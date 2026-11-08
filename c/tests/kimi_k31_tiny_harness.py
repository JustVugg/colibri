#!/usr/bin/env python3
"""Kimi K3.1's DSA indexer on the tiny fixture (tools/make_kimi_k3_tiny.py --k31).

K3.1 is K3 with DeepSeek Sparse Attention on its gated MLA layers; the fixture's
index_topk is 4, so every prompt past 4 tokens attends to a selection. Moonshot
ships no modeling code for K3.1, so the oracle is tools/k3_ref.py, numpy written
from the definition apart from the C engine. The engine runs in its exact-math
configuration (f32 weights, no integer dot products, greedy).

  1. every layer's hidden state matches the oracle's on injected inputs;
  2. the oracle bites: the dense attention (KIMI_DSA_INDEXER=0) does not match it;
  3. while the context holds no more than index_topk positions the attention is
     K3's dense one, bit for bit;
  4. a chunked prefill gives the tokens the token-at-a-time one gives;
  5. two runs give the same tokens;
  6. served, two conversations decoded at once (KV_SLOTS=2, each row on its own
     indexer keys) answer as each does alone.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tools"))
import k3_ref  # noqa: E402

EXACT = {"K3_BITS": "32", "K3_MLA_BITS": "32", "K3_HEAD_BITS": "32", "K3_IDOT": "0", "COLI_TEMP": "0"}
PROMPT = [5, 7, 9, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73]


def run(binary, fixture, ids, ngen, extra=None):
    env = dict(os.environ, **EXACT)
    env.pop("KIMI_DSA_INDEXER", None)
    env.update(extra or {})
    r = subprocess.run([binary, str(fixture), "--ids", " ".join(map(str, ids)), "--ngen", str(ngen)],
                       env=env, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise AssertionError(f"engine exited {r.returncode}:\n{r.stderr[-2000:]}")
    return [int(t) for t in r.stdout.split("TUNE")[0].split()]


def trace(binary, fixture, x0, n, layers, dim, tmp, extra=None):
    out = Path(tmp) / f"trace{len(os.listdir(tmp))}.bin"
    run(binary, fixture, list(range(2, 2 + n)), 0, dict(extra or {}, K3_X0=str(x0), K3_TRACE=str(out)))
    a = np.fromfile(out, dtype=np.float32)
    return a.reshape(n, layers + 1, dim)[:, :layers]   # each token's rows end with the final norm's


def serve(binary, model, slots, prompts, max_new):
    """The served answers of prompts submitted together, one slot each (SERVE=1's
    SUBMIT/DATA/DONE protocol), greedy."""
    env = dict(os.environ, **EXACT, SERVE="1", SNAP=str(model), KV_SLOTS=str(slots))
    p = subprocess.Popen([binary], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, env=env)
    try:
        for i, text in enumerate(prompts):
            body = text.encode("utf-8")
            p.stdin.write(f"SUBMIT r{i} {i} {len(body)} {max_new} 0.0 1.0\n".encode() + body + b"\n")
        p.stdin.flush()
        out, done = {f"r{i}": b"" for i in range(len(prompts))}, set()
        while len(done) < len(prompts):
            line = p.stdout.readline()
            if not line:
                raise AssertionError("the engine closed stdout")
            f = line.decode("utf-8", "replace").split()
            if f and f[0] == "ERROR":
                raise AssertionError(f"engine ERROR: {line!r}")
            if f and f[0] == "DATA" and len(f) == 3:
                out[f[1]] += p.stdout.read(int(f[2]))
                p.stdout.read(1)
            elif f and f[0] == "DONE":
                done.add(f[1])
        return [out[f"r{i}"] for i in range(len(prompts))]
    finally:
        p.stdin.close()
        p.wait(timeout=60)


def rel(a, b):
    return max(float(np.linalg.norm(a[:, i] - b[:, i]) / (np.linalg.norm(b[:, i]) + 1e-30))
               for i in range(a.shape[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", required=True)
    ap.add_argument("--fixture", required=True)
    a = ap.parse_args()
    fixture = Path(a.fixture)
    cfg = json.loads((fixture / "config.json").read_text())
    dsa = cfg["dsa_config"]
    layers, dim, topk = cfg["num_hidden_layers"], cfg["hidden_size"], dsa["index_topk"]
    assert cfg["model_type"] == "kimi_k31" and topk < 12, "not the K3.1 tiny fixture"
    with tempfile.TemporaryDirectory() as tmp:
        n = 12
        x0 = Path(tmp) / "x0.bin"
        rng = np.random.default_rng(7)
        rng.standard_normal((n, dim)).astype(np.float32).tofile(x0)
        ref_path = Path(tmp) / "ref.bin"
        k3_ref.K3Ref(str(fixture), layers).run(np.fromfile(x0, dtype=np.float32).reshape(n, dim), str(ref_path))
        ref = np.fromfile(ref_path, dtype=np.float32).reshape(n, layers, dim)

        got = trace(a.binary, fixture, x0, n, layers, dim, tmp)
        worst = rel(got, ref)
        assert worst < 3e-4, f"the engine's DSA layers do not match the oracle: worst rel-L2 {worst:.3e}"
        print(f"OK oracle: {n} tokens past index_topk {topk}, {layers} layers, worst rel-L2 {worst:.2e}")

        dense = trace(a.binary, fixture, x0, n, layers, dim, tmp, {"KIMI_DSA_INDEXER": "0"})
        bite = rel(dense, ref)
        assert bite > 1e-2, f"the dense attention matches the sparse oracle (rel-L2 {bite:.3e}): the test does not bite"
        print(f"OK bites: the dense attention is {bite:.2e} away from the oracle")

        x4 = Path(tmp) / "x4.bin"
        np.fromfile(x0, dtype=np.float32)[:topk * dim].tofile(x4)
        short = trace(a.binary, fixture, x4, topk, layers, dim, tmp)
        short_dense = trace(a.binary, fixture, x4, topk, layers, dim, tmp, {"KIMI_DSA_INDEXER": "0"})
        assert np.array_equal(short, short_dense), "within index_topk the attention is not the dense one bit for bit"
        print(f"OK dense: {topk} tokens (= index_topk) give the dense attention's bits")

    one = run(a.binary, fixture, PROMPT, 12, {"K3_CHUNK": "1"})
    for chunk in ("3", "32"):
        got = run(a.binary, fixture, PROMPT, 12, {"K3_CHUNK": chunk})
        assert got == one, f"K3_CHUNK={chunk} gives {got}, token at a time {one}"
    assert run(a.binary, fixture, PROMPT, 12) == run(a.binary, fixture, PROMPT, 12), "two runs differ"
    print(f"OK prefill: chunks of 1, 3 and 32 give the same {len(one)} tokens; two runs agree")

    with tempfile.TemporaryDirectory() as tmp:
        model = Path(tmp) / "fixture"
        model.mkdir()
        for name in ("config.json", "model.safetensors"):
            shutil.copy(fixture / name, model / name)
        shutil.copy(HERE / "tok_kimi_tiny.json", model / "tokenizer.json")
        prompts = ["the quick brown fox jumps over the lazy dog " * 4,
                   "a stitch in time saves nine, and so on and so forth " * 3]
        alone = [serve(a.binary, model, 1, [t], 16)[0] for t in prompts]
        together = serve(a.binary, model, 2, prompts, 16)
        assert together == alone, f"two conversations at once answer {together}, alone {alone}"
        assert all(alone), "a served answer is empty"
    print("OK slots: two conversations decoded at once answer as each alone")
    return 0


if __name__ == "__main__":
    sys.exit(main())
