#!/usr/bin/env python3
"""Compare CPU/CUDA full-model logits and greedy replies on unchanged weights.

Requires numpy for the numerical comparison. Writes both raw logits, logs,
replies and a machine-readable report to a new output directory.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

from benchmark_mimo_cuda import PROMPT, sha, verify_cuda


def main():
    import numpy as np
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--engine", required=True, type=Path)
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--shards", default="")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--tokens", type=int, default=32)
    p.add_argument("--cap", type=int, default=88)
    p.add_argument("--threads", type=int, default=24)
    p.add_argument("--release-host", action="store_true")
    p.add_argument("--prompt-file", type=Path, help="UTF-8 fully rendered chat prompt")
    p.add_argument("--chunk", type=int, default=64,
                   help="prompt rows a block (MIMO_CHUNK); 64 or more puts a block's experts, attention "
                        "and matrices on the GPU's prompt kernels")
    p.add_argument("--env", action="append", default=[], metavar="K=V",
                   help="another variable for both runs (MIMO_ATTN_ROWS=64: blocks of 256 rows in pieces "
                        "of 64, the GPU's pipelined layers)")
    a = p.parse_args()
    if a.tokens < 1:
        p.error("--tokens must be positive")
    engine, model, output = a.engine.resolve(), a.model.resolve(), a.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    prompt = a.prompt_file.read_text(encoding="utf-8") if a.prompt_file else PROMPT
    vocab = json.loads((model / "config.json").read_text())["vocab_size"]
    config = {"MIMO_DIRS": a.shards, "MIMO_DENSE_BITS": "0", "MIMO_IDOT": "0",
              "MIMO_DIRECT": "1", "MIMO_READ_THREADS": "16", "MIMO_CHUNK": str(a.chunk),
              "OMP_NUM_THREADS": str(a.threads), "COLI_NO_OMP_TUNE": "1", "COLI_TEMP": "0",
              "MIMO_CTX": "16384", "COLI_VULKAN": "0", "MIMO_CUDA_DENSE_GB": "20",
              "MIMO_CUDA_RESERVE_GB": "2", "SERVE": "0"}
    config["MIMO_CUDA_RELEASE_HOST"] = "1" if a.release_host else "0"
    for kv in a.env:
        k, _, v = kv.partition("=")
        config[k] = v
    for backend in ("cpu", "cuda"):
        env = {k: v for k, v in os.environ.items() if not k.startswith(("MIMO_", "COLI_", "CUDA_"))}
        env.update(config)
        env["MIMO_CUDA"] = "1" if backend == "cuda" else "0"
        env["MIMO_LOGITS"] = str(output / f"{backend}.f32")
        with (output / f"{backend}.log").open("wb") as log, (output / f"{backend}.txt").open("wb") as reply:
            subprocess.run([str(engine), str(model), "--cap", str(a.cap), "--ngen", str(a.tokens),
                            "--prompt", prompt], env=env, stdout=reply, stderr=log, check=True, timeout=1800)
        print(f"{backend}: complete", flush=True)
    verify_cuda((output / "cuda.log").read_text(errors="replace"))
    cpu = np.fromfile(output / "cpu.f32", dtype="<f4").reshape(-1, vocab).astype(np.float64)
    cuda = np.fromfile(output / "cuda.f32", dtype="<f4").reshape(-1, vocab).astype(np.float64)
    if cpu.shape != cuda.shape or not cpu.size or not np.isfinite(cpu).all() or not np.isfinite(cuda).all():
        raise AssertionError("incomplete, mismatched or nonfinite logits")
    matches = int((cpu.argmax(1) == cuda.argmax(1)).sum())
    max_abs = float(np.abs(cpu - cuda).max())
    logp = cpu - cpu.max(1, keepdims=True)
    logq = cuda - cuda.max(1, keepdims=True)
    logp -= np.log(np.exp(logp).sum(1, keepdims=True))
    logq -= np.log(np.exp(logq).sum(1, keepdims=True))
    max_kl = float((np.exp(logp) * (logp - logq)).sum(1).max())
    same_reply = (output / "cpu.txt").read_bytes() == (output / "cuda.txt").read_bytes()
    passed = matches == cpu.shape[0] and max_kl < 1e-5 and same_reply
    report = {"passed": passed, "positions": cpu.shape[0], "argmax_matches": matches,
              "max_abs_logit_difference": max_abs, "max_kl_cpu_to_cuda": max_kl,
              "greedy_reply_equal": same_reply, "requested_tokens": a.tokens,
              "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "config": config,
              "engine_sha256": sha(engine), "files": {f.name: sha(f) for f in output.iterdir() if f.is_file()}}
    dll = engine.parent / "coli_cuda_mimo.dll"
    if dll.exists():
        report["cuda_dll_sha256"] = sha(dll)
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("passed", "positions", "argmax_matches", "max_abs_logit_difference", "max_kl_cpu_to_cuda", "greedy_reply_equal")}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
