#!/usr/bin/env python3
"""Brain tab on Kimi K3: EMAP after READY and STAT, HITS after every turn.

The dashboard reads two lines from the engine's stdout for its Brain tab:
EMAP (the grid: one row per sparse layer, one column per expert, two hex
digits per cell) and HITS (which experts the turn routed, one bit each,
packed 8 per hex pair, after DONE next to PROF). kimi_k3.c emitted neither,
so on Kimi K3 the tab was empty. Same harness as test_kimi_k3_ckpt.py: the
tiny fixture plus tests/tok_kimi_tiny.json as tokenizer.

Exit 0 when the lines arrive in the shape the gateway parses, 1 otherwise.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PROMPT = b"abcabcabc"
MAX_NEW = 3


def read_until(process, kind, limit=400):
    lines = []
    for _ in range(limit):
        line = process.stdout.readline()
        if not line:
            raise RuntimeError("engine closed before " + kind)
        text = line.decode("latin-1").rstrip("\n")
        if text.split(" ", 1)[0] == "DATA":
            process.stdout.read(int(text.split()[2]))
            process.stdout.readline()
            continue
        lines.append(text)
        if text.startswith(kind):
            return lines
    raise RuntimeError(f"no {kind} within {limit} lines:\n" + "\n".join(lines[-10:]))


def only(lines, kind):
    found = [l.split() for l in lines if l.startswith(kind + " ")]
    if len(found) != 1:
        raise AssertionError(f"expected exactly one {kind}, got {found}")
    return found[0]


def check(binary, model, stderr_path):
    env = dict(os.environ, SERVE="1", SNAP=str(model))
    with open(stderr_path, "wb") as stderr_file:
        process = subprocess.Popen([str(binary)], stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=stderr_file, env=env)
        try:
            read_until(process, "\x01\x01READY")
            boot = read_until(process, "EMAP")
            header = f"SUBMIT r1 0 {len(PROMPT)} {MAX_NEW} 0.0 1.0\n".encode()
            process.stdin.write(header + PROMPT + b"\n")
            process.stdin.flush()
            turn = read_until(process, "HITS")
            after = read_until(process, "EMAP")
        finally:
            process.stdin.close()
            process.wait(timeout=60)
    config = json.loads((model / "config.json").read_text())
    text_config = config.get("text_config", config)
    sparse = text_config["num_hidden_layers"] - text_config.get("first_k_dense_replace", 0)
    experts = text_config["num_experts"] if "num_experts" in text_config else text_config["n_routed_experts"]

    kinds = [l.split(" ", 1)[0] for l in boot]
    assert kinds.index("STAT") < kinds.index("EMAP"), "EMAP must follow READY and STAT"
    _, rows, cols, grid = only(boot, "EMAP")
    assert (int(rows), int(cols)) == (sparse, experts), f"EMAP {rows}x{cols}, config says {sparse}x{experts}"
    assert len(grid) == int(rows) * int(cols) * 2, "EMAP: two hex digits per cell"

    kinds = [l.split(" ", 1)[0] for l in turn]
    assert kinds.index("DONE") < kinds.index("HITS"), "HITS belongs after DONE, with PROF"
    _, hrows, hcols, hits = only(turn, "HITS")
    assert (hrows, hcols) == (rows, cols), "HITS must use EMAP's rows and columns"
    assert len(hits) == ((int(rows) * int(cols) + 7) // 8) * 2, "HITS: 8 experts per hex pair"
    assert int(hits, 16) != 0, "a 9-token prompt through the sparse layers routed no expert"

    prof = only(turn, "PROF")
    assert len(prof) == 10, "PROF keeps its ten fields"
    assert int(prof[3]) == MAX_NEW, "completion tokens"
    assert int(prof[9]) == MAX_NEW, "forwards: prefill plus one per token after the first"

    _, _, _, refreshed = only(after, "EMAP")
    assert int(refreshed, 16) != 0, "after a turn the cache holds experts: the refreshed grid shows them"
    return rows, cols, hits


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--binary", required=True)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--tokenizer",
                        default=str(Path(__file__).parent / "tok_kimi_tiny.json"))
    args = parser.parse_args()
    base = Path(args.fixture)
    work = Path(tempfile.mkdtemp(prefix="kimi-k3-dash-"))
    try:
        model = work / "fixture"
        model.mkdir()
        shutil.copy(base / "config.json", model / "config.json")
        shutil.copy(base / "model.safetensors", model / "model.safetensors")
        shutil.copy(args.tokenizer, model / "tokenizer.json")
        try:
            rows, cols, hits = check(args.binary, model, work / "engine.stderr")
        except (AssertionError, RuntimeError) as failure:
            print(f"FAIL: {failure}")
            err = (work / "engine.stderr").read_text(errors="replace")
            if err:
                print(err[-2000:])
            return 1
        print(f"kimi_k3 dashboard: EMAP {rows}x{cols}, HITS {hits}")
        return 0
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
