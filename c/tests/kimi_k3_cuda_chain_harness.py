#!/usr/bin/env python3
"""Kimi K3's CUDA chain (kimi_k3_cuchain.h) against the CPU on the tiny K3 and K3.1 fixtures.

A CUDA build (make CUDA=1 kimi_k3) on a machine with a card. For each fixture (K3, and K3.1 with
its DSA indexer and a top 4, so a 30-token prompt selects in every MLA layer), each trunk (f32,
int8, int4) and each placement (every layer, chunks of 3 rows, a token at a time, the first 3
layers, the first 1): the chain's tokens are the CPU's, with the prompt GEMMs on the exact
kernel (COLI_CUDA_TC_GEMM=0), and the log says how many layers the card ran. With the tensor
cores on (the default: TF32 or f16 inputs, 1e-3 relative) the first token is the CPU's. Then
the K3.1 test (oracle, selection, slots) and the vendor oracle of K3 with the chain on.
  make CUDA=1 kimi-k3-cuda-chain-check
"""
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROMPT = "5 7 9 11 13 17 19 23 29 31 37 41 43 47 53 59 61 67 71 73 79 83 89 97 101 103 107 109 113 127"
TRUNKS = {"f32": {"K3_BITS": "32", "K3_MLA_BITS": "32", "K3_HEAD_BITS": "32"},
          "int8": {"K3_BITS": "8", "K3_MLA_BITS": "8", "K3_HEAD_BITS": "8"},
          "int4": {"K3_BITS": "4"}}
PLACES = {"every layer": {}, "chunks of 3": {"COLI_CUDA_K3_ROWS": "3"}, "a token at a time": {"K3_CHUNK": "1"},
          "first 3 layers": {"COLI_CUDA_CHAIN_LAYERS": "3"}, "first layer": {"COLI_CUDA_CHAIN_LAYERS": "1"}}
CUDA = {"K3_CUDA": "1", "COLI_CUDA": "1", "COLI_VK_TIER_SYNC": "1"}


def run(binary, fixture, extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("K3_", "COLI_CUDA", "KIMI_"))}
    env.update({"K3_IDOT": "0", "COLI_TEMP": "0", "USAGE_SAVE": "0"}, **extra)
    r = subprocess.run([binary, str(fixture), "--ids", PROMPT, "--ngen", "12"], env=env, capture_output=True, text=True,
                       timeout=600)
    if r.returncode != 0:
        raise AssertionError(f"engine exited {r.returncode}:\n{r.stderr[-2000:]}")
    m = re.search(r"kimi_k3 chain: (\d+) of (\d+) layers on the card", r.stderr)
    return r.stdout.split("TUNE")[0].split(), (int(m.group(1)), int(m.group(2))) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", required=True)
    ap.add_argument("--k3", required=True, help="the tiny K3 fixture (with its ref.json)")
    ap.add_argument("--k31", required=True, help="the tiny K3.1 fixture")
    a = ap.parse_args()
    fails = 0
    for fx in (a.k3, a.k31):
        for tn, trunk in TRUNKS.items():
            cpu, on = run(a.binary, fx, trunk)
            assert on is None and cpu, f"{fx} {tn}: the CPU run put layers on the card or gave nothing"
            for pn, place in PLACES.items():
                got, on = run(a.binary, fx, dict(trunk, **CUDA, **place, COLI_CUDA_TC_GEMM="0"))
                want = int(place.get("COLI_CUDA_CHAIN_LAYERS", "0")) or None
                ok = got == cpu and on is not None and (want is None or on[0] == want) and (want is not None or on[0] == on[1])
                print(f"{'OK  ' if ok else 'FAIL'} {Path(fx).name} {tn} {pn}: {on[0] if on else 0} layers on the card, "
                      f"{'the CPU tokens' if got == cpu else 'tokens ' + ' '.join(got) + ', the CPU ' + ' '.join(cpu)}")
                fails += not ok
            got, on = run(a.binary, fx, dict(trunk, **CUDA))
            ok = on is not None and got[:1] == cpu[:1]
            print(f"{'OK  ' if ok else 'FAIL'} {Path(fx).name} {tn} tensor cores: the first token {'the CPU' if ok else got[:1]}"
                  f"'s, {sum(x == y for x, y in zip(got, cpu))} of {len(cpu)} the same")
            fails += not ok
            _, off = run(a.binary, fx, dict(trunk, **CUDA, COLI_CUDA_CHAIN="0"))
            print(f"{'OK  ' if off is None else 'FAIL'} {Path(fx).name} {tn} COLI_CUDA_CHAIN=0: no layer on the card")
            fails += off is not None
    env = dict(os.environ, **CUDA)
    for cmd in ([sys.executable, str(HERE / "kimi_k31_tiny_harness.py"), "--binary", a.binary, "--fixture", a.k31],
                [sys.executable, str(HERE / "test_kimi_k3_tiny.py"), "--binary", a.binary, "--fixture", a.k3]):
        r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1800)
        print(f"{'OK  ' if r.returncode == 0 else 'FAIL'} {Path(cmd[1]).name} with the chain on")
        if r.returncode:
            print(r.stdout[-1500:], r.stderr[-1500:])
        fails += r.returncode != 0
    print("kimi-k3-cuda-chain-check: " + (f"{fails} FAILED" if fails else "all passed"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
