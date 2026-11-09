"""Benchmark llama.cpp --n-cpu-moe against Colibri on the same Q4_K_M GGUF.

Run AFTER Colibri serve is stopped (needs the GPU + ~19 GB RAM):
  .venv/Scripts/python.exe scripts/bench_llamacpp.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

BIN = Path(r"D:\.cache\llama.cpp\bin\llama-bench.exe")
GGUF = Path(r"D:\colibri-models\Qwen3-Coder-30B-A3B\Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf")


def main() -> None:
    for ncmoe in (48, 40, 34):
        cmd = [str(BIN), "-m", str(GGUF), "-ngl", "99", "--n-cpu-moe", str(ncmoe),
               "-p", "512", "-n", "128", "-r", "2"]
        print("RUN:", " ".join(cmd), flush=True)
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
        print(p.stdout[-2000:], flush=True)
        if p.returncode != 0:
            print("STDERR:", p.stderr[-1500:], flush=True)


if __name__ == "__main__":
    sys.exit(main())