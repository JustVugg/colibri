#!/usr/bin/env python3
"""Remove build artifacts. Used by `make clean` so it works from any shell.

Works from cmd.exe, PowerShell, Git Bash, or MSYS2 — no `rm` or POSIX
`for` loop required. Silently ignores files that don't exist.
"""
import glob
import os
import shutil

# Files (relative to c/) to remove if present.
FILES = [
    "colibri", "colibri.exe",
    "olmoe", "olmoe.exe",
    "glm", "glm.exe",
    "iobench", "iobench.exe",
    "backend_cuda.o", "backend_loader.o",
    "backend_cuda_test", "backend_cuda_test.exe",
    "backend_cuda_bench", "backend_cuda_bench.exe",
    "backend_metal.o", "backend_metal_test",
    "coli_cuda.dll", "coli_cuda.lib", "coli_cuda.exp",
    "deepseek_v4", "deepseek_v4.exe",
    "native_quant.o", "native_quant_parallel.o", "native_quant_dual.o",
    "native_quant_batch_avx512.o", "native_quant_fp4_rows16.o",
]
# Test binaries and V4 unit objects match these patterns. Only remove binaries (.exe on Windows,
# no extension on Unix) — never .c or .py source files.
ARTIFACT_GLOBS = ["tests/test_*", "COLI_V4_UNIT_*.o"]
BINARY_EXTENSIONS = {"", ".exe", ".o"}
# Directories to remove.
DIRS = ["tests/__pycache__", "build/ownership"]

removed = 0
for f in FILES:
    if os.path.exists(f):
        os.remove(f)
        removed += 1
for pattern in ARTIFACT_GLOBS:
    for f in glob.glob(pattern):
        if os.path.isfile(f) and os.path.splitext(f)[1] in BINARY_EXTENSIONS:
            os.remove(f)
            removed += 1
for d in DIRS:
    if os.path.isdir(d):
        shutil.rmtree(d)
        removed += 1
print(f"clean: removed {removed} files/dirs")
