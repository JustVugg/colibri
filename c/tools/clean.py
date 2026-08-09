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
    "inkling", "inkling.exe",
    "kimi_k3", "kimi_k3.exe",
    "olmoe", "olmoe.exe",
    "glm", "glm.exe",                       # pre-rename name of the colibri engine
    "iobench", "iobench.exe",
    "backend_cuda.o", "backend_loader.o",
    "backend_cuda_test", "backend_cuda_test.exe",
    "backend_cuda_bench", "backend_cuda_bench.exe",
    "backend_metal.o", "backend_metal_test",
    "coli_cuda.dll", "coli_cuda.lib", "coli_cuda.exp",
    # hipcc emits an import library, export file and PDB alongside the DLL.
    "coli_hip.dll", "coli_hip.lib", "coli_hip.exp", "coli_hip.pdb",
    "deepseek_v4", "deepseek_v4.exe",
    "native_quant.o", "native_quant_parallel.o", "native_quant_dual.o",
    "native_quant_batch_avx512.o", "native_quant_fp4_rows16.o",
]
# Test binaries and V4 unit objects. The test globs deliberately have no
# extension: on Unix that is what a built test IS, and matching only
# "tests/test_*.exe" (as this did) meant `make clean` removed nothing at all on
# Linux and macOS.
#
# That is not a tidiness problem -- it silently invalidates verification. Change a
# compile flag, run `make clean && make test-c`, and the stale binaries built
# with the OLD flags are re-run and reported as passing. CONTRIBUTING's
# `make check` starts with exactly that sequence.
#
# KEEP_EXT is the safety rail: a source file must never match. Everything the
# repo tracks under tests/ carries one of these extensions, and directories
# (tests/fixtures/) are skipped by the isfile() check. Object files are not in
# it, so COLI_V4_UNIT_*.o is removed by the same rule rather than a second one.
ARTIFACT_GLOBS = ["tests/test_*", "tests/bench_*", "tests/fuzz_*",
                  "COLI_V4_UNIT_*.o"]
KEEP_EXT = (".c", ".h", ".cc", ".cpp", ".cu", ".mm", ".py", ".txt", ".json",
            ".md", ".bin", ".sh", ".toml", ".yml", ".yaml")
# Directories to remove.
DIRS = ["tests/__pycache__", "build/ownership"]

removed = 0
for f in FILES:
    if os.path.exists(f):
        os.remove(f)
        removed += 1
for pattern in ARTIFACT_GLOBS:
    for f in glob.glob(pattern):
        if not os.path.isfile(f):          # tests/fixtures/ and friends
            continue
        if f.endswith(KEEP_EXT):           # never a source file
            continue
        os.remove(f)
        removed += 1
for d in DIRS:
    if os.path.isdir(d):
        shutil.rmtree(d)
        removed += 1
print(f"clean: removed {removed} files/dirs")
