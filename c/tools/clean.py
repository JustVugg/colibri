#!/usr/bin/env python3
"""Remove build artifacts. Used by `make clean` so it works from any shell.

Works from cmd.exe, PowerShell, Git Bash, or MSYS2 — no `rm` or POSIX
`for` loop required. Silently ignores files that don't exist.
"""
import glob
import os
import shutil
import sys

C_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if C_DIR not in sys.path:                   # `python tools/clean.py` puts tools/ first
    sys.path.insert(0, C_DIR)
from family_registry import all_families

# The engines are DERIVED, not listed. family_registry is what the release job
# already trusts to decide which engines an archive must contain (release.yml,
# "Verify the packaged archive actually runs"), so deriving from it makes one
# fact true by construction: nothing can be published that clean does not own.
#
# The hand-written list had qwen36 missing. `make clean` therefore left the old
# qwen36 binary in place, and the release Package step copies whatever sits at
# c/qwen36 without asking where it came from -- so a locally staged release
# could ship an engine that was never rebuilt. CI never saw it (fresh runners);
# a maintainer packaging on their own machine would have.
ENGINES = sorted({family.engine_artifact for family in all_families()})

# Files (relative to c/) to remove if present.
FILES = [name for engine in ENGINES for name in (engine, engine + ".exe")] + [
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
#
# tests/*_probe* is here for the same reason. The XDNA physical qualification
# probe is built by an explicit `make tests/xdna_physical_probe$(EXE)` and
# matched none of the test_/bench_/fuzz_ prefixes, so it survived every clean.
# A stale probe does not just waste space -- it is the owner that PRODUCES
# physical execution evidence, and a stale one reports PASS for code that is no
# longer in the tree.
ARTIFACT_GLOBS = ["tests/test_*", "tests/bench_*", "tests/fuzz_*",
                  "tests/*_probe*", "COLI_V4_UNIT_*.o"]
KEEP_EXT = (".c", ".h", ".cc", ".cpp", ".cu", ".mm", ".py", ".txt", ".json",
            ".md", ".bin", ".sh", ".toml", ".yml", ".yaml")
# Directories to remove.
DIRS = ["tests/__pycache__", "build/ownership"]

def clean():
    """Remove everything above, relative to the current directory."""
    removed = 0
    for f in FILES:
        if os.path.exists(f):
            os.remove(f)
            removed += 1
    for pattern in ARTIFACT_GLOBS:
        for f in glob.glob(pattern):
            if not os.path.isfile(f):      # tests/fixtures/ and friends
                continue
            if f.endswith(KEEP_EXT):       # never a source file
                continue
            os.remove(f)
            removed += 1
    for d in DIRS:
        if os.path.isdir(d):
            shutil.rmtree(d)
            removed += 1
    return removed


# Guarded, so that reading FILES (which tests/test_clean_ownership.py does, to
# check it against the release registry) cannot delete the build products the
# rest of the suite is running against. Importing this module must be inert.
if __name__ == "__main__":
    print(f"clean: removed {clean()} files/dirs")
