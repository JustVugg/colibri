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
    "olmoe", "olmoe.exe",
    "colibri", "colibri.exe",
    "glm", "glm.exe",
    "iobench", "iobench.exe",
    "backend_cuda.o", "backend_loader.o",
    "backend_cuda_test", "backend_cuda_test.exe",
    "backend_cuda_bench", "backend_cuda_bench.exe",
    "backend_metal.o", "backend_metal_test",
    "coli_cuda.dll", "coli_cuda.lib", "coli_cuda.exp",
]
# Test binaries are extensionless on Unix and `.exe` on Windows.  Keep the
# basenames explicit so clean can never mistake a source/fixture for an output.
TEST_BASENAMES = [
    "test_json", "test_st", "test_st_pread", "test_st_mirror", "test_tier", "test_grammar",
    "test_schema_gbnf", "test_decode_batch", "test_idot",
    "test_i4_grouped", "test_stops", "test_topp", "test_kv_alloc",
    "test_i4_acc512", "test_compat_direct", "test_dsa_select",
    "test_int3", "test_int3_load", "test_logit_nan", "test_pipe_block",
    "test_sample_nan", "test_tok_o200k", "test_efficiency_report",
    "test_uring", "test_rammap", "test_resource_masks", "test_e8_kernel",
]
ON_DEMAND_BASENAMES = ["bench_topp", "bench_dsa_select"]
TEST_GLOBS = [
    "tests/%s%s" % (name, suffix)
    for name in TEST_BASENAMES + ON_DEMAND_BASENAMES
    for suffix in ("", ".exe")
]
# Directories to remove.
DIRS = ["tests/__pycache__"]

removed = 0
for f in FILES:
    if os.path.exists(f):
        os.remove(f)
        removed += 1
for pattern in TEST_GLOBS:
    for f in glob.glob(pattern):
        os.remove(f)
        removed += 1
for d in DIRS:
    if os.path.isdir(d):
        shutil.rmtree(d)
        removed += 1
print(f"clean: removed {removed} files/dirs")
