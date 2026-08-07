#!/usr/bin/env python3
"""Remove build artifacts. Used by `make clean` so it works from any shell.

Works from cmd.exe, PowerShell, Git Bash, or MSYS2 — no `rm` or POSIX
`for` loop required. Silently ignores files that don't exist.
"""
import glob
import os
import re
import shutil

# Files (relative to c/) to remove if present.
FILES = [
    ".build-config",
    "olmoe", "olmoe.exe",
    "inkling", "inkling.exe",
    "kimi_k3", "kimi_k3.exe",
    "colibri", "colibri.exe",
    "glm", "glm.exe",
    "iobench", "iobench.exe",
    "backend_cuda.o", "backend_cuda_ink.o", "backend_loader.o", "backend_vulkan.o",
    "backend_cuda_test", "backend_cuda_test.exe",
    "ragged_attention_test", "ragged_attention_test.exe",
    "backend_cuda_bench", "backend_cuda_bench.exe",
    "backend_metal.o", "backend_metal_test", "backend_metal_test.exe",
    "gemm_largebatch_test", "gemm_largebatch_test.exe",
    "coli_cuda.dll", "coli_cuda.lib", "coli_cuda.exp",
    "tools/libiq3.so", "tools/libiq3.dylib", "tools/iq3.dll",
    "tools/librans_c.so", "tools/librans_c.dylib", "tools/rans_c.dll",
    # qmatmul.spv is checked in; only the other Vulkan shaders are generated.
    "shaders/qmatmul_gate_up.spv", "shaders/attention_absorb.spv",
    "shaders/rmsnorm.spv",
]
# Test binaries are extensionless on Unix and `.exe` on Windows. The set of
# test binaries is exactly what the Makefile builds under tests/<name>$(EXE), so
# derive it from the Makefile (located via this file's own path, not the CWD, so
# it works when clean is invoked from any directory). This keeps `make clean`
# aligned with the build -- a new test target is cleaned automatically -- while
# still never mistaking a source/fixture for an output. The explicit list is a
# fallback only if the Makefile cannot be read.
_C_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MAKEFILE = os.path.join(_C_DIR, "Makefile")
_EXPLICIT_TEST_BASENAMES = [
    "test_serve_sentinel", "test_ue8m0",
    "test_json", "test_st", "test_st_pread", "test_st_mirror", "test_tier", "test_grammar",
    "test_ablate", "test_schema_gbnf", "test_decode_batch", "test_idot",
    "test_i4_grouped", "test_stops", "test_topp", "test_temp_env", "test_kv_alloc",
    "test_rans", "test_fp8_passthrough", "test_fp8_load", "test_qt_addrow",
    "test_i4_acc512", "test_compat_direct", "test_dsa_select",
    "test_int3", "test_int3_load", "test_logit_nan", "test_router_nan", "test_pipe_block",
    "test_sample_nan", "test_tok_o200k", "test_route_trace", "test_corpus_draft",
    "test_cap_precedence", "test_ssd_probe", "test_pilot_ring",
    "test_uring", "test_rammap", "test_resource_masks", "test_e8_kernel",
]
try:
    with open(_MAKEFILE, encoding="utf-8") as _makefile_handle:
        TEST_BASENAMES = re.findall(
            r"^tests/(test_[a-z0-9_]+)\$\(EXE\):",
            _makefile_handle.read(),
            re.MULTILINE,
        )
except OSError:
    TEST_BASENAMES = list(_EXPLICIT_TEST_BASENAMES)
ON_DEMAND_BASENAMES = [
    "bench_topp", "bench_dsa_select", "bench_idot", "bench_mla_simd", "fuzz_rans",
]
TEST_GLOBS = [
    "tests/%s%s" % (name, suffix)
    for name in TEST_BASENAMES + ON_DEMAND_BASENAMES
    for suffix in ("", ".exe")
]
# Directories to remove.
DIRS = [
    "__pycache__",
    "ramdisk_support/__pycache__",
    "tests/__pycache__",
]

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
