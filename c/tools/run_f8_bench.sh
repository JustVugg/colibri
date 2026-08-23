#!/usr/bin/env bash
# Build + run the fmt=8 kernel bench (c/tests/bench_fp8_cuda.cu) on a CUDA
# host. JSON on stdout, build output and progress on stderr, so the caller can
# pipe stdout straight into a file or jq. Extra make variables pass through:
#
#   bash c/tools/run_f8_bench.sh                    # -arch=native (default)
#   bash c/tools/run_f8_bench.sh CUDA_ARCH=sm_121   # explicit arch
#
# The bench times the old per-(o,s) fmt=8 kernels against the COLI_CUDA_F8_WARP
# rework (LUT decode and, where the toolchain has cuda_fp8.h, the cvt decode)
# at S in {1,4,8,32} on the census expert shapes, reporting weight-goodput
# GB/s and % of the BENCH_PEAK_GBPS roofline.
#
# BENCH_PEAK_GBPS defaults to 273 — the GB10 Spark wire (256-bit LPDDR5X x
# 8533 MT/s), this harness's primary host. It must be an explicit number
# because cudaDevAttrMemoryClockRate is device-inconsistent: on GB10 it
# returns the EFFECTIVE 8,533,000 kHz rate, so the usual 2x DDR factor would
# double-count to ~546 GB/s. Override for any other host, or set it empty
# (BENCH_PEAK_GBPS=) to get pct_peak=-1 plus the raw attributes to judge from.
set -euo pipefail
export BENCH_PEAK_GBPS="${BENCH_PEAK_GBPS-273}"
cd "$(dirname "$0")/.."
make fp8_bench CUDA=1 "$@" 1>&2
./fp8_bench
