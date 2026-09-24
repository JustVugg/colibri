#!/usr/bin/env bash
# bench_fp4_matmul.sh — build & A/B the FP4 expert matmul's SIMD vs scalar arm.
#
# Builds c/deepseek_v4.c twice (default flags = SIMD arm active; then with
# EXTRA_CFLAGS=-mno-avx2 on x86 to force the scalar #else arm), links the
# bench_fp4_matmul.c harness against each, verifies the two arms agree
# bit-exactly, and prints timings at the real DeepSeek-V4-Flash expert shapes.
#
# Usage (from c/):   bash tools/bench_fp4_matmul.sh [S] [I] [O] [iters]
# Needs: gcc or clang, make. CPU-only; no model files required.
set -euo pipefail
cd "$(dirname "$0")/.."

S="${1:-128}"; I="${2:-4096}"; O="${3:-2048}"; IT="${4:-7}"

# Mirror the repo's own unit flags (Makefile.deepseek-v4 / .units).
ARCH=armv8.2-a+dotprod
SCALAR=""
AB=1
case "$(uname -m)" in
    x86_64*|amd64) ARCH=x86-64-v3 ; SCALAR="-mno-avx2" ;;
    arm64|aarch64) SCALAR="-march=armv8-a+nosimd" ;;   # +nosimd undefines __ARM_NEON:
                                                    # the scalar #else arm, against which
                                                    # the NEON arm must be bit-exact (#1696)
    *) echo "unsupported arch $(uname -m)"; exit 1 ;;
esac

UNIT="-DCOLI_V4_UNIT_NATIVE_QUANT"
COMMON="-O2 -fopenmp -I."
# Apple clang lacks -fopenmp; prefer a real gcc when present.
CC=gcc
$CC --version 2>/dev/null | grep -qi "gcc" || CC=gcc-15
command -v $CC >/dev/null || { echo "no gcc with -fopenmp found"; exit 1; }
echo "using $($CC --version | head -1)"

echo "== building SIMD-arm unit =="
$CC $COMMON -march=$ARCH $UNIT -c deepseek_v4.c -o /tmp/bk_unit_simd.o
$CC $COMMON -march=$ARCH -c tools/bench_fp4_matmul.c -o /tmp/bk_main_simd.o
$CC /tmp/bk_unit_simd.o /tmp/bk_main_simd.o -o /tmp/bk_simd -lm -fopenmp

REF=/tmp/bk_ref.f32
rm -f "$REF"
if [ "$AB" = 1 ]; then
    echo "== building scalar-arm unit ($SCALAR) =="
    $CC $COMMON -march=$ARCH $SCALAR $UNIT -c deepseek_v4.c -o /tmp/bk_unit_sca.o
    $CC $COMMON -march=$ARCH $SCALAR -c tools/bench_fp4_matmul.c -o /tmp/bk_main_sca.o
    $CC /tmp/bk_unit_sca.o /tmp/bk_main_sca.o -o /tmp/bk_sca -lm -fopenmp
    echo "== SIMD arm (writes reference) =="
    /tmp/bk_simd "$S" "$I" "$O" "$IT" "$REF"
    echo "== scalar arm (compares bit-exactly + times) =="
    /tmp/bk_sca "$S" "$I" "$O" "$IT" "$REF" | tee /tmp/bk_sca.log
    grep -q "BITEXACT vs .*: IDENTICAL" /tmp/bk_sca.log || { echo "FAIL: SIMD and scalar arms differ"; exit 1; }
    echo "Done. Repeat /tmp/bk_simd ... to re-verify bit-exactness."
else
    echo "== no SIMD arm on this arch/kernel yet (#1696); scalar baseline only =="
    /tmp/bk_simd "$S" "$I" "$O" "$IT"
fi
