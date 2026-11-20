#!/usr/bin/env bash
# The SYCL migration of the CUDA backend, scripted so it can be re-run on upstream changes
# (the migrated files in c/sycl/ are a fork of backend_cuda.cu, cuda_chain.cu and cu_xb.cu;
# re-running this and re-applying the hand edits is how the fork follows the originals).
# No GPU is touched: dpct and icpx live in the oneAPI container, the CUDA headers it parses
# are a header-only extraction under $WORK/cuda-hdr.
#
#   c/sycl/tools/dpct-migrate.sh [c-dir] [work-dir]
#
# 1. dpct-prep.py stages c/ into $WORK/dpct-in with coli_launch expanded (see that file);
# 2. dpct migrates the three backend units and the two GPU tests with COLI_GPU_PORTABLE_ONLY,
#    so the NVIDIA-only kernels (WMMA, mma.sync, PTX) are never seen (backend_gpu_compat.h);
# 3. the results land in c/sycl/ (*.dp.cpp, tests/*.dp.cpp, include/dpct/, the migrated
#    in-root headers that differ from their originals).
set -euo pipefail
C=${1:-$(cd "$(dirname "$0")/../.." && pwd)}
WORK=${2:-${TMPDIR:-/tmp}/colibri-dpct}
IMAGE=${IMAGE:-intel/oneapi-basekit:2025.3.2-0-devel-ubuntu24.04}
CUDA_INC=${CUDA_INC:-$WORK/cuda-hdr/usr/local/cuda-12.8/targets/x86_64-linux/include}
TUS="backend_cuda.cu cuda_chain.cu cu_xb.cu tests/test_cuda_chain.cu tests/test_backend_cuda.cu"

python3 -I "$C/sycl/tools/dpct-prep.py" "$C" "$WORK/dpct-in"
rm -rf "$WORK/dpct-out"
# The tool predefines nvcc's __USE_FAST_MATH__ (no flag clears it): backend_cuda.cu's fmt=8 guard
# tests it only outside COLI_GPU_PORTABLE_ONLY and keeps __FAST_MATH__, which icpx defines under
# its default -fp-model=fast, so the migrated unit still refuses a fast-math build (MIGRATION.md).
# -DCOLI_HIP_NO_WMMA=1 is inert for a CUDA parse (it is read inside the HIP arm only); it stays
# so the command matches the plan's. The portable-only define is the one that matters.
docker run --rm -u "$(id -u):$(id -g)" -e HOME=/tmp -v "$WORK:/work" -v "$CUDA_INC:/cuda-inc:ro" \
    -w /work/dpct-in --entrypoint bash "$IMAGE" -lc "
  source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1
  dpct --cuda-include-path=/cuda-inc --in-root=/work/dpct-in --out-root=/work/dpct-out --gen-helper-function \
       --use-experimental-features=matrix,masked-sub-group-operation,local-memory-kernel-scope-allocation \
       --extra-arg=-I/work/dpct-in --extra-arg=-DCOLI_HIP_NO_WMMA=1 --extra-arg=-DCOLI_GPU_PORTABLE_ONLY=1 \
       $TUS" 2>&1 | tee "$WORK/dpct-migrate.log"

mkdir -p "$C/sycl/tests" "$C/sycl/include"
for f in backend_cuda cuda_chain cu_xb; do cp "$WORK/dpct-out/$f.dp.cpp" "$C/sycl/$f.dp.cpp"; done
for f in test_cuda_chain test_backend_cuda; do cp "$WORK/dpct-out/tests/$f.dp.cpp" "$C/sycl/tests/$f.dp.cpp"; done
rm -rf "$C/sycl/include/dpct"
cp -r "$WORK/dpct-out/include/dpct" "$C/sycl/include/dpct"
# The two in-root headers with device code in them, kept beside the units so a quoted include
# finds the migrated one first. The tool also rewrites the ABI and host headers it saw
# (cuda_chain.h, cu_tier_ops.h, vk_alloc.h, backend_vulkan.h: it names their anonymous structs,
# dpct_type_NNN), which the units must not take: the engine's contract is the original in c/,
# found through -I. Those copies are listed, not kept.
for h in backend_gpu_compat.h cuda_bd.dp.hpp; do cp "$WORK/dpct-out/$h" "$C/sycl/$h"; done
for h in $(cd "$WORK/dpct-out" && ls *.h 2>/dev/null); do
    [ "$h" = backend_gpu_compat.h ] && continue
    cmp -s "$WORK/dpct-out/$h" "$C/$h" || echo "rewritten by the tool, not kept: $h"
done
echo "migration in $C/sycl (log: $WORK/dpct-migrate.log)"
