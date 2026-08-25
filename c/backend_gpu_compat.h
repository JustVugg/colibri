/* backend_gpu_compat.h — one GPU backend source, two vendors.
 * Same pattern as compat.h for Windows: every platform difference lives in
 * this header and backend_cuda.cu stays untouched. Compiled by nvcc this is
 * a pass-through to the CUDA runtime (and mma.h for the tensor-core paths);
 * compiled by hipcc (ROCm, HIP=1) it maps the CUDA runtime surface
 * backend_cuda.cu uses onto HIP 1:1. The kernel language (__global__,
 * __shared__, <<<>>>) is shared syntax.
 *
 * COLI_GPU_HAS_WMMA: the WMMA tensor-core kernels are guarded by
 * __CUDA_ARCH__ >= 700 (device side) and by this flag at the host dispatch
 * sites. Under HIP, __CUDA_ARCH__ is not defined by hipcc, so we define it
 * here to 700 to activate the WMMA kernel bodies (the guard is a minimum
 * threshold; 700 satisfies >= 700). Note: the grouped_s4_wmma kernel has a
 * stricter guard of __CUDA_ARCH__ >= 750, so it stays a no-op even with
 * __CUDA_ARCH__=700 — see HIP-KNOWN-BUGS.md BUG-002.
 * rocWMMA provides float16_t and the fragment/mma_sync API; the CUDA
 * __half and nvcuda::wmma names are mapped to their rocWMMA equivalents.
 * Note: grouped_s4_wmma uses wmma::experimental::precision::s4 (4-bit)
 * which has no rocWMMA equivalent — that kernel stays a no-op under HIP
 * and falls back to quant_matmul. */
#ifndef COLIBRI_BACKEND_GPU_COMPAT_H
#define COLIBRI_BACKEND_GPU_COMPAT_H

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP__)
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
/* rocWMMA requires matrix cores (MFMA: gfx908+, WMMA: gfx11xx); on other
 * targets (gfx906, gfx101x, gfx103x) its headers static_assert. The Makefile
 * passes -DCOLI_HIP_NO_WMMA for those archs: the WMMA kernel bodies stay
 * compiled out (__CUDA_ARCH__ undefined fails the >= 700 guard) and the host
 * dispatch sites fall back via COLI_GPU_HAS_WMMA=0, same as pre-Volta CUDA.
 * The flag must be set for BOTH hipcc passes (host + device) — an arch-macro
 * check here would let the host dispatch kernels whose bodies were compiled
 * out, silently computing nothing. */
#ifndef COLI_HIP_NO_WMMA
#if __has_include(<rocwmma/rocwmma.hpp>)
#include <rocwmma/rocwmma.hpp>
#define COLI_GPU_HAS_WMMA        1
#define __CUDA_ARCH__            700
#define __half                  rocwmma::float16_t
namespace nvcuda { namespace wmma = ::rocwmma; }
#define __syncwarp()            __syncthreads()
#else
/* WMMA-capable arch but no headers: stop loudly rather than silently build
 * a binary with the tensor-core kernels disabled. */
#error "rocWMMA headers not found. Install rocwmma-dev (or rocm-hip-runtime-dev), or build for a non-WMMA arch (see NO_WMMA_ARCHS in the Makefile)."
#endif
#else
/* Arch has no matrix cores: rocWMMA is not needed and not included. */
#define COLI_GPU_HAS_WMMA        0
#define __syncwarp()            __syncthreads()
#endif
#define cudaError_t              hipError_t
#define cudaSuccess              hipSuccess
#define cudaGetErrorString       hipGetErrorString
#define cudaGetLastError         hipGetLastError
#define cudaSetDevice            hipSetDevice
#define cudaGetDeviceCount       hipGetDeviceCount
#define cudaDeviceProp           hipDeviceProp_t
#define cudaGetDeviceProperties  hipGetDeviceProperties
#define cudaMalloc               hipMalloc
#define cudaFree                 hipFree
#define cudaMemcpy               hipMemcpy
#define cudaMemcpy2D             hipMemcpy2D
#define cudaMemcpyAsync          hipMemcpyAsync
#define cudaMemcpyToSymbol       hipMemcpyToSymbol   /* fmt=6 E8 codebook upload */
#define cudaMemcpyHostToDevice   hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost   hipMemcpyDeviceToHost
#define cudaMemGetInfo           hipMemGetInfo
#define cudaStream_t             hipStream_t
#define cudaStreamCreate         hipStreamCreate
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamNonBlocking    hipStreamNonBlocking
#define cudaStreamDestroy        hipStreamDestroy
#define cudaStreamSynchronize    hipStreamSynchronize
#define cudaStreamWaitEvent      hipStreamWaitEvent
#define cudaDeviceSynchronize    hipDeviceSynchronize
#define cudaEvent_t              hipEvent_t
#define cudaEventCreate          hipEventCreate
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventDisableTiming   hipEventDisableTiming
#define cudaEventDestroy         hipEventDestroy
#define cudaEventRecord          hipEventRecord
#define cudaEventSynchronize     hipEventSynchronize
#define cudaEventElapsedTime     hipEventElapsedTime
#define cudaMallocHost           hipHostMalloc
#define cudaFreeHost             hipHostFree
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemcpyPeer           hipMemcpyPeer
#define cudaMemcpyPeerAsync      hipMemcpyPeerAsync
#define cudaMemsetAsync          hipMemsetAsync
/* Device-side kernel abort (weight_at's undecodable-format backstop,
 * backend_cuda.cu). CUDA's __trap() intrinsic has no HIP name-twin; HIP
 * device code aborts via abort() -- the same mapping hipify applies. NVCC
 * never sees this define, so the CUDA-side semantics the fmt trap test pins
 * (kernel aborts, context poisoned) are untouched. Under HIP this makes the
 * refusal COMPILE; its runtime behavior on AMD silicon is not verified by
 * this repo's tests (same hardware caveat as the rest of this HIP arm). */
#define __trap                   abort
#else
#include <cuda_runtime.h>
#include <mma.h>
#define COLI_GPU_HAS_WMMA        1
#endif

#endif /* COLIBRI_BACKEND_GPU_COMPAT_H */
