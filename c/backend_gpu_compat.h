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
/* rocWMMA has no sub-byte fragments (no s4/int4 precision), and the
 * grouped_s4_wmma body is guarded by __CUDA_ARCH__ >= 750, which the 700
 * defined below does not reach: its body is compiled OUT. The host gate
 * used to test rocWMMA availability alone, took the branch, launched three
 * empty kernels and returned the previous call's output untouched (#1499).
 * A second flag for the one kernel that needs more than WMMA. */
#define COLI_GPU_HAS_S4_WMMA     0
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
#define COLI_GPU_HAS_S4_WMMA     0
#define __syncwarp()            __syncthreads()
#endif
#define cudaError_t              hipError_t
#define cudaSuccess              hipSuccess
#define cudaErrorMemoryAllocation hipErrorOutOfMemory
#define cudaErrorNotReady        hipErrorNotReady
#define cudaGetErrorString       hipGetErrorString
#define cudaGetLastError         hipGetLastError
#define cudaPeekAtLastError      hipPeekAtLastError   /* cu_xb.cu */
#define cudaSetDevice            hipSetDevice
#define cudaGetDeviceCount       hipGetDeviceCount
#define cudaDeviceProp           hipDeviceProp_t
#define cudaGetDeviceProperties  hipGetDeviceProperties
#define cudaMalloc               hipMalloc
#define cudaMallocManaged        hipMallocManaged
#define cudaFree                 hipFree
#define cudaMemcpy               hipMemcpy
#define cudaMemcpy2D             hipMemcpy2D
#define cudaFuncSetAttribute     hipFuncSetAttribute
#define cudaGetDevice            hipGetDevice
#define cudaFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize
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
#define cudaEventQuery           hipEventQuery
#define cudaEventElapsedTime     hipEventElapsedTime
#define cudaMallocHost           hipHostMalloc
#define cudaFreeHost             hipHostFree
#define cudaHostAlloc            hipHostMalloc        /* cuda_chain.cu: mapped host buffers */
#define cudaHostAllocMapped      hipHostMallocMapped
#define cudaHostAllocPortable    hipHostMallocPortable   /* cu_xb.cu: the tier's host pool and scratch */
#define cudaHostAllocWriteCombined hipHostMallocWriteCombined
#define cudaHostGetDevicePointer hipHostGetDevicePointer
#define cudaHostRegister         hipHostRegister      /* the qwen36 tier's host pool */
#define cudaHostRegisterPortable hipHostRegisterPortable
#define cudaHostUnregister       hipHostUnregister
#define cudaMemcpyDefault        hipMemcpyDefault
#define cudaMemcpyHostToHost     hipMemcpyHostToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemcpyPeer           hipMemcpyPeer
#define cudaMemcpyPeerAsync      hipMemcpyPeerAsync
#define cudaMemsetAsync          hipMemsetAsync
#define cudaMemset               hipMemset
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
/* nvcuda::wmma::experimental::precision::s4 exists here; the kernel body still
 * needs sm_75+, which the host checks at run time from the device's compute
 * capability (see the grouped_s4_wmma dispatch in backend_cuda.cu). */
#define COLI_GPU_HAS_S4_WMMA     1
#endif

/* Programmatic dependent launch (sm_90 and later). A kernel coli_launch starts may be set up
 * while the kernel before it on the stream ends, so the launch's latency and the last blocks'
 * tail no longer leave the device idle between them. Such a kernel's first statement is
 * COLI_PDL_WAIT(): it returns once every grid before it on the stream has completed and its
 * writes are visible, so the kernel reads and writes what it would after a plain launch.
 * Elsewhere (HIP, older archs, COLI_CUDA_PDL=0, a fatbin whose code for this device was built
 * below sm_90, the legacy default stream) coli_launch is a plain launch and the wait returns at
 * once. */
#if !defined(__HIP_PLATFORM_AMD__) && !defined(__HIP__) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define COLI_PDL_WAIT()          asm volatile("griddepcontrol.wait;" ::: "memory")
#else
#define COLI_PDL_WAIT()          ((void)0)
#endif
#if !defined(__HIP_PLATFORM_AMD__) && !defined(__HIP__)
#include <cstdlib>
__global__ static void coli_pdl_probe_k(void) {}
/* per translation unit (each .cu's fatbin is probed): on where the device is sm_90 or later
 * and this unit's code for it was compiled from sm_90 or later (code JIT-compiled from older
 * PTX has no wait in its kernels) */
static inline int coli_pdl_on(void) {
    static int env = -1;
    static signed char on[16];
    static unsigned char known[16];
    if (env < 0) { const char *e = std::getenv("COLI_CUDA_PDL"); env = !(e && *e == '0'); }
    if (!env) return 0;
    int d = 0;
    if (cudaGetDevice(&d) != cudaSuccess || d < 0 || d >= 16) { (void)cudaGetLastError(); return 0; }
    if (!known[d]) {
        int major = 0;
        cudaFuncAttributes fa;
        on[d] = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, d) == cudaSuccess && major >= 9 &&
                cudaFuncGetAttributes(&fa, coli_pdl_probe_k) == cudaSuccess && fa.ptxVersion >= 90;
        (void)cudaGetLastError();
        known[d] = 1;
    }
    return on[d];
}
template <typename... P, typename... A>
static inline void coli_launch(void (*k)(P...), dim3 g, dim3 b, size_t sm, cudaStream_t st, A... a) {
    if (st && coli_pdl_on()) {   /* not the legacy default stream, which joins the others */
        cudaLaunchAttribute at[1];
        at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        at[0].val.programmaticStreamSerializationAllowed = 1;
        cudaLaunchConfig_t cf = {};
        cf.gridDim = g; cf.blockDim = b; cf.dynamicSmemBytes = sm; cf.stream = st; cf.attrs = at; cf.numAttrs = 1;
        (void)cudaLaunchKernelEx(&cf, k, a...);   /* an error stays for the caller's cudaGetLastError */
        return;
    }
    k<<<g, b, sm, st>>>(a...);
}
#else
template <typename... P, typename... A>
static inline void coli_launch(void (*k)(P...), dim3 g, dim3 b, size_t sm, cudaStream_t st, A... a) { k<<<g, b, sm, st>>>(a...); }
#endif

#endif /* COLIBRI_BACKEND_GPU_COMPAT_H */
