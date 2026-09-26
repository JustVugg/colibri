#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

/* ---- types ---- */
typedef hipStream_t     cudaStream_t;
typedef hipError_t      cudaError_t;
typedef hipEvent_t      cudaEvent_t;
typedef hipGraph_t      cudaGraph_t;
typedef hipGraphExec_t  cudaGraphExec_t;
typedef hipDeviceProp_t cudaDeviceProp;

/* ---- macro aliases ---- */
#define cudaSuccess                            hipSuccess
#define cudaDeviceScheduleSpin                 hipDeviceScheduleSpin
#define cudaSetDeviceFlags                     hipSetDeviceFlags
#define cudaMallocHost(ptr, size)              hipHostMalloc((void**)(ptr), (size))
#define cudaErrorPeerAccessAlreadyEnabled      hipErrorPeerAccessAlreadyEnabled
#define cudaErrorHostMemoryAlreadyRegistered   hipErrorHostMemoryAlreadyRegistered

#define cudaDevAttrIntegrated                              hipDeviceAttributeIntegrated
#define cudaDevAttrPageableMemoryAccessUsesHostPageTables  hipDeviceAttributePageableMemoryAccessUsesHostPageTables

#define cudaStreamNonBlocking               hipStreamNonBlocking
#define cudaStreamCaptureModeThreadLocal    hipStreamCaptureModeThreadLocal
#define cudaEventDisableTiming              hipEventDisableTiming
#define cudaHostRegisterPortable            hipHostRegisterPortable
#define cudaFuncAttributeMaxDynamicSharedMemorySize  hipFuncAttributeMaxDynamicSharedMemorySize

#define cudaMemcpyHostToDevice      hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost      hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice    hipMemcpyDeviceToDevice

#define cudaSetDevice               hipSetDevice
#define cudaGetDevice               hipGetDevice
#define cudaGetDeviceProperties     hipGetDeviceProperties
#define cudaDeviceGetAttribute      hipDeviceGetAttribute
#define cudaDeviceCanAccessPeer     hipDeviceCanAccessPeer
#define cudaDeviceEnablePeerAccess  hipDeviceEnablePeerAccess
#define cudaDeviceSynchronize       hipDeviceSynchronize
#define cudaGetErrorString          hipGetErrorString
#define cudaGetLastError            hipGetLastError

#define cudaStreamCreate            hipStreamCreate
#define cudaStreamCreateWithFlags   hipStreamCreateWithFlags
#define cudaStreamDestroy           hipStreamDestroy
#define cudaStreamSynchronize       hipStreamSynchronize
#define cudaStreamWaitEvent         hipStreamWaitEvent
#define cudaStreamBeginCapture      hipStreamBeginCapture
#define cudaStreamEndCapture        hipStreamEndCapture

#define cudaEventCreate             hipEventCreate
#define cudaEventCreateWithFlags    hipEventCreateWithFlags
#define cudaEventDestroy            hipEventDestroy
#define cudaEventRecord             hipEventRecord
#define cudaEventSynchronize        hipEventSynchronize
#define cudaEventElapsedTime        hipEventElapsedTime

#define cudaMalloc                  hipMalloc
#define cudaFree                    hipFree
#define cudaFreeHost                hipFreeHost
#define cudaHostRegister            hipHostRegister
#define cudaMemcpy                  hipMemcpy
#define cudaMemcpy2D                hipMemcpy2D
#define cudaMemcpyAsync             hipMemcpyAsync
#define cudaMemcpyPeerAsync         hipMemcpyPeerAsync
#define cudaMemcpyToSymbol          hipMemcpyToSymbol
#define cudaMemGetInfo              hipMemGetInfo
#define cudaMemset                  hipMemset
#define cudaMemsetAsync             hipMemsetAsync

#define cudaGraphCreate             hipGraphCreate
#define cudaGraphLaunch             hipGraphLaunch
#define cudaGraphDestroy            hipGraphDestroy
#define cudaGraphExecDestroy        hipGraphExecDestroy

#define cudaProfilerStart           hipProfilerStart
#define cudaProfilerStop            hipProfilerStop

/* ---- warp shuffles (CUDA-style mask ignored on AMD) ---- */
#define __shfl_down_sync(mask, var, delta)  __shfl_down((var), (delta))
#define __shfl_up_sync(mask, var, delta)    __shfl_up((var), (delta))
#define __shfl_sync(mask, var, src)         __shfl((var), (src))
#define __shfl_xor_sync(mask, var, lane)    __shfl_xor((var), (lane))

/* ---- __syncwarp: CUDA allows 0 args; HIP requires a mask ---- */
#define __syncwarp(...) __builtin_amdgcn_wave_barrier()

/* ---- cudaFuncSetAttribute: template so any function-pointer type works ---- */
template <typename T>
static inline hipError_t __coli_cudaFuncSetAttribute(T* f, int a, int v) {
    return hipFuncSetAttribute(reinterpret_cast<const void*>(f), (hipFuncAttribute)a, v);
}
#define cudaFuncSetAttribute __coli_cudaFuncSetAttribute

/* ---- cudaGraphInstantiate: CUDA has 3 args, HIP has 5 ---- */
static inline hipError_t __coli_cudaGraphInstantiate(hipGraphExec_t* e, hipGraph_t g, unsigned long long flags) {
    (void)flags;
    return hipGraphInstantiate(e, g, nullptr, nullptr, 0);
}
#define cudaGraphInstantiate __coli_cudaGraphInstantiate
