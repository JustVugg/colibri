/* backend_gpu_compat.h — one GPU backend source, two vendors.
 * Same pattern as compat.h for Windows: every platform difference lives in
 * this header and backend_cuda.cu stays untouched. Compiled by nvcc this is
 * a pass-through to the CUDA runtime; compiled by hipcc (ROCm, HIP=1) it maps
 * the exact CUDA runtime surface backend_cuda.cu uses onto HIP 1:1. The
 * kernel language (__global__, __shared__, <<<>>>) is shared syntax. */
#ifndef COLIBRI_BACKEND_GPU_COMPAT_H
#define COLIBRI_BACKEND_GPU_COMPAT_H

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP__)
#include <hip/hip_runtime.h>
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
#define cudaMemcpyHostToDevice   hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost   hipMemcpyDeviceToHost
#define cudaMemGetInfo           hipMemGetInfo
#else
#include <cuda_runtime.h>
#endif

#endif
