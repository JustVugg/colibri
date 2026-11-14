#pragma once

#include <stdint.h>
#if defined(__HIP__) || defined(__HIP_PLATFORM_AMD__)
#include "backend_gpu_compat.h"
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>
/* CUDA's fp8 spellings, mapped onto HIP's OCP e4m3. <hip/hip_fp8.h> has
 * __hip_fp8_e4m3 from ROCm 6.3 (llama.cpp's vendors/hip.h gates it the same
 * way); the V4 tier needs it, so older ROCm stops here with the reason. */
#if HIP_VERSION < 60300000
#error "DeepSeek V4 on HIP needs ROCm 6.3 or newer (OCP fp8 in <hip/hip_fp8.h>)"
#endif
typedef __hip_fp8_e4m3 __nv_fp8_e4m3;
#define __NV_SATFINITE __HIP_SATFINITE
#define __NV_E4M3      __HIP_E4M3
static __device__ __forceinline__ __hip_fp8_storage_t
__nv_cvt_float_to_fp8(float v, int sat, int fmt) {
    return __hip_cvt_float_to_fp8(v, (__hip_saturation_t)sat,
                                  (__hip_fp8_interpretation_t)fmt);
}
#else
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#endif

/* Native conversion follows the same approach as Naruto's
 * cuda_deepseek_quantization.cuh. Keep Colibri's ties-to-even and positive
 * zero convention; Naruto's software encoder uses a different tie rule. */
__device__ __forceinline__ float e4m3(uint8_t bits) {
#if defined(__HIP__) || defined(__HIP_PLATFORM_AMD__)
    unsigned exponent = (bits >> 3) & 15, mantissa = bits & 7;
    float value = exponent ? __uint_as_float(((exponent + 120) << 23) |
                                            (mantissa << 20))
                           : mantissa * 0.001953125f;
    return exponent == 15 && mantissa == 7 ? NAN : bits & 128 ? -value : value;
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    return __half2float(__nv_cvt_fp8_to_halfraw(bits, __NV_E4M3));
#else
    unsigned exponent = (bits >> 3) & 15, mantissa = bits & 7;
    float value = exponent ? __uint_as_float(((exponent + 120) << 23) |
                                            (mantissa << 20))
                           : mantissa * 0.001953125f;
    return exponent == 15 && mantissa == 7 ? NAN : bits & 128 ? -value : value;
#endif
}

__device__ __forceinline__ uint8_t e4m3_code(float value) {
    /* Callers clamp finite input to [-448, 448] before conversion. */
    /* Legacy exhaustive search returns zero for nonfinite input. */
    if (!isfinite(value)) return 0;
    uint8_t bits = __nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3);
    return (bits & 127) == 0 ? 0 : bits;
}

__device__ __forceinline__ float e4m3_round(float value) {
    return e4m3(e4m3_code(value));
}

/* A table decode. Building the float from bits made the generic batched
 * MoE's expert compute 1.5x slower in prefill on sm_86 (2x RTX 3090). */
__device__ __forceinline__ float f4(uint8_t q) {
    const float a[8] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
    return q & 8 ? -a[q & 7] : a[q];
}
