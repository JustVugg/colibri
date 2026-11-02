#pragma once

#include <stdint.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

/* Native conversion follows the same approach as Naruto's
 * cuda_deepseek_quantization.cuh. Keep Colibri's ties-to-even and positive
 * zero convention; Naruto's software encoder uses a different tie rule. */
__device__ __forceinline__ float e4m3(uint8_t bits) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
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
