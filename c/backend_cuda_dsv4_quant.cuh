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

__device__ __forceinline__ float f4(uint8_t bits) {
    unsigned magnitude = bits & 7;
    unsigned value = magnitude ? (((magnitude / 2 + 126) << 23) |
                                 ((magnitude > 1 ? magnitude & 1 : 0) << 22)) : 0;
    return __uint_as_float(((unsigned)(bits & 8) << 28) | value);
}
