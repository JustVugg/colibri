/* Metal kernels for the Qwen3.8 engine, compiled at run time from the copy
 * built into qwen38_metal.o (#embed), with the options of qwen36's library
 * (safe math, precise functions). They run in qwen36's command stream. */
#include <metal_stdlib>
using namespace metal;

/* y[i] = x[i]: checks that a q38 dispatch runs in order with q36's. */
kernel void q38_copy(device const float *x [[buffer(0)]],
                     device float *y [[buffer(1)]],
                     constant uint &n [[buffer(2)]],
                     uint i [[thread_position_in_grid]]) {
    if (i < n) y[i] = x[i];
}
