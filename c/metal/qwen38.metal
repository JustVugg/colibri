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

/* y[r][o] = sum_j x[r][j] * bf16(w[o][j]): one 32-lane group a (row, out) cell.
 * Lane l takes the 8-wide chunks l, l+32, ... (j ascending inside a chunk);
 * the exact kernel adds the 32 lane sums in lane order from threadgroup memory
 * (fixed, deterministic order), the fast one with simd_sum. */
struct q38_dot_args { uint rows, in, out; };

static inline float q38_bf16(ushort w) { return as_type<float>(uint(w) << 16); }

static inline float q38_dot_lane(device const float *xr, device const ushort *wo, uint in, uint lane) {
    float s = 0.0f;
    for (uint j = lane * 8; j < in; j += 256) {
        for (uint k = 0; k < 8; k++) s += xr[j + k] * q38_bf16(wo[j + k]);
    }
    return s;
}

kernel void q38_dot_bf16_exact(device const float *x [[buffer(0)]],
                               device const ushort *w [[buffer(1)]],
                               device float *y [[buffer(2)]],
                               constant q38_dot_args &a [[buffer(3)]],
                               uint2 g [[threadgroup_position_in_grid]],
                               uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float part[32];
    uint o = g.x, r = g.y;
    part[lane] = q38_dot_lane(x + (ulong)r * a.in, w + (ulong)o * a.in, a.in, lane);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
        float s = 0.0f;
        for (uint l = 0; l < 32; l++) s += part[l];
        y[(ulong)r * a.out + o] = s;
    }
}

kernel void q38_dot_bf16_fast(device const float *x [[buffer(0)]],
                              device const ushort *w [[buffer(1)]],
                              device float *y [[buffer(2)]],
                              constant q38_dot_args &a [[buffer(3)]],
                              uint2 g [[threadgroup_position_in_grid]],
                              uint lane [[thread_index_in_threadgroup]]) {
    uint o = g.x, r = g.y;
    float s = simd_sum(q38_dot_lane(x + (ulong)r * a.in, w + (ulong)o * a.in, a.in, lane));
    if (lane == 0) y[(ulong)r * a.out + o] = s;
}
