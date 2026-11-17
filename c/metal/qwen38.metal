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

/* Hyper-connection element-wise ops (the CPU's q38_gr_read / q38_gr_apply). One kernel each:
 * the same arithmetic in exact and fast mode. sigmoid is q38_sigmoid's two-branch form. */
static inline float q38_sigmoid(float x) {
    if (x >= 0.0f) { float z = exp(-x); return 1.0f / (1.0f + z); }
    float z = exp(x);
    return z / (1.0f + z);
}

struct q38_hc_args { uint n, D, C; float fc; };

/* y[i] = silu(a[i] / fc) */
kernel void q38_hc_low(device const float *a [[buffer(0)]],
                       device float *y [[buffer(1)]],
                       constant q38_hc_args &p [[buffer(2)]],
                       uint i [[thread_position_in_grid]]) {
    if (i >= p.n) return;
    float v = a[i] / p.fc;
    y[i] = v * q38_sigmoid(v);
}

/* y[i] = 2 * sigmoid(a[i] / fc) */
kernel void q38_hc_inj(device const float *a [[buffer(0)]],
                       device float *y [[buffer(1)]],
                       constant q38_hc_args &p [[buffer(2)]],
                       uint i [[thread_position_in_grid]]) {
    if (i >= p.n) return;
    y[i] = 2.0f * q38_sigmoid(a[i] / p.fc);
}

/* y[r][d] = (sum_k sigmoid(a[r][k][d]) * b[r][k][d]) / fc, k ascending; n = rows * D */
kernel void q38_hc_mix(device const float *a [[buffer(0)]],
                       device const float *b [[buffer(1)]],
                       device float *y [[buffer(2)]],
                       constant q38_hc_args &p [[buffer(3)]],
                       uint i [[thread_position_in_grid]]) {
    if (i >= p.n) return;
    ulong r = i / p.D, d = i % p.D;
    ulong base = r * p.C * p.D + d;
    float v = 0.0f;
    for (uint k = 0; k < p.C; k++) v += q38_sigmoid(a[base + (ulong)k * p.D]) * b[base + (ulong)k * p.D];
    y[i] = v / p.fc;
}

/* hyper[r][k][d] += inject[r][k] * block[r][d]; n = rows * C * D */
kernel void q38_hc_apply(device float *hyper [[buffer(0)]],
                         device const float *inject [[buffer(1)]],
                         device const float *block [[buffer(2)]],
                         constant q38_hc_args &p [[buffer(3)]],
                         uint i [[thread_position_in_grid]]) {
    if (i >= p.n) return;
    ulong rk = i / p.D, d = i % p.D, r = rk / p.C;
    hyper[i] += inject[rk] * block[r * p.D + d];
}

/* y[r*C+b][j] = x * r * (1 + w[b*D + j]), r = 1/sqrt(mean(x^2) + eps) (q38_rms0 per stream).
 * One 256-thread group a (row, stream); thread t sums squares t, t+256, ... and thread 0 adds the
 * 256 partials in order (fixed, deterministic), so exact and fast are the same kernel. */
struct q38_hcn_args { uint D, C; float eps; };

kernel void q38_hc_norm(device const float *x [[buffer(0)]],
                        device const float *w [[buffer(1)]],
                        device float *y [[buffer(2)]],
                        constant q38_hcn_args &p [[buffer(3)]],
                        uint g [[threadgroup_position_in_grid]],
                        uint t [[thread_index_in_threadgroup]]) {
    threadgroup float part[256];
    threadgroup float rs;
    ulong row = g;                                  /* rows * C of them */
    device const float *xr = x + row * p.D;
    device const float *wr = w + (row % p.C) * p.D;
    float s = 0.0f;
    for (uint j = t; j < p.D; j += 256) s += xr[j] * xr[j];
    part[t] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
        float ss = 0.0f;
        for (uint l = 0; l < 256; l++) ss += part[l];
        rs = 1.0f / sqrt(ss / (float)p.D + p.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float r = rs;
    for (uint j = t; j < p.D; j += 256) y[row * p.D + j] = xr[j] * r * (1.0f + wr[j]);
}
