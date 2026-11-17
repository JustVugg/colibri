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

/* ---- PLE (qwen38_core.h's q38_ple from the gate on; shaders/chain_ple.comp) ---- */

struct q38_ple_args { uint S, C, W, H, CK, NG, snap_rows, accumulate; float eps; };

/* The sum of the 256 partials in part[], added in order by thread 0 and returned to all
 * (fixed, deterministic order: exact and fast are the same kernel). */
static inline float q38_tg_sum256(threadgroup float *part, threadgroup float *out, float v, uint t) {
    part[t] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
        float s = 0.0f;
        for (uint l = 0; l < 256; l++) s += part[l];
        *out = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float s = *out;
    threadgroup_barrier(mem_flags::mem_threadgroup);   /* part[] is reused by the next sum */
    return s;
}

/* The gate, one 256-thread group a (row s, stream k), g = s*C + k; H floats of each:
 * kn = rms0(keys[s][k], norm_key[k]), qn = rms0(hyper[s][k], norm_query[k]),
 * dot = kn.qn / sqrt(H), g = sigmoid(copysign(sqrt(max(|dot|, 1e-6)), dot)),
 * gated[s][k] = g * value[s], normv[s][k] = rms0(gated[s][k], norm_conv[k]).
 * prm = norm_key[W] | norm_query[W] | norm_conv[W]. */
kernel void q38_ple_gate(device const float *keys [[buffer(0)]],
                         device const float *hyper [[buffer(1)]],
                         device const float *value [[buffer(2)]],
                         device const float *prm [[buffer(3)]],
                         device float *gated [[buffer(4)]],
                         device float *normv [[buffer(5)]],
                         constant q38_ple_args &p [[buffer(6)]],
                         uint g [[threadgroup_position_in_grid]],
                         uint t [[thread_index_in_threadgroup]]) {
    threadgroup float part[256];
    threadgroup float sum;
    uint s = g / p.C, k = g % p.C;
    ulong o = (ulong)s * p.W + (ulong)k * p.H;
    device const float *kr = keys + o, *hr = hyper + o, *vr = value + (ulong)s * p.H;
    device const float *wk = prm + (ulong)k * p.H, *wq = prm + p.W + (ulong)k * p.H, *wc = prm + 2 * (ulong)p.W + (ulong)k * p.H;
    float a = 0.0f, b = 0.0f;
    for (uint d = t; d < p.H; d += 256) { a += kr[d] * kr[d]; b += hr[d] * hr[d]; }
    float rk = 1.0f / sqrt(q38_tg_sum256(part, &sum, a, t) / (float)p.H + p.eps);
    float rq = 1.0f / sqrt(q38_tg_sum256(part, &sum, b, t) / (float)p.H + p.eps);
    float dt = 0.0f;
    for (uint d = t; d < p.H; d += 256) dt += (kr[d] * rk * (1.0f + wk[d])) * (hr[d] * rq * (1.0f + wq[d]));
    float dot = q38_tg_sum256(part, &sum, dt, t) / sqrt((float)p.H);
    float gt = q38_sigmoid(copysign(sqrt(max(fabs(dot), 1e-6f)), dot));
    float c = 0.0f;
    for (uint d = t; d < p.H; d += 256) {
        float u = gt * vr[d];
        gated[o + d] = u;
        c += u * u;
    }
    float rc = 1.0f / sqrt(q38_tg_sum256(part, &sum, c, t) / (float)p.H + p.eps);
    for (uint d = t; d < p.H; d += 256) normv[o + d] = (gt * vr[d]) * rc * (1.0f + wc[d]);
}

/* The dilated conv, one thread per channel d < W, rows in order. The ring of d holds SL =
 * (CK-1)*NG entries (oldest first) and the taps read every NG-th:
 * a = conv[d][CK-1]*normv[s][d] + sum_{k<CK-1} conv[d][k]*ring[k*NG]; v = gated[s][d] + silu(a);
 * out[s][d] = v or += v (accumulate); the ring drops its oldest and takes normv[s][d]; for
 * s < snap_rows the ring is copied to snap[s]. */
kernel void q38_ple_conv(device float *out [[buffer(0)]],
                         device float *ring [[buffer(1)]],
                         device float *snap [[buffer(2)]],
                         device const float *gated [[buffer(3)]],
                         device const float *normv [[buffer(4)]],
                         device const float *conv [[buffer(5)]],
                         constant q38_ple_args &p [[buffer(6)]],
                         uint d [[thread_position_in_grid]]) {
    if (d >= p.W) return;
    uint SL = (p.CK - 1) * p.NG;
    float rg[32];
    device float *rd = ring + (ulong)d * SL;
    for (uint j = 0; j < SL; j++) rg[j] = rd[j];
    device const float *cw = conv + (ulong)d * p.CK;
    float wl = cw[p.CK - 1];
    for (uint s = 0; s < p.S; s++) {
        ulong i = (ulong)s * p.W + d;
        float nv = normv[i];
        float a = wl * nv;
        for (uint k = 0; k + 1 < p.CK; k++) a += cw[k] * rg[k * p.NG];
        float v = gated[i] + a * q38_sigmoid(a);
        if (p.accumulate) out[i] += v;
        else out[i] = v;
        for (uint j = 0; j + 1 < SL; j++) rg[j] = rg[j + 1];
        rg[SL - 1] = nv;
        if (s < p.snap_rows) {
            device float *sd = snap + ((ulong)s * p.W + d) * SL;
            for (uint j = 0; j < SL; j++) sd[j] = rg[j];
        }
    }
    for (uint j = 0; j < SL; j++) rd[j] = rg[j];
}
