// Qwen3.6 Metal kernels.
//
// Exact kernels keep one fixed arithmetic: operand order, fma placement and
// reduction shapes are deliberate, so their results do not depend on the
// batch size and stay the same bits from version to version. Do not turn a
// serial loop into a SIMD reduction (or the reverse) without checking the
// exact outputs again. Fast kernels deliberately change reduction
// order and are checked numerically and with model likelihood. Fast routing
// can swap nearly tied experts; its quality is checked on the model too.
//
// Layouts (all float32 unless noted, row-major, "rows" = tokens in a batch):
//   dense int8 weights  q[out][in] int8, sc[out] float (one scale per row)
//   expert weights      one slot per expert: planar int4 gate|up|down, then
//                       float scales per 64 inputs (see q36_dot4)
//   DeltaNet state      [vheads][kdim][vdim] high plane, then the low plane
//   conv history        [conv_dim][conv_kernel-1]
//   KV cache            [kv_heads][capacity][head_dim]
#include <metal_stdlib>
using namespace metal;

// Keep these in the same order as their twins in qwen36_metal.m.
struct q36_dense_args { uint n, rows, in, out, mode; float eps; uint group, stride, ld; };   // group, stride, ld: q36_dot_i8_mm
struct q36_state_args {
    uint rows, hidden, qh, kv, hd, qd, rotary, heads;
    uint vh, kh, kd, vd, convk, convd, pos, cap;
    uint snap;      // DeltaNet: copy the state after this row into the snapshot (~0u: no copy)
    uint low;       // DeltaNet: the state has its low plane (exact mode); fast kernels skip it otherwise
};
struct q36_router_args { uint rows, H, E, K, G, T, shared_gate; };
struct q36_expert_args { uint H, I, K, rows, first, n; ulong scales; };

// The place of a lane's two elements in an 8x8 SIMD-group matrix (thread_elements):
// row sm, columns sn and sn + 1.
static inline ushort2 q36_lane_place(ushort lane) {
    ushort qid = lane / 4;
    return ushort2((qid & 2) * 2 + (lane % 2) * 2, (qid & 4) + (lane / 2) % 4);   // (sn, sm)
}

inline float q36_sigmoid(float x) {
    if (x >= 0) return 1.0f / (1.0f + exp(-x));
    float e = exp(x);
    return e / (1.0f + e);
}

inline float q36_softplus(float x) {
    if (x > 20.0f) return x;
    float e = exp(x), u = 1.0f + e;
    return u == 1.0f ? e : log(u) * (e / (u - 1.0f));
}

// acc + a.x*b.x, then + a.y*b.y, + a.z*b.z, + a.w*b.w: each product rounded
// before an ordered addition. The arithmetic of a serial loop, with the data
// loaded four elements at a time.
inline float q36_ordered4(float acc, float4 a, float4 b) {
#pragma clang fp contract(off)
    float4 pr = a * b;
    acc = acc + pr.x;
    acc = acc + pr.y;
    acc = acc + pr.z;
    return acc + pr.w;
}

inline float4 q36_load4(device const float *p) { return float4(*(device const packed_float4 *)p); }

// ---------------------------------------------------------------------------
// Dense trunk: norm, int8 projections, gates, residual.
// ---------------------------------------------------------------------------

// One thread per row. mode 0: Qwen zero-centered gamma (1+w); mode 1: plain w.
// Safe in place (y == x): each element is read before it is written. Loads
// and stores go four at a time; the fma chain keeps its order.
kernel void q36_rmsnorm(device const float *x [[buffer(0)]],
                        device const float *w [[buffer(1)]],
                        device float *y [[buffer(2)]],
                        constant q36_dense_args &p [[buffer(3)]],
                        uint row [[thread_position_in_grid]]) {
    if (row >= p.rows) return;
    device const float *xr = x + ulong(row) * p.in;
    device float *yr = y + ulong(row) * p.in;
    float ss = 0;
    uint j = 0;
    for (; j + 4 <= p.in; j += 4) {
        float4 v = q36_load4(xr + j);
        ss = fma(v.x, v.x, ss);
        ss = fma(v.y, v.y, ss);
        ss = fma(v.z, v.z, ss);
        ss = fma(v.w, v.w, ss);
    }
    for (; j < p.in; j++) ss = fma(xr[j], xr[j], ss);
    float r = 1.0f / sqrt(ss / float(p.in) + p.eps);
    for (j = 0; j + 4 <= p.in; j += 4) {
        float4 v = q36_load4(xr + j), g = q36_load4(w + j);
        *(device packed_float4 *)(yr + j) = packed_float4(v * r * (p.mode ? g : 1.0f + g));
    }
    for (; j < p.in; j++) yr[j] = xr[j] * r * (p.mode ? w[j] : 1.0f + w[j]);
}

// FAST RMSNorm: one 32-lane SIMD group per row. Each lane sums its
// strided elements in FP32; the SIMD reduction changes the serial sum order.
// No dimension alignment or scratch bound is required. y == x is supported:
// all reads for the norm finish before any lane overwrites an input element.
kernel void q36_rmsnorm_fast(device const float *x [[buffer(0)]],
                             device const float *w [[buffer(1)]],
                             device float *y [[buffer(2)]],
                             constant q36_dense_args &p [[buffer(3)]],
                             uint row [[threadgroup_position_in_grid]],
                             ushort lane [[thread_index_in_simdgroup]]) {
    if (row >= p.rows) return;                 // uniform for the whole group
    device const float *xr = x + ulong(row) * p.in;
    device float *yr = y + ulong(row) * p.in;
    float ss = 0;
    for (uint j = lane; j < p.in; j += 32) {
        float v = xr[j];
        ss = fma(v, v, ss);
    }
    ss = simd_sum(ss);
    float r = 1.0f / sqrt(ss / float(p.in) + p.eps);
    threadgroup_barrier(mem_flags::mem_device);
    for (uint j = lane; j < p.in; j += 32) {
        float v = xr[j], g = w[j];
        yr[j] = v * r * (p.mode ? g : 1.0f + g);
    }
}

// q36_rmsnorm_fast for a few rows (decode): a threadgroup of 256 threads per row, four floats
// at a time, the sum of squares added by all of them (SIMD sums, then the 8 SIMD groups'
// sums). One SIMD group took 50 us for a row of 2048 waiting on its loads. Needs dim % 4 == 0
// and 16-byte aligned rows.
kernel void q36_rmsnorm_wide(device const float *x [[buffer(0)]],
                             device const float *w [[buffer(1)]],
                             device float *y [[buffer(2)]],
                             constant q36_dense_args &p [[buffer(3)]],
                             uint row [[threadgroup_position_in_grid]],
                             ushort tid [[thread_index_in_threadgroup]],
                             ushort sg [[simdgroup_index_in_threadgroup]],
                             ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[8];
    device const float4 *xr = (device const float4 *)(x + ulong(row) * p.in);
    device const float4 *w4 = (device const float4 *)w;
    device float4 *yr = (device float4 *)(y + ulong(row) * p.in);
    const uint n4 = p.in / 4;
    float4 s4 = 0;
    for (uint j = tid; j < n4; j += 256) {
        float4 v = xr[j];
        s4 = fma(v, v, s4);
    }
    float ss = simd_sum((s4.x + s4.y) + (s4.z + s4.w));
    if (lane == 0) part[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ss = ((part[0] + part[1]) + (part[2] + part[3])) + ((part[4] + part[5]) + (part[6] + part[7]));
    float r = 1.0f / sqrt(ss / float(p.in) + p.eps);
    for (uint j = tid; j < n4; j += 256) {
        float4 v = xr[j], g = w4[j];
        yr[j] = v * r * (p.mode ? g : 1.0f + g);
    }
}

// y[row][o] = sc[o] * sum_i q[o][i] * x[row][i]. One SIMD group per output.
// Also the lm_head projection (rows = 1).
kernel void q36_dot_i8(device const char *q [[buffer(0)]],
                       device const float *sc [[buffer(1)]],
                       device const float *x [[buffer(2)]],
                       device float *y [[buffer(3)]],
                       constant q36_dense_args &p [[buffer(4)]],
                       uint z [[threadgroup_position_in_grid]],
                       ushort lane [[thread_index_in_simdgroup]]) {
    if (z >= p.rows * p.out) return;
    uint row = z / p.out, o = z % p.out;
    float acc = 0;
    for (uint i = lane; i < p.in; i += 32)
        acc = fma(float(q[ulong(o) * p.in + i]), x[ulong(row) * p.in + i], acc);
    acc = simd_sum(acc);
    if (lane == 0) y[ulong(row) * p.out + o] = acc * sc[o];
}

// q36_dot_i8 for four consecutive outputs of a row per SIMD group (exact mode, out a multiple of 4): each
// output's lane chains, simd_sum and scale are the same operations as in q36_dot_i8, so the same bits; the four
// share each input load and keep four rows of weights in flight. At the 35B model's decode shapes 15-28% less
// GPU time than one output per SIMD group (2048 x 4096: 0.212 -> 0.152 ms; the head 10.6 -> 9.8 ms); eight
// outputs were no faster.
kernel void q36_dot_i8_r4(device const char *q [[buffer(0)]],
                          device const float *sc [[buffer(1)]],
                          device const float *x [[buffer(2)]],
                          device float *y [[buffer(3)]],
                          constant q36_dense_args &p [[buffer(4)]],
                          uint z [[threadgroup_position_in_grid]],
                          ushort lane [[thread_index_in_simdgroup]]) {
    uint base = z * 4;
    if (base >= p.rows * p.out) return;
    uint row = base / p.out, o = base % p.out;
    device const char *q0 = q + ulong(o) * p.in;
    device const float *xr = x + ulong(row) * p.in;
    float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    for (uint i = lane; i < p.in; i += 32) {
        float xv = xr[i];
        a0 = fma(float(q0[i]), xv, a0);
        a1 = fma(float(q0[p.in + i]), xv, a1);
        a2 = fma(float(q0[2 * p.in + i]), xv, a2);
        a3 = fma(float(q0[3 * ulong(p.in) + i]), xv, a3);
    }
    a0 = simd_sum(a0); a1 = simd_sum(a1); a2 = simd_sum(a2); a3 = simd_sum(a3);
    if (lane == 0) {
        device float *yr = y + ulong(row) * p.out + o;
        yr[0] = a0 * sc[o]; yr[1] = a1 * sc[o + 1]; yr[2] = a2 * sc[o + 2]; yr[3] = a3 * sc[o + 3];
    }
}

// Fast GEMV: four consecutive inputs per lane, with independent FMA chains.
// Packed loads avoid imposing any alignment beyond that of the scalar input.
// The reduction order differs from q36_dot_i8; the exact path keeps that kernel.
kernel void q36_dot_i8_vec4(device const char *q [[buffer(0)]],
                            device const float *sc [[buffer(1)]],
                            device const float *x [[buffer(2)]],
                            device float *y [[buffer(3)]],
                            constant q36_dense_args &p [[buffer(4)]],
                            uint z [[threadgroup_position_in_grid]],
                            ushort lane [[thread_index_in_simdgroup]]) {
    if (z >= p.rows * p.out) return;
    uint row = z / p.out, o = z % p.out;
    device const char *qr = q + ulong(o) * p.in;
    device const float *xr = x + ulong(row) * p.in;
    float4 acc = 0;
    for (uint i = uint(lane) * 4; i < p.in; i += 128) {
        char4 w = char4(*(device const packed_char4 *)(qr + i));
        acc = fma(float4(w), q36_load4(xr + i), acc);
    }
    float sum = simd_sum((acc.x + acc.y) + (acc.z + acc.w));
    if (lane == 0) y[ulong(row) * p.out + o] = sum * sc[o];
}

// mode 0: y = x * sigmoid(z) (attention output gate).
// mode 1: y = x * silu(z)    (DeltaNet gated norm).
kernel void q36_gate(device const float *x [[buffer(0)]],
                     device const float *z [[buffer(1)]],
                     device float *y [[buffer(2)]],
                     constant q36_dense_args &p [[buffer(3)]],
                     uint i [[thread_position_in_grid]]) {
    if (i >= p.n) return;
    float a = z[i];
    y[i] = x[i] * q36_sigmoid(a) * (p.mode ? a : 1.0f);
}

kernel void q36_add(device const float *x [[buffer(0)]],
                    device const float *r [[buffer(1)]],
                    device float *y [[buffer(2)]],
                    constant q36_dense_args &p [[buffer(3)]],
                    uint i [[thread_position_in_grid]]) {
    if (i >= p.n) return;
    y[i] = x[i] + r[i];
}

// ---------------------------------------------------------------------------
// Gated DeltaNet (linear attention) layers.
// ---------------------------------------------------------------------------

// Per (row, value head): the a/b projections, then decay = exp(-exp(A_log) *
// softplus(a + dt_bias)) and beta = sigmoid(b).
kernel void q36_dn_aux(device const float *x [[buffer(0)]],
                       device const float *a [[buffer(1)]],
                       device const float *b [[buffer(2)]],
                       device const float *alog [[buffer(3)]],
                       device const float *dt [[buffer(4)]],
                       device float *decay [[buffer(5)]],
                       device float *beta [[buffer(6)]],
                       constant q36_state_args &p [[buffer(7)]],
                       uint z [[threadgroup_position_in_grid]],
                       ushort lane [[thread_index_in_simdgroup]]) {
    if (z >= p.rows * p.vh) return;
    uint row = z / p.vh, h = z % p.vh;
    float av = 0, bv = 0;
    for (uint i = lane; i < p.hidden; i += 32) {
        float v = x[ulong(row) * p.hidden + i];
        av = fma(a[ulong(h) * p.hidden + i], v, av);
        bv = fma(b[ulong(h) * p.hidden + i], v, bv);
    }
    av = simd_sum(av);
    bv = simd_sum(bv);
    if (lane == 0) {
        float ea = exp(alog[h]), sp = q36_softplus(av + dt[h]), gg = -ea * sp;
        decay[z] = exp(gg);
        beta[z] = q36_sigmoid(bv);
    }
}

// q36_dn_aux for 4 rows at a time: the head's a and b rows are read once for the 4 rows. Every
// row's sums are q36_dn_aux's lane chains and simd_sum: the same bits. Rows past the batch read
// its last one and are not written.
kernel void q36_dn_aux4(device const float *x [[buffer(0)]],
                        device const float *a [[buffer(1)]],
                        device const float *b [[buffer(2)]],
                        device const float *alog [[buffer(3)]],
                        device const float *dt [[buffer(4)]],
                        device float *decay [[buffer(5)]],
                        device float *beta [[buffer(6)]],
                        constant q36_state_args &p [[buffer(7)]],
                        uint z [[threadgroup_position_in_grid]],
                        ushort lane [[thread_index_in_simdgroup]]) {
    if (z >= (p.rows + 3) / 4 * p.vh) return;
    const uint r0 = z / p.vh * 4, h = z % p.vh;
    device const float *wa = a + ulong(h) * p.hidden, *wb = b + ulong(h) * p.hidden;
    device const float *x0 = x + ulong(r0) * p.hidden;
    device const float *x1 = x + ulong(min(r0 + 1, p.rows - 1)) * p.hidden;
    device const float *x2 = x + ulong(min(r0 + 2, p.rows - 1)) * p.hidden;
    device const float *x3 = x + ulong(min(r0 + 3, p.rows - 1)) * p.hidden;
    float a0 = 0, a1 = 0, a2 = 0, a3 = 0, b0 = 0, b1 = 0, b2 = 0, b3 = 0;
    for (uint i = lane; i < p.hidden; i += 32) {
        float ca = wa[i], cb = wb[i], v0 = x0[i], v1 = x1[i], v2 = x2[i], v3 = x3[i];
        a0 = fma(ca, v0, a0); b0 = fma(cb, v0, b0);
        a1 = fma(ca, v1, a1); b1 = fma(cb, v1, b1);
        a2 = fma(ca, v2, a2); b2 = fma(cb, v2, b2);
        a3 = fma(ca, v3, a3); b3 = fma(cb, v3, b3);
    }
    float av[4] = {simd_sum(a0), simd_sum(a1), simd_sum(a2), simd_sum(a3)};
    float bv[4] = {simd_sum(b0), simd_sum(b1), simd_sum(b2), simd_sum(b3)};
    if (lane != 0) return;
    float ea = exp(alog[h]);
    for (uint k = 0; k < 4 && r0 + k < p.rows; k++) {
        float sp = q36_softplus(av[k] + dt[h]), gg = -ea * sp;
        decay[(r0 + k) * p.vh + h] = exp(gg);
        beta[(r0 + k) * p.vh + h] = q36_sigmoid(bv[k]);
    }
}

// Causal depthwise conv over the qkv projection, then SiLU. One thread per
// channel walks the rows in order, so the history ring stays in step.
kernel void q36_dn_conv(device const float *x [[buffer(0)]],
                        device const float *w [[buffer(1)]],
                        device float *ring [[buffer(2)]],
                        device float *y [[buffer(3)]],
                        constant q36_state_args &p [[buffer(4)]],
                        device float *snap [[buffer(5)]],
                        uint c [[thread_position_in_grid]]) {
    if (c >= p.convd) return;
    device float *hist = ring + ulong(c) * (p.convk - 1);
    device const float *wc = w + ulong(c) * p.convk;
    // The real model has four taps. Keep its three history samples in
    // registers across the batch; same ordered FMA chain and snapshot row.
    // Small batches and all other convolution sizes keep the old kernel.
    if (p.convk == 4 && p.rows > 8) {
        float h0 = hist[0], h1 = hist[1], h2 = hist[2];
        float w0 = wc[0], w1 = wc[1], w2 = wc[2], w3 = wc[3];
        for (uint row = 0; row < p.rows; row++) {
            float v = 0;
            v = fma(w0, h0, v);
            v = fma(w1, h1, v);
            v = fma(w2, h2, v);
            float raw = x[ulong(row) * p.convd + c];
            v = fma(w3, raw, v);
            y[ulong(row) * p.convd + c] = v * q36_sigmoid(v);
            h0 = h1; h1 = h2; h2 = raw;
            if (row == p.snap) {
                snap[ulong(c) * 3] = h0;
                snap[ulong(c) * 3 + 1] = h1;
                snap[ulong(c) * 3 + 2] = h2;
            }
        }
        hist[0] = h0; hist[1] = h1; hist[2] = h2;
        return;
    }
    for (uint row = 0; row < p.rows; row++) {
        float v = 0;
        for (uint k = 0; k + 1 < p.convk; k++) v = fma(wc[k], hist[k], v);
        float raw = x[ulong(row) * p.convd + c];
        v = fma(wc[p.convk - 1], raw, v);
        y[ulong(row) * p.convd + c] = v * q36_sigmoid(v);
        for (uint k = 0; k + 2 < p.convk; k++) hist[k] = hist[k + 1];
        hist[p.convk - 2] = raw;
        if (row == p.snap)
            for (uint k = 0; k + 1 < p.convk; k++) snap[ulong(c) * (p.convk - 1) + k] = hist[k];
    }
}

// L2-normalize the q and k heads (q also takes the 1/sqrt(kdim) scale) and
// copy v through, so the delta kernel reads one buffer.
kernel void q36_dn_l2(device const float *x [[buffer(0)]],
                      device float *y [[buffer(1)]],
                      constant q36_state_args &p [[buffer(2)]],
                      uint z [[threadgroup_position_in_grid]],
                      ushort lane [[thread_index_in_simdgroup]]) {
    if (z >= p.rows * 2 * p.kh) return;
    uint row = z / (2 * p.kh), head = z % (2 * p.kh);
    ulong base = ulong(row) * p.convd + head * p.kd;
    float sum = 0;
    for (uint k = lane; k < p.kd; k += 32) sum = fma(x[base + k], x[base + k], sum);
    sum = simd_sum(sum);
    float scale = (head < p.kh ? 1.0f / sqrt(float(p.kd)) : 1.0f) / sqrt(sum + 1e-6f);
    for (uint k = lane; k < p.kd; k += 32) y[base + k] = x[base + k] * scale;
    // every SIMD group of the row copies its share of v (one group alone took 20 us in decode)
    ulong vbase = ulong(row) * p.convd + 2 * p.kh * p.kd;
    for (uint v = head * 32 + lane; v < p.vh * p.vd; v += 2 * p.kh * 32) y[vbase + v] = x[vbase + v];
}

// Error-free float sums and products. The state is kept as a high and a low
// plane (about 48 bits of mantissa): over thousands of tokens the plain
// float recurrence drifts away from the reference. The low plane is state,
// not scratch.
inline float2 q36_two_sum(float a, float b) {
    float s = a + b, z = s - a;
    return float2(s, (a - (s - z)) + (b - z));
}
inline float2 q36_dd_add(float2 a, float2 b) {
    float2 s = q36_two_sum(a.x, b.x);
    return q36_two_sum(s.x, (a.y + b.y) + s.y);
}
inline float2 q36_dd_mul(float2 a, float b) {
    float p = a.x * b;
    return q36_two_sum(p, fma(a.x, b, -p) + a.y * b);
}

// The delta rule, one token after the other. A group owns 32 value columns of
// one head; each lane keeps its column of both planes (kdim <= 128) in private
// memory. In threadgroup memory the slice took the whole 32 KiB, so one group
// ran per GPU core and the 128 groups of a layer queued: 3.5x slower.
kernel void q36_dn_delta(device const float *qk [[buffer(0)]],
                         device const float *conv [[buffer(1)]],
                         device const float *g [[buffer(2)]],
                         device const float *beta [[buffer(3)]],
                         device float *state [[buffer(4)]],
                         device float *out [[buffer(5)]],
                         constant q36_state_args &p [[buffer(6)]],
                         device float *snap [[buffer(7)]],
                         uint3 group [[threadgroup_position_in_grid]],
                         ushort lane [[thread_index_in_simdgroup]]) {
    uint v = group.x * 32 + lane, h = group.y;
    if (h >= p.vh || v >= p.vd) return;
    thread float tile[128], low[128];           // the lane's column, both planes (see below)
    uint kh = h / (p.vh / p.kh);
    ulong cells = ulong(p.vh) * p.kd * p.vd;
    for (uint k = 0; k < p.kd; k++) {
        ulong off = (ulong(h) * p.kd + k) * p.vd + v;
        tile[k] = state[off];
        low[k] = state[cells + off];
    }
    for (uint row = 0; row < p.rows; row++) {
        ulong qbase = ulong(row) * p.convd + kh * p.kd, kbase = qbase + p.kh * p.kd;
        float decay = g[ulong(row) * p.vh + h];
        float2 u = 0;
        for (uint k = 0; k < p.kd; k++) {
            float2 st = q36_dd_mul(float2(tile[k], low[k]), decay);
            tile[k] = st.x;
            low[k] = st.y;
            u = q36_dd_add(u, q36_dd_mul(st, qk[kbase + k]));
        }
        float vin = conv[ulong(row) * p.convd + 2 * p.kh * p.kd + h * p.vd + v];
        float2 delta = q36_dd_mul(q36_dd_add(float2(vin, 0), -u), beta[ulong(row) * p.vh + h]);
        float y = 0;
        for (uint k = 0; k < p.kd; k++) {
            float2 st = q36_dd_add(float2(tile[k], low[k]), q36_dd_mul(delta, qk[kbase + k]));
            tile[k] = st.x;
            low[k] = st.y;
            y = fma(qk[qbase + k], st.x, y);
            y = fma(qk[qbase + k], st.y, y);
        }
        out[(ulong(row) * p.vh + h) * p.vd + v] = y;
        if (row == p.snap)
            for (uint k = 0; k < p.kd; k++) {
                ulong off = (ulong(h) * p.kd + k) * p.vd + v;
                snap[off] = tile[k];
                snap[cells + off] = low[k];
            }
    }
    for (uint k = 0; k < p.kd; k++) {
        ulong off = (ulong(h) * p.kd + k) * p.vd + v;
        state[off] = tile[k];
        state[cells + off] = low[k];
    }
}

// A single token does not need to keep a whole column live between tokens.
// Read the old state twice and repeat its decay in the second pass: exactly
// the same arithmetic, without the two 128-element private arrays and their
// register pressure. The first pass never changes state, so the repeated
// decay has identical operands. Each lane still owns one value column.
kernel void q36_dn_delta_row(device const float *qk [[buffer(0)]],
                             device const float *conv [[buffer(1)]],
                             device const float *g [[buffer(2)]],
                             device const float *beta [[buffer(3)]],
                             device float *state [[buffer(4)]],
                             device float *out [[buffer(5)]],
                             constant q36_state_args &p [[buffer(6)]],
                             device float *snap [[buffer(7)]],
                             uint3 group [[threadgroup_position_in_grid]],
                             ushort lane [[thread_index_in_simdgroup]]) {
    uint v = group.x * 32 + lane, h = group.y;
    if (h >= p.vh || v >= p.vd) return;
    uint kh = h / (p.vh / p.kh);
    ulong cells = ulong(p.vh) * p.kd * p.vd, qbase = kh * p.kd, kbase = qbase + p.kh * p.kd;
    float decay = g[h];
    float2 u = 0;
    for (uint k = 0; k < p.kd; k++) {
        ulong off = (ulong(h) * p.kd + k) * p.vd + v;
        float2 st = q36_dd_mul(float2(state[off], state[cells + off]), decay);
        u = q36_dd_add(u, q36_dd_mul(st, qk[kbase + k]));
    }
    float vin = conv[2 * p.kh * p.kd + h * p.vd + v];
    float2 delta = q36_dd_mul(q36_dd_add(float2(vin, 0), -u), beta[h]);
    float y = 0;
    for (uint k = 0; k < p.kd; k++) {
        ulong off = (ulong(h) * p.kd + k) * p.vd + v;
        float2 decayed = q36_dd_mul(float2(state[off], state[cells + off]), decay);
        float2 st = q36_dd_add(decayed, q36_dd_mul(delta, qk[kbase + k]));
        state[off] = st.x;
        state[cells + off] = st.y;
        y = fma(qk[qbase + k], st.x, y);
        y = fma(qk[qbase + k], st.y, y);
        if (p.snap == 0) { snap[off] = st.x; snap[cells + off] = st.y; }
    }
    out[ulong(h) * p.vd + v] = y;
}

// The delta rule in fast mode, as ds4's gdn_scan_r4: a SIMD
// group owns 4 value columns of one head, each lane kd/32 keys of them, and the
// two sums of each token are SIMD sums in plain float, not the exact kernel's
// compensated serial sums. The state goes back as the high plane with a zero
// low plane, which the exact kernel reads as the same values.
kernel void q36_dn_scan(device const float *qk [[buffer(0)]],
                        device const float *conv [[buffer(1)]],
                        device const float *g [[buffer(2)]],
                        device const float *beta [[buffer(3)]],
                        device float *state [[buffer(4)]],
                        device float *out [[buffer(5)]],
                        constant q36_state_args &p [[buffer(6)]],
                        device float *snap [[buffer(7)]],
                        uint3 group [[threadgroup_position_in_grid]],
                        ushort sg [[simdgroup_index_in_threadgroup]],
                        ushort lane [[thread_index_in_simdgroup]]) {
    uint v0 = (group.x * 4 + sg) * 4, h = group.y;
    if (h >= p.vh || v0 >= p.vd) return;
    uint n = p.kd / 32, k0 = lane * n, kh = h / (p.vh / p.kh);
    ulong cells = ulong(p.vh) * p.kd * p.vd;
    float4 s[4];                                // s[i]: key k0 + i, columns v0..v0+3
    for (uint i = 0; i < n; i++) {
        ulong off = (ulong(h) * p.kd + k0 + i) * p.vd + v0;
        s[i] = *(device const float4 *)(state + off);
        if (p.low) s[i] += *(device const float4 *)(state + cells + off);
    }
    for (uint row = 0; row < p.rows; row++) {
        ulong qbase = ulong(row) * p.convd + kh * p.kd + k0, kbase = qbase + p.kh * p.kd;
        float decay = g[ulong(row) * p.vh + h], b = beta[ulong(row) * p.vh + h];
        float kk[4], qq[4];
        for (uint i = 0; i < n; i++) {
            kk[i] = qk[kbase + i];
            qq[i] = qk[qbase + i];
        }
        float4 u = 0;
        for (uint i = 0; i < n; i++) {
            s[i] *= decay;
            u = fma(s[i], float4(kk[i]), u);
        }
        u = simd_sum(u);
        float4 vin = *(device const float4 *)(conv + ulong(row) * p.convd + 2 * p.kh * p.kd + h * p.vd + v0);
        float4 delta = (vin - u) * b, y = 0;
        for (uint i = 0; i < n; i++) {
            s[i] = fma(float4(kk[i]), delta, s[i]);
            y = fma(s[i], float4(qq[i]), y);
        }
        y = simd_sum(y);
        if (lane == 0) *(device float4 *)(out + (ulong(row) * p.vh + h) * p.vd + v0) = y;
        if (row == p.snap)
            for (uint i = 0; i < n; i++) {
                ulong off = (ulong(h) * p.kd + k0 + i) * p.vd + v0;
                *(device float4 *)(snap + off) = s[i];
                if (p.low) *(device float4 *)(snap + cells + off) = 0;
            }
    }
    for (uint i = 0; i < n; i++) {
        ulong off = (ulong(h) * p.kd + k0 + i) * p.vd + v0;
        *(device float4 *)(state + off) = s[i];
        if (p.low) *(device float4 *)(state + cells + off) = 0;
    }
}

// Chunked DeltaNet of fast prompt batches (k and v heads of 128), in float. For a chunk of
// n <= 32 rows of a value head, with the decays g as direct short products (never ratios of
// cumulative products or logs, so a zero decay resets the history without 0/0 or log(0)):
//   C[t][j] = g[t] g[t-1] .. g[j+1] (j <= t),  D[t] = g[t] .. g[0],  E[j] = C[n-1][j],
//   T = I + (beta[t] C[t][j] k_t.k_j below the diagonal),  B = (C[t][j] q_t.k_j, j <= t),
//   U = T^-1 (beta (V - (D K) S0)) = U0 - W S0,  U0 = A (beta V),  W = A (beta D K),  A = T^-1,
//   out = (D Q) S0 + B U,  S = D[n-1] S0 + (E K)^T U.
// Nothing in C, D, E, T, B, A, U0 or W depends on the state: three dispatches compute them for
// up to 8 chunks at once, a threadgroup per chunk and head (q36_dn_chunk_gram: D, E, T, B;
// q36_dn_chunk_inv: A over T; q36_dn_chunk_uw: U0, W). Then, chunk after chunk, one dispatch uses
// the state (q36_dn_chunk_step), reading and writing it once. Chunks of at most 8 rows keep the
// scan. Scratch per prepared chunk and head, in floats: B [32][32], D [32], E [32],
// U0 [32][128], W [32][128], T (then A) [32][32].
constant uint Q36_DNC_B = 0, Q36_DNC_D = 1024, Q36_DNC_E = 1056, Q36_DNC_U = 1088, Q36_DNC_W = 5184;
constant uint Q36_DNC_T = 9280, Q36_DNC_STRIDE = 10304;
struct q36_dn_chunk_args { q36_state_args s; uint first[8], n[8]; uint chunk, capture; };

// A state with both planes enters the chunks folded: high += low, low = 0 (the chunks' update
// leaves the low plane zero anyway).
kernel void q36_dn_fold(device float *state [[buffer(0)]],
                        constant uint &cells [[buffer(1)]],
                        uint i [[thread_position_in_grid]]) {
    if (i >= cells) return;
    state[i] += state[cells + i];
    state[cells + i] = 0.0f;
}

// D, E, T, B. K K^T and Q K^T through threadgroup memory, 32 dimensions at a time, then the
// decay products, each from the diagonal outward. Grid (chunks, value heads), 128 threads,
// 2112 floats of threadgroup memory.
kernel void q36_dn_chunk_gram(device const float *qk [[buffer(0)]],
                              device const float *decay [[buffer(1)]],
                              device const float *beta [[buffer(2)]],
                              device float *scratch [[buffer(3)]],
                              constant q36_dn_chunk_args &a [[buffer(4)]],
                              threadgroup float *shared [[threadgroup(0)]],
                              uint2 tg [[threadgroup_position_in_grid]],
                              ushort tid [[thread_index_in_threadgroup]],
                              ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint c = tg.x, h = tg.y, first = a.first[c], n = a.n[c];
    const uint kh = h / (a.s.vh / a.s.kh), convd = a.s.convd, vh = a.s.vh, nkh = a.s.kh;
    device float *w = scratch + (ulong(c) * vh + h) * Q36_DNC_STRIDE;
    threadgroup float *g = shared, *bt = shared + 32, *ks = shared + 64, *qs = ks + 1024;
    if (tid < 32) {
        g[tid] = tid < n ? decay[ulong(first + tid) * vh + h] : 1.0f;
        bt[tid] = tid < n ? beta[ulong(first + tid) * vh + h] : 0.0f;
    }
    simdgroup_float8x8 akk[4], aqk[4];
    for (uint i = 0; i < 4; i++) {
        akk[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
        aqk[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }
    for (uint k0 = 0; k0 < 128; k0 += 32) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = tid; e < 1024; e += 128) {
            uint t = e / 32, d = e % 32;
            float kv = 0, qv = 0;
            if (t < n) {
                ulong row = ulong(first + t) * convd;
                kv = qk[row + (nkh + kh) * 128 + k0 + d];
                qv = qk[row + kh * 128 + k0 + d];
            }
            ks[e] = kv;
            qs[e] = qv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kb = 0; kb < 4; kb++) {
            simdgroup_float8x8 ak, aq;
            simdgroup_load(ak, ks + 8 * sg * 32 + 8 * kb, 32);
            simdgroup_load(aq, qs + 8 * sg * 32 + 8 * kb, 32);
            for (uint cb = 0; cb < 4; cb++) {
                simdgroup_float8x8 bk;           // (d, j) = k_j[d]
                simdgroup_load(bk, ks + 8 * cb * 32 + 8 * kb, 32, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(akk[cb], ak, bk, akk[cb]);
                simdgroup_multiply_accumulate(aqk[cb], aq, bk, aqk[cb]);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint cb = 0; cb < 4; cb++) {            // the staging is free: K K^T, Q K^T there
        simdgroup_store(akk[cb], ks + 8 * sg * 32 + 8 * cb, 32);
        simdgroup_store(aqk[cb], qs + 8 * sg * 32 + 8 * cb, 32);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32) {                              // D[t] and E[t]
        const uint t = tid;
        float d = 0, e = 0;
        if (t < n) {
            d = 1;
            for (int j = int(t); j >= 0; j--) d *= g[j];
            e = 1;
            for (int j = int(n) - 1; j > int(t); j--) e *= g[j];
        }
        w[Q36_DNC_D + t] = d;
        w[Q36_DNC_E + t] = e;
    }
    // T (unit lower triangular, identity past n) and B (causal): C[t][j] = g[t] g[t-1] .. g[j+1]
    for (uint e = tid; e < 1024; e += 128) {
        uint t = e / 32, j = e % 32;
        float tz = t == j ? 1.0f : 0.0f, bz = 0;
        if (t < n && j <= t) {
            float cf = 1;
            for (uint i = t; i > j; i--) cf *= g[i];
            if (j < t) tz = (bt[t] * cf) * ks[e];
            bz = cf * qs[e];
        }
        w[Q36_DNC_T + e] = tz;
        w[Q36_DNC_B + e] = bz;
    }
}

// A = T^-1 in place of T, a column per lane: x_j = 1, x_i = -sum_{k = j}^{i-1} T[i][k] x_k. A
// threadgroup of one SIMD group and 2048 floats per chunk and head, so that many run at once
// and hide each other's chain of 500 dependent steps. Grid (chunks, value heads), 32 threads.
kernel void q36_dn_chunk_inv(device float *scratch [[buffer(0)]],
                             constant q36_dn_chunk_args &a [[buffer(1)]],
                             threadgroup float *shared [[threadgroup(0)]],
                             uint2 tg [[threadgroup_position_in_grid]],
                             ushort lane [[thread_index_in_simdgroup]]) {
    device float *w = scratch + (ulong(tg.x) * a.s.vh + tg.y) * Q36_DNC_STRIDE + Q36_DNC_T;
    threadgroup float *tm = shared, *am = shared + 1024;
    for (uint e = lane; e < 1024; e += 32) tm[e] = w[e];
    simdgroup_barrier(mem_flags::mem_threadgroup);
    const uint j = lane;
    for (uint i = 0; i < 32; i++) {
        float x = i == j ? 1.0f : 0.0f;
        if (i > j)
            for (uint k = j; k < i; k++) x = fma(-tm[i * 32 + k], am[k * 32 + j], x);
        am[i * 32 + j] = x;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = lane; e < 1024; e += 32) w[e] = am[e];
}

// U0 = A (beta V), then W = A (beta D K): SIMD group sg takes columns 32 sg .. 32 sg + 31; A
// from the scratch, V and K straight into the matrix operand. Grid (chunks, value heads, 2),
// 128 threads.
kernel void q36_dn_chunk_uw(device const float *qk [[buffer(0)]],
                            device const float *conv [[buffer(1)]],
                            device const float *beta [[buffer(2)]],
                            device float *scratch [[buffer(3)]],
                            constant q36_dn_chunk_args &a [[buffer(4)]],
                            uint3 tg [[threadgroup_position_in_grid]],
                            ushort sg [[simdgroup_index_in_threadgroup]],
                            ushort lane [[thread_index_in_simdgroup]]) {
    const uint c = tg.x, h = tg.y, pass = tg.z, first = a.first[c], n = a.n[c];
    const uint kh = h / (a.s.vh / a.s.kh), convd = a.s.convd, vh = a.s.vh, nkh = a.s.kh;
    device float *w = scratch + (ulong(c) * vh + h) * Q36_DNC_STRIDE;
    const ushort2 pl = q36_lane_place(lane);
    const ushort sn = pl.x, sm = pl.y;
    float2 acc[16];
    for (uint i = 0; i < 16; i++) acc[i] = float2(0.f);
    for (uint kb = 0; kb < 4; kb++) {
        simdgroup_float8x8 ma[4];
        for (uint rb = 0; rb < 4; rb++) simdgroup_load(ma[rb], w + Q36_DNC_T + 8 * rb * 32 + 8 * kb, 32);
        const uint t = 8 * kb + sm;
        float2 bf[4];
        float f = 0;
        ulong row = 0;
        if (t < n) {
            row = ulong(first + t) * convd;
            float b = beta[ulong(first + t) * vh + h];
            f = pass == 0 ? b : b * w[Q36_DNC_D + t];
        }
        for (uint cb = 0; cb < 4; cb++) {
            const uint col = 32 * sg + 8 * cb + sn;
            bf[cb] = t < n ? (pass == 0 ? *(device const float2 *)(conv + row + 2 * nkh * 128 + h * 128 + col)
                                        : *(device const float2 *)(qk + row + (nkh + kh) * 128 + col)) * f
                           : float2(0.f);
        }
        for (uint rb = 0; rb < 4; rb++)
            for (uint cb = 0; cb < 4; cb++) {
                simdgroup_float8x8 bm, cmx;
                reinterpret_cast<thread float2 &>(bm.thread_elements()) = bf[cb];
                reinterpret_cast<thread float2 &>(cmx.thread_elements()) = acc[4 * rb + cb];
                simdgroup_multiply_accumulate(cmx, ma[rb], bm, cmx);
                acc[4 * rb + cb] = reinterpret_cast<thread float2 &>(cmx.thread_elements());
            }
    }
    device float *dst = w + (pass == 0 ? Q36_DNC_U : Q36_DNC_W);
    for (uint rb = 0; rb < 4; rb++)
        for (uint cb = 0; cb < 4; cb++)
            *(device float2 *)(dst + (8 * rb + sm) * 128 + 32 * sg + 8 * cb + sn) = acc[4 * rb + cb];
}

// One chunk, everything that uses the state, for 32 of its value columns: U = U0 - W S0,
// out = (D Q) S0 + B U and S = D[n-1] S0 + (E K)^T U, so the state slice is read once and
// written once and U never leaves the chip. A threadgroup per value head and 32 columns; SIMD
// group sg takes rows 8 sg .. of U and out, and state rows 32 sg .. . Grid (4, value heads),
// 128 threads, 1024 floats of threadgroup memory (U).
kernel void q36_dn_chunk_step(device const float *qk [[buffer(0)]],
                              device float *state [[buffer(1)]],
                              device float *out [[buffer(2)]],
                              device float *snap [[buffer(3)]],
                              device const float *scratch [[buffer(4)]],
                              constant q36_dn_chunk_args &a [[buffer(5)]],
                              threadgroup float *um [[threadgroup(0)]],
                              uint2 tg [[threadgroup_position_in_grid]],
                              ushort sg [[simdgroup_index_in_threadgroup]],
                              ushort lane [[thread_index_in_simdgroup]]) {
    const uint c = a.chunk, h = tg.y, c0 = tg.x * 32, first = a.first[c], n = a.n[c];
    const uint kh = h / (a.s.vh / a.s.kh), convd = a.s.convd, vh = a.s.vh, nkh = a.s.kh;
    device const float *w = scratch + (ulong(c) * vh + h) * Q36_DNC_STRIDE;
    device float *s0 = state + ulong(h) * 128 * 128;
    const ulong cells = ulong(vh) * 128 * 128;
    const ushort2 pl = q36_lane_place(lane);
    const ushort sn = pl.x, sm = pl.y;
    // U rows 8 sg .. 8 sg + 7, all 32 columns: U0 - W S0
    float2 acc[16];
    for (uint i = 0; i < 4; i++) acc[i] = float2(0.f);
    for (uint k0 = 0; k0 < 128; k0 += 8) {
        simdgroup_float8x8 ma, mb[4];
        simdgroup_load(ma, w + Q36_DNC_W + 8 * sg * 128 + k0, 128);
        for (uint cb = 0; cb < 4; cb++) simdgroup_load(mb[cb], s0 + k0 * 128 + c0 + 8 * cb, 128);
        for (uint cb = 0; cb < 4; cb++) {
            simdgroup_float8x8 cmx;
            reinterpret_cast<thread float2 &>(cmx.thread_elements()) = acc[cb];
            simdgroup_multiply_accumulate(cmx, ma, mb[cb], cmx);
            acc[cb] = reinterpret_cast<thread float2 &>(cmx.thread_elements());
        }
    }
    for (uint cb = 0; cb < 4; cb++) {
        const uint t = 8 * sg + sm;
        float2 u = *(device const float2 *)(w + Q36_DNC_U + t * 128 + c0 + 8 * cb + sn) - acc[cb];
        *(threadgroup float2 *)(um + t * 32 + 8 * cb + sn) = u;
        acc[cb] = float2(0.f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // out rows 8 sg ..: (D Q) S0 + B U
    for (uint k0 = 0; k0 < 128; k0 += 8) {
        const uint t = 8 * sg + sm;
        float2 af = t < n ? *(device const float2 *)(qk + ulong(first + t) * convd + kh * 128 + k0 + sn) * w[Q36_DNC_D + t]
                          : float2(0.f);
        simdgroup_float8x8 ma, mb[4];
        reinterpret_cast<thread float2 &>(ma.thread_elements()) = af;
        for (uint cb = 0; cb < 4; cb++) simdgroup_load(mb[cb], s0 + k0 * 128 + c0 + 8 * cb, 128);
        for (uint cb = 0; cb < 4; cb++) {
            simdgroup_float8x8 cmx;
            reinterpret_cast<thread float2 &>(cmx.thread_elements()) = acc[cb];
            simdgroup_multiply_accumulate(cmx, ma, mb[cb], cmx);
            acc[cb] = reinterpret_cast<thread float2 &>(cmx.thread_elements());
        }
    }
    for (uint kb = 0; kb < 4; kb++) {
        simdgroup_float8x8 ma, mb[4];
        simdgroup_load(ma, w + Q36_DNC_B + 8 * sg * 32 + 8 * kb, 32);
        for (uint cb = 0; cb < 4; cb++) simdgroup_load(mb[cb], um + 8 * kb * 32 + 8 * cb, 32);
        for (uint cb = 0; cb < 4; cb++) {
            simdgroup_float8x8 cmx;
            reinterpret_cast<thread float2 &>(cmx.thread_elements()) = acc[cb];
            simdgroup_multiply_accumulate(cmx, ma, mb[cb], cmx);
            acc[cb] = reinterpret_cast<thread float2 &>(cmx.thread_elements());
        }
    }
    {
        const uint t = 8 * sg + sm;
        if (t < n) {
            device float *o = out + (ulong(first + t) * vh + h) * 128 + c0 + sn;
            for (uint cb = 0; cb < 4; cb++) *(device float2 *)(o + 8 * cb) = acc[cb];
        }
    }
    // state rows 32 sg ..: D[n-1] S0 + (E K)^T U, written once every SIMD group is done reading S0
    threadgroup_barrier(mem_flags::mem_device);
    for (uint i = 0; i < 16; i++) acc[i] = float2(0.f);
    for (uint kb = 0; kb < 4; kb++) {
        simdgroup_float8x8 mb[4];
        for (uint cb = 0; cb < 4; cb++) simdgroup_load(mb[cb], um + 8 * kb * 32 + 8 * cb, 32);
        const uint t = 8 * kb + sn;              // the lane's two columns of (E K)^T: t, t + 1
        float e0 = t < n ? w[Q36_DNC_E + t] : 0.0f, e1 = t + 1 < n ? w[Q36_DNC_E + t + 1] : 0.0f;
        device const float *k0p = qk + ulong(first + t) * convd + (nkh + kh) * 128;
        for (uint rb = 0; rb < 4; rb++) {
            const uint k = 32 * sg + 8 * rb + sm;
            float2 ak = float2(e0 != 0.0f ? e0 * k0p[k] : 0.0f, e1 != 0.0f ? e1 * k0p[convd + k] : 0.0f);
            simdgroup_float8x8 ma;
            reinterpret_cast<thread float2 &>(ma.thread_elements()) = ak;
            for (uint cb = 0; cb < 4; cb++) {
                simdgroup_float8x8 cmx;
                reinterpret_cast<thread float2 &>(cmx.thread_elements()) = acc[4 * rb + cb];
                simdgroup_multiply_accumulate(cmx, ma, mb[cb], cmx);
                acc[4 * rb + cb] = reinterpret_cast<thread float2 &>(cmx.thread_elements());
            }
        }
    }
    const float d = w[Q36_DNC_D + n - 1];
    for (uint rb = 0; rb < 4; rb++)
        for (uint cb = 0; cb < 4; cb++) {
            ulong off = (ulong(h) * 128 + 32 * sg + 8 * rb + sm) * 128 + c0 + 8 * cb + sn;
            float2 v = acc[4 * rb + cb];
            float2 old = *(device const float2 *)(state + off);
            float2 updated = d == 0.0f ? v : fma(float2(d), old, v);
            *(device float2 *)(state + off) = updated;
            if (a.s.low) *(device float2 *)(state + cells + off) = float2(0.f);
            if (a.capture) {
                *(device float2 *)(snap + off) = updated;
                if (a.s.low) *(device float2 *)(snap + cells + off) = float2(0.f);
            }
        }
}

// ---------------------------------------------------------------------------
// Gated attention layers.
// ---------------------------------------------------------------------------

// The q projection interleaves query and gate per head (qd = 2*hd when the
// output gate is on). Split them and copy k beside.
kernel void q36_attn_split(device const float *q [[buffer(0)]],
                           device float *query [[buffer(1)]],
                           device float *gate [[buffer(2)]],
                           device const float *k [[buffer(3)]],
                           device float *key [[buffer(4)]],
                           constant q36_state_args &p [[buffer(5)]],
                           uint i [[thread_position_in_grid]]) {
    if (i >= p.rows * p.qh * p.hd) return;
    uint d = i % p.hd, h = (i / p.hd) % p.qh, row = i / (p.hd * p.qh);
    ulong off = (ulong(row) * p.qh + h) * p.qd + d;
    query[i] = q[off];
    gate[i] = p.qd > p.hd ? q[off + p.hd] : 0.0f;
    if (i < p.rows * p.kv * p.hd) key[i] = k[i];
}

// Rotate the first `rotary` dims of each head in place, using a cos/sin table
// the host computes in double precision (float angles drift at long positions).
kernel void q36_attn_rope(device float *x [[buffer(0)]],
                          device const float2 *rot [[buffer(1)]],
                          constant q36_state_args &p [[buffer(2)]],
                          uint i [[thread_position_in_grid]]) {
    uint half_dim = p.rotary / 2;
    if (i >= p.rows * p.heads * half_dim) return;
    uint d = i % half_dim, h = (i / half_dim) % p.heads, row = i / (half_dim * p.heads);
    ulong off = (ulong(row) * p.heads + h) * p.hd + d;
    float2 r = rot[ulong(row) * half_dim + d];
    float a = x[off], b = x[off + half_dim];
    x[off] = a * r.x - b * r.y;
    x[off + half_dim] = a * r.y + b * r.x;
}

// Append the rows' keys and values at positions pos.. of the cache.
// KV is float, or half in fast mode (q36_kv_half): the fast attention kernels
// read either; their sums stay in float.
template <typename KV>
kernel void q36_attn_kv(device const float *k [[buffer(0)]],
                        device const float *v [[buffer(1)]],
                        device KV *kc [[buffer(2)]],
                        device KV *vc [[buffer(3)]],
                        constant q36_state_args &p [[buffer(4)]],
                        uint i [[thread_position_in_grid]]) {
    if (i >= p.rows * p.kv * p.hd) return;
    uint d = i % p.hd, h = (i / p.hd) % p.kv, row = i / (p.hd * p.kv);
    ulong dest = (ulong(h) * p.cap + p.pos + row) * p.hd + d;
    kc[dest] = KV(k[i]);
    vc[dest] = KV(v[i]);
}
template [[host_name("q36_attn_kv")]] kernel void q36_attn_kv<float>(device const float *, device const float *,
        device float *, device float *, constant q36_state_args &, uint);
template [[host_name("q36_attn_kv_half")]] kernel void q36_attn_kv<half>(device const float *, device const float *,
        device half *, device half *, constant q36_state_args &, uint);

// Kahan accumulation whose residual is rescaled with the running maximum.
// Small attention weights must survive thousands of later terms.
inline void q36_attn_sum(thread float &sum, thread float &error, float old, float term) {
    float scaled = sum * old, product_error = fma(sum, old, -scaled);
    float adjusted = term - fma(error, old, -product_error), next = scaled + adjusted;
    error = (next - scaled) - adjusted;
    sum = next;
}

// Causal attention of each row over the cache, with an online softmax: no
// score matrix, bounded accumulators, head_dim <= 256.
kernel void q36_attn(device const float *q [[buffer(0)]],
                     device const float *kc [[buffer(1)]],
                     device const float *vc [[buffer(2)]],
                     device float *out [[buffer(3)]],
                     constant q36_state_args &p [[buffer(4)]],
                     uint z [[threadgroup_position_in_grid]],
                     ushort lane [[thread_index_in_simdgroup]]) {
    if (z >= p.rows * p.qh) return;
    uint row = z / p.qh, h = z % p.qh, kh = h / (p.qh / p.kv);
    float qv[8], acc[8], acc_error[8];
    for (uint j = 0; j < 8; j++) {
        uint d = lane + j * 32;
        qv[j] = d < p.hd ? q[(ulong(row) * p.qh + h) * p.hd + d] : 0.0f;
        acc[j] = acc_error[j] = 0;
    }
    float maximum = -INFINITY, denominator = 0, denominator_error = 0;
    float scale = 1.0f / sqrt(float(p.hd));
    for (uint t = 0; t <= p.pos + row; t++) {
        ulong off = (ulong(kh) * p.cap + t) * p.hd;
        float score = 0;
        for (uint j = 0; j < 8; j++) {
            uint d = lane + j * 32;
            if (d < p.hd) score = fma(qv[j], kc[off + d], score);
        }
        score = simd_sum(score) * scale;
        float next = max(maximum, score), old = exp(maximum - next), weight = exp(score - next);
        q36_attn_sum(denominator, denominator_error, old, weight);
        for (uint j = 0; j < 8; j++) {
            uint d = lane + j * 32;
            if (d < p.hd) q36_attn_sum(acc[j], acc_error[j], old, weight * vc[off + d]);
        }
        maximum = next;
    }
    for (uint j = 0; j < 8; j++) {
        uint d = lane + j * 32;
        if (d < p.hd) out[(ulong(row) * p.qh + h) * p.hd + d] = acc[j] / denominator;
    }
}

// Decode (one row) in three steps with q36_attn's arithmetic, so the output is
// the same to the bit. In q36_attn one SIMD group per head walks the cache in
// order, and a long context leaves the GPU almost empty while it does. Here
// the heads that share a KV head (at most 8) sit in one threadgroup, a SIMD
// group each, so they run on one GPU core and read the cache once between
// them, not once per head.
// q36_attn_scores: q.k of every position in parallel, 32 positions per SIMD
// group, each score exactly as q36_attn computes it (the lane's columns, fma
// over j, simd_sum, scale). Grid: (positions / 32, KV heads) x (32 * heads
// per KV head).
kernel void q36_attn_scores(device const float *q [[buffer(0)]],
                            device const float *kc [[buffer(1)]],
                            device float *scores [[buffer(2)]],
                            constant q36_state_args &p [[buffer(3)]],
                            uint2 tg [[threadgroup_position_in_grid]],
                            ushort sg [[simdgroup_index_in_threadgroup]],
                            ushort lane [[thread_index_in_simdgroup]]) {
    uint kh = tg.y, group = p.qh / p.kv, h = kh * group + sg, n = p.pos + 1;
    if (kh >= p.kv || sg >= group) return;
    float qv[8];
    for (uint j = 0; j < 8; j++) {
        uint d = lane + j * 32;
        qv[j] = d < p.hd ? q[ulong(h) * p.hd + d] : 0.0f;
    }
    float scale = 1.0f / sqrt(float(p.hd));
    uint t = tg.x * 32;
    const uint tend = min(tg.x * 32 + 32, n);
    // Heads of 256: four positions at a time, their loads first and their chains side by side, each
    // position's chain and sum the same operations as the loop below (the same bits). One at a time the
    // chain and the sum of a position waited for each other: 0.80 -> 0.31 ms at 8192 positions on the M1.
    if (p.hd == 256)
        for (; t + 4 <= tend; t += 4) {
            float part[4];
            for (uint u = 0; u < 4; u++) {
                ulong off = (ulong(kh) * p.cap + t + u) * p.hd;
                float k8[8];
                for (uint j = 0; j < 8; j++) k8[j] = kc[off + lane + j * 32];
                float score = 0;
                for (uint j = 0; j < 8; j++) score = fma(qv[j], k8[j], score);
                part[u] = score;
            }
            for (uint u = 0; u < 4; u++) {
                float score = simd_sum(part[u]) * scale;
                if (lane == 0) scores[ulong(h) * p.cap + t + u] = score;
            }
        }
    for (; t < tend; t++) {
        ulong off = (ulong(kh) * p.cap + t) * p.hd;
        float score = 0;
        for (uint j = 0; j < 8; j++) {
            uint d = lane + j * 32;
            if (d < p.hd) score = fma(qv[j], kc[off + d], score);
        }
        score = simd_sum(score) * scale;
        if (lane == 0) scores[ulong(h) * p.cap + t] = score;
    }
}

// q36_attn_weights: for every position, q36_attn's factor that rescales what
// came before and the weight of that position. Both are exp() of differences
// with the running maximum, and a maximum does not depend on the order it is
// taken in: the positions are cut in contiguous runs, one per thread, each
// run starts from the maximum of the runs before it, and the values are the
// same bits as in q36_attn. One threadgroup per head, Q36_ATTN_THREADS threads.
// Scratch: scores, then factors, then weights, each [heads][cap].
#define Q36_ATTN_THREADS 256
kernel void q36_attn_weights(device float *scores [[buffer(0)]],
                             constant q36_state_args &p [[buffer(1)]],
                             uint h [[threadgroup_position_in_grid]],
                             ushort tid [[thread_index_in_threadgroup]],
                             ushort sg [[simdgroup_index_in_threadgroup]],
                             ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float tops[Q36_ATTN_THREADS / 32];
    if (h >= p.qh) return;
    uint n = p.pos + 1, run = (n + Q36_ATTN_THREADS - 1) / Q36_ATTN_THREADS;
    uint t0 = min(n, tid * run), t1 = min(n, t0 + run);
    ulong plane = ulong(p.qh) * p.cap;
    device const float *s = scores + ulong(h) * p.cap;
    device float *factor = scores + plane + ulong(h) * p.cap, *weight = factor + plane;
    float top = -INFINITY;
    for (uint t = t0; t < t1; t++) top = max(top, s[t]);
    float before = top;                         // maximum of the runs before this one
    for (ushort k = 1; k < 32; k <<= 1) {
        float other = simd_shuffle_up(before, k);
        if (lane >= k) before = max(before, other);
    }
    if (lane == 31) tops[sg] = before;
    before = simd_shuffle_up(before, 1);
    if (lane == 0) before = -INFINITY;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (ushort k = 0; k < sg; k++) before = max(before, tops[k]);
    float maximum = before;
    for (uint t = t0; t < t1; t++) {
        float score = s[t];
        float next = max(maximum, score);
        factor[t] = exp(maximum - next);
        weight[t] = exp(score - next);
        maximum = next;
    }
}

// q36_attn_values: the sums of q36_attn, one column per thread, the
// denominator alongside as q36_attn keeps it. A threadgroup holds two heads (a
// SIMD group each) for the same 32 columns. It held all the heads of a KV head
// (eight), to read each value of the cache once between them, but groups of
// 64 threads spread over the GPU's cores better: at 8192 positions on the M1
// 1.70 -> 1.51 ms for scores, weights and values, the same bits.
// Grid: (head_dim / 32, heads / 2) x 64.
kernel void q36_attn_values(device const float *scores [[buffer(0)]],
                            device const float *vc [[buffer(1)]],
                            device float *out [[buffer(2)]],
                            constant q36_state_args &p [[buffer(3)]],
                            uint2 tg [[threadgroup_position_in_grid]],
                            ushort sg [[simdgroup_index_in_threadgroup]],
                            ushort lane [[thread_index_in_simdgroup]]) {
    uint h = tg.y * 2 + sg, kh = h / (p.qh / p.kv), d = tg.x * 32 + lane, n = p.pos + 1;
    if (h >= p.qh || d >= p.hd) return;
    ulong plane = ulong(p.qh) * p.cap;
    device const float *factor = scores + plane + ulong(h) * p.cap, *weight = factor + plane;
    device const float *v = vc + ulong(kh) * p.cap * p.hd + d;
    float acc = 0, acc_error = 0, denominator = 0, denominator_error = 0;
    uint t = 0;
    for (; t + 4 <= n; t += 4) {                // the loads of four positions first
        float old[4], w[4], value[4];
        for (uint i = 0; i < 4; i++) {
            old[i] = factor[t + i];
            w[i] = weight[t + i];
            value[i] = v[ulong(t + i) * p.hd];
        }
        for (uint i = 0; i < 4; i++) {
            q36_attn_sum(denominator, denominator_error, old[i], w[i]);
            q36_attn_sum(acc, acc_error, old[i], w[i] * value[i]);
        }
    }
    for (; t < n; t++) {
        q36_attn_sum(denominator, denominator_error, factor[t], weight[t]);
        q36_attn_sum(acc, acc_error, factor[t], weight[t] * v[ulong(t) * p.hd]);
    }
    out[ulong(h) * p.hd + d] = acc / denominator;
}

// ---------------------------------------------------------------------------
// MoE: router, shared expert, routed experts.
// ---------------------------------------------------------------------------

// Router logits. Deliberately serial and uncontracted: this reproduces the
// rounding of the CPU build (products rounded before ordered additions, fma
// only in the tail). A SIMD reduction moves scores enough to swap experts.
// One output per thread waits for its loads at every block: the weights of the
// next 64 inputs are loaded while the current ones are added, in the same
// order (the same bits).
inline float q36_router_row(device const char *qr, device const float *xr, uint in) {
#pragma clang fp contract(off)
    uint rounded = in & ~7u, i = 0;
    float acc = 0;
    if (rounded >= 64) {
        char4 c[16], next[16];
        for (uint k = 0; k < 16; k++) c[k] = char4(*(device const packed_char4 *)(qr + 4 * k));
        for (; i + 64 <= rounded; i += 64) {
            bool more = i + 128 <= rounded;
            if (more)
                for (uint k = 0; k < 16; k++) next[k] = char4(*(device const packed_char4 *)(qr + i + 64 + 4 * k));
            for (uint k = 0; k < 16; k++) acc = q36_ordered4(acc, q36_load4(xr + i + 4 * k), float4(c[k]));
            if (more)
                for (uint k = 0; k < 16; k++) c[k] = next[k];
        }
    }
    for (; i + 16 <= rounded; i += 16)
        for (uint k = 0; k < 4; k++)
            acc = q36_ordered4(acc, q36_load4(xr + i + 4 * k), float4(char4(*(device const packed_char4 *)(qr + i + 4 * k))));
    for (; i < rounded; i++) {
        float product = xr[i] * float(qr[i]);
        acc = acc + product;
    }
    for (; i < in; i++) acc = fma(xr[i], float(qr[i]), acc);
    return acc;
}

kernel void q36_router_dot(device const char *q [[buffer(0)]],
                           device const float *sc [[buffer(1)]],
                           device const float *x [[buffer(2)]],
                           device float *y [[buffer(3)]],
                           constant q36_dense_args &p [[buffer(4)]],
                           uint z [[thread_position_in_grid]]) {
#pragma clang fp contract(off)
    if (z >= p.rows * p.out) return;
    uint row = z / p.out, o = z % p.out;
    y[z] = q36_router_row(q + ulong(o) * p.in, x + ulong(row) * p.in, p.in) * sc[o];
}

// Decode: the layer's router and the next layer's on its residual (the guess
// of the experts to read ahead) in one dispatch. Each output waits only for
// its loads and its chain of additions: side by side, not one after the other.
kernel void q36_router_dot2(device const char *q [[buffer(0)]],
                            device const float *sc [[buffer(1)]],
                            device const float *x [[buffer(2)]],
                            device float *y [[buffer(3)]],
                            constant q36_dense_args &p [[buffer(4)]],
                            device const char *q2 [[buffer(5)]],
                            device const float *sc2 [[buffer(6)]],
                            device const float *x2 [[buffer(7)]],
                            device float *y2 [[buffer(8)]],
                            uint z [[thread_position_in_grid]]) {
#pragma clang fp contract(off)
    if (z >= 2 * p.out) return;
    if (z < p.out) y[z] = q36_router_row(q + ulong(z) * p.in, x, p.in) * sc[z];
    else y2[z - p.out] = q36_router_row(q2 + ulong(z - p.out) * p.in, x2, p.in) * sc2[z - p.out];
}

// Adds the correction bias to the logits (kept, as the CPU does), then a
// softmax. One thread per row; four elements per load, scalar arithmetic in
// the order of the serial loop.
kernel void q36_router_softmax(device float *logits [[buffer(0)]],
                               device const float *bias [[buffer(1)]],
                               device float *prob [[buffer(2)]],
                               constant q36_router_args &p [[buffer(3)]],
                               uint row [[thread_position_in_grid]]) {
    if (row >= p.rows) return;
    device float *l = logits + ulong(row) * p.E, *pr = prob + ulong(row) * p.E;
    float mx = -1e30f, sum = 0;
    uint e = 0;
    for (; e + 4 <= p.E; e += 4) {
        float4 v = q36_load4(l + e) + q36_load4(bias + e);
        *(device packed_float4 *)(l + e) = packed_float4(v);
        mx = max(max(max(max(mx, v.x), v.y), v.z), v.w);
    }
    for (; e < p.E; e++) {
        float v = l[e] + bias[e];
        l[e] = v;
        mx = max(mx, v);
    }
    for (e = 0; e + 4 <= p.E; e += 4) {
        float4 v = q36_load4(l + e);
        float4 u = float4(exp(v.x - mx), exp(v.y - mx), exp(v.z - mx), exp(v.w - mx));
        *(device packed_float4 *)(pr + e) = packed_float4(u);
        sum += u.x;
        sum += u.y;
        sum += u.z;
        sum += u.w;
    }
    for (; e < p.E; e++) {
        float v = exp(l[e] - mx);
        pr[e] = v;
        sum += v;
    }
    for (e = 0; e + 4 <= p.E; e += 4) {
        float4 v = q36_load4(pr + e);
        *(device packed_float4 *)(pr + e) = packed_float4(float4(v.x / sum, v.y / sum, v.z / sum, v.w / sum));
    }
    for (; e < p.E; e++) pr[e] /= sum;
}

// Group-limited top-k (G groups, T of them kept; G == 1 is plain top-k), then
// the chosen probabilities renormalized. Ties go to the lowest expert id.
// One pass in expert order keeps the K best so far, by decreasing probability
// and, among equal ones, increasing id: the experts and the order that K
// passes of "the largest not taken yet" give, so the same sum and weights.
kernel void q36_router_select(device const float *prob [[buffer(0)]],
                              device int *ids [[buffer(1)]],
                              device float *weights [[buffer(2)]],
                              constant q36_router_args &p [[buffer(3)]],
                              uint row [[thread_position_in_grid]]) {
    if (row >= p.rows) return;
    ulong off = ulong(row) * p.E, route = ulong(row) * p.K;
    bool keep[256];
    if (p.G > 1) {
        for (uint e = 0; e < p.E; e++) keep[e] = false;
        float gs[256];
        bool used[256];
        uint per = p.E / p.G;
        for (uint g = 0; g < p.G; g++) {
            float b1 = -1e30f, b2 = -1e30f;
            for (uint e = g * per; e < (g + 1) * per; e++) {
                float v = prob[off + e];
                if (v > b1) { b2 = b1; b1 = v; } else if (v > b2) b2 = v;
            }
            gs[g] = b1 + b2;
            used[g] = false;
        }
        for (uint t = 0; t < p.T; t++) {
            int best = -1;
            float value = -1e30f;
            for (uint g = 0; g < p.G; g++)
                if (!used[g] && gs[g] > value) { best = int(g); value = gs[g]; }
            if (best < 0) break;
            used[best] = true;
            for (uint e = uint(best) * per; e < (uint(best) + 1) * per; e++) keep[e] = true;
        }
    }
    int chosen[8];
    float val[8], sum = 0;
    uint count = 0;
    for (uint e = 0; e < p.E; e++) {
        float v = prob[off + e];
        if ((p.G > 1 && !keep[e]) || !(v > -1e30f) || (count == p.K && !(v > val[p.K - 1]))) continue;
        uint at = count < p.K ? count++ : p.K - 1;
        for (; at > 0 && v > val[at - 1]; at--) {
            chosen[at] = chosen[at - 1];
            val[at] = val[at - 1];
        }
        chosen[at] = int(e);
        val[at] = v;
    }
    for (uint k = 0; k < p.K; k++) {
        if (k >= count) {                       /* fewer eligible experts than K */
            chosen[k] = -1;
            val[k] = -1e30f;
        }
        sum += val[k];
    }
    for (uint k = 0; k < p.K; k++) {
        ids[route + k] = chosen[k];
        weights[route + k] = sum > 0 ? val[k] / sum : val[k];
    }
}

// One-row batches (generation). The kernels above give a row one thread, which
// leaves all but one GPU thread idle and waits on every load in turn. These
// give the row a threadgroup that loads in parallel into threadgroup memory,
// and keep each serial sum on one thread, in the same order, so the results
// are the same bits. Chosen for rows <= 8 by the wrappers.

// q36_rmsnorm, one threadgroup per row (in <= Q36_ROW_MAX).
#define Q36_ROW_MAX 4096
kernel void q36_rmsnorm_row(device const float *x [[buffer(0)]],
                            device const float *w [[buffer(1)]],
                            device float *y [[buffer(2)]],
                            constant q36_dense_args &p [[buffer(3)]],
                            uint row [[threadgroup_position_in_grid]],
                            ushort tid [[thread_index_in_threadgroup]],
                            ushort nt [[threads_per_threadgroup]]) {
    threadgroup float4 xs4[Q36_ROW_MAX / 4];
    threadgroup float scale;
    threadgroup float *xs = (threadgroup float *)xs4;
    if (row >= p.rows) return;
    device const float *xr = x + ulong(row) * p.in;
    for (uint j = tid; j < p.in; j += nt) xs[j] = xr[j];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {                             // sixteen loaded ahead of their sums
        float ss = 0;
        uint j = 0;
        for (; j + 16 <= p.in; j += 16) {
            float4 v[4] = {xs4[j / 4], xs4[j / 4 + 1], xs4[j / 4 + 2], xs4[j / 4 + 3]};
            for (uint k = 0; k < 4; k++) {
                ss = fma(v[k].x, v[k].x, ss);
                ss = fma(v[k].y, v[k].y, ss);
                ss = fma(v[k].z, v[k].z, ss);
                ss = fma(v[k].w, v[k].w, ss);
            }
        }
        for (; j < p.in; j++) ss = fma(xs[j], xs[j], ss);
        scale = 1.0f / sqrt(ss / float(p.in) + p.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float r = scale;
    device float *yr = y + ulong(row) * p.in;
    for (uint j = tid; j < p.in; j += nt) yr[j] = xs[j] * r * (p.mode ? w[j] : 1.0f + w[j]);
}

// q36_router_softmax, one threadgroup per row (E <= Q36_ROW_MAX): the bias,
// the maximum and the exponentials in parallel (a maximum does not depend on
// the order), the sum on one thread in expert order.
kernel void q36_router_softmax_row(device float *logits [[buffer(0)]],
                                   device const float *bias [[buffer(1)]],
                                   device float *prob [[buffer(2)]],
                                   constant q36_router_args &p [[buffer(3)]],
                                   uint row [[threadgroup_position_in_grid]],
                                   ushort tid [[thread_index_in_threadgroup]],
                                   ushort nt [[threads_per_threadgroup]],
                                   ushort sg [[simdgroup_index_in_threadgroup]],
                                   ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float vals[Q36_ROW_MAX];
    threadgroup float tops[32];
    threadgroup float total;
    if (row >= p.rows) return;
    device float *l = logits + ulong(row) * p.E, *pr = prob + ulong(row) * p.E;
    float mx = -1e30f;
    for (uint e = tid; e < p.E; e += nt) {
        float v = l[e] + bias[e];
        l[e] = v;
        vals[e] = v;
        mx = max(mx, v);
    }
    mx = simd_max(mx);
    if (lane == 0) tops[sg] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = 0; k < (nt + 31u) / 32u; k++) mx = max(mx, tops[k]);
    for (uint e = tid; e < p.E; e += nt) vals[e] = exp(vals[e] - mx);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float sum = 0;
        for (uint e = 0; e < p.E; e++) sum += vals[e];
        total = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum = total;
    for (uint e = tid; e < p.E; e += nt) pr[e] = vals[e] / sum;
}

// q36_router_softmax_row in fast mode: the sum in parallel (SIMD sums, then the 8 SIMD groups'
// in a fixed order) instead of on one thread in expert order, which took 20 us. Not the exact
// order of additions.
kernel void q36_router_softmax_fast(device float *logits [[buffer(0)]],
                                    device const float *bias [[buffer(1)]],
                                    device float *prob [[buffer(2)]],
                                    constant q36_router_args &p [[buffer(3)]],
                                    uint row [[threadgroup_position_in_grid]],
                                    ushort tid [[thread_index_in_threadgroup]],
                                    ushort sg [[simdgroup_index_in_threadgroup]],
                                    ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float maxima[8], sums[8];
    if (row >= p.rows) return;
    device float *l = logits + ulong(row) * p.E, *pr = prob + ulong(row) * p.E;
    float mx = -1e30f;
    for (uint e = tid; e < p.E; e += 256) {
        float v = l[e] + bias[e];
        l[e] = v;
        mx = max(mx, v);
    }
    mx = simd_max(mx);
    if (lane == 0) maxima[sg] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = 0; k < 8; k++) mx = max(mx, maxima[k]);
    float sum = 0;
    for (uint e = tid; e < p.E; e += 256) sum += exp(l[e] - mx);
    sum = simd_sum(sum);
    if (lane == 0) sums[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum = ((sums[0] + sums[1]) + (sums[2] + sums[3])) + ((sums[4] + sums[5]) + (sums[6] + sums[7]));
    for (uint e = tid; e < p.E; e += 256) pr[e] = exp(l[e] - mx) / sum;
}

// q36_router_select without groups (G == 1), one threadgroup per row: every
// eligible expert counts the experts that come before it (a larger
// probability, or the same one and a lower id), and that count is its place:
// the experts and the order of the serial insertion. The sum and the weights
// on one thread, in that order.
kernel void q36_router_select_row(device const float *prob [[buffer(0)]],
                                  device int *ids [[buffer(1)]],
                                  device float *weights [[buffer(2)]],
                                  constant q36_router_args &p [[buffer(3)]],
                                  uint row [[threadgroup_position_in_grid]],
                                  ushort tid [[thread_index_in_threadgroup]],
                                  ushort nt [[threads_per_threadgroup]],
                                  ushort sg [[simdgroup_index_in_threadgroup]],
                                  ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float4 vals4[64];
    threadgroup float *vals = (threadgroup float *)vals4;
    threadgroup int chosen[8];
    threadgroup float val[8];
    threadgroup uint counts[8];               // eligible experts per SIMD group (256 threads)
    if (row >= p.rows) return;
    ulong off = ulong(row) * p.E, route = ulong(row) * p.K;
    uint mine = 0;                              // counted, not an atomic per expert: those took 20 us
    for (uint e = tid; e < p.E; e += nt) {
        float v = prob[off + e];
        vals[e] = v;
        mine += v > -1e30f;
    }
    mine = simd_sum(mine);
    if (lane == 0) counts[sg] = mine;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < p.E; e += nt) {
        float v = vals[e];
        if (!(v > -1e30f)) continue;
        uint place = 0, j = 0;
        // four at a time: a loop of 256 loads, one after the other, took 20 us
#pragma unroll 16
        for (; j + 4 <= p.E; j += 4) {
            float4 u = *(threadgroup const float4 *)(vals + j);
            place += (u.x > -1e30f && (u.x > v || (u.x == v && j < e))) +
                     (u.y > -1e30f && (u.y > v || (u.y == v && j + 1 < e))) +
                     (u.z > -1e30f && (u.z > v || (u.z == v && j + 2 < e))) +
                     (u.w > -1e30f && (u.w > v || (u.w == v && j + 3 < e)));
        }
        for (; j < p.E; j++) {
            float u = vals[j];
            place += u > -1e30f && (u > v || (u == v && j < e));
        }
        if (place < p.K) {
            chosen[place] = int(e);
            val[place] = v;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        uint all = 0;
        for (uint i = 0; i < (nt + 31u) / 32u; i++) all += counts[i];
        uint count = min(all, p.K);
        float sum = 0;
        for (uint k = 0; k < p.K; k++) {
            if (k >= count) {                   /* fewer eligible experts than K */
                chosen[k] = -1;
                val[k] = -1e30f;
            }
            sum += val[k];
        }
        for (uint k = 0; k < p.K; k++) {
            ids[route + k] = chosen[k];
            weights[route + k] = sum > 0 ? val[k] / sum : val[k];
        }
    }
}

// q36_shared_gate, one threadgroup per row (H <= Q36_ROW_MAX): the products
// in parallel, rounded as in the serial loop, the ordered sum on one thread.
kernel void q36_shared_gate_row(device const float *x [[buffer(0)]],
                                device const float *w [[buffer(1)]],
                                device float *gate [[buffer(2)]],
                                constant q36_router_args &p [[buffer(3)]],
                                uint row [[threadgroup_position_in_grid]],
                                ushort tid [[thread_index_in_threadgroup]],
                                ushort nt [[threads_per_threadgroup]]) {
#pragma clang fp contract(off)
    threadgroup float4 products4[Q36_ROW_MAX / 4];
    threadgroup float *products = (threadgroup float *)products4;
    if (row >= p.rows) return;
    device const float *xr = x + ulong(row) * p.H;
    uint rounded = p.H & ~7u, i = 0;
    for (uint j = tid; j < rounded; j += nt) products[j] = xr[j] * w[j];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid != 0) return;
    float sum = 0;
    for (; i + 16 <= rounded; i += 16) {        // sixteen loaded ahead of their sums
        float4 v[4] = {products4[i / 4], products4[i / 4 + 1], products4[i / 4 + 2], products4[i / 4 + 3]};
        for (uint k = 0; k < 4; k++) {
            sum = sum + v[k].x;
            sum = sum + v[k].y;
            sum = sum + v[k].z;
            sum = sum + v[k].w;
        }
    }
    for (; i < rounded; i++) sum = sum + products[i];
    for (i = rounded; i < p.H; i++) sum = fma(xr[i], w[i], sum);
    gate[row] = 1.f / (1.f + exp(-sum));
}

// h = silu(g) * u, in the historical division/multiplication order (it also
// gives a finite zero when exp(-g) overflows).
kernel void q36_silu_mul(device const float *u [[buffer(0)]],
                         device const float *g [[buffer(1)]],
                         device float *h [[buffer(2)]],
                         constant q36_dense_args &p [[buffer(3)]],
                         uint i [[thread_position_in_grid]]) {
    if (i >= p.n) return;
    float v = g[i];
    h[i] = (v / (1.f + exp(-v))) * u[i];
}

// Shared expert gate: sigmoid(x . w) per row, serial and uncontracted like
// the router. Without a gate tensor the factor is 1.
kernel void q36_shared_gate(device const float *x [[buffer(0)]],
                            device const float *w [[buffer(1)]],
                            device float *gate [[buffer(2)]],
                            constant q36_router_args &p [[buffer(3)]],
                            uint row [[thread_position_in_grid]]) {
#pragma clang fp contract(off)
    if (row >= p.rows) return;
    float sum = 0;
    if (p.shared_gate) {
        device const float *xr = x + ulong(row) * p.H;
        uint rounded = p.H & ~7u, i = 0;
        for (; i + 4 <= rounded; i += 4) sum = q36_ordered4(sum, q36_load4(xr + i), q36_load4(w + i));
        for (; i < rounded; i++) {
            float product = xr[i] * w[i];
            sum = sum + product;
        }
        for (; i < p.H; i++) sum = fma(xr[i], w[i], sum);
    }
    gate[row] = p.shared_gate ? 1.f / (1.f + exp(-sum)) : 1.f;
}

// x[row][i] *= s[row]
kernel void q36_scale_rows(device float *x [[buffer(0)]],
                           device const float *s [[buffer(1)]],
                           constant q36_router_args &p [[buffer(2)]],
                           uint i [[thread_position_in_grid]]) {
    if (i >= p.rows * p.H) return;
    x[i] *= s[i / p.H];
}

// Planar int4, groups of 64 inputs: 32 bytes per group, input j < 32 in the
// low nibble of byte j, input j >= 32 in the high nibble of byte j-32, value
// q-8, times one float scale per group. Eight accumulators, fixed final
// order: this is the order the CPU kernel (expert_ffn.h) uses.
inline float q36_dot4(device const uchar *w, device const float *s, device const float *x, uint I, uint o) {
    w += ulong(o) * (I / 2);
    s += ulong(o) * (I / 64);
    float a[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (uint b = 0; b < I / 64; b++) {
        float v[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (uint j = 0; j < 64; j++) {
            uchar byte = w[b * 32 + (j & 31)];
            int q = int(j < 32 ? (byte & 15) : (byte >> 4)) - 8;
            v[j & 7] = fma(float(q), x[b * 64 + j], v[j & 7]);
        }
        for (uint j = 0; j < 8; j++) a[j] = fma(v[j], s[b], a[j]);
    }
    return ((a[0] + a[4]) + (a[2] + a[6])) + ((a[1] + a[5]) + (a[3] + a[7]));
}

// Routed experts are computed by expert, not by token: every expert is read
// once per batch, for all the tokens that chose it. A slot holds one expert as
// the cache keeps it: gate (I rows of H), up (I rows), down (H rows of I),
// then the scales of the three from byte p.scales. Up to eight slots are bound
// at buffers 4..11. Each entry of pairs[p.first .. p.first + p.n) names a slot
// (bits 28..30) and a choice row * K + k (bits 0..27).
#define Q36_SLOTS device const uchar *w0 [[buffer(4)]], device const uchar *w1 [[buffer(5)]], \
                  device const uchar *w2 [[buffer(6)]], device const uchar *w3 [[buffer(7)]], \
                  device const uchar *w4 [[buffer(8)]], device const uchar *w5 [[buffer(9)]], \
                  device const uchar *w6 [[buffer(10)]], device const uchar *w7 [[buffer(11)]]

// h[choice][i] = silu(gate_i . x[row]) * (up_i . x[row])
kernel void q36_expert_gate_up(device const float *x [[buffer(0)]],
                               device const uint *pairs [[buffer(1)]],
                               device float *h [[buffer(2)]],
                               constant q36_expert_args &p [[buffer(3)]],
                               Q36_SLOTS,
                               uint2 at [[thread_position_in_grid]]) {
    if (at.x >= p.I || at.y >= p.n) return;
    device const uchar *ws[8] = {w0, w1, w2, w3, w4, w5, w6, w7};
    uint pair = pairs[p.first + at.y], choice = pair & 0x0fffffffu;
    device const uchar *w = ws[pair >> 28];
    device const float *sc = (device const float *)(w + p.scales);
    device const float *row = x + ulong(choice / p.K) * p.H;
    ulong n = ulong(p.H) * p.I;
    float g = q36_dot4(w, sc, row, p.H, at.x), u = q36_dot4(w + n / 2, sc + n / 64, row, p.H, at.x);
    // Same operation order as the CPU SwiGLU: equivalent-looking forms round
    // differently for negative inputs and can move later routing decisions.
    float z = g / (1.f + exp(-g));
    h[ulong(choice) * p.I + at.x] = z * u;
}

// y[choice][o] = down_o . h[choice], not weighted yet
kernel void q36_expert_down(device const float *h [[buffer(0)]],
                            device const uint *pairs [[buffer(1)]],
                            device float *y [[buffer(2)]],
                            constant q36_expert_args &p [[buffer(3)]],
                            Q36_SLOTS,
                            uint2 at [[thread_position_in_grid]]) {
    if (at.x >= p.H || at.y >= p.n) return;
    device const uchar *ws[8] = {w0, w1, w2, w3, w4, w5, w6, w7};
    uint pair = pairs[p.first + at.y], choice = pair & 0x0fffffffu;
    device const uchar *w = ws[pair >> 28];
    device const float *sc = (device const float *)(w + p.scales);
    ulong n = ulong(p.H) * p.I;
    y[ulong(choice) * p.H + at.x] = q36_dot4(w + n, sc + 2 * n / 64, h + ulong(choice) * p.I, p.I, at.x);
}

// out[row][i] = residual[row][i] + (sum_k weight[row][k] * y[row][k][i]) + shared[row][i],
// in exactly that association and in the router's order of k, whatever order
// the experts were computed in.
kernel void q36_expert_sum(device const float *y [[buffer(0)]],
                           device const float *weights [[buffer(1)]],
                           device const float *shared [[buffer(2)]],
                           device const float *residual [[buffer(3)]],
                           device float *out [[buffer(4)]],
                           constant q36_expert_args &p [[buffer(5)]],
                           uint i [[thread_position_in_grid]]) {
    if (i >= p.rows * p.H) return;
    uint row = i / p.H, o = i % p.H;
    float sum = 0;
    for (uint k = 0; k < p.K; k++) {
        float v = y[(ulong(row) * p.K + k) * p.H + o];
        sum = sum + weights[row * p.K + k] * v;
    }
    sum += shared[i];
    sum = residual[i] + sum;
    out[i] = sum;
}

// ---------------------------------------------------------------------------
// Keep-alive (q36_gpu_keepalive): FMAs on registers, one float per thread of
// memory traffic. Run on a second queue while decode waits for the CPU, it
// keeps the power manager from taking those waits for idleness and lowering
// the clocks (the trick of ds4's tensor-parallel keep-alive).
// ---------------------------------------------------------------------------

kernel void q36_keepalive(device float *out [[buffer(0)]],
                          constant uint &iters [[buffer(1)]],
                          uint i [[thread_position_in_grid]]) {
    float a = out[i];
    for (uint k = 0; k < iters; k++) {
        a = fma(a, 1.000001f, 0.000001f);
        a = fma(a, 1.000001f, -0.000001f);
    }
    out[i] = a;
}

// ---------------------------------------------------------------------------
// Fast path of prompt batches. QWEN36_METAL_EXACT=1 keeps the exact kernels.
// ---------------------------------------------------------------------------

// y[row][o] = sc[o] * sum_i q[o][i] * x[row][i] with the GPU's 8x8 matrix operations. A
// threadgroup of 4 SIMD groups computes 32 rows x 128 outputs, each SIMD group all 32 rows of
// its own 32 outputs (16 accumulators). Every lane reads its weights straight from memory into
// the matrix operand (thread_elements): no weight goes through threadgroup memory, and each is
// read once per threadgroup. Only the rows of x do, a step of 32 inputs at a time, in the
// weights' order: lane row m of sub-step j takes input 4m + j, so a lane reads 4 contiguous
// weight bytes per output row. The scale multiplies each sum once, at the end. Float
// throughout, but not q36_dot_i8's order of additions: the results differ from it in the last
// bits. Needs in % 32 == 0 and out % 32 == 0; x 16-byte aligned, y 8-byte aligned. With p.group
// set (a multiple of 32), output o takes weight row (o / group) * stride + o % group: one part of
// each head of a projection whose heads hold several parts. With p.ld set the rows of y are ld
// wide (the first out columns of a product whose others are computed elsewhere).
kernel void q36_dot_i8_mm(device const char *q [[buffer(0)]],
                          device const float *sc [[buffer(1)]],
                          device const float *x [[buffer(2)]],
                          device float *y [[buffer(3)]],
                          constant q36_dense_args &p [[buffer(4)]],
                          threadgroup float *shared [[threadgroup(0)]],
                          uint3 tg [[threadgroup_position_in_grid]],
                          ushort tid [[thread_index_in_threadgroup]],
                          ushort sg [[simdgroup_index_in_threadgroup]],
                          ushort lane [[thread_index_in_simdgroup]]) {
    // shared: the step's x tile as 8x8 blocks [sub-step j][row block][row][m]
    const uint r0 = tg.x * 32, o0 = tg.y * 128 + 32 * sg, nr = min(32u, p.rows - r0);
    // In the last tile of a width that is not a multiple of 128 a SIMD group may have no
    // outputs: it computes on row 0 like the others (a condition in the loop costs 15%) and
    // does not store.
    const bool live = o0 < p.out;
    const ushort2 pl = q36_lane_place(lane);
    const ushort sn = pl.x, sm = pl.y;
    const uint in4 = p.in / 4;
    // the weight row of output o0; the SIMD group's 32 outputs take 32 rows in a row
    const uint w0 = p.group ? o0 / p.group * p.stride + o0 % p.group : o0;
    device const char4 *wq = (device const char4 *)(q + ulong(live ? w0 + sn : sn) * p.in) + sm;
    // x: thread t stages row t / 4 (the last one when the tile is short), inputs 8 (t % 4) .. + 7
    const uint xrow = tid / 4, xc = 8 * (tid % 4);
    device const float4 *xr = (device const float4 *)(x + ulong(r0 + min(xrow, nr - 1)) * p.in + xc);
    float2 acc[16];
#pragma unroll
    for (uint i = 0; i < 16; i++) acc[i] = float2(0.f);
    for (uint k4 = 0; k4 < in4; k4 += 8) {
        float4 xa = xr[k4], xb = xr[k4 + 1];
        char4 wv[8];
#pragma unroll
        for (uint cb = 0; cb < 4; cb++) {
            wv[2 * cb] = wq[ulong(8 * cb) * in4 + k4];
            wv[2 * cb + 1] = wq[ulong(8 * cb + 1) * in4 + k4];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float xv[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
#pragma unroll
        for (uint i = 0; i < 8; i++) {
            uint c = xc + i;
            shared[(c % 4 * 4 + xrow / 8) * 64 + 8 * (xrow % 8) + c / 4] = xv[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
        for (uint j = 0; j < 4; j++) {
            simdgroup_float8x8 a[4];
#pragma unroll
            for (uint rb = 0; rb < 4; rb++) simdgroup_load(a[rb], shared + (j * 4 + rb) * 64, 8);
            float2 bf[4];
#pragma unroll
            for (uint cb = 0; cb < 4; cb++) bf[cb] = float2(float(wv[2 * cb][j]), float(wv[2 * cb + 1][j]));
            // Accumulators live as float2 and become matrices only around each operation: an
            // array of matrices whose elements are taken goes to memory, 30 times slower.
#pragma unroll
            for (uint rb = 0; rb < 4; rb++)
#pragma unroll
                for (uint cb = 0; cb < 4; cb++) {
                    simdgroup_float8x8 bm, cm;
                    reinterpret_cast<thread float2 &>(bm.thread_elements()) = bf[cb];
                    reinterpret_cast<thread float2 &>(cm.thread_elements()) = acc[4 * rb + cb];
                    simdgroup_multiply_accumulate(cm, a[rb], bm, cm);
                    acc[4 * rb + cb] = reinterpret_cast<thread float2 &>(cm.thread_elements());
                }
        }
    }
    if (!live) return;
    const uint ldy = p.ld ? p.ld : p.out;
#pragma unroll
    for (uint cb = 0; cb < 4; cb++) {
        float2 s2 = float2(sc[w0 + 8 * cb + sn], sc[w0 + 8 * cb + sn + 1]);
#pragma unroll
        for (uint rb = 0; rb < 4; rb++) {
            uint row = 8 * rb + sm;
            if (row < nr) *(device float2 *)(y + ulong(r0 + row) * ldy + o0 + 8 * cb + sn) = acc[4 * rb + cb] * s2;
        }
    }
}

// Prompt attention on the matrix units, for a chunk of rows (the wrapper
// offsets q and out and passes the chunk's rows and first position): scores,
// softmax, values as three passes. Float throughout; not q36_attn's order of
// operations (no running maximum, plain sums), so not bit-identical to it.
// Scores and probabilities live in s[row][head][cap].

// s[r][h][t] = scale * q[r][h] . k[kh][t] for t < pos + rows (masked later).
// Four SIMD groups: 16 rows x 32 positions, head_dim 32 at a time.
template <typename KV>
kernel void q36_attn_qk(device const float *q [[buffer(0)]],
                        device const KV *kc [[buffer(1)]],
                        device float *s [[buffer(2)]],
                        constant q36_state_args &p [[buffer(3)]],
                        threadgroup float *shared [[threadgroup(0)]],
                        uint3 tg [[threadgroup_position_in_grid]],
                        ushort tid [[thread_index_in_threadgroup]],
                        ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float *sa = shared, *sb = shared + 32 * 32;
    const uint h = tg.z, kh = h / (p.qh / p.kv), n = p.pos + p.rows;
    const uint t0 = tg.y * 32, r0 = tg.x * 16, nr = min(16u, p.rows - r0), nt = min(32u, n - t0);
    const uint lo = tid / 4, lk = 8 * (tid % 4), lrow = tid / 8, lj = 4 * (tid % 8);
    device const KV *kt = kc + (ulong(kh) * p.cap + t0 + min(lo, nt - 1)) * p.hd + lk;
    device const float *qr = q + (ulong(r0 + min(lrow, nr - 1)) * p.qh + h) * p.hd + lj;
    simdgroup_float8x8 acc[2];
    for (uint i = 0; i < 2; i++) acc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    for (uint k = 0; k < p.hd; k += 32) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = 0; i < 8; i++) sa[64 * (4 * ((lk + i) / 8) + lo / 8) + 8 * ((lk + i) % 8) + lo % 8] = float(kt[k + i]);
        for (uint i = 0; i < 4; i++) sb[64 * (2 * (lj / 8) + lrow / 8) + 8 * (lrow % 8) + lj % 8 + i] = qr[k + i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup const float *ma = sa + 2 * 64 * (sg % 2), *mb = sb + 64 * (sg / 2);
        for (uint kb = 0; kb < 4; kb++) {
            simdgroup_float8x8 a[2], b;
            for (uint i = 0; i < 2; i++) simdgroup_load(a[i], ma + 64 * i, 8);
            simdgroup_load(b, mb, 8);
            for (uint i = 0; i < 2; i++) simdgroup_multiply_accumulate(acc[i], b, a[i], acc[i]);
            ma += 4 * 64;
            mb += 2 * 64;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float *t = shared + 8 * (sg / 2) * 32 + 16 * (sg % 2);
    for (uint i = 0; i < 2; i++) simdgroup_store(acc[i], t + 8 * i, 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float scale = 1.0f / sqrt(float(p.hd));
    for (uint e = tid; e < 16 * 32; e += 128) {
        uint r = e / 32, c = e % 32;
        if (r < nr && c < nt) s[(ulong(r0 + r) * p.qh + h) * p.cap + t0 + c] = shared[e] * scale;
    }
}
template [[host_name("q36_attn_qk")]] kernel void q36_attn_qk<float>(device const float *, device const float *,
        device float *, constant q36_state_args &, threadgroup float *, uint3, ushort, ushort);
template [[host_name("q36_attn_qk_half")]] kernel void q36_attn_qk<half>(device const float *, device const half *,
        device float *, constant q36_state_args &, threadgroup float *, uint3, ushort, ushort);

// Each row's probabilities over its positions t <= pos + row, zero after
// them up to pos + rows. One SIMD group per row and head.
kernel void q36_attn_softmax(device float *s [[buffer(0)]],
                             constant q36_state_args &p [[buffer(1)]],
                             uint z [[threadgroup_position_in_grid]],
                             ushort lane [[thread_index_in_simdgroup]]) {
    if (z >= p.rows * p.qh) return;
    uint row = z / p.qh, h = z % p.qh, valid = p.pos + row + 1, n = p.pos + p.rows;
    device float *x = s + (ulong(row) * p.qh + h) * p.cap;
    float mx = -INFINITY, sum = 0;
    for (uint t = lane; t < valid; t += 32) mx = max(mx, x[t]);
    mx = simd_max(mx);
    for (uint t = lane; t < valid; t += 32) {
        float e = exp(x[t] - mx);
        x[t] = e;
        sum += e;
    }
    float inv = 1.0f / simd_sum(sum);
    for (uint t = lane; t < n; t += 32) x[t] = t < valid ? x[t] * inv : 0.0f;
}

// out[r][h][d] = sum_t s[r][h][t] * v[kh][t][d], t < pos + rows.
// Four SIMD groups: 16 rows x 32 columns, positions 32 at a time.
template <typename KV>
kernel void q36_attn_pv(device const float *s [[buffer(0)]],
                        device const KV *vc [[buffer(1)]],
                        device float *out [[buffer(2)]],
                        constant q36_state_args &p [[buffer(3)]],
                        threadgroup float *shared [[threadgroup(0)]],
                        uint3 tg [[threadgroup_position_in_grid]],
                        ushort tid [[thread_index_in_threadgroup]],
                        ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float *sa = shared, *sb = shared + 32 * 32;
    const uint h = tg.z, kh = h / (p.qh / p.kv), n = p.pos + p.rows;
    const uint d0 = tg.y * 32, r0 = tg.x * 16, nr = min(16u, p.rows - r0);
    // values: position lt, columns lc..lc+7 (contiguous); probabilities as rows
    const uint lt = tid / 4, lc = 8 * (tid % 4), lrow = tid / 8, lj = 4 * (tid % 8);
    device const KV *vt = vc + ulong(kh) * p.cap * p.hd + d0 + lc;
    device const float *pr = s + (ulong(r0 + min(lrow, nr - 1)) * p.qh + h) * p.cap + lj;
    simdgroup_float8x8 acc[2];
    for (uint i = 0; i < 2; i++) acc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    for (uint k = 0; k < n; k += 32) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = 0; i < 8; i++)           /* positions past the cache count as zero */
            sa[64 * (4 * (lt / 8) + (lc + i) / 8) + 8 * (lt % 8) + (lc + i) % 8] =
                k + lt < n ? float(vt[ulong(k + lt) * p.hd + i]) : 0.0f;
        for (uint i = 0; i < 4; i++) sb[64 * (2 * (lj / 8) + lrow / 8) + 8 * (lrow % 8) + lj % 8 + i] = k + lj + i < n ? pr[k + i] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup const float *ma = sa + 2 * 64 * (sg % 2), *mb = sb + 64 * (sg / 2);
        for (uint kb = 0; kb < 4; kb++) {
            simdgroup_float8x8 a[2], b;
            for (uint i = 0; i < 2; i++) simdgroup_load(a[i], ma + 64 * i, 8);
            simdgroup_load(b, mb, 8);
            for (uint i = 0; i < 2; i++) simdgroup_multiply_accumulate(acc[i], b, a[i], acc[i]);
            ma += 4 * 64;
            mb += 2 * 64;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float *t = shared + 8 * (sg / 2) * 32 + 16 * (sg % 2);
    for (uint i = 0; i < 2; i++) simdgroup_store(acc[i], t + 8 * i, 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < nr * 32; e += 128)
        out[(ulong(r0 + e / 32) * p.qh + h) * p.hd + d0 + e % 32] = shared[e];
}
template [[host_name("q36_attn_pv")]] kernel void q36_attn_pv<float>(device const float *, device const float *,
        device float *, constant q36_state_args &, threadgroup float *, uint3, ushort, ushort);
template [[host_name("q36_attn_pv_half")]] kernel void q36_attn_pv<half>(device const float *, device const half *,
        device float *, constant q36_state_args &, threadgroup float *, uint3, ushort, ushort);

// Prompt attention where a KV head serves eight query heads, as on the 35B model, with the
// matrix operations of q36_dot_i8_mm. The 8 query heads of KV head kh are 8 consecutive vectors
// of a row, so 4 rows of them are 32 query vectors: the rows of both products. Two passes per
// chunk of rows: the scores, with each query vector's maximum over every 32 positions, then the
// values, which take the exponentials while staging them, add them up and divide by their sum
// at the end: the softmax has no pass of its own over the scores. tmax[row][head][t / 32]
// follows the scores in the scratch.

// s[r][h][t] = scale * q[r][h] . k[kh][t] for t < pos + rows: a threadgroup of 4 SIMD groups
// takes 32 query vectors x 128 positions, a SIMD group its own 32 positions (16 accumulators).
// Lanes read the keys straight from the cache into the matrix operand: in a step of 32
// dimensions, lane row m of sub-step j takes dimension 4m + j, 4 contiguous elements of each of
// its 8 positions. The query vectors go through threadgroup memory in the same order. Positions
// past the chunk read the last one and are not stored. Grid (ceil(rows / 4), ceil(n / 128), kv).
template <typename KV>
kernel void q36_attn_qk8(device const float *q [[buffer(0)]],
                         device const KV *kc [[buffer(1)]],
                         device float *s [[buffer(2)]],
                         constant q36_state_args &p [[buffer(3)]],
                         device float *tmax [[buffer(4)]],
                         threadgroup float *shared [[threadgroup(0)]],
                         uint3 tg [[threadgroup_position_in_grid]],
                         ushort tid [[thread_index_in_threadgroup]],
                         ushort sg [[simdgroup_index_in_threadgroup]],
                         ushort lane [[thread_index_in_simdgroup]]) {
    const uint kh = tg.z, gq = p.qh / p.kv, n = p.pos + p.rows;
    const uint r0 = tg.x * 4, t0 = tg.y * 128 + 32 * sg;
    const ushort2 pl = q36_lane_place(lane);
    const ushort sn = pl.x, sm = pl.y;
    device const KV *kb = kc + ulong(kh) * p.cap * p.hd + 4 * sm;
    uint tofs[8];                               // the lane's positions t0 + 8 cb + sn (+ 1)
#pragma unroll
    for (uint cb = 0; cb < 4; cb++)
#pragma unroll
        for (uint u = 0; u < 2; u++) tofs[2 * cb + u] = min(t0 + 8 * cb + sn + u, n - 1) * p.hd;
    // staging: thread t loads query vector t / 4 (row r0 + v / 8, head 8 kh + v % 8), dims 8 (t % 4) ..
    const uint qv = tid / 4, qc = 8 * (tid % 4), rr = min(r0 + qv / gq, p.rows - 1);
    device const float4 *qr = (device const float4 *)(q + (ulong(rr) * p.qh + kh * gq + qv % gq) * p.hd + qc);
    float2 acc[16];
#pragma unroll
    for (uint i = 0; i < 16; i++) acc[i] = float2(0.f);
    for (uint k4 = 0; k4 < p.hd / 4; k4 += 8) {
        float4 xa = qr[k4], xb = qr[k4 + 1];
        vec<KV, 4> kv[8];                       // converted only when used: half the registers
#pragma unroll
        for (uint i = 0; i < 8; i++) kv[i] = *(device const vec<KV, 4> *)(kb + tofs[i] + 4 * k4);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float xv[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
#pragma unroll
        for (uint i = 0; i < 8; i++) {
            uint c = qc + i;
            shared[(c % 4 * 4 + qv / 8) * 64 + 8 * (qv % 8) + c / 4] = xv[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
        for (uint j = 0; j < 4; j++) {
            simdgroup_float8x8 a[4];
#pragma unroll
            for (uint rb = 0; rb < 4; rb++) simdgroup_load(a[rb], shared + (j * 4 + rb) * 64, 8);
            float2 bf[4];
#pragma unroll
            for (uint cb = 0; cb < 4; cb++) bf[cb] = float2(float(kv[2 * cb][j]), float(kv[2 * cb + 1][j]));
#pragma unroll
            for (uint rb = 0; rb < 4; rb++)
#pragma unroll
                for (uint cb = 0; cb < 4; cb++) {
                    simdgroup_float8x8 bm, cm;
                    reinterpret_cast<thread float2 &>(bm.thread_elements()) = bf[cb];
                    reinterpret_cast<thread float2 &>(cm.thread_elements()) = acc[4 * rb + cb];
                    simdgroup_multiply_accumulate(cm, a[rb], bm, cm);
                    acc[4 * rb + cb] = reinterpret_cast<thread float2 &>(cm.thread_elements());
                }
        }
    }
    const float scale = 1.0f / sqrt(float(p.hd));
#pragma unroll
    for (uint rb = 0; rb < 4; rb++) {
        uint v = 8 * rb + sm, r = r0 + v / gq;
        if (r >= p.rows) continue;
        device float *sr = s + (ulong(r) * p.qh + kh * gq + v % gq) * p.cap;
#pragma unroll
        for (uint cb = 0; cb < 4; cb++) {
            uint t = t0 + 8 * cb + sn;
            if (t + 1 < n) *(device float2 *)(sr + t) = acc[4 * rb + cb] * scale;
            else if (t < n) sr[t] = acc[4 * rb + cb].x * scale;
        }
    }
    // The maximum of each query vector over its row's positions t <= pos + row among the SIMD
    // group's 32: the 4 lanes of a matrix row (lane bits 0 and 3) combine theirs. The scale is
    // positive, so it is the maximum of the stored scores.
    if (t0 >= n) return;
    const uint ntl = (p.cap + 31) / 32;
#pragma unroll
    for (uint rb = 0; rb < 4; rb++) {
        uint v = 8 * rb + sm, r = r0 + v / gq, valid = p.pos + min(r, p.rows - 1) + 1;
        float m = -INFINITY;
#pragma unroll
        for (uint cb = 0; cb < 4; cb++) {
            uint t = t0 + 8 * cb + sn;
            if (t < valid) m = max(m, acc[4 * rb + cb].x);
            if (t + 1 < valid) m = max(m, acc[4 * rb + cb].y);
        }
        m = max(m, simd_shuffle_xor(m, 1));
        m = max(m, simd_shuffle_xor(m, 8));
        if (sn == 0 && r < p.rows) tmax[(ulong(r) * p.qh + kh * gq + v % gq) * ntl + t0 / 32] = m * scale;
    }
}
template [[host_name("q36_attn_qk8")]] kernel void q36_attn_qk8<float>(device const float *, device const float *,
        device float *, constant q36_state_args &, device float *, threadgroup float *, uint3, ushort, ushort,
        ushort);
template [[host_name("q36_attn_qk8_half")]] kernel void q36_attn_qk8<half>(device const float *, device const half *,
        device float *, constant q36_state_args &, device float *, threadgroup float *, uint3, ushort, ushort,
        ushort);

// out[r][h][d] = sum_t e[r][h][t] * v[kh][t][d] / sum_t e[r][h][t], e = exp(s - max) for
// t <= pos + r, 0 after: the 32 query vectors of q36_attn_qk8 are the rows, and 8 SIMD groups
// take the 256 dimensions of a head, 32 each (16 accumulators), so each exponential is taken
// once per block of query vectors (two threadgroups of 128 dimensions took each twice: 4% slower
// at 8192 positions). The positions of a step of 32 are the matrix rows of the values: lane
// (sm, sn) reads v[t0 + 8j + sm][d0 + 8cb + sn .. + 1] straight from the cache. The thread that
// stages 4 exponentials of a query vector takes them from its scores, and keeps their sum; the
// 8 threads of a query vector reduce its maximum first and its sum at the end. The exponentials
// are the GPU's fast ones (relative error near 1e-6): the precise exp made this kernel slower
// than the separate pass it replaces. Grid (ceil(rows / 4), hd / 256, kv), 256 threads.
template <typename KV>
kernel void q36_attn_pv8(device const float *s [[buffer(0)]],
                         device const KV *vc [[buffer(1)]],
                         device float *out [[buffer(2)]],
                         constant q36_state_args &p [[buffer(3)]],
                         device const float *tmax [[buffer(4)]],
                         threadgroup float *shared [[threadgroup(0)]],
                         uint3 tg [[threadgroup_position_in_grid]],
                         ushort tid [[thread_index_in_threadgroup]],
                         ushort sg [[simdgroup_index_in_threadgroup]],
                         ushort lane [[thread_index_in_simdgroup]]) {
    const uint kh = tg.z, gq = p.qh / p.kv, n = p.pos + p.rows;
    const uint r0 = tg.x * 4, d0 = tg.y * 256 + 32 * sg;
    const ushort2 pl = q36_lane_place(lane);
    const ushort sn = pl.x, sm = pl.y;
    device const KV *vt = vc + ulong(kh) * p.cap * p.hd + d0 + sn;
    const uint qv = tid / 8, qc = 4 * (tid % 8), rr = min(r0 + qv / gq, p.rows - 1);
    device const float *pr = s + (ulong(rr) * p.qh + kh * gq + qv % gq) * p.cap + qc;
    const uint ntl = (p.cap + 31) / 32, valid = p.pos + rr + 1;
    device const float *tm = tmax + (ulong(rr) * p.qh + kh * gq + qv % gq) * ntl;
    float mx = -INFINITY, sum = 0;
    for (uint i = tid % 8; i < (n + 31) / 32; i += 8) mx = max(mx, tm[i]);
    mx = max(mx, simd_shuffle_xor(mx, 1));
    mx = max(mx, simd_shuffle_xor(mx, 2));
    mx = max(mx, simd_shuffle_xor(mx, 4));
    float2 acc[16];
#pragma unroll
    for (uint i = 0; i < 16; i++) acc[i] = float2(0.f);
    for (uint t0 = 0; t0 < n; t0 += 32) {
        float e8[4];
#pragma unroll
        for (uint i = 0; i < 4; i++) {
            e8[i] = t0 + qc + i < valid ? fast::exp(pr[t0 + i] - mx) : 0.0f;
            sum += e8[i];
        }
        vec<KV, 2> vv[16];                      // converted only when used: half the registers
#pragma unroll
        for (uint j = 0; j < 4; j++)
#pragma unroll
            for (uint cb = 0; cb < 4; cb++) {
                uint t = min(t0 + 8 * j + sm, n - 1);
                vv[4 * j + cb] = *(device const vec<KV, 2> *)(vt + ulong(t) * p.hd + 8 * cb);
            }
        threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
        for (uint i = 0; i < 4; i++) {
            uint c = qc + i;
            shared[(c / 8 * 4 + qv / 8) * 64 + 8 * (qv % 8) + c % 8] = e8[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
        for (uint j = 0; j < 4; j++) {
            simdgroup_float8x8 a[4];
#pragma unroll
            for (uint rb = 0; rb < 4; rb++) simdgroup_load(a[rb], shared + (j * 4 + rb) * 64, 8);
            float2 bf[4];
#pragma unroll
            for (uint cb = 0; cb < 4; cb++) bf[cb] = float2(vv[4 * j + cb]);
#pragma unroll
            for (uint rb = 0; rb < 4; rb++)
#pragma unroll
                for (uint cb = 0; cb < 4; cb++) {
                    simdgroup_float8x8 bm, cm;
                    reinterpret_cast<thread float2 &>(bm.thread_elements()) = bf[cb];
                    reinterpret_cast<thread float2 &>(cm.thread_elements()) = acc[4 * rb + cb];
                    simdgroup_multiply_accumulate(cm, a[rb], bm, cm);
                    acc[4 * rb + cb] = reinterpret_cast<thread float2 &>(cm.thread_elements());
                }
        }
    }
    sum += simd_shuffle_xor(sum, 1);
    sum += simd_shuffle_xor(sum, 2);
    sum += simd_shuffle_xor(sum, 4);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid % 8 == 0) shared[qv] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
    for (uint rb = 0; rb < 4; rb++) {
        uint v = 8 * rb + sm, r = r0 + v / gq;
        if (r >= p.rows) continue;
        uint z = r * p.qh + kh * gq + v % gq;
        float w = 1.0f / shared[v];
        device float *o = out + ulong(z) * p.hd + d0 + sn;
#pragma unroll
        for (uint cb = 0; cb < 4; cb++) *(device float2 *)(o + 8 * cb) = acc[4 * rb + cb] * w;
    }
}
template [[host_name("q36_attn_pv8")]] kernel void q36_attn_pv8<float>(device const float *, device const float *,
        device float *, constant q36_state_args &, device const float *, threadgroup float *, uint3, ushort, ushort,
        ushort);
template [[host_name("q36_attn_pv8_half")]] kernel void q36_attn_pv8<half>(device const float *, device const half *,
        device float *, constant q36_state_args &, device const float *, threadgroup float *, uint3, ushort, ushort,
        ushort);

// The choices of each slot of an expert group are contiguous in pairs: slot j has rg.cnt[j] of
// them from pairs[p.first + rg.off[j]].
struct q36_expert_ranges { uint off[8], cnt[8]; };

// Routed experts of a prompt batch, as q36_dot_i8_mm computes the projections: a threadgroup of
// 4 SIMD groups takes up to 64 choices of one slot and 64 columns, SIMD group sg 16 of the
// columns for all the choices (8 row blocks of 8, 2 column blocks), and every lane reads its
// int4 weights straight from the slot into the matrix operand, decoded and scaled on the way
// (the block scales multiply each weight instead of each block sum: not the exact kernels'
// rounding). A decoded weight feeds up to 8 products, and a slot of up to 64 choices reads and
// decodes its weights once (tiles of 32 choices: 4 products, and a second tile for 33 to 64
// choices). The choices' rows go through threadgroup memory 32 inputs at a time, in the
// weights' order: in a block of 64 inputs, lane row m of sub-step j of half hf takes input
// 32 hf + 4m + j, which is nibble hf of byte 4m + j, so a lane reads one word per output row
// and block. Thread t stages row t / 2, inputs 16 (t % 2) .. + 15 of the half. Row blocks of 8
// past the slot's last choice are skipped: a short slot costs its rows, not a whole tile.
//   wrow(cb, t): the word offset of output row t (0, 1) of column block cb in block 0;
//   srow(cb, t): the same row's first scale; xr: the staging thread's row of inputs;
//   nrb: the row blocks of 8 the tile has, a constant: a test inside the loop costs 30%.
// Tried and slower on the M1: loads of the next block or half ahead (-16%, registers), the
// decoding as one fma (-4%), 8 SIMD groups per threadgroup (-8%), inputs read straight from
// memory instead of staged (-8%).
#define Q36_EXPERT_MM_LOOP(inputs, wrow, srow, nrb) \
    for (uint b = 0; b < (inputs) / 64; b++) { \
        uint wv[4]; \
        float sv[4]; \
        _Pragma("unroll") for (uint cb = 0; cb < 2; cb++) \
            _Pragma("unroll") for (uint t = 0; t < 2; t++) { \
                wv[2 * cb + t] = wq[wrow(cb, t) + b * 8]; \
                sv[2 * cb + t] = sq[srow(cb, t) + b]; \
            } \
        _Pragma("unroll") for (uint hf = 0; hf < 2; hf++) { \
            float4 x0 = xr[16 * b + 8 * hf], x1 = xr[16 * b + 8 * hf + 1]; \
            float4 x2 = xr[16 * b + 8 * hf + 2], x3 = xr[16 * b + 8 * hf + 3]; \
            threadgroup_barrier(mem_flags::mem_threadgroup); \
            float xv[16] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w, \
                            x2.x, x2.y, x2.z, x2.w, x3.x, x3.y, x3.z, x3.w}; \
            _Pragma("unroll") for (uint i = 0; i < 16; i++) { \
                uint c = xc + i; \
                shared[(c % 4 * 8 + xrow / 8) * 64 + 8 * (xrow % 8) + c / 4] = xv[i]; \
            } \
            threadgroup_barrier(mem_flags::mem_threadgroup); \
            _Pragma("unroll") for (uint j = 0; j < 4; j++) { \
                float2 bf[2]; \
                _Pragma("unroll") for (uint cb = 0; cb < 2; cb++) \
                    bf[cb] = float2(float(int(extract_bits(wv[2 * cb], 8 * j + 4 * hf, 4)) - 8) * sv[2 * cb], \
                                    float(int(extract_bits(wv[2 * cb + 1], 8 * j + 4 * hf, 4)) - 8) * sv[2 * cb + 1]); \
                _Pragma("unroll") for (uint rb = 0; rb < 8; rb++) { \
                    if (rb >= nrb) continue; \
                    simdgroup_float8x8 a; \
                    simdgroup_load(a, shared + (j * 8 + rb) * 64, 8); \
                    _Pragma("unroll") for (uint cb = 0; cb < 2; cb++) { \
                        simdgroup_float8x8 bm, cm; \
                        reinterpret_cast<thread float2 &>(bm.thread_elements()) = bf[cb]; \
                        reinterpret_cast<thread float2 &>(cm.thread_elements()) = acc[2 * rb + cb]; \
                        simdgroup_multiply_accumulate(cm, a, bm, cm); \
                        acc[2 * rb + cb] = reinterpret_cast<thread float2 &>(cm.thread_elements()); \
                    } \
                } \
            } \
        } \
    }

// The last tile of a slot when it has 1 to 4 row blocks (a slot of up to 32 choices, or the tail
// of a larger one), in kernels of their own (q36_expert_*_mm4): each lane reads its own elements
// of the inputs straight from memory, lane row m of row block rb taking row 8 rb + m and, in half
// hf of block b, the 8 inputs 64 b + 32 hf + 4 sn .. + 7, its two columns of the 4 sub-steps; no
// threadgroup memory and no barriers. The matrix operands hold the values the staging gives, so
// the results are the same bits. 1 to 8 choices of a slot: 75 -> 54 us an expert of the 35B model
// on the M1; 32 choices 161 -> 133. Its registers stay out of the kernels of larger tiles: in one
// kernel together their full tiles were 4% slower.
#define Q36_EXPERT_MM_DIRECT(inputs, wrow, srow, nrb, xrowp) \
    { \
    device const float4 *xd[4]; \
    _Pragma("unroll") for (uint rb = 0; rb < (nrb); rb++) xd[rb] = xrowp(min(8 * rb + sm, nc - 1)); \
    for (uint b = 0; b < (inputs) / 64; b++) { \
        uint wv[4]; \
        float sv[4]; \
        _Pragma("unroll") for (uint cb = 0; cb < 2; cb++) \
            _Pragma("unroll") for (uint t = 0; t < 2; t++) { \
                wv[2 * cb + t] = wq[wrow(cb, t) + b * 8]; \
                sv[2 * cb + t] = sq[srow(cb, t) + b]; \
            } \
        _Pragma("unroll") for (uint hf = 0; hf < 2; hf++) { \
            float4 xa[4], xb[4]; \
            _Pragma("unroll") for (uint rb = 0; rb < (nrb); rb++) { \
                xa[rb] = xd[rb][16 * b + 8 * hf + sn]; \
                xb[rb] = xd[rb][16 * b + 8 * hf + sn + 1]; \
            } \
            _Pragma("unroll") for (uint j = 0; j < 4; j++) { \
                float2 bf[2]; \
                _Pragma("unroll") for (uint cb = 0; cb < 2; cb++) \
                    bf[cb] = float2(float(int(extract_bits(wv[2 * cb], 8 * j + 4 * hf, 4)) - 8) * sv[2 * cb], \
                                    float(int(extract_bits(wv[2 * cb + 1], 8 * j + 4 * hf, 4)) - 8) * sv[2 * cb + 1]); \
                _Pragma("unroll") for (uint rb = 0; rb < (nrb); rb++) { \
                    simdgroup_float8x8 a; \
                    reinterpret_cast<thread float2 &>(a.thread_elements()) = float2(xa[rb][j], xb[rb][j]); \
                    _Pragma("unroll") for (uint cb = 0; cb < 2; cb++) { \
                        simdgroup_float8x8 bm, cm; \
                        reinterpret_cast<thread float2 &>(bm.thread_elements()) = bf[cb]; \
                        reinterpret_cast<thread float2 &>(cm.thread_elements()) = acc[2 * rb + cb]; \
                        simdgroup_multiply_accumulate(cm, a, bm, cm); \
                        acc[2 * rb + cb] = reinterpret_cast<thread float2 &>(cm.thread_elements()); \
                    } \
                } \
            } \
        } \
    } \
    }

#define Q36_EXPERT_MM_DIRECT_SWITCH(inputs, wrow, srow, xrowp) \
    switch (nrb) { \
    case 1: Q36_EXPERT_MM_DIRECT(inputs, wrow, srow, 1u, xrowp) break; \
    case 2: Q36_EXPERT_MM_DIRECT(inputs, wrow, srow, 2u, xrowp) break; \
    case 3: Q36_EXPERT_MM_DIRECT(inputs, wrow, srow, 3u, xrowp) break; \
    default: Q36_EXPERT_MM_DIRECT(inputs, wrow, srow, 4u, xrowp) break; \
    }

#define Q36_EXPERT_MM_SWITCH(inputs, wrow, srow) \
    switch (nrb) { \
    case 5: Q36_EXPERT_MM_LOOP(inputs, wrow, srow, 5u) break; \
    case 6: Q36_EXPERT_MM_LOOP(inputs, wrow, srow, 6u) break; \
    case 7: Q36_EXPERT_MM_LOOP(inputs, wrow, srow, 7u) break; \
    default: Q36_EXPERT_MM_LOOP(inputs, wrow, srow, 8u) break; \
    }

// h[choice][o] = silu(gate_o . x) * (up_o . x): a threadgroup takes 32 outputs, SIMD group sg
// outputs 8 sg .. 8 sg + 7 (column block 0 gate, 1 up). Grid (choices / 64, I / 32, 8).
kernel void q36_expert_gate_up_mm(device const float *x [[buffer(0)]],
                                  device const uint *pairs [[buffer(1)]],
                                  device float *h [[buffer(2)]],
                                  constant q36_expert_args &p [[buffer(3)]],
                                  Q36_SLOTS,
                                  constant q36_expert_ranges &rg [[buffer(12)]],
                                  threadgroup float *shared [[threadgroup(0)]],
                                  uint3 tg [[threadgroup_position_in_grid]],
                                  ushort tid [[thread_index_in_threadgroup]],
                                  ushort sg [[simdgroup_index_in_threadgroup]],
                                  ushort lane [[thread_index_in_simdgroup]]) {
    const uint j0 = tg.z, c0 = tg.x * 64;
    if (j0 >= 8 || c0 >= rg.cnt[j0]) return;
    device const uchar *ws[8] = {w0, w1, w2, w3, w4, w5, w6, w7};
    device const uint *wq = (device const uint *)ws[j0];
    device const float *sq = (device const float *)(ws[j0] + p.scales);
    const ulong n = ulong(p.H) * p.I;
    const uint nc = min(64u, rg.cnt[j0] - c0), nrb = (nc + 7) / 8, o0 = tg.y * 32 + 8 * sg;
    if (nrb <= 4) return;                       // q36_expert_gate_up_mm4's
    const ushort2 pl = q36_lane_place(lane);
    const ushort sn = pl.x, sm = pl.y;
    device const uint *pc = pairs + p.first + rg.off[j0] + c0;
    const uint xrow = tid / 2, xc = 16 * (tid % 2);
    device const float4 *xr = (device const float4 *)(x + ulong((pc[min(xrow, nc - 1)] & 0x0fffffffu) / p.K) * p.H + xc);
    float2 acc[16];
    _Pragma("unroll") for (uint i = 0; i < 16; i++) acc[i] = float2(0.f);
#define Q36_GU_WROW(cb, t) ((cb) ? n / 8 : 0) + ulong(o0 + sn + (t)) * (p.H / 8) + sm
#define Q36_GU_SROW(cb, t) ((cb) ? n / 64 : 0) + ulong(o0 + sn + (t)) * (p.H / 64)
    Q36_EXPERT_MM_SWITCH(p.H, Q36_GU_WROW, Q36_GU_SROW)
#undef Q36_GU_WROW
#undef Q36_GU_SROW
    _Pragma("unroll") for (uint rb = 0; rb < 8; rb++) {
        uint row = 8 * rb + sm;
        if (row < nc) {
            float2 g = acc[2 * rb], u = acc[2 * rb + 1];
            *(device float2 *)(h + ulong(pc[row] & 0x0fffffffu) * p.I + o0 + sn) = g / (1.f + exp(-g)) * u;
        }
    }
}

// y[choice][o] = down_o . h[choice]: a threadgroup takes 64 outputs, SIMD group sg outputs
// 16 sg .. 16 sg + 15. Grid (choices / 64, H / 64, 8).
kernel void q36_expert_down_mm(device const float *hin [[buffer(0)]],
                               device const uint *pairs [[buffer(1)]],
                               device float *y [[buffer(2)]],
                               constant q36_expert_args &p [[buffer(3)]],
                               Q36_SLOTS,
                               constant q36_expert_ranges &rg [[buffer(12)]],
                               threadgroup float *shared [[threadgroup(0)]],
                               uint3 tg [[threadgroup_position_in_grid]],
                               ushort tid [[thread_index_in_threadgroup]],
                               ushort sg [[simdgroup_index_in_threadgroup]],
                               ushort lane [[thread_index_in_simdgroup]]) {
    const uint j0 = tg.z, c0 = tg.x * 64;
    if (j0 >= 8 || c0 >= rg.cnt[j0]) return;
    device const uchar *ws[8] = {w0, w1, w2, w3, w4, w5, w6, w7};
    device const uint *wq = (device const uint *)ws[j0];
    device const float *sq = (device const float *)(ws[j0] + p.scales);
    const ulong n = ulong(p.H) * p.I;
    const uint nc = min(64u, rg.cnt[j0] - c0), nrb = (nc + 7) / 8, o0 = tg.y * 64 + 16 * sg;
    if (nrb <= 4) return;                       // q36_expert_down_mm4's
    const ushort2 pl = q36_lane_place(lane);
    const ushort sn = pl.x, sm = pl.y;
    device const uint *pc = pairs + p.first + rg.off[j0] + c0;
    const uint xrow = tid / 2, xc = 16 * (tid % 2);
    device const float4 *xr = (device const float4 *)(hin + ulong(pc[min(xrow, nc - 1)] & 0x0fffffffu) * p.I + xc);
    float2 acc[16];
    _Pragma("unroll") for (uint i = 0; i < 16; i++) acc[i] = float2(0.f);
#define Q36_DN_WROW(cb, t) n / 4 + ulong(o0 + 8 * (cb) + sn + (t)) * (p.I / 8) + sm
#define Q36_DN_SROW(cb, t) 2 * n / 64 + ulong(o0 + 8 * (cb) + sn + (t)) * (p.I / 64)
    Q36_EXPERT_MM_SWITCH(p.I, Q36_DN_WROW, Q36_DN_SROW)
#undef Q36_DN_WROW
#undef Q36_DN_SROW
    _Pragma("unroll") for (uint cb = 0; cb < 2; cb++)
        _Pragma("unroll") for (uint rb = 0; rb < 8; rb++) {
            uint row = 8 * rb + sm;
            if (row < nc) *(device float2 *)(y + ulong(pc[row] & 0x0fffffffu) * p.H + o0 + 8 * cb + sn) = acc[2 * rb + cb];
        }
}

// The last tile of each slot when it has 1 to 4 row blocks, with the inputs read straight from
// memory (Q36_EXPERT_MM_DIRECT): the same threadgroups and outputs as q36_expert_gate_up_mm.
// Grid (1, I / 32, 8).
kernel void q36_expert_gate_up_mm4(device const float *x [[buffer(0)]],
                                   device const uint *pairs [[buffer(1)]],
                                   device float *h [[buffer(2)]],
                                   constant q36_expert_args &p [[buffer(3)]],
                                   Q36_SLOTS,
                                   constant q36_expert_ranges &rg [[buffer(12)]],
                                   uint3 tg [[threadgroup_position_in_grid]],
                                   ushort sg [[simdgroup_index_in_threadgroup]],
                                   ushort lane [[thread_index_in_simdgroup]]) {
    const uint j0 = tg.z;
    if (j0 >= 8 || !rg.cnt[j0]) return;
    const uint c0 = (rg.cnt[j0] - 1) / 64 * 64, nc = rg.cnt[j0] - c0, nrb = (nc + 7) / 8, o0 = tg.y * 32 + 8 * sg;
    if (nrb > 4) return;
    device const uchar *ws[8] = {w0, w1, w2, w3, w4, w5, w6, w7};
    device const uint *wq = (device const uint *)ws[j0];
    device const float *sq = (device const float *)(ws[j0] + p.scales);
    const ulong n = ulong(p.H) * p.I;
    const ushort2 pl = q36_lane_place(lane);
    const ushort sn = pl.x, sm = pl.y;
    device const uint *pc = pairs + p.first + rg.off[j0] + c0;
    float2 acc[8];
    _Pragma("unroll") for (uint i = 0; i < 8; i++) acc[i] = float2(0.f);
#define Q36_GU_WROW(cb, t) ((cb) ? n / 8 : 0) + ulong(o0 + sn + (t)) * (p.H / 8) + sm
#define Q36_GU_SROW(cb, t) ((cb) ? n / 64 : 0) + ulong(o0 + sn + (t)) * (p.H / 64)
#define Q36_GU_XROW(r) ((device const float4 *)(x + ulong((pc[r] & 0x0fffffffu) / p.K) * p.H))
    Q36_EXPERT_MM_DIRECT_SWITCH(p.H, Q36_GU_WROW, Q36_GU_SROW, Q36_GU_XROW)
#undef Q36_GU_WROW
#undef Q36_GU_SROW
#undef Q36_GU_XROW
    _Pragma("unroll") for (uint rb = 0; rb < 4; rb++) {
        uint row = 8 * rb + sm;
        if (row < nc) {
            float2 g = acc[2 * rb], u = acc[2 * rb + 1];
            *(device float2 *)(h + ulong(pc[row] & 0x0fffffffu) * p.I + o0 + sn) = g / (1.f + exp(-g)) * u;
        }
    }
}

// The same for the down projection, as q36_expert_down_mm. Grid (1, H / 64, 8).
kernel void q36_expert_down_mm4(device const float *hin [[buffer(0)]],
                                device const uint *pairs [[buffer(1)]],
                                device float *y [[buffer(2)]],
                                constant q36_expert_args &p [[buffer(3)]],
                                Q36_SLOTS,
                                constant q36_expert_ranges &rg [[buffer(12)]],
                                uint3 tg [[threadgroup_position_in_grid]],
                                ushort sg [[simdgroup_index_in_threadgroup]],
                                ushort lane [[thread_index_in_simdgroup]]) {
    const uint j0 = tg.z;
    if (j0 >= 8 || !rg.cnt[j0]) return;
    const uint c0 = (rg.cnt[j0] - 1) / 64 * 64, nc = rg.cnt[j0] - c0, nrb = (nc + 7) / 8, o0 = tg.y * 64 + 16 * sg;
    if (nrb > 4) return;
    device const uchar *ws[8] = {w0, w1, w2, w3, w4, w5, w6, w7};
    device const uint *wq = (device const uint *)ws[j0];
    device const float *sq = (device const float *)(ws[j0] + p.scales);
    const ulong n = ulong(p.H) * p.I;
    const ushort2 pl = q36_lane_place(lane);
    const ushort sn = pl.x, sm = pl.y;
    device const uint *pc = pairs + p.first + rg.off[j0] + c0;
    float2 acc[8];
    _Pragma("unroll") for (uint i = 0; i < 8; i++) acc[i] = float2(0.f);
#define Q36_DN_WROW(cb, t) n / 4 + ulong(o0 + 8 * (cb) + sn + (t)) * (p.I / 8) + sm
#define Q36_DN_SROW(cb, t) 2 * n / 64 + ulong(o0 + 8 * (cb) + sn + (t)) * (p.I / 64)
#define Q36_DN_XROW(r) ((device const float4 *)(hin + ulong(pc[r] & 0x0fffffffu) * p.I))
    Q36_EXPERT_MM_DIRECT_SWITCH(p.I, Q36_DN_WROW, Q36_DN_SROW, Q36_DN_XROW)
#undef Q36_DN_WROW
#undef Q36_DN_SROW
#undef Q36_DN_XROW
    _Pragma("unroll") for (uint cb = 0; cb < 2; cb++)
        _Pragma("unroll") for (uint rb = 0; rb < 4; rb++) {
            uint row = 8 * rb + sm;
            if (row < nc) *(device float2 *)(y + ulong(pc[row] & 0x0fffffffu) * p.H + o0 + 8 * cb + sn) = acc[2 * rb + cb];
        }
}

// Fast single-row GQA: eight query heads are the eight matrix rows. All
// operands and matrix accumulators are float32. QK, softmax and PV change
// reduction order; the exact wrapper path never dispatches these kernels.
// Scratch: probabilities [qh][cap], then partial PV [parts][qh][hd].
#define Q36_GQA_PART 256

// Four SIMD groups cover 8 heads x 64 cache positions, two 8x8 matrices
// each. The 32-column input panels use the same blocked layout as prefill.
// Grid (ceil(n/64), kv), 128 threads, (64+8)*32 floats of shared memory.
template <typename KV>
kernel void q36_attn_gqa_qk(device const float *q [[buffer(0)]],
                            device const KV *kc [[buffer(1)]],
                            device float *s [[buffer(2)]],
                            constant q36_state_args &p [[buffer(3)]],
                            threadgroup float *shared [[threadgroup(0)]],
                            uint2 tg [[threadgroup_position_in_grid]],
                            ushort tid [[thread_index_in_threadgroup]],
                            ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint kh = tg.y, t0 = tg.x * 64, n = p.pos + 1;
    if (kh >= p.kv || t0 >= n) return;
    threadgroup float *sk = shared, *sq = shared + 64 * 32;
    const uint kt = tid / 2, kd = 16 * (tid % 2);
    simdgroup_float8x8 acc0 = make_filled_simdgroup_matrix<float, 8>(0.f);
    simdgroup_float8x8 acc1 = make_filled_simdgroup_matrix<float, 8>(0.f);
    for (uint k = 0; k < p.hd; k += 32) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {   // the position's 16 dimensions in four loads of four
            device const vec<KV, 4> *src = (device const vec<KV, 4> *)(kc + (ulong(kh) * p.cap + t0 + kt) * p.hd + k + kd);
            for (uint i4 = 0; i4 < 4; i4++) {
                float4 v = t0 + kt < n ? float4(src[i4]) : float4(0.f);
                for (uint c = 0; c < 4; c++) {
                    uint d = kd + 4 * i4 + c;
                    sk[64 * (8 * (d / 8) + kt / 8) + 8 * (d % 8) + kt % 8] = v[c];
                }
            }
        }
        for (uint e = tid; e < 8 * 32; e += 128) {
            uint h = e / 32, d = e % 32;
            sq[64 * (d / 8) + 8 * h + d % 8] = q[ulong(kh * 8 + h) * p.hd + k + d];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kb = 0; kb < 4; kb++) {
            simdgroup_float8x8 a, b0, b1;
            simdgroup_load(a, sq + kb * 64, 8);
            simdgroup_load(b0, sk + kb * 8 * 64 + sg * 2 * 64, 8);
            simdgroup_load(b1, sk + kb * 8 * 64 + (sg * 2 + 1) * 64, 8);
            simdgroup_multiply_accumulate(acc0, a, b0, acc0);
            simdgroup_multiply_accumulate(acc1, a, b1, acc1);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_store(acc0, shared + sg * 16, 64);
    simdgroup_store(acc1, shared + sg * 16 + 8, 64);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float scale = 1.f / sqrt(float(p.hd));
    for (uint e = tid; e < 8 * 64; e += 128) {
        uint h = e / 64, t = e % 64;
        if (t0 + t < n) s[ulong(kh * 8 + h) * p.cap + t0 + t] = shared[e] * scale;
    }
}
template [[host_name("q36_attn_gqa_qk")]] kernel void q36_attn_gqa_qk<float>(device const float *, device const float *,
        device float *, constant q36_state_args &, threadgroup float *, uint2, ushort, ushort);
template [[host_name("q36_attn_gqa_qk_half")]] kernel void q36_attn_gqa_qk<half>(device const float *, device const half *,
        device float *, constant q36_state_args &, threadgroup float *, uint2, ushort, ushort);

// One full threadgroup per head; the two reductions use distinct shared
// arrays so a sum store cannot race another thread's read of the maxima.
// Grid qh, 256 threads. Probabilities overwrite the scores in place.
kernel void q36_attn_gqa_softmax(device float *s [[buffer(0)]],
                                 constant q36_state_args &p [[buffer(1)]],
                                 uint h [[threadgroup_position_in_grid]],
                                 ushort tid [[thread_index_in_threadgroup]],
                                 ushort sg [[simdgroup_index_in_threadgroup]],
                                 ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float maxima[8], sums[8];
    if (h >= p.qh) return;
    const uint n = p.pos + 1;
    device float *row = s + ulong(h) * p.cap;
    float mx = -INFINITY;
    for (uint t = tid; t < n; t += 256) mx = max(mx, row[t]);
    mx = simd_max(mx);
    if (lane == 0) maxima[sg] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0; i < 8; i++) mx = max(mx, maxima[i]);
    float sum = 0;
    for (uint t = tid; t < n; t += 256) {
        float e = exp(row[t] - mx);
        row[t] = e;
        sum += e;
    }
    sum = simd_sum(sum);
    if (lane == 0) sums[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum = 0;
    for (uint i = 0; i < 8; i++) sum += sums[i];
    const float inv = 1.f / sum;
    for (uint t = tid; t < n; t += 256) row[t] *= inv;
}

// Partial PV for 8 heads x 64 output columns over at most 256 positions.
// Each KV panel is loaded once into shared memory for all eight heads.
// Grid (hd/64, ceil(n/256), kv), 128 threads, (64+8)*32 shared floats.
template <typename KV>
kernel void q36_attn_gqa_pv(device float *s [[buffer(0)]],
                            device const KV *vc [[buffer(1)]],
                            constant q36_state_args &p [[buffer(2)]],
                            threadgroup float *shared [[threadgroup(0)]],
                            uint3 tg [[threadgroup_position_in_grid]],
                            ushort tid [[thread_index_in_threadgroup]],
                            ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint kh = tg.z, d0 = tg.x * 64, t0 = tg.y * Q36_GQA_PART, n = p.pos + 1;
    if (kh >= p.kv || d0 >= p.hd || t0 >= n) return;
    threadgroup float *sv = shared, *sp = shared + 64 * 32;
    const uint vt = tid / 4, vd = 16 * (tid % 4), end = min(n, t0 + Q36_GQA_PART);
    simdgroup_float8x8 acc0 = make_filled_simdgroup_matrix<float, 8>(0.f);
    simdgroup_float8x8 acc1 = make_filled_simdgroup_matrix<float, 8>(0.f);
    for (uint k = t0; k < end; k += 32) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {   // the position's 16 columns in four loads of four
            device const vec<KV, 4> *src = (device const vec<KV, 4> *)(vc + (ulong(kh) * p.cap + k + vt) * p.hd + d0 + vd);
            for (uint i4 = 0; i4 < 4; i4++) {
                float4 v = k + vt < end ? float4(src[i4]) : float4(0.f);
                for (uint c = 0; c < 4; c++) {
                    uint d = vd + 4 * i4 + c;
                    sv[64 * (8 * (vt / 8) + d / 8) + 8 * (vt % 8) + d % 8] = v[c];
                }
            }
        }
        for (uint e = tid; e < 8 * 32; e += 128) {
            uint h = e / 32, t = e % 32;
            sp[64 * (t / 8) + 8 * h + t % 8] =
                k + t < end ? s[ulong(kh * 8 + h) * p.cap + k + t] : 0.f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kb = 0; kb < 4; kb++) {
            simdgroup_float8x8 a, b0, b1;
            simdgroup_load(a, sp + kb * 64, 8);
            simdgroup_load(b0, sv + kb * 8 * 64 + sg * 2 * 64, 8);
            simdgroup_load(b1, sv + kb * 8 * 64 + (sg * 2 + 1) * 64, 8);
            simdgroup_multiply_accumulate(acc0, a, b0, acc0);
            simdgroup_multiply_accumulate(acc1, a, b1, acc1);
        }
    }
    device float *dst = s + ulong(p.qh) * p.cap +
        (ulong(tg.y) * p.qh + kh * 8) * p.hd + d0 + sg * 16;
    simdgroup_store(acc0, dst, p.hd);
    simdgroup_store(acc1, dst + 8, p.hd);
}
template [[host_name("q36_attn_gqa_pv")]] kernel void q36_attn_gqa_pv<float>(device float *, device const float *,
        constant q36_state_args &, threadgroup float *, uint3, ushort, ushort);
template [[host_name("q36_attn_gqa_pv_half")]] kernel void q36_attn_gqa_pv<half>(device float *, device const half *,
        constant q36_state_args &, threadgroup float *, uint3, ushort, ushort);

// All partials contain globally normalized probabilities; simply add them.
kernel void q36_attn_gqa_reduce(device const float *s [[buffer(0)]],
                                device float *out [[buffer(1)]],
                                constant q36_state_args &p [[buffer(2)]],
                                uint i [[thread_position_in_grid]]) {
    const uint width = p.qh * p.hd, parts = (p.pos + Q36_GQA_PART) / Q36_GQA_PART;
    if (i >= width) return;
    device const float *part = s + ulong(p.qh) * p.cap + i;
    float sum = 0;
    for (uint k = 0; k < parts; k++) sum += part[ulong(k) * width];
    out[i] = sum;
}
