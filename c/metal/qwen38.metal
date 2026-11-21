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

#if Q38_COUNTER_TEST == 5   /* counter-test 5: bf16 expands with a half-ulp bias */
static inline float q38_bf16(ushort w) { return as_type<float>((uint(w) << 16) | 0x8000u); }
#else
static inline float q38_bf16(ushort w) { return as_type<float>(uint(w) << 16); }
#endif

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
#if Q38_COUNTER_TEST == 1   /* counter-test 1: hc_mix divides by the wrong fc */
    y[i] = v / (p.fc + 1.0f);
#else
    y[i] = v / p.fc;
#endif
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
#if Q38_COUNTER_TEST == 2   /* counter-test 2: the ring shifts one entry short */
        for (uint j = 0; j + 2 < SL; j++) rg[j] = rg[j + 1];
#else
        for (uint j = 0; j + 1 < SL; j++) rg[j] = rg[j + 1];
#endif
        rg[SL - 1] = nv;
        if (s < p.snap_rows) {
            device float *sd = snap + ((ulong)s * p.W + d) * SL;
            for (uint j = 0; j < SL; j++) sd[j] = rg[j];
        }
    }
    for (uint j = 0; j < SL; j++) rd[j] = rg[j];
}

/* ---- QSA: block keys, scores, selection, sparse attention ----------------- */
/* qwen38_core.h's q38_attention (the indexer at 3482-3533, attention at 3543-3590) and
 * shaders/chain_qsa.comp / chain_attn.comp. rope is q38_rope: the first `rotary` dims, pairs
 * (i, i + half), angle pos / base[i] with base[i] = theta^(2i/rotary) powf'd on the host (the
 * CPU's bits). The division is rounded correctly (a reciprocal and one fma correction), as
 * the CPU's float division is: one ulp of an angle near 9000 is 1e-3 radians. */
struct q38_rope_table { float base[128], inv[128]; };

static inline float q38_rope_angle(float pos, float base, float inv) {
    float q0 = pos * inv;
    float r = fma(-q0, base, pos);
    return fma(r, inv, q0);
}

/* Dim t of rope(v, pos), v a whole vector in threadgroup memory: t < half pairs with t + half. */
static inline float q38_rope_tg(threadgroup const float *v, uint t, uint half_, float pos,
                                constant q38_rope_table &rt) {
    if (t < half_) {
        float ang = q38_rope_angle(pos, rt.base[t], rt.inv[t]);
        return v[t] * cos(ang) - v[t + half_] * sin(ang);
    }
    if (t < 2 * half_) {
        uint i = t - half_;
        float ang = q38_rope_angle(pos, rt.base[i], rt.inv[i]);
        return v[t] * cos(ang) + v[i] * sin(ang);
    }
    return v[t];
}

/* pk[b] = rope(rms0(mean_{r<R} ik[b*R+r], kn), b*R), b = b0 + group. One 256-thread group a
 * block, thread d < ID one dim; the mean sums raw/R in r order from 0 as the CPU does. */
struct q38_pool_args { uint ID, R, b0, half_; float eps; };

kernel void q38_qsa_pool(device const float *ik [[buffer(0)]],
                         device const float *kn [[buffer(1)]],
                         device float *pk [[buffer(2)]],
                         constant q38_pool_args &p [[buffer(3)]],
                         constant q38_rope_table &rt [[buffer(4)]],
                         uint g [[threadgroup_position_in_grid]],
                         uint t [[thread_index_in_threadgroup]]) {
    threadgroup float pool[256], part[256], sum;
    uint b = p.b0 + g;
    float v = 0.0f;
    if (t < p.ID) {
        for (uint r = 0; r < p.R; r++) v += ik[((ulong)b * p.R + r) * p.ID + t] / (float)p.R;
    }
    float rr = 1.0f / sqrt(q38_tg_sum256(part, &sum, v * v, t) / (float)p.ID + p.eps);
    if (t < p.ID) pool[t] = v * rr * (1.0f + kn[t]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t >= p.ID) return;
    pk[(ulong)b * p.ID + t] = q38_rope_tg(pool, t, p.half_, (float)(b * p.R), rt);
}

/* sc[s*nbmax + b] = sum_h max(0, q[s][h] . pk[b]) / sqrt(ID) for b < nblk(s): one thread a
 * (b, s), each dot in d order (both modes). */
struct q38_score_args { uint ID, R, S, pos_base, IQ, nbmax; };

kernel void q38_qsa_score(device const float *q [[buffer(0)]],
                          device const float *pk [[buffer(1)]],
                          device float *sc [[buffer(2)]],
                          constant q38_score_args &p [[buffer(3)]],
                          uint2 id [[thread_position_in_grid]]) {
    uint b = id.x, s = id.y;
    if (s >= p.S || b >= (p.pos_base + s + 1) / p.R) return;
    device const float *qs = q + (ulong)s * p.IQ * p.ID, *kb = pk + (ulong)b * p.ID;
    float score = 0.0f;
    for (uint h = 0; h < p.IQ; h++) {
        float a = 0.0f;
        for (uint d = 0; d < p.ID; d++) a = fma(qs[h * p.ID + d], kb[d], a);
        if (a > 0.0f) score += a;
    }
    sc[(ulong)s * p.nbmax + b] = score / sqrt((float)p.ID);
}

/* The selection, one 256-thread group a row s at pos = pos_base + s: visible = pos + 1,
 * blocks = visible / R, take = min(blocks, budget / R). take == blocks: sel = [-1] (every
 * position). Otherwise chain_qsa.comp's radix select (a byte a pass from the top over the
 * score bits, which order like the scores: they are >= 0) finds the last block taken's score
 * and how many blocks with that score are taken (the lowest ones); the taken blocks are then
 * sorted as q38_block_desc orders them (score desc, block asc) by a bitonic sort of the keys
 * bits << 32 | ~block, so sel = [count, the blocks' R positions in rank order, the tail]. */
#define Q38_SEL_MAX 2048
struct q38_select_args { uint R, S, pos_base, budget, nbmax, sel_row; };

/* The inclusive prefix sum of v over the 256 threads, in thread order. Uniform control flow. */
static inline uint q38_scan256(threadgroup uint *sums, uint v, uint t) {
    sums[t] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint off = 1; off < 256; off <<= 1) {
        uint a = t >= off ? sums[t - off] : 0u;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sums[t] += a;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint r = sums[t];
    threadgroup_barrier(mem_flags::mem_threadgroup);   /* sums is reused by the next scan */
    return r;
}

#if Q38_COUNTER_TEST == 3   /* counter-test 3: equal scores rank the higher block first */
#define Q38_TIE_KEY(b) (b)
#else
#define Q38_TIE_KEY(b) (0xffffffffu - (b))
#endif
kernel void q38_qsa_select(device const float *sc [[buffer(0)]],
                           device int *sel [[buffer(1)]],
                           constant q38_select_args &p [[buffer(2)]],
                           uint g [[threadgroup_position_in_grid]],
                           uint t [[thread_index_in_threadgroup]]) {
    threadgroup atomic_uint hist[256];
    threadgroup uint sums[256];
    threadgroup uint picked_digit, picked_take;
    threadgroup ulong keys[Q38_SEL_MAX];
    uint visible = p.pos_base + g + 1, blocks = visible / p.R, take = min(blocks, p.budget / p.R);
    device int *row = sel + (ulong)g * p.sel_row;
    device const uint *bits = (device const uint *)(sc + (ulong)g * p.nbmax);
    if (take >= blocks) {
        if (t == 0) row[0] = -1;
        return;
    }
    uint cutoff = 0u, mask = 0u, to_take = take;
    if (take > 0) {
        for (uint pass = 0; pass < 4; pass++) {
            uint shift = (3u - pass) * 8u;
            atomic_store_explicit(&hist[t], 0u, memory_order_relaxed);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint b = t; b < blocks; b += 256) {
                uint u = bits[b];
                if ((u & mask) == cutoff) atomic_fetch_add_explicit(&hist[(u >> shift) & 255u], 1u, memory_order_relaxed);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint bin = 255u - t, count = atomic_load_explicit(&hist[bin], memory_order_relaxed);
            uint above = q38_scan256(sums, count, t) - count;   /* blocks in the bins above this one */
            if (above < to_take && to_take <= above + count) {
                picked_digit = bin;
                picked_take = to_take - above;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            cutoff |= picked_digit << shift;
            mask |= 255u << shift;
            to_take = picked_take;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    /* Thread t's contiguous run of blocks: the ties taken are the lowest blocks. */
    uint per = (blocks + 255u) / 256u, first = t * per, end = min(blocks, first + per);
    uint higher = 0u, ties = 0u;
    if (take > 0) {
        for (uint b = first; b < end; b++) {
            uint u = bits[b];
            higher += u > cutoff ? 1u : 0u;
            ties += u == cutoff ? 1u : 0u;
        }
    }
    uint ties_before = q38_scan256(sums, ties, t) - ties;
    uint ties_taken = to_take > ties_before ? min(ties, to_take - ties_before) : 0u;
    uint slot = q38_scan256(sums, higher + ties_taken, t) - higher - ties_taken;
    if (take > 0) {
        for (uint b = first; b < end; b++) {
            uint u = bits[b];
            bool tie = u == cutoff && ties_taken > 0u;
            if (tie) ties_taken--;
            if (u > cutoff || tie) keys[slot++] = (ulong(u) << 32) | ulong(Q38_TIE_KEY(b));
        }
    }
    uint P = 1u;
    while (P < take) P <<= 1;
    for (uint i = take + t; i < P; i += 256) keys[i] = 0ul;   /* below every real key */
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = 2; k <= P; k <<= 1) {
        for (uint j = k >> 1; j > 0; j >>= 1) {
            for (uint i = t; i < P; i += 256) {
                uint l = i ^ j;
                if (l > i) {
                    ulong a = keys[i], c = keys[l];
                    bool down = (i & k) == 0u;   /* the whole list ends descending */
                    if (down ? a < c : a > c) { keys[i] = c; keys[l] = a; }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    for (uint i = t; i < take * p.R; i += 256) {
        uint blk = Q38_TIE_KEY(uint(keys[i / p.R] & 0xfffffffful));
        row[1 + i] = int(blk * p.R + i % p.R);
    }
    uint tail = visible - blocks * p.R;
    for (uint i = t; i < tail; i += 256) row[1 + take * p.R + i] = int(blocks * p.R + i);
    if (t == 0) row[0] = int(take * p.R + tail);
}

/* Attention over a row's list, one 256-thread group a (head h, row s), KV head h / (heads /
 * kv_heads): thread i scores the tile's position i (q . k in d order, / sqrt(D) as the CPU),
 * an online softmax over tiles of 256 (fixed-order max and sum), thread d < D accumulates
 * p_j v_j[d] in list order; out = acc / l * sigmoid(gate). Count -1: positions 0..pos. */
struct q38_attn_args { uint S, pos_base, heads, kv_heads, D, cap, sel_row; };

static inline float q38_tg_max256(threadgroup float *part, threadgroup float *out, float v, uint t) {
    part[t] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
        float m = part[0];
        for (uint l = 1; l < 256; l++) m = max(m, part[l]);
        *out = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float m = *out;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return m;
}

template <typename KT>
static inline void q38_attn_sel_body(device const float *q, device const float *gate, device const KT *kc,
                                     device const KT *vc, device const int *sel, device float *out,
                                     constant q38_attn_args &p, uint h, uint s, uint t,
                                     threadgroup float *qs, threadgroup float *pe, threadgroup uint *pt,
                                     threadgroup float *part, threadgroup float *red) {
    uint kvh = h / (p.heads / p.kv_heads), D = p.D;
    ulong qo = ((ulong)s * p.heads + h) * D;
    if (t < D) qs[t] = q[qo + t];
    device const int *row = sel + (ulong)s * p.sel_row;
    int c = row[0];
    bool list = c >= 0;
    uint n = list ? min(uint(c), p.sel_row - 1) : p.pos_base + s + 1;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ulong kbase = (ulong)kvh * p.cap;
    float m = -INFINITY, l = 0.0f, acc = 0.0f, sd = sqrt((float)D);
    for (uint t0 = 0; t0 < n; t0 += 256) {
        uint i = t0 + t;
        float sv = -INFINITY;
        if (i < n) {
            uint pos = list ? uint(row[1 + i]) : i;
            pt[t] = pos;
            device const KT *kr = kc + (kbase + pos) * D;
            float a = 0.0f;
            for (uint d = 0; d < D; d++) a = fma(qs[d], float(kr[d]), a);
            sv = a / sd;
        }
        float mn = max(m, q38_tg_max256(part, red, sv, t));
        float e = i < n ? exp(sv - mn) : 0.0f;
        pe[t] = e;
        float ts = q38_tg_sum256(part, red, e, t);   /* its barriers publish pe and pt */
        float corr = exp(m - mn);
        l = l * corr + ts;
        acc *= corr;
        uint cnt = min(256u, n - t0);
        if (t < D) {
            for (uint j = 0; j < cnt; j++) acc = fma(pe[j], float(vc[(kbase + pt[j]) * D + t]), acc);
        }
        m = mn;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (t < D) out[qo + t] = (l > 0.0f ? acc / l : 0.0f) * q38_sigmoid(gate[qo + t]);
}

kernel void q38_attn_sel(device const float *q [[buffer(0)]],
                         device const float *gate [[buffer(1)]],
                         device const float *kc [[buffer(2)]],
                         device const float *vc [[buffer(3)]],
                         device const int *sel [[buffer(4)]],
                         device float *out [[buffer(5)]],
                         constant q38_attn_args &p [[buffer(6)]],
                         uint2 g [[threadgroup_position_in_grid]],
                         uint t [[thread_index_in_threadgroup]]) {
    threadgroup float qs[256], pe[256], part[256], red;
    threadgroup uint pt[256];
    q38_attn_sel_body<float>(q, gate, kc, vc, sel, out, p, g.x, g.y, t, qs, pe, pt, part, &red);
}

kernel void q38_attn_sel_half(device const float *q [[buffer(0)]],
                              device const float *gate [[buffer(1)]],
                              device const half *kc [[buffer(2)]],
                              device const half *vc [[buffer(3)]],
                              device const int *sel [[buffer(4)]],
                              device float *out [[buffer(5)]],
                              constant q38_attn_args &p [[buffer(6)]],
                              uint2 g [[threadgroup_position_in_grid]],
                              uint t [[thread_index_in_threadgroup]]) {
    threadgroup float qs[256], pe[256], part[256], red;
    threadgroup uint pt[256];
    q38_attn_sel_body<half>(q, gate, kc, vc, sel, out, p, g.x, g.y, t, qs, pe, pt, part, &red);
}

/* ---- attention glue (q38_attention 3464-3504, 3568-3592) ------------------ */
/* Element-wise, and idx_prep's one fixed-order rms0 sum: exact and fast are the same kernels. */

/* x [rows][heads][D] in place, one thread a rotated pair (row, head, i < half): q38_rope at
 * pos_base + row. */
struct q38_rope_args { uint n, heads, D, half_, pos_base; };

kernel void q38_rope_half(device float *x [[buffer(0)]],
                          constant q38_rope_args &p [[buffer(1)]],
                          constant q38_rope_table &rt [[buffer(2)]],
                          uint id [[thread_position_in_grid]]) {
    if (id >= p.n) return;
    uint i = id % p.half_, rh = id / p.half_, r = rh / p.heads;
    device float *v = x + (ulong)rh * p.D;
    float ang = q38_rope_angle((float)(p.pos_base + r), rt.base[i], rt.inv[i]);
    float co = cos(ang), si = sin(ang), a = v[i], b = v[i + p.half_];
    v[i] = a * co - b * si;
    v[i + p.half_] = b * co + a * si;
}

/* qraw [rows*heads][q D | gate D] -> q, gate [rows*heads][D], one thread an element. */
struct q38_split_args { uint n, D; };

kernel void q38_qgate_split(device const float *qraw [[buffer(0)]],
                            device float *q [[buffer(1)]],
                            device float *gate [[buffer(2)]],
                            constant q38_split_args &p [[buffer(3)]],
                            uint id [[thread_position_in_grid]]) {
    if (id >= p.n) return;
    ulong rh = id / p.D, d = id % p.D, o = rh * 2 * p.D + d;
    q[id] = qraw[o];
    gate[id] = qraw[o + p.D];
}

/* k, v [rows][kv_heads][D] -> caches [kv_heads][cap][D] at pos_base + row, one thread an
 * element. The half form is kv16_round: clamped to +-65504, then the conversion, which rounds
 * to nearest even. */
static inline float q38_kv16_clamp(float x) { return x > 65504.0f ? 65504.0f : x < -65504.0f ? -65504.0f : x; }   /* NaN stays */

struct q38_kva_args { uint n, kv_heads, D, cap, pos_base; };

kernel void q38_kv_append(device float *kc [[buffer(0)]],
                          device float *vc [[buffer(1)]],
                          device const float *k [[buffer(2)]],
                          device const float *v [[buffer(3)]],
                          constant q38_kva_args &p [[buffer(4)]],
                          uint id [[thread_position_in_grid]]) {
    if (id >= p.n) return;
    uint d = id % p.D, rh = id / p.D, h = rh % p.kv_heads, r = rh / p.kv_heads;
    ulong c = ((ulong)h * p.cap + p.pos_base + r) * p.D + d;
    kc[c] = k[id];
    vc[c] = v[id];
}

kernel void q38_kv_append_half(device half *kc [[buffer(0)]],
                               device half *vc [[buffer(1)]],
                               device const float *k [[buffer(2)]],
                               device const float *v [[buffer(3)]],
                               constant q38_kva_args &p [[buffer(4)]],
                               uint id [[thread_position_in_grid]]) {
    if (id >= p.n) return;
    uint d = id % p.D, rh = id / p.D, h = rh % p.kv_heads, r = rh / p.kv_heads;
    ulong c = ((ulong)h * p.cap + p.pos_base + r) * p.D + d;
    kc[c] = half(q38_kv16_clamp(k[id]));
    vc[c] = half(q38_kv16_clamp(v[id]));
}

/* One 256-thread group a (head h, row r) of ip [rows][(IQ+1)*ID]: h < IQ, q[r][h] =
 * rope(rms0(ip[r][h], qn), pos_base + r) (q38_qsa_pool's sum and rope); h == IQ copies the raw
 * key ip[r][IQ] to ikc[pos_base + r]. */
struct q38_idx_args { uint IQ, ID, half_, pos_base; float eps; };

kernel void q38_idx_prep(device const float *ip [[buffer(0)]],
                         device const float *qn [[buffer(1)]],
                         device float *q [[buffer(2)]],
                         device float *ikc [[buffer(3)]],
                         constant q38_idx_args &p [[buffer(4)]],
                         constant q38_rope_table &rt [[buffer(5)]],
                         uint2 g [[threadgroup_position_in_grid]],
                         uint t [[thread_index_in_threadgroup]]) {
    threadgroup float v[256], part[256], sum;
    uint h = g.x, r = g.y, pos = p.pos_base + r;
    device const float *src = ip + ((ulong)r * (p.IQ + 1) + h) * p.ID;
    if (h == p.IQ) {   /* uniform across the group */
        if (t < p.ID) ikc[(ulong)pos * p.ID + t] = src[t];
        return;
    }
    float x = t < p.ID ? src[t] : 0.0f;
    float rr = 1.0f / sqrt(q38_tg_sum256(part, &sum, x * x, t) / (float)p.ID + p.eps);
    if (t < p.ID) v[t] = x * rr * (1.0f + qn[t]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t >= p.ID) return;
    q[((ulong)r * p.IQ + h) * p.ID + t] = q38_rope_tg(v, t, p.half_, (float)pos, rt);
}
