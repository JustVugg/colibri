/* cuda_inkc.cuh -- Inkling's and Inkling-2's ops on the CUDA chain (cuda_chain.h's sconv, relattn,
 * ink2_mix, ink2_clip and ink2_dconv), included once by cuda_chain.cu after cuda_glm53c.cuh, whose
 * grid helper it uses. Each is the Vulkan chain's op of the same name (shaders/chain_sconv.comp,
 * chain_relattn.comp, chain_ink2.comp) with the same fields in the same order; vk_chain.h has
 * their contracts:
 *   sconv      mode 0 the depthwise causal short convolution with its ring (residual inside),
 *              mode 1 x *= fc, mode 2 x /= fc over n floats
 *   relattn    grouped attention with a relative-position bias bank, a per-row tau, a sliding
 *              window over a ring cache, the step's own rows from kvs, and with has_sink a sink
 *              logit per (row, head) in tau at sink_off that starts the online softmax
 *   ink2_mix   the attention residuals' mix of RMS-normalised block states and the stream
 *   ink2_clip  the L2 clip of each segment after an optional per-element scale
 *   ink2_dconv a global layer's dynamic short convolution with its state */

__global__ static void ccg_sconv_k(float *x, const float *w, float *ring, CcSconv p) {
    const long c = (long)(blockIdx.x + blockIdx.y * gridDim.x) * 64 + threadIdx.x;
    if (p.mode != 0) {
        if (c >= p.n) return;
        const float v = x[p.x_off + c];
        x[p.x_off + c] = p.mode == 1 ? v * p.fc : v / p.fc;
        return;
    }
    if (c >= p.C) return;
    const int nh = p.CK - 1;
    float hist[8];
#pragma unroll
    for (int k = 0; k < 8; k++) hist[k] = k < nh ? ring[p.ring_off + c * nh + k] : 0.f;
    const float *wc = w + p.w_off + c * p.CK;
    const float wl = wc[p.CK - 1];
    for (int s = 0; s < p.S; s++) {
        const long xi = p.x_off + (long)s * p.x_row + c;
        const float cur = x[xi];
        float acc = 0.f;
#pragma unroll
        for (int k = 0; k < 8; k++) if (k < nh) acc += wc[k] * hist[k];
        acc += wl * cur;
        x[xi] = acc + cur;
#pragma unroll
        for (int k = 0; k < 7; k++) if (k + 1 < nh) hist[k] = hist[k + 1];
#pragma unroll
        for (int k = 0; k < 8; k++) if (k == nh - 1) hist[k] = cur;
    }
#pragma unroll
    for (int k = 0; k < 8; k++) if (k < nh) ring[p.ring_off + c * nh + k] = hist[k];
}

/* one block of 128 per (head, row): chain_relattn.comp's online softmax over tiles of 128 positions */
__global__ static void ccg_relattn_k(const float *q, const float *kc, const float *vc, float *o, const float *kvs,
                                     const float *r, const float *relp, const float *tau, CcRelAttn p) {
    const int h = (int)blockIdx.x, s = (int)blockIdx.y, tid = (int)threadIdx.x;
    const int kvh = h / (p.H / p.KVH);
    __shared__ float qs[256], rs[64], pe[128], red[128];
    __shared__ int pt[128];
    const int qb = p.q_off + s * p.q_row + h * p.hd;
    for (int d = tid; d < p.hd; d += 128) qs[d] = q[qb + d];
    for (int j = tid; j < p.d_rel; j += 128) rs[j] = r[p.r_off + s * p.r_row + h * p.d_rel + j];
    const int qpos = p.pos_base + s;
    const int t0 = p.window > 0 ? max(0, qpos - p.window + 1) : 0;
    const int n = qpos - t0 + 1;
    const float ts = tau[p.tau_off + s];
    __syncthreads();
    float m = -3.0e38f, l = 0.f, acc0 = 0.f, acc1 = 0.f;
    if (p.has_sink) { m = tau[p.sink_off + s * p.H + h]; l = 1.f; }
    for (int b0 = 0; b0 < n; b0 += 128) {
        const int i = b0 + tid;
        float sv = -3.0e38f;
        if (i < n) {
            const int t = t0 + i;
            pt[tid] = t;
            float a = 0.f;
            if (t < p.pos_base) { const float *kr = kc + p.k_off + ((long)kvh * p.cap + t % p.cap) * p.hd; for (int d = 0; d < p.hd; d++) a += qs[d] * kr[d]; }
            else { const float *kr = kvs + p.ks_off + (long)(t - p.pos_base) * p.kv_row + kvh * p.hd; for (int d = 0; d < p.hd; d++) a += qs[d] * kr[d]; }
            const int dist = qpos - t;
            float bias = 0.f;
            if (dist < p.ext) for (int j = 0; j < p.d_rel; j++) bias += rs[j] * relp[p.relp_off + j * p.ext + dist];
            sv = ts * (a * p.scale + bias);
        }
        red[tid] = sv;
        __syncthreads();
        for (int off = 64; off > 0; off >>= 1) { if (tid < off) red[tid] = fmaxf(red[tid], red[tid + off]); __syncthreads(); }
        const float mn = fmaxf(m, red[0]);
        __syncthreads();
        const float e = i < n ? expf(sv - mn) : 0.f;
        pe[tid] = e; red[tid] = e;
        __syncthreads();
        for (int off = 64; off > 0; off >>= 1) { if (tid < off) red[tid] += red[tid + off]; __syncthreads(); }
        const float corr = expf(m - mn);
        l = l * corr + red[0];
        acc0 *= corr; acc1 *= corr;
        const int cnt = min(128, n - b0), d0 = tid, d1 = tid + 128;
        for (int j = 0; j < cnt; j++) {
            const int t = pt[j];
            const float w = pe[j];
            const float *vr = t < p.pos_base ? vc + p.v_off + ((long)kvh * p.cap + t % p.cap) * p.hd
                                             : kvs + p.vs_off + (long)(t - p.pos_base) * p.kv_row + kvh * p.hd;
            if (d0 < p.hd) acc0 += w * vr[d0];
            if (d1 < p.hd) acc1 += w * vr[d1];
        }
        m = mn;
        __syncthreads();
    }
    const float inv = l > 0.f ? 1.f / l : 0.f;
    const int ob = p.o_off + s * p.o_row + h * p.hd;
    if (tid < p.hd) o[ob + tid] = acc0 * inv;
    if (tid + 128 < p.hd) o[ob + tid + 128] = acc1 * inv;
}

/* one block of 256 per row: every vector's sums in one pass, the warps' joined in order */
__global__ static void ccg_ink2_mix_k(const float *x, const float *blk, const float *prm, float *y, CcInk2Mix p) {
    const int s = (int)(blockIdx.x + blockIdx.y * gridDim.x), tid = (int)threadIdx.x, wp = tid >> 5, ln = tid & 31;
    if (s >= p.S) return;
    __shared__ float pms[8 * 16], pdt[8 * 16], sc[16], rr[16];
    const float *xb = x + p.x_off + (long)s * p.x_row, *bb = blk ? blk + p.b_off + (long)s * p.b_row : xb;
    const float *q = prm + p.w_off;
    float ms[16], dt[16];
#pragma unroll
    for (int e = 0; e < 16; e++) { ms[e] = 0.f; dt[e] = 0.f; }
    for (int d = tid; d < p.D; d += 256) {
        const float qv = q[d];
#pragma unroll
        for (int e = 0; e < 16; e++) {
            if (e > p.nb) break;
            const float v = e < p.nb ? bb[(long)e * p.D + d] : xb[d];
            ms[e] += v * v; dt[e] += v * qv;
        }
    }
#pragma unroll
    for (int e = 0; e < 16; e++) {
        if (e > p.nb) break;
        float a = ms[e], b = dt[e];
        for (int off = 16; off > 0; off >>= 1) { a += __shfl_xor_sync(0xffffffffu, a, off); b += __shfl_xor_sync(0xffffffffu, b, off); }
        if (ln == 0) { pms[wp * 16 + e] = a; pdt[wp * 16 + e] = b; }
    }
    __syncthreads();
    if (tid <= p.nb) {
        float a = 0.f, b = 0.f;
        for (int g = 0; g < 8; g++) { a += pms[g * 16 + tid]; b += pdt[g * 16 + tid]; }
        const float r = 1.f / sqrtf(a / (float)p.D + p.eps);
        rr[tid] = r; sc[tid] = (b * r) * p.scale;
    }
    __syncthreads();
    if (tid == 0) {
        float m = sc[0];
        for (int e = 1; e <= p.nb; e++) if (sc[e] > m) m = sc[e];
        float sum = 0.f;
        for (int e = 0; e <= p.nb; e++) { sc[e] = expf(sc[e] - m); sum += sc[e]; }
        for (int e = 0; e <= p.nb; e++) sc[e] /= sum;
    }
    __syncthreads();
    float *yb = y + p.y_off + (long)s * p.y_row;
    for (int d = tid; d < p.D; d += 256) {
        float a = 0.f;
        for (int e = 0; e < p.nb; e++) a += sc[e] * (bb[(long)e * p.D + d] * rr[e]);
        yb[d] = a + sc[p.nb] * (xb[d] * rr[p.nb]);
    }
}

/* one block of 256 per segment */
__global__ static void ccg_ink2_clip_k(float *x, const float *w, CcInk2Clip p) {
    const int g = (int)(blockIdx.x + blockIdx.y * gridDim.x), tid = (int)threadIdx.x;
    if (g >= p.nseg) return;
    __shared__ float red[256];
    const int j = g % p.per_row;
    float *v = x + p.x_off + (long)(g / p.per_row) * p.x_row + (long)j * p.n;
    const float *sw = p.has_w ? w + p.w_off + (long)j * p.n : nullptr;
    float ss = 0.f;
    for (int i = tid; i < p.n; i += 256) { float a = v[i]; if (sw) a *= sw[i]; ss += a * a; }
    red[tid] = ss;
    __syncthreads();
    for (int off = 128; off > 0; off >>= 1) { if (tid < off) red[tid] += red[tid + off]; __syncthreads(); }
    const float f = p.thr / (sqrtf(red[0]) + 1e-6f);
    for (int i = tid; i < p.n; i += 256) { float a = v[i]; if (sw) a *= sw[i]; v[i] = f < 1.f ? a * f : a; }
}

/* one thread per channel over the step's rows */
__global__ static void ccg_ink2_dconv_k(float *x, const float *dyn, const float *st, float *ring, CcInk2Dconv p) {
    const long c = (long)(blockIdx.x + blockIdx.y * gridDim.x) * 256 + threadIdx.x;
    if (c >= p.C) return;
    const int K = p.K, nh = K - 1, h = (int)(c / p.hd);
    float hist[8];
#pragma unroll
    for (int k = 0; k < 8; k++) hist[k] = k < nh ? ring[p.ring_off + c * nh + k] : 0.f;
    for (int s = 0; s < p.S; s++) {
        const long xi = p.x_off + (long)s * p.x_row + c;
        const float cur = x[xi];
        const float *dw = dyn + p.d_off + (long)s * p.d_row + h * K;
        float acc = cur * (st[p.st_off + c] + dw[0]);
#pragma unroll
        for (int lag = 1; lag < 9; lag++) if (lag < K) acc += hist[nh - lag] * (st[p.st_off + (long)lag * p.C + c] + dw[lag]);
        x[xi] = (acc + cur) / p.fc;
#pragma unroll
        for (int k = 0; k < 7; k++) if (k + 1 < nh) hist[k] = hist[k + 1];
#pragma unroll
        for (int k = 0; k < 8; k++) if (k == nh - 1) hist[k] = cur;
    }
#pragma unroll
    for (int k = 0; k < 8; k++) if (k < nh) ring[p.ring_off + c * nh + k] = hist[k];
}

static int cc_sconv(CcBuf *x, CcBuf *w, CcBuf *ring, const CcSconv *p) {
    CcCtx *c = live();
    if (!c || !p || !x || (p->mode == 0 && (p->CK < 1 || p->CK > 9 || !w || !ring))) return 0;
    const long n = p->mode == 0 ? p->C : p->n;
    if (n <= 0) return 1;
    dim3 g; cg_grid((n + 63) / 64, &g);
    coli_launch(ccg_sconv_k, g, 64, 0, c->stream, fpw(x), p->mode == 0 ? fp(w) : fp(x), p->mode == 0 ? fpw(ring) : fpw(x), *p);
    return launched(c, "sconv");
}
static int cc_relattn(CcBuf *q, CcBuf *kc, CcBuf *vc, CcBuf *o, CcBuf *kvs, CcBuf *r, CcBuf *relp, CcBuf *tau, const CcRelAttn *p) {
    CcCtx *c = live();
    if (!c || !p || p->hd > 256 || p->hd < 1 || p->d_rel > 64 || p->d_rel < 0 || p->KVH < 1 || p->H % p->KVH || p->cap < 1) return 0;
    if (p->S <= 0) return 1;
    coli_launch(ccg_relattn_k, dim3((unsigned)p->H, (unsigned)p->S), 128, 0, c->stream, fp(q), fp(kc), fp(vc), fpw(o), fp(kvs),
                fp(r), fp(relp), fp(tau), *p);
    return launched(c, "relattn");
}
static int cc_ink2_mix(CcBuf *x, CcBuf *blk, CcBuf *prm, CcBuf *y, const CcInk2Mix *p) {
    CcCtx *c = live();
    if (!c || !p || !x || !prm || !y || p->S < 0 || p->D < 1 || p->nb < 0 || p->nb > 15 || (p->nb > 0 && !blk) || y == x || y == blk) return 0;
    if (p->S == 0) return 1;
    dim3 g; cg_grid(p->S, &g);
    coli_launch(ccg_ink2_mix_k, g, 256, 0, c->stream, fp(x), p->nb > 0 ? fp(blk) : nullptr, fp(prm), fpw(y), *p);
    return launched(c, "ink2 mix");
}
static int cc_ink2_clip(CcBuf *x, CcBuf *prm, const CcInk2Clip *p) {
    CcCtx *c = live();
    if (!c || !p || !x || p->nseg < 0 || p->n < 1 || p->per_row < 1 || (p->has_w && !prm)) return 0;
    if (p->nseg == 0) return 1;
    dim3 g; cg_grid(p->nseg, &g);
    coli_launch(ccg_ink2_clip_k, g, 256, 0, c->stream, fpw(x), p->has_w ? fp(prm) : nullptr, *p);
    return launched(c, "ink2 clip");
}
static int cc_ink2_dconv(CcBuf *x, CcBuf *dyn, CcBuf *prm, CcBuf *ring, const CcInk2Dconv *p) {
    CcCtx *c = live();
    if (!c || !p || !x || !dyn || !prm || !ring || p->S < 0 || p->C < 1 || p->K < 1 || p->K > 9 || p->hd < 1 || p->C % p->hd) return 0;
    if (p->S == 0) return 1;
    dim3 g; cg_grid(((long)p->C + 255) / 256, &g);
    coli_launch(ccg_ink2_dconv_k, g, 256, 0, c->stream, fpw(x), fp(dyn), fp(prm), fpw(ring), *p);
    return launched(c, "ink2 dconv");
}
