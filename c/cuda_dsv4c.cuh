/* cuda_dsv4c.cuh -- DeepSeek V4.1 Flash's (and V4's) layer ops on the CUDA chain, included by
 * cuda_chain.cu (its CcCtx, live(), fp(), fpw(), grid1(), launched()). The CUDA twins of the
 * Vulkan chain's shaders/chain_mhc.comp and shaders/chain_dsv4.comp (with chain_dsv4_pa.comp
 * for a prompt chunk's attention), the same arithmetic in the same order, so the CPU's
 * reference (hyper_connections.h, sparse_attn.h, deepseek_v41.c) holds them within the same
 * bounds tests/test_vk_chain.c holds the shaders to:
 *   cc_mhc          the hyper-connections' split (Sinkhorn), collapse, write back, mean, and
 *                   the clamped SwiGLU (CcMhc, modes CC_MHC_*);
 *   cc_ds_attn      sparse attention with a sink over a per-row list: a decode row's (head,
 *                   row) a block, a prompt chunk's rows from CC_DS_PA_MIN in three passes
 *                   over slices of rows (scores on tiles of the listed rows, the weights and
 *                   the denominator in list order, the output on tiles), the same sums;
 *   cc_ds_rope      RoPE on interleaved pairs, inverse and bf16 flags;
 *   cc_ds_compress  the compressor's rolling group and its softmax pooling;
 *   cc_ds_score     the indexer's scores, a thread a (column, head) on a grid, the heads
 *                   joined in their order;
 *   cc_ds_cand      the candidate blocks;
 *   cc_ds_topk      the top-k of each row's scores, ties to the lower column;
 *   cc_ds_engram    the engram gate into each stream. */
#ifndef CUDA_DSV4C_CUH
#define CUDA_DSV4C_CUH

#define CC_DS_NEG_INF (-__int_as_float(0x7f800000))
#define CC_DS_POS_INF (__int_as_float(0x7f800000))

__device__ __forceinline__ float ccd_bf16r(float f) {   /* round to nearest even, as coli_bf16_round */
    unsigned u = __float_as_uint(f);
    if ((u & 0x7f800000u) != 0x7f800000u) u += 0x7fffu + ((u >> 16) & 1u);
    return __uint_as_float(u & 0xffff0000u);
}
__device__ __forceinline__ float ccd_sig(float v) {
    if (v >= 0.f) { float d = expf(-v); return 1.f / (1.f + d); }
    float g = expf(v);
    return g / (1.f + g);
}
__device__ __forceinline__ unsigned ccd_okey(float f) {
    unsigned u = __float_as_uint(f);
    if (u == 0x80000000u) u = 0u;
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
/* a block's sum of v (256 threads), every thread gets it */
__device__ __forceinline__ unsigned ccd_wsum(unsigned v, unsigned *cnt, int tid) {
    cnt[tid] = v;
    __syncthreads();
    for (int off = 128; off > 0; off >>= 1) {
        if (tid < off) cnt[tid] += cnt[tid + off];
        __syncthreads();
    }
    unsigned t = cnt[0];
    __syncthreads();
    return t;
}
/* exclusive scan in thread order */
__device__ __forceinline__ unsigned ccd_wscan(unsigned v, unsigned *cnt, int tid) {
    cnt[tid] = v;
    __syncthreads();
    for (int off = 1; off < 256; off <<= 1) {
        unsigned a = tid >= off ? cnt[tid - off] : 0u;
        __syncthreads();
        cnt[tid] += a;
        __syncthreads();
    }
    unsigned incl = cnt[tid];
    __syncthreads();
    return incl - v;
}

/* ---- mHC ------------------------------------------------------------------------------ */
__global__ void __launch_bounds__(256) ccd_mhc_split_k(const float *x, const float *mx, float *hp, const float *prm, CcMhc p) {
    __shared__ float red[256];
    __shared__ float ms[96];
    const int s = blockIdx.x, tid = threadIdx.x, H = p.H, HD = H * p.D, nm = (2 + H) * H;
    const int xb = p.x_off + s * p.x_row;
    float a = 0.f;
    for (int i = tid; i < HD; i += 256) { float v = x[xb + i]; a += v * v; }
    red[tid] = a;
    __syncthreads();
    for (int off = 128; off > 0; off >>= 1) {
        if (tid < off) red[tid] += red[tid + off];
        __syncthreads();
    }
    const float r = 1.f / sqrtf(red[0] / (float)HD + p.eps);
    for (int i = tid; i < nm; i += 256) ms[i] = mx[p.m_off + s * p.m_row + i] * r;
    __syncthreads();
    if (tid != 0) return;
    const int hb = p.hp_off + s * p.hp_row, pb = p.prm_off;
    for (int i = 0; i < H; i++) {
        hp[hb + i] = ccd_sig(ms[i] * prm[pb] + prm[pb + 3 + i]) + p.hc_eps;
        hp[hb + H + i] = 2.f * ccd_sig(ms[H + i] * prm[pb + 1] + prm[pb + 3 + H + i]);
    }
    const int cb = hb + 2 * H, mo = 2 * H;
    for (int row = 0; row < H; row++) {
        float mxv = -3.0e38f;
        for (int col = 0; col < H; col++) {
            float v = ms[mo + row * H + col] * prm[pb + 2] + prm[pb + 3 + mo + row * H + col];
            hp[cb + row * H + col] = v;
            if (v > mxv) mxv = v;
        }
        float sum = 0.f;
        for (int col = 0; col < H; col++) { float v = expf(hp[cb + row * H + col] - mxv); hp[cb + row * H + col] = v; sum += v; }
        for (int col = 0; col < H; col++) hp[cb + row * H + col] = hp[cb + row * H + col] / sum + p.hc_eps;
    }
    float sums[8];
    for (int col = 0; col < H; col++) { float sm = 0.f; for (int row = 0; row < H; row++) sm += hp[cb + row * H + col]; sums[col] = sm; }
    for (int row = 0; row < H; row++) for (int col = 0; col < H; col++) hp[cb + row * H + col] /= sums[col] + p.hc_eps;
    for (int it = 1; it < p.iters; it++) {
        for (int row = 0; row < H; row++) { float sm = 0.f; for (int col = 0; col < H; col++) sm += hp[cb + row * H + col]; sums[row] = sm; }
        for (int row = 0; row < H; row++) for (int col = 0; col < H; col++) hp[cb + row * H + col] /= sums[row] + p.hc_eps;
        for (int col = 0; col < H; col++) { float sm = 0.f; for (int row = 0; row < H; row++) sm += hp[cb + row * H + col]; sums[col] = sm; }
        for (int row = 0; row < H; row++) for (int col = 0; col < H; col++) hp[cb + row * H + col] /= sums[col] + p.hc_eps;
    }
}
__global__ void ccd_mhc_ew_k(int mode, const float *x, const float *mx, const float *hp, float *y, CcMhc p) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (mode == CC_SWIGLU_CLAMP) {
        if (i >= p.n) return;
        float g = x[p.x_off + i], u = mx[p.m_off + i];
        g = g > p.lim ? p.lim : g;
        u = u < -p.lim ? -p.lim : (u > p.lim ? p.lim : u);
        y[p.y_off + i] = (g / (1.f + expf(-g))) * u;
        return;
    }
    const int H = p.H, D = p.D;
    if (mode == CC_MHC_POST) {
        if (i >= (long long)p.S * H * D) return;
        const int s = (int)(i / (H * D)), r = (int)(i - (long long)s * H * D), j = r / D, d = r - j * D;
        const int xb = p.x_off + s * p.x_row, hb = p.hp_off + s * p.hp_row;
        float v = 0.f;
        for (int k = 0; k < H; k++) v += hp[hb + 2 * H + k * H + j] * x[xb + k * D + d];
        v += hp[hb + H + j] * mx[p.m_off + s * p.m_row + d];
        y[p.y_off + s * p.y_row + j * D + d] = v;
        return;
    }
    if (i >= (long long)p.S * D) return;
    const int s = (int)(i / D), d = (int)(i - (long long)s * D);
    const int xb = p.x_off + s * p.x_row, hb = p.hp_off + s * p.hp_row;
    float v = 0.f;
    if (mode == CC_MHC_COLLAPSE) { for (int k = 0; k < H; k++) v += hp[hb + k] * x[xb + k * D + d]; }
    else { for (int k = 0; k < H; k++) v += x[xb + k * D + d]; v = v / (float)H; }
    y[p.y_off + s * p.y_row + d] = v;
}
static int cc_mhc(int mode, CcBuf *x, CcBuf *m, CcBuf *hp, CcBuf *prm, CcBuf *y, const CcMhc *p) {
    CcCtx *c = live();
    if (!c || !p || !x) return 0;
    if (mode == CC_MHC_SPLIT) {
        if (!m || !hp || !prm || p->S < 1 || p->H < 1 || p->H > 8 || (2 + p->H) * p->H > 96) return 0;
        coli_launch(ccd_mhc_split_k, (unsigned)p->S, 256, 0, c->stream, fp(x), fp(m), fpw(hp), fp(prm), *p);
        return launched(c, "mhc split");
    }
    size_t n;
    if (mode == CC_SWIGLU_CLAMP) { if (!m || !y || p->n < 1) return 0; n = (size_t)p->n; }
    else if (mode == CC_MHC_POST) { if (!m || !hp || !y || p->S < 1) return 0; n = (size_t)p->S * p->H * p->D; }
    else if (mode == CC_MHC_COLLAPSE || mode == CC_MHC_MEAN) { if (!y || (mode == CC_MHC_COLLAPSE && !hp) || p->S < 1) return 0; n = (size_t)p->S * p->D; }
    else return 0;
    coli_launch(ccd_mhc_ew_k, grid1(n, 256), 256, 0, c->stream, mode, fp(x), fp(m), fp(hp), fpw(y), *p);
    return launched(c, "mhc");
}

/* ---- sparse attention ------------------------------------------------------------------- */
__device__ __forceinline__ float ccd_kv(const float *win, const float *cmp, int e, int nwin, int woff, int coff, int hd, int i) {
    return e < nwin ? win[woff + e * hd + i] : cmp[coff + (e - nwin) * hd + i];
}
/* a (head, row) a block: chain_dsv4.comp's mode 0 */
__global__ void __launch_bounds__(256) ccd_attn_k(const float *q, const float *win, const float *cmp, const int *list,
                                                  const float *prm, float *out, CcDsAttn p) {
    extern __shared__ float sm[];   /* hd queries, then cnt scores */
    __shared__ float red[256];
    const int h = blockIdx.x, s = blockIdx.y, tid = threadIdx.x;
    if (s >= p.S || h >= p.H) return;
    const int hd = p.hd, n = p.cnt, lb = p.l_off + s * p.l_row;
    const int qb = p.q_off + s * p.q_row + h * hd, ob = p.o_off + s * p.o_row + h * hd;
    float *sc = sm + hd;
    for (int i = tid; i < hd; i += 256) sm[i] = q[qb + i];
    __syncthreads();
    float best = -1.0e30f;
    for (int j = tid; j < n; j += 256) {
        const int e = list[lb + j];
        float v = CC_DS_NEG_INF;
        if (e >= 0) {
            float d = 0.f;
            if (e < p.nwin) { const int kb = p.w_off + e * hd; for (int i = 0; i < hd; i++) d += sm[i] * win[kb + i]; }
            else { const int kb = p.c_off + (e - p.nwin) * hd; for (int i = 0; i < hd; i++) d += sm[i] * cmp[kb + i]; }
            v = d * p.scale;
            if (v > best) best = v;
        }
        sc[j] = v;
    }
    red[tid] = best;
    __syncthreads();
    for (int off = 128; off > 0; off >>= 1) {
        if (tid < off && red[tid + off] > red[tid]) red[tid] = red[tid + off];
        __syncthreads();
    }
    best = red[0];
    __syncthreads();
    for (int j = tid; j < n; j += 256) { const float v = sc[j]; sc[j] = v > CC_DS_NEG_INF ? expf(v - best) : 0.f; }
    __syncthreads();
    if (tid == 0) {                       /* the denominator in list order, from the sink */
        float den = expf(prm[p.sink_off + h] - best);
        for (int j = 0; j < n; j++) if (list[lb + j] >= 0) den += sc[j];
        red[0] = den;
    }
    __syncthreads();
    const float den = red[0];
    for (int i = tid; i < hd; i += 256) {
        float acc = 0.f;
        for (int j = 0; j < n; j++) {
            const int e = list[lb + j];
            if (e < 0) continue;
            float w = sc[j];
            if (p.flags & 1) w = ccd_bf16r(w);
            acc += w * ccd_kv(win, cmp, e, p.nwin, p.w_off, p.c_off, hd, i);
        }
        const float o = acc / den;
        out[ob + i] = (p.flags & 1) ? ccd_bf16r(o) : o;
    }
}
/* a prompt chunk's rows (chain_dsv4_pa.comp): the scores of 64 heads a block on tiles of 64
 * listed rows, in chunks of 32 floats */
__global__ void __launch_bounds__(256) ccd_pa_scores_k(const float *q, const float *win, const float *cmp, const int *list,
                                                       float *scr, CcDsAttn p) {
    __shared__ float sq[64 * 32];
    __shared__ float sk[64 * 33];
    __shared__ int se[64];
    const int s = blockIdx.x, hg = blockIdx.y * 64, tid = threadIdx.x, H = p.H, hd = p.hd, n = p.cnt;
    const int lb = p.l_off + s * p.l_row, qb = p.q_off + s * p.q_row;
    const int hb = tid >> 4, jb = tid & 15;
    for (int t0 = 0; t0 < n; t0 += 64) {
        __syncthreads();
        if (tid < 64) se[tid] = t0 + tid < n ? list[lb + t0 + tid] : -1;
        float acc[4][4];
        for (int a = 0; a < 4; a++) for (int b = 0; b < 4; b++) acc[a][b] = 0.f;
        for (int c0 = 0; c0 < hd; c0 += 32) {
            __syncthreads();
            for (int k = tid; k < 64 * 32; k += 256) {
                const int hh = k >> 5, ii = k & 31, h = hg + hh;
                sq[k] = h < H ? q[qb + h * hd + c0 + ii] : 0.f;
                const int e = se[hh];
                sk[hh * 33 + ii] = e >= 0 ? ccd_kv(win, cmp, e, p.nwin, p.w_off, p.c_off, hd, c0 + ii) : 0.f;
            }
            __syncthreads();
            for (int ii = 0; ii < 32; ii++) {
                float kk[4];
                for (int b = 0; b < 4; b++) kk[b] = sk[(jb + 16 * b) * 33 + ii];
                for (int a = 0; a < 4; a++) {
                    const float qa = sq[(hb + 16 * a) * 32 + ii];
                    for (int b = 0; b < 4; b++) acc[a][b] += qa * kk[b];
                }
            }
        }
        for (int a = 0; a < 4; a++) {
            const int h = hg + hb + 16 * a;
            if (h >= H) continue;
            for (int b = 0; b < 4; b++) {
                const int j = t0 + jb + 16 * b;
                if (j >= n) continue;
                scr[((size_t)s * H + h) * n + j] = se[jb + 16 * b] >= 0 ? acc[a][b] * p.scale : CC_DS_NEG_INF;
            }
        }
    }
}
/* the weights in place and each (row, head)'s denominator in list order */
__global__ void __launch_bounds__(256) ccd_pa_weights_k(const int *list, const float *prm, float *scr, float *den, CcDsAttn p) {
    __shared__ float red[256];
    __shared__ float sw[3072];
    const int s = blockIdx.x, h = blockIdx.y, tid = threadIdx.x, H = p.H, n = p.cnt;
    const int lb = p.l_off + s * p.l_row;
    float *row = scr + ((size_t)s * H + h) * n;
    float best = -1.0e30f;
    for (int j = tid; j < n; j += 256) { const float v = row[j]; if (v > best) best = v; }
    red[tid] = best;
    __syncthreads();
    for (int off = 128; off > 0; off >>= 1) {
        if (tid < off && red[tid + off] > red[tid]) red[tid] = red[tid + off];
        __syncthreads();
    }
    best = red[0];
    __syncthreads();
    for (int j = tid; j < n; j += 256) { const float v = row[j], w = v > CC_DS_NEG_INF ? expf(v - best) : 0.f; row[j] = w; sw[j] = w; }
    __syncthreads();
    if (tid == 0) {
        float d = expf(prm[p.sink_off + h] - best);
        for (int j = 0; j < n; j++) if (list[lb + j] >= 0) d += sw[j];
        den[(size_t)s * H + h] = d;
    }
}
/* the output: 64 heads by 64 dims a block, the entries in list order on tiles of 32 */
__global__ void __launch_bounds__(256) ccd_pa_out_k(const float *win, const float *cmp, const int *list, const float *scr,
                                                    const float *den, float *out, CcDsAttn p) {
    __shared__ float sk[32 * 65];
    __shared__ float sw[64 * 33];
    __shared__ int se[32];
    const int s = blockIdx.x, d0 = blockIdx.y * 64, hg = blockIdx.z * 64, tid = threadIdx.x, H = p.H, hd = p.hd, n = p.cnt;
    const int lb = p.l_off + s * p.l_row;
    const int hb = tid >> 4, ib = tid & 15;
    float acc[4][4];
    for (int a = 0; a < 4; a++) for (int b = 0; b < 4; b++) acc[a][b] = 0.f;
    for (int t0 = 0; t0 < n; t0 += 32) {
        __syncthreads();
        if (tid < 32) se[tid] = t0 + tid < n ? list[lb + t0 + tid] : -1;
        __syncthreads();
        for (int k = tid; k < 32 * 64; k += 256) {
            const int jj = k >> 6, ii = k & 63, e = se[jj];
            sk[jj * 65 + ii] = e >= 0 ? ccd_kv(win, cmp, e, p.nwin, p.w_off, p.c_off, hd, d0 + ii) : 0.f;
            const int hh = k >> 5, j2 = k & 31, h = hg + hh;
            const float w = h < H && t0 + j2 < n ? scr[((size_t)s * H + h) * n + t0 + j2] : 0.f;
            sw[hh * 33 + j2] = (p.flags & 1) ? ccd_bf16r(w) : w;
        }
        __syncthreads();
        const int m = n - t0 < 32 ? n - t0 : 32;
        for (int jj = 0; jj < m; jj++) {
            if (se[jj] < 0) continue;
            float kk[4];
            for (int b = 0; b < 4; b++) kk[b] = sk[jj * 65 + ib + 16 * b];
            for (int a = 0; a < 4; a++) {
                const float w = sw[(hb + 16 * a) * 33 + jj];
                for (int b = 0; b < 4; b++) acc[a][b] += w * kk[b];
            }
        }
    }
    for (int a = 0; a < 4; a++) {
        const int h = hg + hb + 16 * a;
        if (h >= H) continue;
        const float dn = den[(size_t)s * H + h];
        const int ob = p.o_off + s * p.o_row + h * hd + d0;
        for (int b = 0; b < 4; b++) {
            const float o = acc[a][b] / dn;
            out[ob + ib + 16 * b] = (p.flags & 1) ? ccd_bf16r(o) : o;
        }
    }
}
#define CC_DS_PA_MIN 8
#define CC_DS_PA_MB 64
static CcBuf *g_ccd_scr, *g_ccd_den;
static int cc_ds_attn(CcBuf *q, CcBuf *win, CcBuf *cmp, CcBuf *list, CcBuf *prm, CcBuf *out, const CcDsAttn *p) {
    CcCtx *c = live();
    if (!c || !p || !q || !win || !list || !prm || !out || p->S < 1 || p->H < 1 || p->hd < 1 || p->hd > 1024 ||
        p->cnt < 1 || p->cnt > 3072) return 0;
    const float *cm = fp(cmp ? cmp : win);
    const int *ls = reinterpret_cast<const int *>(fp(list));
    const char *e = getenv("COLI_CUDA_DSV4_PA");   /* 0: a block a (head, row) for every row */
    if (p->S >= CC_DS_PA_MIN && p->hd % 64 == 0 && !(e && *e == '0')) {
        const size_t per = (size_t)p->H * p->cnt * sizeof(float);
        int R = (int)(((size_t)CC_DS_PA_MB << 20) / per);
        if (R < 1) R = 1;
        if (R > p->S) R = p->S;
        if (!cc_reserve(&g_ccd_scr, (size_t)R * per, CC_DEV) || !cc_reserve(&g_ccd_den, (size_t)R * p->H * sizeof(float), CC_DEV))
            return 0;
        const unsigned hgs = (unsigned)((p->H + 63) / 64);
        for (int r0 = 0; r0 < p->S; r0 += R) {
            CcDsAttn x = *p;
            x.S = p->S - r0 < R ? p->S - r0 : R;
            x.l_off += r0 * p->l_row; x.q_off += r0 * p->q_row; x.o_off += r0 * p->o_row;
            coli_launch(ccd_pa_scores_k, dim3((unsigned)x.S, hgs), 256, 0, c->stream, fp(q), fp(win), cm, ls, fpw(g_ccd_scr), x);
            coli_launch(ccd_pa_weights_k, dim3((unsigned)x.S, (unsigned)p->H), 256, 0, c->stream, ls, fp(prm), fpw(g_ccd_scr), fpw(g_ccd_den), x);
            coli_launch(ccd_pa_out_k, dim3((unsigned)x.S, (unsigned)(p->hd / 64), hgs), 256, 0, c->stream, fp(win), cm, ls,
                        fp(g_ccd_scr), fp(g_ccd_den), fpw(out), x);
        }
        return launched(c, "ds attn (prompt)");
    }
    const size_t smem = ((size_t)p->hd + p->cnt) * sizeof(float);
    if (smem > 48 * 1024) return 0;
    coli_launch(ccd_attn_k, dim3((unsigned)p->H, (unsigned)p->S), 256, smem, c->stream, fp(q), fp(win), cm, ls, fp(prm), fpw(out), *p);
    return launched(c, "ds attn");
}

/* ---- RoPE on interleaved pairs ---------------------------------------------------------- */
__global__ void ccd_rope_k(float *x, const float *cs, CcDsRope p) {
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const int half_ = p.rd / 2;
    if (t >= (long long)p.nseg * half_) return;
    const int g = (int)(t / half_), i = (int)(t - (long long)g * half_);
    const int row = g / p.per_row, j = g - row * p.per_row;
    const long long xb = (long long)p.x_off + (long long)row * p.x_row + (long long)j * p.x_seg + 2 * i;
    const long long cb = (long long)p.cs_off + (long long)row * p.cs_row + 2 * i;
    const float c = cs[cb];
    float sn = cs[cb + 1];
    if (p.inverse) sn = -sn;
    const float a = x[xb], b = x[xb + 1];
    float x0 = a * c - b * sn, x1 = a * sn + b * c;
    if (p.flags & 1) { x0 = ccd_bf16r(x0); x1 = ccd_bf16r(x1); }
    x[xb] = x0; x[xb + 1] = x1;
}
static int cc_ds_rope(CcBuf *x, CcBuf *cs, const CcDsRope *p) {
    CcCtx *c = live();
    if (!c || !p || !x || !cs || p->nseg < 0 || p->per_row < 1 || p->rd < 2 || (p->rd & 1)) return 0;
    if (p->nseg == 0) return 1;
    coli_launch(ccd_rope_k, grid1((size_t)p->nseg * (p->rd / 2), 256), 256, 0, c->stream, fpw(x), fp(cs), *p);
    return launched(c, "ds rope");
}

/* ---- the compressor's rolling group (one block, the rows in order) --------------------- */
__global__ void __launch_bounds__(256) ccd_compress_k(const float *kv, const float *sc, float *ring, const float *ape, float *out,
                                                      CcDsComp p) {
    const int tid = threadIdx.x, R = p.ratio, P = p.P, D = p.D, ov = p.overlap;
    const int rows = ov ? 2 * R : R, ro = p.ring_off, so = ro + rows * P;
    for (int s = 0; s < p.S; s++) {
        const int pos = p.pos_base + s, slot = pos % R, srow = ov ? R + slot : slot;
        for (int c = tid; c < P; c += 256) {
            ring[ro + srow * P + c] = kv[p.kv_off + s * p.kv_row + c];
            float v = sc[p.sc_off + s * p.sc_row + c];
            if (p.ape_off >= 0) v = v + ape[p.ape_off + slot * P + c];
            ring[so + srow * P + c] = v;
        }
        __syncthreads();
        if ((pos + 1) % R != 0) continue;
        const int g = pos / R;
        for (int d = tid; d < D; d += 256) {
            float best = CC_DS_NEG_INF;
            for (int k = 0; k < rows; k++) {
                const int col = (ov && k >= R) ? D + d : d;
                const float v = ring[so + k * P + col];
                if (v > best) best = v;
            }
            float tot = 0.f, mix = 0.f;
            for (int k = 0; k < rows; k++) {
                const int col = (ov && k >= R) ? D + d : d;
                const float w = expf(ring[so + k * P + col] - best);
                tot += w;
                mix += w * ring[ro + k * P + col];
            }
            out[p.out_off + g * p.out_row + d] = mix / tot;
        }
        __syncthreads();
        if (ov) {
            for (int c = tid; c < R * P; c += 256) {
                ring[ro + c] = ring[ro + R * P + c];
                ring[so + c] = ring[so + R * P + c];
            }
            __syncthreads();
        }
    }
}
static int cc_ds_compress(CcBuf *kv, CcBuf *sc, CcBuf *ring, CcBuf *prm, CcBuf *out, const CcDsComp *p) {
    CcCtx *c = live();
    if (!c || !p || p->S < 0 || p->ratio < 1 || p->P < 1 || p->D < 1 || p->D > p->P || (p->overlap && p->P < 2 * p->D)) return 0;
    if (p->S == 0) return 1;
    if (!kv || !sc || !ring || !out || (p->ape_off >= 0 && !prm)) return 0;
    coli_launch(ccd_compress_k, 1, 256, 0, c->stream, fp(kv), fp(sc), fpw(ring), fp(prm), fpw(out), *p);
    return launched(c, "ds compress");
}

/* ---- the indexer's scores: a thread a (column, head), the heads joined in their order --- */
__global__ void __launch_bounds__(256) ccd_score_k(const float *iq, const float *hw, const float *keys, const unsigned char *mask, float *scs,
                                                   CcDsScore p) {
    extern __shared__ float sm[];   /* the queries (IH*ID when they fit), then IH weights */
    __shared__ float red[256];
    const int s = blockIdx.y, tid = threadIdx.x, IH = p.IH, ID = p.ID, width = p.width;
    const int pos = p.pos_base + s, lens = (pos + 1) / p.ratio;
    const int qb = p.q_off + s * p.q_row, wb = p.w_off + s * p.w_row, sb = p.sc_off + s * p.sc_row, mb = p.mask_off + s * p.mask_row;
    const int P = 256 / IH > 0 ? 256 / IH : 1, j0 = blockIdx.x * P;
    if (j0 >= width) return;
    const int sh = IH <= 64 && IH * ID <= 4096, wo = sh ? IH * ID : 0;
    if (sh) for (int i = tid; i < IH * ID; i += 256) sm[i] = iq[qb + i];
    for (int h = tid; h < IH; h += 256) sm[wo + h] = hw[wb + h] * p.wscale;
    __syncthreads();
    const int jj = tid / IH, h = tid - jj * IH, j = j0 + jj;
    if (jj < P && j < width && j < lens && (p.mask_row <= 0 || mask[mb + j] != 0)) {
        const int kb = p.k_off + j * p.k_row;
        float d = 0.f;
        if (sh) for (int i = 0; i < ID; i++) d += sm[h * ID + i] * keys[kb + i];
        else for (int i = 0; i < ID; i++) d += iq[qb + h * ID + i] * keys[kb + i];
        red[tid] = d;
    }
    __syncthreads();
    const int jt = j0 + tid;
    if (tid < P && jt < width) {
        float v = CC_DS_NEG_INF;
        if (jt < lens && (p.mask_row <= 0 || mask[mb + jt] != 0)) {
            float tot = 0.f;
            for (int g = 0; g < IH; g++) { const float d = red[tid * IH + g]; if (d > 0.f) tot += d * sm[wo + g]; }
            v = tot;
        }
        scs[sb + jt] = v;
    }
}
static int cc_ds_score(CcBuf *iq, CcBuf *hw, CcBuf *keys, CcBuf *mask, CcBuf *sc, const CcDsScore *p) {
    CcCtx *c = live();
    if (!c || !p || p->S < 1 || p->IH < 1 || p->IH > 256 || p->ID < 1 || p->ratio < 1 || p->width < 0 ||
        p->sc_row < p->width || (p->mask_row > 0 && !mask) || !iq || !hw || !keys || !sc) return 0;
    if (p->mask_row > 0 && (size_t)p->mask_off + (size_t)(p->S - 1) * p->mask_row + (size_t)p->width > mask->bytes) return 0;   /* bytes */
    if (p->width == 0) return 1;
    const int P = 256 / p->IH, sh = p->IH <= 64 && p->IH * p->ID <= 4096;
    const size_t smem = ((sh ? (size_t)p->IH * p->ID : 0) + (size_t)p->IH) * sizeof(float);
    if (smem > 48 * 1024) return 0;
    coli_launch(ccd_score_k, dim3((unsigned)((p->width + P - 1) / P), (unsigned)p->S), 256, smem, c->stream, fp(iq), fp(hw), fp(keys),
                reinterpret_cast<const unsigned char *>(fp(mask)), fpw(sc), *p);
    return launched(c, "ds score");
}

/* ---- the candidate blocks ------------------------------------------------------------------ */
__global__ void __launch_bounds__(256) ccd_cand_k(const float *scs, unsigned char *mask, CcDsCand p) {
    __shared__ float bm[4096];
    __shared__ float red[256];
    __shared__ unsigned cnt[256];
    const int s = blockIdx.x, tid = threadIdx.x;
    const int pos = p.pos_base + s, lens = (pos + 1) / p.ratio;
    const int width = p.width, blk = p.block, topb = p.topb, sb = s * p.sc_row, mb = p.mask_off + s * p.mask_row;
    const int blocks = (width + blk - 1) / blk;
    for (int b = tid; b < blocks; b += 256) {
        float top = CC_DS_NEG_INF;
        for (int i = b * blk; i < (b + 1) * blk && i < width; i++) { const float v = scs[sb + i]; if (v > top) top = v; }
        bm[b] = top;
    }
    for (int i = tid; i < width; i += 256) mask[mb + i] = 0;
    __syncthreads();
    const int last = lens >= 1 ? (lens - 1) / blk : (blk == 1 ? -1 : 0);   /* C's (lens - 1) / block */
    if (tid == 0 && last >= 0 && last < blocks) bm[last] = CC_DS_POS_INF;
    __syncthreads();
    const int wanted = topb < blocks ? topb : blocks;
    for (int pick = 0; pick < wanted; pick++) {   /* the best block above -inf, the lower on a tie */
        float bv = CC_DS_NEG_INF; int bi = -1;
        for (int b = tid; b < blocks; b += 256) {
            const float v = bm[b];
            if (v > CC_DS_NEG_INF && (bi < 0 || v > bv)) { bv = v; bi = b; }
        }
        red[tid] = bv; cnt[tid] = (unsigned)(bi < 0 ? 0x7fffffff : bi);
        __syncthreads();
        for (int off = 128; off > 0; off >>= 1) {
            if (tid < off) {
                const float ov = red[tid + off]; const unsigned oi = cnt[tid + off];
                if (oi != 0x7fffffffu && (cnt[tid] == 0x7fffffffu || ov > red[tid] || (ov == red[tid] && oi < cnt[tid]))) {
                    red[tid] = ov; cnt[tid] = oi;
                }
            }
            __syncthreads();
        }
        const unsigned chosen = cnt[0];
        __syncthreads();
        if (chosen == 0x7fffffffu) break;
        const int cb = (int)chosen;
        for (int i = cb * blk + tid; i < (cb + 1) * blk && i < width; i += 256) mask[mb + i] = 1;
        if (tid == 0) bm[cb] = CC_DS_NEG_INF;
        __syncthreads();
    }
}
static int cc_ds_cand(CcBuf *sc, CcBuf *mask, const CcDsCand *p) {
    CcCtx *c = live();
    if (!c || !p || !sc || !mask || p->S < 1 || p->block < 1 || p->ratio < 1 || p->width < 0 ||
        (p->width + p->block - 1) / p->block > 4096 || p->mask_row < p->width || p->mask_off < 0 ||
        (size_t)p->mask_off + (size_t)(p->S - 1) * p->mask_row + (size_t)p->width > mask->bytes) return 0;
    coli_launch(ccd_cand_k, (unsigned)p->S, 256, 0, c->stream, fp(sc), reinterpret_cast<unsigned char *>(fpw(mask)), *p);
    return launched(c, "ds cand");
}

/* ---- the top-k: the keep-th largest key bit by bit, then the columns in order ---------- */
__global__ void __launch_bounds__(256) ccd_topk_k(const float *scs, int *list, CcDsTopk p) {
    __shared__ unsigned cnt[256];
    extern __shared__ float rank[];   /* order 1: the selected columns */
    const int s = blockIdx.x, tid = threadIdx.x;
    const int width = p.width, topk = p.topk, sb = s * p.sc_row, lb = p.l_off + s * p.l_row;
    unsigned nv = 0u;
    for (int j = tid; j < width; j += 256) if (scs[sb + j] > CC_DS_NEG_INF) nv++;
    const int valid = (int)ccd_wsum(nv, cnt, tid);
    const int keep = topk < valid ? topk : valid;
    for (int k = keep + tid; k < topk; k += 256) list[lb + k] = -1;
    if (keep == 0) return;
    unsigned K = 0u;
    for (int bit = 31; bit >= 0; bit--) {
        const unsigned cand = K | (1u << (unsigned)bit);
        unsigned cn = 0u;
        for (int j = tid; j < width; j += 256) { const float v = scs[sb + j]; if (v > CC_DS_NEG_INF && ccd_okey(v) >= cand) cn++; }
        if (ccd_wsum(cn, cnt, tid) >= (unsigned)keep) K = cand;
    }
    unsigned ab = 0u;
    for (int j = tid; j < width; j += 256) { const float v = scs[sb + j]; if (v > CC_DS_NEG_INF && ccd_okey(v) > K) ab++; }
    const int above = (int)ccd_wsum(ab, cnt, tid), need = keep - above;
    unsigned seen_eq = 0u, placed = 0u;
    for (int j0 = 0; j0 < width; j0 += 256) {
        const int j = j0 + tid;
        const float v = j < width ? scs[sb + j] : CC_DS_NEG_INF;
        const int ok = v > CC_DS_NEG_INF;
        const unsigned key = ok ? ccd_okey(v) : 0u;
        const unsigned eq = ok && key == K ? 1u : 0u;
        const unsigned eq_before = ccd_wscan(eq, cnt, tid) + seen_eq;
        const int sel = ok && (key > K || (eq == 1u && eq_before < (unsigned)need));
        const unsigned sv = sel ? 1u : 0u;
        const unsigned at = ccd_wscan(sv, cnt, tid) + placed;
        if (sel) {
            if (p.order == 0) list[lb + (int)at] = p.base + j;
            else rank[(int)at] = (float)j;
        }
        seen_eq += ccd_wsum(eq, cnt, tid);
        placed += ccd_wsum(sv, cnt, tid);
    }
    if (p.order == 0) return;
    __syncthreads();
    for (int a = tid; a < keep; a += 256) {   /* rank order: the score, then the lower column */
        const int ja = (int)rank[a]; const float va = scs[sb + ja];
        int r = 0;
        for (int b = 0; b < keep; b++) {
            const int jb = (int)rank[b]; const float vb = scs[sb + jb];
            if (vb > va || (vb == va && jb < ja)) r++;
        }
        list[lb + r] = p.base + ja;
    }
}
static int cc_ds_topk(CcBuf *sc, CcBuf *list, const CcDsTopk *p) {
    CcCtx *c = live();
    if (!c || !p || !sc || !list || p->S < 1 || p->width < 0 || p->topk < 1 || p->sc_row < p->width || (p->order && p->topk > 4096)) return 0;
    const size_t smem = p->order ? (size_t)p->topk * sizeof(float) : 0;
    coli_launch(ccd_topk_k, (unsigned)p->S, 256, smem, c->stream, fp(sc), reinterpret_cast<int *>(fpw(list)), *p);
    return launched(c, "ds topk");
}

/* ---- the engram gate ------------------------------------------------------------------------ */
__global__ void __launch_bounds__(256) ccd_engram_k(const float *kv, const float *prm, float *x, CcDsEngram p) {
    __shared__ float sm[768];
    const int cs = blockIdx.x, s = blockIdx.y, tid = threadIdx.x, H = p.H, D = p.D;
    if (s >= p.S || cs >= H) return;
    const int kb = p.kv_off + s * p.kv_row, xb = p.x_off + s * p.x_row + cs * D;
    const int qw = p.qw_off + cs * D, kw = p.kw_off + cs * D;
    float xs = 0.f, ks = 0.f, dt = 0.f;
    for (int i = tid; i < D; i += 256) {
        const float xv = x[xb + i], kvv = kv[kb + cs * D + i];
        xs += xv * xv; ks += kvv * kvv; dt += xv * prm[qw + i] * prm[kw + i] * kvv;
    }
    sm[tid] = xs; sm[256 + tid] = ks; sm[512 + tid] = dt;
    __syncthreads();
    for (int off = 128; off > 0; off >>= 1) {
        if (tid < off) { sm[tid] += sm[tid + off]; sm[256 + tid] += sm[256 + tid + off]; sm[512 + tid] += sm[512 + tid + off]; }
        __syncthreads();
    }
    const float r = (1.f / sqrtf(sm[0] / (float)D + p.eps)) * (1.f / sqrtf(sm[256] / (float)D + p.eps));
    const float t = sm[512] * r / sqrtf((float)D);
    float mag = fabsf(t);
    if (mag < 1.0e-6f) mag = 1.0e-6f;
    float sr = sqrtf(mag);
    if (t < 0.f) sr = -sr;
    const float g = ccd_sig(sr);
    __syncthreads();
    const int vb = kb + H * D;
    for (int i = tid; i < D; i += 256) x[xb + i] = x[xb + i] + g * kv[vb + i];
}
static int cc_ds_engram(CcBuf *kv, CcBuf *prm, CcBuf *x, const CcDsEngram *p) {
    CcCtx *c = live();
    if (!c || !p || !kv || !prm || !x || p->S < 1 || p->H < 1 || p->D < 1) return 0;
    coli_launch(ccd_engram_k, dim3((unsigned)p->H, (unsigned)p->S), 256, 0, c->stream, fp(kv), fp(prm), fpw(x), *p);
    return launched(c, "ds engram");
}
#endif
