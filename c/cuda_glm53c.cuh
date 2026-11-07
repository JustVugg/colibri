/* cuda_glm53c.cuh -- GLM-5.3 Flash's layer ops on the CUDA chain, included by cuda_chain.cu (its
 * CcCtx, live(), fp(), fpw(), launched()) after cuda_dsv4c.cuh (ccd_sig). The CUDA twins of the
 * Vulkan chain's shaders/chain_kda.comp (and chain_kda_rows.spv), the same arithmetic in the same
 * order, so delta_attention.h's coli_kda_step holds them within the bounds tests/test_vk_chain.c
 * holds the shaders to:
 *   cc_kda_conv  the short convolution: a thread a channel, the rows in order, the window
 *                ([C][K], oldest first) carried across calls;
 *   cc_kda_rec   the recurrence and the output norm: a block a head, a thread a column of the
 *                state in registers (KD 64 or 128, VD <= 128); from CC_KDA_ROWS rows a block
 *                of rows at a time with every value the row by row kernel makes;
 * and of chain_mla.comp and chain_dsa.comp's k-pooling (with chain_hgemv.comp on float rows):
 *   cc_mla_core  the attention core over the latent and rope caches, a block a (head, row), the
 *                softmax online over tiles of 64 positions, a causal range or a row's list;
 *   cc_mla_rope, cc_mla_lnorm  RoPE and LayerNorm over segments of rows;
 *   cc_mla_hgemv the per-head block matmuls of a float matrix, rows (with a sigmoid gate) or
 *                transposed;
 *   cc_dsa_pool_keys, cc_dsa_pool_select  the k-pooled indexer: each complete pool's key as the
 *                per-channel softmax mixture of its members', then each row's top pools in rank
 *                order (the scores on a grid first from CC_DSA_GRID pools, the same bits). */
#ifndef CUDA_GLM53C_CUH
#define CUDA_GLM53C_CUH

/* ---- KDA ------------------------------------------------------------------------------ */
/* products and sums nvcc never contracts into an FMA: the row and the rows kernels (and the
 * shader) round the same steps, so the two kernels give the same bits */
#define CG_MUL(a, b) __fmul_rn((a), (b))
#define CG_ADD(a, b) __fadd_rn((a), (b))
#define CG_SUB(a, b) __fsub_rn((a), (b))
__global__ void __launch_bounds__(128) ccg_kda_conv_k(const float *in, const float *w, float *win, float *out, CcKdaConv p) {
    const int c = (int)(blockIdx.x * 128u + threadIdx.x);
    if (c >= p.C) return;
    const int part = c / p.P, j = c - part * p.P;
    float hist[8];
    const int wb = p.win_off + c * p.K, tb = p.w_off + c * p.K;
    #pragma unroll
    for (int k = 0; k < 8; k++) hist[k] = k < p.K ? win[wb + k] : 0.f;
    for (int s = 0; s < p.S; s++) {
        #pragma unroll
        for (int k = 0; k < 7; k++) if (k + 1 < p.K) hist[k] = hist[k + 1];
        const float cur = in[p.in_off + part * p.in_part + s * p.in_row + j];
        #pragma unroll
        for (int k = 0; k < 8; k++) if (k == p.K - 1) hist[k] = cur;
        float sum = 0.f;
        #pragma unroll
        for (int k = 0; k < 8; k++) if (k < p.K) sum = CG_ADD(sum, CG_MUL(w[tb + k], hist[k]));
        out[p.out_off + s * p.out_row + c] = sum / (1.f + expf(-sum));
    }
    #pragma unroll
    for (int k = 0; k < 8; k++) if (k < p.K) win[wb + k] = hist[k];
}

/* chain_kda.comp's rec(): a row at a time */
template <int KD>
__global__ void __launch_bounds__(128) ccg_kda_rec_k(const float *m, const float *fr, const float *br, const float *gr,
                                                     const float *w, float *st, float *o, CcKdaRec p, int flags) {
    __shared__ float sq[KD], sk[KD], al[KD], ov[128], scal[2];
    const int h = blockIdx.x, v = threadIdx.x, Pw = p.P;
    const bool own = v < p.VD;
    float col[KD];
    const int sb = p.st_off + h * KD * p.VD;
    #pragma unroll
    for (int k = 0; k < KD; k++) col[k] = own ? st[sb + k * p.VD + v] : 0.f;
    const float ea = (flags & 1) ? w[p.prm_off + h] : expf(w[p.prm_off + h]);
    const float nw = own ? w[p.prm_off + p.H + Pw + v] : 0.f;
    const float rkd = 1.f / sqrtf((float)KD);
    for (int s = 0; s < p.S; s++) {
        const int mb = p.m_off + s * p.m_row;
        for (int k = v; k < KD; k += 128) {
            sq[k] = m[mb + h * KD + k];
            sk[k] = m[mb + Pw + h * KD + k];
            const float f = fr[p.f_off + s * p.f_row + h * KD + k] + w[p.prm_off + p.H + h * KD + k];
            al[k] = expf(p.lb * ccd_sig(ea * f));
        }
        __syncthreads();
        if (v == 0) {   /* the l2 sums, in order */
            if (flags & 2) {
                float qs = 0.f, ks = 0.f;
                for (int k = 0; k < KD; k++) { qs = CG_ADD(qs, CG_MUL(sq[k], sq[k])); ks = CG_ADD(ks, CG_MUL(sk[k], sk[k])); }
                scal[0] = 1.f / sqrtf(qs + p.neps);
                scal[1] = 1.f / sqrtf(ks + p.neps);
            } else {
                float qs = p.neps, ks = p.neps;
                for (int k = 0; k < KD; k++) { qs = CG_ADD(qs, CG_MUL(sq[k], sq[k])); ks = CG_ADD(ks, CG_MUL(sk[k], sk[k])); }
                scal[0] = rkd / sqrtf(qs);
                scal[1] = 1.f / sqrtf(ks);
            }
        }
        __syncthreads();
        const float qn = scal[0], kn = scal[1];
        const float beta = ccd_sig(br[p.b_off + s * p.b_row + h]);
        const float val = own ? m[mb + 2 * Pw + h * p.VD + v] : 0.f;
        float mem = 0.f, res = 0.f;
        if (flags & 2) {
            for (int k = v; k < KD; k += 128) { sq[k] = CG_MUL(CG_MUL(sq[k], qn), rkd); sk[k] = CG_MUL(sk[k], kn); }
            __syncthreads();
            #pragma unroll
            for (int k = 0; k < KD; k++) { col[k] = CG_MUL(col[k], al[k]); mem = CG_ADD(mem, CG_MUL(sk[k], col[k])); }
            const float vt = CG_MUL(CG_SUB(val, mem), beta);
            #pragma unroll
            for (int k = 0; k < KD; k++) { col[k] = CG_ADD(col[k], CG_MUL(sk[k], vt)); res = CG_ADD(res, CG_MUL(sq[k], col[k])); }
        } else {
            #pragma unroll
            for (int k = 0; k < KD; k++) { col[k] = CG_MUL(col[k], al[k]); mem = CG_ADD(mem, CG_MUL(CG_MUL(sk[k], kn), col[k])); }
            #pragma unroll
            for (int k = 0; k < KD; k++) {
                col[k] = CG_ADD(col[k], CG_MUL(CG_MUL(CG_MUL(sk[k], kn), CG_SUB(val, mem)), beta));
                res = CG_ADD(res, CG_MUL(CG_MUL(sq[k], qn), col[k]));
            }
        }
        ov[v] = own ? res : 0.f;
        __syncthreads();
        if (v == 0) {
            float sum = 0.f;
            for (int d = 0; d < p.VD; d++) sum = CG_ADD(sum, CG_MUL(ov[d], ov[d]));
            scal[0] = 1.f / sqrtf(CG_ADD(sum / (float)p.VD, p.eps));
        }
        __syncthreads();
        if (own) o[p.y_off + s * p.y_row + h * p.VD + v] = res * scal[0] * nw * ccd_sig(gr[p.g_off + s * p.g_row + h * p.VD + v]);
        __syncthreads();
    }
    if (own) {
        #pragma unroll
        for (int k = 0; k < KD; k++) st[sb + k * p.VD + v] = col[k];
    }
}

/* chain_kda.comp's rec_rows(): R rows a block, every value as the row kernel makes it */
#define CC_KDA_R 8
template <int KD>
__global__ void __launch_bounds__(128) ccg_kda_rows_k(const float *m, const float *fr, const float *br, const float *gr,
                                                      const float *w, float *st, float *o, CcKdaRec p, int flags) {
    __shared__ float bq[CC_KDA_R * KD], bk[CC_KDA_R * KD], ba[CC_KDA_R * KD], bv[CC_KDA_R * 128], bo[CC_KDA_R * 128];
    __shared__ float bqn[CC_KDA_R], bkn[CC_KDA_R], bbeta[CC_KDA_R], bon[CC_KDA_R];
    const int h = blockIdx.x, v = threadIdx.x, Pw = p.P;
    const bool own = v < p.VD;
    float col[KD];
    const int sb = p.st_off + h * KD * p.VD;
    #pragma unroll
    for (int k = 0; k < KD; k++) col[k] = own ? st[sb + k * p.VD + v] : 0.f;
    const float ea = (flags & 1) ? w[p.prm_off + h] : expf(w[p.prm_off + h]);
    const float rkd = 1.f / sqrtf((float)KD);
    for (int s0 = 0; s0 < p.S; s0 += CC_KDA_R) {
        const int nr = p.S - s0 < CC_KDA_R ? p.S - s0 : CC_KDA_R;
        for (int i = v; i < nr * KD; i += 128) {
            const int r = i / KD, k = i - r * KD, mb = p.m_off + (s0 + r) * p.m_row;
            bq[i] = m[mb + h * KD + k];
            bk[i] = m[mb + Pw + h * KD + k];
            const float f = fr[p.f_off + (s0 + r) * p.f_row + h * KD + k] + w[p.prm_off + p.H + h * KD + k];
            ba[i] = expf(p.lb * ccd_sig(ea * f));
        }
        for (int i = v; i < nr * 128; i += 128) {
            const int r = i >> 7, d = i & 127;
            bv[i] = d < p.VD ? m[p.m_off + (s0 + r) * p.m_row + 2 * Pw + h * p.VD + d] : 0.f;
        }
        __syncthreads();
        if (v < nr) {   /* row v's l2 sums, in order */
            const int r = v;
            if (flags & 2) {
                float qs = 0.f, ks = 0.f;
                for (int k = 0; k < KD; k++) { qs = CG_ADD(qs, CG_MUL(bq[r * KD + k], bq[r * KD + k])); ks = CG_ADD(ks, CG_MUL(bk[r * KD + k], bk[r * KD + k])); }
                bqn[r] = 1.f / sqrtf(qs + p.neps);
                bkn[r] = 1.f / sqrtf(ks + p.neps);
            } else {
                float qs = p.neps, ks = p.neps;
                for (int k = 0; k < KD; k++) { qs = CG_ADD(qs, CG_MUL(bq[r * KD + k], bq[r * KD + k])); ks = CG_ADD(ks, CG_MUL(bk[r * KD + k], bk[r * KD + k])); }
                bqn[r] = rkd / sqrtf(qs);
                bkn[r] = 1.f / sqrtf(ks);
            }
            bbeta[r] = ccd_sig(br[p.b_off + (s0 + r) * p.b_row + h]);
        }
        __syncthreads();
        if (flags & 2) {
            for (int i = v; i < nr * KD; i += 128) { const int r = i / KD; bq[i] = CG_MUL(CG_MUL(bq[i], bqn[r]), rkd); bk[i] = CG_MUL(bk[i], bkn[r]); }
            __syncthreads();
        }
        for (int r = 0; r < nr; r++) {   /* the rows in order, each column in its own registers */
            const float qn = bqn[r], kn = bkn[r], beta = bbeta[r];
            const float val = own ? bv[r * 128 + v] : 0.f;
            float mem = 0.f, res = 0.f;
            if (flags & 2) {
                #pragma unroll
                for (int k = 0; k < KD; k++) { col[k] = CG_MUL(col[k], ba[r * KD + k]); mem = CG_ADD(mem, CG_MUL(bk[r * KD + k], col[k])); }
                const float vt = CG_MUL(CG_SUB(val, mem), beta);
                #pragma unroll
                for (int k = 0; k < KD; k++) { col[k] = CG_ADD(col[k], CG_MUL(bk[r * KD + k], vt)); res = CG_ADD(res, CG_MUL(bq[r * KD + k], col[k])); }
            } else {
                #pragma unroll
                for (int k = 0; k < KD; k++) { col[k] = CG_MUL(col[k], ba[r * KD + k]); mem = CG_ADD(mem, CG_MUL(CG_MUL(bk[r * KD + k], kn), col[k])); }
                #pragma unroll
                for (int k = 0; k < KD; k++) {
                    col[k] = CG_ADD(col[k], CG_MUL(CG_MUL(CG_MUL(bk[r * KD + k], kn), CG_SUB(val, mem)), beta));
                    res = CG_ADD(res, CG_MUL(CG_MUL(bq[r * KD + k], qn), col[k]));
                }
            }
            bo[r * 128 + v] = own ? res : 0.f;
        }
        __syncthreads();
        if (v < nr) {   /* row v's output norm, its sum in order */
            float sum = 0.f;
            for (int d = 0; d < p.VD; d++) sum = CG_ADD(sum, CG_MUL(bo[v * 128 + d], bo[v * 128 + d]));
            bon[v] = 1.f / sqrtf(CG_ADD(sum / (float)p.VD, p.eps));
        }
        __syncthreads();
        for (int i = v; i < nr * 128; i += 128) {
            const int r = i >> 7, d = i & 127, s = s0 + r;
            if (d < p.VD)
                o[p.y_off + s * p.y_row + h * p.VD + d] = bo[i] * bon[r] * w[p.prm_off + p.H + Pw + d] * ccd_sig(gr[p.g_off + s * p.g_row + h * p.VD + d]);
        }
        __syncthreads();
    }
    if (own) {
        #pragma unroll
        for (int k = 0; k < KD; k++) st[sb + k * p.VD + v] = col[k];
    }
}

static int cc_kda_conv(CcBuf *in, CcBuf *w, CcBuf *win, CcBuf *out, const CcKdaConv *p) {
    CcCtx *c = live();
    if (!c || !p || !in || !w || !win || !out || p->S < 1 || p->C < 1 || p->K < 1 || p->K > 8 || p->P < 1) return 0;
    coli_launch(ccg_kda_conv_k, dim3((unsigned)((p->C + 127) / 128)), 128, 0, c->stream, fp(in), fp(w), fpw(win), fpw(out), *p);
    return launched(c, "kda conv");
}
/* the rows kernel from CC_KDA_ROWS rows (COLI_CUDA_KDA_ROWS=0: the row by row kernel) */
static int cc_kda_rows_on(void) {
    static int on = -1;
    if (on < 0) { const char *e = getenv("COLI_CUDA_KDA_ROWS"); on = !(e && *e == '0'); }
    return on;
}
static int cc_kda_rec(int KD, CcBuf *m, CcBuf *f, CcBuf *b, CcBuf *g, CcBuf *prm, CcBuf *st, CcBuf *y, const CcKdaRec *p, int flags) {
    CcCtx *c = live();
    if (!c || !p || !m || !f || !b || !g || !prm || !st || !y || p->S < 1 || p->H < 1 || p->VD < 1 || p->VD > 128 ||
        (KD != 64 && KD != 128)) return 0;
    const int rows = p->S >= CC_KDA_ROWS && cc_kda_rows_on();
    const dim3 grid((unsigned)p->H);
    if (KD == 128) {
        if (rows) coli_launch(ccg_kda_rows_k<128>, grid, 128, 0, c->stream, fp(m), fp(f), fp(b), fp(g), fp(prm), fpw(st), fpw(y), *p, flags);
        else coli_launch(ccg_kda_rec_k<128>, grid, 128, 0, c->stream, fp(m), fp(f), fp(b), fp(g), fp(prm), fpw(st), fpw(y), *p, flags);
    } else {
        if (rows) coli_launch(ccg_kda_rows_k<64>, grid, 128, 0, c->stream, fp(m), fp(f), fp(b), fp(g), fp(prm), fpw(st), fpw(y), *p, flags);
        else coli_launch(ccg_kda_rec_k<64>, grid, 128, 0, c->stream, fp(m), fp(f), fp(b), fp(g), fp(prm), fpw(st), fpw(y), *p, flags);
    }
    return launched(c, "kda rec");
}
/* ---- MLA (chain_mla.comp, chain_hgemv.comp's float rows) --------------------------------- */
__device__ __forceinline__ float cg_warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
/* mode 0, the core: a block a (head, row), the softmax online over tiles of 64 positions */
__global__ void __launch_bounds__(256) ccg_mla_core_k(const float *qa, const float *qr, const float *lc, const float *rc, const int *sel,
                                                      float *ob, CcMlaCore p) {
    __shared__ float qs[1024], qrs[128], sc[64], red[256];
    __shared__ int tp[64];
    const int h = blockIdx.x, s = blockIdx.y, tid = threadIdx.x, lane = tid & 31, sg = tid >> 5, nsg = 8;
    const int qb = p.qa_off + s * p.qa_row + h * p.qa_seg, rb = p.qr_off + s * p.qr_row + h * p.qr_seg;
    for (int i = tid; i < p.K; i += 256) qs[i] = qa[qb + i];
    for (int i = tid; i < p.R; i += 256) qrs[i] = qr[rb + i];
    int n = p.pos_base + s + 1 - p.kv_start, lb = 0;
    bool list = false;
    if (p.sel_row > 0) {
        lb = p.sel_off + s * p.sel_row;
        const int c = sel[lb];
        if (c >= 0) { n = c; list = true; }
    }
    __syncthreads();
    float m = -3.0e38f, l = 0.f, acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int t0 = 0; t0 < n; t0 += 64) {
        const int cnt = n - t0 < 64 ? n - t0 : 64;
        if (tid < 64) {
            int t = -1;
            if (tid < cnt) t = list ? sel[lb + 1 + t0 + tid] : p.kv_start + t0 + tid;
            tp[tid] = t;
        }
        __syncthreads();
        for (int j = sg; j < cnt; j += nsg) {   /* a warp a position, its lanes split the dot */
            const int t = tp[j];
            float a = 0.f;
            if (t >= 0) {
                const int lo = p.lat_off + t * p.lat_row, ro = p.rope_off + t * p.rope_row;
                for (int i = lane; i < p.K; i += 32) a += qs[i] * lc[lo + i];
                for (int i = lane; i < p.R; i += 32) a += qrs[i] * rc[ro + i];
            }
            const float tot = cg_warp_sum(a);
            if (lane == 0) sc[j] = t >= 0 ? tot * p.scale : -3.0e38f;
        }
        __syncthreads();
        red[tid] = tid < cnt ? sc[tid] : -3.0e38f;
        __syncthreads();
        for (int off = 128; off > 0; off >>= 1) {
            if (tid < off) red[tid] = fmaxf(red[tid], red[tid + off]);
            __syncthreads();
        }
        const float mt = red[0];
        __syncthreads();
        if (mt > -1.0e38f) {   /* a tile with a live position */
            const float mn = fmaxf(m, mt), corr = m > -1.0e38f ? expf(m - mn) : 0.f;
            if (tid < 64) sc[tid] = tid < cnt && tp[tid] >= 0 ? expf(sc[tid] - mn) : 0.f;
            __syncthreads();
            red[tid] = tid < 64 ? sc[tid] : 0.f;
            __syncthreads();
            for (int off = 128; off > 0; off >>= 1) {
                if (tid < off) red[tid] += red[tid + off];
                __syncthreads();
            }
            l = l * corr + red[0];
            for (int k = 0; k < 4; k++) {
                const int i = tid + 256 * k;
                if (i >= p.K) break;
                float a = acc[k] * corr;
                for (int j = 0; j < cnt; j++) {
                    const int t = tp[j];
                    if (t >= 0) a += sc[j] * lc[p.lat_off + t * p.lat_row + i];
                }
                acc[k] = a;
            }
            m = mn;
        }
        __syncthreads();
    }
    const float inv = l > 0.f ? 1.f / l : 0.f;
    const int o0 = p.o_off + s * p.o_row + h * p.o_seg;
    for (int k = 0; k < 4; k++) {
        const int i = tid + 256 * k;
        if (i >= p.K) break;
        ob[o0 + i] = acc[k] * inv;
    }
}
/* modes 1 and 2: a block a segment (g = blockIdx.x + blockIdx.y * gridDim.x) */
__global__ void __launch_bounds__(256) ccg_mla_rope_k(const float *x, const float *cs, float *y, CcMlaRow p, int copy_rest) {
    __shared__ float red[256];
    const int g = (int)(blockIdx.x + blockIdx.y * gridDim.x);
    if (g >= p.nseg) return;
    const int row = g / p.per_row, j = g - row * p.per_row, tid = threadIdx.x, half = p.rd / 2;
    const int xb = p.x_off + row * p.x_row + j * p.x_seg, yb = p.y_off + row * p.y_row + j * p.y_seg;
    for (int i = tid; i < p.rd; i += 256) red[i] = x[xb + i];
    __syncthreads();
    const int cb = p.cs_off + row * p.cs_row;
    for (int i = tid; i < half; i += 256) {
        const float c = cs[cb + 2 * i], sn = cs[cb + 2 * i + 1];
        const float a = p.style == 1 ? red[2 * i] : red[i], b = p.style == 1 ? red[2 * i + 1] : red[i + half];
        y[yb + i] = CG_SUB(CG_MUL(a, c), CG_MUL(b, sn));
        y[yb + i + half] = CG_ADD(CG_MUL(b, c), CG_MUL(a, sn));
    }
    if (copy_rest) for (int i = p.rd + tid; i < p.seg_len; i += 256) y[yb + i] = x[xb + i];
}
__global__ void __launch_bounds__(256) ccg_mla_lnorm_k(const float *x, const float *prm, float *y, CcMlaRow p) {
    __shared__ float red[256];
    const int g = (int)(blockIdx.x + blockIdx.y * gridDim.x);
    if (g >= p.nseg) return;
    const int row = g / p.per_row, j = g - row * p.per_row, tid = threadIdx.x, D = p.seg_len;
    const int xb = p.x_off + row * p.x_row + j * p.x_seg, yb = p.y_off + row * p.y_row + j * p.y_seg;
    float a = 0.f;
    for (int i = tid; i < D; i += 256) a += x[xb + i];
    red[tid] = a;
    __syncthreads();
    for (int off = 128; off > 0; off >>= 1) {
        if (tid < off) red[tid] += red[tid + off];
        __syncthreads();
    }
    const float mean = red[0] / (float)D;
    __syncthreads();
    a = 0.f;
    for (int i = tid; i < D; i += 256) { const float d = x[xb + i] - mean; a += d * d; }
    red[tid] = a;
    __syncthreads();
    for (int off = 128; off > 0; off >>= 1) {
        if (tid < off) red[tid] += red[tid + off];
        __syncthreads();
    }
    const float rr = 1.f / sqrtf(red[0] / (float)D + p.eps);
    __syncthreads();
    for (int i = tid; i < D; i += 256) {
        const float v = (x[xb + i] - mean) * rr * prm[p.w_off + i];
        y[yb + i] = p.has_b ? v + prm[p.b_off + i] : v;
    }
}
/* chain_hgemv.comp on float rows: mode 0 a warp an output row (grid stride), mode 1 a block a
 * (head, row) with x's slice in shared memory and a thread a column */
__global__ void __launch_bounds__(256) ccg_hgemv_rows_k(const float *x, const float *w, float *y, const float *g, CcHgemv p) {
    const int s = blockIdx.y, lane = threadIdx.x & 31, nsg = 8, total = p.H * p.n, stride = (int)gridDim.x * nsg;
    for (int og = (int)blockIdx.x * nsg + (int)(threadIdx.x >> 5); og < total; og += stride) {
        const int h = og / p.n, o = og - h * p.n;
        const long row = (long)h * p.hstride + p.hoff + o;
        const int xb = p.x_off + s * p.x_row + h * p.x_seg;
        const float *wr = w + p.w_off + row * p.I;
        float a = 0.f;
        for (int i = lane; i < p.I; i += 32) a += wr[i] * x[xb + i];
        float tot = cg_warp_sum(a);
        if (lane == 0) {
            if (p.has_gate) tot *= 1.f / (1.f + expf(-g[p.g_off + s * p.g_row + h * p.n + o]));
            y[p.y_off + s * p.y_row + h * p.y_seg + o] = tot;
        }
    }
}
__global__ void __launch_bounds__(256) ccg_hgemv_trans_k(const float *x, const float *w, float *y, CcHgemv p) {
    __shared__ float xs[1024];
    const int h = blockIdx.x, s = blockIdx.y, tid = threadIdx.x;
    const int xb = p.x_off + s * p.x_row + h * p.x_seg;
    for (int d = tid; d < p.n; d += 256) xs[d] = x[xb + d];
    __syncthreads();
    const long r0 = (long)h * p.hstride + p.hoff;
    const int yb = p.y_off + s * p.y_row + h * p.y_seg;
    for (int i = tid; i < p.I; i += 256) {
        float a = 0.f;
        for (int d = 0; d < p.n; d++) a += xs[d] * w[p.w_off + (r0 + d) * p.I + i];
        y[yb + i] = a;
    }
}

/* ---- the k-pooled indexer (chain_dsa.comp modes 1, 2 and 4) ------------------------------- */
__global__ void __launch_bounds__(256) ccg_dsa_pool_keys_k(const float *kc, const float *gt, const float *prm, float *pk, CcDsaPool p) {
    const int pp = p.p0 + (int)blockIdx.x, start = pp * p.pool;
    for (int d = threadIdx.x; d < p.ID; d += 256) {
        float mxv = -3.402823466e38f;
        for (int j = 0; j < p.pool; j++) {
            const float lg = gt[p.g_off + (start + j) * p.g_row + d] + prm[p.ape_off + j * p.ID + d];
            if (lg > mxv) mxv = lg;
        }
        float total = 0.f;
        for (int j = 0; j < p.pool; j++) total = CG_ADD(total, expf(gt[p.g_off + (start + j) * p.g_row + d] + prm[p.ape_off + j * p.ID + d] - mxv));
        float mixed = 0.f;
        for (int j = 0; j < p.pool; j++) {
            const float wgt = expf(gt[p.g_off + (start + j) * p.g_row + d] + prm[p.ape_off + j * p.ID + d] - mxv) / total;
            mixed = CG_ADD(mixed, CG_MUL(wgt, kc[p.k_off + (start + j) * p.k_row + d]));
        }
        pk[p.pk_off + pp * p.ID + d] = mixed;
    }
}
/* mode 4: a thread a (pool, head) for the head's dot, then a thread a pool joining the heads in
 * mode 2's order: the same bits as the selection's own scoring. The dots and the scores, as the
 * pooled keys, round sparse_index.h's steps in its order with nothing contracted into an FMA. */
__global__ void __launch_bounds__(256) ccg_dsa_pool_scores_k(const float *iq, const float *hw, const float *pk, float *sc, CcDsaPick p) {
    __shared__ float qs[4096], ws[64], dh[256];
    const int s = blockIdx.y, tid = threadIdx.x;
    const int P = 256 / p.IH > 0 ? 256 / p.IH : 1, nvis = (p.pos_base + s + 1) / p.pool, p0 = (int)blockIdx.x * P;
    if (p0 >= nvis) return;
    const int nq = p.IH * p.ID, qb = p.q_off + s * p.q_row;
    for (int i = tid; i < nq; i += 256) qs[i] = iq[qb + i];
    for (int h = tid; h < p.IH; h += 256) ws[h] = hw[p.w_off + s * p.w_row + h] / p.wdiv;
    __syncthreads();
    const int j = tid / p.IH, h = tid - j * p.IH, pp = p0 + j;
    if (j < P && pp < nvis) {
        float dot = 0.f;
        for (int d = 0; d < p.ID; d++) dot = CG_ADD(dot, CG_MUL(qs[h * p.ID + d], pk[p.pk_off + pp * p.ID + d]));
        dh[tid] = dot;
    }
    __syncthreads();
    if (tid < P && p0 + tid < nvis) {
        float score = 0.f;
        for (int gg = 0; gg < p.IH; gg++) { const float dot = dh[tid * p.IH + gg]; if (dot > 0.f) score = CG_ADD(score, CG_MUL(CG_MUL(ws[gg], dot), p.scale)); }
        sc[s * p.sc_row + p0 + tid] = score;
    }
}
__device__ float cg_pool_score(const float *qs, const float *ws, const float *pk, int pp, const CcDsaPick &p) {
    float score = 0.f;
    for (int h = 0; h < p.IH; h++) {
        float dot = 0.f;
        for (int d = 0; d < p.ID; d++) dot = CG_ADD(dot, CG_MUL(qs[h * p.ID + d], pk[p.pk_off + pp * p.ID + d]));
        if (dot > 0.f) score = CG_ADD(score, CG_MUL(CG_MUL(ws[h], dot), p.scale));
    }
    return score;
}
/* mode 2: a block a row; the keep-th largest key bit by bit, then the pools above it and those
 * equal to it in pool order, ranked by score and then the lower pool */
__global__ void __launch_bounds__(256) ccg_dsa_pool_select_k(const float *iq, const float *hw, const float *pk, float *sc, int *sel,
                                                             CcDsaPick p, int scored) {
    __shared__ float qs[4096], ws[64];
    __shared__ unsigned cnt[256], pkey[1024];
    __shared__ int plist[1024];
    const int s = blockIdx.x, tid = threadIdx.x, q = p.pos_base + s, pool = p.pool;
    const int nvis = (q + 1) / pool, wanted = p.topk / pool, keep = wanted < nvis ? wanted : nvis;
    const int width = p.tail ? p.topk + pool - 1 : p.topk;
    const int lb = s * p.sel_row, sb = s * p.sc_row;
    for (int i = tid; i < width; i += 256) sel[lb + 1 + i] = -1;
    if (!scored) {
        const int nq = p.IH * p.ID, qb = p.q_off + s * p.q_row;
        for (int i = tid; i < nq; i += 256) qs[i] = iq[qb + i];
        for (int h = tid; h < p.IH; h += 256) ws[h] = hw[p.w_off + s * p.w_row + h] / p.wdiv;
        __syncthreads();
        for (int pp = tid; pp < nvis; pp += 256) sc[sb + pp] = cg_pool_score(qs, ws, pk, pp, p);
        __syncthreads();
    }
    if (keep > 0) {
        unsigned K = 0u;
        for (int b = 31; b >= 0; b--) {
            const unsigned cand = K | (1u << (unsigned)b);
            unsigned c = 0u;
            for (int t = tid; t < nvis; t += 256) if (ccd_okey(sc[sb + t]) >= cand) c++;
            if (ccd_wsum(c, cnt, tid) >= (unsigned)keep) K = cand;
        }
        const int chunk = (nvis + 255) / 256, c0 = tid * chunk, c1 = nvis < c0 + chunk ? nvis : c0 + chunk;
        unsigned above = 0u, at = 0u;
        for (int t = c0; t < c1; t++) { const unsigned k = ccd_okey(sc[sb + t]); above += k > K ? 1u : 0u; at += k == K ? 1u : 0u; }
        const unsigned n1 = ccd_wsum(above, cnt, tid);
        unsigned o = ccd_wscan(above, cnt, tid);
        for (int t = c0; t < c1; t++) { const unsigned k = ccd_okey(sc[sb + t]); if (k > K) { plist[o] = t; pkey[o] = k; o++; } }
        o = n1 + ccd_wscan(at, cnt, tid);
        for (int t = c0; t < c1 && o < (unsigned)keep; t++) { const unsigned k = ccd_okey(sc[sb + t]); if (k == K) { plist[o] = t; pkey[o] = k; o++; } }
        __syncthreads();
        for (int i = tid; i < keep; i += 256) {   /* rank order: the higher score first, then the lower pool */
            const unsigned ki = pkey[i]; const int pi = plist[i];
            int r = 0;
            for (int jj = 0; jj < keep; jj++) { const unsigned kj = pkey[jj]; if (kj > ki || (kj == ki && plist[jj] < pi)) r++; }
            for (int jj = 0; jj < pool; jj++) sel[lb + 1 + r * pool + jj] = pi * pool + jj;
        }
    }
    if (p.tail && tid == 0) {
        const int visible = q + 1, tail = visible % pool, ts = visible - tail;
        for (int jj = 0; jj < tail && jj < pool - 1; jj++) sel[lb + 1 + p.topk + jj] = ts + jj;
    }
    if (tid == 0) sel[lb] = width;
}

static void cg_grid(long n, dim3 *g) {
    const long gx = n < 65535 ? n : 65535;
    *g = dim3((unsigned)(gx > 0 ? gx : 1), (unsigned)((n + gx - 1) / (gx > 0 ? gx : 1)));
}
static int cc_mla_core(CcBuf *qabs, CcBuf *qr, CcBuf *lat, CcBuf *rope, CcBuf *sel, CcBuf *clat, const CcMlaCore *p) {
    CcCtx *c = live();
    if (!c || !p || !qabs || !lat || !clat || p->S < 1 || p->H < 1 || p->K < 1 || p->K > 1024 || p->R < 0 || p->R > 128 ||
        (p->R > 0 && (!rope || !qr)) || p->pos_base < 0 || p->kv_start < 0 || p->kv_start > p->pos_base || p->S > 65535) return 0;
    CcMlaCore b = *p;
    if (!sel) b.sel_row = 0;
    coli_launch(ccg_mla_core_k, dim3((unsigned)p->H, (unsigned)p->S), 256, 0, c->stream, fp(qabs), p->R > 0 ? fp(qr) : fp(qabs),
                fp(lat), p->R > 0 ? fp(rope) : fp(lat), b.sel_row > 0 ? (const int *)fp(sel) : (const int *)nullptr, fpw(clat), b);
    return launched(c, "mla core");
}
static int cc_mla_rope(CcBuf *x, CcBuf *cs, CcBuf *y, const CcMlaRow *p) {
    CcCtx *c = live();
    if (!c || !p || !x || !cs || !y || p->nseg < 0 || p->rd < 2 || p->rd > 256 || (p->rd & 1) || p->rd > p->seg_len || p->per_row < 1) return 0;
    if (p->nseg == 0) return 1;
    dim3 g; cg_grid(p->nseg, &g);
    coli_launch(ccg_mla_rope_k, g, 256, 0, c->stream, (const float *)fp(x), fp(cs), fpw(y), *p, x != y);
    return launched(c, "mla rope");
}
static int cc_mla_lnorm(CcBuf *x, CcBuf *prm, CcBuf *y, const CcMlaRow *p) {
    CcCtx *c = live();
    if (!c || !p || !x || !prm || !y || p->nseg < 0 || p->seg_len < 1 || p->per_row < 1) return 0;
    if (p->nseg == 0) return 1;
    dim3 g; cg_grid(p->nseg, &g);
    coli_launch(ccg_mla_lnorm_k, g, 256, 0, c->stream, (const float *)fp(x), fp(prm), fpw(y), *p);
    return launched(c, "mla lnorm");
}
static int cc_mla_hgemv(CcBuf *w, CcBuf *x, CcBuf *y, CcBuf *gate, const CcHgemv *p) {
    CcCtx *c = live();
    if (!c || !p || !w || !x || !y || p->S < 1 || p->S > 65535 || p->H < 1 || p->H > 65535 || p->n < 1 || p->I < 1 ||
        (p->trans && p->n > 1024) || (p->has_gate && !gate && !p->trans)) return 0;
    if (p->trans) coli_launch(ccg_hgemv_trans_k, dim3((unsigned)p->H, (unsigned)p->S), 256, 0, c->stream, fp(x), fp(w), fpw(y), *p);
    else {
        long rows = (long)p->H * p->n, wg = (rows + 7) / 8;
        if (wg > 4096) wg = 4096;
        coli_launch(ccg_hgemv_rows_k, dim3((unsigned)wg, (unsigned)p->S), 256, 0, c->stream, fp(x), fp(w), fpw(y),
                    p->has_gate ? fp(gate) : fp(x), *p);
    }
    return launched(c, "mla hgemv");
}
#define CC_DSA_GRID 32   /* pools from which the scores go on a grid first */
static int cc_dsa_pool_keys(CcBuf *keys, CcBuf *gates, CcBuf *prm, CcBuf *pk, const CcDsaPool *p) {
    CcCtx *c = live();
    if (!c || !p || p->np < 0 || p->pool < 1 || p->p0 < 0 || p->ID < 1) return 0;
    if (p->np == 0) return 1;
    if (!keys || !gates || !prm || !pk) return 0;
    coli_launch(ccg_dsa_pool_keys_k, dim3((unsigned)p->np), 256, 0, c->stream, fp(keys), fp(gates), fp(prm), fpw(pk), *p);
    return launched(c, "dsa pool keys");
}
static int cc_dsa_pool_select(CcBuf *iq, CcBuf *hw, CcBuf *pk, CcBuf *sc, CcBuf *sel, const CcDsaPick *p) {
    CcCtx *c = live();
    if (!c || !p || !iq || !hw || !pk || !sc || !sel || p->S < 1 || p->S > 65535 || p->IH < 1 || p->IH > 64 || p->ID < 1 ||
        p->IH * p->ID > 4096 || p->pool < 1 || p->topk < p->pool || p->topk % p->pool || p->topk / p->pool > 1024 || p->pos_base < 0 ||
        p->sc_row < (p->pos_base + p->S) / p->pool || p->sel_row < 1 + p->topk + (p->tail ? p->pool - 1 : 0)) return 0;
    const int np = (p->pos_base + p->S) / p->pool;
    int scored = 0;
    if (np >= CC_DSA_GRID) {
        const int P = 256 / p->IH > 0 ? 256 / p->IH : 1;
        coli_launch(ccg_dsa_pool_scores_k, dim3((unsigned)((np + P - 1) / P), (unsigned)p->S), 256, 0, c->stream, fp(iq), fp(hw), fp(pk),
                    fpw(sc), *p);
        if (!launched(c, "dsa pool scores")) return 0;
        scored = 1;
    }
    coli_launch(ccg_dsa_pool_select_k, dim3((unsigned)p->S), 256, 0, c->stream, fp(iq), fp(hw), fp(pk), fpw(sc), (int *)fpw(sel), *p, scored);
    return launched(c, "dsa pool select");
}
#endif
