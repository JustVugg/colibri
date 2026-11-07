/* cuda_glm53c.cuh -- GLM-5.3 Flash's layer ops on the CUDA chain, included by cuda_chain.cu (its
 * CcCtx, live(), fp(), fpw(), launched()) after cuda_dsv4c.cuh (ccd_sig). The CUDA twins of the
 * Vulkan chain's shaders/chain_kda.comp (and chain_kda_rows.spv), the same arithmetic in the same
 * order, so delta_attention.h's coli_kda_step holds them within the bounds tests/test_vk_chain.c
 * holds the shaders to:
 *   cc_kda_conv  the short convolution: a thread a channel, the rows in order, the window
 *                ([C][K], oldest first) carried across calls;
 *   cc_kda_rec   the recurrence and the output norm: a block a head, a thread a column of the
 *                state in registers (KD 64 or 128, VD <= 128); from CC_KDA_ROWS rows a block
 *                of rows at a time with every value the row by row kernel makes. */
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
#endif
