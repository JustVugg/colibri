/* cuda_k3c.cuh -- Kimi K3's two ops on the CUDA chain (cuda_chain.h's ares_mix and situ), included
 * once by cuda_chain.cu after cuda_glm53c.cuh, whose grid helper it uses. The Vulkan chain's
 * vkc_ares_mix and vkc_situ (shaders/chain_ares.comp) with the same fields in the same order:
 *   ares_mix: kimi_k3.c's res_mix. Row s mixes nb + 1 vectors of D floats, the block snapshots
 *     v_e at b_off + s*b_row + e*D (e < nb) and the running prefix v_nb at x_off + s*x_row, by
 *     the softmax of score_e = (v_e . w) / sqrt(mean(v_e^2) + eps) (w at w_off in prm), the sum
 *     over e in order. The dot and the squares in double as the CPU's (a tree here, a loop
 *     there: the same to the last bits of a double), the softmax the CPU's softmax_ (the
 *     maximum, expf(x - max) in order with their sum, then a division). One block a row.
 *   situ: y[y_off + i] = b1*tanh(g/b1)*sigmoid(g)*b2*tanh(u/b2), the CPU's situf_ in its order. */

__global__ static void ccg_ares_mix_k(const float *x, const float *blk, const float *prm, float *y, CcAres p) {
    const int s = (int)(blockIdx.x + blockIdx.y * gridDim.x), tid = (int)threadIdx.x;
    if (s >= p.S) return;   /* uniform over the block */
    __shared__ double rms[256], rdt[256];
    __shared__ float sc[16];
    const float *w = prm + p.w_off;
    for (int e = 0; e <= p.nb; e++) {
        const float *v = e < p.nb ? blk + p.b_off + (size_t)s * p.b_row + (size_t)e * p.D : x + p.x_off + (size_t)s * p.x_row;
        double ms = 0, dt = 0;
        for (int d = tid; d < p.D; d += 256) { const double xv = v[d]; ms += xv * xv; dt += xv * (double)w[d]; }
        rms[tid] = ms; rdt[tid] = dt;
        __syncthreads();
        for (int off = 128; off > 0; off >>= 1) {
            if (tid < off) { rms[tid] += rms[tid + off]; rdt[tid] += rdt[tid + off]; }
            __syncthreads();
        }
        if (tid == 0) sc[e] = (float)(rdt[0] / sqrt(rms[0] / p.D + (double)p.eps));
        __syncthreads();
    }
    if (tid == 0) {
        float m = sc[0];
        for (int e = 1; e <= p.nb; e++) if (sc[e] > m) m = sc[e];
        float sum = 0.f;
        for (int e = 0; e <= p.nb; e++) { sc[e] = expf(sc[e] - m); sum += sc[e]; }
        for (int e = 0; e <= p.nb; e++) sc[e] /= sum;
    }
    __syncthreads();
    const float *xb = x + p.x_off + (size_t)s * p.x_row, *bb = blk + p.b_off + (size_t)s * p.b_row;
    float *yb = y + p.y_off + (size_t)s * p.y_row;
    for (int d = tid; d < p.D; d += 256) {
        float a = 0.f;
        for (int e = 0; e < p.nb; e++) a = __fadd_rn(a, __fmul_rn(sc[e], bb[(size_t)e * p.D + d]));
        yb[d] = __fadd_rn(a, __fmul_rn(sc[p.nb], xb[d]));
    }
}
__global__ static void ccg_situ_k(const float *g, const float *u, float *y, CcSitu p) {
    const long i = (long)(blockIdx.x + blockIdx.y * gridDim.x) * 256 + threadIdx.x;
    if (i >= p.n) return;
    const float gv = g[p.g_off + i], uv = u[p.u_off + i];
    const float sg = 1.f / (1.f + expf(-gv));
    y[p.y_off + i] = __fmul_rn(__fmul_rn(__fmul_rn(__fmul_rn(p.b1, tanhf(gv / p.b1)), sg), p.b2), tanhf(uv / p.b2));
}

static int cc_ares_mix(CcBuf *x, CcBuf *blk, CcBuf *prm, CcBuf *y, const CcAres *p) {
    CcCtx *c = live();
    if (!c || !p || !x || !prm || !y || p->S < 0 || p->D < 1 || p->nb < 0 || p->nb > 15 || (p->nb > 0 && !blk)) return 0;
    if (p->S == 0) return 1;
    dim3 g; cg_grid(p->S, &g);
    coli_launch(ccg_ares_mix_k, g, 256, 0, c->stream, fp(x), p->nb > 0 ? fp(blk) : fp(x), fp(prm), fpw(y), *p);
    return launched(c, "ares mix");
}
static int cc_situ(CcBuf *gb, CcBuf *ub, CcBuf *y, const CcSitu *p) {
    CcCtx *c = live();
    if (!c || !p || !gb || !ub || !y || p->n < 0 || p->b1 == 0.f || p->b2 == 0.f) return 0;
    if (p->n == 0) return 1;
    dim3 g; cg_grid(((long)p->n + 255) / 256, &g);
    coli_launch(ccg_situ_k, g, 256, 0, c->stream, fp(gb), fp(ub), fpw(y), *p);
    return launched(c, "situ");
}
