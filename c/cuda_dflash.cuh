/* cuda_dflash.cuh -- DFlash2's drafter's ops on the CUDA chain (cuda_chain.h's dfconv, dftopk and dftap),
 * included once by cuda_chain.cu after cuda_k3c.cuh.
 *
 * DFlash2's drafter puts one before and one after each attention and MLP sublayer of every layer:
 *   y[r][c] = k0[r][c] x[r][c] + k1[r][c] x[r-1][c],
 * x[r-1] taken as 0 at the first row of each block of B rows (the block is the drafter's unit: the
 * verified token, then the masks). Each tap's coefficient is a learned base (prm[b_off + t*C + c])
 * with a correction the row's own projection gives one group of channels at a time:
 *   kp[kp_off + r*kp_row + t*kp_tap + g*kp_g], g = c / gs (flags 1: g = c % (C / gs)),
 *   k_t = base + kp (flags 2: base * (1 + kp)).
 * One thread an element, no reduction: every output is the two products and their sum in the
 * order written, the CPU's arithmetic with its multiply and add apart. y must not be x (row r
 * reads row r - 1 of x). */

__global__ static void ccf_dfconv_k(const float *__restrict__ x, const float *__restrict__ kp,
                                    const float *__restrict__ w, float *__restrict__ y, CcDfConv p) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)p.n * p.C) return;
    const int r = (int)(i / (size_t)p.C), c = (int)(i % (size_t)p.C), G = p.C / p.gs;
    const int g = (p.flags & 1) ? c % G : c / p.gs;
    const float *kr = kp + p.kp_off + (size_t)r * p.kp_row + (size_t)g * p.kp_g;
    const float b0 = w[p.b_off + c], b1 = w[p.b_off + p.C + c];
    const float c0 = kr[0], c1 = kr[p.kp_tap];
    const float k0 = (p.flags & 2) ? b0 * (1.f + c0) : b0 + c0, k1 = (p.flags & 2) ? b1 * (1.f + c1) : b1 + c1;
    const float xv = x[p.x_off + (size_t)r * p.x_row + c];
    const float xp = r % p.B ? x[p.x_off + (size_t)(r - 1) * p.x_row + c] : 0.f;
    const float a = __fmul_rn(k0, xv), b = __fmul_rn(k1, xp);
    y[p.y_off + (size_t)r * p.y_row + c] = __fadd_rn(a, b);
}
static int cc_dfconv(CcBuf *x, CcBuf *kp, CcBuf *prm, CcBuf *y, const CcDfConv *p) {
    CcCtx *c = live();
    if (!c || !x || !kp || !prm || !y || x == y || !p || p->n < 1 || p->C < 1 || p->B < 1 || p->gs < 1 || p->C % p->gs) return 0;
    const size_t G = (size_t)(p->C / p->gs), last = (size_t)(p->n - 1);
    if ((p->x_off + last * p->x_row + p->C) * 4 > x->bytes || (p->y_off + last * p->y_row + p->C) * 4 > y->bytes ||
        (p->kp_off + last * p->kp_row + (G - 1) * p->kp_g + p->kp_tap + 1) * 4 > kp->bytes ||
        ((size_t)p->b_off + 2 * (size_t)p->C) * 4 > prm->bytes) return 0;
    coli_launch(ccf_dfconv_k, grid1((size_t)p->n * p->C, 256), 256, 0, c->stream, fp(x), fp(kp), fp(prm), fpw(y), *p);
    return launched(c, "dfconv");
}

/* The top K (at most 32) of each of R rows of V scores, best first, a tie to the lower column: row r's
 * values at out[o_off + r*o_row + j], their columns (ints) at out[o_off + r*o_row + K + j]. Two passes:
 * a block a slice of a row (splits of them) keeps its slice's K best in sc, then a block a row merges
 * the slices' candidates. In a block each thread keeps its K best in order, then K rounds take the
 * block's best head. The selector reads its candidates from them (the drafter's logits stay on the card). */
__device__ static inline int ccf_better(float a, int ia, float b, int ib) { return a > b || (a == b && ia < ib); }
/* the K best of n candidates (value at v[i * vs], column at col(i)) into o (values) and o + K (columns) */
template <int NT>
__device__ static void ccf_topk_block(const float *v, int vs, const int *ci, int cs, int base, int n, int K, float *o) {
    const int tid = threadIdx.x;
    float lv[32]; int li[32]; int m = 0;
    for (int i = tid; i < n; i += NT) {
        const float x = v[(size_t)i * vs]; const int c = ci ? ci[(size_t)i * cs] : base + i;
        if (m == K && !ccf_better(x, c, lv[K - 1], li[K - 1])) continue;
        int at = m < K ? m++ : K - 1;
        while (at > 0 && ccf_better(x, c, lv[at - 1], li[at - 1])) { lv[at] = lv[at - 1]; li[at] = li[at - 1]; at--; }
        lv[at] = x; li[at] = c;
    }
    __shared__ float sv[NT]; __shared__ int si[NT], st[NT];
    __shared__ int win;
    int head = 0;
    for (int k = 0; k < K; k++) {
        sv[tid] = head < m ? lv[head] : -INFINITY; si[tid] = head < m ? li[head] : 0x7fffffff; st[tid] = tid;
        __syncthreads();
        for (int off = NT / 2; off > 0; off >>= 1) {
            if (tid < off && ccf_better(sv[tid + off], si[tid + off], sv[tid], si[tid])) {
                sv[tid] = sv[tid + off]; si[tid] = si[tid + off]; st[tid] = st[tid + off];
            }
            __syncthreads();
        }
        if (tid == 0) { o[k] = sv[0]; o[K + k] = __int_as_float(si[0]); win = st[0]; }
        __syncthreads();
        if (tid == win) head++;
        __syncthreads();
    }
}
__global__ static void __launch_bounds__(256) ccf_dftopk1_k(const float *__restrict__ x, float *__restrict__ sc, CcDfTopk p) {
    const int r = blockIdx.x, g = blockIdx.y, per = (p.V + p.splits - 1) / p.splits, lo = g * per;
    const int n = lo >= p.V ? 0 : (p.V - lo < per ? p.V - lo : per);
    ccf_topk_block<256>(x + p.x_off + (size_t)r * p.x_row + lo, 1, NULL, 0, lo, n, p.K, sc + ((size_t)r * p.splits + g) * 2 * p.K);
}
__global__ static void __launch_bounds__(256) ccf_dftopk2_k(const float *__restrict__ sc, float *__restrict__ out, CcDfTopk p) {
    const int r = blockIdx.x, K = p.K;
    const float *row = sc + (size_t)r * p.splits * 2 * K;   /* slice g's values at g 2K .., its columns at g 2K + K .. */
    /* candidate i: slice i / K, place i % K */
    __shared__ float cv[256 * 32 / 4]; __shared__ int cc[256 * 32 / 4];
    const int n = p.splits * K;
    for (int i = threadIdx.x; i < n; i += 256) {
        const int g = i / K, k = i % K;
        cv[i] = row[(size_t)g * 2 * K + k]; cc[i] = __float_as_int(row[(size_t)g * 2 * K + K + k]);
    }
    __syncthreads();
    ccf_topk_block<256>(cv, 1, cc, 1, 0, n, K, out + p.o_off + (size_t)r * p.o_row);
}
static int cc_dftopk(CcBuf *x, CcBuf *sc, CcBuf *out, const CcDfTopk *p) {
    CcCtx *c = live();
    if (!c || !x || !sc || !out || !p || p->R < 1 || p->V < 1 || p->K < 1 || p->K > 32 || p->K > p->V || p->o_row < 2 * p->K ||
        p->splits < 1 || p->splits * p->K > 256 * 32 / 4) return 0;
    if ((p->x_off + (size_t)(p->R - 1) * p->x_row + p->V) * 4 > x->bytes ||
        (p->o_off + (size_t)(p->R - 1) * p->o_row + 2 * (size_t)p->K) * 4 > out->bytes ||
        (size_t)p->R * p->splits * 2 * p->K * 4 > sc->bytes) return 0;
    coli_launch(ccf_dftopk1_k, dim3((unsigned)p->R, (unsigned)p->splits), 256, 0, c->stream, fp(x), fpw(sc), *p);
    if (!launched(c, "dftopk")) return 0;
    coli_launch(ccf_dftopk2_k, (unsigned)p->R, 256, 0, c->stream, fp(sc), fpw(out), *p);
    return launched(c, "dftopk merge");
}

/* The drafter's context rows from a target layer's: y[y_off + r*y_row + c] = the sum over b < C of
 * x[x_off + r*x_row + b*D + c] (C 1: a copy; Qwen3.8-Flash-Next's four hyper-connection streams: their
 * sum, which the drafter's hidden_norm makes their mean). The streams added in order, rounded apart. */
__global__ static void ccf_dftap_k(const float *__restrict__ x, float *__restrict__ y, CcDfTap p) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)p.n * p.D) return;
    const int r = (int)(i / (size_t)p.D), c = (int)(i % (size_t)p.D);
    const float *xr = x + p.x_off + (size_t)r * p.x_row + c;
    float s = xr[0];
    for (int b = 1; b < p.C; b++) s = __fadd_rn(s, xr[(size_t)b * p.D]);
    y[p.y_off + (size_t)r * p.y_row + c] = s;
}
static int cc_dftap(CcBuf *x, CcBuf *y, const CcDfTap *p) {
    CcCtx *c = live();
    if (!c || !x || !y || x == y || !p || p->n < 1 || p->D < 1 || p->C < 1) return 0;
    if ((p->x_off + (size_t)(p->n - 1) * p->x_row + (size_t)p->C * p->D) * 4 > x->bytes ||
        (p->y_off + (size_t)(p->n - 1) * p->y_row + p->D) * 4 > y->bytes) return 0;
    coli_launch(ccf_dftap_k, grid1((size_t)p->n * p->D, 256), 256, 0, c->stream, fp(x), fpw(y), *p);
    return launched(c, "dftap");
}
static int cc_dflash(void) { return 1; }
