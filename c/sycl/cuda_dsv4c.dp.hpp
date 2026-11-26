/* cuda_dsv4c.dp.hpp -- DeepSeek V4.1 Flash's (and V4's) layer ops on the SYCL chain, included by
 * cuda_chain.dp.cpp (its CcCtx, live(), fp(), fpw(), grid1(), dim3(), launched(), CC_K, CC_KSG).
 * cuda_dsv4c.cuh ported by hand, kernel for kernel: the twins of the Vulkan chain's
 * shaders/chain_mhc.comp and shaders/chain_dsv4.comp (with chain_dsv4_pa.comp for a prompt
 * chunk's attention), the same arithmetic in the same order (the same reduction trees over
 * 256 work-items, the same loops, the same sentinels), so the CPU's reference
 * (hyper_connections.h, sparse_attn.h, deepseek_v41.c) holds them within the bounds
 * tests/test_cuda_dsv4c.cu holds the .cuh to:
 *   cc_mhc          the hyper-connections' split (Sinkhorn), collapse, write back, mean, and
 *                   the clamped SwiGLU (CcMhc, modes CC_MHC_*);
 *   cc_ds_attn      sparse attention with a sink over a per-row list: a decode row's (head,
 *                   row) a work-group, a prompt chunk's rows from CC_DS_PA_MIN in three passes
 *                   over slices of rows (scores on tiles of the listed rows, the weights and
 *                   the denominator in list order, the output on tiles), the same sums;
 *   cc_ds_rope      RoPE on interleaved pairs, inverse and bf16 flags;
 *   cc_ds_compress  the compressor's rolling group and its softmax pooling;
 *   cc_ds_score     the indexer's scores, a work-item a (column, head) on a grid, the heads
 *                   joined in their order;
 *   cc_ds_cand      the candidate blocks;
 *   cc_ds_topk      the top-k of each row's scores, ties to the lower column;
 *   cc_ds_engram    the engram gate into each stream.
 * The mapping: a CUDA block is a work-group of 256 (blockIdx.x/y/z = get_group(2/1/0), as
 * dim3() orders the range), __syncthreads a work-group barrier, __shared__ arrays
 * group_local_memory_for_overwrite, an extern __shared__ array the launch's dynamic local
 * memory (CC_KSG's slm_; the kernels read no sub-group, SG is unused). No kernel here uses a
 * warp shuffle, a ballot or an atomic: every reduction is the .cuh's tree in local memory, so
 * the sub-group width (16 or 32) changes nothing. Two kernels share global memory between
 * work-items across a barrier (the compressor's ring, the candidates' mask): their barriers
 * fence global memory too (__syncthreads did); the others fence local memory alone. expf is
 * CCD_EXP: sycl::exp, the precise built-in (the bounds here are tighter than the rest of the
 * chain's, where expf became sycl::native::exp). */
#ifndef CUDA_DSV4C_DP_HPP
#define CUDA_DSV4C_DP_HPP

#define CC_DS_NEG_INF (-sycl::bit_cast<float>(0x7f800000))
#define CC_DS_POS_INF (sycl::bit_cast<float>(0x7f800000))
#define CCD_EXP(x) sycl::exp(x)   /* the .cuh's expf */
/* a work-group's barrier: over local memory, or over local and global memory where work-items
 * hand each other global memory (__syncthreads covered both) */
#define CCD_SYNC(it) (it).barrier(sycl::access::fence_space::local_space)
#define CCD_SYNC_G(it) (it).barrier(sycl::access::fence_space::global_and_local)
#define CCD_LOCAL(T) (*sycl::ext::oneapi::group_local_memory_for_overwrite<T>(sycl::ext::oneapi::this_work_item::get_work_group<3>()))

static inline float ccd_bf16r(float f) {   /* round to nearest even, as coli_bf16_round */
    unsigned u = sycl::bit_cast<unsigned>(f);
    if ((u & 0x7f800000u) != 0x7f800000u) u += 0x7fffu + ((u >> 16) & 1u);
    return sycl::bit_cast<float>(u & 0xffff0000u);
}
static inline float ccd_sig(float v) {
    if (v >= 0.f) { float d = CCD_EXP(-v); return 1.f / (1.f + d); }
    float g = CCD_EXP(v);
    return g / (1.f + g);
}
static inline unsigned ccd_okey(float f) {
    unsigned u = sycl::bit_cast<unsigned>(f);
    if (u == 0x80000000u) u = 0u;
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
/* a work-group's sum of v (256 work-items), every work-item gets it */
static unsigned ccd_wsum(unsigned v, unsigned *cnt, int tid) {
    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    cnt[tid] = v;
    CCD_SYNC(it);
    for (int off = 128; off > 0; off >>= 1) {
        if (tid < off) cnt[tid] += cnt[tid + off];
        CCD_SYNC(it);
    }
    unsigned t = cnt[0];
    CCD_SYNC(it);
    return t;
}
/* exclusive scan in work-item order */
static unsigned ccd_wscan(unsigned v, unsigned *cnt, int tid) {
    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    cnt[tid] = v;
    CCD_SYNC(it);
    for (int off = 1; off < 256; off <<= 1) {
        unsigned a = tid >= off ? cnt[tid - off] : 0u;
        CCD_SYNC(it);
        cnt[tid] += a;
        CCD_SYNC(it);
    }
    unsigned incl = cnt[tid];
    CCD_SYNC(it);
    return incl - v;
}

/* ---- mHC ------------------------------------------------------------------------------ */
static void ccd_mhc_split_k(const float *x, const float *mx, float *hp, const float *prm, CcMhc p) {
    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto &red = CCD_LOCAL(float[256]);
    auto &ms = CCD_LOCAL(float[96]);
    const int s = it.get_group(2), tid = it.get_local_id(2), H = p.H, HD = H * p.D, nm = (2 + H) * H;
    const int xb = p.x_off + s * p.x_row;
    float a = 0.f;
    for (int i = tid; i < HD; i += 256) { float v = x[xb + i]; a += v * v; }
    red[tid] = a;
    CCD_SYNC(it);
    for (int off = 128; off > 0; off >>= 1) {
        if (tid < off) red[tid] += red[tid + off];
        CCD_SYNC(it);
    }
    const float r = 1.f / sycl::sqrt(red[0] / (float)HD + p.eps);
    for (int i = tid; i < nm; i += 256) ms[i] = mx[p.m_off + s * p.m_row + i] * r;
    CCD_SYNC(it);
    if (tid != 0) return;
    const int hb = p.hp_off + s * p.hp_row, pb = p.prm_off;
    for (int i = 0; i < H; i++) {
        hp[hb + i] = ccd_sig(ms[i] * prm[pb] + prm[pb + 3 + i]) + p.hc_eps;
        hp[hb + H + i] = 2.f * ccd_sig(ms[H + i] * prm[pb + 1] + prm[pb + 3 + H + i]);
    }
    /* the comb matrix in registers (H <= 8): the same operations in the same order as on hp[] itself, one write back --
     * each step was a dependent global access (~0.44 ms a call on a decode row, profiled) */
    const int cb = hb + 2 * H, mo = 2 * H;
    float cm[64];
    for (int row = 0; row < H; row++) {
        float mxv = -3.0e38f;
        for (int col = 0; col < H; col++) {
            float v = ms[mo + row * H + col] * prm[pb + 2] + prm[pb + 3 + mo + row * H + col];
            cm[row * H + col] = v;
            if (v > mxv) mxv = v;
        }
        float sum = 0.f;
        for (int col = 0; col < H; col++) { float v = CCD_EXP(cm[row * H + col] - mxv); cm[row * H + col] = v; sum += v; }
        for (int col = 0; col < H; col++) cm[row * H + col] = cm[row * H + col] / sum + p.hc_eps;
    }
    float sums[8];
    for (int col = 0; col < H; col++) { float sm = 0.f; for (int row = 0; row < H; row++) sm += cm[row * H + col]; sums[col] = sm; }
    for (int row = 0; row < H; row++) for (int col = 0; col < H; col++) cm[row * H + col] /= sums[col] + p.hc_eps;
    for (int it2 = 1; it2 < p.iters; it2++) {
        for (int row = 0; row < H; row++) { float sm = 0.f; for (int col = 0; col < H; col++) sm += cm[row * H + col]; sums[row] = sm; }
        for (int row = 0; row < H; row++) for (int col = 0; col < H; col++) cm[row * H + col] /= sums[row] + p.hc_eps;
        for (int col = 0; col < H; col++) { float sm = 0.f; for (int row = 0; row < H; row++) sm += cm[row * H + col]; sums[col] = sm; }
        for (int row = 0; row < H; row++) for (int col = 0; col < H; col++) cm[row * H + col] /= sums[col] + p.hc_eps;
    }
    for (int k = 0; k < H * H; k++) hp[cb + k] = cm[k];
}
/* ccd_mhc_split_k with H a constant (DeepSeek V4.1's hc_mult 4, and 2 / 8): every loop unrolls, so the comb matrix and
 * the sums stay in registers -- with a run-time H, cm[64] / sums[8] are indexed dynamically and live in private
 * (scratch) memory, each Sinkhorn step a dependent memory access (measured: 175 us a call, 10.6 % of a decode's device
 * time on 1-4 work-groups). The operations are ccd_mhc_split_k's, one for one, in its order (-fp-model=precise: no
 * reassociation), so the bits are its bits. COLI_SYCL_MHC_GENERIC=1: ccd_mhc_split_k. */
template <int H>
static void ccd_mhc_split_t(const float *x, const float *mx, float *hp, const float *prm, CcMhc p) {
    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto &red = CCD_LOCAL(float[256]);
    auto &ms = CCD_LOCAL(float[96]);
    constexpr int nm = (2 + H) * H;
    const int s = it.get_group(2), tid = it.get_local_id(2), HD = H * p.D;
    const int xb = p.x_off + s * p.x_row;
    float a = 0.f;
    for (int i = tid; i < HD; i += 256) { float v = x[xb + i]; a += v * v; }
    red[tid] = a;
    CCD_SYNC(it);
    for (int off = 128; off > 0; off >>= 1) {
        if (tid < off) red[tid] += red[tid + off];
        CCD_SYNC(it);
    }
    const float r = 1.f / sycl::sqrt(red[0] / (float)HD + p.eps);
    for (int i = tid; i < nm; i += 256) ms[i] = mx[p.m_off + s * p.m_row + i] * r;
    CCD_SYNC(it);
    if (tid != 0) return;
    const int hb = p.hp_off + s * p.hp_row, pb = p.prm_off;
    float pr[3 + nm];
#pragma unroll
    for (int i = 0; i < 3 + nm; i++) pr[i] = prm[pb + i];
#pragma unroll
    for (int i = 0; i < H; i++) {
        hp[hb + i] = ccd_sig(ms[i] * pr[0] + pr[3 + i]) + p.hc_eps;
        hp[hb + H + i] = 2.f * ccd_sig(ms[H + i] * pr[1] + pr[3 + H + i]);
    }
    const int cb = hb + 2 * H;
    constexpr int mo = 2 * H;
    float cm[H * H];
#pragma unroll
    for (int row = 0; row < H; row++) {
        float mxv = -3.0e38f;
#pragma unroll
        for (int col = 0; col < H; col++) {
            float v = ms[mo + row * H + col] * pr[2] + pr[3 + mo + row * H + col];
            cm[row * H + col] = v;
            if (v > mxv) mxv = v;
        }
        float sum = 0.f;
#pragma unroll
        for (int col = 0; col < H; col++) { float v = CCD_EXP(cm[row * H + col] - mxv); cm[row * H + col] = v; sum += v; }
#pragma unroll
        for (int col = 0; col < H; col++) cm[row * H + col] = cm[row * H + col] / sum + p.hc_eps;
    }
    float sums[H];
#pragma unroll
    for (int col = 0; col < H; col++) { float sm = 0.f;
#pragma unroll
        for (int row = 0; row < H; row++) sm += cm[row * H + col]; sums[col] = sm; }
#pragma unroll
    for (int row = 0; row < H; row++)
#pragma unroll
        for (int col = 0; col < H; col++) cm[row * H + col] /= sums[col] + p.hc_eps;
    for (int it2 = 1; it2 < p.iters; it2++) {
#pragma unroll
        for (int row = 0; row < H; row++) { float sm = 0.f;
#pragma unroll
            for (int col = 0; col < H; col++) sm += cm[row * H + col]; sums[row] = sm; }
#pragma unroll
        for (int row = 0; row < H; row++)
#pragma unroll
            for (int col = 0; col < H; col++) cm[row * H + col] /= sums[row] + p.hc_eps;
#pragma unroll
        for (int col = 0; col < H; col++) { float sm = 0.f;
#pragma unroll
            for (int row = 0; row < H; row++) sm += cm[row * H + col]; sums[col] = sm; }
#pragma unroll
        for (int row = 0; row < H; row++)
#pragma unroll
            for (int col = 0; col < H; col++) cm[row * H + col] /= sums[col] + p.hc_eps;
    }
#pragma unroll
    for (int k = 0; k < H * H; k++) hp[cb + k] = cm[k];
}
/* ccd_mhc_split_t for H*H <= 16 (H 2 and 4), the Sinkhorn spread over a sub-group: lane l < H*H holds the comb's element
 * (l / H, l % H); every row or column sum gathers its H elements with select_from_group in ccd_mhc_split_t's order (from
 * 0.f, index ascending) and each lane divides its own element -- the same operations on the same values, so the same
 * bits, with the ~2*iters*H*H dependent IEEE divisions (-fp-model=precise) 16 at a time instead of one work-item's
 * (measured: 91 us a call on 4 rows). The pre and post weights go one a lane too. COLI_SYCL_MHC_GENERIC=1: the generic. */
/* B 1: the streams' sum of squares with its loads sixteen at a time -- a work-item's HD / 256 elements (80 at
 * DeepSeek V4.1's 4 x 5120) were a load, then an add, eighty times in a row, each load waited for; now sixteen
 * independent loads, then their sixteen adds in the same order from the same start, so the same sum to the bit.
 * COLI_SYCL_MHC_BATCH=0: B 0, the loop as it was. */
template <int H, int B = 0>
static void ccd_mhc_split_sg(const float *x, const float *mx, float *hp, const float *prm, CcMhc p) {
    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
    auto &red = CCD_LOCAL(float[256]);
    auto &ms = CCD_LOCAL(float[96]);
    constexpr int nm = (2 + H) * H, HH = H * H;
    static_assert(HH <= 16, "the comb in one sub-group");
    const int s = it.get_group(2), tid = it.get_local_id(2), HD = H * p.D;
    const int xb = p.x_off + s * p.x_row;
    float a = 0.f;
    if (B) {
        for (int i0 = tid; i0 < HD; i0 += 16 * 256) {
            float xv[16];
#pragma unroll
            for (int k = 0; k < 16; k++) { const int i = i0 + 256 * k; xv[k] = i < HD ? x[xb + i] : 0.f; }
#pragma unroll
            for (int k = 0; k < 16; k++) if (i0 + 256 * k < HD) { float v = xv[k]; a += v * v; }
        }
    } else
        for (int i = tid; i < HD; i += 256) { float v = x[xb + i]; a += v * v; }
    red[tid] = a;
    CCD_SYNC(it);
    for (int off = 128; off > 0; off >>= 1) {
        if (tid < off) red[tid] += red[tid + off];
        CCD_SYNC(it);
    }
    const float r = 1.f / sycl::sqrt(red[0] / (float)HD + p.eps);
    for (int i = tid; i < nm; i += 256) ms[i] = mx[p.m_off + s * p.m_row + i] * r;
    CCD_SYNC(it);
    if (sg.get_group_linear_id() != 0) return;   /* the first sub-group (16 or 32 lanes) goes on, whole */
    const int l = (int)sg.get_local_linear_id(), lc = l < HH ? l : 0, row = lc / H, col = lc % H;
    const int hb = p.hp_off + s * p.hp_row, pb = p.prm_off;
    if (l < H) {
        hp[hb + l] = ccd_sig(ms[l] * prm[pb] + prm[pb + 3 + l]) + p.hc_eps;
        hp[hb + H + l] = 2.f * ccd_sig(ms[H + l] * prm[pb + 1] + prm[pb + 3 + H + l]);
    }
    constexpr int mo = 2 * H;
    float c = ms[mo + lc] * prm[pb + 2] + prm[pb + 3 + mo + lc];
    float mxv = -3.0e38f;
#pragma unroll
    for (int k = 0; k < H; k++) { const float v = sycl::select_from_group(sg, c, row * H + k); if (v > mxv) mxv = v; }
    c = CCD_EXP(c - mxv);
    float sum = 0.f;
#pragma unroll
    for (int k = 0; k < H; k++) sum += sycl::select_from_group(sg, c, row * H + k);
    c = c / sum + p.hc_eps;
    float sm = 0.f;
#pragma unroll
    for (int k = 0; k < H; k++) sm += sycl::select_from_group(sg, c, k * H + col);
    c /= sm + p.hc_eps;
    for (int it2 = 1; it2 < p.iters; it2++) {
        sm = 0.f;
#pragma unroll
        for (int k = 0; k < H; k++) sm += sycl::select_from_group(sg, c, row * H + k);
        c /= sm + p.hc_eps;
        sm = 0.f;
#pragma unroll
        for (int k = 0; k < H; k++) sm += sycl::select_from_group(sg, c, k * H + col);
        c /= sm + p.hc_eps;
    }
    if (l < HH) hp[hb + 2 * H + l] = c;
}
static int ccd_mhc_generic(void) {   /* read at each launch: the test runs both kernels in one process */
    const char *e = getenv("COLI_SYCL_MHC_GENERIC");
    return e && atoi(e) > 0;
}
static int ccd_mhc_batch(void) {   /* read at each launch, as ccd_mhc_generic */
    const char *e = getenv("COLI_SYCL_MHC_BATCH");
    return !(e && *e == '0');
}
static void ccd_mhc_ew_k(int mode, const float *x, const float *mx, const float *hp, float *y, CcMhc p) {
    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const long long i = (long long)it.get_group(2) * it.get_local_range(2) + it.get_local_id(2);
    if (mode == CC_SWIGLU_CLAMP) {
        if (i >= p.n) return;
        float g = x[p.x_off + i], u = mx[p.m_off + i];
        g = g > p.lim ? p.lim : g;
        u = u < -p.lim ? -p.lim : (u > p.lim ? p.lim : u);
        y[p.y_off + i] = (g / (1.f + CCD_EXP(-g))) * u;
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
    CC_OWN(c, "cc_mhc", x, m, hp, prm, y);
    if (!c || !p || !x) return 0;
    const float *xp = fp(x), *mp = fp(m), *pp_ = fp(prm); const CcMhc pp = *p;
    if (mode == CC_MHC_SPLIT) {
        if (!m || !hp || !prm || p->S < 1 || p->H < 1 || p->H > 8 || (2 + p->H) * p->H > 96) return 0;
        float *hw = fpw(hp);
        const int bt = ccd_mhc_batch();
        if (!ccd_mhc_generic() && p->H == 4 && bt) CC_KSG(c, dim3((unsigned)p->S), 256, 0, ccd_mhc_split_sg<4, 1>(xp, mp, hw, pp_, pp));
        else if (!ccd_mhc_generic() && p->H == 4) CC_KSG(c, dim3((unsigned)p->S), 256, 0, ccd_mhc_split_sg<4>(xp, mp, hw, pp_, pp));   /* a fixed sub-group width (16 or 32) */
        else if (!ccd_mhc_generic() && p->H == 2 && bt) CC_KSG(c, dim3((unsigned)p->S), 256, 0, ccd_mhc_split_sg<2, 1>(xp, mp, hw, pp_, pp));
        else if (!ccd_mhc_generic() && p->H == 2) CC_KSG(c, dim3((unsigned)p->S), 256, 0, ccd_mhc_split_sg<2>(xp, mp, hw, pp_, pp));
        else if (!ccd_mhc_generic() && p->H == 8) CC_K(c, dim3((unsigned)p->S), 256, ccd_mhc_split_t<8>(xp, mp, hw, pp_, pp));
        else CC_K(c, dim3((unsigned)p->S), 256, ccd_mhc_split_k(xp, mp, hw, pp_, pp));
        return launched(c, "mhc split");
    }
    size_t n;
    if (mode == CC_SWIGLU_CLAMP) { if (!m || !y || p->n < 1) return 0; n = (size_t)p->n; }
    else if (mode == CC_MHC_POST) { if (!m || !hp || !y || p->S < 1) return 0; n = (size_t)p->S * p->H * p->D; }
    else if (mode == CC_MHC_COLLAPSE || mode == CC_MHC_MEAN) { if (!y || (mode == CC_MHC_COLLAPSE && !hp) || p->S < 1) return 0; n = (size_t)p->S * p->D; }
    else return 0;
    const float *hr = fp(hp); float *yp = fpw(y); const int md = mode;
    CC_K(c, dim3(grid1(n, 256)), 256, ccd_mhc_ew_k(md, xp, mp, hr, yp, pp));
    return launched(c, "mhc");
}

/* ---- a chain-MoE layer's last frame in one launch (cc_moe_tail) ------------------------------
 * The three kernels the frame ran one after the other, a work-item a (row, output) as the first of them:
 *   moe_sum_k (backend_cuda.dp.cpp, cc_moe parts 2)   y = the host's row, then each held slot's w * down, in rank order
 *   cc_ew_k op 0 (ADD)                                br = y + ds
 *   ccd_mhc_ew_k CC_MHC_POST                          xn[j] = sum_k comb[k][j] * xs[k] + post[j] * br, for the H streams
 * each statement theirs, in their order, on the values they read back from memory (a float stored and loaded is the
 * same float), so y, br and xn have their bits (the ops test compares the three buffers byte for byte). Why: the frame
 * is submitted when the host's misses are done and the card is idle, so each of its commands waited for the host's
 * submit before it ran -- 6 to 18 us of idle card before each (a frame dump), ~0.11 ms a layer for 25 us of work. */
static void ccd_moe_tail_k(const CcExpert *tab, const int *ids, const float *w, const float *hid, const float *hy, float *y,
                           const float *ds, float *br, const float *xs, const float *hp, float *xn, CcMoe p, CcMhc q) {
    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int s = it.get_group(1), d = it.get_group(2) * 256 + it.get_local_id(2);
    if (d >= p.D) return;
    /* moe_sum_k, line for line */
    size_t dn = (size_t)(p.lay_S > 0 ? p.lay_S : p.S) * p.K * p.I;
    const int hs = p.lay_s0 + s;
    float v = p.h_off >= 0 ? hy[p.h_off + (size_t)s * p.D + d] : 0.f;
    for (int k0 = 0; k0 < p.K; k0 += 8) {
        int e[8];
        bool on[8];
        float wk[8], hk[8];
        #pragma unroll
        for (int j = 0; j < 8; j++) e[j] = k0 + j < p.K ? ids[p.i_off + s * p.i_row + k0 + j] : -1;
        #pragma unroll
        for (int j = 0; j < 8; j++) on[j] = e[j] >= 0 && tab[p.t_off + e[j]].g;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            wk[j] = on[j] ? w[p.w_off + s * p.w_row + k0 + j] : 0.f;
            hk[j] = on[j] ? hid[dn + (size_t)(hs * p.K + k0 + j) * p.D + d] : 0.f;
        }
        #pragma unroll
        for (int j = 0; j < 8; j++) if (on[j]) v += wk[j] * hk[j];
    }
    y[p.y_off + (size_t)s * p.y_row + d] = v;
    /* cc_ew_k op 0 on S * D floats from offset 0: br = y + ds */
    const float b = v + ds[(size_t)s * p.D + d];
    br[(size_t)s * p.D + d] = b;
    /* ccd_mhc_ew_k CC_MHC_POST, its work-items (s, j, d) for j = 0..H-1 */
    const int H = q.H, D = q.D;
    const int xb = q.x_off + s * q.x_row, hb = q.hp_off + s * q.hp_row;
    for (int j = 0; j < H; j++) {
        float t = 0.f;
        for (int k = 0; k < H; k++) t += hp[hb + 2 * H + k * H + j] * xs[xb + k * D + d];
        t += hp[hb + H + j] * b;
        xn[q.y_off + s * q.y_row + j * D + d] = t;
    }
}
static int ccd_moe_tail_on(void) {   /* read at each launch: the test runs both in one process */
    const char *e = getenv("COLI_SYCL_MOE_TAIL");
    return !(e && *e == '0');
}
/* 1 done, -1 not taken (the caller's cc_moe parts 2, ew ADD and mhc POST), 0 a failure. Taken where cc_moe's parts 2 is
 * moe_sum_k on its own: the tier's formats (fmt < 100: a GGUF's blocks have their own launch), a decode step's rows
 * (no slots sorted by expert: S and lay_S up to CC_MOE_FRAME_ROWS), the buffers of the shapes the three ops are given */
static int cc_moe_tail(CcBuf *tab, CcBuf *ids, CcBuf *w, CcBuf *hid, CcBuf *hy, CcBuf *y, CcBuf *ds, CcBuf *br, CcBuf *xs,
                       CcBuf *hp, CcBuf *xn, const CcMoe *p, const CcMhc *po) {
    CcCtx *c = live();
    CC_OWN(c, "cc_moe_tail", tab, ids, w, hid, hy, y, ds, br, xs, hp, xn);
    if (p) cc_own_table(c, "cc_moe_tail", tab, p->t_off, p->E);
    if (!c) return 0;
    if (!ccd_moe_tail_on() || !tab || !ids || !w || !hid || !y || !ds || !br || !xs || !hp || !xn || !p || !po) return -1;
    const int LS = p->lay_S > 0 ? p->lay_S : p->S;
    if (p->parts != 2 || p->fmt >= 100 || p->S < 1 || p->K < 1 || p->K > CC_ROUTE_KMAX || LS > CC_MOE_FRAME_ROWS || p->lay_S < 0 ||
        p->lay_s0 < 0 || p->lay_s0 > LS - p->S || (p->h_off >= 0 && !hy) || p->D < 1 || p->I < 1) return -1;
    if (po->S != p->S || po->D != p->D || po->H < 1 || po->H > 8 || xs == xn || br == y || br == xn || y == xn) return -1;
    if (cc_moe_hid_floats(LS, p->K, p->D, p->I, p->E) * 4 > hid->bytes) return -1;
    if (((size_t)p->y_off + (size_t)(p->S - 1) * p->y_row + p->D) * 4 > y->bytes || (size_t)p->S * p->D * 4 > ds->bytes ||
        (size_t)p->S * p->D * 4 > br->bytes ||
        ((size_t)po->x_off + (size_t)(p->S - 1) * po->x_row + (size_t)po->H * po->D) * 4 > xs->bytes ||
        ((size_t)po->y_off + (size_t)(p->S - 1) * po->y_row + (size_t)po->H * po->D) * 4 > xn->bytes ||
        ((size_t)po->hp_off + (size_t)(p->S - 1) * po->hp_row + 2 * (size_t)po->H + (size_t)po->H * po->H) * 4 > hp->bytes) return -1;
    const CcExpert *tp = (const CcExpert *)tab->d; const int *ip = (const int *)ids->d;
    const float *wp = fp(w), *hd_ = fp(hid), *hyp = fp(hy), *dsp = fp(ds), *xp = fp(xs), *hpp = fp(hp);
    float *yp = fpw(y), *bp = fpw(br), *xnp = fpw(xn);
    const CcMoe pp = *p; const CcMhc qq = *po;
    CC_K(c, dim3((unsigned)((p->D + 255) / 256), (unsigned)p->S), 256, ccd_moe_tail_k(tp, ip, wp, hd_, hyp, yp, dsp, bp, xp, hpp, xnp, pp, qq));
    return launched(c, "moe sum + add + mhc post");
}

/* ---- sparse attention ------------------------------------------------------------------- */
static inline float ccd_kv(const float *win, const float *cmp, int e, int nwin, int woff, int coff, int hd, int i) {
    return e < nwin ? win[woff + e * hd + i] : cmp[coff + (e - nwin) * hd + i];
}
/* a (head, row) a work-group: chain_dsv4.comp's mode 0; slm the launch's (hd + cnt) floats */
static void ccd_attn_k(const float *q, const float *win, const float *cmp, const int *list,
                       const float *prm, float *out, CcDsAttn p, uint8_t *slm) {
    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    float *sm = (float *)slm;   /* hd queries, then cnt scores */
    auto &red = CCD_LOCAL(float[256]);
    const int h = it.get_group(2), s = it.get_group(1), tid = it.get_local_id(2);
    if (s >= p.S || h >= p.H) return;
    const int hd = p.hd, n = p.cnt, lb = p.l_off + s * p.l_row;
    const int qb = p.q_off + s * p.q_row + h * hd, ob = p.o_off + s * p.o_row + h * hd;
    float *sc = sm + hd;
    for (int i = tid; i < hd; i += 256) sm[i] = q[qb + i];
    CCD_SYNC(it);
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
    CCD_SYNC(it);
    for (int off = 128; off > 0; off >>= 1) {
        if (tid < off && red[tid + off] > red[tid]) red[tid] = red[tid + off];
        CCD_SYNC(it);
    }
    best = red[0];
    CCD_SYNC(it);
    for (int j = tid; j < n; j += 256) { const float v = sc[j]; sc[j] = v > CC_DS_NEG_INF ? CCD_EXP(v - best) : 0.f; }
    CCD_SYNC(it);
    if (tid == 0) {                       /* the denominator in list order, from the sink */
        float den = CCD_EXP(prm[p.sink_off + h] - best);
        for (int j = 0; j < n; j++) if (list[lb + j] >= 0) den += sc[j];
        red[0] = den;
    }
    CCD_SYNC(it);
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
/* ccd_attn_k for HB heads a work-group: the heads share the window and compressed rows (one latent of hd), so each
 * listed row is read once for HB heads instead of once a head. Every (head, row) value goes through ccd_attn_k's
 * operations in its order -- the dot product d += q[i] * k[i] over i, the max, the exp, the denominator in list order
 * from the sink, the output acc += w * kv over the list in order, / den -- so the bits are ccd_attn_k's.
 * slm: HB*hd queries, then HB*cnt scores. */
#define CCD_ATTN_HB 8
static void ccd_attn_hb_k(const float *q, const float *win, const float *cmp, const int *list,
                          const float *prm, float *out, CcDsAttn p, uint8_t *slm) {
    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    float *sq = (float *)slm;                 /* [HB][hd] */
    auto &red = CCD_LOCAL(float[CCD_ATTN_HB * 256]);
    auto &dens = CCD_LOCAL(float[CCD_ATTN_HB]);
    const int h0 = it.get_group(2) * CCD_ATTN_HB, s = it.get_group(1), tid = it.get_local_id(2);
    if (s >= p.S || h0 >= p.H) return;
    const int nh = p.H - h0 < CCD_ATTN_HB ? p.H - h0 : CCD_ATTN_HB;
    const int hd = p.hd, n = p.cnt, lb = p.l_off + s * p.l_row;
    float *sc = sq + CCD_ATTN_HB * hd;        /* [HB][cnt] */
    for (int k = tid; k < nh * hd; k += 256) sq[k] = q[p.q_off + s * p.q_row + h0 * hd + k];
    CCD_SYNC(it);
    float best[CCD_ATTN_HB];
#pragma unroll
    for (int hh = 0; hh < CCD_ATTN_HB; hh++) best[hh] = -1.0e30f;
    for (int j = tid; j < n; j += 256) {
        const int e = list[lb + j];
        float d[CCD_ATTN_HB];
#pragma unroll
        for (int hh = 0; hh < CCD_ATTN_HB; hh++) d[hh] = 0.f;
        if (e >= 0) {
            const float *kr = e < p.nwin ? win + p.w_off + e * hd : cmp + p.c_off + (e - p.nwin) * hd;
            for (int i = 0; i < hd; i++) {
                const float kv = kr[i];
#pragma unroll
                for (int hh = 0; hh < CCD_ATTN_HB; hh++) if (hh < nh) d[hh] += sq[hh * hd + i] * kv;
            }
        }
#pragma unroll
        for (int hh = 0; hh < CCD_ATTN_HB; hh++) {
            if (hh >= nh) continue;
            float v = CC_DS_NEG_INF;
            if (e >= 0) { v = d[hh] * p.scale; if (v > best[hh]) best[hh] = v; }
            sc[hh * n + j] = v;
        }
    }
#pragma unroll
    for (int hh = 0; hh < CCD_ATTN_HB; hh++) red[hh * 256 + tid] = best[hh];
    CCD_SYNC(it);
    for (int off = 128; off > 0; off >>= 1) {
        if (tid < off)
#pragma unroll
            for (int hh = 0; hh < CCD_ATTN_HB; hh++) if (red[hh * 256 + tid + off] > red[hh * 256 + tid]) red[hh * 256 + tid] = red[hh * 256 + tid + off];
        CCD_SYNC(it);
    }
#pragma unroll
    for (int hh = 0; hh < CCD_ATTN_HB; hh++) best[hh] = red[hh * 256];
    CCD_SYNC(it);
    for (int j = tid; j < n; j += 256)
#pragma unroll
        for (int hh = 0; hh < CCD_ATTN_HB; hh++) {
            if (hh >= nh) continue;
            const float v = sc[hh * n + j];
            sc[hh * n + j] = v > CC_DS_NEG_INF ? CCD_EXP(v - best[hh]) : 0.f;
        }
    CCD_SYNC(it);
    if (tid < nh) {                           /* a head's denominator in list order, from the sink: ccd_attn_k's */
        float den = CCD_EXP(prm[p.sink_off + h0 + tid] - best[tid]);
        for (int j = 0; j < n; j++) if (list[lb + j] >= 0) den += sc[tid * n + j];
        dens[tid] = den;
    }
    CCD_SYNC(it);
    for (int i = tid; i < hd; i += 256) {
        float acc[CCD_ATTN_HB];
#pragma unroll
        for (int hh = 0; hh < CCD_ATTN_HB; hh++) acc[hh] = 0.f;
        for (int j = 0; j < n; j++) {
            const int e = list[lb + j];
            if (e < 0) continue;
            const float kv = ccd_kv(win, cmp, e, p.nwin, p.w_off, p.c_off, hd, i);
#pragma unroll
            for (int hh = 0; hh < CCD_ATTN_HB; hh++) {
                if (hh >= nh) continue;
                float w = sc[hh * n + j];
                if (p.flags & 1) w = ccd_bf16r(w);
                acc[hh] += w * kv;
            }
        }
#pragma unroll
        for (int hh = 0; hh < CCD_ATTN_HB; hh++) {
            if (hh >= nh) continue;
            const float o = acc[hh] / dens[hh];
            out[p.o_off + s * p.o_row + (h0 + hh) * hd + i] = (p.flags & 1) ? ccd_bf16r(o) : o;
        }
    }
}
/* a prompt chunk's rows (chain_dsv4_pa.comp): the scores of 64 heads a work-group on tiles of
 * 64 listed rows, in chunks of 32 floats */
static void ccd_pa_scores_k(const float *q, const float *win, const float *cmp, const int *list,
                            float *scr, CcDsAttn p) {
    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto &sq = CCD_LOCAL(float[64 * 32]);
    auto &sk = CCD_LOCAL(float[64 * 33]);
    auto &se = CCD_LOCAL(int[64]);
    const int s = it.get_group(2), hg = it.get_group(1) * 64, tid = it.get_local_id(2), H = p.H, hd = p.hd, n = p.cnt;
    const int lb = p.l_off + s * p.l_row, qb = p.q_off + s * p.q_row;
    const int hb = tid >> 4, jb = tid & 15;
    for (int t0 = 0; t0 < n; t0 += 64) {
        CCD_SYNC(it);
        if (tid < 64) se[tid] = t0 + tid < n ? list[lb + t0 + tid] : -1;
        float acc[4][4];
        for (int a = 0; a < 4; a++) for (int b = 0; b < 4; b++) acc[a][b] = 0.f;
        for (int c0 = 0; c0 < hd; c0 += 32) {
            CCD_SYNC(it);
            for (int k = tid; k < 64 * 32; k += 256) {
                const int hh = k >> 5, ii = k & 31, h = hg + hh;
                sq[k] = h < H ? q[qb + h * hd + c0 + ii] : 0.f;
                const int e = se[hh];
                sk[hh * 33 + ii] = e >= 0 ? ccd_kv(win, cmp, e, p.nwin, p.w_off, p.c_off, hd, c0 + ii) : 0.f;
            }
            CCD_SYNC(it);
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
static void ccd_pa_weights_k(const int *list, const float *prm, float *scr, float *den, CcDsAttn p) {
    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto &red = CCD_LOCAL(float[256]);
    auto &sw = CCD_LOCAL(float[3072]);
    const int s = it.get_group(2), h = it.get_group(1), tid = it.get_local_id(2), H = p.H, n = p.cnt;
    const int lb = p.l_off + s * p.l_row;
    float *row = scr + ((size_t)s * H + h) * n;
    float best = -1.0e30f;
    for (int j = tid; j < n; j += 256) { const float v = row[j]; if (v > best) best = v; }
    red[tid] = best;
    CCD_SYNC(it);
    for (int off = 128; off > 0; off >>= 1) {
        if (tid < off && red[tid + off] > red[tid]) red[tid] = red[tid + off];
        CCD_SYNC(it);
    }
    best = red[0];
    CCD_SYNC(it);
    for (int j = tid; j < n; j += 256) { const float v = row[j], w = v > CC_DS_NEG_INF ? CCD_EXP(v - best) : 0.f; row[j] = w; sw[j] = w; }
    CCD_SYNC(it);
    if (tid == 0) {
        float d = CCD_EXP(prm[p.sink_off + h] - best);
        for (int j = 0; j < n; j++) if (list[lb + j] >= 0) d += sw[j];
        den[(size_t)s * H + h] = d;
    }
}
/* the output: 64 heads by 64 dims a work-group, the entries in list order on tiles of 32 */
static void ccd_pa_out_k(const float *win, const float *cmp, const int *list, const float *scr,
                         const float *den, float *out, CcDsAttn p) {
    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto &sk = CCD_LOCAL(float[32 * 65]);
    auto &sw = CCD_LOCAL(float[64 * 33]);
    auto &se = CCD_LOCAL(int[32]);
    const int s = it.get_group(2), d0 = it.get_group(1) * 64, hg = it.get_group(0) * 64, tid = it.get_local_id(2), H = p.H, hd = p.hd, n = p.cnt;
    const int lb = p.l_off + s * p.l_row;
    const int hb = tid >> 4, ib = tid & 15;
    float acc[4][4];
    for (int a = 0; a < 4; a++) for (int b = 0; b < 4; b++) acc[a][b] = 0.f;
    for (int t0 = 0; t0 < n; t0 += 32) {
        CCD_SYNC(it);
        if (tid < 32) se[tid] = t0 + tid < n ? list[lb + t0 + tid] : -1;
        CCD_SYNC(it);
        for (int k = tid; k < 32 * 64; k += 256) {
            const int jj = k >> 6, ii = k & 63, e = se[jj];
            sk[jj * 65 + ii] = e >= 0 ? ccd_kv(win, cmp, e, p.nwin, p.w_off, p.c_off, hd, d0 + ii) : 0.f;
            const int hh = k >> 5, j2 = k & 31, h = hg + hh;
            const float w = h < H && t0 + j2 < n ? scr[((size_t)s * H + h) * n + t0 + j2] : 0.f;
            sw[hh * 33 + j2] = (p.flags & 1) ? ccd_bf16r(w) : w;
        }
        CCD_SYNC(it);
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
static CcBuf *g_ccd_spl;   /* the split decode attention's partials: H x splits x (hd + 2) floats */
/* A decode row's attention split over its list (flash decoding; S == 1, flags 0): pass 1, a work-group a (head, slice of
 * the list), its scores as ccd_attn_k's dot products, the slice's max m, l = sum exp(v - m) and acc[i] = sum exp(v - m) *
 * kv[i] over its rows in list order; pass 2, a work-group a head, M = the max of the sink and the slices', den =
 * exp(sink - M) + sum l exp(m - M), out = sum acc exp(m - M) / den. Not ccd_attn_k's bits (the sums meet per slice):
 * quality-gated; the multi-row and prompt forms keep theirs. */
static void ccd_attn_sp1_k(const float *q, const float *win, const float *cmp, const int *list, float *spl, CcDsAttn p,
                           int per, uint8_t *slm) {
    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    float *sq = (float *)slm, *sc = sq + p.hd;   /* hd queries, then per scores */
    auto &red = CCD_LOCAL(float[256]);
    const int h = it.get_group(2), sp = it.get_group(1), tid = it.get_local_id(2), ns = it.get_group_range(1);
    const int hd = p.hd, j0 = sp * per, j1 = sycl::min(p.cnt, j0 + per), lb = p.l_off;
    for (int i = tid; i < hd; i += 256) sq[i] = q[p.q_off + h * hd + i];
    CCD_SYNC(it);
    float best = -1.0e30f;
    for (int j = j0 + tid; j < j1; j += 256) {
        const int e = list[lb + j];
        float v = CC_DS_NEG_INF;
        if (e >= 0) {
            const float *kr = e < p.nwin ? win + p.w_off + e * hd : cmp + p.c_off + (e - p.nwin) * hd;
            float d = 0.f;
            for (int i = 0; i < hd; i++) d += sq[i] * kr[i];
            v = d * p.scale;
            if (v > best) best = v;
        }
        sc[j - j0] = v;
    }
    red[tid] = best;
    CCD_SYNC(it);
    for (int off = 128; off > 0; off >>= 1) { if (tid < off && red[tid + off] > red[tid]) red[tid] = red[tid + off]; CCD_SYNC(it); }
    const float m = red[0];
    CCD_SYNC(it);
    for (int j = j0 + tid; j < j1; j += 256) { const float v = sc[j - j0]; sc[j - j0] = v > CC_DS_NEG_INF ? CCD_EXP(v - m) : 0.f; }
    CCD_SYNC(it);
    float *o = spl + ((size_t)h * ns + sp) * (hd + 2);
    if (tid == 0) { float l = 0.f; for (int j = j0; j < j1; j++) if (list[lb + j] >= 0) l += sc[j - j0]; o[hd] = m; o[hd + 1] = l; }
    for (int i = tid; i < hd; i += 256) {
        float acc = 0.f;
        for (int j = j0; j < j1; j++) {
            const int e = list[lb + j];
            if (e < 0) continue;
            acc += sc[j - j0] * ccd_kv(win, cmp, e, p.nwin, p.w_off, p.c_off, hd, i);
        }
        o[i] = acc;
    }
}
static void ccd_attn_sp2_k(const float *spl, const float *prm, float *out, CcDsAttn p, int ns) {
    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int h = it.get_group(2), tid = it.get_local_id(2), hd = p.hd;
    const float *b = spl + (size_t)h * ns * (hd + 2);
    const float sink = prm[p.sink_off + h];
    float M = sink;
    for (int k = 0; k < ns; k++) { const float mk = b[(size_t)k * (hd + 2) + hd]; if (b[(size_t)k * (hd + 2) + hd + 1] > 0.f && mk > M) M = mk; }
    float den = CCD_EXP(sink - M);
    for (int k = 0; k < ns; k++) { const float lk = b[(size_t)k * (hd + 2) + hd + 1]; if (lk > 0.f) den += lk * CCD_EXP(b[(size_t)k * (hd + 2) + hd] - M); }
    for (int i = tid; i < hd; i += 256) {
        float acc = 0.f;
        for (int k = 0; k < ns; k++) { const float lk = b[(size_t)k * (hd + 2) + hd + 1]; if (lk > 0.f) acc += b[(size_t)k * (hd + 2) + i] * CCD_EXP(b[(size_t)k * (hd + 2) + hd] - M); }
        out[p.o_off + h * hd + i] = acc / den;
    }
}
/* ccd_attn_sp1r_k / ccd_attn_sp2r_k: the split decode attention for R rows of different conversations in one launch
 * (row r's window and compressed rows at its own offsets wo[r], co[r]; its query, list and output rows r rows on), and
 * the scores taken a sub-group a list entry (its lanes over hd, reduce_over_group) instead of a work-item an entry
 * (ccd_attn_sp1_k: 64 of 256 work-items busy, each a serial dot of hd). Measured: 8 launches a 4-row layer, 19 % of the
 * frame. Pass 1 is otherwise ccd_attn_sp1_k's (the slice's max, exp, l in list order, acc in list order), pass 2
 * ccd_attn_sp2_k's. Quality-gated as the split is: the dot products sum in another order than ccd_attn_k's. One row
 * (cc_ds_attn S 1) goes here too; COLI_CUDA_DSV4_ATTN_V2=0: ccd_attn_sp1_k / sp2_k and a launch pair a row. */
#define CCD_ROWS_MAX 8
struct CcdRowOffs { int w[CCD_ROWS_MAX], c[CCD_ROWS_MAX]; };
template <int SG>
static void ccd_attn_sp1r_k(const float *q, const float *win, const float *cmp, const int *list, float *spl, CcDsAttn p,
                            CcdRowOffs ro, int per, uint8_t *slm) {
    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
    float *sq = (float *)slm, *sc = sq + p.hd;
    auto &red = CCD_LOCAL(float[256]);
    const int h = it.get_group(2), sp = it.get_group(1), r = it.get_group(0), tid = it.get_local_id(2), ns = it.get_group_range(1), H = it.get_group_range(2);
    const int lane = (int)sg.get_local_linear_id(), sgi = (int)sg.get_group_linear_id(), nsg = 256 / SG;
    const int hd = p.hd, j0 = sp * per, j1 = sycl::min(p.cnt, j0 + per), lb = p.l_off + r * p.l_row, wo = ro.w[r], co = ro.c[r];
    for (int i = tid; i < hd; i += 256) sq[i] = q[p.q_off + r * p.q_row + h * hd + i];
    CCD_SYNC(it);
    for (int j = j0 + sgi; j < j1; j += nsg) {   /* uniform in the sub-group */
        const int e = list[lb + j];
        float v = CC_DS_NEG_INF;
        if (e >= 0) {
            const float *kr = e < p.nwin ? win + wo + e * hd : cmp + co + (e - p.nwin) * hd;
            float d = 0.f;
            for (int i = lane; i < hd; i += SG) d += sq[i] * kr[i];
            d = sycl::reduce_over_group(sg, d, sycl::plus<float>());
            v = d * p.scale;
        }
        if (lane == 0) sc[j - j0] = v;
    }
    CCD_SYNC(it);
    float best = -1.0e30f;
    for (int j = j0 + tid; j < j1; j += 256) { const float v = sc[j - j0]; if (v > CC_DS_NEG_INF && v > best) best = v; }
    red[tid] = best;
    CCD_SYNC(it);
    for (int off = 128; off > 0; off >>= 1) { if (tid < off && red[tid + off] > red[tid]) red[tid] = red[tid + off]; CCD_SYNC(it); }
    const float m = red[0];
    CCD_SYNC(it);
    for (int j = j0 + tid; j < j1; j += 256) { const float v = sc[j - j0]; sc[j - j0] = v > CC_DS_NEG_INF ? CCD_EXP(v - m) : 0.f; }
    CCD_SYNC(it);
    float *o = spl + (((size_t)r * H + h) * ns + sp) * (hd + 2);
    if (tid == 0) { float l = 0.f; for (int j = j0; j < j1; j++) if (list[lb + j] >= 0) l += sc[j - j0]; o[hd] = m; o[hd + 1] = l; }
    for (int i = tid; i < hd; i += 256) {
        float acc = 0.f;
        for (int j = j0; j < j1; j++) {
            const int e = list[lb + j];
            if (e < 0) continue;
            acc += sc[j - j0] * ccd_kv(win, cmp, e, p.nwin, wo, co, hd, i);
        }
        o[i] = acc;
    }
}
static void ccd_attn_sp2r_k(const float *spl, const float *prm, float *out, CcDsAttn p, int ns) {
    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int h = it.get_group(2), r = it.get_group(1), H = it.get_group_range(2), tid = it.get_local_id(2), hd = p.hd;
    const float *b = spl + ((size_t)r * H + h) * ns * (hd + 2);
    const float sink = prm[p.sink_off + h];
    float M = sink;
    for (int k = 0; k < ns; k++) { const float mk = b[(size_t)k * (hd + 2) + hd]; if (b[(size_t)k * (hd + 2) + hd + 1] > 0.f && mk > M) M = mk; }
    float den = CCD_EXP(sink - M);
    for (int k = 0; k < ns; k++) { const float lk = b[(size_t)k * (hd + 2) + hd + 1]; if (lk > 0.f) den += lk * CCD_EXP(b[(size_t)k * (hd + 2) + hd] - M); }
    for (int i = tid; i < hd; i += 256) {
        float acc = 0.f;
        for (int k = 0; k < ns; k++) { const float lk = b[(size_t)k * (hd + 2) + hd + 1]; if (lk > 0.f) acc += b[(size_t)k * (hd + 2) + i] * CCD_EXP(b[(size_t)k * (hd + 2) + hd] - M); }
        out[p.o_off + r * p.o_row + h * hd + i] = acc / den;
    }
}
static int ccd_attn_v2(void) { const char *e = getenv("COLI_CUDA_DSV4_ATTN_V2"); return !(e && *e == '0'); }
/* n decode rows (p: row 0's, S ignored; rows r: q_off + r q_row, l_off + r l_row, o_off + r o_row; window / compressed rows
 * at wo[r] / co[r]) in one launch pair: 1, -1 not taken (flags, list length, local memory, n), 0 a failure */
static int ccd_attn_rows(CcCtx *c, const float *qp, const float *wp, const float *cm, const int *ls, const float *pp_, float *op,
                         const CcDsAttn *p, const int *wo, const int *co, int n) {
    const int per = 64, ns = (p->cnt + per - 1) / per;
    const size_t sm1 = ((size_t)p->hd + per) * sizeof(float);
    if (n < 1 || n > CCD_ROWS_MAX || (p->flags & 1) || ns < 2 || ns > 64 || sm1 > c->slm) return -1;
    if (!cc_reserve(&g_ccd_spl, (size_t)n * p->H * ns * (p->hd + 2) * sizeof(float), CC_DEV)) return 0;
    CcdRowOffs ro;
    for (int r = 0; r < CCD_ROWS_MAX; r++) { ro.w[r] = r < n ? wo[r] : 0; ro.c[r] = r < n ? co[r] : 0; }
    float *sl = fpw(g_ccd_spl);
    const CcDsAttn pp = *p; const int nsp = ns;
    if (!CC_KSG(c, dim3((unsigned)p->H, (unsigned)ns, (unsigned)n), 256, sm1, ccd_attn_sp1r_k<SG>(qp, wp, cm, ls, sl, pp, ro, per, slm_)))
        return launched(c, "ds attn rows");
    CC_K(c, dim3((unsigned)p->H, (unsigned)n), 256, ccd_attn_sp2r_k(sl, pp_, op, pp, nsp));
    return launched(c, "ds attn rows");
}
static int cc_ds_attn_rows(CcBuf *q, CcBuf *win, CcBuf *cmp, CcBuf *list, CcBuf *prm, CcBuf *out, const CcDsAttn *p,
                           const int *wo, const int *co, int n) {
    CcCtx *c = live();
    CC_OWN(c, "cc_ds_attn_rows", q, win, cmp, list, prm, out);
    if (!c || !p || !q || !win || !list || !prm || !out || !wo || !co || p->H < 1 || p->hd < 1 || p->hd > 1024 || p->cnt < 1 ||
        p->cnt > 3072) return 0;
    if (!ccd_attn_v2()) return -1;
    return ccd_attn_rows(c, fp(q), fp(win), fp(cmp ? cmp : win), reinterpret_cast<const int *>(fp(list)), fp(prm), fpw(out), p, wo, co, n);
}
static int cc_ds_attn(CcBuf *q, CcBuf *win, CcBuf *cmp, CcBuf *list, CcBuf *prm, CcBuf *out, const CcDsAttn *p) {
    CcCtx *c = live();
    CC_OWN(c, "cc_ds_attn", q, win, cmp, list, prm, out);
    if (!c || !p || !q || !win || !list || !prm || !out || p->S < 1 || p->H < 1 || p->hd < 1 || p->hd > 1024 ||
        p->cnt < 1 || p->cnt > 3072) return 0;
    const float *qp = fp(q), *wp = fp(win), *cm = fp(cmp ? cmp : win), *pp_ = fp(prm);
    const int *ls = reinterpret_cast<const int *>(fp(list));
    float *op = fpw(out);
    const char *e = getenv("COLI_CUDA_DSV4_PA");   /* 0: a work-group a (head, row) for every row */
    if (p->S >= CC_DS_PA_MIN && p->hd % 64 == 0 && !(e && *e == '0')) {
        const size_t per = (size_t)p->H * p->cnt * sizeof(float);
        int R = (int)(((size_t)CC_DS_PA_MB << 20) / per);
        if (R < 1) R = 1;
        if (R > p->S) R = p->S;
        if (!cc_reserve(&g_ccd_scr, (size_t)R * per, CC_DEV) || !cc_reserve(&g_ccd_den, (size_t)R * p->H * sizeof(float), CC_DEV))
            return 0;
        float *scr = fpw(g_ccd_scr), *dn = fpw(g_ccd_den);
        const unsigned hgs = (unsigned)((p->H + 63) / 64), H = (unsigned)p->H, d64 = (unsigned)(p->hd / 64);
        for (int r0 = 0; r0 < p->S; r0 += R) {
            CcDsAttn x = *p;
            x.S = p->S - r0 < R ? p->S - r0 : R;
            x.l_off += r0 * p->l_row; x.q_off += r0 * p->q_row; x.o_off += r0 * p->o_row;
            const unsigned xs = (unsigned)x.S;
            if (!CC_K(c, dim3(xs, hgs), 256, ccd_pa_scores_k(qp, wp, cm, ls, scr, x)) ||
                !CC_K(c, dim3(xs, H), 256, ccd_pa_weights_k(ls, pp_, scr, dn, x)) ||
                !CC_K(c, dim3(xs, d64, hgs), 256, ccd_pa_out_k(wp, cm, ls, scr, dn, op, x))) break;
        }
        return launched(c, "ds attn (prompt)");
    }
    const CcDsAttn pp = *p;
    {   /* a decode row (S 1, flags 0): split over the list (COLI_CUDA_DSV4_ATTN_SPLIT=0: off) */
        static int sp_on = -1;
        if (sp_on < 0) { const char *es = getenv("COLI_CUDA_DSV4_ATTN_SPLIT"); sp_on = !(es && *es == '0'); }
        const int per = 64, ns = (p->cnt + per - 1) / per;
        const size_t sm1 = ((size_t)p->hd + per) * sizeof(float);
        if (sp_on && p->S == 1 && ccd_attn_v2()) {
            const int r = ccd_attn_rows(c, qp, wp, cm, ls, pp_, op, p, &p->w_off, &p->c_off, 1);
            if (r >= 0) return r;
        }
        if (sp_on && p->S == 1 && !(p->flags & 1) && ns >= 2 && ns <= 64 && sm1 <= c->slm &&
            cc_reserve(&g_ccd_spl, (size_t)p->H * ns * (p->hd + 2) * sizeof(float), CC_DEV)) {
            float *sl = fpw(g_ccd_spl);
            const int nsp = ns;
            if (!CC_KSG(c, dim3((unsigned)p->H, (unsigned)ns), 256, sm1, ccd_attn_sp1_k(qp, wp, cm, ls, sl, pp, per, slm_))) return launched(c, "ds attn split");
            CC_K(c, dim3((unsigned)p->H), 256, ccd_attn_sp2_k(sl, pp_, op, pp, nsp));
            return launched(c, "ds attn split");
        }
    }
    {   /* CCD_ATTN_HB heads a work-group where its local memory fits */
        static int hb_on = -1;
        /* opt-in (COLI_CUDA_DSV4_ATTN_HB=1): bit-identical, but a decode step measured slower -- one decode row leaves
         * H/8 work-groups (8 on the card), and the rows' re-reads it saves were L2 hits */
        if (hb_on < 0) { const char *eh = getenv("COLI_CUDA_DSV4_ATTN_HB"); hb_on = eh && *eh == '1'; }
        const size_t sm8 = (size_t)CCD_ATTN_HB * ((size_t)p->hd + p->cnt) * sizeof(float);
        if (hb_on && p->H >= 2 && sm8 <= 48 * 1024 && sm8 <= c->slm) {
            CC_KSG(c, dim3((unsigned)((p->H + CCD_ATTN_HB - 1) / CCD_ATTN_HB), (unsigned)p->S), 256, sm8,
                   ccd_attn_hb_k(qp, wp, cm, ls, pp_, op, pp, slm_));
            return launched(c, "ds attn");
        }
    }
    const size_t smem = ((size_t)p->hd + p->cnt) * sizeof(float);
    if (smem > 48 * 1024 || smem > c->slm) return 0;
    CC_KSG(c, dim3((unsigned)p->H, (unsigned)p->S), 256, smem, ccd_attn_k(qp, wp, cm, ls, pp_, op, pp, slm_));
    return launched(c, "ds attn");
}

/* ---- RoPE on interleaved pairs ---------------------------------------------------------- */
static void ccd_rope_k(float *x, const float *cs, CcDsRope p) {
    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const long long t = (long long)it.get_group(2) * it.get_local_range(2) + it.get_local_id(2);
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
    CC_OWN(c, "cc_ds_rope", x, cs);
    if (!c || !p || !x || !cs || p->nseg < 0 || p->per_row < 1 || p->rd < 2 || (p->rd & 1)) return 0;
    if (p->nseg == 0) return 1;
    float *xp = fpw(x); const float *csp = fp(cs); const CcDsRope pp = *p;
    CC_K(c, dim3(grid1((size_t)p->nseg * (p->rd / 2), 256)), 256, ccd_rope_k(xp, csp, pp));
    return launched(c, "ds rope");
}

/* ---- the compressor's rolling group (one work-group, the rows in order) ----------------- */
/* the ring is global memory one work-item writes and others read: global barriers */
static void ccd_compress_k(const float *kv, const float *sc, float *ring, const float *ape, float *out, CcDsComp p) {
    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int tid = it.get_local_id(2), R = p.ratio, P = p.P, D = p.D, ov = p.overlap;
    const int rows = ov ? 2 * R : R, ro = p.ring_off, so = ro + rows * P;
    for (int s = 0; s < p.S; s++) {
        const int pos = p.pos_base + s, slot = pos % R, srow = ov ? R + slot : slot;
        for (int c = tid; c < P; c += 256) {
            ring[ro + srow * P + c] = kv[p.kv_off + s * p.kv_row + c];
            float v = sc[p.sc_off + s * p.sc_row + c];
            if (p.ape_off >= 0) v = v + ape[p.ape_off + slot * P + c];
            ring[so + srow * P + c] = v;
        }
        CCD_SYNC_G(it);
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
                const float w = CCD_EXP(ring[so + k * P + col] - best);
                tot += w;
                mix += w * ring[ro + k * P + col];
            }
            out[p.out_off + g * p.out_row + d] = mix / tot;
        }
        CCD_SYNC_G(it);
        if (ov) {
            for (int c = tid; c < R * P; c += 256) {
                ring[ro + c] = ring[ro + R * P + c];
                ring[so + c] = ring[so + R * P + c];
            }
            CCD_SYNC_G(it);
        }
    }
}
static int cc_ds_compress(CcBuf *kv, CcBuf *sc, CcBuf *ring, CcBuf *prm, CcBuf *out, const CcDsComp *p) {
    CcCtx *c = live();
    CC_OWN(c, "cc_ds_compress", kv, sc, ring, prm, out);
    if (!c || !p || p->S < 0 || p->ratio < 1 || p->P < 1 || p->D < 1 || p->D > p->P || (p->overlap && p->P < 2 * p->D)) return 0;
    if (p->S == 0) return 1;
    if (!kv || !sc || !ring || !out || (p->ape_off >= 0 && !prm)) return 0;
    const float *kp = fp(kv), *sp = fp(sc), *ap = fp(prm); float *rp = fpw(ring), *op = fpw(out); const CcDsComp pp = *p;
    CC_K(c, dim3(1), 256, ccd_compress_k(kp, sp, rp, ap, op, pp));
    return launched(c, "ds compress");
}

/* ---- the indexer's scores: a work-item a (column, head), the heads joined in their order */
static void ccd_score_k(const float *iq, const float *hw, const float *keys, const unsigned char *mask, float *scs,
                        CcDsScore p, uint8_t *slm) {
    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    float *sm = (float *)slm;   /* the queries (IH*ID when they fit), then IH weights */
    auto &red = CCD_LOCAL(float[256]);
    const int s = it.get_group(1), tid = it.get_local_id(2), IH = p.IH, ID = p.ID, width = p.width;
    const int pos = p.pos_base + s, lens = (pos + 1) / p.ratio;
    const int qb = p.q_off + s * p.q_row, wb = p.w_off + s * p.w_row, sb = p.sc_off + s * p.sc_row, mb = p.mask_off + s * p.mask_row;
    const int P = 256 / IH > 0 ? 256 / IH : 1, j0 = it.get_group(2) * P;
    if (j0 >= width) return;
    const int sh = IH <= 64 && IH * ID <= 4096, wo = sh ? IH * ID : 0;
    if (sh) for (int i = tid; i < IH * ID; i += 256) sm[i] = iq[qb + i];
    for (int h = tid; h < IH; h += 256) sm[wo + h] = hw[wb + h] * p.wscale;
    CCD_SYNC(it);
    const int jj = tid / IH, h = tid - jj * IH, j = j0 + jj;
    if (jj < P && j < width && j < lens && (p.mask_row <= 0 || mask[mb + j] != 0)) {
        const int kb = p.k_off + j * p.k_row;
        float d = 0.f;
        if (sh) for (int i = 0; i < ID; i++) d += sm[h * ID + i] * keys[kb + i];
        else for (int i = 0; i < ID; i++) d += iq[qb + h * ID + i] * keys[kb + i];
        red[tid] = d;
    }
    CCD_SYNC(it);
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
    CC_OWN(c, "cc_ds_score", iq, hw, keys, mask, sc);
    if (!c || !p || p->S < 1 || p->IH < 1 || p->IH > 256 || p->ID < 1 || p->ratio < 1 || p->width < 0 ||
        p->sc_row < p->width || (p->mask_row > 0 && !mask) || !iq || !hw || !keys || !sc) return 0;
    if (p->mask_row > 0 && (size_t)p->mask_off + (size_t)(p->S - 1) * p->mask_row + (size_t)p->width > mask->bytes) return 0;   /* bytes */
    if (p->width == 0) return 1;
    const int P = 256 / p->IH, sh = p->IH <= 64 && p->IH * p->ID <= 4096;
    const size_t smem = ((sh ? (size_t)p->IH * p->ID : 0) + (size_t)p->IH) * sizeof(float);
    if (smem > 48 * 1024 || smem > c->slm) return 0;
    const float *qp = fp(iq), *wp = fp(hw), *kp = fp(keys);
    const unsigned char *mp = reinterpret_cast<const unsigned char *>(fp(mask));
    float *sp = fpw(sc); const CcDsScore pp = *p;
    CC_KSG(c, dim3((unsigned)((p->width + P - 1) / P), (unsigned)p->S), 256, smem, ccd_score_k(qp, wp, kp, mp, sp, pp, slm_));
    return launched(c, "ds score");
}

/* ---- the candidate blocks ------------------------------------------------------------------ */
/* the mask is global memory work-items zero and others then set: global barriers */
static void ccd_cand_k(const float *scs, unsigned char *mask, CcDsCand p) {
    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto &bm = CCD_LOCAL(float[4096]);
    auto &red = CCD_LOCAL(float[256]);
    auto &cnt = CCD_LOCAL(unsigned[256]);
    const int s = it.get_group(2), tid = it.get_local_id(2);
    const int pos = p.pos_base + s, lens = (pos + 1) / p.ratio;
    const int width = p.width, blk = p.block, topb = p.topb, sb = s * p.sc_row, mb = p.mask_off + s * p.mask_row;
    const int blocks = (width + blk - 1) / blk;
    for (int b = tid; b < blocks; b += 256) {
        float top = CC_DS_NEG_INF;
        for (int i = b * blk; i < (b + 1) * blk && i < width; i++) { const float v = scs[sb + i]; if (v > top) top = v; }
        bm[b] = top;
    }
    for (int i = tid; i < width; i += 256) mask[mb + i] = 0;
    CCD_SYNC_G(it);
    const int last = lens >= 1 ? (lens - 1) / blk : (blk == 1 ? -1 : 0);   /* C's (lens - 1) / block */
    if (tid == 0 && last >= 0 && last < blocks) bm[last] = CC_DS_POS_INF;
    CCD_SYNC_G(it);
    const int wanted = topb < blocks ? topb : blocks;
    for (int pick = 0; pick < wanted; pick++) {   /* the best block above -inf, the lower on a tie */
        float bv = CC_DS_NEG_INF; int bi = -1;
        for (int b = tid; b < blocks; b += 256) {
            const float v = bm[b];
            if (v > CC_DS_NEG_INF && (bi < 0 || v > bv)) { bv = v; bi = b; }
        }
        red[tid] = bv; cnt[tid] = (unsigned)(bi < 0 ? 0x7fffffff : bi);
        CCD_SYNC_G(it);
        for (int off = 128; off > 0; off >>= 1) {
            if (tid < off) {
                const float ov = red[tid + off]; const unsigned oi = cnt[tid + off];
                if (oi != 0x7fffffffu && (cnt[tid] == 0x7fffffffu || ov > red[tid] || (ov == red[tid] && oi < cnt[tid]))) {
                    red[tid] = ov; cnt[tid] = oi;
                }
            }
            CCD_SYNC_G(it);
        }
        const unsigned chosen = cnt[0];
        CCD_SYNC_G(it);
        if (chosen == 0x7fffffffu) break;
        const int cb = (int)chosen;
        for (int i = cb * blk + tid; i < (cb + 1) * blk && i < width; i += 256) mask[mb + i] = 1;
        if (tid == 0) bm[cb] = CC_DS_NEG_INF;
        CCD_SYNC_G(it);
    }
}
static int cc_ds_cand(CcBuf *sc, CcBuf *mask, const CcDsCand *p) {
    CcCtx *c = live();
    CC_OWN(c, "cc_ds_cand", sc, mask);
    if (!c || !p || !sc || !mask || p->S < 1 || p->block < 1 || p->ratio < 1 || p->width < 0 ||
        (p->width + p->block - 1) / p->block > 4096 || p->mask_row < p->width || p->mask_off < 0 ||
        (size_t)p->mask_off + (size_t)(p->S - 1) * p->mask_row + (size_t)p->width > mask->bytes) return 0;
    const float *sp = fp(sc); unsigned char *mp = reinterpret_cast<unsigned char *>(fpw(mask)); const CcDsCand pp = *p;
    CC_K(c, dim3((unsigned)p->S), 256, ccd_cand_k(sp, mp, pp));
    return launched(c, "ds cand");
}

/* ---- the top-k: the keep-th largest key bit by bit, then the columns in order ---------- */
static void ccd_topk_k(const float *scs, int *list, CcDsTopk p, uint8_t *slm) {
    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto &cnt = CCD_LOCAL(unsigned[256]);
    float *rank = (float *)slm;   /* order 1: the selected columns (the launch's topk floats) */
    const int s = it.get_group(2), tid = it.get_local_id(2);
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
    CCD_SYNC(it);
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
    CC_OWN(c, "cc_ds_topk", sc, list);
    if (!c || !p || !sc || !list || p->S < 1 || p->width < 0 || p->topk < 1 || p->sc_row < p->width || (p->order && p->topk > 4096)) return 0;
    const size_t smem = p->order ? (size_t)p->topk * sizeof(float) : 0;
    const float *sp = fp(sc); int *lp = reinterpret_cast<int *>(fpw(list)); const CcDsTopk pp = *p;
    CC_KSG(c, dim3((unsigned)p->S), 256, smem, ccd_topk_k(sp, lp, pp, slm_));
    return launched(c, "ds topk");
}

/* ---- the engram gate ------------------------------------------------------------------------ */
static void ccd_engram_k(const float *kv, const float *prm, float *x, CcDsEngram p) {
    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto &sm = CCD_LOCAL(float[768]);
    const int cs = it.get_group(2), s = it.get_group(1), tid = it.get_local_id(2), H = p.H, D = p.D;
    if (s >= p.S || cs >= H) return;
    const int kb = p.kv_off + s * p.kv_row, xb = p.x_off + s * p.x_row + cs * D;
    const int qw = p.qw_off + cs * D, kw = p.kw_off + cs * D;
    float xs = 0.f, ks = 0.f, dt = 0.f;
    for (int i = tid; i < D; i += 256) {
        const float xv = x[xb + i], kvv = kv[kb + cs * D + i];
        xs += xv * xv; ks += kvv * kvv; dt += xv * prm[qw + i] * prm[kw + i] * kvv;
    }
    sm[tid] = xs; sm[256 + tid] = ks; sm[512 + tid] = dt;
    CCD_SYNC(it);
    for (int off = 128; off > 0; off >>= 1) {
        if (tid < off) { sm[tid] += sm[tid + off]; sm[256 + tid] += sm[256 + tid + off]; sm[512 + tid] += sm[512 + tid + off]; }
        CCD_SYNC(it);
    }
    const float r = (1.f / sycl::sqrt(sm[0] / (float)D + p.eps)) * (1.f / sycl::sqrt(sm[256] / (float)D + p.eps));
    const float t = sm[512] * r / sycl::sqrt((float)D);
    float mag = sycl::fabs(t);
    if (mag < 1.0e-6f) mag = 1.0e-6f;
    float sr = sycl::sqrt(mag);
    if (t < 0.f) sr = -sr;
    const float g = ccd_sig(sr);
    CCD_SYNC(it);
    const int vb = kb + H * D;
    for (int i = tid; i < D; i += 256) x[xb + i] = x[xb + i] + g * kv[vb + i];
}
static int cc_ds_engram(CcBuf *kv, CcBuf *prm, CcBuf *x, const CcDsEngram *p) {
    CcCtx *c = live();
    CC_OWN(c, "cc_ds_engram", kv, prm, x);
    if (!c || !p || !kv || !prm || !x || p->S < 1 || p->H < 1 || p->D < 1) return 0;
    const float *kp = fp(kv), *pp_ = fp(prm); float *xp = fpw(x); const CcDsEngram pp = *p;
    CC_K(c, dim3((unsigned)p->H, (unsigned)p->S), 256, ccd_engram_k(kp, pp_, xp, pp));
    return launched(c, "ds engram");
}
#endif
