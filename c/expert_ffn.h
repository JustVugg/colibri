/* expert_ffn.h — the routed-expert FFN kernel shared by the MoE engines.
 *
 * One weight layout, one activation contract, one way of running a layer:
 *
 *   layout   "planar unsigned int4, gs=64": a row of I weights (I % 64 == 0) is
 *            I/64 blocks of 32 bytes; byte k of a block holds element k in its
 *            low nibble and element k+32 in its high nibble, both stored as
 *            v+8 (0..15). One f32 scale per (row, block), row-major [O][I/64].
 *            The same layout quant.h calls "fmt 4 planar" (K1b): `and 0x0F`
 *            yields elements 0..31 already in order, `srli 4` the other 32,
 *            so no unpack instruction is ever paid, and the unsigned nibbles
 *            feed vpdpbusd natively through dot(v,x) = dot(u,x) - 8*sum(x).
 *   f32      activations stay f32 (default): the weight is exact in f32, one
 *            fma per element, one fma per block for the scale. The only
 *            difference from a pair-layout f32 kernel is accumulation order.
 *   i8       activations quantized once per token to int8 with one scale per
 *            row plus the per-block sums (opt-in, `mode` 1): integer dots,
 *            2 dpbusd per block. Not bit-identical to f32: same policy as
 *            IDOT elsewhere in the tree (opt-in until an ablation blesses it).
 *   layer    xf_moe_run() runs a whole MoE layer: gate+up fused over the
 *            experts the routing touched, silu, down, then the K contributions
 *            of each token summed in rank order. Threads split (expert, row
 *            chunk) items, so a layer costs two OpenMP regions instead of
 *            3*K, and a prompt row never re-reads a weight another row of the
 *            same expert just used.
 *
 * Why it exists: measured on the real Qwen3.6 container at full residency the
 * MoE cost 34 ms/token in steady state against a DRAM floor of 17 ms for the
 * int8-unpacked experts it was reading and 9 ms for the int4 bytes actually
 * on disk. Unpacking int4 to int8 at load doubled both the bytes per token
 * and the resident set; per-GEMV OpenMP regions (24 per layer) and one
 * reduction per 64-block did the rest. The same shape (int4 experts, f32
 * hidden state, top-k routing) is every engine here, which is why this is a
 * header and not a function in one of them.
 *
 * Header-only, all static, pure compute: no Model, no I/O, no allocation
 * beyond what the caller hands in. AVX2/FMA, AVX-VNNI or AVX512-VNNI where built,
 * with a scalar fallback that defines the numerics every SIMD path must
 * reproduce (f32 paths: same per-lane accumulation order; i8 paths: exact
 * integer sums, so identical by construction). */
#ifndef COLI_EXPERT_FFN_H
#define COLI_EXPERT_FFN_H

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <math.h>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#include <immintrin.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

#define XF_BLOCK 64                       /* elements per planar block and per scale group */
#define XF_BLOCK_BYTES 32

/* ---- layout helpers ------------------------------------------------------- */

static inline size_t xf_row_bytes(int I) { return (size_t)I / 2; }
static inline int    xf_groups(int I)    { return I / XF_BLOCK; }
static inline int    xf_layout_ok(int I) { return I > 0 && (I % XF_BLOCK) == 0; }

/* Repack one row from the "signed pairs" container layout (byte j = element
 * 2j in the low nibble, 2j+1 in the high nibble, two's-complement 4-bit — the
 * qwen36 converter's pack_int4) into planar unsigned. src and dst may not
 * overlap. I % 64 == 0. */
static inline void xf_repack_row_pairs_signed(uint8_t *dst, const uint8_t *src, int I) {
    int b = 0;
#if defined(__AVX2__)
    /* 16 pair bytes hold elements 0..31: the low nibbles are the even ones,
     * the high nibbles the odd ones, so interleaving the two nibble vectors
     * byte-wise puts elements 0..31 in order. Same for 32..63 from the next 16
     * bytes; xor 8 turns the signed nibble into v+8; the block is lo | hi<<4. */
    const __m128i m4 = _mm_set1_epi8(0x0F);
    const __m256i e8 = _mm256_set1_epi8(8);
    for (; b < I / XF_BLOCK; b++) {
        const uint8_t *s = src + (size_t)b * XF_BLOCK_BYTES;
        __m128i a = _mm_loadu_si128((const __m128i *)s), c = _mm_loadu_si128((const __m128i *)(s + 16));
        __m128i al = _mm_and_si128(a, m4), ah = _mm_and_si128(_mm_srli_epi16(a, 4), m4);
        __m128i cl = _mm_and_si128(c, m4), ch = _mm_and_si128(_mm_srli_epi16(c, 4), m4);
        __m256i e0 = _mm256_set_m128i(_mm_unpackhi_epi8(al, ah), _mm_unpacklo_epi8(al, ah));   /* elements 0..31 */
        __m256i e1 = _mm256_set_m128i(_mm_unpackhi_epi8(cl, ch), _mm_unpacklo_epi8(cl, ch));   /* elements 32..63 */
        e0 = _mm256_xor_si256(e0, e8); e1 = _mm256_xor_si256(e1, e8);
        _mm256_storeu_si256((__m256i *)(dst + (size_t)b * XF_BLOCK_BYTES), _mm256_or_si256(e0, _mm256_slli_epi16(e1, 4)));
    }
#endif
    for (; b < I / XF_BLOCK; b++) {
        const uint8_t *s = src + (size_t)b * XF_BLOCK_BYTES;
        uint8_t *d = dst + (size_t)b * XF_BLOCK_BYTES;
        for (int k = 0; k < 32; k++) {
            /* element k of the block: byte k/2, nibble k&1; element k+32: byte 16 + k/2 */
            unsigned lo = (unsigned)(s[k >> 1]      >> ((k & 1) * 4)) & 0xF;
            unsigned hi = (unsigned)(s[16 + (k >> 1)] >> ((k & 1) * 4)) & 0xF;
            /* signed 4-bit -> v+8 is the nibble xor 8 */
            d[k] = (uint8_t)((lo ^ 8u) | ((hi ^ 8u) << 4));
        }
    }
}
static inline void xf_repack_pairs_signed(uint8_t *dst, const uint8_t *src, int O, int I) {
    for (int o = 0; o < O; o++)
        xf_repack_row_pairs_signed(dst + (size_t)o * xf_row_bytes(I), src + (size_t)o * xf_row_bytes(I), I);
}
/* Decode one element from a planar row (tests and scalar paths). */
static inline int xf_get(const uint8_t *row, int i) {
    const uint8_t *blk = row + (size_t)(i / XF_BLOCK) * XF_BLOCK_BYTES;
    int k = i % XF_BLOCK;
    unsigned u = k < 32 ? (blk[k] & 0xF) : (blk[k - 32] >> 4);
    return (int)u - 8;
}

/* ---- activation contract for the i8 mode ---------------------------------- */

/* Quantize x[I] to int8 with one scale (amax/127), and the per-block sums as
 * f32 (exact: |sum| <= 64*127). Returns the scale. */
static inline float xf_act_i8(const float *x, int I, int8_t *xq, float *xsum) {
    float am = 0.f;
    for (int i = 0; i < I; i++) { float a = fabsf(x[i]); if (a > am) am = a; }
    float s = am > 0.f ? am / 127.f : 1.f, inv = 1.f / s;
    for (int i = 0; i < I; i++) xq[i] = (int8_t)lrintf(x[i] * inv);
    for (int g = 0; g < I / XF_BLOCK; g++) {
        int32_t t = 0;
        for (int i = 0; i < XF_BLOCK; i++) t += xq[g * XF_BLOCK + i];
        xsum[g] = (float)t;
    }
    return s;
}

/* ---- scalar reference kernels (the numerics contract) --------------------- */

/* f32: per block, 8 lanes l=0..7 accumulate elements l, l+8, ..., l+56 in
 * order with fma; the 8 lane sums are scaled by the block scale into 8 lane
 * accumulators; the row ends with the fixed 8->1 tree of xf_hsum8. The SIMD
 * paths reproduce exactly this. */
static inline float xf_hsum8_scalar(const float *l) {
    float a0 = l[0] + l[4], a1 = l[1] + l[5], a2 = l[2] + l[6], a3 = l[3] + l[7];
    float b0 = a0 + a2, b1 = a1 + a3;
    return b0 + b1;
}
static inline float xf_dot_f32_ref(const uint8_t *w, const float *sc, const float *x, int I) {
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int g = 0; g < I / XF_BLOCK; g++) {
        const uint8_t *blk = w + (size_t)g * XF_BLOCK_BYTES;
        const float *xb = x + g * XF_BLOCK;
        float lane[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (int i = 0; i < XF_BLOCK; i++) {
            int v = (i < 32 ? (blk[i] & 0xF) : (blk[i - 32] >> 4)) - 8;
            lane[i & 7] = fmaf((float)v, xb[i], lane[i & 7]);
        }
        for (int l = 0; l < 8; l++) acc[l] = fmaf(lane[l], sc[g], acc[l]);
    }
    return xf_hsum8_scalar(acc);
}
/* i8: y = sx * sum_g sc[g] * (dot_u(g) - 8*xsum[g]); the per-block integer
 * dot is exact, then one f32 fma per block per lane exactly as the f32 path
 * (the 8 lanes hold the 8 int32 partial sums vpdpbusd produces: lane l sums
 * elements 4l..4l+3 and 32+4l..32+4l+3 of the block). */
static inline float xf_dot_i8_ref(const uint8_t *w, const float *sc, const int8_t *xq, const float *xsum, float sx, int I) {
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    float corr = 0.f;
    for (int g = 0; g < I / XF_BLOCK; g++) {
        const uint8_t *blk = w + (size_t)g * XF_BLOCK_BYTES;
        const int8_t *xb = xq + g * XF_BLOCK;
        int32_t lane[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (int k = 0; k < 32; k++) {
            lane[k >> 2] += (int32_t)(blk[k] & 0xF) * xb[k];
            lane[k >> 2] += (int32_t)(blk[k] >> 4)  * xb[k + 32];
        }
        for (int l = 0; l < 8; l++) acc[l] = fmaf((float)lane[l], sc[g], acc[l]);
        corr = fmaf(sc[g], xsum[g], corr);
    }
    return (xf_hsum8_scalar(acc) - 8.f * corr) * sx;
}

/* ---- SIMD kernels ---------------------------------------------------------- */

#if defined(__AVX2__) && defined(__FMA__)
#define XF_HAVE_AVX2 1
static inline float xf_hsum8(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
    __m128 a = _mm_add_ps(lo, hi);                       /* l+l+4 */
    __m128 b = _mm_add_ps(a, _mm_movehl_ps(a, a));       /* (0+4)+(2+6), (1+5)+(3+7) */
    __m128 c = _mm_add_ss(b, _mm_shuffle_ps(b, b, 1));
    return _mm_cvtss_f32(c);
}
/* nibbles of one 32-byte block as two 32-lane signed int8 vectors (v = u-8) */
static inline void xf_nib_s8(__m256i bb, __m256i *lo, __m256i *hi) {
    const __m256i m4 = _mm256_set1_epi8(0x0F), e8 = _mm256_set1_epi8(8);
    *lo = _mm256_sub_epi8(_mm256_and_si256(bb, m4), e8);
    *hi = _mm256_sub_epi8(_mm256_and_si256(_mm256_srli_epi16(bb, 4), m4), e8);
}
/* No 512-bit f32 path on purpose: element i must land in lane i&7 in the
 * reference's order, and folding 16 lanes onto 8 changes the fma order. The
 * i8 path is exact integers and could widen; it is DRAM-bound already. */

/* f32 row dot, AVX2: 8 lanes per element octet, lane = i&7 */
static inline float xf_dot_f32_avx2(const uint8_t *w, const float *sc, const float *x, int I) {
    __m256 acc = _mm256_setzero_ps();
    for (int g = 0; g < I / XF_BLOCK; g++) {
        __m256i bb = _mm256_loadu_si256((const __m256i *)(w + (size_t)g * XF_BLOCK_BYTES));
        __m256i lo, hi; xf_nib_s8(bb, &lo, &hi);
        const float *xb = x + g * XF_BLOCK;
        __m128i lo0 = _mm256_castsi256_si128(lo), lo1 = _mm256_extracti128_si256(lo, 1);
        __m128i hi0 = _mm256_castsi256_si128(hi), hi1 = _mm256_extracti128_si256(hi, 1);
        __m256 lane = _mm256_setzero_ps();
        lane = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo0)),                    _mm256_loadu_ps(xb),      lane);
        lane = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo0, 8))), _mm256_loadu_ps(xb + 8),  lane);
        lane = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo1)),                    _mm256_loadu_ps(xb + 16), lane);
        lane = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo1, 8))), _mm256_loadu_ps(xb + 24), lane);
        lane = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi0)),                    _mm256_loadu_ps(xb + 32), lane);
        lane = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi0, 8))), _mm256_loadu_ps(xb + 40), lane);
        lane = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi1)),                    _mm256_loadu_ps(xb + 48), lane);
        lane = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi1, 8))), _mm256_loadu_ps(xb + 56), lane);
        acc = _mm256_fmadd_ps(lane, _mm256_set1_ps(sc[g]), acc);
    }
    return xf_hsum8(acc);
}

/* i8 row dot: unsigned nibbles x int8 activations, 8 int32 lanes per block */
#if defined(__AVXVNNI__) || (defined(__AVX512VNNI__) && defined(__AVX512VL__))
#define XF_HAVE_VNNI 1
#if defined(__AVXVNNI__) && !defined(__AVX512VNNI__)
#define xf_dpbusd(acc, u, x) _mm256_dpbusd_avx_epi32(acc, u, x)
#else
#define xf_dpbusd(acc, u, x) _mm256_dpbusd_epi32(acc, u, x)
#endif
#else
static inline __m256i xf_dpbusd(__m256i acc, __m256i u, __m256i x) {
    /* u <= 15, |x| <= 127: a pair sum fits int16 (<= 3810) */
    return _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(u, x), _mm256_set1_epi16(1)));
}
#endif
static inline float xf_dot_i8_avx2(const uint8_t *w, const float *sc, const int8_t *xq, const float *xsum, float sx, int I) {
    const __m256i m4 = _mm256_set1_epi8(0x0F), z = _mm256_setzero_si256();
    __m256 acc = _mm256_setzero_ps();
    int ng = I / XF_BLOCK, g = 0;
    for (; g < ng; g++) {
        __m256i bb = _mm256_loadu_si256((const __m256i *)(w + (size_t)g * XF_BLOCK_BYTES));
        const int8_t *xb = xq + g * XF_BLOCK;
        __m256i d = xf_dpbusd(z, _mm256_and_si256(bb, m4), _mm256_loadu_si256((const __m256i *)xb));
        d = xf_dpbusd(d, _mm256_and_si256(_mm256_srli_epi16(bb, 4), m4), _mm256_loadu_si256((const __m256i *)(xb + 32)));
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(d), _mm256_set1_ps(sc[g]), acc);
    }
    /* corr = sum_g sc[g]*xsum[g] in the scalar reference's order (sequential fma) */
    float c = 0.f;
    for (g = 0; g < ng; g++) c = fmaf(sc[g], xsum[g], c);
    return (xf_hsum8(acc) - 8.f * c) * sx;
}
#endif /* AVX2 */

/* ---- dispatch -------------------------------------------------------------- */

static inline float xf_dot_f32(const uint8_t *w, const float *sc, const float *x, int I) {
#ifdef XF_HAVE_AVX2
    return xf_dot_f32_avx2(w, sc, x, I);
#else
    return xf_dot_f32_ref(w, sc, x, I);
#endif
}
static inline float xf_dot_i8(const uint8_t *w, const float *sc, const int8_t *xq, const float *xsum, float sx, int I) {
#ifdef XF_HAVE_AVX2
    return xf_dot_i8_avx2(w, sc, xq, xsum, sx, I);
#else
    return xf_dot_i8_ref(w, sc, xq, xsum, sx, I);
#endif
}

/* ---- one expert ------------------------------------------------------------- */

typedef struct {
    const uint8_t *g4, *u4, *d4;   /* planar rows: gate/up [F][H], down [H][F] */
    const float *gs, *us, *ds;     /* block scales: [F][H/64], [F][H/64], [H][F/64] */
} XfExpert;

/* Activations of one token, prepared once per layer (both modes carry x). */
typedef struct {
    const float *x;      /* [H] f32 */
    const int8_t *xq;    /* [H] int8, i8 mode */
    const float *xsum;   /* [H/64] block sums, i8 mode */
    float sx;            /* activation scale, i8 mode */
} XfAct;

/* gate+up rows [r0,r1) of one expert for one token: g[r], u[r]. Both matrices
 * share the activation, so their rows are interleaved in the same pass. */
static inline void xf_gate_up_rows(float *g, float *u, const XfExpert *e, const XfAct *a, int H, int r0, int r1, int mode) {
    size_t rb = xf_row_bytes(H); int ng = xf_groups(H);
    for (int r = r0; r < r1; r++) {
        const uint8_t *wg = e->g4 + (size_t)r * rb, *wu = e->u4 + (size_t)r * rb;
        const float *sg = e->gs + (size_t)r * ng, *su = e->us + (size_t)r * ng;
        if (mode) { g[r] = xf_dot_i8(wg, sg, a->xq, a->xsum, a->sx, H); u[r] = xf_dot_i8(wu, su, a->xq, a->xsum, a->sx, H); }
        else      { g[r] = xf_dot_f32(wg, sg, a->x, H);                 u[r] = xf_dot_f32(wu, su, a->x, H); }
    }
}
/* silu(g)*u over [0,F): the hidden row, in the engines' historical form */
static inline void xf_swiglu(float *h, const float *g, const float *u, int F) {
    for (int i = 0; i < F; i++) { float gv = g[i]; h[i] = (gv / (1.f + expf(-gv))) * u[i]; }
}
/* down rows [r0,r1) of one expert for one hidden row */
static inline void xf_down_rows(float *y, const XfExpert *e, const XfAct *h, int F, int r0, int r1, int mode) {
    size_t rb = xf_row_bytes(F); int ng = xf_groups(F);
    for (int r = r0; r < r1; r++) {
        const uint8_t *wd = e->d4 + (size_t)r * rb; const float *sd = e->ds + (size_t)r * ng;
        y[r] = mode ? xf_dot_i8(wd, sd, h->xq, h->xsum, h->sx, F) : xf_dot_f32(wd, sd, h->x, F);
    }
}

/* ---- one layer ---------------------------------------------------------------
 *
 * S tokens, K routed experts each. idx[s*K+k] is the expert id (or -1), val the
 * router weight, experts[s*K+k] the resident expert (NULL iff idx < 0).
 * out[s][H] = sum_k val[s][k] * expert_k(x[s]) accumulated in k order, as the
 * per-token loops it replaces did. Callers own `scratch` (xf_moe_scratch_bytes).
 *
 * Work is split into (expert, row chunk) items twice: gate+up over F rows, then
 * down over H rows. A prompt of S rows routed to the same expert reads that
 * expert's weights from DRAM once per chunk, the other rows hit cache. */
static inline size_t xf_moe_scratch_bytes(int S, int K, int H, int F) {
    size_t n = (size_t)S * K;
    return n * (2 * (size_t)F + (size_t)H) * sizeof(float)      /* g,u per (s,k); contrib per (s,k) */
         + n * (size_t)F * sizeof(float)                           /* h per (s,k) */
         + n * ((size_t)F + 64) + n * (size_t)(F / XF_BLOCK) * sizeof(float) /* hq, hsum (i8 mode) */
         + (size_t)S * ((size_t)H + 64) + (size_t)S * (size_t)(H / XF_BLOCK) * sizeof(float) /* xq, xsum */
         + n * sizeof(int) * 4 + 4096;
}

static inline void xf_moe_run(float *out, const float *x, int S, int K, int H, int F,
                              const int *idx, const float *val, const XfExpert *const *experts,
                              int mode, void *scratch) {
    const size_t n = (size_t)S * K;
    char *p = (char *)scratch;
#define XF_TAKE(T, count) ((T *)p); p += (((size_t)(count) * sizeof(T)) + 63) & ~(size_t)63
    float *g    = XF_TAKE(float, n * F);
    float *u    = XF_TAKE(float, n * F);
    float *h    = XF_TAKE(float, n * F);
    float *ctb  = XF_TAKE(float, n * H);
    int8_t *hq  = XF_TAKE(int8_t, n * F);
    float *hsum = XF_TAKE(float, n * (F / XF_BLOCK));
    float *hsx  = XF_TAKE(float, n);
    int8_t *xq  = XF_TAKE(int8_t, (size_t)S * H);
    float *xsum = XF_TAKE(float, (size_t)S * (H / XF_BLOCK));
    float *xsx  = XF_TAKE(float, S);
    int *uniq   = XF_TAKE(int, n);          /* distinct expert ids, in first-seen order */
    int *head   = XF_TAKE(int, n);          /* first (s,k) pair index of each distinct expert */
    int *next   = XF_TAKE(int, n);          /* chain of pairs sharing an expert */
    int *cnt    = XF_TAKE(int, n);
#undef XF_TAKE

    /* activations once per token */
    if (mode) for (int s = 0; s < S; s++) xsx[s] = xf_act_i8(x + (size_t)s * H, H, xq + (size_t)s * H, xsum + (size_t)s * (H / XF_BLOCK));

    /* group the (s,k) pairs by expert */
    int nu = 0;
    for (int s = 0; s < S; s++) for (int k = 0; k < K; k++) {
        int i = s * K + k; next[i] = -1;
        if (idx[i] < 0 || !experts[i]) continue;
        int j = 0; for (; j < nu; j++) if (uniq[j] == idx[i]) break;
        if (j == nu) { uniq[nu] = idx[i]; head[nu] = i; cnt[nu] = 1; nu++; }
        else { int t = head[j]; while (next[t] >= 0) t = next[t]; next[t] = i; cnt[j]++; }
    }
    if (nu == 0) { memset(out, 0, (size_t)S * H * sizeof(float)); return; }

    int T = 1;
#ifdef _OPENMP
    T = omp_get_max_threads();
#endif
    /* chunking: enough items to feed every thread, rows in multiples of 8 */
    int cF = (4 * T + nu - 1) / nu; if (cF < 1) cF = 1; if (cF > F / 8) cF = F / 8 > 0 ? F / 8 : 1;
    int cH = (4 * T + nu - 1) / nu; if (cH < 1) cH = 1; if (cH > H / 8) cH = H / 8 > 0 ? H / 8 : 1;
    int rowsF = (F + cF - 1) / cF; rowsF = (rowsF + 7) & ~7; cF = (F + rowsF - 1) / rowsF;
    int rowsH = (H + cH - 1) / cH; rowsH = (rowsH + 7) & ~7; cH = (H + rowsH - 1) / rowsH;

    /* phase 1: gate+up for every (expert, chunk), all rows routed to that expert */
    #pragma omp parallel for schedule(dynamic, 1)
    for (int it = 0; it < nu * cF; it++) {
        int e = it / cF, c = it % cF;
        int r0 = c * rowsF, r1 = r0 + rowsF; if (r1 > F) r1 = F;
        for (int i = head[e]; i >= 0; i = next[i]) {
            int s = i / K;
            XfAct a = { x + (size_t)s * H, xq + (size_t)s * H, xsum + (size_t)s * (H / XF_BLOCK), mode ? xsx[s] : 0.f };
            xf_gate_up_rows(g + (size_t)i * F, u + (size_t)i * F, experts[i], &a, H, r0, r1, mode);
        }
    }
    /* hidden rows (cheap, per pair) */
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < (int)n; i++) {
        if (idx[i] < 0 || !experts[i]) continue;
        xf_swiglu(h + (size_t)i * F, g + (size_t)i * F, u + (size_t)i * F, F);
        if (mode) hsx[i] = xf_act_i8(h + (size_t)i * F, F, hq + (size_t)i * F, hsum + (size_t)i * (F / XF_BLOCK));
    }
    /* phase 2: down for every (expert, chunk) */
    #pragma omp parallel for schedule(dynamic, 1)
    for (int it = 0; it < nu * cH; it++) {
        int e = it / cH, c = it % cH;
        int r0 = c * rowsH, r1 = r0 + rowsH; if (r1 > H) r1 = H;
        for (int i = head[e]; i >= 0; i = next[i]) {
            XfAct a = { h + (size_t)i * F, hq + (size_t)i * F, hsum + (size_t)i * (F / XF_BLOCK), mode ? hsx[i] : 0.f };
            xf_down_rows(ctb + (size_t)i * H, experts[i], &a, F, r0, r1, mode);
        }
    }
    /* rank-order sum per token: out[s] = sum_k val[s][k] * ctb[s][k] */
    #pragma omp parallel for schedule(static)
    for (int s = 0; s < S; s++) {
        float *os = out + (size_t)s * H;
        memset(os, 0, (size_t)H * sizeof(float));
        for (int k = 0; k < K; k++) {
            int i = s * K + k; if (idx[i] < 0 || !experts[i]) continue;
            float w = val[i]; const float *c = ctb + (size_t)i * H;
            for (int d = 0; d < H; d++) os[d] += w * c[d];
        }
    }
}

#endif /* COLI_EXPERT_FFN_H */
