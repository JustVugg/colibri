/* The dense chain's ops (vk_chain.c and shaders/chain_*.comp) against CPU references,
 * on any Vulkan device (CI: Lavapipe). Every op runs inside recorded frames the way an
 * engine uses them -- several ops per submission, a submission not waited for, the next
 * one ordered after it -- and its output is compared with a plain C version of what
 * qwen36.c / qwen38_core.h compute, within a float tolerance (the device sums in float
 * and in another order; the references sum in double):
 *   matmul   GEMV (chain_gemv.comp's vectorized one and qmatmul.comp's) and tiled
 *            GEMM, int8 rows, int4-g64, f32, bf16, offsets the device can bind and
 *            offsets it cannot (through the frame's temporaries)
 *   norm     zero-centred, plain, no weight, L2; heads with a gate between them, a
 *            weight slice per stream, in place
 *   rope     rotate-half from a host table, heads at a stride
 *   attn     causal GQA with the output gate over a cache at an offset, prefill rows
 *            after earlier ones, and a selection list; MiMo's form (vkc_attn_w): a
 *            sliding window, a ring of rows, position-major rows, V's own head dim and
 *            a sink logit per head
 *   dnconv   both orders of the sum, the ring carried across calls, the snapshot row
 *   dnrec    KD 8 and 128, silu and sigmoid gates, the state carried, the snapshot
 *   ew       every element-wise op (SCALE: MiMo's value scale)
 *   qsa      block keys, and the selection with ties broken as the CPU sorts
 *   ple      the gate and the dilated convolution with its ring and snapshot
 *   frames   vkc_write, vkc_read, a frame left in flight and one ordered after it
 *   mla      the per-head weight blocks in every format, both ways, with a gate; RoPE
 *            interleaved and rotate-half, in place and into a cache row; LayerNorm;
 *            the layer op against colibri.c's absorbed attention over several
 *            geometries (q latent or none, NoPE, kv_b or the split halves, selection
 *            lists with skipped entries, a gate, prefill rows, a long context from
 *            kv_start, latents up to 1024)
 *   dsa      the indexer's top-k selection bit for bit, ties included
 *   kpool    the k-pooled indexer (GLM-5.3): pooled keys, the pools' rank order and
 *            the tail, slot for slot against sparse_index.h
 *   kda      GLM-5.3's KDA layer around delta_attention.h's step: the convolution
 *            and its window, the recurrence and its state over two submissions, the
 *            decay, beta, output norm and gate
 *   mhc      hyper_connections.h's split, collapse and write back, the mean, and the
 *            clamped SwiGLU
 *   kda k3   Kimi K3's KDA layer (kimi_k3.c's kda_forward): exp(A_log) as the engine
 *            keeps it, its order of the l2 norms and the update, over two submissions
 *   ares     Kimi K3's attention residuals (res_mix) over 0 to 15 block snapshots, and
 *            SiTU-GLU on values both sides of its two constants
 *   sconv    inkling's short convolution (residual inside, the ring carried across
 *            calls) and its scalar multiply and divide
 *   relattn  inkling's attention: the relative-position bias, tau, a sliding window
 *            over a ring the step wraps, the step's rows read from its own K/V, and a
 *            global layer; the shared experts' weighted add (HC_APPLY over one stream)
 *   inkling's hidden size: the GEMV and the norm at D = 6144, the GEMV at I = 24576
 *   dsv4     DeepSeek V4.1 / V4 (deepseek_v41.c): the sparse attention with a sink over
 *            window and compressed rows (V4's bf16 roundings too), interleaved RoPE
 *            both ways, the compressor's ring (V4's overlapping form), the indexer's
 *            scores, candidate blocks and top-k slot for slot, the engram gate
 *   kvs      the KV cache split between the device and the host (vk_kvsplit.h): decode
 *            and prefill steps over a device holding a few blocks, the device's part, the
 *            host's and their merge against the full attention, GQA (a gate, a sink,
 *            position-major rows, a window, lists with pinned blocks) and MLA (NoPE, a
 *            nonzero start, lists, latents up to 1024)
 *   dsv4 rounding  DeepSeek V4's roundings (deepseek_v4.c): bf16, E4M3 and E2M1 per block,
 *            the Hadamard transform, bit for bit; its SwiGLU within one bf16 step
 *
 *   make vk-chain-check VK=1   (VK_ICD_FILENAMES=.../lvp_icd.json for Lavapipe) */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include "../backend_vulkan.h"
#include "../vk_chain.h"
#include "../vk_kvsplit.h"        /* the KV cache split between the device and the host */
#include "../delta_attention.h"      /* the KDA step the KDA ops follow */
#include "../hyper_connections.h"    /* the mHC arithmetic */
#include "../sparse_index.h"         /* GLM-5.3's k-pooled indexer */
#include "../sparse_attn.h"          /* DeepSeek V4.1's attention kernel */

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fails++; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

static unsigned rng = 12345;
static unsigned rnd(void) { rng = rng * 1103515245u + 12345u; return rng >> 8; }
static float frnd(void) { return (float)((int)(rnd() % 2001) - 1000) / 1000.0f; }
static float *fvec(size_t n, float scale) { float *v = malloc(n * sizeof *v); for (size_t i = 0; i < n; i++) v[i] = frnd() * scale; return v; }
static float sigm(float x) { return 1.f / (1.f + expf(-x)); }

/* max |a-b| / (max |b| + floor) */
static double relerr(const float *a, const float *b, size_t n, double floor) {
    double m = 0, e = 0;
    for (size_t i = 0; i < n; i++) { double d = fabs((double)a[i] - b[i]); if (d > e) e = d; if (fabs(b[i]) > m) m = fabs(b[i]); }
    return e / (m + floor);
}
static int bad(const float *a, size_t n) { for (size_t i = 0; i < n; i++) if (!isfinite(a[i])) return 1; return 0; }

static VkcBuf *up(const float *v, size_t n) {   /* a device buffer holding v */
    VkcBuf *b = vkc_buf(n * 4, VKC_DEV);
    if (!b) return NULL;
    vkc_begin(); vkc_write(b, 0, v, n * 4); vkc_submit(1);
    return b;
}
static float *down(VkcBuf *b, size_t off, size_t n) {
    float *v = malloc(n * sizeof *v);
    if (!vkc_read(b, off, v, n * 4)) { memset(v, 0xff, n * sizeof *v); }
    return v;
}

/* ---- matmul ----------------------------------------------------------------------- */
static uint16_t f2bf(float f) { uint32_t u; memcpy(&u, &f, 4); return (uint16_t)((u + 0x7fff + ((u >> 16) & 1)) >> 16); }
static float bf2f(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f; }
static int g_test_dev;   /* the device test_matmul uploads to (the chain's current one) */
static float *g_mm_out;  /* test_matmul's output, kept when set to non-NULL (the two devices' bits) */
static double g_mm_tol = 2e-4;   /* chain_gemm.comp's cases: x rounded to f16 */
static void test_matmul(int fmt, int S, int I, int O, size_t xo, size_t yo) {
    float *x = fvec((size_t)S * I, 1.f), *W = malloc((size_t)O * I * sizeof(float));
    void *codes = NULL; float *sc = NULL; int gs = 0;
    if (fmt == 1) {                      /* int8 rows, one scale per row */
        int8_t *q = malloc((size_t)O * I); sc = malloc(O * sizeof *sc);
        for (int o = 0; o < O; o++) { sc[o] = 0.01f + (rnd() % 100) / 5000.f;
            for (int i = 0; i < I; i++) { q[(size_t)o * I + i] = (int8_t)((int)(rnd() % 255) - 127); W[(size_t)o * I + i] = q[(size_t)o * I + i] * sc[o]; } }
        codes = q;
    } else if (fmt == 4) {               /* int4 (v+8 nibbles, low = even column), one scale per 64 */
        gs = 64; int ng = (I + 63) / 64;
        uint8_t *q = calloc((size_t)O * ((I + 1) / 2), 1); sc = malloc((size_t)O * ng * sizeof *sc);
        for (int o = 0; o < O; o++) for (int g = 0; g < ng; g++) sc[o * ng + g] = 0.02f + (rnd() % 100) / 3000.f;
        for (int o = 0; o < O; o++) for (int i = 0; i < I; i++) {
            int v = (int)(rnd() % 16); q[(size_t)o * ((I + 1) / 2) + i / 2] |= (uint8_t)(v << ((i & 1) * 4));
            W[(size_t)o * I + i] = (v - 8) * sc[o * ng + i / 64];
        }
        codes = q;
    } else if (fmt == 10) {
        float *w = fvec((size_t)O * I, 0.05f); memcpy(W, w, (size_t)O * I * sizeof(float)); codes = w;
    } else {                             /* 11 bf16 */
        uint16_t *w = malloc((size_t)O * I * 2);
        for (size_t i = 0; i < (size_t)O * I; i++) { w[i] = f2bf(frnd() * 0.05f); W[i] = bf2f(w[i]); }
        codes = w;
    }
    ColiVkTensor *t = NULL;
    if (!(g_test_dev ? coli_vk_tensor_ensure2(&t, codes, sc, fmt, I, O, gs) : coli_vk_tensor_ensure(&t, codes, sc, fmt, I, O, gs))) {
        CHECK(0, "matmul fmt %d: upload", fmt); return; }
    float *ref = malloc((size_t)S * O * sizeof *ref);
    for (int s = 0; s < S; s++) for (int o = 0; o < O; o++) {
        double a = 0; for (int i = 0; i < I; i++) a += (double)x[(size_t)s * I + i] * W[(size_t)o * I + i];
        ref[(size_t)s * O + o] = (float)a;
    }
    VkcBuf *xb = vkc_buf((xo + (size_t)S * I) * 4, VKC_DEV), *yb = vkc_buf((yo + (size_t)S * O) * 4, VKC_DOWN);
    vkc_begin();
    vkc_write(xb, xo, x, (size_t)S * I * 4);
    int ok = vkc_matmul(t, xb, xo, yb, yo, S);
    vkc_submit(1);
    float *y = (float *)vkc_ptr(yb) + yo;
    double e = relerr(y, ref, (size_t)S * O, 1e-3);
    CHECK(ok && e < g_mm_tol && !bad(y, (size_t)S * O), "matmul fmt %d S %d I %d O %d xo %zu yo %zu: ok %d err %.2e", fmt, S, I, O, xo, yo, ok, e);
    if (g_mm_out) memcpy(g_mm_out, y, (size_t)S * O * sizeof(float));
    vkc_free(xb); vkc_free(yb); coli_vk_tensor_free(t);
    free(x); free(W); free(codes); free(sc); free(ref);
}

/* ---- norm ------------------------------------------------------------------------- */
static void test_norm(int flags, int inplace) {
    int rows = 3, per = 4, D = 40, seg = 52, rowst = per * seg + 7;  /* a gap after each head, like q's gate */
    size_t n = (size_t)rows * rowst;
    float *x = fvec(n, 2.f), *w = fvec((size_t)per * D, 0.5f), *ref = malloc(n * sizeof *ref);
    memcpy(ref, x, n * sizeof *ref);
    int wmod = flags & VKC_NORM_L2 ? 0 : per;
    for (int r = 0; r < rows; r++) for (int j = 0; j < per; j++) {
        const float *xs = x + r * rowst + j * seg; float *ys = ref + r * rowst + j * seg;
        double ss = 0; for (int i = 0; i < D; i++) ss += (double)xs[i] * xs[i];
        float rr = flags & VKC_NORM_L2 ? 1.f / sqrtf((float)ss + 1e-6f) : 1.f / sqrtf((float)(ss / D) + 1e-6f);
        const float *ws = w + (wmod ? (j % wmod) * D : 0);
        for (int i = 0; i < D; i++) {
            float wv = flags & VKC_NORM_NOW ? 1.f : flags & VKC_NORM_ADD1 ? 1.f + ws[i] : ws[i];
            ys[i] = xs[i] * rr * wv * 0.5f;
        }
    }
    VkcBuf *xb = up(x, n), *wb = up(w, (size_t)per * D), *yb = inplace ? xb : vkc_buf(n * 4, VKC_DEV);
    if (!inplace) { vkc_begin(); vkc_write(yb, 0, x, n * 4); vkc_submit(1); }
    VkcNorm p = {rows * per, D, per, 0, rowst, seg, 0, rowst, seg, 0, wmod, flags, 1e-6f, 0.5f};
    vkc_begin(); int ok = vkc_norm(xb, wb, yb, &p); vkc_submit(1);
    float *y = down(yb, 0, n);
    double e = relerr(y, ref, n, 1e-3);
    CHECK(ok && e < 1e-5, "norm flags %d inplace %d: err %.2e", flags, inplace, e);
    vkc_free(xb); vkc_free(wb); if (!inplace) vkc_free(yb);
    free(x); free(w); free(ref); free(y);
}

/* ---- rope ------------------------------------------------------------------------- */
static void test_rope(void) {
    int rows = 5, heads = 3, hd = 16, seg = 24, half = 4, rowst = heads * seg;
    size_t n = (size_t)rows * rowst;
    float *x = fvec(n, 1.f), *ref = malloc(n * sizeof *ref), *cs = malloc((size_t)rows * half * 2 * sizeof *cs);
    memcpy(ref, x, n * sizeof *ref);
    for (int r = 0; r < rows; r++) for (int j = 0; j < half; j++) {
        float ang = (float)(1000 + 37 * r) * powf(10000.f, -2.f * j / (2 * half));
        cs[(r * half + j) * 2] = cosf(ang); cs[(r * half + j) * 2 + 1] = sinf(ang);
        for (int h = 0; h < heads; h++) {
            float *v = ref + r * rowst + h * seg, a = v[j], b = v[j + half];
            v[j] = a * cs[(r * half + j) * 2] - b * cs[(r * half + j) * 2 + 1];
            v[j + half] = b * cs[(r * half + j) * 2] + a * cs[(r * half + j) * 2 + 1];
        }
    }
    (void)hd;
    VkcBuf *xb = up(x, n), *cb = up(cs, (size_t)rows * half * 2);
    VkcRope p = {rows * heads, heads, 0, rowst, seg, half, 0, half * 2};
    vkc_begin(); int ok = vkc_rope(xb, cb, &p); vkc_submit(1);
    float *y = down(xb, 0, n);
    double e = relerr(y, ref, n, 1e-3);
    CHECK(ok && e < 1e-6, "rope: err %.2e", e);
    vkc_free(xb); vkc_free(cb); free(x); free(ref); free(cs); free(y);
}

/* An automatic chunk must not force 64 rows when even one row exhausts the
 * measured budget. Explicit overrides and small caps remain exact. */
static void test_chunk_rows(void) {
    setenv("COLI_VK_CHAIN_ROWS", "auto", 1);
    setenv("COLI_VK_CHAIN_ROWS_MAX", "8192", 1);
    CHECK(vkc_chunk_rows("test-tight-memory", SIZE_MAX / 4) == 1,
          "automatic chunk exceeded a one-row memory budget");
    setenv("COLI_VK_CHAIN_ROWS_MAX", "7", 1);
    CHECK(vkc_chunk_rows("test-small-cap", 1) == 7, "automatic chunk exceeded its cap");
    setenv("COLI_VK_CHAIN_ROWS", "3", 1);
    CHECK(vkc_chunk_rows("test-explicit", SIZE_MAX / 4) == 3, "explicit chunk ignored");
    unsetenv("COLI_VK_CHAIN_ROWS"); unsetenv("COLI_VK_CHAIN_ROWS_MAX");
}

/* ---- attention ---------------------------------------------------------------------- */
static void test_attn_hk(int S, int pos_base, int H, int KVH, int hd, int use_list) {
    int cap = 300, gsz = hd, qseg = hd + gsz, koff = 64;
    int T = pos_base + S;
    size_t kn = (size_t)koff + (size_t)KVH * cap * hd;
    float *q = fvec((size_t)S * H * qseg, 1.f), *kc = fvec(kn, 1.f), *vc = fvec(kn, 1.f);
    float scale = 1.f / sqrtf((float)hd);
    int selrow = 1 + T;
    int *sel = calloc((size_t)S * selrow, sizeof *sel);
    for (int s = 0; s < S; s++) {
        int vis = pos_base + s + 1;
        if (!use_list || s % 2) { sel[s * selrow] = -1; continue; }
        int n = 0;
        for (int t = 0; t < vis; t++) if (t % 3 != 1 || t == vis - 1) sel[s * selrow + 1 + n++] = t;
        sel[s * selrow] = n;
    }
    float *ref = malloc((size_t)S * H * hd * sizeof *ref);
    for (int s = 0; s < S; s++) for (int h = 0; h < H; h++) {
        int kvh = h / (H / KVH), vis = pos_base + s + 1;
        const int *list = sel + s * selrow;
        int n = list[0] >= 0 ? list[0] : vis;
        double *sc = malloc(n * sizeof *sc), mx = -1e300, sum = 0;
        for (int j = 0; j < n; j++) {
            int t = list[0] >= 0 ? list[1 + j] : j;
            double a = 0; for (int d = 0; d < hd; d++) a += (double)q[(size_t)s * H * qseg + h * qseg + d] * kc[koff + ((size_t)kvh * cap + t) * hd + d];
            sc[j] = a * scale; if (sc[j] > mx) mx = sc[j];
        }
        for (int j = 0; j < n; j++) { sc[j] = exp(sc[j] - mx); sum += sc[j]; }
        for (int d = 0; d < hd; d++) {
            double a = 0;
            for (int j = 0; j < n; j++) { int t = list[0] >= 0 ? list[1 + j] : j; a += sc[j] / sum * vc[koff + ((size_t)kvh * cap + t) * hd + d]; }
            ref[((size_t)s * H + h) * hd + d] = (float)a * sigm(q[(size_t)s * H * qseg + h * qseg + hd + d]);
        }
        free(sc);
    }
    VkcBuf *qb = up(q, (size_t)S * H * qseg), *kb = up(kc, kn), *vb = up(vc, kn), *ob = vkc_buf((size_t)S * H * hd * 4, VKC_DOWN);
    VkcBuf *lb = vkc_buf((size_t)S * selrow * 4, VKC_DEV);
    vkc_begin(); vkc_write(lb, 0, sel, (size_t)S * selrow * 4); vkc_submit(1);
    VkcAttn p = {S, H, KVH, hd, pos_base, cap, 0, H * qseg, qseg, hd, H * qseg, qseg, 1, 0, H * hd, 0, use_list ? selrow : 0, scale, koff, koff};
    vkc_begin(); int ok = vkc_attn(qb, kb, vb, ob, qb, lb, &p); vkc_submit(1);
    float *o = vkc_ptr(ob);
    /* chain_attn_flash's operands are f16: its bound where it may have run */
    int fl = vkc_attn_flash_rows();
    double e = relerr(o, ref, (size_t)S * H * hd, 1e-3);
    double tol = fl > 0 && S >= fl && !use_list && hd % 64 == 0 && 16 % (H / KVH) == 0 ? 2e-3 : 2e-5;
    CHECK(ok && e < tol, "attn S %d pos %d H %d/%d hd %d list %d (block from %d, flash from %d): err %.2e", S, pos_base, H, KVH, hd,
          use_list, vkc_attn_block_rows(), fl, e);
    vkc_free(qb); vkc_free(kb); vkc_free(vb); vkc_free(ob); vkc_free(lb);
    free(q); free(kc); free(vc); free(sel); free(ref);
}
static void test_attn(int S, int pos_base, int hd, int use_list) { test_attn_hk(S, pos_base, 4, 2, hd, use_list); }

/* chain_vae.comp (Qwen-Image's VAE): a band's 3x3 taps, with and without the 2x upsample,
 * and the DupUp3D shortcut added; both are copies, so the bytes must be the CPU's. */
static void test_vae_ops(int H, int W, int C, int u2, int y0, int nb) {
    int Ho = H << u2, Wo = W << u2, C9 = 9 * C;
    size_t xn = (size_t)H * W * C, qn = (size_t)nb * Wo * C9;
    float *x = fvec(xn + 5, 1.f), *ref = malloc(qn * sizeof *ref);
    for (int px = 0; px < nb * Wo; px++)
        for (int kx = 0; kx < 3; kx++) for (int ky = 0; ky < 3; ky++) for (int c = 0; c < C; c++) {
            int row = y0 + px / Wo, col = px % Wo, sy = row + ky - 1, sx = col + kx - 1;
            ref[(size_t)px * C9 + kx * 3 * C + ky * C + c] =
                sy >= 0 && sy < Ho && sx >= 0 && sx < Wo ? x[5 + ((size_t)(sy >> u2) * W + (sx >> u2)) * C + c] : 0.f;
        }
    VkcBuf *xb = up(x, xn + 5), *qb = vkc_buf((qn + 3) * 4, VKC_DOWN);
    VkcVae p = {0, C, H, W, u2, Ho, Wo, y0, (int)qn, 5, 3, 0, 0};
    vkc_begin(); int ok = vkc_vae(xb, qb, NULL, &p); ok = vkc_submit(1) && ok;
    CHECK(ok && memcmp((float *)vkc_ptr(qb) + 3, ref, qn * 4) == 0, "vae taps H %d W %d C %d up %d rows %d+%d", H, W, C, u2, y0, nb);
    /* DupUp3D: the half-size map x (C channels) into a 2H x 2W map of Co channels */
    int Co = 4, *tab = malloc((size_t)Co * 4 * sizeof *tab);
    for (int i = 0; i < Co * 4; i++) tab[i] = (i * 7 + 3) % C;
    size_t yn = (size_t)(2 * H) * (2 * W) * Co;
    float *y = fvec(yn, 1.f), *yr = malloc(yn * sizeof *yr);
    for (int px = 0; px < 4 * H * W; px++) for (int o = 0; o < Co; o++) {
        int row = px / (2 * W), col = px % (2 * W), src = (row >> 1) * W + (col >> 1), sub = (row & 1) * 2 + (col & 1);
        yr[(size_t)px * Co + o] = y[(size_t)px * Co + o] + x[5 + (size_t)src * C + tab[o * 4 + sub]];
    }
    VkcBuf *yb = up(y, yn), *tb = vkc_buf((size_t)Co * 4 * sizeof(int), VKC_DEV);
    VkcVae d = {1, C, H, W, 1, 2 * H, 2 * W, 0, (int)yn, 5, 0, Co, C};
    vkc_begin(); ok = vkc_write(tb, 0, tab, (size_t)Co * 4 * sizeof(int)) && vkc_vae(xb, yb, tb, &d); ok = vkc_submit(1) && ok;
    float *got = down(yb, 0, yn);
    CHECK(ok && memcmp(got, yr, yn * 4) == 0, "vae dup H %d W %d C %d", H, W, C);
    vkc_free(xb); vkc_free(qb); vkc_free(yb); vkc_free(tb);
    free(x); free(ref); free(tab); free(y); free(yr); free(got);
}

/* GATE_ADD (Qwen-Image's gated residual): y = a + e[col] * b, a gate per column. */
static void test_ew_gate(void) {
    int R = 5, D = 70;
    size_t n = (size_t)R * D;
    float *a = fvec(n, 2.f), *b = fvec(n, 2.f), *e = fvec(D + 3, 1.f), *ref = malloc(n * sizeof *ref);
    for (size_t i = 0; i < n; i++) ref[i] = a[i] + e[3 + i % D] * b[i];
    VkcBuf *ab = up(a, n), *bb = up(b, n), *eb = up(e, D + 3);
    VkcEw p = {VKC_EW_GATE_ADD, (int)n, D, 1, 0, 0, 0, 0, 0, 0, 3, 1.f};
    vkc_begin(); int ok = vkc_ew(ab, ab, bb, NULL, eb, &p); vkc_submit(1);   /* in place, as the chain runs it */
    float *y = down(ab, 0, n);
    double err = relerr(y, ref, n, 1e-3);
    CHECK(ok && err < 1e-6, "ew gate-add: err %.2e", err);
    free(y); vkc_free(ab); vkc_free(bb); vkc_free(eb); free(a); free(b); free(e); free(ref);
}

/* A diffusion step's attention (vkc_attn_full, Qwen-Image's DiT): S query rows, each over
 * all T key rows, token-major in one buffer (q, then k and v at their offsets), heads of
 * hd; S and T away from the shader's tiles of 64. */
static void test_attn_full_k(int S, int T, int H, int hd, int coop);
static void test_attn_full(int S, int T, int H, int hd) { test_attn_full_k(S, T, H, hd, 0); }
static void test_attn_full_k(int S, int T, int H, int hd, int coop) {
    int D = H * hd, qrow = D + 8, kvrow = D + 24, qoff = 4, koff = qoff + S * qrow + 16, voff = koff + T * kvrow + 8;
    size_t n = (size_t)voff + (size_t)T * kvrow + 8;
    float *x = fvec(n, 1.f), scale = 1.f / sqrtf((float)hd);
    float *ref = malloc((size_t)S * D * sizeof *ref);
    double *sc = malloc((size_t)T * sizeof *sc);
    for (int s = 0; s < S; s++) for (int h = 0; h < H; h++) {
        double mx = -1e300, sum = 0;
        for (int t = 0; t < T; t++) {
            double a = 0;
            for (int d = 0; d < hd; d++) a += (double)x[qoff + (size_t)s * qrow + h * hd + d] * x[koff + (size_t)t * kvrow + h * hd + d];
            sc[t] = a * scale; if (sc[t] > mx) mx = sc[t];
        }
        for (int t = 0; t < T; t++) { sc[t] = exp(sc[t] - mx); sum += sc[t]; }
        for (int d = 0; d < hd; d++) {
            double a = 0;
            for (int t = 0; t < T; t++) a += sc[t] / sum * x[voff + (size_t)t * kvrow + h * hd + d];
            ref[(size_t)s * D + h * hd + d] = (float)a;
        }
    }
    VkcBuf *xb = up(x, n), *ob = vkc_buf(((size_t)S * D + 32) * 4, VKC_DOWN);
    VkcAttnFull p = {S, T, H, hd, qoff, qrow, koff, voff, kvrow, 32, D, scale};
    vkc_begin(); int ok = coop ? vkc_attn_full_coop(xb, ob, &p) : vkc_attn_full(xb, ob, &p); ok = vkc_submit(1) && ok;
    double e = relerr((float *)vkc_ptr(ob) + 32, ref, (size_t)S * D, 1e-3);
    /* the matrix units' operands are f16: a bound of f16's rounding */
    CHECK(ok && e < (coop ? 4e-3 : 2e-5), "attn full%s S %d T %d H %d hd %d: err %.2e", coop ? " coop" : "", S, T, H, hd, e);
    vkc_free(xb); vkc_free(ob); free(x); free(ref); free(sc);
}

/* MiMo's attention (vkc_attn_w): row s at pos = pos_base + s sees the positions
 * max(0, pos - win + 1)..pos (win 0: from 0); position t sits in row t % ring of the
 * cache (ring 0: row t), rows position-major (kv_pm) or head-major; V has its own head
 * dim vd; with a sink, one more logit per head joins the softmax denominator. */
static void test_attn_win(int S, int pos_base, int H, int KVH, int hd, int vd, int win, int ring, int sink, int kv_pm) {
    int rows = ring ? ring : pos_base + S + 3, koff = 32, voff = 16, qrow = H * hd + 40;
    size_t kn = (size_t)koff + (size_t)rows * KVH * hd, vn = (size_t)voff + (size_t)rows * KVH * vd;
    float *q = fvec((size_t)S * qrow, 1.f), *kc = fvec(kn, 1.f), *vc = fvec(vn, 1.f), *snk = fvec(H + 5, 2.f);
    float scale = 1.f / sqrtf((float)hd);
    float *ref = malloc((size_t)S * H * vd * sizeof *ref);
    for (int s = 0; s < S; s++) for (int h = 0; h < H; h++) {
        int kvh = h / (H / KVH), pos = pos_base + s, first = win ? (pos - win + 1 > 0 ? pos - win + 1 : 0) : 0, n = pos - first + 1;
        double *sc = malloc(n * sizeof *sc), mx = sink ? snk[5 + h] : -1e300, sum = 0;
        for (int j = 0; j < n; j++) {
            int r = ring ? (first + j) % ring : first + j;
            size_t kb = koff + (kv_pm ? ((size_t)r * KVH + kvh) : ((size_t)kvh * rows + r)) * hd;
            double a = 0; for (int d = 0; d < hd; d++) a += (double)q[(size_t)s * qrow + h * hd + d] * kc[kb + d];
            sc[j] = a * scale; if (sc[j] > mx) mx = sc[j];
        }
        if (sink) sum = exp(snk[5 + h] - mx);
        for (int j = 0; j < n; j++) { sc[j] = exp(sc[j] - mx); sum += sc[j]; }
        for (int d = 0; d < vd; d++) {
            double a = 0;
            for (int j = 0; j < n; j++) {
                int r = ring ? (first + j) % ring : first + j;
                size_t vb = voff + (kv_pm ? ((size_t)r * KVH + kvh) : ((size_t)kvh * rows + r)) * vd;
                a += sc[j] / sum * vc[vb + d];
            }
            ref[((size_t)s * H + h) * vd + d] = (float)a;
        }
        free(sc);
    }
    VkcBuf *qb = up(q, (size_t)S * qrow), *kb = up(kc, kn), *vb = up(vc, vn), *sb = up(snk, H + 5);
    VkcBuf *ob = vkc_buf((size_t)S * H * vd * 4, VKC_DOWN);
    VkcAttnW p = {{S, H, KVH, hd, pos_base, rows, 0, qrow, hd, 0, 0, 0, 0, 0, H * vd, 0, 0, scale, koff, voff},
                  win, ring, vd, kv_pm, sink, 5};
    vkc_begin(); int ok = vkc_attn_w(qb, kb, vb, ob, NULL, NULL, sink ? sb : NULL, &p); vkc_submit(1);
    float *o = vkc_ptr(ob);
    double e = relerr(o, ref, (size_t)S * H * vd, 1e-3);
    CHECK(ok && e < 2e-5, "attn window S %d pos %d hd %d vd %d win %d ring %d sink %d pm %d: err %.2e",
          S, pos_base, hd, vd, win, ring, sink, kv_pm, e);
    if (S >= 16 && !getenv("COLI_VK_ATTN_SLICE")) {
        size_t bytes = (size_t)S * H * vd * sizeof(float);
        float *whole = malloc(bytes); memcpy(whole, o, bytes);
        setenv("COLI_VK_ATTN_SLICE", "1", 1);
        vkc_begin(); ok = vkc_attn_w(qb, kb, vb, ob, NULL, NULL, sink ? sb : NULL, &p);
        ok = vkc_submit(1) && ok;
        CHECK(ok && memcmp(whole, vkc_ptr(ob), bytes) == 0,
              "attention slices changed bytes S %d H %d/%d win %d sink %d", S, H, KVH, win, sink);
        unsetenv("COLI_VK_ATTN_SLICE"); free(whole);
    }
    vkc_free(qb); vkc_free(kb); vkc_free(vb); vkc_free(sb); vkc_free(ob);
    free(q); free(kc); free(vc); free(snk); free(ref);
}

/* ---- DeltaNet: convolution and recurrence ------------------------------------------ */
static void conv_ref(int S, int CD, int CK, const float *in, const float *w, float *ring, float *out, int order, int snap_row, float *snap) {
    for (int s = 0; s < S; s++) for (int c = 0; c < CD; c++) {
        float *rg = ring + c * (CK - 1); const float *wc = w + c * CK; float cur = in[s * CD + c], a;
        if (order == 0) { a = 0; for (int k = 0; k < CK - 1; k++) a += wc[k] * rg[k]; a += wc[CK - 1] * cur; }
        else { a = wc[CK - 1] * cur; for (int k = 0; k < CK - 1; k++) a += wc[k] * rg[k]; }
        out[s * CD + c] = a / (1.f + expf(-a));
        for (int k = 0; k < CK - 2; k++) rg[k] = rg[k + 1];
        rg[CK - 2] = cur;
        if (s == snap_row) memcpy(snap + c * (CK - 1), rg, (CK - 1) * sizeof(float));
    }
}
static void test_dnconv(int order) {
    int CD = 37, CK = 4, S1 = 5, S2 = 1, nh = CK - 1;
    float *w = fvec((size_t)CD * CK, 0.5f), *in = fvec((size_t)(S1 + S2) * CD, 1.f);
    float *ring = calloc((size_t)CD * nh, sizeof *ring), *out = malloc((size_t)(S1 + S2) * CD * sizeof *out);
    float *snap = calloc((size_t)CD * nh, sizeof *snap);
    conv_ref(S1, CD, CK, in, w, ring, out, order, 2, snap);
    float *snap_ref = malloc((size_t)CD * nh * sizeof *snap_ref); memcpy(snap_ref, snap, (size_t)CD * nh * sizeof *snap_ref);
    conv_ref(S2, CD, CK, in + S1 * CD, w, ring, out + S1 * CD, order, -1, NULL);
    VkcBuf *wb = up(w, (size_t)CD * CK), *ib = up(in, (size_t)(S1 + S2) * CD);
    VkcBuf *rb = vkc_buf((size_t)CD * nh * 2 * 4 + 64, VKC_DEV), *ob = vkc_buf((size_t)(S1 + S2) * CD * 4, VKC_DEV);
    int ro = 16, so = 16 + CD * nh;   /* ring and its snapshot in one buffer, at offsets */
    VkcDnConv p1 = {S1, CD, CK, 0, CD, 0, CD, 2, order, 0, ro, so};
    VkcDnConv p2 = {S2, CD, CK, S1 * CD, CD, S1 * CD, CD, -1, order, 0, ro, so};
    vkc_begin(); int ok = vkc_dnconv(ib, wb, rb, ob, rb, &p1); vkc_submit(0);     /* left in flight */
    vkc_begin(); ok &= vkc_dnconv(ib, wb, rb, ob, rb, &p2); vkc_submit(1);
    float *o = down(ob, 0, (size_t)(S1 + S2) * CD), *r = down(rb, ro, (size_t)CD * nh), *sn = down(rb, so, (size_t)CD * nh);
    double e1 = relerr(o, out, (size_t)(S1 + S2) * CD, 1e-3), e2 = relerr(r, ring, (size_t)CD * nh, 1e-3), e3 = relerr(sn, snap_ref, (size_t)CD * nh, 1e-3);
    CHECK(ok && e1 < 1e-6 && e2 == 0 && e3 == 0, "dnconv order %d: out %.2e ring %.2e snap %.2e", order, e1, e2, e3);
    vkc_free(wb); vkc_free(ib); vkc_free(rb); vkc_free(ob);
    free(w); free(in); free(ring); free(out); free(snap); free(snap_ref); free(o); free(r); free(sn);
}

static float softplus(float z) { return z > 20.f ? z : log1pf(expf(z)); }
/* qwen36.c's deltanet recurrence for S rows, from the conv output */
static void rec_ref(int S, int VH, int KH, int KD, int VD, const float *cv, const float *b, const float *a, const float *z,
                    const float *alog, const float *dtb, const float *nw, float *st, float *y, int sig_gate, int snap_row, float *snap) {
    int Kt = KH * KD, rep = VH / KH, CD = 2 * Kt + VH * VD;
    for (int s = 0; s < S; s++) for (int h = 0; h < VH; h++) {
        const float *row = cv + (size_t)s * CD;
        float q[256], k[256], o[128];
        double sq = 1e-6, sk = 1e-6;
        for (int d = 0; d < KD; d++) { q[d] = row[(h / rep) * KD + d]; k[d] = row[Kt + (h / rep) * KD + d]; sq += (double)q[d] * q[d]; sk += (double)k[d] * k[d]; }
        for (int d = 0; d < KD; d++) { q[d] = (float)(q[d] / sqrt(sq) / sqrt((double)KD)); k[d] = (float)(k[d] / sqrt(sk)); }
        float beta = sigm(b[s * VH + h]), decay = expf(-expf(alog[h]) * softplus(a[s * VH + h] + dtb[h]));
        float *Sh = st + (size_t)h * KD * VD;
        for (int t = 0; t < KD * VD; t++) Sh[t] *= decay;
        for (int v = 0; v < VD; v++) {
            float kv = 0; for (int d = 0; d < KD; d++) kv += k[d] * Sh[d * VD + v];
            float delta = (row[2 * Kt + h * VD + v] - kv) * beta;
            for (int d = 0; d < KD; d++) Sh[d * VD + v] += k[d] * delta;
        }
        double ms = 0;
        for (int v = 0; v < VD; v++) { float acc = 0; for (int d = 0; d < KD; d++) acc += q[d] * Sh[d * VD + v]; o[v] = acc; ms += (double)acc * acc; }
        float r = 1.f / sqrtf((float)(ms / VD) + 1e-6f);
        for (int v = 0; v < VD; v++) {
            float zv = z[(size_t)s * VH * VD + h * VD + v], g = sig_gate ? sigm(zv) : zv / (1.f + expf(-zv));
            y[(size_t)s * VH * VD + h * VD + v] = o[v] * r * nw[v] * g;
        }
        if (s == snap_row) memcpy(snap + (size_t)h * KD * VD, Sh, (size_t)KD * VD * sizeof(float));
    }
}
static void test_dnrec(int KD, int VD, int VH, int KH, int sig_gate) {
    int S1 = 3, S2 = 2, S = S1 + S2, Kt = KH * KD, CD = 2 * Kt + VH * VD;
    float *cv = fvec((size_t)S * CD, 1.f), *ab = fvec((size_t)S * VH * 2, 2.f), *z = fvec((size_t)S * VH * VD, 2.f);
    float *prm = fvec((size_t)2 * VH + VD, 1.f);
    float *st = fvec((size_t)VH * KD * VD, 0.1f), *st0 = malloc((size_t)VH * KD * VD * sizeof *st0);
    memcpy(st0, st, (size_t)VH * KD * VD * sizeof *st0);
    float *b = malloc((size_t)S * VH * sizeof *b), *a = malloc((size_t)S * VH * sizeof *a);
    for (int s = 0; s < S; s++) for (int h = 0; h < VH; h++) { b[s * VH + h] = ab[s * 2 * VH + h]; a[s * VH + h] = ab[s * 2 * VH + VH + h]; }
    float *y = malloc((size_t)S * VH * VD * sizeof *y), *snap = malloc((size_t)VH * KD * VD * sizeof *snap);
    rec_ref(S1, VH, KH, KD, VD, cv, b, a, z, prm, prm + VH, prm + 2 * VH, st, y, sig_gate, 1, snap);
    rec_ref(S2, VH, KH, KD, VD, cv + (size_t)S1 * CD, b + S1 * VH, a + S1 * VH, z + (size_t)S1 * VH * VD, prm, prm + VH, prm + 2 * VH,
            st, y + (size_t)S1 * VH * VD, sig_gate, -1, NULL);
    VkcBuf *cb = up(cv, (size_t)S * CD), *abb = up(ab, (size_t)S * VH * 2), *zb = up(z, (size_t)S * VH * VD), *pb = up(prm, (size_t)2 * VH + VD);
    size_t SN = (size_t)VH * KD * VD;
    VkcBuf *sb = vkc_buf(2 * SN * 4, VKC_DEV), *yb = vkc_buf((size_t)S * VH * VD * 4, VKC_DEV);
    vkc_begin(); vkc_write(sb, 0, st0, SN * 4); vkc_submit(1);
    VkcDnRec p1 = {S1, VH, KH, VD, Kt, 0, CD, 0, 2 * VH, VH, 2 * VH, 0, VH * VD, 0, VH * VD, 1, sig_gate, 1e-6f,
                   1.f / sqrtf((float)KD), 0, (int)SN, 0};
    VkcDnRec p2 = p1;
    p2.S = S2; p2.cv_off = S1 * CD; p2.b_off = S1 * 2 * VH; p2.a_off = S1 * 2 * VH + VH; p2.z_off = S1 * VH * VD; p2.y_off = S1 * VH * VD; p2.snap_row = -1;
    vkc_begin(); int ok = vkc_dnrec(KD, cb, abb, zb, sb, pb, yb, sb, &p1); vkc_submit(0);
    vkc_begin(); ok &= vkc_dnrec(KD, cb, abb, zb, sb, pb, yb, sb, &p2); vkc_submit(1);
    float *yd = down(yb, 0, (size_t)S * VH * VD), *sd = down(sb, 0, SN), *nd = down(sb, SN, SN);
    double e1 = relerr(yd, y, (size_t)S * VH * VD, 1e-3), e2 = relerr(sd, st, SN, 1e-3), e3 = relerr(nd, snap, SN, 1e-3);
    CHECK(ok && e1 < 5e-5 && e2 < 5e-6 && e3 < 5e-6, "dnrec KD %d VD %d gate %d: y %.2e state %.2e snap %.2e", KD, VD, sig_gate, e1, e2, e3);
    vkc_free(cb); vkc_free(abb); vkc_free(zb); vkc_free(pb); vkc_free(sb); vkc_free(yb);
    free(cv); free(ab); free(z); free(prm); free(st); free(st0); free(b); free(a); free(y); free(snap); free(yd); free(sd); free(nd);
}

/* ---- element-wise ------------------------------------------------------------------- */
static void test_ew(void) {
    int R = 3, D = 50, C = 4, W = C * D;
    size_t nw = (size_t)R * W;
    float *a = fvec(nw, 2.f), *b = fvec(nw, 2.f), *c = fvec(nw, 2.f), *e = fvec(R, 3.f), *ref = malloc(nw * sizeof *ref);
    VkcBuf *ab = up(a, nw), *bb = up(b, nw), *cb = up(c, nw), *eb = up(e, R), *yb = vkc_buf(nw * 4, VKC_DEV);
    struct { int op, flags, n; const char *name; } ops[] = {
        {VKC_EW_ADD, 0, R * D, "add"}, {VKC_EW_COMBINE, 1 | 2 | 4, R * D, "combine"}, {VKC_EW_COMBINE, 2 | 8, R * D, "combine-noresid"},
        {VKC_EW_COMBINE, 1, R * D, "combine-routed"}, {VKC_EW_SWIGLU, 0, R * D, "swiglu"}, {VKC_EW_HC_LOW, 0, R * D, "hc-low"},
        {VKC_EW_HC_MIX, 0, R * D, "hc-mix"}, {VKC_EW_HC_INJ, 0, R * C, "hc-inj"}, {VKC_EW_HC_APPLY, 0, R * W, "hc-apply"},
        {VKC_EW_SCALE, 0, R * D, "scale"}};
    for (size_t k = 0; k < sizeof ops / sizeof *ops; k++) {
        int n = ops[k].n, op = ops[k].op, f = ops[k].flags;
        for (int i = 0; i < n; i++) {
            int r = i / D, d = i % D;
            switch (op) {
            case VKC_EW_ADD: ref[i] = a[i] + b[i]; break;
            case VKC_EW_COMBINE: { float t = f & 1 ? b[r * D + d] : 0.f;
                if (f & 2) t = t + (f & 4 ? sigm(e[r]) : 1.f) * c[r * D + d];
                ref[i] = f & 8 ? t : a[i] + t; break; }
            case VKC_EW_SWIGLU: ref[i] = a[i] / (1.f + expf(-a[i])) * b[i]; break;
            case VKC_EW_HC_LOW: { float v = a[i] / C; ref[i] = v * sigm(v); break; }
            case VKC_EW_HC_MIX: { float v = 0; for (int s = 0; s < C; s++) v += sigm(a[r * W + s * D + d]) * b[r * W + s * D + d]; ref[i] = v / C; break; }
            case VKC_EW_HC_INJ: ref[i] = 2.f * sigm(a[i] / C); break;
            case VKC_EW_HC_APPLY: { int rr = i / W, rem = i % W, s = rem / D, dd = rem % D;
                ref[i] = c[i] + a[rr * C + s] * b[rr * D + dd]; break; }
            case VKC_EW_SCALE: ref[i] = a[i] * (float)C; break;
            }
        }
        if (op == VKC_EW_HC_APPLY) { vkc_begin(); vkc_write(yb, 0, c, nw * 4); vkc_submit(1); }
        VkcEw p = {op, n, D, C, f, 1, 0, 0, 0, 0, 0, (float)C};
        vkc_begin(); int ok = vkc_ew(yb, ab, bb, cb, eb, &p); vkc_submit(1);
        float *y = down(yb, 0, n);
        double err = relerr(y, ref, n, 1e-3);
        CHECK(ok && err < 1e-6, "ew %s: err %.2e", ops[k].name, err);
        free(y);
    }
    vkc_free(ab); vkc_free(bb); vkc_free(cb); vkc_free(eb); vkc_free(yb);
    free(a); free(b); free(c); free(e); free(ref);
}

/* ---- inkling: the short convolution, the scale, the attention ----------------------- */
/* inkling.c's sconv_apply for S rows of C channels, in place, the ring (CK-1 raw inputs) carried */
static void sconv_ref(int S, int C, int CK, float *x, const float *w, float *ring) {
    int P = CK - 1;
    float *col = malloc((size_t)(P + S) * sizeof *col);
    for (int c = 0; c < C; c++) {
        for (int j = 0; j < P; j++) col[j] = ring[c * P + j];
        for (int t = 0; t < S; t++) col[P + t] = x[(size_t)t * C + c];
        for (int t = 0; t < S; t++) {
            float acc = 0.f;
            for (int j = 0; j < CK; j++) acc += w[c * CK + j] * col[t + j];
            x[(size_t)t * C + c] = acc + col[P + t];
        }
        for (int j = 0; j < P; j++) ring[c * P + j] = col[S + j];
    }
    free(col);
}
static void test_sconv(int C, int CK) {
    int S1 = 6, S2 = 1, P = CK - 1, xo = 8, ro = 4;
    size_t nx = (size_t)(S1 + S2) * C;
    float *w = fvec((size_t)C * CK, 0.5f), *x = fvec(nx, 1.f), *ring = fvec((size_t)C * P, 1.f);
    float *xr = malloc(nx * sizeof *xr), *rr = malloc((size_t)C * P * sizeof *rr);
    memcpy(xr, x, nx * sizeof *xr); memcpy(rr, ring, (size_t)C * P * sizeof *rr);
    sconv_ref(S1, C, CK, xr, w, rr);
    sconv_ref(S2, C, CK, xr + (size_t)S1 * C, w, rr);
    VkcBuf *wb = up(w, (size_t)C * CK), *xb = vkc_buf((xo + nx) * 4, VKC_DEV), *rb = vkc_buf((ro + (size_t)C * P) * 4, VKC_DEV);
    vkc_begin(); vkc_write(xb, xo, x, nx * 4); vkc_write(rb, ro, ring, (size_t)C * P * 4); vkc_submit(1);
    VkcSconv p1 = {0, S1, C, CK, xo, C, 0, ro, 0, 1.f}, p2 = {0, S2, C, CK, xo + S1 * C, C, 0, ro, 0, 1.f};
    vkc_begin(); int ok = vkc_sconv(xb, wb, rb, &p1); vkc_submit(0);     /* left in flight */
    vkc_begin(); ok &= vkc_sconv(xb, wb, rb, &p2); vkc_submit(1);
    float *o = down(xb, xo, nx), *r = down(rb, ro, (size_t)C * P);
    double e1 = relerr(o, xr, nx, 1e-3), e2 = relerr(r, rr, (size_t)C * P, 1e-3);
    CHECK(ok && e1 < 1e-6 && e2 == 0, "sconv C %d CK %d: out %.2e ring %.2e", C, CK, e1, e2);
    /* the scalar multiply and divide over n floats at an offset */
    int n = 3 * C + 5; float fc = 24.f;
    for (int mode = 1; mode <= 2; mode++) {
        vkc_begin(); vkc_write(xb, xo, x, (size_t)n * 4); vkc_submit(1);
        VkcSconv ps = {mode, 0, 0, 0, xo, 0, 0, 0, n, fc};
        vkc_begin(); ok = vkc_sconv(xb, NULL, NULL, &ps); vkc_submit(1);
        float *y = down(xb, xo, n), *ref = malloc((size_t)n * sizeof *ref);
        for (int i = 0; i < n; i++) ref[i] = mode == 1 ? x[i] * fc : x[i] / fc;
        double e = relerr(y, ref, n, 1e-6);
        CHECK(ok && e < 1e-6, "sconv mode %d: err %.2e", mode, e);
        free(y); free(ref);
    }
    vkc_free(wb); vkc_free(xb); vkc_free(rb);
    free(w); free(x); free(ring); free(xr); free(rr); free(o); free(r);
}
/* inkling.c's attention() for S rows from pos_base: the K/V of every position in kt/vt
 * ([T][KVH*hd]), r [S][H*dr], relp [dr][ext], tau [S]; the device sees the positions
 * before pos_base through a ring of cap rows (t % cap, the latest such t < pos_base)
 * and the step's own rows in a scratch, k at ks, v at vs, kvd floats a row. */
static void test_relattn(int S, int pos_base, int window, int cap, int hd, int dr, int ext) {
    int H = 4, KVH = 2, kvd = KVH * hd, T = pos_base + S, koff = 32, roff = 16, poff = 24, toff = 8;
    float *q = fvec((size_t)S * H * hd, 1.f), *kt = fvec((size_t)T * kvd, 1.f), *vt = fvec((size_t)T * kvd, 1.f);
    float *r = fvec((size_t)S * H * dr, 0.5f), *relp = fvec((size_t)dr * ext, 0.5f), *tau = malloc(S * sizeof *tau);
    for (int s = 0; s < S; s++) tau[s] = 1.f + (rnd() % 1000) / 2000.f;
    float scale = 1.f / (float)hd;
    float *ref = malloc((size_t)S * H * hd * sizeof *ref);
    for (int s = 0; s < S; s++) for (int h = 0; h < H; h++) {
        int kvh = h / (H / KVH), qpos = pos_base + s, t0 = window > 0 && qpos - window + 1 > 0 ? qpos - window + 1 : 0;
        int n = qpos - t0 + 1;
        double *sc = malloc(n * sizeof *sc), mx = -1e300, sum = 0;
        for (int j = 0; j < n; j++) {
            int t = t0 + j, dist = qpos - t;
            double a = 0; for (int d = 0; d < hd; d++) a += (double)q[((size_t)s * H + h) * hd + d] * kt[(size_t)t * kvd + kvh * hd + d];
            double b = 0; if (dist < ext) for (int k = 0; k < dr; k++) b += (double)r[((size_t)s * H + h) * dr + k] * relp[k * ext + dist];
            sc[j] = tau[s] * (a * scale + b); if (sc[j] > mx) mx = sc[j];
        }
        for (int j = 0; j < n; j++) { sc[j] = exp(sc[j] - mx); sum += sc[j]; }
        for (int d = 0; d < hd; d++) {
            double a = 0; for (int j = 0; j < n; j++) a += sc[j] / sum * vt[(size_t)(t0 + j) * kvd + kvh * hd + d];
            ref[((size_t)s * H + h) * hd + d] = (float)a;
        }
        free(sc);
    }
    /* the ring as the host keeps it after the positions before pos_base, and the scratch */
    size_t nr = (size_t)koff + (size_t)KVH * cap * hd;
    float *kr = calloc(nr, sizeof *kr), *vr = calloc(nr, sizeof *vr);
    for (int t = 0; t < pos_base; t++) for (int h = 0; h < KVH; h++) {
        memcpy(kr + koff + ((size_t)h * cap + t % cap) * hd, kt + (size_t)t * kvd + h * hd, hd * sizeof(float));
        memcpy(vr + koff + ((size_t)h * cap + t % cap) * hd, vt + (size_t)t * kvd + h * hd, hd * sizeof(float));
    }
    size_t ks = 4, vs = ks + (size_t)S * kvd, nsc = vs + (size_t)S * kvd;
    float *scr = calloc(nsc, sizeof *scr);
    memcpy(scr + ks, kt + (size_t)pos_base * kvd, (size_t)S * kvd * sizeof(float));
    memcpy(scr + vs, vt + (size_t)pos_base * kvd, (size_t)S * kvd * sizeof(float));
    float *rbuf = calloc(roff + (size_t)S * H * dr, sizeof *rbuf), *pbuf = calloc(poff + (size_t)dr * ext, sizeof *pbuf), *tbuf = calloc(toff + S, sizeof *tbuf);
    memcpy(rbuf + roff, r, (size_t)S * H * dr * sizeof(float)); memcpy(pbuf + poff, relp, (size_t)dr * ext * sizeof(float));
    memcpy(tbuf + toff, tau, S * sizeof(float));
    VkcBuf *qb = up(q, (size_t)S * H * hd), *kb = up(kr, nr), *vb = up(vr, nr), *sb = up(scr, nsc), *rb = up(rbuf, roff + (size_t)S * H * dr);
    VkcBuf *pb = up(pbuf, poff + (size_t)dr * ext), *tb = up(tbuf, toff + S), *ob = vkc_buf((size_t)S * H * hd * 4, VKC_DOWN);
    VkcRelAttn p = {S, H, KVH, hd, pos_base, cap, window, ext, dr, 0, H * hd, 0, H * hd, koff, koff, (int)ks, (int)vs, kvd,
                    roff, H * dr, poff, toff, scale};
    vkc_begin(); int ok = vkc_relattn(qb, kb, vb, ob, sb, rb, pb, tb, &p); vkc_submit(1);
    double e = relerr((float *)vkc_ptr(ob), ref, (size_t)S * H * hd, 1e-3);
    CHECK(ok && e < 2e-5, "relattn S %d pos %d window %d cap %d hd %d d_rel %d ext %d: err %.2e", S, pos_base, window, cap, hd, dr, ext, e);
    vkc_free(qb); vkc_free(kb); vkc_free(vb); vkc_free(sb); vkc_free(rb); vkc_free(pb); vkc_free(tb); vkc_free(ob);
    free(q); free(kt); free(vt); free(r); free(relp); free(tau); free(ref); free(kr); free(vr); free(scr); free(rbuf); free(pbuf); free(tbuf);
}
/* the shared experts joining the routed sum: y = routed; y += w_j[r] * sh_j[r] in order */
static void test_weighted_add(void) {
    int R = 3, D = 70, NS = 2;
    float *routed = fvec((size_t)R * D, 2.f), *sh = fvec((size_t)NS * R * D, 2.f), *w = fvec((size_t)NS * R, 1.f), *ref = malloc((size_t)R * D * sizeof *ref);
    memcpy(ref, routed, (size_t)R * D * sizeof *ref);
    for (int j = 0; j < NS; j++) for (int r = 0; r < R; r++) for (int d = 0; d < D; d++) ref[r * D + d] += w[j * R + r] * sh[((size_t)j * R + r) * D + d];
    VkcBuf *yb = up(routed, (size_t)R * D), *sb = up(sh, (size_t)NS * R * D), *wb = up(w, (size_t)NS * R);
    vkc_begin(); int ok = 1;
    for (int j = 0; j < NS; j++) { VkcEw p = {VKC_EW_HC_APPLY, R * D, D, 1, 0, 1, 0, j * R, j * R * D, 0, 0, 1.f}; ok &= vkc_ew(yb, wb, sb, NULL, NULL, &p); }
    vkc_submit(1);
    float *y = down(yb, 0, (size_t)R * D);
    double e = relerr(y, ref, (size_t)R * D, 1e-3);
    CHECK(ok && e < 1e-6, "weighted add: err %.2e", e);
    vkc_free(yb); vkc_free(sb); vkc_free(wb); free(routed); free(sh); free(w); free(ref); free(y);
}
static void test_inkling(void) {
    test_sconv(37, 4); test_sconv(6144, 4); test_sconv(9, 1); test_sconv(5, 9);
    test_relattn(1, 0, 0, 64, 16, 4, 32);              /* the first position, global */
    test_relattn(5, 200, 0, 300, 32, 16, 64);          /* global, distances past ext (bias 0) */
    test_relattn(1, 40, 16, 16, 32, 4, 16);            /* decode, a sliding window over a wrapped ring */
    test_relattn(20, 30, 16, 16, 16, 4, 16);           /* a step that wraps the ring more than once */
    test_relattn(130, 3, 0, 200, 24, 4, 200);          /* two tiles of positions */
    test_relattn(3, 513, 512, 512, 128, 16, 512);      /* the real window and ring, hd 128 */
    test_relattn(2, 9, 0, 64, 256, 64, 8);             /* the largest head and bank */
    test_weighted_add();
    for (int k = 0; k < 4; k++) { int f[4] = {1, 4, 10, 11}; test_matmul(f[k], 1, 6144, 48, 0, 0); }
    test_matmul(10, 1, 24576, 16, 0, 0);               /* past chain_gemv's staging: qmatmul.comp */
    test_matmul(4, 1, 24576, 16, 0, 0);
    VkcNorm np = {2, 6144, 1, 0, 6144, 6144, 0, 6144, 6144, 0, 0, 0, 1e-6f, 1.f};
    float *x = fvec(2 * 6144, 1.f), *w = fvec(6144, 1.f), *ref = malloc(2 * 6144 * sizeof *ref);
    for (int r = 0; r < 2; r++) { double ms = 0; for (int i = 0; i < 6144; i++) ms += (double)x[r * 6144 + i] * x[r * 6144 + i];
        float rr = 1.f / sqrtf((float)(ms / 6144) + 1e-6f); for (int i = 0; i < 6144; i++) ref[r * 6144 + i] = x[r * 6144 + i] * rr * w[i]; }
    VkcBuf *xb = up(x, 2 * 6144), *wb = up(w, 6144), *yb = vkc_buf(2 * 6144 * 4, VKC_DOWN);
    vkc_begin(); int ok = vkc_norm(xb, wb, yb, &np); vkc_submit(1);
    double e = relerr((float *)vkc_ptr(yb), ref, 2 * 6144, 1e-3);
    CHECK(ok && e < 1e-5, "norm D 6144: err %.2e", e);
    vkc_free(xb); vkc_free(wb); vkc_free(yb); free(x); free(w); free(ref);
}

/* ---- QSA: block keys and selection ----------------------------------------------------- */
static void test_qsa(void) {
    int ID = 12, R = 2, half = 3, cap = 40, nbmax = cap / R, IQ = 2, budget = 6;   /* take = 3 blocks */
    float *ik = fvec((size_t)cap * ID, 1.f), *w = fvec(ID, 0.3f), *pk = calloc((size_t)nbmax * ID, sizeof *pk);
    int nb = 13;   /* blocks 0..12 */
    float *cs = malloc((size_t)nb * half * 2 * sizeof *cs);
    for (int b = 0; b < nb; b++) for (int j = 0; j < half; j++) {
        float ang = (float)(b * R) * powf(10000.f, -2.f * j / (2 * half));
        cs[(b * half + j) * 2] = cosf(ang); cs[(b * half + j) * 2 + 1] = sinf(ang);
    }
    for (int b = 0; b < nb; b++) {
        float pool[64];
        for (int d = 0; d < ID; d++) { pool[d] = 0; for (int r = 0; r < R; r++) pool[d] += ik[(b * R + r) * ID + d] / R; }
        double ss = 0; for (int d = 0; d < ID; d++) ss += (double)pool[d] * pool[d];
        float rr = 1.f / sqrtf((float)(ss / ID) + 1e-6f);
        for (int d = 0; d < ID; d++) pool[d] = pool[d] * rr * (1.f + w[d]);
        float *o = pk + b * ID;
        for (int d = 0; d < ID; d++) o[d] = pool[d];
        for (int j = 0; j < half; j++) {
            float c = cs[(b * half + j) * 2], s = cs[(b * half + j) * 2 + 1], x0 = pool[j], x1 = pool[j + half];
            o[j] = x0 * c - x1 * s; o[j + half] = x1 * c + x0 * s;
        }
    }
    VkcBuf *ib = up(ik, (size_t)cap * ID), *wb = up(w, ID), *pb = vkc_buf((size_t)nbmax * ID * 4, VKC_DEV), *cb = up(cs, (size_t)nb * half * 2);
    /* blocks in two steps, as two forwards complete them */
    VkcQsa p0 = {0, ID, R, 0, half, 0, 0, 0, 0, 0, 0, 0, 0, 1e-6f, 0, 0, 0, 7};
    VkcQsa p1 = p0; p1.b0 = 7; p1.nb = nb - 7;
    VkcBuf *cb2 = up(cs + 7 * half * 2, (size_t)(nb - 7) * half * 2);
    vkc_begin(); int ok = vkc_qsa(ib, wb, pb, cb, NULL, NULL, &p0); ok &= vkc_qsa(ib, wb, pb, cb2, NULL, NULL, &p1); vkc_submit(1);
    float *pd = down(pb, 0, (size_t)nb * ID);
    double e = relerr(pd, pk, (size_t)nb * ID, 1e-3);
    CHECK(ok && e < 1e-5, "qsa block keys: err %.2e", e);
    /* selection: 4 query rows at positions 20..23; give two blocks the same key to test ties */
    memcpy(pk + 5 * ID, pk + 2 * ID, ID * sizeof(float));
    vkc_begin(); vkc_write(pb, 0, pk, (size_t)nbmax * ID * 4); vkc_submit(1);
    int S = 4, pos_base = 20, selrow = 1 + budget + R - 1;
    float *qi = fvec((size_t)S * IQ * ID, 1.f);
    VkcBuf *qb = up(qi, (size_t)S * IQ * ID), *sb = vkc_buf((size_t)S * 2 * nbmax * 4, VKC_DEV), *lb = vkc_buf((size_t)S * selrow * 4, VKC_DEV);
    VkcQsa ps = {1, ID, R, 0, 0, S, pos_base, budget, IQ, 0, IQ * ID, nbmax, selrow, 1e-6f, 0, 0, 0, 0};
    vkc_begin(); ok = vkc_qsa(qb, NULL, pb, NULL, sb, lb, &ps); vkc_submit(1);
    int *sel = malloc((size_t)S * selrow * sizeof *sel);
    ok &= vkc_read(lb, 0, sel, (size_t)S * selrow * 4);
    for (int s = 0; s < S; s++) {
        int vis = pos_base + s + 1, blocks = vis / R, take = blocks < budget / R ? blocks : budget / R;
        float score[64]; int sel_ref[64], n = 0, chosen[64] = {0};
        for (int b = 0; b < blocks; b++) {
            float sc = 0; for (int h = 0; h < IQ; h++) { float a = 0; for (int d = 0; d < ID; d++) a += qi[(s * IQ + h) * ID + d] * pk[b * ID + d]; if (a > 0) sc += a; }
            score[b] = sc / sqrtf((float)ID);
        }
        for (int z = 0; z < take; z++) {   /* the CPU's sort: score desc, block asc */
            int best = -1; for (int b = 0; b < blocks; b++) if (!chosen[b] && (best < 0 || score[b] > score[best])) best = b;
            chosen[best] = 1;
        }
        for (int b = 0; b < blocks; b++) if (chosen[b]) for (int r = 0; r < R; r++) sel_ref[n++] = b * R + r;
        for (int t = blocks * R; t < vis; t++) sel_ref[n++] = t;
        int same = sel[s * selrow] == n;
        for (int j = 0; same && j < n; j++) same = sel[s * selrow + 1 + j] == sel_ref[j];
        CHECK(ok && same, "qsa selection row %d: count %d vs %d", s, sel[s * selrow], n);
    }
    vkc_free(ib); vkc_free(wb); vkc_free(pb); vkc_free(cb); vkc_free(cb2); vkc_free(qb); vkc_free(sb); vkc_free(lb);
    free(ik); free(w); free(pk); free(cs); free(pd); free(qi); free(sel);
}

/* ---- PLE ---------------------------------------------------------------------------- */
static void test_ple(void) {
    int S = 4, C = 3, H = 20, W = C * H, CK = 4, NG = 3, SL = (CK - 1) * NG;
    float *keys = fvec((size_t)S * W, 1.f), *hyp = fvec((size_t)S * W, 1.f), *val = fvec((size_t)S * H, 1.f);
    float *prm = fvec((size_t)3 * W, 0.3f), *conv = fvec((size_t)W * CK, 0.5f), *ring = fvec((size_t)W * SL, 0.5f);
    float *ring0 = malloc((size_t)W * SL * sizeof *ring0); memcpy(ring0, ring, (size_t)W * SL * sizeof *ring0);
    float *href = malloc((size_t)S * W * sizeof *href); memcpy(href, hyp, (size_t)S * W * sizeof *href);
    float *snap = malloc((size_t)W * SL * sizeof *snap);
    float *gated = malloc((size_t)S * W * sizeof *gated), *normv = malloc((size_t)S * W * sizeof *normv);
    for (int s = 0; s < S; s++) for (int b = 0; b < C; b++) {
        float kn[64], qn[64]; double sk = 0, sq = 0;
        for (int d = 0; d < H; d++) { sk += (double)keys[s * W + b * H + d] * keys[s * W + b * H + d]; sq += (double)hyp[s * W + b * H + d] * hyp[s * W + b * H + d]; }
        float rk = 1.f / sqrtf((float)(sk / H) + 1e-6f), rq = 1.f / sqrtf((float)(sq / H) + 1e-6f);
        for (int d = 0; d < H; d++) { kn[d] = keys[s * W + b * H + d] * rk * (1.f + prm[b * H + d]); qn[d] = hyp[s * W + b * H + d] * rq * (1.f + prm[W + b * H + d]); }
        float dot = 0; for (int d = 0; d < H; d++) dot += kn[d] * qn[d]; dot /= sqrtf((float)H);
        float g = sigm(copysignf(sqrtf(fmaxf(fabsf(dot), 1e-6f)), dot));
        double sg = 0;
        for (int d = 0; d < H; d++) { gated[s * W + b * H + d] = g * val[s * H + d]; sg += (double)gated[s * W + b * H + d] * gated[s * W + b * H + d]; }
        float rg = 1.f / sqrtf((float)(sg / H) + 1e-6f);
        for (int d = 0; d < H; d++) normv[s * W + b * H + d] = gated[s * W + b * H + d] * rg * (1.f + prm[2 * W + b * H + d]);
    }
    for (int s = 0; s < S; s++) for (int d = 0; d < W; d++) {
        float a = conv[d * CK + CK - 1] * normv[s * W + d];
        for (int k = 0; k < CK - 1; k++) a += conv[d * CK + k] * ring[d * SL + k * NG];
        href[s * W + d] += gated[s * W + d] + a * sigm(a);
        float *rg = ring + d * SL; for (int k = 0; k < SL - 1; k++) rg[k] = rg[k + 1]; rg[SL - 1] = normv[s * W + d];
        if (s == 1) memcpy(snap + d * SL, rg, SL * sizeof(float));
    }
    VkcBuf *kb = up(keys, (size_t)S * W), *hb = up(hyp, (size_t)S * W), *vb = up(val, (size_t)S * H), *pb = up(prm, (size_t)3 * W);
    VkcBuf *cb = up(conv, (size_t)W * CK), *gb = vkc_buf((size_t)S * W * 4, VKC_DEV), *nb = vkc_buf((size_t)S * W * 4, VKC_DEV);
    VkcBuf *rb = vkc_buf((size_t)2 * W * SL * 4, VKC_DEV);
    vkc_begin(); vkc_write(rb, 0, ring0, (size_t)W * SL * 4); vkc_submit(1);
    VkcPle p0 = {0, S, C, H, CK, NG, 0, 0, 0, -1, 0, 1e-6f, 0, 0, 0};
    VkcPle p1 = {1, S, C, H, CK, NG, 0, 0, 0, 1, W * SL, 1e-6f, 0, 0, 0};
    vkc_begin(); int ok = vkc_ple(kb, hb, vb, pb, gb, nb, NULL, NULL, &p0) && vkc_ple(NULL, hb, NULL, NULL, gb, nb, cb, rb, &p1); vkc_submit(1);
    float *hd = down(hb, 0, (size_t)S * W), *rd = down(rb, 0, (size_t)W * SL), *sd = down(rb, (size_t)W * SL, (size_t)W * SL);
    double e1 = relerr(hd, href, (size_t)S * W, 1e-3), e2 = relerr(rd, ring, (size_t)W * SL, 1e-3), e3 = relerr(sd, snap, (size_t)W * SL, 1e-3);
    CHECK(ok && e1 < 2e-5 && e2 < 2e-5 && e3 < 2e-5, "ple: hyper %.2e ring %.2e snap %.2e", e1, e2, e3);
    vkc_free(kb); vkc_free(hb); vkc_free(vb); vkc_free(pb); vkc_free(cb); vkc_free(gb); vkc_free(nb); vkc_free(rb);
    free(keys); free(hyp); free(val); free(prm); free(conv); free(ring); free(ring0); free(href); free(snap); free(gated); free(normv);
    free(hd); free(rd); free(sd);
}

/* ---- MLA: weight blocks, RoPE and LayerNorm rows, the layer op, the DSA selection ------ */
static float e4m3f(uint8_t b) {   /* the CPU's E4M3_LUT */
    int e = (b >> 3) & 15, m = b & 7;
    float v = e == 0 ? m * 0.001953125f : ldexpf(1.f + m * 0.125f, e - 7);
    return (b & 0x80) ? -v : v;
}
/* A resident [O x I] tensor in fmt with random codes (values about `scale`); W gets the
 * dequantized weights. Groups of 64 for the grouped formats. */
static ColiVkTensor *mk_tensor(int fmt, int O, int I, float scale, float *W) {
    void *codes = NULL; float *sc = NULL; int gs = 0, ng64 = (I + 63) / 64;
    size_t n = (size_t)O * I;
    if (fmt == 1 || fmt == 13) {
        int8_t *q = malloc(n); gs = fmt == 13 ? 64 : 0;
        int ns = fmt == 13 ? O * ng64 : O;
        sc = malloc((size_t)ns * sizeof *sc);
        for (int k = 0; k < ns; k++) sc[k] = scale * (0.5f + (rnd() % 100) / 100.f) / 127.f;
        for (int o = 0; o < O; o++) for (int i = 0; i < I; i++) {
            q[(size_t)o * I + i] = (int8_t)((int)(rnd() % 255) - 127);
            W[(size_t)o * I + i] = q[(size_t)o * I + i] * sc[fmt == 13 ? o * ng64 + i / 64 : o];
        }
        codes = q;
    } else if (fmt == 2 || fmt == 4) {
        size_t rb = (size_t)(I + 1) / 2; uint8_t *q = calloc((size_t)O * rb, 1); gs = fmt == 4 ? 64 : 0;
        int ns = fmt == 4 ? O * ng64 : O;
        sc = malloc((size_t)ns * sizeof *sc);
        for (int k = 0; k < ns; k++) sc[k] = scale * (0.5f + (rnd() % 100) / 100.f) / 7.f;
        for (int o = 0; o < O; o++) for (int i = 0; i < I; i++) {
            int v = (int)(rnd() % 16);
            q[(size_t)o * rb + i / 2] |= (uint8_t)(v << ((i & 1) * 4));
            W[(size_t)o * I + i] = (v - 8) * sc[fmt == 4 ? o * ng64 + i / 64 : o];
        }
        codes = q;
    } else if (fmt == 5) {               /* int3-g64: 16 bytes of 2-bit lows, 8 of high bits, a scale per 64 */
        size_t rb = (size_t)ng64 * 24; uint8_t *q = calloc((size_t)O * rb, 1);
        sc = malloc((size_t)O * ng64 * sizeof *sc);
        for (int k = 0; k < O * ng64; k++) sc[k] = scale * (0.5f + (rnd() % 100) / 100.f) / 4.f;
        for (int o = 0; o < O; o++) for (int i = 0; i < I; i++) {
            int u = (int)(rnd() % 8), g = i / 64, j = i % 64;
            uint8_t *gb = q + (size_t)o * rb + (size_t)g * 24;
            gb[j / 4] |= (uint8_t)((u & 3) << ((j & 3) * 2));
            gb[16 + j / 8] |= (uint8_t)(((u >> 2) & 1) << (j & 7));
            W[(size_t)o * I + i] = (u - 4) * sc[o * ng64 + g];
        }
        codes = q;
    } else if (fmt == 12) {              /* fp8 e4m3, a scale per 64 */
        uint8_t *q = malloc(n); gs = 64;
        sc = malloc((size_t)O * ng64 * sizeof *sc);
        for (int k = 0; k < O * ng64; k++) sc[k] = scale * (0.5f + (rnd() % 100) / 100.f) / 8.f;
        for (int o = 0; o < O; o++) for (int i = 0; i < I; i++) {
            uint8_t b; do b = (uint8_t)rnd(); while ((b & 0x7f) == 0x7f || ((b >> 3) & 15) > 10);
            q[(size_t)o * I + i] = b; W[(size_t)o * I + i] = e4m3f(b) * sc[o * ng64 + i / 64];
        }
        codes = q;
    } else if (fmt == 10) {
        float *w = fvec(n, scale); memcpy(W, w, n * sizeof *W); codes = w;
    } else {                             /* 11 bf16 */
        uint16_t *w = malloc(n * 2);
        for (size_t i = 0; i < n; i++) { w[i] = f2bf(frnd() * scale); W[i] = bf2f(w[i]); }
        codes = w;
    }
    ColiVkTensor *t = NULL;
    int ok = coli_vk_tensor_ensure(&t, codes, sc, fmt, I, O, gs);
    free(codes); free(sc);
    return ok ? t : NULL;
}

/* the per-head weight blocks, both directions, every format, a gate */
static void test_hgemv(int fmt, int trans) {
    int S = 3, H = 3, n = trans ? 24 : 20, I = 128, hstride = n + 9, hoff = 5, O = (H - 1) * hstride + hoff + n + 2;
    float *W = malloc((size_t)O * I * sizeof *W);
    ColiVkTensor *t = mk_tensor(fmt, O, I, 0.1f, W);
    if (!t) { CHECK(0, "hgemv fmt %d: upload", fmt); free(W); return; }
    int xin = trans ? n : I, xseg = xin + 3, xrow = H * xseg + 1, yout = trans ? I : n, yseg = yout + 2, yrow = H * yseg;
    float *x = fvec((size_t)S * xrow + 7, 1.f), *g = fvec((size_t)S * H * n, 2.f);
    float *ref = calloc((size_t)S * yrow, sizeof *ref);
    for (int s = 0; s < S; s++) for (int h = 0; h < H; h++) {
        const float *xs = x + 7 + (size_t)s * xrow + h * xseg;
        float *ys = ref + (size_t)s * yrow + h * yseg;
        if (trans) for (int i = 0; i < I; i++) {
            double a = 0; for (int d = 0; d < n; d++) a += (double)W[(size_t)(h * hstride + hoff + d) * I + i] * xs[d];
            ys[i] = (float)a;
        } else for (int o = 0; o < n; o++) {
            double a = 0; for (int i = 0; i < I; i++) a += (double)W[(size_t)(h * hstride + hoff + o) * I + i] * xs[i];
            ys[o] = (float)a * sigm(g[(size_t)s * H * n + h * n + o]);
        }
    }
    VkcBuf *xb = up(x, (size_t)S * xrow + 7), *gb = up(g, (size_t)S * H * n), *yb = vkc_buf((size_t)S * yrow * 4, VKC_DEV);
    VkcHgemv p = {trans, S, H, n, hstride, hoff, 7, xrow, xseg, 0, yrow, yseg, 0, H * n, !trans};
    vkc_begin(); int ok = vkc_mla_hgemv(t, xb, yb, trans ? NULL : gb, &p); vkc_submit(1);
    float *y = down(yb, 0, (size_t)S * yrow);
    double e = relerr(y, ref, (size_t)S * yrow, 1e-3);
    CHECK(ok && e < 2e-5, "hgemv fmt %d trans %d: ok %d err %.2e", fmt, trans, ok, e);
    vkc_free(xb); vkc_free(gb); vkc_free(yb); coli_vk_tensor_free(t);
    free(W); free(x); free(g); free(ref); free(y);
}

/* interleaved and rotate-half RoPE, in place and into another place; LayerNorm */
static void test_mla_rows(void) {
    int rows = 4, per = 3, seg = 20, rd = 12, xrow = per * seg + 5, yrow = per * (seg + 2), half = rd / 2;
    float *x = fvec((size_t)rows * xrow, 1.f), *cs = malloc((size_t)rows * rd * sizeof *cs);
    for (int r = 0; r < rows; r++) for (int j = 0; j < half; j++) {
        float ang = (float)(500 + 91 * r) * powf(10000.f, -2.f * j / rd);
        cs[r * rd + 2 * j] = cosf(ang); cs[r * rd + 2 * j + 1] = sinf(ang);
    }
    VkcBuf *cb = up(cs, (size_t)rows * rd);
    for (int style = 0; style < 2; style++) for (int inplace = 0; inplace < 2; inplace++) {
        int yr = inplace ? xrow : yrow, ys = inplace ? seg : seg + 2;
        float *ref = calloc((size_t)rows * yr, sizeof *ref);
        if (inplace) memcpy(ref, x, (size_t)rows * xrow * sizeof *ref);
        for (int r = 0; r < rows; r++) for (int j = 0; j < per; j++) {
            const float *v = x + r * xrow + j * seg; float *o = ref + r * yr + j * ys;
            for (int i = 0; i < half; i++) {
                float a = style ? v[2 * i] : v[i], b = style ? v[2 * i + 1] : v[i + half];
                float c = cs[r * rd + 2 * i], sn = cs[r * rd + 2 * i + 1];
                o[i] = a * c - b * sn; o[i + half] = b * c + a * sn;
            }
            for (int i = rd; i < seg; i++) o[i] = v[i];
        }
        VkcBuf *xb = up(x, (size_t)rows * xrow), *yb = inplace ? xb : vkc_buf((size_t)rows * yr * 4, VKC_DEV);
        VkcMlaRow p = {rows * per, per, rd, seg, style, 0, xrow, seg, 0, yr, ys, 0, rd, 0, 0, 0, 0.f};
        vkc_begin(); int ok = vkc_mla_rope(xb, cb, yb, &p); vkc_submit(1);
        float *y = down(yb, 0, (size_t)rows * yr);
        double e = relerr(y, ref, (size_t)rows * yr, 1e-3);
        CHECK(ok && e < 1e-6, "mla rope style %d in place %d: ok %d err %.2e", style, inplace, ok, e);
        if (!inplace) vkc_free(yb);
        vkc_free(xb); free(ref); free(y);
    }
    /* LayerNorm, with and without the bias, into the next segment's place */
    float *prm = fvec(2 * seg, 1.f);
    VkcBuf *pb = up(prm, 2 * seg);
    for (int hb = 0; hb < 2; hb++) {
        float *ref = calloc((size_t)rows * yrow, sizeof *ref);
        for (int r = 0; r < rows; r++) for (int j = 0; j < per; j++) {
            const float *v = x + r * xrow + j * seg; float *o = ref + r * yrow + j * (seg + 2);
            double mu = 0, var = 0; for (int i = 0; i < seg; i++) mu += v[i]; mu /= seg;
            for (int i = 0; i < seg; i++) var += (v[i] - mu) * (v[i] - mu); var /= seg;
            float rr = 1.f / sqrtf((float)var + 1e-6f);
            for (int i = 0; i < seg; i++) o[i] = (float)(v[i] - mu) * rr * prm[i] + (hb ? prm[seg + i] : 0.f);
        }
        VkcBuf *xb = up(x, (size_t)rows * xrow), *yb = vkc_buf((size_t)rows * yrow * 4, VKC_DEV);
        VkcMlaRow p = {rows * per, per, 0, seg, 0, 0, xrow, seg, 0, yrow, seg + 2, 0, 0, 0, seg, hb, 1e-6f};
        vkc_begin(); int ok = vkc_mla_lnorm(xb, pb, yb, &p); vkc_submit(1);
        float *y = down(yb, 0, (size_t)rows * yrow);
        double e = relerr(y, ref, (size_t)rows * yrow, 1e-3);
        CHECK(ok && e < 2e-6, "mla layernorm bias %d: ok %d err %.2e", hb, ok, e);
        vkc_free(xb); vkc_free(yb); free(ref); free(y);
    }
    vkc_free(pb); vkc_free(cb); free(prm); free(x); free(cs);
}

/* The MLA layer op against a double reference of colibri.c's absorbed attention:
 * geometry and formats from the caller; `split` gives the weights as [H*K x Q] and
 * [H*V x K] instead of kv_b; `list` gives every other row a selection list with a
 * skipped entry; `gate` a sigmoid gate on the values. The cache holds random rows
 * before pos_base (from kv_start); the step writes S more. Two frames: the step's
 * projections, then (after the host's selection lists) the attention. */
typedef struct { int H, Q, R, V, K, q_lora, D, S, pos_base, kv_start, style, split, list, gate, fq, fkv, fo; } MlaCase;
static void rmsref(double *y, const double *x, const float *w, int n, float eps) {
    double ss = 0; for (int i = 0; i < n; i++) ss += x[i] * x[i];
    double r = 1.0 / sqrt(ss / n + eps); for (int i = 0; i < n; i++) y[i] = x[i] * r * w[i];
}
static void test_mla(MlaCase c) {
    int H = c.H, Q = c.Q, R = c.R, V = c.V, K = c.K, D = c.D, S = c.S, pb = c.pos_base, QR = Q + R, KR = K + R;
    int cap = pb + S + 3, qin = c.q_lora > 0 ? c.q_lora : D, half = R / 2;
    float eps = 1e-5f, scale = 1.f / sqrtf((float)QR) * 1.1f;
    float *Wqa = c.q_lora ? malloc((size_t)c.q_lora * D * 4) : NULL, *Wqb = malloc((size_t)H * QR * qin * 4);
    float *Wkva = malloc((size_t)KR * D * 4), *Wo = malloc((size_t)D * H * V * 4);
    float *Wkvb = malloc((size_t)H * (Q + V) * K * 4), *Wk = malloc((size_t)H * K * Q * 4), *Wv = malloc((size_t)H * V * K * 4);
    VkcMla m = {H, Q, R, V, K, D, c.q_lora, eps, scale, c.style, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, 0, 0};
    if (c.q_lora) m.q_a = mk_tensor(c.fq, c.q_lora, D, 1.f / sqrtf((float)D) * 2.f, Wqa);
    m.q_b = mk_tensor(c.fq, H * QR, qin, 1.f / sqrtf((float)qin) * 2.f, Wqb);
    m.kv_a = mk_tensor(c.fkv, KR, D, 1.f / sqrtf((float)D) * 2.f, Wkva);
    if (c.split) { m.k_abs = mk_tensor(c.fkv, H * K, Q, 1.f / sqrtf((float)Q), Wk); m.v_abs = mk_tensor(c.fkv, H * V, K, 1.f / sqrtf((float)K), Wv); }
    else {
        m.kv_b = mk_tensor(c.fkv, H * (Q + V), K, 1.f / sqrtf((float)K), Wkvb);
        for (int h = 0; h < H; h++) {      /* the same weights seen as the two halves */
            for (int d = 0; d < Q; d++) for (int i = 0; i < K; i++) Wk[((size_t)h * K + i) * Q + d] = Wkvb[((size_t)h * (Q + V) + d) * K + i];
            for (int v = 0; v < V; v++) for (int i = 0; i < K; i++) Wv[((size_t)h * V + v) * K + i] = Wkvb[((size_t)h * (Q + V) + Q + v) * K + i];
        }
    }
    m.o = mk_tensor(c.fo, D, H * V, 1.f / sqrtf((float)(H * V)), Wo);
    float *prm = malloc((size_t)(qin + K + 8) * 4);
    for (int i = 0; i < qin + K + 8; i++) prm[i] = 0.5f + (rnd() % 100) / 100.f;
    m.prm = up(prm, (size_t)qin + K + 8); m.q_norm = 3; m.kv_norm = 3 + qin;
    /* the cache: rows before pos_base as the device holds them (normalized, rotated) */
    float *lat = fvec((size_t)cap * K, 1.f), *rope = R ? fvec((size_t)cap * R, 1.f) : NULL;
    float *x = fvec((size_t)S * D, 1.f), *cs = R ? malloc((size_t)S * R * 4) : NULL, *gate = fvec((size_t)S * H * V, 2.f);
    for (int s = 0; s < S; s++) for (int j = 0; j < half; j++) {
        float ang = (float)(pb + s) * powf(10000.f, -2.f * j / R);
        cs[s * R + 2 * j] = cosf(ang); cs[s * R + 2 * j + 1] = sinf(ang);
    }
    int selrow = 2 + cap;
    int *sel = calloc((size_t)S * selrow, sizeof *sel);
    for (int s = 0; s < S; s++) {
        int vis = pb + s + 1, n = 0;
        if (!c.list || s % 2) { sel[s * selrow] = -1; continue; }
        for (int t = vis - 1; t >= c.kv_start; t -= 2) sel[s * selrow + 1 + n++] = t;   /* newest first, every other */
        sel[s * selrow + 1 + n++] = -1;                                              /* a skipped entry */
        sel[s * selrow] = n;
    }
    /* reference */
    double *q = calloc((size_t)S * H * QR, 8), *kvr = calloc((size_t)S * KR, 8), *qa = calloc((size_t)S * qin, 8);
    double *latr = malloc((size_t)cap * K * 8), *roper = malloc((size_t)cap * (R ? R : 1) * 8);
    for (size_t i = 0; i < (size_t)cap * K; i++) latr[i] = lat[i];
    for (size_t i = 0; R && i < (size_t)cap * R; i++) roper[i] = rope[i];
    for (int s = 0; s < S; s++) {
        const float *xs = x + (size_t)s * D;
        if (c.q_lora) {
            double *a = qa + (size_t)s * qin;
            for (int o = 0; o < c.q_lora; o++) { double t = 0; for (int i = 0; i < D; i++) t += (double)Wqa[(size_t)o * D + i] * xs[i]; a[o] = t; }
            rmsref(a, a, prm + 3, c.q_lora, eps);
        } else for (int i = 0; i < D; i++) qa[(size_t)s * qin + i] = xs[i];
        for (int o = 0; o < H * QR; o++) { double t = 0; for (int i = 0; i < qin; i++) t += (double)Wqb[(size_t)o * qin + i] * qa[(size_t)s * qin + i]; q[(size_t)s * H * QR + o] = t; }
        for (int o = 0; o < KR; o++) { double t = 0; for (int i = 0; i < D; i++) t += (double)Wkva[(size_t)o * D + i] * xs[i]; kvr[(size_t)s * KR + o] = t; }
        rmsref(latr + (size_t)(pb + s) * K, kvr + (size_t)s * KR, prm + 3 + qin, K, eps);
        for (int h = 0; h <= H; h++) {      /* the heads' rotated parts, then the shared key's */
            double *v = h < H ? q + (size_t)s * H * QR + h * QR + Q : kvr + (size_t)s * KR + K, in[256];
            double *o = h < H ? v : roper + (size_t)(pb + s) * R;
            for (int i = 0; i < R; i++) in[i] = v[i];
            for (int i = 0; i < half; i++) {
                double a = c.style ? in[2 * i] : in[i], b = c.style ? in[2 * i + 1] : in[i + half];
                double cc = cs[s * R + 2 * i], sn = cs[s * R + 2 * i + 1];
                o[i] = a * cc - b * sn; o[i + half] = b * cc + a * sn;
            }
        }
    }
    float *ref = malloc((size_t)S * D * 4);
    double *ctx = calloc((size_t)H * V, 8);
    for (int s = 0; s < S; s++) {
        for (int h = 0; h < H; h++) {
            const double *qs = q + (size_t)s * H * QR + h * QR;
            double qabs[1024], clat[1024], *sc = malloc((size_t)cap * 8);
            for (int i = 0; i < K; i++) { double t = 0; for (int d = 0; d < Q; d++) t += (double)Wk[((size_t)h * K + i) * Q + d] * qs[d]; qabs[i] = t; }
            const int *ls = sel + s * selrow;
            int n = ls[0] >= 0 ? ls[0] : pb + s + 1 - c.kv_start, used = 0;
            int *pos = malloc((size_t)(n + 1) * sizeof *pos);
            for (int j = 0; j < n; j++) { int t = ls[0] >= 0 ? ls[1 + j] : c.kv_start + j; if (t >= 0) pos[used++] = t; }
            double mx = -1e300, sum = 0;
            for (int j = 0; j < used; j++) {
                double a = 0; int t = pos[j];
                for (int i = 0; i < K; i++) a += qabs[i] * latr[(size_t)t * K + i];
                for (int i = 0; i < R; i++) a += qs[Q + i] * roper[(size_t)t * R + i];
                sc[j] = a * scale; if (sc[j] > mx) mx = sc[j];
            }
            for (int j = 0; j < used; j++) { sc[j] = exp(sc[j] - mx); sum += sc[j]; }
            for (int i = 0; i < K; i++) { double a = 0; for (int j = 0; j < used; j++) a += sc[j] / sum * latr[(size_t)pos[j] * K + i]; clat[i] = a; }
            for (int v = 0; v < V; v++) {
                double a = 0; for (int i = 0; i < K; i++) a += (double)Wv[((size_t)h * V + v) * K + i] * clat[i];
                if (c.gate) a *= 1.0 / (1.0 + exp(-(double)gate[(size_t)s * H * V + h * V + v]));
                ctx[h * V + v] = a;
            }
            free(sc); free(pos);
        }
        for (int o = 0; o < D; o++) { double t = 0; for (int i = 0; i < H * V; i++) t += (double)Wo[(size_t)o * H * V + i] * ctx[i]; ref[(size_t)s * D + o] = (float)t; }
    }
    /* the device */
    VkcMlaCache cc = {up(lat, (size_t)cap * K), R ? up(rope, (size_t)cap * R) : NULL, cap};
    VkcMlaScratch scr = {0};
    VkcBuf *xb = vkc_buf((size_t)(S * D + 16) * 4, VKC_DEV), *csb = R ? up(cs, (size_t)S * R) : NULL;
    VkcBuf *gb = up(gate, (size_t)S * H * V), *ob = vkc_buf((size_t)S * D * 4, VKC_DEV);
    VkcBuf *dn = vkc_buf((size_t)S * KR * 4, VKC_DOWN), *sb = vkc_buf((size_t)S * selrow * 4, VKC_DEV);
    int ok = vkc_mla_scratch(&scr, &m, S) && vkc_begin() && vkc_write(xb, 16, x, (size_t)S * D * 4) &&
             vkc_mla_qkv(&m, &scr, xb, 16, S, pb, csb, &cc, dn, 0) && vkc_submit(0) &&
             vkc_begin() && vkc_write(sb, 0, sel, (size_t)S * selrow * 4) &&
             vkc_mla_attn(&m, &scr, S, pb, c.kv_start, &cc, c.list ? sb : NULL, 0, selrow, c.gate ? gb : NULL, 0, ob, 0) &&
             vkc_submit(1);
    float *y = down(ob, 0, (size_t)S * D);
    double e = relerr(y, ref, (size_t)S * D, 1e-3);
    /* the new cache rows, as the host's copy receives them */
    float *nl = (float *)vkc_ptr(dn);
    double el = 0, er = 0;
    for (int s = 0; s < S; s++) {
        for (int i = 0; i < K; i++) { double d = fabs(nl[(size_t)s * K + i] - latr[(size_t)(pb + s) * K + i]); if (d > el) el = d; }
        for (int i = 0; i < R; i++) { double d = fabs(nl[(size_t)S * K + (size_t)s * R + i] - roper[(size_t)(pb + s) * R + i]); if (d > er) er = d; }
    }
    CHECK(ok && e < 2e-5 && el < 2e-5 && er < 2e-5 && !bad(y, (size_t)S * D),
          "mla H %d Q %d R %d V %d K %d q_lora %d S %d pos %d kv_start %d style %d split %d list %d gate %d fmt %d/%d/%d: "
          "ok %d err %.2e latent %.2e rope %.2e", H, Q, R, V, K, c.q_lora, S, pb, c.kv_start, c.style, c.split, c.list, c.gate,
          c.fq, c.fkv, c.fo, ok, e, el, er);
    if (getenv("VKC_TEST_VERBOSE")) printf("mla case H %d K %d S %d: err %.2e latent %.2e rope %.2e\n", H, K, S, e, el, er);
    /* the same layer over the split cache (vk_kvsplit.h): a device of a few blocks of 4,
     * the positions before them in the host's rows (lat, rope as given) */
    if (vkc_kvs_ready()) {
        int rows = (c.list ? 16 : 8) * S + 2;   /* blocks of 1, a step of S rows an eighth of the window */
        char rv[32]; snprintf(rv, sizeof rv, "%d", rows); setenv("COLI_VK_KV_DEVICE_ROWS", rv, 1); setenv("COLI_VK_KV_BLOCK", "1", 1);
        VkcKvSplit ks; memset(&ks, 0, sizeof ks);
        int plan = vkc_kv_plan(&ks, "test", 1, (size_t)KR * 4, cap, S, c.list, 0);
        unsetenv("COLI_VK_KV_DEVICE_ROWS"); unsetenv("COLI_VK_KV_BLOCK");
        if (plan == 2) {
            VkcMlaCache dc = {vkc_buf((size_t)ks.rows * K * 4, VKC_DEV), R ? vkc_buf((size_t)ks.rows * R * 4, VKC_DEV) : NULL, ks.rows};
            VkcMlaCache tc = {vkc_buf((size_t)S * K * 4, VKC_DEV), R ? vkc_buf((size_t)S * R * 4, VKC_DEV) : NULL, S};
            VkcKvPart pt[2] = {{1, K, lat, 0, dc.lat, 0}, {1, R ? R : 1, rope, 0, dc.rope, 0}};
            VkcBuf *ob2 = vkc_buf((size_t)S * D * 4, VKC_DEV), *dn2 = vkc_buf((size_t)S * KR * 4, VKC_DOWN);
            vkc_kv_place(&ks, 0, pb, S);
            unsigned long long sp = ks.splits;
            int ok2 = vkc_kv_parts(&ks, (size_t)S * H * (K + 2)) && vkc_begin() && vkc_kv_push(&ks, 0, pt, R ? 2 : 1, pb) &&
                      vkc_write(sb, 0, sel, (size_t)S * selrow * 4) &&
                      vkc_kv_mla_qkv(&ks, 0, &m, &scr, xb, 16, S, pb, csb, &dc, &tc, dn2, 0) &&
                      vkc_kv_mla_attn(&ks, 0, &m, &scr, S, pb, c.kv_start, &dc, c.list ? sb : NULL, 0, selrow, c.gate ? gb : NULL, 0,
                                      ob2, 0, lat, rope) && vkc_submit(1);
            float *y2 = down(ob2, 0, (size_t)S * D);
            double e2 = relerr(y2, ref, (size_t)S * D, 1e-3), el2 = 0;
            float *nl2 = (float *)vkc_ptr(dn2);
            for (size_t i = 0; i < (size_t)S * KR; i++) { double d = fabs(nl2[i] - nl[i]); if (d > el2) el2 = d; }
            CHECK(ok2 && e2 < 2e-5 && el2 == 0 && (pb <= ks.rows || ks.splits > sp),
                  "mla over the split cache H %d K %d S %d pos %d list %d: ok %d err %.2e, new rows %.2e, %llu host parts",
                  H, K, S, pb, c.list, ok2, e2, el2, ks.splits - sp);
            vkc_free(dc.lat); vkc_free(dc.rope); vkc_free(tc.lat); vkc_free(tc.rope); vkc_free(ob2); vkc_free(dn2); free(y2);
        } else CHECK(cap <= rows, "mla over the split cache: no split for cap %d rows %d", cap, rows);
        vkc_kv_free(&ks);
    }
    vkc_mla_scratch_free(&scr);
    vkc_free(cc.lat); vkc_free(cc.rope); vkc_free(xb); vkc_free(csb); vkc_free(gb); vkc_free(ob); vkc_free(dn); vkc_free(sb);
    vkc_free(m.prm);
    ColiVkTensor *ts[7] = {m.q_a, m.q_b, m.kv_a, m.kv_b, m.k_abs, m.v_abs, m.o};
    for (int i = 0; i < 7; i++) if (ts[i]) coli_vk_tensor_free(ts[i]);
    free(Wqa); free(Wqb); free(Wkva); free(Wo); free(Wkvb); free(Wk); free(Wv); free(prm); free(lat); free(rope);
    free(x); free(cs); free(gate); free(sel); free(q); free(kvr); free(qa); free(latr); free(roper); free(ref); free(ctx); free(y);
}

/* The DSA selection against colibri.c's, bit for bit: quarter-step queries and keys
 * and power-of-two scales make every score exact on both sides, and the many zero
 * scores (every head negative) make ties the selection must break as the CPU does. */
/* negw: every head weight <= 0, so every score is 0 or below and the zeros tie at the
 * threshold far beyond top-k (the second band's order over long chunks) */
static void test_dsa(int S, int pos_base, int IH, int ID, int topk, int force, int negw) {
    int T = pos_base + S, krow = ID + 3, qrow = IH * ID + 1, wrow = IH + 2, selrow = 1 + topk + 2, scrow = T + 5;
    /* the buffers start at offsets 1 (queries), 2 (weights) and 3 (keys) */
    float *q = malloc((size_t)(S * qrow + 1) * 4), *k = malloc((size_t)(T * krow + 3) * 4), *w = malloc((size_t)(S * wrow + 2) * 4);
    for (int i = 0; i < S * qrow + 1; i++) q[i] = ((int)(rnd() % 17) - 8) * 0.25f;
    for (int i = 0; i < T * krow + 3; i++) k[i] = ((int)(rnd() % 17) - 8) * 0.25f;
    for (int i = 0; i < S * wrow + 2; i++) w[i] = ((int)(rnd() % 9) - (negw ? 8 : 4)) * 0.5f;
    for (int t = 0; t < T; t += 5) for (int i = 0; i < ID; i++) k[3 + t * krow + i] = k[3 + (t / 5) * krow + i];   /* equal keys: ties */
    float qs = 0.25f, wsc = 0.5f;
    int *ref = calloc((size_t)S * selrow, sizeof *ref), fails0 = fails;
    for (int s = 0; s < S; s++) {
        int nk = pos_base + s + 1, *dst = ref + s * selrow;
        if (nk <= topk && !force) { dst[0] = -1; continue; }
        int keep = nk < topk ? nk : topk;
        float *isc = malloc((size_t)nk * 4), *tmp = malloc((size_t)nk * 4);
        for (int t = 0; t < nk; t++) {
            float a = 0;
            for (int h = 0; h < IH; h++) {
                float d0 = 0; for (int i = 0; i < ID; i++) d0 += q[1 + s * qrow + h * ID + i] * k[3 + t * krow + i];
                d0 *= qs; if (d0 > 0) a += w[2 + s * wrow + h] * d0;
            }
            isc[t] = a * wsc;
        }
        memcpy(tmp, isc, (size_t)nk * 4);
        for (int a = 0; a < nk; a++) for (int b = a + 1; b < nk; b++) if (tmp[b] > tmp[a]) { float x = tmp[a]; tmp[a] = tmp[b]; tmp[b] = x; }
        float thr = tmp[keep - 1];
        int nd = 0;
        for (int t = 0; t < nk && nd < keep; t++) if (isc[t] > thr) dst[1 + nd++] = t;
        for (int t = 0; t < nk && nd < keep; t++) if (isc[t] == thr) dst[1 + nd++] = t;
        dst[0] = nd;
        free(isc); free(tmp);
    }
    VkcBuf *qb = up(q, (size_t)S * qrow + 1), *kb = up(k, (size_t)T * krow + 3), *wb = up(w, (size_t)S * wrow + 2);
    VkcBuf *scb = vkc_buf((size_t)S * scrow * 4, VKC_DEV), *sb = vkc_buf((size_t)S * selrow * 4, VKC_DOWN);
    VkcDsa p = {S, pos_base, IH, ID, topk, force, 1, qrow, 2, wrow, 3, krow, scrow, selrow, qs, wsc};
    vkc_begin(); int ok = vkc_dsa_select(qb, wb, kb, scb, sb, &p); vkc_submit(1);
    const int *got = (const int *)vkc_ptr(sb);
    CHECK(ok, "dsa S %d pos %d: not recorded", S, pos_base);
    for (int s = 0; ok && s < S && fails == fails0; s++) {
        int n = ref[s * selrow] < 0 ? 0 : ref[s * selrow];
        int same = got[s * selrow] == ref[s * selrow];
        for (int j = 0; same && j < n; j++) same = got[s * selrow + 1 + j] == ref[s * selrow + 1 + j];
        CHECK(same, "dsa S %d pos %d IH %d ID %d topk %d force %d: row %d differs (count %d, expected %d)",
              S, pos_base, IH, ID, topk, force, s, got[s * selrow], ref[s * selrow]);
        if (!same && getenv("DSA_DEBUG")) { for (int j = 0; j < n; j++) printf(" %d/%d", got[s * selrow + 1 + j], ref[s * selrow + 1 + j]); printf("\n"); }
    }
    vkc_free(qb); vkc_free(kb); vkc_free(wb); vkc_free(scb); vkc_free(sb);
    free(q); free(k); free(w); free(ref);
}

/* ---- KDA, mHC, k-pooling ----------------------------------------------------------------- */
/* glm53.c's KDA layer around delta_attention.h's step, S rows over two calls (the state
 * and the window carried on the device between them) */
static void test_kda(int H, int KD, int CK, float xs) {   /* xs: the inputs' scale (small: the l2 eps counts) */
    int VD = KD, P = H * KD, C = 3 * P, S = 5;
    float lb = -5.f, neps = 1e-6f, eps = 1e-5f;
    float *qkv = fvec((size_t)S * C, xs), *f = fvec((size_t)S * P, 1.f), *b = fvec((size_t)S * H, 2.f), *g = fvec((size_t)S * P, 2.f);
    float *taps = fvec((size_t)C * CK, 0.7f), *win = fvec((size_t)C * CK, 1.f), *state = fvec((size_t)H * KD * VD, 0.3f);
    float *prm = malloc((size_t)(H + P + VD) * 4);
    for (int i = 0; i < H; i++) prm[i] = frnd() * 0.5f;
    for (int i = 0; i < P; i++) prm[H + i] = frnd() * 0.5f;
    for (int i = 0; i < VD; i++) prm[H + P + i] = 0.5f + (rnd() % 100) / 100.f;
    /* reference: glm53.c's kda_layer row by row */
    float *rs = malloc((size_t)H * KD * VD * 4), *rw = malloc((size_t)C * CK * 4), *ref = malloc((size_t)S * P * 4);
    memcpy(rs, state, (size_t)H * KD * VD * 4); memcpy(rw, win, (size_t)C * CK * 4);
    float *scratch = malloc((size_t)coli_kda_scratch_floats(H, KD, VD) * 4), *core = malloc((size_t)P * 4);
    float *dk = malloc((size_t)P * 4), *bt = malloc((size_t)H * 4);
    for (int s = 0; s < S; s++) {
        for (int h = 0; h < H; h++) for (int d = 0; d < KD; d++) {
            int i = h * KD + d; float sv = expf(prm[h]) * (f[s * P + i] + prm[H + i]);
            float sg = sv >= 0 ? 1.f / (1.f + expf(-sv)) : expf(sv) / (1.f + expf(sv));
            dk[i] = lb * sg;
        }
        for (int h = 0; h < H; h++) { float v = b[s * H + h]; bt[h] = v >= 0 ? 1.f / (1.f + expf(-v)) : expf(v) / (1.f + expf(v)); }
        coli_kda_step(core, rs, rw, qkv + (size_t)s * C, taps, dk, bt, H, KD, VD, CK, neps, scratch);
        for (int h = 0; h < H; h++) {
            float sq = 0; for (int d = 0; d < VD; d++) sq += core[h * VD + d] * core[h * VD + d];
            float inv = 1.f / sqrtf(sq / VD + eps);
            for (int d = 0; d < VD; d++) {
                float gv = g[s * P + h * VD + d], sg = gv >= 0 ? 1.f / (1.f + expf(-gv)) : expf(gv) / (1.f + expf(gv));
                ref[s * P + h * VD + d] = core[h * VD + d] * inv * prm[H + P + d] * sg;
            }
        }
    }
    /* the device: q, k, v in three blocks [3][S][P] (in_part = S*P) */
    float *blk = malloc((size_t)3 * S * P * 4);
    for (int s = 0; s < S; s++) for (int part = 0; part < 3; part++)
        memcpy(blk + (size_t)part * S * P + (size_t)s * P, qkv + (size_t)s * C + (size_t)part * P, (size_t)P * 4);
    VkcBuf *ib = up(blk, (size_t)3 * S * P), *tb = up(taps, (size_t)C * CK), *wb = up(win, (size_t)C * CK), *sb = up(state, (size_t)H * KD * VD);
    VkcBuf *fb = up(f, (size_t)S * P), *bb = up(b, (size_t)S * H), *gb = up(g, (size_t)S * P), *pb = up(prm, (size_t)H + P + VD);
    VkcBuf *mb = vkc_buf((size_t)S * C * 4, VKC_DEV), *yb = vkc_buf((size_t)S * P * 4, VKC_DEV);
    int ok = 1, s0 = 0;
    for (int part = 0; part < 2 && ok; part++) {
        int n = part ? S - 3 : 3;
        VkcKdaConv cp = {n, C, CK, P, s0 * P, P, S * P, 0, C, 0, 0};
        VkcKdaRec rp = {n, H, VD, P, 0, C, s0 * P, P, s0 * H, H, s0 * P, P, s0 * P, P, 0, 0, lb, neps, eps};
        ok = vkc_begin() && vkc_kda_conv(ib, tb, wb, mb, &cp) && vkc_kda_rec(KD, mb, fb, bb, gb, pb, sb, yb, &rp) && vkc_submit(part);
        s0 += n;
    }
    float *y = down(yb, 0, (size_t)S * P), *st2 = down(sb, 0, (size_t)H * KD * VD), *w2 = down(wb, 0, (size_t)C * CK);
    double e = relerr(y, ref, (size_t)S * P, 1e-3), es = relerr(st2, rs, (size_t)H * KD * VD, 1e-3), ew = relerr(w2, rw, (size_t)C * CK, 1e-3);
    CHECK(ok && e < 1e-5 && es < 1e-5 && ew == 0 && !bad(y, (size_t)S * P), "kda H %d KD %d K %d: ok %d out %.2e state %.2e window %.2e",
          H, KD, CK, ok, e, es, ew);
    if (getenv("VKC_TEST_VERBOSE")) printf("kda H %d KD %d: out %.2e state %.2e\n", H, KD, e, es);
    vkc_free(ib); vkc_free(tb); vkc_free(wb); vkc_free(sb); vkc_free(fb); vkc_free(bb); vkc_free(gb); vkc_free(pb); vkc_free(mb); vkc_free(yb);
    free(qkv); free(f); free(b); free(g); free(taps); free(win); free(state); free(prm); free(rs); free(rw); free(ref);
    free(scratch); free(core); free(dk); free(bt); free(blk); free(y); free(st2); free(w2);
}

/* hyper_connections.h's pre and post against the split, collapse and write back, the
 * mean, and the clamped SwiGLU */
static void test_mhc(int H, int D, int iters) {
    int S = 3, HD = H * D, nm = (2 + H) * H;
    float eps = 1e-5f, hce = 1e-6f, lim = 1.5f;
    float *x = fvec((size_t)S * HD, 1.f), *fn = fvec((size_t)nm * HD, 0.05f), *br = fvec((size_t)S * D, 1.f);
    float *prm = malloc((size_t)(3 + nm) * 4);
    for (int i = 0; i < 3; i++) prm[i] = 0.5f + (rnd() % 100) / 100.f;
    for (int i = 0; i < nm; i++) prm[3 + i] = frnd() * 0.5f;
    float *rpre = malloc((size_t)S * D * 4), *rpost = malloc((size_t)S * H * 4), *rcomb = malloc((size_t)S * H * H * 4);
    float *rout = malloc((size_t)S * HD * 4), *rmean = malloc((size_t)S * D * 4);
    for (int s = 0; s < S; s++) {
        coli_hc_pre(rpre + s * D, rpost + s * H, rcomb + s * H * H, x + s * HD, fn, prm, prm + 3, H, D, iters, eps, hce);
        coli_hc_post(rout + s * HD, br + s * D, x + s * HD, rpost + s * H, rcomb + s * H * H, H, D);
        for (int d = 0; d < D; d++) { float sm = 0; for (int i = 0; i < H; i++) sm += x[s * HD + i * D + d]; rmean[s * D + d] = sm / H; }
    }
    ColiVkTensor *t = NULL;
    if (!coli_vk_tensor_ensure(&t, fn, NULL, 10, HD, nm, 0)) { CHECK(0, "mhc: upload"); return; }
    VkcBuf *xb = up(x, (size_t)S * HD), *bb = up(br, (size_t)S * D), *pb = up(prm, 3 + (size_t)nm);
    VkcBuf *mb = vkc_buf((size_t)S * nm * 4, VKC_DEV), *hb = vkc_buf((size_t)S * (2 * H + H * H) * 4, VKC_DEV);
    VkcBuf *cb = vkc_buf((size_t)S * D * 4, VKC_DEV), *ob = vkc_buf((size_t)S * HD * 4, VKC_DEV), *eb = vkc_buf((size_t)S * D * 4, VKC_DEV);
    int hrow = 2 * H + H * H;
    VkcMhc sp = {S, H, D, iters, 0, HD, 0, nm, 0, hrow, 0, D, 0, 0, eps, hce, 0.f};
    VkcMhc po = {S, H, D, iters, 0, HD, 0, D, 0, hrow, 0, HD, 0, 0, eps, hce, 0.f};
    int ok = vkc_begin() && vkc_matmul(t, xb, 0, mb, 0, S) && vkc_mhc(VKC_MHC_SPLIT, xb, mb, hb, pb, NULL, &sp) &&
             vkc_mhc(VKC_MHC_COLLAPSE, xb, NULL, hb, NULL, cb, &sp) && vkc_mhc(VKC_MHC_POST, xb, bb, hb, NULL, ob, &po) &&
             vkc_mhc(VKC_MHC_MEAN, xb, NULL, NULL, NULL, eb, &sp) && vkc_submit(1);
    float *hp = down(hb, 0, (size_t)S * hrow), *col = down(cb, 0, (size_t)S * D), *out = down(ob, 0, (size_t)S * HD), *mean = down(eb, 0, (size_t)S * D);
    /* Isolate the nonlinear split from the preceding matrix product: use the
     * device's raw mixes here. End-to-end pre/post/collapse/write-back references
     * above still include the CPU matrix product and retain their own bounds. */
    float *rp = malloc((size_t)S * hrow * 4), *mixs = malloc((size_t)nm * 4);
    float *device_mix = down(mb, 0, (size_t)S * nm);
    for (int s = 0; s < S; s++) {
        float ms = 0; for (int i = 0; i < HD; i++) ms += x[s * HD + i] * x[s * HD + i];
        float ir = 1.f / sqrtf(ms / HD + eps);
        for (int r = 0; r < nm; r++) mixs[r] = device_mix[s * nm + r] * ir;
        coli_hc_split_sinkhorn(rp + s * hrow, rp + s * hrow + H, rp + s * hrow + 2 * H, mixs, prm, prm + 3, H, iters, hce);
    }
    free(device_mix);
    double epre = 0;
    for (int s = 0; s < S; s++) for (int i = 0; i < hrow; i++) { double d = fabs(hp[s * hrow + i] - rp[s * hrow + i]); if (d > epre) epre = d; }
    free(rp); free(mixs);
    double ep = 0, ec = relerr(col, rpre, (size_t)S * D, 1e-3), eo = relerr(out, rout, (size_t)S * HD, 1e-3), em = relerr(mean, rmean, (size_t)S * D, 1e-3);
    for (int s = 0; s < S; s++) {
        for (int i = 0; i < H; i++) { double d = fabs(hp[s * hrow + H + i] - rpost[s * H + i]); if (d > ep) ep = d; }
        for (int i = 0; i < H * H; i++) { double d = fabs(hp[s * hrow + 2 * H + i] - rcomb[s * H * H + i]); if (d > ep) ep = d; }
    }
    CHECK(ok && ep < 2e-5 && epre < 5e-7 && ec < 2e-5 && eo < 2e-5 && em < 1e-6,
          "mhc H %d D %d iters %d: ok %d split %.2e post/comb %.2e collapse %.2e write back %.2e mean %.2e",
          H, D, iters, ok, epre, ep, ec, eo, em);
    if (getenv("VKC_TEST_VERBOSE")) printf("mhc H %d D %d: split %.2e post/comb %.2e collapse %.2e write back %.2e mean %.2e\n", H, D, epre, ep, ec, eo, em);
    /* the clamped SwiGLU, on values both sides of the limit */
    int n = 333;
    float *ga = fvec(n, 3.f), *ua = fvec(n, 3.f), *rr = malloc((size_t)n * 4);
    for (int i = 0; i < n; i++) { float gg = ga[i] > lim ? lim : ga[i], uu = ua[i] < -lim ? -lim : (ua[i] > lim ? lim : ua[i]); rr[i] = gg / (1.f + expf(-gg)) * uu; }
    VkcBuf *gab = up(ga, n), *uab = up(ua, n), *yb = vkc_buf((size_t)n * 4, VKC_DEV);
    VkcMhc sw = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, n, 0.f, 0.f, lim};
    ok = vkc_begin() && vkc_mhc(VKC_SWIGLU_CLAMP, gab, uab, NULL, NULL, yb, &sw) && vkc_submit(1);
    float *yy = down(yb, 0, n);
    double es = relerr(yy, rr, n, 1e-3);
    CHECK(ok && es < 1e-6, "swiglu clamp: ok %d err %.2e", ok, es);
    vkc_free(xb); vkc_free(bb); vkc_free(pb); vkc_free(mb); vkc_free(hb); vkc_free(cb); vkc_free(ob); vkc_free(eb);
    vkc_free(gab); vkc_free(uab); vkc_free(yb); coli_vk_tensor_free(t);
    free(x); free(fn); free(br); free(prm); free(rpre); free(rpost); free(rcomb); free(rout); free(rmean);
    free(hp); free(col); free(out); free(mean); free(ga); free(ua); free(rr); free(yy);
}

/* k-pooling against sparse_index.h. Equal gates and no bias make each pooled key the
 * exact mean of its members, so with quarter-step values and power-of-two scales the
 * scores are exact on both sides and the selection must be the CPU's slot for slot;
 * a second pass with random gates and bias checks the pooled keys to rounding. */
static void test_kpool(int S, int pos_base, int IH, int ID, int pool, int topk, int tail) {
    int T = pos_base + S, width = coli_sparse_index_width(topk, pool, tail), selrow = width + 1;
    int npool = T / pool, scrow = npool + 1;
    float *keys = malloc((size_t)T * ID * 4), *gates = calloc((size_t)T * ID, 4), *ape = calloc((size_t)pool * ID, 4);
    float *q = malloc((size_t)S * IH * ID * 4), *hw = malloc((size_t)S * IH * 4);
    for (int i = 0; i < T * ID; i++) keys[i] = ((int)(rnd() % 17) - 8) * 0.25f;
    for (int i = 0; i < S * IH * ID; i++) q[i] = ((int)(rnd() % 17) - 8) * 0.25f;
    for (int i = 0; i < S * IH; i++) hw[i] = ((int)(rnd() % 9) - 4) * 0.5f * sqrtf((float)IH);
    for (int t = 0; t < T; t += 7) for (int i = 0; i < ID; i++) keys[t * ID + i] = keys[(t % 3) * ID + i];   /* equal pools: ties */
    unsigned char *valid = malloc((size_t)T); memset(valid, 1, (size_t)T);
    int *ref = malloc((size_t)S * width * sizeof(int));
    float *hwd = malloc((size_t)S * IH * 4);
    for (int i = 0; i < S * IH; i++) hwd[i] = hw[i] / sqrtf((float)IH);
    coli_sparse_index_select_range(ref, q, keys, gates, hwd, ape, valid, T, IH, ID, pool, topk, tail, pos_base, T);
    VkcBuf *kb = up(keys, (size_t)T * ID), *gb = up(gates, (size_t)T * ID), *ab = up(ape, (size_t)pool * ID);
    VkcBuf *qb = up(q, (size_t)S * IH * ID), *wb = up(hw, (size_t)S * IH), *pk = vkc_buf((size_t)(npool + 1) * ID * 4, VKC_DEV);
    VkcBuf *scb = vkc_buf((size_t)S * scrow * 4, VKC_DEV), *sb = vkc_buf((size_t)S * selrow * 4, VKC_DOWN);
    VkcDsaPool kp = {npool, pool, 0, ID, 0, ID, 0, 0, 0, ID};
    VkcDsaPick sp = {S, pos_base, IH, ID, topk, pool, 0, IH * ID, 0, IH, 0, tail, scrow, selrow, sqrtf((float)IH), 1.f / sqrtf((float)ID)};
    int ok = vkc_begin() && vkc_dsa_pool_keys(kb, gb, ab, pk, &kp) && vkc_dsa_pool_select(qb, wb, pk, scb, sb, &sp) && vkc_submit(1);
    const int *got = (const int *)vkc_ptr(sb);
    int same = ok;
    for (int s = 0; same && s < S; s++) {
        same = got[s * selrow] == width;
        for (int i = 0; same && i < width; i++) same = got[s * selrow + 1 + i] == ref[s * width + i];
        if (!same && getenv("DSA_DEBUG")) { for (int i = 0; i < width; i++) printf(" %d/%d", got[s * selrow + 1 + i], ref[s * width + i]); printf("\n"); }
    }
    CHECK(same, "kpool S %d pos %d IH %d ID %d pool %d topk %d tail %d: ok %d, the selection differs", S, pos_base, IH, ID, pool, topk, tail, ok);
    /* the pooled keys with real gates and bias, against sparse_index.h's arithmetic */
    for (int i = 0; i < T * ID; i++) gates[i] = frnd() * 2.f;
    for (int i = 0; i < pool * ID; i++) ape[i] = frnd() * 0.5f;
    float *rk = malloc((size_t)npool * ID * 4);
    for (int p = 0; p < npool; p++) for (int d = 0; d < ID; d++) {
        float mx = -FLT_MAX, tot = 0, mix = 0;
        for (int j = 0; j < pool; j++) { float l = gates[(p * pool + j) * ID + d] + ape[j * ID + d]; if (l > mx) mx = l; }
        for (int j = 0; j < pool; j++) tot += expf(gates[(p * pool + j) * ID + d] + ape[j * ID + d] - mx);
        for (int j = 0; j < pool; j++) mix += expf(gates[(p * pool + j) * ID + d] + ape[j * ID + d] - mx) / tot * keys[(p * pool + j) * ID + d];
        rk[p * ID + d] = mix;
    }
    ok = vkc_begin() && vkc_write(gb, 0, gates, (size_t)T * ID * 4) && vkc_write(ab, 0, ape, (size_t)pool * ID * 4) &&
         vkc_dsa_pool_keys(kb, gb, ab, pk, &kp) && vkc_submit(1);
    float *gk = down(pk, 0, (size_t)npool * ID);
    double e = relerr(gk, rk, (size_t)npool * ID, 1e-3);
    CHECK(ok && e < 2e-6, "kpool keys pool %d ID %d: ok %d err %.2e", pool, ID, ok, e);
    vkc_free(kb); vkc_free(gb); vkc_free(ab); vkc_free(qb); vkc_free(wb); vkc_free(pk); vkc_free(scb); vkc_free(sb);
    free(keys); free(gates); free(ape); free(q); free(hw); free(valid); free(ref); free(hwd); free(rk); free(gk);
}

/* Kimi K3's KDA layer, kimi_k3.c's kda_forward row by row (its scalar path): the window
 * shifted and the taps summed in order, silu; q and k l2-normalized with the eps after
 * the squares, q then scaled; alpha from A = exp(A_log) as the engine keeps it; the
 * recurrence k * ((v - kS) * beta); the output RMSNorm (its sum in double) and gate.
 * S rows over two calls (the state and the window carried on the device). */
static void test_kda_k3(int H, int KD, int CK, float xs) {
    int VD = KD, P = H * KD, C = 3 * P, S = 6;
    float lb = -5.f, neps = 1e-6f, eps = 1e-5f, qscale = 1.f / sqrtf((float)KD);
    float *qkv = fvec((size_t)S * C, xs), *f = fvec((size_t)S * P, 1.f), *b = fvec((size_t)S * H, 2.f), *g = fvec((size_t)S * P, 2.f);
    float *taps = fvec((size_t)C * CK, 0.7f), *win = fvec((size_t)C * CK, 1.f), *state = fvec((size_t)H * KD * VD, 0.3f);
    float *prm = malloc((size_t)(H + P + VD) * 4);
    for (int i = 0; i < H; i++) prm[i] = expf(frnd() * 0.5f);          /* A = exp(A_log), as kimi_k3.c keeps it */
    for (int i = 0; i < P; i++) prm[H + i] = frnd() * 0.5f;
    for (int i = 0; i < VD; i++) prm[H + P + i] = 0.5f + (rnd() % 100) / 100.f;
    float *rs = malloc((size_t)H * KD * VD * 4), *rw = malloc((size_t)C * CK * 4), *ref = malloc((size_t)S * P * 4);
    memcpy(rs, state, (size_t)H * KD * VD * 4); memcpy(rw, win, (size_t)C * CK * 4);
    float *vec = malloc((size_t)C * 4), *kS = malloc((size_t)VD * 4), *oh = malloc((size_t)VD * 4);
    float *qn = malloc((size_t)KD * 4), *kn = malloc((size_t)KD * 4), *al = malloc((size_t)KD * 4);
    for (int s = 0; s < S; s++) {
        for (int c = 0; c < C; c++) {               /* the short convolution, the window oldest first */
            float *wd = rw + (size_t)c * CK;
            for (int j = 0; j < CK - 1; j++) wd[j] = wd[j + 1];
            wd[CK - 1] = qkv[(size_t)s * C + c];
            float acc = 0; for (int j = 0; j < CK; j++) acc += taps[(size_t)c * CK + j] * wd[j];
            vec[c] = acc / (1.f + expf(-acc));
        }
        for (int h = 0; h < H; h++) {
            const float *qh = vec + h * KD, *kh = vec + P + h * KD, *vh = vec + 2 * P + h * VD;
            float sq = 0, sk = 0;
            for (int i = 0; i < KD; i++) { sq += qh[i] * qh[i]; sk += kh[i] * kh[i]; }
            sq = 1.f / sqrtf(sq + neps); sk = 1.f / sqrtf(sk + neps);
            for (int i = 0; i < KD; i++) { qn[i] = qh[i] * sq * qscale; kn[i] = kh[i] * sk; }
            for (int i = 0; i < KD; i++) al[i] = expf(lb * sigm(prm[h] * (f[(size_t)s * P + h * KD + i] + prm[H + h * KD + i])));
            float beta = sigm(b[(size_t)s * H + h]);
            float *St = rs + (size_t)h * KD * VD;
            memset(kS, 0, (size_t)VD * 4);
            for (int k = 0; k < KD; k++) { float *row = St + (size_t)k * VD; for (int v = 0; v < VD; v++) { row[v] *= al[k]; kS[v] += kn[k] * row[v]; } }
            for (int v = 0; v < VD; v++) kS[v] = (vh[v] - kS[v]) * beta;
            memset(oh, 0, (size_t)VD * 4);
            for (int k = 0; k < KD; k++) { float *row = St + (size_t)k * VD; for (int v = 0; v < VD; v++) { row[v] += kn[k] * kS[v]; oh[v] += qn[k] * row[v]; } }
            double ms = 0; for (int v = 0; v < VD; v++) ms += (double)oh[v] * oh[v];
            float r = 1.f / sqrtf((float)(ms / VD) + eps);
            for (int v = 0; v < VD; v++) ref[(size_t)s * P + h * VD + v] = oh[v] * r * prm[H + P + v] * sigm(g[(size_t)s * P + h * VD + v]);
        }
    }
    /* the device: q, k, v in three blocks [3][S][P] (in_part = S*P), as an engine's three matmuls leave them */
    float *blk = malloc((size_t)3 * S * P * 4);
    for (int s = 0; s < S; s++) for (int part = 0; part < 3; part++)
        memcpy(blk + (size_t)part * S * P + (size_t)s * P, qkv + (size_t)s * C + (size_t)part * P, (size_t)P * 4);
    VkcBuf *ib = up(blk, (size_t)3 * S * P), *tb = up(taps, (size_t)C * CK), *wb = up(win, (size_t)C * CK), *sb = up(state, (size_t)H * KD * VD);
    VkcBuf *fb = up(f, (size_t)S * P), *bb = up(b, (size_t)S * H), *gb = up(g, (size_t)S * P), *pb = up(prm, (size_t)H + P + VD);
    VkcBuf *mb = vkc_buf((size_t)S * C * 4, VKC_DEV), *yb = vkc_buf((size_t)S * P * 4, VKC_DEV);
    int ok = 1, s0 = 0;
    for (int part = 0; part < 2 && ok; part++) {
        int n = part ? S - 1 : 1;
        VkcKdaConv cp = {n, C, CK, P, s0 * P, P, S * P, 0, C, 0, 0};
        VkcKdaRec rp = {n, H, VD, P, 0, C, s0 * P, P, s0 * H, H, s0 * P, P, s0 * P, P, 0, 0, lb, neps, eps};
        ok = vkc_begin() && vkc_kda_conv(ib, tb, wb, mb, &cp) &&
             vkc_kda_rec_flags(KD, mb, fb, bb, gb, pb, sb, yb, &rp, VKC_KDA_EXP_A | VKC_KDA_K3) && vkc_submit(part);
        s0 += n;
    }
    float *y = down(yb, 0, (size_t)S * P), *st2 = down(sb, 0, (size_t)H * KD * VD), *w2 = down(wb, 0, (size_t)C * CK);
    double e = relerr(y, ref, (size_t)S * P, 1e-3), es = relerr(st2, rs, (size_t)H * KD * VD, 1e-3), ew = relerr(w2, rw, (size_t)C * CK, 1e-3);
    CHECK(ok && e < 1e-5 && es < 1e-5 && ew == 0 && !bad(y, (size_t)S * P), "kda k3 H %d KD %d K %d: ok %d out %.2e state %.2e window %.2e",
          H, KD, CK, ok, e, es, ew);
    if (getenv("VKC_TEST_VERBOSE")) printf("kda k3 H %d KD %d: out %.2e state %.2e\n", H, KD, e, es);
    vkc_free(ib); vkc_free(tb); vkc_free(wb); vkc_free(sb); vkc_free(fb); vkc_free(bb); vkc_free(gb); vkc_free(pb); vkc_free(mb); vkc_free(yb);
    free(qkv); free(f); free(b); free(g); free(taps); free(win); free(state); free(prm); free(rs); free(rw); free(ref);
    free(vec); free(kS); free(oh); free(qn); free(kn); free(al); free(blk); free(y); free(st2); free(w2);
}

/* kimi_k3.c's res_mix for S rows over nb block snapshots at a row stride, and SiTU-GLU */
static void test_ares(int S, int D, int nb) {
    int nbmax = nb > 0 ? nb + 1 : 1, xrow = D + 3, brow = nbmax * D, yrow = D + 5;
    float eps = 1e-5f;
    float *x = fvec((size_t)S * xrow, 2.f), *bl = fvec((size_t)S * brow, 1.5f), *w = fvec((size_t)D + 7, 0.3f);
    for (int s = 0; s < S; s++) for (int d = 0; d < D; d++) x[(size_t)s * xrow + d] *= 1.f + (float)s;   /* rows of different sizes */
    float *ref = calloc((size_t)S * yrow, 4);
    for (int s = 0; s < S; s++) {
        const float *v[16]; float sc[16];
        for (int e = 0; e < nb; e++) v[e] = bl + (size_t)s * brow + (size_t)e * D;
        v[nb] = x + (size_t)s * xrow;
        for (int e = 0; e <= nb; e++) {
            double ms = 0, dot = 0;
            for (int d = 0; d < D; d++) { double a = v[e][d]; ms += a * a; dot += a * (double)w[7 + d]; }
            sc[e] = (float)(dot / sqrt(ms / D + eps));
        }
        float m = sc[0]; for (int e = 1; e <= nb; e++) if (sc[e] > m) m = sc[e];
        float sum = 0; for (int e = 0; e <= nb; e++) { sc[e] = expf(sc[e] - m); sum += sc[e]; }
        for (int e = 0; e <= nb; e++) sc[e] /= sum;
        for (int d = 0; d < D; d++) { float a = 0; for (int e = 0; e <= nb; e++) a += sc[e] * v[e][d]; ref[(size_t)s * yrow + d] = a; }
    }
    VkcBuf *xb = up(x, (size_t)S * xrow), *bb = up(bl, (size_t)S * brow), *wb = up(w, (size_t)D + 7), *yb = vkc_buf((size_t)S * yrow * 4, VKC_DEV);
    VkcAres ap = {S, D, nb, 0, xrow, 0, brow, 7, 0, yrow, eps};
    int ok = vkc_begin() && vkc_ares_mix(xb, bb, wb, yb, &ap) && vkc_submit(1);
    float *y = down(yb, 0, (size_t)S * yrow);
    for (int s = 0; s < S; s++) for (int d = D; d < yrow; d++) y[(size_t)s * yrow + d] = 0.f;   /* the gaps are not written */
    double e = relerr(y, ref, (size_t)S * yrow, 1e-3);
    CHECK(ok && e < 2e-6 && !bad(y, (size_t)S * yrow), "ares S %d D %d nb %d: ok %d err %.2e", S, D, nb, ok, e);
    if (getenv("VKC_TEST_VERBOSE")) printf("ares S %d D %d nb %d: %.2e\n", S, D, nb, e);
    vkc_free(xb); vkc_free(bb); vkc_free(wb); vkc_free(yb);
    free(x); free(bl); free(w); free(ref); free(y);
}
static void test_situ(float b1, float b2) {
    int n = 777;
    float *ga = fvec(n, 3.f * b1), *ua = fvec(n, 3.f * b2), *rr = malloc((size_t)n * 4);
    const float tiny[] = {0.f, 0x1p-40f, -0x1p-40f, 0x1p-20f, -0x1p-20f, 0x1p-13f, -0x1p-13f};
    for (int i = 0; i < 7; i++) { ga[i] = b1 * tiny[i]; ua[i] = b2 * tiny[i]; }
    for (int i = 0; i < n; i++) rr[i] = b1 * tanhf(ga[i] / b1) * sigm(ga[i]) * b2 * tanhf(ua[i] / b2);
    VkcBuf *gb = up(ga, n), *ub = up(ua, n), *yb = vkc_buf((size_t)n * 4 + 64, VKC_DEV);
    VkcSitu sp = {n, 0, 0, 16, b1, b2};
    int ok = vkc_begin() && vkc_situ(gb, ub, yb, &sp) && vkc_submit(1);
    float *y = down(yb, 16, n);
    double e = relerr(y, rr, n, 1e-3);
    CHECK(ok && e < 1e-6 && !bad(y, n), "situ b1 %g b2 %g: ok %d err %.2e", b1, b2, ok, e);
    for (int i = 0; i < 7; i++)
        CHECK(fabs((double)y[i] - rr[i]) <= 2e-6 * fabs(rr[i]),
              "situ tiny b1 %g b2 %g at %d: got %.9g ref %.9g", b1, b2, i, y[i], rr[i]);
    vkc_free(gb); vkc_free(ub); vkc_free(yb); free(ga); free(ua); free(rr); free(y);
}

/* ---- DeepSeek V4.1 / V4 attention (chain_dsv4.comp) ------------------------------------- */
static float bf16r(float f) { uint32_t u; memcpy(&u, &f, 4); if ((u & 0x7f800000u) != 0x7f800000u) u += 0x7fffu + ((u >> 16) & 1u);
                              u &= 0xffff0000u; memcpy(&f, &u, 4); return f; }
/* sparse_attn.h's scalar kernel (and DeepSeek V4's: the weights and the output to bf16)
 * over lists of window rows (e < nwin), compressed rows (nwin + j) and skipped entries */
static void test_ds_attn(int S, int H, int hd, int nwin, int ncmp, int cnt, int flags) {
    int qrow = H * hd + 3, lrow = cnt + 2, orow = H * hd + 3;
    float *q = fvec((size_t)S * qrow, 1.f), *win = fvec((size_t)(nwin * hd + 5), 1.f), *cmp = fvec((size_t)(ncmp * hd + 7), 1.f);
    float *sink = fvec((size_t)H + 2, 2.f), *ref = calloc((size_t)S * orow, 4), *sc = malloc((size_t)cnt * 4);
    int *list = malloc((size_t)S * lrow * 4);
    float *kvall = malloc((size_t)(nwin + ncmp) * hd * 4);
    memcpy(kvall, win + 5, (size_t)nwin * hd * 4); memcpy(kvall + (size_t)nwin * hd, cmp + 7, (size_t)ncmp * hd * 4);
    for (int s = 0; s < S; s++)
        for (int j = 0; j < lrow; j++) {
            int r = (int)(rnd() % 5);
            list[s * lrow + j] = s == S - 1 && S > 1 ? -1                                     /* a row with nothing to read */
                               : r == 0 ? -1 : r < 3 ? (int)(rnd() % (unsigned)nwin) : nwin + (int)(rnd() % (unsigned)ncmp);
        }
    for (int s = 0; s < S; s++)
        for (int h = 0; h < H; h++) {
            const float *qq = q + (size_t)s * qrow + 1 + h * hd;
            float *o = ref + (size_t)s * orow + 2 + h * hd;
            const int *idx = list + s * lrow + 1;
            if (!(flags & 1)) { coli_sparse_attend_scalar(o, qq, kvall, idx, cnt, hd, sink[1 + h], 1.f / sqrtf((float)hd), sc); continue; }
            float best = -1e30f;   /* DeepSeek V4's reference: w to bf16 before the value sum, the output to bf16 */
            for (int j = 0; j < cnt; j++) {
                if (idx[j] < 0) { sc[j] = -INFINITY; continue; }
                float d = 0; for (int i = 0; i < hd; i++) d += qq[i] * kvall[(size_t)idx[j] * hd + i];
                sc[j] = d * (1.f / sqrtf((float)hd)); if (sc[j] > best) best = sc[j];
            }
            float den = expf(sink[1 + h] - best);
            for (int i = 0; i < hd; i++) o[i] = 0;
            for (int j = 0; j < cnt; j++) {
                if (idx[j] < 0) continue;
                float w = expf(sc[j] - best); den += w; w = bf16r(w);
                for (int i = 0; i < hd; i++) o[i] += w * kvall[(size_t)idx[j] * hd + i];
            }
            for (int i = 0; i < hd; i++) o[i] = bf16r(o[i] / den);
        }
    VkcBuf *qb = up(q, (size_t)S * qrow), *wb = up(win, (size_t)nwin * hd + 5), *cb = up(cmp, (size_t)ncmp * hd + 7);
    VkcBuf *sb = up(sink, (size_t)H + 2), *ob = vkc_buf((size_t)S * orow * 4, VKC_DEV), *lb = vkc_buf((size_t)S * lrow * 4, VKC_DEV);
    vkc_begin(); vkc_write(lb, 0, list, (size_t)S * lrow * 4); vkc_submit(1);
    VkcDsAttn p = {S, H, hd, cnt, 1, lrow, nwin, 5, 7, 1, qrow, 2, orow, 1, flags, 1.f / sqrtf((float)hd)};
    vkc_begin(); int ok = vkc_dsv4_attn(qb, wb, cb, lb, sb, ob, &p); vkc_submit(1);
    float *got = down(ob, 0, (size_t)S * orow);
    double e = 0;
    for (int s = 0; s < S; s++) { double r = relerr(got + (size_t)s * orow + 2, ref + (size_t)s * orow + 2, (size_t)H * hd, 1e-3); if (r > e) e = r; }
    double tol = (flags & 1) ? 8e-3 : 2e-5;
    CHECK(ok && e < tol && !bad(got, (size_t)S * orow), "dsv4 attn S %d H %d hd %d cnt %d flags %d: rel err %.2e", S, H, hd, cnt, flags, e);
    if (!(flags & 1) && S > 1) {   /* the empty row is exactly zero */
        int z = 1; for (int i = 0; i < H * hd; i++) z &= got[(size_t)(S - 1) * orow + 2 + i] == 0.f;
        CHECK(z, "dsv4 attn: a row with no entry is not zero");
    }
    printf("  dsv4 attn S %d H %d hd %d cnt %d flags %d: rel err %.2e\n", S, H, hd, cnt, flags, e);
    vkc_free(qb); vkc_free(wb); vkc_free(cb); vkc_free(sb); vkc_free(ob); vkc_free(lb);
    free(q); free(win); free(cmp); free(sink); free(ref); free(sc); free(list); free(kvall); free(got);
}

/* deepseek_v41.c's rope_apply: interleaved pairs, forward and inverse, segments at a stride */
static void test_ds_rope(int flags) {
    int rows = 3, per = 4, seg = 40, rd = 16, xrow = per * seg + 2, csrow = rd + 1;
    float *x = fvec((size_t)rows * xrow + 3, 1.f), *ref = malloc(((size_t)rows * xrow + 3) * 4), *cs = malloc(((size_t)rows * csrow + 2) * 4);
    for (int r = 0; r < rows; r++) for (int i = 0; i < rd / 2; i++) {
        float a = (r * 7 + 3) * powf(10000.f, -2.f * i / rd);
        cs[2 + r * csrow + 2 * i] = cosf(a); cs[2 + r * csrow + 2 * i + 1] = sinf(a);
    }
    for (int inv = 0; inv < 2; inv++) {
        memcpy(ref, x, ((size_t)rows * xrow + 3) * 4);
        for (int r = 0; r < rows; r++) for (int j = 0; j < per; j++) {
            float *v = ref + 3 + r * xrow + j * seg + (seg - rd);
            for (int i = 0; i < rd / 2; i++) {
                float c = cs[2 + r * csrow + 2 * i], sn = cs[2 + r * csrow + 2 * i + 1];
                if (inv) sn = -sn;
                float a = v[2 * i], b = v[2 * i + 1];
                v[2 * i] = a * c - b * sn; v[2 * i + 1] = a * sn + b * c;
                if (flags & 1) { v[2 * i] = bf16r(v[2 * i]); v[2 * i + 1] = bf16r(v[2 * i + 1]); }
            }
        }
        VkcBuf *xb = up(x, (size_t)rows * xrow + 3), *cb = up(cs, (size_t)rows * csrow + 2);
        VkcDsRope p = {rows * per, per, rd, 3 + seg - rd, xrow, seg, 2, csrow, inv, flags};
        vkc_begin(); int ok = vkc_dsv4_rope(xb, cb, &p); vkc_submit(1);
        float *got = down(xb, 0, (size_t)rows * xrow + 3);
        double e = relerr(got, ref, (size_t)rows * xrow + 3, 1e-3);
        CHECK(ok && e < ((flags & 1) ? 8e-3 : 1e-6), "dsv4 rope inverse %d flags %d: rel err %.2e", inv, flags, e);
        vkc_free(xb); vkc_free(cb); free(got);
    }
    free(x); free(ref); free(cs);
}

/* deepseek_v41.c's compressor_run (its decode form; the prefill form gives the same groups)
 * and DeepSeek V4's overlapping one (ratio 4: the previous group's rows in the first half,
 * a channel of its own each, the ape bias), the ring carried over three calls */
static void test_ds_compress(int ratio, int overlap, int ape) {
    int D = 24, P = overlap ? 2 * D : D, rows = overlap ? 2 * ratio : ratio, T = 23, pb0 = 3;
    float *kv = fvec((size_t)T * P, 1.f), *sc = fvec((size_t)T * P, 2.f), *ap = fvec((size_t)ratio * P + 4, 1.f);
    float *rk = calloc((size_t)rows * P, 4), *rs = malloc((size_t)rows * P * 4);
    for (int i = 0; i < rows * P; i++) rs[i] = -INFINITY;
    int G = (pb0 + T) / ratio + 1;
    float *ref = calloc((size_t)G * D, 4);
    for (int t = 0; t < T; t++) {
        int pos = pb0 + t, slot = pos % ratio, sr = overlap ? ratio + slot : slot;
        for (int c = 0; c < P; c++) { rk[sr * P + c] = kv[t * P + c]; rs[sr * P + c] = sc[t * P + c] + (ape ? ap[4 + slot * P + c] : 0.f); }
        if ((pos + 1) % ratio) continue;
        for (int d = 0; d < D; d++) {
            float best = -INFINITY;
            for (int k = 0; k < rows; k++) { int col = overlap && k >= ratio ? D + d : d; if (rs[k * P + col] > best) best = rs[k * P + col]; }
            float tot = 0, mix = 0;
            for (int k = 0; k < rows; k++) { int col = overlap && k >= ratio ? D + d : d; float w = expf(rs[k * P + col] - best); tot += w; mix += w * rk[k * P + col]; }
            ref[(size_t)(pos / ratio) * D + d] = mix / tot;
        }
        if (overlap) { memcpy(rk, rk + (size_t)ratio * P, (size_t)ratio * P * 4); memcpy(rs, rs + (size_t)ratio * P, (size_t)ratio * P * 4); }
    }
    VkcBuf *kb = up(kv, (size_t)T * P), *sb = up(sc, (size_t)T * P), *ab = up(ap, (size_t)ratio * P + 4);
    VkcBuf *ring = vkc_buf((size_t)(2 * rows * P + 5) * 4, VKC_DEV), *ob = vkc_buf((size_t)G * D * 4, VKC_DEV);
    float *init = malloc((size_t)(2 * rows * P + 5) * 4);
    for (int i = 0; i < 2 * rows * P + 5; i++) init[i] = i >= 5 + rows * P ? -INFINITY : 0.f;
    vkc_begin(); vkc_write(ring, 0, init, (size_t)(2 * rows * P + 5) * 4); vkc_submit(1);
    int cuts[4] = {0, 5, 6, T}, ok = 1;
    for (int k = 0; k < 3; k++) {
        int n = cuts[k + 1] - cuts[k];
        VkcDsComp p = {n, pb0 + cuts[k], ratio, P, D, overlap, cuts[k] * P, P, cuts[k] * P, P, ape ? 4 : -1, 0, D, 5};
        vkc_begin(); ok &= vkc_dsv4_compress(kb, sb, ring, ab, ob, &p); vkc_submit(k == 2);
    }
    float *got = down(ob, 0, (size_t)G * D);
    int g0 = (pb0 + 1 + ratio - 1) / ratio;   /* the first group completed by a row here */
    double e = relerr(got + (size_t)g0 * D, ref + (size_t)g0 * D, (size_t)(G - 1 - g0) * D, 1e-3);
    CHECK(ok && e < 2e-6, "dsv4 compress ratio %d overlap %d ape %d: rel err %.2e", ratio, overlap, ape, e);
    vkc_free(kb); vkc_free(sb); vkc_free(ab); vkc_free(ring); vkc_free(ob);
    free(kv); free(sc); free(ap); free(rk); free(rs); free(ref); free(got); free(init);
}

/* deepseek_v41.c's indexer_run and candidate_blocks: the scores with the reach and the
 * mask, the candidate blocks, and the top-k list slot for slot. Inputs on a grid of 0.25
 * (weights 0.5, a power-of-two scale) so every score is exact in float on both sides,
 * with equal keys for ties. */
static void ds_topk_ref(const float *score, int width, int topk, int base, int order, int *row) {
    int taken = 0;
    for (int k = 0; k < topk; k++) row[k] = -1;
    for (int k = 0; k < topk; k++) {
        int best = -1;
        for (int j = 0; j < width; j++) {
            if (score[j] == -INFINITY) continue;
            int already = 0; for (int u = 0; u < taken; u++) if (row[u] == j) { already = 1; break; }
            if (already) continue;
            if (best < 0 || score[j] > score[best]) best = j;
        }
        if (best < 0) break;
        row[taken++] = best;
    }
    if (order == 0)
        for (int a = 0; a < taken; a++) for (int b = a + 1; b < taken; b++) if (row[b] < row[a]) { int t = row[a]; row[a] = row[b]; row[b] = t; }
    for (int k = 0; k < taken; k++) row[k] += base;
}
static void ds_cand_ref(const float *score, int width, int lens, int block, int topb, int *keep) {
    int blocks = (width + block - 1) / block;
    float *best = malloc((size_t)(blocks + 1) * 4);
    for (int b = 0; b < blocks; b++) { float top = -INFINITY; for (int i = b * block; i < (b + 1) * block && i < width; i++) if (score[i] > top) top = score[i]; best[b] = top; }
    int last = (lens - 1) / block;
    if (last >= 0 && last < blocks) best[last] = INFINITY;
    int wanted = topb < blocks ? topb : blocks;
    for (int i = 0; i < width; i++) keep[i] = 0;
    for (int pick = 0; pick < wanted; pick++) {
        int chosen = -1;
        for (int b = 0; b < blocks; b++) if (best[b] > -INFINITY && (chosen < 0 || best[b] > best[chosen])) chosen = b;
        if (chosen < 0) break;
        for (int i = chosen * block; i < (chosen + 1) * block && i < width; i++) keep[i] = 1;
        best[chosen] = -INFINITY;
    }
    free(best);
}
static void test_ds_index(int S, int pos_base, int ratio, int IH, int ID, int topk, int block, int topb, int order) {
    int width = (pos_base + S) / ratio, krow = ID + 3, qrow = IH * ID + 1, wrow = IH + 2, scrow = width + 3, lrow = topk + 5, base = 11;
    if (width < 1) width = 1;
    float *q = malloc((size_t)(S * qrow + 1) * 4), *k = malloc((size_t)(width * krow + 3) * 4), *w = malloc((size_t)(S * wrow + 2) * 4);
    for (int i = 0; i < S * qrow + 1; i++) q[i] = ((int)(rnd() % 17) - 8) * 0.25f;
    for (int i = 0; i < width * krow + 3; i++) k[i] = ((int)(rnd() % 17) - 8) * 0.25f;
    for (int i = 0; i < S * wrow + 2; i++) w[i] = ((int)(rnd() % 9) - 3) * 0.5f;
    for (int t = 0; t < width; t += 4) for (int i = 0; i < ID; i++) k[3 + t * krow + i] = k[3 + (t / 4) * krow + i];   /* ties */
    float wscale = 0.25f;
    float *ref = malloc((size_t)S * scrow * 4), *refm = malloc((size_t)S * scrow * 4);
    int *keep = calloc((size_t)S * scrow, 4), *lst = malloc((size_t)S * lrow * 4), *lstm = malloc((size_t)S * lrow * 4);
    for (int s = 0; s < S; s++) {
        int lens = (pos_base + s + 1) / ratio;
        float wt[128];
        for (int h = 0; h < IH; h++) { wt[h] = w[2 + s * wrow + h]; wt[h] *= wscale; }
        for (int j = 0; j < width; j++) {
            if (j >= lens) { ref[s * scrow + j] = -INFINITY; continue; }
            float total = 0;
            for (int h = 0; h < IH; h++) {
                float dot = 0; for (int i = 0; i < ID; i++) dot += q[1 + s * qrow + h * ID + i] * k[3 + j * krow + i];
                if (dot > 0.f) total += dot * wt[h];
            }
            ref[s * scrow + j] = total;
        }
        ds_cand_ref(ref + s * scrow, width, lens, block, topb, keep + s * scrow);
        for (int j = 0; j < width; j++) refm[s * scrow + j] = keep[s * scrow + j] ? ref[s * scrow + j] : -INFINITY;
        ds_topk_ref(ref + s * scrow, width, topk, base, order, lst + s * lrow + 2);
        ds_topk_ref(refm + s * scrow, width, topk, base, order, lstm + s * lrow + 2);
    }
    VkcBuf *qb = up(q, (size_t)S * qrow + 1), *kb = up(k, (size_t)width * krow + 3), *wb = up(w, (size_t)S * wrow + 2);
    VkcBuf *scb = vkc_buf((size_t)S * scrow * 4, VKC_DEV), *mb = vkc_buf((size_t)S * scrow * 4, VKC_DEV);
    VkcBuf *lb = vkc_buf((size_t)S * lrow * 4, VKC_DEV), *lb2 = vkc_buf((size_t)S * lrow * 4, VKC_DEV), *scb2 = vkc_buf((size_t)S * scrow * 4, VKC_DEV);
    VkcDsScore sp = {S, pos_base, ratio, IH, ID, width, 1, qrow, 2, wrow, 3, krow, 0, scrow, wscale, 0, 0};
    VkcDsCand cp = {S, pos_base, ratio, width, block, topb, scrow, scrow};
    VkcDsTopk tp = {S, width, topk, scrow, 2, lrow, base, order};
    VkcDsScore sm = sp; sm.mask_row = scrow;
    vkc_begin();
    int ok = vkc_dsv4_score(qb, wb, kb, NULL, scb, &sp) && vkc_dsv4_cand(scb, mb, &cp) && vkc_dsv4_topk(scb, lb, &tp);
    if (S > 1)   /* the masked scores one row at a time, at offsets (a speculative step's rows read different keys) */
        for (int s = 0; s < S && ok; s++) {
            VkcDsScore r1 = sm; r1.S = 1; r1.pos_base = pos_base + s; r1.q_off = 1 + s * qrow; r1.w_off = 2 + s * wrow;
            r1.sc_off = s * scrow; r1.mask_off = s * scrow;
            ok = vkc_dsv4_score(qb, wb, kb, mb, scb2, &r1);
        }
    else ok = ok && vkc_dsv4_score(qb, wb, kb, mb, scb2, &sm);
    ok = ok && vkc_dsv4_topk(scb2, lb2, &tp);
    vkc_submit(1);
    float *gs = down(scb, 0, (size_t)S * scrow);
    int *gm = (int *)down(mb, 0, (size_t)S * scrow), *gl = (int *)down(lb, 0, (size_t)S * lrow), *gl2 = (int *)down(lb2, 0, (size_t)S * lrow);
    int same_sc = 1, same_m = 1, same_l = 1, same_l2 = 1;
    for (int s = 0; s < S; s++) {
        for (int j = 0; j < width; j++) {
            same_sc &= gs[s * scrow + j] == ref[s * scrow + j];
            same_m &= gm[s * scrow + j] == keep[s * scrow + j];
        }
        for (int t = 0; t < topk; t++) { same_l &= gl[s * lrow + 2 + t] == lst[s * lrow + 2 + t]; same_l2 &= gl2[s * lrow + 2 + t] == lstm[s * lrow + 2 + t]; }
    }
    CHECK(ok && same_sc && same_m && same_l && same_l2, "dsv4 index S %d pos %d ratio %d IH %d ID %d topk %d block %d/%d order %d: "
          "scores %d mask %d list %d masked list %d", S, pos_base, ratio, IH, ID, topk, block, topb, order, same_sc, same_m, same_l, same_l2);
    vkc_free(qb); vkc_free(kb); vkc_free(wb); vkc_free(scb); vkc_free(mb); vkc_free(lb); vkc_free(lb2); vkc_free(scb2);
    free(q); free(k); free(w); free(ref); free(refm); free(keep); free(lst); free(lstm); free(gs); free(gm); free(gl); free(gl2);
}

/* deepseek_v41.c's engram_run gate, in place on the streams */
static void test_ds_engram(int S, int H, int D) {
    int kvrow = (H + 1) * D + 2, xrow = H * D + 3;
    float *kv = fvec((size_t)S * kvrow + 1, 1.f), *x = fvec((size_t)S * xrow + 4, 1.f), *prm = fvec((size_t)2 * H * D + 6, 1.f);
    float *ref = malloc(((size_t)S * xrow + 4) * 4), eps = 1e-6f;
    memcpy(ref, x, ((size_t)S * xrow + 4) * 4);
    for (int s = 0; s < S; s++) {
        const float *k0 = kv + 1 + (size_t)s * kvrow, *value = k0 + (size_t)H * D;
        for (int c = 0; c < H; c++) {
            const float *key = k0 + (size_t)c * D, *qw = prm + 6 + (size_t)c * D, *kw = prm + 6 + (size_t)H * D + (size_t)c * D;
            float *st = ref + 4 + (size_t)s * xrow + (size_t)c * D;
            double ss = 0, ks = 0, dot = 0;
            for (int i = 0; i < D; i++) { ss += (double)st[i] * st[i]; ks += (double)key[i] * key[i]; dot += (double)st[i] * qw[i] * kw[i] * key[i]; }
            float rstd = (1.0f / sqrtf((float)(ss / D) + eps)) * (1.0f / sqrtf((float)(ks / D) + eps));
            float scaled = (float)dot * rstd / sqrtf((float)D);
            float mag = fabsf(scaled); if (mag < 1e-6f) mag = 1e-6f;
            float sr = sqrtf(mag); if (scaled < 0) sr = -sr;
            float g = sigm(sr);
            for (int i = 0; i < D; i++) st[i] += g * value[i];
        }
    }
    VkcBuf *kb = up(kv, (size_t)S * kvrow + 1), *xb = up(x, (size_t)S * xrow + 4), *pb = up(prm, (size_t)2 * H * D + 6);
    VkcDsEngram p = {S, H, D, 1, kvrow, 4, xrow, 6, 6 + H * D, eps};
    vkc_begin(); int ok = vkc_dsv4_engram(kb, pb, xb, &p); vkc_submit(1);
    float *got = down(xb, 0, (size_t)S * xrow + 4);
    double e = relerr(got, ref, (size_t)S * xrow + 4, 1e-3);
    CHECK(ok && e < 2e-6, "dsv4 engram S %d H %d D %d: rel err %.2e", S, H, D, e);
    vkc_free(kb); vkc_free(xb); vkc_free(pb);
    free(kv); free(x); free(prm); free(ref); free(got);
}

/* ---- DeepSeek V4's roundings (chain_dsv4.comp modes 7, 8) ------------------------------
 * References transcribed from deepseek_v4.c (COLI_V4_UNIT_NATIVE_QUANT, the MATH unit's
 * coli_v4_swiglu): the device must give their bits. */
static float ds_e4m3_decode(uint8_t v) {
    int sign = v >> 7, exponent = (v >> 3) & 15, mantissa = v & 7;
    if (exponent == 15 && mantissa == 7) return NAN;
    float n = !exponent ? ldexpf((float)mantissa, -9) : ldexpf(1.0f + (float)mantissa / 8.0f, exponent - 7);
    return sign ? -n : n;
}
static uint8_t ds_e4m3_encode(float value) {
    if (isnan(value)) return 0x7f;
    int negative = signbit(value) != 0;
    float magnitude = fabsf(value);
    if (!magnitude) return negative ? 0x80 : 0;
    if (magnitude >= 448.0f) return (uint8_t)((negative ? 0x80 : 0) | 0x7e);
    uint8_t best;
    if (magnitude < 0.015625f) {
        float scaled = magnitude * 512.0f;
        uint8_t rounded = (uint8_t)scaled;
        float fraction = scaled - rounded;
        if (fraction > 0.5f || (fraction == 0.5f && (rounded & 1))) rounded++;
        best = rounded;
    } else {
        uint32_t bits; memcpy(&bits, &magnitude, 4);
        int exponent = (int)((bits >> 23) & 0xff) - 127;
        uint32_t significand = 0x800000u | (bits & 0x7fffffu), rounded = significand >> 20, remainder = significand & 0xfffffu;
        if (remainder > 0x80000u || (remainder == 0x80000u && (rounded & 1u))) rounded++;
        if (rounded == 16u) { rounded = 8u; exponent++; }
        best = (uint8_t)((exponent + 7) * 8 + (int)rounded - 8);
    }
    return (uint8_t)(best | (negative ? 0x80 : 0));
}
static int ds_ceil_log2(float value) { int e; float f = frexpf(value, &e); return f == 0.5f ? e - 1 : e; }
static void ds_fp8_qdq(float *out, const float *in, size_t n, size_t block) {
    for (size_t base = 0; base < n; base += block) {
        size_t count = n - base < block ? n - base : block;
        float maximum = 0.0f;
        for (size_t i = 0; i < count; i++) maximum = fmaxf(maximum, fabsf(in[base + i]));
        maximum = fmaxf(maximum, 1e-4f);
        int e = ds_ceil_log2(maximum / 448.0f);
        if (e < -127) e = -127;
        if (e > 127) e = 127;
        float scale = ldexpf(1.0f, e);
        for (size_t i = 0; i < count; i++)
            out[base + i] = ds_e4m3_decode(ds_e4m3_encode(fmaxf(-448.0f, fminf(448.0f, in[base + i] / scale)))) * scale;
    }
}
static float ds_e2m1_decode(int n) {
    static const float v[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f, 0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
    return v[n & 15];
}
static void ds_fp4_qdq(float *out, const float *in, size_t n, size_t block) {
    for (size_t base = 0; base < n; base += block) {
        size_t count = n - base < block ? n - base : block;
        float maximum = 0.0f;
        for (size_t i = 0; i < count; i++) maximum = fmaxf(maximum, fabsf(in[base + i]));
        maximum = fmaxf(maximum, 6.0f * ldexpf(1.0f, -126));
        int e = ds_ceil_log2(maximum / 6.0f);
        if (e < -127) e = -127;
        if (e > 127) e = 127;
        float scale = ldexpf(1.0f, e);
        for (size_t i = 0; i < count; i++) {
            float value = fmaxf(-6.0f, fminf(6.0f, in[base + i] / scale));
            int best = 0; float distance = fabsf(value - ds_e2m1_decode(0));
            for (int code = 1; code < 16; code++) {
                float c = fabsf(value - ds_e2m1_decode(code));
                if (c < distance) { distance = c; best = code; }
            }
            out[base + i] = ds_e2m1_decode(best) * scale;
        }
    }
}
static void ds_hadamard(float *v, size_t n) {
    for (size_t w = 1; w < n; w *= 2)
        for (size_t base = 0; base < n; base += 2 * w)
            for (size_t i = 0; i < w; i++) { float l = v[base + i], r = v[base + w + i]; v[base + i] = l + r; v[base + w + i] = l - r; }
    float scale = 1.0f / sqrtf((float)n);
    for (size_t i = 0; i < n; i++) v[i] = bf16r(v[i] * scale);
}
static float ds_sigmoid(float v) { if (v >= 0.0f) { float d = expf(-v); return 1.0f / (1.0f + d); } float g = expf(v); return g / (1.0f + g); }

/* A block's values: random magnitudes over 40 binades, and blocks built on purpose:
 * all zero, the amax on the scale's boundary (1.75 * 2^E for E4M3, 1.5 * 2^E for E2M1)
 * and one step above it, exact ties of the target grid (E4M3 mantissa midpoints and
 * subnormal midpoints; E2M1 midpoints), signed zeros, an amax under E4M3's floor. (The
 * bf16 rounding after E4M3 or E2M1 is the CPU's step but changes nothing: both grids
 * are bf16 values.) */
static void ds_fill(float *v, int n, int kind, int pattern) {
    float s = ldexpf(1.0f, (int)(rnd() % 30) - 15);
    for (int i = 0; i < n; i++) v[i] = frnd() * ldexpf(1.0f, (int)(rnd() % 40) - 20);
    if (pattern == 1) for (int i = 0; i < n; i++) v[i] = 0.0f;
    if (pattern == 2 || pattern == 3) {
        float top = (kind == VKC_DS_E2M1 ? 1.5f : 1.75f) * 4.0f * s;
        if (pattern == 3) { uint32_t u; memcpy(&u, &top, 4); u++; memcpy(&top, &u, 4); }
        v[0] = top;
    }
    if (pattern == 4) {
        float amax = (kind == VKC_DS_E2M1 ? 6.0f : 448.0f) * s;
        static const float ties2[7] = {0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f};
        v[0] = amax;
        for (int i = 1; i < n; i++) {
            float t;
            if (kind == VKC_DS_E2M1) t = ties2[rnd() % 7];
            else if (rnd() % 3 == 0) t = (float)(2 * (int)(rnd() % 8) + 1) * ldexpf(1.0f, -10);   /* subnormal midpoints */
            else t = (1.0f + (float)(2 * (int)(rnd() % 8) + 1) / 16.0f) * ldexpf(1.0f, (int)(rnd() % 14) - 6);
            v[i] = (rnd() & 1 ? -t : t) * s;
        }
        v[n / 2] = -0.0f; if (n > 3) v[3] = 0.0f;
    }
    if (pattern == 5) for (int i = 0; i < n; i++) v[i] = frnd() * ldexpf(1.0f, -20 - (int)(rnd() % 4));   /* under the floor */
}
/* kind, nseg segments of len (per_row a row, at strides), block, flags; in place or not */
static void test_ds_round(int kind, int nseg, int per_row, int len, int block, int flags, int inplace) {
    int rows = (nseg + per_row - 1) / per_row, xseg = len + 3, xrow = per_row * xseg + 5, yseg = len + 1, yrow = per_row * yseg + 2;
    size_t xn = (size_t)rows * xrow + 7, yn = inplace ? xn : (size_t)rows * yrow + 4;
    int xo = 7, yo = inplace ? 7 : 4, ys = inplace ? xseg : yseg, yr = inplace ? xrow : yrow;
    float *x = fvec(xn, 1.f), *ref = malloc(xn > yn ? xn * 4 : yn * 4), *seg = malloc((size_t)len * 4);
    int bl = kind == VKC_DS_E4M3 || kind == VKC_DS_E2M1 ? block : len, nb = (len + bl - 1) / bl;
    for (int g = 0; g < nseg; g++) {
        float *xs = x + xo + (g / per_row) * xrow + (g % per_row) * xseg;
        for (int b = 0; b < nb; b++) ds_fill(xs + b * bl, len - b * bl < bl ? len - b * bl : bl, kind, (g * nb + b) % 6);
    }
    if (inplace) memcpy(ref, x, xn * 4);
    else for (size_t i = 0; i < yn; i++) ref[i] = -7.0f;     /* what the op must not touch */
    for (int g = 0; g < nseg; g++) {
        const float *xs = x + xo + (g / per_row) * xrow + (g % per_row) * xseg;
        float *ys2 = ref + yo + (g / per_row) * yr + (g % per_row) * ys;
        memcpy(seg, xs, (size_t)len * 4);
        if (kind == VKC_DS_BF16) for (int i = 0; i < len; i++) seg[i] = bf16r(seg[i]);
        else if (kind == VKC_DS_E4M3) ds_fp8_qdq(seg, xs, (size_t)len, (size_t)block);
        else if (kind == VKC_DS_E2M1) ds_fp4_qdq(seg, xs, (size_t)len, (size_t)block);
        else ds_hadamard(seg, (size_t)len);
        if ((flags & 1) && (kind == VKC_DS_E4M3 || kind == VKC_DS_E2M1)) for (int i = 0; i < len; i++) seg[i] = bf16r(seg[i]);
        memcpy(ys2, seg, (size_t)len * 4);
    }
    VkcBuf *xb = up(x, xn), *yb = inplace ? xb : NULL;
    if (!inplace) { float *init = malloc(yn * 4); for (size_t i = 0; i < yn; i++) init[i] = -7.0f; yb = up(init, yn); free(init); }
    VkcDsRound p = {kind, nseg, per_row, len, block, flags, xo, xrow, xseg, yo, yr, ys, 1.0f / sqrtf((float)len)};
    vkc_begin(); int ok = vkc_dsv4_round(xb, yb, &p); vkc_submit(1);
    float *got = down(yb, 0, yn);
    size_t diff = 0, first = 0;
    for (size_t i = 0; i < yn; i++) if (memcmp(&got[i], &ref[i], 4)) { if (!diff) first = i; diff++; }
    CHECK(ok && !diff, "dsv4 round kind %d nseg %d len %d block %d flags %d inplace %d: %zu of %zu values differ (first at %zu: %a, want %a)",
          kind, nseg, len, block, flags, inplace, diff, yn, first, (double)got[first], (double)ref[first]);
    vkc_free(xb); if (!inplace) vkc_free(yb);
    free(x); free(ref); free(seg); free(got);
}
/* The SwiGLU: within one bf16 step of the CPU (the device's exp is not the CPU's expf),
 * and how many values are the CPU's bits */
static void test_ds_swiglu(int n, float lim) {
    float *a = fvec((size_t)n + 3, 14.f), *b = fvec((size_t)n + 5, 14.f), *ref = malloc((size_t)n * 4);
    for (int i = 0; i < n; i++) {
        float g = bf16r(a[3 + i]), u = bf16r(b[5 + i]);
        if (lim > 0.0f) { g = fminf(g, lim); u = fmaxf(-lim, fminf(u, lim)); }
        ref[i] = bf16r(g * ds_sigmoid(g) * u);
    }
    VkcBuf *ab = up(a, (size_t)n + 3), *bb = up(b, (size_t)n + 5), *yb = vkc_buf((size_t)(n + 2) * 4, VKC_DEV);
    VkcDsSwiglu p = {n, 3, 5, 2, lim};
    vkc_begin(); int ok = vkc_dsv4_swiglu(ab, bb, yb, &p); vkc_submit(1);
    float *got = down(yb, 2, (size_t)n);
    int same = 0, far = 0;
    for (int i = 0; i < n; i++) {
        uint32_t u, v; memcpy(&u, &got[i], 4); memcpy(&v, &ref[i], 4);
        same += u == v;
        long d = (long)(u >> 16) - (long)(v >> 16);
        if ((u & 0xffff) || d > 1 || d < -1) far++;
    }
    CHECK(ok && !far, "dsv4 swiglu n %d lim %g: %d values further than one bf16 step (or not bf16)", n, lim, far);
    printf("  dsv4 swiglu n %d lim %g: %d of %d values the CPU's bits, the rest one bf16 step away\n", n, lim, same, n);
    vkc_free(ab); vkc_free(bb); vkc_free(yb);
    free(a); free(b); free(ref); free(got);
}

/* ---- frames: many ops, frames in flight, ordering ---------------------------------------- */
static void test_frames(void) {
    int n = 1000;
    float *a = fvec(n, 1.f), *acc = calloc(n, sizeof *acc);
    VkcBuf *x = vkc_buf((size_t)n * 4, VKC_DEV), *y = vkc_buf((size_t)n * 4, VKC_DEV);
    int ok = 1;
    for (int f = 0; f < 11; f++) {        /* more frames than the ring: reuse while others are in flight */
        ok &= vkc_begin();
        ok &= vkc_write(y, 0, a, (size_t)n * 4);
        for (int k = 0; k < 3; k++) {
            VkcEw p = {VKC_EW_ADD, n, n, 1, 0, 1, 0, 0, 0, 0, 0, 1.f};
            ok &= vkc_ew(x, x, y, NULL, NULL, &p);
            for (int i = 0; i < n; i++) acc[i] += a[i];
        }
        ok &= vkc_submit(f == 10);
        for (int i = 0; i < n; i++) a[i] = a[i] * 0.5f + 0.25f;
    }
    float *got = down(x, 0, n);
    double e = relerr(got, acc, n, 1e-3);
    VkcStats st; vkc_stats(&st);
    CHECK(ok && e < 1e-5, "frames: err %.2e (%llu frames, %llu barriers)", e, st.frames, st.barriers);
    vkc_free(x); vkc_free(y); free(a); free(acc); free(got);
}

/* ---- the KV cache split between the device and the host (chain_kvs, vk_kvsplit.h) --
 * A host cache of every position, a device holding `rows` of them (vkc_kv_plan under
 * COLI_VK_KV_DEVICE_ROWS), decode or prefill steps one after another as an engine runs
 * them: the window placed, the rows below the step uploaded, the step's own rows stored
 * from a scratch, then the device's part (F2) and the host's part (from the queries F1
 * brought down) merged in F3, or one frame when the device holds every visible position.
 * Each step's output against a double-precision reference of the full attention. */
/* row s's positions: its list (count >= 0) or the causal range */
static int kvs_positions(const int *sel, int sel_row, int s, int lo, int pos, int *out) {
    if (sel && sel[(size_t)s * sel_row] >= 0) {
        int n = 0;
        for (int j = 0; j < sel[(size_t)s * sel_row]; j++) { int t = sel[(size_t)s * sel_row + 1 + j]; if (t >= 0) out[n++] = t; }
        return n;
    }
    int n = 0;
    for (int t = lo; t <= pos; t++) out[n++] = t;
    return n;
}

/* A query value fixed by its position and index (not by the order the steps draw them):
 * the same row gets the same query however the positions are cut into steps. */
static float hval(unsigned a, unsigned b) {
    unsigned x = a * 2654435761u ^ (b + 0x9e3779b9u) * 2246822519u;
    x ^= x >> 15; x *= 2654435761u; x ^= x >> 13;
    return (float)((int)(x % 2001u) - 1000) / 1000.0f;
}
/* each step's output rows, kept by position (kvs_keep_row floats a position) */
static float *kvs_keep; static size_t kvs_keep_row;
static int kvs_start; /* seeded canonical prefix; zero for the sequential tests */
static float *hvec(size_t n, unsigned seed, float scale) { float *v = malloc(n * sizeof *v); for (size_t i = 0; i < n; i++) v[i] = hval(seed, (unsigned)i) * scale; return v; }
/* The host's cache as an engine's holds it: a step's rows arrive after the step (the
 * rows not there yet are NaN, which a host part reading them would carry into the
 * result). copy rows [t0, t1) of a [seg][T][len] or [T][len] array. */
static float *kvs_host(size_t n) { float *v = malloc(n * sizeof *v); for (size_t i = 0; i < n; i++) v[i] = NAN; return v; }
static void kvs_arrive(float *dst, const float *src, int nseg, size_t seg, int len, int t0, int t1) {
    for (int h = 0; h < nseg; h++) memcpy(dst + h * seg + (size_t)t0 * len, src + h * seg + (size_t)t0 * len, (size_t)(t1 - t0) * len * 4);
}
static void kvs_keep_rows(const float *o, int pb, int S) { if (kvs_keep) memcpy(kvs_keep + (size_t)pb * kvs_keep_row, o, (size_t)S * kvs_keep_row * 4); }

/* GQA: steps of `step` rows from position 0 to T, the device holding `rows` positions. */
/* COLI_VK_KV_COLD=device (test_kvs_cold, or a family run with it): every split step's host part must run on
 * the device, but where pins are on (they need the CPU's lists) */
static int kvs_cold_wanted(void) { const char *c = getenv("COLI_VK_KV_COLD"); return c && !strcmp(c, "device"); }
static void test_kvs_gqa(int T, int step, int H, int KVH, int hd, int vd, int B, int rows, int win, int sink, int kv_pm,
                         int gated, int lists, int selects) {
    char rv[32]; snprintf(rv, sizeof rv, "%d", rows); setenv("COLI_VK_KV_DEVICE_ROWS", rv, 1);
    char bv[32]; snprintf(bv, sizeof bv, "%d", B); setenv("COLI_VK_KV_BLOCK", bv, 1);
    VkcKvSplit ks; memset(&ks, 0, sizeof ks);
    int r = vkc_kv_plan(&ks, "test", 1, (size_t)KVH * (hd + vd) * 4, T, step, selects, 0);
    unsetenv("COLI_VK_KV_DEVICE_ROWS"); unsetenv("COLI_VK_KV_BLOCK");
    CHECK(r == 2 && ks.on, "kvs gqa: the plan did not split (%d)", r);
    if (r != 2) { vkc_kv_free(&ks); return; }
    if (step > ks.chunk) step = ks.chunk;   /* as the engines clamp a step to the split's */
    float *Kh = hvec((size_t)T * KVH * hd, 1u << 30, 1.f), *Vh = hvec((size_t)T * KVH * vd, 1u << 29, 1.f), *snk = hvec(H + 3, 1u << 28, 2.f);
    float *Kx = kvs_host((size_t)T * KVH * hd), *Vx = kvs_host((size_t)T * KVH * vd);
    int qrow = H * 2 * hd, sel_row = lists ? 1 + T : 0;
    float scale = 1.f / sqrtf((float)hd);
    VkcBuf *kc = vkc_buf((size_t)KVH * ks.rows * hd * 4, VKC_DEV), *vc = vkc_buf((size_t)KVH * ks.rows * vd * 4, VKC_DEV);
    VkcBuf *qb = vkc_buf((size_t)step * qrow * 4, VKC_DEV), *nk = vkc_buf((size_t)step * KVH * hd * 4, VKC_DEV);
    VkcBuf *nv = vkc_buf((size_t)step * KVH * vd * 4, VKC_DEV), *ob = vkc_buf((size_t)step * H * vd * 4, VKC_DEV);
    VkcBuf *sb = up(snk, H + 3), *lb = vkc_buf((size_t)step * (sel_row ? sel_row : 1) * 4, VKC_DEV);
    int ok = vkc_kv_parts(&ks, (size_t)step * H * (vd + 2));
    double worst = 0; int splits = 0, fins = 0;
    int *posl = malloc((size_t)(T + 1) * sizeof *posl), *sel = calloc((size_t)step * (sel_row ? sel_row : 1), sizeof *sel);
    float *ref = malloc((size_t)step * H * vd * sizeof *ref), *nkh = malloc((size_t)step * KVH * hd * 4), *nvh = malloc((size_t)step * KVH * vd * 4);
    if (kvs_start) {
        kvs_arrive(Kx, Kh, kv_pm ? 1 : KVH, (size_t)T * hd, kv_pm ? KVH * hd : hd, 0, kvs_start);
        kvs_arrive(Vx, Vh, kv_pm ? 1 : KVH, (size_t)T * vd, kv_pm ? KVH * vd : vd, 0, kvs_start);
    }
    for (int pb = kvs_start; pb < T && ok; pb += step) {
        int S = T - pb < step ? T - pb : step;
        float *q = malloc((size_t)S * qrow * 4);
        for (int s = 0; s < S; s++) for (int i = 0; i < qrow; i++) q[(size_t)s * qrow + i] = hval(pb + s, i);
        for (int s = 0; s < S; s++) for (int h = 0; h < KVH; h++) {   /* the step's rows, as its projections leave them */
            const float *ks_ = kv_pm ? Kh + ((size_t)(pb + s) * KVH + h) * hd : Kh + ((size_t)h * T + pb + s) * hd;
            const float *vs_ = kv_pm ? Vh + ((size_t)(pb + s) * KVH + h) * vd : Vh + ((size_t)h * T + pb + s) * vd;
            memcpy(nkh + ((size_t)s * KVH + h) * hd, ks_, hd * 4); memcpy(nvh + ((size_t)s * KVH + h) * vd, vs_, vd * 4);
        }
        if (lists) for (int s = 0; s < S; s++) {   /* odd rows causal; even rows every third position and a few far ones */
            int pos = pb + s, n = 0, *rw = sel + (size_t)s * sel_row;
            if (pos % 2) { rw[0] = -1; continue; }
            for (int t = 0; t <= pos; t++) if (t % 3 == 0 || t == pos || t < 2 * B) rw[1 + n++] = t;
            rw[1 + n++] = -1;   /* a skipped entry */
            rw[0] = n;
        }
        for (int s = 0; s < S; s++) for (int h = 0; h < H; h++) {
            int kvh = h / (H / KVH), pos = pb + s, lo = win ? (pos - win + 1 > 0 ? pos - win + 1 : 0) : 0;
            int n = kvs_positions(lists ? sel : NULL, sel_row, s, lo, pos, posl);
            double mx = sink ? snk[3 + h] : -1e300, sum = 0, *sc = malloc((n + 1) * sizeof *sc);
            for (int j = 0; j < n; j++) {
                const float *kr = kv_pm ? Kh + ((size_t)posl[j] * KVH + kvh) * hd : Kh + ((size_t)kvh * T + posl[j]) * hd;
                double a = 0; for (int d = 0; d < hd; d++) a += (double)q[(size_t)s * qrow + h * 2 * hd + d] * kr[d];
                sc[j] = a * scale; if (sc[j] > mx) mx = sc[j];
            }
            if (sink) sum = exp(snk[3 + h] - mx);
            for (int j = 0; j < n; j++) { sc[j] = exp(sc[j] - mx); sum += sc[j]; }
            for (int d = 0; d < vd; d++) {
                double a = 0;
                for (int j = 0; j < n; j++) {
                    const float *vr = kv_pm ? Vh + ((size_t)posl[j] * KVH + kvh) * vd : Vh + ((size_t)kvh * T + posl[j]) * vd;
                    a += sc[j] * vr[d];
                }
                double v = sum > 0 ? a / sum : 0;
                if (gated) v *= sigm(q[(size_t)s * qrow + h * 2 * hd + hd + d]);
                ref[((size_t)s * H + h) * vd + d] = (float)v;
            }
            free(sc);
        }
        /* the engine's order: the window placed, the rows below uploaded, the new rows stored */
        VkcKvPart pk = {kv_pm ? 1 : KVH, kv_pm ? KVH * hd : hd, Kx, (size_t)T * hd, kc, 0};
        VkcKvPart pv = {kv_pm ? 1 : KVH, kv_pm ? KVH * vd : vd, Vx, (size_t)T * vd, vc, 0};
        VkcKvPart parts[2] = {pk, pv};
        vkc_kv_place(&ks, 0, pb, S);
        ok = vkc_begin() && vkc_kv_push(&ks, 0, parts, 2, pb) &&
             vkc_write(qb, 0, q, (size_t)S * qrow * 4) && vkc_write(nk, 0, nkh, (size_t)S * KVH * hd * 4) &&
             vkc_write(nv, 0, nvh, (size_t)S * KVH * vd * 4) && (!lists || vkc_write(lb, 0, sel, (size_t)S * sel_row * 4)) &&
             vkc_kv_store(&ks, 0, &pk, nk, 0, (size_t)KVH * hd, kv_pm ? 0 : (size_t)hd, pb, S) &&
             vkc_kv_store(&ks, 0, &pv, nv, 0, (size_t)KVH * vd, kv_pm ? 0 : (size_t)vd, pb, S);
        unsigned long long sp = ks.splits;
        VkcKvGqa ga = {&ks, 0, qb, kc, vc, qb, lists ? lb : NULL, sb, ob, S, H, KVH, hd, vd, pb, win, kv_pm, sink, 3,
                       qrow, 2 * hd, gated, hd, qrow, 2 * hd, H * vd, 0, sel_row, scale,
                       Kx, Vx, kv_pm ? (size_t)hd : (size_t)T * hd, kv_pm ? (size_t)KVH * hd : (size_t)hd,
                       kv_pm ? (size_t)vd : (size_t)T * vd, kv_pm ? (size_t)KVH * vd : (size_t)vd};
        ok = ok && vkc_kv_gqa(&ga);
        if (ks.splits > sp) splits++; else fins++;
        ok = ok && vkc_submit(1);
        vkc_kv_done(&ks, 0, pb + S);
        if (kv_pm) { kvs_arrive(Kx, Kh, 1, 0, KVH * hd, pb, pb + S); kvs_arrive(Vx, Vh, 1, 0, KVH * vd, pb, pb + S); }
        else { kvs_arrive(Kx, Kh, KVH, (size_t)T * hd, hd, pb, pb + S); kvs_arrive(Vx, Vh, KVH, (size_t)T * vd, vd, pb, pb + S); }
        float *o = down(ob, 0, (size_t)S * H * vd);
        kvs_keep_rows(o, pb, S);
        double e = relerr(o, ref, (size_t)S * H * vd, 1e-3);
        if (bad(o, (size_t)S * H * vd)) e = 1e9;   /* a NaN: the host part read a row not there yet */
        if (e > worst) worst = e;
        free(o); free(q);
    }
    CHECK(ok && worst < 2e-5 && splits > 0, "kvs gqa T %d step %d H %d/%d hd %d vd %d B %d rows %d win %d sink %d pm %d gate %d lists %d: "
          "err %.2e (%d split steps, %d whole)", T, step, H, KVH, hd, vd, B, rows, win, sink, kv_pm, gated, lists, worst, splits, fins);
    printf("  kvs gqa T %d step %d rows %d lists %d: rel err %.2e, %d split steps, %d whole, %llu host positions, %llu pinned%s\n",
           T, step, rows, lists, worst, splits, fins, ks.host_pos, ks.pins, ks.cold_dev ? ", the host's part on the device" : "");
    CHECK(!kvs_cold_wanted() || ks.np || (ks.cold_dev && ks.dev_cold > 0 && ks.host_pos == 0), "kvs gqa: the host's part did not run on the device");
    CHECK(!ks.np || !selects || ks.pins > 0, "kvs gqa lists: no block was pinned");
    vkc_free(kc); vkc_free(vc); vkc_free(qb); vkc_free(nk); vkc_free(nv); vkc_free(ob); vkc_free(sb); vkc_free(lb);
    vkc_kv_free(&ks);
    free(Kh); free(Vh); free(Kx); free(Vx); free(snk); free(posl); free(sel); free(ref); free(nkh); free(nvh);
}

/* MLA: the latent and the rope key per position; lists with skipped entries; kv_start. */
static void test_kvs_mla(int T, int step, int H, int K, int R, int B, int rows, int kv_start, int lists) {
    char rv[32]; snprintf(rv, sizeof rv, "%d", rows); setenv("COLI_VK_KV_DEVICE_ROWS", rv, 1);
    char bv[32]; snprintf(bv, sizeof bv, "%d", B); setenv("COLI_VK_KV_BLOCK", bv, 1);
    VkcKvSplit ks; memset(&ks, 0, sizeof ks);
    int r = vkc_kv_plan(&ks, "test", 2, (size_t)(K + R) * 4, T, step, lists, 0);
    unsetenv("COLI_VK_KV_DEVICE_ROWS"); unsetenv("COLI_VK_KV_BLOCK");
    CHECK(r == 2, "kvs mla: the plan did not split (%d)", r);
    if (r != 2) { vkc_kv_free(&ks); return; }
    if (step > ks.chunk) step = ks.chunk;
    float *Lh = hvec((size_t)T * K, 1u << 30, 1.f), *Rh = hvec((size_t)T * (R ? R : 1), 1u << 29, 1.f);
    float *Lx = kvs_host((size_t)T * K), *Rx = kvs_host((size_t)T * (R ? R : 1));
    int sel_row = lists ? 1 + T : 0, li = 1;
    float scale = 0.7f / sqrtf((float)(K + R));
    VkcBuf *lat = vkc_buf((size_t)ks.rows * K * 4, VKC_DEV), *rope = vkc_buf((size_t)ks.rows * (R ? R : 1) * 4, VKC_DEV);
    VkcBuf *qab = vkc_buf((size_t)step * H * (K + R) * 4, VKC_DEV), *nl = vkc_buf((size_t)step * (K + R) * 4, VKC_DEV);
    VkcBuf *ob = vkc_buf((size_t)step * H * K * 4, VKC_DEV), *lb = vkc_buf((size_t)step * (sel_row ? sel_row : 1) * 4, VKC_DEV);
    int ok = vkc_kv_parts(&ks, (size_t)step * H * (K + 2));
    double worst = 0; int splits = 0;
    int *posl = malloc((size_t)(T + 1) * sizeof *posl), *sel = calloc((size_t)step * (sel_row ? sel_row : 1), sizeof *sel);
    float *ref = malloc((size_t)step * H * K * sizeof *ref), *nh = malloc((size_t)step * (K + R) * 4);
    for (int pb = 0; pb < T && ok; pb += step) {
        int S = T - pb < step ? T - pb : step;
        float *q = malloc((size_t)S * H * (K + R) * 4);   /* qa [S][H][K], then qr [S][H][R] */
        for (int s = 0; s < S; s++) {
            for (int i = 0; i < H * K; i++) q[(size_t)s * H * K + i] = hval(pb + s, i);
            for (int i = 0; i < H * R; i++) q[(size_t)S * H * K + (size_t)s * H * R + i] = hval(pb + s, 100000 + i);
        }
        for (int s = 0; s < S; s++) {
            memcpy(nh + (size_t)s * K, Lh + (size_t)(pb + s) * K, K * 4);
            if (R) memcpy(nh + (size_t)S * K + (size_t)s * R, Rh + (size_t)(pb + s) * R, R * 4);
        }
        if (lists) for (int s = 0; s < S; s++) {
            int pos = pb + s, n = 0, *rw = sel + (size_t)s * sel_row;
            if (pos % 3 == 2) { rw[0] = -1; continue; }
            for (int t = 0; t <= pos; t++) if (t % 4 == 1 || t == pos || (t >= B && t < 3 * B)) rw[1 + n++] = t;
            rw[1 + n++] = -1;
            rw[0] = n;
        }
        for (int s = 0; s < S; s++) for (int h = 0; h < H; h++) {
            int n = kvs_positions(lists ? sel : NULL, sel_row, s, kv_start, pb + s, posl);
            double mx = -1e300, sum = 0, *sc = malloc((n + 1) * sizeof *sc);
            for (int j = 0; j < n; j++) {
                double a = 0;
                for (int d = 0; d < K; d++) a += (double)q[((size_t)s * H + h) * K + d] * Lh[(size_t)posl[j] * K + d];
                for (int d = 0; d < R; d++) a += (double)q[(size_t)S * H * K + ((size_t)s * H + h) * R + d] * Rh[(size_t)posl[j] * R + d];
                sc[j] = a * scale; if (sc[j] > mx) mx = sc[j];
            }
            for (int j = 0; j < n; j++) { sc[j] = exp(sc[j] - mx); sum += sc[j]; }
            for (int d = 0; d < K; d++) {
                double a = 0;
                for (int j = 0; j < n; j++) a += sc[j] * Lh[(size_t)posl[j] * K + d];
                ref[((size_t)s * H + h) * K + d] = (float)(sum > 0 ? a / sum : 0);
            }
            free(sc);
        }
        VkcKvPart pl = {1, K, Lx, 0, lat, 0}, pr = {1, R ? R : 1, Rx, 0, rope, 0};
        VkcKvPart parts[2] = {pl, pr};
        vkc_kv_place(&ks, li, pb, S);
        ok = vkc_begin() && vkc_kv_push(&ks, li, parts, R ? 2 : 1, pb) && vkc_write(qab, 0, q, (size_t)S * H * (K + R) * 4) &&
             vkc_write(nl, 0, nh, (size_t)S * (K + R) * 4) && (!lists || vkc_write(lb, 0, sel, (size_t)S * sel_row * 4)) &&
             vkc_kv_store(&ks, li, &pl, nl, 0, (size_t)K, 0, pb, S) &&
             (!R || vkc_kv_store(&ks, li, &pr, nl, (size_t)S * K, (size_t)R, 0, pb, S));
        unsigned long long sp = ks.splits;
        VkcKvMla ma = {&ks, li, qab, qab, lat, rope, lists ? lb : NULL, ob, S, H, K, R, pb, kv_start,
                       0, H * K, K, S * H * K, H * R, R, 0, H * K, K, 0, sel_row, scale, Lx, Rx, 0};
        ok = ok && vkc_kv_mla(&ma);
        if (ks.splits > sp) splits++;
        ok = ok && vkc_submit(1);
        vkc_kv_done(&ks, li, pb + S);
        kvs_arrive(Lx, Lh, 1, 0, K, pb, pb + S);
        if (R) kvs_arrive(Rx, Rh, 1, 0, R, pb, pb + S);
        float *o = down(ob, 0, (size_t)S * H * K);
        kvs_keep_rows(o, pb, S);
        double e = relerr(o, ref, (size_t)S * H * K, 1e-3);
        if (bad(o, (size_t)S * H * K)) e = 1e9;   /* a NaN: the host part read a row not there yet */
        if (e > worst) worst = e;
        free(o); free(q);
    }
    CHECK(ok && worst < 2e-5 && splits > 0, "kvs mla T %d step %d H %d K %d R %d B %d rows %d start %d lists %d: err %.2e (%d split steps)",
          T, step, H, K, R, B, rows, kv_start, lists, worst, splits);
    printf("  kvs mla T %d step %d K %d rows %d lists %d: rel err %.2e, %d split steps, %llu host positions, %llu pinned%s\n",
           T, step, K, rows, lists, worst, splits, ks.host_pos, ks.pins, ks.cold_dev ? ", the host's part on the device" : "");
    CHECK(!lists || !ks.np || ks.pins > 0, "kvs mla lists: no block was pinned");
    CHECK(!kvs_cold_wanted() || ks.np || (ks.cold_dev && ks.dev_cold > 0 && ks.host_pos == 0), "kvs mla: the host's part did not run on the device");
    vkc_free(lat); vkc_free(rope); vkc_free(qab); vkc_free(nl); vkc_free(ob); vkc_free(lb);
    vkc_kv_free(&ks);
    free(Lh); free(Rh); free(Lx); free(Rx); free(posl); free(sel); free(ref); free(nh);
}
/* Inkling's global layer over the split cache: the relative-position bias for distances
 * below ext, tau per row; steps from position 0 to T. */
static void test_kvs_rel(int T, int step, int H, int KVH, int hd, int ext, int d_rel, int B, int rows) {
    char rv[32]; snprintf(rv, sizeof rv, "%d", rows); setenv("COLI_VK_KV_DEVICE_ROWS", rv, 1);
    char bv[32]; snprintf(bv, sizeof bv, "%d", B); setenv("COLI_VK_KV_BLOCK", bv, 1);
    VkcKvSplit ks; memset(&ks, 0, sizeof ks);
    int r = vkc_kv_plan(&ks, "test", 1, (size_t)KVH * hd * 2 * 4, T, step, 0, 0);
    unsetenv("COLI_VK_KV_DEVICE_ROWS"); unsetenv("COLI_VK_KV_BLOCK");
    CHECK(r == 2, "kvs rel: the plan did not split (%d)", r);
    if (r != 2) { vkc_kv_free(&ks); return; }
    if (step > ks.chunk) step = ks.chunk;
    float *Kh = hvec((size_t)T * KVH * hd, 1u << 30, 1.f), *Vh = hvec((size_t)T * KVH * hd, 1u << 29, 1.f);
    float *relp = hvec((size_t)d_rel * ext + 3, 1u << 28, 0.5f);
    float *Kx = kvs_host((size_t)T * KVH * hd), *Vx = kvs_host((size_t)T * KVH * hd);
    float scale = 1.f / (float)hd;
    VkcBuf *kc = vkc_buf((size_t)KVH * ks.rows * hd * 4, VKC_DEV), *vc = vkc_buf((size_t)KVH * ks.rows * hd * 4, VKC_DEV);
    VkcBuf *qb = vkc_buf((size_t)step * H * hd * 4, VKC_DEV), *rb = vkc_buf((size_t)step * H * (d_rel ? d_rel : 1) * 4, VKC_DEV);
    VkcBuf *nk = vkc_buf((size_t)step * KVH * hd * 4, VKC_DEV), *nv = vkc_buf((size_t)step * KVH * hd * 4, VKC_DEV);
    VkcBuf *ob = vkc_buf((size_t)step * H * hd * 4, VKC_DEV), *tb = vkc_buf((size_t)(step + 2) * 4, VKC_UP), *pb_ = up(relp, (size_t)d_rel * ext + 3);
    int ok = vkc_kv_parts(&ks, (size_t)step * H * (hd + 2));
    double worst = 0; int splits = 0;
    float *ref = malloc((size_t)step * H * hd * 4), *nkh = malloc((size_t)step * KVH * hd * 4), *nvh = malloc((size_t)step * KVH * hd * 4);
    float *tau = malloc((size_t)(step + 2) * 4);
    for (int pb = 0; pb < T && ok; pb += step) {
        int S = T - pb < step ? T - pb : step;
        float *q = malloc((size_t)S * H * hd * 4), *rr = malloc((size_t)S * H * (d_rel ? d_rel : 1) * 4);
        for (int s = 0; s < S; s++) {
            for (int i = 0; i < H * hd; i++) q[(size_t)s * H * hd + i] = hval(pb + s, i);
            for (int i = 0; i < H * (d_rel ? d_rel : 1); i++) rr[(size_t)s * H * (d_rel ? d_rel : 1) + i] = hval(pb + s, 50000 + i);
        }
        for (int s = 0; s < S; s++) { tau[2 + s] = 1.f + 0.3f * logf(1.f + (float)(pb + s) / 7.f); }
        for (int s = 0; s < S; s++) for (int h = 0; h < KVH; h++) {
            memcpy(nkh + ((size_t)s * KVH + h) * hd, Kh + ((size_t)h * T + pb + s) * hd, hd * 4);
            memcpy(nvh + ((size_t)s * KVH + h) * hd, Vh + ((size_t)h * T + pb + s) * hd, hd * 4);
        }
        for (int s = 0; s < S; s++) for (int h = 0; h < H; h++) {
            int kvh = h / (H / KVH), pos = pb + s;
            double mx = -1e300, sum = 0, *sc = malloc((pos + 1) * sizeof *sc);
            for (int t = 0; t <= pos; t++) {
                double a = 0; for (int d = 0; d < hd; d++) a += (double)q[((size_t)s * H + h) * hd + d] * Kh[((size_t)kvh * T + t) * hd + d];
                double bias = 0; int dist = pos - t;
                if (dist < ext) for (int j = 0; j < d_rel; j++) bias += (double)rr[((size_t)s * H + h) * d_rel + j] * relp[3 + (size_t)j * ext + dist];
                sc[t] = tau[2 + s] * (a * scale + bias); if (sc[t] > mx) mx = sc[t];
            }
            for (int t = 0; t <= pos; t++) { sc[t] = exp(sc[t] - mx); sum += sc[t]; }
            for (int d = 0; d < hd; d++) {
                double a = 0; for (int t = 0; t <= pos; t++) a += sc[t] * Vh[((size_t)kvh * T + t) * hd + d];
                ref[((size_t)s * H + h) * hd + d] = (float)(a / sum);
            }
            free(sc);
        }
        VkcKvPart pk = {KVH, hd, Kx, (size_t)T * hd, kc, 0}, pv = {KVH, hd, Vx, (size_t)T * hd, vc, 0};
        VkcKvPart parts[2] = {pk, pv};
        vkc_kv_place(&ks, 0, pb, S);
        memcpy(vkc_ptr(tb), tau, (size_t)(S + 2) * 4);
        ok = vkc_begin() && vkc_kv_push(&ks, 0, parts, 2, pb) && vkc_write(qb, 0, q, (size_t)S * H * hd * 4) &&
             (!d_rel || vkc_write(rb, 0, rr, (size_t)S * H * d_rel * 4)) &&
             vkc_write(nk, 0, nkh, (size_t)S * KVH * hd * 4) && vkc_write(nv, 0, nvh, (size_t)S * KVH * hd * 4) &&
             vkc_kv_store(&ks, 0, &pk, nk, 0, (size_t)KVH * hd, (size_t)hd, pb, S) &&
             vkc_kv_store(&ks, 0, &pv, nv, 0, (size_t)KVH * hd, (size_t)hd, pb, S);
        unsigned long long sp = ks.splits;
        VkcKvRel a = {&ks, 0, qb, kc, vc, rb, tb, pb_, ob, S, H, KVH, hd, pb, ext, d_rel, H * hd, H * d_rel, 3, 2, H * hd, scale,
                      Kx, Vx, (size_t)T * hd, (size_t)hd, (size_t)T * hd, (size_t)hd, tau + 2, relp + 3, 0};
        ok = ok && vkc_kv_rel(&a) && vkc_submit(1);
        if (ks.splits > sp) splits++;
        vkc_kv_done(&ks, 0, pb + S);
        kvs_arrive(Kx, Kh, KVH, (size_t)T * hd, hd, pb, pb + S); kvs_arrive(Vx, Vh, KVH, (size_t)T * hd, hd, pb, pb + S);
        float *o = down(ob, 0, (size_t)S * H * hd);
        kvs_keep_rows(o, pb, S);
        double e = relerr(o, ref, (size_t)S * H * hd, 1e-3);
        if (bad(o, (size_t)S * H * hd)) e = 1e9;   /* a NaN: the host part read a row not there yet */
        if (e > worst) worst = e;
        free(o); free(q); free(rr);
    }
    CHECK(ok && worst < 2e-5 && splits > 0, "kvs rel T %d step %d H %d/%d hd %d ext %d d_rel %d B %d rows %d: err %.2e (%d split steps)",
          T, step, H, KVH, hd, ext, d_rel, B, rows, worst, splits);
    printf("  kvs rel T %d step %d rows %d ext %d: rel err %.2e, %d split steps, %llu host positions%s\n", T, step, rows, ext, worst,
           splits, ks.host_pos, ks.cold_dev ? ", the host's part on the device" : "");
    CHECK(!kvs_cold_wanted() || ks.np || (ks.cold_dev && ks.dev_cold > 0 && ks.host_pos == 0), "kvs rel: the host's part did not run on the device");
    vkc_free(kc); vkc_free(vc); vkc_free(qb); vkc_free(rb); vkc_free(nk); vkc_free(nv); vkc_free(ob); vkc_free(tb); vkc_free(pb_);
    vkc_kv_free(&ks);
    free(Kh); free(Vh); free(Kx); free(Vx); free(relp); free(ref); free(nkh); free(nvh); free(tau);
}

/* DeepSeek's sparse attention over a window ring and compressed rows split between the
 * device and the host: against the scalar kernel (V4's rounding: its reference), and with
 * every compressed row on the device, bit for bit against vkc_dsv4_attn. */
static void test_kvs_ds(int S, int H, int hd, int nwin, int ncmp, int cnt, int v4, int B, int rows) {
    int qrow = H * hd + 3, lrow = cnt + 2, orow = H * hd + 3;
    float *q = fvec((size_t)S * qrow, 1.f), *win = fvec((size_t)nwin * hd, 1.f), *cmp = fvec((size_t)ncmp * hd, 1.f);
    float *sink = fvec((size_t)H + 2, 2.f), *ref = calloc((size_t)S * orow, 4), *sc = malloc((size_t)cnt * 4);
    int *list = malloc((size_t)S * lrow * 4);
    float *kvall = malloc((size_t)(nwin + ncmp) * hd * 4);
    memcpy(kvall, win, (size_t)nwin * hd * 4); memcpy(kvall + (size_t)nwin * hd, cmp, (size_t)ncmp * hd * 4);
    for (int s = 0; s < S; s++)
        for (int j = 0; j < lrow; j++) {
            int r = (int)(rnd() % 6);
            list[s * lrow + j] = r == 0 ? -1 - (int)(rnd() % 7) : r < 2 ? (int)(rnd() % (unsigned)nwin) : nwin + (int)(rnd() % (unsigned)ncmp);
        }
    float scale = 1.f / sqrtf((float)hd);
    for (int s = 0; s < S; s++) for (int h = 0; h < H; h++) {
        const float *qq = q + (size_t)s * qrow + 1 + h * hd;
        float *o = ref + (size_t)s * orow + 2 + h * hd;
        const int *idx = list + s * lrow + 1;
        if (!v4) { coli_sparse_attend_scalar(o, qq, kvall, idx, cnt, hd, sink[1 + h], scale, sc); continue; }
        float best = -1e30f;
        for (int j = 0; j < cnt; j++) {
            if (idx[j] < 0) { sc[j] = -INFINITY; continue; }
            float d = 0; for (int i = 0; i < hd; i++) d += qq[i] * kvall[(size_t)idx[j] * hd + i];
            sc[j] = d * scale; if (sc[j] > best) best = sc[j];
        }
        float den = expf(sink[1 + h] - best);
        for (int i = 0; i < hd; i++) o[i] = 0;
        for (int j = 0; j < cnt; j++) {
            if (idx[j] < 0) continue;
            float w = expf(sc[j] - best); den += w; w = bf16r(w);
            for (int i = 0; i < hd; i++) o[i] += w * kvall[(size_t)idx[j] * hd + i];
        }
        for (int i = 0; i < hd; i++) o[i] = bf16r(o[i] / den);
    }
    VkcBuf *qb = up(q, (size_t)S * qrow), *wb = up(win, (size_t)nwin * hd), *cb = up(cmp, (size_t)ncmp * hd);
    VkcBuf *sb = up(sink, (size_t)H + 2), *lb = vkc_buf((size_t)S * lrow * 4, VKC_DEV);
    VkcBuf *ow = vkc_buf((size_t)S * orow * 4, VKC_DEV), *os = vkc_buf((size_t)S * orow * 4, VKC_DEV);
    vkc_begin(); vkc_write(lb, 0, list, (size_t)S * lrow * 4); vkc_submit(1);
    VkcDsAttn p = {S, H, hd, cnt, 1, lrow, nwin, 0, 0, 1, qrow, 2, orow, 1, v4, scale};
    int ok = vkc_begin() && vkc_dsv4_attn(qb, wb, cb, lb, sb, ow, &p) && vkc_submit(1);
    float *whole = down(ow, 0, (size_t)S * orow);
    /* twice: a device holding `rows` compressed rows (a host part), then all of them (fin) */
    for (int pass = 0; pass < 2; pass++) {
        int rws = pass ? B * ((ncmp + B - 1) / B) : rows;
        char rv[32]; snprintf(rv, sizeof rv, "%d", rws); setenv("COLI_VK_KV_DEVICE_ROWS", rv, 1);
        char bv[32]; snprintf(bv, sizeof bv, "%d", B); setenv("COLI_VK_KV_BLOCK", bv, 1);
        VkcKvSplit ks; memset(&ks, 0, sizeof ks);
        int r = vkc_kv_plan(&ks, "test", 1, (size_t)hd * 4, ncmp + 2 * B, 1, 1, 0);
        unsetenv("COLI_VK_KV_DEVICE_ROWS"); unsetenv("COLI_VK_KV_BLOCK");
        CHECK(r == 2, "kvs ds: the plan did not split (%d)", r);
        if (r != 2) { vkc_kv_free(&ks); continue; }
        if (pass) { ks.nw = ks.ns; ks.np = 0; }        /* every block in the window */
        VkcBuf *dc = vkc_buf((size_t)ks.rows * hd * 4, VKC_DEV);
        VkcKvPart pc = {1, hd, cmp, 0, dc, 0};
        vkc_kv_place(&ks, 0, ncmp - 1, 1);
        unsigned long long sp = ks.splits;
        ok = vkc_kv_parts(&ks, (size_t)S * H * (hd + 2)) && vkc_begin() && vkc_kv_push(&ks, 0, &pc, 1, ncmp);
        /* The current chunk's resident rows may not have reached canonical RAM
         * yet. Poison them there: staging must read only nonresident rows. */
        float *hostcmp = malloc((size_t)ncmp * hd * sizeof(float));
        memcpy(hostcmp, cmp, (size_t)ncmp * hd * sizeof(float));
        for (int j = 0; j < ncmp; j++) if (ks.t[0].bt[j / B] >= 0)
            for (int d = 0; d < hd; d++) hostcmp[(size_t)j * hd + d] = NAN;
        VkcKvDs a = {&ks, 0, qb, wb, dc, sb, lb, os, S, H, hd, cnt, 1, lrow, nwin, 0, 1, qrow, 1, v4, 2, orow, ncmp, scale, hostcmp};
        ok = ok && vkc_kv_ds(&a) && vkc_submit(1);
        float *got = down(os, 0, (size_t)S * orow);
        double e = 0, ew = 0;
        for (int s = 0; s < S; s++) {
            double r1 = relerr(got + (size_t)s * orow + 2, ref + (size_t)s * orow + 2, (size_t)H * hd, 1e-3); if (r1 > e) e = r1;
            double r2 = relerr(got + (size_t)s * orow + 2, whole + (size_t)s * orow + 2, (size_t)H * hd, 1e-3); if (r2 > ew) ew = r2;
        }
        int same = 1;
        for (int s = 0; s < S; s++) same &= !memcmp(got + (size_t)s * orow + 2, whole + (size_t)s * orow + 2, (size_t)H * hd * 4);
        double tol = 2e-5;
        if (pass) CHECK(ok && same && ks.splits == sp, "kvs ds fin S %d H %d hd %d cnt %d v4 %d: not vkc_dsv4_attn's bits (err %.2e)", S, H, hd, cnt, v4, ew);
        else CHECK(ok && same && e < tol && ks.splits > sp && ks.staged_rows > 0, "kvs ds S %d H %d hd %d cnt %d v4 %d rows %d: err %.2e (%llu host parts)", S, H, hd, cnt, v4, rows, e, ks.splits - sp);
        printf("  kvs ds S %d hd %d cnt %d v4 %d rows %d: rel err %.2e against the reference, %.2e against the whole op%s\n",
               S, hd, cnt, v4, rws, e, ew, pass ? (same ? " (its bits)" : " (NOT its bits)") : "");
        vkc_free(dc); free(got); free(hostcmp);
        vkc_kv_free(&ks);
    }
    vkc_free(qb); vkc_free(wb); vkc_free(cb); vkc_free(sb); vkc_free(lb); vkc_free(ow); vkc_free(os);
    free(q); free(win); free(cmp); free(sink); free(ref); free(sc); free(list); free(kvall); free(whole);
}
/* The same positions in steps of different sizes give the same bits (the split's
 * partition follows each row's own position): GQA, MLA and Inkling's, causal, and a
 * list engine's with no pins. */
static void test_kvs_steps(void) {
    int T = 70, H = 4, hd = 32;
    float *a = malloc((size_t)T * H * 64 * 4), *b = malloc((size_t)T * H * 64 * 4);
    kvs_keep_row = (size_t)H * hd;
    kvs_keep = a; test_kvs_gqa(T, 1, H, 2, hd, hd, 4, 32, 0, 1, 0, 1, 0, 0);
    kvs_keep = b; test_kvs_gqa(T, 3, H, 2, hd, hd, 4, 32, 0, 1, 0, 1, 0, 0);
    CHECK(!memcmp(a, b, (size_t)T * kvs_keep_row * 4), "kvs: GQA rows in steps of 1 and 3 differ");
    setenv("COLI_VK_KV_SLICE", "20000", 1);   /* the device's rows in slices of a row or two, a submission each */
    kvs_keep = b; test_kvs_gqa(T, 3, H, 2, hd, hd, 4, 32, 0, 1, 0, 1, 0, 0);
    unsetenv("COLI_VK_KV_SLICE");
    CHECK(!memcmp(a, b, (size_t)T * kvs_keep_row * 4), "kvs: GQA rows in slices differ");
    unsetenv("COLI_VK_KV_PIN");   /* default: a deterministic partition even for lists */
    kvs_keep = a; test_kvs_gqa(T, 1, H, 2, hd, hd, 4, 32, 0, 0, 0, 1, 1, 1);
    kvs_keep = b; test_kvs_gqa(T, 4, H, 2, hd, hd, 4, 32, 0, 0, 0, 1, 1, 1);
    unsetenv("COLI_VK_KV_PIN");
    CHECK(!memcmp(a, b, (size_t)T * kvs_keep_row * 4), "kvs: listed GQA rows in steps of 1 and 4 (no pins) differ");
    kvs_keep_row = (size_t)2 * 64;
    kvs_keep = a; test_kvs_mla(T, 1, 2, 64, 8, 4, 32, 0, 0);
    kvs_keep = b; test_kvs_mla(T, 2, 2, 64, 8, 4, 32, 0, 0);
    CHECK(!memcmp(a, b, (size_t)T * kvs_keep_row * 4), "kvs: MLA rows in steps of 1 and 2 differ");
    kvs_keep_row = (size_t)H * hd;
    kvs_keep = a; test_kvs_rel(T, 1, H, 2, hd, 16, 4, 4, 32);
    kvs_keep = b; test_kvs_rel(T, 4, H, 2, hd, 16, 4, 4, 32);
    CHECK(!memcmp(a, b, (size_t)T * kvs_keep_row * 4), "kvs: Inkling's rows in steps of 1 and 4 differ");
    kvs_keep = NULL;
    free(a); free(b);
    printf("  kvs: the same bits for every position in steps of 1, 2, 3 and 4, and in slices of rows\n");
}
static void test_kvs(void) {
    setenv("COLI_VK_KV_PIN", "1", 1); /* explicitly exercise read-based pins */
    /* T step H KVH hd vd B rows win sink pm gate lists selects */
    test_kvs_gqa(40, 1, 4, 2, 32, 32, 4, 12, 0, 0, 0, 1, 0, 0);       /* decode over a window of three blocks, a gate */
    test_kvs_gqa(41, 3, 4, 2, 32, 32, 4, 8, 0, 0, 0, 1, 0, 0);        /* the smallest device: steps across a block */
    test_kvs_gqa(70, 7, 4, 2, 48, 32, 8, 32, 0, 1, 1, 0, 0, 0);       /* prefill chunks, MiMo's position-major rows, a sink */
    test_kvs_gqa(300, 1, 8, 2, 64, 64, 16, 64, 0, 0, 0, 1, 0, 0);     /* past a tile of the device's rows */
    test_kvs_gqa(600, 50, 4, 4, 16, 16, 16, 128, 0, 0, 0, 0, 0, 0);   /* host tiles of many positions, prefill rows */
    test_kvs_gqa(60, 3, 4, 2, 32, 32, 4, 24, 0, 0, 0, 1, 1, 1);       /* selection lists, pinned blocks */
    test_kvs_gqa(50, 2, 4, 2, 256, 256, 4, 16, 0, 0, 0, 1, 1, 1);     /* the largest head */
    test_kvs_gqa(60, 2, 4, 2, 32, 24, 4, 16, 20, 1, 1, 0, 0, 0);      /* a window wider than the device's rows */
    test_kvs_gqa(48, 12, 4, 2, 32, 32, 4, 40, 0, 0, 0, 1, 0, 0);      /* prompt chunks on the device whole, then split */
    /* T step H K R B rows kv_start lists */
    test_kvs_mla(40, 1, 4, 32, 8, 4, 12, 0, 0);
    test_kvs_mla(90, 5, 2, 64, 16, 8, 40, 0, 1);
    test_kvs_mla(60, 4, 3, 48, 0, 4, 16, 0, 0);          /* NoPE */
    test_kvs_mla(70, 2, 2, 512, 64, 8, 32, 5, 0);        /* GLM-5.2's latent, a nonzero start */
    test_kvs_mla(40, 3, 1, 1024, 8, 4, 16, 0, 1);        /* the largest latent */
    /* T step H KVH hd ext d_rel B rows */
    test_kvs_rel(60, 1, 4, 2, 32, 8, 4, 4, 16);          /* decode: the bias reaches back 8, the device holds 16 */
    test_kvs_rel(80, 5, 4, 2, 16, 40, 16, 8, 32);        /* the bias reaching into the host's rows */
    test_kvs_rel(50, 3, 2, 1, 128, 16, 64, 4, 24);       /* the largest bank, Inkling's head */
    /* S H hd nwin ncmp cnt v4 B rows */
    test_kvs_ds(1, 4, 64, 8, 40, 30, 0, 4, 12);
    test_kvs_ds(3, 2, 512, 16, 200, 300, 0, 16, 64);
    test_kvs_ds(2, 3, 128, 8, 90, 2900, 0, 8, 32);
    test_kvs_ds(3, 4, 64, 8, 60, 40, 1, 4, 16);          /* DeepSeek V4's roundings */
    test_kvs_ds(2, 2, 1024, 4, 30, 20, 1, 4, 12);        /* the largest head */
    unsetenv("COLI_VK_KV_PIN");
    test_kvs_steps();
    /* Cross 64K with only 512 device positions, starting from a canonical RAM
     * prefix. This also crosses host tiles and changes OpenMP task grouping.
     * Rows not yet produced are NaN, as in the shorter sequential tests. */
    int T = 65543, first = 65531, H = 2, hd = 16;
    float *a = calloc((size_t)T * H * hd, sizeof(float));
    float *b = calloc((size_t)T * H * hd, sizeof(float));
    kvs_start = first; kvs_keep_row = (size_t)H * hd;
    kvs_keep = a; test_kvs_gqa(T, 1, H, 1, hd, hd, 64, 512, 0, 0, 0, 0, 0, 0);
    kvs_keep = b; test_kvs_gqa(T, 7, H, 1, hd, hd, 64, 512, 0, 0, 0, 0, 0, 0);
    CHECK(!memcmp(a + (size_t)first * H * hd, b + (size_t)first * H * hd,
                  (size_t)(T - first) * H * hd * sizeof(float)), "kvs: rows crossing 64K differ with chunk size");
    printf("  kvs: compared all %d rows crossing 64K bit for bit across chunk sizes\n", T - first);
    kvs_start = 0; kvs_keep = NULL; free(a); free(b);
}

/* COLI_VK_KV_COLD=device: the host's part on the device, from the shadow of the host's
 * rows. Every case without pins against the same double-precision reference (sinks,
 * gates, MiMo's position-major rows, windows, lists, MLA with and without a rope key and
 * a nonzero start, Inkling's bias reaching into the host's rows), each split step's host
 * part on the device and none on the CPU, and the same bits for every position however
 * the forward is cut (steps of 1 to 4, slices of rows, crossing 64K). */
static void test_kvs_cold(void) {
    if (!coli_vk_import_alignment()) { printf("  kvs cold: no host memory the device reads in place here, skipped\n"); return; }
    setenv("COLI_VK_KV_COLD", "device", 1);
    /* T step H KVH hd vd B rows win sink pm gate lists selects */
    test_kvs_gqa(40, 1, 4, 2, 32, 32, 4, 12, 0, 0, 0, 1, 0, 0);
    test_kvs_gqa(41, 3, 4, 2, 32, 32, 4, 8, 0, 0, 0, 1, 0, 0);
    test_kvs_gqa(70, 7, 4, 2, 48, 32, 8, 32, 0, 1, 1, 0, 0, 0);
    test_kvs_gqa(300, 1, 8, 2, 64, 64, 16, 64, 0, 0, 0, 1, 0, 0);
    test_kvs_gqa(600, 50, 4, 4, 16, 16, 16, 128, 0, 0, 0, 0, 0, 0);
    test_kvs_gqa(60, 3, 4, 2, 32, 32, 4, 24, 0, 0, 0, 1, 1, 1);       /* lists, no pins */
    test_kvs_gqa(50, 2, 4, 2, 256, 256, 4, 16, 0, 0, 0, 1, 1, 1);
    test_kvs_gqa(60, 2, 4, 2, 32, 24, 4, 16, 20, 1, 1, 0, 0, 0);
    test_kvs_gqa(48, 12, 4, 2, 32, 32, 4, 40, 0, 0, 0, 1, 0, 0);
    /* T step H K R B rows kv_start lists */
    test_kvs_mla(40, 1, 4, 32, 8, 4, 12, 0, 0);
    test_kvs_mla(90, 5, 2, 64, 16, 8, 40, 0, 1);
    test_kvs_mla(60, 4, 3, 48, 0, 4, 16, 0, 0);
    test_kvs_mla(70, 2, 2, 512, 64, 8, 32, 5, 0);
    test_kvs_mla(40, 3, 1, 1024, 8, 4, 16, 0, 1);
    /* T step H KVH hd ext d_rel B rows */
    test_kvs_rel(60, 1, 4, 2, 32, 8, 4, 4, 16);
    test_kvs_rel(80, 5, 4, 2, 16, 40, 16, 8, 32);
    test_kvs_rel(50, 3, 2, 1, 128, 16, 64, 4, 24);
    test_kvs_steps();
    int T = 65543, first = 65531, H = 2, hd = 16;
    float *a = calloc((size_t)T * H * hd, sizeof(float));
    float *b = calloc((size_t)T * H * hd, sizeof(float));
    kvs_start = first; kvs_keep_row = (size_t)H * hd;
    kvs_keep = a; test_kvs_gqa(T, 1, H, 1, hd, hd, 64, 512, 0, 0, 0, 0, 0, 0);
    kvs_keep = b; test_kvs_gqa(T, 7, H, 1, hd, hd, 64, 512, 0, 0, 0, 0, 0, 0);
    CHECK(!memcmp(a + (size_t)first * H * hd, b + (size_t)first * H * hd,
                  (size_t)(T - first) * H * hd * sizeof(float)), "kvs cold: rows crossing 64K differ with chunk size");
    kvs_start = 0; kvs_keep = NULL; free(a); free(b);
    unsetenv("COLI_VK_KV_COLD");
    printf("  kvs cold: every case with the host's part on the device, the same bits however the forward is cut\n");
}

/* COLI_VK_CHAIN_BENCH=1: decode-shaped GEMVs back to back in one frame (the chain's
 * situation: no host gap between matrices), per weight format, in GB/s of weights. */
#include <time.h>
static double bnow(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec / 1e9; }
static void bench_gemv(void) {
    int shapes[3][2] = {{2560, 10240}, {6144, 2560}, {2560, 640}};
    int fmts[3] = {1, 4, 11};
    for (int f = 0; f < 3; f++) for (int k = 0; k < 3; k++) {
        int I = shapes[k][0], O = shapes[k][1], fmt = fmts[f], gs = fmt == 4 ? 64 : 0, nm = 24;
        size_t rb = fmt == 1 ? (size_t)I : fmt == 4 ? (size_t)I / 2 : (size_t)I * 2;
        ColiVkTensor **t = calloc(nm, sizeof *t);
        uint8_t *w = malloc(rb * O); float *sc = malloc((size_t)O * (I / 64 + 1) * sizeof(float));
        for (size_t i = 0; i < rb * O; i++) w[i] = (uint8_t)rnd();
        for (size_t i = 0; i < (size_t)O * (I / 64 + 1); i++) sc[i] = 0.01f;
        for (int j = 0; j < nm; j++) coli_vk_tensor_ensure(&t[j], w, sc, fmt, I, O, gs);
        float *x = fvec(I, 1.f);
        VkcBuf *xb = up(x, I), *yb = vkc_buf((size_t)O * 4 * nm, VKC_DEV);
        double best = 1e9;
        for (int r = 0; r < 4; r++) {
            vkc_begin();
            for (int j = 0; j < nm; j++) vkc_matmul(t[j], xb, 0, yb, (size_t)j * O, 1);
            double t0 = bnow(); vkc_submit(1); double dt = bnow() - t0;
            if (dt < best) best = dt;
        }
        printf("bench fmt %2d [%5d x %5d] x %d: %.3f ms per matrix, %.1f GB/s of weights\n", fmt, O, I, nm,
               best * 1e3 / nm, (double)rb * O * nm / best / 1e9);
        for (int j = 0; j < nm; j++) coli_vk_tensor_free(t[j]);
        vkc_free(xb); vkc_free(yb); free(t); free(w); free(sc); free(x);
    }
}

/* ---- a second device (vkc_device(1), COLI_VK_DEV2) -------------------------------------
 * The chain's context on COLI_VK_DEV2's device: the ops against their references there,
 * the matmul's bits equal to device 0's, and the boundary held (a buffer of one device
 * bound or copied on the other fails; vkc_free takes it back on its own device). */
static void test_dev2(const char *spv) {
    const char *d2 = getenv("COLI_VK_DEV2");
    if (!coli_vk_init_dev2(spv, strcmp(d2, "auto") ? atoi(d2) : -1)) { CHECK(0, "dev2: the second device did not open"); return; }
    vkc_device(1);
    if (!vkc_init()) { CHECK(0, "dev2: the chain's pipelines did not come up there"); vkc_device(0); return; }
    vkc_device(0);
    int fmts[4] = {1, 4, 10, 11};
    for (int k = 0; k < 4; k++) {
        int shapes[3][3] = {{1, 192, 70}, {40, 256, 160}, {2, 3072, 40}};
        for (int j = 0; j < 3; j++) {
            int S = shapes[j][0], I = shapes[j][1], O = shapes[j][2];
            float *y0 = malloc((size_t)S * O * sizeof(float)), *y1 = malloc((size_t)S * O * sizeof(float));
            unsigned seed = rng;
            g_mm_out = y0; test_matmul(fmts[k], S, I, O, 0, 0);
            rng = seed; vkc_device(1); g_test_dev = 1;
            g_mm_out = y1; test_matmul(fmts[k], S, I, O, 0, 0);
            g_test_dev = 0; vkc_device(0); g_mm_out = NULL;
            CHECK(!memcmp(y0, y1, (size_t)S * O * sizeof(float)), "dev2: matmul fmt %d S %d: device 1's bits differ from device 0's", fmts[k], S);
            free(y0); free(y1);
        }
    }
    vkc_device(1);
    test_norm(VKC_NORM_ADD1, 0); test_norm(0, 0);
    test_rope();
    test_attn(1, 140, 32, 0); test_attn(5, 200, 64, 0);
    setenv("COLI_VK_ATTN_BLOCK", "16", 1); test_attn_hk(40, 7, 16, 2, 256, 0); unsetenv("COLI_VK_ATTN_BLOCK");
    test_dnrec(8, 8, 8, 4, 0); test_dnrec(32, 100, 6, 3, 1);
    if (vkc_kvs_ready()) test_kvs_steps();
    /* the boundary: device 0's buffer on device 1 */
    vkc_device(0);
    float v[64]; for (int i = 0; i < 64; i++) v[i] = (float)i;
    VkcBuf *a0 = up(v, 64);
    vkc_device(1);
    VkcBuf *a1 = up(v, 64), *b1 = vkc_buf(64 * 4, VKC_DEV);
    vkc_begin();
    int cross = vkc_copy(b1, 0, a0, 0, 64);
    vkc_submit(1);
    vkc_begin();
    int same = vkc_copy(b1, 0, a1, 0, 64);
    vkc_submit(1);
    CHECK(!cross && same, "dev2: a copy across devices %s, one on device 1 %s", cross ? "ran" : "refused", same ? "ran" : "failed");
    VkcNorm nm = {1, 64, 1, 0, 64, 64, 0, 64, 64, 0, 0, VKC_NORM_NOW, 1e-6f, 1.f};
    vkc_begin();
    int bound = vkc_norm(a0, a1, b1, &nm);
    vkc_submit(1);
    CHECK(!bound, "dev2: an op bound device 0's buffer on device 1");
    vkc_free(a1); vkc_free(b1);
    vkc_device(0);
    vkc_free(a0);
    vkc_device(1);
    vkc_free(NULL);
    VkcStats st; vkc_stats(&st);
    printf("dev2: chain on the second device, %llu frames, %llu ops, %llu matmuls\n", st.frames, st.ops, st.matmuls);
    vkc_shutdown();
    vkc_device(0);
}

/* A speculative verify's rows through the decode GEMV: each of S rows (2..4, one
 * workgroup) must carry the bits a one-row call gives that row. */
static void test_gemv_rows(int fmt, int S, int I, int O) {
    float *x = fvec((size_t)S * I, 1.f);
    void *codes = NULL; float *sc = NULL; int gs = 0;
    if (fmt == 1) {
        int8_t *q = malloc((size_t)O * I); sc = malloc(O * sizeof *sc);
        for (size_t i = 0; i < (size_t)O * I; i++) q[i] = (int8_t)((int)(rnd() % 255) - 127);
        for (int o = 0; o < O; o++) sc[o] = 0.01f + (rnd() % 100) / 5000.f;
        codes = q;
    } else {
        gs = 64; int ng = I / 64;
        uint8_t *q = malloc((size_t)O * (I / 2)); sc = malloc((size_t)O * ng * sizeof *sc);
        for (size_t i = 0; i < (size_t)O * (I / 2); i++) q[i] = (uint8_t)rnd();
        for (int i = 0; i < O * ng; i++) sc[i] = 0.02f + (rnd() % 100) / 3000.f;
        codes = q;
    }
    ColiVkTensor *t = NULL;
    if (!coli_vk_tensor_ensure(&t, codes, sc, fmt, I, O, gs)) { CHECK(0, "gemv rows fmt %d: upload", fmt); return; }
    VkcBuf *xb = vkc_buf((size_t)S * I * 4, VKC_DEV), *ya = vkc_buf((size_t)S * O * 4, VKC_DOWN), *y1 = vkc_buf((size_t)S * O * 4, VKC_DOWN);
    vkc_gemm_rows(0);   /* every row on the GEMV, as a verify's */
    vkc_begin(); vkc_write(xb, 0, x, (size_t)S * I * 4);
    int ok = vkc_matmul(t, xb, 0, ya, 0, S);
    for (int s = 0; s < S; s++) ok = ok && vkc_matmul(t, xb, (size_t)s * I, y1, (size_t)s * O, 1);
    vkc_submit(1);
    vkc_gemm_rows(-1);
    int same = ok && !memcmp(vkc_ptr(ya), vkc_ptr(y1), (size_t)S * O * 4);
    CHECK(same, "gemv rows fmt %d S %d I %d O %d: a row's bits depend on the rows beside it", fmt, S, I, O);
    vkc_free(xb); vkc_free(ya); vkc_free(y1); coli_vk_tensor_free(t);
    free(x); free(codes); free(sc);
}
int main(int argc, char **argv) {
    const char *spv = argc > 1 ? argv[1] : "shaders/qmatmul.spv";
    if (!coli_vk_init(spv)) { printf("FAIL: no Vulkan device (shaders %s)\n", spv); return 1; }
    if (!vkc_init()) { printf("FAIL: the chain's pipelines did not come up\n"); return 1; }
    if (getenv("COLI_VK_CHAIN_BENCH")) { bench_gemv(); vkc_shutdown(); coli_vk_shutdown(); return 0; }
    int fmts[4] = {1, 4, 10, 11};
    for (int k = 0; k < 4; k++) {
        test_matmul(fmts[k], 1, 192, 70, 0, 0);
        test_matmul(fmts[k], 3, 128, 96, 64, 128);       /* GEMV rows at bindable offsets */
        test_matmul(fmts[k], 40, 256, 160, 0, 0);        /* the tiled GEMM */
        test_matmul(fmts[k], 2, 64, 33, 3, 5);           /* offsets the device cannot bind */
        test_matmul(fmts[k], 1, 100, 33, 0, 0);          /* rows that are not whole 16-byte steps */
        test_matmul(fmts[k], 2, 3072, 40, 0, 0);         /* a long row (chain_gemv's staging) */
    }
    /* int8 and int4 prompt GEMMs whose shapes chain_gemm.comp takes (I % 64, O % 128), on
     * a device with cooperative matrices at subgroup size 64: full and partial row tiles,
     * bindable offsets; its x is rounded to f16, hence the bound */
    {
        VkcStats s0; vkc_stats(&s0);
        g_mm_tol = 2e-3;
        for (int k = 0; k < 2; k++) {
            test_matmul(fmts[k], 64, 256, 256, 0, 0);
            test_matmul(fmts[k], 70, 512, 384, 64, 128);
            test_matmul(fmts[k], 130, 192, 128, 0, 0);
        }
        g_mm_tol = 2e-4;
        VkcStats s1; vkc_stats(&s1);
        printf("  prompt GEMMs on chain_gemm: %llu\n", s1.tile_gemms - s0.tile_gemms);
    }
    printf("matmul done\n");
    test_norm(VKC_NORM_ADD1, 0); test_norm(VKC_NORM_ADD1, 1); test_norm(0, 0); test_norm(VKC_NORM_NOW, 1); test_norm(VKC_NORM_L2 | VKC_NORM_NOW, 0);
    printf("norm done\n");
    for (int k = 0; k < 2; k++) {   /* a verify's rows: the decode step's bits each */
        int f = k ? 4 : 1;
        test_gemv_rows(f, 2, 2048, 512); test_gemv_rows(f, 3, 4096, 256); test_gemv_rows(f, 4, 2048, 300);
    }
    test_rope();
    test_chunk_rows();
    test_attn(1, 0, 16, 0); test_attn(1, 140, 32, 0); test_attn(5, 200, 64, 0); test_attn(6, 9, 256, 1); test_attn(130, 3, 24, 0);
    /* prompt chunks on the matrix units where the device has them (chain_attn_flash, from
     * 16 rows): Qwen3.6's heads (hd 256, 8 query heads per kv head) from position 0 and
     * after earlier rows, a chunk over several 64-key blocks with a partial last token
     * group, hd 64 with GQA 2, hd 128 with GQA 16; then cut in slices */
    {
        VkcStats s0; vkc_stats(&s0);
        test_attn_hk(16, 0, 16, 2, 256, 0); test_attn_hk(37, 200, 16, 2, 256, 0); test_attn_hk(130, 3, 4, 2, 64, 0);
        test_attn_hk(21, 70, 16, 1, 128, 0);
        setenv("COLI_VK_ATTN_SLICE", "60000", 1);
        test_attn_hk(40, 7, 16, 2, 256, 0); test_attn_hk(130, 3, 4, 2, 64, 0);
        unsetenv("COLI_VK_ATTN_SLICE");
        VkcStats s1; vkc_stats(&s1);
        printf("  attention on the matrix units: %llu calls\n", s1.attn_flash - s0.attn_flash);
    }
    setenv("COLI_VK_CHAIN_FLASH", "0", 1);   /* the blocked and plain cases below: their own shaders */
    /* MiMo: a prompt's rows through a window (from position 0, and past it), a decode row
     * over a ring as large as the window, prefill rows over a larger ring, full attention
     * with V's own head dim, a window wider than a tile, the head-major layout with a
     * window, the release's head dims (192 / 128) over a ring of 128 */
    test_attn_win(5, 0, 4, 2, 48, 32, 8, 0, 1, 1);  test_attn_win(5, 20, 4, 2, 48, 32, 8, 0, 1, 1);
    test_attn_win(1, 30, 4, 4, 48, 32, 8, 8, 1, 1); test_attn_win(1, 3, 4, 4, 48, 32, 8, 8, 1, 1);
    test_attn_win(3, 40, 4, 2, 48, 32, 8, 10, 0, 1); test_attn_win(4, 6, 4, 2, 64, 48, 0, 0, 1, 1);
    test_attn_win(2, 300, 4, 2, 16, 16, 150, 0, 1, 1); test_attn_win(130, 3, 4, 2, 24, 40, 16, 0, 0, 0);
    test_attn_win(1, 500, 8, 4, 192, 128, 128, 128, 1, 1);
    /* prompt chunks through the blocked attention (chain_attnb.comp, from 16 rows): Qwen3.6's
     * and Qwen3.8's head groups (8 and 12 a KV head), one head a KV head, lists, the window,
     * the ring, position-major rows and the sink; then the same cases on chain_attn */
    for (int pass = 0; pass < 2; pass++) {
        setenv("COLI_VK_ATTN_BLOCK", pass ? "0" : "16", 1);
        test_attn_hk(40, 7, 16, 2, 256, 0); test_attn_hk(33, 0, 24, 2, 64, 0); test_attn_hk(64, 100, 4, 4, 32, 0);
        test_attn_hk(37, 5, 16, 2, 128, 1); test_attn_hk(16, 200, 24, 2, 24, 1);
        test_attn_win(40, 0, 4, 2, 48, 32, 8, 0, 1, 1); test_attn_win(33, 20, 8, 2, 64, 48, 0, 0, 1, 0);
        test_attn_win(50, 100, 4, 4, 48, 32, 16, 40, 0, 1); test_attn_win(20, 3, 8, 4, 192, 128, 128, 0, 1, 1);
    }
    unsetenv("COLI_VK_ATTN_BLOCK");
    { VkcStats st; vkc_stats(&st); CHECK(st.attn_blocked >= 9, "attn: %llu calls took the blocked attention", st.attn_blocked);
      printf("  blocked attention: %llu calls\n", st.attn_blocked); }
    /* a long attention cut over several submissions (COLI_VK_ATTN_SLICE): the same cases,
     * the blocked attention and chain_attn, slices of a few rows each */
    {
        VkcStats s0; vkc_stats(&s0);
        setenv("COLI_VK_ATTN_SLICE", "60000", 1);
        for (int pass = 0; pass < 2; pass++) {
            setenv("COLI_VK_ATTN_BLOCK", pass ? "0" : "16", 1);
            test_attn_hk(40, 7, 16, 2, 256, 0); test_attn_hk(37, 5, 16, 2, 128, 1); test_attn_hk(64, 100, 4, 4, 32, 0);
            test_attn_win(50, 100, 4, 4, 48, 32, 16, 40, 0, 1); test_attn_win(33, 20, 8, 2, 64, 48, 0, 0, 1, 0);
        }
        unsetenv("COLI_VK_ATTN_BLOCK");
        VkcStats s1; vkc_stats(&s1);
        CHECK(s1.attn_slices > s0.attn_slices + 10, "attn: %llu slices", s1.attn_slices - s0.attn_slices);
        printf("  attention in slices: %llu extra submissions\n", s1.attn_slices - s0.attn_slices);
        unsetenv("COLI_VK_ATTN_SLICE");
    }
    unsetenv("COLI_VK_CHAIN_FLASH");
    printf("attn done\n");
    test_dnconv(0); test_dnconv(1);
    test_dnrec(8, 8, 8, 4, 0); test_dnrec(4, 4, 4, 2, 1); test_dnrec(128, 128, 4, 2, 0); test_dnrec(32, 100, 6, 3, 1);
    printf("deltanet done\n");
    test_ew();
    test_qsa();
    test_ple();
    test_frames();
    if (!vkc_mla_ready()) { fails++; printf("FAIL: the MLA shaders did not load\n"); }
    else {
        int hf[9] = {1, 2, 4, 5, 7, 10, 11, 12, 13};
        for (int k = 0; k < 9; k++) if (hf[k] != 7) { test_hgemv(hf[k], 0); test_hgemv(hf[k], 1); }
        test_mla_rows();
        printf("mla ops done\n");
        /* H Q R V K q_lora D S pos kv_start style split list gate fq fkv fo */
        MlaCase cases[] = {
            {4, 24, 8, 32, 32, 64, 128, 1, 12, 0, 1, 0, 0, 0, 10, 10, 10},     /* colibri's tiny, decode */
            {4, 24, 8, 32, 32, 64, 128, 5, 7, 0, 1, 0, 0, 0, 10, 10, 10},      /* prefill rows after earlier ones */
            {4, 24, 8, 32, 32, 64, 128, 1, 70, 0, 1, 0, 1, 0, 10, 10, 10},     /* a selection list */
            {3, 16, 16, 20, 48, 0, 96, 3, 9, 0, 0, 1, 0, 1, 1, 1, 1},          /* no q latent, rotate-half, split, gate, int8 */
            {2, 64, 0, 64, 128, 64, 128, 4, 66, 0, 0, 1, 1, 0, 4, 4, 4},       /* NoPE, int4-g64, lists */
            {2, 32, 64, 16, 64, 128, 128, 1, 300, 5, 1, 0, 0, 0, 2, 2, 11},    /* a long context from kv_start, int4 rows, bf16 */
            {2, 192, 64, 256, 512, 128, 256, 2, 70, 0, 1, 0, 0, 0, 4, 4, 4},   /* GLM-5.2's heads */
            {1, 16, 8, 16, 1024, 32, 64, 1, 20, 0, 1, 0, 0, 1, 10, 5, 10},     /* the largest latent, int3-g64 */
        };
        for (size_t k = 0; k < sizeof cases / sizeof *cases; k++) test_mla(cases[k]);
        {   /* the core over a long chunk in slices of rows (COLI_VK_ATTN_SLICE) */
            VkcStats s0; vkc_stats(&s0);
            setenv("COLI_VK_ATTN_SLICE", "200000", 1);
            test_mla(cases[1]); test_mla(cases[2]); test_mla(cases[4]); test_mla(cases[6]);
            MlaCase big = {4, 24, 8, 32, 32, 64, 128, 40, 30, 0, 1, 0, 0, 0, 10, 10, 10};   /* 40 rows after 30 */
            test_mla(big);
            unsetenv("COLI_VK_ATTN_SLICE");
            VkcStats s1; vkc_stats(&s1);
            CHECK(s1.attn_slices > s0.attn_slices, "mla: no slice");
            printf("  mla core in slices: %llu extra submissions\n", s1.attn_slices - s0.attn_slices);
        }
        printf("mla done\n");
        test_dsa(1, 30, 2, 16, 8, 0, 0); test_dsa(4, 40, 4, 32, 16, 0, 0); test_dsa(3, 5, 2, 16, 64, 0, 0);
        test_dsa(3, 5, 2, 16, 64, 1, 0); test_dsa(2, 700, 3, 64, 100, 0, 0); test_dsa(1, 9, 64, 64, 4, 0, 0);
        test_dsa(2, 900, 2, 16, 50, 0, 1);
        printf("dsa done\n");
        test_kpool(3, 20, 4, 16, 2, 8, 1); test_kpool(2, 9, 1, 16, 4, 8, 1); test_kpool(4, 300, 4, 64, 4, 64, 1);
        test_kpool(1, 2, 4, 16, 2, 4, 0); test_kpool(2, 30, 16, 16, 2, 6, 1);
        printf("kpool done\n");
    }
    if (!vkc_kda_ready() || !vkc_mhc_ready()) { fails++; printf("FAIL: the KDA or mHC shaders did not load\n"); }
    else {
        test_kda(3, 16, 4, 1.f); test_kda(2, 32, 4, 1.f); test_kda(4, 128, 3, 1.f); test_kda(1, 64, 2, 1.f);
        test_kda(2, 16, 4, 0.002f);
        printf("kda done\n");
        test_mhc(2, 64, 3); test_mhc(4, 96, 20); test_mhc(3, 40, 1);
        printf("mhc done\n");
        test_kda_k3(2, 16, 4, 1.f); test_kda_k3(3, 32, 4, 1.f); test_kda_k3(2, 128, 4, 1.f); test_kda_k3(1, 64, 2, 1.f);
        test_kda_k3(2, 16, 4, 0.002f);
        printf("kda k3 done\n");
    }
    if (!vkc_ares_ready()) { fails++; printf("FAIL: the AttnRes shader did not load\n"); }
    else {
        test_ares(1, 128, 0); test_ares(3, 128, 1); test_ares(5, 300, 4); test_ares(2, 7168, 8); test_ares(4, 96, 15);
        test_situ(1.f, 1.f); test_situ(4.f, 25.f);
        printf("ares done\n");
    }
    if (vkc_sconv_ready() && vkc_relattn_ready()) {
        test_inkling();
        VkcStats s0; vkc_stats(&s0);
        setenv("COLI_VK_ATTN_SLICE", "30000", 1);   /* the relative attention over slices of rows */
        test_relattn(20, 30, 16, 16, 16, 4, 16); test_relattn(130, 3, 0, 200, 24, 4, 200); test_relattn(40, 10, 0, 64, 32, 16, 64);
        unsetenv("COLI_VK_ATTN_SLICE");
        VkcStats s1; vkc_stats(&s1);
        CHECK(s1.attn_slices > s0.attn_slices, "relattn: no slice");
        printf("  relative attention in slices: %llu extra submissions\n", s1.attn_slices - s0.attn_slices);
        printf("inkling done\n");
    }
    else CHECK(0, "inkling's ops: chain_sconv.spv or chain_relattn.spv did not load");
    if (!vkc_dsv4_ready()) { fails++; printf("FAIL: the DeepSeek V4 shader did not load\n"); }
    else {
        test_ds_attn(1, 4, 64, 8, 5, 13, 0); test_ds_attn(3, 2, 512, 128, 40, 300, 0); test_ds_attn(2, 3, 100, 6, 9, 20, 0);
        test_ds_attn(2, 2, 64, 8, 12, 2900, 0); test_ds_attn(3, 4, 64, 8, 5, 13, 1);
        test_ds_rope(0); test_ds_rope(1);
        test_ds_compress(2, 0, 0); test_ds_compress(1, 0, 0); test_ds_compress(4, 0, 0); test_ds_compress(4, 1, 1);
        test_ds_compress(3, 0, 1);
        test_ds_index(1, 30, 1, 4, 32, 4, 2, 2, 0); test_ds_index(5, 40, 2, 4, 32, 4, 2, 2, 0); test_ds_index(3, 0, 1, 2, 16, 8, 2, 3, 0);
        test_ds_index(2, 900, 1, 3, 64, 100, 4, 8, 0); test_ds_index(4, 300, 4, 16, 64, 16, 1, 3, 1); test_ds_index(3, 1, 2, 2, 16, 4, 2, 1, 0);
        test_ds_engram(1, 4, 128); test_ds_engram(3, 2, 300);
        test_ds_index(3, 37, 4, 64, 128, 16, 2, 2, 1); test_ds_index(2, 60, 4, 96, 64, 8, 2, 2, 0);   /* DeepSeek V4's 64 x 128, past the staging */
        printf("dsv4 done\n");
        int in;
        for (in = 0; in < 2; in++) {
            test_ds_round(VKC_DS_BF16, 6, 2, 50, 0, 0, in); test_ds_round(VKC_DS_BF16, 300, 1, 1000, 0, 0, in);
            test_ds_round(VKC_DS_E4M3, 6, 2, 300, 128, 0, in); test_ds_round(VKC_DS_E4M3, 9, 3, 448, 64, 1, in);
            test_ds_round(VKC_DS_E4M3, 40, 1, 4096, 128, 0, in); test_ds_round(VKC_DS_E4M3, 5, 1, 70, 256, 1, in);
            test_ds_round(VKC_DS_E2M1, 6, 2, 128, 32, 1, in); test_ds_round(VKC_DS_E2M1, 10, 5, 100, 32, 0, in);
            test_ds_round(VKC_DS_HADAMARD, 6, 2, 128, 0, 0, in); test_ds_round(VKC_DS_HADAMARD, 3, 1, 4096, 0, 0, in);
            test_ds_round(VKC_DS_HADAMARD, 4, 4, 32, 0, 0, in); test_ds_round(VKC_DS_HADAMARD, 2, 1, 1, 0, 0, in);
        }
        test_ds_swiglu(5000, 10.0f); test_ds_swiglu(3000, 0.0f);
        printf("dsv4 rounding done\n");
    }
    if (!vkc_kvs_ready()) { fails++; printf("FAIL: the split KV shader did not load\n"); }
    else { test_kvs(); test_kvs_cold(); printf("kvs done\n"); }
    if (getenv("COLI_VK_DEV2")) { test_dev2(spv); printf("dev2 done\n"); }
    /* a diffusion step's attention: one tile, partial tiles of rows and keys, every head
     * dim (last: the shared random stream the tests above draw from stays theirs) */
    test_attn_full(5, 9, 2, 32); test_attn_full(64, 64, 1, 128); test_attn_full(130, 200, 3, 128);
    test_attn_full(70, 129, 2, 64); test_attn_full(1, 300, 4, 128);
    printf("attn full done\n");
    if (vkc_attn_full_coop_ready(128)) {   /* on the matrix units, where the device has them */
        test_attn_full_k(64, 64, 1, 128, 1); test_attn_full_k(130, 200, 3, 128, 1);
        test_attn_full_k(70, 129, 2, 64, 1); test_attn_full_k(1, 300, 4, 128, 1);
        printf("attn full coop done\n");
    } else printf("attn full coop: no matrix units here\n");
    test_ew_gate();
    test_vae_ops(5, 7, 3, 0, 0, 5); test_vae_ops(5, 7, 3, 1, 3, 4); test_vae_ops(4, 4, 16, 1, 0, 8);
    printf("vae ops done\n");
    VkcStats st; vkc_stats(&st);
    printf("chain: %llu frames, %llu ops, %llu matmuls (%llu GEMM), %llu barriers\n", st.frames, st.ops, st.matmuls, st.gemms, st.barriers);
    vkc_shutdown();
    coli_vk_shutdown();
    printf(fails ? "FAIL (%d)\n" : "PASS\n", fails);
    return fails ? 1 : 0;
}
