/* COLI_VK_KV_QUANT's codec (chain_dsv4.comp modes 13 and 14, and the quantized readers
 * of modes 0's attention / 3, 9, 10's index scores) against c/dsv41_kvq.h's reference,
 * the formats DeepInfer's cache defines (a window row 528 B of MXFP8, a compressed row
 * 288 B of FP4, an index key 68 B of FP4 in block-wide sections):
 *   pack     the device's pack of fp32 rows must be the host's bit for bit (the same
 *            amax, ceil-pow2/e4m3 scale and RNE codes), including the edge rows below
 *   unpack   the device's unpack of those bytes must be the host's (rel_l2 <= 1e-5; the
 *            decode is exact arithmetic, so it is 0 in practice)
 *   upload   a mirror the host packed and wrote up (the chain's re-upload) reads back
 *            the same rows -- the little-endian word lanes the shader assumes, end to end
 *   again    a row re-packed after its dequantization keeps its bytes (idempotence),
 *            which the pull's write-back and a later re-upload rely on
 *   scores   the index scores over the quantized keys are the scores over their
 *            dequantized rows bit for bit, and track the original rows' top-k (the
 *            overlap, as test_vk_chain's index tests compare lists), which is what the
 *            approximate mode costs
 * The edge rows: an amax of zero (all-zero, -0 and denormal rows), an amax exactly a
 * power of two times the format's max (448 and 896 for the window, 6 and 12 for the FP4
 * rows), the e4m3-scale clamp floor (an all-zero compressed row) and the RNE ties on
 * both grids. Skips (exit 0) when no Vulkan device or the shader is missing. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "../backend_vulkan.h"
#include "../vk_chain.h"
#include "../dsv41_kvq.h"

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fails++; printf("FAIL: " __VA_ARGS__); putchar('\n'); } } while (0)

static VkcBuf *upf(const void *v, size_t bytes) {   /* a device buffer holding v */
    VkcBuf *b = vkc_buf(bytes, VKC_DEV);
    if (!b) return NULL;
    vkc_begin(); vkc_write(b, 0, v, bytes); vkc_submit(1);
    return b;
}
static void *grab(VkcBuf *b, size_t bytes) {
    void *p = malloc(bytes);
    if (!p || !vkc_read(b, 0, p, bytes)) { free(p); return NULL; }
    return p;
}
static double rel_l2(const float *a, const float *b, int n) {
    double num = 0, den = 0;
    for (int i = 0; i < n; i++) { double d = (double)a[i] - b[i]; num += d * d; den += (double)b[i] * b[i]; }
    return sqrt(num / (den + 1e-30));
}

/* R rows of hd floats: random magnitudes, exact grid and tie values, then the edge rows */
static void fill_rows(float *x, int R, int hd, int e4m3) {
    srand(11);
    for (int i = 0; i < R * hd; i++) {
        int k = rand() % 4;
        if (k == 0) x[i] = (float)((rand() % 2001) - 1000) / 500.0f * ldexpf(1.0f, rand() % 7 - 3);
        else if (k == 1) x[i] = (float)((rand() % 65) - 32) / 16.0f;          /* ties on both grids */
        else if (k == 2) x[i] = (float)((rand() % 17) - 8) * 0.5f;
        else x[i] = (float)((rand() % 33) - 16) / 512.0f;
    }
    int e = 0;
#define EDGE(f) do { if (e >= R) return; for (int i = 0; i < hd; i++) x[e * hd + i] = (f); e++; } while (0)
    EDGE(0.0f);                                             /* a zero amax: the margin's scale */
    EDGE(-0.0f);                                            /* negative zero keeps its sign */
    EDGE((i & 1 ? -1.0f : 1.0f) * 1e-42f);                  /* denormals, below every margin */
    EDGE((i & 1 ? -1.0f : 1.0f) * (e4m3 ? 448.0f : 6.0f));  /* the format's max */
    EDGE((i & 1 ? -1.0f : 1.0f) * (e4m3 ? 896.0f : 12.0f)); /* an amax a power of two up */
    EDGE((i & 1 ? -1.0f : 1.0f) * (e4m3 ? 447.99f : 5.99f));/* just under the clamp */
    EDGE((i & 3) * (i & 4 ? -0.25f : 0.25f) + 0.5f * (i & 1));   /* the FP4 tie ladder */
    EDGE((float)((i % 16) - 8) / 16.0f);                    /* the e4m3 mantissa's ties */
    EDGE(0.0f);                                             /* a second zero row: the e4m3-scale floor */
#undef EDGE
}

/* the win/ckv round trip (a row packs to rowb contiguous bytes). ring: the rows land in
 * ring slots (dst_tok + r) % cap (the window ring's scatter); else at token dst_tok + r. */
static void codec(const char *what, int kind, int ring, int R, int hd, size_t rowb, int cap, int dst_tok,
                  void (*pack_row)(uint8_t *, const float *), void (*unpack_row)(float *, const uint8_t *)) {
    float *x = malloc((size_t)R * hd * sizeof(float));
    uint8_t *hp = calloc((size_t)cap, rowb), again[KVQ_WIN_BYTES];
    float *hu = malloc((size_t)R * hd * sizeof(float)), *hu2 = malloc((size_t)R * hd * sizeof(float));
    VkcBuf *src = NULL, *m = NULL, *dq = NULL, *m2 = NULL, *dq2 = NULL;
    if (!x || !hp || !hu || !hu2 || rowb > sizeof again) { CHECK(0, "%s: out of memory", what); goto done; }
    fill_rows(x, R, hd, kind != 2);
    for (int r = 0; r < R; r++) pack_row(hp + (size_t)(ring ? (dst_tok + r) % cap : dst_tok + r) * rowb, x + (size_t)r * hd);
    for (int r = 0; r < R; r++) unpack_row(hu + (size_t)r * hd, hp + (size_t)(ring ? (dst_tok + r) % cap : dst_tok + r) * rowb);
    if (!(src = upf(x, (size_t)R * hd * sizeof(float))) || !(m = vkc_buf((size_t)cap * rowb, VKC_DEV)) ||
        !(dq = vkc_buf((size_t)R * hd * sizeof(float), VKC_DEV)) || !(m2 = vkc_buf((size_t)cap * rowb, VKC_DEV))) {
        CHECK(0, "%s: device memory refused", what); goto done;
    }
    vkc_begin();
    int ok = vkc_dsv4_quant(m, src, &(VkcDsQuant){kind, R, hd, 0, hd, dst_tok, ring ? cap : 0}) &&
             vkc_dsv4_dequant(dq, m, &(VkcDsDequant){kind, R, hd, dst_tok, ring ? cap : 0, 0, hd});
    vkc_write(m2, 0, hp, (size_t)cap * rowb);                    /* the host's pack, as a re-upload */
    vkc_submit(1);
    uint8_t *gp = grab(m, (size_t)cap * rowb);
    float *gd = grab(dq, (size_t)R * hd * sizeof(float));
    CHECK(ok && gp && !memcmp(gp, hp, (size_t)cap * rowb), "%s: the device's pack is not the host's bit for bit", what);
    double rl = gd ? rel_l2(gd, hu, R * hd) : 1;
    CHECK(rl <= 1e-5, "%s: the device's unpack differs from the host's (rel_l2 %.3e)", what, rl);
    if ((dq2 = vkc_buf((size_t)R * hd * sizeof(float), VKC_DEV))) {   /* the host-written mirror */
        vkc_begin();
        int ok2 = vkc_dsv4_dequant(dq2, m2, &(VkcDsDequant){kind, R, hd, dst_tok, ring ? cap : 0, 0, hd});
        vkc_submit(1);
        float *gd2 = grab(dq2, (size_t)R * hd * sizeof(float));
        double rl2 = gd2 ? rel_l2(gd2, hu, R * hd) : 1;
        CHECK(ok2 && rl2 <= 1e-5, "%s: a host-written mirror unpacks differently (rel_l2 %.3e)", what, rl2);
        free(gd2);
    } else CHECK(0, "%s: device memory refused (2)", what);
    int stab = 1;   /* pack(dequant(pack)) is a fixpoint in VALUES: a group whose amax
     * landed at half the scale's span (224 of 448) re-packs one scale step down -- the
     * same decoded values in different bytes -- so what the pull's write-back and a
     * re-upload rely on is that a dequantized row re-packs to a row that decodes to it */
    for (int r = 0; r < R && stab; r++) {
        pack_row(again, hu + (size_t)r * hd);
        unpack_row(hu2 + (size_t)r * hd, again);
        stab = !memcmp(hu2 + (size_t)r * hd, hu + (size_t)r * hd, (size_t)hd * sizeof(float));
    }
    CHECK(stab, "%s: a dequantized row does not re-pack to its own values", what);
    printf("OK kv-quant %s: %d rows pack bit for bit, unpack rel_l2 %.2e, the re-pack a fixpoint in values\n", what, R, rl);
    free(gd); free(gp);
done:
    vkc_free(src); vkc_free(m); vkc_free(dq); vkc_free(m2); vkc_free(dq2);
    free(x); free(hp); free(hu); free(hu2);
}

/* the index keys' round trip in their block-wide sections (a row packs into a mirror at
 * a token, not to contiguous bytes), tokens 5..44 crossing the 32-token block boundary */
static void codec_ikey(void) {
    const int R = 40, hd = 128, tok = 5, cap = tok + R;
    const size_t mb = kvq_ikey_mirror_bytes((size_t)cap);   /* the blocks are whole */
    float *x = malloc((size_t)R * hd * sizeof(float));
    uint8_t *hp = calloc(mb, 1), *again = calloc(KVQ_IKEY_BLOCK * KVQ_IKEY_BYTES, 1);
    float *hu = malloc((size_t)R * hd * sizeof(float)), *hu2 = malloc((size_t)R * hd * sizeof(float));
    VkcBuf *src = NULL, *m = NULL, *dq = NULL, *m2 = NULL, *dq2 = NULL;
    if (!x || !hp || !again || !hu || !hu2) { CHECK(0, "ikey: out of memory"); goto done; }
    fill_rows(x, R, hd, 0);
    for (int r = 0; r < R; r++) kvq_pack_ikey(hp, tok + r, x + (size_t)r * hd);
    for (int r = 0; r < R; r++) kvq_unpack_ikey(hu + (size_t)r * hd, hp, tok + r);
    if (!(src = upf(x, (size_t)R * hd * sizeof(float))) || !(m = vkc_buf(mb, VKC_DEV)) ||
        !(dq = vkc_buf((size_t)R * hd * sizeof(float), VKC_DEV)) || !(m2 = vkc_buf(mb, VKC_DEV))) {
        CHECK(0, "ikey: device memory refused"); goto done;
    }
    vkc_begin();
    int ok = vkc_dsv4_quant(m, src, &(VkcDsQuant){2, R, hd, 0, hd, tok, 0}) &&
             vkc_dsv4_dequant(dq, m, &(VkcDsDequant){2, R, hd, tok, 0, 0, hd});
    vkc_write(m2, 0, hp, mb);
    vkc_submit(1);
    uint8_t *gp = grab(m, mb);
    float *gd = grab(dq, (size_t)R * hd * sizeof(float));
    CHECK(ok && gp && !memcmp(gp, hp, mb), "ikey: the device's pack is not the host's bit for bit");
    double rl = gd ? rel_l2(gd, hu, R * hd) : 1;
    CHECK(rl <= 1e-5, "ikey: the device's unpack differs from the host's (rel_l2 %.3e)", rl);
    if ((dq2 = vkc_buf((size_t)R * hd * sizeof(float), VKC_DEV))) {
        vkc_begin();
        int ok2 = vkc_dsv4_dequant(dq2, m2, &(VkcDsDequant){2, R, hd, tok, 0, 0, hd});
        vkc_submit(1);
        float *gd2 = grab(dq2, (size_t)R * hd * sizeof(float));
        double rl2 = gd2 ? rel_l2(gd2, hu, R * hd) : 1;
        CHECK(ok2 && rl2 <= 1e-5, "ikey: a host-written mirror unpacks differently (rel_l2 %.3e)", rl2);
        free(gd2);
    } else CHECK(0, "ikey: device memory refused (2)");
    int stab = 1;   /* a dequantized row re-packs to a row that decodes to it (values, as codec's note) */
    for (int r = 0; r < R && stab; r++) {
        memset(again, 0, KVQ_IKEY_BLOCK * KVQ_IKEY_BYTES);
        kvq_pack_ikey(again, tok + r - (tok + r) / KVQ_IKEY_BLOCK * KVQ_IKEY_BLOCK, hu + (size_t)r * hd);
        kvq_unpack_ikey(hu2 + (size_t)r * hd, again, (tok + r) % KVQ_IKEY_BLOCK);
        stab = !memcmp(hu2 + (size_t)r * hd, hu + (size_t)r * hd, (size_t)hd * sizeof(float));
    }
    CHECK(stab, "ikey: a dequantized row does not re-pack to its own values");
    printf("OK kv-quant ikey (from token %d): %d rows pack bit for bit across the 32-token block boundary, unpack rel_l2 %.2e\n",
           tok, R, rl);
    free(gd); free(gp);
done:
    vkc_free(src); vkc_free(m); vkc_free(dq); vkc_free(m2); vkc_free(dq2);
    free(x); free(hp); free(again); free(hu); free(hu2);
}

/* the index scores over the quantized keys: bit for bit the scores over the rows the
 * pack decoded, and the original rows' top-k tracked (the overlap test_vk_chain's index
 * tests make bit for bit; the quantized keys cannot, so the overlap) */
static void score_case(int S, int pos_base, int IH) {
    const int ID = 128, ratio = 4, topk = 16;
    int width = (pos_base + S) / ratio;
    float wscale = 0.25f, scrow = width + 3;
    float *q = malloc((size_t)S * IH * ID * sizeof(float)), *w = malloc((size_t)S * IH * sizeof(float));
    float *k = malloc((size_t)width * ID * sizeof(float)), *kd = malloc((size_t)width * ID * sizeof(float));
    const size_t kmb = kvq_ikey_mirror_bytes((size_t)width);   /* the keys' blocks are whole */
    uint8_t *km = calloc(kmb, 1);
    float *ref = malloc((size_t)S * width * sizeof(float)), *ref0 = malloc((size_t)S * width * sizeof(float));
    float *refc = malloc((size_t)S * width * sizeof(float)), *ref0c = malloc((size_t)S * width * sizeof(float));
    int *top1 = malloc((size_t)S * topk * sizeof(int)), *top0 = malloc((size_t)S * topk * sizeof(int));
    VkcBuf *qb = NULL, *wb = NULL, *kq = NULL, *kmq = NULL, *s1 = NULL, *s2 = NULL;
    if (!q || !w || !k || !kd || !km || !ref || !ref0 || !top1 || !top0) { CHECK(0, "scores: out of memory"); goto done; }
    srand(13);
    for (int i = 0; i < S * IH * ID; i++) q[i] = (float)((rand() % 33) - 16) / 16.0f;
    for (int i = 0; i < S * IH; i++) w[i] = (float)((rand() % 9) - 4) * 0.5f;
    for (int i = 0; i < width * ID; i++) k[i] = (float)((rand() % 33) - 16) / 16.0f * (float)(1 << (rand() % 3));
    for (int t = 0; t < width; t++) { kvq_pack_ikey(km, t, k + (size_t)t * ID); kvq_unpack_ikey(kd + (size_t)t * ID, km, t); }
    for (int s = 0; s < S; s++) {                       /* the CPU's scores, over the decoded and the original rows
                                                         * (the sums in double: the device's are float and in its own
                                                         * order, compared within a tolerance, as test_vk_chain does) */
        int lens = (pos_base + s + 1) / ratio;
        for (int j = 0; j < width; j++) {
            double t1 = 0, t0 = 0;
            if (j < lens) {
                for (int h = 0; h < IH; h++) {
                    double d1 = 0, d0 = 0;
                    float wt = w[(size_t)s * IH + h] * wscale;
                    for (int i = 0; i < ID; i++) { d1 += (double)q[(size_t)s * IH * ID + h * ID + i] * kd[(size_t)j * ID + i];
                                                    d0 += (double)q[(size_t)s * IH * ID + h * ID + i] * k[(size_t)j * ID + i]; }
                    if (d1 > 0) t1 += d1 * wt;
                    if (d0 > 0) t0 += d0 * wt;
                }
            } else { t1 = -INFINITY; t0 = -INFINITY; }
            ref[(size_t)s * width + j] = (float)t1; ref0[(size_t)s * width + j] = (float)t0;
        }
        memcpy(refc + (size_t)s * width, ref + (size_t)s * width, (size_t)width * sizeof(float));
        memcpy(ref0c + (size_t)s * width, ref0 + (size_t)s * width, (size_t)width * sizeof(float));
        for (int t = 0; t < topk; t++) {                /* each row's top-k, ties to the lower column */
            int b1 = -1, b0 = -1;
            for (int j = 0; j < width; j++) {
                if (b1 < 0 || refc[(size_t)s * width + j] > refc[(size_t)s * width + b1]) b1 = j;
                if (b0 < 0 || ref0c[(size_t)s * width + j] > ref0c[(size_t)s * width + b0]) b0 = j;
            }
            if (b1 < 0) { top1[(size_t)s * topk + t] = top0[(size_t)s * topk + t] = -1; continue; }
            top1[(size_t)s * topk + t] = b1; top0[(size_t)s * topk + t] = b0;
            refc[(size_t)s * width + b1] = -INFINITY; ref0c[(size_t)s * width + b0] = -INFINITY;
        }
    }
    if (!(qb = upf(q, (size_t)S * IH * ID * sizeof(float))) || !(wb = upf(w, (size_t)S * IH * sizeof(float))) ||
        !(kq = upf(kd, (size_t)width * ID * sizeof(float))) || !(kmq = upf(km, kmb)) ||
        !(s1 = vkc_buf((size_t)S * scrow * sizeof(float), VKC_DEV)) || !(s2 = vkc_buf((size_t)S * scrow * sizeof(float), VKC_DEV))) {
        CHECK(0, "scores: device memory refused"); goto done;
    }
    VkcDsScore sp = {S, pos_base, ratio, IH, ID, width, 0, IH * ID, 0, IH, 0, ID, 0, scrow, wscale, 0, 0, 0};
    VkcDsScore sq = sp; sq.quant = 1;
    vkc_begin();
    int ok = vkc_dsv4_score(qb, wb, kq, NULL, s1, &sp) && vkc_dsv4_score(qb, wb, kmq, NULL, s2, &sq);
    vkc_submit(1);
    float *g1 = grab(s1, (size_t)S * scrow * sizeof(float)), *g2 = grab(s2, (size_t)S * scrow * sizeof(float));
    int same = 1;
    double num = 0, den = 0, num0 = 0;
    for (int s = 0; s < S; s++)
        for (int j = 0; j < width; j++) {
            int i = s * scrow + j;
            same &= g1 && g2 && g1[i] == g2[i];   /* the reader decodes the packer's bits */
            if (isfinite(ref[(size_t)s * width + j])) {
                double d = (double)g1[i] - ref[(size_t)s * width + j];
                num += d * d; den += (double)ref[(size_t)s * width + j] * ref[(size_t)s * width + j];
                double d0 = (double)g1[i] - ref0[(size_t)s * width + j];
                num0 += d0 * d0;
            }
        }
    int over = 0;
    for (int s = 0; s < S; s++) for (int t = 0; t < topk; t++) {
        int a = top1[(size_t)s * topk + t];
        for (int u = 0; u < topk; u++) if (top0[(size_t)s * topk + u] == a) { over++; break; }
    }
    CHECK(ok && same, "scores (S %d pos %d IH %d): the quantized keys' scores are not the decoded rows' bit for bit", S, pos_base, IH);
    double rel = sqrt(num / (den + 1e-30)), rel0 = sqrt(num0 / (den + 1e-30));
    double ov = (double)over / (S * topk);
    CHECK(rel <= 1e-5, "scores (S %d pos %d IH %d): rel_l2 to the reference %.3e", S, pos_base, IH, rel);
    /* the drift against the ORIGINAL rows is the FP4 format's own noise on random data
     * (a group's per-32 scale washes out its small values: 2-3 significant bits), which
     * the overlap below shows does not move the selection much */
    CHECK(rel0 <= 2e-1, "scores (S %d pos %d IH %d): rel_l2 to the original rows' scores %.3e", S, pos_base, IH, rel0);
    CHECK(ov >= 0.7, "scores (S %d pos %d IH %d): the top-%d overlap %.0f%% is below 70%%", S, pos_base, IH, topk, ov * 100);
    printf("OK kv-quant scores (S %d pos %d IH %d): the decoded rows' bits, rel_l2 %.2e vs the reference and %.2e vs the original rows, top-%d overlap %.0f%%\n",
           S, pos_base, IH, rel, rel0, topk, ov * 100);
    free(g1); free(g2);
done:
    vkc_free(qb); vkc_free(wb); vkc_free(kq); vkc_free(kmq); vkc_free(s1); vkc_free(s2);
    free(q); free(w); free(k); free(kd); free(km); free(ref); free(ref0); free(refc); free(ref0c); free(top1); free(top0);
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);   /* a crash's prints must not sit in the buffer */
    if (!coli_vk_init("shaders/chain_dsv4.spv") || !coli_vk_available()) {
        fprintf(stderr, "vk-kv-quant: no Vulkan device -- skipped\n"); return 0;
    }
    if (!vkc_init() || !vkc_dsv4_ready()) {
        fprintf(stderr, "vk-kv-quant: the chain_dsv4 shader is missing -- skipped\n"); coli_vk_shutdown(); return 0;
    }
    codec("win (the ring scattered)", 0, 1, 16, 512, KVQ_WIN_BYTES, 16, 13, kvq_pack_win, kvq_unpack_win);
    codec("ckv (from token 7)", 1, 0, 16, 512, KVQ_CKV_BYTES, 23, 7, kvq_pack_ckv, kvq_unpack_ckv);
    codec_ikey();
    score_case(3, 512, 8);    /* a wide table: the grid's mode 10 */
    score_case(2, 64, 8);     /* a narrow one: mode 3 */
    score_case(2, 512, 65);   /* the wide queries: mode 9 */
    if (fails) { printf("FAIL: kv-quant (%d)\n", fails); coli_vk_shutdown(); return 1; }
    printf("PASS: kv-quant\n");
    coli_vk_shutdown();
    return 0;
}
