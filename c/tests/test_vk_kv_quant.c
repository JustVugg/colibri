/* The packed cache's codec (chain_dsv4.comp modes 13 and 14, and the quantized readers
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
 * rows), the e4m3-scale clamp floor (an all-zero compressed row), the RNE ties on both
 * grids, and the ckv subnormal-scale band [9*2^-9, 10*2^-9] with its edges -- the slice
 * kvq_pack_ckv's demote buys the re-pack fixpoint in. Skips (exit 0, the host half's
 * verdict aside) when no Vulkan device or the shader is missing. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "../backend_vulkan.h"
#include "../vk_chain.h"
#include "../dsv41_kvq.h"
#include "../park.h"

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
/* the device half never runs: the host half's verdict stands, so a host failure is not
 * swallowed by the skip (the gate reads the last line, this prints FAIL there) */
static int skip(const char *why) {
    fprintf(stderr, "vk-kv-quant: %s -- skipped\n", why);
    if (fails) { printf("FAIL: kv-quant (%d)\n", fails); return 1; }
    return 0;
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
    /* the ckv subnormal-scale band and its edges (kvq_pack_ckv's demote note): a group
     * amax in [9*2^-9, 10*2^-9] picks sf 0x02 whose scaled amax sits at the e2m1 top tie
     * or under it -- the demote steps it to 0x01 -- and 15*2^-9 with the outsides 8.9
     * and 11 hold their code; every group of these rows carries the edge amax exactly */
    EDGE((i & 1 ? -1.0f : 1.0f) * ldexpf(9.0f, -9));        /* the band's left edge: the tie rounds to sf 0x02 */
    EDGE((i & 1 ? -1.0f : 1.0f) * ldexpf(9.5f, -9));        /* inside the band */
    EDGE((i & 1 ? -1.0f : 1.0f) * ldexpf(10.0f, -9));       /* the right edge: the scaled amax is the 5.0 tie */
    EDGE((i & 1 ? -1.0f : 1.0f) * ldexpf(15.0f, -9));       /* the sf 0x03 tie point: even, so 0x02 holds */
    EDGE((i & 1 ? -1.0f : 1.0f) * ldexpf(8.9f, -9));        /* just under the band: sf 0x01 */
    EDGE((i & 1 ? -1.0f : 1.0f) * ldexpf(11.0f, -9));       /* just over it: sf 0x02 at a scaled amax over the tie */
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
    const int R = 48, hd = 128, tok = 5, cap = tok + R;
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
static void score_case(int S, int pos_base, int IH, int topk) {
    const int ID = 128, ratio = 4;
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
     * (a group's per-32 scale washes out its small values: 2 significant bits), which
     * the overlap below shows does not move the selection much. The thresholds are the
     * measured band: 1.01-1.28e-1 / 89-100% for the five arms below, and 1.46-1.88e-1 /
     * 81-92% with the scale rule floored instead of ceiled in BOTH the header and
     * chain_dsv4.comp's ue8m0c (the mutation the old 2e-1 / 0.7 gates let through: the
     * suite then printed PASS with all five arms in range) */
    CHECK(rel0 <= 1.35e-1, "scores (S %d pos %d IH %d): rel_l2 to the original rows' scores %.3e", S, pos_base, IH, rel0);
    CHECK(ov >= 0.80, "scores (S %d pos %d IH %d): the top-%d overlap %.0f%% is below 80%%", S, pos_base, IH, topk, ov * 100);
    printf("OK kv-quant scores (S %d pos %d IH %d): the decoded rows' bits, rel_l2 %.2e vs the reference and %.2e vs the original rows, top-%d overlap %.0f%%\n",
           S, pos_base, IH, rel, rel0, topk, ov * 100);
    free(g1); free(g2);
done:
    vkc_free(qb); vkc_free(wb); vkc_free(kq); vkc_free(kmq); vkc_free(s1); vkc_free(s2);
    free(q); free(w); free(k); free(kd); free(km); free(ref); free(ref0); free(refc); free(ref0c); free(top1); free(top0);
}

/* The host-side block rules, no device needed (the packed cache, the host arrays in
 * these formats):
 *   staging  a mirror's rows through 68-byte linear staging and back, bit for bit --
 *            the pull's write-back and a raw index-key upload go through these
 *   prefix   the whole blocks covering rows [0, n) are the linear prefix
 *            kvq_ikey_mirror_bytes(n) -- park/reset/zero keep whole blocks
 *   slide    a window ring's two-arm slide (the prefill chunk copy) is the same bytes
 *            at 528-byte rows: the ring stays linear, only its row changed width
 *   re-pack  a compressed row dequantized and packed again decodes to itself, on
 *            fill_rows' rows (its six band edges are where kvq_pack_ckv's demote does
 *            the work), here so a device-less box runs it (the codec's CHECK needs the
 *            device) */
static void host_block_tests(void) {
    enum { R = 70, BLK = KVQ_IKEY_BLOCK, IKB = BLK * KVQ_IKEY_BYTES };   /* 3 blocks, a partial last */
    const size_t mb = kvq_ikey_mirror_bytes(R);
    float x[R * 128], a[128], b[128];
    uint8_t *m1 = malloc(mb), *m2 = malloc(mb), row[KVQ_IKEY_BYTES], row2[KVQ_IKEY_BYTES];
    CHECK(m1 && m2, "staging: no memory");
    if (m1 && m2) {
        memset(m1, 0, mb); memset(m2, 0, mb);
        srand(29);
        for (int i = 0; i < R * 128; i++) x[i] = (float)((rand() % 2001) - 1000) / 500.0f * ldexpf(1.0f, rand() % 5 - 2);
        for (int t = 0; t < R; t++) kvq_pack_ikey(m1, t, x + (size_t)t * 128);
        for (int t = 0; t < R; t++) {   /* every row through the staging, blocks apart */
            kvq_ikey_row_gather(row, m1, t);
            kvq_ikey_row_scatter(m2, t, row);
            kvq_unpack_ikey(a, m1, t); kvq_unpack_ikey(b, m2, t);
            CHECK(memcmp(a, b, sizeof a) == 0, "staging: row %d decodes differently after gather/scatter", t);
        }
        CHECK(memcmp(m1, m2, mb) == 0, "staging: the scattered mirror is not the packed one");
        int over = 0;   /* a partial row cut at 2048 strands scales: the prefix must round */
        for (int rows = 1; rows <= R + BLK; rows++) {
            size_t want = (size_t)((rows + BLK - 1) / BLK) * IKB;
            if (kvq_ikey_mirror_bytes((size_t)rows) != want) over++;
        }
        CHECK(over == 0, "prefix: kvq_ikey_mirror_bytes is not whole blocks from byte 0");
        memset(m2, 0, mb);   /* the block-rounded prefix copy is a plain memcpy */
        memcpy(m2, m1, kvq_ikey_mirror_bytes((size_t)33));   /* keep rows [0,33): blocks 0..1, rows 0..63, whole */
        int moved = 0;
        for (int t = 0; t < 64; t++) {
            kvq_unpack_ikey(a, m1, t); kvq_unpack_ikey(b, m2, t);
            if (memcmp(a, b, sizeof a)) moved++;
        }
        CHECK(moved == 0, "prefix: %d of the 64 rows a 33-row keep covers moved", moved);
        kvq_ikey_row_gather(row, m1, 33); kvq_ikey_row_scatter(m2, 33, row);   /* a write inside the kept blocks */
        kvq_ikey_row_gather(row2, m2, 33);
        CHECK(memcmp(row, row2, KVQ_IKEY_BYTES) == 0, "prefix: a staged write inside the kept blocks did not land");
    }
    free(m1); free(m2);
    /* park's block rounding: a row a `per` positions, whole blocks kept */
    static const struct { int n, rows; size_t want; } pk[] = {
        {1, 96, (size_t)32 * KVQ_IKEY_BYTES}, {63, 96, (size_t)32 * KVQ_IKEY_BYTES},
        {64, 96, (size_t)32 * KVQ_IKEY_BYTES}, {65, 96, (size_t)64 * KVQ_IKEY_BYTES},
        {193, 96, (size_t)96 * KVQ_IKEY_BYTES}, {10000, 96, (size_t)96 * KVQ_IKEY_BYTES},
    };
    for (size_t i = 0; i < sizeof pk / sizeof *pk; i++) {
        ColiParkPart p = coli_park_per_blocks((void *)1, KVQ_IKEY_BYTES, pk[i].rows, 2, KVQ_IKEY_BLOCK);
        CHECK(coli_park_part_bytes(&p, pk[i].n) == pk[i].want,
              "park: %d positions keep %zu B, not the whole-block %zu", pk[i].n,
              coli_park_part_bytes(&p, pk[i].n), pk[i].want);
    }
    /* the ckv re-pack fixpoint on the host: pack, dequantize, pack again, dequantize --
     * the two decodes must be the same values, which is what the pull's write-back and a
     * later re-upload rely on (the codec's CHECK needs a device; this one does not).
     * fill_rows puts its edges first, so its six band-edge rows are rows 9-14, and they
     * are where kvq_pack_ckv's demote does the work: without it 9, 9.5 and 10*2^-9 pick
     * sf 0x02, decode 8*2^-9, and re-pack to sf 0x01, hopping their large elements to
     * 6*2^-9 -- this CHECK fails on three rows. */
    {
        enum { BR = 24, BHD = 512 };   /* the codec's ckv rows: 15 edges (6 of them the band), 9 random */
        float *bx = malloc((size_t)BR * BHD * sizeof(float)), *bu = malloc((size_t)BHD * sizeof(float)),
              *bu2 = malloc((size_t)BHD * sizeof(float));
        uint8_t brow[KVQ_CKV_BYTES];
        int hop = 0;
        CHECK(bx && bu && bu2, "ckv: out of memory");
        if (bx && bu && bu2) {
            fill_rows(bx, BR, BHD, 1);   /* the codec's ckv rows, bit for bit */
            for (int r = 0; r < BR; r++) {
                kvq_pack_ckv(brow, bx + (size_t)r * BHD);
                kvq_unpack_ckv(bu, brow);
                kvq_pack_ckv(brow, bu);
                kvq_unpack_ckv(bu2, brow);
                hop += memcmp(bu, bu2, (size_t)BHD * sizeof(float)) != 0;
            }
            CHECK(hop == 0, "ckv: %d of %d rows (the band edges 9-14 among them) do not re-pack to their own values",
                  hop, BR);
        }
        free(bx); free(bu); free(bu2);
    }
    /* the window ring's slide: the last W of N rows in, position p to slot p % win.
     * The fp32 code's two memcpy arms land there; the packed writer (a pack a row from
     * the fp32 scratch) must land on the same slots -- the ring stays linear, only its
     * row changed width, so a raw two-arm byte slide would agree too */
    enum { W = 8, N = 11 };
    float kv[N * 512];
    uint8_t ring[W * KVQ_WIN_BYTES], ref[W * KVQ_WIN_BYTES];
    srand(31);
    for (int i = 0; i < N * 512; i++) kv[i] = (float)((rand() % 2001) - 1000) / 500.0f;
    memset(ref, 0, sizeof ref);
    for (int p = 0; p < N; p++) kvq_pack_win(ref + (size_t)(p % W) * KVQ_WIN_BYTES, kv + (size_t)p * 512);
    memset(ring, 0, sizeof ring);
    for (int p = N - W; p < N; p++) kvq_pack_win(ring + (size_t)(p % W) * KVQ_WIN_BYTES, kv + (size_t)p * 512);
    CHECK(memcmp(ring, ref, sizeof ring) == 0, "slide: the packed tail is not the per-position pack");
    printf("OK kv-quant host blocks: staging bit for bit, the prefix whole, park rounded, the slide linear, "
           "the re-pack a fixpoint on the host\n");
}

/* a row's attention as sparse_attn.h's formula (and chain_dsv4.comp's mode 0): the score
 * (q . k) * scale with the sink in the denominator only, the weights exp(sc - best), the
 * value sum and the denominator in list order, entry e < 0 skipped -- the dot over i
 * ascending, as mode 0's one thread a (row, entry) does it. sc: cnt floats of scratch */
static void attn_row(float *o, const float *q, const float *kv, const int *idx, int n, int hd, float sink, float scale,
                     float *sc) {
    float best = -1e30f;
    for (int j = 0; j < n; j++) {
        if (idx[j] < 0) { sc[j] = -INFINITY; continue; }
        float d = 0;
        for (int i = 0; i < hd; i++) d += q[i] * kv[(size_t)idx[j] * hd + i];
        sc[j] = d * scale;
        if (sc[j] > best) best = sc[j];
    }
    float den = expf(sink - best);
    for (int i = 0; i < hd; i++) o[i] = 0;
    for (int j = 0; j < n; j++) {
        if (idx[j] < 0) continue;
        float w = expf(sc[j] - best);
        den += w;
        for (int i = 0; i < hd; i++) o[i] += w * kv[(size_t)idx[j] * hd + i];
    }
    for (int i = 0; i < hd; i++) o[i] /= den;
}

/* mode 0 over the packed mirrors (vkc_dsv4_attn's flags 2) against two arms that differ
 * from it in exactly one thing -- how the row's floats are produced:
 *   fp32   mode 0 over the rows kvq_unpack_win/kvq_unpack_ckv decode from the SAME bytes
 *          (a separate buffer), so the two runs are the same kernel on the same values and
 *          their outputs must be equal BIT FOR BIT. A wrong code, scale offset or group
 *          index in the shader's decoder breaks it; nothing else in the kernel differs.
 *   scalar attn_row over those decoded rows: the formula, and the only tolerances are
 *          expf's (libm against llvmpipe's exp) and a contracted fma -- 2e-5, as mode 0
 *          against sparse_attn.h's kernel (tests/test_vk_chain.c's test_ds_attn).
 * The list mixes window and compressed rows (row 0's walks every row of both -- a window
 * ring of exactly nwin slots read whole), with duplicates and skipped entries.
 * one: a single entry a row and the sink -inf, so the weight is exp(0) = 1, the denominator
 * 1 and the output the value row's own floats, compared bit for bit.
 * mut (VK_KV_QUANT_MUT=win|ckv): one scale byte of the packed mirror is changed by one
 * code AFTER the fp32 arm's rows were decoded from the original bytes, and the packed arm
 * must move -- the plain run's "the arms agree" is a test only if this run makes them
 * differ, so a run with the variable set fails where the plain one passes. */
static void attn_packed(const char *what, int S, int H, int nwin, int ncmp, int cnt, int one, int mut) {
    const int hd = 512;
    const size_t wby = (size_t)nwin * KVQ_WIN_BYTES, cby = (size_t)ncmp * KVQ_CKV_BYTES;
    const size_t rowb = (size_t)H * hd * 4, allb = (size_t)S * rowb;
    uint8_t *pw = malloc(wby), *pc = malloc(cby);
    float *kw = malloc((size_t)nwin * hd * 4), *kc = malloc((size_t)ncmp * hd * 4);
    float *kv = malloc((size_t)(nwin + ncmp) * hd * 4);
    float *qq = malloc((size_t)S * H * hd * 4), *sk = malloc((size_t)H * 4);
    float *ref = malloc(allb), *sc = malloc((size_t)(cnt > 0 ? cnt : 1) * 4);
    int *li = malloc((size_t)S * cnt * 4);
    VkcBuf *qd = NULL, *wd = NULL, *cd = NULL, *wf = NULL, *cf = NULL, *ld = NULL, *sd = NULL, *op = NULL, *of = NULL, *wm = NULL, *cd2 = NULL, *om = NULL;
    if (!pw || !pc || !kw || !kc || !kv || !qq || !sk || !ref || !sc || !li) { CHECK(0, "%s: out of memory", what); goto done; }
    srand(41);
    for (int i = 0; i < nwin * hd; i++) kw[i] = (float)((rand() % 2001) - 1000) / 500.0f * ldexpf(1.f, rand() % 7 - 3);
    for (int i = 0; i < ncmp * hd; i++) kc[i] = (float)((rand() % 2001) - 1000) / 500.0f * ldexpf(1.f, rand() % 7 - 3);
    if (nwin > 2) for (int i = 0; i < hd; i++) { kw[hd + i] = 0.0f; kw[2 * hd + i] = (i & 1) ? -448.0f : 448.0f; }
    if (ncmp > 1) for (int i = 0; i < hd; i++) kc[hd + i] = 0.0f;
    srand(47);
    for (int i = 0; i < S * H * hd; i++) qq[i] = (float)((rand() % 2001) - 1000) / 500.0f * ldexpf(1.f, rand() % 3 - 1);
    srand(53);
    for (int h = 0; h < H; h++) sk[h] = one ? -INFINITY : (float)((rand() % 9) - 4) * 0.5f;
    srand(59);
    for (int s = 0; s < S; s++)
        for (int j = 0; j < cnt; j++) {
            if (one) { li[(size_t)s * cnt + j] = j ? -1 : (int)(rand() % (unsigned)(nwin + ncmp)); continue; }
            if (s == 0 && j < nwin + ncmp) { li[j] = j; continue; }   /* every row of both, in a row */
            int r = rand() % 6;
            li[(size_t)s * cnt + j] = r == 0 ? -1 : r < 3 ? (int)(rand() % (unsigned)nwin) : nwin + (int)(rand() % (unsigned)ncmp);
        }
    for (int r = 0; r < nwin; r++) kvq_pack_win(pw + (size_t)r * KVQ_WIN_BYTES, kw + (size_t)r * hd);
    for (int r = 0; r < ncmp; r++) kvq_pack_ckv(pc + (size_t)r * KVQ_CKV_BYTES, kc + (size_t)r * hd);
    for (int r = 0; r < nwin; r++) kvq_unpack_win(kv + (size_t)r * hd, pw + (size_t)r * KVQ_WIN_BYTES);
    for (int r = 0; r < ncmp; r++) kvq_unpack_ckv(kv + (size_t)(nwin + r) * hd, pc + (size_t)r * KVQ_CKV_BYTES);
    for (int s = 0; s < S; s++)
        for (int h = 0; h < H; h++)
            attn_row(ref + ((size_t)s * H + h) * hd, qq + ((size_t)s * H + h) * hd, kv, li + (size_t)s * cnt, cnt, hd,
                     sk[h], 1.f / sqrtf((float)hd), sc);
    if (!(qd = upf(qq, allb)) || !(wd = upf(pw, wby)) || !(cd = upf(pc, cby)) ||
        !(wf = upf(kv, (size_t)nwin * hd * 4)) || !(cf = upf(kv + (size_t)nwin * hd, (size_t)ncmp * hd * 4)) ||
        !(ld = upf(li, (size_t)S * cnt * 4)) || !(sd = upf(sk, (size_t)H * 4)) ||
        !(op = vkc_buf(allb, VKC_DEV)) || !(of = vkc_buf(allb, VKC_DEV))) {
        CHECK(0, "%s: device memory refused", what); goto done;
    }
    VkcDsAttn p2 = {S, H, hd, cnt, 0, cnt, nwin, 0, 0, 0, H * hd, 0, H * hd, 0, 2, 1.f / sqrtf((float)hd)};
    VkcDsAttn p0 = p2; p0.flags = 0;   /* the same rows as fp32: w_off/c_off 0, hd the stride */
    vkc_begin();
    int ok = vkc_dsv4_attn(qd, wd, cd, ld, sd, op, &p2) && vkc_dsv4_attn(qd, wf, cf, ld, sd, of, &p0);
    vkc_submit(1);
    float *gp = grab(op, allb), *gf = grab(of, allb);
    int same = gp && gf && !memcmp(gp, gf, allb);
    double e = gp ? rel_l2(gp, ref, S * H * hd) : 1;
    CHECK(ok && gp && gf, "%s: the dispatches were refused", what);
    if (one) CHECK(same && gp && !memcmp(gp, ref, allb), "%s: a one-entry row is not its value row's bits", what);
    else {
        CHECK(same, "%s: the packed arm is not the fp32 arm's bits (the shader's decode is not dsv41_kvq.h's)", what);
        CHECK(e <= 2e-5, "%s: rel_l2 %.3e to the scalar attention over the decoded rows", what, e);
    }
    if (mut) {   /* one scale code, only where the packed arm reads it */
        VkcBuf *mw = wd, *mc = cd;   /* the mutated mirror, above the one the fp32 arm's rows came from */
        if (mut == 1 && nwin) { pw[512] ^= 1u; if (!(wm = upf(pw, wby))) goto mutfail; mw = wm; }
        else if (mut == 2 && ncmp) { pc[256] ^= 1u; if (!(cd2 = upf(pc, cby))) goto mutfail; mc = cd2; }
        else { CHECK(0, "%s: the mutation needs a row", what); goto mutdone; }
        if ((om = vkc_buf(allb, VKC_DEV))) {
            vkc_begin(); int ok2 = vkc_dsv4_attn(qd, mw, mc, ld, sd, om, &p2); vkc_submit(1);
            float *gm = grab(om, allb);
            int moved = gm && gp && memcmp(gm, gp, allb) != 0;
            CHECK(ok2 && moved, "%s: a one-code change in a %s scale moved nothing", what, mut == 1 ? "window" : "compressed");
            if (ok2 && moved) printf("  %s: a one-code change in a %s scale moves the packed arm\n", what,
                                     mut == 1 ? "window" : "compressed");
            free(gm);
            goto mutdone;
        }
    mutfail:
        CHECK(0, "%s: device memory refused (the mutation)", what);
    mutdone: ;
    }
    printf("OK kv-quant attn %s (S %d H %d nwin %d ncmp %d cnt %d): the packed arm's bits are the fp32 arm's, "
           "rel_l2 %.2e to the scalar attention over kvq_unpack's rows\n", what, S, H, nwin, ncmp, cnt, e);
    free(gp); free(gf);
    done:
    vkc_free(qd); vkc_free(wd); vkc_free(cd); vkc_free(wf); vkc_free(cf); vkc_free(ld); vkc_free(sd);
    vkc_free(op); vkc_free(of); vkc_free(wm); vkc_free(cd2); vkc_free(om);
    free(pw); free(pc); free(kw); free(kc); free(kv); free(qq); free(sk); free(ref); free(sc); free(li);
}

/* a row's top-k by repeated scan: the largest, then the next (ties to the lower column, as
 * vkc_dsv4_topk); a column at -inf is never picked and a short row leaves -1s */
static void topk_scan(const float *v, int width, int topk, int *row) {
    float *c = malloc((size_t)width * 4);
    if (!c) { for (int t = 0; t < topk; t++) row[t] = -1; return; }
    memcpy(c, v, (size_t)width * 4);
    for (int t = 0; t < topk; t++) {
        int b = -1;
        for (int j = 0; j < width; j++) if (b < 0 || c[j] > c[b]) b = j;
        if (b < 0 || c[b] == -INFINITY) { for (int u = t; u < topk; u++) row[u] = -1; break; }
        row[t] = b; c[b] = -INFINITY;
    }
    free(c);
}

/* the indexer's scores over the packed keys (vkc_dsv4_score's quant) against (a) the same
 * call over the rows kvq_unpack_ikey decodes from the mirror -- bit for bit -- and (b) a
 * scalar float sum in the shader's own order (a thread a column, its dot over i ascending,
 * the heads ascending with the weight w * wscale precomputed, score/score_wide/
 * score_grid), also bit for bit: the inputs are on a 0.25 grid with weights on 0.5, so
 * every partial sum is exact in float whatever the order or an fma contraction does.
 * The mode is the parameter's own (vkc_dsv4_score): width < 32 keeps mode 3 (IH <= 64 and
 * IH * ID <= 4096) or mode 9 (IH > 64, the queries from memory); width >= 32 with IH <= 256
 * takes mode 10's grid. width not a multiple of 32 leaves the mirror a partial last block.
 * Reported: the top-k overlap (the checkpoint's index_topk, 512, on a pool wider than it)
 * between the scores over the ORIGINAL fp32 rows and the packed keys' -- the format's own
 * selection cost, which a decoding fault cannot hide behind.
 * mut (VK_KV_QUANT_MUT=ikey): one scale byte of the mirror changes a code after the fp32
 * arm was packed from the original bytes; the packed arm must move. */
static void index_quant(const char *what, int S, int pos_base, int ratio, int IH, int width, int topk, int mut, int wide) {
    const int ID = 128, qrow = IH * ID, wrow = IH, scrow = width + 3;
    const float wscale = 0.25f;
    const size_t mb = kvq_ikey_mirror_bytes((size_t)width), sn = (size_t)S * scrow;
    float *q = malloc((size_t)S * qrow * 4), *w = malloc((size_t)S * wrow * 4);
    float *k = malloc((size_t)width * ID * 4), *kd = malloc((size_t)width * ID * 4);
    float *ref = calloc(sn, 4), *ref0 = calloc(sn, 4), *zero = calloc(sn, 4);
    uint8_t *km = calloc(mb, 1);
    int *t1 = malloc((size_t)topk * 4), *t0 = malloc((size_t)topk * 4);
    VkcBuf *qb = NULL, *wb = NULL, *kq = NULL, *kmq = NULL, *s1 = NULL, *s2 = NULL, *kmut = NULL, *sm2 = NULL;
    if (!q || !w || !k || !kd || !ref || !ref0 || !zero || !km || !t1 || !t0) { CHECK(0, "%s: out of memory", what); goto done; }
    srand(61);
    for (int i = 0; i < S * qrow; i++) q[i] = (float)((rand() % 9) - 4) * 0.25f;
    for (int i = 0; i < S * wrow; i++) w[i] = (float)((rand() % 9) - 4) * 0.5f;
    for (int i = 0; i < width * ID; i++)
        k[i] = wide ? (float)((rand() % 33) - 16) / 16.0f * (float)(1 << (rand() % 7))   /* past FP4's grid */
                    : (float)((rand() % 9) - 4) * 0.25f;
    /* a power of two a token, cycling: the 32 tokens of a block then carry 32 different
     * scale bytes, so a reader that takes a neighbouring token's scales (an off-by-one on
     * the 4 scale bytes at block + 2048 + 4t) cannot agree by coincidence -- equal tokens
     * would hide it. Every value stays a multiple of 1/16, so the sums below stay exact. */
    for (int t = 0; t < width; t++) {
        float f = ldexpf(1.f, t % 4 - 1);
        for (int i = 0; i < ID; i++) k[(size_t)t * ID + i] *= f;
    }
    for (int t = 0; t < width; t++) { kvq_pack_ikey(km, t, k + (size_t)t * ID); kvq_unpack_ikey(kd + (size_t)t * ID, km, t); }
    for (int s = 0; s < S; s++) {
        int lens = (pos_base + s + 1) / ratio;
        float wt[IH];
        for (int h = 0; h < IH; h++) wt[h] = w[(size_t)s * wrow + h] * wscale;
        for (int j = 0; j < width; j++) {
            float t1v = -INFINITY, t0v = -INFINITY;
            if (j < lens) {
                t1v = 0; t0v = 0;
                for (int h = 0; h < IH; h++) {
                    float d1 = 0, d0 = 0;
                    for (int i = 0; i < ID; i++) {
                        d1 += q[(size_t)s * qrow + h * ID + i] * kd[(size_t)j * ID + i];
                        d0 += q[(size_t)s * qrow + h * ID + i] * k[(size_t)j * ID + i];
                    }
                    if (d1 > 0) t1v += d1 * wt[h];
                    if (d0 > 0) t0v += d0 * wt[h];
                }
            }
            ref[(size_t)s * scrow + j] = t1v; ref0[(size_t)s * scrow + j] = t0v;
        }
    }
    int over = 0;
    double num0 = 0, den0 = 0;
    for (int s = 0; s < S; s++) {
        topk_scan(ref + (size_t)s * scrow, width, topk, t1);
        topk_scan(ref0 + (size_t)s * scrow, width, topk, t0);
        for (int a = 0; a < topk; a++) for (int b = 0; b < topk; b++) if (t1[a] >= 0 && t1[a] == t0[b]) { over++; break; }
        for (int j = 0; j < width; j++) {
            if (!isfinite(ref0[(size_t)s * scrow + j])) continue;
            double d = (double)ref[(size_t)s * scrow + j] - ref0[(size_t)s * scrow + j];
            num0 += d * d; den0 += (double)ref0[(size_t)s * scrow + j] * ref0[(size_t)s * scrow + j];
        }
    }
    double ov = (double)over / ((double)S * topk), rel0 = sqrt(num0 / (den0 + 1e-30));
    if (!(qb = upf(q, (size_t)S * qrow * 4)) || !(wb = upf(w, (size_t)S * wrow * 4)) ||
        !(kq = upf(kd, (size_t)width * ID * 4)) || !(kmq = upf(km, mb)) ||
        !(s1 = vkc_buf(sn * 4, VKC_DEV)) || !(s2 = vkc_buf(sn * 4, VKC_DEV))) {
        CHECK(0, "%s: device memory refused", what); goto done;
    }
    vkc_begin(); vkc_write(s1, 0, zero, sn * 4); vkc_write(s2, 0, zero, sn * 4); vkc_submit(1);
    VkcDsScore sp = {S, pos_base, ratio, IH, ID, width, 0, qrow, 0, wrow, 0, ID, 0, scrow, wscale, 0, 0, 0};
    VkcDsScore sq = sp; sq.quant = 1;
    vkc_begin();
    int ok = vkc_dsv4_score(qb, wb, kq, NULL, s1, &sp) && vkc_dsv4_score(qb, wb, kmq, NULL, s2, &sq);
    vkc_submit(1);
    float *g1 = grab(s1, sn * 4), *g2 = grab(s2, sn * 4);
    int bits = g1 && g2 && !memcmp(g1, g2, sn * 4), cbits = g1 && !memcmp(g1, ref, sn * 4);
    double rnum = 0, rden = 0;   /* the -inf columns are equal on both sides but not subtractable */
    if (g1) for (size_t i = 0; i < sn; i++) {
        if (!isfinite(ref[i])) { if (g1[i] != ref[i]) rnum = INFINITY; continue; }
        double d = (double)g1[i] - ref[i]; rnum += d * d; rden += (double)ref[i] * ref[i];
    }
    double relc = sqrt(rnum / (rden + 1e-30));
    CHECK(ok && g1 && g2, "%s: the dispatches were refused", what);
    CHECK(bits, "%s: the packed keys' scores are not the fp32 rows' bits", what);
    /* exact grids: bit for bit; the wide rows' sums need more bits than float has at every
     * partial, so there an order or an fma contraction may round the last place */
    if (wide) CHECK(relc <= 1e-6, "%s: rel_l2 %.3e to the scalar sum over kvq_unpack_ikey's rows", what, relc);
    else CHECK(cbits, "%s: the scores are not the scalar sum over kvq_unpack_ikey's rows, bit for bit", what);
    CHECK(ov >= 0.80, "%s: the top-%d overlap %.1f%% is below 80%%", what, topk, ov * 100);
    if (mut == 3) {
        km[2048] ^= 1u;   /* token 0's first group scale, a code up */
        if ((kmut = upf(km, mb)) && (sm2 = vkc_buf(sn * 4, VKC_DEV))) {
            vkc_begin(); int ok2 = vkc_dsv4_score(qb, wb, kmut, NULL, sm2, &sq); vkc_submit(1);
            float *gm = grab(sm2, sn * 4);
            int moved = gm && g2 && memcmp(gm, g2, sn * 4) != 0;
            CHECK(ok2 && moved, "%s: a one-code change in an index-key scale moved nothing", what);
            if (ok2 && moved) printf("  %s: a one-code change in an index-key scale moves the packed scores\n", what);
            free(gm);
        } else CHECK(0, "%s: device memory refused (the mutation)", what);
    }
    printf("OK kv-quant index %s (S %d pos %d ratio %d IH %d width %d topk %d): the packed keys' scores are the fp32 "
           "rows' bits%s, rel_l2 %.2e to the scalar sum over kvq_unpack_ikey's rows, top-%d overlap %.1f%% and rel_l2 "
           "%.2e against the original rows' scores\n",
           what, S, pos_base, ratio, IH, width, topk, cbits ? " and the scalar sum's" : "", relc, topk, ov * 100, rel0);
    free(g1); free(g2);
    done:
    vkc_free(qb); vkc_free(wb); vkc_free(kq); vkc_free(kmq); vkc_free(s1); vkc_free(s2); vkc_free(kmut); vkc_free(sm2);
    free(q); free(w); free(k); free(kd); free(ref); free(ref0); free(zero); free(km); free(t1); free(t0);
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);   /* a crash's prints must not sit in the buffer */
    host_block_tests();   /* no device needed -- and its failures are not a skip */
    if (!coli_vk_init("shaders/chain_dsv4.spv") || !coli_vk_available()) return skip("no Vulkan device");
    if (!vkc_init() || !vkc_dsv4_ready()) {
        coli_vk_shutdown();
        return skip("the chain_dsv4 shader is missing");
    }
    codec("win (the ring scattered)", 0, 1, 24, 512, KVQ_WIN_BYTES, 24, 13, kvq_pack_win, kvq_unpack_win);
    codec("ckv (from token 7)", 1, 0, 24, 512, KVQ_CKV_BYTES, 31, 7, kvq_pack_ckv, kvq_unpack_ckv);
    codec_ikey();
    score_case(3, 512, 8, 16);      /* a wide table: mode 10's grid arm (dsa_grid default-on
                                     * takes IH<=256 there; mode 9's quantized reader is the
                                     * same expression, exercised through mode 10 here) */
    score_case(2, 64, 8, 16);       /* a narrow one: mode 3 -- the only arm below VKC_DSA_GRID */
    score_case(2, 512, 65, 16);     /* the wide queries: mode 10 */
    score_case(2, 8190, 8, 16);     /* a pool ABOVE VKC_DSA_GRID (2048 keys, top-16: the
                                     * selection can move) */
    score_case(2, 32766, 8, 512);   /* the real geometry: 8192 keys, top-512 (index_topk,
                                     * config.json:124) -- k/pool 6.25%, as on the checkpoint */
    {   /* mode 0's packed read and the indexer's, which test_vk_chain.c never takes: its
         * test_ds_attn runs flags 0 and 1 only (no packed cache) and its VkcDsScore
         * initializers leave quant 0, so the decode-on-read paths ship with no test that
         * could fail. VK_KV_QUANT_MUT=win|ckv|ikey: the same run with one scale code
         * changed under the packed arm, which must part from the others. */
        const char *m = getenv("VK_KV_QUANT_MUT");
        int mut = !m || !*m ? 0 : !strcmp(m, "win") ? 1 : !strcmp(m, "ckv") ? 2 : !strcmp(m, "ikey") ? 3 : -1;
        CHECK(mut >= 0, "VK_KV_QUANT_MUT=%s: not win, ckv or ikey", m);
        attn_packed("mixed rows, a window ring of exactly nwin", 2, 3, 8, 5, 16, 0, mut == 1 || mut == 2 ? mut : 0);
        attn_packed("V4.1-Flash's 64 heads of 512", 2, 64, 8, 5, 13, 0, 0);
        attn_packed("a list past the kernel's 256 threads", 2, 4, 8, 40, 300, 0, 0);
        attn_packed("one entry a row and the sink -inf: the value's bits", 2, 2, 1, 1, 3, 1, 0);
        index_quant("mode 3 (the default, 23 keys: a partial last block)", 1, 88, 4, 32, 23, 16, mut == 3 ? 3 : 0, 0);
        index_quant("mode 9 (the wide queries, 65 heads)", 1, 92, 4, 65, 23, 16, 0, 0);
        index_quant("mode 10 (the grid, 1005 keys)", 2, 4016, 4, 32, 1005, 512, 0, 0);
        index_quant("mode 10 (the grid, 2048 keys, index_topk 512)", 1, 8191, 4, 32, 2048, 512, 0, 0);
        index_quant("mode 10 (the grid, 8 heads, 64 keys)", 3, 252, 4, 8, 64, 16, 0, 0);
        index_quant("mode 10 (the grid, 2048 keys off FP4's grid)", 1, 8191, 4, 32, 2048, 512, 0, 1);
    }
    if (fails) { printf("FAIL: kv-quant (%d)\n", fails); coli_vk_shutdown(); return 1; }
    printf("PASS\n");   /* the plain word, as the gates grep for */
    coli_vk_shutdown();
    return 0;
}
