/* The routed-expert tier on a CUDA device (cu_tier.c: vk_tier.c over cu_xb.cu) against a CPU
 * reference: the sections of tests/test_vk_tier.c the CUDA backend serves, the same synthetic
 * MoE (L layers of E experts, top-K) and the same decoders, written apart from the tier and the
 * backend. Gates:
 *   formats  every VktSrc kind and the three activations: each row the device returns equals the
 *            reference expert, the rank-order sum of a step the all-CPU sum, within 2e-3;
 *   blocks   a GGUF's block types (GQ_EXACT=1: their float form), the same;
 *   adapt    a third of the experts fit and the hot set moves halfway: evictions, the new hot
 *            set resident, the budget held, every row the reference's;
 *   v4       DeepSeek V4's expert with its roundings, against the engine's CPU arithmetic;
 *   host     the host pool (mapped page-locked RAM the card reads over the bus), four experts on
 *            the device: the CPU computes the pool's pairs of small steps in place (the
 *            hybrid), the card reads the rest; with COLI_VK_HYBRID=0 the card reads them all;
 *            every row the reference's.
 *   make tests/test_cu_tier CUDA=1 && ./tests/test_cu_tier */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include "../backend_cuda.h"
#include "../cu_tier.h"
#include "../ggml_quants.h"   /* the block formats' CPU decoders, the reference of section blocks */

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fails++; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

enum { L = 2, E = 10, K = 3, HF_MAX = 4096 };
static int H = 192, F = 128;   /* hidden and expert width; grouped() widens H */
static unsigned rng = 7;
static unsigned rnd(void) { rng = rng * 1103515245u + 12345u; return rng >> 8; }
static float frnd(void) { return (float)((int)(rnd() % 2001) - 1000) / 1000.0f; }

/* ---- one matrix in a source format, and its own decode ------------------------- */
typedef struct { VktFmt f; int I, O; uint8_t *codes; void *scales; float *w; } Mat;
static size_t row_bytes(VktSrc k, int I) {
    switch (k) {
    case VKT_SRC_I8_ROW: case VKT_SRC_I8_GS: case VKT_SRC_I8_AS_I4_ROW: case VKT_SRC_I8_AS_I4_GS:
    case VKT_SRC_FP8_GS: case VKT_SRC_FP8_BLOCK: return (size_t)I;
    case VKT_SRC_I3_G64: return ((size_t)I + 63) / 64 * 24;
    case VKT_SRC_BF16: return (size_t)I * 2;
    case VKT_SRC_F32: return (size_t)I * 4;
    default: return ((size_t)I + 1) / 2;
    }
}
static float e4m3(uint8_t b) {
    int e = (b >> 3) & 15, m = b & 7;
    float v = e == 0 ? m / 512.0f : ldexpf(1.0f + m / 8.0f, e - 7);
    return (b & 0x80) ? -v : v;
}
static int nib(const uint8_t *row, int i) { return (row[i >> 1] >> ((i & 1) * 4)) & 15; }
/* the weight value (o, i) before scales, as the format defines it */
static float code(const Mat *m, int o, int i) {
    const uint8_t *row = m->codes + (size_t)o * row_bytes(m->f.kind, m->I);
    switch (m->f.kind) {
    case VKT_SRC_I8_ROW: case VKT_SRC_I8_GS: case VKT_SRC_I8_AS_I4_ROW: case VKT_SRC_I8_AS_I4_GS:
        return (float)(int8_t)row[i];
    case VKT_SRC_I4S_PAIRS_ROW: case VKT_SRC_I4S_PAIRS_GS: { int v = nib(row, i); return (float)(v >= 8 ? v - 16 : v); }
    case VKT_SRC_I4U_PAIRS_ROW: case VKT_SRC_I4U_PAIRS_GS: return (float)(nib(row, i) - 8);
    case VKT_SRC_I4U_PLANAR64: {
        int full = m->I / 64 * 64;
        if (i >= full) return (float)(nib(row + full / 2, i - full) - 8);
        const uint8_t *blk = row + (size_t)(i / 64) * 32; int k = i % 64;
        return (float)((k < 32 ? (blk[k] & 15) : (blk[k - 32] >> 4)) - 8);
    }
    case VKT_SRC_I3_G64: {
        const uint8_t *lo = row + (size_t)(i / 64) * 24, *hi = lo + 16; int j = i % 64;
        return (float)((int)(((lo[j >> 2] >> ((j & 3) * 2)) & 3u) | (((hi[j >> 3] >> (j & 7)) & 1u) << 2)) - 4);
    }
    case VKT_SRC_MXFP4_F32: case VKT_SRC_MXFP4_E8M0: {
        static const float lut[8] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
        int v = nib(row, i); return (v & 8) ? -lut[v & 7] : lut[v & 7];
    }
    case VKT_SRC_FP8_GS: case VKT_SRC_FP8_BLOCK: return e4m3(row[i]);
    case VKT_SRC_BF16: { uint16_t h; memcpy(&h, row + 2 * i, 2); uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f; }
    case VKT_SRC_F32: { float f; memcpy(&f, row + 4 * i, 4); return f; }
    default: return 0;
    }
}
static int per_row(VktSrc k) { return k == VKT_SRC_I8_ROW || k == VKT_SRC_I8_AS_I4_ROW || k == VKT_SRC_I4S_PAIRS_ROW || k == VKT_SRC_I4U_PAIRS_ROW; }
static float scale(const Mat *m, int o, int i) {
    VktSrc k = m->f.kind;
    if (k == VKT_SRC_BF16 || k == VKT_SRC_F32) return 1.0f;
    if (per_row(k)) return ((const float *)m->scales)[o];
    int gs = k == VKT_SRC_I4U_PLANAR64 || k == VKT_SRC_I3_G64 ? 64 : m->f.gs, ng = (m->I + gs - 1) / gs;
    if (k == VKT_SRC_MXFP4_E8M0) return ldexpf(1.0f, ((const uint8_t *)m->scales)[(size_t)o * ng + i / gs] - 127);
    if (k == VKT_SRC_FP8_BLOCK) return ((const float *)m->scales)[(size_t)(o / gs) * ng + i / gs];
    return ((const float *)m->scales)[(size_t)o * ng + i / gs];
}
static void mat_make(Mat *m, VktFmt f, int I, int O) {
    m->f = f; m->I = I; m->O = O;
    size_t rb = row_bytes(f.kind, I);
    m->codes = malloc(rb * O);
    for (size_t i = 0; i < rb * O; i++) m->codes[i] = (uint8_t)rnd();
    if (f.kind == VKT_SRC_I8_AS_I4_ROW || f.kind == VKT_SRC_I8_AS_I4_GS)   /* an unpacked int4: [-8, 7] */
        for (size_t i = 0; i < rb * O; i++) m->codes[i] = (uint8_t)(int8_t)((int)(m->codes[i] % 16) - 8);
    if (f.kind == VKT_SRC_FP8_GS || f.kind == VKT_SRC_FP8_BLOCK)
        for (size_t i = 0; i < rb * O; i++) if ((m->codes[i] & 0x7f) == 0x7f) m->codes[i] ^= 1;
    if (f.kind == VKT_SRC_BF16 || f.kind == VKT_SRC_F32)
        for (int i = 0; i < I * O; i++) {
            float v = frnd() * 0.1f;
            if (f.kind == VKT_SRC_F32) memcpy(m->codes + 4 * (size_t)i, &v, 4);
            else { uint32_t u; memcpy(&u, &v, 4); uint16_t h = (uint16_t)(u >> 16); memcpy(m->codes + 2 * (size_t)i, &h, 2); }
        }
    int gs = f.kind == VKT_SRC_I4U_PLANAR64 || f.kind == VKT_SRC_I3_G64 ? 64 : f.gs, ng = gs ? (I + gs - 1) / gs : 1;
    size_t ns = per_row(f.kind) ? (size_t)O : f.kind == VKT_SRC_FP8_BLOCK ? (size_t)((O + gs - 1) / gs) * ng : (size_t)O * ng;
    if (f.kind == VKT_SRC_MXFP4_E8M0) {
        uint8_t *s = malloc(ns); for (size_t i = 0; i < ns; i++) s[i] = (uint8_t)(120 + rnd() % 4); m->scales = s;
    } else if (f.kind == VKT_SRC_BF16 || f.kind == VKT_SRC_F32) m->scales = NULL;
    else { float *s = malloc(ns * 4); for (size_t i = 0; i < ns; i++) s[i] = 0.004f + (rnd() % 100) / 25000.0f; m->scales = s; }
    m->w = malloc((size_t)I * O * sizeof(float));
    for (int o = 0; o < O; o++) for (int i = 0; i < I; i++) m->w[(size_t)o * I + i] = code(m, o, i) * scale(m, o, i);
}
static void mat_free(Mat *m) { free(m->codes); free(m->scales); free(m->w); }

/* ---- the synthetic model ---------------------------------------------------------- */
typedef struct { Mat g, u, d; } Ex;
static Ex ex[L][E];
static int g_act; static float g_limit;
static void expert_ref(const Ex *e, const float *x, float *y) {
    float h[HF_MAX];
    for (int o = 0; o < F; o++) {
        double a = 0, b = 0;
        for (int i = 0; i < H; i++) { a += (double)e->g.w[(size_t)o * H + i] * x[i]; b += (double)e->u.w[(size_t)o * H + i] * x[i]; }
        float g = (float)a, u = (float)b;
        if (g_act == VKT_ACT_SITU) h[o] = 4.f * tanhf(g / 4.f) * (1.f / (1.f + expf(-g))) * (25.f * tanhf(u / 25.f));
        else {
            if (g_limit > 0) { if (g > g_limit) g = g_limit; if (u > g_limit) u = g_limit; if (u < -g_limit) u = -g_limit; }
            h[o] = g / (1.f + expf(-g)) * u;
        }
    }
    for (int o = 0; o < H; o++) { double a = 0; for (int i = 0; i < F; i++) a += (double)e->d.w[(size_t)o * F + i] * h[i]; y[o] = (float)a; }
}
static VktExpertSrc src_of(const Ex *e) {
    return (VktExpertSrc){e->g.codes, e->u.codes, e->d.codes, e->g.scales, e->u.scales, e->d.scales};
}
static void layer_make(int l, VktFmt gu, VktFmt dn) {
    for (int e = 0; e < E; e++) { mat_make(&ex[l][e].g, gu, H, F); mat_make(&ex[l][e].u, gu, H, F); mat_make(&ex[l][e].d, dn, F, H); }
}
static void model_make(VktFmt gu, VktFmt dn) { for (int l = 0; l < L; l++) layer_make(l, gu, dn); }
static void model_free(void) {
    for (int l = 0; l < L; l++) for (int e = 0; e < E; e++) { mat_free(&ex[l][e].g); mat_free(&ex[l][e].u); mat_free(&ex[l][e].d); }
}
static double rel(const float *a, const float *b, int n) {
    double num = 0, den = 0;
    for (int i = 0; i < n; i++) { double d = (double)a[i] - b[i]; num += d * d; den += (double)b[i] * b[i]; }
    return den > 0 ? sqrt(num / den) : sqrt(num);
}

/* One layer step as an engine runs it (vk_tier.h step 3): issue, CPU pairs with a
 * note each, join, rank-order sum. Returns the worst relative error of a device row
 * against the reference and of the sum against the all-CPU sum. */
typedef struct { unsigned long long routed, dev; } Books;
static double step(int layer, int S, const int *idx, const float *w, Books *bk) {
    float *x = calloc((size_t)S * H, sizeof(float)), *cpu = calloc((size_t)S * K * H, sizeof(float));
    float *out = calloc((size_t)S * H, sizeof(float)), *ref = calloc((size_t)S * H, sizeof(float)), *y = malloc(sizeof(float) * H);
    uint8_t *taken = malloc((size_t)S * K); const float **dev = malloc(sizeof(*dev) * S * K);
    for (int i = 0; i < S * H; i++) x[i] = frnd();
    int n = cut_issue(layer, x, S, K, idx, taken);
    for (int i = 0; i < S * K; i++) {
        if (taken[i]) continue;
        expert_ref(&ex[layer][idx[i]], x + (size_t)(i / K) * H, cpu + (size_t)i * H);
        VktExpertSrc s = src_of(&ex[layer][idx[i]]); cut_note(layer, idx[i], &s);
    }
    int joined = n ? cut_join(dev) : 1;
    CHECK(joined, "join failed");
    double worst = 0;
    for (int s = 0; s < S; s++)
        for (int k = 0; k < K; k++) {
            int i = s * K + k;
            expert_ref(&ex[layer][idx[i]], x + (size_t)s * H, y);
            for (int d = 0; d < H; d++) ref[(size_t)s * H + d] += w[i] * y[d];
            const float *c = taken[i] && joined ? dev[i] : cpu + (size_t)i * H;
            if (taken[i] && joined) { double r = rel(c, y, H); if (!(r <= worst)) worst = r; }   /* a NaN counts as the worst */
            for (int d = 0; d < H; d++) out[(size_t)s * H + d] += w[i] * c[d];
            bk->routed++; bk->dev += taken[i] != 0;
        }
    double r = rel(out, ref, S * H);
    if (!(r <= worst)) worst = r;
    if (getenv("VKT_TEST_DUMP") && n) fprintf(stderr, "dump: ref %g %g dev %g %g\n", ref[0], ref[1], out[0], out[1]);
    free(x); free(cpu); free(out); free(ref); free(y); free(taken); free(dev);
    return worst;
}
/* K distinct experts per row from a skewed draw over [lo, lo+span) */
static void route(int S, int lo, int span, int *idx, float *w) {
    for (int s = 0; s < S; s++)
        for (int k = 0; k < K; k++) {
            int e, dup;
            do { unsigned r = rnd() % 100; e = lo + (r < 60 ? (int)(r % 2) : (int)(r % span)) % span; e %= E;
                 dup = 0; for (int j = 0; j < k; j++) dup |= idx[s * K + j] == e; } while (dup);
            idx[s * K + k] = e; w[s * K + k] = 0.2f + (rnd() % 100) / 200.0f;
        }
}

static VktConfig cfg_of(VktFmt gu, VktFmt dn, int act, float limit) {
    return (VktConfig){.engine = "test", .layers = L, .experts = E, .hidden = H, .inter = F, .topk = K,
                       .gate_up = gu, .down = dn, .act = act, .act_limit = limit, .act_a = 4.f, .act_b = 25.f,
                       .max_rows = 1 << 20};
}
static void set_budget(int experts, VktFmt gu, VktFmt dn) {
    char b[64]; snprintf(b, sizeof b, "%.12f", (experts + 0.5) * cut_expert_bytes(H, F, gu, dn) / 1073741824.0);
    setenv("COLI_VK_TIER_GB", b, 1);
}


/* every format: the whole model fits, experts are promoted as they pass by */
static void formats(void) {
    struct { VktFmt gu, dn; int act; float limit; const char *name; } cs[] = {
        {{VKT_SRC_I8_ROW, 0}, {VKT_SRC_I8_ROW, 0}, VKT_ACT_SWIGLU, 0, "int8 per row"},
        {{VKT_SRC_I8_GS, 32}, {VKT_SRC_I8_GS, 32}, VKT_ACT_SWIGLU, 0, "int8 gs32"},
        {{VKT_SRC_I8_AS_I4_ROW, 0}, {VKT_SRC_I8_AS_I4_ROW, 0}, VKT_ACT_SWIGLU, 0, "unpacked int4 per row"},
        {{VKT_SRC_I8_AS_I4_GS, 64}, {VKT_SRC_I8_GS, 32}, VKT_ACT_SWIGLU, 0, "unpacked int4 gs64, int8 down"},
        {{VKT_SRC_I4S_PAIRS_ROW, 0}, {VKT_SRC_I4S_PAIRS_ROW, 0}, VKT_ACT_SWIGLU, 0, "signed int4 pairs per row"},
        {{VKT_SRC_I4S_PAIRS_GS, 32}, {VKT_SRC_I4S_PAIRS_GS, 32}, VKT_ACT_SWIGLU, 0, "signed int4 pairs gs32"},
        {{VKT_SRC_I4U_PAIRS_ROW, 0}, {VKT_SRC_I4U_PAIRS_ROW, 0}, VKT_ACT_SWIGLU, 0, "int4 v+8 per row"},
        {{VKT_SRC_I4U_PAIRS_GS, 64}, {VKT_SRC_I4U_PAIRS_GS, 64}, VKT_ACT_SWIGLU, 2.5f, "int4 v+8 gs64, SwiGLU limit"},
        {{VKT_SRC_I4U_PLANAR64, 64}, {VKT_SRC_I4U_PLANAR64, 64}, VKT_ACT_SWIGLU, 0, "planar int4-g64"},
        {{VKT_SRC_I3_G64, 0}, {VKT_SRC_I3_G64, 0}, VKT_ACT_SWIGLU, 0, "int3-g64"},
        {{VKT_SRC_MXFP4_F32, 32}, {VKT_SRC_MXFP4_F32, 32}, VKT_ACT_SWIGLU, 0, "MXFP4 f32 scales"},
        {{VKT_SRC_MXFP4_E8M0, 32}, {VKT_SRC_MXFP4_E8M0, 32}, VKT_ACT_SITU, 0, "MXFP4 ue8m0, SiTU-GLU"},
        {{VKT_SRC_FP8_GS, 32}, {VKT_SRC_FP8_GS, 32}, VKT_ACT_SWIGLU, 0, "fp8 gs32"},
        {{VKT_SRC_FP8_BLOCK, 128}, {VKT_SRC_FP8_BLOCK, 128}, VKT_ACT_SWIGLU, 0, "fp8 128x128 blocks"},
        {{VKT_SRC_BF16, 0}, {VKT_SRC_BF16, 0}, VKT_ACT_SWIGLU, 0, "bf16"},
        {{VKT_SRC_F32, 0}, {VKT_SRC_F32, 0}, VKT_ACT_SWIGLU, 0, "f32"},
    };
    for (size_t c = 0; c < sizeof cs / sizeof *cs; c++) {
        model_make(cs[c].gu, cs[c].dn);
        g_act = cs[c].act; g_limit = cs[c].limit;
        set_budget(L * E, cs[c].gu, cs[c].dn);
        setenv("COLI_VK_TIER_RATE", "64", 1);
        /* each step waits for the uploads staged so far: the numbers are this section's
         * point, and twelve tiny steps can end before the first upload lands (with
         * staged uploads a commit is a submit and a fence wait: 8.5 ms for the first,
         * 0.6 ms after, on an Iris Xe through Dozen); warm, adapt and partial keep the
         * uploader running free */
        setenv("COLI_VK_TIER_SYNC", "1", 1);
        VktConfig vc = cfg_of(cs[c].gu, cs[c].dn, cs[c].act, cs[c].limit);
        int on = cut_init(&vc, NULL);
        unsetenv("COLI_VK_TIER_SYNC");
        CHECK(on, "%s: the tier did not start", cs[c].name);
        if (!on) { model_free(); continue; }
        Books bk = {0, 0}; double worst = 0;
        static const int Ss[] = {1, 3, 20, 1, 2, 24};
        for (int t = 0; t < 6; t++)
            for (int l = 0; l < L; l++) {
                int idx[24 * K]; float w[24 * K];
                route(Ss[t], 0, E, idx, w);
                double r = step(l, Ss[t], idx, w, &bk);
                if (r > worst) worst = r;
            }
        printf("  %-34s device %3llu of %3llu assignments, worst relative error %.2e\n", cs[c].name, bk.dev, bk.routed, worst);
        CHECK(bk.dev > 0, "%s: nothing ran on the device", cs[c].name);
        CHECK(worst < 2e-3, "%s: device rows off the reference (%.3g)", cs[c].name, worst);
        cut_shutdown(); model_free();
    }
}

/* ---- block formats (shaders/blockdec.glsl): ggml block types and colibri's low-bit ---- */
/* a block's scale fields set to sane values over random bytes (the decoders read f16 /
 * e8m0 / ue4m3 scales at these places) */
static void block_fix(int t, uint8_t *b) {
    const uint8_t h0 = 0x00, h1 = 0x24;   /* f16 0x2400 = 2^-6 */
    #define F16AT(o) do { b[o] = h0; b[(o) + 1] = (uint8_t)(h1 + rnd() % 3); } while (0)
    switch (t) {
    case GQ_Q4_1: case GQ_Q5_1: case GQ_Q4_K: case GQ_Q5_K: F16AT(0); F16AT(2); break;
    case GQ_Q2_K: F16AT(80); F16AT(82); break;
    case GQ_Q3_K: F16AT(108); break;
    case GQ_Q6_K: F16AT(208); break;
    case GQ_TQ1_0: F16AT(52); break;
    case GQ_TQ2_0: F16AT(64); break;
    case GQ_MXFP4: b[0] = (uint8_t)(118 + rnd() % 4); break;
    case GQ_NVFP4: for (int i = 0; i < 4; i++) b[i] = (uint8_t)(0x28 + rnd() % 8); break;
    case GQ_IQ1_M: {   /* d is the top nibble of the four u16 scale words at 48..54 */
        uint16_t d = (uint16_t)(0x2400 + rnd() % 0x300);
        for (int i = 0; i < 4; i++) b[48 + 2 * i + 1] = (uint8_t)((b[48 + 2 * i + 1] & 0x0f) | (((d >> (4 * i)) & 15) << 4));
        break; }
    default: F16AT(0); break;
    }
    #undef F16AT
}

static void mat_make_block(Mat *m, VktFmt f, int I, int O) {
    m->f = f; m->I = I; m->O = O;
    m->w = malloc((size_t)I * O * sizeof(float));
    if (f.kind >= VKT_SRC_GGML) {
        int t = (int)(f.kind - VKT_SRC_GGML); size_t rb = (size_t)gq_row_bytes(t, I);
        int bl = GQ_TYPES[t].blck, by = GQ_TYPES[t].bytes;
        m->codes = malloc(rb * O); m->scales = NULL;
        for (size_t i = 0; i < rb * O; i++) m->codes[i] = (uint8_t)rnd();
        for (int o = 0; o < O; o++) {
            uint8_t *row = m->codes + (size_t)o * rb;
            for (int b = 0; b < I / bl; b++) block_fix(t, row + (size_t)b * by);
            gq_dequant_row(t, row, m->w + (size_t)o * I, I);
        }
        return;
    }
    /* colibri's low-bit forms (fmt 200-203) have no CUDA form yet */
    memset(m->w, 0, (size_t)I * O * sizeof(float)); m->codes = NULL; m->scales = NULL;
}
static void blocks(void) {
    const int H0 = H, F0 = F;
    H = 512; F = 256;   /* whole blocks of 256 in both widths, and of 128 for the low-bit forms */
    struct { int gu, dn; const char *name; } cs[] = {
        {GQ_Q4_0, GQ_Q4_0, "Q4_0"}, {GQ_Q4_1, GQ_Q4_1, "Q4_1"}, {GQ_Q5_0, GQ_Q5_0, "Q5_0"}, {GQ_Q5_1, GQ_Q5_1, "Q5_1"},
        {GQ_Q8_0, GQ_Q8_0, "Q8_0"}, {GQ_IQ4_NL, GQ_IQ4_NL, "IQ4_NL"}, {GQ_MXFP4, GQ_MXFP4, "MXFP4 (ggml)"},
        {GQ_NVFP4, GQ_NVFP4, "NVFP4"}, {GQ_Q1_0, GQ_Q1_0, "Q1_0"}, {GQ_Q2_0, GQ_Q2_0, "Q2_0"},
        {GQ_Q2_K, GQ_Q2_K, "Q2_K"}, {GQ_Q3_K, GQ_Q3_K, "Q3_K"}, {GQ_Q4_K, GQ_Q6_K, "Q4_K, Q6_K down"},
        {GQ_Q5_K, GQ_Q5_K, "Q5_K"}, {GQ_IQ2_XXS, GQ_IQ2_XXS, "IQ2_XXS"}, {GQ_IQ2_XS, GQ_IQ2_XS, "IQ2_XS"},
        {GQ_IQ2_S, GQ_IQ2_S, "IQ2_S"}, {GQ_IQ3_XXS, GQ_IQ3_XXS, "IQ3_XXS"}, {GQ_IQ3_S, GQ_IQ3_S, "IQ3_S"},
        {GQ_IQ1_S, GQ_IQ1_S, "IQ1_S"}, {GQ_IQ1_M, GQ_IQ1_M, "IQ1_M"}, {GQ_IQ4_XS, GQ_IQ4_XS, "IQ4_XS"},
        {GQ_TQ1_0, GQ_TQ1_0, "TQ1_0"}, {GQ_TQ2_0, GQ_TQ2_0, "TQ2_0"},
    };
    for (size_t c = 0; c < sizeof cs / sizeof *cs; c++) {
        VktFmt gu, dn;
        if (cs[c].gu >= 0) { gu.kind = (VktSrc)(VKT_SRC_GGML + cs[c].gu); dn.kind = (VktSrc)(VKT_SRC_GGML + cs[c].dn); }
        else { VktSrc k = cs[c].gu == -1 ? VKT_SRC_LB_I2G32 : cs[c].gu == -2 ? VKT_SRC_LB_I2G16 : cs[c].gu == -3 ? VKT_SRC_LB_I3G16 : VKT_SRC_LB_VQ2; gu.kind = dn.kind = k; }
        gu.gs = dn.gs = 0;
        for (int l = 0; l < L; l++) for (int e = 0; e < E; e++) {
            mat_make_block(&ex[l][e].g, gu, H, F); mat_make_block(&ex[l][e].u, gu, H, F); mat_make_block(&ex[l][e].d, dn, F, H);
        }
        g_act = VKT_ACT_SWIGLU; g_limit = 0;
        set_budget(L * E, gu, dn);
        setenv("COLI_VK_TIER_RATE", "64", 1);
        setenv("COLI_VK_TIER_SYNC", "1", 1);
        VktConfig vc = cfg_of(gu, dn, VKT_ACT_SWIGLU, 0);
        int on = cut_init(&vc, NULL);
        unsetenv("COLI_VK_TIER_SYNC");
        CHECK(on, "%s: the tier did not start", cs[c].name);
        if (!on) { model_free(); continue; }
        Books bk = {0, 0}; double worst = 0;
        static const int Ss[] = {1, 3, 20, 1, 2, 24};
        for (int t = 0; t < 6; t++)
            for (int l = 0; l < L; l++) {
                int idx[24 * K]; float w[24 * K];
                route(Ss[t], 0, E, idx, w);
                double r = step(l, Ss[t], idx, w, &bk);
                if (r > worst) worst = r;
            }
        printf("  %-34s device %3llu of %3llu assignments, worst relative error %.2e\n", cs[c].name, bk.dev, bk.routed, worst);
        CHECK(bk.dev > 0, "%s: nothing ran on the device", cs[c].name);
        CHECK(worst < 2e-3, "%s: device rows off the reference (%.3g)", cs[c].name, worst);
        cut_shutdown(); model_free();
    }
    H = H0; F = F0;
}


/* a third of the experts fit; the hot set moves halfway */
static void adapt(void) {
    VktFmt f = {VKT_SRC_FP8_GS, 32};
    model_make(f, f); g_act = VKT_ACT_SWIGLU; g_limit = 0;
    set_budget(L * E / 3, f, f);
    setenv("COLI_VK_TIER_RATE", "4", 1);
    VktConfig vc = cfg_of(f, f, VKT_ACT_SWIGLU, 0);
    CHECK(cut_init(&vc, NULL), "adapt: the tier did not start");
    Books a = {0, 0}, b1 = {0, 0}, b2 = {0, 0}; double worst = 0;
    int idx[4 * K]; float w[4 * K];
    /* LFRU: frequency first, halved every 1024 tokens, so an established hot set
     * yields over a few hundred steps of a new one, not at its first appearance */
    for (int t = 0; t < 100; t++)                     /* phase A: experts 0..4 */
        for (int l = 0; l < L; l++) { route(4, 0, 5, idx, w); double r = step(l, 4, idx, w, &a); if (r > worst) worst = r; }
    int resA = 0; for (int l = 0; l < L; l++) for (int e = 0; e < 5; e++) resA += cut_resident(l, e);
    for (int t = 0; t < 400; t++)                     /* phase B: experts 5..9 */
        for (int l = 0; l < L; l++) { route(4, 5, 5, idx, w); double r = step(l, 4, idx, w, t < 200 ? &b1 : &b2); if (r > worst) worst = r; }
    int resB = 0, resA2 = 0, res = 0;
    for (int l = 0; l < L; l++) for (int e = 0; e < E; e++) { int v = cut_resident(l, e); res += v; if (e >= 5) resB += v; else resA2 += v; }
    printf("  adapt: phase A device %.0f%%, resident A %d; phase B device %.0f%% then %.0f%%, resident B %d, A %d, all %d (budget %d)\n",
           100.0 * a.dev / a.routed, resA, 100.0 * b1.dev / b1.routed, 100.0 * b2.dev / b2.routed, resB, resA2, res, L * E / 3);
    cut_report("test", 0, 0);
    CHECK(resA > 0 && a.dev > 0, "adapt: phase A promoted nothing");
    CHECK(resB > resA2, "adapt: the new hot set did not take the device (B %d, A %d)", resB, resA2);
    CHECK(res <= L * E / 3, "adapt: %d resident over a budget of %d", res, L * E / 3);
    CHECK(b2.dev * b1.routed > b1.dev * b2.routed, "adapt: the device share did not rise in phase B");
    CHECK(worst < 2e-3, "adapt: relative error %.3g", worst);
    cut_shutdown(); model_free();
}

/* DeepSeek V4's activation (VKT_ACT_SWIGLU_V4) through the tier: MXFP4 ue8m0 experts,
 * x rounded to E4M3 per 128 here as the engine rounds it, the route weight applied on
 * the device (cut_issue_w), each device row rounded to bf16 and added with no weight
 * of its own. Every row against the engine's CPU arithmetic written out here (bf16
 * gate and up, the clamped SwiGLU, weight, bf16, E4M3 per 128, down, bf16): the two
 * differ only where their summation orders put a value on the other side of one of
 * those roundings, so most rows come back bit-identical and the rest close. */
static float bf16r(float v) {
    uint32_t b; memcpy(&b, &v, 4);
    if ((b & 0x7f800000u) != 0x7f800000u) b += 0x7fffu + ((b >> 16) & 1u);
    b &= 0xffff0000u; memcpy(&v, &b, 4);
    return v;
}
static float e4m3r(float value) {
    int neg = signbit(value) != 0;
    float a = fabsf(value), r;
    if (!a) return value;
    if (a >= 448.0f) r = 448.0f;
    else if (a < 0.015625f) {
        float sc = a * 512.0f; unsigned q = (unsigned)sc; float fr = sc - (float)q;
        if (fr > 0.5f || (fr == 0.5f && (q & 1))) q++;
        r = ldexpf((float)q, -9);
    } else {
        int e; float m = frexpf(a, &e);                     /* a = m 2^e, m in [0.5, 1) */
        float q = nearbyintf(m * 16.0f);                     /* 1.mmm: 4 bits, ties to even */
        r = ldexpf(q, e - 4);
    }
    return neg ? -r : r;
}
static void qdq128(float *v, int n) {
    for (int b = 0; b < n; b += 128) {
        int c = n - b < 128 ? n - b : 128;
        float mx = 0;
        for (int i = 0; i < c; i++) mx = fmaxf(mx, fabsf(v[b + i]));
        mx = fmaxf(mx, 1e-4f);
        int ex; float fr = frexpf(mx / 448.0f, &ex);
        float sc = ldexpf(1.0f, fr == 0.5f ? ex - 1 : ex);
        for (int i = 0; i < c; i++) v[b + i] = e4m3r(fmaxf(-448.0f, fminf(448.0f, v[b + i] / sc))) * sc;
    }
}
static void expert_v4(const Ex *e, const float *xq, float w, float limit, float *y) {
    float h[HF_MAX];
    for (int o = 0; o < F; o++) {
        double a = 0, b = 0;
        for (int i = 0; i < H; i++) { a += (double)e->g.w[(size_t)o * H + i] * xq[i]; b += (double)e->u.w[(size_t)o * H + i] * xq[i]; }
        float g = bf16r((float)a), u = bf16r((float)b);
        if (limit > 0) { g = fminf(g, limit); u = fmaxf(-limit, fminf(u, limit)); }
        float sg = g >= 0 ? 1.0f / (1.0f + expf(-g)) : expf(g) / (1.0f + expf(g));
        h[o] = bf16r(g * sg * u * w);
    }
    qdq128(h, F);
    for (int o = 0; o < H; o++) { double a = 0; for (int i = 0; i < F; i++) a += (double)e->d.w[(size_t)o * F + i] * h[i]; y[o] = bf16r((float)a); }
}
static void v4_act(void) {
    VktFmt f = {VKT_SRC_MXFP4_E8M0, 32};
    model_make(f, f);
    const float limit = 10.0f;
    set_budget(L * E, f, f);
    setenv("COLI_VK_TIER_RATE", "64", 1);
    VktConfig vc = cfg_of(f, f, VKT_ACT_SWIGLU_V4, limit);
    int on = cut_init(&vc, NULL);
    CHECK(on, "v4: the tier did not start");
    if (!on) { model_free(); return; }
    unsigned long long dev = 0, rows_same = 0, routed = 0; double worst = 0, worst_sum = 0;
    static const int Ss[] = {1, 3, 20, 1, 2, 24};
    for (int t = 0; t < 6; t++)
        for (int l = 0; l < L; l++) {
            int S = Ss[t], idx[24 * K]; float w[24 * K];
            route(S, 0, E, idx, w);
            float *x = malloc(sizeof(float) * S * H), *xq = malloc(sizeof(float) * S * H);
            float *cpu = calloc((size_t)S * K * H, sizeof(float)), *y = malloc(sizeof(float) * H);
            float *out = calloc((size_t)S * H, sizeof(float)), *ref = calloc((size_t)S * H, sizeof(float));
            uint8_t taken[24 * K]; const float *rows[24 * K];
            for (int i = 0; i < S * H; i++) x[i] = frnd() * 4.0f;
            memcpy(xq, x, sizeof(float) * S * H);
            for (int s = 0; s < S; s++) qdq128(xq + (size_t)s * H, H);   /* the engine's rounding of x */
            int n = cut_issue_w(l, xq, S, K, idx, w, taken);
            for (int i = 0; i < S * K; i++) {
                if (taken[i]) continue;
                expert_v4(&ex[l][idx[i]], xq + (size_t)(i / K) * H, w[i], limit, cpu + (size_t)i * H);
                VktExpertSrc src = src_of(&ex[l][idx[i]]); cut_note(l, idx[i], &src);
            }
            int joined = n ? cut_join(rows) : 1;
            CHECK(joined, "v4: join failed");
            for (int s = 0; s < S; s++)
                for (int k = 0; k < K; k++) {
                    int i = s * K + k;
                    expert_v4(&ex[l][idx[i]], xq + (size_t)s * H, w[i], limit, y);
                    for (int d = 0; d < H; d++) ref[(size_t)s * H + d] += y[d];
                    if (taken[i] && joined) {
                        float dr[HF_MAX];
                        for (int d = 0; d < H; d++) dr[d] = bf16r(rows[i][d]);
                        double r = rel(dr, y, H); if (r > worst) worst = r;
                        rows_same += !memcmp(dr, y, (size_t)H * sizeof(float));
                        for (int d = 0; d < H; d++) out[(size_t)s * H + d] += dr[d];
                        dev++;
                    } else for (int d = 0; d < H; d++) out[(size_t)s * H + d] += cpu[(size_t)i * H + d];
                    routed++;
                }
            double r = rel(out, ref, S * H); if (r > worst_sum) worst_sum = r;
            free(x); free(xq); free(cpu); free(y); free(out); free(ref);
        }
    printf("  %-34s device %3llu of %3llu assignments, %llu rows bit-identical to the CPU's arithmetic, worst row %.2e, worst sum %.2e\n",
           "MXFP4 ue8m0, DeepSeek V4 roundings", dev, routed, rows_same, worst, worst_sum);
    CHECK(dev > 0, "v4: nothing ran on the device");
    CHECK(rows_same * 2 > dev, "v4: only %llu of %llu device rows equal the CPU's arithmetic", rows_same, dev);
    CHECK(worst < 2e-2 && worst_sum < 2e-2, "v4: device rows off the reference (row %.3g, sum %.3g)", worst, worst_sum);
    cut_shutdown(); model_free();
}

/* DeepSeek V4's activation on a GGUF's experts at the model's size (V4 Flash: hidden 4096, expert
 * width 2048, IQ1_S gate and up, IQ3_XXS down), as v4_act does on MXFP4 */
static void v4_gguf(VktFmt gu, VktFmt f, int h, int w, const char *nm) {
    const int H0 = H, F0 = F;
    H = h; F = w;
    const int blk = gu.kind >= VKT_SRC_GGML;
    for (int l = 0; l < L; l++) for (int e = 0; e < E; e++) {
        if (blk) { mat_make_block(&ex[l][e].g, gu, H, F); mat_make_block(&ex[l][e].u, gu, H, F); mat_make_block(&ex[l][e].d, f, F, H); }
        else { mat_make(&ex[l][e].g, gu, H, F); mat_make(&ex[l][e].u, gu, H, F); mat_make(&ex[l][e].d, f, F, H); }
    }
    const float limit = 10.0f;
    set_budget(L * E, gu, f);
    setenv("COLI_VK_TIER_RATE", "64", 1);
    VktConfig vc = cfg_of(gu, f, VKT_ACT_SWIGLU_V4, limit);
    int on = cut_init(&vc, NULL);
    CHECK(on, "v4: the tier did not start");
    if (!on) { model_free(); H = H0; F = F0; return; }
    unsigned long long dev = 0, rows_same = 0, routed = 0; double worst = 0, worst_sum = 0;
    static const int Ss[] = {1, 3, 20, 1, 2, 24};
    for (int t = 0; t < 6; t++)
        for (int l = 0; l < L; l++) {
            int S = Ss[t], idx[24 * K]; float w[24 * K];
            route(S, 0, E, idx, w);
            float *x = malloc(sizeof(float) * S * H), *xq = malloc(sizeof(float) * S * H);
            float *cpu = calloc((size_t)S * K * H, sizeof(float)), *y = malloc(sizeof(float) * H);
            float *out = calloc((size_t)S * H, sizeof(float)), *ref = calloc((size_t)S * H, sizeof(float));
            uint8_t taken[24 * K]; const float *rows[24 * K];
            for (int i = 0; i < S * H; i++) x[i] = frnd() * 4.0f;
            memcpy(xq, x, sizeof(float) * S * H);
            for (int s = 0; s < S; s++) qdq128(xq + (size_t)s * H, H);   /* the engine's rounding of x */
            int n = cut_issue_w(l, xq, S, K, idx, w, taken);
            for (int i = 0; i < S * K; i++) {
                if (taken[i]) continue;
                expert_v4(&ex[l][idx[i]], xq + (size_t)(i / K) * H, w[i], limit, cpu + (size_t)i * H);
                VktExpertSrc src = src_of(&ex[l][idx[i]]); cut_note(l, idx[i], &src);
            }
            int joined = n ? cut_join(rows) : 1;
            CHECK(joined, "v4: join failed");
            for (int s = 0; s < S; s++)
                for (int k = 0; k < K; k++) {
                    int i = s * K + k;
                    expert_v4(&ex[l][idx[i]], xq + (size_t)s * H, w[i], limit, y);
                    for (int d = 0; d < H; d++) ref[(size_t)s * H + d] += y[d];
                    if (taken[i] && joined) {
                        float dr[HF_MAX];
                        for (int d = 0; d < H; d++) dr[d] = bf16r(rows[i][d]);
                        double r = rel(dr, y, H); if (r > worst) worst = r;
                        rows_same += !memcmp(dr, y, (size_t)H * sizeof(float));
                        for (int d = 0; d < H; d++) out[(size_t)s * H + d] += dr[d];
                        dev++;
                    } else for (int d = 0; d < H; d++) out[(size_t)s * H + d] += cpu[(size_t)i * H + d];
                    routed++;
                }
            double r = rel(out, ref, S * H); if (r > worst_sum) worst_sum = r;
            free(x); free(xq); free(cpu); free(y); free(out); free(ref);
        }
    printf("  %-34s device %3llu of %3llu assignments, %llu rows bit-identical to the CPU's arithmetic, worst row %.2e, worst sum %.2e\n",
           nm, dev, routed, rows_same, worst, worst_sum);
    CHECK(dev > 0, "v4: nothing ran on the device");
    CHECK(rows_same * 2 > dev, "v4: only %llu of %llu device rows equal the CPU's arithmetic", rows_same, dev);
    CHECK(worst < 2e-2 && worst_sum < 2e-2, "v4: device rows off the reference (row %.3g, sum %.3g)", worst, worst_sum);
    cut_shutdown(); model_free();
    H = H0; F = F0;
}


static void report_line(char *out, size_t n) {
    out[0] = 0;
    FILE *t = tmpfile();
    if (!t) return;
    fflush(stderr);
    int saved = dup(2);
    if (saved < 0) { fclose(t); return; }
    dup2(fileno(t), 2);
    cut_report("test", 0, 0);
    fflush(stderr);
    dup2(saved, 2); close(saved);
    rewind(t);
    char line[4096];
    while (fgets(line, sizeof line, t)) { fputs(line, stderr); if (strstr(line, "] tier test test:")) snprintf(out, n, "%s", line); }
    fclose(t);
}
static unsigned long long after(const char *line, const char *key) {
    const char *p = strstr(line, key);
    return p ? strtoull(p + strlen(key), NULL, 10) : ~0ull;
}
/* the report's "hybrid: N steps, P host pool pairs": P, ~0 without the part */
static unsigned long long hy_pairs(const char *line) {
    const char *p = strstr(line, "| hybrid: ");
    if (!p) return ~0ull;
    p = strstr(p, "steps, ");
    return p ? strtoull(p + 7, NULL, 10) : ~0ull;
}
static void hybrid_case(VktFmt gu, VktFmt dn, const char *nm) {
    const int H0 = H, F0 = F;
    H = 512; F = 256;   /* whole blocks of 256 */
    const int blk = gu.kind >= VKT_SRC_GGML;
    for (int l = 0; l < L; l++) for (int e = 0; e < E; e++) {
        if (blk) { mat_make_block(&ex[l][e].g, gu, H, F); mat_make_block(&ex[l][e].u, gu, H, F); mat_make_block(&ex[l][e].d, dn, F, H); }
        else { mat_make(&ex[l][e].g, gu, H, F); mat_make(&ex[l][e].u, gu, H, F); mat_make(&ex[l][e].d, dn, F, H); }
    }
    g_act = VKT_ACT_SWIGLU; g_limit = 0;
    const double eb = (double)cut_expert_bytes(H, F, gu, dn) / 1073741824.0;
    char gb[64];
    setenv("COLI_VK_TIER_HOST", "1", 1);
    setenv("COLI_VK_TIER_SYNC", "1", 1);
    setenv("COLI_VK_TIER_PROMOTE", "0", 1);   /* the device keeps its four: the pool's experts stay there */
    setenv("GQ_EXACT", "1", 1);               /* the CPU's float form, against the float reference */
    uint32_t hist[L][E], *rows[L];
    for (int l = 0; l < L; l++) { rows[l] = hist[l]; for (int e = 0; e < E; e++) hist[l][e] = e < 2 ? 100 : 1; }
    for (int c = 0; c < 2; c++) {   /* the hybrid on, then COLI_VK_HYBRID=0 */
        if (c) setenv("COLI_VK_HYBRID", "0", 1);
        set_budget(4, gu, dn);
        snprintf(gb, sizeof gb, "%.12f", (L * E + 0.5) * eb);
        setenv("COLI_VK_TIER_HOST_GB", gb, 1);
        VktConfig vc = cfg_of(gu, dn, VKT_ACT_SWIGLU, 0);
        int on = cut_init(&vc, rows);
        CHECK(on, "hybrid: the tier did not start");
        if (!on) break;
        int pl[L * E], pe[L * E], n = cut_plan(pl, pe, L * E);
        for (int i = 0; i < n; i++) { VktExpertSrc src = src_of(&ex[pl[i]][pe[i]]); cut_put(pl[i], pe[i], &src); }
        cut_put_done();
        Books bk = {0, 0}; double worst = 0;
        static const int Ss[] = {1, 3, 8, 1, 2, 8};
        int idx[20 * K]; float w[20 * K];
        for (int t = 0; t < 6; t++)
            for (int l = 0; l < L; l++) { route(Ss[t], 0, E, idx, w); double r = step(l, Ss[t], idx, w, &bk); if (r > worst) worst = r; }
        char line[4096]; report_line(line, sizeof line);
        unsigned long long pairs = hy_pairs(line);
        for (int l = 0; l < L; l++) { route(20, 0, E, idx, w); double r = step(l, 20, idx, w, &bk); if (r > worst) worst = r; }
        char line2[4096]; report_line(line2, sizeof line2);
        unsigned long long pairs2 = hy_pairs(line2);
        printf("  %s experts in the host pool, the hybrid %s: %llu pairs computed by the CPU, %llu after a 20-row step; "
               "%llu of %llu assignments taken, worst relative error %.2e\n", nm, c ? "off" : "on",
               pairs == ~0ull ? 0 : pairs, pairs2 == ~0ull ? 0 : pairs2, bk.dev, bk.routed, worst);
        CHECK(worst < 2e-3, "hybrid: relative error %.3g", worst);
        CHECK(bk.dev == bk.routed, "hybrid: the tier took %llu of %llu", bk.dev, bk.routed);
        if (!c) {
            CHECK(pairs != ~0ull && pairs > 0, "hybrid: no host pool pair computed by the CPU");
            CHECK(pairs2 == pairs, "hybrid: a 20-row step computed %llu pairs on the CPU", pairs2 - pairs);
        } else
            CHECK(pairs == ~0ull, "hybrid: COLI_VK_HYBRID=0 computed %llu pairs on the CPU", pairs);
        cut_shutdown();
    }
    unsetenv("COLI_VK_HYBRID"); unsetenv("GQ_EXACT"); unsetenv("COLI_VK_TIER_PROMOTE");
    unsetenv("COLI_VK_TIER_HOST_GB"); unsetenv("COLI_VK_TIER_SYNC");
    setenv("COLI_VK_TIER_HOST", "0", 1);
    model_free();
    H = H0; F = F0;
}


int main(void) {
    /* the float form throughout: the tier's CPU pairs and the card's block GEMVs read GQ_EXACT
     * once, at their start */
    setenv("GQ_EXACT", "1", 1);
    int dev = 0;
    if (!coli_cuda_init(&dev, 1)) { printf("FAIL: no CUDA device\n"); return 1; }
    setenv("COLI_VK_TIER_RESERVE_GB", "0", 1);
    setenv("COLI_VK_TIER_HOST", "0", 1);   /* the budgets below are the device's; host_case sets its own */
    printf("formats:\n"); formats();
    printf("blocks:\n"); blocks();
    printf("adapt:\n"); adapt();
    printf("DeepSeek V4:\n"); v4_act();
    v4_gguf((VktFmt){(VktSrc)(VKT_SRC_GGML + GQ_IQ1_S), 0}, (VktFmt){(VktSrc)(VKT_SRC_GGML + GQ_IQ3_XXS), 0}, 4096, 2048, "IQ1_S/IQ3_XXS 4096x2048, V4");
    v4_gguf((VktFmt){(VktSrc)(VKT_SRC_GGML + GQ_IQ1_S), 0}, (VktFmt){(VktSrc)(VKT_SRC_GGML + GQ_IQ3_XXS), 0}, 512, 256, "IQ1_S/IQ3_XXS 512x256, V4");
    v4_gguf((VktFmt){(VktSrc)(VKT_SRC_GGML + GQ_IQ1_S), 0}, (VktFmt){(VktSrc)(VKT_SRC_GGML + GQ_IQ2_XXS), 0}, 4096, 2048, "IQ1_S/IQ2_XXS 4096x2048, V4");
    v4_gguf((VktFmt){VKT_SRC_MXFP4_E8M0, 32}, (VktFmt){VKT_SRC_MXFP4_E8M0, 32}, 4096, 2048, "MXFP4 4096x2048, V4");
    printf("host pool:\n");
    hybrid_case((VktFmt){(VktSrc)(VKT_SRC_GGML + GQ_Q4_K), 0}, (VktFmt){(VktSrc)(VKT_SRC_GGML + GQ_Q6_K), 0}, "Q4_K/Q6_K");
    hybrid_case((VktFmt){VKT_SRC_I4U_PAIRS_GS, 64}, (VktFmt){VKT_SRC_I4U_PAIRS_GS, 64}, "int4-g64");
    hybrid_case((VktFmt){VKT_SRC_I8_ROW, 0}, (VktFmt){VKT_SRC_I8_ROW, 0}, "int8");
    coli_cuda_shutdown();
    printf(fails ? "FAIL (%d)\n" : "PASS\n", fails);
    return fails != 0;
}
