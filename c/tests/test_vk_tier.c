/* The Vulkan routed-expert tier (vk_tier.c) against a CPU reference, on any Vulkan
 * device (CI: Lavapipe). A small synthetic MoE -- L layers of E experts, top-K --
 * whose weights are random in every source format the tier accepts, decoded here
 * on their own, independently of the tier and of the backend. Gates:
 *   formats  every VktSrc kind and both activations (uploads awaited, COLI_VK_TIER_SYNC):
 *            each row the device returns
 *            equals the reference expert output, and the rank-order sum of a mixed
 *            step (device and CPU rows) equals the all-CPU sum, within 2e-3;
 *   warm     a history fills the budget in heat order (vkt_plan / vkt_put);
 *   adapt    a budget of a third of the experts and a hot set that moves halfway:
 *            experts are evicted, the new hot set becomes resident and its steps
 *            go to the device;
 *   partial  a step with more assignments than max_rows: the device takes
 *            max_rows of them, the CPU the rest, the sum is still right;
 *   sync     COLI_VK_TIER_SYNC=1 and a budget of two experts: a promotion that
 *            displaces a resident while a batch is in flight finds its room at
 *            the join (no failed upload, the budget holds);
 *   books    device + CPU = routed, resident <= budget, no failed upload;
 *   v4       DeepSeek V4's activation (route weight on the device, its bf16 and E4M3
 *            roundings): device rows against the engine's CPU arithmetic;
 *   extra    an extra layer (an MTP head's) in another, bigger form: each layer's rows
 *            on the device equal the reference, its experts sit in a pool of their
 *            own with its share of the budget (each pool evicts its own kind, no
 *            upload refused), it gets its expert in beside a full main pool on its
 *            own promotion rate, and with no device form for it the tier serves the
 *            main layers;
 *   stream   big prefill steps with streaming (vkt_issue's sub-batches, the staging
 *            slots, prefetch, experts cut into parts): every device row, resident or
 *            streamed, equals the reference, the sum the all-CPU sum, the cold experts
 *            below the rule's rows stay on the CPU, and the resident set is the warm
 *            start's before and after (streaming promotes nothing).
 *
 *   make tests/test_vk_tier VK=1 && VK_ICD_FILENAMES=.../lvp_icd.json ./tests/test_vk_tier */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include "../backend_vulkan.h"
#include "../vk_tier.h"

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fails++; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

enum { L = 2, E = 10, H = 192, F = 128, K = 3 };
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
    float h[F];
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
    int n = vkt_issue(layer, x, S, K, idx, taken);
    for (int i = 0; i < S * K; i++) {
        if (taken[i]) continue;
        expert_ref(&ex[layer][idx[i]], x + (size_t)(i / K) * H, cpu + (size_t)i * H);
        VktExpertSrc s = src_of(&ex[layer][idx[i]]); vkt_note(layer, idx[i], &s);
    }
    int joined = n ? vkt_join(dev) : 1;
    CHECK(joined, "join failed");
    double worst = 0;
    for (int s = 0; s < S; s++)
        for (int k = 0; k < K; k++) {
            int i = s * K + k;
            expert_ref(&ex[layer][idx[i]], x + (size_t)s * H, y);
            for (int d = 0; d < H; d++) ref[(size_t)s * H + d] += w[i] * y[d];
            const float *c = taken[i] && joined ? dev[i] : cpu + (size_t)i * H;
            if (taken[i] && joined) { double r = rel(c, y, H); if (r > worst) worst = r; }
            for (int d = 0; d < H; d++) out[(size_t)s * H + d] += w[i] * c[d];
            bk->routed++; bk->dev += taken[i] != 0;
        }
    double r = rel(out, ref, S * H);
    if (r > worst) worst = r;
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
    char b[64]; snprintf(b, sizeof b, "%.12f", (experts + 0.5) * vkt_expert_bytes(H, F, gu, dn) / 1073741824.0);
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
        int on = vkt_init(&vc, NULL);
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
        vkt_shutdown(); model_free();
    }
}

/* a history fills the budget in heat order */
static void warm(void) {
    VktFmt f = {VKT_SRC_I4U_PAIRS_GS, 64};
    model_make(f, f); g_act = VKT_ACT_SWIGLU; g_limit = 0;
    set_budget(7, f, f);
    uint32_t hist[L][E], *rows[L];
    for (int l = 0; l < L; l++) { rows[l] = hist[l]; for (int e = 0; e < E; e++) hist[l][e] = (uint32_t)(1000 - 37 * (l * E + e) % 700); }
    VktConfig vc = cfg_of(f, f, VKT_ACT_SWIGLU, 0);
    CHECK(vkt_init(&vc, rows), "warm: the tier did not start");
    int pl[L * E], pe[L * E];
    int n = vkt_plan(pl, pe, L * E);
    CHECK(n == 7, "warm: planned %d experts for a budget of 7", n);
    /* the plan is the 7 hottest, hottest first */
    int ok = 1;
    for (int i = 1; i < n; i++) ok &= hist[pl[i - 1]][pe[i - 1]] >= hist[pl[i]][pe[i]];
    for (int l = 0; l < L; l++) for (int e = 0; e < E; e++) {
        int planned = 0; for (int i = 0; i < n; i++) planned |= pl[i] == l && pe[i] == e;
        if (!planned) ok &= hist[l][e] <= hist[pl[n - 1]][pe[n - 1]];
    }
    CHECK(ok, "warm: the plan is not the hottest experts in heat order");
    for (int i = 0; i < n; i++) { VktExpertSrc s = src_of(&ex[pl[i]][pe[i]]); vkt_put(pl[i], pe[i], &s); }
    vkt_put_done();
    int res = 0; for (int i = 0; i < n; i++) res += vkt_resident(pl[i], pe[i]);
    CHECK(res == n, "warm: %d of %d planned experts resident", res, n);
    Books bk = {0, 0};
    int idx[K] = {pe[0], (pe[0] + 1) % E, (pe[0] + 2) % E}; float w[K] = {0.5f, 0.3f, 0.2f};
    double r = step(pl[0], 1, idx, w, &bk);
    CHECK(bk.dev >= 1 && r < 2e-3, "warm: the hottest expert is not served from the device (%llu, %.3g)", bk.dev, r);
    printf("  warm start: %d experts planned in heat order and resident, the hottest served by the device\n", n);
    vkt_shutdown(); model_free();
}

/* a third of the experts fit; the hot set moves halfway */
static void adapt(void) {
    VktFmt f = {VKT_SRC_FP8_GS, 32};
    model_make(f, f); g_act = VKT_ACT_SWIGLU; g_limit = 0;
    set_budget(L * E / 3, f, f);
    setenv("COLI_VK_TIER_RATE", "4", 1);
    VktConfig vc = cfg_of(f, f, VKT_ACT_SWIGLU, 0);
    CHECK(vkt_init(&vc, NULL), "adapt: the tier did not start");
    Books a = {0, 0}, b1 = {0, 0}, b2 = {0, 0}; double worst = 0;
    int idx[4 * K]; float w[4 * K];
    /* LFRU: frequency first, halved every 1024 tokens, so an established hot set
     * yields over a few hundred steps of a new one, not at its first appearance */
    for (int t = 0; t < 100; t++)                     /* phase A: experts 0..4 */
        for (int l = 0; l < L; l++) { route(4, 0, 5, idx, w); double r = step(l, 4, idx, w, &a); if (r > worst) worst = r; }
    int resA = 0; for (int l = 0; l < L; l++) for (int e = 0; e < 5; e++) resA += vkt_resident(l, e);
    for (int t = 0; t < 400; t++)                     /* phase B: experts 5..9 */
        for (int l = 0; l < L; l++) { route(4, 5, 5, idx, w); double r = step(l, 4, idx, w, t < 200 ? &b1 : &b2); if (r > worst) worst = r; }
    int resB = 0, resA2 = 0, res = 0;
    for (int l = 0; l < L; l++) for (int e = 0; e < E; e++) { int v = vkt_resident(l, e); res += v; if (e >= 5) resB += v; else resA2 += v; }
    printf("  adapt: phase A device %.0f%%, resident A %d; phase B device %.0f%% then %.0f%%, resident B %d, A %d, all %d (budget %d)\n",
           100.0 * a.dev / a.routed, resA, 100.0 * b1.dev / b1.routed, 100.0 * b2.dev / b2.routed, resB, resA2, res, L * E / 3);
    vkt_report("test", 0, 0);
    CHECK(resA > 0 && a.dev > 0, "adapt: phase A promoted nothing");
    CHECK(resB > resA2, "adapt: the new hot set did not take the device (B %d, A %d)", resB, resA2);
    CHECK(res <= L * E / 3, "adapt: %d resident over a budget of %d", res, L * E / 3);
    CHECK(b2.dev * b1.routed > b1.dev * b2.routed, "adapt: the device share did not rise in phase B");
    CHECK(worst < 2e-3, "adapt: relative error %.3g", worst);
    vkt_shutdown(); model_free();
}

/* more assignments than max_rows: the rest stays on the CPU */
static void partial(void) {
    VktFmt f = {VKT_SRC_I8_ROW, 0};
    model_make(f, f); g_act = VKT_ACT_SWIGLU; g_limit = 0;
    set_budget(L * E, f, f);
    VktConfig vc = cfg_of(f, f, VKT_ACT_SWIGLU, 0); vc.max_rows = 5;
    setenv("COLI_VK_TIER_RATE", "64", 1);              /* every expert resident after a few steps */
    CHECK(vkt_init(&vc, NULL), "partial: the tier did not start");
    Books bk = {0, 0}; int idx[8 * K]; float w[8 * K];
    for (int t = 0; t < 8; t++) for (int l = 0; l < L; l++) { route(8, 0, E, idx, w); step(l, 8, idx, w, &bk); }
    Books one = {0, 0};
    route(8, 0, E, idx, w);
    double r = step(0, 8, idx, w, &one);
    printf("  partial: a step of %llu assignments, the device took %llu (max_rows 5), relative error %.2e\n", one.routed, one.dev, r);
    CHECK(one.dev > 0 && one.dev <= 5, "partial: the device took %llu of a 5-row cap", one.dev);
    CHECK(r < 2e-3, "partial: relative error %.3g", r);
    vkt_shutdown(); model_free();
}

/* vkt_report's evictions and failed uploads, read back from its stderr line */
static void report_counts(unsigned long long *evictions, unsigned long long *failed) {
    *evictions = *failed = ~0ull;
    FILE *t = tmpfile();
    if (!t) return;
    fflush(stderr);
    int saved = dup(2);
    if (saved < 0) { fclose(t); return; }
    dup2(fileno(t), 2);
    vkt_report("test", 0, 0);
    fflush(stderr);
    dup2(saved, 2); close(saved);
    rewind(t);
    char line[4096];
    while (fgets(line, sizeof line, t)) {
        fputs(line, stderr);
        const char *e = strstr(line, "evictions "), *f = strstr(line, "failed ");
        if (e) *evictions = strtoull(e + 10, NULL, 10);
        if (f) *failed = strtoull(f + 7, NULL, 10);
    }
    fclose(t);
}

/* COLI_VK_TIER_SYNC=1 and a budget of two experts. Phase A makes experts 0 and 1 of
 * layer 0 resident; in phase B every step routes 0 (resident: the step's batch is in
 * flight) with 5 and 6, which the CPU computes and notes, so the promotion of 5 that
 * displaces the cold 1 is decided while a batch runs and 1 is freed only at the
 * join. The uploader must wait for that free: it used to try at once, meet the full
 * pool and give up (sync mode does not retry), the upload counted as failed and the
 * budget shrank to the one expert left, so 5 never became resident. */
static void sync_evict(void) {
    VktFmt f = {VKT_SRC_I8_ROW, 0};
    model_make(f, f); g_act = VKT_ACT_SWIGLU; g_limit = 0;
    set_budget(2, f, f);
    setenv("COLI_VK_TIER_RATE", "64", 1);
    setenv("COLI_VK_TIER_SYNC", "1", 1);
    VktConfig vc = cfg_of(f, f, VKT_ACT_SWIGLU, 0);
    int on = vkt_init(&vc, NULL);
    unsetenv("COLI_VK_TIER_SYNC");
    CHECK(on, "sync: the tier did not start");
    if (!on) { model_free(); return; }
    Books a = {0, 0}, b = {0, 0}; double worst = 0;
    int idx_a[K] = {0, 1, 2}, idx_b[K] = {0, 5, 6}; float w[K] = {0.5f, 0.3f, 0.2f};
    for (int t = 0; t < 12; t++) { vkt_begin_forward(); double r = step(0, 1, idx_a, w, &a); if (r > worst) worst = r; }
    int res_a = vkt_resident(0, 0) + vkt_resident(0, 1);
    for (int t = 0; t < 40; t++) { vkt_begin_forward(); double r = step(0, 1, idx_b, w, &b); if (r > worst) worst = r; }
    int res = 0; for (int l = 0; l < L; l++) for (int e = 0; e < E; e++) res += vkt_resident(l, e);
    unsigned long long ev, failed;
    report_counts(&ev, &failed);
    printf("  sync: phase A resident %d of 2; phase B device %llu of %llu, resident 0 %d, 1 %d, 5 %d, all %d (budget 2), evictions %llu, failed %llu\n",
           res_a, b.dev, b.routed, vkt_resident(0, 0), vkt_resident(0, 1), vkt_resident(0, 5), res, ev, failed);
    CHECK(res_a == 2, "sync: phase A made %d of experts 0 and 1 resident", res_a);
    CHECK(ev >= 1 && failed == 0, "sync: %llu evictions, %llu failed uploads (a promotion gave up on its victim's room)", ev, failed);
    CHECK(vkt_resident(0, 5) && !vkt_resident(0, 1) && res == 2,
          "sync: expert 5 did not displace expert 1 (resident 1 %d, 5 %d, all %d of a budget of 2)", vkt_resident(0, 1), vkt_resident(0, 5), res);
    CHECK(worst < 2e-3, "sync: relative error %.3g", worst);
    vkt_shutdown(); model_free();
}

/* An extra layer: layer 0 is the model's (int4 v+8 gs64), layer 1 an MTP head's held
 * as f32 (VktConfig.extra_layers), eight times the bytes. Its experts sit in a pool of
 * their own with the extra layer's share of the budget; uploads awaited throughout. */
static void extra(void) {
    VktFmt fm = {VKT_SRC_I4U_PAIRS_GS, 64}, fx = {VKT_SRC_F32, 0};
    layer_make(0, fm, fm); layer_make(1, fx, fx); g_act = VKT_ACT_SWIGLU; g_limit = 0;
    size_t mb = vkt_expert_bytes(H, F, fm, fm), xb = vkt_expert_bytes(H, F, fx, fx);
    int per = (int)((xb + mb - 1) / mb);
    unsigned long long ev, failed;
    int idx[4 * K]; float w[4 * K];
    VktConfig vc = cfg_of(fm, fm, VKT_ACT_SWIGLU, 0);
    vc.layers = 1; vc.extra_layers = 1; vc.extra_gate_up = fx; vc.extra_down = fx;
    /* Half of a budget of 2 * per + 4 main experts is the extra layer's: one f32 expert,
     * whose room leaves the main layer all ten of its own. Both layers route, the hot
     * set moves halfway: each pool evicts in its own kind, every row is right. */
    int units = 2 * per + 4;
    set_budget(units, fm, fm);
    setenv("COLI_VK_TIER_RATE", "64", 1);
    setenv("COLI_VK_TIER_SYNC", "1", 1);
    int on = vkt_init(&vc, NULL);
    unsetenv("COLI_VK_TIER_SYNC");
    CHECK(on && vkt_layers() == 2, "extra: the tier did not start with its extra layer (%d layers)", vkt_layers());
    if (!on) { model_free(); return; }
    Books b0 = {0, 0}, b1 = {0, 0}; double worst = 0;
    for (int t = 0; t < 140; t++)   /* the new hot set needs a while to outweigh the old one */
        for (int l = 0; l < 2; l++) {
            route(4, t < 40 ? 0 : 5, 5, idx, w);
            double r = step(l, 4, idx, w, l ? &b1 : &b0);
            if (r > worst) worst = r;
        }
    int r0 = 0, r1 = 0;
    for (int e = 0; e < E; e++) { r0 += vkt_resident(0, e); r1 += vkt_resident(1, e); }
    report_counts(&ev, &failed);
    ColiVkPoolStats xp; coli_vk_pool_stats(4, &xp);
    printf("  extra: an f32 expert is %d int4 ones; device main %llu of %llu, extra %llu of %llu; resident main %d, "
           "extra %d (its pool %zu of %zu bytes), evictions %llu, failed %llu, worst relative error %.2e\n",
           per, b0.dev, b0.routed, b1.dev, b1.routed, r0, r1, xp.used, xp.limit, ev, failed, worst);
    CHECK(per > 1, "extra: the f32 form takes no more room than the int4 one (%zu, %zu bytes)", xb, mb);
    CHECK(b0.dev > 0 && b1.dev > 0, "extra: a layer ran nothing on the device (main %llu, extra %llu)", b0.dev, b1.dev);
    CHECK(r0 == E && r1 == 1, "extra: resident main %d (want %d), extra %d (want 1)", r0, E, r1);
    CHECK(xp.used > 0 && xp.used <= xp.limit, "extra: the extra pool holds %zu bytes of %zu", xp.used, xp.limit);
    CHECK(ev >= 1 && failed == 0, "extra: %llu evictions, %llu failed uploads", ev, failed);
    CHECK(worst < 2e-3, "extra: relative error %.3g", worst);
    vkt_shutdown();
    /* A budget of per + 4: the extra layer's share is under one of its experts, so it
     * gets one (two main ones still fit) and the main layer the four places left. The
     * main layer fills them, then keeps routing and spends each forward's promotion
     * (COLI_VK_TIER_RATE=1) before the extra layer runs: the extra layer's expert must
     * still get in, on its own promotion. */
    units = per + 4;
    set_budget(units, fm, fm);
    setenv("COLI_VK_TIER_RATE", "1", 1);
    setenv("COLI_VK_TIER_SYNC", "1", 1);
    on = vkt_init(&vc, NULL);
    unsetenv("COLI_VK_TIER_SYNC");
    CHECK(on && vkt_layers() == 2, "extra: the tier did not start for the full-tier case");
    if (on) {
        Books c0 = {0, 0}, c1 = {0, 0}; worst = 0;
        for (int t = 0; t < 40; t++) { route(4, 0, E, idx, w); double r = step(0, 4, idx, w, &c0); if (r > worst) worst = r; }
        int full = 0; for (int e = 0; e < E; e++) full += vkt_resident(0, e);
        for (int t = 0; t < 60; t++) {
            route(4, 0, E, idx, w); double r = step(0, 4, idx, w, &c0); if (r > worst) worst = r;
            route(4, 0, 5, idx, w); r = step(1, 4, idx, w, &c1); if (r > worst) worst = r;
        }
        r0 = r1 = 0;
        for (int e = 0; e < E; e++) { r0 += vkt_resident(0, e); r1 += vkt_resident(1, e); }
        report_counts(&ev, &failed);
        printf("  extra, full tier: main resident %d after the first phase; then main %d + extra %d, "
               "extra device %llu of %llu, evictions %llu, failed %llu, worst %.2e\n",
               full, r0, r1, c1.dev, c1.routed, ev, failed, worst);
        CHECK(full == 4 && r0 == 4, "extra: the main layer holds %d then %d of its 4 places", full, r0);
        CHECK(r1 == 1 && c1.dev > 0, "extra: no extra expert got in beside a full main pool (resident %d, device %llu)", r1, c1.dev);
        CHECK(ev >= 1 && failed == 0, "extra, full tier: %llu evictions, %llu failed uploads", ev, failed);
        CHECK(worst < 2e-3, "extra, full tier: relative error %.3g", worst);
        vkt_shutdown();
    }
    /* an extra form with no device form: the main layers only */
    vc.extra_gate_up = vc.extra_down = (VktFmt){VKT_SRC_NONE, 0};
    on = vkt_init(&vc, NULL);
    CHECK(on && vkt_layers() == 1, "extra: no device form, the tier has %d layers (want the main one)", vkt_layers());
    if (on) {
        float x[H] = {0}; int id[K] = {0, 1, 2}; uint8_t taken[K];
        CHECK(vkt_issue(1, x, 1, K, id, taken) == 0, "extra: a layer past the tier's took rows");
        vkt_shutdown();
    }
    model_free();
}

/* DeepSeek V4's activation (VKT_ACT_SWIGLU_V4) through the tier: MXFP4 ue8m0 experts,
 * x rounded to E4M3 per 128 here as the engine rounds it, the route weight applied on
 * the device (vkt_issue_w), each device row rounded to bf16 and added with no weight
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
    float h[F];
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
    int on = vkt_init(&vc, NULL);
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
            int n = vkt_issue_w(l, xq, S, K, idx, w, taken);
            for (int i = 0; i < S * K; i++) {
                if (taken[i]) continue;
                expert_v4(&ex[l][idx[i]], xq + (size_t)(i / K) * H, w[i], limit, cpu + (size_t)i * H);
                VktExpertSrc src = src_of(&ex[l][idx[i]]); vkt_note(l, idx[i], &src);
            }
            int joined = n ? vkt_join(rows) : 1;
            CHECK(joined, "v4: join failed");
            for (int s = 0; s < S; s++)
                for (int k = 0; k < K; k++) {
                    int i = s * K + k;
                    expert_v4(&ex[l][idx[i]], xq + (size_t)s * H, w[i], limit, y);
                    for (int d = 0; d < H; d++) ref[(size_t)s * H + d] += y[d];
                    if (taken[i] && joined) {
                        float dr[H];
                        for (int d = 0; d < H; d++) dr[d] = bf16r(rows[i][d]);
                        double r = rel(dr, y, H); if (r > worst) worst = r;
                        rows_same += !memcmp(dr, y, sizeof dr);
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
    vkt_shutdown(); model_free();
}

/* ---- streaming ----------------------------------------------------------------------
 * The engine's load hook: the expert's bytes as its RAM holds them. */
static int g_loads, g_batch_left, g_partial_serial;
static int load_cb(void *ctx, int layer, int eid, VktExpertSrc *src, void **h) {
    if (g_batch_left >= 2) g_partial_serial++;
    if (g_batch_left) g_batch_left--;
    (void)ctx; *src = src_of(&ex[layer][eid]); *h = &ex[layer][eid]; g_loads++; return 1;
}
static void release_cb(void *ctx, void *h) { (void)ctx; (void)h; }
/* An engine with two RAM slots: a device upload group needs several batches. */
static int g_batches;
static int load_batch_cb(void *ctx, int layer, const int *eids, int n, VktExpertSrc *srcs, void **h) {
    (void)ctx; g_batches++;
    int got = n < 2 ? n : 2;
    g_batch_left = n - got;
    for (int i = 0; i < got; i++) { srcs[i] = src_of(&ex[layer][eids[i]]); h[i] = &ex[layer][eids[i]]; g_loads++; }
    return got;
}
/* One big step (vk_tier.h step 3, with the prefetch while "attention" runs); the device
 * rows of cold experts counted apart. */
static double stream_step(int layer, int S, const int *idx, const float *w, int v4, Books *bk,
                          unsigned long long *streamed, unsigned long long *kept, int prefetch) {
    float *x = calloc((size_t)S * H, sizeof(float)), *xq = calloc((size_t)S * H, sizeof(float));
    float *cpu = calloc((size_t)S * K * H, sizeof(float)), *out = calloc((size_t)S * H, sizeof(float));
    float *ref = calloc((size_t)S * H, sizeof(float)), *y = malloc(sizeof(float) * H);
    uint8_t *taken = malloc((size_t)S * K); const float **dev = malloc(sizeof(*dev) * S * K);
    uint8_t *res = malloc((size_t)S * K);
    for (int i = 0; i < S * H; i++) x[i] = frnd() * (v4 ? 4.0f : 1.0f);
    memcpy(xq, x, sizeof(float) * S * H);
    if (v4) for (int s = 0; s < S; s++) qdq128(xq + (size_t)s * H, H);
    for (int i = 0; i < S * K; i++) res[i] = (uint8_t)vkt_resident(layer, idx[i]);
    if (prefetch) vkt_stream_prefetch(layer, S);
    int n = v4 ? vkt_issue_w(layer, xq, S, K, idx, w, taken) : vkt_issue(layer, xq, S, K, idx, taken);
    for (int i = 0; i < S * K; i++) {
        if (taken[i]) continue;
        if (v4) expert_v4(&ex[layer][idx[i]], xq + (size_t)(i / K) * H, w[i], 10.0f, cpu + (size_t)i * H);
        else expert_ref(&ex[layer][idx[i]], xq + (size_t)(i / K) * H, cpu + (size_t)i * H);
        VktExpertSrc s = src_of(&ex[layer][idx[i]]); vkt_note(layer, idx[i], &s);
        (*kept)++;
    }
    int joined = n ? vkt_join(dev) : 1;
    CHECK(joined, "stream: join failed");
    double worst = 0;
    for (int s = 0; s < S; s++)
        for (int k = 0; k < K; k++) {
            int i = s * K + k;
            if (v4) expert_v4(&ex[layer][idx[i]], xq + (size_t)s * H, w[i], 10.0f, y);
            else expert_ref(&ex[layer][idx[i]], xq + (size_t)s * H, y);
            float dr[H];
            const float *c = cpu + (size_t)i * H;
            if (taken[i] && joined) {
                for (int d = 0; d < H; d++) dr[d] = v4 ? bf16r(dev[i][d]) : dev[i][d];
                c = dr;
                double r = rel(c, y, H); if (r > worst) worst = r;
                if (!res[i]) (*streamed)++;
            }
            for (int d = 0; d < H; d++) { ref[(size_t)s * H + d] += (v4 ? 1.0f : w[i]) * y[d]; out[(size_t)s * H + d] += (v4 ? 1.0f : w[i]) * c[d]; }
            bk->routed++; bk->dev += taken[i] != 0;
        }
    double r = rel(out, ref, S * H);
    if (r > worst) worst = r;
    free(x); free(xq); free(cpu); free(out); free(ref); free(y); free(taken); free(dev); free(res);
    return worst;
}
static void stream(void) {
    struct { VktFmt gu, dn; int act; const char *name; } cs[] = {
        {{VKT_SRC_I4U_PLANAR64, 64}, {VKT_SRC_I4U_PLANAR64, 64}, VKT_ACT_SWIGLU, "planar int4-g64"},
        {{VKT_SRC_I8_AS_I4_GS, 64}, {VKT_SRC_I8_GS, 32}, VKT_ACT_SWIGLU, "unpacked int4 gs64, int8 down"},
        {{VKT_SRC_FP8_BLOCK, 128}, {VKT_SRC_FP8_BLOCK, 128}, VKT_ACT_SWIGLU, "fp8 128x128 blocks"},
        {{VKT_SRC_MXFP4_E8M0, 32}, {VKT_SRC_MXFP4_E8M0, 32}, VKT_ACT_SITU, "MXFP4 ue8m0, SiTU-GLU"},
        {{VKT_SRC_BF16, 0}, {VKT_SRC_BF16, 0}, VKT_ACT_SWIGLU, "bf16"},
        {{VKT_SRC_MXFP4_E8M0, 32}, {VKT_SRC_MXFP4_E8M0, 32}, VKT_ACT_SWIGLU_V4, "MXFP4, DeepSeek V4 roundings"},
    };
    for (size_t c = 0; c < sizeof cs / sizeof *cs; c++) {
        int v4 = cs[c].act == VKT_ACT_SWIGLU_V4;
        model_make(cs[c].gu, cs[c].dn); g_act = cs[c].act; g_limit = 0;
        /* Eight residents at most, four warm; batch cases have eight upload slots
         * but only two RAM slots, so a group must continue after a partial load. */
        set_budget(c % 2 ? 16 : 12, cs[c].gu, cs[c].dn);
        setenv("COLI_VK_TIER_RATE", "0", 1);
        setenv("COLI_VK_TIER_STREAM_SLOTS", c % 2 ? "8" : "4", 1);
        setenv("COLI_VK_TIER_STREAM_ROWS", "3", 1);    /* a cold expert with 3 rows or more streams */
        setenv("COLI_VK_TIER_STREAM_HALF", "40", 1);   /* sub-batches of 40 rows: several a step, experts in parts */
        VktConfig vc = cfg_of(cs[c].gu, cs[c].dn, cs[c].act, v4 ? 10.0f : 0.f);
        vc.load = load_cb; vc.release = release_cb;
        if (c % 2) vc.load_batch = load_batch_cb;   /* half the cases: the batch hook too */
        g_batches = g_batch_left = g_partial_serial = 0;
        uint32_t hist[L][E], *rows[L];
        for (int l = 0; l < L; l++) { rows[l] = hist[l]; for (int e = 0; e < E; e++) hist[l][e] = e < 2 ? 100 - e : 0; }
        int on = vkt_init(&vc, rows);
        unsetenv("COLI_VK_TIER_STREAM_HALF");
        CHECK(on, "stream %s: the tier did not start", cs[c].name);
        if (!on) { model_free(); continue; }
        int pl[L * E], pe[L * E], np = vkt_plan(pl, pe, L * E);
        for (int i = 0; i < np; i++) { VktExpertSrc s = src_of(&ex[pl[i]][pe[i]]); vkt_put(pl[i], pe[i], &s); }
        vkt_put_done();
        uint8_t before[L][E];
        for (int l = 0; l < L; l++) for (int e = 0; e < E; e++) before[l][e] = (uint8_t)vkt_resident(l, e);
        Books bk = {0, 0}; unsigned long long streamed = 0, kept = 0; double worst = 0;
        g_loads = 0;
        static const int Ss[] = {48, 33, 1, 64, 2, 40};
        for (int t = 0; t < 6; t++)
            for (int l = 0; l < L; l++) {
                int S = Ss[t], idx[64 * K]; float w[64 * K];
                route(S, 0, E, idx, w);
                vkt_begin_forward();
                double r = stream_step(l, S, idx, w, v4, &bk, &streamed, &kept, t >= 3);
                if (r > worst) worst = r;
            }
        int same = 1;
        for (int l = 0; l < L; l++) for (int e = 0; e < E; e++) same &= before[l][e] == (uint8_t)vkt_resident(l, e);
        printf("  %-34s device %3llu of %3llu assignments (%llu streamed), %llu on the CPU, %d loads (%d batches), %d warm, worst %.2e\n",
               cs[c].name, bk.dev, bk.routed, streamed, kept, g_loads, g_batches, np, worst);
        vkt_report("test", 0, 0);
        CHECK(np == 4, "stream %s: %d warm experts", cs[c].name, np);
        CHECK(!(c % 2) || g_batches > 0, "stream %s: the batch hook was never called", cs[c].name);
        CHECK(g_partial_serial == 0, "stream %s: %d reads serialized after a partial batch", cs[c].name, g_partial_serial);
        CHECK(streamed > 0, "stream %s: nothing streamed", cs[c].name);
        CHECK(kept > 0, "stream %s: no cold expert stayed on the CPU", cs[c].name);
        CHECK(same, "stream %s: the resident set moved", cs[c].name);
        CHECK(worst < (v4 ? 2e-2 : 2e-3), "stream %s: device rows off the reference (%.3g)", cs[c].name, worst);
        vkt_shutdown(); model_free();
        unsetenv("COLI_VK_TIER_STREAM_SLOTS"); unsetenv("COLI_VK_TIER_STREAM_ROWS");
    }
}

/* A large requested step under a deliberately tiny scratch budget: reserve a
 * smaller batch before allocating, then keep the backend usable for the format
 * and streaming reference checks below. Reusing those buffers needs no growth. */
static void scratch_budget(void) {
    CHECK(coli_vk_xb_init(H, F, COLI_VK_ACT_SWIGLU, 0, 0, 0), "scratch budget: init failed");
    const size_t budget = 512 * 1024;
    ColiVkXbStats before, after; coli_vk_xb_stats(&before);
    int rows = coli_vk_xb_sub_fit(4096, 2, budget);
    CHECK(rows > 0 && rows < 4096, "scratch budget: got %d rows", rows);
    CHECK(rows > 0 && coli_vk_xb_sub_reserve(rows, 2), "scratch budget: reserve failed");
    coli_vk_xb_stats(&after);
    CHECK(after.scratch_bytes > before.scratch_bytes && after.scratch_bytes - before.scratch_bytes <= budget,
          "scratch budget: grew from %zu to %zu beyond %zu bytes", before.scratch_bytes, after.scratch_bytes, budget);
    CHECK(coli_vk_xb_sub_fit(rows, 2, 0) == rows, "scratch budget: existing buffers should need no extra bytes");
    printf("scratch budget: %d of 4096 rows, %zu additional bytes (limit %zu)\n", rows,
           after.scratch_bytes - before.scratch_bytes, budget);
}

/* A failed multi-expert upload may have copied only one projection. None of
 * that group may run; its complete CPU replay must be exact. Reusing the same
 * staging slots on the next step must replace every matrix before publishing. */
static void stream_commit_failure(const char *spv) {
    setenv("COLI_VK_STAGED", "1", 1);
    setenv("COLI_VK_STAGED_FAULT", "commit:1", 1);
    int ready = coli_vk_init(spv);
    CHECK(ready && coli_vk_staged(), "stream fault: staged device did not start");
    if (!ready) return;
    VktFmt f = {VKT_SRC_FP8_BLOCK, 128};
    model_make(f, f); g_act = VKT_ACT_SWIGLU; g_limit = 0;
    set_budget(12, f, f);
    setenv("COLI_VK_TIER_RATE", "0", 1);
    setenv("COLI_VK_TIER_STREAM_SLOTS", "4", 1);
    setenv("COLI_VK_TIER_STREAM_ROWS", "1", 1);
    VktConfig vc = cfg_of(f, f, VKT_ACT_SWIGLU, 0);
    vc.load = load_cb; vc.release = release_cb; vc.load_batch = load_batch_cb;
    int on = vkt_init(&vc, NULL);
    CHECK(on, "stream fault: tier did not start");
    if (on) {
        enum { S = 32 };
        int idx[S * K]; float w[S * K];
        for (int i = 0; i < S * K; i++) { idx[i] = i % 2; w[i] = 1.0f / K; }
        Books books = {0, 0}; unsigned long long streamed = 0, kept = 0;
        double error = stream_step(0, S, idx, w, 0, &books, &streamed, &kept, 0);
        CHECK(books.dev == 0 && kept == S * K && error == 0,
              "stream fault: partial upload used (GPU %llu, CPU %llu, error %.3g)", books.dev, kept, error);
        books = (Books){0, 0}; streamed = kept = 0;
        vkt_begin_forward();
        error = stream_step(0, S, idx, w, 0, &books, &streamed, &kept, 0);
        CHECK(books.dev == S * K && !kept && error < 2e-3,
              "stream fault: refill did not recover (GPU %llu, CPU %llu, error %.3g)", books.dev, kept, error);
        puts("stream fault: whole group replayed on CPU, staging slots refilled correctly");
    }
    vkt_shutdown(); model_free(); coli_vk_shutdown();
    unsetenv("COLI_VK_STAGED_FAULT");
    unsetenv("COLI_VK_TIER_STREAM_SLOTS"); unsetenv("COLI_VK_TIER_STREAM_ROWS");
}

int main(int argc, char **argv) {
    char buf[1024];
    const char *spv = argc > 1 ? argv[1] : coli_vk_shader_path(buf, sizeof buf);
    if (!coli_vk_init(spv)) { printf("FAIL: no Vulkan device (shaders %s)\n", spv); return 1; }
    setenv("COLI_VK_TIER_RESERVE_GB", "0", 1);
    scratch_budget();
    printf("formats:\n"); formats();
    printf("warm:\n"); warm();
    printf("adapt:\n"); adapt();
    printf("partial:\n"); partial();
    printf("sync:\n"); sync_evict();
    printf("extra:\n"); extra();
    printf("DeepSeek V4:\n"); v4_act();
    printf("stream:\n"); stream();
    ColiVkPoolStats ps, px; coli_vk_pool_stats(1, &ps); coli_vk_pool_stats(4, &px);
    CHECK(ps.live == 0 && px.live == 0, "%d + %d tier ranges still live after every shutdown", ps.live, px.live);
    coli_vk_shutdown();   /* with staged uploads: where the experts were ("[VK] memory at exit") */
    stream_commit_failure(spv);
    printf(fails ? "FAIL (%d)\n" : "PASS\n", fails);
    return fails != 0;
}
