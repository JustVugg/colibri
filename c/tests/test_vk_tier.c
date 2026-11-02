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
 *            start's before and after (streaming promotes nothing);
 *   layers   formats that change by layer (VktConfig.layer_gate_up/layer_down, a GGUF typed
 *            by layer): every device row equals the reference with the model resident,
 *            with a budget of four (the smaller form's experts take the rooms the bigger
 *            form's give up, no upload refused), and streamed through slots that change
 *            format between the layers' steps;
 *   timeout  a batch's wait gives up (COLI_VK_WAIT_FAULT): the tier stops and the experts
 *            that batch names keep their device memory until the device is idle;
 *   promote  a host pool that holds every expert (COLI_VK_TIER_HOST=1), the device four:
 *            the warm start puts every expert in the pool and the history's four on the
 *            device too; a batch step's routed pool experts move up to the device and the
 *            coldest there go back to their pool copy (no upload refused, every expert
 *            resident throughout, every device row the reference's, from either place);
 *            the dense chain's books (vkt_dev_count, then vkt_settle at a frame's start)
 *            move them the same way, the table whole at every read; with a pool short of
 *            one expert the device keeps the warm start's set;
 *   hybrid   GGUF block types (Q4_K, Q6_K down), then colibri's int4-g64 and int8, every
 *            expert in the host pool, the device
 *            four: in steps of up to 8 rows the pool's experts outside the device's share are
 *            computed by the CPU at the join (vkt_host_pairs, GQ_EXACT=1's float form) and
 *            the rest by the device, every row the reference's; a prompt step (20 rows) and
 *            COLI_VK_HYBRID=0 leave them all to the device;
 *   pool copy the same experts and the streaming's slots: a prompt step's host pool experts
 *            are copied into the slots on the device (coli_vk_xb_sub_copy: at the head of the
 *            batch with staged uploads, the CPU through the mapping otherwise) and computed
 *            there, those of at most COLI_VK_HYBRID_BIG_ROWS rows (2) by the CPU in place at
 *            the join, every row the reference's; COLI_VK_TIER_POOL_COPY=0 and
 *            COLI_VK_HYBRID_BIG_ROWS=0 read them in place;
 *   host pairs vkt_host_pairs on the CPU alone (Q4_K/Q6_K, IQ2_XXS/IQ3_XXS, colibri's int4-g64
 *            pairs, int8 a row, int3-g64, MXFP4 and FP8): the
 *            pairs of one expert computed together give the bits of each pair computed on its
 *            own, and the dequantized reference within the form's error. Here the float form
 *            (GQ_EXACT=1, set for the whole run: vk_tier.c reads it once); with the argument
 *            host-pairs, alone and before anything reads it, the integer form.
 *
 *   make tests/test_vk_tier VK=1 && VK_ICD_FILENAMES=.../lvp_icd.json ./tests/test_vk_tier */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include "../backend_vulkan.h"
#include "../vk_tier.h"
#include "../ggml_quants.h"   /* the block formats' CPU decoders, the reference of section blocks */

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fails++; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

enum { L = 2, E = 10, K = 3, HF_MAX = 512 };
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
        ColiVkXbStats x0, x1; coli_vk_xb_stats(&x0);
        static const int Ss[] = {1, 3, 20, 1, 2, 24};
        for (int t = 0; t < 6; t++)
            for (int l = 0; l < L; l++) {
                int idx[24 * K]; float w[24 * K];
                route(Ss[t], 0, E, idx, w);
                double r = step(l, Ss[t], idx, w, &bk);
                if (r > worst) worst = r;
            }
        coli_vk_xb_stats(&x1);   /* the steps of a few rows: the chain's kernels where the device has them */
        printf("  %-34s device %3llu of %3llu assignments, worst relative error %.2e, %llu batches on the chain's kernels\n",
               cs[c].name, bk.dev, bk.routed, worst, x1.chain_gemv_batches - x0.chain_gemv_batches);
        CHECK(bk.dev > 0, "%s: nothing ran on the device", cs[c].name);
        CHECK(worst < 2e-3, "%s: device rows off the reference (%.3g)", cs[c].name, worst);
        vkt_shutdown(); model_free();
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
static int lb_bits(VktSrc k) { return k == VKT_SRC_LB_I3G16 ? 3 : 2; }
/* vq2's codebooks for the test: per coordinate max|a| + max|b| = 127, as the converter scales them */
static int8_t g_vq_a[256][8], g_vq_b[256][8];
static void vq_books(void) {
    for (int j = 0; j < 8; j++) {
        int ta = 30 + (int)(rnd() % 70), tb = 127 - ta;
        for (int e = 0; e < 256; e++) {
            g_vq_a[e][j] = (int8_t)(e < 2 ? (e ? -ta : ta) : (int)(rnd() % (2 * ta + 1)) - ta);
            g_vq_b[e][j] = (int8_t)(e < 2 ? (e ? -tb : tb) : (int)(rnd() % (2 * tb + 1)) - tb);
        }
    }
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
    /* colibri low-bit: codes, then as "scales" the group scales (I/32 bytes a row, vq2 I/64)
     * and the f32 row scales; decoded here from lowbit_g16.h's and vq2_g32.h's descriptions */
    int bits = lb_bits(f.kind), g32 = f.kind == VKT_SRC_LB_I2G32, vq = f.kind == VKT_SRC_LB_VQ2;
    size_t rb = (size_t)I * bits / 8, qb = (size_t)I / (vq ? 64 : 32);
    m->codes = malloc(rb * O);
    for (size_t i = 0; i < rb * O; i++) m->codes[i] = (uint8_t)rnd();
    uint8_t *sc = malloc((size_t)O * (qb + 4)); m->scales = sc;
    for (size_t i = 0; i < (size_t)O * qb; i++) sc[i] = (uint8_t)(g32 ? 1 + rnd() % 255 : rnd());
    float *d = (float *)(sc + (size_t)O * qb);
    static const int l2[4] = {-10, -3, 3, 10}, l3[8] = {-26, -16, -9, -3, 3, 9, 16, 26};
    for (int o = 0; o < O; o++) {
        d[o] = 0.0005f + (rnd() % 100) / 100000.0f;
        const uint8_t *row = m->codes + (size_t)o * rb, *q = sc + (size_t)o * qb;
        for (int i = 0; vq && i < I; i++) {   /* vq2: index bytes a, b per 8, a nibble per 32 */
            const uint8_t *v = row + (size_t)(i / 8) * 2;
            int qs = (q[i / 64] >> (4 * ((i / 32) & 1))) & 15;
            m->w[(size_t)o * I + i] = d[o] * (float)qs * (float)(g_vq_a[v[0]][i % 8] + g_vq_b[v[1]][i % 8]);
        }
        for (int i = 0; !vq && i < I; i++) {
            const uint8_t *blk = row + (size_t)(i / 128) * (bits == 3 ? 48 : 32);
            int k = i % 128, c = (blk[k % 32] >> (2 * (k / 32))) & 3;
            if (bits == 3) c |= ((blk[32 + k % 16] >> (k / 16)) & 1) << 2;
            int qs = g32 ? q[i / 32] : (q[i / 32] >> (4 * ((i / 16) & 1))) & 15;
            m->w[(size_t)o * I + i] = d[o] * (float)qs * (float)(bits == 3 ? l3[c] : l2[c]);
        }
    }
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
        {-1, -1, "colibri int2-g32"}, {-2, -2, "colibri int2-g16"}, {-3, -3, "colibri int3-g16"},
        {-4, -4, "colibri vq2"},
    };
    vq_books();   /* before the first block-format tensor builds the device's table */
    CHECK(!coli_vk_set_vq2_books(&g_vq_a[0][0], &g_vq_b[0][0]), "vq2: the codebooks did not reach the table");
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
        int on = vkt_init(&vc, NULL);
        unsetenv("COLI_VK_TIER_SYNC");
        CHECK(on, "%s: the tier did not start", cs[c].name);
        if (!on) { model_free(); continue; }
        Books bk = {0, 0}; double worst = 0;
        ColiVkXbStats x0, x1; coli_vk_xb_stats(&x0);
        static const int Ss[] = {1, 3, 20, 1, 2, 24};
        for (int t = 0; t < 6; t++)
            for (int l = 0; l < L; l++) {
                int idx[24 * K]; float w[24 * K];
                route(Ss[t], 0, E, idx, w);
                double r = step(l, Ss[t], idx, w, &bk);
                if (r > worst) worst = r;
            }
        coli_vk_xb_stats(&x1);   /* the steps of a few rows: the chain's kernels where the device has them */
        printf("  %-34s device %3llu of %3llu assignments, worst relative error %.2e, %llu batches on the chain's kernels\n",
               cs[c].name, bk.dev, bk.routed, worst, x1.chain_gemv_batches - x0.chain_gemv_batches);
        CHECK(bk.dev > 0, "%s: nothing ran on the device", cs[c].name);
        CHECK(worst < 2e-3, "%s: device rows off the reference (%.3g)", cs[c].name, worst);
        vkt_shutdown(); model_free();
    }
    H = H0; F = F0;
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

/* #2081: the warm start fills the device whatever the history knows. No history: a
 * budget of 7 takes 7 experts, one layer after the other; a history's experts go
 * first; a budget that holds every expert plans every one; a layer without a row
 * (no experts) is never planned. */
static void fill(void) {
    VktFmt f = {VKT_SRC_I4U_PAIRS_GS, 64};
    model_make(f, f); g_act = VKT_ACT_SWIGLU; g_limit = 0;
    static const char *const what[4] = {"no history", "a history of two", "every expert", "a layer without experts"};
    for (int c = 0; c < 4; c++) {
        set_budget(c < 2 ? 7 : L * E, f, f);
        uint32_t hist[L][E], *rows[L];
        for (int l = 0; l < L; l++) {
            for (int e = 0; e < E; e++) hist[l][e] = c == 1 && e == 0 ? (uint32_t)(50 - l) : 0;
            rows[l] = c == 3 && l == 1 ? NULL : hist[l];
        }
        VktConfig vc = cfg_of(f, f, VKT_ACT_SWIGLU, 0);
        CHECK(vkt_init(&vc, rows), "fill %s: the tier did not start", what[c]);
        int pl[L * E], pe[L * E], n = vkt_plan(pl, pe, L * E);
        const int want = c < 2 ? 7 : c == 2 ? L * E : E;
        CHECK(n == want, "fill %s: planned %d experts, not %d", what[c], n, want);
        int per[L] = {0}, ok = 1;
        for (int i = 0; i < n; i++) per[pl[i]]++;
        if (c == 0) ok = per[0] - per[1] <= 1 && per[1] - per[0] <= 1;          /* one layer after the other */
        if (c == 1) ok = n >= 2 && pl[0] == 0 && pe[0] == 0 && pl[1] == 1 && pe[1] == 0;   /* the history first */
        if (c == 3) ok = per[1] == 0;                                                /* no row, no plan */
        CHECK(ok, "fill %s: the plan's order or layers are wrong", what[c]);
        for (int i = 0; i < n; i++) { VktExpertSrc src = src_of(&ex[pl[i]][pe[i]]); vkt_put(pl[i], pe[i], &src); }
        vkt_put_done();
        int res = 0; for (int i = 0; i < n; i++) res += vkt_resident(pl[i], pe[i]);
        CHECK(res == n, "fill %s: %d of %d planned experts resident", what[c], res, n);
        printf("  fill, %s: %d experts planned and resident\n", what[c], n);
        vkt_shutdown();
    }
    model_free();
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
        float x[H]; memset(x, 0, sizeof x);   /* H is a variable (grouped() widens it) */
        int id[K] = {0, 1, 2}; uint8_t taken[K];
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
            float dr[HF_MAX];
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
    if (!(r <= worst)) worst = r;
    if (getenv("VKT_TEST_DUMP") && n) fprintf(stderr, "dump: ref %g %g dev %g %g\n", ref[0], ref[1], out[0], out[1]);
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
        setenv("COLI_VK_TIER_FILL", "0", 1);           /* four warm, the rest cold: what streams */
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
        unsetenv("COLI_VK_TIER_STREAM_SLOTS"); unsetenv("COLI_VK_TIER_STREAM_ROWS"); unsetenv("COLI_VK_TIER_FILL");
    }
}

/* formats by layer: layer 0 Q4_K gate/up and Q5_K down (the main form), layer 1 the
 * bigger Q5_K and Q6_K, as a published GGUF gives a few layers */
static void layers(void) {
    const int H0 = H, F0 = F;
    H = 512; F = 256;
    VktFmt lgu[L], ldn[L];
    lgu[0] = (VktFmt){(VktSrc)(VKT_SRC_GGML + GQ_Q4_K), 0}; ldn[0] = (VktFmt){(VktSrc)(VKT_SRC_GGML + GQ_Q5_K), 0};
    lgu[1] = (VktFmt){(VktSrc)(VKT_SRC_GGML + GQ_Q5_K), 0}; ldn[1] = (VktFmt){(VktSrc)(VKT_SRC_GGML + GQ_Q6_K), 0};
    for (int mode = 0; mode < 3; mode++) {   /* resident, evicting, streaming */
        for (int l = 0; l < L; l++) for (int e = 0; e < E; e++) {
            mat_make_block(&ex[l][e].g, lgu[l], H, F); mat_make_block(&ex[l][e].u, lgu[l], H, F);
            mat_make_block(&ex[l][e].d, ldn[l], F, H);
        }
        g_act = VKT_ACT_SWIGLU; g_limit = 0;
        set_budget(mode == 0 ? L * E : 4, lgu[1], ldn[1]);   /* counted in the biggest form */
        setenv("COLI_VK_TIER_RATE", mode == 2 ? "0" : "64", 1);
        setenv("COLI_VK_TIER_SYNC", "1", 1);
        if (mode == 2) { setenv("COLI_VK_TIER_STREAM_SLOTS", "4", 1); setenv("COLI_VK_TIER_STREAM_ROWS", "3", 1);
                         setenv("COLI_VK_TIER_STREAM_HALF", "40", 1); }
        VktConfig vc = cfg_of(lgu[0], ldn[0], VKT_ACT_SWIGLU, 0);
        vc.layer_gate_up = lgu; vc.layer_down = ldn;
        if (mode == 2) { vc.load = load_cb; vc.release = release_cb; }
        uint32_t hist[L][E], *hr[L];
        for (int l = 0; l < L; l++) { hr[l] = hist[l]; for (int e = 0; e < E; e++) hist[l][e] = l == 0 && e < 4 ? 100 - e : 0; }
        int on = vkt_init(&vc, mode == 2 ? hr : NULL);
        unsetenv("COLI_VK_TIER_SYNC"); unsetenv("COLI_VK_TIER_STREAM_HALF");
        CHECK(on, "layers: the tier did not start");
        if (!on) { model_free(); break; }
        if (mode == 2) {   /* the warm start: layer 0's four, the slots then stream layer 1's experts */
            int pl[L * E], pe[L * E], np = vkt_plan(pl, pe, L * E);
            for (int i = 0; i < np; i++) { VktExpertSrc s = src_of(&ex[pl[i]][pe[i]]); vkt_put(pl[i], pe[i], &s); }
            vkt_put_done();
        }
        Books bk = {0, 0}; double worst = 0; unsigned long long streamed = 0, kept = 0;
        int res1 = 0;
        static const int Ss[2][6] = {{1, 3, 20, 1, 2, 24}, {48, 33, 1, 64, 2, 40}};
        if (mode == 1)   /* layer 1's (the bigger form's) experts settle, then layer 0's take their rooms (LFRU: a few
                          * hundred steps); a bigger form's never takes a smaller one's room, which would not hold it */
            for (int t = 0; t < 360; t++) {
                int lay = t < 60 ? 1 : 0, idx[4 * K]; float w[4 * K];
                route(4, 0, 5, idx, w);
                vkt_begin_forward();
                double r = step(lay, 4, idx, w, &bk);
                if (!(r <= worst)) worst = r;
            }
        else
            for (int t = 0; t < 6; t++)
                for (int l = 0; l < L; l++) {
                    int S = Ss[mode / 2][t], idx[64 * K]; float w[64 * K];
                    route(S, 0, E, idx, w);
                    vkt_begin_forward();
                    double r = mode == 2 ? stream_step(l, S, idx, w, 0, &bk, &streamed, &kept, 0) : step(l, S, idx, w, &bk);
                    if (!(r <= worst)) worst = r;
                }
        for (int e = 0; e < E; e++) res1 += vkt_resident(mode == 1 ? 0 : 1, e);   /* the budget of four: layer 0's */
        static const char *mn[3] = {"resident", "budget of four", "streaming"};
        printf("  Q4_K/Q5_K and Q5_K/Q6_K layers, %-15s device %3llu of %3llu (%llu streamed), %d of layer %d resident, worst %.2e\n",
               mn[mode], bk.dev, bk.routed, streamed, res1, mode == 1 ? 0 : 1, worst);
        unsigned long long ev, failed;
        report_counts(&ev, &failed);
        int counted = failed != ~0ull;   /* the report read back (not where stderr cannot be redirected: mingw) */
        CHECK(bk.dev > 0, "layers %s: nothing ran on the device", mn[mode]);
        CHECK(!counted || failed == 0, "layers %s: %llu uploads refused", mn[mode], failed);
        CHECK(worst < 2e-3, "layers %s: device rows off the reference (%.3g)", mn[mode], worst);
        if (mode == 1) CHECK(res1 > 0 && (!counted || ev > 0), "layers: no expert of the smaller form took a freed room of the "
                                                "bigger one (%d resident, %llu evictions)", res1, ev);
        if (mode == 2) CHECK(streamed > 0, "layers: nothing streamed");
        vkt_shutdown(); model_free();
        unsetenv("COLI_VK_TIER_STREAM_SLOTS"); unsetenv("COLI_VK_TIER_STREAM_ROWS");
    }
    H = H0; F = F0;
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

/* COLI_VK_WAIT_FAULT=1: the first batch's wait gives up at once, as a wait that runs out
 * of time does, and the batch may still be running (#1988). The tier stops; the experts
 * that batch names, freed by the tier's shutdown, keep their device ranges until the
 * device is idle (coli_vk_shutdown), not from the moment the wait gave up. */
static void wait_timeout(const char *spv) {
    setenv("COLI_VK_WAIT_FAULT", "1", 1);
    int ready = coli_vk_init(spv);
    CHECK(ready, "wait timeout: no device");
    if (!ready) { unsetenv("COLI_VK_WAIT_FAULT"); return; }
    VktFmt f = {VKT_SRC_I4U_PAIRS_GS, 64};
    model_make(f, f); g_act = VKT_ACT_SWIGLU; g_limit = 0;
    set_budget(L * E, f, f);
    VktConfig vc = cfg_of(f, f, VKT_ACT_SWIGLU, 0);
    int on = vkt_init(&vc, NULL);
    CHECK(on, "wait timeout: the tier did not start");
    if (on) {
        for (int e = 0; e < E; e++) { VktExpertSrc s = src_of(&ex[0][e]); vkt_put(0, e, &s); }
        vkt_put_done();
        enum { S = 4 };
        int idx[S * K]; float w[S * K]; uint8_t taken[S * K]; const float *dev[S * K];
        float x[S * HF_MAX];
        route(S, 0, E, idx, w);
        for (int i = 0; i < S * H; i++) x[i] = frnd();
        int n = vkt_issue(0, x, S, K, idx, taken);
        int joined = n ? vkt_join(dev) : -1;
        CHECK(n > 0 && joined == 0, "wait timeout: %d rows issued, join %d (the device should take them and the join fail)", n, joined);
        CHECK(vkt_devices() == 0, "wait timeout: the tier is still on");
        ColiVkPoolStats before, after; coli_vk_pool_stats(1, &before);
        vkt_shutdown();
        coli_vk_pool_stats(1, &after);
        CHECK(before.live > 0 && after.live == before.live && after.frees == before.frees,
              "wait timeout: the tier's shutdown freed experts under the batch (%d live ranges and %llu frees "
              "before it, %d and %llu after)", before.live, (unsigned long long)before.frees, after.live,
              (unsigned long long)after.frees);
        printf("  the wait gave up: tier stopped, %d expert ranges kept until the device is idle\n", after.live);
    }
    model_free(); coli_vk_shutdown();
    unsetenv("COLI_VK_WAIT_FAULT");
}

/* The expert batch's grouped GEMM (qmatmul_grp.comp) and a big step's whole-step
 * buffers (coli_vk_xb_step_*): a hidden width of 256 (multiples of 128 take the grouped
 * route), every format it computes (device fmt 1, 2, 4).
 *   single  steps of 32 rows x top-3 (96 assignments, over its 64) as one batch;
 *   whole   prompt steps of 40 and 64 rows with every expert resident: sub-batches
 *           read their x through the assignment map, write y in place, and
 *           vkt_join_sum returns the rows summed on the device, compared with the
 *           reference sum and with the engines' own loop over the device rows;
 *   mixed   the same steps with a budget of four experts: cold ones stream or stay on
 *           the CPU, no device sum (*sum NULL), the rows still right.
 * Inputs and weights are rounded to f16 on this route, hence the looser bound. A device
 * without it (no cooperative matrices at subgroup size 64 or bufferDeviceAddress, e.g.
 * Lavapipe) runs the same steps on the per-expert route and vkt_join_sum gives NULL. */
static double sum_step(int layer, int S, const int *idx, const float *w, Books *bk, int *summed) {
    float *x = calloc((size_t)S * H, sizeof(float)), *cpu = calloc((size_t)S * K * H, sizeof(float));
    float *ref = calloc((size_t)S * H, sizeof(float)), *loop = calloc((size_t)S * H, sizeof(float)), *y = malloc(sizeof(float) * H);
    uint8_t *taken = malloc((size_t)S * K); const float **dev = malloc(sizeof(*dev) * S * K);
    for (int i = 0; i < S * H; i++) x[i] = frnd();
    int n = vkt_issue_w(layer, x, S, K, idx, w, taken);
    for (int i = 0; i < S * K; i++) {
        if (taken[i]) continue;
        expert_ref(&ex[layer][idx[i]], x + (size_t)(i / K) * H, cpu + (size_t)i * H);
        VktExpertSrc s = src_of(&ex[layer][idx[i]]); vkt_note(layer, idx[i], &s);
    }
    const float *sum = NULL;
    int joined = n ? vkt_join_sum(dev, w, &sum) : 1;
    CHECK(joined, "grouped: join failed");
    double worst = 0;
    for (int s = 0; s < S; s++)
        for (int k = 0; k < K; k++) {
            int i = s * K + k;
            expert_ref(&ex[layer][idx[i]], x + (size_t)s * H, y);
            const float *c = taken[i] && joined ? dev[i] : cpu + (size_t)i * H;
            if (taken[i] && joined) { double r = rel(c, y, H); if (!(r <= worst)) worst = r; }   /* a NaN counts as the worst */
            for (int d = 0; d < H; d++) { ref[(size_t)s * H + d] += w[i] * y[d]; loop[(size_t)s * H + d] += w[i] * c[d]; }
            bk->routed++; bk->dev += taken[i] != 0;
        }
    double r = rel(loop, ref, S * H);
    if (r > worst) worst = r;
    if (sum) {
        (*summed)++;
        /* the device's sum: the reference's within the bound, the host loop's to rounding */
        double rs = rel(sum, ref, S * H), rl = rel(sum, loop, S * H);
        if (rs > worst) worst = rs;
        CHECK(rl < 1e-6, "grouped: the device sum is off the host loop over its rows (%.3g)", rl);
    }
    free(x); free(cpu); free(ref); free(loop); free(y); free(taken); free(dev);
    return worst;
}
static void grouped(void) {
    struct { VktFmt gu, dn; float limit; const char *name; } cs[] = {
        {{VKT_SRC_I8_ROW, 0}, {VKT_SRC_I8_ROW, 0}, 0, "int8 per row"},
        {{VKT_SRC_I8_AS_I4_ROW, 0}, {VKT_SRC_I4S_PAIRS_ROW, 0}, 0, "int4 per row"},
        {{VKT_SRC_I4U_PAIRS_GS, 64}, {VKT_SRC_I4U_PAIRS_GS, 64}, 2.5f, "int4 v+8 gs64, SwiGLU limit"},
        {{VKT_SRC_I4S_PAIRS_GS, 32}, {VKT_SRC_I4S_PAIRS_GS, 32}, 0, "signed int4 pairs gs32"},
        {{VKT_SRC_I4U_PLANAR64, 64}, {VKT_SRC_I4U_PLANAR64, 64}, 0, "planar int4-g64"},
    };
    int h0 = H; H = 256;
    int any_grouped = 0, any_summed = 0, any_gemv = 0;
    for (size_t c = 0; c < sizeof cs / sizeof *cs; c++) {
        for (int mode = 0; mode < 4; mode++) {   /* single, whole, mixed, decode */
            model_make(cs[c].gu, cs[c].dn);
            g_act = VKT_ACT_SWIGLU; g_limit = cs[c].limit;
            set_budget(mode == 2 ? 4 : L * E, cs[c].gu, cs[c].dn);
            setenv("COLI_VK_TIER_RATE", mode == 1 || mode == 2 ? "0" : "64", 1);
            setenv("COLI_VK_TIER_SYNC", "1", 1);
            if (mode == 1 || mode == 2) { setenv("COLI_VK_TIER_STREAM_SLOTS", "4", 1); setenv("COLI_VK_TIER_STREAM_ROWS", "6", 1);
                        setenv("COLI_VK_TIER_STREAM_HALF", "70", 1); }
            VktConfig vc = cfg_of(cs[c].gu, cs[c].dn, VKT_ACT_SWIGLU, cs[c].limit);
            if (mode == 1 || mode == 2) { vc.load = load_cb; vc.release = release_cb; }
            uint32_t hist[L][E], *hr[L];
            for (int l = 0; l < L; l++) { hr[l] = hist[l]; for (int e = 0; e < E; e++) hist[l][e] = 100 - e; }
            int on = vkt_init(&vc, mode == 1 || mode == 2 || mode == 3 ? hr : NULL);
            unsetenv("COLI_VK_TIER_SYNC"); unsetenv("COLI_VK_TIER_STREAM_HALF");
            CHECK(on, "grouped %s: the tier did not start", cs[c].name);
            if (!on) { model_free(); continue; }
            if (mode) {   /* the warm start: whole and decode = every expert, mixed = the four hottest */
                int pl[L * E], pe[L * E], np = vkt_plan(pl, pe, L * E);
                for (int i = 0; i < np; i++) { VktExpertSrc s = src_of(&ex[pl[i]][pe[i]]); vkt_put(pl[i], pe[i], &s); }
                vkt_put_done();
            }
            ColiVkXbStats a, b; coli_vk_xb_stats(&a);
            Books bk = {0, 0}; double worst = 0; int summed = 0;
            static const int Ss[4][4] = {{32, 32, 32, 32}, {40, 64, 40, 64}, {40, 64, 40, 64}, {1, 2, 1, 3}};
            for (int t = 0; t < 4; t++)
                for (int l = 0; l < L; l++) {
                    int S = Ss[mode][t], idx[64 * K]; float w[64 * K];
                    route(S, 0, E, idx, w);
                    vkt_begin_forward();
                    double r = sum_step(l, S, idx, w, &bk, &summed);
                    if (r > worst) worst = r;
                }
            coli_vk_xb_stats(&b);
            unsigned long long gb = b.grouped_batches - a.grouped_batches, gv = b.gemv_batches - a.gemv_batches,
                               cg = b.chain_gemv_batches - a.chain_gemv_batches;   /* subgroups of 32: the chain's decode kernel */
            static const char *mn[4] = {"single", "whole", "mixed", "decode"};
            printf("  %-30s %-6s device %3llu of %3llu, %3llu grouped GEMM / %3llu GEMV batches, %d summed on the device, worst %.2e\n",
                   cs[c].name, mn[mode], bk.dev, bk.routed, gb, gv, summed, worst);
            CHECK(bk.dev > 0, "grouped %s %s: nothing ran on the device", cs[c].name, mn[mode]);
            CHECK(worst < (gb ? 1e-2 : 2e-3), "grouped %s %s: off the reference (%.3g)", cs[c].name, mn[mode], worst);
            any_gemv |= gv > 0 || cg > 0;
            any_grouped |= gb > 0; any_summed |= summed > 0;
            vkt_shutdown(); model_free();
            unsetenv("COLI_VK_TIER_STREAM_SLOTS"); unsetenv("COLI_VK_TIER_STREAM_ROWS");
        }
    }
    printf("  grouped route %s, device sums %s\n", any_grouped ? "taken" : "not on this device",
           any_summed ? "taken" : "not on this device");
    CHECK(!any_grouped || any_summed, "grouped: the device took the grouped route but never summed a whole step");
    const char *gve = getenv("COLI_VK_XB_GEMV");
    CHECK(!any_grouped || any_gemv || (gve && *gve == '0'), "grouped: the device took the grouped GEMM but never a grouped GEMV (its own or the chain's)");
    H = h0;
}

/* vkt_report's line, read back from its stderr into out */
static void report_line(char *out, size_t n) {
    out[0] = 0;
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
    while (fgets(line, sizeof line, t)) { fputs(line, stderr); if (strstr(line, "] tier test test:")) snprintf(out, n, "%s", line); }
    fclose(t);
}
static unsigned long long after(const char *line, const char *key) {
    const char *p = strstr(line, key);
    return p ? strtoull(p + strlen(key), NULL, 10) : ~0ull;
}
/* the experts of each layer on the device's memory (not the host pool), as a mask */
static unsigned dev_mask(int l) {
    uint8_t host[E]; unsigned m = 0;
    if (vkt_dev_host(l, host) < 0) return ~0u;
    for (int e = 0; e < E; e++) if (vkt_resident(l, e) && !host[e]) m |= 1u << e;
    return m;
}
static void promote_case(void) {
    VktFmt f = {VKT_SRC_I8_ROW, 0};
    model_make(f, f); g_act = VKT_ACT_SWIGLU; g_limit = 0;
    const double eb = (double)vkt_expert_bytes(H, F, f, f) / 1073741824.0;
    char gb[64];
    setenv("COLI_VK_TIER_HOST", "1", 1);
    setenv("COLI_VK_TIER_SYNC", "1", 1);
    uint32_t hist[L][E], *rows[L];
    for (int l = 0; l < L; l++) { rows[l] = hist[l]; for (int e = 0; e < E; e++) hist[l][e] = e < 2 ? 100 : 1; }
    for (int c = 0; c < 2; c++) {   /* the pool: every expert, then one short of them */
        set_budget(4, f, f);
        snprintf(gb, sizeof gb, "%.12f", (c ? L * E - 0.5 : L * E + 0.5) * eb);
        setenv("COLI_VK_TIER_HOST_GB", gb, 1);
        VktConfig vc = cfg_of(f, f, VKT_ACT_SWIGLU, 0);
        int on = vkt_init(&vc, rows);
        CHECK(on, "promote: the tier did not start");
        if (!on) break;
        int pl[L * E], pe[L * E], n = vkt_plan(pl, pe, L * E);
        CHECK(n == L * E, "promote: planned %d experts of %d", n, L * E);
        for (int i = 0; i < n; i++) { VktExpertSrc src = src_of(&ex[pl[i]][pe[i]]); vkt_put(pl[i], pe[i], &src); }
        vkt_put_done();
        unsigned m0 = dev_mask(0), m1 = dev_mask(1);
        CHECK(m0 == 3u && m1 == 3u, "promote: the warm start's device set is %#x / %#x, not the history's 0 and 1", m0, m1);
        /* phase A: batch steps on experts 5..9 (5 and 6 the hottest) */
        Books bk = {0, 0}; double worst = 0;
        int idx[8 * K]; float w[8 * K];
        for (int t = 0; t < 40; t++)
            for (int l = 0; l < L; l++) { route(4, 5, 5, idx, w); double r = step(l, 4, idx, w, &bk); if (r > worst) worst = r; }
        unsigned a0 = dev_mask(0), a1 = dev_mask(1);
        int res = 0; for (int l = 0; l < L; l++) for (int e = 0; e < E; e++) res += vkt_resident(l, e);
        /* phase B: the dense chain's books on experts 2..4 (8 rows a step: they pass 5 and 6), the
         * table read at each frame */
        unsigned ep = vkt_epoch(); int whole = 1;
        uint32_t *tab = malloc((size_t)E * COLI_VK_XB_ENTRY * sizeof(uint32_t));
        for (int t = 0; t < 40 && tab; t++) {
            vkt_settle();
            for (int l = 0; l < L; l++) whole &= vkt_dev_table(l, tab) == E;
            for (int l = 0; l < L; l++) { route(8, 2, 3, idx, w); vkt_dev_count(l, 8, K, idx); }
        }
        vkt_settle();
        free(tab);
        unsigned b0 = dev_mask(0), b1 = dev_mask(1);
        char line[4096]; report_line(line, sizeof line);
        unsigned long long ups = after(line, "read from it, "), failed = after(line, "failed ");
        printf("  promote, %s: device sets %#x / %#x after the warm start, %#x / %#x after batch steps on 5..9 "
               "(device %llu of %llu routed, relative error %.2e), %#x / %#x after the chain's books on 2..4; "
               "%llu moved up, failed %llu, resident %d\n", c ? "a pool short of one expert" : "every expert in the pool",
               m0, m1, a0, a1, bk.dev, bk.routed, worst, b0, b1, ups == ~0ull ? 0 : ups, failed, res);
        CHECK(worst < 2e-3, "promote: relative error %.3g", worst);
        CHECK(failed == 0, "promote: %llu failed uploads", failed);
        CHECK(res == L * E && bk.dev == bk.routed, "promote: %d of %d resident, the device took %llu of %llu",
              res, L * E, bk.dev, bk.routed);
        CHECK(whole, "promote: a table read in the chain's books was not whole");
        if (!c) {
            CHECK(ups != ~0ull && ups >= 8, "promote: %llu experts moved up", ups);
            CHECK(!(a0 & 3u) && !(a1 & 3u) && (a0 & 0x60u) && (a1 & 0x60u),
                  "promote: after the batch steps on 5..9 the device holds %#x / %#x", a0, a1);
            CHECK(!((b0 | b1) & 0x3e0u) && (b0 & 0x1cu) && (b1 & 0x1cu),
                  "promote: after the chain's books on 2..4 the device holds %#x / %#x", b0, b1);
            CHECK(vkt_epoch() != ep, "promote: the epoch did not move");
        } else
            CHECK(ups == ~0ull && a0 == 3u && a1 == 3u && b0 == 3u && b1 == 3u,
                  "promote: a pool short of one expert moved them (%#x / %#x, %#x / %#x)", a0, a1, b0, b1);
        vkt_shutdown();
        ColiVkPoolStats ph; coli_vk_pool_stats(5, &ph);
        CHECK(ph.live == 0, "promote: %d host pool ranges still live after the shutdown", ph.live);
    }
    unsetenv("COLI_VK_TIER_HOST_GB"); unsetenv("COLI_VK_TIER_SYNC");
    setenv("COLI_VK_TIER_HOST", "0", 1);
    model_free();
}

/* the report's "hybrid: N steps, P host pool pairs": P, ~0 without the part */
static unsigned long long hy_pairs(const char *line) {
    const char *p = strstr(line, "| hybrid: ");
    if (!p) return ~0ull;
    p = strstr(p, "steps, ");
    return p ? strtoull(p + 7, NULL, 10) : ~0ull;
}
/* vkt_host_pairs alone: 12 pairs of three experts over five rows, all at once (an expert's
 * pairs grouped, its rows read once) and one by one; the same bits, and the float reference
 * within tol (the float form's or the integer form's error, whichever vk_tier.c computes) */
static void host_pairs_case(double tol) {
    enum { HP = 512, FP = 256, NE = 3, NR = 5, NP = 12 };
    static const int pe[NP] = {0, 1, 2, 0, 1, 0, 2, 1, 0, 2, 2, 1}, pr[NP] = {0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 4};
    /* a GGUF's block types, and colibri's int4-g64 pairs (fmt 4) and int8 a row (fmt 1), the
     * host pool's layouts of VKT_SRC_I4U_PAIRS_GS and VKT_SRC_I8_ROW */
    static const int ty[7][2] = {{GQ_Q4_K, GQ_Q6_K}, {GQ_IQ2_XXS, GQ_IQ3_XXS}, {-4, -4}, {-1, -1}, {-5, -5}, {-7, -7}, {-12, -12}};
    static const char *nm[7] = {"Q4_K/Q6_K", "IQ2_XXS/IQ3_XXS", "int4-g64 pairs", "int8 a row", "int3-g64", "MXFP4 g32", "FP8 g128"};
    static const VktFmt cf[7] = {{0, 0}, {0, 0}, {VKT_SRC_I4U_PAIRS_GS, 64}, {VKT_SRC_I8_ROW, 0}, {VKT_SRC_I3_G64, 64},
                                 {VKT_SRC_MXFP4_F32, 32}, {VKT_SRC_FP8_GS, 128}};
    for (int c = 0; c < 7; c++) {
        const int blk = ty[c][0] >= 0;
        VktFmt gu = blk ? (VktFmt){(VktSrc)(VKT_SRC_GGML + ty[c][0]), 0} : cf[c];
        VktFmt dn = blk ? (VktFmt){(VktSrc)(VKT_SRC_GGML + ty[c][1]), 0} : gu;
        const int gsv = blk || c == 3 ? 0 : cf[c].gs;
        Ex xe[NE]; VktHostExpert hv[NE], hp[NP]; const VktHostExpert *hx[NP]; const float *xr[NP];
        for (int e = 0; e < NE; e++) {
            if (blk) { mat_make_block(&xe[e].g, gu, HP, FP); mat_make_block(&xe[e].u, gu, HP, FP); mat_make_block(&xe[e].d, dn, FP, HP); }
            else { mat_make(&xe[e].g, gu, HP, FP); mat_make(&xe[e].u, gu, HP, FP); mat_make(&xe[e].d, dn, FP, HP); }
            hv[e] = (VktHostExpert){xe[e].g.codes, xe[e].u.codes, xe[e].d.codes, ty[c][0], ty[c][1],
                                    blk ? (size_t)gq_row_bytes(ty[c][0], HP) : row_bytes(gu.kind, HP),
                                    blk ? (size_t)gq_row_bytes(ty[c][1], FP) : row_bytes(dn.kind, FP),
                                    xe[e].g.scales, xe[e].u.scales, xe[e].d.scales, gsv, gsv};
        }
        float *x = malloc((size_t)NR * HP * sizeof(float)), *gs = malloc((size_t)NP * 2 * FP * sizeof(float)),
              *hs = malloc((size_t)NP * FP * sizeof(float)), *all = malloc((size_t)NP * HP * sizeof(float)),
              *one = malloc((size_t)NP * HP * sizeof(float));
        for (int i = 0; i < NR * HP; i++) x[i] = frnd();
        for (int p = 0; p < NP; p++) { hp[p] = hv[pe[p]]; hx[p] = &hp[p]; xr[p] = x + (size_t)pr[p] * HP; }
        int ok = vkt_host_pairs(hx, xr, NP, HP, FP, 0.f, 3, gs, hs, all);
        for (int p = 0; p < NP && ok; p++) ok = vkt_host_pairs(&hx[p], &xr[p], 1, HP, FP, 0.f, 2, gs, hs, one + (size_t)p * HP);
        CHECK(ok, "host pairs %s: vkt_host_pairs failed", nm[c]);
        int diff = ok ? memcmp(all, one, (size_t)NP * HP * sizeof(float)) != 0 : 1;
        double worst = 0;
        float *hr = malloc((size_t)FP * sizeof(float));
        for (int p = 0; p < NP && ok; p++) {   /* the float reference from the dequantized rows */
            const Ex *m = &xe[pe[p]]; const float *xi = xr[p]; double mx = 0, er = 0;
            for (int o = 0; o < FP; o++) {
                double g = 0, u = 0;
                for (int i = 0; i < HP; i++) { g += (double)m->g.w[(size_t)o * HP + i] * xi[i]; u += (double)m->u.w[(size_t)o * HP + i] * xi[i]; }
                hr[o] = (float)(g / (1.0 + exp(-g)) * u);
            }
            for (int o = 0; o < HP; o++) {
                double y = 0;
                for (int i = 0; i < FP; i++) y += (double)m->d.w[(size_t)o * FP + i] * hr[i];
                if (fabs(y) > mx) mx = fabs(y);
                if (fabs(y - all[(size_t)p * HP + o]) > er) er = fabs(y - all[(size_t)p * HP + o]);
            }
            if (mx > 0 && er / mx > worst) worst = er / mx;
        }
        printf("  %-16s %d pairs of %d experts: grouped %s one by one, worst relative error %.2e against the float reference\n",
               nm[c], NP, NE, diff ? "DIFFERS from" : "the same bits as", worst);
        CHECK(!diff, "host pairs %s: the grouped pairs are not the bits of each on its own", nm[c]);
        CHECK(worst < tol, "host pairs %s: relative error %.3g", nm[c], worst);
        free(x); free(gs); free(hs); free(all); free(one); free(hr);
        for (int e = 0; e < NE; e++) { mat_free(&xe[e].g); mat_free(&xe[e].u); mat_free(&xe[e].d); }
    }
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
    const double eb = (double)vkt_expert_bytes(H, F, gu, dn) / 1073741824.0;
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
        int on = vkt_init(&vc, rows);
        CHECK(on, "hybrid: the tier did not start");
        if (!on) break;
        int pl[L * E], pe[L * E], n = vkt_plan(pl, pe, L * E);
        for (int i = 0; i < n; i++) { VktExpertSrc src = src_of(&ex[pl[i]][pe[i]]); vkt_put(pl[i], pe[i], &src); }
        vkt_put_done();
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
        vkt_shutdown();
    }
    unsetenv("COLI_VK_HYBRID"); unsetenv("GQ_EXACT"); unsetenv("COLI_VK_TIER_PROMOTE");
    unsetenv("COLI_VK_TIER_HOST_GB"); unsetenv("COLI_VK_TIER_SYNC");
    setenv("COLI_VK_TIER_HOST", "0", 1);
    model_free();
    H = H0; F = F0;
}

/* DeepSeek V4's expert in the hybrid: MXFP4 ue8m0 experts, four on the device and the rest
 * in the host pool, the CPU computing the pool's pairs in place with the device's roundings
 * (vk_tier.c hy_act_v4, the route weight in the row). The float form (GQ_EXACT=1) against
 * expert_v4, the engine's arithmetic: most rows bit-identical once rounded to bf16, every
 * one close; then the same steps with COLI_VK_HYBRID=0. */
static void v4_hybrid_case(void) {
    VktFmt f = {VKT_SRC_MXFP4_E8M0, 32};
    model_make(f, f);
    const float limit = 10.0f;
    const double eb = (double)vkt_expert_bytes(H, F, f, f) / 1073741824.0;
    char gb[64];
    setenv("COLI_VK_TIER_HOST", "1", 1); setenv("COLI_VK_TIER_SYNC", "1", 1);
    setenv("COLI_VK_TIER_PROMOTE", "0", 1); setenv("GQ_EXACT", "1", 1);
    uint32_t hist[L][E], *hrows[L];
    for (int l = 0; l < L; l++) { hrows[l] = hist[l]; for (int e = 0; e < E; e++) hist[l][e] = e < 2 ? 100 : 1; }
    for (int c = 0; c < 2; c++) {   /* the hybrid on, then COLI_VK_HYBRID=0 */
        if (c) setenv("COLI_VK_HYBRID", "0", 1);
        set_budget(4, f, f);
        snprintf(gb, sizeof gb, "%.12f", (L * E + 0.5) * eb);
        setenv("COLI_VK_TIER_HOST_GB", gb, 1);
        VktConfig vc = cfg_of(f, f, VKT_ACT_SWIGLU_V4, limit);
        int on = vkt_init(&vc, hrows);
        CHECK(on, "v4 hybrid: the tier did not start");
        if (!on) break;
        int pl[L * E], pe[L * E], n = vkt_plan(pl, pe, L * E);
        for (int i = 0; i < n; i++) { VktExpertSrc src = src_of(&ex[pl[i]][pe[i]]); vkt_put(pl[i], pe[i], &src); }
        vkt_put_done();
        unsigned long long dev = 0, same = 0, routed = 0; double worst = 0;
        static const int Ss[] = {1, 3, 8, 1, 2, 8};
        for (int t = 0; t < 6; t++)
            for (int l = 0; l < L; l++) {
                int S = Ss[t], idx[8 * K]; float w[8 * K];
                route(S, 0, E, idx, w);
                float *xq = malloc(sizeof(float) * S * H), *y = malloc(sizeof(float) * H);
                uint8_t taken[8 * K]; const float *rows[8 * K];
                for (int i = 0; i < S * H; i++) xq[i] = frnd() * 4.0f;
                for (int s = 0; s < S; s++) qdq128(xq + (size_t)s * H, H);   /* the engine's rounding of x */
                int k = vkt_issue_w(l, xq, S, K, idx, w, taken);
                int joined = k ? vkt_join(rows) : 1;
                CHECK(joined, "v4 hybrid: join failed");
                for (int i = 0; i < S * K; i++) {
                    routed++;
                    if (!taken[i] || !joined) continue;
                    float dr[HF_MAX];
                    expert_v4(&ex[l][idx[i]], xq + (size_t)(i / K) * H, w[i], limit, y);
                    for (int d = 0; d < H; d++) dr[d] = bf16r(rows[i][d]);
                    double r = rel(dr, y, H); if (r > worst) worst = r;
                    same += !memcmp(dr, y, (size_t)H * sizeof(float));
                    dev++;
                }
                free(xq); free(y);
            }
        char line[4096]; report_line(line, sizeof line);
        unsigned long long pairs = hy_pairs(line);
        printf("  MXFP4 ue8m0 experts in the host pool, DeepSeek V4 roundings, the hybrid %s: %llu pairs computed by the CPU, "
               "%llu of %llu assignments taken, %llu rows bit-identical to the engine's arithmetic, worst row %.2e\n",
               c ? "off" : "on", pairs == ~0ull ? 0 : pairs, dev, routed, same, worst);
        CHECK(dev == routed, "v4 hybrid: the tier took %llu of %llu", dev, routed);
        CHECK(same * 2 > dev, "v4 hybrid: only %llu of %llu rows equal the engine's arithmetic", same, dev);
        CHECK(worst < 2e-2, "v4 hybrid: a row off the reference (%.3g)", worst);
        if (!c) CHECK(pairs != ~0ull && pairs > 0, "v4 hybrid: no host pool pair computed by the CPU");
        else CHECK(pairs == ~0ull, "v4 hybrid: COLI_VK_HYBRID=0 computed %llu pairs on the CPU", pairs);
        vkt_shutdown();
    }
    unsetenv("COLI_VK_HYBRID"); unsetenv("GQ_EXACT"); unsetenv("COLI_VK_TIER_PROMOTE");
    unsetenv("COLI_VK_TIER_HOST_GB"); unsetenv("COLI_VK_TIER_SYNC");
    setenv("COLI_VK_TIER_HOST", "0", 1);
    model_free();
}

/* the report's ", and S prompt steps' P pairs": P, 0 without the part */
static unsigned long long big_pairs(const char *line) {
    const char *p = strstr(line, "prompt steps' ");
    return p ? strtoull(p + 14, NULL, 10) : 0;
}
/* the report's "N host pool experts copied into the slots": N, 0 without the part */
static unsigned long long pc_count(const char *line) {
    const char *p = strstr(line, " host pool experts copied into the slots");
    if (!p) return 0;
    while (p > line && p[-1] >= '0' && p[-1] <= '9') p--;
    return strtoull(p, NULL, 10);
}
static void pool_copy_case(void) {
    const int H0 = H, F0 = F;
    H = 512; F = 256;   /* whole blocks of 256 */
    VktFmt gu = {(VktSrc)(VKT_SRC_GGML + GQ_Q4_K), 0}, dn = {(VktSrc)(VKT_SRC_GGML + GQ_Q6_K), 0};
    for (int l = 0; l < L; l++) for (int e = 0; e < E; e++) {
        mat_make_block(&ex[l][e].g, gu, H, F); mat_make_block(&ex[l][e].u, gu, H, F); mat_make_block(&ex[l][e].d, dn, F, H);
    }
    g_act = VKT_ACT_SWIGLU; g_limit = 0;
    const double eb = (double)vkt_expert_bytes(H, F, gu, dn) / 1073741824.0;
    char gb[64];
    setenv("COLI_VK_TIER_HOST", "1", 1);
    setenv("COLI_VK_TIER_SYNC", "1", 1);
    setenv("COLI_VK_TIER_PROMOTE", "0", 1);
    setenv("COLI_VK_TIER_RATE", "0", 1);
    setenv("COLI_VK_TIER_STREAM_SLOTS", "4", 1);
    setenv("COLI_VK_TIER_STREAM_HALF", "40", 1);   /* sub-batches of 40 rows: the slots refilled within a step */
    uint32_t hist[L][E], *rows[L];
    for (int l = 0; l < L; l++) { rows[l] = hist[l]; for (int e = 0; e < E; e++) hist[l][e] = e < 2 ? 100 : 1; }
    setenv("GQ_EXACT", "1", 1);   /* the CPU's pairs in its float form, against the float reference */
    for (int c = 0; c < 2; c++) {   /* the copies and the CPU's few-row pairs on, then both off */
        setenv("COLI_VK_HYBRID_BIG_ROWS", c ? "0" : "2", 1);
        if (c) setenv("COLI_VK_TIER_POOL_COPY", "0", 1);
        set_budget(8, gu, dn);
        snprintf(gb, sizeof gb, "%.12f", (L * E + 0.5) * eb);
        setenv("COLI_VK_TIER_HOST_GB", gb, 1);
        VktConfig vc = cfg_of(gu, dn, VKT_ACT_SWIGLU, 0);
        vc.load = load_cb; vc.release = release_cb;
        int on = vkt_init(&vc, rows);
        CHECK(on, "pool copy: the tier did not start");
        if (!on) break;
        int pl[L * E], pe[L * E], n = vkt_plan(pl, pe, L * E);
        for (int i = 0; i < n; i++) { VktExpertSrc src = src_of(&ex[pl[i]][pe[i]]); vkt_put(pl[i], pe[i], &src); }
        vkt_put_done();
        Books bk = {0, 0}; unsigned long long streamed = 0, kept = 0; double worst = 0;
        static const int Ss[] = {20, 33, 48, 20};
        int idx[48 * K]; float w[48 * K];
        for (int t = 0; t < 4; t++)
            for (int l = 0; l < L; l++) {
                route(Ss[t], 0, E, idx, w);
                vkt_begin_forward();
                double r = stream_step(l, Ss[t], idx, w, 0, &bk, &streamed, &kept, 0);
                if (r > worst) worst = r;
            }
        char line[4096]; report_line(line, sizeof line);
        unsigned long long copies = pc_count(line), cpu = big_pairs(line);
        printf("  Q4_K/Q6_K experts in the host pool, %s: %llu copied into the slots, %llu pairs on the CPU in place, "
               "%llu of %llu assignments taken, worst relative error %.2e\n", c ? "both off" : "the copies on, 2 rows",
               copies, cpu, bk.dev, bk.routed, worst);
        CHECK(worst < 2e-3, "pool copy: relative error %.3g", worst);
        CHECK(bk.dev == bk.routed, "pool copy: the device took %llu of %llu", bk.dev, bk.routed);
        if (!c) CHECK(copies > 0 && cpu > 0, "pool copy: %llu copied into a slot, %llu pairs on the CPU", copies, cpu);
        else CHECK(copies == 0 && cpu == 0, "pool copy: off, yet %llu copied and %llu pairs on the CPU", copies, cpu);
        vkt_shutdown();
    }
    unsetenv("COLI_VK_TIER_POOL_COPY"); unsetenv("COLI_VK_TIER_PROMOTE"); unsetenv("COLI_VK_TIER_RATE");
    unsetenv("COLI_VK_HYBRID_BIG_ROWS"); unsetenv("GQ_EXACT");
    unsetenv("COLI_VK_TIER_STREAM_SLOTS"); unsetenv("COLI_VK_TIER_STREAM_HALF");
    unsetenv("COLI_VK_TIER_HOST_GB"); unsetenv("COLI_VK_TIER_SYNC");
    setenv("COLI_VK_TIER_HOST", "0", 1);
    model_free();
    H = H0; F = F0;
}

/* The probe of the host-visible window (vk_window_decide, place_decide): COLI_VK_WINDOW_PROBE_MB=n says what the
 * type took, on any device. Under 9/10 of its heap the device takes the staged path, at or over it the mapped one,
 * and COLI_VK_STAGED=0 or 1 decides whatever the window is. The last two rows run the real loop
 * (COLI_VK_WINDOW_ASK_MB asks a few GiB at most): one the device gives, one refused past COLI_VK_DEVICE_CAP_MB.
 * A device that is staged on its own account (no Resizable BAR, COLI_VK_HOST_VISIBLE_CAP_MB) never gets to the
 * probe: the case skips the rows that want the mapped path. */
static void window_case(const char *spv) {
    /* probe: COLI_VK_WINDOW_PROBE_MB, the answer without asking; ask: COLI_VK_WINDOW_ASK_MB, how much the real loop asks;
     * cap: COLI_VK_DEVICE_CAP_MB, which refuses the blocks past it as a full card does */
    static const struct { const char *staged, *probe, *ask, *cap; int want; const char *what; } t[] = {
        {NULL, "1", NULL, NULL, 1, "a window of 1 MiB"},
        {NULL, "100000000", NULL, NULL, 0, "a window of the whole heap and more"},
        {"0", "1", NULL, NULL, 0, "COLI_VK_STAGED=0 over a window of 1 MiB"},
        {"1", "100000000", NULL, NULL, 1, "COLI_VK_STAGED=1 over a window of the whole heap"},
        {NULL, NULL, "1024", NULL, 0, "a real probe of 1 GiB, which the device gives"},
        {NULL, NULL, "3072", "2048", 1, "a real probe of 3 GiB under a device cap of 2 GiB, which refuses the third"}};
    static const char *const var[] = {"COLI_VK_STAGED", "COLI_VK_WINDOW_PROBE_MB", "COLI_VK_WINDOW_ASK_MB", "COLI_VK_DEVICE_CAP_MB"};
    char *was[4];
    for (int k = 0; k < 4; k++) { const char *v = getenv(var[k]); was[k] = v ? strdup(v) : NULL; }
    for (size_t i = 0; i < sizeof t / sizeof *t; i++) {
        const char *set[4] = {t[i].staged, t[i].probe, t[i].ask, t[i].cap};
        for (int k = 0; k < 4; k++) { if (set[k]) setenv(var[k], set[k], 1); else unsetenv(var[k]); }
        int ready = coli_vk_init(spv);
        CHECK(ready, "window, %s: no device", t[i].what);
        if (!ready) continue;
        const int staged = coli_vk_staged();
        if (!t[i].want && !t[i].staged && staged) {   /* staged on its own account, whatever the window: nothing to tell apart */
            printf("  %s: skipped, the device takes staged uploads\n", t[i].what);
            coli_vk_shutdown();
            continue;
        }
        CHECK(staged == t[i].want, "window, %s: the device is %s, wanted %s", t[i].what,
              staged ? "staged" : "mapped", t[i].want ? "staged" : "mapped");
        printf("  %s: %s\n", t[i].what, staged ? "staged" : "mapped");
        coli_vk_shutdown();
    }
    for (int k = 0; k < 4; k++) { if (was[k]) { setenv(var[k], was[k], 1); free(was[k]); } else unsetenv(var[k]); }
}

int main(int argc, char **argv) {
    char buf[1024];
    const char *spv = argc > 1 ? argv[1] : coli_vk_shader_path(buf, sizeof buf);
    if (argc > 2 && !strcmp(argv[2], "host-pairs")) {   /* the CPU's integer form: no device, GQ_EXACT unread */
        unsetenv("GQ_EXACT");
        printf("host pairs, the integer form:\n"); host_pairs_case(5e-2);
        printf(fails ? "FAIL (%d)\n" : "PASS\n", fails);
        return fails != 0;
    }
    /* a device that shares the host's RAM (Lavapipe) has a type for the host pool only with
     * COLI_VK_TIER_HOST=1 at the device's start (backend_vulkan.c): promote_case's pool */
    int host_set = getenv("COLI_VK_TIER_HOST") != NULL;
    if (!host_set) setenv("COLI_VK_TIER_HOST", "1", 1);
    /* the CPU's pairs of the hybrid steps (any case with a host pool: GGUF block types and
     * colibri's int8 and int4) in their float form, against the float references; vk_tier.c
     * reads GQ_EXACT once, so before any case (the integer form: the argument host-pairs) */
    setenv("GQ_EXACT", "1", 1);
    if (!coli_vk_init(spv)) { printf("FAIL: no Vulkan device (shaders %s)\n", spv); return 1; }
    setenv("COLI_VK_TIER_RESERVE_GB", "0", 1);
    /* the budgets, evictions and promotions below are the device's: the host pool (a
     * discrete card's system RAM) would take what the device leaves */
    if (!host_set) setenv("COLI_VK_TIER_HOST", "0", 1);
    scratch_budget();
    printf("formats:\n"); formats();
    printf("blocks:\n"); blocks();
    printf("grouped:\n"); grouped();
    printf("warm:\n"); warm();
    printf("adapt:\n"); adapt();
    printf("partial:\n"); partial();
    printf("sync:\n"); sync_evict();
    printf("extra:\n"); extra();
    printf("fill:\n"); fill();   /* after extra: its exact pool capacities assume no tier came before with more experts */
    printf("DeepSeek V4:\n"); v4_act();
    printf("stream:\n"); stream();
    printf("layers:\n"); layers();
    printf("promote:\n"); promote_case();
    printf("hybrid:\n");
    hybrid_case((VktFmt){(VktSrc)(VKT_SRC_GGML + GQ_Q4_K), 0}, (VktFmt){(VktSrc)(VKT_SRC_GGML + GQ_Q6_K), 0}, "Q4_K/Q6_K");
    hybrid_case((VktFmt){VKT_SRC_I4U_PAIRS_GS, 64}, (VktFmt){VKT_SRC_I4U_PAIRS_GS, 64}, "int4-g64");
    hybrid_case((VktFmt){VKT_SRC_I8_ROW, 0}, (VktFmt){VKT_SRC_I8_ROW, 0}, "int8");
    v4_hybrid_case();
    printf("pool copy:\n"); pool_copy_case();
    printf("host pairs:\n"); host_pairs_case(2e-3);
    ColiVkPoolStats ps, px; coli_vk_pool_stats(1, &ps); coli_vk_pool_stats(4, &px);
    CHECK(ps.live == 0 && px.live == 0, "%d + %d tier ranges still live after every shutdown", ps.live, px.live);
    coli_vk_shutdown();   /* with staged uploads: where the experts were ("[VK] memory at exit") */
    stream_commit_failure(spv);
    printf("wait timeout:\n"); wait_timeout(spv);
    printf("window:\n"); window_case(spv);
    printf(fails ? "FAIL (%d)\n" : "PASS\n", fails);
    return fails != 0;
}
