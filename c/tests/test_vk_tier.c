/* The Vulkan routed-expert tier (vk_tier.c) against a CPU reference, on any Vulkan
 * device (CI: Lavapipe). A small synthetic MoE -- L layers of E experts, top-K --
 * whose weights are random in every source format the tier accepts, decoded here
 * on their own, independently of the tier and of the backend. Gates:
 *   formats  every VktSrc kind and both activations (uploads awaited, COLI_VK_TIER_SYNC):
 *            each row the device returns
 *            equals the reference expert output, and the rank-order sum of a mixed
 *            step (device and CPU rows) equals the all-CPU sum, within 2e-3;
 *   warm     a history fills the budget in heat order (vkt_plan / vkt_put);
 *   hot set  COLI_VK_TIER_HOT: the history's top share of the residents pinned against
 *            every victim path -- the pins survive a promotion storm that evicts the rest
 *            (the same storm evicts them with the knob unset), a fully pinned tier refuses
 *            promotions and still serves, COLI_VK_TIER_PIN_LAYER's layer pins whole and
 *            holds, and the ranking is global on the RAW counts (HOT_SCOPE=layer and
 *            HOT_HASH the variants; the scaled 1..31 heat would make it a per-layer film);
 *   pool     COLI_VK_TIER_POOL_LAYER (the dmoe arm's Stage C lever): the named layer's
 *            experts planned into the host pool ahead of the history, resident there and
 *            counted by vkt_dev_table whole (the dmoe gate) with no device place taken,
 *            staying put through the books' promotions, and a pool one place short of them
 *            placing none (all or nothing, like the gate);
 *   bank     COLI_VK_TIER_BANK_LAYER (the VRAM bank): the stream slots carved for the
 *            banked layer (the ledger line's slot pair), its experts planned whole to
 *            the device ahead of the history and counted by vkt_dev_table (the dmoe gate),
 *            the at-most-once refill of what the warm start left un-read through the load
 *            hooks, the layer-switch rotation (the bitmap clears, the old layer's places
 *            free, the new layer's routed experts refill at most once, both ways), and
 *            the declines (out of range, no room): one line each, the plan the history's;
 *   dgrow    COLI_VK_TIER_DECODE_GROW: the resident set grows over a decode sequence by
 *            the decode-only rank (VktConfig.decode_heat, warmer than the merged history
 *            the warm start filled by) -- free VRAM first, then stream slots retired a
 *            place at a time, the budget and the rank cutoff respected both ways (the
 *            below-cutoff routed expert never installs), and the env unset growing
 *            nothing;
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
 *   ring     COLI_VK_TIER_STREAM_RING, the fork's ring on the port's slots: with the env
 *            off the ring's reader is never called (the default's own leg); with it on,
 *            every step's rows byte-equal the inline prefetch's (the ring changes
 *            timing, never bytes) through cut and refusing readers alike (the partial
 *            contract and the inline fallback); the pipeline hands the next layer's
 *            predicted set (T.pred) to the worker in the prediction's order and the
 *            step consumes it without one inline load; a step of another layer runs
 *            beside a parked job (the join waits only for what it needs) while the next
 *            prefetch takes it; COLI_VK_TIER_SYNC's quiescent point drains the ring;
 *            the teardown cancels between reads and never waits out the rest;
 *            COLI_VK_TIER_STREAM_RING_COLD fills the cold branch too: a next layer with
 *            no prediction behind it is planned by its heat (the whole non-resident set,
 *            the st_per cap) and the step takes it from the slots -- the routed expert
 *            below the cap stays the inline fallback's, the default hands nothing over
 *            and fills all four inline, and where a prediction exists the plan is the
 *            prediction's, COLD armed or not;
 *   layers   formats that change by layer (VktConfig.layer_gate_up/layer_down, a GGUF typed
 *            by layer): every device row equals the reference with the model resident,
 *            with a budget of four (the smaller form's experts take the rooms the bigger
 *            form's give up, no upload refused), and streamed through slots that change
 *            format between the layers' steps;
 *   host     the pool in system RAM the device reads over PCIe, sized from the engine's
 *            RAM plan (VktConfig.ram_planned/ram_budget/ram_spare): the plan's room
 *            bounds it (vkt_host_room's arithmetic, no device), a plan that leaves
 *            nothing turns it off with a line that says so, and COLI_VK_TIER_HOST_GB
 *            wins over the plan with one warning;
 *   timeout  a batch's wait gives up (COLI_VK_WAIT_FAULT): the tier stops and the experts
 *            that batch names keep their device memory until the device is idle;
 *   promote2 a host pool SHORT of every expert (COLI_VK_TIER_HOST_GB caps it at half),
 *            COLI_VK_TIER_PROMOTE on: a routed pooled expert moves up to the device and
 *            the coldest resident gives way (its device copy freed: no pool copy to take
 *            back), every row still the reference's and the device count inside its budget;
 *            by default (PROMOTE unset) a partial pool promotes nothing
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
#include <time.h>
#ifndef _WIN32
#include <sys/mman.h>   /* the VA-reuse leg's MAP_FIXED remap: a deterministic same-address realloc */
#endif
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
/* gap-wave4: the suite pins and restores the default levers' envs parent-transparently.
 * The strict gate's hatch case runs the suite once with every lever at =0 and once
 * with COLI_VK_LEGACY=1 and byte-compares: a case's unsetenv would erase the caller's
 * value, and the unset resolution then differs between the two legs (=0's parent is
 * "off by value", LEGACY's is "off by hatch") -- so every case that touches these
 * envs saves them at entry and restores them at exit ("" = unset). The suite's own
 * standing between cases is whatever the caller brought, exactly as before the flip. */
static char g_pin_experts[64], g_pin_dg[64], g_pin_dgh[64], g_pin_dgt[64], g_pin_dggb[64], g_pin_dgr[64];
static const char *env_or_empty(const char *n) { const char *e = getenv(n); return e ? e : ""; }
static void lever_pin_save(void) {
    snprintf(g_pin_experts, sizeof g_pin_experts, "%s", env_or_empty("COLI_VK_TIER_EXPERTS"));
    snprintf(g_pin_dg, sizeof g_pin_dg, "%s", env_or_empty("COLI_VK_TIER_DECODE_GROW"));
    snprintf(g_pin_dgh, sizeof g_pin_dgh, "%s", env_or_empty("COLI_VK_TIER_DECODE_GROW_HEADROOM"));
    snprintf(g_pin_dgt, sizeof g_pin_dgt, "%s", env_or_empty("COLI_VK_TIER_DECODE_GROW_TARGET"));
    snprintf(g_pin_dggb, sizeof g_pin_dggb, "%s", env_or_empty("COLI_VK_TIER_DECODE_GROW_GB"));
    snprintf(g_pin_dgr, sizeof g_pin_dgr, "%s", env_or_empty("COLI_VK_TIER_DECODE_GROW_RESERVE_GB"));
}
static void lever_pin_restore(void) {
    if (g_pin_experts[0]) setenv("COLI_VK_TIER_EXPERTS", g_pin_experts, 1); else unsetenv("COLI_VK_TIER_EXPERTS");
    if (g_pin_dg[0]) setenv("COLI_VK_TIER_DECODE_GROW", g_pin_dg, 1); else unsetenv("COLI_VK_TIER_DECODE_GROW");
    if (g_pin_dgh[0]) setenv("COLI_VK_TIER_DECODE_GROW_HEADROOM", g_pin_dgh, 1); else unsetenv("COLI_VK_TIER_DECODE_GROW_HEADROOM");
    if (g_pin_dgt[0]) setenv("COLI_VK_TIER_DECODE_GROW_TARGET", g_pin_dgt, 1); else unsetenv("COLI_VK_TIER_DECODE_GROW_TARGET");
    if (g_pin_dggb[0]) setenv("COLI_VK_TIER_DECODE_GROW_GB", g_pin_dggb, 1); else unsetenv("COLI_VK_TIER_DECODE_GROW_GB");
    if (g_pin_dgr[0]) setenv("COLI_VK_TIER_DECODE_GROW_RESERVE_GB", g_pin_dgr, 1); else unsetenv("COLI_VK_TIER_DECODE_GROW_RESERVE_GB");
}
static double rel(const float *a, const float *b, int n) {
    double num = 0, den = 0;
    for (int i = 0; i < n; i++) { double d = (double)a[i] - b[i]; num += d * d; den += (double)b[i] * b[i]; }
    return den > 0 ? sqrt(num / den) : sqrt(num);
}
/* rel()'s known hole (AJ, the v4 case's compare): a NaN never raises a `>` bar, so a
 * non-finite row reads as passing it. The new legs take their worst through this -- a
 * non-finite error becomes infinity and fails any threshold. */
static double no_nan(double r) { return r <= 1e300 ? r : INFINITY; }

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
    /* the armed default (gap-wave4): a rows-1 step defers to the finish site, so the
     * helper owes the group exactly as the engine's token finish does (vk_tier.c's
     * own contract); took == 0 there means nothing flew -- every selection the CPU's */
    int def = S == 1 && vkt_cs_armed();
    int grp = def ? vkt_issue_group(layer, x, S, K, idx, NULL, NULL, taken) : 0;
    for (int i = 0; i < S * K; i++) {
        if (taken[i]) continue;
        expert_ref(&ex[layer][idx[i]], x + (size_t)(i / K) * H, cpu + (size_t)i * H);
        VktExpertSrc s = src_of(&ex[layer][idx[i]]); vkt_note(layer, idx[i], &s);
    }
    int joined = grp ? vkt_join(dev) : (def ? 1 : (n ? vkt_join(dev) : 1));
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
                       .max_rows = 1 << 20,
                       /* the helper owes the deferred group at its finish (step() calls
                        * vkt_issue_group exactly as an engine's token finish does) */
                       .cs_finish = 1};
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

/* COLI_VK_TIER_PIN_LAYER (test-only, the COLI_VK_DMOE arm's scaffolding): the pinned
 * layer's experts planned ahead of the history, whatever the heat says, and resident
 * after the warm start so vkt_dev_table counts them all (vk_dmoe.h's all-or-nothing
 * gate); without the lever the plan is the history's, as before. */
static void pin_layer(void) {
    VktFmt f = {VKT_SRC_I4U_PAIRS_GS, 64};
    model_make(f, f); g_act = VKT_ACT_SWIGLU; g_limit = 0;
    set_budget(E, f, f);                     /* one layer's experts fit, of the two */
    uint32_t hist[L][E], *rows[L];           /* layer 0 the hot one: the pin must override it */
    for (int l = 0; l < L; l++) { rows[l] = hist[l]; for (int e = 0; e < E; e++) hist[l][e] = l ? 1u : 1000u; }
    VktConfig vc = cfg_of(f, f, VKT_ACT_SWIGLU, 0);
    setenv("COLI_VK_TIER_PIN_LAYER", "1", 1);
    CHECK(vkt_init(&vc, rows), "pin: the tier did not start");
    int pl[L * E], pe[L * E], n = vkt_plan(pl, pe, L * E);
    int per[L] = {0}, first = 1;
    for (int i = 0; i < n; i++) { per[pl[i]]++; first &= pl[i] == 1; }
    CHECK(n == E && per[1] == E && per[0] == 0 && first, "pin: planned %d experts (layer counts %d/%d), not layer 1's %d first",
          n, per[0], per[1], E);
    for (int i = 0; i < n; i++) { VktExpertSrc s = src_of(&ex[pl[i]][pe[i]]); vkt_put(pl[i], pe[i], &s); }
    vkt_put_done();
    uint32_t *tab = malloc((size_t)E * COLI_VK_XB_ENTRY * sizeof(uint32_t));
    CHECK(tab && vkt_dev_table(1, tab) == E, "pin: the tier does not hold all %d of layer 1 for the dmoe gate", E);
    int res = 0; for (int i = 0; i < n; i++) res += vkt_resident(pl[i], pe[i]);
    CHECK(res == n, "pin: %d of %d pinned experts resident", res, n);
    printf("  pin layer: %d of %d experts of layer 1 planned first and resident, the table whole\n", res, E);
    free(tab);
    vkt_shutdown();
    /* without the lever (the default): the history's hottest layer takes the budget */
    unsetenv("COLI_VK_TIER_PIN_LAYER");
    CHECK(vkt_init(&vc, rows), "pin: the tier did not start again");
    n = vkt_plan(pl, pe, L * E);
    per[0] = per[1] = 0;
    for (int i = 0; i < n; i++) per[pl[i]]++;
    CHECK(n == E && per[0] == E && per[1] == 0, "pin: the default plan took layer 1 (%d/%d of %d)", per[0], per[1], n);
    vkt_shutdown();
    model_free();
}
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
    /* gap-wave4: the case's own subject is the extra layers' residency -- an armed
     * compute site defers their rows-1 steps too, so the =0 half pins the standing
     * path this case asserts (the unset default runs it through the cs cases) */
    lever_pin_save();
    setenv("COLI_VK_TIER_EXPERTS", "0", 1);
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
    lever_pin_restore();   /* the caller's standing back: the armed default, or the strict legs' =0 / LEGACY=1 */
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
            int def = S == 1 && vkt_cs_armed();   /* the armed default defers: the finish site's group (the engine's order) */
            int grp = def ? vkt_issue_group(l, xq, S, K, idx, w, NULL, taken) : 0;
            for (int i = 0; i < S * K; i++) {
                if (taken[i]) continue;
                expert_v4(&ex[l][idx[i]], xq + (size_t)(i / K) * H, w[i], limit, cpu + (size_t)i * H);
                VktExpertSrc src = src_of(&ex[l][idx[i]]); vkt_note(l, idx[i], &src);
            }
            int joined = grp ? vkt_join(rows) : (def ? 1 : (n ? vkt_join(rows) : 1));
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
/* The load-path contract's own hooks (stream_batch): a loader that counts its holds so a
 * leak fails the test, and a batch reader that returns a cut prefix -- g_cut experts of
 * the n asked, a partial batch whenever the group was bigger -- the way an engine whose
 * reads run through a bounded pool of stage buffers must. */
static int g_holds_live, g_cut = 1, g_cut_partial;
/* COLI_VK_TIER_STREAM_GROUP's own reader: records the widths it was asked (a fill group's
 * gather is the ask) -- the schedule's own evidence, the loads all succeed. */
static int g_group_ask_max, g_group_asks; static unsigned long long g_group_ask_rows;
static int load_group_cb(void *ctx, int layer, const int *eids, int n, VktExpertSrc *srcs, void **hs) {
    (void)ctx;
    if (n > 64) n = 64;
    g_group_asks++;
    g_group_ask_rows += (unsigned long long)n;
    if (n > g_group_ask_max) g_group_ask_max = n;
    for (int i = 0; i < n; i++) { srcs[i] = src_of(&ex[layer][eids[i]]); hs[i] = &ex[layer][eids[i]]; }
    return n;
}
static int load_live_cb(void *ctx, int layer, int eid, VktExpertSrc *src, void **h) {
    (void)ctx; *src = src_of(&ex[layer][eid]); *h = &ex[layer][eid]; g_holds_live++; return 1;
}
static void release_live_cb(void *ctx, void *h) { (void)ctx; (void)h; g_holds_live--; }
static int load_cut_cb(void *ctx, int layer, const int *eids, int n, VktExpertSrc *srcs, void **h) {
    (void)ctx;
    int got = n < g_cut ? n : g_cut;
    if (got < n) g_cut_partial++;
    for (int i = 0; i < got; i++) { srcs[i] = src_of(&ex[layer][eids[i]]); h[i] = &ex[layer][eids[i]]; g_holds_live++; }
    return got;
}
/* One big step (vk_tier.h step 3, with the prefetch while "attention" runs); the device
 * rows of cold experts counted apart. capture (optional) receives the step's output
 * rows as they were summed, for the load-path tests' byte-exact comparisons. */
static double stream_step(int layer, int S, const int *idx, const float *w, int v4, Books *bk,
                          unsigned long long *streamed, unsigned long long *kept, int prefetch,
                          float *capture) {
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
    int def = S == 1 && vkt_cs_armed();   /* the armed default defers: the finish site's group (the engine's order) */
    int grp = def ? vkt_issue_group(layer, xq, S, K, idx, v4 ? w : NULL, NULL, taken) : 0;
    for (int i = 0; i < S * K; i++) {
        if (taken[i]) continue;
        if (v4) expert_v4(&ex[layer][idx[i]], xq + (size_t)(i / K) * H, w[i], 10.0f, cpu + (size_t)i * H);
        else expert_ref(&ex[layer][idx[i]], xq + (size_t)(i / K) * H, cpu + (size_t)i * H);
        VktExpertSrc s = src_of(&ex[layer][idx[i]]); vkt_note(layer, idx[i], &s);
        (*kept)++;
    }
    int joined = grp ? vkt_join(dev) : (def ? 1 : (n ? vkt_join(dev) : 1));
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
                double r = rel(c, y, H); if (!(r <= worst)) worst = r;   /* a NaN counts as the worst */
                if (!res[i]) (*streamed)++;
            }
            for (int d = 0; d < H; d++) { ref[(size_t)s * H + d] += (v4 ? 1.0f : w[i]) * y[d]; out[(size_t)s * H + d] += (v4 ? 1.0f : w[i]) * c[d]; }
            bk->routed++; bk->dev += taken[i] != 0;
        }
    double r = rel(out, ref, S * H);
    if (!(r <= worst)) worst = r;
    if (capture) memcpy(capture, out, sizeof(float) * (size_t)S * H);
    if (getenv("VKT_TEST_DUMP") && n) fprintf(stderr, "dump: ref %g %g dev %g %g\n", ref[0], ref[1], out[0], out[1]);
    free(x); free(xq); free(cpu); free(out); free(ref); free(y); free(taken); free(dev); free(res);
    return worst;
}
/* COLI_VK_TIER_BANK_LAYER (the VRAM bank, S1's substrate): the stream slots exchanged
 * for one layer's experts. The carve: the boot ledger line says the slot budget's pair
 * (12 -> 2 here), the armed plan places the banked layer's experts whole at the device
 * ahead of the history (all or nothing, like the dmoe gate whose table then counts them
 * -- vkt_dev_table's n == E with zero new gate code), and the warm start fills them. The
 * refill: a step of the banked layer fills the experts the warm start left un-read
 * (still queued) through the load hooks, each at most once a tenure -- a second call
 * loads nothing, and another layer's step is not the bank's (the engine's seam never
 * switches: S1's bank holds the env's layer for the run). The rotation (the fork's
 * layer-switch discipline, the F6 case) is the explicit seam's, vkt_bank_switch: the
 * bitmap clears, the old layer's experts free their device places, the new layer's
 * routed experts refill at most once (both ways), and every step's rows stay the
 * reference's. Declines: an out-of-range layer and a slot
 * budget too small for the bank each print their one line and leave the plan the
 * history's. */
static void bank_layer_case(void) {
    rng = 7;   /* deterministic routing: the refill's expectations count the routed set */
    VktFmt f = {VKT_SRC_I4U_PAIRS_GS, 64};
    model_make(f, f); g_act = VKT_ACT_SWIGLU; g_limit = 0;
    uint32_t hist[L][E], *rows[L];           /* layer 0's first two hot; the bank's layer is 1 */
    for (int l = 0; l < L; l++) { rows[l] = hist[l]; for (int e = 0; e < E; e++) hist[l][e] = l == 0 && e < 2 ? 100u : 0u; }
    setenv("COLI_VK_TIER_RATE", "0", 1);          /* no promotions: residency moves with the bank alone */
    setenv("COLI_VK_TIER_FILL", "0", 1);         /* no fill: the history's two, plus the bank's layer */
    setenv("COLI_VK_TIER_STREAM_SLOTS", "12", 1);
    setenv("COLI_VK_TIER_BANK_LAYER", "1", 1);
    set_budget(24, f, f);                         /* 24 experts: 12 slots, 12 residents, the carve trades 10 */
    VktConfig vc = cfg_of(f, f, VKT_ACT_SWIGLU, 0);
    vc.load = load_cb; vc.release = release_cb; vc.load_batch = load_batch_cb;
    uint32_t *tab = malloc((size_t)E * COLI_VK_XB_ENTRY * sizeof(uint32_t));
    /* the carve's ledger line prints at the tier's own boot, not at the plan */
    FILE *t = tmpfile(); fflush(stderr);
    int saved = t ? dup(2) : -1;
    if (saved >= 0) dup2(fileno(t), 2);
    int on = vkt_init(&vc, rows);
    char buf[4096], line[4096] = "";
    if (saved >= 0) {
        fflush(stderr); dup2(saved, 2); close(saved); rewind(t);
        while (fgets(buf, sizeof buf, t)) if (strstr(buf, "bank=layer 1")) snprintf(line, sizeof line, "%s", buf);
        fclose(t);
    }
    fputs(line, stderr);
    CHECK(on, "bank: the tier did not start");
    int pl[L * E], pe[L * E], n = vkt_plan(pl, pe, L * E);
    CHECK((int)strlen(line) > 0 && strstr(line, "the stream slots' budget 12 -> 2"),
          "bank: the carve's ledger line missing or wrong: %s", line);
    fputs(line, stderr);
    int per[L] = {0}, first = 1;
    for (int i = 0; i < n; i++) { per[pl[i]]++; if (i < E) first &= pl[i] == 1; }
    CHECK(n == E + 2 && per[1] == E && per[0] == 2 && first,
          "bank: planned %d experts (layers %d/%d), not the banked layer's %d ahead of the history's 2",
          n, per[0], per[1], E);
    for (int i = 0; i < n; i++) {   /* the warm start, three of the banked layer's experts un-read (still queued) */
        if (pl[i] == 1 && pe[i] >= E - 3) continue;
        VktExpertSrc s = src_of(&ex[pl[i]][pe[i]]); vkt_put(pl[i], pe[i], &s);
    }
    vkt_put_done();
    int res = 0;
    for (int e = 0; e < E; e++) res += vkt_resident(1, e);
    CHECK(res == E - 3, "bank: %d of the banked layer's %d resident after the warm start (3 left queued)", res, E);
    CHECK(tab && vkt_dev_table(1, tab) == E - 3, "bank: the table counts %d of %d (the gate stays shut until the refill)",
          tab ? vkt_dev_table(1, tab) : -1, E);
    /* the refill at the routed step (post-router, pre-issue): the queued experts load
     * through the hooks, the resident ones never re-read */
    Books bk = {0, 0}; double worst = 0;
    int idx[8 * K]; float w[8 * K];
    route(4, 0, E, idx, w);
    idx[0] = E - 3; idx[1] = E - 2; idx[2] = E - 1;   /* the queued trio routed: the refill's expectation */
    int want = 0;
    for (int i = 0; i < 4 * K; i++) {   /* the distinct queued experts the step routes */
        int dup = 0;
        for (int j = 0; j < i; j++) dup |= idx[j] == idx[i];
        want += !dup && idx[i] >= E - 3;
    }
    g_loads = 0;
    vkt_bank_fill(1, idx, 4 * K);
    CHECK(g_loads == want, "bank: the refill loaded %d experts, not the %d queued ones the step routed", g_loads, want);
    res = 0;
    for (int e = 0; e < E; e++) res += vkt_resident(1, e);
    CHECK(res == E && vkt_dev_table(1, tab) == E, "bank: %d of %d resident after the refill, the table whole (the gate opens)", res, E);
    g_loads = 0;
    vkt_bank_fill(1, idx, 4 * K);   /* at most once a tenure: nothing left to load */
    CHECK(g_loads == 0, "bank: the second refill loaded %d experts -- each is at most once a tenure", g_loads);
    vkt_begin_forward();
    worst = step(1, 4, idx, w, &bk);
    CHECK(worst < 2e-3, "bank: the banked layer's rows off the reference (%.3g)", worst);
    /* another layer's step is not the bank's: S1's engine seam never switches (a chunk-outer
     * loop visits every layer each chunk, and a bank that followed the visit order would
     * thrash a place a layer) -- nothing loads, nothing moves */
    int idx0[8 * K]; float w0[8 * K];
    route(4, 0, E, idx0, w0);
    g_loads = 0;
    vkt_bank_fill(0, idx0, 4 * K);
    res = 0;
    for (int e = 0; e < E; e++) res += vkt_resident(1, e);
    CHECK(g_loads == 0 && res == E, "bank: another layer's step loaded %d experts or moved the bank (%d of %d still resident)",
          g_loads, res, E);
    /* the rotation (r2 F6), the explicit seam: switch, then the new layer's routed experts
     * refill at most once -- the bitmap cleared, the old layer's places freed */
    int miss0 = 0;
    for (int i = 0; i < 4 * K; i++) { int dup = 0; for (int j = 0; j < i; j++) dup |= idx0[j] == idx0[i]; miss0 += !dup && !vkt_resident(0, idx0[i]); }
    vkt_bank_switch(0);
    g_loads = 0;
    vkt_bank_fill(0, idx0, 4 * K);
    CHECK(g_loads == miss0, "bank: the switch refilled %d experts, not the %d missing the step routed", g_loads, miss0);
    res = 0;
    int res1 = 0;
    for (int e = 0; e < E; e++) { res += vkt_resident(0, e); res1 += vkt_resident(1, e); }
    CHECK(res1 == 0, "bank: %d of the old banked layer's experts still hold device places after the switch", res1);
    CHECK(res >= 2 + miss0, "bank: the new banked layer holds %d resident (its 2 warm and %d refilled)", res, miss0);
    g_loads = 0;
    vkt_bank_fill(0, idx0, 4 * K);   /* at most once holds per layer tenure */
    CHECK(g_loads == 0, "bank: the switched layer's second refill loaded %d experts", g_loads);
    vkt_begin_forward();
    double r0 = step(0, 4, idx0, w0, &bk);
    if (r0 > worst) worst = r0;
    CHECK(r0 < 2e-3, "bank: the switched layer's rows off the reference (%.3g)", r0);
    /* and back: the rotation both ways, the first layer's experts refilled again */
    vkt_bank_switch(1);
    g_loads = 0;
    vkt_bank_fill(1, idx, 4 * K);
    CHECK(g_loads > 0, "bank: the switch back refilled nothing (the first layer's places were freed)");
    res1 = 0;
    for (int e = 0; e < E; e++) res1 += vkt_resident(1, e);
    CHECK(res1 >= E - 3, "bank: %d of the first banked layer's %d back after the switch back", res1, E);
    vkt_report("test", 0, 0);
    vkt_shutdown();
    /* the declines: out of range and no room, one line each, the plan the lever-free tier's */
    unsetenv("COLI_VK_TIER_BANK_LAYER");
    unsetenv("COLI_VK_TIER_STREAM_SLOTS");
    CHECK(vkt_init(&vc, rows), "bank: the tier did not start lever-free");
    int fl[L * E], fe[L * E], n0 = vkt_plan(fl, fe, L * E);
    vkt_shutdown();
    for (int c = 0; c < 2; c++) {
        if (!c) setenv("COLI_VK_TIER_BANK_LAYER", "2", 1);
        else { setenv("COLI_VK_TIER_BANK_LAYER", "1", 1); setenv("COLI_VK_TIER_STREAM_SLOTS", "4", 1); }
        t = tmpfile(); fflush(stderr);
        saved = t ? dup(2) : -1;
        if (saved >= 0) dup2(fileno(t), 2);
        int on2 = vkt_init(&vc, rows);
        char dline[4096] = "";
        int said = 0;
        if (saved >= 0) {
            fflush(stderr); dup2(saved, 2); close(saved); rewind(t);
            while (fgets(buf, sizeof buf, t)) {
                fputs(buf, stderr);
                if (strstr(buf, "COLI_VK_TIER_BANK_LAYER")) { said++; if (!dline[0]) snprintf(dline, sizeof dline, "%s", buf); }
            }
            fclose(t);
        }
        CHECK(on2, "bank: the tier did not start again (decline %d)", c);
        int ql[L * E], qe[L * E], m = vkt_plan(ql, qe, L * E);
        int same = m == n0, qper[L] = {0};
        for (int i = 0; i < m; i++) qper[ql[i]]++;
        for (int i = 0; same && i < m; i++) same = ql[i] == fl[i] && qe[i] == fe[i];
        CHECK(said == 1 && strstr(dline, c ? "no room" : "out of range"),
              "bank: the decline said %d lines, not the one %s", said, c ? "no-room" : "out-of-range");
        CHECK(same && !qper[1], "bank: the declined lever moved the plan (%d experts, the lever-free %d, layer 1's %d of them)",
              m, n0, qper[1]);
        printf("  bank layer, %s: the plan the lever-free tier's %d experts exactly -- nothing placed\n", c ? "no room" : "out of range", m);
        vkt_shutdown();
        unsetenv("COLI_VK_TIER_STREAM_SLOTS");
    }
    free(tab);
    unsetenv("COLI_VK_TIER_BANK_LAYER"); unsetenv("COLI_VK_TIER_STREAM_SLOTS");
    unsetenv("COLI_VK_TIER_FILL"); unsetenv("COLI_VK_TIER_RATE");
    setenv("COLI_VK_TIER_HOST", "0", 1);
    model_free();
}

/* COLI_VK_TIER_DECODE_GROW: the armed tier's resident set grows over a decode sequence
 * by the DECODE-ONLY rank (VktConfig.decode_heat), which here ranks differently than the
 * merged history the warm start filled by -- free VRAM first (leg A: a capped probe)
 * and the stream slots only beyond (leg B: the reserve kills the free part, the installs
 * retire slots a place at a time), the budget respected, the below-cutoff routed experts
 * never installed, and the env unset growing nothing (leg C, the default).
 * COLI_VK_TIER_DECODE_GROW_HEADROOM (legs D-F): the free places are the live guard's --
 * made one install at a time against the card's free VRAM (re-probed fork-style), the
 * TARGET env the residents' count it may reach (leg D grows the guard's two, leg E the
 * target's ceiling biting at one), and with no target the operator's byte cap the bound
 * (leg F: cutoff 0 admits every demand, the cap stops the second place).
 * COLI_VK_TIER_PROMO_CAP (leg G): the rate's 1-a-forward starves the same growth (the
 * W2 class -- installs refused with room standing) until the cap is lifted, a fixed N
 * or negative (none); both legs end at the target's count.
 * The headroom legs (D-G) ride their own `dgrow` invocation: their guard-made places
 * allocate as installs proceed, so the pool's block count and frag, and the no-target
 * arm line's probe-derived place count, differ between ANY two runs -- and the strict
 * gate byte-compares the plain invocation's output of its unset-vs-0 pair. Legs A-C
 * (the standing decode-grow case, deterministic) stay in the plain invocation. */
static int dg_res(void) { int n = 0; for (int l = 0; l < L; l++) for (int e = 0; e < E; e++) n += vkt_resident(l, e); return n; }
static void report_line(char *out, size_t n);   /* below: the tier's report line as a string */
static void decode_grow(int dg) {
    lever_pin_save();   /* the legs set/unset the growth envs: restore the caller's standing at exit */
    rng = 7;
    VktFmt f = {VKT_SRC_I8_ROW, 0};
    model_make(f, f); g_act = VKT_ACT_SWIGLU; g_limit = 0;
    const double eb = (double)vkt_expert_bytes(H, F, f, f);
    char gbs[2][64];
    /* the merged history warms e<3 of each layer; the decode-only table ranks 4..7:
     * 40, 30, 20, 10 -- the growth must follow the decode rank, not the merged one */
    uint32_t hist[L][E], dhist[L][E], *rows[L], *drows[L];
    for (int l = 0; l < L; l++) {
        rows[l] = hist[l]; drows[l] = dhist[l];
        for (int e = 0; e < E; e++) {
            hist[l][e] = e < 3 ? 100u : 0u;
            dhist[l][e] = e >= 4 && e < 8 ? 40u - 10u * (uint32_t)(e - 4) : 0u;
        }
    }
    /* --- leg A: the growth comes from free VRAM (no stream: no slots to retire) --- */
    setenv("COLI_VK_TIER_SYNC", "1", 1);
    setenv("COLI_VK_TIER_RATE", "64", 1);
    set_budget(6, f, f);   /* the warm start fills it exactly: no free place to promote into */
    snprintf(gbs[0], sizeof gbs[0], "%.12f", 2.5 * eb / 1073741824.0);   /* two places */
    setenv("COLI_VK_TIER_DECODE_GROW", "1", 1);
    setenv("COLI_VK_TIER_DECODE_GROW_GB", gbs[0], 1);
    setenv("COLI_VK_TIER_DECODE_GROW_RESERVE_GB", "0.000001", 1);
    VktConfig vc = cfg_of(f, f, VKT_ACT_SWIGLU, 0);
    vc.decode_heat = drows;
    CHECK(vkt_init(&vc, rows), "dgrow: the tier did not start");
    int pl[L * E], pe[L * E], n = vkt_plan(pl, pe, L * E);
    for (int i = 0; i < n; i++) { VktExpertSrc s = src_of(&ex[pl[i]][pe[i]]); vkt_put(pl[i], pe[i], &s); }
    vkt_put_done();
    CHECK(n == 6 && dg_res() == 6, "dgrow A: %d warm of %d planned, budget 6", dg_res(), n);
    /* a decode sequence that routes the BELOW-cutoff experts first and often: they must
     * never take a place; the hottest (40) installs in whatever order it arrives */
    Books bk = {0, 0}; double worst = 0;
    int curve[8]; int nc = 0;
    static const int seq[4][K] = {{7, 5, 6}, {6, 5, 7}, {5, 4, 7}, {4, 6, 5}};
    for (int t = 0; t < 4; t++) {
        for (int l = 0; l < L; l++) {
            float w[K] = {0.5f, 0.3f, 0.2f};
            vkt_begin_forward();
            double r = step(l, 1, seq[t], w, &bk);
            if (r > worst) worst = r;
        }
        vkt_settle();
        curve[nc++] = dg_res();
    }
    char line[4096]; report_line(line, sizeof line);
    const char *clause = strstr(line, " | decode grow: ");
    int mono = 1;
    for (int i = 1; i < nc; i++) mono &= curve[i] >= curve[i - 1];
    printf("  decode grow A (free VRAM): residents 6 -> %d over the sequence (curve", dg_res());
    for (int i = 0; i < nc; i++) printf(" %d", curve[i]);
    printf("), worst %.2e, report \"%s\"\n", worst, clause ? clause + 3 : "no clause");
    CHECK(dg_res() == 8, "dgrow A: %d residents, not the warm 6 + 2", dg_res());
    CHECK(vkt_resident(0, 4) && vkt_resident(1, 4), "dgrow A: the hottest expert of each layer is not resident");
    CHECK(!vkt_resident(0, 5) && !vkt_resident(1, 5) && !vkt_resident(0, 6) && !vkt_resident(0, 7),
          "dgrow A: a below-cutoff routed expert installed (5/6/7)");
    CHECK(mono, "dgrow A: the resident set shrank");
    CHECK(worst < 2e-3, "dgrow A: rows off the reference (%.3g)", worst);
    CHECK(clause && strstr(line, "decode grow: 2 installed by decode-only rank (cutoff 40)"), "dgrow A: the report's growth clause");
    CHECK(strstr(line, "evictions 0"), "dgrow A: growth evicted");
    vkt_shutdown();
    /* --- leg B: the free part gone, the slots beyond -- installs retire them --- */
    snprintf(gbs[1], sizeof gbs[1], "%.12f", 3.5 * eb / 1073741824.0);   /* three places */
    setenv("COLI_VK_TIER_DECODE_GROW_GB", gbs[1], 1);
    setenv("COLI_VK_TIER_DECODE_GROW_RESERVE_GB", "99", 1);
    setenv("COLI_VK_TIER_STREAM_SLOTS", "6", 1);
    setenv("COLI_VK_TIER_STREAM_ROWS", "3", 1);
    setenv("COLI_VK_TIER_FILL", "0", 1);
    vc = cfg_of(f, f, VKT_ACT_SWIGLU, 0);
    vc.load = load_cb; vc.release = release_cb;
    vc.decode_heat = drows;
    set_budget(12, f, f);   /* six warm, six slots: the budget exact, no free place */
    CHECK(vkt_init(&vc, rows), "dgrow: the tier did not start again");
    n = vkt_plan(pl, pe, L * E);
    for (int i = 0; i < n; i++) { VktExpertSrc s = src_of(&ex[pl[i]][pe[i]]); vkt_put(pl[i], pe[i], &s); }
    vkt_put_done();
    CHECK(n == 6 && dg_res() == 6, "dgrow B: %d warm of %d planned, budget 6", dg_res(), n);
    int big[64 * K]; float w[64 * K];
    unsigned long long streamed = 0, kept = 0;
    for (int l = 0; l < L; l++) {   /* a big step a layer: the slots allocate, the arm sees them */
        route(64, 0, E, big, w);
        vkt_begin_forward();
        stream_step(l, 64, big, w, 0, &bk, &streamed, &kept, 0, NULL);
    }
    vkt_settle();
    CHECK(dg_res() == 6, "dgrow B: the big steps installed something (%d residents)", dg_res());
    static const int seqB[3][K] = {{7, 6, 5}, {7, 5, 4}, {4, 6, 7}};
    for (int t = 0; t < 3; t++) {
        float ww[K] = {0.5f, 0.3f, 0.2f};
        vkt_begin_forward();
        double r = step(t < 2 ? t : 0, 1, seqB[t], ww, &bk);   /* layer 0, then 1, then 0 again */
        if (r > worst) worst = r;
        vkt_settle();
    }
    report_line(line, sizeof line);
    const char *sl = strstr(line, "(slots ");
    int s0 = sl ? atoi(sl + 7) : -1, s1 = sl && strstr(sl, " -> ") ? atoi(strstr(sl, " -> ") + 4) : -1;
    printf("  decode grow B (slots beyond): residents 6 -> %d, slots %d -> %d, the fourth hot refused by the cap, "
           "worst %.2e\n", dg_res(), s0, s1, worst);
    CHECK(dg_res() == 9, "dgrow B: %d residents, not the warm 6 + 3", dg_res());
    CHECK(s0 == 6 && s1 == 3, "dgrow B: the slot pair %d -> %d, not 6 -> 3", s0, s1);
    CHECK(vkt_resident(0, 5) && vkt_resident(1, 5) && vkt_resident(1, 4), "dgrow B: the hot installs are missing");
    CHECK(!vkt_resident(0, 4), "dgrow B: the fourth hot expert took a place past the cap");
    CHECK(!vkt_resident(0, 6) && !vkt_resident(0, 7) && !vkt_resident(1, 6) && !vkt_resident(1, 7),
          "dgrow B: a below-cutoff expert installed (6/7)");
    CHECK(strstr(line, "decode grow: 3 installed by decode-only rank (cutoff 30)"), "dgrow B: the report's growth clause");
    vkt_shutdown();
    unsetenv("COLI_VK_TIER_STREAM_SLOTS"); unsetenv("COLI_VK_TIER_STREAM_ROWS"); unsetenv("COLI_VK_TIER_FILL");
    /* --- leg C: the off half -- the growth is the DEFAULT since gap-wave4, so the
     * "nothing grows" invariant is the =0 hatch's (and COLI_VK_LEGACY=1's: same
     * resolution, backend_vulkan.h's policy) --- */
    setenv("COLI_VK_TIER_DECODE_GROW", "0", 1); unsetenv("COLI_VK_TIER_DECODE_GROW_GB");
    unsetenv("COLI_VK_TIER_DECODE_GROW_RESERVE_GB");
    set_budget(6, f, f);
    vc = cfg_of(f, f, VKT_ACT_SWIGLU, 0);
    vc.decode_heat = drows;
    CHECK(vkt_init(&vc, rows), "dgrow C: the tier did not start");
    n = vkt_plan(pl, pe, L * E);
    for (int i = 0; i < n; i++) { VktExpertSrc s = src_of(&ex[pl[i]][pe[i]]); vkt_put(pl[i], pe[i], &s); }
    vkt_put_done();
    for (int t = 0; t < 2; t++)
        for (int l = 0; l < L; l++) {
            float ww[K] = {0.5f, 0.3f, 0.2f};
            vkt_begin_forward();
            step(l, 1, seq[t], ww, &bk);
        }
    vkt_settle();
    report_line(line, sizeof line);
    printf("  decode grow C (the =0 hatch): residents %d after the same decode sequence, no growth clause\n", dg_res());
    CHECK(dg_res() == 6, "dgrow C: the =0 hatch grew (%d residents)", dg_res());
    CHECK(!strstr(line, "decode grow:"), "dgrow C: the =0 hatch printed a growth clause");
    vkt_shutdown();
    if (dg) {   /* the dgrow invocation's own legs -- see the case's comment above */
        /* --- legs D-F: COLI_VK_TIER_DECODE_GROW_HEADROOM -- the live guard owns the free
         * places (made one install at a time against the card's free VRAM, less the tiny
         * reserve), the TARGET env the residents' count it may reach; no stream, so a slot
         * can never stand in for the guard --- */
        setenv("COLI_VK_TIER_DECODE_GROW", "1", 1);
        setenv("COLI_VK_TIER_DECODE_GROW_RESERVE_GB", "0.000001", 1);
        setenv("COLI_VK_TIER_DECODE_GROW_HEADROOM", "1", 1);
        for (int leg = 0; leg < 3; leg++) {
            static const int targets[3] = {8, 7, 99};   /* two guard places, the ceiling at one, the bar past the table */
            char tg[16]; snprintf(tg, sizeof tg, "%d", targets[leg]);
            setenv("COLI_VK_TIER_DECODE_GROW_TARGET", tg, 1);
            if (leg == 2) {   /* F only: the operator's byte cap stops the guard after one place */
                char gb[64]; snprintf(gb, sizeof gb, "%.12f", 1.5 * eb / 1073741824.0);
                setenv("COLI_VK_TIER_DECODE_GROW_GB", gb, 1);
            } else unsetenv("COLI_VK_TIER_DECODE_GROW_GB");
            set_budget(6, f, f);
            vc = cfg_of(f, f, VKT_ACT_SWIGLU, 0);
            vc.decode_heat = drows;
            CHECK(vkt_init(&vc, rows), "dgrow %c: the tier did not start", 'D' + leg);
            n = vkt_plan(pl, pe, L * E);
            for (int i = 0; i < n; i++) { VktExpertSrc s = src_of(&ex[pl[i]][pe[i]]); vkt_put(pl[i], pe[i], &s); }
            vkt_put_done();
            CHECK(n == 6 && dg_res() == 6, "dgrow %c: %d warm of %d planned, budget 6", 'D' + leg, dg_res(), n);
            double worst = 0;
            int curve[8]; int nc = 0;
            for (int t = 0; t < 4; t++) {
                for (int l = 0; l < L; l++) {
                    float ww[K] = {0.5f, 0.3f, 0.2f};
                    vkt_begin_forward();
                    double r = step(l, 1, seq[t], ww, &bk);
                    if (!(r <= worst)) worst = r;   /* a NaN counts as the worst */
                }
                vkt_settle();
                curve[nc++] = dg_res();
            }
            report_line(line, sizeof line);
            const char *hclause = strstr(line, " | decode grow: ");
            int grew = 1;
            for (int i = 1; i < nc; i++) grew &= curve[i] >= curve[i - 1];
            printf("  decode grow %c (headroom, target %d%s): residents 6 -> %d (curve", 'D' + leg, targets[leg],
                   leg == 2 ? ", byte cap" : "", dg_res());
            for (int i = 0; i < nc; i++) printf(" %d", curve[i]);
            printf("), worst %.2e, report \"%s\"\n", no_nan(worst), hclause ? hclause + 3 : "no clause");
            if (leg == 0) {
                CHECK(dg_res() == 8, "dgrow D: %d residents, not the guard's 6 + 2", dg_res());
                CHECK(vkt_resident(0, 4) && vkt_resident(1, 4), "dgrow D: the hottest expert of each layer is not resident");
                CHECK(!vkt_resident(0, 5) && !vkt_resident(0, 6) && !vkt_resident(0, 7) && !vkt_resident(1, 5),
                      "dgrow D: a below-cutoff routed expert installed (5/6/7)");
                CHECK(grew, "dgrow D: the resident set shrank");
                CHECK(strstr(line, "decode grow: 2 installed by decode-only rank (cutoff 40)"), "dgrow D: the report's growth clause");
                CHECK(strstr(line, "2 by the live guard"), "dgrow D: the report's headroom clause");
                CHECK(strstr(line, "evictions 0"), "dgrow D: growth evicted");
            } else if (leg == 1) {
                CHECK(dg_res() == 7, "dgrow E: %d residents, not the target's ceiling of 7", dg_res());
                CHECK(vkt_resident(0, 4) && !vkt_resident(1, 4), "dgrow E: the target's one place went to both layers");
                CHECK(strstr(line, "1 by the live guard"), "dgrow E: the report's headroom clause");
            } else {
                CHECK(dg_res() == 7, "dgrow F: %d residents, not the byte cap's one place", dg_res());
                CHECK(vkt_resident(0, 7) && !vkt_resident(0, 4) && !vkt_resident(0, 5),
                      "dgrow F: cutoff 0 admitted every demand; the cap stopped the second place");
                CHECK(strstr(line, "decode grow: 1 installed by decode-only rank (cutoff 0)"), "dgrow F: the report's growth clause");
                CHECK(strstr(line, "1 by the live guard"), "dgrow F: the report's headroom clause");
            }
            CHECK(no_nan(worst) < 2e-3, "dgrow %c: rows off the reference (%.3g)", 'D' + leg, no_nan(worst));
            vkt_shutdown();
        }
        /* --- the caps gone (no target, no byte cap): cutoff 0, every routed demand installs
         * -- the guard's places bounded by the card alone, the fork's demand class --- */
        unsetenv("COLI_VK_TIER_DECODE_GROW_TARGET"); unsetenv("COLI_VK_TIER_DECODE_GROW_GB");
        set_budget(6, f, f);
        vc = cfg_of(f, f, VKT_ACT_SWIGLU, 0);
        vc.decode_heat = drows;
        CHECK(vkt_init(&vc, rows), "dgrow F2: the tier did not start");
        n = vkt_plan(pl, pe, L * E);
        for (int i = 0; i < n; i++) { VktExpertSrc s = src_of(&ex[pl[i]][pe[i]]); vkt_put(pl[i], pe[i], &s); }
        vkt_put_done();
        for (int l = 0; l < L; l++) {
            float ww[K] = {0.5f, 0.3f, 0.2f};
            vkt_begin_forward();
            step(l, 1, seq[0], ww, &bk);   /* {7,5,6}: below leg A's bar, first at the cap-less guard */
        }
        vkt_settle();
        report_line(line, sizeof line);
        printf("  decode grow F2 (headroom, no target): one round of the below-bar experts installs all six, residents %d\n", dg_res());
        CHECK(dg_res() == 12, "dgrow F2: %d residents, not every routed demand", dg_res());
        CHECK(strstr(line, "6 by the live guard"), "dgrow F2: the report's headroom clause");
        vkt_shutdown();
        /* --- leg G: COLI_VK_TIER_PROMO_CAP -- the rate's 1-a-forward starves the same
         * growth (W2's `skipped 312 rate` class: notes refused with the guard's room
         * standing) until the cap is lifted, a fixed N or negative (none). One
         * vkt_begin_forward a round here, both layers' notes against the one cap -- the
         * model's decode-token shape, not the per-step forward of the legs above --- */
        setenv("COLI_VK_TIER_DECODE_GROW_TARGET", "10", 1);
        setenv("COLI_VK_TIER_RATE", "1", 1);
        static const int seqG[2][K] = {{4, 5, 6}, {6, 5, 7}};
        for (int leg = 0; leg < 3; leg++) {
            if (leg == 0) unsetenv("COLI_VK_TIER_PROMO_CAP");
            else {
                char pc[8]; snprintf(pc, sizeof pc, "%d", leg == 1 ? 6 : -1);
                setenv("COLI_VK_TIER_PROMO_CAP", pc, 1);
            }
            set_budget(6, f, f);
            vc = cfg_of(f, f, VKT_ACT_SWIGLU, 0);
            vc.decode_heat = drows;
            CHECK(vkt_init(&vc, rows), "dgrow G%d: the tier did not start", leg);
            n = vkt_plan(pl, pe, L * E);
            for (int i = 0; i < n; i++) { VktExpertSrc s = src_of(&ex[pl[i]][pe[i]]); vkt_put(pl[i], pe[i], &s); }
            vkt_put_done();
            double worst = 0;
            for (int t = 0; t < 2; t++) {
                vkt_begin_forward();
                for (int l = 0; l < L; l++) {
                    float ww[K] = {0.5f, 0.3f, 0.2f};
                    double r = step(l, 1, seqG[t], ww, &bk);
                    if (!(r <= worst)) worst = r;   /* a NaN counts as the worst */
                }
                vkt_settle();
            }
            report_line(line, sizeof line);
            /* the skip counters cut before the timing span (" | device "): those fields jitter
             * between any two runs, and the strict gate byte-compares this test's stdout of its
             * unset-vs-0 pair -- the growth clause beside them carries no timings */
            char sk[64];
            const char *s0 = strstr(line, "skipped ");
            if (s0) {
                const char *d0 = strstr(s0, " | device ");
                size_t n = d0 && (size_t)(d0 - s0) < sizeof sk - 1 ? (size_t)(d0 - s0) : sizeof sk - 1;
                memcpy(sk, s0, n); sk[n] = 0;
            } else snprintf(sk, sizeof sk, "no skips");
            const char *gclause = strstr(line, " | decode grow: ");
            printf("  decode grow G%d (promo cap %s): residents 6 -> %d, %s, %s\n", leg,
                   leg == 0 ? "unset" : leg == 1 ? "6" : "-1", dg_res(), sk,
                   gclause ? gclause + 3 : "no clause");
            if (leg == 0) {
                CHECK(dg_res() == 8, "dgrow G0: %d residents under the rate's 1-a-forward, not 8", dg_res());
                CHECK(strstr(line, "skipped 0 queue + 9 rate"), "dgrow G0: the rate-starved run line");
            } else {
                CHECK(dg_res() == 10, "dgrow G%d: %d residents, not the target's 10 under the lifted cap", leg, dg_res());
                CHECK(strstr(line, "skipped 0 queue + 0 rate"), "dgrow G%d: a note still refused by the rate", leg);
            }
            CHECK(no_nan(worst) < 2e-3, "dgrow G%d: rows off the reference (%.3g)", leg, no_nan(worst));
            vkt_shutdown();
        }
        unsetenv("COLI_VK_TIER_DECODE_GROW"); unsetenv("COLI_VK_TIER_DECODE_GROW_RESERVE_GB");
        unsetenv("COLI_VK_TIER_DECODE_GROW_HEADROOM"); unsetenv("COLI_VK_TIER_DECODE_GROW_TARGET");
        unsetenv("COLI_VK_TIER_PROMO_CAP");
    }
    unsetenv("COLI_VK_TIER_SYNC"); unsetenv("COLI_VK_TIER_RATE");
    model_free();
}

/* COLI_VK_TIER_ADAPTIVE: the forward-boundary re-probe (the fork's request-boundary
 * reorder). Leg A grows dmax[0] into free VRAM above reserve+margin and the routed
 * demand then installs; leg B sheds the coldest residents and shrinks dmax when free
 * falls below the reserve (a co-tenant's reclaim), leaving the in-flight group of the
 * very step untouched -- it still joins and its rows are the reference's; leg C keeps
 * the env-unset default silent and unchanged. */
static FILE *cap_begin(int *saved) {
    FILE *t = tmpfile();
    if (!t) { *saved = -1; return NULL; }
    fflush(stderr);
    *saved = dup(2);
    if (*saved < 0) { fclose(t); return NULL; }
    dup2(fileno(t), 2);
    return t;
}
static void cap_end(FILE *t, int saved, char *out, size_t n, const char *needle) {
    out[0] = 0;
    if (!t) return;
    fflush(stderr);
    dup2(saved, 2); close(saved);
    rewind(t);
    char line[4096];
    while (fgets(line, sizeof line, t)) {
        fputs(line, stderr);
        if (!out[0] && strstr(line, needle)) snprintf(out, n, "%s", line);
    }
    fclose(t);
}
static void tier_adaptive_case(void) {
    VktFmt f = {VKT_SRC_I8_ROW, 0};
    model_make(f, f); g_act = VKT_ACT_SWIGLU; g_limit = 0;
    const double eb = (double)vkt_expert_bytes(H, F, f, f);
    /* the history warms e<3 of each layer, exactly the six-resident budget */
    uint32_t hist[L][E], *rows[L];
    for (int l = 0; l < L; l++) {
        rows[l] = hist[l];
        for (int e = 0; e < E; e++) hist[l][e] = e < 3 ? 100u : 0u;
    }
    setenv("COLI_VK_TIER_SYNC", "1", 1);
    setenv("COLI_VK_TIER_RATE", "64", 1);
    setenv("COLI_VK_TIER_ADAPTIVE", "1", 1);
    setenv("COLI_VK_TIER_ADAPTIVE_RESERVE_GB", "0.000001", 1);
    setenv("COLI_VK_TIER_ADAPTIVE_MARGIN_GB", "0", 1);
    int pl[L * E], pe[L * E];
    /* --- leg A: free above reserve + margin -> grow, the routed demand installs --- */
    set_budget(6, f, f);
    VktConfig vc = cfg_of(f, f, VKT_ACT_SWIGLU, 0);
    CHECK(vkt_init(&vc, rows), "adaptive A: the tier did not start");
    int n = vkt_plan(pl, pe, L * E);
    for (int i = 0; i < n; i++) { VktExpertSrc s = src_of(&ex[pl[i]][pe[i]]); vkt_put(pl[i], pe[i], &s); }
    vkt_put_done();
    CHECK(n == 6 && dg_res() == 6, "adaptive A: %d warm of %d planned, budget 6", dg_res(), n);
    char freemb[64]; snprintf(freemb, sizeof freemb, "%.12f", 2.5 * eb / 1048576.0);   /* two places */
    setenv("COLI_VK_TIER_FREE_MB", freemb, 1);
    static const int seqA[K] = {6, 7, 2};   /* 2 resident (the device takes it); 6,7 install into the grown places */
    float w[K] = {0.5f, 0.3f, 0.2f};
    Books bk = {0, 0};
    int saved; FILE *t = cap_begin(&saved);
    vkt_begin_forward();
    double worst = step(0, 1, seqA, w, &bk);
    vkt_settle();
    char said[4096]; cap_end(t, saved, said, sizeof said, "tier adaptive:");
    printf("  adaptive A (grow on free): residents 6 -> %d, line \"%s\", worst %.2e\n", dg_res(), said, worst);
    CHECK(strstr(said, "grow 2"), "adaptive A: the grow line (\"%s\")", said);
    CHECK(dg_res() == 8, "adaptive A: %d residents, not the warm 6 + 2", dg_res());
    CHECK(vkt_resident(0, 6) && vkt_resident(0, 7), "adaptive A: the routed demand did not install");
    CHECK(worst < 2e-3, "adaptive A: rows off the reference (%.3g)", worst);
    vkt_shutdown();
    /* --- leg B: free below reserve -> shed, dmax shrinks, no in-flight group broken --- */
    setenv("COLI_VK_TIER_RATE", "0", 1);   /* no re-promotion: the shed's places stay freed */
    set_budget(6, f, f);
    vc = cfg_of(f, f, VKT_ACT_SWIGLU, 0);
    CHECK(vkt_init(&vc, rows), "adaptive B: the tier did not start");
    n = vkt_plan(pl, pe, L * E);
    for (int i = 0; i < n; i++) { VktExpertSrc s = src_of(&ex[pl[i]][pe[i]]); vkt_put(pl[i], pe[i], &s); }
    vkt_put_done();
    CHECK(n == 6 && dg_res() == 6, "adaptive B: %d warm of %d planned, budget 6", dg_res(), n);
    setenv("COLI_VK_TIER_FREE_MB", "0", 1);
    static const int seqB[K] = {1, 2, 3};   /* 1,2 resident: the step's group flies after the shed */
    bk = (Books){0, 0};
    t = cap_begin(&saved);
    vkt_begin_forward();
    worst = step(0, 1, seqB, w, &bk);
    vkt_settle();
    cap_end(t, saved, said, sizeof said, "tier adaptive:");
    printf("  adaptive B (shed under pressure): residents 6 -> %d, line \"%s\", worst %.2e\n", dg_res(), said, worst);
    CHECK(strstr(said, "shed 1"), "adaptive B: the shed line (\"%s\")", said);
    CHECK(dg_res() == 5, "adaptive B: %d residents, not the warm 6 - 1", dg_res());
    CHECK(worst < 2e-3, "adaptive B: rows off the reference (%.3g)", worst);
    vkt_shutdown();
    /* --- leg C: the env unset -- no arm, no decision line, the warm set stands --- */
    unsetenv("COLI_VK_TIER_ADAPTIVE"); unsetenv("COLI_VK_TIER_FREE_MB");
    setenv("COLI_VK_TIER_RATE", "64", 1);
    set_budget(6, f, f);
    vc = cfg_of(f, f, VKT_ACT_SWIGLU, 0);
    CHECK(vkt_init(&vc, rows), "adaptive C: the tier did not start");
    n = vkt_plan(pl, pe, L * E);
    for (int i = 0; i < n; i++) { VktExpertSrc s = src_of(&ex[pl[i]][pe[i]]); vkt_put(pl[i], pe[i], &s); }
    vkt_put_done();
    bk = (Books){0, 0};
    t = cap_begin(&saved);
    vkt_begin_forward();
    worst = step(0, 1, seqA, w, &bk);
    vkt_settle();
    cap_end(t, saved, said, sizeof said, "tier adaptive:");
    printf("  adaptive C (default off): residents %d, no adaptive line\n", dg_res());
    CHECK(!said[0], "adaptive C: the unset default printed an adaptive line (\"%s\")", said);
    CHECK(dg_res() == 6, "adaptive C: the default changed the warm set (%d residents)", dg_res());
    vkt_shutdown();
    unsetenv("COLI_VK_TIER_ADAPTIVE_RESERVE_GB"); unsetenv("COLI_VK_TIER_ADAPTIVE_MARGIN_GB");
    unsetenv("COLI_VK_TIER_SYNC"); unsetenv("COLI_VK_TIER_RATE");
    model_free();
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
                double r = stream_step(l, S, idx, w, v4, &bk, &streamed, &kept, t >= 3, NULL);
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
                    double r = mode == 2 ? stream_step(l, S, idx, w, 0, &bk, &streamed, &kept, 0, NULL) : step(l, S, idx, w, &bk);
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
/* S1, the group reader's contract: a batch hook may return any prefix and cut a group
 * anywhere, and whichever expert's bytes the tier asked for must land in that expert's
 * slot -- so the same steps over the same model must give the same output rows with the
 * serial loader and with batch loaders cut at one, two and four experts, byte for byte,
 * and no hold may outlive the step that asked for it. */
static void stream_batch(void) {
    VktFmt f = {VKT_SRC_I4U_PAIRS_GS, 64};
    model_make(f, f); g_act = VKT_ACT_SWIGLU; g_limit = 0;
    set_budget(16, f, f);
    setenv("COLI_VK_TIER_RATE", "0", 1);
    setenv("COLI_VK_TIER_STREAM_SLOTS", "8", 1);
    setenv("COLI_VK_TIER_STREAM_ROWS", "3", 1);   /* a cold expert with 3 rows or more streams */
    setenv("COLI_VK_TIER_FILL", "0", 1);          /* two warm a layer, the rest cold: what streams */
    VktConfig vc = cfg_of(f, f, VKT_ACT_SWIGLU, 0);
    vc.load = load_live_cb; vc.release = release_live_cb;
    uint32_t hist[L][E], *rows[L];
    for (int l = 0; l < L; l++) { rows[l] = hist[l]; for (int e = 0; e < E; e++) hist[l][e] = e < 2 ? 100 - e : 0; }
    static const int Ss[] = {48, 33, 64, 40};
    enum { T = sizeof Ss / sizeof *Ss };
    float *refs[T * L] = {0};
    int rows_of[T * L] = {0};
    for (int mode = 0; mode < 4; mode++) {   /* the serial loader, then batches cut at 1, 2 and 3: the sub-batches hold 4 */
        g_cut = mode ? mode : 1; g_holds_live = 0; g_cut_partial = 0;
        vc.load_batch = mode ? load_cut_cb : NULL;
        rng = 7;                     /* the same routing and the same inputs every mode */
        int on = vkt_init(&vc, rows);
        CHECK(on, "stream batch %d: the tier did not start", mode);
        if (!on) break;
        int pl[L * E], pe[L * E], np = vkt_plan(pl, pe, L * E);
        for (int i = 0; i < np; i++) { VktExpertSrc s = src_of(&ex[pl[i]][pe[i]]); vkt_put(pl[i], pe[i], &s); }
        vkt_put_done();
        unsigned long long streamed = 0;
        float *mine[T * L];
        int at = 0;
        for (int t = 0; t < T; t++)
            for (int l = 0; l < L; l++) {
                int S = Ss[t], idx[64 * K]; float w[64 * K];
                route(S, 0, E, idx, w);
                vkt_begin_forward();
                Books bk = {0, 0}; unsigned long long kept = 0;
                mine[at] = malloc(sizeof(float) * (size_t)S * H);
                rows_of[at] = S;
                stream_step(l, S, idx, w, 0, &bk, &streamed, &kept, 0, mine[at]);
                CHECK(g_holds_live == 0, "stream batch %d: %d holds outlive a step", mode, g_holds_live);
                at++;
            }
        if (mode)
            for (int i = 0; i < T * L; i++) {
                CHECK(refs[i] && mine[i] &&
                      memcmp(refs[i], mine[i], sizeof(float) * (size_t)rows_of[i] * H) == 0,
                      "stream batch %d: step %d's rows differ from the serial load's", mode, i);
                free(mine[i]);
            }
        else
            for (int i = 0; i < T * L; i++) refs[i] = mine[i];
        CHECK(streamed > 0, "stream batch %d: nothing streamed", mode);
        if (mode) CHECK(g_cut_partial > 0, "stream batch %d: no partial batch was exercised", mode);
        printf("  %s: %llu assignments streamed, %d partial batches\n", mode ? "cut" : "serial", streamed, g_cut_partial);
        vkt_shutdown();
    }
    for (int i = 0; i < T * L; i++) free(refs[i]);
    model_free();
    unsetenv("COLI_VK_TIER_RATE"); unsetenv("COLI_VK_TIER_STREAM_SLOTS");
    unsetenv("COLI_VK_TIER_STREAM_ROWS"); unsetenv("COLI_VK_TIER_FILL");
}
/* COLI_VK_TIER_STREAM_ROWS_ADAPTIVE: the R rule's floor. The same routing twice --
 * six cold experts of exactly 1 row, two of 21 -- once with R fixed at 3 (the tests'
 * usual lever) and once with the adaptive rule armed before any measurement exists
 * (R=1): the 1-row experts the fixed rule keeps on the CPU stream under the adaptive
 * one, and every output row is byte-identical either way (the rule picks who computes,
 * never how). One step, so the measurements the later rule would use never land. */
static void stream_rows_adaptive(void) {
    VktFmt f = {VKT_SRC_I4U_PAIRS_GS, 64};
    model_make(f, f); g_act = VKT_ACT_SWIGLU; g_limit = 0;
    set_budget(16, f, f);
    setenv("COLI_VK_TIER_RATE", "0", 1);
    setenv("COLI_VK_TIER_STREAM_SLOTS", "8", 1);
    setenv("COLI_VK_TIER_FILL", "0", 1);
    uint32_t hist[L][E], *rows[L];
    for (int l = 0; l < L; l++) { rows[l] = hist[l]; for (int e = 0; e < E; e++) hist[l][e] = e < 2 ? 100 - e : 0; }
    enum { S = 16, N = S * K };   /* a big step (S above the GEMM's rows), 48 assignments */
    int idx[N]; float w[N];
    for (int i = 0; i < N; i++) w[i] = 0.5f;
    for (int i = 0; i < 6; i++) idx[i] = 2 + i;            /* six cold experts, one row each */
    for (int i = 6; i < N; i++) idx[i] = 8 + (i & 1);      /* two cold experts, 21 rows each */
    float *cap[2] = {0};
    unsigned long long kept[2] = {0, 0}, streamed[2] = {0, 0};
    for (int mode = 0; mode < 2; mode++) {
        if (mode) { unsetenv("COLI_VK_TIER_STREAM_ROWS"); setenv("COLI_VK_TIER_STREAM_ROWS_ADAPTIVE", "1", 1); }
        else { setenv("COLI_VK_TIER_STREAM_ROWS", "3", 1); unsetenv("COLI_VK_TIER_STREAM_ROWS_ADAPTIVE"); }
        rng = 7;   /* the same inputs both legs */
        VktConfig vc = cfg_of(f, f, VKT_ACT_SWIGLU, 0);
        vc.load = load_cb; vc.release = release_cb;
        int on = vkt_init(&vc, rows);
        CHECK(on, "stream rows %d: the tier did not start", mode);
        if (!on) break;
        int pl[L * E], pe[L * E], np = vkt_plan(pl, pe, L * E);
        for (int i = 0; i < np; i++) { VktExpertSrc s = src_of(&ex[pl[i]][pe[i]]); vkt_put(pl[i], pe[i], &s); }
        vkt_put_done();
        CHECK(np == 4, "stream rows %d: %d warm experts", mode, np);
        cap[mode] = malloc(sizeof(float) * (size_t)S * H);
        vkt_begin_forward();
        Books bk = {0, 0};
        double worst = stream_step(0, S, idx, w, 0, &bk, &streamed[mode], &kept[mode], 0, cap[mode]);
        CHECK(worst < 2e-3, "stream rows %d: rows off the reference (%.3g)", mode, worst);
        CHECK(bk.dev == (mode ? N : N - 6), "stream rows %d: %llu of %d assignments on the device", mode, bk.dev, N);
        vkt_shutdown();
    }
    CHECK(kept[0] == 6 && streamed[0] == N - 6,
          "stream rows fixed: the rule kept %llu rows and streamed %llu (6 and 42 expected)", kept[0], streamed[0]);
    CHECK(kept[1] == 0 && streamed[1] == N,
          "stream rows adaptive: R=1 kept %llu rows and streamed %llu (0 and %d expected)", kept[1], streamed[1], N);
    CHECK(cap[0] && cap[1] && rel(cap[0], cap[1], S * H) < 2e-3,
          "stream rows adaptive: the outputs differ from the fixed rule's (%.3g)",
          cap[0] && cap[1] ? rel(cap[0], cap[1], S * H) : -1.0);
    printf("  fixed R=3: %llu of %d assignments streamed (6 one-row experts kept); adaptive R=1: %llu streamed, rows equal within %.1e\n",
           streamed[0], N, streamed[1], cap[0] && cap[1] ? rel(cap[0], cap[1], S * H) : -1.0);
    free(cap[0]); free(cap[1]);
    model_free();
    unsetenv("COLI_VK_TIER_RATE"); unsetenv("COLI_VK_TIER_STREAM_SLOTS");
    unsetenv("COLI_VK_TIER_STREAM_ROWS"); unsetenv("COLI_VK_TIER_STREAM_ROWS_ADAPTIVE");
    unsetenv("COLI_VK_TIER_FILL");
}
/* COLI_VK_TIER_STREAM_GROUP: the inline fill groups' width. Two tiers over the same
 * routing -- every cold expert well above R, so all eight stream -- once at the default
 * (a group is the sub-batch's remaining room: 4 of the 8 slots) and once with groups
 * of 8: the reader's asks go 4-wide to 8-wide, the outputs byte-equal (the schedule
 * moved, nothing else: which sub-batch computes what is the nst_b rule's, unchanged). */
static void stream_group(void) {
    VktFmt f = {VKT_SRC_I4U_PAIRS_GS, 64};
    model_make(f, f); g_act = VKT_ACT_SWIGLU; g_limit = 0;
    set_budget(16, f, f);
    setenv("COLI_VK_TIER_RATE", "0", 1);
    setenv("COLI_VK_TIER_STREAM_SLOTS", "8", 1);   /* st_per 4: the default group's room */
    setenv("COLI_VK_TIER_STREAM_ROWS", "3", 1);
    setenv("COLI_VK_TIER_FILL", "0", 1);
    uint32_t hist[L][E], *rows[L];
    for (int l = 0; l < L; l++) { rows[l] = hist[l]; for (int e = 0; e < E; e++) hist[l][e] = e < 2 ? 100 - e : 0; }
    enum { S = 48, N = S * K };   /* 144 assignments over the 8 cold experts, all well above R */
    int idx[N]; float w[N];
    float *cap[2] = {0};
    int ask_max[2] = {0, 0}, asks[2] = {0, 0};
    unsigned long long streamed[2] = {0, 0}, kept[2] = {0, 0};
    for (int mode = 0; mode < 2; mode++) {
        if (mode) setenv("COLI_VK_TIER_STREAM_GROUP", "8", 1); else unsetenv("COLI_VK_TIER_STREAM_GROUP");
        rng = 7;   /* the same routing and inputs both legs */
        VktConfig vc = cfg_of(f, f, VKT_ACT_SWIGLU, 0);
        vc.load = load_cb; vc.release = release_cb; vc.load_batch = load_group_cb;
        g_group_ask_max = g_group_asks = 0; g_group_ask_rows = 0;
        int on = vkt_init(&vc, rows);
        CHECK(on, "stream group %d: the tier did not start", mode);
        if (!on) break;
        int pl[L * E], pe[L * E], np = vkt_plan(pl, pe, L * E);
        for (int i = 0; i < np; i++) { VktExpertSrc s = src_of(&ex[pl[i]][pe[i]]); vkt_put(pl[i], pe[i], &s); }
        vkt_put_done();
        route(S, 2, 8, idx, w);
        cap[mode] = malloc(sizeof(float) * (size_t)S * H);
        vkt_begin_forward();
        Books bk = {0, 0};
        double worst = stream_step(0, S, idx, w, 0, &bk, &streamed[mode], &kept[mode], 0, cap[mode]);
        CHECK(worst < 2e-3, "stream group %d: rows off the reference (%.3g)", mode, worst);
        CHECK(kept[mode] == 0 && streamed[mode] == N,
              "stream group %d: %llu streamed, %llu kept (%d and 0 expected)", mode, streamed[mode], kept[mode], N);
        ask_max[mode] = g_group_ask_max; asks[mode] = g_group_asks;
        if (mode) vkt_report("test", 0, 0);   /* the DPROF line's groups-of-N experts, when armed */
        vkt_shutdown();
    }
    CHECK(ask_max[0] <= 4, "stream group default: a fill group asked %d wide (the sub-batch's room is 4)", ask_max[0]);
    CHECK(ask_max[1] == 8, "stream group armed: the widest ask %d, not the 8 the group allows", ask_max[1]);
    CHECK(asks[1] <= asks[0], "stream group armed: %d asks against the default's %d", asks[1], asks[0]);
    CHECK(cap[0] && cap[1] && memcmp(cap[0], cap[1], sizeof(float) * (size_t)S * H) == 0,
          "stream group: the outputs differ between the room-bound and the wide groups");
    printf("  default: fill asks %d wide (%d asks); armed 8: %d wide (%d asks), outputs byte-identical\n",
           ask_max[0], asks[0], ask_max[1], asks[1]);
    free(cap[0]); free(cap[1]);
    model_free();
    unsetenv("COLI_VK_TIER_RATE"); unsetenv("COLI_VK_TIER_STREAM_SLOTS");
    unsetenv("COLI_VK_TIER_STREAM_ROWS"); unsetenv("COLI_VK_TIER_STREAM_GROUP");
    unsetenv("COLI_VK_TIER_FILL");
}
/* COLI_VK_TIER_STREAM_PRED: the prefetch plan's coverage. A layer's prediction is
 * established by one step, then the next same-shape step's prefetch runs twice -- at
 * the default floor (R) and tightened (3x): the plan keeps only the experts whose
 * predicted rows clear the higher floor, the trimmed ones fall to the inline fill at
 * the step, and the step's rows are byte-identical either way. */
static void stream_pred(void) {
    VktFmt f = {VKT_SRC_I4U_PAIRS_GS, 64};
    model_make(f, f); g_act = VKT_ACT_SWIGLU; g_limit = 0;
    set_budget(16, f, f);
    setenv("COLI_VK_TIER_RATE", "0", 1);
    setenv("COLI_VK_TIER_STREAM_SLOTS", "8", 1);
    setenv("COLI_VK_TIER_STREAM_ROWS", "3", 1);
    setenv("COLI_VK_TIER_FILL", "0", 1);
    uint32_t hist[L][E], *rows[L];
    for (int l = 0; l < L; l++) { rows[l] = hist[l]; for (int e = 0; e < E; e++) hist[l][e] = e < 2 ? 100 - e : 0; }
    enum { S = 16, N = S * K };
    int idx[N]; float w[N];
    /* experts 2..9 all above R (3): their predicted rows are 6,6,6,6,6,6,6,6 -- one of
     * them (9 rows) is the only one above a 3x floor */
    for (int i = 0; i < N; i++) w[i] = 0.5f;
    for (int i = 0; i < 9; i++) idx[i] = 9;                      /* the one hot expert: 9 rows */
    for (int i = 9; i < 15; i++) idx[i] = 2;
    for (int i = 15; i < 21; i++) idx[i] = 3;
    for (int i = 21; i < 27; i++) idx[i] = 4;
    for (int i = 27; i < 33; i++) idx[i] = 5;
    for (int i = 33; i < 39; i++) idx[i] = 6;
    for (int i = 39; i < 45; i++) idx[i] = 7;
    for (int i = 45; i < 48; i++) idx[i] = 8;
    float *cap[2] = {0};
    int got[2] = {0, 0}, loads[2] = {0, 0};
    for (int mode = 0; mode < 2; mode++) {
        if (mode) setenv("COLI_VK_TIER_STREAM_PRED", "3", 1); else unsetenv("COLI_VK_TIER_STREAM_PRED");
        rng = 7;
        VktConfig vc = cfg_of(f, f, VKT_ACT_SWIGLU, 0);
        vc.load = load_cb; vc.release = release_cb;
        int on = vkt_init(&vc, rows);
        CHECK(on, "stream pred %d: the tier did not start", mode);
        if (!on) break;
        int pl[L * E], pe[L * E], np = vkt_plan(pl, pe, L * E);
        for (int i = 0; i < np; i++) { VktExpertSrc s = src_of(&ex[pl[i]][pe[i]]); vkt_put(pl[i], pe[i], &s); }
        vkt_put_done();
        cap[mode] = malloc(sizeof(float) * (size_t)S * H);
        {   /* the prediction's step: same shape, no prefetch yet */
            Books bk = {0, 0}; unsigned long long st = 0, kp = 0;
            vkt_begin_forward();
            stream_step(1, S, idx, w, 0, &bk, &st, &kp, 0, NULL);
        }
        g_loads = 0;
        got[mode] = vkt_stream_prefetch(1, S);
        int pref_loads = g_loads;   /* the prefetch's own reads, apart from the step's */
        Books bk = {0, 0}; unsigned long long streamed = 0, kept = 0;
        vkt_begin_forward();
        double worst = stream_step(1, S, idx, w, 0, &bk, &streamed, &kept, 0, cap[mode]);
        CHECK(worst < 2e-3, "stream pred %d: rows off the reference (%.3g)", mode, worst);
        CHECK(streamed == N && kept == 0, "stream pred %d: %llu streamed, %llu kept", mode, streamed, kept);
        CHECK(pref_loads == got[mode], "stream pred %d: the prefetch read %d of the %d it filled", mode, pref_loads, got[mode]);
        loads[mode] = g_loads - pref_loads;
        vkt_shutdown();
    }
    CHECK(got[0] == 8, "stream pred default: the plan prefetched %d of the 8 above-R experts", got[0]);
    CHECK(got[1] == 1, "stream pred armed: the 3x floor kept %d experts (the one 9-row expert expected)", got[1]);
    CHECK(loads[0] == 0 && loads[1] == 7,
          "stream pred: %d inline loads at the default, %d armed (the trimmed fall back)", loads[0], loads[1]);
    CHECK(cap[0] && cap[1] && memcmp(cap[0], cap[1], sizeof(float) * (size_t)S * H) == 0,
          "stream pred: the outputs differ between the floors");
    printf("  default floor R: %d of 8 prefetched, %d step loads; 3x floor: %d prefetched, %d step loads, outputs byte-identical\n",
           got[0], loads[0], got[1], loads[1]);
    free(cap[0]); free(cap[1]);
    model_free();
    unsetenv("COLI_VK_TIER_RATE"); unsetenv("COLI_VK_TIER_STREAM_SLOTS");
    unsetenv("COLI_VK_TIER_STREAM_ROWS"); unsetenv("COLI_VK_TIER_STREAM_PRED");
    unsetenv("COLI_VK_TIER_FILL");
}
/* ---- the stream ring (COLI_VK_TIER_STREAM_RING) ---------------------------------- */
/* The ring's own reader: a counting hook that can cut its batches (the partial-batch
 * contract), park inside a read until the test lets it go or a bound passes (the
 * timing legs: a step of another layer must coexist with the running job, the sync
 * drain must join it, the teardown must cancel it), or refuse everything (the inline
 * fallback). g_ring_layer/g_ring_eids record the order one watched job filled. */
static int g_ring_calls, g_ring_n, g_ring_cut, g_ring_fail, g_ring_park_ms;
static volatile int g_ring_go, g_ring_in;
static int g_ring_layer = -1, g_ring_eids[64];
static double ring_now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }
static int ring_batch_cb(void *ctx, int layer, const int *eids, int n, VktExpertSrc *srcs, void **hs) {
    (void)ctx;
    __atomic_add_fetch(&g_ring_calls, 1, __ATOMIC_RELAXED);
    __atomic_store_n(&g_ring_in, 1, __ATOMIC_RELAXED);
    for (int waited = 0; !__atomic_load_n(&g_ring_go, __ATOMIC_RELAXED) && waited < g_ring_park_ms; waited++)
        usleep(1000);
    __atomic_store_n(&g_ring_in, 0, __ATOMIC_RELAXED);
    if (g_ring_fail) return 0;
    int got = g_ring_cut > 0 && n > g_ring_cut ? g_ring_cut : n;
    if (got > 64) got = 64;
    for (int i = 0; i < got; i++) {
        srcs[i] = src_of(&ex[layer][eids[i]]); hs[i] = &ex[layer][eids[i]];
        if (layer == g_ring_layer && g_ring_n < 64) g_ring_eids[g_ring_n++] = eids[i];
    }
    return got;
}
/* A routing that keeps a stable candidate set: experts 2..5 routed three rows or
 * more each (the four the ring can hold, st_per 4), the residents and the rest cold
 * and below the rule. Fills g_ring_idx/g_ring_w and returns the per-expert counts. */
static int g_ring_idx[64 * K]; static float g_ring_w[64 * K];
static void route_ring(int cnt[E]) {
    memset(cnt, 0, E * sizeof(int));
    for (int s = 0; s < 48; s++) {
        for (int k = 0; k < K; k++) {
            int e = 2 + (s / 12 + 3 * k) % 4;
            cnt[e]++;
            g_ring_idx[s * K + k] = e;
            g_ring_w[s * K + k] = 0.2f + (float)((s + k) % 5) / 10.0f;
        }
    }
}
/* The cold branch's routing: experts 2, 3 and 4 round-robin and 9 on every third slot
 * (48 rows, the head by count), so the prediction's set (9 in, 5 out) is not the
 * heat's -- the two plans the cold leg must tell apart. */
static int g_cold_idx[64 * K]; static float g_cold_w[64 * K];
static void route_cold(int cnt[E]) {
    memset(cnt, 0, E * sizeof(int));
    for (int s = 0; s < 48; s++)
        for (int k = 0; k < K; k++) {
            int e = k == 2 ? 9 : 2 + (s / 3 + k) % 3;
            cnt[e]++;
            g_cold_idx[s * K + k] = e;
            g_cold_w[s * K + k] = 0.2f + (float)((s + k) % 5) / 10.0f;
        }
}
static void stream_ring(void) {
    VktFmt f = {VKT_SRC_I4U_PAIRS_GS, 64};
    model_make(f, f); g_act = VKT_ACT_SWIGLU; g_limit = 0;
    set_budget(16, f, f);
    setenv("COLI_VK_TIER_RATE", "0", 1);
    setenv("COLI_VK_TIER_STREAM_SLOTS", "8", 1);   /* st_per 4: the ring's half */
    setenv("COLI_VK_TIER_STREAM_ROWS", "3", 1);
    setenv("COLI_VK_TIER_FILL", "0", 1);
    uint32_t hist[L][E], *rows[L];
    for (int l = 0; l < L; l++) { rows[l] = hist[l]; for (int e = 0; e < E; e++) hist[l][e] = e < 2 ? 100 - e : 0; }
    /* mode 0: the hook wired, the env off -- today's inline prefetch, the default's own
     * leg; mode 1: the ring on, its reader cutting every batch to one (the partial
     * contract); mode 2: the ring refusing every read -- the inline fill at the step
     * covers. Every step's rows byte-equal mode 0's: the ring changes timing, never
     * bytes. */
    static const int Ss[] = {48, 33, 64, 40};
    enum { T = sizeof Ss / sizeof *Ss };
    float *refs[T * L] = {0};
    int rows_of[T * L] = {0};
    for (int mode = 0; mode < 3; mode++) {
        if (mode) setenv("COLI_VK_TIER_STREAM_RING", "1", 1); else unsetenv("COLI_VK_TIER_STREAM_RING");
        g_ring_cut = mode == 1 ? 1 : 0; g_ring_fail = mode == 2;
        g_ring_park_ms = 0; __atomic_store_n(&g_ring_go, 1, __ATOMIC_RELAXED);
        __atomic_store_n(&g_ring_calls, 0, __ATOMIC_RELAXED);
        rng = 7;
        VktConfig vc = cfg_of(f, f, VKT_ACT_SWIGLU, 0);
        vc.load = load_cb; vc.release = release_cb; vc.ring_batch = ring_batch_cb;
        int on = vkt_init(&vc, rows);
        CHECK(on, "stream ring %d: the tier did not start", mode);
        if (!on) break;
        int pl[L * E], pe[L * E], np = vkt_plan(pl, pe, L * E);
        for (int i = 0; i < np; i++) { VktExpertSrc s = src_of(&ex[pl[i]][pe[i]]); vkt_put(pl[i], pe[i], &s); }
        vkt_put_done();
        unsigned long long streamed = 0;
        float *mine[T * L];
        int at = 0;
        for (int t = 0; t < T; t++)
            for (int l = 0; l < L; l++) {
                int S = Ss[t], idx[64 * K]; float w[64 * K];
                route(S, 0, E, idx, w);
                vkt_begin_forward();
                Books bk = {0, 0}; unsigned long long kept = 0;
                mine[at] = malloc(sizeof(float) * (size_t)S * H);
                rows_of[at] = S;
                g_loads = 0;
                stream_step(l, S, idx, w, 0, &bk, &streamed, &kept, 1, mine[at]);
                at++;
            }
        if (mode)
            for (int i = 0; i < T * L; i++) {
                CHECK(refs[i] && mine[i] &&
                      memcmp(refs[i], mine[i], sizeof(float) * (size_t)rows_of[i] * H) == 0,
                      "stream ring %d: step %d's rows differ from the inline prefetch's", mode, i);
                free(mine[i]);
            }
        else
            for (int i = 0; i < T * L; i++) refs[i] = mine[i];
        CHECK(streamed > 0, "stream ring %d: nothing streamed", mode);
        if (mode) {
            CHECK(__atomic_load_n(&g_ring_calls, __ATOMIC_RELAXED) > 0, "stream ring %d: the ring never read", mode);
            CHECK(g_loads > 0, "stream ring %d: the inline fallback never engaged", mode);
        } else
            CHECK(__atomic_load_n(&g_ring_calls, __ATOMIC_RELAXED) == 0,
                  "stream ring off: the ring's reader was called anyway");
        printf("  %s: %llu assignments streamed, %d ring reads, %d inline loads\n",
               mode ? (mode == 2 ? "refusing" : "cutting") : "inline", streamed,
               __atomic_load_n(&g_ring_calls, __ATOMIC_RELAXED), g_loads);
        vkt_shutdown();
    }
    for (int i = 0; i < T * L; i++) free(refs[i]);
    model_free();

    /* --- the pipeline: the prediction's order, one layer ahead, taken at the next
     * prefetch and consumed by the step without one inline load ------------------- */
    setenv("COLI_VK_TIER_STREAM_RING", "1", 1);
    g_ring_cut = 0; g_ring_fail = 0; g_ring_park_ms = 0; __atomic_store_n(&g_ring_go, 1, __ATOMIC_RELAXED);
    model_make(f, f);
    rng = 3;
    { VktConfig vc = cfg_of(f, f, VKT_ACT_SWIGLU, 0);
      vc.load = load_cb; vc.release = release_cb; vc.ring_batch = ring_batch_cb;
      CHECK(vkt_init(&vc, rows), "ring pipeline: the tier did not start"); }
    { int pl[L * E], pe[L * E], np = vkt_plan(pl, pe, L * E);
      for (int i = 0; i < np; i++) { VktExpertSrc s = src_of(&ex[pl[i]][pe[i]]); vkt_put(pl[i], pe[i], &s); }
      vkt_put_done(); }
    int cnt[E];
    route_ring(cnt);
    Books bk = {0, 0}; unsigned long long streamed = 0, kept = 0;
    vkt_begin_forward();
    stream_step(1, 48, g_ring_idx, g_ring_w, 0, &bk, &streamed, &kept, 0, NULL);   /* pred[1] established */
    int order[E], no = 0;   /* the plan's own rule: est desc, equals keep ascending id */
    for (int e = 0; e < E; e++)
        if (cnt[e] >= 3 && !vkt_resident(1, e)) {
            int at = no++;
            while (at > 0 && cnt[order[at - 1]] < cnt[e]) { order[at] = order[at - 1]; at--; }
            order[at] = e;
        }
    g_ring_layer = 1; g_ring_n = 0;
    int handed = vkt_stream_prefetch(0, 48);
    CHECK(handed == no && handed > 0, "ring pipeline: %d experts handed, %d predicted", handed, no);
    stream_step(0, 48, g_ring_idx, g_ring_w, 0, &bk, &streamed, &kept, 0, NULL);   /* coexists, no join */
    vkt_stream_prefetch(1, 48);   /* the take; the last layer submits nothing */
    g_loads = 0;
    stream_step(1, 48, g_ring_idx, g_ring_w, 0, &bk, &streamed, &kept, 0, NULL);
    CHECK(g_ring_n == handed, "ring pipeline: %d of %d handed experts filled", g_ring_n, handed);
    CHECK(g_loads == 0, "ring pipeline: %d inline loads though the ring held the step's set", g_loads);
    int order_ok = g_ring_n == handed;
    for (int i = 0; order_ok && i < handed; i++) order_ok &= g_ring_eids[i] == order[i];
    CHECK(order_ok, "ring pipeline: the fill order is not the prediction's (first %d vs %d)",
          g_ring_n > 0 ? g_ring_eids[0] : -1, no > 0 ? order[0] : -1);
    printf("  pipeline: %d experts one layer ahead, order", handed);
    for (int i = 0; i < handed; i++) printf(" %d", g_ring_eids[i]);
    printf("\n");
    vkt_shutdown(); model_free();
    g_ring_layer = -1;

    /* --- the join waits only for what it needs: a step of another layer runs beside
     * the parked job; the next prefetch takes it ---------------------------------- */
    model_make(f, f);
    rng = 5;
    { VktConfig vc = cfg_of(f, f, VKT_ACT_SWIGLU, 0);
      vc.load = load_cb; vc.release = release_cb; vc.ring_batch = ring_batch_cb;
      CHECK(vkt_init(&vc, rows), "ring coexist: the tier did not start"); }
    { int pl[L * E], pe[L * E], np = vkt_plan(pl, pe, L * E);
      for (int i = 0; i < np; i++) { VktExpertSrc s = src_of(&ex[pl[i]][pe[i]]); vkt_put(pl[i], pe[i], &s); }
      vkt_put_done(); }
    route_ring(cnt);
    vkt_begin_forward();
    stream_step(1, 48, g_ring_idx, g_ring_w, 0, &bk, &streamed, &kept, 0, NULL);
    __atomic_store_n(&g_ring_go, 0, __ATOMIC_RELAXED);
    g_ring_park_ms = 1500; __atomic_store_n(&g_ring_calls, 0, __ATOMIC_RELAXED);
    g_ring_layer = 1; g_ring_n = 0;
    handed = vkt_stream_prefetch(0, 48);
    CHECK(handed > 0, "ring coexist: nothing handed");
    double t0 = ring_now();
    stream_step(0, 48, g_ring_idx, g_ring_w, 0, &bk, &streamed, &kept, 0, NULL);
    double wall = ring_now() - t0;
    CHECK(wall < 700, "ring coexist: a step of another layer waited %.0f ms on the running job", wall);
    CHECK(__atomic_load_n(&g_ring_calls, __ATOMIC_RELAXED) == 1,
          "ring coexist: the job went past its parked read (%d calls)",
          __atomic_load_n(&g_ring_calls, __ATOMIC_RELAXED));
    __atomic_store_n(&g_ring_go, 1, __ATOMIC_RELAXED);
    vkt_stream_prefetch(1, 48);   /* the take: the read this step needs */
    CHECK(g_ring_n == handed, "ring coexist: the take did not finish the job (%d of %d)", g_ring_n, handed);
    g_loads = 0;
    stream_step(1, 48, g_ring_idx, g_ring_w, 0, &bk, &streamed, &kept, 0, NULL);
    CHECK(g_loads == 0, "ring coexist: %d inline loads after the take", g_loads);
    printf("  coexist: the other layer's step ran beside the parked job, the take finished it\n");
    vkt_shutdown(); model_free();
    g_ring_layer = -1; g_ring_park_ms = 0;

    /* --- COLI_VK_TIER_SYNC's drain: the quiescent point joins the ring too -------- */
    setenv("COLI_VK_TIER_SYNC", "1", 1);
    model_make(f, f);
    rng = 9;
    { VktConfig vc = cfg_of(f, f, VKT_ACT_SWIGLU, 0);
      vc.load = load_cb; vc.release = release_cb; vc.ring_batch = ring_batch_cb;
      CHECK(vkt_init(&vc, rows), "ring sync: the tier did not start"); }
    { int pl[L * E], pe[L * E], np = vkt_plan(pl, pe, L * E);
      for (int i = 0; i < np; i++) { VktExpertSrc s = src_of(&ex[pl[i]][pe[i]]); vkt_put(pl[i], pe[i], &s); }
      vkt_put_done(); }
    route_ring(cnt);
    vkt_begin_forward();
    stream_step(1, 48, g_ring_idx, g_ring_w, 0, &bk, &streamed, &kept, 0, NULL);
    __atomic_store_n(&g_ring_go, 0, __ATOMIC_RELAXED);
    g_ring_park_ms = 1000;
    vkt_stream_prefetch(0, 48);
    t0 = ring_now();
    vkt_settle();   /* a quiescent point: the sync drain joins the worker */
    wall = ring_now() - t0;
    CHECK(wall >= 400, "ring sync: the quiescent point did not wait for the ring (%.0f ms)", wall);
    g_loads = 0;
    stream_step(1, 48, g_ring_idx, g_ring_w, 0, &bk, &streamed, &kept, 0, NULL);
    CHECK(g_loads == 0, "ring sync: %d inline loads after the drain", g_loads);
    printf("  sync drain: the quiescent point waited out the parked read for the ring\n");
    vkt_shutdown(); model_free();
    unsetenv("COLI_VK_TIER_SYNC");
    __atomic_store_n(&g_ring_go, 1, __ATOMIC_RELAXED); g_ring_park_ms = 0;

    /* --- the teardown cancels: never wait out a discarded stream (the fork's #4) -- */
    model_make(f, f);
    rng = 13;
    { VktConfig vc = cfg_of(f, f, VKT_ACT_SWIGLU, 0);
      vc.load = load_cb; vc.release = release_cb; vc.ring_batch = ring_batch_cb;
      CHECK(vkt_init(&vc, rows), "ring teardown: the tier did not start"); }
    { int pl[L * E], pe[L * E], np = vkt_plan(pl, pe, L * E);
      for (int i = 0; i < np; i++) { VktExpertSrc s = src_of(&ex[pl[i]][pe[i]]); vkt_put(pl[i], pe[i], &s); }
      vkt_put_done(); }
    route_ring(cnt);
    vkt_begin_forward();
    stream_step(1, 48, g_ring_idx, g_ring_w, 0, &bk, &streamed, &kept, 0, NULL);
    __atomic_store_n(&g_ring_go, 0, __ATOMIC_RELAXED);
    g_ring_park_ms = 300; g_ring_cut = 1;   /* 4 reads of 300 ms: 1200 ms the teardown must not pay */
    __atomic_store_n(&g_ring_calls, 0, __ATOMIC_RELAXED);
    CHECK(vkt_stream_prefetch(0, 48) > 0, "ring teardown: nothing handed");
    usleep(100000);   /* the worker is inside its first read */
    t0 = ring_now();
    vkt_shutdown();
    wall = ring_now() - t0;
    int calls = __atomic_load_n(&g_ring_calls, __ATOMIC_RELAXED);
    CHECK(wall < 900, "ring teardown: waited %.0f ms on a discarded stream", wall);
    CHECK(calls <= 2, "ring teardown: %d reads ran after the cancel", calls);
    printf("  teardown: cancelled, bounded to the read in flight\n");
    model_free();

    /* --- COLI_VK_TIER_STREAM_RING_COLD: the cold branch fills, not the inline --------
     * A next layer with no prediction behind it (the prompt's first chunk) is the landed
     * ring's one gap: it hands nothing over and the whole fill runs inline at the step.
     * COLD plans it by its heat instead -- the next layer's whole non-resident set, the
     * cap (st_per 4) and the half unchanged -- and where a prediction exists the plan
     * stays the prediction's. The history gives layer 1's 2 and 3 the only heat above the
     * rule's rows (4..9 below it, 9 the coldest of the four the routing names) while
     * layer 0's outranks them all for the warm start's eight places: layer 1 keeps 0,1
     * resident and 2..9 cold. So the cold plan, whole, is 2,3 then the below-rule tail
     * 4,5 (the fork's unconditional coverage -- a heat plan under the rule would hand
     * only 2,3), and the prediction's set is 9,2,3,4 (9 the head by rows). */
    uint32_t chist[L][E], *crows[L];
    for (int l = 0; l < L; l++) {
        crows[l] = chist[l];
        for (int e = 0; e < E; e++) chist[l][e] = !l ? 100 - e : e < 2 ? 100 - e : (int[]){0, 0, 62, 46, 3, 3, 2, 2, 1, 1}[e];
    }
    int ccnt[E];
    route_cold(ccnt);
    setenv("COLI_VK_TIER_STREAM_RING", "1", 1);
    g_ring_cut = 0; g_ring_fail = 0; g_ring_park_ms = 0; __atomic_store_n(&g_ring_go, 1, __ATOMIC_RELAXED);
    float *cold_ref[2] = {0};   /* the default's rows: COLD changes timing, never bytes */
    model_make(f, f);   /* one model: the two legs' rows are the same experts' */
    for (int cold = 0; cold < 2; cold++) {
        if (cold) setenv("COLI_VK_TIER_STREAM_RING_COLD", "1", 1);
        else unsetenv("COLI_VK_TIER_STREAM_RING_COLD");
        rng = 11;
        { VktConfig vc = cfg_of(f, f, VKT_ACT_SWIGLU, 0);
          vc.load = load_cb; vc.release = release_cb; vc.ring_batch = ring_batch_cb;
          CHECK(vkt_init(&vc, crows), "ring cold %d: the tier did not start", cold); }
        { int pl[L * E], pe[L * E], np = vkt_plan(pl, pe, L * E);
          for (int i = 0; i < np; i++) { VktExpertSrc s = src_of(&ex[pl[i]][pe[i]]); vkt_put(pl[i], pe[i], &s); }
          vkt_put_done(); }
        vkt_begin_forward();
        Books cbk = {0, 0}; unsigned long long cstreamed = 0, ckept = 0;
        float *rows[2] = {malloc(sizeof(float) * (size_t)48 * H), malloc(sizeof(float) * (size_t)48 * H)};
        __atomic_store_n(&g_ring_calls, 0, __ATOMIC_RELAXED);
        g_ring_layer = 1; g_ring_n = 0; g_loads = 0;
        CHECK(stream_step(0, 48, g_cold_idx, g_cold_w, 0, &cbk, &cstreamed, &ckept, 1, rows[0]) < 2e-3,
              "ring cold %d: layer 0's rows are not the reference's", cold);
        /* layer 1's step takes the job at its own prefetch, so the counts are read after it */
        g_loads = 0;   /* only the cold step's own fills count */
        CHECK(stream_step(1, 48, g_cold_idx, g_cold_w, 0, &cbk, &cstreamed, &ckept, 1, rows[1]) < 2e-3,
              "ring cold %d: layer 1's rows are not the reference's", cold);
        if (cold) {
            static const int want[4] = {2, 3, 4, 5};   /* the heat's order, 4 and 5 below the rule */
            int heat_ok = g_ring_n == 4;
            for (int i = 0; heat_ok && i < 4; i++) heat_ok &= g_ring_eids[i] == want[i];
            CHECK(heat_ok, "ring cold: %d experts handed over, the whole non-resident set's four warmest is"
                  " 2,3,4,5 with 4 and 5 below the rule (first %d)", g_ring_n, g_ring_n > 0 ? g_ring_eids[0] : -1);
            CHECK(g_loads == 1, "ring cold: %d inline loads (9 is below the cap, the inline fallback's own)", g_loads);
        } else {
            CHECK(g_ring_n == 0 && __atomic_load_n(&g_ring_calls, __ATOMIC_RELAXED) == 0,
                  "ring cold default: the cold branch handed %d experts over (%d reads)", g_ring_n,
                  __atomic_load_n(&g_ring_calls, __ATOMIC_RELAXED));
            CHECK(g_loads == 4, "ring cold default: %d inline loads, not the four the cold step routed", g_loads);
        }
        if (!cold) cold_ref[0] = rows[0], cold_ref[1] = rows[1];
        else {
            CHECK(memcmp(cold_ref[0], rows[0], sizeof(float) * (size_t)48 * H) == 0 &&
                  memcmp(cold_ref[1], rows[1], sizeof(float) * (size_t)48 * H) == 0,
                  "ring cold: a step's rows differ from the default's (the ring changes timing, never bytes)");
            free(rows[0]); free(rows[1]);
        }
        vkt_shutdown();
        if (cold)
            printf("  cold on: %d experts one layer ahead, 4 and 5 of them below the rule, %d inline load"
                   " at the cold step\n", g_ring_n, g_loads);
        else
            printf("  cold default: nothing handed over, %d inline loads at the cold step\n", g_loads);
    }
    model_free();
    free(cold_ref[0]); free(cold_ref[1]);
    g_ring_layer = -1;

    /* --- with a prediction the plan is the prediction's, COLD armed or not: 9 in, 5 out,
     * and the step consumes the job's set without one inline load ------------------- */
    model_make(f, f);
    rng = 11;
    { VktConfig vc = cfg_of(f, f, VKT_ACT_SWIGLU, 0);
      vc.load = load_cb; vc.release = release_cb; vc.ring_batch = ring_batch_cb;
      CHECK(vkt_init(&vc, crows), "ring cold pred: the tier did not start"); }
    { int pl[L * E], pe[L * E], np = vkt_plan(pl, pe, L * E);
      for (int i = 0; i < np; i++) { VktExpertSrc s = src_of(&ex[pl[i]][pe[i]]); vkt_put(pl[i], pe[i], &s); }
      vkt_put_done(); }
    { Books pbk = {0, 0}; unsigned long long ps = 0, pk = 0;
      vkt_begin_forward();
      stream_step(1, 48, g_cold_idx, g_cold_w, 0, &pbk, &ps, &pk, 0, NULL);   /* pred[1] established */
      g_ring_layer = 1; g_ring_n = 0;
      CHECK(vkt_stream_prefetch(0, 48) == 4, "ring cold pred: the prediction's set is not the four routed");
      stream_step(0, 48, g_cold_idx, g_cold_w, 0, &pbk, &ps, &pk, 0, NULL);   /* coexists, no join */
      vkt_stream_prefetch(1, 48);   /* the take; the last layer submits nothing */
      int porder[E], pn = 0;   /* the plan's own rule: est desc, equals count desc */
      for (int e = 0; e < E; e++)
          if (ccnt[e] >= 3 && !vkt_resident(1, e)) {
              int at = pn++;
              while (at > 0 && ccnt[porder[at - 1]] < ccnt[e]) { porder[at] = porder[at - 1]; at--; }
              porder[at] = e;
          }
      int pred_ok = g_ring_n == pn && pn == 4;
      for (int i = 0; pred_ok && i < pn; i++) pred_ok &= g_ring_eids[i] == porder[i];
      CHECK(pred_ok, "ring cold pred: %d of %d handed, the order is not the prediction's (first %d, want %d)",
            g_ring_n, pn, g_ring_n > 0 ? g_ring_eids[0] : -1, pn > 0 ? porder[0] : -1);
      g_loads = 0;
      CHECK(stream_step(1, 48, g_cold_idx, g_cold_w, 0, &pbk, &ps, &pk, 0, NULL) < 2e-3,
            "ring cold pred: layer 1's rows are not the reference's");
      CHECK(g_loads == 0, "ring cold pred: %d inline loads though the ring held the step's set", g_loads);
      printf("  cold with a prediction: %d experts one layer ahead, the prediction's order", g_ring_n);
      for (int i = 0; i < g_ring_n; i++) printf(" %d", g_ring_eids[i]);
      printf("\n"); }
    vkt_shutdown(); model_free();
    g_ring_layer = -1;

    unsetenv("COLI_VK_TIER_STREAM_RING");
    unsetenv("COLI_VK_TIER_STREAM_RING_COLD");
    unsetenv("COLI_VK_TIER_RATE"); unsetenv("COLI_VK_TIER_STREAM_SLOTS");
    unsetenv("COLI_VK_TIER_STREAM_ROWS"); unsetenv("COLI_VK_TIER_STREAM_FILL");
    __atomic_store_n(&g_ring_go, 1, __ATOMIC_RELAXED); g_ring_cut = 0; g_ring_park_ms = 0; g_ring_fail = 0;
}
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
        double error = stream_step(0, S, idx, w, 0, &books, &streamed, &kept, 0, NULL);
        CHECK(books.dev == 0 && kept == S * K && error == 0,
              "stream fault: partial upload used (GPU %llu, CPU %llu, error %.3g)", books.dev, kept, error);
        books = (Books){0, 0}; streamed = kept = 0;
        vkt_begin_forward();
        error = stream_step(0, S, idx, w, 0, &books, &streamed, &kept, 0, NULL);
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
    int def = S == 1 && vkt_cs_armed();   /* the armed default defers: the finish site's group (the engine's order) */
    int grp = def ? vkt_issue_group(layer, x, S, K, idx, w, NULL, taken) : 0;
    for (int i = 0; i < S * K; i++) {
        if (taken[i]) continue;
        expert_ref(&ex[layer][idx[i]], x + (size_t)(i / K) * H, cpu + (size_t)i * H);
        VktExpertSrc s = src_of(&ex[layer][idx[i]]); vkt_note(layer, idx[i], &s);
    }
    const float *sum = NULL;
    int joined = grp ? vkt_join_sum(dev, w, &sum) : (def ? 1 : (n ? vkt_join_sum(dev, w, &sum) : 1));
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
        {{VKT_SRC_MXFP4_F32, 32}, {VKT_SRC_MXFP4_F32, 32}, 7.0f, "MXFP4 f32 gs32 (the grouped GEMM's, mode 4)"},
        {{VKT_SRC_MXFP4_E8M0, 32}, {VKT_SRC_MXFP4_E8M0, 32}, 7.0f, "MXFP4 ue8m0 gs32 (mode 4, packed scales)"},
    };
    int h0 = H; H = 256;
    int any_grouped = 0, any_summed = 0, any_gemv = 0;
    for (size_t c = 0; c < sizeof cs / sizeof *cs; c++) {
        int mx = cs[c].gu.kind == VKT_SRC_MXFP4_F32 || cs[c].gu.kind == VKT_SRC_MXFP4_E8M0;   /* its GEMM needs no streaming cases */
        for (int mode = 0; mode < (mx ? 1 : 4); mode++) {   /* single, whole, mixed, decode */
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
            any_grouped |= gb > 0 && !mx; any_summed |= summed > 0;   /* the MX GEMM carries no sum contract */
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

/* ---- the compute site (COLI_VK_TIER_EXPERTS, CS-1b) -----------------------------
 * One decode token's routed experts as ONE expert-batch group issued at the finish
 * site: vkt_issue defers the S==1 step, vkt_issue_group submits the residents'
 * cached handles beside zero-copy aliases of imported store slabs (fmt 15/7), and
 * the join retires the group. The device rows run against the same double reference
 * as the ordinary paths -- the v4 MXFP4 class at 2e-2 and the non-fp4 rows at 2e-3,
 * armed and unarmed (the unarmed control runs the ordinary issue/join path). A
 * non-finite row is a failure outright: rel()'s NaN-blind `>` never gets to pass it. */
/* the record layout of one expert's three matrices at 4 KiB offsets: woff/soff and
 * the buffer's whole cap -- every expert of a layer shares it, so a rewrite of the
 * same buffer at the same offsets keeps the cache key (the VA-reuse and scan-skip
 * legs below fill one buffer twice with it) */
static size_t cs_alias_slab_geometry(int layer, int eid, size_t woff[3], size_t soff[3]) {
    const Mat *m[3] = {&ex[layer][eid].g, &ex[layer][eid].u, &ex[layer][eid].d};
    size_t cap = 0;
    for (int k = 0; k < 3; k++) {
        size_t rb = row_bytes(m[k]->f.kind, m[k]->I), wb = rb * (size_t)m[k]->O;
        size_t nb = (size_t)m[k]->O * (((size_t)m[k]->I + 31) / 32);   /* fmt 15: one ue8m0 byte a group */
        size_t sb = (nb + 3) / 4 * 4;
        woff[k] = (cap + 4095) / 4096 * 4096; cap = woff[k] + wb;
        soff[k] = (cap + 4095) / 4096 * 4096; cap = soff[k] + sb;
    }
    return (cap + 4095) / 4096 * 4096;
}
/* one expert's records into a buffer at those offsets (the slab a store slot holds
 * after its read of this expert) */
static void cs_alias_slab_fill(int layer, int eid, void *slab, size_t cap, const size_t woff[3], const size_t soff[3]) {
    const Mat *m[3] = {&ex[layer][eid].g, &ex[layer][eid].u, &ex[layer][eid].d};
    memset(slab, 0, cap);
    for (int k = 0; k < 3; k++) {
        size_t rb = row_bytes(m[k]->f.kind, m[k]->I);
        size_t nb = (size_t)m[k]->O * (((size_t)m[k]->I + 31) / 32);
        memcpy((uint8_t *)slab + woff[k], m[k]->codes, rb * (size_t)m[k]->O);
        memcpy((uint8_t *)slab + soff[k], m[k]->scales, nb);
    }
}
static void cs_alias_src_of(int layer, int eid, const void *slab, size_t cap, const size_t woff[3], const size_t soff[3], VktAliasSrc *src) {
    const Mat *m[3] = {&ex[layer][eid].g, &ex[layer][eid].u, &ex[layer][eid].d};
    memset(src, 0, sizeof(*src));
    src->slab = slab; src->bytes = cap; src->fmt = 15; src->gs = 32;
    for (int k = 0; k < 3; k++) {
        src->woff[k] = woff[k]; src->soff[k] = soff[k];
        src->I[k] = m[k]->I; src->O[k] = m[k]->O;
    }
}
/* One aliased expert's store slab (fmt 15) in its own allocation: *slab_out is the
 * caller's, freed after the join; the imported Vk buffer is the tier's. */
static int cs_alias_slab(int layer, int eid, VktAliasSrc *src, void **slab_out) {
    if (!coli_vk_import_alignment()) return 0;
    size_t woff[3], soff[3], cap = cs_alias_slab_geometry(layer, eid, woff, soff);
    uint8_t *slab = NULL;
    if (posix_memalign(slab_out, 4096, cap) != 0 || !(slab = *slab_out)) return 0;
    cs_alias_slab_fill(layer, eid, slab, cap, woff, soff);
    cs_alias_src_of(layer, eid, slab, cap, woff, soff, src);
    return 1;
}
/* One S==1 step through the compute-site group. w is the reference's route weight
 * (always non-NULL); the device sees it only for the v4 activation, where it is
 * applied on-device. armed selects the deferral path: on, vkt_issue defers and
 * vkt_issue_group submits; off, vkt_issue submits the ordinary batch and the group
 * is never called. *alias_hits, when given, counts taken selections that were not
 * resident -- the aliased entries of the group. */
static double cs_group_step(int layer, const int *idx, const float *w, const VktAliasSrc *src,
                            int v4, float limit, int armed, Books *bk, unsigned long long *alias_hits) {
    int S = 1;
    float *x = malloc((size_t)S * H * sizeof(float)), *xq = malloc((size_t)S * H * sizeof(float));
    float *cpu = calloc((size_t)S * K * H, sizeof(float)), *y = malloc(H * sizeof(float));
    float *out = calloc((size_t)S * H, sizeof(float)), *ref = calloc((size_t)S * H, sizeof(float));
    uint8_t taken[K]; const float *rows[K];
    for (int i = 0; i < S * H; i++) x[i] = frnd() * (v4 ? 4.0f : 1.0f);
    memcpy(xq, x, sizeof(float) * (size_t)S * H);
    if (v4) for (int s = 0; s < S; s++) qdq128(xq + (size_t)s * H, H);
    int n = v4 ? vkt_issue_w(layer, xq, S, K, idx, w, taken)
               : vkt_issue(layer, xq, S, K, idx, taken);
    int took = armed ? vkt_issue_group(layer, xq, S, K, idx, v4 ? w : NULL, src, taken) : 0;
    (void)n;
    for (int i = 0; i < S * K; i++) {
        if (taken[i]) continue;
        if (v4) expert_v4(&ex[layer][idx[i]], xq + (size_t)(i / K) * H, w[i], limit, cpu + (size_t)i * H);
        else expert_ref(&ex[layer][idx[i]], xq + (size_t)(i / K) * H, cpu + (size_t)i * H);
        VktExpertSrc s = src_of(&ex[layer][idx[i]]); vkt_note(layer, idx[i], &s);
    }
    int joined = (armed ? took : n) ? vkt_join(rows) : 1;
    CHECK(joined, "cs: join failed");
    double worst = 0;
    for (int s = 0; s < S; s++)
        for (int k = 0; k < K; k++) {
            int i = s * K + k;
            if (v4) expert_v4(&ex[layer][idx[i]], xq + (size_t)s * H, w[i], limit, y);
            else expert_ref(&ex[layer][idx[i]], xq + (size_t)s * H, y);
            for (int d = 0; d < H; d++) ref[(size_t)s * H + d] += (v4 ? 1.0f : w[i]) * y[d];
            const float *c = taken[i] && joined ? rows[i] : cpu + (size_t)i * H;
            if (taken[i] && joined) {
                float dr[HF_MAX];
                for (int d = 0; d < H; d++) dr[d] = v4 ? bf16r(rows[i][d]) : rows[i][d];
                double r = rel(dr, y, H);
                if (!(r <= worst)) worst = r;   /* a NaN counts as the worst */
                for (int d = 0; d < H; d++) out[(size_t)s * H + d] += (v4 ? 1.0f : w[i]) * dr[d];
                if (alias_hits && !vkt_resident(layer, idx[i])) (*alias_hits)++;
            } else for (int d = 0; d < H; d++) out[(size_t)s * H + d] += (v4 ? 1.0f : w[i]) * c[d];
            bk->routed++; bk->dev += taken[i] != 0;
        }
    double r = rel(out, ref, S * H);
    if (!(r <= worst)) worst = r;
    free(x); free(xq); free(cpu); free(out); free(ref); free(y);
    return worst;
}
/* the report's cs: clause, parsed for the engagement assertion */
static void cs_report(const char *line, unsigned long long *g, unsigned long long *r,
                      unsigned long long *a, unsigned long long *f, unsigned long long *x) {
    *g = *r = *a = *f = *x = 0;
    const char *p = strstr(line, "cs: ");
    if (!p || sscanf(p, "cs: %llu groups, %llu resident + %llu aliased entries on the device, "
                     "%llu cpu-fallback (%llu alias refusals)", g, r, a, f, x) != 5)
        *g = *r = *a = *f = *x = ~0ull;
}
static void cs_group_v4(int armed) {
    rng = 7;
    VktFmt f = {VKT_SRC_MXFP4_E8M0, 32};
    model_make(f, f); g_act = VKT_ACT_SWIGLU_V4; g_limit = 10.0f;
    uint32_t hist[L][E], *rows[L];
    for (int l = 0; l < L; l++) { rows[l] = hist[l]; for (int e = 0; e < E; e++) hist[l][e] = 0; }
    hist[0][0] = 100; hist[0][1] = 90;   /* two resident places: layer 0's experts 0 and 1 */
    lever_pin_save();
    /* gap-wave4: the unarmed half pins =0 -- unset is the ARMED default now, and a
     * deferred step without its finish-site group leaves the residents' rows unjoined */
    if (armed) setenv("COLI_VK_TIER_EXPERTS", "1", 1); else setenv("COLI_VK_TIER_EXPERTS", "0", 1);
    setenv("COLI_VK_TIER_SYNC", "1", 1);
    setenv("COLI_VK_TIER_RATE", "64", 1);
    set_budget(2, f, f);
    VktConfig vc = cfg_of(f, f, VKT_ACT_SWIGLU_V4, 10.0f);
    CHECK(vkt_init(&vc, rows), "cs v4: the tier did not start (%s)", armed ? "armed" : "unarmed");
    int pl[L * E], pe[L * E], n = vkt_plan(pl, pe, L * E);
    for (int i = 0; i < n; i++) { VktExpertSrc s = src_of(&ex[pl[i]][pe[i]]); vkt_put(pl[i], pe[i], &s); }
    vkt_put_done();
    CHECK(n == 2, "cs v4: %d warm of 2 planned (%s)", n, armed ? "armed" : "unarmed");
    VktAliasSrc asrc[E]; void *slabs[E];
    for (int e = 0; e < E; e++) slabs[e] = NULL;
    if (armed) for (int e = 2; e <= 4; e++)
        CHECK(cs_alias_slab(0, e, &asrc[e], &slabs[e]), "cs v4: alias slab for expert %d", e);
    static const int routes[][K] = {{0, 1, 2}, {0, 1, 3}, {1, 2, 4}, {0, 2, 3}};
    float w[K] = {0.5f, 0.3f, 0.2f};
    Books bk = {0, 0}; double worst = 0; unsigned long long aliased = 0;
    for (int t = 0; t < 4; t++) {
        VktAliasSrc src[K];
        for (int k = 0; k < K; k++) {
            int e = routes[t][k];
            src[k] = (armed && e >= 2) ? asrc[e] : (VktAliasSrc){0};
        }
        vkt_begin_forward();
        double r = cs_group_step(0, routes[t], w, src, 1, 10.0f, armed, &bk, &aliased);
        if (!(r <= worst)) worst = r;
    }
    char line[4096]; report_line(line, sizeof line);
    unsigned long long cg, cr, ca, cf, cx; cs_report(line, &cg, &cr, &ca, &cf, &cx);
    if (armed)
        printf("  cs v4 (armed): device %llu of %llu assignments, %llu aliased, worst %.2e, "
               "cs: %llu groups, %llu resident + %llu aliased, %llu fallback, %llu refusals\n",
               bk.dev, bk.routed, aliased, no_nan(worst), cg, cr, ca, cf, cx);
    else
        printf("  cs v4 (unarmed): device %llu of %llu assignments, %llu aliased, worst %.2e\n",
               bk.dev, bk.routed, aliased, no_nan(worst));
    CHECK(bk.dev > 0, "cs v4 (%s): nothing ran on the device", armed ? "armed" : "unarmed");
    if (armed)
        CHECK(cg == 4 && cr == 6 && ca == 6 && cf == 0 && cx == 0,
              "cs v4 armed: the group did not engage (cs: %llu groups, %llu resident + %llu aliased, "
              "%llu fallback, %llu refusals)", cg, cr, ca, cf, cx);
    CHECK(no_nan(worst) < 2e-2, "cs v4 (%s): rows off the reference (%.3g)", armed ? "armed" : "unarmed", no_nan(worst));
    vkt_shutdown();
    for (int e = 0; e < E; e++) free(slabs[e]);
    lever_pin_restore();
    model_free();
}
static void cs_group_i8(int armed) {
    rng = 7;
    VktFmt f = {VKT_SRC_I8_ROW, 0};
    model_make(f, f); g_act = VKT_ACT_SWIGLU; g_limit = 0;
    uint32_t hist[L][E], *rows[L];
    for (int l = 0; l < L; l++) { rows[l] = hist[l]; for (int e = 0; e < E; e++) hist[l][e] = 0; }
    hist[0][0] = 100; hist[0][1] = 90; hist[0][2] = 80;   /* three resident places: the whole routed set */
    lever_pin_save();
    if (armed) setenv("COLI_VK_TIER_EXPERTS", "1", 1); else setenv("COLI_VK_TIER_EXPERTS", "0", 1);   /* gap-wave4: the unarmed half pins =0 (unset arms) */
    setenv("COLI_VK_TIER_SYNC", "1", 1);
    setenv("COLI_VK_TIER_RATE", "64", 1);
    set_budget(3, f, f);
    VktConfig vc = cfg_of(f, f, VKT_ACT_SWIGLU, 0);
    CHECK(vkt_init(&vc, rows), "cs i8: the tier did not start (%s)", armed ? "armed" : "unarmed");
    int pl[L * E], pe[L * E], n = vkt_plan(pl, pe, L * E);
    for (int i = 0; i < n; i++) { VktExpertSrc s = src_of(&ex[pl[i]][pe[i]]); vkt_put(pl[i], pe[i], &s); }
    vkt_put_done();
    CHECK(n == 3, "cs i8: %d warm of 3 planned (%s)", n, armed ? "armed" : "unarmed");
    static const int routes[][K] = {{0, 1, 2}, {2, 0, 1}, {1, 2, 0}};
    float w[K] = {0.5f, 0.3f, 0.2f};
    Books bk = {0, 0}; double worst = 0;
    for (int t = 0; t < 3; t++) {
        vkt_begin_forward();
        double r = cs_group_step(0, routes[t], w, NULL, 0, 0, armed, &bk, NULL);
        if (!(r <= worst)) worst = r;
    }
    char line[4096]; report_line(line, sizeof line);
    unsigned long long cg, cr, ca, cf, cx; cs_report(line, &cg, &cr, &ca, &cf, &cx);
    if (armed)
        printf("  cs i8 (armed): device %llu of %llu assignments, worst %.2e, "
               "cs: %llu groups, %llu resident + %llu aliased, %llu fallback, %llu refusals\n",
               bk.dev, bk.routed, no_nan(worst), cg, cr, ca, cf, cx);
    else
        printf("  cs i8 (unarmed): device %llu of %llu assignments, worst %.2e\n",
               bk.dev, bk.routed, no_nan(worst));
    CHECK(bk.dev > 0, "cs i8 (%s): nothing ran on the device", armed ? "armed" : "unarmed");
    if (armed)
        CHECK(cg == 3 && cr == 9 && ca == 0 && cf == 0 && cx == 0,
              "cs i8 armed: the group did not engage (cs: %llu groups, %llu resident + %llu aliased, "
              "%llu fallback, %llu refusals)", cg, cr, ca, cf, cx);
    CHECK(no_nan(worst) < 2e-3, "cs i8 (%s): rows off the reference (%.3g)", armed ? "armed" : "unarmed", no_nan(worst));
    vkt_shutdown(); model_free();
}
/* CS x decode-growth composition (CS-1b x T2): EXPERTS and DECODE_GROW both armed -- the
 * finish-site group's aliased selections must still install by the decode-only rank
 * (the ordinary path's vkt_note never sees them). Warm two residents, grow two free
 * places, and route the two hot aliases: both install, the below-cutoff alias does
 * not, and the group's rows stay on the reference. */
static void cs_grow(void) {
    lever_pin_save();
    rng = 7;
    VktFmt f = {VKT_SRC_MXFP4_E8M0, 32};
    model_make(f, f); g_act = VKT_ACT_SWIGLU_V4; g_limit = 10.0f;
    const double eb = (double)vkt_expert_bytes(H, F, f, f);
    uint32_t hist[L][E], dhist[L][E], *rows[L], *drows[L];
    for (int l = 0; l < L; l++) {
        rows[l] = hist[l]; drows[l] = dhist[l];
        for (int e = 0; e < E; e++) {
            hist[l][e] = (l == 0 && e < 2) ? 100u : 0u;   /* the warm start: layer 0's experts 0 and 1 */
            dhist[l][e] = (l == 0 && e >= 2 && e <= 4) ? 40u - 10u * (uint32_t)(e - 2) : 0u;
        }
    }
    setenv("COLI_VK_TIER_EXPERTS", "1", 1);
    setenv("COLI_VK_TIER_SYNC", "1", 1);
    setenv("COLI_VK_TIER_RATE", "64", 1);
    set_budget(2, f, f);   /* exactly the two warm residents: no free place to promote into */
    char gb[64]; snprintf(gb, sizeof gb, "%.12f", 2.5 * eb / 1073741824.0);   /* two grown places */
    setenv("COLI_VK_TIER_DECODE_GROW", "1", 1);
    setenv("COLI_VK_TIER_DECODE_GROW_GB", gb, 1);
    setenv("COLI_VK_TIER_DECODE_GROW_RESERVE_GB", "0.000001", 1);
    VktConfig vc = cfg_of(f, f, VKT_ACT_SWIGLU_V4, 10.0f);
    vc.decode_heat = drows;
    CHECK(vkt_init(&vc, rows), "cs grow: the tier did not start");
    int pl[L * E], pe[L * E], n = vkt_plan(pl, pe, L * E);
    for (int i = 0; i < n; i++) { VktExpertSrc s = src_of(&ex[pl[i]][pe[i]]); vkt_put(pl[i], pe[i], &s); }
    vkt_put_done();
    CHECK(n == 2 && dg_res() == 2, "cs grow: %d warm of 2 planned", n);
    VktAliasSrc asrc[E]; void *slabs[E];
    for (int e = 0; e < E; e++) slabs[e] = NULL;
    for (int e = 2; e <= 4; e++)
        CHECK(cs_alias_slab(0, e, &asrc[e], &slabs[e]), "cs grow: alias slab for expert %d", e);
    static const int routes[][K] = {{0, 1, 2}, {0, 1, 3}, {1, 2, 4}, {0, 2, 3}};
    float w[K] = {0.5f, 0.3f, 0.2f};
    Books bk = {0, 0}; double worst = 0; unsigned long long aliased = 0;
    for (int t = 0; t < 4; t++) {
        VktAliasSrc src[K];
        for (int k = 0; k < K; k++) {
            int e = routes[t][k];
            src[k] = e >= 2 ? asrc[e] : (VktAliasSrc){0};
        }
        vkt_begin_forward();
        double r = cs_group_step(0, routes[t], w, src, 1, 10.0f, 1, &bk, &aliased);
        if (!(r <= worst)) worst = r;
        vkt_settle();
    }
    char line[4096]; report_line(line, sizeof line);
    const char *grow = strstr(line, "decode grow: ");
    printf("  cs grow (EXPERTS + DECODE_GROW): residents 2 -> %d, aliased %llu, worst %.2e, \"%s\"\n",
           dg_res(), aliased, no_nan(worst), grow ? grow : "no clause");
    CHECK(dg_res() == 4, "cs grow: %d residents, not the warm 2 + 2", dg_res());
    CHECK(vkt_resident(0, 2) && vkt_resident(0, 3), "cs grow: the hot aliases did not install");
    CHECK(!vkt_resident(0, 4), "cs grow: a below-cutoff alias installed");
    CHECK(strstr(line, "decode grow: 2 installed by decode-only rank (cutoff 30)"), "cs grow: the growth clause");
    CHECK(no_nan(worst) < 2e-2, "cs grow: rows off the reference (%.3g)", no_nan(worst));
    vkt_shutdown();
    for (int e = 0; e < E; e++) free(slabs[e]);
    model_free();
    unsetenv("COLI_VK_TIER_DECODE_GROW"); unsetenv("COLI_VK_TIER_DECODE_GROW_GB");
    unsetenv("COLI_VK_TIER_DECODE_GROW_RESERVE_GB");
    unsetenv("COLI_VK_TIER_EXPERTS");
    lever_pin_restore();
}
/* COLI_VK_TIER_CS_DEVSUM: the whole group's rows summed ON THE DEVICE -- each bf16
 * rounded, added in rank order from 0, exactly the finish site's host merge
 * (deepseek_v4.c's output[i] += coli_bf16_round(dev[k][i])) -- read back as ONE row.
 * The gate is bit-for-bit: each group's device-merged row equals the host merge of the
 * SAME group's rows (memcmp), and the two legs' host merges equal each other (the
 * device rows are deterministic). The unarmed control joins with no merged row at
 * all; a group with a selection left the CPU's (a refused alias: not whole) keeps the
 * rows path, its rows against the reference as cs_group's; the alias cache composes
 * (armed throughout, the keepalive books unaffected by the tail). */
static void cs_devsum(void) {
    VktFmt f = {VKT_SRC_MXFP4_E8M0, 32};
    enum { T = 4 };
    static const int routes[][K] = {{0, 1, 2}, {0, 1, 3}, {1, 2, 4}, {0, 2, 3}};
    float w[K] = {0.5f, 0.3f, 0.2f};
    static float first[T][HF_MAX];   /* the unarmed control's host merges, the armed leg's second yardstick */
    for (int leg = 0; leg < 2; leg++) {
        rng = 7;
        model_make(f, f); g_act = VKT_ACT_SWIGLU_V4; g_limit = 10.0f;
        float xq[T][H];
        for (int t = 0; t < T; t++) {   /* fixed inputs: the same rng stream both legs */
            for (int i = 0; i < H; i++) xq[t][i] = frnd() * 4.0f;
            qdq128(xq[t], H);
        }
        uint32_t hist[L][E], *hrows[L];
        for (int l = 0; l < L; l++) { hrows[l] = hist[l]; for (int e = 0; e < E; e++) hist[l][e] = 0; }
        hist[0][0] = 100; hist[0][1] = 90;   /* two resident places: layer 0's experts 0 and 1 */
        setenv("COLI_VK_TIER_EXPERTS", "1", 1);
        setenv("COLI_VK_TIER_SYNC", "1", 1);
        setenv("COLI_VK_TIER_RATE", "64", 1);
        setenv("COLI_VK_TIER_CS_ALIAS_CACHE", "1", 1);   /* composes: the tail changes nothing of the keepalive */
        if (leg) setenv("COLI_VK_TIER_CS_DEVSUM", "1", 1); else unsetenv("COLI_VK_TIER_CS_DEVSUM");
        set_budget(2, f, f);
        VktConfig vc = cfg_of(f, f, VKT_ACT_SWIGLU_V4, 10.0f);
        CHECK(vkt_init(&vc, hrows), "cs devsum: the tier did not start (%s)", leg ? "armed" : "unarmed");
        int pl[L * E], pe[L * E], n = vkt_plan(pl, pe, L * E);
        for (int i = 0; i < n; i++) { VktExpertSrc s = src_of(&ex[pl[i]][pe[i]]); vkt_put(pl[i], pe[i], &s); }
        vkt_put_done();
        CHECK(n == 2, "cs devsum: %d warm of 2 planned (%s)", n, leg ? "armed" : "unarmed");
        VktAliasSrc asrc[E]; void *slabs[E];
        for (int e = 0; e < E; e++) slabs[e] = NULL;
        for (int e = 2; e <= 4; e++)
            CHECK(cs_alias_slab(0, e, &asrc[e], &slabs[e]), "cs devsum: alias slab for expert %d", e);
        float merged[T][H];
        int summed = 0;
        for (int t = 0; t < T; t++) {
            VktAliasSrc src[K];
            for (int k = 0; k < K; k++) { int e = routes[t][k]; src[k] = e >= 2 ? asrc[e] : (VktAliasSrc){0}; }
            uint8_t taken[K];
            vkt_begin_forward();
            (void)vkt_issue_w(0, xq[t], 1, K, routes[t], w, taken);
            int took = vkt_issue_group(0, xq[t], 1, K, routes[t], w, src, taken);
            CHECK(took == K, "cs devsum: group %d did not fly whole (%d of %d)", t, took, K);
            for (int k = 0; k < K; k++)
                if (!taken[k]) { VktExpertSrc s = src_of(&ex[0][routes[t][k]]); vkt_note(0, routes[t][k], &s); }
            const float *dev[K], *sum = NULL;
            CHECK(vkt_join_merged(dev, &sum), "cs devsum: join %d failed (%s)", t, leg ? "armed" : "unarmed");
            if (leg) CHECK(sum != NULL, "cs devsum: group %d carried no device sum", t);
            else CHECK(!sum, "cs devsum: the unarmed control handed back a merged row");
            summed += sum != NULL;
            memset(merged[t], 0, sizeof merged[t]);
            for (int k = 0; k < K; k++)   /* the finish site's host merge, itself */
                if (taken[k])
                    for (int i = 0; i < H; i++) merged[t][i] += bf16r(dev[k][i]);
            if (sum)
                CHECK(memcmp(sum, merged[t], (size_t)H * sizeof(float)) == 0,
                      "cs devsum: group %d's merged row is not bit-for-bit the host merge", t);
            if (!leg) memcpy(first[t], merged[t], (size_t)H * sizeof(float));
            else CHECK(memcmp(merged[t], first[t], (size_t)H * sizeof(float)) == 0,
                       "cs devsum: group %d's host merge moved between the legs", t);
        }
        char line[4096]; report_line(line, sizeof line);
        const char *dm = strstr(line, "device-merged (");
        if (leg) {
            unsigned long long gn = 0, gr = 0;
            if (dm) {   /* the count sits before the keyword: \"... refusals), 4 device-merged (12 rows)\" */
                const char *p = dm;
                while (p > line && p[-1] == ' ') p--;   /* the count's space, then its digits */
                while (p > line && p[-1] >= '0' && p[-1] <= '9') p--;
                gn = strtoull(p, NULL, 10);
                sscanf(dm, "device-merged (%llu rows)", &gr);
            }
            printf("  cs devsum (armed): %d groups merged on the device, books %llu (%llu rows)\n", summed, gn, gr);
            CHECK(gn == (unsigned long long)T && gr == (unsigned long long)T * K,
                  "cs devsum: the books say %llu merged (%llu rows), not %d (%d)", gn, gr, T, T * K);
        } else {
            printf("  cs devsum (unarmed): %d groups, no merged row, books \"%s\"\n", summed, dm ? dm : "no clause");
            CHECK(!dm, "cs devsum: the unarmed control engaged the device sum");
        }
        vkt_shutdown();
        for (int e = 0; e < E; e++) free(slabs[e]);
        model_free();
    }
    /* the not-whole group: expert 5 has no slab and no resident place, so its selection
     * stays the CPU's and the tail must not fly (the host merge has a raw CPU row among
     * the rounded device ones -- a device sum would reassociate it) */
    rng = 7;
    model_make(f, f); g_act = VKT_ACT_SWIGLU_V4; g_limit = 10.0f;
    float xq[H];
    for (int i = 0; i < H; i++) xq[i] = frnd() * 4.0f;
    qdq128(xq, H);
    uint32_t hist[L][E], *hrows[L];
    for (int l = 0; l < L; l++) { hrows[l] = hist[l]; for (int e = 0; e < E; e++) hist[l][e] = 0; }
    hist[0][0] = 100; hist[0][1] = 90;
    setenv("COLI_VK_TIER_CS_DEVSUM", "1", 1);
    set_budget(2, f, f);
    VktConfig vc = cfg_of(f, f, VKT_ACT_SWIGLU_V4, 10.0f);
    CHECK(vkt_init(&vc, hrows), "cs devsum: the mixed group's tier did not start");
    int pl[L * E], pe[L * E], n = vkt_plan(pl, pe, L * E);
    for (int i = 0; i < n; i++) { VktExpertSrc s = src_of(&ex[pl[i]][pe[i]]); vkt_put(pl[i], pe[i], &s); }
    vkt_put_done();
    {
        static const int mix[K] = {0, 1, 5};
        VktAliasSrc src[K];
        for (int k = 0; k < K; k++) src[k] = (VktAliasSrc){0};   /* no slabs here: 0 and 1 resident, 5 the CPU's */
        uint8_t taken[K];
        vkt_begin_forward();
        (void)vkt_issue_w(0, xq, 1, K, mix, w, taken);
        int took = vkt_issue_group(0, xq, 1, K, mix, w, src, taken);
        CHECK(took == 2 && !taken[2], "cs devsum: the mixed group took %d (expert 5 should be the CPU's)", took);
        const float *dev[K], *sum = (const float *)1;
        CHECK(vkt_join_merged(dev, &sum), "cs devsum: the mixed group's join failed");
        CHECK(!sum, "cs devsum: a group with a CPU row carried a device sum");
        float y[HF_MAX];
        double worst = 0;
        for (int k = 0; k < 2; k++) {
            expert_v4(&ex[0][mix[k]], xq, w[k], 10.0f, y);
            float dr[HF_MAX];
            for (int i = 0; i < H; i++) dr[i] = bf16r(dev[k][i]);
            double r = rel(dr, y, H);
            if (!(r <= worst)) worst = r;
        }
        CHECK(no_nan(worst) < 2e-2, "cs devsum: the mixed group's rows off the reference (%.3g)", no_nan(worst));
        printf("  cs devsum (mixed): %d on the device, expert 5 the CPU's, no merged row, worst %.2e\n", took, no_nan(worst));
    }
    vkt_shutdown(); model_free();
    unsetenv("COLI_VK_TIER_CS_DEVSUM");
    unsetenv("COLI_VK_TIER_CS_ALIAS_CACHE");
}
static void cs_group(void) {
    cs_group_v4(1);   /* the v4 MXFP4 class: aliased + resident selections in one group, < 2e-2 */
    cs_group_v4(0);
    cs_group_i8(1);   /* the non-fp4 rows: the resident selections through the same group, < 2e-3 */
    cs_group_i8(0);
    cs_grow();        /* EXPERTS + DECODE_GROW: the aliased selections still install, > 2e-2 */
    cs_devsum();      /* CS_DEVSUM: the whole group's rows merge on the device, bit-for-bit the host merge */
    unsetenv("COLI_VK_TIER_SYNC");   /* the suite's next cases (the stream ring) expect it unset */
}

/* One aliased expert's store slab carrying TWO copies of the same records at different
 * offsets (the same slab pointer, two geometries): the slot-rewrite leg of the alias
 * cache reuses the pointer with a new geometry and must invalidate the old entry. */
static int cs_alias_slab2(int layer, int eid, VktAliasSrc *a, VktAliasSrc *b, void **slab_out) {
    if (!coli_vk_import_alignment()) return 0;
    const Ex *e = &ex[layer][eid];
    const Mat *m[3] = {&e->g, &e->u, &e->d};
    size_t woff[2][3], soff[2][3], cap = 0;
    for (int copy = 0; copy < 2; copy++) {
        for (int k = 0; k < 3; k++) {
            size_t rb = row_bytes(m[k]->f.kind, m[k]->I), wb = rb * (size_t)m[k]->O;
            size_t nb = (size_t)m[k]->O * (((size_t)m[k]->I + 31) / 32);   /* fmt 15: one ue8m0 byte a group */
            size_t sb = (nb + 3) / 4 * 4;
            woff[copy][k] = (cap + 4095) / 4096 * 4096; cap = woff[copy][k] + wb;
            soff[copy][k] = (cap + 4095) / 4096 * 4096; cap = soff[copy][k] + sb;
        }
    }
    cap = (cap + 4095) / 4096 * 4096;
    uint8_t *slab = NULL;
    if (posix_memalign(slab_out, 4096, cap) != 0 || !(slab = *slab_out)) return 0;
    memset(slab, 0, cap);
    for (int copy = 0; copy < 2; copy++) {
        for (int k = 0; k < 3; k++) {
            size_t rb = row_bytes(m[k]->f.kind, m[k]->I);
            size_t nb = (size_t)m[k]->O * (((size_t)m[k]->I + 31) / 32);
            memcpy(slab + woff[copy][k], m[k]->codes, rb * (size_t)m[k]->O);
            memcpy(slab + soff[copy][k], m[k]->scales, nb);
        }
    }
    memset(a, 0, sizeof(*a));
    memset(b, 0, sizeof(*b));
    a->slab = b->slab = slab; a->bytes = b->bytes = cap; a->fmt = b->fmt = 15; a->gs = b->gs = 32;
    for (int k = 0; k < 3; k++) {
        a->woff[k] = woff[0][k]; a->soff[k] = soff[0][k];
        b->woff[k] = woff[1][k]; b->soff[k] = soff[1][k];
        a->I[k] = b->I[k] = m[k]->I; a->O[k] = b->O[k] = m[k]->O;
    }
    return 1;
}
/* the report's cs alias cache clause, parsed for the keepalive books; skips, when
 * given, reads the fast path's trailing counter (-1: the arm was off, no text) */
static void cs_cache_report(const char *line, int *n, size_t *bytes,
                            unsigned long long *hits, unsigned long long *misses,
                            unsigned long long *evicts, long long *skips) {
    *n = -1; *bytes = 0; *hits = *misses = *evicts = ~0ull;
    if (skips) *skips = -1;
    const char *p = strstr(line, "cs alias cache: ");
    if (!p || sscanf(p, "cs alias cache: %d entries, %zu bytes (%*[^)]), %llu hits, %llu misses, %llu evictions",
                     n, bytes, hits, misses, evicts) != 5)
        { *n = -1; *bytes = 0; *hits = *misses = *evicts = ~0ull; }
    else if (skips) {
        int cn2; size_t cb2; unsigned long long ch2, cm2, ce2, cf;
        if (sscanf(p, "cs alias cache: %d entries, %zu bytes (%*[^)]), %llu hits, %llu misses, %llu evictions, %llu forgotten, %lld scans skipped",
                   &cn2, &cb2, &ch2, &cm2, &ce2, &cf, skips) != 7)
            *skips = -1;   /* the fast path's arm was off: no trailing counter */
    }
}
/* COLI_VK_TIER_CS_ALIAS_CACHE: the tier-owned keepalive for the compute site's aliased
 * slabs -- a decode expert's store slab is imported once at its exact geometry and reused
 * across joins (a hit), a slot rewritten in place to another geometry invalidates the old
 * entry and re-imports, and the byte bound retires the coldest entries at the join. */
static void cs_alias_cache(void) {
    lever_pin_save();
    VktFmt f = {VKT_SRC_MXFP4_E8M0, 32};
    static const int route[K] = {0, 1, 2};
    float w[K] = {0.5f, 0.3f, 0.2f};
    /* leg A: the hit and the slot rewrite (same slab pointer, a second copy's geometry) */
    rng = 7;
    model_make(f, f); g_act = VKT_ACT_SWIGLU_V4; g_limit = 10.0f;
    uint32_t hist[L][E], *rows[L];
    for (int l = 0; l < L; l++) { rows[l] = hist[l]; for (int e = 0; e < E; e++) hist[l][e] = 0; }
    hist[0][0] = 100; hist[0][1] = 90;
    setenv("COLI_VK_TIER_EXPERTS", "1", 1);
    setenv("COLI_VK_TIER_CS_ALIAS_CACHE", "1", 1);
    unsetenv("COLI_VK_TIER_CS_ALIAS_CACHE_MB");
    setenv("COLI_VK_TIER_SYNC", "1", 1);
    setenv("COLI_VK_TIER_RATE", "64", 1);
    set_budget(2, f, f);
    VktConfig vc = cfg_of(f, f, VKT_ACT_SWIGLU_V4, 10.0f);
    CHECK(vkt_init(&vc, rows), "cs alias cache: the tier did not start");
    int pl[L * E], pe[L * E], n = vkt_plan(pl, pe, L * E);
    for (int i = 0; i < n; i++) { VktExpertSrc s = src_of(&ex[pl[i]][pe[i]]); vkt_put(pl[i], pe[i], &s); }
    vkt_put_done();
    CHECK(n == 2, "cs alias cache: %d warm of 2 planned", n);
    VktAliasSrc a, b; void *slab = NULL;
    CHECK(cs_alias_slab2(0, 2, &a, &b, &slab), "cs alias cache: the dual-copy slab did not build");
    VktAliasSrc src[K] = {{0}, {0}, a};
    Books bk = {0, 0}; double worst = 0; unsigned long long aliased = 0;
    char line[4096]; int cn; size_t cb; unsigned long long ch, cm, cev;

    vkt_begin_forward();
    double r = cs_group_step(0, route, w, src, 1, 10.0f, 1, &bk, &aliased);
    if (r > worst) worst = r;
    report_line(line, sizeof line);
    cs_cache_report(line, &cn, &cb, &ch, &cm, &cev, NULL);
    CHECK(ch == 0 && cm == 1 && cn == 1 && cev == 0, "cs alias cache: first use %llu hits, %llu misses, %d entries, %llu evictions (want 0/1/1/0)", ch, cm, cn, cev);

    vkt_begin_forward();
    r = cs_group_step(0, route, w, src, 1, 10.0f, 1, &bk, &aliased);
    if (r > worst) worst = r;
    report_line(line, sizeof line);
    cs_cache_report(line, &cn, &cb, &ch, &cm, &cev, NULL);
    CHECK(ch == 1 && cm == 1 && cn == 1 && cev == 0, "cs alias cache: the same geometry reused %llu hits, %llu misses, %d entries, %llu evictions (want 1/1/1/0)", ch, cm, cn, cev);

    src[2] = b;   /* the slot rewritten in place: same slab, the second copy's geometry */
    vkt_begin_forward();
    r = cs_group_step(0, route, w, src, 1, 10.0f, 1, &bk, &aliased);
    if (r > worst) worst = r;
    report_line(line, sizeof line);
    cs_cache_report(line, &cn, &cb, &ch, &cm, &cev, NULL);
    CHECK(ch == 1 && cm == 2 && cn == 1 && cev == 1, "cs alias cache: the slot rewrite re-imported %llu hits, %llu misses, %d entries, %llu evictions (want 1/2/1/1)", ch, cm, cn, cev);

    vkt_begin_forward();
    r = cs_group_step(0, route, w, src, 1, 10.0f, 1, &bk, &aliased);
    if (r > worst) worst = r;
    report_line(line, sizeof line);
    cs_cache_report(line, &cn, &cb, &ch, &cm, &cev, NULL);
    CHECK(ch == 2 && cm == 2 && cn == 1 && cev == 1, "cs alias cache: the rewritten geometry reused %llu hits, %llu misses, %d entries, %llu evictions (want 2/2/1/1)", ch, cm, cn, cev);
    CHECK(no_nan(worst) < 2e-2, "cs alias cache: rows off the reference (%.3g)", no_nan(worst));
    CHECK(aliased > 0, "cs alias cache: no aliased selection flew");
    vkt_shutdown();
    free(slab);
    model_free();

    /* leg B: the byte bound retires the coldest entries at the join (1 MiB over 30 slabs) */
    rng = 7;
    model_make(f, f); g_act = VKT_ACT_SWIGLU_V4; g_limit = 10.0f;
    for (int l = 0; l < L; l++) { rows[l] = hist[l]; for (int e = 0; e < E; e++) hist[l][e] = 0; }
    hist[0][0] = 100; hist[0][1] = 90;
    setenv("COLI_VK_TIER_CS_ALIAS_CACHE_MB", "1", 1);
    set_budget(2, f, f);
    CHECK(vkt_init(&vc, rows), "cs alias cache bound: the tier did not start");
    n = vkt_plan(pl, pe, L * E);
    for (int i = 0; i < n; i++) { VktExpertSrc s = src_of(&ex[pl[i]][pe[i]]); vkt_put(pl[i], pe[i], &s); }
    vkt_put_done();
    CHECK(n == 2, "cs alias cache bound: %d warm of 2 planned", n);
    enum { NB = 30 };
    void *slabs[NB];
    for (int i = 0; i < NB; i++) slabs[i] = NULL;
    Books bkb = {0, 0}; double worstb = 0;
    for (int i = 0; i < NB; i++) {
        VktAliasSrc as;
        CHECK(cs_alias_slab(0, 2, &as, &slabs[i]), "cs alias cache bound: slab %d did not build", i);
        VktAliasSrc srcc[K] = {{0}, {0}, as};
        vkt_begin_forward();
        double rb = cs_group_step(0, route, w, srcc, 1, 10.0f, 1, &bkb, NULL);
        if (rb > worstb) worstb = rb;
    }
    report_line(line, sizeof line);
    cs_cache_report(line, &cn, &cb, &ch, &cm, &cev, NULL);
    CHECK(cev > 0, "cs alias cache bound: no byte-bound eviction (%d entries, %zu bytes, %llu evictions)", cn, cb, cev);
    CHECK(cb <= 1048576u, "cs alias cache bound: %zu bytes held over the 1 MiB cap", cb);
    CHECK(cn < NB, "cs alias cache bound: %d entries still held of %d", cn, NB);
    CHECK(no_nan(worstb) < 2e-2, "cs alias cache bound: rows off the reference (%.3g)", no_nan(worstb));
    vkt_shutdown();
    for (int i = 0; i < NB; i++) free(slabs[i]);
    model_free();
    unsetenv("COLI_VK_TIER_CS_ALIAS_CACHE");
    unsetenv("COLI_VK_TIER_CS_ALIAS_CACHE_MB");
    unsetenv("COLI_VK_TIER_SYNC");
    lever_pin_restore();
}
/* The VA-reuse guard (vkt_slab_forget, gap-wave1's mandatory hardening): the cache's
 * geometry key cannot tell two generations of one address apart. A slab is aliased
 * and cached, its pages are freed (the store's slot-free path: vkt_slab_forget first)
 * and the SAME address comes back for the same size with the SAME geometry and
 * different content -- the next alias there MUST MISS (the stale entry and the
 * import dedup under it are gone) and its rows must be the NEW pages' expert, not
 * the old import's. mmap MAP_FIXED makes the same-address realloc deterministic;
 * the fixture's experts of a layer share the record layout, so both generations use
 * one geometry. A POSIX-only leg: the Windows build skips it (no MAP_FIXED). */
static void cs_alias_va_reuse(void) {
#ifndef _WIN32
    lever_pin_save();
    VktFmt f = {VKT_SRC_MXFP4_E8M0, 32};
    rng = 7;
    model_make(f, f); g_act = VKT_ACT_SWIGLU_V4; g_limit = 10.0f;
    uint32_t hist[L][E], *rows[L];
    for (int l = 0; l < L; l++) { rows[l] = hist[l]; for (int e = 0; e < E; e++) hist[l][e] = 0; }
    hist[0][0] = 100; hist[0][1] = 90;   /* experts 0 and 1 resident: the alias path is the others' */
    setenv("COLI_VK_TIER_EXPERTS", "1", 1);
    setenv("COLI_VK_TIER_CS_ALIAS_CACHE", "1", 1);
    unsetenv("COLI_VK_TIER_CS_ALIAS_CACHE_MB");
    setenv("COLI_VK_TIER_SYNC", "1", 1);
    setenv("COLI_VK_TIER_RATE", "64", 1);
    set_budget(2, f, f);
    VktConfig vc = cfg_of(f, f, VKT_ACT_SWIGLU_V4, 10.0f);
    CHECK(vkt_init(&vc, rows), "cs va reuse: the tier did not start");
    int pl[L * E], pe[L * E], n = vkt_plan(pl, pe, L * E);
    for (int i = 0; i < n; i++) { VktExpertSrc s = src_of(&ex[pl[i]][pe[i]]); vkt_put(pl[i], pe[i], &s); }
    vkt_put_done();
    CHECK(n == 2, "cs va reuse: %d warm of 2 planned", n);
    size_t woff[3], soff[3], cap = cs_alias_slab_geometry(0, 2, woff, soff);
    void *pages = mmap(NULL, cap, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    CHECK(pages != MAP_FAILED, "cs va reuse: the first generation's pages did not map");
    if (pages == MAP_FAILED) { vkt_shutdown(); model_free(); goto out; }
    cs_alias_slab_fill(0, 2, pages, cap, woff, soff);   /* generation 1: expert 2's records */
    VktAliasSrc a;
    cs_alias_src_of(0, 2, pages, cap, woff, soff, &a);
    float w[K] = {0.5f, 0.3f, 0.2f};
    static const int route2[K] = {0, 1, 2}, route4[K] = {0, 1, 4};
    VktAliasSrc src[K] = {{0}, {0}, a};
    Books bk = {0, 0}; double worst = 0;
    char line[4096]; int cn; size_t cb; unsigned long long ch, cm, cev;

    vkt_begin_forward();
    double r = cs_group_step(0, route2, w, src, 1, 10.0f, 1, &bk, NULL);
    if (r > worst) worst = r;
    report_line(line, sizeof line);
    cs_cache_report(line, &cn, &cb, &ch, &cm, &cev, NULL);
    CHECK(ch == 0 && cm == 1 && cn == 1, "cs va reuse: the first generation cached %llu hits, %llu misses, %d entries (want 0/1/1)", ch, cm, cn);

    vkt_slab_forget(pages);   /* the store's slot-free path, then the pages go */
    munmap(pages, cap);
    void *again = mmap(pages, cap, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
    CHECK(again == pages, "cs va reuse: MAP_FIXED did not return the same address");
    if (again != pages) { munmap(again, cap); vkt_shutdown(); model_free(); goto out; }
    cs_alias_slab_fill(0, 4, pages, cap, woff, soff);   /* generation 2: expert 4's records, the SAME geometry */
    cs_alias_src_of(0, 4, pages, cap, woff, soff, &a);
    src[2] = a;
    vkt_begin_forward();
    r = cs_group_step(0, route4, w, src, 1, 10.0f, 1, &bk, NULL);
    if (r > worst) worst = r;
    report_line(line, sizeof line);
    cs_cache_report(line, &cn, &cb, &ch, &cm, &cev, NULL);
    CHECK(ch == 0 && cm == 2 && cn == 1, "cs va reuse: the re-taken address MUST MISS, not hit the stale entry (%llu hits, %llu misses, %d entries; want 0/2/1)", ch, cm, cn);

    vkt_begin_forward();   /* the fresh entry serves its own generation: a hit now */
    r = cs_group_step(0, route4, w, src, 1, 10.0f, 1, &bk, NULL);
    if (r > worst) worst = r;
    report_line(line, sizeof line);
    cs_cache_report(line, &cn, &cb, &ch, &cm, &cev, NULL);
    CHECK(ch == 1 && cm == 2, "cs va reuse: the second generation reused %llu hits, %llu misses (want 1/2)", ch, cm);
    CHECK(no_nan(worst) < 2e-2, "cs va reuse: rows off the reference, the fresh pages must serve (%.3g)", no_nan(worst));
    vkt_shutdown();
    munmap(pages, cap);
    model_free();
out:
    unsetenv("COLI_VK_TIER_CS_ALIAS_CACHE");
    unsetenv("COLI_VK_TIER_CS_ALIAS_CACHE_MB");
    unsetenv("COLI_VK_TIER_SYNC");
    lever_pin_restore();
#else
    printf("  (skipped on Windows: no MAP_FIXED)\n");
#endif
}
/* The hit fast path (COLI_VK_TIER_CS_ALIAS_SCAN_SKIP): a cache hit serving the SAME
 * expert the entry's f16 verdicts were scanned for passes them to the tensor build
 * and skips the scale scan (a full pass over the slab's scale bytes, three tensors an
 * alias); any other hit -- the slot rewritten to ANOTHER expert at the same geometry,
 * the key's blind spot -- must rescan, because the verdicts belong to the content that
 * earned them and a wrong one is silent wrong math (a wide scale on a coopmat f16
 * GEMM infs its output column). One slab, two experts, one geometry: the miss records
 * the first expert's verdicts, its repeat skips, the rewrite to the second expert
 * rescans (and re-records), and its repeat skips again -- the rows right throughout. */
static void cs_alias_scan_skip(int dprof) {
    lever_pin_save();
    VktFmt f = {VKT_SRC_MXFP4_E8M0, 32};
    rng = 7;
    model_make(f, f); g_act = VKT_ACT_SWIGLU_V4; g_limit = 10.0f;
    uint32_t hist[L][E], *rows[L];
    for (int l = 0; l < L; l++) { rows[l] = hist[l]; for (int e = 0; e < E; e++) hist[l][e] = 0; }
    hist[0][0] = 100; hist[0][1] = 90;
    setenv("COLI_VK_TIER_EXPERTS", "1", 1);
    setenv("COLI_VK_TIER_CS_ALIAS_CACHE", "1", 1);
    setenv("COLI_VK_TIER_CS_ALIAS_SCAN_SKIP", "1", 1);
    unsetenv("COLI_VK_TIER_CS_ALIAS_CACHE_MB");
    setenv("COLI_VK_TIER_SYNC", "1", 1);
    setenv("COLI_VK_TIER_RATE", "64", 1);
    set_budget(2, f, f);
    VktConfig vc = cfg_of(f, f, VKT_ACT_SWIGLU_V4, 10.0f);
    CHECK(vkt_init(&vc, rows), "cs scan skip: the tier did not start");
    int pl[L * E], pe[L * E], n = vkt_plan(pl, pe, L * E);
    for (int i = 0; i < n; i++) { VktExpertSrc s = src_of(&ex[pl[i]][pe[i]]); vkt_put(pl[i], pe[i], &s); }
    vkt_put_done();
    CHECK(n == 2, "cs scan skip: %d warm of 2 planned", n);
    void *slab = NULL;
    CHECK(cs_alias_slab(0, 3, &(VktAliasSrc){0}, &slab), "cs scan skip: the slab did not build");
    if (!slab) { vkt_shutdown(); model_free(); goto out2; }
    size_t woff[3], soff[3], cap = cs_alias_slab_geometry(0, 3, woff, soff);
    VktAliasSrc a;
    cs_alias_src_of(0, 3, slab, cap, woff, soff, &a);
    float w[K] = {0.5f, 0.3f, 0.2f};
    static const int route3[K] = {0, 1, 3}, route4[K] = {0, 1, 4};
    VktAliasSrc src[K] = {{0}, {0}, a};
    Books bk = {0, 0}; double worst = 0;
    char line[4096]; int cn; size_t cb; unsigned long long ch, cm, cev; long long cs;

    vkt_begin_forward();   /* the miss: scanned, verdicts recorded for expert 3 */
    double r = cs_group_step(0, route3, w, src, 1, 10.0f, 1, &bk, NULL);
    if (r > worst) worst = r;
    report_line(line, sizeof line);
    cs_cache_report(line, &cn, &cb, &ch, &cm, &cev, &cs);
    CHECK(ch == 0 && cm == 1 && cs == 0, "cs scan skip: the miss scanned (%llu hits, %llu misses, %lld skips; want 0/1/0)", ch, cm, cs);

    vkt_begin_forward();   /* the same expert again: a hit of the recorded content, the scan skipped */
    r = cs_group_step(0, route3, w, src, 1, 10.0f, 1, &bk, NULL);
    if (r > worst) worst = r;
    report_line(line, sizeof line);
    cs_cache_report(line, &cn, &cb, &ch, &cm, &cev, &cs);
    CHECK(ch == 1 && cm == 1 && cs == 1, "cs scan skip: the same expert's hit skipped the scan (%llu hits, %llu misses, %lld skips; want 1/1/1)", ch, cm, cs);

    /* the slot rewritten in place to ANOTHER expert at the SAME geometry: the key
     * hits, the content did not -- the rescan is the correctness gate, and its rows
     * the second expert's (the falsifier: a stale verdict or a stale import reads the
     * first expert's weights here) */
    cs_alias_slab_fill(0, 4, slab, cap, woff, soff);
    cs_alias_src_of(0, 4, slab, cap, woff, soff, &a);
    src[2] = a;
    vkt_begin_forward();
    r = cs_group_step(0, route4, w, src, 1, 10.0f, 1, &bk, NULL);
    if (r > worst) worst = r;
    report_line(line, sizeof line);
    cs_cache_report(line, &cn, &cb, &ch, &cm, &cev, &cs);
    CHECK(ch == 2 && cm == 1 && cs == 1, "cs scan skip: the rewritten slot's hit RESCANNED (%llu hits, %llu misses, %lld skips; want 2/1/1)", ch, cm, cs);

    vkt_begin_forward();   /* the second expert's repeat: its own verdicts now, skipped */
    r = cs_group_step(0, route4, w, src, 1, 10.0f, 1, &bk, NULL);
    if (r > worst) worst = r;
    report_line(line, sizeof line);
    cs_cache_report(line, &cn, &cb, &ch, &cm, &cev, &cs);
    CHECK(ch == 3 && cm == 1 && cs == 2, "cs scan skip: the second expert's hit skipped (%llu hits, %llu misses, %lld skips; want 3/1/2)", ch, cm, cs);
    if (dprof) {   /* the DPROF split (c): the alias bucket by its outcome -- hit + miss
                    * close on the total, both books fed by the same deltas */
        double tot = 0, hv = 0, mv = 0;
        const char *p = strstr(line, "; alias ");   /* the dprof clause's own 'alias', never the cache clause's */
        int ok = p && sscanf(p, "; alias %lf (hit %lf / miss %lf)", &tot, &hv, &mv) == 3;
        CHECK(ok, "cs scan skip: the dprof alias split did not parse");
        if (ok) {
            CHECK(hv >= 0 && mv >= 0, "cs scan skip: a negative alias split (%.3f / %.3f)", hv, mv);
            CHECK(tot <= 0 || fabs(hv + mv - tot) <= 0.01 + 1e-3 * tot,
                  "cs scan skip: the split misses the total (%.3f + %.3f vs %.3f)", hv, mv, tot);
        }
    }
    CHECK(no_nan(worst) < 2e-2, "cs scan skip: rows off the reference (%.3g)", no_nan(worst));
    vkt_shutdown();
    free(slab);
    model_free();
out2:
    unsetenv("COLI_VK_TIER_CS_ALIAS_CACHE");
    unsetenv("COLI_VK_TIER_CS_ALIAS_CACHE_MB");
    unsetenv("COLI_VK_TIER_CS_ALIAS_SCAN_SKIP");
    unsetenv("COLI_VK_TIER_SYNC");
    lever_pin_restore();
}
/* The hot set overlay (COLI_VK_TIER_HOT): the history's top share of the resident places,
 * ranked on the RAW counts, marked on the slots and skipped by every victim path. The
 * history names layer 0 ten times hotter than layer 1, so the global ranking and the
 * scaled 1..31 heat disagree -- the heat would make every layer's hottest expert equal
 * and the pin set a per-layer film, the counts keep it layer 0's top. */
static int hot_res(int layer, int lo, int n) {   /* residents of layer in [lo, lo+n) */
    int r = 0;
    for (int e = lo; e < lo + n && e < E; e++) r += vkt_resident(layer, e);
    return r;
}
static void hotset(void) {
    VktFmt f = {VKT_SRC_I4U_PAIRS_GS, 64};
    model_make(f, f); g_act = VKT_ACT_SWIGLU; g_limit = 0;
    setenv("COLI_VK_TIER_SYNC", "1", 1);   /* the evictions below are the point: not the uploader's timing */
    setenv("COLI_VK_TIER_RATE", "64", 1);
    uint32_t hist[L][E], *rows[L];
    for (int l = 0; l < L; l++) { rows[l] = hist[l]; for (int e = 0; e < E; e++) hist[l][e] = (uint32_t)(l ? 100 - 9 * e : 1000 - 90 * e); }
    int idx[4 * K]; float w[4 * K];
    unsigned long long ev, failed;
    /* (a) HOT=50 of a budget of 8: four pins, the history's top. A storm of the coldest
     * experts (layer 1's, none resident) churns the four unpinned places; the pins stay.
     * With the knob unset the same storm takes the head too: the overlay is what holds it. */
    for (int arm = 0; arm < 2; arm++) {
        if (arm) setenv("COLI_VK_TIER_HOT", "50", 1);
        set_budget(8, f, f);
        VktConfig vc = cfg_of(f, f, VKT_ACT_SWIGLU, 0);
        CHECK(vkt_init(&vc, rows), "hot set: the tier did not start");
        int pl[L * E], pe[L * E], n = vkt_plan(pl, pe, L * E);
        CHECK(n == 8, "hot set: planned %d experts for a budget of 8", n);
        for (int i = 0; i < n; i++) { VktExpertSrc s = src_of(&ex[pl[i]][pe[i]]); vkt_put(pl[i], pe[i], &s); }
        vkt_put_done();
        CHECK(hot_res(0, 0, 4) == 4, "hot set: the warm start did not take the history's top");
        Books bk = {0, 0}; double worst = 0;
        for (int t = 0; t < 300; t++)
            for (int l = 0; l < L; l++) { route(4, 4, 6, idx, w); double r = step(l, 4, idx, w, &bk); if (r > worst) worst = r; }
        report_counts(&ev, &failed);
        int res = 0; for (int l = 0; l < L; l++) res += hot_res(l, 0, E);
        printf("  %s: after the storm the history's top holds %d of 4 places, resident %d of 8, "
               "evictions %llu, failed %llu, worst %.2e\n", arm ? "HOT=50" : "HOT unset",
               hot_res(0, 0, 4), res, ev, failed, worst);
        CHECK(worst < 2e-3, "hot set: relative error %.3g", worst);
        CHECK(failed == 0, "hot set: %llu failed uploads", failed);
        CHECK(ev > 0, "hot set: the storm evicted nothing (the tail never churned)");
        CHECK(res <= 8, "hot set: %d resident over a budget of 8", res);
        if (arm) {
            CHECK(hot_res(0, 0, 4) == 4, "hot set: a pin gave way to the storm");
            char line[4096]; report_line(line, sizeof line);
            CHECK(after(line, "pins ") == 4, "hot set: the report says %llu pins, not 4", after(line, "pins "));
            CHECK(strstr(line, "decode:"), "hot set: the report's decode share is missing");
        } else
            CHECK(hot_res(0, 0, 4) < 4, "hot set: the storm left the whole history top resident with no pins");
        vkt_shutdown();
        unsetenv("COLI_VK_TIER_HOT");
    }
    /* (a2) HOT_UPDATE=1: the same pins re-ranked every forward, the rows still right */
    setenv("COLI_VK_TIER_HOT", "50", 1);
    setenv("COLI_VK_TIER_HOT_UPDATE", "1", 1);
    set_budget(8, f, f);
    { VktConfig vc = cfg_of(f, f, VKT_ACT_SWIGLU, 0);
      CHECK(vkt_init(&vc, rows), "hot set update: the tier did not start");
      int pl[L * E], pe[L * E], n = vkt_plan(pl, pe, L * E);
      for (int i = 0; i < n; i++) { VktExpertSrc s = src_of(&ex[pl[i]][pe[i]]); vkt_put(pl[i], pe[i], &s); }
      vkt_put_done();
      CHECK(hot_res(0, 0, 4) == 4, "hot set update: the warm start did not take the history's top");
      Books bk = {0, 0}; double worst = 0;
      for (int t = 0; t < 60; t++)
          for (int l = 0; l < L; l++) { route(4, 4, 6, idx, w); double r = step(l, 4, idx, w, &bk); if (r > worst) worst = r; }
      CHECK(hot_res(0, 0, 4) == 4 && worst < 2e-3, "hot set update: the re-pinned head did not hold (%d, %.3g)",
            hot_res(0, 0, 4), worst);
      vkt_shutdown(); }
    unsetenv("COLI_VK_TIER_HOT_UPDATE");
    unsetenv("COLI_VK_TIER_HOT");
    /* (b) HOT=100 of a budget of 4: every resident pinned, the tier full -- promotions
     * are refused (no victim), nothing is evicted, the residents still serve beside the
     * CPU's rows */
    setenv("COLI_VK_TIER_HOT", "100", 1);
    set_budget(4, f, f);
    { VktConfig vc = cfg_of(f, f, VKT_ACT_SWIGLU, 0);
      CHECK(vkt_init(&vc, rows), "hot set full: the tier did not start");
      int pl[L * E], pe[L * E], n = vkt_plan(pl, pe, L * E);
      for (int i = 0; i < n; i++) { VktExpertSrc s = src_of(&ex[pl[i]][pe[i]]); vkt_put(pl[i], pe[i], &s); }
      vkt_put_done();
      Books bk = {0, 0}; double worst = 0;
      for (int t = 0; t < 100; t++)
          for (int l = 0; l < L; l++) { route(4, 0, E, idx, w); double r = step(l, 4, idx, w, &bk); if (r > worst) worst = r; }
      report_counts(&ev, &failed);
      int res = hot_res(0, 0, 4) + hot_res(0, 4, 6) + hot_res(1, 0, E);
      printf("  HOT=100 of 4: resident %d (all pinned), evictions %llu, device %llu of %llu, worst %.2e\n",
             res, ev, bk.dev, bk.routed, worst);
      CHECK(res == 4 && hot_res(0, 0, 4) == 4, "hot set full: %d resident, the history's top %d", res, hot_res(0, 0, 4));
      CHECK(ev == 0, "hot set full: an all-pinned tier evicted %llu residents", ev);
      CHECK(bk.dev > 0 && worst < 2e-3, "hot set full: the residents did not serve (%llu of %llu, %.3g)", bk.dev, bk.routed, worst);
      vkt_shutdown(); }
    unsetenv("COLI_VK_TIER_HOT");
    /* (c) COLI_VK_TIER_PIN_LAYER (the dmoe arm's lever, the explicit pin): layer 1's ten
     * experts planned first whatever the history says and pinned whole; a storm on layer
     * 0's coldest takes the two unpinned places, never the layer */
    setenv("COLI_VK_TIER_PIN_LAYER", "1", 1);
    set_budget(12, f, f);
    { VktConfig vc = cfg_of(f, f, VKT_ACT_SWIGLU, 0);
      CHECK(vkt_init(&vc, rows), "pin layer: the tier did not start");
      int pl[L * E], pe[L * E], n = vkt_plan(pl, pe, L * E);
      int ok = n == 12;
      for (int i = 0; i < E; i++) ok &= pl[i] == 1;   /* the layer ahead of the history */
      CHECK(ok, "pin layer: the plan did not take layer 1 whole first");
      for (int i = 0; i < n; i++) { VktExpertSrc s = src_of(&ex[pl[i]][pe[i]]); vkt_put(pl[i], pe[i], &s); }
      vkt_put_done();
      Books bk = {0, 0}; double worst = 0;
      for (int t = 0; t < 200; t++) { route(4, 7, 3, idx, w); double r = step(0, 4, idx, w, &bk); if (r > worst) worst = r; }
      report_counts(&ev, &failed);
      printf("  PIN_LAYER=1: layer 1 holds %d of 10 through the storm, layer 0's hot head %d of 2, "
             "evictions %llu, worst %.2e\n", hot_res(1, 0, E), hot_res(0, 0, 2), ev, worst);
      CHECK(hot_res(1, 0, E) == 10, "pin layer: the pinned layer lost %d of its 10 places", 10 - hot_res(1, 0, E));
      CHECK(hot_res(0, 0, 2) < 2, "pin layer: the unpinned places never moved (%d of 2)", hot_res(0, 0, 2));
      CHECK(ev > 0 && worst < 2e-3 && failed == 0, "pin layer: evictions %llu, failed %llu, worst %.3g", ev, failed, worst);
      char line[4096]; report_line(line, sizeof line);
      CHECK(after(line, "pins ") == E, "pin layer: the report says %llu pins, not %d", after(line, "pins "), E);
      vkt_shutdown(); }
    unsetenv("COLI_VK_TIER_PIN_LAYER");
    /* (d) the ranking's scope: global keeps layer 0's top whatever layer 1's storm does;
     * per layer (HOT_SCOPE=layer) each layer keeps its own two; HOT_HASH=1 hands the
     * global set to layer 1, the hash-routed layer 0 left out */
    static const char *const what[3] = {"global", "HOT_SCOPE=layer", "HOT_HASH=1"};
    for (int arm = 0; arm < 3; arm++) {
        if (arm == 1) setenv("COLI_VK_TIER_HOT_SCOPE", "layer", 1);
        if (arm == 2) setenv("COLI_VK_TIER_HOT_HASH", "1", 1);
        setenv("COLI_VK_TIER_HOT", "50", 1);
        set_budget(8, f, f);
        VktConfig vc = cfg_of(f, f, VKT_ACT_SWIGLU, 0);
        CHECK(vkt_init(&vc, rows), "hot set %s: the tier did not start", what[arm]);
        int pl[L * E], pe[L * E], n = vkt_plan(pl, pe, L * E);
        for (int i = 0; i < n; i++) { VktExpertSrc s = src_of(&ex[pl[i]][pe[i]]); vkt_put(pl[i], pe[i], &s); }
        vkt_put_done();
        Books bk = {0, 0}; double worst = 0;
        for (int t = 0; t < 300; t++) { route(4, 0, E, idx, w); double r = step(1, 4, idx, w, &bk); if (r > worst) worst = r; }
        int h0 = hot_res(0, 0, 4), r11 = hot_res(1, 0, 2);
        printf("  %s: after layer 1's storm layer 0's top holds %d, layer 1's first two %d, worst %.2e\n",
               what[arm], h0, r11, worst);
        CHECK(worst < 2e-3, "hot set %s: relative error %.3g", what[arm], worst);
        if (arm == 0) {
            CHECK(h0 == 4, "hot set global: the RAW-count pins are not layer 0's top four (%d of 4)", h0);
            CHECK(!hot_res(0, 4, 4), "hot set global: the unpinned residents of layer 0 held too");
        } else if (arm == 1)
            CHECK(hot_res(0, 0, 2) == 2 && !vkt_resident(0, 2) && !vkt_resident(0, 3) && r11 == 2,
                  "hot set layer: the per-layer pins are not two a layer (layer 0's top %d, layer 1's first two %d)",
                  hot_res(0, 0, 2), r11);
        else
            CHECK(!h0 && hot_res(1, 0, 4) == 4, "hot set hash: the hash layer kept %d resident and layer 1's top held %d of 4", h0,
                  hot_res(1, 0, 4));
        vkt_shutdown();
        unsetenv("COLI_VK_TIER_HOT");
        unsetenv("COLI_VK_TIER_HOT_SCOPE");
        unsetenv("COLI_VK_TIER_HOT_HASH");
    }
    unsetenv("COLI_VK_TIER_SYNC");
    unsetenv("COLI_VK_TIER_RATE");
    model_free();
}

/* the experts of each layer on the device's memory (not the host pool), as a mask */
static unsigned dev_mask(int l) {
    uint8_t host[E]; unsigned m = 0;
    if (vkt_dev_host(l, host) < 0) return ~0u;
    for (int e = 0; e < E; e++) if (vkt_resident(l, e) && !host[e]) m |= 1u << e;
    return m;
}
/* COLI_VK_TIER_POOL_LAYER (the dmoe arm's Stage C lever, the residency activator that is
 * not a device pin): the named layer's E experts planned into the HOST POOL ahead of the
 * history, resident there -- place 2, vkt_dev_host's marks on them, the device's mask
 * empty -- and counted by vkt_dev_table whole, so vk_dmoe.h's all-or-nothing gate opens
 * with no device place taken. Promotions leave them there (hot_pin's pin, promote's skip)
 * while the device keeps its own residents, and a pool one place short of E places none of
 * them: all or nothing, like the gate it serves. */
static void pool_layer(void) {
    VktFmt f = {VKT_SRC_I4U_PAIRS_GS, 64};
    model_make(f, f); g_act = VKT_ACT_SWIGLU; g_limit = 0;
    const double eb = (double)vkt_expert_bytes(H, F, f, f) / 1073741824.0;
    char gb[64];
    uint32_t hist[L][E], *rows[L];           /* layer 0 the hot one: the lever must override it */
    for (int l = 0; l < L; l++) { rows[l] = hist[l]; for (int e = 0; e < E; e++) hist[l][e] = l ? 1u : 1000u; }
    setenv("COLI_VK_TIER_HOST", "1", 1);
    setenv("COLI_VK_TIER_POOL_LAYER", "1", 1);
    setenv("COLI_VK_TIER_SYNC", "1", 1);
    setenv("COLI_VK_TIER_PROMOTE", "8", 1);  /* on: the placed layer must stay in the pool anyway */
    uint32_t *tab = malloc((size_t)E * COLI_VK_XB_ENTRY * sizeof(uint32_t));
    for (int c = 0; c < 2; c++) {   /* the pool holds the layer whole, then one place short of it */
        set_budget(4, f, f);                   /* the device holds four, layer 0's */
        snprintf(gb, sizeof gb, "%.12f", (c ? E - 0.5 : E + 0.5) * eb);
        setenv("COLI_VK_TIER_HOST_GB", gb, 1);
        VktConfig vc = cfg_of(f, f, VKT_ACT_SWIGLU, 0);
        CHECK(vkt_init(&vc, rows), "pool layer: the tier did not start");
        /* the placement decision's own line, once, whichever way it went */
        FILE *t = tmpfile(); fflush(stderr);
        int saved = t ? dup(2) : -1;
        if (saved >= 0) dup2(fileno(t), 2);
        int pl[L * E], pe[L * E], n = vkt_plan(pl, pe, L * E);
        char buf[4096], line[4096] = "";
        int said = 0;
        if (saved >= 0) {
            fflush(stderr); dup2(saved, 2); close(saved); rewind(t);
            while (fgets(buf, sizeof buf, t)) { fputs(buf, stderr); if (strstr(buf, "COLI_VK_TIER_POOL_LAYER=1")) { said++; snprintf(line, sizeof line, "%s", buf); } }
            fclose(t);
        }
        int per[L] = {0}, first = 1;
        for (int i = 0; i < n; i++) { per[pl[i]]++; if (i < E) first &= pl[i] == 1; }
        CHECK(said == 1 && strstr(line, c ? "nothing placed" : "placed in the host pool"),
              "pool layer: the placement decision said %d lines, not the one %s", said, c ? "declining" : "placing");
        for (int i = 0; i < n; i++) { VktExpertSrc s = src_of(&ex[pl[i]][pe[i]]); vkt_put(pl[i], pe[i], &s); }
        vkt_put_done();
        if (c) {   /* declined: nothing placed, the plan exactly the lever-free tier's, the gate shut */
            int held = tab ? vkt_dev_table(1, tab) : -1;
            CHECK(held >= 0 && held < E, "pool layer: a pool one short still holds layer 1 whole for the gate (%d of %d)", held, E);
            vkt_shutdown();
            unsetenv("COLI_VK_TIER_POOL_LAYER");
            CHECK(vkt_init(&vc, rows), "pool layer: the tier did not start again");
            int ql[L * E], qe[L * E], m = vkt_plan(ql, qe, L * E);
            int same = n == m;
            for (int i = 0; same && i < n; i++) same = pl[i] == ql[i] && pe[i] == qe[i];
            CHECK(same && !first, "pool layer: the declined lever moved the plan (%d experts, layer 1's %d of %d, first %d)",
                  m, per[1], E, first);
            printf("  pool layer, the pool one place short: the plan the lever-free tier's %d experts exactly, layer 1's %d of %d "
                   "in it by the fill, the table %d of %d -- nothing placed\n", m, per[1], E, held, E);
        } else {
            uint8_t host[E];
            int nh = vkt_dev_host(1, host), in_pool = 1, res = 0;
            for (int e = 0; e < E; e++) { in_pool &= host[e]; res += vkt_resident(1, e); }
            CHECK(n == E + 4 && per[1] == E && per[0] == 4 && first,
                  "pool layer: planned %d experts (layers %d/%d), not layer 1's %d in the pool ahead of layer 0's four",
                  n, per[0], per[1], E);
            CHECK(res == E && nh == E && in_pool && dev_mask(1) == 0u,
                  "pool layer: %d of %d resident, %d of them in the table's host places, the device's mask %#x",
                  res, E, nh, dev_mask(1));
            CHECK(tab && vkt_dev_table(1, tab) == E, "pool layer: the tier does not count all %d of layer 1 for the dmoe gate", E);
            /* the layer routed, its rows read from the pool, and the books' promotions on:
             * it stays in the pool and the device keeps its own four */
            Books bk = {0, 0}; double worst = 0; int idx[8 * K]; float w[8 * K];
            for (int t2 = 0; t2 < 40; t2++) { route(4, 0, E, idx, w); double r = step(1, 4, idx, w, &bk); if (r > worst) worst = r; vkt_settle(); }
            for (int t2 = 0; t2 < 8; t2++) { route(8, 0, E, idx, w); vkt_dev_count(1, 8, K, idx); vkt_settle(); }
            CHECK(dev_mask(1) == 0u && dev_mask(0) != 0u && vkt_dev_table(1, tab) == E,
                  "pool layer: the promotions moved the placed layer (its device mask %#x, layer 0's %#x, the table %d of %d)",
                  dev_mask(1), dev_mask(0), vkt_dev_table(1, tab), E);
            CHECK(worst < 2e-3, "pool layer: the pool's rows off the reference (%.3g)", worst);
            printf("  pool layer: layer 1's %d experts in the host pool (the device's mask %#x, layer 0's %#x), the table whole "
                   "through %llu routed rows and the books' promotions, worst %.2e\n", nh, dev_mask(1), dev_mask(0), bk.routed, worst);
        }
        vkt_shutdown();
    }
    free(tab);
    unsetenv("COLI_VK_TIER_HOST_GB"); unsetenv("COLI_VK_TIER_SYNC"); unsetenv("COLI_VK_TIER_PROMOTE");
    unsetenv("COLI_VK_TIER_POOL_LAYER");
    setenv("COLI_VK_TIER_HOST", "0", 1);
    model_free();
}

/* The model's geometry (PW-Dfix, the W1 P0's regression test): the hybrid host-pool path
 * of fmt 15 at DeepSeek V4 Flash's shapes -- hidden 4096, inter 2048, gs 32, the V4
 * roundings. The P0: host_pairs advanced each output row's scale pointer by the f32-scale
 * stride while fmt 15's scales are packed ue8m0 bytes (a quarter of it), so every output
 * row past the first read its bytes 4x too deep -- past the scale range into the host
 * pool's other bytes, up to 2^127-scale garbage and non-finite expert rows; at model scale
 * the router one layer below took them as input and failed (AI-pw-w1-2026-10-25). Every
 * tier-held row is checked finite AND against the reference -- rel()'s NaN comparison is
 * false, so a NaN row never raises a plain `worst` (that is how the unit gates stayed green
 * while the model died); the hybrid pairs are forced with COLI_VK_HYBRID_PCIE=0 beside a
 * device budget too small for the layer. */
static unsigned long long hy_pairs(const char *line);   /* below: the report's hybrid pairs */
static void model_hybrid_case(void) {
    enum { MH = 4096, MF = 2048, ME = 8, MK = 6, GS = 32 };   /* the model's geometry, a layer's slice */
    lever_pin_save();
    setenv("COLI_VK_TIER_EXPERTS", "0", 1);   /* gap-wave4: the hybrid classification is the case's subject; the armed compute site skips it at rows-1 */
    const float limit = 10.0f;
    static const float lut[8] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
    static const VktFmt mf = {VKT_SRC_MXFP4_E8M0, GS};
    struct { uint8_t *gc, *uc, *dc, *gs, *us, *ds; } m[ME];
    size_t rg = (size_t)(MH + 1) / 2, rd = (size_t)(MF + 1) / 2, sg = (size_t)MF * (MH / GS), sd = (size_t)MH * (MF / GS);
    int ok = 1;
    for (int e = 0; e < ME; e++) {
        m[e].gc = malloc(rg * MF); m[e].uc = malloc(rg * MF); m[e].dc = malloc(rd * MH);
        m[e].gs = malloc(sg); m[e].us = malloc(sg); m[e].ds = malloc(sd);
        for (size_t i = 0; i < rg * MF; i++) { m[e].gc[i] = (uint8_t)rnd(); m[e].uc[i] = (uint8_t)rnd(); }
        for (size_t i = 0; i < rd * MH; i++) m[e].dc[i] = (uint8_t)rnd();
        for (size_t i = 0; i < sg; i++) { m[e].gs[i] = (uint8_t)(118 + rnd() % 4); m[e].us[i] = (uint8_t)(118 + rnd() % 4); }
        for (size_t i = 0; i < sd; i++) m[e].ds[i] = (uint8_t)(118 + rnd() % 4);
        ok &= m[e].gc && m[e].uc && m[e].dc && m[e].gs && m[e].us && m[e].ds;
    }
    if (!ok) { CHECK(0, "model hybrid: no memory for the model's experts"); goto free_model; }
    {
    setenv("COLI_VK_TIER_FMT15", "1", 1);
    setenv("COLI_VK_TIER_HOST", "1", 1); setenv("COLI_VK_TIER_SYNC", "1", 1);
    setenv("COLI_VK_TIER_PROMOTE", "0", 1); setenv("GQ_EXACT", "1", 1);
    setenv("COLI_VK_HYBRID_PCIE", "0", 1);   /* every host pool pair the CPU's: the corrupt path */
    const double one = (double)vkt_expert_bytes(MH, MF, mf, mf);
    char b[64];
    snprintf(b, sizeof b, "%.12f", 2.5 * one / 1073741824.0); setenv("COLI_VK_TIER_GB", b, 1);   /* 2 of 8 on the device */
    snprintf(b, sizeof b, "%.12f", (ME + 0.5) * one / 1073741824.0); setenv("COLI_VK_TIER_HOST_GB", b, 1);
    VktConfig vc = {.engine = "test", .layers = 1, .experts = ME, .hidden = MH, .inter = MF, .topk = MK,
                    .gate_up = mf, .down = mf, .act = VKT_ACT_SWIGLU_V4, .act_limit = limit,
                    .act_a = 4.f, .act_b = 25.f, .max_rows = 1 << 20, .ram_reserve = (size_t)4 << 30};
    int on = vkt_init(&vc, NULL);
    CHECK(on, "model hybrid: the tier did not start");
    unsigned long long dev = 0, bad = 0, nf = 0, hy = 0;
    double worst = 0;
    float *x = malloc(sizeof(float) * MH), *y = malloc(sizeof(float) * MH), *h = malloc(sizeof(float) * MF);
    if (on && x && y && h) {
        for (int e = 0; e < ME; e++) { VktExpertSrc src = {m[e].gc, m[e].uc, m[e].dc, m[e].gs, m[e].us, m[e].ds}; vkt_put(0, e, &src); }
        vkt_put_done();
        for (int t = 0; t < 4; t++) {   /* decode-class steps over the layer */
            int idx[MK]; float w[MK]; uint8_t taken[MK]; const float *rows[MK];
            for (int k = 0; k < MK; k++) { idx[k] = (t * 3 + k) % ME; w[k] = 0.2f + (rnd() % 100) / 200.0f; }
            for (int i = 0; i < MH; i++) x[i] = frnd() * 4.0f;
            qdq128(x, MH);   /* the engine's rounding of the activation */
            int n = vkt_issue_w(0, x, 1, MK, idx, w, taken);
            for (int i = 0; i < MK; i++)
                if (!taken[i]) { VktExpertSrc src = {m[idx[i]].gc, m[idx[i]].uc, m[idx[i]].dc,
                                                     m[idx[i]].gs, m[idx[i]].us, m[idx[i]].ds}; vkt_note(0, idx[i], &src); }
            CHECK(n ? vkt_join(rows) : 1, "model hybrid: join failed");
            for (int k = 0; k < MK; k++) {
                if (!taken[k]) continue;
                /* the reference: the engine's arithmetic (v4_act's at these shapes) */
                for (int o = 0; o < MF; o++) {
                    double ga = 0, ua = 0;
                    const uint8_t *gr = m[idx[k]].gc + (size_t)o * rg, *ur = m[idx[k]].uc + (size_t)o * rg;
                    for (int g = 0; g < MH / GS; g++) {
                        double gp = 0, up = 0;
                        for (int j = 0; j < GS; j++) {
                            int i = g * GS + j;
                            int nib = (i & 1) ? gr[i >> 1] >> 4 : gr[i >> 1] & 15;
                            gp += x[i] * ((nib & 8) ? -lut[nib & 7] : lut[nib & 7]);
                            nib = (i & 1) ? ur[i >> 1] >> 4 : ur[i >> 1] & 15;
                            up += x[i] * ((nib & 8) ? -lut[nib & 7] : lut[nib & 7]);
                        }
                        uint32_t u; float s;
                        u = (uint32_t)m[idx[k]].gs[(size_t)o * (MH / GS) + g] << 23; memcpy(&s, &u, 4); ga += gp * s;
                        u = (uint32_t)m[idx[k]].us[(size_t)o * (MH / GS) + g] << 23; memcpy(&s, &u, 4); ua += up * s;
                    }
                    float g = bf16r((float)ga), u = bf16r((float)ua);
                    g = fminf(g, limit); u = fmaxf(-limit, fminf(u, limit));
                    float sg = g >= 0 ? 1.0f / (1.0f + expf(-g)) : expf(g) / (1.0f + expf(g));
                    h[o] = bf16r(g * sg * u * w[k]);
                }
                qdq128(h, MF);
                for (int o = 0; o < MH; o++) {
                    double a = 0;
                    const uint8_t *dr = m[idx[k]].dc + (size_t)o * rd;
                    for (int g = 0; g < MF / GS; g++) {
                        double p = 0;
                        for (int j = 0; j < GS; j++) {
                            int i = g * GS + j;
                            int nib = (i & 1) ? dr[i >> 1] >> 4 : dr[i >> 1] & 15;
                            p += h[i] * ((nib & 8) ? -lut[nib & 7] : lut[nib & 7]);
                        }
                        uint32_t u = (uint32_t)m[idx[k]].ds[(size_t)o * (MF / GS) + g] << 23; float s; memcpy(&s, &u, 4);
                        a += p * s;
                    }
                    y[o] = bf16r((float)a);
                }
                for (int o = 0; o < MH; o++) if (!isfinite(rows[k][o])) nf++;
                double r = rel(rows[k], y, MH);
                if (!isfinite(r) || r > 2e-2) bad++;   /* a NaN row counts: rel's NaN compare is false */
                if (isfinite(r) && r > worst) worst = r;
                dev++;
            }
        }
    }
    {
        char line[4096]; report_line(line, sizeof line);   /* this case's own hybrid pairs */
        unsigned long long pairs = hy_pairs(line);
        hy = pairs == ~0ull ? 0 : pairs;
    }
    vkt_shutdown();
    unsetenv("COLI_VK_HYBRID_PCIE"); unsetenv("GQ_EXACT"); unsetenv("COLI_VK_TIER_PROMOTE");
    unsetenv("COLI_VK_TIER_SYNC"); unsetenv("COLI_VK_TIER_HOST_GB"); unsetenv("COLI_VK_TIER_GB");
    unsetenv("COLI_VK_TIER_FMT15");
    setenv("COLI_VK_TIER_HOST", "0", 1);
    printf("  fmt 15 at the model's geometry (hidden %d, inter %d), %llu hybrid pairs: %llu rows, %llu off the "
           "reference, %llu non-finite, worst rel %.2e\n", MH, MF, hy, dev, bad, nf, worst);
    CHECK(hy > 0, "model hybrid: no host pool pair ran on the CPU (the P0's path)");
    CHECK(dev > 0, "model hybrid: the tier took nothing");
    CHECK(bad == 0 && nf == 0, "model hybrid: the host pool's rows corrupt at the model's geometry");
    free(x); free(y); free(h);
    }
free_model:
    for (int e = 0; e < ME; e++) { free(m[e].gc); free(m[e].uc); free(m[e].dc); free(m[e].gs); free(m[e].us); free(m[e].ds); }
    lever_pin_restore();   /* gap-wave4: the hybrid pin lifts */
}

/* A pool short of every expert, promotions on (COLI_VK_TIER_PROMOTE): a routed pooled
 * expert moves up to the device, the coldest resident gives way -- with no pool copy of
 * its own, its device copy is freed and the slot empties -- and every row stays the
 * reference's. Unset, a partial pool promotes nothing (the default). */
static void promote_partial_case(void) {
    VktFmt f = {VKT_SRC_I8_ROW, 0};
    model_make(f, f); g_act = VKT_ACT_SWIGLU; g_limit = 0;
    const double eb = (double)vkt_expert_bytes(H, F, f, f) / 1073741824.0;
    char gb[64];
    setenv("COLI_VK_TIER_HOST", "1", 1);
    setenv("COLI_VK_TIER_SYNC", "1", 1);
    setenv("COLI_VK_TIER_RATE", "64", 1);   /* the notes' admission: an earlier case may have left it at 0 */
    for (int d = 0; d < 2; d++) {   /* d 1: the default, no promotion without the knob */
        set_budget(4, f, f);
        snprintf(gb, sizeof gb, "%.12f", L * E / 2.0 * eb);
        setenv("COLI_VK_TIER_HOST_GB", gb, 1);
        if (d) unsetenv("COLI_VK_TIER_PROMOTE"); else setenv("COLI_VK_TIER_PROMOTE", "8", 1);
        VktConfig vc = cfg_of(f, f, VKT_ACT_SWIGLU, 0);
        int on = vkt_init(&vc, NULL);
        CHECK(on, "promote2: the tier did not start");
        if (!on) break;
        CHECK(vkt_host_pool_on(), "promote2: the pool is off");
        int res0 = 0;
        for (int l = 0; l < L; l++) for (int e = 0; e < E; e++) res0 += vkt_resident(l, e);
        CHECK(res0 < L * E, "promote2: the pool holds every expert (%d of %d)", res0, L * E);
        (void)0;
        /* fill the device with 0..3, then route 4..6: they land in the pool (the device
         * is full), their heat grows, and the promotion moves them up over the coldest
         * of 0..3 -- every row still the reference's */
        Books bk = {0, 0}; double worst = 0;
        for (int t = 0; t < 16; t++) {
            int idx[K]; float w[K];
            int base = t < 6 ? 0 : 4;
            for (int k = 0; k < K; k++) { idx[k] = base + k % 4; w[k] = 1.f / (k + 1); }
            vkt_begin_forward();
            double r = step(0, 1, idx, w, &bk);
            if (r > worst) worst = r;
        }
        vkt_settle();   /* the last promotion's copy lands at a quiescent point */
        char line[4096]; report_line(line, sizeof line);
        unsigned long long ups = ~0ull;   /* its count sits before the words */
        for (char *m = strstr(line, " moved up to the device ("); m && m > line; ) {
            char *d = m; while (d > line && d[-1] >= '0' && d[-1] <= '9') d--;
            ups = strtoull(d, NULL, 10);
            break;
        }
        if (!d) {
            CHECK(ups != ~0ull && ups >= 1, "promote2: %llu experts moved up", ups == ~0ull ? 0ull : ups);
            CHECK(worst < 2e-3, "promote2: rows off the reference (%.3g)", worst);
        } else {
            CHECK(ups == ~0ull, "promote2: the default moved %llu up", ups == ~0ull ? 0ull : ups);
        }
        int res1 = 0;
        for (int l = 0; l < L; l++) for (int e = 0; e < E; e++) res1 += vkt_resident(l, e);
        printf("  %s: resident %d of %d (device 4 + the pool), ups %llu, worst %.2e\n", d ? "default" : "PROMOTE=8",
               res1, L * E, ups == ~0ull ? 0 : ups, worst);
        vkt_shutdown();
    }
    unsetenv("COLI_VK_TIER_HOST_GB"); unsetenv("COLI_VK_TIER_SYNC"); unsetenv("COLI_VK_TIER_RATE");
    setenv("COLI_VK_TIER_HOST", "0", 1);
    model_free();
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
    lever_pin_save();
    setenv("COLI_VK_TIER_EXPERTS", "0", 1);   /* gap-wave4: the hybrid classification is the case's subject; the armed compute site skips it at rows-1 */
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
    lever_pin_restore();   /* gap-wave4: the hybrid pin lifts */
}

/* DeepSeek V4's expert in the hybrid: MXFP4 ue8m0 experts, four on the device and the rest
 * in the host pool, the CPU computing the pool's pairs in place with the device's roundings
 * (vk_tier.c hy_act_v4, the route weight in the row). The float form (GQ_EXACT=1) against
 * expert_v4, the engine's arithmetic: most rows bit-identical once rounded to bf16, every
 * one close; then the same steps with COLI_VK_HYBRID=0. */
static void v4_hybrid_case(void) {
    VktFmt f = {VKT_SRC_MXFP4_E8M0, 32};
    lever_pin_save();
    setenv("COLI_VK_TIER_EXPERTS", "0", 1);   /* gap-wave4: the hybrid classification is the case's subject; the armed compute site skips it at rows-1 */
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
                    double r = rel(dr, y, H); if (!(r <= worst)) worst = r;   /* a NaN counts as the worst */
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
    lever_pin_restore();   /* gap-wave4: the hybrid pin lifts */
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
                double r = stream_step(l, Ss[t], idx, w, 0, &bk, &streamed, &kept, 0, NULL);
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

/* The host pool's room, pure arithmetic (vkt_host_room, no device). */
static void host_room(void) {
    const size_t GiB = 1073741824, MiB = 1048576;
    CHECK(vkt_host_room(30 * GiB, 18 * GiB, 350 * MiB, 1, 0) == 350 * MiB,
          "host room: a plan that leaves 350 MiB gave %zu", vkt_host_room(30 * GiB, 18 * GiB, 350 * MiB, 1, 0));
    CHECK(vkt_host_room(30 * GiB, 1 * GiB, 2 * GiB, 1, 0) == 1 * GiB,
          "host room: a spare above the budget is not budget-bound");
    CHECK(vkt_host_room(30 * GiB, 18 * GiB, 0, 1, 0) == 0,
          "host room: a plan that leaves nothing gave no room");
    CHECK(vkt_host_room(30 * GiB, 50 * GiB, 40 * GiB, 1, 0) == 30 * GiB,
          "host room: room above what is available is clamped to it");
    CHECK(vkt_host_room(30 * GiB, 0, 0, 0, 1 * GiB) == 25 * GiB,
          "host room: without a plan, MemAvailable less the cache growth and 4 GiB");
    CHECK(vkt_host_room(4 * GiB, 0, 0, 0, 2 * GiB) == 0 && vkt_host_room(4 * GiB, 0, 0, 0, 0) == 0,
          "host room: the fallback went negative");
    /* The run the fix is for: an 18 GiB budget the planner had spent to 17.65 GiB,
     * 15.1 MiB an expert, MemAvailable 29.75 GiB. The old pool took MemAvailable
     * less 4 GiB = 25.75 GiB beside it and the kernel killed the process; the plan
     * leaves 0.35 GiB, so the pool holds at most 23 experts and stays inside the
     * budget. */
    size_t budget = 18 * GiB, projected = (size_t)(17.65 * (double)GiB), per = (size_t)(15.1 * (double)MiB),
           avail = (size_t)(29.75 * (double)GiB);
    size_t spare = budget - projected, room = vkt_host_room(avail, budget, spare, 1, 0);
    CHECK(projected + room <= budget, "host room: the plan and the pool exceed the budget (%zu + %zu of %zu)",
          projected, room, budget);
    CHECK(room / per <= 23, "host room: the pool would hold %zu experts, not 23", room / per);
    CHECK(vkt_host_room(avail, 0, 0, 0, 0) > 25 * GiB,
          "host room: without the plan the room is the 25.75 GiB that killed the run");
    printf("host room: the plan leaves %zu of %zu, %zu experts; without it %.2f GiB\n",
           room, budget, room / per, (double)vkt_host_room(avail, 0, 0, 0, 0) / GiB);
}

/* vkt_init with stderr captured, so a test can read the tier's [VK] lines; the capture
 * lands in host_cap (the boot lines the host pool's cases are about). */
static char host_cap[16384];
static int cap_vkt_init(const VktConfig *vc, uint32_t *const *heat) {
    fflush(stderr);
    host_cap[0] = 0;
    int save = dup(2), on;
    char path[] = "/tmp/coli_vkt_host.XXXXXX";
    int fd = mkstemp(path);
    if (save < 0 || fd < 0) { if (fd >= 0) close(fd); if (save >= 0) close(save); unlink(path); return vkt_init(vc, heat); }
    dup2(fd, 2);
    on = vkt_init(vc, heat);
    fflush(stderr);
    dup2(save, 2); close(save);
    lseek(fd, 0, SEEK_SET);
    ssize_t got = read(fd, host_cap, sizeof host_cap - 1);
    if (got > 0) host_cap[got] = 0;
    close(fd); unlink(path);
    return on;
}

/* The pool's bytes as the boot line states them ("host pool on, N experts in X of
 * system RAM"; *experts its count), -1 when the line is not there to read. */
static double cap_pool_bytes(const char *cap, int *experts) {
    const char *p = strstr(cap, "host pool on, ");
    int n; double v; char unit[8] = "";
    if (experts) *experts = -1;
    if (!p || sscanf(p, "host pool on, %d experts in %lf %3s", &n, &v, unit) != 3) return -1;
    if (experts) *experts = n;
    const double GiB = 1073741824.0, MiB = 1048576.0, KiB = 1024.0;
    return v * (!strcmp(unit, "GiB") ? GiB : !strcmp(unit, "MiB") ? MiB : !strcmp(unit, "KiB") ? KiB : 1.0);
}

/* The host pool from a RAM plan (COLI_VK_TIER_HOST=1, so also on a device that
 * shares the host's RAM). The fixture is tiny, so the plan is stated in its scale; the
 * run that OOMed is host_room's above. */
static void host_plan(void) {
    const size_t GiB = 1073741824, MiB = 1048576, KiB = 1024;
    VktFmt f = {VKT_SRC_F32, 0};
    size_t exp = vkt_expert_bytes(H, F, f, f);
    model_make(f, f); g_act = VKT_ACT_SWIGLU; g_limit = 0;
    set_budget(4, f, f);   /* the device holds 4 of the 20: the pool has experts to take */
    setenv("COLI_VK_TIER_HOST", "1", 1);
    VktConfig vc = cfg_of(f, f, VKT_ACT_SWIGLU, 0);
    vc.ram_planned = 1; vc.ram_budget = GiB; vc.ram_spare = 350 * MiB;
    int on = cap_vkt_init(&vc, NULL), n = -1;
    CHECK(on, "host plan: the tier did not start");
    if (on) {
        CHECK(vkt_host_pool_on(), "host plan: the pool is off with 350 MiB of plan room");
        CHECK(strstr(host_cap, "RAM plan"), "host plan: the boot line does not name the plan");
        double b = cap_pool_bytes(host_cap, &n);
        CHECK(b >= 0 && b <= 350.0 * MiB, "host plan: the pool takes %.0f, over the plan's room", b);
        printf("  plan of 350 MiB: pool %d experts, %.1f KiB\n", n, b / KiB);
        vkt_shutdown();
    }
    /* a plan that leaves nothing: the pool is off, the line says why */
    vc.ram_spare = 0;
    on = cap_vkt_init(&vc, NULL);
    CHECK(on, "host plan: the tier did not start (no spare)");
    if (on) {
        CHECK(!vkt_host_pool_on(), "host plan: the pool is on with no plan room");
        CHECK(strstr(host_cap, "host pool off: the RAM plan leaves"),
              "host plan: no line when the plan leaves nothing");
        vkt_shutdown();
    }
    /* the operator's size wins over the plan, with one warning */
    setenv("COLI_VK_TIER_HOST_GB", "0.001", 1);   /* ~1 MiB: an expert fits, the plan's room does not */
    vc.ram_spare = 100 * KiB;
    on = cap_vkt_init(&vc, NULL);
    CHECK(on, "host plan: the tier did not start (override)");
    if (on) {
        double b = cap_pool_bytes(host_cap, &n);
        CHECK(vkt_host_pool_on(), "host plan: the explicit size did not override the plan");
        CHECK(strstr(host_cap, "over the RAM plan"), "host plan: no warning for the size over the plan");
        CHECK(b >= 0 && b > 100.0 * KiB + (double)exp, "host plan: the override gave %.0f, under the plan's room", b);
        printf("  override 0.001 GiB over a 100 KiB plan: pool %.1f KiB, one warning\n", b / KiB);
        vkt_shutdown();
    }
    unsetenv("COLI_VK_TIER_HOST_GB");
    setenv("COLI_VK_TIER_HOST", "0", 1);   /* the rest of the suite's guard again */
    model_free();
}

/* fmt 15 (the tier's packed MXFP4: ue8m0 scale bytes) against fmt 7 (f32 scales): the
 * round trip is the identity -- the byte's exponent lands in the float's exponent field
 * (vk_tier.c's expansion, the shaders' decode) -- and the packed form holds 3/4 less

 * scale bytes, the fork's 12.75 MiB expert against the f32 form's 15.0.
 *
 * The two halves live on different legs: the identity and the scale-word counts are
 * properties of the formats themselves and run in every leg; the byte-shrink half
 * reads them through vkt_expert_bytes, whose dev_fmt() routes MXFP4_E8M0 to fmt 15
 * -- the default again since PW-Dfix fixed the W1 P0's host_pairs stride and undid
 * the opt-in hotfix (d73b82e1); tier_fmt15() re-reads the env on every call, no
 * cache. The armed "fmt15" leg still runs the shrink half as the pin: it unsets
 * the env before the measure, so the assertion reads the tree's default and holds
 * whichever way the default later lands. */
static void packed_scales(int bytes) {

    int ident = 1;
    for (int e = 0; e < 256; e++) {   /* byte -> f32 -> byte, both directions the code takes */
        uint32_t u = (uint32_t)e << 23; float f; memcpy(&f, &u, 4);
        uint32_t b; memcpy(&b, &f, 4);
        if ((b >> 23) != (uint32_t)e) ident = 0;
        uint32_t w = 0; memcpy(&w, (const uint8_t[4]){(uint8_t)e, 0, 0, 0}, 4);   /* the shader's word read, byte 0 */
        float s = ldexpf(1.0f, (int)((w & 0xffu)) - 127);
        if (s != f && (e != 0)) ident = 0;   /* e 0: the expansion is +0.0 (a zero exponent field), ldexp's 2^-127 differs */
        if (e == 0 && f != 0.0f) ident = 0;
    }
    CHECK(ident, "packed scales: the ue8m0 <-> f32 round trip is not the identity");
    /* the scale words the pool counts: the bytes padded to the pool's 4-byte unit */
    int hs[2] = {192, 4096}, fs[2] = {128, 2048};   /* the harness's model and DeepSeek V4 */
    for (int g = 0; g < 2; g++)
        for (int k = 0; k < 2; k++) {
            int I = k ? fs[g] : hs[g], O = k ? hs[g] : fs[g];
            size_t nb = (size_t)O * (((size_t)I + 31) / 32);
            size_t nw = coli_vk_tensor_scale_count(15, I, O, 32);
            CHECK(nw * 4 >= nb && nw * 4 - nb < 4, "packed scales: the scale words hold O*ng bytes");
            CHECK(coli_vk_tensor_scale_count(7, I, O, 32) * 4 == nb * 4,
                  "packed scales: fmt 7 holds 4x the bytes of the ue8m0 store");
        }
    if (!bytes) return;   /* the shrink half is the armed leg's: the comment above */
    /* DeepSeek V4's geometry: the packed form is the fork's 12.75 MiB expert against 15.0 */

    VktFmt pk = {VKT_SRC_MXFP4_E8M0, 32}, f32s = {VKT_SRC_MXFP4_F32, 32};
    size_t bp = vkt_expert_bytes(4096, 2048, pk, pk), bf = vkt_expert_bytes(4096, 2048, f32s, f32s);
    unsetenv("COLI_VK_TIER_FMT15");
    printf("  expert bytes: ue8m0-packed %.2f MiB, f32 scales %.2f MiB (x %.4f)\n",
           bp / 1048576.0, bf / 1048576.0, (double)bf / (double)bp);
    CHECK(bp < bf, "packed scales: the packed form did not shrink the expert");
    CHECK((double)bf / bp > 1.176 && (double)bf / bp < 1.177, "packed scales: the shrink is not the scales' 4x on 12 of 15 MiB");
    /* the same budget holds the same GiB in more experts: fit scales with the bytes */
    double more = (double)bf / bp;
    CHECK(more * bp >= bf - 4096, "packed scales: the capacity multiplier does not hold the budget");
    unsetenv("COLI_VK_TIER_FMT15");
}

/* ---- the dedicated S=1 GEMV (COLI_VK_V4_S1GEMV) ----------------------------------- */
/* The fork's qmatmul_s1_mx.comp (its qmatmul_s1_fp4.comp) on the down of a decode row,
 * A/B'd bit for bit against the generic qmatmul dispatch through the tier's own decode
 * groups: one model, every expert pre-loaded so both arms take the same rows, the PRNG
 * rewound so route and x match, coli_vk_xb_s1gemv_force the only difference. The arm's
 * own parity leg (backend_vulkan.c xb_s1gemv_parity, run when the lever arms) already
 * asserts the two kernels on the device; this adds the batch's plumbing -- the sets, the
 * dynamic offsets and the pick itself -- and, without the lever, that nothing armed. */
static void s1gemv_case(int leg) {
    if (!leg) {
        /* gap-wave4: unset ARMS the kernel (the parity gate is the lever's own probe),
         * so the off-state invariants are the explicit halves': COLI_VK_V4_S1GEMV=0
         * and COLI_VK_LEGACY=1 (vk-tier-check's legacy leg runs them) assert the
         * nothing-armed/no-dispatch pair, the unset default the armed kernel */
        ColiVkXbStats st; coli_vk_xb_stats(&st);
        const char *e = getenv("COLI_VK_V4_S1GEMV");
        int off = (e && *e && !atoi(e)) || coli_vk_legacy();
        if (off) {
            CHECK(!coli_vk_xb_s1gemv_armed(0), "s1gemv: armed with the lever off");
            CHECK(st.s1_gemvs == 0, "s1gemv: %llu dispatches with the lever off", st.s1_gemvs);
        } else {
            CHECK(coli_vk_xb_s1gemv_armed(0),
                  "s1gemv: the unset default armed nothing (shader absent or the parity leg refused)");
            printf("  %-44s the default armed the kernel (%llu dispatches so far)\n",
                   "MXFP4 ue8m0 S=1 down: the default arm", st.s1_gemvs);
        }
        return;
    }
    int h0 = H, f0 = F;
    VktFmt f = {VKT_SRC_MXFP4_E8M0, 32};
    model_make(f, f);
    const float limit = 10.0f;
    setenv("COLI_VK_TIER_SYNC", "1", 1);
    setenv("COLI_VK_TIER_RATE", "64", 1);
    set_budget(L * E, f, f);
    /* v4_act's own step mix: the wide steps warm the tier (its residents are what a
     * decode group takes), the S=1 steps are the pick's; the S>1 steps' experts carry
     * more than one row, so the pick correctly stays off them */
    static const int Ss[] = {1, 3, 20, 1, 2, 24};
    enum { ST = 6, MR = 51 * K * L };   /* sum(Ss) = 51 rows a layer */
    float *cap[2] = {malloc((size_t)MR * HF_MAX * sizeof(float)), malloc((size_t)MR * HF_MAX * sizeof(float))};
    uint8_t *tk[2] = {malloc(MR), malloc(MR)};
    unsigned save = rng;
    int ran = cap[0] && cap[1] && tk[0] && tk[1], nrows = 0;
    unsigned long long base = 0, fired = 0;
    for (int arm = 0; arm < 2 && ran; arm++) {
        VktConfig vc = cfg_of(f, f, VKT_ACT_SWIGLU_V4, limit);
        int on = vkt_init(&vc, NULL);   /* a fresh tier an arm: the same residents, the same takes */
        CHECK(on, "s1gemv: the tier did not start (arm %d)", arm);
        if (arm) CHECK(coli_vk_xb_s1gemv_armed(0),
                      "s1gemv: the lever armed nothing (shader absent or the parity leg refused)");
        if (!on) break;
        coli_vk_xb_s1gemv_force(arm ? 1 : 0);   /* arm 0: the generic form */
        rng = save;
        int ri = 0;
        for (int t = 0; t < ST; t++)
            for (int l = 0; l < L; l++) {
                int S = Ss[t], idx[24 * K]; float w[24 * K];
                route(S, 0, E, idx, w);
                float *xq = malloc(sizeof(float) * S * H);
                uint8_t taken[24 * K]; const float *rows[24 * K];
                for (int i = 0; i < S * H; i++) xq[i] = frnd() * 4.0f;
                for (int s = 0; s < S; s++) qdq128(xq + (size_t)s * H, H);
                int nq = vkt_issue_w(l, xq, S, K, idx, w, taken);
                int def = S == 1 && vkt_cs_armed();   /* the armed default defers: the finish site's group (the engine's order) */
                int grp = def ? vkt_issue_group(l, xq, S, K, idx, w, NULL, taken) : 0;
                for (int i = 0; i < S * K; i++) {   /* the engine's share: the tier learns these bytes (v4_act's flow) */
                    if (taken[i]) continue;
                    VktExpertSrc src = src_of(&ex[l][idx[i]]); vkt_note(l, idx[i], &src);
                }
                int joined = grp ? vkt_join(rows) : (def ? 1 : (nq ? vkt_join(rows) : 1));
                CHECK(joined, "s1gemv: join failed (arm %d)", arm);
                for (int i = 0; i < S * K; i++) {
                    if (!arm) {
                        tk[0][ri] = (uint8_t)(taken[i] && joined);
                        if (tk[0][ri]) memcpy(cap[0] + (size_t)ri * H, rows[i], (size_t)H * sizeof(float));
                    } else {
                        tk[1][ri] = (uint8_t)(taken[i] && joined);
                        if (tk[1][ri]) memcpy(cap[1] + (size_t)ri * H, rows[i], (size_t)H * sizeof(float));
                    }
                    ri++;
                }
                free(xq);
            }
        if (!arm) nrows = ri;
        else CHECK(ri == nrows, "s1gemv: %d rows against the first arm's %d", ri, nrows);
        vkt_shutdown();
        ColiVkXbStats stp; coli_vk_xb_stats(&stp);
        if (!arm) base = stp.s1_gemvs;
        else fired = stp.s1_gemvs - base;   /* the tier's shutdowns keep the batch's counters: the delta is this case's */
    }
    coli_vk_xb_s1gemv_force(-1);
    int same = 0;
    if (ran)
        for (int ri = 0; ri < nrows; ri++) {
            CHECK(tk[0][ri] == tk[1][ri], "s1gemv: the arms disagree on row %d (%d vs %d)", ri, tk[0][ri], tk[1][ri]);
            if (tk[0][ri] && tk[1][ri]) {
                CHECK(!memcmp(cap[0] + (size_t)ri * H, cap[1] + (size_t)ri * H, (size_t)H * sizeof(float)),
                      "s1gemv: row %d not bit-identical between the generic and the dedicated arm", ri);
                same++;
            }
        }
    printf("  %-44s %d of %d rows bit-identical between the arms, %llu dedicated down GEMVs\n",
           "MXFP4 ue8m0 S=1 down: qmatmul vs qmatmul_s1_mx", same, nrows, fired);
    CHECK(fired > 0, "s1gemv: the dedicated GEMV never fired (%llu)", fired);
    CHECK(same > 0, "s1gemv: no row ran on the device in both arms");
    free(cap[0]); free(cap[1]); free(tk[0]); free(tk[1]);
    unsetenv("COLI_VK_TIER_SYNC"); unsetenv("COLI_VK_TIER_RATE");
    H = h0; F = f0;
    model_free();
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
    /* the armed leg: the whole default suite with fmt 15 pinned on -- the default
     * itself again since PW-Dfix, so the env is belt-and-braces: every case runs the
     * packed path, and packed_scales' shrink half asserts it at the unset default.
     * tier_fmt15() re-reads the env on every call (no cache), so the set's timing is
     * free; it stays at main's top regardless. */
    /* the counters-split leg (c's): one invocation with COLI_VK_CHAIN_DPROF on from
     * the top -- vk_tier.c caches the env at its first timing call, so it must be set
     * before any tier activity; the whole suite runs with its buckets on (counts and
     * walls only, the checked values unaffected) */
    int dprof_leg = argc > 2 && !strcmp(argv[2], "dprof");
    if (dprof_leg) setenv("COLI_VK_CHAIN_DPROF", "1", 1);
    int fmt15_leg = argc > 2 && !strcmp(argv[2], "fmt15");
    if (fmt15_leg) setenv("COLI_VK_TIER_FMT15", "1", 1);
    /* the decode-growth headroom legs' own invocation (see decode_grow's comment): their
     * run-to-run nondeterminism must stay out of the plain one the strict gate compares */
    int dg_leg = argc > 2 && !strcmp(argv[2], "dgrow");
    /* the dedicated S=1 GEMV's own invocation: the whole suite with the fork's kernel
     * armed on every decode down (each case's own reference CHECKs then cover it),
     * plus the A/B case below -- the arm's parity leg has already run at the arm */
    int s1_leg = argc > 2 && !strcmp(argv[2], "s1gemv");
    if (s1_leg) setenv("COLI_VK_V4_S1GEMV", "1", 1);
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
    printf("host room:\n"); host_room();   /* pure arithmetic: needs no device */
    printf("packed scales:\n"); packed_scales(fmt15_leg);   /* the round trip and the byte arithmetic, before the device's fill */
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
    printf("hot set:\n"); hotset();
    printf("pin layer:\n"); pin_layer();
    printf("pool layer:\n"); pool_layer();
    printf("bank layer:\n"); bank_layer_case();
    printf("decode grow:\n"); decode_grow(dg_leg);
    printf("tier adaptive:\n"); tier_adaptive_case();
    printf("DeepSeek V4:\n"); v4_act();
    printf("compute site:\n"); cs_group();
    printf("compute site alias cache:\n"); cs_alias_cache();
    printf("compute site VA reuse:\n"); cs_alias_va_reuse();
    printf("compute site scan skip:\n"); cs_alias_scan_skip(dprof_leg);
    printf("s1 GEMV:\n"); s1gemv_case(s1_leg);
    printf("stream:\n"); stream();
    printf("stream batch:\n"); stream_batch();
    printf("stream rows adaptive:\n"); stream_rows_adaptive();
    printf("stream group:\n"); stream_group();
    printf("stream pred:\n"); stream_pred();
    printf("stream ring:\n"); stream_ring();
    printf("layers:\n"); layers();
    printf("promote2:\n"); promote_partial_case();
    printf("promote:\n"); promote_case();
    printf("hybrid:\n");
    hybrid_case((VktFmt){(VktSrc)(VKT_SRC_GGML + GQ_Q4_K), 0}, (VktFmt){(VktSrc)(VKT_SRC_GGML + GQ_Q6_K), 0}, "Q4_K/Q6_K");
    hybrid_case((VktFmt){VKT_SRC_I4U_PAIRS_GS, 64}, (VktFmt){VKT_SRC_I4U_PAIRS_GS, 64}, "int4-g64");
    hybrid_case((VktFmt){VKT_SRC_I8_ROW, 0}, (VktFmt){VKT_SRC_I8_ROW, 0}, "int8");
    v4_hybrid_case();
    printf("model geometry:\n"); model_hybrid_case();
    printf("pool copy:\n"); pool_copy_case();
    printf("host pairs:\n"); host_pairs_case(2e-3);
    printf("host pool:\n"); host_plan();
    ColiVkPoolStats ps, px; coli_vk_pool_stats(1, &ps); coli_vk_pool_stats(4, &px);
    CHECK(ps.live == 0 && px.live == 0, "%d + %d tier ranges still live after every shutdown", ps.live, px.live);
    coli_vk_shutdown();   /* with staged uploads: where the experts were ("[VK] memory at exit") */
    stream_commit_failure(spv);
    printf("wait timeout:\n"); wait_timeout(spv);
    printf("window:\n"); window_case(spv);
    printf(fails ? "FAIL (%d)\n" : "PASS\n", fails);
    return fails != 0;
}
