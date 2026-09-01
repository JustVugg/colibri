/* qwen38_tier.c -- CUDA VRAM expert tier for the qwen38 engine.
 *
 * See qwen38_tier.h for why this exists and what it deliberately does not do.
 * This file is compiled only under -DCOLI_CUDA; without it the header's inline
 * stubs keep the engine exactly as it is today.
 *
 * THE SHAPE OF THE PROBLEM, measured on this hardware
 *
 * Qwen3.8-Flash-Next is hidden 2560, moe_intermediate 640, 512 experts across
 * 48 layers, top-k 10. One expert is gate[640,2560] + up[640,2560] +
 * down[2560,640] = 4.69 MiB of fp8, so the whole routed set is ~112.6 GiB and a
 * 24 GB card holds roughly a fifth of it. Residency is a bet on routing skew,
 * exactly as it is for the GLM tier.
 *
 * TOP-K 10 AGAINST AN 8-ROW API. backend_cuda.h:141 states the expert-group
 * contract plainly: "Small totals only (<=8 rows); one outstanding issue per
 * device." qwen36's tier never meets this because its top-k is smaller - its
 * per-device input block is sized 8*D and that is the whole budget. Qwen3.8
 * routes TEN experts per token per layer, so at most 8 can be issued and the
 * remainder must stay on the CPU. That is not a workaround, it is the API's
 * documented limit, and the split is made explicit here rather than discovered
 * as a silent truncation later.
 *
 * NO EVICTION IN THIS VERSION, and that is a deliberate omission rather than an
 * oversight. Experts are uploaded first-come until the budget is spent, then
 * every later miss stays on the CPU. A heat-ranked LFRU policy is what
 * qwen36_tier.c does and it is the obvious next step, but it changes which
 * experts are resident from token to token and would make the first
 * correctness comparison against the CPU path harder to read. Correctness
 * first, policy second.
 */
#include "qwen38_tier.h"

#ifdef COLI_CUDA

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "backend_cuda.h"
/* E4M3_LUT lives here. quant.h is header-only and its table is `static const`,
 * so this translation unit gets its own 1 KB copy rather than a link
 * dependency - the same arrangement colibri.c uses at its own call site. */
#include "quant.h"

#define Q38T_MAX_DEV   COLI_CUDA_MAX_DEVICES
/* The API's own ceiling (backend_cuda.h:141), not a number chosen here. */
#define Q38T_MAX_ISSUE 8

typedef struct {
    ColiCudaTensor *tg, *tu, *td;
    int resident;                 /* 1 once all three projections are uploaded */
} Q38Slot;

static struct {
    int on, ndev, dev[Q38T_MAX_DEV];
    int layers, experts, D, I, topk;
    Q38Slot *slot;                /* [layers * experts] */
    float *xrep[Q38T_MAX_DEV];    /* Q38T_MAX_ISSUE copies of x, per device */
    /* per-device issue bookkeeping, valid between issue() and take() */
    int is_cnt[Q38T_MAX_DEV];
    int is_k[Q38T_MAX_DEV][Q38T_MAX_ISSUE];
    /* counters, reported by q38t_stats */
    unsigned long long hits, miss, over_cap, uploads, upload_fail;
    size_t vram_used;      /* RAW weight bytes uploaded, for reporting only */
    size_t byte_ceiling;   /* optional user cap on the above, 0 = none */
} G;

static Q38Slot *qs(int layer, int eid) {
    return &G.slot[(size_t)layer * (size_t)G.experts + (size_t)eid];
}

/* One expert's three projections, in RAW weight bytes. fp8 = one byte per
 * weight. This is what the tensors CONTAIN, and deliberately NOT what they COST
 * on the device - see headroom_ok(). */
static size_t expert_bytes(void) {
    return (size_t)G.I * (size_t)G.D          /* gate */
         + (size_t)G.I * (size_t)G.D          /* up   */
         + (size_t)G.D * (size_t)G.I;         /* down */
}

/* MEASURE THE CARD, DO NOT MODEL IT.
 *
 * The first version of this file budgeted by adding expert_bytes() per upload.
 * That undercounts: each expert is THREE separate allocations and CUDA rounds
 * every one up to a 2 MiB page, so 4.69 MiB of weights costs 6.00 MiB of
 * device memory - a 1.28x overcommit. The tier filled 22.5 GB of a 24 GB card
 * by its own accounting while actually consuming all of it, and decode went
 * from 3.9 s to 40 s to hanging as the driver thrashed.
 *
 * Asking the driver how much is free is immune to all of that - page
 * granularity, per-tensor descriptors, fragmentation, and anything else a
 * model would have to predict correctly. It costs one cudaMemGetInfo per
 * upload, and uploads stop entirely once the tier is full.
 *
 * RESERVE is what decode itself needs on top of the resident experts: the
 * activation buffers, the pinned result rows, and whatever the driver keeps.
 * 2 GB matches what colibri's own GLM tier holds back. */
#define Q38T_RESERVE_BYTES ((size_t)2e9)

static int headroom_ok(int device) {
    size_t freeb = 0, totb = 0;
    if (!coli_cuda_mem_info(device, &freeb, &totb)) return 0;   /* unknown: refuse */
    /* The expert about to be uploaded must fit AND leave the reserve intact.
     * expert_bytes() understates the real cost, so this is deliberately the
     * conservative side of the comparison. */
    return freeb > Q38T_RESERVE_BYTES + expert_bytes();
}

static int parse_devices(void) {
    const char *many = getenv("COLI_GPUS"), *one = getenv("COLI_GPU");
    const char *list = many ? many : one;
    if (!list || !*list) { G.dev[0] = 0; return 1; }
    int n = 0;
    for (const char *p = list; *p && n < Q38T_MAX_DEV; ) {
        char *end = NULL;
        long v = strtol(p, &end, 10);
        if (end == p || v < 0) return 0;
        G.dev[n++] = (int)v;
        p = end;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == ',') p++; else break;
    }
    return n;
}

/* An optional hard ceiling on raw expert bytes, for a user who wants the tier
 * smaller than the card allows - sharing the GPU with something else, say.
 * Zero (the default, and CUDA_EXPERT_GB=auto) means no ceiling: growth is
 * governed by headroom_ok() alone, which measures the device instead of
 * predicting it. */
static size_t byte_ceiling(void) {
    const char *e = getenv("CUDA_EXPERT_GB");
    if (!e || !*e || strcmp(e, "auto") == 0) return 0;
    double g = atof(e);
    return g > 0 ? (size_t)(g * 1e9) : 0;
}

int q38t_init(int n_layers, int n_experts, int hidden, int inter, int topk) {
    const char *on = getenv("COLI_CUDA");
    if (!on || atoi(on) == 0) return 0;                   /* opt-in, always */
    if (n_layers < 1 || n_experts < 1 || hidden < 1 || inter < 1) return 0;

    memset(&G, 0, sizeof G);
    G.ndev = parse_devices();
    if (G.ndev < 1) {
        fprintf(stderr, "[q38tier] invalid COLI_GPUS -> CPU path\n");
        return 0;
    }
    if (!coli_cuda_init(G.dev, G.ndev)) {
        fprintf(stderr, "[q38tier] coli_cuda_init failed -> CPU path\n");
        return 0;
    }
    /* fmt=8 uploads are REFUSED until the e4m3 decode table is published
     * (backend_cuda.h:80). Publishing it after an upload would leave kernels
     * decoding against a zero table, which is silently wrong rather than
     * loudly broken - so this must happen here, before anything is uploaded. */
    if (!coli_cuda_fp8_set_lut(E4M3_LUT)) {
        fprintf(stderr, "[q38tier] fp8 LUT publish failed -> CPU path\n");
        return 0;
    }

    G.layers = n_layers; G.experts = n_experts;
    G.D = hidden; G.I = inter; G.topk = topk;
    G.slot = (Q38Slot *)calloc((size_t)n_layers * (size_t)n_experts, sizeof(Q38Slot));
    if (!G.slot) return 0;
    for (int i = 0; i < G.ndev; i++) {
        G.xrep[i] = (float *)malloc((size_t)Q38T_MAX_ISSUE * (size_t)G.D * sizeof(float));
        if (!G.xrep[i]) return 0;
    }
    G.byte_ceiling = byte_ceiling();
    G.on = 1;

    fprintf(stderr,
            "[q38tier] %d device(s), %d layers x %d experts, top-k %d "
            "(<=%d issued per call), %.2f MiB/expert, growth capped by measured "
            "free VRAM (%.1f GB reserve)%s\n",
            G.ndev, n_layers, n_experts, topk, Q38T_MAX_ISSUE,
            expert_bytes() / (1024.0 * 1024.0),
            Q38T_RESERVE_BYTES / 1e9,
            G.byte_ceiling ? " plus an explicit CUDA_EXPERT_GB ceiling" : "");
    if (topk > Q38T_MAX_ISSUE)
        fprintf(stderr,
                "[q38tier] top-k %d exceeds the %d-row expert-group limit: "
                "%d expert(s) per token stay on the CPU path\n",
                topk, Q38T_MAX_ISSUE, topk - Q38T_MAX_ISSUE);
    return 1;
}

int q38t_ready(void) { return G.on; }

int q38t_is_resident(int layer, int eid) {
    if (!G.on || layer < 0 || layer >= G.layers || eid < 0 || eid >= G.experts) return 0;
    return qs(layer, eid)->resident;
}

void q38t_note(int layer, int eid,
               const void *g, const float *gs,
               const void *u, const float *us,
               const void *d, const float *ds) {
    if (!G.on || layer < 0 || layer >= G.layers || eid < 0 || eid >= G.experts) return;
    /* A checkpoint may legitimately carry non-FP8 experts; those have no scale
     * table and simply stay on the CPU path rather than being forced. */
    if (!g || !u || !d || !gs || !us || !ds) return;

    Q38Slot *s = qs(layer, eid);
    if (s->resident) return;

    int dv = G.dev[eid % G.ndev];        /* one home device per expert, no duplicates */
    /* Ask the card, do not model it. Once full this returns 0 for every later
     * expert and the tier simply stops growing - decode carries on at the CPU
     * speed it had before, rather than degrading past it. */
    if (G.byte_ceiling && G.vram_used + expert_bytes() > G.byte_ceiling) {
        G.over_cap++; return;                /* explicit user ceiling, if set */
    }
    if (!headroom_ok(dv)) { G.over_cap++; return; }
    /* upload(tensor, weights, scales, fmt, I, O, device) - note I then O. */
    if (coli_cuda_tensor_upload(&s->tg, g, gs, 8, G.D, G.I, dv)
     && coli_cuda_tensor_upload(&s->tu, u, us, 8, G.D, G.I, dv)
     && coli_cuda_tensor_upload(&s->td, d, ds, 8, G.I, G.D, dv)) {
        s->resident = 1;
        G.vram_used += expert_bytes();
        G.uploads++;
    } else {
        s->tg = s->tu = s->td = NULL;
        G.upload_fail++;
    }
}

uint32_t q38t_issue(int layer, const int *eids, int K, const float *x) {
    if (!G.on || !eids || !x || K < 1 || K > 32) return 0;

    ColiCudaTensor *tg[Q38T_MAX_DEV][Q38T_MAX_ISSUE];
    ColiCudaTensor *tu[Q38T_MAX_DEV][Q38T_MAX_ISSUE];
    ColiCudaTensor *td[Q38T_MAX_DEV][Q38T_MAX_ISSUE];
    static int rows[Q38T_MAX_ISSUE];
    for (int i = 0; i < Q38T_MAX_ISSUE; i++) rows[i] = 1;   /* S=1 per expert at decode */

    for (int i = 0; i < G.ndev; i++) G.is_cnt[i] = 0;

    uint32_t mask = 0;
    for (int k = 0; k < K; k++) {
        Q38Slot *s = qs(layer, eids[k]);
        if (!s->resident) { G.miss++; continue; }
        int di = (eids[k] % G.ndev);
        int c = G.is_cnt[di];
        /* The 8-row ceiling is per device per issue. Anything past it is a
         * CPU expert this token; it is counted so the split is visible in
         * q38t_stats rather than being an invisible truncation. */
        if (c >= Q38T_MAX_ISSUE) { G.over_cap++; continue; }
        tg[di][c] = s->tg; tu[di][c] = s->tu; td[di][c] = s->td;
        G.is_k[di][c] = k;
        G.is_cnt[di] = c + 1;
        mask |= 1u << k;
        G.hits++;
    }

    for (int di = 0; di < G.ndev; di++) {
        int c = G.is_cnt[di];
        if (!c) continue;
        float *xr = G.xrep[di];
        for (int j = 0; j < c; j++) memcpy(xr + (size_t)j * G.D, x, (size_t)G.D * sizeof(float));
        if (!coli_cuda_expert_group_issue(tg[di], tu[di], td[di], rows, c, xr)) {
            /* Hand this device's experts back to the CPU rather than dropping
             * their contribution - a missing expert is a wrong answer, not a
             * slow one. */
            for (int j = 0; j < c; j++) mask &= ~(1u << G.is_k[di][j]);
            G.is_cnt[di] = 0;
        }
    }
    return mask;
}

void q38t_take(uint32_t mask, const float *route_gates, int K, float *out) {
    (void)K;
    if (!G.on || !mask || !route_gates || !out) return;
    for (int di = 0; di < G.ndev; di++) {
        int c = G.is_cnt[di];
        if (!c) continue;
        const float *y = coli_cuda_expert_group_take(G.dev[di]);
        if (!y) { G.is_cnt[di] = 0; continue; }
        for (int j = 0; j < c; j++) {
            float w = route_gates[G.is_k[di][j]];
            const float *row = y + (size_t)j * G.D;
            for (int d = 0; d < G.D; d++) out[d] += w * row[d];
        }
        G.is_cnt[di] = 0;
    }
}

void q38t_stats(void) {
    if (!G.on) return;
    unsigned long long total = G.hits + G.miss;
    fprintf(stderr,
            "[q38tier] resident %llu expert(s), %.2f GB of raw expert weight\n"
            "[q38tier] group hits %llu, misses %llu (%.1f%% hit), "
            "declined %llu, upload failures %llu\n",
            G.uploads, G.vram_used / 1e9,
            G.hits, G.miss, total ? 100.0 * (double)G.hits / (double)total : 0.0,
            G.over_cap, G.upload_fail);
}

void q38t_shutdown(void) {
    if (!G.on) return;
    q38t_stats();
    free(G.slot); G.slot = NULL;
    for (int i = 0; i < G.ndev; i++) { free(G.xrep[i]); G.xrep[i] = NULL; }
    G.on = 0;
}

#endif /* COLI_CUDA */
