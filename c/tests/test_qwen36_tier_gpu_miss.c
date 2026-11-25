/* QT_GPU_MISS on the fake backend: routed misses computed in the device's group.
 *
 * Pinned here: the switch needs the arena and the async overwrite (else off, one
 * line, nothing allocated); at init it takes topk transient arena slots per device
 * out of the expert budget and one pinned staging buffer each; a staged miss's
 * group row equals the CPU expert row (the fake group computes MXFP4 with quant.h
 * from the bytes the tensors point at); a slot is not reused before qt_take (no
 * free slot -> 0, fallback counted); a refused copy or a refused group leaves
 * those ranks to the CPU, the counters right and the stream drained at the take;
 * shutdown frees the pinned memory and the slabs. Rows at once (a verify): rows
 * routed to one expert share its group entry, a missed expert is staged once for
 * all of them, each row takes back its own ranks. QT_GPU_MISS_PROMOTE=1: a swap
 * qt_note decides for a staged miss is copied from its transient slot into the
 * victim's after the group launch, with no host staging copy. No GPU. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../compat.h"
#include "../quant.h"

#define coli_cuda_expert_group_issue fixture_issue
#define coli_cuda_expert_group_take fixture_take
#include "qwen36_fake_cuda.h"
#undef coli_cuda_expert_group_issue
#undef coli_cuda_expert_group_take

enum { NL = 2, NE = 8, D = 64, IH = 32, CAP = 1, TOPK = 4 };
#define MB (D * IH / 2)
#define NGU (IH * (D / 32))
#define ND (D * (IH / 32))

static unsigned char wts[NE][3 * MB];
static unsigned char e8[NE][2 * NGU + ND];

static void expert_row(float *y, const float *x, const uint8_t *gw, const uint8_t *uw, const uint8_t *dw,
                       const uint8_t *gs, const uint8_t *us, const uint8_t *ds) {
    float g[IH], u[IH];
    matmul_mxfp4(g, x, gw, gs, 1, D, IH);
    matmul_mxfp4(u, x, uw, us, 1, D, IH);
    for (int i = 0; i < IH; i++) g[i] = (g[i] / (1.f + expf(-g[i]))) * u[i];
    matmul_mxfp4(y, g, dw, ds, 1, IH, D);
}
static void cpu_row(float *y, const float *x, int e) {
    expert_row(y, x, wts[e], wts[e] + MB, wts[e] + 2 * MB, e8[e], e8[e] + NGU, e8[e] + 2 * NGU);
}

/* the fake group: rows[c] consecutive input rows per expert, every row from the bytes
 * its tensors point at (pinned staging for a transient slot, the uploader's copy for
 * a resident) */
static float res[32 * D];
static int issues, issue_refuse, pending, last_rows[32], last_count, async_at_issue;
static int pend_rows, takes, appends, append_refuse;   /* the pending group's rows; takes; QT_GPU_MISS_FOLD's appends */
int coli_cuda_expert_group_issue(ColiCudaTensor *const *g, ColiCudaTensor *const *u,
                                 ColiCudaTensor *const *d, const int *rows, int count, const float *x) {
    issues++; async_at_issue = fake_async_overwrites;
    if (issue_refuse) return 0;
    int off = 0;
    for (int c = 0; c < count; c++) {
        last_rows[c] = rows[c];
        for (int r = 0; r < rows[c]; r++, off++)
            expert_row(res + (size_t)off * D, x + (size_t)off * D, g[c]->w, u[c]->w, d[c]->w,
                       (const uint8_t *)g[c]->sc, (const uint8_t *)u[c]->sc, (const uint8_t *)d[c]->sc);
    }
    last_count = count;
    pending = 1; pend_rows = off;
    return 1;
}
static int take_fail;   /* 1: the next take answers NULL (a failed device) */
const float *coli_cuda_expert_group_take(int device) {
    (void)device;
    takes++;
    if (take_fail) { take_fail = 0; pending = 0; return NULL; }
    if (!pending) return NULL;
    pending = 0;
    return res;
}
/* QT_GPU_MISS_FOLD: the fake's coli_cuda_expert_group_append (fake_append_hook): its rows
 * land after the pending group's, nothing is taken; all-NULL arguments are the probe */
static int fixture_append(ColiCudaTensor *const *g, ColiCudaTensor *const *u,
                          ColiCudaTensor *const *d, const int *rows, int count, const float *x) {
    if (!g && !x) return 1;
    if (append_refuse || !pending) return 0;
    int off = pend_rows;
    for (int c = 0; c < count; c++)
        for (int r = 0; r < rows[c]; r++, off++)
            expert_row(res + (size_t)off * D, x + (size_t)(off - pend_rows) * D, g[c]->w, u[c]->w, d[c]->w,
                       (const uint8_t *)g[c]->sc, (const uint8_t *)u[c]->sc, (const uint8_t *)d[c]->sc);
    pend_rows = off; appends++;
    return 1;
}

#include "../qwen36_tier.c"

static int fails;
static void check(int ok, const char *what) {
    if (!ok) { printf("  FAIL: %s\n", what); fails++; }
}
static void note(int l, int e) {
    qt_note(l, e, wts[e], wts[e] + MB, wts[e] + 2 * MB,
            (const float *)e8[e], (const float *)(e8[e] + NGU), (const float *)(e8[e] + 2 * NGU));
}
static int mstage_rows(int l, const int *as, int na, int e) {
    return qt_miss_stage(l, as, na, e, wts[e], wts[e] + MB, wts[e] + 2 * MB,
                         (const float *)e8[e], (const float *)(e8[e] + NGU), (const float *)(e8[e] + 2 * NGU));
}
static int mstage(int l, int k, int e) { return mstage_rows(l, &k, 1, e); }
static void counters(uint64_t *gm, uint64_t *fb) { QtCounters q; qt_counters(&q); *gm = q.gpu_miss; *fb = q.gpu_miss_fallback; }

/* QT_GPU_MISS=2: each expert also as one record in an engine-like layout (scales, then
 * weights, as qwen38's WEX records order them: gate, down, up), every part 256-aligned */
enum { REC = 3840 };
static const size_t ROFF[6] = {768, 2816, 1792, 0, 512, 256};   /* gate/up/down weights, gate/up/down scales */
static unsigned char rec[NE][REC];
static int pstage_rows(int l, const int *as, int na, int e, int pinned) {
    const unsigned char *b = rec[e];
    return qt_miss_stage_src(l, as, na, e, b + ROFF[0], b + ROFF[1], b + ROFF[2], (const float *)(b + ROFF[3]),
                             (const float *)(b + ROFF[4]), (const float *)(b + ROFF[5]), pinned);
}
/* one decode row routed {5,1,2,3} with 5 resident: 1 and 2 staged (src 0: the plain
 * arrays through the staging buffers, 1: the pinned records), 3 computed here; out =
 * the CPU's rank, then the take. mode 1 begins and commits once, mode 2 launches the
 * hit at routing (qt_issue) and commits the misses after. Returns the final mask. */
static uint32_t row_run(int mode, int src, const float *x, const float *val, float *out, int *issued_at_route) {
    int route[4] = {5, 1, 2, 3}, is0 = issues;
    float row[D];
    uint32_t m = mode == 2 ? qt_issue(0, route, 4, x) : qt_issue_begin(0, route, 4, x);
    *issued_at_route = issues - is0;
    for (int k = 1; k <= 2; k++) {
        int ok = src ? pstage_rows(0, &k, 1, route[k], 1) : mstage(0, k, route[k]);
        if (!ok) printf("  (stage of rank %d declined)\n", k);
    }
    m = qt_issue_commit();
    memset(out, 0, D * sizeof(float));
    for (int k = 0; k < 4; k++) if (!((m >> k) & 1u)) { cpu_row(row, x, route[k]); for (int d = 0; d < D; d++) out[d] += val[k] * row[d]; }
    if (!qt_take(m, val, 4, out)) printf("  (take failed)\n");
    return m;
}

int main(void) {
    fake_host_reg_ok = 1;
    for (int e = 0; e < NE; e++) {
        for (size_t i = 0; i < sizeof wts[e]; i++) wts[e][i] = (unsigned char)((i * 7 + e * 13 + (i >> 5)) & 0xFF);
        for (size_t i = 0; i < sizeof e8[e]; i++) e8[e][i] = (unsigned char)(120 + (i * 3 + e) % 12);
        for (int m = 0; m < 3; m++) {
            memcpy(rec[e] + ROFF[m], wts[e] + m * MB, MB);
            memcpy(rec[e] + ROFF[3 + m], e8[e] + m * NGU, m < 2 ? NGU : ND);
        }
    }
    float x[D];
    for (int i = 0; i < D; i++) x[i] = (float)((i * 37) % 19) / 9.f - 1.f;
    setenv("COLI_CUDA", "1", 1);
    setenv("COLI_GPUS", "0", 1);
    setenv("QT_NO_WARMSTART", "1", 1);
    setenv("COLI_PLACE", "off", 1);
    setenv("HEAT_FILE", "", 1);
    setenv("QT_UPLOAD_SYNC", "1", 1);
    unsetenv("CUDA_EXPERT_GB");
    fake_ndev = 1;

    printf(" 1. switch off\n");
    setenv("QT_ARENA", "1", 1);
    unsetenv("QT_GPU_MISS");
    check(qt_init_mxfp4(NL, NE, D, IH, CAP, TOPK), "tier up");
    check(!qt_gpu_miss_on(TOPK) && fake_host_allocs == 0 && fake_pipe_allocs == 0 && G.used[0] == 0,
          "QT_GPU_MISS unset: no transient slot, no pinned staging, nothing out of the budget");
    int ids[3] = {1, 2, 3};
    uint32_t m0 = qt_issue_begin(0, ids, 3, x);
    check(m0 == 0 && !mstage(0, 0, 1) && G.is_cnt[0] == 0 && qt_issue_commit() == 0 && issues == 0,
          "off: qt_miss_stage declines, nothing is launched");
    check(qt_take(0, NULL, 3, NULL) && fake_pipe_syncs == 0, "off: the take has nothing to drain");
    uint64_t gm, fb; counters(&gm, &fb);
    check(gm == 0 && fb == 0, "off: the GPU-miss counters stay 0");
    qt_shutdown();

    setenv("QT_GPU_MISS", "1", 1);
    unsetenv("QT_ARENA");
    check(qt_init_mxfp4(NL, NE, D, IH, CAP, TOPK) && !qt_gpu_miss_on(TOPK) && fake_host_allocs == 0,
          "arena off: the switch stays off");
    qt_shutdown();
    setenv("QT_ARENA", "1", 1);
    fake_async_absent = 1;
    check(qt_init_mxfp4(NL, NE, D, IH, CAP, TOPK) && !qt_gpu_miss_on(TOPK) && fake_host_allocs == 0 && G.used[0] == 0,
          "older backend (no async overwrite): the switch stays off");
    qt_shutdown();
    fake_async_absent = 0;

    printf(" 2. switch on: init\n");
    int pa0 = fake_pipe_allocs, ua0 = fake_upload_view_n; size_t pb0 = fake_pipe_bytes;
    check(qt_init_mxfp4(NL, NE, D, IH, CAP, TOPK) && qt_gpu_miss_on(TOPK), "QT_GPU_MISS=1 + QT_ARENA=1: on");
    check(qt_gpu_miss_on(32) && !qt_gpu_miss_on(33) && !qt_gpu_miss_on(0), "a group of up to 32 rows fits a device group");
    check(G_gm.n == TOPK && fake_host_allocs == 1, "topk slots, one pinned allocation per device");
    check(fake_upload_view_n - ua0 == 3 * TOPK && fake_pipe_allocs - pa0 == 1, "three arena views per slot, in one slab");
    check(fake_pipe_bytes - pb0 == (size_t)TOPK * G_ar.slot && G_ar.d[0].nslot == TOPK && G_ar.d[0].nfr == 0,
          "that slab holds the transient slots only, not a full arena slab");
    check(G.used[0] == (size_t)TOPK * G.exp_bytes, "the transient slots come out of the expert budget");
    size_t exp_b = G.exp_bytes;
    size_t slot_bytes = G_ar.slot;
    int resident = 0;
    for (int l = 0; l < NL; l++) for (int e = 0; e < NE; e++) resident += qs(l, e)->resident;
    check(resident == 0, "and are no expert's residency");

    printf(" 3. a staged miss is the CPU expert row\n");
    note(0, 5); qt_fill_wait();
    check(qt_is_resident(0, 5), "expert 5 resident");
    int route[4] = {5, 1, 2, 3};
    float val[4] = {0.4f, 0.3f, 0.2f, 0.1f};
    uint32_t hit = qt_issue_begin(0, route, 4, x);
    check(hit == 1u && issues == 0, "begin: expert 5 is the hit, nothing launched yet");
    check(mstage(0, 1, 1) && mstage(0, 2, 2), "two misses staged");
    uint8_t *slot0 = G_gm.h[0];
    check(G.is_cnt[0] == 3 && G.is_k[0][1] == 1 && G.is_k[0][2] == 2 && G.is_tg[0][1] == G_gm.tg[0][0],
          "they join the device's group as ranks 1 and 2, in transient slot 0 and 1");
    check(!memcmp(slot0, wts[1], 3 * MB) && !memcmp(slot0 + G_gm.wb, e8[1], 2 * NGU + ND),
          "slot 0's pinned staging holds expert 1's nibbles and scale bytes unchanged");
    uint32_t mask = qt_issue_commit();
    check(mask == 7u && issues == 1, "commit: one group, hits + staged");
    float out[D], want[D], row[D];
    memset(out, 0, sizeof out); memset(want, 0, sizeof want);
    cpu_row(row, x, 3); for (int d = 0; d < D; d++) want[d] += val[3] * row[d];   /* the CPU's rank */
    memcpy(out, want, sizeof out);
    int same = 1;
    for (int k = 0; k < 3; k++) {
        cpu_row(row, x, route[k]);
        same &= !memcmp(res + (size_t)k * D, row, sizeof row);
        for (int d = 0; d < D; d++) want[d] += val[k] * row[d];
    }
    check(same, "every group row is bit-identical to the CPU expert row");
    check(qt_take(mask, val, 4, out) && !memcmp(out, want, sizeof out), "the take adds the GPU ranks to the CPU's");
    counters(&gm, &fb);
    check(gm == 2 && fb == 0 && fake_pipe_syncs == 0, "gpu_miss 2, fallback 0, nothing to drain");

    printf(" 4. a slot is reused only after the take\n");
    int r4[4] = {1, 2, 3, 4};
    check(qt_issue_begin(1, r4, 4, x) == 0, "layer 1: four misses");
    for (int k = 0; k < 4; k++) check(mstage(1, k, r4[k]), "staged");
    check(!mstage(1, 3, 6), "a fifth miss finds no free slot: 0, the CPU computes it");
    check(!memcmp(slot0, wts[1], 3 * MB), "slot 0 still holds the open group's first expert");
    counters(&gm, &fb);
    check(gm == 2 && fb == 1, "staged misses count at the commit: gpu_miss 2, fallback 1");
    mask = qt_issue_commit();
    counters(&gm, &fb);
    check(gm == 6 && fb == 1, "gpu_miss 6 after the commit");
    check(mask == 15u && qt_take(mask, val, 4, out), "group of four, taken");
    check(qt_issue_begin(0, r4, 1, x) == 0 && mstage(0, 0, 7) && G.is_tg[0][0] == G_gm.tg[0][0] && !memcmp(slot0, wts[7], 3 * MB),
          "after the take slot 0 takes the next miss");
    mask = qt_issue_commit();
    check(qt_take(mask, val, 1, out), "and is taken again");

    printf(" 5. failures leave the ranks to the CPU\n");
    int syncs = fake_pipe_syncs;
    check(qt_issue_begin(0, r4, 2, x) == 0, "two misses");
    fake_async_fail_at = fake_async_overwrites + 2;     /* the up matrix of the first */
    check(!mstage(0, 0, 1), "a refused copy: 0");
    check(G.is_cnt[0] == 0 && G_gm.used[0] == 1, "no group row; its slot stays taken until the take");
    fake_async_fail_at = 0;
    check(mstage(0, 1, 2) && G.is_tg[0][0] == G_gm.tg[0][1], "the next miss gets slot 1");
    counters(&gm, &fb);
    check(gm == 7 && fb == 2, "gpu_miss 7, fallback 2");
    issue_refuse = 1;
    mask = qt_issue_commit();
    check(mask == 0 && G.is_cnt[0] == 0, "a refused group clears its staged ranks: the caller computes them");
    counters(&gm, &fb);
    check(gm == 7 && fb == 3, "the CPU recomputes the refused group's staged miss: fallback, not gpu_miss");
    check(qt_take(mask, val, 2, out) && fake_pipe_syncs == syncs + 1 && G_gm.used[0] == 0,
          "the take drains the stream for the copies no group took, and frees the slots");
    issue_refuse = 0;

    printf(" 6. two rows in one group (a verify)\n");
    {
        /* row 0 routes 5 1 2 3, row 1 routes 1 5 6 2; 5 is resident. 1 and 2 are staged
         * once for both their rows, 6 for row 1, 3 stays on the CPU. */
        int rr[8] = {5, 1, 2, 3, 1, 5, 6, 2};
        float x2[2 * D], v2[2][4] = {{0.4f, 0.3f, 0.2f, 0.1f}, {0.35f, 0.3f, 0.2f, 0.15f}};
        for (int i = 0; i < D; i++) { x2[i] = x[i]; x2[D + i] = (float)((i * 11) % 13) / 7.f - 0.9f; }
        uint32_t hm[2], fm[2];
        int is0 = issues; uint64_t gm0, fb0; counters(&gm0, &fb0);
        qt_issue_begin_rows(0, rr, 2, 4, x2, hm);
        check(hm[0] == 1u && hm[1] == 2u && issues == is0, "begin: the hit of each row, nothing launched");
        check(G.is_ne[0] == 1 && G.is_rows[0][0] == 2 && G.is_cnt[0] == 2,
              "both rows of resident 5 share one entry of two rows");
        int a1[2] = {1, 4}, a2[2] = {2, 7}, a6 = 6;
        check(mstage_rows(0, a1, 2, 1) && mstage_rows(0, a2, 2, 2) && mstage_rows(0, &a6, 1, 6),
              "1, 2 and 6 staged");
        check(G_gm.used[0] == 3 && G.is_ne[0] == 4 && G.is_cnt[0] == 7, "one transient slot per expert, seven group rows");
        memcpy(fm, hm, sizeof fm);
        qt_issue_commit_rows(fm);
        check(fm[0] == 7u && fm[1] == 15u && issues == is0 + 1, "commit: one group for both rows");
        check(last_count == 4 && last_rows[0] == 2 && last_rows[1] == 2 && last_rows[2] == 2 && last_rows[3] == 1,
              "rows[] per expert: 5, 1 and 2 two rows each, 6 one");
        int ent[7] = {5, 5, 1, 1, 2, 2, 6}, row[7] = {0, 1, 0, 1, 0, 1, 1}, same = 1;
        float y[D];
        for (int j = 0; j < 7; j++) { cpu_row(y, x2 + (size_t)row[j] * D, ent[j]); same &= !memcmp(res + (size_t)j * D, y, sizeof y); }
        check(same, "every group row is the CPU row of its expert and input row");
        /* each row: its CPU rank first, then its device ranks in group order */
        float o2[2][D], w2[2][D];
        memset(w2, 0, sizeof w2);
        cpu_row(y, x2, 3); for (int d = 0; d < D; d++) w2[0][d] += v2[0][3] * y[d];
        memcpy(o2, w2, sizeof o2);
        for (int j = 0; j < 7; j++) {
            int r = row[j], k = r ? (ent[j] == 1 ? 0 : ent[j] == 5 ? 1 : ent[j] == 6 ? 2 : 3) : (ent[j] == 5 ? 0 : ent[j] == 1 ? 1 : 2);
            for (int d = 0; d < D; d++) w2[r][d] += v2[r][k] * res[(size_t)j * D + d];
        }
        check(qt_take_row(0, 0, v2[0], o2[0]) && G.issue_open && !memcmp(o2[0], w2[0], sizeof o2[0]),
              "row 0 takes its ranks, the group stays open");
        check(qt_take_row(1, 1, v2[1], o2[1]) && !memcmp(o2[1], w2[1], sizeof o2[1]),
              "row 1 takes its own from the same drained result");
        check(!G.issue_open && G_gm.used[0] == 0 && G.is_cnt[0] == 0, "the last row closes the group and frees the slots");
        counters(&gm, &fb);
        check(gm == gm0 + 5 && fb == fb0, "gpu_miss counts the five staged assignments");

        int syncs2 = fake_pipe_syncs;
        qt_issue_begin_rows(0, rr, 2, 4, x2, hm);
        check(mstage_rows(0, a1, 2, 1), "staged again");
        issue_refuse = 1;
        memcpy(fm, hm, sizeof fm);
        qt_issue_commit_rows(fm);
        issue_refuse = 0;
        check(fm[0] == 0 && fm[1] == 0, "a refused group: every rank of both rows back to the CPU");
        counters(&gm, &fb);
        check(gm == gm0 + 5 && fb == fb0 + 2, "its two staged assignments count as fallback");
        check(qt_take_row(0, 0, v2[0], o2[0]) && qt_take_row(1, 1, v2[1], o2[1]) && fake_pipe_syncs == syncs2 + 1 && !G.issue_open,
              "the takes drain the stream once and close the group");
    }

    printf(" 7. shutdown\n");
    qt_shutdown();
    check(fake_host_frees == fake_host_allocs && fake_pipe_frees == fake_pipe_allocs && !G_gm.on,
          "pinned staging and slabs freed, switch off");

    printf(" 7b. QT_GPU_MISS_SLOTS\n");
    setenv("QT_GPU_MISS_SLOTS", "6", 1);
    check(qt_init_mxfp4(NL, NE, D, IH, CAP, TOPK) && qt_gpu_miss_on(TOPK) && G_gm.n == 6 &&
          G.used[0] == 6 * exp_b, "6 asked: 6 transient slots out of the budget");
    qt_shutdown();
    setenv("QT_GPU_MISS_SLOTS", "2", 1);
    check(qt_init_mxfp4(NL, NE, D, IH, CAP, TOPK) && G_gm.n == TOPK, "fewer than topk asked: topk");
    qt_shutdown();
    setenv("QT_GPU_MISS_SLOTS", "99", 1);
    check(qt_init_mxfp4(NL, NE, D, IH, CAP, TOPK) && G_gm.n <= 32, "more than a group: capped");
    qt_shutdown();
    unsetenv("QT_GPU_MISS_SLOTS");

    printf(" 8. a budget below the transient slots\n");
    char gb[64]; snprintf(gb, sizeof gb, "%.15f", (double)(TOPK - 1) * (slot_bytes + 64) / 1073741824.0);
    setenv("CUDA_EXPERT_GB", gb, 1);
    int ha = fake_host_allocs;
    check(qt_init_mxfp4(NL, NE, D, IH, CAP, TOPK) && !qt_gpu_miss_on(TOPK) && fake_host_allocs == ha && G.used[0] == 0,
          "too small: the switch stays off, nothing taken");
    qt_shutdown();
    unsetenv("CUDA_EXPERT_GB");
    fake_host_alloc_fail = 1;
    check(qt_init_mxfp4(NL, NE, D, IH, CAP, TOPK) && !qt_gpu_miss_on(TOPK) && G.used[0] == 0 && G_ar.d[0].nslot == 0,
          "no pinned memory: the switch stays off");
    qt_shutdown();
    fake_host_alloc_fail = 0;

    printf(" 9. QT_GPU_MISS_PROMOTE: a staged miss promoted on the device\n");
    snprintf(gb, sizeof gb, "%.15f", (double)((TOPK + 1) * exp_b + exp_b / 2) / 1073741824.0);
    setenv("CUDA_EXPERT_GB", gb, 1);   /* the transient slots and one resident */
    for (int promote = 0; promote < 2; promote++) {
        if (promote) setenv("QT_GPU_MISS_PROMOTE", "1", 1); else unsetenv("QT_GPU_MISS_PROMOTE");
        check(qt_init_mxfp4(NL, NE, D, IH, CAP, TOPK) && qt_gpu_miss_on(TOPK) && G_gm.promote == promote, "tier up");
        note(0, 5); qt_fill_wait();
        QSlot *v = qs(0, 5), *s = qs(0, 1);
        ColiCudaTensor *vt[3] = {v->tg, v->tu, v->td}; uint32_t var = v->ar;
        check(v->resident && var && G.used[0] + G.exp_bytes > G.budget[0], "expert 5 resident, the budget full");
        int route[4] = {1, 2, 3, 4}, k0 = 0, ups = (int)G.uploads, ow = fake_overwrites;
        float val[4] = {0.4f, 0.3f, 0.2f, 0.1f}, out[D];
        check(qt_issue_begin(0, route, 4, x) == 0 && mstage_rows(0, &k0, 1, 1), "expert 1 staged");
        int aw = fake_async_overwrites;
        qt_note_n(0, 1, 10, wts[1], wts[1] + MB, wts[1] + 2 * MB,
                  (const float *)e8[1], (const float *)(e8[1] + NGU), (const float *)(e8[1] + 2 * NGU));
        if (promote) {
            check(!v->resident && s->queued && G.qn == 0 && G.inflight == 0 && fake_async_overwrites == aw,
                  "the swap is taken over: victim 5 out, nothing queued for the uploader, no copy yet");
            check(qt_issue_commit() == 1u && async_at_issue == aw && fake_async_overwrites == aw + 3,
                  "commit: the group launches, then the three device copies");
            const uint8_t *src = ar_ptr(0, G_gm.ar[0][0] - 1);
            check(fake_async_last_t == vt[2] && fake_async_last_w == src + 2 * G_ar.o_w &&
                  fake_async_last_sc == (const float *)(src + G_ar.o_s[2]),
                  "from transient slot 0's device bytes into the victim's down matrix");
            check(s->resident && !s->queued && s->tg == vt[0] && s->td == vt[2] && s->ar == var && !v->tg && !v->ar,
                  "expert 1 resident in the victim's tensors and arena slot");
            check(G_gm.promoted == 1 && (int)G.uploads == ups + 1 && fake_overwrites == ow, "counted, and no host upload");
            memset(out, 0, sizeof out);
            check(qt_take(1u, val, 4, out), "taken");
        } else {
            check(s->queued && G.qn + G.inflight > 0 && qt_issue_commit() == 1u && fake_async_overwrites == aw,
                  "switch off: the host path queues the swap, no device copy");
            memset(out, 0, sizeof out);
            check(qt_take(1u, val, 4, out), "taken");
            qt_fill_wait();
            check(s->resident && !v->resident && G_gm.promoted == 0 && fake_overwrites == ow + 3,
                  "the uploader swapped it with three host overwrites");
        }
        qt_shutdown();
    }
    unsetenv("QT_GPU_MISS_PROMOTE"); unsetenv("CUDA_EXPERT_GB");

    printf("10. QT_GPU_MISS=2: hits at routing, misses as a second group\n");
    {
        float v4[4] = {0.4f, 0.3f, 0.2f, 0.1f}, o1[D], o2[D];
        int at_route;
        /* mode 1 reference */
        check(qt_init_mxfp4(NL, NE, D, IH, CAP, TOPK) && qt_gpu_miss_on(TOPK) == 1, "QT_GPU_MISS=1: mode 1");
        note(0, 5); qt_fill_wait();
        uint32_t m1 = row_run(1, 0, x, v4, o1, &at_route);
        check(m1 == 7u && at_route == 0, "mode 1: nothing launched at routing, one group of three");
        qt_shutdown();

        setenv("QT_GPU_MISS", "2", 1);
        int ha = fake_host_allocs;
        check(qt_init_mxfp4(NL, NE, D, IH, CAP, TOPK) && qt_gpu_miss_on(TOPK) == 2 && G_gm.mode == 2, "QT_GPU_MISS=2: mode 2");
        check(fake_host_allocs == ha + 1 && G_gm.hy[0] && !G_gm.lay && G_gm.ow[2] == 2 * G_ar.o_w && G_gm.os[1] == G_ar.o_s[1],
              "same pinned staging, a hit-row buffer, the arena layout without qt_miss_src_layout");
        note(0, 5); qt_fill_wait();
        int is0 = issues;
        uint32_t m2 = row_run(2, 0, x, v4, o2, &at_route);
        check(m2 == 7u && at_route == 1 && issues == is0 + 2 && last_count == 2, "the hit launched at routing, the two misses as their own group");
        check(!memcmp(o1, o2, sizeof o1), "the row is bit-identical to mode 1's (hit rows, then miss rows)");
        uint64_t gm, fb; counters(&gm, &fb);
        check(gm == 2 && fb == 0 && G_gm.direct == 0, "gpu_miss 2; plain sources took the staging copy");
        check(!G_gm.launched && !G_gm.ht[0] && !G_gm.mp[0] && !G.is_xp && G_gm.used[0] == 0, "the take closes everything");

        /* a row without misses: the hit group stays in flight until the take */
        int r5[1] = {5}; is0 = issues;
        uint32_t m = qt_issue(0, r5, 1, x);
        check(m == 1u && qt_issue_commit() == 1u && issues == is0 + 1 && pending, "no miss: the second commit takes nothing");
        memset(o2, 0, sizeof o2);
        check(qt_take(m, v4, 1, o2) && !pending, "the take collects the hit group");
        /* a row without hits: the misses' group alone */
        int r12[2] = {1, 2}, k1 = 1; is0 = issues;
        m = qt_issue(0, r12, 2, x);
        int k0 = 0;
        check(m == 0 && issues == is0 && pstage_rows(0, &k0, 1, 1, 0) && pstage_rows(0, &k1, 1, 2, 0), "no hit: nothing launched, two staged");
        m = qt_issue_commit();
        check(m == 3u && issues == is0 + 1 && !G_gm.ht[0], "one group, no hit group taken first");
        float want[D], row[D]; memset(want, 0, sizeof want); memset(o2, 0, sizeof o2);
        for (int k = 0; k < 2; k++) { cpu_row(row, x, r12[k]); for (int d = 0; d < D; d++) want[d] += v4[k] * row[d]; }
        check(qt_take(m, v4, 2, o2) && !memcmp(o2, want, sizeof want), "taken as mode 1 would");

        /* failures: a refused misses' group keeps the hit, the misses go to the CPU */
        counters(&gm, &fb);
        int route[4] = {5, 1, 2, 3}; is0 = issues;
        m = qt_issue(0, route, 4, x);
        k1 = 1; check(mstage(0, k1, 1), "staged");
        issue_refuse = 1; m = qt_issue_commit(); issue_refuse = 0;
        check(m == 1u && G_gm.ht[0] && !G_gm.mp[0] && G.is_cnt[0] == 1, "refused: the hit stays (taken into hy), the miss is the CPU's");
        uint64_t gm2, fb2; counters(&gm2, &fb2);
        check(gm2 == gm && fb2 == fb + 1, "counted as fallback");
        memset(o2, 0, sizeof o2); memset(want, 0, sizeof want);
        cpu_row(row, x, 5); for (int d = 0; d < D; d++) want[d] += v4[0] * row[d];
        int syncs = fake_pipe_syncs;
        check(qt_take(m, v4, 4, o2) && !memcmp(o2, want, sizeof want) && fake_pipe_syncs == syncs, "the take adds the hit row; no drain needed");
        /* a refused hit group: its rank goes to the CPU, the misses still run */
        issue_refuse = 1; m = qt_issue(0, route, 4, x); issue_refuse = 0;
        check(m == 0 && G_gm.c1[0] == 0, "hit group refused at routing");
        check(mstage(0, k1, 1) && qt_issue_commit() == 2u && !G_gm.ht[0] && G_gm.mp[0], "the miss runs as a group taken the plain way");
        memset(o2, 0, sizeof o2);
        check(qt_take(2u, v4, 4, o2), "taken");
        /* a hit take that fails at the second commit: the take reports it */
        m = qt_issue(0, route, 4, x);
        check(mstage(0, k1, 1), "staged");
        take_fail = 1; m = qt_issue_commit();
        check(G_gm.bad && m == 1u, "the hit take failed: the miss goes back to the CPU");
        memset(o2, 0, sizeof o2);
        check(!qt_take(m, v4, 4, o2) && !G_gm.bad, "the take fails, and the next group starts clean");
        qt_shutdown();
        check(!G_gm.hy[0] && fake_host_frees == fake_host_allocs, "shutdown frees the hit-row buffer and the staging");
    }

    printf("11. QT_GPU_MISS=2: misses straight from pinned records\n");
    {
        float v4[4] = {0.4f, 0.3f, 0.2f, 0.1f}, o1[D], o2[D];
        int at_route;
        check(qt_init_mxfp4(NL, NE, D, IH, CAP, TOPK), "tier up (no layout)");
        note(0, 5); qt_fill_wait();
        int ao = fake_async_overwrites, cp = fake_copies;
        uint8_t *slot1 = G_gm.h[0] + G_gm.stride;
        row_run(2, 1, x, v4, o1, &at_route);   /* records pinned, the arena layout: matrix by matrix */
        int zero = 1; for (size_t i = 0; i < G_gm.wb; i++) zero &= slot1[i] == 0;
        check(G_gm.direct == 2 && G_gm.one == 0 && fake_async_overwrites == ao + 6 && fake_copies == cp && zero,
              "no layout: three async copies an expert from the record itself, staging untouched");
        qt_shutdown();

        size_t off[6]; memcpy(off, ROFF, sizeof off);
        qt_miss_src_layout(off, REC);
        check(qt_init_mxfp4(NL, NE, D, IH, CAP, TOPK) && G_gm.lay && G_gm.span == REC && G_gm.ow[1] == ROFF[1] && G_gm.os[2] == ROFF[5],
              "the record layout fits the arena slot: transient slots take it");
        ColiCudaTensor *tu0 = G_gm.tu[0][0];
        check(tu0->dw == (void *)(ar_ptr(0, G_gm.ar[0][0] - 1) + ROFF[1]) && tu0->ds == (void *)(ar_ptr(0, G_gm.ar[0][0] - 1) + ROFF[4]) &&
              tu0->dwb == MB && tu0->dsb == NGU, "slot 0's up matrix views the record's offsets at their exact sizes");
        note(0, 5); qt_fill_wait();
        ao = fake_async_overwrites; cp = fake_copies;
        uint32_t m = row_run(2, 1, x, v4, o2, &at_route);
        check(m == 7u && G_gm.direct == 2 && G_gm.one == 2 && fake_copies == cp + 2 && fake_async_overwrites == ao,
              "one copy an expert, nothing staged");
        check(!memcmp(o1, o2, sizeof o1), "the same row as the matrix-by-matrix copies");
        /* plain arrays flagged pinned: not the layout, so matrix by matrix from them */
        ao = fake_async_overwrites;
        int route[4] = {5, 1, 2, 3}, k1 = 1;
        m = qt_issue(0, route, 4, x);
        check(qt_miss_stage_src(0, &k1, 1, 1, wts[1], wts[1] + MB, wts[1] + 2 * MB, (const float *)e8[1],
                                (const float *)(e8[1] + NGU), (const float *)(e8[1] + 2 * NGU), 1)
              && fake_async_overwrites == ao + 3 && G_gm.one == 2 && G_gm.direct == 3, "off-layout pinned source: three copies");
        m = qt_issue_commit(); memset(o2, 0, sizeof o2);
        check(m == 3u && qt_take(m, v4, 4, o2), "taken");
        /* an older DLL without the single copy: matrix by matrix from then on */
        fake_copy_absent = 1; ao = fake_async_overwrites;
        row_run(2, 1, x, v4, o2, &at_route);
        check(!G_gm.lay && fake_async_overwrites == ao + 6 && !memcmp(o1, o2, sizeof o1), "no copy_async: three copies an expert, same row");
        fake_copy_absent = 0;
        qt_shutdown();

        /* mode 1 ignores the pinned flag (staging, as before) */
        setenv("QT_GPU_MISS", "1", 1);
        check(qt_init_mxfp4(NL, NE, D, IH, CAP, TOPK) && !G_gm.lay, "mode 1: arena layout even with qt_miss_src_layout");
        note(0, 5); qt_fill_wait();
        row_run(1, 1, x, v4, o2, &at_route);
        check(G_gm.direct == 0 && !memcmp(G_gm.h[0], wts[1], 3 * MB) && !memcmp(o1, o2, sizeof o1),
              "mode 1: the pinned record staged like any source, same row");
        check(!qt_pin_host(rec[1], REC) && fake_host_regs == 0, "mode 1 pins nothing");
        qt_shutdown();
        size_t zoff[6] = {0}; qt_miss_src_layout(zoff, 0);
    }

    printf("12. QT_GPU_MISS=2: pinning within QT_GPU_MISS_PIN_GB\n");
    {
        setenv("QT_GPU_MISS", "2", 1);
        check(qt_init_mxfp4(NL, NE, D, IH, CAP, TOPK) && G_pin.cap == SIZE_MAX, "no cap set: the engine's RAM plan bounds it");
        check(qt_pin_host(rec[0], REC) && qt_pin_host(rec[1], REC) && fake_host_regs == 2 && G_pin.n == 2 && G_pin.bytes == 2 * REC, "two slots pinned");
        qt_unpin_host(rec[1], REC);
        check(fake_host_unregs == 1 && G_pin.n == 1 && G_pin.bytes == REC, "unpinned before a free");
        qt_unpin_host(rec[0], REC);
        qt_shutdown();
        char gb[64]; snprintf(gb, sizeof gb, "%.15f", (double)(REC + REC / 2) / 1073741824.0);
        setenv("QT_GPU_MISS_PIN_GB", gb, 1);
        check(qt_init_mxfp4(NL, NE, D, IH, CAP, TOPK), "cap of one and a half slots");
        check(qt_pin_host(rec[0], REC) && !qt_pin_host(rec[1], REC) && fake_host_regs == 3, "the second slot is past the cap: not pinned");
        qt_unpin_host(rec[0], REC);
        qt_shutdown();
        setenv("QT_GPU_MISS_PIN_GB", "0", 1);
        check(qt_init_mxfp4(NL, NE, D, IH, CAP, TOPK) && !qt_pin_host(rec[0], REC) && fake_host_regs == 3, "QT_GPU_MISS_PIN_GB=0: none");
        qt_shutdown();
        unsetenv("QT_GPU_MISS_PIN_GB");
        fake_host_reg_fail = 1;
        check(qt_init_mxfp4(NL, NE, D, IH, CAP, TOPK) && !qt_pin_host(rec[0], REC) && G_pin.fail && G_pin.bytes == 0, "a refused registration");
        fake_host_reg_fail = 0;
        check(!qt_pin_host(rec[1], REC) && fake_host_regs == 3, "is not retried");
        qt_shutdown();
    }

    printf("13. QT_GPU_MISS=2: two rows (a verify)\n");
    {
        int rr[8] = {5, 1, 2, 3, 1, 5, 6, 2}, a1[2] = {1, 4}, a2[2] = {2, 7}, a6 = 6;
        float x2[2 * D], v2[2][4] = {{0.4f, 0.3f, 0.2f, 0.1f}, {0.35f, 0.3f, 0.2f, 0.15f}}, o[2][2][D];
        for (int i = 0; i < D; i++) { x2[i] = x[i]; x2[D + i] = (float)((i * 11) % 13) / 7.f - 0.9f; }
        for (int mode = 1; mode <= 2; mode++) {
            setenv("QT_GPU_MISS", mode == 2 ? "2" : "1", 1);
            check(qt_init_mxfp4(NL, NE, D, IH, CAP, TOPK) && qt_gpu_miss_on(8) == mode, "tier up");
            note(0, 5); qt_fill_wait();
            uint32_t hm[2], fm[2]; int is0 = issues;
            qt_issue_begin_rows(0, rr, 2, 4, x2, hm);
            if (mode == 2) { qt_issue_commit_rows(hm); check(issues == is0 + 1 && hm[0] == 1u && hm[1] == 2u, "mode 2: the shared hit entry launched at routing"); }
            check(mstage_rows(0, a1, 2, 1) && mstage_rows(0, a2, 2, 2) && mstage_rows(0, &a6, 1, 6), "1, 2 and 6 staged");
            memcpy(fm, hm, sizeof fm);
            qt_issue_commit_rows(fm);
            check(fm[0] == 7u && fm[1] == 15u && issues == is0 + mode, mode == 2 ? "a second group of the three misses" : "one group");
            if (mode == 2) check(last_count == 3 && last_rows[0] == 2 && last_rows[2] == 1, "rows[] per missed expert: 2, 2, 1");
            float y[D];
            for (int r = 0; r < 2; r++) {
                memset(o[mode - 1][r], 0, sizeof o[0][0]);
                if (r == 0) { cpu_row(y, x2, 3); for (int d = 0; d < D; d++) o[mode - 1][0][d] += v2[0][3] * y[d]; }
                check(qt_take_row(r, r == 1, v2[r], o[mode - 1][r]), "row taken");
            }
            qt_shutdown();
        }
        check(!memcmp(o[0], o[1], sizeof o[0]), "both rows bit-identical to mode 1's");
    }

    printf("14. QT_GPU_MISS=2 + QT_GPU_MISS_PROMOTE: a promotion reads the record layout\n");
    {
        size_t off[6]; memcpy(off, ROFF, sizeof off);
        qt_miss_src_layout(off, REC);
        setenv("QT_GPU_MISS", "2", 1); setenv("QT_GPU_MISS_PROMOTE", "1", 1);
        snprintf(gb, sizeof gb, "%.15f", (double)((TOPK + 1) * exp_b + exp_b / 2) / 1073741824.0);
        setenv("CUDA_EXPERT_GB", gb, 1);
        check(qt_init_mxfp4(NL, NE, D, IH, CAP, TOPK) && G_gm.lay && G_gm.promote, "tier up, record layout, promote");
        note(0, 5); qt_fill_wait();
        QSlot *v = qs(0, 5), *s = qs(0, 1);
        ColiCudaTensor *vt[3] = {v->tg, v->tu, v->td};
        int route[4] = {1, 2, 3, 4}, k0 = 0;
        float v4[4] = {0.4f, 0.3f, 0.2f, 0.1f}, out[D];
        uint32_t m = qt_issue(0, route, 4, x);
        check(m == 0 && pstage_rows(0, &k0, 1, 1, 1) && G_gm.one == 1, "expert 1 staged from its record in one copy");
        qt_note_n(0, 1, 10, rec[1] + ROFF[0], rec[1] + ROFF[1], rec[1] + ROFF[2], (const float *)(rec[1] + ROFF[3]),
                  (const float *)(rec[1] + ROFF[4]), (const float *)(rec[1] + ROFF[5]));
        check(s->queued && !v->resident, "the swap is taken over");
        m = qt_issue_commit();
        const uint8_t *src = ar_ptr(0, G_gm.ar[0][0] - 1);
        check(m == 1u && fake_async_last_t == vt[2] && fake_async_last_w == src + ROFF[2] && fake_async_last_sc == (const float *)(src + ROFF[5]),
              "the victim's down matrix copied from the transient slot's record offsets");
        check(s->resident && s->td == vt[2], "expert 1 resident");
        memset(out, 0, sizeof out);
        check(qt_take(m, v4, 4, out), "taken");
        qt_shutdown();
        unsetenv("QT_GPU_MISS_PROMOTE"); unsetenv("CUDA_EXPERT_GB");
        off[3] = 8;   /* a scale table off the 16-byte grid */
        qt_miss_src_layout(off, REC);
        check(qt_init_mxfp4(NL, NE, D, IH, CAP, TOPK) && !G_gm.lay && G_gm.os[0] == G_ar.o_s[0], "a misaligned part: the arena layout");
        qt_shutdown();
        off[3] = 0; qt_miss_src_layout(off, G_ar.slot + 1);
        check(qt_init_mxfp4(NL, NE, D, IH, CAP, TOPK) && !G_gm.lay, "a record larger than the arena slot: the arena layout");
        qt_shutdown();
        size_t zoff[6] = {0}; qt_miss_src_layout(zoff, 0);
    }

    printf("15. QT_GPU_MISS=2 + QT_GPU_MISS_FOLD: the miss group behind the hit group, one take\n");
    {
        float v4[4] = {0.4f, 0.3f, 0.2f, 0.1f}, off_[D], on_[D], o2[D];
        int at_route, route[4] = {5, 1, 2, 3}, k1 = 1;
        setenv("QT_GPU_MISS", "2", 1); unsetenv("QT_GPU_MISS_FOLD");
        check(qt_init_mxfp4(NL, NE, D, IH, CAP, TOPK) && !G_gm.fold, "switch unset: off");
        note(0, 5); qt_fill_wait();
        int t0 = takes, is0 = issues;
        uint32_t m_off = row_run(2, 0, x, v4, off_, &at_route);
        check(m_off == 7u && takes - t0 == 2 && issues - is0 == 2 && appends == 0, "off: the hit group taken, then the miss group (two takes)");
        qt_shutdown();
        setenv("QT_GPU_MISS_FOLD", "1", 1);
        check(qt_init_mxfp4(NL, NE, D, IH, CAP, TOPK) && !G_gm.fold, "a backend without coli_cuda_expert_group_append: off");
        qt_shutdown();
        fake_append_hook = fixture_append;
        check(qt_init_mxfp4(NL, NE, D, IH, CAP, TOPK) && G_gm.fold, "switch on, entry point there: on");
        note(0, 5); qt_fill_wait();
        t0 = takes; is0 = issues;
        uint32_t m_on = row_run(2, 0, x, v4, on_, &at_route);
        uint64_t promoted, folded; qt_gpu_miss_more(&promoted, &folded);
        check(at_route == 1 && issues - is0 == 1 && appends == 1 && takes - t0 == 1 && folded == 1,
              "the hit launched at routing, the two misses appended behind it, one take for both");
        check(m_on == m_off && !memcmp(on_, off_, sizeof on_), "the row is bit-identical to the switch-off row");
        uint64_t gm, fb; counters(&gm, &fb);
        check(gm == 2 && fb == 0 && !G_gm.launched && !G_gm.ht[0] && !G_gm.mp[0] && G_gm.used[0] == 0, "counted as GPU misses; the take closes everything");
        /* a refused append (no room in the backend's buffers): the old way, same bits */
        append_refuse = 1; t0 = takes; is0 = issues;
        uint32_t m_r = row_run(2, 0, x, v4, o2, &at_route);
        append_refuse = 0;
        qt_gpu_miss_more(&promoted, &folded);
        check(m_r == m_off && takes - t0 == 2 && issues - is0 == 2 && folded == 1 && !memcmp(o2, off_, sizeof o2),
              "append refused: the hit group is taken first as before, same row");
        /* no hit on the device: nothing in flight to append to, the plain issue */
        int r12[2] = {1, 2}, k0 = 0; is0 = issues; int ap0 = appends;
        uint32_t m = qt_issue(0, r12, 2, x);
        check(m == 0 && mstage(0, k0, 1) && mstage(0, k1, 2) && qt_issue_commit() == 3u && issues == is0 + 1 && appends == ap0,
              "no hit: the misses' group is issued, not appended");
        memset(o2, 0, sizeof o2);
        check(qt_take(3u, v4, 2, o2), "taken");
        /* a take that fails after an append: the layer fails as any failed take */
        m = qt_issue(0, route, 4, x);
        check(mstage(0, k1, 1) && qt_issue_commit() == 3u && appends == ap0 + 1, "appended");
        take_fail = 1; memset(o2, 0, sizeof o2);
        check(!qt_take(3u, v4, 4, o2), "the one take failed: reported");
        m = qt_issue(0, route, 4, x); memset(o2, 0, sizeof o2);
        check(m == 1u && qt_issue_commit() == 1u && qt_take(1u, v4, 4, o2), "and the next group starts clean");
        qt_shutdown();
        fake_append_hook = NULL; unsetenv("QT_GPU_MISS_FOLD");
    }
    unsetenv("QT_GPU_MISS");

    if (fails) { printf("test_qwen36_tier_gpu_miss: %d failure(s)\n", fails); return 1; }
    printf("test_qwen36_tier_gpu_miss: ok\n");
    return 0;
}
