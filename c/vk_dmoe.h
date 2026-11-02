/* vk_dmoe.h -- a layer's routed experts inside an engine's dense chain, with no host step
 * (vk_chain.h: vkc_route, vkc_moe; vk_tier.h: vkt_dev_table, vkt_dev_count). Included by
 * the engines' *_chain.h in a COLI_VULKAN build.
 *
 * A chain normally ends its frame at every MoE layer: the router's logits go down, the
 * host routes and joins the tier's device batch with the CPU's experts, the routed sum
 * comes back up. For a step of up to VKC_DMOE_ROWS rows (decode, a short verify) and a
 * layer whose experts the tier holds all of on the chain's device, the frame instead
 * routes (vkc_route) and computes the experts (vkc_moe) itself and goes on to the next
 * layer: one frame for the whole forward when every layer qualifies, which the tier's
 * warm start makes the usual case on a device that holds the model. An engine that sets
 * big gets a prompt chunk's rows the same way, each layer whose experts are block formats
 * (a GGUF's) through vkc_moe_rows' grouped GEMMs; a layer whose formats have none
 * takes the host's step in that chunk, as every layer did before.
 *
 * The engine keeps what is its own: its router's parameters (VkcRoute), its expert's
 * activation (VkcMoe), the conditions under which its host routing has state the device
 * does not keep (cache routing, a momentum PILOT reads, ROUTE_TRACE), its K/V rows, and
 * its books after the frame, from the ids that come down with it (vkc_dmoe_ids):
 * vkt_dev_count for the tier, and whatever its router counts (the history, HITS).
 *
 *   VkcDmoe d = {0};
 *   if (vkc_dmoe_prepare(&d, lo, L, E, K, D, I, rows)) ...      once per step
 *   vkc_dmoe_upload(&d)                                          in every frame the step begins
 *   if (vkc_dmoe_on(&d, i)) vkc_dmoe_layer(&d, i, n, lg, bias, h2, &route, &moe)
 *                                                                then d.y holds the routed sum
 *   after the frame: vkc_dmoe_ids(&d, i) for each such layer
 *
 * The table holds device addresses of the tier's experts. A layer the host steps through
 * (it holds only part of its experts) runs the tier, which may evict and free experts and
 * upload others; a frame that read the table from before would read freed memory (on a
 * software device, the process's own: a crash, wrong tokens, a stalled frame). So every
 * frame the step begins, the first one and each after a host layer, calls
 * vkc_dmoe_upload, which reads the table again when the tier's residents changed
 * (vkt_epoch), and vkc_dmoe_on turns a layer down while its table is not the current one.
 *
 * COLI_VK_CHAIN_MOE=0 turns it off for every engine. */
#ifndef COLI_VK_DMOE_H
#define COLI_VK_DMOE_H
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <math.h>
#include "vk_chain.h"
#include "vk_tier.h"
#include "backend_vulkan.h"
#include "ggml_quants.h"

#define VKC_DMOE_ROWS 8

/* Hybrid steps (COLI_VK_HYBRID, on where the device has timeline semaphores): in a step of
 * up to VKC_DMOE_ROWS rows, the experts of the tier's host pool (system RAM the device
 * would read over PCIe, about 10 GB/s on an RTX PRO 4000 at PCIe 4.0 x8) are computed by
 * the CPU in place, at its memory's speed, while the device computes the ones it holds:
 *   frame A  ... the router, the route, the input rows copied down; signals the device's
 *            timeline (vkc_hybrid_submit)
 *   frame B  the device's experts (a table where the host pool's are zero)
 *   frame C  waits for the host's timeline, adds the CPU's sum, goes on
 * The engine's own thread computes the pairs, one layer behind its recording: the step's
 * job is kept (vkc_hyb_post) and run (vkc_hyb_flush: wait for frame A, the pairs as the
 * CPU computes a GGUF's matmuls — ggml_quants.h: int8 activations, integer dots — their
 * weighted sum, the host's signal) when the next layer's step begins, when a table is
 * read again, before any wait for a frame (vkc_hybrid_set_flush: every frame from C on
 * waits for the host's value), and at the end. The thread has its OpenMP team warm and
 * waits for the device otherwise; frame B covers the pairs when they take about as long
 * as the device's experts. A worker thread with a team of its own took the steps before:
 * beside the engine's spinning team (OMP_WAIT_POLICY=ACTIVE) it lost its cores — 789 us a
 * step with 12 threads, 328 with 2, the same pairs alone 50 (a Threadripper 3945WX, Qwen3.6
 * int4, 1.4 pairs a step) — and the handoff (a condition variable, the driver's semaphore
 * wake) came on top. A prompt chunk keeps the full table. */
#include "vk_hyb_gate.h"
struct VkcDmoe_;
typedef struct {
    struct VkcDmoe_ *d; int i, n; uint64_t g;
    int act; float limit;
    int flags;                         /* the step inside its frame (flags), or frames A/B/C (timelines) */
    unsigned long long serial;         /* flags: the frame the step is in (the host submits it before it waits) */
} VkcHybJob;
typedef struct {
    VkcHybJob job[2]; int nq, err;     /* the steps kept, oldest first: one, or two while the next step is recorded */
    unsigned long long steps, pairs;   /* steps done, host pool pairs computed */
    double cpu_ms, wait_ms;            /* the pairs' compute; waiting for the device's frame A */
    float *gu, *hh, *dn, *ys; int cap; /* scratch: gate and up, the hidden rows, down, the rows' sums, for cap pairs */
    int ycap;
} VkcHyb;
/* one for every engine's steps on the device: the host's timeline rises in their order */
static VkcHyb g_vkc_hyb;
static uint64_t g_vkc_hyb_seq;
static double vkc_hyb_now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }

typedef struct VkcDmoe_ {
    int lo, L, E, K;               /* the chain's layers lo..lo+L-1, E experts each, top K */
    uint8_t *on;                   /* per layer: the tier holds all of its experts here */
    uint32_t *tabh;                /* their entries, as the tier gave them at `epoch` */
    unsigned epoch; int read, dirty, off;
    VkcBuf *tab, *rid, *rwt, *hid, *y;   /* the table, ids [L][cap][K] (down), weights, hidden rows, the routed sum */
    unsigned long long steps;      /* layer steps that ran so */
    /* big: the engine takes a step of more than VKC_DMOE_ROWS rows too (a prompt chunk),
     * its layers through vkc_moe_rows when their experts are block formats; cap the rows
     * the buffers hold a layer (VKC_DMOE_ROWS, or the step's) */
    int big, cap;
    int big_step;                  /* this step's rows are more than VKC_DMOE_ROWS */
    uint8_t *rows_off;             /* per layer, in a big step: its formats have no grouped GEMM (the host's step) */
    VkcMoeRows rows;
    unsigned long long big_steps;  /* layer steps of more than VKC_DMOE_ROWS rows */
    /* hybrid steps: the table where the host pool's experts are zero (tabhh, tabhyb on the
     * device), their rows for the CPU (hx [L][E]), the input rows copied down (hxx), the
     * CPU's sum (cy), its worker, and the timeline's last value */
    int hybrid;
    uint32_t *tabhh; VktHostExpert *hx; VkcBuf *tabhyb, *hxx, *cy;
    int ev;                        /* the steps on events, inside their frame (vkc_dmoe_flags at the first read) */
    int *hx_n;                     /* per layer: its experts the CPU computes (0: the full table's step) */
    int D, I;
    unsigned long long hyb_steps;
    /* the hybrid steps measured against the device's own reads of the host pool
     * (vkc_dmoe_hyb_gate): the decode forwards timed with the steps on and then off, the
     * faster kept, measured again after a while. On an RTX 3070 at PCIe 4.0 x16 with an
     * int4 Qwen3.6 the device read a pool expert faster than the CPU computed one (18.5 ms a
     * token against 33 to 50 with the hybrid), on an RTX PRO 4000 at x8 the hybrid won. */
    VkcHybGate gate;
    /* a prompt chunk's staging of the host pool (vkc_dmoe_stage_read): the table with its
     * host pool entries pointing into stg (tabs; tabsh the host's copy), the copies a layer
     * makes (fr: uvec4s, fro / frn / frmax a layer's first, count and widest), whether a
     * layer's host pool experts all moved (full), stg_off: none from here on */
    int stg_off, stg_on, stg_big, stg_said; VkcBuf *stg, *tabs, *fr; uint64_t stg_addr; size_t stg_cap, stg_peak;
    uint32_t *tabsh, *frh; int *fro, *frn; uint32_t *frmax; uint8_t *full, *hostm; int nfr;
    unsigned long long stg_steps; double stg_bytes;
    /* the layer whose host pool experts stg holds (-1 none): a prompt run layer by layer
     * stages each layer once for all its chunks (qwen38_chain.h, q38c_forward_lm); a table
     * read again or the kept experts' fill drop it */
    int stg_layer; unsigned long long stg_reused;
    /* between prompt chunks: stg holds the host pool's hottest experts of block types
     * (vkc_dmoe_keep): koff [L][E] their places in it (-1 none), frk the copies, nkr of
     * them, kmax the widest, keep_valid whether stg has them, keep_fetch: copy them */
    int64_t *koff; uint32_t *frk; int nkr, nkeep, keep_valid, keep_fetch; uint32_t kmax;
    unsigned long long keep_fills;
    /* a decode step's table patched with the tier's changes (vkc_dmoe_patch): chg the slots
     * they name, wr the entries to write, nwr of them; reads whole, patches, entries patched */
    int *chg, *wr; int nwr;
    unsigned long long reads, patches, patched;
} VkcDmoe;

static int vkc_dmoe_wanted(void) {
    static int on = -1;
    if (on < 0) { const char *e = getenv("COLI_VK_CHAIN_MOE"); on = !(e && *e == '0'); }
    return on;
}
/* COLI_VK_DMOE: an engine's opt-in single-layer arm of this (scaffolding for a
 * measurement: one layer, the decode path) -- the one layer whose routed experts run
 * in the frame. A layer index >= 1 arms layer n; 0, unset or anything else is off, the
 * host's step as before. Distinct from COLI_VK_CHAIN_MOE, which the engines that follow
 * this header default on. */
static int vkc_dmoe_layer_parse(const char *v) {
    int n = v && *v ? atoi(v) : 0;
    return n > 0 ? n : 0;
}
static int vkc_dmoe_layer_env(void) {
    static int n = -1;
    if (n < 0) n = vkc_dmoe_layer_parse(getenv("COLI_VK_DMOE"));
    return n;
}
/* COLI_VK_DMOE=all -- the multi-layer form (the layer-outer loop's arm): every layer of
 * the engine's chain in one prepare (lo..L, vk_dmoe.h's own range support), the per-layer
 * decline staying the engine's (its mask knows the hash layers and the gates). S1's arm
 * holds exactly one layer for the run; this is the lo..L form the reorder drives. */
static int vkc_dmoe_all_env(void) {
    static int on = -1;
    if (on < 0) { const char *e = getenv("COLI_VK_DMOE"); on = e && !strcmp(e, "all"); }
    return on;
}
/* COLI_VK_DMOE_ROWS_MAX=R: the amdgpu boundary as a cap -- at most R of a prompt chunk's
 * rows per submission, vkc_submit(0) + vkc_begin between the groups (the
 * vkc_layer_recorded pattern). A whole chunk's grouped GEMMs are seconds of GPU work in
 * one submission at 2048 rows (K1: ~2-4 s of it hard-recovers on amdgpu; the same load
 * passes split 8 layers a frame; U \u00a72: the armed chunk's 3.75 GiB of reads lost the
 * device), and the compute reads VRAM at ~0.1 s a layer-visit -- so R near 512 keeps a
 * frame in the ~1 s class. 0 (the default): the whole chunk in one submission, the bits
 * and frames of today. */
static int vkc_dmoe_rows_max_parse(const char *v) {
    int r = v && *v ? atoi(v) : 0;
    return r > 0 ? r : 0;
}
static int vkc_dmoe_rows_max(void) {
    static int r = -1;
    if (r < 0) r = vkc_dmoe_rows_max_parse(getenv("COLI_VK_DMOE_ROWS_MAX"));
    return r;
}
static int vkc_dmoe_hyb_gated(void) {
    static int on = -1;
    if (on < 0) { const char *e = getenv("COLI_VK_HYBRID_GATE"); on = !(e && *e == '0'); }
    return on;
}
static int vkc_hybrid_wanted(void) {
    static int on = -1;
    if (on < 0) { const char *e = getenv("COLI_VK_HYBRID"); on = !(e && *e == '0'); }
    return on && vkc_hybrid_ready();
}
/* ---- the hybrid steps' CPU side ---- */
/* the pairs' team: COLI_VK_HYBRID_THREADS, else the engine thread's (OpenMP's) */
static int vkc_hyb_threads(void) {
    static int n = -1;
    if (n < 0) { const char *e = getenv("COLI_VK_HYBRID_THREADS"); n = e && atoi(e) > 0 ? atoi(e) : 0; }
#ifdef _OPENMP
    return n > 0 ? n : omp_get_max_threads();
#else
    return 1;
#endif
}
/* layer i's host pool pairs of the step's n rows, their weighted sum into the CPU's rows */
static void vkc_hyb_compute(VkcDmoe *d, const VkcHybJob *j) {
    int E = d->E, K = d->K, D = d->D, I = d->I, n = j->n;
    const int *ids = (const int *)vkc_ptr(d->rid) + (size_t)j->i * d->cap * K;
    const float *wt = (const float *)vkc_ptr(d->rwt), *x = (const float *)vkc_ptr(d->hxx);
    /* the sum goes to the device's buffer (written by the host, read by the device: memory the
     * CPU does not cache, on a discrete card often its BAR) in one copy at the end; reading
     * it back as the sum grew cost about 70 us a step of Flint's three pairs */
    float *y = (float *)vkc_ptr(d->cy);
    int ps[VKC_DMOE_ROWS * VKC_ROUTE_KMAX], pk[VKC_DMOE_ROWS * VKC_ROUTE_KMAX], np = 0;
    const VktHostExpert *hx = d->hx + (size_t)j->i * E;
    for (int s = 0; s < n; s++)
        for (int k = 0; k < K; k++) { int e = ids[s * K + k]; if (e >= 0 && e < E && hx[e].g) { ps[np] = s; pk[np] = k; np++; } }
    VkcHyb *w = &g_vkc_hyb;
    /* the sum in the host's own memory (cy is the device's: its mapping is uncached for the
     * CPU, and reading it back line by line through the BAR cost more than the pairs) */
    if (w->ycap < n * D) {
        free(w->ys); w->ys = malloc((size_t)n * D * sizeof(float)); w->ycap = w->ys ? n * D : 0;
        if (!w->ycap) { w->err = 1; return; }
    }
    memset(w->ys, 0, (size_t)n * D * sizeof(float));
    if (!np) { memcpy(y, w->ys, (size_t)n * D * sizeof(float)); return; }
    w->pairs += (unsigned long long)np;
    if (w->cap < np) {
        free(w->gu); free(w->hh); free(w->dn);
        w->gu = malloc((size_t)np * 2 * I * sizeof(float)); w->hh = malloc((size_t)np * I * sizeof(float)); w->dn = malloc((size_t)np * D * sizeof(float));
        w->cap = w->gu && w->hh && w->dn ? np : 0;
        if (!w->cap) { w->err = 1; return; }
    }
    float *dn = w->dn;
    const VktHostExpert *hp[VKC_DMOE_ROWS * VKC_ROUTE_KMAX]; const float *xp[VKC_DMOE_ROWS * VKC_ROUTE_KMAX];
    for (int p = 0; p < np; p++) { hp[p] = &hx[ids[ps[p] * K + pk[p]]]; xp[p] = x + (size_t)ps[p] * D; }
    if (!vkt_host_pairs(hp, xp, np, D, I, j->limit, vkc_hyb_threads(), w->gu, w->hh, dn)) w->err = 1;
    for (int p = 0; p < np; p++) {   /* in rank order within a row */
        float wv = wt[ps[p] * K + pk[p]]; float *ys = w->ys + (size_t)ps[p] * D; const float *o = dn + (size_t)p * D;
        for (int c = 0; c < D; c++) ys[c] += wv * o[c];
    }
    memcpy(y, w->ys, (size_t)n * D * sizeof(float));
}
/* the hybrid steps inside their frame, on events (vkc_ev_signal / vkc_ev_wait: the device
 * sets one once the route and the rows are down, waits for the host's after its experts),
 * as the CUDA chain does them with its flags; COLI_VK_HYBRID_EVENTS=0: frames A/B/C and
 * the timelines. Measured on an RTX 3070 (Qwen3.6 int4, 1.3 pool pairs a step): the three
 * frames and two timeline round trips a step left the device idle about 300 us a step,
 * more than the pairs it spared it (33 ms a token against 18 with the device's own reads). */
static int vkc_dmoe_flags(void) {
    static int on = -1;
    if (on < 0) { const char *e = getenv("COLI_VK_HYBRID_EVENTS"); on = !(e && *e == '0'); }
    return on && vkc_ev_ready();
}
/* the host's side of an event step: the device's event, then the pairs, then the host's */
static int vkc_hyb_flag_wait(VkcDmoe *d, uint64_t g) {
    (void)d;
    double t0 = vkc_hyb_now_ms();
    for (;;) {
        int r = vkc_ev_host_done(g);
        if (r > 0) return 1;
        if (r < 0 || vkc_hyb_now_ms() - t0 > 20000.0) return 0;   /* lost, or the device never got there */
    }
}
static void vkc_hyb_flag_signal(VkcDmoe *d, uint64_t g) {
    (void)d;
    __atomic_thread_fence(__ATOMIC_SEQ_CST);   /* cy before the event (the mapping may be write-combined) */
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_sfence();
#endif
    if (!vkc_ev_host_set(g)) g_vkc_hyb.err = 1;
}
/* the oldest kept step: frame A done (the device's event set), the pairs, the host's signal
 * (always: frame C, or the wait in the frame, waits for it) */
static int g_vkc_hyb_in;   /* a flush under way: the frame it submits waits for the ring's oldest frame (vkc_begin),
                              and that wait's flush (vkc_hybrid_set_flush) must not take the next step before this
                              one's signal -- the device would wait for this step's event while the host polled the
                              next step's, which comes after it. The frames the ring waits for are older than any
                              step kept, so nothing is lost by leaving the queue to the flush under way. */
static void vkc_hyb_flush_oldest(void) {
    VkcHyb *w = &g_vkc_hyb;
    if (!w->nq || g_vkc_hyb_in) return;
    g_vkc_hyb_in = 1;
    VkcHybJob j = w->job[0];
    w->job[0] = w->job[1]; w->nq--;
    if (j.flags && j.serial > vkc_serial()) {   /* its frame is still being recorded: off it goes */
        if (!(vkc_submit(0) && vkc_begin())) { w->err = 1; g_vkc_hyb_in = 0; return; }
    }
    double t0 = vkc_hyb_now_ms();
    int got = j.flags ? vkc_hyb_flag_wait(j.d, j.g) : vkc_hybrid_host_wait(j.g);
    if (!got) w->err = 1;
    else {
        double t1 = vkc_hyb_now_ms();
        w->wait_ms += t1 - t0;
        vkc_hyb_compute(j.d, &j);
        w->cpu_ms += vkc_hyb_now_ms() - t1;
    }
    if (j.flags) vkc_hyb_flag_signal(j.d, j.g); else vkc_hybrid_host_signal(j.g);
    w->steps++;
    g_vkc_hyb_in = 0;
}
/* every step kept so far (before the host waits for a frame, the table changes, the end) */
static void vkc_hyb_flush(void) { while (g_vkc_hyb.nq && !g_vkc_hyb_in) vkc_hyb_flush_oldest(); }
/* The steps a layer about to be recorded waits for. An event step is run once the next
 * layer's step is recorded (vkc_dmoe_layer, after it: its frame then holds both, and goes
 * off before the host polls), so the device has the next frame while this one's experts
 * run and the host records while the device computes; run here only when it is two layers
 * old (a layer between took no step). A timeline step runs here: frame A is a frame of its
 * own, and the recording since was its frame B and C's start. */
static void vkc_hyb_flush_before(int layer) {
    VkcHyb *w = &g_vkc_hyb;
    while (w->nq && !g_vkc_hyb_in && (!w->job[0].flags || w->job[0].i < layer - 1)) vkc_hyb_flush_oldest();
}
static void vkc_hyb_drain(void) { vkc_hyb_flush(); }
static int vkc_hyb_start(void) { vkc_hybrid_set_flush(vkc_hyb_flush); return 1; }
static void vkc_hyb_stop(void) { vkc_hyb_flush(); }
/* a step's job, kept until its flush; two at most */
static int vkc_hyb_post(const VkcHybJob *j) {
    VkcHyb *w = &g_vkc_hyb;
    if (w->nq == 2) vkc_hyb_flush_oldest();
    w->job[w->nq++] = *j;
    return 1;
}


/* The tier's table for layers lo..lo+L-1, read again when its residents changed since. */
static int vkc_dmoe_stage_read(VkcDmoe *d);   /* below: the prompt chunks' staging of the host pool */
static void vkc_dmoe_keep(VkcDmoe *d);         /* and what stg holds between them */
static void vkc_keep_entry(VkcDmoe *d, size_t j);
/* A decode step's table after a decode step's (not a prompt chunk's staging): the entries
 * the tier changed since it was read (vkt_dev_changes), each read again, the hybrid steps'
 * and the kept experts' with it; the frame writes those (vkc_dmoe_upload). The tier moves a
 * few experts a token (vk_tier.c's promotions), and reading every layer's table whole for
 * them cost a fifth of the decode (Qwen3.8 IQ2_XS on an RTX PRO 4000, a code prompt: 37.5 to
 * 40.3 against 46.4 to 48.0 tok/s). 0 = read it whole: a layer that loses an expert or gains
 * its last, an error, or more changes than the tier logged. */
static int vkc_dmoe_patch(VkcDmoe *d) {
    if (d->big_step || d->stg_big || d->stg_on) return 0;
    int L = d->L, E = d->E;
    const size_t ent = COLI_VK_XB_ENTRY;
    if (!d->chg) d->chg = malloc((size_t)L * E * sizeof(int));
    if (!d->wr) d->wr = malloc((size_t)L * E * sizeof(int));
    if (!d->chg || !d->wr) return 0;
    int n = vkt_dev_changes(d->epoch, d->chg, L * E);
    if (n < 0) return 0;
    vkc_hyb_drain();   /* the worker reads the host pool's rows of the table it had */
    for (int k = 0; k < n; k++) {
        int l = d->chg[k] / E, e = d->chg[k] % E;
        if (l < d->lo || l >= d->lo + L) continue;
        int i = l - d->lo;
        if (!d->on[i]) return 0;   /* a layer the host steps through: whole when it may gain its last */
        size_t j = (size_t)i * E + e;
        int had = d->hybrid && d->hx && d->hx[j].g;
        uint8_t host = 0;
        int r = vkt_dev_entry(l, e, d->tabh + j * ent, d->hybrid ? d->tabhh + j * ent : NULL, d->hybrid ? d->hx + j : NULL, &host);
        if (r <= 0) return 0;
        if (d->hybrid && d->hx_n) d->hx_n[i] += (d->hx[j].g != NULL) - had;
        if (d->hostm) d->hostm[j] = host;
        if (d->keep_valid) vkc_keep_entry(d, j);
        if (d->nwr < L * E) d->wr[d->nwr++] = (int)j;
    }
    d->patches++; d->patched += (unsigned long long)n;
    d->epoch = vkt_epoch();
    return 1;
}
static int vkc_dmoe_read(VkcDmoe *d) {
    if (d->read && d->epoch == vkt_epoch()) return 1;
    if (d->read && vkc_dmoe_patch(d)) return 1;
    d->reads++; d->nwr = 0;
    int L = d->L, E = d->E;
    if (!d->on) d->on = calloc((size_t)L, 1);
    if (!d->tabh) d->tabh = malloc((size_t)L * E * COLI_VK_XB_ENTRY * sizeof(uint32_t));
    if (!d->on || !d->tabh) { d->off = 1; return 0; }
    vkc_hyb_drain();   /* the worker reads the host pool's rows of the table it had */
    for (int i = 0; i < L; i++) {
        int n = vkt_dev_table(d->lo + i, d->tabh + (size_t)i * E * COLI_VK_XB_ENTRY);
        if (n < 0) { d->off = 1; return 0; }
        d->on[i] = n == E;
    }
    if (d->hybrid) {
        if (!d->tabhh) d->tabhh = malloc((size_t)L * E * COLI_VK_XB_ENTRY * sizeof(uint32_t));
        if (!d->hx) d->hx = calloc((size_t)L * E, sizeof *d->hx);
        if (!d->hx_n) d->hx_n = calloc((size_t)L, sizeof *d->hx_n);
        if (!d->tabhh || !d->hx || !d->hx_n) d->hybrid = 0;
        for (int i = 0; i < L && d->hybrid; i++) {
            if (vkt_dev_table_hybrid(d->lo + i, d->tabhh + (size_t)i * E * COLI_VK_XB_ENTRY, d->hx + (size_t)i * E) < 0) { d->hybrid = 0; break; }
            d->hx_n[i] = 0;
            for (int e = 0; e < E; e++) d->hx_n[i] += d->hx[(size_t)i * E + e].g != NULL;
        }
    }
    d->stg_on = d->big && d->big_step && vkc_dmoe_stage_read(d);
    d->stg_layer = -1;   /* the staged regions may have moved */
    if (!d->big_step) vkc_dmoe_keep(d);
    d->stg_big = d->big_step;
    d->epoch = vkt_epoch(); d->read = 1; d->dirty = 1;
    return 1;
}
/* layer i's gate/up and down formats, from its first entry (a layer's experts share them) */
/* layer i's format word w (12 gate/up, 15 down): the tier's form for the layer, the same in
 * every resident entry, read from the first resident one (0 when none is) */
static int vkc_dmoe_fmt(const VkcDmoe *d, int i, int w) {
    const uint32_t *t = d->tabh + (size_t)i * d->E * COLI_VK_XB_ENTRY;
    for (int e = 0; e < d->E; e++, t += COLI_VK_XB_ENTRY) if (t[18]) return (int)t[w];
    return 0;
}
/* A prompt chunk routes nearly every expert of a layer, and the grouped GEMMs of the block
 * types drew about 4 GB/s from the bus reading the host pool in place: a chunk of more
 * than VKC_DMOE_ROWS rows copies a layer's host pool experts of block types into stg first
 * (vkc_fetch, at the bus's rate), the table it runs on pointing there (tabs), and its
 * GEMMs read VRAM (Qwen3.8 IQ2_S/Q2_0 on an RTX PRO 4000, 19 GiB in the host pool: a
 * 507-token prompt 116-119 -> 139-149 tok/s). Our formats' GEMMs read the pool near the
 * bus's rate already, and the copy before them only cost (Qwen3.6 int4, 9 GiB in the pool:
 * 513 -> 394 tok/s): they keep reading it. stg holds the most a layer has, within the room
 * the tier's budget left (vkt_stage_room; past it an entry reads the pool): a barrier
 * before each copy, the layer before done with it. COLI_VK_STAGE=0 off. */
static int vkc_stage_wanted(void) {
    static int on = -1;
    if (on < 0) { const char *e = getenv("COLI_VK_STAGE"); on = !(e && *e == '0'); }
    return on;
}
/* the bytes of a matrix's rows for an entry's format word fmt (row words rw) at `rows`
 * rows: the block types' (their scales are in the blocks, their tables shared); 0 for
 * the others, which keep reading the pool */
static int vkc_stage_sizes(uint32_t fmt, uint32_t rw, uint32_t gs, int rows, int n, size_t *wb, size_t *sb) {
    (void)gs; (void)n;
    *wb = (size_t)rows * rw * 4; *sb = 0;
    return fmt >= 100 && fmt < 164;
}
static size_t vkc_stage_up(size_t b) { return (b + 255) & ~(size_t)255; }
static int vkc_dmoe_stage_read(VkcDmoe *d) {
    int L = d->L, E = d->E, D = d->D, I = d->I;
    const size_t ent = COLI_VK_XB_ENTRY;
    size_t room = vkt_stage_room();   /* what the tier's budget left for it */
    if (d->stg_off || !vkc_stage_wanted() || !vkt_host_pool_on() || !room || D < 1 || I < 1) return 0;
    if (!d->hostm && !(d->hostm = malloc((size_t)L * E))) return 0;
    if (!d->tabsh) d->tabsh = malloc((size_t)L * E * ent * sizeof(uint32_t));
    if (!d->fro) { d->fro = calloc((size_t)L, sizeof(int)); d->frn = calloc((size_t)L, sizeof(int));
                   d->frmax = calloc((size_t)L, sizeof(uint32_t)); d->full = calloc((size_t)L, 1); }
    if (!d->tabsh || !d->fro || !d->frn || !d->frmax || !d->full) return 0;
    /* pass 1: each layer's bytes and regions */
    size_t need = 0; int regions = 0;
    for (int i = 0; i < L; i++) {
        if (vkt_dev_host(d->lo + i, d->hostm + (size_t)i * E) < 0) return 0;
        size_t lb = 0;
        for (int e = 0; e < E; e++) {
            const uint32_t *o = d->tabh + ((size_t)i * E + e) * ent;
            size_t w[3], s[3];
            if (!d->hostm[(size_t)i * E + e] || !o[18]) continue;
            if (!vkc_stage_sizes(o[12], o[13], o[14], I, D, &w[0], &s[0]) || !vkc_stage_sizes(o[15], o[16], o[17], D, I, &w[2], &s[2])) continue;
            w[1] = w[0]; s[1] = s[0];
            size_t eb = 0;
            for (int k = 0; k < 3; k++) eb += vkc_stage_up(w[k]) + (s[k] ? vkc_stage_up(s[k]) : 0);
            if (lb + eb > room) continue;   /* past the room: this one reads the pool */
            lb += eb;
            for (int k = 0; k < 3; k++) regions += 1 + (s[k] != 0);
        }
        if (lb > need) need = lb;
    }
    if (!need) return 0;
    if (need > d->stg_cap) {
        vkc_free(d->stg); d->stg = NULL; d->stg_cap = 0;
        /* not into the device's last memory (the driver moves the rest out then) */
        double used = 0, bud = 0;
        if (coli_vk_mem_budget(&used, &bud) && (bud - used) * 1e9 < (double)need + 67108864.0) {
            if (!d->stg_said) fprintf(stderr, "[VK] chain: %.0f MiB free on the device, short of the host pool experts' staging "
                                              "(%.0f MiB); a prompt chunk reads them over PCIe\n", (bud - used) * 1e9 / 1048576.0, need / 1048576.0);
            d->stg_said = 1;
            return 0;
        }
        if (room > need) need = room;   /* the whole room: between chunks it holds the hottest (vkc_dmoe_keep) */
        if (!(d->stg = vkc_buf_addr(need, &d->stg_addr))) {
            fprintf(stderr, "[VK] chain: no VRAM to stage the host pool's experts in (%.0f MiB); a prompt chunk reads them over PCIe\n",
                    need / 1048576.0);
            d->stg_off = 1;
            return 0;
        }
        d->stg_cap = need;
        if (need > d->stg_peak) d->stg_peak = need;
    }
    uint32_t *fr = realloc(d->frh, (size_t)regions * 4 * sizeof(uint32_t));
    if (!fr) return 0;
    d->frh = fr;
    /* pass 2: the entries pointing into stg, the copies */
    int k = 0;
    for (int i = 0; i < L; i++) {
        size_t off = 0; int full = 1;
        d->fro[i] = k; d->frmax[i] = 0;
        for (int e = 0; e < E; e++) {
            size_t j = ((size_t)i * E + e) * ent;
            const uint32_t *o = d->tabh + j;
            uint32_t *t = d->tabsh + j;
            memcpy(t, o, ent * sizeof(uint32_t));
            if (!d->hostm[(size_t)i * E + e] || !o[18]) continue;
            size_t w[3], s[3];
            if (!vkc_stage_sizes(o[12], o[13], o[14], I, D, &w[0], &s[0]) || !vkc_stage_sizes(o[15], o[16], o[17], D, I, &w[2], &s[2])) {
                full = 0; continue;
            }
            w[1] = w[0]; s[1] = s[0];
            size_t eb = 0;
            for (int m = 0; m < 3; m++) eb += vkc_stage_up(w[m]) + (s[m] ? vkc_stage_up(s[m]) : 0);
            if (off + eb > d->stg_cap) { full = 0; continue; }
            for (int m = 0; m < 3; m++) {   /* rows at words 4m, 4m + 1; scales at 4m + 2, 4m + 3 */
                for (int sc = 0; sc < 2; sc++) {
                    size_t bytes = sc ? s[m] : w[m];
                    if (!bytes) continue;
                    uint64_t src = (uint64_t)o[4 * m + 2 * sc] | (uint64_t)o[4 * m + 2 * sc + 1] << 32, dst = d->stg_addr + off;
                    fr[4 * k] = (uint32_t)src; fr[4 * k + 1] = (uint32_t)(src >> 32);
                    fr[4 * k + 2] = (uint32_t)(off / 4); fr[4 * k + 3] = (uint32_t)(bytes / 4);
                    if (bytes / 4 > d->frmax[i]) d->frmax[i] = (uint32_t)(bytes / 4);
                    t[4 * m + 2 * sc] = (uint32_t)dst; t[4 * m + 2 * sc + 1] = (uint32_t)(dst >> 32);
                    off += vkc_stage_up(bytes); k++;
                }
            }
        }
        d->frn[i] = k - d->fro[i]; d->full[i] = full;
    }
    d->nfr = k;
    if (!vkc_reserve(&d->tabs, (size_t)L * E * ent * sizeof(uint32_t), VKC_DEV) ||
        !vkc_reserve(&d->fr, (size_t)L * E * 6 * 4 * sizeof(uint32_t), VKC_DEV)) { d->stg_off = 1; return 0; }   /* the most regions: never grown in a frame */
    return 1;
}
/* Between prompt chunks stg would sit idle: it holds the host pool's hottest experts of
 * block types (the tier's order, vkt_score) as more of the device's, the decode's tables
 * pointing there and the CPU's hybrid share leaving them. So the room the tier's budget
 * gave up for the staging comes back to the decode (Qwen3.8 IQ2_S/Q2_0 on an RTX PRO 4000,
 * a 2000-token prompt and 200 teacher-forced steps: the CPU's host pool pairs 31920 ->
 * 23642). Chosen and copied at the first table read after a chunk staged into stg (or the
 * first ever), applied again at every read until the next chunk. */
static int vkc_keep_cmp(const void *a, const void *b) {
    uint64_t x = ((const uint64_t *)a)[0], y = ((const uint64_t *)b)[0];
    return x < y ? 1 : x > y ? -1 : 0;
}
/* entry j (layer * E + expert) of the tables, a kept expert still in the host pool: stg's */
static void vkc_keep_entry(VkcDmoe *d, size_t j) {
    const size_t ent = COLI_VK_XB_ENTRY;
    int E = d->E, D = d->D, I = d->I;
    if (!d->koff || !d->hostm || d->koff[j] < 0 || !d->hostm[j]) return;
    uint32_t *o = d->tabh + j * ent;
    if (!o[18]) return;
    size_t w[3], s0, off = (size_t)d->koff[j];
    vkc_stage_sizes(o[12], o[13], o[14], I, D, &w[0], &s0);
    vkc_stage_sizes(o[15], o[16], o[17], D, I, &w[2], &s0);
    w[1] = w[0];
    for (int m = 0; m < 3; m++) {
        uint64_t dst = d->stg_addr + off;
        o[4 * m] = (uint32_t)dst; o[4 * m + 1] = (uint32_t)(dst >> 32);
        off += vkc_stage_up(w[m]);
    }
    if (d->hybrid && d->tabhh && d->hx) {   /* the device's in the hybrid steps, not the CPU's */
        memcpy(d->tabhh + j * ent, o, ent * sizeof(uint32_t));
        if (d->hx[j].g) { d->hx_n[j / E]--; memset(&d->hx[j], 0, sizeof d->hx[j]); }
    }
}
static void vkc_dmoe_keep(VkcDmoe *d) {
    int L = d->L, E = d->E, D = d->D, I = d->I;
    const size_t ent = COLI_VK_XB_ENTRY;
    size_t room = vkt_stage_room();
    if (d->stg_off || !vkc_stage_wanted() || !room || D < 1 || I < 1) return;
    if (!d->hostm && !(d->hostm = malloc((size_t)L * E))) return;
    if (!d->stg) {   /* stg before any prompt chunk: the room's size, as a chunk would make it */
        if (!(d->stg = vkc_buf_addr(room, &d->stg_addr))) { d->stg_off = 1; return; }
        d->stg_cap = room;
        if (room > d->stg_peak) d->stg_peak = room;
    }
    if (!vkc_reserve(&d->fr, (size_t)L * E * 6 * 4 * sizeof(uint32_t), VKC_DEV)) { d->stg_off = 1; return; }
    for (int i = 0; i < L; i++) if (vkt_dev_host(d->lo + i, d->hostm + (size_t)i * E) < 0) return;
    if (!d->keep_valid) {
        /* the choice: the host pool's experts of block types, hottest first, while stg holds them */
        uint64_t (*c)[2] = malloc((size_t)L * E * sizeof *c), *sc = malloc((size_t)E * sizeof *sc);
        if (!d->koff) d->koff = malloc((size_t)L * E * sizeof *d->koff);
        if (!d->frk) d->frk = malloc((size_t)L * E * 3 * 4 * sizeof *d->frk);
        if (!c || !sc || !d->koff || !d->frk) { free(c); free(sc); return; }
        size_t n = 0;
        for (int i = 0; i < L; i++) {
            if (vkt_score(d->lo + i, sc) < 0) { free(c); free(sc); return; }
            for (int e = 0; e < E; e++) {
                const uint32_t *o = d->tabh + ((size_t)i * E + e) * ent;
                if (d->hostm[(size_t)i * E + e] && o[18] && o[12] >= 100 && o[12] < 164 && o[15] >= 100 && o[15] < 164) {
                    c[n][0] = sc[e]; c[n][1] = (uint64_t)i * E + e; n++;
                }
            }
        }
        qsort(c, n, sizeof *c, vkc_keep_cmp);
        for (size_t k = 0; k < (size_t)L * E; k++) d->koff[k] = -1;
        size_t off = 0; int r = 0; d->kmax = 0; d->nkeep = 0;
        for (size_t k = 0; k < n; k++) {
            size_t j = (size_t)c[k][1];
            const uint32_t *o = d->tabh + j * ent;
            size_t w[3], s0;
            vkc_stage_sizes(o[12], o[13], o[14], I, D, &w[0], &s0);
            vkc_stage_sizes(o[15], o[16], o[17], D, I, &w[2], &s0);
            w[1] = w[0];
            size_t eb = vkc_stage_up(w[0]) + vkc_stage_up(w[1]) + vkc_stage_up(w[2]);
            if (off + eb > d->stg_cap) continue;
            d->koff[j] = (int64_t)off;
            for (int m = 0; m < 3; m++) {
                uint64_t src = (uint64_t)o[4 * m] | (uint64_t)o[4 * m + 1] << 32;
                d->frk[4 * r] = (uint32_t)src; d->frk[4 * r + 1] = (uint32_t)(src >> 32);
                d->frk[4 * r + 2] = (uint32_t)(off / 4); d->frk[4 * r + 3] = (uint32_t)(w[m] / 4);
                if (w[m] / 4 > d->kmax) d->kmax = (uint32_t)(w[m] / 4);
                off += vkc_stage_up(w[m]); r++;
            }
            d->nkeep++;
        }
        free(c); free(sc);
        d->nkr = r; d->keep_valid = 1; d->keep_fetch = 1;
    }
    /* the tables: a kept expert still in the host pool reads stg (one the tier moved
     * onto the device keeps its own entry) */
    for (size_t j = 0; j < (size_t)L * E; j++) vkc_keep_entry(d, j);
}
/* For a step of `rows` rows on the primary device: whether any of layers lo..lo+L-1 takes
 * it; reads the tier's table again when its residents changed. 0 for good when the
 * device cannot read the tier's experts by address. */
static int vkc_dmoe_prepare(VkcDmoe *d, int lo, int L, int E, int K, int D, int I, int rows) {
    int big = rows > VKC_DMOE_ROWS;
    if (!vkc_dmoe_wanted() || d->off || (big && !d->big) || L < 1 || vkc_device_now() != 0 || !vkt_ready() ||
        !vkc_moe_ready() || !vkc_route_ready() || K < 1 || K > VKC_ROUTE_KMAX || E < K || E > VKC_ROUTE_EMAX) return 0;
    if (d->on && (d->lo != lo || d->L != L || d->E != E || d->K != K)) { d->off = 1; return 0; }
    if (!d->on) d->stg_layer = -1;   /* before the first read: no layer staged */
    d->lo = lo; d->L = L; d->E = E; d->K = K;
    if (d->D != D || d->I != I) d->read = 0;   /* the staging's sizes follow D and I */
    d->D = D; d->I = I;
    /* the staging's table is a prompt's: read again when the steps change kind */
    d->big_step = big;
    if (d->stg_big != big) d->read = 0;
    if (!vkc_dmoe_read(d)) return 0;
    int any = 0;
    for (int i = 0; i < L; i++) any |= d->on[i];
    if (!any) return 0;
    /* a big step: the layers whose formats have the grouped GEMMs run here, the others
     * (a GGUF's IQ1_M layers, say) take the host's step */
    if (big && (D % 64 || I % 64)) return 0;
    if (big) {
        if (!d->rows_off && !(d->rows_off = calloc((size_t)L, 1))) return 0;
        int some = 0;
        for (int i = 0; i < L; i++) {
            d->rows_off[i] = d->on[i] && !vkc_moe_rows_ready(vkc_dmoe_fmt(d, i, 12), vkc_dmoe_fmt(d, i, 15));
            some |= d->on[i] && !d->rows_off[i];
        }
        if (!some) return 0;
    }
    d->big_step = big;
    d->D = D; d->I = I;
    size_t r = VKC_DMOE_ROWS, rc = big ? (size_t)rows : r;
    /* the hybrid steps' table and rows (the weights where the worker reads them) */
    if (!d->tab && !d->hybrid && vkc_hybrid_wanted()) {
        d->hybrid = 1; vkc_hyb_gate_init(&d->gate, vkc_dmoe_hyb_gated());
        if (!vkc_hyb_start() || !(d->tabhyb = vkc_buf((size_t)L * E * COLI_VK_XB_ENTRY * sizeof(uint32_t), VKC_DEV)) ||
            !vkc_reserve(&d->hxx, r * (size_t)D * sizeof(float), VKC_DOWN) || !vkc_reserve(&d->cy, r * (size_t)D * sizeof(float), VKC_UP))
            d->hybrid = 0;
        else { d->read = 0; if (!vkc_dmoe_read(d)) return 0; }
        d->ev = d->hybrid && vkc_dmoe_flags();
    }
    if ((!d->tab && !(d->tab = vkc_buf((size_t)L * E * COLI_VK_XB_ENTRY * sizeof(uint32_t), VKC_DEV))) ||
        !vkc_reserve(&d->rid, (size_t)L * rc * K * sizeof(int), VKC_DOWN) || !vkc_reserve(&d->rwt, rc * K * sizeof(float), d->hybrid ? VKC_DOWN : VKC_DEV) ||
        !vkc_reserve(&d->hid, r * K * ((size_t)I + D) * sizeof(float), VKC_DEV) || !vkc_reserve(&d->y, rc * (size_t)D * sizeof(float), VKC_DEV)) {
        d->off = 1;
        return 0;
    }
    d->cap = (int)rc;
    return 1;
}
/* The table into the device, in a frame just begun: read again first when the tier's
 * residents changed since (a host layer before it ran the tier), written when it changed.
 * 0 only when the write fails; a table that cannot be read turns the layers down. */
static int vkc_dmoe_upload(VkcDmoe *d) {
    if (d->off || !d->tab) return 1;
    vkt_settle();   /* the experts that moved up to the device since the last frame */
    if (!vkc_dmoe_read(d)) return 1;   /* d->off: every layer takes the host's step from here */
    if (!d->dirty) {   /* the entries vkc_dmoe_patch read again */
        const size_t ent = COLI_VK_XB_ENTRY;
        for (int k = 0; k < d->nwr; k++) {
            size_t o = (size_t)d->wr[k] * ent;
            if (!vkc_write(d->tab, o, d->tabh + o, ent * sizeof(uint32_t)) ||
                (d->hybrid && !vkc_write(d->tabhyb, o, d->tabhh + o, ent * sizeof(uint32_t)))) return 0;
        }
        d->nwr = 0;
        return 1;
    }
    d->nwr = 0;
    if (!vkc_write(d->tab, 0, d->tabh, (size_t)d->L * d->E * COLI_VK_XB_ENTRY * sizeof(uint32_t))) return 0;
    if (d->stg_on && (!vkc_write(d->tabs, 0, d->tabsh, (size_t)d->L * d->E * COLI_VK_XB_ENTRY * sizeof(uint32_t)) ||
                      !vkc_write(d->fr, 0, d->frh, (size_t)d->nfr * 4 * sizeof(uint32_t)))) return 0;
    if (d->hybrid && !vkc_write(d->tabhyb, 0, d->tabhh, (size_t)d->L * d->E * COLI_VK_XB_ENTRY * sizeof(uint32_t))) return 0;
    /* the kept experts into stg, before any step reads it (by address: the barriers are ours) */
    if (d->keep_fetch && d->nkr) {
        if (!vkc_write(d->fr, 0, d->frk, (size_t)d->nkr * 4 * sizeof(uint32_t)) || !vkc_barrier() ||
            !vkc_fetch(d->fr, 0, d->nkr, d->kmax, d->stg) || !vkc_barrier()) return 0;
        d->keep_fills++; d->stg_layer = -1;
    }
    d->keep_fetch = 0;
    d->dirty = 0;
    return 1;
}
/* Whether layer `layer` runs its experts in the frame: all of them on the device, in the
 * table the frame has (the tier's residents unchanged since it was read). */
/* 1 when a prompt step's host pool experts stage into VRAM (the tables read for big steps) */
static int vkc_dmoe_staging(const VkcDmoe *d) { return d->on && !d->off && d->stg_on; }
static int vkc_dmoe_on(const VkcDmoe *d, int layer) {
    return d->on && !d->off && d->epoch == vkt_epoch() && layer >= d->lo && layer < d->lo + d->L && d->on[layer - d->lo] &&
           !(d->big_step && d->rows_off && d->rows_off[layer - d->lo]);
}
/* Layer `layer` for n rows: route the logits lg (n rows of E from 0, bias as rt's
 * has_bias says) and run the experts on x (n rows of D from 0) into d->y. rt gives the
 * router's kind, bias, groups, norm, temp, scale and eps; mp the activation (act, limit,
 * a, b), D and I; the rest is filled here. */
static int vkc_dmoe_layer(VkcDmoe *d, int layer, int n, VkcBuf *lg, VkcBuf *bias, VkcBuf *x,
                          const VkcRoute *rt, const VkcMoe *mp) {
    int i = layer - d->lo, io = i * d->cap * d->K;
    vkc_hyb_flush_before(i);   /* the steps before this layer's that are due (vkc_hyb_flush_before) */
    VkcRoute r = *rt;
    r.S = n; r.E = d->E; r.K = d->K; r.l_off = 0; r.l_row = d->E; r.i_off = io; r.i_row = d->K; r.w_off = 0; r.w_row = d->K;
    VkcMoe m = *mp;
    m.S = n; m.K = d->K; m.t_off = i * d->E; m.x_off = 0; m.x_row = mp->D; m.i_off = io; m.i_row = d->K;
    m.w_off = 0; m.w_row = d->K; m.y_off = 0; m.y_row = mp->D;
    if (!m.gfmt && d->tabh) m.gfmt = vkc_dmoe_fmt(d, i, 12);   /* the layer's gate/up format */
    if (!m.dfmt && d->tabh) m.dfmt = vkc_dmoe_fmt(d, i, 15);   /* and down's (the integer form wants both) */
    if (n > VKC_DMOE_ROWS) {   /* a prompt chunk: the grouped GEMMs (vkc_dmoe_prepare checked the layer's formats) */
        m.E = d->E;
        /* With a host pool the GEMMs read some experts over PCIe, and the floats' were faster
         * there: Qwen3.8 Next IQ2_XS on an RTX PRO 4000 (19 GiB of experts in the host pool), a
         * 507-token prompt, 115-116 tok/s against 93-94 on the matrix units; without the pool the
         * two were even (104-114 against 109-120). */
        int stg = d->stg_on && d->frn[i] > 0, have = stg && d->stg_layer == layer;
        m.floats = vkt_host_pool_on() && !(d->stg_on && d->full[i]);   /* staged: VRAM, the matrix units' GEMMs */
        /* COLI_VK_DMOE_ROWS_MAX's frame split (vkc_dmoe_rows_max above): each group routes
         * its own rows and computes them into its own slice of y -- the buffers are the
         * chunk's (vkc_dmoe_prepare sized them), the groups disjoint, so at equal rows the
         * split computes what the whole chunk does (the unit's split-vs-unsplit seam). */
        const int rmax = vkc_dmoe_rows_max();
        for (int off = 0, first = 1; off < n; first = 0, off += rmax > 0 ? rmax : n) {
            int gn = n - off < rmax || rmax <= 0 ? n - off : rmax;
            VkcRoute rg = r; VkcMoe mg = m;
            rg.S = gn; rg.l_off = off * d->E; rg.i_off = io + off * d->K; rg.w_off = off * d->K;
            mg.S = gn; mg.x_off = off * mp->D; mg.i_off = io + off * d->K; mg.w_off = off * d->K; mg.y_off = off * mp->D;
            if (!first && !(vkc_submit(0) && vkc_begin())) return 0;   /* a frame per group: no submission carries the whole chunk */
            if (!vkc_route(lg, bias, d->rid, d->rwt, &rg)) return 0;
            /* the layer's host pool experts into stg: once, after the layer before is done reading it,
             * before this one's GEMMs read it (both by address, so the barriers are ours); skipped
             * when stg still holds this layer's copies (stg_reused) */
            if (first && stg && !have && !(vkc_barrier() && vkc_fetch(d->fr, d->fro[i], d->frn[i], d->frmax[i], d->stg) && vkc_barrier())) return 0;
            if (!vkc_moe_rows(x, d->rid, d->rwt, d->stg_on ? d->tabs : d->tab, &d->rows, d->y, &mg)) return 0;
        }
        if (stg && have) d->stg_reused++;
        else if (stg) { d->keep_valid = 0; d->stg_steps++; for (int k = 0; k < d->frn[i]; k++) d->stg_bytes += 4.0 * d->frh[4 * (d->fro[i] + k) + 3]; }
        d->stg_layer = stg ? layer : -1;
        d->steps++; d->big_steps++;
        return 1;
    }
    if (d->hybrid && d->gate.use && d->hx_n && d->hx_n[i] > 0 && m.act == 0) {
        /* frame A: the route and the input rows down, then the device's timeline; frame B:
         * the device's experts; frame C, after the host's: the CPU's sum added */
        uint64_t g = ++g_vkc_hyb_seq;
        VkcEw add = {VKC_EW_ADD, n * d->D, d->D, 1, 0, 0, 0, 0, 0, 0, 0, 1.f};
        if (!vkc_route(lg, bias, d->rid, d->rwt, &r) || !vkc_copy(d->hxx, 0, x, 0, (size_t)n * d->D)) return 0;
        if (!vkc_hyb_start()) return 0;
        if (d->ev) {
            /* in the frame: the device's event, its experts, the wait for the host's event, the sum */
            if (!vkc_ev_signal(g)) return 0;
            VkcHybJob j = {d, i, n, g, m.act, m.limit, 1, vkc_serial() + 1};
            vkc_hyb_post(&j);
            if (!vkc_moe(x, d->rid, d->rwt, d->tabhyb, d->hid, d->y, &m) || !vkc_ev_wait(g) ||
                !vkc_ew(d->y, d->y, d->cy, NULL, NULL, &add)) return 0;
            d->steps++; d->hyb_steps++;
            /* the layer before's step now: the frame with both steps goes off, the device has it
             * while the host computes the earlier pairs and then records the next layer */
            while (g_vkc_hyb.nq > 1 && !g_vkc_hyb_in) vkc_hyb_flush_oldest();
            return 1;
        }
        VkcHybJob j = {d, i, n, g, m.act, m.limit, 0, 0};
        if (!vkc_hybrid_submit(g)) return 0;
        vkc_hyb_post(&j);   /* frame A is on its way: g will come */
        if (!vkc_begin() || !vkc_moe(x, d->rid, d->rwt, d->tabhyb, d->hid, d->y, &m) || !vkc_submit(0)) return 0;
        vkc_hybrid_wait_next(g);
        if (!vkc_begin() || !vkc_ew(d->y, d->y, d->cy, NULL, NULL, &add)) return 0;
        d->steps++; d->hyb_steps++;
        return 1;
    }
    if (!vkc_route(lg, bias, d->rid, d->rwt, &r) || !vkc_moe(x, d->rid, d->rwt, d->tab, d->hid, d->y, &m)) return 0;
    d->steps++;
    return 1;
}
/* Layer `layer`'s ids (n rows of K, rank order, -1 for an empty slot) once its frame is done. */
static const int *vkc_dmoe_ids(const VkcDmoe *d, int layer) {
    return (const int *)vkc_ptr(d->rid) + (size_t)(layer - d->lo) * d->cap * d->K;
}
static void vkc_dmoe_free(VkcDmoe *d) {
    vkc_hyb_drain();   /* the worker reads its buffers */
    vkc_free(d->tab); vkc_free(d->rid); vkc_free(d->rwt); vkc_free(d->hid); vkc_free(d->y);
    vkc_free(d->tabhyb); vkc_free(d->hxx); vkc_free(d->cy);
    vkc_free(d->stg); vkc_free(d->tabs); vkc_free(d->fr); free(d->koff); free(d->frk);
    vkc_moe_rows_free(&d->rows);
    free(d->on); free(d->tabh); free(d->rows_off); free(d->tabhh); free(d->hx); free(d->hx_n);
    free(d->tabsh); free(d->frh); free(d->fro); free(d->frn); free(d->frmax); free(d->full); free(d->hostm);
    free(d->chg); free(d->wr);
    memset(d, 0, sizeof *d);
}
/* the gate on the hybrid steps (vk_hyb_gate.h; COLI_VK_HYBRID_GATE=0: always on) */
static void vkc_dmoe_forward_done(VkcDmoe *d, int rows, double ms) {
    if (!d || !d->hybrid || rows < 1 || rows > VKC_DMOE_ROWS) return;
    static int dbg = -1;
    if (dbg < 0) { const char *e = getenv("COLI_VK_HYB_GATE_DEBUG"); dbg = e && *e == '1'; }
    if (dbg && d->gate.phase != 2) fprintf(stderr, "[VK] hyb gate: phase %d forward %d: %.2f ms%s\n", d->gate.phase, d->gate.n, ms, d->gate.use ? " (hybrid)" : "");
    if (vkc_hyb_gate_forward(&d->gate, ms) && !d->gate.use) vkc_hyb_stop();
}
/* the hybrid steps' line: steps, pairs the CPU computed, its time */
static void vkc_dmoe_hyb_report(const char *engine, const VkcDmoe *d) {
    const VkcHyb *w = &g_vkc_hyb;
    if (d->stg_steps)
        fprintf(stderr, "[VK] %s chain: %llu prompt layer steps staged the host pool's experts (%.2f GiB copied over PCIe, "
                        "%.0f MiB of staging, the tier's room %.0f MiB)\n",
                engine, d->stg_steps, d->stg_bytes / 1073741824.0, d->stg_peak / 1048576.0, vkt_stage_room() / 1048576.0);
    if (d->stg_reused)
        fprintf(stderr, "[VK] %s chain: %llu prompt layer steps read the experts their layer had staged already\n",
                engine, d->stg_reused);
    if (d->keep_fills)
        fprintf(stderr, "[VK] %s chain: between prompt chunks the staging held the host pool's hottest %d experts (%llu fills)\n",
                engine, d->nkeep, d->keep_fills);
    if (d->patches)
        fprintf(stderr, "[VK] %s chain: the experts' table read whole %llu times, patched %llu times (%llu entries the tier changed)\n",
                engine, d->reads, d->patches, d->patched);
    if (!d->hyb_steps && !d->gate.probes) return;
    if (d->hyb_steps) {
        fprintf(stderr, "[VK] %s chain: %llu hybrid layer steps (%s), %llu host pool pairs on the CPU in %.1f ms (%.1f us a step), "
                        "%.1f ms waiting for the device%s\n",
                engine, d->hyb_steps, d->ev ? "in the frame, on events" : "frames A/B/C, by timelines",
                w->pairs, w->cpu_ms, w->steps ? w->cpu_ms * 1e3 / (double)w->steps : 0.0, w->wait_ms,
                w->err ? ", errors seen" : "");
    }
    if (d->gate.probes)
        fprintf(stderr, "[VK] %s chain: the hybrid steps measured %d time(s): a decode forward %.1f ms with them, %.1f ms with the "
                        "device reading the host pool itself; %s (COLI_VK_HYBRID_GATE=0 keeps them on)\n",
                engine, d->gate.probes, d->gate.kept[0], d->gate.kept[1],
                d->gate.use ? "kept" : "the device's reads kept");
}
#endif
