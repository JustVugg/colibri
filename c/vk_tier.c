/* vk_tier.c -- the Vulkan routed-expert tier shared by the MoE engines. See
 * vk_tier.h for what it does and how an engine calls it.
 *
 * Threads. Everything runs on the engine thread except two things: the uploader
 * thread, which turns staged copies of promoted experts into device tensors, and
 * vkt_put, which a warm start may call from many loader threads. Neither touches
 * a slot: they hand finished tensors back through the `done` list, and the engine
 * thread makes them resident at a quiescent point (no batch in flight), where it
 * also frees evicted experts. So the slot table, the counters and every Vulkan
 * call but the pool's allocation are single-threaded; the lock guards the queue,
 * the done list, the room-made signal and the count of victims not freed yet
 * only.
 *
 * Two devices. With COLI_VK_DEV2 the tier holds experts on a second device too
 * (backend_vulkan.c's batch on that device, its own budget): a slot says which
 * device holds it, the warm start fills the primary device with the hottest experts
 * and the second with the next ones, a promotion goes where there is room or takes
 * the coldest resident's place on either device, and a step sends each device its
 * own batch, both in flight at once. Streaming (big prefill steps' cold experts)
 * stays on the primary device. A second device that fails gives its experts back
 * to the CPU and the tier goes on with the primary one. */
#ifdef COLI_VULKAN
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <time.h>
#include "compat.h"
#include <errno.h>
#include "vk_tier.h"
#include "backend_vulkan.h"
#include "tier.h"

#define VKT_QCAP 32                 /* promotions staged and waiting for the uploader */
#define VKT_CAND 64                 /* victim candidates kept per refresh */
#define VKT_DECAY_TOKENS 1024u      /* tier.h's decay period, in tokens */

enum { VS_NONE = 0, VS_QUEUED, VS_RESIDENT, VS_EVICT };

typedef struct {
    ColiVkExpert *ex;
    uint32_t heat, last;
    uint8_t state;
    uint8_t dev;                       /* the device it is on or on its way to (0, or 1: COLI_VK_DEV2's) */
} VSlot;
typedef struct { int layer, eid, dev; uint8_t *buf; } VQ;
typedef struct { int layer, eid, ok, copy_failed; ColiVkTensor *g, *u, *d; } VDone;
/* streaming: a staging slot (an expert's three tensors on the device, filled again for
 * every expert it carries) and a sub-batch in flight */
enum { SS_FREE = 0, SS_FILLED, SS_BUSY };
typedef struct { ColiVkTensor *t[3]; ColiVkExpert *ex; int layer, eid, state, pending, pref; } VStage;
typedef struct { int n, nslot, cap, *slot; float **yout; } VSub;

static struct {
    int on;
    VktConfig c;
    char engine[32];
    /* the forms of the experts, as the engine holds them and on the device: [0] the main
     * layers', [1] the extra layers' (VktConfig.extra_layers, an MTP head's), from
     * main_layers on */
    int main_layers;
    struct { VktFmt gu, dn; int gu_fmt, gu_gs, dn_fmt, dn_gs; size_t gu_codes, gu_scales, d_codes, d_scales, stage_bytes; } form[2];
    /* the extra layers' experts: on the primary device, in a pool of their own (another
     * size than the main ones', so neither leaves holes the other cannot use), at most
     * xmax of x_bytes each in x_budget */
    int xmax, xres, xque;
    size_t x_bytes, x_budget;
    unsigned long long extra_served, extra_routed;
    size_t exp_bytes, budget;
    int max_resident, resident, queued, rate, uma;   /* totals over the devices */
    /* per device: the most it holds, resident, queued (dmax[1] = 0 without a second one) */
    int dmax[2], dres[2], dque[2];
    /* the second device (COLI_VK_DEV2) */
    int d2;                            /* the tier holds experts there */
    char d2_name[64];
    size_t d2_budget, d2_exp_bytes;
    int d2_rows;                       /* the most rows of a big step its one batch takes */
    int d0_inflight, d2_inflight, d2_rows0;   /* a batch in flight on each; where the second's rows start in by[] */
    int nd2; int *d2_idx; int cd2;     /* a big step's rows on it: their assignments */
    const float **d2_y; ColiVkExpert **d2_bex; int *d2_brows; const float **d2_bx; float *d2_bw; int d2_cap, d2_cexp;
    unsigned long long d2_served, d2_steps, d2_big, d2_uploads, d2_lost;
    /* exclusive RAM/VRAM (vkt_ram_first): on, and the RAM slots given up for it. routed[l*E+e]
     * is the issue (seq) that last routed the expert, layer_seq[l] the layer's last issue: an
     * expert routed in its layer's current step may be one the CPU is computing from its RAM
     * slot (handed back by the balance, past a batch's rows), so it is never offered */
    int excl; unsigned long long ram_gave;
    uint32_t *routed_at, *layer_seq, seq;
    double d2_ms;
    VSlot *s;
    uint32_t tick, decay_at;          /* tokens seen (rows of the forward's first layer) */
    int last_layer, first_layer, promos[2], promo_cap;   /* promotions this forward, the main layers' and the extra ones', each up to the cap */
    int begin;                         /* vkt_begin_forward: the next issue starts a forward */
    /* uploader */
    pthread_t th; int th_on, stop;
    pthread_mutex_t mx; pthread_cond_t cv, cv_room, cv_done;
    int busy;                          /* the uploader is between taking an entry and push_done */
    int sync;                          /* COLI_VK_TIER_SYNC=1: quiescent points wait for the uploader */
    int evict_pending;                 /* victims chosen and not freed yet (a batch is in flight) */
    VQ q[VKT_QCAP]; int qh, qn;
    VDone *done; int ndone, cdone;
    unsigned long room_gen;
    /* evicted, freed at the next quiescent point */
    int *evict; int nevict, cevict;
    /* the coldest residents, refreshed once per forward */
    int cand[2][VKT_CAND]; uint64_t cand_score[2][VKT_CAND]; int ncand[2];   /* per kind: main, extra */
    /* the step in flight */
    int inflight, S, K;
    int *map; int cmap;
    int *grp; int *touched;              /* expert -> group of this step, and the groups' experts */
    ColiVkExpert **bex; int *brows; int *bfirst; int cb;
    const float **bx, **by; int *rowsrc; int cbx;
    float *bw;                           /* the rows' route weights (vkt_issue_w), beside bx */
    int max_dev_rows;
    double t_issued;
    /* the balance: the share of a step's resident experts the device takes, moved by
     * what each join waited (the device was the slower side) or did not */
    int balance, can_balance; float share; unsigned long long handed;
    /* accounting */
    unsigned long long routed, served, steps, uploads, upload_bytes, evictions, qfull, rated, refused, failed, warm;
    double dev_ms, cpu_ms, wait_ms;
    unsigned long long rep_routed, rep_served;
    /* streaming (big prefill steps; see "streaming" below) */
    int st_ok, st_slots, st_n, st_per, st_half, st_rows_env, gemm_rows, st_par;
    VStage *st;
    double bw_up, cpu_row_ms;             /* measured: bytes per ms into a slot; CPU ms per (row, expert) */
    int big, big_fail, big_n, big_taken, said_fwd;
    unsigned sk;                          /* sub-batches issued in this step */
    VSub hf[2];                           /* the two in flight */
    float *sy; size_t csy;                /* the step's device rows, S*K*hidden */
    float *sy_step;                       /* the backend's whole-step rows instead (coli_vk_xb_step_begin), or NULL */
    int big_valid;                        /* the step's routed assignments (idx >= 0) */
    int *bcnt, *bofs, *blist, *bcls, cblist; /* per expert of the step: rows, first row, class; the rows grouped */
    uint32_t *pred; uint8_t *pred_ok;     /* per layer: the last big step's rows per expert */
    unsigned long long st_steps, st_experts, st_rows, st_bytes, st_subs, st_sums, st_kept, st_kept_rows, pf_n, pf_used;
    double st_up_ms;
} T;

/* "2.31 GiB", "35.0 MiB", "9.5 KiB": the budget lines read on a 24 GB card and a tiny fixture alike */
static const char *human(double b, char *buf, size_t n) {
    if (b >= 1073741824.0) snprintf(buf, n, "%.2f GiB", b / 1073741824.0);
    else if (b >= 1048576.0) snprintf(buf, n, "%.1f MiB", b / 1048576.0);
    else snprintf(buf, n, "%.1f KiB", b / 1024.0);
    return buf;
}
static double vkt_now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }
static VSlot *slot(int layer, int eid) { return &T.s[(size_t)layer * T.c.experts + eid]; }
static int kind_of(int layer) { return layer >= T.main_layers; }   /* the form of a layer's experts */

/* ---- source formats ---------------------------------------------------------- */
static int src_per_row(VktSrc k) {
    return k == VKT_SRC_I8_ROW || k == VKT_SRC_I8_AS_I4_ROW || k == VKT_SRC_I4S_PAIRS_ROW || k == VKT_SRC_I4U_PAIRS_ROW;
}
static size_t src_row_bytes(VktSrc k, int I) {
    switch (k) {
    case VKT_SRC_I8_ROW: case VKT_SRC_I8_GS: case VKT_SRC_I8_AS_I4_ROW: case VKT_SRC_I8_AS_I4_GS:
    case VKT_SRC_FP8_GS: case VKT_SRC_FP8_BLOCK: return (size_t)I;
    case VKT_SRC_I3_G64: return ((size_t)I + 63) / 64 * 24;
    case VKT_SRC_BF16: return (size_t)I * 2;
    case VKT_SRC_F32: return (size_t)I * 4;
    default: return ((size_t)I + 1) / 2;                 /* the int4 and fp4 kinds */
    }
}
static size_t src_scale_bytes(VktFmt f, int I, int O) {
    if (f.kind == VKT_SRC_BF16 || f.kind == VKT_SRC_F32) return 0;
    if (src_per_row(f.kind)) return (size_t)O * 4;
    if (f.kind == VKT_SRC_I4U_PLANAR64 || f.kind == VKT_SRC_I3_G64) return (size_t)O * (((size_t)I + 63) / 64) * 4;
    size_t ng = ((size_t)I + f.gs - 1) / f.gs;
    if (f.kind == VKT_SRC_MXFP4_E8M0) return (size_t)O * ng;
    if (f.kind == VKT_SRC_FP8_BLOCK) return (((size_t)O + f.gs - 1) / f.gs) * ng * 4;
    return (size_t)O * ng * 4;
}
/* The backend format a source kind lands in; 0 when the combination is not one. */
static int dev_fmt(VktFmt f, int *fmt, int *gs) {
    int g = f.gs;
    switch (f.kind) {
    case VKT_SRC_I8_ROW: *fmt = 1; *gs = 0; return 1;
    case VKT_SRC_I8_GS: *fmt = 13; *gs = g; return g >= 4 && g % 4 == 0;
    case VKT_SRC_I8_AS_I4_ROW: case VKT_SRC_I4S_PAIRS_ROW: case VKT_SRC_I4U_PAIRS_ROW: *fmt = 2; *gs = 0; return 1;
    case VKT_SRC_I8_AS_I4_GS: case VKT_SRC_I4S_PAIRS_GS: case VKT_SRC_I4U_PAIRS_GS: *fmt = 4; *gs = g; return g >= 8 && g % 8 == 0;
    case VKT_SRC_I4U_PLANAR64: *fmt = 4; *gs = 64; return 1;
    case VKT_SRC_I3_G64: *fmt = 5; *gs = 0; return 1;
    case VKT_SRC_MXFP4_F32: case VKT_SRC_MXFP4_E8M0: *fmt = 7; *gs = g; return g >= 8 && g % 8 == 0;
    case VKT_SRC_FP8_GS: case VKT_SRC_FP8_BLOCK: *fmt = 12; *gs = g; return g >= 4 && g % 4 == 0;
    case VKT_SRC_BF16: *fmt = 11; *gs = 0; return 1;
    case VKT_SRC_F32: *fmt = 10; *gs = 0; return 1;
    default: return 0;
    }
}

/* One matrix [O x I] from its RAM form into a device tensor's mapping: rows at
 * `stride` bytes (already zeroed past the row), scales as the backend reads them.
 * Pure byte work; the values are never requantized. */
static void convert_par(VktFmt f, int I, int O, const uint8_t *codes, const void *scales,
                        uint8_t *rows, size_t stride, float *sc, int par);
static void convert(VktFmt f, int I, int O, const uint8_t *codes, const void *scales,
                    uint8_t *rows, size_t stride, float *sc) {
    convert_par(f, I, O, codes, scales, rows, stride, sc, 0);
}
/* par: the rows split over the OpenMP threads (streaming, on the engine thread) */
static void convert_par(VktFmt f, int I, int O, const uint8_t *codes, const void *scales,
                        uint8_t *rows, size_t stride, float *sc, int par) {
    size_t rb = src_row_bytes(f.kind, I);
    par = par && O >= 64;
    switch (f.kind) {
    case VKT_SRC_I8_AS_I4_ROW: case VKT_SRC_I8_AS_I4_GS:   /* int8 in [-8,7] -> v+8 nibble pairs */
        #pragma omp parallel for schedule(static) if (par)
        for (int o = 0; o < O; o++) {
            const int8_t *src = (const int8_t *)codes + (size_t)o * I;
            uint8_t *dst = rows + (size_t)o * stride;
            for (int j = 0; j < I / 2; j++)
                dst[j] = (uint8_t)(((src[2 * j] + 8) & 15) | (((src[2 * j + 1] + 8) & 15) << 4));
            if (I & 1) dst[I / 2] = (uint8_t)((src[I - 1] + 8) & 15);
        }
        break;
    case VKT_SRC_I4S_PAIRS_ROW: case VKT_SRC_I4S_PAIRS_GS:  /* two's complement nibble -> v+8: xor 8 */
        #pragma omp parallel for schedule(static) if (par)
        for (int o = 0; o < O; o++) {
            const uint8_t *src = codes + (size_t)o * rb;
            uint8_t *dst = rows + (size_t)o * stride;
            for (size_t j = 0; j < rb; j++) dst[j] = src[j] ^ 0x88u;
        }
        break;
    case VKT_SRC_I4U_PLANAR64:   /* planar blocks (byte k = elements k, k+32) -> pairs (2j, 2j+1) */
        #pragma omp parallel for schedule(static) if (par)
        for (int o = 0; o < O; o++) {
            const uint8_t *src = codes + (size_t)o * rb;
            uint8_t *dst = rows + (size_t)o * stride;
            int nb = I / 64;
            for (int b = 0; b < nb; b++) {
                const uint8_t *blk = src + (size_t)b * 32;
                for (int p = 0; p < 32; p++) {
                    int e0 = 2 * p, e1 = 2 * p + 1;
                    unsigned a = e0 < 32 ? (blk[e0] & 15u) : (unsigned)(blk[e0 - 32] >> 4);
                    unsigned c = e1 < 32 ? (blk[e1] & 15u) : (unsigned)(blk[e1 - 32] >> 4);
                    dst[(size_t)b * 32 + p] = (uint8_t)(a | (c << 4));
                }
            }
            memcpy(dst + (size_t)nb * 32, src + (size_t)nb * 32, rb - (size_t)nb * 32);   /* tail: pairs already */
        }
        break;
    default:   /* the same bytes */
        #pragma omp parallel for schedule(static) if (par)
        for (int o = 0; o < O; o++) memcpy(rows + (size_t)o * stride, codes + (size_t)o * rb, rb);
        break;
    }
    if (f.kind == VKT_SRC_BF16 || f.kind == VKT_SRC_F32) { sc[0] = 1.0f; return; }
    if (f.kind == VKT_SRC_MXFP4_E8M0) {        /* ue8m0 -> f32 2^(s-127), quant.h's mx4_scale */
        size_t n = (size_t)O * (((size_t)I + f.gs - 1) / f.gs);
        const uint8_t *e = scales;
        for (size_t i = 0; i < n; i++) { uint32_t u = (uint32_t)e[i] << 23; memcpy(&sc[i], &u, 4); }
        return;
    }
    if (f.kind == VKT_SRC_FP8_BLOCK) {         /* [O/bs][I/bs] blocks -> one row each */
        size_t nb = ((size_t)I + f.gs - 1) / f.gs;
        const float *b = scales;
        for (int o = 0; o < O; o++) memcpy(sc + (size_t)o * nb, b + (size_t)(o / f.gs) * nb, nb * 4);
        return;
    }
    memcpy(sc, scales, src_scale_bytes(f, I, O));
}

/* Upload one expert from its RAM form to device dev's pool (any thread). 0 when the pool
 * refused, -1 when the copy to the device failed (staged uploads: the room was there,
 * and is given back). */
static int upload(int dev, int kind, const uint8_t *g, const uint8_t *u, const uint8_t *d,
                  const void *gs, const void *us, const void *ds, ColiVkTensor *t[3]) {
    const int H = T.c.hidden, F = T.c.inter;
    const __typeof__(T.form[0]) *fm = &T.form[kind];
    const uint8_t *codes[3] = {g, u, d}; const void *sc[3] = {gs, us, ds};
    t[0] = t[1] = t[2] = NULL;
    for (int k = 0; k < 3; k++) {
        VktFmt f = k < 2 ? fm->gu : fm->dn;
        int fmt = k < 2 ? fm->gu_fmt : fm->dn_fmt, fgs = k < 2 ? fm->gu_gs : fm->dn_gs;
        int I = k < 2 ? H : F, O = k < 2 ? F : H;
        uint8_t *rows; size_t stride; float *scales;
        if (!(kind ? coli_vk_tier_tensor_extra(&t[k], fmt, I, O, fgs, &rows, &stride, &scales)   /* its own pool */
                   : coli_vk_tier_tensor_dev(dev, &t[k], fmt, I, O, fgs, &rows, &stride, &scales))) {
            for (int j = 0; j < k; j++) { coli_vk_tensor_free(t[j]); t[j] = NULL; }
            return 0;
        }
        convert(f, I, O, codes[k], sc[k], rows, stride, scales);
    }
    /* staged uploads: the three host images go to device memory now (nothing to do on
     * mapped memory) */
    if (!coli_vk_tensor_commit(t, 3)) {
        for (int k = 0; k < 3; k++) { coli_vk_tensor_free(t[k]); t[k] = NULL; }
        return -1;
    }
    return 1;
}

/* One expert's bytes on a device whose weight ranges start at `a` bytes. */
static size_t expert_bytes_at(int H, int F, VktFmt gu, VktFmt dn, size_t a) {
    int gf, gg, df, dg;
    if (!dev_fmt(gu, &gf, &gg) || !dev_fmt(dn, &df, &dg)) return 0;
    size_t b = 0;
    for (int k = 0; k < 3; k++) {
        int fmt = k < 2 ? gf : df, gs = k < 2 ? gg : dg, I = k < 2 ? H : F, O = k < 2 ? F : H;
        size_t rows = (coli_vk_tensor_row_bytes(fmt, I) + 3) / 4 * 4 * (size_t)O;
        size_t sc = coli_vk_tensor_scale_count(fmt, I, O, gs) * 4;
        b += (rows + a - 1) / a * a + (sc + a - 1) / a * a;   /* two ranges, each aligned */
    }
    return b;
}
size_t vkt_expert_bytes(int H, int F, VktFmt gu, VktFmt dn) {
    return expert_bytes_at(H, F, gu, dn, coli_vk_buffer_alignment());
}

/* ---- the uploader -------------------------------------------------------------- */
static void push_done(int layer, int eid, int ok, int copy_failed, ColiVkTensor *t[3]) {
    pthread_mutex_lock(&T.mx);
    if (T.ndone == T.cdone) {
        int nc = T.cdone ? 2 * T.cdone : 64;
        VDone *n = realloc(T.done, (size_t)nc * sizeof(*n));
        if (!n) {   /* cannot even record it: the slot stays queued and is never used */
            T.busy = 0; pthread_cond_broadcast(&T.cv_done);
            pthread_mutex_unlock(&T.mx);
            for (int k = 0; k < 3; k++) if (t[k]) coli_vk_tensor_free(t[k]);
            return;
        }
        T.done = n; T.cdone = nc;
    }
    T.done[T.ndone++] = (VDone){layer, eid, ok, copy_failed, t[0], t[1], t[2]};
    T.busy = 0; pthread_cond_broadcast(&T.cv_done);
    pthread_mutex_unlock(&T.mx);
}

static void *uploader(void *arg) {
    (void)arg;
    for (;;) {
        pthread_mutex_lock(&T.mx);
        while (!T.qn && !T.stop) pthread_cond_wait(&T.cv, &T.mx);
        /* COLI_VK_TIER_SYNC=1: a victim chosen while a batch is in flight is freed at that
         * batch's join, before the join waits for the uploads staged so far: wait for
         * that free here, so the promotion that displaced it finds its room. Uploading
         * first would meet the full pool, and a refusal is final in this mode. */
        while (T.sync && T.evict_pending && !T.stop) pthread_cond_wait(&T.cv_room, &T.mx);
        if (T.stop) { pthread_mutex_unlock(&T.mx); return NULL; }
        VQ e = T.q[T.qh]; T.qh = (T.qh + 1) % VKT_QCAP; T.qn--; T.busy = 1;
        pthread_mutex_unlock(&T.mx);
        const uint8_t *b = e.buf;
        const int kd = kind_of(e.layer);
        const size_t guc = T.form[kd].gu_codes, dc = T.form[kd].d_codes, gus = T.form[kd].gu_scales;
        const uint8_t *g = b, *u = g + guc, *d = u + guc;
        const uint8_t *gs = d + dc, *us = gs + gus, *ds = us + gus;
        ColiVkTensor *t[3];
        int ok = 0, r = 0;
        /* Refused: the pool is at its budget until the engine thread frees the victim
         * this promotion displaced, at its next quiescent point (a join: every layer
         * in decode, after a chunk's CPU work in prefill). Wait for room to be made
         * and try again; refused again after three frees, or nothing freed for a
         * minute (the engine stopped stepping), the pool is full for real. */
        for (int frees = 0, waited = 0; !ok; ) {
            r = upload(e.dev, kd, g, u, d, gs, us, ds, t);
            ok = r > 0;
            /* sync: every free decided so far came first (above), and the engine now waits
             * on us, not we on it: the pool is full for real. A failed copy is no question
             * of room: no waiting for one. */
            if (ok || r < 0 || frees >= 3 || waited >= 600 || T.sync) break;
            pthread_mutex_lock(&T.mx);
            unsigned long gen = T.room_gen;
            while (T.room_gen == gen && !T.stop && waited < 600) {
                struct timespec until; clock_gettime(CLOCK_REALTIME, &until);
                until.tv_nsec += 100 * 1000000L; if (until.tv_nsec >= 1000000000L) { until.tv_sec++; until.tv_nsec -= 1000000000L; }
                if (pthread_cond_timedwait(&T.cv_room, &T.mx, &until) == ETIMEDOUT) waited++;
            }
            if (T.room_gen != gen) frees++;
            int stop = T.stop;
            pthread_mutex_unlock(&T.mx);
            if (stop) break;
        }
        free(e.buf);
        if (!ok) t[0] = t[1] = t[2] = NULL;
        push_done(e.layer, e.eid, ok, r < 0, t);
    }
}

/* ---- quiescent points: nothing in flight ---------------------------------------- */
static void quiesce(void) {
    if (T.inflight) return;
    int freed = 0, had = T.nevict;
    for (int i = 0; i < T.nevict; i++) {
        VSlot *v = &T.s[T.evict[i]];
        if (v->ex) { coli_vk_xb_expert_free(v->ex); v->ex = NULL; freed = 1; }
        v->state = VS_NONE;
    }
    T.nevict = 0;
    if (had) {   /* room made: an upload may be waiting for it */
        pthread_mutex_lock(&T.mx);
        if (freed) T.room_gen++;
        T.evict_pending = 0;
        pthread_cond_broadcast(&T.cv_room);
        pthread_mutex_unlock(&T.mx);
    }
    /* finished uploads become resident; COLI_VK_TIER_SYNC=1 waits for every staged one
     * first, so what is resident depends on the routing alone, not on thread timing
     * (tests: a fixture's whole run can be over before the uploader is scheduled) */
    pthread_mutex_lock(&T.mx);
    if (T.sync && T.th_on) while ((T.qn || T.busy) && !T.stop) pthread_cond_wait(&T.cv_done, &T.mx);
    int n = T.ndone; VDone *d = n ? malloc((size_t)n * sizeof(*d)) : NULL;
    if (d) { memcpy(d, T.done, (size_t)n * sizeof(*d)); T.ndone = 0; } else n = 0;
    pthread_mutex_unlock(&T.mx);
    for (int i = 0; i < n; i++) {
        VSlot *v = slot(d[i].layer, d[i].eid);
        int dv = v->dev, kd = kind_of(d[i].layer);
        T.queued--; if (kd) T.xque--; else T.dque[dv]--;
        /* an upload to a second device that has stopped meanwhile (d2_stop) is dropped */
        ColiVkExpert *e = d[i].ok && (dv == 0 || T.d2) ? coli_vk_xb_expert(d[i].g, d[i].u, d[i].d) : NULL;
        if (e) {
            v->ex = e; v->state = VS_RESIDENT; T.resident++; if (kd) T.xres++; else T.dres[dv]++;
            T.uploads++; T.upload_bytes += coli_vk_tensor_bytes(d[i].g) + coli_vk_tensor_bytes(d[i].u) + coli_vk_tensor_bytes(d[i].d);
            if (dv) T.d2_uploads++;
        } else {
            for (int k = 0; k < 3; k++) { ColiVkTensor *t = k == 0 ? d[i].g : k == 1 ? d[i].u : d[i].d; if (t) coli_vk_tensor_free(t); }
            v->state = VS_NONE; v->dev = 0;
            if (dv && !T.d2) continue;
            T.failed++;
            /* refused: the budget holds fewer than the count said (fragmentation, or the
             * device ran out first): stop planning past what is there. A failed copy gave
             * its room back: the budget stands, the expert may come again. */
            if (!d[i].copy_failed && kd && T.xmax > T.xres + T.xque) T.xmax = T.xres + T.xque;
            if (!d[i].copy_failed && !kd && T.dmax[dv] > T.dres[dv] + T.dque[dv]) {
                T.max_resident -= T.dmax[dv] - (T.dres[dv] + T.dque[dv]);
                T.dmax[dv] = T.dres[dv] + T.dque[dv];
            }
        }
    }
    free(d);
}

/* The second device stopped (a batch there failed, or the device did): its residents
 * are freed (nothing is in flight there when this runs) and go back to the CPU, uploads
 * on their way there are dropped when they land, and the tier goes on with the primary
 * device. */
static void d2_stop(const char *why) {
    if (!T.d2) return;
    T.d2 = 0; T.d2_lost++;
    size_t n = (size_t)T.c.layers * T.c.experts;
    for (size_t i = 0; i < n; i++) {
        VSlot *v = &T.s[i];
        if (v->dev != 1 || v->state != VS_RESIDENT) continue;
        if (v->ex) { coli_vk_xb_expert_free(v->ex); v->ex = NULL; }
        v->state = VS_NONE; v->dev = 0; T.resident--; T.dres[1]--;
    }
    T.max_resident -= T.dmax[1]; T.dmax[1] = 0;
    fprintf(stderr, "[VK] tier %s: %s, its experts go back to the CPU and the tier goes on with %s\n",
            T.engine, why, coli_vk_device_name());
}

/* The device a newcomer of layer `layer` goes to when there is room: a main layer's to
 * the primary device first, then the second one; an extra layer's to its pool on the
 * primary device; -1 when they are full. */
static int room_dev(int layer) {
    if (kind_of(layer)) return T.xres + T.xque < T.xmax ? 0 : -1;
    if (T.dres[0] + T.dque[0] < T.dmax[0]) return 0;
    if (T.d2 && T.dres[1] + T.dque[1] < T.dmax[1]) return 1;
    return -1;
}

/* ---- placement ------------------------------------------------------------------ */
static uint64_t score(const VSlot *v) { return tier_lfru_score(v->heat, v->last, T.tick); }

/* The coldest residents of each kind, ascending, once per forward. */
static void refresh_candidates(void) {
    T.ncand[0] = T.ncand[1] = 0;
    size_t n = (size_t)T.c.layers * T.c.experts;
    for (size_t i = 0; i < n; i++) {
        VSlot *v = &T.s[i];
        if (v->state != VS_RESIDENT) continue;
        int k = kind_of((int)(i / (size_t)T.c.experts)), *c = T.cand[k];
        uint64_t sc = score(v), *cs = T.cand_score[k];
        if (T.ncand[k] == VKT_CAND && sc >= cs[VKT_CAND - 1]) continue;
        int at = T.ncand[k] < VKT_CAND ? T.ncand[k]++ : VKT_CAND - 1;
        while (at > 0 && cs[at - 1] > sc) { c[at] = c[at - 1]; cs[at] = cs[at - 1]; at--; }
        c[at] = (int)i; cs[at] = sc;
    }
}
/* The resident a newcomer of kind k and score hs would displace, or -1: the coldest
 * candidate of its kind (each kind has its own pool) still resident, when the newcomer
 * beats it by tier.h's LFRU margin (25% + 4 counts). It stays first in the list until
 * drop_victim takes it. */
static void drop_victim(int k) {
    memmove(T.cand[k], T.cand[k] + 1, (size_t)--T.ncand[k] * sizeof(int));
    memmove(T.cand_score[k], T.cand_score[k] + 1, (size_t)T.ncand[k] * sizeof(uint64_t));
}
static int peek_victim(int k, uint64_t hs) {
    while (T.ncand[k]) {
        int i = T.cand[k][0];
        VSlot *v = &T.s[i];
        if (v->state != VS_RESIDENT) { drop_victim(k); continue; }
        uint64_t cs = score(v);
        return hs > cs + (cs >> 2) + (4u << 8) ? i : -1;
    }
    return -1;
}

/* A forward starts when the layer index goes back (a layer that issues several
 * blocks of rows, or rows one by one, is the same forward). The tick counts tokens:
 * the rows of the forward's first layer, every block of them. */
static void new_forward(void) {
    T.promos[0] = T.promos[1] = 0; T.promo_cap = 0; T.said_fwd = 0;
    if (T.tick >= T.decay_at) {
        size_t n = (size_t)T.c.layers * T.c.experts;
        for (size_t i = 0; i < n; i++) T.s[i].heat = tier_decay_value(T.s[i].heat);
        T.decay_at = T.tick + VKT_DECAY_TOKENS;
    }
    refresh_candidates();
}

/* The bytes of an expert the CPU just computed pass by: promote it when there is
 * room, or when it is hot enough to displace the coldest resident. The copy is the
 * only cost on this thread; the uploader does the rest. */
void vkt_note(int layer, int eid, const VktExpertSrc *src) {
    if (!T.on || !src || layer < 0 || layer >= T.c.layers || eid < 0 || eid >= T.c.experts) return;
    VSlot *v = slot(layer, eid);
    if (v->state != VS_NONE || !src->g || !src->u || !src->d) return;
    /* the extra layers (an MTP head's) come last in a forward: their promotions are
     * counted apart, or the model's layers would always have spent the rate first */
    const int kd = kind_of(layer);
    if (T.promos[kd] >= T.promo_cap) { T.rated++; return; }
    const __typeof__(T.form[0]) *fm = &T.form[kd];
    if (fm->gu_scales && (!src->gs || !src->us || !src->ds)) return;
    int victim = -1, dev = room_dev(layer);
    if (dev < 0 && (!v->heat || (victim = peek_victim(kd, score(v))) < 0)) return;
    if (victim >= 0) dev = T.s[victim].dev;   /* the newcomer takes its place, on its device */
    pthread_mutex_lock(&T.mx);
    int qfull = T.qn >= VKT_QCAP;
    pthread_mutex_unlock(&T.mx);
    if (qfull) { T.qfull++; return; }
    uint8_t *buf = malloc(fm->stage_bytes);
    if (!buf) return;
    if (victim >= 0) drop_victim(kd);
    uint8_t *p = buf;
    memcpy(p, src->g, fm->gu_codes); p += fm->gu_codes;
    memcpy(p, src->u, fm->gu_codes); p += fm->gu_codes;
    memcpy(p, src->d, fm->d_codes);  p += fm->d_codes;
    if (fm->gu_scales) { memcpy(p, src->gs, fm->gu_scales); p += fm->gu_scales; memcpy(p, src->us, fm->gu_scales); p += fm->gu_scales; }
    if (fm->d_scales) memcpy(p, src->ds, fm->d_scales);
    if (victim >= 0) {
        VSlot *w = &T.s[victim];
        w->state = VS_EVICT; T.resident--; if (kd) T.xres--; else T.dres[w->dev]--; T.evictions++;
        if (T.nevict == T.cevict) {
            int nc = T.cevict ? 2 * T.cevict : 64;
            int *n = realloc(T.evict, (size_t)nc * sizeof(int));
            if (!n) { w->state = VS_RESIDENT; T.resident++; if (kd) T.xres++; else T.dres[w->dev]++; T.evictions--; free(buf); return; }
            T.evict = n; T.cevict = nc;
        }
        T.evict[T.nevict++] = victim;
        pthread_mutex_lock(&T.mx); T.evict_pending = T.nevict; pthread_mutex_unlock(&T.mx);
        quiesce();   /* nothing in flight: free it right away; else at the join */
    }
    v->state = VS_QUEUED; v->dev = (uint8_t)dev; T.queued++; if (kd) T.xque++; else T.dque[dev]++; T.promos[kd]++;
    pthread_mutex_lock(&T.mx);
    T.q[(T.qh + T.qn) % VKT_QCAP] = (VQ){layer, eid, dev, buf}; T.qn++;
    pthread_cond_signal(&T.cv);
    pthread_mutex_unlock(&T.mx);
}

/* vkt_note's decision without its copy: the same tests, in the same order. */
int vkt_wants(int layer, int eid) {
    if (!T.on || layer < 0 || layer >= T.c.layers || eid < 0 || eid >= T.c.experts) return 0;
    VSlot *v = slot(layer, eid);
    if (v->state != VS_NONE || T.promos[kind_of(layer)] >= T.promo_cap) return 0;
    if (room_dev(layer) < 0 && (!v->heat || peek_victim(kind_of(layer), score(v)) < 0)) return 0;
    pthread_mutex_lock(&T.mx);
    int qfull = T.qn >= VKT_QCAP;
    pthread_mutex_unlock(&T.mx);
    return !qfull;
}

/* ---- warm start ------------------------------------------------------------------ */
static const uint32_t *const *g_plan_heat;
static int g_plan_E;
static int plan_cmp(const void *a, const void *b) {
    int x = *(const int *)a, y = *(const int *)b;
    uint32_t hx = g_plan_heat[x / g_plan_E][x % g_plan_E], hy = g_plan_heat[y / g_plan_E][y % g_plan_E];
    return hx < hy ? 1 : hx > hy ? -1 : (x > y) - (x < y);
}
static uint32_t *const *g_heat_hist;
int vkt_plan(int *layers, int *eids, int max) {
    if (!T.on || !g_heat_hist) return 0;
    int L = T.main_layers, E = T.c.experts, n = 0;   /* the history: the main layers' */
    int *order = malloc((size_t)L * E * sizeof(int));
    if (!order) return 0;
    for (int l = 0; l < L; l++)
        if (g_heat_hist[l]) for (int e = 0; e < E; e++) if (g_heat_hist[l][e]) order[n++] = l * E + e;
    g_plan_heat = (const uint32_t *const *)g_heat_hist; g_plan_E = E;
    qsort(order, (size_t)n, sizeof(int), plan_cmp);
    int k = 0;
    for (int i = 0; i < n && k < max && room_dev(0) >= 0; i++) {
        VSlot *v = &T.s[order[i]];
        if (v->state != VS_NONE) continue;
        int dev = room_dev(0);   /* the hottest fill the primary device, the next ones the second */
        v->state = VS_QUEUED; v->dev = (uint8_t)dev; T.queued++; T.dque[dev]++;
        layers[k] = order[i] / E; eids[k] = order[i] % E; k++;
    }
    free(order);
    /* With the chain's device MoE (COLI_VK_CHAIN_MOE=1, qwen36), when the primary device
     * holds every expert, the ones without history come too: a step can then route on
     * the device (vkt_all_resident) from the first token. Otherwise only history counts. */
    const char *dm = getenv("COLI_VK_CHAIN_MOE");
    int cold = 0;
    if (dm && *dm == '1') for (int i = 0; i < L * E; i++) cold += T.s[i].state == VS_NONE;
    if (cold && !T.d2 && T.dmax[0] - (T.dres[0] + T.dque[0]) >= cold && k + cold <= max)
        for (int i = 0; i < L * E; i++) {
            VSlot *v = &T.s[i];
            if (v->state != VS_NONE) continue;
            v->state = VS_QUEUED; v->dev = 0; T.queued++; T.dque[0]++;
            layers[k] = i / E; eids[k] = i % E; k++;
        }
    return k;
}
int vkt_put(int layer, int eid, const VktExpertSrc *src) {
    if (!T.on || layer < 0 || layer >= T.c.layers || eid < 0 || eid >= T.c.experts) return 0;
    ColiVkTensor *t[3] = {NULL, NULL, NULL};
    int dev = slot(layer, eid)->dev;   /* vkt_plan's choice */
    int r = src && src->g && src->u && src->d ? upload(dev, kind_of(layer), src->g, src->u, src->d, src->gs, src->us, src->ds, t) : 0;
    int ok = r > 0;
    push_done(layer, eid, ok, r < 0, t);
    if (ok) __atomic_add_fetch(&T.warm, 1, __ATOMIC_RELAXED);
    return ok;
}
void vkt_put_done(void) { if (T.on) quiesce(); }

/* ---- one layer step -------------------------------------------------------------- */
/* The per-step arrays: assignments (map), groups (bex, brows, bfirst, touched) and
 * rows (bx, by), each family grown together. */
static int grow_to(int need, int *cap) {
    int nc = *cap ? *cap : 64;
    while (nc < need) nc *= 2;
    return nc;
}
static int grow_map(int need) {
    if (need <= T.cmap) return 1;
    int nc = grow_to(need, &T.cmap);
    int *m = realloc(T.map, (size_t)nc * sizeof(int));
    if (!m) return 0;
    T.map = m; T.cmap = nc;
    return 1;
}
static int grow_groups(int need) {
    if (need <= T.cb) return 1;
    int nc = grow_to(need, &T.cb);
    ColiVkExpert **a = realloc(T.bex, (size_t)nc * sizeof(*a)); if (a) T.bex = a;
    int *b = realloc(T.brows, (size_t)nc * sizeof(int)); if (b) T.brows = b;
    int *c = realloc(T.bfirst, (size_t)nc * sizeof(int)); if (c) T.bfirst = c;
    int *d = realloc(T.touched, (size_t)nc * sizeof(int)); if (d) T.touched = d;
    if (!a || !b || !c || !d) return 0;
    T.cb = nc;
    return 1;
}
static int grow_rows(int need) {
    if (need <= T.cbx) return 1;
    int nc = grow_to(need, &T.cbx);
    const float **a = realloc(T.bx, (size_t)nc * sizeof(*a)); if (a) T.bx = a;
    const float **b = realloc(T.by, (size_t)nc * sizeof(*b)); if (b) T.by = b;
    float *c = realloc(T.bw, (size_t)nc * sizeof(*c)); if (c) T.bw = c;
    if (!a || !b || !c) return 0;
    T.cbx = nc;
    return 1;
}

/* ---- streaming: a big prefill step's cold experts on the device -------------------------
 * A prompt chunk of thousands of rows routes most of a layer's experts, each to dozens of
 * rows or more. The resident ones run on the device as always; a cold one (not resident)
 * the CPU would read from its RAM cache or the disk and multiply for every one of its
 * rows. Streaming hands it to the device instead: the engine loads it as its CPU path
 * would (VktConfig.load), the tier copies it into one of a few staging slots (st_n of
 * them, carved out of the budget) and runs its rows there with the tiled GEMM, then the
 * slot is free for the next one. The resident set and its LFRU never see a streamed
 * expert: it is not offered for promotion (a prompt's whole working set passes through
 * the slots and would flush the residents), and its routings count as heat, as every
 * routing does.
 *
 * Such a step runs as sub-batches (backend_vulkan.c's coli_vk_xb_sub_*): the resident
 * experts first, then the streamed ones, at most st_per streamed experts and st_half
 * rows a sub-batch; sub-batch k goes to half k % 2 of the scratch, so while the device
 * computes one the engine thread loads and uploads the experts of the next (double
 * buffering). An expert with more rows than a sub-batch holds is cut into parts.
 *
 * The rule, per cold expert of a step: stream it when it has at least R rows,
 *     R = max(G, ceil(upload / cpu_row)),
 * G the rows from which the device takes the tiled GEMM (COLI_VK_TIER_GEMM_ROWS, 16:
 * fewer rows would run the per-row GEMV), upload the time one expert takes to reach a
 * slot (its bytes at the bandwidth measured over the uploads so far), cpu_row the time
 * the CPU spent per (row, expert) pair it computed in the prefill steps so far (its share
 * of a step between issue and join, loads included). Below R the CPU keeps the expert:
 * its rows cost the CPU less than the upload. Until both are measured R = G.
 * COLI_VK_TIER_STREAM_ROWS=n fixes R at n (tests). A [VK] line per forward says what the
 * rule gave, and the run's line how much streamed. */
static double vkt_now_ms(void);
static int stream_min_rows(void) {
    if (T.st_rows_env > 0) return T.st_rows_env;
    int r = T.gemm_rows;
    if (T.bw_up > 0 && T.cpu_row_ms > 0) {
        double up = (double)T.exp_bytes / T.bw_up, need = up / T.cpu_row_ms;
        if (need > 1e6) need = 1e6;
        int n = (int)need + (need > (int)need);
        if (n > r) r = n;
    }
    return r;
}
int vkt_step_rows(int S, int block) {
    if (T.on && T.st_ok && S >= T.gemm_rows) return S;
    return S < block ? S : block;
}

/* the slots, made at the first step that streams */
static int st_alloc(void) {
    if (T.st) return T.st_n >= 2;
    T.st = calloc((size_t)T.st_slots, sizeof(VStage));
    if (!T.st) { T.st_ok = 0; return 0; }
    int H = T.c.hidden, F = T.c.inter, n = 0;
    for (int i = 0; i < T.st_slots; i++) {
        VStage *v = &T.st[n];
        int ok = 1;
        for (int k = 0; k < 3 && ok; k++) {
            int fmt = k < 2 ? T.form[0].gu_fmt : T.form[0].dn_fmt, gs = k < 2 ? T.form[0].gu_gs : T.form[0].dn_gs;
            uint8_t *rows; size_t stride; float *sc;
            ok = coli_vk_tier_tensor(&v->t[k], fmt, k < 2 ? H : F, k < 2 ? F : H, gs, &rows, &stride, &sc);
            if (!ok) v->t[k] = NULL;
        }
        if (ok) v->ex = coli_vk_xb_expert(v->t[0], v->t[1], v->t[2]);
        if (!ok || !v->ex) {
            for (int k = 0; k < 3; k++) if (v->t[k]) coli_vk_tensor_free(v->t[k]);
            memset(v, 0, sizeof *v);
            break;
        }
        n++;
    }
    T.st_n = n;
    if (n < 2) {
        fprintf(stderr, "[VK] tier %s: no device memory for the streaming slots, cold experts stay on the CPU\n", T.engine);
        T.st_ok = 0;
        return 0;
    }
    if (n < T.st_slots) fprintf(stderr, "[VK] tier %s: %d of %d streaming slots\n", T.engine, n, T.st_slots);
    T.st_per = n / 2;
    return 1;
}
/* Experts' bytes into slots: eids[i] into T.st[slots[i]], ok[i] = 1 when it is there
 * (else the CPU computes it). The engine loads them on this thread, several at once
 * when it can (load_batch: its parallel reads); the conversions into the slots run in
 * parallel, each expert into its own slot. */
static int st_put(int layer, int eid, VStage *v, const VktExpertSrc *src, int par) {
    const int H = T.c.hidden, F = T.c.inter;
    if (!src->g || !src->u || !src->d || (T.form[0].gu_scales && (!src->gs || !src->us || !src->ds))) return 0;
    const uint8_t *codes[3] = {src->g, src->u, src->d}; const void *scs[3] = {src->gs, src->us, src->ds};
    for (int k = 0; k < 3; k++) {
        uint8_t *rows; size_t stride; float *sc;
        if (!coli_vk_tensor_refill(v->t[k], &rows, &stride, &sc)) return 0;
        convert_par(k < 2 ? T.form[0].gu : T.form[0].dn, k < 2 ? H : F, k < 2 ? F : H, codes[k], scs[k], rows, stride, sc, par);
    }
    (void)layer; (void)eid;
    return 1;
}
static void st_fill_group(int layer, const int *eids, const int *slots, int n, uint8_t *ok) {
    VktExpertSrc srcs[64]; void *hs[64];
    if (n > 64) n = 64;
    double ms = 0;
    /* A device upload group can exceed the RAM cache. Consume each cache-sized
     * batch completely before loading the next, which may reuse those RAM slots.
     * A partial batch is capacity, not a reason to serialize all remaining reads. */
    for (int base = 0; base < n;) {
        int left = n - base;
        int got = T.c.load_batch && left >= 2
                ? T.c.load_batch(T.c.load_ctx, layer, eids + base, left, srcs, hs) : 0;
        if (got < 0 || got > left) got = 0;
        if (got) {
            double t0 = vkt_now_ms();
            #pragma omp parallel for schedule(dynamic, 1) if (T.st_par)
            for (int j = 0; j < got; j++)
                ok[base + j] = (uint8_t)st_put(layer, eids[base + j], &T.st[slots[base + j]], &srcs[j], 0);
            ms += vkt_now_ms() - t0;
            for (int j = 0; j < got; j++) T.c.release(T.c.load_ctx, hs[j]);
            base += got;
            continue;
        }
        /* A lone expert, or an engine without a usable batch hook. Its rows can
         * still convert in parallel (COLI_VK_TIER_STREAM_PAR=0 keeps one thread). */
        void *h = NULL; VktExpertSrc src;
        ok[base] = 0;
        if (T.c.load(T.c.load_ctx, layer, eids[base], &src, &h)) {
            double t0 = vkt_now_ms();
            ok[base] = (uint8_t)st_put(layer, eids[base], &T.st[slots[base]], &src, T.st_par);
            ms += vkt_now_ms() - t0;
            T.c.release(T.c.load_ctx, h);
        }
        base++;
    }
    double t0 = vkt_now_ms();
    /* Fill the uploader's bounded staging window across the complete group.
     * Committing each expert separately drains that window and waits for a
     * fence after just three matrices, serializing transfers on discrete GPUs.
     * A failed group is wholly left to the CPU; no partial expert is published. */
    ColiVkTensor *tensors[3 * 64]; int nt = 0;
    for (int i = 0; i < n; i++) if (ok[i])
        for (int k = 0; k < 3; k++) tensors[nt++] = T.st[slots[i]].t[k];
    if (nt && !coli_vk_tensor_commit(tensors, nt)) memset(ok, 0, (size_t)n);
    int done = 0;
    for (int i = 0; i < n; i++) {
        VStage *v = &T.st[slots[i]];
        if (!ok[i]) { v->state = SS_FREE; continue; }
        v->layer = layer; v->eid = eids[i]; v->state = SS_FILLED; v->pending = 0; v->pref = 0;
        done++;
    }
    ms += vkt_now_ms() - t0;
    T.st_up_ms += ms; T.st_bytes += (unsigned long long)done * T.exp_bytes;
    if (done && ms > 0) { double bw = (double)done * T.exp_bytes / ms; T.bw_up = T.bw_up > 0 ? 0.8 * T.bw_up + 0.2 * bw : bw; }
}
/* a sub-batch in flight joined: its rows copied to their places, its slots free */
static int st_join_half(int h) {
    VSub *b = &T.hf[h];
    if (!coli_vk_xb_sub_busy(h)) return 1;
    double dms = 0, t0 = vkt_now_ms();
    int ok = coli_vk_xb_sub_join(h, b->yout, &dms);
    T.wait_ms += vkt_now_ms() - t0;
    T.dev_ms += dms;
    for (int i = 0; i < b->nslot; i++) {
        VStage *v = &T.st[b->slot[i]];
        if (--v->pending <= 0) { v->pending = 0; v->state = SS_FREE; }
    }
    b->n = b->nslot = 0;
    if (!ok) T.big_fail = 1;
    return ok;
}
static int grow_int(int **p, int *cap, int need) {
    if (need <= *cap) return 1;
    int nc = *cap ? *cap : 64; while (nc < need) nc *= 2;
    int *n = realloc(*p, (size_t)nc * sizeof(int));
    if (!n) return 0;
    *p = n; *cap = nc;
    return 1;
}

/* One big step: every resident expert and every cold one the rule streams, as
 * sub-batches; the CPU gets the rest (taken stays 0). The second device's residents run
 * there as one batch beside them, up to d2_rows rows (past that the rule decides). */
enum { BC_NONE = 0, BC_DEV, BC_STREAM, BC_CPU, BC_DEV2 };
/* the second device's batch of a big step: its experts' rows, in its one batch */
static int d2_issue_big(int layer, const float *x, int K, const float *w, const int *cnt, const int *ofs,
                        const int *cls, uint8_t *taken) {
    const int E = T.c.experts, H = T.c.hidden;
    int ne = 0, nr = 0;
    for (int e = 0; e < E; e++) if (cls[e] == BC_DEV2) { ne++; nr += cnt[e]; }
    if (!ne) return 0;
    if (ne > T.d2_cexp) {
        ColiVkExpert **a = realloc(T.d2_bex, (size_t)ne * sizeof(*a)); if (a) T.d2_bex = a;
        int *b = realloc(T.d2_brows, (size_t)ne * sizeof(int)); if (b) T.d2_brows = b;
        if (!a || !b) return 0;
        T.d2_cexp = ne;
    }
    if (nr > T.d2_cap) {
        const float **a = realloc(T.d2_bx, (size_t)nr * sizeof(*a)); if (a) T.d2_bx = a;
        const float **b = realloc(T.d2_y, (size_t)nr * sizeof(*b)); if (b) T.d2_y = b;
        float *c = realloc(T.d2_bw, (size_t)nr * sizeof(float)); if (c) T.d2_bw = c;
        if (!a || !b || !c) return 0;
        T.d2_cap = nr;
    }
    if (!grow_int(&T.d2_idx, &T.cd2, nr)) return 0;
    int c = 0, j = 0;
    for (int e = 0; e < E; e++) {
        if (cls[e] != BC_DEV2) continue;
        T.d2_bex[c] = slot(layer, e)->ex; T.d2_brows[c] = cnt[e]; c++;
        for (int r = 0; r < cnt[e]; r++, j++) {
            int i = T.blist[ofs[e] + r];
            T.d2_bx[j] = x + (size_t)(i / K) * H; T.d2_bw[j] = w ? w[i] : 1.0f; T.d2_idx[j] = i;
        }
    }
    if (!coli_vk_xb_issue_dev(1, T.d2_bex, T.d2_brows, c, T.d2_bx, w ? T.d2_bw : NULL)) {
        if (!coli_vk_xb_ready_dev(1)) d2_stop("the second device's batch failed");
        return 0;
    }
    for (int q = 0; q < j; q++) { taken[T.d2_idx[q]] = 1; T.map[T.d2_idx[q]] = T.d2_idx[q]; }
    T.d2_inflight = 1; T.nd2 = j; T.inflight = 1;
    T.d2_served += (unsigned long long)j; T.d2_steps++; T.d2_big++;
    return j;
}
static int issue_big(int layer, const float *x, int S, int K, const int *idx, const float *w, uint8_t *taken) {
    const int E = T.c.experts, H = T.c.hidden, n = S * K;
    int *cnt = T.bcnt, *ofs = T.bofs, *cls = T.bcls;
    memset(cnt, 0, (size_t)E * sizeof(int));
    for (int i = 0; i < n; i++) {
        int e = idx[i];
        if (e < 0 || e >= E) continue;
        T.routed++;
        VSlot *v = slot(layer, e);
        if (v->heat < 0xFFFFFFFFu) v->heat++;
        v->last = T.tick;
        cnt[e]++;
    }
    /* the rows grouped by expert, each expert's in routing order; its class */
    if (!grow_int(&T.blist, &T.cblist, n) || !grow_map(n)) return 0;
    int at = 0, R = stream_min_rows();
    for (int e = 0; e < E; e++) {
        ofs[e] = at; at += cnt[e];
        VSlot *v = slot(layer, e);
        cls[e] = !cnt[e] ? BC_NONE : v->state == VS_RESIDENT && v->ex ? (v->dev ? BC_DEV2 : BC_DEV)
               : T.st_ok && cnt[e] >= R ? BC_STREAM : BC_CPU;
    }
    /* the second device's one batch holds d2_rows rows; past them the rule decides */
    for (int e = 0, d2r = 0; e < E; e++)
        if (cls[e] == BC_DEV2) {
            if (d2r + cnt[e] > T.d2_rows) cls[e] = T.st_ok && cnt[e] >= R ? BC_STREAM : BC_CPU;
            else d2r += cnt[e];
        }
    int *fill = malloc((size_t)E * sizeof(int));
    if (!fill) return 0;
    memcpy(fill, ofs, (size_t)E * sizeof(int));
    for (int i = 0; i < n; i++) { int e = idx[i]; if (e >= 0 && e < E) T.blist[fill[e]++] = i; }
    free(fill);
    if (T.pred) {   /* the prediction for the next big step at this layer */
        for (int e = 0; e < E; e++) T.pred[(size_t)layer * E + e] = (uint32_t)cnt[e];
        T.pred_ok[layer] = 1;
    }
    int ndev = 0, nst = 0, nkept = 0, kept_rows = 0, nd2 = 0;
    for (int e = 0; e < E; e++) {
        ndev += cls[e] == BC_DEV; nst += cls[e] == BC_STREAM; nd2 += cls[e] == BC_DEV2;
        if (cls[e] == BC_CPU) { nkept++; kept_rows += cnt[e]; }
    }
    /* prefetched slots this step does not stream go back */
    for (int i = 0; T.st && i < T.st_n; i++) {
        VStage *v = &T.st[i];
        if (v->state == SS_FILLED && (v->layer != layer || cls[v->eid] != BC_STREAM)) v->state = SS_FREE;
    }
    if (nst && !st_alloc()) {
        for (int e = 0; e < E; e++) if (cls[e] == BC_STREAM) { cls[e] = BC_CPU; nkept++; kept_rows += cnt[e]; }
        nst = 0;
    }
    if (!T.said_fwd) {
        T.said_fwd = 1;
        char hb[32];
        fprintf(stderr, "[VK] tier %s stream: a step of %d rows; cold experts with %d rows or more go to the device (",
                T.engine, S, R);
        if (T.st_rows_env > 0) fprintf(stderr, "COLI_VK_TIER_STREAM_ROWS");
        else if (T.bw_up > 0 && T.cpu_row_ms > 0)
            fprintf(stderr, "an upload of %s at %.1f GB/s takes %.3f ms, the CPU %.4f ms a row; the GEMM from %d rows",
                    human((double)T.exp_bytes, hb, sizeof hb), T.bw_up / 1e6, (double)T.exp_bytes / T.bw_up,
                    T.cpu_row_ms, T.gemm_rows);
        else fprintf(stderr, "the GEMM's %d rows: the upload and the CPU's rows not measured yet", T.gemm_rows);
        fprintf(stderr, "); layer %d: %d resident, %d streamed, %d kept on the CPU (%d rows)", layer, ndev, nst, nkept, kept_rows);
        if (T.d2) fprintf(stderr, ", %d on the second device", nd2);
        fprintf(stderr, "\n");
    }
    T.st_kept += (unsigned long long)nkept; T.st_kept_rows += (unsigned long long)kept_rows;
    if (!ndev && !nst && !nd2) return 0;
    /* The backend counts both halves, activation buffers, descriptor windows and
     * growth before allocating. Leave half of available RAM for CPU work too. */
    int half = T.st_half < n ? T.st_half : n;
    if (ndev || nst) {
        size_t ram = (size_t)(compat_mem_available_gb() * 1e9);
        half = coli_vk_xb_sub_fit(half, E + 1, ram ? ram / 2 : SIZE_MAX);
        if (half < 1 || !coli_vk_xb_sub_reserve(half, E + 1)) {
            fprintf(stderr, "[VK] tier %s: no scratch for a step of %d rows, its experts stay on the CPU\n", T.engine, S);
            return 0;
        }
    }
    if ((size_t)n * H > T.csy) {
        float *ny = realloc(T.sy, (size_t)n * H * sizeof(float));
        if (!ny) return 0;
        T.sy = ny; T.csy = (size_t)n * H;
    }
    for (int i = 0; i < n; i++) T.map[i] = -1;
    T.big = 1; T.big_fail = 0; T.big_n = n; T.big_taken = 0; T.sk = 0;
    T.big_valid = 0;
    for (int e = 0; e < E; e++) T.big_valid += cnt[e];
    /* the step's token rows once and its output rows in place, where the backend can
     * (not beside the second device: its rows come back as before) */
    T.sy_step = !nd2 && (ndev || nst) ? coli_vk_xb_step_begin(S, K, x) : NULL;
    float *SY = T.sy_step ? T.sy_step : T.sy;
    T.can_balance = 0;
    /* the order: resident experts, then the streamed ones already in a slot, then the rest */
    int *order = malloc((size_t)E * sizeof(int)), no = 0;
    ColiVkExpert **bex = malloc((size_t)(E + 1) * sizeof(*bex));
    int *brows = malloc((size_t)(E + 1) * sizeof(int));
    const float **bx = malloc((size_t)half * sizeof(*bx)); float *bw = malloc((size_t)half * sizeof(float));
    int *bidx = malloc((size_t)half * sizeof(int));
    int *eslot = malloc((size_t)E * sizeof(int));
    int ok = order && bex && brows && bx && bw && bidx && eslot;
    for (int h = 0; h < 2 && ok; h++) {   /* a sub-batch's slots and output places */
        if (T.hf[h].cap < half) {
            float **ny = realloc(T.hf[h].yout, (size_t)half * sizeof(float *));
            if (ny) { T.hf[h].yout = ny; T.hf[h].cap = half; } else ok = 0;
        }
        if (ok && !T.hf[h].slot) ok = (T.hf[h].slot = malloc((size_t)(E + 1) * sizeof(int))) != NULL;
    }
    if (!ok) { free(order); free(bex); free(brows); free(bx); free(bw); free(bidx); free(eslot); T.big = 0; return 0; }
    for (int e = 0; e < E; e++) { eslot[e] = -1; if (cls[e] == BC_DEV) order[no++] = e; }
    for (int i = 0; T.st && i < T.st_n; i++) {
        VStage *v = &T.st[i];
        if (v->state == SS_FILLED && v->layer == layer && cls[v->eid] == BC_STREAM && eslot[v->eid] < 0) {
            eslot[v->eid] = i; order[no++] = v->eid;
            if (v->pref) T.pf_used++;
        }
    }
    for (int e = 0; e < E; e++) if (cls[e] == BC_STREAM && eslot[e] < 0) order[no++] = e;
    /* the second device's batch first, so it runs beside every sub-batch */
    int d2_taken = nd2 ? d2_issue_big(layer, x, K, w, cnt, ofs, cls, taken) : 0;
    /* sub-batches */
    int cnt_b = 0, rows_b = 0, nst_b = 0, total = 0;
    VSub *cur = NULL;
    int curh = 0;
#define SUBMIT_CURRENT() do { \
        if (cnt_b) { \
            int okk = !T.big_fail && (T.sy_step ? coli_vk_xb_sub_issue_step(curh, bex, brows, cnt_b, bidx, bw) \
                                                : coli_vk_xb_sub_issue(curh, bex, brows, cnt_b, bx, bw)); \
            if (okk) { \
                for (int q = 0; q < rows_b; q++) { taken[bidx[q]] = 1; T.map[bidx[q]] = bidx[q]; } \
                total += rows_b; T.st_subs++; T.sk++; \
                T.inflight = 1; \
            } else { \
                T.big_fail = 1; \
                for (int q = 0; q < cur->nslot; q++) { VStage *vv = &T.st[cur->slot[q]]; if (--vv->pending <= 0) { vv->pending = 0; vv->state = SS_FREE; } } \
                cur->nslot = 0; cur->n = 0; \
            } \
        } \
        cnt_b = rows_b = nst_b = 0; cur = NULL; \
    } while (0)
    for (int o = 0; o < no && !T.big_fail; o++) {
        int e = order[o];
        /* An earlier group fill can have reassigned this expert to the CPU.
         * Its entry remains in order[], but has no resident device tensor. */
        if (cls[e] == BC_CPU) continue;
        int sl = -1;
        if (cls[e] == BC_STREAM) {
            if (nst_b >= T.st_per) SUBMIT_CURRENT();
            if (T.big_fail) break;
            if (eslot[e] < 0) {
                /* this expert and the next ones without a slot, as many as the sub-batch has
                 * room for, loaded and uploaded together; free slots first, joining the
                 * oldest sub-batch in flight when there are not enough */
                int room = T.st_per - nst_b, gid[64], gsl[64], gn = 0;
                if (room > 64) room = 64;
                for (int o2 = o; o2 < no && gn < room; o2++)
                    if (cls[order[o2]] == BC_STREAM && eslot[order[o2]] < 0) gid[gn++] = order[o2];
                int nfree = 0;
                for (int tries = 0; tries < 3; tries++) {
                    nfree = 0;
                    for (int i = 0; i < T.st_n && nfree < gn; i++) if (T.st[i].state == SS_FREE) gsl[nfree++] = i;
                    if (nfree >= gn) break;
                    int old = coli_vk_xb_sub_busy(T.sk % 2) ? (int)(T.sk % 2) : (int)((T.sk + 1) % 2);
                    if (cur && old == curh && cnt_b) old = !curh;   /* never the half being filled */
                    if (!coli_vk_xb_sub_busy(old)) break;
                    st_join_half(old);
                }
                if (T.big_fail) break;
                if (nfree < gn) gn = nfree;
                uint8_t gok[64];
                st_fill_group(layer, gid, gsl, gn, gok);
                for (int i = 0; i < gn; i++) {
                    if (gok[i]) eslot[gid[i]] = gsl[i];
                    else { cls[gid[i]] = BC_CPU; T.st_kept++; T.st_kept_rows += (unsigned long long)cnt[gid[i]]; }
                }
                if (cls[e] != BC_STREAM || eslot[e] < 0) {   /* not loaded (or no slot): the CPU's */
                    if (cls[e] == BC_STREAM) { cls[e] = BC_CPU; T.st_kept++; T.st_kept_rows += (unsigned long long)cnt[e]; }
                    continue;
                }
            }
            sl = eslot[e];
            T.st_experts++; T.st_rows += (unsigned long long)cnt[e];
        }
        ColiVkExpert *ex = sl >= 0 ? T.st[sl].ex : slot(layer, e)->ex;
        for (int r0 = 0; r0 < cnt[e] && !T.big_fail; ) {
            if (rows_b && rows_b + (cnt[e] - r0 < half ? cnt[e] - r0 : half) > half) SUBMIT_CURRENT();
            if (T.big_fail) break;
            if (!cur) {   /* a new sub-batch in half sk % 2: the one there before is joined first */
                curh = (int)(T.sk % 2);
                if (coli_vk_xb_sub_busy(curh) && !st_join_half(curh)) break;
                cur = &T.hf[curh]; cur->n = cur->nslot = 0;
            }
            int take = cnt[e] - r0; if (take > half - rows_b) take = half - rows_b;
            bex[cnt_b] = ex; brows[cnt_b] = take; cnt_b++;
            for (int q = 0; q < take; q++) {
                int i = T.blist[ofs[e] + r0 + q];
                bx[rows_b + q] = x + (size_t)(i / K) * H;
                bw[rows_b + q] = w ? w[i] : 1.0f;
                bidx[rows_b + q] = i;
                cur->yout[rows_b + q] = SY + (size_t)i * H;
            }
            rows_b += take; cur->n = rows_b;
            if (sl >= 0) {
                if (!T.st[sl].pending || T.st[sl].state != SS_BUSY) { T.st[sl].state = SS_BUSY; }
                T.st[sl].pending++;
                cur->slot[cur->nslot++] = sl;
                if (r0 == 0) nst_b++;
            }
            r0 += take;
        }
    }
    SUBMIT_CURRENT();
#undef SUBMIT_CURRENT
    free(order); free(bex); free(brows); free(bx); free(bw); free(bidx); free(eslot);
    total += d2_taken;
    T.big_taken = total;
    if (!total && !T.inflight) {   /* nothing went (or everything failed before a submit) */
        T.big = 0;
        if (T.sy_step) { coli_vk_xb_step_end(); T.sy_step = NULL; }
        if (T.big_fail) {
            fprintf(stderr, "[VK] tier %s: a batch failed, its experts are recomputed on the CPU and the tier stops\n", T.engine);
            T.on = 0;
        }
        return 0;
    }
    T.steps++; T.served += (unsigned long long)total; T.st_steps++;
    T.S = S; T.K = K;
    T.t_issued = vkt_now_ms();
    return total;
}
static int join_big(const float **rows) {
    double t0 = vkt_now_ms(), tc = t0 - T.t_issued;
    T.cpu_ms += tc;
    /* the oldest first */
    int a = (int)(T.sk % 2), b = !a;
    st_join_half(a); st_join_half(b);
    int ok1 = 1;
    if (T.d2_inflight) {   /* the second device's rows to their places */
        double dms = 0, t1 = vkt_now_ms();
        ok1 = coli_vk_xb_join_dev(1, T.d2_y, &dms);
        T.wait_ms += vkt_now_ms() - t1; T.dev_ms += dms; T.d2_ms += dms;
        if (ok1) for (int q = 0; q < T.nd2; q++) memcpy((T.sy_step ? T.sy_step : T.sy) + (size_t)T.d2_idx[q] * T.c.hidden, T.d2_y[q], (size_t)T.c.hidden * sizeof(float));
        T.d2_inflight = 0;
    }
    int cpu_pairs = T.big_n - T.big_taken;
    if (cpu_pairs >= 32 && tc > 0) {
        double r = tc / cpu_pairs;
        T.cpu_row_ms = T.cpu_row_ms > 0 ? 0.8 * T.cpu_row_ms + 0.2 * r : r;
    }
    T.inflight = 0; T.big = 0;
    int ok = !T.big_fail;
    const float *SY = T.sy_step ? T.sy_step : T.sy;
    for (int i = 0; i < T.big_n; i++) rows[i] = ok && ok1 && T.map[i] >= 0 ? SY + (size_t)i * T.c.hidden : NULL;
    for (int i = 0; T.st && i < T.st_n; i++) if (T.st[i].state == SS_BUSY) { T.st[i].state = SS_FREE; T.st[i].pending = 0; }
    if (!ok) {
        fprintf(stderr, "[VK] tier %s: a batch failed, its experts are recomputed on the CPU and the tier stops\n", T.engine);
        T.on = 0;
        return 0;
    }
    if (!ok1) {
        d2_stop("a batch on the second device failed");
        return 0;
    }
    quiesce();
    return 1;
}

/* layer's likely streamed experts into the free slots, while the device runs its attention */
int vkt_stream_prefetch(int layer, int S) {
    if (!T.on || !T.st_ok || T.inflight || layer < 0 || layer >= T.main_layers || S < T.gemm_rows) return 0;
    if (!coli_vk_xb_ready() || !st_alloc()) return 0;
    const int E = T.c.experts, K = T.c.topk;
    int R = stream_min_rows();
    /* the slots a step before left filled are free again */
    for (int i = 0; i < T.st_n; i++) if (T.st[i].state != SS_BUSY) T.st[i].state = SS_FREE;
    double *est = malloc((size_t)E * sizeof(double));
    int *cand = malloc((size_t)E * sizeof(int));
    if (!est || !cand) { free(est); free(cand); return 0; }
    double tot = 0;
    if (T.pred && T.pred_ok[layer]) {
        double sum = 0;
        for (int e = 0; e < E; e++) sum += T.pred[(size_t)layer * E + e];
        for (int e = 0; e < E; e++) est[e] = sum > 0 ? (double)T.pred[(size_t)layer * E + e] * S * K / sum : 0;
        tot = sum;
    } else {
        for (int e = 0; e < E; e++) tot += slot(layer, e)->heat;
        for (int e = 0; e < E; e++) est[e] = tot > 0 ? (double)slot(layer, e)->heat * S * K / tot : 0;
    }
    int nc = 0;
    if (tot > 0)
        for (int e = 0; e < E; e++) {
            VSlot *v = slot(layer, e);
            if (v->state == VS_RESIDENT || est[e] < R) continue;
            int at = nc++;
            while (at > 0 && est[cand[at - 1]] < est[e]) { cand[at] = cand[at - 1]; at--; }
            cand[at] = e;
        }
    int got = 0, used = 0;
    if (nc > T.st_n) nc = T.st_n;
    while (used < nc) {   /* groups of a sub-batch's worth, into the slots in order */
        int gn = nc - used < T.st_per ? nc - used : T.st_per, gsl[64]; uint8_t gok[64];
        if (gn > 64) gn = 64;
        for (int i = 0; i < gn; i++) gsl[i] = used + i;
        st_fill_group(layer, cand + used, gsl, gn, gok);
        for (int i = 0; i < gn; i++) if (gok[i]) { T.st[gsl[i]].pref = 1; got++; }
        used += gn;
    }
    T.pf_n += (unsigned long long)got;
    free(est); free(cand);
    return got;
}

static int issue(int layer, const float *x, int S, int K, const int *idx, const float *w, uint8_t *taken) {
    if (S > 0 && K > 0) memset(taken, 0, (size_t)S * K);
    if (!T.on || T.inflight || layer < 0 || layer >= T.c.layers || S < 1 || K < 1) return 0;
    if (!coli_vk_xb_ready()) {
        if (T.on) { fprintf(stderr, "[VK] tier %s: the device stopped answering, the experts stay on the CPU\n", T.engine); T.on = 0; }
        return 0;
    }
    if (T.d2 && !coli_vk_xb_ready_dev(1)) d2_stop("the second device stopped answering");
    if (layer < T.last_layer || T.begin) { T.begin = 0; T.first_layer = layer; new_forward(); }
    if (T.routed_at) {   /* this step's routing: its experts are not offered to the RAM cache's eviction */
        uint32_t q = ++T.seq;
        for (int i = 0; i < S * K; i++)
            if (idx[i] >= 0 && idx[i] < T.c.experts) __atomic_store_n(&T.routed_at[(size_t)layer * T.c.experts + idx[i]], q, __ATOMIC_RELAXED);
        __atomic_store_n(&T.layer_seq[layer], q, __ATOMIC_RELEASE);
    }
    if (layer == T.first_layer) {   /* COLI_VK_TIER_RATE promotions per token of the forward */
        T.tick += (uint32_t)S;
        long cap = (long)T.promo_cap + (long)T.rate * S;
        T.promo_cap = cap > (1 << 30) ? 1 << 30 : (int)cap;
    }
    T.last_layer = layer;
    quiesce();
    if (T.st_ok && S >= T.gemm_rows && layer < T.main_layers) return issue_big(layer, x, S, K, idx, w, taken);
    int E = T.c.experts, H = T.c.hidden, n = S * K;
    if (!grow_map(n)) return 0;
    /* heat for every routed expert, groups for the resident ones (first seen first) */
    int ng = 0, total = 0, rows1 = 0;
    for (int i = 0; i < n; i++) {
        T.map[i] = -1;
        int e = idx[i];
        if (e < 0 || e >= E) continue;
        T.routed++;
        if (layer >= T.main_layers) T.extra_routed++;
        VSlot *v = slot(layer, e);
        if (v->heat < 0xFFFFFFFFu) v->heat++;
        v->last = T.tick;
        if (v->state != VS_RESIDENT) continue;
        if (v->dev && rows1 >= T.d2_rows) continue;    /* the second device's batch is as large as it gets */
        int g = T.grp[e];
        if (g < 0) {
            if (total >= T.max_dev_rows || !grow_groups(ng + 1)) continue;   /* the batch is as large as it gets */
            g = ng++; T.grp[e] = g; T.touched[g] = e; T.bex[g] = v->ex; T.brows[g] = 0;
        } else if (total >= T.max_dev_rows) continue;
        T.brows[g]++; total++; rows1 += v->dev;
        T.map[i] = g;                                   /* group for now, row index below */
    }
    if (!ng) return 0;
    /* The balance: when the device has been the slower side, the experts the CPU
     * also holds in RAM beyond the device's share of the step's rows go back to the
     * CPU, whole experts (the CPU reads one once for all its rows); the ones only
     * the device holds stay (the CPU would read them from disk). */
    T.can_balance = 0;
    if (T.balance) {
        int cap = (int)(T.share * total + 0.999f), kept = 0, keep_n = 0;
        uint8_t *gone = NULL;
        for (int g = 0; g < ng; g++) {
            if (!T.c.in_ram(T.c.ram_ctx, layer, T.touched[g])) { kept += T.brows[g]; continue; }
            T.can_balance = 1;
            /* the first expert always stays: a step with nothing on the device would
             * measure nothing, and the share could never come back up */
            if (!kept || kept + T.brows[g] <= cap) { kept += T.brows[g]; continue; }
            if (!gone && !(gone = calloc((size_t)ng, 1))) break;
            gone[g] = 1;
        }
        if (gone) {
            int *newg = malloc((size_t)ng * sizeof(int));
            if (newg) {
                for (int g = 0; g < ng; g++) {
                    if (gone[g]) { newg[g] = -1; T.grp[T.touched[g]] = -1; T.handed += (unsigned long long)T.brows[g]; total -= T.brows[g]; continue; }
                    newg[g] = keep_n;
                    T.bex[keep_n] = T.bex[g]; T.brows[keep_n] = T.brows[g]; T.touched[keep_n] = T.touched[g]; keep_n++;
                }
                for (int i = 0; i < n; i++) if (T.map[i] >= 0) T.map[i] = newg[T.map[i]];
                ng = keep_n;
                free(newg);
            }
            free(gone);
        }
        if (!ng) return 0;
    }
    /* With a second device: the primary device's groups first, then the second's, so each
     * device's batch is a contiguous run of groups and of rows (rows0 rows on the first). */
    int n0 = ng, rows0 = total;
    if (T.d2) {
        n0 = rows0 = 0;
        for (int g = 0; g < ng; g++) if (!slot(layer, T.touched[g])->dev) { n0++; rows0 += T.brows[g]; }
        if (n0 && n0 < ng) {
            int *newg = malloc((size_t)ng * sizeof(int)), *tr = malloc((size_t)ng * sizeof(int)), *tt = malloc((size_t)ng * sizeof(int));
            ColiVkExpert **tb = malloc((size_t)ng * sizeof(*tb));
            if (!newg || !tr || !tt || !tb) {
                free(newg); free(tr); free(tt); free(tb);
                for (int g = 0; g < ng; g++) T.grp[T.touched[g]] = -1;
                for (int i = 0; i < n; i++) T.map[i] = -1;
                return 0;
            }
            for (int g = 0, a0 = 0, a1 = n0; g < ng; g++) newg[g] = slot(layer, T.touched[g])->dev ? a1++ : a0++;
            for (int g = 0; g < ng; g++) { tb[newg[g]] = T.bex[g]; tr[newg[g]] = T.brows[g]; tt[newg[g]] = T.touched[g]; }
            memcpy(T.bex, tb, (size_t)ng * sizeof(*tb)); memcpy(T.brows, tr, (size_t)ng * sizeof(int));
            memcpy(T.touched, tt, (size_t)ng * sizeof(int));
            for (int i = 0; i < n; i++) if (T.map[i] >= 0) T.map[i] = newg[T.map[i]];
            free(newg); free(tr); free(tt); free(tb);
        }
    }
    /* rows expert by expert, each expert's in routing order */
    if (!grow_rows(total)) {
        for (int g = 0; g < ng; g++) T.grp[T.touched[g]] = -1;
        for (int i = 0; i < n; i++) T.map[i] = -1;
        return 0;
    }
    int at = 0;
    for (int g = 0; g < ng; g++) { T.bfirst[g] = at; at += T.brows[g]; T.brows[g] = 0; }
    for (int i = 0; i < n; i++) {
        int g = T.map[i];
        if (g < 0) continue;
        int j = T.bfirst[g] + T.brows[g]++;
        T.bx[j] = x + (size_t)(i / K) * H;
        T.bw[j] = w ? w[i] : 1.0f;
        T.map[i] = j;
    }
    for (int g = 0; g < ng; g++) T.grp[T.touched[g]] = -1;
    /* one batch per device, both in flight at once */
    T.d0_inflight = n0 > 0 && coli_vk_xb_issue_w(T.bex, T.brows, n0, T.bx, w ? T.bw : NULL);
    T.d2_inflight = 0; T.d2_rows0 = rows0;
    if (n0 < ng) {
        T.d2_inflight = coli_vk_xb_issue_dev(1, T.bex + n0, T.brows + n0, ng - n0, T.bx + rows0, w ? T.bw + rows0 : NULL);
        if (!T.d2_inflight && !coli_vk_xb_ready_dev(1)) d2_stop("the second device's batch failed");
    }
    int took = 0;
    for (int i = 0; i < n; i++) {
        int j = T.map[i];
        if (j < 0) continue;
        if (j < rows0 ? !T.d0_inflight : !T.d2_inflight) { T.map[i] = -1; continue; }   /* not submitted: the CPU's */
        taken[i] = 1; took++;
    }
    if (!took) return 0;
    T.inflight = 1; T.S = S; T.K = K; T.steps++; T.served += (unsigned long long)took;
    if (layer >= T.main_layers) T.extra_served += (unsigned long long)took;
    if (T.d2_inflight) { T.d2_served += (unsigned long long)(total - rows0); T.d2_steps++; }
    T.t_issued = vkt_now_ms();
    return took;
}
int vkt_issue(int layer, const float *x, int S, int K, const int *idx, uint8_t *taken) {
    return issue(layer, x, S, K, idx, NULL, taken);
}
int vkt_issue_w(int layer, const float *x, int S, int K, const int *idx, const float *w, uint8_t *taken) {
    return issue(layer, x, S, K, idx, w, taken);
}

int vkt_join(const float **rows) {
    if (!T.inflight) return 0;
    if (T.big) {
        int ok = join_big(rows);
        if (T.sy_step) coli_vk_xb_step_end();   /* rows stay readable until the next step */
        return ok;
    }
    double t0 = vkt_now_ms(), dms = 0, dms1 = 0, tc = t0 - T.t_issued;
    T.cpu_ms += tc;
    int ok = !T.d0_inflight || coli_vk_xb_join(T.by, &dms);
    int ok1 = !T.d2_inflight || coli_vk_xb_join_dev(1, T.by + T.d2_rows0, &dms1);
    T.d0_inflight = T.d2_inflight = 0;
    double tw = vkt_now_ms() - t0;
    T.wait_ms += tw;
    T.dev_ms += dms + dms1; T.d2_ms += dms1;
    if (dms1 > dms) dms = dms1;   /* the two ran side by side: the step waited for the slower */
    /* The balance moves only on steps where it had a choice. Waiting for more than
     * a tenth of the CPU's own time: give the CPU more. The device done well before
     * the CPU (its timestamps say so; without them, nothing waited): take more. */
    if (T.balance && T.can_balance) {
        if (tw > 0.1 * tc + 0.02) T.share *= 0.92f;
        else if (dms > 0 ? dms < 0.8 * tc : tw < 0.01) T.share = T.share * 1.04f + 0.01f;
        if (T.share > 1.f) T.share = 1.f;
        if (T.share < 0.05f) T.share = 0.05f;
    }
    T.inflight = 0;
    int n = T.S * T.K;
    for (int i = 0; i < n; i++) rows[i] = ok && ok1 && T.map[i] >= 0 ? T.by[T.map[i]] : NULL;
    if (!ok) {
        fprintf(stderr, "[VK] tier %s: a batch failed, its experts are recomputed on the CPU and the tier stops\n", T.engine);
        T.on = 0;
        return 0;
    }
    if (!ok1) {   /* the step's device rows are recomputed on the CPU (join's 0), the tier goes on */
        d2_stop("a batch on the second device failed");
        return 0;
    }
    quiesce();
    return 1;
}

/* vkt_join, and for a big step that ran wholly on the device (every routed assignment
 * taken) on the backend's whole-step buffers, the routed sum per row on the device:
 * *sum = S rows of sum over k in rank order of w[s*K+k] * expert(s, k), as an fma chain
 * from 0 (the engines' own loop); else *sum = NULL and rows as vkt_join gives them. */
int vkt_join_sum(const float **rows, const float *w, const float **sum) {
    *sum = NULL;
    if (!T.inflight) return 0;
    if (!T.big || !T.sy_step) return vkt_join(rows);
    int ok = join_big(rows);
    if (ok && T.big_taken == T.big_valid && T.big_valid > 0) {
        uint8_t *use = malloc((size_t)T.big_n);
        if (use) {
            for (int i = 0; i < T.big_n; i++) use[i] = T.map[i] >= 0;
            double dms = 0, t0 = vkt_now_ms();
            *sum = coli_vk_xb_step_sum(w, use, &dms);
            T.wait_ms += vkt_now_ms() - t0; T.dev_ms += dms;
            if (*sum) T.st_sums++;
            free(use);
        }
    }
    coli_vk_xb_step_end();
    return ok;
}

/* Every expert of every layer resident on the primary device and nothing on its way in
 * or out: a caller may then route on the device and use the experts' entries for the
 * whole step (no promotion can start, so no eviction either). */
int vkt_all_resident(void) {
    if (!T.on || T.inflight || T.d2) return 0;
    return T.resident == T.c.layers * T.c.experts;
}
const void *vkt_expert(int layer, int eid) {
    if (!T.on || layer < 0 || layer >= T.c.layers || eid < 0 || eid >= T.c.experts) return NULL;
    VSlot *v = slot(layer, eid);
    return v->state == VS_RESIDENT && !v->dev ? v->ex : NULL;
}
/* The routings of a step that ran on the device, for the heat (the LFRU's counts). */
void vkt_note_routed(int layer, const int *idx, int n) {
    if (!T.on || layer < 0 || layer >= T.c.layers) return;
    for (int i = 0; i < n; i++) {
        if (idx[i] < 0 || idx[i] >= T.c.experts) continue;
        VSlot *v = slot(layer, idx[i]);
        if (v->heat < 0xFFFFFFFFu) v->heat++;
        v->last = T.tick;
        T.routed++; T.served++;
    }
}

void vkt_begin_forward(void) { if (T.on) T.begin = 1; }
int vkt_resident(int layer, int eid) {
    return T.on && layer >= 0 && layer < T.c.layers && eid >= 0 && eid < T.c.experts &&
           slot(layer, eid)->state == VS_RESIDENT;
}
int vkt_ready(void) { return T.on; }

/* ---- budget ---------------------------------------------------------------------- */
static size_t mem_available(void) { return (size_t)(compat_mem_available_gb() * 1e9); }
static double env_gb(const char *name, double def) {
    const char *e = getenv(name);
    return e && *e ? atof(e) : def;
}

int vkt_wanted(void) {
    const char *on = getenv("COLI_VK_TIER");
    return !(on && *on == '0');
}

/* A form of the experts: the device formats and the bytes of one expert as the engine
 * holds it. 0 when it has no device form. */
static int form_init(int k, VktFmt gu, VktFmt dn) {
    __typeof__(T.form[0]) *fm = &T.form[k];
    const int H = T.c.hidden, F = T.c.inter;
    if (!dev_fmt(gu, &fm->gu_fmt, &fm->gu_gs) || !dev_fmt(dn, &fm->dn_fmt, &fm->dn_gs)) return 0;
    fm->gu = gu; fm->dn = dn;
    fm->gu_codes = src_row_bytes(gu.kind, H) * (size_t)F;
    fm->d_codes = src_row_bytes(dn.kind, F) * (size_t)H;
    fm->gu_scales = src_scale_bytes(gu, H, F);
    fm->d_scales = src_scale_bytes(dn, F, H);
    fm->stage_bytes = 2 * fm->gu_codes + fm->d_codes + 2 * fm->gu_scales + fm->d_scales;
    return 1;
}

/* COLI_VK_DEV2: the second device's share. Its budget is what it has free less
 * COLI_VK_RESERVE2_GB (0.5 GiB: its batch's scratch, the driver), an integrated GPU's a
 * quarter of what the RAM leaves (as the primary's), capped by COLI_VK_EXPERTS2 experts
 * when set; it holds at most the experts the primary device does not. */
static void d2_init(long long fit0, long long all, int act) {
    if (!getenv("COLI_VK_DEV2") || !*getenv("COLI_VK_DEV2")) return;
    const char *eng = T.engine;
    if (!coli_vk_dev2_open_env()) {
        fprintf(stderr, "[VK] tier %s: COLI_VK_DEV2=%s brought no second device up, the tier stays on %s\n",
                eng, getenv("COLI_VK_DEV2"), coli_vk_device_name());
        return;
    }
    ColiVkDevInfo di;
    if (!coli_vk_dev_info(1, &di) || !coli_vk_xb_init_dev(1, T.c.hidden, T.c.inter, act, T.c.act_limit, T.c.act_a, T.c.act_b)) {
        fprintf(stderr, "[VK] tier %s: no expert batch on the second device (shaders?), the tier stays on %s\n",
                eng, coli_vk_device_name());
        return;
    }
    snprintf(T.d2_name, sizeof T.d2_name, "%s", di.name ? di.name : "?");
    const double GiB = 1073741824.0;
    T.d2_exp_bytes = expert_bytes_at(T.c.hidden, T.c.inter, T.c.gate_up, T.c.down, di.buf_align ? di.buf_align : 256);
    double heap_free = di.has_budget ? (double)di.free_bytes : (double)di.local_bytes * 0.8;
    double room = heap_free - env_gb("COLI_VK_RESERVE2_GB", 0.5) * GiB, want = room;
    if (di.shares_ram) {
        double avail = (double)mem_available();
        double ram_room = avail > 0 ? avail - (double)T.c.ram_reserve - (double)T.c.dense_bytes - 2.0 * GiB : room;
        if (ram_room / 4 < want) want = ram_room / 4;
    }
    if (want < 0) want = 0;
    size_t blk = coli_vk_block_bytes_dev(1, (size_t)256 << 20);
    if ((size_t)want < blk) blk = (size_t)want;
    long long fit = 0;
    if (T.d2_exp_bytes && blk >= T.d2_exp_bytes) {
        size_t full = (size_t)want / blk, rest = (size_t)want - full * blk;
        fit = (long long)full * (long long)(blk / T.d2_exp_bytes) + (long long)(rest / T.d2_exp_bytes);
    }
    long long fit_raw = fit, left = all - fit0;
    const char *cap = getenv("COLI_VK_EXPERTS2");
    if (cap && *cap && atoll(cap) >= 0 && fit > atoll(cap)) fit = atoll(cap);
    if (fit > left) fit = left;
    char hb[32], he[32];
    if (fit < 1) {
        fprintf(stderr, "[VK] tier %s: second device %s: %s, the tier stays on %s\n", eng, T.d2_name,
                left < 1 ? "the primary device holds every expert" : "no room for experts (COLI_VK_RESERVE2_GB sets its reserve)",
                coli_vk_device_name());
        return;
    }
    /* a pool no bigger than its experts when they all fit in one block (a small model) */
    size_t lim = (size_t)want, need = (size_t)(fit + 1) * T.d2_exp_bytes;
    if (fit_raw >= left && need < lim && need <= coli_vk_block_bytes_dev(1, (size_t)256 << 20)) lim = need;
    coli_vk_tier_pool_limit_dev(1, lim);
    T.d2 = 1; T.dmax[1] = (int)fit; T.max_resident += (int)fit; T.d2_budget = (size_t)want;
    /* a big step's rows on it: their x rows within 128 MiB */
    long hr = (128L << 20) / ((long)T.c.hidden * 4);
    T.d2_rows = hr < 64 ? 64 : hr > 65535 ? 65535 : (int)hr;
    fprintf(stderr, "[VK] tier %s: second device %s, budget %s = %d experts of %s (%s), the experts after the primary device's\n",
            eng, T.d2_name, human((double)T.d2_budget, hb, sizeof hb), T.dmax[1], human((double)T.d2_exp_bytes, he, sizeof he),
            di.shares_ram ? "shared RAM" : "device memory");
}

int vkt_init(const VktConfig *cfg, uint32_t *const *heat) {
    if (T.on || !cfg) return 0;
    if (!vkt_wanted()) return 0;
    if (!coli_vk_available()) return 0;
    const char *eng = cfg->engine ? cfg->engine : "engine";
    memset(&T, 0, sizeof T);
    T.c = *cfg;
    snprintf(T.engine, sizeof T.engine, "%s", eng);
    if (T.c.layers < 1 || T.c.experts < 1 || T.c.hidden < 1 || T.c.inter < 1 || T.c.topk < 1) return 0;
    T.main_layers = T.c.layers;
    if (!form_init(0, T.c.gate_up, T.c.down)) {
        fprintf(stderr, "[VK] tier %s: expert format %d/%d (gs %d/%d) has no device form, the experts stay on the CPU\n",
                eng, T.c.gate_up.kind, T.c.down.kind, T.c.gate_up.gs, T.c.down.gs);
        return 0;
    }
    int act = T.c.act == VKT_ACT_SITU ? COLI_VK_ACT_SITU
            : T.c.act == VKT_ACT_SWIGLU_V4 ? COLI_VK_ACT_SWIGLU_V4 : COLI_VK_ACT_SWIGLU;
    if (!coli_vk_xb_init(T.c.hidden, T.c.inter, act, T.c.act_limit, T.c.act_a, T.c.act_b)) {
        fprintf(stderr, "[VK] tier %s: no expert batch on this device (shaders?), the experts stay on the CPU\n", eng);
        return 0;
    }
    int H = T.c.hidden, F = T.c.inter;
    T.exp_bytes = vkt_expert_bytes(H, F, T.c.gate_up, T.c.down);
    /* the extra layers (an MTP head's): on the tier when their form has a device form */
    if (T.c.extra_layers > 0) {
        if (form_init(1, T.c.extra_gate_up, T.c.extra_down) && T.exp_bytes &&
            (T.x_bytes = vkt_expert_bytes(H, F, T.c.extra_gate_up, T.c.extra_down)))
            T.c.layers += T.c.extra_layers;
        else
            fprintf(stderr, "[VK] tier %s: the extra layers' expert format %d/%d (gs %d/%d) has no device form, their experts stay on the CPU\n",
                    eng, T.c.extra_gate_up.kind, T.c.extra_down.kind, T.c.extra_gate_up.gs, T.c.extra_down.gs);
    }
    if (T.main_layers == T.c.layers) T.c.extra_layers = 0;

    /* The budget. Discrete: what VK_EXT_memory_budget says is free on the device,
     * minus a reserve (scratch, KV mirrors, driver) and the dense weights still to
     * come. Integrated (device memory is host RAM): a quarter of what the RAM has
     * left after the engine's expert cache has grown to its size and the dense
     * weights are placed, so the tier never takes what the CPU cache needs.
     * COLI_VK_TIER_GB replaces either, within what the device can hold. */
    const double GiB = 1073741824.0;
    double used = 0, bud = 0;
    int have = coli_vk_mem_budget(&used, &bud);
    double heap_free = have ? (bud - used) * 1e9 : (double)coli_vk_device_local_bytes() * 0.8;
    T.uma = coli_vk_device_shares_ram();
    double reserve = env_gb("COLI_VK_TIER_RESERVE_GB", 1.0) * GiB;
    double room = heap_free - reserve - (double)T.c.dense_bytes;
    double want;
    const char *cap = getenv("COLI_VK_TIER_GB");
    if (cap && *cap) want = atof(cap) * GiB;
    else if (T.uma) {
        double avail = (double)mem_available();
        double ram_room = avail > 0 ? avail - (double)T.c.ram_reserve - (double)T.c.dense_bytes - 2.0 * GiB : room;
        want = ram_room / 4;
    } else want = room;
    if (want > room) want = room;
    if (want < 0) want = 0;
    /* whole experts per pool block, so the count matches what the blocks can hold */
    size_t blk = coli_vk_block_bytes((size_t)256 << 20);   /* the pool's blocks (smaller under COLI_VK_DEVICE_CAP_MB) */
    if ((size_t)want < blk) blk = (size_t)want;
    long long fit = 0;
    if (T.exp_bytes && blk >= T.exp_bytes) {
        size_t full = (size_t)want / blk, rest = (size_t)want - full * blk;
        fit = (long long)full * (long long)(blk / T.exp_bytes) + (long long)(rest / T.exp_bytes);
    }
    long long all = (long long)T.main_layers * T.c.experts, fit_raw = fit;
    /* The extra layers' pool: all their experts when the budget holds them beside every
     * main one, else their layers' share of the budget (at least one expert, when two
     * main ones still fit beside it). Its room, in main experts, leaves the main count. */
    if (T.c.extra_layers) {
        long long xall = (long long)T.c.extra_layers * T.c.experts, xn;
        double bud = (double)fit_raw * T.exp_bytes;
        if (bud >= (double)all * T.exp_bytes + (double)xall * T.x_bytes) xn = xall;
        else xn = (long long)(bud * T.c.extra_layers / (T.main_layers + T.c.extra_layers) / T.x_bytes);
        if (xn < 1 && bud >= (double)T.x_bytes + 2.0 * T.exp_bytes) xn = 1;
        if (xn > xall) xn = xall;
        if (xn < 1) {
            char hx[32];
            fprintf(stderr, "[VK] tier %s: no room for an extra layer's expert (%s), they stay on the CPU\n",
                    eng, human((double)T.x_bytes, hx, sizeof hx));
            T.c.layers = T.main_layers; T.c.extra_layers = 0;
        } else {
            long long take = (long long)(((size_t)xn * T.x_bytes + T.exp_bytes - 1) / T.exp_bytes);
            T.xmax = (int)xn; T.x_budget = (size_t)take * T.exp_bytes; fit_raw -= take;
        }
    }
    if (fit > all) fit = all;
    if (fit > fit_raw) fit = fit_raw;
    if (fit < 2) {
        char hb[32], he[32];
        fprintf(stderr, "[VK] tier %s: no room for experts (budget %s, %s each; COLI_VK_TIER_GB sets it), the experts stay on the CPU\n",
                eng, human(want, hb, sizeof hb), human((double)T.exp_bytes, he, sizeof he));
        return 0;
    }
    if (T.c.max_experts > 0 && fit > T.c.max_experts) fit = T.c.max_experts;   /* the engine's count cap */
    /* Streaming (big prefill steps, see "streaming" above): COLI_VK_TIER_STREAM=0 turns
     * it off; it needs the engine's load hook and the GEMM route. Its staging slots come
     * out of the budget: the residents give up what the budget cannot hold beside them. */
    {
        ColiVkXbStats xs; coli_vk_xb_stats(&xs);
        T.gemm_rows = xs.gemm_rows;
        const char *se = getenv("COLI_VK_TIER_STREAM"), *sn = getenv("COLI_VK_TIER_STREAM_SLOTS");
        const char *sr = getenv("COLI_VK_TIER_STREAM_ROWS");
        int want_st = !(se && *se == '0') && T.c.load && T.c.release && T.gemm_rows > 0;
        int slots = sn && *sn ? atoi(sn) : 64;
        if (slots > fit_raw / 2) slots = (int)(fit_raw / 2);
        if (want_st && slots >= 2) {
            if (fit > fit_raw - slots) fit = fit_raw - slots;
            T.st_ok = fit >= 2;
            if (!T.st_ok) fit = fit_raw > all ? all : fit_raw;
        }
        if (T.st_ok) {
            T.st_slots = slots;
            T.st_rows_env = sr && *sr ? atoi(sr) : 0;
            /* a sub-batch's rows: its x rows within 32 MiB (a card without Resizable BAR
             * keeps them in its host-visible window) */
            long hr = (32L << 20) / ((long)T.c.hidden * 4);
            T.st_half = hr < 1 ? 1 : hr > 65535 ? 65535 : (int)hr;
            const char *sp = getenv("COLI_VK_TIER_STREAM_PAR");
            T.st_par = !(sp && *sp == '0');
            const char *sh = getenv("COLI_VK_TIER_STREAM_HALF");   /* tests: small sub-batches */
            if (sh && *sh && atoi(sh) > 0) T.st_half = atoi(sh) > 65535 ? 65535 : atoi(sh);
            T.bcnt = malloc((size_t)T.c.experts * sizeof(int)); T.bofs = malloc((size_t)T.c.experts * sizeof(int));
            T.bcls = malloc((size_t)T.c.experts * sizeof(int));
            T.pred = calloc((size_t)T.c.layers * T.c.experts, sizeof(uint32_t)); T.pred_ok = calloc((size_t)T.c.layers, 1);
            if (!T.bcnt || !T.bofs || !T.bcls || !T.pred || !T.pred_ok) T.st_ok = 0;
        } else if (want_st)
            fprintf(stderr, "[VK] tier %s: no room for streaming slots beside the residents, cold experts stay on the CPU\n", eng);
    }
    T.max_resident = (int)fit;
    T.dmax[0] = (int)fit;
    T.budget = (size_t)want;
    /* Every expert fits in one block's worth: the pool's limit (and so its one block)
     * is what they take, not the budget, so a small model does not hold a 256 MB
     * block for a few experts. */
    size_t lim = T.budget - T.x_budget, need = (size_t)(all + 1 + (T.st_ok ? T.st_slots : 0)) * T.exp_bytes;
    if (fit + (T.st_ok ? T.st_slots : 0) >= all && fit_raw >= all + (T.st_ok ? T.st_slots : 0) &&
        need < lim && need <= coli_vk_block_bytes((size_t)256 << 20)) lim = need;
    coli_vk_tier_pool_limit(lim);
    if (T.xmax) coli_vk_tier_extra_pool_limit(T.x_budget);
    const char *ex = getenv("COLI_VK_TIER_EXCLUSIVE");
    T.excl = !(ex && *ex == '0');
    if (T.excl) {
        T.routed_at = calloc((size_t)T.c.layers * T.c.experts, sizeof(uint32_t));
        T.layer_seq = calloc((size_t)T.c.layers, sizeof(uint32_t));
        if (!T.routed_at || !T.layer_seq) { free(T.routed_at); free(T.layer_seq); T.routed_at = T.layer_seq = NULL; T.excl = 0; }
    }
    const char *bal = getenv("COLI_VK_TIER_BALANCE");
    T.balance = T.c.in_ram != NULL && !(bal && *bal == '0');
    T.share = 1.f;
    const char *r = getenv("COLI_VK_TIER_RATE");
    T.rate = r && *r ? atoi(r) : 16;
    if (T.rate < 0) T.rate = 0;
    T.s = calloc((size_t)T.c.layers * T.c.experts, sizeof(VSlot));
    T.grp = malloc((size_t)T.c.experts * sizeof(int));
    if (!T.s || !T.grp) { free(T.s); free(T.grp); return 0; }
    for (int e = 0; e < T.c.experts; e++) T.grp[e] = -1;
    T.max_dev_rows = T.c.max_rows > 0 ? T.c.max_rows : 1 << 20;
    T.last_layer = 1 << 30; T.first_layer = -1; T.decay_at = VKT_DECAY_TOKENS;
    /* The history, scaled: the hottest expert of a layer starts at 32 and the rest in
     * proportion, so a few minutes of a new workload can displace it (raw counts
     * from a long history would hold the tier for hours). */
    g_heat_hist = heat;
    if (heat)
        for (int l = 0; l < T.main_layers; l++) {   /* the history has the main layers */
            if (!heat[l]) continue;
            uint32_t mx = 0;
            for (int e = 0; e < T.c.experts; e++) if (heat[l][e] > mx) mx = heat[l][e];
            if (mx) for (int e = 0; e < T.c.experts; e++)
                if (heat[l][e]) T.s[(size_t)l * T.c.experts + e].heat = 1 + (uint32_t)(31.0 * heat[l][e] / mx);
        }
    if (pthread_mutex_init(&T.mx, NULL) || pthread_cond_init(&T.cv, NULL) || pthread_cond_init(&T.cv_room, NULL) ||
        pthread_cond_init(&T.cv_done, NULL)) return 0;
    { const char *sy = getenv("COLI_VK_TIER_SYNC"); T.sync = sy && *sy == '1'; }
    if (pthread_create(&T.th, NULL, uploader, NULL)) return 0;
    T.th_on = 1;
    T.on = 1;
    char hb[32], he[32];
    fprintf(stderr, "[VK] tier %s: on, %s, budget %s = %d experts of %s (fmt %d", eng,
            coli_vk_device_name(), human((double)T.budget, hb, sizeof hb), T.dmax[0],
            human((double)T.exp_bytes, he, sizeof he), T.form[0].gu_fmt);
    if (T.form[0].gu_gs) fprintf(stderr, " gs %d", T.form[0].gu_gs);
    fprintf(stderr, ", down fmt %d", T.form[0].dn_fmt);
    if (T.form[0].dn_gs) fprintf(stderr, " gs %d", T.form[0].dn_gs);
    fprintf(stderr, "), %s, %s queue, up to %d promotions per token%s%s",
            T.uma ? (cap && *cap ? "shared RAM (COLI_VK_TIER_GB)" : "shared RAM: a quarter of what the expert cache leaves")
                  : "device memory", coli_vk_xb_queue_shared() ? "shared" : "own", T.rate,
            T.balance ? ", balanced against the CPU" : "", T.sync ? ", uploads awaited (COLI_VK_TIER_SYNC)" : "");
    if (T.st_ok) fprintf(stderr, ", streaming cold experts of prompt steps through %d slots", T.st_slots);
    if (T.xmax) {
        char hx[32], hp[32];
        fprintf(stderr, "; the extra layers' experts (%d, fmt %d): %d of %s in a pool of %s", T.c.extra_layers,
                T.form[1].gu_fmt, T.xmax, human((double)T.x_bytes, hx, sizeof hx), human((double)T.x_budget, hp, sizeof hp));
    }
    fprintf(stderr, "\n");
    d2_init(fit, all, act);   /* COLI_VK_DEV2: the experts after these on a second device */
    return 1;
}

void vkt_report(const char *scope, unsigned long long ram_hits, unsigned long long disk_loads) {
    if (!T.s) return;
    ColiVkPoolStats ps; coli_vk_pool_stats(1, &ps);
    unsigned long long r = T.routed - T.rep_routed, sv = T.served - T.rep_served;
    T.rep_routed = T.routed; T.rep_served = T.served;
    double hidden = T.dev_ms > T.wait_ms ? T.dev_ms - T.wait_ms : 0;
    char hu[32], hb[32], hl[32];
    fprintf(stderr, "[VK] tier %s %s: device %llu of %llu routed experts (%.1f%%; this %s %llu of %llu) | "
            "CPU RAM hits %llu, disk loads %llu | resident %d (budget %d, %s of %s, %d blocks, frag %.2f) | "
            "uploads %llu (%s, %llu warm), evictions %llu, skipped %llu queue + %llu rate, failed %llu | "
            "device %.1f ms, CPU share %.1f ms, waited %.1f ms (%.0f%% of device time hidden)",
            T.engine, scope, T.served, T.routed, T.routed ? 100.0 * T.served / T.routed : 0.0, scope, sv, r,
            ram_hits, disk_loads, T.dres[0], T.dmax[0], human((double)ps.used, hu, sizeof hu),
            human((double)T.budget, hb, sizeof hb), ps.blocks, ps.frag, T.uploads,
            human((double)T.upload_bytes, hl, sizeof hl), T.warm, T.evictions, T.qfull, T.rated, T.failed,
            T.dev_ms, T.cpu_ms, T.wait_ms, T.dev_ms > 0 ? 100.0 * hidden / T.dev_ms : 0.0);
    if (T.balance) fprintf(stderr, " | balance: device share %.2f, %llu rows handed to the CPU", T.share, T.handed);
    if (T.c.extra_layers)
        fprintf(stderr, " | extra layers (%d): %llu of %llu routed experts on the device, resident %d (budget %d)",
                T.c.extra_layers, T.extra_served, T.extra_routed, T.xres, T.xmax);
    if (T.excl) fprintf(stderr, " | exclusive: %llu RAM copies of device experts given up first",
                        __atomic_load_n(&T.ram_gave, __ATOMIC_RELAXED));
    if (T.st_ok && T.st_steps) {
        char hs[32];
        fprintf(stderr, " | stream: %llu steps, %llu cold experts (%s, %.2f GB/s) and %llu rows streamed in %llu sub-batches, "
                "prefetched %llu (%llu used), %llu cold experts (%llu rows) kept on the CPU",
                T.st_steps, T.st_experts, human((double)T.st_bytes, hs, sizeof hs),
                T.st_up_ms > 0 ? (double)T.st_bytes / T.st_up_ms / 1e6 : 0.0, T.st_rows, T.st_subs, T.pf_n, T.pf_used,
                T.st_kept, T.st_kept_rows);
        if (T.st_sums) fprintf(stderr, ", %llu steps summed on the device", T.st_sums);
    }
    if (T.d2 || T.d2_lost) {
        ColiVkPoolStats p2; coli_vk_pool_stats(3, &p2);
        char h2[32], b2[32];
        fprintf(stderr, " | second device %s: resident %d (budget %d, %s of %s), %llu rows in %llu batches (%llu beside prompt steps), "
                "uploads %llu, %.1f ms%s", T.d2_name, T.dres[1], T.dmax[1], human((double)p2.used, h2, sizeof h2),
                human((double)T.d2_budget, b2, sizeof b2), T.d2_served, T.d2_steps, T.d2_big, T.d2_uploads, T.d2_ms,
                T.d2 ? "" : " (stopped)");
    }
    ColiVkXbStats xs; coli_vk_xb_stats(&xs);
    if (xs.cooperative_matmuls)
        fprintf(stderr, " | cooperative matmuls: %llu", xs.cooperative_matmuls);
    fprintf(stderr, "\n");
}

int vkt_devices(void) { return !T.on ? 0 : T.d2 ? 2 : 1; }
int vkt_layers(void) { return T.on ? T.c.layers : 0; }
static VSlot *g_old_slots;   /* the slot table of the tier shut down last (vkt_shutdown) */
static uint32_t *g_old_routed, *g_old_seq;

/* The engine's prefetch workers ask too, while the engine thread changes the slot table:
 * the state byte is read atomically (a stale answer costs one eviction order or one
 * read, never a wrong expert: the engine's own cache decides what its slots hold). */
int vkt_ram_first(int layer, int eid) {
    VSlot *s = __atomic_load_n(&T.s, __ATOMIC_ACQUIRE);
    uint32_t *routed = __atomic_load_n(&T.routed_at, __ATOMIC_ACQUIRE), *seq = __atomic_load_n(&T.layer_seq, __ATOMIC_ACQUIRE);
    int L = T.c.layers, E = T.c.experts;
    if (!__atomic_load_n(&T.on, __ATOMIC_RELAXED) || !T.excl || !s || !routed || !seq ||
        layer < 0 || layer >= L || eid < 0 || eid >= E) return 0;
    size_t i = (size_t)layer * E + eid;
    if (__atomic_load_n(&routed[i], __ATOMIC_RELAXED) == __atomic_load_n(&seq[layer], __ATOMIC_ACQUIRE)) return 0;
    return __atomic_load_n(&s[i].state, __ATOMIC_RELAXED) == VS_RESIDENT;
}
void vkt_ram_gave(void) { __atomic_add_fetch(&T.ram_gave, 1, __ATOMIC_RELAXED); }

void vkt_shutdown(void) {
    if (!T.s) return;
    if (T.th_on) {
        pthread_mutex_lock(&T.mx); T.stop = 1; pthread_cond_broadcast(&T.cv); pthread_cond_broadcast(&T.cv_room); pthread_cond_broadcast(&T.cv_done); pthread_mutex_unlock(&T.mx);
        pthread_join(T.th, NULL);
        T.th_on = 0;
    }
    int big = T.inflight && T.big;
    if (big) { st_join_half(0); st_join_half(1); }
    else if (T.inflight && T.d0_inflight) coli_vk_xb_join(T.by, NULL);
    if (T.d2_inflight) coli_vk_xb_join_dev(1, big ? T.d2_y : T.by + T.d2_rows0, NULL);   /* the second device's batch */
    T.inflight = T.big = T.d0_inflight = T.d2_inflight = 0;
    for (int i = 0; i < T.qn; i++) free(T.q[(T.qh + i) % VKT_QCAP].buf);
    T.qn = 0;
    if (coli_vk_available()) {
        quiesce();
        size_t n = (size_t)T.c.layers * T.c.experts;
        for (size_t i = 0; i < n; i++) if (T.s[i].ex) { coli_vk_xb_expert_free(T.s[i].ex); T.s[i].ex = NULL; }
        for (int i = 0; T.st && i < T.st_n; i++) if (T.st[i].ex) coli_vk_xb_expert_free(T.st[i].ex);
    }
    pthread_mutex_destroy(&T.mx); pthread_cond_destroy(&T.cv); pthread_cond_destroy(&T.cv_room); pthread_cond_destroy(&T.cv_done);
    /* the slot table outlives the tier by one shutdown: a prefetch worker may be in
     * vkt_ram_first with the pointer it read before this */
    free(g_old_slots); g_old_slots = T.s;
    free(g_old_routed); g_old_routed = T.routed_at;   /* the same for the routing marks */
    free(g_old_seq); g_old_seq = T.layer_seq;
    free(T.grp); free(T.map); free(T.touched); free(T.bex); free(T.brows); free(T.bfirst);
    free(T.bx); free(T.by); free(T.bw); free(T.evict); free(T.done);
    free(T.st); free(T.sy); free(T.bcnt); free(T.bofs); free(T.bcls); free(T.blist); free(T.pred); free(T.pred_ok);
    for (int h = 0; h < 2; h++) { free(T.hf[h].slot); free(T.hf[h].yout); }
    free(T.d2_idx); free(T.d2_y); free(T.d2_bex); free(T.d2_brows); free(T.d2_bx); free(T.d2_bw);
    memset(&T, 0, sizeof T);
}
#endif /* COLI_VULKAN */
