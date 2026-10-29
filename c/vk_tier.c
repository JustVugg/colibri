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
#include "ggml_quants.h"
/* the tag of the tier's lines: cu_tier.c compiles this file again for a CUDA device as "[CUDA]" */
#ifndef VKT_TAG
#define VKT_TAG "[VK]"
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

#define VKT_QCAP 32                 /* promotions staged and waiting for the uploader */
#define VKT_CAND 64                 /* victim candidates kept per refresh */
#define VKT_DECAY_TOKENS 1024u      /* tier.h's decay period, in tokens */
/* T.full's promotions (promote): from steps of up to this many rows (decode, a verify; a
 * prompt chunk routes nearly every expert once), and a newcomer beats the coldest resident
 * by this many routings. Qwen3.8 IQ2_XS on an RTX PRO 4000, a frozen history: without the
 * margin the prompt that wrote the history lost 4% to swaps that found nothing more on the
 * card; with it that prompt keeps its speed and another topic its gain. */
#define VKT_UP_ROWS 8
#define VKT_UP_MARGIN 2u

enum { VS_NONE = 0, VS_QUEUED, VS_RESIDENT, VS_EVICT };

typedef struct {
    ColiVkExpert *ex;
    uint32_t heat, last;
    uint8_t state;
    uint8_t dev;                       /* the device it is on or on its way to (0, or 1: COLI_VK_DEV2's) */
    uint8_t host;                      /* dev 0 with its tensors in the host pool (coli_vk_tier_host_pool): the
                                        * primary device reads them over PCIe, in a batch and in the dense chain */
    uint8_t up;                        /* T.full: on its way from the host pool to the device (promote) */
    ColiVkExpert *hex;                 /* T.full: its copy in the host pool (ex itself while host) */
} VSlot;
/* dev: the place (place_of); src: a promotion's host pool copy, copied up instead of buf */
typedef struct { int layer, eid, dev; uint8_t *buf; ColiVkExpert *src; } VQ;
/* hg, hu, hd: vkt_put's copy in the host pool beside a device's (T.full) */
typedef struct { int layer, eid, ok, copy_failed; ColiVkTensor *g, *u, *d, *hg, *hu, *hd; } VDone;
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
     * main_layers on; [2..nform) the main layers' other forms when their formats differ
     * by layer (VktConfig.layer_gate_up/layer_down), lform[l] naming each layer's. Then
     * the main layers fall in two classes of rooms (lcls[l]): 0 those of the form most
     * layers have, their tensors at their own size, 1 the others, each tensor in the room
     * of its role's biggest form among them (pad_w, pad_s [class][role]: [0] gate and up,
     * [1] down). A newcomer displaces a resident of its own class (vcls_of), so a freed
     * expert's ranges fit the one that takes its place. */
    int main_layers;
    struct { VktFmt gu, dn; int gu_fmt, gu_gs, dn_fmt, dn_gs; size_t gu_codes, gu_scales, d_codes, d_scales, stage_bytes; } form[VKT_FORMS];
    int nform, per_layer;
    uint8_t lform[VKT_MAX_LAYERS], lcls[VKT_MAX_LAYERS];
    size_t pad_w[2][2], pad_s[2][2];
    int h_exact; size_t h_bytes, h_max;   /* the host pool's experts at their own forms' sizes (host_init): the average, the largest */
    /* the extra layers' experts: on the primary device, in a pool of their own (another
     * size than the main ones', so neither leaves holes the other cannot use), at most
     * xmax of x_bytes each in x_budget */
    int xmax, xres, xque;
    size_t x_bytes, x_budget;
    unsigned long long extra_served, extra_routed;
    size_t exp_bytes, budget;
    int max_resident, resident, queued, rate, uma;   /* totals over the devices */
    /* per place: the most it holds, resident, queued. 0 the primary device's memory, 1 the
     * second device's (dmax[1] = 0 without one), 2 the host pool (dmax[2] = 0 without it) */
    int dmax[3], dres[3], dque[3];
    size_t h_budget;                   /* the host pool's bytes */
    /* Every expert in the host pool (host_init): the device's experts keep a copy there, so
     * the routed ones the device reads from the pool move up to its memory as a conversation
     * routes them (promote) and the coldest there give way by going back to their copy. full:
     * on; promote_rate: the most a forward (COLI_VK_TIER_PROMOTE); ups, up_victims: done, and
     * residents that gave way; dfree: the device copies of those, freed at a quiescent point;
     * vram_routed: routed pairs the dense chain found in the device's memory */
    int full, promote_rate, upromos;
    ColiVkExpert **dfree; int ndfree, cdfree;
    unsigned long long ups, up_victims, vram_routed;
    unsigned long long dec_routed, dec_vram;   /* the same in steps of up to VKT_UP_ROWS rows (decode, a verify) */
    size_t stage_room;                 /* left out of the budget for a prompt chunk's staging (VktConfig.stage) */
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
    int cand[3][VKT_CAND]; uint64_t cand_score[3][VKT_CAND]; int ncand[3];   /* per class (vcls_of): main, extra, the main layers' other forms */
    /* the step in flight */
    int inflight, S, K;
    int *map; int cmap;
    int *grp; int *touched;              /* expert -> group of this step, and the groups' experts */
    ColiVkExpert **bex; int *brows; int *bfirst; int cb;
    const float **bx, **by; int *rowsrc; int cbx;
    float *bw;                           /* the rows' route weights (vkt_issue_w), beside bx */
    /* hybrid steps (hybrid: COLI_VK_HYBRID, SwiGLU or DeepSeek V4's, a host pool): in a step of up to
     * VKT_UP_ROWS rows, the pairs of the host pool's experts of block types the CPU computes
     * in place at the join (vkt_host_pairs) while the device computes the ones it holds,
     * instead of the device reading them over PCIe; hy_of[i] pair i's place among them (-1
     * none), hy_v their rows, hy_x their input rows, hy_w their route weights (VKT_ACT_SWIGLU_V4
     * applies them as the device does), hy_out their outputs; hy_xb the input rows copied at
     * the issue (hy_keep_x: an engine may free x once the issue returns, as DeepSeek V4 does),
     * hy_src their sources, hy_nx how many */
    int hybrid, hy_n, chy, cho; int *hy_of;
    int hy_big;                          /* a prompt step's host pool experts of at most hy_big rows on the CPU too (0: none) */
    unsigned long long hyb_steps, hyb_pairs;
    VktHostExpert *hy_v; const VktHostExpert **hy_hx; const float **hy_x;
    float *hy_gu, *hy_hh, *hy_out, *hy_w, *hy_xb; const float **hy_src; int hy_nx, hy_xcap;
    unsigned long long hy_steps, hy_pairs; double hy_ms;
    int max_dev_rows;
    double t_issued;
    /* the balance: the share of a step's resident experts the device takes, moved by
     * what each join waited (the device was the slower side) or did not */
    int balance, can_balance; float share; unsigned long long handed;
    unsigned epoch;                      /* bumps when a resident comes or goes (vkt_epoch) */
    /* The slots whose entries changed (note_change), published together with an epoch at a
     * quiescent point (publish), and the log of what each epoch changed (vkt_dev_changes),
     * whole from log_from on: a chain updates those entries of its table and reads no other */
    int *pend; int npend, cpend, pend_all;
    int *log_slot; unsigned *log_ep; int nlog, clog; unsigned log_from;
    unsigned long long chain_served;     /* pairs the dense chain computed (vkt_dev_count) */
    /* accounting */
    unsigned long long routed, served, steps, uploads, upload_bytes, evictions, qfull, rated, refused, failed, warm;
    /* failed, by what: the pool refused the expert (no room as it is laid out), the copy
     * to the device failed, or the batch could not take the uploaded expert's sets */
    unsigned long long f_pool, f_copy, f_sets;
    int plan_hist;                        /* the last plan's experts from the history (vkt_plan) */
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
    /* pool_copy (COLI_VK_TIER_POOL_COPY): a big step's host pool experts of block types go
     * through the slots too, copied there on the device (st_copy_group) instead of read in
     * place by the GEMMs; bpool[e] marks them in the step */
    int pool_copy; uint8_t *bpool;
    unsigned long long pc_experts; double pc_bytes;
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
static int kind_of(int layer) { return layer >= T.main_layers; }   /* main (0) or extra (1): the pool */
/* where a slot's tensors are or go: its device, or 2 for the primary device's host pool */
static int place_of(const VSlot *v) { return v->host ? 2 : v->dev; }
static void set_place(VSlot *v, int pl) { v->dev = (uint8_t)(pl == 2 ? 0 : pl); v->host = (uint8_t)(pl == 2); }
static int form_of(int layer) { return T.per_layer && layer < T.main_layers ? T.lform[layer] : kind_of(layer); }
/* the class of the residents a newcomer of `layer` displaces: 0 the main layers of the most
 * common form, 2 those of the other forms, 1 the extra layers (each class's rooms alike) */
static int vcls_of(int layer) { return kind_of(layer) ? 1 : T.per_layer && T.lcls[layer] ? 2 : 0; }
/* a main tensor's room (forms_per_layer): its class's, 0 with one form (its own size) */
static size_t pad_w_of(int layer, int down) { return T.per_layer && !kind_of(layer) ? T.pad_w[T.lcls[layer]][down] : 0; }
static size_t pad_s_of(int layer, int down) { return T.per_layer && !kind_of(layer) ? T.pad_s[T.lcls[layer]][down] : 0; }
/* the room of any main expert: a staging slot refilled with every form */
static size_t pad_w_max(int down) { return T.pad_w[0][down] > T.pad_w[1][down] ? T.pad_w[0][down] : T.pad_w[1][down]; }
static size_t pad_s_max(int down) { return T.pad_s[0][down] > T.pad_s[1][down] ? T.pad_s[0][down] : T.pad_s[1][down]; }

/* ---- source formats ---------------------------------------------------------- */
static int src_per_row(VktSrc k) {
    return k == VKT_SRC_I8_ROW || k == VKT_SRC_I8_AS_I4_ROW || k == VKT_SRC_I4S_PAIRS_ROW || k == VKT_SRC_I4U_PAIRS_ROW;
}
static int src_lowbit(VktSrc k) { return k == VKT_SRC_LB_I2G32 || k == VKT_SRC_LB_I2G16 || k == VKT_SRC_LB_I3G16 || k == VKT_SRC_LB_VQ2; }
static size_t src_lb_qbytes(VktSrc k, int I) { return (size_t)I / (k == VKT_SRC_LB_VQ2 ? 64 : 32); }   /* group scales a row */
static size_t src_row_bytes(VktSrc k, int I) {
    if (k >= VKT_SRC_GGML) return coli_vk_tensor_row_bytes(100 + (int)(k - VKT_SRC_GGML), I);   /* as the device holds it */
    if (src_lowbit(k)) return (size_t)I * (k == VKT_SRC_LB_I3G16 ? 3 : 2) / 8;
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
    if (f.kind == VKT_SRC_BF16 || f.kind == VKT_SRC_F32 || f.kind >= VKT_SRC_GGML) return 0;
    if (src_lowbit(f.kind)) return (size_t)O * (src_lb_qbytes(f.kind, I) + 4);   /* group scales, then f32 row scales */
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
    if (f.kind >= VKT_SRC_GGML) { *fmt = 100 + (int)(f.kind - VKT_SRC_GGML); *gs = 0; return coli_vk_block_elems(*fmt) > 0; }
    if (src_lowbit(f.kind)) { *fmt = f.kind == VKT_SRC_LB_I2G32 ? 200 : f.kind == VKT_SRC_LB_I2G16 ? 201 : f.kind == VKT_SRC_LB_VQ2 ? 203 : 202; *gs = 0; return 1; }
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
    if (src_lowbit(f.kind)) {   /* device row: codes | group scales | f32 row scale (blockdec.glsl) */
        const uint8_t *q = scales; const size_t qb = src_lb_qbytes(f.kind, I);
        const uint8_t *d = q + (size_t)O * qb;
        #pragma omp parallel for schedule(static) if (par)
        for (int o = 0; o < O; o++) {
            uint8_t *dst = rows + (size_t)o * stride;
            memcpy(dst, codes + (size_t)o * rb, rb);
            memcpy(dst + rb, q + (size_t)o * qb, qb);
            memcpy(dst + rb + qb, d + (size_t)o * 4, 4);
        }
        sc[0] = 1.0f;
        return;
    }
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
    if (f.kind == VKT_SRC_BF16 || f.kind == VKT_SRC_F32 || f.kind >= VKT_SRC_GGML) { sc[0] = 1.0f; return; }
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
static int upload(int dev, int layer, const uint8_t *g, const uint8_t *u, const uint8_t *d,
                  const void *gs, const void *us, const void *ds, ColiVkTensor *t[3]) {
    const int H = T.c.hidden, F = T.c.inter, kind = kind_of(layer);
    const __typeof__(T.form[0]) *fm = &T.form[form_of(layer)];
    const uint8_t *codes[3] = {g, u, d}; const void *sc[3] = {gs, us, ds};
    t[0] = t[1] = t[2] = NULL;
    for (int k = 0; k < 3; k++) {
        VktFmt f = k < 2 ? fm->gu : fm->dn;
        int fmt = k < 2 ? fm->gu_fmt : fm->dn_fmt, fgs = k < 2 ? fm->gu_gs : fm->dn_gs;
        int I = k < 2 ? H : F, O = k < 2 ? F : H;
        uint8_t *rows; size_t stride; float *scales;
        if (!(kind ? coli_vk_tier_tensor_extra(&t[k], fmt, I, O, fgs, &rows, &stride, &scales)   /* its own pool */
                   : coli_vk_tier_tensor_pad(dev, &t[k], fmt, I, O, fgs, dev == 2 && T.h_exact ? 0 : pad_w_of(layer, k == 2),
                                             dev == 2 && T.h_exact ? 0 : pad_s_of(layer, k == 2), &rows, &stride, &scales))) {
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
static void push_done(int layer, int eid, int ok, int copy_failed, ColiVkTensor *t[3], ColiVkTensor *h[3]);
/* A promotion (promote): the expert's host pool copy to new device tensors, by the device's
 * copy engine. Refused for room, it waits for the victim's free as an upload does. */
static void promote_copy(const VQ *e) {
    ColiVkTensor *src[3], *t[3] = {NULL, NULL, NULL};
    coli_vk_xb_expert_tensors(e->src, &src[0], &src[1], &src[2]);
    int ok = 0, copy_failed = 0;
    for (int frees = 0, waited = 0; ; ) {
        int k = 0;
        for (; k < 3; k++) if (!src[k] || !coli_vk_tier_tensor_like(0, &t[k], src[k], pad_w_of(e->layer, k == 2), pad_s_of(e->layer, k == 2))) break;
        if (k == 3) { ok = 1; break; }
        for (int j = 0; j < k; j++) { coli_vk_tensor_free(t[j]); t[j] = NULL; }
        if (!src[k] || frees >= 3 || waited >= 600 || T.sync) break;
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
    if (ok && !coli_vk_tensor_copy(t, src, 3)) {
        for (int k = 0; k < 3; k++) { coli_vk_tensor_free(t[k]); t[k] = NULL; }
        ok = 0; copy_failed = 1;
    }
    push_done(e->layer, e->eid, ok, copy_failed, t, NULL);
}
static void push_done(int layer, int eid, int ok, int copy_failed, ColiVkTensor *t[3], ColiVkTensor *h[3]) {
    pthread_mutex_lock(&T.mx);
    if (T.ndone == T.cdone) {
        int nc = T.cdone ? 2 * T.cdone : 64;
        VDone *n = realloc(T.done, (size_t)nc * sizeof(*n));
        if (!n) {   /* cannot even record it: the slot stays queued and is never used */
            T.busy = 0; pthread_cond_broadcast(&T.cv_done);
            pthread_mutex_unlock(&T.mx);
            for (int k = 0; k < 3; k++) { if (t[k]) coli_vk_tensor_free(t[k]); if (h && h[k]) coli_vk_tensor_free(h[k]); }
            return;
        }
        T.done = n; T.cdone = nc;
    }
    T.done[T.ndone++] = (VDone){layer, eid, ok, copy_failed, t[0], t[1], t[2], h ? h[0] : NULL, h ? h[1] : NULL, h ? h[2] : NULL};
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
        if (e.src) { promote_copy(&e); continue; }
        const uint8_t *b = e.buf;
        const int fi = form_of(e.layer);
        const size_t guc = T.form[fi].gu_codes, dc = T.form[fi].d_codes, gus = T.form[fi].gu_scales;
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
            r = upload(e.dev, e.layer, g, u, d, gs, us, ds, t);
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
        push_done(e.layer, e.eid, ok, r < 0, t, NULL);
    }
}

/* ---- quiescent points: nothing in flight ---------------------------------------- */
#define VKT_LOG_MAX 65536
static void note_change(int i) {
    if (T.pend_all) return;
    if (T.npend == T.cpend) {
        int nc = T.cpend ? 2 * T.cpend : 256;
        int *n = nc <= VKT_LOG_MAX ? realloc(T.pend, (size_t)nc * sizeof(int)) : NULL;
        if (!n) { T.pend_all = 1; return; }
        T.pend = n; T.cpend = nc;
    }
    T.pend[T.npend++] = i;
}
/* The changes noted since the last epoch become the next one's; a change that could not be
 * noted, or a log grown past its bound, starts the log again from that epoch. */
static void publish(void) {
    if (!T.npend && !T.pend_all) return;
    T.epoch++;
    int all = T.pend_all || T.nlog + T.npend > VKT_LOG_MAX;
    if (!all && T.nlog + T.npend > T.clog) {
        int nc = T.clog ? T.clog : 1024;
        while (nc < T.nlog + T.npend) nc *= 2;
        int *a = realloc(T.log_slot, (size_t)nc * sizeof(int));
        if (a) T.log_slot = a;
        unsigned *b = a ? realloc(T.log_ep, (size_t)nc * sizeof(unsigned)) : NULL;
        if (b) { T.log_ep = b; T.clog = nc; } else all = 1;
    }
    if (all) { T.nlog = 0; T.log_from = T.epoch; }
    else for (int k = 0; k < T.npend; k++) { T.log_slot[T.nlog] = T.pend[k]; T.log_ep[T.nlog++] = T.epoch; }
    T.npend = 0; T.pend_all = 0;
}
/* T.full: the device copies of residents that went back to the host pool (promote), freed
 * when nothing is in flight: an upload waiting for their room goes on. */
static void free_victims(void) {
    if (T.inflight || !T.ndfree) return;
    for (int i = 0; i < T.ndfree; i++) coli_vk_xb_expert_free(T.dfree[i]);
    T.ndfree = 0;
    pthread_mutex_lock(&T.mx);
    T.room_gen++; T.evict_pending = T.nevict;
    pthread_cond_broadcast(&T.cv_room);
    pthread_mutex_unlock(&T.mx);
}
static void quiesce(void) {
    if (T.inflight) return;
    int freed = 0, had = T.nevict + T.ndfree;
    for (int i = 0; i < T.nevict; i++) {
        VSlot *v = &T.s[T.evict[i]];
        if (v->ex) { coli_vk_xb_expert_free(v->ex); freed = 1; }
        if (v->hex && v->hex != v->ex) coli_vk_xb_expert_free(v->hex);
        v->ex = v->hex = NULL;
        v->state = VS_NONE;
        note_change(T.evict[i]);
    }
    T.nevict = 0;
    /* the device copies of residents that went back to the host pool (promote): their
     * entries changed then, published below with the rest */
    for (int i = 0; i < T.ndfree; i++) { coli_vk_xb_expert_free(T.dfree[i]); freed = 1; }
    T.ndfree = 0;
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
    /* a device whose batch is still in flight cannot take an expert's sets now: what was
     * uploaded there waits for the next quiet point, rather than being freed and counted
     * failed (#2081) */
    const int busy[2] = {coli_vk_xb_busy_dev(0), T.d2 ? coli_vk_xb_busy_dev(1) : 0};
    int kept = 0;
    for (int i = 0; i < n; i++) {
        VSlot *v = slot(d[i].layer, d[i].eid);
        int dv = v->dev, pl = place_of(v), kd = kind_of(d[i].layer);
        if (d[i].ok && (dv == 0 || T.d2) && busy[dv ? 1 : 0]) { d[kept++] = d[i]; continue; }
        if (v->up) {   /* a promotion from the host pool: the device's copy takes the slot, the pool's stays */
            v->up = 0; T.queued--; T.dque[0]--;
            ColiVkExpert *e = d[i].ok ? coli_vk_xb_expert(d[i].g, d[i].u, d[i].d) : NULL;
            if (e) {
                v->ex = e; v->host = 0; T.dres[0]++; T.dres[2]--; T.ups++;
                note_change(d[i].layer * T.c.experts + d[i].eid);
                T.uploads++; T.upload_bytes += coli_vk_tensor_bytes(d[i].g) + coli_vk_tensor_bytes(d[i].u) + coli_vk_tensor_bytes(d[i].d);
                continue;
            }
            for (int k = 0; k < 3; k++) { ColiVkTensor *t = k == 0 ? d[i].g : k == 1 ? d[i].u : d[i].d; if (t) coli_vk_tensor_free(t); }
            T.failed++;
            if (d[i].ok) T.f_sets++; else if (d[i].copy_failed) T.f_copy++; else T.f_pool++;
            if (!d[i].ok && !d[i].copy_failed && T.dmax[0] > T.dres[0] + T.dque[0]) {
                T.max_resident -= T.dmax[0] - (T.dres[0] + T.dque[0]);
                T.dmax[0] = T.dres[0] + T.dque[0];
            }
            continue;
        }
        T.queued--; if (kd) T.xque--; else T.dque[pl]--;
        /* an upload to a second device that has stopped meanwhile (d2_stop) is dropped */
        ColiVkExpert *e = d[i].ok && (dv == 0 || T.d2) ? coli_vk_xb_expert(d[i].g, d[i].u, d[i].d) : NULL;
        /* T.full: a device's expert with its copy in the host pool (vkt_put); one put in the pool is its own copy */
        ColiVkExpert *h = e && d[i].hg ? coli_vk_xb_expert(d[i].hg, d[i].hu, d[i].hd) : NULL;
        if (!h) for (int k = 0; k < 3; k++) { ColiVkTensor *t = k == 0 ? d[i].hg : k == 1 ? d[i].hu : d[i].hd; if (t) coli_vk_tensor_free(t); }
        if (e) {
            v->hex = h ? h : T.full && v->host ? e : NULL;
            v->ex = e; v->state = VS_RESIDENT; T.resident++; if (kd) T.xres++; else T.dres[pl]++;
            note_change(d[i].layer * T.c.experts + d[i].eid);
            T.uploads++; T.upload_bytes += coli_vk_tensor_bytes(d[i].g) + coli_vk_tensor_bytes(d[i].u) + coli_vk_tensor_bytes(d[i].d);
            if (dv) T.d2_uploads++;
        } else {
            for (int k = 0; k < 3; k++) { ColiVkTensor *t = k == 0 ? d[i].g : k == 1 ? d[i].u : d[i].d; if (t) coli_vk_tensor_free(t); }
            v->state = VS_NONE; set_place(v, 0);
            if (dv && !T.d2) continue;
            T.failed++;
            if (d[i].ok) T.f_sets++; else if (d[i].copy_failed) T.f_copy++; else T.f_pool++;
            /* refused: the budget holds fewer than the count said (fragmentation, or the
             * device ran out first): stop planning past what is there. A failed copy gave
             * its room back: the budget stands, the expert may come again. */
            if (!d[i].copy_failed && kd && T.xmax > T.xres + T.xque) T.xmax = T.xres + T.xque;
            if (!d[i].copy_failed && !kd && T.dmax[pl] > T.dres[pl] + T.dque[pl]) {
                T.max_resident -= T.dmax[pl] - (T.dres[pl] + T.dque[pl]);
                T.dmax[pl] = T.dres[pl] + T.dque[pl];
            }
        }
    }
    if (kept) {   /* back in front of what finished meanwhile, in the order they finished */
        pthread_mutex_lock(&T.mx);
        if (T.ndone + kept > T.cdone) {
            VDone *g = realloc(T.done, (size_t)(T.ndone + kept) * sizeof(*g));
            if (g) { T.done = g; T.cdone = T.ndone + kept; }
        }
        if (T.ndone + kept <= T.cdone) {
            memmove(T.done + kept, T.done, (size_t)T.ndone * sizeof(*T.done));
            memcpy(T.done, d, (size_t)kept * sizeof(*d));
            T.ndone += kept;
            kept = 0;
        }
        pthread_mutex_unlock(&T.mx);
        for (int i = 0; i < kept; i++) {   /* no memory to keep them: as before, freed and counted */
            for (int k = 0; k < 3; k++) { ColiVkTensor *t = k == 0 ? d[i].g : k == 1 ? d[i].u : d[i].d; if (t) coli_vk_tensor_free(t); }
            for (int k = 0; k < 3; k++) { ColiVkTensor *t = k == 0 ? d[i].hg : k == 1 ? d[i].hu : d[i].hd; if (t) coli_vk_tensor_free(t); }
            VSlot *v = slot(d[i].layer, d[i].eid);
            int kd = kind_of(d[i].layer);
            T.failed++; T.f_sets++;
            if (v->up) { v->up = 0; T.queued--; T.dque[0]--; continue; }   /* a promotion: it stays in the pool */
            T.queued--; if (kd) T.xque--; else T.dque[place_of(v)]--;
            v->state = VS_NONE; set_place(v, 0);
        }
    }
    free(d);
    publish();
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
    T.max_resident -= T.dmax[1]; T.dmax[1] = 0; T.pend_all = 1; publish();
    fprintf(stderr, VKT_TAG " tier %s: %s, its experts go back to the CPU and the tier goes on with %s\n",
            T.engine, why, coli_vk_device_name());
}

/* The place a newcomer of layer `layer` goes to when there is room: a main layer's to
 * the primary device first, then the second one, then the host pool (2); an extra
 * layer's to its pool on the primary device; -1 when they are full. */
static int room_dev(int layer) {
    if (kind_of(layer)) return T.xres + T.xque < T.xmax ? 0 : -1;
    if (T.dres[0] + T.dque[0] < T.dmax[0]) return 0;
    if (T.d2 && T.dres[1] + T.dque[1] < T.dmax[1]) return 1;
    if (T.dres[2] + T.dque[2] < T.dmax[2]) return 2;
    return -1;
}

/* ---- placement ------------------------------------------------------------------ */
static uint64_t score(const VSlot *v) { return tier_lfru_score(v->heat, v->last, T.tick); }

/* The coldest residents of each kind, ascending, once per forward (T.full: on the device,
 * the host pool holding every expert anyway). */
static void refresh_candidates(void) {
    T.ncand[0] = T.ncand[1] = T.ncand[2] = 0;
    size_t n = (size_t)T.c.layers * T.c.experts;
    for (size_t i = 0; i < n; i++) {
        VSlot *v = &T.s[i];
        if (v->state != VS_RESIDENT || (T.full && (v->host || !v->hex))) continue;
        int k = vcls_of((int)(i / (size_t)T.c.experts)), *c = T.cand[k];
        uint64_t sc = score(v), *cs = T.cand_score[k];
        if (T.ncand[k] == VKT_CAND && sc >= cs[VKT_CAND - 1]) continue;
        int at = T.ncand[k] < VKT_CAND ? T.ncand[k]++ : VKT_CAND - 1;
        while (at > 0 && cs[at - 1] > sc) { c[at] = c[at - 1]; cs[at] = cs[at - 1]; at--; }
        c[at] = (int)i; cs[at] = sc;
    }
}
/* The resident a newcomer of class k and score hs would displace, or -1: the coldest
 * candidate of its class (vcls_of: its own pool, or rooms of its size) still resident, when the newcomer
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
        if (v->state != VS_RESIDENT || (T.full && v->host)) { drop_victim(k); continue; }
        uint64_t cs = score(v);
        /* T.full: a swap copies nothing back and no CPU time goes into it, so any hotter one wins */
        return hs > (T.full ? cs + ((uint64_t)VKT_UP_MARGIN << 8) : cs + (cs >> 2) + (4u << 8)) ? i : -1;
    }
    return -1;
}

/* The resident a newcomer of `layer` with score hs displaces, or -1, and *from its class:
 * one of its own class first; failing that, with forms by layer, one of the other main
 * class whose rooms hold its tensors (a smaller form's newcomer in a bigger form's room,
 * never the other way: the freed ranges would not hold it). */
static int rooms_hold(int a, int b) {   /* pad class a's rooms hold pad class b's tensors */
    for (int r = 0; r < 2; r++) if (T.pad_w[a][r] < T.pad_w[b][r] || T.pad_s[a][r] < T.pad_s[b][r]) return 0;
    return 1;
}
static int pick_victim(int layer, uint64_t hs, int *from) {
    const int k = vcls_of(layer);
    int v = peek_victim(k, hs);
    *from = k;
    if (v >= 0 || k == 1 || !T.per_layer) return v;
    const int o = k == 0 ? 2 : 0;
    if (!rooms_hold(o == 2, k == 2)) return -1;
    *from = o;
    return peek_victim(o, hs);
}

/* A forward starts when the layer index goes back (a layer that issues several
 * blocks of rows, or rows one by one, is the same forward). The tick counts tokens:
 * the rows of the forward's first layer, every block of them. */
static void new_forward(void) {
    T.promos[0] = T.promos[1] = 0; T.promo_cap = 0; T.said_fwd = 0; T.upromos = 0;
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
    const __typeof__(T.form[0]) *fm = &T.form[form_of(layer)];
    if (fm->gu_scales && (!src->gs || !src->us || !src->ds)) return;
    int victim = -1, dev = room_dev(layer), vc = vcls_of(layer);
    /* T.full: the device's residents give way to promote's, which keeps their host pool copy */
    if (dev < 0 && (T.full || !v->heat || (victim = pick_victim(layer, score(v), &vc)) < 0)) return;
    if (victim >= 0) dev = place_of(&T.s[victim]);   /* the newcomer takes its place, on its device or in the host pool */
    pthread_mutex_lock(&T.mx);
    int qfull = T.qn >= VKT_QCAP;
    pthread_mutex_unlock(&T.mx);
    if (qfull) { T.qfull++; return; }
    uint8_t *buf = malloc(fm->stage_bytes);
    if (!buf) return;
    if (victim >= 0) drop_victim(vc);
    uint8_t *p = buf;
    memcpy(p, src->g, fm->gu_codes); p += fm->gu_codes;
    memcpy(p, src->u, fm->gu_codes); p += fm->gu_codes;
    memcpy(p, src->d, fm->d_codes);  p += fm->d_codes;
    if (fm->gu_scales) { memcpy(p, src->gs, fm->gu_scales); p += fm->gu_scales; memcpy(p, src->us, fm->gu_scales); p += fm->gu_scales; }
    if (fm->d_scales) memcpy(p, src->ds, fm->d_scales);
    if (victim >= 0) {
        VSlot *w = &T.s[victim];
        w->state = VS_EVICT; T.resident--; if (kd) T.xres--; else T.dres[place_of(w)]--; T.evictions++;
        if (T.nevict == T.cevict) {
            int nc = T.cevict ? 2 * T.cevict : 64;
            int *n = realloc(T.evict, (size_t)nc * sizeof(int));
            if (!n) { w->state = VS_RESIDENT; T.resident++; if (kd) T.xres++; else T.dres[place_of(w)]++; T.evictions--; free(buf); return; }
            T.evict = n; T.cevict = nc;
        }
        T.evict[T.nevict++] = victim;
        pthread_mutex_lock(&T.mx); T.evict_pending = T.nevict; pthread_mutex_unlock(&T.mx);
        quiesce();   /* nothing in flight: free it right away; else at the join */
    }
    v->state = VS_QUEUED; set_place(v, dev); T.queued++; if (kd) T.xque++; else T.dque[dev]++; T.promos[kd]++;
    pthread_mutex_lock(&T.mx);
    T.q[(T.qh + T.qn) % VKT_QCAP] = (VQ){layer, eid, dev, buf, NULL}; T.qn++;
    pthread_cond_signal(&T.cv);
    pthread_mutex_unlock(&T.mx);
}

/* vkt_note's decision without its copy: the same tests, in the same order. */
int vkt_wants(int layer, int eid) {
    if (!T.on || layer < 0 || layer >= T.c.layers || eid < 0 || eid >= T.c.experts) return 0;
    VSlot *v = slot(layer, eid);
    if (v->state != VS_NONE || T.promos[kind_of(layer)] >= T.promo_cap) return 0;
    int from;
    if (room_dev(layer) < 0 && (T.full || !v->heat || pick_victim(layer, score(v), &from) < 0)) return 0;
    pthread_mutex_lock(&T.mx);
    int qfull = T.qn >= VKT_QCAP;
    pthread_mutex_unlock(&T.mx);
    return !qfull;
}

/* T.full: a routed expert the device reads from the host pool moves up to its memory, into
 * a free place or the coldest resident's there (refresh_candidates), which goes back to its
 * host pool copy at once: nothing is copied back. Up to T.promote_rate a forward. The bytes
 * go from the pool to the device by its copy engine on the uploader's thread (promote_copy)
 * while the engine goes on; the expert keeps its host pool entry until they landed (quiesce).
 * Only where the victim's device copy is read by nothing in flight before it is freed: the
 * dense chain's books after its frame (vkt_dev_count), or a batch's step after its issue. */
static void promote(int layer, int e) {
    VSlot *v = slot(layer, e);
    if (!T.full || v->state != VS_RESIDENT || !v->host || v->up || !v->hex || kind_of(layer) || T.upromos >= T.promote_rate) return;
    int victim = -1, from = vcls_of(layer);
    if (T.dres[0] + T.dque[0] >= T.dmax[0] && (victim = pick_victim(layer, score(v), &from)) < 0) return;
    pthread_mutex_lock(&T.mx);
    int qfull = T.qn >= VKT_QCAP;
    pthread_mutex_unlock(&T.mx);
    if (qfull) { T.qfull++; return; }
    if (victim >= 0) {
        VSlot *w = &T.s[victim];
        if (T.ndfree == T.cdfree) {
            int nc = T.cdfree ? 2 * T.cdfree : 64;
            ColiVkExpert **n = realloc(T.dfree, (size_t)nc * sizeof(*n));
            if (!n) return;
            T.dfree = n; T.cdfree = nc;
        }
        drop_victim(from);
        T.dfree[T.ndfree++] = w->ex;
        w->ex = w->hex; w->host = 1; T.dres[0]--; T.dres[2]++; T.evictions++; T.up_victims++;
        note_change(victim);   /* published at the next quiescent point: a chain's frame start (vkt_settle) */
        pthread_mutex_lock(&T.mx); T.evict_pending = T.nevict + T.ndfree; pthread_mutex_unlock(&T.mx);
        free_victims();   /* nothing in flight: freed right away; else at the join */
    }
    v->up = 1; T.queued++; T.dque[0]++; T.upromos++;
    pthread_mutex_lock(&T.mx);
    T.q[(T.qh + T.qn) % VKT_QCAP] = (VQ){layer, e, 0, NULL, v->hex}; T.qn++;
    pthread_cond_signal(&T.cv);
    pthread_mutex_unlock(&T.mx);
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
/* The experts the warm start puts on the device: the history's, hottest first, then
 * (COLI_VK_TIER_FILL, on by default) whatever room is left with the experts no history
 * named, one of each layer in turn. A first run with no history used to start with an
 * empty device and fill it at the promotion rate: on a 3090 Ti, Qwen3.6's first run
 * had 16.5% of its routed experts there and decoded at the CPU's speed, the second 97%
 * at three times it (#2081). A model whose experts all fit goes up whole now; one that
 * does not starts with the device full, and the promotions trade cold residents for
 * hot ones as before. */
int vkt_plan(int *layers, int *eids, int max) {
    T.plan_hist = 0;
    if (!T.on) return 0;
    int L = T.main_layers, E = T.c.experts, n = 0, k = 0;   /* the main layers' */
    /* the extra layers' experts first where they stay whole (extra_whole): their own pool,
     * and no history names them */
    if (T.c.extra_whole && T.xmax >= T.c.extra_layers * E)
        for (int l = L; l < T.c.layers; l++)
            for (int e = 0; e < E && k < max && room_dev(l) >= 0; e++) {
                VSlot *v = slot(l, e);
                if (v->state != VS_NONE) continue;
                v->state = VS_QUEUED; set_place(v, 0); T.queued++; T.xque++;
                layers[k] = l; eids[k] = e; k++;
            }
    int kx = k;
    if (g_heat_hist) {
        int *order = malloc((size_t)L * E * sizeof(int));
        if (!order) return 0;
        for (int l = 0; l < L; l++)
            if (g_heat_hist[l]) for (int e = 0; e < E; e++) if (g_heat_hist[l][e]) order[n++] = l * E + e;
        g_plan_heat = (const uint32_t *const *)g_heat_hist; g_plan_E = E;
        qsort(order, (size_t)n, sizeof(int), plan_cmp);
        for (int i = 0; i < n && k < max && room_dev(0) >= 0; i++) {
            VSlot *v = &T.s[order[i]];
            if (v->state != VS_NONE) continue;
            int dev = room_dev(0);   /* the hottest fill the primary device, the next ones the second, then the host pool */
            v->state = VS_QUEUED; set_place(v, dev); T.queued++; T.dque[dev]++;
            layers[k] = order[i] / E; eids[k] = order[i] % E; k++;
        }
        free(order);
    }
    T.plan_hist = k - kx;
    /* only the layers the heat table has a row for: a NULL row is a layer without
     * experts (route_trace.h's rt_drop_row), and with no table at all nothing is known */
    const char *fill = getenv("COLI_VK_TIER_FILL");
    if ((fill && *fill == '0') || !g_heat_hist) return k;
    for (int e = 0; e < E && k < max && room_dev(0) >= 0; e++)
        for (int l = 0; l < L && k < max && room_dev(0) >= 0; l++) {
            if (!g_heat_hist[l]) continue;
            VSlot *v = &T.s[(size_t)l * E + e];
            if (v->state != VS_NONE) continue;
            int dev = room_dev(0);
            v->state = VS_QUEUED; set_place(v, dev); T.queued++; T.dque[dev]++;
            layers[k] = l; eids[k] = e; k++;
        }
    return k;
}
/* The warm start's line, for every engine: "warm start, N experts" stays its head. */
void vkt_warm_report(const char *engine, int n, double seconds) {
    if (!T.on || n <= 0) return;
    int all = 0;   /* the experts of the layers that have them */
    for (int l = 0; l < T.main_layers; l++) if (g_heat_hist && g_heat_hist[l]) all += T.c.experts;
    fprintf(stderr, VKT_TAG " tier %s: warm start, %d experts in %.1fs (%d from the history, %d more to fill the device%s)\n",
            engine, n, seconds, T.plan_hist < n ? T.plan_hist : n, n > T.plan_hist ? n - T.plan_hist : 0,
            all && n >= all ? ": every expert of the model" : "");
}
int vkt_put(int layer, int eid, const VktExpertSrc *src) {
    if (!T.on || layer < 0 || layer >= T.c.layers || eid < 0 || eid >= T.c.experts) return 0;
    ColiVkTensor *t[3] = {NULL, NULL, NULL};
    int dev = place_of(slot(layer, eid));   /* vkt_plan's choice: a device, or the host pool */
    int r = src && src->g && src->u && src->d ? upload(dev, layer, src->g, src->u, src->d, src->gs, src->us, src->ds, t) : 0;
    int ok = r > 0;
    /* T.full: the device's experts keep a copy in the host pool too (promote); without one an
     * expert stays where it is put */
    ColiVkTensor *h[3] = {NULL, NULL, NULL};
    if (ok && T.full && dev == 0 && !kind_of(layer) &&
        upload(2, layer, src->g, src->u, src->d, src->gs, src->us, src->ds, h) <= 0) h[0] = h[1] = h[2] = NULL;
    push_done(layer, eid, ok, r < 0, t, h);
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
            ok = coli_vk_tier_tensor_pad(0, &v->t[k], fmt, k < 2 ? H : F, k < 2 ? F : H, gs, pad_w_max(k == 2), pad_s_max(k == 2),
                                         &rows, &stride, &sc);
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
        fprintf(stderr, VKT_TAG " tier %s: no device memory for the streaming slots, cold experts stay on the CPU\n", T.engine);
        T.st_ok = 0;
        return 0;
    }
    if (n < T.st_slots) fprintf(stderr, VKT_TAG " tier %s: %d of %d streaming slots\n", T.engine, n, T.st_slots);
    T.st_per = n / 2;
    return 1;
}
/* Experts' bytes into slots: eids[i] into T.st[slots[i]], ok[i] = 1 when it is there
 * (else the CPU computes it). The engine loads them on this thread, several at once
 * when it can (load_batch: its parallel reads); the conversions into the slots run in
 * parallel, each expert into its own slot. */
static int st_put(int layer, int eid, VStage *v, const VktExpertSrc *src, int par) {
    const int H = T.c.hidden, F = T.c.inter;
    /* the slots hold form 0; with forms by layer a slot takes each layer's in turn */
    const __typeof__(T.form[0]) *fm = &T.form[T.per_layer ? form_of(layer) : 0];
    if (!src->g || !src->u || !src->d || (fm->gu_scales && (!src->gs || !src->us || !src->ds))) return 0;
    const uint8_t *codes[3] = {src->g, src->u, src->d}; const void *scs[3] = {src->gs, src->us, src->ds};
    for (int k = 0; k < 3; k++) {
        uint8_t *rows; size_t stride; float *sc;
        int I = k < 2 ? H : F, O = k < 2 ? F : H;
        if (!(T.per_layer ? coli_vk_tensor_refill_as(v->t[k], k < 2 ? fm->gu_fmt : fm->dn_fmt, I, O, k < 2 ? fm->gu_gs : fm->dn_gs,
                                                     &rows, &stride, &sc)
                          : coli_vk_tensor_refill(v->t[k], &rows, &stride, &sc))) return 0;
        convert_par(k < 2 ? fm->gu : fm->dn, I, O, codes[k], scs[k], rows, stride, sc, par);
    }
    (void)eid;
    return 1;
}
/* A resident of the host pool whose three tensors are block types: a big step copies it
 * into a slot (pool_copy) */
static int pool_block(const VSlot *v) {
    if (v->state != VS_RESIDENT || !v->ex || !v->host || v->dev) return 0;
    ColiVkTensor *t[3]; ColiVkTensorInfo ti;
    coli_vk_xb_expert_tensors(v->ex, &t[0], &t[1], &t[2]);
    for (int k = 0; k < 3; k++)
        if (!t[k] || !coli_vk_tensor_info(t[k], &ti) || ti.fmt < 100 || ti.fmt >= 164) return 0;
    return 1;
}
/* host pool experts copied into slots on the device: their bytes cross the bus once, where a
 * big step's grouped GEMMs read them in place at a third of its rate. The copies go at the
 * head of sub-batch half `half`'s next submit (coli_vk_xb_sub_copy), the one these slots
 * serve first: the engine thread goes on to the step's cold experts meanwhile. */
static void st_copy_group(int layer, const int *eids, const int *slots, int n, uint8_t *ok, int half) {
    ColiVkTensor *dst[3 * 64], *src[3 * 64];
    int nt = 0;
    double bytes = 0;
    for (int i = 0; i < n; i++) {
        ColiVkTensor *t[3];
        coli_vk_xb_expert_tensors(slot(layer, eids[i])->ex, &t[0], &t[1], &t[2]);
        for (int k = 0; k < 3; k++) {
            ColiVkTensorInfo ti;
            if (coli_vk_tensor_info(t[k], &ti)) bytes += (double)coli_vk_tensor_payload(ti.fmt, ti.I, ti.O, ti.gs);
            dst[nt] = T.st[slots[i]].t[k]; src[nt++] = t[k];
        }
    }
    const int good = nt > 0 && coli_vk_xb_sub_copy(half, dst, src, nt);
    for (int i = 0; i < n; i++) {
        VStage *v = &T.st[slots[i]];
        ok[i] = (uint8_t)good;
        if (!good) { v->state = SS_FREE; continue; }
        v->layer = layer; v->eid = eids[i]; v->state = SS_FILLED; v->pending = 0; v->pref = 0;
    }
    if (good) { T.pc_experts += (unsigned long long)n; T.pc_bytes += bytes; }
}
/* experts into slots: pool[i] set (or NULL), the host pool's copied on the device, the rest
 * loaded by the engine and uploaded */
static void st_fill_group(int layer, const int *eids, const int *slots, int n, uint8_t *ok, const uint8_t *pool, int half) {
    VktExpertSrc srcs[64]; void *hs[64];
    if (n > 64) n = 64;
    if (pool) {
        int ce[64], cs[64], ci[64], nc = 0, le[64], ls[64], li[64], nl = 0;
        uint8_t got[64];
        for (int i = 0; i < n; i++) {
            if (pool[i]) { ci[nc] = i; ce[nc] = eids[i]; cs[nc++] = slots[i]; }
            else { li[nl] = i; le[nl] = eids[i]; ls[nl++] = slots[i]; }
        }
        if (nc) { st_copy_group(layer, ce, cs, nc, got, half); for (int j = 0; j < nc; j++) ok[ci[j]] = got[j]; }
        if (nl) { st_fill_group(layer, le, ls, nl, got, NULL, half); for (int j = 0; j < nl; j++) ok[li[j]] = got[j]; }
        return;
    }
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
static int host_view_share(int layer, int e, VktHostExpert *hx, int share);   /* below, with the hybrid steps */
static int grow_hy(int n, int np);
static int grow_hyx(int S);
static const float *hy_keep_x(const float *src);
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
    int npool = 0;
    for (int e = 0; e < E; e++) {
        ofs[e] = at; at += cnt[e];
        VSlot *v = slot(layer, e);
        cls[e] = !cnt[e] ? BC_NONE : v->state == VS_RESIDENT && v->ex ? (v->dev ? BC_DEV2 : BC_DEV)
               : T.st_ok && cnt[e] >= R ? BC_STREAM : BC_CPU;
        if (T.bpool) T.bpool[e] = 0;
        if (cls[e] == BC_DEV && T.pool_copy && pool_block(v)) { cls[e] = BC_STREAM; T.bpool[e] = 1; npool++; }   /* copied into a slot */
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
    /* A host pool expert of block types with at most hy_big rows: the CPU computes its pairs in
     * place at the join (vkt_host_pairs), while it waits for the device anyway, instead of its
     * bytes crossing the bus for a row or two (the device's alternative is a copy, so the
     * decode's PCIe share does not apply) */
    T.hy_n = T.hy_nx = 0;
    int nhy = 0;
    if (T.hybrid && T.hy_big > 0 && grow_hy(n, 0) && grow_hyx(S)) {
        for (int i = 0; i < n; i++) T.hy_of[i] = -1;
        for (int e = 0; e < E; e++) {
            if (cnt[e] < 1 || cnt[e] > T.hy_big || (cls[e] != BC_DEV && cls[e] != BC_STREAM) || !slot(layer, e)->host) continue;
            VktHostExpert hv;
            if (!host_view_share(layer, e, &hv, 0) || !grow_hy(n, T.hy_n + cnt[e])) continue;
            for (int r = 0; r < cnt[e]; r++) {
                const int i = T.blist[ofs[e] + r];
                T.hy_v[T.hy_n] = hv; T.hy_x[T.hy_n] = hy_keep_x(x + (size_t)(i / K) * H); T.hy_w[T.hy_n] = w ? w[i] : 1.f;
                T.hy_of[i] = T.hy_n++;
            }
            if (T.bpool && T.bpool[e]) { T.bpool[e] = 0; npool--; }
            cls[e] = BC_NONE; nhy++;   /* none of the device's */
        }
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
    if (nst && !st_alloc()) {   /* no slots: the host pool's are read in place again */
        for (int e = 0; e < E; e++)
            if (cls[e] == BC_STREAM && T.bpool && T.bpool[e]) { cls[e] = BC_DEV; T.bpool[e] = 0; ndev++; }
            else if (cls[e] == BC_STREAM) { cls[e] = BC_CPU; nkept++; kept_rows += cnt[e]; }
        nst = npool = 0;
    }
    if (!T.said_fwd) {
        T.said_fwd = 1;
        char hb[32];
        fprintf(stderr, VKT_TAG " tier %s stream: a step of %d rows; cold experts with %d rows or more go to the device (",
                T.engine, S, R);
        if (T.st_rows_env > 0) fprintf(stderr, "COLI_VK_TIER_STREAM_ROWS");
        else if (T.bw_up > 0 && T.cpu_row_ms > 0)
            fprintf(stderr, "an upload of %s at %.1f GB/s takes %.3f ms, the CPU %.4f ms a row; the GEMM from %d rows",
                    human((double)T.exp_bytes, hb, sizeof hb), T.bw_up / 1e6, (double)T.exp_bytes / T.bw_up,
                    T.cpu_row_ms, T.gemm_rows);
        else fprintf(stderr, "the GEMM's %d rows: the upload and the CPU's rows not measured yet", T.gemm_rows);
        fprintf(stderr, "); layer %d: %d resident, %d streamed (%d of them copied from the host pool), %d kept on the CPU (%d rows)",
                layer, ndev, nst, npool, nkept, kept_rows);
        if (nhy) fprintf(stderr, ", %d of the host pool on the CPU in place (%d rows or fewer)", nhy, T.hy_big);
        if (T.d2) fprintf(stderr, ", %d on the second device", nd2);
        fprintf(stderr, "\n");
    }
    T.st_kept += (unsigned long long)nkept; T.st_kept_rows += (unsigned long long)kept_rows;
    if (!ndev && !nst && !nd2 && !T.hy_n) return 0;
    /* The backend counts both halves, activation buffers, descriptor windows and
     * growth before allocating. Leave half of available RAM for CPU work too. */
    int half = T.st_half < n ? T.st_half : n;
    if (ndev || nst) {
        size_t ram = (size_t)(compat_mem_available_gb() * 1e9);
        half = coli_vk_xb_sub_fit(half, E + 1, ram ? ram / 2 : SIZE_MAX);
        if (half < 1 || !coli_vk_xb_sub_reserve(half, E + 1)) {
            fprintf(stderr, VKT_TAG " tier %s: no scratch for a step of %d rows, its experts stay on the CPU\n", T.engine, S);
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
                uint8_t gok[64], gpool[64];
                for (int i = 0; i < gn; i++) gpool[i] = T.bpool ? T.bpool[gid[i]] : 0;
                st_fill_group(layer, gid, gsl, gn, gok, gpool, cur ? curh : (int)(T.sk % 2));   /* the half they go to */
                for (int i = 0; i < gn; i++) {
                    if (gok[i]) eslot[gid[i]] = gsl[i];
                    else if (gpool[i]) { cls[gid[i]] = BC_DEV; T.bpool[gid[i]] = 0; }   /* not copied: read in place */
                    else { cls[gid[i]] = BC_CPU; T.st_kept++; T.st_kept_rows += (unsigned long long)cnt[gid[i]]; }
                }
                if (cls[e] == BC_STREAM && eslot[e] < 0) {   /* no slot: the CPU's, a host pool expert read in place */
                    if (T.bpool && T.bpool[e]) { cls[e] = BC_DEV; T.bpool[e] = 0; }
                    else { cls[e] = BC_CPU; T.st_kept++; T.st_kept_rows += (unsigned long long)cnt[e]; }
                }
                if (cls[e] == BC_CPU) continue;
            }
            if (cls[e] == BC_STREAM) {
                sl = eslot[e];
                if (!T.bpool || !T.bpool[e]) { T.st_experts++; T.st_rows += (unsigned long long)cnt[e]; }   /* the cold ones */
            }
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
    coli_vk_xb_sub_copy_drop();   /* copies no submit took (a step that ended early): their slots go back free */
    free(order); free(bex); free(brows); free(bx); free(bw); free(bidx); free(eslot);
    total += d2_taken;
    T.big_taken = total;
    if (T.hy_n) {   /* the join computes them */
        for (int i = 0; i < n; i++) if (T.hy_of[i] >= 0) taken[i] = 1;
        for (int p = 0; p < T.hy_n; p++) T.hy_hx[p] = &T.hy_v[p];
        T.inflight = 1;
    }
    if (!total && !T.inflight) {   /* nothing went (or everything failed before a submit) */
        T.big = 0;
        if (T.sy_step) { coli_vk_xb_step_end(); T.sy_step = NULL; }
        if (T.big_fail) {
            fprintf(stderr, VKT_TAG " tier %s: a batch failed, its experts are recomputed on the CPU and the tier stops\n", T.engine);
            T.on = 0;
        }
        return 0;
    }
    T.steps++; T.served += (unsigned long long)total; T.st_steps++;
    T.S = S; T.K = K;
    T.t_issued = vkt_now_ms();
    return total + T.hy_n;
}
static int host_pairs(const VktHostExpert *const *hx, const float *const *xr, int np, int D, int I,
                      float limit, int nt, float *gu, float *hh, float *dn, const float *pw);
static int join_big(const float **rows) {
    double t0 = vkt_now_ms(), tc = t0 - T.t_issued;
    T.cpu_ms += tc;
    /* the host pool's pairs of few rows, while the device computes its batches */
    const int hn = T.hy_n;
    int hok = 1;
    if (hn) {
        hok = host_pairs(T.hy_hx, T.hy_x, hn, T.c.hidden, T.c.inter, T.c.act_limit, 0, T.hy_gu, T.hy_hh, T.hy_out,
                         T.c.act == VKT_ACT_SWIGLU_V4 ? T.hy_w : NULL);
        double th = vkt_now_ms() - t0;
        T.hy_ms += th; T.cpu_ms += th; T.hyb_steps++; T.hyb_pairs += (unsigned long long)hn;
        T.hy_n = 0;
    }
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
    int cpu_pairs = T.big_n - T.big_taken - hn;   /* the engine's own */
    if (cpu_pairs >= 32 && tc > 0) {
        double r = tc / cpu_pairs;
        T.cpu_row_ms = T.cpu_row_ms > 0 ? 0.8 * T.cpu_row_ms + 0.2 * r : r;
    }
    T.inflight = 0; T.big = 0;
    int ok = !T.big_fail;
    const float *SY = T.sy_step ? T.sy_step : T.sy;
    for (int i = 0; i < T.big_n; i++)
        rows[i] = !(ok && ok1 && hok) ? NULL : T.map[i] >= 0 ? SY + (size_t)i * T.c.hidden
                : hn && T.hy_of[i] >= 0 ? T.hy_out + (size_t)T.hy_of[i] * T.c.hidden : NULL;
    for (int i = 0; T.st && i < T.st_n; i++) if (T.st[i].state == SS_BUSY) { T.st[i].state = SS_FREE; T.st[i].pending = 0; }
    if (!ok) {
        fprintf(stderr, VKT_TAG " tier %s: a batch failed, its experts are recomputed on the CPU and the tier stops\n", T.engine);
        T.on = 0;
        return 0;
    }
    if (!ok1) {
        d2_stop("a batch on the second device failed");
        return 0;
    }
    quiesce();
    if (!hok) {   /* the CPU computes the step's pairs from its own copies */
        fprintf(stderr, VKT_TAG " tier %s: a prompt step's host pool pairs could not be computed in place, the step goes to the CPU\n", T.engine);
        return 0;
    }
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
        st_fill_group(layer, cand + used, gsl, gn, gok, NULL, 0);
        for (int i = 0; i < gn; i++) if (gok[i]) { T.st[gsl[i]].pref = 1; got++; }
        used += gn;
    }
    T.pf_n += (unsigned long long)got;
    free(est); free(cand);
    return got;
}

/* the share of the host pool's experts the device keeps reading over PCIe in a hybrid
 * step (COLI_VK_HYBRID_PCIE, 0..1; 0: the CPU computes every one). The CPU's rows by eight
 * (ggml_quants.h gq_rows_f8) read the pool at its memory's speed, faster than the device
 * reads it over PCIe, and a step's few pairs split by a hash leave the device the slower
 * side on the steps that drew two of them. An RTX PRO 4000 (PCIe 4.0 x8) and an i5-13500,
 * warm 400-token answers, the share at 0.3 against 0: Flint UD-Q3_K_XL with 7.4 GB of the
 * card 44.6 against 49.4 tok/s (0.15 47.5, 0.45 40.8), with 12 GB 51.6 against 56.2;
 * Qwen3.6-35B-A3B Q4_K_M 47.7 against 55.5 and 71.3 against 74.4; Qwen3.8 IQ2_XS on the
 * whole card 49.1 against 49.0 (0.15 48.8). */
static double hybrid_pcie_share(void) {
    static double f = -1;
    if (f < 0) { const char *e = getenv("COLI_VK_HYBRID_PCIE"); f = e && *e ? atof(e) : 0; if (f < 0) f = 0; if (f > 1) f = 1; }
    return f;
}
/* The device formats the CPU computes from the host pool's rows: a GGUF's block types, and
 * colibri's int8 (fmt 1 a row's scale, 13 a group's) and int4 pairs (fmt 2, 4), whose groups
 * are whole blocks of 32 of the activations */
static int host_fmt_ok(const ColiVkTensorInfo *ti) {
    if (ti->fmt >= 100 && ti->fmt < 164) return 1;
    if (ti->I % 32) return 0;
    if (ti->fmt == 1 || ti->fmt == 2) return 1;
    if (ti->fmt == 5) return ti->I % 64 == 0;
    return (ti->fmt == 4 || ti->fmt == 13 || ti->fmt == 7 || ti->fmt == 12) && ti->gs >= 32 && ti->gs % 32 == 0;
}
static int host_type(const ColiVkTensorInfo *ti) { return ti->fmt >= 100 ? ti->fmt - 100 : -ti->fmt; }
static int host_gs(const ColiVkTensorInfo *ti) {
    return ti->fmt == 5 ? 64 : ti->fmt == 4 || ti->fmt == 13 || ti->fmt == 7 || ti->fmt == 12 ? ti->gs : 0;
}
/* A resident expert of the host pool the CPU computes in place in a hybrid step: outside the
 * device's share (by a hash of (layer, expert): the same split at every step and every read
 * of a table, the CPU's share spread over the experts), its three tensors block types the
 * CPU has rows of (a GGUF's, as the file has them), gate and up of one type. *hx its rows. */
static int host_view_share(int layer, int e, VktHostExpert *hx, int share) {
    VSlot *v = slot(layer, e);
    if (v->state != VS_RESIDENT || v->dev != 0 || !v->ex || !v->host) return 0;
    uint32_t hsh = (uint32_t)(layer * 2654435761u) ^ (uint32_t)(e * 2246822519u);
    hsh ^= hsh >> 15; hsh *= 2654435761u; hsh ^= hsh >> 13;
    if (share && (double)(hsh & 0xffff) / 65536.0 < hybrid_pcie_share()) return 0;
    ColiVkTensor *t[3]; ColiVkTensorInfo ti[3]; const void *r[3]; int ok = 1;
    const void *sc[3] = {NULL, NULL, NULL};
    coli_vk_xb_expert_tensors(v->ex, &t[0], &t[1], &t[2]);
    for (int k = 0; k < 3 && ok; k++)
        ok = t[k] && coli_vk_tensor_info(t[k], &ti[k]) && host_fmt_ok(&ti[k]) &&
             (r[k] = coli_vk_tensor_host_rows(t[k], &sc[k])) != NULL && (ti[k].fmt >= 100 || sc[k]);
    if (!ok || ti[0].fmt != ti[1].fmt || ti[0].gs != ti[1].gs) return 0;
    *hx = (VktHostExpert){r[0], r[1], r[2], host_type(&ti[0]), host_type(&ti[2]),
                          (size_t)ti[0].rowWords * 4, (size_t)ti[2].rowWords * 4,
                          sc[0], sc[1], sc[2], host_gs(&ti[0]), host_gs(&ti[2])};
    return 1;
}
static int host_view(int layer, int e, VktHostExpert *hx) { return host_view_share(layer, e, hx, 1); }

static int hy_threads(void) {   /* the CPU's team: COLI_VK_HYBRID_THREADS, else OpenMP's */
    static int n = -1;
    if (n < 0) { const char *e = getenv("COLI_VK_HYBRID_THREADS"); n = e && atoi(e) > 0 ? atoi(e) : 0; }
#ifdef _OPENMP
    return n > 0 ? n : omp_get_max_threads();
#else
    return 1;
#endif
}
/* colibri's int8 (fmt 1, 13), int4 (fmt 2, 4), int3-g64 (fmt 5), MXFP4 (fmt 7) and FP8 (fmt 12)
 * rows. dev_codes: an int4 row's codes in element order (the pairs' low nibble first), 0..15
 * (the value + 8), into u (I bytes). */
static void dev_codes(uint8_t *u, const uint8_t *row, int I) {
    int j = 0;
#if defined(__AVX2__)
    const __m128i m4 = _mm_set1_epi8(0x0f);
    for (; j + 16 <= I / 2; j += 16) {
        const __m128i b = _mm_loadu_si128((const __m128i *)(row + j));
        const __m128i lo = _mm_and_si128(b, m4), hi = _mm_and_si128(_mm_srli_epi16(b, 4), m4);
        _mm_storeu_si128((__m128i *)(u + 2 * j), _mm_unpacklo_epi8(lo, hi));
        _mm_storeu_si128((__m128i *)(u + 2 * j + 16), _mm_unpackhi_epi8(lo, hi));
    }
#endif
    for (; j < I / 2; j++) { u[2 * j] = row[j] & 15; u[2 * j + 1] = row[j] >> 4; }
}
/* an int3-g64 row's codes (16 bytes of 2-bit low planes, then 8 of high bits, a group of 64),
 * 0..7 (the value + 4) */
static void dev_codes3(uint8_t *u, const uint8_t *row, int I) {
    for (int g = 0; g < I / 64; g++) {
        const uint8_t *lo = row + (size_t)g * 24, *hi = lo + 16;
        uint8_t *o = u + (size_t)g * 64;
        for (int j = 0; j < 64; j++) o[j] = (uint8_t)(((lo[j >> 2] >> ((j & 3) * 2)) & 3u) | (((hi[j >> 3] >> (j & 7)) & 1u) << 2));
    }
}
/* an MXFP4 row's values times 2, as int8 (e2m1: 0, 0.5, 1, 1.5, 2, 3, 4, 6 and the sign) */
static void dev_codes_fp4(int8_t *w, const uint8_t *row, int I) {
    static const int8_t v2[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
    for (int j = 0; j < I / 2; j++) { w[2 * j] = v2[row[j] & 15]; w[2 * j + 1] = v2[row[j] >> 4]; }
}
static float dev_fp8(uint8_t b) {   /* e4m3 as the backend decodes it */
    const int e = (b >> 3) & 15, m = b & 7;
    const float v = e == 0 ? m / 512.0f : ldexpf(1.0f + m / 8.0f, e - 7);
    return (b & 0x80) ? -v : v;
}
/* Activation row q (int8 per 32, dx its scales, xs its sums per 16) against a row's codes:
 * int8 w8, or int4 u4 (0..15, the value + 8: the bias comes back with the activation's
 * sums). rs: the row's scale (gs 0) or its groups' of gs columns. Integer sums a block, then
 * the block's scale: dx times the group's (the row's at the end). */
static float dev_dot(const int8_t *w8, const uint8_t *u4, int zero, const float *rs, int gs, const int8_t *q, const float *dx,
                     const int32_t *xs, int I) {
    float bias = 0.f;
#if defined(__AVX2__)
    const __m256i one16 = _mm256_set1_epi16(1);
    __m256 acc = _mm256_setzero_ps();
    for (int b = 0; b < I / 32; b++) {
        const float sc = dx[b] * (gs ? rs[b * 32 / gs] : 1.f);
        const __m256i x = _mm256_loadu_si256((const __m256i *)(q + b * 32));
        __m256i p;
        if (u4) {
            p = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_loadu_si256((const __m256i *)(u4 + b * 32)), x), one16);
            bias += sc * (float)(xs[2 * b] + xs[2 * b + 1]);
        } else {
            const __m256i w = _mm256_loadu_si256((const __m256i *)(w8 + b * 32));
            p = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_sign_epi8(w, w), _mm256_sign_epi8(x, w)), one16);
        }
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_set1_ps(sc), acc);
    }
    __m128 h = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    h = _mm_add_ps(h, _mm_movehl_ps(h, h)); h = _mm_add_ss(h, _mm_movehdup_ps(h));
    float r = _mm_cvtss_f32(h);
#else
    float r = 0.f;
    for (int b = 0; b < I / 32; b++) {
        const float sc = dx[b] * (gs ? rs[b * 32 / gs] : 1.f);
        int32_t p = 0;
        for (int i = 0; i < 32; i++) p += (u4 ? (int32_t)u4[b * 32 + i] : (int32_t)w8[b * 32 + i]) * q[b * 32 + i];
        if (u4) bias += sc * (float)(xs[2 * b] + xs[2 * b + 1]);
        r += (float)p * sc;
    }
#endif
    r -= (float)zero * bias;
    return gs ? r : r * rs[0];
}
/* the float reference: the row dequantized (its codes times their scale) */
static void dev_dequant(float *o, const uint8_t *row, int fmt, const float *rs, int gs, int I) {
    static const float e2m1[8] = {0, 0.5f, 1, 1.5f, 2, 3, 4, 6};
    for (int i = 0; i < I; i++) {
        float w;
        if (fmt == 1 || fmt == 13) w = (float)((const int8_t *)row)[i];
        else if (fmt == 12) w = dev_fp8(row[i]);
        else if (fmt == 5) {   /* 3/8 of a byte a weight: no nibble to read */
            const uint8_t *lo = row + (size_t)(i >> 6) * 24, *hi = lo + 16; const int j = i & 63;
            w = (float)((int)(((lo[j >> 2] >> ((j & 3) * 2)) & 3u) | (((hi[j >> 3] >> (j & 7)) & 1u) << 2)) - 4);
        } else {
            const int nib = (i & 1) ? row[i >> 1] >> 4 : row[i >> 1] & 15;
            w = fmt == 7 ? ((nib & 8) ? -e2m1[nib & 7] : e2m1[nib & 7]) : (float)(nib - 8);
        }
        o[i] = w * (gs ? rs[i / gs] : rs[0]);
    }
}
/* a row of type t (I inputs; rs its scales, gs their group for colibri's formats) against
 * act a's rows (xf[s] their floats): the CPU's integer form, or with GQ_EXACT=1 its float
 * reference, as its own matmuls would take it. Each activation row its own dot, the same
 * bits whatever rows share the call. tmp: I floats, and for an int4 row I bytes after them. */
static inline void host_rows(float *y, int ys, const gq_act *a, const float *const *xf, const uint8_t *row, int t, int I,
                             float *tmp, const float *rs, int gs) {
    if (t < 0 && (-t == 12 || (-t == 7 && I / gs > 2048)) && !gq_exact_mode()) {   /* FP8 has no integer form: its floats against x's */
        dev_dequant(tmp, row, -t, rs, gs, I);
        for (int s = 0; s < a->S; s++) {
            const float *x = xf[s];
            float acc = 0.f;
            for (int i = 0; i < I; i++) acc += tmp[i] * x[i];
            y[(size_t)s * ys] = acc;
        }
        return;
    }
    if (t < 0 && !gq_exact_mode()) {
        const int fmt = -t;
        uint8_t *u = NULL; const int8_t *w8 = (const int8_t *)row; int zero = 0;
        float half[2048], *rsx = (float *)rs;
        if (fmt == 2 || fmt == 4) { u = (uint8_t *)(tmp + I); dev_codes(u, row, I); zero = 8; }
        else if (fmt == 5) { u = (uint8_t *)(tmp + I); dev_codes3(u, row, I); zero = 4; }
        else if (fmt == 7) {   /* the values doubled: the group's scale halved */
            int8_t *w = (int8_t *)(tmp + I); dev_codes_fp4(w, row, I); w8 = w;
            for (int g = 0; g < I / gs; g++) half[g] = 0.5f * rs[g];   /* at most 2048: above */
            rsx = half;
        }
        for (int s = 0; s < a->S; s++)
            y[(size_t)s * ys] = dev_dot(w8, u, zero, rsx, gs, a->q + (size_t)s * I, a->d + (size_t)s * (I / 32),
                                        a->s + (size_t)s * (I / 16), I);
        return;
    }
    if (!gq_exact_mode()) { gq_row_all(y, ys, a, row, t, I); return; }
    if (t < 0) dev_dequant(tmp, row, -t, rs, gs, I);
    else gq_dequant_row(t, row, tmp, I);
    for (int s = 0; s < a->S; s++) {
        double acc = 0; const float *x = xf[s];
        for (int i = 0; i < I; i++) acc += (double)tmp[i] * x[i];
        y[(size_t)s * ys] = (float)acc;
    }
}
typedef struct { const uint8_t *g; int p; } HyKey;
static int hy_key_cmp(const void *a, const void *b) {
    const HyKey *x = a, *y = b;
    if (x->g != y->g) return (uintptr_t)x->g < (uintptr_t)y->g ? -1 : 1;
    return x->p - y->p;
}
/* DeepSeek V4's expert between gate/up and down, as shaders/expert_act.comp (act 2) and
 * expert_act_v4.comp compute it and its CPU kernel does: gate and up to bf16 (nearest
 * even), the clamped SwiGLU with the sigmoid's two forms, times the route weight and to
 * bf16, then E4M3 and back per 128 inputs under a power-of-two scale (the smallest 2^e
 * with max(amax, 1e-4) <= 448 * 2^e). Each step is exact in f32 or rounds once. */
static float hy_bf16(float v) {
    uint32_t b; memcpy(&b, &v, 4);
    if ((b & 0x7f800000u) != 0x7f800000u) b += 0x7fffu + ((b >> 16) & 1u);
    b &= 0xffff0000u; memcpy(&v, &b, 4);
    return v;
}
static float hy_e4m3(float x) {   /* |x| <= 448: the nearest E4M3 value, ties to even */
    float a = fabsf(x), r;
    if (a == 0.f) return x;
    if (a >= 448.f) r = 448.f;
    else if (a < 0.015625f) {   /* subnormal codes, steps of 2^-9 */
        float t = a * 512.f, f = floorf(t), fr = t - f;
        if (fr > 0.5f || (fr == 0.5f && ((int)f & 1))) f += 1.f;
        r = f * 0.001953125f;
    } else {
        uint32_t b; memcpy(&b, &a, 4);
        const int E = (int)((b >> 23) & 0xffu) - 127;
        uint32_t sig = 0x800000u | (b & 0x7fffffu), q = sig >> 20, rem = sig & 0xfffffu;
        if (rem > 0x80000u || (rem == 0x80000u && (q & 1u))) q++;
        r = ldexpf((float)q, E - 3);
    }
    return x < 0.f ? -r : r;
}
static void hy_act_v4(float *h, const float *g, const float *u, int I, float limit, float w) {
    for (int o = 0; o < I; o++) {
        float gv = hy_bf16(g[o]), uv = hy_bf16(u[o]);
        if (limit > 0.f) { gv = gv < limit ? gv : limit; uv = uv < -limit ? -limit : uv > limit ? limit : uv; }
        const float sg = gv >= 0.f ? 1.f / (1.f + expf(-gv)) : expf(gv) / (1.f + expf(gv));
        h[o] = hy_bf16(gv * sg * uv * w);
    }
    for (int b0 = 0; b0 < I; b0 += 128) {
        const int b1 = b0 + 128 < I ? b0 + 128 : I;
        float amax = 0.f;
        for (int o = b0; o < b1; o++) amax = fmaxf(amax, fabsf(h[o]));
        const float m = fmaxf(amax, 1e-4f);
        uint32_t mb; memcpy(&mb, &m, 4);
        const int E = (int)((mb >> 23) & 0xffu) - 127;
        int e = (mb & 0x7fffffu) <= 0x600000u ? E - 8 : E - 7;
        e = e < -126 ? -126 : e > 126 ? 126 : e;
        for (int o = b0; o < b1; o++) {
            float v = ldexpf(h[o], -e);
            v = v < -448.f ? -448.f : v > 448.f ? 448.f : v;
            h[o] = ldexpf(hy_e4m3(v), e);
        }
    }
}
/* pw: the pairs' route weights for DeepSeek V4's expert (hy_act_v4, the weight in the row as
 * the device's), NULL for SwiGLU */
static int host_pairs(const VktHostExpert *const *hx, const float *const *xr, int np, int D, int I,
                      float limit, int nt, float *gu, float *hh, float *dn, const float *pw) {
    if (np < 1) return 1;
    /* The pairs of one expert side by side (q: their place in ord), a group reading the
     * expert's rows once for all of its pairs: a prompt step's experts of a few rows, or the
     * rows of a decode step's draft that share one. gu and hh go in that order, dn in p's. */
    int err = 0, nx = 0, ng = 0, ident = 1;
    int *xa = malloc((size_t)np * sizeof(int)), *ord = malloc((size_t)np * sizeof(int)), *gs = malloc((size_t)(np + 1) * sizeof(int)),
        *gx = malloc((size_t)np * sizeof(int));
    HyKey *key = malloc((size_t)np * sizeof(HyKey));
    const float **xo = malloc((size_t)np * sizeof(float *)), **ho = malloc((size_t)np * sizeof(float *));
    gq_act ax = {0}, *ag = calloc((size_t)np, sizeof(gq_act)), *ah = calloc((size_t)np, sizeof(gq_act));
    float *xb = NULL, *dq = NULL;
    if (!xa || !ord || !gs || !gx || !key || !xo || !ho || !ag || !ah) { err = 2; goto out; }
    for (int p = 0; p < np; p++) key[p] = (HyKey){hx[p]->g, p};
    qsort(key, (size_t)np, sizeof(HyKey), hy_key_cmp);
    for (int q = 0; q < np; q++) {
        ord[q] = key[q].p; ident &= ord[q] == q;
        if (!q || key[q].g != key[q - 1].g) gs[ng++] = q;
        xo[q] = xr[ord[q]]; ho[q] = hh + (size_t)q * I;
    }
    gs[ng] = np;
    for (int p = 0; p < np; p++) {   /* a row's activation once: the pairs of a row share it */
        int r = 0;
        while (r < p && xr[r] != xr[p]) r++;
        xa[p] = r < p ? xa[r] : nx++;
    }
    if (!(xb = malloc((size_t)nx * D * sizeof(float))) || (!ident && !(dq = malloc((size_t)np * D * sizeof(float))))) { err = 2; goto out; }
    for (int p = 0; p < np; p++) memcpy(xb + (size_t)xa[p] * D, xr[p], (size_t)D * sizeof(float));
    if (gq_act_set(&ax, xb, nx, D)) { err = 1; goto out; }
    for (int g = 0; g < ng && !err; g++) {   /* each group's rows of ax */
        for (int q = gs[g]; q < gs[g + 1]; q++) gx[q - gs[g]] = xa[ord[q]];
        if (gq_act_gather(&ag[g], &ax, gx, gs[g + 1] - gs[g])) err = 2;
    }
    if (err) goto out;
    float *dy = dq ? dq : dn;
    /* gate and up of every group, the activation, then down: one parallel region, the
     * rows handed out in chunks (the cores of a hybrid CPU do not run at one speed) */
    #pragma omp parallel num_threads(nt > 0 ? nt : hy_threads())
    {
        const int W = D > I ? D : I;   /* the float reference's row, an int4 row's codes after it */
        float *tmp = malloc((size_t)W * sizeof(float) + (size_t)W);
        if (!tmp) {
            #pragma omp atomic write
            err = 2;
        }
        /* a group's rows by eight (gq_rows_f8) where the type has the kernel, one by one where
         * not or where eight would cross gate into up; 32 rows a grab as before */
        const int cgu = (2 * I + 7) / 8, cd = (D + 7) / 8;
        #pragma omp for schedule(dynamic, 4)
        for (int64_t c = 0; c < (int64_t)ng * cgu; c++) {
            if (!tmp) continue;
            const int g = (int)(c / cgu), o0 = (int)(c % cgu) * 8, o1 = o0 + 8 < 2 * I ? o0 + 8 : 2 * I, q0 = gs[g];
            const VktHostExpert *h = hx[ord[q0]];
            if (o1 - o0 == 8 && (o1 <= I || o0 >= I) && h->tg >= 0 && gq_f8_ok(h->tg, D)) {
                const int oo = o0 < I ? o0 : o0 - I;
                gq_rows_f8(gu + (size_t)q0 * 2 * I + o0, 2 * I, &ag[g], (o0 < I ? h->g : h->u) + (size_t)oo * h->rgu, (int64_t)h->rgu, h->tg, D, 8);
                continue;
            }
            for (int o = o0; o < o1; o++) {
                const int oo = o < I ? o : o - I;
                const uint8_t *row = (o < I ? h->g : h->u) + (size_t)oo * h->rgu;
                const float *rs = h->tg < 0 ? (o < I ? h->sg : h->su) + (size_t)oo * (h->gsgu ? (D + h->gsgu - 1) / h->gsgu : 1) : NULL;
                host_rows(gu + (size_t)q0 * 2 * I + o, 2 * I, &ag[g], xo + q0, row, h->tg, D, tmp, rs, h->gsgu);
            }
        }
        #pragma omp for schedule(static)
        for (int q = 0; q < np; q++) {
            const float *g = gu + (size_t)q * 2 * I, *u = g + I; float *h = hh + (size_t)q * I;
            if (pw) { hy_act_v4(h, g, u, I, limit, pw[ord[q]]); continue; }
            for (int o = 0; o < I; o++) {
                float gv = g[o], uv = u[o];
                if (limit > 0.f) { gv = gv < limit ? gv : limit; uv = uv < -limit ? -limit : uv > limit ? limit : uv; }
                h[o] = gv / (1.f + expf(-gv)) * uv;
            }
        }
        #pragma omp for schedule(static)
        for (int g = 0; g < ng; g++)
            if (gq_act_set(&ah[g], hh + (size_t)gs[g] * I, gs[g + 1] - gs[g], I)) {
                #pragma omp atomic write
                err = 1;
            }
        #pragma omp for schedule(dynamic, 8)
        for (int64_t c = 0; c < (int64_t)ng * cd; c++) {
            if (!tmp) continue;
            const int g = (int)(c / cd), o0 = (int)(c % cd) * 8, o1 = o0 + 8 < D ? o0 + 8 : D, q0 = gs[g];
            const VktHostExpert *h = hx[ord[q0]];
            if (o1 - o0 == 8 && h->td >= 0 && gq_f8_ok(h->td, I)) {
                gq_rows_f8(dy + (size_t)q0 * D + o0, D, &ah[g], h->d + (size_t)o0 * h->rd, (int64_t)h->rd, h->td, I, 8);
                continue;
            }
            for (int o = o0; o < o1; o++) {
                const float *rs = h->td < 0 ? h->sd + (size_t)o * (h->gsd ? (I + h->gsd - 1) / h->gsd : 1) : NULL;
                host_rows(dy + (size_t)q0 * D + o, D, &ah[g], ho + q0, h->d + (size_t)o * h->rd, h->td, I, tmp, rs, h->gsd);
            }
        }
        free(tmp);
    }
    if (dq && !err) for (int q = 0; q < np; q++) memcpy(dn + (size_t)ord[q] * D, dq + (size_t)q * D, (size_t)D * sizeof(float));
out:
    gq_act_free(&ax);
    for (int g = 0; ag && ah && g < np; g++) { gq_act_free(&ag[g]); gq_act_free(&ah[g]); }
    free(xa); free(ord); free(gs); free(gx); free(key); free(xo); free(ho); free(ag); free(ah); free(xb); free(dq);
    return !err;
}
int vkt_host_pairs(const VktHostExpert *const *hx, const float *const *xr, int np, int D, int I,
                   float limit, int nt, float *gu, float *hh, float *dn) {
    return host_pairs(hx, xr, np, D, I, limit, nt, gu, hh, dn, NULL);
}
int vkt_host_pairs_w(const VktHostExpert *const *hx, const float *const *xr, int np, int D, int I,
                     float limit, int nt, float *gu, float *hh, float *dn, const float *w) {
    return host_pairs(hx, xr, np, D, I, limit, nt, gu, hh, dn, w);
}
/* room for a step's S input rows of the CPU's pairs (hy_keep_x), before any is taken */
static int grow_hyx(int S) {
    if (S <= T.hy_xcap) return 1;
    float *xb = realloc(T.hy_xb, (size_t)S * T.c.hidden * sizeof(float)); if (xb) T.hy_xb = xb;
    const float **sr = realloc(T.hy_src, (size_t)S * sizeof(*sr)); if (sr) T.hy_src = sr;
    if (!xb || !sr) return 0;
    T.hy_xcap = S;
    return 1;
}
/* room for n pairs of the step (hy_of) and for np of them on the CPU */
static int grow_hy(int n, int np) {
    if (n > T.chy) {
        int nc = grow_to(n, &T.chy), *a = realloc(T.hy_of, (size_t)nc * sizeof(int));
        if (!a) return 0;
        T.hy_of = a; T.chy = nc;
    }
    if (np > T.cho) {
        int nc = grow_to(np, &T.cho), H = T.c.hidden, I = T.c.inter;
        VktHostExpert *v = realloc(T.hy_v, (size_t)nc * sizeof(*v)); if (v) T.hy_v = v;
        const VktHostExpert **h = realloc(T.hy_hx, (size_t)nc * sizeof(*h)); if (h) T.hy_hx = h;
        const float **x = realloc(T.hy_x, (size_t)nc * sizeof(*x)); if (x) T.hy_x = x;
        float *g = realloc(T.hy_gu, (size_t)nc * 2 * I * sizeof(float)); if (g) T.hy_gu = g;
        float *u = realloc(T.hy_hh, (size_t)nc * I * sizeof(float)); if (u) T.hy_hh = u;
        float *o = realloc(T.hy_out, (size_t)nc * H * sizeof(float)); if (o) T.hy_out = o;
        float *w = realloc(T.hy_w, (size_t)nc * sizeof(float)); if (w) T.hy_w = w;
        if (!v || !h || !x || !g || !u || !o || !w) return 0;
        T.cho = nc;
    }
    return 1;
}

/* a hybrid pair's input row, copied once a step (grow_hyx has made room for the step's rows) */
static const float *hy_keep_x(const float *src) {
    const int H = T.c.hidden;
    for (int r = 0; r < T.hy_nx; r++) if (T.hy_src[r] == src) return T.hy_xb + (size_t)r * H;
    float *dst = T.hy_xb + (size_t)T.hy_nx * H;
    memcpy(dst, src, (size_t)H * sizeof(float));
    T.hy_src[T.hy_nx++] = src;
    return dst;
}
static int issue(int layer, const float *x, int S, int K, const int *idx, const float *w, uint8_t *taken) {
    if (S > 0 && K > 0) memset(taken, 0, (size_t)S * K);
    if (!T.on || T.inflight || layer < 0 || layer >= T.c.layers || S < 1 || K < 1) return 0;
    if (!coli_vk_xb_ready()) {
        if (T.on) { fprintf(stderr, VKT_TAG " tier %s: the device stopped answering, the experts stay on the CPU\n", T.engine); T.on = 0; }
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
    /* a hybrid step: the host pool's experts the CPU can compute in place are its, at the join */
    const int hyb = T.hybrid && S <= VKT_UP_ROWS && layer < T.main_layers && grow_hy(n, 0) && grow_hyx(S);
    T.hy_n = T.hy_nx = 0;
    /* heat for every routed expert, groups for the resident ones (first seen first) */
    int ng = 0, total = 0, rows1 = 0;
    for (int i = 0; i < n; i++) {
        T.map[i] = -1;
        if (hyb) T.hy_of[i] = -1;
        int e = idx[i];
        if (e < 0 || e >= E) continue;
        T.routed++;
        if (layer >= T.main_layers) T.extra_routed++;
        VSlot *v = slot(layer, e);
        if (v->heat < 0xFFFFFFFFu) v->heat++;
        v->last = T.tick;
        if (v->state != VS_RESIDENT) continue;
        if (hyb && v->host && grow_hy(n, T.hy_n + 1) && host_view(layer, e, &T.hy_v[T.hy_n])) {
            T.hy_x[T.hy_n] = hy_keep_x(x + (size_t)(i / K) * H); T.hy_w[T.hy_n] = w ? w[i] : 1.f;
            T.hy_of[i] = T.hy_n++;
            continue;
        }
        if (v->dev && rows1 >= T.d2_rows) continue;    /* the second device's batch is as large as it gets */
        int g = T.grp[e];
        if (g < 0) {
            if (total >= T.max_dev_rows || !grow_groups(ng + 1)) continue;   /* the batch is as large as it gets */
            g = ng++; T.grp[e] = g; T.touched[g] = e; T.bex[g] = v->ex; T.brows[g] = 0;
        } else if (total >= T.max_dev_rows) continue;
        T.brows[g]++; total++; rows1 += v->dev;
        T.map[i] = g;                                   /* group for now, row index below */
    }
    if (!ng && !T.hy_n) return 0;
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
        if (!ng && !T.hy_n) return 0;
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
    T.served += (unsigned long long)took;
    if (T.hy_n) {   /* the join computes them */
        for (int i = 0; i < n; i++) if (T.hy_of[i] >= 0) taken[i] = 1;
        for (int p = 0; p < T.hy_n; p++) T.hy_hx[p] = &T.hy_v[p];
        took += T.hy_n;
    }
    if (!took) return 0;
    T.inflight = 1; T.S = S; T.K = K; T.steps++;
    /* T.full: the batch is in flight with the experts it took: a victim's device copy is
     * freed at its join */
    for (int i = 0; T.full && S <= VKT_UP_ROWS && i < n; i++)
        if (idx[i] >= 0 && idx[i] < E && layer < T.main_layers) promote(layer, idx[i]);
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
    /* a hybrid step's host pool pairs, while the device computes its own */
    const int hn = T.hy_n;
    int hok = 1;
    if (hn) {
        hok = host_pairs(T.hy_hx, T.hy_x, hn, T.c.hidden, T.c.inter, T.c.act_limit, 0,
                         T.hy_gu, T.hy_hh, T.hy_out, T.c.act == VKT_ACT_SWIGLU_V4 ? T.hy_w : NULL);
        double th = vkt_now_ms() - t0;
        T.hy_ms += th; T.hy_steps++; T.hy_pairs += (unsigned long long)hn;
        tc += th; t0 += th;
        T.hy_n = 0;
    }
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
    for (int i = 0; i < n; i++)
        rows[i] = !(ok && ok1 && hok) ? NULL : T.map[i] >= 0 ? T.by[T.map[i]]
                : hn && T.hy_of[i] >= 0 ? T.hy_out + (size_t)T.hy_of[i] * T.c.hidden : NULL;
    if (!ok) {
        fprintf(stderr, VKT_TAG " tier %s: a batch failed, its experts are recomputed on the CPU and the tier stops\n", T.engine);
        T.on = 0;
        return 0;
    }
    if (!ok1) {   /* the step's device rows are recomputed on the CPU (join's 0), the tier goes on */
        d2_stop("a batch on the second device failed");
        return 0;
    }
    quiesce();
    if (!hok) {   /* the same: the CPU computes the step's pairs from its own copies */
        fprintf(stderr, VKT_TAG " tier %s: a hybrid step's host pool pairs could not be computed in place, the step goes to the CPU\n", T.engine);
        return 0;
    }
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
    /* a block format reads whole blocks of a row */
    int bg = coli_vk_block_elems(fm->gu_fmt), bd = coli_vk_block_elems(fm->dn_fmt);
    if ((bg && H % bg) || (bd && F % bd)) return 0;
    fm->gu = gu; fm->dn = dn;
    fm->gu_codes = src_row_bytes(gu.kind, H) * (size_t)F;
    fm->d_codes = src_row_bytes(dn.kind, F) * (size_t)H;
    fm->gu_scales = src_scale_bytes(gu, H, F);
    fm->d_scales = src_scale_bytes(dn, F, H);
    fm->stage_bytes = 2 * fm->gu_codes + fm->d_codes + 2 * fm->gu_scales + fm->d_scales;
    return 1;
}

static int fmt_eq(VktFmt a, VktFmt b) { return a.kind == b.kind && a.gs == b.gs; }
/* VktConfig.layer_gate_up/layer_down: each main layer's form, the ones past form 0 from 2
 * on, and the room every main tensor takes (the biggest form's of its role). 0 when a
 * layer's form has no device form, or there are more than VKT_FORMS. */
static int forms_per_layer(void) {
    const int H = T.c.hidden, F = T.c.inter;
    T.nform = 2;
    if (!T.c.layer_gate_up || !T.c.layer_down) return 1;
    if (T.main_layers > VKT_MAX_LAYERS) return 0;
    for (int l = 0; l < T.main_layers; l++) {
        VktFmt a = T.c.layer_gate_up[l], b = T.c.layer_down[l];
        int f = fmt_eq(a, T.c.gate_up) && fmt_eq(b, T.c.down) ? 0 : -1;
        for (int j = 2; f < 0 && j < T.nform; j++) if (fmt_eq(a, T.form[j].gu) && fmt_eq(b, T.form[j].dn)) f = j;
        if (f < 0) {
            if (T.nform == VKT_FORMS || !form_init(T.nform, a, b)) return 0;
            f = T.nform++;
        }
        T.lform[l] = (uint8_t)f;
    }
    if (T.nform == 2) return 1;   /* every layer has form 0 */
    T.per_layer = 1;
    /* the form most layers have: class 0, its rooms its own size; the others class 1 */
    int count[VKT_FORMS] = {0}, major = 0;
    for (int l = 0; l < T.main_layers; l++) count[T.lform[l]]++;
    for (int j = 0; j < T.nform; j++) if (j != 1 && count[j] > count[major]) major = j;
    for (int l = 0; l < T.main_layers; l++) T.lcls[l] = (uint8_t)(T.lform[l] != major);
    for (int j = 0; j < T.nform; j++) {
        if (j == 1 || !count[j]) continue;
        const int c = j != major;
        size_t sg, sd, wg = coli_vk_tensor_alloc_bytes(T.form[j].gu_fmt, H, F, T.form[j].gu_gs, &sg),
               wd = coli_vk_tensor_alloc_bytes(T.form[j].dn_fmt, F, H, T.form[j].dn_gs, &sd);
        if (wg > T.pad_w[c][0]) T.pad_w[c][0] = wg;
        if (sg > T.pad_s[c][0]) T.pad_s[c][0] = sg;
        if (wd > T.pad_w[c][1]) T.pad_w[c][1] = wd;
        if (sd > T.pad_s[c][1]) T.pad_s[c][1] = sd;
    }
    return 1;
}
/* One main expert's bytes on a device whose ranges start at `a`: its form's, or with
 * forms by layer its class's room, on average over the layers (the budget's count; a pool
 * that fills before it refuses the rest, and the count comes down to what it holds). */
static size_t tier_expert_bytes_at(size_t a) {
    if (!T.per_layer) return expert_bytes_at(T.c.hidden, T.c.inter, T.c.gate_up, T.c.down, a);
    size_t room[2] = {0, 0}; long long n[2] = {0, 0};
    for (int c = 0; c < 2; c++)
        for (int k = 0; k < 3; k++) {
            size_t w = T.pad_w[c][k == 2], sc = T.pad_s[c][k == 2];
            room[c] += (w + a - 1) / a * a + (sc + a - 1) / a * a;
        }
    for (int l = 0; l < T.main_layers; l++) n[T.lcls[l]]++;
    return n[0] + n[1] ? (size_t)((room[0] * (unsigned long long)n[0] + room[1] * (unsigned long long)n[1] + n[0] + n[1] - 1) /
                                  (unsigned long long)(n[0] + n[1])) : room[0];
}

/* One main expert's bytes at its own layer's form (no room of a bigger one), on average over
 * the layers, and the largest (*mx): a host pool that holds every expert and so frees none
 * (host_init) keeps them at these sizes. */
static size_t tier_expert_bytes_own(size_t a, size_t *mx) {
    const int H = T.c.hidden, F = T.c.inter;
    unsigned long long sum = 0; size_t big = 0;
    for (int l = 0; l < T.main_layers; l++) {
        const __typeof__(T.form[0]) *fm = &T.form[form_of(l)];
        size_t sg, sd, wg = coli_vk_tensor_alloc_bytes(fm->gu_fmt, H, F, fm->gu_gs, &sg), wd = coli_vk_tensor_alloc_bytes(fm->dn_fmt, F, H, fm->dn_gs, &sd);
        const size_t b = 2 * ((wg + a - 1) / a * a + (sg + a - 1) / a * a) + (wd + a - 1) / a * a + (sd + a - 1) / a * a;
        sum += b; if (b > big) big = b;
    }
    *mx = big;
    return T.main_layers ? (size_t)((sum + (unsigned long long)T.main_layers - 1) / (unsigned long long)T.main_layers) : 0;
}

/* COLI_VK_DEV2: the second device's share. Its budget is what it has free less
 * COLI_VK_RESERVE2_GB (0.5 GiB: its batch's scratch, the driver), an integrated GPU's a
 * quarter of what the RAM leaves (as the primary's), capped by COLI_VK_EXPERTS2 experts
 * when set; it holds at most the experts the primary device does not. */
static void d2_init(long long fit0, long long all, int act) {
    if (!getenv("COLI_VK_DEV2") || !*getenv("COLI_VK_DEV2")) return;
    const char *eng = T.engine;
    if (!coli_vk_dev2_open_env()) {
        fprintf(stderr, VKT_TAG " tier %s: COLI_VK_DEV2=%s brought no second device up, the tier stays on %s\n",
                eng, getenv("COLI_VK_DEV2"), coli_vk_device_name());
        return;
    }
    ColiVkDevInfo di;
    if (!coli_vk_dev_info(1, &di) || !coli_vk_xb_init_dev(1, T.c.hidden, T.c.inter, act, T.c.act_limit, T.c.act_a, T.c.act_b)) {
        fprintf(stderr, VKT_TAG " tier %s: no expert batch on the second device (shaders?), the tier stays on %s\n",
                eng, coli_vk_device_name());
        return;
    }
    snprintf(T.d2_name, sizeof T.d2_name, "%s", di.name ? di.name : "?");
    const double GiB = 1073741824.0;
    T.d2_exp_bytes = tier_expert_bytes_at(di.buf_align ? di.buf_align : 256);
    double heap_free = di.has_budget ? (double)di.free_bytes : (double)di.local_bytes * 0.8;
    double room = heap_free - env_gb("COLI_VK_RESERVE2_GB", 0.5) * GiB, want = room;
    if (di.shares_ram) {
        double avail = (double)mem_available();
        double ram_room = avail > 0 ? avail - (double)T.c.ram_reserve - (double)T.c.dense_bytes - (double)T.c.ram_later - 2.0 * GiB : room;
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
        fprintf(stderr, VKT_TAG " tier %s: second device %s: %s, the tier stays on %s\n", eng, T.d2_name,
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
    fprintf(stderr, VKT_TAG " tier %s: second device %s, budget %s = %d experts of %s (%s), the experts after the primary device's\n",
            eng, T.d2_name, human((double)T.d2_budget, hb, sizeof hb), T.dmax[1], human((double)T.d2_exp_bytes, he, sizeof he),
            di.shares_ram ? "shared RAM" : "device memory");
}

/* The host pool (backend_vulkan.h: coli_vk_tier_host_pool): the main layers' experts that
 * neither device's memory holds, in system RAM the primary device reads by address over
 * PCIe. They count as the primary device's (dev 0): its batch takes them, and the dense
 * chain computes a layer's routed experts in its frame once every one of them is on the
 * device or here (vk_dmoe.h), where without the pool a model larger than the card ends
 * every layer's frame for the host. Only beside the exclusive RAM/VRAM books
 * (COLI_VK_TIER_EXCLUSIVE): the engine's RAM cache gives up the copies of what the
 * device holds, so the pool takes RAM the cache would have, not RAM beside it. Its size:
 * what RAM has free less 4 GiB (the engine's own growth, the page cache the loads go
 * through), at most what the devices leave; COLI_VK_TIER_HOST_GB sets it and
 * COLI_VK_TIER_HOST=0 turns it off. COLI_VK_TIER_HOST=1 turns it on where the device
 * shares the host's RAM too (tests: the same memory, the paths of a card). */
static void host_init(long long all) {
    const char *e = getenv("COLI_VK_TIER_HOST"), *g = getenv("COLI_VK_TIER_HOST_GB");
    if ((e && *e == '0') || (T.uma && !(e && !strcmp(e, "1"))) || !T.excl || !T.exp_bytes || !coli_vk_tier_host_pool()) return;
    long long left = all - T.dmax[0] - T.dmax[1];
    if (left <= 0) return;
    const double GiB = 1073741824.0;
    double want = g && *g ? atof(g) * GiB : (double)mem_available() - 4.0 * GiB - (double)T.c.ram_later;
    /* The driver's heap for the system RAM the device reads (coli_vk_tier_host_heap_room) is
     * not all the RAM: about half of it on Windows, three quarters with NVIDIA's on Linux.
     * Past its room the pool's blocks are refused, and the experts planned there were in
     * neither the pool nor the engine's cache, which had given them up (ram_given): they came
     * from disk. #2314: Kimi K3 and GLM-5.3 on 256 GB under Windows held 170 and 147 GB
     * against 214 and 223, read 1.5 to 3 times as much and decoded at 0.3 and 0.4 tok/s
     * against 0.4 and 0.6. The pool takes at most the heap's room less 2 GiB (the uploads'
     * staging lives there too); the RAM it does not take (cut) stays the engine's cache's. */
    size_t heap = coli_vk_tier_host_heap_room();
    double cut = 0;
    char hh[32] = "";
    if (heap && want > (double)heap - 2.0 * GiB) {
        cut = want - ((double)heap - 2.0 * GiB);
        want -= cut;
        human((double)heap, hh, sizeof hh);
    }
    /* what it leaves for the state the engine allocates once it serves (ram_later) */
    char hl[96] = "";
    if (T.c.ram_later && !(g && *g)) {
        char x[32];
        snprintf(hl, sizeof hl, "; %s of RAM left for the KV and state the engine allocates later",
                 human((double)T.c.ram_later, x, sizeof x));
    }
    size_t blk = (size_t)256 << 20;   /* the host pool's blocks (system RAM: no device cap shrinks them) */
    if (want < (double)T.exp_bytes || blk < T.exp_bytes) {
        if (hh[0]) fprintf(stderr, VKT_TAG " tier %s: host pool off: the driver leaves %s of system RAM for the device\n", T.engine, hh);
        else if (hl[0]) fprintf(stderr, VKT_TAG " tier %s: host pool off: no room past the RAM it leaves (%s)\n", T.engine, hl + 2);
        return;
    }
    size_t full = (size_t)want / blk, rest = (size_t)want - full * blk;
    long long fit = (long long)full * (long long)(blk / T.exp_bytes) + (long long)(rest / T.exp_bytes);
    /* Every expert, the device's too, where RAM holds them (T.full, not beside a second
     * device): a conversation's routed experts then move up from the pool to the device and
     * its coldest go back to their copy (promote). Without it the device keeps the warm
     * start's set, and on another topic than the history's it reads most of a token's
     * experts over the bus. COLI_VK_TIER_PROMOTE=0 keeps that set, =n sets the rate. */
    const char *pe = getenv("COLI_VK_TIER_PROMOTE");
    int rate = pe && *pe ? atoi(pe) : 32;
    /* A pool that holds every expert frees none (promote moves the device's copies), so its
     * experts need not the room of a bigger form's: with forms by layer it keeps each at its
     * own (h_exact), a block's end losing at most the largest. Flint UD-Q3_K_XL: 24.50 GiB in
     * rooms, 18.5 GB of experts. */
    size_t hmax = T.exp_bytes, hx = T.per_layer ? tier_expert_bytes_own(coli_vk_buffer_alignment(), &hmax) : T.exp_bytes;
    if (rate > 0 && !T.d2 && T.per_layer && hx && hx < T.exp_bytes && blk >= hmax) {
        const long long fx = (long long)(full * ((blk - hmax) / hx + 1) + rest / hx);
        if (fx >= all) { T.h_exact = 1; T.h_bytes = hx; T.h_max = hmax; fit = fx; }
    }
    T.full = rate > 0 && !T.d2 && fit >= all;
    if (!T.full) T.h_exact = 0;
    /* Short of every expert the device leaves: the CPU computes the rest from the engine's
     * RAM caches, which were sized from the same free RAM. Their least (ram_floor) comes out of
     * the pool's room and the pool's bytes out of them (ram_given); an engine that cannot lower
     * them keeps ram_reserve, and the pool takes what is left. */
    size_t given = 0;
    if (fit < left && !(g && *g)) {
        /* the caches' least out of the RAM's share (want + cut): what the heap cut off is RAM
         * beside the pool already */
        double least = T.c.ram_given ? (double)T.c.ram_floor : (double)T.c.ram_reserve;
        want -= least > cut ? least - cut : 0;
        if (want < (double)T.exp_bytes) return;
        full = (size_t)want / blk; rest = (size_t)want - full * blk;
        fit = (long long)full * (long long)(blk / T.exp_bytes) + (long long)(rest / T.exp_bytes);
        given = T.c.ram_given ? 1 : 0;
    }
    long long held = T.full ? all : fit < left ? fit : left;
    if (held < 1) { T.full = 0; return; }
    if (fit > left) fit = left;
    /* the blocks those experts take: whole blocks of blk / exp_bytes, the last one partly
     * (h_exact: of (blk - the largest) / the average and one, every one at its own size) */
    long long per = T.h_exact ? (long long)((blk - T.h_max) / T.h_bytes + 1) : (long long)(blk / T.exp_bytes), nb = (held + per - 1) / per;
    size_t lim = (size_t)nb * blk;
    if (held < per) lim = (size_t)held * (T.h_exact ? T.h_bytes : T.exp_bytes) + (T.h_exact ? T.h_max : 0) + 4096;
    coli_vk_tier_pool_limit_dev(2, lim);
    T.dmax[2] = (int)fit; T.max_resident += (int)fit; T.h_budget = lim;
    T.promote_rate = T.full ? rate : 0;
    if (given) T.c.ram_given(T.c.ram_ctx, lim);
    char hb[32];
    if (T.full)
        fprintf(stderr, VKT_TAG " tier %s: host pool on, %lld experts in %s of system RAM the device reads over PCIe (every expert "
                        "of the model: the %lld its memory leaves and a copy of its own, from which the routed ones move up, up "
                        "to %d a forward; COLI_VK_TIER_PROMOTE sets it, COLI_VK_TIER_HOST=0 off%s)\n",
                T.engine, held, human((double)lim, hb, sizeof hb), left, T.promote_rate, hl);
    else
        fprintf(stderr, VKT_TAG " tier %s: host pool on, %d experts in %s of system RAM the device reads over PCIe "
                        "(the %lld its memory leaves%s%s%s%s; COLI_VK_TIER_HOST=0 off, COLI_VK_TIER_HOST_GB sets it)\n",
                T.engine, T.dmax[2], human((double)lim, hb, sizeof hb), left, fit < left ? ", not all of them" : "",
                hh[0] ? ": the driver's heap for it leaves " : "", hh, hl);
}

int vkt_host_pool_on(void) { return T.on && T.dmax[2] > 0; }
size_t vkt_stage_room(void) { return vkt_host_pool_on() ? T.stage_room : 0; }

/* vkt_give_back: whole blocks of the tier's pool on the primary device, the coldest first (the
 * scores of the residents with a range in each; an empty one first of all), until they hold
 * `bytes`. Their residents go back to their host pool copies (T.full: the hybrid steps compute
 * them on the CPU, as promote's victims) or out of the tier; then dmax and the pool's limit
 * drop by what left, which releases the emptied blocks to the driver. The uploader is let
 * finish first, so no copy is landing in a block being emptied. */
#ifndef VKT_NO_GIVE_BACK
static int gb_cmp_heat(const void *a, const void *b) {
    const double *h = (const double *)a, *k = (const double *)b;   /* {heat, index} pairs */
    return h[0] < k[0] ? -1 : h[0] > k[0] ? 1 : (h[1] < k[1] ? -1 : h[1] > k[1]);
}
#endif
size_t vkt_give_back(size_t bytes) {
#ifdef VKT_NO_GIVE_BACK
    (void)bytes; return 0;
#else
    if (!T.on || !bytes || T.inflight) return 0;
    pthread_mutex_lock(&T.mx);
    while ((T.qn || T.busy) && T.th_on && !T.stop) pthread_cond_wait(&T.cv_done, &T.mx);
    pthread_mutex_unlock(&T.mx);
    quiesce();
    if (T.inflight || T.nevict || T.ndfree) return 0;
    const int nb = coli_vk_tier_pool_blocks(NULL, 0);
    if (nb <= 0) return 0;
    uint64_t *bb = calloc((size_t)nb, sizeof *bb); double *hp = calloc((size_t)nb, 2 * sizeof *hp); uint8_t *pick = calloc((size_t)nb, 1);
    if (!bb || !hp || !pick) { free(bb); free(hp); free(pick); return 0; }
    coli_vk_tier_pool_blocks(bb, nb);
    const size_t n = (size_t)T.c.layers * T.c.experts;
    for (int k = 0; k < nb; k++) { hp[2 * k] = 0; hp[2 * k + 1] = k; }
    for (size_t i = 0; i < n; i++) {
        VSlot *v = &T.s[i];
        if (v->state != VS_RESIDENT || v->dev != 0 || !v->ex || v->host || kind_of((int)(i / T.c.experts))) continue;
        int blk[6], m = coli_vk_xb_expert_blocks(v->ex, blk, 6);
        for (int j = 0; j < m; j++) if (blk[j] >= 0 && blk[j] < nb) hp[2 * blk[j]] += 1.0 + (double)score(v);
    }
    qsort(hp, (size_t)nb, 2 * sizeof *hp, gb_cmp_heat);
    uint64_t got = 0;
    for (int q = 0; q < nb && got < bytes; q++) { const int k = (int)hp[2 * q + 1]; if (bb[k]) { pick[k] = 1; got += bb[k]; } }
    int moved = 0, out = 0;
    for (size_t i = 0; i < n && got; i++) {
        VSlot *v = &T.s[i];
        if (v->state != VS_RESIDENT || v->dev != 0 || !v->ex || v->host || kind_of((int)(i / T.c.experts))) continue;
        int blk[6], m = coli_vk_xb_expert_blocks(v->ex, blk, 6), hit = 0;
        for (int j = 0; j < m; j++) hit |= blk[j] >= 0 && blk[j] < nb && pick[blk[j]];
        if (!hit) continue;
        if (T.full && v->hex && v->hex != v->ex) {   /* to its host pool copy, as promote's victim */
            if (T.ndfree == T.cdfree) {
                int nc = T.cdfree ? 2 * T.cdfree : 64;
                ColiVkExpert **g = realloc(T.dfree, (size_t)nc * sizeof(*g));
                if (!g) continue;
                T.dfree = g; T.cdfree = nc;
            }
            T.dfree[T.ndfree++] = v->ex;
            v->ex = v->hex; v->host = 1; T.dres[0]--; T.dres[2]++; moved++;
        } else {   /* out of the tier: the engine's CPU path has it */
            if (T.nevict == T.cevict) {
                int nc = T.cevict ? 2 * T.cevict : 64;
                int *g = realloc(T.evict, (size_t)nc * sizeof(int));
                if (!g) continue;
                T.evict = g; T.cevict = nc;
            }
            v->state = VS_EVICT; T.resident--; T.dres[0]--;
            T.evict[T.nevict++] = (int)i; out++;
        }
        T.evictions++;
        note_change((int)i);
    }
    free(hp);
    const int drop = T.dmax[0] - (T.dres[0] + T.dque[0]);
    if (drop > 0) { T.dmax[0] -= drop; T.max_resident -= drop; }
    pthread_mutex_lock(&T.mx); T.evict_pending = T.nevict + T.ndfree; pthread_mutex_unlock(&T.mx);
    quiesce();   /* nothing in flight: the device copies freed, the changes published */
    ColiVkPoolStats st; coli_vk_pool_stats(1, &st);
    const uint64_t base = st.limit ? st.limit : st.total;
    coli_vk_tier_pool_limit(base > got + 4096 ? (size_t)(base - got) : 4096);   /* releases the emptied blocks */
    uint64_t *now = calloc((size_t)nb, sizeof *now), gave = 0;
    if (now) { coli_vk_tier_pool_blocks(now, nb); for (int k = 0; k < nb; k++) if (pick[k] && !now[k]) gave += bb[k]; }
    free(now); free(pick); free(bb);
    T.budget = T.budget > gave ? T.budget - (size_t)gave : 0;
    coli_vk_pool_stats(1, &st);
    char hb[32], hc[32];
    fprintf(stderr, VKT_TAG " tier %s: %s of device memory back for the dense chain (%d experts to the host pool, %d out of the tier; "
            "the tier's pool now %s in %d blocks)\n", T.engine, human((double)gave, hb, sizeof hb), moved, out,
            human((double)st.total, hc, sizeof hc), st.blocks);
    return (size_t)gave;
#endif
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
        fprintf(stderr, VKT_TAG " tier %s: expert format %d/%d (gs %d/%d) has no device form, the experts stay on the CPU\n",
                eng, T.c.gate_up.kind, T.c.down.kind, T.c.gate_up.gs, T.c.down.gs);
        return 0;
    }
    int act = T.c.act == VKT_ACT_SITU ? COLI_VK_ACT_SITU
            : T.c.act == VKT_ACT_SWIGLU_V4 ? COLI_VK_ACT_SWIGLU_V4 : COLI_VK_ACT_SWIGLU;
    if (!coli_vk_xb_init(T.c.hidden, T.c.inter, act, T.c.act_limit, T.c.act_a, T.c.act_b)) {
        fprintf(stderr, VKT_TAG " tier %s: no expert batch on this device (shaders?), the experts stay on the CPU\n", eng);
        return 0;
    }
    int H = T.c.hidden, F = T.c.inter;
    if (!forms_per_layer()) {
        fprintf(stderr, VKT_TAG " tier %s: the layers' expert formats (more than %d forms, or one with no device form) do not "
                        "all go to the device, the experts stay on the CPU\n", eng, VKT_FORMS - 1);
        return 0;
    }
    T.exp_bytes = T.per_layer ? tier_expert_bytes_at(coli_vk_buffer_alignment()) : vkt_expert_bytes(H, F, T.c.gate_up, T.c.down);
    /* the extra layers (an MTP head's): on the tier when their form has a device form */
    if (T.c.extra_layers > 0) {
        if (form_init(1, T.c.extra_gate_up, T.c.extra_down) && T.exp_bytes &&
            (T.x_bytes = vkt_expert_bytes(H, F, T.c.extra_gate_up, T.c.extra_down)))
            T.c.layers += T.c.extra_layers;
        else
            fprintf(stderr, VKT_TAG " tier %s: the extra layers' expert format %d/%d (gs %d/%d) has no device form, their experts stay on the CPU\n",
                    eng, T.c.extra_gate_up.kind, T.c.extra_down.kind, T.c.extra_gate_up.gs, T.c.extra_down.gs);
    }
    if (T.main_layers == T.c.layers) T.c.extra_layers = 0;

    /* The budget. Discrete: what VK_EXT_memory_budget says is free on the device,
     * minus a reserve (scratch, KV mirrors, driver), the dense weights still to come and
     * what the dense chain allocates at its first forward (chain_bytes). Integrated (device memory is host RAM): a quarter of what the RAM has
     * left after the engine's expert cache has grown to its size and the dense
     * weights are placed, so the tier never takes what the CPU cache needs.
     * COLI_VK_TIER_GB replaces either, within what the device can hold. */
    const double GiB = 1073741824.0;
    double used = 0, bud = 0;
    int have = coli_vk_mem_budget(&used, &bud);
    double heap_free = have ? (bud - used) * 1e9 : (double)coli_vk_device_local_bytes() * 0.8;
    T.uma = coli_vk_device_shares_ram();
    double reserve = env_gb("COLI_VK_TIER_RESERVE_GB", 1.0) * GiB;
    double room = heap_free - reserve - (double)T.c.dense_bytes - (double)T.c.chain_bytes;
    double want;
    const char *cap = getenv("COLI_VK_TIER_GB");
    if (cap && *cap) want = atof(cap) * GiB;
    else if (T.uma) {
        double avail = (double)mem_available();
        double ram_room = avail > 0 ? avail - (double)T.c.ram_reserve - (double)T.c.dense_bytes - (double)T.c.ram_later - 2.0 * GiB : room;
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
        else if (T.c.extra_whole && (double)xall * T.x_bytes <= bud / 5) xn = xall;   /* the head's layer whole */
        else xn = (long long)(bud * T.c.extra_layers / (T.main_layers + T.c.extra_layers) / T.x_bytes);
        if (xn < 1 && bud >= (double)T.x_bytes + 2.0 * T.exp_bytes) xn = 1;
        if (xn > xall) xn = xall;
        if (xn < 1) {
            char hx[32];
            fprintf(stderr, VKT_TAG " tier %s: no room for an extra layer's expert (%s), they stay on the CPU\n",
                    eng, human((double)T.x_bytes, hx, sizeof hx));
            T.c.layers = T.main_layers; T.c.extra_layers = 0;
        } else {
            /* every expert of the extra layers: their pool's blocks hold whole experts, so it
             * takes whole blocks (512 of 5 MiB need 11 blocks of 256 MiB, not 10) */
            size_t xbytes = (size_t)xn * T.x_bytes, xb = coli_vk_block_bytes((size_t)256 << 20);
            if (xn == xall && xb >= T.x_bytes) {
                size_t per = xb / T.x_bytes;
                xbytes = (size_t)((xall + (long long)per - 1) / (long long)per) * xb;
            }
            long long take = (long long)((xbytes + T.exp_bytes - 1) / T.exp_bytes);
            if (take > fit_raw - 2) { take = (long long)(((size_t)xn * T.x_bytes + T.exp_bytes - 1) / T.exp_bytes); }
            T.xmax = (int)xn; T.x_budget = (size_t)take * T.exp_bytes; fit_raw -= take;
        }
    }
    if (fit > all) fit = all;
    if (fit > fit_raw) fit = fit_raw;
    /* streaming (below): the engine's load hook and the GEMM route, unless COLI_VK_TIER_STREAM=0 */
    ColiVkXbStats xs0; coli_vk_xb_stats(&xs0);
    const char *se0 = getenv("COLI_VK_TIER_STREAM");
    int can_stream = !(se0 && *se0 == '0') && T.c.load && T.c.release && xs0.gemm_rows > 0;
    if (fit_raw < 4 && fit < all && can_stream && T.exp_bytes && !T.xmax && room >= 4.0 * (double)T.exp_bytes) {
        /* fewer experts than streaming's least (two residents, two staging slots), with the
         * device's room for that least: stream every cold expert through it, the budget
         * raised to it (#2183: a budget below one expert, set so that a test streams; below
         * two it turned the tier off, below four it left the cold experts on the CPU) */
        char hb[32], hn[32];
        double least = 4.0 * (double)T.exp_bytes;
        fprintf(stderr, VKT_TAG " tier %s: the budget (%s) holds fewer experts than streaming needs; %s for two residents "
                        "and two staging slots, every other expert streamed\n",
                eng, human(want, hb, sizeof hb), human(least, hn, sizeof hn));
        want = least; fit = fit_raw = 4;
    }
    if (fit < 2) {
        char hb[32], he[32];
        fprintf(stderr, VKT_TAG " tier %s: no room for experts (budget %s, %s each; COLI_VK_TIER_GB sets it), the experts stay on the CPU\n",
                eng, human(want, hb, sizeof hb), human((double)T.exp_bytes, he, sizeof he));
        return 0;
    }
    if (T.c.max_experts > 0 && fit > T.c.max_experts) fit = T.c.max_experts;   /* the engine's count cap */
    /* A prompt chunk's staging (VktConfig.stage, vk_dmoe.h): where the host pool will hold
     * the experts the budget leaves, a layer's share of them and a quarter more come out of
     * it. Allocated beside a full budget, the staging took the card to its last memory and
     * the decode after a prompt lost 4% (Qwen3.8 IQ2_S on an RTX PRO 4000, 48.8 -> 46.6
     * tok/s); with the room left it kept 48.8, the prompt 112 -> 148 tok/s. */
    {
        const char *he = getenv("COLI_VK_TIER_HOST"), *xe = getenv("COLI_VK_TIER_EXCLUSIVE"), *sg = getenv("COLI_VK_STAGE");
        int pool = !(he && *he == '0') && !(T.uma && !(he && !strcmp(he, "1"))) && !(xe && *xe == '0') && coli_vk_tier_host_pool();
        int blocks = 0;   /* the staging copies block types only (vk_dmoe.h) */
        for (int l = 0; l < T.main_layers; l++) blocks |= T.form[form_of(l)].gu_fmt >= 100 || T.form[form_of(l)].dn_fmt >= 100;
        long long per = (all - fit + T.main_layers - 1) / T.main_layers;
        per += per / 4;
        if (per > T.c.experts) per = T.c.experts;
        if (T.c.stage && blocks && !(sg && *sg == '0') && pool && fit < all && per > 0 && fit - per >= 2) {
            fit -= per; fit_raw -= per;
            want -= (double)per * T.exp_bytes;
            T.stage_room = (size_t)per * T.exp_bytes;
        }
    }
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
            fprintf(stderr, VKT_TAG " tier %s: no room for streaming slots beside the residents, cold experts stay on the CPU\n", eng);
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
    fprintf(stderr, VKT_TAG " tier %s: on, %s, budget %s = %d experts of %s (fmt %d", eng,
            coli_vk_device_name(), human((double)T.budget, hb, sizeof hb), T.dmax[0],
            human((double)T.exp_bytes, he, sizeof he), T.form[0].gu_fmt);
    if (T.form[0].gu_gs) fprintf(stderr, " gs %d", T.form[0].gu_gs);
    fprintf(stderr, ", down fmt %d", T.form[0].dn_fmt);
    if (T.form[0].dn_gs) fprintf(stderr, " gs %d", T.form[0].dn_gs);
    if (T.per_layer) {
        int other = 0, forms = 0;
        for (int l = 0; l < T.main_layers; l++) other += T.lcls[l];
        for (int j = 0; j < T.nform; j++) {
            int used = 0;
            for (int l = 0; l < T.main_layers && !used; l++) used = j != 1 && T.lform[l] == j && T.lcls[l];
            forms += used;
        }
        fprintf(stderr, "; %d layer%s in %d form%s other than most layers', their experts in the room of the biggest of those, the rest at their own size",
                other, other == 1 ? "" : "s", forms, forms == 1 ? "" : "s");
    }
    if (T.c.chain_bytes) fprintf(stderr, "; %s left for the dense chain's KV mirror and a prompt chunk",
                                 human((double)T.c.chain_bytes, hb, sizeof hb));
    if (T.stage_room) fprintf(stderr, "; %s for a prompt chunk's host pool experts", human((double)T.stage_room, hb, sizeof hb));
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
    host_init(all);           /* and the rest in system RAM the device reads */
    /* hybrid steps: a decode step's or a verify's host pool experts of block types on the CPU,
     * in place, while the device computes the ones it holds (COLI_VK_HYBRID=0: the device
     * reads them over PCIe). SwiGLU, and DeepSeek V4's expert with the device's roundings
     * (hy_act_v4); Kimi K3's SiTU stays on the device. */
    const char *hy = getenv("COLI_VK_HYBRID");
    T.hybrid = T.dmax[2] > 0 && (T.c.act == VKT_ACT_SWIGLU || T.c.act == VKT_ACT_SWIGLU_V4) && !(hy && *hy == '0');
    /* and a prompt step's host pool experts of at most COLI_VK_HYBRID_BIG_ROWS rows */
    const char *hbr = getenv("COLI_VK_HYBRID_BIG_ROWS");
    T.hy_big = hbr && *hbr ? atoi(hbr) : 2;
    /* a prompt step's host pool experts of block types copied into the streaming's slots by
     * the copy engine (12.6 GB/s on a PCIe 4.0 x8 card), where the grouped GEMMs read them
     * in place at about 4 (COLI_VK_TIER_POOL_COPY=0: in place) */
    const char *pc = getenv("COLI_VK_TIER_POOL_COPY");
    T.pool_copy = T.st_ok && T.dmax[2] > 0 && !(pc && *pc == '0') &&
                  (T.bpool = calloc((size_t)T.c.experts, 1)) != NULL;
    return 1;
}

void vkt_report(const char *scope, unsigned long long ram_hits, unsigned long long disk_loads) {
    if (!T.s) return;
    ColiVkPoolStats ps; coli_vk_pool_stats(1, &ps);
    unsigned long long r = T.routed - T.rep_routed, sv = T.served - T.rep_served;
    T.rep_routed = T.routed; T.rep_served = T.served;
    double hidden = T.dev_ms > T.wait_ms ? T.dev_ms - T.wait_ms : 0;
    char hu[32], hb[32], hl[32];
    fprintf(stderr, VKT_TAG " tier %s %s: device %llu of %llu routed experts (%.1f%%; this %s %llu of %llu) | "
            "CPU RAM hits %llu, disk loads %llu | resident %d (budget %d, %s of %s, %d blocks, frag %.2f) | "
            "uploads %llu (%s, %llu warm), evictions %llu, skipped %llu queue + %llu rate, failed %llu | "
            "device %.1f ms, CPU share %.1f ms, waited %.1f ms (%.0f%% of device time hidden)",
            T.engine, scope, T.served, T.routed, T.routed ? 100.0 * T.served / T.routed : 0.0, scope, sv, r,
            ram_hits, disk_loads, T.dres[0], T.dmax[0], human((double)ps.used, hu, sizeof hu),
            human((double)T.budget, hb, sizeof hb), ps.blocks, ps.frag, T.uploads,
            human((double)T.upload_bytes, hl, sizeof hl), T.warm, T.evictions, T.qfull, T.rated, T.failed,
            T.dev_ms, T.cpu_ms, T.wait_ms, T.dev_ms > 0 ? 100.0 * hidden / T.dev_ms : 0.0);
    if (T.failed)
        fprintf(stderr, " | failed: %llu refused by the pool, %llu copies to the device, %llu not taken by the batch",
                T.f_pool, T.f_copy, T.f_sets);
    if (T.balance) fprintf(stderr, " | balance: device share %.2f, %llu rows handed to the CPU", T.share, T.handed);
    if (T.chain_served) fprintf(stderr, " | in the dense chain: %llu routed experts with no host step", T.chain_served);
    if (T.hy_steps || T.hyb_steps) {
        fprintf(stderr, " | hybrid: %llu steps, %llu host pool pairs computed by the CPU in place (%.1f ms, %.0f us a pair)",
                T.hy_steps, T.hy_pairs, T.hy_ms, T.hy_ms * 1e3 / (double)(T.hy_pairs + T.hyb_pairs));
        if (T.hyb_steps) fprintf(stderr, ", and %llu prompt steps' %llu pairs", T.hyb_steps, T.hyb_pairs);
    }
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
        if (T.pc_experts) {
            char hc[32];
            fprintf(stderr, ", %llu host pool experts copied into the slots (%s)", T.pc_experts, human(T.pc_bytes, hc, sizeof hc));
        }
    }
    if (T.dmax[2]) {
        ColiVkPoolStats ph; coli_vk_pool_stats(5, &ph);
        char hh[32], hb[32];
        if (T.full)   /* every expert, and those the device reads from it (the ones on their way up among them) */
            fprintf(stderr, " | host pool: every expert (%s of %s), %d read from it, %llu moved up to the device "
                            "(%llu residents gave way)", human((double)ph.used, hh, sizeof hh),
                    human((double)T.h_budget, hb, sizeof hb), T.dres[2], T.ups, T.up_victims);
        else
            fprintf(stderr, " | host pool: resident %d (budget %d, %s of %s)", T.dres[2], T.dmax[2],
                    human((double)ph.used, hh, sizeof hh), human((double)T.h_budget, hb, sizeof hb));
        if (T.chain_served)
            fprintf(stderr, "; of the chain's routed experts %.1f%% in the device's memory, %.1f%% of a decode step's",
                    100.0 * T.vram_routed / T.chain_served, T.dec_routed ? 100.0 * T.dec_vram / T.dec_routed : 0.0);
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

unsigned vkt_epoch(void) { return T.epoch; }
/* one expert's entry (zero where it is not on the primary device): 1 resident there, 0 not, -1 an error */
static int dev_entry(int layer, int e, uint32_t *o) {
    VSlot *v = slot(layer, e);
    memset(o, 0, COLI_VK_XB_ENTRY * sizeof *o);
    if (v->state != VS_RESIDENT || v->dev != 0 || !v->ex) return 0;
    return coli_vk_xb_expert_entry(v->ex, o) ? 1 : -1;
}
int vkt_dev_table(int layer, uint32_t *ent) {
    if (!T.on || layer < 0 || layer >= T.c.layers || !ent) return -1;
    int n = 0;
    for (int e = 0; e < T.c.experts; e++) {
        int r = dev_entry(layer, e, ent + (size_t)e * COLI_VK_XB_ENTRY);
        if (r < 0) return -1;
        n += r;
    }
    return n;
}
int vkt_dev_host(int layer, uint8_t *host) {
    if (!T.on || layer < 0 || layer >= T.c.layers || !host) return -1;
    int n = 0;
    for (int e = 0; e < T.c.experts; e++) {
        VSlot *v = slot(layer, e);
        host[e] = v->state == VS_RESIDENT && v->dev == 0 && v->ex && v->host;
        n += host[e];
    }
    return n;
}
int vkt_score(int layer, uint64_t *sc) {
    if (!T.on || layer < 0 || layer >= T.c.layers || !sc) return -1;
    for (int e = 0; e < T.c.experts; e++) sc[e] = score(slot(layer, e));
    return 0;
}
static int dev_entry_hybrid(int layer, int e, uint32_t *o, VktHostExpert *hx);
int vkt_dev_table_hybrid(int layer, uint32_t *ent, VktHostExpert *hx) {
    if (!T.on || layer < 0 || layer >= T.c.layers || !ent || !hx) return -1;
    int n = 0;
    for (int e = 0; e < T.c.experts; e++) {
        int r = dev_entry_hybrid(layer, e, ent + (size_t)e * COLI_VK_XB_ENTRY, &hx[e]);
        if (r < 0) return -1;
        n += r;
    }
    return n;
}
/* the published changes since epoch `since`: the slots (layer * experts + expert) of the
 * epochs after it, oldest first, a slot again for each change; -1 when the log does not
 * reach back to it or they are more than max (read the tables whole) */
int vkt_dev_changes(unsigned since, int *slots, int max) {
    if (!T.on || since < T.log_from || since > T.epoch) return -1;
    int k = T.nlog;
    while (k > 0 && T.log_ep[k - 1] > since) k--;
    int n = T.nlog - k;
    if (n > max) return -1;
    if (n) memcpy(slots, T.log_slot + k, (size_t)n * sizeof(int));
    return n;
}
int vkt_dev_entry(int layer, int eid, uint32_t *ent, uint32_t *ent_hybrid, VktHostExpert *hx, uint8_t *host) {
    if (!T.on || layer < 0 || layer >= T.c.layers || eid < 0 || eid >= T.c.experts || !ent) return -1;
    int r = dev_entry(layer, eid, ent);
    if (r >= 0 && ent_hybrid && hx && dev_entry_hybrid(layer, eid, ent_hybrid, hx) < 0) r = -1;
    VSlot *v = slot(layer, eid);
    if (host) *host = v->state == VS_RESIDENT && v->dev == 0 && v->ex && v->host;
    return r;
}
/* one expert's entry in the hybrid steps' table (zero for one the CPU computes, hx its rows):
 * 1 resident on the primary device, 0 not, -1 an error */
static int dev_entry_hybrid(int layer, int e, uint32_t *o, VktHostExpert *hx) {
    VSlot *v = slot(layer, e);
    memset(o, 0, COLI_VK_XB_ENTRY * sizeof *o);
    memset(hx, 0, sizeof *hx);
    if (v->state != VS_RESIDENT || v->dev != 0 || !v->ex) return 0;
    if (host_view(layer, e, hx)) return 1;
    return coli_vk_xb_expert_entry(v->ex, o) ? 1 : -1;
}
/* issue()'s books for pairs it never saw: a forward's start, the tick, heat, routed and
 * served (the device took every resident one) */
void vkt_dev_count(int layer, int S, int K, const int *idx) {
    if (!T.on || layer < 0 || layer >= T.c.layers || S < 1 || K < 1 || !idx) return;
    if (layer < T.last_layer || T.begin) { T.begin = 0; T.first_layer = layer; new_forward(); }
    if (T.routed_at) {
        uint32_t q = ++T.seq;
        for (int i = 0; i < S * K; i++)
            if (idx[i] >= 0 && idx[i] < T.c.experts) __atomic_store_n(&T.routed_at[(size_t)layer * T.c.experts + idx[i]], q, __ATOMIC_RELAXED);
        __atomic_store_n(&T.layer_seq[layer], q, __ATOMIC_RELEASE);
    }
    if (layer == T.first_layer) {
        T.tick += (uint32_t)S;
        long cap = (long)T.promo_cap + (long)T.rate * S;
        T.promo_cap = cap > (1 << 30) ? 1 << 30 : (int)cap;
    }
    T.last_layer = layer;
    for (int i = 0; i < S * K; i++) {
        int e = idx[i];
        if (e < 0 || e >= T.c.experts) continue;
        T.routed++;
        if (layer >= T.main_layers) T.extra_routed++;
        VSlot *v = slot(layer, e);
        if (v->heat < 0xFFFFFFFFu) v->heat++;
        v->last = T.tick;
        if (v->state != VS_RESIDENT || v->dev != 0) continue;
        T.served++; T.chain_served++;
        if (layer >= T.main_layers) T.extra_served++;
        if (!v->host) T.vram_routed++;
        if (S > VKT_UP_ROWS) continue;
        T.dec_routed++; T.dec_vram += !v->host;
        if (v->host) promote(layer, e);   /* the frame is done: a victim's device copy is read by nothing now */
    }
}
/* T.full: the promotions that landed into the slot table, at a frame's start (vk_dmoe.h)
 * where nothing reads the experts: the table read next has them. */
void vkt_settle(void) { if (T.on && T.full) quiesce(); }
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
        for (size_t i = 0; i < n; i++) {
            if (T.s[i].hex && T.s[i].hex != T.s[i].ex) coli_vk_xb_expert_free(T.s[i].hex);
            T.s[i].hex = NULL;
            if (T.s[i].ex) { coli_vk_xb_expert_free(T.s[i].ex); T.s[i].ex = NULL; }
        }
        for (int i = 0; T.st && i < T.st_n; i++) if (T.st[i].ex) coli_vk_xb_expert_free(T.st[i].ex);
    }
    pthread_mutex_destroy(&T.mx); pthread_cond_destroy(&T.cv); pthread_cond_destroy(&T.cv_room); pthread_cond_destroy(&T.cv_done);
    /* the slot table outlives the tier by one shutdown: a prefetch worker may be in
     * vkt_ram_first with the pointer it read before this */
    free(g_old_slots); g_old_slots = T.s;
    free(g_old_routed); g_old_routed = T.routed_at;   /* the same for the routing marks */
    free(g_old_seq); g_old_seq = T.layer_seq;
    free(T.grp); free(T.map); free(T.touched); free(T.bex); free(T.brows); free(T.bfirst);
    free(T.bx); free(T.by); free(T.bw); free(T.evict); free(T.done); free(T.dfree);
    free(T.st); free(T.sy); free(T.bcnt); free(T.bofs); free(T.bcls); free(T.blist); free(T.pred); free(T.pred_ok);
    for (int h = 0; h < 2; h++) { free(T.hf[h].slot); free(T.hf[h].yout); }
    free(T.d2_idx); free(T.d2_y); free(T.d2_bex); free(T.d2_brows); free(T.d2_bx); free(T.d2_bw);
    free(T.hy_of); free(T.hy_v); free(T.hy_hx); free(T.hy_x); free(T.hy_gu); free(T.hy_hh); free(T.hy_out); free(T.hy_w); free(T.hy_xb); free(T.hy_src); free(T.bpool);
    memset(&T, 0, sizeof T);
}
#endif /* COLI_VULKAN */
