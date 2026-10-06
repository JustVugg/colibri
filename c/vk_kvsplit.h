/* vk_kvsplit.h -- a chain engine's KV cache past the device's budget: the part read
 * most on the device, the rest in the host's RAM (header-only; a *_chain.h includes it
 * after vk_chain.h).
 *
 * The chain keeps the host's KV cache canonical and mirrors each attention layer's
 * cache on the device behind a watermark (docs/vulkan.md, "The dense chain"). A long
 * context makes those mirrors larger than the device can hold. Then each split layer
 * keeps `rows` = ns * B rows on the device: ns slots of B positions, nw of them a window
 * over the newest blocks (the step's own rows always land there) and np pinned to the
 * blocks the engine's selection (QSA, DSA) reads most. The rest of the positions are
 * only in the host's cache, which holds every row anyway. A step's attention then runs
 * in two parts at once: the device over its share (vkc_kvs_attn / vkc_kvs_mla, frame
 * F2) and the CPU over the rest (vkc_kv_host_attn, from the queries frame F1 brought
 * down), and vkc_kvs_merge joins them through their softmax statistics (frame F3): the
 * full attention up to the order of the sums. A step whose rows see nothing outside the
 * device's share takes one frame as before (VKC_KVS_FIN).
 *
 * The device's share of a row depends on the row's position only: the `anchor` + 1
 * newest blocks up to the row's (all of them in the window while a step holds the row,
 * since a step writes at most `chunk` rows), plus, from a list, the pinned blocks. The
 * host takes the rest, in chunks fixed by position. So a row's bits do not depend on
 * how a forward is cut into steps (a prompt in one chunk or many, a decode step or a
 * verify, a resumed prefix or a cold one), except through the pins, which follow the
 * reads: pins are off by default, COLI_VK_KV_PIN=1 opts in. DeepSeek stages only
 * selected cold rows to preserve its complete sparse attention on the device.
 *
 * The plan (vkc_kv_plan): whole mirrors when they fit the device's budget (the chain
 * as it was, bit for bit), else the split, sized from the budget and resized as the
 * host's cache grows. Knobs:
 *   COLI_VK_KV_SPLIT=0          never split: past the budget the chain declines as before
 *   COLI_VK_KV_DEVICE_ROWS=N    keep N rows per layer on the device whenever the cache is
 *                               larger (the tests' small device; a measurement's split)
 *   COLI_VK_KV_BLOCK=B          positions per block (default 64)
 *   COLI_VK_KV_PIN=1            opt in to read-based pins (GQA/MLA rounding can then
 *                               depend on read history; the default is deterministic)
 *   COLI_VK_KV_COLD=device      the host's part on the device too (vkc_kv_shadow): each
 *                               split layer keeps a copy of the host's rows in memory the
 *                               device reads in place, kept up to date row by row, and the
 *                               device computes the same positions the CPU would; on a
 *                               device that can import host memory (not with staged
 *                               uploads), without pins. Unset: the CPU, as before
 *
 * The residency (VkcKvTab, per layer): bt (block -> slot or -1) and rb (slot -> block
 * or -1), uploaded to `tab` when they change; `valid`, the watermark: every resident
 * row below it equals the host's. A step from pos_base first uploads the resident rows
 * in [valid, pos_base) and the whole of a block that just got its slot (vkc_kv_push),
 * then writes its new rows into their slots (vkc_kv_store); a CPU step lowers valid
 * (vkc_kv_lower); a new host cache resets the tables (vkc_kv_reset). */
#ifndef COLI_VK_KVSPLIT_H
#define COLI_VK_KVSPLIT_H
#include "vk_chain.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

typedef struct {
    int nblk, wb0, wb1;            /* blocks the table covers; the window's first and last block */
    int *bt, *rb;                  /* block -> slot (-1), slot -> block (-1) */
    unsigned char *pin, *fresh;    /* per slot: pinned by reads; given its block since the last push */
    unsigned *hits;                /* per block: reads seen in the selection lists */
    unsigned long long counted;    /* lists counted (the hits halve every 128) */
    int valid;                     /* resident rows below it equal the host's */
    int dirty;                     /* bt / rb changed since their upload */
} VkcKvTab;

/* One array of a layer's cache: per position nseg segments of seglen floats (the KV heads
 * of a head-major cache; 1 segment for a position-major one). The host's: segment h of
 * position t at host + h*host_seg + t*seglen. The device's: segment h of row r at
 * dev_off + (h*rows + r)*seglen in dev. */
typedef struct {
    int nseg, seglen;
    const float *host; size_t host_seg;
    VkcBuf *dev; size_t dev_off;
} VkcKvPart;

/* COLI_VK_KV_COLD=device: a split layer's host rows in memory the device reads in place,
 * per array (K and V, or the latent and the rope key): nseg segments of cap positions of
 * seglen floats, head-major (segment g, position t at (g*cap + t)*seglen), page-aligned and
 * imported once (b, off floats into it); rows below `valid` equal the host's. */
typedef struct {
    float *p[2]; VkcBuf *b[2]; size_t off[2], n[2];
    int nseg[2], seglen[2], cap, valid;
} VkcKvShadow;
typedef struct {
    int on;                        /* split: every split layer holds `rows` rows on the device */
    int B, ns, nw, np, rows, nl;   /* block size, slots (window, pinned), rows = ns*B, layers */
    int cap;                       /* host positions the tables cover */
    int chunk;                     /* the most rows a step may write */
    int anchor;                    /* a row's device share: blocks pos/B - anchor .. pos/B */
    VkcKvTab *t;
    VkcBuf *tab; size_t tab_stride;   /* per layer: bt[cap/B+1] then rb[ns], ints */
    VkcBuf *dpart, *cpart;         /* the device's part (F2) and the host's (written in F3) */
    VkcBuf *qd, *sd;               /* DOWN: the queries and the lists the host's part reads */
    VkcBuf *cold_rows, *cold_list; /* selected sparse rows from RAM, reused one query at a time */
    float *hpart; size_t hpart_n;
    int *cold, *lo, *cnt, *causal; size_t cold_n, lo_n;
    unsigned long long steps, splits, host_pos, pins, staged_rows; double host_ms, wait_ms;
    int cold_dev;                  /* COLI_VK_KV_COLD=device: the host's part on the device, from sh */
    VkcKvShadow *sh;               /* per split layer (cold_dev) */
    VkcBuf *cscr;                  /* its chunks' parts (vkc_attn_part_chunks), joined into cpart */
    unsigned long long dev_cold, shadow_rows; size_t shadow_bytes; double shadow_ms;
    const char *engine;
} VkcKvSplit;

static double vkc_kv_now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }

static void vkc_kv_shadow_free(VkcKvSplit *ks) {
    if (ks->sh) for (int i = 0; i < ks->nl; i++)
        for (int k = 0; k < 2; k++) {
            vkc_free(ks->sh[i].b[k]);   /* after the frames that read it, before its pages go */
            free(ks->sh[i].p[k]);
        }
    free(ks->sh); ks->sh = NULL; ks->shadow_bytes = 0;
    vkc_free(ks->cscr); ks->cscr = NULL;
}
static void vkc_kv_free(VkcKvSplit *ks) {
    vkc_kv_shadow_free(ks);
    if (ks->t) for (int i = 0; i < ks->nl; i++) {
        VkcKvTab *t = &ks->t[i];
        free(t->bt); free(t->rb); free(t->pin); free(t->fresh); free(t->hits);
    }
    free(ks->t); ks->t = NULL;
    vkc_free(ks->tab); vkc_free(ks->dpart); vkc_free(ks->cpart); vkc_free(ks->qd); vkc_free(ks->sd);
    vkc_free(ks->cold_rows); vkc_free(ks->cold_list); ks->cold_rows = ks->cold_list = NULL;
    ks->tab = ks->dpart = ks->cpart = ks->qd = ks->sd = NULL;
    free(ks->hpart); free(ks->cold); free(ks->lo); free(ks->cnt); free(ks->causal);
    ks->hpart = NULL; ks->cold = ks->lo = ks->cnt = ks->causal = NULL; ks->hpart_n = ks->cold_n = ks->lo_n = 0;
    ks->on = 0;
}

/* Every table empty: nothing resident (a new host cache, or a reset). */
static void vkc_kv_reset(VkcKvSplit *ks) {
    if (!ks->on) return;
    for (int i = 0; i < ks->nl; i++) {
        VkcKvTab *t = &ks->t[i];
        for (int b = 0; b < t->nblk; b++) t->bt[b] = -1;
        for (int j = 0; j < ks->ns; j++) { t->rb[j] = -1; t->pin[j] = 0; t->fresh[j] = 0; }
        memset(t->hits, 0, (size_t)t->nblk * sizeof *t->hits);
        t->valid = 0; t->dirty = 1; t->wb0 = t->wb1 = 0; t->counted = 0;
        if (ks->sh) ks->sh[i].valid = 0;
    }
}

static long vkc_kv_env(const char *name, long def) {
    const char *e = getenv(name);
    return e && *e ? atol(e) : def;
}

/* Decide the device's share of `nl` layers' caches of `cap` positions, `row_bytes` a
 * position and layer on the device (every array), steps of up to `chunk` rows; `selects`:
 * the engine picks positions (pinned slots pay off); `held`: the bytes of the mirrors the
 * engine holds now (freed before the new ones). Returns 1 for whole mirrors (ks->on = 0:
 * the chain as before), 2 for the split (ks->on = 1, tables ready, every block in the
 * host's RAM until a step places it), 0 when not even the split can run. */
/* The current device's free bytes: the driver's budget less what is in use, else the
 * local heap less the weights and the chain's buffers. */
static size_t vkc_dev_avail(void) {
    double used = 0, bud = 0;
    if (coli_vk_mem_budget_dev(vkc_device_now(), &used, &bud)) return bud > used ? (size_t)((bud - used) * 1e9) : 0;
    size_t dev = coli_vk_device_local_bytes_dev(vkc_device_now()), w = 0, n = 0;
    VkcStats st; vkc_stats(&st);
    coli_vk_mem_info_dev(vkc_device_now(), &w, &n);
    return dev > w + st.dev_bytes ? dev - w - st.dev_bytes : 0;
}
/* vkc_kv_plan_need: the same with the whole mirrors' bytes given (need_bytes; 0: nl * cap
 * * row_bytes), for layers whose caches differ (DeepSeek's compressed rows, one ratio a
 * layer): the decision compares those bytes with the budget, the tables cover cap
 * positions and the split keeps the same rows in every layer. */
static int vkc_kv_plan_need(VkcKvSplit *ks, const char *engine, int nl, size_t row_bytes, int cap, int chunk, int selects,
                            size_t held, size_t need_bytes) {
    vkc_kv_free(ks);
    ks->engine = engine;
    if (nl < 1 || cap < 1 || !row_bytes) return 1;
    size_t need = need_bytes ? need_bytes : (size_t)nl * (size_t)cap * row_bytes;
    long forced = vkc_kv_env("COLI_VK_KV_DEVICE_ROWS", 0);
    int split_ok = vkc_kv_env("COLI_VK_KV_SPLIT", 1) != 0 && vkc_kvs_ready();
    int B = (int)vkc_kv_env("COLI_VK_KV_BLOCK", 64);
    if (B < 1) B = 1;
    long rows;
    if (forced > 0) {
        if (forced >= cap) return 1;
        rows = forced;
    } else {
        size_t avail = vkc_dev_avail() + held;
        size_t room = avail / 10 * 8;          /* a fifth for scratch, frames and the rest */
        if (need <= room) return 1;
        rows = (long)(room / ((size_t)nl * row_bytes));
    }
    if (!split_ok) return 1;                     /* as before: the whole mirrors, or the decline */
    (void)chunk;   /* the step's rows follow the split (ks->chunk), so that a row's share is the same in every step */
    int ns = (int)(rows / B);
    if (ns < 3) ns = 3;
    int nw = ns, np = 0;
    if (selects && vkc_kv_env("COLI_VK_KV_PIN", 0) != 0) { nw = ns / 2 > 3 ? ns / 2 : 3; np = ns - nw; }
    if ((long)ns * B >= cap) return 1;
    ks->B = B; ks->ns = ns; ks->nw = nw; ks->np = np; ks->rows = ns * B; ks->nl = nl; ks->cap = cap;
    /* A step writes at most cb blocks' worth of rows, so its rows span at most cb blocks
     * past its first. A row's share reaches back `anchor` blocks: every block of it stays
     * in the window while a step holds the row (anchor <= nw - 1 - cb), and it covers the
     * step's earlier rows (anchor >= cb), which reach the host's cache only after the
     * step. cb is the largest both allow: a prompt's chunk is a step, and every chunk
     * carries its own frames and expert loads (with an eighth of the window, Qwen3.6's
     * 7.6K-token prompt on a Radeon 780M took 170 s to its first token at 1024 rows on the
     * device and 183 s at 4096; with this, 122 s and 142 s; docs/vulkan.md). */
    int cb = (nw - 1) / 2 > 1 ? (nw - 1) / 2 : 1;
    ks->chunk = cb * B;
    ks->anchor = nw - 1 - cb;
    int nblk = (cap + B - 1) / B + 1;
    ks->tab_stride = (size_t)nblk + (size_t)ns;
    ks->t = (VkcKvTab *)calloc((size_t)nl, sizeof *ks->t);
    if (!ks->t) return 0;
    for (int i = 0; i < nl; i++) {
        VkcKvTab *t = &ks->t[i];
        t->nblk = nblk;
        t->bt = (int *)malloc((size_t)nblk * sizeof(int)); t->rb = (int *)malloc((size_t)ns * sizeof(int));
        t->pin = (unsigned char *)calloc((size_t)ns, 1); t->fresh = (unsigned char *)calloc((size_t)ns, 1);
        t->hits = (unsigned *)calloc((size_t)nblk, sizeof(unsigned));
        if (!t->bt || !t->rb || !t->pin || !t->fresh || !t->hits) { vkc_kv_free(ks); return 0; }
    }
    ks->tab = vkc_buf((size_t)nl * ks->tab_stride * sizeof(int), VKC_DEV);
    if (!ks->tab) { vkc_kv_free(ks); return 0; }
    { const char *cm = getenv("COLI_VK_KV_COLD");
      ks->cold_dev = cm && !strcmp(cm, "device") && np == 0 && coli_vk_import_alignment() > 0;
      if (cm && !strcmp(cm, "device") && !ks->cold_dev)
          fprintf(stderr, "[VK] %s chain: COLI_VK_KV_COLD=device needs host memory the device reads in place and no pins; "
                          "the CPU computes the host's part\n", engine);
      if (ks->cold_dev && !(ks->sh = (VkcKvShadow *)calloc((size_t)nl, sizeof *ks->sh))) { vkc_kv_free(ks); return 0; } }
    ks->on = 1;
    vkc_kv_reset(ks);
    fprintf(stderr, "[VK] %s chain: the KV cache split past the device's budget: %d of %d positions a layer on the device "
                    "(%d blocks of %d, %d for pins by reads; a row's newest %d positions on the device), the rest in RAM "
                    "(%.1f MiB on the device instead of %.1f)%s\n",
            engine, ks->rows, cap, ns, B, np, (ks->anchor + 1) * B, (double)nl * ks->rows * row_bytes / 1048576.0,
            need / 1048576.0, ks->cold_dev ? ", its attention on the device too (COLI_VK_KV_COLD=device)" : "");
    return 2;
}
static int vkc_kv_plan(VkcKvSplit *ks, const char *engine, int nl, size_t row_bytes, int cap, int chunk, int selects,
                       size_t held) {
    return vkc_kv_plan_need(ks, engine, nl, row_bytes, cap, chunk, selects, held, 0);
}

/* A slot for a block of layer li: a free one, else the oldest block behind the window
 * that nothing pinned, else the pinned block behind the window read least. */
static int vkc_kv_take(VkcKvSplit *ks, VkcKvTab *t) {
    int best = -1;
    for (int j = 0; j < ks->ns; j++) if (t->rb[j] < 0) return j;
    for (int j = 0; j < ks->ns; j++)
        if (!t->pin[j] && t->rb[j] < t->wb0 && (best < 0 || t->rb[j] < t->rb[best])) best = j;
    if (best >= 0) return best;
    for (int j = 0; j < ks->ns; j++)
        if (t->pin[j] && t->rb[j] < t->wb0 && (best < 0 || t->hits[t->rb[j]] < t->hits[t->rb[best]])) best = j;
    return best;
}
static void vkc_kv_assign(VkcKvTab *t, int j, int b, int pinned) {
    if (t->rb[j] >= 0) t->bt[t->rb[j]] = -1;
    t->rb[j] = b; t->bt[b] = j; t->pin[j] = (unsigned char)pinned; t->fresh[j] = 1; t->dirty = 1;
}

/* Place the window of a step writing [pos_base, pos_base + S), S <= ks->chunk: its
 * blocks and the ones its rows' shares reach back to get slots. */
static void vkc_kv_place(VkcKvSplit *ks, int li, int pos_base, int S) {
    VkcKvTab *t = &ks->t[li];
    /* the shadow too: rows from pos_base on may be rewritten from here (this step's, or a
     * new request's after a rewind, whether or not this step has a host part) */
    if (ks->sh && ks->sh[li].valid > pos_base) ks->sh[li].valid = pos_base > 0 ? pos_base : 0;
    if (pos_base + S <= 0) { t->wb0 = t->wb1 = 0; return; }
    int B = ks->B, wb1 = (pos_base + S - 1) / B, wb0 = wb1 - ks->nw + 1;
    if (wb0 < 0) wb0 = 0;
    if (wb1 >= t->nblk) wb1 = t->nblk - 1;
    t->wb0 = wb0; t->wb1 = wb1;
    for (int j = 0; j < ks->ns; j++)            /* blocks past the step (a rewind): nothing reads them */
        if (t->rb[j] > wb1) { t->bt[t->rb[j]] = -1; t->rb[j] = -1; t->pin[j] = 0; t->dirty = 1; }
    for (int b = wb0; b <= wb1; b++) {
        if (t->bt[b] >= 0) continue;
        int j = vkc_kv_take(ks, t);
        if (j >= 0) vkc_kv_assign(t, j, b, 0);
    }
}

/* Selection engines, after vkc_kv_run (the step's host part has used the tables): count
 * the step's reads and pin up to two of the blocks read most behind the newest row's
 * share (anchored: the GQA and MLA lists; else DeepSeek's, whose share is the
 * residency, so only blocks in the host's RAM), ahead of pinned blocks read less.
 * vkc_kv_count: lists as the GQA/MLA ops read them (row s: sel[s*sel_row] = count, the
 * positions after; a negative count reads every position); vkc_kv_count_fixed:
 * DeepSeek's (cnt entries a row, no count; entries below base are the window's, the
 * others index e - base). */
static void vkc_kv_promote(VkcKvSplit *ks, VkcKvTab *t, int anchored) {
    if (++t->counted % 128 == 0) for (int b = 0; b < t->nblk; b++) t->hits[b] >>= 1;
    int lim = anchored ? t->wb1 - ks->anchor : t->wb0;
    for (int k = 0; k < 2; k++) {
        int best = -1;
        for (int b = 0; b < lim && b < t->nblk; b++) {
            int j = t->bt[b];
            if (!t->hits[b] || (j >= 0 && (t->pin[j] || !anchored))) continue;
            if (best < 0 || t->hits[b] > t->hits[best]) best = b;
        }
        if (best < 0) return;
        int npin = 0, low = -1;
        for (int j = 0; j < ks->ns; j++) if (t->pin[j]) { npin++; if (low < 0 || t->hits[t->rb[j]] < t->hits[t->rb[low]]) low = j; }
        if (npin >= ks->np) {
            if (low < 0 || t->hits[t->rb[low]] >= t->hits[best]) return;
            t->pin[low] = 0; t->dirty = 1;    /* behind the window it may now be evicted; in it, it stays */
        }
        int j = t->bt[best];
        if (j >= 0) { t->pin[j] = 1; t->dirty = 1; ks->pins++; continue; }   /* on the device already */
        for (int q = 0; q < ks->ns && j < 0; q++) if (t->rb[q] < 0) j = q;
        for (int q = 0; q < ks->ns; q++)
            if (!t->pin[q] && t->rb[q] >= 0 && t->rb[q] < t->wb0 && (j < 0 || (t->rb[j] >= 0 && t->rb[q] < t->rb[j]))) j = q;
        if (j < 0) return;
        vkc_kv_assign(t, j, best, 1);
        ks->pins++;
    }
}
static void vkc_kv_count(VkcKvSplit *ks, int li, const int *sel, int S, int sel_row) {
    if (!ks->on || ks->np < 1 || !sel) return;
    VkcKvTab *t = &ks->t[li];
    for (int s = 0; s < S; s++) {
        const int *r = sel + (size_t)s * sel_row;
        for (int j = 0; j < r[0]; j++) if (r[1 + j] >= 0 && r[1 + j] / ks->B < t->nblk) t->hits[r[1 + j] / ks->B]++;
    }
    vkc_kv_promote(ks, t, 1);
}
static void vkc_kv_count_fixed(VkcKvSplit *ks, int li, const int *sel, int S, int sel_row, int cnt, int base) {
    if (!ks->on || ks->np < 1 || !sel) return;
    VkcKvTab *t = &ks->t[li];
    for (int s = 0; s < S; s++) {
        const int *r = sel + (size_t)s * sel_row;
        for (int j = 0; j < cnt; j++) if (r[j] >= base && (r[j] - base) / ks->B < t->nblk) t->hits[(r[j] - base) / ks->B]++;
    }
    vkc_kv_promote(ks, t, 0);
}

/* Record the uploads that make layer li's resident rows below pos_base the host's, and
 * its table when it changed. Into the open frame. */
static int vkc_kv_push(VkcKvSplit *ks, int li, const VkcKvPart *pt, int npart, int pos_base) {
    VkcKvTab *t = &ks->t[li];
    int B = ks->B, ok = 1;
    for (int j = 0; j < ks->ns && ok; j++) {
        int b = t->rb[j];
        if (b < 0) continue;
        int t0 = t->fresh[j] ? b * B : (t->valid > b * B ? t->valid : b * B), t1 = b * B + B < pos_base ? b * B + B : pos_base;
        t->fresh[j] = 0;
        if (t1 <= t0) continue;
        for (int k = 0; k < npart && ok; k++)
            for (int h = 0; h < pt[k].nseg && ok; h++)
                ok = vkc_write(pt[k].dev, pt[k].dev_off + ((size_t)h * ks->rows + (size_t)j * B + (t0 - b * B)) * pt[k].seglen,
                               pt[k].host + h * pt[k].host_seg + (size_t)t0 * pt[k].seglen,
                               (size_t)(t1 - t0) * pt[k].seglen * sizeof(float));
    }
    t->valid = pos_base;
    if (ok && t->dirty) {   /* bt with bit 30 on the pinned blocks, then rb */
        size_t o = (size_t)li * ks->tab_stride;
        int *bt = (int *)malloc((size_t)t->nblk * sizeof(int));
        if (!bt) return 0;
        for (int b = 0; b < t->nblk; b++) bt[b] = t->bt[b] >= 0 && t->pin[t->bt[b]] ? (t->bt[b] | 0x40000000) : t->bt[b];
        ok = vkc_write(ks->tab, o, bt, (size_t)t->nblk * sizeof(int)) &&
             vkc_write(ks->tab, o + (size_t)t->nblk, t->rb, (size_t)ks->ns * sizeof(int));
        free(bt);
        t->dirty = !ok;
    }
    return ok;
}

/* Record the copies of a step's new rows into their slots: row s, segment h at src_off
 * + s*src_row + h*src_seg in src (seglen floats each). */
static int vkc_kv_store(VkcKvSplit *ks, int li, const VkcKvPart *pt, VkcBuf *src, size_t src_off, size_t src_row,
                        size_t src_seg, int pos_base, int S) {
    VkcKvTab *t = &ks->t[li];
    VkcRegion *rg = (VkcRegion *)malloc(sizeof *rg * (size_t)S * pt->nseg);
    if (!rg) return 0;
    int n = 0;
    for (int s = 0; s < S; s++) {
        int p = pos_base + s, j = t->bt[p / ks->B];
        if (j < 0) { free(rg); return 0; }   /* the window holds every row of a step */
        for (int h = 0; h < pt->nseg; h++)
            rg[n++] = (VkcRegion){pt->dev_off + ((size_t)h * ks->rows + (size_t)j * ks->B + p % ks->B) * pt->seglen,
                                  src_off + (size_t)s * src_row + (size_t)h * src_seg, (size_t)pt->seglen};
    }
    int ok = vkc_copy_regions(pt->dev, src, rg, n);
    free(rg);
    return ok;
}
static void vkc_kv_done(VkcKvSplit *ks, int li, int end) { if (ks->on) ks->t[li].valid = end; }
static void vkc_kv_lower(VkcKvSplit *ks, int li, int pos_base) {
    if (ks->on && ks->t[li].valid > pos_base) ks->t[li].valid = pos_base;
    if (ks->on && ks->sh && ks->sh[li].valid > pos_base) ks->sh[li].valid = pos_base;
}

/* 1 when a row of the step may read a position outside the device's share: causal
 * rows see [first, pos] (first = pos - win + 1 with a window, else kv_start), the share
 * starts at block pos/B - anchor; with lists (lists 1) the same, conservatively. */
static int vkc_kv_cold(const VkcKvSplit *ks, int li, int pos_base, int S, int win, int kv_start, int lists) {
    (void)li; (void)lists;
    for (int s = 0; s < S; s++) {
        int pos = pos_base + s, first = win > 0 ? pos - win + 1 : kv_start;
        if (first < kv_start) first = kv_start;
        if (first < 0) first = 0;
        if ((pos / ks->B - ks->anchor) * ks->B > first) return 1;
    }
    return 0;
}

/* 1 when a block below index n is not on the device (DeepSeek's compressed rows: any of
 * them may be listed). */
static int vkc_kv_cold_upto(const VkcKvSplit *ks, int li, int n) {
    const VkcKvTab *t = &ks->t[li];
    for (int b = 0; b * ks->B < n && b < t->nblk; b++) if (t->bt[b] < 0) return 1;
    return 0;
}
static int vkc_kv_rows_alloc(VkcKvSplit *ks, int S, size_t need) {
    if (ks->lo_n < (size_t)S) {
        int *a = (int *)realloc(ks->lo, (size_t)S * sizeof(int));
        if (a) ks->lo = a;
        int *b = a ? (int *)realloc(ks->cnt, (size_t)S * sizeof(int)) : NULL;
        if (b) ks->cnt = b;
        int *c = b ? (int *)realloc(ks->causal, (size_t)S * sizeof(int)) : NULL;
        if (!c) return 0;
        ks->causal = c; ks->lo_n = (size_t)S;
    }
    if (ks->cold_n < need) {
        int *c = (int *)realloc(ks->cold, (need ? need : 1) * sizeof(int));
        if (!c) return 0;
        ks->cold = c; ks->cold_n = need;
    }
    return 1;
}
/* The host's entries of DeepSeek's lists: row s reads ks->cold[ks->lo[s] + j], j <
 * ks->cnt[s]: its entries e >= base (sel[s*sel_row + j], j < cnt) whose index e - base
 * is in a block the device does not hold, as e - base, in list order. */
static int vkc_kv_cold_fixed(VkcKvSplit *ks, int li, int S, const int *sel, int sel_row, int cnt, int base) {
    const VkcKvTab *t = &ks->t[li];
    if (!vkc_kv_rows_alloc(ks, S, (size_t)S * (cnt > 0 ? cnt : 1))) return 0;
    size_t nc = 0;
    for (int s = 0; s < S; s++) {
        const int *r = sel + (size_t)s * sel_row;
        ks->lo[s] = (int)nc; ks->cnt[s] = 0; ks->causal[s] = 0;
        for (int j = 0; j < cnt; j++) {
            int e = r[j] - base;
            if (r[j] >= base && (e / ks->B >= t->nblk || t->bt[e / ks->B] < 0)) { ks->cold[nc++] = e; ks->cnt[s]++; }
        }
    }
    return 1;
}

/* The host's positions for each row of the step: row s reads ks->cold[ks->lo[s] + j],
 * j < ks->cnt[s]. Causal (sel NULL, or a row whose count is negative): [first, the
 * start of the row's share) in order (first as in vkc_kv_cold); listed: the row's
 * entries outside its share (older than its anchored blocks and not pinned), in list
 * order. */
static int vkc_kv_cold_rows(VkcKvSplit *ks, int li, int pos_base, int S, int win, int kv_start, const int *sel, int sel_row) {
    const VkcKvTab *t = &ks->t[li];
    int B = ks->B, end = pos_base + S, start = kv_start > 0 ? kv_start : 0;
    if (!vkc_kv_rows_alloc(ks, S, (size_t)end + (sel ? (size_t)S * sel_row : 0))) return 0;
    for (int p = start; p < end; p++) ks->cold[p - start] = p;   /* the causal list: every position, ascending */
    size_t nc = (size_t)(end > start ? end - start : 0);
    for (int s = 0; s < S; s++) {
        int pos = pos_base + s, first = win > 0 ? pos - win + 1 : start, a = pos / B - ks->anchor;
        if (first < start) first = start;
        const int *r = sel ? sel + (size_t)s * sel_row : NULL;
        if (r && r[0] >= 0) {
            ks->lo[s] = (int)nc; ks->cnt[s] = 0; ks->causal[s] = 0;
            for (int j = 0; j < r[0]; j++) {
                int p = r[1 + j], b = p / B;
                if (p < 0 || p > pos || b >= a) continue;
                if (b < t->nblk && t->bt[b] >= 0 && t->pin[t->bt[b]]) continue;
                ks->cold[nc++] = p; ks->cnt[s]++;
            }
            continue;
        }
        int lo = a * B > first ? a * B : first;
        ks->lo[s] = first - start; ks->cnt[s] = lo - first; ks->causal[s] = 1;
    }
    return 1;
}

/* The host's part: rows s < S, heads h < H in groups of G sharing one key row (G = H /
 * KV heads; MLA: H). Head h of row s queries q1 (d1 floats at q1 + s*q1_row + h*q1_seg)
 * and q2 (d2, may be 0); position t of group g has keys k1 (k1 + g*k1_grp + t*k1_pos), k2
 * and the value v (dv floats). Positions: ks->cold / lo / cnt (vkc_kv_cold_rows).
 * Writes acc [S][H][dv] (unnormalized) and (m, l) [S][H][2]: the device's layout. */
typedef struct {
    int S, H, G, d1, d2, dv; float scale;
    const float *q1; size_t q1_row, q1_seg;
    const float *q2; size_t q2_row, q2_seg;
    const float *k1; size_t k1_grp, k1_pos;
    const float *k2; size_t k2_grp, k2_pos;
    const float *v;  size_t v_grp, v_pos;
    /* Inkling: score = tau[s] * (q . k * scale + bias), bias = sum_j r[s*r_row + h*r_seg + j]
     * * relp[j*ext + d] for d = pos_base + s - t < ext (NULL tau / r: none) */
    const float *tau, *r; size_t r_row, r_seg; const float *relp; int d_rel, ext, pos_base;
    int bf16w;   /* DeepSeek V4: the value sum's weights rounded to bf16 (two passes a row) */
} VkcKvHostAttn;
static float vkc_kv_bf16(float f) {   /* nearest even, as coli_bf16_round */
    uint32_t u; memcpy(&u, &f, 4);
    if ((u & 0x7f800000u) != 0x7f800000u) u += 0x7fffu + ((u >> 16) & 1u);
    u &= 0xffff0000u; memcpy(&f, &u, 4);
    return f;
}
/* a dot product in eight running sums (vectorizable without reassociating a single sum) */
static inline float vkc_kv_dot(const float *a, const float *b, int n) {
    float s[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    int d = 0;
    for (; d + 8 <= n; d += 8) for (int k = 0; k < 8; k++) s[k] += a[d + k] * b[d + k];
    float t = ((s[0] + s[4]) + (s[1] + s[5])) + ((s[2] + s[6]) + (s[3] + s[7]));
    for (; d < n; d++) t += a[d] * b[d];
    return t;
}
/* one position's score for head h of row s */
static inline float vkc_kv_score(const VkcKvHostAttn *p, int s, int h, int grp, int t, const float *qa, const float *qb) {
    float a = vkc_kv_dot(qa, p->k1 + (size_t)grp * p->k1_grp + (size_t)t * p->k1_pos, p->d1);
    if (qb) a += vkc_kv_dot(qb, p->k2 + (size_t)grp * p->k2_grp + (size_t)t * p->k2_pos, p->d2);
    float v = a * p->scale;
    if (p->r) {
        int dist = p->pos_base + s - t;
        float bias = 0.f;
        if (dist < p->ext) {
            const float *rr = p->r + (size_t)s * p->r_row + (size_t)h * p->r_seg;
            for (int j = 0; j < p->d_rel; j++) bias += rr[j] * p->relp[(size_t)j * p->ext + dist];
        }
        v = v + bias;
    }
    return p->tau ? p->tau[s] * v : v;
}
#define VKC_KV_TILE 64
#define VKC_KV_HCH 2048   /* the host's chunks: positions [k*HCH, (k+1)*HCH) of a causal row, HCH entries of a list */
/* o = o (+) c: two partials (m, l, acc[dv]) joined through their statistics */
static void vkc_kv_join(float *o, const float *c, int dv) {
    if (c[0] <= -1.0e38f) return;
    float mn = o[0] > c[0] ? o[0] : c[0];
    float ea = o[0] > -1.0e38f ? expf(o[0] - mn) : 0.f, eb = expf(c[0] - mn);
    for (int d = 0; d < dv; d++) o[2 + d] = o[2 + d] * ea + c[2 + d] * eb;
    o[1] = o[1] * ea + c[1] * eb; o[0] = mn;
}
/* row s's chunks and the entries [*j0, *j1) of chunk c */
static int vkc_kv_nchunks(const VkcKvSplit *ks, int s, int bf16w) {
    int n = ks->cnt[s];
    if (bf16w || n < 1) return 1;
    if (!ks->causal[s]) return (n + VKC_KV_HCH - 1) / VKC_KV_HCH;
    int p0 = ks->cold[ks->lo[s]];
    return (p0 + n - 1) / VKC_KV_HCH - p0 / VKC_KV_HCH + 1;
}
static void vkc_kv_chunk(const VkcKvSplit *ks, int s, int bf16w, int c, long *j0, long *j1) {
    int n = ks->cnt[s];
    if (bf16w || n < 1) { *j0 = 0; *j1 = n; return; }
    if (!ks->causal[s]) { *j0 = (long)c * VKC_KV_HCH; *j1 = *j0 + VKC_KV_HCH < n ? *j0 + VKC_KV_HCH : n; return; }
    long p0 = ks->cold[ks->lo[s]], a = (p0 / VKC_KV_HCH + c) * VKC_KV_HCH;
    *j0 = a - p0 > 0 ? a - p0 : 0;
    *j1 = a + VKC_KV_HCH - p0 < n ? a + VKC_KV_HCH - p0 : n;
}
/* One chunk's partial for the G heads of group grp of row s, from nothing, into o. */
static void vkc_kv_host_chunk(const VkcKvSplit *ks, const VkcKvHostAttn *p, int s, int grp, long j0, long j1, float *o) {
    int G = p->G, dv = p->dv;
    const int *pos = ks->cold + ks->lo[s];
    float sc[VKC_KV_TILE];
    for (int g = 0; g < G; g++) {
        float *og = o + (size_t)g * (dv + 2);
        og[0] = -3.0e38f; og[1] = 0.f;
        memset(og + 2, 0, (size_t)dv * sizeof(float));
    }
    if (j1 <= j0) return;
    if (p->bf16w) {   /* two passes: the row's max, then the weights rounded against it */
        for (int g = 0; g < G; g++) {
            int h = grp * G + g;
            const float *qa = p->q1 + (size_t)s * p->q1_row + (size_t)h * p->q1_seg;
            const float *qb = p->d2 ? p->q2 + (size_t)s * p->q2_row + (size_t)h * p->q2_seg : NULL;
            float *og = o + (size_t)g * (dv + 2), *ac = og + 2, mx = -3.0e38f, l = 0.f;
            for (long j = j0; j < j1; j++) { float v = vkc_kv_score(p, s, h, grp, pos[j], qa, qb); if (v > mx) mx = v; }
            for (long j = j0; j < j1; j++) {
                float e = expf(vkc_kv_score(p, s, h, grp, pos[j], qa, qb) - mx), w = vkc_kv_bf16(e);
                const float *vr = p->v + (size_t)grp * p->v_grp + (size_t)pos[j] * p->v_pos;
                l += e;
                for (int d = 0; d < dv; d++) ac[d] += w * vr[d];
            }
            og[0] = mx; og[1] = l;
        }
        return;
    }
    for (long t0 = j0; t0 < j1; t0 += VKC_KV_TILE) {
        int n = (int)(j1 - t0 < VKC_KV_TILE ? j1 - t0 : VKC_KV_TILE);
        for (int g = 0; g < G; g++) {
            int h = grp * G + g;
            const float *qa = p->q1 + (size_t)s * p->q1_row + (size_t)h * p->q1_seg;
            const float *qb = p->d2 ? p->q2 + (size_t)s * p->q2_row + (size_t)h * p->q2_seg : NULL;
            float mt = -3.0e38f;
            for (int i = 0; i < n; i++) {
                sc[i] = vkc_kv_score(p, s, h, grp, pos[t0 + i], qa, qb);
                if (sc[i] > mt) mt = sc[i];
            }
            float *og = o + (size_t)g * (dv + 2), *ac = og + 2;
            float mn = og[0] > mt ? og[0] : mt;
            float corr = og[0] > -1.0e38f ? expf(og[0] - mn) : 0.f;
            float l = og[1] * corr;
            if (corr != 1.f) for (int d = 0; d < dv; d++) ac[d] *= corr;
            for (int i = 0; i < n; i++) {
                float e = expf(sc[i] - mn);
                const float *vr = p->v + (size_t)grp * p->v_grp + (size_t)pos[t0 + i] * p->v_pos;
                l += e;
                for (int d = 0; d < dv; d++) ac[d] += e * vr[d];
            }
            og[0] = mn; og[1] = l;
        }
    }
}
/* Each (row, head)'s chunks are joined in chunk order from nothing, whatever runs them:
 * a task per (row, group) when there are enough of them to fill the threads, else a task
 * per chunk and the joins after: the same bits either way, and for a causal row the
 * same chunks whatever the step around it. */
static void vkc_kv_host_attn(VkcKvSplit *ks, const VkcKvHostAttn *p, float *acc, float *st) {
    int S = p->S, H = p->H, G = p->G, NG = H / G, dv = p->dv;
    long total = 0;
    for (int s = 0; s < S; s++) total += ks->cnt[s];
    total *= NG;
    ks->host_pos += (unsigned long long)total;
    int nth = 1;
#ifdef _OPENMP
    nth = omp_get_max_threads();
#endif
    size_t per = (size_t)G * (dv + 2);
    long *tstart = (long *)malloc((size_t)(S * NG + 1) * sizeof(long));
    if (!tstart) { fprintf(stderr, "[VK] %s chain: out of memory for the host's attention\n", ks->engine); exit(1); }
    long k = 0;
    for (int s = 0; s < S; s++) for (int g = 0; g < NG; g++) { tstart[s * NG + g] = k; k += vkc_kv_nchunks(ks, s, p->bf16w); }
    tstart[S * NG] = k;
    int by_row = S * NG >= 2 * nth;
    long ntask = by_row ? (long)S * NG : k;
    float *tm = (float *)malloc((size_t)ntask * per * sizeof(float));
    if (!tm) { fprintf(stderr, "[VK] %s chain: out of memory for the host's attention\n", ks->engine); exit(1); }
    double work = (double)total * G * (p->d1 + p->d2 + p->dv);   /* a team only for work worth waking it */
    #pragma omp parallel for schedule(dynamic, 1) if (ntask > 1 && work > 4e6)
    for (long task = 0; task < ntask; task++) {
        float *out = tm + (size_t)task * per;
        if (by_row) {
            int s = (int)(task / NG), grp = (int)(task % NG), nch = vkc_kv_nchunks(ks, s, p->bf16w);
            float *cbuf = (float *)malloc(per * sizeof(float));
            if (!cbuf) { fprintf(stderr, "[VK] %s chain: out of memory for the host's attention\n", ks->engine); exit(1); }
            for (int g = 0; g < G; g++) { float *og = out + (size_t)g * (dv + 2); og[0] = -3.0e38f; og[1] = 0.f; memset(og + 2, 0, (size_t)dv * sizeof(float)); }
            for (int c = 0; c < nch; c++) {
                long j0, j1;
                vkc_kv_chunk(ks, s, p->bf16w, c, &j0, &j1);
                vkc_kv_host_chunk(ks, p, s, grp, j0, j1, cbuf);
                for (int g = 0; g < G; g++) vkc_kv_join(out + (size_t)g * (dv + 2), cbuf + (size_t)g * (dv + 2), dv);
            }
            free(cbuf);
            continue;
        }
        int sg = 0;
        { int a = 0, z = S * NG; while (a + 1 < z) { int m = (a + z) / 2; if (tstart[m] <= task) a = m; else z = m; } sg = a; }
        int s = sg / NG, grp = sg % NG;
        long j0, j1;
        vkc_kv_chunk(ks, s, p->bf16w, (int)(task - tstart[sg]), &j0, &j1);
        vkc_kv_host_chunk(ks, p, s, grp, j0, j1, out);
    }
    for (int s = 0; s < S; s++) for (int grp = 0; grp < NG; grp++) for (int g = 0; g < G; g++) {
        int h = grp * G + g;
        float *o = (float *)malloc((size_t)(dv + 2) * sizeof(float));
        if (!o) { fprintf(stderr, "[VK] %s chain: out of memory for the host's attention\n", ks->engine); exit(1); }
        o[0] = -3.0e38f; o[1] = 0.f; memset(o + 2, 0, (size_t)dv * sizeof(float));
        if (by_row) vkc_kv_join(o, tm + (size_t)(s * NG + grp) * per + (size_t)g * (dv + 2), dv);
        else for (long task = tstart[s * NG + grp]; task < tstart[s * NG + grp + 1]; task++)
            vkc_kv_join(o, tm + (size_t)task * per + (size_t)g * (dv + 2), dv);
        memcpy(acc + ((size_t)s * H + h) * dv, o + 2, (size_t)dv * sizeof(float));
        st[((size_t)s * H + h) * 2] = o[0]; st[((size_t)s * H + h) * 2 + 1] = o[1];
        free(o);
    }
    free(tstart); free(tm);
}

/* Partials of `part_n` floats: the device's (F2) in ks->dpart, the host's in ks->cpart.
 * Reserve them with the scratch, before a frame records anything that binds them. */
static int vkc_kv_parts(VkcKvSplit *ks, size_t part_n) {
    if (!vkc_reserve(&ks->dpart, part_n * sizeof(float), VKC_DEV) || !vkc_reserve(&ks->cpart, part_n * sizeof(float), VKC_DEV))
        return 0;
    if (ks->hpart_n < part_n) {
        float *h = (float *)realloc(ks->hpart, part_n * sizeof(float));
        if (!h) return 0;
        ks->hpart = h; ks->hpart_n = part_n;
    }
    return 1;
}

/* The split step of one layer. The open frame F1 holds everything up to the attention.
 * Copies the host part's inputs down (n_q floats of q from q_off, then n_q2 of q2 from
 * q2_off when q2, into ks->qd; n_sel ints of sel from sel_off, when sel, into ks->sd),
 * submits F1, records dev(u) in a frame F2 and submits it, waits for F1, runs host(u, q,
 * sel) while the device runs F2 (it fills ks->hpart; q is ks->qd's floats, q2's after
 * the first n_q), and opens F3 with ks->hpart uploaded to ks->cpart (part_n floats,
 * vkc_kv_parts first). Returns 1 with F3 open, 0 when the device failed. */
static int vkc_kv_run2(VkcKvSplit *ks, VkcBuf *q, size_t q_off, size_t n_q, VkcBuf *q2, size_t q2_off, size_t n_q2,
                       VkcBuf *sel, size_t sel_off, size_t n_sel,
                       int (*dev)(void *u), int (*host)(void *u, const float *q, const int *sel), void *u, size_t part_n) {
    if (!q2) n_q2 = 0;
    if (!vkc_reserve(&ks->qd, (n_q + n_q2 ? n_q + n_q2 : 1) * sizeof(float), VKC_DOWN) ||
        (sel && !vkc_reserve(&ks->sd, (n_sel ? n_sel : 1) * sizeof(int), VKC_DOWN))) return 0;
    if (!vkc_copy(ks->qd, 0, q, q_off, n_q) || (n_q2 && !vkc_copy(ks->qd, n_q, q2, q2_off, n_q2)) ||
        (sel && !vkc_copy(ks->sd, 0, sel, sel_off, n_sel)) || !vkc_submit(0)) return 0;
    unsigned long long f1 = vkc_serial();
    if (!vkc_begin() || !dev(u) || !vkc_submit(0)) return 0;
    double t0 = vkc_kv_now();
    if (!vkc_wait_serial(f1)) return 0;
    double t1 = vkc_kv_now();
    int ok = host(u, (const float *)vkc_ptr(ks->qd), sel ? (const int *)vkc_ptr(ks->sd) : NULL);
    ks->wait_ms += t1 - t0; ks->host_ms += vkc_kv_now() - t1; ks->splits++;
    return ok && vkc_begin() && vkc_write(ks->cpart, 0, ks->hpart, part_n * sizeof(float));
}
static int vkc_kv_run(VkcKvSplit *ks, VkcBuf *q, size_t q_off, size_t n_q, VkcBuf *sel, size_t sel_off, size_t n_sel,
                      int (*dev)(void *u), int (*host)(void *u, const float *q, const int *sel), void *u, size_t part_n) {
    return vkc_kv_run2(ks, q, q_off, n_q, NULL, 0, 0, sel, sel_off, n_sel, dev, host, u, part_n);
}

/* ---- the host's part on the device (COLI_VK_KV_COLD=device) ------------------------
 * Layer li's shadow up to row `upto` (exclusive: a step's cold positions are all below its
 * first row, which the host holds): nparts arrays, part k's segment g of position t at
 * host[k] + g*grp[k] + t*pstride[k], seglen[k] floats. Made at the first use, sized to the
 * tables' positions; rows from `valid` on are copied (a step lowers `valid` to its first
 * row: the rows from there may be rewritten, by it or by a rollback). NULL when the memory
 * or the import fails (the CPU's part then, for good on this layer). */
static VkcKvShadow *vkc_kv_shadow(VkcKvSplit *ks, int li, int nparts, const float *const *host, const size_t *grp,
                                  const size_t *pstride, const int *nseg, const int *seglen, int upto) {
    VkcKvShadow *sh = &ks->sh[li];
    if (sh->cap < 0) return NULL;
    if (!sh->cap) {
        size_t al = coli_vk_import_alignment();
        for (int k = 0; k < nparts; k++) {
            size_t n = (size_t)nseg[k] * ks->cap * seglen[k], bytes = (n * sizeof(float) + al - 1) / al * al;
            void *p = NULL;
            if (!al || posix_memalign(&p, al, bytes)) p = NULL;
            sh->p[k] = (float *)p;
            if (p) { memset(p, 0, bytes); sh->b[k] = vkc_host(p, bytes, &sh->off[k]); }
            if (!p || !sh->b[k]) {
                fprintf(stderr, "[VK] %s chain: no shadow for layer %d's host rows (%.1f MiB), the CPU computes its part\n",
                        ks->engine, li, bytes / 1048576.0);
                for (int j = 0; j <= k; j++) { vkc_free(sh->b[j]); free(sh->p[j]); sh->b[j] = NULL; sh->p[j] = NULL; }
                sh->cap = -1;
                return NULL;
            }
            sh->off[k] /= sizeof(float); sh->n[k] = n; sh->nseg[k] = nseg[k]; sh->seglen[k] = seglen[k];
            ks->shadow_bytes += bytes;
        }
        sh->cap = ks->cap; sh->valid = 0;
    }
    if (upto > sh->cap) upto = sh->cap;
    if (sh->valid > upto) sh->valid = upto;
    if (sh->valid < upto) {
        double t0 = vkc_kv_now();
        long rows = upto - sh->valid;
        for (int k = 0; k < nparts; k++) {
            int G = nseg[k], L = seglen[k];
            #pragma omp parallel for collapse(2) schedule(static) if (rows * G * L > (1 << 20))
            for (int g = 0; g < G; g++)
                for (long t = sh->valid; t < upto; t++)
                    memcpy(sh->p[k] + ((size_t)g * sh->cap + t) * L, host[k] + g * grp[k] + t * pstride[k], (size_t)L * sizeof(float));
        }
        ks->shadow_rows += (unsigned long long)rows; ks->shadow_ms += vkc_kv_now() - t0;
        sh->valid = upto;
    }
    return sh;
}

/* ---- one layer's attention over the split cache, the forms the engines use ----------- */
/* The device's attention of S rows in slices of rows, each slice in a submission of its
 * own (the rows are independent: the same bits), so that no one submission runs long
 * enough for a GPU watchdog (amdgpu resets a ring after 10 s; one 512-row prompt chunk
 * over a few thousand positions per row came near that on a Radeon 780M). A slice has at
 * most COLI_VK_KV_SLICE (default 2^30) score and value multiply-adds; per row: H heads,
 * up to n positions, w floats a position. rec(u, s0, ns) records rows [s0, s0 + ns);
 * between slices the open frame is submitted and a new one opened, so the caller's frame
 * is open on return as it was on entry. */
static int vkc_kv_sliced(int S, int H, long n, int w, int (*rec)(void *u, int s0, int ns), void *u) {
    double per = (double)H * (n > 0 ? n : 1) * (w > 0 ? w : 1);
    double lim = (double)vkc_kv_env("COLI_VK_KV_SLICE", 1L << 30);
    int rows = per >= lim ? 1 : (int)(lim / per);
    if (rows >= S) return rec(u, 0, S);
    for (int s0 = 0; s0 < S; s0 += rows) {
        if (s0 && (!vkc_submit(0) || !vkc_begin())) return 0;
        if (!rec(u, s0, S - s0 < rows ? S - s0 : rows)) return 0;
    }
    return 1;
}
/* GQA (vkc_attn_w's form). q at s*q_row + h*q_seg of q (q_n floats a step's rows bring
 * down); the output at s*o_row + h*vd of out; the gate (gated) at g_off + s*g_row +
 * h*g_seg; lists at sel_off + s*sel_row of sel (sel_row 0: causal). The host's cache:
 * K of group g, position t at Kh + g*k_grp + t*k_pos (V likewise). */
typedef struct {
    VkcKvSplit *ks; int li;
    VkcBuf *q, *kc, *vc, *gate, *sel, *snk, *out;
    int S, H, KVH, hd, vd, pos_base, win, kv_pm, sink, sink_off;
    int q_row, q_seg, gated, g_off, g_row, g_seg, o_row, sel_off, sel_row;
    float scale;
    const float *Kh, *Vh; size_t k_grp, k_pos, v_grp, v_pos;
} VkcKvGqa;
/* rows [s0, s0 + ns) of the device's part (fin: of the output) */
static int vkc_kv_gqa_rec(VkcKvGqa *a, int s0, int ns, int fin) {
    VkcKvSplit *ks = a->ks;
    int bt = (int)((size_t)a->li * ks->tab_stride), nb = ks->t[a->li].nblk;
    VkcKvsAttn p = {ns, a->H, a->KVH, a->hd, a->vd, a->pos_base + s0, ks->rows, s0 * a->q_row, a->q_row, a->q_seg,
                    a->sel_off + s0 * a->sel_row, a->sel_row, a->scale, 0, 0, a->win, a->kv_pm, a->sink, a->sink_off, ks->B, ks->ns,
                    bt, nb, ks->anchor, 0, 0, 0, 0, a->g_off + s0 * a->g_row, a->g_row, a->g_seg};
    if (fin) { p.o_off = s0 * a->o_row; p.o_row = a->o_row; p.flags = VKC_KVS_FIN | (a->gated ? VKC_KVS_GATE : 0); }
    else { p.o_off = s0 * a->H * a->vd; p.o_row = a->H * a->vd; p.st_off = a->S * a->H * a->vd + s0 * a->H * 2; }
    return vkc_kvs_attn(a->q, a->kc, a->vc, fin ? a->out : ks->dpart, fin ? a->gate : NULL, a->sel, a->snk, ks->tab, &p);
}
static int vkc_kv_gqa_part(void *u, int s0, int ns) { return vkc_kv_gqa_rec((VkcKvGqa *)u, s0, ns, 0); }
/* rows [s0, s0 + ns) of the host's part on the device, from the shadow, into ks->cpart */
static int vkc_kv_gqa_cold(void *u, int s0, int ns) {
    VkcKvGqa *a = (VkcKvGqa *)u; VkcKvSplit *ks = a->ks; VkcKvShadow *sh = &ks->sh[a->li];
    int bt = (int)((size_t)a->li * ks->tab_stride), nb = ks->t[a->li].nblk;
    VkcKvsAttn p = {ns, a->H, a->KVH, a->hd, a->vd, a->pos_base + s0, sh->cap, s0 * a->q_row, a->q_row, a->q_seg,
                    a->sel_off + s0 * a->sel_row, a->sel_row, a->scale, (int)sh->off[0], (int)sh->off[1], a->win, 0, 0, 0,
                    ks->B, ks->ns, bt, nb, ks->anchor, s0 * a->H * a->vd, a->H * a->vd, a->S * a->H * a->vd + s0 * a->H * 2,
                    VKC_KVS_COLD, 0, 0, 0};
    return vkc_kvs_attn(a->q, sh->b[0], sh->b[1], ks->cpart, NULL, a->sel, NULL, ks->tab, &p);
}
/* The same without lists through the blocked attention (vkc_attn_part_chunks): the
 * shadow's K and V rows read once per block of rows, not once per (head, row), and the
 * positions in chunks of VKC_KV_CCH, one workgroup each, joined in order (vkc_kvs_join):
 * a decode row's thousands of positions over many workgroups. The chunks are fixed by
 * position, so a row's bits do not depend on the step around it. Rows go in slices whose
 * chunk parts fit 64 MiB. */
#define VKC_KV_CCH 512
static int vkc_kv_gqa_cold_blocked(VkcKvGqa *a) {
    VkcKvSplit *ks = a->ks; VkcKvShadow *sh = &ks->sh[a->li];
    int H = a->H, vd = a->vd, last = a->pos_base + a->S - 1;
    int end = (last / ks->B - ks->anchor) * ks->B;   /* the last row's cold positions end there */
    int nz = end > 0 ? (end + VKC_KV_CCH - 1) / VKC_KV_CCH : 0;
    size_t per_row = (size_t)(nz > 0 ? nz : 1) * H * (vd + 2) * sizeof(float);
    int rows = (int)(((size_t)64 << 20) / per_row);
    if (rows < 1) rows = 1;
    if (rows > a->S) rows = a->S;
    if (!vkc_reserve(&ks->cscr, (size_t)rows * per_row, VKC_DEV)) return 0;
    for (int r0 = 0; r0 < a->S; r0 += rows) {
        int n = a->S - r0 < rows ? a->S - r0 : rows, o_z = n * H * vd, st_z = n * H * 2, sa_off = nz * o_z;
        if (nz > 0) {
            VkcAttnW w; memset(&w, 0, sizeof w);
            w.a = (VkcAttn){n, H, a->KVH, a->hd, a->pos_base + r0, sh->cap, r0 * a->q_row, a->q_row, a->q_seg, 0, 0, 0, 0,
                            0, H * vd, 0, 0, a->scale, (int)sh->off[0], (int)sh->off[1]};
            w.win = a->win; w.vd = vd;
            if (!vkc_attn_part_chunks(a->q, sh->b[0], sh->b[1], ks->cscr, &w, ks->B, ks->anchor, sa_off, VKC_KV_CCH, nz,
                                      o_z, st_z)) return 0;
        }
        VkcKvsJoin j = {n * H, vd, nz, 0, o_z, sa_off, st_z, r0 * H * vd, a->S * H * vd + r0 * H * 2};
        if (!vkc_kvs_join(ks->cscr, ks->cpart, &j)) return 0;
    }
    return 1;
}
static int vkc_kv_gqa_fin(void *u, int s0, int ns) { return vkc_kv_gqa_rec((VkcKvGqa *)u, s0, ns, 1); }
static long vkc_kv_span(const VkcKvSplit *ks, int win, int sel_row) {   /* the most positions a row's device part reads */
    long n = (long)(ks->anchor + 1) * ks->B;
    if (win > 0 && win < n) n = win;
    if (sel_row > 0 && sel_row > n) n = sel_row;
    return n;
}
static int vkc_kv_gqa_dev(void *u) {
    VkcKvGqa *a = (VkcKvGqa *)u;
    return vkc_kv_sliced(a->S, a->H, vkc_kv_span(a->ks, a->win, a->sel_row), a->hd + a->vd, vkc_kv_gqa_part, a);
}
static int vkc_kv_gqa_host(void *u, const float *q, const int *sel) {
    VkcKvGqa *a = (VkcKvGqa *)u; VkcKvSplit *ks = a->ks;
    if (!vkc_kv_cold_rows(ks, a->li, a->pos_base, a->S, a->win, 0, sel, a->sel_row)) return 0;
    VkcKvHostAttn h = {a->S, a->H, a->H / a->KVH, a->hd, 0, a->vd, a->scale, q, (size_t)a->q_row, (size_t)a->q_seg,
                       NULL, 0, 0, a->Kh, a->k_grp, a->k_pos, NULL, 0, 0, a->Vh, a->v_grp, a->v_pos,
                       NULL, NULL, 0, 0, NULL, 0, 0, 0, 0};
    vkc_kv_host_attn(ks, &h, ks->hpart, ks->hpart + (size_t)a->S * a->H * a->vd);
    return 1;
}
/* Recorded where the engine's attention op would be, into the open frame; returns with
 * a frame open (the same one, or F3 after a split). */
static int vkc_kv_gqa(VkcKvGqa *a) {
    VkcKvSplit *ks = a->ks;
    if (a->S > ks->chunk) {   /* a row's share would reach past the window: the engine must clamp its steps */
        fprintf(stderr, "[VK] %s chain: a step of %d rows past the split's %d\n", ks->engine, a->S, ks->chunk);
        return 0;
    }
    ks->steps++;
    if (!vkc_kv_cold(ks, a->li, a->pos_base, a->S, a->win, 0, a->sel_row > 0))
        return vkc_kv_sliced(a->S, a->H, vkc_kv_span(ks, a->win, a->sel_row), a->hd + a->vd, vkc_kv_gqa_fin, a);
    size_t part = (size_t)a->S * a->H * (a->vd + 2);
    if (ks->cold_dev) {   /* the host's part on the device too: one frame, no round trip */
        const float *host[2] = {a->Kh, a->Vh};
        size_t grp[2] = {a->k_grp, a->v_grp}, ps[2] = {a->k_pos, a->v_pos};
        int nseg[2] = {a->KVH, a->KVH}, sl[2] = {a->hd, a->vd};
        if (vkc_kv_shadow(ks, a->li, 2, host, grp, ps, nseg, sl, a->pos_base)) {
            ks->dev_cold++; ks->splits++;   /* a step with a host part, on the device */
            VkcKvsMerge mg = {a->S * a->H, a->H, a->vd, 0, a->H * a->vd, a->vd, a->S * a->H * a->vd, 0, a->S * a->H * a->vd,
                              0, a->o_row, a->vd, a->gated ? VKC_KVS_GATE : 0, a->g_off, a->g_row, a->g_seg};
            return vkc_kv_parts(ks, part) && vkc_kv_gqa_dev(a) &&
                   (a->sel_row > 0 ? vkc_kv_sliced(a->S, a->H, (long)a->pos_base + a->S, a->hd + a->vd, vkc_kv_gqa_cold, a)
                                   : vkc_kv_gqa_cold_blocked(a)) &&
                   vkc_kvs_merge(ks->dpart, ks->cpart, a->gate, a->out, &mg);
        }
    }
    if (!vkc_kv_parts(ks, part) ||
        !vkc_kv_run(ks, a->q, 0, (size_t)a->S * a->q_row, a->sel_row > 0 ? a->sel : NULL, (size_t)a->sel_off,
                    (size_t)a->S * a->sel_row, vkc_kv_gqa_dev, vkc_kv_gqa_host, a, part)) return 0;
    VkcKvsMerge mg = {a->S * a->H, a->H, a->vd, 0, a->H * a->vd, a->vd, a->S * a->H * a->vd, 0, a->S * a->H * a->vd,
                      0, a->o_row, a->vd, a->gated ? VKC_KVS_GATE : 0, a->g_off, a->g_row, a->g_seg};
    if (!vkc_kvs_merge(ks->dpart, ks->cpart, a->gate, a->out, &mg)) return 0;
    if (a->sel_row > 0) vkc_kv_count(ks, a->li, (const int *)vkc_ptr(ks->sd), a->S, a->sel_row);
    return 1;
}

/* The MLA core (vkc_mla_core's form): qa at qa_off + s*qa_row + h*qa_seg of qa (K
 * floats), qr at qr_off + s*qr_row + h*qr_seg of qr (R floats; may be qa's buffer), the
 * latent context at o_off + s*o_row + h*o_seg of out; lists at sel_off + s*sel_row of
 * sel. The host's cache: the latent of position t at Lh + t*K, the rope key at Rh + t*R. */
typedef struct {
    VkcKvSplit *ks; int li;
    VkcBuf *qa, *qr, *lat, *rope, *sel, *out;
    int S, H, K, R, pos_base, kv_start;
    int qa_off, qa_row, qa_seg, qr_off, qr_row, qr_seg, o_off, o_row, o_seg, sel_off, sel_row;
    float scale;
    const float *Lh, *Rh;
    size_t nqa;                    /* (set by vkc_kv_mla) the qa floats F1 brings down */
} VkcKvMla;
static int vkc_kv_mla_rec(VkcKvMla *a, int s0, int ns, int fin) {
    VkcKvSplit *ks = a->ks;
    int bt = (int)((size_t)a->li * ks->tab_stride), nb = ks->t[a->li].nblk;
    VkcKvsMla p = {ns, a->H, a->K, a->R, a->pos_base + s0, a->kv_start, a->qa_off + s0 * a->qa_row, a->qa_row, a->qa_seg,
                   a->qr_off + s0 * a->qr_row, a->qr_row, a->qr_seg, 0, a->K, 0, a->R, a->sel_off + s0 * a->sel_row, a->sel_row,
                   0, 0, a->K, a->scale, ks->B, ks->ns, bt, nb, ks->anchor, 0, 0};
    if (fin) { p.o_off = a->o_off + s0 * a->o_row; p.o_row = a->o_row; p.o_seg = a->o_seg; p.flags = VKC_KVS_FIN; }
    else { p.o_off = s0 * a->H * a->K; p.o_row = a->H * a->K; p.st_off = a->S * a->H * a->K + s0 * a->H * 2; }
    return vkc_kvs_mla(a->qa, a->qr, a->lat, a->rope, a->sel, fin ? a->out : ks->dpart, ks->tab, &p);
}
static int vkc_kv_mla_part(void *u, int s0, int ns) { return vkc_kv_mla_rec((VkcKvMla *)u, s0, ns, 0); }
static int vkc_kv_mla_cold(void *u, int s0, int ns) {
    VkcKvMla *a = (VkcKvMla *)u; VkcKvSplit *ks = a->ks; VkcKvShadow *sh = &ks->sh[a->li];
    int bt = (int)((size_t)a->li * ks->tab_stride), nb = ks->t[a->li].nblk;
    VkcKvsMla p = {ns, a->H, a->K, a->R, a->pos_base + s0, a->kv_start, a->qa_off + s0 * a->qa_row, a->qa_row, a->qa_seg,
                   a->qr_off + s0 * a->qr_row, a->qr_row, a->qr_seg, (int)sh->off[0], a->K, (int)sh->off[1], a->R,
                   a->sel_off + s0 * a->sel_row, a->sel_row, s0 * a->H * a->K, a->H * a->K, a->K, a->scale, ks->B, ks->ns,
                   bt, nb, ks->anchor, VKC_KVS_COLD, a->S * a->H * a->K + s0 * a->H * 2};
    return vkc_kvs_mla(a->qa, a->qr, sh->b[0], a->R > 0 ? sh->b[1] : NULL, a->sel, ks->cpart, ks->tab, &p);
}
static int vkc_kv_mla_fin(void *u, int s0, int ns) { return vkc_kv_mla_rec((VkcKvMla *)u, s0, ns, 1); }
static int vkc_kv_mla_dev(void *u) {
    VkcKvMla *a = (VkcKvMla *)u;
    return vkc_kv_sliced(a->S, a->H, vkc_kv_span(a->ks, 0, a->sel_row), 2 * a->K + a->R, vkc_kv_mla_part, a);
}
static int vkc_kv_mla_host(void *u, const float *q, const int *sel) {
    VkcKvMla *a = (VkcKvMla *)u; VkcKvSplit *ks = a->ks;
    if (!vkc_kv_cold_rows(ks, a->li, a->pos_base, a->S, 0, a->kv_start, sel, a->sel_row)) return 0;
    VkcKvHostAttn h = {a->S, a->H, a->H, a->K, a->R, a->K, a->scale, q, (size_t)a->qa_row, (size_t)a->qa_seg,
                       q + a->nqa, (size_t)a->qr_row, (size_t)a->qr_seg, a->Lh, 0, (size_t)a->K, a->Rh, 0, (size_t)a->R,
                       a->Lh, 0, (size_t)a->K, NULL, NULL, 0, 0, NULL, 0, 0, 0, 0};
    vkc_kv_host_attn(ks, &h, ks->hpart, ks->hpart + (size_t)a->S * a->H * a->K);
    return 1;
}
static int vkc_kv_mla(VkcKvMla *a) {
    VkcKvSplit *ks = a->ks;
    if (a->S > ks->chunk) {   /* a row's share would reach past the window: the engine must clamp its steps */
        fprintf(stderr, "[VK] %s chain: a step of %d rows past the split's %d\n", ks->engine, a->S, ks->chunk);
        return 0;
    }
    ks->steps++;
    if (!vkc_kv_cold(ks, a->li, a->pos_base, a->S, 0, a->kv_start, a->sel_row > 0))
        return vkc_kv_sliced(a->S, a->H, vkc_kv_span(ks, 0, a->sel_row), 2 * a->K + a->R, vkc_kv_mla_fin, a);
    size_t part = (size_t)a->S * a->H * (a->K + 2);
    if (ks->cold_dev) {   /* the host's part on the device too */
        const float *host[2] = {a->Lh, a->Rh};
        size_t grp[2] = {0, 0}, ps[2] = {(size_t)a->K, (size_t)a->R};
        int nseg[2] = {1, 1}, sl[2] = {a->K, a->R};
        if (vkc_kv_shadow(ks, a->li, a->R > 0 ? 2 : 1, host, grp, ps, nseg, sl, a->pos_base)) {
            ks->dev_cold++; ks->splits++;   /* a step with a host part, on the device */
            VkcKvsMerge mg = {a->S * a->H, a->H, a->K, 0, a->H * a->K, a->K, a->S * a->H * a->K, 0, a->S * a->H * a->K,
                              a->o_off, a->o_row, a->o_seg, 0, 0, 0, 0};
            return vkc_kv_parts(ks, part) && vkc_kv_mla_dev(a) &&
                   vkc_kv_sliced(a->S, a->H, (long)a->pos_base + a->S, 2 * a->K + a->R, vkc_kv_mla_cold, a) &&
                   vkc_kvs_merge(ks->dpart, ks->cpart, NULL, a->out, &mg);
        }
    }
    a->nqa = (size_t)(a->S - 1) * a->qa_row + (size_t)(a->H - 1) * a->qa_seg + a->K;
    size_t nqr = a->R > 0 ? (size_t)(a->S - 1) * a->qr_row + (size_t)(a->H - 1) * a->qr_seg + a->R : 0;
    if (!vkc_kv_parts(ks, part) ||
        !vkc_kv_run2(ks, a->qa, (size_t)a->qa_off, a->nqa, a->R > 0 ? a->qr : NULL, (size_t)a->qr_off, nqr,
                     a->sel_row > 0 ? a->sel : NULL, (size_t)a->sel_off, (size_t)a->S * a->sel_row,
                     vkc_kv_mla_dev, vkc_kv_mla_host, a, part)) return 0;
    VkcKvsMerge mg = {a->S * a->H, a->H, a->K, 0, a->H * a->K, a->K, a->S * a->H * a->K, 0, a->S * a->H * a->K,
                      a->o_off, a->o_row, a->o_seg, 0, 0, 0, 0};
    if (!vkc_kvs_merge(ks->dpart, ks->cpart, NULL, a->out, &mg)) return 0;
    if (a->sel_row > 0) vkc_kv_count(ks, a->li, (const int *)vkc_ptr(ks->sd), a->S, a->sel_row);
    return 1;
}

/* Inkling's global layers (vkc_kvs_rel's form): q at s*q_row + h*hd of q, r at s*r_row +
 * h*d_rel of r, tau[s] at tau_off + s of tau (floats; the host's copy in tau_h), relp at
 * relp_off of relp (the host's [d_rel][ext] in relp_h); the output at s*o_row + h*hd of
 * out. The host's cache as VkcKvGqa's. */
typedef struct {
    VkcKvSplit *ks; int li;
    VkcBuf *q, *kc, *vc, *r, *tau, *relp, *out;
    int S, H, KVH, hd, pos_base, ext, d_rel, q_row, r_row, relp_off, tau_off, o_row;
    float scale;
    const float *Kh, *Vh; size_t k_grp, k_pos, v_grp, v_pos;
    const float *tau_h, *relp_h;
    size_t nq;                     /* (set by vkc_kv_rel) */
} VkcKvRel;
static int vkc_kv_rel_rec(VkcKvRel *a, int s0, int ns, int fin) {
    VkcKvSplit *ks = a->ks;
    int bt = (int)((size_t)a->li * ks->tab_stride), nb = ks->t[a->li].nblk;
    VkcKvsRel p = {ns, a->H, a->KVH, a->hd, a->pos_base + s0, ks->rows, s0 * a->q_row, a->q_row, a->scale, 0, 0, a->ext, a->d_rel,
                   s0 * a->r_row, a->r_row, a->relp_off, a->tau_off + s0, ks->B, bt, nb, ks->anchor, 0, 0, 0, 0};
    if (fin) { p.o_off = s0 * a->o_row; p.o_row = a->o_row; p.flags = VKC_KVS_FIN; }
    else { p.o_off = s0 * a->H * a->hd; p.o_row = a->H * a->hd; p.st_off = a->S * a->H * a->hd + s0 * a->H * 2; }
    return vkc_kvs_rel(a->q, a->kc, a->vc, fin ? a->out : ks->dpart, a->r, a->tau, a->relp, ks->tab, &p);
}
static int vkc_kv_rel_part(void *u, int s0, int ns) { return vkc_kv_rel_rec((VkcKvRel *)u, s0, ns, 0); }
static int vkc_kv_rel_cold(void *u, int s0, int ns) {
    VkcKvRel *a = (VkcKvRel *)u; VkcKvSplit *ks = a->ks; VkcKvShadow *sh = &ks->sh[a->li];
    int bt = (int)((size_t)a->li * ks->tab_stride), nb = ks->t[a->li].nblk;
    VkcKvsRel p = {ns, a->H, a->KVH, a->hd, a->pos_base + s0, sh->cap, s0 * a->q_row, a->q_row, a->scale, (int)sh->off[0],
                   (int)sh->off[1], a->ext, a->d_rel, s0 * a->r_row, a->r_row, a->relp_off, a->tau_off + s0, ks->B, bt, nb,
                   ks->anchor, s0 * a->H * a->hd, a->H * a->hd, a->S * a->H * a->hd + s0 * a->H * 2, VKC_KVS_COLD};
    return vkc_kvs_rel(a->q, sh->b[0], sh->b[1], ks->cpart, a->r, a->tau, a->relp, ks->tab, &p);
}
static int vkc_kv_rel_fin(void *u, int s0, int ns) { return vkc_kv_rel_rec((VkcKvRel *)u, s0, ns, 1); }
static int vkc_kv_rel_dev(void *u) {
    VkcKvRel *a = (VkcKvRel *)u;
    return vkc_kv_sliced(a->S, a->H, vkc_kv_span(a->ks, 0, 0), 2 * a->hd + a->d_rel, vkc_kv_rel_part, a);
}
static int vkc_kv_rel_host(void *u, const float *q, const int *sel) {
    VkcKvRel *a = (VkcKvRel *)u; VkcKvSplit *ks = a->ks;
    (void)sel;
    if (!vkc_kv_cold_rows(ks, a->li, a->pos_base, a->S, 0, 0, NULL, 0)) return 0;
    VkcKvHostAttn h = {a->S, a->H, a->H / a->KVH, a->hd, 0, a->hd, a->scale, q, (size_t)a->q_row, (size_t)a->hd,
                       NULL, 0, 0, a->Kh, a->k_grp, a->k_pos, NULL, 0, 0, a->Vh, a->v_grp, a->v_pos,
                       a->tau_h, a->d_rel > 0 ? q + a->nq : NULL, (size_t)a->r_row, (size_t)a->d_rel, a->relp_h, a->d_rel,
                       a->ext, a->pos_base, 0};
    vkc_kv_host_attn(ks, &h, ks->hpart, ks->hpart + (size_t)a->S * a->H * a->hd);
    return 1;
}
static int vkc_kv_rel(VkcKvRel *a) {
    VkcKvSplit *ks = a->ks;
    if (a->S > ks->chunk) {   /* a row's share would reach past the window: the engine must clamp its steps */
        fprintf(stderr, "[VK] %s chain: a step of %d rows past the split's %d\n", ks->engine, a->S, ks->chunk);
        return 0;
    }
    ks->steps++;
    if (!vkc_kv_cold(ks, a->li, a->pos_base, a->S, 0, 0, 0))
        return vkc_kv_sliced(a->S, a->H, vkc_kv_span(ks, 0, 0), 2 * a->hd + a->d_rel, vkc_kv_rel_fin, a);
    size_t part = (size_t)a->S * a->H * (a->hd + 2);
    if (ks->cold_dev) {   /* the host's part on the device too */
        const float *host[2] = {a->Kh, a->Vh};
        size_t grp[2] = {a->k_grp, a->v_grp}, ps[2] = {a->k_pos, a->v_pos};
        int nseg[2] = {a->KVH, a->KVH}, sl[2] = {a->hd, a->hd};
        if (vkc_kv_shadow(ks, a->li, 2, host, grp, ps, nseg, sl, a->pos_base)) {
            ks->dev_cold++; ks->splits++;   /* a step with a host part, on the device */
            VkcKvsMerge mg = {a->S * a->H, a->H, a->hd, 0, a->H * a->hd, a->hd, a->S * a->H * a->hd, 0, a->S * a->H * a->hd,
                              0, a->o_row, a->hd, 0, 0, 0, 0};
            return vkc_kv_parts(ks, part) && vkc_kv_rel_dev(a) &&
                   vkc_kv_sliced(a->S, a->H, (long)a->pos_base + a->S, 2 * a->hd + a->d_rel, vkc_kv_rel_cold, a) &&
                   vkc_kvs_merge(ks->dpart, ks->cpart, NULL, a->out, &mg);
        }
    }
    a->nq = (size_t)a->S * a->q_row;
    if (!vkc_kv_parts(ks, part) ||
        !vkc_kv_run2(ks, a->q, 0, a->nq, a->d_rel > 0 ? a->r : NULL, 0, (size_t)a->S * a->r_row, NULL, 0, 0,
                     vkc_kv_rel_dev, vkc_kv_rel_host, a, part)) return 0;
    VkcKvsMerge mg = {a->S * a->H, a->H, a->hd, 0, a->H * a->hd, a->hd, a->S * a->H * a->hd, 0, a->S * a->H * a->hd,
                      0, a->o_row, a->hd, 0, 0, 0, 0};
    return vkc_kvs_merge(ks->dpart, ks->cpart, NULL, a->out, &mg);
}

/* DeepSeek's sparse attention (vkc_kvs_ds's form): q at q_off + s*q_row + h*hd, the list
 * at l_off + s*l_row (cnt entries), the window ring win, the split compressed rows cmp
 * (ks->rows of hd), the sinks at sink_off of prm; the output at o_off + s*o_row + h*hd of
 * out. The host's compressed rows of the layer at Ch + c*hd; n_cmp of them exist. v4:
 * DeepSeek V4's bf16 weights and output. */
typedef struct {
    VkcKvSplit *ks; int li;
    VkcBuf *q, *win, *cmp, *prm, *list, *out;
    int S, H, hd, cnt, l_off, l_row, nwin, w_off, q_off, q_row, sink_off, v4, o_off, o_row, n_cmp;
    float scale;
    const float *Ch;
} VkcKvDs;
/* Sparse scratch grows exactly, without vkc_reserve's geometric over-allocation.
 * Include the existing allocation when checking the driver budget before replacement. */
static int vkc_kv_sparse_reserve(VkcBuf **b, size_t bytes) {
    size_t old = vkc_bytes(*b);
    if (old >= bytes) return 1;
    double used = 0, budget = 0;
    size_t free_bytes;
    if (coli_vk_mem_budget_dev(vkc_device_now(), &used, &budget)) free_bytes = budget > used ? (size_t)((budget - used) * 1e9) : 0;
    else {
        size_t total = coli_vk_device_local_bytes_dev(vkc_device_now()), weights = 0, count = 0;
        VkcStats st; vkc_stats(&st); coli_vk_mem_info_dev(vkc_device_now(), &weights, &count);
        free_bytes = total > weights + st.dev_bytes ? total - weights - st.dev_bytes : 0;
    }
    if (bytes - old > free_bytes) return 0;
    vkc_free(*b); *b = vkc_buf(bytes, VKC_DEV);
    return *b != NULL;
}
/* Sparse attention must retain its original list order, independently of residency.
 * In particular, V4 rounds weights against the maximum of the complete list: partial
 * softmax results rounded against separate maxima cannot be merged equivalently.
 * Download the lists, stage only their missing compressed rows and let the original
 * device arithmetic consume resident and staged rows in that order. Scratch is reused
 * after each query and capped at 32 MiB; a larger requirement declines safely to the
 * engine's CPU fallback instead of allocating another whole KV mirror. The current
 * chunk's new compressed rows stay on the device and are never read from stale RAM. */
static int vkc_kv_ds_stage(VkcKvDs *a) {
    VkcKvSplit *ks = a->ks;
    VkcKvTab *t = &ks->t[a->li];
    size_t nl = (size_t)(a->S - 1) * a->l_row + a->cnt;
    if (!vkc_reserve(&ks->sd, (nl ? nl : 1) * sizeof(int), VKC_DOWN) ||
        !vkc_copy(ks->sd, 0, a->list, (size_t)a->l_off, nl)) return 0;
    double before = vkc_kv_now();
    if (!vkc_submit(1) || !vkc_begin()) return 0;
    ks->wait_ms += vkc_kv_now() - before;
    const int *list = (const int *)vkc_ptr(ks->sd);
    size_t limit = (size_t)32 * 1024 * 1024 / ((size_t)a->hd * sizeof(float));
    if (limit > (size_t)a->cnt) limit = (size_t)a->cnt;
    int *mapped = (int *)malloc((size_t)(a->cnt ? a->cnt : 1) * sizeof(int));
    float *cold = (float *)malloc((limit ? limit : 1) * (size_t)a->hd * sizeof(float));
    int ok = mapped && cold;
    for (int s = 0; ok && s < a->S; s++) {
        double start = vkc_kv_now();
        size_t n = 0;
        for (int j = 0; j < a->cnt; j++) {
            int e = list[(size_t)s * a->l_row + j];
            mapped[j] = e < 0 ? -1 : e; /* all original negative entries mean skipped */
            if (e < a->nwin) continue;
            int r = e - a->nwin, b = r / ks->B;
            if (r >= a->n_cmp || b >= t->nblk) { ok = 0; break; }
            if (t->bt[b] >= 0) continue;
            if (n == limit) { ok = 0; break; }
            memcpy(cold + n * a->hd, a->Ch + (size_t)r * a->hd, (size_t)a->hd * sizeof(float));
            mapped[j] = -2 - (int)n++;
        }
        ks->host_ms += vkc_kv_now() - start;
        if (!ok) break;
        if (!vkc_kv_sparse_reserve(&ks->cold_rows, (n ? n : 1) * (size_t)a->hd * sizeof(float)) ||
            !vkc_kv_sparse_reserve(&ks->cold_list, (size_t)(a->cnt ? a->cnt : 1) * sizeof(int)) ||
            (n && !vkc_write(ks->cold_rows, 0, cold, n * (size_t)a->hd * sizeof(float))) ||
            !vkc_write(ks->cold_list, 0, mapped, (size_t)a->cnt * sizeof(int))) { ok = 0; break; }
        VkcKvsDs p = {1, a->H, a->hd, a->cnt, 0, a->cnt, a->nwin, a->w_off, 0,
                      a->q_off + s * a->q_row, a->q_row, a->sink_off,
                      VKC_KVS_DSFIN | VKC_KVS_DSCOLD | (a->v4 ? VKC_KVS_V4 : 0), a->scale,
                      ks->B, ks->ns, (int)((size_t)a->li * ks->tab_stride), t->nblk,
                      a->o_off + s * a->o_row, a->o_row, 0};
        ok = vkc_kvs_ds_cold(a->q, a->win, a->cmp, ks->cold_rows, a->out,
                             a->prm, ks->cold_list, ks->tab, &p);
        ks->staged_rows += n;
        /* Bound the frame's upload ring as well as the reusable device scratch. */
        if (ok && s + 1 < a->S) ok = vkc_submit(0) && vkc_begin();
    }
    if (ok) { ks->splits++; vkc_kv_count_fixed(ks, a->li, list, a->S, a->l_row, a->cnt, a->nwin); }
    free(mapped); free(cold);
    return ok;
}
static int vkc_kv_ds(VkcKvDs *a) {
    VkcKvSplit *ks = a->ks;
    int bt = (int)((size_t)a->li * ks->tab_stride), nb = ks->t[a->li].nblk;
    ks->steps++;
    if (!vkc_kv_cold_upto(ks, a->li, a->n_cmp)) {
        VkcKvsDs p = {a->S, a->H, a->hd, a->cnt, a->l_off, a->l_row, a->nwin, a->w_off, 0, a->q_off, a->q_row, a->sink_off,
                      VKC_KVS_DSFIN | (a->v4 ? VKC_KVS_V4 : 0), a->scale, ks->B, ks->ns, bt, nb, a->o_off, a->o_row, 0};
        return vkc_kvs_ds(a->q, a->win, a->cmp, a->out, a->prm, a->list, ks->tab, &p);
    }
    return vkc_kv_ds_stage(a);
}

/* ---- the MLA layer op over the split cache (vk_chain.h's vkc_mla_qkv / vkc_mla_attn) --
 * vkc_kv_mla_qkv: vkc_mla_qkv with the step's new rows written into `tmp` (a cache of at
 * least S rows: the latent and rope rows of the step at rows 0..S-1, copied to down for
 * the host as vkc_mla_qkv copies them), then stored into their slots of the split cache
 * c (ks->rows rows). vkc_kv_mla_attn: vkc_mla_attn with the core in two parts
 * (vkc_kv_mla: the device's rows and the host's, Lh / Rh the host's latent and rope rows
 * of the layer, position-major) and the rest as vkc_mla_attn records it. */
static int vkc_kv_mla_qkv(VkcKvSplit *ks, int li, const VkcMla *m, VkcMlaScratch *s, VkcBuf *x, size_t x_off, int S,
                          int pos_base, VkcBuf *cs, VkcMlaCache *c, VkcMlaCache *tmp, VkcBuf *down, size_t down_off) {
    VkcMlaCache t = {tmp->lat, tmp->rope, S};
    if (tmp->cap < S || !vkc_mla_qkv(m, s, x, x_off, S, 0, cs, &t, down, down_off)) return 0;
    VkcKvPart pl = {1, m->K, NULL, 0, c->lat, 0}, pr = {1, m->R, NULL, 0, c->rope, 0};
    return vkc_kv_store(ks, li, &pl, tmp->lat, 0, (size_t)m->K, 0, pos_base, S) &&
           (m->R == 0 || vkc_kv_store(ks, li, &pr, tmp->rope, 0, (size_t)m->R, 0, pos_base, S));
}
static int vkc_kv_mla_attn(VkcKvSplit *ks, int li, const VkcMla *m, VkcMlaScratch *s, int S, int pos_base, int kv_start,
                           VkcMlaCache *c, VkcBuf *sel, size_t sel_off, int sel_row, VkcBuf *gate, size_t gate_off,
                           VkcBuf *out, size_t out_off, const float *Lh, const float *Rh) {
    int H = m->H, QR = m->Q + m->R, HK = H * m->K, HV = H * m->V;
    if (S < 1 || S > s->rows) return 0;
    int ok;
    if (m->kv_b) {   /* qa[s][h][i] = sum_d kv_b[h*(Q+V) + d][i] q[s][h][d], d < Q */
        VkcHgemv a = {1, S, H, m->Q, m->Q + m->V, 0, 0, H * QR, QR, 0, HK, m->K, 0, 0, 0};
        ok = vkc_mla_hgemv(m->kv_b, s->q, s->qabs, NULL, &a);
    } else {         /* qa[s][h][i] = k_abs[h*K + i] . q[s][h][0..Q) */
        VkcHgemv a = {0, S, H, m->K, m->K, 0, 0, H * QR, QR, 0, HK, m->K, 0, 0, 0};
        ok = vkc_mla_hgemv(m->k_abs, s->q, s->qabs, NULL, &a);
    }
    VkcKvMla a = {ks, li, s->qabs, s->q, c->lat, c->rope, sel, s->clat, S, H, m->K, m->R, pos_base, kv_start,
                  0, HK, m->K, m->Q, H * QR, QR, 0, HK, m->K, (int)sel_off, sel ? sel_row : 0, m->scale, Lh, Rh, 0};
    ok = ok && vkc_kv_mla(&a);
    VkcHgemv v = {0, S, H, m->V, m->kv_b ? m->Q + m->V : m->V, m->kv_b ? m->Q : 0, 0, HK, m->K, 0, HV, m->V,
                  (int)gate_off, HV, gate != NULL};
    ok = ok && vkc_mla_hgemv(m->kv_b ? m->kv_b : m->v_abs, s->clat, s->ctx, gate, &v);
    if (ok && m->o && out) ok = vkc_matmul(m->o, s->ctx, 0, out, out_off, S);
    return ok;
}

static void vkc_kv_report(const VkcKvSplit *ks) {
    if (!ks->on || !ks->steps) return;
    fprintf(stderr, "[VK] %s chain: KV split: %llu layer steps, %llu with a host part (%.1f ms waiting for the queries, "
                    "%.1f ms of host work over %llu positions, %llu sparse rows staged), %llu blocks pinned by reads",
            ks->engine, ks->steps, ks->splits, ks->wait_ms, ks->host_ms, ks->host_pos, ks->staged_rows, ks->pins);
    if (ks->cold_dev)
        fprintf(stderr, " | the host's part on the device: %llu layer steps, %llu rows copied to the shadow in %.1f ms "
                        "(%.1f MiB held)", ks->dev_cold, ks->shadow_rows, ks->shadow_ms, ks->shadow_bytes / 1048576.0);
    fprintf(stderr, "\n");
}
#endif
