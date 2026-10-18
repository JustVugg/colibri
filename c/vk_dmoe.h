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
 * warm start makes the usual case on a device that holds the model.
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
#include "vk_chain.h"
#include "vk_tier.h"
#include "backend_vulkan.h"

#define VKC_DMOE_ROWS 8

typedef struct {
    int lo, L, E, K;               /* the chain's layers lo..lo+L-1, E experts each, top K */
    uint8_t *on;                   /* per layer: the tier holds all of its experts here */
    uint32_t *tabh;                /* their entries, as the tier gave them at `epoch` */
    unsigned epoch; int read, dirty, off;
    VkcBuf *tab, *rid, *rwt, *hid, *y;   /* the table, ids [L][rows][K] (down), weights, hidden rows, the routed sum */
    unsigned long long steps;      /* layer steps that ran so */
} VkcDmoe;

static int vkc_dmoe_wanted(void) {
    static int on = -1;
    if (on < 0) { const char *e = getenv("COLI_VK_CHAIN_MOE"); on = !(e && *e == '0'); }
    return on;
}

/* The tier's table for layers lo..lo+L-1, read again when its residents changed since. */
static int vkc_dmoe_read(VkcDmoe *d) {
    if (d->read && d->epoch == vkt_epoch()) return 1;
    int L = d->L, E = d->E;
    if (!d->on) d->on = calloc((size_t)L, 1);
    if (!d->tabh) d->tabh = malloc((size_t)L * E * COLI_VK_XB_ENTRY * sizeof(uint32_t));
    if (!d->on || !d->tabh) { d->off = 1; return 0; }
    for (int i = 0; i < L; i++) {
        int n = vkt_dev_table(d->lo + i, d->tabh + (size_t)i * E * COLI_VK_XB_ENTRY);
        if (n < 0) { d->off = 1; return 0; }
        d->on[i] = n == E;
    }
    d->epoch = vkt_epoch(); d->read = 1; d->dirty = 1;
    return 1;
}
/* For a step of `rows` rows on the primary device: whether any of layers lo..lo+L-1 takes
 * it; reads the tier's table again when its residents changed. 0 for good when the
 * device cannot read the tier's experts by address. */
static int vkc_dmoe_prepare(VkcDmoe *d, int lo, int L, int E, int K, int D, int I, int rows) {
    if (!vkc_dmoe_wanted() || d->off || rows > VKC_DMOE_ROWS || L < 1 || vkc_device_now() != 0 || !vkt_ready() ||
        !vkc_moe_ready() || !vkc_route_ready() || K < 1 || K > VKC_ROUTE_KMAX || E < K || E > VKC_ROUTE_EMAX) return 0;
    if (d->on && (d->lo != lo || d->L != L || d->E != E || d->K != K)) { d->off = 1; return 0; }
    d->lo = lo; d->L = L; d->E = E; d->K = K;
    if (!vkc_dmoe_read(d)) return 0;
    int any = 0;
    for (int i = 0; i < L; i++) any |= d->on[i];
    if (!any) return 0;
    size_t r = VKC_DMOE_ROWS;
    if ((!d->tab && !(d->tab = vkc_buf((size_t)L * E * COLI_VK_XB_ENTRY * sizeof(uint32_t), VKC_DEV))) ||
        !vkc_reserve(&d->rid, (size_t)L * r * K * sizeof(int), VKC_DOWN) || !vkc_reserve(&d->rwt, r * K * sizeof(float), VKC_DEV) ||
        !vkc_reserve(&d->hid, r * K * ((size_t)I + D) * sizeof(float), VKC_DEV) || !vkc_reserve(&d->y, r * (size_t)D * sizeof(float), VKC_DEV)) {
        d->off = 1;
        return 0;
    }
    return 1;
}
/* The table into the device, in a frame just begun: read again first when the tier's
 * residents changed since (a host layer before it ran the tier), written when it changed.
 * 0 only when the write fails; a table that cannot be read turns the layers down. */
static int vkc_dmoe_upload(VkcDmoe *d) {
    if (d->off || !d->tab) return 1;
    if (!vkc_dmoe_read(d)) return 1;   /* d->off: every layer takes the host's step from here */
    if (!d->dirty) return 1;
    if (!vkc_write(d->tab, 0, d->tabh, (size_t)d->L * d->E * COLI_VK_XB_ENTRY * sizeof(uint32_t))) return 0;
    d->dirty = 0;
    return 1;
}
/* Whether layer `layer` runs its experts in the frame: all of them on the device, in the
 * table the frame has (the tier's residents unchanged since it was read). */
static int vkc_dmoe_on(const VkcDmoe *d, int layer) {
    return d->on && !d->off && d->epoch == vkt_epoch() && layer >= d->lo && layer < d->lo + d->L && d->on[layer - d->lo];
}
/* Layer `layer` for n rows: route the logits lg (n rows of E from 0, bias as rt's
 * has_bias says) and run the experts on x (n rows of D from 0) into d->y. rt gives the
 * router's kind, bias, groups, norm, temp, scale and eps; mp the activation (act, limit,
 * a, b), D and I; the rest is filled here. */
static int vkc_dmoe_layer(VkcDmoe *d, int layer, int n, VkcBuf *lg, VkcBuf *bias, VkcBuf *x,
                          const VkcRoute *rt, const VkcMoe *mp) {
    int i = layer - d->lo, io = i * VKC_DMOE_ROWS * d->K;
    VkcRoute r = *rt;
    r.S = n; r.E = d->E; r.K = d->K; r.l_off = 0; r.l_row = d->E; r.i_off = io; r.i_row = d->K; r.w_off = 0; r.w_row = d->K;
    VkcMoe m = *mp;
    m.S = n; m.K = d->K; m.t_off = i * d->E; m.x_off = 0; m.x_row = mp->D; m.i_off = io; m.i_row = d->K;
    m.w_off = 0; m.w_row = d->K; m.y_off = 0; m.y_row = mp->D;
    if (!m.gfmt && d->tabh) m.gfmt = (int)d->tabh[(size_t)i * d->E * COLI_VK_XB_ENTRY + 12];   /* the layer's gate/up format */
    if (!vkc_route(lg, bias, d->rid, d->rwt, &r) || !vkc_moe(x, d->rid, d->rwt, d->tab, d->hid, d->y, &m)) return 0;
    d->steps++;
    return 1;
}
/* Layer `layer`'s ids (n rows of K, rank order, -1 for an empty slot) once its frame is done. */
static const int *vkc_dmoe_ids(const VkcDmoe *d, int layer) {
    return (const int *)vkc_ptr(d->rid) + (size_t)(layer - d->lo) * VKC_DMOE_ROWS * d->K;
}
static void vkc_dmoe_free(VkcDmoe *d) {
    vkc_free(d->tab); vkc_free(d->rid); vkc_free(d->rwt); vkc_free(d->hid); vkc_free(d->y);
    free(d->on); free(d->tabh);
    memset(d, 0, sizeof *d);
}
#endif
