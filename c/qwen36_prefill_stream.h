/* Fast prompt layers: the trunk and router run in windows of up to 64 rows
 * (128 in a batch of 1024 or more, 256 in one of 1536), and after each window
 * the layer loader starts reading the experts it chose, while the next windows
 * run. The experts then run on the whole batch. Their slots stay borrowed
 * until the layer's GPU work is done (qg_settle). A later batch of a long
 * prompt's layer finds the layer's experts loaded: its trunk runs in one
 * window, the matrix kernels' best case, with one sync instead of 8. The GPU
 * kernels sum each row the same way in any window. Included after qg_trunk
 * and the legacy layer path. */
#ifndef QWEN36_PREFILL_STREAM_H
#define QWEN36_PREFILL_STREAM_H

/* The routed experts the CPU's matrix unit computes (qwen36_cpu_mm.h): those
 * with the most choices, as long as by these estimates the CPU, which also
 * computes the shared expert, finishes before the GPU's share. Measured on the
 * M1 with the 35B model's experts: the CPU takes 8.4 us a choice, 0.67 ms an
 * expert (its weights decoded, its first products) and 0.38 ms more every 512
 * choices, 7.2 us a row for the shared expert, and 10% more while the GPU
 * runs. The GPU's prompt kernels take about 4.4 us a choice in full tiles,
 * more in the many small experts it keeps; its estimate, 4.0 us a choice, is
 * where the prompts ran fastest (against 4.8: 8192 tokens +2.1%, 2048 +2.6%;
 * 4.4 was in between). An expert of fewer than 128
 * choices stays on the GPU: the CPU's unit is slow on few rows. On a real text
 * the router is skewed: in a batch of 2048 rows a few experts often take
 * hundreds of choices each, up to every row. In the first batch of a layer
 * only the experts its first window chose, the first to load: a choice that
 * followed which experts had loaded would follow the disk's timing, and since
 * the CPU and the GPU round differently the results would change from run to
 * run. The chosen experts move to the end of the layer's order, and the GPU
 * takes the ranks below the returned one. */
static int qg_cpu_experts(Model *m, Qwen36Graph *g, int il, int rows, const int *count) {
    int ne = g->stream_ne;
#if QCM_ON
    if (!qg_cpu(g, rows) || rows < 128) return ne;
    int K = m->c.topk, cand[256], nc = 0, nt = 0;
    for (int r = 0; r < ne; r++)
        if (count[g->stream_order[r]] >= 128 && (g->stream_warm || r < g->stream_early)) cand[nc++] = g->stream_order[r];
    for (int i = 1; i < nc; i++) {              /* most choices first */
        int v = cand[i], k = i;
        while (k > 0 && count[cand[k - 1]] < count[v]) { cand[k] = cand[k - 1]; k--; }
        cand[k] = v;
    }
    double gpu = 4.0 * rows * K, cpu = 1.1 * 7.2 * rows;
    unsigned char take[256] = {0};
    for (int i = 0; i < nc && nt < QCM_MAX_EXPERTS; i++) {
        int e = cand[i], n = count[e];
        double us = 1.1 * (670 + 8.4 * n + 380 * ((n - 1) / QCM_EXPERT_ROWS));
        if (cpu + us > gpu - 4.0 * n) continue;
        take[e] = 1;
        nt++;
        cpu += us;
        gpu -= 4.0 * n;
    }
    if (!nt) return ne;
    int order[256], k = 0;
    for (int r = 0; r < ne; r++) if (!take[g->stream_order[r]]) order[k++] = g->stream_order[r];
    for (int r = 0; r < ne; r++) if (take[g->stream_order[r]]) order[k++] = g->stream_order[r];
    memcpy(g->stream_order, order, (size_t)ne * sizeof(int));
    return ne - nt;
#else
    (void)m; (void)il; (void)rows; (void)count;
    return ne;
#endif
}

/* The CPU's part of the MoE (qwen36_cpu_mm.h): the shared expert, and the
 * routed experts of ranks from .. stream_ne, from post as the trunk's last
 * sync left it, while the GPU runs the rest. 0 if the GPU computes it all; if
 * the routed ones cannot be handed over, *from becomes stream_ne. */
static int qg_cpu_moe_start(Model *m, Qwen36Graph *g, QgLayer *L, int il, int rows, int *from, const int *first) {
#if QCM_ON
    QcmShared *j = &g->cpu_shared;
    j->nex = 0;
    if (*from < g->stream_ne) {
        const uint32_t *pairs = q36_tensor_data(g->pairs);
        int ok = 1, E = m->c.n_experts;
        for (int r = *from; ok && r < g->stream_ne; r += 8) {
            int n = g->stream_ne - r < 8 ? g->stream_ne - r : 8, keys[8];
            void *payloads[8];
            for (int k = 0; k < n; k++) keys[k] = il * E + g->stream_order[r + k];
            ok = qll_wait(&g->stream_loader, keys, (unsigned)n, payloads);
            for (int k = 0; ok && k < n; k++) {
                Slot *slot = payloads[k];
                ok = slot && slot->gpu;
                if (ok) j->ex[j->nex++] = (QcmExpert){q36_tensor_data(slot->gpu), pairs + first[r + k],
                                                      first[r + k + 1] - first[r + k]};
            }
        }
        if (!ok) {
            j->nex = 0;
            *from = g->stream_ne;
        }
        j->ey = q36_tensor_data(g->ey);
        j->I = m->c.inter;
        j->K = m->c.topk;
        j->scales = g->slot_scales;
    }
#endif
    return qg_shared_cpu_start(m, g, L, rows);
}

/* The GPU's routed experts of ranks from .. to. The groups whose experts are
 * loaded run in one concurrent region: every gate/up, a barrier, every down
 * (serial dispatches left the GPU idle at each of the 64 ends of a batch of
 * 1024 rows). The region stops at a group still loading, and its work is
 * submitted before that wait. In the later batches of a long prompt the
 * layer's experts are all loaded: one region. Groups start at multiples of
 * topk, as the slots in the pairs do. */
static int qg_stream_groups(Model *m, Qwen36Graph *g, int il, int rows, int from, int to, const int *first) {
    Cfg *c = &m->c;
    int H = c->hidden, K = c->topk, E = c->n_experts, ok = 1;
    static q36_tensor *gpu[QLL_MAX_KEYS][8];   /* a group per topk experts: up to 256 groups */
    for (int r = from; ok && r < to;) {
        int end = r;
        q36_gpu_concurrent(1);
        while (ok && end < to) {
            int n = to-end < K ? to-end : K, keys[8];
            void *payloads[8] = {0};
            q36_tensor **gg = gpu[(end-r)/K];
            for (int j = 0; j < n; j++) keys[j] = il*E + g->stream_order[end+j];
            if (end > r && !qll_ready(&g->stream_loader, keys, (unsigned)n)) break;
            q36_gpu_keepalive(0.25);
            double t = tm_on() ? tm_now() : 0;
            ok = qll_wait(&g->stream_loader, keys, (unsigned)n, payloads);
            if (tm_on()) g->ms[6] += tm_now()-t;
            for (int j = 0; j < 8; j++) gg[j] = NULL;
            for (int j = 0; ok && j < n; j++) {
                Slot *slot = payloads[j];
                ok = (gg[j] = slot ? slot->gpu : NULL) != NULL;
            }
            ok = ok && q36_expert_gate_up(g->eh, g->post, g->pairs, first[end], first[end+n]-first[end],
                                         gg, rows, H, c->inter, K, g->slot_scales);
            end += n;
        }
        q36_gpu_barrier();
        for (int s = r; ok && s < end; s += K) {
            int n = end-s < K ? end-s : K;
            ok = q36_expert_down(g->ey, g->eh, g->pairs, first[s], first[s+n]-first[s],
                                 gpu[(s-r)/K], rows, H, c->inter, K, g->slot_scales);
        }
        q36_gpu_concurrent(0);
        /* Submit useful work before a possible next-group IO wait. No sync is
         * required: all union payloads remain protected through final sum. */
        if (ok && end < to) ok = qg_cut(g, 2) && q36_gpu_flush();
        r = end;
    }
    return ok;
}

/* Build full-batch pair ranges in discovery order. Choice indexes remain in
 * router order within each expert; expert_sum retains the original top-k sum. */
static int qg_stream_moe(Model *m, Qwen36Graph *g, int il, int rows, FILE *lf) {
    Cfg *c = &m->c; QgLayer *L = &g->L[il];
    int H = c->hidden, K = c->topk, E = c->n_experts;
    int count[256] = {0}, rank[256], first[257] = {0}, fill[256];
    const int32_t *ids = q36_tensor_data(g->ids);
    uint32_t *pairs = q36_tensor_data(g->pairs);
    for (int e = 0; e < E; e++) rank[e] = -1;
    for (int p = 0; p < rows*K; p++) {
        if (ids[p] < 0 || ids[p] >= E) return 0;
        count[ids[p]]++; ehit_mark(m, il, ids[p]);
    }
    int from = qg_cpu_experts(m, g, il, rows, count);   /* the CPU's experts last */
    for (int r = 0; r < g->stream_ne; r++) {
        int e = g->stream_order[r];
        if (e < 0 || e >= E || rank[e] >= 0 || !count[e]) return 0;
        rank[e] = r; first[r+1] = first[r] + count[e]; fill[e] = first[r];
    }
    if (first[g->stream_ne] != rows*K) return 0;
    for (int p = 0; p < rows*K; p++) {
        int e = ids[p]; if (rank[e] < 0) return 0;
        pairs[fill[e]++] = (uint32_t)p | (uint32_t)(rank[e]%K) << 28;
    }
    /* The trunk's last sync left post complete: the CPU's matrix unit can
     * compute the shared expert and its routed experts while the GPU runs the
     * others. */
    int cpu = qg_cpu_moe_start(m, g, L, il, rows, &from, first);
    int ok = cpu || (qg_shared_gpu(m, g, L, rows) && qg_cut(g, 3) && q36_gpu_flush());
    if (!cpu) from = g->stream_ne;
    ok = ok && qg_stream_groups(m, g, il, rows, 0, from, first);
    if (cpu) {
        /* The GPU's experts go now, before the wait: the last region of groups
         * was only encoded (qg_cut submits only under COLI_TIMERS), and the
         * GPU sat idle until the CPU's job ended, about 1.1 s of an
         * 8192-token prompt. */
        if (ok) ok = q36_gpu_flush();
        ok = qg_shared_cpu_finish(m, g, L, rows) && ok;
#if QCM_ON
        /* routed experts the CPU could not compute (memory): the GPU does */
        if (!g->cpu_shared.ex_ok) ok = ok && qg_stream_groups(m, g, il, rows, from, g->stream_ne, first);
#endif
    }
    ok = ok && q36_expert_sum(g->x, g->ey, g->weights, g->shared, g->sum, rows, H, K) && qg_cut(g, 2);
    if (ok) {
        /* A long prompt keeps the layer's experts for its next batch. */
        g->stream_pending = !g->stream_more; g->dump_pending = lf; g->dump_x = g->x; g->dump_row = rows-1;
        QgcStats stats = qgc_snapshot(m->global_cache); m->hits = stats.hits; m->miss = stats.misses;
    }
    return ok;
}

static int qg_layer_stream(Model *m, Qwen36Graph *g, int il, int rows, int pos, FILE *lf) {
    Cfg *c = &m->c;
    if (!g->stream_initialized) {
        if (!qll_init(&g->stream_loader, m->global_cache)) return 0;
        g->stream_initialized = 1;
    }
    unsigned char selected[256] = {0};
    g->stream_ne = 0; int ok = 1;
    /* Measured on the 35B model: 128-row windows made a 1024-row batch 2-3% faster,
     * a 512-row one, which waits for its experts more, 5% slower; 256-row windows
     * made a 2048-row batch 6.8% faster than 128 and a 1536-row one 4.4% (the
     * CPU's products and the GPU's kernels do better on more rows, and there are
     * fewer syncs), a 1024-row one 3% slower (its experts start loading later). */
    int window = g->stream_warm ? rows : rows >= 1536 ? 256 : rows >= 1024 ? 128 : 64;
    for (int first = 0; ok && first < rows; first += window) {
        int n = rows-first < window ? rows-first : window;
        int snap = g->snap_rows-first; if (snap < 1 || snap > n) snap = 0;
        QgTrunk t;
        if (!qg_trunk_bind(&t, g, c, c->is_attn[il], first, n, 1)) { ok = 0; break; }
        int encoded = qg_trunk(m, g, &t, il, n, pos+first, snap, 0);
        int terminal = q36_gpu_sync(); /* Also drains partial encoding on error. */
        qg_trunk_release(&t); ok = encoded && terminal;
        qg_settle(m, g, ok); /* First prefix settles the PREVIOUS layer only. */
        if (!ok) break;
        if (!first && !qll_held(&g->stream_loader) && !qll_begin(&g->stream_loader)) { ok = 0; break; }
        int keys[256], nk = 0; unsigned char discovered[256] = {0};
        qg_route_degrade(m, g, il, first, n);
        const int32_t *ids = q36_tensor_data(g->ids);
        for (int p = first*c->topk; p < (first+n)*c->topk; p++) {
            int e = ids[p]; if (e < 0 || e >= c->n_experts) { ok = 0; break; }
            if (!selected[e]) discovered[e] = 1;
        }
        if (!ok) break;
        for (int e = 0; e < c->n_experts; e++) if (discovered[e]) {
            selected[e] = 1; g->stream_order[g->stream_ne++] = e;
            keys[nk++] = il*c->n_experts+e;
        }
        if (nk && !qll_append(&g->stream_loader, keys, (unsigned)nk)) { ok = 0; break; }
        if (!first) g->stream_early = g->stream_ne;
    }
    if (ok) {
        qg_dump_row(lf, g->projected, rows-1, c->hidden);
        qg_dump_row(lf, g->sum, rows-1, c->hidden);
        ok = qg_stream_moe(m, g, il, rows, lf);
    }
    if (!ok) {
        q36_gpu_sync(); qg_settle(m, g, 0);
        if (!qll_finish(&g->stream_loader, 1)) abort(); /* no-op if begin never succeeded */
    }
    return ok;
}

static int qg_layer(Model *m, Qwen36Graph *g, int il, int rows, int pos, FILE *lf) {
    if (g->stream_enabled && (rows > 8 || g->layer_major) && !q36_gpu_is_exact())
        return qg_layer_stream(m, g, il, rows, pos, lf);
    return qg_layer_legacy(m, g, il, rows, pos, lf);
}
#endif
