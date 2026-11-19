/* M3: the Qwen3.8 Metal chain's routed experts on the GPU (METAL=1 builds; included by
 * qwen38_metal_chain.h). One expert cache for the whole model (qwen36_global_cache.h),
 * keys layer * experts + e, groups of top-k, every slot one GPU buffer view of `payload`
 * bytes: the expert's gate | up | down rows as the GGUF stores them (q38_gg_expert_tensors,
 * expert e at e * eb in each), read with three preads (gguf_read) by the selected loader's
 * worker (qwen36_selected_loader.h; a group's misses on up to four threads at once).
 *
 * Per layer (q38x_moe), after the sync that follows the router:
 *   the GPU routed (q36_router_softmax + q36_router_select, encoded with the router: the CPU's
 *   rule, see q38x_route); the choices are counting-sorted by expert (pairs = choice | slot << 28,
 *   the slot within its kernel sub-group of 8); the first group's loads start, the shared expert
 *   is encoded and flushed while they read; each group of at most top-k experts runs
 *   q36_expert_gate_up_gq / q36_expert_down_gq in sub-groups of 8 (the layer's types, scale 1:
 *   qwen38's GGUF experts have no ".scale"); a group before the last is synced and released, the
 *   last one stays leased until the next sync (q38x_settle). Then q36_expert_sum(blk, y, weights,
 *   shared, zero): sum_k w_k y_k in the router's order, + gate x shared, + 0 = the CPU's
 *   q38_moe_ex(routed_only) followed by routed + gate x shared (qwen38_metal_chain.h's M2 add).
 *
 * Off (one line, `[qwen38-metal] GPU experts are off: <reason>; the routed experts run on the
 * CPU`, and the M2 path): QWEN38_METAL_EXPERTS=0, a safetensors checkpoint, an expert type
 * other than Q8_0 (8), IQ4_NL (20), IQ4_XS (23), a cache smaller than one group, a slot that
 * does not allocate at the start (Q38M_FAIL_SLOT_ALLOC=1, debug only, fakes that). A layer whose
 * GPU routes are not E valid ids (non-finite logits) runs the M2 path for that step (the CPU's
 * rt_router_pick degrades the same rows). A read that fails mid-run stops the process (exit 3).
 *
 * Size: QWEN38_EXPERT_CACHE_BYTES, else min(all experts' bytes, max(6 GiB, physical RAM - the
 * dense resident bytes - 12 GiB)); slots = bytes / payload, at most keys + group (that many when the
 * bytes hold every expert: no expert is read twice). */
#ifndef QWEN38_METAL_EXPERTS_H
#define QWEN38_METAL_EXPERTS_H
#include <dispatch/dispatch.h>
#include <sys/sysctl.h>
#include "qwen36_global_cache.h"
#include "qwen36_selected_loader.h"

#define Q38X_CHUNK_BYTES ((size_t)256 << 20)   /* slots are views of buffers of up to this size */

static struct {
    int on, E, K, H, I, layers, group;
    size_t payload, nslots;
    const gguf_tensor *(*t)[3];   /* [layer][gate, up, down] */
    uint64_t (*eb)[3];            /* an expert's bytes in each */
    q36_gq_experts *ty;           /* [layer]: the types, scales 1 */
    q36_tensor **slot, **chunk;
    int nchunks;
    QwenGlobalCache cache;
    QwenSelectedLoader loader;
    int cache_on, loader_on, pending;
    q36_tensor *prob, *ids, *wts, *pairs, *eh, *ey, *zero;
    int rows;
    unsigned long long choices, groups, loads, hits, cpu_layers;
    int most;
    double wait_s;
} g_q38x;

static void q38x_off(const char *why) {
    fprintf(stderr, "[qwen38-metal] GPU experts are off: %s; the routed experts run on the CPU\n", why);
}

/* one expert's three parts into a slot: gate | up | down, each at e * eb of its tensor */
static int q38x_read(int key, void *payload) {
    const int l = key / g_q38x.E, e = key % g_q38x.E;
    char *p = (char *)q36_tensor_data((q36_tensor *)payload);
    for (int k = 0; k < 3; k++) {
        const gguf_tensor *t = g_q38x.t[l][k];
        const uint64_t eb = g_q38x.eb[l][k];
        if (gguf_read(q38_gg_ctx_of(t), t, p, (int64_t)e * (int64_t)eb, (int64_t)eb)) {
            fprintf(stderr, "[qwen38-metal] GPU experts: reading %s (expert %d) failed\n", t->name, e);
            return 0;
        }
        p += eb;
    }
    return 1;
}
static void *q38x_allocate(void *ctx, size_t slot) { (void)ctx; return slot < g_q38x.nslots ? g_q38x.slot[slot] : NULL; }
static int q38x_load(void *ctx, int key, void *payload) { (void)ctx; return q38x_read(key, payload); }
static void q38x_release(void *ctx, void *payload) { (void)ctx; (void)payload; }   /* the chunks are freed with the cache */
typedef struct { const int *keys; void *const *payload; unsigned n; int next, ok; } Q38xMany;
static void q38x_load_some(void *ctx, size_t worker) {
    Q38xMany *w = (Q38xMany *)ctx;
    (void)worker;
    for (;;) {
        int i = __atomic_fetch_add(&w->next, 1, __ATOMIC_RELAXED);
        if (i >= (int)w->n) return;
        if (!q38x_read(w->keys[i], w->payload[i])) __atomic_store_n(&w->ok, 0, __ATOMIC_RELAXED);
    }
}
/* a group's misses on up to four threads */
static int q38x_load_many(void *ctx, const int *keys, void *const *payload, unsigned n) {
    (void)ctx;
    Q38xMany w = { keys, payload, n, 0, 1 };
    dispatch_apply_f(n < 4 ? n : 4, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), &w, q38x_load_some);
    return __atomic_load_n(&w.ok, __ATOMIC_RELAXED);
}

static int q38x_type_ok(int type) { return type == 8 || type == 20 || type == 23; }

/* the GPU is done with the last group's slots (after a sync): its lease goes back */
static void q38x_settle(int cancel) {
    if (!g_q38x.pending) return;
    qsl_finish(&g_q38x.loader, cancel);
    g_q38x.pending = 0;
}

static void q38x_free_scratch(void) {
    q36_tensor **t[] = { &g_q38x.prob, &g_q38x.ids, &g_q38x.wts, &g_q38x.pairs, &g_q38x.eh, &g_q38x.ey, &g_q38x.zero };
    for (size_t i = 0; i < sizeof t / sizeof t[0]; i++) { q36_tensor_free(*t[i]); *t[i] = NULL; }
    g_q38x.rows = 0;
}
/* scratch for n rows (the GPU idle) */
static int q38x_scratch(int n) {
    if (!g_q38x.on || n <= g_q38x.rows) return 1;
    q38x_free_scratch();
    const size_t r = (size_t)n, E = g_q38x.E, K = g_q38x.K, H = g_q38x.H, I = g_q38x.I;
    g_q38x.prob = q36_tensor_new(r * E * 4); g_q38x.ids = q36_tensor_new(r * K * 4);
    g_q38x.wts = q36_tensor_new(r * K * 4); g_q38x.pairs = q36_tensor_new(r * K * 4);
    g_q38x.eh = q36_tensor_new(r * K * I * 4); g_q38x.ey = q36_tensor_new(r * K * H * 4);
    g_q38x.zero = q36_tensor_new(r * H * 4);
    if (!g_q38x.prob || !g_q38x.ids || !g_q38x.wts || !g_q38x.pairs || !g_q38x.eh || !g_q38x.ey || !g_q38x.zero) {
        q38x_free_scratch();
        return 0;
    }
    g_q38x.rows = n;
    return 1;
}

static void q38x_free(void) {
    if (g_q38x.loader_on) { q38x_settle(1); qsl_destroy(&g_q38x.loader); }
    if (g_q38x.cache_on) {
        QgcStats s = qgc_snapshot(&g_q38x.cache);
        g_q38x.loads = s.loads; g_q38x.hits = s.hits;
        qgc_destroy(&g_q38x.cache);
    }
    if (g_q38x.slot) for (size_t i = 0; i < g_q38x.nslots; i++) q36_tensor_free(g_q38x.slot[i]);
    for (int i = 0; i < g_q38x.nchunks; i++) q36_tensor_free(g_q38x.chunk[i]);
    free(g_q38x.slot); free(g_q38x.chunk); free(g_q38x.t); free(g_q38x.eb); free(g_q38x.ty);
    q38x_free_scratch();
    g_q38x.slot = g_q38x.chunk = NULL; g_q38x.t = NULL; g_q38x.eb = NULL; g_q38x.ty = NULL;
    g_q38x.nslots = 0; g_q38x.nchunks = 0;
    g_q38x.on = g_q38x.cache_on = g_q38x.loader_on = g_q38x.pending = 0;
}

/* the default size: min(all experts' bytes, max(6 GiB, RAM - dense resident - 12 GiB)) */
static size_t q38x_default_bytes(const Model *m, size_t all) {
    uint64_t ram = 0;
    size_t len = sizeof ram;
    if (sysctlbyname("hw.memsize", &ram, &len, NULL, 0)) ram = 0;
    const uint64_t gib = (uint64_t)1 << 30, dense = m->resident_weight_bytes;
    uint64_t room = ram > dense + 12 * gib ? ram - dense - 12 * gib : 0, b = room > 6 * gib ? room : 6 * gib;
    return b < all ? (size_t)b : all;
}

/* After the chain's start: 1 with the GPU experts on, else 0 with the one line. */
static int q38x_start(Model *m) {
    const Cfg *c = &m->c;
    const int E = c->experts, K = c->topk, H = c->hidden, I = c->inter, L = c->layers;
    char why[192];
    memset(&g_q38x, 0, sizeof g_q38x);
    const char *on = getenv("QWEN38_METAL_EXPERTS");
    if (on && !strcmp(on, "0")) { q38x_off("QWEN38_METAL_EXPERTS=0"); return 0; }
    if (!g_q38_gguf) { q38x_off("the routed experts are not a GGUF's blocks"); return 0; }
    if (E < 1 || E > 512 || K < 1 || K > 16 || K > E || K > QGC_MAX_GROUP || L < 1 || L > INT_MAX / E || I < 1 || H < 1) {
        snprintf(why, sizeof why, "%d experts, top-k %d (the GPU router takes up to 512 and 16)", E, K);
        q38x_off(why); return 0;
    }
    g_q38x.t = (const gguf_tensor *(*)[3])calloc((size_t)L, sizeof *g_q38x.t);
    g_q38x.eb = (uint64_t (*)[3])calloc((size_t)L, sizeof *g_q38x.eb);
    g_q38x.ty = (q36_gq_experts *)calloc((size_t)L, sizeof *g_q38x.ty);
    if (!g_q38x.t || !g_q38x.eb || !g_q38x.ty) { q38x_free(); q38x_off("no memory for the experts' table"); return 0; }
    size_t payload = 0, all = 0;
    for (int l = 0; l < L; l++) {
        q38_gg_expert_tensors(m, l, g_q38x.t[l]);   /* the CPU's own check: it exits on a bad tensor */
        const int ins[3] = { H, H, I }, outs[3] = { I, I, H };
        int ty[3];
        uint64_t sum = 0;
        for (int k = 0; k < 3; k++) {
            ty[k] = g_q38x.t[l][k]->type;
            uint64_t rb = q38x_type_ok(ty[k]) ? q36_gq_row_bytes(ty[k], ins[k]) : 0;
            if (!rb) {
                snprintf(why, sizeof why, "layer %d's routed experts are of GGML type %d (the GPU takes Q8_0, IQ4_NL and IQ4_XS)", l, ty[k]);
                q38x_free(); q38x_off(why); return 0;
            }
            g_q38x.eb[l][k] = rb * (uint64_t)outs[k];
            sum += g_q38x.eb[l][k];
        }
        q36_gq_experts t = { ty[0], ty[1], ty[2], {0}, {0}, {0} };
        for (int j = 0; j < 8; j++) t.sg[j] = t.su[j] = t.sd[j] = 1.f;
        g_q38x.ty[l] = t;
        if (sum > payload) payload = (size_t)sum;
        all += (size_t)sum * (size_t)E;
    }
    payload = (payload + 255) & ~(size_t)255;   /* every view on a 256-byte boundary */
    const size_t keys = (size_t)L * E, group = (size_t)K;
    size_t bytes;
    const char *cb = getenv("QWEN38_EXPERT_CACHE_BYTES");
    if (cb && *cb) {
        char *end = NULL;
        errno = 0;
        unsigned long long v = strtoull(cb, &end, 10);
        if (errno || end == cb || *end || !v || *cb == '-') {
            q38x_free(); q38x_off("QWEN38_EXPERT_CACHE_BYTES is not a positive number of bytes"); return 0;
        }
        bytes = (size_t)v;
    } else bytes = q38x_default_bytes(m, all);
    size_t slots = bytes / payload;
    if (slots < group) {
        snprintf(why, sizeof why, "an expert cache of %zu bytes holds less than one group of %zu experts of %zu bytes", bytes, group, payload);
        q38x_free(); q38x_off(why); return 0;
    }
    if (bytes >= all || slots > keys + group) slots = keys + group;   /* every expert, and a group's window */
    /* every slot now: a refusal turns the GPU experts off here, not mid-run */
    const size_t per = Q38X_CHUNK_BYTES / payload > 0 ? Q38X_CHUNK_BYTES / payload : 1;
    const int nchunks = (int)((slots + per - 1) / per);
    const char *fa = getenv("Q38M_FAIL_SLOT_ALLOC");
    const int fake = fa && !strcmp(fa, "1");
    g_q38x.slot = (q36_tensor **)calloc(slots, sizeof *g_q38x.slot);
    g_q38x.chunk = (q36_tensor **)calloc((size_t)nchunks, sizeof *g_q38x.chunk);
    if (!g_q38x.slot || !g_q38x.chunk) { q38x_free(); q38x_off("no memory for the expert cache's table"); return 0; }
    g_q38x.nslots = slots;
    for (int i = 0; i < nchunks; i++) {
        size_t n = slots - (size_t)i * per < per ? slots - (size_t)i * per : per;
        q36_tensor *ch = fake ? NULL : q36_tensor_new(n * payload);
        if (!ch) {
            if (fake) snprintf(why, sizeof why, "a slot of the GPU expert cache could not be allocated (Q38M_FAIL_SLOT_ALLOC)");
            else snprintf(why, sizeof why, "a slot of the GPU expert cache could not be allocated (%zu slots of %zu bytes)", slots, payload);
            q38x_free(); q38x_off(why); return 0;
        }
        g_q38x.chunk[g_q38x.nchunks++] = ch;
        for (size_t j = 0; j < n; j++) {
            q36_tensor *v = q36_tensor_view(ch, j * payload, payload);
            if (!v) {
                snprintf(why, sizeof why, "a slot of the GPU expert cache could not be allocated (a view)");
                q38x_free(); q38x_off(why); return 0;
            }
            g_q38x.slot[(size_t)i * per + j] = v;
        }
    }
    g_q38x.E = E; g_q38x.K = K; g_q38x.H = H; g_q38x.I = I; g_q38x.layers = L; g_q38x.group = (int)group;
    g_q38x.payload = payload;
    if (!qgc_init(&g_q38x.cache, keys, slots, group, payload, payload, (uint64_t)L * K * 8, NULL,
                  q38x_allocate, q38x_load, q38x_release)) {
        q38x_free(); q38x_off("the expert cache did not start"); return 0;
    }
    g_q38x.cache_on = 1;
    g_q38x.cache.load_many = q38x_load_many;
    /* decode keeps the experts by their expected reuse, as qwen36's (qwen36_expected_policy.h) */
    if (!qgc_expected_enable(&g_q38x.cache, (size_t)L, (size_t)E, (size_t)K)) {
        q38x_free(); q38x_off("no memory for the expert cache's reuse estimates"); return 0;
    }
    if (!qsl_init(&g_q38x.loader, &g_q38x.cache)) { q38x_free(); q38x_off("the expert loader's thread did not start"); return 0; }
    g_q38x.loader_on = 1;
    g_q38x.on = 1;
    /* the CPU's own expert cache serves only a step the chain hands back: one slot a layer */
    for (int l = 0; l < L; l++) {
        LCache *lc = &m->cache[l];
        if (lc->slots && lc->cap > 1) lc->cap = lc->n > 1 ? lc->n : 1;
    }
    fprintf(stderr, "[qwen38-metal] GPU experts: %zu slots of %zu bytes (%.2f GiB%s), groups of %d; the CPU's expert cache: 1 slot a layer\n",
            slots, payload, (double)slots * payload / 1073741824.0, cb && *cb ? ", QWEN38_EXPERT_CACHE_BYTES" : "", K);
    return 1;
}

/* the GPU's routes of n rows from the router logits (lg): softmax over all E, top-k by strict >
 * (ties to the lowest id), weights val / sum of the top (Qwen3.8's norm_topk; without it the
 * chain keeps the M2 path, q38x_start) */
static int q38x_route(q36_tensor *lg, int n) {
    return q36_router_softmax(g_q38x.prob, lg, NULL, n, g_q38x.E) &&
           q36_router_select(g_q38x.ids, g_q38x.wts, g_q38x.prob, n, g_q38x.E, g_q38x.K, 1, 1);
}

/* the routed experts of layer i's n rows on the GPU, after the sync: their sum with the shared
 * expert into blk. 1 done, 0 a kernel refused, -1 the routes are not valid ids (the caller runs
 * the M2 path; nothing encoded). shared() encodes the shared expert (gate x shared in shd). */
static int q38x_moe(Model *m, int i, int n, q36_tensor *x, q36_tensor *shd, q36_tensor *blk,
                    int (*shared)(Model *, int, int)) {
    const int E = g_q38x.E, K = g_q38x.K, H = g_q38x.H, I = g_q38x.I, nc = n * K;
    const int32_t *ids = (const int32_t *)q36_tensor_data(g_q38x.ids);
    uint32_t *pairs = (uint32_t *)q36_tensor_data(g_q38x.pairs);
    q38x_settle(0);   /* the previous layer's last group: the GPU is past it */
    for (int p = 0; p < nc; p++)
        if (ids[p] < 0 || ids[p] >= E) return -1;
    int *first = (int *)calloc((size_t)E + 1, sizeof(int)), *fill = (int *)malloc((size_t)E * sizeof(int));
    int *experts = (int *)malloc((size_t)E * sizeof(int)), ne = 0;
    if (!first || !fill || !experts) { free(first); free(fill); free(experts); return 0; }
    for (int p = 0; p < nc; p++) first[ids[p] + 1]++;
    for (int e = 0; e < E; e++) {
        if (first[e + 1]) experts[ne++] = e;
        first[e + 1] += first[e];
    }
    memcpy(fill, first, (size_t)E * sizeof(int));
    for (int p = 0; p < nc; p++) pairs[fill[ids[p]]++] = (uint32_t)p;
    /* each expert's slot in its kernel sub-group: groups of K experts, sub-groups of 8 in a group */
    for (int j = 0; j < ne; j++)
        for (int q = first[experts[j]]; q < first[experts[j] + 1]; q++) pairs[q] |= (uint32_t)((j % K) % 8) << 28;
    if (ne > g_q38x.most) g_q38x.most = ne;
    g_q38x.choices += (unsigned long long)nc;
    const int G = g_q38x.group;
    int ok = 1;
    for (int g0 = 0; ok && g0 < ne; g0 += G) {
        const int ng = ne - g0 < G ? ne - g0 : G;
        int keys[QGC_MAX_GROUP];
        void *payload[QGC_MAX_GROUP] = { 0 };
        for (int j = 0; j < ng; j++) keys[j] = i * E + experts[g0 + j];
        if (!qsl_start(&g_q38x.loader, keys, (unsigned)ng)) {
            fprintf(stderr, "[qwen38-metal] GPU experts: the loader refused layer %d's group; stopping\n", i);
            exit(3);
        }
        g_q38x.groups++;
        if (g0 == 0) ok = (!shared || shared(m, i, n)) && q36_gpu_flush();   /* while the reads run */
        double t0 = now_s();
        int got = qsl_wait(&g_q38x.loader, NULL, NULL, payload);
        g_q38x.wait_s += now_s() - t0;
        q38_tm_add(m, Q38_TM_EXPERT_READ, t0);
        g_q38x.pending = 1;
        if (!got) {
            q36_gpu_sync();
            q38x_settle(1);
            fprintf(stderr, "[qwen38-metal] GPU experts: layer %d's experts did not reach the cache (a read or a slot failed); stopping\n", i);
            exit(3);
        }
        for (int s0 = 0; ok && s0 < ng; s0 += 8) {
            const int nk = ng - s0 < 8 ? ng - s0 : 8;
            q36_tensor *slots[8] = { 0 };
            for (int j = 0; j < nk; j++) slots[j] = (q36_tensor *)payload[s0 + j];
            const int a = first[experts[g0 + s0]], b = first[experts[g0 + s0 + nk - 1] + 1];
            ok = q36_expert_gate_up_gq(g_q38x.eh, x, g_q38x.pairs, a, b - a, slots, n, H, I, K, &g_q38x.ty[i]) &&
                 q36_expert_down_gq(g_q38x.ey, g_q38x.eh, g_q38x.pairs, a, b - a, slots, n, H, I, K, &g_q38x.ty[i]);
        }
        if (g0 + ng < ne) {   /* the next group may take these slots: the GPU done with them first */
            ok = q36_gpu_sync() && ok;
            q38x_settle(!ok);
        } else ok = ok && q36_gpu_flush();
    }
    free(first); free(fill); free(experts);
    return ok && q36_expert_sum(blk, g_q38x.ey, g_q38x.wts, shd ? shd : g_q38x.zero, g_q38x.zero, n, H, K) ? 1 : 0;
}

#endif
