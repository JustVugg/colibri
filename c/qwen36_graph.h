/* Qwen3.6 on Metal: the forward pass as one sequence of GPU operations.
 *
 * Included by qwen36.c in Metal builds. The model semantics are those of the
 * CPU engine; the kernels are metal/qwen36.metal, reached through
 * qwen36_gpu.h. The tensors stay on the GPU for the whole forward. The CPU
 * writes the token embeddings and a RoPE table, reads the router's choices
 * (the chosen experts have to come from disk), computes the final norm of
 * the last token and reads the logits; in a fast prompt batch its matrix unit
 * also takes part of the products (qwen36_cpu_mm.h). Decode is prefill with
 * one row.
 *
 * One layer:
 *   1. trunk and router, one batch: norm, projections, DeltaNet or attention,
 *      output projection, residual, norm, router logits, top-k;
 *   2. sync, read the choices and sort them by expert; the loader thread
 *      reads the first experts from disk while the GPU computes the shared
 *      expert;
 *   3. the routed experts, by expert: each one is read once per batch and
 *      computes every token that chose it. A last kernel adds them up per
 *      token, with the shared expert and the residual, and writes the next
 *      layer's input in place of x. */
#ifndef QWEN36_GRAPH_H
#define QWEN36_GRAPH_H
#ifdef QWEN36_METAL_GRAPH
#include "qwen36_layer_loader.h"
#include "qwen36_advice_schedule.h"
#include "qwen36_cpu_mm.h"
#include "qwen36_fast_experts.h"

/* A dense matrix: INT8 rows q with a scale per row sc (canonical.qc's, or converted at start), or with gq a
 * GGUF's (qg_dense_gguf): q the file's rows of ggml blocks of `type` times gs, or of floats (type GQ_F32), and
 * sc NULL. */
typedef struct { q36_tensor *q, *sc; int in, out; int gq, type; float gs; } QgDense;

typedef struct {
    q36_tensor *in_ln, *post_ln;
    QgDense q, k, v, o;                         /* attention */
    q36_tensor *qn, *kn, *kc, *vc;              /* q/k norms, KV cache [kv][cap][hd] */
    QgDense qkv, z, out;                        /* DeltaNet */
    q36_tensor *a, *b, *conv, *alog, *dt, *norm;
    q36_tensor *state, *ring;                   /* recurrent state (2 planes), conv history */
    q36_tensor *snap_state, *snap_ring;         /* their copy at the snapshot (qg_snapshot_arm) */
    q36_tensor *spec_state[Q36_SPEC_ROWS - 1], *spec_ring[Q36_SPEC_ROWS - 1];  /* their copy after each of a verify's first rows (qg_spec_alloc) */
    QgDense router, sh_g, sh_u, sh_d;
    q36_tensor *bias, *sh_gate;                 /* bias is zeros when the model has none */
} QgLayer;

struct Qwen36Graph {
    int rows, cap;                              /* rows per batch, KV capacity */
    q36_map *map;                               /* the dense cache file */
    const char *map_base;
    size_t map_bytes;
    q36_map **gmap;                             /* a GGUF's parts (qwen_gguf_map), and the bytes seen in them */
    size_t gmap_bytes;
    QgLayer *L;
    QgDense head;
    /* batch tensors, `rows` rows each */
    q36_tensor *x, *normal, *p0, *p1, *p2, *query, *gate, *key, *core, *normout, *tmp;
    q36_tensor *projected, *sum, *post, *decay, *beta, *conv, *qk, *rot;
    q36_tensor *logits, *prob, *ids, *weights, *sg, *su, *sh, *shared, *scalar;
    q36_tensor *pairs, *eh, *ey;                /* routed experts: choices by expert, h and y per choice */
    q36_tensor *last, *head_out;
    q36_tensor *ahead, *ahead_logits;           /* decode: the next layer's router on this residual */
    q36_tensor *scores;                         /* attention scores, see qg_scores */
    size_t slot_scales;                         /* byte offset of the scales in a slot */
    QwenSelectedLoader loader;
    QasPending ahead_advice;                    /* decode: the guess, hinted once the next trunk is submitted */
    int defer_decode_advice;                    /* set per forward: a one-token continuation */
    /* The loader keeps its lease on the last group of experts until the next
     * sync: the GPU runs batches in order, so once a later batch completes the
     * experts batch has too, and the slots can go back to the cache. The
     * DUMP_LAYERS row of that MoE output is written at the same point. */
    int lease_pending;
    size_t kv_bytes;                            /* a KV element: 4, or 2 where q36_kv_half */
    QwenLayerLoader stream_loader;              /* fast prompt layers, see qwen36_prefill_stream.h */
    int stream_enabled, stream_initialized, stream_pending;
    int stream_order[256], stream_ne;           /* the layer's experts in the order they were chosen */
    int layer_major, stream_more;               /* a long prompt layer by layer; more batches of this layer follow */
    int stream_warm;                            /* a later batch of a layer: its experts are loaded */
    int stream_early;                           /* the experts the batch's first window chose, the first loaded */
    FILE *dump_pending;
    q36_tensor *dump_x;                         /* the tensor that row is in */
    int dump_row;
    double ms[7];                               /* COLI_TIMERS of the current forward, see qg_cut */
    /* Speculative decoding (q36_spec_step in qwen36.c). A verify forward (verify =
     * its rows) runs the picked token and its draft as one batch, every row through
     * the last layer, and returns every row's logits; the DeltaNet kernels copy the
     * state after each of its first rows into spec_state/spec_ring, and a rejected
     * draft swaps in the copy after the last accepted row (qg_spec_rollback). The MTP
     * head (m->mtp) reads the final residual of each verified row with the token
     * after it: a forward keeps its rows' residuals as pending rows, and qg_mtp_draft
     * runs them through the head once the token after the last is known. */
    int verify, no_tail, tail_keep;             /* tail_keep: a prompt's last rows that run the last layer whole (qg_layer_tail) */
    int vadv_layer, vadv_n, vadv_ids[8 * Q36_SPEC_ROWS];    /* a verify's guesses for the next layer (qg_advise_ahead) */
    int guess_layer, guess_n, guess_ids[8 * Q36_SPEC_ROWS];  /* QWEN36_FAST_GUESSED, QWEN36_PREFETCH: the
                                                              * guesses for guess_layer */
    q36_tensor *last_rows, *head_rows;          /* final norms and logits of a verify's rows */
    float *h_rows; size_t h_cap;                /* host: final residuals of the forward's rows */
    QgDense mtp_fc;
    q36_tensor *mtp_in;                         /* [rows][2 hidden]: the normed embedding, then the normed residual */
    float *mtp_pend;                            /* pending rows' residuals, Q36_SPEC_ROWS rows */
    int mtp_pend_n, mtp_pend_pos, mtp_pend_tok[Q36_SPEC_ROWS], mtp_len;
    int mtp_off;                                /* the head's K/V row of position p is row p - mtp_off (qg_mtp_window) */
    int snap_at, snap_rows;                     /* the forward copies the state after snap_at tokens */
    int *snap_ids, snap_len;                    /* the tokens the copy holds */
    int gguf_locked;                            /* qg_gguf_lock: 1 while a GGUF's dense matrices are locked in
                                                 * memory, -1 once mlock failed */
#if QCM_ON
    int cpu_mm;                                 /* qwen36_cpu_mm.h: products of fast prompt batches on the CPU */
    QcmWeights cpu_z, cpu_o;
    QcmShared cpu_shared;
    QcmAttn cpu_attn;
#endif
};

static void qg_settle(Model *, Qwen36Graph *, int);
static void qg_gguf_lock(Model *, Qwen36Graph *, int, int);
static int qg_fast_experts(int topk);
static int qg_fast_guessed(int topk);

static q36_tensor *qg_upload(const float *data, size_t count) {
    return data ? q36_tensor_upload(data, count * sizeof(float)) : NULL;
}

/* Whether [p, p + bytes) lies inside the dense cache file mapping. */
static int qg_mapped(const Qwen36Graph *g, const void *p, size_t bytes) {
    uintptr_t at = (uintptr_t)p, base = (uintptr_t)g->map_base;
    return g->map && at >= base && at - base <= g->map_bytes && bytes <= g->map_bytes - (at - base);
}

/* A GGUF's matrix: its blocks seen in place, each in a window of its own (the experts lie between the dense
 * tensors, and the GPU holds a window resident whole); a matrix of floats as the CPU engine loaded it
 * (load_t_n: F32, F16 and BF16 in F32, times its ".scale"), copied once. */
static int qg_dense_gguf(Qwen36Graph *g, const QW *w, QgDense *out) {
    out->in = w->I;
    out->out = w->O;
    out->gq = 1;
    out->gs = 1.f;
    if (w->w) {
        out->type = GQ_F32;
        return (out->q = q36_tensor_upload(w->w, (size_t)w->I * w->O * sizeof(float))) != NULL;
    }
    size_t bytes = (size_t)q36_gq_row_bytes(w->gq_type, w->I) * w->O;
    for (int p = 0; w->gq && bytes && g->gmap && p < g_gg.n_files; p++) {
        uintptr_t at = (uintptr_t)w->gq, base = (uintptr_t)g_gg_map[p];
        if (at < base || at - base >= g_gg_map_bytes[p]) continue;
        out->type = w->gq_type;
        out->gs = w->gs;
        out->q = q36_map_view_own(g->gmap[p], (size_t)(at - base), bytes);
        if (out->q) g->gmap_bytes += bytes;
        return out->q != NULL;
    }
    return 0;
}

/* A matrix of canonical.qc is seen in place; without the file it is copied
 * once. */
static int qg_dense(Qwen36Graph *g, const QW *w, QgDense *out) {
    if (g_gguf) return qg_dense_gguf(g, w, out);
    if (!w->q || !w->sc) return 0;
    size_t qb = (size_t)w->I * w->O, sb = (size_t)w->O * sizeof(float);
    const char *q = (const char *)w->q, *sc = (const char *)w->sc;
    out->in = w->I;
    out->out = w->O;
    if (qg_mapped(g, q, qb) && qg_mapped(g, sc, sb)) {
        out->q = q36_map_view(g->map, (size_t)(q - g->map_base), qb);
        out->sc = q36_map_view(g->map, (size_t)(sc - g->map_base), sb);
    } else {
        out->q = q36_tensor_upload(q, qb);
        out->sc = q36_tensor_upload(sc, sb);
    }
    return out->q && out->sc;
}

static void qg_dense_free(QgDense *d) {
    q36_tensor_free(d->q);
    q36_tensor_free(d->sc);
    memset(d, 0, sizeof(*d));
}

/* The products with a GGUF's dense matrices and with its routed experts (a dispatch each), for the line at exit
 * (qg_gguf_report): a test sees that the GPU computed them. */
static unsigned long long qg_gguf_products, qg_gguf_experts;

static int qg_mm(q36_tensor *y, q36_tensor *x, const QgDense *w, int rows) {
    if (w->gq) {
        qg_gguf_products++;
        return w->type == GQ_F32 ? q36_dot_f32(y, x, w->q, rows, w->in, w->out)
                                 : q36_dot_gq(y, x, w->q, w->type, w->gs, rows, w->in, w->out);
    }
    return q36_dot_i8(y, x, w->q, w->sc, rows, w->in, w->out);
}

/* The router's logits: INT8 with ordered sums in exact mode (q36_router_dot), a GGUF's as qg_mm computes them
 * (for floats the CPU engine's arithmetic in both modes, q36_dot_f32). The shared expert's products likewise. */
static int qg_router(q36_tensor *y, q36_tensor *x, const QgDense *w, int rows) {
    return w->gq ? qg_mm(y, x, w, rows) : q36_router_dot(y, x, w->q, w->sc, rows, w->in, w->out);
}
static int qg_shared_mm(q36_tensor *y, q36_tensor *x, const QgDense *w, int rows) {
    return w->gq ? qg_mm(y, x, w, rows) : q36_shared_dot(y, x, w->q, w->sc, rows, w->in, w->out);
}

static unsigned qg_gguf_locks, qg_gguf_unlocks[2];   /* qg_gguf_lock: after a decode step, at a request's end */
static void qg_gguf_report(void) {
    fprintf(stderr, "[qwen36-metal] GGUF: %llu products with the file's dense matrices and %llu with its routed "
                    "experts ran on the GPU\n", qg_gguf_products, qg_gguf_experts);
    if (qg_gguf_locks)
        fprintf(stderr, "[qwen36-metal] GGUF: the dense matrices locked for %u prompts, unlocked after %u first decode "
                        "steps and at the end of %u requests; %u still locked at exit\n", qg_gguf_locks,
                qg_gguf_unlocks[0], qg_gguf_unlocks[1], qg_gguf_locks - qg_gguf_unlocks[0] - qg_gguf_unlocks[1]);
}

/* Whether the CPU's matrix unit takes products of this batch (qwen36_cpu_mm.h). */
static int qg_cpu(const Qwen36Graph *g, int rows) {
#if QCM_ON
    return g->cpu_mm && rows >= QCM_MIN_ROWS && !q36_gpu_is_exact();
#else
    (void)g; (void)rows;
    return 0;
#endif
}

/* y = x w on the CPU, x being written before the mark: what was encoded after
 * it goes to the GPU first, and runs meanwhile. Without memory for the float
 * weights the GPU computes it after that. */
static int qg_cpu_mm(Qwen36Graph *g, q36_tensor *y, q36_tensor *x, const QgDense *w, int rows, int mark) {
#if QCM_ON
    const int8_t *q = q36_tensor_data(w->q);
    const float *sc = q36_tensor_data(w->sc);
    if (!q36_gpu_flush()) return 0;
    if (!qcm_convert(&g->cpu_z, 1, &q, &sc, w->out, w->in)) return q36_gpu_wait_mark(mark) && qg_mm(y, x, w, rows);
    if (!q36_gpu_wait_mark(mark)) return 0;
    qcm_mm(q36_tensor_data(y), w->out, q36_tensor_data(x), w->in, g->cpu_z.w, rows, w->in, w->out);
    return 1;
#else
    (void)g;
    return q36_gpu_wait_mark(mark) && qg_mm(y, x, w, rows);
#endif
}

/* The gate parts of q (the second half of each head) on the CPU, x being
 * written before the mark, while the GPU runs what was encoded after it. The
 * converted rows share z's buffer: a layer has one or the other. Without memory
 * for them the GPU computes the gates. */
static int qg_cpu_gate(Qwen36Graph *g, q36_tensor *gate, q36_tensor *x, const QgDense *w, int rows, int hd, int qd,
                       int mark) {
    int heads = w->out / qd, width = heads * hd;
    if (!q36_gpu_flush()) return 0;
#if QCM_ON
    if (qcm_convert_rows(&g->cpu_z, q36_tensor_data(w->q), q36_tensor_data(w->sc), width, w->in, hd, qd, hd)) {
        if (!q36_gpu_wait_mark(mark)) return 0;
        qcm_mm(q36_tensor_data(gate), width, q36_tensor_data(x), w->in, g->cpu_z.w, rows, w->in, width);
        return 1;
    }
#endif
    /* the gate parts are the query parts of heads that start hd rows later */
    q36_tensor *qv = q36_tensor_view(w->q, (size_t)hd * w->in, q36_tensor_bytes(w->q) - (size_t)hd * w->in);
    q36_tensor *sv = q36_tensor_view(w->sc, (size_t)hd * sizeof(float), q36_tensor_bytes(w->sc) - (size_t)hd * sizeof(float));
    int ok = qv && sv && q36_dot_i8_heads(gate, x, qv, sv, rows, w->in, width, hd, qd);
    q36_tensor_free(qv);
    q36_tensor_free(sv);
    return ok;
}

/* The last rows of a long prompt batch whose attention the CPU computes
 * (qwen36_cpu_mm.h), while the GPU computes the others: as many as by these
 * estimates both finish together, the CPU after the gate half of q, the GPU
 * after the query half, k and v. Measured on the M1 with the 35B model (the
 * crossing points at 2048, 4096 and 6144 positions came out at 350, 375 and
 * 400 of 2048 rows): a row's attention takes the GPU about macs / 482 ns per
 * position and the CPU macs / 134 (its softmax included), macs = 2 qh hd; the
 * projections before it hidden (qh + 2 kv) hd / 700 ns a row on the GPU and the
 * gate's hidden qh hd / 318 on the CPU. A CPU row costs about 3.6 GPU rows, and
 * too many rows cost more than too few: rounded down to steps of 32. Batches of
 * 512 rows or more after 1024 positions, on the GPU's eight-heads-per-KV-head
 * path; 0 when it does not pay. */
static int qg_cpu_attn_rows(const Qwen36Graph *g, const Cfg *c, int rows, int pos) {
    if (!qg_cpu(g, rows) || rows < 512 || pos < 1024 || c->q_heads != 8 * c->kv_heads || c->head_dim % 256) return 0;
    double ctx = pos + rows / 2.0, macs = 2.0 * c->q_heads * c->head_dim;
    double gpu = ctx * macs / 482.0, cpu = ctx * macs / 134.0;                    /* ns a row */
    double gpu0 = (double)c->hidden * (c->q_heads + 2 * c->kv_heads) * c->head_dim / 700.0;
    double cpu0 = (double)c->hidden * c->q_heads * c->head_dim / 318.0;
    int r = (int)(rows * (gpu0 + gpu - cpu0) / (cpu + gpu)) / 32 * 32;
    if (r > rows / 2) r = rows / 2 / 32 * 32;
    return r >= 64 ? r : 0;
}

/* A layer's output projection, which the residual right after it reads: in a
 * prompt batch of 1024 rows or more run as one window (a later batch of a
 * long prompt), where the CPU is free after its z or its attention, the GPU
 * computes the first 1408 of every 2048 columns and the CPU's matrix unit the
 * others at the same time (the GPU computes a product about 1.6 times as fast
 * as the CPU while both run; qwen36_cpu_mm.h). In the windows of a first batch
 * the extra wait cost more than it saved. Without memory for the float weights
 * the GPU computes every column. */
static int qg_out(Qwen36Graph *g, q36_tensor *y, q36_tensor *x, const QgDense *w, int rows) {
#if QCM_ON
    int cols = w->out * 11 / 16 / 128 * 128;
    if (qg_cpu(g, rows) && rows >= 1024 && cols >= 128 && cols < w->out) {
        const int8_t *q = (const int8_t *)q36_tensor_data(w->q) + (size_t)cols * w->in;
        const float *sc = (const float *)q36_tensor_data(w->sc) + cols;
        int n = w->out - cols, mark = qcm_convert(&g->cpu_o, 1, &q, &sc, n, w->in) ? q36_gpu_mark() : 0;
        if (mark && q36_dot_i8_cols(y, w->out, x, w->q, w->sc, rows, w->in, cols)) {
            if (!q36_gpu_flush() || !q36_gpu_wait_mark(mark)) return 0;
            qcm_mm((float *)q36_tensor_data(y) + cols, w->out, q36_tensor_data(x), w->in, g->cpu_o.w, rows, w->in, n);
            return 1;
        }
    }
#endif
    return qg_mm(y, x, w, rows);
}

/* Rows of a prompt batch; a longer prompt is cut in batches. Each batch reads
 * nearly every expert of every layer once, so a larger batch reads fewer
 * experts per prompt token; its tensors take about 330 KB per row (340 MB at
 * 1024). With 2048 rows instead of 1024 the routed experts get 64 rows each
 * instead of 32 on average and run 10-15% more of their products per second:
 * an 8192-token prompt of the 35B model ran 4.3% faster, for about 330 MB more
 * while the prompt runs (the results are the same bits). So 2048 while the
 * system reports no memory pressure (kern.memorystatus_vm_pressure_level 1),
 * 1024 under warning or critical pressure. The available memory the system
 * reports is no guide: reading experts fills the file cache, and it went from
 * 72% to 30% within a prompt with no pressure at all. QWEN36_PROMPT_ROWS=1024
 * or 2048 sets it. */
static int qg_prompt_rows(void) {
    const char *v = getenv("QWEN36_PROMPT_ROWS");
    if (v && (!strcmp(v, "1024") || !strcmp(v, "2048"))) return atoi(v);
#ifdef __APPLE__
    int level = 0;
    size_t len = sizeof(level);
    if (!sysctlbyname("kern.memorystatus_vm_pressure_level", &level, &len, NULL, 0) && level == 1) return 2048;
#endif
    return 1024;
}

/* The batch tensors hold the rows of one forward: a prompt's batch (up to
 * qg_prompt_rows, about 330 KB a row) or a decode step's (one row, or eight
 * with the MTP head: see qg_rows). A forward resizes them when its batch is
 * different, so the 340 to 680 MB of a long prompt are not kept through the
 * answer: the system gets them back, and caches experts in them. The GPU must
 * be idle. */
static void qg_free_rows(Qwen36Graph *g) {
    q36_tensor **t[] = {&g->x, &g->normal, &g->p0, &g->p1, &g->p2, &g->query, &g->gate, &g->key,
                        &g->core, &g->normout, &g->tmp, &g->projected, &g->sum, &g->post, &g->decay,
                        &g->beta, &g->conv, &g->qk, &g->rot, &g->logits, &g->prob, &g->ids, &g->weights,
                        &g->sg, &g->su, &g->sh, &g->shared, &g->scalar, &g->pairs, &g->eh, &g->ey, &g->mtp_in};
    for (size_t k = 0; k < sizeof(t) / sizeof(t[0]); k++) { q36_tensor_free(*t[k]); *t[k] = NULL; }
    g->rows = 0;
#if QCM_ON
    qcm_weights_free(&g->cpu_z);
    qcm_weights_free(&g->cpu_o);
    qcm_shared_release(&g->cpu_shared);
    qcm_attn_free(&g->cpu_attn);
#endif
}

/* Attention scores: up to Q36_ATTN_CHUNK rows of every head over the whole
 * cache for a prompt batch, three planes for a decode step, then a float per
 * row and head of a chunk (q36_attn). */
static int qg_scores(Model *m) {
    Qwen36Graph *g = m->graph;
    if (!g->cap) return 1;
    /* at least the three planes of one row: a prompt's last row runs its last layer alone (qg_layer_tail) */
    size_t planes = g->rows > 3 ? (g->rows < Q36_ATTN_CHUNK ? (size_t)g->rows : Q36_ATTN_CHUNK) : 3;
    size_t tail = g->rows > 1 ? planes * (((size_t)g->cap + 31) / 32) : 0;   /* the prompt's maxima */
    if (tail < Q36_ATTN_CHUNK) tail = Q36_ATTN_CHUNK;
    size_t bytes = (planes * (size_t)g->cap + tail) * m->c.q_heads * sizeof(float);
    if (g->scores && q36_tensor_bytes(g->scores) == bytes) return 1;
    q36_tensor *scores = q36_tensor_new(bytes);
    if (!scores) return 0;
    q36_tensor_free(g->scores);
    g->scores = scores;
    return 1;
}

static int qg_rows(Model *m, int R) {
    Qwen36Graph *g = m->graph;
    Cfg *c = &m->c;
    /* decode alternates one-row steps, verifies and MTP rows: a small batch keeps its tensors, and with the
     * MTP head a small batch takes tensors for a verify's rows at once (a verify that reallocated them would
     * cost more) */
    if (R == g->rows || (R < g->rows && g->rows <= Q36_SPEC_ROWS)) return 1;
    if (m->mtp && R < Q36_SPEC_ROWS) R = Q36_SPEC_ROWS;
    if (R == g->rows) return 1;
    if (!q36_gpu_sync()) return 0;
    qg_free_rows(g);
    int H = c->hidden, K = c->topk;
    int aw = c->q_heads * c->head_dim, dw = c->dn_vheads * c->dn_vdim;
    int wide = aw > dw ? aw : dw;
    int p0 = c->q_heads * c->q_head_dim > c->dn_conv_dim ? c->q_heads * c->q_head_dim : c->dn_conv_dim;
    int kvw = c->kv_heads * c->head_dim, p1 = kvw > dw ? kvw : dw;
    size_t F = sizeof(float);
#define QG_NEW(field, count) do { if (!(g->field = q36_tensor_new((size_t)(count) * F))) goto failed; } while (0)
    QG_NEW(x, R * H); QG_NEW(normal, R * H); QG_NEW(projected, R * H); QG_NEW(sum, R * H); QG_NEW(post, R * H);
    QG_NEW(p0, R * p0); QG_NEW(p1, R * p1); QG_NEW(p2, R * kvw); QG_NEW(key, R * kvw);
    QG_NEW(query, R * wide); QG_NEW(gate, R * wide); QG_NEW(core, R * wide); QG_NEW(normout, R * wide);
    QG_NEW(tmp, R * wide);
    QG_NEW(decay, R * c->dn_vheads); QG_NEW(beta, R * c->dn_vheads);
    QG_NEW(conv, R * c->dn_conv_dim); QG_NEW(qk, R * c->dn_conv_dim);
    QG_NEW(rot, R * (c->rotary_dim > 0 ? c->rotary_dim : 2));
    QG_NEW(logits, R * c->n_experts); QG_NEW(prob, R * c->n_experts);
    QG_NEW(ids, R * K); QG_NEW(weights, R * K);
    QG_NEW(sg, R * c->shared_inter); QG_NEW(su, R * c->shared_inter); QG_NEW(sh, R * c->shared_inter);
    QG_NEW(shared, R * H); QG_NEW(scalar, R);
    QG_NEW(pairs, R * K);                       /* uint32, as wide as a float */
    QG_NEW(eh, R * K * c->inter); QG_NEW(ey, R * K * H);
    if (m->mtp) QG_NEW(mtp_in, R * 2 * H);
#undef QG_NEW
    g->rows = R;
    return qg_scores(m);
failed:
    qg_free_rows(g);
    return 0;
}

static void qg_free(Model *m) {
    Qwen36Graph *g = m->graph;
    if (!g) return;
    qas_clear(&g->ahead_advice);
    q36_gpu_sync();
    if (!qll_destroy(&g->stream_loader)) abort();
    qsl_destroy(&g->loader);
    for (int i = 0; g->L && i < m->c.n_layers + m->mtp; i++) {
        QgLayer *L = &g->L[i];
        q36_tensor *t[] = {L->in_ln, L->post_ln, L->qn, L->kn, L->kc, L->vc, L->a, L->b, L->conv,
                           L->alog, L->dt, L->norm, L->state, L->ring, L->bias, L->sh_gate,
                           L->snap_state, L->snap_ring};
        for (size_t k = 0; k < sizeof(t) / sizeof(t[0]); k++) q36_tensor_free(t[k]);
        for (int k = 0; k < Q36_SPEC_ROWS - 1; k++) { q36_tensor_free(L->spec_state[k]); q36_tensor_free(L->spec_ring[k]); }
        QgDense *d[] = {&L->q, &L->k, &L->v, &L->o, &L->qkv, &L->z, &L->out,
                        &L->router, &L->sh_g, &L->sh_u, &L->sh_d};
        for (size_t k = 0; k < sizeof(d) / sizeof(d[0]); k++) qg_dense_free(d[k]);
    }
    free(g->L);
    qg_dense_free(&g->head);
    qg_free_rows(g);
    q36_tensor *t[] = {g->last, g->head_out, g->ahead, g->ahead_logits, g->scores, g->last_rows, g->head_rows};
    for (size_t k = 0; k < sizeof(t) / sizeof(t[0]); k++) q36_tensor_free(t[k]);
    qg_dense_free(&g->mtp_fc);
    free(g->h_rows); free(g->mtp_pend);
    q36_map_free(g->map);
    for (int p = 0; g->gmap && p < g_gg.n_files; p++) q36_map_free(g->gmap[p]);
    free(g->gmap);
    free(g->snap_ids);
#if QCM_ON
    if (g->cpu_shared.queue) dispatch_release(g->cpu_shared.queue);
    if (g->cpu_shared.group) dispatch_release(g->cpu_shared.group);
#endif
    free(g);
    m->graph = NULL;
}

static int qg_create(Model *m) {
    Cfg *c = &m->c;
    int H = c->hidden, K = c->topk;
    if (!m->global_cache || K > 8 || c->n_experts > 256 || c->head_dim > 256 || c->dn_kdim > 128 ||
        H % 64 || c->inter % 64 || c->shared_inter < 1) {
        fprintf(stderr, "[qwen36-metal] geometry outside the Metal graph (top-k<=8, experts<=256, "
                        "head_dim<=256, kdim<=128, hidden/inter multiples of 64, global expert cache)\n");
        return 0;
    }
    Qwen36Graph *g = calloc(1, sizeof(*g));
    if (!g) return 0;
    m->graph = g;
    g->guess_layer = -1;                        /* nothing guessed before the first decode step */
    /* Fast mode keeps the KV cache in halves where its kernels read them. */
    q36_shape kvs = {0};
    kvs.q_heads = c->q_heads; kvs.kv_heads = c->kv_heads; kvs.head_dim = c->head_dim;
    g->kv_bytes = q36_kv_half(&kvs) ? 2 : sizeof(float);
    /* QWEN36_PREFILL_STREAM=0 keeps the eight-expert prompt schedule, to measure. */
    const char *stream = getenv("QWEN36_PREFILL_STREAM");
    if (!stream || strcmp(stream, "0")) {
        g->stream_enabled = qll_cache_eligible(m->global_cache);
        if (!g->stream_enabled)
            fprintf(stderr, "[qwen36-metal] expert cache below %d retained slots: prompt experts load eight at a time\n",
                    QLL_MAX_KEYS);
    }
#if QCM_ON
    /* a GGUF's products are all the GPU's: the CPU's take INT8 weights */
    const char *cpu_mm = getenv("QWEN36_CPU_MATMUL");
    g->cpu_mm = !g_gguf && (!cpu_mm || strcmp(cpu_mm, "0"));
    if (g->cpu_mm) qcm_init();
#endif
    /* The matrices of canonical.qc are seen in place through its mapping, a GGUF's through its parts'. */
    if (m->dense_file.base) {
        g->map_base = (const char *)m->dense_file.base;
        g->map_bytes = m->dense_file.bytes;
        g->map = q36_map_new(g->map_base, g->map_bytes);
        if (!g->map) goto failed;
    }
    if (g_gguf) {
        if (!g_gg_map || !(g->gmap = calloc((size_t)g_gg.n_files, sizeof *g->gmap))) goto failed;
        for (int p = 0; p < g_gg.n_files; p++)
            if (!(g->gmap[p] = q36_map_new(g_gg_map[p], g_gg_map_bytes[p]))) goto failed;
    }
    g->L = calloc((size_t)c->n_layers + m->mtp, sizeof(QgLayer));
    if (!g->L) goto failed;
    size_t cells = (size_t)c->dn_vheads * c->dn_kdim * c->dn_vdim;
    for (int i = 0; i < c->n_layers + m->mtp; i++) {     /* the MTP head's layer last (an attention layer) */
        Layer *l = i < c->n_layers ? &m->L[i] : &m->mtp_L;
        QgLayer *L = &g->L[i];
        L->in_ln = qg_upload(l->in_ln, (size_t)H);
        L->post_ln = qg_upload(l->post_ln, (size_t)H);
        if (!L->in_ln || !L->post_ln) goto failed;
        if (c->is_attn[i]) {
            if (!qg_dense(g, &l->q, &L->q) || !qg_dense(g, &l->k, &L->k) || !qg_dense(g, &l->v, &L->v) ||
                !qg_dense(g, &l->o, &L->o)) goto failed;
            L->qn = qg_upload(l->qn, (size_t)c->head_dim);
            L->kn = qg_upload(l->kn, (size_t)c->head_dim);
        } else {
            if (!qg_dense(g, &l->dn_qkv, &L->qkv) || !qg_dense(g, &l->dn_z, &L->z) ||
                !qg_dense(g, &l->dn_out, &L->out)) goto failed;
            L->a = qg_upload(l->dn_a, (size_t)c->dn_vheads * H);
            L->b = qg_upload(l->dn_b, (size_t)c->dn_vheads * H);
            L->conv = qg_upload(l->dn_conv, (size_t)c->dn_conv_dim * c->dn_convk);
            L->alog = qg_upload(l->dn_alog, (size_t)c->dn_vheads);
            L->dt = qg_upload(l->dn_dtbias, (size_t)c->dn_vheads);
            L->norm = qg_upload(l->dn_norm, (size_t)c->dn_vdim);
            /* The low plane of the compensated state is the exact kernels'
             * only; fast mode keeps the high plane (60 MB less on the 35B). */
            q36_shape dn = {0};
            dn.k_dim = c->dn_kdim; dn.v_dim = c->dn_vdim; dn.conv_dim = c->dn_conv_dim;
            L->state = q36_tensor_new((q36_dn_high_only(&dn) ? 1 : 2) * cells * sizeof(float));
            L->ring = q36_tensor_new((size_t)c->dn_conv_dim * (c->dn_convk - 1) * sizeof(float));
            if (!L->a || !L->b || !L->conv || !L->alog || !L->dt || !L->norm || !L->state || !L->ring) goto failed;
        }
        if (!qg_dense(g, &l->gate, &L->router) || !qg_dense(g, &l->sh_g, &L->sh_g) ||
            !qg_dense(g, &l->sh_u, &L->sh_u) || !qg_dense(g, &l->sh_d, &L->sh_d)) goto failed;
        L->bias = c->has_bias && l->gate_bias ? qg_upload(l->gate_bias, (size_t)c->n_experts)
                                              : q36_tensor_new((size_t)c->n_experts * sizeof(float));
        L->sh_gate = qg_upload(l->sh_gate, (size_t)H);
        if (!L->bias) goto failed;
        /* Without canonical.qc the matrices were copied: the CPU's copies go, so the
         * dense weights are in memory once (a graph that fails stops the start, and
         * the graph never runs the CPU engine's forward). A GGUF's blocks stay in the
         * file's mapping (qw_free leaves them), its floats go. */
        if (!g->map) {
            QW *w[] = {&l->q, &l->k, &l->v, &l->o, &l->dn_qkv, &l->dn_z, &l->dn_out,
                       &l->gate, &l->sh_g, &l->sh_u, &l->sh_d};
            for (size_t k = 0; k < sizeof(w) / sizeof(w[0]); k++) qw_free(w[k]);
        }
    }
    if (!m->final_norm || !qg_dense(g, &m->lm_head, &g->head)) goto failed;
    if (m->mtp) {
        if (!qg_dense(g, &m->mtp_fc, &g->mtp_fc) || !(g->mtp_pend = malloc((size_t)Q36_SPEC_ROWS * H * sizeof(float))))
            goto failed;
        qw_free(&m->mtp_fc);
        /* the head runs on the GPU only: its CPU copies go */
        Layer *l = &m->mtp_L;
        QW *w[] = {&l->q, &l->k, &l->v, &l->o, &l->gate, &l->sh_g, &l->sh_u, &l->sh_d};
        for (size_t k = 0; k < sizeof(w) / sizeof(w[0]); k++) qw_free(w[k]);
    }
    qw_free(&m->lm_head);                       /* the GPU sees it; a CPU copy is not needed */

    if (!(g->last = q36_tensor_new((size_t)H * sizeof(float))) ||
        !(g->head_out = q36_tensor_new((size_t)c->vocab * sizeof(float))) ||
        !(g->ahead = q36_tensor_new((size_t)Q36_SPEC_ROWS * H * sizeof(float))) ||
        !(g->ahead_logits = q36_tensor_new((size_t)Q36_SPEC_ROWS * c->n_experts * sizeof(float)))) goto failed;
    g->slot_scales = (size_t)3 * H * c->inter / 2;
    if (!qsl_init(&g->loader, m->global_cache)) goto failed;
    fprintf(stderr, "[qwen36-metal] graph ready on %s: rows<=%d dense=%s kv=%s (%zu bytes mapped); the experts in the "
                    "global cache, not in the per-layer one\n",
            q36_gpu_device_name(), qg_prompt_rows(), g->map || g_gguf ? "mapped" : "copied",
            g->kv_bytes == 2 ? "half" : "float", g_gguf ? g->gmap_bytes : g->map_bytes);
    if (g_gguf) {
        fprintf(stderr, "[qwen36-metal] GGUF: the dense matrices read in place as the file's ggml blocks, and the "
                        "routed experts as its blocks too (layer 0: gate %s, up %s, down %s), with no second "
                        "quantization\n", gq_type_name(m->gqx[0].hg), gq_type_name(m->gqx[0].hu),
                gq_type_name(m->gqx[0].hd));
        atexit(qg_gguf_report);
    }
    qg_fast_experts(c->topk);                   /* its line, if QWEN36_FAST_EXPERTS is set */
    qg_fast_guessed(c->topk);                   /* and QWEN36_FAST_GUESSED's */
    return 1;
failed:
    fprintf(stderr, "[qwen36-metal] cannot build the Metal graph (weights or GPU memory)\n");
    qg_free(m);
    return 0;
}

/* KV capacity: 256, 512 or 1024 rows, then multiples of 1024, so a context of
 * 8192 tokens holds 9216 rows instead of 16384 (40 MB per 1024 rows on the 35B
 * model, 20 MB with the halves of fast mode). A request reserves its prompt and
 * answer at once (ensure_kv); the rows already written move along. */
static int qg_reserve(Model *m, int need) {
    Qwen36Graph *g = m->graph;
    Cfg *c = &m->c;
    if (need <= g->cap) return 1;
    int cap = g->cap > 256 ? g->cap : 256;
    while (cap < need && cap < 1024) cap *= 2;
    if (cap < need) cap = (need + 1023) / 1024 * 1024;
    if (!q36_gpu_sync()) return 0;
    size_t head = (size_t)c->head_dim * g->kv_bytes;
    for (int i = 0; i < c->n_layers + m->mtp; i++) {
        if (!c->is_attn[i]) continue;
        QgLayer *L = &g->L[i];
        q36_tensor *kc = q36_tensor_new((size_t)c->kv_heads * cap * head);
        q36_tensor *vc = q36_tensor_new((size_t)c->kv_heads * cap * head);
        if (!kc || !vc) { q36_tensor_free(kc); q36_tensor_free(vc); return 0; }
        for (int h = 0; L->kc && h < c->kv_heads; h++) {
            memcpy((char *)q36_tensor_data(kc) + (size_t)h * cap * head,
                   (char *)q36_tensor_data(L->kc) + (size_t)h * g->cap * head, (size_t)g->cap * head);
            memcpy((char *)q36_tensor_data(vc) + (size_t)h * cap * head,
                   (char *)q36_tensor_data(L->vc) + (size_t)h * g->cap * head, (size_t)g->cap * head);
        }
        q36_tensor_free(L->kc);
        q36_tensor_free(L->vc);
        L->kc = kc;
        L->vc = vc;
    }
    g->cap = cap;
    return qg_scores(m);
}

/* A new sequence: zero the recurrent state. KV rows are simply overwritten. */
static void qg_reset(Model *m) {
    Qwen36Graph *g = m->graph;
    qas_clear(&g->ahead_advice);
    g->defer_decode_advice = 0;
    qg_settle(m, g, q36_gpu_sync());
    if (qll_held(&g->stream_loader) && !qll_finish(&g->stream_loader, 1)) abort();
    g->stream_pending = 0;
    g->mtp_len = g->mtp_pend_n = g->mtp_off = 0;
    for (int i = 0; i < m->c.n_layers; i++) {
        QgLayer *L = &g->L[i];
        if (L->state) memset(q36_tensor_data(L->state), 0, q36_tensor_bytes(L->state));
        if (L->ring) memset(q36_tensor_data(L->ring), 0, q36_tensor_bytes(L->ring));
    }
}

/* cos/sin per row and rotary pair. Frequency and angle are computed in
 * double and only the result is rounded to float: float phases are already
 * off by a visible amount below 2048 tokens. */
static void qg_rope_table(const Cfg *c, Qwen36Graph *g, int rows, int pos) {
    float *t = q36_tensor_data(g->rot);
    for (int d = 0; d < c->rotary_dim / 2; d++) {
        double f = pow((double)c->theta, -2.0 * (double)d / (double)c->rotary_dim);
        for (int r = 0; r < rows; r++) {
            float *p = t + (size_t)r * c->rotary_dim + 2 * d;
            double angle = (double)(pos + r) * f;
            p[0] = (float)cos(angle);
            p[1] = (float)sin(angle);
        }
    }
}

static void qg_dump_row(FILE *lf, q36_tensor *t, int row, int width) {
    if (lf) fwrite((const float *)q36_tensor_data(t) + (size_t)row * width, sizeof(float), (size_t)width, lf);
}

/* After a sync: write the deferred MoE dump row, release the deferred lease
 * and the copies lent to the layer before (QWEN36_PREFETCH). */
static void qg_settle(Model *m, Qwen36Graph *g, int ok) {
    if (g->dump_pending && ok) qg_dump_row(g->dump_pending, g->dump_x, g->dump_row, m->c.hidden);
    g->dump_pending = NULL;
    if (!ok) qas_clear(&g->ahead_advice);
    if (g->lease_pending) qsl_finish(&g->loader, !ok);
    g->lease_pending = 0;
    if (g->stream_pending && !qll_finish(&g->stream_loader, !ok)) abort();
    g->stream_pending = 0;
    qwen_prefetch_settle();
}

/* A copy of the sequence state after the first tokens of a prompt, so a later
 * prompt that begins with the same tokens does not read them again (serve
 * mode, qwen36.c). The forward takes it on its way: in the batch that holds
 * that token the DeltaNet kernels also write the state and conv history of
 * their layer as they are after it (q36_shape.snap). The KV rows before it
 * are not copied: nothing written at later positions reaches them. */
static int qg_snapshot_arm(Model *m, int len) {
    Qwen36Graph *g = m->graph;
    qas_clear(&g->ahead_advice);
    g->snap_len = 0;
    for (int i = 0; i < m->c.n_layers; i++) {
        QgLayer *L = &g->L[i];
        if (!L->state) continue;
        if (!L->snap_state) L->snap_state = q36_tensor_new(q36_tensor_bytes(L->state));
        if (!L->snap_ring) L->snap_ring = q36_tensor_new(q36_tensor_bytes(L->ring));
        if (!L->snap_state || !L->snap_ring) return 0;
    }
    g->snap_at = len;
    return 1;
}

/* After the forward: the copy holds the first snap_at of these tokens. */
static void qg_snapshot_taken(Model *m, const int *ids) {
    Qwen36Graph *g = m->graph;
    int len = g->snap_at, *copy = realloc(g->snap_ids, (size_t)len * sizeof(int));
    g->snap_at = 0;
    if (!copy) return;
    g->snap_ids = copy;
    memcpy(copy, ids, (size_t)len * sizeof(int));
    g->snap_len = len;
}

/* A sequence that starts from the copy: its state back in place, and kv_len
 * where the copy ends. */
static void qg_snapshot_load(Model *m) {
    Qwen36Graph *g = m->graph;
    qas_clear(&g->ahead_advice);
    qg_settle(m, g, q36_gpu_sync());
    for (int i = 0; i < m->c.n_layers; i++) {
        QgLayer *L = &g->L[i];
        if (!L->state) continue;
        memcpy(q36_tensor_data(L->state), q36_tensor_data(L->snap_state), q36_tensor_bytes(L->state));
        memcpy(q36_tensor_data(L->ring), q36_tensor_data(L->snap_ring), q36_tensor_bytes(L->ring));
    }
    m->kv_len = g->snap_len;
}

/* COLI_TIMERS=1: what was encoded so far becomes a batch of its own, and the
 * GPU's own report of its time goes to g->ms[idx]: the phases of qwen36.c's
 * timers (0 DeltaNet, 1 attention, 2 routed experts, 3 shared expert,
 * 4 router, 5 lm_head); 6 is the wall time spent waiting for the disk.
 * Nothing waits for the GPU, so a timed run keeps its pace. */
static int qg_cut(Qwen36Graph *g, int idx) {
    return !tm_on() || q36_gpu_flush_timed(&g->ms[idx]);
}

/* Experts [from, to) of the sorted list that are not in the cache are
 * announced to the kernel, so the SSD reads them while the current group is
 * loaded and computed. Advisory only. */
static void qg_advise(Model *m, int il, const int *experts, int from, int to) {
    for (int i = from; i < to; i++)
        if (!qgc_contains(m->global_cache, il * m->c.n_experts + experts[i])) qwen_expert_advise(m, il, experts[i]);
}

/* Only an already encoded batch can be submitted. No GPU command is
 * encoded between this flush and the existing sync, so its flush is empty. */
static int qg_advice_submit(void *context) {
    (void)context;
    return q36_gpu_flush();
}
static void qg_advice_emit(void *context, int layer, const int *ids, int from, int to) {
    qg_advise(context, layer, ids, from, to);
}

/* A speculative verify's rows read many experts together already: each row
 * announces its four likeliest guesses, against a step's six. */
#define QG_AHEAD_GUESSES_VERIFY 4
#define QG_MTP_EXPERTS 2               /* routed experts the MTP head reads, see qg_moe */
#define QG_MTP_WINDOW 32               /* a prompt's last positions the MTP head reads, see qg_mtp_window */
#define QG_MTP_VOCAB 32768             /* token ids the MTP head drafts from, see qg_mtp_draft */

/* Fast mode, decode: QWEN36_FAST_EXPERTS=J reads a token's J likeliest experts
 * and uses the rest of its top-k only when the expert cache holds them
 * (qwen36_fast_experts.h), with one line at start. Unset or 0, the default, and
 * exact mode use all of them; a value that is not a whole number from 1 to
 * top-k - 1 is refused with a line. */
static int qg_fast_experts(int topk) {
    static int j = -1;
    if (j < 0) {
        const char *v = getenv("QWEN36_FAST_EXPERTS");
        char *end = NULL;
        long n = v && *v ? strtol(v, &end, 10) : 0;
        j = 0;
        if (v && *v && (*end || n < 0 || n >= topk))
            fprintf(stderr, "[qwen36-metal] QWEN36_FAST_EXPERTS=%s ignored: J is a whole number from 1 to %d\n", v,
                    topk - 1);
        else if (n > 0 && q36_gpu_is_exact())
            fprintf(stderr, "[qwen36-metal] QWEN36_FAST_EXPERTS=%ld ignored: exact mode uses every chosen expert\n", n);
        else if (n > 0) {
            j = (int)n;
            fprintf(stderr, "[qwen36-metal] QWEN36_FAST_EXPERTS=%d: in decode a token uses its %d likeliest routed "
                            "experts, the rest of its top-%d only when the expert cache holds them\n", j, j, topk);
        }
    }
    return j;
}

/* QWEN36_FAST_GUESSED's counts for its line at exit: the rows the rule ran
 * for (a token at a layer) and the middle-rank choices that stayed only
 * because the layer before guessed them. */
static unsigned long long qg_guessed_rows, qg_guessed_kept;
static void qg_guessed_report(void) {
    fprintf(stderr, "[qwen36-metal] QWEN36_FAST_GUESSED: the rule ran %llu times (a token at a layer); %llu "
                    "middle-rank choices stayed because the layer before guessed them\n", qg_guessed_rows,
            qg_guessed_kept);
}

/* Fast experts, decode: QWEN36_FAST_GUESSED=R uses a token's R - 1 likeliest
 * experts, its next ones up to the J likeliest only when the expert cache
 * holds them or they were guessed for its layer (qg_advise_ahead: their reads
 * were announced while the layer before computed, so keeping them costs
 * little), and the rest of its top-k as fast experts do (qfe_keep_guessed),
 * with one line at start and one at exit (qg_guessed_report). Unset or 0, the
 * default, leaves fast experts as they are; exact mode, the variable without
 * QWEN36_FAST_EXPERTS of 2 or more, and a value that is not a whole number
 * from 2 to J are refused with a line. */
static int qg_fast_guessed(int topk) {
    static int r = -1;
    if (r < 0) {
        const char *v = getenv("QWEN36_FAST_GUESSED");
        char *end = NULL;
        long n = v && *v ? strtol(v, &end, 10) : 0;
        int J = qg_fast_experts(topk);
        r = 0;
        if (!v || !*v || (!*end && n == 0))
            ;
        else if (q36_gpu_is_exact())
            fprintf(stderr, "[qwen36-metal] QWEN36_FAST_GUESSED=%s ignored: exact mode uses every chosen expert\n", v);
        else if (J < 2)
            fprintf(stderr, "[qwen36-metal] QWEN36_FAST_GUESSED=%s ignored: it needs QWEN36_FAST_EXPERTS of 2 or more\n",
                    v);
        else if (*end || n < 2 || n > J)
            fprintf(stderr, "[qwen36-metal] QWEN36_FAST_GUESSED=%s ignored: R is a whole number from 2 to %d\n", v, J);
        else {
            r = (int)n;
            fprintf(stderr, "[qwen36-metal] QWEN36_FAST_GUESSED=%d: in decode a token uses its %d likeliest routed "
                            "experts, the next ones up to its %d likeliest only when the expert cache holds "
                            "them or the layer before guessed them\n", r, r - 1, J);
            atexit(qg_guessed_report);
        }
    }
    return r;
}

/* Decode: the next layer's router run on this layer's residual (qg_layer).
 * The residual changes little from one layer to the next, so its likeliest
 * experts are mostly the ones the next layer will choose. Of the six
 * likeliest (QG_AHEAD_GUESSES, qwen36_advice_schedule.h), those not in the
 * cache are announced once the next layer's trunk is submitted, and the SSD
 * reads them while the GPU computes it. A wrong guess costs a read, never a
 * result; on the 35B model the seventh and eighth guesses were chosen 56% and
 * 41% of the time on code (65% and 49% on prose), too seldom for their reads
 * to pay on an 8 GB M1 (docs/experiments/qwen36-ahead-six-2026-10-06). */
static void qg_advise_ahead(Model *m, Qwen36Graph *g, int il, int row) {
    const float *l = (const float *)q36_tensor_data(g->ahead_logits) + (size_t)row * m->c.n_experts;
    int K = g->verify ? QG_AHEAD_GUESSES_VERIFY : qas_ahead_guesses(m->c.topk), top[8], n;
    if (K > m->c.topk) K = m->c.topk;
    /* with fast experts the others are used only when the cache holds them: nothing to announce */
    if (qg_fast_experts(m->c.topk) && K > qg_fast_experts(m->c.topk)) K = qg_fast_experts(m->c.topk);
    n = qas_top(l, m->c.n_experts, K, top);
    if (qg_fast_guessed(m->c.topk) || qpf_on) {   /* the next layer's guesses, every row's, once each */
        if (g->guess_layer != il + 1) { g->guess_layer = il + 1; g->guess_n = 0; }
        for (int j = 0; j < n; j++) {
            int k = 0;
            while (k < g->guess_n && g->guess_ids[k] != top[j]) k++;
            if (k == g->guess_n && k < (int)(sizeof g->guess_ids / sizeof *g->guess_ids))
                g->guess_ids[g->guess_n++] = top[j];
        }
    }
    if (g->defer_decode_advice && !g->verify && n <= 8) {
        /* The next trunk overwrites ahead_logits. Save only the selected IDs
         * and emit them after that trunk and the current experts are submitted. */
        qas_save(&g->ahead_advice, il + 1, top, n);
        return;
    }
    if (g->verify || g->defer_decode_advice) {
        /* A verify's rows: the union of their guesses waits for the next trunk too (qg_layer_legacy). Said at
         * once, their reads would hold up the verify's own, the GPU idle meanwhile. */
        if (g->vadv_layer != il + 1) { g->vadv_layer = il + 1; g->vadv_n = 0; }
        for (int j = 0; j < n; j++) {
            int k = 0;
            while (k < g->vadv_n && g->vadv_ids[k] != top[j]) k++;
            if (k == g->vadv_n && k < (int)(sizeof g->vadv_ids / sizeof *g->vadv_ids)) g->vadv_ids[g->vadv_n++] = top[j];
        }
        return;
    }
    qg_advise(m, il + 1, top, 0, n);
}

/* The shared expert of the batch on the GPU, into g->shared. */
static int qg_shared_gpu(Model *m, Qwen36Graph *g, QgLayer *L, int rows) {
    int H = m->c.hidden, SI = m->c.shared_inter;
    return qg_shared_mm(g->sg, g->post, &L->sh_g, rows) && qg_shared_mm(g->su, g->post, &L->sh_u, rows) &&
           q36_silu_mul(g->sh, g->sg, g->su, rows * SI) && qg_shared_mm(g->shared, g->sh, &L->sh_d, rows) &&
           q36_shared_gate(g->scalar, g->post, L->sh_gate, rows, H) &&
           q36_scale_rows(g->shared, g->scalar, rows, H);
}

/* The shared expert on the CPU's matrix unit (qwen36_cpu_mm.h), from post as
 * the last sync left it, while the GPU runs the routed experts: 0 if the GPU
 * computes it. */
static int qg_shared_cpu_start(Model *m, Qwen36Graph *g, QgLayer *L, int rows) {
#if QCM_ON
    if (!qg_cpu(g, rows)) return 0;
    QcmShared *j = &g->cpu_shared;
    const QgDense *d[3] = {&L->sh_g, &L->sh_u, &L->sh_d};
    for (int k = 0; k < 3; k++) {
        j->q[k] = q36_tensor_data(d[k]->q);
        j->sc[k] = q36_tensor_data(d[k]->sc);
    }
    j->post = q36_tensor_data(g->post);
    j->shared = q36_tensor_data(g->shared);
    j->gate = L->sh_gate ? q36_tensor_data(L->sh_gate) : NULL;
    j->rows = rows;
    j->H = m->c.hidden;
    j->SI = m->c.shared_inter;
    return qcm_shared_start(j);
#else
    (void)m; (void)g; (void)L; (void)rows;
    return 0;
#endif
}

/* Waits for the CPU's shared expert; when it could not run (memory), the GPU
 * computes it. */
static int qg_shared_cpu_finish(Model *m, Qwen36Graph *g, QgLayer *L, int rows) {
#if QCM_ON
    if (qcm_shared_finish(&g->cpu_shared)) return 1;
#endif
    return qg_shared_gpu(m, g, L, rows);
}

/* A row the router could not fill (a NaN in its input, a corrupt tile) gets
 * id -1 from the selection kernel. As in the CPU engine (moe in qwen36.c),
 * such a choice becomes the slot's own expert at weight 0, the row's other
 * weights are scaled to sum 1, and one line says so; rows the router filled
 * are not touched. */
static void qg_route_degrade(Model *m, Qwen36Graph *g, int il, int first, int rows) {
    int K = m->c.topk, E = m->c.n_experts;
    int32_t *ids = (int32_t *)q36_tensor_data(g->ids) + (size_t)first * K;
    float *w = (float *)q36_tensor_data(g->weights) + (size_t)first * K;
    static int warned;
    for (int r = 0; r < rows; r++, ids += K, w += K) {
        int bad = 0;
        float sum = 0;
        for (int k = 0; k < K; k++)
            if (ids[k] < 0 || ids[k] >= E) bad = 1;
            else sum += w[k];
        if (!bad) continue;
        if (!warned) {
            warned = 1;
            fprintf(stderr, "[router] non-finite logits at layer %d, or fewer than top-k "
                            "experts eligible: selection degraded\n", il);
        }
        for (int k = 0; k < K; k++)
            if (ids[k] < 0 || ids[k] >= E) ids[k] = k, w[k] = 0;
            else if (sum > 0) w[k] /= sum;
    }
}

/* The routed experts of slots gpu[0..8) for pairs [first, first + n): gate and up into eh, or down (down = 1)
 * into ey. A GGUF's in its layer's types, with the ".scale" factors of each slot's expert, eids[j] for slot j
 * (-1: none). */
static int qg_experts(Model *m, Qwen36Graph *g, int il, int down, int first, int n, q36_tensor *const gpu[8],
                      const int eids[8], int rows) {
    Cfg *c = &m->c;
    int H = c->hidden, I = c->inter, K = c->topk;
    if (!g_gguf)
        return down ? q36_expert_down(g->ey, g->eh, g->pairs, first, n, gpu, rows, H, I, K, g->slot_scales)
                    : q36_expert_gate_up(g->eh, g->post, g->pairs, first, n, gpu, rows, H, I, K, g->slot_scales);
    const GqExp *x = &m->gqx[il];
    q36_gq_experts t = {x->hg, x->hu, x->hd, {0}, {0}, {0}};
    for (int j = 0; j < 8; j++) {
        int e = eids[j];
        t.sg[j] = x->sg && e >= 0 ? x->sg[e] : 1.f;
        t.su[j] = x->su && e >= 0 ? x->su[e] : 1.f;
        t.sd[j] = x->sd && e >= 0 ? x->sd[e] : 1.f;
    }
    qg_gguf_experts++;
    return down ? q36_expert_down_gq(g->ey, g->eh, g->pairs, first, n, gpu, rows, H, I, K, &t)
                : q36_expert_gate_up_gq(g->eh, g->post, g->pairs, first, n, gpu, rows, H, I, K, &t);
}

/* The routed experts, by expert. The batch's choices are sorted by expert;
 * the distinct experts are read in groups of top-k (one cache lease each, the
 * first while the GPU computes the shared expert) and each group computes all
 * the choices routed to it. expert_sum then adds them up per token in the
 * router's order, so the grouping does not change a bit of the result. */
/* A verify's experts, all in one lease (qg_moe): those the cache holds are computed while the misses read, as a
 * step's are. The hits' choices go first in pairs, then the misses', each part in sub-groups of top-k slots
 * whose choices' slot fields count within the sub-group. Each choice is computed alone and expert_sum adds them
 * in the router's order: the same bits in any order. */
static int qg_moe_verify(Model *m, Qwen36Graph *g, int il, int rows, int ne, const int *experts, const int *first,
                         uint32_t *pairs) {
    Cfg *c = &m->c;
    int K = c->topk, n = rows * K, ord[QGC_MAX_GROUP], at[QGC_MAX_GROUP + 1], o = 0, ok = 1;
    uint32_t moved[8 * Q36_SPEC_ROWS];
    Slot *hit[QGC_MAX_GROUP] = {0}, *slots[QGC_MAX_GROUP] = {0};
    q36_tensor *gpu[8] = {0};
    if (n > (int)(sizeof moved / sizeof *moved) || ne > QGC_MAX_GROUP) return 0;
    int held = (int)qsl_wait_hits(&g->loader, (void **)hit);
    for (int j = 0; j < ne; j++) if (hit[j]) ord[o++] = j;
    for (int j = 0; j < ne; j++) if (!hit[j]) ord[o++] = j;
    o = 0;
    for (int i = 0; i < ne; i++) {
        int e = experts[ord[i]], sub = (i < held ? i : i - held) % K;
        at[i] = o;
        for (int q = first[e]; q < first[e + 1]; q++) moved[o++] = (pairs[q] & 0x0fffffffu) | (uint32_t)sub << 28;
    }
    at[ne] = o;
    memcpy(pairs, moved, (size_t)o * sizeof(uint32_t));
    for (int part = 0; ok && part < 2; part++) {
        int from = part ? held : 0, to = part ? ne : held;
        if (part) {
            if (held) ok = q36_gpu_flush();
            double t = tm_on() ? tm_now() : 0;
            ok = qsl_wait(&g->loader, NULL, NULL, (void **)slots) && ok;
            if (tm_on()) g->ms[6] += tm_now() - t;
            if (ok && il + 1 < c->n_layers)
                for (int r = 0; r < rows; r++) qg_advise_ahead(m, g, il, r);
        }
        for (int s0 = from; ok && s0 < to; s0 += K) {
            int nk = to - s0 < K ? to - s0 : K;
            int eids[8];
            for (int j = 0; j < 8; j++) eids[j] = j < nk ? experts[ord[s0 + j]] : -1;
            for (int j = 0; j < K; j++) {
                Slot *s = j >= nk ? NULL : part ? slots[ord[s0 + j]] : hit[ord[s0 + j]];
                gpu[j] = s ? s->gpu : NULL;
                if (j < nk && !gpu[j]) ok = 0;
            }
            ok = ok && qg_experts(m, g, il, 0, at[s0], at[s0 + nk] - at[s0], gpu, eids, rows) &&
                 qg_experts(m, g, il, 1, at[s0], at[s0 + nk] - at[s0], gpu, eids, rows);
        }
    }
    return ok;
}

static int qg_moe(Model *m, Qwen36Graph *g, int il, int rows, FILE *lf) {
    Cfg *c = &m->c;
    QgLayer *L = &g->L[il];
    int H = c->hidden, K = c->topk, E = c->n_experts, n = rows * K;
    qg_route_degrade(m, g, il, 0, rows);   /* first: qfe_keep indexes by the ids */
    if (E <= 256 && K <= 8) {
        int32_t *iw = q36_tensor_data(g->ids);
        float *w = q36_tensor_data(g->weights);
        unsigned char present[256];
        if (il == c->n_layers) {
            /* The MTP head reads its 2 heaviest routed experts only (QG_MTP_EXPERTS): with all 8 its reads take
             * the cache's slots from the model's layers. A draft only, in both modes; its other six choices point
             * at the heaviest expert with weight 0, whose FFN runs once more for each (qfe_keep). */
            memset(present, 0, (size_t)E);
            for (int r = 0; r < rows; r++) qfe_keep(iw + (size_t)r * K, w + (size_t)r * K, K, QG_MTP_EXPERTS, present);
        } else if (qg_fast_experts(K) && ((g->defer_decode_advice && rows == 1) || (g->verify && rows == g->verify))) {
            /* Fast experts (qwen36_fast_experts.h). A verify's later rows also count what its earlier rows chose,
             * which the verify reads anyway, and with J above 3 its draft rows keep two sure experts fewer (3 with
             * J = 5), except in the last quarter of the layers. With QWEN36_FAST_GUESSED the first row keeps its
             * middle ranks only when the cache holds them or they were guessed for this layer; the first layer of a
             * step, which nothing guessed, keeps the rule above. */
            int J = qg_fast_experts(K), late = il >= c->n_layers - c->n_layers / 4;
            int R = g->guess_layer == il ? qg_fast_guessed(K) : 0;
            unsigned char cheap[256];
            qgc_present(m->global_cache, il * E, E, present);
            if (R) {
                memcpy(cheap, present, (size_t)E);
                for (int q = 0; q < g->guess_n; q++) cheap[g->guess_ids[q]] = 1;
            }
            for (int r = 0; r < rows; r++) {
                if (!r && R) {
                    int kept = 0;
                    qfe_keep_guessed(iw + (size_t)r * K, w + (size_t)r * K, K, J, R - 1, present, cheap, &kept);
                    qg_guessed_rows++;
                    qg_guessed_kept += (unsigned long long)kept;
                } else qfe_keep(iw + (size_t)r * K, w + (size_t)r * K, K, r && !late && J > 3 ? J - 2 : J, present);
                for (int k = 0; k < K; k++) present[iw[(size_t)r * K + k]] = 1;
            }
        }
    }
    const int32_t *ids = q36_tensor_data(g->ids);
    uint32_t *pairs = q36_tensor_data(g->pairs);
    int first[257] = {0}, fill[256], experts[256], ne = 0, keys[QGC_MAX_GROUP];
    for (int p = 0; p < n; p++) {
        if (ids[p] < 0 || ids[p] >= E) return 0;
        first[ids[p] + 1]++;
        if (il < c->n_layers) ehit_mark(m, il, ids[p]);
    }
    for (int e = 0; e < E; e++) {
        if (first[e + 1]) experts[ne++] = e;
        first[e + 1] += first[e];
    }
    memcpy(fill, first, (size_t)E * sizeof(int));
    for (int p = 0; p < n; p++) pairs[fill[ids[p]]++] = (uint32_t)p;
    for (int i = 0; i < ne; i++)                /* the slot of each expert in its group */
        for (int q = first[experts[i]]; q < first[experts[i] + 1]; q++) pairs[q] |= (uint32_t)(i % K) << 28;
    /* A small batch (a speculative verify) leases every expert of the layer at once when the cache takes
     * groups that large (qwen_global_cache_init): the reads go together and the GPU computes them in
     * sub-groups of top-k slots (each choice's slot field counts within its sub-group) with no wait between. */
    int whole = rows > 1 && rows <= Q36_SPEC_ROWS && ne > K && ne <= (int)m->global_cache->group_cap;
    int ng = whole ? ne : ne < K ? ne : K;
    for (int j = 0; j < ng; j++) keys[j] = il * E + experts[j];
    /* QWEN36_PREFETCH: a decode step borrows the copies of its experts done before their loads start */
    q36_tensor *lent[8] = {0};
    if (qpf_on && rows == 1 && !g->verify && g->defer_decode_advice && il < c->n_layers && ng == ne)
        for (int j = 0; j < ng; j++) lent[j] = qwen_prefetch_lend(keys[j]);
    if (!qsl_start(&g->loader, keys, (unsigned)ng)) return 0;
    qg_advise(m, il, experts, ng, ng + K < ne ? ng + K : ne);
    /* The shared expert runs on the GPU while the worker reads from disk. */
    int ok = qg_shared_gpu(m, g, L, rows) && qg_cut(g, 3) && q36_gpu_flush();
    if (rows == g->verify && ng == ne) {
        if (rows > 1) q36_gpu_keepalive(0.25);
        ok = ok && qg_moe_verify(m, g, il, rows, ne, experts, first, pairs);
    } else for (int g0 = 0;;) {
        /* Fast prompt kernels also leave gaps while the next experts load. */
        if (rows > 1) q36_gpu_keepalive(0.25);
        int a = first[experts[g0]], done = 0;
        Slot *slots[QGC_MAX_GROUP] = {0};
        q36_tensor *gpu[8] = {0};
        /* Decode: the experts the cache already holds, and those whose copy
         * was lent to the group, are computed while the misses read, in a GPU
         * batch of their own; the misses' pairs follow theirs. An expert's
         * pairs stay together: with fast experts a left-out choice is another
         * pair of the heaviest expert. Each choice is computed alone and
         * expert_sum adds them in the router's order: the same bits in any
         * order. */
        if (ok && rows == 1 && ng == ne) {
            Slot *hit[8] = {0};
            unsigned held = qsl_wait_hits(&g->loader, (void **)hit), copied = 0;
            for (int j = 0; j < ng; j++) copied += !hit[j] && lent[j];
            if (held + copied > 0 && held + copied < (unsigned)ng) {
                uint32_t order[8];
                int o = 0;
                for (int j = 0; j < ng; j++) gpu[j] = hit[j] ? hit[j]->gpu : lent[j];
                for (int j = 0; j < ng; j++)
                    if (gpu[j]) for (int q = first[experts[j]]; q < first[experts[j] + 1]; q++) order[o++] = pairs[q];
                done = o;
                for (int j = 0; j < ng; j++)
                    if (!gpu[j]) for (int q = first[experts[j]]; q < first[experts[j] + 1]; q++) order[o++] = pairs[q];
                memcpy(pairs + a, order, (size_t)o * sizeof(uint32_t));
                qpf_early += copied;
                int eids[8];
                for (int j = 0; j < 8; j++) eids[j] = j < ng ? experts[j] : -1;
                ok = qg_experts(m, g, il, 0, a, done, gpu, eids, rows) &&
                     qg_experts(m, g, il, 1, a, done, gpu, eids, rows) && q36_gpu_flush();
            }
        }
        double t = tm_on() ? tm_now() : 0;
        ok = qsl_wait(&g->loader, NULL, NULL, (void **)slots) && ok;
        if (tm_on()) g->ms[6] += tm_now() - t;
        if (ok && (rows == 1 || rows == g->verify) && il + 1 < c->n_layers)
            for (int r = 0; r < rows; r++) qg_advise_ahead(m, g, il, r);
        /* QWEN36_PREFETCH: a decode step's guesses start copying now, while this layer's experts compute */
        if (ok && rows == 1 && !g->verify && g->defer_decode_advice && il + 1 < c->n_layers && g->guess_layer == il + 1)
            qwen_prefetch_issue(m, il + 1, g->guess_ids, g->guess_n);
        for (int s0 = 0; ok && s0 < ng; s0 += K) {      /* sub-groups of top-k slots (one when ng <= K) */
            int nk = ng - s0 < K ? ng - s0 : K;
            int a1 = s0 ? first[experts[g0 + s0]] : a + done, b1 = first[experts[g0 + s0 + nk - 1] + 1], eids[8];
            for (int j = 0; ok && j < K; j++) ok = j >= nk || (gpu[j] = slots[s0 + j] ? slots[s0 + j]->gpu : NULL) != NULL;
            for (int j = 0; j < 8; j++) eids[j] = j < nk ? experts[g0 + s0 + j] : -1;
            ok = ok && qg_experts(m, g, il, 0, a1, b1 - a1, gpu, eids, rows) &&
                 qg_experts(m, g, il, 1, a1, b1 - a1, gpu, eids, rows);
        }
        if (!ok || (g0 += ng) == ne) break;
        ok = qg_cut(g, 2);
        ok = q36_gpu_sync() && ok;            /* the GPU is done with the slots before release, even after a failure */
        qsl_finish(&g->loader, !ok);
        if (!ok) return 0;
        ng = ne - g0 < K ? ne - g0 : K;
        for (int j = 0; j < ng; j++) keys[j] = il * E + experts[g0 + j];
        if (!qsl_start(&g->loader, keys, (unsigned)ng)) return 0;
        qg_advise(m, il, experts, g0 + ng, g0 + ng + K < ne ? g0 + ng + K : ne);
    }
    ok = ok && q36_expert_sum(g->x, g->ey, g->weights, g->shared, g->sum, rows, H, K) && qg_cut(g, 2);
    /* The last lease and the dump row are settled at the next sync. */
    g->lease_pending = 1;
    g->dump_pending = ok ? lf : NULL;
    g->dump_x = g->x;
    g->dump_row = rows - 1;
    QgcStats cache = qgc_snapshot(m->global_cache);
    m->hits = cache.hits;
    m->miss = cache.misses;
    return ok;
}

/* Non-owning base pointers or owned small tensor-view descriptors. Never copy
 * Qwen36Graph, whose loader mutexes and lease identities are not copyable. */
typedef struct {
    q36_tensor *x, *normal, *p0, *p1, *p2, *query, *gate, *key, *core, *normout, *tmp, *projected, *sum, *post, *decay, *beta, *conv, *qk, *rot, *logits, *prob, *ids, *weights;
    int owns_views;
} QgTrunk;
static void qg_trunk_release(QgTrunk *t) {
    if (t->owns_views) {
        q36_tensor_free(t->x);
        q36_tensor_free(t->normal);
        q36_tensor_free(t->p0);
        q36_tensor_free(t->p1);
        q36_tensor_free(t->p2);
        q36_tensor_free(t->query);
        q36_tensor_free(t->gate);
        q36_tensor_free(t->key);
        q36_tensor_free(t->core);
        q36_tensor_free(t->normout);
        q36_tensor_free(t->tmp);
        q36_tensor_free(t->projected);
        q36_tensor_free(t->sum);
        q36_tensor_free(t->post);
        q36_tensor_free(t->decay);
        q36_tensor_free(t->beta);
        q36_tensor_free(t->conv);
        q36_tensor_free(t->qk);
        q36_tensor_free(t->rot);
        q36_tensor_free(t->logits);
        q36_tensor_free(t->prob);
        q36_tensor_free(t->ids);
        q36_tensor_free(t->weights);
    }
    memset(t, 0, sizeof(*t));
}
static int qg_trunk_bind(QgTrunk *t, Qwen36Graph *g, const Cfg *c,
                         int attn, int first, int rows, int views) {
    memset(t, 0, sizeof(*t)); t->owns_views = views;
    int aw = c->q_heads*c->head_dim, dw = c->dn_vheads*c->dn_vdim;
    int width = attn ? aw : dw, kv = c->kv_heads*c->head_dim;
#define QG_VIEW(field, stride) do { \
    size_t sw = (size_t)(stride); \
    t->field = views ? q36_tensor_view(g->field, (size_t)first*sw*4, (size_t)rows*sw*4) : g->field; \
    if (!t->field) { qg_trunk_release(t); return 0; } \
} while (0)
    QG_VIEW(x, c->hidden);
    QG_VIEW(normal, c->hidden);
    QG_VIEW(p0, attn ? c->q_heads*c->q_head_dim : c->dn_conv_dim);
    QG_VIEW(p1, attn ? kv : dw);
    QG_VIEW(p2, kv);
    QG_VIEW(query, aw);
    QG_VIEW(gate, aw);
    QG_VIEW(key, kv);
    QG_VIEW(core, width);
    QG_VIEW(normout, dw);
    QG_VIEW(tmp, width);
    QG_VIEW(projected, c->hidden);
    QG_VIEW(sum, c->hidden);
    QG_VIEW(post, c->hidden);
    QG_VIEW(decay, c->dn_vheads);
    QG_VIEW(beta, c->dn_vheads);
    QG_VIEW(conv, c->dn_conv_dim);
    QG_VIEW(qk, c->dn_conv_dim);
    QG_VIEW(rot, c->rotary_dim > 0 ? c->rotary_dim : 2);
    QG_VIEW(logits, c->n_experts);
    QG_VIEW(prob, c->n_experts);
    QG_VIEW(ids, c->topk);
    QG_VIEW(weights, c->topk);
#undef QG_VIEW
    return 1;
}
/* The attention of rows first .. rows on the CPU, the queries and the cache
 * written before the mark. Without memory the GPU computes them too. */
static int qg_cpu_attention(Qwen36Graph *g, QgLayer *L, QgTrunk *t, const q36_shape *s, int first, int mark) {
    if (!q36_gpu_flush() || !q36_gpu_wait_mark(mark)) return 0;
#if QCM_ON
    if (qcm_attention(&g->cpu_attn, q36_tensor_data(t->core), q36_tensor_data(t->query), q36_tensor_data(L->kc),
                      q36_tensor_data(L->vc), g->kv_bytes == 2, first, s->rows, s->pos, s->q_heads, s->kv_heads,
                      s->head_dim, s->cap)) return 1;
#endif
    size_t row = (size_t)s->q_heads * s->head_dim * sizeof(float);
    q36_tensor *qv = q36_tensor_view(t->query, first * row, (size_t)(s->rows - first) * row);
    q36_tensor *ov = q36_tensor_view(t->core, first * row, (size_t)(s->rows - first) * row);
    q36_shape r = *s;
    r.rows = s->rows - first;
    r.pos = s->pos + first;
    int ok = qv && ov && q36_attn(ov, qv, L->kc, L->vc, g->scores, &r);
    q36_tensor_free(qv);
    q36_tensor_free(ov);
    return ok;
}

/* Whether a verify runs its DeltaNet recurrence row by row (qg_dn_rows): with more than one draft, and in exact
 * mode with one too (the exact batch kernel keeps a column of both state planes per thread, slower for two rows
 * than two one-row kernels). */
static int qg_verify_rows(const Qwen36Graph *g) {
    return g->verify > 2 || (g->verify == 2 && q36_gpu_is_exact());
}

/* A verify of more than one draft: the DeltaNet convolution, norms and recurrence row after row, each from the
 * state the row before left, the state and ring after each of the first rows copied (snap) into its spec copy, so
 * that a rejected draft can swap in the state after the last accepted row (qg_spec_rollback). Every row runs a
 * decode step's one-row kernels, so it has a step's bits. */
static int qg_dn_rows(Model *m, QgLayer *L, QgTrunk *t, const q36_shape *s) {
    Cfg *c = &m->c;
    size_t cd = (size_t)c->dn_conv_dim * sizeof(float), vh = (size_t)c->dn_vheads * sizeof(float);
    size_t core = (size_t)c->dn_vheads * c->dn_vdim * sizeof(float);
    int ok = 1;
    for (int r = 0; ok && r < s->rows; r++) {
        q36_shape s1 = *s;
        s1.rows = 1; s1.pos = s->pos + r; s1.snap = r + 1 < s->rows;
        q36_tensor *p0 = q36_tensor_view(t->p0, r * cd, cd), *cv = q36_tensor_view(t->conv, r * cd, cd);
        q36_tensor *qk = q36_tensor_view(t->qk, r * cd, cd), *dc = q36_tensor_view(t->decay, r * vh, vh);
        q36_tensor *bt = q36_tensor_view(t->beta, r * vh, vh), *co = q36_tensor_view(t->core, r * core, core);
        q36_tensor *ss = s1.snap ? L->spec_state[r] : NULL, *sr = s1.snap ? L->spec_ring[r] : NULL;
        ok = p0 && cv && qk && dc && bt && co && (!s1.snap || (ss && sr)) &&
             q36_dn_conv(cv, L->ring, p0, L->conv, sr, &s1) && q36_dn_l2(qk, cv, &s1) &&
             q36_dn_delta(co, L->state, qk, cv, dc, bt, ss, &s1);
        /* encoded commands keep the buffers; this frees the views' descriptors only */
        q36_tensor_free(p0); q36_tensor_free(cv); q36_tensor_free(qk);
        q36_tensor_free(dc); q36_tensor_free(bt); q36_tensor_free(co);
    }
    return ok;
}

static int qg_trunk(Model *m, Qwen36Graph *g, QgTrunk *t, int il, int rows, int pos, int snap, int decode_ahead) {
    Cfg *c = &m->c;
    QgLayer *L = &g->L[il];
    int H = c->hidden;
    q36_shape s = {rows, H, c->q_heads, c->kv_heads, c->head_dim, c->q_head_dim, c->rotary_dim,
                   c->dn_vheads, c->dn_kheads, c->dn_kdim, c->dn_vdim, c->dn_convk, c->dn_conv_dim, pos, g->cap,
                   snap, g_dn_tiled, g_ssm_a_direct};
    int ok = q36_rmsnorm(t->normal, t->x, L->in_ln, rows, H, c->eps, 0), width;
    const QgDense *wo;
    if (c->is_attn[il]) {
        width = c->q_heads * c->head_dim;
        /* The gate half of q is read only after the attention: the CPU computes it while the GPU
         * computes the query half, k, v and the attention, each straight into its tensor. */
        int mark = 0;
        int cpu = qg_cpu(g, rows) && c->q_head_dim == 2 * c->head_dim && c->head_dim % 32 == 0 &&
                  (mark = q36_gpu_mark()) &&
                  q36_dot_i8_heads(t->query, t->normal, L->q.q, L->q.sc, rows, H, width, c->head_dim, c->q_head_dim);
        /* the query half goes to the GPU now: after the mark it had only the short input norm */
        if (cpu) ok = ok && q36_gpu_flush() && qg_mm(t->key, t->normal, &L->k, rows) && qg_mm(t->p2, t->normal, &L->v, rows);
        else
            ok = ok && qg_mm(t->p0, t->normal, &L->q, rows) && qg_mm(t->p1, t->normal, &L->k, rows) &&
                 qg_mm(t->p2, t->normal, &L->v, rows) && q36_attn_split(t->query, t->gate, t->key, t->p0, t->p1, &s);
        if (ok && L->qn) ok = q36_rmsnorm(t->query, t->query, L->qn, rows * c->q_heads, c->head_dim, c->eps, 0);
        if (ok && L->kn) ok = q36_rmsnorm(t->key, t->key, L->kn, rows * c->kv_heads, c->head_dim, c->eps, 0);
        if (ok && c->rotary_dim)
            ok = q36_attn_rope(t->query, t->rot, c->q_heads, &s) && q36_attn_rope(t->key, t->rot, c->kv_heads, &s);
        ok = ok && q36_attn_kv(L->kc, L->vc, t->key, t->p2, &s);
        /* In a long prompt the CPU also takes the attention of the last rows. */
        int arows = cpu ? qg_cpu_attn_rows(g, c, rows, pos) : 0, amark = arows && ok ? q36_gpu_mark() : 0;
        q36_shape sa = s;
        if (amark) sa.rows = rows - arows;
        ok = ok && q36_attn(t->core, t->query, L->kc, L->vc, g->scores, &sa) &&
             (!cpu || qg_cpu_gate(g, t->gate, t->normal, &L->q, rows, c->head_dim, c->q_head_dim, mark)) &&
             (!amark || qg_cpu_attention(g, L, t, &s, rows - arows, amark)) &&
             q36_gate(t->tmp, t->core, t->gate, rows * width, 0);
        wo = &L->o;
    } else {
        width = c->dn_vheads * c->dn_vdim;
        /* z is read only by the gate: the CPU can compute it during the recurrence */
        int cpu = qg_cpu(g, rows), mark = cpu ? q36_gpu_mark() : 0;
        /* qkv goes to the GPU at once: after the mark it had only the short input norm, and
         * idled about 0.3 ms while the rest was encoded (673 times in an 8192-token prompt) */
        ok = ok && (!cpu || mark) && qg_mm(t->p0, t->normal, &L->qkv, rows) && (!cpu || q36_gpu_flush()) &&
             (cpu || qg_mm(t->p1, t->normal, &L->z, rows)) &&
             q36_dn_aux(t->decay, t->beta, t->normal, L->a, L->b, L->alog, L->dt, &s) &&
             (qg_verify_rows(g) && rows == g->verify ? qg_dn_rows(m, L, t, &s) :
              q36_dn_conv(t->conv, L->ring, t->p0, L->conv, L->snap_ring, &s) && q36_dn_l2(t->qk, t->conv, &s) &&
              q36_dn_delta(t->core, L->state, t->qk, t->conv, t->decay, t->beta, L->snap_state, &s)) &&
             q36_rmsnorm(t->normout, t->core, L->norm, rows * c->dn_vheads, c->dn_vdim, c->eps, 1) &&
             (!cpu || qg_cpu_mm(g, t->p1, t->normal, &L->z, rows, mark)) &&
             q36_gate(t->tmp, t->normout, t->p1, rows * width, 1);
        wo = &L->out;
    }
    int groups = c->n_group > 1 ? c->n_group : 1;
    ok = ok && qg_out(g, t->projected, t->tmp, wo, rows) && qg_cut(g, c->is_attn[il] ? 1 : 0) &&
         q36_add(t->sum, t->projected, t->x, rows * H) &&
         q36_rmsnorm(t->post, t->sum, L->post_ln, rows, H, c->eps, 0);
    /* Decode: the next layer's router on this layer's residual guesses the
     * experts it will choose, so qg_moe can start reading them early. The two
     * routers go in one dispatch. */
    const QgDense *next = il + 1 < c->n_layers ? &g->L[il + 1].router : NULL;
    if (ok && decode_ahead && rows == 1 && next)   /* a GGUF's two routers in two dispatches */
        ok = q36_rmsnorm(g->ahead, t->sum, g->L[il + 1].post_ln, 1, H, c->eps, 0) &&
             (L->router.gq ? qg_router(t->logits, t->post, &L->router, 1) &&
                             qg_router(g->ahead_logits, g->ahead, next, 1)
                           : q36_router_dot2(t->logits, t->post, L->router.q, L->router.sc, g->ahead_logits, g->ahead,
                                             next->q, next->sc, H, c->n_experts));
    else if (ok && rows == g->verify && next)   /* a verify: the same guess for each row */
        ok = qg_router(t->logits, t->post, &L->router, rows) &&
             q36_rmsnorm(g->ahead, t->sum, g->L[il + 1].post_ln, rows, H, c->eps, 0) &&
             qg_router(g->ahead_logits, g->ahead, next, rows);
    else ok = ok && qg_router(t->logits, t->post, &L->router, rows);
    ok = ok && q36_router_softmax(t->prob, t->logits, L->bias, rows, c->n_experts) &&
         q36_router_select(t->ids, t->weights, t->prob, rows, c->n_experts, c->topk, groups, c->topk_group) &&
         qg_cut(g, 4);
    return ok;
}
/* A prompt's last layer, when it is an attention layer: only the last row's output goes on, to
 * the final norm and the head (with the MTP head, the rows it reads: g->tail_keep, or every row
 * with g->no_tail). The other rows only write their keys and values to the cache:
 * the input norm, the k and v projections, the k norm and RoPE, as qg_trunk computes them. */
static int qg_trunk_kv(Model *m, Qwen36Graph *g, QgTrunk *t, int il, int rows, int pos) {
    Cfg *c = &m->c;
    QgLayer *L = &g->L[il];
    q36_shape s = {rows, c->hidden, c->q_heads, c->kv_heads, c->head_dim, c->q_head_dim, c->rotary_dim,
                   c->dn_vheads, c->dn_kheads, c->dn_kdim, c->dn_vdim, c->dn_convk, c->dn_conv_dim, pos, g->cap, 0,
                   g_dn_tiled, g_ssm_a_direct};
    int ok = q36_rmsnorm(t->normal, t->x, L->in_ln, rows, c->hidden, c->eps, 0) &&
             qg_mm(t->p1, t->normal, &L->k, rows) && qg_mm(t->p2, t->normal, &L->v, rows);
    if (ok && L->kn) ok = q36_rmsnorm(t->p1, t->p1, L->kn, rows * c->kv_heads, c->head_dim, c->eps, 0);
    if (ok && c->rotary_dim) ok = q36_attn_rope(t->p1, t->rot, c->kv_heads, &s);
    return ok && q36_attn_kv(L->kc, L->vc, t->p1, t->p2, &s);
}

static int qg_layer_legacy(Model *m, Qwen36Graph *g, int il, int rows, int pos, FILE *lf);

/* The last layer for a batch of a prompt: keys and values of its rows, and with last (the
 * prompt's final batch) the whole layer for its last g->tail_keep rows alone (its last row; with
 * the MTP head, the rows the head reads, qg_mtp_window), at their positions. Every row is
 * computed as in a whole batch, so exact mode keeps every bit; the last layer's queries,
 * attention, output projection and experts of the other rows are skipped. */
static int qg_layer_tail(Model *m, Qwen36Graph *g, int il, int rows, int pos, int last, FILE *lf) {
    int keep = !last ? 0 : g->tail_keep < 1 ? 1 : g->tail_keep < rows ? g->tail_keep : rows;
    int kv_rows = rows - keep, ok = 1;
    if (kv_rows > 0) {
        QgTrunk t;
        if (!qg_trunk_bind(&t, g, &m->c, 1, 0, kv_rows, 0)) return 0;
        ok = qg_trunk_kv(m, g, &t, il, kv_rows, pos);
        qg_trunk_release(&t);
    }
    if (!ok || !last) return ok;
    size_t H = (size_t)m->c.hidden, rd = m->c.rotary_dim > 0 ? (size_t)m->c.rotary_dim : 2;
    q36_tensor *x = g->x, *rot = g->rot;
    q36_tensor *xv = q36_tensor_view(x, (size_t)kv_rows * H * sizeof(float), (size_t)keep * H * sizeof(float));
    q36_tensor *rv = q36_tensor_view(rot, (size_t)kv_rows * rd * sizeof(float), (size_t)keep * rd * sizeof(float));
    if (xv && rv) {
        g->x = xv; g->rot = rv;
        ok = qg_layer_legacy(m, g, il, keep, pos + kv_rows, lf);
        g->x = x; g->rot = rot;
        if (g->dump_x == xv) { g->dump_x = x; g->dump_row = rows - 1; }   /* the same row, read at the next sync */
    } else ok = 0;
    /* encoded commands keep the buffer; q36_tensor_free only drops the views' descriptors */
    q36_tensor_free(xv); q36_tensor_free(rv);
    return ok;
}
static int qg_is_tail(const Qwen36Graph *g, const Cfg *c, int il, int rows) {
    return il == c->n_layers - 1 && c->is_attn[il] && rows > 1 && !g->no_tail;
}

static int qg_layer_legacy(Model *m, Qwen36Graph *g, int il, int rows, int pos, FILE *lf) {
    QgTrunk t;
    if (!qg_trunk_bind(&t, g, &m->c, m->c.is_attn[il], 0, rows, 0)) return 0;
    int ok = qg_trunk(m, g, &t, il, rows, pos, g->snap_rows, rows == 1);
    if (ok && g->ahead_advice.n)
        ok = qas_submit_pending(&g->ahead_advice, il, m, qg_advice_submit, qg_advice_emit);
    if (ok && g->vadv_n) {                      /* a verify's guesses for this layer, while its trunk runs */
        int n = g->vadv_layer == il ? g->vadv_n : 0;
        g->vadv_n = 0;
        if (n && (ok = q36_gpu_flush())) qg_advise(m, il, g->vadv_ids, 0, n);
    }
    if (!ok || !q36_gpu_sync()) return 0;
    qg_settle(m, g, 1);
    qg_dump_row(lf, g->projected, rows - 1, m->c.hidden);
    qg_dump_row(lf, g->sum, rows - 1, m->c.hidden);
    return qg_moe(m, g, il, rows, lf);
}

#include "qwen36_prefill_stream.h"

/* A fast prompt longer than one batch runs layer by layer: all its batches
 * through layer 0, then all through layer 1, and so on. A batch chooses
 * nearly every expert of a layer, and they do not stay in the cache until
 * the next batch: batch by batch, every batch read them all again (about
 * 18 GB on the 35B model). Here a layer's experts stay borrowed while all its
 * batches run (qwen36_prefill_stream.h), so the prompt reads each once. The
 * residual of every row waits in one buffer between layers, 4 bytes per
 * hidden unit and token (64 MB at 8192 tokens), with the RoPE table of all
 * the positions; both are freed at the end of the prompt. Every batch of a
 * layer sees the same inputs, state and KV rows as batch by batch, so the
 * results are the same bits. */
typedef struct { q36_tensor *x, *rot, *x0, *rot0, **xv, **rv; int nb; } QgPrompt;

static void qg_prompt_done(Qwen36Graph *g, QgPrompt *p) {
    if (p->x0) { g->x = p->x0; g->rot = p->rot0; }
    for (int b = 0; b < p->nb; b++) {
        if (p->xv) q36_tensor_free(p->xv[b]);
        if (p->rv) q36_tensor_free(p->rv[b]);
    }
    free(p->xv); free(p->rv);
    q36_tensor_free(p->x); q36_tensor_free(p->rot);
    g->layer_major = g->stream_more = g->stream_warm = 0;
    memset(p, 0, sizeof(*p));
}

/* The dense weights' pages. A prompt's expert reads fill the file cache, and
 * by its end canonical.qc's pages are out of memory: the head's, which the
 * prompt does not read (the first head after it waited about 300 ms for the
 * 509 MB of the 35B model's head, against 8 ms in memory), and in a prompt run
 * layer by layer the earlier layers' too (the first decode step after an
 * 8192-token prompt took 510 ms against about 140). The prompt asks for them
 * near its end: a prompt run layer by layer for the whole file in the second
 * batch of its second-to-last layer, when that layer's experts are loaded and
 * the disk is free; a shorter one when its last four layers start, sharing the
 * disk with their experts: the whole file from 1536 tokens (the first decode
 * step after 1536 and 2048 tokens 85 and 65 ms sooner, the prompt the same),
 * the head alone below (after 1024 tokens the whole file made the first decode
 * step 100 ms sooner but the prompt 90 ms longer). madvise reads the pages
 * before it returns (about 300 ms for the head), so another thread calls it,
 * at the priority of the prompt: the system slows down the reads of lower
 * ones. */
static struct { void *at; size_t bytes; } g_dense_pages;
static void qg_dense_pages(void *arg) {
    (void)arg;
    madvise(g_dense_pages.at, g_dense_pages.bytes, MADV_WILLNEED);
}
static void qg_prefetch_dense(Qwen36Graph *g, int whole) {
    int mapped = g->map || (g->head.gq && g->head.type != GQ_F32);   /* a GGUF's head in its file */
    if (!mapped || !g->head.q) return;          /* copied weights stay in memory */
    uintptr_t page = (uintptr_t)getpagesize(), at = (uintptr_t)q36_tensor_data(g->head.q);
    size_t bytes = q36_tensor_bytes(g->head.q);
    if (whole && g->map) {                      /* a GGUF: the head only, its file holds the experts too */
        at = (uintptr_t)g->map_base;
        bytes = g->map_bytes;
    }
    uintptr_t lo = at & ~(page - 1), hi = (at + bytes + page - 1) & ~(page - 1);
    g_dense_pages.at = (void *)lo;
    g_dense_pages.bytes = hi - lo;
    dispatch_async_f(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), NULL, qg_dense_pages);
}

/* A GGUF's dense matrices are locked in memory (mlock) from the start of a
 * prompt to the end of the decode step after it. The file keeps the routed
 * experts beside them, and a prompt's expert reads pushed the dense pages out
 * of memory, for the prompt itself and the first decode step to read back from
 * disk: on the 35B model on an 8 GB M1, 0.3 to 1.2 GB of the 1.9 GB were out
 * after a 512-token prompt, and that step took 0.7 to 1.5 s (0.3 to 0.5 s with
 * the pages locked). In a file of their own (tried with a copy at the same
 * offsets) they stayed, as canonical.qc's did beside experts.qc in the same
 * runs. During decode the GPU holds them in memory anyway; between requests
 * the system may take them back (a request that ends without a decode step
 * unlocks them at its end, q36_spec_end). In most runs the decode after a
 * locked prompt found fewer of its experts in the file cache (up to 9% more
 * read from disk, a few percent slower). A failed mlock is said once and not
 * tried again. The report at exit counts the locks and the unlocks, by where
 * they happened, and what stayed locked. */
static void qg_gguf_range(const QgDense *d, int lock, size_t *bytes, int *error) {
    if (!d->gq || !d->q || d->type == GQ_F32) return;   /* floats are copies */
    uintptr_t page = (uintptr_t)getpagesize(), at = (uintptr_t)q36_tensor_data(d->q);
    uintptr_t lo = at & ~(page - 1), hi = (at + q36_tensor_bytes(d->q) + page - 1) & ~(page - 1);
    if (!(lock ? mlock((void *)lo, hi - lo) : munlock((void *)lo, hi - lo))) *bytes += hi - lo;
    else if (!*error) *error = errno;
}
static size_t qg_gguf_pages(Model *m, Qwen36Graph *g, int lock, int *error) {
    size_t bytes = 0;
    *error = 0;
    qg_gguf_range(&g->head, lock, &bytes, error);
    for (int i = 0; i < m->c.n_layers; i++) {
        QgLayer *L = &g->L[i];
        const QgDense *d[] = {&L->q, &L->k, &L->v, &L->o, &L->qkv, &L->z, &L->out,
                              &L->router, &L->sh_g, &L->sh_u, &L->sh_d};
        for (size_t k = 0; k < sizeof(d) / sizeof(d[0]); k++) qg_gguf_range(d[k], lock, &bytes, error);
    }
    return bytes;
}
static void qg_gguf_lock(Model *m, Qwen36Graph *g, int lock, int where) {
    if (!g_gguf || g->gguf_locked < 0 || g->gguf_locked == lock) return;
    int error, ignored;
    size_t bytes = qg_gguf_pages(m, g, lock, &error);
    if (lock && error) {                        /* what was locked goes back */
        qg_gguf_pages(m, g, 0, &ignored);
        g->gguf_locked = -1;
        fprintf(stderr, "[qwen36-metal] GGUF: the dense matrices could not be locked in memory (mlock: %s); a "
                        "prompt's expert reads may push them out of it\n", strerror(error));
        return;
    }
    g->gguf_locked = lock;
    if (lock) qg_gguf_locks++;
    else qg_gguf_unlocks[where]++;
    static int said;
    if (lock && !said++)
        fprintf(stderr, "[qwen36-metal] GGUF: the dense matrices locked in memory (%zu bytes) from each prompt to "
                        "the decode step after it\n", bytes);
}

/* The forward waits for the GPU and calls qg_prompt_done whatever happens. */
static int qg_prompt_layers(Model *m, const int *ids, int S, int pos, FILE *lf, QgPrompt *p, int *last_rows) {
    Qwen36Graph *g = m->graph;
    Cfg *c = &m->c;
    int H = c->hidden, rd = c->rotary_dim > 0 ? c->rotary_dim : 2, R = g->rows;
    p->nb = (S + R - 1) / R;
    p->x = q36_tensor_new((size_t)S * H * sizeof(float));
    p->rot = q36_tensor_new((size_t)S * rd * sizeof(float));
    p->xv = calloc((size_t)p->nb, sizeof(*p->xv));
    p->rv = calloc((size_t)p->nb, sizeof(*p->rv));
    int ok = p->x && p->rot && p->xv && p->rv;
    for (int b = 0; ok && b < p->nb; b++) {
        size_t r0 = (size_t)b * R, n = (size_t)S - r0 < (size_t)R ? (size_t)S - r0 : (size_t)R;
        ok = (p->xv[b] = q36_tensor_view(p->x, r0 * H * sizeof(float), n * H * sizeof(float))) &&
             (p->rv[b] = q36_tensor_view(p->rot, r0 * rd * sizeof(float), n * rd * sizeof(float)));
    }
    /* x and the RoPE table are written by the CPU: nothing may be in flight */
    ok = q36_gpu_sync() && ok;
    qg_settle(m, g, ok);
    if (ok) ok = qgc_expected_phase(m->global_cache, S, pos);
    if (!ok) return 0;
    p->x0 = g->x; p->rot0 = g->rot;
    float *x = q36_tensor_data(p->x);
    for (int r = 0; ok && r < S; r++) ok = qwen_embedding_row(m, ids[r], x + (size_t)r * H);
    if (ok && c->rotary_dim) { g->rot = p->rot; qg_rope_table(c, g, S, pos); }
    g->layer_major = 1;
    for (int il = 0; ok && il < c->n_layers; il++)
        for (int b = 0; ok && b < p->nb; b++) {
            if (il == c->n_layers - 2 && b == 1) qg_prefetch_dense(g, 1);
            int s0 = b * R, rows = S - s0 < R ? S - s0 : R;
            int before = g->snap_at - (pos + s0);   /* rows of this batch up to the snapshot */
            g->snap_rows = before >= 1 && before <= rows ? before : 0;
            g->x = p->xv[b]; g->rot = p->rv[b];
            g->stream_more = b + 1 < p->nb;
            g->stream_warm = b > 0;
            q36_gpu_keepalive(0.25);
            FILE *dump = b + 1 == p->nb ? lf : NULL;
            ok = qg_is_tail(g, c, il, rows) ? qg_layer_tail(m, g, il, rows, pos + s0, b + 1 == p->nb, dump)
                                         : qg_layer(m, g, il, rows, pos + s0, dump);
            *last_rows = rows;
        }
    return ok;
}

/* The final residuals of the forward's rows first.. (n of them, from x), those from
 * row lo on, at h_rows[row - lo]: a step's or a verify's rows (lo 0), a prompt's
 * last QG_MTP_WINDOW + 1 (qg_mtp_window). */
static int qg_keep(Qwen36Graph *g, const float *x, int first, int n, int lo, int H) {
    int cut = lo - first < 0 ? 0 : lo - first < n ? lo - first : n;
    x += (size_t)cut * H;
    first += cut;
    n -= cut;
    size_t need = (size_t)(first - lo + n) * H;
    if (n && need > g->h_cap) {
        float *h = realloc(g->h_rows, need * sizeof(float));
        if (!h) return 0;
        g->h_rows = h;
        g->h_cap = need;
    }
    if (n) memcpy(g->h_rows + (size_t)(first - lo) * H, x, (size_t)n * H * sizeof(float));
    return 1;
}
static int qg_mtp_rows(Model *m, const float *h, const int *next, int n, int pos, float *logits, int vocab);
static int qg_mtp_window(Qwen36Graph *g, int pairs, int pos);
static int qg_mtp_idle_prompt(const Model *m, const int *ids, int S);

/* S tokens at positions pos..; logits of the last one, or of every row in a
 * verify (g->verify). lf: DUMP_LAYERS file, three rows per layer for the last
 * token (projection, residual, MoE output). With the MTP head the final residual
 * of every row it reads is kept (qg_keep), a prompt's pairs go through the head,
 * and the rows left wait for the token after them (qg_mtp_draft). */
static int qg_forward(Model *m, const int *ids, int S, int pos, float *logits, FILE *lf) {
    Qwen36Graph *g = m->graph;
    Cfg *c = &m->c;
    int verify = g->verify;
    qas_clear(&g->ahead_advice);
    g->vadv_n = 0;
    g->guess_layer = -1;                        /* a forward's guesses are its own (QWEN36_FAST_GUESSED) */
    g->guess_n = 0;
    g->defer_decode_advice = S == 1 && pos > 0;
    int R = S > 1 && !verify ? qg_prompt_rows() : S;
    /* A prompt's last layer runs whole for its last row only; with the MTP head, for the rows whose final
     * residual the head reads (all of them, or its window; the last row alone where the head will idle,
     * qg_mtp_idle_prompt), when they are in the prompt's last batch. */
    g->no_tail = verify;
    g->tail_keep = 1;
    int lo = 0;                                 /* the first row whose final residual the head reads */
    int idle = m->mtp && !verify && S > 1 && qg_mtp_idle_prompt(m, ids, S);
    if (m->mtp && !verify && S > 1) {
        int w = QG_MTP_WINDOW, last_rows = S - ((S - 1) / R) * R, need = idle ? 1 : S - 1 > w ? w + 1 : S;
        if (need <= last_rows) g->tail_keep = need;
        else g->no_tail = 1;
        lo = S - need;
    }
    int H = c->hidden, rows = 0, ok = S >= 1 && pos >= 0 && qg_reserve(m, pos + S) && qg_rows(m, S < R ? S : R);
    if (ok && S > 1 && !verify) qg_gguf_lock(m, g, 1, 0);
    /* Decode leaves the GPU waiting at every layer, and a prompt waits for
     * its experts too: keep the clocks up until a quarter of a second after
     * the last renewal (every step, and every layer and group of a prompt). */
    q36_gpu_keepalive(0.25);
    /* A one-draft verify's kernels copy the DeltaNet state after its first row into the first spec copies (one
     * batch, through the snapshot copies); a longer verify, or an exact one, runs the recurrence row by row
     * (qg_dn_rows). */
    int batch_snap = verify == 2 && !qg_verify_rows(g);
    if (batch_snap)
        for (int i = 0; i < c->n_layers; i++) {
            QgLayer *L = &g->L[i];
            if (!L->state) continue;
            q36_tensor *s = L->snap_state, *r = L->snap_ring;
            L->snap_state = L->spec_state[0]; L->snap_ring = L->spec_ring[0];
            L->spec_state[0] = s; L->spec_ring[0] = r;
        }
    QgPrompt lp = {0};
    if (ok && S > g->rows && g->stream_enabled && !q36_gpu_is_exact())
        ok = qg_prompt_layers(m, ids, S, pos, lf, &lp, &rows);
    else for (int s0 = 0; ok && s0 < S; s0 += rows) {
        int prev = rows;
        rows = S - s0 < g->rows ? S - s0 : g->rows;
        int before = g->snap_at - (pos + s0);   /* rows of this batch up to the snapshot */
        g->snap_rows = batch_snap ? 1 : verify ? 0 : before >= 1 && before <= rows ? before : 0;
        /* x and the RoPE table are rewritten by the CPU: nothing may be in flight */
        ok = q36_gpu_sync();
        qg_settle(m, g, ok);
        if (ok && m->mtp && s0 > 0) ok = qg_keep(g, q36_tensor_data(g->x), s0 - prev, prev, lo, H);
        /* a verify is decode for the expert cache's reuse estimates */
        if (ok && s0 == 0) ok = qgc_expected_phase(m->global_cache, verify ? 1 : S, pos);
        float *x = q36_tensor_data(g->x);
        for (int r = 0; ok && r < rows; r++) ok = qwen_embedding_row(m, ids[s0 + r], x + (size_t)r * H);
        if (ok && c->rotary_dim) qg_rope_table(c, g, rows, pos + s0);
        for (int il = 0; ok && il < c->n_layers; il++) {
            if (S > 1) q36_gpu_keepalive(0.25);
            if (S > 64 && il == c->n_layers - 4 && s0 + rows == S) qg_prefetch_dense(g, S >= 1536);
            FILE *dump = s0 + rows == S ? lf : NULL;
            ok = qg_is_tail(g, c, il, rows) ? qg_layer_tail(m, g, il, rows, pos + s0, s0 + rows == S, dump)
                                            : qg_layer(m, g, il, rows, pos + s0, dump);
        }
    }
    /* The final norm is the CPU engine's double sum of squares. The head
     * runs on the GPU, with ordered sums in exact mode and vector sums in
     * fast mode. It costs one more wait for the GPU per forward. */
    ok = q36_gpu_sync() && ok;
    qg_settle(m, g, ok);
    qas_clear(&g->ahead_advice);
    g->defer_decode_advice = 0;
    g->snap_rows = 0;
    if (batch_snap)
        for (int i = 0; i < c->n_layers; i++) {
            QgLayer *L = &g->L[i];
            if (!L->state) continue;
            q36_tensor *s = L->snap_state, *r = L->snap_ring;
            L->snap_state = L->spec_state[0]; L->snap_ring = L->spec_ring[0];
            L->spec_state[0] = s; L->spec_ring[0] = r;
        }
    if (ok && m->mtp)       /* the rows' final residuals: the whole prompt layer by layer, else the last batch */
        ok = lp.x ? qg_keep(g, q36_tensor_data(lp.x), 0, S, lo, H)
                  : qg_keep(g, q36_tensor_data(g->x), S - rows, rows, lo, H);
    int nl = verify ? S : 1;                    /* rows whose logits come back */
    q36_tensor *last = verify ? g->last_rows : g->last, *out = verify ? g->head_rows : g->head_out;
    for (int r = 0; ok && r < nl; r++)
        rmsnorm_row((float *)q36_tensor_data(last) + (size_t)r * H,
                    (const float *)q36_tensor_data(g->x) + (size_t)(rows - nl + r) * H, m->final_norm, H, c->eps);
    qg_prompt_done(g, &lp);
    ok = ok && qg_mm(out, last, &g->head, nl) && qg_cut(g, 5) && q36_gpu_sync();
    if (ok) memcpy(logits, q36_tensor_data(out), (size_t)nl * c->vocab * sizeof(float));
    if (tm_on()) {      /* as on the CPU, the MoE time holds the router, the shared expert and the disk */
        const double *t = g->ms;
        tm_add(S, 0, t[0]); tm_add(S, 1, t[1]); tm_add(S, 2, t[2] + t[3] + t[4] + t[6]);
        tm_add(S, 3, t[3]); tm_add(S, 4, t[4]); tm_add(S, 5, t[5]);
        if (S == 1) { g_xf_load += t[6]; g_xf_run += t[2]; g_tm_dec_tokens++; } else g_tm_pre_tokens += S;
        memset(g->ms, 0, sizeof(g->ms));
    }
    if (g->stream_initialized) {
        /* The readers sleep on the cache's condition: join them at the end of
         * the prompt, so that decode does not wake four idle threads. */
        if (!qll_destroy(&g->stream_loader)) abort();
        g->stream_initialized = 0;
    }
    if (ok && m->mtp) {
        if (S > 1 && !verify) {
            /* a prompt: its pairs through the head (the last QG_MTP_WINDOW only, see qg_mtp_window: h_rows holds
             * the rows from there, lo), none where the head will idle (its attention restarts where it runs
             * again); its last row waits for the next token */
            int skip = idle ? S - 1 : qg_mtp_window(g, S - 1, pos);
            ok = skip == lo && (idle || qg_mtp_rows(m, g->h_rows, ids + 1 + skip, S - 1 - skip, pos + skip, NULL, 0));
            if (idle) g->mtp_len = -1;
            memcpy(g->mtp_pend, g->h_rows + (size_t)(S - 1 - skip) * H, (size_t)H * sizeof(float));
            g->mtp_pend_n = 1;
            g->mtp_pend_pos = pos + S - 1;
            if (g->h_cap > (size_t)Q36_SPEC_ROWS * H) { free(g->h_rows); g->h_rows = NULL; g->h_cap = 0; }
        } else {
            /* a step or a verify: its rows wait; a verify's drafts are settled by q36_spec_step (qg_mtp_keep) */
            memcpy(g->mtp_pend, g->h_rows, (size_t)S * H * sizeof(float));
            for (int j = 0; j + 1 < S; j++) g->mtp_pend_tok[j] = ids[j + 1];
            g->mtp_pend_n = S;
            g->mtp_pend_pos = pos;
        }
    }
    if (S > 1 && !verify) q36_gpu_keepalive(0);       /* also on a failed prefill; a verify is decode */
    if (S == 1 || verify) qg_gguf_lock(m, g, 0, 0);   /* the step after a prompt: the GPU holds them now */
    return ok;
}

/* ---- speculative decoding (q36_spec_step) --------------------------------- */

/* The copies a verify of k drafts needs, allocated the first time: the DeltaNet
 * state after each of its first k rows (each as large as the state: 63 MiB on the
 * 35B model in fast mode, 123 MiB in exact mode) and the final norms and logits of
 * its rows. 0 = no memory. */
static int qg_spec_alloc(Model *m, int k) {
    Qwen36Graph *g = m->graph;
    Cfg *c = &m->c;
    if (k < 1 || k > Q36_SPEC_ROWS - 1) return 0;
    for (int i = 0; i < c->n_layers; i++) {
        QgLayer *L = &g->L[i];
        if (!L->state) continue;
        for (int j = 0; j < k; j++) {
            if (!L->spec_state[j] && !(L->spec_state[j] = q36_tensor_new(q36_tensor_bytes(L->state)))) return 0;
            if (!L->spec_ring[j] && !(L->spec_ring[j] = q36_tensor_new(q36_tensor_bytes(L->ring)))) return 0;
        }
    }
    if (!g->last_rows && !(g->last_rows = q36_tensor_new((size_t)Q36_SPEC_ROWS * c->hidden * sizeof(float)))) return 0;
    if (!g->head_rows && !(g->head_rows = q36_tensor_new((size_t)Q36_SPEC_ROWS * c->vocab * sizeof(float)))) return 0;
    return 1;
}

/* A verify's draft was rejected: the state after its first keep rows comes back. */
static void qg_spec_rollback(Model *m, int keep) {
    Qwen36Graph *g = m->graph;
    qg_settle(m, g, q36_gpu_sync());
    if (keep < 1 || keep > Q36_SPEC_ROWS - 1) abort();
    for (int i = 0; i < m->c.n_layers; i++) {
        QgLayer *L = &g->L[i];
        if (!L->state) continue;
        q36_tensor *s = L->state, *r = L->ring;
        L->state = L->spec_state[keep - 1]; L->ring = L->spec_ring[keep - 1];
        L->spec_state[keep - 1] = s; L->spec_ring[keep - 1] = r;
    }
}

/* The MTP head over n pairs at positions pos..: row r pairs the model's final
 * residual h[r] at pos + r with the token next[r] at pos + r + 1. Their K/V
 * rows go into the head's attention (its own cache, layer c.n_layers), the
 * routed experts through the expert cache; with logits, the last row's logits
 * come back: what the head says follows next[n - 1]. The input is the
 * checkpoint's: the embedding normed by pre_fc_norm_embedding, then the residual
 * after the model's final norm (as the head was trained) normed by
 * pre_fc_norm_hidden, both into fc.
 *
 * A prompt's pairs go through the head for its last QG_MTP_WINDOW positions only,
 * and the head's attention restarts there (its K/V rows are stored from row 0 for
 * that position: mtp_off; RoPE keeps the true positions): the head sees the recent
 * context, and the prompt pays 32 rows of the head instead of all of them. Returns
 * the pairs skipped. */
static int qg_mtp_window(Qwen36Graph *g, int pairs, int pos) {
    int w = QG_MTP_WINDOW;
    if (pairs <= w) return 0;
    int skip = pairs - w;
    g->mtp_off = pos + skip;
    g->mtp_len = pos + skip;
    return skip;
}

static int qg_mtp_rows(Model *m, const float *h, const int *next, int n, int pos, float *logits, int vocab) {
    Qwen36Graph *g = m->graph;
    Cfg *c = &m->c;
    int H = c->hidden, il = c->n_layers, rows = 0;
    int R = n <= g->rows ? g->rows : n < qg_prompt_rows() ? n : qg_prompt_rows();   /* no new tensors when they hold n rows */
    float *e = malloc((size_t)H * sizeof(float)), *hn = malloc((size_t)H * sizeof(float));
    int ok = e && hn && n > 0 && qg_reserve(m, pos + n) && qg_rows(m, R);
    g->snap_rows = 0;
    /* A pair that does not follow the head's rows (a turn resumed from the copy of an earlier position, a prompt
     * after the row a generation left pending): the head's attention restarts there, as at a prompt's window. */
    if (g->mtp_len != pos) g->mtp_off = g->mtp_len = pos;
    for (int b0 = 0; ok && b0 < n; b0 += rows) {
        rows = n - b0 < g->rows ? n - b0 : g->rows;
        ok = q36_gpu_sync();
        qg_settle(m, g, ok);
        float *in = q36_tensor_data(g->mtp_in);
        for (int r = 0; ok && r < rows; r++) {
            float *row = in + (size_t)r * 2 * H;
            ok = qwen_embedding_row(m, next[b0 + r], e);
            rmsnorm_row(row, e, m->mtp_norm_emb, H, c->eps);
            rmsnorm_row(hn, h + (size_t)(b0 + r) * H, m->final_norm, H, c->eps);
            rmsnorm_row(row + H, hn, m->mtp_norm_hid, H, c->eps);
        }
        if (ok && c->rotary_dim) qg_rope_table(c, g, rows, pos + b0);   /* RoPE at the true positions */
        ok = ok && qg_mm(g->x, g->mtp_in, &g->mtp_fc, rows) &&
             qg_layer_legacy(m, g, il, rows, pos + b0 - g->mtp_off, NULL);
    }
    ok = q36_gpu_sync() && ok;
    qg_settle(m, g, ok);
    if (ok) g->mtp_len = pos + n;
    if (ok && logits) {
        rmsnorm_row(q36_tensor_data(g->last), (const float *)q36_tensor_data(g->x) + (size_t)(rows - 1) * H,
                    m->mtp_norm, H, c->eps);
        QgDense head = g->head;
        head.out = vocab;                       /* the first `vocab` rows of the head */
        ok = qg_mm(g->head_out, g->last, &head, 1) && q36_gpu_sync();
        if (ok) memcpy(logits, q36_tensor_data(g->head_out), (size_t)vocab * sizeof(float));
    }
    free(e);
    free(hn);
    return ok;
}

/* After a verify: of its rows, the first keep stood. */
static void qg_mtp_keep(Model *m, int keep) {
    Qwen36Graph *g = m->graph;
    if (keep < g->mtp_pend_n) g->mtp_pend_n = keep;
}

/* The token ids the head drafts from: the first QG_MTP_VOCAB, or Q36_MTP_VOCAB (tests), at most the vocabulary. */
static int qg_mtp_vocab(int V) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("Q36_MTP_VOCAB");
        v = e && *e ? atoi(e) : QG_MTP_VOCAB;
        if (v < 1) v = QG_MTP_VOCAB;
    }
    return v < V ? v : V;
}

/* Whether the head idles at the step after a prompt whatever its token (q36_spec_step: half or more of the last 16
 * tokens past the ids it drafts from): 8 or more of the prompt's last 15 tokens are (of all of them when it has
 * fewer: the tokens before can only add). Its window over the prompt (qg_mtp_window) would then go unused, its
 * attention starting again where it runs again: the prompt skips it. */
static int qg_mtp_idle_prompt(const Model *m, const int *ids, int S) {
    int past = 0, vocab = qg_mtp_vocab(m->c.vocab);
    for (int i = S > 15 ? S - 15 : 0; i < S; i++) past += ids[i] >= vocab;
    return past >= 8;
}

/* The pending rows meet the token after the last, tok: through the head, whose
 * most likely next token among the first QG_MTP_VOCAB ids (qg_mtp_vocab) is the
 * draft, with its probability under a softmax of the head's logits for those ids
 * (the draft's confidence): a draft reads 67 MB of the model's 0.5 GB lm_head.
 * Those ids hold most tokens of English, other Latin-script text and code, but no
 * Chinese character, and few Arabic or Korean ones: where most tokens lie past
 * them, q36_spec_step leaves the head idle. 0 = nothing pending or a failure.
 * Q36_MTP_DUMP (tests) appends the record of the other engines' head
 * (qwen36_mtp.h): the row tok pairs with, tok, the logits (past those ids -inf). */
static int qg_mtp_draft(Model *m, int tok, int *draft, float *prob) {
    Qwen36Graph *g = m->graph;
    int n = g->mtp_pend_n, next[Q36_SPEC_ROWS], V = m->c.vocab, vocab = qg_mtp_vocab(V);
    if (!n) return 0;
    for (int j = 0; j + 1 < n; j++) next[j] = g->mtp_pend_tok[j];
    next[n - 1] = tok;
    float *logits = malloc((size_t)V * sizeof(float));
    int ok = logits && qg_mtp_rows(m, g->mtp_pend, next, n, g->mtp_pend_pos, logits, vocab);
    g->mtp_pend_n = 0;
    const char *dump = getenv("Q36_MTP_DUMP");
    if (ok && dump && *dump) {
        int row = g->mtp_pend_pos + n - 1;
        for (int i = vocab; i < V; i++) logits[i] = -INFINITY;
        FILE *f = fopen(dump, "ab");
        if (!f || fwrite(&row, 4, 1, f) != 1 || fwrite(&tok, 4, 1, f) != 1 ||
            fwrite(logits, 4, (size_t)V, f) != (size_t)V) { fprintf(stderr, "Q36_MTP_DUMP write failed\n"); exit(1); }
        fclose(f);
    }
    if (ok) {
        int best = 0;
        for (int i = 1; i < vocab; i++) if (logits[i] > logits[best]) best = i;
        double z = 0;
        for (int i = 0; i < vocab; i++) z += exp((double)logits[i] - logits[best]);
        *draft = best;
        *prob = (float)(1.0 / z);
    }
    free(logits);
    return ok;
}

#endif
#endif
