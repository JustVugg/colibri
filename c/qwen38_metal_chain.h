/* The Qwen3.8 dense chain on the Apple GPU (METAL=1 builds, QWEN38_METAL): every
 * layer, the final mixer and lm_head on qwen36's Metal device, through the q36_* and
 * q38_* kernels (qwen36_gpu.h, qwen38_gpu.h), as qwen38_chain.h does on Vulkan and
 * qwen38_cuda_chain.h on CUDA. Included by qwen38_core.h; one conversation only.
 *
 * Startup (qwen38.c, after the Vulkan block): q38m_decide picks the chain or prints
 * the one line that says why the CPU engine runs the model, then q38m_start sets up
 * its device state. q38_forward offers every step to q38m_forward; a 0 there means
 * the CPU runs that step from the host state (q38m_sync_host brings it down first).
 *
 * So far: the decision, the report and the dense weights on the GPU (q38m_start) are
 * real; q38m_forward declines every step. */
#ifndef QWEN38_METAL_CHAIN_H
#define QWEN38_METAL_CHAIN_H
#include "qwen36_gpu.h"
#include "qwen38_gpu.h"

#define Q38M_UNUSED __attribute__((unused))

static int g_mtl_chain;                        /* 1 when the chain runs steps */
static unsigned long long g_q38m_forwards, g_q38m_declined;

static void q38m_off(const char *why) {
    fprintf(stderr, "[qwen38-metal] the Metal chain is off: %s; the CPU engine runs the model\n", why);
}

static const char *q38m_kind_name(const Q38Weight *w, char *buf, size_t n) {
    static const char *names[] = { "NONE", "F32", "BF16", "FP8", "INT4G64", "GGML", "INT2G32",
                                   "LOWBIT16", "VQ2", "VQ1", "MXFP4" };
    if (w->kind == Q38_WEIGHT_GGML) snprintf(buf, n, "GGML type %d", w->gq_type);
    else if ((unsigned)w->kind < sizeof names / sizeof names[0]) snprintf(buf, n, "%s", names[w->kind]);
    else snprintf(buf, n, "%d", (int)w->kind);
    return buf;
}

/* A trunk weight the chain's kernels read: GGML rows of Q8_0 (8), Q4_K (12), Q5_K (13),
 * Q6_K (14), IQ4_NL (20) or IQ4_XS (23), BF16 or F32. An absent one (NONE) is fine. */
static int q38m_weight_ok(const Q38Weight *w) {
    switch (w->kind) {
    case Q38_WEIGHT_NONE: case Q38_WEIGHT_F32: case Q38_WEIGHT_BF16: return 1;
    case Q38_WEIGHT_GGML:
        switch (w->gq_type) { case 8: case 12: case 13: case 14: case 20: case 23: return 1; default: return 0; }
    default: return 0;
    }
}

/* A layer's dense matrices, in the order the chain keeps them (Q38mW L[layer][j]), and the
 * tail's (final mixer, lm_head). An absent one is NONE. */
enum { Q38M_NW = 22, Q38M_NT = 4 };
static const char *const q38m_wname[Q38M_NW] = {
    "attn_gr.down", "attn_gr.up", "attn_gr.inject", "mlp_gr.down", "mlp_gr.up", "mlp_gr.inject",
    "router", "sh_g", "sh_u", "sh_d", "q", "k", "v", "o", "idx_qk",
    "dn_qkv", "dn_z", "dn_b", "dn_a", "dn_out", "ple_key", "ple_value",
};
static const char *const q38m_tname[Q38M_NT] = { "final_gr.down", "final_gr.up", "final_gr.inject", "lm_head" };
static void q38m_layer_srcs(const Layer *l, const Q38Weight *w[Q38M_NW]) {
    const Q38Weight *s[Q38M_NW] = {
        &l->attn_gr.down, &l->attn_gr.up, &l->attn_gr.inject,
        &l->mlp_gr.down, &l->mlp_gr.up, &l->mlp_gr.inject,
        &l->router, &l->sh_g, &l->sh_u, &l->sh_d,
        &l->q, &l->k, &l->v, &l->o, &l->idx_qk,
        &l->dn_qkv, &l->dn_z, &l->dn_b, &l->dn_a, &l->dn_out,
        &l->ple_key, &l->ple_value,
    };
    memcpy(w, s, sizeof s);
}
static void q38m_tail_srcs(const Model *m, const Q38Weight *w[Q38M_NT]) {
    w[0] = &m->final_gr.down; w[1] = &m->final_gr.up; w[2] = &m->final_gr.inject; w[3] = &m->lm_head;
}

/* The first dense weight the chain cannot read, NULL when it reads them all. The
 * embedding stays on the host (its rows are gathered on the CPU). */
static const Q38Weight *q38m_dense_unsupported(const Model *m) {
    const Q38Weight *w[Q38M_NW], *t[Q38M_NT];
    for (int i = 0; i < m->c.layers; i++) {
        q38m_layer_srcs(&m->L[i], w);
        for (int j = 0; j < Q38M_NW; j++)
            if (!q38m_weight_ok(w[j])) return w[j];
    }
    q38m_tail_srcs(m, t);
    for (int j = 0; j < Q38M_NT; j++)
        if (!q38m_weight_ok(t[j])) return t[j];
    return NULL;
}

static int q38m_decide(Model *m) {
    const char *why = NULL;
    char buf[96];
    const char *opt = getenv("COLI_METAL");
    const char *ks = getenv("KV_SLOTS");
    const char *mtp = getenv("Q38_MTP");
    const Q38Weight *bad;
    if (opt && *opt && !strcmp(opt, "0")) why = "COLI_METAL=0";
    /* KV_SLOTS as the serve reads it (g_q38_mux_slots); a run that is not a serve ignores
     * it, and the chain steps aside for it there too */
    else if (ks && *ks && atoi(ks) > 1) why = "KV_SLOTS>1 (one conversation on the GPU)";
    else if (m->vis_ready) why = "vision tower";
    else if (mtp && *mtp && strcmp(mtp, "0")) why = "MTP (Q38_MTP=1) needs the CPU engine until M5";
    else if ((bad = q38m_dense_unsupported(m))) {
        char kind[32];
        snprintf(buf, sizeof buf, "a dense weight of kind %s is not supported", q38m_kind_name(bad, kind, sizeof kind));
        why = buf;
    } else {
        const char *device = q36_gpu_default_device_name();
        if (!*device) why = "no Metal device";
        else if (strstr(device, "Apple Paravirtual device")) why = "Apple Paravirtual device";
        else if (!q38_gpu_init()) why = "q38_gpu_init failed";
    }
    if (why) q38m_off(why);
    return !why;
}

/* ---- dense weights -------------------------------------------------------------------
 * Where the bytes are: qwen38 reads a GGUF with pread, not mmap, so a GGML matrix's bytes
 * are the CPU's own heap copy (Q38Weight.data, q38_gg_weight_parts), which the CPU keeps for
 * its gq_matmul for the life of the model (q38_trunk_cpu_int8 skips a GGUF's blocks). Under
 * QWEN38_METAL that copy is allocated page aligned and page rounded (q38_gg_alloc), and the
 * chain maps it in place (q36_map_new + q36_map_view_own): the GPU reads the CPU's bytes,
 * no second copy. BF16 and F32 matrices are uploaded (copied): q38_trunk_cpu_int8 replaces
 * them with int8 rows and frees them, after the chain's start (qwen38.c). A matrix that is
 * not page aligned (or a map that fails) is uploaded too, so every weight sits at offset 0
 * of its buffer (q36_dot_gq wants an even offset). qwen38's dense GGML matrices have no
 * ".scale" companion (the loader reads none), so scale is 1. */
enum { Q38M_GQ = 1, Q38M_BF16, Q38M_F32 };
typedef struct { q36_tensor *t; int kind; /* Q38M_GQ | Q38M_BF16 | Q38M_F32 */ int type; float scale; int in, out; } Q38mW;

static struct {
    int layers;
    Q38mW (*L)[Q38M_NW];        /* [layer][j], q38m_wname order; t NULL for an absent matrix */
    Q38mW tail[Q38M_NT];
    q36_map **maps; int nmaps, cmaps;   /* the in-place maps, freed with the chain */
    size_t mapped, uploaded;    /* bytes */
    int n_mapped, n_uploaded, n_f32_from_bf16;
} g_q38m;

static int q38m_keep_map(q36_map *map) {
    if (g_q38m.nmaps == g_q38m.cmaps) {
        int c = g_q38m.cmaps ? 2 * g_q38m.cmaps : 64;
        q36_map **p = (q36_map **)realloc(g_q38m.maps, (size_t)c * sizeof *p);
        if (!p) return 0;
        g_q38m.maps = p; g_q38m.cmaps = c;
    }
    g_q38m.maps[g_q38m.nmaps++] = map;
    return 1;
}

/* in-place map over the bytes the CPU holds when they are page aligned, else upload; 0 =
 * unsupported kind (or no device memory) */
static int q38m_weight(Q38mW *w, const Q38Weight *src, int in, int out) {
    memset(w, 0, sizeof *w);
    w->scale = 1.f; w->in = in; w->out = out;
    if (!q38m_weight_ok(src) || src->kind == Q38_WEIGHT_NONE || !src->data || src->rows != out || src->cols != in) return 0;
    size_t n = (size_t)in * (size_t)out;
    if (src->kind == Q38_WEIGHT_GGML) {
        int64_t rb = gq_row_bytes(src->gq_type, in);
        if (rb <= 0) return 0;
        size_t bytes = (size_t)rb * (size_t)out;
        long page = sysconf(_SC_PAGESIZE);
        if (page > 0 && !((uintptr_t)src->data % (size_t)page)) {
            q36_map *map = q36_map_new(src->data, bytes);
            q36_tensor *t = map ? q36_map_view_own(map, 0, bytes) : NULL;
            if (t && q38m_keep_map(map)) { w->t = t; g_q38m.mapped += bytes; g_q38m.n_mapped++; }
            else { q36_tensor_free(t); q36_map_free(map); }
        }
        if (!w->t) {
            if (!(w->t = q36_tensor_upload(src->data, bytes))) return 0;
            g_q38m.uploaded += bytes; g_q38m.n_uploaded++;
        }
        w->kind = Q38M_GQ; w->type = src->gq_type;
        return 1;
    }
    if (src->kind == Q38_WEIGHT_BF16 && in % 8 == 0) {
        if (!(w->t = q36_tensor_upload(src->data, n * sizeof(uint16_t)))) return 0;
        w->kind = Q38M_BF16; g_q38m.uploaded += n * sizeof(uint16_t); g_q38m.n_uploaded++;
        return 1;
    }
    if (src->kind == Q38_WEIGHT_BF16) {   /* q38_dot_bf16 reads 8 inputs at a time: as F32 */
        float *f = (float *)malloc(n * sizeof(float));
        if (!f) return 0;
        const uint16_t *b = (const uint16_t *)src->data;
        for (size_t i = 0; i < n; i++) f[i] = bf16_to_f32(b[i]);
        w->t = q36_tensor_upload(f, n * sizeof(float));
        free(f);
        if (!w->t) return 0;
        w->kind = Q38M_F32; g_q38m.uploaded += n * sizeof(float); g_q38m.n_uploaded++; g_q38m.n_f32_from_bf16++;
        return 1;
    }
    if (!(w->t = q36_tensor_upload(src->data, n * sizeof(float)))) return 0;
    w->kind = Q38M_F32; g_q38m.uploaded += n * sizeof(float); g_q38m.n_uploaded++;
    return 1;
}

/* y (rows x out) = x (rows x in) . w^T on the GPU, encoded: GQ -> q36_dot_gq(type, scale),
 * BF16 -> q38_dot_bf16, F32 -> q36_dot_f32. 0, nothing encoded, for a shape it does not take. */
static int q38m_mm(q36_tensor *y, q36_tensor *x, const Q38mW *w, int rows) {
    if (!w->t) return 0;
    switch (w->kind) {
    case Q38M_GQ: return q36_dot_gq(y, x, w->t, w->type, w->scale, rows, w->in, w->out);
    case Q38M_BF16: return q38_dot_bf16(y, x, w->t, rows, w->in, w->out);
    case Q38M_F32: return q36_dot_f32(y, x, w->t, rows, w->in, w->out);
    default: return 0;
    }
}

static void q38m_free_weights(void) {
    for (int i = 0; i < g_q38m.layers; i++)
        for (int j = 0; j < Q38M_NW; j++) q36_tensor_free(g_q38m.L[i][j].t);
    for (int j = 0; j < Q38M_NT; j++) q36_tensor_free(g_q38m.tail[j].t);
    for (int i = 0; i < g_q38m.nmaps; i++) q36_map_free(g_q38m.maps[i]);
    free(g_q38m.L); free(g_q38m.maps);
    memset(&g_q38m, 0, sizeof g_q38m);
}

/* Q38M_CHECK_WEIGHTS=1 (debug only): one random row through every dense matrix of layers 0
 * and 3 on the GPU and through the CPU's q38_dense_matmul (run before q38_trunk_cpu_int8,
 * so on the same bytes), the worst error of an output over its Ruling 9 scale,
 * sum_i |w[o][i] x[i]|. Prints `[q38m-check] <name> worst=<e>`, FAIL over 2e-5. */
static void q38m_check_one(Model *m, const char *name, const Q38Weight *src, const Q38mW *w) {
    const int in = w->in, out = w->out;
    float *x = (float *)malloc((size_t)in * sizeof(float)), *yc = (float *)malloc((size_t)out * sizeof(float));
    float *row = (float *)malloc((size_t)in * sizeof(float));
    if (!x || !yc || !row) { fprintf(stderr, "[q38m-check] %s OOM FAIL\n", name); free(x); free(yc); free(row); return; }
    uint32_t r = 0x9e3779b9u ^ (uint32_t)(in * 7919 + out);
    for (int i = 0; i < in; i++) { r = r * 1664525u + 1013904223u; x[i] = (float)((r >> 8) / 16777216.0 * 2.0 - 1.0); }
    q36_tensor *xt = q36_tensor_upload(x, (size_t)in * sizeof(float)), *yt = q36_tensor_new((size_t)out * sizeof(float));
    int ok = xt && yt && q38m_mm(yt, xt, w, 1) && q36_gpu_sync();
    q38_dense_matmul(m, yc, x, src, 1, in, out);
    double worst = ok ? 0.0 : INFINITY;
    if (ok) {
        const float *yg = (const float *)q36_tensor_data(yt);
        for (int o = 0; o < out; o++) {
            q38_weight_row(src, o, row);
            double s = 0.0;
            for (int i = 0; i < in; i++) s += fabs((double)row[i] * x[i]);
            double e = fabs((double)yg[o] - yc[o]) / (s + 1e-30);
            if (!(e <= worst)) worst = e;   /* a NaN is the worst */
        }
    }
    fprintf(stderr, "[q38m-check] %s worst=%.3g%s\n", name, worst, worst <= 2e-5 ? "" : " FAIL");
    q36_tensor_free(xt); q36_tensor_free(yt);
    free(x); free(yc); free(row);
}

static void q38m_check_weights(Model *m) {
    const int layers[2] = { 0, 3 };
    const Q38Weight *src[Q38M_NW];
    char name[64];
    for (int k = 0; k < 2; k++) {
        int i = layers[k];
        if (i >= m->c.layers) continue;
        q38m_layer_srcs(&m->L[i], src);
        for (int j = 0; j < Q38M_NW; j++) {
            if (src[j]->kind == Q38_WEIGHT_NONE) continue;
            snprintf(name, sizeof name, "L%d.%s", i, q38m_wname[j]);
            q38m_check_one(m, name, src[j], &g_q38m.L[i][j]);
        }
    }
}

/* The chain's device state: every dense matrix on the GPU (above). Called before
 * q38_trunk_cpu_int8 frees the BF16 rows it uploads. 0, with the off line, on a failure. */
static int q38m_start(Model *m) {
    const int L = m->c.layers;
    const Q38Weight *src[Q38M_NW], *tail[Q38M_NT];
    char kind[32], why[160];
    g_q38m.L = (Q38mW (*)[Q38M_NW])calloc((size_t)(L > 0 ? L : 1), sizeof *g_q38m.L);
    if (!g_q38m.L) { q38m_off("no memory for the dense weights' table"); return 0; }
    g_q38m.layers = L;
    for (int i = 0; i < L; i++) {
        q38m_layer_srcs(&m->L[i], src);
        for (int j = 0; j < Q38M_NW; j++) {
            if (src[j]->kind == Q38_WEIGHT_NONE) continue;
            if (!q38m_weight(&g_q38m.L[i][j], src[j], src[j]->cols, src[j]->rows)) {
                snprintf(why, sizeof why, "layer %d's %s (%s) did not reach the GPU", i, q38m_wname[j],
                         q38m_kind_name(src[j], kind, sizeof kind));
                q38m_free_weights(); q38m_off(why); return 0;
            }
        }
    }
    q38m_tail_srcs(m, tail);
    for (int j = 0; j < Q38M_NT; j++) {
        if (tail[j]->kind == Q38_WEIGHT_NONE) continue;
        if (!q38m_weight(&g_q38m.tail[j], tail[j], tail[j]->cols, tail[j]->rows)) {
            snprintf(why, sizeof why, "the %s (%s) did not reach the GPU", q38m_tname[j], q38m_kind_name(tail[j], kind, sizeof kind));
            q38m_free_weights(); q38m_off(why); return 0;
        }
    }
    fprintf(stderr, "[qwen38-metal] dense weights: %d mapped in place (%.1f MiB), %d uploaded (%.1f MiB%s)\n",
            g_q38m.n_mapped, g_q38m.mapped / 1048576.0, g_q38m.n_uploaded, g_q38m.uploaded / 1048576.0,
            g_q38m.n_f32_from_bf16 ? ", some BF16 as F32" : "");
    const char *chk = getenv("Q38M_CHECK_WEIGHTS");
    if (chk && !strcmp(chk, "1")) q38m_check_weights(m);
    return 1;
}

/* 0: not taken (the CPU runs the step from the host state) | 1: done, the logits of
 * the last nlogits rows in logit, the final streams in hyper_h when want_streams,
 * every row's mixed in mixed_h when it is not NULL. */
static int q38m_forward(Model *m, const int *ids, int S, int pos_base, int nlogits,
                        float *hyper_h, int want_streams, float *mixed_h, float *logit) {
    (void)m; (void)ids; (void)S; (void)pos_base; (void)nlogits;
    (void)hyper_h; (void)want_streams; (void)mixed_h; (void)logit;
    g_q38m_declined++;
    return 0;
}

/* device -> host: DN_rec, DN_conv, the PLE ring, K/V/IK up to kv_len. Nothing lives on
 * the device yet. */
static void q38m_sync_host(Model *m) { (void)m; }

/* The host state is authoritative (a restored prefix, a reset): the next forward
 * uploads it again; zero, when it is all zeros. */
static Q38M_UNUSED void q38m_host_wrote(Model *m, int zero) { (void)m; (void)zero; }

/* slot == -1 only (MTP off, one conversation): lower the KV watermarks to len. */
static Q38M_UNUSED void q38m_rollback(Model *m, int slot, int len) { (void)m; (void)slot; (void)len; }

static void q38m_report(void) {
    if (!g_mtl_chain) return;
    fprintf(stderr, "[metal] qwen38 chain: %llu forwards, %llu declined\n", g_q38m_forwards, g_q38m_declined);
}

#endif
