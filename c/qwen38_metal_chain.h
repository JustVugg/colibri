/* The Qwen3.8 dense chain on the Apple GPU (METAL=1 builds, QWEN38_METAL): every
 * layer, the final mixer and lm_head on qwen36's Metal device, through the q36_* and
 * q38_* kernels (qwen36_gpu.h, qwen38_gpu.h), as qwen38_chain.h does on Vulkan and
 * qwen38_cuda_chain.h on CUDA. Included by qwen38_core.h; one conversation only.
 *
 * Startup (qwen38.c, before q38_trunk_cpu_int8): q38m_decide picks the chain or prints
 * the one line that says why the CPU engine runs the model, then q38m_start puts the
 * dense weights, norms and state on the GPU. q38_forward offers every step to
 * q38m_forward; a 0 there means the CPU runs that step from the host state
 * (q38m_cpu_step brings it down first). M3: the routed experts on the GPU from a GPU
 * expert cache (qwen38_metal_experts.h) where the GGUF's expert types allow; M2 (else, and
 * QWEN38_METAL_EXPERTS=0): on the CPU, inside the step (q38_moe_ex), from the router logits
 * the GPU computed.
 *
 * Switches: COLI_METAL=0 the CPU engine; QWEN38_METAL_EXACT=1 (or QWEN36_METAL_EXACT=1)
 * the exact kernels; QWEN38_METAL_EXPERTS=0 the routed experts on the CPU (M2);
 * QWEN38_EXPERT_CACHE_BYTES the GPU expert cache. Debug only (read once, no effect unset):
 * Q38M_CHECK_WEIGHTS, Q38M_LAYER_CHECK (1, 2), Q38M_FORCE_ROUTES, Q38M_DECLINE_EVERY,
 * Q38M_FAIL_SLOT_ALLOC, Q38M_REFUSE_AT, Q38M_FAIL_AT_STEP, Q38M_AHEAD_EARLY, Q38M_AHEAD_FAIL. */
#ifndef QWEN38_METAL_CHAIN_H
#define QWEN38_METAL_CHAIN_H
#include "qwen36_gpu.h"
#include "qwen38_gpu.h"
#include "qwen38_metal_experts.h"   /* M3: the routed experts on the GPU */

static int g_mtl_chain;                        /* 1 when the chain runs steps */
static unsigned long long g_q38m_forwards, g_q38m_declined;

static char g_q38m_off_why[192];               /* the off line's reason, for the METAL frame */
/* serve mode (qwen38.c's serve_loop sets it): a failure with the recurrent state on the GPU only
 * turns the chain off and sets g_q38m_lost instead of exiting; the turn then ends with an ERROR
 * and the conversation starts over on the CPU (qwen38.c, q38_chain_lost_end). Elsewhere it exits. */
static int g_q38m_serve, g_q38m_lost;
static void q38m_off(const char *why) {
    snprintf(g_q38m_off_why, sizeof g_q38m_off_why, "%s", why);
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

/* A BF16 matrix q38_dot_bf16 does not take: more than 65535 outputs (its grid's rows; a
 * checkpoint's BF16 lm_head over a large vocabulary). One with in % 8 != 0 goes up as F32. */
static const Q38Weight *q38m_bf16_too_wide(const Model *m) {
    const Q38Weight *w[Q38M_NW], *t[Q38M_NT];
    for (int i = 0; i < m->c.layers; i++) {
        q38m_layer_srcs(&m->L[i], w);
        for (int j = 0; j < Q38M_NW; j++)
            if (w[j]->kind == Q38_WEIGHT_BF16 && w[j]->cols % 8 == 0 && w[j]->rows > 65535) return w[j];
    }
    q38m_tail_srcs(m, t);
    for (int j = 0; j < Q38M_NT; j++)
        if (t[j]->kind == Q38_WEIGHT_BF16 && t[j]->cols % 8 == 0 && t[j]->rows > 65535) return t[j];
    return NULL;
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

/* A geometry the kernels take: head sizes (q38_attn_sel, q38_idx_prep: 256), the DeltaNet's key
 * size (q36_dn_delta: 128), the PLE ring (q38_ple_conv: 32 a channel), the QSA budget
 * (q38_qsa_select: 2048 blocks), one indexer key head (q38_idx_prep). NULL when it fits. */
static const char *q38m_geometry(const Cfg *c) {
    if (c->idx_kheads != 1) return "idx_kheads != 1";
    if (c->head_dim > 256 || c->idx_dim > 256 || c->rotary_dim % 2 || c->rotary_dim > c->head_dim ||
        c->rotary_dim > c->idx_dim || c->dn_kdim > 128 || c->idx_ratio < 1 || c->idx_budget / c->idx_ratio > 2048 ||
        c->ngram_heads * c->ngram_head_dim != c->ple_dim || (c->ple_convk - 1) * c->ngram_size > 32 ||
        c->q_heads % (c->kv_heads > 0 ? c->kv_heads : 1) || (c->dn_kheads > 0 && c->dn_vheads % c->dn_kheads))
        return "a layer shape the Metal kernels do not take";
    return NULL;
}

/* slots: the serve's conversations (g_q38_mux_slots; 1 outside a serve, whatever KV_SLOTS says) */
static int q38m_decide(Model *m, int slots) {
    const char *why = NULL;
    char buf[96];
    const char *opt = getenv("COLI_METAL");
    const char *mtp = getenv("Q38_MTP");
    const Q38Weight *bad;
    if (opt && *opt && !strcmp(opt, "0")) why = "COLI_METAL=0";
    else if (slots > 1) why = "KV_SLOTS>1 (one conversation on the GPU)";
#ifdef COLI_VULKAN
    /* one GPU backend a process: a build with both takes Metal only where Vulkan is not asked for */
    else if (getenv("COLI_VULKAN") && *getenv("COLI_VULKAN") && strcmp(getenv("COLI_VULKAN"), "0"))
        why = "COLI_VULKAN is on (one GPU backend at a time)";
#endif
    else if (m->vis_ready) why = "vision tower";
    else if ((why = q38m_geometry(&m->c))) {}
    else if (mtp && *mtp && strcmp(mtp, "0")) why = "MTP (Q38_MTP=1) needs the CPU engine until M5";
    else if ((bad = q38m_dense_unsupported(m))) {
        char kind[32];
        snprintf(buf, sizeof buf, "a dense weight of kind %s is not supported", q38m_kind_name(bad, kind, sizeof kind));
        why = buf;
    } else if ((bad = q38m_bf16_too_wide(m))) {
        snprintf(buf, sizeof buf, "a BF16 matrix with %d outputs (the BF16 kernel takes 65535)", bad->rows);
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
static int q38m_setup(Model *m, char *why, size_t nwhy);   /* below */
static int q38m_start(Model *m) {
    const int L = m->c.layers;
    const char *ex = getenv("QWEN38_METAL_EXACT");
    if (ex && !strcmp(ex, "1")) q36_gpu_set_exact(1);   /* before the state's planes are chosen (q36_dn_high_only) */
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
    if (!q38m_setup(m, why, sizeof why)) { q38m_free_weights(); q38m_off(why); return 0; }
    if (!m->c.norm_topk) q38x_off("norm_topk_prob is off (the GPU router's weights are over the top-k)");
    else q38x_start(m);
    fprintf(stderr, "[qwen38-metal] dense weights: %d mapped in place (%.1f MiB), %d uploaded (%.1f MiB%s)\n",
            g_q38m.n_mapped, g_q38m.mapped / 1048576.0, g_q38m.n_uploaded, g_q38m.uploaded / 1048576.0,
            g_q38m.n_f32_from_bf16 ? ", some BF16 as F32" : "");
    const char *chk = getenv("Q38M_CHECK_WEIGHTS");
    if (chk && !strcmp(chk, "1")) q38m_check_weights(m);
    return 1;
}

/* ---- the forward (M2): the trunk on the GPU, the routed experts on the CPU --------------
 * Per step of S rows, in chunks of at most Q38M_CHUNK rows (a chunk sees the K/V rows of
 * the chunks before it, the DeltaNet and PLE state they left, as one step of their rows):
 *   the streams the host embedded (hyper_h) and the PLE rows the host gathers (its n-gram
 *   history stays on the host, q38m_ple_rows) go up; then every layer: the PLE step on its
 *   layer, the gated-residual read, the attention (QSA) or DeltaNet mixer, the gated
 *   residual's apply, the MLP's read, the router logits and the shared expert. Then one sync:
 *   the CPU's q38_moe_ex routes from the GPU's logits and sums the routed experts (routed
 *   only), straight from and into the shared buffers, and the GPU adds the shared expert and
 *   applies the residual. After the last layer the final mixer and lm_head.
 * The CPU's code and the GPU's op for each piece (qwen38_core.h):
 *   q38_gr_read 1815        q38_hc_norm, q38m_mm down, q38_hc_low, q38m_mm up, q38_hc_mix;
 *                           q38m_mm inject, q38_hc_inj
 *   q38_gr_apply 1841       q38_hc_apply
 *   q38_ple 3115            q38m_mm key/value, q38_ple_gate, q38_ple_conv (accumulate: hyper +=)
 *   q38_attention 3473      q38m_mm q/k/v/idx_qk, q38_qgate_split, q38_hc_norm (C 1: rms0 by head,
 *                           qn/kn), q38_rope_half, q38_kv_append (f16 where q38_kv16), q38_idx_prep,
 *                           q38_qsa_pool (the blocks this step completes), q38_qsa_score,
 *                           q38_qsa_select, q38_attn_sel, q38m_mm o
 *   q38_deltanet 3262       q38m_mm qkv/z/b/a; q36_dn_aux twice over identity weights (decay
 *                           from a, beta from b: the projections are the matrices' kinds,
 *                           q36_dn_aux's dot then reads one weight 1 a lane, exact);
 *                           q36_dn_conv, q36_dn_l2, q36_dn_delta (tiled / a_direct = the GGUF's
 *                           g_q38_dn_tiled / g_q38_a_direct); q36_rmsnorm plain (dn_norm),
 *                           q36_gate sigmoid (q38_rmsg's sigmoid_gate 1), q38m_mm out
 *   q38_moe_decode_ex 4972  q38m_mm router; q38m_mm sh_g/sh_u, q36_silu_mul, q38m_mm sh_d,
 *                           q36_shared_gate + q36_scale_rows; the CPU: q38_moe_ex(.., logits,
 *                           routed_only 1) (its own top-k, so the routes are the CPU's rule);
 *                           q36_add (routed + gate x shared, the CPU's order)
 *   q38_forward 5520        final q38_gr_read (no inject), q38m_mm lm_head
 * State: the device holds the DeltaNet states and rings, the PLE ring (`where`: HOST, DEV or
 * BOTH) and the K/V/IK rows and pooled index keys of every attention layer. The host's K, V
 * and IK rows are written after every chunk (the host's caches are always current; the
 * device's below kv_valid are the host's); the recurrent state comes down only when the host
 * needs it (q38m_sync_host). A CPU step (a declined one) or a host that rewrote its state
 * (q38m_host_wrote, q38m_rollback) moves the state's authority to the host: it goes up again
 * before the next chain step. */
#define Q38M_CHUNK 256
enum { Q38M_HOST = 0, Q38M_DEV, Q38M_BOTH };

typedef struct {
    q36_tensor *attn_norm, *mlp_norm;                /* the gated residuals' norms [W] */
    q36_tensor *qn, *kn, *idx_qn, *idx_kn;           /* attention */
    q36_tensor *dn_conv, *alog, *dt, *dn_norm;       /* DeltaNet */
    q36_tensor *sh_gate;                             /* the shared expert's gate [H], NULL: none */
    q36_tensor *ple_prm, *ple_conv;                  /* the PLE layer: norms key|query|conv [3W], conv [W][CK] */
    q36_tensor *state, *ring;                        /* DeltaNet state (dn_planes planes) and conv ring */
    q36_tensor *kc, *vc, *ikc, *pk;                  /* attention: [KVH][dcap][D] (f16 when half), [dcap][ID], [dcap/R][ID] */
    int kv_valid, pk_valid;                          /* device rows (blocks) equal to the host's */
} Q38mL;

static struct {
    int on;
    Q38mL *L;
    q36_tensor *final_norm, *eye, *ple_ring;
    int dn_planes, dcap, half, where, host_zero;
    int rows;   /* the scratch's rows */
    q36_tensor *hyper, *hn, *low, *mix, *mixed, *inj_a, *inj_m, *blk, *routed;
    q36_tensor *qraw, *qs, *gate, *qno, *kraw, *kno, *vraw, *ip, *qidx, *heads, *sel;
    q36_tensor *qkv, *z, *aa, *bb, *decay, *beta, *junk, *convo, *qk, *core, *normo, *gated;
    q36_tensor *emb, *keys, *val, *pg, *pn;
    q36_tensor *lg, *sg, *su, *sh, *shd, *sgate;
    q36_tensor *sc, *logits;
    size_t sc_floats, logit_floats;
    long decline_every;   /* Q38M_DECLINE_EVERY (debug): every k-th step offered is declined */
    int layer_check;      /* Q38M_LAYER_CHECK (debug): the first step against q38_layer_forward;
                           * 2: also the CPU against itself from inputs one ulp away */
    long refuse_at;       /* Q38M_REFUSE_AT (debug): the k-th step offered has its device memory
                           * refused, so the chain turns off mid-run (the off path's test) */
    long fail_at;         /* Q38M_FAIL_AT_STEP (debug): the k-th step offered fails before the first
                           * layer of its last chunk, as a declining kernel does (the failure path's test) */
    int force_routes;     /* Q38M_FORCE_ROUTES (debug, with the layer check): 1 the reference run's
                           * routes, 2 also its QSA selections past the budget */
} g_q38f;

static void q38_layer_forward(Model *m,int i,float *hyper,const int *ids,int S,int pos_base,
                              float *mixed,float *inject,float *block);   /* qwen38_core.h, after this file */

static q36_tensor *q38m_new(size_t floats) { return q36_tensor_new((floats ? floats : 1) * sizeof(float)); }
static q36_tensor *q38m_up(const float *p, size_t floats) {
    return p ? q36_tensor_upload(p, floats * sizeof(float)) : NULL;
}
static float *q38m_f(q36_tensor *t) { return (float *)q36_tensor_data(t); }
static int q38m_selrow(const Cfg *c) { return c->idx_budget + c->idx_ratio; }
static size_t q38m_ple_cells(const Cfg *c) { return (size_t)c->hc_width * (c->ple_convk - 1) * c->ngram_size; }
static int q38m_has_ple(const Cfg *c) { return c->ple_layer >= 0 && c->ple_layer < c->layers; }
static size_t q38m_dn_cells(const Cfg *c) { return (size_t)c->dn_vheads * c->dn_kdim * c->dn_vdim; }
static size_t q38m_conv_cells(const Cfg *c) { return (size_t)c->dn_conv_dim * (c->dn_convk - 1); }

/* a layer's DeltaNet shape for the q36_dn_* kernels */
static q36_shape q38m_dn_shape(const Cfg *c, int rows) {
    q36_shape s;
    memset(&s, 0, sizeof s);
    s.rows = rows; s.hidden = c->hidden;
    s.v_heads = c->dn_vheads; s.k_heads = c->dn_kheads; s.k_dim = c->dn_kdim; s.v_dim = c->dn_vdim;
    s.conv_kernel = c->dn_convk; s.conv_dim = c->dn_conv_dim;
    s.tiled = g_q38_dn_tiled; s.a_direct = g_q38_a_direct;
    return s;
}
static q38_qsa_shape q38m_qsa_shape(const Cfg *c) {
    q38_qsa_shape s = { c->idx_qheads, c->idx_dim, c->idx_ratio, c->idx_budget, c->rotary_dim, c->theta, c->eps };
    return s;
}

static void q38m_free_scratch(void) {
    q36_tensor **t[] = { &g_q38f.hyper, &g_q38f.hn, &g_q38f.low, &g_q38f.mix, &g_q38f.mixed, &g_q38f.inj_a,
        &g_q38f.inj_m, &g_q38f.blk, &g_q38f.routed, &g_q38f.qraw, &g_q38f.qs, &g_q38f.gate, &g_q38f.qno,
        &g_q38f.kraw, &g_q38f.kno, &g_q38f.vraw, &g_q38f.ip, &g_q38f.qidx, &g_q38f.heads, &g_q38f.sel,
        &g_q38f.qkv, &g_q38f.z, &g_q38f.aa, &g_q38f.bb, &g_q38f.decay, &g_q38f.beta, &g_q38f.junk,
        &g_q38f.convo, &g_q38f.qk, &g_q38f.core, &g_q38f.normo, &g_q38f.gated, &g_q38f.emb, &g_q38f.keys,
        &g_q38f.val, &g_q38f.pg, &g_q38f.pn, &g_q38f.lg, &g_q38f.sg, &g_q38f.su, &g_q38f.sh, &g_q38f.shd,
        &g_q38f.sgate };
    for (size_t i = 0; i < sizeof t / sizeof t[0]; i++) { q36_tensor_free(*t[i]); *t[i] = NULL; }
    g_q38f.rows = 0;
}

/* scratch for n rows (the GPU idle: called between chunks, after a sync) */
static int q38m_scratch(const Cfg *c, int n) {
    if (n <= g_q38f.rows) return 1;
    q38m_free_scratch();
    const size_t r = (size_t)n, W = c->hc_width, H = c->hidden, C = c->hc_count, R = c->hc_rank;
    const size_t QD = (size_t)c->q_heads * c->head_dim, KD = (size_t)c->kv_heads * c->head_dim;
    const size_t IP = (size_t)(c->idx_qheads + 1) * c->idx_dim, IQ = (size_t)c->idx_qheads * c->idx_dim;
    const size_t CD = c->dn_conv_dim, V = (size_t)c->dn_vheads * c->dn_vdim, VH = c->dn_vheads;
    const size_t SI = c->shared_inter, E = c->experts, PE = c->ple_dim;
    g_q38f.hyper = q38m_new(r * W); g_q38f.hn = q38m_new(r * W); g_q38f.low = q38m_new(r * R);
    g_q38f.mix = q38m_new(r * W); g_q38f.mixed = q38m_new(r * H); g_q38f.inj_a = q38m_new(r * C);
    g_q38f.inj_m = q38m_new(r * C); g_q38f.blk = q38m_new(r * H); g_q38f.routed = q38m_new(r * H);
    g_q38f.qraw = q38m_new(r * 2 * QD); g_q38f.qs = q38m_new(r * QD); g_q38f.gate = q38m_new(r * QD);
    g_q38f.qno = q38m_new(r * QD); g_q38f.kraw = q38m_new(r * KD); g_q38f.kno = q38m_new(r * KD);
    g_q38f.vraw = q38m_new(r * KD); g_q38f.ip = q38m_new(r * IP); g_q38f.qidx = q38m_new(r * IQ);
    g_q38f.heads = q38m_new(r * QD); g_q38f.sel = q38m_new(r * (size_t)q38m_selrow(c));
    g_q38f.qkv = q38m_new(r * CD); g_q38f.z = q38m_new(r * V); g_q38f.aa = q38m_new(r * VH);
    g_q38f.bb = q38m_new(r * VH); g_q38f.decay = q38m_new(r * VH); g_q38f.beta = q38m_new(r * VH);
    g_q38f.junk = q38m_new(r * VH); g_q38f.convo = q38m_new(r * CD); g_q38f.qk = q38m_new(r * CD);
    g_q38f.core = q38m_new(r * V); g_q38f.normo = q38m_new(r * V); g_q38f.gated = q38m_new(r * V);
    g_q38f.emb = q38m_new(r * PE); g_q38f.keys = q38m_new(r * W); g_q38f.val = q38m_new(r * H);
    g_q38f.pg = q38m_new(r * W); g_q38f.pn = q38m_new(r * W);
    g_q38f.lg = q38m_new(r * E); g_q38f.sg = q38m_new(r * SI); g_q38f.su = q38m_new(r * SI);
    g_q38f.sh = q38m_new(r * SI); g_q38f.shd = q38m_new(r * H); g_q38f.sgate = q38m_new(r);
    q36_tensor *all[] = { g_q38f.hyper, g_q38f.hn, g_q38f.low, g_q38f.mix, g_q38f.mixed, g_q38f.inj_a, g_q38f.inj_m,
        g_q38f.blk, g_q38f.routed, g_q38f.qraw, g_q38f.qs, g_q38f.gate, g_q38f.qno, g_q38f.kraw, g_q38f.kno,
        g_q38f.vraw, g_q38f.ip, g_q38f.qidx, g_q38f.heads, g_q38f.sel, g_q38f.qkv, g_q38f.z, g_q38f.aa, g_q38f.bb,
        g_q38f.decay, g_q38f.beta, g_q38f.junk, g_q38f.convo, g_q38f.qk, g_q38f.core, g_q38f.normo, g_q38f.gated,
        g_q38f.emb, g_q38f.keys, g_q38f.val, g_q38f.pg, g_q38f.pn, g_q38f.lg, g_q38f.sg, g_q38f.su, g_q38f.sh,
        g_q38f.shd, g_q38f.sgate };
    for (size_t i = 0; i < sizeof all / sizeof all[0]; i++)
        if (!all[i]) { q38m_free_scratch(); return 0; }
    g_q38f.rows = n;
    return 1;
}

/* a buffer of at least `floats` floats in *t (cap in *have); contents not kept */
static int q38m_grow(q36_tensor **t, size_t *have, size_t floats) {
    if (*t && *have >= floats) return 1;
    q36_tensor_free(*t);
    *t = q38m_new(floats);
    *have = *t ? floats : 0;
    return *t != NULL;
}

/* The device's K/V/IK rows for `need` positions: 256, doubled to 1024, then 1024 more at a
 * time (qwen36's qg_reserve), the GPU idle. The device's rows are dropped (kv_valid = pk_valid =
 * 0): q38m_push uploads them again from the host's K/V/IK, which are always current, and
 * q38m_attention pools the index keys again from them (q38_qsa_pool from block 0). */
static unsigned long long g_q38m_kv_grown, g_q38m_kv_regrow, g_q38m_kv_regrow_rows;   /* growths; those with valid rows, the rows */
static int q38m_kv_reserve(Model *m, int need) {
    const Cfg *c = &m->c;
    if (need <= g_q38f.dcap) return 1;
    int cap = g_q38f.dcap ? g_q38f.dcap : 256;
    while (cap < need) cap = cap < 1024 ? 2 * cap : cap + 1024;
    const size_t D = c->head_dim, KVH = c->kv_heads, ID = c->idx_dim, R = c->idx_ratio, es = g_q38f.half ? 2 : 4;
    /* every layer's new buffers first: a refusal leaves every layer as it was */
    q36_tensor **nb = (q36_tensor **)calloc((size_t)c->layers * 4, sizeof *nb);
    int ok = nb != NULL;
    for (int i = 0; ok && i < c->layers; i++) {
        if (!c->is_attn[i]) continue;
        q36_tensor **t = nb + (size_t)i * 4;
        ok = (t[0] = q36_tensor_new(KVH * cap * D * es)) && (t[1] = q36_tensor_new(KVH * cap * D * es)) &&
             (t[2] = q38m_new((size_t)cap * ID)) && (t[3] = q38m_new(((size_t)cap / R + 1) * ID));
    }
    if (!ok) {
        if (nb) for (size_t k = 0; k < (size_t)c->layers * 4; k++) q36_tensor_free(nb[k]);
        free(nb);
        return 0;
    }
    int dropped = 0;
    for (int i = 0; i < c->layers; i++) {
        if (!c->is_attn[i]) continue;
        Q38mL *L = &g_q38f.L[i];
        q36_tensor *kc = nb[(size_t)i * 4], *vc = nb[(size_t)i * 4 + 1], *ikc = nb[(size_t)i * 4 + 2], *pk = nb[(size_t)i * 4 + 3];
        if (L->kv_valid > dropped) dropped = L->kv_valid;
        L->kv_valid = L->pk_valid = 0;
        q36_tensor_free(L->kc); q36_tensor_free(L->vc); q36_tensor_free(L->ikc); q36_tensor_free(L->pk);
        L->kc = kc; L->vc = vc; L->ikc = ikc; L->pk = pk;
    }
    free(nb);
    g_q38f.dcap = cap;
    g_q38m_kv_grown++;
    if (dropped > 0) { g_q38m_kv_regrow++; g_q38m_kv_regrow_rows += (unsigned long long)dropped; }
    return 1;
}

/* f32 <-> the device cache's element (f16 when half: the host's rows are kv16-rounded
 * already, so both ways are exact) */
static void q38m_kv_put(void *dst, const float *src, size_t n) {
    if (!g_q38f.half) { memcpy(dst, src, n * sizeof(float)); return; }
    _Float16 *d = (_Float16 *)dst;
    for (size_t i = 0; i < n; i++) d[i] = (_Float16)src[i];
}
static void q38m_kv_get(float *dst, const void *src, size_t n) {
    if (!g_q38f.half) { memcpy(dst, src, n * sizeof(float)); return; }
    const _Float16 *s = (const _Float16 *)src;
    for (size_t i = 0; i < n; i++) dst[i] = (float)s[i];
}

/* rows [t0, t1) of attention layer i, host -> device (up) or device -> host */
static void q38m_kv_rows(Model *m, int i, int t0, int t1, int up) {
    const Cfg *c = &m->c;
    const size_t D = c->head_dim, ID = c->idx_dim, es = g_q38f.half ? 2 : 4;
    Q38mL *L = &g_q38f.L[i];
    if (t1 <= t0) return;
    for (int h = 0; h < c->kv_heads; h++) {
        size_t hs = ((size_t)h * m->kv_cap + t0) * D, ds = ((size_t)h * g_q38f.dcap + t0) * D;
        char *kd = (char *)q36_tensor_data(L->kc) + ds * es, *vd = (char *)q36_tensor_data(L->vc) + ds * es;
        size_t n = (size_t)(t1 - t0) * D;
        if (up) { q38m_kv_put(kd, m->K[i] + hs, n); q38m_kv_put(vd, m->V[i] + hs, n); }
        else { q38m_kv_get(m->K[i] + hs, kd, n); q38m_kv_get(m->V[i] + hs, vd, n); }
    }
    float *ik = q38m_f(L->ikc) + (size_t)t0 * ID, *hk = m->IK[i] + (size_t)t0 * ID;
    if (up) memcpy(ik, hk, (size_t)(t1 - t0) * ID * sizeof(float));
    else memcpy(hk, ik, (size_t)(t1 - t0) * ID * sizeof(float));
}

/* Before a chunk of rows from pb (the GPU idle): the recurrent state up where the host holds
 * it, and every attention layer's rows below pb that the device does not have. */
static void q38m_push(Model *m, int pb) {
    const Cfg *c = &m->c;
    const int R = c->idx_ratio;
    if (g_q38f.where == Q38M_HOST) {
        const size_t nr = q38m_dn_cells(c), nc = q38m_conv_cells(c);
        for (int i = 0; i < c->layers; i++) {
            if (c->is_attn[i]) continue;
            Q38mL *L = &g_q38f.L[i];
            memset(q36_tensor_data(L->state), 0, (size_t)g_q38f.dn_planes * nr * sizeof(float));
            if (g_q38f.host_zero) memset(q36_tensor_data(L->ring), 0, nc * sizeof(float));
            else {
                memcpy(q36_tensor_data(L->state), m->DN_rec[i], nr * sizeof(float));
                memcpy(q36_tensor_data(L->ring), m->DN_conv[i], nc * sizeof(float));
            }
        }
        if (g_q38f.ple_ring) {
            if (g_q38f.host_zero || !m->PLE_conv_state) memset(q36_tensor_data(g_q38f.ple_ring), 0, q38m_ple_cells(c) * sizeof(float));
            else memcpy(q36_tensor_data(g_q38f.ple_ring), m->PLE_conv_state, q38m_ple_cells(c) * sizeof(float));
        }
        g_q38f.where = Q38M_BOTH;
    }
    for (int i = 0; i < c->layers; i++) {
        if (!c->is_attn[i]) continue;
        Q38mL *L = &g_q38f.L[i];
        if (L->pk_valid > pb / R) L->pk_valid = pb / R;   /* the blocks this step rewrites */
        if (L->kv_valid < pb) {
            q38m_kv_rows(m, i, L->kv_valid, pb, 1);
            if (L->pk_valid > L->kv_valid / R) L->pk_valid = L->kv_valid / R;
            L->kv_valid = pb;
        }
    }
}

/* device -> host: DN_rec (the high plane plus the low), DN_conv, the PLE ring. The K/V/IK
 * rows are the host's already (written after every chunk). */
static void q38m_sync_host(Model *m) {
    if (!g_q38f.on || g_q38f.where != Q38M_DEV) return;
    const Cfg *c = &m->c;
    const size_t nr = q38m_dn_cells(c), nc = q38m_conv_cells(c);
    for (int i = 0; i < c->layers; i++) {
        if (c->is_attn[i]) continue;
        const float *st = q38m_f(g_q38f.L[i].state);
        if (g_q38f.dn_planes == 2) for (size_t k = 0; k < nr; k++) m->DN_rec[i][k] = st[k] + st[nr + k];
        else memcpy(m->DN_rec[i], st, nr * sizeof(float));
        memcpy(m->DN_conv[i], q36_tensor_data(g_q38f.L[i].ring), nc * sizeof(float));
    }
    if (g_q38f.ple_ring && m->PLE_conv_state)
        memcpy(m->PLE_conv_state, q36_tensor_data(g_q38f.ple_ring), q38m_ple_cells(c) * sizeof(float));
    g_q38f.where = Q38M_BOTH;
}

/* The host state is authoritative (a restored prefix, a reset): the next forward uploads it
 * again; zero, when it is all zeros. */
static void q38m_host_wrote(Model *m, int zero) {
    (void)m;
    if (!g_q38f.on) return;
    g_q38f.where = Q38M_HOST; g_q38f.host_zero = zero;
}

/* A CPU step from pos_base (q38_forward, a declined step): the host's state current first,
 * then the host's; the device's rows from pos_base on are the CPU's to write. */
static void q38m_cpu_step(Model *m, int pos_base) {
    if (!g_q38f.on) return;
    q38m_sync_host(m);
    g_q38f.where = Q38M_HOST; g_q38f.host_zero = 0;
    const Cfg *c = &m->c;
    for (int i = 0; i < c->layers; i++) {
        if (!c->is_attn[i]) continue;
        Q38mL *L = &g_q38f.L[i];
        if (L->kv_valid > pos_base) L->kv_valid = pos_base;
        if (L->pk_valid > pos_base / c->idx_ratio) L->pk_valid = pos_base / c->idx_ratio;
    }
}

/* len positions stand: the KV and pooled-key watermarks come down to them (the rows past len
 * go up again from the host). slot >= 0 (a speculative verify's rollback, which the CPU ran:
 * the chain declines a verify) swaps the host's state in: the host's again. */
static void q38m_rollback(Model *m, int slot, int len) {
    if (!g_q38f.on) return;
    const Cfg *c = &m->c;
    for (int i = 0; i < c->layers; i++) {
        if (!c->is_attn[i]) continue;
        Q38mL *L = &g_q38f.L[i];
        if (L->kv_valid > len) L->kv_valid = len;
        if (L->pk_valid > len / c->idx_ratio) L->pk_valid = len / c->idx_ratio;
    }
    if (slot >= 0) { q38m_sync_host(m); g_q38f.where = Q38M_HOST; g_q38f.host_zero = 0; }
}

static void q38m_free_forward(const Cfg *c) {
    if (g_q38f.L)
        for (int i = 0; i < c->layers; i++) {
            Q38mL *L = &g_q38f.L[i];
            q36_tensor *t[] = { L->attn_norm, L->mlp_norm, L->qn, L->kn, L->idx_qn, L->idx_kn, L->dn_conv, L->alog,
                                L->dt, L->dn_norm, L->sh_gate, L->ple_prm, L->ple_conv, L->state, L->ring, L->kc,
                                L->vc, L->ikc, L->pk };
            for (size_t k = 0; k < sizeof t / sizeof t[0]; k++) q36_tensor_free(t[k]);
        }
    free(g_q38f.L);
    q36_tensor_free(g_q38f.final_norm); q36_tensor_free(g_q38f.eye); q36_tensor_free(g_q38f.ple_ring);
    q36_tensor_free(g_q38f.sc); q36_tensor_free(g_q38f.logits);
    q38m_free_scratch();
    memset(&g_q38f, 0, sizeof g_q38f);
}

/* the chain's whole device state, before q38_model_free (the in-place maps read the model's
 * bytes) */
static void q38m_free(Model *m) {
    if (!g_q38f.on && !g_q38m.layers) return;
    q36_gpu_sync();
    q38x_free();
    q38m_free_forward(&m->c);
    q38m_free_weights();
    g_mtl_chain = 0;
}

/* the norms, small vectors and state of every layer on the GPU; 0 with the reason in why */
static int q38m_setup(Model *m, char *why, size_t nwhy) {
    const Cfg *c = &m->c;
    const int W = c->hc_width, H = c->hidden, VH = c->dn_vheads;
    memset(&g_q38f, 0, sizeof g_q38f);
    g_q38f.L = (Q38mL *)calloc((size_t)c->layers, sizeof *g_q38f.L);
    if (!g_q38f.L) { snprintf(why, nwhy, "no memory for the layers' table"); return 0; }
    q36_shape ds = q38m_dn_shape(c, 1);
    g_q38f.dn_planes = q36_dn_high_only(&ds) ? 1 : 2;
    g_q38f.half = q38_kv16();
    g_q38f.where = Q38M_HOST; g_q38f.host_zero = 1;
    int ok = (g_q38f.final_norm = q38m_up(m->final_gr.norm, (size_t)W)) != NULL;
    if (ok && VH > 0) {   /* q36_dn_aux over a and b already projected: identity weights */
        float *e = (float *)calloc((size_t)VH * VH, sizeof(float));
        ok = e != NULL;
        if (ok) { for (int h = 0; h < VH; h++) e[(size_t)h * VH + h] = 1.f; g_q38f.eye = q38m_up(e, (size_t)VH * VH); ok = g_q38f.eye != NULL; }
        free(e);
    }
    if (ok && q38m_has_ple(c)) {
        Layer *l = &m->L[c->ple_layer];
        Q38mL *L = &g_q38f.L[c->ple_layer];
        float *prm = (float *)malloc((size_t)3 * W * sizeof(float));
        ok = prm && l->ple_norm_key && l->ple_norm_query && l->ple_norm_conv && l->ple_conv;
        if (ok) {
            memcpy(prm, l->ple_norm_key, (size_t)W * sizeof(float));
            memcpy(prm + W, l->ple_norm_query, (size_t)W * sizeof(float));
            memcpy(prm + 2 * W, l->ple_norm_conv, (size_t)W * sizeof(float));
            ok = (L->ple_prm = q38m_up(prm, (size_t)3 * W)) && (L->ple_conv = q38m_up(l->ple_conv, (size_t)W * c->ple_convk)) &&
                 (g_q38f.ple_ring = q38m_new(q38m_ple_cells(c)));
        }
        free(prm);
    }
    for (int i = 0; ok && i < c->layers; i++) {
        Layer *l = &m->L[i];
        Q38mL *L = &g_q38f.L[i];
        ok = (L->attn_norm = q38m_up(l->attn_gr.norm, (size_t)W)) && (L->mlp_norm = q38m_up(l->mlp_gr.norm, (size_t)W));
        if (ok && l->sh_gate) ok = (L->sh_gate = q38m_up(l->sh_gate, (size_t)H)) != NULL;
        if (!ok) break;
        if (c->is_attn[i])
            ok = (L->qn = q38m_up(l->qn, (size_t)c->head_dim)) && (L->kn = q38m_up(l->kn, (size_t)c->head_dim)) &&
                 (L->idx_qn = q38m_up(l->idx_qn, (size_t)c->idx_dim)) && (L->idx_kn = q38m_up(l->idx_kn, (size_t)c->idx_dim));
        else
            ok = (L->dn_conv = q38m_up(l->dn_conv, (size_t)c->dn_conv_dim * c->dn_convk)) &&
                 (L->alog = q38m_up(l->dn_alog, (size_t)VH)) && (L->dt = q38m_up(l->dn_dtbias, (size_t)VH)) &&
                 (L->dn_norm = q38m_up(l->dn_norm, (size_t)c->dn_vdim)) &&
                 (L->state = q38m_new((size_t)g_q38f.dn_planes * q38m_dn_cells(c))) &&
                 (L->ring = q38m_new(q38m_conv_cells(c)));
    }
    if (!ok) { q38m_free_forward(c); snprintf(why, nwhy, "the layers' norms and state did not reach the GPU"); return 0; }
    const char *d = getenv("Q38M_DECLINE_EVERY"), *lc = getenv("Q38M_LAYER_CHECK");
    g_q38f.decline_every = d && *d ? atol(d) : 0;
    const char *ra = getenv("Q38M_REFUSE_AT");
    g_q38f.refuse_at = ra && *ra ? atol(ra) : 0;
    const char *fa = getenv("Q38M_FAIL_AT_STEP");
    g_q38f.fail_at = fa && *fa ? atol(fa) : 0;
    g_q38f.layer_check = lc && (!strcmp(lc, "1") || !strcmp(lc, "2")) ? atoi(lc) : 0;
    const char *fr = getenv("Q38M_FORCE_ROUTES");
    g_q38f.force_routes = fr && (!strcmp(fr, "1") || !strcmp(fr, "2")) ? atoi(fr) : 0;
    g_q38f.on = 1;
    return 1;
}

/* the n-gram rows of the chunk's rows [c0, c0+n) into emb, the host's history moved on, as
 * q38_ple does it (qwen38_chain.h's q38c_ple_rows) */
static void q38m_ple_rows(Model *m, const int *ids, int c0, int n, float *emb) {
    const Cfg *c = &m->c;
    const int E = c->ngram_heads * c->ngram_head_dim;
    for (int r = 0; r < n; r++) {
        int s = c0 + r;
        float *e = emb + (size_t)r * E;
        int64_t p1 = m->ple_history_len >= 1 ? m->ple_history[m->ple_history_len - 1] : c->eos_id;
        int64_t p2 = m->ple_history_len >= 2 ? m->ple_history[m->ple_history_len - 2] : c->eos_id;
        if (m->ple_pref && s < m->ple_pref_rows)
            memcpy(e, m->ple_pref + (int64_t)s * E, (size_t)E * sizeof(float));
        else for (int h = 0; h < c->ngram_heads; h++) {
            int ng = h < c->heads_per_ngram ? 2 : 3;
            q38_ple_row(m, q38_hash_row(m, h, ng, ids[s], p1, p2), e + (int64_t)h * c->ngram_head_dim);
        }
        if (ids[s] == c->eos_id) m->ple_history_len = 0;
        else if (m->ple_history_len == 0) { m->ple_history[0] = ids[s]; m->ple_history_len = 1; }
        else if (m->ple_history_len == 1) { m->ple_history[1] = ids[s]; m->ple_history_len = 2; }
        else { m->ple_history[0] = m->ple_history[1]; m->ple_history[1] = ids[s]; }
    }
}

/* q38_gr_read: mixed [n][H] from the streams, inject [n][C] when inj */
static int q38m_gr_read(const Cfg *c, const Q38mW *w /* down, up, inject */, q36_tensor *norm, int n, q36_tensor *inj) {
    const int H = c->hidden, C = c->hc_count, R = c->hc_rank;
    int ok = q38_hc_norm(g_q38f.hn, g_q38f.hyper, norm, n, H, C, c->eps) && q38m_mm(g_q38f.low, g_q38f.hn, &w[0], n) &&
             q38_hc_low(g_q38f.low, g_q38f.low, n * R, (float)C) && q38m_mm(g_q38f.mix, g_q38f.low, &w[1], n) &&
             q38_hc_mix(g_q38f.mixed, g_q38f.mix, g_q38f.hn, n, H, C, (float)C);
    if (ok && inj) ok = q38m_mm(inj, g_q38f.hn, &w[2], n) && q38_hc_inj(inj, inj, n * C, (float)C);
    return ok;
}

/* Q38M_FORCE_ROUTES=2 (layer check): the chunk's rows past the budget take the reference run's
 * selection (the GPU's batch flushed first) */
static int g_q38m_lc_qforced, g_q38m_lc_qset, g_q38m_lc_qswap, g_q38m_lc_qrows;   /* qrows: the rows past the budget held */
static double g_q38m_lc_qgap;   /* the largest |score| gap between two blocks the two selections swapped */
static int q38m_lc_force_sel(int i, int n, int pb);   /* below, beside the layer check */

/* q38_attention over mixed into blk, rows at pb.. */
static int q38m_attention(Model *m, int i, int n, int pb) {
    const Cfg *c = &m->c;
    const Q38mW *w = g_q38m.L[i];
    Q38mL *L = &g_q38f.L[i];
    const int QH = c->q_heads, KVH = c->kv_heads, D = c->head_dim, R = c->idx_ratio, selrow = q38m_selrow(c);
    const float theta = c->theta;   /* the MTP head (mtp_theta) stays on the CPU, off beside the chain */
    q38_qsa_shape qs = q38m_qsa_shape(c);
    int ok = q38m_mm(g_q38f.qraw, g_q38f.mixed, &w[10], n) && q38m_mm(g_q38f.kraw, g_q38f.mixed, &w[11], n) &&
             q38m_mm(g_q38f.vraw, g_q38f.mixed, &w[12], n) && q38m_mm(g_q38f.ip, g_q38f.mixed, &w[14], n) &&
             q38_qgate_split(g_q38f.qs, g_q38f.gate, g_q38f.qraw, n, QH, D) &&
             q38_hc_norm(g_q38f.qno, g_q38f.qs, L->qn, n * QH, D, 1, c->eps) &&
             q38_hc_norm(g_q38f.kno, g_q38f.kraw, L->kn, n * KVH, D, 1, c->eps) &&
             q38_rope_half(g_q38f.qno, n, QH, D, c->rotary_dim, pb, theta) &&
             q38_rope_half(g_q38f.kno, n, KVH, D, c->rotary_dim, pb, theta) &&
             q38_kv_append(L->kc, L->vc, g_q38f.kno, g_q38f.vraw, n, pb, KVH, D, g_q38f.dcap, g_q38f.half) &&
             q38_idx_prep(g_q38f.qidx, L->ikc, g_q38f.ip, L->idx_qn, n, pb, g_q38f.dcap, &qs);
    const int blocks = (pb + n) / R, nbmax = blocks > 0 ? blocks : 1;
    if (ok && blocks > L->pk_valid) {   /* the blocks this chunk completes */
        ok = q38_qsa_pool(L->pk, L->ikc, L->idx_kn, L->pk_valid, blocks - L->pk_valid, &qs);
        if (ok) L->pk_valid = blocks;
    }
    ok = ok && q38m_grow(&g_q38f.sc, &g_q38f.sc_floats, (size_t)n * nbmax) &&
         q38_qsa_score(g_q38f.sc, g_q38f.qidx, L->pk, n, pb, nbmax, &qs) &&
         q38_qsa_select(g_q38f.sel, g_q38f.sc, n, pb, nbmax, selrow, &qs) &&
         q38m_lc_force_sel(i, n, pb) &&
         q38_attn_sel(g_q38f.heads, g_q38f.qno, g_q38f.gate, L->kc, L->vc, g_q38f.sel, n, pb, QH, KVH, D, g_q38f.dcap,
                      selrow, g_q38f.half) &&
         q38m_mm(g_q38f.blk, g_q38f.heads, &w[13], n);
    return ok;
}

/* q38_deltanet over mixed into blk */
static int q38m_deltanet(Model *m, int i, int n) {
    const Cfg *c = &m->c;
    const Q38mW *w = g_q38m.L[i];
    Q38mL *L = &g_q38f.L[i];
    q36_shape s = q38m_dn_shape(c, n), sa = s;
    sa.hidden = c->dn_vheads;   /* q36_dn_aux over the projections, identity weights */
    const int VH = c->dn_vheads, VD = c->dn_vdim;
    return q38m_mm(g_q38f.qkv, g_q38f.mixed, &w[15], n) && q38m_mm(g_q38f.z, g_q38f.mixed, &w[16], n) &&
           q38m_mm(g_q38f.bb, g_q38f.mixed, &w[17], n) && q38m_mm(g_q38f.aa, g_q38f.mixed, &w[18], n) &&
           q36_dn_aux(g_q38f.decay, g_q38f.junk, g_q38f.aa, g_q38f.eye, g_q38f.eye, L->alog, L->dt, &sa) &&
           q36_dn_aux(g_q38f.junk, g_q38f.beta, g_q38f.bb, g_q38f.eye, g_q38f.eye, L->alog, L->dt, &sa) &&
           q36_dn_conv(g_q38f.convo, L->ring, g_q38f.qkv, L->dn_conv, NULL, &s) &&
           q36_dn_l2(g_q38f.qk, g_q38f.convo, &s) &&
           q36_dn_delta(g_q38f.core, L->state, g_q38f.qk, g_q38f.convo, g_q38f.decay, g_q38f.beta, NULL, &s) &&
           q36_rmsnorm(g_q38f.normo, g_q38f.core, L->dn_norm, n * VH, VD, c->eps, 1) &&
           q36_gate(g_q38f.gated, g_q38f.normo, g_q38f.z, n * VH * VD, 0) &&
           q38m_mm(g_q38f.blk, g_q38f.gated, &w[19], n);
}

/* the PLE step on its layer: hyper += gated + silu(conv) (q38_ple, then hyper += ple) */
static int q38m_ple(Model *m, int n) {
    const Cfg *c = &m->c;
    const Q38mW *w = g_q38m.L[c->ple_layer];
    Q38mL *L = &g_q38f.L[c->ple_layer];
    q38_ple_shape ps = { n, c->hc_count, c->hc_width, c->hidden, c->ple_convk, c->ngram_size, 0, c->eps, 1 };
    return q38m_mm(g_q38f.keys, g_q38f.emb, &w[20], n) && q38m_mm(g_q38f.val, g_q38f.emb, &w[21], n) &&
           q38_ple_gate(g_q38f.pg, g_q38f.pn, g_q38f.keys, g_q38f.hyper, g_q38f.val, L->ple_prm, &ps) &&
           q38_ple_conv(g_q38f.hyper, g_q38f.ple_ring, NULL, g_q38f.pg, g_q38f.pn, L->ple_conv, &ps);
}

/* the shared expert (gate x shared in shd), when the layer has one */
static int q38m_has_shared(const Model *m, int i) { return m->c.shared_inter > 0 && g_q38m.L[i][7].t; }
static int q38m_mlp_shared(Model *m, int i, int n) {
    const Cfg *c = &m->c;
    const Q38mW *w = g_q38m.L[i];
    Q38mL *L = &g_q38f.L[i];
    if (!q38m_has_shared(m, i)) return 1;
    return q38m_mm(g_q38f.sg, g_q38f.mixed, &w[7], n) && q38m_mm(g_q38f.su, g_q38f.mixed, &w[8], n) &&
           q36_silu_mul(g_q38f.sh, g_q38f.sg, g_q38f.su, n * c->shared_inter) &&
           q38m_mm(g_q38f.shd, g_q38f.sh, &w[9], n) &&
           q36_shared_gate(g_q38f.sgate, g_q38f.mixed, L->sh_gate, n, c->hidden) &&
           q36_scale_rows(g_q38f.shd, g_q38f.sgate, n, c->hidden);
}

/* router logits (and, with the GPU experts, the GPU's routes); M2: the shared expert too, before
 * the sync (with the GPU experts it runs while their reads do, q38x_moe) */
static int q38m_mlp_dense(Model *m, int i, int n) {
    int ok = q38m_mm(g_q38f.lg, g_q38f.mixed, &g_q38m.L[i][6], n);
    if (g_q38x.on) {
        ok = ok && q38x_route(g_q38f.lg, n);
        /* read-ahead (decode): the next layer's router on this row (qwen38_metal_experts.h); advice
         * only, so a router that declines turns read-ahead off and the step goes on */
        if (ok && q38x_ahead_wanted(i, n) &&
            !(q38m_mm(g_q38x.alg, g_q38f.mixed, &g_q38m.L[i + 1][6], 1) && q38x_ahead_route(i)))
            q38x_ahead_off("its router did not run on the GPU");
        return ok;
    }
    return ok && q38m_mlp_shared(m, i, n);
}

/* Q38M_LAYER_CHECK: the CPU's streams after each layer for the first step (q38_layer_forward on
 * a copy of the embedded rows, the PLE history and prefetched rows put back after), and the
 * worst |gpu - cpu| and max |cpu| of each layer.
 * The routes too: a top-k of 512 experts has rows whose k-th and (k+1)-th router logits are
 * a few ulps apart, and two f32 computations of the same row (the CPU's, the GPU's) can pick
 * different experts there; the rest of the step then carries that row's difference forward
 * (DeltaNet state, attention), far over 1e-4 at long prompts. Each layer's line says how many
 * rows the GPU routed unlike the reference run, unlike the CPU's router over the GPU's own
 * row, and the smallest k-th/(k+1)-th margin. Q38M_LAYER_CHECK=2 also runs the CPU step again
 * from the embedded rows one ulp away (`[q38m-cpu-ulp]` lines): the CPU's own sensitivity,
 * the floor the chain is measured against (Plan 2 Task 7a). */
typedef struct { double diff, max, margin, diff_lo, diff_hi, qmargin, chunk[64]; int wrow, mrow, off_ref, off_cpu, forced, qrow, qzero, gpu_off; } Q38mLc;
static float *g_q38m_lc_ref;
static int *g_q38m_lc_idx, g_q38m_lc_S;   /* the reference run's routes [layer][S][topk] */
static Q38mLc *g_q38m_lc;
static int **g_q38m_lc_sel;   /* Q38M_FORCE_ROUTES=2: the reference run's selections [layer][S][selrow], the chain's format */
static int g_q38m_lc_selrow, g_q38m_lc_take, g_q38m_lc_R;

/* g_q38_sel_hook: the rows past the budget as [count, positions], the others [-1] (the chain's own) */
static void q38m_lc_sel_hook(int layer, int pos_base, int S, const int *sel, const int *nsel, int maxsel) {
    if (!g_q38m_lc_sel || S != g_q38m_lc_S) return;
    int *d = g_q38m_lc_sel[layer] = (int *)malloc((size_t)S * g_q38m_lc_selrow * sizeof(int));
    if (!d) { fprintf(stderr, "[q38m-layer] OOM\n"); exit(1); }
    for (int s = 0; s < S; s++) {
        int *row = d + (size_t)s * g_q38m_lc_selrow;
        if ((pos_base + s + 1) / g_q38m_lc_R <= g_q38m_lc_take || nsel[s] > g_q38m_lc_selrow - 1) { row[0] = -1; continue; }
        row[0] = nsel[s];
        memcpy(row + 1, sel + (size_t)s * maxsel, (size_t)nsel[s] * sizeof(int));
    }
}

static void q38m_lc_topk(const float *lg, int E, int K, int *idx) {   /* q38_moe_decode_ex's rule */
    for (int z = 0; z < K; z++) {
        int best = -1; float bv = -1.f;
        for (int e = 0; e < E; e++) {
            int used = 0;
            for (int j = 0; j < z; j++) used |= idx[j] == e;
            if (!used && (best < 0 || lg[e] > bv)) { bv = lg[e]; best = e; }
        }
        idx[z] = best;
    }
}
static int q38m_lc_same(const int *a, const int *b, int K) {   /* the same set */
    for (int z = 0; z < K; z++) {
        int f = 0;
        for (int y = 0; y < K; y++) f |= a[z] == b[y];
        if (!f) return 0;
    }
    return 1;
}
/* the CPU's routes of S rows of layer i from their mixed rows into idx [S][topk] */
static void q38m_lc_route_rows(Model *m, int i, const float *mixed, int S, int *idx) {
    const Cfg *c = &m->c;
    float *lg = falloc(c->experts);
    for (int r = 0; r < S; r++) {
        q38_dense_matmul(m, lg, mixed + (size_t)r * c->hidden, &m->L[i].router, 1, c->hidden, c->experts);
        q38m_lc_topk(lg, c->experts, c->topk, idx + (size_t)r * c->topk);
    }
    free(lg);
}
/* the recurrent state zero, as the step's start (pos_base 0) had it */
static void q38m_lc_zero_state(Model *m) {
    const Cfg *c = &m->c;
    for (int i = 0; i < c->layers; i++) {
        if (c->is_attn[i]) continue;
        memset(m->DN_rec[i], 0, q38m_dn_cells(c) * sizeof(float));
        memset(m->DN_conv[i], 0, q38m_conv_cells(c) * sizeof(float));
    }
    if (m->PLE_conv_state) memset(m->PLE_conv_state, 0, q38m_ple_cells(c) * sizeof(float));
}
static void q38m_layer_check_ref(Model *m, const int *ids, int S, int pos_base, const float *hyper_h) {
    const Cfg *c = &m->c;
    const size_t SW = (size_t)S * c->hc_width;
    const int K = c->topk;
    g_q38m_lc_S = S;
    g_q38m_lc_ref = (float *)malloc((size_t)c->layers * SW * sizeof(float));
    g_q38m_lc = (Q38mLc *)calloc((size_t)c->layers, sizeof *g_q38m_lc);
    g_q38m_lc_idx = (int *)malloc((size_t)c->layers * S * K * sizeof(int));
    int *idx = (int *)malloc((size_t)S * K * sizeof(int));
    float *hy = (float *)malloc(SW * sizeof(float)), *mixed = falloc((int64_t)S * c->hidden);
    float *inject = falloc((int64_t)S * c->hc_count), *block = falloc((int64_t)S * c->hidden);
    if (!g_q38m_lc_ref || !g_q38m_lc || !g_q38m_lc_idx || !idx || !hy) { fprintf(stderr, "[q38m-layer] OOM\n"); exit(1); }
    for (int i = 0; i < c->layers; i++) g_q38m_lc[i].margin = g_q38m_lc[i].qmargin = INFINITY;
    if (g_q38f.force_routes == 2) {
        g_q38m_lc_sel = (int **)calloc((size_t)c->layers, sizeof *g_q38m_lc_sel);
        if (!g_q38m_lc_sel) { fprintf(stderr, "[q38m-layer] OOM\n"); exit(1); }
        g_q38m_lc_selrow = q38m_selrow(c); g_q38m_lc_take = c->idx_budget / c->idx_ratio; g_q38m_lc_R = c->idx_ratio;
        g_q38_sel_hook = q38m_lc_sel_hook;
    }
    memcpy(hy, hyper_h, SW * sizeof(float));
    int64_t hist[2] = { m->ple_history[0], m->ple_history[1] };
    int hlen = m->ple_history_len, pref_rows = m->ple_pref_rows;
    float *pref = m->ple_pref;
    m->ple_pref = NULL; m->ple_pref_rows = 0;   /* q38_ple reads its rows itself and frees none of ours */
    for (int i = 0; i < c->layers; i++) {
        q38_layer_forward(m, i, hy, ids, S, pos_base, mixed, inject, block);
        memcpy(g_q38m_lc_ref + (size_t)i * SW, hy, SW * sizeof(float));
        q38m_lc_route_rows(m, i, mixed, S, g_q38m_lc_idx + (size_t)i * S * K);   /* mixed: the MLP's */
    }
    g_q38_sel_hook = NULL;
    m->ple_pref = pref; m->ple_pref_rows = pref_rows;
    m->ple_history[0] = hist[0]; m->ple_history[1] = hist[1]; m->ple_history_len = hlen;
    if (g_q38f.layer_check == 2) {
        /* the CPU against itself: the step again from the embedded rows one ulp up, the
         * recurrent state zero again (the check runs at pos_base 0 from a zero host state) */
        q38m_lc_zero_state(m);
        m->ple_pref = NULL; m->ple_pref_rows = 0;
        for (size_t k = 0; k < SW; k++) hy[k] = nextafterf(hyper_h[k], INFINITY);
        for (int i = 0; i < c->layers; i++) {
            q38_layer_forward(m, i, hy, ids, S, pos_base, mixed, inject, block);
            const float *r = g_q38m_lc_ref + (size_t)i * SW;
            double d = 0, mx = 0;
            int wr = -1, moved = 0, first = -1;
            for (size_t k = 0; k < SW; k++) {
                double e = fabs((double)hy[k] - r[k]), a = fabs((double)r[k]);
                if (!(e <= d)) { d = e; wr = (int)(k / c->hc_width); }
                if (a > mx) mx = a;
            }
            q38m_lc_route_rows(m, i, mixed, S, idx);
            for (int row = 0; row < S; row++)
                if (!q38m_lc_same(idx + (size_t)row * K, g_q38m_lc_idx + ((size_t)i * S + row) * K, K)) { moved++; if (first < 0) first = row; }
            fprintf(stderr, "[q38m-cpu-ulp] L%d worst=%.3g (max |cpu| %.3g) at row %d; routes moved in %d rows (first %d)\n",
                    i, d / (mx > 0 ? mx : 1.0), mx, wr, moved, first);
        }
        q38m_lc_zero_state(m);   /* the chain's step starts from the zero state (host_zero) */
        m->ple_pref = pref; m->ple_pref_rows = pref_rows;
        m->ple_history[0] = hist[0]; m->ple_history[1] = hist[1]; m->ple_history_len = hlen;
    }
    free(idx); free(hy); free(mixed); free(inject); free(block);
}

/* the GPU's routes of the chunk's n rows from c0 (its router logits, the CPU's rule) against
 * the reference run's, and against the CPU's router over the GPU's own mixed rows */
static void q38m_lc_routes(Model *m, int i, int n, int c0) {
    const Cfg *c = &m->c;
    const int E = c->experts, K = c->topk;
    const float *gl = q38m_f(g_q38f.lg);
    Q38mLc *lc = &g_q38m_lc[i];
    int *a = (int *)malloc((size_t)K * sizeof(int)), *cpu = (int *)malloc((size_t)n * K * sizeof(int));
    if (!a || !cpu) { fprintf(stderr, "[q38m-layer] OOM\n"); exit(1); }
    q38m_lc_route_rows(m, i, q38m_f(g_q38f.mixed), n, cpu);
    for (int r = 0; r < n; r++) {
        const float *g = gl + (size_t)r * E;
        q38m_lc_topk(g, E, K, a);
        lc->off_ref += !q38m_lc_same(a, g_q38m_lc_idx + ((size_t)i * g_q38m_lc_S + c0 + r) * K, K);
        lc->off_cpu += !q38m_lc_same(a, cpu + (size_t)r * K, K);
        float kth = g[a[K - 1]], out = -INFINITY;   /* the k-th against the best one left out */
        for (int e = 0; e < E; e++) {
            int u = 0;
            for (int z = 0; z < K; z++) u |= a[z] == e;
            if (!u && g[e] > out) out = g[e];
        }
        if (K < E && kth - out < lc->margin) { lc->margin = kth - out; lc->mrow = c0 + r; }
    }
    free(a); free(cpu);
}

/* the GPU experts: rows whose GPU routes (q38x_route, in order) are not the CPU's rule over the
 * same logits (q38_moe_decode_ex's top-k of the softmax: the logits' order) */
static int q38m_lc_gpu_routes(Model *m, int n) {
    const int E = m->c.experts, K = m->c.topk;
    const float *gl = q38m_f(g_q38f.lg);
    const int32_t *ids = (const int32_t *)q36_tensor_data(g_q38x.ids);
    int *a = (int *)malloc((size_t)K * sizeof(int)), off = 0;
    if (!a) { fprintf(stderr, "[q38m-layer] OOM\n"); exit(1); }
    for (int r = 0; r < n; r++) {
        q38m_lc_topk(gl + (size_t)r * E, E, K, a);
        for (int z = 0; z < K; z++) if (a[z] != ids[(size_t)r * K + z]) { off++; break; }
    }
    free(a);
    return off;
}

static int q38m_lc_fdesc(const void *a, const void *b) {
    float x = *(const float *)a, y = *(const float *)b;
    return x > y ? -1 : x < y;
}
/* an attention layer's QSA selection past the budget: per row the gap between the last block
 * taken and the first left out (the GPU's scores), the smallest over the step, and the rows
 * whose cut falls between two scores of 0 (blocks with no positive head: an exact tie the
 * block order breaks) */
static void q38m_lc_qsa(Model *m, int i, int n, int pb) {
    const Cfg *c = &m->c;
    const int R = c->idx_ratio, take = c->idx_budget / R;
    const int nbmax = (pb + n) / R > 0 ? (pb + n) / R : 1;
    const float *sc = q38m_f(g_q38f.sc);
    float *v = (float *)malloc((size_t)nbmax * sizeof(float));
    if (!v) { fprintf(stderr, "[q38m-layer] OOM\n"); exit(1); }
    Q38mLc *lc = &g_q38m_lc[i];
    for (int r = 0; r < n; r++) {
        const int blocks = (pb + r + 1) / R;
        if (blocks <= take) continue;
        memcpy(v, sc + (size_t)r * nbmax, (size_t)blocks * sizeof(float));
        qsort(v, (size_t)blocks, sizeof(float), q38m_lc_fdesc);
        double gap = (double)v[take - 1] - v[take];
        if (v[take - 1] == 0.f) lc->qzero++;
        if (gap < lc->qmargin) { lc->qmargin = gap; lc->qrow = pb + r; }
    }
    free(v);
}

static int q38m_lc_force_sel(int i, int n, int pb) {
    if (!g_q38m_lc_sel || !g_q38m_lc_sel[i]) return 1;
    if (!q36_gpu_sync()) return 0;
    int *sel = (int *)q36_tensor_data(g_q38f.sel);
    for (int r = 0; r < n; r++) {
        const int *src = g_q38m_lc_sel[i] + (size_t)(pb + r) * g_q38m_lc_selrow;
        if (src[0] < 0) continue;
        g_q38m_lc_qrows++;
        int *dst = sel + (size_t)r * g_q38m_lc_selrow;
        if (dst[0] != src[0] || memcmp(dst + 1, src + 1, (size_t)src[0] * sizeof(int))) {
            g_q38m_lc_qforced++;
            /* the same blocks in another order, or other blocks: by block (a block's R positions
             * are taken together), the GPU's scores of the blocks only one side took */
            const int R = g_q38m_lc_R, nb = (pb + n) / R, take = g_q38m_lc_take;
            const float *sc = q38m_f(g_q38f.sc) + (size_t)r * (nb > 0 ? nb : 1);
            unsigned char *in = (unsigned char *)calloc((size_t)nb + 1, 1);
            if (!in) { fprintf(stderr, "[q38m-layer] OOM\n"); exit(1); }
            int cnt = dst[0] < 0 ? 0 : dst[0];
            for (int z = 0; z < cnt && z < take * R; z += R) if (dst[1 + z] / R < nb) in[dst[1 + z] / R] |= 1;
            for (int z = 0; z < src[0] && z < take * R; z += R) if (src[1 + z] / R < nb) in[src[1 + z] / R] |= 2;
            double hi_g = -INFINITY, lo_c = INFINITY, hi_c = -INFINITY, lo_g = INFINITY;
            int sw = 0;
            for (int b = 0; b < nb; b++) {
                if (in[b] == 1) { sw++; if (sc[b] > hi_g) hi_g = sc[b]; if (sc[b] < lo_g) lo_g = sc[b]; }
                if (in[b] == 2) { if (sc[b] > hi_c) hi_c = sc[b]; if (sc[b] < lo_c) lo_c = sc[b]; }
            }
            if (sw) {
                g_q38m_lc_qset++; g_q38m_lc_qswap += sw;
                double gap = fmax(fabs(hi_g - lo_c), fabs(hi_c - lo_g));
                if (gap > g_q38m_lc_qgap) g_q38m_lc_qgap = gap;
            }
            free(in);
        }
        memcpy(dst, src, (size_t)(1 + src[0]) * sizeof(int));
    }
    return 1;
}

/* Q38M_FORCE_ROUTES: the chunk's router logits moved so the CPU's top-k rule picks the
 * reference run's experts: every expert outside them is held under the smallest of them
 * (by 1e-3 of it, so the softmax's expf keeps them apart). The chosen experts' logits stay
 * the GPU's, so with norm_topk their gates are the GPU's; without it the denominator moves
 * on the rows held. Router near-ties then cannot flip, and the check measures the rest. */
static int q38m_lc_force(Model *m, int i, int n, int c0) {
    const Cfg *c = &m->c;
    const int E = c->experts, K = c->topk;
    float *gl = q38m_f(g_q38f.lg);
    int held = 0;
    for (int r = 0; r < n; r++) {
        float *g = gl + (size_t)r * E;
        const int *rr = g_q38m_lc_idx + ((size_t)i * g_q38m_lc_S + c0 + r) * K;
        float lo = INFINITY;
        for (int z = 0; z < K; z++) if (g[rr[z]] < lo) lo = g[rr[z]];
        const float cap = lo - 1e-3f * (1.f + fabsf(lo));
        int moved = 0;
        for (int e = 0; e < E; e++) {
            int in = 0;
            for (int z = 0; z < K; z++) in |= rr[z] == e;
            if (!in && g[e] > cap) { g[e] = cap; moved = 1; }
        }
        held += moved;
    }
    return held;
}

/* 0: not taken (the CPU runs the step from the host state) | 1: done, the logits of
 * the last nlogits rows in logit, the final streams in hyper_h when want_streams,
 * every row's mixed in mixed_h when it is not NULL. hyper_h holds the rows' embedded
 * streams on the way in. */
static int q38m_forward(Model *m, const int *ids, int S, int pos_base, int nlogits,
                        float *hyper_h, int want_streams, float *mixed_h, float *logit) {
    const Cfg *c = &m->c;
    const int H = c->hidden, W = c->hc_width, C = c->hc_count, V = c->vocab, L = c->layers;
    static unsigned long long offered;
    offered++;
    /* a verify's per-row copies (snap_rows) and several conversations' rows stay on the CPU */
    if (!g_q38f.on || S < 1 || nlogits < 1 || nlogits > S || m->snap_rows > 0 || m->mux_rows ||
        (g_q38f.decline_every > 0 && offered % (unsigned long long)g_q38f.decline_every == 0)) {
        g_q38m_declined++;
        return 0;
    }
    const int was_dev = g_q38f.where == Q38M_DEV;
    const int64_t hist0[2] = { m->ple_history[0], m->ple_history[1] };
    const int hlen0 = m->ple_history_len;
    const int check = g_q38f.layer_check && !g_q38m_forwards && pos_base == 0 && g_q38f.where == Q38M_HOST && g_q38f.host_zero;
    /* every buffer the step needs, before anything is encoded or the host's state moves: a
     * refusal here leaves the state as the step found it, and the CPU takes over (q38_forward
     * then runs q38m_cpu_step, which brings the GPU's state down) */
    if ((g_q38f.refuse_at > 0 && offered == (unsigned long long)g_q38f.refuse_at) ||
        !q38m_scratch(c, S < Q38M_CHUNK ? S : Q38M_CHUNK) || !q38m_kv_reserve(m, pos_base + S) ||
        !q38m_grow(&g_q38f.logits, &g_q38f.logit_floats, (size_t)nlogits * V) ||
        !q38x_scratch(S < Q38M_CHUNK ? S : Q38M_CHUNK)) {
        char why[128];
        snprintf(why, sizeof why, "device memory for a step of %d rows at position %d was refused", S, pos_base);
        q38m_off(why);
        q36_gpu_sync();
        q38x_chain_off();
        g_mtl_chain = 0;
        g_q38m_declined++;
        return 0;
    }
    if (check) q38m_layer_check_ref(m, ids, S, pos_base, hyper_h);
    if (g_q38x.on) { qgc_expected_phase(&g_q38x.cache, S, pos_base); q38x_ahead_reset(); }   /* no lease in flight between steps */
    const char *what = NULL;
    int failed_layer = -1, lo_first = S - nlogits, c0;
    for (c0 = 0; c0 < S; c0 += Q38M_CHUNK) {
        if (coli_prompt_break()) { g_q38m_declined++; return 0; }   /* a served prompt's stop, between two chunks */
        if (c0 + Q38M_CHUNK >= S && g_q38f.fail_at > 0 && offered == (unsigned long long)g_q38f.fail_at) {   /* its last chunk */
            what = "a step forced to fail (Q38M_FAIL_AT_STEP)"; failed_layer = 0; goto fail;
        }
        const int n = S - c0 < Q38M_CHUNK ? S - c0 : Q38M_CHUNK, pb = pos_base + c0;
        memcpy(q36_tensor_data(g_q38f.hyper), hyper_h + (size_t)c0 * W, (size_t)n * W * sizeof(float));
        q38m_push(m, pb);
        if (q38m_has_ple(c)) q38m_ple_rows(m, ids, c0, n, q38m_f(g_q38f.emb));
        for (int i = 0; i < L; i++) {
            const Q38mW *w = g_q38m.L[i];
            Q38mL *Ld = &g_q38f.L[i];
            failed_layer = i;
            if (i == c->ple_layer && !q38m_ple(m, n)) { what = "the PLE step"; goto fail; }
            if (!q38m_gr_read(c, &w[0], Ld->attn_norm, n, g_q38f.inj_a)) { what = "the attention's gated-residual read"; goto fail; }
            if (c->is_attn[i] ? !q38m_attention(m, i, n, pb) : !q38m_deltanet(m, i, n)) { what = c->is_attn[i] ? "the attention" : "the DeltaNet"; goto fail; }
            if (!q38_hc_apply(g_q38f.hyper, g_q38f.inj_a, g_q38f.blk, n, H, C) ||
                !q38m_gr_read(c, &w[3], Ld->mlp_norm, n, g_q38f.inj_m) || !q38m_mlp_dense(m, i, n)) { what = "the MLP's dense part"; goto fail; }
            q38x_ahead_emit(i, 1);   /* the guesses for this layer, once its trunk is submitted */
            if (!q36_gpu_sync()) { what = "a GPU batch"; goto fail; }
            q38x_settle(0);   /* the GPU is past the last layer's experts */
            q38x_ahead_take(i, n);
            if (check) {
                if (c->is_attn[i]) q38m_lc_qsa(m, i, n, pb);
                q38m_lc_routes(m, i, n, c0);
                if (g_q38f.force_routes) {
                    g_q38m_lc[i].forced += q38m_lc_force(m, i, n, c0);
                    /* the GPU routes again from the logits held */
                    if (g_q38x.on && (!q38x_route(g_q38f.lg, n) || !q36_gpu_sync())) { what = "the router"; goto fail; }
                }
                if (g_q38x.on) g_q38m_lc[i].gpu_off += q38m_lc_gpu_routes(m, n);
            }
            int sh = q38m_has_shared(m, i), r = -1;
            if (g_q38x.on) {
                r = q38x_moe(m, i, n, g_q38f.mixed, sh ? g_q38f.shd : NULL, g_q38f.blk, q38m_mlp_shared);
                if (!r) { what = "the routed experts"; goto fail; }
                if (r < 0) {   /* routes that are not valid ids (non-finite logits): the CPU's, as M2 */
                    g_q38x.cpu_layers++;
                    if (!q38m_mlp_shared(m, i, n) || !q36_gpu_sync()) { what = "the shared expert"; goto fail; }
                }
            }
            if (r > 0) {
                if (!q38_hc_apply(g_q38f.hyper, g_q38f.inj_m, g_q38f.blk, n, H, C)) { what = "the MoE's sum"; goto fail; }
            } else {
                q38_moe_ex(m, &m->L[i], i, q38m_f(g_q38f.mixed), n, q38m_f(g_q38f.routed), q38m_f(g_q38f.lg), 1);
                if ((sh ? !q36_add(g_q38f.blk, g_q38f.routed, g_q38f.shd, n * H) : 0) ||
                    !q38_hc_apply(g_q38f.hyper, g_q38f.inj_m, sh ? g_q38f.blk : g_q38f.routed, n, H, C)) { what = "the MoE's sum"; goto fail; }
            }
            if (check) {
                if (!q36_gpu_sync()) { what = "a GPU batch"; goto fail; }
                q38x_settle(0);
                const float *g = q38m_f(g_q38f.hyper), *r = g_q38m_lc_ref + (size_t)i * S * W + (size_t)c0 * W;
                for (size_t k = 0; k < (size_t)n * W; k++) {
                    double d = fabs((double)g[k] - r[k]), a = fabs((double)r[k]);
                    if (!(d <= g_q38m_lc[i].diff)) { g_q38m_lc[i].diff = d; g_q38m_lc[i].wrow = c0 + (int)(k / W); }   /* a NaN is the worst */
                    double *part = c0 + (int)(k / W) < 2048 ? &g_q38m_lc[i].diff_lo : &g_q38m_lc[i].diff_hi;
                    if (!(d <= *part)) *part = d;
                    if (c0 / Q38M_CHUNK < 64 && !(d <= g_q38m_lc[i].chunk[c0 / Q38M_CHUNK])) g_q38m_lc[i].chunk[c0 / Q38M_CHUNK] = d;
                    if (a > g_q38m_lc[i].max) g_q38m_lc[i].max = a;
                }
            }
        }
        failed_layer = L;
        /* the final mixer for the rows that need it: the read-out wants every row, the logits
         * the last nlogits */
        int lo0 = lo_first > c0 ? lo_first - c0 : 0;
        if (mixed_h || lo0 < n) {
            if (!q38m_gr_read(c, g_q38m.tail, g_q38f.final_norm, n, NULL)) { what = "the final mixer"; goto fail; }
            if (lo0 < n) {
                size_t rows = (size_t)(n - lo0), dst = (size_t)(c0 + lo0 - lo_first);
                q36_tensor *xv = q36_tensor_view(g_q38f.mixed, (size_t)lo0 * H * sizeof(float), rows * H * sizeof(float));
                q36_tensor *yv = q36_tensor_view(g_q38f.logits, dst * V * sizeof(float), rows * V * sizeof(float));
                int ok = xv && yv && q38m_mm(yv, xv, &g_q38m.tail[3], (int)rows);
                q36_tensor_free(xv); q36_tensor_free(yv);
                if (!ok) { what = "lm_head"; goto fail; }
            }
        }
        if (!q36_gpu_sync()) { what = "a GPU batch"; goto fail; }
        q38x_settle(0);
        if (want_streams) memcpy(hyper_h + (size_t)c0 * W, q36_tensor_data(g_q38f.hyper), (size_t)n * W * sizeof(float));
        if (mixed_h) memcpy(mixed_h + (size_t)c0 * H, q36_tensor_data(g_q38f.mixed), (size_t)n * H * sizeof(float));
        for (int i = 0; i < L; i++)
            if (c->is_attn[i]) { q38m_kv_rows(m, i, pb, pb + n, 0); g_q38f.L[i].kv_valid = pb + n; }
        g_q38f.where = Q38M_DEV; g_q38f.host_zero = 0;
    }
    memcpy(logit, q36_tensor_data(g_q38f.logits), (size_t)nlogits * V * sizeof(float));
    if (check) {
        for (int i = 0; i < L; i++) {
            const Q38mLc *lc = &g_q38m_lc[i];
            double e = lc->diff / (lc->max > 0 ? lc->max : 1.0);
            const double mx = lc->max > 0 ? lc->max : 1.0;
            fprintf(stderr, "[q38m-layer] L%d worst=%.3g (max |cpu| %.3g)%s at row %d (rows < 2048 %.3g, from 2048 %.3g); routes off the reference run's in %d rows, "
                            "off the CPU's on the same rows in %d, smallest margin %.3g at row %d%s\n",
                    i, e, lc->max, e <= 1e-4 ? "" : " FAIL", lc->wrow, lc->diff_lo / mx, lc->diff_hi / mx, lc->off_ref, lc->off_cpu,
                    lc->margin, lc->mrow, g_q38f.force_routes ? "; forced" : "");
            if (g_q38f.force_routes) fprintf(stderr, "[q38m-layer] L%d forced: %d rows held to the reference run's routes\n", i, lc->forced);
            if (g_q38x.on) fprintf(stderr, "[q38m-layer] L%d GPU routes: %d rows unlike the CPU's rule on the same logits\n", i, lc->gpu_off);
            if (S > Q38M_CHUNK) {   /* the worst of each chunk of rows, over max |cpu| */
                char line[1024]; int o = 0;
                for (int k = 0; k * Q38M_CHUNK < S && k < 64 && o < (int)sizeof line - 16; k++)
                    o += snprintf(line + o, sizeof line - o, " %.2g", lc->chunk[k] / mx);
                fprintf(stderr, "[q38m-layer] L%d by chunk of %d rows:%s\n", i, Q38M_CHUNK, line);
            }
            if (c->is_attn[i] && g_q38m_lc_sel)
                fprintf(stderr, "[q38m-layer] L%d QSA forced: the reference run's selection on the rows past the budget\n", i);
            if (c->is_attn[i] && lc->qmargin < INFINITY)
                fprintf(stderr, "[q38m-layer] L%d QSA past the budget: smallest gap at the cut %.3g at row %d; cut between zero scores in %d rows\n",
                        i, lc->qmargin, lc->qrow, lc->qzero);
        }
        if (g_q38m_lc_sel) {
            fprintf(stderr, "[q38m-layer] QSA forced: %d rows past the budget held; %d rows' selections differed from the reference run's (%d in their blocks, "
                            "%d blocks swapped, the largest score gap between swapped blocks %.3g; the rest in order only)\n",
                    g_q38m_lc_qrows, g_q38m_lc_qforced, g_q38m_lc_qset, g_q38m_lc_qswap, g_q38m_lc_qgap);
            for (int i = 0; i < L; i++) free(g_q38m_lc_sel[i]);
            free(g_q38m_lc_sel); g_q38m_lc_sel = NULL;
        }
        free(g_q38m_lc_ref); free(g_q38m_lc); free(g_q38m_lc_idx);
        g_q38m_lc_ref = NULL; g_q38m_lc = NULL; g_q38m_lc_idx = NULL;
    }
    g_q38m_forwards++;
    return 1;
fail:
    /* A kernel that declined: a shape or a size the chain does not take. The device's state
     * may have moved past the host's: the step can be handed to the CPU only when the host
     * held the whole state at the step's start (its first chunk). */
    q36_gpu_sync();
    q38x_settle(1);
    int lost = 0;
    {
        char why[192];
        snprintf(why, sizeof why, "%s of layer %d at position %d did not run on the GPU", what ? what : "a step", failed_layer,
                 pos_base + c0);   /* the failing chunk's first row */
        if (was_dev || c0 > 0) {   /* the recurrent state moved on the GPU only: nothing to hand the CPU */
            if (!g_q38m_serve) {
                fprintf(stderr, "[qwen38-metal] the Metal chain failed: %s; the recurrent state is on the GPU only, "
                                "stopping\n", why);
                exit(1);
            }
            /* a serve keeps serving: the chain off and its memory freed below, the turn ended
             * with an ERROR and the conversation read again on the CPU (q38_chain_lost_end) */
            lost = 1;
            g_q38m_lost = 1;
            fprintf(stderr, "[qwen38-metal] the Metal chain failed with the recurrent state on the GPU only: this turn "
                            "ends with an error and the next request is read again on the CPU\n");
        }
        q38m_off(why);
    }
    m->ple_history[0] = hist0[0]; m->ple_history[1] = hist0[1]; m->ple_history_len = hlen0;
    g_q38f.where = Q38M_HOST; g_q38f.host_zero = 0;
    for (int i = 0; i < L; i++)
        if (c->is_attn[i]) { if (g_q38f.L[i].kv_valid > pos_base) g_q38f.L[i].kv_valid = pos_base; }
    q38x_chain_off();
    g_mtl_chain = 0;
    g_q38m_declined++;
    if (lost) q38m_free(m);   /* the state it held is gone: the device memory goes too */
    return 0;
}

/* the chain's stats, also after it turned off mid-run (the steps it ran) */
static void q38m_report(void) {
    if (!g_mtl_chain && !g_q38m_forwards) return;
    if (g_q38m_kv_grown)
        fprintf(stderr, "[metal] qwen38 kv: device caches grown %llu times to %d rows, %llu of them with valid rows "
                        "(%llu rows uploaded again from the host)\n", g_q38m_kv_grown, g_q38f.dcap, g_q38m_kv_regrow, g_q38m_kv_regrow_rows);
    if (!g_q38x.on && !g_q38x.choices) { fprintf(stderr, "[metal] qwen38 chain: %llu forwards, %llu declined\n", g_q38m_forwards, g_q38m_declined); return; }
    unsigned long long loads = g_q38x.loads, hits = g_q38x.hits;   /* q38x_free's, when the chain turned off */
    if (g_q38x.on) { QgcStats s = qgc_snapshot(&g_q38x.cache); loads = s.loads; hits = s.hits; }
    fprintf(stderr, "[metal] qwen38 chain: %llu forwards, %llu declined, experts on GPU: %llu, loads: %llu, hits: %llu; "
                    "cache groups %llu, most experts in one layer %d, expert reads waited %.2f s%s\n",
            g_q38m_forwards, g_q38m_declined, g_q38x.choices, loads, hits,
            g_q38x.groups, g_q38x.most, g_q38x.wait_s, g_q38x.cpu_layers ? " (some layers' routes on the CPU: non-finite logits)" : "");
    if (g_q38x.guesses)
        fprintf(stderr, "[metal] qwen38 read-ahead: guesses: %llu, right: %llu (%.1f%%), announced (not cached): %llu\n",
                g_q38x.guesses, g_q38x.right, 100.0 * (double)g_q38x.right / (double)g_q38x.guesses, g_q38x.advised);
}

/* The METAL frame (docs/serve_protocol.md): the chain's state for the gateway's /health, one
 * line on stdout after every finished request. loads, hits, wait_s, guesses and right are the
 * turn's (since the previous frame); forwards, declined and kv_regrow the run's. reason: why
 * the chain is off, URL-encoded, or "-". */
static void q38m_frame(void) {
    static unsigned long long p_loads, p_hits, p_guesses, p_right;
    static double p_wait;
    unsigned long long loads = g_q38x.loads, hits = g_q38x.hits;   /* q38x_free's, once the cache is gone */
    if (g_q38x.cache_on) { QgcStats s = qgc_snapshot(&g_q38x.cache); loads = s.loads; hits = s.hits; }
    char reason[3 * sizeof g_q38m_off_why + 1];
    size_t w = 0;
    for (const unsigned char *p = (const unsigned char *)g_q38m_off_why; *p; p++) {
        if ((*p >= 'A' && *p <= 'Z') || (*p >= 'a' && *p <= 'z') || (*p >= '0' && *p <= '9') || strchr("-._~", *p))
            reason[w++] = (char)*p;
        else { reason[w++] = '%'; reason[w++] = "0123456789ABCDEF"[*p >> 4]; reason[w++] = "0123456789ABCDEF"[*p & 15]; }
    }
    if (!w) reason[w++] = '-';
    reason[w] = 0;
    printf("METAL chain=%d reason=%s forwards=%llu declined=%llu experts_gpu=%d cache_gb=%.3f slots=%zu loads=%llu hits=%llu "
           "wait_s=%.4f ahead=%d guesses=%llu right=%llu kv_regrow=%llu\n",
           g_mtl_chain, reason, g_q38m_forwards, g_q38m_declined, g_q38x.on,
           (double)g_q38x.nslots * (double)g_q38x.payload / 1073741824.0, g_q38x.nslots,
           loads - p_loads, hits - p_hits, g_q38x.wait_s - p_wait, g_q38x.ahead,
           g_q38x.guesses - p_guesses, g_q38x.right - p_right, g_q38m_kv_regrow);
    fflush(stdout);
    p_loads = loads; p_hits = hits; p_wait = g_q38x.wait_s; p_guesses = g_q38x.guesses; p_right = g_q38x.right;
}

#endif
