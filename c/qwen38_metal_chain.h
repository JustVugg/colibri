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
 * This file is the skeleton: the decision and the report are real, q38m_forward
 * declines every step and q38m_start allocates nothing yet. */
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

/* The first dense weight the chain cannot read, NULL when it reads them all. The
 * embedding stays on the host (its rows are gathered on the CPU). */
static const Q38Weight *q38m_dense_unsupported(const Model *m) {
    const Cfg *c = &m->c;
    for (int i = 0; i < c->layers; i++) {
        const Layer *l = &m->L[i];
        const Q38Weight *w[] = {
            &l->attn_gr.down, &l->attn_gr.up, &l->attn_gr.inject,
            &l->mlp_gr.down, &l->mlp_gr.up, &l->mlp_gr.inject,
            &l->router, &l->sh_g, &l->sh_u, &l->sh_d,
            &l->q, &l->k, &l->v, &l->o, &l->idx_qk,
            &l->dn_qkv, &l->dn_z, &l->dn_b, &l->dn_a, &l->dn_out,
            &l->ple_key, &l->ple_value,
        };
        for (size_t j = 0; j < sizeof w / sizeof w[0]; j++)
            if (!q38m_weight_ok(w[j])) return w[j];
    }
    const Q38Weight *tail[] = { &m->final_gr.down, &m->final_gr.up, &m->final_gr.inject, &m->lm_head };
    for (size_t j = 0; j < sizeof tail / sizeof tail[0]; j++)
        if (!q38m_weight_ok(tail[j])) return tail[j];
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

/* The chain's device state. Nothing to allocate yet (q38m_forward declines every step). */
static int q38m_start(Model *m) {
    (void)m;
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
