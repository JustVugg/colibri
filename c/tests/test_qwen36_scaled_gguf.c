/* A GGUF matrix with a per-tensor scale (its <name>.scale: e.g. NVFP4 projections converted from
 * Basalt's pack carry 1 / the global scale on every NVFP4 projection) goes to the CUDA tier
 * and the dense chain with the scale on its tensor. Before, qdw_gq_fmt refused it: the
 * whole trunk stayed on the CPU, the chain could not start, and DFlash with it.
 *
 * A two-layer dense qwen35 model in memory (a DeltaNet layer, a gated-attention layer, each with
 * its MLP), every projection NVFP4 blocks with a scale of its own (k_proj none), on the fake
 * backend (qwen36_fake_cuda.h, fake_dense_compute: a block tensor's product is ggml_quants.h's float
 * reference times the tensor's scale, as the CUDA kernels apply it). GQ_EXACT=1: the CPU's products
 * are that float reference too. Checked:
 *   A. a scaled NVFP4 matrix has a device form and is placed with its scale on the tensor; an
 *      unscaled one with none; a backend without the call (an older DLL) keeps a scaled matrix on
 *      the CPU and frees what it uploaded, and still places an unscaled one;
 *   B. the DeltaNet in_proj: qkv and z of two scales go up as two tensors, each with its own;
 *      of one scale as one tensor;
 *   C. a scaled row, its reference: sum over the NVFP4 values (gq_blk_nvfp4) times x, then times the
 *      scale, in double, against the CPU's product and the device path's (qt_dense_matmul_batch);
 *   D. the chain takes every scaled matrix, and a prompt and decode tokens through it give the
 *      logits of the CPU path that ran the same model before anything was placed.
 * Include order as in test_qwen36_cuda_chain.c. */
#define main qwen36_main_unused
#include "../qwen36.c"
#undef main

#include "../compat.h"
#include "qwen36_fake_cuda.h"
#include "../qwen36_tier.c"

static int fails;
static void ck(int ok, const char *what) { if (ok) { printf("  ok   %s\n", what); return; } printf("  FAIL %s\n", what); fails++; }

enum { D = 64, H = 4, KV = 2, HD = 16, QDIM = 2 * HD, ROT = 8, VH = 4, VK = 2, KD = 16, VD = 16, CONVK = 4 };
enum { CD = 2 * VK * KD + VH * VD, VDIM = VH * VD, SI = 64, V = 64, NL = 2, CAP = 32 };

static float frand(uint32_t *s) { *s = *s * 1664525u + 1013904223u; return ((*s >> 8) & 0xFFFF) / 65535.f * 2.f - 1.f; }
static float *fvec(int n, uint32_t *s, float scale, float base) { float *v = falloc(n); for (int i = 0; i < n; i++) v[i] = base + scale * frand(s); return v; }
static double maxdiff(const float *a, const float *b, size_t n) { double d = 0; for (size_t i = 0; i < n; i++) { double x = fabs((double)a[i] - b[i]); if (x > d) d = x; } return d; }
static double maxabs(const float *a, size_t n) { double d = 0; for (size_t i = 0; i < n; i++) if (fabs((double)a[i]) > d) d = fabs((double)a[i]); return d; }
static int argmax(const float *a, int n) { int b = 0; for (int i = 1; i < n; i++) if (a[i] > a[b]) b = i; return b; }

/* an NVFP4 matrix [O x I] as the file stores it (ggml blocks of 64: four UE4M3 scales, 32 code bytes) and its scale */
static float g_gs_next = 0.02f;
static void mk_nvfp4(QW *w, int I, int O, uint32_t *s, int scaled) {
    memset(w, 0, sizeof *w);
    w->I = I; w->O = O; w->gq_type = GQ_NVFP4;
    const size_t nb = (size_t)O * (I / 64) * 36;
    w->gq = malloc(nb);
    for (size_t b = 0; b < nb; b += 36) {
        for (int k = 0; k < 4; k++) w->gq[b + k] = (uint8_t)(0x28 + ((*s = *s * 1664525u + 1013904223u) >> 24) % 16);   /* 1/16 .. 15/16 */
        for (int k = 4; k < 36; k++) w->gq[b + k] = (uint8_t)((*s = *s * 1664525u + 1013904223u) >> 24);
    }
    w->gs = scaled ? g_gs_next : 1.f;
    if (scaled) g_gs_next += 0.0037f;
}

typedef struct { Layer L[NL]; float *embed, *final_norm; QW lm_head; } Weights;
static void make_weights(Weights *w) {
    uint32_t s = 11;
    memset(w, 0, sizeof *w);
    for (int i = 0; i < NL; i++) {
        Layer *l = &w->L[i];
        l->in_ln = fvec(D, &s, 0.1f, 0.f); l->post_ln = fvec(D, &s, 0.1f, 0.f);
        if (i == 1) {
            mk_nvfp4(&l->q, D, H * QDIM, &s, 1); mk_nvfp4(&l->k, D, KV * HD, &s, 0);   /* k_proj: no .scale */
            mk_nvfp4(&l->v, D, KV * HD, &s, 1); mk_nvfp4(&l->o, H * HD, D, &s, 1);
            l->qn = fvec(HD, &s, 0.1f, 0.f); l->kn = fvec(HD, &s, 0.1f, 0.f);
        } else {
            mk_nvfp4(&l->dn_qkv, D, CD, &s, 1); mk_nvfp4(&l->dn_z, D, VDIM, &s, 1); mk_nvfp4(&l->dn_out, VDIM, D, &s, 1);
            l->dn_b = fvec(VH * D, &s, 0.1f, 0.f); l->dn_a = fvec(VH * D, &s, 0.1f, 0.f);
            l->dn_conv = fvec(CD * CONVK, &s, 0.5f, 0.f); l->dn_alog = fvec(VH, &s, 0.5f, -1.f); l->dn_dtbias = fvec(VH, &s, 0.2f, 0.f);
            l->dn_norm = fvec(VD, &s, 0.2f, 1.f);
        }
        mk_nvfp4(&l->sh_g, D, SI, &s, 1); mk_nvfp4(&l->sh_u, D, SI, &s, 1); mk_nvfp4(&l->sh_d, SI, D, &s, 1); l->sh_I = SI;
    }
    w->embed = fvec(V * D, &s, 0.5f, 0.f); w->final_norm = fvec(D, &s, 0.1f, 0.f);
    mk_nvfp4(&w->lm_head, D, V, &s, 1);
}

static void build_model(Model *m, Weights *w) {
    memset(m, 0, sizeof *m);
    Cfg *c = &m->c;
    c->hidden = D; c->n_layers = NL; c->n_active = NL; c->q_heads = H; c->kv_heads = KV; c->head_dim = HD; c->q_head_dim = QDIM;
    c->k_head_dim = HD; c->v_head_dim = HD; c->o_in = H * HD; c->rope_dim = ROT; c->rotary_dim = ROT; c->theta = 10000.f; c->eps = 1e-6f;
    c->n_experts = 0; c->topk = 0; c->inter = 0; c->shared_inter = SI; c->vocab = V; c->has_qk_norm = 1; c->attn_output_gate = 1;
    c->dn_vheads = VH; c->dn_kheads = VK; c->dn_kdim = KD; c->dn_vdim = VD; c->dn_convk = CONVK; c->dn_conv_dim = CD;
    c->is_attn = calloc(NL, 1); c->is_attn[1] = 1;
    m->L = calloc(NL, sizeof(Layer));
    for (int i = 0; i < NL; i++) m->L[i] = w->L[i];
    m->embed = w->embed; m->final_norm = w->final_norm; m->lm_head = w->lm_head;
    m->active_of = calloc(NL, sizeof(int)); for (int i = 0; i < NL; i++) m->active_of[i] = i;
    m->is_pinned = calloc(1, 1); m->is_queued = calloc(1, 1); m->seen = calloc(1, 1);
    m->cache = calloc(NL, sizeof(LCache));
    m->DN_rec = calloc(NL, sizeof(float *)); m->DN_conv = calloc(NL, sizeof(float *));
    m->DN_rec[0] = calloc((size_t)VH * KD * VD, sizeof(float)); m->DN_conv[0] = calloc((size_t)CD * (CONVK - 1), sizeof(float));
    m->dn_dev_fresh = calloc(NL, 1); m->dn_host_stale = calloc(NL, 1);
    m->max_t = CAP; m->kv_cap = CAP;
    ensure_kv(m);
    m->attn_sc_thr = 1;
#ifdef _OPENMP
    m->attn_sc_thr = omp_get_max_threads();
#endif
    m->attn_sc = falloc((int64_t)m->attn_sc_thr * m->max_t);
    kv_prefix_alloc(&m->kvp, CAP);
    m->first_step = 1;
}

static int same_logits(const float *a, const float *b, int n, const char *what) {
    double d = maxdiff(a, b, (size_t)n), s = maxabs(b, (size_t)n);
    int ok = argmax(a, n) == argmax(b, n) && d <= 1e-4 * (s > 1 ? s : 1);
    char buf[200]; snprintf(buf, sizeof buf, "%s (max diff %.2e of %.2e, argmax %d/%d)", what, d, s, argmax(a, n), argmax(b, n));
    ck(ok, buf);
    return ok;
}

/* the fake's tensor behind a handle */
static ColiCudaTensor *ten(int hp1) { int d = -1; return hp1 > 0 ? qt_dense_tensor(hp1 - 1, &d) : NULL; }

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    setenv("Q36_KV", "f32", 1);
    setenv("COLI_LMHEAD_GPU", "0", 1);           /* lm_head on device 0 */
    setenv("GQ_EXACT", "1", 1);                  /* the CPU's block products: the float reference, as the fake's */
    setenv("COLI_CUDA", "1", 1); setenv("COLI_GPUS", "0", 1); setenv("QT_NO_WARMSTART", "1", 1);
    setenv("COLI_DENSE_IDOT", "0", 1);
    setenv("COLI_CUDA_CHAIN", "1", 1);
    unsetenv("HEAT_FILE");
    fake_ndev = 1; fake_dense_compute = 1; fake_block_elems = 64;
    Weights w; make_weights(&w);
    const int ids[8] = {3, 17, 42, 9, 21, 60, 5, 33};

    /* the reference: the CPU path, before anything is placed */
    Model ref; build_model(&ref, &w);
    float *want[5];
    want[0] = step_ex(&ref, ids, 4, 0, 1); ref.kv_len = 4;
    for (int t = 4; t < 8; t++) { want[t - 3] = step_ex(&ref, ids + t, 1, t, 1); ref.kv_len = t + 1; }

    ck(qt_init(NL, 0, D, 0, 1, 1, 0, -1), "the tier is up for a dense model's trunk");
    Model m; build_model(&m, &w);
    Layer *l0 = &m.L[0], *l1 = &m.L[1];

    printf(" A. a scaled matrix has a device form and takes its scale along\n");
    g_gguf = 1;   /* qdw_gq_fmt: a GGUF's matrices (the forward does not read the flag for these) */
    ck(qdw_gq_fmt(&l1->q) == 100 + GQ_NVFP4 && qdw_gs(&l1->q) == l1->q.gs && l1->q.gs != 1.f, "q_proj (scaled NVFP4): format 140, its scale");
    int sc0 = fake_out_scales;
    l1->qth_q = qdw_place(&l1->q, 0);
    ck(l1->qth_q > 0 && ten(l1->qth_q)->fmt == 140 && ten(l1->qth_q)->os == l1->q.gs && fake_out_scales == sc0 + 1,
       "placed, the tensor carrying q_proj's scale");
    l1->qth_k = qdw_place(&l1->k, 0);
    ck(l1->qth_k > 0 && ten(l1->qth_k)->os == 0.f && fake_out_scales == sc0 + 1, "k_proj (no .scale): placed, no scale set");
    fake_out_scale_absent = 1;
    int frees0 = fake_tensor_frees, n0 = qt_dense_count();
    ck(qdw_place(&l1->v, 0) == 0 && fake_tensor_frees == frees0 + 1, "an older DLL: v_proj (scaled) stays on the CPU, its upload freed");
    QW unscaled = l1->v; unscaled.gs = 1.f;
    int hu = qdw_place(&unscaled, 0);
    ck(hu > 0 && qt_dense_count() == n0 + 1, "... and an unscaled matrix still goes up");
    fake_out_scale_absent = 0;
    l1->qth_v = qdw_place(&l1->v, 0); l1->qth_o = qdw_place(&l1->o, 0);
    l0->qth_dnout = qdw_place(&l0->dn_out, 0);
    for (int i = 0; i < NL; i++) {
        m.L[i].qth_shg = qdw_place(&m.L[i].sh_g, 0); m.L[i].qth_shu = qdw_place(&m.L[i].sh_u, 0); m.L[i].qth_shd = qdw_place(&m.L[i].sh_d, 0);
    }
    ck(l1->qth_v && l1->qth_o && l0->qth_dnout && m.L[0].qth_shd && m.L[1].qth_shd && ten(m.L[1].qth_shd)->os == m.L[1].sh_d.gs,
       "every other projection placed with its scale");
    {
        const int fmt = qdw_gq_fmt(&m.lm_head);
        uint8_t *rows; const void *src = qdw_gq_rows(&m.lm_head, fmt, &rows);
        ck(src && qt_lmhead_init_fmt(src, NULL, NULL, fmt, D, V, qdw_gs(&m.lm_head)), "lm_head (scaled) on the card");
        free(rows);
        int d = -1;
        ck(qt_lmhead_tensor(&d) && qt_lmhead_tensor(&d)->os == m.lm_head.gs, "... with its scale");
    }

    printf(" B. the DeltaNet in_proj: two scales, two tensors\n");
    double vram = 0;
    ck(l0->dn_qkv.gs != l0->dn_z.gs && q36_dnproj_place_gguf(&m, 0, 0, &vram) == 2, "qkv and z of two scales: two tensors");
    {
        int d = -1;
        ColiCudaTensor *tq = qt_dnproj_tensor(0, &d), *tz = qt_dnproj_z_tensor(0, &d);
        ck(tq && tz && tq->O == CD && tz->O == VDIM && tq->os == l0->dn_qkv.gs && tz->os == l0->dn_z.gs, "each with its own scale");
    }
    {   /* one scale: one tensor (on layer 1's slot of the tier, given back after) */
        Model one = m;
        Layer L1 = m.L[0];
        L1.dn_z.gs = L1.dn_qkv.gs;
        Layer arr[NL] = {m.L[0], L1};
        one.L = arr;
        ck(q36_dnproj_place_gguf(&one, 1, 0, &vram) == 1, "qkv and z of one scale: one tensor");
        int d = -1;
        ColiCudaTensor *t1 = qt_dnproj_tensor(1, &d);
        ck(t1 && t1->O == CD + VDIM && t1->os == L1.dn_qkv.gs && !qt_dnproj_z_tensor(1, &d), "... qkv's rows then z's, the one scale");
        coli_cuda_tensor_free(G_dnp[1].t); G_dnp[1].t = NULL; G_dnp[1].on = 0;
    }
    g_gguf = 0;

    printf(" C. a scaled row: the reference, the CPU's product, the device path's\n");
    {
        uint32_t s = 5;
        float x[3 * D], ycpu[3 * H * QDIM], ydev[3 * H * QDIM];
        for (int i = 0; i < 3 * D; i++) x[i] = frand(&s);
        const QW *q = &l1->q;
        double worst = 0;
        for (int o = 0; o < q->O; o++) {
            float blk[64]; double acc = 0;
            for (int b = 0; b < D / 64; b++) {
                gq_blk_nvfp4(q->gq + ((size_t)o * (D / 64) + b) * 36, blk);
                for (int k = 0; k < 64; k++) acc += (double)blk[k] * x[b * 64 + k];
            }
            acc *= q->gs;   /* the sum, then the scale: what the tensor's kernels do */
            matmul_d(ycpu, x, q, 1, D, q->O);
            double e = fabs(ycpu[o] - acc) / (fabs(acc) > 1e-3 ? fabs(acc) : 1e-3);
            if (e > worst) worst = e;
        }
        char buf[120]; snprintf(buf, sizeof buf, "the CPU's scaled NVFP4 rows equal the reference (worst relative %.1e)", worst);
        ck(worst < 1e-5, buf);
        matmul_d(ycpu, x, q, 3, D, q->O);
        ck(qt_dense_matmul_batch(l1->qth_q - 1, ydev, x, 3, D, q->O) && !memcmp(ycpu, ydev, sizeof ycpu),
           "the device path's three rows (the tensor's scale) are the CPU's bits");
    }

    printf(" D. the chain takes the scaled trunk: its logits are the CPU path's\n");
    Q36CChain *ch = q36cc_setup(&m, 1, NULL);
    ck(ch != NULL && ch->ok && ch->n == NL && ch->head, "the chain set itself up with every scaled matrix and the head");
    if (!ch) { printf("qwen36 scaled gguf: %d FAIL\n", fails); return 1; }
    ck(ch->t_dnz[0] != NULL, "layer 0's in_proj as two tensors (qkv, z)");
    g_cuda_chain = 1;
    int frames0 = fake_chain_frames;
    float *got = step_ex(&m, ids, 4, 0, 1); m.kv_len = 4;
    ck(fake_chain_frames > frames0, "the prompt ran through the chain");
    same_logits(got, want[0], V, "the prompt's logits equal the CPU path's");
    free(got);
    for (int t = 4; t < 8; t++) {
        got = step_ex(&m, ids + t, 1, t, 1); m.kv_len = t + 1;
        char what[64]; snprintf(what, sizeof what, "decode token %d", t);
        same_logits(got, want[t - 3], V, what);
        free(got);
    }
    ck(ch->forwards >= 5 && !ch->failed && g_cuda_chain, "the chain took every step");
    g_cuda_chain = 0;
    for (int i = 0; i < 5; i++) free(want[i]);
    printf(fails ? "qwen36 scaled gguf: %d FAIL\n" : "qwen36 scaled gguf: ok\n", fails);
    return fails != 0;
}
