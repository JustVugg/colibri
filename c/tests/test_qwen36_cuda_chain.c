/* The CUDA dense chain (COLI_CUDA_CHAIN=1, qwen36_cuda_chain.h) computes what the CPU
 * path computes, and the host state and the device state never disagree.
 *
 * An in-memory two-layer model (a DeltaNet layer, then a gated-attention layer, four
 * routed experts and a gated shared expert per layer, lm_head), no container, driven
 * through step_ex() the way step() drives it, with the whole trunk placed on the fake
 * CUDA tier (qwen36_fake_cuda.h, fake_dense_compute) and the chain's ops answered by
 * the fake's host-side table -- the same arithmetic as cuda_chain.cu's kernels. The
 * properties:
 *   - a prompt and decode tokens through the chain give the CPU path's logits;
 *   - the host's DeltaNet state is not touched while the chain runs, and reads back
 *     (q36cc_sync_host, the pinned snapshot's call) as the CPU path's state; the K/V
 *     rows the chain wrote into the host cache are the CPU path's;
 *   - a CPU step after chain steps (reset, then a prompt on the CPU) and a chain step
 *     after a host write (pin_restore's upload) continue from the right state;
 *   - a speculative verify copies the state per row on the device and a rejected draft
 *     rolls the device buffers back with the host's (q36cc_rollback);
 *   - a frame that fails loses the device: the state is rebuilt on the CPU from the
 *     prefix record, the step and every later one run there, and the logits still
 *     equal the CPU path's;
 *   - a backend without the chain (an older DLL) leaves the engine on the per-matrix path.
 * Include order as in test_qwen36_tier_int8_engine.c. */
#define main qwen36_main_unused
#include "../qwen36.c"
#undef main

#include "../compat.h"
#include "qwen36_fake_cuda.h"
#include "../qwen36_tier.c"

static int fails;
static void ck(int ok, const char *what) { if (ok) { printf("  ok   %s\n", what); return; } printf("  FAIL %s\n", what); fails++; }

enum { D = 64, H = 4, KV = 2, HD = 16, QDIM = 2 * HD, ROT = 8, VH = 4, VK = 2, KD = 16, VD = 16, CONVK = 4 };
enum { CD = 2 * VK * KD + VH * VD, VDIM = VH * VD, PD = CD + VDIM, E = 4, TOPK = 2, INTER = 32, SI = 32, V = 64, NL = 2, CAP = 48 };

static float frand(uint32_t *s) { *s = *s * 1664525u + 1013904223u; return ((*s >> 8) & 0xFFFF) / 65535.f * 2.f - 1.f; }
static float *fvec(int n, uint32_t *s, float scale, float base) { float *v = falloc(n); for (int i = 0; i < n; i++) v[i] = base + scale * frand(s); return v; }
static void mk_qw(QW *w, int I, int O, uint32_t *s) {
    memset(w, 0, sizeof *w);
    w->I = I; w->O = O; w->q = malloc((size_t)I * O); w->sc = malloc((size_t)O * sizeof(float));
    for (size_t i = 0; i < (size_t)I * O; i++) w->q[i] = (int8_t)(frand(s) * 60);
    for (int o = 0; o < O; o++) w->sc[o] = 0.02f + 0.01f * frand(s);
}
static double maxdiff(const float *a, const float *b, size_t n) { double d = 0; for (size_t i = 0; i < n; i++) { double x = fabs((double)a[i] - b[i]); if (x > d) d = x; } return d; }
static double maxabs(const float *a, size_t n) { double d = 0; for (size_t i = 0; i < n; i++) if (fabs((double)a[i]) > d) d = fabs((double)a[i]); return d; }
static int argmax(const float *a, int n) { int b = 0; for (int i = 1; i < n; i++) if (a[i] > a[b]) b = i; return b; }

/* the weights every model instance shares */
typedef struct { Layer L[NL]; float *embed, *final_norm; QW lm_head; int8_t *eg[NL][E], *eu[NL][E], *ed[NL][E]; } Weights;
static void make_weights(Weights *w) {
    uint32_t s = 7;
    memset(w, 0, sizeof *w);
    for (int i = 0; i < NL; i++) {
        Layer *l = &w->L[i];
        l->in_ln = fvec(D, &s, 0.1f, 0.f); l->post_ln = fvec(D, &s, 0.1f, 0.f);
        if (i == 1) {
            mk_qw(&l->q, D, H * QDIM, &s); mk_qw(&l->k, D, KV * HD, &s); mk_qw(&l->v, D, KV * HD, &s); mk_qw(&l->o, H * HD, D, &s);
            l->qn = fvec(HD, &s, 0.1f, 0.f); l->kn = fvec(HD, &s, 0.1f, 0.f);
        } else {
            mk_qw(&l->dn_qkv, D, CD, &s); mk_qw(&l->dn_z, D, VDIM, &s); mk_qw(&l->dn_out, VDIM, D, &s);
            l->dn_b = fvec(VH * D, &s, 0.1f, 0.f); l->dn_a = fvec(VH * D, &s, 0.1f, 0.f);
            l->dn_conv = fvec(CD * CONVK, &s, 0.5f, 0.f); l->dn_alog = fvec(VH, &s, 0.5f, -1.f); l->dn_dtbias = fvec(VH, &s, 0.2f, 0.f);
            l->dn_norm = fvec(VD, &s, 0.2f, 1.f);
        }
        mk_qw(&l->gate, D, E, &s);
        mk_qw(&l->sh_g, D, SI, &s); mk_qw(&l->sh_u, D, SI, &s); mk_qw(&l->sh_d, SI, D, &s);
        l->sh_gate = fvec(D, &s, 0.1f, 0.f);
        for (int e = 0; e < E; e++) {
            w->eg[i][e] = malloc((size_t)INTER * D); w->eu[i][e] = malloc((size_t)INTER * D); w->ed[i][e] = malloc((size_t)D * INTER);
            for (size_t k = 0; k < (size_t)INTER * D; k++) { w->eg[i][e][k] = (int8_t)(frand(&s) * 40); w->eu[i][e][k] = (int8_t)(frand(&s) * 40); w->ed[i][e][k] = (int8_t)(frand(&s) * 40); }
        }
    }
    w->embed = fvec(V * D, &s, 0.5f, 0.f); w->final_norm = fvec(D, &s, 0.1f, 0.f);
    mk_qw(&w->lm_head, D, V, &s);
}

/* a model instance: the shared weights, its own state */
static void build_model(Model *m, Weights *w) {
    memset(m, 0, sizeof *m);
    Cfg *c = &m->c;
    c->hidden = D; c->n_layers = NL; c->n_active = NL; c->q_heads = H; c->kv_heads = KV; c->head_dim = HD; c->q_head_dim = QDIM;
    c->k_head_dim = HD; c->v_head_dim = HD; c->o_in = H * HD; c->rope_dim = ROT; c->rotary_dim = ROT; c->theta = 10000.f; c->eps = 1e-6f;
    c->n_experts = E; c->topk = TOPK; c->inter = INTER; c->shared_inter = SI; c->vocab = V; c->has_qk_norm = 1; c->attn_output_gate = 1;
    c->dn_vheads = VH; c->dn_kheads = VK; c->dn_kdim = KD; c->dn_vdim = VD; c->dn_convk = CONVK; c->dn_conv_dim = CD;
    c->is_attn = calloc(NL, 1); c->is_attn[1] = 1;
    m->L = calloc(NL, sizeof(Layer));
    for (int i = 0; i < NL; i++) m->L[i] = w->L[i];
    m->embed = w->embed; m->final_norm = w->final_norm; m->lm_head = w->lm_head;
    m->active_of = calloc(NL, sizeof(int)); for (int i = 0; i < NL; i++) m->active_of[i] = i;
    m->is_pinned = calloc((size_t)NL * E, 1); m->is_queued = calloc((size_t)NL * E, 1); m->seen = calloc((size_t)NL * E, 1);
    m->cache = calloc(NL, sizeof(LCache));
    for (int i = 0; i < NL; i++) {
        LCache *lc = &m->cache[i];
        lc->cap = E; lc->n = E; lc->slots = calloc(E, sizeof(Slot)); lc->slot_by_expert = malloc(E * sizeof(int));
        for (int e = 0; e < E; e++) {
            Slot *sl = &lc->slots[e];
            slot_ensure_allocated(m, sl);
            memcpy(sl->g, w->eg[i][e], (size_t)INTER * D); memcpy(sl->u, w->eu[i][e], (size_t)INTER * D); memcpy(sl->d, w->ed[i][e], (size_t)D * INTER);
            for (int k = 0; k < INTER; k++) { sl->gs[k] = 0.03f; sl->us[k] = 0.03f; }
            for (int k = 0; k < D; k++) sl->ds[k] = 0.03f;
            sl->eid = e; sl->pinned = 1; sl->used = ++m->clock; lc->slot_by_expert[e] = e;
        }
    }
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

/* the trunk on the fake tier: every matrix the chain needs, one device */
static void place_trunk(Model *m, Weights *w) {
    int8_t *qf = malloc((size_t)PD * D); float *sf = malloc((size_t)PD * sizeof(float));
    memcpy(qf, w->L[0].dn_qkv.q, (size_t)CD * D); memcpy(qf + (size_t)CD * D, w->L[0].dn_z.q, (size_t)VDIM * D);
    memcpy(sf, w->L[0].dn_qkv.sc, CD * sizeof(float)); memcpy(sf + CD, w->L[0].dn_z.sc, VDIM * sizeof(float));
    ck(qt_dnproj_init(0, qf, sf, D, PD, 0), "the DeltaNet in_proj on the card");
    free(qf); free(sf);
    m->L[0].qth_dnout = qt_dense_init(w->L[0].dn_out.q, w->L[0].dn_out.sc, VDIM, D, 0) + 1;
    m->L[1].qth_q = qt_dense_init(w->L[1].q.q, w->L[1].q.sc, D, H * QDIM, 0) + 1;
    m->L[1].qth_k = qt_dense_init(w->L[1].k.q, w->L[1].k.sc, D, KV * HD, 0) + 1;
    m->L[1].qth_v = qt_dense_init(w->L[1].v.q, w->L[1].v.sc, D, KV * HD, 0) + 1;
    m->L[1].qth_o = qt_dense_init(w->L[1].o.q, w->L[1].o.sc, H * HD, D, 0) + 1;
    for (int i = 0; i < NL; i++) {
        m->L[i].qth_shg = qt_dense_init(w->L[i].sh_g.q, w->L[i].sh_g.sc, D, SI, 0) + 1;
        m->L[i].qth_shu = qt_dense_init(w->L[i].sh_u.q, w->L[i].sh_u.sc, D, SI, 0) + 1;
        m->L[i].qth_shd = qt_dense_init(w->L[i].sh_d.q, w->L[i].sh_d.sc, SI, D, 0) + 1;
    }
    ck(m->L[0].qth_dnout && m->L[1].qth_q && m->L[1].qth_k && m->L[1].qth_v && m->L[1].qth_o && m->L[1].qth_shd, "every projection and the shared expert placed");
    ck(qt_lmhead_init(w->lm_head.q, w->lm_head.sc, D, V), "lm_head on the card");
}

static int same_logits(const float *a, const float *b, int n, const char *what) {
    double d = maxdiff(a, b, (size_t)n), s = maxabs(b, (size_t)n);
    int ok = argmax(a, n) == argmax(b, n) && d <= 1e-3 * (s > 1 ? s : 1);
    char buf[160]; snprintf(buf, sizeof buf, "%s (max diff %.2e of %.2e, argmax %d/%d)", what, d, s, argmax(a, n), argmax(b, n));
    ck(ok, buf);
    return ok;
}

int main(void) {
    setenv("COLI_CUDA", "1", 1); setenv("COLI_GPUS", "0", 1); setenv("QT_NO_WARMSTART", "1", 1);
    setenv("COLI_LMHEAD_GPU", "0", 1);
    setenv("COLI_DENSE_IDOT", "0", 1);          /* f32 activations on the CPU too: the chain's arithmetic */
    setenv("COLI_CUDA_CHAIN", "1", 1);
    fake_ndev = 1; fake_dense_compute = 1;
    Weights w; make_weights(&w);
    const int ids[12] = {3, 17, 42, 9, 21, 60, 5, 33, 12, 48, 7, 29};   /* the verify reads three from index 6 */

    ck(qt_init(NL, E, D, INTER, E, TOPK, 0, 0), "the tier is up");
    Model ref; build_model(&ref, &w);
    place_trunk(&ref, &w);        /* the reference uses the same resident copies, matrix by matrix */
    Model m; build_model(&m, &w);
    for (int i = 0; i < NL; i++) { m.L[i].qth_dnout = ref.L[i].qth_dnout; m.L[i].qth_q = ref.L[i].qth_q; m.L[i].qth_k = ref.L[i].qth_k; m.L[i].qth_v = ref.L[i].qth_v;
                                   m.L[i].qth_o = ref.L[i].qth_o; m.L[i].qth_shg = ref.L[i].qth_shg; m.L[i].qth_shu = ref.L[i].qth_shu; m.L[i].qth_shd = ref.L[i].qth_shd; }

    printf(" A. the chain against the CPU path\n");
    Q36CChain *ch = q36cc_setup(&m, 1);
    ck(ch != NULL && ch->ok && ch->n == NL && ch->head, "the chain set itself up on the fake device, every layer and the head");
    g_cuda_chain = 1;
    float *lr = step_ex(&ref, ids, 4, 0, 1);
    ref.kv_len = 4;
    int frames0 = fake_chain_frames;
    float *lc = step_ex(&m, ids, 4, 0, 1);
    m.kv_len = 4;
    ck(fake_chain_frames > frames0, "the prompt ran through the chain");
    same_logits(lc, lr, V, "the prompt's logits equal the CPU path's");
    free(lr); free(lc);
    { int allzero = 1; for (size_t i = 0; i < (size_t)VH * KD * VD; i++) if (m.DN_rec[0][i] != 0.f) allzero = 0; ck(allzero, "the host's DeltaNet state was not touched"); }
    ck(maxdiff(m.K[1], ref.K[1], (size_t)KV * CAP * HD) < 1e-4 && maxdiff(m.V[1], ref.V[1], (size_t)KV * CAP * HD) < 1e-4, "the K/V rows reached the host cache as the CPU path wrote them");
    for (int t = 4; t < 7; t++) {
        lr = step_ex(&ref, ids + t, 1, t, 1); ref.kv_len = t + 1;
        lc = step_ex(&m, ids + t, 1, t, 1); m.kv_len = t + 1;
        char what[64]; snprintf(what, sizeof what, "decode token %d", t);
        same_logits(lc, lr, V, what);
        free(lr); free(lc);
    }
    q36cc_sync_host(&m);
    ck(maxdiff(m.DN_rec[0], ref.DN_rec[0], (size_t)VH * KD * VD) < 1e-3 && maxdiff(m.DN_conv[0], ref.DN_conv[0], (size_t)CD * (CONVK - 1)) < 1e-4,
       "the state read back from the device equals the CPU path's");
    ck(ch->dn_where == Q36CC_BOTH, "both sides hold it now");

    printf(" B. a host write, then the chain again; a CPU prompt, then the chain again\n");
    reset_recurrent(&ref); reset_recurrent(&m); ref.kv_len = m.kv_len = 0;
    ck(ch->dn_where == Q36CC_HOST && ch->host_zero, "the reset wrote zeros on the host: the device fills its copy with zeros before the next step");
    lr = step_ex(&ref, ids, 3, 0, 1); ref.kv_len = 3;
    lc = step_ex(&m, ids, 3, 0, 1); m.kv_len = 3;
    same_logits(lc, lr, V, "after the reset the chain starts from zeros too");
    free(lr); free(lc);
    g_cuda_chain = 0;                                      /* the CPU takes the next step (as a declined chain) */
    q36cc_cpu_step(&m, 3);
    lr = step_ex(&ref, ids + 3, 2, 3, 1); ref.kv_len = 5;
    lc = step_ex(&m, ids + 3, 2, 3, 1); m.kv_len = 5;
    same_logits(lc, lr, V, "a CPU step continues from the device's state");
    free(lr); free(lc);
    g_cuda_chain = 1;
    lr = step_ex(&ref, ids + 5, 1, 5, 1); ref.kv_len = 6;
    lc = step_ex(&m, ids + 5, 1, 5, 1); m.kv_len = 6;
    same_logits(lc, lr, V, "the chain continues from the host's state, the KV rows the CPU wrote uploaded first");
    free(lr); free(lc);

    printf(" C. a verify's copies and the rollback of a rejected draft\n");
    ck(q36_spec_alloc(&ref, 2) && q36_spec_alloc(&m, 2), "verify slots on the host");
    ref.snap_rows = 2; m.snap_rows = 2; g_q36_rowwise = 1;
    lr = step_ex(&ref, ids + 6, 3, 6, 3);
    lc = step_ex(&m, ids + 6, 3, 6, 3);
    ref.snap_rows = m.snap_rows = 0; g_q36_rowwise = 0;
    ck(same_logits(lc, lr, V, "row 0 of the verify") && same_logits(lc + V, lr + V, V, "row 1") && same_logits(lc + 2 * V, lr + 2 * V, V, "row 2"), "a verify's three rows through the chain");
    free(lr); free(lc);
    ck(ch->snap_valid == 2, "the device copied the state after rows 0 and 1");
    q36_spec_rollback(&ref, 7, 1); q36_spec_rollback(&m, 7, 1);   /* the draft after row 0 rejected: 7 positions stand */
    lr = step_ex(&ref, ids + 7, 1, 7, 1); ref.kv_len = 8;
    lc = step_ex(&m, ids + 7, 1, 7, 1); m.kv_len = 8;
    same_logits(lc, lr, V, "after the rollback the next token continues from the state after row 0");
    free(lr); free(lc);

    printf(" D. a frame that fails: the state rebuilt on the CPU, the step and the rest there\n");
    q36cc_sync_host(&ref);   /* no chain on the reference: a no-op */
    fake_chain_fail_at = fake_chain_frames + 2;
    lr = step_ex(&ref, ids + 2, 1, 8, 1); ref.kv_len = 9;
    lc = step_ex(&m, ids + 2, 1, 8, 1); m.kv_len = 9;
    fake_chain_fail_at = 0;
    ck(!g_cuda_chain && ch->failed, "the chain is off after the lost device");
    same_logits(lc, lr, V, "the step was computed on the CPU from the rebuilt state");
    free(lr); free(lc);
    lr = step_ex(&ref, ids + 1, 1, 9, 1);
    lc = step_ex(&m, ids + 1, 1, 9, 1);
    same_logits(lc, lr, V, "and the CPU keeps going");
    free(lr); free(lc);

    printf(" E. a backend without the chain\n");
    { Model m2; build_model(&m2, &w);
      for (int i = 0; i < NL; i++) m2.L[i] = m.L[i];
      fake_chain_absent = 1;
      /* cc_ops() caches the table once per process: ask the engine's gate directly */
      ck(coli_cuda_chain_ops() == NULL, "the backend has no table");
      fake_chain_absent = 0;
      ck(q36cc_setup(&m2, 2) == NULL, "KV_SLOTS > 1 keeps the chain off");
      (void)m2; }

    qt_shutdown();
    if (fails) { printf("test_qwen36_cuda_chain: %d failure(s)\n", fails); return 1; }
    printf("OK test_qwen36_cuda_chain: the CUDA chain gives the CPU path's logits; host and device state agree; a lost device rebuilds\n");
    return 0;
}
