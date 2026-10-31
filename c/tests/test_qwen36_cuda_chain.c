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
 *   - a backend without the chain (an older DLL) leaves the engine on the per-matrix path;
 *   - speculative steps (prompt lookup, the MTP head) run under the tier only when it holds
 *     every routed expert and the chain runs them (q36_cuda_tier_drafts);
 *   - routed experts in the frame, all on the device or one layer's served by the host, give
 *     the CPU path's logits and route books, with one copy of the route ids a frame;
 *   - the MTP head on the chain picks its drafts on the device (cc_pick): the CPU head's ids,
 *     and its probabilities to 1e-5; a listed vocabulary, Q36_MTP_DUMP or a backend without the
 *     op keep bringing the scores down. With the adaptive list (Q36_MTP_VOCAB) it scores the
 *     listed rows of the device's lm_head (cc_matmul_rows), and the device's copy of the list
 *     follows every append and every rebuild, across requests too; a backend without
 *     matmul_rows scores the whole vocabulary.
 *   - the host pool follows the residents before a prompt chunk of Q36CC_FOLLOW_ROWS rows or
 *     more, not before a smaller one, and a change stales the staged table (q36cc_follow);
 *   - the promotions' brake holds only where the warm start's set came from a history (this
 *     model's heat file); COLI_CUDA_CHAIN_PROMOTE_HITS, when set, wins;
 *   - the staging buffer for a prompt chunk's host-pool experts grows into the old one's place: a small card
 *     has no room for both (q36cc_stage_reserve).
 * Include order as in test_qwen36_tier_int8_engine.c. */
#define main qwen36_main_unused
#include "../qwen36.c"
#undef main

#include "../compat.h"
#include "qwen36_fake_cuda.h"
#include "../qwen36_tier.c"

static int fails;
static void ck(int ok, const char *what) { if (ok) { printf("  ok   %s\n", what); return; } printf("  FAIL %s\n", what); fails++; }
/* the MTP head's device copy of its adaptive list (the fake's CC_DEV buffers live on the host) is the list */
static int vids_same(void) {
    const DraftVocab *v = g_q36_mtp.voc;
    return v && g_q36_mc.vids && g_q36_mc.vids_n == v->list_n &&
           !memcmp(g_q36_mc.vids->d, v->list, (size_t)v->list_n * sizeof(int));
}

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
        mk_qw(&l->sh_g, D, SI, &s); mk_qw(&l->sh_u, D, SI, &s); mk_qw(&l->sh_d, SI, D, &s); l->sh_I = SI;
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

/* a hybrid layer on the fake, whose ops run when called: the host answers inside the device's
 * wait (the engine serves it again after the next layer, the same rows) */
static Q36CChain *serve_ch; static Model *serve_m; static int serve_n;
static int serve_layer1(unsigned v) { return q36cc_dm_serve(serve_ch, serve_m, 1, serve_n, v); }

/* K's tier: planar int4 per 64 as on a Qwen3.6 container (the host pool wants it), eight experts
 * a layer, half of them on the card (as test_qwen36_tier_host_pool.c's) */
enum { HE = 8, HI = 64, HMB = D * HI / 2, HSC = 2 * HI * (D / 64) + D * (HI / 64) };
static unsigned char hp_w[NL][HE][3 * HMB];
static float hp_sc[NL][HE][HSC];
static int hp_pooled(int l, int e) { const void *w; const float *s; size_t a, b; return qt_chain_host(l, e, &w, &a, &s, &b); }
static void hp_note(int l, int e, int how) {   /* 0: offered to the card, 1: to the pool, 2: a miss */
    const unsigned char *g = hp_w[l][e]; const float *s = hp_sc[l][e];
    if (how == 1) qt_note_host(l, e, g, g + HMB, g + 2 * HMB, s, s + HI, s + 2 * HI);
    else if (how == 2) qt_note(l, e, g, g + HMB, g + 2 * HMB, s, s + HI, s + 2 * HI);
    else qt_note_block(l, e, g, g + HMB, g + 2 * HMB, s, s + HI, s + 2 * HI);
}
static void hp_route(int l, int a, int b) {   /* a row's routes, the misses offered as the engine's miss path does */
    int ids[2] = {a, b};
    qt_chain_count(l, ids, 2);
    for (int k = 0; k < 2; k++) if (!qt_is_resident(l, ids[k])) hp_note(l, ids[k], 2);
}

/* the reference's resident copies, matrix by matrix (sections G and H write the same lines out) */
static void share_trunk(Model *to, const Model *from) {
    for (int i = 0; i < NL; i++) {
        to->L[i].qth_dnout = from->L[i].qth_dnout; to->L[i].qth_q = from->L[i].qth_q; to->L[i].qth_k = from->L[i].qth_k;
        to->L[i].qth_v = from->L[i].qth_v; to->L[i].qth_o = from->L[i].qth_o; to->L[i].qth_shg = from->L[i].qth_shg;
        to->L[i].qth_shu = from->L[i].qth_shu; to->L[i].qth_shd = from->L[i].qth_shd;
    }
}

/* The lm_head computes on the card and one copy takes its logits to the host's buffer: no matmul of
 * the chain's steps stores through the bus, and the logits equal the CPU path's after a prompt read
 * in chunks, a decode token and a verify whose three logits rows lie in two chunks. */
static void head_logits_on_the_card(Weights *w, const Model *shared, const int *ids) {
    g_q36_mtp.on = 0; memset(&g_q36_mc, 0, sizeof g_q36_mc);   /* section H's MTP head is not part of these steps */
    Model r, m; build_model(&r, w); build_model(&m, w);
    share_trunk(&r, shared); share_trunk(&m, shared);
    Q36CChain *ch = q36cc_setup(&m, 1);
    ck(ch != NULL && ch->ok, "chain model ready for the head's logits");
    g_cuda_chain = 1;
    int stores0 = fake_chain_host_stores;
    ch->rows_max = 3;
    float *lr = step_ex(&r, ids, 7, 0, 1), *lc = step_ex(&m, ids, 7, 0, 1);
    r.kv_len = m.kv_len = 7;
    same_logits(lc, lr, V, "a prompt of seven tokens in chunks of three: its logits equal the CPU path's");
    free(lr); free(lc);
    lr = step_ex(&r, ids + 7, 1, 7, 1); lc = step_ex(&m, ids + 7, 1, 7, 1);
    r.kv_len = m.kv_len = 8;
    same_logits(lc, lr, V, "the decode token after it");
    free(lr); free(lc);
    ck(q36_spec_alloc(&r, 2) && q36_spec_alloc(&m, 2), "verify slots on the host");
    r.snap_rows = m.snap_rows = 2; g_q36_rowwise = 1;
    ch->rows_max = 2;   /* row 0 and 1 in the first chunk, row 2 in the second */
    lr = step_ex(&r, ids + 8, 3, 8, 3); lc = step_ex(&m, ids + 8, 3, 8, 3);
    r.snap_rows = m.snap_rows = 0; g_q36_rowwise = 0;
    ck(same_logits(lc, lr, V, "row 0 of a verify split in two chunks") && same_logits(lc + V, lr + V, V, "row 1") &&
       same_logits(lc + 2 * V, lr + 2 * V, V, "row 2"), "a verify's three rows, two chunks");
    free(lr); free(lc);
    ck(ch->forwards == 3 && !ch->failed && g_cuda_chain, "the chain took the three steps, none went to the CPU");
    ck(fake_chain_host_stores == stores0, "no matmul of those steps stored into host memory: the head's logits come down in a copy");
    g_cuda_chain = 0;
}

int main(void) {
    setenv("Q36_KV", "f32", 1);   /* the host's K/V rows compared below as floats */
    setenv("COLI_CUDA", "1", 1); setenv("COLI_GPUS", "0", 1); setenv("QT_NO_WARMSTART", "1", 1);
    unsetenv("HEAT_FILE");                      /* G: the tier starts without a history */
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
    {   /* the four positions written, per head (the rows past them are malloc's: not zero on Windows) */
        int kv_ok = 1;
        for (int h = 0; h < KV; h++) {
            size_t off = (size_t)h * CAP * HD;
            if (maxdiff((const float *)m.K[1] + off, (const float *)ref.K[1] + off, (size_t)4 * HD) > 1e-4 ||
                maxdiff((const float *)m.V[1] + off, (const float *)ref.V[1] + off, (size_t)4 * HD) > 1e-4) kv_ok = 0;
        }
        ck(kv_ok, "the K/V rows reached the host cache as the CPU path wrote them");
    }
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
      /* KV_SLOTS > 1: the chain stays on, each conversation's state in a place of its own
       * (q36cc_conv_*, serve_mux) */
      Q36CChain *c2 = q36cc_setup(&m2, 2);
      ck(c2 != NULL && c2->ok, "KV_SLOTS > 1 keeps the chain on");
      (void)m2; }

    printf(" F. speculative steps under the tier\n");
    g_cuda_chain = 1;
    ck(!q36_cuda_tier_drafts(&m), "routed experts off the device: no speculative steps (residency would follow the drafts)");
    for (int i = 0; i < NL; i++)
        for (int e = 0; e < E; e++) {
            const Slot *sl = &m.cache[i].slots[e];
            qt_note_block(i, e, (const uint8_t *)sl->g, (const uint8_t *)sl->u, (const uint8_t *)sl->d, sl->gs, sl->us, sl->ds);
        }
    qt_fill_wait();
    ck(q36_cuda_tier_drafts(&m), "every routed expert on the device, the chain on: speculative steps");
    g_cuda_chain = 0;
    ck(!q36_cuda_tier_drafts(&m), "the chain off: none");

    printf(" G. routed experts in the frame: logits, route books, one ids copy a frame\n");
    fake_chain_table.size = offsetof(ColiCudaChainOps, tensor_dev) + sizeof fake_chain_table.tensor_dev;
    for (int hyb = 0; hyb < 2; hyb++) {   /* every expert on the device, then layer 1 with one on the host */
        Model r3, m3; build_model(&r3, &w); build_model(&m3, &w);
        for (int i = 0; i < NL; i++) {
            r3.L[i].qth_dnout = m3.L[i].qth_dnout = ref.L[i].qth_dnout;
            r3.L[i].qth_q = m3.L[i].qth_q = ref.L[i].qth_q; r3.L[i].qth_k = m3.L[i].qth_k = ref.L[i].qth_k;
            r3.L[i].qth_v = m3.L[i].qth_v = ref.L[i].qth_v; r3.L[i].qth_o = m3.L[i].qth_o = ref.L[i].qth_o;
            r3.L[i].qth_shg = m3.L[i].qth_shg = ref.L[i].qth_shg; r3.L[i].qth_shu = m3.L[i].qth_shu = ref.L[i].qth_shu;
            r3.L[i].qth_shd = m3.L[i].qth_shd = ref.L[i].qth_shd;
        }
        r3.freq = calloc((size_t)NL * E, sizeof(uint64_t)); m3.freq = calloc((size_t)NL * E, sizeof(uint64_t));
        r3.resident_collecting = m3.resident_collecting = 1;
        if (hyb) setenv("COLI_CUDA_CHAIN_PROMOTE", "0", 1);   /* the chain promotes nothing */
        Q36CChain *c3 = q36cc_setup(&m3, 1);
        ck(c3 != NULL && c3->ok && c3->dm_on, "routed experts in the frame enabled");
        if (hyb) {   /* expert 3 of layer 1 on the host, and kept there: with the residents made hot,
                      * the tier's swap (a quarter hotter than the coldest, tier_should_promote)
                      * never takes it back */
            res_set(qs(1, 3), 0);
            for (int i = 0; i < NL; i++) for (int e = 0; e < E; e++) if (qs(i, e)->resident) qs(i, e)->heat = 1u << 20;
            serve_ch = c3; serve_m = &m3; fake_wait_hook = serve_layer1;
        }
        /* a step's downloads: the attention layer's K and V, the hybrid layer's ids, weights and
         * rows, then the ids of every layer in one copy, and the head's logits */
        const int want = 2 + 3 * hyb + 1 + 1;
        g_cuda_chain = 1;
        float *lr3 = step_ex(&r3, ids, 3, 0, 1); r3.kv_len = 3;
        int down0 = fake_chain_down_copies;
        serve_n = 3; float *lc3 = step_ex(&m3, ids, 3, 0, 1); m3.kv_len = 3;
        same_logits(lc3, lr3, V, "three prompt rows with routed experts in the frame");
        ck(fake_chain_down_copies - down0 == want, hyb ? "prompt chunk: the hybrid layer's three copies, then one ids copy, the logits" :
                                                         "prompt chunk: one route-ids download, no route-weights download, the logits");
        int books = !memcmp(m3.seen, r3.seen, (size_t)NL * E) && !memcmp(m3.freq, r3.freq, (size_t)NL * E * sizeof(uint64_t));
        free(lr3); free(lc3);
        /* five decode rows, the books checked after each: ids read at the old offsets
         * (i * rows * K, rows the chunk's capacity) would be another step's */
        int one = 1;
        for (int t = 3; t < 8; t++) {
            lr3 = step_ex(&r3, ids + t, 1, t, 1); r3.kv_len = t + 1;
            down0 = fake_chain_down_copies;
            serve_n = 1; lc3 = step_ex(&m3, ids + t, 1, t, 1); m3.kv_len = t + 1;
            same_logits(lc3, lr3, V, "decode row with routed experts in the frame");
            one &= fake_chain_down_copies - down0 == want;
            books &= !memcmp(m3.seen, r3.seen, (size_t)NL * E) && !memcmp(m3.freq, r3.freq, (size_t)NL * E * sizeof(uint64_t));
            free(lr3); free(lc3);
        }
        ck(one && c3->dm_steps == 6 * NL && (hyb ? c3->dm_hyb > 0 : c3->dm_hyb == 0),
           hyb ? "decode steps: the hybrid layer's three copies, then one ids copy, the logits" :
                 "decode steps: one route-ids download across all layers, the logits");
        ck(books, "route books (seen and freq) equal the CPU path's after the prompt and every step");
        if (hyb) { fake_wait_hook = NULL; res_set(qs(1, 3), 1); unsetenv("COLI_CUDA_CHAIN_PROMOTE"); }
        g_cuda_chain = 0;
    }
    fake_chain_table.size = CC_OPS_BASE_SIZE;

    printf(" H. the MTP head on the CUDA chain: its drafts picked on the device\n");
    /* the whole vocabulary or the adaptive list: picked on the device; a listed vocabulary, or a
     * backend without the op: the scores come down and the host picks, as before; the adaptive
     * list on a backend without matmul_rows: the whole vocabulary. Each against the CPU head. */
    for (int hc = 0; hc < 5; hc++) {
        static const char *what[5] = {"the whole vocabulary", "a listed vocabulary", "a backend without the op",
                                      "an adaptive list", "an adaptive list, a backend without matmul_rows"};
        fake_chain_table.size = hc == 2 ? offsetof(ColiCudaChainOps, pick) :
                                hc == 4 ? offsetof(ColiCudaChainOps, matmul_rows) : sizeof(ColiCudaChainOps);
        uint32_t ms = 99;
        Q36Mtp *p = &g_q36_mtp;
        memset(p, 0, sizeof *p);
        p->on = 1; p->E = E; p->K = TOPK; p->I = INTER; p->ish = SI;
        if (hc == 1) {   /* Q36_MTP_VOCAB_IDS: row j of dvoc scores id vid[j] */
            p->nvoc = 16; p->vid = malloc(16 * sizeof(int));
            for (int j = 0; j < 16; j++) p->vid[j] = 3 + 2 * j;
            mk_qw(&p->dvoc, D, 16, &ms);
        }
        p->enorm = fvec(D, &ms, 0.1f, 0.f); p->hnorm = fvec(D, &ms, 0.1f, 0.f); p->head_norm = fvec(D, &ms, 0.1f, 0.f);
        p->L.in_ln = fvec(D, &ms, 0.1f, 0.f); p->L.post_ln = fvec(D, &ms, 0.1f, 0.f);
        p->L.qn = fvec(HD, &ms, 0.1f, 0.f); p->L.kn = fvec(HD, &ms, 0.1f, 0.f);
        mk_qw(&p->eh, 2 * D, D, &ms);
        mk_qw(&p->L.q, D, H * QDIM, &ms); mk_qw(&p->L.k, D, KV * HD, &ms);
        mk_qw(&p->L.v, D, KV * HD, &ms); mk_qw(&p->L.o, H * HD, D, &ms);
        mk_qw(&p->sh_g, D, SI, &ms); mk_qw(&p->sh_u, D, SI, &ms); mk_qw(&p->sh_d, SI, D, &ms);
        p->router = fvec(E * D, &ms, 0.2f, 0.f); p->sh_gate = fvec(D, &ms, 0.1f, 0.f);
        p->eg = calloc(E, sizeof(QW)); p->eu = calloc(E, sizeof(QW)); p->ed = calloc(E, sizeof(QW));
        for (int e = 0; e < E; e++) { mk_qw(&p->eg[e], D, INTER, &ms); mk_qw(&p->eu[e], D, INTER, &ms); mk_qw(&p->ed[e], INTER, D, &ms); }
        p->pend = falloc(D);
        if (hc == 3) {   /* the CPU head's list too: ids 40..63, so no draft's place in it is its id */
            p->voc = draft_vocab_new(V, 16);
            for (int t = 40; t < V; t++) draft_vocab_count(p->voc, t);
            const int hi[3] = {40, 51, 63};
            q36_mtp_vocab_begin(hi, 3);
        }
        Model r3, m3; build_model(&r3, &w); build_model(&m3, &w);
        for (int i = 0; i < NL; i++) {
            r3.L[i].qth_dnout = m3.L[i].qth_dnout = ref.L[i].qth_dnout;
            r3.L[i].qth_q = m3.L[i].qth_q = ref.L[i].qth_q; r3.L[i].qth_k = m3.L[i].qth_k = ref.L[i].qth_k;
            r3.L[i].qth_v = m3.L[i].qth_v = ref.L[i].qth_v; r3.L[i].qth_o = m3.L[i].qth_o = ref.L[i].qth_o;
            r3.L[i].qth_shg = m3.L[i].qth_shg = ref.L[i].qth_shg; r3.L[i].qth_shu = m3.L[i].qth_shu = ref.L[i].qth_shu;
            r3.L[i].qth_shd = m3.L[i].qth_shd = ref.L[i].qth_shd;
        }
        Q36CChain *c3 = q36cc_setup(&m3, 1);
        ck(c3 != NULL && c3->ok, "chain model ready for the MTP head");

        float *hid_in = fvec(3 * D, &ms, 0.3f, 0.f), h_cpu[D], h_gpu[D];
        memset(&g_q36_mc, 0, sizeof g_q36_mc);   /* the reference on the CPU, not the last case's head */
        g_cuda_chain = 0;
        q36_mtp_feed_rows(&r3, ids, 3, 0, hid_in);
        int d1_cpu = q36_mtp_draft(&r3, ids[3], 3, h_cpu);
        float p1_cpu = p->last_p;
        int d2_cpu = q36_mtp_draft_more(&r3, h_cpu, d1_cpu, 4);
        float p2_cpu = p->last_p;

        g_cuda_chain = 1;
        memset(&g_q36_mc, 0, sizeof g_q36_mc);
        p->on = 1;
        if (hc == 4) { p->voc = draft_vocab_new(V, 16); q36_mtp_vocab_begin(ids, 3); }   /* dropped at init */
        q36_mtp_feed_rows(&m3, ids, 3, 0, hid_in);
        int d1_gpu = q36_mtp_draft(&m3, ids[3], 3, h_gpu);
        float p1_gpu = p->last_p;
        int d2_gpu = q36_mtp_draft_more(&m3, h_gpu, d1_gpu, 4);
        float p2_gpu = p->last_p;

        int dev = hc == 0 || hc >= 3;
        if (hc == 4) ck(!p->voc, "an adaptive list, a backend without matmul_rows: the list is dropped");
        char msg[160];
        snprintf(msg, sizeof msg, "%s: %s", what[hc], dev ? "the logits stay in device memory, an 8-byte pick comes down" :
                                                            "the logits come down to the host, as before");
        ck(g_q36_mc.on && g_q36_mc.pick_dev == dev && g_q36_mc.outd->kind == (dev ? CC_DEV : CC_DOWN) &&
           (!dev || (g_q36_mc.pick->kind == CC_DOWN && g_q36_mc.pick->bytes == 8)), msg);
        snprintf(msg, sizeof msg, "%s: draft 1 and draft 2 equal the CPU head's, their probabilities to 1e-5", what[hc]);
        ck(d1_gpu == d1_cpu && d2_gpu == d2_cpu && fabsf(p1_gpu - p1_cpu) <= 1e-5f * p1_cpu && fabsf(p2_gpu - p2_cpu) <= 1e-5f * p2_cpu,
           msg);
        if (hc == 3) ck(d1_gpu >= 40 && d2_gpu >= 40, "an adaptive list: the drafts are the list's ids, not their places in it");
        if (hc == 0) {
            setenv("Q36_MTP_DUMP", "/tmp/unused", 1);
            memset(&g_q36_mc, 0, sizeof g_q36_mc);
            ck(q36_mc_init(&m3) && !g_q36_mc.pick_dev && g_q36_mc.outd->kind == CC_DOWN,
               "Q36_MTP_DUMP set when the head is set up: the logits buffer on the host, as before");
            unsetenv("Q36_MTP_DUMP");
        }
        if (hc == 3) {   /* the device's copy of the list (read straight from the fake's buffer) */
            q36_mtp_vocab_step(7);   /* an id outside the list: appended at its tail */
            int d3 = q36_mtp_draft_more(&m3, h_gpu, d2_gpu, 5);
            ck(p->voc->list[p->voc->list_n - 1] == 7 && vids_same(), "an adaptive list: an appended id reaches the device");
            while (!p->voc->run.rebuilds) q36_mtp_vocab_step(5);   /* the counts halve: a rebuilt list */
            int d4 = q36_mtp_draft_more(&m3, h_gpu, d3, 6);
            ck(vids_same(), "an adaptive list: a rebuilt list reaches the device");
            const int next[3] = {60, 33, 48};
            q36_mtp_vocab_begin(next, 3);   /* the next request: after one rebuild the list changes again */
            q36_mtp_draft_more(&m3, h_gpu, d4, 7);
            ck(vids_same(), "an adaptive list: the next request's list reaches the device");
        }
        if (p->voc) { draft_vocab_free(p->voc); p->voc = NULL; }
        free(hid_in);
        g_cuda_chain = 0;
    }
    fake_chain_table.size = CC_OPS_BASE_SIZE;

    printf(" I. the head's logits: computed on the card, then copied to the host\n");
    head_logits_on_the_card(&w, &ref, ids);

    printf(" J. the staging buffer grows without room for a second copy beside it\n");
    {   /* residents spread over the layers start the staging small and every promotion raises it; on a small card
         * (room for 400 here) the bigger buffer takes the old one's place, it does not wait beside it */
        CcBuf *st = NULL;
        fake_chain_dev_cap = fake_chain_dev_bytes + 400;
        ck(q36cc_stage_reserve(&st, 320) && cc_bytes(st) >= 320, "a staging buffer of 320");
        ck(q36cc_stage_reserve(&st, 347) && cc_bytes(st) >= 347, "grown to 347 with room for 400: no second copy beside it");
        ck(q36cc_stage_reserve(&st, 300) && cc_bytes(st) >= 347, "a smaller ask keeps the buffer");
        ck(!q36cc_stage_reserve(&st, 500), "an ask over the card's room fails");
        cc_free(st); fake_chain_dev_cap = 0;
    }

    qt_shutdown();

    printf(" K. the host pool follows the residents before a large prompt chunk only\n");
    {
        setenv("HEAT_FILE", "", 1); setenv("COLI_PLACE", "off", 1); setenv("QT_UPLOAD_SYNC", "", 1);
        fake_host_reg_ok = 1; fake_planar_rows = 1;
        int regs0 = fake_host_regs, unregs0 = fake_host_unregs;
        for (int i = 0; i < NL; i++) for (int e = 0; e < HE; e++) {
            for (size_t k = 0; k < sizeof hp_w[i][e]; k++) hp_w[i][e][k] = (unsigned char)((k * 37u + (unsigned)i * 13u + (unsigned)e * 7u) & 0xFFu);
            for (int k = 0; k < HSC; k++) hp_sc[i][e][k] = 0.25f * (float)(k + e + 1);
        }
        size_t xb = 3 * dev_alloc_footprint(HMB) + 3 * dev_alloc_footprint((HSC / 3) * sizeof(float));
        char gb[64]; snprintf(gb, sizeof gb, "%.15f", (double)(HE * xb + xb / 2) / 1073741824.0);
        setenv("CUDA_EXPERT_GB", gb, 1);
        qt_source_planar(1);
        ck(qt_init(NL, HE, D, HI, HE, TOPK, 64, 1), "a planar tier");
        for (int e = 0; e < HE; e++) for (int i = 0; i < NL; i++) hp_note(i, e, 0);
        qt_fill_wait();
        for (int i = 0; i < NL; i++) for (int e = 0; e < HE; e++) if (!qt_is_resident(i, e)) hp_note(i, e, 1);
        ck(qt_chain_host_count() == HE, "the warm start: half the experts on the card, the other half page-locked in the pool");
        for (int t = 0; t < 3; t++) { hp_route(0, 4, 5); hp_route(1, 0, 5); }
        QtPromo pr[8];
        int np = qt_chain_promote(8, pr);
        for (int k = 0; k < np; k++) qt_chain_promote_done(pr[k].ci, pr[k].vi, 1);
        int vl = np > 0 ? pr[0].vi / HE : 0, ve = np > 0 ? pr[0].vi % HE : 0;
        ck(np == 3 && !qt_is_resident(vl, ve) && !hp_pooled(vl, ve), "three promotions: the victims left the card and are not in the pool");

        Q36CChain fc; memset(&fc, 0, sizeof fc);
        fc.dm_stage_dirty = 1;   /* q36cc_dm_begin read the residents' moves */
        int regs = fake_host_regs;
        q36cc_follow(&fc, Q36CC_STAGE_ROWS + 4);
        ck(fc.hp_follows == 0 && fake_host_regs == regs && !hp_pooled(vl, ve),
           "a chunk of 36 rows (a chat's next question) leaves the pool behind: no page-lock");
        fc.dm_stage_dirty = 0;   /* that chunk's staged table, rebuilt as before (q36cc_stage_table) */
        q36cc_follow(&fc, Q36CC_FOLLOW_ROWS);
        ck(fc.hp_follows == 1 && fc.hp_joined == 3 && fc.hp_left == 3 && hp_pooled(vl, ve),
           "the next chunk of Q36CC_FOLLOW_ROWS rows: the pool follows, though the table was rebuilt since the moves");
        ck(fc.dm_stage_dirty, "and the staged table is stale again");
        fc.dm_stage_dirty = 0;
        q36cc_follow(&fc, 512);
        ck(fc.hp_follows == 1 && !fc.dm_stage_dirty && fake_host_regs == regs + 6, "nothing moved since: the pool and the table stand");
        qt_source_planar(0);
        qt_shutdown();
        ck(fake_host_regs - regs0 == fake_host_unregs - unregs0, "the exit releases every page-lock the pool took");
    }
    printf(" L. the promotions' brake only where the warm start came from a history\n");
    unsetenv("COLI_CUDA_CHAIN_PROMOTE_HITS");
    ck(!qt_chain_history() && q36cc_promote_hits() == 0, "no heat file: no history, no brake");
    setenv("COLI_CUDA_CHAIN_PROMOTE_HITS", "60", 1);
    ck(q36cc_promote_hits() == 60, "the variable, when set, wins: 60 without a history");
    unsetenv("COLI_CUDA_CHAIN_PROMOTE_HITS");
    static const char heat[] = "test_qwen36_cuda_chain_heat.bin";
    for (int own = 0; own < 2; own++) {   /* another model's heat file (an expert more a layer), then this one's */
        uint32_t hdr[3] = {0x51544831u, NL, own ? E : E + 1}, h = 7;
        FILE *f = fopen(heat, "wb");
        fwrite(hdr, 4, 3, f);
        for (uint32_t i = 0; i < hdr[1] * hdr[2]; i++) fwrite(&h, 4, 1, f);
        fclose(f);
        setenv("HEAT_FILE", heat, 1);
        ck(qt_init(NL, E, D, INTER, E, TOPK, 0, 0), "the tier is up again, a heat file given");
        ck(qt_chain_history() == own && q36cc_promote_hits() == (own ? Q36CC_PROMO_HITS : 0),
           own ? "this model's heat file: a history, the brake at 75" : "another model's heat file is not read: no brake");
        if (own) {
            setenv("COLI_CUDA_CHAIN_PROMOTE_HITS", "0", 1);
            ck(q36cc_promote_hits() == 0, "the variable wins: 0 with a history");
            unsetenv("COLI_CUDA_CHAIN_PROMOTE_HITS");
        }
        unsetenv("HEAT_FILE");   /* qt_shutdown would write the heat back */
        qt_shutdown();
    }
    remove(heat);
    if (fails) { printf("test_qwen36_cuda_chain: %d failure(s)\n", fails); return 1; }
    printf("OK test_qwen36_cuda_chain: the CUDA chain gives the CPU path's logits; host and device state agree; a lost device rebuilds; the host pool follows before a large prompt chunk only; the promotions' brake only with a history\n");
    return 0;
}
