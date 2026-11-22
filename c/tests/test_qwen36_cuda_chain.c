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
 *   - the staged table of a prompt chunk (q36cc_dm_begin) follows the tier's residents when the warm
 *     start pooled every expert (qt_chain_host_all): after a promotion and a give-back every expert off
 *     the card is staged, none of the layers goes to the host's step whole, and the follow has nothing
 *     to do; without it a chunk of 36 rows leaves the layer holding a victim to the host;
 *   - the promotions' brake holds only where the warm start's set came from a history (this
 *     model's heat file); COLI_CUDA_CHAIN_PROMOTE_HITS, when set, wins;
 *   - the staging buffer for a prompt chunk's host-pool experts grows into the old one's place: a small card
 *     has no room for both (q36cc_stage_reserve).
 *   - the staged table puts the host pool's experts in the staging's one or two banks (q36cc_stage_table):
 *     two where two layers' worth fit in a layer's and the card has room, the staged layers taking them in
 *     turn, each expert at its bank's offset in both layouts; the chain's calls (first copy of each bank, then
 *     one after each layer) never write a bank before its layer is done with it.
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

static int g_cap = CAP;   /* the models' context (section M and --probe: a long prompt's) */
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
    m->max_t = g_cap; m->kv_cap = g_cap;
    ensure_kv(m);
    m->attn_sc_thr = 1;
#ifdef _OPENMP
    m->attn_sc_thr = omp_get_max_threads();
#endif
    m->attn_sc = falloc((int64_t)m->attn_sc_thr * m->max_t);
    kv_prefix_alloc(&m->kvp, g_cap);
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

/* P's tier: HL layers of the same experts, to see where q36cc_stage_table puts the host pool's in the staging */
enum { HL = 5 };
static unsigned char sq_w[HL][HE][3 * HMB];
static float sq_sc[HL][HE][HSC];
/* the staging's ops on the fake, a state machine for each bank: the layer a fetch put in it, which the layer's
 * wait must find and its done must release before the next fetch lands there */
static struct { int hold[2], bad, fetches, waits, expect, maxend, firstat[HL], routed; Q36CChain *ch; } sg;
static int sg_layer(const CcFetch *r) {   /* the layer whose copies these are */
    if (sg.routed) return sg.expect;   /* a short chunk's list is built apart (q36cc_routed_fetches) */
    for (int i = 0; i < HL; i++) if (sg.ch->dm_frn[i] > 0 && r == sg.ch->dm_fr + sg.ch->dm_fro[i]) return i;
    return -1;
}
static int sg_fetch_at(CcBuf *dst, const CcFetch *r, int n, int bank) {
    int i = sg_layer(r);
    if (bank < 0 || bank > 1 || i < 0 || sg.hold[bank] >= 0) sg.bad++;   /* a bank still unread, or copies of no layer */
    else { sg.hold[bank] = i; sg.fetches++; sg.firstat[i] = n ? (int)r[0].dst : -1; }
    for (int k = 0; k < n; k++) {
        if (r[k].dst + r[k].bytes > cc_bytes(dst)) sg.bad++;   /* past the buffer */
        if (r[k].dst + r[k].bytes > (size_t)sg.maxend) sg.maxend = (int)(r[k].dst + r[k].bytes);
    }
    return 1;
}
static int sg_wait_at(int bank) { sg.waits++; if (bank < 0 || bank > 1 || sg.hold[bank] != sg.expect) sg.bad++; return 1; }
static int sg_done_at(int bank) { if (bank < 0 || bank > 1 || sg.hold[bank] != sg.expect) sg.bad++; else sg.hold[bank] = -1; return 1; }
static int sg_fetch(CcBuf *dst, const CcFetch *r, int n) { return sg_fetch_at(dst, r, n, 0); }
static int sg_wait(void) { return sg_wait_at(0); }
static int sg_done(void) { return sg_done_at(0); }

/* A tier of HL layers with `res` experts of the HE on the card (the first ones, in every layer) and the rest in the
 * host pool, `cap` bytes of device memory for the staging (0: no limit), one layer (unstaged_l, -1: none) with an
 * expert in neither, a backend whose table has the banks' ops (at_ops) or ends before them. Builds the staged table,
 * then runs the staged layers as the chain does: the first copy of each bank, and after a layer's experts the copy
 * of the layer that takes its bank. */
static void stage_table_case(int res, int planar, int unstaged_l, size_t cap, int at_ops, int want_banks, const char *what) {
    char gb[64], msg[240];
    setenv("HEAT_FILE", "", 1); setenv("COLI_PLACE", "off", 1); setenv("QT_UPLOAD_SYNC", "", 1);
    fake_host_reg_ok = 1; fake_planar_rows = planar;
    for (int i = 0; i < HL; i++) for (int e = 0; e < HE; e++) {
        for (size_t k = 0; k < sizeof sq_w[i][e]; k++) sq_w[i][e][k] = (unsigned char)((k * 31u + (unsigned)i * 11u + (unsigned)e * 5u) & 0xFFu);
        for (int k = 0; k < HSC; k++) sq_sc[i][e][k] = 0.5f * (float)(k + e + i + 1);
    }
    size_t xb = 3 * dev_alloc_footprint(HMB) + 3 * dev_alloc_footprint((HSC / 3) * sizeof(float));
    snprintf(gb, sizeof gb, "%.15f", (double)((size_t)HL * res * xb + xb / 2) / 1073741824.0);
    setenv("CUDA_EXPERT_GB", gb, 1);
    qt_source_planar(planar);
    int up = qt_init(HL, HE, D, HI, HE, TOPK, 64, 1);
    for (int e = 0; e < HE; e++) for (int i = 0; i < HL; i++) {
        const unsigned char *g = sq_w[i][e]; const float *s = sq_sc[i][e];
        qt_note_block(i, e, g, g + HMB, g + 2 * HMB, s, s + HI, s + 2 * HI);
    }
    qt_fill_wait();
    int pool = 0, resident = 0;
    for (int i = 0; i < HL; i++) for (int e = 0; e < HE; e++) {
        if (qt_is_resident(i, e)) { resident++; continue; }
        if (i == unstaged_l && e == HE - 1) continue;   /* in neither */
        const unsigned char *g = sq_w[i][e]; const float *s = sq_sc[i][e];
        qt_note_host(i, e, g, g + HMB, g + 2 * HMB, s, s + HI, s + 2 * HI);
        pool++;
    }
    snprintf(msg, sizeof msg, "%s: %d experts a layer on the card, the rest in the pool", what, res);
    ck(up && resident == HL * res && qt_chain_host_count() == pool, msg);

    static Q36CChain ch; static Model sm;
    memset(&ch, 0, sizeof ch); memset(&sm, 0, sizeof sm);
    sm.c.n_layers = HL; sm.c.n_experts = HE; sm.c.hidden = D; sm.c.inter = HI;
    ch.dm_fmt = 4; ch.dm_gs = 64;
    ch.dm_host = calloc((size_t)HL * HE, sizeof(CcExpert)); ch.dm_res = calloc((size_t)HL * HE, 1);
    for (int i = 0; i < HL; i++) for (int e = 0; e < HE; e++) ch.dm_res[i * HE + e] = (uint8_t)qt_is_resident(i, e);
    memset(&sg, 0, sizeof sg); sg.hold[0] = sg.hold[1] = -1; sg.ch = &ch;
    ColiCudaChainOps saved = fake_chain_table;
    fake_chain_table.size = at_ops ? sizeof fake_chain_table : offsetof(ColiCudaChainOps, fetch_at);
    fake_chain_table.fetch = sg_fetch; fake_chain_table.fetch_wait = sg_wait; fake_chain_table.fetch_done = sg_done;
    fake_chain_table.fetch_at = at_ops ? sg_fetch_at : NULL; fake_chain_table.fetch_wait_at = at_ops ? sg_wait_at : NULL;
    fake_chain_table.fetch_done_at = at_ops ? sg_done_at : NULL;
    fake_chain_dev_cap = cap ? fake_chain_dev_bytes + cap : 0;

    const int hmax = HE - res, staged = HL - (unstaged_l >= 0);
    const size_t wb = 3 * HMB, wa = (wb + 255) / 256 * 256, stride = wa + (HSC * sizeof(float) + 255) / 256 * 256, bank = (size_t)hmax * stride;
    ck(q36cc_stage_table(&ch, &sm), what);
    snprintf(msg, sizeof msg, "%s: %d bank%s of %d experts", what, want_banks, want_banks == 1 ? "" : "s", hmax);
    ck(ch.dm_banks == want_banks && cc_bytes(ch.dm_stage) >= want_banks * bank, msg);
    int banks_ok = 1, seen = 0, at_ok = 1;
    const char *base = (const char *)cc_dev_addr(ch.dm_stage);
    for (int i = 0; i < HL; i++) {
        const int st = q36cc_stages(&ch, ch.dm_tabb, i);
        banks_ok = banks_ok && (st == (i != unstaged_l)) && ch.dm_bank[i] == (st ? seen % want_banks : -1);
        for (int e = 0; e < HE && st; e++) {   /* the staged expert h of a layer at (bank * hmax + h) * stride, as the pool holds it */
            const CcExpert *x = &ch.dm_hostb[(size_t)i * HE + e];
            if (ch.dm_res[i * HE + e]) continue;
            const size_t off = (size_t)((const char *)x->g - base), h = off / stride - (size_t)ch.dm_bank[i] * hmax;
            at_ok = at_ok && off % stride == 0 && h < (size_t)hmax && x->u == (const char *)x->g + HMB && x->d == (const char *)x->g + 2 * HMB &&
                    (const char *)x->gs == base + off + wa && x->us == x->gs + HI && x->ds == x->gs + 2 * HI && x->planar == planar;
        }
        seen += st;
    }
    snprintf(msg, sizeof msg, "%s: the staged layers take the banks in turn, an unstaged layer takes none", what);
    ck(banks_ok && seen == staged, msg);
    snprintf(msg, sizeof msg, "%s: each expert in its layer's bank, its matrices and scales after it (%s)", what, planar ? "planar" : "pairs");
    ck(at_ok, msg);

    const int banks = q36cc_banks(&ch);
    int ok = 1, first_ok = 1;
    for (int s = 1; s <= banks; s++) ok = ok && q36cc_stage_from(&ch, &sm, 0, s);
    for (int i = 0; i < HL; i++) {
        if (!q36cc_stages(&ch, ch.dm_tabb, i)) continue;
        sg.expect = i;
        ok = ok && q36cc_fetch_wait(&ch, i) && q36cc_fetch_done(&ch, i) && q36cc_stage_from(&ch, &sm, i + 1, banks);
        first_ok = first_ok && sg.firstat[i] == ch.dm_bank[i] * (int)bank;   /* the layer's copy starts at its bank */
    }
    snprintf(msg, sizeof msg, "%s: each copy lands in a free bank at its layer's offset and is waited for by its layer", what);
    ck(ok && sg.bad == 0 && sg.fetches == staged && sg.waits == staged && sg.hold[0] < 0 && sg.hold[1] < 0 &&
       sg.maxend <= (int)(want_banks * bank) && first_ok, msg);

    /* a short chunk's routed copy, layer by layer as q36cc_dm_layer runs it: after the router, the layer's bank, and
     * the wait and done of that same bank (a copy with another bank's events leaves its bank's state unmatched) */
    sm.c.topk = TOPK;
    fake_chain_dev_cap = 0;
    const int hot = HE - 1;   /* not on the card in any layer: the last of the pool's, the layer's last staged expert */
    ok = cc_reserve(&ch.dm_idsd, (size_t)HL * TOPK * sizeof(int), CC_DEV);
    for (int i = 0; ok && i < HL; i++) { ((int *)cc_ptr(ch.dm_idsd))[i * TOPK] = 0; ((int *)cc_ptr(ch.dm_idsd))[i * TOPK + 1] = hot; }
    memset(&sg.hold, 0xff, sizeof sg.hold); sg.bad = sg.fetches = sg.waits = sg.maxend = 0; sg.routed = 1;
    int routed_at_ok = 1, routed_n = 0;
    for (int i = 0; ok && i < HL; i++) {
        if (!q36cc_stages(&ch, ch.dm_tabb, i)) continue;
        sg.expect = i; routed_n++;
        ok = q36cc_stage_routed(&ch, &sm, i, 1) && q36cc_fetch_wait(&ch, i) && q36cc_fetch_done(&ch, i);
        const size_t at = (size_t)sg.firstat[i], lo = (size_t)ch.dm_bank[i] * bank, h = (size_t)ch.dm_hostb[(size_t)i * HE + hot].g - (size_t)base;
        routed_at_ok = routed_at_ok && at == h && at >= lo && at + stride <= lo + bank;   /* the routed expert, in its layer's bank */
    }
    sg.routed = 0;
    snprintf(msg, sizeof msg, "%s: a short chunk's routed copy takes its layer's bank, with that bank's events", what);
    ck(ok && routed_n == staged && sg.bad == 0 && sg.fetches == staged && sg.waits == staged && sg.hold[0] < 0 && sg.hold[1] < 0 && routed_at_ok, msg);

    fake_chain_dev_cap = 0;
    cc_free(ch.dm_idsd);
    cc_free(ch.dm_stage); cc_free(ch.dm_tabb); free(ch.dm_hostb); free(ch.dm_partb); free(ch.dm_fro); free(ch.dm_frn);
    free(ch.dm_bank); free(ch.dm_frx); free(ch.dm_used); free(ch.dm_rfr); free(ch.dm_fr); free(ch.dm_host); free(ch.dm_res);
    fake_chain_table = saved;
    qt_source_planar(0);
    qt_shutdown();
}

/* the reference's resident copies, matrix by matrix (sections G and H write the same lines out) */
static void share_trunk(Model *to, const Model *from) {
    for (int i = 0; i < NL; i++) {
        to->L[i].qth_dnout = from->L[i].qth_dnout; to->L[i].qth_q = from->L[i].qth_q; to->L[i].qth_k = from->L[i].qth_k;
        to->L[i].qth_v = from->L[i].qth_v; to->L[i].qth_o = from->L[i].qth_o; to->L[i].qth_shg = from->L[i].qth_shg;
        to->L[i].qth_shu = from->L[i].qth_shu; to->L[i].qth_shd = from->L[i].qth_shd;
    }
}

/* the MTP head of section H (E routed experts, the whole vocabulary), on */
static void mtp_head(void) {
    uint32_t ms = 99;
    Q36Mtp *p = &g_q36_mtp;
    memset(p, 0, sizeof *p);
    p->on = 1; p->E = E; p->K = TOPK; p->I = INTER; p->ish = SI;
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
    memset(&g_q36_mc, 0, sizeof g_q36_mc);
}

#ifdef _WIN32
/* --probe S (--probe-cpu S): a prompt of S rows through step_ex, the MTP head on (the chain's, or the
 * CPU's), and the process's commit around it: its private bytes before and after, their peak, and the
 * fake card's bytes (host memory here, apart). One prompt a process: the peak is the process's. */
static int probe(int S, int cpu) {
    setenv("Q36_KV", "f32", 1); setenv("COLI_CUDA", "1", 1); setenv("COLI_GPUS", "0", 1); setenv("QT_NO_WARMSTART", "1", 1);
    setenv("COLI_LMHEAD_GPU", "0", 1); setenv("COLI_DENSE_IDOT", "0", 1); setenv("COLI_CUDA_CHAIN", "1", 1);
    fake_ndev = 1; fake_dense_compute = 1; fake_chain_table.size = sizeof(ColiCudaChainOps);
    g_cap = S + 16;
    Weights w; make_weights(&w);
    if (!qt_init(NL, E, D, INTER, E, TOPK, 0, 0)) return 1;
    Model m; build_model(&m, &w); place_trunk(&m, &w);
    if (!q36cc_setup(&m, 1, NULL)) return 1;
    g_cuda_chain = !cpu;
    mtp_head();
    int *tk = malloc((size_t)S * sizeof(int));
    for (int s = 0; s < S; s++) tk[s] = (s * 7 + 3) % V;
    PROCESS_MEMORY_COUNTERS_EX a, b;
    memset(&a, 0, sizeof a); memset(&b, 0, sizeof b); a.cb = b.cb = sizeof a;
    GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS *)&a, sizeof a);
    size_t dev0 = fake_chain_dev_bytes;
    free(step_ex(&m, tk, S, 0, 1));
    GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS *)&b, sizeof b);
    printf("PROBE %s S=%d private_before=%zu private_after=%zu peak=%zu peak_rise=%zu dev_before=%zu dev_after=%zu "
           "head_dev_rows=%zu last_n=%d\n", cpu ? "cpu" : "chain", S, (size_t)a.PrivateUsage, (size_t)b.PrivateUsage,
           (size_t)b.PeakPagefileUsage, (size_t)(b.PeakPagefileUsage - a.PrivateUsage), dev0, fake_chain_dev_bytes,
           g_q36_mc.emb ? cc_bytes(g_q36_mc.emb) / (D * sizeof(float)) : (size_t)0, g_q36_mtp.last_n);
    return 0;
}
#endif

/* Shape selection only: handles refer to already resident projections. Setup reads
 * the norms and uploads its arena, but no 35B projection is evaluated. */
static int selector_request, selector_calls;
static int record_selector(int on) {
    if (!selector_calls++) selector_request = on;
    return fcc_select_qwen36_profile(on);
}
static int selector_init_failure(int dev) { (void)dev; return 0; }

static int write_fixture_file(const char *path, const char *contents) {
    FILE *f = fopen(path, "wb");
    if (!f) return 0;
    int ok = fputs(contents, f) >= 0;
    return fclose(f) == 0 && ok;
}

static void test_cuda_model_filter(const Model *trunk) {
    enum Shape { native, gguf, int8_experts, group32, muse, lfm3, width, depth, experts, picks,
                 middle, q_heads, kv_heads, q_width, k_width, mux2, draft, mtp, allowed_env,
                 no_init, no_memory, other_gpu, different_weights, missing_checkpoint };

    char fixture_dir[] = "qwen36-checkpoint-XXXXXX";
    if (!mkdtemp(fixture_dir)) { ck(0, "make checkpoint selector fixture"); return; }
    char weight_path[256], config_path[256], cache_path[256];
    snprintf(weight_path, sizeof weight_path, "%s/model-00000.safetensors", fixture_dir);
    snprintf(config_path, sizeof config_path, "%s/config.json", fixture_dir);
    snprintf(cache_path, sizeof cache_path, "%s/%s", fixture_dir, Q36_CHECKPOINT_CACHE);
    if (!write_fixture_file(weight_path, "abc") || !write_fixture_file(config_path, "{}")) {
        ck(0, "write checkpoint selector fixture");
        return;
    }
    int fd = open(weight_path, COMPAT_O_RDONLY);
    if (fd < 0) { ck(0, "open checkpoint selector fixture"); return; }
#if defined(__linux__) || defined(__APPLE__)
    if (q36_checkpoint_cache_fs(fd)) sleep(2);  /* allow the fixture to qualify for the cache */
#endif
    const Q36CheckpointFile fixture_files[] = {
        {"model-00000.safetensors", 3, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"},
        {"config.json", 2, "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"},
    };
    const Q36Checkpoint fixture_checkpoint = {fixture_dir, fixture_files, 1, 2};
    shards fixture_shards = {.nfd = 1, .fds = {fd}, .paths = {weight_path}, .sizes = {3}};
    const ColiCudaChainOps saved = fake_chain_table;
    const int prev_gguf = g_gguf, prev_int4 = g_expert_is_int4;
    const size_t prev_cap = fake_chain_dev_cap;
    const char *val = getenv("COLI_DFLASH"); char *draft_env = val ? strdup(val) : NULL;
    val = getenv("Q36_MTP"); char *mtp_env = val ? strdup(val) : NULL;
    float norm[2048] = {0};
    Layer layers[40] = {0};
    uint8_t attention[40]; memset(attention, 1, sizeof attention);
    for (int i = 0; i < 40; i++) {
        Layer *l = &layers[i];
        l->qth_q = l->qth_ag = l->qth_ld = l->qth_lu = trunk->L[1].qth_q;
        l->qth_k = trunk->L[1].qth_k; l->qth_v = trunk->L[1].qth_v; l->qth_o = trunk->L[1].qth_o;
        l->in_ln = l->post_ln = l->ra_a = l->ra_b = l->rf_a = l->rf_b = norm;
        l->pre_en = l->post_fn = l->post_op = l->post_ffn = norm;
        l->gate = trunk->L[1].gate;
    }
    fake_chain_table.size = sizeof fake_chain_table;
    fake_chain_table.select_qwen36_profile = record_selector;
    fake_chain_table.muse = fake_chain_table.lfm3 = fcc_ready;
    for (int trial = native; trial <= missing_checkpoint; trial++) {
        Model m = {0}; Cfg *c = &m.c;
        c->hidden = trial == width ? 2016 : 2048; c->n_layers = trial == depth ? 39 : 40;
        c->n_experts = trial == experts ? 255 : 256; c->topk = trial == picks ? 7 : 8;
        c->inter = trial == middle ? 256 : 512; c->expert_gs = trial == group32 ? 32 : 64;
        c->q_heads = trial == q_heads ? 8 : 16; c->kv_heads = trial == kv_heads ? 1 : 2;
        c->head_dim = trial == q_width ? 128 : 256; c->k_head_dim = trial == k_width ? 128 : 256;
        c->q_head_dim = 512; c->is_attn = attention; c->muse = trial == muse; c->lfm3 = trial == lfm3;
        m.S = fixture_shards;
        m.L = layers; m.final_norm = norm; g_gguf = trial == gguf; g_expert_is_int4 = trial != int8_experts;
        if (trial == draft) setenv("COLI_DFLASH", "draft", 1);
        else if (trial == allowed_env) setenv("COLI_DFLASH", "", 1);
        else unsetenv("COLI_DFLASH");
        if (trial == mtp || trial == allowed_env) setenv("Q36_MTP", trial == mtp ? "1" : "2", 1);
        else unsetenv("Q36_MTP");
        fake_chain_table.init = trial == no_init ? selector_init_failure : saved.init;
        fake_chain_dev_cap = trial == no_memory ? fake_chain_dev_bytes + 1 : prev_cap;
        selector_request = -1; selector_calls = 0;
        const int old_device = fake_device_profile;
        if (trial == other_gpu) {
            unlink(cache_path);
            fake_device_profile = Q36_GPU_NONE;
        }
        if (trial == different_weights) ck(write_fixture_file(weight_path, "xyz"), "replace the weights with the same number of bytes");
        Q36CChain *ch = q36cc_setup(&m, trial == mux2 ? 2 : 1,
                                  trial == missing_checkpoint ? NULL : &fixture_checkpoint);
        if (trial == other_gpu) ck(access(cache_path, F_OK) != 0, "another GPU does not verify checkpoint files");
        if (trial == different_weights) ck(write_fixture_file(weight_path, "abc"), "restore checkpoint fixture");
        fake_device_profile = old_device;
        const int early = trial == q_width || trial == k_width || trial == no_init;
        const int enabled = trial == native || trial == allowed_env;
        int ok = early ? (!ch && !selector_calls && !fake_active_profile) :
                 trial == no_memory ? (!ch && selector_request == 1 && selector_calls == 2 && !fake_active_profile) :
                 (ch && (ch->device_profile != 0) == enabled && selector_request == enabled && (fake_active_profile != 0) == enabled);
        char what[136];
        snprintf(what, sizeof what, "Qwen selector trial %d: request, active dispatch and setup result", trial);
        ck(ok, what);
        if (ch) { q36cc_free_own(ch, &m); cc_select_qwen36_profile(0); }
        Q36CChain *a = q36cc_of(&m);
        if (a) {
            void *owned[] = {a->o_in, a->o_post, a->o_qn, a->o_kn, a->o_conv, a->o_dn, a->o_snk, a->o_ra, a->o_rf,
                a->o_pre, a->o_pf, a->t_ag, a->t_ld, a->t_lu, a->t_sci, a->t_sco, a->o_po, a->o_pff, a->t_ab,
                a->t_sg, a->t_gate, a->rgf, a->t_q, a->t_k, a->t_v, a->t_o, a->t_dnp, a->t_dno, a->t_dnz,
                a->t_shg, a->t_shu, a->t_shd, a->rec, a->ring, a->kc, a->vc, a->kv_valid, a->attn_ord, a->kv_rows};
            for (size_t j = 0; j < sizeof owned / sizeof *owned; j++) free(owned[j]);
            free(a);
        }
    }
    struct stat cache_before, cache_after;
#if defined(__linux__) || defined(__APPLE__)
    if (q36_checkpoint_cache_fs(fd)) sleep(2);  /* allow the restored fixture's timestamps to settle */
#endif
    ck(q36_checkpoint_matches(&fixture_shards, &fixture_checkpoint), "verify exact fixture files");
    if (q36_checkpoint_cache_fs(fd)) {
        ck(stat(cache_path, &cache_before) == 0 && q36_checkpoint_matches(&fixture_shards, &fixture_checkpoint) &&
           stat(cache_path, &cache_after) == 0 && cache_before.st_ino == cache_after.st_ino,
           "unchanged verified files reuse the cache without replacing it");
    }
    ck(write_fixture_file(weight_path, "xyz") && !q36_checkpoint_matches(&fixture_shards, &fixture_checkpoint),
       "a same-size weight change invalidates existing cache entries");
#ifndef _WIN32
    ck(unlink(weight_path) == 0 && write_fixture_file(weight_path, "abc"),
       "replace the path with a byte-identical new file");
    ck(!q36_checkpoint_matches(&fixture_shards, &fixture_checkpoint),
       "a replaced path cannot attest to the shard descriptor previously opened");
    close(fd);
    fd = open(weight_path, COMPAT_O_RDONLY);
    fixture_shards.fds[0] = fd;
    ck(q36_checkpoint_matches(&fixture_shards, &fixture_checkpoint),
       "a byte-identical replacement requalifies through a new descriptor");
#endif
    ck(write_fixture_file(config_path, "[]") && !q36_checkpoint_matches(&fixture_shards, &fixture_checkpoint),
       "a same-size configuration change invalidates the checkpoint");
    close(fd);
    unlink(weight_path);
    unlink(config_path);
    unlink(cache_path);
    rmdir(fixture_dir);

    const int old_profile = fake_device_profile;
    const unsigned old_mask = fake_kernel_mask;
    fake_device_profile = Q36_GPU_L4;
    ck(cc_probe_qwen36_profile() == Q36_GPU_L4 && !fake_active_profile && fake_kernel_mask == old_mask,
       "querying an eligible CUDA device leaves the engine on standard dispatch");
    fake_device_profile = Q36_GPU_NONE;
    ck(cc_probe_qwen36_profile() == Q36_GPU_NONE && !fake_active_profile && fake_kernel_mask == old_mask,
       "querying another CUDA device leaves the engine on standard dispatch");
    fake_device_profile = old_profile;
    fake_chain_table.size = offsetof(ColiCudaChainOps, probe_qwen36_profile);
    selector_calls = 0;
    ck(!cc_probe_qwen36_profile() && !selector_calls && !fake_active_profile,
       "a table without a device probe cannot enable a profile");
    fake_chain_table.size = offsetof(ColiCudaChainOps, set_kernel_mask);
    selector_calls = 0;
    ck(!cc_select_qwen36_profile(1) && !selector_calls,
       "table with the selector but without request permissions declines the device profile");
    fake_chain_table.size = sizeof fake_chain_table;
    fake_chain_table.set_kernel_mask = NULL;
    ck(!cc_select_qwen36_profile(1) && !selector_calls,
       "table with a null request permissions callback declines the device profile");

    fake_chain_table = saved; fake_chain_dev_cap = prev_cap; g_gguf = prev_gguf; g_expert_is_int4 = prev_int4;
    if (draft_env) { setenv("COLI_DFLASH", draft_env, 1); free(draft_env); } else unsetenv("COLI_DFLASH");
    if (mtp_env) { setenv("Q36_MTP", mtp_env, 1); free(mtp_env); } else unsetenv("Q36_MTP");
}

/* The lm_head computes on the card and one copy takes its logits to the host's buffer: no matmul of
 * the chain's steps stores through the bus, and the logits equal the CPU path's after a prompt read
 * in chunks, a decode token and a verify whose three logits rows lie in two chunks. */
static void head_logits_on_the_card(Weights *w, const Model *shared, const int *ids) {
    g_q36_mtp.on = 0; memset(&g_q36_mc, 0, sizeof g_q36_mc);   /* section H's MTP head is not part of these steps */
    Model r, m; build_model(&r, w); build_model(&m, w);
    share_trunk(&r, shared); share_trunk(&m, shared);
    Q36CChain *ch = q36cc_setup(&m, 1, NULL);
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

/* a short prompt chunk's copies for one layer (q36cc_routed_fetches): the experts its rows route
 * to that have regions in the staged table, two regions each, in the table's order. Only the
 * list is tested here: the copy itself runs on a card (the T4 bench of the experiment record). */
static void test_routed_fetches(void) {
    enum { RE = 8 };
    CcFetch fr[8], out[2 * RE]; uint8_t used[RE];
    memset(used, 1, sizeof used);   /* the scratch as the layer before left it */
    for (int k = 0; k < 8; k++) fr[k] = (CcFetch){(const void *)(uintptr_t)(0x1000 + k), (size_t)k, (size_t)(100 + k)};
    const int frx[RE] = {-1, 0, -1, 2, 4, -1, 6, -1};   /* 1, 3, 4 and 6 in the pool; the rest resident */
    const int ids[6] = {6, 0, 1, 6, -1, 9};              /* two rows of three: 6 twice, a resident, an empty slot, an id past E */
    int k = q36cc_routed_fetches(ids, 6, RE, frx, fr, used, out);
    ck(k == 4 && out[0].src == fr[0].src && out[1].src == fr[1].src && out[2].src == fr[6].src && out[3].src == fr[7].src,
       "the routed experts in the pool, each once, in the table's order: weights, then scales");
    const int res[3] = {0, 2, 5};
    ck(q36cc_routed_fetches(res, 3, RE, frx, fr, used, out) == 0, "routes only to residents: nothing to copy (the scratch starts clean)");
    const int all[RE] = {7, 6, 5, 4, 3, 2, 1, 0};
    k = q36cc_routed_fetches(all, RE, RE, frx, fr, used, out);
    int whole = k == 8;
    for (int j = 0; j < 8 && whole; j++) whole = out[j].src == fr[j].src && out[j].dst == fr[j].dst && out[j].bytes == fr[j].bytes;
    ck(whole, "routes to every expert: the layer's whole copy, as a longer chunk makes it");
    ck(Q36CC_ROUTED_ROWS > Q36CC_STAGE_ROWS, "a chunk the staging takes can be short enough to copy only its routes");
    ck(q36cc_routes_only(Q36CC_STAGE_ROWS + 1, 4) && q36cc_routes_only(Q36CC_ROUTED_ROWS, 4) &&
       !q36cc_routes_only(Q36CC_ROUTED_ROWS + 1, 4), "an int4 per group chunk copies only its routes up to Q36CC_ROUTED_ROWS rows");
    ck(q36cc_routes_only(Q36CC_STAGE_ROWS + 1, 2), "so does a per-row int4's (the same two regions an expert, planar blocks or pairs)");
    ck(!q36cc_routes_only(Q36CC_STAGE_ROWS + 1, 100) && !q36cc_routes_only(Q36CC_STAGE_ROWS + 1, 142),
       "a GGUF's chunk copies whole layers: no list of its experts' regions");
    ck(!q36cc_routes_only(Q36CC_STAGE_ROWS + 1, 1) && !q36cc_routes_only(Q36CC_STAGE_ROWS + 1, 8),
       "the formats with no staged table (int8, fp8) take no such list either");
    ck(Q36CC_DM_ROWS <= Q36CC_STAGE_ROWS,
       "no staged chunk has a layer the host serves inside the frame: the wait for the routes cannot block");
}

/* Make new mirror storage distinguishable from an explicitly zeroed attention ring. */
static CcBuf *mirror_marked_buf(size_t bytes, int kind) {
    CcBuf *b = fcc_buf(bytes, kind);
    if (b) memset(b->d, 0x3c, b->bytes);
    return b;
}
static int mirror_halves(void) { return 1; }

/* Send a one-token request with a varying generation budget. The empty mux input
 * queue isolates serve_start's control poll from the process's standard input. */
static void host_growth_requests(Weights *w, const Model *shared, int grow_cache, int ring) {
    Model m; build_model(&m, w); share_trunk(&m, shared);
    for (int i = 0; i < NL; i++) { free(m.K[i]); free(m.V[i]); }
    free(m.K); free(m.V); m.K = m.V = NULL; m.max_t = m.kv_cap = 0;
    Q36CChain policy = {0}; policy.device_profile = grow_cache ? Q36_GPU_L4 : Q36_GPU_NONE; m.cchain = &policy;
    uint8_t window[NL] = {0, 1};
    if (ring) { m.c.window = 4; m.c.win_layer = window; }
    const int old_slots = g_q36_mux_slots, old_eof = g_q36_held_eof;
    g_q36_mux_slots = 2; g_q36_held_eof = 1;
    const int needs[] = {5, 1024, 2048, 2049, 4090, 4200};
    /* The prompt cache rounds 4090 to its next 1024-slot block. Both policies cap 4200 at 4097. */
    const int expected[2][6] = {{1024, 1024, 2048, 3072, 4097, 4097}, {2048, 2048, 2048, 3072, 4096, 4097}};
    int ok = 1;
    for (int k = 0; k < 6 && ok; k++) {
        void *old_k = k ? m.K[1] : NULL, *old_v = k ? m.V[1] : NULL;
        int old_cap = m.kv_cap, old_rows = q36_kv_rows(&m, 1);
        if (k) {
            m.kvp.len = 7;
            for (int v = 0; v < 2; v++) for (int h = 0; h < KV; h++) for (int j = 0; j < HD; j++) {
                float *buf = v ? old_v : old_k;
                buf[((size_t)h * old_rows + 6) * HD + j] = (float)(100 * v + 31 * h + j + 7 * k);
            }
        }
        ServeReq req = {0}; strcpy(req.id, "cache-growth"); req.payload = "A"; req.plen = 1; req.max_tok = needs[k] - 1;
        Q36ReqClock clk = {0}; int *tokens = NULL, count = 0; float *logits = NULL;
        ok = q36_serve_start(&m, &req, &tokens, &count, &logits, &clk) && count == 1 && logits &&
             m.kv_cap == expected[grow_cache][k] && m.max_t == m.kv_cap;
        for (int j = 0; ok && j < V; j++) ok = isfinite(logits[j]);
        free(tokens); free(logits);
        int rows = q36_kv_rows(&m, 1);
        if (ok && ring) ok = rows == m.kv_ring && m.kv_ring > 6 && m.kv_ring < m.max_t;
        if (k && ok) {
            if (ring || old_cap == m.kv_cap) ok &= m.K[1] == old_k && m.V[1] == old_v;
            for (int v = 0; v < 2; v++) for (int h = 0; h < KV; h++) for (int j = 0; j < HD; j++) {
                const float *buf = v ? m.V[1] : m.K[1];
                float x = buf[((size_t)h * rows + 6) * HD + j];
                ok &= isfinite(x) && x == (float)(100 * v + 31 * h + j + 7 * k);
            }
        }
    }
    g_q36_mux_slots = old_slots; g_q36_held_eof = old_eof;
    for (int i = 0; i < NL; i++) { free(m.K[i]); free(m.V[i]); }
    free(m.K); free(m.V); free(m.ring_pos); free(m.attn_sc); kv_prefix_free(&m.kvp);
    ck(ok, "served budgets grow or reuse host KV at the expected capacity and preserve per-head K/V, including rings");
}

static void mirror_growth_cases(int half) {
    enum { layers = 3, heads = 2, width = 4, keep = 3, ceiling = 4097 };
    Model m = {0};
    Q36CChain ch = {0};
    uint8_t attn[layers] = {1, 1, 1}, window[layers] = {0, 0, 1};
    CcBuf *keys[layers] = {0}, *values[layers] = {0};
    int rows[layers] = {0}, valid[layers] = {0};
    m.c.n_layers = layers; m.c.kv_heads = heads; m.c.k_head_dim = width;
    m.c.is_attn = attn; m.c.win_layer = window; m.max_t = m.kv_cap = 512;
    ch.kc = keys; ch.vc = values; ch.kv_rows = rows; ch.kv_valid = valid; ch.device_profile = Q36_GPU_L4;
    ColiCudaChainOps saved = fake_chain_table;
    int saved_kv = g_q36_kv16;
    size_t baseline = fake_chain_dev_bytes, es = half ? 2 : 4;
    fake_chain_table.size = sizeof fake_chain_table; fake_chain_table.kvh = mirror_halves;
    fake_chain_table.buf = mirror_marked_buf; g_q36_kv16 = half;
    int ok = q36cc_kves() == es && q36cc_mirror(&ch, &m) && ch.cap == 3072;
    if (!ok) goto done;
    {
        CcBuf *first = keys[0], *last = values[layers - 1];
        int frames = fake_chain_frames;
        size_t held = fake_chain_dev_bytes;
        ok = q36cc_mirror(&ch, &m);
        m.max_t = m.kv_cap = 2000;
        ok &= q36cc_mirror(&ch, &m) && ch.host_cap == 2000 && ch.cap == 3072 &&
              keys[0] == first && values[layers - 1] == last && fake_chain_frames == frames && fake_chain_dev_bytes == held;
        for (int i = 0; i < layers; i++) {
            valid[i] = keep;
            for (int v = 0; v < 2; v++) for (int h = 0; h < heads; h++) for (int j = 0; j < keep * width; j++) {
                CcBuf *buf = v ? values[i] : keys[i];
                int at = h * rows[i] * width + j;
                float x = (float)(1 + i * 400 + v * 100 + h * 20 + j);
                if (half) ((uint16_t *)buf->d)[at] = kv16_from(x); else buf->d[at] = x;
            }
        }
        m.kv_ring = 17; m.max_t = m.kv_cap = 3500;
        size_t old_pair = 2 * heads * 3072 * width * es, new_pair = 2 * heads * ceiling * width * es;
        /* The second layer fits only after the first copy is submitted and its old pair is freed. */
        fake_chain_dev_cap = held + 2 * new_pair - old_pair;
        ok &= q36cc_mirror(&ch, &m) && ch.cap == ceiling && fake_chain_frames == frames + 2;
        fake_chain_dev_cap = 0;
        if (!ok) goto done;
        for (int i = 0; i < layers; i++) {
            int ring = window[i], n = ring ? m.kv_ring * width : keep * width;
            ok &= valid[i] == (ring ? 0 : keep) && rows[i] == (ring ? m.kv_ring : ceiling);
            for (int v = 0; v < 2; v++) for (int h = 0; h < heads; h++) for (int j = 0; j < n; j++) {
                CcBuf *buf = v ? values[i] : keys[i];
                int at = h * rows[i] * width + j;
                float x = half ? kv16_to(((uint16_t *)buf->d)[at]) : buf->d[at];
                ok &= isfinite(x) && x == (ring ? 0.f : (float)(1 + i * 400 + v * 100 + h * 20 + j));
            }
        }
        ch.device_profile = 0;
        ok &= q36cc_mirror(&ch, &m) && ch.cap == 3500 && rows[0] == 3500 && rows[1] == 3500 &&
              rows[2] == m.kv_ring && !valid[0] && !valid[1] && !valid[2];
    }
done:
    fake_chain_dev_cap = 0;
    for (int i = 0; i < layers; i++) { cc_free(keys[i]); cc_free(values[i]); }
    fake_chain_table = saved; g_q36_kv16 = saved_kv;
    ck(ok && fake_chain_dev_bytes == baseline,
       "mirror slack, clamped growth under pressure, per-head K/V, cleared ring and exact size");
}

static void test_step_kernel_mask(Model *m, Model *ref, Q36CChain *ch, const int *ids) {
    const size_t saved_size = fake_chain_table.size;
    fake_chain_table.size = sizeof fake_chain_table;
    ch->device_profile = cc_select_qwen36_profile(1);
    q36cc_end_request(ch);
    const unsigned defaults = fake_kernel_mask;
    for (int rows = 1; rows <= 2; rows++) {
        const int pos = m->kv_len;
        float *logits = step(m, ids + pos, rows, pos);
        float *expected = step_ex(ref, ids + pos, rows, pos, 1);
        m->kv_len = ref->kv_len = pos + rows;
        ck(logits && !ch->request_phase && fake_kernel_mask == defaults,
           "a standalone step releases its request and restores the default kernel mask");
        same_logits(logits, expected, V, "standalone CUDA step agrees with CPU");
        free(logits); free(expected);
    }
    q36cc_begin_request(ch, 0, m->kv_len + 1, 0);
    const unsigned restricted = defaults & ~Q36_PROMPT_CACHE;
    q36cc_set_permissions(ch, restricted);
    const int pos = m->kv_len;
    float *logits = step(m, ids + pos, 1, pos);
    float *expected = step_ex(ref, ids + pos, 1, pos, 1);
    m->kv_len = ref->kv_len = pos + 1;
    ck(logits && ch->request_phase && fake_kernel_mask == restricted,
       "a nested step retains the caller's request and backend permissions");
    same_logits(logits, expected, V, "nested CUDA step agrees with CPU");
    free(logits); free(expected);
    q36cc_end_request(ch);
    ck(!ch->request_phase && fake_kernel_mask == defaults,
       "the caller releases the request kernel mask after the nested step");
    ch->device_profile = cc_select_qwen36_profile(0);
    fake_chain_table.size = saved_size;
}

int main(int argc, char **argv) {
#ifdef _WIN32
    if (argc == 3 && !strcmp(argv[1], "--probe")) return probe(atoi(argv[2]), 0);
    if (argc == 3 && !strcmp(argv[1], "--probe-cpu")) return probe(atoi(argv[2]), 1);
#endif
    (void)argc; (void)argv;
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
    /* the staging's second bank: only where both fit in the 256 experts q36cc_rows_max sets aside for it
     * (a Colab T4 with the residents spread, 91 a layer: two; whole layers, 256: one; 190 a layer: one) */
    ck(q36cc_stage_banks(91, 256) == 2 && q36cc_stage_banks(128, 256) == 2 && q36cc_stage_banks(129, 256) == 1 &&
       q36cc_stage_banks(190, 256) == 1 && q36cc_stage_banks(256, 256) == 1 && q36cc_stage_banks(0, 256) == 1,
       "the staging has two banks where two layers' worth fit in a layer's, else one");
    static Q36CChain zp;   /* zeroed, as a staged table that writes no banks leaves it */
    ck(q36cc_banks(&zp) == 1, "a staged table that sets no banks copies its layers in one");
    zp.dm_banks = 2; ck(q36cc_banks(&zp) == 2, "two banks stay two");
    zp.dm_banks = 1; ck(q36cc_banks(&zp) == 1, "one bank stays one");
    Model ref; build_model(&ref, &w);
    place_trunk(&ref, &w);        /* the reference uses the same resident copies, matrix by matrix */
    Model m; build_model(&m, &w);
    for (int i = 0; i < NL; i++) { m.L[i].qth_dnout = ref.L[i].qth_dnout; m.L[i].qth_q = ref.L[i].qth_q; m.L[i].qth_k = ref.L[i].qth_k; m.L[i].qth_v = ref.L[i].qth_v;
                                   m.L[i].qth_o = ref.L[i].qth_o; m.L[i].qth_shg = ref.L[i].qth_shg; m.L[i].qth_shu = ref.L[i].qth_shu; m.L[i].qth_shd = ref.L[i].qth_shd; }

    test_cuda_model_filter(&ref);

    printf(" A. the chain against the CPU path\n");
    Q36CChain *ch = q36cc_setup(&m, 1, NULL);
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
    test_step_kernel_mask(&m, &ref, ch, ids);

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
      Q36CChain *c2 = q36cc_setup(&m2, 2, NULL);
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
        Q36CChain *c3 = q36cc_setup(&m3, 1, NULL);
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
        Q36CChain *c3 = q36cc_setup(&m3, 1, NULL);
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

    printf(" M. a short prompt chunk copies only the experts its rows route to\n");
    test_routed_fetches();
    printf(" N. the staged table follows the residents when every expert is pooled\n");
    {   /* the ops through dev_addr: the fake's table, tensor_dev and the staging's address, no copy (fetch is not there) */
        fake_chain_table.size = offsetof(ColiCudaChainOps, dev_addr) + sizeof fake_chain_table.dev_addr;
        for (int all = 1; all >= 0; all--) {
            char w[160];
            int regs0 = fake_host_regs, unregs0 = fake_host_unregs;
            qt_source_planar(1);
            ck(qt_init(NL, HE, D, HI, HE, TOPK, 64, 1), "a planar tier");
            for (int e = 0; e < HE; e++) for (int i = 0; i < NL; i++) hp_note(i, e, 0);
            qt_fill_wait();
            for (int i = 0; i < NL; i++) for (int e = 0; e < HE; e++) if (!qt_is_resident(i, e)) hp_note(i, e, 1);
            if (all) qt_chain_host_all();
            Model km; memset(&km, 0, sizeof km);   /* the chunk's dimensions, as the tier's */
            km.c.hidden = D; km.c.inter = HI; km.c.n_layers = NL; km.c.n_experts = HE; km.c.topk = TOPK;
            Q36CChain fc; memset(&fc, 0, sizeof fc);
            fc.rows = 64;
            ck(qt_chain_format(&fc.dm_fmt, &fc.dm_gs) && fc.dm_fmt == 4 && qt_chain_devices() == 1, "the chain reads the tier's format");
            ck(q36cc_dm_begin(&fc, &km, 40), "a chunk of 40 rows: the table built before anything moved");
            q36cc_dm_end(&fc);
            int clean = 1;
            for (int i = 0; i < NL; i++) clean = clean && !fc.dm_partb[i];
            ck(clean && fc.dm_staged == 0 && !fc.dm_stage_off, "before the moves: no layer goes to the host's step");
            for (int t = 0; t < 3; t++) { hp_route(0, 4, 5); hp_route(1, 0, 5); }
            QtPromo pr[8];
            int np = qt_chain_promote(8, pr);
            for (int k = 0; k < np; k++) qt_chain_promote_done(pr[k].ci, pr[k].vi, 1);
            int moved = np > 0;
            for (int round = 0; round < 2; round++) {   /* after the promotions, then after the KV mirrors' give-back */
                if (round) qt_chain_give_back(2 * G.exp_bytes - 1, NULL);
                ck(q36cc_dm_begin(&fc, &km, 40), "a chunk of 40 rows after the moves");
                q36cc_dm_end(&fc);
                int agrees = 1, hostl = 0;   /* the table says what the tier says: resident, staged, or the host's */
                for (int i = 0; i < NL; i++) {
                    hostl += fc.dm_partb[i];
                    for (int e = 0; e < HE; e++) {
                        const int res = qt_is_resident(i, e), pool = hp_pooled(i, e), j = i * HE + e;
                        agrees = agrees && fc.dm_res[j] == res && (fc.dm_hostb[j].g != NULL) == (res || pool);
                    }
                }
                if (all) {
                    snprintf(w, sizeof w, "every expert pooled, %s: the staged table equals the residents, every expert off the card staged, no layer to the host", round ? "after the give-back" : "after the promotions");
                    ck(moved && agrees && hostl == 0 && fc.hp_follows == 0, w);
                } else if (!round) {
                    ck(moved && agrees && hostl > 0 && fc.hp_follows == 0,
                       "the non-residents only (a 40-row chunk does not follow): a layer holding a victim goes to the host's step whole");
                }
            }
            if (all) {
                ck(q36cc_dm_begin(&fc, &km, Q36CC_FOLLOW_ROWS), "a chunk of Q36CC_FOLLOW_ROWS rows");
                q36cc_dm_end(&fc);
                ck(fc.hp_follows == 0 && !fc.dm_partb[0] && !fc.dm_partb[1], "every expert pooled: the follow did nothing, the layers stay on the device");
            }
            cc_free(fc.dm_stage); cc_free(fc.dm_tabb);
            qt_source_planar(0);
            qt_shutdown();
            ck(fake_host_regs - regs0 == fake_host_unregs - unregs0, "the exit releases every page-lock the pool took");
        }
        fake_chain_table.size = CC_OPS_BASE_SIZE;
    }
    printf(" O. a prompt longer than Q36_MTP_FEED_ROWS: the MTP head holds a chunk's rows, its K/V rows as before\n");
    {
        unsetenv("CUDA_EXPERT_GB"); unsetenv("COLI_PLACE"); fake_planar_rows = 0;
        fake_chain_table.size = sizeof(ColiCudaChainOps);
        const int S = Q36_MTP_FEED_ROWS + 300;
        g_cap = S + 16;
        ck(qt_init(NL, E, D, INTER, E, TOPK, 0, 0), "the tier is up for a long context");
        Model mm; build_model(&mm, &w); place_trunk(&mm, &w);
        Q36CChain *cm = q36cc_setup(&mm, 1, NULL);
        ck(cm != NULL && cm->ok, "a chain for the long context");
        g_cuda_chain = 1;
        mtp_head();
        Q36Mtp *p = &g_q36_mtp;
        int *tk = malloc((size_t)S * sizeof(int));
        for (int s = 0; s < S; s++) tk[s] = (s * 7 + 3) % V;
        uint32_t hs = 5;
        float *res = fvec(S * D, &hs, 0.5f, 0.f), *hid = falloc((int64_t)S * D);
        for (int s = 0; s < S; s++) rmsnorm_row(hid + (size_t)s * D, res + (size_t)s * D, mm.final_norm, D, mm.c.eps);
        q36_mtp_feed(&mm, tk, S, 0, res);   /* a forward's residual rows */
        cc_finish();
        ck(g_q36_mc.on && cc_bytes(g_q36_mc.emb) <= (size_t)Q36_MTP_FEED_ROWS * D * sizeof(float),
           "the head's rows on the card: a chunk's, not the prompt's");
        ck(p->last_n == Q36_MTP_FEED_ROWS && p->last_base == S - Q36_MTP_FEED_ROWS && p->has_pend && p->len == S - 1 &&
           !memcmp(p->pend, hid + (size_t)(S - 1) * D, D * sizeof(float)), "the forward's last rows kept, the last one pending");
        ck(q36_mtp_ready(S) && q36_mtp_ready(S - 100) && !q36_mtp_ready(S - Q36_MTP_FEED_ROWS - 10),
           "a later forward from inside the kept rows finds its row there; one from further back waits");
        const size_t kvn = (size_t)KV * g_q36_mc.cap * HD, cut = (size_t)(S - 1) * HD;
        float *kc = malloc(kvn * sizeof(float)), *vc = malloc(kvn * sizeof(float));
        memcpy(kc, g_q36_mc.kc->d, kvn * sizeof(float)); memcpy(vc, g_q36_mc.vc->d, kvn * sizeof(float));
        memset(g_q36_mc.kc->d, 0, kvn * sizeof(float)); memset(g_q36_mc.vc->d, 0, kvn * sizeof(float));
        q36_mtp_rows(&mm, tk + 1, hid, S - 1, 1, NULL, NULL);   /* the prompt's rows in one call, as before */
        cc_finish();
        int same = 0;   /* and not zeros on both sides */
        for (size_t i = 0; i < cut; i++) same |= kc[i] != 0.f;
        for (int h = 0; h < KV; h++) {
            const size_t o = (size_t)h * g_q36_mc.cap * HD;
            same &= !memcmp(kc + o, g_q36_mc.kc->d + o, cut * sizeof(float)) && !memcmp(vc + o, g_q36_mc.vc->d + o, cut * sizeof(float));
        }
        ck(same, "the head's K/V rows of the prompt, a chunk at a time, equal one call's over the whole prompt (bits)");
        free(kc); free(vc); free(res); free(hid); free(tk);

        fake_chain_dev_cap = fake_chain_dev_bytes;   /* the card full: no room for a verify's copies */
        const int slots0 = cm->snap_slots;
        ck(!q36_spec_alloc(&mm, 3) && cm->snap_slots == slots0 && !cm->failed && g_cuda_chain,
           "a full card refuses a verify's copies: no drafts (a plain step on the chain), not a verify on the CPU");
        fake_chain_dev_cap = 0;
        ck(q36_spec_alloc(&mm, 3) && cm->snap_slots >= 3, "with room the copies are made before the drafts");

        /* the CPU's layers (a chain off or lost): a prompt longer than Q36_CPU_ROWS a chunk at a time */
        p->on = 0; g_cuda_chain = 0;
        const int SC = Q36_CPU_ROWS + 300;
        Model ma, mb; build_model(&ma, &w); place_trunk(&ma, &w); build_model(&mb, &w); place_trunk(&mb, &w);
        int *tc = malloc((size_t)SC * sizeof(int));
        for (int s = 0; s < SC; s++) tc[s] = (s * 5 + 1) % V;
        float *la = step_ex(&ma, tc, SC, 0, 1);
        float *xb = falloc((int64_t)SC * D), last[D], lb[V];
        for (int s = 0; s < SC; s++) q36_embed_row(&mb, tc[s], s, xb + (size_t)s * D);
        layers_forward_range(&mb, xb, SC, 0, 0, NL, 1, NULL);   /* the prompt whole, as before */
        rmsnorm_row(last, xb + (size_t)(SC - 1) * D, mb.final_norm, D, mb.c.eps);
        if (!qt_lmhead_matmul(lb, last, D, V)) matmul_d(lb, last, &mb.lm_head, 1, D, V);
        q36_logits_out(&mb.c, lb, V);
        int kv_same = ma.kv_cap == mb.kv_cap;   /* a head's rows start kv_cap rows apart (ensure_kv rounds it up from g_cap) */
        for (int h = 0; kv_same && h < KV; h++) {
            const size_t o = (size_t)h * ma.kv_cap * HD, n = (size_t)SC * HD * sizeof(float);
            kv_same &= !memcmp((float *)ma.K[1] + o, (float *)mb.K[1] + o, n) && !memcmp((float *)ma.V[1] + o, (float *)mb.V[1] + o, n);
        }
        ck(!memcmp(la, lb, sizeof lb) && kv_same && !memcmp(ma.DN_rec[0], mb.DN_rec[0], (size_t)VH * KD * VD * sizeof(float)) &&
           !memcmp(ma.DN_conv[0], mb.DN_conv[0], (size_t)CD * (CONVK - 1) * sizeof(float)),
           "the CPU's layers a chunk at a time: the prompt's last logits, K/V rows and DeltaNet state as one pass's (bits)");
        free(la); free(xb); free(tc);
        g_cap = CAP;
        fake_chain_table.size = CC_OPS_BASE_SIZE;
        qt_shutdown();
    }
    printf(" P. the staged table: the host pool's experts in the staging's banks\n");
    stage_table_case(4, 1, -1, 0, 1, 2, "planar, half the experts in the pool (two layers' worth fit in a layer's)");
    stage_table_case(4, 0, -1, 0, 1, 2, "the residents' pairs, the same");
    stage_table_case(4, 1, 2, 0, 1, 2, "planar, a layer with an expert in neither");
    stage_table_case(3, 1, -1, 0, 1, 1, "planar, five of eight in the pool (two would not fit in a layer's)");
    stage_table_case(4, 1, -1, 40000, 1, 1, "planar, room for one bank only");
    stage_table_case(4, 1, -1, 0, 0, 1, "planar, a backend whose table ends before the banks' ops");

    printf(" Q. warmstart pre-allocates scratch and mirror growth preserves valid KV rows in VRAM\n");
    {
        unsetenv("CUDA_EXPERT_GB"); unsetenv("COLI_PLACE"); fake_planar_rows = 0;
        fake_chain_table.size = CC_OPS_BASE_SIZE;   /* this fixture keeps its routed experts on the host */
        ck(qt_init(NL, E, D, INTER, E, TOPK, 0, 0), "tier up for warmstart and mirror growth");
        Model mw, mr; build_model(&mw, &w); build_model(&mr, &w);
        place_trunk(&mr, &w); share_trunk(&mw, &mr);
        Q36CChain *cw = q36cc_setup(&mw, 1, NULL);
        ck(cw != NULL && cw->ok, "chain setup for warmstart");
        g_cuda_chain = 1;
        cw->device_profile = fcc_select_qwen36_profile(1);   /* exercise the cache growth policy on the tiny model */
        q36cc_warmstart(&mw);
        ck(cw->cap >= Q36CC_CONV_ROWS && cw->rows > 0, "q36cc_warmstart pre-allocates KV mirrors and prompt scratch");
        float *lr = step_ex(&mr, ids, 4, 0, 1), *lw = step_ex(&mw, ids, 4, 0, 1);
        mr.kv_len = mw.kv_len = 4; mw.kvp.len = mr.kvp.len = 4;
        same_logits(lw, lr, V, "warmstarted chain prompt logits match CPU");
        free(lr); free(lw);
        /* Grow host kv_cap past cw->cap (3072): q36cc_mirror must copy the 4 valid rows in VRAM in 1 frame and keep kv_valid[1] == 4 */
        mr.max_t = mw.max_t = 4096;
        ensure_kv(&mr); ensure_kv(&mw);
        int f0 = fake_chain_frames;
        ck(q36cc_mirror(cw, &mw) && cw->cap >= 4096 && cw->kv_valid[1] == 4 && fake_chain_frames - f0 == 1,
           "q36cc_mirror grows in VRAM in a single batched frame and preserves kv_valid[1] == 4 without host re-upload");
        lr = step_ex(&mr, ids + 4, 2, 4, 1); lw = step_ex(&mw, ids + 4, 2, 4, 1);
        same_logits(lw, lr, V, "continuation after VRAM mirror growth matches CPU");
        free(lr); free(lw);
        {
            const char *limit = getenv("Q36_MAXT");
            char *saved = limit ? strdup(limit) : NULL;
            setenv("Q36_MAXT", "4097", 1);
            for (int mode = 0; mode < 2; mode++)
                for (int ring = 0; ring < 2; ring++) host_growth_requests(&w, &mr, mode, ring);
            mirror_growth_cases(0);
            mirror_growth_cases(1);
            if (saved) { setenv("Q36_MAXT", saved, 1); free(saved); } else unsetenv("Q36_MAXT");
        }

        /* Corrupt one preserved device KV entry and check that the continuation still reads it. */
        float saved = cw->kc[1]->d[0];
        cw->kc[1]->d[0] += 100.0f;
        lw = step_ex(&mw, ids + 4, 2, 4, 1);
        lr = step_ex(&mr, ids + 4, 2, 4, 1);
        ck(maxdiff(lw, lr, V) > 1e-3, "corrupting preserved device KV row 0 changes continuation logits without a host re-upload");
        cw->kc[1]->d[0] = saved;
        free(lr); free(lw);
        g_cuda_chain = 0;
        qt_shutdown();
        fake_chain_table.size = CC_OPS_BASE_SIZE;
    }
    if (fails) { printf("test_qwen36_cuda_chain: %d failure(s)\n", fails); return 1; }
    printf("OK test_qwen36_cuda_chain: the CUDA chain gives the CPU path's logits; host and device state agree; a lost device rebuilds; the host pool follows before a large prompt chunk only; the promotions' brake only with a history; the MTP head and the CPU's layers read a long prompt a chunk at a time\n");
    return 0;
}
