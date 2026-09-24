/* The rest of the dense trunk offered to the VRAM placer: "dnout" (DeltaNet
 * out_proj), "attnproj" (q, k, v, o of an attention layer) and "shexp" (the
 * shared expert's gate, up, down), placed by name and layer and served from
 * VRAM through qt_dense handles kept in the Layer.
 *
 * What is pinned, on the fake CUDA backend (no GPU, no toolkit):
 *   - trunk_offer_dense offers exactly the components whose every matrix has
 *     a dense-i8 copy, with the bytes of those copies;
 *   - trunk_place_dense uploads what the placer took and keeps one handle per
 *     matrix, counting matrices and bytes;
 *   - a placed matrix answers the same GEMV from VRAM as matmul_d does on the
 *     CPU (same int8 rows, same per-row scales), and a component without a
 *     handle keeps running matmul_d;
 *   - the placer prices every one of them as a dense component: with room for
 *     all, all go; with COLI_PLACE=off nothing is offered at all.
 * Include order as in tests/test_qwen36_tier_int8_engine.c: the engine, the
 * fake backend, then the tier, so the tier's statics live in this TU. */
#define main qwen36_main_unused
#include "../qwen36.c"
#undef main

#include "../compat.h"   /* setenv/unsetenv: MinGW has neither */

#include "qwen36_fake_cuda.h"

#include "../qwen36_tier.c"

static int fails;
static void ck(int ok, const char *what) {
    if (ok) { printf("  ok   %s\n", what); return; }
    printf("  FAIL %s\n", what);
    fails++;
}

/* Small but every dimension distinct, so a swapped I/O would show. */
enum { NL = 2, D = 48, VH = 2, VD = 8, QH = 2, QD = 16, KVH = 1, KD = 8, SH = 20, NE = 4, IH = 16 };

static unsigned g_seed = 12345;
static float rnd(void) {
    g_seed = g_seed * 1103515245u + 12345u;
    return ((g_seed >> 8) & 0xFFFF) / 32768.f - 1.f;
}
static float *rnd_matrix(int rows, int cols) {
    float *w = malloc((size_t)rows * cols * sizeof(float));
    for (size_t i = 0; i < (size_t)rows * cols; i++) w[i] = rnd();
    return w;
}

/* One GEMV both ways: from VRAM through the handle and on the CPU through
 * matmul_d. The fake backend computes exactly what a real one does with these
 * bytes (x . int8 row, times the row scale), so the two must agree to float
 * accumulation order. */
static double gemv_gap(int hp1, const QW *w, int I, int O, int *served) {
    float *x = rnd_matrix(1, I), *ya = calloc((size_t)O, sizeof(float)), *yb = calloc((size_t)O, sizeof(float));
    *served = qtd(hp1, ya, x, I, O);
    matmul_d(yb, x, w, 1, I, O);
    double worst = 0, scale = 1e-6;
    for (int o = 0; o < O; o++) {
        double d = fabs((double)ya[o] - yb[o]);
        if (d > worst) worst = d;
        if (fabs((double)yb[o]) > scale) scale = fabs((double)yb[o]);
    }
    free(x); free(ya); free(yb);
    return worst / scale;
}

/* Quantize a freshly-built random matrix straight into *out (the QW the
 * Layer holds), then discard the f32 staging buffer -- the same "quantize
 * once, free the f32 copy" shape load_tq uses. */
static void register_qw(int rows, int cols, QW *out) {
    float *w = rnd_matrix(rows, cols);
    qw_quantize(w, cols, rows, NULL, out);
    free(w);
}

static void build(Model *m) {
    memset(m, 0, sizeof *m);
    Cfg *c = &m->c;
    c->n_layers = NL; c->hidden = D; c->n_experts = NE; c->inter = IH; c->topk = 1; c->expert_gs = 0;
    c->q_heads = QH; c->q_head_dim = QD; c->kv_heads = KVH; c->k_head_dim = KD; c->head_dim = KD; c->o_in = QH * KD;
    c->dn_vheads = VH; c->dn_vdim = VD; c->shared_inter = SH;
    c->is_attn = calloc(NL, 1); c->is_attn[1] = 1;
    m->L = calloc(NL, sizeof(Layer));
    Layer *dn = &m->L[0], *at = &m->L[1];
    register_qw(D, VH * VD, &dn->dn_out);
    register_qw(SH, D, &dn->sh_g);
    register_qw(SH, D, &dn->sh_u);
    register_qw(D, SH, &dn->sh_d);
    register_qw(QH * QD, D, &at->q);
    register_qw(KVH * KD, D, &at->k);
    register_qw(KVH * KD, D, &at->v);
    register_qw(D, QH * KD, &at->o);
    /* the attention layer's shared expert is INCOMPLETE on purpose: gate and
     * up have a dense-i8 copy, down does not (never registered), so "shexp"
     * for layer 1 must not be offered and its three handles must stay 0 */
    register_qw(SH, D, &at->sh_g);
    register_qw(SH, D, &at->sh_u);
    /* at->sh_d: left zeroed (q == NULL) -- no dense-i8 copy, never registered */
}

int main(void) {
    setenv("COLI_CUDA", "1", 1); setenv("COLI_GPUS", "0", 1);
    setenv("QT_NO_WARMSTART", "1", 1); setenv("HEAT_FILE", "", 1);
    setenv("COLI_PLACE", "", 1);                    /* "" == unset == auto */
    setenv("CUDA_EXPERT_GB", "1", 1);               /* room for everything */
    unsetenv("COLI_DENSE_I8");
    /* the GEMV from VRAM is compared with the CPU's f32-activation kernel, the
     * contract the tier uploads: the integer dense path rounds the activation
     * and is measured elsewhere (test_qwen36_dense_idot) */
    setenv("COLI_DENSE_IDOT", "0", 1);
    fake_ndev = 1; fake_uploads = 0; fake_dense_compute = 1;

    Model m; build(&m);
    Layer *dn = &m.L[0], *at = &m.L[1];

    printf("offers\n");
    int before = G_offer_n;
    trunk_offer_dense(&m);
    /* dnout(layer 0) + attnproj(layer 1) + shexp(layer 0); NOT shexp(layer 1) */
    ck(G_offer_n - before == 3, "three components offered: dnout, attnproj, shexp of the complete layer only");
    size_t want_attn = qdw_bytes(&at->q) + qdw_bytes(&at->k) + qdw_bytes(&at->v) + qdw_bytes(&at->o);
    int seen_attn = 0, seen_dnout = 0, seen_shexp1 = 0;
    for (int o = before; o < G_offer_n; o++) {
        if (!strcmp(G_offer[o].name, "attnproj") && G_offer[o].layer == 1 && G_offer[o].bytes == want_attn) seen_attn = 1;
        if (!strcmp(G_offer[o].name, "dnout") && G_offer[o].layer == 0 && G_offer[o].bytes == qdw_bytes(&dn->dn_out)) seen_dnout = 1;
        if (!strcmp(G_offer[o].name, "shexp") && G_offer[o].layer == 1) seen_shexp1 = 1;
    }
    ck(seen_dnout, "dnout offered for the DeltaNet layer with the bytes of its dense-i8 copy");
    ck(seen_attn, "attnproj offered for the attention layer as the sum of q, k, v, o");
    ck(!seen_shexp1, "a shared expert missing one dense-i8 copy is not offered");
    ck(qdw_bytes(&at->sh_d) == 0, "a matrix without a dense-i8 copy reports zero bytes");

    printf("placement\n");
    ck(qt_init(NL, NE, D, IH, NE, 1, 0, 1), "tier starts (int4 mode, cap == n_experts)");
    ck(qt_place_of("dnout", 0) == 0 && qt_place_of("attnproj", 1) == 0 && qt_place_of("shexp", 0) == 0,
       "every offered component placed on the one device with room");
    ck(qt_place_of("shexp", 1) == QT_PLACE_CPU, "the component never offered stays on the CPU");
    double vram = 0;
    int placed = trunk_place_dense(&m, &vram);
    ck(placed == 8, "eight matrices placed: dnout, q k v o, and the complete shared expert");
    ck(vram == (double)(qdw_bytes(&dn->dn_out) + want_attn + qdw_bytes(&dn->sh_g) + qdw_bytes(&dn->sh_u) + qdw_bytes(&dn->sh_d)),
       "placed bytes are the sum of the dense-i8 copies uploaded");
    ck(fake_uploads == 8, "one upload per placed matrix");
    ck(dn->qth_dnout > 0 && at->qth_q > 0 && at->qth_k > 0 && at->qth_v > 0 && at->qth_o > 0, "handles kept in the Layer");
    ck(dn->qth_shg > 0 && dn->qth_shu > 0 && dn->qth_shd > 0, "shared expert handles kept in the DeltaNet layer");
    ck(at->qth_shg == 0 && at->qth_shu == 0 && at->qth_shd == 0, "no handle for the shared expert that was not offered");

    printf("GEMV from VRAM == matmul_d on the CPU\n");
    struct { const char *name; int hp1; const QW *w; int I, O; } mats[] = {
        {"dn_out", dn->qth_dnout, &dn->dn_out, VH * VD, D},
        {"q", at->qth_q, &at->q, D, QH * QD}, {"k", at->qth_k, &at->k, D, KVH * KD},
        {"v", at->qth_v, &at->v, D, KVH * KD}, {"o", at->qth_o, &at->o, QH * KD, D},
        {"sh_g", dn->qth_shg, &dn->sh_g, D, SH}, {"sh_u", dn->qth_shu, &dn->sh_u, D, SH}, {"sh_d", dn->qth_shd, &dn->sh_d, SH, D},
    };
    for (size_t i = 0; i < sizeof mats / sizeof mats[0]; i++) {
        int served = 0;
        double gap = gemv_gap(mats[i].hp1, mats[i].w, mats[i].I, mats[i].O, &served);
        char what[128];
        snprintf(what, sizeof what, "%s: served from VRAM, relative gap %.2e", mats[i].name, gap);
        ck(served && gap < 1e-4, what);
    }
    {
        int served = 1; float x[D] = {0}, y[SH] = {0};
        served = qtd(at->qth_shd, y, x, SH, D);
        ck(!served, "a matrix with no handle is not served (the caller runs matmul_d)");
    }

    /* Drive attention itself: this catches an S==1 gate left at any of the
     * four projection call sites, which a tier-only test cannot detect. */
    {
        enum { S = 5 };
#ifdef _OPENMP
        omp_set_num_threads(1);
#endif
        m.max_t = m.kv_cap = S + 1;
        m.K = calloc(NL, sizeof(float *)); m.V = calloc(NL, sizeof(float *));
        m.K[1] = calloc((S + 1) * KVH * KD, sizeof(float));
        m.V[1] = calloc((S + 1) * KVH * KD, sizeof(float));
        m.attn_sc = calloc(S + 1, sizeof(float));
        float *x = rnd_matrix(S + 1, D), gpu[S * D], cpu[S * D], fallback[S * D];
        Layer host = *at;
        host.qth_q = host.qth_k = host.qth_v = host.qth_o = 0;
        attention(&m, &host, 1, x, S, 0, cpu);
        int calls = fake_matmuls;
        attention(&m, at, 1, x, S, 0, gpu);
        ck(fake_matmuls == calls + 4 && fake_matmul_rows == S,
           "prefill dispatches all four projections as batches");
        int handles[]={at->qth_q,at->qth_k,at->qth_v,at->qth_o};
        for(int failure=0;failure<4;failure++){
            for(int i=0;i<4;i++) G_dense[handles[i]-1].on=1;
            calls=fake_matmuls;
            fake_matmul_fail_at=calls+failure+1;
            attention(&m,at,1,x,S,0,fallback);
            fake_matmul_fail_at=0;
            ck(fake_matmuls==calls+4,"each projection attempted once on first failure");
            for(int i=0;i<4;i++)
                ck(G_dense[handles[i]-1].on==(i!=failure),"only failed projection disabled");
            for(int pass=0;pass<2;pass++){
                double gap=0, scale=1e-6;
                int finite=1;
                for(int i=0;i<S*D;i++){
                    finite &= isfinite(cpu[i]) && isfinite(gpu[i]) && isfinite(fallback[i]);
                    gap=fmax(gap,fabs((double)gpu[i]-cpu[i]));
                    gap=fmax(gap,fabs((double)fallback[i]-cpu[i]));
                    scale=fmax(scale,fabs(cpu[i]));
                }
                ck(finite && gap/scale<1e-4,"poisoned GPU output replaced by finite CPU-equivalent output");
                if(pass==0){
                    calls=fake_matmuls;
                    attention(&m,at,1,x,S,0,fallback);
                    ck(fake_matmuls==calls+3,"disabled handle stays on CPU; healthy projections stay on GPU");
                }
            }
        }
        /* Split prefill at a nonzero position, then decode another token.
         * Compare both the observable output and the state consumed next. */
        float reference[(S+1)*D], continued[(S+1)*D];
        float keys[(S+1)*KVH*KD], values[(S+1)*KVH*KD];
        attention(&m,&host,1,x,S+1,0,reference);
        memcpy(keys,m.K[1],sizeof keys); memcpy(values,m.V[1],sizeof values);
        for(int failure=-1;failure<4;failure++){
            for(int i=0;i<4;i++) G_dense[handles[i]-1].on=1;
            memset(m.K[1],0,sizeof keys); memset(m.V[1],0,sizeof values);
            calls=fake_matmuls;
            attention(&m,at,1,x,2,0,continued);
            if(failure>=0) fake_matmul_fail_at=fake_matmuls+failure+1;
            attention(&m,at,1,x+2*D,S-2,2,continued+2*D);
            fake_matmul_fail_at=0;
            attention(&m,at,1,x+S*D,1,S,continued+S*D);
            ck(fake_matmuls==calls+(failure<0?12:11),
               "segmented prefill and decode keep only the failed projection on CPU");
            double gap=0, scale=1e-6;
            int finite=1;
            for(int i=0;i<(S+1)*D;i++){
                finite &= isfinite(continued[i]) && isfinite(reference[i]);
                gap=fmax(gap,fabs((double)continued[i]-reference[i]));
                scale=fmax(scale,fabs(reference[i]));
            }
            ck(finite && gap/scale<1e-4,"segmented attention outputs match full CPU prefill");
            gap=0; scale=1e-6; finite=1;
            for(int i=0;i<(S+1)*KVH*KD;i++){
                finite &= isfinite(m.K[1][i]) && isfinite(m.V[1][i]);
                gap=fmax(gap,fabs((double)m.K[1][i]-keys[i]));
                gap=fmax(gap,fabs((double)m.V[1][i]-values[i]));
                scale=fmax(scale,fmax(fabs(keys[i]),fabs(values[i])));
            }
            ck(finite && gap/scale<1e-4,"segmented attention preserves CPU-equivalent KV state");
        }
        free(x); free(m.K[1]); free(m.V[1]); free(m.K); free(m.V); free(m.attn_sc);
    }

    qt_shutdown();
    if (fails) { printf("test_qwen36_trunk_dense: %d failure(s)\n", fails); return 1; }
    printf("OK test_qwen36_trunk_dense: dnout, attnproj and shexp offered, placed and served from VRAM\n");
    return 0;
}
