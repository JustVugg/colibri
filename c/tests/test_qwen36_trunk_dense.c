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
enum { NL = 2, D = 48, VH = 2, VD = 8, QH = 2, QD = 12, KVH = 1, KD = 8, SH = 20, NE = 4, IH = 16 };

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
static double gemv_gap(int hp1, const float *W, int I, int O, int *served) {
    float *x = rnd_matrix(1, I), *ya = calloc((size_t)O, sizeof(float)), *yb = calloc((size_t)O, sizeof(float));
    *served = qtd(hp1, ya, x, I, O);
    matmul_d(yb, x, W, 1, I, O);
    double worst = 0, scale = 1e-6;
    for (int o = 0; o < O; o++) {
        double d = fabs((double)ya[o] - yb[o]);
        if (d > worst) worst = d;
        if (fabs((double)yb[o]) > scale) scale = fabs((double)yb[o]);
    }
    free(x); free(ya); free(yb);
    return worst / scale;
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
    dn->dn_out = rnd_matrix(D, VH * VD);          qdw_register(dn->dn_out, VH * VD, D);
    dn->sh_g = rnd_matrix(SH, D);                 qdw_register(dn->sh_g, D, SH);
    dn->sh_u = rnd_matrix(SH, D);                 qdw_register(dn->sh_u, D, SH);
    dn->sh_d = rnd_matrix(D, SH);                 qdw_register(dn->sh_d, SH, D);
    at->q = rnd_matrix(QH * QD, D);               qdw_register(at->q, D, QH * QD);
    at->k = rnd_matrix(KVH * KD, D);              qdw_register(at->k, D, KVH * KD);
    at->v = rnd_matrix(KVH * KD, D);              qdw_register(at->v, D, KVH * KD);
    at->o = rnd_matrix(D, QH * KD);               qdw_register(at->o, QH * KD, D);
    /* the attention layer's shared expert is INCOMPLETE on purpose: gate and
     * up have a dense-i8 copy, down does not (never registered), so "shexp"
     * for layer 1 must not be offered and its three handles must stay 0 */
    at->sh_g = rnd_matrix(SH, D);                 qdw_register(at->sh_g, D, SH);
    at->sh_u = rnd_matrix(SH, D);                 qdw_register(at->sh_u, D, SH);
    at->sh_d = rnd_matrix(D, SH);                 /* no qdw_register */
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
    size_t want_attn = qdw_bytes(at->q) + qdw_bytes(at->k) + qdw_bytes(at->v) + qdw_bytes(at->o);
    int seen_attn = 0, seen_dnout = 0, seen_shexp1 = 0;
    for (int o = before; o < G_offer_n; o++) {
        if (!strcmp(G_offer[o].name, "attnproj") && G_offer[o].layer == 1 && G_offer[o].bytes == want_attn) seen_attn = 1;
        if (!strcmp(G_offer[o].name, "dnout") && G_offer[o].layer == 0 && G_offer[o].bytes == qdw_bytes(dn->dn_out)) seen_dnout = 1;
        if (!strcmp(G_offer[o].name, "shexp") && G_offer[o].layer == 1) seen_shexp1 = 1;
    }
    ck(seen_dnout, "dnout offered for the DeltaNet layer with the bytes of its dense-i8 copy");
    ck(seen_attn, "attnproj offered for the attention layer as the sum of q, k, v, o");
    ck(!seen_shexp1, "a shared expert missing one dense-i8 copy is not offered");
    ck(qdw_bytes(at->sh_d) == 0, "a matrix without a dense-i8 copy reports zero bytes");

    printf("placement\n");
    ck(qt_init(NL, NE, D, IH, NE, 1, 0, 1), "tier starts (int4 mode, cap == n_experts)");
    ck(qt_place_of("dnout", 0) == 0 && qt_place_of("attnproj", 1) == 0 && qt_place_of("shexp", 0) == 0,
       "every offered component placed on the one device with room");
    ck(qt_place_of("shexp", 1) == QT_PLACE_CPU, "the component never offered stays on the CPU");
    double vram = 0;
    int placed = trunk_place_dense(&m, &vram);
    ck(placed == 8, "eight matrices placed: dnout, q k v o, and the complete shared expert");
    ck(vram == (double)(qdw_bytes(dn->dn_out) + want_attn + qdw_bytes(dn->sh_g) + qdw_bytes(dn->sh_u) + qdw_bytes(dn->sh_d)),
       "placed bytes are the sum of the dense-i8 copies uploaded");
    ck(fake_uploads == 8, "one upload per placed matrix");
    ck(dn->qth_dnout > 0 && at->qth_q > 0 && at->qth_k > 0 && at->qth_v > 0 && at->qth_o > 0, "handles kept in the Layer");
    ck(dn->qth_shg > 0 && dn->qth_shu > 0 && dn->qth_shd > 0, "shared expert handles kept in the DeltaNet layer");
    ck(at->qth_shg == 0 && at->qth_shu == 0 && at->qth_shd == 0, "no handle for the shared expert that was not offered");

    printf("GEMV from VRAM == matmul_d on the CPU\n");
    struct { const char *name; int hp1; const float *w; int I, O; } mats[] = {
        {"dn_out", dn->qth_dnout, dn->dn_out, VH * VD, D},
        {"q", at->qth_q, at->q, D, QH * QD}, {"k", at->qth_k, at->k, D, KVH * KD},
        {"v", at->qth_v, at->v, D, KVH * KD}, {"o", at->qth_o, at->o, QH * KD, D},
        {"sh_g", dn->qth_shg, dn->sh_g, D, SH}, {"sh_u", dn->qth_shu, dn->sh_u, D, SH}, {"sh_d", dn->qth_shd, dn->sh_d, SH, D},
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

    qt_shutdown();
    if (fails) { printf("test_qwen36_trunk_dense: %d failure(s)\n", fails); return 1; }
    printf("OK test_qwen36_trunk_dense: dnout, attnproj and shexp offered, placed and served from VRAM\n");
    return 0;
}
