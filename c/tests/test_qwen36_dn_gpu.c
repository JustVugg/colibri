/* The DeltaNet layer on the GPU (Q36_DN_GPU=1) computes what the CPU path
 * computes, and the host state and the device state never disagree.
 *
 * One in-memory DeltaNet layer (no container), driven through deltanet() the
 * way step() drives it, against the fake CUDA backend whose coli_cuda_dn_step
 * is a host-side reference of the same arithmetic. The properties:
 *   - decode tokens through the GPU path give the CPU path's outputs;
 *   - the host copy is not touched while the card advances (dn_host_stale),
 *     and reads back exactly the state the CPU path would have;
 *   - a CPU step after GPU steps (a prefill, S > 1) pulls the state first and
 *     continues from it; a GPU step after a CPU step pushes it first;
 *   - a failing GPU step turns the layer off and the CPU continues from the
 *     state the card holds; reset_recurrent invalidates the device copy.
 * Include order as in test_qwen36_tier_int8_engine.c. */
#define main qwen36_main_unused
#include "../qwen36.c"
#undef main

#include "../compat.h"
#include "qwen36_fake_cuda.h"
#include "../qwen36_tier.c"

static int fails;
static void ck(int ok, const char *what) { if (ok) { printf("  ok   %s\n", what); return; } printf("  FAIL %s\n", what); fails++; }

enum { D = 64, VK = 2, VH = 4, KD = 16, VD = 16, CONVK = 4 };
enum { CONV_DIM = 2 * VK * KD + VH * VD, VALUE_DIM = VH * VD, PROJ_DIM = CONV_DIM + VALUE_DIM, T = 6 };

static float frand(uint32_t *s) { *s = *s * 1664525u + 1013904223u; return ((*s >> 8) & 0xFFFF) / 65535.f * 2.f - 1.f; }

typedef struct { int8_t *qkv, *z, *o; float *sqkv, *sz, *so, *b, *a, *conv, *alog, *dtb, *norm; } Weights;

static void make_weights(Weights *w) {
    uint32_t s = 7;
    w->qkv = malloc((size_t)CONV_DIM * D); w->z = malloc((size_t)VALUE_DIM * D); w->o = malloc((size_t)D * VALUE_DIM);
    for (size_t i = 0; i < (size_t)CONV_DIM * D; i++) w->qkv[i] = (int8_t)(frand(&s) * 60);
    for (size_t i = 0; i < (size_t)VALUE_DIM * D; i++) w->z[i] = (int8_t)(frand(&s) * 60);
    for (size_t i = 0; i < (size_t)D * VALUE_DIM; i++) w->o[i] = (int8_t)(frand(&s) * 60);
    w->sqkv = malloc(CONV_DIM * sizeof(float)); w->sz = malloc(VALUE_DIM * sizeof(float)); w->so = malloc(D * sizeof(float));
    for (int i = 0; i < CONV_DIM; i++) w->sqkv[i] = 0.02f + 0.01f * frand(&s);
    for (int i = 0; i < VALUE_DIM; i++) w->sz[i] = 0.02f + 0.01f * frand(&s);
    for (int i = 0; i < D; i++) w->so[i] = 0.02f + 0.01f * frand(&s);
    w->b = malloc((size_t)VH * D * sizeof(float)); w->a = malloc((size_t)VH * D * sizeof(float));
    for (int i = 0; i < VH * D; i++) { w->b[i] = 0.1f * frand(&s); w->a[i] = 0.1f * frand(&s); }
    w->conv = malloc((size_t)CONV_DIM * CONVK * sizeof(float));
    for (int i = 0; i < CONV_DIM * CONVK; i++) w->conv[i] = 0.5f * frand(&s);
    w->alog = malloc(VH * sizeof(float)); w->dtb = malloc(VH * sizeof(float)); w->norm = malloc(VD * sizeof(float));
    for (int h = 0; h < VH; h++) { w->alog[h] = -1.f + 0.5f * frand(&s); w->dtb[h] = 0.2f * frand(&s); }
    for (int i = 0; i < VD; i++) w->norm[i] = 1.f + 0.2f * frand(&s);
}

static void build_model(Model *m, Layer *l, Weights *w) {
    memset(m, 0, sizeof *m); memset(l, 0, sizeof *l);
    m->c.n_layers = 1; m->c.hidden = D; m->c.eps = 1e-6f;
    m->c.dn_vheads = VH; m->c.dn_kheads = VK; m->c.dn_kdim = KD; m->c.dn_vdim = VD; m->c.dn_convk = CONVK; m->c.dn_conv_dim = CONV_DIM;
    m->c.is_attn = calloc(1, 1);
    m->DN_rec = calloc(1, sizeof(float *)); m->DN_conv = calloc(1, sizeof(float *));
    m->DN_rec[0] = calloc((size_t)VH * KD * VD, sizeof(float)); m->DN_conv[0] = calloc((size_t)CONV_DIM * (CONVK - 1), sizeof(float));
    m->dn_dev_fresh = calloc(1, 1); m->dn_host_stale = calloc(1, 1);
    l->dn_qkv = (QW){ .q = w->qkv, .sc = w->sqkv, .I = D, .O = CONV_DIM };
    l->dn_z   = (QW){ .q = w->z,   .sc = w->sz,   .I = D, .O = VALUE_DIM };
    l->dn_out = (QW){ .q = w->o,   .sc = w->so,   .I = VALUE_DIM, .O = D };
    l->dn_b = w->b; l->dn_a = w->a; l->dn_conv = w->conv; l->dn_alog = w->alog; l->dn_dtbias = w->dtb; l->dn_norm = w->norm;
}
static void free_model(Model *m) { free(m->c.is_attn); free(m->DN_rec[0]); free(m->DN_conv[0]); free(m->DN_rec); free(m->DN_conv); free(m->dn_dev_fresh); free(m->dn_host_stale); }

static double maxdiff(const float *a, const float *b, size_t n) { double d = 0; for (size_t i = 0; i < n; i++) { double x = fabs((double)a[i] - b[i]); if (x > d) d = x; } return d; }

int main(void) {
    setenv("COLI_CUDA", "1", 1); setenv("COLI_GPUS", "0", 1); setenv("QT_NO_WARMSTART", "1", 1);
    setenv("COLI_DENSE_IDOT", "0", 1);          /* f32 activations on the CPU too: same arithmetic as the reference */
    setenv("Q36_DN_GPU", "1", 1);
    fake_ndev = 1; fake_dense_compute = 1;

    Weights w; make_weights(&w);
    float xs[T][D]; uint32_t s = 99;
    for (int t = 0; t < T; t++) for (int i = 0; i < D; i++) xs[t][i] = frand(&s);

    /* reference: everything on the CPU */
    Model ref; Layer lref; build_model(&ref, &lref, &w);
    float out_ref[T][D];
    for (int t = 0; t < T; t++) deltanet(&ref, &lref, 0, xs[t], 1, t, out_ref[t]);
    ck(fabs(out_ref[T-1][0]) > 0 || fabs(out_ref[T-1][1]) > 0, "the reference produces something");

    /* the layer on the (fake) card */
    Model m; Layer l; build_model(&m, &l, &w);
    int8_t *qf = malloc((size_t)PROJ_DIM * D); float *sf = malloc(PROJ_DIM * sizeof(float));
    memcpy(qf, w.qkv, (size_t)CONV_DIM * D); memcpy(qf + (size_t)CONV_DIM * D, w.z, (size_t)VALUE_DIM * D);
    memcpy(sf, w.sqkv, CONV_DIM * sizeof(float)); memcpy(sf + CONV_DIM, w.sz, VALUE_DIM * sizeof(float));
    ck(qt_dnproj_init(0, qf, sf, D, PROJ_DIM, 0), "in_proj on the card");
    int h = qt_dense_init(w.o, w.so, VALUE_DIM, D, 0);
    ck(h >= 0, "out_proj on the card");
    l.qth_dnout = h + 1;
    ck(qt_dn_gpu_init(0, VH, VK, KD, VD, CONV_DIM, CONVK, D, w.conv, w.norm, m.c.eps, l.qth_dnout), "the DeltaNet layer joins them");
    ck(qt_dn_gpu_ready(0), "and reports ready");
    m.dn_dev = 1;

    /* A. decode tokens through the GPU path */
    float out[D]; double worst = 0;
    for (int t = 0; t < T; t++) { deltanet(&m, &l, 0, xs[t], 1, t, out); double d = maxdiff(out, out_ref[t], D); if (d > worst) worst = d; }
    ck(fake_dn_steps == T, "every decode token ran on the card");
    ck(worst < 1e-4, "GPU outputs equal the CPU outputs");
    ck(m.dn_host_stale[0] == 1 && m.dn_dev_fresh[0] == 1, "the card is ahead; the host copy was left alone");
    { int allzero = 1; for (size_t i = 0; i < (size_t)VH * KD * VD; i++) if (m.DN_rec[0][i] != 0.f) allzero = 0; ck(allzero, "the host state was not touched by the GPU steps"); }
    dn_gpu_pull(&m, 0);
    ck(m.dn_host_stale[0] == 0, "pulled");
    ck(maxdiff(m.DN_rec[0], ref.DN_rec[0], (size_t)VH * KD * VD) < 1e-4 && maxdiff(m.DN_conv[0], ref.DN_conv[0], (size_t)CONV_DIM * (CONVK - 1)) < 1e-5,
       "the pulled state equals the CPU path's state");

    /* B. GPU, then a two-row prefill on the CPU, then GPU again */
    Model m2; Layer l2; build_model(&m2, &l2, &w); l2.qth_dnout = l.qth_dnout; m2.dn_dev = 1;
    reset_recurrent(&m2);                      /* clears the flags too: the card holds another model's state */
    int steps0 = fake_dn_steps;
    for (int t = 0; t < 3; t++) deltanet(&m2, &l2, 0, xs[t], 1, t, out);
    ck(fake_dn_steps == steps0 + 3, "three tokens on the card");
    float x2[2 * D]; memcpy(x2, xs[3], sizeof xs[3]); memcpy(x2 + D, xs[4], sizeof xs[4]);
    float out2[2 * D];
    deltanet(&m2, &l2, 0, x2, 2, 3, out2);   /* S > 1: the CPU path, which must pull first */
    ck(fake_dn_steps == steps0 + 3, "the prefill did not run on the card");
    ck(maxdiff(out2, out_ref[3], D) < 1e-4 && maxdiff(out2 + D, out_ref[4], D) < 1e-4, "the CPU prefill continued from the card's state");
    ck(m2.dn_dev_fresh[0] == 0 && m2.dn_host_stale[0] == 0, "after the CPU step the device copy is old, the host current");
    deltanet(&m2, &l2, 0, xs[5], 1, 5, out);
    ck(fake_dn_steps == steps0 + 4 && maxdiff(out, out_ref[5], D) < 1e-4, "the next GPU token pushed the host state first and matches");

    /* C. a failing step: the layer turns itself off, the CPU continues from the card's state */
    Model m3; Layer l3; build_model(&m3, &l3, &w); l3.qth_dnout = l.qth_dnout; m3.dn_dev = 1; reset_recurrent(&m3);
    for (int t = 0; t < 2; t++) deltanet(&m3, &l3, 0, xs[t], 1, t, out);
    fake_dn_fail = 1;
    deltanet(&m3, &l3, 0, xs[2], 1, 2, out);
    fake_dn_fail = 0;
    ck(!qt_dn_gpu_ready(0), "after the failure the layer is off");
    ck(maxdiff(out, out_ref[2], D) < 1e-4, "the failed token was computed on the CPU from the pulled state");
    deltanet(&m3, &l3, 0, xs[3], 1, 3, out);
    ck(maxdiff(out, out_ref[3], D) < 1e-4, "and the CPU keeps going");

    qt_shutdown();
    free(qf); free(sf); free_model(&ref); free_model(&m); free_model(&m2); free_model(&m3);
    if (fails) { printf("test_qwen36_dn_gpu: %d failure(s)\n", fails); return 1; }
    printf("OK test_qwen36_dn_gpu: the DeltaNet layer on the card matches the CPU, host and device state agree\n");
    return 0;
}
