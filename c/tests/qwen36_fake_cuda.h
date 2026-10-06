/* qwen36_fake_cuda.h -- fake CUDA backend shared by the qwen36 tier tests.
 *
 * Defines every coli_cuda_* symbol qwen36_tier.c links against (its
 * signatures come from backend_cuda.h, which the tier includes on its own)
 * and RECORDS what it receives, so a test can assert on real upload/issue
 * traffic without a GPU or the CUDA toolkit. A test that only checked
 * "qt_init returns 1" would pass even with the tier fully broken.
 *
 * Three settable hooks beyond plain recording:
 *   fake_ndev        - device count returned by coli_cuda_available_device_count
 *                       and coli_cuda_device_count (default 1).
 *   fake_issue_hook   - called by coli_cuda_expert_group_issue with the issuing
 *                       device (taken from g[0]->device), the row count and the
 *                       input pointer; its return value is what issue returns.
 *                       NULL (the default) reproduces the old always-0 stub.
 *   fake_upload_hook  - called at the start of every tensor upload, on the
 *                       uploader thread, with the tensor's fmt. A test that
 *                       needs an upload to take TIME (a real cudaMemcpy does)
 *                       sleeps here; NULL (the default) uploads instantly. */
#ifndef QWEN36_FAKE_CUDA_H
#define QWEN36_FAKE_CUDA_H
#include <math.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#include "../backend_cuda.h"

struct ColiCudaTensor { int fmt, I, O, device, gs; const void *w; const float *sc; };

static int fake_uploads;
static int fake_overwrites;      /* coli_cuda_tensor_overwrite calls (in-place swaps) */
static int fake_overwrite_fail;  /* 1: refuse overwrites, as a backend without the symbol */
static int last_fmt = -1;
static size_t last_bytes;
static unsigned char captured[4096];
static size_t captured_len;

static int fake_ndev = 1;
static size_t fake_free_bytes = 2ull << 30;    /* what coli_cuda_mem_info reports as free */
static int (*fake_issue_hook)(int device, int count, const float *x) = NULL;
static void (*fake_upload_hook)(int fmt) = NULL;

/* fake_dense_compute=1: coli_cuda_matmul really computes fmt 1 (int8 per
 * row) from the uploaded bytes, so an engine test can put a trunk on the fake
 * tier and demand the same tokens as the CPU int8 reference. The engine
 * frees its int8 rows right after qt_dense_init, so the upload keeps a copy. */
static int fake_dense_compute;
static int upload_common(ColiCudaTensor **t, const void *w, const float *sc, int fmt,
                         int I, int O, int device, int gs) {
    if (fake_upload_hook) fake_upload_hook(fmt);
    ColiCudaTensor *n = (ColiCudaTensor *)calloc(1, sizeof *n);
    n->fmt = fmt; n->I = I; n->O = O; n->device = device; n->gs = gs; n->w = w; n->sc = sc;
    if (fake_dense_compute && fmt == 1 && w && sc) {
        int8_t *q = (int8_t *)malloc((size_t)I * O); float *s = (float *)malloc((size_t)O * sizeof(float));
        if (q && s) { memcpy(q, w, (size_t)I * O); memcpy(s, sc, (size_t)O * sizeof(float)); n->w = q; n->sc = s; }
        else { free(q); free(s); n->w = NULL; n->sc = NULL; }
    }
    *t = n;
    fake_uploads++;
    last_fmt = fmt;
    last_bytes = (size_t)I * O / ((fmt == 1 || fmt == 8) ? 1 : 2);
    if (fake_uploads == 1) {
        captured_len = last_bytes < sizeof captured ? last_bytes : sizeof captured;
        memcpy(captured, w, captured_len);
    }
    return 1;
}
int coli_cuda_tensor_upload(ColiCudaTensor **t, const void *w, const float *s,
                            int fmt, int I, int O, int device) {
    return upload_common(t, w, s, fmt, I, O, device, 0);
}
int coli_cuda_tensor_upload_g(ColiCudaTensor **t, const void *w, const float *s,
                              int fmt, int I, int O, int device, int gs) {
    return upload_common(t, w, s, fmt, I, O, device, gs);
}
void coli_cuda_tensor_free(ColiCudaTensor *t) {
    if (t && fake_dense_compute && t->fmt == 1) { free((void *)t->w); free((void *)t->sc); }
    free(t);
}
int coli_cuda_tensor_overwrite(ColiCudaTensor *t, const void *w, const float *sc) {
    if (!t || !w || fake_overwrite_fail) return 0;
    if (fake_upload_hook) fake_upload_hook(t->fmt);
    if (fake_dense_compute && t->fmt == 1 && t->w && t->sc) {
        memcpy((void *)t->w, w, (size_t)t->I * t->O); memcpy((void *)t->sc, sc, (size_t)t->O * sizeof(float));
    } else { t->w = w; t->sc = sc; }
    fake_overwrites++;
    return 1;
}
int coli_cuda_available_device_count(void) { return fake_ndev; }
int coli_cuda_device_count(void) { return fake_ndev; }
int coli_cuda_init(const int *d, int n) { (void)d; (void)n; return 1; }
static int fake_lut_published;
int coli_cuda_fp8_set_lut(const float *lut) { fake_lut_published = lut != NULL; return lut != NULL; }
void coli_cuda_shutdown(void) {}
int coli_cuda_mem_info(int device, size_t *freeb, size_t *total) {
    (void)device;
    *freeb = fake_free_bytes; *total = 4ull << 30;   /* 2 GiB liberi by default */
    return 1;
}
int coli_cuda_expert_group_issue(ColiCudaTensor *const *g, ColiCudaTensor *const *u,
                                 ColiCudaTensor *const *d, const int *rows,
                                 int count, const float *x) {
    (void)u; (void)d; (void)rows;
    if (fake_issue_hook) return fake_issue_hook(count > 0 ? g[0]->device : -1, count, x);
    return 0;
}
const float *coli_cuda_expert_group_take(int device) { (void)device; return NULL; }
void coli_cuda_group_stats(uint64_t *calls, uint64_t *experts, uint64_t *rows,
                           double *h2d, double *kernel, double *d2h) {
    if (calls) *calls = 0; if (experts) *experts = 0; if (rows) *rows = 0;
    if (h2d) *h2d = 0; if (kernel) *kernel = 0; if (d2h) *d2h = 0;
}
void coli_cuda_stats(int device, size_t *count, size_t *bytes) {
    (void)device; if (count) *count = 0; if (bytes) *bytes = 0;
}

/* dense GEMV on a resident tensor (lm_head / DeltaNet projections placed on a
 * device). Counted, never computed: the placement tests check WHERE work
 * went; the arithmetic has its own oracle in the CUDA build. Parameters are
 * unused on purpose (CFLAGS carry -Wno-unused-parameter). */
static int fake_matmuls, fake_matmul_fail, fake_matmul_rows, fake_matmul_fail_at;
int coli_cuda_matmul(ColiCudaTensor **tensor, float *y, const float *x, const void *weights, const float *scales, int fmt, int S, int I, int O, int device, int gs) {
    fake_matmuls++;
    fake_matmul_rows = S;
    if (fake_matmul_fail || fake_matmuls == fake_matmul_fail_at) {
        for(int i=0;i<S*O;i++) y[i]=NAN;
        return 0;
    }
    ColiCudaTensor *t = tensor ? *tensor : NULL;
    if (fake_dense_compute && t && t->fmt == 1 && t->w && t->sc && t->I == I && t->O == O) {
        const int8_t *q = (const int8_t *)t->w;
        for (int s = 0; s < S; s++) for (int o = 0; o < O; o++) {
            const int8_t *w = q + (size_t)o * I; const float *xs = x + (size_t)s * I; float a = 0.f;
            for (int i = 0; i < I; i++) a += xs[i] * (float)w[i];
            y[(size_t)s * O + o] = a * t->sc[o];
        }
    }
    return 1;
}


/* Gated delta layer, host-side reference: the same arithmetic as the CUDA
 * kernels (and as qwen36.c's deltanet()), so an engine test can put the layer
 * on the fake tier and demand the same tokens as the CPU path. Matmuls go
 * through the fake's fmt=1 compute when fake_dense_compute is on. */
struct ColiCudaDn { int device, vh, vk, kdim, vdim, conv_dim, convk, hidden; float eps; int gate_sigmoid;
                    float *ring, *rec, *conv_w, *norm_w; };
static int fake_dn_steps, fake_dn_fail;
static void fake_gemv_i8(float *y, const float *x, const ColiCudaTensor *t) {
    const int8_t *q = (const int8_t *)t->w;
    for (int o = 0; o < t->O; o++) { float a = 0.f; const int8_t *w = q + (size_t)o * t->I;
        for (int i = 0; i < t->I; i++) a += x[i] * (float)w[i]; y[o] = a * t->sc[o]; }
}
ColiCudaDn *coli_cuda_dn_create(int device, int vh, int vk, int kdim, int vdim, int conv_dim, int convk, int hidden,
                                const float *conv_w, const float *norm_w, float eps, int gate_sigmoid) {
    if (fake_dn_fail || vh < 1 || vk < 1 || vh % vk || convk < 2 || conv_dim != 2 * vk * kdim + vh * vdim || !conv_w || !norm_w) return NULL;
    ColiCudaDn *d = (ColiCudaDn *)calloc(1, sizeof *d);
    d->device = device; d->vh = vh; d->vk = vk; d->kdim = kdim; d->vdim = vdim; d->conv_dim = conv_dim; d->convk = convk; d->hidden = hidden; d->eps = eps; d->gate_sigmoid = gate_sigmoid;
    d->ring = (float *)calloc((size_t)conv_dim * (convk - 1), sizeof(float));
    d->rec = (float *)calloc((size_t)vh * kdim * vdim, sizeof(float));
    d->conv_w = (float *)malloc((size_t)conv_dim * convk * sizeof(float)); memcpy(d->conv_w, conv_w, (size_t)conv_dim * convk * sizeof(float));
    d->norm_w = (float *)malloc((size_t)vdim * sizeof(float)); memcpy(d->norm_w, norm_w, (size_t)vdim * sizeof(float));
    return d;
}
void coli_cuda_dn_free(ColiCudaDn *d) { if (!d) return; free(d->ring); free(d->rec); free(d->conv_w); free(d->norm_w); free(d); }
int coli_cuda_dn_set_state(ColiCudaDn *d, const float *ring, const float *rec) {
    if (!d) return 0;
    size_t rn = (size_t)d->conv_dim * (d->convk - 1), sn = (size_t)d->vh * d->kdim * d->vdim;
    if (ring) memcpy(d->ring, ring, rn * sizeof(float)); else memset(d->ring, 0, rn * sizeof(float));
    if (rec) memcpy(d->rec, rec, sn * sizeof(float)); else memset(d->rec, 0, sn * sizeof(float));
    return 1;
}
int coli_cuda_dn_get_state(ColiCudaDn *d, float *ring, float *rec) {
    if (!d || !ring || !rec) return 0;
    memcpy(ring, d->ring, (size_t)d->conv_dim * (d->convk - 1) * sizeof(float));
    memcpy(rec, d->rec, (size_t)d->vh * d->kdim * d->vdim * sizeof(float));
    return 1;
}
int coli_cuda_dn_step(ColiCudaDn *d, ColiCudaTensor *proj, ColiCudaTensor *projz, ColiCudaTensor *outp, const float *x, float *out, const float *egh, const float *beta) {
    if (!d || !proj || !outp || !x || !out || fake_dn_fail) return 0;
    if (!fake_dense_compute || !proj->w || !outp->w || (projz && !projz->w)) return 0;      /* nothing to compute with: the engine keeps the CPU path */
    int vh = d->vh, kdim = d->kdim, vdim = d->vdim, rep = vh / d->vk, key_dim_tot = d->vk * kdim, conv_dim = d->conv_dim, convk = d->convk;
    size_t proj_dim = (size_t)conv_dim + (size_t)vh * vdim, want_o = projz ? (size_t)conv_dim : proj_dim;
    if (proj->fmt != 1 || proj->I != d->hidden || (size_t)proj->O != want_o || outp->fmt != 1 || (size_t)outp->I != (size_t)vh * vdim || outp->O != d->hidden) return 0;
    if (projz && (projz->fmt != 1 || projz->I != d->hidden || (size_t)projz->O != (size_t)vh * vdim)) return 0;
    float *qkvz = (float *)malloc(proj_dim * sizeof(float)), *conv_out = (float *)malloc((size_t)conv_dim * sizeof(float));
    float *outr = (float *)malloc((size_t)vh * vdim * sizeof(float));
    fake_gemv_i8(qkvz, x, proj);
    if (projz) fake_gemv_i8(qkvz + conv_dim, x, projz);
    const float *z = qkvz + conv_dim;
    for (int cc = 0; cc < conv_dim; cc++) {
        const float *w = d->conv_w + (size_t)cc * convk; float *rg = d->ring + (size_t)cc * (convk - 1);
        float acc = 0.f; for (int kk = 0; kk < convk - 1; kk++) acc += w[kk] * rg[kk];
        acc += w[convk - 1] * qkvz[cc]; conv_out[cc] = acc / (1.f + expf(-acc));
        for (int kk = 0; kk < convk - 2; kk++) rg[kk] = rg[kk + 1];
        rg[convk - 2] = qkvz[cc];
    }
    float scale = 1.f / sqrtf((float)kdim);
    for (int h = 0; h < vh; h++) {
        const float *qs = conv_out + (size_t)(h / rep) * kdim, *ks = conv_out + key_dim_tot + (size_t)(h / rep) * kdim, *vs = conv_out + 2 * (size_t)key_dim_tot + (size_t)h * vdim;
        float q[256], k[256], delta[256], o[256];
        double sq = 1e-6, sk = 1e-6;
        for (int t = 0; t < kdim; t++) { sq += (double)qs[t] * qs[t]; sk += (double)ks[t] * ks[t]; }
        double nq = sqrt(sq), nk = sqrt(sk);
        for (int t = 0; t < kdim; t++) { q[t] = (float)((double)qs[t] / nq * scale); k[t] = (float)((double)ks[t] / nk); }
        float *Sh = d->rec + (size_t)h * kdim * vdim;
        for (int t = 0; t < vdim; t++) { float kvsum = 0.f; for (int kk = 0; kk < kdim; kk++) kvsum += k[kk] * (Sh[(size_t)kk * vdim + t] * egh[h]); delta[t] = (vs[t] - kvsum) * beta[h]; }
        for (int t = 0; t < vdim; t++) { float acc = 0.f; for (int kk = 0; kk < kdim; kk++) { float s = Sh[(size_t)kk * vdim + t] * egh[h] + k[kk] * delta[t]; Sh[(size_t)kk * vdim + t] = s; acc += q[kk] * s; } o[t] = acc; }
        double ms = 0; for (int t = 0; t < vdim; t++) ms += (double)o[t] * o[t];
        float r = 1.f / sqrtf((float)(ms / vdim) + d->eps);
        for (int t = 0; t < vdim; t++) { float zz = z[(size_t)h * vdim + t]; float gate = d->gate_sigmoid ? 1.f / (1.f + expf(-zz)) : zz / (1.f + expf(-zz)); outr[(size_t)h * vdim + t] = (o[t] * r * d->norm_w[t]) * gate; }
    }
    fake_gemv_i8(out, outr, outp);
    free(qkvz); free(conv_out); free(outr);
    fake_dn_steps++;
    return 1;
}

#endif /* QWEN36_FAKE_CUDA_H */
