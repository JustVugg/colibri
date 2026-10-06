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
    if (fake_dense_compute && fmt == 0 && w) {   /* f32 rows (the chain's b|a, gate rows): the caller frees its copy */
        float *f = (float *)malloc((size_t)I * O * sizeof(float));
        if (f) { memcpy(f, w, (size_t)I * O * sizeof(float)); n->w = f; n->sc = NULL; } else n->w = NULL;
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

/* ---- the dense chain's table, host-side (cuda_chain.h) ------------------------
 * The same ops as cuda_chain.cu's kernels, in plain C in the kernels' order of sums,
 * so an engine test can run qwen36's CUDA chain (qwen36_cuda_chain.h) on the fake and
 * demand the CPU path's tokens. Buffers are plain memory (every kind), a frame is
 * nothing (every op runs when called), the matmul is the fake's fmt 1 compute or the
 * f32 rows of a fmt 0 tensor. fake_chain_fail_at: the n-th cc_submit fails (a lost
 * device); fake_chain_frames counts the submits. */
#include "../cuda_chain.h"
struct CcBuf { int kind; size_t bytes; float *d; };
static int fake_chain_up, fake_chain_lost, fake_chain_dev = -1, fake_chain_frames, fake_chain_fail_at, fake_chain_ops;
static size_t fake_chain_dev_bytes;
static int fcc_init(int device) { fake_chain_up = 1; fake_chain_lost = 0; fake_chain_dev = device; return 1; }
static int fcc_ready(void) { return fake_chain_up && !fake_chain_lost; }
static int fcc_lost(void) { return !fake_chain_up || fake_chain_lost; }
static void fcc_shutdown(void) { fake_chain_up = 0; }
static int fcc_device(int d) { int was = fake_chain_dev; fake_chain_dev = d; return was; }
static int fcc_device_now(void) { return fake_chain_dev; }
static CcBuf *fcc_buf(size_t bytes, int kind) {
    if (!bytes) bytes = 4;
    CcBuf *b = (CcBuf *)calloc(1, sizeof *b);
    b->kind = kind; b->bytes = bytes; b->d = (float *)calloc(bytes, 1);
    if (kind == CC_DEV) fake_chain_dev_bytes += bytes;
    return b;
}
static void fcc_free(CcBuf *b) { if (!b) return; if (b->kind == CC_DEV) fake_chain_dev_bytes -= b->bytes; free(b->d); free(b); }
static int fcc_reserve(CcBuf **b, size_t bytes, int kind) {
    if (*b && (*b)->bytes >= bytes && (*b)->kind == kind) return 1;
    CcBuf *n = fcc_buf(bytes, kind); fcc_free(*b); *b = n; return 1;
}
static void *fcc_ptr(const CcBuf *b) { return b ? b->d : NULL; }
static size_t fcc_bytes(const CcBuf *b) { return b ? b->bytes : 0; }
static int fcc_begin(void) { return fcc_ready(); }
static int fcc_submit(int wait) {
    (void)wait;
    if (!fcc_ready()) return 0;
    fake_chain_frames++;
    if (fake_chain_fail_at && fake_chain_frames == fake_chain_fail_at) { fake_chain_lost = 1; return 0; }
    return 1;
}
static int fcc_finish(void) { return fcc_ready(); }
static int fcc_copy(CcBuf *dst, size_t doff, CcBuf *src, size_t soff, size_t n) {
    if (!fcc_ready() || !dst || !src || (doff + n) * 4 > dst->bytes || (soff + n) * 4 > src->bytes) return 0;
    memmove(dst->d + doff, src->d + soff, n * 4); fake_chain_ops++; return 1;
}
static int fcc_zero(CcBuf *dst, size_t off, size_t n) {
    if (!fcc_ready() || !dst || (off + n) * 4 > dst->bytes) return 0;
    memset(dst->d + off, 0, n * 4); return 1;
}
static int fcc_copy_regions(CcBuf *dst, CcBuf *src, const CcRegion *r, int n) {
    for (int i = 0; i < n; i++) if (!fcc_copy(dst, r[i].dst, src, r[i].src, r[i].n)) return 0;
    return 1;
}
static int fcc_write(CcBuf *dst, size_t off, const void *src, size_t bytes) {
    if (!fcc_ready() || !dst || off * 4 + bytes > dst->bytes) return 0;
    memcpy((char *)dst->d + off * 4, src, bytes); return 1;
}
static int fcc_read(CcBuf *src, size_t off, void *dst, size_t bytes) {
    if (!fcc_ready() || !src || off * 4 + bytes > src->bytes) return 0;
    memcpy(dst, (const char *)src->d + off * 4, bytes); return 1;
}
static int fcc_matmul(ColiCudaTensor *t, CcBuf *x, size_t xo, CcBuf *y, size_t yo, int S) {
    if (!fcc_ready() || !t || !x || !y || S < 1 || !t->w) return 0;
    if ((xo + (size_t)S * t->I) * 4 > x->bytes || (yo + (size_t)S * t->O) * 4 > y->bytes) return 0;
    fake_chain_ops++;
    for (int s = 0; s < S; s++) {
        const float *xs = x->d + xo + (size_t)s * t->I; float *ys = y->d + yo + (size_t)s * t->O;
        if (t->fmt == 1 && t->sc) {
            const int8_t *q = (const int8_t *)t->w;
            for (int o = 0; o < t->O; o++) { const int8_t *w = q + (size_t)o * t->I; float a = 0.f;
                for (int i = 0; i < t->I; i++) a += xs[i] * (float)w[i]; ys[o] = a * t->sc[o]; }
        } else if (t->fmt == 0) {
            const float *w = (const float *)t->w;
            for (int o = 0; o < t->O; o++) { const float *wr = w + (size_t)o * t->I; float a = 0.f;
                for (int i = 0; i < t->I; i++) a += xs[i] * wr[i]; ys[o] = a; }
        } else return 0;
    }
    return 1;
}
static int fcc_norm(CcBuf *x, CcBuf *w, CcBuf *y, const CcNorm *p) {
    if (!fcc_ready() || !x || !y || !p) return 0;
    fake_chain_ops++;
    for (int g = 0; g < p->nseg; g++) {
        int row = g / p->per_row, j = g - row * p->per_row;
        const float *xb = x->d + p->x_off + row * p->x_row + j * p->x_seg;
        float *yb = y->d + p->y_off + row * p->y_row + j * p->y_seg;
        const float *wb = w ? w->d + p->w_off + (p->w_mod > 0 ? (j % p->w_mod) * p->D : 0) : NULL;
        float acc = 0.f; for (int i = 0; i < p->D; i++) acc += xb[i] * xb[i];
        float r = (p->flags & 4) ? 1.f / sqrtf(acc + p->eps) : 1.f / sqrtf(acc / (float)p->D + p->eps);
        for (int i = 0; i < p->D; i++) {
            float wv = (p->flags & 2) ? 1.f : ((p->flags & 1) ? 1.f + wb[i] : wb[i]);
            yb[i] = xb[i] * r * wv * p->post;
        }
    }
    return 1;
}
static int fcc_rope(CcBuf *x, CcBuf *cs, const CcRope *p) {
    if (!fcc_ready() || !x || !cs || !p) return 0;
    fake_chain_ops++;
    for (int seg = 0; seg < p->nseg; seg++) {
        int row = seg / p->per_row, hh = seg - row * p->per_row;
        float *base = x->d + p->x_off + row * p->x_row + hh * p->x_seg;
        const float *cb = cs->d + p->cs_off + row * p->cs_row;
        for (int j = 0; j < p->half_; j++) {
            float c = cb[2 * j], s = cb[2 * j + 1], a = base[j], b = base[j + p->half_];
            base[j] = a * c - b * s; base[j + p->half_] = b * c + a * s;
        }
    }
    return 1;
}
static int fcc_attn(CcBuf *q, CcBuf *kc, CcBuf *vc, CcBuf *o, CcBuf *gate, CcBuf *sel, const CcAttn *p) {
    if (!fcc_ready() || !q || !kc || !vc || !o || !p || (p->has_gate && !gate)) return 0;
    fake_chain_ops++;
    int rep = p->H / p->KVH;
    for (int s = 0; s < p->S; s++) for (int h = 0; h < p->H; h++) {
        int kvh = h / rep, kbase = kvh * p->cap;
        const float *qs = q->d + p->q_off + s * p->q_row + h * p->q_seg;
        int n = p->pos_base + s + 1, list = 0, lb = 0;
        if (p->sel_row > 0 && sel) { lb = p->sel_off + s * p->sel_row; int c = ((const int *)sel->d)[lb]; if (c >= 0) { n = c; list = 1; } }
        float m = -3.0e38f, l = 0.f, acc[256]; for (int d = 0; d < p->hd; d++) acc[d] = 0.f;
        for (int t0 = 0; t0 < n; t0 += 128) {
            int cnt = n - t0 < 128 ? n - t0 : 128; float sv[128], pe[128]; int pt[128];
            float mx = -3.0e38f;
            for (int i = 0; i < cnt; i++) {
                int t = list ? ((const int *)sel->d)[lb + 1 + t0 + i] : t0 + i;
                pt[i] = t;
                const float *kb = kc->d + p->k_off + (kbase + t) * p->hd;
                float a = 0.f; for (int d = 0; d < p->hd; d++) a += qs[d] * kb[d];
                sv[i] = a * p->scale; if (sv[i] > mx) mx = sv[i];
            }
            float mn = m > mx ? m : mx, sum = 0.f;
            for (int i = 0; i < cnt; i++) { pe[i] = expf(sv[i] - mn); sum += pe[i]; }
            float corr = expf(m - mn);
            l = l * corr + sum;
            for (int d = 0; d < p->hd; d++) acc[d] *= corr;
            for (int i = 0; i < cnt; i++) { const float *vb = vc->d + p->v_off + (kbase + pt[i]) * p->hd; for (int d = 0; d < p->hd; d++) acc[d] += pe[i] * vb[d]; }
            m = mn;
        }
        float inv = l > 0.f ? 1.f / l : 0.f;
        float *ob = o->d + p->o_off + s * p->o_row + h * p->hd;
        const float *gb = p->has_gate ? gate->d + p->g_off + s * p->g_row + h * p->g_seg : NULL;
        for (int d = 0; d < p->hd; d++) { float v = acc[d] * inv; if (gb) v *= 1.f / (1.f + expf(-gb[d])); ob[d] = v; }
    }
    return 1;
}
static int fcc_dnconv(CcBuf *in, CcBuf *w, CcBuf *ring, CcBuf *out, CcBuf *snap, const CcDnConv *p) {
    if (!fcc_ready() || !in || !w || !ring || !out || !p || (p->snap_row >= 0 && !snap)) return 0;
    fake_chain_ops++;
    int nh = p->CK - 1;
    for (int c = 0; c < p->CD; c++) {
        float hist[8] = {0};
        for (int k = 0; k < nh; k++) hist[k] = ring->d[p->ring_off + c * nh + k];
        const float *wb = w->d + p->w_off + c * p->CK; float wl = wb[p->CK - 1];
        for (int s = 0; s < p->S; s++) {
            float cur = in->d[p->in_off + s * p->in_row + c], acc;
            if (p->order == 0) { acc = 0.f; for (int k = 0; k < nh; k++) acc += wb[k] * hist[k]; acc += wl * cur; }
            else { acc = wl * cur; for (int k = 0; k < nh; k++) acc += wb[k] * hist[k]; }
            out->d[p->out_off + s * p->out_row + c] = acc / (1.f + expf(-acc));
            for (int k = 0; k + 1 < nh; k++) hist[k] = hist[k + 1];
            if (nh > 0) hist[nh - 1] = cur;
            if (s == p->snap_row) for (int k = 0; k < nh; k++) snap->d[p->snap_off + c * nh + k] = hist[k];
        }
        for (int k = 0; k < nh; k++) ring->d[p->ring_off + c * nh + k] = hist[k];
    }
    return 1;
}
static float fcc_softplus(float x) { if (x > 20.f) return x; float e = expf(x); return e < 1e-4f ? e - 0.5f * e * e : logf(1.f + e); }
static int fcc_dnrec(int KD, CcBuf *cv, CcBuf *ab, CcBuf *z, CcBuf *st, CcBuf *prm, CcBuf *y, CcBuf *snap, const CcDnRec *p) {
    if (!fcc_ready() || !cv || !ab || !z || !st || !prm || !y || !p || (p->snap_row >= 0 && !snap) || KD > 256 || p->VD > 128) return 0;
    fake_chain_ops++;
    for (int h = 0; h < p->VH; h++) {
        int kh = h / (p->VH / p->KH);
        float *S = st->d + p->st_off + h * KD * p->VD;
        float alog = prm->d[p->prm_off + h], dtb = prm->d[p->prm_off + p->VH + h];
        const float *nw = prm->d + p->prm_off + 2 * p->VH;
        for (int s = 0; s < p->S; s++) {
            const float *cb = cv->d + p->cv_off + s * p->cv_row;
            float qn[256], kn[256], qa = 0.f, ka = 0.f;
            for (int k = 0; k < KD; k++) { qn[k] = cb[kh * KD + k]; kn[k] = cb[p->Ktot + kh * KD + k]; qa += qn[k] * qn[k]; ka += kn[k] * kn[k]; }
            float qsc = 1.f / sqrtf(qa + 1e-6f) * p->qscale, ksc = 1.f / sqrtf(ka + 1e-6f);
            for (int k = 0; k < KD; k++) { qn[k] *= qsc; kn[k] *= ksc; }
            float bv = ab->d[p->b_off + s * p->b_row + h], av = ab->d[p->a_off + s * p->a_row + h];
            float beta = 1.f / (1.f + expf(-bv)), decay = expf(-expf(alog) * fcc_softplus(av + dtb));
            float o[128], ms = 0.f;
            for (int v = 0; v < p->VD; v++) {
                float vin = cb[2 * p->Ktot + h * p->VD + v], kvs = 0.f;
                for (int k = 0; k < KD; k++) { S[k * p->VD + v] *= decay; kvs += kn[k] * S[k * p->VD + v]; }
                float delta = (vin - kvs) * beta;
                for (int k = 0; k < KD; k++) S[k * p->VD + v] += kn[k] * delta;
                float ov = 0.f; for (int k = 0; k < KD; k++) ov += qn[k] * S[k * p->VD + v];
                o[v] = ov; ms += ov * ov;
            }
            float r = 1.f / sqrtf(ms / (float)p->VD + p->eps);
            for (int v = 0; v < p->VD; v++) {
                float zv = z->d[p->z_off + s * p->z_row + h * p->VD + v];
                float g = (p->flags & 1) ? 1.f / (1.f + expf(-zv)) : zv / (1.f + expf(-zv));
                y->d[p->y_off + s * p->y_row + h * p->VD + v] = o[v] * r * nw[v] * g;
            }
            if (s == p->snap_row) memcpy(snap->d + p->snap_off + h * KD * p->VD, S, (size_t)KD * p->VD * sizeof(float));
        }
    }
    return 1;
}
static float fcc_sig(float x) { return 1.f / (1.f + expf(-x)); }
static int fcc_ew(CcBuf *y, CcBuf *a, CcBuf *b, CcBuf *c, CcBuf *e, const CcEw *p) {
    if (!fcc_ready() || !y || !p) return 0;
    fake_chain_ops++;
    float *Y = y->d; const float *A = a ? a->d : NULL, *B = b ? b->d : NULL, *Cc = c ? c->d : NULL, *E = e ? e->d : NULL;
    for (int i = 0; i < p->n; i++) {
        if (p->op == 0) Y[p->y_off + i] = A[p->a_off + i] + B[p->b_off + i];
        else if (p->op == 1) {
            int r = i / p->D, d = i - r * p->D;
            float t = (p->flags & 1) ? B[p->b_off + r * p->D + d] : 0.f;
            if (p->flags & 2) { float g = (p->flags & 4) ? fcc_sig(E[p->e_off + r * p->e_row]) : 1.f; t = t + g * Cc[p->c_off + r * p->D + d]; }
            Y[p->y_off + i] = (p->flags & 8) ? t : A[p->a_off + i] + t;
        } else if (p->op == 2) { float g = A[p->a_off + i]; Y[p->y_off + i] = (g / (1.f + expf(-g))) * B[p->b_off + i]; }
        else if (p->op == 3) { float v = A[p->a_off + i] / p->fc; Y[p->y_off + i] = v * fcc_sig(v); }
        else if (p->op == 4) { int r = i / p->D, d = i - r * p->D, W = p->C * p->D; float v = 0.f;
            for (int k = 0; k < p->C; k++) { int j = r * W + k * p->D + d; v += fcc_sig(A[p->a_off + j]) * B[p->b_off + j]; }
            Y[p->y_off + i] = v / p->fc; }
        else if (p->op == 5) Y[p->y_off + i] = 2.f * fcc_sig(A[p->a_off + i] / p->fc);
        else if (p->op == 6) { int W = p->C * p->D, r = i / W, rem = i - r * W, k = rem / p->D, d = rem - k * p->D;
            Y[p->y_off + i] += A[p->a_off + r * p->C + k] * B[p->b_off + r * p->D + d]; }
        else if (p->op == 7) Y[p->y_off + i] = A[p->a_off + i] * p->fc;
        else if (p->op == 8) Y[p->y_off + i] = A[p->a_off + i] + E[p->e_off + i % p->D] * B[p->b_off + i];
        else return 0;
    }
    return 1;
}
static void fcc_stats(CcStats *st) { memset(st, 0, sizeof *st); st->frames = (unsigned long long)fake_chain_frames; st->ops = (unsigned long long)fake_chain_ops; st->dev_bytes = fake_chain_dev_bytes; }
static const ColiCudaChainOps fake_chain_table = {
    sizeof(ColiCudaChainOps), fcc_init, fcc_ready, fcc_lost, fcc_shutdown, fcc_device, fcc_device_now,
    fcc_buf, fcc_free, fcc_reserve, fcc_ptr, fcc_bytes, fcc_begin, fcc_submit, fcc_finish,
    fcc_copy, fcc_zero, fcc_copy_regions, fcc_write, fcc_read,
    fcc_matmul, fcc_norm, fcc_rope, fcc_attn, fcc_dnconv, fcc_dnrec, fcc_ew, fcc_stats,
};
static int fake_chain_absent;   /* 1: a backend without the chain (an older DLL) */
const ColiCudaChainOps *coli_cuda_chain_ops(void) { return fake_chain_absent ? NULL : &fake_chain_table; }

#endif /* QWEN36_FAKE_CUDA_H */
