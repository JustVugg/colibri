/* The dense chain's ops on a CUDA device (cuda_chain.cu) against CPU references, the
 * references being the same arithmetic in the same order (tests/qwen36_fake_cuda.h's
 * host-side table, copied here so the real backend can be linked):
 *   buffers   DEV, UP and DOWN; write, read, copy, zero, copy_regions, reserve
 *   frames    ops in a frame left in flight, the next frame ordered after it
 *   matmul    int8 rows (fmt 1) and f32 rows (fmt 0), one row and several
 *   norm      zero-centred, plain, no weight, L2; heads at a stride, in place
 *   rope      rotate-half from a host table, heads at a stride
 *   attn      causal GQA with the output gate over a cache at an offset, a prefill of
 *             several rows after earlier ones, a selection list
 *   dnconv    both orders, the ring carried across calls, the snapshot row
 *   dnrec     KD 8 and 128, silu and sigmoid gates, the state carried, the snapshot
 *   ew        every element-wise op
 *   lost      a failed launch (COLI_GPU_FAIL_AFTER) marks the device lost to the chain
 * Needs a device: run by hand (make cuda-chain-check CUDA=1); gpu-compile builds it. */
#include "../backend_cuda.h"
#define COLI_CUDA_CHAIN_NO_WRAPPERS
#include "../cuda_chain.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

static int fails;
static void ck(int ok, const char *what) { if (ok) { printf("  ok   %s\n", what); return; } printf("  FAIL %s\n", what); fails++; }
static float frand(unsigned *s) { *s = *s * 1664525u + 1013904223u; return ((*s >> 8) & 0xFFFF) / 65535.f * 2.f - 1.f; }
static double maxdiff(const float *a, const float *b, size_t n) { double d = 0; for (size_t i = 0; i < n; i++) { double x = fabs((double)a[i] - b[i]); if (x > d) d = x; } return d; }
static const ColiCudaChainOps *T;

/* ---- references ---------------------------------------------------------------- */
static void ref_norm(const float *x, const float *w, float *y, const CcNorm *p) {
    for (int g = 0; g < p->nseg; g++) {
        int row = g / p->per_row, j = g - row * p->per_row;
        const float *xb = x + p->x_off + row * p->x_row + j * p->x_seg; float *yb = y + p->y_off + row * p->y_row + j * p->y_seg;
        const float *wb = w ? w + p->w_off + (p->w_mod > 0 ? (j % p->w_mod) * p->D : 0) : NULL;
        float acc = 0.f; for (int i = 0; i < p->D; i++) acc += xb[i] * xb[i];
        float r = (p->flags & 4) ? 1.f / sqrtf(acc + p->eps) : 1.f / sqrtf(acc / (float)p->D + p->eps);
        for (int i = 0; i < p->D; i++) { float wv = (p->flags & 2) ? 1.f : ((p->flags & 1) ? 1.f + wb[i] : wb[i]); yb[i] = xb[i] * r * wv * p->post; }
    }
}
static void ref_rope(float *x, const float *cs, const CcRope *p) {
    for (int seg = 0; seg < p->nseg; seg++) {
        int row = seg / p->per_row, hh = seg - row * p->per_row;
        float *base = x + p->x_off + row * p->x_row + hh * p->x_seg; const float *cb = cs + p->cs_off + row * p->cs_row;
        for (int j = 0; j < p->half_; j++) { float c = cb[2 * j], s = cb[2 * j + 1], a = base[j], b = base[j + p->half_]; base[j] = a * c - b * s; base[j + p->half_] = b * c + a * s; }
    }
}
static void ref_attn(const float *q, const float *kc, const float *vc, float *o, const float *gate, const int *sel, const CcAttn *p) {
    int rep = p->H / p->KVH;
    for (int s = 0; s < p->S; s++) for (int h = 0; h < p->H; h++) {
        int kvh = h / rep, kbase = kvh * p->cap;
        const float *qs = q + p->q_off + s * p->q_row + h * p->q_seg;
        int n = p->pos_base + s + 1, list = 0, lb = 0;
        if (p->sel_row > 0) { lb = p->sel_off + s * p->sel_row; int c = sel[lb]; if (c >= 0) { n = c; list = 1; } }
        float m = -3.0e38f, l = 0.f, acc[256]; for (int d = 0; d < p->hd; d++) acc[d] = 0.f;
        for (int t0 = 0; t0 < n; t0 += 128) {
            int cnt = n - t0 < 128 ? n - t0 : 128; float sv[128], pe[128]; int pt[128]; float mx = -3.0e38f;
            for (int i = 0; i < cnt; i++) {
                int t = list ? sel[lb + 1 + t0 + i] : t0 + i; pt[i] = t;
                const float *kb = kc + p->k_off + (kbase + t) * p->hd; float a = 0.f;
                for (int d = 0; d < p->hd; d++) a += qs[d] * kb[d];
                sv[i] = a * p->scale; if (sv[i] > mx) mx = sv[i];
            }
            float mn = m > mx ? m : mx, sum = 0.f;
            for (int i = 0; i < cnt; i++) { pe[i] = expf(sv[i] - mn); sum += pe[i]; }
            float corr = expf(m - mn); l = l * corr + sum;
            for (int d = 0; d < p->hd; d++) acc[d] *= corr;
            for (int i = 0; i < cnt; i++) { const float *vb = vc + p->v_off + (kbase + pt[i]) * p->hd; for (int d = 0; d < p->hd; d++) acc[d] += pe[i] * vb[d]; }
            m = mn;
        }
        float inv = l > 0.f ? 1.f / l : 0.f; float *ob = o + p->o_off + s * p->o_row + h * p->hd;
        const float *gb = p->has_gate ? gate + p->g_off + s * p->g_row + h * p->g_seg : NULL;
        for (int d = 0; d < p->hd; d++) { float v = acc[d] * inv; if (gb) v *= 1.f / (1.f + expf(-gb[d])); ob[d] = v; }
    }
}
static void ref_dnconv(const float *in, const float *w, float *ring, float *out, float *snap, const CcDnConv *p) {
    int nh = p->CK - 1;
    for (int c = 0; c < p->CD; c++) {
        float hist[8] = {0}; for (int k = 0; k < nh; k++) hist[k] = ring[p->ring_off + c * nh + k];
        const float *wb = w + p->w_off + c * p->CK; float wl = wb[p->CK - 1];
        for (int s = 0; s < p->S; s++) {
            float cur = in[p->in_off + s * p->in_row + c], acc;
            if (p->order == 0) { acc = 0.f; for (int k = 0; k < nh; k++) acc += wb[k] * hist[k]; acc += wl * cur; }
            else { acc = wl * cur; for (int k = 0; k < nh; k++) acc += wb[k] * hist[k]; }
            out[p->out_off + s * p->out_row + c] = acc / (1.f + expf(-acc));
            for (int k = 0; k + 1 < nh; k++) hist[k] = hist[k + 1];
            if (nh > 0) hist[nh - 1] = cur;
            if (s == p->snap_row) for (int k = 0; k < nh; k++) snap[p->snap_off + c * nh + k] = hist[k];
        }
        for (int k = 0; k < nh; k++) ring[p->ring_off + c * nh + k] = hist[k];
    }
}
static float softplus(float x) { if (x > 20.f) return x; float e = expf(x); return e < 1e-4f ? e - 0.5f * e * e : logf(1.f + e); }
static void ref_dnrec(int KD, const float *cv, const float *ab, const float *z, float *st, const float *prm, float *y, float *snap, const CcDnRec *p) {
    for (int h = 0; h < p->VH; h++) {
        int kh = h / (p->VH / p->KH); float *S = st + p->st_off + h * KD * p->VD;
        float alog = prm[p->prm_off + h], dtb = prm[p->prm_off + p->VH + h]; const float *nw = prm + p->prm_off + 2 * p->VH;
        for (int s = 0; s < p->S; s++) {
            const float *cb = cv + p->cv_off + s * p->cv_row; float qn[256], kn[256], qa = 0.f, ka = 0.f;
            for (int k = 0; k < KD; k++) { qn[k] = cb[kh * KD + k]; kn[k] = cb[p->Ktot + kh * KD + k]; qa += qn[k] * qn[k]; ka += kn[k] * kn[k]; }
            float qsc = 1.f / sqrtf(qa + 1e-6f) * p->qscale, ksc = 1.f / sqrtf(ka + 1e-6f);
            for (int k = 0; k < KD; k++) { qn[k] *= qsc; kn[k] *= ksc; }
            float bv = ab[p->b_off + s * p->b_row + h], av = ab[p->a_off + s * p->a_row + h];
            float beta = 1.f / (1.f + expf(-bv)), decay = expf(-expf(alog) * softplus(av + dtb));
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
            for (int v = 0; v < p->VD; v++) { float zv = z[p->z_off + s * p->z_row + h * p->VD + v]; float g = (p->flags & 1) ? 1.f / (1.f + expf(-zv)) : zv / (1.f + expf(-zv)); y[p->y_off + s * p->y_row + h * p->VD + v] = o[v] * r * nw[v] * g; }
            if (s == p->snap_row) memcpy(snap + p->snap_off + h * KD * p->VD, S, (size_t)KD * p->VD * sizeof(float));
        }
    }
}
static float sig(float x) { return 1.f / (1.f + expf(-x)); }
static void ref_ew(float *Y, const float *A, const float *B, const float *Cc, const float *E, const CcEw *p) {
    for (int i = 0; i < p->n; i++) {
        if (p->op == 0) Y[p->y_off + i] = A[p->a_off + i] + B[p->b_off + i];
        else if (p->op == 1) { int r = i / p->D, d = i - r * p->D; float t = (p->flags & 1) ? B[p->b_off + r * p->D + d] : 0.f;
            if (p->flags & 2) { float g = (p->flags & 4) ? sig(E[p->e_off + r * p->e_row]) : 1.f; t = t + g * Cc[p->c_off + r * p->D + d]; }
            Y[p->y_off + i] = (p->flags & 8) ? t : A[p->a_off + i] + t; }
        else if (p->op == 2) { float g = A[p->a_off + i]; Y[p->y_off + i] = (g / (1.f + expf(-g))) * B[p->b_off + i]; }
        else if (p->op == 3) { float v = A[p->a_off + i] / p->fc; Y[p->y_off + i] = v * sig(v); }
        else if (p->op == 4) { int r = i / p->D, d = i - r * p->D, W = p->C * p->D; float v = 0.f; for (int k = 0; k < p->C; k++) { int j = r * W + k * p->D + d; v += sig(A[p->a_off + j]) * B[p->b_off + j]; } Y[p->y_off + i] = v / p->fc; }
        else if (p->op == 5) Y[p->y_off + i] = 2.f * sig(A[p->a_off + i] / p->fc);
        else if (p->op == 6) { int W = p->C * p->D, r = i / W, rem = i - r * W, k = rem / p->D, d = rem - k * p->D; Y[p->y_off + i] += A[p->a_off + r * p->C + k] * B[p->b_off + r * p->D + d]; }
        else if (p->op == 7) Y[p->y_off + i] = A[p->a_off + i] * p->fc;
        else if (p->op == 8) Y[p->y_off + i] = A[p->a_off + i] + E[p->e_off + i % p->D] * B[p->b_off + i];
    }
}

static void ref_qsa(const float *src, const float *w, float *pk, const float *cs, float *sc, int *sel, const CcQsa *p) {
    if (p->mode == 0) {
        for (int bi = 0; bi < p->nb; bi++) {
            int b = p->b0 + bi; float pool[256], ss = 0.f;
            for (int d = 0; d < p->ID; d++) { float v = 0.f; for (int r = 0; r < p->R; r++) v += src[p->src_off + (b * p->R + r) * p->ID + d] / (float)p->R; pool[d] = v; ss += v * v; }
            float rr = 1.f / sqrtf(ss / (float)p->ID + p->eps);
            for (int d = 0; d < p->ID; d++) pool[d] = pool[d] * rr * (1.f + w[p->w_off + d]);
            int cb = bi * 2 * p->half_;
            for (int d = 0; d < p->ID; d++) {
                float v = pool[d];
                if (d < p->half_) { float c = cs[cb + 2 * d], sn = cs[cb + 2 * d + 1]; v = pool[d] * c - pool[d + p->half_] * sn; }
                else if (d < 2 * p->half_) { int j = d - p->half_; float c = cs[cb + 2 * j], sn = cs[cb + 2 * j + 1]; v = pool[d] * c + pool[j] * sn; }
                pk[p->pk_off + b * p->ID + d] = v;
            }
        }
        return;
    }
    for (int s = 0; s < p->S; s++) {
        int visible = p->pos_base + s + 1, blocks = visible / p->R, take = blocks < p->budget / p->R ? blocks : p->budget / p->R, lb = s * p->sel_row;
        if (take >= blocks) { sel[lb] = -1; continue; }
        int sb = s * 2 * p->nbmax, qb = p->q_off + s * p->q_row;
        for (int b = 0; b < blocks; b++) { float score = 0.f; for (int h = 0; h < p->IQ; h++) { float a = 0.f; for (int d = 0; d < p->ID; d++) a += src[qb + h * p->ID + d] * pk[p->pk_off + b * p->ID + d]; if (a > 0.f) score += a; } sc[sb + b] = score / sqrtf((float)p->ID); }
        for (int b = 0; b < blocks; b++) { float v = sc[sb + b]; int rank = 0; for (int b2 = 0; b2 < blocks; b2++) { float v2 = sc[sb + b2]; if (v2 > v || (v2 == v && b2 < b)) rank++; } sc[sb + p->nbmax + b] = rank < take ? 1.f : 0.f; }
        int n = 0;
        for (int b = 0; b < blocks; b++) if (sc[sb + p->nbmax + b] != 0.f) for (int r = 0; r < p->R; r++) sel[lb + 1 + n++] = b * p->R + r;
        for (int t = blocks * p->R; t < visible; t++) sel[lb + 1 + n++] = t;
        sel[lb] = n;
    }
}
static void ref_ple(const float *keys, float *hyp, const float *val, const float *prm, float *gated, float *normv, const float *conv, float *ring, const CcPle *p) {
    int W = p->C * p->H;
    if (p->mode == 0) {
        for (int g = 0; g < p->S * p->C; g++) {
            int s = g / p->C, k = g - s * p->C, kb = p->keys_off + s * W + k * p->H, hb = p->hyp_off + s * W + k * p->H, vb = p->val_off + s * p->H;
            int wk = p->prm_off + k * p->H, wq = p->prm_off + W + k * p->H, wc = p->prm_off + 2 * W + k * p->H;
            float a = 0.f, b = 0.f;
            for (int d = 0; d < p->H; d++) { float u = keys[kb + d], v = hyp[hb + d]; a += u * u; b += v * v; }
            float rk = 1.f / sqrtf(a / (float)p->H + p->eps), rq = 1.f / sqrtf(b / (float)p->H + p->eps), dt = 0.f;
            for (int d = 0; d < p->H; d++) dt += (keys[kb + d] * rk * (1.f + prm[wk + d])) * (hyp[hb + d] * rq * (1.f + prm[wq + d]));
            float dot = dt / sqrtf((float)p->H), shaped = (dot > 0.f ? 1.f : dot < 0.f ? -1.f : 0.f) * sqrtf(fmaxf(fabsf(dot), 1e-6f));
            if (dot == 0.f) shaped = sqrtf(1e-6f);
            float gt = sig(shaped), c3 = 0.f;
            for (int d = 0; d < p->H; d++) { float u = gt * val[vb + d]; gated[s * W + k * p->H + d] = u; c3 += u * u; }
            float rc = 1.f / sqrtf(c3 / (float)p->H + p->eps);
            for (int d = 0; d < p->H; d++) normv[s * W + k * p->H + d] = gated[s * W + k * p->H + d] * rc * (1.f + prm[wc + d]);
        }
        return;
    }
    int SL = (p->CK - 1) * p->NG;
    for (int d = 0; d < W; d++) {
        float rg[32] = {0}; for (int t = 0; t < SL; t++) rg[t] = ring[p->ring_off + d * SL + t];
        int cw = p->conv_off + d * p->CK; float wl = conv[cw + p->CK - 1];
        for (int s = 0; s < p->S; s++) {
            float nv = normv[s * W + d], a = wl * nv;
            for (int t = 0; t < p->CK - 1; t++) a += conv[cw + t] * rg[t * p->NG < 32 ? t * p->NG : 0];
            hyp[p->hyp_off + s * W + d] += gated[s * W + d] + a * sig(a);
            for (int t = 0; t + 1 < SL; t++) rg[t] = rg[t + 1];
            if (SL > 0) rg[SL - 1] = nv;
            if (s == p->snap_row) for (int t = 0; t < SL; t++) ring[p->snap_off + d * SL + t] = rg[t];
        }
        for (int t = 0; t < SL; t++) ring[p->ring_off + d * SL + t] = rg[t];
    }
}

/* ---- helpers --------------------------------------------------------------------- */
static CcBuf *up(const std::vector<float> &v, int kind = CC_DEV) {
    CcBuf *b = T->buf(v.size() * 4, kind);
    if (!b || !T->begin() || !T->write(b, 0, v.data(), v.size() * 4) || !T->submit(1)) { printf("  FAIL upload\n"); fails++; }
    return b;
}
static std::vector<float> down(CcBuf *b, size_t n) {
    std::vector<float> v(n);
    if (!T->read(b, 0, v.data(), n * 4)) { printf("  FAIL read back\n"); fails++; }
    return v;
}
static std::vector<float> rnd(size_t n, unsigned seed, float scale = 1.f) { std::vector<float> v(n); unsigned s = seed; for (auto &x : v) x = frand(&s) * scale; return v; }

int main(void) {
    int dev = 0;
    if (!coli_cuda_init(&dev, 1)) { printf("no CUDA device: skipped\n"); return 0; }
    T = coli_cuda_chain_ops();
    ck(T != NULL && T->size == sizeof(ColiCudaChainOps), "the backend exports the chain's table");
    if (!T) return 1;
    ck(T->init(0), "a context on device 0");
    ck(T->ready() && !T->lost(), "ready, not lost");

    printf("buffers and frames\n");
    {
        std::vector<float> a = rnd(1000, 1);
        CcBuf *d = up(a), *u = T->buf(1000 * 4, CC_UP), *dn = T->buf(1000 * 4, CC_DOWN);
        ck(d && u && dn, "DEV, UP and DOWN buffers");
        memcpy(T->ptr(u), a.data(), 1000 * 4);
        CcBuf *d2 = T->buf(1000 * 4, CC_DEV);
        ck(T->begin() && T->copy(d2, 0, u, 0, 1000) && T->submit(0), "a frame left in flight: UP -> DEV");
        ck(T->begin() && T->copy(dn, 0, d2, 0, 1000) && T->submit(1), "the next frame ordered after it: DEV -> DOWN");
        ck(maxdiff((const float *)T->ptr(dn), a.data(), 1000) == 0, "the bytes went round");
        CcRegion rg[3] = {{0, 500, 10}, {100, 0, 10}, {990, 10, 10}};
        std::vector<float> want = a; for (int i = 0; i < 10; i++) { want[i] = a[500 + i]; want[100 + i] = a[i]; want[990 + i] = a[10 + i]; }
        ck(T->begin() && T->copy_regions(d2, d, rg, 3) && T->zero(d2, 200, 5) && T->submit(1), "regions copied and a range zeroed");
        for (int i = 0; i < 5; i++) want[200 + i] = 0;
        ck(maxdiff(down(d2, 1000).data(), want.data(), 1000) == 0, "as the references say");
        CcBuf *r = NULL;
        ck(T->reserve(&r, 64, CC_DEV) && T->bytes(r) >= 64 && T->reserve(&r, 32, CC_DEV) && T->bytes(r) >= 64 && T->reserve(&r, 4096, CC_DEV) && T->bytes(r) >= 4096, "reserve grows, never shrinks");
        T->free(d); T->free(u); T->free(dn); T->free(d2); T->free(r);
    }
    printf("matmul\n");
    {
        const int I = 96, O = 40, S = 3; unsigned s = 5;
        std::vector<int8_t> q((size_t)I * O); for (auto &v : q) v = (int8_t)(frand(&s) * 60);
        std::vector<float> sc(O); for (auto &v : sc) v = 0.02f + 0.01f * frand(&s);
        std::vector<float> x = rnd((size_t)S * I + 7, 9), ref((size_t)S * O);
        for (int r = 0; r < S; r++) for (int o = 0; o < O; o++) { float a = 0; for (int i = 0; i < I; i++) a += x[7 + r * I + i] * q[(size_t)o * I + i]; ref[r * O + o] = a * sc[o]; }
        ColiCudaTensor *t = NULL;
        ck(coli_cuda_tensor_upload(&t, q.data(), sc.data(), 1, I, O, 0), "an int8 tensor resident");
        CcBuf *xb = up(x), *yb = T->buf(((size_t)S * O + 3) * 4, CC_DEV);
        ck(T->begin() && T->matmul(t, xb, 7, yb, 3, S) && T->submit(1), "y = x @ W^T at offsets, three rows");
        std::vector<float> y = down(yb, (size_t)S * O + 3);
        ck(maxdiff(y.data() + 3, ref.data(), (size_t)S * O) < 1e-3, "equals the CPU GEMV");
        ck(T->begin() && T->matmul(t, xb, 7, yb, 3, 1) && T->submit(1) && maxdiff(down(yb, O + 3).data() + 3, ref.data(), O) < 1e-3, "and one row");
        {   /* a verify's rows: each row of a small S gets exactly the one-row call's bits */
            std::vector<float> one((size_t)S * O);
            for (int r = 0; r < S; r++) { T->begin(); T->matmul(t, xb, 7 + (size_t)r * I, yb, 0, 1); T->submit(1); std::vector<float> yr = down(yb, O); memcpy(&one[(size_t)r * O], yr.data(), O * 4); }
            ck(maxdiff(y.data() + 3, one.data(), (size_t)S * O) == 0, "three rows at once are bit for bit the three one-row calls");
        }
        {   /* a width the warp GEMV does not take (not a multiple of 16): the block kernel */
            const int I2 = 100; std::vector<int8_t> q2((size_t)I2 * O); for (auto &v : q2) v = (int8_t)(frand(&s) * 60);
            std::vector<float> x2 = rnd(I2, 13), ref2(O);
            for (int o = 0; o < O; o++) { float a = 0; for (int i = 0; i < I2; i++) a += x2[i] * q2[(size_t)o * I2 + i]; ref2[o] = a * sc[o]; }
            ColiCudaTensor *t2 = NULL; CcBuf *x2b = up(x2);
            ck(coli_cuda_tensor_upload(&t2, q2.data(), sc.data(), 1, I2, O, 0) && T->begin() && T->matmul(t2, x2b, 0, yb, 0, 1) && T->submit(1) &&
               maxdiff(down(yb, O).data(), ref2.data(), O) < 1e-3, "int8 rows of a width the warp GEMV leaves to the block kernel");
            coli_cuda_tensor_free(t2); T->free(x2b);
        }
        std::vector<float> w = rnd((size_t)I * O, 11), ones(O, 1.f), ref0(O);
        for (int o = 0; o < O; o++) { float a = 0; for (int i = 0; i < I; i++) a += x[7 + i] * w[(size_t)o * I + i]; ref0[o] = a; }
        ColiCudaTensor *t0 = NULL;
        ck(coli_cuda_tensor_upload(&t0, w.data(), ones.data(), 0, I, O, 0), "an f32 tensor resident");
        ck(T->begin() && T->matmul(t0, xb, 7, yb, 0, 1) && T->submit(1) && maxdiff(down(yb, O).data(), ref0.data(), O) < 1e-4, "f32 rows: the CPU's dot");
        coli_cuda_tensor_free(t); coli_cuda_tensor_free(t0); T->free(xb); T->free(yb);
    }
    printf("norm\n");
    {
        const int rows = 3, H = 4, hd = 32, D = 64;
        std::vector<float> x = rnd((size_t)rows * (H * hd + 8), 21, 2.f), w = rnd(D + hd, 22), ref = x;
        CcBuf *xb = up(x), *wb = up(w), *yb = up(x);   /* y starts as x: the elements past D of a row stay */
        CcNorm p = {rows, D, 1, 0, H * hd + 8, D, 0, H * hd + 8, D, 0, 0, CC_NORM_ADD1, 1e-6f, 1.f};
        ref_norm(x.data(), w.data(), ref.data(), &p);
        ck(T->begin() && T->norm(xb, wb, yb, &p) && T->submit(1) && maxdiff(down(yb, x.size()).data(), ref.data(), x.size()) < 1e-5, "zero-centred rows");
        CcNorm ph = {rows * H, hd, H, 0, H * hd + 8, hd, 0, H * hd + 8, hd, D, 0, 0, 1e-6f, 1.f};
        std::vector<float> refh = x; ref_norm(x.data(), w.data(), refh.data(), &ph);
        ck(T->begin() && T->norm(xb, wb, xb, &ph) && T->submit(1) && maxdiff(down(xb, x.size()).data(), refh.data(), x.size()) < 1e-5, "heads at a stride, plain weight, in place");
        CcNorm pn = {rows, D, 1, 0, H * hd + 8, D, 0, H * hd + 8, D, 0, 0, CC_NORM_NOW | CC_NORM_L2, 1e-6f, 0.5f};
        std::vector<float> refn = refh; ref_norm(refh.data(), NULL, refn.data(), &pn);
        ck(T->begin() && T->write(yb, 0, refh.data(), refh.size() * 4) && T->norm(xb, NULL, yb, &pn) && T->submit(1) && maxdiff(down(yb, x.size()).data(), refn.data(), x.size()) < 1e-5, "L2, no weight, post scale");
        T->free(xb); T->free(wb); T->free(yb);
    }
    printf("rope\n");
    {
        const int rows = 3, H = 4, qdim = 40, half = 8;
        std::vector<float> x = rnd((size_t)rows * H * qdim, 31), cs((size_t)rows * 2 * half), ref = x;
        for (int r = 0; r < rows; r++) for (int j = 0; j < half; j++) { float ang = (r + 5) * powf(10000.f, -2.f * j / (2 * half)); cs[(r * half + j) * 2] = cosf(ang); cs[(r * half + j) * 2 + 1] = sinf(ang); }
        CcRope p = {rows * H, H, 0, H * qdim, qdim, half, 0, 2 * half};
        ref_rope(ref.data(), cs.data(), &p);
        CcBuf *xb = up(x), *cb = up(cs, CC_UP);
        ck(T->begin() && T->rope(xb, cb, &p) && T->submit(1) && maxdiff(down(xb, x.size()).data(), ref.data(), x.size()) < 1e-5, "rotate-half from the host's table");
        T->free(xb); T->free(cb);
    }
    printf("attn\n");
    {
        const int H = 4, KV = 2, hd = 32, qdim = 2 * hd, cap = 300, pos_base = 140, S = 5, k_off = 64;
        std::vector<float> q = rnd((size_t)S * H * qdim, 41), kc = rnd((size_t)k_off + (size_t)KV * cap * hd, 42), vc = rnd((size_t)k_off + (size_t)KV * cap * hd, 43);
        std::vector<float> o((size_t)S * H * hd), ref = o;
        CcAttn p = {S, H, KV, hd, pos_base, cap, 0, H * qdim, qdim, hd, H * qdim, qdim, 1, 0, H * hd, 0, 0, 1.f / sqrtf((float)hd), k_off, k_off};
        ref_attn(q.data(), kc.data(), vc.data(), ref.data(), q.data(), NULL, &p);
        CcBuf *qb = up(q), *kb = up(kc), *vb = up(vc), *ob = T->buf(o.size() * 4, CC_DEV);
        ck(T->begin() && T->attn(qb, kb, vb, ob, qb, NULL, &p) && T->submit(1) && maxdiff(down(ob, o.size()).data(), ref.data(), o.size()) < 1e-4, "causal GQA with the gate, five rows after 140");
        std::vector<int> sel((size_t)S * 8); for (int s = 0; s < S; s++) { sel[s * 8] = s == 2 ? -1 : 6; for (int i = 0; i < 6; i++) sel[s * 8 + 1 + i] = (s * 17 + i * 23) % (pos_base + s + 1); }
        CcAttn ps = p; ps.sel_row = 8; ps.has_gate = 0;
        ref_attn(q.data(), kc.data(), vc.data(), ref.data(), NULL, sel.data(), &ps);
        CcBuf *sb = T->buf(sel.size() * 4, CC_DEV);
        ck(T->begin() && T->write(sb, 0, sel.data(), sel.size() * 4) && T->attn(qb, kb, vb, ob, NULL, sb, &ps) && T->submit(1) && maxdiff(down(ob, o.size()).data(), ref.data(), o.size()) < 1e-4, "a selection list, one row causal, no gate");
        T->free(qb); T->free(kb); T->free(vb); T->free(ob); T->free(sb);
    }
    printf("dnconv\n");
    {
        const int CD = 200, CK = 4, S = 5, PD = CD + 16;
        std::vector<float> in = rnd((size_t)S * PD, 51), w = rnd((size_t)CD * CK, 52), ring = rnd((size_t)CD * (CK - 1), 53), out((size_t)S * CD), snap((size_t)CD * (CK - 1));
        std::vector<float> rr = ring, ro = out, rs = snap;
        CcDnConv p = {S, CD, CK, 0, PD, 0, CD, 2, 0, 0, 0, 0};
        ref_dnconv(in.data(), w.data(), rr.data(), ro.data(), rs.data(), &p);
        CcBuf *ib = up(in), *wb = up(w), *rb = up(ring), *ob = T->buf(out.size() * 4, CC_DEV), *sb = T->buf(snap.size() * 4, CC_DEV);
        ck(T->begin() && T->dnconv(ib, wb, rb, ob, sb, &p) && T->submit(1), "the convolution over five rows");
        ck(maxdiff(down(ob, out.size()).data(), ro.data(), out.size()) < 1e-5 && maxdiff(down(rb, ring.size()).data(), rr.data(), ring.size()) == 0 && maxdiff(down(sb, snap.size()).data(), rs.data(), snap.size()) == 0, "outputs, the ring carried, the snapshot after row 2");
        CcDnConv p1 = p; p1.order = 1; p1.snap_row = -1;
        ref_dnconv(in.data(), w.data(), rr.data(), ro.data(), rs.data(), &p1);
        ck(T->begin() && T->dnconv(ib, wb, rb, ob, NULL, &p1) && T->submit(1) && maxdiff(down(ob, out.size()).data(), ro.data(), out.size()) < 1e-5 && maxdiff(down(rb, ring.size()).data(), rr.data(), ring.size()) == 0, "the other order, from the carried ring");
        T->free(ib); T->free(wb); T->free(rb); T->free(ob); T->free(sb);
    }
    printf("dnrec\n");
    for (int KD = 8; KD <= 128; KD += 120) {
        const int VH = 4, KH = 2, VD = KD == 8 ? 16 : 128, S = 4, Ktot = KH * KD, CD = 2 * Ktot + VH * VD, vd = VH * VD;
        std::vector<float> cv = rnd((size_t)S * CD, 61), ab = rnd((size_t)S * 2 * VH, 62), z = rnd((size_t)S * (CD + vd), 63), st = rnd((size_t)VH * KD * VD, 64, 0.1f);
        std::vector<float> prm(2 * VH + VD); unsigned s = 65; for (int h = 0; h < VH; h++) { prm[h] = -1.f + 0.5f * frand(&s); prm[VH + h] = 0.2f * frand(&s); } for (int v = 0; v < VD; v++) prm[2 * VH + v] = 1.f + 0.2f * frand(&s);
        std::vector<float> y((size_t)S * vd), snap((size_t)VH * KD * VD), rst = st, ry = y, rs = snap;
        CcDnRec p = {S, VH, KH, VD, Ktot, 0, CD, 0, 2 * VH, VH, 2 * VH, CD, CD + vd, 0, vd, 1, 0, 1e-6f, 1.f / sqrtf((float)KD), 0, 0, 0};
        ref_dnrec(KD, cv.data(), ab.data(), z.data(), rst.data(), prm.data(), ry.data(), rs.data(), &p);
        CcBuf *cb = up(cv), *bb = up(ab), *zb = up(z), *sb = up(st), *pb = up(prm), *yb = T->buf(y.size() * 4, CC_DEV), *nb = T->buf(snap.size() * 4, CC_DEV);
        char what[96];
        snprintf(what, sizeof what, "KD %d: four rows, the silu gate, z inside a wider row", KD);
        ck(T->begin() && T->dnrec(KD, cb, bb, zb, sb, pb, yb, nb, &p) && T->submit(1) && maxdiff(down(yb, y.size()).data(), ry.data(), y.size()) < 1e-4, what);
        ck(maxdiff(down(sb, st.size()).data(), rst.data(), st.size()) < 1e-4 && maxdiff(down(nb, snap.size()).data(), rs.data(), snap.size()) < 1e-4, "the state carried, the snapshot after row 1");
        CcDnRec p2 = p; p2.flags = 1; p2.snap_row = -1;
        ref_dnrec(KD, cv.data(), ab.data(), z.data(), rst.data(), prm.data(), ry.data(), rs.data(), &p2);
        ck(T->begin() && T->dnrec(KD, cb, bb, zb, sb, pb, yb, NULL, &p2) && T->submit(1) && maxdiff(down(yb, y.size()).data(), ry.data(), y.size()) < 1e-4 && maxdiff(down(sb, st.size()).data(), rst.data(), st.size()) < 1e-4, "the sigmoid gate, from the carried state");
        T->free(cb); T->free(bb); T->free(zb); T->free(sb); T->free(pb); T->free(yb); T->free(nb);
    }
    printf("ew\n");
    {
        const int R = 3, D = 40, C = 4, n = R * D;
        std::vector<float> a = rnd((size_t)R * C * D, 71), b = rnd((size_t)R * C * D, 72), c = rnd((size_t)n, 73), e = rnd((size_t)R * C + D, 74), y0 = rnd((size_t)R * C * D, 75);
        CcBuf *ab = up(a), *bb = up(b), *cb = up(c), *eb = up(e), *yb = T->buf(y0.size() * 4, CC_DEV);
        struct { int op, n, C, flags, e_row; float fc; const char *what; } cases[] = {
            {CC_EW_ADD, n, 1, 0, 1, 1.f, "ADD"}, {CC_EW_COMBINE, n, 1, 7, 1, 1.f, "COMBINE: routed, gated shared, residual"},
            {CC_EW_COMBINE, n, 1, 10, 1, 1.f, "COMBINE: the shared expert alone, no residual"}, {CC_EW_SWIGLU, n, 1, 0, 1, 1.f, "SWIGLU"},
            {CC_EW_HC_LOW, n, 1, 0, 1, 4.f, "HC_LOW"}, {CC_EW_HC_MIX, n, C, 0, 1, 4.f, "HC_MIX"}, {CC_EW_HC_INJ, n, 1, 0, 1, 4.f, "HC_INJ"},
            {CC_EW_HC_APPLY, R * C * D, C, 0, 1, 1.f, "HC_APPLY"}, {CC_EW_SCALE, n, 1, 0, 1, 0.25f, "SCALE"}, {CC_EW_GATE_ADD, n, 1, 0, 1, 1.f, "GATE_ADD"} };
        for (size_t k = 0; k < sizeof cases / sizeof *cases; k++) {
            CcEw p = {cases[k].op, cases[k].n, D, cases[k].C, cases[k].flags, cases[k].e_row, 0, 0, 0, 0, 0, cases[k].fc};
            std::vector<float> ref = y0;
            ref_ew(ref.data(), a.data(), b.data(), c.data(), e.data(), &p);
            int ok = T->begin() && T->write(yb, 0, y0.data(), y0.size() * 4) && T->ew(yb, ab, bb, cb, eb, &p) && T->submit(1);
            ck(ok && maxdiff(down(yb, y0.size()).data(), ref.data(), (size_t)p.n) < 1e-5, cases[k].what);
        }
        T->free(ab); T->free(bb); T->free(cb); T->free(eb); T->free(yb);
    }
    printf("qsa\n");
    {
        const int ID = 32, R = 4, nb = 7, half = 8, IQ = 3, S = 3, pos_base = 25, budget = 12, nbmax = 16, selrow = 1 + budget + R - 1;
        std::vector<float> ik = rnd((size_t)(nb + 2) * R * ID, 91), w = rnd(ID, 92), cs((size_t)nb * 2 * half), pk((size_t)nbmax * ID), rpk = pk;
        for (int b = 0; b < nb; b++) for (int j = 0; j < half; j++) { float ang = (float)(b * R) / powf(10000.f, (float)(2 * j) / (2 * half)); cs[(b * half + j) * 2] = cosf(ang); cs[(b * half + j) * 2 + 1] = sinf(ang); }
        CcQsa p0 = {0, ID, R, 0, half, 0, 0, 0, 0, 0, 0, 0, 0, 1e-6f, 0, 0, 0, nb};
        ref_qsa(ik.data(), w.data(), rpk.data(), cs.data(), NULL, NULL, &p0);
        CcBuf *ib = up(ik), *wb = up(w), *cb = up(cs, CC_UP), *pb = T->buf(pk.size() * 4, CC_DEV);
        ck(T->begin() && T->qsa(ib, wb, pb, cb, NULL, NULL, &p0) && T->submit(1) && maxdiff(down(pb, pk.size()).data(), rpk.data(), (size_t)nb * ID) < 1e-5, "seven pooled block keys, normalized and rotated");
        std::vector<float> iq = rnd((size_t)S * (IQ + 1) * ID, 93), sc((size_t)S * 2 * nbmax), rsc = sc; std::vector<int> sel((size_t)S * selrow), rsel = sel;
        CcQsa p1 = {1, ID, R, 0, 0, S, pos_base, budget, IQ, 0, (IQ + 1) * ID, nbmax, selrow, 1e-6f, 0, 0, 0, 0};
        ref_qsa(iq.data(), NULL, rpk.data(), NULL, rsc.data(), rsel.data(), &p1);
        CcBuf *qb = up(iq), *sb = T->buf(sc.size() * 4, CC_DEV), *lb = T->buf(sel.size() * 4, CC_DEV);
        int ok = T->begin() && T->qsa(qb, NULL, pb, NULL, sb, lb, &p1) && T->submit(1);
        std::vector<int> got(sel.size()); ok = ok && T->read(lb, 0, got.data(), got.size() * 4);
        int same = 1; for (size_t i = 0; i < got.size(); i++) if (got[i] != rsel[i]) same = 0;
        ck(ok && same, "three rows' selections: the taken blocks' positions, then the tail, as the CPU sorts");
        CcQsa p2 = p1; p2.pos_base = 5;   /* every block taken: the causal range */
        ref_qsa(iq.data(), NULL, rpk.data(), NULL, rsc.data(), rsel.data(), &p2);
        ok = T->begin() && T->qsa(qb, NULL, pb, NULL, sb, lb, &p2) && T->submit(1) && T->read(lb, 0, got.data(), got.size() * 4);
        ck(ok && got[0] == -1 && got[selrow] == -1 && got[2 * selrow] == -1, "few positions: every row attends to all of them (-1)");
        T->free(ib); T->free(wb); T->free(cb); T->free(pb); T->free(qb); T->free(sb); T->free(lb);
    }
    printf("ple\n");
    {
        const int S = 3, C = 4, H = 24, W = C * H, CK = 3, NG = 3, SL = (CK - 1) * NG;
        std::vector<float> keys = rnd((size_t)S * W, 101), hyp = rnd((size_t)S * W, 102), val = rnd((size_t)S * H, 103), prm = rnd(3 * W, 104, 0.2f);
        std::vector<float> conv = rnd((size_t)W * CK, 105, 0.5f), ring = rnd((size_t)2 * W * SL, 106), gated((size_t)S * W), normv((size_t)S * W);
        std::vector<float> rh = hyp, rg = gated, rn = normv, rr = ring;
        CcPle g = {0, S, C, H, CK, NG, 0, 0, 0, -1, 0, 1e-6f, 0, 0, 0};
        ref_ple(keys.data(), rh.data(), val.data(), prm.data(), rg.data(), rn.data(), NULL, NULL, &g);
        CcBuf *kb = up(keys), *hb = up(hyp), *vb = up(val), *pb = up(prm), *gb = T->buf(gated.size() * 4, CC_DEV), *nb = T->buf(normv.size() * 4, CC_DEV);
        CcBuf *cb = up(conv), *rb = up(ring);
        ck(T->begin() && T->ple(kb, hb, vb, pb, gb, nb, NULL, NULL, &g) && T->submit(1) &&
           maxdiff(down(gb, gated.size()).data(), rg.data(), gated.size()) < 1e-5 && maxdiff(down(nb, normv.size()).data(), rn.data(), normv.size()) < 1e-5, "the gate over twelve (row, stream) pairs");
        CcPle cv = {1, S, C, H, CK, NG, 0, 0, 0, 1, W * SL, 1e-6f, 0, 0, 0};
        ref_ple(NULL, rh.data(), NULL, NULL, rg.data(), rn.data(), conv.data(), rr.data(), &cv);
        ck(T->begin() && T->ple(NULL, hb, NULL, NULL, gb, nb, cb, rb, &cv) && T->submit(1) &&
           maxdiff(down(hb, hyp.size()).data(), rh.data(), hyp.size()) < 1e-5 && maxdiff(down(rb, ring.size()).data(), rr.data(), ring.size()) < 1e-6, "the dilated convolution, the ring carried, the copy after row 1");
        T->free(kb); T->free(hb); T->free(vb); T->free(pb); T->free(gb); T->free(nb); T->free(cb); T->free(rb);
    }
    printf("experts\n");
    {
        /* the route's op against a CPU reference: three int4 experts with per-row scales (the
         * container's fmt 2), then gate/up grouped (fmt 4, gs 32) over an int8 down (--down-bits),
         * one expert alone, and a refusal that must not lose the device */
        const int D = 64, I = 96, N = 3; unsigned s = 77;
        std::vector<int8_t> q4[N][3]; std::vector<uint8_t> pk[N][3]; std::vector<float> sc[N][3];
        /* the host's nibbles are offset-binary (value + 8), as the container stores them; the upload turns them signed */
        auto pack = [&](std::vector<int8_t> &q, std::vector<uint8_t> &p) { p.assign(q.size() / 2, 0); for (size_t b = 0; b < p.size(); b++) p[b] = (uint8_t)(((q[2 * b] + 8) & 15) | (((q[2 * b + 1] + 8) & 15) << 4)); };
        for (int c = 0; c < N; c++) for (int k = 0; k < 3; k++) {
            size_t n = (size_t)D * I; q4[c][k].resize(n); for (auto &v : q4[c][k]) v = (int8_t)floorf(frand(&s) * 8.f); for (auto &v : q4[c][k]) if (v > 7) v = 7;
            pack(q4[c][k], pk[c][k]);
            int O = k < 2 ? I : D; sc[c][k].resize(O); for (auto &v : sc[c][k]) v = 0.02f + 0.01f * frand(&s);
        }
        std::vector<float> x = rnd(D, 78), vals = {0.5f, 0.3f, 0.2f};
        /* the reference, in double; fmt 2 everywhere */
        auto ref_y = [&](int count, const std::vector<float> *gsc, const std::vector<float> *usc, int gs, const std::vector<int8_t> *d8) {
            std::vector<float> y(D, 0.f);
            for (int c = 0; c < count; c++) {
                std::vector<double> h(I);
                for (int o = 0; o < I; o++) {
                    double g = 0, u = 0;
                    for (int i = 0; i < D; i++) {
                        double gw = q4[c][0][(size_t)o * D + i], uw = q4[c][1][(size_t)o * D + i];
                        if (gs) { gw *= gsc[c][(size_t)o * (D / gs) + i / gs]; uw *= usc[c][(size_t)o * (D / gs) + i / gs]; }
                        g += x[i] * gw; u += x[i] * uw;
                    }
                    if (!gs) { g *= sc[c][0][o]; u *= sc[c][1][o]; }
                    h[o] = (g / (1.0 + exp(-g))) * u;
                }
                for (int o = 0; o < D; o++) {
                    double a = 0;
                    for (int i = 0; i < I; i++) a += h[i] * (double)(d8 ? d8[c][(size_t)o * I + i] : q4[c][2][(size_t)o * I + i]);
                    y[o] += (float)(vals[c] * a * sc[c][2][o]);
                }
            }
            return y;
        };
        ColiCudaTensor *tg[N] = {0}, *tu[N] = {0}, *td[N] = {0};
        int okup = 1;
        for (int c = 0; c < N; c++) okup &= coli_cuda_tensor_upload(&tg[c], pk[c][0].data(), sc[c][0].data(), 2, D, I, 0) &&
                                          coli_cuda_tensor_upload(&tu[c], pk[c][1].data(), sc[c][1].data(), 2, D, I, 0) &&
                                          coli_cuda_tensor_upload(&td[c], pk[c][2].data(), sc[c][2].data(), 2, I, D, 0);
        ck(okup, "nine int4 tensors resident");
        CcBuf *xb = up(x), *yb = T->buf((size_t)(D + 5) * 4, CC_DEV);
        std::vector<float> want = ref_y(N, NULL, NULL, 0, NULL);
        ck(T->begin() && T->experts(tg, tu, td, vals.data(), N, xb, 0, yb, 5) && T->submit(1), "three experts of one row, y at an offset");
        std::vector<float> got = down(yb, D + 5);
        { char b[120]; snprintf(b, sizeof b, "equals the CPU reference (per-row int4): max diff %.2e", maxdiff(got.data() + 5, want.data(), D)); ck(maxdiff(got.data() + 5, want.data(), D) < 1e-4, b); }
        want = ref_y(1, NULL, NULL, 0, NULL);
        ck(T->begin() && T->experts(tg, tu, td, vals.data(), 1, xb, 0, yb, 0) && T->submit(1) && maxdiff(down(yb, D).data(), want.data(), D) < 1e-4, "one expert alone");
        /* grouped gate/up (gs 32: two groups per row) over an int8 down */
        std::vector<float> gsc[N], usc[N]; std::vector<int8_t> d8[N];
        ColiCudaTensor *tg4[N] = {0}, *tu4[N] = {0}, *td8[N] = {0};
        for (int c = 0; c < N; c++) {
            gsc[c].resize((size_t)I * (D / 32)); usc[c].resize((size_t)I * (D / 32));
            for (auto &v : gsc[c]) v = 0.02f + 0.01f * frand(&s); for (auto &v : usc[c]) v = 0.02f + 0.01f * frand(&s);
            d8[c].resize((size_t)D * I); for (auto &v : d8[c]) v = (int8_t)(frand(&s) * 60);
            okup &= coli_cuda_tensor_upload_g(&tg4[c], pk[c][0].data(), gsc[c].data(), 4, D, I, 0, 32) &&
                    coli_cuda_tensor_upload_g(&tu4[c], pk[c][1].data(), usc[c].data(), 4, D, I, 0, 32) &&
                    coli_cuda_tensor_upload(&td8[c], d8[c].data(), sc[c][2].data(), 1, I, D, 0);
        }
        ck(okup, "grouped int4 and int8 tensors resident");
        want = ref_y(N, gsc, usc, 32, d8);
        ck(T->begin() && T->experts(tg4, tu4, td8, vals.data(), N, xb, 0, yb, 0) && T->submit(1) && maxdiff(down(yb, D).data(), want.data(), D) < 1e-4,
           "grouped gate/up scales over an int8 down (--down-bits) equal the reference");
        {   /* gs 16: a 16-byte load would straddle two groups; refused without a launch, the device kept */
            ColiCudaTensor *bad = NULL; std::vector<float> bsc((size_t)I * (D / 16), 0.02f);
            ck(coli_cuda_tensor_upload_g(&bad, pk[0][0].data(), bsc.data(), 4, D, I, 0, 16), "a gs-16 tensor resident");
            ColiCudaTensor *bg[1] = {bad};
            ck(T->begin() && !T->experts(bg, tu4, td8, vals.data(), 1, xb, 0, yb, 0) && !T->lost() && T->submit(1), "refused, the device not lost");
            if (bad) coli_cuda_tensor_free(bad);
        }
        CcStats st; T->stats(&st);
        ck(st.experts == 7, "the counter saw seven experts");
        for (int c = 0; c < N; c++) { coli_cuda_tensor_free(tg[c]); coli_cuda_tensor_free(tu[c]); coli_cuda_tensor_free(td[c]); coli_cuda_tensor_free(tg4[c]); coli_cuda_tensor_free(tu4[c]); coli_cuda_tensor_free(td8[c]); }
        T->free(xb); T->free(yb);
    }
    printf("lost\n");
    {
        CcStats st; T->stats(&st);
        ck(st.frames > 10 && st.ops > 30 && st.matmuls == 7, "the counters saw the frames, ops and matmuls");
        setenv("COLI_GPU_FAIL_AFTER", "0", 1);
        std::vector<float> w = rnd(64, 81), ones(8, 1.f);
        ColiCudaTensor *t = NULL;
        coli_cuda_tensor_upload(&t, w.data(), ones.data(), 0, 8, 8, 0);
        CcBuf *xb = T->buf(64, CC_DEV), *yb = T->buf(64, CC_DEV);
        int r = T->begin() && T->matmul(t, xb, 0, yb, 0, 1);
        ck(!r && T->lost(), "a failed launch loses the device to the chain");
        ck(!T->begin() && !T->norm(xb, NULL, yb, NULL), "every later call returns 0");
        unsetenv("COLI_GPU_FAIL_AFTER");
        if (t) coli_cuda_tensor_free(t);
    }
    T->shutdown();
    coli_cuda_shutdown();
    if (fails) { printf("test_cuda_chain: %d failure(s)\n", fails); return 1; }
    printf("OK test_cuda_chain: every op of the CUDA chain matches its reference\n");
    return 0;
}
