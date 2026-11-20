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
 *   dnconv    both orders, the ring carried across calls, the snapshot row; LFM3's (order 2)
 *   dnrec     KD 8 and 128, silu and sigmoid gates, the state carried, the snapshot
 *   ew        every element-wise op
 *   lost      a failed launch (COLI_GPU_FAIL_AFTER) marks the device lost to the chain
 * Needs a device: run by hand (make sycl-chain-check SYCL=1, or the binary with --device N /
 * COLI_GPUS=N; --list prints every GPU's profile and exits, the first guarded contact); sycl-test
 * builds it. tests/test_cuda_chain.cu's migration (c/sycl/tools/dpct-migrate.sh): the ops are the
 * ABI's, the one SYCL thing here is the planar experts' host memory (host USM). */
#include "../../backend_cuda.h"
#define COLI_CUDA_CHAIN_NO_WRAPPERS
#include "../../cuda_chain.h"
#include "../coli_sycl.h"
#include "sycl_test_args.h"
#define KV16_INTEGER   /* kv16.h's integer rounding: the reference of ew's CC_EW_F16 */
#include "../../kv16.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>

static int fails;
static void ck(int ok, const char *what) { if (ok) { printf("  ok   %s\n", what); return; } printf("  FAIL %s\n", what); fails++; }
static float frand(unsigned *s) { *s = *s * 1664525u + 1013904223u; return ((*s >> 8) & 0xFFFF) / 65535.f * 2.f - 1.f; }
/* a bit-for-bit comparison's diagnostics: the count of differing words, the first one (row, column
 * of a D-wide row), both values and their distance in ulps (the first card: addnorm's norm differed
 * by one ulp in some rows, a different 1 / sqrt lowering per kernel; MIGRATION.md section 13) */
static int bits_differ(const char *what, const float *a, const float *b, size_t n, int D) {
    size_t bad = 0, first = 0; int maxulp = 0;
    for (size_t i = 0; i < n; i++) {
        uint32_t ua, ub; memcpy(&ua, &a[i], 4); memcpy(&ub, &b[i], 4);
        if (ua == ub) continue;
        if (!bad) first = i;
        bad++;
        int32_t ia = (int32_t)(ua & 0x7fffffffu), ib = (int32_t)(ub & 0x7fffffffu);
        if ((ua ^ ub) & 0x80000000u) ia = -ia;
        int ulp = ia > ib ? ia - ib : ib - ia;
        if (ulp > maxulp) maxulp = ulp;
    }
    if (bad)
        printf("       %s: %zu of %zu words differ, up to %d ulp; first at row %zu col %zu: %a (%.9g) vs %a (%.9g)\n",
               what, bad, n, maxulp, D > 0 ? first / (size_t)D : 0, D > 0 ? first % (size_t)D : first, a[first], a[first], b[first], b[first]);
    return bad != 0;
}
static double maxdiff(const float *a, const float *b, size_t n) { double d = 0; for (size_t i = 0; i < n; i++) { double x = fabs((double)a[i] - b[i]); if (x > d) d = x; } return d; }
static const ColiCudaChainOps *T;
static int g_dev;   /* the device under test: --device N, COLI_GPUS=N, else 0 */

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
/* ref_attn with a sliding window of win positions (0: all), a sink logit per head
 * (snk[sink_off + h], NULL: none) and a ring of `ring` rows (position t in row t % ring; 0: none) */
static void ref_attn_w(const float *q, const float *kc, const float *vc, float *o, const CcAttn *p, int win, const float *snk,
                       int sink_off, int ring) {
    int rep = p->H / p->KVH;
    std::vector<float> sv((size_t)p->pos_base + p->S);
    for (int s = 0; s < p->S; s++) for (int h = 0; h < p->H; h++) {
        int kvh = h / rep, pos = p->pos_base + s, lo = win > 0 && pos - win + 1 > 0 ? pos - win + 1 : 0;
        const float *qs = q + p->q_off + s * p->q_row + h * p->q_seg;
        float m = snk ? snk[sink_off + h] : -3.0e38f;
        for (int t = lo; t <= pos; t++) {
            const float *kb = kc + p->k_off + (size_t)(kvh * p->cap + (ring ? t % ring : t)) * p->hd; float a = 0.f;
            for (int d = 0; d < p->hd; d++) a += qs[d] * kb[d];
            sv[t] = a * p->scale; if (sv[t] > m) m = sv[t];
        }
        double l = snk ? exp((double)snk[sink_off + h] - m) : 0.0, acc[256] = {0};
        for (int t = lo; t <= pos; t++) {
            double e = exp((double)sv[t] - m); l += e;
            const float *vb = vc + p->v_off + (size_t)(kvh * p->cap + (ring ? t % ring : t)) * p->hd;
            for (int d = 0; d < p->hd; d++) acc[d] += e * vb[d];
        }
        float *ob = o + p->o_off + s * p->o_row + h * p->hd;
        for (int d = 0; d < p->hd; d++) ob[d] = (float)(acc[d] / l);
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
            const float *r = in + p->in_off + s * p->in_row;   /* order 2 (LFM3): B | C | x */
            float cur = p->order == 2 ? r[c] * r[c + 2 * p->CD] : r[c], acc;
            if (p->order != 1) { acc = 0.f; for (int k = 0; k < nh; k++) acc += wb[k] * hist[k]; acc += wl * cur; }
            else { acc = wl * cur; for (int k = 0; k < nh; k++) acc += wb[k] * hist[k]; }
            out[p->out_off + s * p->out_row + c] = p->order == 2 ? r[c + p->CD] * acc : acc / (1.f + expf(-acc));
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
        else if (p->op == 12) Y[p->y_off + i] = A[p->a_off + i] * sig(B[p->b_off + i / p->D]);
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

/* ---- the decision engines' encoders (chain_enc.comp's modes; the engines' own sums) --- */
static float enc_act(float x, int k) {
    if (k == 1) return 0.5f * x * (1.0f + erff(x * 0.70710678118654752f));
    if (k == 2) return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
    if (k == 3) return x > 0.f ? x : 0.f;
    return x;
}
static void ref_enc_norm(float *a, const float *b, const float *prm, float *y, const CcEncNorm *p) {
    std::vector<float> t((size_t)p->D);
    for (int r = 0; r < p->rows; r++) {
        double s = 0, q = 0;
        for (int i = 0; i < p->D; i++) {
            float v = a[(size_t)p->a_off + (size_t)r * p->a_row + i];
            if (p->flags & CC_ENC_ADD) v = v + b[(size_t)p->b_off + (size_t)r * p->b_row + i];
            t[i] = v; s += v;
        }
        double mean = s / p->D;
        for (int i = 0; i < p->D; i++) { double c = t[i] - mean; q += c * c; }
        float rstd = (float)(1.0 / sqrt(q / p->D + (double)p->eps));
        for (int i = 0; i < p->D; i++) {
            if (p->flags & CC_ENC_SUM) a[(size_t)p->a_off + (size_t)r * p->a_row + i] = t[i];
            float o = (float)(t[i] - mean) * rstd * prm[p->w_off + i];
            if (p->bias_off >= 0) o += prm[p->bias_off + i];
            y[(size_t)p->y_off + (size_t)r * p->y_row + i] = o;
        }
    }
}
static void ref_enc_attn(const float *qkv, const float *c2p, const float *p2c, float *o, const int *rng, const int *ridx,
                         const int *pos, const CcEncAttn *p) {
    const int S = p->S, hd = p->hd;
    for (int h = 0; h < p->H; h++)
        for (int i = 0; i < S; i++) {
            float *out = o + (size_t)p->o_off + (size_t)i * p->o_row + (size_t)h * hd;
            int lo = rng[2 * i], hi = rng[2 * i + 1];
            for (int d = 0; d < hd; d++) out[d] = 0.f;
            if (lo > hi) continue;
            std::vector<double> sc((size_t)(hi - lo + 1));
            double mx = -1e300, sum = 0;
            const float *q = qkv + (size_t)p->q_off + (size_t)i * p->kv_row + (size_t)h * hd;
            for (int j = lo; j <= hi; j++) {
                const float *k = qkv + (size_t)p->k_off + (size_t)j * p->kv_row + (size_t)h * hd;
                double s = 0;
                for (int d = 0; d < hd; d++) s += (double)q[d] * k[d];
                if (p->flags & (CC_ENC_C2P | CC_ENC_P2C)) {
                    int rr = ridx[pos[i] - pos[j] + p->rc] - p->r0;
                    if (p->flags & CC_ENC_C2P) s += c2p[(size_t)p->c_off + ((size_t)h * S + i) * p->nr + rr];
                    if (p->flags & CC_ENC_P2C) s += p2c[(size_t)p->p_off + ((size_t)h * S + j) * p->nr + rr];
                }
                sc[j - lo] = s * p->scale;
                if (sc[j - lo] > mx) mx = sc[j - lo];
            }
            for (auto &v : sc) { v = exp(v - mx); sum += v; }
            for (int d = 0; d < hd; d++) {
                double acc = 0;
                for (int j = lo; j <= hi; j++) acc += sc[j - lo] * qkv[(size_t)p->v_off + (size_t)j * p->kv_row + (size_t)h * hd + d];
                out[d] = (float)(acc / sum);
            }
        }
}
static void ref_enc_rel(const float *x, const float *pt, float *y, const CcEncRel *p) {
    for (int h = 0; h < p->H; h++)
        for (int i = 0; i < p->S; i++)
            for (int t = 0; t < p->nr; t++) {
                double s = 0;
                for (int d = 0; d < p->hd; d++)
                    s += (double)x[(size_t)p->x_off + (size_t)i * p->x_row + (size_t)h * p->hd + d] *
                         pt[(size_t)p->p_off + (size_t)(p->r0 + t) * p->p_row + (size_t)h * p->hd + d];
                y[(size_t)p->y_off + ((size_t)h * p->S + i) * p->nr + t] = (float)s;
            }
}
static CcBuf *upi(const std::vector<int> &v) {
    CcBuf *b = T->buf(v.size() * 4, CC_DEV);
    if (!b || !T->begin() || !T->write(b, 0, v.data(), v.size() * 4) || !T->submit(1)) { printf("  FAIL upload\n"); fails++; }
    return b;
}
/* A chunk as long as an 8192-token prompt's last, Qwen3.6's attention: 2043 rows after 6144
 * positions, H 16, KV 2, hd 256, on the tensor cores where the card has them (in one pass from
 * sm_75; a T4's blocks of 16 keys). 2043 rows leave the last block of 8 tokens 3, the ragged
 * edge. Four rows within 1e-4 of the reference (the kernel's error is ~4e-6, and half a k16
 * wrong misses by ~8e-3: the means over 8000 keys are small), and a float16 cache of the same
 * values (attn_h) gives the float32 cache's bits. */
static void prompt_rows_one_pass_long(void) {
    const int H = 16, KV = 2, hd = 256, S = 2043, pos = 6144, cap = 8384, qdim = 2 * hd, k_off = 32;
    std::vector<float> q = rnd((size_t)S * H * qdim, 61), kc = rnd((size_t)k_off + (size_t)KV * cap * hd, 62),
                       vc = rnd(kc.size(), 63);
    kv16_round(kc.data(), (int64_t)kc.size()); kv16_round(vc.data(), (int64_t)vc.size());   /* the same values in both caches */
    size_t n = (size_t)S * H * hd;
    CcAttn p = {S, H, KV, hd, pos, cap, 0, H * qdim, qdim, hd, H * qdim, qdim, 1, 0, H * hd, 0, 0, 1.f / sqrtf((float)hd), k_off, k_off};
    CcBuf *qb = up(q), *kb = up(kc), *vb = up(vc), *ob = T->buf(n * 4, CC_DEV);
    int ran = T->begin() && T->attn(qb, kb, vb, ob, qb, NULL, &p) && T->submit(1);
    std::vector<float> o = down(ob, n);
    double d = 0;
    for (int r : {0, 1, 1024, 2042}) {   /* a row alone: one position more each */
        CcAttn pr = p; pr.S = 1; pr.pos_base = pos + r; pr.q_off = r * p.q_row; pr.g_off = p.g_off + r * p.g_row; pr.o_off = 0;
        std::vector<float> ref((size_t)H * hd);
        ref_attn(q.data(), kc.data(), vc.data(), ref.data(), q.data(), NULL, &pr);
        d = fmax(d, maxdiff(o.data() + (size_t)r * H * hd, ref.data(), ref.size()));
    }
    ck(ran && d < 1e-4, "prompt rows, a long chunk: H 16 KV 2 hd 256, 2043 rows after 6144, rows 0, 1, 1024 and 2042");
    printf("  the long chunk, rows 0, 1, 1024 and 2042: max |diff| %.2e\n", d);
    if (T->size < offsetof(ColiCudaChainOps, copy_regions_h) + sizeof T->copy_regions_h || !T->kvh()) printf("  (no float16 cache here)\n");
    else {
        std::vector<uint16_t> kh(kc.size()), vh(vc.size());
        kv16_pack(kh.data(), kc.data(), (int64_t)kc.size()); kv16_pack(vh.data(), vc.data(), (int64_t)vc.size());
        CcBuf *khb = T->buf(kh.size() * 2, CC_DEV), *vhb = T->buf(vh.size() * 2, CC_DEV), *o16 = T->buf(n * 4, CC_DEV);
        int ok = ran && khb && vhb && o16 && T->begin() && T->write(khb, 0, kh.data(), kh.size() * 2) &&
                 T->write(vhb, 0, vh.data(), vh.size() * 2) && T->attn_h(qb, khb, vhb, o16, qb, NULL, &p) && T->submit(1);
        ck(ok && !memcmp(o.data(), down(o16, n).data(), n * 4), "and from a float16 cache of the same values, the float32 cache's bits");
        T->free(khb); T->free(vhb); T->free(o16);
    }
    T->free(qb); T->free(kb); T->free(vb); T->free(ob);
}

/* the router's pick (chain_route.comp's rules: the lower expert on a tie, groups by their two best) */
static void ref_route(const float *l, const float *bias, int *ids, float *w, const CcRoute *p) {
    std::vector<float> sc(p->E), pk(p->E);
    for (int s = 0; s < p->S; s++) {
        const float *lr = l + p->l_off + s * p->l_row;
        auto lg = [&](int e) { float v = lr[e]; if (p->has_bias == 2) v += bias[p->b_off + e]; return v / p->temp; };
        if (p->kind == 0) {
            float m = -3.4e38f; for (int e = 0; e < p->E; e++) m = fmaxf(m, lg(e));
            double sum = 0; for (int e = 0; e < p->E; e++) { sc[e] = expf(lg(e) - m); sum += sc[e]; }
            for (int e = 0; e < p->E; e++) sc[e] = (float)(sc[e] / sum);
        } else for (int e = 0; e < p->E; e++) { float v = lg(e); sc[e] = p->kind == 1 ? 1.f / (1.f + expf(-v)) : sqrtf(v > 20.f ? v : logf(1.f + expf(v))); }
        for (int e = 0; e < p->E; e++) pk[e] = sc[e] + (p->has_bias == 1 ? bias[p->b_off + e] : 0.f);
        if (p->n_group > 1) {
            int per = p->E / p->n_group; std::vector<float> gv(p->n_group); std::vector<int> keep(p->n_group, 0);
            for (int g = 0; g < p->n_group; g++) { float b1 = -3.4e38f, b2 = -3.4e38f; for (int e = g * per; e < g * per + per; e++) { if (pk[e] > b1) { b2 = b1; b1 = pk[e]; } else if (pk[e] > b2) b2 = pk[e]; } gv[g] = b1 + b2; }
            for (int k = 0; k < p->topk_group; k++) { int bg = -1; float bv = -3.4e38f; for (int g = 0; g < p->n_group; g++) if (!keep[g] && gv[g] > bv) { bv = gv[g]; bg = g; } keep[bg] = 1; }
            for (int e = 0; e < p->E; e++) if (!keep[e / per]) pk[e] = -3.4e38f;
        }
        std::vector<int> ch(p->K); float sum = 0;
        for (int k = 0; k < p->K; k++) {
            int be = -1; float bv = -3.4e38f;
            for (int e = 0; e < p->E; e++) { int used = 0; for (int j = 0; j < k; j++) used |= ch[j] == e; if (!used && pk[e] > bv) { bv = pk[e]; be = e; } }
            ch[k] = be; sum += be >= 0 ? sc[be] : 0.f;
        }
        float div = sum + p->norm_eps; int nrm = (p->norm == 1 || (p->norm == 2 && p->K > 1)) && div > 0.f;
        for (int k = 0; k < p->K; k++) { float x = ch[k] >= 0 ? sc[ch[k]] : 0.f; if (nrm) x /= div; ids[p->i_off + s * p->i_row + k] = ch[k]; w[p->w_off + s * p->w_row + k] = x * p->scale; }
    }
}
/* e4m3's values (quant.h's E4M3_LUT): sign, four exponent bits (bias 7), three mantissa bits */
static float e4m3(int b) {
    int sg = b & 0x80, ex = (b >> 3) & 15, m = b & 7;
    float v = ex == 0 ? ldexpf((float)m / 8.f, -6) : (ex == 15 && m == 7) ? NAN : ldexpf(1.f + m / 8.f, ex - 7);
    return sg ? -v : v;
}
/* one expert matrix [O][I] in a tier format, its values for the reference (dq) and the tensor */
/* moe_mat's int4-per-gs matrix (the same seed, the same codes and scales) in expert_ffn.h's
 * planar blocks (byte k of a 64-weight block: weight k low, k + 32 high, both v + 8), in
 * host USM the device reads over the bus (its device address the host's): its scales likewise.
 * The pointers to free with sycl::free on the device's queue. */
static int moe_planar(int gs, int I, int O, unsigned seed, uint8_t **hw,
                      float **hs, const void **dw, const float **ds) {
    unsigned s = seed; int ng = I / gs;
    std::vector<float> sc((size_t)O * ng);
    for (auto &v : sc) v = 0.05f + 0.02f * frand(&s);
    *hw = (uint8_t *)sycl::malloc_host((size_t)O * I / 2, coli_sycl_queue(g_dev));
    *hs = (float *)sycl::malloc_host(sc.size() * 4, coli_sycl_queue(g_dev));
    if (!*hw || !*hs) return 0;
    memset(*hw, 0, (size_t)O * I / 2);
    for (int o = 0; o < O; o++) for (int i = 0; i < I; i += 2) {
        int a = (int)(frand(&s) * 7.9f), b = (int)(frand(&s) * 7.9f);
        for (int t = 0; t < 2; t++) {
            int e = i + t, v = (t ? b : a) + 8, blk = e / 64, k = e % 64;
            uint8_t &byte = (*hw)[(size_t)o * I / 2 + (size_t)blk * 32 + (k & 31)];
            byte = (uint8_t)(k < 32 ? (byte & 0xF0) | v : (byte & 0x0F) | (v << 4));
        }
    }
    memcpy(*hs, sc.data(), sc.size() * 4);
    *dw = *hw; *ds = *hs;
    return 1;
}
static ColiCudaTensor *moe_mat(int fmt, int gs, int I, int O, unsigned seed, std::vector<float> &dq) {
    unsigned s = seed; dq.assign((size_t)O * I, 0.f);
    ColiCudaTensor *t = NULL; int ok = 0;
    if (fmt == 1) {
        std::vector<int8_t> q((size_t)O * I); std::vector<float> sc(O);
        for (int o = 0; o < O; o++) { sc[o] = 0.01f + 0.005f * frand(&s); for (int i = 0; i < I; i++) { q[(size_t)o * I + i] = (int8_t)(frand(&s) * 100); dq[(size_t)o * I + i] = q[(size_t)o * I + i] * sc[o]; } }
        ok = coli_cuda_tensor_upload(&t, q.data(), sc.data(), 1, I, O, g_dev);
    } else if (fmt == 2 || fmt == 4) {
        int ng = fmt == 4 ? I / gs : 1;
        std::vector<uint8_t> q((size_t)O * I / 2); std::vector<float> sc((size_t)O * ng);
        for (auto &v : sc) v = 0.05f + 0.02f * frand(&s);
        for (int o = 0; o < O; o++) for (int i = 0; i < I; i += 2) {
            int a = (int)(frand(&s) * 7.9f), b = (int)(frand(&s) * 7.9f);
            q[((size_t)o * I + i) / 2] = (uint8_t)((a + 8) | ((b + 8) << 4));   /* offset binary, the upload's form */
            float sa = sc[(size_t)o * ng + (fmt == 4 ? i / gs : 0)], sb = sc[(size_t)o * ng + (fmt == 4 ? (i + 1) / gs : 0)];
            dq[(size_t)o * I + i] = a * sa; dq[(size_t)o * I + i + 1] = b * sb;
        }
        ok = fmt == 4 ? coli_cuda_tensor_upload_g(&t, q.data(), sc.data(), 4, I, O, g_dev, gs) : coli_cuda_tensor_upload(&t, q.data(), sc.data(), 2, I, O, g_dev);
    } else {
        int nbI = (I + 127) / 128, nbO = (O + 127) / 128;
        std::vector<uint8_t> q((size_t)O * I); std::vector<float> sc((size_t)nbO * nbI);
        for (auto &v : sc) v = 0.004f + 0.002f * frand(&s);
        for (int o = 0; o < O; o++) for (int i = 0; i < I; i++) {
            int b; do { b = (int)((frand(&s) * 0.5f + 0.5f) * 255.f) & 255; } while ((b & 0x7f) == 0x7f || ((b >> 3) & 15) > 12);
            q[(size_t)o * I + i] = (uint8_t)b; dq[(size_t)o * I + i] = e4m3(b) * sc[(size_t)(o / 128) * nbI + i / 128];
        }
        ok = coli_cuda_tensor_upload(&t, q.data(), sc.data(), 8, I, O, g_dev);
    }
    return ok ? t : NULL;
}

/* COLI_CUDA_CHAIN_BENCH=adec: one decode row's attention over L = 10 layers of caches of their
 * own, float16 (attn_h; COLI_CUDA_BENCH_KV16=0 float32), COLI_CUDA_BENCH_ADEC "H KVH hd pos"
 * (Qwen3.6's 16 2 256 18000 by default): the time a layer, the best of five frames */
static void bench_adec(void) {
    int H = 16, KVH = 2, hd = 256, pos = 18000, n = 10;
    const char *sh = getenv("COLI_CUDA_BENCH_ADEC"), *k16 = getenv("COLI_CUDA_BENCH_KV16");
    if (sh && *sh) sscanf(sh, "%d %d %d %d", &H, &KVH, &hd, &pos);
    int h16 = !(k16 && *k16 == '0') && T->size >= offsetof(ColiCudaChainOps, copy_regions_h) + sizeof T->copy_regions_h && T->kvh();
    size_t kv = (size_t)KVH * (pos + 1) * hd;
    std::vector<float> q = rnd((size_t)H * hd, 5), k = rnd(kv, 6);
    std::vector<uint16_t> kh(kv);
    kv16_round(k.data(), (int64_t)kv); kv16_pack(kh.data(), k.data(), (int64_t)kv);
    CcBuf *qb = up(q), *ob = T->buf((size_t)H * hd * 4, CC_DEV), *kb[10], *vb[10];
    int ok = qb && ob;
    for (int l = 0; l < n; l++) {
        kb[l] = T->buf(kv * (h16 ? 2 : 4), CC_DEV); vb[l] = T->buf(kv * (h16 ? 2 : 4), CC_DEV);
        ok = ok && kb[l] && vb[l] && T->begin() && T->write(kb[l], 0, h16 ? (const void *)kh.data() : (const void *)k.data(), kv * (h16 ? 2 : 4)) &&
             T->write(vb[l], 0, h16 ? (const void *)kh.data() : (const void *)k.data(), kv * (h16 ? 2 : 4)) && T->submit(1);
    }
    double best = 1e9;
    for (int r = 0; r < 5 && ok; r++) {
        ok = T->begin();
        for (int l = 0; l < n && ok; l++) {
            CcAttn a = {1, H, KVH, hd, pos, pos + 1, 0, H * hd, hd, 0, 0, 0, 0, 0, H * hd, 0, 0, 1.f / sqrtf((float)hd), 0, 0};
            ok = h16 ? T->attn_h(qb, kb[l], vb[l], ob, NULL, NULL, &a) : T->attn(qb, kb[l], vb[l], ob, NULL, NULL, &a);
        }
        auto t0 = std::chrono::steady_clock::now();
        ok = T->submit(1) && ok;
        double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (dt < best) best = dt;
    }
    if (ok) printf("bench adec H %d/%d hd %d at %d positions%s: %.1f us a layer (%.1f GB/s of K and V)\n", H, KVH, hd, pos,
                   h16 ? ", float16" : "", best * 1e6 / n, 2.0 * kv * (h16 ? 2 : 4) * n / best / 1e9);
    else printf("bench adec: the frame failed\n");
    for (int l = 0; l < n; l++) { T->free(kb[l]); T->free(vb[l]); }
    T->free(qb); T->free(ob);
}

/* COLI_CUDA_CHAIN_BENCH=route: 48 routes in a frame, each its own row (E 256, K 8 by default:
 * COLI_CUDA_BENCH_ROUTE "E K"), the best of five. RTX PRO 4000: E 256 K 8 4.9 us, E 64 K 8 3.6;
 * one warp a row with no __syncthreads (as chain_route.comp's few experts) took 6.0 and 5.0 */
static void bench_route(void) {
    int E = 256, K = 8, nl = 48;
    const char *sh = getenv("COLI_CUDA_BENCH_ROUTE");
    if (sh && *sh) sscanf(sh, "%d %d", &E, &K);
    std::vector<float> l = rnd((size_t)E * nl, 7, 3.f);
    CcBuf *lb = up(l), *ib = T->buf((size_t)K * 4 * nl, CC_DEV), *wb = T->buf((size_t)K * 4 * nl, CC_DEV);
    int ok = lb && ib && wb;
    double best = 1e9;
    for (int r = -1; r < 5 && ok; r++) {
        ok = T->begin();
        for (int i = 0; i < nl && ok; i++) {
            CcRoute p = {1, E, K, 0, i * E, E, 0, 0, i * K, K, i * K, K, 1, 1, 1, 1.f, 1.f, 0.f};
            ok = T->route(lb, NULL, ib, wb, &p);
        }
        auto t0 = std::chrono::steady_clock::now();
        ok = T->submit(1) && ok;
        double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (r >= 0 && dt < best) best = dt;
    }
    if (ok) printf("bench route E %d K %d: %.1f us a route (%d in a frame)\n", E, K, best * 1e6 / nl, nl);
    else printf("bench route: the frame failed\n");
    T->free(lb); T->free(ib); T->free(wb);
}


/* ---- the performance pass's benches (perf m-kernels, 2026-11-07): the engine's decode-critical
 * kernels through the chain's own table, pipelined in one frame as a decode token pays them (the
 * host clock around begin .. submit(1): submissions, launch gaps and the wait included), rotating
 * over enough copies of the operands to run past the device's L2 (a 16 MB projection would sit in
 * Ponte Vecchio's 108 MB otherwise). COLI_CUDA_BENCH_N launches a frame (200), the best of five
 * frames; every mode checks its last answer against the CPU. ---------------------------------- */
static double bench_now(void) { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static int bench_n(void) { const char *e = getenv("COLI_CUDA_BENCH_N"); int n = e && *e ? atoi(e) : 200; return n < 1 ? 1 : n; }
static size_t bench_pool(void) { const char *e = getenv("COLI_CUDA_BENCH_POOL_MB"); return (size_t)(e && *e ? atoi(e) : 768) << 20; }

/* Independent CPU reference for the opt-in prompt activation policy. Keep the
 * original f32-vs-f16 gate intact when the option is off. */
static bool prompt_s8(int I, int S, int fmt) {
    const char *e = getenv("COLI_SYCL_XMX_S8");
    const auto &p = coli_sycl_profile_for(g_dev);
    return e && atoi(e) == 1 && fmt == 1 && S > 8 && I % 64 == 0 && I <= 131071 &&
           coli_sycl_xmx_supported(p, COLI_SYCL_DT_F16) && (p.xmx_shapes & COLI_SYCL_XMX_INT8);
}
static std::vector<float> s8_reference_rows(const std::vector<float> &x, int I) {
    std::vector<float> q(x.size());
    for (size_t r = 0; r < x.size(); r += I) {
        float mx = 0; for (int i = 0; i < I; i++) mx = fmaxf(mx, fabsf(x[r + i]));
        float scale = mx > 0 ? mx / 127.f : 1.f;
        for (int i = 0; i < I; i++)
            q[r + i] = fminf(127.f, fmaxf(-127.f, nearbyintf(x[r + i] / scale))) * scale;
    }
    return q;
}

/* COLI_CUDA_CHAIN_BENCH=gemv: the int8 GEMV of a decode step's one row (cc_matmul: gemv_i8_w4 with
 * its split, or gemv_i8_rows, as COLI_CUDA_GEMV_W4 / COLI_CUDA_GEMV_SK choose; COLI_SYCL_SG the
 * sub-group) over the shapes of COLI_CUDA_BENCH_GEMV ("OxI,OxI[,S]..": Qwen3.6-35B's projections and
 * lm_head by default), S rows (1; a third number "OxIxS"). Bytes: the rows, the scales, x and y. */
static void bench_gemv(void) {
    const char *sh = getenv("COLI_CUDA_BENCH_GEMV");
    struct Shape { int O, I, S; };
    std::vector<Shape> shapes;
    if (sh && *sh) {
        for (const char *c = sh; *c; ) {
            Shape t = {0, 0, 1};
            if (sscanf(c, "%dx%dx%d", &t.O, &t.I, &t.S) < 2) break;
            shapes.push_back(t);
            while (*c && *c != ',') c++;
            if (*c == ',') c++;
        }
    } else shapes = {{8192, 2048, 1}, {2048, 8192, 1}, {12288, 2048, 1}, {2048, 4096, 1}, {512, 2048, 1}, {248320, 2048, 1}};
    const int N = bench_n();
    for (const Shape &sp : shapes) {
        const int O = sp.O, I = sp.I, S = sp.S;
        if (O < 1 || I < 16 || I % 16 || S < 1) { printf("bench gemv: bad shape %d x %d x %d\n", O, I, S); continue; }
        size_t bytes = (size_t)O * I + (size_t)O * 4;
        int nc = (int)((bench_pool() + bytes - 1) / bytes); if (nc < 1) nc = 1; if (nc > 1024) nc = 1024;
        std::vector<int8_t> q((size_t)O * I);
        for (size_t i = 0; i < q.size(); i++) q[i] = (int8_t)((i * 37u + (i >> 7)) % 201u) - 100;
        std::vector<float> sc(O); for (int o = 0; o < O; o++) sc[o] = 0.01f + 0.001f * (o % 7);
        std::vector<ColiCudaTensor *> t(nc, nullptr); int ok = 1;
        for (int c = 0; c < nc && ok; c++) ok = coli_cuda_tensor_upload(&t[c], q.data(), sc.data(), 1, I, O, g_dev);
        std::vector<float> x = rnd((size_t)S * I, 3);
        CcBuf *xb = up(x), *yb = T->buf((size_t)S * O * 4, CC_DEV);
        ok = ok && xb && yb;
        double best = 1e9, sub = 0;
        for (int r = -1; r < 5 && ok; r++) {
            double t0 = bench_now();
            ok = T->begin();
            for (int i = 0; i < N && ok; i++) ok = T->matmul(t[i % nc], xb, 0, yb, 0, S);
            double t1 = bench_now();
            ok = T->submit(1) && ok;
            double dt = bench_now() - t0;
            if (r >= 0 && dt < best) { best = dt; sub = t1 - t0; }
        }
        int same = 0;
        if (ok) {   /* the last launch's tensor against the CPU, the last row */
            std::vector<float> y = down(yb, (size_t)S * O);
            std::vector<float> xr = x;
            const auto &p = coli_sycl_profile_for(g_dev);
            if (prompt_s8(I, S, 1)) xr = s8_reference_rows(x, I);
            else if (S > 8 && I % 64 == 0 && coli_sycl_xmx_supported(p, COLI_SYCL_DT_F16))
                for (float &v : xr) v = (float)sycl::half(fminf(65504.f, fmaxf(-65504.f, v)));
            const int8_t *w = q.data(); double worst = 0, scale = 0;
            for (int o = 0; o < O; o += (O > 4096 ? 97 : 1)) {
                double a = 0; for (int i = 0; i < I; i++) a += (double)xr[(size_t)(S - 1) * I + i] * w[(size_t)o * I + i];
                a *= sc[o]; double d = fabs(a - y[(size_t)(S - 1) * O + o]); if (d > worst) worst = d; if (fabs(a) > scale) scale = fabs(a);
            }
            same = worst <= 1e-4 * scale + 1e-5;
        }
        double per = best / N, gbs = (double)(bytes + (size_t)S * (I + O) * 4) / per / 1e9;
        if (ok) printf("bench gemv %d x %d int8, %d row%s (%d copies, %d launches a frame): %.2f us a launch pipelined (%.2f us of it submitting), %.0f GB/s%s\n",
                       O, I, S, S > 1 ? "s" : "", nc, N, per * 1e6, sub / N * 1e6, gbs, same ? "" : "  [FAIL: the CPU's dot differs]");
        if (ok) {   /* the bits of every row of the last launch (FNV-1a over y): equal across kernels that keep a decode step's bits */
            std::vector<float> yh = down(yb, (size_t)S * O);
            uint64_t h = 1469598103934665603ull, h0 = h; const unsigned char *yp = (const unsigned char *)yh.data();
            for (size_t i = 0; i < yh.size() * sizeof(float); i++) { h ^= yp[i]; h *= 1099511628211ull; }
            for (size_t i = 0; i < (size_t)O * sizeof(float); i++) { h0 ^= yp[i]; h0 *= 1099511628211ull; }   /* the first row alone */
            printf("GEMV HASH %d x %d x %d: all rows %016llx, row 0 %016llx\n", O, I, S, (unsigned long long)h, (unsigned long long)h0);
        }
        else printf("bench gemv %d x %d: the frame failed\n", O, I);
        for (int c = 0; c < nc; c++) if (t[c]) coli_cuda_tensor_free(t[c]);
        T->free(xb); T->free(yb);
    }
}

/* an int4-per-64 expert matrix [O][I] in the backend's upload form (offset-binary nibbles, weight
 * 2j low), its dequantized values kept for the reference when dq is given */
static ColiCudaTensor *bench_expert(int I, int O, int gs, unsigned seed, std::vector<float> *dq) {
    unsigned s = seed; int ng = I / gs;
    std::vector<uint8_t> q((size_t)O * I / 2); std::vector<float> sc((size_t)O * ng);
    for (auto &v : sc) v = 0.05f + 0.02f * frand(&s);
    if (dq) dq->assign((size_t)O * I, 0.f);
    for (size_t j = 0; j < q.size(); j++) {
        s = s * 1664525u + 1013904223u;
        int a = (int)((s >> 8) & 15) - 8, b = (int)((s >> 12) & 15) - 8;
        if (a < -7) a = -7; if (b < -7) b = -7;
        q[j] = (uint8_t)((a + 8) | ((b + 8) << 4));
        if (dq) { size_t o = (2 * j) / I, i = (2 * j) % I; (*dq)[2 * j] = a * sc[o * ng + i / gs]; (*dq)[2 * j + 1] = b * sc[o * ng + (i + 1) / gs]; }
    }
    ColiCudaTensor *t = NULL;
    return coli_cuda_tensor_upload_g(&t, q.data(), sc.data(), 4, I, O, g_dev, gs) ? t : NULL;
}
/* COLI_CUDA_CHAIN_BENCH=moe: a decode step's routed experts (cc_moe parts 3: moe_gv_gu_k, moe_gv_dn_k
 * and moe_sum_k; COLI_CUDA_MOE_GV=0 the warp-a-row pair) of K experts of D inputs and I hidden in int4
 * per 64 (COLI_CUDA_BENCH_MOE "D I K L": Qwen3.6-35B's 2048 512 8 and L 40 layers of experts of their
 * own, the pool past the L2). Bytes: the three matrices and their scales of the K experts, x, y. */
static void bench_moe(void) {
    int D = 2048, I = 512, K = 8, L = 40;
    const char *sh = getenv("COLI_CUDA_BENCH_MOE");
    if (sh && *sh) sscanf(sh, "%d %d %d %d", &D, &I, &K, &L);
    const int N = bench_n(), gs = 64;
    if (D % 64 || I % 64 || K < 1 || K > CC_ROUTE_KMAX || L < 1) { printf("bench moe: bad shape\n"); return; }
    std::vector<ColiCudaTensor *> ts; ts.reserve((size_t)L * K * 3);
    std::vector<CcExpert> tab((size_t)L * K + 2);
    std::vector<std::vector<float>> G0(K), U0(K), D0(K);   /* layer 0's values, the reference */
    unsigned seed = 900; int made = 1;
    for (int l = 0; l < L && made; l++) for (int e = 0; e < K && made; e++) {
        ColiCudaTensor *g = bench_expert(D, I, gs, seed++, l ? NULL : &G0[e]), *u = bench_expert(D, I, gs, seed++, l ? NULL : &U0[e]), *d = bench_expert(I, D, gs, seed++, l ? NULL : &D0[e]);
        made = g && u && d;
        if (!made) break;
        ts.push_back(g); ts.push_back(u); ts.push_back(d);
        CcExpert &x = tab[2 + (size_t)l * K + e]; memset(&x, 0, sizeof x);
        T->tensor_dev(g, &x.g, &x.gs, NULL); T->tensor_dev(u, &x.u, &x.us, NULL); T->tensor_dev(d, &x.d, &x.ds, NULL);
    }
    std::vector<float> x = rnd(D, 901), w(K, 1.f / K);
    std::vector<int> ids(K); for (int k = 0; k < K; k++) ids[k] = k;
    CcBuf *tb = T->buf(tab.size() * sizeof(CcExpert), CC_DEV), *xb = up(x), *ib = T->buf((size_t)K * 4, CC_DEV), *wb = up(w);
    CcBuf *hb = T->buf((size_t)K * (I + D) * 4, CC_DEV), *yb = T->buf((size_t)D * 4, CC_DEV);
    int ok = made && tb && xb && ib && wb && hb && yb && T->begin() && T->write(tb, 0, tab.data(), tab.size() * sizeof(CcExpert)) &&
             T->write(ib, 0, ids.data(), ids.size() * 4) && T->submit(1);
    double best = 1e9, sub = 0;
    for (int r = -1; r < 5 && ok; r++) {
        double t0 = bench_now();
        ok = T->begin();
        for (int i = 0; i < N && ok; i++) {
            CcMoe m = {1, K, D, I, 4, gs, 0, 3, 2 + (i % L) * K, 0, D, 0, K, 0, K, 0, D, -1, K, 0, 0, 0, 0};
            ok = T->moe(tb, xb, ib, wb, hb, yb, NULL, &m);
        }
        double t1 = bench_now();
        ok = T->submit(1) && ok;
        double dt = bench_now() - t0;
        if (r >= 0 && dt < best) { best = dt; sub = t1 - t0; }
    }
    int same = 0;
    if (ok) {   /* layer 0 once more, against the CPU */
        CcMoe m = {1, K, D, I, 4, gs, 0, 3, 2, 0, D, 0, K, 0, K, 0, D, -1, K, 0, 0, 0, 0};
        ok = T->begin() && T->moe(tb, xb, ib, wb, hb, yb, NULL, &m) && T->submit(1);
        std::vector<float> y = down(yb, D), ref(D, 0.f), h(I);
        for (int e = 0; e < K; e++) {
            for (int o = 0; o < I; o++) {
                double g = 0, u = 0;
                for (int i = 0; i < D; i++) { g += x[i] * G0[e][(size_t)o * D + i]; u += x[i] * U0[e][(size_t)o * D + i]; }
                h[o] = (float)(g / (1 + exp(-g)) * u);
            }
            for (int o = 0; o < D; o++) { double v = 0; for (int i = 0; i < I; i++) v += h[i] * D0[e][(size_t)o * I + i]; ref[o] += (float)(v * w[e]); }
        }
        double scale = 0; for (float v : ref) scale = fmax(scale, fabs(v));
        same = ok && maxdiff(y.data(), ref.data(), D) < 2e-5 * scale + 1e-6;
    }
    size_t bytes = (size_t)K * (3 * (size_t)D * I / 2 + 3 * ((size_t)D * I / gs) * 4) + (size_t)(D + D) * 4;
    double per = best / N;
    if (ok) printf("bench moe %d experts of %d x %d int4-g64, one row (%d layers, %d calls a frame): %.2f us a call pipelined (%.2f us of it submitting), %.0f GB/s%s\n",
                   K, D, I, L, N, per * 1e6, sub / N * 1e6, bytes / per / 1e9, same ? "" : "  [FAIL: the CPU's experts differ]");
    else printf("bench moe: the frame failed\n");
    for (ColiCudaTensor *t : ts) coli_cuda_tensor_free(t);
    T->free(tb); T->free(xb); T->free(ib); T->free(wb); T->free(hb); T->free(yb);
}

/* COLI_CUDA_CHAIN_BENCH=dnrec: the Gated DeltaNet recurrence of a decode row (cc_dnrec_k<KD>: a block
 * of 128 a value head, the state's column in registers) over L layers' states of their own
 * (COLI_CUDA_BENCH_DNREC "VH KH KD VD L S": Qwen3.6-35B's 32 16 128 128, L 30, S 1; S from 8 takes
 * the prompt's split kernels, a per-row figure to compare). Bytes: the state read and written, the
 * row's q, k, v, z, y. */
static void bench_dnrec(void) {
    int VH = 32, KH = 16, KD = 128, VD = 128, L = 30, S = 1;
    const char *sh = getenv("COLI_CUDA_BENCH_DNREC");
    if (sh && *sh) sscanf(sh, "%d %d %d %d %d %d", &VH, &KH, &KD, &VD, &L, &S);
    const int N = bench_n();
    if (VH < 1 || KH < 1 || VH % KH || VD < 1 || VD > 128 || (KD != 64 && KD != 128 && KD != 8) || L < 1 || S < 1) { printf("bench dnrec: bad shape\n"); return; }
    const int Ktot = KH * KD, CD = 2 * Ktot + VH * VD, vd = VH * VD;
    std::vector<float> cv = rnd((size_t)S * CD, 61), ab = rnd((size_t)S * 2 * VH, 62), z = rnd((size_t)S * (CD + vd), 63), st0 = rnd((size_t)VH * KD * VD, 64, 0.1f);
    std::vector<float> prm(2 * VH + VD); unsigned s = 65;
    for (int h = 0; h < VH; h++) { prm[h] = -1.f + 0.5f * frand(&s); prm[VH + h] = 0.2f * frand(&s); }
    for (int v = 0; v < VD; v++) prm[2 * VH + v] = 1.f + 0.2f * frand(&s);
    CcDnRec p = {S, VH, KH, VD, Ktot, 0, CD, 0, 2 * VH, VH, 2 * VH, CD, CD + vd, 0, vd, -1, 0, 1e-6f, 1.f / sqrtf((float)KD), 0, 0, 0};
    CcBuf *cb = up(cv), *bb = up(ab), *zb = up(z), *pb = up(prm), *yb = T->buf((size_t)S * vd * 4, CC_DEV);
    std::vector<CcBuf *> sb(L, nullptr); int ok = cb && bb && zb && pb && yb;
    for (int l = 0; l < L && ok; l++) { sb[l] = up(st0); ok = sb[l] != nullptr; }
    double best = 1e9, sub = 0;
    for (int r = -1; r < 5 && ok; r++) {
        double t0 = bench_now();
        ok = T->begin();
        for (int i = 0; i < N && ok; i++) ok = T->dnrec(KD, cb, bb, zb, sb[i % L], pb, yb, NULL, &p);
        double t1 = bench_now();
        ok = T->submit(1) && ok;
        double dt = bench_now() - t0;
        if (r >= 0 && dt < best) { best = dt; sub = t1 - t0; }
    }
    int same = 0;
    if (ok) {   /* a fresh state once, against the CPU */
        CcBuf *fb = up(st0);
        std::vector<float> rst = st0, ry((size_t)S * vd);
        ref_dnrec(KD, cv.data(), ab.data(), z.data(), rst.data(), prm.data(), ry.data(), NULL, &p);
        ok = fb && T->begin() && T->dnrec(KD, cb, bb, zb, fb, pb, yb, NULL, &p) && T->submit(1);
        same = ok && maxdiff(down(yb, ry.size()).data(), ry.data(), ry.size()) < 1e-4 && maxdiff(down(fb, rst.size()).data(), rst.data(), rst.size()) < 1e-4;
        T->free(fb);
    }
    size_t bytes = 2 * (size_t)VH * KD * VD * 4 + (size_t)S * (CD + 2 * VH + CD + vd + vd) * 4;
    double per = best / N;
    if (ok) printf("bench dnrec VH %d KH %d KD %d VD %d, %d row%s (%d states, %d calls a frame): %.2f us a call pipelined (%.2f us of it submitting), %.1f us a row, %.0f GB/s%s\n",
                   VH, KH, KD, VD, S, S > 1 ? "s" : "", L, N, per * 1e6, sub / N * 1e6, per * 1e6 / S, bytes / per / 1e9, same ? "" : "  [FAIL: the CPU's recurrence differs]");
    else printf("bench dnrec: the frame failed\n");
    T->free(cb); T->free(bb); T->free(zb); T->free(pb); T->free(yb);
    for (int l = 0; l < L; l++) if (sb[l]) T->free(sb[l]);
}

int main(int argc, char **argv) {
    int devices[COLI_CUDA_MAX_DEVICES];
    if (sycl_test_devices(argc, argv, devices, COLI_CUDA_MAX_DEVICES) < 0) return 0;   /* --list */
    g_dev = devices[0];
    if (!coli_cuda_init(&g_dev, 1)) { printf("no GPU (device %d): skipped\n", g_dev); return 0; }
    T = coli_cuda_chain_ops();
    ck(T != NULL && T->size == sizeof(ColiCudaChainOps), "the backend exports the chain's table");
    if (!T) return 1;
    ck(T->init(g_dev), "a context on the device");
    ck(T->ready() && !T->lost(), "ready, not lost");
    ck(T->preload && T->preload() && T->ready() && !T->lost(), "the chain's and the backend's kernels loaded before their first launch");
    const char *be = getenv("COLI_CUDA_CHAIN_BENCH");
    if (be && !strcmp(be, "adec")) { bench_adec(); return 0; }
    if (be && !strcmp(be, "route")) { bench_route(); return 0; }
    if (be && !strcmp(be, "gemv")) { bench_gemv(); return 0; }
    if (be && !strcmp(be, "moe")) { bench_moe(); return 0; }
    if (be && !strcmp(be, "dnrec")) { bench_dnrec(); return 0; }

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
    {   /* copies of up to 1 MiB by a kernel (cc_copy_k: no copy engine), past it by the engine, at
         * offsets 16-byte aligned or not; scattered regions out of mapped memory by a kernel too */
        const size_t N = ((size_t)1 << 18) + 1000;
        std::vector<float> a = rnd(N, 71);
        CcBuf *u = T->buf(N * 4, CC_UP), *d = T->buf(N * 4, CC_DEV), *dn = T->buf(N * 4, CC_DOWN), *d3 = T->buf(4096 * 4, CC_DEV);
        int ok = u && d && dn && d3;
        if (ok) memcpy(T->ptr(u), a.data(), N * 4);
        ok = ok && T->begin() && T->copy(d, 0, u, 0, N) && T->copy(dn, 3, d, 5, 777) && T->copy(dn, 1000, d, 1000, (size_t)1 << 18) && T->submit(1);
        int same = ok;
        for (size_t i = 0; i < 777 && same; i++) same = ((const float *)T->ptr(dn))[3 + i] == a[5 + i];
        for (size_t i = 0; i < ((size_t)1 << 18) && same; i++) same = ((const float *)T->ptr(dn))[1000 + i] == a[1000 + i];
        ck(same, "1 MiB + 4000 bytes by the engine, 777 floats unaligned and 1 MiB by the kernel");
        CcRegion rg[3] = {{7, 33, 5}, {200, 1, 13}, {50, 900, 3}};
        std::vector<float> want(4096, 0.f);
        for (auto &x : rg) memcpy(want.data() + x.dst, a.data() + x.src, x.n * 4);
        ck(ok && T->begin() && T->zero(d3, 0, 4096) && T->copy_regions(d3, u, rg, 3) && T->submit(1) &&
           maxdiff(down(d3, 4096).data(), want.data(), 4096) == 0, "scattered regions out of mapped memory");
        T->free(u); T->free(d); T->free(dn); T->free(d3);
    }
    {   /* K rows into a head-major cache: a region per row and head, one kernel for the grid
         * (two regions or more, from a decode step's KV heads up to a prompt chunk); off the
         * grid, the copies as before */
        for (int S : {1, 2, 100}) {
            const int H = S < 10 ? 2 : 4, hd = 32, DR = 256, pb = 7;
            for (int shuffle = 0; shuffle < (S == 100 ? 2 : 1); shuffle++) {
                std::vector<float> src = rnd((size_t)S * H * hd, 61 + S + shuffle), dst = rnd((size_t)H * DR * hd, 63);
                std::vector<CcRegion> rg((size_t)S * H);
                for (int s = 0; s < S; s++) for (int h = 0; h < H; h++)
                    rg[(size_t)s * H + h] = {((size_t)h * DR + pb + s) * hd, (size_t)s * H * hd + (size_t)h * hd, (size_t)hd};
                if (shuffle) std::swap(rg[0], rg[(size_t)S * H - 1]);
                std::vector<float> want = dst;
                for (auto &x : rg) memcpy(want.data() + x.dst, src.data() + x.src, x.n * 4);
                CcBuf *sb = up(src), *db = up(dst);
                ck(T->begin() && T->copy_regions(db, sb, rg.data(), (int)rg.size()) && T->submit(1) &&
                   maxdiff(down(db, dst.size()).data(), want.data(), dst.size()) == 0,
                   shuffle ? "400 regions off the grid: a copy each" :
                   S == 1 ? "2 regions on a grid (decode KV): one kernel" :
                   S == 2 ? "4 regions on a grid (verify KV): one kernel" : "400 regions on a grid: one kernel");
                T->free(sb); T->free(db);
            }
        }
    }
    if (T->size >= offsetof(ColiCudaChainOps, host_up_done) + sizeof T->host_up_done) {
        /* host_up: page-locked host bytes to a device address on the copy stream, a ticket each;
         * many in flight (past the marks' ring), each landing where it was sent */
        const size_t n = 1 << 18;
        std::vector<float> b = rnd(n, 42);
        CcBuf *db = T->buf(n * 4, CC_DEV), *hb2 = T->buf(n * 4, CC_UP);
        int ok = db && hb2;
        if (ok) memcpy(T->ptr(hb2), b.data(), n * 4);
        float *dd = ok ? (float *)T->dev_addr(db) : NULL;
        unsigned t = 0, first = 0;
        ok = ok && dd && T->host_up(dd, T->ptr(hb2), n * 4, &first);   /* b everywhere, then its halves again, 1500 times */
        for (int r = 0; ok && r < 1500; r++) ok = T->host_up(dd + (r & 1) * (n / 2), (const float *)T->ptr(hb2) + (r & 1) * (n / 2), n * 2, &t);
        ok = ok && T->begin() && T->submit(1);
        int landed = 0;
        auto t0 = std::chrono::steady_clock::now();   /* 1501 copies of 0.5 to 1 MiB: tens of ms over the bus */
        while (ok && !landed && std::chrono::steady_clock::now() - t0 < std::chrono::seconds(5)) landed = T->host_up_done(t);
        int early = T->host_up_done(first);
        if (!(ok && landed && early)) printf("  (ok %d landed %d early %d, tickets %u and %u)\n", ok, landed, early, first, t);
        ck(ok && landed && early, "host_up: 1501 copies on the copy stream, the last ticket landed, an early one too");
        ck(ok && maxdiff(down(db, n).data(), b.data(), n) == 0, "host_up: the bytes where they were sent");
        T->free(db); T->free(hb2);
    }
    if (T->size >= offsetof(ColiCudaChainOps, host_up_n) + sizeof T->host_up_n && T->host_up_n) {
        /* host_up_n: 96 pieces of a page-locked buffer, sizes 4 bytes to 11 KiB and apart, to places of a device
         * buffer in another order, one call and one ticket; after a zero the frame recorded first */
        const int np = 96; const size_t n = 1 << 18;
        std::vector<float> b = rnd(n, 43);
        CcBuf *db = T->buf(n * 4, CC_DEV), *hb2 = T->buf(n * 4, CC_UP);
        int ok = db && hb2;
        if (ok) memcpy(T->ptr(hb2), b.data(), n * 4);
        float *dd = ok ? (float *)T->dev_addr(db) : NULL;
        void *dst[np]; const void *src[np]; size_t by[np];
        std::vector<float> want(n, 0.f);
        for (int i = 0; i < np && ok; i++) {   /* piece i: floats [2730 i, +len) of the host buffer to the place of piece np-1-i */
            const size_t len = 1 + (size_t)(i * 97) % 2730, from = (size_t)i * 2730, to = (size_t)(np - 1 - i) * 2730;
            dst[i] = dd + to; src[i] = (const float *)T->ptr(hb2) + from; by[i] = len * 4;
            for (size_t j = 0; j < len; j++) want[to + j] = b[from + j];
        }
        unsigned t = 0;
        ok = ok && T->begin() && T->zero(db, 0, n) && T->host_up_n(dst, src, by, np, &t) && T->submit(1);
        int landed = 0;
        auto t0 = std::chrono::steady_clock::now();
        while (ok && !landed && std::chrono::steady_clock::now() - t0 < std::chrono::seconds(5)) landed = T->host_up_done(t);
        ck(ok && landed, "host_up_n: 96 copies in one call, their one ticket landed");
        ck(ok && landed && maxdiff(down(db, n).data(), want.data(), n) == 0, "host_up_n: each piece where it was sent, after the frame's zero");
        ck(!T->host_up_n(dst, src, by, 0, &t) && !T->host_up_n(NULL, src, by, 1, &t), "host_up_n: no copies, or none to, refused");
        T->free(db); T->free(hb2);
    }
    printf("matmul\n");
    {
        const int I = 96, O = 40, S = 3; unsigned s = 5;
        std::vector<int8_t> q((size_t)I * O); for (auto &v : q) v = (int8_t)(frand(&s) * 60);
        std::vector<float> sc(O); for (auto &v : sc) v = 0.02f + 0.01f * frand(&s);
        std::vector<float> x = rnd((size_t)S * I + 7, 9), ref((size_t)S * O);
        for (int r = 0; r < S; r++) for (int o = 0; o < O; o++) { float a = 0; for (int i = 0; i < I; i++) a += x[7 + r * I + i] * q[(size_t)o * I + i]; ref[r * O + o] = a * sc[o]; }
        ColiCudaTensor *t = NULL;
        ck(coli_cuda_tensor_upload(&t, q.data(), sc.data(), 1, I, O, g_dev), "an int8 tensor resident");
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
        {   /* few outputs of long int8 rows (Qwen3.8's hyper-connection read, 10240 x 320): the inputs split
             * (gemv_i8_sk), a verify's four rows each the bits of its one-row call, and the CPU's dot */
            const int I5 = 10240, O5 = 320, S5 = 4;
            std::vector<int8_t> q5((size_t)I5 * O5); for (auto &v : q5) v = (int8_t)(frand(&s) * 60);
            std::vector<float> sc5(O5), x5 = rnd((size_t)S5 * I5, 19), r5((size_t)S5 * O5);
            for (int o = 0; o < O5; o++) sc5[o] = 0.01f + 0.001f * (o % 7);
            for (int r = 0; r < S5; r++) for (int o = 0; o < O5; o++) {
                double a = 0; for (int i = 0; i < I5; i++) a += (double)x5[(size_t)r * I5 + i] * q5[(size_t)o * I5 + i]; r5[(size_t)r * O5 + o] = (float)(a * sc5[o]);
            }
            ColiCudaTensor *t5 = NULL;
            CcBuf *x5b = up(x5), *y5b = T->buf((size_t)S5 * O5 * 4, CC_DEV), *y1b = T->buf((size_t)O5 * 4, CC_DEV);
            int ok5 = coli_cuda_tensor_upload(&t5, q5.data(), sc5.data(), 1, I5, O5, g_dev) && T->begin() && T->matmul(t5, x5b, 0, y5b, 0, S5) && T->submit(1);
            std::vector<float> all5 = ok5 ? down(y5b, (size_t)S5 * O5) : std::vector<float>();
            double mx5 = 0; for (float v : r5) mx5 = fmax(mx5, fabs(v));
            ck(ok5 && maxdiff(all5.data(), r5.data(), r5.size()) < 1e-4 * mx5, "few outputs of long int8 rows, four rows: the CPU's dot");
            int same5 = ok5;
            for (int r = 0; r < S5 && same5; r++) {
                same5 = T->begin() && T->matmul(t5, x5b, (size_t)r * I5, y1b, 0, 1) && T->submit(1);
                if (same5) same5 = !memcmp(down(y1b, O5).data(), all5.data() + (size_t)r * O5, (size_t)O5 * 4);
            }
            ck(same5, "and each row the bits of its one-row call");
            if (t5) coli_cuda_tensor_free(t5);
            T->free(x5b); T->free(y5b); T->free(y1b);
        }
        {   /* a width the warp GEMV does not take (not a multiple of 16): the block kernel */
            const int I2 = 100; std::vector<int8_t> q2((size_t)I2 * O); for (auto &v : q2) v = (int8_t)(frand(&s) * 60);
            std::vector<float> x2 = rnd(I2, 13), ref2(O);
            for (int o = 0; o < O; o++) { float a = 0; for (int i = 0; i < I2; i++) a += x2[i] * q2[(size_t)o * I2 + i]; ref2[o] = a * sc[o]; }
            ColiCudaTensor *t2 = NULL; CcBuf *x2b = up(x2);
            ck(coli_cuda_tensor_upload(&t2, q2.data(), sc.data(), 1, I2, O, g_dev) && T->begin() && T->matmul(t2, x2b, 0, yb, 0, 1) && T->submit(1) &&
               maxdiff(down(yb, O).data(), ref2.data(), O) < 1e-3, "int8 rows of a width the warp GEMV leaves to the block kernel");
            coli_cuda_tensor_free(t2); T->free(x2b);
        }
        {   /* a prompt chunk's rows (more than the warp GEMV's 8): the tensor cores on sm_80+, TF32's 10-bit inputs */
            const int S2 = 70, I2 = 256, O2 = 100;
            std::vector<float> x2 = rnd((size_t)S2 * I2, 17), dq;
            for (int fmt : {1, 2, 4}) {
                ColiCudaTensor *tt = moe_mat(fmt, fmt == 4 ? 64 : 0, I2, O2, 700 + fmt, dq);
                const bool s8 = prompt_s8(I2, S2, fmt);
                const std::vector<float> xr = s8 ? s8_reference_rows(x2, I2) : x2;
                std::vector<float> r2((size_t)S2 * O2); double mx = 0;
                for (int r = 0; r < S2; r++) for (int o = 0; o < O2; o++) { double a = 0; for (int i = 0; i < I2; i++) a += (double)xr[(size_t)r * I2 + i] * dq[(size_t)o * I2 + i]; r2[(size_t)r * O2 + o] = (float)a; mx = fmax(mx, fabs(a)); }
                CcBuf *xb2 = up(x2), *yb2 = T->buf((size_t)S2 * O2 * 4, CC_DEV);
                int ok2 = tt && T->begin() && T->matmul(tt, xb2, 0, yb2, 0, S2) && T->submit(1);
                double e2 = ok2 ? maxdiff(down(yb2, (size_t)S2 * O2).data(), r2.data(), (size_t)S2 * O2) / mx : 1;
                if (e2 >= 2e-3) printf("  (prompt matmul fmt %d: relative error %.3g)\n", fmt, e2);
                if (s8) printf("  (int8 prompt reference uses rowwise s8 activation quantization)\n");
                ck(ok2 && e2 < 2e-3, fmt == 1 ? "seventy rows of int8 rows (a prompt chunk)" : fmt == 2 ? "seventy rows of int4 rows" : "seventy rows of int4 with a scale per 64");
                if (tt) coli_cuda_tensor_free(tt); T->free(xb2); T->free(yb2);
            }
        }
        std::vector<float> w = rnd((size_t)I * O, 11), ones(O, 1.f), ref0(O);
        for (int o = 0; o < O; o++) { float a = 0; for (int i = 0; i < I; i++) a += x[7 + i] * w[(size_t)o * I + i]; ref0[o] = a; }
        ColiCudaTensor *t0 = NULL;
        ck(coli_cuda_tensor_upload(&t0, w.data(), ones.data(), 0, I, O, g_dev), "an f32 tensor resident");
        ck(T->begin() && T->matmul(t0, xb, 7, yb, 0, 1) && T->submit(1) && maxdiff(down(yb, O).data(), ref0.data(), O) < 1e-4, "f32 rows: the CPU's dot");
        {   /* a prompt chunk's rows of an f32 matrix (the tiled GEMM): widths that are no tile's multiple */
            const int S3 = 70, I3 = 200, O3 = 100;
            std::vector<float> x3 = rnd((size_t)S3 * I3, 12), w3 = rnd((size_t)I3 * O3, 13), one3(O3, 1.f), r3((size_t)S3 * O3);
            for (int s = 0; s < S3; s++) for (int o = 0; o < O3; o++) {
                double a = 0; for (int i = 0; i < I3; i++) a += (double)x3[(size_t)s * I3 + i] * w3[(size_t)o * I3 + i]; r3[(size_t)s * O3 + o] = (float)a;
            }
            ColiCudaTensor *t3 = NULL;
            CcBuf *xb3 = T->buf((size_t)S3 * I3 * 4, CC_DEV), *yb3 = T->buf((size_t)S3 * O3 * 4, CC_DOWN);
            int ok3 = xb3 && yb3 && coli_cuda_tensor_upload(&t3, w3.data(), one3.data(), 0, I3, O3, g_dev) && T->begin() &&
                      T->write(xb3, 0, x3.data(), x3.size() * 4) && T->matmul(t3, xb3, 0, yb3, 0, S3) && T->submit(1);
            ck(ok3 && maxdiff((const float *)T->ptr(yb3), r3.data(), r3.size()) < 1e-4, "f32 rows, a prompt chunk's 70 (the tiled GEMM)");
            coli_cuda_tensor_free(t3); T->free(xb3); T->free(yb3);
        }
        {   /* few outputs of long f32 rows, a decode step's and a verify's rows (the wide GEMV): each row its one-row bits */
            const int S4 = 3, I4 = 10240, O4 = 4;
            std::vector<float> x4 = rnd((size_t)S4 * I4, 14), w4 = rnd((size_t)I4 * O4, 15), one4(O4, 1.f), r4((size_t)S4 * O4);
            for (int s = 0; s < S4; s++) for (int o = 0; o < O4; o++) {
                double a = 0; for (int i = 0; i < I4; i++) a += (double)x4[(size_t)s * I4 + i] * w4[(size_t)o * I4 + i]; r4[(size_t)s * O4 + o] = (float)a;
            }
            ColiCudaTensor *t4 = NULL;
            CcBuf *xb4 = T->buf((size_t)S4 * I4 * 4, CC_DEV), *yb4 = T->buf((size_t)S4 * O4 * 4, CC_DOWN), *yb1 = T->buf((size_t)O4 * 4, CC_DOWN);
            int ok4 = xb4 && yb4 && yb1 && coli_cuda_tensor_upload(&t4, w4.data(), one4.data(), 0, I4, O4, g_dev) && T->begin() &&
                      T->write(xb4, 0, x4.data(), x4.size() * 4) && T->matmul(t4, xb4, 0, yb4, 0, S4) && T->submit(1);
            ck(ok4 && maxdiff((const float *)T->ptr(yb4), r4.data(), r4.size()) < 1e-3, "f32 rows, 4 outputs of 10240 (the GEMV for few outputs)");
            int same = ok4;
            for (int s = 0; s < S4 && same; s++)
                same = T->begin() && T->matmul(t4, xb4, (size_t)s * I4, yb1, 0, 1) && T->submit(1) &&
                       !memcmp(T->ptr(yb1), (const float *)T->ptr(yb4) + (size_t)s * O4, (size_t)O4 * 4);
            ck(same, "and each of its three rows the one-row call's bits");
            {   /* a prompt chunk's 100 rows of it (f32_few_k, a block a row): each row the one-row call's bits */
                const int S8 = 100;
                std::vector<float> x8 = rnd((size_t)S8 * I4, 16);
                CcBuf *xb8 = T->buf((size_t)S8 * I4 * 4, CC_DEV), *yb8 = T->buf((size_t)S8 * O4 * 4, CC_DOWN);
                int ok8 = xb8 && yb8 && T->begin() && T->write(xb8, 0, x8.data(), x8.size() * 4) && T->matmul(t4, xb8, 0, yb8, 0, S8) && T->submit(1);
                for (int r = 0; r < S8 && ok8; r += 11)   /* ten of them */
                    ok8 = T->begin() && T->matmul(t4, xb8, (size_t)r * I4, yb1, 0, 1) && T->submit(1) &&
                          !memcmp(T->ptr(yb1), (const float *)T->ptr(yb8) + (size_t)r * O4, (size_t)O4 * 4);
                ck(ok8, "f32 rows, 4 outputs of 10240, a prompt chunk's 100 rows: each the one-row call's bits");   /* 11 matmuls */
                T->free(xb8); T->free(yb8);
            }
            coli_cuda_tensor_free(t4); T->free(xb4); T->free(yb4); T->free(yb1);
        }
        {   /* BF16 rows (a GGUF's type 30, fmt 130): a decode step's and a verify's rows (bd_gemv_bf16_k, each row
             * the one-row call's bits), and a prompt chunk's (the f16 tiles), against the CPU's dot of the same weights */
            const int I6 = 2560, O6 = 1000, S6 = 5, S7 = 70;
            std::vector<float> x6 = rnd((size_t)S7 * I6, 23), wf = rnd((size_t)I6 * O6, 24);
            std::vector<uint16_t> wb16((size_t)I6 * O6);
            for (size_t k = 0; k < wb16.size(); k++) { uint32_t u; memcpy(&u, &wf[k], 4); wb16[k] = (uint16_t)(u >> 16); u = (uint32_t)wb16[k] << 16; memcpy(&wf[k], &u, 4); }
            std::vector<float> r6((size_t)S7 * O6);
            for (int s = 0; s < S7; s++) for (int o = 0; o < O6; o++) {
                double a = 0; for (int i = 0; i < I6; i++) a += (double)x6[(size_t)s * I6 + i] * wf[(size_t)o * I6 + i]; r6[(size_t)s * O6 + o] = (float)a;
            }
            double mx = 0; for (float v : r6) mx = fmax(mx, fabs(v));
            ColiCudaTensor *t6 = NULL;
            CcBuf *xb6 = T->buf((size_t)S7 * I6 * 4, CC_DEV), *yb6 = T->buf((size_t)S7 * O6 * 4, CC_DOWN), *yb7 = T->buf((size_t)O6 * 4, CC_DOWN);
            int ok6 = xb6 && yb6 && yb7 && coli_cuda_tensor_upload(&t6, wb16.data(), NULL, 130, I6, O6, g_dev) && T->begin() &&
                      T->write(xb6, 0, x6.data(), x6.size() * 4) && T->matmul(t6, xb6, 0, yb6, 0, S6) && T->submit(1);
            ck(ok6 && maxdiff((const float *)T->ptr(yb6), r6.data(), (size_t)S6 * O6) < 1e-3, "BF16 rows, a verify's 5 rows: the CPU's dot");
            int same = ok6;
            std::vector<float> five((const float *)T->ptr(yb6), (const float *)T->ptr(yb6) + (size_t)S6 * O6);
            for (int s = 0; s < S6 && same; s++)
                same = T->begin() && T->matmul(t6, xb6, (size_t)s * I6, yb7, 0, 1) && T->submit(1) &&
                       !memcmp(T->ptr(yb7), five.data() + (size_t)s * O6, (size_t)O6 * 4);
            ck(same, "and each of its rows the one-row call's bits");
            int ok7 = ok6 && T->begin() && T->matmul(t6, xb6, 0, yb6, 0, S7) && T->submit(1);
            double e7 = ok7 ? maxdiff((const float *)T->ptr(yb6), r6.data(), r6.size()) / mx : 1;
            if (e7 >= 2e-3) printf("  (prompt matmul BF16: relative error %.3g)\n", e7);
            ck(ok7 && e7 < 2e-3, "BF16 rows, a prompt chunk's 70 (the f16 tiles)");
            if (t6) coli_cuda_tensor_free(t6);
            T->free(xb6); T->free(yb6); T->free(yb7);
        }
        for (int shape = 0; shape < 3; shape++) {   /* and quant_matmul's bits (coli_cuda_matmul still runs it) */
            static const int sI[3] = {10240, 2048, 2048}, sO[3] = {4, 256, 1};
            const int I5 = sI[shape], O5 = sO[shape], S5 = 2;
            std::vector<float> x5 = rnd((size_t)S5 * I5, 16 + shape), w5 = rnd((size_t)I5 * O5, 19 + shape), one5(O5, 1.f), q5((size_t)S5 * O5);
            ColiCudaTensor *t5 = NULL, *tq = NULL;
            CcBuf *xb5 = T->buf((size_t)S5 * I5 * 4, CC_DEV), *yb5 = T->buf((size_t)S5 * O5 * 4, CC_DOWN);
            int ok5 = xb5 && yb5 && coli_cuda_tensor_upload(&t5, w5.data(), one5.data(), 0, I5, O5, g_dev) && T->begin() &&
                      T->write(xb5, 0, x5.data(), x5.size() * 4) && T->matmul(t5, xb5, 0, yb5, 0, S5) && T->submit(1) &&
                      coli_cuda_matmul(&tq, q5.data(), x5.data(), w5.data(), one5.data(), 0, S5, I5, O5, g_dev, 0);
            char what[96]; snprintf(what, sizeof what, "f32 rows, %d outputs of %d: quant_matmul's bits", O5, I5);
            ck(ok5 && !memcmp(T->ptr(yb5), q5.data(), q5.size() * 4), what);
            if (ok5 && memcmp(T->ptr(yb5), q5.data(), q5.size() * 4))
                printf("  (largest difference %.3g)\n", maxdiff((const float *)T->ptr(yb5), q5.data(), q5.size()));
            coli_cuda_tensor_free(t5); coli_cuda_tensor_free(tq); T->free(xb5); T->free(yb5);
        }
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
    printf("addnorm\n");
    /* the branch (normalised first or not) into x and x's norm in one pass: the same bits as the
     * norm, the add and the norm apart, on this device */
    for (int it = 0; it < 6; it++) {
        const int pre = it & 1, D = it < 2 ? 64 : it < 4 ? 2048 : 2560, rows = 5;
        std::vector<float> x = rnd((size_t)rows * D, 81, 2.f), t = rnd((size_t)rows * D, 82), w = rnd(2 * (size_t)D, 83, 0.3f);
        CcBuf *xa = up(x), *xb = up(x), *ta = up(t), *tb = up(t), *wb = up(w), *ya = up(x), *yb = up(x);
        CcNorm np1 = {rows, D, 1, 0, D, D, 0, D, D, 0, 0, CC_NORM_ADD1, 1e-6f, 1.f}, np2 = {rows, D, 1, 0, D, D, 0, D, D, D, 0, CC_NORM_ADD1, 1e-6f, 1.f};
        CcEw add = {CC_EW_ADD, rows * D, D, 1, 0, 1, 0, 0, 0, 0, 0, 1.f};
        CcAddNorm an = {rows, D, 0, 0, 0, 0, D, pre, 1e-6f};
        int ok = T->begin() && (!pre || T->norm(ta, wb, ta, &np1)) && T->ew(xa, xa, ta, NULL, NULL, &add) && T->norm(xa, wb, ya, &np2) &&
                 T->addnorm(xb, tb, wb, yb, &an) == 1 && T->submit(1);
        char what[96];
        snprintf(what, sizeof what, "D %d, %s: x and its norm bit for bit", D, pre ? "the branch normalised" : "the branch as it is");
        std::vector<float> xa_v = down(xa, x.size()), xb_v = down(xb, x.size()), ya_v = down(ya, x.size()), yb_v = down(yb, x.size());
        const int same_x = ok && !bits_differ("x (the add)", xa_v.data(), xb_v.data(), x.size(), D),
                  same_y = ok && !bits_differ("its norm", ya_v.data(), yb_v.data(), x.size(), D);
        ck(ok && same_x && same_y, what);
        T->free(xa); T->free(xb); T->free(ta); T->free(tb); T->free(wb); T->free(ya); T->free(yb);
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
    printf("float16 KV rows\n");
    {
        /* ew's CC_EW_F16 against kv16.h, every bit: ordinary values, past +-65504 (clamped), and
         * small ones down through float16's subnormals to zero */
        const int n = 4096;
        std::vector<float> x = rnd(n, 51, 70000.f);
        for (int i = 0; i < n / 4; i++) x[i] *= 1e-4f;
        for (int i = n / 4; i < n / 2; i++) x[i] *= 1e-9f;
        CcEw r = {CC_EW_F16, n, n, 0, 0, 0, 0, 0, 0, 0, 0, 1.f};
        CcBuf *xb = up(x), *yb = up(std::vector<float>(n));
        int ok = T->begin() && T->ew(yb, xb, NULL, NULL, NULL, &r) && T->submit(1), bad = 0;
        std::vector<float> y = down(yb, n);
        for (int i = 0; i < n && ok; i++) {
            float c = kv16_clamp(x[i]); uint32_t cu, want, got; memcpy(&cu, &c, 4); want = kv16_round_bits(cu); memcpy(&got, &y[i], 4);
            bad += got != want;
        }
        ck(ok && !bad, "ew's float16 rounding: the bits of kv16.h's");
        T->free(xb); T->free(yb);
        /* rope_f16: the rope and both rounds in one launch, the bits of the three launches; a
         * partial rotation (16 of 40 values a head), as Qwen3.6's 64 of 256 */
        if (T->size >= offsetof(ColiCudaChainOps, rope_f16) + sizeof T->rope_f16 && T->rope_f16) {
            const int rows = 3, KV = 2, kvd = 40, half = 8;
            std::vector<float> k = rnd((size_t)rows * KV * kvd, 61, 3.f), v = rnd((size_t)rows * KV * kvd, 62, 3.f), cs((size_t)rows * 2 * half);
            for (int q = 0; q < rows; q++) for (int j = 0; j < half; j++) { float ang = (q + 7) * powf(10000.f, -2.f * j / (2 * half)); cs[(q * half + j) * 2] = cosf(ang); cs[(q * half + j) * 2 + 1] = sinf(ang); }
            CcRope p = {rows * KV, KV, 0, KV * kvd, kvd, half, 0, 2 * half};
            CcEw rr = {CC_EW_F16, rows * KV * kvd, KV * kvd, 0, 0, 0, 0, 0, 0, 0, 0, 1.f};
            CcBuf *k3 = up(k), *v3 = up(v), *k1 = up(k), *v1 = up(v), *cb = up(cs, CC_UP);
            int three = T->begin() && T->rope(k3, cb, &p) && T->ew(k3, k3, NULL, NULL, NULL, &rr) && T->ew(v3, v3, NULL, NULL, NULL, &rr) && T->submit(1);
            int one = T->begin() && T->rope_f16(k1, v1, cb, &p, kvd) && T->submit(1);
            std::vector<float> ka = down(k3, k.size()), kb = down(k1, k.size()), va = down(v3, v.size()), vb = down(v1, v.size());
            ck(three && one && !memcmp(ka.data(), kb.data(), k.size() * 4) && !memcmp(va.data(), vb.data(), v.size() * 4),
               "rope_f16: K rotated and rounded, V rounded, the bits of rope and two ew");
            ck(!T->rope_f16(k1, v1, cb, &p, 2 * half - 1), "rope_f16: a segment narrower than the rotation refused");
            T->free(k3); T->free(v3); T->free(k1); T->free(v1); T->free(cb);
        } else ck(0, "the backend has rope_f16");
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
    {   /* a prompt chunk's rows with selection lists, a block per kv head (cc_attn_gk, from 32 rows): each
         * row the bits of that row alone (cc_attn_k), Qwen3.8's QSA shape (24 / 2 heads of 256, the gate) */
        const int H = 24, KV = 2, hd = 256, qdim = 2 * hd, cap = 1200, pos_base = 700, S = 40, NS = 400, SR = NS + 1;
        std::vector<float> q = rnd((size_t)S * H * qdim, 61), kc = rnd((size_t)KV * cap * hd, 62), vc = rnd((size_t)KV * cap * hd, 63);
        std::vector<int> sel((size_t)S * SR);
        for (int r = 0; r < S; r++) {
            const int vis = pos_base + r + 1, c = r == 3 ? -1 : r % 5 == 0 ? 97 : NS;
            sel[(size_t)r * SR] = c;
            for (int i = 0; i < NS; i++) sel[(size_t)r * SR + 1 + i] = (int)(((long long)r * 131 + (long long)i * 37) % vis);
        }
        CcAttn p = {S, H, KV, hd, pos_base, cap, 0, H * qdim, qdim, hd, H * qdim, qdim, 1, 0, H * hd, 0, SR, 1.f / sqrtf((float)hd), 0, 0};
        std::vector<float> o((size_t)S * H * hd), ref = o;
        ref_attn(q.data(), kc.data(), vc.data(), ref.data(), q.data(), sel.data(), &p);
        CcBuf *qb = up(q), *kb = up(kc), *vb = up(vc), *ob = T->buf(o.size() * 4, CC_DEV), *sb = T->buf(sel.size() * 4, CC_DEV);
        setenv("COLI_CUDA_ATTN_SEL_TC", "0", 1);   /* cc_attn_gk: these lists are in no order, with repeats */
        int ok = T->begin() && T->write(sb, 0, sel.data(), sel.size() * 4) && T->attn(qb, kb, vb, ob, qb, sb, &p) && T->submit(1);
        std::vector<float> all = down(ob, o.size());
        ck(ok && maxdiff(all.data(), ref.data(), o.size()) < 1e-4, "40 rows with selection lists, a block per kv head, against the reference");
        CcBuf *o1 = T->buf((size_t)H * hd * 4, CC_DEV);
        int same = 1;
        setenv("COLI_CUDA_ATTN_DEC_SEL", "0", 1);   /* a row alone through cc_attn_k */
        for (int r = 0; r < S && same; r++) {
            CcAttn p1 = p; p1.S = 1; p1.pos_base = pos_base + r; p1.q_off = r * p.q_row; p1.g_off = p.g_off + r * p.g_row; p1.o_off = 0; p1.sel_off = r * SR;
            std::vector<float> one;
            same = T->begin() && T->attn(qb, kb, vb, o1, qb, sb, &p1) && T->submit(1);
            if (same) { one = down(o1, (size_t)H * hd); same = !memcmp(one.data(), all.data() + (size_t)r * H * hd, (size_t)H * hd * 4); }
        }
        unsetenv("COLI_CUDA_ATTN_DEC_SEL");
        ck(same, "each row the bits of that row alone (a block per query head, COLI_CUDA_ATTN_DEC_SEL=0)");
        /* a verify's rows with their lists in blocks of positions (cc_attn_dec_k): 1 to 4 rows from row 3 on
         * (row 3 without a list: its causal positions; row 5's list 97 long, the others 400), against the reference */
        double worst = 0;
        int okd = 1;
        for (int S2 = 1; S2 <= 4 && okd; S2++) {
            CcAttn p2 = p; p2.S = S2; p2.pos_base = pos_base + 3; p2.q_off = 3 * p.q_row; p2.g_off = p.g_off + 3 * p.g_row; p2.o_off = 0;
            p2.sel_off = 3 * SR;
            CcBuf *o2 = T->buf((size_t)S2 * H * hd * 4, CC_DEV);
            okd = o2 && T->begin() && T->attn(qb, kb, vb, o2, qb, sb, &p2) && T->submit(1);
            std::vector<float> got = okd ? down(o2, (size_t)S2 * H * hd) : std::vector<float>();
            if (okd) { double d = maxdiff(got.data(), ref.data() + (size_t)3 * H * hd, (size_t)S2 * H * hd); if (d > worst) worst = d; }
            T->free(o2);
        }
        ck(okd && worst < 1e-4, "a verify's rows with selection lists in blocks of positions, against the reference");
        unsetenv("COLI_CUDA_ATTN_SEL_TC");
        T->free(qb); T->free(kb); T->free(vb); T->free(ob); T->free(sb); T->free(o1);
    }
    {   /* a prompt chunk's rows with selection lists on the tensor cores (cc_attn_f2 with them, three warps
         * for 12 query heads a kv head): lists in ascending order as cc_qsa writes them, blocks of four
         * positions and the row's own last ones, a row without a list, short lists, key blocks no row of a
         * block selects, 41 rows (a block of tokens cut short); against the reference, f16's error */
        const int H = 24, KV = 2, hd = 256, qdim = 2 * hd, cap = 1600, pos_base = 1100, S = 41, NS = 520, SR = NS + 1;
        std::vector<float> q = rnd((size_t)S * H * qdim, 71), kc = rnd((size_t)KV * cap * hd, 72), vc = rnd((size_t)KV * cap * hd, 73);
        std::vector<int> sel((size_t)S * SR, 0);
        for (int r = 0; r < S; r++) {
            const int vis = pos_base + r + 1;
            if (r == 5) { sel[(size_t)r * SR] = -1; continue; }   /* no list: every position up to the row's */
            std::vector<int> pos;
            const int want = r % 7 == 0 ? 37 : NS - 8;
            for (int b = 0; b * 4 < vis - 4 && (int)pos.size() + 4 <= want; b++)   /* blocks of 4, two in five, more near the end */
                if ((b * 2654435761u + (unsigned)r * 40503u) % 5 < 2 || b * 4 > vis - 200)
                    for (int i = 0; i < 4; i++) pos.push_back(b * 4 + i);
            for (int t = (vis - 4) / 4 * 4; t < vis && (int)pos.size() < want; t++) if (pos.empty() || t > pos.back()) pos.push_back(t);
            sel[(size_t)r * SR] = (int)pos.size();
            for (size_t i = 0; i < pos.size(); i++) sel[(size_t)r * SR + 1 + i] = pos[i];
        }
        CcAttn p = {S, H, KV, hd, pos_base, cap, 0, H * qdim, qdim, hd, H * qdim, qdim, 1, 0, H * hd, 0, SR, 1.f / sqrtf((float)hd), 0, 0};
        std::vector<float> ref((size_t)S * H * hd);
        ref_attn(q.data(), kc.data(), vc.data(), ref.data(), q.data(), sel.data(), &p);
        CcBuf *qb = up(q), *kb = up(kc), *vb = up(vc), *ob = T->buf(ref.size() * 4, CC_DEV), *sb = T->buf(sel.size() * 4, CC_DEV);
        int ok = qb && kb && vb && ob && sb && T->begin() && T->write(sb, 0, sel.data(), sel.size() * 4) && T->attn(qb, kb, vb, ob, qb, sb, &p) && T->submit(1);
        const double d = ok ? maxdiff(down(ob, ref.size()).data(), ref.data(), ref.size()) : 1;
        char what[160]; snprintf(what, sizeof what, "41 rows with ascending selection lists on the tensor cores, against the reference (|d| %.1e)", d);
        ck(ok && d < 4e-3, what);
        T->free(qb); T->free(kb); T->free(vb); T->free(ob); T->free(sb);
    }
    {   /* a decode's or a verify's rows over many positions: in blocks of positions (cc_attn_dec_k) */
        struct { int H, KV, hd, S, pos_base, cap, gate; } cs[] = {{16, 2, 256, 3, 2500, 3000, 1}, {4, 2, 32, 1, 140, 300, 1},
                                                                   {8, 8, 128, 4, 700, 800, 0}, {16, 2, 256, 6, 700, 800, 1},
                                                                   {16, 2, 128, 6, 125, 300, 1}, {16, 2, 128, 6, 4093, 4200, 0}};
        for (auto &k : cs) {
            const int qdim = 2 * k.hd, k_off = 32;
            std::vector<float> q = rnd((size_t)k.S * k.H * qdim, 51), kc = rnd((size_t)k_off + (size_t)k.KV * k.cap * k.hd, 52),
                               vc = rnd((size_t)k_off + (size_t)k.KV * k.cap * k.hd, 53);
            std::vector<float> o((size_t)k.S * k.H * k.hd), ref = o;
            CcAttn p = {k.S, k.H, k.KV, k.hd, k.pos_base, k.cap, 0, k.H * qdim, qdim, k.hd, k.H * qdim, qdim, k.gate, 0, k.H * k.hd, 0, 0,
                        1.f / sqrtf((float)k.hd), k_off, k_off};
            ref_attn(q.data(), kc.data(), vc.data(), ref.data(), k.gate ? q.data() : NULL, NULL, &p);
            CcBuf *qb = up(q), *kb = up(kc), *vb = up(vc), *ob = T->buf(o.size() * 4, CC_DEV);
            char what[96];
            snprintf(what, sizeof what, "H %d/%d hd %d: %d rows after %d positions, in blocks", k.H, k.KV, k.hd, k.S, k.pos_base);
            ck(T->begin() && T->attn(qb, kb, vb, ob, k.gate ? qb : NULL, NULL, &p) && T->submit(1) &&
               maxdiff(down(ob, o.size()).data(), ref.data(), o.size()) < 1e-4, what);
            /* a verify's rows: each the bits of a decode step of that row alone (the blocks of positions
             * its own count chooses, past 128 positions and across the doubling of the block at 4096) */
            if (k.S > 1) {
                std::vector<float> all = down(ob, o.size());
                CcBuf *o1 = T->buf((size_t)k.H * k.hd * 4, CC_DEV);
                int same = 1;
                for (int r = 0; r < k.S && same; r++) {
                    CcAttn p1 = p; p1.S = 1; p1.pos_base = k.pos_base + r; p1.q_off = p.q_off + r * p.q_row; p1.g_off = p.g_off + r * p.g_row; p1.o_off = 0;
                    same = T->begin() && T->attn(qb, kb, vb, o1, k.gate ? qb : NULL, NULL, &p1) && T->submit(1) &&
                           !memcmp(down(o1, (size_t)k.H * k.hd).data(), all.data() + (size_t)r * k.H * k.hd, (size_t)k.H * k.hd * 4);
                }
                snprintf(what, sizeof what, "and each of the %d rows a one-row step's bits", k.S);
                ck(same, what);
                T->free(o1);
            }
            T->free(qb); T->free(kb); T->free(vb); T->free(ob);
        }
    }
    {   /* a prompt chunk's rows (COLI_CUDA_CHAIN_FLASH, 16 rows): the tensor cores where the device has them
         * (f16 Q, K and V: within 4e-3 of the reference), cc_attn_k elsewhere */
        struct { int H, KV, hd, S, pos_base, gate; } cs[] = {{4, 2, 64, 40, 0, 1}, {16, 2, 256, 70, 33, 1}, {8, 8, 128, 17, 200, 0},
                                                              {16, 4, 128, 64, 0, 0}};
        for (auto &k : cs) {
            const int qdim = 2 * k.hd, cap = k.pos_base + k.S + 9, k_off = 32;
            std::vector<float> q = rnd((size_t)k.S * k.H * qdim, 51), kc = rnd((size_t)k_off + (size_t)k.KV * cap * k.hd, 52),
                               vc = rnd((size_t)k_off + (size_t)k.KV * cap * k.hd, 53);
            std::vector<float> o((size_t)k.S * k.H * k.hd), ref = o;
            CcAttn p = {k.S, k.H, k.KV, k.hd, k.pos_base, cap, 0, k.H * qdim, qdim, k.hd, k.H * qdim, qdim, k.gate, 0, k.H * k.hd, 0, 0,
                        1.f / sqrtf((float)k.hd), k_off, k_off};
            ref_attn(q.data(), kc.data(), vc.data(), ref.data(), q.data(), NULL, &p);
            CcBuf *qb = up(q), *kb = up(kc), *vb = up(vc), *ob = T->buf(o.size() * 4, CC_DEV);
            char what[128];
            snprintf(what, sizeof what, "prompt rows: H %d KV %d hd %d, %d rows after %d%s", k.H, k.KV, k.hd, k.S, k.pos_base, k.gate ? ", the gate" : "");
            float d = 1e30f;
            if (T->begin() && T->attn(qb, kb, vb, ob, qb, NULL, &p) && T->submit(1)) d = maxdiff(down(ob, o.size()).data(), ref.data(), o.size());
            ck(d < 4e-3f, what);
            printf("  %s: max |diff| %.2e\n", what, d);
            T->free(qb); T->free(kb); T->free(vb); T->free(ob);
        }
    }
    {   /* a prompt chunk with a sliding window and a sink per head (Flint), its cache linear or a
         * ring: the tensor cores where the device has them (within 4e-3 of the reference) */
        struct { int H, KV, hd, S, pos_base, win, ring; } cs[] = {{32, 4, 128, 300, 700, 129, 0}, {32, 4, 128, 90, 900, 129, 230},
                                                                  {16, 2, 256, 70, 300, 100, 0}, {8, 8, 64, 33, 0, 20, 0},
                                                                  {32, 4, 128, 200, 0, 513, 0}};
        for (auto &k : cs) {
            const int cap = k.ring ? k.ring : k.pos_base + k.S + 9, k_off = 32, so = 3;
            std::vector<float> q = rnd((size_t)k.S * k.H * k.hd, 71), kc = rnd((size_t)k_off + (size_t)k.KV * cap * k.hd, 72),
                               vc = rnd((size_t)k_off + (size_t)k.KV * cap * k.hd, 73), sk = rnd((size_t)k.H + so, 74, 2.f);
            std::vector<float> o((size_t)k.S * k.H * k.hd), ref = o;
            CcAttn a = {k.S, k.H, k.KV, k.hd, k.pos_base, cap, 0, k.H * k.hd, k.hd, 0, 0, 0, 0, 0, k.H * k.hd, 0, 0,
                        1.f / sqrtf((float)k.hd), k_off, k_off};
            CcAttnW w = {a, k.win, 1, so};
            ref_attn_w(q.data(), kc.data(), vc.data(), ref.data(), &a, k.win, sk.data(), so, k.ring);
            CcBuf *qb = up(q), *kb = up(kc), *vb = up(vc), *sb = up(sk), *ob = T->buf(o.size() * 4, CC_DEV);
            char what[160];
            snprintf(what, sizeof what, "prompt rows, window %d and sinks: H %d KV %d hd %d, %d rows after %d%s", k.win, k.H, k.KV, k.hd,
                     k.S, k.pos_base, k.ring ? ", a ring" : "");
            float d = 1e30f;
            int ok = T->begin() && (k.ring ? T->size >= offsetof(ColiCudaChainOps, attn_wr) + sizeof T->attn_wr && T->attn_wr(qb, kb, vb, ob, NULL, sb, &w, k.ring)
                                           : T->attn_w(qb, kb, vb, ob, NULL, sb, &w)) && T->submit(1);
            if (ok) d = maxdiff(down(ob, o.size()).data(), ref.data(), o.size());
            ck(d < 4e-3f, what);
            printf("  %s: max |diff| %.2e\n", what, d);
            T->free(qb); T->free(kb); T->free(vb); T->free(sb); T->free(ob);
        }
    }
    {   /* a decode's or a verify's rows with a window and a sink per head (Flint) after many positions:
         * in blocks of positions, a KV head's query heads a block, float32 (within 1e-4); a verify's
         * rows each a one-row step's bits */
        struct { int H, KV, hd, S, pos_base, win, ring; } cs[] = {{32, 4, 128, 1, 17000, 0, 0}, {32, 4, 128, 3, 5000, 128, 0},
                                                                  {32, 4, 128, 1, 4000, 200, 236}, {16, 2, 256, 4, 2000, 0, 0}};
        for (auto &k : cs) {
            const int cap = k.ring ? k.ring : k.pos_base + k.S + 9, k_off = 32, so = 3;
            std::vector<float> q = rnd((size_t)k.S * k.H * k.hd, 81), kc = rnd((size_t)k_off + (size_t)k.KV * cap * k.hd, 82),
                               vc = rnd((size_t)k_off + (size_t)k.KV * cap * k.hd, 83), sk = rnd((size_t)k.H + so, 84, 2.f);
            std::vector<float> o((size_t)k.S * k.H * k.hd), ref = o;
            CcAttn a = {k.S, k.H, k.KV, k.hd, k.pos_base, cap, 0, k.H * k.hd, k.hd, 0, 0, 0, 0, 0, k.H * k.hd, 0, 0,
                        1.f / sqrtf((float)k.hd), k_off, k_off};
            CcAttnW w = {a, k.win, 1, so};
            ref_attn_w(q.data(), kc.data(), vc.data(), ref.data(), &a, k.win, sk.data(), so, k.ring);
            CcBuf *qb = up(q), *kb = up(kc), *vb = up(vc), *sb = up(sk), *ob = T->buf(o.size() * 4, CC_DEV);
            const int wr = k.ring && T->size >= offsetof(ColiCudaChainOps, attn_wr) + sizeof T->attn_wr;
            auto run = [&](CcBuf *out, const CcAttnW *x) {
                return T->begin() && (k.ring ? wr && T->attn_wr(qb, kb, vb, out, NULL, sb, x, k.ring) : T->attn_w(qb, kb, vb, out, NULL, sb, x)) &&
                       T->submit(1);
            };
            char what[160];
            snprintf(what, sizeof what, "decode rows, window %d and sinks: H %d KV %d hd %d, %d rows after %d%s", k.win, k.H, k.KV, k.hd,
                     k.S, k.pos_base, k.ring ? ", a ring" : "");
            float d = 1e30f;
            if (run(ob, &w)) d = maxdiff(down(ob, o.size()).data(), ref.data(), o.size());
            ck(d < 1e-4f, what);
            printf("  %s: max |diff| %.2e\n", what, d);
            if (k.S > 1) {
                std::vector<float> all = down(ob, o.size());
                CcBuf *o1 = T->buf((size_t)k.H * k.hd * 4, CC_DEV);
                int same = 1;
                for (int r = 0; r < k.S && same; r++) {
                    CcAttnW w1 = w; w1.a.S = 1; w1.a.pos_base = k.pos_base + r; w1.a.q_off = r * a.q_row; w1.a.o_off = 0;
                    same = run(o1, &w1) && !memcmp(down(o1, (size_t)k.H * k.hd).data(), all.data() + (size_t)r * k.H * k.hd, (size_t)k.H * k.hd * 4);
                }
                ck(same, "and each row a one-row step's bits");
                T->free(o1);
            }
            T->free(qb); T->free(kb); T->free(vb); T->free(sb); T->free(ob);
        }
    }
    prompt_rows_one_pass_long();
    {   /* a float16 KV cache (kvh: attn_h, attn_wh, copy_regions_h): K and V of float16 values give
         * attn's bits from a float32 cache of the same values on every path: a decode's blocks by KV
         * head and by head, a prompt chunk on the tensor cores, a window with sinks, linear and in a ring */
        if (T->size < offsetof(ColiCudaChainOps, copy_regions_h) + sizeof T->copy_regions_h || !T->kvh()) printf("  (no float16 cache here)\n");
        else {
            struct { int H, KV, hd, S, pos_base, win, ring, sink; } cs[] = {
                {16, 2, 256, 3, 2500, 0, 0, 0}, {8, 8, 128, 4, 700, 0, 0, 0}, {16, 2, 256, 70, 33, 0, 0, 0}, {4, 2, 64, 40, 0, 0, 0, 0},
                {32, 4, 128, 1, 4000, 200, 236, 1}, {32, 4, 128, 90, 900, 129, 230, 1}, {32, 4, 128, 3, 5000, 128, 0, 1}};
            for (auto &k : cs) {
                const int cap = k.ring ? k.ring : k.pos_base + k.S + 9, k_off = 32, so = 3;
                std::vector<float> q = rnd((size_t)k.S * k.H * k.hd, 91), kc = rnd((size_t)k_off + (size_t)k.KV * cap * k.hd, 92),
                                   vc = rnd(kc.size(), 93), sk = rnd((size_t)k.H + so, 94, 2.f), o((size_t)k.S * k.H * k.hd);
                kv16_round(kc.data(), (int64_t)kc.size()); kv16_round(vc.data(), (int64_t)vc.size());
                std::vector<uint16_t> kh(kc.size()), vh(vc.size());
                kv16_pack(kh.data(), kc.data(), (int64_t)kc.size()); kv16_pack(vh.data(), vc.data(), (int64_t)vc.size());
                CcBuf *qb = up(q), *kb = up(kc), *vb = up(vc), *sb = up(sk), *o32 = T->buf(o.size() * 4, CC_DEV), *o16 = T->buf(o.size() * 4, CC_DEV);
                CcBuf *khb = T->buf(kh.size() * 2, CC_DEV), *vhb = T->buf(vh.size() * 2, CC_DEV);
                int ok = khb && vhb && T->begin() && T->write(khb, 0, kh.data(), kh.size() * 2) && T->write(vhb, 0, vh.data(), vh.size() * 2) &&
                         T->submit(1);
                CcAttn a = {k.S, k.H, k.KV, k.hd, k.pos_base, cap, 0, k.H * k.hd, k.hd, 0, 0, 0, 0, 0, k.H * k.hd, 0, 0,
                            1.f / sqrtf((float)k.hd), k_off, k_off};
                CcAttnW w = {a, k.win, k.sink, so};
                if (k.win || k.sink)
                    ok = ok && T->begin() && (k.ring ? T->attn_wr(qb, kb, vb, o32, NULL, sb, &w, k.ring) : T->attn_w(qb, kb, vb, o32, NULL, sb, &w)) &&
                         T->attn_wh(qb, khb, vhb, o16, NULL, sb, &w, k.ring) && T->submit(1);
                else ok = ok && T->begin() && T->attn(qb, kb, vb, o32, NULL, NULL, &a) && T->attn_h(qb, khb, vhb, o16, NULL, NULL, &a) && T->submit(1);
                char what[160];
                snprintf(what, sizeof what, "a float16 cache: H %d KV %d hd %d, %d rows after %d%s%s: a float32 cache's bits", k.H, k.KV, k.hd,
                         k.S, k.pos_base, k.win ? ", a window and sinks" : "", k.ring ? " in a ring" : "");
                ck(ok && !memcmp(down(o32, o.size()).data(), down(o16, o.size()).data(), o.size() * 4), what);
                T->free(qb); T->free(kb); T->free(vb); T->free(sb); T->free(o32); T->free(o16); T->free(khb); T->free(vhb);
            }
            /* copy_regions_h: a step's K rows into a head-major cache (a grid), and one region alone */
            const int n = 5, KV = 2, kvd = 64, rows = 20, pb = 7;
            std::vector<float> src = rnd((size_t)n * KV * kvd, 95);
            std::vector<CcRegion> rg;
            for (int s2 = 0; s2 < n; s2++)
                for (int h = 0; h < KV; h++) rg.push_back(CcRegion{((size_t)h * rows + pb + s2) * kvd, (size_t)s2 * KV * kvd + (size_t)h * kvd, (size_t)kvd});
            rg.push_back(CcRegion{(size_t)KV * rows * kvd - 10, 3, 10});
            std::vector<uint16_t> want((size_t)KV * rows * kvd, 0), got(want.size());
            for (auto &r : rg) kv16_pack(want.data() + r.dst, src.data() + r.src, (int64_t)r.n);
            CcBuf *sbuf = up(src), *dbuf = T->buf(want.size() * 2, CC_DEV);
            int ok = dbuf && T->begin() && T->copy_regions_h(dbuf, sbuf, rg.data(), (int)rg.size()) && T->submit(1) &&
                     T->read(dbuf, 0, got.data(), got.size() * 2);
            ck(ok && got == want, "copy_regions_h: a grid of rows and a region alone, each value its float16");
            T->free(sbuf); T->free(dbuf);
        }
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
    if (T->size >= offsetof(ColiCudaChainOps, lfm3) + sizeof T->lfm3 && T->lfm3 && T->lfm3()) {   /* LFM3's gated convolution */
        const int CD = 200, CK = 7, S = 5;
        std::vector<float> in = rnd((size_t)S * 3 * CD, 54), w = rnd((size_t)CD * CK, 55), ring = rnd((size_t)CD * (CK - 1), 56), out((size_t)S * CD), snap((size_t)CD * (CK - 1));
        std::vector<float> rr = ring, ro = out, rs = snap;
        CcDnConv p = {S, CD, CK, 0, 3 * CD, 0, CD, 1, 2, 0, 0, 0};
        ref_dnconv(in.data(), w.data(), rr.data(), ro.data(), rs.data(), &p);
        CcBuf *ib = up(in), *wb = up(w), *rb = up(ring), *ob = T->buf(out.size() * 4, CC_DEV), *sb = T->buf(snap.size() * 4, CC_DEV);
        ck(T->begin() && T->dnconv(ib, wb, rb, ob, sb, &p) && T->submit(1) &&
           maxdiff(down(ob, out.size()).data(), ro.data(), out.size()) < 1e-5 && maxdiff(down(rb, ring.size()).data(), rr.data(), ring.size()) == 0 &&
           maxdiff(down(sb, snap.size()).data(), rs.data(), snap.size()) == 0, "LFM3's order 2: C * conv(B * x), the ring of B * x, the snapshot after row 1");
        T->free(ib); T->free(wb); T->free(rb); T->free(ob); T->free(sb);
    }
    printf("dnrec\n");
    /* four rows (cc_dnrec_k) and 37 (the prompt's kernels: cc_dnrec_rows_k for KD 8, cc_dnrec_split_k
     * for 64 and 128; blocks of rows past the first, the last part full, the snapshot inside one) */
    for (int it = 0; it < 6; it++) {
        const int KD = it % 3 == 1 ? 128 : it % 3 == 2 ? 64 : 8, S = it < 3 ? 4 : 37, snap_at = S == 4 ? 1 : 20;
        const int VH = 4, KH = 2, VD = KD == 8 ? 16 : 128, Ktot = KH * KD, CD = 2 * Ktot + VH * VD, vd = VH * VD;
        std::vector<float> cv = rnd((size_t)S * CD, 61), ab = rnd((size_t)S * 2 * VH, 62), z = rnd((size_t)S * (CD + vd), 63), st = rnd((size_t)VH * KD * VD, 64, 0.1f);
        std::vector<float> prm(2 * VH + VD); unsigned s = 65; for (int h = 0; h < VH; h++) { prm[h] = -1.f + 0.5f * frand(&s); prm[VH + h] = 0.2f * frand(&s); } for (int v = 0; v < VD; v++) prm[2 * VH + v] = 1.f + 0.2f * frand(&s);
        std::vector<float> y((size_t)S * vd), snap((size_t)VH * KD * VD), rst = st, ry = y, rs = snap;
        CcDnRec p = {S, VH, KH, VD, Ktot, 0, CD, 0, 2 * VH, VH, 2 * VH, CD, CD + vd, 0, vd, snap_at, 0, 1e-6f, 1.f / sqrtf((float)KD), 0, 0, 0};
        ref_dnrec(KD, cv.data(), ab.data(), z.data(), rst.data(), prm.data(), ry.data(), rs.data(), &p);
        CcBuf *cb = up(cv), *bb = up(ab), *zb = up(z), *sb = up(st), *pb = up(prm), *yb = T->buf(y.size() * 4, CC_DEV), *nb = T->buf(snap.size() * 4, CC_DEV);
        char what[96];
        snprintf(what, sizeof what, "KD %d: %d rows, the silu gate, z inside a wider row", KD, S);
        ck(T->begin() && T->dnrec(KD, cb, bb, zb, sb, pb, yb, nb, &p) && T->submit(1) && maxdiff(down(yb, y.size()).data(), ry.data(), y.size()) < 1e-4, what);
        ck(maxdiff(down(sb, st.size()).data(), rst.data(), st.size()) < 1e-4 && maxdiff(down(nb, snap.size()).data(), rs.data(), snap.size()) < 1e-4, "the state carried, the snapshot");
        CcDnRec p2 = p; p2.flags = 1; p2.snap_row = -1;
        ref_dnrec(KD, cv.data(), ab.data(), z.data(), rst.data(), prm.data(), ry.data(), rs.data(), &p2);
        ck(T->begin() && T->dnrec(KD, cb, bb, zb, sb, pb, yb, NULL, &p2) && T->submit(1) && maxdiff(down(yb, y.size()).data(), ry.data(), y.size()) < 1e-4 && maxdiff(down(sb, st.size()).data(), rst.data(), st.size()) < 1e-4, "the sigmoid gate, from the carried state");
        T->free(cb); T->free(bb); T->free(zb); T->free(sb); T->free(pb); T->free(yb); T->free(nb);
    }
    /* dnconv_mx, dnrec_mx: the decode row of each of five conversations in one launch against five launches,
     * to the bit (output rows, rings, states); one buffer given twice, a snapshot and two rows are not taken */
    if (T->size >= offsetof(ColiCudaChainOps, dnrec_mx) + sizeof T->dnrec_mx && T->dnconv_mx && T->dnrec_mx) {
        printf("dnconv_mx, dnrec_mx\n");
        for (int it = 0; it < 3; it++) {
            const int N = 5, KD = it == 0 ? 8 : it == 1 ? 64 : 128, VH = 4, KH = 2, VD = KD == 8 ? 16 : 128, Ktot = KH * KD;
            const int CD = 2 * Ktot + VH * VD, vd = VH * VD, PD = CD + vd, CK = 4;
            std::vector<float> in = rnd((size_t)N * PD, 71), w = rnd((size_t)CD * CK, 72, 0.3f), ab = rnd((size_t)N * 2 * VH, 73);
            std::vector<float> prm(2 * VH + VD); unsigned s = 74; for (int h = 0; h < VH; h++) { prm[h] = -1.f + 0.5f * frand(&s); prm[VH + h] = 0.2f * frand(&s); } for (int v = 0; v < VD; v++) prm[2 * VH + v] = 1.f + 0.2f * frand(&s);
            CcBuf *ib = up(in), *wb = up(w), *bb = up(ab), *pb = up(prm);
            CcBuf *cv1 = T->buf((size_t)N * CD * 4, CC_DEV), *cvm = T->buf((size_t)N * CD * 4, CC_DEV), *y1 = T->buf((size_t)N * vd * 4, CC_DEV), *ym = T->buf((size_t)N * vd * 4, CC_DEV);
            CcBuf *r1[N], *rm[N], *s1[N], *sm[N]; int ok = ib && wb && bb && pb && cv1 && cvm && y1 && ym;
            for (int j = 0; j < N; j++) {
                std::vector<float> ring = rnd((size_t)CD * (CK - 1), 80 + j, 0.5f), st = rnd((size_t)VH * KD * VD, 90 + j, 0.1f);
                r1[j] = up(ring); rm[j] = up(ring); s1[j] = up(st); sm[j] = up(st); ok = ok && r1[j] && rm[j] && s1[j] && sm[j];
            }
            ok = ok && T->begin();
            for (int j = 0; j < N && ok; j++) {
                CcDnConv cp = {1, CD, CK, j * PD, PD, j * CD, CD, -1, 0, 0, 0, 0};
                CcDnRec rp = {1, VH, KH, VD, Ktot, j * CD, CD, j * 2 * VH, 2 * VH, VH + j * 2 * VH, 2 * VH, CD + j * PD, PD, j * vd, vd, -1, 0, 1e-6f, 1.f / sqrtf((float)KD), 0, 0, 0};
                ok = T->dnconv(ib, wb, r1[j], cv1, NULL, &cp) && T->dnrec(KD, cv1, bb, ib, s1[j], pb, y1, NULL, &rp);
            }
            ok = ok && T->submit(1);
            CcDnConv cp0 = {1, CD, CK, 0, PD, 0, CD, -1, 0, 0, 0, 0};
            CcDnRec rp0 = {1, VH, KH, VD, Ktot, 0, CD, 0, 2 * VH, VH, 2 * VH, CD, PD, 0, vd, -1, 0, 1e-6f, 1.f / sqrtf((float)KD), 0, 0, 0};
            int rc = 0, rr = 0;
            ok = ok && T->begin() && (rc = T->dnconv_mx(ib, wb, rm, N, cvm, &cp0)) != 0 && (rr = T->dnrec_mx(KD, cvm, bb, ib, sm, N, pb, ym, &rp0)) != 0 && T->submit(1);
            char what[128];
            {   /* a recurrence kernel other than the default's (COLI_CUDA_DNREC_ROWS=0, _DEC=0, _MIN<=1) is not taken as one launch */
                const char *er = getenv("COLI_CUDA_DNREC_ROWS"), *ed = getenv("COLI_CUDA_DNREC_DEC"), *em = getenv("COLI_CUDA_DNREC_MIN");
                if (ok && rr == -1 && ((er && *er == '0') || (ed && *ed == '0') || (em && *em && atoi(em) <= 1))) {
                    printf("  skip KD %d: five conversations as one launch: the recurrence's kernel is not the default's (dnrec_mx -1)\n", KD);
                    for (int j = 0; j < N; j++) { T->free(r1[j]); T->free(rm[j]); T->free(s1[j]); T->free(sm[j]); }
                    T->free(ib); T->free(wb); T->free(bb); T->free(pb); T->free(cv1); T->free(cvm); T->free(y1); T->free(ym);
                    continue;
                }
            }
            snprintf(what, sizeof what, "KD %d: five conversations, the convolution and the recurrence taken as one launch each", KD);
            ck(ok && rc == 1 && rr == 1, what);
            int same = ok && !memcmp(down(cv1, (size_t)N * CD).data(), down(cvm, (size_t)N * CD).data(), (size_t)N * CD * 4) &&
                       !memcmp(down(y1, (size_t)N * vd).data(), down(ym, (size_t)N * vd).data(), (size_t)N * vd * 4);
            for (int j = 0; j < N && same; j++)
                same = !memcmp(down(r1[j], (size_t)CD * (CK - 1)).data(), down(rm[j], (size_t)CD * (CK - 1)).data(), (size_t)CD * (CK - 1) * 4) &&
                       !memcmp(down(s1[j], (size_t)VH * KD * VD).data(), down(sm[j], (size_t)VH * KD * VD).data(), (size_t)VH * KD * VD * 4);
            ck(same, "their outputs, rings and states those of five launches, to the bit");
            if (it == 0 && ok) {
                CcBuf *twice[2] = {sm[0], sm[0]};
                CcDnRec snapped = rp0; snapped.snap_row = 0;
                CcDnRec two = rp0; two.S = 2;
                ck(T->begin() && T->dnrec_mx(KD, cvm, bb, ib, twice, 2, pb, ym, &rp0) == -1 && T->dnrec_mx(KD, cvm, bb, ib, sm, N, pb, ym, &snapped) == -1 &&
                   T->dnrec_mx(KD, cvm, bb, ib, sm, N, pb, ym, &two) == -1 && T->dnrec_mx(KD, cvm, bb, ib, sm, 1, pb, ym, &rp0) == -1 && T->submit(1),
                   "one state twice, a snapshot, two rows and one conversation are not taken");
                CcBuf *ring2[2] = {rm[0], rm[0]};
                CcDnConv csnap = cp0; csnap.snap_row = 0;
                CcDnConv ctwo = cp0; ctwo.S = 2;
                ck(T->begin() && T->dnconv_mx(ib, wb, ring2, 2, cvm, &cp0) == -1 && T->dnconv_mx(ib, wb, rm, N, cvm, &csnap) == -1 &&
                   T->dnconv_mx(ib, wb, rm, N, cvm, &ctwo) == -1 && T->dnconv_mx(ib, wb, rm, 1, cvm, &cp0) == -1 && T->submit(1),
                   "the convolution too: one ring twice, a snapshot, two rows and one conversation are not taken");
            }
            for (int j = 0; j < N; j++) { T->free(r1[j]); T->free(rm[j]); T->free(s1[j]); T->free(sm[j]); }
            T->free(ib); T->free(wb); T->free(bb); T->free(pb); T->free(cv1); T->free(cvm); T->free(y1); T->free(ym);
        }
    }
    /* attn_h_mx, copy_regions_h_mx: the decode row of each of five conversations at its own position in float16
     * caches of its own length: the K and V rows into the caches as one launch a cache and the attention as one
     * call, against five calls of each on copies of the caches, to the bit (outputs, whole caches). Blocks by KV
     * head (G 8, 2, 4) and by head (G 1); positions 0, 63 and 64 (one position, a full block, a block and one),
     * and a last conversation whose blocks the rule doubles in the second and third shapes where the others keep
     * 64, so the shorter ones leave empty blocks in the scratch. What is not a decode step's default is not taken */
    if (T->size >= offsetof(ColiCudaChainOps, copy_regions_h_mx) + sizeof T->copy_regions_h_mx && T->attn_h_mx && T->copy_regions_h_mx &&
        T->kvh && T->kvh()) {
        printf("attn_h_mx, copy_regions_h_mx\n");
        struct { int H, KV, hd, gate, last; } cs[] = {
            {16, 2, 256, 1, 2500}, {32, 4, 128, 1, 5000}, {8, 8, 64, 0, 9000}, {4, 2, 128, 0, 900}, {8, 2, 64, 1, 900}};
        int refusals = 1;
        for (auto &k : cs) {
            const int N = 5, qdim = k.gate ? 2 * k.hd : k.hd, qo = k.H * qdim, kvo = k.KV * k.hd, oo = k.H * k.hd;
            const int pos[N] = {0, 63, 64, 700, k.last}, cap[N] = {8, 64, 100, 1024, k.last + 100};
            const float scale = 1.f / sqrtf((float)k.hd);
            std::vector<float> q = rnd((size_t)N * qo, 131), kr = rnd((size_t)N * kvo, 132), vr = rnd((size_t)N * kvo, 133);
            CcBuf *qb = up(q), *krb = up(kr), *vrb = up(vr), *o1 = T->buf((size_t)N * oo * 4, CC_DEV), *om = T->buf((size_t)N * oo * 4, CC_DEV);
            CcBuf *gb = k.gate ? qb : NULL, *k1[N], *km[N], *v1[N], *vm[N];
            int ok = qb && krb && vrb && o1 && om;
            std::vector<CcRegion> rg;
            for (int j = 0; j < N; j++) {
                const size_t ne = (size_t)k.KV * cap[j] * k.hd;
                std::vector<float> kc = rnd(ne, 140 + j), vc = rnd(ne, 150 + j);
                std::vector<uint16_t> kh(ne), vh(ne);
                kv16_pack(kh.data(), kc.data(), (int64_t)ne); kv16_pack(vh.data(), vc.data(), (int64_t)ne);
                k1[j] = T->buf(ne * 2, CC_DEV); km[j] = T->buf(ne * 2, CC_DEV); v1[j] = T->buf(ne * 2, CC_DEV); vm[j] = T->buf(ne * 2, CC_DEV);
                ok = ok && k1[j] && km[j] && v1[j] && vm[j] && T->begin() &&
                     T->write(k1[j], 0, kh.data(), ne * 2) && T->write(km[j], 0, kh.data(), ne * 2) &&
                     T->write(v1[j], 0, vh.data(), ne * 2) && T->write(vm[j], 0, vh.data(), ne * 2) && T->submit(1);
                for (int h = 0; h < k.KV; h++)
                    rg.push_back(CcRegion{((size_t)h * cap[j] + pos[j]) * k.hd, (size_t)j * kvo + (size_t)h * k.hd, (size_t)k.hd});
            }
            ok = ok && T->begin();
            for (int j = 0; j < N && ok; j++) {
                CcAttn a = {1, k.H, k.KV, k.hd, pos[j], cap[j], j * qo, qo, qdim, j * qo + k.hd, qo, qdim, k.gate, j * oo, oo, 0, 0, scale, 0, 0};
                ok = T->copy_regions_h(k1[j], krb, rg.data() + (size_t)j * k.KV, k.KV) && T->copy_regions_h(v1[j], vrb, rg.data() + (size_t)j * k.KV, k.KV) &&
                     T->attn_h(qb, k1[j], v1[j], o1, gb, NULL, &a);
            }
            ok = ok && T->submit(1);
            CcAttn a0 = {1, k.H, k.KV, k.hd, 0, 0, 0, qo, qdim, k.hd, qo, qdim, k.gate, 0, oo, 0, 0, scale, 0, 0};
            int rk = 0, rv = 0, ra = 0;
            ok = ok && T->begin() && (rk = T->copy_regions_h_mx(km, N, krb, rg.data(), k.KV)) != 0 &&
                 (rv = T->copy_regions_h_mx(vm, N, vrb, rg.data(), k.KV)) != 0 &&
                 (ra = T->attn_h_mx(qb, km, vm, pos, cap, N, om, gb, &a0)) != 0 && T->submit(1);
            char what[200];
            snprintf(what, sizeof what, "H %d KV %d hd %d%s: five conversations' K and V rows taken as one launch a cache", k.H, k.KV, k.hd,
                     k.gate ? ", gated" : "");
            ck(ok && rk == 1 && rv == 1, what);
            int same = ok;
            for (int j = 0; j < N && same; j++) {
                const size_t ne = (size_t)k.KV * cap[j] * k.hd;
                std::vector<uint16_t> a(ne), b(ne);
                same = T->read(k1[j], 0, a.data(), ne * 2) && T->read(km[j], 0, b.data(), ne * 2) && !memcmp(a.data(), b.data(), ne * 2) &&
                       T->read(v1[j], 0, a.data(), ne * 2) && T->read(vm[j], 0, b.data(), ne * 2) && !memcmp(a.data(), b.data(), ne * 2);
            }
            ck(same, "their caches those of five launches a cache, to the bit");
            /* an attention kernel other than a decode row's default (COLI_CUDA_ATTN_DEC=0|1, COLI_CUDA_ATTN_DEC_MIN above
             * one position, COLI_CUDA_CHAIN_FLASH=1) is not taken as one call */
            const char *ed = getenv("COLI_CUDA_ATTN_DEC"), *em = getenv("COLI_CUDA_ATTN_DEC_MIN"), *ef = getenv("COLI_CUDA_CHAIN_FLASH");
            if (ok && ra == -1 && ((ed && (*ed == '0' || *ed == '1')) || (em && *em && atoi(em) > 1) || (ef && *ef && atoi(ef) == 1)))
                printf("  skip H %d KV %d hd %d: the attention as one call: a decode row's kernel is not the default's (attn_h_mx -1)\n", k.H, k.KV, k.hd);
            else {
                ck(ok && ra == 1, "their attention taken as one call");
                std::vector<float> x1 = down(o1, (size_t)N * oo), xm = down(om, (size_t)N * oo);
                const int so = ok && !memcmp(x1.data(), xm.data(), x1.size() * 4);
                ck(so, "their outputs those of five calls, to the bit");
            }
            if (refusals && ok) {
                refusals = 0;
                CcAttn two = a0; two.S = 2;
                CcAttn lst = a0; lst.sel_row = 4;
                int past[N]; for (int j = 0; j < N; j++) past[j] = pos[j];
                past[1] = cap[1];
                CcBuf *k9[CC_MX_ATTN_N + 1], *v9[CC_MX_ATTN_N + 1]; int p9[CC_MX_ATTN_N + 1], c9[CC_MX_ATTN_N + 1];
                for (int j = 0; j < CC_MX_ATTN_N + 1; j++) { k9[j] = km[j % N]; v9[j] = vm[j % N]; p9[j] = pos[j % N]; c9[j] = cap[j % N]; }
                ck(T->begin() && T->attn_h_mx(qb, km, vm, pos, cap, N, om, gb, &two) == -1 && T->attn_h_mx(qb, km, vm, pos, cap, N, om, gb, &lst) == -1 &&
                   T->attn_h_mx(qb, km, vm, pos, cap, 1, om, gb, &a0) == -1 && T->attn_h_mx(qb, k9, v9, p9, c9, CC_MX_ATTN_N + 1, om, gb, &a0) == -1 &&
                   T->attn_h_mx(qb, km, vm, past, cap, N, om, gb, &a0) == -1 && T->submit(1),
                   "the attention: two rows, a selection list, one conversation, one too many and a position past its cache are not taken");
                CcBuf *twice[2] = {km[0], km[0]};
                std::vector<CcRegion> odd(rg.begin(), rg.begin() + 2 * k.KV);
                odd[1].n -= 1;   /* no grid: the lengths differ */
                ck(T->begin() && T->copy_regions_h_mx(twice, 2, krb, rg.data(), k.KV) == -1 && T->copy_regions_h_mx(km, 1, krb, rg.data(), k.KV) == -1 &&
                   T->copy_regions_h_mx(km, 2, krb, odd.data(), k.KV) == -1 && T->submit(1),
                   "the rows: one cache twice, one conversation and regions that are no grid are not taken");
            }
            for (int j = 0; j < N; j++) { T->free(k1[j]); T->free(km[j]); T->free(v1[j]); T->free(vm[j]); }
            T->free(qb); T->free(krb); T->free(vrb); T->free(o1); T->free(om);
        }
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
            {CC_EW_HC_APPLY, R * C * D, C, 0, 1, 1.f, "HC_APPLY"}, {CC_EW_SCALE, n, 1, 0, 1, 0.25f, "SCALE"}, {CC_EW_GATE_ADD, n, 1, 0, 1, 1.f, "GATE_ADD"},
            {CC_EW_HEADGATE, n, 1, 0, 1, 1.f, "HEADGATE (LFM3)"} };
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
        {   /* 1100 blocks (Qwen3.8's 4400 positions, budget 2048): of 37 keys and some zero ones, so many
             * scores tie, the take-th among them; the device's select against the CPU's ranking */
            const int NB = 1100, S3 = 2, B3 = 2048, nbm3 = NB + 4, sr3 = 1 + B3 + R - 1;
            std::vector<float> kk = rnd((size_t)37 * ID, 94), pk3((size_t)nbm3 * ID, 0.f);
            for (int b = 0; b < NB; b++) if (b % 11) for (int d = 0; d < ID; d++) pk3[(size_t)b * ID + d] = kk[(size_t)(b % 37) * ID + d];
            std::vector<float> iq3 = rnd((size_t)S3 * (IQ + 1) * ID, 95), sc3((size_t)S3 * 2 * nbm3), rsc3 = sc3;
            std::vector<int> sel3((size_t)S3 * sr3), rsel3 = sel3, got3(sel3.size());
            CcQsa p3 = {1, ID, R, 0, 0, S3, NB * R + 1, B3, IQ, 0, (IQ + 1) * ID, nbm3, sr3, 1e-6f, 0, 0, 0, 0};
            ref_qsa(iq3.data(), NULL, pk3.data(), NULL, rsc3.data(), rsel3.data(), &p3);
            CcBuf *pb3 = up(pk3), *qb3 = up(iq3), *sb3 = T->buf(sc3.size() * 4, CC_DEV), *lb3 = T->buf(sel3.size() * 4, CC_DEV);
            ok = T->begin() && T->qsa(qb3, NULL, pb3, NULL, sb3, lb3, &p3) && T->submit(1) && T->read(lb3, 0, got3.data(), got3.size() * 4);
            same = 1; for (size_t i = 0; i < got3.size(); i++) if (got3[i] != rsel3[i]) same = 0;
            ck(ok && same && rsel3[0] == B3 + 2 && rsel3[sr3] == B3 + 3, "1100 blocks of tied scores: the take-th key's blocks by index, then the tail, as the CPU ranks them");
            T->free(pb3); T->free(qb3); T->free(sb3); T->free(lb3);
        }
        {   /* a prompt chunk's 70 rows (cc_qsa_score_t, tiles of 64 rows by 64 blocks), Qwen3.8's index heads (4 of 128):
             * the same selections as a thread a (row, block) (COLI_CUDA_QSA_TILES=0) and as the CPU ranks them */
            const int ID4 = 128, IQ4 = 4, S4 = 70, PB4 = 2000, B4 = 512, nbm4 = 600, sr4 = 1 + B4 + R - 1, NB4 = (PB4 + S4) / R;
            std::vector<float> pk4 = rnd((size_t)nbm4 * ID4, 96), iq4 = rnd((size_t)S4 * (IQ4 + 1) * ID4, 97);
            for (int b = NB4; b < nbm4; b++) for (int d = 0; d < ID4; d++) pk4[(size_t)b * ID4 + d] = 0.f;
            std::vector<float> rsc4((size_t)S4 * 2 * nbm4);
            std::vector<int> rsel4((size_t)S4 * sr4), t4(rsel4.size()), k4(rsel4.size());
            CcQsa p4 = {1, ID4, R, 0, 0, S4, PB4, B4, IQ4, 0, (IQ4 + 1) * ID4, nbm4, sr4, 1e-6f, 0, 0, 0, 0};
            ref_qsa(iq4.data(), NULL, pk4.data(), NULL, rsc4.data(), rsel4.data(), &p4);
            CcBuf *pb4 = up(pk4), *qb4 = up(iq4), *sb4 = T->buf(rsc4.size() * 4, CC_DEV), *lb4 = T->buf(rsel4.size() * 4, CC_DEV);
            ok = T->begin() && T->qsa(qb4, NULL, pb4, NULL, sb4, lb4, &p4) && T->submit(1) && T->read(lb4, 0, t4.data(), t4.size() * 4);
            setenv("COLI_CUDA_QSA_TILES", "0", 1);
            ok = ok && T->begin() && T->qsa(qb4, NULL, pb4, NULL, sb4, lb4, &p4) && T->submit(1) && T->read(lb4, 0, k4.data(), k4.size() * 4);
            unsetenv("COLI_CUDA_QSA_TILES");
            ck(ok && !memcmp(t4.data(), k4.data(), t4.size() * 4), "70 rows' selections in tiles: those of a thread a (row, block), to the bit");
            same = 1; for (size_t i = 0; i < t4.size(); i++) if (t4[i] != rsel4[i]) same = 0;
            ck(ok && same, "and as the CPU ranks them");
            T->free(pb4); T->free(qb4); T->free(sb4); T->free(lb4);
        }
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
    printf("route\n");
    {
        struct { int S, E, K, kind, has_bias, n_group, topk_group, norm; float temp, scale; const char *what; } cases[] = {
            {3, 256, 8, 0, 0, 0, 0, 1, 1.f, 1.f, "softmax, top 8 of 256, normalized (Qwen3.6)"},
            {2, 512, 10, 0, 0, 0, 0, 1, 1.f, 1.f, "softmax, top 10 of 512 (Qwen3.8)"},
            {2, 64, 6, 1, 1, 8, 4, 1, 1.f, 2.5f, "sigmoid, a bias for the pick, 4 of 8 groups, scaled"},
            {1, 100, 4, 2, 2, 0, 0, 2, 0.7f, 1.f, "sqrt(softplus), the bias in the logit, a temperature"},
            {5, 64, 8, 0, 0, 0, 0, 0, 1.f, 1.f, "softmax, top 8 of 64 (OLMoE)"},
            {3, 128, 8, 1, 1, 0, 0, 1, 1.f, 2.5f, "sigmoid, a bias for the pick, top 8 of 128, scaled"},
            {2, 256, 32, 0, 0, 0, 0, 0, 1.f, 1.f, "softmax, top 32 of 256, not normalized"},
            {2, 200, 6, 0, 2, 0, 0, 1, 1.3f, 1.f, "softmax, the bias in the logit, a temperature, 200 experts"},
        };
        for (auto &cs : cases) {
            CcRoute p = {cs.S, cs.E, cs.K, cs.kind, 5, cs.E + 3, 2, cs.has_bias, 1, cs.K + 2, 3, cs.K + 1, cs.n_group, cs.topk_group, cs.norm, cs.temp, cs.scale, 1e-20f};
            std::vector<float> l = rnd((size_t)5 + cs.S * (cs.E + 3), 101 + cs.E, 4.f), b = rnd(cs.E + 2, 202, 0.5f);
            std::vector<int> rid((size_t)1 + cs.S * (cs.K + 2), -7); std::vector<float> rw((size_t)3 + cs.S * (cs.K + 1), -7.f);
            ref_route(l.data(), b.data(), rid.data(), rw.data(), &p);
            CcBuf *lb = up(l), *bb = up(b), *ib = T->buf(rid.size() * 4, CC_DEV), *wb = T->buf(rw.size() * 4, CC_DEV);
            int ok = T->begin() && T->route(lb, cs.has_bias ? bb : NULL, ib, wb, &p) && T->submit(1);
            std::vector<int> gi(rid.size()); std::vector<float> gw = down(wb, rw.size());
            T->read(ib, 0, gi.data(), gi.size() * 4);
            int same = 1; double wd = 0;
            for (int s = 0; s < cs.S; s++) for (int k = 0; k < cs.K; k++) {
                same &= gi[1 + s * (cs.K + 2) + k] == rid[1 + s * (cs.K + 2) + k];
                wd = fmax(wd, fabs(gw[3 + s * (cs.K + 1) + k] - rw[3 + s * (cs.K + 1) + k]));
            }
            if (!(ok && same && wd < 1e-6)) {
                printf("  (route: ok %d, weights off by %.3g)\n", ok, wd);
                for (int s = 0; s < cs.S; s++) for (int k = 0; k < cs.K; k++) {
                    int a = gi[1 + s * (cs.K + 2) + k], b = rid[1 + s * (cs.K + 2) + k];
                    if (a != b) printf("  (row %d slot %d: device %d, CPU %d, scores %.9g %.9g)\n", s, k, a, b, a >= 0 && a < cs.E ? l[5 + s * (cs.E + 3) + a] : 0.f, b >= 0 ? l[5 + s * (cs.E + 3) + b] : 0.f);
                }
            }
            ck(ok && same && wd < 1e-6, cs.what);
            T->free(lb); T->free(bb); T->free(ib); T->free(wb);
        }
    }
    printf("moe\n");
    {
        { float lut[256]; for (int b = 0; b < 256; b++) lut[b] = e4m3(b); coli_cuda_fp8_set_lut(lut); }   /* fmt 8 uploads need it */
        const int S = 2, K = 3, E = 6, D = 256, I = 128;
        unsigned seed = 300;
        for (int fmt : {1, 2, 4, 8}) {
            int gs = fmt == 4 ? 64 : 0;
            std::vector<std::vector<float>> G(E), U(E), Dn(E);
            std::vector<ColiCudaTensor *> tg(E), tu(E), td(E);
            std::vector<CcExpert> tab(E + 2);   /* the layer's entries start at t_off 2 */
            int made = 1;
            for (int e = 0; e < E; e++) {
                tg[e] = moe_mat(fmt, gs, D, I, seed++, G[e]); tu[e] = moe_mat(fmt, gs, D, I, seed++, U[e]); td[e] = moe_mat(fmt, gs, I, D, seed++, Dn[e]);
                made &= tg[e] && tu[e] && td[e];
                CcExpert &x = tab[2 + e]; memset(&x, 0, sizeof x);
                if (made) { T->tensor_dev(tg[e], &x.g, &x.gs, NULL); T->tensor_dev(tu[e], &x.u, &x.us, NULL); T->tensor_dev(td[e], &x.d, &x.ds, NULL); }
            }
            ck(made, fmt == 1 ? "int8 experts resident" : fmt == 2 ? "int4 experts resident" : fmt == 4 ? "int4 experts with a scale per 64 resident" : "fp8 experts resident");
            if (!made) continue;
            std::vector<float> x = rnd((size_t)S * D + 4, 400 + fmt), hy = rnd((size_t)S * D, 500 + fmt), w = {0.5f, 0.3f, 0.2f, 0.7f, 0.2f, 0.1f};
            std::vector<int> ids = {4, 1, -1, 0, 5, 2};   /* row 0's last slot empty */
            /* experts 1 and 5 not on the device: the host's (its rows in hy) */
            std::vector<CcExpert> part = tab; memset(&part[2 + 1], 0, sizeof(CcExpert)); memset(&part[2 + 5], 0, sizeof(CcExpert));
            auto expert = [&](int e, const float *xr, std::vector<float> &out) {
                std::vector<float> h(I); out.assign(D, 0.f);
                for (int o = 0; o < I; o++) {
                    double g = 0, u = 0; for (int i = 0; i < D; i++) {
                        g += xr[i] * G[e][(size_t)o * D + i];
                        u += xr[i] * U[e][(size_t)o * D + i];
                    } h[o] = (float)(g / (1 + expf(-g)) * u);
                }
                for (int o = 0; o < D; o++) { double v = 0; for (int i = 0; i < I; i++) v += h[i] * Dn[e][(size_t)o * I + i]; out[o] = (float)v; }
            };
            std::vector<float> ref_full((size_t)S * D, 0.f), ref_part((size_t)S * D, 0.f), ye;
            for (int s = 0; s < S; s++) {
                for (int d = 0; d < D; d++) ref_part[(size_t)s * D + d] = hy[(size_t)s * D + d];
                for (int k = 0; k < K; k++) {
                    int e = ids[s * K + k]; if (e < 0) continue;
                    expert(e, x.data() + 4 + (size_t)s * D, ye);
                    for (int d = 0; d < D; d++) { ref_full[(size_t)s * D + d] += w[s * K + k] * ye[d]; if (e != 1 && e != 5) ref_part[(size_t)s * D + d] += w[s * K + k] * ye[d]; }
                }
            }
            CcBuf *tb = T->buf(tab.size() * sizeof(CcExpert), CC_DEV), *pb = T->buf(part.size() * sizeof(CcExpert), CC_DEV);
            CcBuf *xb = up(x), *ib = T->buf(ids.size() * 4, CC_DEV), *wb = up(w), *hb = T->buf((size_t)S * K * (I + D) * 4, CC_DEV);
            CcBuf *yb = T->buf(((size_t)S * D + 8) * 4, CC_DEV), *hyb = up(hy, CC_UP);
            int ok = tb && pb && T->begin() && T->write(tb, 0, tab.data(), tab.size() * sizeof(CcExpert)) && T->write(pb, 0, part.data(), part.size() * sizeof(CcExpert)) &&
                     T->write(ib, 0, ids.data(), ids.size() * 4) && T->submit(1);
            CcMoe m = {S, K, D, I, fmt, gs, 0, 3, 2, 4, D, 0, K, 0, K, 8, D, -1, E, 0, 0, 0, 0};
            ok = ok && T->begin() && T->moe(tb, xb, ib, wb, hb, yb, NULL, &m) && T->submit(1);
            std::vector<float> y = down(yb, (size_t)S * D + 8);
            double scale = 0; for (float v : ref_full) scale = fmax(scale, fabs(v));
            ck(ok && maxdiff(y.data() + 8, ref_full.data(), (size_t)S * D) < 2e-5 * scale + 1e-6, "every slot on the device, rank order, an empty slot");
            CcMoe m1 = m; m1.parts = 1; CcMoe m2 = m; m2.parts = 2; m2.h_off = 0;
            ok = T->begin() && T->moe(pb, xb, ib, wb, hb, yb, hyb, &m1) && T->submit(0) && T->begin() && T->moe(pb, xb, ib, wb, hb, yb, hyb, &m2) && T->submit(1);
            y = down(yb, (size_t)S * D + 8);
            ck(ok && maxdiff(y.data() + 8, ref_part.data(), (size_t)S * D) < 2e-5 * scale + 1e-6, "two of the table's experts the host's: its rows plus the device's slots");
            {   /* a prompt chunk's rows (more than CC_MOE_FRAME_ROWS): sorted by expert, the tensor cores' GEMMs */
                const int S2 = 40;
                unsigned rs = 900 + fmt;
                std::vector<int> ids2((size_t)S2 * K); std::vector<float> w2((size_t)S2 * K), x2 = rnd((size_t)S2 * D, 950 + fmt), hy2 = rnd((size_t)S2 * D, 960 + fmt);
                for (int a = 0; a < S2 * K; a++) { int v = (int)((frand(&rs) * 0.5f + 0.5f) * (E - 0.01f)); ids2[a] = a % 17 == 5 ? -1 : v; w2[a] = 0.1f + 0.3f * (frand(&rs) * 0.5f + 0.5f); }
                std::vector<float> rf((size_t)S2 * D, 0.f), rp((size_t)S2 * D, 0.f);
                for (int s2 = 0; s2 < S2; s2++) {
                    for (int d = 0; d < D; d++) rp[(size_t)s2 * D + d] = hy2[(size_t)s2 * D + d];
                    for (int k = 0; k < K; k++) {
                        int e = ids2[s2 * K + k]; if (e < 0) continue;
                        expert(e, x2.data() + (size_t)s2 * D, ye);
                        for (int d = 0; d < D; d++) { rf[(size_t)s2 * D + d] += w2[s2 * K + k] * ye[d]; if (e != 1 && e != 5) rp[(size_t)s2 * D + d] += w2[s2 * K + k] * ye[d]; }
                    }
                }
                double sc2 = 0; for (float v : rf) sc2 = fmax(sc2, fabs(v));
                CcBuf *x2b = up(x2), *i2b = T->buf(ids2.size() * 4, CC_DEV), *w2b = up(w2), *h2b = T->buf(cc_moe_hid_floats(S2, K, D, I, E) * 4, CC_DEV);
                CcBuf *y2b = T->buf((size_t)S2 * D * 4, CC_DEV), *hy2b = up(hy2, CC_UP);
                CcMoe m2 = {S2, K, D, I, fmt, gs, 0, 3, 2, 0, D, 0, K, 0, K, 0, D, -1, E, 0, 0, 0, 0};
                int ok2 = T->begin() && T->write(i2b, 0, ids2.data(), ids2.size() * 4) && T->moe(tb, x2b, i2b, w2b, h2b, y2b, NULL, &m2) && T->submit(1);
                double ef = ok2 ? maxdiff(down(y2b, (size_t)S2 * D).data(), rf.data(), (size_t)S2 * D) / sc2 : 1;
                CcMoe a2 = m2, b2 = m2; a2.parts = 1; b2.parts = 2; b2.h_off = 0;
                int okp = T->begin() && T->moe(pb, x2b, i2b, w2b, h2b, y2b, hy2b, &a2) && T->moe(pb, x2b, i2b, w2b, h2b, y2b, hy2b, &b2) && T->submit(1);
                double ep = okp ? maxdiff(down(y2b, (size_t)S2 * D).data(), rp.data(), (size_t)S2 * D) / sc2 : 1;
                if (ef >= 2e-3 || ep >= 2e-3) printf("  (prompt rows fmt %d: relative errors %.3g, %.3g)\n", fmt, ef, ep);
                ck(ok2 && ef < 2e-3, "forty rows: the slots sorted by expert, their GEMMs, the sum in rank order");
                ck(okp && ep < 2e-3, "forty rows, two experts the host's: its rows plus the device's slots");
                {   /* a prompt chunk's two lanes (CcMoe.lay_S): the forty rows' parts 1 above, then the sums of
                     * rows 0..15 and 16..39 apart over that layout, each with its own rows' offsets */
                    std::vector<float> yall = down(y2b, (size_t)S2 * D);
                    const int r0 = 16;
                    CcMoe la = b2, lb = b2;
                    la.S = r0; la.lay_S = S2;
                    lb.S = S2 - r0; lb.lay_S = S2; lb.lay_s0 = r0; lb.i_off = r0 * K; lb.w_off = r0 * K; lb.y_off = r0 * D; lb.h_off = r0 * D;
                    CcBuf *y3b = T->buf((size_t)S2 * D * 4, CC_DEV);
                    int okl = okp && y3b && T->begin() && T->moe(pb, x2b, i2b, w2b, h2b, y3b, hy2b, &la) && T->moe(pb, x2b, i2b, w2b, h2b, y3b, hy2b, &lb) && T->submit(1);
                    std::vector<float> yl = okl ? down(y3b, (size_t)S2 * D) : std::vector<float>();
                    ck(okl && !memcmp(yl.data(), yall.data(), (size_t)S2 * D * 4), "a chunk's two lanes' sums over one parts 1 of its rows: the chunk's sum's bits");
                    CcMoe bad = a2; bad.lay_S = S2 + 8;   /* a parts 1 has its own layout */
                    CcMoe past = lb; past.lay_s0 = r0 + 1;   /* rows past the layout's */
                    int refused = T->begin() && !T->moe(pb, x2b, i2b, w2b, h2b, y3b, hy2b, &bad) && !T->moe(pb, x2b, i2b, w2b, h2b, y3b, hy2b, &past) && T->submit(1);
                    ck(refused, "a parts 1 over another layout, and rows past the layout's, refused");
                }
                if (fmt == 4) {   /* experts 1 and 5 in planar blocks in host memory (the tier's RAM cache), read over the bus */
                    std::vector<CcExpert> tp = tab;
                    std::vector<void *> hosts;
                    uint8_t *hwv[2][3] = {}; float *hsv[2][3] = {};
                    int okh = 1;
                    for (int k = 0; k < 2; k++) {
                        int e = k ? 5 : 1;
                        CcExpert &xp = tp[2 + e];
                        unsigned s0 = 300 + 3 * (unsigned)e + (fmt == 4 ? 3u * E * 2u : 0u);   /* moe_mat's seeds for this expert and format */
                        uint8_t **hw = hwv[k]; float **hs = hsv[k];
                        okh &= moe_planar(gs, D, I, s0, &hw[0], &hs[0], &xp.g, &xp.gs) && moe_planar(gs, D, I, s0 + 1, &hw[1], &hs[1], &xp.u, &xp.us) &&
                               moe_planar(gs, I, D, s0 + 2, &hw[2], &hs[2], &xp.d, &xp.ds);
                        xp.planar = 1;
                        for (int j = 0; j < 3; j++) { hosts.push_back(hw[j]); hosts.push_back(hs[j]); }
                    }
                    CcBuf *tpb = T->buf(tp.size() * sizeof(CcExpert), CC_DEV);
                    CcMoe m3 = m2; m3.planar = 1;   /* the table holds planar entries: the f16 GEMMs */
                    int okq = okh && tpb && T->begin() && T->write(tpb, 0, tp.data(), tp.size() * sizeof(CcExpert)) &&
                              T->moe(tpb, x2b, i2b, w2b, h2b, y2b, NULL, &m3) && T->submit(1);
                    double eq = okq ? maxdiff(down(y2b, (size_t)S2 * D).data(), rf.data(), (size_t)S2 * D) / sc2 : 1;
                    if (eq >= 2e-3) printf("  (prompt rows, planar experts in host memory: relative error %.3g)\n", eq);
                    ck(okq && eq < 2e-3, "forty rows, two experts in planar blocks in host memory, read over the bus");
                    /* the same two staged into VRAM from those blocks (fetch: the tier's host pool for a prompt
                     * chunk), twice in one frame in swapped places: the second copy waits for the first
                     * table's GEMMs (fetch_done), or the first one reads half-copied weights */
                    if (okh && T->size >= offsetof(ColiCudaChainOps, dev_addr) + sizeof T->dev_addr) {
                        size_t mb = (size_t)D * I / 2, nsc = (size_t)D * I / gs, wa = (3 * mb + 255) / 256 * 256;
                        size_t st = wa + (3 * nsc * 4 + 255) / 256 * 256;
                        CcBuf *sg = T->buf(2 * st, CC_DEV);
                        const char *base = sg ? (const char *)T->dev_addr(sg) : NULL;
                        std::vector<CcExpert> t1 = tab, t2 = tab;
                        std::vector<CcFetch> f1, f2;
                        for (int k = 0; k < 2 && base; k++) for (int r = 0; r < 2; r++) {
                            int at = r ? 1 - k : k;   /* round r: expert k at place k, then at the other */
                            CcExpert &xp = (r ? t2 : t1)[2 + (k ? 5 : 1)];
                            std::vector<CcFetch> &f = r ? f2 : f1;
                            const char *g = base + at * st;
                            xp.g = g; xp.u = g + mb; xp.d = g + 2 * mb;
                            xp.gs = (const float *)(g + wa); xp.us = xp.gs + nsc; xp.ds = xp.gs + 2 * nsc; xp.planar = 1;
                            for (int j = 0; j < 3; j++) {
                                f.push_back({hwv[k][j], at * st + j * mb, mb});
                                f.push_back({hsv[k][j], at * st + wa + j * nsc * 4, nsc * 4});
                            }
                        }
                        CcBuf *t1b = T->buf(t1.size() * sizeof(CcExpert), CC_DEV), *t2b = T->buf(t2.size() * sizeof(CcExpert), CC_DEV);
                        CcBuf *y3b = T->buf((size_t)S2 * D * 4, CC_DEV);
                        int oks = base && t1b && t2b && y3b && T->begin() && T->write(t1b, 0, t1.data(), t1.size() * sizeof(CcExpert)) &&
                                  T->write(t2b, 0, t2.data(), t2.size() * sizeof(CcExpert)) && T->submit(1);
                        oks = oks && T->begin() && T->fetch(sg, f1.data(), (int)f1.size()) && T->fetch_wait() &&
                              T->moe(t1b, x2b, i2b, w2b, h2b, y2b, NULL, &m3) && T->fetch_done() &&
                              T->fetch(sg, f2.data(), (int)f2.size()) && T->fetch_wait() &&
                              T->moe(t2b, x2b, i2b, w2b, h2b, y3b, NULL, &m3) && T->fetch_done() && T->submit(1);
                        double e1 = oks ? maxdiff(down(y2b, (size_t)S2 * D).data(), rf.data(), (size_t)S2 * D) / sc2 : 1;
                        double e2 = oks ? maxdiff(down(y3b, (size_t)S2 * D).data(), rf.data(), (size_t)S2 * D) / sc2 : 1;
                        if (e1 >= 2e-3 || e2 >= 2e-3) printf("  (prompt rows, staged experts: relative errors %.3g, %.3g)\n", e1, e2);
                        ck(oks && e1 < 2e-3 && e2 < 2e-3, "forty rows, two experts staged into VRAM from host memory, twice in swapped places");
                        CcFetch bad = {hwv[0][0], 2 * st - 8, 16};
                        ck(!sg || !T->fetch(sg, &bad, 1), "a fetch past its buffer refused");
                        T->free(sg); T->free(t1b); T->free(t2b); T->free(y3b);
                    }
                    for (void *h : hosts) sycl::free(h, coli_sycl_queue(g_dev));
                    T->free(tpb);
                }
                T->free(x2b); T->free(i2b); T->free(w2b); T->free(h2b); T->free(y2b); T->free(hy2b);
            }
            for (int e = 0; e < E; e++) { coli_cuda_tensor_free(tg[e]); coli_cuda_tensor_free(tu[e]); coli_cuda_tensor_free(td[e]); }
            T->free(tb); T->free(pb); T->free(xb); T->free(ib); T->free(wb); T->free(hb); T->free(yb); T->free(hyb);
        }
    }
    printf("signal, wait_host\n");
    {
        /* a hybrid step's shape: the device signals, the host answers in an UP buffer and
         * its word, the device's next op reads the answer; eight rounds in one frame */
        /* (the answer copied UP -> DEV: a copy between two host buffers is the runtime's on the
         * host, after the stream's earlier work, which here waits for the host) */
        CcBuf *df = T->buf(64, CC_DOWN), *hf = T->buf(64, CC_UP), *ans = T->buf(64 * 4, CC_UP), *dst = T->buf(8 * 64 * 4, CC_DEV);
        volatile unsigned *dw = (volatile unsigned *)T->ptr(df), *hw = (volatile unsigned *)T->ptr(hf);
        float *aw = (float *)T->ptr(ans);
        int ok = df && hf && ans && dst && T->begin();
        for (int r = 0; r < 8 && ok; r++) ok = T->signal(df, 3, 100 + r) && T->wait_host(hf, 5, 100 + r) && T->copy(dst, (size_t)r * 64, ans, 0, 64);
        ok = ok && T->submit(0);
        std::thread host([&] {
            for (int r = 0; r < 8; r++) {
                while (dw[3] < (unsigned)(100 + r)) { }
                for (int i = 0; i < 64; i++) aw[i] = (float)(r * 1000 + i);
                std::atomic_thread_fence(std::memory_order_seq_cst);
                hw[5] = 100 + r;
                std::atomic_thread_fence(std::memory_order_seq_cst);
                /* the copy has read the answer once the next round's signal is seen (or the frame ends) */
                if (r < 7) while (dw[3] < (unsigned)(101 + r)) { }
            }
        });
        ok = ok && T->finish();
        host.join();
        std::vector<float> gotv = down(dst, 8 * 64); const float *got = gotv.data(); int same = 1;
        for (int r = 0; r < 8; r++) for (int i = 0; i < 64; i++) same &= got[r * 64 + i] == (float)(r * 1000 + i);
        ck(ok && same, "eight rounds: each answer read after its word, before the next");
        T->free(df); T->free(hf); T->free(ans); T->free(dst);
    }
    printf("pick\n");
    {
        const int V = 248320;
        std::vector<float> lg = rnd((size_t)V, 91, 10.f);
        lg[173421] = 22.f;   /* a realistic peak above the bulk */
        int best = 0;
        for (int i = 1; i < V; i++) if (lg[i] > lg[best]) best = i;
        double z = 0.0; float mx = lg[best];
        for (int i = 0; i < V; i++) if (lg[i] - mx > -20.f) z += expf(lg[i] - mx);
        float p_ref = z > 0.0 ? (float)(1.0 / z) : 0.f;

        CcBuf *lb = up(lg), *pb = T->buf(8, CC_DOWN), *small = T->buf(4, CC_DOWN);
        int ok = T->pick && T->begin() && T->pick(lb, pb, V) && T->submit(1);
        const void *pk = T->ptr(pb);
        int got_id = pk ? ((const int *)pk)[0] : -1;
        float got_p = pk ? ((const float *)pk)[1] : 0.f;
        ck(ok && got_id == best && fabsf(got_p - p_ref) <= 1e-5f * p_ref,
           "248320 logits: argmax id and softmax probability (within 20 of max) into an 8-byte DOWN buffer");

        /* equal maxima across warps, within a warp and on the same thread: the lowest id wins */
        int ties[3][2] = {{137, 190000}, {137, 138}, {137, 137 + 256 * 50}};
        int tie_ok = 1;
        for (int c = 0; c < 3; c++) {
            std::vector<float> lt = lg;
            lt[ties[c][0]] = 35.f;
            lt[ties[c][1]] = 35.f;
            double zt = 0.0;
            for (int i = 0; i < V; i++) if (lt[i] - 35.f > -20.f) zt += expf(lt[i] - 35.f);
            float pt = (float)(1.0 / zt);
            int okc = T->begin() && T->write(lb, 0, lt.data(), (size_t)V * 4) && T->pick(lb, pb, V) && T->submit(1);
            int idc = ((const int *)pk)[0];
            float pc = ((const float *)pk)[1];
            tie_ok &= okc && idc == ties[c][0] && fabsf(pc - pt) <= 1e-5f * pt;
        }
        ck(tie_ok, "equal maxima (across warps, within a warp, same thread): lowest index wins");
        ck(!T->pick(lb, small, V) && !T->pick(lb, pb, V + 1), "pick past its buffers refused");
        T->free(lb); T->free(pb); T->free(small);
    }
    printf("pick_rows\n");
    {
        const int V = 248320, R = 4, off = 3;   /* rows from float off: a buffer's rows past the first */
        std::vector<float> lg = rnd((size_t)(off + R * V), 93, 10.f);
        lg[off + 173421] = 22.f; lg[off + V + 9] = 30.f; lg[off + 2 * V + 248319] = 25.f;
        lg[off + 3 * V + 5000] = 28.f; lg[off + 3 * V + 4000] = 28.f;   /* a tie: the lower place */
        int ids_ref[R]; float p_ref[R];
        for (int r = 0; r < R; r++) {
            const float *x = lg.data() + off + (size_t)r * V;
            int best = 0;
            for (int i = 1; i < V; i++) if (x[i] > x[best]) best = i;
            double z = 0.0;
            for (int i = 0; i < V; i++) if (x[i] - x[best] > -20.f) z += expf(x[i] - x[best]);
            ids_ref[r] = best; p_ref[r] = (float)(1.0 / z);
        }
        CcBuf *lb = up(lg), *pb = T->buf(8 * R, CC_DOWN), *small = T->buf(8 * R - 8, CC_DOWN);
        int ok = T->size >= offsetof(ColiCudaChainOps, pick_rows) + sizeof T->pick_rows && T->pick_rows &&
                 T->begin() && T->pick_rows(lb, off, pb, V, R) && T->submit(1);
        const void *pk = T->ptr(pb);
        int same = ok && pk;
        for (int r = 0; same && r < R; r++)
            same = ((const int *)pk)[2 * r] == ids_ref[r] && fabsf(((const float *)pk)[2 * r + 1] - p_ref[r]) <= 1e-5f * p_ref[r];
        ck(same, "four rows of 248320 logits from an offset: each row's argmax (the lower place on a tie) and probability, pick's");
        ck(ok && ids_ref[3] == 4000, "the tie row's reference is the lower place");
        ck(ok && !T->pick_rows(lb, off, small, V, R) && !T->pick_rows(lb, off + 1, pb, V, R), "pick_rows past its buffers refused");
        T->free(lb); T->free(pb); T->free(small);
    }
    printf("matmul_rows\n");
    {
        for (int c = 0; c < 3; c++) {
            int fmt = c == 0 ? 0 : 1, I = c == 2 ? 518 : 512, O = 1024, n = 137;
            std::vector<float> dq, x = rnd((size_t)I, 120 + c), want((size_t)n, 0.f);
            ColiCudaTensor *t = NULL;
            if (fmt == 0) {
                dq = rnd((size_t)O * I, 130 + c);
                std::vector<float> ones((size_t)O, 1.f);
                coli_cuda_tensor_upload(&t, dq.data(), ones.data(), 0, I, O, g_dev);
            } else {
                t = moe_mat(1, 0, I, O, 140 + c, dq);
            }
            std::vector<int> ids((size_t)n);
            for (int j = 0; j < n; j++) {
                ids[j] = (j * 7 + 19) % O;
                double s = 0;
                for (int i = 0; i < I; i++) s += (double)x[i] * dq[(size_t)ids[j] * I + i];
                want[j] = (float)s;
            }
            CcBuf *ib = T->buf((size_t)n * 4, CC_DEV), *xb = up(x), *yb = T->buf((size_t)n * 4, CC_DEV);
            int ok = T->matmul_rows && t && ib && xb && yb && T->begin() && T->write(ib, 0, ids.data(), (size_t)n * 4) &&
                     T->matmul_rows(t, ib, xb, yb, n) && T->submit(1);
            std::vector<float> got = down(yb, (size_t)n);
            ck(ok && maxdiff(got.data(), want.data(), (size_t)n) < 2e-4,
               c == 0 ? "matmul_rows fmt 0 (f32)" : c == 1 ? "matmul_rows fmt 1 (i8, I % 16 == 0)" : "matmul_rows fmt 1 (i8, I % 16 != 0)");
            if (c == 0) ck(!T->matmul_rows(t, ib, xb, yb, n + 1), "matmul_rows past its buffer refused");
            if (t) coli_cuda_tensor_free(t);
            T->free(ib); T->free(xb); T->free(yb);
        }
    }
    {   /* a GGUF's blocks (Q8_0, fmt 108): each listed row the bits the whole GEMV gives it, an id past
         * the matrix -inf (an MTP head's draft list scored from a block lm_head) */
        const int I = 2560, O = 3000, n = 211, rb = I / 32 * 34;
        std::vector<uint8_t> w8((size_t)O * rb);
        unsigned sq = 7;
        static const uint16_t scl[3] = {0x2400, 0x2800, 0x2C00};   /* 1/64, 1/32, 1/16 as float16 */
        for (int o = 0; o < O; o++)
            for (int b = 0; b < I / 32; b++) {
                uint8_t *blk = w8.data() + (size_t)o * rb + (size_t)b * 34;
                memcpy(blk, &scl[(o + b) % 3], 2);
                for (int k = 0; k < 32; k++) blk[2 + k] = (uint8_t)(int8_t)(frand(&sq) * 120);
            }
        std::vector<float> x = rnd((size_t)I, 150);
        std::vector<int> ids((size_t)n);
        for (int j = 0; j < n; j++) ids[j] = j == 17 ? O + 5 : (j * 13 + 3) % O;
        ColiCudaTensor *t = NULL;
        CcBuf *ib = T->buf((size_t)n * 4, CC_DEV), *xb = up(x), *yb = T->buf((size_t)n * 4, CC_DEV), *fb = T->buf((size_t)O * 4, CC_DEV);
        int ok = T->matmul_rows && coli_cuda_tensor_upload(&t, w8.data(), NULL, 108, I, O, g_dev) && ib && xb && yb && fb && T->begin() &&
                 T->write(ib, 0, ids.data(), (size_t)n * 4) && T->matmul_rows(t, ib, xb, yb, n) && T->matmul(t, xb, 0, fb, 0, 1) && T->submit(1);
        std::vector<float> got = down(yb, (size_t)n), full = down(fb, (size_t)O);
        int same = ok;
        for (int j = 0; j < n && same; j++) {
            const float wnt = ids[j] < O ? full[ids[j]] : -INFINITY;
            same = !memcmp(&got[j], &wnt, 4);
        }
        ck(same, "matmul_rows fmt 108 (Q8_0 blocks): each listed row the whole GEMV's bits, an id past the matrix -inf");
        if (t) coli_cuda_tensor_free(t);
        T->free(ib); T->free(xb); T->free(yb); T->free(fb);
    }
    printf("the decision engines' encoders\n");
    if (!T->enc_rel || T->size < offsetof(ColiCudaChainOps, enc_rel) + sizeof T->enc_rel) ck(0, "the table has the encoder ops");
    else {
        {   /* LayerNorm: alone, after an add, the sum written back, into the added rows (an engine's x += h; h = LN(x)) */
            const int R = 5;
            for (int D : {1000, 4096}) for (int f = 0; f < 4; f++) {
                int flags = f == 0 ? 0 : f == 1 ? CC_ENC_ADD : CC_ENC_ADD | CC_ENC_SUM, into_b = f == 3, row = D + 3;
                std::vector<float> a = rnd((size_t)R * row, 900 + D + f, 3.f), b = rnd((size_t)R * row, 910 + D + f), prm = rnd((size_t)3 * D, 920 + f);
                for (auto &v : a) v += 0.7f;   /* a mean away from zero */
                std::vector<float> y((size_t)R * row, 0.f), ra = a, rb = b, ry = y;
                CcEncNorm np = {0, R, D, 0, row, 0, row, 0, row, 0, f == 2 ? -1 : D, flags, 1e-5f};
                CcBuf *ab = up(a), *bb = up(b), *pb = up(prm), *yb = into_b ? bb : up(y);
                int ok = T->begin() && T->enc_norm(ab, flags ? bb : NULL, pb, yb, &np) && T->submit(1);
                ref_enc_norm(ra.data(), rb.data(), prm.data(), into_b ? rb.data() : ry.data(), &np);
                std::vector<float> gy = down(yb, (size_t)R * row), ga = down(ab, (size_t)R * row);
                double dy = maxdiff(gy.data(), (into_b ? rb : ry).data(), gy.size()), da = maxdiff(ga.data(), ra.data(), ga.size());
                char what[160];
                snprintf(what, sizeof what, "enc_norm D %d%s%s%s (|dy| %.1e)", D, flags & CC_ENC_ADD ? ", + b" : "", flags & CC_ENC_SUM ? ", the sum kept" : "",
                         into_b ? ", into b" : f == 2 ? ", no bias" : "", dy);
                ck(ok && dy < 2e-5 && da == 0, what);
                T->free(ab); T->free(bb); T->free(pb); if (!into_b) T->free(yb);
            }
        }
        {   /* bias and activation in place, a row wider than its outputs; GeGLU */
            const int R = 7, O = 300, row = 320;
            for (int act = 0; act < 4; act++) for (int nb = 0; nb < 2; nb++) {
                std::vector<float> y = rnd((size_t)R * row, 930 + act, 3.f), prm = rnd(O + 5, 940), want = y;
                CcEncBias bp = {1, R * O, O, 0, row, nb ? -1 : 5, act};
                for (int r = 0; r < R; r++) for (int o = 0; o < O; o++) { float &v = want[(size_t)r * row + o]; v = enc_act(nb ? v : v + prm[5 + o], act); }
                CcBuf *yb = up(y), *pb = up(prm);
                int ok = T->begin() && T->enc_bias(yb, pb, &bp) && T->submit(1);
                char what[96]; snprintf(what, sizeof what, "enc_bias act %d%s", act, nb ? ", no bias" : "");
                ck(ok && maxdiff(down(yb, y.size()).data(), want.data(), y.size()) < 1e-6, what);
                T->free(yb); T->free(pb);
            }
            const int I = 200;
            for (int act = 1; act < 3; act++) for (int nb = 0; nb < 2; nb++) {
                std::vector<float> a = rnd((size_t)R * 2 * I, 950 + act, 3.f), prm = rnd(2 * I + 3, 960), y((size_t)R * (I + 8), 0.f), want = y;
                CcEncGeglu gp = {2, R * I, I, 0, 2 * I, 0, I + 8, nb ? -1 : 3, act};
                for (int r = 0; r < R; r++) for (int j = 0; j < I; j++) {
                    float u = a[(size_t)r * 2 * I + j], g = a[(size_t)r * 2 * I + I + j];
                    if (!nb) { u += prm[3 + j]; g += prm[3 + I + j]; }
                    want[(size_t)r * (I + 8) + j] = enc_act(u, act) * g;
                }
                CcBuf *ab = up(a), *pb = up(prm), *yb = up(y);
                int ok = T->begin() && T->enc_geglu(ab, pb, yb, &gp) && T->submit(1);
                char what[96]; snprintf(what, sizeof what, "enc_geglu act %d%s", act, nb ? ", no bias" : "");
                ck(ok && maxdiff(down(yb, y.size()).data(), want.data(), y.size()) < 1e-6, what);
                T->free(ab); T->free(pb); T->free(yb);
            }
        }
        {   /* RoPE on the query and key heads of a fused row, a table row per position; a type row per row */
            const int R = 9, H = 3, hd = 16, half = hd / 2, row = 3 * H * hd, P = 12;
            std::vector<float> x = rnd((size_t)R * row, 970), tab((size_t)2 * P * half), want = x;
            std::vector<int> pos((size_t)R);
            for (int i = 0; i < P * half; i++) { float ang = (float)(i / half) * powf(10000.f, -(float)(i % half) / half); tab[i] = cosf(ang); tab[P * half + i] = sinf(ang); }
            for (int r = 0; r < R; r++) pos[r] = (r * 5 + 2) % P;
            for (int r = 0; r < R; r++) for (int part = 0; part < 2; part++) for (int h = 0; h < H; h++) for (int k = 0; k < half; k++) {
                float *v = &want[(size_t)r * row + part * H * hd + h * hd];
                float c = tab[pos[r] * half + k], sn = tab[P * half + pos[r] * half + k], x0 = v[k], x1 = v[k + half];
                v[k] = x0 * c - x1 * sn; v[k + half] = x1 * c + x0 * sn;
            }
            CcEncRope rp = {3, R * H * hd, H, hd, 0, row, H * hd, 0, P * half};
            CcBuf *xb = up(x), *tb = up(tab), *pb = upi(pos);
            int ok = T->begin() && T->enc_rope(xb, tb, pb, &rp) && T->submit(1);
            ck(ok && maxdiff(down(xb, x.size()).data(), want.data(), x.size()) < 1e-6, "enc_rope: the query and key heads, the table's rows");   /* a fused multiply-add's rounding */
            T->free(xb); T->free(tb); T->free(pb);
            const int D = 40;
            std::vector<float> y = rnd((size_t)R * D, 975), t2 = rnd(3 * D + 7, 976), want2 = y;
            std::vector<int> idx((size_t)R);
            for (int r = 0; r < R; r++) { idx[r] = r % 3; for (int d = 0; d < D; d++) want2[(size_t)r * D + d] += t2[7 + idx[r] * D + d]; }
            CcEncAddRow ap = {4, R * D, D, 0, D, 7};
            CcBuf *yb = up(y), *tb2 = up(t2), *ib = upi(idx);
            ok = T->begin() && T->enc_addrow(yb, tb2, ib, &ap) && T->submit(1);
            ck(ok && maxdiff(down(yb, y.size()).data(), want2.data(), y.size()) == 0, "enc_addrow: a type row per row");
            T->free(yb); T->free(tb2); T->free(ib);
        }
        {   /* the attention: sequences side by side, each row over its sequence or a window in it,
             * with DeBERTa's two relative terms and without; head dims under, at and past a warp's tile */
            const int lens[3] = {37, 1, 50}, S = 88, nr = 21, r0 = 3, maxp = 60;
            for (int hd : {40, 64, 128}) for (int mode = 0; mode < 3; mode++) {
                const int H = hd == 128 ? 2 : 3, row = 3 * H * hd;
                std::vector<float> qkv = rnd((size_t)S * row, 980 + hd + mode, 1.5f), o((size_t)S * H * hd, 0.f), want = o;
                std::vector<float> c2p = rnd((size_t)H * S * nr + 4, 990 + mode, 2.f), p2c = rnd((size_t)H * S * nr + 6, 995 + mode, 2.f);
                std::vector<int> rng((size_t)2 * S), pos((size_t)S), ridx((size_t)2 * maxp + 1);
                for (int s = 0, r = 0; s < 3; s++)
                    for (int i = 0; i < lens[s]; i++, r++) {
                        int st = r - i, lo = mode == 1 ? (i - 6 < 0 ? 0 : i - 6) : 0, hi = mode == 1 ? (i + 6 > lens[s] - 1 ? lens[s] - 1 : i + 6) : lens[s] - 1;
                        rng[2 * r] = st + lo; rng[2 * r + 1] = st + hi; pos[r] = i;
                    }
                rng[2 * 40] = 5; rng[2 * 40 + 1] = 4;   /* an empty range: zeros */
                for (int k = 0; k <= 2 * maxp; k++) ridx[k] = r0 + (k * 7 + 3) % nr;
                int flags = mode == 2 ? CC_ENC_C2P | CC_ENC_P2C : 0;
                CcEncAttn ap = {5, S, H, hd, 0, H * hd, 2 * H * hd, row, 0, H * hd, flags, 4, 6, nr, maxp, 1.f / sqrtf((float)hd), r0};
                ref_enc_attn(qkv.data(), c2p.data(), p2c.data(), want.data(), rng.data(), ridx.data(), pos.data(), &ap);
                CcBuf *qb = up(qkv), *ob = up(o), *cb = up(c2p), *pb = up(p2c), *rb = upi(rng), *xb = upi(ridx), *sb = upi(pos);
                int ok = T->begin() && T->enc_attn(qb, flags ? cb : NULL, flags ? pb : NULL, ob, rb, flags ? xb : NULL, flags ? sb : NULL, &ap) && T->submit(1);
                double d = maxdiff(down(ob, o.size()).data(), want.data(), o.size());
                char what[128];
                snprintf(what, sizeof what, "enc_attn hd %d, %s (|d| %.1e)", hd, mode == 0 ? "the sequences" : mode == 1 ? "windows" : "c2p and p2c", d);
                ck(ok && d < 2e-5, what);
                T->free(qb); T->free(ob); T->free(cb); T->free(pb); T->free(rb); T->free(xb); T->free(sb);
            }
            CcEncAttn bad = {5, S, 2, 129, 0, 0, 0, 3 * 129, 0, 129, 0, 0, 0, 0, 0, 1.f, 0};
            CcBuf *qb = T->buf(4, CC_DEV);
            ck(!T->enc_attn(qb, NULL, NULL, qb, qb, NULL, NULL, &bad), "enc_attn past 128 a head refused");
            T->free(qb);
        }
        {   /* the relative products: rows by buckets of every head, buckets past a block's 64 */
            const int S = 40, H = 3, hd = 48, nr = 70, r0 = 5, xrow = H * hd + 4, prow = H * hd + 2;
            std::vector<float> x = rnd((size_t)S * xrow, 1001), pt = rnd((size_t)(r0 + nr + 1) * prow, 1002), y((size_t)H * S * nr + 3, 0.f), want = y;
            CcEncRel rp = {6, S, H, hd, 4, xrow, 2, prow, 3, nr, r0};
            ref_enc_rel(x.data(), pt.data(), want.data(), &rp);
            CcBuf *xb = up(x), *pb = up(pt), *yb = up(y);
            int ok = T->begin() && T->enc_rel(xb, pb, yb, &rp) && T->submit(1);
            double d = maxdiff(down(yb, y.size()).data(), want.data(), y.size());
            char what[96]; snprintf(what, sizeof what, "enc_rel (|d| %.1e)", d);
            ck(ok && d < 2e-5, what);
            T->free(xb); T->free(pb); T->free(yb);
        }
    }
    printf("hyper-connection mix\n");
    int hcmix_taken = 0;   /* the fused passes the backend took (its matmuls counted below) */
    {   /* a hyper-connection read's up projection and mix in one pass (matmul_hcmix): Q8_0 rows of 320 inputs, four
         * streams of 256, 300 rows (tiles of 256 rows) and 100 (of 128); against matmul then CC_EW_HC_MIX, to the bit */
        const int I = 320, H = 256, C = 4, O = C * H;
        unsigned sd = 31;
        std::vector<uint8_t> q8((size_t)O * (I / 32) * 34);
        for (int o = 0; o < O; o++)
            for (int b = 0; b < I / 32; b++) {
                uint8_t *blk = q8.data() + ((size_t)o * (I / 32) + b) * 34;
                const uint16_t dh = (uint16_t)(0x2000u + (unsigned)(frand(&sd) * 2048.f + 2048.f));   /* a small positive half */
                memcpy(blk, &dh, 2);
                for (int k = 0; k < 32; k++) blk[2 + k] = (uint8_t)(int8_t)(frand(&sd) * 120.f);
            }
        ColiCudaTensor *tq = NULL;
        ck(coli_cuda_tensor_upload(&tq, q8.data(), NULL, 108, I, O, g_dev), "a Q8_0 tensor of 320 inputs and 4 x 256 outputs resident");
        for (int S : {300, 100}) {
            std::vector<float> x = rnd((size_t)S * I, 41), hn = rnd((size_t)S * O, 42);
            CcBuf *xb = up(x), *hb = up(hn), *mb = T->buf((size_t)S * O * 4, CC_DEV), *ya = T->buf((size_t)S * H * 4, CC_DEV), *yb = T->buf((size_t)S * H * 4, CC_DEV);
            CcEw mx = {CC_EW_HC_MIX, S * H, H, C, 0, 1, 0, 0, 0, 0, 0, (float)C};
            int ok = T->begin() && T->matmul(tq, xb, 0, mb, 0, S) && T->ew(ya, mb, hb, NULL, NULL, &mx) && T->submit(1);
            int r = ok && T->matmul_hcmix ? T->matmul_hcmix(tq, xb, 0, hb, yb, S, H, C, (float)C) : 0;
            char what[128]; snprintf(what, sizeof what, "%d rows: the up projection and the mix in one pass, those of the two ops to the bit", S);
            if (ok && r == -1) {
                /* the contract's "not taken" (cuda_chain.h): this backend has no fused kernel (the CUDA
                 * hcm_mix_k was f16 mma.sync over GGUF blocks, MIGRATION.md section 10) and the engine runs
                 * the matmul and CC_EW_HC_MIX apart, as this frame just did; nothing to compare, said so */
                printf("  skip %s: matmul_hcmix not taken on this backend (-1)\n", what);
                ok = T->submit(1);
                if (!ok) ck(0, what);
            } else {
                ok = ok && r == 1 && T->submit(1);
                std::vector<float> a = down(ya, (size_t)S * H), b = down(yb, (size_t)S * H);
                ck(ok && !memcmp(a.data(), b.data(), a.size() * 4), what);
                hcmix_taken += ok;
            }
            T->free(xb); T->free(hb); T->free(mb); T->free(ya); T->free(yb);
        }
        coli_cuda_tensor_free(tq);
    }
    printf("int8 GEMV subgroup boundaries\n");
    /* Small matrices first: exercise both sides of PVC's SG16 shape rule and NR staging.
     * I=2048 also crosses W4's minimum matrix size and exercises split-K at both widths.
     * The one-row global-x variants must retain every bit of the staged verify rows. */
    for (int I : {64, 2048}) for (int O : {1023, 1024, 2048, 2049}) {
        const int S = 4;
        unsigned seed = 73;
        std::vector<int8_t> w((size_t)O * I);
        for (auto &v : w) v = (int8_t)(frand(&seed) * 60);
        std::vector<float> scales(O, 0.01f), x = rnd((size_t)S * I, 74), ref((size_t)S * O);
        for (int r = 0; r < S; r++) for (int o = 0; o < O; o++) {
            double a = 0;
            for (int i = 0; i < I; i++) a += (double)x[(size_t)r * I + i] * w[(size_t)o * I + i];
            ref[(size_t)r * O + o] = (float)(a * scales[o]);
        }
        ColiCudaTensor *tensor = nullptr;
        CcBuf *xb = up(x), *yb = T->buf((size_t)S * O * 4, CC_DEV), *one = T->buf((size_t)O * 4, CC_DEV);
        int ok = xb && yb && one && coli_cuda_tensor_upload(&tensor, w.data(), scales.data(), 1, I, O, g_dev)
                 && T->begin() && T->matmul(tensor, xb, 0, yb, 0, S) && T->submit(1);
        std::vector<float> all = ok ? down(yb, (size_t)S * O) : std::vector<float>();
        char what[128];
        snprintf(what, sizeof what, "O=%d I=%d: four int8 rows agree with the CPU", O, I);
        ck(ok && maxdiff(all.data(), ref.data(), ref.size()) < (I == 64 ? 1e-5 : 1e-4), what);
        int same = ok;
        for (int r = 0; r < S && same; r++) {
            same = T->begin() && T->matmul(tensor, xb, (size_t)r * I, one, 0, 1) && T->submit(1);
            if (same) same = !memcmp(down(one, O).data(), all.data() + (size_t)r * O, (size_t)O * 4);
        }
        snprintf(what, sizeof what, "O=%d I=%d: verify rows equal one-row calls bit for bit", O, I);
        ck(same, what);
        if (tensor) coli_cuda_tensor_free(tensor);
        T->free(xb); T->free(yb); T->free(one);
    }
    printf("lost\n");
    {
        CcStats st; T->stats(&st);
        /* Original 48 plus hcmix passes taken, plus 8 shapes x (1 verify + 4 decode) above. */
        ck(st.frames > 10 && st.ops > 30 && st.matmuls == (unsigned long long)(88 + hcmix_taken), "the counters saw the frames, ops and matmuls");
        setenv("COLI_GPU_FAIL_AFTER", "0", 1);
        std::vector<float> w = rnd(64, 81), ones(8, 1.f);
        ColiCudaTensor *t = NULL;
        coli_cuda_tensor_upload(&t, w.data(), ones.data(), 0, 8, 8, g_dev);
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
