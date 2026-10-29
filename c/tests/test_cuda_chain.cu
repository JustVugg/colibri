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
#include "../backend_gpu_compat.h"   /* the runtime's names under HIP too (the planar blocks' host memory) */
#define COLI_CUDA_CHAIN_NO_WRAPPERS
#include "../cuda_chain.h"
#define KV16_INTEGER   /* kv16.h's integer rounding: the reference of ew's CC_EW_F16 */
#include "../kv16.h"
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
 * page-locked host memory the device reads over the bus: its device address in *dev, its scales
 * likewise. The host pointers to free with cudaFreeHost. */
static int moe_planar(int gs, int I, int O, unsigned seed, uint8_t **hw, float **hs, const void **dw, const float **ds) {
    unsigned s = seed; int ng = I / gs;
    std::vector<float> sc((size_t)O * ng);
    for (auto &v : sc) v = 0.05f + 0.02f * frand(&s);
    if (cudaHostAlloc((void **)hw, (size_t)O * I / 2, cudaHostAllocMapped) != cudaSuccess ||
        cudaHostAlloc((void **)hs, sc.size() * 4, cudaHostAllocMapped) != cudaSuccess) return 0;
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
    void *a = NULL, *b = NULL;
    if (cudaHostGetDevicePointer(&a, *hw, 0) != cudaSuccess || cudaHostGetDevicePointer(&b, *hs, 0) != cudaSuccess) return 0;
    *dw = a; *ds = (const float *)b;
    return 1;
}
static ColiCudaTensor *moe_mat(int fmt, int gs, int I, int O, unsigned seed, std::vector<float> &dq) {
    unsigned s = seed; dq.assign((size_t)O * I, 0.f);
    ColiCudaTensor *t = NULL; int ok = 0;
    if (fmt == 1) {
        std::vector<int8_t> q((size_t)O * I); std::vector<float> sc(O);
        for (int o = 0; o < O; o++) { sc[o] = 0.01f + 0.005f * frand(&s); for (int i = 0; i < I; i++) { q[(size_t)o * I + i] = (int8_t)(frand(&s) * 100); dq[(size_t)o * I + i] = q[(size_t)o * I + i] * sc[o]; } }
        ok = coli_cuda_tensor_upload(&t, q.data(), sc.data(), 1, I, O, 0);
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
        ok = fmt == 4 ? coli_cuda_tensor_upload_g(&t, q.data(), sc.data(), 4, I, O, 0, gs) : coli_cuda_tensor_upload(&t, q.data(), sc.data(), 2, I, O, 0);
    } else {
        int nbI = (I + 127) / 128, nbO = (O + 127) / 128;
        std::vector<uint8_t> q((size_t)O * I); std::vector<float> sc((size_t)nbO * nbI);
        for (auto &v : sc) v = 0.004f + 0.002f * frand(&s);
        for (int o = 0; o < O; o++) for (int i = 0; i < I; i++) {
            int b; do { b = (int)((frand(&s) * 0.5f + 0.5f) * 255.f) & 255; } while ((b & 0x7f) == 0x7f || ((b >> 3) & 15) > 12);
            q[(size_t)o * I + i] = (uint8_t)b; dq[(size_t)o * I + i] = e4m3(b) * sc[(size_t)(o / 128) * nbI + i / 128];
        }
        ok = coli_cuda_tensor_upload(&t, q.data(), sc.data(), 8, I, O, 0);
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

int main(void) {
    int dev = 0;
    if (!coli_cuda_init(&dev, 1)) { printf("no CUDA device: skipped\n"); return 0; }
    T = coli_cuda_chain_ops();
    ck(T != NULL && T->size == sizeof(ColiCudaChainOps), "the backend exports the chain's table");
    if (!T) return 1;
    ck(T->init(0), "a context on device 0");
    ck(T->ready() && !T->lost(), "ready, not lost");
    const char *be = getenv("COLI_CUDA_CHAIN_BENCH");
    if (be && !strcmp(be, "adec")) { bench_adec(); return 0; }
    if (be && !strcmp(be, "route")) { bench_route(); return 0; }

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
        {   /* a prompt chunk's rows (more than the warp GEMV's 8): the tensor cores on sm_80+, TF32's 10-bit inputs */
            const int S2 = 70, I2 = 256, O2 = 100;
            std::vector<float> x2 = rnd((size_t)S2 * I2, 17), dq;
            for (int fmt : {1, 2, 4}) {
                ColiCudaTensor *tt = moe_mat(fmt, fmt == 4 ? 64 : 0, I2, O2, 700 + fmt, dq);
                std::vector<float> r2((size_t)S2 * O2); double mx = 0;
                for (int r = 0; r < S2; r++) for (int o = 0; o < O2; o++) { double a = 0; for (int i = 0; i < I2; i++) a += (double)x2[(size_t)r * I2 + i] * dq[(size_t)o * I2 + i]; r2[(size_t)r * O2 + o] = (float)a; mx = fmax(mx, fabs(a)); }
                CcBuf *xb2 = up(x2), *yb2 = T->buf((size_t)S2 * O2 * 4, CC_DEV);
                int ok2 = tt && T->begin() && T->matmul(tt, xb2, 0, yb2, 0, S2) && T->submit(1);
                double e2 = ok2 ? maxdiff(down(yb2, (size_t)S2 * O2).data(), r2.data(), (size_t)S2 * O2) / mx : 1;
                if (e2 >= 2e-3) printf("  (prompt matmul fmt %d: relative error %.3g)\n", fmt, e2);
                ck(ok2 && e2 < 2e-3, fmt == 1 ? "seventy rows of int8 rows (a prompt chunk)" : fmt == 2 ? "seventy rows of int4 rows" : "seventy rows of int4 with a scale per 64");
                if (tt) coli_cuda_tensor_free(tt); T->free(xb2); T->free(yb2);
            }
        }
        std::vector<float> w = rnd((size_t)I * O, 11), ones(O, 1.f), ref0(O);
        for (int o = 0; o < O; o++) { float a = 0; for (int i = 0; i < I; i++) a += x[7 + i] * w[(size_t)o * I + i]; ref0[o] = a; }
        ColiCudaTensor *t0 = NULL;
        ck(coli_cuda_tensor_upload(&t0, w.data(), ones.data(), 0, I, O, 0), "an f32 tensor resident");
        ck(T->begin() && T->matmul(t0, xb, 7, yb, 0, 1) && T->submit(1) && maxdiff(down(yb, O).data(), ref0.data(), O) < 1e-4, "f32 rows: the CPU's dot");
        {   /* a prompt chunk's rows of an f32 matrix (the tiled GEMM): widths that are no tile's multiple */
            const int S3 = 70, I3 = 200, O3 = 100;
            std::vector<float> x3 = rnd((size_t)S3 * I3, 12), w3 = rnd((size_t)I3 * O3, 13), one3(O3, 1.f), r3((size_t)S3 * O3);
            for (int s = 0; s < S3; s++) for (int o = 0; o < O3; o++) {
                double a = 0; for (int i = 0; i < I3; i++) a += (double)x3[(size_t)s * I3 + i] * w3[(size_t)o * I3 + i]; r3[(size_t)s * O3 + o] = (float)a;
            }
            ColiCudaTensor *t3 = NULL;
            CcBuf *xb3 = T->buf((size_t)S3 * I3 * 4, CC_DEV), *yb3 = T->buf((size_t)S3 * O3 * 4, CC_DOWN);
            int ok3 = xb3 && yb3 && coli_cuda_tensor_upload(&t3, w3.data(), one3.data(), 0, I3, O3, 0) && T->begin() &&
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
            int ok4 = xb4 && yb4 && yb1 && coli_cuda_tensor_upload(&t4, w4.data(), one4.data(), 0, I4, O4, 0) && T->begin() &&
                      T->write(xb4, 0, x4.data(), x4.size() * 4) && T->matmul(t4, xb4, 0, yb4, 0, S4) && T->submit(1);
            ck(ok4 && maxdiff((const float *)T->ptr(yb4), r4.data(), r4.size()) < 1e-3, "f32 rows, 4 outputs of 10240 (the GEMV for few outputs)");
            int same = ok4;
            for (int s = 0; s < S4 && same; s++)
                same = T->begin() && T->matmul(t4, xb4, (size_t)s * I4, yb1, 0, 1) && T->submit(1) &&
                       !memcmp(T->ptr(yb1), (const float *)T->ptr(yb4) + (size_t)s * O4, (size_t)O4 * 4);
            ck(same, "and each of its three rows the one-row call's bits");
            coli_cuda_tensor_free(t4); T->free(xb4); T->free(yb4); T->free(yb1);
        }
        for (int shape = 0; shape < 3; shape++) {   /* and quant_matmul's bits (coli_cuda_matmul still runs it) */
            static const int sI[3] = {10240, 2048, 2048}, sO[3] = {4, 256, 1};
            const int I5 = sI[shape], O5 = sO[shape], S5 = 2;
            std::vector<float> x5 = rnd((size_t)S5 * I5, 16 + shape), w5 = rnd((size_t)I5 * O5, 19 + shape), one5(O5, 1.f), q5((size_t)S5 * O5);
            ColiCudaTensor *t5 = NULL, *tq = NULL;
            CcBuf *xb5 = T->buf((size_t)S5 * I5 * 4, CC_DEV), *yb5 = T->buf((size_t)S5 * O5 * 4, CC_DOWN);
            int ok5 = xb5 && yb5 && coli_cuda_tensor_upload(&t5, w5.data(), one5.data(), 0, I5, O5, 0) && T->begin() &&
                      T->write(xb5, 0, x5.data(), x5.size() * 4) && T->matmul(t5, xb5, 0, yb5, 0, S5) && T->submit(1) &&
                      coli_cuda_matmul(&tq, q5.data(), x5.data(), w5.data(), one5.data(), 0, S5, I5, O5, 0, 0);
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
    printf("dnrec\n");
    /* four rows (cc_dnrec_k) and 37 (cc_dnrec_rows_k: blocks past the first, the last part
     * full, the snapshot inside one) */
    for (int it = 0; it < 4; it++) {
        const int KD = it & 1 ? 128 : 8, S = it < 2 ? 4 : 37, snap_at = S == 4 ? 1 : 20;
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
                for (int o = 0; o < I; o++) { double g = 0, u = 0; for (int i = 0; i < D; i++) { g += xr[i] * G[e][(size_t)o * D + i]; u += xr[i] * U[e][(size_t)o * D + i]; } h[o] = (float)(g / (1 + exp(-g)) * u); }
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
            CcMoe m = {S, K, D, I, fmt, gs, 0, 3, 2, 4, D, 0, K, 0, K, 8, D, -1, E, 0};
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
                CcMoe m2 = {S2, K, D, I, fmt, gs, 0, 3, 2, 0, D, 0, K, 0, K, 0, D, -1, E, 0};
                int ok2 = T->begin() && T->write(i2b, 0, ids2.data(), ids2.size() * 4) && T->moe(tb, x2b, i2b, w2b, h2b, y2b, NULL, &m2) && T->submit(1);
                double ef = ok2 ? maxdiff(down(y2b, (size_t)S2 * D).data(), rf.data(), (size_t)S2 * D) / sc2 : 1;
                CcMoe a2 = m2, b2 = m2; a2.parts = 1; b2.parts = 2; b2.h_off = 0;
                int okp = T->begin() && T->moe(pb, x2b, i2b, w2b, h2b, y2b, hy2b, &a2) && T->moe(pb, x2b, i2b, w2b, h2b, y2b, hy2b, &b2) && T->submit(1);
                double ep = okp ? maxdiff(down(y2b, (size_t)S2 * D).data(), rp.data(), (size_t)S2 * D) / sc2 : 1;
                if (ef >= 2e-3 || ep >= 2e-3) printf("  (prompt rows fmt %d: relative errors %.3g, %.3g)\n", fmt, ef, ep);
                ck(ok2 && ef < 2e-3, "forty rows: the slots sorted by expert, their GEMMs, the sum in rank order");
                ck(okp && ep < 2e-3, "forty rows, two experts the host's: its rows plus the device's slots");
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
                    for (void *h : hosts) cudaFreeHost(h);
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
    printf("matmul_rows\n");
    {
        for (int c = 0; c < 3; c++) {
            int fmt = c == 0 ? 0 : 1, I = c == 2 ? 518 : 512, O = 1024, n = 137;
            std::vector<float> dq, x = rnd((size_t)I, 120 + c), want((size_t)n, 0.f);
            ColiCudaTensor *t = NULL;
            if (fmt == 0) {
                dq = rnd((size_t)O * I, 130 + c);
                std::vector<float> ones((size_t)O, 1.f);
                coli_cuda_tensor_upload(&t, dq.data(), ones.data(), 0, I, O, 0);
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
    printf("lost\n");
    {
        CcStats st; T->stats(&st);
        ck(st.frames > 10 && st.ops > 30 && st.matmuls == 21, "the counters saw the frames, ops and matmuls");
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
