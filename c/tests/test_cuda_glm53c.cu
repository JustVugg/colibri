/* GLM-5.3's layer ops on the CUDA chain (cuda_glm53c.cuh) against the CPU's references, the cases
 * and bounds tests/test_vk_chain.c holds the Vulkan shaders to:
 *   kda  glm53.c's KDA layer around delta_attention.h's coli_kda_step (the decay, beta and
 *        output gate as glm53.c makes them), S rows over two calls (the state and the window
 *        carried on the device between them); the block of rows kernel (CC_KDA_ROWS rows and
 *        more in one call) against calls of fewer rows, bit for bit, GLM's form and Kimi K3's;
 *   mla  the attention core against a double reference over a causal range and over row lists
 *        with a skipped entry; RoPE in both styles in place and elsewhere; LayerNorm with and
 *        without the bias; the per-head matmuls of a float matrix, rows with the gate and
 *        transposed;
 *   kpool sparse_index.h's k-pooled selection slot for slot (exact inputs, ties among equal
 *        pools), with and without the tail, below and past CC_DSA_GRID pools; the pooled keys
 *        with real gates and bias against its arithmetic;
 *   dsa  colibri.c's selection (GLM-5.2 and GLM-5.3, not pooled) position for position (exact
 *        inputs, ties at the threshold), rows below topk reading every position, DSA_FORCE's; the
 *        per-head matmul on a GGUF's Q8_0 rows against the float one on the same rows, bit for bit.
 * The kernels round the CPU's steps in its order and contract nothing into an FMA (the reference is
 * built without contraction too): with an exp correctly rounded in their place the outputs are the
 * CPU's within 5e-7. CUDA's expf, within 2 ulp, gives 2-5e-5 on the outputs once the recurrence and
 * the output norm have carried it: the outputs' bound is 1e-4, the state's 1e-5.
 * make CUDA=1 cuda-glm53c-check (the references in tests/cuda_glm53c_ref.c) */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include "../backend_cuda.h"
#include "../cuda_chain.h"
extern "C" {   /* tests/cuda_glm53c_ref.c */
int glm53c_ref_kda_scratch(int H, int KD, int VD);
int glm53c_ref_kda_step(float *out, float *state, float *window, const float *qkv, const float *conv_w, const float *gate,
                        const float *beta, int H, int KD, int VD, int K, float neps, float *scratch);
int glm53c_ref_width(int topk, int pool, int tail);
int glm53c_ref_kpool(int *out, const float *q, const float *keys, const float *gates, const float *hw, const float *ape,
                     const unsigned char *valid, int T, int IH, int ID, int pool, int topk, int tail, int q_from, int q_to);
void glmdsa_ref_select(int *sel, int *nsel, const float *q, const float *hw, const float *keys, int S, int pb, int IH, int ID,
                       int topk, int force);
}

static const ColiCudaChainOps *T;
static int fails;
static void ck(int ok, const char *what) { printf("  %s %s\n", ok ? "ok  " : "FAIL", what); if (!ok) fails++; }
static unsigned g_rng = 12345;
static unsigned rnd() { g_rng = g_rng * 1103515245u + 12345u; return (g_rng >> 8) & 0xffffff; }
static float frnd() { return ((int)(rnd() % 2001) - 1000) / 1000.0f; }
static std::vector<float> fvec(size_t n, float scale) { std::vector<float> v(n); for (auto &x : v) x = frnd() * scale; return v; }
static CcBuf *up(const void *v, size_t bytes) {
    CcBuf *b = T->buf(bytes ? bytes : 4, CC_DEV);
    if (!b || !T->begin() || !T->write(b, 0, v, bytes) || !T->submit(1)) { printf("  FAIL upload\n"); fails++; }
    return b;
}
static CcBuf *upf(const std::vector<float> &v) { return up(v.data(), v.size() * 4); }
template <class X> static std::vector<X> down(CcBuf *b, size_t n) {
    std::vector<X> v(n);
    if (!T->read(b, 0, v.data(), n * sizeof(X))) { printf("  FAIL read back\n"); fails++; }
    return v;
}
static double relerr(const float *a, const float *b, size_t n, double floor_) {
    double w = 0;
    for (size_t i = 0; i < n; i++) {
        if (std::isinf(a[i]) || std::isinf(b[i])) { if (a[i] != b[i]) return 1e9; continue; }
        double d = fabs((double)a[i] - b[i]) / (fabs((double)b[i]) > floor_ ? fabs((double)b[i]) : floor_);
        if (!(d <= w)) w = d;
    }
    return w;
}
/* the largest difference against the largest reference value: outputs that are sums of terms of
 * both signs (an average of keys, a softmax mixture), whose small values carry the rounding of
 * the large ones */
static double relmax(const float *a, const float *b, size_t n) {
    double d = 0, m = 0;
    for (size_t i = 0; i < n; i++) { const double e = fabs((double)a[i] - b[i]); if (!(e <= d)) d = e; if (fabs((double)b[i]) > m) m = fabs((double)b[i]); }
    return m > 0 ? d / m : d;
}
static int bad(const float *a, size_t n) { for (size_t i = 0; i < n; i++) if (!std::isfinite(a[i])) return 1; return 0; }
static float sigm(float v) { if (v >= 0) { float d = expf(-v); return 1.f / (1.f + d); } float g = expf(v); return g / (1.f + g); }

/* the device's inputs of one KDA layer step: q, k, v in three blocks [3][S][P] (in_part = S*P) */
struct Kda {
    int H, KD, VD, P, C, S, K;
    std::vector<float> qkv, f, b, g, taps, win, state, prm;
};
static Kda kda_make(int H, int KD, int K, int S, float xs) {
    Kda k; k.H = H; k.KD = KD; k.VD = KD; k.P = H * KD; k.C = 3 * k.P; k.S = S; k.K = K;
    k.qkv = fvec((size_t)S * k.C, xs); k.f = fvec((size_t)S * k.P, 1.f); k.b = fvec((size_t)S * H, 2.f); k.g = fvec((size_t)S * k.P, 2.f);
    k.taps = fvec((size_t)k.C * K, 0.7f); k.win = fvec((size_t)k.C * K, 1.f); k.state = fvec((size_t)H * KD * k.VD, 0.3f);
    k.prm.resize((size_t)H + k.P + k.VD);
    for (int i = 0; i < H; i++) k.prm[i] = frnd() * 0.5f;
    for (int i = 0; i < k.P; i++) k.prm[H + i] = frnd() * 0.5f;
    for (int i = 0; i < k.VD; i++) k.prm[H + k.P + i] = 0.5f + (rnd() % 100) / 100.f;
    return k;
}
/* the device's run of the S rows in calls of `per` rows; y, the state and the window back */
static int kda_run(const Kda &k, int per, int flags, std::vector<float> &y, std::vector<float> &st, std::vector<float> &win) {
    const int S = k.S, P = k.P, C = k.C;
    const float lb = -5.f, neps = 1e-6f, eps = 1e-5f;
    std::vector<float> blk((size_t)3 * S * P);
    for (int s = 0; s < S; s++) for (int part = 0; part < 3; part++)
        memcpy(&blk[(size_t)part * S * P + (size_t)s * P], &k.qkv[(size_t)s * C + (size_t)part * P], (size_t)P * 4);
    CcBuf *ib = upf(blk), *tb = upf(k.taps), *wb = upf(k.win), *sb = upf(k.state);
    CcBuf *fb = upf(k.f), *bb = upf(k.b), *gb = upf(k.g), *pb = upf(k.prm);
    CcBuf *mb = T->buf((size_t)S * C * 4, CC_DEV), *yb = T->buf((size_t)S * P * 4, CC_DEV);
    int ok = 1;
    for (int s0 = 0; s0 < S && ok; s0 += per) {
        const int n = S - s0 < per ? S - s0 : per;
        CcKdaConv cp = {n, C, k.K, P, s0 * P, P, S * P, 0, C, 0, 0};
        CcKdaRec rp = {n, k.H, k.VD, P, 0, C, s0 * P, P, s0 * k.H, k.H, s0 * P, P, s0 * P, P, 0, 0, lb, neps, eps};
        ok = T->begin() && T->kda_conv(ib, tb, wb, mb, &cp) && T->kda_rec(k.KD, mb, fb, bb, gb, pb, sb, yb, &rp, flags) && T->submit(1);
    }
    y = down<float>(yb, (size_t)S * P); st = down<float>(sb, (size_t)k.H * k.KD * k.VD); win = down<float>(wb, (size_t)C * k.K);
    CcBuf *all[] = {ib, tb, wb, sb, fb, bb, gb, pb, mb, yb};
    for (CcBuf *x : all) T->free(x);
    return ok;
}
static void test_kda(int H, int KD, int K, float xs) {
    Kda k = kda_make(H, KD, K, 5, xs);
    const int S = k.S, P = k.P, C = k.C, VD = k.VD;
    const float lb = -5.f, neps = 1e-6f, eps = 1e-5f;
    /* reference: glm53.c's kda_layer row by row */
    std::vector<float> rs = k.state, rw = k.win, ref((size_t)S * P), core(P), dk(P), bt(H);
    std::vector<float> scratch((size_t)glm53c_ref_kda_scratch(H, KD, VD));
    for (int s = 0; s < S; s++) {
        for (int h = 0; h < H; h++) for (int d = 0; d < KD; d++) {
            const int i = h * KD + d; const float sv = expf(k.prm[h]) * (k.f[s * P + i] + k.prm[H + i]);
            dk[i] = lb * sigm(sv);
        }
        for (int h = 0; h < H; h++) bt[h] = sigm(k.b[s * H + h]);
        glm53c_ref_kda_step(core.data(), rs.data(), rw.data(), &k.qkv[(size_t)s * C], k.taps.data(), dk.data(), bt.data(), H, KD, VD, K, neps,
                            scratch.data());
        for (int h = 0; h < H; h++) {
            float sq = 0; for (int d = 0; d < VD; d++) sq += core[h * VD + d] * core[h * VD + d];
            const float inv = 1.f / sqrtf(sq / VD + eps);
            for (int d = 0; d < VD; d++) ref[s * P + h * VD + d] = core[h * VD + d] * inv * k.prm[H + P + d] * sigm(k.g[s * P + h * VD + d]);
        }
    }
    std::vector<float> y, st, win;
    const int ok = kda_run(k, 3, 0, y, st, win);
    const double e = relerr(y.data(), ref.data(), y.size(), 1e-3), es = relerr(st.data(), rs.data(), st.size(), 1e-3),
                 ew = relerr(win.data(), rw.data(), win.size(), 1e-3);
    char what[160];
    snprintf(what, sizeof what, "kda H %d KD %d K %d xs %g: out %.2e state %.2e window %.2e", H, KD, K, xs, e, es, ew);
    ck(ok && e < 1e-4 && es < 1e-5 && ew == 0 && !bad(y.data(), y.size()), what);
}
/* the rows kernel (a call of S >= CC_KDA_ROWS rows) against calls of fewer rows: the same bits */
static void test_kda_rows(int H, int KD, int S, int flags) {
    Kda k = kda_make(H, KD, 4, S, 1.f);
    std::vector<float> y1, st1, w1, y2, st2, w2;
    const int ok = kda_run(k, S, flags, y1, st1, w1) && kda_run(k, CC_KDA_ROWS - 1, flags, y2, st2, w2);
    const int same = ok && !memcmp(y1.data(), y2.data(), y1.size() * 4) && !memcmp(st1.data(), st2.data(), st1.size() * 4) &&
                     !memcmp(w1.data(), w2.data(), w1.size() * 4) && !bad(y1.data(), y1.size());
    char what[160];
    snprintf(what, sizeof what, "kda rows H %d KD %d S %d flags %d: the row kernel's bits", H, KD, S, flags);
    ck(same, what);
}

/* the attention core over random caches: a causal range from kv_start, or every other row a list
 * of every other position newest first with a skipped entry */
static void test_mla_core(int H, int K, int R, int S, int pb, int kv_start, int list) {
    const int cap = pb + S + 3, qseg = K + 5, qrow = H * qseg + 2, rseg = R + 3, rrow = H * rseg + 1, oseg = K + 1, orow = H * oseg;
    const float scale = 0.07f;
    std::vector<float> qa = fvec((size_t)S * qrow + 4, 1.f), qr = fvec((size_t)S * rrow + 4, 1.f);
    std::vector<float> lat = fvec((size_t)cap * K + 2, 1.f), rope = fvec((size_t)cap * (R ? R : 1) + 2, 1.f);
    const int selrow = 2 + cap;
    std::vector<int> sel((size_t)S * selrow, 0);
    for (int s = 0; s < S; s++) {
        const int vis = pb + s + 1; int n = 0;
        if (!list || s % 2) { sel[(size_t)s * selrow] = -1; continue; }
        for (int t = vis - 1; t >= kv_start; t -= 2) sel[(size_t)s * selrow + 1 + n++] = t;
        sel[(size_t)s * selrow + 1 + n++] = -1;
        sel[(size_t)s * selrow] = n;
    }
    std::vector<float> ref((size_t)S * orow, 0.f);
    for (int s = 0; s < S; s++) for (int h = 0; h < H; h++) {
        const float *q = &qa[4 + (size_t)s * qrow + h * qseg], *qrr = &qr[4 + (size_t)s * rrow + h * rseg];
        const int *ls = &sel[(size_t)s * selrow];
        const int n = ls[0] >= 0 ? ls[0] : pb + s + 1 - kv_start;
        std::vector<int> pos; std::vector<double> sc;
        for (int j = 0; j < n; j++) { const int t = ls[0] >= 0 ? ls[1 + j] : kv_start + j; if (t >= 0) pos.push_back(t); }
        double mx = -1e300, sum = 0;
        for (int t : pos) {
            double a = 0;
            for (int i = 0; i < K; i++) a += (double)q[i] * lat[2 + (size_t)t * K + i];
            for (int i = 0; i < R; i++) a += (double)qrr[i] * rope[2 + (size_t)t * R + i];
            sc.push_back(a * scale); if (a * scale > mx) mx = a * scale;
        }
        for (double &v : sc) { v = exp(v - mx); sum += v; }
        for (int i = 0; i < K; i++) {
            double a = 0;
            for (size_t j = 0; j < pos.size(); j++) a += sc[j] / sum * lat[2 + (size_t)pos[j] * K + i];
            ref[(size_t)s * orow + h * oseg + i] = (float)a;
        }
    }
    CcBuf *qb = upf(qa), *rb = upf(qr), *lb = upf(lat), *pb2 = upf(rope), *sb = up(sel.data(), sel.size() * 4);
    CcBuf *ob = T->buf((size_t)S * orow * 4, CC_DEV);
    CcMlaCore cp = {S, H, K, R, pb, kv_start, 4, qrow, qseg, 4, rrow, rseg, 2, K, 2, R, 0, list ? selrow : 0, 0, orow, oseg, scale};
    const int ok = T->begin() && T->zero(ob, 0, (size_t)S * orow) && T->mla_core(qb, rb, lb, R ? pb2 : NULL, list ? sb : NULL, ob, &cp) && T->submit(1);
    std::vector<float> y = down<float>(ob, (size_t)S * orow);
    const double e = relmax(y.data(), ref.data(), y.size());
    char what[160];
    snprintf(what, sizeof what, "mla core H %d K %d R %d S %d pos %d kv_start %d list %d: err %.2e", H, K, R, S, pb, kv_start, list, e);
    ck(ok && e < 2e-6 && !bad(y.data(), y.size()), what);
    CcBuf *all[] = {qb, rb, lb, pb2, sb, ob};
    for (CcBuf *x : all) T->free(x);
}
/* interleaved and rotate-half RoPE, in place and elsewhere; LayerNorm with and without the bias */
static void test_mla_rows(void) {
    const int rows = 4, per = 3, seg = 20, rd = 12, xrow = per * seg + 5, yrow = per * (seg + 2), half = rd / 2;
    std::vector<float> x = fvec((size_t)rows * xrow, 1.f), cs((size_t)rows * rd);
    for (int r = 0; r < rows; r++) for (int j = 0; j < half; j++) {
        const float ang = (float)(500 + 91 * r) * powf(10000.f, -2.f * j / rd);
        cs[r * rd + 2 * j] = cosf(ang); cs[r * rd + 2 * j + 1] = sinf(ang);
    }
    CcBuf *cb = upf(cs);
    for (int style = 0; style < 2; style++) for (int inplace = 0; inplace < 2; inplace++) {
        const int yr = inplace ? xrow : yrow, ys = inplace ? seg : seg + 2;
        std::vector<float> ref((size_t)rows * yr, 0.f);
        if (inplace) ref = x;
        for (int r = 0; r < rows; r++) for (int j = 0; j < per; j++) {
            const float *v = &x[(size_t)r * xrow + j * seg]; float *o = &ref[(size_t)r * yr + j * ys];
            float tmp[256]; for (int i = 0; i < seg; i++) tmp[i] = v[i];
            for (int i = 0; i < half; i++) {
                const float a = style ? tmp[2 * i] : tmp[i], b = style ? tmp[2 * i + 1] : tmp[i + half];
                const float c = cs[r * rd + 2 * i], sn = cs[r * rd + 2 * i + 1];
                o[i] = a * c - b * sn; o[i + half] = b * c + a * sn;
            }
            for (int i = rd; i < seg; i++) o[i] = tmp[i];
        }
        CcBuf *xb = upf(x), *yb = inplace ? xb : T->buf((size_t)rows * yr * 4, CC_DEV);
        CcMlaRow p = {rows * per, per, rd, seg, style, 0, xrow, seg, 0, yr, ys, 0, rd, 0, 0, 0, 0.f};
        const int ok = T->begin() && (inplace || T->zero(yb, 0, (size_t)rows * yr)) && T->mla_rope(xb, cb, yb, &p) && T->submit(1);
        std::vector<float> y = down<float>(yb, (size_t)rows * yr);
        const double e = relerr(y.data(), ref.data(), y.size(), 1e-3);
        char what[120];
        snprintf(what, sizeof what, "mla rope style %d in place %d: err %.2e", style, inplace, e);
        ck(ok && e < 1e-6, what);
        if (!inplace) T->free(yb);
        T->free(xb);
    }
    std::vector<float> prm = fvec(2 * seg, 1.f);
    CcBuf *pb = upf(prm);
    for (int hb = 0; hb < 2; hb++) {
        std::vector<float> ref((size_t)rows * yrow, 0.f);
        for (int r = 0; r < rows; r++) for (int j = 0; j < per; j++) {
            const float *v = &x[(size_t)r * xrow + j * seg]; float *o = &ref[(size_t)r * yrow + j * (seg + 2)];
            double mu = 0, var = 0; for (int i = 0; i < seg; i++) mu += v[i]; mu /= seg;
            for (int i = 0; i < seg; i++) var += (v[i] - mu) * (v[i] - mu); var /= seg;
            const float rr = 1.f / sqrtf((float)var + 1e-6f);
            for (int i = 0; i < seg; i++) o[i] = (float)(v[i] - mu) * rr * prm[i] + (hb ? prm[seg + i] : 0.f);
        }
        CcBuf *xb = upf(x), *yb = T->buf((size_t)rows * yrow * 4, CC_DEV);
        CcMlaRow p = {rows * per, per, 0, seg, 0, 0, xrow, seg, 0, yrow, seg + 2, 0, 0, 0, seg, hb, 1e-6f};
        const int ok = T->begin() && T->zero(yb, 0, (size_t)rows * yrow) && T->mla_lnorm(xb, pb, yb, &p) && T->submit(1);
        std::vector<float> y = down<float>(yb, (size_t)rows * yrow);
        const double e = relerr(y.data(), ref.data(), y.size(), 1e-3);
        char what[120];
        snprintf(what, sizeof what, "mla layernorm bias %d: err %.2e", hb, e);
        ck(ok && e < 2e-6, what);
        T->free(xb); T->free(yb);
    }
    T->free(pb); T->free(cb);
}
/* the per-head block matmuls of a float matrix, rows (the gate) and transposed */
static void test_hgemv(int trans) {
    const int S = 3, H = 3, n = trans ? 24 : 20, I = 128, hstride = n + 9, hoff = 5, O = (H - 1) * hstride + hoff + n + 2, woff = 6;
    std::vector<float> W = fvec((size_t)O * I + woff, 0.1f);
    const int xin = trans ? n : I, xseg = xin + 3, xrow = H * xseg + 1, yout = trans ? I : n, yseg = yout + 2, yrow = H * yseg;
    std::vector<float> x = fvec((size_t)S * xrow + 7, 1.f), g = fvec((size_t)S * H * n, 2.f), ref((size_t)S * yrow, 0.f);
    for (int s = 0; s < S; s++) for (int h = 0; h < H; h++) {
        const float *xs = &x[7 + (size_t)s * xrow + h * xseg]; float *ys = &ref[(size_t)s * yrow + h * yseg];
        if (trans) for (int i = 0; i < I; i++) {
            double a = 0; for (int d = 0; d < n; d++) a += (double)W[woff + (size_t)(h * hstride + hoff + d) * I + i] * xs[d];
            ys[i] = (float)a;
        } else for (int o = 0; o < n; o++) {
            double a = 0; for (int i = 0; i < I; i++) a += (double)W[woff + (size_t)(h * hstride + hoff + o) * I + i] * xs[i];
            ys[o] = (float)a * sigm(g[(size_t)s * H * n + h * n + o]);
        }
    }
    CcBuf *wb = upf(W), *xb = upf(x), *gb = upf(g), *yb = T->buf((size_t)S * yrow * 4, CC_DEV);
    CcHgemv p = {trans, S, H, n, hstride, hoff, 7, xrow, xseg, 0, yrow, yseg, 0, H * n, !trans, I, woff};
    const int ok = T->begin() && T->zero(yb, 0, (size_t)S * yrow) && T->mla_hgemv(wb, xb, yb, trans ? NULL : gb, &p) && T->submit(1);
    std::vector<float> y = down<float>(yb, (size_t)S * yrow);
    const double e = relmax(y.data(), ref.data(), y.size());
    char what[120];
    snprintf(what, sizeof what, "hgemv float trans %d: err %.2e", trans, e);
    ck(ok && e < 2e-6, what);
    CcBuf *all[] = {wb, xb, gb, yb};
    for (CcBuf *b : all) T->free(b);
}
/* the k-pooled selection slot for slot against sparse_index.h, then the pooled keys */
/* the Q8_0 rows kernel against the float rows kernel on the same rows dequantized (d * q in f32, as
 * ggml_quants.h): the same bits */
static void test_hgemv_q8(int H, int n, int I, int hstride, int hoff) {
    const int S = 3, O = (H - 1) * hstride + hoff + n, nb = I / 32, xseg = I + 3, xrow = H * xseg + 1, yseg = n + 2, yrow = H * yseg;
    std::vector<unsigned char> raw((size_t)O * nb * 34);
    std::vector<float> W((size_t)O * I);
    for (int r = 0; r < O; r++) for (int b = 0; b < nb; b++) {
        unsigned char *blk = &raw[((size_t)r * nb + b) * 34];
        /* a scale a half holds exactly (k / 4096, k < 2048), its bits made by hand: no half type */
        const int k = 20 + (int)(rnd() % 80);
        const float df = (float)k / 4096.0f;
        unsigned fb; memcpy(&fb, &df, 4);
        const unsigned short u = (unsigned short)(((fb >> 16) & 0x8000u) | ((((fb >> 23) & 255u) - 112u) << 10) | ((fb >> 13) & 1023u));
        blk[0] = (unsigned char)(u & 255); blk[1] = (unsigned char)(u >> 8);
        for (int j = 0; j < 32; j++) {
            const signed char q = (signed char)((int)(rnd() % 255) - 127);
            blk[2 + j] = (unsigned char)q;
            W[(size_t)r * I + b * 32 + j] = df * (float)q;
        }
    }
    std::vector<float> x = fvec((size_t)S * xrow + 5, 1.f);
    CcBuf *wf = upf(W), *wq = up(raw.data(), (raw.size() + 3) & ~(size_t)3), *xb = upf(x);
    CcBuf *ya = T->buf((size_t)S * yrow * 4, CC_DEV), *yb = T->buf((size_t)S * yrow * 4, CC_DEV);
    CcHgemv p = {0, S, H, n, hstride, hoff, 5, xrow, xseg, 0, yrow, yseg, 0, 0, 0, I, 0};
    const int ok = T->begin() && T->zero(ya, 0, (size_t)S * yrow) && T->zero(yb, 0, (size_t)S * yrow) &&
                   T->mla_hgemv(wf, xb, ya, NULL, &p) && T->mla_hgemv_q8(wq, xb, yb, &p) && T->submit(1);
    std::vector<float> a = down<float>(ya, (size_t)S * yrow), b = down<float>(yb, (size_t)S * yrow);
    char what[120];
    snprintf(what, sizeof what, "hgemv q8 H %d n %d I %d: the float kernel's bits", H, n, I);
    ck(ok && !memcmp(a.data(), b.data(), a.size() * 4) && !bad(a.data(), a.size()), what);
    CcBuf *all[] = {wf, wq, xb, ya, yb};
    for (CcBuf *c : all) T->free(c);
}

static void test_kpool(int S, int pb, int IH, int ID, int pool, int topk, int tail) {
    const int Tn = pb + S, width = glm53c_ref_width(topk, pool, tail), selrow = width + 1, npool = Tn / pool, scrow = npool + 1;
    std::vector<float> keys((size_t)Tn * ID), gates((size_t)Tn * ID, 0.f), ape((size_t)pool * ID, 0.f), q((size_t)S * IH * ID), hw((size_t)S * IH);
    for (auto &v : keys) v = ((int)(rnd() % 17) - 8) * 0.25f;
    for (auto &v : q) v = ((int)(rnd() % 17) - 8) * 0.25f;
    for (auto &v : hw) v = ((int)(rnd() % 9) - 4) * 0.5f * sqrtf((float)IH);
    for (int t = 0; t < Tn; t += 7) for (int i = 0; i < ID; i++) keys[(size_t)t * ID + i] = keys[(size_t)(t % 3) * ID + i];   /* equal pools: ties */
    std::vector<unsigned char> valid((size_t)Tn, 1);
    std::vector<int> ref((size_t)S * width);
    std::vector<float> hwd((size_t)S * IH);
    for (int i = 0; i < S * IH; i++) hwd[i] = hw[i] / sqrtf((float)IH);
    glm53c_ref_kpool(ref.data(), q.data(), keys.data(), gates.data(), hwd.data(), ape.data(), valid.data(), Tn, IH, ID, pool, topk, tail, pb, Tn);
    CcBuf *kb = upf(keys), *gb = upf(gates), *ab = upf(ape), *qb = upf(q), *wb = upf(hw);
    CcBuf *pk = T->buf((size_t)(npool + 1) * ID * 4, CC_DEV), *scb = T->buf((size_t)S * scrow * 4, CC_DEV), *sb = T->buf((size_t)S * selrow * 4, CC_DEV);
    CcDsaPool kp = {npool, pool, 0, ID, 0, ID, 0, 0, 0, ID};
    CcDsaPick sp = {S, pb, IH, ID, topk, pool, 0, IH * ID, 0, IH, 0, tail, scrow, selrow, sqrtf((float)IH), 1.f / sqrtf((float)ID)};
    int ok = T->begin() && T->dsa_pool_keys(kb, gb, ab, pk, &kp) && T->dsa_pool_select(qb, wb, pk, scb, sb, &sp) && T->submit(1);
    std::vector<int> got = down<int>(sb, (size_t)S * selrow);
    int same = ok;
    for (int s = 0; same && s < S; s++) {
        same = got[(size_t)s * selrow] == width;
        for (int i = 0; same && i < width; i++) same = got[(size_t)s * selrow + 1 + i] == ref[(size_t)s * width + i];
    }
    char what[160];
    snprintf(what, sizeof what, "kpool S %d pos %d IH %d ID %d pool %d topk %d tail %d: the selection", S, pb, IH, ID, pool, topk, tail);
    ck(same, what);
    for (auto &v : gates) v = frnd() * 2.f;
    for (auto &v : ape) v = frnd() * 0.5f;
    std::vector<float> rk((size_t)npool * ID);
    for (int pp = 0; pp < npool; pp++) for (int d = 0; d < ID; d++) {
        float mx = -3.402823466e38f, tot = 0, mix = 0;
        for (int j = 0; j < pool; j++) { const float l = gates[(size_t)(pp * pool + j) * ID + d] + ape[(size_t)j * ID + d]; if (l > mx) mx = l; }
        for (int j = 0; j < pool; j++) tot += expf(gates[(size_t)(pp * pool + j) * ID + d] + ape[(size_t)j * ID + d] - mx);
        for (int j = 0; j < pool; j++) mix += expf(gates[(size_t)(pp * pool + j) * ID + d] + ape[(size_t)j * ID + d] - mx) / tot * keys[(size_t)(pp * pool + j) * ID + d];
        rk[(size_t)pp * ID + d] = mix;
    }
    ok = T->begin() && T->write(gb, 0, gates.data(), gates.size() * 4) && T->write(ab, 0, ape.data(), ape.size() * 4) &&
         T->dsa_pool_keys(kb, gb, ab, pk, &kp) && T->submit(1);
    std::vector<float> gk = down<float>(pk, (size_t)npool * ID);
    const double e = relmax(gk.data(), rk.data(), gk.size());
    snprintf(what, sizeof what, "kpool keys pool %d ID %d: err %.2e", pool, ID, e);
    ck(ok && e < 2e-6, what);
    CcBuf *all[] = {kb, gb, ab, qb, wb, pk, scb, sb};
    for (CcBuf *b : all) T->free(b);
}

static void test_dsa_select(int S, int pb, int IH, int ID, int topk, int force) {
    const int Tn = pb + S, selrow = topk + 1;
    std::vector<float> keys((size_t)Tn * ID), q((size_t)S * IH * ID), hw((size_t)S * IH);
    for (auto &v : keys) v = ((int)(rnd() % 17) - 8) * 0.25f;
    for (auto &v : q) v = ((int)(rnd() % 17) - 8) * 0.25f;
    for (auto &v : hw) v = ((int)(rnd() % 9) - 4) * 0.5f;
    for (int t = 0; t < Tn; t += 5) for (int i = 0; i < ID; i++) keys[(size_t)t * ID + i] = keys[(size_t)(t % 3) * ID + i];   /* equal keys: ties */
    std::vector<int> ref((size_t)S * topk), nref((size_t)S);
    glmdsa_ref_select(ref.data(), nref.data(), q.data(), hw.data(), keys.data(), S, pb, IH, ID, topk, force);
    CcBuf *kb = upf(keys), *qb = upf(q), *wb = upf(hw);
    CcBuf *scb = T->buf((size_t)S * Tn * 4, CC_DEV), *sb = T->buf((size_t)S * selrow * 4, CC_DEV);
    CcDsaSel ds = {S, pb, IH, ID, topk, force, 0, IH * ID, 0, IH, 0, Tn, selrow, 1.f / sqrtf((float)ID), 1.f / sqrtf((float)IH)};
    const int ok = T->begin() && T->dsa_select(qb, wb, kb, scb, sb, &ds) && T->submit(1);
    std::vector<int> got = down<int>(sb, (size_t)S * selrow);
    int same = ok, lists = 0;
    for (int s = 0; same && s < S; s++) {
        const int c = got[(size_t)s * selrow];
        if (nref[s] == 0) { same = c == -1; continue; }
        lists++;
        same = c == nref[s];
        for (int i = 0; same && i < c; i++) same = got[(size_t)s * selrow + 1 + i] == ref[(size_t)s * topk + i];
    }
    char what[160];
    snprintf(what, sizeof what, "dsa S %d pos %d IH %d ID %d topk %d%s: %d rows' lists position for position", S, pb, IH, ID, topk,
             force ? " force" : "", lists);
    ck(same, what);
    CcBuf *all[] = {kb, qb, wb, scb, sb};
    for (CcBuf *b : all) T->free(b);
}

int main(void) {
    int dev = 0;
    if (!coli_cuda_init(&dev, 1)) { printf("no CUDA device: skipped\n"); return 0; }
    T = coli_cuda_chain_ops();
    if (!T || T->size < offsetof(ColiCudaChainOps, dsa_pool_select) + sizeof T->dsa_pool_select || !T->dsa_pool_select) {
        printf("FAIL: the backend has no GLM-5.3 ops\n"); return 1; }
    if (!T->init(0) || !T->ready()) { printf("FAIL: no chain context\n"); return 1; }
    printf("kda\n");
    test_kda(2, 64, 4, 1.f); test_kda(3, 128, 4, 1.f); test_kda(2, 128, 4, 0.01f); test_kda(4, 64, 3, 0.05f);
    test_kda_rows(3, 128, 21, 0); test_kda_rows(2, 64, 8, 0); test_kda_rows(3, 128, 19, 2); test_kda_rows(2, 64, 30, 3);
    printf("mla\n");
    test_mla_core(3, 64, 0, 3, 20, 0, 0); test_mla_core(2, 512, 0, 4, 300, 0, 1); test_mla_core(4, 512, 64, 2, 900, 100, 1);
    test_mla_core(64, 512, 0, 1, 2100, 0, 1); test_mla_core(1, 1000, 32, 5, 70, 7, 0);
    test_mla_rows(); test_hgemv(0); test_hgemv(1);
    printf("kpool\n");
    test_kpool(3, 20, 4, 16, 2, 8, 1); test_kpool(2, 9, 1, 16, 4, 8, 1); test_kpool(4, 300, 4, 64, 4, 64, 1);
    test_kpool(1, 2, 4, 16, 2, 4, 0); test_kpool(2, 30, 16, 16, 2, 6, 1); test_kpool(2, 1000, 4, 32, 4, 128, 1);
    test_kpool(2, 5000, 32, 128, 4, 2048, 1);
    printf("dsa\n");
    if (T->size < offsetof(ColiCudaChainOps, mla_hgemv_q8) + sizeof T->mla_hgemv_q8 || !T->dsa_select || !T->mla_hgemv_q8) {
        printf("FAIL: the backend has no GLM-5.2 / GLM-5.3 ops\n"); fails++; }
    else {
        test_dsa_select(3, 6, 2, 16, 4, 0); test_dsa_select(4, 2, 2, 16, 4, 0); test_dsa_select(2, 3, 2, 16, 8, 1);
        test_dsa_select(3, 300, 4, 64, 64, 0); test_dsa_select(2, 2100, 32, 128, 2048, 0); test_dsa_select(5, 4000, 32, 128, 2048, 0);
        test_dsa_select(1, 9000, 32, 128, 2048, 0); test_dsa_select(2, 30, 64, 64, 2048, 1);
        test_hgemv_q8(4, 32, 192, 32, 0); test_hgemv_q8(64, 256, 512, 256, 0); test_hgemv_q8(3, 20, 64, 29, 5);
    }
    T->shutdown();
    printf(fails ? "FAIL (%d)\n" : "PASS\n", fails);
    return fails != 0;
}
