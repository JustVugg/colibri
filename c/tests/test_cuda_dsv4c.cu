/* DeepSeek V4.1's layer ops on the CUDA chain (cuda_dsv4c.cuh) against the CPU's references,
 * the cases and bounds tests/test_vk_chain.c holds the Vulkan shaders to:
 *   mhc      the split (hyper_connections.h's coli_hc_split_sinkhorn on the same mixes), the
 *            collapse, the write back (coli_hc_post), the mean, the clamped SwiGLU;
 *   attn     sparse_attn.h's coli_sparse_attend_scalar (flags 0) and DeepSeek V4's bf16 form
 *            (flags 1), a decode row and a prompt chunk's rows (the three passes, and their
 *            bits against the one-block kernel's, COLI_CUDA_DSV4_PA=0);
 *   rope     interleaved pairs, inverse and bf16;
 *   compress the rolling group over three calls, overlap and ape;
 *   index    the scores, the candidate blocks and the top-k list slot for slot (exact inputs);
 *   engram   the gate in place on the streams.
 * make CUDA=1 cuda-dsv4c-check (the references in tests/cuda_dsv4c_ref.c) */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include "../backend_cuda.h"
#include "../cuda_chain.h"
#ifdef _WIN32
/* MSVC has no POSIX setenv/unsetenv */
static int setenv(const char *name, const char *value, int overwrite) {
    (void)overwrite; return _putenv_s(name, value);
}
static int unsetenv(const char *name) { return _putenv_s(name, ""); }
#endif
extern "C" {   /* tests/cuda_dsv4c_ref.c */
int dsv4c_ref_split(float *pre, float *post, float *comb, const float *mixes, const float *scale, const float *base, int hc, int iters, float eps);
int dsv4c_ref_post(float *out, const float *branch, const float *residual, const float *post, const float *comb, int hc, int D);
void dsv4c_ref_attend(float *out, const float *q, const float *kv, const int *idx, int count, int hd, float sink, float scale, float *score);
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
static float bf16r(float f) {
    unsigned u; memcpy(&u, &f, 4);
    if ((u & 0x7f800000u) != 0x7f800000u) u += 0x7fffu + ((u >> 16) & 1u);
    u &= 0xffff0000u; memcpy(&f, &u, 4); return f;
}
static float sigm(float v) { if (v >= 0) { float d = expf(-v); return 1.f / (1.f + d); } float g = expf(v); return g / (1.f + g); }

static void test_mhc(int H, int D, int iters) {
    const int S = 3, HD = H * D, nm = (2 + H) * H, hrow = 2 * H + H * H;
    const float eps = 1e-5f, hce = 1e-6f, lim = 1.5f;
    std::vector<float> x = fvec((size_t)S * HD, 1.f), mix = fvec((size_t)S * nm, 2.f), br = fvec((size_t)S * D, 1.f);
    std::vector<float> prm(3 + nm);
    for (int i = 0; i < 3; i++) prm[i] = 0.5f + (rnd() % 100) / 100.f;
    for (int i = 0; i < nm; i++) prm[3 + i] = frnd() * 0.5f;
    std::vector<float> rp((size_t)S * hrow), rcol((size_t)S * D), rout((size_t)S * HD), rmean((size_t)S * D), ms(nm);
    for (int s = 0; s < S; s++) {
        float a = 0; for (int i = 0; i < HD; i++) a += x[s * HD + i] * x[s * HD + i];
        const float r = 1.f / sqrtf(a / HD + eps);
        for (int i = 0; i < nm; i++) ms[i] = mix[s * nm + i] * r;
        dsv4c_ref_split(&rp[s * hrow], &rp[s * hrow + H], &rp[s * hrow + 2 * H], ms.data(), prm.data(), prm.data() + 3, H, iters, hce);
        for (int d = 0; d < D; d++) {
            float c = 0, m = 0;
            for (int i = 0; i < H; i++) { c += rp[s * hrow + i] * x[s * HD + i * D + d]; m += x[s * HD + i * D + d]; }
            rcol[s * D + d] = c; rmean[s * D + d] = m / H;
        }
        dsv4c_ref_post(&rout[s * HD], &br[s * D], &x[s * HD], &rp[s * hrow + H], &rp[s * hrow + 2 * H], H, D);
    }
    CcBuf *xb = upf(x), *mb = upf(mix), *bb = upf(br), *pb = upf(prm);
    CcBuf *hb = T->buf((size_t)S * hrow * 4, CC_DEV), *cb = T->buf((size_t)S * D * 4, CC_DEV), *ob = T->buf((size_t)S * HD * 4, CC_DEV),
          *eb = T->buf((size_t)S * D * 4, CC_DEV);
    CcMhc sp = {S, H, D, iters, 0, HD, 0, nm, 0, hrow, 0, D, 0, 0, eps, hce, 0.f};
    CcMhc po = {S, H, D, iters, 0, HD, 0, D, 0, hrow, 0, HD, 0, 0, eps, hce, 0.f};
    int ok = T->begin() && T->mhc(CC_MHC_SPLIT, xb, mb, hb, pb, NULL, &sp) && T->mhc(CC_MHC_COLLAPSE, xb, NULL, hb, NULL, cb, &sp) &&
             T->mhc(CC_MHC_POST, xb, bb, hb, NULL, ob, &po) && T->mhc(CC_MHC_MEAN, xb, NULL, NULL, NULL, eb, &sp) && T->submit(1);
    auto hp = down<float>(hb, (size_t)S * hrow), col = down<float>(cb, (size_t)S * D), out = down<float>(ob, (size_t)S * HD), mean = down<float>(eb, (size_t)S * D);
    double es = 0;
    for (size_t i = 0; i < hp.size(); i++) { double d = fabs(hp[i] - rp[i]); if (d > es) es = d; }
    const double ec = relerr(col.data(), rcol.data(), col.size(), 1e-3), eo = relerr(out.data(), rout.data(), out.size(), 1e-3),
                 em = relerr(mean.data(), rmean.data(), mean.size(), 1e-3);
    char m[200]; snprintf(m, sizeof m, "mhc H %d D %d iters %d: split %.2e collapse %.2e write back %.2e mean %.2e", H, D, iters, es, ec, eo, em);
    ck(ok && es < 5e-7 && ec < 2e-4 && eo < 2e-4 && em < 1e-6, m);   /* fma sums: float rounding, not the CPU's order */
    const int n = 333;
    std::vector<float> ga = fvec(n, 3.f), ua = fvec(n, 3.f), rr(n);
    for (int i = 0; i < n; i++) { float gg = ga[i] > lim ? lim : ga[i], uu = ua[i] < -lim ? -lim : (ua[i] > lim ? lim : ua[i]); rr[i] = gg / (1.f + expf(-gg)) * uu; }
    CcBuf *gb = upf(ga), *ub = upf(ua), *yb = T->buf((size_t)n * 4, CC_DEV);
    CcMhc sw = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, n, 0.f, 0.f, lim};
    ok = T->begin() && T->mhc(CC_SWIGLU_CLAMP, gb, ub, NULL, NULL, yb, &sw) && T->submit(1);
    auto y = down<float>(yb, n);
    snprintf(m, sizeof m, "clamped swiglu: rel err %.2e", relerr(y.data(), rr.data(), n, 1e-3));
    ck(ok && relerr(y.data(), rr.data(), n, 1e-3) < 1e-6, m);
    for (CcBuf *b : {xb, mb, bb, pb, hb, cb, ob, eb, gb, ub, yb}) T->free(b);
}

static void test_attn(int S, int H, int hd, int nwin, int ncmp, int cnt, int flags) {
    const int qrow = H * hd + 3, lrow = cnt + 2, orow = H * hd + 3;
    std::vector<float> q = fvec((size_t)S * qrow, 1.f), win = fvec((size_t)nwin * hd + 5, 1.f), cmp = fvec((size_t)ncmp * hd + 7, 1.f),
                       sink = fvec((size_t)H + 2, 2.f), ref((size_t)S * orow, 0.f), sc(cnt), kvall((size_t)(nwin + ncmp) * hd);
    std::vector<int> list((size_t)S * lrow);
    memcpy(kvall.data(), win.data() + 5, (size_t)nwin * hd * 4); memcpy(kvall.data() + (size_t)nwin * hd, cmp.data() + 7, (size_t)ncmp * hd * 4);
    for (int s = 0; s < S; s++)
        for (int j = 0; j < lrow; j++) {
            int r = (int)(rnd() % 5);
            list[s * lrow + j] = s == S - 1 && S > 1 ? -1 : r == 0 ? -1 : r < 3 ? (int)(rnd() % (unsigned)nwin) : nwin + (int)(rnd() % (unsigned)ncmp);
        }
    for (int s = 0; s < S; s++)
        for (int h = 0; h < H; h++) {
            const float *qq = &q[(size_t)s * qrow + 1 + h * hd];
            float *o = &ref[(size_t)s * orow + 2 + h * hd];
            const int *idx = &list[s * lrow + 1];
            if (!(flags & 1)) { dsv4c_ref_attend(o, qq, kvall.data(), idx, cnt, hd, sink[1 + h], 1.f / sqrtf((float)hd), sc.data()); continue; }
            float best = -1e30f;
            for (int j = 0; j < cnt; j++) {
                if (idx[j] < 0) { sc[j] = -INFINITY; continue; }
                float d = 0; for (int i = 0; i < hd; i++) d += qq[i] * kvall[(size_t)idx[j] * hd + i];
                sc[j] = d * (1.f / sqrtf((float)hd)); if (sc[j] > best) best = sc[j];
            }
            float den = expf(sink[1 + h] - best);
            for (int i = 0; i < hd; i++) o[i] = 0;
            for (int j = 0; j < cnt; j++) {
                if (idx[j] < 0) continue;
                float w = expf(sc[j] - best); den += w; w = bf16r(w);
                for (int i = 0; i < hd; i++) o[i] += w * kvall[(size_t)idx[j] * hd + i];
            }
            for (int i = 0; i < hd; i++) o[i] = bf16r(o[i] / den);
        }
    CcBuf *qb = upf(q), *wb = upf(win), *cb = upf(cmp), *sb = upf(sink), *lb = up(list.data(), list.size() * 4);
    CcBuf *ob = T->buf((size_t)S * orow * 4, CC_DEV), *o0 = T->buf((size_t)S * orow * 4, CC_DEV);
    CcDsAttn p = {S, H, hd, cnt, 1, lrow, nwin, 5, 7, 1, qrow, 2, orow, 1, flags, 1.f / sqrtf((float)hd)};
    int ok = T->begin() && T->ds_attn(qb, wb, cb, lb, sb, ob, &p) && T->submit(1);
    auto got = down<float>(ob, (size_t)S * orow);
    double e = 0;
    for (int s = 0; s < S; s++) {   /* flags 1: a value of bf16 is an ulp of 2^-8 apart; measured above a floor of 1e-2 */
        double r = relerr(&got[(size_t)s * orow + 2], &ref[(size_t)s * orow + 2], (size_t)H * hd, (flags & 1) ? 1e-2 : 1e-3); if (r > e) e = r;
    }
    char m[200]; snprintf(m, sizeof m, "attn S %d H %d hd %d cnt %d flags %d: rel err %.2e", S, H, hd, cnt, flags, e);
    ck(ok && e < ((flags & 1) ? 8e-3 : 2e-4), m);
    if (S >= 8 && hd % 64 == 0) {   /* the prompt passes against the one-block kernel, bit for bit */
        setenv("COLI_CUDA_DSV4_PA", "0", 1);
        ok = T->begin() && T->ds_attn(qb, wb, cb, lb, sb, o0, &p) && T->submit(1);
        unsetenv("COLI_CUDA_DSV4_PA");
        auto g0 = down<float>(o0, (size_t)S * orow);
        int same = 1;
        for (int s = 0; s < S; s++) same &= !memcmp(&got[(size_t)s * orow + 2], &g0[(size_t)s * orow + 2], (size_t)H * hd * 4);
        snprintf(m, sizeof m, "attn S %d H %d hd %d cnt %d flags %d: the prompt passes give the one-block kernel's bits", S, H, hd, cnt, flags);
        ck(ok && same, m);
    }
    for (CcBuf *b : {qb, wb, cb, sb, lb, ob, o0}) T->free(b);
}

static void test_rope(int flags) {
    const int rows = 3, per = 4, seg = 40, rd = 16, xrow = per * seg + 2, csrow = rd + 1;
    std::vector<float> x = fvec((size_t)rows * xrow + 3, 1.f), cs((size_t)rows * csrow + 2);
    for (int r = 0; r < rows; r++) for (int i = 0; i < rd / 2; i++) {
        float a = (r * 7 + 3) * powf(10000.f, -2.f * i / rd);
        cs[2 + r * csrow + 2 * i] = cosf(a); cs[2 + r * csrow + 2 * i + 1] = sinf(a);
    }
    for (int inv = 0; inv < 2; inv++) {
        std::vector<float> ref = x;
        for (int r = 0; r < rows; r++) for (int j = 0; j < per; j++) {
            float *v = &ref[3 + r * xrow + j * seg + (seg - rd)];
            for (int i = 0; i < rd / 2; i++) {
                float c = cs[2 + r * csrow + 2 * i], sn = cs[2 + r * csrow + 2 * i + 1];
                if (inv) sn = -sn;
                float a = v[2 * i], b = v[2 * i + 1];
                v[2 * i] = a * c - b * sn; v[2 * i + 1] = a * sn + b * c;
                if (flags & 1) { v[2 * i] = bf16r(v[2 * i]); v[2 * i + 1] = bf16r(v[2 * i + 1]); }
            }
        }
        CcBuf *xb = upf(x), *cb = upf(cs);
        CcDsRope p = {rows * per, per, rd, 3 + seg - rd, xrow, seg, 2, csrow, inv, flags};
        int ok = T->begin() && T->ds_rope(xb, cb, &p) && T->submit(1);
        auto got = down<float>(xb, x.size());
        double e = relerr(got.data(), ref.data(), x.size(), 1e-3);
        char m[160]; snprintf(m, sizeof m, "rope inverse %d flags %d: rel err %.2e", inv, flags, e);
        ck(ok && e < ((flags & 1) ? 8e-3 : 1e-6), m);
        T->free(xb); T->free(cb);
    }
}

static void test_compress(int ratio, int overlap, int ape, int Tn) {
    const int D = 24, P = overlap ? 2 * D : D, rows = overlap ? 2 * ratio : ratio, pb0 = 3;
    std::vector<float> kv = fvec((size_t)Tn * P, 1.f), sc = fvec((size_t)Tn * P, 2.f), ap = fvec((size_t)ratio * P + 4, 1.f);
    std::vector<float> rk((size_t)rows * P, 0.f), rs((size_t)rows * P, -INFINITY);
    const int G = (pb0 + Tn) / ratio + 1;
    std::vector<float> ref((size_t)G * D, 0.f);
    for (int t = 0; t < Tn; t++) {
        int pos = pb0 + t, slot = pos % ratio, sr = overlap ? ratio + slot : slot;
        for (int c = 0; c < P; c++) { rk[sr * P + c] = kv[t * P + c]; rs[sr * P + c] = sc[t * P + c] + (ape ? ap[4 + slot * P + c] : 0.f); }
        if ((pos + 1) % ratio) continue;
        for (int d = 0; d < D; d++) {
            float best = -INFINITY;
            for (int k = 0; k < rows; k++) { int col = overlap && k >= ratio ? D + d : d; if (rs[k * P + col] > best) best = rs[k * P + col]; }
            float tot = 0, mix = 0;
            for (int k = 0; k < rows; k++) { int col = overlap && k >= ratio ? D + d : d; float w = expf(rs[k * P + col] - best); tot += w; mix += w * rk[k * P + col]; }
            ref[(size_t)(pos / ratio) * D + d] = mix / tot;
        }
        if (overlap) { memcpy(rk.data(), rk.data() + (size_t)ratio * P, (size_t)ratio * P * 4); memcpy(rs.data(), rs.data() + (size_t)ratio * P, (size_t)ratio * P * 4); }
    }
    std::vector<float> init((size_t)2 * rows * P + 5);
    for (size_t i = 0; i < init.size(); i++) init[i] = (int)i >= 5 + rows * P ? -INFINITY : 0.f;
    CcBuf *kb = upf(kv), *sb = upf(sc), *ab = upf(ap), *ring = upf(init), *ob = T->buf((size_t)G * D * 4, CC_DEV);
    int cuts[4] = {0, 5, 6, Tn}, ok = 1;
    for (int k = 0; k < 3; k++) {
        CcDsComp p = {cuts[k + 1] - cuts[k], pb0 + cuts[k], ratio, P, D, overlap, cuts[k] * P, P, cuts[k] * P, P, ape ? 4 : -1, 0, D, 5};
        ok &= T->begin() && T->ds_compress(kb, sb, ring, ab, ob, &p) && T->submit(1);
    }
    auto got = down<float>(ob, (size_t)G * D);
    const int g0 = (pb0 + 1 + ratio - 1) / ratio;
    double e = relerr(&got[(size_t)g0 * D], &ref[(size_t)g0 * D], (size_t)(G - 1 - g0) * D, 1e-3);
    char m[160]; snprintf(m, sizeof m, "compress ratio %d overlap %d ape %d: rel err %.2e", ratio, overlap, ape, e);
    ck(ok && e < 1e-5, m);
    for (CcBuf *b : {kb, sb, ab, ring, ob}) T->free(b);
}

static void topk_ref(const float *score, int width, int topk, int base, int order, int *row) {
    int taken = 0;
    for (int k = 0; k < topk; k++) row[k] = -1;
    for (int k = 0; k < topk; k++) {
        int best = -1;
        for (int j = 0; j < width; j++) {
            if (score[j] == -INFINITY) continue;
            int already = 0; for (int u = 0; u < taken; u++) if (row[u] == j) { already = 1; break; }
            if (already) continue;
            if (best < 0 || score[j] > score[best]) best = j;
        }
        if (best < 0) break;
        row[taken++] = best;
    }
    if (order == 0) for (int a = 0; a < taken; a++) for (int b = a + 1; b < taken; b++) if (row[b] < row[a]) { int t = row[a]; row[a] = row[b]; row[b] = t; }
    for (int k = 0; k < taken; k++) row[k] += base;
}
static void cand_ref(const float *score, int width, int lens, int block, int topb, int *keep) {
    int blocks = (width + block - 1) / block;
    std::vector<float> best(blocks + 1);
    for (int b = 0; b < blocks; b++) { float top = -INFINITY; for (int i = b * block; i < (b + 1) * block && i < width; i++) if (score[i] > top) top = score[i]; best[b] = top; }
    int last = (lens - 1) / block;
    if (last >= 0 && last < blocks) best[last] = INFINITY;
    int wanted = topb < blocks ? topb : blocks;
    for (int i = 0; i < width; i++) keep[i] = 0;
    for (int pick = 0; pick < wanted; pick++) {
        int chosen = -1;
        for (int b = 0; b < blocks; b++) if (best[b] > -INFINITY && (chosen < 0 || best[b] > best[chosen])) chosen = b;
        if (chosen < 0) break;
        for (int i = chosen * block; i < (chosen + 1) * block && i < width; i++) keep[i] = 1;
        best[chosen] = -INFINITY;
    }
}
static void test_index(int S, int pos_base, int ratio, int IH, int ID, int topk, int block, int topb, int order) {
    int width = (pos_base + S) / ratio;
    if (width < 1) width = 1;
    const int krow = ID + 3, qrow = IH * ID + 1, wrow = IH + 2, scrow = width + 3, lrow = topk + 5, base = 11;
    std::vector<float> q((size_t)S * qrow + 1), k((size_t)width * krow + 3), w((size_t)S * wrow + 2);
    for (auto &v : q) v = ((int)(rnd() % 17) - 8) * 0.25f;
    for (auto &v : k) v = ((int)(rnd() % 17) - 8) * 0.25f;
    for (auto &v : w) v = ((int)(rnd() % 9) - 3) * 0.5f;
    for (int t = 0; t < width; t += 4) for (int i = 0; i < ID; i++) k[3 + t * krow + i] = k[3 + (t / 4) * krow + i];   /* ties */
    const float wscale = 0.25f;
    std::vector<float> ref((size_t)S * scrow), refm((size_t)S * scrow);
    std::vector<int> keep((size_t)S * scrow), lst((size_t)S * lrow), lstm((size_t)S * lrow);
    for (int s = 0; s < S; s++) {
        int lens = (pos_base + s + 1) / ratio;
        for (int j = 0; j < width; j++) {
            if (j >= lens) { ref[s * scrow + j] = -INFINITY; continue; }
            float total = 0;
            for (int h = 0; h < IH; h++) {
                float dot = 0; for (int i = 0; i < ID; i++) dot += q[1 + s * qrow + h * ID + i] * k[3 + j * krow + i];
                if (dot > 0.f) total += dot * (w[2 + s * wrow + h] * wscale);
            }
            ref[s * scrow + j] = total;
        }
        cand_ref(&ref[s * scrow], width, lens, block, topb, &keep[s * scrow]);
        for (int j = 0; j < width; j++) refm[s * scrow + j] = keep[s * scrow + j] ? ref[s * scrow + j] : -INFINITY;
        topk_ref(&ref[s * scrow], width, topk, base, order, &lst[s * lrow + 2]);
        topk_ref(&refm[s * scrow], width, topk, base, order, &lstm[s * lrow + 2]);
    }
    CcBuf *qb = upf(q), *kb = upf(k), *wb = upf(w);
    CcBuf *scb = T->buf((size_t)S * scrow * 4, CC_DEV), *mb = T->buf((size_t)S * scrow * 4, CC_DEV), *lb = T->buf((size_t)S * lrow * 4, CC_DEV),
          *lb2 = T->buf((size_t)S * lrow * 4, CC_DEV), *scb2 = T->buf((size_t)S * scrow * 4, CC_DEV);
    CcDsScore sp = {S, pos_base, ratio, IH, ID, width, 1, qrow, 2, wrow, 3, krow, 0, scrow, wscale, 0, 0};
    const int moff = 3;   /* the mask a byte an entry, at an offset no word aligns */
    CcDsCand cp = {S, pos_base, ratio, width, block, topb, scrow, scrow, moff};
    CcDsTopk tp = {S, width, topk, scrow, 2, lrow, base, order};
    CcDsScore sm = sp; sm.mask_row = scrow; sm.mask_off = moff;
    int ok = T->begin() && T->ds_score(qb, wb, kb, NULL, scb, &sp) && T->ds_cand(scb, mb, &cp) && T->ds_topk(scb, lb, &tp) &&
             T->ds_score(qb, wb, kb, mb, scb2, &sm) && T->ds_topk(scb2, lb2, &tp) && T->submit(1);
    auto gs = down<float>(scb, (size_t)S * scrow);
    auto gm = down<unsigned char>(mb, (size_t)S * scrow + moff);
    auto gl = down<int>(lb, (size_t)S * lrow), gl2 = down<int>(lb2, (size_t)S * lrow);
    int same_sc = 1, same_m = 1, same_l = 1, same_l2 = 1;
    for (int s = 0; s < S; s++) {
        for (int j = 0; j < width; j++) { same_sc &= gs[s * scrow + j] == ref[s * scrow + j]; same_m &= gm[moff + s * scrow + j] == keep[s * scrow + j]; }
        for (int t = 0; t < topk; t++) { same_l &= gl[s * lrow + 2 + t] == lst[s * lrow + 2 + t]; same_l2 &= gl2[s * lrow + 2 + t] == lstm[s * lrow + 2 + t]; }
    }
    char m[240]; snprintf(m, sizeof m, "index S %d pos %d ratio %d IH %d ID %d topk %d block %d/%d order %d: scores %d mask %d list %d masked list %d",
                          S, pos_base, ratio, IH, ID, topk, block, topb, order, same_sc, same_m, same_l, same_l2);
    ck(ok && same_sc && same_m && same_l && same_l2, m);
    for (CcBuf *b : {qb, kb, wb, scb, mb, lb, lb2, scb2}) T->free(b);
}

static void test_engram(int S, int H, int D) {
    const int kvrow = (H + 1) * D + 2, xrow = H * D + 3;
    const float eps = 1e-6f;
    std::vector<float> kv = fvec((size_t)S * kvrow + 1, 1.f), x = fvec((size_t)S * xrow + 4, 1.f), prm = fvec((size_t)2 * H * D + 6, 1.f);
    std::vector<float> ref = x;
    for (int s = 0; s < S; s++) {
        const float *k0 = &kv[1 + (size_t)s * kvrow], *value = k0 + (size_t)H * D;
        for (int c = 0; c < H; c++) {
            const float *key = k0 + (size_t)c * D, *qw = &prm[6 + (size_t)c * D], *kw = &prm[6 + (size_t)H * D + (size_t)c * D];
            float *st = &ref[4 + (size_t)s * xrow + (size_t)c * D];
            double ss = 0, ks = 0, dot = 0;
            for (int i = 0; i < D; i++) { ss += (double)st[i] * st[i]; ks += (double)key[i] * key[i]; dot += (double)st[i] * qw[i] * kw[i] * key[i]; }
            float rstd = (1.0f / sqrtf((float)(ss / D) + eps)) * (1.0f / sqrtf((float)(ks / D) + eps));
            float scaled = (float)dot * rstd / sqrtf((float)D), mag = fabsf(scaled);
            if (mag < 1e-6f) mag = 1e-6f;
            float sr = sqrtf(mag); if (scaled < 0) sr = -sr;
            float g = sigm(sr);
            for (int i = 0; i < D; i++) st[i] += g * value[i];
        }
    }
    CcBuf *kb = upf(kv), *xb = upf(x), *pb = upf(prm);
    CcDsEngram p = {S, H, D, 1, kvrow, 4, xrow, 6, 6 + H * D, eps};
    int ok = T->begin() && T->ds_engram(kb, pb, xb, &p) && T->submit(1);
    auto got = down<float>(xb, x.size());
    double e = relerr(got.data(), ref.data(), x.size(), 1e-3);
    char m[160]; snprintf(m, sizeof m, "engram S %d H %d D %d: rel err %.2e", S, H, D, e);
    ck(ok && e < 2e-4, m);   /* the block's tree sums against the reference's double sums */
    for (CcBuf *b : {kb, xb, pb}) T->free(b);
}

int main(void) {
    int dev = 0;
    if (!coli_cuda_init(&dev, 1)) { printf("no CUDA device: skipped\n"); return 0; }
    T = coli_cuda_chain_ops();
    if (!T || T->size < offsetof(ColiCudaChainOps, ds_engram) + sizeof T->ds_engram || !T->ds_engram) { printf("FAIL: the backend has no V4 ops\n"); return 1; }
    if (!T->init(0) || !T->ready()) { printf("FAIL: no chain context\n"); return 1; }
    printf("mhc\n");
    test_mhc(4, 64, 20); test_mhc(4, 5120 / 8, 20); test_mhc(2, 33, 3);
    printf("attention\n");
    test_attn(1, 4, 64, 8, 5, 13, 0); test_attn(3, 2, 512, 128, 40, 300, 0); test_attn(2, 3, 100, 6, 9, 20, 0);
    test_attn(2, 2, 64, 8, 12, 2900, 0); test_attn(3, 4, 64, 8, 5, 13, 1);
    test_attn(40, 64, 512, 128, 700, 640, 0); test_attn(9, 70, 128, 8, 90, 200, 0);
    test_attn(12, 8, 64, 8, 40, 2900, 0); test_attn(40, 64, 512, 128, 700, 640, 1);
    printf("rope\n"); test_rope(0); test_rope(1);
    printf("compress\n"); test_compress(4, 0, 0, 23); test_compress(4, 1, 1, 23); test_compress(128, 0, 1, 300);
    printf("index\n");
    test_index(1, 63, 4, 64, 128, 8, 4, 2, 0); test_index(5, 40, 4, 8, 16, 6, 3, 2, 1); test_index(3, 900, 4, 64, 128, 512, 16, 8, 0);
    test_index(2, 130, 1, 4, 8, 20, 5, 3, 1);
    printf("engram\n"); test_engram(3, 4, 64); test_engram(2, 4, 5120);
    T->shutdown();
    printf(fails ? "FAIL (%d)\n" : "PASS\n", fails);
    return fails != 0;
}
