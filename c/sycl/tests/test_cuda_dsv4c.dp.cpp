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
 * make CUDA=1 cuda-dsv4c-check (the references in tests/cuda_dsv4c_ref.c).
 * The SYCL build of tests/test_cuda_dsv4c.cu (sycl/cuda_dsv4c.dp.hpp, the .cuh's port): the same
 * cases and bounds, the ops the ABI's; only the includes and the device choice differ (--device N /
 * COLI_GPUS=N, --list prints every GPU's profile and exits: sycl_test_args.h). Needs a device:
 * make SYCL=1 sycl-dsv4c-check, or make SYCL=1 cuda_dsv4c_test and run the binary. */
#include <cstdio>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include "../../backend_cuda.h"
#include "../../cuda_chain.h"
#include "../coli_sycl.h"
#include "sycl_test_args.h"
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
    {   /* the split with H a constant (ccd_mhc_split_t) against the run-time-H kernel: the same bits */
        CcBuf *hg = T->buf((size_t)S * hrow * 4, CC_DEV);
        setenv("COLI_SYCL_MHC_GENERIC", "1", 1);
        int okg = T->begin() && T->mhc(CC_MHC_SPLIT, xb, mb, hg, pb, NULL, &sp) && T->submit(1);
        unsetenv("COLI_SYCL_MHC_GENERIC");
        auto hpg = down<float>(hg, (size_t)S * hrow);
        size_t diff = 0;
        for (size_t i = 0; i < hp.size(); i++) diff += memcmp(&hp[i], &hpg[i], 4) != 0;
        snprintf(m, sizeof m, "mhc split H %d iters %d: constant-H kernel vs generic, %zu of %zu values differ", H, iters, diff, hp.size());
        ck(okg && diff == 0, m);
        T->free(hg);
    }
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

/* ds_attn_rows: n decode rows of different conversations (each its own window and compressed rows) in one call, against
 * n one-row ds_attn calls (the same kernel: bit for bit) and against the old split kernel (COLI_CUDA_DSV4_ATTN_V2=0:
 * the split's quality bound) */
static void test_attn_rows(int n, int H, int hd, int nwin, int ncmp, int cnt) {
    const int qrow = H * hd + 3, lrow = cnt + 2, orow = H * hd + 3;
    std::vector<float> q = fvec((size_t)n * qrow, 1.f), win = fvec((size_t)n * nwin * hd + 5, 1.f), cmp = fvec((size_t)n * ncmp * hd + 7, 1.f),
                       sink = fvec((size_t)H + 2, 2.f);
    std::vector<int> list((size_t)n * lrow), wo(n), co(n);
    for (int r = 0; r < n; r++) { wo[r] = 5 + r * nwin * hd; co[r] = 7 + r * ncmp * hd; }
    for (int r = 0; r < n; r++)
        for (int j = 0; j < lrow; j++) { int x = (int)(rnd() % 5); list[r * lrow + j] = x == 0 ? -1 : x < 3 ? (int)(rnd() % (unsigned)nwin) : nwin + (int)(rnd() % (unsigned)ncmp); }
    CcBuf *qb = upf(q), *wb = upf(win), *cb = upf(cmp), *sb = upf(sink), *lb = up(list.data(), list.size() * 4);
    CcBuf *oa = T->buf((size_t)n * orow * 4, CC_DEV), *ob = T->buf((size_t)n * orow * 4, CC_DEV), *oc = T->buf((size_t)n * orow * 4, CC_DEV);
    char m[200];
    if (!(T->size >= offsetof(ColiCudaChainOps, ds_attn_rows) + sizeof T->ds_attn_rows && T->ds_attn_rows)) { ck(0, "ds_attn_rows in the table"); return; }
    CcDsAttn p = {1, H, hd, cnt, 1, lrow, nwin, wo[0], co[0], 1, qrow, 2, orow, 1, 0, 1.f / sqrtf((float)hd)};
    int ok = T->begin() && T->ds_attn_rows(qb, wb, cb, lb, sb, oa, &p, wo.data(), co.data(), n) == 1 && T->submit(1);
    ok = ok && T->begin();
    for (int r = 0; ok && r < n; r++) {
        CcDsAttn a = p; a.l_off = 1 + r * lrow; a.q_off = 1 + r * qrow; a.o_off = 2 + r * orow; a.w_off = wo[r]; a.c_off = co[r];
        ok = T->ds_attn(qb, wb, cb, lb, sb, ob, &a);
    }
    ok = ok && T->submit(1);
    setenv("COLI_CUDA_DSV4_ATTN_V2", "0", 1);
    ok = ok && T->begin();
    for (int r = 0; ok && r < n; r++) {
        CcDsAttn a = p; a.l_off = 1 + r * lrow; a.q_off = 1 + r * qrow; a.o_off = 2 + r * orow; a.w_off = wo[r]; a.c_off = co[r];
        ok = T->ds_attn(qb, wb, cb, lb, sb, oc, &a);
    }
    ok = ok && T->submit(1);
    unsetenv("COLI_CUDA_DSV4_ATTN_V2");
    auto ga = down<float>(oa, (size_t)n * orow), gb = down<float>(ob, (size_t)n * orow), gc = down<float>(oc, (size_t)n * orow);
    size_t diff = 0; double e = 0;
    for (int r = 0; r < n; r++) {
        for (int i = 0; i < H * hd; i++) diff += memcmp(&ga[(size_t)r * orow + 2 + i], &gb[(size_t)r * orow + 2 + i], 4) != 0;
        double x = relerr(&ga[(size_t)r * orow + 2], &gc[(size_t)r * orow + 2], (size_t)H * hd, 1e-3); if (x > e) e = x;
    }
    snprintf(m, sizeof m, "attn rows n %d H %d hd %d cnt %d: one call vs a call a row %zu values differ; vs the old split rel err %.2e", n, H, hd, cnt, diff, e);
    ck(ok && diff == 0 && e < 2e-4, m);
    for (CcBuf *b : {qb, wb, cb, sb, lb, oa, ob, oc}) T->free(b);
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

/* ---- the layer frame's merged and faster small kernels, bit for bit against the kernels they stand for ---- */
#include <chrono>
static double now_us() { return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
/* cc_norm's rows of 2049..5120 through the registers' path (cc_norm_k<20>) against the loops' (COLI_SYCL_NORM_NV=8) */
static void test_norm_nv(int rows, int D, int flags) {
    std::vector<float> x = fvec((size_t)rows * D, 2.f), w = fvec((size_t)D, 1.f);
    CcBuf *xb = upf(x), *wb = upf(w), *ya = T->buf((size_t)rows * D * 4, CC_DEV), *yb = T->buf((size_t)rows * D * 4, CC_DEV);
    CcNorm p = {rows, D, 1, 0, D, D, 0, D, D, 0, 0, flags, 1e-6f, 1.f};
    setenv("COLI_SYCL_NORM_NV", "8", 1);
    int ok = T->begin() && T->norm(xb, wb, ya, &p) && T->submit(1);
    unsetenv("COLI_SYCL_NORM_NV");
    ok = ok && T->begin() && T->norm(xb, wb, yb, &p) && T->submit(1);
    auto a = down<float>(ya, (size_t)rows * D), b = down<float>(yb, (size_t)rows * D);
    size_t diff = 0, nz = 0;
    for (size_t i = 0; i < a.size(); i++) { diff += memcmp(&a[i], &b[i], 4) != 0; nz += a[i] != 0.f; }
    char m[200]; snprintf(m, sizeof m, "norm rows %d D %d flags %d: the registers' path vs the loops', %zu of %zu values differ (%zu non-zero)",
                          rows, D, flags, diff, a.size(), nz);
    ck(ok && diff == 0 && nz > a.size() / 2, m);
    for (CcBuf *q : {xb, wb, ya, yb}) T->free(q);
}
static int have_moe_tail() { return T->size >= offsetof(ColiCudaChainOps, moe_tail) + sizeof T->moe_tail && T->moe_tail; }
/* a chain-MoE layer's last frame: cc_moe parts 2, ew ADD, mhc POST against moe_tail's one launch (y, br and the streams) */
struct TailCase {
    int S, K, D, I, H, E, hrow;
    CcBuf *tab, *ids, *w, *hid, *hy, *ds, *xs, *hp, *y[2], *br[2], *xn[2];
    CcMoe p; CcMhc po; CcEw add;
};
static TailCase tail_make(int S, int K, int D, int I, int H, int hoff_on) {
    TailCase t; memset(&t, 0, sizeof t);
    t.S = S; t.K = K; t.D = D; t.I = I; t.H = H; t.E = 24; t.hrow = 2 * H + H * H;
    const int toff = 5;
    std::vector<CcExpert> tab((size_t)toff + t.E);
    memset(tab.data(), 0, tab.size() * sizeof(CcExpert));
    for (int e = 0; e < t.E; e++) if (e % 3 != 1) tab[toff + e].g = (const void *)(uintptr_t)64;   /* held: parts 2 reads g as a flag only */
    std::vector<int> ids((size_t)S * K);
    for (int s = 0; s < S; s++) for (int k = 0; k < K; k++) ids[s * K + k] = (s * 7 + k * 5) % 11 == 3 ? -1 : (int)(rnd() % t.E);
    const size_t hf = cc_moe_hid_floats(S, K, D, I, t.E);
    std::vector<float> w = fvec((size_t)S * K, 1.f), hid = fvec(hf, 3.f), hy = fvec((size_t)S * D, 2.f), ds = fvec((size_t)S * D, 2.f),
                       xs = fvec((size_t)S * H * D, 2.f), hp = fvec((size_t)S * t.hrow, 1.f);
    t.tab = up(tab.data(), tab.size() * sizeof(CcExpert)); t.ids = up(ids.data(), ids.size() * 4);
    t.w = upf(w); t.hid = upf(hid); t.hy = upf(hy); t.ds = upf(ds); t.xs = upf(xs); t.hp = upf(hp);
    for (int v = 0; v < 2; v++) {
        t.y[v] = T->buf((size_t)S * D * 4, CC_DEV); t.br[v] = T->buf((size_t)S * D * 4, CC_DEV); t.xn[v] = T->buf((size_t)S * H * D * 4, CC_DEV);
    }
    CcMoe p; memset(&p, 0, sizeof p);
    p.S = S; p.K = K; p.D = D; p.I = I; p.fmt = 7; p.gs = 32; p.act = 1; p.parts = 2; p.t_off = toff; p.x_row = D; p.i_row = K; p.w_row = K;
    p.y_row = D; p.h_off = hoff_on ? 0 : -1; p.E = t.E; p.act_lim = 10.f;
    t.p = p;
    CcMhc po = {S, H, D, 20, 0, H * D, 0, D, 0, t.hrow, 0, H * D, 0, 0, 1e-6f, 1e-6f, 0.f};
    t.po = po;
    CcEw add = {CC_EW_ADD, S * D, D, 1, 0, 1, 0, 0, 0, 0, 0, 1.f};
    t.add = add;
    return t;
}
static int tail_three(TailCase &t, int v) {
    return T->moe(t.tab, NULL, t.ids, t.w, t.hid, t.y[v], t.hy, &t.p) && T->ew(t.br[v], t.y[v], t.ds, NULL, NULL, &t.add) &&
           T->mhc(CC_MHC_POST, t.xs, t.br[v], t.hp, NULL, t.xn[v], &t.po);
}
static int tail_one(TailCase &t, int v) {
    return T->moe_tail(t.tab, t.ids, t.w, t.hid, t.hy, t.y[v], t.ds, t.br[v], t.xs, t.hp, t.xn[v], &t.p, &t.po) == 1;
}
static void tail_free(TailCase &t) {
    for (CcBuf *q : {t.tab, t.ids, t.w, t.hid, t.hy, t.ds, t.xs, t.hp, t.y[0], t.y[1], t.br[0], t.br[1], t.xn[0], t.xn[1]}) T->free(q);
}
static size_t bitdiff(CcBuf *a, CcBuf *b, size_t n, size_t *nz) {
    auto x = down<float>(a, n), y = down<float>(b, n);
    size_t d = 0;
    for (size_t i = 0; i < n; i++) { d += memcmp(&x[i], &y[i], 4) != 0; if (nz) *nz += x[i] != 0.f; }
    return d;
}
/* the split's sum of squares with its loads sixteen at a time (ccd_mhc_split_sg<H, 1>) against the loop (COLI_SYCL_MHC_BATCH=0) */
static void test_mhc_batch(int S, int H, int D, int iters) {
    const int HD = H * D, nm = (2 + H) * H, hrow = 2 * H + H * H;
    std::vector<float> x = fvec((size_t)S * HD, 1.f), mix = fvec((size_t)S * nm, 2.f), prm(3 + nm);
    for (int i = 0; i < 3; i++) prm[i] = 0.5f + (rnd() % 100) / 100.f;
    for (int i = 0; i < nm; i++) prm[3 + i] = frnd() * 0.5f;
    CcBuf *xb = upf(x), *mb = upf(mix), *pb = upf(prm), *ha = T->buf((size_t)S * hrow * 4, CC_DEV), *hb = T->buf((size_t)S * hrow * 4, CC_DEV);
    CcMhc sp = {S, H, D, iters, 0, HD, 0, nm, 0, hrow, 0, D, 0, 0, 1e-5f, 1e-6f, 0.f};
    setenv("COLI_SYCL_MHC_BATCH", "0", 1);
    int ok = T->begin() && T->mhc(CC_MHC_SPLIT, xb, mb, ha, pb, NULL, &sp) && T->submit(1);
    unsetenv("COLI_SYCL_MHC_BATCH");
    ok = ok && T->begin() && T->mhc(CC_MHC_SPLIT, xb, mb, hb, pb, NULL, &sp) && T->submit(1);
    size_t nz = 0;
    const size_t d = bitdiff(ha, hb, (size_t)S * hrow, &nz);
    char m[200]; snprintf(m, sizeof m, "mhc split S %d H %d D %d iters %d: loads sixteen at a time vs the loop, %zu of %zu values differ", S, H, D, iters, d, (size_t)S * hrow);
    ck(ok && d == 0 && nz > (size_t)S * hrow / 2, m);
    for (CcBuf *q : {xb, mb, pb, ha, hb}) T->free(q);
}
static void test_moe_tail(int S, int K, int D, int I, int H, int hoff_on) {
    char m[240];
    if (!have_moe_tail()) { snprintf(m, sizeof m, "moe tail S %d: the backend has no moe_tail", S); ck(0, m); return; }
    TailCase t = tail_make(S, K, D, I, H, hoff_on);
    int ok = T->begin() && tail_three(t, 0) && T->submit(1);
    ok = ok && T->begin() && tail_one(t, 1) && T->submit(1);
    size_t nz = 0;
    const size_t dy = bitdiff(t.y[0], t.y[1], (size_t)S * D, &nz), db = bitdiff(t.br[0], t.br[1], (size_t)S * D, &nz),
                 dx = bitdiff(t.xn[0], t.xn[1], (size_t)S * H * D, &nz);
    snprintf(m, sizeof m, "moe tail S %d K %d D %d H %d host rows %d: one launch vs moe sum, add, mhc post: y %zu, br %zu, streams %zu values "
                          "differ (%zu non-zero of %zu)", S, K, D, H, hoff_on, dy, db, dx, nz, (size_t)S * D * (2 + H));
    ck(ok && !dy && !db && !dx && nz > (size_t)S * D * (2 + H) / 2, m);
    setenv("COLI_SYCL_MOE_TAIL", "0", 1);   /* the switch: not taken, nothing launched */
    const int r = T->begin() ? T->moe_tail(t.tab, t.ids, t.w, t.hid, t.hy, t.y[1], t.ds, t.br[1], t.xs, t.hp, t.xn[1], &t.p, &t.po) : 0;
    unsetenv("COLI_SYCL_MOE_TAIL");
    ok = T->submit(1);
    snprintf(m, sizeof m, "moe tail S %d: COLI_SYCL_MOE_TAIL=0 is not taken (%d)", S, r);
    ck(ok && r == -1, m);
    tail_free(t);
}
/* mark / reached: a frame left running, its mark after the first ops; the host polls: the rows those ops wrote to DOWN
 * memory are there when the mark is reached, and the frame's later ops were not held (the frame ends after it) */
static void test_mark(int later) {
    char m[240];
    if (T->size < offsetof(ColiCudaChainOps, reached) + sizeof T->reached || !T->mark || !T->reached) { ck(0, "mark: the backend has no mark / reached"); return; }
    const int D = 5120, H = 4, HD = H * D, hrow = 2 * H + H * H;
    std::vector<float> x = fvec((size_t)HD, 1.f), hp = fvec(hrow, 1.f), br = fvec(D, 1.f);
    CcBuf *xb = upf(x), *hb = upf(hp), *bb = upf(br), *dn = T->buf((size_t)HD * 4, CC_DOWN), *yb = T->buf((size_t)HD * 4, CC_DEV);
    CcMhc po = {1, H, D, 20, 0, HD, 0, D, 0, hrow, 0, HD, 0, 0, 1e-6f, 1e-6f, 0.f};
    int bad = 0, never = 0; double t_mark = 0, t_end = 0; long polls = 0;
    const int reps = 50;
    for (int rep = 0; rep < reps && !bad && !never; rep++) {
        float *d = (float *)T->ptr(dn);
        memset(d, 0, (size_t)HD * 4);
        std::vector<float> xr = fvec((size_t)HD, 1.f);
        int ok = T->begin() && T->write(xb, 0, xr.data(), (size_t)HD * 4) && T->copy(dn, 0, xb, 0, (size_t)HD) && T->mark();
        for (int i = 0; ok && i < later; i++) ok = T->mhc(CC_MHC_POST, xb, bb, hb, NULL, yb, &po);
        ok = ok && T->submit(0);
        const double t0 = now_us();
        int r = 0;
        while (ok && (r = T->reached()) == 0) { polls++; if (now_us() - t0 > 5e6) { never = 1; break; } }
        const double t1 = now_us();
        if (!ok || r != 1 || memcmp(d, xr.data(), (size_t)HD * 4)) bad = 1;   /* at the mark: the copy's rows are down */
        if (!T->finish()) bad = 1;
        t_mark += t1 - t0; t_end += now_us() - t0;
    }
    snprintf(m, sizeof m, "mark with %d ops after it: reached %.1f us after the submit (%ld polls a frame), the frame done after %.1f us; "
                          "the rows before the mark down at the mark", later, t_mark / reps, polls / reps, t_end / reps);
    ck(!bad && !never, m);
    for (CcBuf *q : {xb, hb, bb, dn, yb}) T->free(q);
}
/* Times, the card's (a line each, no verdict): `queued` = n launches in one frame, waited once (the device's time a
 * launch with the next one already submitted); `live` = a frame a repetition, waited each time (what a frame costs
 * when the card is idle as it is submitted: the layer's last frame after the host's misses) */
template <class F> static void time_queued(const char *what, int n, F f) {
    int ok = T->begin();
    for (int i = 0; ok && i < 16; i++) ok = f();
    ok = ok && T->submit(1) && T->begin();
    const double t0 = now_us();
    for (int i = 0; ok && i < n; i++) ok = f();
    ok = ok && T->submit(1);
    printf("  FRAME-TIME %-58s queued %7.1f us%s\n", what, (now_us() - t0) / n, ok ? "" : " (FAILED)");
}
template <class F> static void time_live(const char *what, int n, F f) {
    int ok = 1;
    for (int i = 0; ok && i < 16; i++) ok = T->begin() && f() && T->submit(1);
    const double t0 = now_us();
    for (int i = 0; ok && i < n; i++) ok = T->begin() && f() && T->submit(1);
    printf("  FRAME-TIME %-58s live   %7.1f us%s\n", what, (now_us() - t0) / n, ok ? "" : " (FAILED)");
}
static void frame_times() {
    const int D = 5120, H = 4, HD = H * D, nm = (2 + H) * H, hrow = 2 * H + H * H, N = 400;
    for (int S : {1, 4}) {
        char m[120];
        std::vector<float> x = fvec((size_t)S * HD, 1.f), w = fvec((size_t)D, 1.f), mix = fvec((size_t)S * nm, 2.f), prm = fvec(3 + nm, 0.5f);
        CcBuf *xb = upf(x), *wb = upf(w), *mb = upf(mix), *pb = upf(prm), *yb = T->buf((size_t)S * HD * 4, CC_DEV),
              *hb = T->buf((size_t)S * hrow * 4, CC_DEV), *cb = T->buf((size_t)S * D * 4, CC_DEV);
        CcNorm np = {S, D, 1, 0, D, D, 0, D, D, 0, 0, 0, 1e-6f, 1.f};
        CcNorm ns = {S * 10, 512, 10, 0, D, 512, 0, D, 512, 0, 0, 0, 1e-6f, 1.f};   /* ten rows of 512 a row: a short norm */
        CcMhc sp = {S, H, D, 20, 0, HD, 0, nm, 0, hrow, 0, D, 0, 0, 1e-6f, 1e-6f, 0.f};
        CcMhc po = {S, H, D, 20, 0, HD, 0, D, 0, hrow, 0, HD, 0, 0, 1e-6f, 1e-6f, 0.f};
        setenv("COLI_SYCL_NORM_NV", "8", 1);
        snprintf(m, sizeof m, "norm D 5120 rows %d, the loops (COLI_SYCL_NORM_NV=8)", S);
        time_queued(m, N, [&] { return T->norm(xb, wb, yb, &np); });
        unsetenv("COLI_SYCL_NORM_NV");
        snprintf(m, sizeof m, "norm D 5120 rows %d, the registers", S);
        time_queued(m, N, [&] { return T->norm(xb, wb, yb, &np); });
        snprintf(m, sizeof m, "norm D 512 x 10 a row, rows %d", S);
        time_queued(m, N, [&] { return T->norm(xb, wb, yb, &ns); });
        snprintf(m, sizeof m, "mhc split H 4 D 5120 iters 20 rows %d", S);
        time_queued(m, N, [&] { return T->mhc(CC_MHC_SPLIT, xb, mb, hb, pb, NULL, &sp); });
        setenv("COLI_SYCL_MHC_BATCH", "0", 1);
        snprintf(m, sizeof m, "mhc split H 4 D 5120 iters 20 rows %d, the loop (BATCH=0)", S);
        time_queued(m, N, [&] { return T->mhc(CC_MHC_SPLIT, xb, mb, hb, pb, NULL, &sp); });
        unsetenv("COLI_SYCL_MHC_BATCH");
        setenv("COLI_SYCL_MHC_GENERIC", "1", 1);
        snprintf(m, sizeof m, "mhc split H 4 D 5120 iters 20 rows %d, generic", S);
        time_queued(m, N, [&] { return T->mhc(CC_MHC_SPLIT, xb, mb, hb, pb, NULL, &sp); });
        unsetenv("COLI_SYCL_MHC_GENERIC");
        CcMhc sp1 = sp; sp1.iters = 1;
        snprintf(m, sizeof m, "mhc split H 4 D 5120 iters 1 rows %d", S);
        time_queued(m, N, [&] { return T->mhc(CC_MHC_SPLIT, xb, mb, hb, pb, NULL, &sp1); });
        snprintf(m, sizeof m, "mhc collapse rows %d", S);
        time_queued(m, N, [&] { return T->mhc(CC_MHC_COLLAPSE, xb, NULL, hb, NULL, cb, &sp); });
        snprintf(m, sizeof m, "mhc post rows %d", S);
        time_queued(m, N, [&] { return T->mhc(CC_MHC_POST, xb, cb, hb, NULL, yb, &po); });
        snprintf(m, sizeof m, "copy %d floats", S * HD);
        time_queued(m, N, [&] { return T->copy(yb, 0, xb, 0, (size_t)S * HD); });
        if (have_moe_tail()) {
            TailCase t = tail_make(S, 6, D, 2304, H, 1);
            snprintf(m, sizeof m, "last frame rows %d: moe sum, add, mhc post", S);
            time_queued(m, N, [&] { return tail_three(t, 0); });
            time_live(m, N, [&] { return tail_three(t, 0); });
            snprintf(m, sizeof m, "last frame rows %d: moe_tail", S);
            time_queued(m, N, [&] { return tail_one(t, 1); });
            time_live(m, N, [&] { return tail_one(t, 1); });
            tail_free(t);
        }
        for (CcBuf *q : {xb, wb, mb, pb, yb, hb, cb}) T->free(q);
    }
}

int main(int argc, char **argv) {
    int devices[COLI_CUDA_MAX_DEVICES];
    if (sycl_test_devices(argc, argv, devices, COLI_CUDA_MAX_DEVICES) < 0) return 0;   /* --list */
    int dev = devices[0];
    if (!coli_cuda_init(&dev, 1)) { printf("no GPU (device %d): skipped\n", dev); return 0; }
    T = coli_cuda_chain_ops();
    if (!T || T->size < offsetof(ColiCudaChainOps, ds_engram) + sizeof T->ds_engram || !T->ds_engram) { printf("FAIL: the backend has no V4 ops\n"); return 1; }
    if (!T->init(dev) || !T->ready()) { printf("FAIL: no chain context\n"); return 1; }
    printf("mhc\n");
    test_mhc(4, 64, 20); test_mhc(4, 5120 / 8, 20); test_mhc(2, 33, 3);
    printf("attention\n");
    test_attn(1, 4, 64, 8, 5, 13, 0); test_attn(3, 2, 512, 128, 40, 300, 0); test_attn(2, 3, 100, 6, 9, 20, 0);
    test_attn(2, 2, 64, 8, 12, 2900, 0); test_attn(3, 4, 64, 8, 5, 13, 1);
    test_attn(40, 64, 512, 128, 700, 640, 0); test_attn(9, 70, 128, 8, 90, 200, 0);
    test_attn(12, 8, 64, 8, 40, 2900, 0); test_attn(40, 64, 512, 128, 700, 640, 1);
    /* a decode row's split over its list (S 1, flags 0): DeepSeek V4.1's shape, an odd one, a list with holes */
    test_attn(1, 64, 512, 128, 700, 640, 0); test_attn(1, 7, 96, 8, 300, 200, 0); test_attn(1, 64, 512, 128, 2000, 1500, 0);
    printf("rope\n"); test_rope(0); test_rope(1);
    printf("compress\n"); test_compress(4, 0, 0, 23); test_compress(4, 1, 1, 23); test_compress(128, 0, 1, 300);
    printf("index\n");
    test_index(1, 63, 4, 64, 128, 8, 4, 2, 0); test_index(5, 40, 4, 8, 16, 6, 3, 2, 1); test_index(3, 900, 4, 64, 128, 512, 16, 8, 0);
    test_index(2, 130, 1, 4, 8, 20, 5, 3, 1);
    printf("engram\n"); test_engram(3, 4, 64); test_engram(2, 4, 5120);
    /* last, so the cases above draw the same random numbers as before: the constant-H split at H 8 and the generic H 3 */
    printf("mhc (more)\n"); test_mhc(8, 40, 20); test_mhc(3, 17, 5);
    printf("attention rows\n"); test_attn_rows(4, 64, 512, 128, 700, 640); test_attn_rows(3, 7, 96, 8, 300, 200); test_attn_rows(2, 64, 512, 128, 2000, 1500);
    /* the layer-frame kernels, after everything the suite had (the same random numbers for those) */
    printf("norm, the registers' path\n");
    for (int fl : {0, CC_NORM_ADD1, CC_NORM_NOW, CC_NORM_L2, CC_NORM_PRESCALE}) test_norm_nv(3, 5120, fl);
    test_norm_nv(4, 2049, 0); test_norm_nv(2, 4100, 0); test_norm_nv(1, 5119, CC_NORM_ADD1); test_norm_nv(5, 3000, CC_NORM_PRESCALE);
    printf("mhc split, loads sixteen at a time\n");
    test_mhc_batch(1, 4, 5120, 20); test_mhc_batch(4, 4, 5120, 20); test_mhc_batch(3, 4, 5120 / 8, 20); test_mhc_batch(2, 2, 33, 3); test_mhc_batch(3, 4, 1000, 7);
    printf("the last frame in one launch\n");
    test_moe_tail(1, 6, 5120, 2304, 4, 1); test_moe_tail(4, 6, 5120, 2304, 4, 1); test_moe_tail(3, 6, 5120, 2304, 4, 1);
    test_moe_tail(2, 9, 320, 96, 2, 1); test_moe_tail(4, 6, 5120, 2304, 4, 0); test_moe_tail(8, 3, 1056, 64, 8, 1);
    printf("a mark the host polls\n");
    test_mark(0); test_mark(40); test_mark(400);
    if (!getenv("DSV4C_NO_TIMES")) { printf("times\n"); frame_times(); }
    T->shutdown();
    printf(fails ? "FAIL (%d)\n" : "PASS\n", fails);
    return fails != 0;
}
