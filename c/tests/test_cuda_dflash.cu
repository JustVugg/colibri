/* DFlash2's drafter's two ops on the CUDA chain (cuda_dflash.cuh) against the CPU's.
 * dfconv, the two-tap dynamic convolution:
 * blocks of 8 rows (a drafter's block) and of 1 (no predecessor at all), several blocks in a call, the
 * correction a group of 16 channels (interleaved and tiled), added or scaling, the corrections laid
 * out [tap][group] and [group][tap], rows with offsets and strides. Each output is two products and
 * their sum, rounded apart: the bits must be the CPU's. And a call that names its input as its output,
 * or reads past a buffer, is refused.
 * dftopk, the selector's candidates: the top 16 (and 1 and 32) of Qwen3.8's 248320 logits a row, best
 * first, ties to the lower column, rows with offsets and a stride, ties planted, a row of equal values.
 * dftap, the context rows: a copy (one stream) and Flash-Next's four streams summed, rows with offsets
 * and strides, the bits of the CPU's sum in order.
 * make CUDA=1 cuda-dflash-check */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include "../backend_cuda.h"
#include "../cuda_chain.h"

static const ColiCudaChainOps *T;
static int fails;
static void ck(int ok, const char *what) { printf("  %s %s\n", ok ? "ok  " : "FAIL", what); if (!ok) fails++; }
static unsigned g_rng = 2026;
static unsigned rnd() { g_rng = g_rng * 1103515245u + 12345u; return (g_rng >> 8) & 0xffffff; }
static float frnd() { return ((int)(rnd() % 2001) - 1000) / 1000.0f; }
static std::vector<float> fvec(size_t n, float scale) { std::vector<float> v(n); for (auto &x : v) x = frnd() * scale; return v; }
static CcBuf *upf(const std::vector<float> &v) {
    CcBuf *b = T->buf(v.empty() ? 4 : v.size() * 4, CC_DEV);
    if (!b || !T->begin() || !T->write(b, 0, v.data(), v.size() * 4) || !T->submit(1)) { printf("  FAIL upload\n"); fails++; }
    return b;
}
static std::vector<float> down(CcBuf *b, size_t n) {
    std::vector<float> v(n);
    if (!T->read(b, 0, v.data(), n * 4)) { printf("  FAIL read back\n"); fails++; }
    return v;
}

/* the CPU's: volatile keeps each product and the sum rounded apart (no fused multiply-add) */
static float ref_one(const CcDfConv &p, const float *x, const float *kp, const float *w, int r, int c) {
    const int G = p.C / p.gs, g = (p.flags & 1) ? c % G : c / p.gs;
    const float *kr = kp + p.kp_off + (size_t)r * p.kp_row + (size_t)g * p.kp_g;
    const float b0 = w[p.b_off + c], b1 = w[p.b_off + p.C + c];
    const float k0 = (p.flags & 2) ? b0 * (1.f + kr[0]) : b0 + kr[0];
    const float k1 = (p.flags & 2) ? b1 * (1.f + kr[p.kp_tap]) : b1 + kr[p.kp_tap];
    const float xv = x[p.x_off + (size_t)r * p.x_row + c], xp = r % p.B ? x[p.x_off + (size_t)(r - 1) * p.x_row + c] : 0.f;
    volatile float a = k0 * xv, b = k1 * xp;
    volatile float s = a + b;
    return s;
}

static void test_conv(const char *what, int n, int C, int B, int gs, int flags, int g_major, int x_pad, int y_pad) {
    const int G = C / gs;
    CcDfConv p = {};
    p.n = n; p.C = C; p.B = B; p.gs = gs; p.flags = flags;
    p.x_off = 5; p.x_row = C + x_pad;
    p.kp_off = 3; p.kp_row = 4 * G + 2;               /* a row: [pre, post] x [tap 0, tap 1] x G (or G x 2 x 2) */
    p.kp_tap = g_major ? 1 : G; p.kp_g = g_major ? 4 : 1;
    p.y_off = 7; p.y_row = C + y_pad; p.b_off = 11;
    std::vector<float> x = fvec((size_t)p.x_off + (size_t)n * p.x_row, 2.f);
    std::vector<float> kp = fvec((size_t)p.kp_off + (size_t)n * p.kp_row + 8, 0.1f);
    std::vector<float> w = fvec((size_t)p.b_off + 4 * (size_t)C, 1.f);
    for (int c = 0; c < C; c++) w[p.b_off + c] += 1.f;   /* the base's own tap near 1, as a trained drafter's */
    CcBuf *bx = upf(x), *bk = upf(kp), *bw = upf(w), *by = T->buf(((size_t)p.y_off + (size_t)n * p.y_row) * 4, CC_DEV);
    int ok = by && T->begin() && T->zero(by, 0, (size_t)p.y_off + (size_t)n * p.y_row);
    for (int post = 0; ok && post < 2; post++) {   /* the pre and the post kernels: kp_off and b_off moved */
        CcDfConv q = p;
        q.kp_off += post * (g_major ? 2 : 2 * G); q.b_off += post * 2 * C;
        ok = T->dfconv(bx, bk, bw, by, &q) && T->submit(1);
        std::vector<float> y = down(by, (size_t)p.y_off + (size_t)n * p.y_row);
        size_t bad = 0;
        for (int r = 0; r < n; r++) for (int c = 0; c < C; c++) {
            float want = ref_one(q, x.data(), kp.data(), w.data(), r, c), got = y[q.y_off + (size_t)r * q.y_row + c];
            if (memcmp(&want, &got, 4)) bad++;
        }
        char msg[200]; snprintf(msg, sizeof msg, "%s, %s: %zu of %d outputs differ from the CPU's bits", what, post ? "post" : "pre", bad, n * C);
        ck(ok && bad == 0, msg);
        ok = ok && T->begin();
    }
    if (ok) T->submit(1);
    T->free(bx); T->free(bk); T->free(bw); T->free(by);
}

static void test_topk(const char *what, int R, int V, int K, int ties, int splits = 64) {
    CcDfTopk p = {R, V, K, 9, V + 3, 5, 2 * K + 1, splits};
    std::vector<float> x = fvec((size_t)p.x_off + (size_t)R * p.x_row, 30.f);
    for (int r = 0; r < R && ties; r++)   /* the best value planted twice, and a block of equal ones */
        for (int j = 0; j < V; j += V / 7) x[p.x_off + (size_t)r * p.x_row + j] = ties == 2 ? 1.f : 31.f;
    if (ties == 2) for (int r = 0; r < R; r++) for (int j = 0; j < V; j++) x[p.x_off + (size_t)r * p.x_row + j] = 1.f;
    CcBuf *bx = upf(x), *bo = T->buf(((size_t)p.o_off + (size_t)R * p.o_row) * 4, CC_DEV);
    CcBuf *bs = T->buf((size_t)R * splits * 2 * K * 4, CC_DEV);
    int ok = bo && bs && T->begin() && T->dftopk(bx, bs, bo, &p) && T->submit(1);
    std::vector<float> o = down(bo, (size_t)p.o_off + (size_t)R * p.o_row);
    int bad = 0;
    for (int r = 0; r < R; r++) {
        std::vector<int> idx(V);
        for (int j = 0; j < V; j++) idx[j] = j;
        const float *xr = x.data() + p.x_off + (size_t)r * p.x_row;
        std::partial_sort(idx.begin(), idx.begin() + K, idx.end(), [&](int a, int b) { return xr[a] > xr[b] || (xr[a] == xr[b] && a < b); });
        for (int k = 0; k < K; k++) {
            float v = o[p.o_off + (size_t)r * p.o_row + k]; int id;
            float f = o[p.o_off + (size_t)r * p.o_row + K + k]; memcpy(&id, &f, 4);
            if (id != idx[k] || v != xr[idx[k]]) bad++;
        }
    }
    char msg[200]; snprintf(msg, sizeof msg, "%s: %d of %d picks differ", what, bad, R * K);
    ck(ok && bad == 0, msg);
    T->free(bx); T->free(bo); T->free(bs);
}

static void test_tap(const char *what, int n, int D, int C) {
    CcDfTap p = {n, D, C, 3, C * D + 5, 7, 5 * D + 2};
    std::vector<float> x = fvec((size_t)p.x_off + (size_t)n * p.x_row, 4.f);
    CcBuf *bx = upf(x), *by = T->buf(((size_t)p.y_off + (size_t)n * p.y_row) * 4, CC_DEV);
    int ok = by && T->begin() && T->dftap(bx, by, &p) && T->submit(1);
    std::vector<float> y = down(by, (size_t)p.y_off + (size_t)n * p.y_row);
    size_t bad = 0;
    for (int r = 0; r < n; r++) for (int c = 0; c < D; c++) {
        volatile float s = x[p.x_off + (size_t)r * p.x_row + c];
        for (int b = 1; b < C; b++) s = s + x[p.x_off + (size_t)r * p.x_row + (size_t)b * D + c];
        float want = s, got = y[p.y_off + (size_t)r * p.y_row + c];
        if (memcmp(&want, &got, 4)) bad++;
    }
    char msg[200]; snprintf(msg, sizeof msg, "%s: %zu of %d values differ", what, bad, n * D);
    ck(ok && bad == 0, msg);
    T->free(bx); T->free(by);
}

int main() {
    T = coli_cuda_chain_ops();
    if (!T || !T->init(0) || !T->ready()) { printf("cuda-dflash-check: no CUDA chain on this machine\n"); return 1; }
    ck(T->size >= offsetof(ColiCudaChainOps, dfconv) + sizeof T->dfconv && T->dflash && T->dflash(), "the module has the op");
    test_conv("8-row blocks, Qwen3.8-27B's width, interleaved groups, added", 24, 5120, 8, 16, 0, 0, 0, 0);
    test_conv("8-row blocks, Flash-Next's width, offsets and strides", 16, 2560, 8, 16, 0, 0, 13, 9);
    test_conv("one block of 8, groups tiled, scaling", 8, 512, 8, 16, 3, 0, 3, 1);
    test_conv("rows alone (B 1: no predecessor), [group][tap] layout", 10, 256, 1, 16, 0, 1, 0, 2);
    test_conv("a block cut short (7 rows of an 8-row block)", 7, 128, 8, 16, 2, 1, 1, 0);
    test_tap("a copy of 8 rows (Qwen3.8-27B, one stream)", 8, 5120, 1);
    test_tap("Flash-Next's four streams summed, 33 rows", 33, 2560, 4);
    test_topk("top 16 of 7 rows of 248320 (a drafter's block)", 7, 248320, 16, 0);
    test_topk("top 16, the best planted at several columns", 7, 248320, 16, 1);
    test_topk("top 16 of rows all equal: the lowest columns", 3, 5000, 16, 2);
    test_topk("top 1 and top 32", 2, 248320, 1, 0);
    test_topk("top 32", 2, 248320, 32, 1, 32);
    test_topk("a short row (V 40, K 16; slices empty past it)", 4, 40, 16, 0);
    test_topk("one slice (splits 1)", 3, 248320, 16, 1, 1);
    {
        std::vector<float> v = fvec(64 * 4, 1.f);
        CcBuf *b = upf(v), *k = upf(v), *w = upf(v);
        CcDfConv p = {};
        p.n = 2; p.C = 64; p.B = 8; p.gs = 16; p.x_row = 64; p.kp_row = 16; p.kp_tap = 4; p.kp_g = 1; p.y_row = 64;
        ck(T->begin() && !T->dfconv(b, k, w, b, &p), "the input as the output is refused");
        CcDfConv q = p; q.n = 5;   /* 5 rows of 64 floats past a buffer of 256 */
        ck(!T->dfconv(b, k, w, upf(std::vector<float>(64)), &q), "a call past its buffers is refused");
        T->submit(1);
    }
    printf("%s\n", fails ? "FAILED" : "all ok");
    return fails ? 1 : 0;
}
