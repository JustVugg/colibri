/* coli_cuda_expert_group_append (QT_GPU_MISS_FOLD): a second MXFP4 group queued behind the
 * one in flight, one take for both. Ten experts at Qwen3.8's shapes (2560 x 640), random
 * e2m1 nibbles and ue8m0 scales, are computed three ways and must give the same bits:
 *   A  one issue of every row, one take;
 *   B  an issue of the first experts, the rest appended (in one append, and in two), ONE take;
 *   C  issue + take of the first, issue + take of the rest (QT_GPU_MISS=2 without the fold).
 * Row mixes: one row an expert (a decode step) and several rows an expert (a verify's shared
 * entries, the kernels' 4-row tiles). B runs first on a fresh device context for the small
 * mix, so the first append also takes the regrow path (buffers sized by a smaller group).
 * Refusals: nothing in flight, more than 32 rows in all; both leave the pending rows intact.
 * Real kernels. Exit 0: all bit-exact. 1: a refusal or a mismatch. 77: no GPU. */
#include "../backend_cuda.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

enum { NE = 10, D = 2560, H = 640 };

static uint32_t rng_state = 0x2545F491u;
static uint32_t rnd(void) { rng_state ^= rng_state << 13; rng_state ^= rng_state >> 17; rng_state ^= rng_state << 5; return rng_state; }
static float frand(float lo, float hi) { return lo + (hi - lo) * ((rnd() >> 8) * (1.0f / 16777216.0f)); }

/* one MXFP4 matrix: O rows of I/2 nibble bytes, O x I/32 ue8m0 scale bytes (2^-7 .. 2^-4) */
static int upload(ColiCudaTensor **t, int I, int O) {
    std::vector<uint8_t> w((size_t)O * (I / 2)), s((size_t)O * (I / 32));
    for (auto &b : w) b = (uint8_t)rnd();
    for (auto &b : s) b = (uint8_t)(120 + rnd() % 4);
    return coli_cuda_tensor_upload(t, w.data(), (const float *)s.data(), 7, I, O, 0);
}

static ColiCudaTensor *tg[NE], *tu[NE], *td[NE];

/* rows[e] rows per expert; `first` experts issued, the rest appended in `parts` appends */
static int run_mix(const char *name, const int *rows, int first, int parts) {
    int total = 0, r_first = 0;
    for (int e = 0; e < NE; e++) { total += rows[e]; if (e < first) r_first += rows[e]; }
    std::vector<float> x((size_t)total * D);
    for (auto &v : x) v = frand(-1.f, 1.f);
    std::vector<float> ya((size_t)total * D), yb((size_t)total * D), yc((size_t)total * D);
    size_t bytes = ya.size() * sizeof(float);

    /* B first: on a fresh context the first append meets buffers sized for `first` rows */
    int b = coli_cuda_expert_group_issue(tg, tu, td, rows, first, x.data());
    int at = first, r_at = r_first;
    for (int p = 0; p < parts && b; p++) {
        int n = p == parts - 1 ? NE - at : (NE - at) / 2, r_n = 0;
        for (int e = at; e < at + n; e++) r_n += rows[e];
        b = coli_cuda_expert_group_append(tg + at, tu + at, td + at, rows + at, n, x.data() + (size_t)r_at * D);
        at += n; r_at += r_n;
    }
    const float *r = coli_cuda_expert_group_take(0);   /* also clears the group after a refused append */
    if (b && r) std::memcpy(yb.data(), r, bytes); else b = 0;

    int a = coli_cuda_expert_group_issue(tg, tu, td, rows, NE, x.data());
    r = a ? coli_cuda_expert_group_take(0) : nullptr;
    if (r) std::memcpy(ya.data(), r, bytes); else a = 0;

    int c = coli_cuda_expert_group_issue(tg, tu, td, rows, first, x.data());
    r = c ? coli_cuda_expert_group_take(0) : nullptr;
    if (r) std::memcpy(yc.data(), r, (size_t)r_first * D * sizeof(float)); else c = 0;
    if (c) {
        c = coli_cuda_expert_group_issue(tg + first, tu + first, td + first, rows + first, NE - first, x.data() + (size_t)r_first * D);
        r = c ? coli_cuda_expert_group_take(0) : nullptr;
        if (r) std::memcpy(yc.data() + (size_t)r_first * D, r, (size_t)(total - r_first) * D * sizeof(float)); else c = 0;
    }

    size_t dab = 0, dcb = 0; float maxv = 0.f;
    for (size_t i = 0; i < yb.size(); i++) {
        if (std::memcmp(&ya[i], &yb[i], sizeof(float))) dab++;
        if (std::memcmp(&yc[i], &yb[i], sizeof(float))) dcb++;
        float m = std::fabs(yb[i]); if (m > maxv) maxv = m;
    }
    int exact = a && b && c && dab == 0 && dcb == 0 && maxv > 0.f && std::isfinite(maxv);
    std::printf("%-34s %2d rows, %d issued + %d appended in %d: one-issue=%s append=%s two-takes=%s  "
                "append vs one issue: %zu differ, vs two takes: %zu differ  %s\n",
                name, total, r_first, total - r_first, parts, a ? "ok" : "REFUSED", b ? "ok" : "REFUSED", c ? "ok" : "REFUSED",
                dab, dcb, exact ? "BIT-EXACT" : "FAIL");
    return exact ? 0 : 1;
}

int main(void) {
    int dev = 0;
    if (!coli_cuda_init(&dev, 1)) { std::printf("expert group append: no CUDA device, skipped\n"); return 77; }
    int failed = 0;
    if (!coli_cuda_expert_group_append(nullptr, nullptr, nullptr, nullptr, 0, nullptr)) { std::printf("probe answered 0: FAIL\n"); failed++; }
    for (int e = 0; e < NE; e++)
        if (!upload(&tg[e], D, H) || !upload(&tu[e], D, H) || !upload(&td[e], H, D)) { std::printf("upload failed\n"); coli_cuda_shutdown(); return 1; }
    int one[NE]; for (int e = 0; e < NE; e++) one[e] = 1;
    std::vector<float> x((size_t)32 * D, 0.25f);
    if (coli_cuda_expert_group_append(tg, tu, td, one, 3, x.data())) { std::printf("append with nothing in flight accepted: FAIL\n"); failed++; }

    static const int verify[NE] = {3, 2, 1, 3, 1, 2, 3, 1, 1, 2};   /* 19 rows: tiles of 3, 2 and 1 */
    static const int wide[NE] = {4, 4, 4, 4, 2, 2, 5, 3, 2, 2};     /* 32 rows: a full group, a 5-row entry (two tiles) */
    failed += run_mix("decode step, 7 hits + 3 misses", one, 7, 1);
    failed += run_mix("decode step, 1 hit + 9 misses", one, 1, 1);
    failed += run_mix("decode step, 9 hits + 1 miss", one, 9, 1);
    failed += run_mix("decode step, 4 + 3 + 3", one, 4, 2);
    failed += run_mix("verify rows, 6 entries + 4", verify, 6, 1);
    failed += run_mix("verify rows, 3 entries + 3 + 4", verify, 3, 2);
    failed += run_mix("full group, 16 rows + 16", wide, 4, 1);

    /* past 32 rows: refused, and the pending group's rows are still its take's */
    std::vector<float> ref((size_t)32 * D), got((size_t)32 * D);
    for (auto &v : x) v = frand(-1.f, 1.f);
    int ok = coli_cuda_expert_group_issue(tg, tu, td, wide, NE, x.data());
    const float *r = ok ? coli_cuda_expert_group_take(0) : nullptr;
    if (r) std::memcpy(ref.data(), r, ref.size() * sizeof(float)); else ok = 0;
    ok = ok && coli_cuda_expert_group_issue(tg, tu, td, wide, NE, x.data());
    int refused = ok && !coli_cuda_expert_group_append(tg, tu, td, one, 1, x.data());
    r = ok ? coli_cuda_expert_group_take(0) : nullptr;
    if (r) std::memcpy(got.data(), r, got.size() * sizeof(float)); else ok = 0;
    int intact = ok && refused && !std::memcmp(ref.data(), got.data(), ref.size() * sizeof(float));
    std::printf("a 33rd row: %s, pending rows %s\n", refused ? "refused" : "ACCEPTED", intact ? "intact" : "CHANGED");
    failed += !intact;

    for (int e = 0; e < NE; e++) { coli_cuda_tensor_free(tg[e]); coli_cuda_tensor_free(tu[e]); coli_cuda_tensor_free(td[e]); }
    coli_cuda_shutdown();
    std::printf("expert group append: %s (%d failed)\n", failed ? "FAIL" : "all bit-exact", failed);
    return failed ? 1 : 0;
}
