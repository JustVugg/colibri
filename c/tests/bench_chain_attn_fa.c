/* Microbenchmark: DeepSeek V4/V4.1's sparse attention (vkc_dsv4_attn) on the flash tile
 * (COLI_VK_CHAIN_ATTN_FA, chain_dsv4_fa.comp) against mode 0's scalar two-pass kernel,
 * at AG §4's geometry — the model's decode shape S 1 / H 64 / hd 512 / window 128 /
 * pool 640 / cnt 768 (every list entry live: the window whole, the pool whole), and the
 * prefill chunk of the same rows at S 256 (COLI_VK_BENCH_ROWS). A rep is one
 * frame of N recorded dispatches (COLI_VK_BENCH_N; 200 at S 1, 3 on the chunk
 * arms — mode 0's chunk dispatch runs ~0.3 s, and this box's amdgpu hard-recovers
 * a submission past ~2-4 s of GPU work (ENVIRONMENT's COLI_VK_DMOE_ROWS_MAX row,
 * the K1/P1 boundary), so the default keeps one frame under ~1 s): the best of 3
 * reps after a one-dispatch warm-up frame that also grows arm 1's f16 mirror
 * scratch, whose steady state re-gathers every dispatch.
 * The lever is set per arm around the recorded frame; VkcStats.attn_fa must move by the
 * dispatch count on the arms and not at all on mode 0 — a device without the tile says
 * so instead of reporting a 1x that is really mode 0.
 *
 * NOT a unit test: test_vk_chain's test_ds_attn_fa gates the tile's bits. Build on
 * demand: make tests/bench_chain_attn_fa VK=1
 * Run from c/:  ./tests/bench_chain_attn_fa [path/to/shaders]
 *               COLI_VK_BENCH_ROWS=256 ./tests/bench_chain_attn_fa   (the chunk case) */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "../backend_vulkan.h"
#include "../vk_chain.h"

static unsigned rng = 20261025u;
static unsigned rnd(void) { rng = rng * 1103515245u + 12345u; return rng >> 8; }
static float *fvec(size_t n) { float *v = malloc(n * sizeof *v); for (size_t i = 0; i < n; i++) v[i] = (float)((int)(rnd() % 2001) - 1000) / 1000.0f; return v; }
static double bnow(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec / 1e9; }

static VkcBuf *qb, *wb, *cb, *sb, *lb, *ob;
static VkcDsAttn P;

/* one arm's per-dispatch time: the warm-up frame, then the best of `reps` frames of N
 * recorded dispatches, timed at the submit (the fence wait covers the device's work) */
static double arm(const char *lever, int N, int reps, unsigned long long *fa) {
    if (lever) setenv("COLI_VK_CHAIN_ATTN_FA", lever, 1); else unsetenv("COLI_VK_CHAIN_ATTN_FA");
    VkcStats s0, s1; vkc_stats(&s0);
    int ok = vkc_begin();
    ok = ok && vkc_dsv4_attn(qb, wb, cb, lb, sb, ob, &P);   /* the warm-up: the pipelines, arm 1's mirror scratch */
    ok = ok && vkc_submit(1);
    double best = 1e30;
    for (int r = 0; ok && r < reps; r++) {
        ok = vkc_begin();
        for (int k = 0; ok && k < N; k++) ok = vkc_dsv4_attn(qb, wb, cb, lb, sb, ob, &P);
        double t0 = bnow();
        ok = ok && vkc_submit(1);
        double dt = bnow() - t0;
        if (dt < best) best = dt;
    }
    vkc_stats(&s1);
    *fa = s1.attn_fa - s0.attn_fa;
    if (lever) unsetenv("COLI_VK_CHAIN_ATTN_FA");
    if (!ok) { printf("FAIL: a dispatch was refused (the shape, or the shader missing)\n"); exit(1); }
    return best / N;
}

int main(int argc, char **argv) {
    const char *spv = argc > 1 ? argv[1] : "shaders/qmatmul.spv";
    int S = 1, H = 64, N = 200, reps = 3, nwin = 128, ncmp = 640, hd = 512, cnt = nwin + ncmp;
    const char *se = getenv("COLI_VK_BENCH_ROWS"), *he = getenv("COLI_VK_BENCH_VH"), *ne = getenv("COLI_VK_BENCH_N");
    if (se && atoi(se) > 0) S = atoi(se);
    if (he && atoi(he) > 0) H = atoi(he);
    if (ne && atoi(ne) > 0) N = atoi(ne);
    else if (S > 1) N = 3;   /* the chunk arms: mode 0 ~0.3 s a dispatch, so 3 keeps a frame under ~1 s of GPU work, clear of this box's amdgpu hard-recovery boundary (~2-4 s) */
    if (!coli_vk_init(spv)) { printf("FAIL: no Vulkan device (shaders %s)\n", spv); return 1; }
    if (!vkc_init()) { printf("FAIL: the chain's pipelines did not come up\n"); return 1; }

    int qrow = H * hd + 4, lrow = cnt + 2, orow = H * hd + 4;
    float *q = fvec((size_t)S * qrow), *win = fvec((size_t)nwin * hd + 8), *cmp = fvec((size_t)ncmp * hd + 8), *sink = fvec((size_t)H + 2);
    int *list = malloc((size_t)S * lrow * 4);
    for (int j = 0; j < cnt; j++) list[2 + j] = j < nwin ? j : nwin + (j - nwin) % ncmp;   /* every entry live */
    for (int s = 1; s < S; s++) memcpy(list + (size_t)s * lrow, list, (size_t)lrow * 4);
    P = (VkcDsAttn){S, H, hd, cnt, 2, lrow, nwin, 8, 8, 4, qrow, 4, orow, 1, 1, 1.f / sqrtf((float)hd)};

    {   /* the uploads (the test's up(): one frame a buffer) */
        qb = vkc_buf((size_t)S * qrow * 4, VKC_DEV); wb = vkc_buf(((size_t)nwin * hd + 8) * 4, VKC_DEV);
        cb = vkc_buf(((size_t)ncmp * hd + 8) * 4, VKC_DEV); sb = vkc_buf(((size_t)H + 2) * 4, VKC_DEV);
        lb = vkc_buf((size_t)S * lrow * 4, VKC_DEV); ob = vkc_buf((size_t)S * orow * 4, VKC_DEV);
        if (!qb || !wb || !cb || !sb || !lb || !ob) { printf("FAIL: out of device memory\n"); return 1; }
        vkc_begin();
        int ok = vkc_write(qb, 0, q, (size_t)S * qrow * 4) & vkc_write(wb, 0, win, ((size_t)nwin * hd + 8) * 4);
        ok &= vkc_write(cb, 0, cmp, ((size_t)ncmp * hd + 8) * 4) & vkc_write(sb, 0, sink, ((size_t)H + 2) * 4);
        ok &= vkc_write(lb, 0, list, (size_t)S * lrow * 4);
        if (!vkc_submit(1) || !ok) { printf("FAIL: the uploads\n"); return 1; }
    }

    printf("bench fa S %d H %d hd %d window %d pool %d cnt %d, %d dispatches a frame, best of %d:\n",
           S, H, hd, nwin, ncmp, cnt, N, reps);
    unsigned long long fa0, fa1, fa2;
    double m0 = arm(NULL, N, reps, &fa0);
    double m1 = arm("1", N, reps, &fa1);
    double m2 = arm("2", N, reps, &fa2);
    if (fa0) printf("  mode 0 moved attn_fa by %llu -- the lever is leaking\n", fa0);
    unsigned long long want = (unsigned long long)N * reps + 1;   /* the warm-up dispatch too */
    if (fa1 != want || fa2 != want)
        printf("  WARN: attn_fa moved %llu (arm 1) / %llu (arm 2), expected %llu: the tile did not engage\n", fa1, fa2, want);
    printf("  mode 0 (scalar two-pass):  %7.1f us a dispatch\n", m0 * 1e6);
    printf("  arm 1  (tile + f16 mirror): %7.1f us a dispatch, %.1fx\n", m1 * 1e6, m0 / m1);
    printf("  arm 2  (tile over fp32):    %7.1f us a dispatch, %.1fx\n", m2 * 1e6, m0 / m2);
    if (S == 1) printf("  a token: mode 0 %.1f ms over 43 layers, arm 1 %.1f ms\n", m0 * 43e3, m1 * 43e3);

    vkc_free(qb); vkc_free(wb); vkc_free(cb); vkc_free(sb); vkc_free(lb); vkc_free(ob);
    free(q); free(win); free(cmp); free(sink); free(list);
    vkc_shutdown(); coli_vk_shutdown();
    return 0;
}
