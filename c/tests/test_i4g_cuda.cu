/* i4_tiled_f32 (fmt=4 grouped int4) vs the CPU decoder in quant.h.
 *
 * Both GPU paths read the SAME uploaded weights and scales, so running the GEMV
 * and the tiled kernel against the CPU on identical inputs isolates the tiling
 * from everything else. coli_cuda_set_tile_min() selects between them, so one
 * binary exercises both -- and using the API rather than the environment is the
 * point: the tiled dispatch is opt-in precisely so it cannot reach the GLM
 * engine, which shares coli_cuda_matmul.
 *
 * Shapes are K3's real dense-trunk projections at K3_BITS=4 (gs=64).
 *
 *   nvcc -O3 -std=c++17 -arch=native backend_cuda.cu tests/test_i4g_cuda.cu \
 *        tests/i4g_ref.o -o test_i4g_cuda -Xcompiler -fopenmp
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include "../backend_cuda.h"

extern "C" void i4g_ref(float *y, const float *x, const unsigned char *q4,
                        const float *scale, int S, int I, int O, int gs);

/* Deterministic, so a failure reproduces without carrying a seed file. */
static unsigned long long g_seed = 0x9E3779B97F4A7C15ULL;
static float rnd(void) {
    g_seed = g_seed * 6364136223846793005ULL + 1442695040888963407ULL;
    return (float)((g_seed >> 40) % 2000001) / 1000000.0f - 1.0f;
}

/* Relative L2 over the whole output: one bad tile shows up here, whereas a
 * max-relative on near-zero entries mostly reports cancellation noise. */
static double rel_l2(const float *got, const float *want, size_t n) {
    double num = 0, den = 0;
    for (size_t i = 0; i < n; i++) {
        double d = (double)got[i] - (double)want[i];
        num += d * d;
        den += (double)want[i] * (double)want[i];
    }
    return den > 0 ? std::sqrt(num / den) : (num > 0 ? 1.0 : 0.0);
}

static int fail = 0;

static void case_(const char *label, int S, int I, int O, int gs, double bound) {
    int rb = (I + 1) / 2, ng = (I + gs - 1) / gs;
    unsigned char *q4 = (unsigned char *)std::malloc((size_t)O * rb);
    float *scale = (float *)std::malloc((size_t)O * ng * sizeof(float));
    float *x = (float *)std::malloc((size_t)S * I * sizeof(float));
    float *want = (float *)std::calloc((size_t)S * O, sizeof(float));
    float *gemv = (float *)std::calloc((size_t)S * O, sizeof(float));
    float *tile = (float *)std::calloc((size_t)S * O, sizeof(float));
    if (!q4 || !scale || !x || !want || !gemv || !tile) { std::printf("  OOM\n"); fail = 1; return; }

    for (size_t i = 0; i < (size_t)O * rb; i++) {
        g_seed = g_seed * 6364136223846793005ULL + 1442695040888963407ULL;
        q4[i] = (unsigned char)(g_seed >> 40);
    }
    /* Scales spread over ~2 decades: a per-row-only index would still pass on a
     * flat scale field, so make the groups actually differ. */
    for (size_t i = 0; i < (size_t)O * ng; i++) scale[i] = 0.002f * (1.0f + 9.0f * std::fabs(rnd()));
    for (size_t i = 0; i < (size_t)S * I; i++) x[i] = rnd();

    i4g_ref(want, x, q4, scale, S, I, O, gs);

    ColiCudaTensor *t = nullptr;
    coli_cuda_set_tile_min(0);                                /* force GEMV */
    int ok_g = coli_cuda_matmul(&t, gemv, x, q4, scale, 4, S, I, O, 0, gs);
    coli_cuda_set_tile_min(1);                                /* force tiles */
    int ok_t = coli_cuda_matmul(&t, tile, x, q4, scale, 4, S, I, O, 0, gs);

    if (!ok_g || !ok_t) {
        std::printf("  FAIL %-28s S=%-3d rejected (gemv=%d tiled=%d)\n", label, S, ok_g, ok_t);
        fail = 1;
    } else {
        double eg = rel_l2(gemv, want, (size_t)S * O);
        double et = rel_l2(tile, want, (size_t)S * O);
        int bad = !(et <= bound) || !(eg <= 1e-5);
        std::printf("  %-4s %-28s S=%-3d I=%-5d O=%-5d  gemv %.2e  tiled %.2e  (bound %.0e)\n",
                    bad ? "FAIL" : "ok", label, S, I, O, eg, et, bound);
        if (bad) fail = 1;
    }
    if (t) coli_cuda_tensor_free(t);
    std::free(q4); std::free(scale); std::free(x);
    std::free(want); std::free(gemv); std::free(tile);
}

int main(void) {
    int dev[1] = {0};
    if (!coli_cuda_init(dev, 1)) { std::fprintf(stderr, "no CUDA device - skipped\n"); return 0; }

    /* fp32 throughout, so the tiled path should sit near the GEMV's own 1.8e-07
     * and differ from the CPU only by summation order. 1e-5 leaves room for that
     * reordering over K=7168 terms while still failing loudly on a wrong scale
     * index, which moves entries by whole decades here.
     *
     * The predecessor of this kernel used fp16 Tensor Cores and passed at 2.6e-04
     * against a 1e-3 bound -- and produced 5.9% logit drift with argmax flips on
     * the real model. Hence the tight bound: a kernel-level number this loose is
     * not evidence of anything at 93 layers. */
    const double B = 1e-5;
    std::printf("fmt=4 grouped int4, fp32 shared-memory tiling vs CPU (gs=64)\n");
    case_("lat_down 7168->3584",   32, 7168, 3584, 64, B);
    case_("lat_up   3584->7168",   32, 3584, 7168, 64, B);
    case_("sh_gate  7168->6144",   32, 7168, 6144, 64, B);
    case_("kda_qkv  7168->12288",  32, 7168, 12288, 64, B);
    /* S below/at/above the tile edge: 16 is exactly one tile, 33 forces a
     * partial tile, 1 is the decode shape the tiled path must still get right
     * when the threshold is forced down to it. */
    case_("tile edge S=16",        16, 3584, 3584, 64, B);
    case_("partial tile S=33",     33, 3584, 3584, 64, B);
    case_("degenerate S=1",         1, 3584, 3584, 64, B);
    /* gs=128 straddles nothing; gs=16 makes every k-tile its own group, which is
     * the indexing most likely to be off by one. */
    case_("gs=128",                32, 3584, 3584, 128, B);
    case_("gs=16",                 32, 3584, 3584, 16, B);

    std::printf("%s\n", fail ? "test_i4g_cuda: FAIL" : "test_i4g_cuda: ok");
    return fail;
}
