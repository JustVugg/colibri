/* fmt=1 (int8, per-row scale) on CUDA vs the CPU decoder in quant.h.
 *
 * This path was shipped untested by test_i4g_cuda, which only covered fmt=4.
 * K3's MLA projections are fmt=1 at the default K3_MLA_BITS=8, so K3_CUDA_DENSE
 * routes 24 layers x 5 tensors through it -- and the dense prefill logits came
 * out 5% off the CPU with argmax flips even after the int4 kernel was proven
 * correct to 1e-06. If fmt=1 disagrees, that is the explanation.
 *
 *   nvcc -O3 -std=c++17 -arch=native backend_cuda.cu tests/test_i8_cuda.cu \
 *        tests/i4g_ref.o -o test_i8_cuda -lgomp
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "../backend_cuda.h"

extern "C" void i8_ref(float *y, const float *x, const signed char *q8,
                       const float *scale, int S, int I, int O);

static unsigned long long sd = 0x9E3779B97F4A7C15ULL;
static float rnd(void) {
    sd = sd * 6364136223846793005ULL + 1442695040888963407ULL;
    return (float)((sd >> 40) % 2000001) / 1000000.0f - 1.0f;
}
static double rel_l2(const float *a, const float *b, size_t n) {
    double num = 0, den = 0;
    for (size_t i = 0; i < n; i++) {
        double d = (double)a[i] - (double)b[i];
        num += d * d; den += (double)b[i] * (double)b[i];
    }
    return den > 0 ? std::sqrt(num / den) : (num > 0 ? 1.0 : 0.0);
}

static int fail = 0;
static void case_(const char *nm, int S, int I, int O) {
    signed char *q8 = (signed char *)std::malloc((size_t)O * I);
    float *sc = (float *)std::malloc((size_t)O * sizeof(float));
    float *x = (float *)std::malloc((size_t)S * I * sizeof(float));
    float *want = (float *)std::calloc((size_t)S * O, sizeof(float));
    float *got = (float *)std::calloc((size_t)S * O, sizeof(float));
    for (size_t i = 0; i < (size_t)O * I; i++) { sd = sd*6364136223846793005ULL+1442695040888963407ULL; q8[i] = (signed char)(sd >> 40); }
    for (int o = 0; o < O; o++) sc[o] = 0.002f * (1.0f + std::fabs(rnd()));
    for (size_t i = 0; i < (size_t)S * I; i++) x[i] = rnd();

    i8_ref(want, x, q8, sc, S, I, O);
    ColiCudaTensor *t = nullptr;
    int ok = coli_cuda_matmul(&t, got, x, q8, sc, 1, S, I, O, 0, 0);
    double e = ok ? rel_l2(got, want, (size_t)S * O) : 1.0;
    int bad = !ok || !(e <= 1e-5);
    std::printf("  %-4s %-24s S=%-3d I=%-5d O=%-5d  rel_l2 %.3e%s\n",
                bad ? "FAIL" : "ok", nm, S, I, O, e, ok ? "" : "  (REJECTED)");
    if (bad) {
        fail = 1;
        for (int i = 0; i < 4 && i < S * O; i++)
            std::printf("        [%d] gpu %.6f  cpu %.6f  ratio %.6f\n",
                        i, got[i], want[i], want[i] != 0 ? got[i] / want[i] : 0.0);
    }
    if (t) coli_cuda_tensor_free(t);
    std::free(q8); std::free(sc); std::free(x); std::free(want); std::free(got);
}

int main(void) {
    int d[1] = {0};
    if (!coli_cuda_init(d, 1)) { std::fprintf(stderr, "no CUDA device - skipped\n"); return 0; }
    std::printf("fmt=1 int8 per-row vs CPU\n");
    case_("mla_qa  7168->1536",  32, 7168, 1536);
    case_("mla_qb  1536->12288", 32, 1536, 12288);
    case_("mla_kva 7168->576",   32, 7168, 576);
    case_("S=1 decode shape",     1, 7168, 1536);
    case_("S=8 below threshold",  8, 7168, 1536);
    std::printf("%s\n", fail ? "test_i8_cuda: FAIL" : "test_i8_cuda: ok");
    return fail;
}
