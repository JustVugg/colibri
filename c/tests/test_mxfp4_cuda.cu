/* MXFP4 (fmt=7) on CUDA, checked against the CPU decoder that already ships.
 *
 * Kimi K3's routed experts are QAT in MXFP4 and passed through un-re-encoded,
 * so fmt=7 is the format its expert tier runs on. Until now only the Vulkan
 * shader could decode it; the CUDA backend understood fmt 0/1/2/3/4/6 and
 * nothing else, so a CUDA host had no expert tier for that engine at all.
 *
 * The reference is quant.h's matmul_mxfp4 -- the CPU path the engine already
 * uses and trusts. That makes this a differential test rather than a
 * self-consistent one: if both sides are wrong in the same way the test says
 * nothing, but a decode that disagrees with what the engine already computes
 * is caught immediately, which is the failure that matters when adding a
 * second implementation of an existing format.
 *
 * Tolerance is 1e-4 relative. It is NOT bit-exactness: the two accumulate in a
 * different order (CPU strides columns serially, CUDA strides them across
 * threads and reduces), so the float sums legitimately differ in the last
 * bits. Decode errors are not subtle at this scale -- a wrong nibble, a
 * swapped parity or a misread exponent moves a result by whole factors, not by
 * an ulp.
 *
 *   make -C c cuda-test CUDA_ARCH=native      (runs this among the CUDA tests)
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>

/* quant.h is C (it uses _Thread_local, which nvcc's C++ front end rejects), so
 * the reference is compiled separately as C and reached through this one
 * declaration -- see tests/mxfp4_ref.c. */
extern "C" void mxfp4_ref(float *y, const float *x, const unsigned char *q4,
                          const unsigned char *e8s, int S, int I, int O);
#include "../backend_cuda.h"

static int fails;

#define CK(call)                                                                       \
    do {                                                                               \
        cudaError_t e_ = (call);                                                       \
        if (e_ != cudaSuccess) {                                                       \
            printf("  FAIL %s: %s\n", #call, cudaGetErrorString(e_));                  \
            return 1;                                                                  \
        }                                                                              \
    } while (0)

static uint64_t rng_state = 0x9E3779B97F4A7C15ull;
static uint32_t rnd(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (uint32_t)(rng_state >> 32);
}

/* One case: random MXFP4 weights, random activations, CPU vs CUDA. */
static void one_case(const char *what, int S, int I, int O, int fixed_exp) {
    int rb = (I + 1) / 2, ng = (I + 31) / 32;
    uint8_t *q4 = (uint8_t *)malloc((size_t)O * rb);
    uint8_t *e8 = (uint8_t *)malloc((size_t)O * ng);
    float *x = (float *)malloc(sizeof(float) * S * I);
    float *y_cpu = (float *)calloc((size_t)S * O, sizeof(float));
    float *y_gpu = (float *)calloc((size_t)S * O, sizeof(float));
    if (!q4 || !e8 || !x || !y_cpu || !y_gpu) { printf("  FAIL oom\n"); fails++; return; }

    for (int i = 0; i < O * rb; i++) q4[i] = (uint8_t)(rnd() & 0xFF);
    for (int i = 0; i < O * ng; i++)
        /* fixed_exp < 0 keeps exponents in the normal range; a fixed value
         * pins the edge cases the CPU comment documents (0 -> +0, 255 -> inf). */
        e8[i] = fixed_exp >= 0 ? (uint8_t)fixed_exp : (uint8_t)(110 + (rnd() % 30));
    for (int i = 0; i < S * I; i++) x[i] = ((float)(rnd() % 2001) - 1000.0f) / 1000.0f;

    mxfp4_ref(y_cpu, x, q4, e8, S, I, O);

    if (!coli_cuda_matmul_mxfp4(y_gpu, x, q4, e8, S, I, O)) {
        printf("  FAIL %-28s CUDA path refused the call\n", what);
        fails++;
        goto done;
    }

    {
        double worst = 0.0;
        int bad = 0;
        for (int i = 0; i < S * O; i++) {
            float a = y_cpu[i], b = y_gpu[i];
            if (isinf(a) || isinf(b)) {           /* exponent 255: both must agree it is inf */
                if (isinf(a) != isinf(b) || (isinf(a) && ((a > 0) != (b > 0)))) bad++;
                continue;
            }
            double den = fabs(a) > 1e-6 ? fabs(a) : 1e-6;
            double rel = fabs((double)a - (double)b) / den;
            if (rel > worst) worst = rel;
            if (rel > 1e-4) bad++;
        }
        if (bad) {
            printf("  FAIL %-28s %d/%d elements differ, worst rel %.3e\n", what, bad, S * O, worst);
            fails++;
        } else {
            printf("  ok   %-28s S=%-3d I=%-5d O=%-4d worst rel %.2e\n", what, S, I, O, worst);
        }
    }
done:
    free(q4); free(e8); free(x); free(y_cpu); free(y_gpu);
}

/* Every e2m1 code, decoded in isolation: one column set to code c, x = 1.0,
 * so the result IS the decoded value times the group scale. */
static void all_codes(void) {
    static const float want[16] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f,
                                   -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
    int I = 32, O = 16, rb = I / 2, ng = 1;
    uint8_t q4[16 * 16], e8[16];
    float x[32], y_cpu[16], y_gpu[16];
    memset(q4, 0, sizeof q4);
    for (int o = 0; o < O; o++) { q4[o * rb] = (uint8_t)o; e8[o] = 127; }   /* 2^0 = 1 */
    for (int i = 0; i < I; i++) x[i] = (i == 0) ? 1.0f : 0.0f;

    mxfp4_ref(y_cpu, x, q4, e8, 1, I, O);
    if (!coli_cuda_matmul_mxfp4(y_gpu, x, q4, e8, 1, I, O)) {
        printf("  FAIL all-16-e2m1-codes            CUDA refused\n");
        fails++;
        return;
    }
    int bad = 0;
    for (int o = 0; o < O; o++)
        if (y_gpu[o] != want[o] || y_cpu[o] != want[o]) {
            printf("  FAIL code %2d: want %.2f cpu %.2f gpu %.2f\n", o, want[o], y_cpu[o], y_gpu[o]);
            bad++;
        }
    if (bad) fails++;
    else printf("  ok   all 16 e2m1 codes decode exactly (cpu == gpu == spec)\n");
}

int main(void) {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) {
        printf("test_mxfp4_cuda: no CUDA device, skipping\n");
        return 0;
    }
    int dev0 = 0;
    if (!coli_cuda_init(&dev0, 1)) {
        printf("test_mxfp4_cuda: coli_cuda_init failed, skipping\n");
        return 0;
    }

    all_codes();
    one_case("decode + matmul",            1,   64,   32, -1);
    one_case("multi-row batch",            4,  128,   64, -1);
    one_case("wide rows (many groups)",    1, 2048,   16, -1);
    one_case("non-multiple-of-32 columns", 1,   96,    8, -1);   /* 3 groups exactly */
    one_case("tail group (I%32 != 0)",     1,   80,    8, -1);   /* 2.5 groups */
    one_case("exponent 0 -> +0",           1,   64,   16,  0);
    one_case("exponent 255 -> inf",        1,   64,   16, 255);
    one_case("exponent 127 -> unit scale", 1,   64,   16, 127);

    printf(fails ? "test_mxfp4_cuda: %d failure(s)\n" : "test_mxfp4_cuda: ok\n", fails);
    return fails != 0;
}
