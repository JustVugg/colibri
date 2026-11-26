/* fmt=9 (bf16) — decode, parity and scale-free accounting oracle.
 *
 * The qwen38 dense side (attention q/k/v/o, DeltaNet in/out_proj, the gated
 * residual, the shared expert, the router and lm_head) is bf16 end to end and
 * carries NO scales, while this backend had no bf16 format at all: every dense
 * matmul therefore stayed on the CPU, which is 86.9% of the per-token decode
 * byte traffic. fmt=9 is the enabling brick, and it is additive -- it rides the
 * generic weight_at branch of quant_matmul, so the only way it can be wrong is
 * in the decode itself or in the scale bookkeeping that assumed "fmt != 0 has a
 * scale buffer".
 *
 * Four claims, each independently falsifiable:
 *   1. DECODE -- bf16_at() reproduces st.h's bf16_to_f32 BIT FOR BIT on
 *      all 65536 bf16 patterns, NaN/inf/subnormals included. Compared as raw
 *      u32 so NaN != NaN cannot hide a mismatch. A shift-by-15, a byte swap or
 *      a float16 decoder all die here.
 *   2. PARITY -- coli_cuda_matmul(fmt=9) against a double CPU reference built
 *      on bf16_to_f32 itself, over a shape with an odd I (tail) and S>1.
 *   3. SCALE-FREE -- the tensor must upload with scales == NULL, charge ZERO
 *      scale bytes (coli_cuda_tensor_bytes == I*O*2), and accept a refresh with
 *      a NULL scale pointer. This is what the fmt_scale_free() consolidation
 *      claims; before it, fmt=9 would have cudaMalloc'd O floats, copied from
 *      NULL, and the trailing `partial[0] * scales[o]` would have dereferenced
 *      a null pointer on every output.
 *   4. NEGATIVE CONTROL -- the widened guard must NOT admit a scale-bearing
 *      format with no scales: fmt=1 + NULL must still be refused. Without this,
 *      claim 3 would be satisfied just as well by deleting the guard.
 *
 * Build: nvcc -O2 -std=c++17 -arch=native tests/test_bf16_cuda.cu -o bf16_cuda_test
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>

#include "../backend_cuda.cu"

/* Byte-for-byte st.h's bf16_to_f32 -- the CPU decoder every qwen38 dense path
 * already uses. Restated here because st.h is C and this TU is C++. */
static inline float ref_bf16_to_f32(uint16_t h) {
    uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f;
}
static inline uint32_t bits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

/* Claim 1 runs weight_at itself, not a copy of it: one row of 65536 bf16
 * weights, one output per pattern. */
__global__ static void decode_all(const void *w, float *out, int n) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) out[i] = bf16_at(static_cast<const uint8_t *>(w), (size_t)i);
}

int main(void) {
    int bad = 0;
    int devs[1] = {0};
    if (!coli_cuda_init(devs, 1)) { printf("FAIL cuda init\n"); return 1; }

    /* ---- claim 1: exhaustive bit-exact decode ---- */
    {
        const int N = 65536;
        uint16_t *hw = (uint16_t *)malloc((size_t)N * 2);
        for (int i = 0; i < N; i++) hw[i] = (uint16_t)i;
        void *dw = NULL; float *dy = NULL;
        float *hy = (float *)malloc((size_t)N * 4);
        if (cudaMalloc(&dw, (size_t)N * 2) != cudaSuccess ||
            cudaMalloc(&dy, (size_t)N * 4) != cudaSuccess ||
            cudaMemcpy(dw, hw, (size_t)N * 2, cudaMemcpyHostToDevice) != cudaSuccess) {
            printf("FAIL decode setup\n"); return 1; }
        decode_all<<<(N + 255) / 256, 256>>>(dw, dy, N);
        if (cudaGetLastError() != cudaSuccess ||
            cudaMemcpy(hy, dy, (size_t)N * 4, cudaMemcpyDeviceToHost) != cudaSuccess) {
            printf("FAIL decode launch\n"); return 1; }
        int mism = 0;
        for (int i = 0; i < N; i++)
            if (bits(hy[i]) != bits(ref_bf16_to_f32((uint16_t)i))) {
                if (mism < 5) printf("  decode mismatch pattern 0x%04x: got 0x%08x want 0x%08x\n",
                                     i, bits(hy[i]), bits(ref_bf16_to_f32((uint16_t)i)));
                mism++;
            }
        if (mism) { printf("FAIL decode: %d/%d patterns\n", mism, N); bad += mism; }
        else printf("  decode: 65536/65536 bf16 patterns bit-exact\n");
        cudaFree(dw); cudaFree(dy); free(hw); free(hy);
    }

    /* ---- claims 2+3: GEMM parity and scale-free residency ---- */
    srand(9);
    const int S = 3, I = 257, O = 64;          /* odd I: the row stride is I*2, not padded */
    uint16_t *w = (uint16_t *)malloc((size_t)I * O * 2);
    float *x = (float *)malloc((size_t)S * I * 4);
    float *y = (float *)malloc((size_t)S * O * 4);
    /* Draw the weights as bf16 patterns of real magnitude rather than random
     * 16-bit noise, which would be mostly inf/NaN and make the parity check
     * vacuous. */
    for (size_t i = 0; i < (size_t)I * O; i++) {
        float v = (rand() / (float)RAND_MAX - .5f) * 2.f;
        w[i] = (uint16_t)(bits(v) >> 16);
    }
    for (size_t i = 0; i < (size_t)S * I; i++) x[i] = (rand() / (float)RAND_MAX - .5f) * 2.f;

    ColiCudaTensor *t = NULL;
    if (!coli_cuda_matmul(&t, y, x, w, NULL, 9, S, I, O, 0, 0)) {
        printf("FAIL matmul fmt=9 (scales==NULL refused?)\n"); return 1; }

    int par = 0;
    for (int s = 0; s < S; s++)
        for (int o = 0; o < O; o++) {
            double a = 0;
            for (int i = 0; i < I; i++)
                a += (double)x[(size_t)s * I + i] * (double)ref_bf16_to_f32(w[(size_t)o * I + i]);
            float got = y[(size_t)s * O + o];
            /* f32 tree reduction over 257 terms vs a double serial sum: a few
             * ulp. A double-applied scale or a wrong row stride moves the
             * output by O(1). */
            if (fabs(got - a) > 1e-4 * (fabs(a) + 1e-3)) {
                if (par < 5) printf("  parity mismatch s=%d o=%d: got %g want %g\n", s, o, got, a);
                par++;
            }
        }
    if (par) { printf("FAIL parity: %d/%d outputs\n", par, S * O); bad += par; }
    else printf("  parity: %d outputs vs the double CPU reference\n", S * O);

    size_t want_bytes = (size_t)I * O * 2, got_bytes = coli_cuda_tensor_bytes(t);
    if (got_bytes != want_bytes) {
        printf("FAIL accounting: tensor_bytes %zu, expected %zu (scale bytes charged?)\n",
               got_bytes, want_bytes); bad++;
    } else printf("  accounting: %zu bytes, no scale buffer charged\n", got_bytes);

    if (!coli_cuda_tensor_update(t, w, NULL)) {
        printf("FAIL refresh: fmt=9 update with NULL scales refused\n"); bad++;
    } else printf("  refresh: in-place update with NULL scales accepted\n");

    /* ---- claim 4: the guard is not over-widened ---- */
    {
        ColiCudaTensor *t8 = NULL;
        int8_t *q = (int8_t *)calloc((size_t)I * O, 1);
        if (coli_cuda_tensor_upload(&t8, q, NULL, 1, I, O, 0)) {
            printf("FAIL negative control: fmt=1 accepted a NULL scale pointer\n");
            bad++; coli_cuda_tensor_free(t8);
        } else printf("  negative control: fmt=1 + NULL scales still refused\n");
        free(q);
    }

    coli_cuda_tensor_free(t);
    free(w); free(x); free(y);
    printf("bf16 (fmt=9) oracle: %d failures\n", bad);
    if (bad) { printf("FAIL\n"); return 1; }
    printf("ok\n");
    return 0;
}
