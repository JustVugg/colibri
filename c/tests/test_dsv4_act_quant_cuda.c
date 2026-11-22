/* The 128-block fp8 calls of the CUDA backend against the CPU reference itself, bit for bit:
 * dsv4_cuda_matvec, _matvec_grouped (wo_a, fmt 9), the dsv4_cuda_wo_decode chain,
 * _proj_batch and _matmul_batch against coli_fp8_matvec_ref (which rounds its input to E4M3
 * per 128 first) and coli_bf16_round_array. The operands make every sum exact in fp32 in
 * any order: weights are integers -8..8 times 2^-1 or 2^0, and each 128-block of the raw
 * input is one +-448 * 2^k plus integers |j| <= 16 times 2^k, each moved by 2% (a 0 by
 * 1e-4 * 2^k) so that the rounding brings it back. Equal bits then mean equal operands.
 * With DSV4_CUDA_ACT_QUANT=0 (the raw input, the behaviour before) every call must differ
 * instead, which shows the inputs are sensitive to the rounding.
 * Windows: gcc $(CFLAGS) tests/test_dsv4_act_quant_cuda.c backend_loader_dsv4.c
 * tests/deepseek_v4_native_quant.o $(LDFLAGS), with COLI_DSV4_DLL naming the DLL (the
 * DeepGEMM build refuses the batch shape here: its own kernels round);
 * Linux: make dsv4-cuda-test. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../backend_cuda_dsv4.h"
#include "../native_quant.h"

static int failures, raw_mode;
static uint32_t rng = 20261111u;
static uint32_t next(void) { rng = rng * 1664525u + 1013904223u; return rng >> 8; }

typedef struct { uint8_t *w, *s; float *fs; ColiTensorView v; Dsv4CudaTensor *t; } Mat;

/* sparse: one weight per row (column 37 o + 11 mod I), the rest 0 */
static int mat(Mat *m, int O, int I, int fmt9, int sparse, int dev) {
    size_t blocks = (size_t)(O / 128) * (I / 128);
    m->t = NULL;
    m->w = calloc((size_t)O * I, 1); m->s = malloc(blocks); m->fs = malloc(blocks * sizeof(float));
    if (!m->w || !m->s || !m->fs) return 0;
    for (size_t i = 0; i < (size_t)O * I; i++)
        if (!sparse || (int)(i % I) == (int)((37 * (i / I) + 11) % I))
            m->w[i] = coli_e4m3fn_encode((float)((int)(next() % 17) - 8));
    for (size_t i = 0; i < blocks; i++) { m->s[i] = (uint8_t)(126 + next() % 2); m->fs[i] = ldexpf(1.f, m->s[i] - 127); }
    m->v = (ColiTensorView){COLI_TENSOR_FP8_E4M3_BLOCK, COLI_SCALE_F32, m->w, m->fs, (size_t)O * I,
                            blocks * sizeof(float), O, I, 128, 128, NULL};
    return fmt9 ? dsv4_cuda_upload_fp8_bf16(&m->t, m->w, m->s, O, I, dev)
                : dsv4_cuda_upload_fp8(&m->t, m->w, m->s, O, I, dev);
}
static void mat_free(Mat *m) { dsv4_cuda_tensor_free(m->t); free(m->w); free(m->s); free(m->fs); }

static void raw_input(float *x, size_t n) {
    for (size_t b = 0; b < n; b += 128) {
        int k = -4 + (int)(next() % 2);
        for (size_t i = b; i < b + 128; i++) {
            int j = (int)(next() % 33) - 16;
            x[i] = ldexpf(j ? j * (next() & 1 ? 1.02f : 0.98f) : (next() & 1 ? 1e-4f : -1e-4f), k);
        }
        x[b + next() % 128] = ldexpf(next() & 1 ? 448.f : -448.f, k);
    }
}

/* the CPU reference of a grouped (wo_a) matvec: one row-block view per group */
static int cpu_grouped(float *y, const Mat *m, const float *x, int groups, int O, int I) {
    for (int g = 0; g < groups; g++) {
        ColiTensorView v = m->v;
        int rows = O / groups;
        v.rows = rows;
        v.data = m->w + (size_t)g * rows * I;
        v.scales = m->fs + (size_t)g * (rows / 128) * (I / 128);
        v.data_bytes = (size_t)rows * I;
        v.scale_bytes = (size_t)(rows / 128) * (I / 128) * sizeof(float);
        if (coli_fp8_matvec_ref(y + (size_t)g * rows, &v, x + (size_t)g * I)) return 0;
    }
    return 1;
}

static void verdict(const char *what, const float *got, const float *want, size_t n) {
    size_t bad = 0;
    for (size_t i = 0; i < n; i++) bad += memcmp(got + i, want + i, sizeof *got) != 0;
    int pass = raw_mode ? bad > 0 : bad == 0;
    printf("  %-34s %6zu of %6zu differ from the CPU: %s\n", what, bad, n, pass ? "ok" : "FAIL");
    failures += !pass;
}
static void refused(const char *what) { printf("  %-34s refused: FAIL\n", what); failures++; }

int main(void) {
    int dev = 0;
    const char *env = getenv("DSV4_CUDA_ACT_QUANT");
    raw_mode = env && *env && !atoi(env);
    if (!dsv4_cuda_init(&dev, 1)) {   /* a named DLL that does not load is a failure */
        const char *dll = getenv("COLI_DSV4_DLL");
        printf("%s: no CUDA backend%s%s\n", dll ? "FAIL" : "SKIP", dll ? " from " : "", dll ? dll : "");
        return dll != NULL;
    }
    printf("backend %s, %s\n", dsv4_cuda_backend_name(),
           raw_mode ? "DSV4_CUDA_ACT_QUANT=0: every call must differ" : "every call bit-exact");
    enum { H = 4096, R = 1024, K = 512, G = 4, OR = 256, GW = 512, T = 5 };
    float *x = malloc((size_t)T * H * 4), *y = malloc((size_t)T * H * 4), *ref = malloc((size_t)T * H * 4);
    float *y2 = malloc((size_t)T * H * 4), *ref2 = malloc((size_t)T * H * 4), *stage = malloc((size_t)(R + K) * 4);
    Mat a, b, wa, wb, mm;
    if (!x || !y || !ref || !y2 || !ref2 || !stage || !mat(&a, R, H, 0, 0, dev) || !mat(&b, K, H, 0, 0, dev) ||
        !mat(&wa, G * OR, GW, 1, 0, dev) || !mat(&wb, K, G * OR, 0, 1, dev) || !mat(&mm, K, R, 0, 0, dev)) {
        puts("FAIL setup"); return 1;
    }

    raw_input(x, H);   /* wq_a shape */
    if (!dsv4_cuda_matvec(a.t, y, x) || coli_fp8_matvec_ref(ref, &a.v, x)) refused("matvec 1024x4096");
    else verdict("matvec 1024x4096", y, ref, R);

    raw_input(x, (size_t)G * GW);   /* wo_a: 4 groups of 256x512, fmt 9 */
    if (!dsv4_cuda_matvec_grouped(wa.t, y, x, G) || !cpu_grouped(ref, &wa, x, G, G * OR, GW))
        refused("matvec_grouped 4x(256x512) fmt 9");
    else verdict("matvec_grouped 4x(256x512) fmt 9", y, ref, (size_t)G * OR);

    /* wo_decode: wo_a, bf16, wo_b (one weight per row), bf16 -- the host chain */
    if (!dsv4_cuda_wo_decode(wa.t, wb.t, G, y, x) || !cpu_grouped(ref2, &wa, x, G, G * OR, GW))
        refused("wo_decode");
    else {
        coli_bf16_round_array(ref2, (size_t)G * OR);
        if (coli_fp8_matvec_ref(ref, &wb.v, ref2)) refused("wo_decode (CPU wo_b)");
        else { coli_bf16_round_array(ref, K); verdict("wo_decode 4x(256x512) -> 512", y, ref, K); }
    }

    /* proj_batch: wq_a and wkv shapes over one input */
    {
        Dsv4CudaTensor *ts[2] = {a.t, b.t};
        int rows[2] = {R, K};
        raw_input(x, H);
        if (!dsv4_cuda_proj_batch(2, ts, rows, x, stage) || coli_fp8_matvec_ref(ref, &a.v, x) ||
            coli_fp8_matvec_ref(ref2, &b.v, x)) refused("proj_batch");
        else {
            verdict("proj_batch 1024x4096", stage, ref, R);
            verdict("proj_batch 512x4096", stage + R, ref2, K);
        }
    }

    /* matmul_batch: T rows, each rounded per 128 as coli_fp8_matvec_ref rounds it */
    {
        Dsv4CudaActivation *in = dsv4_cuda_activation_create(dev, (long long)T * R);
        Dsv4CudaActivation *out = dsv4_cuda_activation_create(dev, (long long)T * K);
        raw_input(x, (size_t)T * R);
        int ran = in && out && dsv4_cuda_activation_upload(in, x, (long long)T * R) &&
                  dsv4_cuda_matmul_batch(mm.t, in, T, out);
        if (!ran && strstr(dsv4_cuda_backend_name(), "deepgemm"))
            printf("  %-34s refused by the DeepGEMM build (its kernels round): skipped\n", "matmul_batch 512x1024 T=5");
        else if (!ran || !dsv4_cuda_activation_download(y, out, (long long)T * K) ||
                 !dsv4_cuda_activation_download(y2, in, (long long)T * R) || !dsv4_cuda_activation_sync(out))
            refused("matmul_batch 512x1024 T=5");
        else {
            for (int t = 0; t < T; t++)
                if (coli_fp8_matvec_ref(ref + (size_t)t * K, &mm.v, x + (size_t)t * R)) refused("matmul_batch (CPU)");
            verdict("matmul_batch 512x1024 T=5", y, ref, (size_t)T * K);
            int kept = !memcmp(y2, x, (size_t)T * R * 4);
            printf("  %-34s %s\n", "matmul_batch leaves its input raw", kept ? "ok" : "FAIL");
            failures += !kept;
        }
        dsv4_cuda_activation_free(in); dsv4_cuda_activation_free(out);
    }

    mat_free(&a); mat_free(&b); mat_free(&wa); mat_free(&wb); mat_free(&mm);
    free(x); free(y); free(ref); free(y2); free(ref2); free(stage);
    dsv4_cuda_shutdown();
    printf("%s: dsv4 act_quant of the 128-block fp8 calls\n", failures ? "FAIL" : "PASS");
    return failures != 0;
}
