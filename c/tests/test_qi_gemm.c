/* qi_gemm against the naive triple loop, every format, ragged shapes; and a
 * throughput line for the DiT's biggest shape (argv[1] = "bench"). */
#include "../qi_gemm.h"
#include <math.h>
#include <time.h>

static float frand(uint32_t *s){ *s = *s * 1664525u + 1013904223u; return ((*s >> 8) / 16777216.f) * 2.f - 1.f; }
static uint16_t to_bf16(float f){ union { float f; uint32_t u; } v; v.f = f; return (uint16_t)((v.u + 0x7fff + ((v.u >> 16) & 1)) >> 16); }

static int check(int M, int N, int K, int fmt, int use_bias){
    uint32_t s = 1234u + M * 7 + N * 13 + K * 17 + fmt;
    float *X = malloc(sizeof(float) * M * K), *Wf = malloc(sizeof(float) * N * K);
    float *Y = malloc(sizeof(float) * M * N), *R = malloc(sizeof(float) * M * N), *b = malloc(sizeof(float) * N);
    uint16_t *Wb = malloc(2 * (size_t)N * K); int8_t *Wq = malloc((size_t)N * K); float *sc = malloc(sizeof(float) * N);
    for (int i = 0; i < M * K; i++) X[i] = frand(&s);
    for (int i = 0; i < N * K; i++) Wf[i] = frand(&s);
    for (int i = 0; i < N; i++) b[i] = frand(&s);
    QiMat W = { fmt, N, K, Wf, NULL, 0 };
    if (fmt == QI_BF16) { for (int i = 0; i < N * K; i++) { Wb[i] = to_bf16(Wf[i]); Wf[i] = qi_bf16(Wb[i]); } W.w = Wb; }
    if (fmt == QI_I8) { qi_quantize_i8(Wf, N, K, Wq, sc); for (int n = 0; n < N; n++) for (int k = 0; k < K; k++) Wf[(int64_t)n*K+k] = Wq[(int64_t)n*K+k] * sc[n]; W.w = Wq; W.sc = sc; }
    qi_gemm(Y, X, M, &W, use_bias ? b : NULL);
    double maxerr = 0;
    for (int m = 0; m < M; m++) for (int n = 0; n < N; n++) {
        double acc = use_bias ? b[n] : 0;
        for (int k = 0; k < K; k++) acc += (double)X[(int64_t)m*K+k] * Wf[(int64_t)n*K+k];
        double e = fabs(acc - Y[(int64_t)m*N+n]) / (1.0 + fabs(acc));
        if (e > maxerr) maxerr = e;
    }
    int ok = maxerr < 1e-4;
    printf("%s M=%d N=%d K=%d fmt=%d bias=%d maxrel=%.2e\n", ok ? "ok  " : "FAIL", M, N, K, fmt, use_bias, maxerr);
    free(X); free(Wf); free(Y); free(R); free(b); free(Wb); free(Wq); free(sc);
    return ok;
}


static int check_act8(int M, int N, int K){
#ifdef QI_HAVE_VNNI
    uint32_t s = 99u + M + N * 3 + K * 5;
    float *X = malloc(sizeof(float) * M * K), *Wf = malloc(sizeof(float) * N * K), *Y = malloc(sizeof(float) * M * N);
    float *b = malloc(sizeof(float) * N); int8_t *q = malloc((size_t)N * K); float *sc = malloc(sizeof(float) * N);
    for (int i = 0; i < M * K; i++) X[i] = frand(&s) * (i % 7 == 0 ? 4.f : 1.f);
    for (int i = 0; i < N * K; i++) Wf[i] = frand(&s);
    for (int i = 0; i < N; i++) b[i] = frand(&s);
    qi_quantize_i8(Wf, N, K, q, sc);
    QiMat W = { QI_I8, N, K, q, sc, 0 };
    qi_gemm_act8(Y, N, X, K, M, &W, b);
    double maxerr = 0;
    for (int m = 0; m < M; m++) {
        float am = 0; for (int k = 0; k < K; k++) { float a = fabsf(X[(int64_t)m*K+k]); if (a > am) am = a; }
        float sx = am / 127.f, inv = 1.f / sx;
        for (int n = 0; n < N; n++) {
            long acc = 0;
            for (int k = 0; k < K; k++) { float v = X[(int64_t)m*K+k] * inv; int iv = (int)(v < 0 ? v - 0.5f : v + 0.5f); acc += (long)iv * q[(int64_t)n*K+k]; }
            double want = (double)acc * sx * sc[n] + b[n];
            double e = fabs(want - Y[(int64_t)m*N+n]) / (1.0 + fabs(want));
            if (e > maxerr) maxerr = e;
        }
    }
    int ok = maxerr < 1e-5;
    printf("%s act8 M=%d N=%d K=%d maxrel=%.2e\n", ok ? "ok  " : "FAIL", M, N, K, maxerr);
    free(X); free(Wf); free(Y); free(b); free(q); free(sc);
    return ok;
#else
    printf("skip act8 (no VNNI)\n"); return 1;
#endif
}

/* QI_GG: rows the caller's decoder unpacks while a panel is packed (a GGUF's blocks in the
 * engine). Here the "blocks" are f32 rows and the decoder a copy, so the f32 product must be
 * QI_F32's bytes, and the int8 x int8 one (the decoded rows quantized with the load's scales)
 * QI_I8's bytes. */
static int gg_copy(int type, const uint8_t *src, float *dst, int64_t n){ (void)type; memcpy(dst, src, (size_t)n * 4); return 0; }
static int check_gg(int M, int N, int K){
    uint32_t s = 4242u + (uint32_t)(M * 31 + N * 7 + K);
    float *X = malloc(sizeof(float) * (size_t)M * K), *Wf = malloc(sizeof(float) * (size_t)N * K);
    float *Y0 = malloc(sizeof(float) * (size_t)M * N), *Y1 = malloc(sizeof(float) * (size_t)M * N);
    int8_t *q = malloc((size_t)N * K); float *sc = malloc(sizeof(float) * N);
    for (size_t i = 0; i < (size_t)M * K; i++) X[i] = frand(&s);
    for (size_t i = 0; i < (size_t)N * K; i++) Wf[i] = frand(&s) * 0.1f;
    qi_quantize_i8(Wf, N, K, q, sc);
    QiGG g = { (const uint8_t *)Wf, (int64_t)K * 4, 0, gg_copy };
    QiMat G = { QI_GG, N, K, &g, sc, 0 }, F = { QI_F32, N, K, Wf, NULL, 0 }, I = { QI_I8, N, K, q, sc, 0 };
    qi_gemm(Y0, X, M, &F, NULL);
    qi_gemm(Y1, X, M, &G, NULL);
    int ok = !memcmp(Y0, Y1, sizeof(float) * (size_t)M * N);
    printf("%s gg f32 M=%d N=%d K=%d\n", ok ? "ok  " : "FAIL", M, N, K);
#ifdef QI_HAVE_VNNI
    qi_gemm_act8(Y0, N, X, K, M, &I, NULL);
    qi_gemm_act8(Y1, N, X, K, M, &G, NULL);
    int ok8 = !memcmp(Y0, Y1, sizeof(float) * (size_t)M * N);
    printf("%s gg act8 M=%d N=%d K=%d\n", ok8 ? "ok  " : "FAIL", M, N, K);
    ok &= ok8;
#endif
    free(X); free(Wf); free(Y0); free(Y1); free(q); free(sc);
    return ok;
}

int main(int argc, char **argv){
    if (argc > 1 && !strcmp(argv[1], "bench")) {
        int M = argc > 2 ? atoi(argv[2]) : 1296, N = argc > 3 ? atoi(argv[3]) : 12288, K = argc > 4 ? atoi(argv[4]) : 4096;
        float *X = malloc(sizeof(float) * (size_t)M * K), *Y = malloc(sizeof(float) * (size_t)M * N);
        int8_t *q = malloc((size_t)N * K); float *sc = malloc(sizeof(float) * N);
        uint32_t s = 7; for (size_t i = 0; i < (size_t)M * K; i++) X[i] = frand(&s);
        for (size_t i = 0; i < (size_t)N * K; i++) q[i] = (int8_t)(frand(&s) * 127); for (int i = 0; i < N; i++) sc[i] = 0.01f;
        QiMat W = { QI_I8, N, K, q, sc, 0 };
        int act8 = argc > 5 && !strcmp(argv[5], "act8");
        if (act8) qi_gemm_act8(Y, N, X, K, M, &W, NULL); else qi_gemm(Y, X, M, &W, NULL);
        struct timespec a, b; clock_gettime(CLOCK_MONOTONIC, &a);
        int reps = 3; for (int r = 0; r < reps; r++) { if (act8) qi_gemm_act8(Y, N, X, K, M, &W, NULL); else qi_gemm(Y, X, M, &W, NULL); }
        clock_gettime(CLOCK_MONOTONIC, &b);
        double t = (b.tv_sec - a.tv_sec) + (b.tv_nsec - a.tv_nsec) * 1e-9;
        printf("M=%d N=%d K=%d %s: %.1f ms, %.1f GFLOP/s\n", M, N, K, act8 ? "int8 x int8 (VNNI)" : "int8 weights, f32 act", t / reps * 1e3, 2.0 * M * N * K * reps / t * 1e-9);
        return 0;
    }
    int ok = 1;
    int shapes[][3] = { {1,1,1}, {5,17,33}, {6,16,256}, {7,31,257}, {64,48,300}, {193,40,520}, {13,100,64} };
    for (unsigned i = 0; i < sizeof shapes / sizeof shapes[0]; i++)
        for (int fmt = 0; fmt < 3; fmt++) ok &= check(shapes[i][0], shapes[i][1], shapes[i][2], fmt, i & 1);
    /* The last three are wide in N with multiple m-blocks: they exercise the
     * hoisted-pack path, including a ragged n-panel tail. */
    int act_shapes[][3] = { {1,1,1}, {7,31,257}, {6,16,256}, {193,40,522}, {13,100,64},
                            {193,2065,260}, {385,4096,130}, {201,3073,130} };
    for (unsigned i = 0; i < sizeof act_shapes / sizeof act_shapes[0]; i++) ok &= check_act8(act_shapes[i][0], act_shapes[i][1], act_shapes[i][2]);
    /* whole 256-column panels (qi_gg_ok); narrow and wide N (the act8 pack per tile, and
     * hoisted out of the m-loop), several m-blocks, a ragged n-panel */
    int gg_shapes[][3] = { {1, 1, 256}, {7, 31, 512}, {193, 40, 768}, {385, 2065, 256}, {64, 17, 1024} };
    for (unsigned i = 0; i < sizeof gg_shapes / sizeof gg_shapes[0]; i++) ok &= check_gg(gg_shapes[i][0], gg_shapes[i][1], gg_shapes[i][2]);
    printf(ok ? "qi_gemm: all ok\n" : "qi_gemm: FAILED\n");
    return !ok;
}
