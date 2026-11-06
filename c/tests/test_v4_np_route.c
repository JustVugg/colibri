/* V4_ROUTER_AVX2 (v4_np_route.h): the AVX2 router must give
 * exactly what the scalar route gives (coli_v4_route_bf16, the ROUTE_BF16 unit
 * linked as built for the engine): the same return value, the same indices in
 * the same order, the same weights bit for bit.
 *   (default)          random gates; near ties (a row one bf16 ulp from the
 *                      K-th pick, placed at the K/K+1 boundary); exact ties
 *                      (duplicate rows and biases); odd shapes (cols % 32 != 0,
 *                      few experts, topk 1 and topk == experts); extreme scales;
 *                      the logit bound holds on every logit
 *   --real G I         real gate rows (G: "V4NG" gates.bin) and real MoE inputs
 *                      (I: "V4NI" inputs.bin): each input of layer L through
 *                      layer L+1's gate (K=8) and L+2's (K=6), as a router runs them
 *                      runs them
 *   --bench            time per call, scalar vs AVX2, gates cycled beyond L3
 * Prints "v4 np route avx2: ok" and exits 0 on success. */
#include "../v4_np_route.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef V4_NP_ROUTE_AVX2
int main(void) {
    printf("v4 np route avx2: skipped (built without AVX2/FMA)\n");
    return 0;
}
#else

static int failures;
static void router_mode(int enabled) {
#ifdef _WIN32
    _putenv_s("V4_ROUTER_AVX2", enabled ? "1" : "0");
#else
    setenv("V4_ROUTER_AVX2", enabled ? "1" : "0", 1);
#endif
}
#define CHECK(c, ...) do { if (!(c)) { failures++; if (failures <= 20) { \
    fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
    fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } } } while (0)

static uint64_t rng = 0x9e3779b97f4a7c15ull;
static uint64_t next_u64(void) {
    rng ^= rng >> 12; rng ^= rng << 25; rng ^= rng >> 27;
    return rng * 0x2545f4914f6cdd1dull;
}
static double uniform(void) { return (double)(next_u64() >> 11) / 9007199254740992.0; }
static float gauss(void) {
    double u = uniform() + 1e-300, v = uniform();
    return (float)(sqrt(-2.0 * log(u)) * cos(6.283185307179586 * v));
}
static uint16_t to_bf16(float x) {   /* round to nearest even */
    uint32_t bits;
    memcpy(&bits, &x, 4);
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}
static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* the same bound the header uses, for the bound check */
static float logit_bound(float magnitude_sum, int cols) {
    const double u = 1.0 / 16777216.0;
    double m = (double)(cols / 16 + 8 + (cols & 31)), n = (double)cols + 1.0;
    double gm = m * u / (1.0 - m * u), gn = n * u / (1.0 - n * u);
    return (float)((gm + gn) / (1.0 - gm) * 1.001) * magnitude_sum + 1e-35f;
}

typedef struct {
    long calls, plain_differs, refined_sum, refined_max, fallback;
    double bound_ratio_max;
} Tally;

/* One comparison: scalar vs AVX2 (must be identical) and vs the plain AVX2
 * logits without the scalar recompute (counted, allowed to differ). */
static void compare(Tally *t, const uint16_t *gate, const float *bias, const float *hidden,
                    int experts, int cols, int topk, float scale, const char *what,
                    int check_bound) {
    float w1[V4_NP_ROUTE_MAX_EXPERTS], w2[V4_NP_ROUTE_MAX_EXPERTS], w3[V4_NP_ROUTE_MAX_EXPERTS];
    int i1[V4_NP_ROUTE_MAX_EXPERTS], i2[V4_NP_ROUTE_MAX_EXPERTS], i3[V4_NP_ROUTE_MAX_EXPERTS];
    float *scratch = malloc((size_t)cols * sizeof(float));
    memset(w1, 0, sizeof w1); memset(w2, 0, sizeof w2);
    memset(i1, 0, sizeof i1); memset(i2, 0, sizeof i2);
    int r1 = coli_v4_route_bf16_scalar(w1, i1, hidden, gate, bias, NULL, experts, cols, topk, scale);
    for (int mode=0;mode<=1;mode++) {
        router_mode(mode);
        int public_result=coli_v4_route_bf16(w3,i3,hidden,gate,bias,NULL,experts,cols,topk,scale);
        CHECK(public_result==r1,"%s: public router mode=%d return differs",what,mode);
        if(!r1 && !public_result) {
            CHECK(!memcmp(i1,i3,(size_t)topk*sizeof(int)),"%s: public router mode=%d IDs differ",what,mode);
            CHECK(!memcmp(w1,w3,(size_t)topk*sizeof(float)),"%s: public router mode=%d weights differ",what,mode);
        }
    }
    router_mode(0);
    int refined = -2;
    int r2 = v4_np_route_bf16_avx2(w2, i2, hidden, gate, bias, experts, cols, topk, scale,
                                   scratch, &refined);
    t->calls++;
    CHECK(r1 == r2, "%s: return %d vs %d", what, r1, r2);
    if (r1 == 0 && r2 == 0) {
        CHECK(!memcmp(i1, i2, (size_t)topk * sizeof(int)), "%s: indices differ (rank0 %d vs %d)",
              what, i1[0], i2[0]);
        CHECK(!memcmp(w1, w2, (size_t)topk * sizeof(float)), "%s: weights differ", what);
    }
    if (refined < 0) t->fallback++;
    else {
        CHECK(refined >= topk && refined <= experts, "%s: refined %d", what, refined);
        t->refined_sum += refined;
        if (refined > t->refined_max) t->refined_max = refined;
    }
    /* the plain vector logits straight into the selection: what the refine buys */
    float fast[V4_NP_ROUTE_MAX_EXPERTS], sums[V4_NP_ROUTE_MAX_EXPERTS];
    v4_np_gemv_bf16(fast, sums, gate, hidden, scratch, experts, cols);
    int r3 = coli_v4_route_bf16_logits(w3, i3, hidden, gate, bias, NULL, experts, cols, topk,
                                       scale, fast);
    if (r3 != r1 || (r1 == 0 && memcmp(i1, i3, (size_t)topk * sizeof(int))))
        t->plain_differs++;
    if (check_bound) {
        float exact[V4_NP_ROUTE_MAX_EXPERTS];
        int rows[V4_NP_ROUTE_MAX_EXPERTS];
        for (int e = 0; e < experts; e++) rows[e] = e;
        coli_v4_route_bf16_rows(exact, gate, hidden, rows, experts, cols);
        for (int e = 0; e < experts; e++) {
            if (!isfinite(fast[e]) || !isfinite(exact[e])) continue;
            double b = logit_bound(sums[e], cols), diff = fabs((double)fast[e] - exact[e]);
            CHECK(diff <= b, "%s: expert %d |fast-scalar| %.3g over bound %.3g", what, e, diff, b);
            if (b > 0 && diff / b > t->bound_ratio_max) t->bound_ratio_max = diff / b;
        }
    }
    free(scratch);
}

static void fill(uint16_t *gate, float *bias, float *hidden, int experts, int cols,
                 float gate_sigma, float hidden_sigma) {
    for (size_t i = 0; i < (size_t)experts * cols; i++) gate[i] = to_bf16(gauss() * gate_sigma);
    for (int e = 0; e < experts; e++) bias[e] = gauss() * 0.05f;
    for (int c = 0; c < cols; c++) hidden[c] = gauss() * hidden_sigma;
}

static void report(const char *name, const Tally *t) {
    printf("  %-26s calls=%-6ld refined avg=%.2f max=%ld fallback=%ld plain_avx2_differs=%ld"
           " bound_use_max=%.3f\n", name, t->calls,
           t->calls - t->fallback ? (double)t->refined_sum / (double)(t->calls - t->fallback) : 0.0,
           t->refined_max, t->fallback, t->plain_differs, t->bound_ratio_max);
}

static void run_synthetic(void) {
    enum { E = 256, D = 4096, DMAX = 4224 };   /* DMAX: the widest odd shape */
    uint16_t *gate = malloc((size_t)E * DMAX * sizeof(uint16_t));
    float *bias = malloc(E * sizeof(float)), *hidden = malloc(DMAX * sizeof(float));
    char what[96];
    /* 1. random gates, the engine's shape, K = 8 and 6, with and without bias */
    Tally random = {0};
    for (int trial = 0; trial < 400; trial++) {
        if (trial % 20 == 0) fill(gate, bias, hidden, E, D, 0.02f, 1.0f);
        else for (int c = 0; c < D; c++) hidden[c] = gauss();
        snprintf(what, sizeof what, "random %d", trial);
        compare(&random, gate, trial & 1 ? bias : NULL, hidden, E, D, trial & 2 ? 6 : 8, 1.5f,
                what, trial % 10 == 0);
    }
    report("random 256x4096", &random);
    /* 2. near ties at the boundary: the best unpicked expert becomes the K-th
     * pick's row with one element moved by one bf16 ulp (same bias), so the two
     * selection values differ in the last bits only */
    Tally near = {0};
    for (int trial = 0; trial < 300; trial++) {
        fill(gate, bias, hidden, E, D, 0.02f, 1.0f);
        int k = 6 + (trial % 3), idx[V4_NP_ROUTE_MAX_EXPERTS];
        float w[V4_NP_ROUTE_MAX_EXPERTS];
        if (coli_v4_route_bf16_scalar(w, idx, hidden, gate, bias, NULL, E, D, k + 1, 1.5f)) continue;
        int last = idx[k - 1], rival = idx[k];
        memcpy(gate + (size_t)rival * D, gate + (size_t)last * D, D * sizeof(uint16_t));
        bias[rival] = bias[last];
        /* half the trials at the column of the smallest |hidden|: a gap of
         * ~1e-8, far inside the vector sum's rounding */
        int c = (int)(next_u64() % D);
        if (trial & 1)
            for (int j = 0; j < D; j++)
                if (fabsf(hidden[j]) < fabsf(hidden[c])) c = j;
        gate[(size_t)rival * D + c] += (trial & 4) ? 1 : (uint16_t)-1;
        snprintf(what, sizeof what, "near tie %d", trial);
        compare(&near, gate, bias, hidden, E, D, k, 1.5f, what, trial % 10 == 0);
    }
    report("near ties", &near);
    /* 3. exact ties: duplicate rows and biases (tie-break by the lower id) */
    Tally ties = {0};
    for (int trial = 0; trial < 100; trial++) {
        fill(gate, bias, hidden, E, D, 0.02f, 1.0f);
        int idx[V4_NP_ROUTE_MAX_EXPERTS];
        float w[V4_NP_ROUTE_MAX_EXPERTS];
        if (coli_v4_route_bf16_scalar(w, idx, hidden, gate, bias, NULL, E, D, 8, 1.5f)) continue;
        for (int r = 0; r < 4; r++) {
            int from = idx[4 + r], to = (int)(next_u64() % E);
            memcpy(gate + (size_t)to * D, gate + (size_t)from * D, D * sizeof(uint16_t));
            bias[to] = bias[from];
        }
        snprintf(what, sizeof what, "exact tie %d", trial);
        compare(&ties, gate, trial & 1 ? bias : NULL, hidden, E, D, 8, 1.5f, what, 0);
    }
    for (int c = 0; c < D; c++) hidden[c] = 0.0f;   /* every logit 0: all tied */
    compare(&ties, gate, NULL, hidden, E, D, 8, 1.5f, "zero input", 1);
    report("exact ties", &ties);
    /* 4. odd shapes and scales */
    Tally odd = {0};
    static const int shapes[][3] = {   /* experts, cols, topk */
        {256, 4000, 8}, {256, 100, 6}, {256, 33, 6}, {256, 7, 4}, {256, 31, 16},
        {64, 4096, 6}, {6, 4096, 6}, {16, 64, 16}, {256, 4096, 1}, {1, 32, 1},
        {200, 2048, 16}, {256, 4128, 8},
    };
    for (size_t s = 0; s < sizeof shapes / sizeof shapes[0]; s++)
        for (int trial = 0; trial < 20; trial++) {
            int e = shapes[s][0], d = shapes[s][1], k = shapes[s][2];
            fill(gate, bias, hidden, e, d, 0.02f, 1.0f);
            snprintf(what, sizeof what, "shape %dx%d k%d #%d", e, d, k, trial);
            compare(&odd, gate, trial & 1 ? bias : NULL, hidden, e, d, k, 1.5f, what, 1);
        }
    static const float scales[][2] = {{0.02f, 1e-3f}, {0.02f, 1e3f}, {2.0f, 30.0f},
                                      {1e-4f, 1.0f}, {0.3f, 0.3f}};
    for (size_t s = 0; s < sizeof scales / sizeof scales[0]; s++)
        for (int trial = 0; trial < 20; trial++) {
            fill(gate, bias, hidden, E, D, scales[s][0], scales[s][1]);
            snprintf(what, sizeof what, "scale %g/%g #%d", scales[s][0], scales[s][1], trial);
            compare(&odd, gate, bias, hidden, E, D, 8, 1.5f, what, 1);
        }
    report("odd shapes and scales", &odd);
    /* 5. what falls back to the scalar route: non-finite input, too many experts */
    Tally edge = {0};
    fill(gate, bias, hidden, E, D, 0.02f, 1.0f);
    hidden[17] = NAN;
    compare(&edge, gate, bias, hidden, E, D, 8, 1.5f, "nan input", 0);
    hidden[17] = INFINITY;
    compare(&edge, gate, bias, hidden, E, D, 8, 1.5f, "inf input", 0);
    CHECK(edge.fallback == 2, "non-finite inputs fell back %ld/2", edge.fallback);
    {
        uint16_t *big = calloc((size_t)300 * 64, sizeof(uint16_t));
        float big_bias[300];
        for (int e = 0; e < 300; e++) big_bias[e] = 0.001f * (float)e;
        for (int i = 0; i < 300 * 64; i++) big[i] = to_bf16(gauss() * 0.1f);
        float w1[16], w2[16], h[64], scratch[64];
        int i1[16], i2[16], refined = 0;
        for (int c = 0; c < 64; c++) h[c] = gauss();
        int r1 = coli_v4_route_bf16_scalar(w1, i1, h, big, big_bias, NULL, 300, 64, 8, 1.5f);
        int r2 = v4_np_route_bf16_avx2(w2, i2, h, big, big_bias, 300, 64, 8, 1.5f, scratch,
                                       &refined);
        CHECK(r1 == r2 && refined == -1 && !memcmp(i1, i2, sizeof(int) * 8) &&
              !memcmp(w1, w2, sizeof(float) * 8), "300 experts: not the scalar fallback");
        router_mode(1);
        r2=coli_v4_route_bf16(w2,i2,h,big,big_bias,NULL,300,64,8,1.5f);
        CHECK(r1==r2&&!memcmp(i1,i2,sizeof(int)*8)&&!memcmp(w1,w2,sizeof(float)*8),"public 300-expert fallback differs");
        int forced[2]={255,0};
        r1=coli_v4_route_bf16_scalar(w1,i1,h,big,big_bias,forced,256,64,2,1.5f);
        r2=coli_v4_route_bf16(w2,i2,h,big,big_bias,forced,256,64,2,1.5f);
        CHECK(r1==r2&&!memcmp(i1,i2,sizeof(int)*2)&&!memcmp(w1,w2,sizeof(float)*2),"forced router differs");
        router_mode(0);
        free(big);
    }
    {
        uint16_t *wide=calloc((size_t)2*8193,sizeof(uint16_t));
        float *h=calloc(8193,sizeof(float)),w1[2],w2[2];int i1[2],i2[2];
        CHECK(wide&&h,"wide fallback allocation");
        if(wide&&h) {
            router_mode(1);
            int r1=coli_v4_route_bf16_scalar(w1,i1,h,wide,NULL,NULL,2,8193,2,1);
            int r2=coli_v4_route_bf16(w2,i2,h,wide,NULL,NULL,2,8193,2,1);
            CHECK(r1==r2&&!memcmp(w1,w2,sizeof(w1))&&!memcmp(i1,i2,sizeof(i1)),"8193-column fallback differs");
            CHECK(coli_v4_route_bf16(w2,i2,NULL,wide,NULL,NULL,2,8193,2,1)==
                  coli_v4_route_bf16_scalar(w1,i1,NULL,wide,NULL,NULL,2,8193,2,1),"null-input fallback differs");
            router_mode(0);
        }
        free(wide);free(h);
    }
    report("fallbacks", &edge);
    free(gate); free(bias); free(hidden);
}

static void *slurp(const char *path, size_t *size) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    void *data = n > 0 ? malloc((size_t)n) : NULL;
    if (data && fread(data, 1, (size_t)n, f) != (size_t)n) { free(data); data = NULL; }
    fclose(f);
    *size = data ? (size_t)n : 0;
    return data;
}

/* gates.bin: "V4NG", i32 count, n, d; per layer: i32 layer, bf16[n*d], f32[n].
 * inputs.bin: "V4NI", i32 count, d; per record: i32 layer, i32 token, f32[d]. */
static int run_real(const char *gates_path, const char *inputs_path) {
    size_t gsize = 0, isize = 0;
    unsigned char *g = slurp(gates_path, &gsize), *in = slurp(inputs_path, &isize);
    if (!g || !in || gsize < 16 || isize < 12 || memcmp(g, "V4NG", 4) || memcmp(in, "V4NI", 4)) {
        fprintf(stderr, "cannot read %s / %s\n", gates_path, inputs_path);
        return 2;
    }
    int32_t count, n, d, records, di;
    memcpy(&count, g + 4, 4); memcpy(&n, g + 8, 4); memcpy(&d, g + 12, 4);
    memcpy(&records, in + 4, 4); memcpy(&di, in + 8, 4);
    size_t per = 4 + (size_t)n * d * 2 + (size_t)n * 4;
    if (di != d || n > V4_NP_ROUTE_MAX_EXPERTS || gsize != 16 + (size_t)count * per ||
        isize != 12 + (size_t)records * (8 + (size_t)d * 4)) {
        fprintf(stderr, "unexpected sizes\n");
        return 2;
    }
    const uint16_t *gate_of[128] = {0};
    const float *bias_of[128] = {0};
    for (int i = 0; i < count; i++) {
        const unsigned char *p = g + 16 + (size_t)i * per;
        int32_t layer;
        memcpy(&layer, p, 4);
        if (layer < 0 || layer >= 128) continue;
        gate_of[layer] = (const uint16_t *)(p + 4);
        bias_of[layer] = (const float *)(p + 4 + (size_t)n * d * 2);
    }
    Tally d1 = {0}, d2 = {0};
    double scalar_s = 0, vector_s = 0;
    float *hidden = malloc((size_t)d * sizeof(float)), *scratch = malloc((size_t)d * sizeof(float));
    for (int r = 0; r < records; r++) {
        const unsigned char *p = in + 12 + (size_t)r * (8 + (size_t)d * 4);
        int32_t layer, token;
        memcpy(&layer, p, 4); memcpy(&token, p + 4, 4);
        memcpy(hidden, p + 8, (size_t)d * 4);
        char what[64];
        for (int depth = 1; depth <= 2; depth++) {
            int target = layer + depth, k = depth == 1 ? 8 : 6;
            if (target >= 128 || !gate_of[target]) continue;
            snprintf(what, sizeof what, "real L%d->%d tok %d", layer, target, token);
            compare(depth == 1 ? &d1 : &d2, gate_of[target], bias_of[target], hidden, n, d, k,
                    1.5f, what, r % 16 == 0);
            float w[16];
            int idx[16], refined;
            double t0 = now();
            coli_v4_route_bf16_scalar(w, idx, hidden, gate_of[target], bias_of[target], NULL, n, d, k,
                               1.5f);
            double t1 = now();
            v4_np_route_bf16_avx2(w, idx, hidden, gate_of[target], bias_of[target], n, d, k,
                                  1.5f, scratch, &refined);
            scalar_s += t1 - t0;
            vector_s += now() - t1;
        }
    }
    report("real depth 1 (K=8)", &d1);
    report("real depth 2 (K=6)", &d2);
    long calls = d1.calls + d2.calls;
    printf("  real per call: scalar %.3f ms, avx2 %.3f ms (gates warm in cache order)\n",
           calls ? scalar_s / calls * 1e3 : 0.0, calls ? vector_s / calls * 1e3 : 0.0);
    free(hidden); free(scratch); free(g); free(in);
    return 0;
}

static void run_bench(void) {
    enum { E = 256, D = 4096, NG = 80 };   /* 160 MiB of gates: beyond L3 */
    uint16_t *gates = malloc((size_t)NG * E * D * sizeof(uint16_t));
    __attribute__((aligned(32))) float bias[E], hidden[D], scratch[D], fast[E], sums[E];
    __attribute__((aligned(32))) float w[16];
    int idx[16], refined;
    for (size_t i = 0; i < (size_t)NG * E * D; i++) gates[i] = to_bf16(gauss() * 0.02f);
    for (int e = 0; e < E; e++) bias[e] = gauss() * 0.05f;
    for (int c = 0; c < D; c++) hidden[c] = gauss();
    const int reps = 400;
    double t0 = now();
    for (int i = 0; i < reps; i++)
        coli_v4_route_bf16_scalar(w, idx, hidden, gates + (size_t)(i % NG) * E * D, bias, NULL, E, D, 8, 1.5f);
    double scalar = (now() - t0) / reps;
    long refined_sum = 0;
    t0 = now();
    for (int i = 0; i < reps; i++) {
        v4_np_route_bf16_avx2(w, idx, hidden, gates + (size_t)(i % NG) * E * D, bias, E, D, 8, 1.5f,
                              scratch, &refined);
        refined_sum += refined;
    }
    double vector = (now() - t0) / reps;
    t0 = now();
    for (int i = 0; i < reps; i++)
        v4_np_gemv_bf16(fast, sums, gates + (size_t)(i % NG) * E * D, hidden, scratch, E, D);
    double gemv = (now() - t0) / reps;
    printf("  bench per call (256x4096, K=8, cold gates): scalar route %.3f ms, avx2 route %.3f ms"
           " (gemv %.3f ms, refined avg %.2f)\n", scalar * 1e3, vector * 1e3, gemv * 1e3,
           (double)refined_sum / reps);
    free(gates);
}

int main(int argc, char **argv) {
    if (argc > 1 && !strcmp(argv[1], "--bench")) { run_bench(); return 0; }
    if (argc > 3 && !strcmp(argv[1], "--real")) {
        int rc = run_real(argv[2], argv[3]);
        if (rc) return rc;
    } else {
        run_synthetic();
    }
    if (failures) {
        printf("v4 np route avx2: %d FAILURES\n", failures);
        return 1;
    }
    printf("v4 np route avx2: ok\n");
    return 0;
}
#endif
