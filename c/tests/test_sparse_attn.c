/* The vector attention kernel against the scalar reading of it.
 *
 * They are not bit-identical and are not meant to be: the vector path folds the
 * dot into four partial sums and fuses its multiply-adds. What is checked is that
 * the difference stays at fp32 accumulation noise across every shape the engine
 * can hand the kernel -- widths with and without a tail, lists that are entirely
 * unreachable, a single position, and a sink large enough to hold the softmax
 * down on its own.
 */
#include "../sparse_attn.h"

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static uint64_t seed = 0x9E3779B97F4A7C15ull;
static float next_float(void) {
    seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
    return (float)((double)(seed >> 11) / 9007199254740992.0) * 2.0f - 1.0f;
}

/* Relative to the magnitude of the row the kernel sums, not to 1: the outputs are
 * a convex combination of kv rows, so that is the scale the error lives on. */
static int agrees(const float *a, const float *b, int n, float tolerance, const char *what) {
    float worst = 0.0f;
    for (int i = 0; i < n; i++) {
        float delta = fabsf(a[i] - b[i]);
        if (delta > worst) worst = delta;
    }
    if (!(worst <= tolerance)) {
        fprintf(stderr, "%s: worst |delta| %.3e exceeds %.3e\n", what, worst, tolerance);
        return 0;
    }
    return 1;
}

static int one_case(int hd, int count, int rows, int holes, float sink, const char *what) {
    float *q = malloc((size_t)hd * sizeof(float));
    float *kv = malloc((size_t)rows * hd * sizeof(float));
    int *idx = malloc((size_t)count * sizeof(int));
    float *score_a = malloc((size_t)count * sizeof(float));
    float *score_b = malloc((size_t)count * sizeof(float));
    float *out_a = malloc((size_t)hd * sizeof(float));
    float *out_b = malloc((size_t)hd * sizeof(float));
    if (!q || !kv || !idx || !score_a || !score_b || !out_a || !out_b) {
        fprintf(stderr, "out of memory\n"); exit(1); }

    for (int i = 0; i < hd; i++) q[i] = next_float();
    for (int i = 0; i < rows * hd; i++) kv[i] = next_float();
    for (int j = 0; j < count; j++)
        idx[j] = (holes && (j % holes) == 0) ? -1 : (int)(seed % (uint64_t)rows);
    /* the scale the engine uses */
    float scale = 1.0f / sqrtf((float)hd);

    coli_sparse_attend_scalar(out_a, q, kv, idx, count, hd, sink, scale, score_a);
    coli_sparse_attend(out_b, q, kv, idx, count, hd, sink, scale, score_b);

    int ok = agrees(out_a, out_b, hd, 1e-5f, what);
    free(q); free(kv); free(idx); free(score_a); free(score_b); free(out_a); free(out_b);
    return ok;
}

/* A list with nothing reachable in it: every score is -INFINITY, the sink is the
 * whole denominator, and the output must be exactly zero -- not a NaN, which is
 * what a denominator of zero would give. */
static int all_holes(void) {
    enum { HD = 64, COUNT = 12 };
    float q[HD], kv[HD], out[HD], score[COUNT];
    int idx[COUNT];
    for (int i = 0; i < HD; i++) { q[i] = next_float(); kv[i] = next_float(); }
    for (int j = 0; j < COUNT; j++) idx[j] = -1;
    coli_sparse_attend(out, q, kv, idx, COUNT, HD, 0.0f, 1.0f, score);
    for (int i = 0; i < HD; i++)
        if (out[i] != 0.0f) {
            fprintf(stderr, "unreachable list: out[%d] is %g, expected 0\n", i, out[i]);
            return 0;
        }
    return 1;
}

int main(void) {
    struct { int hd, count, rows, holes; float sink; const char *what; } cases[] = {
        { 512, 640, 700, 0,  0.0f,  "released shape (hd 512, window + index_topk)" },
        { 512, 640, 700, 5,  0.0f,  "released shape with unreachable slots" },
        {  64, 128, 130, 0,  0.0f,  "fixture shape (hd 64)" },
        {  32,  33,  40, 0,  0.0f,  "hd 32, one vector block, no tail" },
        {  40,  17,  20, 3,  0.0f,  "hd 40, an 8-wide block past the 32-wide loop" },
        {  33,   9,  12, 0,  0.0f,  "hd 33, scalar tail of one" },
        {   7,   5,   6, 2,  0.0f,  "hd 7, entirely scalar tail" },
        {   1,   3,   4, 0,  0.0f,  "hd 1" },
        { 512,   1,   2, 0,  0.0f,  "a single reachable position" },
        {  64,  48,  50, 0, 30.0f,  "a sink that dominates the softmax" },
        {  64,  48,  50, 0, -30.0f, "a sink that does not reach it" },
    };
    int failures = 0;
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++)
        if (!one_case(cases[i].hd, cases[i].count, cases[i].rows, cases[i].holes,
                      cases[i].sink, cases[i].what)) failures++;
    if (!all_holes()) failures++;
    if (failures) { fprintf(stderr, "sparse attention: %d case(s) failed\n", failures); return 1; }
    printf("sparse attention: vector and scalar agree on %zu shapes, and an "
           "unreachable list gives zero\n", sizeof(cases) / sizeof(cases[0]) + 1);
    return 0;
}
