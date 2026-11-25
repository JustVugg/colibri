/* deepseek_v41.c: v41_hc_post_team selects the calling thread (team = 1) at
 * single-token decode (n == 1) and the default OpenMP team (team = 0) for
 * multi-token batches, and coli_hc_post_team(..., 1) is bit-identical to
 * coli_hc_post_team(..., 0) and coli_hc_post(...) across V4.1, V4, tiny, and
 * odd shapes. */
#define main dsv41_main_unused
#include "../deepseek_v41.c"
#undef main

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static int g_fails;
static void check(int ok, const char *what) {
    printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) g_fails++;
}

static uint32_t g_rng = 20261113u;
static float uniform(void) {
    g_rng = g_rng * 1664525u + 1013904223u;
    return (float)((g_rng >> 8) & 0xffffff) / 8388608.0f - 1.0f;
}

static void check_shape(int hc, int dim) {
    size_t hd = (size_t)hc * (size_t)dim;
    float *residual = malloc(hd * sizeof(float));
    float *branch = malloc((size_t)dim * sizeof(float));
    float *post = malloc((size_t)hc * sizeof(float));
    float *comb = malloc((size_t)hc * (size_t)hc * sizeof(float));
    float *out_mt = malloc(hd * sizeof(float));
    float *out_st = malloc(hd * sizeof(float));
    float *out_ref = malloc(hd * sizeof(float));
    if (!residual || !branch || !post || !comb || !out_mt || !out_st || !out_ref) {
        fprintf(stderr, "oom\n");
        exit(2);
    }

    for (size_t i = 0; i < hd; i++)
        residual[i] = uniform() * ldexpf(1.0f, (int)(i % 23) - 11);
    for (int i = 0; i < dim; i++)
        branch[i] = uniform() * 4.0f;
    for (int i = 0; i < hc; i++)
        post[i] = 2.0f * (uniform() + 1.0f);
    for (int i = 0; i < hc * hc; i++)
        comb[i] = (uniform() + 1.0f) / (float)hc;

    memset(out_mt, 0xa5, hd * sizeof(float));
    memset(out_st, 0x5a, hd * sizeof(float));
    memset(out_ref, 0x3c, hd * sizeof(float));

    int r_mt = coli_hc_post_team(out_mt, branch, residual, post, comb, hc, dim, v41_hc_post_team(4));
    int r_st = coli_hc_post_team(out_st, branch, residual, post, comb, hc, dim, v41_hc_post_team(1));
    int r_ref = coli_hc_post(out_ref, branch, residual, post, comb, hc, dim);

    char msg[128];
    snprintf(msg, sizeof(msg), "hc_post_team %dx%d: n=1 bit-identical to multi-threaded and coli_hc_post", hc, dim);
    check(r_mt == 0 && r_st == 0 && r_ref == 0 &&
          memcmp(out_mt, out_st, hd * sizeof(float)) == 0 &&
          memcmp(out_ref, out_st, hd * sizeof(float)) == 0, msg);

    free(residual);
    free(branch);
    free(post);
    free(comb);
    free(out_mt);
    free(out_st);
    free(out_ref);
}

int main(void) {
    check(v41_hc_post_team(1) == 1, "v41_hc_post_team(1) selects calling thread (1)");
    check(v41_hc_post_team(2) == 0 && v41_hc_post_team(4) == 0 &&
          v41_hc_post_team(16) == 0 && v41_hc_post_team(128) == 0,
          "v41_hc_post_team(n > 1) selects default OpenMP team (0)");

    check_shape(4, 5120);
    check_shape(4, 4096);
    check_shape(4, 128);
    check_shape(4, 5121);
    check_shape(3, 7);
    check_shape(1, 1);

    float x[8] = {0};
#ifdef _OPENMP
    int before = omp_get_max_threads();
    check(coli_hc_post_team(x, x, x, x, x, 1, 1, v41_hc_post_team(1)) == 0 &&
          omp_get_max_threads() == before,
          "coli_hc_post_team(..., 1) preserves omp_get_max_threads()");
#endif
    check(coli_hc_post_team(NULL, x, x, x, x, 1, 1, 1) == -1 &&
          coli_hc_post_team(x, x, x, x, x, 0, 1, 1) == -1 &&
          coli_hc_post_team(x, x, x, x, x, 1, 0, 1) == -1,
          "coli_hc_post_team refuses NULL and empty shapes");

    printf("%s: %d failure(s)\n", g_fails ? "FAIL" : "OK", g_fails);
    return g_fails != 0;
}
