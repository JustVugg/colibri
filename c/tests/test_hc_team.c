/* hyper_connections.h: coli_hc_pre_team / coli_hc_post_team give the same
 * bits whatever the team (0 the default OpenMP team, 1 the calling thread,
 * 3 and 8 a small team for the mix), and the unsuffixed entry points are the
 * default team. test_hc_team.py builds this with the engines' flags. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "../hyper_connections.h"

static uint32_t state = 0x9e3779b9u;
static float next(void) {
    state = state * 1664525u + 1013904223u;
    return (float)((int32_t)(state >> 8) - (1 << 23)) / (float)(1 << 23);
}

static int same(const float *a, const float *b, size_t count, const char *what,
                int hc, int dimension, int team) {
    if (!memcmp(a, b, count * sizeof(*a))) return 1;
    for (size_t i = 0; i < count; i++)
        if (memcmp(a + i, b + i, sizeof(*a))) {
            printf("test_hc_team: %s differs at %zu (hc=%d dimension=%d team=%d): %a against %a\n",
                   what, i, hc, dimension, team, (double)a[i], (double)b[i]);
            break;
        }
    return 0;
}

enum { TEAMS = 4 };
static const int teams[TEAMS] = {0, 1, 3, 8};

/* not inlined, so the shape reaches the header as it does from an engine:
 * known only at run time */
#if defined(__GNUC__)
__attribute__((noinline))
#endif
static int check(int hc, int dimension) {
    int flattened = hc * dimension, mix_count = (2 + hc) * hc;
    float *input = malloc((size_t)flattened * sizeof(float));
    float *fn = malloc((size_t)mix_count * flattened * sizeof(float));
    float *base = malloc((size_t)mix_count * sizeof(float));
    float *branch = malloc((size_t)dimension * sizeof(float));
    float *out[TEAMS + 1], *post[TEAMS + 1], *comb[TEAMS + 1], *streams[TEAMS + 1];
    for (int t = 0; t <= TEAMS; t++) {
        out[t] = malloc((size_t)dimension * sizeof(float));
        post[t] = malloc((size_t)hc * sizeof(float));
        comb[t] = malloc((size_t)hc * hc * sizeof(float));
        streams[t] = malloc((size_t)flattened * sizeof(float));
    }
    float scale[3] = {0.75f, 1.25f, 0.5f};
    for (int i = 0; i < flattened; i++) input[i] = 3.0f * next();
    for (int i = 0; i < mix_count * flattened; i++) fn[i] = 0.05f * next();
    for (int i = 0; i < mix_count; i++) base[i] = next();
    for (int i = 0; i < dimension; i++) branch[i] = 2.0f * next();
    int ok = 1;
    for (int t = 0; t < TEAMS; t++) {
        if (coli_hc_pre_team(out[t], post[t], comb[t], input, fn, scale, base,
                             hc, dimension, 20, 1e-6f, 1e-6f, teams[t]) != 0) ok = 0;
        if (coli_hc_post_team(streams[t], branch, input, post[t], comb[t],
                              hc, dimension, teams[t]) != 0) ok = 0;
    }
    /* the unsuffixed entry points, in the last slot */
    if (coli_hc_pre(out[TEAMS], post[TEAMS], comb[TEAMS], input, fn, scale, base,
                    hc, dimension, 20, 1e-6f, 1e-6f) != 0 ||
        coli_hc_post(streams[TEAMS], branch, input, post[TEAMS], comb[TEAMS],
                     hc, dimension) != 0)
        ok = 0;
    for (int t = 1; t <= TEAMS; t++) {
        int team = t < TEAMS ? teams[t] : -1;
        ok = same(post[0], post[t], (size_t)hc, "post weights", hc, dimension, team) && ok;
        ok = same(comb[0], comb[t], (size_t)hc * hc, "comb", hc, dimension, team) && ok;
        ok = same(out[0], out[t], (size_t)dimension, "pre output", hc, dimension, team) && ok;
        ok = same(streams[0], streams[t], (size_t)flattened, "post output", hc, dimension, team) && ok;
    }
    for (int t = 0; t <= TEAMS; t++) {
        free(out[t]); free(post[t]); free(comb[t]); free(streams[t]);
    }
    free(branch); free(base); free(fn); free(input);
    return ok;
}

int main(void) {
    static const int shapes[][2] = {
        {4, 4096},   /* DeepSeek V4 Flash: 24 mix rows */
        {4, 1021},   /* a dimension that is no multiple of a vector width */
        {3, 37},     /* 15 rows: more rows than a team of 8, not a multiple of it */
        {8, 5},      /* 80 rows, the largest hc a loader accepts */
        {1, 9},      /* 3 rows: fewer rows than threads */
        {2, 64},
    };
    int ok = 1;
    for (size_t i = 0; i < sizeof(shapes) / sizeof(shapes[0]); i++)
        for (int repeat = 0; repeat < 3; repeat++)
            ok = check(shapes[i][0], shapes[i][1]) && ok;
    if (!ok) return 1;
    puts("test_hc_team: ok");
    return 0;
}
