/* The SERVE sampler (k1_sample) as a component.
 *
 *   - a row with any NaN / +Inf / -Inf yields -1 (no token is ever drawn from it), greedy or sampled
 *   - temperature <= 0: the first maximum (lowest id on a tie), the greedy rule
 *   - top_k keeps every logit >= the K-th largest SCALED (logit / temperature, FP32) value (the vendor stack's rule: ties at the
 *     K-th value all stay), so top_k 1 samples among the logits tied at the maximum
 *   - top_p keeps the smallest descending prefix whose mass reaches p of the survivors; among equal logits the
 *     highest id comes first (reverse of the vendor stack's ascending stable sort: it removes the lowest id first)
 *   - the selection path (top_k < V) draws the same token as an independent full-sort reference,
 *     on random rows with many exact ties, for the same random stream
 *   - empirical frequencies match the analytic softmax within 5 sigma; the same seed gives the same stream
 *   tests/test_kolibri1_serve_sampler */
#define _GNU_SOURCE
#define KOLIBRI1_NO_MAIN
#include "../kolibri1.c"

static int failures;
#define CHECK(cond, ...) do { if (!(cond)) { failures++; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

static uint64_t g_state = 1;
static double rnd(void) { return k1_unit(&g_state); }

/* the reference: full sort by (logit desc, id desc), threshold = k-th value, softmax, cut, draw with the same stream */
static int ref_sample(const float *lg0, int V, float temp, float top_p, int top_k, uint64_t *rng) {
    float *lg = malloc((size_t)V * sizeof(float));       /* FP32 temperature scaling first, as the vendor stack does */
    for (int i = 0; i < V; i++) lg[i] = temp > 0 ? lg0[i] / temp : lg0[i];
    int *order = malloc((size_t)V * sizeof(int));
    for (int i = 0; i < V; i++) order[i] = i;
    for (int i = 1; i < V; i++) {                       /* insertion sort, stable on id */
        int x = order[i], j = i - 1;
        while (j >= 0 && (lg[order[j]] < lg[x] || (lg[order[j]] == lg[x] && order[j] < x))) { order[j + 1] = order[j]; j--; }
        order[j + 1] = x;
    }
    int n = V;
    if (top_k > 0 && top_k < V) { n = 0; while (n < V && lg[order[n]] >= lg[order[top_k - 1]]) n++; }
    long double top = lg[order[0]], total = 0;
    for (int i = 0; i < n; i++) total += expl(((long double)lg[order[i]] - top));
    long double cut = (top_p > 0 && top_p < 1) ? top_p * total : total, kept = 0;
    int m = 0;
    while (m < n && kept < cut) { kept += expl(((long double)lg[order[m]] - top)); m++; }
    long double draw = k1_unit(rng) * kept, acc = 0;
    int pick = order[m - 1];
    for (int i = 0; i < m; i++) { acc += expl(((long double)lg[order[i]] - top)); if (acc >= draw) { pick = order[i]; break; } }
    free(order); free(lg);
    return pick;
}

int main(void) {
    enum { V = 512 };
    float lg[V];
    K1Cand *cand = malloc(sizeof(K1Cand) * V);
    uint64_t rng = 123;

    /* non-finite rows */
    for (int kind = 0; kind < 3; kind++)
        for (int where = 0; where < V; where += 97) {
            for (int i = 0; i < V; i++) lg[i] = (float)(i % 7);
            lg[where] = kind == 0 ? NAN : kind == 1 ? INFINITY : -INFINITY;
            CHECK(k1_sample(lg, V, 0.0f, 1.0f, 0, &rng, cand) == -1, "greedy accepted a non-finite row");
            CHECK(k1_sample(lg, V, 1.0f, 0.9f, 128, &rng, cand) == -1, "sampled accepted a non-finite row");
        }

    /* greedy, ties */
    for (int i = 0; i < V; i++) lg[i] = 1.0f;
    lg[40] = lg[300] = 5.0f;
    CHECK(k1_sample(lg, V, 0.0f, 1.0f, 0, &rng, cand) == 40, "greedy tie rule");
    CHECK(k1_sample(lg, V, -1.0f, 1.0f, 0, &rng, cand) == 40, "negative temperature is greedy");
    int hit40 = 0, hit300 = 0;                           /* top_k 1 keeps BOTH maxima, like the vendor stack, and samples among them */
    for (int t = 0; t < 400; t++) {
        int p = k1_sample(lg, V, 3.0f, 1.0f, 1, &rng, cand);
        CHECK(p == 40 || p == 300, "top_k 1 kept %d", p);
        hit40 += p == 40; hit300 += p == 300;
    }
    CHECK(hit40 > 120 && hit300 > 120, "top_k 1 is uniform over the tied maxima: %d / %d", hit40, hit300);
    /* ... and top_p <  1 over that tie: ascending removal takes the lowest id first, the highest survives */
    for (int t = 0; t < 200; t++) CHECK(k1_sample(lg, V, 3.0f, 0.4f, 1, &rng, cand) == 300, "top_k 1 with top_p 0.4 over a 2-way tie");
    for (int t = 0; t < 200; t++) { int p = k1_sample(lg, V, 100.0f, 1.0f, 2, &rng, cand); CHECK(p == 40 || p == 300, "top_k 2 kept %d", p); }
    /* A tie created by the FP32 temperature division. 1.5000001192092896 and 1.500000238418579 are
     * adjacent floats; both divided by 1.5f give the same float, so top_k 1 (top_p 1) keeps BOTH. */
    {
        for (int i = 0; i < V; i++) lg[i] = -10.0f;
        lg[11] = 1.5000001192092896f; lg[12] = 1.500000238418579f;
        CHECK(lg[11] != lg[12] && lg[11] / 1.5f == lg[12] / 1.5f, "the pair is equal after the FP32 division");
        int s11 = 0, s12 = 0;
        for (int t = 0; t < 600; t++) {
            int p = k1_sample(lg, V, 1.5f, 1.0f, 1, &rng, cand);
            CHECK(p == 11 || p == 12, "temperature-created tie kept %d", p);
            s11 += p == 11; s12 += p == 12;
        }
        CHECK(s11 > 200 && s12 > 200, "both admitted: %d / %d", s11, s12);
        for (int t = 0; t < 50; t++) CHECK(k1_sample(lg, V, 1.0f, 1.0f, 1, &rng, cand) == 12, "at T=1 they differ: only the larger");
    }
    /* the k best, a tie at the cut: every logit equal to the K-th stays (as in the vendor stack) */
    for (int i = 0; i < V; i++) lg[i] = 0.0f;
    lg[10] = 9; lg[20] = 8; lg[30] = lg[31] = lg[32] = 7;
    int seen[V] = {0};
    for (int t = 0; t < 3000; t++) seen[k1_sample(lg, V, 50.0f, 1.0f, 4, &rng, cand)]++;
    CHECK(seen[10] && seen[20] && seen[30] && seen[31] && seen[32] && !seen[0], "top_k 4 over a 3-way tie: ids 30,31,32 all in");

    /* top_p on a known distribution: probabilities .5 .3 .15 .05 (logits ln p) */
    for (int i = 0; i < V; i++) lg[i] = -50.0f;
    lg[7] = logf(0.5f); lg[3] = logf(0.3f); lg[9] = logf(0.15f); lg[1] = logf(0.05f);
    int c[V] = {0}; int N = 40000;
    for (int t = 0; t < N; t++) c[k1_sample(lg, V, 1.0f, 0.7f, 0, &rng, cand)]++;
    CHECK(!c[9] && !c[1], "top_p 0.7 keeps {.5,.3}: got %d %d", c[9], c[1]);
    double f7 = (double)c[7] / N, sd = sqrt(0.625 * 0.375 / N);
    CHECK(fabs(f7 - 0.5 / 0.8) < 5 * sd, "renormalised .5/.8: %.4f", f7);
    memset(c, 0, sizeof c);
    for (int t = 0; t < N; t++) c[k1_sample(lg, V, 1.0f, 1.0f, 0, &rng, cand)]++;
    CHECK(fabs((double)c[7] / N - 0.5) < 5 * sqrt(0.25 / N) && fabs((double)c[1] / N - 0.05) < 5 * sqrt(0.05 * 0.95 / N), "plain softmax");
    memset(c, 0, sizeof c);                               /* temperature 0.5: p ~ p^2, 0.25 .09 .0225 .0025 */
    for (int t = 0; t < N; t++) c[k1_sample(lg, V, 0.5f, 1.0f, 0, &rng, cand)]++;
    double z = 0.25 + 0.09 + 0.0225 + 0.0025;
    CHECK(fabs((double)c[7] / N - 0.25 / z) < 5 * sqrt(0.25 / N), "temperature 0.5");

    /* the selection path equals the full-sort reference on random rows with ties */
    int mismatches = 0, draws = 0;
    for (int row = 0; row < 300; row++) {
        for (int i = 0; i < V; i++) lg[i] = (float)((int)(rnd() * 24)) * 0.5f - 3.0f;      /* 24 levels: heavy ties */
        if (row % 3 == 0) for (int i = 0; i < V; i++) lg[i] += (float)(rnd() * 0.01);
        int ks[] = {0, 1, 2, 5, 128, 511, 512, 9999};
        float ts[] = {0.7f, 1.0f, 2.0f}, ps[] = {0.3f, 0.9f, 0.97f, 1.0f};
        for (int a = 0; a < 8; a++) for (int b = 0; b < 3; b++) for (int d = 0; d < 4; d++) {
            uint64_t s1 = 1000 + (uint64_t)row * 7 + a, s2 = s1;
            int x = k1_sample(lg, V, ts[b], ps[d], ks[a], &s1, cand);
            int y = ref_sample(lg, V, ts[b], ps[d], ks[a], &s2);
            draws++; if (x != y) { mismatches++; if (mismatches < 5) printf("mismatch row %d k %d T %g p %g: %d vs %d\n", row, ks[a], ts[b], ps[d], x, y); }
        }
    }
    CHECK(mismatches == 0, "%d of %d draws differ from the reference", mismatches, draws);

    /* the stream */
    uint64_t a = 99, b = 99, d = 100;
    for (int i = 0; i < V; i++) lg[i] = (float)(i % 13) * 0.3f;
    int same = 1, diff = 0;
    for (int t = 0; t < 64; t++) {
        int x = k1_sample(lg, V, 2.0f, 1.0f, 0, &a, cand), y = k1_sample(lg, V, 2.0f, 1.0f, 0, &b, cand), w = k1_sample(lg, V, 2.0f, 1.0f, 0, &d, cand);
        if (x != y) same = 0;
        if (x != w) diff = 1;
    }
    CHECK(same, "one seed, two streams");
    CHECK(diff, "two seeds, one stream");
    double u = 0; for (int t = 0; t < 100000; t++) { double v = k1_unit(&a); if (v < 0 || v >= 1) failures++; u += v; }
    CHECK(fabs(u / 100000 - 0.5) < 0.01, "uniform mean");

    free(cand);
    if (failures) { printf("serve sampler: %d FAILURES\n", failures); return 1; }
    printf("serve sampler: ok (%d reference draws)\n", draws);
    return 0;
}
