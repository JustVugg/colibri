/* v4_topk_desc against the indexer's qsort ranking (score descending, index
 * ascending): same indices in the same order on random rows, rows full of
 * ties, signed zeros, infinities and subnormals; NaN rows refused. Prints the
 * time of both at the decode and prefill sizes.
 *
 * Built and run by `make -C c check`; alone, from the repository root:
 *   make -C c tests/test_v4_topk && c/tests/test_v4_topk     (add .exe on Windows) */
#include "../v4_topk.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct { float score; int index; } IndexScore;

static int descending_score(const void *left, const void *right) {
    const IndexScore *a = left, *b = right;
    if (a->score < b->score) return 1;
    if (a->score > b->score) return -1;
    return a->index - b->index;
}

static int reference(int *out, const float *scores, int count, int k,
                     IndexScore *ranked) {
    for (int i = 0; i < count; i++) ranked[i] = (IndexScore){scores[i], i};
    qsort(ranked, (size_t)count, sizeof(*ranked), descending_score);
    int n = count < k ? count : k;
    for (int i = 0; i < n; i++) out[i] = ranked[i].index;
    return n;
}

static uint64_t rng = 0x9e3779b97f4a7c15ull;
static uint32_t next(void) {
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return (uint32_t)(rng >> 16);
}

static float bits_float(uint32_t bits) {
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

/* kind 0: spread values; 1: few distinct values (ties everywhere);
 * 2: signed zeros, infinities, subnormals, extremes; 3: random bit patterns
 * without NaN; 4: all equal. */
static void fill(float *scores, int count, int kind) {
    static const float specials[] = {0.0f, -0.0f, INFINITY, -INFINITY,
                                     1e-45f, -1e-45f, 1.17549435e-38f,
                                     -1.17549435e-38f, 3.4028235e38f,
                                     -3.4028235e38f, 1.0f, -1.0f};
    for (int i = 0; i < count; i++) {
        switch (kind) {
        case 0: scores[i] = ((float)next() / 4294967296.0f - 0.25f) * 37.0f; break;
        case 1: scores[i] = (float)(next() % 7) * 0.5f - 1.0f; break;
        case 2: scores[i] = specials[next() % (sizeof(specials) / sizeof(specials[0]))]; break;
        case 3: {
            uint32_t bits;
            do bits = next() << 16 ^ next(); while ((bits & 0x7fffffffu) > 0x7f800000u);
            scores[i] = bits_float(bits);
            break;
        }
        default: scores[i] = 0.125f; break;
        }
    }
}

static double now_ms(void) {
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return value.tv_sec * 1e3 + value.tv_nsec * 1e-6;
}

int main(void) {
    enum { MAX = 20000 };
    float *scores = malloc(MAX * sizeof(*scores));
    int *got = malloc(MAX * sizeof(*got)), *want = malloc(MAX * sizeof(*want));
    IndexScore *ranked = malloc(MAX * sizeof(*ranked));
    uint64_t *work = malloc(2 * (size_t)MAX * sizeof(*work));
    if (!scores || !got || !want || !ranked || !work) return 1;

    static const int counts[] = {1, 2, 3, 7, 255, 256, 511, 512, 513, 600,
                                 1064, 2048, 4096, 16384};
    static const int ks[] = {1, 64, 512, 2048, MAX};
    long long rows = 0;
    for (size_t c = 0; c < sizeof(counts) / sizeof(counts[0]); c++)
        for (size_t q = 0; q < sizeof(ks) / sizeof(ks[0]); q++)
            for (int kind = 0; kind < 5; kind++)
                for (int trial = 0; trial < 4; trial++) {
                    int count = counts[c], k = ks[q];
                    fill(scores, count, kind);
                    int n = v4_topk_desc(got, scores, count, k, work);
                    int m = reference(want, scores, count, k, ranked);
                    if (n != m || memcmp(got, want, (size_t)n * sizeof(*got))) {
                        fprintf(stderr, "top-k mismatch count=%d k=%d kind=%d trial=%d n=%d m=%d\n",
                                count, k, kind, trial, n, m);
                        return 2;
                    }
                    rows++;
                }

    /* A NaN anywhere: refused, output untouched. */
    fill(scores, 1064, 0);
    scores[777] = bits_float(0x7fc00000u);
    memset(got, 0xff, 1064 * sizeof(*got));
    if (v4_topk_desc(got, scores, 1064, 512, work) != -1 || got[0] != -1) {
        fprintf(stderr, "NaN row not refused\n");
        return 3;
    }
    scores[777] = bits_float(0xffc00001u);
    if (v4_topk_desc(got, scores, 1064, 512, work) != -1) {
        fprintf(stderr, "negative NaN row not refused\n");
        return 3;
    }
    if (v4_topk_desc(got, scores, 0, 512, work) != 0 ||
        v4_topk_desc(got, scores, 10, 0, work) != 0) {
        fprintf(stderr, "empty selection not empty\n");
        return 4;
    }
    printf("v4_topk_desc == qsort ranking on %lld rows, NaN refused: OK\n", rows);

    static const int timed[] = {1064, 4096, 16384};
    for (size_t c = 0; c < sizeof(timed) / sizeof(timed[0]); c++) {
        int count = timed[c], repeats = 4000000 / count;
        fill(scores, count, 0);
        double t0 = now_ms();
        for (int r = 0; r < repeats; r++) reference(want, scores, count, 512, ranked);
        double t1 = now_ms();
        for (int r = 0; r < repeats; r++) v4_topk_desc(got, scores, count, 512, work);
        double t2 = now_ms();
        printf("count=%d k=512: qsort %.1f us, v4_topk_desc %.1f us per row\n",
               count, (t1 - t0) * 1e3 / repeats, (t2 - t1) * 1e3 / repeats);
    }
    free(work); free(ranked); free(want); free(got); free(scores);
    return 0;
}
