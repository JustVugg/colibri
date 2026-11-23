/* Exact bounded nucleus selection. The caller retains softmax and RNG.
 * A fast prefix is accepted only if its entire selected order and mass are
 * identical to the original full qsort. Ties and uncertain boundaries use
 * the original sort, with its original index input and tie behavior.
 * No probability approximation, activation quantization or new RNG draw. */
#ifndef GLM53_EXACT_NUCLEUS_H
#define GLM53_EXACT_NUCLEUS_H
#include <math.h>
#include <stdlib.h>
#ifndef G53_NUCLEUS_HEAP
#define G53_NUCLEUS_HEAP 1024
#endif
#if G53_NUCLEUS_HEAP < 2 || G53_NUCLEUS_HEAP > 65536
#error G53_NUCLEUS_HEAP must be between 2 and 65536
#endif
static inline void g53_nucleus_sift(int *heap, int n, int pos, const float *p) {
    const int value = heap[pos];
    for (;;) {
        int child = 2 * pos + 1;
        if (child >= n) break;
        if (child + 1 < n && p[heap[child + 1]] < p[heap[child]]) child++;
        if (p[value] <= p[heap[child]]) break;
        heap[pos] = heap[child]; pos = child;
    }
    heap[pos] = value;
}
/* On decline only order scratch may change; keep, mass and p do not.
 * Early screening avoids heap work for tied maxima and diffuse rows. This
 * chooses the old path, not a relaxed numerical or probability condition. */
static int g53_nucleus_prefix(const float *p, int vocab, float top_p, int *order,
                              int *keep, double *mass,
                              int (*compare)(const void *, const void *)) {
    const int k = G53_NUCLEUS_HEAP;
    if (vocab <= k || !(top_p > 0.f && top_p < 1.f) || !isfinite(top_p)) return 0;
    float peak = -1.f; int peak_count = 0;
    for (int i = 0; i < vocab; i++) {
        if (!isfinite(p[i]) || p[i] < 0.f) return 0;
        if (p[i] > peak) { peak = p[i]; peak_count = 1; }
        else if (p[i] == peak) peak_count++;
    }
    if (peak_count != 1 || (double)peak * (k - 1) < (double)top_p) return 0;
    for (int i = 0; i < k; i++) order[i] = i;
    for (int i = k / 2; i > 0; i--) g53_nucleus_sift(order, k, i - 1, p);
    for (int i = k; i < vocab; i++) {
        if (p[i] > p[order[0]]) {
            order[0] = i;
            g53_nucleus_sift(order, k, 0, p);
        }
    }
    qsort(order, (size_t)k, sizeof(*order), compare);
    double sum = 0.0;
    for (int i = 0; i < k - 1; i++) {
        if (i && p[order[i]] == p[order[i - 1]]) return 0;
        sum += p[order[i]];
        if (sum >= top_p) {
            if (!(p[order[i]] > p[order[i + 1]])) return 0;
            *keep = i + 1; *mass = sum; return 1;
        }
    }
    return 0;
}
static int g53_nucleus_order(const float *p, int vocab, float top_p, int *order,
                             int *keep, double *mass,
                             int (*compare)(const void *, const void *)) {
    if (g53_nucleus_prefix(p, vocab, top_p, order, keep, mass, compare)) return 1;
    /* Reset the input even if prefix selection changed only some indices. */
    for (int i = 0; i < vocab; i++) order[i] = i;
    qsort(order, (size_t)vocab, sizeof(*order), compare);
    double sum = 0.0; int n = vocab;
    for (int i = 0; i < vocab; i++) {
        sum += p[order[i]];
        if (sum >= top_p) { n = i + 1; break; }
    }
    *keep = n; *mass = sum; return 0;
}
#endif
