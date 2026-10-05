/* The serve sampler (qwen36_sample.h) against a reference that sorts the whole vocabulary, as the sampler did
 * before: for the same logits and the same rand() values the same tokens, over many logit shapes, temperatures
 * and top_p values at Qwen3.6's vocabulary size; argmax at temperature 0; equal probabilities ordered by token id.
 * The reference sorts once per logits and temperature. Every case draws after its own srand(), with seeds spread
 * over the range of rand() so that the draws land anywhere in the nucleus, then four draws in a row from one
 * srand(), as the engine draws the tokens of a reply. */
#include "../qwen36_sample.h"

static int ref_desc(const void *a, const void *b) {
    const SampleProb *x = a, *y = b;
    if (x->p != y->p) return (y->p > x->p) - (x->p > y->p);
    return (x->id > y->id) - (x->id < y->id);
}
static double ref_sort(SampleProb *rank, const float *lo, int V, float temp) {
    float mx = lo[0];
    for (int i = 1; i < V; i++) if (lo[i] > mx) mx = lo[i];
    double sum = 0;
    for (int i = 0; i < V; i++) { float p = expf((lo[i] - mx) / temp); sum += p; rank[i] = (SampleProb){p, i}; }
    qsort(rank, (size_t)V, sizeof(SampleProb), ref_desc);
    return sum;
}
static int ref_draw(const SampleProb *rank, double sum, int V, float top_p) {
    double cut = (top_p > 0.f && top_p < 1.f) ? top_p * sum : sum, kept = 0;
    int n = 0;
    while (n < V && kept < cut) kept += rank[n++].p;
    double r = ((double)rand() / RAND_MAX) * kept, acc = 0;
    int pick = rank[0].id;
    for (int i = 0; i < n; i++) { acc += rank[i].p; if (acc >= r) { pick = rank[i].id; break; } }
    return pick;
}

int main(void) {
    const int V = 248320;
    float *lo = malloc((size_t)V * sizeof(float));
    SampleProb *rank = malloc((size_t)V * sizeof(SampleProb));
    const float temps[] = {0.2f, 0.7f, 1.0f, 2.0f}, tops[] = {0.3f, 0.9f, 0.99f, 1.0f};
    int failures = 0, cases = 0;
    for (int trial = 0; trial < 13; trial++) {
        srand(500 + trial);
        for (int i = 0; i < V; i++) {               /* long tail plus a few peaks; trial 11 quantized: many ties */
            double u = (rand() + 0.5) / ((double)RAND_MAX + 1);
            lo[i] = (float)(-2.0 * log(-log(u)));
            if (trial == 11) lo[i] = floorf(lo[i] * 2.0f) * 0.5f;
        }
        if (trial == 12)                           /* the nucleus all ties: 50 tokens of exactly the same logit */
            for (int i = 0; i < 50; i++) lo[(i * 4967 + 13) % V] = 30.0f;
        else
            for (int i = 0; i < 25; i++) lo[rand() % V] += (float)((2 + trial) * (1.0 - i / 30.0) * 2.5);
        for (int a = 0; a < 4; a++) {
            double sum = ref_sort(rank, lo, V, temps[a]);
            for (int b = 0; b < 4; b++) {
                unsigned base = 9000u + (unsigned)(trial * 200 + a * 40 + b * 8);
                /* top_p 1 takes the full sort, the reference's own algorithm: one draw is enough there */
                int n = tops[b] < 1.f ? 6 : 1, want[10], got[10];
                for (int d = 0; d < n; d++) {
                    srand((base + d) * 2654435761u); want[d] = ref_draw(rank, sum, V, tops[b]);
                    srand((base + d) * 2654435761u); got[d] = serve_sample(lo, V, temps[a], tops[b]);
                }
                if (tops[b] < 1.f) {                /* four draws in a row from one srand(), as in a reply */
                    srand((base + 7) * 2654435761u); for (int k = 0; k < 4; k++) want[n + k] = ref_draw(rank, sum, V, tops[b]);
                    srand((base + 7) * 2654435761u); for (int k = 0; k < 4; k++) got[n + k] = serve_sample(lo, V, temps[a], tops[b]);
                    n += 4;
                }
                for (int d = 0; d < n; d++) {
                    cases++;
                    if (want[d] != got[d]) {
                        failures++;
                        if (failures <= 5) printf("FAIL trial %d T %.1f top_p %.2f draw %d: %d instead of %d\n", trial, temps[a], tops[b], d, got[d], want[d]);
                    }
                }
            }
        }
    }
    srand(1);
    int arg = 0;
    for (int i = 1; i < V; i++) if (lo[i] > lo[arg]) arg = i;
    if (serve_sample(lo, V, 0.f, 0.9f) != arg) { failures++; printf("FAIL temperature 0 is not the argmax\n"); }
    printf("serve sampler: %d of %d draws as the full sort%s\n", cases - (failures > 0 ? failures : 0), cases,
           failures ? "" : ", argmax at temperature 0");
    printf("%s\n", failures ? "FAILED" : "PASSED");
    free(rank);
    free(lo);
    return failures != 0;
}
