/* nucleus.c: how many tokens the serve sampler sorts on the logits of sample_bench.c. For every vector: the
 * nucleus (the tokens the draw can pick, as the full sort finds them) and the candidates this change sorts (the
 * tokens of the exponent buckets down to the one that passes top_p, plus one, as in qwen36_sample.h).
 *
 *   clang -O3 -I<engine c/ dir> nucleus.c -o nucleus -lm
 *   ./nucleus logits-it.f32 logits-en.f32 logits-code.f32 -- reply-it.f32 reply-en.f32 reply-code.f32 */
#include "qwen36_sample.h"

static void count(const float *lo, int V, float temp, float top_p, SampleProb *rank, double *top, int *n_out, int *m_out) {
    float mx = lo[0];
    for (int i = 1; i < V; i++) if (lo[i] > mx) mx = lo[i];
    double sum = 0, bucket[256] = {0};
    for (int i = 0; i < V; i++) {
        float p = expf((lo[i] - mx) / temp);
        uint32_t bits;
        memcpy(&bits, &p, 4);
        sum += p;
        rank[i] = (SampleProb){p, i};
        bucket[255 - ((bits >> 23) & 255)] += p;
    }
    double cut = top_p * sum, kept = 0, total = 0;
    int last = 0;
    while (last < 256) { total += bucket[last]; if (total >= cut) break; last++; }
    if (last < 255) last++;
    int m = 0;
    for (int i = 0; i < V; i++) {
        uint32_t bits;
        memcpy(&bits, &rank[i].p, 4);
        if (255 - (int)((bits >> 23) & 255) <= last) m++;
    }
    qsort(rank, (size_t)V, sizeof(SampleProb), sample_prob_desc);
    int n = 0;
    while (n < V && kept < cut) kept += rank[n++].p;
    *top = rank[0].p / sum;
    *n_out = n;
    *m_out = m;
}
static int icmp(const void *a, const void *b) { return (*(const int *)a > *(const int *)b) - (*(const int *)a < *(const int *)b); }

int main(int argc, char **argv) {
    enum { V = 248320, MAX_GEN = 4096 };
    static const float temps[] = {0.7f, 0.6f, 1.0f, 1.5f, 2.0f}, tops[] = {0.9f, 0.95f, 1.0f, 0.95f, 0.9f};
    float *lo = malloc((size_t)V * sizeof(float));
    SampleProb *rank = malloc((size_t)V * sizeof(SampleProb));
    int *ns = malloc(MAX_GEN * sizeof(int)), *ms = malloc(MAX_GEN * sizeof(int)), gen = 0;
    if (!lo || !rank || !ns || !ms) return 1;
    for (int f = 1; f < argc; f++) {
        if (!strcmp(argv[f], "--")) { gen = 1; continue; }
        FILE *in = fopen(argv[f], "rb");
        if (!in) { perror(argv[f]); return 1; }
        if (!gen) {
            if (fread(lo, sizeof(float), V, in) != V) { fprintf(stderr, "%s: not %d floats\n", argv[f], V); return 1; }
            fclose(in);
            for (int s = 0; s < 5; s++) {
                double top;
                int n, m;
                count(lo, V, temps[s], tops[s] < 1.f ? tops[s] : 1.f, rank, &top, &n, &m);
                printf("%s temperature %.2f top_p %.2f: top token %.4f%%, nucleus %d tokens, candidates %s%d\n", argv[f],
                       temps[s], tops[s], top * 100, n, tops[s] < 1.f ? "" : "(the full sort) ", tops[s] < 1.f ? m : V);
            }
            continue;
        }
        int nv = 0, one = 0, upto4 = 0, upto16 = 0, upto64 = 0;
        while (nv < MAX_GEN && fread(lo, sizeof(float), V, in) == V) {
            double top;
            count(lo, V, 0.7f, 0.9f, rank, &top, &ns[nv], &ms[nv]);
            one += ns[nv] == 1;
            upto4 += ns[nv] <= 4;
            upto16 += ns[nv] <= 16;
            upto64 += ns[nv] <= 64;
            nv++;
        }
        fclose(in);
        if (!nv) { fprintf(stderr, "%s: no vector\n", argv[f]); return 1; }
        qsort(ns, nv, sizeof(int), icmp);
        qsort(ms, nv, sizeof(int), icmp);
        printf("%s temperature 0.70 top_p 0.90: %d tokens; nucleus of 1 token %d, up to 4 %d, up to 16 %d, up to 64 %d, "
               "largest %d; candidates median %d, 90th percentile %d, largest %d\n",
               argv[f], nv, one, upto4, upto16, upto64, ns[nv - 1], ms[nv / 2], ms[nv * 9 / 10], ms[nv - 1]);
    }
    free(ms);
    free(ns);
    free(rank);
    free(lo);
    return 0;
}
