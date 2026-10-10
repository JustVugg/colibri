/* picks.c: the draws that two builds of sample_bench.c do not share (their PICKS= files, same files and order),
 * and for each such draw whether the two tokens have exactly the same probability: the one case where this change
 * may pick another token (ties are now in token id order, which the full sort left to qsort). It also counts the
 * draws of the first file that are not the most likely token, which a sampler returning the argmax would miss.
 *
 *   clang -O3 picks.c -o picks -lm
 *   ./picks dev.picks pr.picks logits-it.f32 logits-en.f32 logits-code.f32 -- reply-it.f32 reply-en.f32 reply-code.f32 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { V = 248320, CALLS = 100, GEN_CALLS = 5 };
static FILE *pa, *pb;
static long diff_all, ties_all, other_all;

/* n calls on one vector at one setting: how many picks of the first file are not the most likely token, how many
 * picks differ, and how many of those are exact ties */
static void compare(const float *lo, float temp, int n, long *other, long *diff, long *ties) {
    float mx = lo[0];
    int top = 0;
    for (int i = 1; i < V; i++) if (lo[i] > mx) { mx = lo[i]; top = i; }
    for (int k = 0; k < n; k++) {
        int32_t a, b;
        if (fread(&a, 4, 1, pa) != 1 || fread(&b, 4, 1, pb) != 1) { fprintf(stderr, "a picks file is too short\n"); exit(1); }
        *other += a != top;
        if (a == b) continue;
        (*diff)++;
        if (a < 0 || a >= V || b < 0 || b >= V) continue;
        if (expf((lo[a] - mx) / temp) == expf((lo[b] - mx) / temp)) (*ties)++;
    }
}

int main(int argc, char **argv) {
    static const float temps[] = {0.7f, 0.6f, 1.0f, 1.5f, 2.0f}, tops[] = {0.9f, 0.95f, 1.0f, 0.95f, 0.9f};
    if (argc < 4) { fprintf(stderr, "usage: %s <a.picks> <b.picks> <one-vector.f32>... [-- <reply.f32>...]\n", argv[0]); return 2; }
    pa = fopen(argv[1], "rb");
    pb = fopen(argv[2], "rb");
    float *lo = malloc((size_t)V * sizeof(float));
    if (!pa || !pb || !lo) { fprintf(stderr, "cannot open the picks files\n"); return 1; }
    int gen = 0;
    long calls = 0;
    for (int f = 3; f < argc; f++) {
        if (!strcmp(argv[f], "--")) { gen = 1; continue; }
        FILE *in = fopen(argv[f], "rb");
        if (!in) { perror(argv[f]); return 1; }
        if (!gen) {
            if (fread(lo, sizeof(float), V, in) != V) { fprintf(stderr, "%s: not %d floats\n", argv[f], V); return 1; }
            for (int s = 0; s < 5; s++) {
                long o = 0, d = 0, t = 0;
                compare(lo, temps[s], CALLS, &o, &d, &t);
                calls += CALLS;
                other_all += o;
                diff_all += d;
                ties_all += t;
                printf("%s temperature %.2f top_p %.2f: %ld of %d draws not the most likely token; %ld differ, %ld of them two "
                       "tokens of exactly the same probability\n", argv[f], temps[s], tops[s], o, CALLS, d, t);
            }
        } else {
            long o = 0, d = 0, t = 0;
            int nv = 0;
            while (fread(lo, sizeof(float), V, in) == V) { compare(lo, 0.7f, GEN_CALLS, &o, &d, &t); nv++; }
            calls += (long)nv * GEN_CALLS;
            other_all += o;
            diff_all += d;
            ties_all += t;
            printf("%s temperature 0.70 top_p 0.90: %ld of %d draws not the most likely token; %ld differ, %ld of them two "
                   "tokens of exactly the same probability\n", argv[f], o, nv * GEN_CALLS, d, t);
        }
        fclose(in);
    }
    int32_t x;
    if (fread(&x, 4, 1, pa) == 1 || fread(&x, 4, 1, pb) == 1) { fprintf(stderr, "a picks file is too long\n"); return 1; }
    printf("in all: %ld of %ld draws not the most likely token; %ld differ, %ld of them two tokens of exactly the same "
           "probability\n", other_all, calls, diff_all, ties_all);
    return diff_all != ties_all;
}
