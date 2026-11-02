/* sample_bench.c: how long the qwen36 serve sampler takes for one token, on logits of the real model
 * (Qwen3.6-35B-A3B: 248,320 float32 per vector). It includes qwen36.c, as the engine's own tests do, and calls
 * the serve_sample of that qwen36.c.
 *
 *   clang -O3 -Xclang -fopenmp -I<libomp include> -I<engine c/ dir> sample_bench.c -o sample_bench \
 *         -lm -L<libomp lib> -lomp
 *   ./sample_bench logits-it.f32 logits-en.f32 logits-code.f32 -- reply-it.f32 reply-en.f32 reply-code.f32
 *
 * Before "--": files of one vector (the last token of a prompt). Each at five settings, 100 calls per setting;
 * the median time of a call. After "--": files of many vectors (every token of a reply, sample_gen.c), at the
 * gateway's default (temperature 0.7, top_p 0.9): 5 calls per vector, and over the vectors the mean, median,
 * 90th percentile and largest of each vector's median, and over all their vectors the draws per second. Before
 * every call srand() with a seed spread over the range of rand(), so the draws land anywhere in the nucleus. At
 * the end a checksum of every drawn token: equal for two versions that draw the same tokens. PICKS=<file> also
 * writes every drawn token (int32, in the order of the calls), to find which draws two versions do not share. */
#define main qwen36_engine_main
#include "qwen36.c"
#undef main

static int bench_cmp(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv) {
    enum { V = 248320, CALLS = 100, GEN_CALLS = 5, MAX_GEN = 4096 };
    static const float temps[] = {0.7f, 0.6f, 1.0f, 1.5f, 2.0f}, tops[] = {0.9f, 0.95f, 1.0f, 0.95f, 0.9f};
    if (argc < 2) { fprintf(stderr, "usage: %s <one-vector.f32>... [-- <reply.f32>...]\n", argv[0]); return 2; }
    float *lo = malloc((size_t)V * sizeof(float));
    double *per = malloc(MAX_GEN * sizeof(double));
    if (!lo || !per) return 1;
    uint64_t h = 0xcbf29ce484222325ull;
    const char *pk = getenv("PICKS");
    FILE *picks = pk && *pk ? fopen(pk, "wb") : NULL;
    if (pk && *pk && !picks) { perror(pk); return 1; }
    unsigned call = 0;
    int gen = 0, gen_tokens = 0;
    double gen_s = 0;
    for (int f = 1; f < argc; f++) {
        if (!strcmp(argv[f], "--")) { gen = 1; continue; }
        FILE *in = fopen(argv[f], "rb");
        if (!in) { perror(argv[f]); return 1; }
        if (!gen) {
            if (fread(lo, sizeof(float), V, in) != V) { fprintf(stderr, "%s: not %d floats\n", argv[f], V); return 1; }
            fclose(in);
            for (int s = 0; s < 5; s++) {
                double t[CALLS];
                for (int k = 0; k < CALLS; k++) {
                    srand((1000u + call++) * 2654435761u);
                    double a = now_s();
                    int pick = serve_sample(lo, V, temps[s], tops[s]);
                    t[k] = now_s() - a;
                    for (int b = 0; b < 4; b++) h = (h ^ (uint8_t)(pick >> (8 * b))) * 0x100000001b3ull;
                    if (picks) fwrite(&pick, sizeof pick, 1, picks);
                }
                qsort(t, CALLS, sizeof t[0], bench_cmp);
                printf("%s temperature %.2f top_p %.2f: median %.3f ms a call (min %.3f, max %.3f), %d calls\n",
                       argv[f], temps[s], tops[s], t[CALLS / 2] * 1e3, t[0] * 1e3, t[CALLS - 1] * 1e3, CALLS);
            }
            continue;
        }
        int nv = 0;
        double sum = 0;
        while (nv < MAX_GEN && fread(lo, sizeof(float), V, in) == V) {
            double t[GEN_CALLS];
            for (int k = 0; k < GEN_CALLS; k++) {
                srand((1000u + call++) * 2654435761u);
                double a = now_s();
                int pick = serve_sample(lo, V, 0.7f, 0.9f);
                t[k] = now_s() - a;
                for (int b = 0; b < 4; b++) h = (h ^ (uint8_t)(pick >> (8 * b))) * 0x100000001b3ull;
                if (picks) fwrite(&pick, sizeof pick, 1, picks);
            }
            qsort(t, GEN_CALLS, sizeof t[0], bench_cmp);
            per[nv++] = t[GEN_CALLS / 2];
            sum += t[GEN_CALLS / 2];
        }
        fclose(in);
        if (!nv) { fprintf(stderr, "%s: no vector\n", argv[f]); return 1; }
        gen_tokens += nv;
        gen_s += sum;
        qsort(per, nv, sizeof per[0], bench_cmp);
        printf("%s temperature 0.70 top_p 0.90: %d tokens, a token mean %.3f ms, median %.3f, 90th percentile %.3f, "
               "largest %.3f (each the median of %d calls)\n",
               argv[f], nv, sum / nv * 1e3, per[nv / 2] * 1e3, per[nv * 9 / 10] * 1e3, per[nv - 1] * 1e3, GEN_CALLS);
    }
    if (gen_tokens)
        printf("every reply token at the gateway default: %d tokens in %.3f s = %.2f draws per second\n", gen_tokens,
               gen_s, gen_tokens / gen_s);
    printf("checksum of every drawn token: %016llx\n", (unsigned long long)h);
    if (picks && fclose(picks)) { perror(pk); return 1; }
    free(per);
    free(lo);
    return 0;
}
