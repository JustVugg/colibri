/* sample_gen.c: the logits of every token of a chat reply of Qwen3.6-35B-A3B, generated the way the gateway asks
 * by default (temperature 0.7, top_p 0.9), so that the sampler can then be timed on the logits of a real reply.
 * It includes qwen36.c, as the engine's own tests do, and uses its tokenizer, forward pass and serve_sample.
 *
 *   clang -O3 -Xclang -fopenmp -I<libomp include> -I<engine c/ dir> sample_gen.c -o sample_gen \
 *         -lm -L<libomp lib> -lomp
 *   SNAP=<model> ./sample_gen <prompt-file> <tokens> <seed> <out.f32> > reply.txt
 *
 * The prompt file holds the text the gateway sends the engine for one user message (Qwen3.6's chat template,
 * thinking off: its default). The prompt is read from an empty state; then srand(seed) once and, as serve_one
 * does, serve_sample picks every token from the logits of the previous step, until <|im_end|> or <tokens>
 * tokens. Every logit vector the sampler drew from (248,320 float32) is appended to <out.f32>, in order; the
 * reply goes to stdout. */
#define main qwen36_engine_main
#include "qwen36.c"
#undef main

int main(int argc, char **argv) {
    const char *snap = getenv("SNAP");
    if (!snap || argc != 5) {
        fprintf(stderr, "usage: SNAP=<model> %s <prompt-file> <tokens> <seed> <out.f32>\n", argv[0]);
        return 2;
    }
    int G = atoi(argv[2]);
    unsigned seed = (unsigned)strtoul(argv[3], NULL, 10);
    char tok[2048];
    snprintf(tok, sizeof tok, "%s/tokenizer.json", snap);
    load_tokenizer(tok);
    if (!g_tok) { fprintf(stderr, "no tokenizer: %s\n", tok); return 1; }
    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror(argv[1]); return 1; }
    char text[65536];
    size_t got = fread(text, 1, sizeof text - 1, f);
    fclose(f);
    text[got] = 0;
    int *ids = NULL, np = 0;
    encode_text(text, &ids, &np);
    if (G < 1 || np < 1 || np + G > qwen36_max_ctx()) { fprintf(stderr, "bad lengths: prompt %d, tokens %d\n", np, G); return 2; }

    static Model m;
    model_init(&m, snap, 16, 4);
    g_expert_gs = m.c.expert_gs;
    int V = m.c.vocab, eos[4], neos = serve_eos_ids(eos, 4);
    m.max_t = np + G;                          /* an empty state, as the bench harness starts each length */
    reset_recurrent(&m);
    ensure_kv(&m);
    m.kv_len = 0;
    m.first_step = 1;
    if (m.seen) memset(m.seen, 0, (size_t)m.c.n_layers * m.c.n_experts);
    if (m.momentum_logits) memset(m.momentum_logits, 0, (size_t)m.c.n_layers * m.c.n_experts * sizeof(float));

    FILE *out = fopen(argv[4], "wb");
    if (!out) { perror(argv[4]); return 1; }
    double t0 = now_s();
    float *lo = step(&m, ids, np, 0);
    srand(seed);
    int n = 0, ended = 0;
    for (int g = 0; g < G; g++) {
        if (fwrite(lo, sizeof(float), (size_t)V, out) != (size_t)V) { fprintf(stderr, "cannot write %s\n", argv[4]); return 1; }
        int tk = serve_sample(lo, V, 0.7f, 0.9f);
        free(lo);
        n++;
        for (int e = 0; e < neos; e++) ended |= tk == eos[e];
        if (ended) break;
        unsigned char piece[256];
        int pn = 0;
        decode_id_to_bytes(tk, piece, &pn);
        fwrite(piece, 1, (size_t)pn, stdout);
        fflush(stdout);
        if (g + 1 < G) lo = step(&m, &tk, 1, np + g);
    }
    if (fclose(out)) { fprintf(stderr, "cannot write %s\n", argv[4]); return 1; }
    printf("\n");
    fprintf(stderr, "[sample_gen] prompt %d tokens; %d logit vectors written to %s (%s); %.1f s\n", np, n, argv[4],
            ended ? "the reply ended" : "token limit", now_s() - t0);
    free(ids);
    return 0;
}
