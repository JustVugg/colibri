/* qwen36-bench (local tool, not part of the repository): how fast the Qwen3.6
 * engine reads a prompt and generates, measured the way ds4-bench does. The
 * engine is compiled in, as tools/qwen36_canonical.c does, and used as is.
 *
 *   SNAP=<model-dir> qwen36-bench <text-file> [lengths=64,512] [gen=32]
 *
 * For each length N the first N tokens of the text are read (timed), then
 * `gen` tokens are generated, always the most likely one and never an
 * end-of-text token (timed, the first step apart). Lengths run in one
 * process, in order, so the expert cache carries over as in ds4-bench. One
 * CSV row per length on stdout.
 *
 * By default every length starts from an empty state: the prefill of a new
 * prompt of N tokens. PHOTO=1 does what ds4-bench does: after each prefill the
 * sequence state is copied out of the GPU (the "photo"), the generation runs,
 * the photo is put back and the next length only reads the tokens it adds:
 * the prefill of a growing conversation. Photo time is not timed. The engine
 * gives the same numbers however a prompt is cut into pieces, so with PHOTO=1
 * the fingerprint of each length must equal the one without it.
 *
 * fingerprint: FNV-1a over every logit of the run and the chosen tokens. An
 * optimization that must not change results leaves it unchanged.
 * PHOTO_SAVE=<file> also writes the photo taken after the last prefill to a
 * file, with the logits of that prompt; PHOTO_LOAD=<file> starts from it: the
 * first length must be the photo's, its prompt is not read again (prefill
 * columns 0) and its fingerprint equals the one of the run that saved it.
 * So a long context is read once, and every run measures at that length.
 * TEACHER=1 feeds the text's own next tokens instead of the most likely ones
 * (ds4-bench's teacher-forced decode) and reports their mean negative log
 * likelihood (nll, nats per token) and how often the most likely token was the
 * text's (top1): the quality check of kernels that are not bit-identical.
 * tokens is the fingerprint of the chosen tokens alone: equal when the same
 * words come out even though the logits differ in the last bits.
 * BENCH_DUMP=<file> writes the last logits of the first length, to compare
 * with the engine's DUMP. With COLI_TIMERS=1 the engine's phase report is
 * printed at the end. */
#define main qwen36_engine_main
#include "qwen36.c"
#undef main

static uint64_t bench_fnv(uint64_t h, const void *data, size_t n) {
    const unsigned char *p = (const unsigned char *)data;
    for (size_t i = 0; i < n; i++) h = (h ^ p[i]) * 0x100000001b3ull;
    return h;
}

#ifdef QWEN36_METAL_GRAPH
/* The state of a sequence in the graph: DeltaNet state (both planes) and
 * convolution history, the KV rows already written. With buf NULL only the
 * bytes are counted. After step() returns the GPU has finished everything. */
#define BENCH_PART(gpu, n) do {                                         \
        size_t n_ = (n);                                                \
        if (buf && save) memcpy(buf + at, (gpu), n_);                   \
        else if (buf) memcpy((gpu), buf + at, n_);                      \
        at += n_;                                                       \
    } while (0)
static size_t bench_state(Model *m, unsigned char *buf, int kv_len, int save) {
    Qwen36Graph *g = m->graph;
    Cfg *c = &m->c;
    size_t row = (size_t)c->head_dim * sizeof(float), at = 0;
    for (int i = 0; i < c->n_layers; i++) {
        QgLayer *L = &g->L[i];
        int half = c->is_attn[i] &&
                   q36_tensor_bytes(L->kc) < (size_t)c->kv_heads * g->cap * c->head_dim * sizeof(float);
        if (c->is_attn[i] && !half) {
            for (int h = 0; h < c->kv_heads; h++) {   /* [kv][cap][head_dim] */
                BENCH_PART((char *)q36_tensor_data(L->kc) + (size_t)h * g->cap * row, (size_t)kv_len * row);
                BENCH_PART((char *)q36_tensor_data(L->vc) + (size_t)h * g->cap * row, (size_t)kv_len * row);
            }
        } else if (c->is_attn[i]) {
            /* Fast mode keeps the KV cache in halves; the photo keeps floats. */
            size_t n = (size_t)kv_len * c->head_dim;
            for (int h = 0; h < c->kv_heads; h++) {
                _Float16 *cache[2] = {(_Float16 *)q36_tensor_data(L->kc) + (size_t)h * g->cap * c->head_dim,
                                      (_Float16 *)q36_tensor_data(L->vc) + (size_t)h * g->cap * c->head_dim};
                for (int kvp = 0; kvp < 2; kvp++) {
                    float *photo = buf ? (float *)(buf + at) : NULL;
                    if (photo && save) for (size_t j = 0; j < n; j++) photo[j] = (float)cache[kvp][j];
                    else if (photo) for (size_t j = 0; j < n; j++) cache[kvp][j] = (_Float16)photo[j];
                    at += n * sizeof(float);
                }
            }
        } else {
            /* The photo keeps both planes of the state. In fast mode the engine
             * keeps the high one only: the low plane folds into it on load (the
             * same addition the fast kernel makes) and saves as zeros. */
            size_t plane = (size_t)c->dn_vheads * c->dn_kdim * c->dn_vdim * sizeof(float);
            if (q36_tensor_bytes(L->state) >= 2 * plane) {
                BENCH_PART(q36_tensor_data(L->state), 2 * plane);
            } else {
                float *hi = q36_tensor_data(L->state);
                if (buf && save) { memcpy(buf + at, hi, plane); memset(buf + at + plane, 0, plane); }
                else if (buf) {
                    const float *ph = (const float *)(buf + at), *pl = (const float *)(buf + at + plane);
                    for (size_t i = 0; i < plane / sizeof(float); i++) hi[i] = ph[i] + pl[i];
                }
                at += 2 * plane;
            }
            BENCH_PART(q36_tensor_data(L->ring), q36_tensor_bytes(L->ring));
        }
    }
    return at;
}

typedef struct { unsigned char *buf; size_t bytes; int kv_len; } BenchPhoto;

static void bench_photo_save(Model *m, BenchPhoto *p) {
    q36_gpu_sync();
    size_t need = bench_state(m, NULL, m->kv_len, 1);
    if (need > p->bytes) {
        free(p->buf);
        p->buf = malloc(need);
        p->bytes = need;
        if (!p->buf) { fprintf(stderr, "no memory for the photo (%zu bytes)\n", need); exit(1); }
    }
    bench_state(m, p->buf, m->kv_len, 1);
    p->kv_len = m->kv_len;
}

static void bench_photo_load(Model *m, const BenchPhoto *p) {
    q36_gpu_sync();
    bench_state(m, p->buf, p->kv_len, 0);
    m->kv_len = p->kv_len;
}

/* PHOTO_SAVE / PHOTO_LOAD: the photo on disk with the logits of its last
 * prompt token, so a long context is read once and reloaded by every run.
 * The header ties it to the model's shape and to the tokens it holds. */
typedef struct {
    char magic[8];                              /* "Q36PHOT1" */
    uint32_t kv_len, vocab, layers, hidden;
    uint64_t state_bytes, tokens_hash;
} BenchPhotoHeader;

static uint64_t bench_tokens_hash(const int *ids, int n) {
    return bench_fnv(0xcbf29ce484222325ull, ids, (size_t)n * sizeof(int));
}

static void bench_photo_write(const char *path, Model *m, const BenchPhoto *p, const float *logits, const int *ids) {
    BenchPhotoHeader h = {{'Q', '3', '6', 'P', 'H', 'O', 'T', '1'}, (uint32_t)p->kv_len, (uint32_t)m->c.vocab,
                          (uint32_t)m->c.n_layers, (uint32_t)m->c.hidden, bench_state(m, NULL, p->kv_len, 1),
                          bench_tokens_hash(ids, p->kv_len)};
    FILE *f = fopen(path, "wb");
    if (!f || fwrite(&h, sizeof h, 1, f) != 1 || fwrite(logits, sizeof(float), h.vocab, f) != h.vocab ||
        fwrite(p->buf, 1, h.state_bytes, f) != h.state_bytes || fclose(f)) {
        fprintf(stderr, "cannot write the photo %s\n", path);
        exit(1);
    }
    fprintf(stderr, "[bench] photo of %d tokens written to %s (%.0f MB)\n", p->kv_len, path,
            (double)(sizeof h + (size_t)h.vocab * 4 + h.state_bytes) / 1e6);
}

/* The state goes into the graph and into *p; the logits are returned. */
static float *bench_photo_read(const char *path, Model *m, BenchPhoto *p, const int *ids, int nids) {
    BenchPhotoHeader h;
    FILE *f = fopen(path, "rb");
    if (!f || fread(&h, sizeof h, 1, f) != 1 || memcmp(h.magic, "Q36PHOT1", 8)) {
        fprintf(stderr, "%s: not a photo\n", path);
        exit(1);
    }
    if (h.vocab != (uint32_t)m->c.vocab || h.layers != (uint32_t)m->c.n_layers || h.hidden != (uint32_t)m->c.hidden ||
        (int)h.kv_len > nids || h.tokens_hash != bench_tokens_hash(ids, (int)h.kv_len) ||
        h.state_bytes != bench_state(m, NULL, (int)h.kv_len, 1)) {
        fprintf(stderr, "%s: made from another model or another text\n", path);
        exit(1);
    }
    float *logits = malloc((size_t)h.vocab * sizeof(float));
    free(p->buf);
    p->buf = malloc(h.state_bytes);
    p->bytes = h.state_bytes;
    if (!logits || !p->buf || fread(logits, sizeof(float), h.vocab, f) != h.vocab ||
        fread(p->buf, 1, h.state_bytes, f) != h.state_bytes) {
        fprintf(stderr, "%s: cannot read the photo\n", path);
        exit(1);
    }
    fclose(f);
    p->kv_len = (int)h.kv_len;
    bench_photo_load(m, p);
    return logits;
}
#endif

/* An empty sequence with room for `need` tokens, as serve_one() starts one. */
static void bench_fresh(Model *m, int need) {
    m->max_t = need;
    reset_recurrent(m);
    ensure_kv(m);
    m->kv_len = 0;
    m->first_step = 1;
    if (m->seen) memset(m->seen, 0, (size_t)m->c.n_layers * m->c.n_experts);
    if (m->momentum_logits) memset(m->momentum_logits, 0, (size_t)m->c.n_layers * m->c.n_experts * sizeof(float));
}

int main(int argc, char **argv) {
    const char *snap = getenv("SNAP");
    if (!snap || argc < 2) {
        fprintf(stderr, "usage: SNAP=<model-dir> [PHOTO=1] [PHOTO_SAVE=f] [PHOTO_LOAD=f] %s <text-file> [lengths=64,512] [gen=32]\n", argv[0]);
        return 2;
    }
    const char *photo_save = getenv("PHOTO_SAVE"), *photo_load = getenv("PHOTO_LOAD");
    if (photo_save && !*photo_save) photo_save = NULL;
    if (photo_load && !*photo_load) photo_load = NULL;
    int photo = (getenv("PHOTO") && atoi(getenv("PHOTO")) == 1) || photo_save || photo_load;
    int teacher = getenv("TEACHER") && atoi(getenv("TEACHER")) == 1;
    int lens[16], nl = 0, maxlen = 0, gen = argc > 3 ? atoi(argv[3]) : 32;
    char spec[256];
    snprintf(spec, sizeof spec, "%s", argc > 2 ? argv[2] : "64,512");
    for (char *t = strtok(spec, ","); t && nl < 16; t = strtok(NULL, ",")) {
        lens[nl] = atoi(t);
        if (lens[nl] < 1 || (photo && nl && lens[nl] <= lens[nl - 1])) {
            fprintf(stderr, "bad length %s (with PHOTO=1 lengths must grow)\n", t);
            return 2;
        }
        if (lens[nl] > maxlen) maxlen = lens[nl];
        nl++;
    }
    if (!nl || gen < 0 || maxlen + gen > qwen36_max_ctx()) {
        fprintf(stderr, "need lengths >= 1, gen >= 0, longest + gen <= %d (Q36_MAXT)\n", qwen36_max_ctx());
        return 2;
    }

    const char *tok = getenv("TOK");
    char tok_default[2048];
    if (!tok || !*tok) { snprintf(tok_default, sizeof tok_default, "%s/tokenizer.json", snap); tok = tok_default; }
    load_tokenizer(tok);
    if (!g_tok) { fprintf(stderr, "no tokenizer: %s\n", tok); return 1; }
    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror(argv[1]); return 1; }
    size_t cap = (size_t)maxlen * 32 + 4096;      /* only the start of the text is needed */
    char *text = malloc(cap + 1);
    size_t got = text ? fread(text, 1, cap, f) : 0;
    fclose(f);
    if (!text) return 1;
    text[got] = 0;
    int *ids = NULL, nids = 0;
    encode_text(text, &ids, &nids);
    free(text);
    if (nids < maxlen + (teacher ? gen : 0)) { fprintf(stderr, "the text gives %d tokens, %d needed\n", nids, maxlen + (teacher ? gen : 0)); return 1; }

    double t0 = now_s();
    static Model m;
    model_init(&m, snap, 16, 4);
    g_expert_gs = m.c.expert_gs;
    int eos[4], neos = serve_eos_ids(eos, 4);
#ifdef QWEN36_METAL_GRAPH
    const char *engine = m.graph ? "Metal graph" : "CPU";
    BenchPhoto ph = {0};
    if (photo && !m.graph) { fprintf(stderr, "PHOTO=1 needs the Metal graph\n"); return 2; }
#else
    const char *engine = "CPU";
    if (photo) { fprintf(stderr, "PHOTO=1 needs the Metal graph\n"); return 2; }
#endif
    fprintf(stderr, "[bench] %s ready in %.2f s; text %d tokens; lengths %s; gen %d; %s\n",
            engine, now_s() - t0, nids, argc > 2 ? argv[2] : "64,512", gen,
            photo ? "photo: each length continues the previous one" : "each length from an empty state");

    printf("ctx_tokens,prefill_tokens,prefill_s,prefill_tps,gen_tokens,gen_first_ms,gen_steady_tps,"
           "expert_miss,disk_s,fingerprint,tokens,nll,top1\n");
    fflush(stdout);
    if (photo) bench_fresh(&m, maxlen + gen);
    float *loaded = NULL;                       /* logits of the photo read from disk */
#ifdef QWEN36_METAL_GRAPH
    if (photo_load) {
        loaded = bench_photo_read(photo_load, &m, &ph, ids, nids);
        if (ph.kv_len != lens[0]) { fprintf(stderr, "the photo holds %d tokens: the lengths must start there\n", ph.kv_len); return 2; }
        fprintf(stderr, "[bench] photo of %d tokens read from %s\n", ph.kv_len, photo_load);
    }
#endif
    for (int li = 0, prev = 0; li < nl; li++) {
        int N = lens[li], V = m.c.vocab;
        if (!photo) { bench_fresh(&m, N + gen); prev = 0; }
#ifdef QWEN36_METAL_GRAPH
        else if (li) bench_photo_load(&m, &ph);   /* back to the end of the previous prompt */
#endif
        unsigned long long miss0 = (unsigned long long)m.miss;
        double disk0 = m.t_disk;

        double a = now_s(), prefill = 0, first = 0, steady = 0;
        float *lo;
        if (li == 0 && loaded) {                /* the prompt is the photo: nothing to read */
            lo = loaded;
            prev = N;
        } else {
            lo = step(&m, ids + prev, N - prev, prev);
            prefill = now_s() - a;
#ifdef QWEN36_METAL_GRAPH
            if (photo && (li + 1 < nl || photo_save)) bench_photo_save(&m, &ph);
#endif
        }
#ifdef QWEN36_METAL_GRAPH
        if (photo_save && li == nl - 1) bench_photo_write(photo_save, &m, &ph, lo, ids);
#endif
        uint64_t h = bench_fnv(0xcbf29ce484222325ull, lo, (size_t)V * sizeof(float)), ht = 0xcbf29ce484222325ull;
        double nll = 0;
        int top1 = 0;
        for (int g = 0; g < gen; g++) {
            int best = -1;
            for (int i = 0; i < V; i++) {
                int skip = 0;
                for (int e = 0; e < neos; e++) skip |= i == eos[e];
                if (!skip && (best < 0 || lo[i] > lo[best])) best = i;
            }
            if (teacher) {                      /* the text's token, and how likely the model found it */
                int want = ids[N + g];
                double mx = lo[0], z = 0;
                for (int i = 1; i < V; i++) if (lo[i] > mx) mx = lo[i];
                for (int i = 0; i < V; i++) z += exp((double)lo[i] - mx);
                nll += -((double)lo[want] - mx - log(z));
                top1 += best == want;
                best = want;
            }
            h = bench_fnv(h, &best, sizeof best);
            ht = bench_fnv(ht, &best, sizeof best);
            free(lo);
            double b = now_s();
            lo = step(&m, &best, 1, N + g);
            double d = now_s() - b;
            if (g == 0) first = d; else steady += d;
            h = bench_fnv(h, lo, (size_t)V * sizeof(float));
        }
        const char *dump = getenv("BENCH_DUMP");
        if (li == 0 && dump && *dump) {
            FILE *df = fopen(dump, "wb");
            if (!df || fwrite(lo, sizeof(float), (size_t)V, df) != (size_t)V || fclose(df)) {
                fprintf(stderr, "cannot write %s\n", dump);
                return 1;
            }
        }
        free(lo);
        printf("%d,%d,%.3f,%.2f,%d,%.1f,%.3f,%llu,%.2f,%016llx,%016llx,%.5f,%.3f\n", N, N - prev, prefill,
               prefill > 0 ? (N - prev) / prefill : 0.0, gen, first * 1e3, gen > 1 && steady > 0 ? (gen - 1) / steady : 0.0,
               (unsigned long long)m.miss - miss0, m.t_disk - disk0, (unsigned long long)h, (unsigned long long)ht,
               teacher && gen ? nll / gen : 0.0, teacher && gen ? (double)top1 / gen : 0.0);
        fflush(stdout);
        if (photo) prev = N;
    }
    tm_report();
    fprintf(stderr, "[bench] peak RSS %.2f GB\n", rss_gb());
    return 0;
}
