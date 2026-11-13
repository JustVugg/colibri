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

/* First-step diagnosis (2026-10-20): which pages of the file's dense weights are in memory around the first
 * decode step and during the decode, and the system's page-ins and the experts read during the prompt, that step
 * and the steps after it, with the time of day at the end of those lines. BENCH_DIAG=1 turns the lines on.
 * BENCH_PREFETCH_DENSE=1 asks for the dense weights (madvise WILLNEED, untimed) before the first step;
 * BENCH_MLOCK_DENSE locks them in memory (mlock): =1 once the model is loaded, =2 from the start of each prompt
 * to its end, =3 from the start of each prompt to the end of the decode step after it (unlocked outside that
 * step's time), =4 as 3 but through a second mapping of the file, unmapped after that step, =5 from the start of
 * each prompt to the end of the last decode step after it (unlocked outside the time); BENCH_MADV_RANDOM=1 marks
 * a GGUF's mapping MADV_RANDOM; BENCH_WRITE_DENSE_COPY=<file> writes a sparse copy of a GGUF without its routed
 * experts and exits; BENCH_PAUSE_AFTER_FIRST=<ms> sleeps that long after the first decode step, untimed. */
#include <sys/mman.h>
#include <mach/mach.h>
#ifdef QWEN36_METAL_GRAPH
static int diag_file(const Model *m, const void *p) {
    const Qwen36Graph *g = m->graph;
    uintptr_t a = (uintptr_t)p;
    if (g->map && a >= (uintptr_t)g->map_base && a - (uintptr_t)g->map_base < g->map_bytes) return 1;
    for (int i = 0; g_gguf && g_gg_map && i < g_gg.n_files; i++)
        if (g_gg_map[i] && a >= (uintptr_t)g_gg_map[i] && a - (uintptr_t)g_gg_map[i] < g_gg_map_bytes[i]) return 1;
    return 0;
}
typedef void (*diag_fn)(void *at, size_t bytes, double *in, double *all);
static void diag_count(void *at, size_t bytes, double *in, double *all) {
    size_t page = (size_t)getpagesize();
    uintptr_t lo = (uintptr_t)at & ~(page - 1), hi = ((uintptr_t)at + bytes + page - 1) & ~(page - 1);
    size_t n = (hi - lo) / page;
    char *v = malloc(n);
    if (!v) return;
    if (mincore((caddr_t)lo, hi - lo, v) == 0) {
        size_t k = 0;
        for (size_t i = 0; i < n; i++) k += (v[i] & MINCORE_INCORE) != 0;
        *in += (double)k * page / 1e6;
        *all += (double)n * page / 1e6;
    }
    free(v);
}
static void diag_mlock(void *at, size_t bytes, double *in, double *all) {
    size_t page = (size_t)getpagesize();
    uintptr_t lo = (uintptr_t)at & ~(page - 1), hi = ((uintptr_t)at + bytes + page - 1) & ~(page - 1);
    if (mlock((void *)lo, hi - lo) == 0) *in += (double)(hi - lo) / 1e6;
    *all += (double)(hi - lo) / 1e6;
}
static void diag_munlock(void *at, size_t bytes, double *in, double *all) {
    size_t page = (size_t)getpagesize();
    uintptr_t lo = (uintptr_t)at & ~(page - 1), hi = ((uintptr_t)at + bytes + page - 1) & ~(page - 1);
    if (munlock((void *)lo, hi - lo) == 0) *in += (double)(hi - lo) / 1e6;
    *all += (double)(hi - lo) / 1e6;
}
static void diag_willneed(void *at, size_t bytes, double *in, double *all) {
    size_t page = (size_t)getpagesize();
    uintptr_t lo = (uintptr_t)at & ~(page - 1), hi = ((uintptr_t)at + bytes + page - 1) & ~(page - 1);
    madvise((void *)lo, hi - lo, MADV_WILLNEED);
    *all += (double)(hi - lo) / 1e6;
    (void)in;
}
/* BENCH_MLOCK_DENSE=4: the dense pages locked through a second mapping of each GGUF file, unmapped after the
 * first decode step (the graph's own mapping never holds them locked) */
static char *diag_map2[16];
static void *diag_second(void *at) {
    for (int i = 0; g_gguf && g_gg_map && i < g_gg.n_files && i < 16; i++)
        if (g_gg_map[i] && (uintptr_t)at >= (uintptr_t)g_gg_map[i] && (uintptr_t)at - (uintptr_t)g_gg_map[i] < g_gg_map_bytes[i])
            return diag_map2[i] ? diag_map2[i] + ((uintptr_t)at - (uintptr_t)g_gg_map[i]) : NULL;
    return NULL;
}
static void diag_mlock2(void *at, size_t bytes, double *in, double *all) {
    void *p = diag_second(at);
    if (p) diag_mlock(p, bytes, in, all);
    else *all += (double)bytes / 1e6;
}
static int diag_map2_open(void) {
    int ok = 1;
    for (int i = 0; g_gguf && g_gg_map && i < g_gg.n_files && i < 16; i++) {
        void *b = mmap(NULL, g_gg_map_bytes[i], PROT_READ, MAP_SHARED, g_gg.fds[i], 0);
        diag_map2[i] = b == MAP_FAILED ? NULL : (char *)b;
        ok &= diag_map2[i] != NULL;
    }
    return ok;
}
static void diag_map2_close(void) {          /* munmap unlocks what it unmaps */
    for (int i = 0; i < 16; i++)
        if (diag_map2[i]) { munmap(diag_map2[i], g_gg_map_bytes[i]); diag_map2[i] = NULL; }
}
/* the file's dense matrices: [0] the head, [1..4] the layers by quarters */
static void diag_walk(Model *m, diag_fn fn, double in[5], double all[5]) {
    Qwen36Graph *g = m->graph;
    for (int b = 0; b < 5; b++) in[b] = all[b] = 0;
    q36_tensor *ht[2] = {g->head.q, g->head.sc};
    for (int t = 0; t < 2; t++)
        if (ht[t] && diag_file(m, q36_tensor_data(ht[t]))) fn(q36_tensor_data(ht[t]), q36_tensor_bytes(ht[t]), &in[0], &all[0]);
    for (int il = 0; il < m->c.n_layers; il++) {
        QgLayer *L = &g->L[il];
        QgDense *d[] = {&L->q, &L->k, &L->v, &L->o, &L->qkv, &L->z, &L->out, &L->router, &L->sh_g, &L->sh_u, &L->sh_d};
        int b = 1 + il * 4 / m->c.n_layers;
        for (int k = 0; k < (int)(sizeof d / sizeof d[0]); k++) {
            q36_tensor *t2[2] = {d[k]->q, d[k]->sc};
            for (int t = 0; t < 2; t++)
                if (t2[t] && diag_file(m, q36_tensor_data(t2[t])))
                    fn(q36_tensor_data(t2[t]), q36_tensor_bytes(t2[t]), &in[b], &all[b]);
        }
    }
}
static void diag_resident(Model *m, const char *when) {
    double in[5], all[5];
    diag_walk(m, diag_count, in, all);
    fprintf(stderr, "[diag] %s: in memory head %.0f/%.0f MB, layers 0-9 %.0f/%.0f, 10-19 %.0f/%.0f, 20-29 %.0f/%.0f, "
            "30-39 %.0f/%.0f\n", when, in[0], all[0], in[1], all[1], in[2], all[2], in[3], all[3], in[4], all[4]);
}
#include <sys/time.h>
static double diag_epoch(void) { struct timeval tv; gettimeofday(&tv, NULL); return tv.tv_sec + tv.tv_usec * 1e-6; }
static uint64_t diag_pageins(void) {
    vm_statistics64_data_t vs;
    mach_msg_type_number_t n = HOST_VM_INFO64_COUNT;
    return host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t)&vs, &n) == KERN_SUCCESS ? vs.pageins : 0;
}
#endif

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
    int diag = getenv("BENCH_DIAG") && atoi(getenv("BENCH_DIAG")) == 1;
    int prefetch = getenv("BENCH_PREFETCH_DENSE") && atoi(getenv("BENCH_PREFETCH_DENSE")) == 1;
    int mlock_mode = getenv("BENCH_MLOCK_DENSE") ? atoi(getenv("BENCH_MLOCK_DENSE")) : 0;
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

    {   /* a model directory made by tools/gguf_import.py (its gguf.json names the file), or a GGUF file, opened
         * as the engine's main opens it (2026-10-19, for the GGUF port's measures) */
        static char gguf_path[4096];
        int gguf_dir = 0;
        char gj[4096];
        snprintf(gj, sizeof gj, "%s/gguf.json", snap);
        FILE *gf = fopen(gj, "rb");
        if (gf) {
            char buf[8192];
            size_t n = fread(buf, 1, sizeof buf - 1, gf);
            fclose(gf);
            buf[n] = 0;
            char *arena = NULL;
            jval *r = json_parse(buf, &arena);
            const char *g = r ? jstr(r, "gguf") : NULL;
            if (!g || !*g) { fprintf(stderr, "%s: no \"gguf\" path\n", gj); return 1; }
            if (g[0] == '/') snprintf(gguf_path, sizeof gguf_path, "%s", g);
            else snprintf(gguf_path, sizeof gguf_path, "%s/%s", snap, g);
            free(arena);
            gguf_dir = 1;
        }
        if (gguf_dir || gguf_is_file(snap)) {
            const char *gp = gguf_dir ? gguf_path : snap;
            if (gguf_open(&g_gg, gp)) { fprintf(stderr, "GGUF: %s\n", g_gg.err); return 1; }
            g_gguf = 1; g_ssm_a_direct = 1; g_dn_tiled = 1;
            fprintf(stderr, "[bench] GGUF %s: %d file(s), %d tensors\n", gp, g_gg.n_files, g_gg.n_tensors);
            const char *cp = getenv("BENCH_WRITE_DENSE_COPY");   /* dense-copy.patch maps it */
            if (cp && *cp) {
                struct stat st;
                int out = open(cp, O_WRONLY | O_CREAT | O_EXCL, 0644);
                if (g_gg.n_files != 1 || out < 0 || fstat(g_gg.fds[0], &st) || ftruncate(out, st.st_size)) {
                    fprintf(stderr, "[diag] cannot write the copy %s\n", cp); return 1; }
                size_t cap = (size_t)64 << 20;
                char *buf = malloc(cap);
                double mb = 0;
                for (int i = 0; buf && i < g_gg.n_tensors; i++) {
                    const gguf_tensor *t = &g_gg.t[i];
                    if (strstr(t->name, "_exps")) continue;
                    for (int64_t at = 0; at < t->nbytes; ) {
                        size_t n = (size_t)(t->nbytes - at) < cap ? (size_t)(t->nbytes - at) : cap;
                        if (pread(g_gg.fds[0], buf, n, (off_t)(t->off + at)) != (ssize_t)n ||
                            pwrite(out, buf, n, (off_t)(t->off + at)) != (ssize_t)n) {
                            fprintf(stderr, "[diag] copy failed at %s\n", t->name); return 1; }
                        at += (int64_t)n;
                    }
                    mb += (double)t->nbytes / 1e6;
                }
                if (!buf || close(out)) { fprintf(stderr, "[diag] copy failed\n"); return 1; }
                fprintf(stderr, "[diag] %s: %.0f MB of tensors (all but the routed experts) at their offsets\n", cp, mb);
                return 0;
            }
        }
    }
    const char *tok = getenv("TOK");
    char tok_default[2048];
    if (!tok || !*tok) { snprintf(tok_default, sizeof tok_default, "%s/tokenizer.json", snap); tok = tok_default; }
    load_tokenizer(tok);
    if (!g_tok) { fprintf(stderr, "no tokenizer: %s\n", tok); return 1; }
    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror(argv[1]); return 1; }
    size_t cap = (size_t)(maxlen + (teacher ? gen : 0)) * 32 + 4096;   /* only the start of the text is needed */
    char *text = malloc(cap + 1);
    size_t got = text ? fread(text, 1, cap, f) : 0;
    fclose(f);
    if (!text) return 1;
    text[got] = 0;
    int *ids = NULL, nids = 0;
    encode_text(text, &ids, &nids);
    free(text);
    if (nids < maxlen + (teacher ? gen : 0)) { fprintf(stderr, "the text gives %d tokens, %d needed\n", nids, maxlen + (teacher ? gen : 0)); return 1; }
    {   /* BENCH_IDS=<file>: the token ids the run reads (the longest prompt and, with TEACHER=1, the forced
         * words), one per line; BENCH_IDS_ONLY=1 stops there, before the model is loaded (2026-10-19: the
         * tokenizer.json of a GGUF import against the model's own) */
        const char *ip = getenv("BENCH_IDS");
        if (ip && *ip) {
            FILE *fi = fopen(ip, "w");
            if (!fi) { perror(ip); return 1; }
            for (int i = 0; i < maxlen + (teacher ? gen : 0); i++) fprintf(fi, "%d\n", ids[i]);
            fclose(fi);
            if (getenv("BENCH_IDS_ONLY") && atoi(getenv("BENCH_IDS_ONLY")) == 1) return 0;
        }
    }

    double t0 = now_s();
    static Model m;
    model_init(&m, snap, 16, 4);
    g_expert_gs = m.c.expert_gs;
    int eos[4], neos = serve_eos_ids(eos, 4);
#ifdef QWEN36_METAL_GRAPH
    if (getenv("BENCH_MADV_RANDOM") && atoi(getenv("BENCH_MADV_RANDOM")) == 1 && m.graph && g_gguf && g_gg_map) {
        int ok = 1;
        for (int i = 0; i < g_gg.n_files; i++)
            ok &= g_gg_map[i] && madvise((void *)g_gg_map[i], g_gg_map_bytes[i], MADV_RANDOM) == 0;
        fprintf(stderr, "[diag] the GGUF mapping marked MADV_RANDOM: %s\n", ok ? "ok" : "failed");
    }
#endif
#ifdef QWEN36_METAL_GRAPH
    if (getenv("BENCH_MLOCK_DENSE") && atoi(getenv("BENCH_MLOCK_DENSE")) == 1 && m.graph) {
        double in[5], all[5], t = now_s();
        diag_walk(&m, diag_mlock, in, all);
        fprintf(stderr, "[diag] the dense pages locked in memory (mlock): %.0f of %.0f MB in %.0f ms\n",
                in[0] + in[1] + in[2] + in[3] + in[4], all[0] + all[1] + all[2] + all[3] + all[4], (now_s() - t) * 1e3);
    }
#endif
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
        uint64_t pin_p = 0;
#ifdef QWEN36_METAL_GRAPH
        if (diag && m.graph) { diag_resident(&m, "before the prompt"); pin_p = diag_pageins(); }
        if (m.graph && mlock_mode >= 2 && mlock_mode <= 5) {
            double in[5], all[5], t = now_s();
            if (mlock_mode == 4 && !diag_map2_open()) fprintf(stderr, "[diag] the second mapping failed\n");
            diag_walk(&m, mlock_mode == 4 ? diag_mlock2 : diag_mlock, in, all);
            fprintf(stderr, "[diag] locked for the prompt%s: %.0f of %.0f MB in %.0f ms\n",
                    mlock_mode == 5 ? " (to the end of the decode)" : mlock_mode == 4 ? " (a second mapping)" : mlock_mode == 3 ? " (to the end of the first step)" : "",
                    in[0] + in[1] + in[2] + in[3] + in[4], all[0] + all[1] + all[2] + all[3] + all[4], (now_s() - t) * 1e3);
        }
#endif

        double a = now_s(), prefill = 0, first = 0, steady = 0;
        float *lo;
        if (li == 0 && loaded) {                /* the prompt is the photo: nothing to read */
            lo = loaded;
            prev = N;
        } else {
            lo = step(&m, ids + prev, N - prev, prev);
            prefill = now_s() - a;
#ifdef QWEN36_METAL_GRAPH
            if (m.graph && getenv("BENCH_MLOCK_DENSE") && atoi(getenv("BENCH_MLOCK_DENSE")) == 2) {
                double in[5], all[5], t = now_s();
                diag_walk(&m, diag_munlock, in, all);
                fprintf(stderr, "[diag] unlocked after the prompt: %.0f of %.0f MB in %.0f ms\n",
                        in[0] + in[1] + in[2] + in[3] + in[4], all[0] + all[1] + all[2] + all[3] + all[4], (now_s() - t) * 1e3);
            }
            if (diag && m.graph)
                fprintf(stderr, "[diag] prompt of %d tokens: %.0f ms, %llu experts read, page-ins %.0f MB, at %.3f\n", N - prev,
                        prefill * 1e3, (unsigned long long)m.miss - miss0,
                        (double)(diag_pageins() - pin_p) * getpagesize() / 1e6, diag_epoch());
#endif
#ifdef QWEN36_METAL_GRAPH
            if (photo && (li + 1 < nl || photo_save)) bench_photo_save(&m, &ph);
#endif
        }
#ifdef QWEN36_METAL_GRAPH
        if (photo_save && li == nl - 1) bench_photo_write(photo_save, &m, &ph, lo, ids);
#endif
        uint64_t h = bench_fnv(0xcbf29ce484222325ull, lo, (size_t)V * sizeof(float)), ht = 0xcbf29ce484222325ull;
        /* speculative decoding as generate() runs it: q36_spec_step drafts and verifies (on the Metal graph
         * the MTP head, when the container has mtp.safetensors; QWEN36_MTP=0 decodes plainly). Older
         * versions have no drafts on the graph: there it is a plain step. */
        Q36Spec sp;
        q36_spec_begin(&m, &sp, ids, N);
        double nll = 0;
        int top1 = 0;
        /* BENCH_NLL=<file>: with TEACHER=1, one line per generated word of the first length: its nll and
         * whether the first choice was the text's (paired comparisons between two versions, analizza_nll.py) */
        const char *nll_path = getenv("BENCH_NLL");
        FILE *nf = teacher && li == 0 && nll_path && *nll_path ? fopen(nll_path, "w") : NULL;
        /* BENCH_TEXT=<file>: the words chosen after the first length's prompt, as text (to read what a version
         * generates: a faster answer that repeats itself is not a gain) */
        const char *text_path = getenv("BENCH_TEXT");
        int *chosen = li == 0 && text_path && *text_path ? malloc(sizeof(int) * (size_t)(gen > 0 ? gen : 1)) : NULL;
        uint64_t pin_rest = 0;                  /* first-step diagnosis: steps 2 to gen */
        unsigned long long miss_rest = 0;
        double disk_rest = 0;
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
                if (nf) fprintf(nf, "%d %.6f %d\n", g, -((double)lo[want] - mx - log(z)), best == want);
                best = want;
            }
            if (chosen) chosen[g] = best;
            h = bench_fnv(h, &best, sizeof best);
            ht = bench_fnv(ht, &best, sizeof best);
            free(lo);
            uint64_t pin0 = 0;
            unsigned long long miss_s = 0;
            double disk_s0 = 0;
#ifdef QWEN36_METAL_GRAPH
            if (diag && m.graph && g == 0) {
                diag_resident(&m, "before the first step");
                if (prefetch) {
                    double in[5], all[5], t = now_s();
                    diag_walk(&m, diag_willneed, in, all);
                    fprintf(stderr, "[diag] the dense pages asked for (madvise WILLNEED, %.0f MB) in %.0f ms\n",
                            all[0] + all[1] + all[2] + all[3] + all[4], (now_s() - t) * 1e3);
                    diag_resident(&m, "after the request");
                }
                pin0 = diag_pageins();
                miss_s = (unsigned long long)m.miss;
                disk_s0 = m.t_disk;
            }
#endif
            double b = now_s();
            lo = q36_spec_step(&m, &sp, best, N + g, gen - 1 - g);
            double d = now_s() - b;
#ifdef QWEN36_METAL_GRAPH
            if (diag && m.graph && g == 0) {
                uint64_t pin1 = diag_pageins();
                fprintf(stderr, "[diag] first step %.1f ms: %llu experts read, disk %.1f ms, page-ins %.0f MB, at %.3f\n",
                        d * 1e3, (unsigned long long)m.miss - miss_s, (m.t_disk - disk_s0) * 1e3,
                        (double)(pin1 - pin0) * getpagesize() / 1e6, diag_epoch());
                diag_resident(&m, "after the first step");
                pin_rest = diag_pageins();
                miss_rest = (unsigned long long)m.miss;
                disk_rest = m.t_disk;
            }
            if (g == 0 && m.graph && (mlock_mode == 3 || mlock_mode == 4)) {   /* untimed */
                double in[5], all[5], t = now_s();
                if (mlock_mode == 3) diag_walk(&m, diag_munlock, in, all);
                else { diag_walk(&m, diag_count, in, all); diag_map2_close(); }
                fprintf(stderr, "[diag] unlocked after the first step%s: %.0f of %.0f MB in %.0f ms\n",
                        mlock_mode == 4 ? " (the second mapping unmapped; MB in memory)" : "",
                        in[0] + in[1] + in[2] + in[3] + in[4], all[0] + all[1] + all[2] + all[3] + all[4], (now_s() - t) * 1e3);
            }
            if (g == 0 && getenv("BENCH_PAUSE_AFTER_FIRST") && atoi(getenv("BENCH_PAUSE_AFTER_FIRST")) > 0)
                usleep((useconds_t)atoi(getenv("BENCH_PAUSE_AFTER_FIRST")) * 1000);   /* untimed: before step 2 */
            if (diag && m.graph && (g == 31 || g == 95 || g == gen - 1) && gen > 32) {   /* the dense pages in decode */
                char when[48];
                snprintf(when, sizeof when, "at step %d", g + 1);
                diag_resident(&m, when);
            }
            if (diag && m.graph && g == 31 && gen > 32)
                fprintf(stderr, "[diag] steps 2-32: %llu experts read, disk %.1f ms, page-ins %.0f MB, at %.3f\n",
                        (unsigned long long)m.miss - miss_rest, (m.t_disk - disk_rest) * 1e3,
                        (double)(diag_pageins() - pin_rest) * getpagesize() / 1e6, diag_epoch());
            if (diag && m.graph && g == gen - 1 && gen > 1)
                fprintf(stderr, "[diag] steps 2-%d: %llu experts read, disk %.1f ms, page-ins %.0f MB, at %.3f\n", gen,
                        (unsigned long long)m.miss - miss_rest, (m.t_disk - disk_rest) * 1e3,
                        (double)(diag_pageins() - pin_rest) * getpagesize() / 1e6, diag_epoch());
            if (g == gen - 1 && m.graph && mlock_mode == 5) {   /* untimed: after the last step of this length */
                double in[5], all[5], t = now_s();
                diag_walk(&m, diag_munlock, in, all);
                fprintf(stderr, "[diag] unlocked after the last step: %.0f of %.0f MB in %.0f ms\n",
                        in[0] + in[1] + in[2] + in[3] + in[4], all[0] + all[1] + all[2] + all[3] + all[4], (now_s() - t) * 1e3);
            }
#endif
            if (g == 0) first = d; else steady += d;
            h = bench_fnv(h, lo, (size_t)V * sizeof(float));
        }
        if (nf) fclose(nf);
        if (chosen) {
            size_t cap = (size_t)gen * 255 + 1;
            char *text = malloc(cap);
            FILE *tf = fopen(text_path, "w");
            if (text && tf) {
                int len = decode_range(chosen, 0, gen, text, (int)cap);
                fwrite(text, 1, len > 0 ? (size_t)len : 0, tf);
            }
            if (tf) fclose(tf);
            free(text); free(chosen);
        }
        q36_spec_end(&m, &sp);
        char scope[32];
        snprintf(scope, sizeof scope, "%d tokens", N);
        q36_spec_report(&sp, scope);
        q36_spec_free(&sp);
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
