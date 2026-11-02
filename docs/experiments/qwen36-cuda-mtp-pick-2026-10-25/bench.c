/* qwen36-bench: how fast the Qwen3.6 engine reads a prompt and generates, measured the way antirez's
 * ds4-bench does (ds4, MIT). One process, one version of the engine, compiled in and used as is. The
 * `bench` driver beside this file builds it, runs two versions in alternating order and reads the CSV;
 * README.md gives the CSV contract every engine's harness follows.
 *
 *   qwen36-bench --model <dir> --text <file> [--lengths 64,512] [--gen 32] [--photo] [--teacher]
 *                [--steps <file>] [--nll <file>] [--words <file>] [--pause <s>]
 *   qwen36-bench --model <dir> --session <file> [--steps <file>] [--nll <file>] [--pause <s>]
 *                [--tokens-only]
 *
 * For each length N the first N tokens of the text are read (timed), then --gen tokens follow (timed, the
 * first step apart): the most likely one, never an end-of-text token, or with --teacher the text's own
 * next token, with the mean negative log likelihood of those tokens (nll, nats a token) and how often the
 * most likely one was the text's (top1). Lengths run in one process, in order, so the expert cache carries
 * over. Each length starts from an empty state; with --photo it continues the previous one, as ds4-bench's
 * context frontiers do: after each prefill the sequence state is copied out of the GPU (the "photo"), the
 * generation runs, the photo is put back and the next length reads only the tokens it adds. One CSV row per
 * length on stdout.
 *
 * fingerprint: FNV-1a over every logit of the run and the chosen tokens; tokens: over the chosen tokens
 * alone (equal when the same words come out though the logits differ in the last bits). A change that must
 * not alter results leaves the fingerprint unchanged. In the exact mode (QWEN36_METAL_EXACT=1) the engine gives
 * the same numbers however a prompt is cut, so with --photo each length's fingerprint equals the one without;
 * the fast mode's prompt kernels may differ in the last bits when the cut changes (seen on the tests' model).
 *
 * --session replays a chat turn by turn the way the server runs it (serve_one), each answer teacher-forced.
 * A session file (sessione.py writes one from a transcript) frames each turn by bytes,
 *
 *   TURN <k> <prompt_bytes> <answer_bytes>\n<prompt><answer>\n
 *
 * where a turn's prompt is the whole conversation before its answer, as the server renders it, and its
 * tokens must begin those of prompt + answer. The prompt goes through q36_serve_start, which keeps what the
 * state already holds (the live prefix, or the graph's copy at the previous prompt's last <|im_start|>) and
 * reads the rest; then the answer is fed token by token, without its end-of-turn token, as in serve_one:
 * the next prompt carries it. One CSV row per turn: the 13 columns of a length, with ctx_tokens the
 * prompt's tokens, prefill_tokens those read and prefill_s the time q36_serve_start took (the time to the
 * first token); then reused (the prompt's tokens not read again), kv_len (the state after the turn), rss_gb
 * (the peak so far), greedy_prefix (the answer's tokens before the first one the engine's own choice at
 * temperature 0, end-of-turn tokens included, would not have given), tool_calls (<tool_call> spans in the
 * answer), tool_calls_ok (those where every token is that choice), stop_ok (1 when that choice after the
 * answer is an end-of-turn token). The protocol's lines (ACCEPT, ERROR) go to stderr, stdin is /dev/null.
 *
 * --steps writes a line per step, "<length|turn> <step> <experts read> <reads' s> <step ms>", the prompt as
 * step -1: where the reads go. --nll writes "<step> <nll> <top1>" for each generated token of the first
 * length (a session: "<turn> <step> <nll> <top1>"), for paired comparisons (analizza_nll.py). --words writes
 * the first length's generated words as text: a faster answer that repeats itself is not a gain. --pause
 * rests the engine s seconds before every length after the first (with --photo) or every turn after the
 * first, never timed. Q36_MAXT, when unset, is raised to what the run needs. The engine's own variables
 * (QWEN36_*, COLI_TIMERS, ...) stay in the environment; with COLI_TIMERS=1 its phase report ends stderr. */
/* Off the Metal graph a backend starts inline in the engine's main, after model_init (the CUDA tier with its
 * experts in VRAM, the chain, the trunk's placement), not in a function the harness could call. There the
 * harness runs that main with SERVE=1 and takes its model where main hands it to serve_loop, whose first call
 * is coli_serve_binary_mode: back in bench_model with longjmp, main's setup done. serve_codec.h, the other
 * header that calls it, comes first with compat.h, which defines it: both keep their names. */
#define _GNU_SOURCE
#include <setjmp.h>
#include "serve_codec.h"
static jmp_buf bench_main_done;
static void *bench_main_model;
#define coli_serve_binary_mode() (bench_main_model = m, longjmp(bench_main_done, 1))
#define main qwen36_engine_main
#include "qwen36.c"
#undef main
#undef coli_serve_binary_mode

static uint64_t bench_fnv(uint64_t h, const void *data, size_t n) {
    const unsigned char *p = (const unsigned char *)data;
    for (size_t i = 0; i < n; i++) h = (h ^ p[i]) * 0x100000001b3ull;
    return h;
}

typedef struct {
    const char *model, *tokenizer, *text, *lengths, *session, *steps, *nll, *words;
    int gen, photo, teacher, pause, tokens_only;
} BenchOpts;

/* An idle engine for s seconds, outside every timed window: the GPU has finished first. */
static void bench_pause(int s, const char *what, int which) {
    if (s <= 0) return;
#ifdef QWEN36_METAL_GRAPH
    q36_gpu_sync();
#endif
    fprintf(stderr, "[bench] pause %d s before %s %d\n", s, what, which);
    sleep((unsigned)s);
}

/* The run needs `need` tokens of context: Q36_MAXT unset is raised to it (the graph allocates what a sequence
 * needs, not the ceiling, so this changes no measurement); one set too low is an error. 0, or an exit code. */
static int bench_context(int need, const char *why) {
    if (need <= qwen36_max_ctx()) return 0;
    const char *e = getenv("Q36_MAXT");
    if (e && *e) {
        fprintf(stderr, "%s needs %d tokens of context, Q36_MAXT=%s holds %d\n", why, need, e, qwen36_max_ctx());
        return 2;
    }
    char v[32];
    snprintf(v, sizeof v, "%d", need + 64);
    setenv("Q36_MAXT", v, 1);
    fprintf(stderr, "[bench] Q36_MAXT=%s (%s)\n", v, why);
    return 0;
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

/* The model, ready on its backend: the Metal graph starts in model_init; any other backend in the engine's main,
 * which loads the tokenizer again (the first copy stays in memory). cap and bits as the records run the graph
 * (16, 4), and as docs/qwen36-cuda-tier.md runs the engine elsewhere (256, 4). */
static Model *bench_model(const BenchOpts *o) {
#ifdef QWEN36_METAL_GRAPH
    static Model m;
    model_init(&m, o->model, 16, 4);
    g_expert_gs = m.c.expert_gs;
    return &m;
#else
    char *argv[] = {"qwen36", "256", "4", NULL};
    setenv("SNAP", o->model, 1);
    if (o->tokenizer) setenv("TOK", o->tokenizer, 1);
    setenv("SERVE", "1", 1);
    if (!setjmp(bench_main_done)) {
        int rc = qwen36_engine_main(3, argv);
        fprintf(stderr, "the engine's main returned %d before its serve loop\n", rc);
        exit(rc ? rc : 1);
    }
    unsetenv("SERVE");
    return bench_main_model;
#endif
}

static const char *bench_engine(Model *m) {
#ifdef QWEN36_METAL_GRAPH
    return m->graph ? "Metal graph" : "CPU";
#else
    (void)m;
    return qt_ready() ? "CUDA" : "CPU";
#endif
}

/* The model's file and its tokenizer, as the engine's main opens them: a model directory, a directory made
 * by tools/gguf_import.py (its gguf.json names the file) or a GGUF file. 0, or the exit code after a message. */
static int bench_open(const BenchOpts *o) {
    const char *snap = o->model;
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
    }
    char tok_default[2048];
    const char *tok = o->tokenizer;
    if (!tok) { snprintf(tok_default, sizeof tok_default, "%s/tokenizer.json", snap); tok = tok_default; }
    load_tokenizer(tok);
    if (!g_tok) { fprintf(stderr, "no tokenizer: %s\n", tok); return 1; }
    return 0;
}

static FILE *bench_out(const char *path) {
    if (!path) return NULL;
    FILE *f = fopen(path, "w");
    if (!f) { perror(path); exit(1); }
    return f;
}

/* --session: a session file's turns, tokenized. */
typedef struct {
    char *prompt;                               /* the prompt's bytes, then a 0 */
    int *ids, np;                               /* its tokens */
    int *full, nfull;                           /* the tokens of prompt + answer */
} BenchTurn;

/* An added token's id (<tool_call>, ...), -1 when the tokenizer has none. */
static int bench_added_id(const char *s) {
    for (int k = 0; k < g_nspecial; k++)
        if (!strcmp(g_sp_str[k], s)) return g_sp_id[k];
    return -1;
}

/* The turns of a session file; how many, 0 after a message. */
static int bench_session_read(const char *path, BenchTurn **out) {
    FILE *f = fopen(path, "rb");
    long size = -1;
    if (!f || fseek(f, 0, SEEK_END) || (size = ftell(f)) <= 0 || fseek(f, 0, SEEK_SET)) {
        fprintf(stderr, "%s: cannot read the session\n", path);
        return 0;
    }
    char *buf = malloc((size_t)size + 1);
    if (!buf || fread(buf, 1, (size_t)size, f) != (size_t)size) {
        fprintf(stderr, "%s: cannot read the session\n", path);
        return 0;
    }
    fclose(f);
    buf[size] = 0;
    BenchTurn *t = NULL;
    int nt = 0;
    for (size_t at = 0; at < (size_t)size;) {
        char head[64];
        const char *nl = memchr(buf + at, '\n', (size_t)size - at < sizeof head ? (size_t)size - at : sizeof head);
        int k = 0, pb = 0, ab = 0, used = -1;
        size_t body = nl ? (size_t)(nl - buf) + 1 : 0;
        if (nl) {
            memcpy(head, buf + at, body - 1 - at);
            head[body - 1 - at] = 0;
            sscanf(head, "TURN %d %d %d%n", &k, &pb, &ab, &used);
        }
        if (!nl || used != (int)strlen(head) || k != nt + 1 || pb < 1 || ab < 1 ||
            body + (size_t)pb + (size_t)ab >= (size_t)size || buf[body + pb + ab] != '\n' ||
            memchr(buf + body, 0, (size_t)pb + ab)) {
            fprintf(stderr, "%s: turn %d is not a frame TURN <k> <prompt_bytes> <answer_bytes>\n", path, nt + 1);
            return 0;
        }
        BenchTurn *grown = realloc(t, (size_t)(nt + 1) * sizeof *t);
        char *both = malloc((size_t)pb + ab + 1);
        if (!grown || !both) { fprintf(stderr, "no memory for the session\n"); return 0; }
        t = grown;
        BenchTurn *u = &t[nt];
        memcpy(both, buf + body, (size_t)pb + ab);
        both[pb + ab] = 0;
        u->prompt = strndup(both, (size_t)pb);
        encode_text(u->prompt, &u->ids, &u->np);
        encode_text(both, &u->full, &u->nfull);
        free(both);
        if (u->nfull <= u->np || memcmp(u->full, u->ids, (size_t)u->np * sizeof(int))) {
            fprintf(stderr, "%s: the renderer and the tokenizer disagree at the boundary of turn %d\n", path, k);
            return 0;
        }
        nt++;
        at = body + pb + ab + 1;
    }
    free(buf);
    if (!nt) fprintf(stderr, "%s: no turns\n", path);
    *out = t;
    return nt;
}

/* How many of the prompt's tokens q36_serve_start will not read again: the
 * live prefix (kv_prefix_reuse), or the graph's copy at the previous prompt's
 * last <|im_start|> when it holds more. Its decision, taken before it for the
 * report (the log's [PREFIX] and "from the previous turn" lines say what it
 * did); ensure_kv, which it calls first, keeps both. */
static int bench_reuse(Model *m, const int *ids, int np) {
    if (kv_prefix_off()) return 0;
    int reuse = kv_prefix_reuse(&m->kvp, ids, np);
#ifdef QWEN36_METAL_GRAPH
    if (m->graph) {
        Qwen36Graph *g = m->graph;
        int turn = serve_turn_id(), cut = np - 1;
        for (int i = np - 1; i > 0; i--) if (ids[i] == turn) { cut = i; break; }
        if (g->snap_len > 0 && g->snap_len <= cut && !memcmp(ids, g->snap_ids, (size_t)g->snap_len * sizeof(int)) &&
            reuse < g->snap_len)
            reuse = g->snap_len;
    }
#endif
    return reuse;
}

static int bench_session(const BenchOpts *o) {
    {   int rc = bench_open(o); if (rc) return rc; }
    BenchTurn *t = NULL;
    int nt = bench_session_read(o->session, &t), longest = 0, answers = 0;
    if (!nt) return 1;
    for (int k = 0; k < nt; k++) {
        if (t[k].nfull > longest) longest = t[k].nfull;
        answers += t[k].nfull - t[k].np;
    }
    fprintf(stderr, "[bench] session: %d turns, longest context %d tokens, %d answer tokens in all\n", nt, longest,
            answers);
    /* the end-of-turn token's logits need a row too */
    {   int rc = bench_context(longest + 1, "the longest turn"); if (rc) return rc; }
    if (o->tokens_only) return 0;
    /* stdout keeps the rows; what the engine prints there (ACCEPT, ERROR) goes to stderr */
    fflush(stdout);
    int fd = dup(1);
    FILE *csv = fd >= 0 ? fdopen(fd, "w") : NULL;
    if (!csv || dup2(2, 1) < 0 || !freopen("/dev/null", "r", stdin)) { perror("--session"); return 1; }
    setenv("COLI_PREFIX_LOG", "1", 0);

    double t0 = now_s();
    Model *m = bench_model(o);
    int eos[4], neos = serve_eos_ids(eos, 4);
    fprintf(stderr, "[bench] %s ready in %.2f s; session of %d turns, each through the serve path\n",
            bench_engine(m), now_s() - t0, nt);
    fprintf(csv, "ctx_tokens,prefill_tokens,prefill_s,prefill_tps,gen_tokens,gen_first_ms,gen_steady_tps,"
                 "expert_miss,disk_s,fingerprint,tokens,nll,top1,reused,kv_len,rss_gb,greedy_prefix,tool_calls,"
                 "tool_calls_ok,stop_ok\n");
    fflush(csv);
    FILE *nf = bench_out(o->nll), *sf = bench_out(o->steps);
    int V = m->c.vocab, open_id = bench_added_id("<tool_call>"), close_id = bench_added_id("</tool_call>");
    for (int k = 0; k < nt; k++) {
        if (k > 0) bench_pause(o->pause, "turn", k + 1);
        BenchTurn *u = &t[k];
        int n = u->nfull - u->np, *ids = NULL, np = 0;
        const int *want = u->full + u->np;
        ServeReq q;
        memset(&q, 0, sizeof q);
        snprintf(q.id, sizeof q.id, "t%d", k + 1);
        q.max_tok = n + 1;                      /* the answer, then its end-of-turn token */
        q.temp = 0;
        q.top_p = 1;
        q.payload = u->prompt;
        q.plen = (int)strlen(u->prompt);
        int reuse = bench_reuse(m, u->ids, u->np);
        unsigned long long miss0 = (unsigned long long)m->miss;
        double disk0 = m->t_disk, first = 0, steady = 0, nll = 0;
        float *lo = NULL;
        Q36ReqClock clk;
        double a = now_s();
        if (!q36_serve_start(m, &q, &ids, &np, &lo, &clk)) {
            fprintf(stderr, "turn %d was refused: the engine's ERROR line is above\n", k + 1);
            return 1;
        }
        double prefill = now_s() - a;
        if (sf) fprintf(sf, "%d -1 %llu %.4f %.3f\n", k + 1, (unsigned long long)m->miss - miss0, m->t_disk - disk0,
                        prefill * 1e3);
        if (np != u->np || memcmp(ids, u->ids, (size_t)np * sizeof(int))) {
            fprintf(stderr, "turn %d: the engine read other tokens than the bench\n", k + 1);
            return 1;
        }
        uint64_t h = bench_fnv(0xcbf29ce484222325ull, lo, (size_t)V * sizeof(float)), ht = 0xcbf29ce484222325ull;
        int top1 = 0, greedy = n, calls = 0, calls_ok = 0, in_call = 0, call_ok = 0;
        Q36Spec sp;
        q36_spec_begin(m, &sp, ids, np);
        for (int g = 0; g < n; g++) {
            int w = want[g], best = -1, pick = 0;
            double mx = lo[0], z = 0;
            for (int i = 0; i < V; i++) {       /* best: the normal mode's choice, never an end-of-turn token */
                int skip = 0;
                for (int e = 0; e < neos; e++) skip |= i == eos[e];
                if (!skip && (best < 0 || lo[i] > lo[best])) best = i;
                if (lo[i] > lo[pick]) pick = i;   /* pick: serve_sample's at temperature 0 */
                if (lo[i] > mx) mx = lo[i];
            }
            for (int i = 0; i < V; i++) z += exp((double)lo[i] - mx);
            double l = -((double)lo[w] - mx - log(z));
            nll += l;
            top1 += best == w;
            if (greedy == n && pick != w) greedy = g;
            if (w == open_id && open_id >= 0) { in_call = 1; call_ok = 1; }
            if (in_call) call_ok &= pick == w;
            if (in_call && w == close_id) { calls++; calls_ok += call_ok; in_call = 0; }
            if (nf) fprintf(nf, "%d %d %.6f %d\n", k + 1, g, l, best == w);
            h = bench_fnv(h, &w, sizeof w);
            ht = bench_fnv(ht, &w, sizeof w);
            free(lo);
            unsigned long long sm = (unsigned long long)m->miss;
            double sd = m->t_disk;
            double b = now_s();
            lo = q36_spec_step(m, &sp, w, np + g, n - g);   /* the room serve_one leaves with max_tok n + 1 */
            double d = now_s() - b;
            if (sf) fprintf(sf, "%d %d %llu %.4f %.3f\n", k + 1, g, (unsigned long long)m->miss - sm, m->t_disk - sd,
                            d * 1e3);
            if (g == 0) first = d; else steady += d;
            h = bench_fnv(h, lo, (size_t)V * sizeof(float));
        }
        int pick = 0, stop = 0;
        for (int i = 1; i < V; i++) if (lo[i] > lo[pick]) pick = i;
        for (int e = 0; e < neos; e++) stop |= pick == eos[e];
        free(lo);
        q36_spec_end(m, &sp);
        char scope[32];
        snprintf(scope, sizeof scope, "turn %d", k + 1);
        q36_spec_report(&sp, scope);
        q36_spec_free(&sp);
        free(ids);
        fprintf(csv, "%d,%d,%.3f,%.2f,%d,%.1f,%.3f,%llu,%.2f,%016llx,%016llx,%.5f,%.3f,%d,%d,%.2f,%d,%d,%d,%d\n", np,
                np - reuse, prefill, prefill > 0 ? (np - reuse) / prefill : 0.0, n, first * 1e3,
                n > 1 && steady > 0 ? (n - 1) / steady : 0.0, (unsigned long long)m->miss - miss0, m->t_disk - disk0,
                (unsigned long long)h, (unsigned long long)ht, nll / n, (double)top1 / n, reuse, m->kv_len, rss_gb(),
                greedy, calls, calls_ok, stop);
        fflush(csv);
        fprintf(stderr, "[bench] turn %d: prompt %d tokens (%d reused, %d read in %.2f s), answer %d tokens: "
                "first step %.0f ms, then %.2f tok/s; nll %.4f, top1 %.3f, greedy prefix %d of %d, tool calls exact "
                "%d of %d, stop %s; kv_len %d, peak RSS %.2f GB\n", k + 1, np, reuse, np - reuse, prefill, n,
                first * 1e3, n > 1 && steady > 0 ? (n - 1) / steady : 0.0, nll / n, (double)top1 / n, greedy, n,
                calls_ok, calls, stop ? "yes" : "no", m->kv_len, rss_gb());
    }
    if (nf) fclose(nf);
    if (sf) fclose(sf);
    tm_report();
    fprintf(stderr, "[bench] peak RSS %.2f GB\n", rss_gb());
    fclose(csv);
    return 0;
}

static int bench_usage(const char *argv0) {
    fprintf(stderr,
            "usage: %s --model <dir> --text <file> [--lengths 64,512] [--gen 32] [--photo] [--teacher]\n"
            "          [--steps <file>] [--nll <file>] [--words <file>] [--pause <s>] [--tokenizer <file>]\n"
            "       %s --model <dir> --session <file> [--steps <file>] [--nll <file>] [--pause <s>] [--tokens-only]\n",
            argv0, argv0);
    return 2;
}

int main(int argc, char **argv) {
    BenchOpts o = {.lengths = "64,512", .gen = 32};
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i], **s = NULL;
        int *n = NULL;
        if (!strcmp(a, "--photo")) { o.photo = 1; continue; }
        if (!strcmp(a, "--teacher")) { o.teacher = 1; continue; }
        if (!strcmp(a, "--tokens-only")) { o.tokens_only = 1; continue; }
        if (!strcmp(a, "--model")) s = &o.model;
        else if (!strcmp(a, "--tokenizer")) s = &o.tokenizer;
        else if (!strcmp(a, "--text")) s = &o.text;
        else if (!strcmp(a, "--lengths")) s = &o.lengths;
        else if (!strcmp(a, "--session")) s = &o.session;
        else if (!strcmp(a, "--steps")) s = &o.steps;
        else if (!strcmp(a, "--nll")) s = &o.nll;
        else if (!strcmp(a, "--words")) s = &o.words;
        else if (!strcmp(a, "--gen")) n = &o.gen;
        else if (!strcmp(a, "--pause")) n = &o.pause;
        else { fprintf(stderr, "unknown option %s\n", a); return bench_usage(argv[0]); }
        if (++i >= argc) { fprintf(stderr, "%s needs a value\n", a); return bench_usage(argv[0]); }
        if (s) *s = argv[i];
        else {
            char *end;
            long v = strtol(argv[i], &end, 10);
            if (*end || v < 0 || v > 1000000) { fprintf(stderr, "%s %s: a whole number\n", a, argv[i]); return 2; }
            *n = (int)v;
        }
    }
    if (!o.model || !!o.text == !!o.session) return bench_usage(argv[0]);
    if (o.session) {
        if (o.photo || o.words) { fprintf(stderr, "--session takes no --photo or --words\n"); return 2; }
        return bench_session(&o);
    }
    if (o.tokens_only) { fprintf(stderr, "--tokens-only goes with --session\n"); return 2; }
    if (o.nll && !o.teacher) { fprintf(stderr, "--nll needs --teacher: it scores the text's own tokens\n"); return 2; }
    if (o.pause && !o.photo) { fprintf(stderr, "--pause needs --photo (or --session): lengths from empty states "
                                               "are separate prompts\n"); return 2; }

    int lens[16], nl = 0, maxlen = 0, gen = o.gen;
    char spec[256];
    snprintf(spec, sizeof spec, "%s", o.lengths);
    for (char *t = strtok(spec, ","); t; t = strtok(NULL, ",")) {
        if (nl == 16) { fprintf(stderr, "at most 16 lengths\n"); return 2; }
        lens[nl] = atoi(t);
        if (lens[nl] < 1 || (o.photo && nl && lens[nl] <= lens[nl - 1])) {
            fprintf(stderr, "bad length %s (with --photo lengths must grow)\n", t);
            return 2;
        }
        if (lens[nl] > maxlen) maxlen = lens[nl];
        nl++;
    }
    if (!nl) { fprintf(stderr, "no lengths\n"); return 2; }
    {   int rc = bench_context(maxlen + gen, "the longest length and the generated tokens"); if (rc) return rc; }

    {   int rc = bench_open(&o); if (rc) return rc; }
    FILE *f = fopen(o.text, "rb");
    if (!f) { perror(o.text); return 1; }
    size_t cap = (size_t)(maxlen + (o.teacher ? gen : 0)) * 32 + 4096;   /* only the start of the text is needed */
    char *text = malloc(cap + 1);
    size_t got = text ? fread(text, 1, cap, f) : 0;
    fclose(f);
    if (!text) return 1;
    text[got] = 0;
    int *ids = NULL, nids = 0;
    encode_text(text, &ids, &nids);
    free(text);
    if (nids < maxlen + (o.teacher ? gen : 0)) {
        fprintf(stderr, "the text gives %d tokens, %d needed\n", nids, maxlen + (o.teacher ? gen : 0));
        return 1;
    }

#ifndef QWEN36_METAL_GRAPH
    if (o.photo) { fprintf(stderr, "--photo needs the Metal graph\n"); return 2; }
#endif
    double t0 = now_s();
    Model *m = bench_model(&o);
    int eos[4], neos = serve_eos_ids(eos, 4);
#ifdef QWEN36_METAL_GRAPH
    BenchPhoto ph = {0};
    if (o.photo && !m->graph) { fprintf(stderr, "--photo needs the Metal graph\n"); return 2; }
#endif
    fprintf(stderr, "[bench] %s ready in %.2f s; text %d tokens; lengths %s; gen %d; %s\n", bench_engine(m),
            now_s() - t0, nids, o.lengths, gen, o.photo ? "photo: each length continues the previous one"
                                          : "each length from an empty state");

    printf("ctx_tokens,prefill_tokens,prefill_s,prefill_tps,gen_tokens,gen_first_ms,gen_steady_tps,"
           "expert_miss,disk_s,fingerprint,tokens,nll,top1\n");
    fflush(stdout);
    FILE *sf = bench_out(o.steps);
    if (o.photo) bench_fresh(m, maxlen + gen);
    for (int li = 0, prev = 0; li < nl; li++) {
        int N = lens[li], V = m->c.vocab;
        if (o.photo && li > 0) bench_pause(o.pause, "length", N);
        if (!o.photo) { bench_fresh(m, N + gen); prev = 0; }
#ifdef QWEN36_METAL_GRAPH
        else if (li) bench_photo_load(m, &ph);   /* back to the end of the previous prompt */
#endif
        unsigned long long miss0 = (unsigned long long)m->miss;
        double disk0 = m->t_disk;

        double a = now_s(), prefill = 0, first = 0, steady = 0;
        float *lo = step(m, ids + prev, N - prev, prev);
        prefill = now_s() - a;
        if (sf) fprintf(sf, "%d -1 %llu %.4f %.3f\n", N, (unsigned long long)m->miss - miss0, m->t_disk - disk0,
                        prefill * 1e3);
#ifdef QWEN36_METAL_GRAPH
        if (o.photo && li + 1 < nl) bench_photo_save(m, &ph);
#endif
        uint64_t h = bench_fnv(0xcbf29ce484222325ull, lo, (size_t)V * sizeof(float)), ht = 0xcbf29ce484222325ull;
        /* speculative decoding as generate() runs it: q36_spec_step drafts and verifies (on the Metal graph
         * the MTP head, when the model has mtp.safetensors; QWEN36_MTP=0 decodes plainly) */
        Q36Spec sp;
        q36_spec_begin(m, &sp, ids, N);
        double nll = 0;
        int top1 = 0;
        FILE *nf = o.teacher && li == 0 ? bench_out(o.nll) : NULL;
        int *chosen = li == 0 && o.words ? malloc(sizeof(int) * (size_t)(gen > 0 ? gen : 1)) : NULL;
        for (int g = 0; g < gen; g++) {
            int best = -1;
            for (int i = 0; i < V; i++) {
                int skip = 0;
                for (int e = 0; e < neos; e++) skip |= i == eos[e];
                if (!skip && (best < 0 || lo[i] > lo[best])) best = i;
            }
            if (o.teacher) {                    /* the text's token, and how likely the model found it */
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
            unsigned long long sm = (unsigned long long)m->miss;
            double sd = m->t_disk;
            double b = now_s();
            lo = q36_spec_step(m, &sp, best, N + g, gen - 1 - g);
            double d = now_s() - b;
            if (sf) fprintf(sf, "%d %d %llu %.4f %.3f\n", N, g, (unsigned long long)m->miss - sm, m->t_disk - sd,
                            d * 1e3);
            if (g == 0) first = d; else steady += d;
            h = bench_fnv(h, lo, (size_t)V * sizeof(float));
        }
        if (nf) fclose(nf);
        if (chosen) {
            size_t wcap = (size_t)gen * 255 + 1;
            char *words = malloc(wcap);
            FILE *wf = bench_out(o.words);
            if (words) {
                int len = decode_range(chosen, 0, gen, words, (int)wcap);
                fwrite(words, 1, len > 0 ? (size_t)len : 0, wf);
            }
            fclose(wf);
            free(words); free(chosen);
        }
        q36_spec_end(m, &sp);
        char scope[32];
        snprintf(scope, sizeof scope, "%d tokens", N);
        q36_spec_report(&sp, scope);
        q36_spec_free(&sp);
        free(lo);
        printf("%d,%d,%.3f,%.2f,%d,%.1f,%.3f,%llu,%.2f,%016llx,%016llx,%.5f,%.3f\n", N, N - prev, prefill,
               prefill > 0 ? (N - prev) / prefill : 0.0, gen, first * 1e3, gen > 1 && steady > 0 ? (gen - 1) / steady : 0.0,
               (unsigned long long)m->miss - miss0, m->t_disk - disk0, (unsigned long long)h, (unsigned long long)ht,
               o.teacher && gen ? nll / gen : 0.0, o.teacher && gen ? (double)top1 / gen : 0.0);
        fflush(stdout);
        if (o.photo) prev = N;
    }
    if (sf) fclose(sf);
    tm_report();
    fprintf(stderr, "[bench] peak RSS %.2f GB\n", rss_gb());
    return 0;
}
