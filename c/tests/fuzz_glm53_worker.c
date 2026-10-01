/* Structured-mutation fuzz for the GLM-5.3 expert worker's wire parser.
 *
 * The worker is the one part of glm53 that reads bytes from another machine,
 * so it is the part where a parsing slip is a remote memory bug rather than a
 * wrong answer. This drives glm53_worker_serve_one() -- the exact function a
 * worker runs on every accepted connection -- over a socketpair, with a real
 * (tiny) model loaded, under ASan+UBSan.
 *
 * Every case starts from well-formed traffic built from the model's own shape
 * (handshakes and batches of distinct experts), then mutates it: random byte
 * flips, header fields forced to boundary values (0, 1, n_experts,
 * 65536, 0xFFFFFFFF...), duplicated experts, truncation, junk prefixes and
 * concatenated requests. Well-formed, unmutated requests must also get a
 * well-formed reply: magic, version, status 0, and every expert back in
 * request order with its row count.
 *
 * The expert cache is held at two slots per layer so batches span several
 * cache blocks, and slots are reused within one request.
 *
 * Seeded and bounded (FUZZ_ITERS, default 3000). Needs the int4 streaming
 * fixture:
 *   python3 tools/make_glm53_multimodal_tiny.py --output /tmp/glm53_mm
 *   python3 tools/make_glm53_streaming_pair.py --fixture /tmp/glm53_mm --output /tmp/glm53_stream
 *   make fuzz-glm53-worker GLM53_FUZZ_FIXTURE=/tmp/glm53_stream-i4
 * Exits 2 (SKIP) without it, nonzero on a malformed reply to a valid request;
 * sanitizers abort on any memory or UB report. NOT part of TEST_BINS.
 *
 * Built with -DGLM53_LIBFUZZER -fsanitize=fuzzer it is a libFuzzer target
 * instead (fixture from COLI_GLM53_FIXTURE), for open-ended runs. */
#define main glm53_engine_main
#include "../glm53.c"
#undef main

#include <sys/socket.h>
#include <sys/un.h>

static GModel g_model;

static uint64_t rng_state = 0x243F6A8885A308D3ULL;
static uint32_t rnd(void) {
    rng_state ^= rng_state << 13; rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17; return (uint32_t)(rng_state >> 32);
}

/* ---------- a growable byte buffer ---------- */
typedef struct { uint8_t *p; size_t n, cap; } Buf;
static void put(Buf *b, const void *src, size_t n) {
    if (b->n + n > b->cap) {
        b->cap = (b->n + n) * 2 + 64;
        b->p = realloc(b->p, b->cap);
        if (!b->p) { fprintf(stderr, "OOM\n"); exit(1); }
    }
    memcpy(b->p + b->n, src, n);
    b->n += n;
}
static void put_u32(Buf *b, uint32_t v) { v = htonl(v); put(b, &v, 4); }

/* ---------- well-formed traffic ---------- */
static void hello(Buf *b) {
    put(b, GLM53_CLUSTER_MAGIC, 8);
    put_u32(b, GLM53_CLUSTER_VERSION); put_u32(b, GLM53_CLUSTER_HELLO);
    put_u32(b, (uint32_t)g_model.c.hidden); put_u32(b, (uint32_t)g_model.c.moe_inter);
    put_u32(b, 0);
}

/* A batch of `n` distinct experts, each with 1..4 rows of small floats.
 * Returns the layer; eids/rows receive what was sent. */
static int batch(Buf *b, int n, int *eids, int *rows) {
    const Cfg *c = &g_model.c;
    const int layer = c->first_dense + (int)(rnd() % (unsigned)(c->n_layers - c->first_dense));
    int pool[512], np = c->n_experts < 512 ? c->n_experts : 512;
    for (int i = 0; i < np; i++) pool[i] = i;
    for (int i = np - 1; i > 0; i--) { int j = (int)(rnd() % (unsigned)(i + 1)), t = pool[i]; pool[i] = pool[j]; pool[j] = t; }
    if (n > np) n = np;
    put(b, GLM53_CLUSTER_MAGIC, 8);
    put_u32(b, GLM53_CLUSTER_VERSION); put_u32(b, (uint32_t)layer);
    put_u32(b, (uint32_t)c->hidden); put_u32(b, (uint32_t)c->moe_inter);
    put_u32(b, (uint32_t)n);
    for (int j = 0; j < n; j++) {
        eids[j] = pool[j];
        rows[j] = 1 + (int)(rnd() % 4);
        put_u32(b, (uint32_t)eids[j]); put_u32(b, (uint32_t)rows[j]);
        for (int r = 0; r < rows[j] * c->hidden; r++) {
            float v = ((float)(rnd() % 2001) - 1000.0f) / 1000.0f;
            put(b, &v, 4);
        }
    }
    return layer;
}

/* ---------- one exchange over a socketpair ---------- */
typedef struct { int fd; const uint8_t *data; size_t n; } Feed;
typedef struct { int fd; Buf out; } Drain;

static void *feed_run(void *arg) {
    Feed *f = arg;
    size_t at = 0;
    while (at < f->n) {
        ssize_t w = send(f->fd, f->data + at, f->n - at, MSG_NOSIGNAL);
        if (w <= 0) break;              /* the worker dropped the connection */
        at += (size_t)w;
    }
    shutdown(f->fd, SHUT_WR);
    return NULL;
}

static void *drain_run(void *arg) {
    Drain *d = arg;
    uint8_t buf[65536];
    ssize_t r;
    while ((r = recv(d->fd, buf, sizeof(buf), 0)) > 0) put(&d->out, buf, (size_t)r);
    return NULL;
}

/* Feeds `data` to a fresh worker connection and returns everything the
 * worker wrote back (caller frees .p). */
static Buf exchange(const uint8_t *data, size_t n) {
    int sv[2];
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sv)) { perror("socketpair"); exit(1); }
    Feed f = { sv[0], data, n };
    Drain d = { dup(sv[0]), { 0 } };
    pthread_t tf, td;
    pthread_create(&tf, NULL, feed_run, &f);
    pthread_create(&td, NULL, drain_run, &d);
    while (!glm53_worker_serve_one(&g_model, sv[1], 7)) {}
    close(sv[1]);                       /* the drain sees EOF */
    pthread_join(tf, NULL);
    pthread_join(td, NULL);
    close(d.fd);
    close(sv[0]);
    return d.out;
}

/* ---------- reply check for well-formed requests ---------- */
static uint32_t get_u32(const Buf *b, size_t *at, int *ok) {
    uint32_t v = 0;
    if (*at + 4 > b->n) { *ok = 0; return 0; }
    memcpy(&v, b->p + *at, 4);
    *at += 4;
    return ntohl(v);
}

static int reply_ok(const Buf *b, int n, const int *eids, const int *rows) {
    size_t at = 0;
    int ok = 1;
    if (b->n < 8 || memcmp(b->p, GLM53_CLUSTER_MAGIC, 8)) return 0;
    at = 8;
    if (get_u32(b, &at, &ok) != GLM53_CLUSTER_VERSION || get_u32(b, &at, &ok) != 0 ||
        get_u32(b, &at, &ok) != (uint32_t)n || !ok)
        return 0;
    for (int j = 0; j < n; j++) {
        if (get_u32(b, &at, &ok) != (uint32_t)eids[j] ||
            get_u32(b, &at, &ok) != (uint32_t)rows[j] || !ok)
            return 0;
        const size_t bytes = (size_t)rows[j] * g_model.c.hidden * sizeof(float);
        if (at + bytes > b->n) return 0;
        for (size_t k = 0; k < bytes / 4; k++) {
            float v;
            memcpy(&v, b->p + at + 4 * k, 4);
            if (!isfinite(v)) return 0;
        }
        at += bytes;
    }
    return at == b->n;
}

/* ---------- mutations ---------- */
static const uint32_t edge[] = { 0, 1, 2, 7, 64, 255, 256, 65535, 65536, 65537,
                                 0x7FFFFFFF, 0x80000000, 0xFFFFFFFE, 0xFFFFFFFF };

static void mutate(Buf *b) {
    const int kind = (int)(rnd() % 6);
    if (!b->n) return;
    switch (kind) {
    case 0:                                     /* flip a few bytes */
        for (int k = 1 + (int)(rnd() % 4); k > 0; k--) b->p[rnd() % b->n] ^= (uint8_t)(1u << (rnd() % 8));
        break;
    case 1: {                                   /* a header u32 to an edge value */
        size_t at = 8 + 4 * (rnd() % 7);        /* version..first item */
        if (at + 4 <= b->n) {
            uint32_t v = edge[rnd() % (sizeof(edge) / sizeof(edge[0]))];
            if (rnd() % 3 == 0) v = (uint32_t)g_model.c.n_experts - (rnd() % 2);
            v = htonl(v);
            memcpy(b->p + at, &v, 4);
        }
        break;
    }
    case 2:                                     /* truncate */
        b->n = rnd() % b->n;
        break;
    case 3: {                                   /* junk before a valid message */
        Buf j = { 0 };
        for (int k = (int)(rnd() % 24); k > 0; k--) { uint8_t x = (uint8_t)rnd(); put(&j, &x, 1); }
        put(&j, b->p, b->n);
        free(b->p); *b = j;
        break;
    }
    case 4: {                                   /* first item's expert repeated in the second */
        size_t first = 8 + 5 * 4, second;
        if (first + 8 > b->n) break;
        uint32_t nr; memcpy(&nr, b->p + first + 4, 4);
        second = first + 8 + (size_t)ntohl(nr) * g_model.c.hidden * 4;
        if (second + 4 <= b->n) memcpy(b->p + second, b->p + first, 4);
        break;
    }
    default:                                    /* a random byte anywhere */
        b->p[rnd() % b->n] = (uint8_t)rnd();
        break;
    }
}

static int setup(const char *fixture) {
    g_cap_override = 2;             /* two slots: batches span several cache blocks */
    return glm53_worker_open(&g_model, fixture);
}

#ifdef GLM53_LIBFUZZER
int LLVMFuzzerInitialize(int *argc, char ***argv) {
    (void)argc; (void)argv;
    const char *fixture = getenv("COLI_GLM53_FIXTURE");
    if (!fixture || setup(fixture)) { fprintf(stderr, "set COLI_GLM53_FIXTURE to the int4 fixture\n"); exit(2); }
    return 0;
}
int LLVMFuzzerTestOneInput(const uint8_t *data, size_t n) {
    Buf out = exchange(data, n);
    free(out.p);
    return 0;
}
#else
int main(int argc, char **argv) {
    const char *fixture = argc > 1 ? argv[1] : getenv("COLI_GLM53_FIXTURE");
    char probe[4096];
    snprintf(probe, sizeof(probe), "%s/config.json", fixture ? fixture : "");
    if (!fixture || !*fixture || access(probe, R_OK)) {
        printf("SKIP: no int4 GLM-5.3 fixture (argument or COLI_GLM53_FIXTURE)\n");
        return 2;
    }
    if (setup(fixture)) return 1;
    const char *it = getenv("FUZZ_ITERS");
    const int iters = it ? atoi(it) : 3000;
    int valid = 0, mutated = 0, bad_replies = 0;
    for (int i = 0; i < iters; i++) {
        Buf msg = { 0 };
        int eids[512], rows[512], n = 0;
        const int shape = (int)(rnd() % 4);
        if (shape == 0) hello(&msg);
        else {
            n = 1 + (int)(rnd() % (unsigned)(g_model.c.n_experts < 6 ? g_model.c.n_experts : 6));
            batch(&msg, n, eids, rows);
            if (shape == 3) { int e2[512], r2[512]; batch(&msg, 1 + (int)(rnd() % 2), e2, r2); }
        }
        const int mutate_it = rnd() % 4 != 0;       /* one in four stays well-formed */
        if (mutate_it) { mutate(&msg); mutated++; }
        Buf out = exchange(msg.p, msg.n);
        if (!mutate_it && shape != 0 && shape != 3) {
            valid++;
            if (!reply_ok(&out, n, eids, rows)) {
                bad_replies++;
                fprintf(stderr, "case %d: malformed reply to a well-formed %d-expert request\n", i, n);
            }
        }
        free(out.p);
        free(msg.p);
    }
    printf("fuzz_glm53_worker: %d cases (%d mutated, %d well-formed checked), %d bad replies\n",
           iters, mutated, valid, bad_replies);
    return bad_replies ? 1 : 0;
}
#endif
