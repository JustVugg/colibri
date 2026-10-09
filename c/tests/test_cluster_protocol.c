/* COLIEX01 cluster wire contract: network-order headers, raw f32 activations
 * (v1) or the v2 header's activation format (f32 or q8 blocks of 32), and one
 * request/response over a socketpair. No model fixture required. */
#define main coli_engine_main_unused
#include "../colibri.c"
#undef main

#include <assert.h>

#if defined(__APPLE__) || defined(__linux__) || defined(__FreeBSD__)
#include <sys/wait.h>

static void test_disconnected_peer_is_an_io_error(void)
{
    /* Use a child with the default signal disposition: a caller must receive
     * the error, even when its launcher has not globally ignored SIGPIPE. */
    pid_t child = fork();
    assert(child >= 0);
    if (child == 0) {
        int sockets[2];
        assert(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0);
        close(sockets[1]);
        signal(SIGPIPE, SIG_DFL);
        char byte = 'x';
        int result = cluster_io(sockets[0], &byte, 1, 1);
        close(sockets[0]);
        assert(signal(SIGPIPE, SIG_DFL) == SIG_DFL);
        _exit(result == -1 ? 0 : 2);
    }
    int status;
    while (waitpid(child, &status, 0) < 0) assert(errno == EINTR);
    assert(WIFEXITED(status) && WEXITSTATUS(status) == 0);
}

typedef struct { int fd; int failed; } ClusterProtocolArgs;

static void *cluster_protocol_worker(void *opaque)
{
    ClusterProtocolArgs *args = opaque;
    int fd = args->fd;
    char magic[8];
    uint32_t version, layer, D, I, n, eid, nr;
    float input[6], output[6];

    /* Request: magic(8) version layer D I n, then eid nr and nr*D inputs. */
    if (cluster_io(fd, magic, sizeof(magic), 0) || memcmp(magic, COLI_CLUSTER_MAGIC, 8) ||
        cluster_u32(fd, &version, 0) || version != COLI_CLUSTER_VERSION ||
        cluster_u32(fd, &layer, 0) || layer != 7 ||
        cluster_u32(fd, &D, 0) || D != 3 ||
        cluster_u32(fd, &I, 0) || I != 5 ||
        cluster_u32(fd, &n, 0) || n != 1 ||
        cluster_u32(fd, &eid, 0) || eid != 42 ||
        cluster_u32(fd, &nr, 0) || nr != 2 ||
        cluster_io(fd, input, sizeof(input), 0)) {
        args->failed = 1;
        return NULL;
    }
    for (int i = 0; i < 6; i++) output[i] = input[i] * 2.0f;

    /* Response: magic(8) version status(0) n, then eid nr and nr*D outputs. */
    version = COLI_CLUSTER_VERSION;
    if (cluster_io(fd, (void *)COLI_CLUSTER_MAGIC, 8, 1) ||
        cluster_u32(fd, &version, 1) ||
        cluster_u32(fd, &(uint32_t){0}, 1) ||
        cluster_u32(fd, &n, 1) ||
        cluster_u32(fd, &eid, 1) ||
        cluster_u32(fd, &nr, 1) ||
        cluster_io(fd, output, sizeof(output), 1))
        args->failed = 1;
    return NULL;
}

static void test_wire_round_trip(void)
{
    int sockets[2];
    assert(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0);
    ClusterProtocolArgs args = {sockets[1], 0};
    pthread_t thread;
    assert(pthread_create(&thread, NULL, cluster_protocol_worker, &args) == 0);

    uint32_t value;
    float input[6] = {1.0f, -2.0f, 0.5f, 3.0f, -4.0f, 0.25f}, output[6] = {0};

    assert(cluster_io(sockets[0], (void *)COLI_CLUSTER_MAGIC, 8, 1) == 0);
    value = COLI_CLUSTER_VERSION; assert(cluster_u32(sockets[0], &value, 1) == 0);
    value = 7;  assert(cluster_u32(sockets[0], &value, 1) == 0); /* layer */
    value = 3;  assert(cluster_u32(sockets[0], &value, 1) == 0); /* D */
    value = 5;  assert(cluster_u32(sockets[0], &value, 1) == 0); /* moe_inter */
    value = 1;  assert(cluster_u32(sockets[0], &value, 1) == 0); /* n */
    value = 42; assert(cluster_u32(sockets[0], &value, 1) == 0); /* eid */
    value = 2;  assert(cluster_u32(sockets[0], &value, 1) == 0); /* nr */
    assert(cluster_io(sockets[0], input, sizeof(input), 1) == 0);

    char magic[8];
    assert(cluster_io(sockets[0], magic, 8, 0) == 0);
    assert(memcmp(magic, COLI_CLUSTER_MAGIC, 8) == 0);
    value = 0; assert(cluster_u32(sockets[0], &value, 0) == 0 && value == COLI_CLUSTER_VERSION);
    value = 1; assert(cluster_u32(sockets[0], &value, 0) == 0 && value == 0);  /* status */
    value = 0; assert(cluster_u32(sockets[0], &value, 0) == 0 && value == 1);  /* n */
    value = 0; assert(cluster_u32(sockets[0], &value, 0) == 0 && value == 42); /* eid */
    value = 0; assert(cluster_u32(sockets[0], &value, 0) == 0 && value == 2);  /* nr */
    assert(cluster_io(sockets[0], output, sizeof(output), 0) == 0);
    assert(output[0] == 2.0f && output[1] == -4.0f && output[2] == 1.0f &&
           output[3] == 6.0f && output[4] == -8.0f && output[5] == 0.5f);

    assert(pthread_join(thread, NULL) == 0);
    assert(args.failed == 0);
    close(sockets[0]);
    close(sockets[1]);
}

/* Deterministic rows (a fixed LCG) so a failure reproduces. */
static void fill_rows(float *x, size_t n, uint32_t seed)
{
    for (size_t i = 0; i < n; i++) {
        seed = seed * 1664525u + 1013904223u;
        x[i] = ((float)(seed >> 8) / 16777216.0f - 0.5f) * 8.0f;   /* [-4, 4) */
    }
}

/* q8 codec: every block of 32 on the wire is qrow_i8's scale and int8 bytes
 * (the engine's own activation rounding), decoding lands within half a block
 * scale of the source, a partial last block (D not a multiple of 32) carries
 * only its D%32 values, and an all-zero row decodes to exact zeros. */
static void test_q8_codec_matches_qrow_i8(void)
{
    static const uint32_t dims[] = {256, 70, 1};
    for (size_t t = 0; t < sizeof(dims) / sizeof(dims[0]); t++) {
        uint32_t D = dims[t], nr = 3, nb = (D + COLI_Q8_BLOCK - 1) / COLI_Q8_BLOCK;
        size_t row_bytes = nb * sizeof(float) + D;
        assert(cluster_act_bytes(COLI_ACT_Q8, nr, D) == nr * row_bytes);
        assert(cluster_act_bytes(COLI_ACT_F32, nr, D) == (size_t)nr * D * sizeof(float));

        float *x = malloc((size_t)nr * D * sizeof(float)), *y = malloc((size_t)nr * D * sizeof(float));
        uint8_t *wire = malloc(nr * row_bytes);
        fill_rows(x, (size_t)nr * D, 7u + D);
        for (uint32_t i = 0; i < D; i++) x[D + i] = 0.0f;              /* row 1: all zeros */
        cluster_q8_encode(x, nr, D, wire);

        for (uint32_t r = 0; r < nr; r++) {
            const uint8_t *row = wire + r * row_bytes;
            for (uint32_t b = 0; b < nb; b++) {
                uint32_t n = D - b * COLI_Q8_BLOCK; if (n > COLI_Q8_BLOCK) n = COLI_Q8_BLOCK;
                int8_t q[COLI_Q8_BLOCK]; float wire_scale;
                float s = qrow_i8(x + r * D + b * COLI_Q8_BLOCK, q, (int)n);
                memcpy(&wire_scale, row + b * sizeof(float), sizeof(wire_scale));
                assert(wire_scale == s);                                              /* exact scale bytes */
                assert(memcmp(row + nb * sizeof(float) + b * COLI_Q8_BLOCK, q, n) == 0); /* exact int8 bytes */
            }
        }
        cluster_q8_decode(wire, nr, D, y);
        for (uint32_t r = 0; r < nr; r++)
            for (uint32_t b = 0; b < nb; b++) {
                float s; memcpy(&s, wire + r * row_bytes + b * sizeof(float), sizeof(s));
                uint32_t n = D - b * COLI_Q8_BLOCK; if (n > COLI_Q8_BLOCK) n = COLI_Q8_BLOCK;
                for (uint32_t i = 0; i < n; i++) {
                    size_t k = (size_t)r * D + b * COLI_Q8_BLOCK + i;
                    assert(fabsf(y[k] - x[k]) <= 0.5f * s * (1.0f + 1e-6f));
                }
            }
        for (uint32_t i = 0; i < D; i++) assert(y[D + i] == 0.0f);
        free(x); free(y); free(wire);
    }
}

/* A v2 header whose act is f32 carries v1's raw rows: no scales, no int8. */
static void test_v2_f32_rows_are_raw(void)
{
    int sockets[2];
    assert(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0);
    float input[6] = {1.0f, -2.0f, 0.5f, 3.0f, -4.0f, 0.25f}, raw[6] = {0}, back[6] = {0};
    assert(cluster_act_bytes(COLI_ACT_F32, 2, 3) == sizeof(raw));
    assert(cluster_act_send(sockets[0], COLI_ACT_F32, input, 2, 3) == 0);
    assert(cluster_io(sockets[1], raw, sizeof(raw), 0) == 0);
    assert(memcmp(raw, input, sizeof(raw)) == 0);
    assert(cluster_io(sockets[1], input, sizeof(input), 1) == 0);
    assert(cluster_act_recv(sockets[0], COLI_ACT_F32, back, 2, 3) == 0);
    assert(memcmp(back, input, sizeof(back)) == 0);
    close(sockets[0]);
    close(sockets[1]);
}

static const float q8_sent[6] = {1.0f, -2.0f, 0.5f, 3.0f, -4.0f, 0.25f};

static void *cluster_protocol_worker_q8(void *opaque)
{
    ClusterProtocolArgs *args = opaque;
    int fd = args->fd;
    char magic[8];
    uint32_t version, act, layer, D, I, n, eid, nr;
    float input[6], output[6];
    uint8_t wire[14], expect[14];   /* 2 rows x (1 scale + 3 int8): 14 bytes, not 24 */

    /* v2 request: magic(8) version act layer D I n, then eid nr and nr rows in act. */
    if (cluster_io(fd, magic, sizeof(magic), 0) || memcmp(magic, COLI_CLUSTER_MAGIC, 8) ||
        cluster_u32(fd, &version, 0) || version != COLI_CLUSTER_VERSION_ACT ||
        cluster_u32(fd, &act, 0) || act != COLI_ACT_Q8 ||
        cluster_u32(fd, &layer, 0) || layer != 7 ||
        cluster_u32(fd, &D, 0) || D != 3 ||
        cluster_u32(fd, &I, 0) || I != 5 ||
        cluster_u32(fd, &n, 0) || n != 1 ||
        cluster_u32(fd, &eid, 0) || eid != 42 ||
        cluster_u32(fd, &nr, 0) || nr != 2 ||
        cluster_act_bytes(act, nr, D) != sizeof(wire) ||
        cluster_io(fd, wire, sizeof(wire), 0)) {
        args->failed = 1;
        return NULL;
    }
    /* The bytes on the wire are exactly the q8 encoding of the sender's rows. */
    cluster_q8_encode(q8_sent, nr, D, expect);
    if (memcmp(wire, expect, sizeof(wire))) { args->failed = 1; return NULL; }
    cluster_q8_decode(wire, nr, D, input);
    for (int i = 0; i < 6; i++) output[i] = input[i] * 2.0f;

    /* v2 response: magic(8) version act status(0) n, then eid nr and nr rows in act. */
    version = COLI_CLUSTER_VERSION_ACT;
    if (cluster_io(fd, (void *)COLI_CLUSTER_MAGIC, 8, 1) ||
        cluster_u32(fd, &version, 1) ||
        cluster_u32(fd, &act, 1) ||
        cluster_u32(fd, &(uint32_t){0}, 1) ||
        cluster_u32(fd, &n, 1) ||
        cluster_u32(fd, &eid, 1) ||
        cluster_u32(fd, &nr, 1) ||
        cluster_act_send(fd, act, output, nr, D))
        args->failed = 1;
    return NULL;
}

static void test_wire_round_trip_q8(void)
{
    int sockets[2];
    assert(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0);
    ClusterProtocolArgs args = {sockets[1], 0};
    pthread_t thread;
    assert(pthread_create(&thread, NULL, cluster_protocol_worker_q8, &args) == 0);

    uint32_t value;
    float output[6] = {0};

    assert(cluster_io(sockets[0], (void *)COLI_CLUSTER_MAGIC, 8, 1) == 0);
    value = COLI_CLUSTER_VERSION_ACT; assert(cluster_u32(sockets[0], &value, 1) == 0);
    value = COLI_ACT_Q8; assert(cluster_u32(sockets[0], &value, 1) == 0); /* act */
    value = 7;  assert(cluster_u32(sockets[0], &value, 1) == 0); /* layer */
    value = 3;  assert(cluster_u32(sockets[0], &value, 1) == 0); /* D */
    value = 5;  assert(cluster_u32(sockets[0], &value, 1) == 0); /* moe_inter */
    value = 1;  assert(cluster_u32(sockets[0], &value, 1) == 0); /* n */
    value = 42; assert(cluster_u32(sockets[0], &value, 1) == 0); /* eid */
    value = 2;  assert(cluster_u32(sockets[0], &value, 1) == 0); /* nr */
    assert(cluster_act_send(sockets[0], COLI_ACT_Q8, q8_sent, 2, 3) == 0);

    char magic[8];
    assert(cluster_io(sockets[0], magic, 8, 0) == 0);
    assert(memcmp(magic, COLI_CLUSTER_MAGIC, 8) == 0);
    value = 0; assert(cluster_u32(sockets[0], &value, 0) == 0 && value == COLI_CLUSTER_VERSION_ACT);
    value = 0; assert(cluster_u32(sockets[0], &value, 0) == 0 && value == COLI_ACT_Q8); /* act */
    value = 1; assert(cluster_u32(sockets[0], &value, 0) == 0 && value == 0);  /* status */
    value = 0; assert(cluster_u32(sockets[0], &value, 0) == 0 && value == 1);  /* n */
    value = 0; assert(cluster_u32(sockets[0], &value, 0) == 0 && value == 42); /* eid */
    value = 0; assert(cluster_u32(sockets[0], &value, 0) == 0 && value == 2);  /* nr */
    assert(cluster_act_recv(sockets[0], COLI_ACT_Q8, output, 2, 3) == 0);

    /* 2x the q8-decoded input, itself q8 on the way back. A row's block scale
     * is amax/127 (in: amax, out: 2*amax), so each value lands within
     * amax/127 + amax/127 of exact; and it really is lossy, not raw f32. */
    static const float expected[6] = {2.0f, -4.0f, 1.0f, 6.0f, -8.0f, 0.5f}, amax[2] = {2.0f, 4.0f};
    for (int i = 0; i < 6; i++) assert(fabsf(output[i] - expected[i]) <= 2.0f * amax[i / 3] / 127.0f * (1.0f + 1e-5f));
    assert(output[0] != 2.0f);

    assert(pthread_join(thread, NULL) == 0);
    assert(args.failed == 0);
    close(sockets[0]);
    close(sockets[1]);
}

/* ---- v2: the hello and the shared-expert item ---- */

/* A v1 worker in miniature: the v1 engine's header check closed the connection
 * on any version but 1, silently. This one does the same, and hands back the
 * version the hello carried. */
static void *fake_v1_worker(void *opaque)
{
    int fd = *(int *)opaque;
    char magic[8];
    uint32_t version = 0;
    if (cluster_io(fd, magic, 8, 0) == 0) cluster_u32(fd, &version, 0);
    close(fd);
    return (void *)(uintptr_t)version;
}

/* A v2 worker answering the hello: an empty v2 request gets an empty response
 * in the same act. */
static void *fake_v2_hello_worker(void *opaque)
{
    int fd = *(int *)opaque;
    char magic[8];
    uint32_t v, act, layer, D, I, n, zero = 0;
    if (cluster_io(fd, magic, 8, 0) || memcmp(magic, COLI_CLUSTER_MAGIC, 8) ||
        cluster_u32(fd, &v, 0) || v != COLI_CLUSTER_VERSION_ACT ||
        cluster_u32(fd, &act, 0) || act != COLI_ACT_F32 ||
        cluster_u32(fd, &layer, 0) || cluster_u32(fd, &D, 0) || cluster_u32(fd, &I, 0) ||
        cluster_u32(fd, &n, 0) || n != 0)
        return (void *)1;
    v = COLI_CLUSTER_VERSION_ACT;
    if (cluster_io(fd, (void *)COLI_CLUSTER_MAGIC, 8, 1) || cluster_u32(fd, &v, 1) ||
        cluster_u32(fd, &act, 1) || cluster_u32(fd, &zero, 1) || cluster_u32(fd, &zero, 1))
        return (void *)1;
    return NULL;
}

/* stderr into an unlinked temp file for the duration of one call, so the test
 * can read what the engine said about the worker it refused. */
static int stderr_capture_begin(int *saved)
{
    char path[] = "/tmp/coli_cluster_stderr_XXXXXX";
    int fd = mkstemp(path);
    assert(fd >= 0);
    unlink(path);
    fflush(stderr);
    *saved = dup(2);
    assert(*saved >= 0 && dup2(fd, 2) == 2);
    return fd;
}
static void stderr_capture_end(int saved, int fd, char *buf, size_t cap)
{
    fflush(stderr);
    assert(dup2(saved, 2) == 2);
    close(saved);
    assert(lseek(fd, 0, SEEK_SET) == 0);
    ssize_t n = read(fd, buf, cap - 1);
    buf[n > 0 ? n : 0] = 0;
    close(fd);
}

static void test_hello_refuses_a_v1_worker_by_name(void)
{
    int sockets[2];
    assert(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0);
    pthread_t thread;
    assert(pthread_create(&thread, NULL, fake_v1_worker, &sockets[1]) == 0);
    ClusterWorker w = {sockets[0], "mac-b", 9101, 0};
    int saved, capture = stderr_capture_begin(&saved);
    int rc = cluster_hello(&w);
    char said[1024];
    stderr_capture_end(saved, capture, said, sizeof(said));
    void *carried;
    assert(pthread_join(thread, &carried) == 0);
    assert((uint32_t)(uintptr_t)carried == COLI_CLUSTER_VERSION_ACT);  /* the hello was v2 */
    assert(rc == -1);
    assert(strstr(said, "mac-b:9101") != NULL);   /* refused by name */
    assert(strstr(said, "(v1)") != NULL);         /* and by version */
    close(sockets[0]);
}

static void test_hello_accepts_a_v2_worker(void)
{
    int sockets[2];
    assert(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0);
    pthread_t thread;
    assert(pthread_create(&thread, NULL, fake_v2_hello_worker, &sockets[1]) == 0);
    ClusterWorker w = {sockets[0], "mac-a", 9100, 0};
    assert(cluster_hello(&w) == 0);
    void *failed;
    assert(pthread_join(thread, &failed) == 0 && failed == NULL);
    close(sockets[0]);
    close(sockets[1]);
}

/* A worker for one request of layer 7 (D=3, I=5) in v1 or v2 (act f32): every
 * routed item comes back as input*2, the shared one as input*10, and what it
 * saw on the wire -- version, item ids and row counts -- is kept for the test. */
typedef struct { int fd, failed; uint32_t ver, n, eid[4], nr[4]; } FakeV2Worker;
static void *fake_v2_worker(void *opaque)
{
    FakeV2Worker *w = opaque;
    int fd = w->fd;
    char magic[8];
    uint32_t act = COLI_ACT_F32, layer, D, I, zero = 0;
    float *rows[4] = {0};
    if (cluster_io(fd, magic, 8, 0) || memcmp(magic, COLI_CLUSTER_MAGIC, 8) || cluster_u32(fd, &w->ver, 0) ||
        (w->ver != COLI_CLUSTER_VERSION && w->ver != COLI_CLUSTER_VERSION_ACT) ||
        (w->ver == COLI_CLUSTER_VERSION_ACT && (cluster_u32(fd, &act, 0) || act != COLI_ACT_F32)) ||
        cluster_u32(fd, &layer, 0) || layer != 7 || cluster_u32(fd, &D, 0) || D != 3 ||
        cluster_u32(fd, &I, 0) || I != 5 || cluster_u32(fd, &w->n, 0) || w->n > 4) {
        w->failed = 1;
        return NULL;
    }
    for (uint32_t j = 0; j < w->n; j++) {
        if (cluster_u32(fd, &w->eid[j], 0) || cluster_u32(fd, &w->nr[j], 0) || w->nr[j] > 8) { w->failed = 1; break; }
        rows[j] = calloc((size_t)w->nr[j] * D, sizeof(float));
        if (cluster_io(fd, rows[j], (size_t)w->nr[j] * D * sizeof(float), 0)) { w->failed = 1; break; }
        float k = w->eid[j] == COLI_CLUSTER_EID_SHARED ? 10.0f : 2.0f;
        for (uint32_t z = 0; z < w->nr[j] * D; z++) rows[j][z] *= k;
    }
    uint32_t v = w->ver;
    if (!w->failed &&
        (cluster_io(fd, (void *)COLI_CLUSTER_MAGIC, 8, 1) || cluster_u32(fd, &v, 1) ||
         (w->ver == COLI_CLUSTER_VERSION_ACT && cluster_u32(fd, &act, 1)) ||
         cluster_u32(fd, &zero, 1) || cluster_u32(fd, &w->n, 1)))
        w->failed = 1;
    for (uint32_t j = 0; j < w->n && !w->failed; j++)
        if (cluster_u32(fd, &w->eid[j], 1) || cluster_u32(fd, &w->nr[j], 1) ||
            cluster_io(fd, rows[j], (size_t)w->nr[j] * D * sizeof(float), 1))
            w->failed = 1;
    for (uint32_t j = 0; j < 4; j++) free(rows[j]);
    return NULL;
}

/* Drives the real cluster_moe_batch over a socketpair: two rows, both routed
 * to expert 42 of layer 7, through one worker, with f32 activations. With a
 * shared buffer the request is v2 and carries a second item -- the shared
 * expert over every row -- whose rows come back unweighted into that buffer
 * while the routed rows are weighted into out; without the buffer the request
 * keeps v1's bytes exactly, as the default must. */
static void test_shared_expert_rides_the_routed_batch(void)
{
    static Model m;
    memset(&m, 0, sizeof(m));
    m.c.hidden = 3;
    m.c.moe_inter = 5;
    float x[6] = {1.0f, -2.0f, 0.5f, 3.0f, -4.0f, 0.25f};
    int idxs[2] = {42, 42}, keff[2] = {1, 1}, uniq[1] = {42};
    float ws[2] = {0.5f, 0.25f};
    assert(g_cluster_act == COLI_ACT_F32);
    for (int pass = 0; pass < 2; pass++) {
        int sockets[2];
        assert(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0);
        g_cluster_n = 1;
        g_cluster_workers[0].fd = sockets[0];
        snprintf(g_cluster_workers[0].host, sizeof(g_cluster_workers[0].host), "mac-a");
        g_cluster_workers[0].port = 9100;
        FakeV2Worker w = {sockets[1], 0, 0, 0, {0}, {0}};
        pthread_t thread;
        assert(pthread_create(&thread, NULL, fake_v2_worker, &w) == 0);
        float out[6] = {0}, sh[6] = {0};
        int delivered = cluster_moe_batch(&m, 7, x, 2, out, idxs, ws, keff, 1, uniq, 0, 1, pass ? NULL : sh);
        assert(pthread_join(thread, NULL) == 0);
        assert(w.failed == 0);
        assert(w.eid[0] == 42 && w.nr[0] == 2);
        for (int r = 0; r < 2; r++)
            for (int d = 0; d < 3; d++)
                assert(out[r * 3 + d] == ws[r] * 2.0f * x[r * 3 + d]);
        if (pass == 0) {
            assert(w.ver == COLI_CLUSTER_VERSION_ACT);
            assert(delivered == 1 && w.n == 2);
            assert(w.eid[1] == COLI_CLUSTER_EID_SHARED && w.nr[1] == 2);
            for (int z = 0; z < 6; z++) assert(sh[z] == 10.0f * x[z]);
        } else {
            assert(w.ver == COLI_CLUSTER_VERSION);   /* no shared expert, f32: v1 on the wire */
            assert(delivered == 0 && w.n == 1);
        }
        g_cluster_n = 0;
        close(sockets[0]);
        close(sockets[1]);
    }
}

/* ---- after a failure the link is dead until reconnect (ds4 f0962e3) ---- */

/* A worker for one v1 request of layer 7 (D=3, I=5, one item: expert 42 over
 * two rows) that answers it wrongly in one of two ways, then waits for a
 * second request and counts whatever arrives:
 *  FAIL_STALE: the reply's header claims two items for a one-item request (a
 *              one-sided mix-up), nothing follows it, and a well-formed reply
 *              to the same request is sent right behind it -- 52 bytes a
 *              coordinator that kept reading would take as the answer to its
 *              NEXT request;
 *  FAIL_EOF:   the reply stops after the first of its two rows and the write
 *              side is shut, so the coordinator sees EOF mid-reply while the
 *              read side stays open. */
enum { FAIL_STALE = 0, FAIL_EOF = 1 };
typedef struct { int fd, mode, failed; ssize_t bytes_after; } FailingWorker;

static int failing_worker_read_request(int fd, float *input /* 6 */)
{
    char magic[8];
    uint32_t version, layer, D, I, n, eid, nr;
    return cluster_io(fd, magic, 8, 0) || memcmp(magic, COLI_CLUSTER_MAGIC, 8) ||
           cluster_u32(fd, &version, 0) || version != COLI_CLUSTER_VERSION ||
           cluster_u32(fd, &layer, 0) || layer != 7 ||
           cluster_u32(fd, &D, 0) || D != 3 || cluster_u32(fd, &I, 0) || I != 5 ||
           cluster_u32(fd, &n, 0) || n != 1 ||
           cluster_u32(fd, &eid, 0) || eid != 42 || cluster_u32(fd, &nr, 0) || nr != 2 ||
           cluster_io(fd, input, 6 * sizeof(float), 0);
}

static int failing_worker_write_reply(int fd, uint32_t n, const float *output, size_t floats)
{
    uint32_t version = COLI_CLUSTER_VERSION, zero = 0, eid = 42, nr = 2;
    if (cluster_io(fd, (void *)COLI_CLUSTER_MAGIC, 8, 1) || cluster_u32(fd, &version, 1) ||
        cluster_u32(fd, &zero, 1) || cluster_u32(fd, &n, 1))
        return -1;
    if (floats == 0) return 0;            /* a header and nothing behind it */
    return cluster_u32(fd, &eid, 1) || cluster_u32(fd, &nr, 1) ||
           cluster_io(fd, (void *)output, floats * sizeof(float), 1);
}

static void *failing_worker(void *opaque)
{
    FailingWorker *w = opaque;
    float input[6], output[6];
    if (failing_worker_read_request(w->fd, input)) { w->failed = 1; return NULL; }
    for (int i = 0; i < 6; i++) output[i] = input[i] * 2.0f;
    if (w->mode == FAIL_STALE) {
        if (failing_worker_write_reply(w->fd, 2, NULL, 0) ||
            failing_worker_write_reply(w->fd, 1, output, 6)) { w->failed = 1; return NULL; }
    } else {
        if (failing_worker_write_reply(w->fd, 1, output, 3)) { w->failed = 1; return NULL; }
        shutdown(w->fd, SHUT_WR);
    }
    /* Anything the coordinator sends now is a request it must not have made.
     * The test closing its end is what ends this wait. */
    char buf[256];
    ssize_t got;
    while ((got = recv(w->fd, buf, sizeof(buf), 0)) > 0) w->bytes_after += got;
    return NULL;
}

static void test_failed_link_refuses_later_exchanges_without_io(void)
{
    static Model m;
    memset(&m, 0, sizeof(m));
    m.c.hidden = 3;
    m.c.moe_inter = 5;
    float x[6] = {1.0f, -2.0f, 0.5f, 3.0f, -4.0f, 0.25f};
    int idxs[2] = {42, 42}, keff[2] = {1, 1};
    float ws[2] = {0.5f, 0.25f};
    assert(g_cluster_act == COLI_ACT_F32);
    for (int mode = FAIL_STALE; mode <= FAIL_EOF; mode++) {
        int sockets[2];
        assert(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0);
        ClusterWorker w = {sockets[0], "mac-a", 9100, 0};
        FailingWorker fw = {sockets[1], mode, 0, 0};
        pthread_t thread;
        assert(pthread_create(&thread, NULL, failing_worker, &fw) == 0);

        /* The first exchange fails on the worker's reply. */
        ClusterItem items[1];
        memset(items, 0, sizeof(items));
        assert(cluster_item(idxs, ws, keff, 1, 2, 42, &items[0], 3, x) == 1);
        float out[6] = {0};
        assert(cluster_exchange(&w, COLI_CLUSTER_VERSION, 7, 3, 5, items, 1, out, NULL) == -1);
        assert(w.failed == 1);
        for (int z = 0; z < 6; z++) assert(out[z] == 0.0f);   /* nothing of a failed reply lands */

        /* In FAIL_STALE the worker's well-formed 52-byte reply is queued on
         * the link before the second exchange begins (wait for it, so the
         * claim below is about bytes that were there to be read). */
        char peek[256];
        ssize_t pending = 0;
        for (int tries = 0; mode == FAIL_STALE && pending < 52 && tries < 500; tries++) {
            pending = recv(sockets[0], peek, sizeof(peek), MSG_PEEK | MSG_DONTWAIT);
            if (pending < 52) usleep(10000);
        }
        if (mode == FAIL_STALE) assert(pending == 52);

        /* The second exchange on the same link is refused before any I/O: it
         * returns at once (the alarm turns a blocked read into a failure), it
         * sends the worker nothing, and it reads nothing -- in FAIL_STALE the
         * well-formed 52-byte reply the worker queued is still there, unread. */
        memset(items, 0, sizeof(items));
        assert(cluster_item(idxs, ws, keff, 1, 2, 42, &items[0], 3, x) == 1);
        float out2[6] = {0};
        alarm(5);
        int rc = cluster_exchange(&w, COLI_CLUSTER_VERSION, 7, 3, 5, items, 1, out2, NULL);
        alarm(0);
        assert(rc == -1);
        assert(w.failed == 1);
        for (int z = 0; z < 6; z++) assert(out2[z] == 0.0f);
        cluster_item_free(&items[0]);
        pending = recv(sockets[0], peek, sizeof(peek), MSG_PEEK | MSG_DONTWAIT);
        if (mode == FAIL_STALE) {
            assert(pending == 52);                     /* magic 8 + 4 u32 + eid nr + 6 f32 */
            assert(memcmp(peek, COLI_CLUSTER_MAGIC, 8) == 0);
        } else {
            assert(pending == 0);                      /* EOF: the worker shut its write side */
        }

        close(sockets[0]);                             /* ends the worker's wait */
        assert(pthread_join(thread, NULL) == 0);
        close(sockets[1]);
        assert(fw.failed == 0);
        assert(fw.bytes_after == 0);                   /* no second request reached it */
    }
}
#endif

int main(void)
{
#if defined(__APPLE__) || defined(__linux__) || defined(__FreeBSD__)
    test_disconnected_peer_is_an_io_error();
    test_wire_round_trip();
    test_q8_codec_matches_qrow_i8();
    test_v2_f32_rows_are_raw();
    test_wire_round_trip_q8();
    test_hello_refuses_a_v1_worker_by_name();
    test_hello_accepts_a_v2_worker();
    test_shared_expert_rides_the_routed_batch();
    test_failed_link_refuses_later_exchanges_without_io();
    puts("cluster protocol tests: ok");
#else
    puts("cluster protocol tests: skipped on Windows");
#endif
    return 0;
}
