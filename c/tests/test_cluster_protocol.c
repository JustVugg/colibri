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
#endif

int main(void)
{
#if defined(__APPLE__) || defined(__linux__) || defined(__FreeBSD__)
    test_disconnected_peer_is_an_io_error();
    test_wire_round_trip();
    test_q8_codec_matches_qrow_i8();
    test_v2_f32_rows_are_raw();
    test_wire_round_trip_q8();
    puts("cluster protocol tests: ok");
#else
    puts("cluster protocol tests: skipped on Windows");
#endif
    return 0;
}
