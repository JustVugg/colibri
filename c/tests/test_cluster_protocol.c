/* COLIEX01 cluster wire contract: network-order headers, raw f32 activations,
 * and one request/response over a socketpair. No model fixture required. */
#define main coli_engine_main_unused
#include "../colibri.c"
#undef main

#include <assert.h>

#if defined(__APPLE__) || defined(__linux__) || defined(__FreeBSD__)
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
#endif

int main(void)
{
#if defined(__APPLE__) || defined(__linux__) || defined(__FreeBSD__)
    test_wire_round_trip();
    puts("cluster protocol tests: ok");
#else
    puts("cluster protocol tests: skipped on Windows");
#endif
    return 0;
}
