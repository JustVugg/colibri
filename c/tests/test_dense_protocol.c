/* Dense shard wire contract: network-order headers, raw f32 activations, and
 * one request/response over a persistent stream. No model fixture required. */
#define main coli_engine_main_unused
#include "../colibri.c"
#undef main

#include <assert.h>

#if defined(__APPLE__) || defined(__linux__) || defined(__FreeBSD__)
typedef struct { int fd; int failed; } DenseProtocolArgs;

static void *dense_protocol_worker(void *opaque)
{
    DenseProtocolArgs *args = opaque;
    char magic[8];
    uint32_t version, layer, hidden, intermediate, rows;
    float input[3], output[3];

    if (dense_io(args->fd, magic, sizeof(magic), 0) || memcmp(magic, COLI_DENSE_MAGIC, 8) ||
        dense_u32(args->fd, &version, 0) || version != 1 ||
        dense_u32(args->fd, &layer, 0) || layer != 2 ||
        dense_u32(args->fd, &hidden, 0) || hidden != 3 ||
        dense_u32(args->fd, &intermediate, 0) || intermediate != 5 ||
        dense_u32(args->fd, &rows, 0) || rows != 1 ||
        dense_io(args->fd, input, sizeof(input), 0)) {
        args->failed = 1;
        return NULL;
    }
    for (int i = 0; i < 3; i++) output[i] = input[i] * 2.0f;
    version = 1;
    if (dense_io(args->fd, (void *)COLI_DENSE_MAGIC, 8, 1) ||
        dense_u32(args->fd, &version, 1) ||
        dense_u32(args->fd, &(uint32_t){0}, 1) ||
        dense_u32(args->fd, &(uint32_t){1}, 1) ||
        dense_io(args->fd, output, sizeof(output), 1))
        args->failed = 1;
    return NULL;
}

static void test_wire_round_trip(void)
{
    int sockets[2];
    assert(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0);
    DenseProtocolArgs args = {sockets[1], 0};
    pthread_t thread;
    assert(pthread_create(&thread, NULL, dense_protocol_worker, &args) == 0);

    uint32_t value;
    float input[3] = {1.0f, -2.0f, 0.5f}, output[3] = {0};
    assert(dense_io(sockets[0], (void *)COLI_DENSE_MAGIC, 8, 1) == 0);
    value = 1; assert(dense_u32(sockets[0], &value, 1) == 0);
    value = 2; assert(dense_u32(sockets[0], &value, 1) == 0);
    value = 3; assert(dense_u32(sockets[0], &value, 1) == 0);
    value = 5; assert(dense_u32(sockets[0], &value, 1) == 0);
    value = 1; assert(dense_u32(sockets[0], &value, 1) == 0);
    assert(dense_io(sockets[0], input, sizeof(input), 1) == 0);

    char magic[8];
    assert(dense_io(sockets[0], magic, 8, 0) == 0);
    assert(memcmp(magic, COLI_DENSE_MAGIC, 8) == 0);
    value = 0; assert(dense_u32(sockets[0], &value, 0) == 0 && value == 1);
    value = 1; assert(dense_u32(sockets[0], &value, 0) == 0 && value == 0);
    value = 0; assert(dense_u32(sockets[0], &value, 0) == 0 && value == 1);
    assert(dense_io(sockets[0], output, sizeof(output), 0) == 0);
    assert(output[0] == 2.0f && output[1] == -4.0f && output[2] == 1.0f);

    assert(pthread_join(thread, NULL) == 0);
    assert(args.failed == 0);
    close(sockets[0]);
    close(sockets[1]);
}

static void test_layer_ownership(void)
{
    memset(g_dense_shards, 0, sizeof(g_dense_shards));
    g_dense_n = 2;
    g_dense_shards[0].first = 0; g_dense_shards[0].last = 2;
    g_dense_shards[1].first = 3; g_dense_shards[1].last = 5;
    assert(dense_owner(0) == 0);
    assert(dense_owner(2) == 0);
    assert(dense_owner(3) == 1);
    assert(dense_owner(5) == 1);
    assert(dense_owner(6) == -1);
    g_dense_n = 0;
}
#endif

int main(void)
{
#if defined(__APPLE__) || defined(__linux__) || defined(__FreeBSD__)
    test_wire_round_trip();
    test_layer_ownership();
    puts("dense protocol tests: ok");
#else
    puts("dense protocol tests: skipped on Windows");
#endif
    return 0;
}
