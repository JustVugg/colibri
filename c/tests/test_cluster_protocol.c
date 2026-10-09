/* COLIEX01 cluster wire contract: network-order headers, raw f32 activations,
 * and one request/response over a socketpair. No model fixture required. */
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

/* ---- fan-out: every request is on the wire before any reply is read ---- */

/* Two workers that will not answer until BOTH have received their request. A
 * coordinator that serves workers in turn -- send to one, wait for its reply,
 * send to the next -- never gets past the first: the barrier times out (2 s)
 * with `late` set and the test fails. A fan-out coordinator passes straight
 * through. Each worker then holds its reply for `hold_ms`, so with fan-out the
 * batch takes about one hold; served in turn it would take two. */
typedef struct { pthread_mutex_t mu; pthread_cond_t cv; int arrived, late; } FanoutBarrier;
typedef struct { int fd, eid, hold_ms; FanoutBarrier *bar; int failed; } FanoutWorker;

static void *fanout_worker(void *opaque)
{
    FanoutWorker *w = opaque;
    int fd = w->fd;
    char magic[8];
    uint32_t version, layer, D, I, n, eid, nr, zero = 0;
    float rows[2 * 3];
    /* v1 request for layer 0, D=3, I=5: one item, this worker's expert, both rows. */
    if (cluster_io(fd, magic, 8, 0) || memcmp(magic, COLI_CLUSTER_MAGIC, 8) ||
        cluster_u32(fd, &version, 0) || version != COLI_CLUSTER_VERSION ||
        cluster_u32(fd, &layer, 0) || layer != 0 || cluster_u32(fd, &D, 0) || D != 3 ||
        cluster_u32(fd, &I, 0) || I != 5 || cluster_u32(fd, &n, 0) || n != 1 ||
        cluster_u32(fd, &eid, 0) || eid != (uint32_t)w->eid ||
        cluster_u32(fd, &nr, 0) || nr != 2 || cluster_io(fd, rows, sizeof(rows), 0)) {
        w->failed = 1;
        return NULL;
    }
    struct timespec deadline;
    clock_gettime(CLOCK_REALTIME, &deadline);
    deadline.tv_sec += 2;
    pthread_mutex_lock(&w->bar->mu);
    w->bar->arrived++;
    pthread_cond_broadcast(&w->bar->cv);
    while (w->bar->arrived < 2 && !w->bar->late)
        if (pthread_cond_timedwait(&w->bar->cv, &w->bar->mu, &deadline) == ETIMEDOUT) w->bar->late = 1;
    pthread_mutex_unlock(&w->bar->mu);
    struct timespec hold = {w->hold_ms / 1000, (w->hold_ms % 1000) * 1000000L};
    nanosleep(&hold, NULL);
    for (int i = 0; i < 6; i++) rows[i] *= (float)(w->eid + 2);   /* expert 0: 2x, expert 1: 3x */
    version = COLI_CLUSTER_VERSION;
    if (cluster_io(fd, (void *)COLI_CLUSTER_MAGIC, 8, 1) || cluster_u32(fd, &version, 1) ||
        cluster_u32(fd, &zero, 1) || cluster_u32(fd, &n, 1) || cluster_u32(fd, &eid, 1) ||
        cluster_u32(fd, &nr, 1) || cluster_io(fd, rows, sizeof(rows), 1))
        w->failed = 1;
    return NULL;
}

static double fanout_now_ms(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1e3 + (double)t.tv_nsec / 1e6;
}

static void test_fanout_puts_every_request_on_the_wire_first(void)
{
    static Model m;
    memset(&m, 0, sizeof(m));
    m.c.hidden = 3;
    m.c.moe_inter = 5;
    /* Two rows, top-2, both rows to experts 0 and 1; at layer 0 expert 0 lives
     * on worker 0 and expert 1 on worker 1 ((eid+layer) % 2). */
    float x[6] = {1.0f, -2.0f, 0.5f, 3.0f, -4.0f, 0.25f};
    int idxs[4] = {0, 1, 0, 1}, keff[2] = {2, 2}, uniq[2] = {0, 1};
    float ws[4] = {0.5f, 0.25f, 0.125f, 1.0f};
    const int hold_ms = 200;
    int s0[2], s1[2];
    assert(socketpair(AF_UNIX, SOCK_STREAM, 0, s0) == 0);
    assert(socketpair(AF_UNIX, SOCK_STREAM, 0, s1) == 0);
    FanoutBarrier bar = {PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER, 0, 0};
    FanoutWorker w0 = {s0[1], 0, hold_ms, &bar, 0}, w1 = {s1[1], 1, hold_ms, &bar, 0};
    pthread_t t0, t1;
    assert(pthread_create(&t0, NULL, fanout_worker, &w0) == 0);
    assert(pthread_create(&t1, NULL, fanout_worker, &w1) == 0);
    g_cluster_n = 2;
    g_cluster_workers[0].fd = s0[0];
    snprintf(g_cluster_workers[0].host, sizeof(g_cluster_workers[0].host), "mac-a");
    g_cluster_workers[0].port = 9100;
    g_cluster_workers[1].fd = s1[0];
    snprintf(g_cluster_workers[1].host, sizeof(g_cluster_workers[1].host), "mac-b");
    g_cluster_workers[1].port = 9101;
    float out[6] = {0};
    double t = fanout_now_ms();
    cluster_moe_batch(&m, 0, x, 2, out, idxs, ws, keff, 2, uniq, 0, 2);
    double elapsed = fanout_now_ms() - t;
    assert(pthread_join(t0, NULL) == 0);
    assert(pthread_join(t1, NULL) == 0);
    g_cluster_n = 0;
    assert(w0.failed == 0 && w1.failed == 0);
    assert(bar.late == 0);                       /* both requests were on the wire before any reply */
    /* row r: ws[r][0] * 2x (expert 0, worker 0) + ws[r][1] * 3x (expert 1, worker 1), in that order */
    for (int r = 0; r < 2; r++)
        for (int d = 0; d < 3; d++)
            assert(out[r * 3 + d] == ws[r * 2] * 2.0f * x[r * 3 + d] + ws[r * 2 + 1] * 3.0f * x[r * 3 + d]);
    printf("fan-out: 2 workers holding %d ms each, the batch took %.0f ms (served in turn: %d ms)\n",
           hold_ms, elapsed, 2 * hold_ms);
    assert(elapsed < 1.5 * hold_ms);             /* toward max(), not sum() */
    close(s0[0]); close(s0[1]); close(s1[0]); close(s1[1]);
}
#endif

int main(void)
{
#if defined(__APPLE__) || defined(__linux__) || defined(__FreeBSD__)
    test_disconnected_peer_is_an_io_error();
    test_wire_round_trip();
    test_fanout_puts_every_request_on_the_wire_first();
    puts("cluster protocol tests: ok");
#else
    puts("cluster protocol tests: skipped on Windows");
#endif
    return 0;
}
