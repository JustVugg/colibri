/* CPU ordering fixture for the production advisory scheduling helpers.
 * The queue model counts commits exactly when a nonempty batch is flushed;
 * it does not model GPU performance or execute Metal, model or file I/O.
 * clang -O2 -std=c11 c/tests/test_qwen36_advice_schedule.c -o /tmp/qwen-advice-schedule
 */
#include <stdio.h>
#include <stdlib.h>
#include "../qwen36_advice_schedule.h"
#define REQUIRE(c) do { if (!(c)) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #c); abort(); \
} } while (0)
enum { ENCODE = 1, FLUSH, ADVICE, SYNC, RELEASE, START, MAX_EVENTS = 256 };
typedef struct {
    int encoded, in_flight, commands, commits, syncs, submits, emissions;
    int events[MAX_EVENTS], event_n;
    int ids[128], layers[128], id_n;
    int require_submitted, fail_submit, lease_live;
} Queue;
static void event(Queue *q, int value) {
    REQUIRE(q->event_n < MAX_EVENTS); q->events[q->event_n++] = value;
}
static void encode(Queue *q) { q->encoded++; q->commands++; event(q, ENCODE); }
static int submit(void *ctx) {
    Queue *q = ctx; q->submits++; event(q, FLUSH);
    if (q->encoded) { q->commits++; q->in_flight += q->encoded; q->encoded = 0; }
    return !q->fail_submit;
}
static void emit(void *ctx, int layer, const int *ids, int from, int to) {
    Queue *q = ctx;
    if (q->require_submitted) {
        REQUIRE(q->event_n && q->events[q->event_n - 1] == FLUSH);
        REQUIRE(!q->encoded && q->in_flight);
    }
    REQUIRE(from >= 0 && to >= from && to - from <= 64);
    event(q, ADVICE); q->emissions++;
    for (int i = from; i < to; i++) {
        REQUIRE(q->id_n < 128); q->ids[q->id_n] = ids[i]; q->layers[q->id_n++] = layer;
    }
}
static int sync_queue(Queue *q) {
    int ok = submit(q); q->syncs++; event(q, SYNC);
    q->in_flight = 0; return ok;       /* a failed GPU command is still terminal */
}
static void release_lease(Queue *q) {
    REQUIRE(q->lease_live && !q->encoded && !q->in_flight);
    q->lease_live = 0; event(q, RELEASE);
}
static void start_lease(Queue *q) {
    REQUIRE(!q->lease_live); q->lease_live = 1; event(q, START);
}
static void same_work(const Queue *a, const Queue *b) {
    REQUIRE(a->commands == b->commands && a->commits == b->commits && a->syncs == b->syncs);
    REQUIRE(a->emissions == b->emissions && a->id_n == b->id_n);
    REQUIRE(!memcmp(a->ids, b->ids, (size_t)a->id_n * sizeof(*a->ids)));
    REQUIRE(!memcmp(a->layers, b->layers, (size_t)a->id_n * sizeof(*a->layers)));
}
static Queue decode(int deferred, int timed) {
    Queue q = {0}; QasPending p = {0}; int ids[8] = {29, 3, 71, 8, 19, 20, 14, 96};
    int expected[8]; memcpy(expected, ids, sizeof(ids)); start_lease(&q);
    if (deferred) REQUIRE(qas_save(&p, 7, ids, 8));
    else emit(&q, 7, ids, 0, 8);
    encode(&q); if (timed) REQUIRE(submit(&q));  /* previous experts/sum */
    encode(&q); if (timed) REQUIRE(submit(&q));  /* next trunk */
    encode(&q); if (timed) REQUIRE(submit(&q));  /* next router */
    /* The next layer has reused the source logits/ID scratch before advice. */
    for (int i = 0; i < 8; i++) ids[i] = 255 - i;
    if (deferred) {
        q.require_submitted = 1;
        REQUIRE(qas_submit_pending(&p, 7, &q, submit, emit));
        REQUIRE(!p.n && !memcmp(q.ids, expected, sizeof(expected)));
    }
    REQUIRE(sync_queue(&q)); release_lease(&q);
    int submissions = q.submits, emissions = q.emissions;
    REQUIRE(qas_submit_pending(&p, 7, &q, submit, emit));
    REQUIRE(q.submits == submissions && q.emissions == emissions);
    return q;
}
static Queue prefill(int deferred, int timed, int experts, int B) {
    Queue q = {0}; int ids[128], K = 8;
    for (int i = 0; i < experts; i++) ids[i] = i;
    int g0 = 0, n = experts < B ? experts : B; start_lease(&q);
    if (!deferred && n < experts) emit(&q, 4, ids, n, n + B < experts ? n + B : experts);
    encode(&q); REQUIRE(submit(&q));           /* existing shared flush */
    for (;;) {
        for (int sub = 0; sub < n; sub += K) { encode(&q); encode(&q); } /* gate/up, down */
        g0 += n;
        if (g0 == experts) break;
        if (timed) REQUIRE(submit(&q));       /* existing qg_cut */
        if (deferred) {
            q.require_submitted = 1;
            REQUIRE(qas_submit_advice(&q, submit, emit, 4, ids, g0, g0 + B < experts ? g0 + B : experts));
        }
        REQUIRE(sync_queue(&q)); release_lease(&q);
        n = experts - g0 < B ? experts - g0 : B; start_lease(&q);
        if (!deferred && g0 + n < experts)
            emit(&q, 4, ids, g0 + n, g0 + n + B < experts ? g0 + n + B : experts);
    }
    encode(&q); if (timed) REQUIRE(submit(&q)); /* expert_sum and existing cut */
    REQUIRE(sync_queue(&q)); release_lease(&q);
    return q;
}
static void ordering_and_commits(void) {
    for (int timed = 0; timed <= 1; timed++) {
        Queue a = decode(0, timed), b = decode(1, timed); same_work(&a, &b);
        REQUIRE(a.commits == (timed ? 3 : 1) && a.syncs == 1);
        int sizes[] = {8, 9, 16, 25, 63};
        for (unsigned i = 0; i < sizeof(sizes) / sizeof(sizes[0]); i++) {
            Queue p = prefill(0, timed, sizes[i], 8), q = prefill(1, timed, sizes[i], 8);
            same_work(&p, &q);
            REQUIRE(q.emissions == (sizes[i] - 1) / 8);
            for (int j = 0; j < q.id_n; j++) REQUIRE(q.ids[j] == 8 + j);
        }
    }
}
static void cancellation_and_failure(void) {
    QasPending p = {0}; int ids[8] = {1, 2, 3, 4, 5, 6, 7, 8}; Queue q = {0};
    REQUIRE(qas_save(&p, 2, ids, 8)); qas_clear(&p);
    REQUIRE(qas_submit_pending(&p, 2, &q, submit, emit));
    REQUIRE(!q.submits && !q.emissions);       /* reset/snapshot/free clear */
    REQUIRE(qas_save(&p, 2, ids, 8));
    REQUIRE(qas_submit_pending(&p, 3, &q, submit, emit));
    REQUIRE(!p.n && !q.submits && !q.emissions); /* wrong layer discarded */
    REQUIRE(qas_save(&p, 2, ids, 8)); start_lease(&q); encode(&q); q.fail_submit = 1;
    REQUIRE(!qas_submit_pending(&p, 2, &q, submit, emit));
    REQUIRE(!p.n && !q.emissions && q.in_flight);
    REQUIRE(!sync_queue(&q)); release_lease(&q); /* error drains before release */
    REQUIRE(!qas_save(&p, -1, ids, 8) && !p.n);
    REQUIRE(!qas_save(&p, 2, ids, 9) && !p.n);
    REQUIRE(!qas_save(&p, 2, NULL, 8) && !p.n);
    q = (Queue){0}; start_lease(&q); encode(&q); q.fail_submit = 1;
    REQUIRE(!qas_submit_advice(&q, submit, emit, 3, ids, 0, 8));
    REQUIRE(!q.emissions && q.in_flight); REQUIRE(!sync_queue(&q)); release_lease(&q);
}
int main(void) {
    ordering_and_commits(); cancellation_and_failure();
    puts("advice schedule: submit/advice/wait order, unchanged modeled command buffers, copied IDs, bounds and failed-submit cleanup passed");
    return 0;
}
