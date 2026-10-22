/* qwen36_prefetch_ring.h without a model: an entry reserved, copied and
 * taken, the swap that hands the copy to the cache's slot, the entries a copy
 * may reuse, a copy lent before its load, and a load that waits for a copy
 * still reading. A slot here is a box whose storage holds the key it was
 * loaded with. A load that waits is seen waiting (qpr_waiting) before its
 * copy ends, so the checks do not depend on how fast a thread starts. */
#define _POSIX_C_SOURCE 200809L
#include "../qwen36_prefetch_ring.h"
#include <errno.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>

static int failures, checks;
#define CHECK(c, ...) do { checks++; if (!(c)) { failures++; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

typedef struct { int *data; } Box;
static int swaps;
static void swap_boxes(void *a, void *b) {
    Box *x = a, *y = b;
    int *t = x->data;
    x->data = y->data;
    y->data = t;
    swaps++;
}

static void *box_storage(void *b) { return ((Box *)b)->data; }   /* what a loan reads of a slot */

enum { N = 3 };
static int storage[N + 1];
static Box boxes[N], slot;
static void ring_init(QprRing *r) {
    void *payloads[N];
    for (int i = 0; i <= N; i++) storage[i] = -1;
    for (int i = 0; i < N; i++) { boxes[i].data = &storage[i]; payloads[i] = &boxes[i]; }
    slot.data = &storage[N];
    swaps = 0;
    if (!qpr_init(r, payloads, N)) { printf("FAIL: qpr_init\n"); exit(2); }
}
/* A copy of key for issue, read at once: the entry it took. */
static int copy(QprRing *r, int key, uint64_t issue) {
    void *payload = NULL;
    int at = qpr_reserve(r, key, issue, &payload);
    if (at >= 0) {
        *((Box *)payload)->data = key;
        qpr_done(r, at, 1);
    }
    return at;
}

typedef struct {
    QprRing *ring;
    int key, result, finished;
    pthread_mutex_t mutex;
    pthread_cond_t changed;
    pthread_t thread;
} Taker;
static void *take_main(void *arg) {
    Taker *t = arg;
    int result = qpr_take(t->ring, t->key, &slot, swap_boxes);
    pthread_mutex_lock(&t->mutex);
    t->result = result;
    t->finished = 1;
    pthread_cond_broadcast(&t->changed);
    pthread_mutex_unlock(&t->mutex);
    return NULL;
}
static void take_start(Taker *t, QprRing *r, int key) {
    memset(t, 0, sizeof *t);
    t->ring = r;
    t->key = key;
    pthread_mutex_init(&t->mutex, NULL);
    pthread_cond_init(&t->changed, NULL);
    if (pthread_create(&t->thread, NULL, take_main, t)) { printf("FAIL: pthread_create\n"); exit(2); }
}
/* 1 when the load finished within ms milliseconds: for a check that it does
 * not, while the copy it waits for is still reading. */
static int finished_within(Taker *t, long ms) {
    struct timespec limit;
    clock_gettime(CLOCK_REALTIME, &limit);
    limit.tv_sec += ms / 1000;
    limit.tv_nsec += (ms % 1000) * 1000000L;
    if (limit.tv_nsec >= 1000000000L) { limit.tv_sec++; limit.tv_nsec -= 1000000000L; }
    pthread_mutex_lock(&t->mutex);
    while (!t->finished && !pthread_cond_timedwait(&t->changed, &t->mutex, &limit)) {}
    int finished = t->finished;
    pthread_mutex_unlock(&t->mutex);
    return finished;
}
static void take_join(Taker *t) {
    pthread_join(t->thread, NULL);
    pthread_cond_destroy(&t->changed);
    pthread_mutex_destroy(&t->mutex);
}

/* Until the load waits for the copy (the watchdog below ends a hang). */
static void until_waiting(QprRing *r) {
    struct timespec nap = {0, 1000000};
    while (qpr_waiting(r) < 1) nanosleep(&nap, NULL);
}

/* A hang must fail, not sit forever in CI (as test_qwen36_global_cache.c does). */
static void *watchdog(void *arg) {
    (void)arg;
    struct timespec limit = {60, 0};
    while (nanosleep(&limit, &limit) && errno == EINTR) {}
    (void)!write(2, "FAIL: hung for 60 s\n", 20);
    _exit(1);
}

int main(void) {
    pthread_t dog;
    if (!pthread_create(&dog, NULL, watchdog, NULL)) pthread_detach(dog);
    QprRing r;
    void *none[1] = {NULL}, *many[QPR_MAX + 1];
    for (int i = 0; i <= QPR_MAX; i++) many[i] = &boxes[0];
    CHECK(!qpr_init(&r, none, 1), "a ring with a missing payload");
    CHECK(!qpr_init(&r, many, 0) && !qpr_init(&r, many, QPR_MAX + 1), "a ring of 0 or QPR_MAX + 1 entries");

    /* Reserved, copied, taken: the slot gets the copy's storage, the entry the slot's, and is free again. */
    ring_init(&r);
    void *payload = NULL;
    int at = qpr_reserve(&r, 40, 1, &payload);
    CHECK(at == 0 && payload == &boxes[0] && r.state[0] == QPR_COPYING, "reserve: entry %d", at);
    *((Box *)payload)->data = 40;
    qpr_done(&r, at, 1);
    CHECK(r.state[0] == QPR_READY && r.key[0] == 40, "done: the entry does not hold its copy");
    CHECK(qpr_take(&r, 41, &slot, swap_boxes) == 0 && swaps == 0, "a key the ring does not hold was taken");
    CHECK(qpr_take(&r, 40, &slot, swap_boxes) == 1 && swaps == 1, "the copy was not taken");
    CHECK(slot.data == &storage[0] && *slot.data == 40, "the slot does not hold the copy (%d)", *slot.data);
    CHECK(boxes[0].data == &storage[N], "the entry did not get the slot's storage");
    CHECK(r.state[0] == QPR_FREE && r.key[0] == -1, "a taken entry is not free");
    CHECK(qpr_take(&r, 40, &slot, swap_boxes) == 0 && swaps == 1, "a copy was taken twice");

    /* A key copied or copying is not copied again; a failed copy frees its entry. */
    at = qpr_reserve(&r, 42, 1, &payload);
    CHECK(at >= 0 && qpr_reserve(&r, 42, 1, &payload) == -1, "a key being copied was reserved again");
    qpr_done(&r, at, 1);
    CHECK(qpr_reserve(&r, 42, 2, &payload) == -1, "a key the ring holds was reserved again");
    at = qpr_reserve(&r, 43, 2, &payload);
    qpr_done(&r, at, 0);
    CHECK(r.state[at] == QPR_FREE && qpr_take(&r, 43, &slot, swap_boxes) == 0, "a failed copy was kept");
    QprStats s = qpr_snapshot(&r);
    CHECK(s.copies == 2 && s.taken == 1 && s.failed == 1 && s.waited == 0 && s.busy == 0,
          "stats: %llu copies, %llu taken, %llu failed, %llu waited, %llu busy", (unsigned long long)s.copies,
          (unsigned long long)s.taken, (unsigned long long)s.failed, (unsigned long long)s.waited,
          (unsigned long long)s.busy);
    qpr_destroy(&r);

    /* Which entry a copy takes: a free one first, then the copy of an earlier issue reserved first; never a copy of
     * its own issue or a later one, nor one still reading. */
    ring_init(&r);
    CHECK(copy(&r, 10, 1) == 0 && copy(&r, 11, 1) == 1 && copy(&r, 12, 1) == 2, "three copies of issue 1");
    CHECK(copy(&r, 13, 1) == -2 && r.key[0] == 10 && r.key[1] == 11 && r.key[2] == 12,
          "a copy of issue 1 took another of issue 1");
    CHECK(copy(&r, 20, 2) == 0 && r.key[0] == 20, "a copy of issue 2 did not take the first of issue 1");
    CHECK(qpr_take(&r, 12, &slot, swap_boxes) == 1, "take 12");
    CHECK(copy(&r, 21, 2) == 2 && r.key[1] == 11, "a copy did not take the free entry first");
    CHECK(copy(&r, 30, 3) == 1 && r.key[1] == 30, "a copy of issue 3 did not take the last of issue 1");
    CHECK(copy(&r, 22, 2) == -2, "a copy of issue 2 took one of issue 2 or 3");
    CHECK(qpr_take(&r, 20, &slot, swap_boxes) == 1 && *slot.data == 20, "take 20");
    CHECK(qpr_reserve(&r, 31, 3, &payload) == 0, "reserve 31");
    CHECK(copy(&r, 40, 4) == 2 && copy(&r, 41, 4) == 1 && copy(&r, 42, 4) == -2,
          "a copy took an entry still reading");
    qpr_done(&r, 0, 1);
    s = qpr_snapshot(&r);
    CHECK(s.busy == 3 && s.copies == 9, "stats: %llu busy, %llu copies", (unsigned long long)s.busy,
          (unsigned long long)s.copies);
    qpr_destroy(&r);

    /* A copy lent before its load: no later copy reuses its entry until the load takes it or the loan ends; a copy
     * still reading is not lent. */
    ring_init(&r);
    CHECK(copy(&r, 60, 1) == 0 && copy(&r, 61, 1) == 1 && copy(&r, 62, 1) == 2, "three copies of issue 1");
    payload = NULL;
    CHECK(qpr_hold(&r, 60, &payload, NULL) == 1 && payload == &boxes[0] && *((Box *)payload)->data == 60,
          "the copy of 60 was not lent");
    CHECK(qpr_hold(&r, 63, &payload, NULL) == 0, "a key the ring does not hold was lent");
    CHECK(copy(&r, 70, 2) == 1 && r.key[0] == 60, "a copy reused a lent entry");
    CHECK(qpr_take(&r, 60, &slot, swap_boxes) == 1 && *slot.data == 60 && r.lent[0] == 0,
          "the lent copy was not taken, or stayed lent");
    CHECK(copy(&r, 71, 2) == 0, "the entry of a lent copy taken by its load is not free");
    CHECK(qpr_hold(&r, 62, &payload, NULL) == 1 && copy(&r, 72, 2) == -2, "a copy reused a lent entry");
    qpr_release(&r);
    CHECK(copy(&r, 72, 2) == 2 && r.key[2] == 72, "the entry of an ended loan was not reused");
    at = qpr_reserve(&r, 80, 3, &payload);
    CHECK(at == 1 && qpr_hold(&r, 80, &payload, NULL) == 0, "a copy still reading was lent (entry %d)", at);
    *((Box *)payload)->data = 80;
    qpr_done(&r, at, 1);
    payload = NULL;
    CHECK(qpr_hold(&r, 80, &payload, box_storage) == 1 && payload == &storage[1] && *(int *)payload == 80,
          "a done copy was not lent, or the loan did not read its storage");
    s = qpr_snapshot(&r);
    CHECK(s.busy == 1 && s.taken == 1 && s.copies == 7, "stats: %llu busy, %llu taken, %llu copies",
          (unsigned long long)s.busy, (unsigned long long)s.taken, (unsigned long long)s.copies);
    qpr_destroy(&r);

    /* A load waits for a copy still reading, then takes it; if the copy fails, it reads the key itself. */
    ring_init(&r);
    at = qpr_reserve(&r, 50, 1, &payload);
    Taker t;
    take_start(&t, &r, 50);
    until_waiting(&r);
    CHECK(!finished_within(&t, 200), "a load did not wait for the copy still reading");
    *((Box *)payload)->data = 50;
    qpr_done(&r, at, 1);
    take_join(&t);
    CHECK(t.result == 1 && *slot.data == 50 && swaps == 1, "the load that waited did not take the copy");
    at = qpr_reserve(&r, 51, 1, &payload);
    take_start(&t, &r, 51);
    until_waiting(&r);
    CHECK(!finished_within(&t, 200), "a load did not wait for the copy still reading");
    qpr_done(&r, at, 0);
    take_join(&t);
    CHECK(t.result == 0 && swaps == 1 && *slot.data == 50, "a load took a failed copy");
    s = qpr_snapshot(&r);
    CHECK(s.taken == 1 && s.waited == 1 && s.failed == 1, "stats: %llu taken, %llu waited, %llu failed",
          (unsigned long long)s.taken, (unsigned long long)s.waited, (unsigned long long)s.failed);
    qpr_destroy(&r);

    printf("test_qwen36_prefetch_ring: %d checks, %d failures\n", checks, failures);
    return failures != 0;
}
