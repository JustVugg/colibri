/* QWEN36_PREFETCH (qwen36.c): copies of the next layer's guessed experts,
 * made while a decode step's layer computes, in a few slots of their own
 * beside the expert cache. A copy reserves an entry for its key
 * (qpr_reserve), reads the record into the entry's payload without the
 * ring's lock and marks it done (qpr_done). When the cache then loads that
 * key into one of its slots, the load takes the entry instead of reading
 * (qpr_take): it waits for a copy still reading, and the caller's swap trades
 * the two payloads' storage, so that the cache's slot holds the copied bytes
 * and the entry the slot's old storage, for a later copy. The cache never
 * sees the ring: it borrows, evicts and loads as it does without it, and its
 * slots end up with the same bytes; only where a load's bytes come from
 * changes. A copy can also be lent before its load (qpr_hold): no later copy
 * reuses its entry until the load takes it or qpr_release ends the loan, so
 * its storage can be read meanwhile. Header-only, without a model: the
 * engine supplies the payloads, the swap and what a loan reads of a payload.
 */
#ifndef QWEN36_PREFETCH_RING_H
#define QWEN36_PREFETCH_RING_H
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define QPR_MAX 8
enum { QPR_FREE, QPR_COPYING, QPR_READY };
typedef struct {
    uint64_t copies, taken, waited, busy, failed;
} QprStats;
typedef struct {
    pthread_mutex_t mutex;
    pthread_cond_t changed;
    int n, initialized, waiting;  /* waiting: loads waiting for a copy still reading (qpr_take) */
    int key[QPR_MAX], state[QPR_MAX], lent[QPR_MAX];  /* lent: a READY copy that no copy may reuse (qpr_hold) */
    uint64_t issue[QPR_MAX], order[QPR_MAX], clock;  /* the issue a copy was made for; when it was reserved */
    void *payload[QPR_MAX];
    QprStats stats;
} QprRing;

/* n entries (1 to QPR_MAX), each with its payload, all free. */
static inline int qpr_init(QprRing *r, void *const *payloads, int n) {
    memset(r, 0, sizeof *r);
    if (!payloads || n < 1 || n > QPR_MAX) return 0;
    for (int i = 0; i < n; i++) {
        if (!payloads[i]) return 0;
        r->payload[i] = payloads[i];
        r->key[i] = -1;
    }
    if (pthread_mutex_init(&r->mutex, NULL)) return 0;
    if (pthread_cond_init(&r->changed, NULL)) {
        pthread_mutex_destroy(&r->mutex);
        return 0;
    }
    r->n = n;
    r->initialized = 1;
    return 1;
}

/* An entry for a copy of key made for issue (a count that grows with each
 * layer's guesses): its index, with *payload the storage to read into; the
 * entry is COPYING until qpr_done. -1 when the ring holds key already or is
 * copying it. A copy takes a free entry, or else the one reserved first among
 * the copies of earlier issues, which their layer did not take; -2 when there
 * is none: every entry is copying, lent, or holds a copy of this issue or a
 * later one. */
static inline int qpr_reserve(QprRing *r, int key, uint64_t issue, void **payload) {
    int pick = -1;
    pthread_mutex_lock(&r->mutex);
    for (int i = 0; i < r->n; i++)
        if (r->state[i] != QPR_FREE && r->key[i] == key) {
            pthread_mutex_unlock(&r->mutex);
            return -1;
        }
    for (int i = 0; i < r->n; i++) {
        if (r->state[i] == QPR_COPYING || r->lent[i] || (r->state[i] == QPR_READY && r->issue[i] >= issue)) continue;
        if (r->state[i] == QPR_FREE) {
            pick = i;
            break;
        }
        if (pick < 0 || r->order[i] < r->order[pick]) pick = i;
    }
    if (pick < 0) {
        r->stats.busy++;
        pthread_mutex_unlock(&r->mutex);
        return -2;
    }
    r->key[pick] = key;
    r->state[pick] = QPR_COPYING;
    r->issue[pick] = issue;
    r->order[pick] = ++r->clock;
    *payload = r->payload[pick];
    pthread_mutex_unlock(&r->mutex);
    return pick;
}

/* The copy into entry i ended. Done, the entry holds its key until a load
 * takes it or a later copy reuses it; failed, it is free again. */
static inline void qpr_done(QprRing *r, int i, int ok) {
    pthread_mutex_lock(&r->mutex);
    if (i < 0 || i >= r->n || r->state[i] != QPR_COPYING) abort();
    if (ok) {
        r->state[i] = QPR_READY;
        r->stats.copies++;
    } else {
        r->state[i] = QPR_FREE;
        r->key[i] = -1;
        r->stats.failed++;
    }
    pthread_cond_broadcast(&r->changed);
    pthread_mutex_unlock(&r->mutex);
}

/* A load of key into target: 1 when the ring held a copy of it, once swap
 * has traded that entry's payload storage with target's (the entry is free
 * again); a copy still reading is waited for first. 0 when the ring has no
 * copy of key, or its copy failed: the caller reads it. swap runs under the
 * ring's lock. */
static inline int qpr_take(QprRing *r, int key, void *target, void (*swap)(void *, void *)) {
    int taken = 0, waited = 0;
    pthread_mutex_lock(&r->mutex);
    for (;;) {
        int i = 0;
        while (i < r->n && (r->state[i] == QPR_FREE || r->key[i] != key)) i++;
        if (i == r->n) break;
        if (r->state[i] == QPR_COPYING) {
            waited = 1;
            r->waiting++;
            if (pthread_cond_wait(&r->changed, &r->mutex)) abort();
            r->waiting--;
            continue;
        }
        swap(r->payload[i], target);
        r->state[i] = QPR_FREE;
        r->key[i] = -1;
        r->lent[i] = 0;
        r->stats.taken++;
        if (waited) r->stats.waited++;
        taken = 1;
        break;
    }
    pthread_mutex_unlock(&r->mutex);
    return taken;
}

/* The copy of key, if one is done: 1 with *out its payload, or what get reads
 * of it, under the ring's lock (a take trades the payloads' storage under the
 * same lock). The copy is lent until a load takes it (the storage then moves
 * to the load's target) or qpr_release ends the loan; meanwhile no copy
 * reuses the entry. 0 when the ring has no finished copy of key. */
static inline int qpr_hold(QprRing *r, int key, void **out, void *(*get)(void *payload)) {
    int i = 0;
    pthread_mutex_lock(&r->mutex);
    while (i < r->n && (r->state[i] != QPR_READY || r->key[i] != key)) i++;
    if (i < r->n) {
        r->lent[i] = 1;
        *out = get ? get(r->payload[i]) : r->payload[i];
    }
    pthread_mutex_unlock(&r->mutex);
    return i < r->n;
}

/* Ends every loan a load did not take: those copies can be reused again. */
static inline void qpr_release(QprRing *r) {
    pthread_mutex_lock(&r->mutex);
    for (int i = 0; i < r->n; i++) r->lent[i] = 0;
    pthread_mutex_unlock(&r->mutex);
}

/* How many loads wait for a copy still reading. */
static inline int qpr_waiting(QprRing *r) {
    pthread_mutex_lock(&r->mutex);
    int n = r->waiting;
    pthread_mutex_unlock(&r->mutex);
    return n;
}

static inline QprStats qpr_snapshot(QprRing *r) {
    pthread_mutex_lock(&r->mutex);
    QprStats s = r->stats;
    pthread_mutex_unlock(&r->mutex);
    return s;
}

/* No copy may be in flight; the payloads stay the caller's. */
static inline void qpr_destroy(QprRing *r) {
    if (!r->initialized) return;
    pthread_cond_destroy(&r->changed);
    pthread_mutex_destroy(&r->mutex);
    memset(r, 0, sizeof *r);
}
#endif
