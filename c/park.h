/* park.h — keep a conversation aside while a short request runs (COLI_PARK).
 *
 * THE PROBLEM. An engine serving one conversation holds one state: the K/V rows and whatever
 * else its layers keep for the tokens it has read. A client's short request between two turns
 * of a conversation (Hermes asks for the conversation's title after the first turn; other
 * agents ask for a summary or a topic the same way) leaves that conversation after a few
 * tokens and takes the state for itself. The conversation's next turn then reads every token
 * again: 86 s for 14,785 tokens on Qwen3.8-Flash-Next, minutes on the largest models.
 *
 * THE IDEA. Such a request writes only the first rows of each cache: its prompt and its answer,
 * a few hundred positions. Before it runs, the conversation's copy of those rows goes aside
 * with its tokens and the state that is not rows (a recurrence, a convolution's window); when
 * the request's turn ends, all of it goes back, and the record says the conversation again.
 * Its next turn continues from where it was and reads only what is new.
 *
 * WHAT THIS FILE KNOWS AND WHAT IT DOES NOT. An engine describes its state as parts (below):
 * a cache of rows by position, a ring of rows by position modulo its size, a block saved whole.
 * Here are the conditions, the copies and the record. The engine brings its device copies to
 * the host before coli_park_save (the state the copies are taken from must be the newest) and
 * tells its devices after coli_park_restore that the host's rows and state are theirs again.
 *
 * One conversation at a time; a conversation of at least COLI_PARK_MIN tokens (1024 unless
 * set); a request whose rows fit before the conversation's end (its prompt, up to
 * COLI_PARK_GEN of its answer, the rows a verify writes past the tokens that stand).
 * COLI_PARK=0 never parks. qwen38.c and qwen36.c park the same way with their own code. */
#ifndef COLI_PARK_H
#define COLI_PARK_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "kv_prefix.h"

#define COLI_PARK_MIN_DEFAULT 1024
#define COLI_PARK_GEN 4096   /* a request's tokens whose rows are kept apart: past them it is not a short one */

/* One piece of state: `heads` blocks `head_stride` bytes apart, each `rows` rows of `row`
 * bytes. A cache by position (row t at t, or at t % rows for a ring) keeps its rows [0, n)
 * apart, all of them when it has fewer; a block saved whole is one row of its size. */
typedef struct { void *base; size_t row; int rows, heads; size_t head_stride; } ColiParkPart;

typedef struct {
    int on, len, kv_len, n, nparts;
    int *fed;
    ColiParkPart *part;
    void **copy;
} ColiPark;

static inline ColiParkPart coli_park_rows(void *base, size_t row, int rows) {
    ColiParkPart p = {base, row, rows, 1, 0}; return p;
}
static inline ColiParkPart coli_park_heads(void *base, size_t row, int rows, int heads, size_t head_stride) {
    ColiParkPart p = {base, row, rows, heads, head_stride}; return p;
}
static inline ColiParkPart coli_park_whole(void *base, size_t bytes) {
    ColiParkPart p = {base, bytes, 1, 1, 0}; return p;
}

static inline void coli_park_free(ColiPark *p) {
    for (int i = 0; p->copy && i < p->nparts; i++) free(p->copy[i]);
    free(p->copy); free(p->part); free(p->fed);
    memset(p, 0, sizeof *p);
}

/* The rows a request of np prompt tokens and up to max_tok more keeps apart, spec of them a
 * verify's, when the conversation held (kvp, kv_len) is one to park for it; 0 when it is not. */
static inline int coli_park_rows_for(const ColiPark *p, const kv_prefix *kvp, int kv_len, int np, int max_tok, int spec) {
    const char *e = getenv("COLI_PARK"), *mn = getenv("COLI_PARK_MIN");
    const int least = mn && atoi(mn) > 0 ? atoi(mn) : COLI_PARK_MIN_DEFAULT,
              gen = max_tok < COLI_PARK_GEN ? max_tok : COLI_PARK_GEN, H = kvp ? kvp->len : 0;
    const int n = np + (gen > 0 ? gen : 0) + spec;
    if ((e && *e == '0') || p->on || !kvp || !kvp->fed || kvp->tainted || H < least || kv_len != H || n >= H) return 0;
    return n;
}

static inline size_t coli_park_part_bytes(const ColiParkPart *q, int n) {
    return (size_t)(n < q->rows ? n : q->rows) * q->row;
}
static inline void coli_park_copy(const ColiParkPart *q, void *copy, int n, int out) {
    const size_t b = coli_park_part_bytes(q, n);
    for (int h = 0; h < q->heads; h++) {
        char *a = (char *)q->base + (size_t)h * q->head_stride, *c = (char *)copy + (size_t)h * b;
        if (out) memcpy(c, a, b); else memcpy(a, c, b);
    }
}

/* The conversation's tokens and its parts' first n rows aside. 1 when parked; 0 (nothing
 * kept) when there is no memory for it, and then the request simply takes the state. */
static inline int coli_park_save(ColiPark *p, const kv_prefix *kvp, int kv_len, const ColiParkPart *parts, int nparts, int n) {
    if (p->on || n <= 0 || nparts <= 0) return 0;
    p->part = (ColiParkPart *)malloc((size_t)nparts * sizeof *parts);
    p->copy = (void **)calloc((size_t)nparts, sizeof(void *));
    p->fed = (int *)malloc((size_t)kvp->len * sizeof(int));
    int ok = p->part && p->copy && p->fed;
    p->nparts = nparts;
    for (int i = 0; ok && i < nparts; i++) {
        p->part[i] = parts[i];
        if (!parts[i].base) continue;
        ok = (p->copy[i] = malloc(coli_park_part_bytes(&parts[i], n) * (size_t)parts[i].heads)) != NULL;
        if (ok) coli_park_copy(&parts[i], p->copy[i], n, 1);
    }
    if (!ok) { coli_park_free(p); return 0; }
    memcpy(p->fed, kvp->fed, (size_t)kvp->len * sizeof(int));
    p->len = kvp->len; p->kv_len = kv_len; p->n = n; p->on = 1;
    if (getenv("COLI_PREFIX_LOG"))
        fprintf(stderr, "[PREFIX] the %d-token conversation parked for a short request (%d rows kept apart)\n", p->len, n);
    return 1;
}

/* The request's turn is over, `written` the positions it holds (spec more a verify may have
 * written past them) and parts its state as it is now: the conversation's rows, tokens and
 * record back. 1 when they are back; 0 when the request wrote past the rows kept apart or the
 * caches moved (then the conversation is read again by its next turn). Frees the copy. */
static inline int coli_park_restore(ColiPark *p, kv_prefix *kvp, int *kv_len, int written, int spec,
                                    const ColiParkPart *parts, int nparts) {
    if (!p->on) return 0;
    int ok = written + spec <= p->n && nparts == p->nparts && kvp->cap >= p->len;
    for (int i = 0; ok && i < nparts; i++)
        ok = parts[i].base == p->part[i].base && parts[i].row == p->part[i].row && parts[i].rows == p->part[i].rows &&
             parts[i].heads == p->part[i].heads && parts[i].head_stride == p->part[i].head_stride;
    if (!ok) {
        if (getenv("COLI_PREFIX_LOG"))
            fprintf(stderr, "[PREFIX] the request wrote past the rows kept apart: the conversation will be read again\n");
        coli_park_free(p);
        return 0;
    }
    for (int i = 0; i < nparts; i++) if (p->copy[i]) coli_park_copy(&p->part[i], p->copy[i], p->n, 0);
    kv_prefix_clear(kvp);
    kv_prefix_record(kvp, p->fed, 0, p->len);
    *kv_len = p->kv_len;
    if (getenv("COLI_PREFIX_LOG")) fprintf(stderr, "[PREFIX] the %d-token conversation is back\n", p->len);
    coli_park_free(p);
    return 1;
}

#endif /* COLI_PARK_H */
