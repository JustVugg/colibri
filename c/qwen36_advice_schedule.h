/* Advisory scheduling only: which guessed experts a decode step announces,
 * bounded CPU IDs and a submit-before-advice ordering hook. Callers perform
 * their existing GPU wait afterwards, without encoding another command in
 * between. No storage, worker or cache policy. */
#ifndef QWEN36_ADVICE_SCHEDULE_H
#define QWEN36_ADVICE_SCHEDULE_H
#include <string.h>

typedef struct { int layer, n, ids[8]; } QasPending;
typedef int (*QasSubmit)(void *);
typedef void (*QasEmit)(void *, int, const int *, int, int);

/* A decode step announces the next layer's six likeliest guesses
 * (qg_advise_ahead in qwen36_graph.h): on the 35B model the seventh and
 * eighth were chosen too seldom for their reads to pay on an 8 GB M1
 * (docs/experiments/qwen36-ahead-six-2026-10-06). */
#define QG_AHEAD_GUESSES 6

static int qas_ahead_guesses(int topk) { return topk < QG_AHEAD_GUESSES ? topk : QG_AHEAD_GUESSES; }

/* The `want` largest of n router logits, most likely first (a tie keeps the
 * lower index), into top[]; want is at most 8. Returns how many. */
static int qas_top(const float *logits, int n, int want, int top[8]) {
    int k = 0;
    if (want > 8) want = 8;
    if (want < 1) return 0;
    for (int e = 0; e < n; e++) {
        int i;
        if (k < want) i = k++;
        else if (logits[e] > logits[top[want - 1]]) i = want - 1;
        else continue;
        while (i > 0 && logits[e] > logits[top[i - 1]]) { top[i] = top[i - 1]; i--; }
        top[i] = e;
    }
    return k;
}

static void qas_clear(QasPending *p) { p->n = 0; p->layer = -1; }

static int qas_save(QasPending *p, int layer, const int *ids, int n) {
    qas_clear(p);
    if (!ids || layer < 0 || n < 1 || n > 8) return 0;
    memcpy(p->ids, ids, (size_t)n * sizeof(*ids));
    p->layer = layer; p->n = n;
    return 1;
}

static int qas_submit_advice(void *context, QasSubmit submit, QasEmit emit,
                             int layer, const int *ids, int from, int to) {
    if (!submit(context)) return 0;
    emit(context, layer, ids, from, to);
    return 1;
}

static int qas_submit_pending(QasPending *p, int layer, void *context,
                              QasSubmit submit, QasEmit emit) {
    if (!p->n) return 1;
    int n = p->n, target = p->layer;
    qas_clear(p);                         /* one-shot even if submit fails */
    if (target != layer) return 1;        /* never issue a stale layer hint */
    return qas_submit_advice(context, submit, emit, target, p->ids, 0, n);
}
#endif
