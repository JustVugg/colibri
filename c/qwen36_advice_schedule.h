/* Advisory scheduling only: bounded CPU IDs and a submit-before-advice
 * ordering hook. Callers perform their existing GPU wait afterwards, without
 * encoding another command in between. No storage, worker or cache policy. */
#ifndef QWEN36_ADVICE_SCHEDULE_H
#define QWEN36_ADVICE_SCHEDULE_H
#include <string.h>

typedef struct { int layer, n, ids[8]; } QasPending;
typedef int (*QasSubmit)(void *);
typedef void (*QasEmit)(void *, int, const int *, int, int);

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
