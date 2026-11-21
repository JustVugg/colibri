/* qwen38_pg.h -- Q38_PREFILL_GPU_PERCARD's placement helpers (a worker thread a card for the experts that a prompt
 * chunk, or a step of several conversations, has resident on the cards). Kept apart from qwen38_core.h so that
 * tests/test_qwen38_pg_percard.c can hold them without the engine. Placement only: which card or the CPU computes an
 * expert never changes its arithmetic (the cards' FP8 kernels give the CPU's bits; q38_prefill_gpu_exact). */
#ifndef QWEN38_PG_H
#define QWEN38_PG_H
#include <stddef.h>

#define Q38_PG_MAX_CARDS 8   /* as QT_MAX_DEV */

/* The cards holding resident experts in this list (home[u] the card of expert u, -1 none; cards past the limit are
 * not counted: their experts go to the CPU) */
static int q38_pg_cards(const int *home, int n) {
    int present[Q38_PG_MAX_CARDS] = {0}, nc = 0;
    for (int u = 0; u < n; u++) {
        const int h = home[u];
        if (h >= 0 && h < Q38_PG_MAX_CARDS && !present[h]) { present[h] = 1; nc++; }
    }
    return nc;
}

/* The balance with a worker a card: the cards together take `want` rows of the chunk (the share their combined rows a
 * second earn against the CPU's), split equally between the cards that hold experts here, except that a card with
 * fewer resident rows than its part keeps all of them and the rest of its part goes to the others (water-filling);
 * each card's experts with the most rows first, an expert past its card's part to the CPU (home -1). counts[u]: expert
 * u's rows; order: scratch of n ints. The ordering is the single-worker balance's (rows descending, stable). */
static void q38_pg_balance_percard(int *home, const int *counts, int n, int *order, double want) {
    for (int u = 0; u < n; u++) order[u] = u;
    for (int i = 1; i < n; i++) {   /* insertion: at most a few hundred */
        const int v = order[i]; int j = i - 1;
        while (j >= 0 && counts[order[j]] < counts[v]) { order[j + 1] = order[j]; j--; }
        order[j + 1] = v;
    }
    long avail[Q38_PG_MAX_CARDS] = {0}, taken[Q38_PG_MAX_CARDS] = {0};
    int present[Q38_PG_MAX_CARDS] = {0};
    for (int u = 0; u < n; u++) {
        const int h = home[u];
        if (h >= 0 && h < Q38_PG_MAX_CARDS) { present[h] = 1; avail[h] += counts[u]; }
    }
    int idx[Q38_PG_MAX_CARDS], nc = 0;
    for (int h = 0; h < Q38_PG_MAX_CARDS; h++) if (present[h]) idx[nc++] = h;
    for (int i = 1; i < nc; i++) {   /* by resident rows, ascending */
        const int v = idx[i]; int j = i - 1;
        while (j >= 0 && avail[idx[j]] > avail[v]) { idx[j + 1] = idx[j]; j--; }
        idx[j + 1] = v;
    }
    double part[Q38_PG_MAX_CARDS] = {0}, left = want > 0 ? want : 0.0;
    for (int i = 0; i < nc; i++) {
        const int h = idx[i];
        const double fair = left / (double)(nc - i);
        part[h] = (double)avail[h] < fair ? (double)avail[h] : fair;
        left -= part[h];
    }
    for (int i = 0; i < n; i++) {
        const int u = order[i], h = home[u];
        if (h < 0) continue;
        if (h >= Q38_PG_MAX_CARDS || (double)taken[h] >= part[h]) home[u] = -1;   /* past its card's part: the CPU's */
        else taken[h] += counts[u];
    }
}

/* The workers' ranges of the grouped list (experts grouped by card, devs[i] the card of entry i, counts[i] its rows):
 * one range a run of equal cards, its first entry, its length and the rows before it (where its inputs and outputs
 * start in the shared scratch). Returns the number of ranges, at most `max`; entries past them get no worker (their
 * experts fall back to the CPU after the join). */
static int q38_pg_ranges(const int *devs, const int *counts, int n, int *start, int *len, size_t *rbase, int max) {
    int nr = 0; size_t rows = 0;
    for (int i0 = 0; i0 < n && nr < max;) {
        int i1 = i0; size_t rr = 0;
        while (i1 < n && devs[i1] == devs[i0]) { rr += (size_t)counts[i1]; i1++; }
        start[nr] = i0; len[nr] = i1 - i0; rbase[nr] = rows; nr++;
        rows += rr; i0 = i1;
    }
    return nr;
}

#endif /* QWEN38_PG_H */
