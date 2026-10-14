/* spec_draft.h -- what a speculative verify should carry, and whether drafting pays.
 *
 * An engine with a verify forward (several rows in one forward, each computed the way a
 * decode step computes it, so the rows that stand give the logits plain decoding gives)
 * can take its drafts from any source: its MTP head, or the context itself. This header
 * holds the parts that do not depend on the engine; it has no engine state and no clock.
 *
 *   spec_lookup   prompt lookup: the longest n-gram (max_n down to min_n tokens) that ends
 *                 the token history and also occurs earlier in it, at its most recent earlier
 *                 occurrence; the up to k tokens that followed it there are the proposal.
 *                 Code edits, quotes and repeated structure give long runs of accepted
 *                 drafts; free text gives few proposals, and short ones.
 *
 *   SpecGate      how many drafts a verify should carry. Per source it keeps the running
 *                 probability a_j that draft j is accepted when the drafts before it were,
 *                 the measured seconds T(n) of a forward of n rows, and the measured seconds
 *                 d a source spends on each draft past its first. The depth k it picks
 *                 maximizes the expected tokens per second of a verify against a plain step:
 *                     value(k) = E(k) / cost(k)
 *                     E(k)     = 1 + a_1 + a_1 a_2 + ... + a_1 ... a_k   (tokens it yields)
 *                     cost(k)  = (T(k+1) + (k-1) d) / T(1)              (in decode steps)
 *                 and drafts only when value(k) > 1 (k = 0 is a plain step, value 1). A row
 *                 count not measured yet costs T(1) (1 + (n-1) rho), rho fitted from the
 *                 multi-row forwards that ran (SPEC_RHO0 until one has). Every SPEC_PROBE
 *                 picks the gate tries one draft deeper than its best (so an estimate the
 *                 gate stopped feeding can recover; every SPEC_PROBE_NEW while that deeper
 *                 verify's cost has fewer than 3 measurements), and after SPEC_REFRESH
 *                 forwards with no plain step it takes one, so T(1) stays measured.
 *
 * Neither ever changes a token: every proposal goes through the engine's verify, which
 * keeps exactly what plain decoding produces. They change how many forwards that takes.
 *
 * Callers: qwen38_core.h (MTP drafts and prompt lookup, combined) and qwen36.c (prompt
 * lookup). docs/speculative.md describes how another engine with a verify path wires it. */
#ifndef COLI_SPEC_DRAFT_H
#define COLI_SPEC_DRAFT_H

#include <stdlib.h>
#include <string.h>

#define SPEC_MAX_DRAFTS 8          /* positions the gate tracks (an engine's verify may take fewer) */
#define SPEC_PROBE      16         /* every this many picks, one draft deeper than the best */
#define SPEC_PROBE_NEW  4          /* the same while that deeper row count is barely measured */
#define SPEC_REFRESH    64         /* forwards with no plain step before the gate takes one */
#define SPEC_RHO0       0.5        /* a row's extra cost, relative to one row, before any is measured */
#define SPEC_ACC_W      16.0       /* the acceptance's memory, in verifies */
#define SPEC_T_W        8.0        /* a forward time's memory, in forwards */

enum { SPEC_SRC_MTP = 0, SPEC_SRC_LOOKUP = 1, SPEC_NSRC = 2 };

typedef struct {
    double acc[SPEC_NSRC][SPEC_MAX_DRAFTS];   /* a_{j+1}: P(draft j+1 accepted | drafts 1..j accepted) */
    double acc_w[SPEC_NSRC][SPEC_MAX_DRAFTS]; /* weight behind it (a prior's pseudo-count first) */
    double t[SPEC_MAX_DRAFTS + 2];            /* T(n), seconds of a forward of n rows; t_w 0 = not measured */
    double t_w[SPEC_MAX_DRAFTS + 2];
    double rho, rho_w;                        /* a row's extra cost relative to one row */
    double d[SPEC_NSRC], d_w[SPEC_NSRC];      /* seconds a source spends per draft past its first */
    int picks[SPEC_NSRC];                     /* picks so far (the probe's clock) */
    int since_plain;                          /* forwards since the last plain step */
    int off;                                  /* 1: every proposal drafted in full (tests, fixed depths) */
    /* the run's counts, for the engines' report lines */
    unsigned long long prop[SPEC_NSRC][SPEC_MAX_DRAFTS], hit[SPEC_NSRC][SPEC_MAX_DRAFTS];
    unsigned long long verifies[SPEC_NSRC], probes[SPEC_NSRC], declined[SPEC_NSRC];
} SpecGate;

static inline void spec_gate_init(SpecGate *g, int off) {
    memset(g, 0, sizeof *g);
    g->off = off;
    for (int j = 0; j < SPEC_MAX_DRAFTS; j++) {
        /* priors, worth two verifies each: an MTP head is trained to be right, and each
         * draft past the first reads the head's own state instead of the model's */
        g->acc[SPEC_SRC_MTP][j] = j == 0 ? 0.8 : j == 1 ? 0.6 : 0.5;
        g->acc[SPEC_SRC_LOOKUP][j] = 0.5;
        g->acc_w[SPEC_SRC_MTP][j] = g->acc_w[SPEC_SRC_LOOKUP][j] = 2.0;
    }
    g->rho = SPEC_RHO0; g->rho_w = 1.0;
}

static inline void spec_ema(double *x, double *w, double v, double memory) {
    if (*w < memory) *w += 1.0;
    *x += (v - *x) / *w;
}

/* A forward of `rows` rows took `seconds` (a plain step, or a verify). */
static inline void spec_gate_forward(SpecGate *g, int rows, double seconds) {
    if (rows < 1 || rows > SPEC_MAX_DRAFTS + 1 || !(seconds > 0.0)) return;
    spec_ema(&g->t[rows], &g->t_w[rows], seconds, SPEC_T_W);
    if (rows == 1) { g->since_plain = 0; return; }
    g->since_plain++;
    if (g->t_w[1] > 0.0 && g->t[1] > 0.0) {
        double r = (seconds / g->t[1] - 1.0) / (rows - 1);
        if (r < 0.0) r = 0.0;
        if (r > 4.0) r = 4.0;
        spec_ema(&g->rho, &g->rho_w, r, SPEC_T_W);
    }
}

/* A source spent `seconds` on one draft past its first (an MTP head's extra step). */
static inline void spec_gate_draft_cost(SpecGate *g, int src, double seconds) {
    if (src < 0 || src >= SPEC_NSRC || !(seconds >= 0.0)) return;
    spec_ema(&g->d[src], &g->d_w[src], seconds, SPEC_T_W);
}

/* A verify's drafts from `src` were judged: the first `accepted` stood, and `judged` of
 * them were looked at (accepted + 1 after a rejection; accepted when every draft stood, or
 * when the generation ended before the rest were reached: those say nothing). */
static inline void spec_gate_result(SpecGate *g, int src, int judged, int accepted) {
    if (src < 0 || src >= SPEC_NSRC || judged < 1) return;
    if (judged > SPEC_MAX_DRAFTS) judged = SPEC_MAX_DRAFTS;
    if (accepted > judged) accepted = judged;
    g->verifies[src]++;
    for (int j = 0; j < judged; j++) {
        g->prop[src][j]++;
        if (j < accepted) g->hit[src][j]++;
        spec_ema(&g->acc[src][j], &g->acc_w[src][j], j < accepted ? 1.0 : 0.0, SPEC_ACC_W);
    }
}

/* T(n) in units of T(1): measured, or the fitted line. */
static inline double spec_gate_rows_cost(const SpecGate *g, int rows) {
    if (rows <= 1) return 1.0;
    if (rows <= SPEC_MAX_DRAFTS + 1 && g->t_w[rows] > 0.0 && g->t_w[1] > 0.0 && g->t[1] > 0.0)
        return g->t[rows] / g->t[1];
    return 1.0 + (rows - 1) * g->rho;
}

/* The expected tokens a verify of k drafts from src yields, and that per unit of time
 * against a plain step (value 1). */
static inline double spec_gate_value(const SpecGate *g, int src, int k, double *expect) {
    double e = 1.0, p = 1.0;
    for (int j = 0; j < k && j < SPEC_MAX_DRAFTS; j++) { p *= g->acc[src][j]; e += p; }
    double cost = spec_gate_rows_cost(g, k + 1);
    if (k > 1 && g->d_w[src] > 0.0 && g->t_w[1] > 0.0 && g->t[1] > 0.0) cost += (k - 1) * g->d[src] / g->t[1];
    if (expect) *expect = e;
    return k > 0 ? e / cost : 1.0;
}

/* How many of the up to maxk drafts src can offer the next verify should carry: 0 for a
 * plain step. *value gets that depth's value (1 for a plain step). */
static inline int spec_gate_pick(SpecGate *g, int src, int maxk, double *value) {
    if (value) *value = 1.0;
    if (maxk <= 0 || src < 0 || src >= SPEC_NSRC) return 0;
    if (maxk > SPEC_MAX_DRAFTS) maxk = SPEC_MAX_DRAFTS;
    if (g->off) { if (value) *value = spec_gate_value(g, src, maxk, NULL); return maxk; }
    if (g->since_plain >= SPEC_REFRESH) { g->declined[src]++; return 0; }   /* keep T(1) measured */
    int best = 0; double bv = 1.0;
    for (int k = 1; k <= maxk; k++) {
        double v = spec_gate_value(g, src, k, NULL);
        if (v > bv) { bv = v; best = k; }
    }
    /* explore one deeper: every SPEC_PROBE picks, every SPEC_PROBE_NEW while that row
     * count has fewer than 3 measurements (its cost is still the fitted line's guess) */
    int every = best < maxk && g->t_w[best + 2] < 3.0 ? SPEC_PROBE_NEW : SPEC_PROBE;
    if (++g->picks[src] % every == 0 && best < maxk) {
        best++; bv = spec_gate_value(g, src, best, NULL); g->probes[src]++;
    }
    if (!best) g->declined[src]++;
    if (value) *value = bv;
    return best;
}

/* What the gate measured, for a report line: T(1), the cost of the row counts seen (in
 * units of T(1)), the fitted rho, a source's cost per extra draft and its acceptance
 * estimates by position. */
#include <stdio.h>
static inline void spec_gate_describe(const SpecGate *g, int src, char *buf, size_t cap) {
    size_t at = 0;
    if (!cap) return;
    buf[0] = 0;
#define SPEC_PUT(...) do { int w_ = snprintf(buf + at, cap - at, __VA_ARGS__); \
                           if (w_ < 0 || (size_t)w_ >= cap - at) return; at += (size_t)w_; } while (0)
    SPEC_PUT("T(1) %.1f ms, rows", g->t_w[1] > 0.0 ? 1e3 * g->t[1] : 0.0);
    for (int n = 2; n <= SPEC_MAX_DRAFTS + 1; n++)
        if (g->t_w[n] > 0.0) SPEC_PUT(" %d:%.2f", n, spec_gate_rows_cost(g, n));
    SPEC_PUT(", rho %.2f", g->rho);
    if (g->d_w[src] > 0.0) SPEC_PUT(", %.1f ms per extra draft", 1e3 * g->d[src]);
    SPEC_PUT(", acceptance by position");
    for (int j = 0; j < 5; j++) SPEC_PUT(" %.2f", g->acc[src][j]);
#undef SPEC_PUT
}

/* Prompt lookup: up to k tokens proposed for what follows h[0..len), from the most recent
 * earlier occurrence of the longest suffix of n tokens, max_n >= n >= min_n, that has one
 * with at least one token after it. Returns how many were written to out. */
static inline int spec_lookup(const int *h, int len, int min_n, int max_n, int k, int *out) {
    if (!h || !out || k < 1 || min_n < 1 || len < min_n + 1) return 0;
    if (max_n > len - 1) max_n = len - 1;
    for (int n = max_n; n >= min_n; n--) {
        const int *suf = h + len - n;
        for (int i = len - n - 1; i >= 0; i--) {   /* h[i..i+n) against the suffix, h[i+n] after it */
            if (h[i + n - 1] != suf[n - 1] || memcmp(h + i, suf, (size_t)n * sizeof(int))) continue;
            int g = 0;
            for (int j = i + n; j < len && g < k; j++) out[g++] = h[j];
            return g;
        }
    }
    return 0;
}

#endif /* COLI_SPEC_DRAFT_H */
