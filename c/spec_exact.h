/* spec_exact.h -- exact speculative sampling for drafts that are SAMPLED, not argmax.
 *
 * Every speculative loop in this tree drafted greedily: the draft is the argmax of
 * the drafter's logits, so its proposal distribution q is a point mass and the
 * lossless rule is "keep the draft with probability p(d), else draw from p with d
 * removed" (Leviathan et al.).  That rule keeps p(d) of the drafts, which on a flat
 * row is little even when the drafter knows the row well.  Drawing the draft from
 * the drafter's distribution q instead and keeping it with probability
 * min(1, p(d)/q(d)), else drawing from the residual max(p - q, 0) normalised, emits
 * tokens that follow p exactly as well, and keeps sum_i min(p_i, q_i) = 1 - TV(p, q)
 * of the drafts (Chen et al. 2023, "Accelerating LLM decoding with speculative
 * sampling").  Basalt 0.4.0 (MIT, src/kernels/cuda/sampler.cu, drawn_pick) runs this
 * rule for its block drafter; this is the same rule on the CPU, for colibri's drafters.
 *
 * Neither rule is always ahead.  The point mass keeps p(argmax q); the drawn draft
 * keeps sum min(p, q).  With a drafter flatter than the model (p = .9/.1, q = .5/.5)
 * the point mass keeps .9 and the drawn draft .6; with one about as sharp as the model
 * the drawn draft wins (tests/test_spec_exact.c measures both).  Both expectations
 * can be read off every verified row for free -- the verify computes p, the drafter
 * computed q -- so SpecSoftGate keeps a running mean of each and drafts with the rule
 * that has been keeping more.  Either way the tokens follow p: the gate only moves
 * the acceptance rate.  Greedy requests (temperature 0) never draw: there the draft
 * is the argmax and the check is equality, as before.
 *
 * Header-only, no globals; the caller hands in its own uniforms so each engine keeps
 * its RNG (sample.h's rndu, rand()).
 *
 *   COLI_SPEC_SOFT  auto (default where an engine enables it): the gate picks per row
 *                   0: always the point mass (the old behaviour, RNG use unchanged)
 *                   1: always draw drafts from q (tests, measurements)
 */
#ifndef SPEC_EXACT_H
#define SPEC_EXACT_H

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define SPEC_Q_MAX 32   /* a draft's q keeps its top 32 tokens: past that the mass is noise */

/* A drafter's proposal distribution for one row: ids by descending probability
 * (equal probabilities by ascending id), probabilities summing to 1. */
typedef struct { int n; int id[SPEC_Q_MAX]; float p[SPEC_Q_MAX]; } SpecQ;

static inline void specq_point(SpecQ *q, int tok) { q->n = 1; q->id[0] = tok; q->p[0] = 1.f; }

/* q = softmax(lo / temp), cut to its top kmax tokens and to the nucleus top_p (the cut
 * measured against the whole softmax, as the target sampler measures it), renormalised.
 * temp <= 0 (or a row with no finite logit) gives the point mass on the argmax. */
static void specq_from_logits(SpecQ *q, const float *lo, int V, float temp, float top_p, int kmax) {
    if (kmax > SPEC_Q_MAX) kmax = SPEC_Q_MAX;
    if (kmax < 1) kmax = 1;
    int amx = -1; float mx = 0.f;
    for (int i = 0; i < V; i++)
        if (isfinite(lo[i]) && (amx < 0 || lo[i] > mx)) { mx = lo[i]; amx = i; }
    if (amx < 0) { specq_point(q, 0); return; }
    if (!(temp > 0.f)) { specq_point(q, amx); return; }
    double kp[SPEC_Q_MAX]; int ki[SPEC_Q_MAX]; int n = 0;
    double sum = 0.0, inv = 1.0 / (double)temp;
    for (int i = 0; i < V; i++) {
        if (!isfinite(lo[i])) continue;
        double e = exp(((double)lo[i] - (double)mx) * inv);
        sum += e;
        if (n == kmax && e <= kp[n - 1]) continue;   /* equal: the lower id, already in, stays */
        int j = n < kmax ? n++ : n - 1;
        while (j > 0 && kp[j - 1] < e) { kp[j] = kp[j - 1]; ki[j] = ki[j - 1]; j--; }
        kp[j] = e; ki[j] = i;
    }
    double cut = (top_p > 0.f && top_p < 1.f) ? (double)top_p * sum : sum, kept = 0.0;
    int m = 0;
    while (m < n && kept < cut) kept += kp[m++];
    q->n = m;
    for (int j = 0; j < m; j++) { q->id[j] = ki[j]; q->p[j] = (float)(kp[j] / kept); }
}

/* The token at uniform u in [0, 1) of q. */
static inline int specq_draw(const SpecQ *q, double u) {
    double total = 0.0;
    for (int j = 0; j < q->n; j++) total += q->p[j];
    double thr = u * total, cum = 0.0;
    for (int j = 0; j < q->n; j++) { cum += q->p[j]; if (thr < cum) return q->id[j]; }
    return q->id[q->n - 1];
}

static inline double specq_prob(const SpecQ *q, int tok) {
    for (int j = 0; j < q->n; j++) if (q->id[j] == tok) return q->p[j];
    return 0.0;
}

/* Settle one drafted row.  p: the target distribution over the vocabulary, dense and
 * normalised, exactly what the engine's plain sampler draws from (temperature and
 * nucleus applied).  d: the draft, drawn from q.  Keeps d with probability
 * min(1, p(d)/q(d)) at u_keep; else returns the token at u_res of max(p - q, 0).
 * *kept says which.  On a rejection p is overwritten with the residual.  A draft
 * proposed from a point mass (specq_point) reduces to the old rule: keep with
 * probability p(d), else draw from p without d. */
static int spec_exact_settle(float *p, int V, const SpecQ *q, int d,
                             double u_keep, double u_res, int *kept) {
    double qd = specq_prob(q, d), pd = (d >= 0 && d < V) ? (double)p[d] : 0.0;
    if (qd > 0.0 && (pd >= qd || u_keep * qd < pd)) { *kept = 1; return d; }
    *kept = 0;
    for (int j = 0; j < q->n; j++) {
        int i = q->id[j];
        if (i < 0 || i >= V) continue;
        float r = p[i] - q->p[j];
        p[i] = r > 0.f ? r : 0.f;
    }
    double mass = 0.0;
    for (int i = 0; i < V; i++) mass += p[i];
    /* no residual mass means p <= q everywhere, i.e. p == q up to rounding, where the
     * draft is kept with probability 1 anyway */
    if (!(mass > 0.0)) { *kept = 1; return d; }
    double thr = u_res * mass, cum = 0.0; int last = d;
    for (int i = 0; i < V; i++) {
        if (p[i] <= 0.f) continue;
        cum += p[i]; last = i;
        if (thr < cum) return i;
    }
    return last;
}

/* How much of a row each rule would keep, from the row's p and the drafter's q:
 * the drawn draft sum_i min(p_i, q_i), the point mass p(argmax q). */
static inline double spec_exact_keep_drawn(const float *p, int V, const SpecQ *q) {
    double s = 0.0;
    for (int j = 0; j < q->n; j++) {
        int i = q->id[j];
        if (i < 0 || i >= V) continue;
        s += (double)p[i] < (double)q->p[j] ? (double)p[i] : (double)q->p[j];
    }
    return s;
}
static inline double spec_exact_keep_point(const float *p, int V, const SpecQ *q) {
    return (q->n > 0 && q->id[0] >= 0 && q->id[0] < V) ? (double)p[q->id[0]] : 0.0;
}

/* The target distribution as a nucleus sampler sees it: softmax(lo / temp), the
 * tokens by descending probability (equal ones by ascending id) kept while their sum
 * is under top_p of the total, renormalised; everything else 0.  Same rule as
 * qwen36_sample.h's and qwen38.c's serve_sample, so a settle draws from the
 * distribution those draw from.  Only the nucleus is sorted: probabilities are
 * bucketed by binary exponent first, as qwen36_sample.h does. */
typedef struct { float p; int id; } SpecExactRank;
static int spec_exact_rank_desc(const void *a, const void *b) {
    const SpecExactRank *x = a, *y = b;
    if (x->p != y->p) return (y->p > x->p) - (x->p > y->p);
    return (x->id > y->id) - (x->id < y->id);
}
static int spec_exact_target(float *p, const float *lo, int V, float temp, float top_p) {
    float mx = lo[0];
    for (int i = 1; i < V; i++) if (lo[i] > mx) mx = lo[i];
    double sum = 0.0;
    for (int i = 0; i < V; i++) { p[i] = expf((lo[i] - mx) / temp); sum += p[i]; }
    if (!(sum > 0.0) || !isfinite(sum)) return -1;
    if (!(top_p > 0.f && top_p < 1.f)) {
        for (int i = 0; i < V; i++) p[i] = (float)(p[i] / sum);
        return 0;
    }
    double cut = (double)top_p * sum, bucket[256] = {0};
    for (int i = 0; i < V; i++) { uint32_t b; memcpy(&b, &p[i], 4); bucket[255 - ((b >> 23) & 255)] += p[i]; }
    int last = 0; double total = 0.0;
    while (last < 256) { total += bucket[last]; if (total >= cut) break; last++; }
    if (last < 255) last++;
    if (last > 255) last = 255;
    SpecExactRank *rank = NULL;
    int m = 0, n = 0; double kept = 0.0;
    for (;;) {   /* a second pass over every token when rounding left the candidates short of the cut */
        int cap = 0;
        for (int i = 0; i < V; i++) { uint32_t b; memcpy(&b, &p[i], 4); if (255 - (int)((b >> 23) & 255) <= last) cap++; }
        rank = malloc((size_t)(cap > 0 ? cap : 1) * sizeof(*rank));
        if (!rank) return -1;
        m = 0;
        for (int i = 0; i < V; i++) {
            uint32_t b; memcpy(&b, &p[i], 4);
            if (255 - (int)((b >> 23) & 255) <= last) rank[m++] = (SpecExactRank){p[i], i};
        }
        qsort(rank, (size_t)m, sizeof(*rank), spec_exact_rank_desc);
        n = 0; kept = 0.0;
        while (n < m && kept < cut) kept += rank[n++].p;
        if (kept >= cut || m == V || last == 255) break;
        free(rank); last = 255;
    }
    for (int i = 0; i < V; i++) p[i] = 0.f;
    for (int i = 0; i < n; i++) p[rank[i].id] = (float)(rank[i].p / kept);
    free(rank);
    return 0;
}

/* Which rule drafts the next round.  Both expectations are read off every verified
 * row (spec_soft_gate_observe), whichever rule drafted it, so the gate compares like
 * with like and can switch either way.  The first rows use the plain mean, then an
 * exponential one (1/32) that follows a change of domain within a few verifies. */
typedef struct {
    int mode;                 /* 0 point, 1 drawn, 2 auto */
    uint64_t rows;            /* verified rows observed */
    double keep_drawn, keep_point;
    uint64_t drawn_drafts, drawn_kept;   /* drafts settled with the drawn rule, and kept */
} SpecSoftGate;

#define SPEC_SOFT_WARMUP 16

/* env: COLI_SPEC_SOFT (0, 1, auto); `fallback` is the mode an unset variable means */
static void spec_soft_gate_init(SpecSoftGate *g, int fallback) {
    memset(g, 0, sizeof *g);
    g->mode = fallback;
    const char *e = getenv("COLI_SPEC_SOFT");
    if (e && *e) g->mode = !strcmp(e, "auto") ? 2 : atoi(e) ? 1 : 0;
}
/* 1 = draw this round's drafts from q */
static inline int spec_soft_gate_drawn(const SpecSoftGate *g, float temp) {
    if (!(temp > 0.f) || g->mode == 0) return 0;
    if (g->mode == 1) return 1;
    return g->rows >= SPEC_SOFT_WARMUP && g->keep_drawn > g->keep_point;
}
static inline void spec_soft_gate_observe(SpecSoftGate *g, const float *p, int V, const SpecQ *q) {
    double a = spec_exact_keep_drawn(p, V, q), b = spec_exact_keep_point(p, V, q);
    g->rows++;
    double w = g->rows < 32 ? 1.0 / (double)g->rows : 1.0 / 32.0;
    g->keep_drawn += w * (a - g->keep_drawn);
    g->keep_point += w * (b - g->keep_point);
}

#endif /* SPEC_EXACT_H */
