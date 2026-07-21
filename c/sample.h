/* sample.h — sampling (temperature + nucleus) and stop-set management.
 * Header-only: all functions are static — include from the main engine file. */
#ifndef SAMPLE_H
#define SAMPLE_H

#include <math.h>
#include <stdio.h>
#include <string.h>
#include "tok.h"

/* ---- RNG (xorshift64*) -------------------------------------------------- */
static uint64_t g_rng = 0x9E3779B97F4A7C15ULL;
static inline double rndu(void){
    g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17;
    return (double)(g_rng >> 11) * (1.0 / 9007199254740992.0);
}

/* ---- argmax over a float vector ----------------------------------------- */
static inline int argmax_v(const float *lo, int V){
    int b=-1; float bv=-INFINITY;
    for(int i=0;i<V;i++){ float x=lo[i]; if(x==x && x>bv){ bv=x; b=i; } }
    return b<0?0:b;
}

/* ---- distribution buffers (reused, single-threaded decode) --------------- */
static float *g_pbuf = NULL;
static int   *g_pidx = NULL;

/* sift-down on max-heap in h[0..n), key = g_pbuf[h[i]] (#335: partial top-p).
 * "hole" variant: carries the root value and deposits only at the end, so
 * heapify is O(V) and each pop is O(log n) without qsort on the full vocab. */
static void topp_siftdown(int *h, int n, int i){
    int iv = h[i]; float kv = g_pbuf[iv];
    for (;;) {
        int l = 2*i + 1;
        if (l >= n) break;
        int b = l; if (l+1 < n && g_pbuf[h[l+1]] > g_pbuf[h[l]]) b = l+1;
        if (g_pbuf[h[b]] <= kv) break;
        h[i] = h[b]; i = b;
    }
    h[i] = iv;
}

/* build the target distribution in g_pbuf: softmax(lo/temp) truncated to
 * top-p g_nuc. Invariant: g_pbuf stays indexed by token-id (never reordered);
 * the truncated tail is zeroed (dist_sample reads by id directly).
 * Requires: g_temp, g_nuc, falloc() — declared in the main engine file. */
static void dist_build(const float *lo, int V){
    if (!g_pbuf) { g_pbuf = falloc(V); g_pidx = malloc(V * sizeof(int)); }
    int mxi = -1; float mx = 0;
    for (int i = 0; i < V; i++)
        if (isfinite(lo[i]) && (mxi < 0 || lo[i] > mx)) { mx = lo[i]; mxi = i; }
    double s = 0; float invt = 1.f / (g_temp > 1e-4f ? g_temp : 1e-4f);
    if (mxi >= 0) {
        for (int i = 0; i < V; i++) {
            g_pbuf[i] = isfinite(lo[i]) ? expf((lo[i] - mx) * invt) : 0.f;
            s += g_pbuf[i];
        }
    }
    if (mxi < 0 || !isfinite(s) || s <= 0.0) {
        static int warned = 0;
        if (!warned) { warned = 1; fprintf(stderr,
            "[SAMPLE] warning: non-finite logits (NaN/Inf) — falling back to argmax; "
            "output may be degraded. This usually means a numerical blow-up upstream.\n"); }
        int a = (mxi >= 0) ? mxi : 0;
        for (int i = 0; i < V; i++) g_pbuf[i] = 0.f;
        g_pbuf[a] = 1.f;
        return;
    }
    for (int i = 0; i < V; i++) g_pbuf[i] /= (float)s;
    if (g_nuc > 0 && g_nuc < 1.f) {
        for (int i = 0; i < V; i++) g_pidx[i] = i;
        for (int i = V/2-1; i >= 0; i--) topp_siftdown(g_pidx, V, i);
        double s2 = 0, cum = 0; int out = V;
        do {
            int root = g_pidx[0];
            g_pidx[0] = g_pidx[--out]; g_pidx[out] = root;
            s2 += g_pbuf[root]; cum += g_pbuf[root];
            if (out > 0) topp_siftdown(g_pidx, out, 0);
        } while (cum < g_nuc && out > 0);
        for (int i = 0; i < out; i++) g_pbuf[g_pidx[i]] = 0;
        float s2f = (float)s2;
        for (int i = out; i < V; i++) g_pbuf[g_pidx[i]] /= s2f;
    }
}

/* sample from g_pbuf; ban>=0 excludes that token (renormalizing on the fly) */
static int dist_sample(int V, int ban){
    double z = 1.0 - (ban >= 0 ? g_pbuf[ban] : 0.0);
    if (z <= 1e-12) z = 1e-12;
    double u = rndu() * z, cum = 0;
    for (int i = 0; i < V; i++) { if (i == ban) continue; cum += g_pbuf[i]; if (cum >= u) return i; }
    for (int i = V-1; i >= 0; i--) if (i != ban && g_pbuf[i] > 0) return i;
    return 0;
}

/* next token from logits: greedy if g_temp<=0, sampling otherwise.
 * ban = token excluded because it was rejected by speculative verification. */
static int pick_tok(const float *lo, int V, int ban){
    if (g_temp <= 0) return argmax_v(lo, V);
    dist_build(lo, V);
    return dist_sample(V, ban);
}

/* ---- stop set ----------------------------------------------------------- */
static int g_stop[64], g_nstop = 0;
static inline int is_stop(int t){
    for (int i = 0; i < g_nstop; i++) if (t == g_stop[i]) return 1;
    return 0;
}
/* T=NULL -> config stops only (validation/oracle, where the tokenizer is not needed). */
static void stops_arm_tok(const Cfg *c, int tok_eos, Tok *T){
    g_nstop = 0;
    for (int i = 0; i < c->n_stop && g_nstop < 64; i++) g_stop[g_nstop++] = c->stop_ids[i];
    if (tok_eos >= 0 && !is_stop(tok_eos) && g_nstop < 64) g_stop[g_nstop++] = tok_eos;
    int nsp = 0;
    if (T) for (int id = 0; id < T->n_ids && g_nstop < 64; id++)
        if (T->id_special[id] && !is_stop(id)) { g_stop[g_nstop++] = id; nsp++; }
    fprintf(stderr, "[stop] %d stop tokens:", g_nstop);
    for (int i = 0; i < g_nstop; i++) fprintf(stderr, " %d", g_stop[i]);
    if (nsp) fprintf(stderr, " (%d from the tokenizer's special set)", nsp);
    fprintf(stderr, "\n");
}
static void stops_arm(const Cfg *c, int tok_eos){ stops_arm_tok(c, tok_eos, NULL); }

/* ---- log-prob of a target token given the logit vector ------------------- */
static double logprob_target(const float *lo, int V, int target, int *am){
    float mx = lo[0]; int best = 0;
    for (int i = 1; i < V; i++) if (lo[i] > mx) { mx = lo[i]; best = i; }
    double se = 0;
    for (int i = 0; i < V; i++) se += exp((double)lo[i] - mx);
    if (am) *am = (best == target);
    return (double)(lo[target] - mx) - log(se);
}

/* "glm" in model_type, case-insensitive */
static int mt_is_glm(const char *s){
    if (s) for (; *s; s++)
        if ((s[0]|32) == 'g' && (s[1]|32) == 'l' && (s[2]|32) == 'm') return 1;
    return 0;
}

#endif /* SAMPLE_H */
