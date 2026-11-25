/* tests/test_dflash_select.c -- DFlash2's host path selector (dflash_select.h):
 *   - greedy paths equal the selector dflash_cuda.h had before extraction (copied below), bit
 *     for bit, drafts and probabilities, with and without q requested;
 *   - each position's q is softmax(score / temp) over the valid candidates, the greedy pick first;
 *   - drawn paths follow q: position 0 by frequency, position 1 conditioned on position 0's draw;
 *   - a drawn path settled by spec_exact.h against a target emits the target's joint law. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../dflash_select.h"

static int g_fail = 0;
#define CHECK(c, ...) do { if (!(c)) { g_fail++; printf("FAIL: " __VA_ARGS__); printf("\n"); } } while (0)

static uint64_t g_s = 0x9E3779B97F4A7C15ULL;
static double U(void) { g_s ^= g_s << 13; g_s ^= g_s >> 7; g_s ^= g_s << 17; return (double)(g_s >> 11) * (1.0 / 9007199254740992.0); }
static double gauss(void) { double a = U() + 1e-12, b = U(); return sqrt(-2 * log(a)) * cos(6.283185307179586 * b); }
static uint16_t bf16(float f) { uint32_t u; memcpy(&u, &f, 4); return (uint16_t)(u >> 16); }

enum { V = 40, K = 16, RANK = 8, T = 4 };
static float tp[T][2 * K], hg[T][RANK];
static uint16_t A[V][RANK], Bk[V][RANK];

/* the selector as dflash_cuda.h had it (greedy only) */
static void old_select(int tok, int k, int *out, float *prob) {
    int prev = tok;
    for (int t = 0; t < k; t++) {
        const float *v = tp[t], *gt = hg[t];
        float ag[RANK];
        for (int j = 0; j < RANK; j++) ag[j] = df_sel_bf16(A[prev][j]) * gt[j];
        int best = 0; float bs = -INFINITY, z = 0.f;
        for (int c = 0; c < K; c++) {
            int id; memcpy(&id, v + K + c, 4);
            if (id < 0 || id >= V) continue;
            float s = v[c];
            for (int j = 0; j < RANK; j++) s += ag[j] * df_sel_bf16(Bk[id][j]);
            if (s > bs) { bs = s; best = c; }
        }
        for (int c = 0; c < K; c++) z += expf(v[c] - v[0]);
        int id; memcpy(&id, v + K + best, 4);
        out[t] = id; prob[t] = z > 0.f ? expf(v[best] - v[0]) / z : 0.f;
        prev = id;
    }
}

static void fill(int invalid) {
    for (int i = 0; i < V; i++) for (int j = 0; j < RANK; j++) { A[i][j] = bf16((float)gauss()); Bk[i][j] = bf16((float)(0.5 * gauss())); }
    for (int t = 0; t < T; t++) {
        int perm[V]; for (int i = 0; i < V; i++) perm[i] = i;
        for (int i = V - 1; i > 0; i--) { int j = (int)(U() * (i + 1)); int x = perm[i]; perm[i] = perm[j]; perm[j] = x; }
        float top = 3.f;
        for (int c = 0; c < K; c++) {
            top -= (float)(0.3 * U());
            tp[t][c] = top;
            int id = (invalid && c == 5) ? V + 3 : perm[c];
            memcpy(&tp[t][K + c], &id, 4);
        }
        for (int j = 0; j < RANK; j++) hg[t][j] = (float)(0.4 * gauss());
    }
}

static double scores(int t, int prev, double *sc, int *ids) {   /* reference q's scores */
    double best = -INFINITY;
    for (int c = 0; c < K; c++) {
        int id; memcpy(&id, &tp[t][K + c], 4); ids[c] = id;
        if (id < 0 || id >= V) { sc[c] = -INFINITY; continue; }
        float ag, s = tp[t][c];
        for (int j = 0; j < RANK; j++) { ag = df_sel_bf16(A[prev][j]) * hg[t][j]; s += ag * df_sel_bf16(Bk[id][j]); }
        sc[c] = s; if (s > best) best = s;
    }
    return best;
}

int main(void) {
    int out[T], o2[T]; float pr[T], p2[T]; SpecQ qs[T];
    /* greedy: bit-identical to the old selector */
    int same = 1;
    for (int trial = 0; trial < 200; trial++) {
        fill(trial % 3 == 0);
        int tok = (int)(U() * V);
        old_select(tok, T, out, pr);
        df_select_path(&tp[0][0], &hg[0][0], &A[0][0], &Bk[0][0], RANK, K, V, tok, T, o2, p2, NULL, 0, 0.f, NULL);
        same &= !memcmp(out, o2, sizeof out) && !memcmp(pr, p2, sizeof pr);
        df_select_path(&tp[0][0], &hg[0][0], &A[0][0], &Bk[0][0], RANK, K, V, tok, T, o2, p2, qs, 0, 0.7f, U);
        same &= !memcmp(out, o2, sizeof out) && !memcmp(pr, p2, sizeof pr);
        for (int t = 0; t < T; t++) same &= qs[t].id[0] == out[t];
    }
    printf("greedy path: %s over 200 random blocks (with and without q)\n", same ? "bit-identical" : "DIFFERS");
    CHECK(same, "greedy path differs from the old selector");

    /* q = softmax(score / temp) over the valid candidates */
    fill(1);
    float temp = 0.8f; int tok = 7;
    df_select_path(&tp[0][0], &hg[0][0], &A[0][0], &Bk[0][0], RANK, K, V, tok, 1, o2, p2, qs, 0, temp, U);
    double sc[K]; int ids[K]; double best = scores(0, tok, sc, ids), sum = 0, err = 0;
    for (int c = 0; c < K; c++) if (sc[c] > -INFINITY) sum += exp((sc[c] - best) / temp);
    for (int c = 0; c < K; c++) if (sc[c] > -INFINITY) err = fmax(err, fabs(specq_prob(&qs[0], ids[c]) - exp((sc[c] - best) / temp) / sum));
    printf("q of position 0: %d candidates, max |q - softmax(score/T)| %.2e\n", qs[0].n, err);
    CHECK(qs[0].n == K - 1 && err < 1e-6, "q differs from softmax of the scores");

    /* drawn: position 0 by frequency, position 1 conditioned on position 0 */
    enum { N = 400000 };
    static double h0[V], h01[V][V];
    for (int n = 0; n < N; n++) {
        df_select_path(&tp[0][0], &hg[0][0], &A[0][0], &Bk[0][0], RANK, K, V, tok, 2, o2, p2, qs, 1, temp, U);
        h0[o2[0]] += 1; h01[o2[0]][o2[1]] += 1;
    }
    double tv0 = 0;
    SpecQ q0; df_select_path(&tp[0][0], &hg[0][0], &A[0][0], &Bk[0][0], RANK, K, V, tok, 1, o2, p2, &q0, 0, temp, U);
    for (int i = 0; i < V; i++) tv0 += fabs(h0[i] / N - specq_prob(&q0, i));
    tv0 /= 2;
    /* position 1 given the most frequent first draft */
    int a = 0; for (int i = 1; i < V; i++) if (h0[i] > h0[a]) a = i;
    double sc1[K]; int ids1[K]; double b1 = scores(1, a, sc1, ids1), s1 = 0, tv1 = 0;
    for (int c = 0; c < K; c++) if (sc1[c] > -INFINITY) s1 += exp((sc1[c] - b1) / temp);
    for (int c = 0; c < K; c++) if (sc1[c] > -INFINITY) tv1 += fabs(h01[a][ids1[c]] / h0[a] - exp((sc1[c] - b1) / temp) / s1);
    tv1 /= 2;
    printf("drawn path: TV(position 0, q) %.4f, TV(position 1 | %d, q) %.4f\n", tv0, a, tv1);
    CHECK(tv0 < 0.006 && tv1 < 0.012, "drawn path does not follow q");

    /* settled against a target: the emitted pair follows p1(x) p2(y | x) */
    float pt0[V], pt1[V][V], s[V];
    for (int i = 0; i < V; i++) pt0[i] = (float)(1.5 * gauss());
    spec_exact_target(pt0, pt0, V, 1.f, 1.f);
    for (int x = 0; x < V; x++) { for (int i = 0; i < V; i++) pt1[x][i] = (float)(1.5 * gauss()); spec_exact_target(pt1[x], pt1[x], V, 1.f, 1.f); }
    static double joint[V][V]; long kept = 0;
    for (int n = 0; n < N; n++) {
        df_select_path(&tp[0][0], &hg[0][0], &A[0][0], &Bk[0][0], RANK, K, V, tok, 2, o2, p2, qs, 1, temp, U);
        int k; memcpy(s, pt0, sizeof s);
        int x = spec_exact_settle(s, V, &qs[0], o2[0], U(), U(), &k), y;
        kept += k;
        if (k) { memcpy(s, pt1[x], sizeof s); y = spec_exact_settle(s, V, &qs[1], o2[1], U(), U(), &k); }
        else { double u = U(), c = 0; y = V - 1; for (int i = 0; i < V; i++) { c += pt1[x][i]; if (u < c) { y = i; break; } } }
        joint[x][y] += 1;
    }
    double tvj = 0;
    for (int x = 0; x < V; x++) for (int y = 0; y < V; y++) tvj += fabs(joint[x][y] / N - (double)pt0[x] * pt1[x][y]);
    tvj /= 2;
    printf("drawn path settled: TV(joint, target) %.4f, first draft kept %.3f\n", tvj, (double)kept / N);
    CHECK(tvj < 0.03, "settled drawn path does not follow the target");
    if (g_fail) { printf("test_dflash_select: %d FAILED\n", g_fail); return 1; }
    printf("test_dflash_select: ok\n");
    return 0;
}
