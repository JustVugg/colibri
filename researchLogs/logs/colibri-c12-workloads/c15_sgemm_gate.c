/* c15_sgemm_gate.c -- the C15 GEMM re-shape's bench-level go/no-go gate (plan doc,
 * C15 section "Sequencing" bullet): the scalar prefill-attention loop the engine
 * runs today (q36_kv_dot scores + kv16_axpy context, the exact kv16.h kernels)
 * versus an Accelerate cblas_sgemm re-shape of the same work, on the model's real
 * attention shapes, with score/context maxdiff measured on real fp16-grid data.
 *
 *   gate: sgemm combined (scores+softmax+context+aux widening/packing) >= 2x faster
 *         than the scalar loop combined, AND score maxdiff (relative to the row's
 *         score magnitude) ~= 1e-3 or below -> proceed to the in-tree
 *         Q36_ATTN_SGEMM implementation; otherwise record as a negative result.
 *
 * Shapes are the model's: head_dim 256, H=16 q-heads / KVH=2 kv-heads
 * (q_per_kv=8), causal prefill (row s attends t <= s, pos_base 0, one chunk).
 * The context pass is [rows x T] . [T x 256] -- the GEMV-heavy direction, timed
 * separately from the scores pass [rows x 256] . [256 x T]^T as the re-scope's
 * open question (a) requires.
 *
 * The scalar cells replicate the engine's loop exactly: per (q-head, row) an
 * ascending-t kv16_dot_lanes over the fp16 K rows (scale applied after), an
 * engine-identical softmax_row, then an ascending-t kv16_axpy accumulation into
 * ctx. The sgemm cells replicate the re-scope sketch: K/V widened fp16->f32 once
 * per pass (timed), queries packed per q-head (timed), causal row-blocks of
 * C15_BLOCK rows: sgemm scores -> per-row softmax (scale, zeros past the
 * diagonal) -> sgemm context.
 *
 * Build (from repo root):
 *   clang -O3 -Xclang -fopenmp -I/opt/homebrew/opt/libomp/include \
 *     researchLogs/logs/colibri-c12-workloads/c15_sgemm_gate.c -o /tmp/c15gate \
 *     -Ic -lm -framework Accelerate -L/opt/homebrew/opt/libomp/lib -lomp
 * Runs: OMP_NUM_THREADS=6 VECLIB_MAXIMUM_THREADS=6 (the serving config) and
 * VECLIB_MAXIMUM_THREADS unset (Accelerate's own all-core pool) for the ceiling.
 */
#include "kv16.h"   /* kv16_dot_lanes / kv16_axpy / kv16_pack / kv16_unpack: the engine's exact scalar kernels */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <Accelerate/Accelerate.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

#define HD      256
#define H       16
#define KVH     2
#define Q_PER_KV 8
#define C15_BLOCK 512   /* causal row-block for the sgemm path (re-scope sketch) */

static double now(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}
static uint64_t rs = 0x9E3779B97F4A7C15ULL;
static inline float frand(void) {   /* [-1,1), deterministic */
    rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17;
    return (float)((double)(int64_t)(rs >> 40) / 8388608.0 - 1.0);
}
static void *xalloc(size_t n) {
    void *p; if (posix_memalign(&p, 64, n)) { fprintf(stderr, "OOM %zu\n", n); exit(1); }
    return p;
}
/* the engine's softmax_row (qwen36.c) */
static void softmax_row(float *x, int n) {
    float m = -1e30f; for (int i = 0; i < n; i++) if (x[i] > m) m = x[i];
    float s = 0; for (int i = 0; i < n; i++) { x[i] = expf(x[i]-m); s += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= s;
}

/* ---- the two pipelines ------------------------------------------------------ */

typedef struct {
    double sc, sm, cx, aux;   /* scalar scores / softmax / context; sgemm aux (widen+pack) */
    double t_scores, t_ctx, t_aux;   /* sgemm stage times */
} Times;

/* scalar: exactly the engine's per-(hh,s) work, OMP over (hh,s) like the engine's
 * collapse(2). probs_out (may be NULL) receives the scaled probabilities of head
 * `only_hh` (all heads if -1) for the maxdiff/context cells. */
static double scalar_pipeline(int S, const float *Q, const uint16_t *Kg,
                              const uint16_t *Vg, float *probs_out, int only_hh,
                              double *t_sc, double *t_sm, double *t_cx, double *chk) {
    const float scale = 1.f / sqrtf((float)HD);
    int nth = 1;
#if defined(_OPENMP)
    nth = omp_get_max_threads();
#endif
    float *scr = xalloc((size_t)nth * S * sizeof(float));    /* per-thread score row */
    float *ctx = xalloc((size_t)nth * HD * sizeof(float));   /* per-thread ctx row */
    double c_sc = 0, c_sm = 0, c_cx = 0; uint64_t sum = 0;
    double t0 = now();
#if defined(_OPENMP)
#pragma omp parallel for collapse(2) schedule(static) reduction(+:c_sc,c_sm,c_cx)
#endif
    for (int hh = 0; hh < H; hh++) {
        for (int s = 0; s < S; s++) {
#if defined(_OPENMP)
            int tid = omp_get_thread_num();
#else
            int tid = 0;
#endif
            if (only_hh >= 0 && hh != only_hh) continue;
            int kvh = hh / Q_PER_KV;
            const float *qv = Q + ((size_t)s * H + hh) * HD;
            const uint16_t *Kk = Kg + (size_t)kvh * S * HD;   /* T == S here */
            const uint16_t *Vk = Vg + (size_t)kvh * S * HD;
            float *sc = scr + (size_t)tid * S;
            double a0 = now();
            for (int t = 0; t <= s; t++)
                sc[t] = kv16_dot_lanes(qv, Kk + (size_t)t * HD, HD) * scale;
            c_sc += now() - a0;
            float *cx = ctx + (size_t)tid * HD;
            double a1 = now();
            softmax_row(sc, s + 1);
            c_sm += now() - a1;   /* NOTE: thread-seconds across the OMP team (reported /nth) */
            double a2 = now();
            for (int d = 0; d < HD; d++) cx[d] = 0.f;
            for (int t = 0; t <= s; t++)
                kv16_axpy(cx, sc[t], Vk + (size_t)t * HD, HD);
            c_cx += now() - a2;
            sum += (uint64_t)(cx[0] * 4096.f) + (uint64_t)(cx[HD-1] * 4096.f);
            if (probs_out && hh == only_hh)
                memcpy(probs_out + (size_t)s * S, sc, (size_t)(s + 1) * sizeof(float));
        }
    }
    double el = now() - t0;
    free(scr); free(ctx);
    *t_sc = c_sc; *t_sm = c_sm; *t_cx = c_cx;
    *chk += (double)sum;
    return el;
}

/* sgemm re-shape: widen K,V once (timed into aux), pack Q per head (aux), then
 * causal row-blocks: sgemm scores -> softmax in place (scale, zero tail) ->
 * sgemm context. probs_out receives head only_hh's probabilities like above. */
static double sgemm_pipeline(int S, const float *Q, const uint16_t *Kg,
                             const uint16_t *Vg, float *probs_out, int only_hh,
                             double *t_aux, double *t_sc, double *t_sm, double *t_cx,
                             double *chk) {
    const float scale = 1.f / sqrtf((float)HD);
    /* widen the two KV heads' K and V grids (T x HD each) */
    float *Kf = xalloc((size_t)KVH * S * HD * sizeof(float));
    float *Vf = xalloc((size_t)KVH * S * HD * sizeof(float));
    double a0 = now();
    for (int kvh = 0; kvh < KVH; kvh++) {
        kv16_unpack(Kf + (size_t)kvh * S * HD, Kg + (size_t)kvh * S * HD, (int64_t)S * HD);
        kv16_unpack(Vf + (size_t)kvh * S * HD, Vg + (size_t)kvh * S * HD, (int64_t)S * HD);
    }
    /* pack the queries per head: QP[hh][s*HD] contiguous rows */
    float *QP = xalloc((size_t)H * S * HD * sizeof(float));
    for (int s = 0; s < S; s++)
        for (int hh = 0; hh < H; hh++)
            memcpy(QP + ((size_t)hh * S + s) * HD, Q + ((size_t)s * H + hh) * HD, HD * sizeof(float));
    double aux = now() - a0;
    float *blk = xalloc((size_t)C15_BLOCK * S * sizeof(float));   /* score/prob block */
    float *cb  = xalloc((size_t)C15_BLOCK * HD * sizeof(float));  /* ctx block */
    uint64_t sum = 0;
    double t_s = 0, t_m = 0, t_c = 0;
    double t0 = now();
    for (int hh = 0; hh < H; hh++) {
        int kvh = hh / Q_PER_KV;
        const float *Kk = Kf + (size_t)kvh * S * HD, *Vk = Vf + (size_t)kvh * S * HD;
        for (int s0 = 0; s0 < S; s0 += C15_BLOCK) {
            int rb = S - s0 < C15_BLOCK ? S - s0 : C15_BLOCK;
            int s1 = s0 + rb;                       /* causal: cols [0, s1) */
            const float *Ab = QP + (size_t)hh * S * HD + (size_t)s0 * HD;
            double b0 = now();
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        rb, s1, HD, 1.0f, Ab, HD, Kk, HD, 0.0f, blk, s1);
            t_s += now() - b0;
            double b1 = now();
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
            for (int i = 0; i < rb; i++) {          /* scale+softmax in place, zero tail */
                float *row = blk + (size_t)i * s1;
                for (int t = 0; t <= s0 + i; t++) row[t] *= scale;
                softmax_row(row, s0 + i + 1);
                for (int t = s0 + i + 1; t < s1; t++) row[t] = 0.f;
                if (probs_out && hh == only_hh)
                    memcpy(probs_out + (size_t)(s0 + i) * S, row, (size_t)s1 * sizeof(float));
            }
            t_m += now() - b1;
            double b2 = now();
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        rb, HD, s1, 1.0f, blk, s1, Vk, HD, 0.0f, cb, HD);
            t_c += now() - b2;
            sum += (uint64_t)(cb[0] * 4096.f) + (uint64_t)(cb[(size_t)(rb-1)*HD + HD-1] * 4096.f);
        }
    }
    double el = now() - t0;
    free(Kf); free(Vf); free(QP); free(blk); free(cb);
    *t_aux = aux; *t_sc = t_s; *t_sm = t_m; *t_cx = t_c;
    *chk += (double)sum;
    return el;
}

/* ---- maxdiff cell: scalar scores/ctx vs sgemm scores/ctx on one head -------- */

static void maxdiff_cell(int S, const float *Q, const uint16_t *Kg, const uint16_t *Vg,
                         double *sc_abs, double *sc_rel, double *ctx_abs, double *ctx_rel) {
    const float scale = 1.f / sqrtf((float)HD);
    float *Kf = xalloc((size_t)KVH * S * HD * sizeof(float));
    float *Vf = xalloc((size_t)KVH * S * HD * sizeof(float));
    for (int kvh = 0; kvh < KVH; kvh++) {
        kv16_unpack(Kf + (size_t)kvh * S * HD, Kg + (size_t)kvh * S * HD, (int64_t)S * HD);
        kv16_unpack(Vf + (size_t)kvh * S * HD, Vg + (size_t)kvh * S * HD, (int64_t)S * HD);
    }
    float *QP = xalloc((size_t)H * S * HD * sizeof(float));
    for (int s = 0; s < S; s++)
        for (int hh = 0; hh < H; hh++)
            memcpy(QP + ((size_t)hh * S + s) * HD, Q + ((size_t)s * H + hh) * HD, HD * sizeof(float));
    float *blk = xalloc((size_t)C15_BLOCK * S * sizeof(float));
    float *cb  = xalloc((size_t)C15_BLOCK * HD * sizeof(float));
    float *sc_ref = xalloc((size_t)S * sizeof(float));
    float *cx_ref = xalloc((size_t)C15_BLOCK * HD * sizeof(float));   /* the block's scalar ctx rows */
    double a = 0, r = 0, ca = 0, cr = 0;
    for (int hh = 0; hh < H; hh++) {
        int kvh = hh / Q_PER_KV;
        const float *Kk = Kf + (size_t)kvh * S * HD, *Vk = Vf + (size_t)kvh * S * HD;
        for (int s0 = 0; s0 < S; s0 += C15_BLOCK) {
            int rb = S - s0 < C15_BLOCK ? S - s0 : C15_BLOCK;
            int s1 = s0 + rb;
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        rb, s1, HD, scale, QP + (size_t)hh * S * HD + (size_t)s0 * HD, HD,
                        Kk, HD, 0.0f, blk, s1);
            /* scalar reference rows + ctx for the whole block */
            for (int i = 0; i < rb; i++) {
                int s = s0 + i;
                const float *qv = Q + ((size_t)s * H + hh) * HD;
                float mx = 0;
                for (int t = 0; t <= s; t++) {
                    sc_ref[t] = kv16_dot_lanes(qv, Kg + ((size_t)kvh * S + t) * HD, HD) * scale;
                    float d = fabsf(sc_ref[t] - blk[(size_t)i * s1 + t]);
                    if (d > a) a = d;
                    if (fabsf(sc_ref[t]) > mx) mx = fabsf(sc_ref[t]);
                }
                for (int t = 0; t <= s; t++) {
                    float rel = fabsf(sc_ref[t] - blk[(size_t)i * s1 + t]) / (mx > 0 ? mx : 1);
                    if (rel > r) r = rel;
                }
                float *cx = cx_ref + (size_t)i * HD;
                softmax_row(sc_ref, s + 1);            /* ctx reads the softmaxed probs, as the engine does */
                for (int d = 0; d < HD; d++) cx[d] = 0.f;
                for (int t = 0; t <= s; t++)
                    kv16_axpy(cx, sc_ref[t], Vg + ((size_t)kvh * S + t) * HD, HD);
            }
            /* softmax the block, sgemm ctx, compare per row */
            for (int i = 0; i < rb; i++) {
                float *row = blk + (size_t)i * s1;
                softmax_row(row, s0 + i + 1);
                for (int t = s0 + i + 1; t < s1; t++) row[t] = 0.f;
            }
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        rb, HD, s1, 1.0f, blk, s1, Vk, HD, 0.0f, cb, HD);
            for (int i = 0; i < rb; i++) {
                float mxc = 0;
                for (int d = 0; d < HD; d++) if (fabsf(cx_ref[(size_t)i * HD + d]) > mxc) mxc = fabsf(cx_ref[(size_t)i * HD + d]);
                for (int d = 0; d < HD; d++) {
                    float d_ = fabsf(cx_ref[(size_t)i * HD + d] - cb[(size_t)i * HD + d]);
                    if (d_ > ca) ca = d_;
                    float rel = d_ / (mxc > 0 ? mxc : 1);
                    if (rel > cr) cr = rel;
                }
            }
        }
    }
    free(Kf); free(Vf); free(QP); free(blk); free(cb); free(sc_ref); free(cx_ref);
    if (a > *sc_abs) *sc_abs = a;
    if (r > *sc_rel) *sc_rel = r;
    if (ca > *ctx_abs) *ctx_abs = ca;
    if (cr > *ctx_rel) *ctx_rel = cr;
}

/* ---- driver ------------------------------------------------------------------ */

int main(void) {
    int shapes[] = {2048, 4096, 12288};
    int reps[]   = {3, 2, 1};
    int nth = 1;
#if defined(_OPENMP)
    nth = omp_get_max_threads();
#endif
    printf("c15 sgemm gate: HD=%d H=%d KVH=%d block=%d omp_threads=%d\n",
           HD, H, KVH, C15_BLOCK, nth);
    printf("scalar = engine loop (kv16_dot_lanes scores, softmax_row, kv16_axpy ctx)\n");
    printf("sgemm  = widen K/V once + pack Q, per-head causal %d-row blocks, Accelerate cblas_sgemm\n\n", C15_BLOCK);
    for (int si = 0; si < 3; si++) {
        int S = shapes[si];
        /* data: engine-layout Q [S*H*HD], per-kv-head fp16 grids [S*HD] (T == S, pos_base 0) */
        float *Q = xalloc((size_t)S * H * HD * sizeof(float));
        for (size_t i = 0; i < (size_t)S * H * HD; i++) Q[i] = frand();
        uint16_t *Kg = xalloc((size_t)KVH * S * HD * sizeof(uint16_t));
        uint16_t *Vg = xalloc((size_t)KVH * S * HD * sizeof(uint16_t));
        float *tmp = xalloc((size_t)S * HD * sizeof(float));
        for (int kvh = 0; kvh < KVH; kvh++) {
            for (size_t i = 0; i < (size_t)S * HD; i++) tmp[i] = frand();
            kv16_pack(Kg + (size_t)kvh * S * HD, tmp, (int64_t)S * HD);
            for (size_t i = 0; i < (size_t)S * HD; i++) tmp[i] = frand();
            kv16_pack(Vg + (size_t)kvh * S * HD, tmp, (int64_t)S * HD);
        }
        free(tmp);
        double useful_sc = 2.0 * HD * (double)H * ((double)S * (S + 1) / 2.0);   /* causal score pairs */
        double useful_cx = useful_sc;                                           /* same pair count for ctx */
        double tsc = 0, tsm = 0, tcx = 0, taux = 0, tgs = 0, tgm = 0, tgc = 0, tga = 0;
        double chk1 = 0, chk2 = 0;
        for (int r = 0; r < reps[si]; r++) {
            double a, b, c;
            double e1 = scalar_pipeline(S, Q, Kg, Vg, NULL, -1, &a, &b, &c, &chk1);
            tsc += a; tsm += b; tcx += c;
            double e2 = sgemm_pipeline(S, Q, Kg, Vg, NULL, -1, &taux, &tgs, &tgm, &tgc, &chk2);
            tga += taux;
            if (r == 0)
                printf("shape S=T=%d: scalar wall %.2fs | sgemm wall %.2fs (+aux %.3fs)\n",
                       S, e1, e2, taux);
        }
        tsc /= nth; tsm /= nth; tcx /= nth;          /* thread-seconds -> wall (balanced team) */
        double scal = tsc + tsm + tcx, sg = tgs + tgm + tgc;
        printf("  scalar (wall, %d threads): scores %.2fs (%.1f GFLOPS) | softmax %.2fs | ctx %.2fs (%.1f GFLOPS) | total %.2fs\n",
               nth, tsc, useful_sc / 1e9 / tsc, tsm, tcx, useful_cx / 1e9 / tcx, scal);
        printf("  sgemm  (wall): scores %.2fs (%.1f GFLOPS) | softmax+mask %.2fs | ctx %.2fs (%.1f GFLOPS) | aux %.3fs | total %.2fs\n",
               tgs, useful_sc / 1e9 / tgs, tgm, tgc, useful_cx / 1e9 / tgc, tga, sg + tga);
        printf("  combined scalar/sgemm speedup: %.2fx\n\n", scal / (sg + tga));
        if (S == 4096) {
            double sa = 0, sr = 0, ca = 0, cr = 0;
            maxdiff_cell(S, Q, Kg, Vg, &sa, &sr, &ca, &cr);
            printf("  maxdiff @S=4096 (all 16 heads): scaled scores abs %.3e rel-to-rowmax %.3e | ctx abs %.3e rel %.3e\n\n",
                   sa, sr, ca, cr);
        }
        free(Q); free(Kg); free(Vg);
    }
    return 0;
}
