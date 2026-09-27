/* fmt=8 LUT-gate state machine, pinned on the host with no CUDA toolchain.
 *
 * Why this exists: the gate that stops a fmt=8 tensor reaching a kernel whose
 * device has an unwritten e4m3 table is implemented in backend_cuda.cu, which
 * compiles only under nvcc/hipcc. Every test that exercised it therefore ran
 * nowhere in CPU CI, and a rebase was able to change the surrounding
 * init semantics while three sites in the tree went on asserting the old ones.
 * The two decisions the gate rests on are pure predicates in backend_cuda.h;
 * backend_cuda.cu calls them at the real sites, so what is pinned below is the
 * engine's own logic, not a restatement of it.
 *
 * What this does NOT pin: that backend_cuda.cu calls the predicates in the
 * right order, or at all. That needs a CUDA toolchain (tests/test_fp8_cuda.cu
 * Phase 3 covers it there). What it does pin is the decision table itself and
 * the one invariant the whole gate rests on:
 *
 *     no live context set  =>  the LUT flag is clear
 *
 * which holds because shutdown is the only writer that clears the flag and it
 * clears g_nctx in the same breath, and because init refuses to rebuild
 * contexts underneath a live set.
 */
#include <stdio.h>
#include "../backend_cuda.h"

static int fails;

static void ck(int cond, const char *what) {
    if (!cond) { printf("FAIL %s\n", what); fails++; }
}

/* A model of the three public transitions, written in terms of the SAME
 * predicates the engine calls. `nctx`/`ready` are the engine's two globals. */
typedef struct { int nctx; int ready; int dev[8]; } Gate;

static int g_init(Gate *g, const int *want, int count) {
    int d = coli_cuda_init_disposition(g->nctx, count, want, g->dev);
    if (d == COLI_CUDA_INIT_REFUSE) return 0;          /* touches nothing */
    if (d == COLI_CUDA_INIT_ACCEPT) return 1;          /* touches nothing */
    for (int i = 0; i < count; i++) g->dev[i] = want[i];
    g->nctx = count;                                   /* flag NOT written here */
    return 1;
}
static void g_set_lut(Gate *g) { if (g->nctx > 0) g->ready = 1; }
static void g_shutdown(Gate *g) { g->nctx = 0; g->ready = 0; }
static int g_upload(Gate *g, int fmt) {
    return g->nctx > 0 && coli_cuda_fp8_gate_admits(fmt, g->ready);
}

int main(void) {
    /* --- the disposition table, directly ------------------------------- */
    {
        int live1[1] = {0}, want1[1] = {0}, want2[2] = {0, 1}, wantB[1] = {1};
        ck(coli_cuda_init_disposition(0, 1, want1, live1) == COLI_CUDA_INIT_BUILD,
           "nothing live -> BUILD");
        ck(coli_cuda_init_disposition(1, 1, want1, live1) == COLI_CUDA_INIT_ACCEPT,
           "same set -> ACCEPT");
        ck(coli_cuda_init_disposition(1, 2, want2, live1) == COLI_CUDA_INIT_REFUSE,
           "widening set -> REFUSE");
        ck(coli_cuda_init_disposition(1, 1, wantB, live1) == COLI_CUDA_INIT_REFUSE,
           "same size, different device -> REFUSE");
        int live2[2] = {0, 1};
        ck(coli_cuda_init_disposition(2, 1, want1, live2) == COLI_CUDA_INIT_REFUSE,
           "narrowing set -> REFUSE");
        ck(coli_cuda_init_disposition(2, 2, want2, live2) == COLI_CUDA_INIT_ACCEPT,
           "same two-device set -> ACCEPT");
    }

    /* --- the upload gate ------------------------------------------------ */
    for (int fmt = -3; fmt <= 9; fmt++) {
        if (fmt == 8) continue;
        ck(coli_cuda_fp8_gate_admits(fmt, 0) && coli_cuda_fp8_gate_admits(fmt, 1),
           "non-fmt=8 is admitted regardless of the LUT flag");
    }
    ck(!coli_cuda_fp8_gate_admits(8, 0), "fmt=8 refused while the LUT is unpublished");
    ck(coli_cuda_fp8_gate_admits(8, 1), "fmt=8 admitted once the LUT is published");

    /* --- the three lifecycle edges, as sequences ------------------------ */
    {
        Gate g = {0, 0, {0}};
        int d0[1] = {0}, d01[2] = {0, 1};

        ck(g_init(&g, d0, 1) == 1, "first init succeeds");
        ck(!g_upload(&g, 8), "fmt=8 refused before the first publish");
        ck(g_upload(&g, 1), "fmt=1 unaffected by the gate");
        g_set_lut(&g);
        ck(g_upload(&g, 8), "fmt=8 admitted after publish");

        /* Edge 2: same-set re-init keeps the flag, because it keeps the
         * contexts the table was published to. */
        ck(g_init(&g, d0, 1) == 1, "same-set re-init returns success");
        ck(g.nctx == 1, "same-set re-init leaves the context count alone");
        ck(g_upload(&g, 8), "fmt=8 still admitted after a same-set re-init");

        /* Edge 3: different-set re-init is refused and changes nothing. */
        ck(g_init(&g, d01, 2) == 0, "widening re-init is refused");
        ck(g.nctx == 1, "refused re-init leaves the context set alone");
        ck(g_upload(&g, 8), "refused re-init leaves the LUT flag alone");

        /* Edge 1: shutdown clears both, and the next span must republish. */
        g_shutdown(&g);
        ck(g.nctx == 0 && g.ready == 0, "shutdown clears the set and the flag");
        ck(g_init(&g, d01, 2) == 1, "after shutdown a WIDER set may be built");
        ck(!g_upload(&g, 8), "fmt=8 refused on the widened set until it republishes");
        g_set_lut(&g);
        ck(g_upload(&g, 8), "fmt=8 admitted on the widened set after republish");
    }

    /* --- the invariant, over every reachable transition sequence -------- */
    {
        /* Exhaustive over sequences of length 6 drawn from
         * {init{0}, init{0,1}, set_lut, shutdown}: no reachable state may have
         * a live-free context set and a set flag, and no state may admit fmt=8
         * without a live set. */
        int d0[1] = {0}, d01[2] = {0, 1};
        int idx[6] = {0, 0, 0, 0, 0, 0};
        long checked = 0;
        for (;;) {
            Gate g = {0, 0, {0}};
            for (int s = 0; s < 6; s++) {
                switch (idx[s]) {
                    case 0: g_init(&g, d0, 1); break;
                    case 1: g_init(&g, d01, 2); break;
                    case 2: g_set_lut(&g); break;
                    default: g_shutdown(&g); break;
                }
                if (g.nctx == 0 && g.ready != 0) { printf("FAIL invariant: no contexts but the LUT flag is set\n"); return 1; }
                if (g.nctx == 0 && g_upload(&g, 8)) { printf("FAIL invariant: fmt=8 admitted with no live context\n"); return 1; }
            }
            checked++;
            int p = 5;
            while (p >= 0 && ++idx[p] > 3) { idx[p] = 0; p--; }
            if (p < 0) break;
        }
        if (checked != 4096) { printf("FAIL sequence enumeration covered %ld, expected 4096\n", checked); return 1; }
        printf("lut-gate invariant holds over %ld transition sequences\n", checked);
    }

    if (fails) { printf("cuda lut-gate tests: %d FAILED\n", fails); return 1; }
    printf("cuda lut-gate tests: ok\n");
    return 0;
}
