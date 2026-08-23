/* test_v4_hybrid_policy.c — the q* fill/execute split contract.
 *
 * coli_v4_hybrid_fill_count() decides, out of m missing experts, how many to
 * upload-and-run-on-GPU versus compute on the host: q* = m * B_P / B_H,
 * clamped to [0, m], with every unmeasured or degenerate input falling back
 * to m — the engine's historical upload-everything behaviour, never a new
 * one. Pure arithmetic, so this runs on machines with no GPU: the CUDA
 * wiring in deepseek_v4.c is validated on real hardware separately. */
#include <stdio.h>
#include "../deepseek_v4_hybrid.h"

static int failures;
#define CHECK(cond, ...) do { if (!(cond)) { \
    fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
    fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); failures++; } } while (0)

int main(void) {
    /* no misses: nothing to decide */
    CHECK(coli_v4_hybrid_fill_count(0, 5.0, 5.0) == 0, "m=0 must fill 0");
    CHECK(coli_v4_hybrid_fill_count(-3, 5.0, 5.0) == 0, "m<0 must fill 0");

    /* unmeasured bandwidths: legacy upload-everything, never something new */
    CHECK(coli_v4_hybrid_fill_count(6, 0.0, 7.0) == 6, "no fill bw -> all");
    CHECK(coli_v4_hybrid_fill_count(6, 7.0, 0.0) == 6, "no host bw -> all");
    CHECK(coli_v4_hybrid_fill_count(6, -1.0, -1.0) == 6, "bad bw -> all");

    /* the closed form: q* = m * B_P / B_H */
    CHECK(coli_v4_hybrid_fill_count(6, 5.0, 10.0) == 3, "half ratio -> m/2");
    CHECK(coli_v4_hybrid_fill_count(6, 1.0, 4.0) == 2,  "6*0.25 -> 2 (round)");
    CHECK(coli_v4_hybrid_fill_count(6, 1.0, 1000.0) == 0,
          "host vastly faster -> compute everything there");

    /* degenerate limit: as B_H approaches (or drops below) B_P the split
     * collapses into plain upload-everything */
    CHECK(coli_v4_hybrid_fill_count(6, 5.0, 5.0) == 6, "equal bw -> all fill");
    CHECK(coli_v4_hybrid_fill_count(6, 10.0, 5.0) == 6, "bus faster -> clamp m");

    /* monotone in the fill bandwidth: a faster bus never fills fewer */
    int previous = 0;
    for (int step = 1; step <= 20; step++) {
        int fill = coli_v4_hybrid_fill_count(8, 0.5 * step, 10.0);
        CHECK(fill >= previous, "fill must be monotone in B_P (step %d)", step);
        previous = fill;
    }

    /* EMA: first valid sample seeds, invalid samples never poison */
    CHECK(coli_v4_hybrid_ema(0.0, 4.0) == 4.0, "first sample seeds");
    CHECK(coli_v4_hybrid_ema(4.0, 0.0) == 4.0, "zero sample ignored");
    CHECK(coli_v4_hybrid_ema(4.0, -1.0) == 4.0, "negative sample ignored");
    double smoothed = coli_v4_hybrid_ema(4.0, 8.0);
    CHECK(smoothed > 4.0 && smoothed < 8.0, "EMA moves between old and new");

    if (failures) { fprintf(stderr, "%d failure(s)\n", failures); return 1; }
    printf("test_v4_hybrid_policy: ok\n");
    return 0;
}
