/* test_k3_fill_budget.c — the adaptive tier-fill budget contract.
 *
 * coli_k3_fill_budget() bounds the INLINE upload investment of K3's
 * fill-once Vulkan expert tier to a fraction of the measured decode step:
 * budget = fraction * step_s / upload_s, clamped to [1, 64], with the legacy
 * fixed cap while either rate is unmeasured — enabling K3_VK_UP=auto can
 * never start worse than the default. Pure arithmetic, GPU-less testable. */
#include <stdio.h>
#include "../hybrid_split.h"

static int failures;
#define CHECK(cond, ...) do { if (!(cond)) { \
    fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
    fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); failures++; } } while (0)

int main(void) {
    /* unmeasured rates: the legacy cap applies verbatim */
    CHECK(coli_k3_fill_budget(0.0, 0.001, 0.25, 8) == 8, "no step -> legacy");
    CHECK(coli_k3_fill_budget(0.010, 0.0, 0.25, 8) == 8, "no upload -> legacy");
    CHECK(coli_k3_fill_budget(-1.0, -1.0, 0.25, 3) == 3, "bad rates -> legacy");

    /* the budget: fraction * step / upload */
    CHECK(coli_k3_fill_budget(0.100, 0.005, 0.25, 8) == 5, "25%% of 100ms at 5ms -> 5");
    CHECK(coli_k3_fill_budget(0.100, 0.001, 0.25, 8) == 25, "fast bus -> more");
    CHECK(coli_k3_fill_budget(0.010, 0.010, 0.25, 8) == 1, "slow bus -> floor 1");

    /* clamps */
    CHECK(coli_k3_fill_budget(10.0, 0.0001, 0.25, 8) == 64, "ceiling 64");
    CHECK(coli_k3_fill_budget(0.100, 0.005, 0.0, 8) == 1, "zero fraction -> floor");

    /* monotone in step time: a slower step affords more uploads */
    int previous = 0;
    for (int step = 1; step <= 20; step++) {
        int budget = coli_k3_fill_budget(0.005 * step, 0.004, 0.25, 8);
        CHECK(budget >= previous, "monotone in step time (step %d)", step);
        previous = budget;
    }

    if (failures) { fprintf(stderr, "%d failure(s)\n", failures); return 1; }
    printf("test_k3_fill_budget: ok\n");
    return 0;
}
