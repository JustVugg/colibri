/* test_v4_bank_pair.c — sequencing contract of the double-buffered bank.
 * Pure decisions, GPU-less: the classic double-buffer failure modes (swap on
 * a mismatched or failed prefetch, prefetch past the last layer, feature
 * leaking when off) must be impossible by construction. */
#include <stdio.h>
#include "../deepseek_v4_bank_pair.h"

static int failures;
#define CHECK(cond, ...) do { if (!(cond)) { \
    fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
    fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); failures++; } } while (0)

int main(void) {
    /* off: never anything but legacy, never a prefetch */
    CHECK(coli_v4_bank_pair_decide(0, 5, 5) == V4_BANK_LEGACY, "off must be legacy");
    CHECK(coli_v4_bank_pair_prefetch_target(0, 3, 61) == -1, "off must not prefetch");

    /* bootstrap: nothing prefetched yet */
    CHECK(coli_v4_bank_pair_decide(1, -1, 0) == V4_BANK_LEGACY, "bootstrap legacy");
    CHECK(coli_v4_bank_pair_prefetch_target(1, 0, 61) == 1, "then prefetch L+1");

    /* steady state: match swaps, and the next prefetch follows */
    CHECK(coli_v4_bank_pair_decide(1, 1, 1) == V4_BANK_SWAP, "match must swap");
    CHECK(coli_v4_bank_pair_prefetch_target(1, 1, 61) == 2, "prefetch L+2");

    /* segment restart: bank holds layer 8, chunk arrives for layer 0 */
    CHECK(coli_v4_bank_pair_decide(1, 8, 0) == V4_BANK_LEGACY, "mismatch legacy");

    /* worker failed mid-prefetch: other_layer reported -1 */
    CHECK(coli_v4_bank_pair_decide(1, -1, 7) == V4_BANK_LEGACY, "failure legacy");

    /* last layer: nothing to prefetch */
    CHECK(coli_v4_bank_pair_prefetch_target(1, 60, 61) == -1, "no L past last");
    CHECK(coli_v4_bank_pair_prefetch_target(1, -1, 61) == -1, "no current layer");

    /* full sweep: every ascending walk is swap after the bootstrap layer */
    int other = -1;
    int swaps = 0, legacies = 0;
    for (int layer = 0; layer < 61; layer++) {
        if (coli_v4_bank_pair_decide(1, other, layer) == V4_BANK_SWAP) swaps++;
        else legacies++;
        other = coli_v4_bank_pair_prefetch_target(1, layer, 61);
    }
    CHECK(legacies == 1 && swaps == 60,
          "ascending walk: 1 bootstrap + 60 swaps, got %d + %d", legacies, swaps);

    if (failures) { fprintf(stderr, "%d failure(s)\n", failures); return 1; }
    printf("test_v4_bank_pair: ok\n");
    return 0;
}
