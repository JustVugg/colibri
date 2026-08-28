/* test_ram_clamp.c -- #759 explicit RAM_GB vs physical memory contract.
 * coli_clamp_ram_gb() (colibri.c) must implement exactly: an explicit budget
 * larger than the memory actually available is clamped down to it, an explicit
 * budget that fits passes through untouched, the <=0 "auto" sentinel always
 * passes through, an unreadable snapshot (0) keeps the literal budget, and
 * COLI_RAM_OVERCOMMIT=1 restores take-it-literally behaviour for benchmarking.
 * It is a pure function (no I/O, no globals), so this exercises it directly
 * with fabricated inputs -- no model, no probe, no GPU hardware needed.
 * Portable: builds and runs on every platform, same as test_cap_precedence.c. */
#define main coli_glm_main_unused
#include "../colibri.c"
#undef main

static int failures = 0;

static void check(const char *label, double ram_gb, double avail, int oc, double want){
    double got = coli_clamp_ram_gb(ram_gb, avail, oc);
    if(fabs(got-want) > 1e-9){
        fprintf(stderr, "FAIL %-34s ram=%.1f avail=%.1f oc=%d -> %.1f (want %.1f)\n",
                label, ram_gb, avail, oc, got, want);
        failures++;
    }
}

int main(void){
    /* #759 repro shape: unified-memory host, #653 shrank the snapshot below the
     * user's explicit budget -- that budget must come down to what exists. */
    check("oversized budget clamps",        100.0, 59.0, 0, 59.0);
    check("oversized on discrete too",      200.0, 120.0, 0, 120.0);

    /* a budget that fits is honored literally */
    check("fitting budget untouched",       100.0, 128.0, 0, 100.0);
    check("exact fit untouched",             59.0, 59.0, 0, 59.0);

    /* auto sentinel and degenerate inputs pass through */
    check("auto sentinel untouched",         0.0, 59.0, 0, 0.0);
    check("negative sentinel untouched",    -1.0, 59.0, 0, -1.0);
    check("unreadable snapshot keeps literal", 100.0, 0.0, 0, 100.0);

    /* documented escape hatch: COLI_RAM_OVERCOMMIT=1 keeps the literal budget */
    check("overcommit keeps oversized",     100.0, 59.0, 1, 100.0);
    check("overcommit keeps fitting",       100.0, 128.0, 1, 100.0);

    if(failures){ fprintf(stderr, "%d failure(s)\n", failures); return 1; }
    printf("test_ram_clamp: ok\n");
    return 0;
}
