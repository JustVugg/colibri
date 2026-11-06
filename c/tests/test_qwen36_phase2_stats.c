#include <stdio.h>
#include <string.h>

#define main qwen36_main_unused
#include "../qwen36.c"
#undef main

int main(void) {
    memset(g_q36_iq2_metal, 0, sizeof g_q36_iq2_metal);
    memset(g_q36_iq2_fallback, 0, sizeof g_q36_iq2_fallback);

    g_q36_perf_enabled = 0;
    g_q36_perf_phase = Q36_PERF_VERIFY;
    q36_perf_iq2(1);
    q36_perf_iq2(0);
    if (g_q36_iq2_metal[Q36_PERF_VERIFY] || g_q36_iq2_fallback[Q36_PERF_VERIFY]) {
        fprintf(stderr, "FAIL: disabled phase2 stats changed counters\n");
        return 1;
    }

    g_q36_perf_enabled = 1;
    g_q36_perf_phase = Q36_PERF_MTP_DRAFT;
    q36_perf_iq2(1);
    g_q36_perf_phase = Q36_PERF_VERIFY;
    q36_perf_iq2(0);
    if (g_q36_iq2_metal[Q36_PERF_MTP_DRAFT] != 1 ||
        g_q36_iq2_fallback[Q36_PERF_MTP_DRAFT] != 0 ||
        g_q36_iq2_metal[Q36_PERF_VERIFY] != 0 ||
        g_q36_iq2_fallback[Q36_PERF_VERIFY] != 1) {
        fprintf(stderr, "FAIL: phase2 stats were not isolated by phase and dispatch\n");
        return 1;
    }

    puts("OK test_qwen36_phase2_stats: opt-in and phase-specific IQ2 counters");
    return 0;
}
