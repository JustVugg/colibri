/* Physical-core OpenMP sizing (#718).
 *
 * The helper is shared by the GLM, Kimi K3, and OLMoE engines.  Exercise its
 * runtime contract directly: cap an SMT host to physical cores when topology is
 * available (to a hybrid CPU's performance cores), preserve an explicit
 * OMP_NUM_THREADS override, and honor the
 * existing COLI_NO_OMP_TUNE kill switch.  Hosts without SMT still validate the
 * two override paths and correctly leave the OpenMP default unchanged. */
#include <stdio.h>
#include <stdlib.h>

#ifdef _OPENMP
#include <omp.h>
#endif
#include "../compat.h"       /* setenv/unsetenv: MinGW has neither */
#include "../omp_tune.h"

#ifdef _WIN32
static int test_windows_variable_records(void)
{
    const size_t record_size =
        offsetof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX, Processor) +
        offsetof(PROCESSOR_RELATIONSHIP, GroupMask) + sizeof(GROUP_AFFINITY);
    unsigned char records[2 * record_size];
    memset(records, 0, sizeof(records));

    for (int i = 0; i < 2; i++) {
        unsigned char *record = records + i * record_size;
        LOGICAL_PROCESSOR_RELATIONSHIP relationship = RelationProcessorCore;
        DWORD size = (DWORD)record_size;
        memcpy(record + offsetof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX, Relationship),
               &relationship, sizeof(relationship));
        memcpy(record + offsetof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX, Size),
               &size, sizeof(size));
    }
    if (coli_count_windows_physical_cores(records, sizeof(records)) != 2) {
        fprintf(stderr, "Windows variable-sized topology records lost a core\n");
        return 1;
    }
    int perf = -1;
    coli_count_windows_physical_cores_ex(records, sizeof(records), &perf);
    if (perf != 0) {
        fprintf(stderr, "one EfficiencyClass read as a hybrid CPU (%d performance cores)\n", perf);
        return 1;
    }
    return 0;
}
/* a hybrid CPU: two cores of EfficiencyClass 1 among five, the others 0 */
static int test_windows_hybrid_records(void)
{
    const size_t hs = offsetof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX, Processor);
    const size_t record_size = hs + offsetof(PROCESSOR_RELATIONSHIP, GroupMask) + sizeof(GROUP_AFFINITY);
    unsigned char records[5 * record_size];
    memset(records, 0, sizeof(records));
    for (int i = 0; i < 5; i++) {
        unsigned char *record = records + i * record_size;
        LOGICAL_PROCESSOR_RELATIONSHIP relationship = RelationProcessorCore;
        DWORD size = (DWORD)record_size;
        memcpy(record + offsetof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX, Relationship),
               &relationship, sizeof(relationship));
        memcpy(record + offsetof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX, Size), &size, sizeof(size));
        record[hs + 1] = (unsigned char)(i == 1 || i == 3);   /* EfficiencyClass */
    }
    int perf = -1, cores = coli_count_windows_physical_cores_ex(records, sizeof(records), &perf);
    if (cores != 5 || perf != 2) {
        fprintf(stderr, "hybrid topology: %d cores, %d performance cores (want 5, 2)\n", cores, perf);
        return 1;
    }
    return 0;
}
#endif

int main(void)
{
#ifndef _OPENMP
    puts("test_omp_tune: ok (OpenMP unavailable; helper is a no-op)");
    return 0;
#else
    int fail = 0;
    int logical = omp_get_num_procs();
    int physical = coli_physical_cores();

#ifdef _WIN32
    fail |= test_windows_variable_records();
    fail |= test_windows_hybrid_records();
#endif

    unsetenv("OMP_NUM_THREADS");
    unsetenv("COLI_NO_OMP_TUNE");
    omp_set_num_threads(logical);
    coli_omp_tune_threads("test");
    int got = omp_get_max_threads();
    int perf = coli_performance_cores();   /* a hybrid CPU's P-cores, else 0 */
    int team = perf > 0 && perf < physical ? perf : physical;
    int want = team > 0 && team < logical ? team : logical;
    if (got != want) {
        fprintf(stderr, "default sizing: got %d threads, want %d (physical=%d performance=%d logical=%d)\n",
                got, want, physical, perf, logical);
        fail = 1;
    }

    int sentinel = logical > 1 ? logical - 1 : 1;
    omp_set_num_threads(sentinel);
    setenv("OMP_NUM_THREADS", "7", 1);
    coli_omp_tune_threads("test");
    if (omp_get_max_threads() != sentinel) {
        fprintf(stderr, "explicit OMP_NUM_THREADS override was not preserved\n");
        fail = 1;
    }
    unsetenv("OMP_NUM_THREADS");

    omp_set_num_threads(sentinel);
    setenv("COLI_NO_OMP_TUNE", "1", 1);
    coli_omp_tune_threads("test");
    if (omp_get_max_threads() != sentinel) {
        fprintf(stderr, "COLI_NO_OMP_TUNE kill switch was not preserved\n");
        fail = 1;
    }
    unsetenv("COLI_NO_OMP_TUNE");

    if (fail) {
        puts("test_omp_tune: FAIL");
        return 1;
    }
    printf("test_omp_tune: ok (physical=%d performance=%d logical=%d)\n", physical, coli_performance_cores(), logical);
    return 0;
#endif
}
