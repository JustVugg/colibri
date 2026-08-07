/* Physical-core OpenMP sizing (#718).
 *
 * The helper is shared by the GLM, Kimi K3, and OLMoE engines.  Exercise its
 * runtime contract directly: cap an SMT host to physical cores when topology is
 * available, preserve an explicit OMP_NUM_THREADS override, and honor the
 * existing COLI_NO_OMP_TUNE kill switch.  Hosts without SMT still validate the
 * two override paths and correctly leave the OpenMP default unchanged. */
#include <stdio.h>
#include <stdlib.h>

#ifdef _OPENMP
#include <omp.h>
#endif
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
    return 0;
}
#endif

static void env_set(const char *name, const char *value)
{
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

static void env_unset(const char *name)
{
#ifdef _WIN32
    _putenv_s(name, "");
#else
    unsetenv(name);
#endif
}

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
#endif

    env_unset("OMP_NUM_THREADS");
    env_unset("COLI_NO_OMP_TUNE");
    omp_set_num_threads(logical);
    coli_omp_tune_threads("test");
    int got = omp_get_max_threads();
    int want = physical > 0 && physical < logical ? physical : logical;
    if (got != want) {
        fprintf(stderr, "default sizing: got %d threads, want %d (physical=%d logical=%d)\n",
                got, want, physical, logical);
        fail = 1;
    }

    int sentinel = logical > 1 ? logical - 1 : 1;
    omp_set_num_threads(sentinel);
    env_set("OMP_NUM_THREADS", "7");
    coli_omp_tune_threads("test");
    if (omp_get_max_threads() != sentinel) {
        fprintf(stderr, "explicit OMP_NUM_THREADS override was not preserved\n");
        fail = 1;
    }
    env_unset("OMP_NUM_THREADS");

    omp_set_num_threads(sentinel);
    env_set("COLI_NO_OMP_TUNE", "1");
    coli_omp_tune_threads("test");
    if (omp_get_max_threads() != sentinel) {
        fprintf(stderr, "COLI_NO_OMP_TUNE kill switch was not preserved\n");
        fail = 1;
    }
    env_unset("COLI_NO_OMP_TUNE");

    if (fail) {
        puts("test_omp_tune: FAIL");
        return 1;
    }
    printf("test_omp_tune: ok (physical=%d logical=%d)\n", physical, logical);
    return 0;
#endif
}
