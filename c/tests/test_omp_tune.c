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

#ifdef _OPENMP
static int test_cgroup_cpu_quota_bound(void)
{
    char base[256];
    snprintf(base, sizeof(base), "test_omp_cg_XXXXXX");
    if (!mkdtemp(base)) return 0;
    char proc_path[512], max_path[512];
    snprintf(proc_path, sizeof(proc_path), "%s/proc", base);
    snprintf(max_path, sizeof(max_path), "%s/cpu.max", base);
    FILE *fp = fopen(proc_path, "w");
    if (!fp) { rmdir(base); return 0; }
    fputs("0::/\n", fp);
    fclose(fp);
    fp = fopen(max_path, "w");
    if (!fp) { remove(proc_path); rmdir(base); return 0; }
    /* 1.0 core quota: 100000 us / 100000 us */
    fputs("100000 100000\n", fp);
    fclose(fp);

    setenv("COLIBRI_CGROUP_ROOT", base, 1);
    setenv("COLIBRI_PROC_CGROUP", proc_path, 1);
    unsetenv("OMP_NUM_THREADS");
    unsetenv("COLI_NO_OMP_TUNE");

    int logical = omp_get_num_procs();
    omp_set_num_threads(logical);
    coli_omp_tune_threads("test_cg");

    int got = omp_get_max_threads();
    int fail = 0;
    if (got != 1) {
        fprintf(stderr, "cgroup cpu.max 1-core bound failed: got %d threads, want 1\n", got);
        fail = 1;
    }

    unsetenv("COLIBRI_CGROUP_ROOT");
    unsetenv("COLIBRI_PROC_CGROUP");
    remove(max_path);
    remove(proc_path);
    rmdir(base);
    return fail;
}
#else
static int test_cgroup_cpu_quota_bound(void) { return 0; }
#endif

/* Apple Silicon's tiers (coli_apple_team): every tier but the efficiency ones by name, the
 * last one taken for efficiency when a name is missing. A pure function, checked on any OS. */
static int test_apple_tiers(void)
{
    static const struct { const char *chip; int n, cpus[3], eff[3], want; } cases[] = {
        {"M1 Max: Performance + Efficiency", 2, {8, 2}, {0, 1}, 8},
        {"M6: Super + Performance + Efficiency (#2194)", 3, {2, 4, 6}, {0, 0, 1}, 6},
        {"M5 Pro: Super + Performance, no efficiency tier (#2938)", 2, {6, 12}, {0, 0}, 18},
        {"no names: the last tier is the efficiency one", 2, {6, 12}, {-1, -1}, 6},
        {"one name missing: the same", 3, {2, 4, 6}, {0, -1, 1}, 6},
        {"one tier: every core", 1, {8}, {-1}, 8},
        {"only efficiency tiers: every core", 1, {4}, {1}, 4},
    };
    int fail = 0;
    for (size_t i = 0; i < sizeof cases / sizeof *cases; i++) {
        const int got = coli_apple_team(cases[i].n, cases[i].cpus, cases[i].eff);
        if (got != cases[i].want) {
            fprintf(stderr, "apple tiers, %s: got %d, want %d\n", cases[i].chip, got, cases[i].want);
            fail = 1;
        }
    }
    return fail;
}

int main(void)
{
    if (test_apple_tiers()) return 1;
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

    fail |= test_cgroup_cpu_quota_bound();

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
