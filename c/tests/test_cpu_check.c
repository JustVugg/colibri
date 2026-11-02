/* cpu_check.h (#1979): the features a build assumes and the processor lacks.
 *
 * The constructor has already run when main starts, so reaching main means the
 * processor has what this test was compiled for. COLI_CPU_MISSING then plays an
 * older processor: the features named there must come back as missing, the
 * others not, and only whole words count ("avx" is not "avx2"). On a target
 * without the check (arm64, clang) there is nothing to ask. */
#include "../cpu_check.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #c); return 1; \
} } while (0)

#ifdef COLI_CPU_CHECK_ACTIVE
static void play_missing(const char *features)
{
#ifdef _WIN32
    _putenv_s("COLI_CPU_MISSING", features);
#else
    setenv("COLI_CPU_MISSING", features, 1);
#endif
}
#endif

int main(void)
{
#ifdef COLI_CPU_CHECK_ACTIVE
    char missing[128];
    play_missing("");
    CHECK(coli_cpu_missing(missing, sizeof missing) == 0);
    CHECK(missing[0] == 0);

    CHECK(coli_cpu_listed("avx2", "avx2"));
    CHECK(coli_cpu_listed("fma avx2,bmi2", "avx2"));
    CHECK(coli_cpu_listed("fma avx2,bmi2", "bmi2"));
    CHECK(!coli_cpu_listed("avx2", "avx"));
    CHECK(!coli_cpu_listed("avx", "avx2"));
    CHECK(!coli_cpu_listed(NULL, "avx2"));

#ifdef __AVX2__
    play_missing("avx2");
    CHECK(coli_cpu_missing(missing, sizeof missing) == 1);
    CHECK(strcmp(missing, "avx2") == 0);
#if defined(__FMA__) && defined(__BMI2__)
    play_missing("bmi2 avx2 fma");
    CHECK(coli_cpu_missing(missing, sizeof missing) == 3);
    CHECK(strcmp(missing, "avx2 fma bmi2") == 0);   /* in the build's order, not the list's */
#endif
#endif
    /* a feature the build does not assume is not missing, whatever is said */
    play_missing("sse9");
    CHECK(coli_cpu_missing(missing, sizeof missing) == 0);
    printf("test_cpu_check: ok\n");
#else
    printf("test_cpu_check: no check on this target\n");
#endif
    return 0;
}
