/* cpu_check.h -- the processor has the instructions this engine was compiled for,
 * or the engine says so before running any of them (#1979).
 *
 * The release engines are compiled for x86-64-v3, which lets the compiler use
 * AVX2, FMA and BMI2 anywhere. On a processor without them (Intel before
 * Haswell, AMD before Excavator: the Xeon E3-1230 V2 of #1979) the first such
 * instruction stops the process -- SIGILL, or 0xC000001D on Windows -- right
 * after the banner, and the server could only report "engine exited
 * unexpectedly".
 *
 * The check is a constructor, so it runs before main and before anything the
 * compiler was free to build with those instructions, and it is compiled for
 * the baseline processor (target arch=x86-64) together with the two helpers it
 * calls. It compares what the compiler was told to assume (__AVX2__, __FMA__,
 * ...) with what the processor reports, and on a gap writes the reason on
 * stderr and, in serve mode, the LOAD_FAIL line of load_fail.h, then exits 1.
 *
 * It writes with write() and copies by hand, on purpose: with _FORTIFY_SOURCE
 * the stdio and string functions are inline wrappers compiled for the
 * engine's processor, which gcc may not inline into a baseline function, and
 * load_fail.h's helpers are compiled for the engine's processor as well.
 *
 * Include it in an engine's main translation unit only: every inclusion is a
 * constructor of its own. COLI_CPU_CHECK=0 skips the check (an emulator that
 * reports fewer features than it runs). COLI_CPU_MISSING="<feature> ..." makes
 * the check treat those features as absent, for the tests. */
#ifndef COLI_CPU_CHECK_H
#define COLI_CPU_CHECK_H

#if (defined(__x86_64__) || defined(__i386__)) && defined(__GNUC__) && !defined(__clang__)
#define COLI_CPU_CHECK_ACTIVE 1
#include <stdlib.h>
#if defined(_WIN32)
#include <io.h>
#define COLI_CPU_WRITE(fd, text, n) _write((fd), (text), (unsigned)(n))
#else
#include <unistd.h>
#define COLI_CPU_WRITE(fd, text, n) write((fd), (text), (size_t)(n))
#endif

#define COLI_CPU_BASELINE __attribute__((target("arch=x86-64"), noinline, unused))

/* appends src to the NUL-terminated dst of capacity cap */
COLI_CPU_BASELINE static void coli_cpu_append(char *dst, int cap, const char *src)
{
    int n = 0;
    while (n < cap - 1 && dst[n])
        n++;
    while (n < cap - 1 && *src)
        dst[n++] = *src++;
    dst[n] = 0;
}

/* is `feature` one of the space-separated words of `list` */
COLI_CPU_BASELINE static int coli_cpu_listed(const char *list, const char *feature)
{
    for (const char *p = list; p && *p;) {
        while (*p == ' ' || *p == ',')
            p++;
        int i = 0;
        while (feature[i] && p[i] == feature[i])
            i++;
        if (!feature[i] && (p[i] == 0 || p[i] == ' ' || p[i] == ','))
            return 1;
        while (*p && *p != ' ' && *p != ',')
            p++;
    }
    return 0;
}

/* The features this build assumes and the processor lacks, as space-separated
 * words in `out`; returns how many. The constructor's question, apart so the
 * tests can ask it (COLI_CPU_MISSING plays the absent features). */
COLI_CPU_BASELINE static int coli_cpu_missing(char *out, int cap)
{
    const char *fake = getenv("COLI_CPU_MISSING");
    int count = 0;
    (void)fake;               /* a build that assumes nothing asks nothing */
    out[0] = 0;
    __builtin_cpu_init();
    /* __builtin_cpu_supports takes a literal, hence the macro */
#define COLI_CPU_NEED(feature)                                                    \
    if (!__builtin_cpu_supports(feature) || coli_cpu_listed(fake, feature)) {    \
        if (out[0]) coli_cpu_append(out, cap, " ");                               \
        coli_cpu_append(out, cap, feature);                                       \
        count++;                                                                  \
    }
#ifdef __AVX__
    COLI_CPU_NEED("avx")
#endif
#ifdef __AVX2__
    COLI_CPU_NEED("avx2")
#endif
#ifdef __FMA__
    COLI_CPU_NEED("fma")
#endif
#ifdef __BMI2__
    COLI_CPU_NEED("bmi2")
#endif
#ifdef __AVX512F__
    COLI_CPU_NEED("avx512f")
#endif
#ifdef __AVX512BW__
    COLI_CPU_NEED("avx512bw")
#endif
#ifdef __AVX512VL__
    COLI_CPU_NEED("avx512vl")
#endif
#ifdef __AVX512VNNI__
    COLI_CPU_NEED("avx512vnni")
#endif
#undef COLI_CPU_NEED
    return count;
}

__attribute__((constructor(101), used, target("arch=x86-64")))
static void coli_cpu_check(void)
{
    const char *skip = getenv("COLI_CPU_CHECK");
    if (skip && skip[0] == '0')
        return;
    char missing[128];
    if (!coli_cpu_missing(missing, sizeof missing))
        return;
    char detail[320] = "";
    coli_cpu_append(detail, sizeof detail, "this processor has no ");
    coli_cpu_append(detail, sizeof detail, missing);
    coli_cpu_append(detail, sizeof detail,
                    ", which this engine build needs; build the engines for it with "
                    "ARCH=x86-64-v2 (docs/quickstart.md, \"Old processors\")");
    char line[400] = "";
    coli_cpu_append(line, sizeof line, "colibri: ");
    coli_cpu_append(line, sizeof line, detail);
    coli_cpu_append(line, sizeof line, "\n");
    int n = 0;
    while (line[n])
        n++;
    if (COLI_CPU_WRITE(2, line, n) < 0) { /* nothing else to say it with */ }
    if (getenv("SERVE")) {
        line[0] = 0;
        coli_cpu_append(line, sizeof line, "LOAD_FAIL kind=unsupported ");
        coli_cpu_append(line, sizeof line, detail);
        coli_cpu_append(line, sizeof line, "\n");
        n = 0;
        while (line[n])
            n++;
        if (COLI_CPU_WRITE(1, line, n) < 0) { /* the server sees EOF instead */ }
    }
    exit(1);
}
#endif

#endif
