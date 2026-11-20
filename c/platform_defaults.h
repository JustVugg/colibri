/* platform_defaults.h -- an engine's measured configuration on a platform, filled in for the variables the
 * environment leaves unset. An explicit value (set and not empty) always wins; COLI_PLATFORM_DEFAULTS=0 keeps
 * every code default. An engine applies its table once, early in main, before anything reads those variables,
 * and prints the list this returns (qwen38: the SYCL build's hybrid). Header-only C, no backend: the test
 * (tests/test_platform_defaults.c) runs it on any host. The SYCL backend's own per-architecture defaults are
 * coli_sycl_tuned_default (sycl/coli_sycl_profile.h). */
#ifndef COLI_PLATFORM_DEFAULTS_H
#define COLI_PLATFORM_DEFAULTS_H
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { const char *name, *value; } ColiDefaultKV;

static inline int coli_platform_defaults_setenv(const char *name, const char *value) {
#ifdef _WIN32
    return _putenv_s(name, value);
#else
    return setenv(name, value, 1);
#endif
}

/* Sets each name of `t` that is unset or empty to its value. Returns how many it set and lists them as
 * "A=1 B=2" in `got` (cap bytes, always terminated; a list too long for it is cut). Sets nothing when
 * COLI_PLATFORM_DEFAULTS is "0". */
static inline int coli_platform_defaults_apply(const ColiDefaultKV *t, int n, char *got, size_t cap) {
    if (got && cap) got[0] = 0;
    const char *off = getenv("COLI_PLATFORM_DEFAULTS");
    if (off && !strcmp(off, "0")) return 0;
    int set = 0;
    size_t len = 0;
    for (int i = 0; i < n; i++) {
        const char *v = getenv(t[i].name);
        if (v && *v) continue;
        if (coli_platform_defaults_setenv(t[i].name, t[i].value) != 0) continue;
        set++;
        if (got && len + 1 < cap) {
            int w = snprintf(got + len, cap - len, "%s%s=%s", len ? " " : "", t[i].name, t[i].value);
            if (w > 0) len += (size_t)w < cap - len ? (size_t)w : cap - len - 1;
        }
    }
    return set;
}

static inline int coli_platform_defaults_unsetenv(const char *name) {
#ifdef _WIN32
    return _putenv_s(name, "");
#else
    return unsetenv(name);
#endif
}

/* The same for variables that only make sense together: sets all of `t` when every one of them is unset or empty,
 * none when any is set (one of them set by hand leaves the others to the caller too), and none when one cannot be
 * set (those already set are unset again: whole or nothing). Returns how many it set (n or 0) and lists them in `got`
 * as above. Sets nothing when COLI_PLATFORM_DEFAULTS is "0". */
static inline int coli_platform_defaults_apply_group(const ColiDefaultKV *t, int n, char *got, size_t cap) {
    if (got && cap) got[0] = 0;
    const char *off = getenv("COLI_PLATFORM_DEFAULTS");
    if (off && !strcmp(off, "0")) return 0;
    for (int i = 0; i < n; i++) {
        const char *v = getenv(t[i].name);
        if (v && *v) return 0;
    }
    for (int i = 0; i < n; i++)
        if (coli_platform_defaults_setenv(t[i].name, t[i].value) != 0) {
            for (int k = 0; k < i; k++) coli_platform_defaults_unsetenv(t[k].name);
            return 0;
        }
    size_t len = 0;
    for (int i = 0; i < n && got && len + 1 < cap; i++) {
        int w = snprintf(got + len, cap - len, "%s%s=%s", len ? " " : "", t[i].name, t[i].value);
        if (w > 0) len += (size_t)w < cap - len ? (size_t)w : cap - len - 1;
    }
    return n;
}

#endif /* COLI_PLATFORM_DEFAULTS_H */
