/* load_fail.h -- the one exit for a load-time failure, classified.
 *
 * An engine that cannot load its model says why on stderr and exits 1. The
 * server, reading stdout for READY, sees only EOF and reports "colibri engine
 * exited unexpectedly" for an out-of-memory host, an unreadable path, a
 * truncated shard and an unsupported dtype alike. coli_load_fail() keeps the
 * stderr line as it was and, in serve mode (SERVE set, so stdout is the
 * handshake channel), writes one classified line on stdout before exiting:
 *
 *   LOAD_FAIL kind=<nomem|io|format|unsupported> <detail>
 *
 * It is the last line the engine writes; READY never follows it.
 *   nomem        the host refused memory: ENOMEM/EAGAIN from malloc, mmap, mlock
 *   io           the file could not be reached or read: ENOENT, EACCES, EIO, ...
 *   format       the file was read but is not what it claims: short, bad header
 *   unsupported  well-formed, but a dtype or geometry this engine does not serve
 *
 * Standalone on purpose: st.h includes it, and st.h is compiled as C, C++
 * (.mm, .cpp) and CUDA, so nothing from the serve codec can come along. */
#ifndef COLI_LOAD_FAIL_H
#define COLI_LOAD_FAIL_H

#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef enum {
    COLI_LOAD_FAIL_NOMEM = 0,
    COLI_LOAD_FAIL_IO,
    COLI_LOAD_FAIL_FORMAT,
    COLI_LOAD_FAIL_UNSUPPORTED
} ColiLoadFailKind;

static inline const char *coli_load_fail_kind_string(ColiLoadFailKind kind)
{
    switch (kind) {
    case COLI_LOAD_FAIL_NOMEM: return "nomem";
    case COLI_LOAD_FAIL_IO: return "io";
    case COLI_LOAD_FAIL_FORMAT: return "format";
    case COLI_LOAD_FAIL_UNSUPPORTED: return "unsupported";
    }
    return "io";
}

/* errno -> kind, for the sites where the OS said why. The host refusing
 * memory is ENOMEM (malloc, mmap, a read into pages it cannot back), EAGAIN
 * (mlock past RLIMIT_MEMLOCK, mmap with MAP_LOCKED) or ENOBUFS; every other
 * errno is the path or the file. */
static inline ColiLoadFailKind coli_load_fail_kind_from_errno(int err)
{
    switch (err) {
    case ENOMEM:
    case EAGAIN:
#ifdef ENOBUFS
    case ENOBUFS:
#endif
        return COLI_LOAD_FAIL_NOMEM;
    default:
        return COLI_LOAD_FAIL_IO;
    }
}

/* The classified line on the handshake channel. A newline inside the detail
 * would end the line early, so it is written as a space. */
static inline int coli_serve_write_load_fail(FILE *output, ColiLoadFailKind kind,
                                             const char *detail)
{
    if (fprintf(output, "LOAD_FAIL kind=%s ", coli_load_fail_kind_string(kind)) < 0)
        return 0;
    for (const char *p = detail ? detail : ""; *p; p++)
        if (fputc(*p == '\n' || *p == '\r' ? ' ' : *p, output) == EOF) return 0;
    return fputc('\n', output) != EOF && fflush(output) == 0;
}

#if defined(__GNUC__) || defined(__clang__)
#define COLI_LOAD_FAIL_NORETURN __attribute__((noreturn))
#define COLI_LOAD_FAIL_PRINTF(fmt, args) __attribute__((format(printf, fmt, args)))
#elif defined(_MSC_VER)
#define COLI_LOAD_FAIL_NORETURN __declspec(noreturn)
#define COLI_LOAD_FAIL_PRINTF(fmt, args)
#else
#define COLI_LOAD_FAIL_NORETURN
#define COLI_LOAD_FAIL_PRINTF(fmt, args)
#endif

/* Says why on stderr, as every load-time exit always has; says the kind on
 * stdout when SERVE is set; exits 1. Never returns. */
static inline COLI_LOAD_FAIL_NORETURN COLI_LOAD_FAIL_PRINTF(2, 3)
void coli_load_fail(ColiLoadFailKind kind, const char *format, ...)
{
    char detail[1024];
    va_list args;
    va_start(args, format);
    vsnprintf(detail, sizeof(detail), format, args);
    va_end(args);
    fprintf(stderr, "%s\n", detail);
    if (getenv("SERVE")) coli_serve_write_load_fail(stdout, kind, detail);
    exit(1);
}

#endif
