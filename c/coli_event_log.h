/* coli_event_log.h -- why a device or a chain stopped, as one line on stderr and the same line appended to a file:
 * by the time someone looks, a server's console may be gone (#2793: "the device stopped answering", and nothing kept).
 *
 * The file: COLI_EVENT_LOG names it; unset, colibri-logs/gpu.log in the home folder (HOME, else USERPROFILE on
 * Windows), the folder made when missing; COLI_EVENT_LOG=0 writes none. Each line starts with the local time and
 * the process id. A file that cannot be opened is skipped quietly: the stderr line is still there. */
#ifndef COLI_EVENT_LOG_H
#define COLI_EVENT_LOG_H
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _WIN32
#include <direct.h>
#include <process.h>
#define coli_ev_mkdir(p) _mkdir(p)
#define coli_ev_pid() _getpid()
#else
#include <sys/stat.h>
#include <unistd.h>
#define coli_ev_mkdir(p) mkdir((p), 0755)
#define coli_ev_pid() getpid()
#endif

static inline const char *coli_event_path(char *buf, size_t n) {
    const char *e = getenv("COLI_EVENT_LOG");
    if (e && *e) return strcmp(e, "0") ? e : NULL;
    const char *home = getenv("HOME");
    if (!home || !*home) home = getenv("USERPROFILE");
    if (!home || !*home) return NULL;
    snprintf(buf, n, "%s/colibri-logs", home);
    (void)coli_ev_mkdir(buf);
    size_t l = strlen(buf);
    snprintf(buf + l, n - l, "/gpu.log");
    return buf;
}
#if defined(__GNUC__) || defined(__clang__)
static inline void coli_event(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
#endif
static inline void coli_event(const char *fmt, ...) {
    char line[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(line, sizeof line, fmt, ap);
    va_end(ap);
    fprintf(stderr, "%s\n", line);
    char pb[1024];
    const char *path = coli_event_path(pb, sizeof pb);
    if (!path) return;
    FILE *f = fopen(path, "a");
    if (!f) return;
    time_t t = time(NULL);
    struct tm tmv;
#ifdef _WIN32
    localtime_s(&tmv, &t);
#else
    localtime_r(&t, &tmv);
#endif
    char ts[32];
    strftime(ts, sizeof ts, "%Y-%m-%d %H:%M:%S", &tmv);
    fprintf(f, "%s pid %d %s\n", ts, (int)coli_ev_pid(), line);
    fclose(f);
}
#endif
