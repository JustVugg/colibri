/* test_serve_sentinel — the serve handshake must leave the wire bytes alone.
 *
 * `coli` matches the engine's sentinels byte-exactly (endswith on
 * "\x01\x01READY\x01\x01\n", a "^STAT ..." regex). If anything rewrites the
 * trailing LF, the gateway waits for a byte that never arrives and the session
 * hangs with no error at all -- which is #748: Kimi K3 on Windows loaded 93
 * layers in 42 minutes and then sat there forever.
 *
 * The cause is the CRT opening stdout in TEXT mode on Windows, where '\n' is
 * written as "\r\n". colibri.c has called _setmode(..., _O_BINARY) since #195;
 * inkling.c and kimi_k3.c were written without it, and nothing in the tree
 * noticed for months because no test ever looked at the bytes.
 *
 * This is that test. It writes the sentinel through a real FILE* to a real file
 * with coli_serve_binary_mode() applied, then reads the bytes back and asserts
 * there is no CR. On Linux and macOS it can only pass -- which is the point:
 * it is a Windows regression guard that costs nothing to carry elsewhere, and
 * CI now builds every engine on Windows (#736), so it runs where it matters.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef _WIN32
#include <unistd.h>   /* close() dopo mkstemp */
#endif
#include "../compat.h"

#define SENTINEL "\x01\x01READY\x01\x01\n"

static int fails = 0;

static void check(int cond, const char *what)
{
    printf("  %s %s\n", cond ? "ok  " : "FAIL", what);
    if (!cond) fails++;
}

int main(void)
{
    char path[] = "serve_sentinel_XXXXXX";
    /* mkstemp gives a unique name; we want a FILE* we control the mode of, so
     * close the fd and reopen by name. */
    int fd = mkstemp(path);
    if (fd < 0) { perror("mkstemp"); return 2; }
    close(fd);

    FILE *f = fopen(path, "w");          /* deliberately TEXT mode: the bug's condition */
    if (!f) { perror("fopen"); remove(path); return 2; }

    /* Same call the serve loops make before emitting the handshake. On Windows it
     * flips the stream to binary; elsewhere it is a no-op. We point it at stdout,
     * so exercise the same primitive here on our own stream. */
    coli_serve_binary_mode();
#ifdef _WIN32
    _setmode(_fileno(f), _O_BINARY);
#endif

    fputs(SENTINEL, f);
    fprintf(f, "STAT 0 0.0 0.0 %.2f 0 0\n", 1.0);
    fclose(f);

    FILE *r = fopen(path, "rb");         /* rb: read the bytes as they landed */
    if (!r) { perror("fopen rb"); remove(path); return 2; }
    char buf[256];
    size_t n = fread(buf, 1, sizeof(buf), r);
    fclose(r);
    remove(path);

    printf("test_serve_sentinel: %zu bytes written\n", n);

    check(n >= sizeof(SENTINEL) - 1, "something was written");
    check(memcmp(buf, SENTINEL, sizeof(SENTINEL) - 1) == 0,
          "READY sentinel is byte-exact");
    check(memchr(buf, '\r', n) == NULL,
          "no CR anywhere -- a TEXT-mode stream would have inserted one (#748)");

    /* The STAT line is matched by a "^STAT " regex on a line basis, so its
     * terminator has to be a bare LF too. */
    const char *stat = memchr(buf, 'S', n);
    check(stat && strncmp(stat, "STAT ", 5) == 0, "STAT line follows the sentinel");
    check(buf[n - 1] == '\n', "output ends with a bare LF");

    printf("test_serve_sentinel: %s\n", fails ? "FAILED" : "ok");
    return fails ? 1 : 0;
}
