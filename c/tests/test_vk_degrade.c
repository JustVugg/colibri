/* COLI_VK_STRICT's helper (coli_vk_degrade, backend_vulkan.c) against its class
 * table, with no Vulkan device: the knob is read once per process, so every case
 * runs in a forked child that sets its own environment before the first call and
 * redirects stderr to a file the parent reads back. Gates:
 *   parse    COLI_VK_STRICT unset, 0, 2 or junk is off; 1 is on;
 *   refuse   COLI_VK_REFUSE returns, never counts as a degradation, and under
 *            strict prints exactly one "[VK] refused:" line per distinct op;
 *   degrade  under strict: one "[VK] strict: aborting, op=" line, exit 70;
 *            unset: silent (nothing on stderr), counted, no abort;
 *   fallback COLI_VK_FALLBACK never aborts and prints nothing, strict or not;
 *   table    64 distinct ops fill the dedup table; the 65th onward records
 *            nothing, changes nothing;
 *   count    coli_vk_degrade_events counts degradations and fallbacks, not refusals.
 *
 *   make tests/test_vk_degrade VK=1 && ./tests/test_vk_degrade   (also in `check`,
 *   and the first case of tests/vulkan_strict.sh) */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/wait.h>
#include "../backend_vulkan.h"

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fails++; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

/* the child's stderr lands here; one capture file, truncated per spawn */
static char cap_path[] = "/tmp/vk_degradeXXXXXX";
static int cap_fd = -1;

static long log_lines(void) {   /* lines in the capture, whatever they say */
    lseek(cap_fd, 0, SEEK_SET);
    FILE *f = fdopen(dup(cap_fd), "r");
    if (!f) return -1;
    long n = 0; int c;
    while ((c = fgetc(f)) != EOF) n += c == '\n';
    fclose(f);
    return n;
}
static int log_has(const char *s) {
    lseek(cap_fd, 0, SEEK_SET);
    FILE *f = fdopen(dup(cap_fd), "r");
    if (!f) return 0;
    char line[512]; int hit = 0;
    while (fgets(line, sizeof line, f)) if (strstr(line, s)) { hit = 1; break; }
    fclose(f);
    return hit;
}

/* Forks, points the child's stderr at the capture and runs fn there; the parent
 * waits and returns the child's exit status (or -1 if it died on a signal). */
static int spawn(void (*fn)(void)) {
    ftruncate(cap_fd, 0);
    lseek(cap_fd, 0, SEEK_SET);   /* the child writes through this offset: no hole */
    pid_t p = fork();
    if (p < 0) { perror("fork"); exit(2); }
    if (!p) {
        dup2(cap_fd, 2);
        fn();
        _exit(0);
    }
    int st;
    if (waitpid(p, &st, 0) != p) return -1;
    return WIFEXITED(st) ? WEXITSTATUS(st) : -1;
}

/* ---- one case per child: the env is set inside, before the first gate call ------ */

static void c_off_unset(void) { unsetenv("COLI_VK_STRICT"); _exit(coli_vk_strict() != 0); }
static void c_off_zero(void)  { setenv("COLI_VK_STRICT", "0", 1); _exit(coli_vk_strict() != 0); }
static void c_off_two(void)   { setenv("COLI_VK_STRICT", "2", 1); _exit(coli_vk_strict() != 0); }
static void c_off_junk(void)  { setenv("COLI_VK_STRICT", "yes", 1); _exit(coli_vk_strict() != 0); }
static void c_on_one(void)    { setenv("COLI_VK_STRICT", "1", 1); _exit(coli_vk_strict() == 0); }

static void c_refuse(void) {   /* strict: one refused line per op, no degradation counted */
    setenv("COLI_VK_STRICT", "1", 1);
    coli_vk_degrade(COLI_VK_REFUSE, "op A", "first");
    coli_vk_degrade(COLI_VK_REFUSE, "op A", "second");
    coli_vk_degrade(COLI_VK_REFUSE, "op B", "other");
    _exit(coli_vk_degrade_events() != 0);   /* a refusal is not a degradation */
}
static void c_degrade_strict(void) {
    setenv("COLI_VK_STRICT", "1", 1);
    coli_vk_degrade(COLI_VK_DEGRADE, "op X", "boom");
    _exit(9);   /* not reached: the gate exits 70 */
}
static void c_degrade_default(void) {   /* unset: silent, counted, no abort */
    unsetenv("COLI_VK_STRICT");
    coli_vk_degrade(COLI_VK_DEGRADE, "op Y", "quiet");
    _exit(coli_vk_degrade_events() != 1);
}
static void c_fallback(void) {   /* strict or not: never fatal, never a line */
    setenv("COLI_VK_STRICT", "1", 1);
    coli_vk_degrade(COLI_VK_FALLBACK, "op Z", "budget");
    _exit(coli_vk_degrade_events() != 1);
}
static void c_table_cap(void) {   /* 64 entries; the rest record nothing and say nothing */
    setenv("COLI_VK_STRICT", "1", 1);
    static char ops[70][16];   /* the table holds the pointer: stable strings, one per op */
    for (int k = 0; k < 70; k++) { snprintf(ops[k], sizeof ops[k], "cap op %d", k); coli_vk_degrade(COLI_VK_REFUSE, ops[k], "fill"); }
    _exit(0);
}
static void c_default_silent(void) {   /* unset: nothing at all reaches stderr */
    unsetenv("COLI_VK_STRICT");
    coli_vk_degrade(COLI_VK_DEGRADE, "op Y", "quiet");
    coli_vk_degrade(COLI_VK_REFUSE, "op A", "quiet");
    coli_vk_degrade(COLI_VK_FALLBACK, "op Z", "quiet");
    _exit(0);
}

int main(void) {
    cap_fd = mkstemp(cap_path);
    CHECK(cap_fd >= 0, "mkstemp");
    if (cap_fd < 0) return 2;
    unlink(cap_path);   /* keep only the descriptor */

    int st;
    st = spawn(c_off_unset);   CHECK(st == 0, "parse: unset must read off (exit %d)", st);
    st = spawn(c_off_zero);    CHECK(st == 0, "parse: 0 must read off (exit %d)", st);
    st = spawn(c_off_two);     CHECK(st == 0, "parse: 2 must read off (exit %d)", st);
    st = spawn(c_off_junk);    CHECK(st == 0, "parse: junk must read off (exit %d)", st);
    st = spawn(c_on_one);      CHECK(st == 0, "parse: 1 must read on (exit %d)", st);

    st = spawn(c_refuse);
    CHECK(st == 0, "refuse: must return (exit %d)", st);
    CHECK(log_lines() == 2, "refuse: one line per distinct op, got %ld", log_lines());
    CHECK(log_has("[VK] refused: op A — first"), "refuse: the first op A line");
    CHECK(log_has("[VK] refused: op B — other"), "refuse: the op B line");

    st = spawn(c_degrade_strict);
    CHECK(st == 70, "degrade: strict must exit 70, got %d", st);
    CHECK(log_lines() == 1, "degrade: exactly one line, got %ld", log_lines());
    CHECK(log_has("[VK] strict: aborting, op=op X — boom"), "degrade: the aborting line");

    st = spawn(c_degrade_default);
    CHECK(st == 0, "degrade: default must not abort (exit %d)", st);
    CHECK(log_lines() == 0, "degrade: default is silent, got %ld lines", log_lines());

    st = spawn(c_fallback);
    CHECK(st == 0, "fallback: never fatal (exit %d)", st);
    CHECK(log_lines() == 0, "fallback: never a line, got %ld", log_lines());

    st = spawn(c_table_cap);
    CHECK(st == 0, "table: 70 distinct ops must not crash (exit %d)", st);
    CHECK(log_lines() == 64, "table: the 64-entry cap, got %ld refused lines", log_lines());

    st = spawn(c_default_silent);
    CHECK(st == 0, "default: every class silent (exit %d)", st);
    CHECK(log_lines() == 0, "default: stderr empty, got %ld lines", log_lines());

    close(cap_fd);
    printf(fails ? "FAIL (%d)\n" : "PASS\n", fails);
    return fails != 0;
}
