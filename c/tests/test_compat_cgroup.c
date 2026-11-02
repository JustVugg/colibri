/* compat_cgroup_room: a container's memory limit, read as the planner reads it
 * (resource_plan.py memory_available). /proc/meminfo shows the whole machine,
 * and a container is killed at its cgroup's limit however much the machine has
 * free. The trees here are written by hand in a temporary directory, so the
 * test runs on every platform: v2 and v1, the walk from the leaf up, the
 * working set less the inactive file cache, unlimited values, files that do
 * not parse. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "../compat.h"

#ifdef _WIN32
#include <direct.h>
#define mkdir_(p) _mkdir(p)
#else
#define mkdir_(p) mkdir((p), 0700)
#endif

static int fails;
static void check(int ok, const char *what) { if (!ok) { printf("  FAIL: %s\n", what); fails++; } else printf("  ok   %s\n", what); }

static char base[256];
static int nwritten;
static char written[64][512];   /* to clean up */
static int ndirs;
static char dirs[64][512];

static void dir_(const char *rel) {
    char p[512]; snprintf(p, sizeof p, "%s/%s", base, rel);
    if (mkdir_(p) == 0 && ndirs < 64) snprintf(dirs[ndirs++], sizeof dirs[0], "%s", p);
}
static void put(const char *rel, const char *text) {
    char p[512]; snprintf(p, sizeof p, "%s/%s", base, rel);
    FILE *f = fopen(p, "w");
    if (!f) { perror(p); exit(1); }
    fputs(text, f); fclose(f);
    if (nwritten < 64) snprintf(written[nwritten++], sizeof written[0], "%s", p);
}
static void clean(void) {
    while (nwritten) remove(written[--nwritten]);
    while (ndirs) rmdir(dirs[--ndirs]);
}
/* a fresh tree under base/t<n>: its root and its /proc/self/cgroup */
static char root[512], proc[512];
static void tree(int n, const char *proc_text) {
    clean();
    char t[32]; snprintf(t, sizeof t, "t%d", n); dir_(t);
    snprintf(root, sizeof root, "%s/t%d", base, n);
    snprintf(proc, sizeof proc, "%s/t%d/proc", base, n);
    if (proc_text) { char r[48]; snprintf(r, sizeof r, "t%d/proc", n); put(r, proc_text); }
}
static void cg(int n, const char *dir, const char *limit_name, const char *limit, const char *cur_name,
               const char *cur, const char *stat) {
    char r[256];
    snprintf(r, sizeof r, "t%d/%s", n, dir); if (*dir) dir_(r);
    if (limit) { snprintf(r, sizeof r, "t%d/%s%s%s", n, dir, *dir ? "/" : "", limit_name); put(r, limit); }
    if (cur) { snprintf(r, sizeof r, "t%d/%s%s%s", n, dir, *dir ? "/" : "", cur_name); put(r, cur); }
    if (stat) { snprintf(r, sizeof r, "t%d/%s%smemory.stat", n, dir, *dir ? "/" : ""); put(r, stat); }
}
#define V2(n, dir, lim, cur, stat) cg(n, dir, "memory.max", lim, "memory.current", cur, stat)
#define V1(n, dir, lim, cur, stat) cg(n, dir, "memory.limit_in_bytes", lim, "memory.usage_in_bytes", cur, stat)

int main(void) {
    snprintf(base, sizeof base, "test_compat_cgroup_XXXXXX");
    if (!mkdtemp(base)) { perror("mkdtemp"); return 1; }
    uint64_t lim, room;

    tree(0, NULL);
    check(!compat_cgroup_room(root, proc, &lim, &room) && lim == 0 && room == 0, "no /proc/self/cgroup: no limit");

    tree(1, "0::/\n");
    V2(1, "", "max\n", "4096\n", NULL);
    check(!compat_cgroup_room(root, proc, &lim, &room), "v2 at the root, memory.max max: no limit");

    tree(2, "0::/\n");
    V2(2, "", "1000\n", "700\n", "anon 300\ninactive_anon 50\ninactive_file 200\nactive_file 100\n");
    check(compat_cgroup_room(root, proc, &lim, &room) && lim == 1000 && room == 600,
          "v2 at the root: 1000 less 700 used, plus 200 inactive and 100 active file cache");

    tree(3, "0::/a/b\n");
    V2(3, "a", "2000\n", "1500\n", NULL);
    V2(3, "a/b", "5000\n", "100\n", NULL);
    check(compat_cgroup_room(root, proc, &lim, &room) && lim == 2000 && room == 500,
          "v2 leaf and parent: the parent's tighter limit and room");

    tree(4, "0::/a/b\n");
    V2(4, "a", "9000\n", "100\n", NULL);
    V2(4, "a/b", "1000\n", "1200\n", NULL);
    check(compat_cgroup_room(root, proc, &lim, &room) && lim == 1000 && room == 0,
          "a leaf with no room left stays at 0 under a looser parent");

    tree(5, "0::/system.slice/docker-1.scope\n");
    V2(5, "", "3000\n", "1000\n", NULL);
    check(compat_cgroup_room(root, proc, &lim, &room) && lim == 3000 && room == 2000,
          "a leaf the mount does not show: the mount's root is the container's");

    tree(6, "12:pids:/docker/x\n4:cpu,memory:/docker/x\n1:name=systemd:/docker/x\n");
    V1(6, "memory", "3000\n", "1000\n",
       "inactive_file 900\nactive_file 800\ntotal_inactive_file 500\ntotal_active_file 200\n");
    check(compat_cgroup_room(root, proc, &lim, &room) && lim == 3000 && room == 2700,
          "v1 under <root>/memory: the total_ file counters, not the cgroup's own");

    tree(7, "4:memory:/\n");
    V1(7, "memory", "9223372036854771712\n", "1000\n", NULL);
    check(!compat_cgroup_room(root, proc, &lim, &room), "v1's unlimited value: no limit");

    tree(8, "0::/\n4:memory:/\n");
    V2(8, "", "max\n", "100\n", NULL);
    V1(8, "memory", "1000\n", "100\n", NULL);
    check(!compat_cgroup_room(root, proc, &lim, &room), "v2's controls seen: v1 is not read");

    tree(9, "0::/\n4:memory:/\n");
    V1(9, "memory", "1000\n", "100\n", NULL);
    check(compat_cgroup_room(root, proc, &lim, &room) && lim == 1000 && room == 900,
          "hybrid: no v2 controls at the root, v1's limit");

    tree(10, "0::/a\n");
    V2(10, "", "1000\n", "12x\n", NULL);
    V2(10, "a", "99999999999999999999999\n", "10\n", NULL);
    check(!compat_cgroup_room(root, proc, &lim, &room), "counters that do not parse add nothing");

    tree(11, "0::/a\n");
    V2(11, "a", "1000\n", NULL, NULL);
    V2(11, "", "4000\n", "1000\n", NULL);
    check(compat_cgroup_room(root, proc, &lim, &room) && lim == 4000 && room == 3000,
          "a limit without its usage adds nothing; the root's pair counts");

    tree(12, "0::/../t12x\n");
    V2(12, "", "4000\n", "1000\n", NULL);
    dir_("t12x");
    put("t12x/memory.max", "1000\n"); put("t12x/memory.current", "10\n");
    check(!compat_cgroup_room(root, proc, &lim, &room), "a cgroup outside the namespace (..): nothing is read");

    tree(13, "0::/..a/b..\n");
    V2(13, "", "4000\n", "1000\n", NULL);
    check(compat_cgroup_room(root, proc, &lim, &room) && lim == 4000, "names that start or end with .. are names");

    tree(14, "0::/\n");
    V2(14, "", "1000\n", "100\n", "inactive_file 500\n");
    check(compat_cgroup_room(root, proc, &lim, &room) && lim == 1000 && room == 1000,
          "an inactive file cache above the usage (the two files read a moment apart): the whole limit is room");

    tree(15, "0::/a/..\n");
    V2(15, "", "4000\n", "1000\n", NULL);
    check(!compat_cgroup_room(root, proc, &lim, &room), "a path that ends in .. is outside too");

    tree(16, "4:memory:/../x\n");
    V1(16, "memory", "1000\n", "100\n", NULL);
    check(!compat_cgroup_room(root, proc, &lim, &room), "a v1 memory path with .. is outside too");

    tree(17, "0::/\n");
    V2(17, "", "18446744073709551616\n", "1\n", NULL);   /* 2^64, one past the largest counter */
    check(!compat_cgroup_room(root, proc, &lim, &room), "a limit past 2^64-1 does not parse");

    tree(18, "0::/\n");
    V2(18, "", "18446744073709551615\n", "1\n", NULL);   /* 2^64-1, which no real limit is */
    check(!compat_cgroup_room(root, proc, &lim, &room), "the largest counter is no limit either");

    tree(19, "4:memory:/\n");
    V1(19, "memory", "9223372036854775807\n", "1000\n", NULL);   /* LONG_MAX, the other unlimited value */
    check(!compat_cgroup_room(root, proc, &lim, &room), "v1's other unlimited value: no limit");

    tree(20, "0::/\n");
    V2(20, "", "\n", "1000\n", NULL);
    check(!compat_cgroup_room(root, proc, &lim, &room), "a limit file with no digits is no limit, not a limit of 0");

    tree(21, "0::/\n");
    V2(21, "", "1000\n", "100\n", "inactive_file 50\nactive_file 500\n");
    check(compat_cgroup_room(root, proc, &lim, &room) && lim == 1000 && room == 1000,
          "an active file cache above what is left of the usage: the whole limit is room");

    { double t = 50.47, a = 24.91;
      compat_cg_bound(&t, &a, 24224202752ull, 24109000000ull);
      check(t > 24.22 && t < 24.23 && a > 24.10 && a < 24.12, "the cgroup bounds the total and the available"); }
    { double t = 50.47, a = 24.91;
      compat_cg_bound(&t, &a, 1000000000000ull, 900000000000ull);
      check(t == 50.47 && a == 24.91, "a cgroup above the machine changes nothing"); }
    { double t = 50.47, a = 24.91;
      compat_cg_bound(&t, &a, 24224202752ull, 0);
      check(a > 0 && a < 1e-6, "no room left is a small number, not 0 (\"not measurable\")"); }
    { double t = 0, a = 0;
      compat_cg_bound(&t, &a, 5000000000ull, 4000000000ull);
      check(t == 5.0 && a == 4.0, "what the machine did not give comes from the cgroup"); }

    clean();
    rmdir(base);
    if (fails) { printf("test_compat_cgroup: %d failed\n", fails); return 1; }
    printf("test_compat_cgroup: ok\n");
    return 0;
}
