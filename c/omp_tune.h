/* omp_tune.h — dimensionamento della squadra OpenMP sui CORE FISICI.
 *
 * PERCHE' SOLO IL DIMENSIONAMENTO, E NON LO SPIN-WAIT.
 * Il blocco di tuning in colibri.c fa due cose che hanno profili di rischio
 * OPPOSTI, e vanno tenute separate:
 *
 *   dimensionamento  OMP_NUM_THREADS = core fisici (niente SMT)
 *                    #718: +2.3x su Zen3 (5950X, 16C/32T) solo cambiando
 *                    il numero di thread. Il guadagno e' cosi' grande che
 *                    "sommerge la maggior parte dei delta che si citano qui".
 *
 *   spin-wait        OMP_WAIT_POLICY=active, GOMP_SPINCOUNT, KMP_BLOCKTIME
 *                    #707: -2.2x sul decode di un host a bassa residenza
 *                          (M1 Max 32 GB, ~10% di expert residenti)
 *                    #116: -39% su Metal      #159: ~3x su x86+CUDA
 *                    #341: 3000% di CPU su FreeBSD con la squadra ferma
 *                    Meccanismo: dove il token e' fatto di byte dal disco,
 *                    una squadra che gira a vuoto ruba i core al pool di I/O
 *                    che sta facendo il lavoro vero.
 *
 * Kimi K3 e OLMoE non avevano NESSUNA delle due. Qui prendono solo la prima:
 * Kimi e' il motore piu' disk-bound del progetto (misurato: 6.7% di hit,
 * 891 GB letti per 32 token), cioe' esattamente il regime in cui la seconda
 * meta' fa danno. Aggiungergliela sarebbe stato un peggioramento misurabile.
 * GLM now calls this same helper after its optional hot-team re-exec, so it
 * receives the physical-core sizing without changing the independent spin-wait
 * policy that its existing block controls.
 *
 * PERCHE' QUI NON SERVE IL RE-EXEC.
 * colibri.c si ri-esegue perche' OMP_WAIT_POLICY & co. le legge il COSTRUTTORE
 * di libgomp, prima di main(): un setenv() dentro main() arriva tardi. Il numero
 * di thread no: omp_set_num_threads() e' una API di runtime e ha effetto subito.
 * La meta' sicura e' anche la meta' semplice.
 *
 * REGOLA SUI FALLIMENTI: se il conteggio dei core fisici non e' determinabile,
 * NON si indovina — si lascia il default di OpenMP. Un conteggio sbagliato e'
 * peggio di nessun conteggio (cfr. #325, dove un fallback silenzioso a 1
 * inchiodava il decode su un core solo).
 */
#ifndef COLI_OMP_TUNE_H
#define COLI_OMP_TUNE_H

#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "compat.h"

#if defined(_WIN32)
#include <windows.h>
#elif defined(__APPLE__) || defined(__FreeBSD__)
#include <sys/types.h>
#include <sys/sysctl.h>
#else
#include <dirent.h>
#endif
#if defined(__linux__) || defined(__FreeBSD__)
#include <sched.h>
#include <unistd.h>
#endif
#if defined(__FreeBSD__)
#include <sys/types.h>
#include <sys/sysctl.h>
#endif

/* Numero di core FISICI, o 0 se non determinabile. Mai un valore inventato. */
#if defined(_WIN32)
/* perf: if non-NULL, the cores of the highest EfficiencyClass (byte 1 of
 * PROCESSOR_RELATIONSHIP, read by offset: older MinGW headers call it Reserved),
 * or 0 when every core has the same class (one kind of core). */
static int coli_count_windows_physical_cores_ex(const void *buf, DWORD bytes, int *perf)
{
    const char *p = (const char *)buf;
    const char *end = p + bytes;
    const size_t header_size = offsetof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX,
                                        Processor);
    int cores = 0, cls_max = -1, cls_min = 256, at_max = 0;

    while ((size_t)(end - p) >= header_size) {
        LOGICAL_PROCESSOR_RELATIONSHIP relationship;
        DWORD record_size;
        memcpy(&relationship,
               p + offsetof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX, Relationship),
               sizeof(relationship));
        memcpy(&record_size,
               p + offsetof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX, Size),
               sizeof(record_size));
        if (record_size < header_size + 2 || (size_t)(end - p) < record_size)
            break; /* Reject a zero, truncated, or otherwise malformed record. */
        if (relationship == RelationProcessorCore) {
            int cls = (unsigned char)p[header_size + 1];
            cores++;
            if (cls > cls_max) { cls_max = cls; at_max = 0; }
            if (cls == cls_max) at_max++;
            if (cls < cls_min) cls_min = cls;
        }
        p += record_size;
    }
    if (perf) *perf = cores && cls_max > cls_min ? at_max : 0;
    return cores;
}
static int coli_count_windows_physical_cores(const void *buf, DWORD bytes)
{
    return coli_count_windows_physical_cores_ex(buf, bytes, NULL);
}
#endif

/* Apple Silicon's team out of its perflevel tiers (fastest first): every tier but the
 * efficiency ones, eff[i] telling which (1), which not (0), or unknown (-1). The names
 * decide when the system gives them all: an M5 Pro's two tiers are "Super" and
 * "Performance" and neither is an efficiency tier (#2938). Without them the last tier
 * is taken for the efficiency one, as on an M1-M4 (P + E) and an M6 (Super +
 * Performance + Efficiency, #2194). One tier, or only efficiency tiers: every core. */
static int coli_apple_team(int n, const int *cpus, const int *eff)
{
    int named = 1, tot = 0, all = 0;
    for (int i = 0; i < n; i++) { named &= eff[i] >= 0; all += cpus[i] > 0 ? cpus[i] : 0; }
    for (int i = 0; i < n; i++)
        if (cpus[i] > 0 && (named ? !eff[i] : (n == 1 || i < n - 1))) tot += cpus[i];
    return tot > 0 ? tot : all;
}

static int coli_physical_cores(void)
{
#if defined(_WIN32)
    DWORD need = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, NULL, &need);
    if (!need) return 0;
    void *buf = malloc(need);
    if (!buf) return 0;
    int cores = GetLogicalProcessorInformationEx(RelationProcessorCore, buf, &need)
                    ? coli_count_windows_physical_cores(buf, need)
                    : 0;
    free(buf);
    return cores;

#elif defined(__APPLE__)
    /* Apple Silicon: every core but the efficiency tier's. hw.physicalcpu counts
     * the E-cores too (10 on an M1 Max), and with a barrier per matmul the slowest
     * thread sets the pace: the E-cores slowed the team (#707, -4.2% decode). The
     * tiers are hw.perflevel0 .. hw.perflevel(N-1), fastest first, each with its
     * name (coli_apple_team picks them). An Intel Mac has no perflevel*:
     * hw.physicalcpu. */
    int n = 0; size_t sz = sizeof(n);
    if (sysctlbyname("hw.nperflevels", &n, &sz, NULL, 0) == 0 && n >= 1) {
        int cpus[16] = {0}, eff[16];
        if (n > 16) n = 16;
        for (int i = 0; i < n; i++) {
            char nm[48], name[32]; snprintf(nm, sizeof nm, "hw.perflevel%d.logicalcpu", i);
            sz = sizeof(cpus[i]);
            if (sysctlbyname(nm, &cpus[i], &sz, NULL, 0) != 0) cpus[i] = 0;
            snprintf(nm, sizeof nm, "hw.perflevel%d.name", i);
            sz = sizeof(name); eff[i] = -1;
            if (sysctlbyname(nm, name, &sz, NULL, 0) == 0 && sz > 0 && sz <= sizeof(name)) {
                name[sz - 1] = 0; eff[i] = !strcmp(name, "Efficiency");
            }
        }
        const int tot = coli_apple_team(n, cpus, eff);
        if (tot > 0) return tot;
    }
    int v = 0; sz = sizeof(v);
    if (sysctlbyname("hw.perflevel0.logicalcpu", &v, &sz, NULL, 0) == 0 && v > 0) return v;
    v = 0; sz = sizeof(v);
    if (sysctlbyname("hw.physicalcpu", &v, &sz, NULL, 0) == 0 && v > 0) return v;
    return 0;

#elif defined(__FreeBSD__)
    /* kern.smp.cores counts a core once, whatever kern.smp.threads_per_core is.
     * There is no /sys to read here, and without this the team stayed at the
     * logical count: both SMT siblings of every core (#718). */
    int v = 0;
    size_t sz = sizeof(v);
    if (sysctlbyname("kern.smp.cores", &v, &sz, NULL, 0) == 0 && v > 0) return v;
    return 0;

#else
    /* Linux: un core fisico = una lista di thread_siblings distinta. Contare le
     * liste uniche deduplica l'SMT senza dover interpretare la topologia. */
    DIR *d = opendir("/sys/devices/system/cpu");
    if (!d) return 0;
    char seen[1024][64];
    int n = 0;
    struct dirent *e;
    while ((e = readdir(d)) && n < 1024) {
        if (strncmp(e->d_name, "cpu", 3) != 0 || e->d_name[3] < '0' || e->d_name[3] > '9')
            continue;
        /* d_name can be up to 255 bytes; leave enough room for the fixed
         * sysfs prefix/suffix so -Wformat-truncation stays honest when this
         * shared helper is compiled into the GLM engine too. */
        char path[512], line[64];
        snprintf(path, sizeof(path),
                 "/sys/devices/system/cpu/%s/topology/thread_siblings_list", e->d_name);
        FILE *f = fopen(path, "r");
        if (!f) continue;
        if (fgets(line, sizeof(line), f)) {
            line[strcspn(line, "\n")] = 0;
            int dup = 0;
            for (int i = 0; i < n; i++) if (strcmp(seen[i], line) == 0) { dup = 1; break; }
            if (!dup) { snprintf(seen[n], sizeof(seen[0]), "%s", line); n++; }
        }
        fclose(f);
    }
    closedir(d);
    return n;
#endif
}

/* The performance cores of a hybrid x86 CPU (Intel's P and E cores), 0 on a CPU with
 * one kind of core or where it cannot be told. A team meets at a barrier after every
 * matmul, so its slowest thread sets the pace, as with Apple's efficiency cores (#707):
 * on an i5-13500 (6 P + 8 E) the 6 P-cores decoded Qwen3.8 IQ2_XS at 9.40 tok/s against
 * 9.00 for all 14 cores (four runs each), prefill 1.7% faster. resource_plan.py's
 * performance_core_count is the launcher's side of the same probe. */
static int coli_performance_cores(void)
{
#if defined(_WIN32)
    DWORD need = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, NULL, &need);
    if (!need) return 0;
    void *buf = malloc(need);
    if (!buf) return 0;
    int perf = 0;
    if (GetLogicalProcessorInformationEx(RelationProcessorCore, buf, &need))
        coli_count_windows_physical_cores_ex(buf, need, &perf);
    free(buf);
    return perf;
#elif defined(__linux__)
    /* a hybrid Intel CPU registers two core PMUs, cpu_core (P) and cpu_atom (E); the
     * P-cores counted once per SMT sibling list */
    char pl[1024] = "", el[64] = "";
    FILE *f = fopen("/sys/devices/cpu_core/cpus", "r");
    if (!f) return 0;
    if (!fgets(pl, sizeof pl, f)) pl[0] = 0;
    fclose(f);
    if (!(f = fopen("/sys/devices/cpu_atom/cpus", "r"))) return 0;
    if (!fgets(el, sizeof el, f)) el[0] = 0;
    fclose(f);
    if (el[0] < '0' || el[0] > '9') return 0;
    char seen[256][64]; int n = 0;
    for (char *t = strtok(pl, ",\n"); t && n < 256; t = strtok(NULL, ",\n")) {
        int lo, hi;
        if (sscanf(t, "%d-%d", &lo, &hi) != 2) { if (sscanf(t, "%d", &lo) != 1) return 0; hi = lo; }
        for (int c = lo; c <= hi && n < 256; c++) {
            char path[128], line[64];
            snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", c);
            FILE *g = fopen(path, "r");
            if (!g) return 0;
            if (!fgets(line, sizeof line, g)) line[0] = 0;
            fclose(g);
            line[strcspn(line, "\n")] = 0;
            int dup = 0;
            for (int i = 0; i < n; i++) if (!strcmp(seen[i], line)) { dup = 1; break; }
            if (!dup) { snprintf(seen[n], sizeof seen[0], "%s", line); n++; }
        }
    }
    return n;
#else
    return 0;   /* macOS: coli_physical_cores counts the performance level already */
#endif
}

/* ---- the team probe: does the machine give this process the CPUs it counted? ----
 *
 * Every count above is of CPUs that exist. None says whether they are this process's to run
 * on: a WSL2 or VM guest sees all of its vCPUs idle while the host runs other work on the
 * cores under them, a quota is not always in a cgroup this process can read, and a guest of
 * a hybrid CPU is shown neither kind of core. A team of every CPU then has members that are
 * not running, and every parallel region waits at its barrier for one to be scheduled: not a
 * few percent, a collapse. Measured under WSL2 on a Core Ultra 9 285K (8 P + 16 E, shown to
 * the guest as 24 equal CPUs, `top` 97% idle) while the Windows host was 65% busy, Qwen3.6
 * Q4_K_M, the same request: 8 threads 7.3 s, 16 threads 11.2 s, 20 threads 91 s, 24 (the
 * default) 146 s; and no engine code is needed to see it, 4000 empty-handed parallel regions
 * take 21 us each with 16 threads and 3356 us with 18.
 *
 * So the team is measured, not only counted: a short region of fixed work per thread is timed
 * at a quarter of the team (the base: those CPUs the machine can nearly always give) and at
 * the whole team. Fixed work per thread means a healthy team's region takes as long whatever
 * its size; a team larger than what the machine gives takes many times longer. Past
 * COLI_OMP_PROBE_SLOW times the base the team steps down by eighths to the largest that is
 * not, and one step more. A healthy machine pays the two measurements, about ten milliseconds, and keeps its
 * team. The probe is at start only: a host that gets busy later is not seen. */

/* The fixed work of one thread in a probed region: a dependent chain, so the compiler cannot
 * fold or vectorize it away, about 50 us. */
#define COLI_OMP_PROBE_WORK (1 << 16)
#define COLI_OMP_PROBE_SLOW 4.0

#ifdef _OPENMP
static inline int coli_omp_probe_cmp(const void *a, const void *b)
{
    const double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}
/* The median time of a parallel region of `team` threads, seconds. Up to 48 regions after
 * two that let the runtime make the team, or as many as 50 ms hold (four at least): a team
 * that does not get its CPUs is slow in nearly every region, so a few are enough to see it
 * and the median keeps one late region from condemning a healthy team. */
static inline double coli_omp_region_s(int team, void *unused)
{
    (void)unused;
    enum { WARM = 2, MOST = 48, LEAST = 4 };
    double t[MOST];
    float sink = 0.f;
    int n = 0;
    const double start = omp_get_wtime();
    for (int r = 0; r < WARM + MOST; r++) {
        const double t0 = omp_get_wtime();
        #pragma omp parallel num_threads(team) reduction(+:sink)
        {
            float a = 1.f;
            for (int i = 0; i < COLI_OMP_PROBE_WORK; i++) a = a * 0.9999999f + (float)(i & 7);
            sink += a;
        }
        const double t1 = omp_get_wtime();
        if (r >= WARM) t[n++] = t1 - t0;
        if (n >= LEAST && t1 - start > 0.05) break;
    }
    if (sink < 0.f) return 0.0;            /* never: the regions' result is used */
    qsort(t, (size_t)n, sizeof t[0], coli_omp_probe_cmp);
    return t[n / 2];
}
#endif

/* The team to run with, given the team counted and a way to time a region at a size
 * (coli_omp_region_s; a table in tests). `base` and `full`, when not NULL, get the two times
 * the decision starts from. A team under 4 is kept: there is no smaller team to compare it to. */
static inline int coli_omp_probe_pick(int team, double (*region_s)(int team, void *ctx), void *ctx,
                                      double *base, double *full)
{
    if (base) *base = 0.0;
    if (full) *full = 0.0;
    if (team < 4 || !region_s) return team;
    const int small = team / 4 < 2 ? 2 : team / 4;
    const double b = region_s(small, ctx), f = region_s(team, ctx);
    if (base) *base = b;
    if (full) *full = f;
    if (!(b > 0.0) || !(f > 0.0) || f <= COLI_OMP_PROBE_SLOW * b) return team;
    for (int k = 7; k >= 3; k--) {
        const int t = team * k / 8;
        if (t <= small) break;
        if (t < team && region_s(t, ctx) <= COLI_OMP_PROBE_SLOW * b) {
            /* one step under the largest that holds: the edge moves with the host's load, and
             * a team just under it is already slower than a smaller one (16 against 8 above) */
            const int under = team * (k - 1) / 8;
            return under > small ? under : small;
        }
    }
    return small;
}

/* Lower the team to what the probe finds the machine gives (COLI_OMP_PROBE=0: not asked). */
static inline void coli_omp_probe_team(const char *engine, int team)
{
#ifdef _OPENMP
    const char *off = getenv("COLI_OMP_PROBE");
    if (off && !strcmp(off, "0")) return;
    double base = 0.0, full = 0.0;
    const int got = coli_omp_probe_pick(team, coli_omp_region_s, NULL, &base, &full);
    if (got >= team) return;
    omp_set_num_threads(got);
    fprintf(stderr, "[OMP] %s: %d threads instead of %d: a parallel region takes %.2f ms with %d and "
                    "%.2f ms with %d, so the machine is not giving this process %d CPUs at once "
                    "(a busy host under WSL or a VM, or a limit it cannot read); "
                    "set OMP_NUM_THREADS=<n> to choose, COLI_OMP_PROBE=0 to keep the count\n",
            engine, got, team, 1e3 * full, team, 1e3 * base, team / 4 < 2 ? 2 : team / 4, team);
#else
    (void)engine; (void)team;
#endif
}

/* Dimensiona la squadra OpenMP sui core fisici. Rispetta OMP_NUM_THREADS se
 * l'utente l'ha impostata, e non fa nulla se il conteggio non e' affidabile.
 * `engine` finisce solo nella riga di log. The launcher sets OMP_NUM_THREADS too, as its
 * default (resource_plan.team_cpu_count): it says so with COLI_OMP_TEAM_AUTO, and that count
 * is then the team to probe, not a user's choice to leave alone. */
static inline void coli_omp_tune_threads(const char *engine)
{
#ifdef _OPENMP
    const char *off = getenv("COLI_NO_OMP_TUNE");
    if (off) return;                       /* stesso kill-switch degli altri motori */
    if (getenv("OMP_NUM_THREADS")) {
        if (getenv("COLI_OMP_TEAM_AUTO")) coli_omp_probe_team(engine, omp_get_max_threads());
        return;                            /* l'utente comanda */
    }

    int phys = coli_physical_cores();
    if (phys <= 0) return;                 /* sconosciuto -> default di OpenMP */
    int perf = coli_performance_cores();   /* a hybrid CPU: its P-cores pace the team */
    int team = perf > 0 && perf < phys ? perf : phys;
    int logical = omp_get_max_threads();

    double cg_quota = 0.0;
    int cg_threads = compat_cgroup_cpu_limit(&cg_quota);
    int bounded_by_cgroup = 0;
    if (cg_threads > 0 && cg_threads < team) {
        team = cg_threads;
        bounded_by_cgroup = 1;
    }

    /* never above OpenMP's own default: that is the affinity mask, a container's cpuset
     * (a quota of 7 CPUs on a cpuset of 4 stays at the 4) */
    if (team >= logical) {                 /* niente da evitare: silenzio */
        coli_omp_probe_team(engine, logical);
        return;
    }

    omp_set_num_threads(team);
    if (bounded_by_cgroup)
        fprintf(stderr, "[OMP] %s: %d threads bounded by container CPU quota "
                        "(cgroup: %.2f cores, %d physical cores on host); "
                        "set OMP_NUM_THREADS=<n> to override\n",
                engine, team, cg_quota, phys);
    else if (team < phys)
        fprintf(stderr, "[OMP] %s: %d performance-core threads instead of %d logical CPUs "
                        "(%d physical cores, the efficiency cores left out: the slowest thread "
                        "sets the pace of every matmul); set OMP_NUM_THREADS=<n> to override\n",
                engine, team, logical, phys);
    else
        fprintf(stderr, "[OMP] %s: %d physical-core threads instead of %d logical CPUs; "
                        "SMT can halve decode throughput on some CPUs (#718); "
                        "set OMP_NUM_THREADS=<n> to override\n",
                engine, phys, logical);
    coli_omp_probe_team(engine, team);
#else
    (void)engine;
#endif
}

/* The CPUs a team sized from the machine may keep busy: OpenMP's processor count (the
 * affinity mask), no more than a container's CPU quota lets run at once
 * (compat_cgroup_cpu_limit, #2914). For the teams a probe or a serving thread pool
 * sizes from the CPU count instead of taking coli_omp_tune_threads's team: under a
 * quota of 7.65 CPUs on a 96-CPU host, omp_get_num_procs() says 96. */
static inline int coli_cpu_budget(void)
{
#ifdef _OPENMP
    const int n = omp_get_num_procs();
    const int q = compat_cgroup_cpu_limit(NULL);
    return q > 0 && q < n ? q : n;
#else
    return 1;
#endif
}

/* 1 when an engine switch such as COLI_CUDA is on (set, and not 0/false/off/no) */
static inline int coli_omp_env_on(const char *name)
{
    const char *v = getenv(name);
    if (!v || !*v) return 0;
    return !(strcmp(v, "0") == 0 || strcmp(v, "false") == 0 || strcmp(v, "off") == 0 || strcmp(v, "no") == 0);
}

/* The hot team colibri.c gives itself, for the engines that only had the sizing above:
 * OMP_WAIT_POLICY=active with GOMP_SPINCOUNT=200000 (spin a moment between the many small
 * matmul regions of a token, then sleep, so a wait on the disk does not burn a core),
 * KMP_BLOCKTIME=200 for LLVM's libomp (#341: an idle team spinning forever), the team on
 * whole cores (OMP_PROC_BIND=close, OMP_PLACES=cores) and of fixed size. libgomp reads them
 * in its constructor, before main(): they are set, then the engine runs itself again, once
 * (Linux, FreeBSD). Not on macOS (#707: slower with Apple's libomp), not with CUDA or Metal
 * on (gpu_on; colibri.c leaves those out too), not under COLI_NO_OMP_TUNE; a variable the
 * user set wins. On Linux the engine keeps its own name (the path /proc/self/exe points to,
 * not "exe": a `pgrep -x olmoe` still finds it). Measured on an i5-13500 (6 P + 8 E cores),
 * served, Q4_K_M on the CPU: OLMoE decode 43.3 and 42.7 -> 47.9 and 47.6 tok/s, LFM3 33.7
 * and 33.8 -> 34.5 and 34.5, Qwen3.6 13.5 -> 13.9, Qwen3.8 IQ2_XS from the disk 7.9-8.1 -> 9.0-9.1
 * (its prompt 4% slower) (docs/tuning.md). */
#if defined(__FreeBSD__)
/* The running executable's path on FreeBSD, for the hot-team re-exec (here and in
 * colibri.c). The re-exec used /proc/curproc/file, which exists only when procfs is
 * mounted, and a default FreeBSD install does not mount it: execv failed, the engine
 * printed "re-exec failed, running untuned" and ran without the hot team. The kernel
 * gives the same path through the kern.proc.pathname sysctl (what `procstat -b`
 * reads), procfs or not. 1 with a NUL-terminated path in buf, else 0. */
static inline int coli_self_exe_path(char *buf, size_t n)
{
    int mib[4] = { CTL_KERN, KERN_PROC, KERN_PROC_PATHNAME, -1 };
    size_t len = n;
    if (n < 2 || sysctl(mib, 4, buf, &len, NULL, 0) != 0 || len < 2) return 0;
    buf[n - 1] = 0;
    return 1;
}
#endif

static inline void coli_omp_hot_team(char **argv, const char *engine, int gpu_on)
{
#if defined(_OPENMP) && !defined(__APPLE__) && (defined(__linux__) || defined(__FreeBSD__))
    if (getenv("COLI_OMP_TUNED") || getenv("COLI_NO_OMP_TUNE") || gpu_on || !argv || !argv[0]) return;
    setenv("OMP_WAIT_POLICY", "active", 0);
    setenv("GOMP_SPINCOUNT", "200000", 0);
    setenv("KMP_BLOCKTIME", "200", 0);
    setenv("OMP_PROC_BIND", "close", 0);
    setenv("OMP_PLACES", "cores", 0);
    setenv("OMP_DYNAMIC", "FALSE", 0);
    setenv("COLI_OMP_TUNED", "1", 1);
    fprintf(stderr, "[OMP] %s: hot-thread tuning, re-exec once (COLI_NO_OMP_TUNE=1 to skip)\n", engine);
#ifdef __linux__
#ifdef CPU_SETSIZE
    {   /* #471: a user's OMP_PROC_BIND already bound this thread to one place; the new image
         * would inherit that mask and put its whole team there */
        cpu_set_t all; CPU_ZERO(&all);
        long n = sysconf(_SC_NPROCESSORS_ONLN);
        if (n > CPU_SETSIZE) n = CPU_SETSIZE;
        for (long i = 0; i < n; i++) CPU_SET((int)i, &all);
        if (sched_setaffinity(0, sizeof all, &all) != 0) perror("[OMP] sched_setaffinity before the re-exec (continuing)");
    }
#endif
    char self[4096];
    ssize_t k = readlink("/proc/self/exe", self, sizeof self - 1);
    if (k > 0) { self[k] = 0; execv(self, argv); }
    execv("/proc/self/exe", argv);
#else
    {   char self[4096];   /* FreeBSD: the path from the kernel, no procfs needed */
        if (coli_self_exe_path(self, sizeof self)) execv(self, argv);
    }
    execv("/proc/curproc/file", argv);   /* the old path, for a system that mounts procfs */
#endif
    perror("[OMP] re-exec failed, running untuned");
#else
    (void)argv; (void)engine; (void)gpu_on;
#endif
}

#endif /* COLI_OMP_TUNE_H */
