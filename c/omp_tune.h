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

#if defined(_WIN32)
#include <windows.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#else
#include <dirent.h>
#endif
#if defined(__linux__) || defined(__FreeBSD__)
#include <sched.h>
#include <unistd.h>
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
     * tiers are hw.perflevel0 .. hw.perflevel(N-1), fastest first, the last the
     * efficiency cores: an M1-M4 has two (perflevel0, its P-cores), an M6 three
     * (2 Super, 4 Performance, 6 Efficiency: 6, not perflevel0's 2, #2194); one
     * tier is every core. An Intel Mac has no perflevel*: hw.physicalcpu. */
    int n = 0; size_t sz = sizeof(n);
    if (sysctlbyname("hw.nperflevels", &n, &sz, NULL, 0) == 0 && n >= 1) {
        int tot = 0, last = n > 1 ? n - 1 : 1;
        for (int i = 0; i < last && i < 16; i++) {
            char nm[48]; snprintf(nm, sizeof nm, "hw.perflevel%d.logicalcpu", i);
            int c = 0; sz = sizeof(c);
            if (sysctlbyname(nm, &c, &sz, NULL, 0) == 0 && c > 0) tot += c;
        }
        if (tot > 0) return tot;
    }
    int v = 0; sz = sizeof(v);
    if (sysctlbyname("hw.perflevel0.logicalcpu", &v, &sz, NULL, 0) == 0 && v > 0) return v;
    v = 0; sz = sizeof(v);
    if (sysctlbyname("hw.physicalcpu", &v, &sz, NULL, 0) == 0 && v > 0) return v;
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

/* Dimensiona la squadra OpenMP sui core fisici. Rispetta OMP_NUM_THREADS se
 * l'utente l'ha impostata, e non fa nulla se il conteggio non e' affidabile.
 * `engine` finisce solo nella riga di log. */
static void coli_omp_tune_threads(const char *engine)
{
#ifdef _OPENMP
    const char *off = getenv("COLI_NO_OMP_TUNE");
    if (off) return;                       /* stesso kill-switch degli altri motori */
    if (getenv("OMP_NUM_THREADS")) return; /* l'utente comanda */

    int phys = coli_physical_cores();
    if (phys <= 0) return;                 /* sconosciuto -> default di OpenMP */
    int perf = coli_performance_cores();   /* a hybrid CPU: its P-cores pace the team */
    int team = perf > 0 && perf < phys ? perf : phys;
    int logical = omp_get_max_threads();
    if (team >= logical) return;           /* niente SMT da evitare: silenzio */

    omp_set_num_threads(team);
    if (team < phys)
        fprintf(stderr, "[OMP] %s: %d performance-core threads instead of %d logical CPUs "
                        "(%d physical cores, the efficiency cores left out: the slowest thread "
                        "sets the pace of every matmul); set OMP_NUM_THREADS=<n> to override\n",
                engine, team, logical, phys);
    else
        fprintf(stderr, "[OMP] %s: %d physical-core threads instead of %d logical CPUs; "
                        "SMT can halve decode throughput on some CPUs (#718); "
                        "set OMP_NUM_THREADS=<n> to override\n",
                engine, phys, logical);
#else
    (void)engine;
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
static void coli_omp_hot_team(char **argv, const char *engine, int gpu_on)
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
    execv("/proc/curproc/file", argv);
#endif
    perror("[OMP] re-exec failed, running untuned");
#else
    (void)argv; (void)engine; (void)gpu_on;
#endif
}

#endif /* COLI_OMP_TUNE_H */
