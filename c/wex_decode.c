/* WEX record reconstruction: the WNM2 model registry, the decode of one record
 * (wex_decode.h), and the loader of the optional CUDA helper that decodes neural
 * records on the GPU. */
#include "wex_decode.h"
#include "wex4.h"
#include "wex_fast_decode.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>

/* ---- the CUDA helper (tools/wex_cuda_bridge.cu; docs/wex.md) -------------------
 * Opened once per process, lazily: the first model registration or neural decode.
 * Only when asked: COLI_WEX_CUDA_LIB=<path> names the library (kaira_wex_cuda.dll /
 * kaira_wex_cuda.so, make wex-cuda-helper); unset or "0", no library is looked for and
 * every record decodes on the CPU. It must report ABI 1 and export wex_cuda_decode_to_host; every other entry
 * point is resolved only when its feature bit is set. The library stays loaded until
 * the process exits (its pools keep CUDA workspaces). Other systems: no helper. */
#if defined(_WIN32) || defined(__linux__)
#define COLI_WEX_CUDA_LOADER 1
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#endif

typedef int (*WexCudaHostFn)(const WexEntry *, unsigned char *, size_t);
typedef int (*WexCudaDeviceFn)(const WexEntry *, const unsigned char *, int, unsigned char *const *,
                               unsigned char *const *);
typedef int (*WexCudaModelFn)(const unsigned char *, size_t);
typedef int (*WexCudaBatchFn)(int, const WexEntry *const *, const unsigned char *const *, int,
                              unsigned char *const *, unsigned char *const *, int *);
typedef int (*WexCudaBatchHostFn)(int, const WexEntry *const *, const unsigned char *const *, int,
                                  unsigned char *const *, int *);
typedef int (*WexCudaRegisterFn)(void *, size_t);
typedef int (*WexCudaUnregisterFn)(void *);
typedef unsigned int (*WexCudaWordFn)(void);

static struct {
    unsigned int features;          /* the loaded helper's usable bits | 0x80000000; 0: none */
    WexCudaHostFn to_host;
    WexCudaDeviceFn to_device;
    WexCudaModelFn model;
    WexCudaBatchFn batch;
    WexCudaBatchHostFn batch_host;
    WexCudaRegisterFn host_register;
    WexCudaUnregisterFn host_unregister;
    uint64_t host_decodes, device_decodes;
    int failure_reported;
} wex_cuda;
static pthread_once_t wex_cuda_once = PTHREAD_ONCE_INIT;

#ifdef COLI_WEX_CUDA_LOADER
static void *wex_cuda_symbol(void *library, const char *name) {
    void *symbol = NULL;
#ifdef _WIN32
    FARPROC address = GetProcAddress((HMODULE)library, name);
    if (address) memcpy(&symbol, &address, sizeof symbol);
#else
    symbol = dlsym(library, name);
#endif
    return symbol;
}

static void wex_cuda_close(void *library) {
#ifdef _WIN32
    FreeLibrary((HMODULE)library);
#else
    dlclose(library);
#endif
}

static void wex_cuda_load(void) {
    const char *configured = getenv("COLI_WEX_CUDA_LIB");
    char path[4096];
    void *library;
    WexCudaWordFn abi, features;
    unsigned int bits, have = 0u;
    if (!configured || !*configured || !strcmp(configured, "0")) return;   /* not asked for: the CPU */
    if (strlen(configured) >= sizeof path) {
        fprintf(stderr, "[wex] COLI_WEX_CUDA_LIB is too long; neural records decode on the CPU\n");
        return;
    }
    snprintf(path, sizeof path, "%s", configured);
#ifdef _WIN32
    /* its dependencies from its own directory and the system's, not the current one */
    library = (void *)LoadLibraryExA(path, NULL, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if (!library) library = (void *)LoadLibraryExA(path, NULL, 0);   /* a relative COLI_WEX_CUDA_LIB */
#else
    library = dlopen(path, RTLD_NOW | RTLD_LOCAL);
#endif
    if (!library) {
        fprintf(stderr, "[wex] the CUDA helper %s does not load; neural records decode on the CPU\n", path);
        return;
    }
    abi = (WexCudaWordFn)wex_cuda_symbol(library, "wex_cuda_helper_abi");
    wex_cuda.to_host = (WexCudaHostFn)wex_cuda_symbol(library, "wex_cuda_decode_to_host");
    if (!abi || abi() != 1u || !wex_cuda.to_host) {
        fprintf(stderr, "[wex] %s is not a CUDA helper of ABI 1; neural records decode on the CPU\n", path);
        wex_cuda.to_host = NULL;
        wex_cuda_close(library);
        return;
    }
    features = (WexCudaWordFn)wex_cuda_symbol(library, "wex_cuda_helper_features");
    bits = features ? features() : 0u;
    if (bits & COLI_WEX_CUDA_DEVICE)
        if ((wex_cuda.to_device = (WexCudaDeviceFn)wex_cuda_symbol(library, "wex_cuda_decode_to_device"))) have |= COLI_WEX_CUDA_DEVICE;
    if (bits & COLI_WEX_CUDA_REGISTER) {
        wex_cuda.host_register = (WexCudaRegisterFn)wex_cuda_symbol(library, "wex_cuda_host_register");
        wex_cuda.host_unregister = (WexCudaUnregisterFn)wex_cuda_symbol(library, "wex_cuda_host_unregister");
        if (wex_cuda.host_register && wex_cuda.host_unregister) have |= COLI_WEX_CUDA_REGISTER;
        else wex_cuda.host_register = NULL, wex_cuda.host_unregister = NULL;
    }
    if (bits & COLI_WEX_CUDA_NR2)
        if ((wex_cuda.model = (WexCudaModelFn)wex_cuda_symbol(library, "wex_cuda_register_model"))) have |= COLI_WEX_CUDA_NR2;
    if (bits & COLI_WEX_CUDA_BATCH)
        if ((wex_cuda.batch = (WexCudaBatchFn)wex_cuda_symbol(library, "wex_cuda_decode_batch_to_device"))) have |= COLI_WEX_CUDA_BATCH;
    if ((bits & COLI_WEX_CUDA_NR3) && (have & COLI_WEX_CUDA_NR2)) have |= COLI_WEX_CUDA_NR3;   /* WNR3 needs the registry */
    if (bits & COLI_WEX_CUDA_BATCH_HOST)
        if ((wex_cuda.batch_host = (WexCudaBatchHostFn)wex_cuda_symbol(library, "wex_cuda_decode_batch_to_host")))
            have |= COLI_WEX_CUDA_BATCH_HOST;
    __atomic_store_n(&wex_cuda.features, have | 0x80000000u, __ATOMIC_RELEASE);
    fprintf(stderr, "[wex] CUDA helper %s: ABI 1, features %u\n", path, have);
}
#else
static void wex_cuda_load(void) {}
#endif

static unsigned int wex_cuda_bits(void) {
    pthread_once(&wex_cuda_once, wex_cuda_load);
    return __atomic_load_n(&wex_cuda.features, __ATOMIC_ACQUIRE) & 0x7fffffffu;
}

COLI_WEX_DECODE_API unsigned int coli_wex_cuda_features(void) { return wex_cuda_bits(); }
COLI_WEX_DECODE_API uint64_t coli_wex_cuda_decode_count(void) {
    return __atomic_load_n(&wex_cuda.host_decodes, __ATOMIC_RELAXED);
}
COLI_WEX_DECODE_API uint64_t coli_wex_cuda_device_decode_count(void) {
    return __atomic_load_n(&wex_cuda.device_decodes, __ATOMIC_RELAXED);
}

/* WNM2 model registry: a few containers per process at most, never freed, so
 * a found pointer stays valid without holding the lock. */
#define WEX_NR2_MAX_MODELS 16
static pthread_mutex_t wex_nr2_mutex = PTHREAD_MUTEX_INITIALIZER;
static WexNr2Model wex_nr2_models[WEX_NR2_MAX_MODELS];
static int wex_nr2_model_count;
static int wex_nr2_in_helper[WEX_NR2_MAX_MODELS];   /* 1: registered with the CUDA helper too */
/* The fast CPU decoder's tables of each model (wex_fast_decode.h, 224 KiB), built when the model
 * is added and the CPU has the decoder's SIMD kernel; NULL: that model's records use the reference. */
static WexFastModel *wex_fast_models[WEX_NR2_MAX_MODELS];

static void wex_stats_arm(void);   /* COLI_WEX_STATS, below */

COLI_WEX_DECODE_API int coli_wex_register_model(const unsigned char *blob, size_t bytes, uint32_t *id) {
    WexNr2Model probe;
    int i, index = -1, result = 0;
    if (wex_nr2_model_open(blob, bytes, &probe)) return -1;
    wex_stats_arm();
    pthread_mutex_lock(&wex_nr2_mutex);
    for (i = 0; i < wex_nr2_model_count; i++)
        if (wex_nr2_models[i].id == probe.id) {
            if (memcmp(wex_nr2_models[i].blob, blob, bytes)) result = -1;   /* hash collision */
            index = i;
            break;
        }
    if (index < 0 && !result) {
        unsigned char *copy = wex_nr2_model_count < WEX_NR2_MAX_MODELS ? (unsigned char *)malloc(bytes) : NULL;
        if (!copy) result = -1;
        else {
            memcpy(copy, blob, bytes);
            index = wex_nr2_model_count;
            if (wex_nr2_model_open(copy, bytes, &wex_nr2_models[index])) { free(copy); result = -1; }
            else {
                if (wexf_cpu_simd() && (wex_fast_models[index] = (WexFastModel *)malloc(sizeof(WexFastModel))))
                    wex_fast_model_init(wex_fast_models[index], &wex_nr2_models[index]);
                wex_nr2_model_count++;
            }
        }
    }
    /* the helper keeps its own copy; one it refuses leaves this model's records on the CPU */
    if (!result && !wex_nr2_in_helper[index] && (wex_cuda_bits() & COLI_WEX_CUDA_NR2))
        wex_nr2_in_helper[index] = wex_cuda.model(wex_nr2_models[index].blob, bytes) == 0;
    pthread_mutex_unlock(&wex_nr2_mutex);
    if (!result && id) *id = probe.id;
    return result;
}

COLI_WEX_DECODE_API int coli_wex4_register_model(const unsigned char *stored, size_t bytes, size_t stored_bytes,
                                                 const unsigned char sha[32], uint32_t id) {
    unsigned char digest[32];
    WexNr2Model probe;
    size_t at;
    if (!stored || !sha || bytes != WEX_NR2_MODEL_BYTES || stored_bytes < bytes) return -1;
    wex4_sha256(stored, bytes, digest);
    if (memcmp(digest, sha, 32u)) return -1;
    for (at = bytes; at < stored_bytes; at++) if (stored[at]) return -1;
    if (wex_nr2_model_open(stored, bytes, &probe) || probe.id != id) return -1;   /* before it is registered */
    return coli_wex_register_model(stored, bytes, NULL);
}

static const WexNr2Model *wex_nr2_find(uint32_t id, int *in_helper) {
    const WexNr2Model *found = NULL;
    int i;
    pthread_mutex_lock(&wex_nr2_mutex);
    for (i = 0; i < wex_nr2_model_count; i++)
        if (wex_nr2_models[i].id == id) {
            found = &wex_nr2_models[i];
            if (in_helper) *in_helper = wex_nr2_in_helper[i];
            break;
        }
    pthread_mutex_unlock(&wex_nr2_mutex);
    return found;
}

COLI_WEX_DECODE_API int coli_wex_container_refused(uint32_t header_flags, const char *who) {
    const char *masked = getenv("DSV4_CUDA_MOE_MASKED");
    if (!(header_flags & WEX_HEADER_CANONICAL_ZERO)) return 0;
    /* -0 weights decode as +0: proven equal on the default CUDA MoE paths, not on
     * the masked DeepGEMM path (non-deterministic, emits -0 itself) */
    if (masked && atoi(masked) != 0) {
        fprintf(stderr, "[%s] value-exact (canonical-zero) container refused: "
                        "DSV4_CUDA_MOE_MASKED is not proven value-exact\n", who);
        return 1;
    }
    fprintf(stderr, "[%s] value-exact (canonical-zero) container\n", who);
    return 0;
}

/* COLI_WEX_FAST_DECODE: unset, empty or not 0 -> the fast CPU decoder for WNR3 records; 0 -> the
 * reference. Read at each decode (a getenv next to a decode of megabytes). */
static int wex_fast_wanted(void) {
    const char *value = getenv("COLI_WEX_FAST_DECODE");
    return value && *value ? atoi(value) != 0 : 1;
}

static uint64_t wex_fast_decodes;
static int wex_cpu_reported;
/* One line, the first time a WNR3 record decodes on the CPU: which decoder. */
static void wex_cpu_report(int fast, int has_kernel) {
    if (__atomic_load_n(&wex_cpu_reported, __ATOMIC_RELAXED) || __atomic_exchange_n(&wex_cpu_reported, 1, __ATOMIC_ACQ_REL))
        return;
    if (fast)
        fprintf(stderr, "[wex] WNR3 records decode on the CPU with the fast decoder (%s; COLI_WEX_FAST_DECODE=0: the reference)\n",
                wex_fast_kernels());
    else
        fprintf(stderr, "[wex] WNR3 records decode on the CPU with the reference decoder (%s)\n",
                has_kernel ? "COLI_WEX_FAST_DECODE=0" : "the fast decoder has no kernel for this CPU");
}

COLI_WEX_DECODE_API const char *coli_wex_cpu_decoder(void) {
    return wexf_cpu_simd() && wex_fast_wanted() ? wex_fast_kernels() : "reference";
}
COLI_WEX_DECODE_API uint64_t coli_wex_fast_decode_count(void) {
    return __atomic_load_n(&wex_fast_decodes, __ATOMIC_RELAXED);
}

/* COLI_WEX_STATS=1: one stderr line when the process exits, with what decoded the neural records
 * coli_wex_decode_record was given, in a form a script can read:
 *   [wex] stats: helper=H device=D fast=F reference=R failed=X cpu=AVX2
 * helper: by the CUDA helper into host slabs (with its host batch), device: by the helper into
 * device buffers, fast / reference: on the CPU, failed: refused or not decoded (-1), cpu: the CPU
 * decoder at exit. Armed by the first model registered or neural record decoded; unset or 0: nothing. */
static uint64_t wex_ref_decodes, wex_failed_decodes;
static int wex_stats_armed;
static void wex_stats_line(void) {
    fprintf(stderr, "[wex] stats: helper=%llu device=%llu fast=%llu reference=%llu failed=%llu cpu=%s\n",
            (unsigned long long)coli_wex_cuda_decode_count(), (unsigned long long)coli_wex_cuda_device_decode_count(),
            (unsigned long long)coli_wex_fast_decode_count(),
            (unsigned long long)__atomic_load_n(&wex_ref_decodes, __ATOMIC_RELAXED),
            (unsigned long long)__atomic_load_n(&wex_failed_decodes, __ATOMIC_RELAXED), coli_wex_cpu_decoder());
}
static void wex_stats_arm(void) {
    const char *value;
    if (__atomic_load_n(&wex_stats_armed, __ATOMIC_RELAXED)) return;
    value = getenv("COLI_WEX_STATS");
    if (!value || !*value || !atoi(value)) return;
    if (!__atomic_exchange_n(&wex_stats_armed, 1, __ATOMIC_ACQ_REL)) atexit(wex_stats_line);
}

/* ---- streams a record (kernel 1's lanes) ----------------------------------------------
 * How many streams kernel 1 interleaves is a matter of speed only (same bytes, same
 * verdicts for every count), and the best count depends on the core: 4 on the i7-7700K
 * where it was tuned, 6 on others. COLI_WEX_FAST_LANES = 1, 2, 3, 4, 6 or 8 fixes it;
 * unset, 0 or anything else: measured on the process's first records. The first
 * WEX_LANE_CHOICES x WEX_LANE_ROUNDS WNR3 records the fast decoder decodes (whichever
 * threads decode them) take the counts in turn, each timed on its thread's CPU clock; once
 * all are in, the count with the lowest time per decoded byte (best of its rounds) is used for every record
 * after, and one stderr line says which. A record that is refused does not count (its
 * turn goes to the next record). Records decoded meanwhile by other threads use
 * WEX_FAST_LANES. The counts share one interleaved sample, so a busy machine slows each
 * of them alike. */
#define WEX_LANE_CHOICES 4
#define WEX_LANE_ROUNDS 4
static const int wex_lane_choice[WEX_LANE_CHOICES] = {3, 4, 6, 8};
static struct {
    int chosen;                              /* the count in use; 0 while measuring */
    unsigned int ticket, done;               /* records handed a count to measure, and measured */
    double best[WEX_LANE_CHOICES];           /* CPU time per decoded MB, best round; 0: none yet */
} wex_lanes;
static pthread_mutex_t wex_lanes_mutex = PTHREAD_MUTEX_INITIALIZER;

/* The calling thread's CPU time in ms (cycles on Windows, where the thread times tick at 15.6 ms; the
 * counts are compared with each other only, so a cycle counts as a unit): time another thread took from
 * this one on a busy machine, e.g. 8 readers on 4 cores, is not counted against the count being measured. */
static double wex_now_ms(void) {
#ifdef _WIN32
    ULONG64 cycles = 0;
    QueryThreadCycleTime(GetCurrentThread(), &cycles);
    return (double)cycles * 1e-6;
#else
    struct timespec t;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t);
    return (double)t.tv_sec * 1e3 + (double)t.tv_nsec * 1e-6;
#endif
}

/* COLI_WEX_FAST_LANES: a count kernel 1 has, or 0 (measure). */
static int wex_lanes_fixed(void) {
    const char *value = getenv("COLI_WEX_FAST_LANES");
    const int n = value && *value ? atoi(value) : 0;
    return n == 1 || n == 2 || n == 3 || n == 4 || n == 6 || n == 8 ? n : 0;
}

/* The count for the next record; *choice >= 0 when that record is one of the measured. */
static int wex_lanes_pick(int *choice) {
    const int fixed = wex_lanes_fixed(), chosen = __atomic_load_n(&wex_lanes.chosen, __ATOMIC_ACQUIRE);
    unsigned int ticket;
    *choice = -1;
    if (fixed) return fixed;
    if (chosen) return chosen;
    ticket = __atomic_fetch_add(&wex_lanes.ticket, 1u, __ATOMIC_RELAXED);
    if (ticket >= WEX_LANE_CHOICES * WEX_LANE_ROUNDS) return WEX_FAST_LANES;
    *choice = (int)(ticket % WEX_LANE_CHOICES);
    return wex_lane_choice[*choice];
}

/* A measured record: decoded in ms, or refused (ms < 0): its ticket is handed out again, so every count gets its
 * rounds from records that decode. The last of them picks the count. */
static void wex_lanes_note(int choice, double ms, const WexEntry *entry) {
    const double mb = (double)entry->scale_bytes / 1e6 + (double)entry->weight_bytes / 1e6;
    int i, pick = 0;
    if (ms < 0.0) {
        __atomic_fetch_sub(&wex_lanes.ticket, 1u, __ATOMIC_RELAXED);
        return;
    }
    pthread_mutex_lock(&wex_lanes_mutex);
    if (!wex_lanes.best[choice] || ms / mb < wex_lanes.best[choice]) wex_lanes.best[choice] = ms / mb;
    if (++wex_lanes.done == WEX_LANE_CHOICES * WEX_LANE_ROUNDS) {
        for (i = 1; i < WEX_LANE_CHOICES; i++)
            if (wex_lanes.best[i] > 0.0 && (!wex_lanes.best[pick] || wex_lanes.best[i] < wex_lanes.best[pick])) pick = i;
        __atomic_store_n(&wex_lanes.chosen, wex_lane_choice[pick], __ATOMIC_RELEASE);
        fprintf(stderr, "[wex] fast decoder: %d streams a record (measured on the first %d records, CPU time per byte "
                        "against it:", wex_lane_choice[pick], WEX_LANE_CHOICES * WEX_LANE_ROUNDS);
        for (i = 0; i < WEX_LANE_CHOICES; i++)
            if (wex_lanes.best[i] > 0.0) fprintf(stderr, " %d: %.2f", wex_lane_choice[i], wex_lanes.best[i] / wex_lanes.best[pick]);
        fprintf(stderr, "; COLI_WEX_FAST_LANES=N fixes it)\n");
    }
    pthread_mutex_unlock(&wex_lanes_mutex);
}

COLI_WEX_DECODE_API int coli_wex_fast_lanes(void) {
    const int fixed = wex_lanes_fixed(), chosen = __atomic_load_n(&wex_lanes.chosen, __ATOMIC_ACQUIRE);
    return fixed ? fixed : chosen;
}

/* Whether the helper decodes this neural record: its model registered with it, and
 * feature 16 for a WNR3 record. */
static int wex_cuda_takes(const WexEntry *entry, int in_helper) {
    const unsigned int bits = wex_cuda_bits();
    return in_helper && (bits & COLI_WEX_CUDA_NR2) &&
           (entry->flags != WEX_ENTRY_NEURAL_RANS3 || (bits & COLI_WEX_CUDA_NR3));
}

static void wex_cuda_failed(int status) {
    if (!__atomic_exchange_n(&wex_cuda.failure_reported, 1, __ATOMIC_ACQ_REL))
        fprintf(stderr, "[wex] the CUDA helper did not decode a record (status %d); decoding it on the CPU\n", status);
}

COLI_WEX_DECODE_API int coli_wex_decode_record(const WexEntry *entry, unsigned char *slab, size_t capacity) {
    const WexNr2Model *model = NULL;
    int in_helper = 0;
    if (entry && wex_entry_nr_model(entry)) {
        /* the record names its model at bytes 4..7 */
        size_t at = wex_record_read_offset(entry);
        if (!slab || !at || at > capacity || entry->total_bytes < WEX_NR2_HEADER_BYTES ||
            entry->total_bytes > capacity - at) return -1;
        wex_stats_arm();
        model = wex_nr2_find(wex_read_u32_le(slab + at + 4u), &in_helper);
        if (!model) { __atomic_fetch_add(&wex_failed_decodes, UINT64_C(1), __ATOMIC_RELAXED); return -1; }
    }
    if (model && wex_cuda_takes(entry, in_helper)) {
        /* on any status but 0 the helper left the slab as it was: the CPU decodes the same record */
        const int status = wex_cuda.to_host(entry, slab, capacity);
        if (!status) {
            __atomic_fetch_add(&wex_cuda.host_decodes, UINT64_C(1), __ATOMIC_RELAXED);
            return 0;
        }
        wex_cuda_failed(status);
    }
    if (model && entry->flags == WEX_ENTRY_NEURAL_RANS3) {
        /* the CPU: the fast decoder where its tables were built (the CPU has its SIMD kernel) and
         * COLI_WEX_FAST_DECODE is not 0, one thread a record (the engines decode records on several
         * threads at once); the same checks and bytes as the reference, the slab untouched on -1 */
        const WexFastModel *tables = wex_fast_models[model - wex_nr2_models];
        const int fast = tables && wex_fast_wanted();
        wex_cpu_report(fast, tables != NULL);
        if (fast) {
            int choice;
            const WexFastOptions one = {1, 1, wex_lanes_pick(&choice), 1};
            const double started = wex_now_ms();
            const int status = wex_fast_decode_record(entry, slab, capacity, model, tables, &one);
            __atomic_fetch_add(status ? &wex_failed_decodes : &wex_fast_decodes, UINT64_C(1), __ATOMIC_RELAXED);
            if (choice >= 0) wex_lanes_note(choice, status ? -1.0 : wex_now_ms() - started, entry);
            return status;
        }
    }
    if (model) {
        const int status = wex_decode_nr2_record(entry, slab, capacity, model);
        __atomic_fetch_add(status ? &wex_failed_decodes : &wex_ref_decodes, UINT64_C(1), __ATOMIC_RELAXED);
        return status;
    }
    return wex_decode_record(entry, slab, capacity);
}

COLI_WEX_DECODE_API int coli_wex_decode_record_to(const WexEntry *entry, const unsigned char *record,
                                                  unsigned char *const scales[3], unsigned char *const weights[3]) {
    const WexNr2Model *model;
    WexNr2Layout layout;
    WexNr2Record view;
    uint64_t raw, aligned;
    int in_helper = 0, m;
    if (!entry || !record || !scales || !weights || !wex_entry_nr_model(entry) ||
        entry->total_bytes < WEX_NR2_HEADER_BYTES || wex_entry_raw_sizes(entry, &raw, &aligned)) return -1;
    for (m = 0; m < 3; m++) if (!scales[m] || !weights[m]) return -1;
    wex_stats_arm();
    model = wex_nr2_find(wex_read_u32_le(record + 4u), &in_helper);   /* the record names its model at bytes 4..7 */
    if (!model) { __atomic_fetch_add(&wex_failed_decodes, UINT64_C(1), __ATOMIC_RELAXED); return -1; }
    if (wex_cuda_takes(entry, in_helper)) {
        /* the helper decodes into a slab: the record at its end, as coli_wex_decode_record has it, the parts copied
         * out on 0; on any other status the CPU decodes the record below */
        const size_t at = wex_record_read_offset(entry);
        unsigned char *slab = at ? (unsigned char *)malloc((size_t)aligned) : NULL;
        if (slab) {
            int status;
            memcpy(slab + at, record, entry->total_bytes);
            status = wex_cuda.to_host(entry, slab, (size_t)aligned);
            if (!status) {
                for (m = 0; m < 3; m++) {
                    memcpy(scales[m], slab + entry->scale_offsets[m],
                           (size_t)entry->scale_shapes[m][0] * entry->scale_shapes[m][1]);
                    memcpy(weights[m], slab + entry->weight_offsets[m],
                           (size_t)entry->weight_shapes[m][0] * entry->weight_shapes[m][1]);
                }
                free(slab);
                __atomic_fetch_add(&wex_cuda.host_decodes, UINT64_C(1), __ATOMIC_RELAXED);
                return 0;
            }
            free(slab);
            wex_cuda_failed(status);
        }
    }
    wex_nr2_layout(entry, &layout);
    if (wex_nrx_parse(record, entry->total_bytes, &layout, model->id, wex_entry_nr_magic(entry), &view)) {
        __atomic_fetch_add(&wex_failed_decodes, UINT64_C(1), __ATOMIC_RELAXED);
        return -1;
    }
    if (entry->flags == WEX_ENTRY_NEURAL_RANS3) {   /* the fast decoder as in coli_wex_decode_record */
        const WexFastModel *tables = wex_fast_models[model - wex_nr2_models];
        const int fast = tables && wex_fast_wanted();
        wex_cpu_report(fast, tables != NULL);
        if (fast) {
            int choice;
            const WexFastOptions one = {1, 1, wex_lanes_pick(&choice), 1};
            const double started = wex_now_ms();
            const int status = wex_fast_decode_with(tables, &view, &layout, model, scales, weights, &one) ? -1 : 0;
            __atomic_fetch_add(status ? &wex_failed_decodes : &wex_fast_decodes, UINT64_C(1), __ATOMIC_RELAXED);
            if (choice >= 0) wex_lanes_note(choice, status ? -1.0 : wex_now_ms() - started, entry);
            return status;
        }
    }
    {
        const int status = wex_nr2_decode(&view, &layout, model, scales, weights) ? -1 : 0;
        __atomic_fetch_add(status ? &wex_failed_decodes : &wex_ref_decodes, UINT64_C(1), __ATOMIC_RELAXED);
        return status;
    }
}

/* A neural record the helper can take into device buffers: 0 yes, else the status 1 or -1. */
static int wex_cuda_device_ready(const WexEntry *entry, const unsigned char *record, unsigned int feature) {
    int in_helper = 0;
    if (!entry || !record || !wex_entry_nr_model(entry) || entry->total_bytes < WEX_NR2_HEADER_BYTES) return -1;
    if (!(wex_cuda_bits() & feature)) return 1;
    if (!wex_nr2_find(wex_read_u32_le(record + 4u), &in_helper)) return -1;   /* no such model: invalid here */
    return wex_cuda_takes(entry, in_helper) ? 0 : 1;
}

COLI_WEX_DECODE_API int coli_wex_cuda_decode_to_device(const WexEntry *entry, const unsigned char *record, int device,
                                                       unsigned char *const dst_scale[3],
                                                       unsigned char *const dst_weight[3]) {
    int status = wex_cuda_device_ready(entry, record, COLI_WEX_CUDA_DEVICE);
    if (status) return status;
    if (!dst_scale || !dst_weight) return -1;
    status = wex_cuda.to_device(entry, record, device, dst_scale, dst_weight);
    if (!status) __atomic_fetch_add(&wex_cuda.device_decodes, UINT64_C(1), __ATOMIC_RELAXED);
    return status;
}

/* The batches: records the helper cannot take report their status here and are not
 * passed; the others go in one call, in their order. */
static int wex_cuda_batch_call(int n, const WexEntry *const *entries, const unsigned char *const *records, int device,
                               unsigned char *const *dst_scale, unsigned char *const *dst_weight,
                               unsigned char *const *dst, int *status, unsigned int feature) {
    int i, k = 0, failed = 0;
    const WexEntry **e;
    const unsigned char **r;
    unsigned char **s = NULL, **w = NULL, **d = NULL;
    int *st, *at;
    if (n <= 0) return 0;
    if (!status) return n;
    if (!entries || !records || !(wex_cuda_bits() & feature)) {
        for (i = 0; i < n; i++) status[i] = 1;
        return n;
    }
    e = (const WexEntry **)malloc((size_t)n * sizeof *e);
    r = (const unsigned char **)malloc((size_t)n * sizeof *r);
    st = (int *)malloc((size_t)n * sizeof *st);
    at = (int *)malloc((size_t)n * sizeof *at);
    if (dst) d = (unsigned char **)malloc((size_t)n * sizeof *d);
    else {
        s = (unsigned char **)malloc((size_t)n * 3u * sizeof *s);
        w = (unsigned char **)malloc((size_t)n * 3u * sizeof *w);
    }
    if (!e || !r || !st || !at || (dst ? !d : !s || !w)) {
        for (i = 0; i < n; i++) status[i] = 1;
        failed = n;
        goto done;
    }
    for (i = 0; i < n; i++) {
        status[i] = wex_cuda_device_ready(entries[i], records[i], feature);
        if (!status[i] && (dst ? !dst[i] : !dst_scale || !dst_weight)) status[i] = -1;
        if (status[i]) { failed++; continue; }
        e[k] = entries[i]; r[k] = records[i]; at[k] = i;
        if (dst) d[k] = dst[i];
        else for (int m = 0; m < 3; m++) { s[3 * k + m] = dst_scale[3 * i + m]; w[3 * k + m] = dst_weight[3 * i + m]; }
        k++;
    }
    if (k) {
        if (dst) wex_cuda.batch_host(k, e, r, device, d, st);
        else wex_cuda.batch(k, e, r, device, s, w, st);
        for (i = 0; i < k; i++) {
            status[at[i]] = st[i];
            if (st[i]) failed++;
            else __atomic_fetch_add(dst ? &wex_cuda.host_decodes : &wex_cuda.device_decodes, UINT64_C(1),
                                    __ATOMIC_RELAXED);
        }
    }
done:
    free(e); free(r); free(st); free(at); free(s); free(w); free(d);
    return failed;
}

COLI_WEX_DECODE_API int coli_wex_cuda_decode_batch_to_device(int n, const WexEntry *const *entries,
                                                             const unsigned char *const *records, int device,
                                                             unsigned char *const *dst_scale,
                                                             unsigned char *const *dst_weight, int *status) {
    return wex_cuda_batch_call(n, entries, records, device, dst_scale, dst_weight, NULL, status, COLI_WEX_CUDA_BATCH);
}

COLI_WEX_DECODE_API int coli_wex_cuda_decode_batch_to_host(int n, const WexEntry *const *entries,
                                                           const unsigned char *const *records, int device,
                                                           unsigned char *const *dst, int *status) {
    if (!dst) {
        for (int i = 0; status && i < n; i++) status[i] = -1;
        return n > 0 ? n : 0;
    }
    return wex_cuda_batch_call(n, entries, records, device, NULL, NULL, dst, status, COLI_WEX_CUDA_BATCH_HOST);
}

COLI_WEX_DECODE_API int coli_wex_cuda_host_register(void *pointer, size_t bytes) {
    if (!pointer || !bytes) return -1;
    if (!(wex_cuda_bits() & COLI_WEX_CUDA_REGISTER)) return 1;
    return wex_cuda.host_register(pointer, bytes) ? 1 : 0;
}

COLI_WEX_DECODE_API int coli_wex_cuda_host_unregister(void *pointer) {
    if (!pointer) return -1;
    if (!(wex_cuda_bits() & COLI_WEX_CUDA_REGISTER)) return 1;
    return wex_cuda.host_unregister(pointer) ? 1 : 0;
}
