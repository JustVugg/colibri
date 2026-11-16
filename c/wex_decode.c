/* WEX record reconstruction: the WNM2 model registry, the decode of one record
 * (wex_decode.h), and the loader of the optional CUDA helper that decodes neural
 * records on the GPU. */
#include "wex_decode.h"
#include "wex4.h"
#include "wex_fast_decode.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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

COLI_WEX_DECODE_API int coli_wex_register_model(const unsigned char *blob, size_t bytes, uint32_t *id) {
    WexNr2Model probe;
    int i, index = -1, result = 0;
    if (wex_nr2_model_open(blob, bytes, &probe)) return -1;
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

/* Whether the helper decodes this neural record: its model registered with it, and
 * feature 16 for a WNR3 record. */
static int wex_cuda_takes(const WexEntry *entry, int in_helper) {
    const unsigned int bits = wex_cuda_bits();
    return in_helper && (bits & COLI_WEX_CUDA_NR2) &&
           (entry->flags != WEX_ENTRY_NEURAL_RANS3 || (bits & COLI_WEX_CUDA_NR3));
}

COLI_WEX_DECODE_API int coli_wex_decode_record(const WexEntry *entry, unsigned char *slab, size_t capacity) {
    const WexNr2Model *model = NULL;
    int in_helper = 0;
    if (entry && wex_entry_nr_model(entry)) {
        /* the record names its model at bytes 4..7 */
        size_t at = wex_record_read_offset(entry);
        if (!slab || !at || at > capacity || entry->total_bytes < WEX_NR2_HEADER_BYTES ||
            entry->total_bytes > capacity - at) return -1;
        model = wex_nr2_find(wex_read_u32_le(slab + at + 4u), &in_helper);
        if (!model) return -1;
    }
    if (model && wex_cuda_takes(entry, in_helper)) {
        /* on any status but 0 the helper left the slab as it was: the CPU decodes the same record */
        const int status = wex_cuda.to_host(entry, slab, capacity);
        if (!status) {
            __atomic_fetch_add(&wex_cuda.host_decodes, UINT64_C(1), __ATOMIC_RELAXED);
            return 0;
        }
        if (!__atomic_exchange_n(&wex_cuda.failure_reported, 1, __ATOMIC_ACQ_REL))
            fprintf(stderr, "[wex] the CUDA helper did not decode a record (status %d); decoding it on the CPU\n",
                    status);
    }
    if (model && entry->flags == WEX_ENTRY_NEURAL_RANS3) {
        /* the CPU: the fast decoder where its tables were built (the CPU has its SIMD kernel) and
         * COLI_WEX_FAST_DECODE is not 0, one thread a record (the engines decode records on several
         * threads at once); the same checks and bytes as the reference, the slab untouched on -1 */
        const WexFastModel *tables = wex_fast_models[model - wex_nr2_models];
        const int fast = tables && wex_fast_wanted();
        wex_cpu_report(fast, tables != NULL);
        if (fast) {
            static const WexFastOptions one = {1, 1, WEX_FAST_LANES, 1};
            const int status = wex_fast_decode_record(entry, slab, capacity, model, tables, &one);
            if (!status) __atomic_fetch_add(&wex_fast_decodes, UINT64_C(1), __ATOMIC_RELAXED);
            return status;
        }
    }
    if (model) return wex_decode_nr2_record(entry, slab, capacity, model);
    return wex_decode_record(entry, slab, capacity);
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
