/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */
#include "wex_decode.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

/* Lazy, process-lifetime helper: no DLL probing for native/V2/palette records.
 * Keep the CPU reference available when CUDA or the optional DLL is absent.
 */
#ifdef _WIN32
typedef int (*WexCudaDecodeFn)(const WexEntry *, unsigned char *, size_t);
typedef unsigned int (*WexCudaAbiFn)(void);
typedef int (*WexCudaDeviceFn)(const WexEntry *, const unsigned char *, int, unsigned char *const *, unsigned char *const *);
typedef int (*WexCudaRegisterFn)(void *, size_t);
typedef int (*WexCudaUnregisterFn)(void *);
typedef int (*WexCudaModelFn)(const unsigned char *, size_t);
typedef int (*WexCudaBatchFn)(int, const WexEntry *const *, const unsigned char *const *, int,
                              unsigned char *const *, unsigned char *const *, int *);
static WexCudaBatchFn wex_cuda_batch_fn;     /* optional feature: a layer's WNR2 misses per call */
typedef int (*WexCudaBatchHostFn)(int, const WexEntry *const *, const unsigned char *const *, int,
                                  unsigned char *const *, int *);
static WexCudaBatchHostFn wex_cuda_batch_host_fn;   /* feature 32: a wave's records to host slabs */
static WexCudaUnregisterFn wex_cuda_unregister_fn;
static WexCudaDecodeFn wex_cuda_decode_fn;
static WexCudaModelFn wex_cuda_model_fn;     /* optional feature: WNR2 records */
static WexCudaDeviceFn wex_cuda_device_fn;   /* optional feature: direct to VRAM */
static WexCudaRegisterFn wex_cuda_register_fn;
static int wex_cuda_nr3;                     /* helper feature 16: WNR3 records (codec 5) */
static volatile LONG64 wex_cuda_device_successes;
static pthread_once_t wex_cuda_once = PTHREAD_ONCE_INIT;
static volatile LONG64 wex_cuda_successes;
static volatile LONG wex_cuda_failure_reported;
static void wex_cuda_initialize(void) {
    const char *configured = getenv("COLI_WEX_CUDA_LIB");
    char path[32768];
    HMODULE library;
    FARPROC symbol;
    WexCudaAbiFn abi = NULL;
    size_t length;
    if (configured && !strcmp(configured, "0")) return;
    if (configured && *configured) {
        length = strlen(configured);
        if (length >= sizeof(path)) return;
        memcpy(path, configured, length + 1u);
    } else {
        DWORD n = GetModuleFileNameA(NULL, path, (DWORD)sizeof(path));
        char *slash;
        if (!n || n >= sizeof(path)) return;
        slash = strrchr(path, '\\');
        if (!slash) return;
        length = (size_t)(slash - path) + 1u;
        if (length + sizeof("kaira_wex_cuda.dll") > sizeof(path)) return;
        memcpy(path + length, "kaira_wex_cuda.dll", sizeof("kaira_wex_cuda.dll"));
    }
    /* Restrict dependency lookup to the helper directory and Windows defaults. */
    library = LoadLibraryExA(path, NULL, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if (!library) {
        fprintf(stderr, "[wex] Neural CUDA helper unavailable; using the CPU decoder.\n");
        return;
    }
    symbol = GetProcAddress(library, "wex_cuda_helper_abi");
    if (symbol) memcpy(&abi, &symbol, sizeof(abi));
    if (!abi || abi() != 1u) { FreeLibrary(library); return; }
    symbol = GetProcAddress(library, "wex_cuda_decode_to_host");
    if (symbol) memcpy(&wex_cuda_decode_fn, &symbol, sizeof(wex_cuda_decode_fn));
    if (!wex_cuda_decode_fn) { FreeLibrary(library); return; }
    {
        WexCudaAbiFn features = NULL;
        symbol = GetProcAddress(library, "wex_cuda_helper_features");
        if (symbol) memcpy(&features, &symbol, sizeof(features));
        if (features && (features() & 1u)) {
            symbol = GetProcAddress(library, "wex_cuda_decode_to_device");
            if (symbol) memcpy(&wex_cuda_device_fn, &symbol, sizeof(wex_cuda_device_fn));
        }
        if (features && (features() & 2u)) {
            symbol = GetProcAddress(library, "wex_cuda_host_register");
            if (symbol) memcpy(&wex_cuda_register_fn, &symbol, sizeof(wex_cuda_register_fn));
            symbol = GetProcAddress(library, "wex_cuda_host_unregister");
            if (symbol) memcpy(&wex_cuda_unregister_fn, &symbol, sizeof(wex_cuda_unregister_fn));
        }
        if (features && (features() & 4u)) {
            symbol = GetProcAddress(library, "wex_cuda_register_model");
            if (symbol) memcpy(&wex_cuda_model_fn, &symbol, sizeof(wex_cuda_model_fn));
        }
        if (features && (features() & 8u)) {
            symbol = GetProcAddress(library, "wex_cuda_decode_batch_to_device");
            if (symbol) memcpy(&wex_cuda_batch_fn, &symbol, sizeof(wex_cuda_batch_fn));
        }
        wex_cuda_nr3 = features && (features() & 16u);
        if (features && (features() & 32u)) {
            symbol = GetProcAddress(library, "wex_cuda_decode_batch_to_host");
            if (symbol) memcpy(&wex_cuda_batch_host_fn, &symbol, sizeof(wex_cuda_batch_host_fn));
        }
    }
    /* The helper owns thread-local CUDA buffers; retain it until process exit. */
    fprintf(stderr, "[wex] Neural CUDA reconstruction enabled: %s\n", path);
}
#endif

/* WNR2 model registry: a few containers per process at most, never freed, so
 * a found pointer stays valid without holding the lock. */
#define WEX_NR2_MAX_MODELS 16
static pthread_mutex_t wex_nr2_mutex = PTHREAD_MUTEX_INITIALIZER;
static WexNr2Model wex_nr2_models[WEX_NR2_MAX_MODELS];
static int wex_nr2_model_count;
#ifdef _WIN32
static int wex_nr2_helper_ids[WEX_NR2_MAX_MODELS];   /* registered with the helper too */
#endif

int coli_wex_register_model(const unsigned char *blob, size_t bytes, uint32_t *id) {
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
        unsigned char *copy = wex_nr2_model_count < WEX_NR2_MAX_MODELS ? malloc(bytes) : NULL;
        if (!copy) result = -1;
        else {
            memcpy(copy, blob, bytes);
            index = wex_nr2_model_count;
            if (wex_nr2_model_open(copy, bytes, &wex_nr2_models[index])) { free(copy); result = -1; }
            else wex_nr2_model_count++;
        }
    }
#ifdef _WIN32
    if (!result) {
        pthread_once(&wex_cuda_once, wex_cuda_initialize);
        if (wex_cuda_model_fn && !wex_nr2_helper_ids[index])
            wex_nr2_helper_ids[index] = !wex_cuda_model_fn(wex_nr2_models[index].blob, bytes);
    }
#endif
    pthread_mutex_unlock(&wex_nr2_mutex);
    if (!result && id) *id = probe.id;
    return result;
}

static const WexNr2Model *wex_nr2_find(uint32_t id, int *in_helper) {
    const WexNr2Model *found = NULL;
    int i;
    pthread_mutex_lock(&wex_nr2_mutex);
    for (i = 0; i < wex_nr2_model_count; i++)
        if (wex_nr2_models[i].id == id) {
            found = &wex_nr2_models[i];
#ifdef _WIN32
            if (in_helper) *in_helper = wex_nr2_helper_ids[i];
#endif
            break;
        }
    pthread_mutex_unlock(&wex_nr2_mutex);
    return found;
}

int coli_wex_cuda_nr2_available(void) {
#ifdef _WIN32
    pthread_once(&wex_cuda_once, wex_cuda_initialize);
    return wex_cuda_model_fn != NULL;
#else
    return 0;
#endif
}

int coli_wex_cuda_nr3_available(void) {
#ifdef _WIN32
    pthread_once(&wex_cuda_once, wex_cuda_initialize);
    return wex_cuda_model_fn != NULL && wex_cuda_nr3;
#else
    return 0;
#endif
}

int coli_wex_cuda_device_available(void) {
#ifdef _WIN32
    pthread_once(&wex_cuda_once, wex_cuda_initialize);
    return wex_cuda_device_fn != NULL;
#else
    return 0;
#endif
}

int coli_wex_cuda_decode_to_device(const WexEntry *entry, const unsigned char *record, int device,
                                   unsigned char *const dst_scale[3], unsigned char *const dst_weight[3]) {
#ifdef _WIN32
    if (!entry || (entry->flags != WEX_ENTRY_NEURAL_RANS && !wex_entry_nr_model(entry))) return -1;
    pthread_once(&wex_cuda_once, wex_cuda_initialize);
    if (!wex_cuda_device_fn || (wex_entry_nr_model(entry) && !wex_cuda_model_fn) ||
        (entry->flags == WEX_ENTRY_NEURAL_RANS3 && !wex_cuda_nr3)) return 1;
    int status = wex_cuda_device_fn(entry, record, device, dst_scale, dst_weight);
    if (!status) InterlockedIncrement64(&wex_cuda_device_successes);
    return status;
#else
    (void)entry; (void)record; (void)device; (void)dst_scale; (void)dst_weight;
    return 1;
#endif
}

int coli_wex_cuda_batch_available(void) {
#ifdef _WIN32
    pthread_once(&wex_cuda_once, wex_cuda_initialize);
    return wex_cuda_batch_fn != NULL && wex_cuda_model_fn != NULL;
#else
    return 0;
#endif
}

int coli_wex_cuda_decode_batch_to_device(int n, const WexEntry *const *entries,
                                         const unsigned char *const *records, int device,
                                         unsigned char *const *dst_scale, unsigned char *const *dst_weight,
                                         int *status) {
    int i, failed;
    if (n <= 0) return 0;
    if (!status) return n;
#ifdef _WIN32
    pthread_once(&wex_cuda_once, wex_cuda_initialize);
    if (wex_cuda_batch_fn && wex_cuda_model_fn) {
        failed = wex_cuda_batch_fn(n, entries, records, device, dst_scale, dst_weight, status);
        for (i = 0; i < n; i++)
            if (!status[i]) InterlockedIncrement64(&wex_cuda_device_successes);
        return failed;
    }
#else
    (void)entries; (void)records; (void)device; (void)dst_scale; (void)dst_weight;
#endif
    for (i = 0; i < n; i++) status[i] = 1;
    failed = n;
    return failed;
}

int coli_wex_cuda_batch_host_available(void) {
#ifdef _WIN32
    pthread_once(&wex_cuda_once, wex_cuda_initialize);
    return wex_cuda_batch_host_fn != NULL && wex_cuda_model_fn != NULL;
#else
    return 0;
#endif
}

int coli_wex_cuda_decode_batch_to_host(int n, const WexEntry *const *entries,
                                       const unsigned char *const *records, int device,
                                       unsigned char *const *dst, int *status) {
    int i, failed;
    if (n <= 0) return 0;
    if (!status) return n;
#ifdef _WIN32
    pthread_once(&wex_cuda_once, wex_cuda_initialize);
    if (wex_cuda_batch_host_fn && wex_cuda_model_fn) {
        failed = wex_cuda_batch_host_fn(n, entries, records, device, dst, status);
        for (i = 0; i < n; i++)
            if (!status[i]) InterlockedIncrement64(&wex_cuda_successes);
        return failed;
    }
#else
    (void)entries; (void)records; (void)device; (void)dst;
#endif
    for (i = 0; i < n; i++) status[i] = 1;
    failed = n;
    return failed;
}

int coli_wex_cuda_host_register(void *pointer, size_t bytes) {
#ifdef _WIN32
    pthread_once(&wex_cuda_once, wex_cuda_initialize);
    return wex_cuda_register_fn ? wex_cuda_register_fn(pointer, bytes) : 1;
#else
    (void)pointer; (void)bytes;
    return 1;
#endif
}

int coli_wex_cuda_host_unregister(void *pointer) {
#ifdef _WIN32
    pthread_once(&wex_cuda_once, wex_cuda_initialize);
    return wex_cuda_unregister_fn ? wex_cuda_unregister_fn(pointer) : 1;
#else
    (void)pointer;
    return 1;
#endif
}

uint64_t coli_wex_cuda_device_decode_count(void) {
#ifdef _WIN32
    return (uint64_t)InterlockedCompareExchange64(&wex_cuda_device_successes, 0, 0);
#else
    return 0;
#endif
}

uint64_t coli_wex_cuda_decode_count(void) {
#ifdef _WIN32
    return (uint64_t)InterlockedCompareExchange64(&wex_cuda_successes, 0, 0);
#else
    return 0;
#endif
}

int coli_wex_container_refused(uint32_t header_flags, const char *who) {
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

int coli_wex_decode_record(const WexEntry *entry, unsigned char *slab, size_t capacity) {
    const WexNr2Model *model = NULL;
    int in_helper = 0;
    if (wex_entry_nr_model(entry)) {
        /* the record names its model at bytes 4..7 */
        size_t at = wex_record_read_offset(entry);
        if (!slab || !at || at > capacity || entry->total_bytes < WEX_NR2_HEADER_BYTES ||
            entry->total_bytes > capacity - at) return -1;
        model = wex_nr2_find(wex_read_u32_le(slab + at + 4u), &in_helper);
        if (!model) return -1;
    }
#ifdef _WIN32
    if (entry && (entry->flags == WEX_ENTRY_NEURAL_RANS ||
                  (entry->flags == WEX_ENTRY_NEURAL_RANS2 && in_helper) ||
                  (entry->flags == WEX_ENTRY_NEURAL_RANS3 && in_helper && wex_cuda_nr3))) {
        pthread_once(&wex_cuda_once, wex_cuda_initialize);
        if (wex_cuda_decode_fn) {
            int status = wex_cuda_decode_fn(entry, slab, capacity);
            if (!status) {
                InterlockedIncrement64(&wex_cuda_successes);
                return 0;
            }
            if (!InterlockedExchange(&wex_cuda_failure_reported, 1))
                fprintf(stderr, "[wex] CUDA reconstruction failed/unavailable (%d); using CPU fallback.\n", status);
        }
    }
#endif
    if (model) return wex_decode_nr2_record(entry, slab, capacity, model);
    return wex_decode_record(entry, slab, capacity);
}
