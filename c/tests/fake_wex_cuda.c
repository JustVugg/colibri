/* A CPU stand-in for the CUDA helper kaira_wex_cuda (tools/wex_cuda_bridge.cu), for
 * tests/test_wex_cuda_loader.c: the same exports and statuses, every decode done by the
 * reference decoder of wex_neural_rans.h, its "device" pointers plain host memory.
 * Built only for the tests, as tests/fake_wex_cuda.dll / .so; never shipped.
 *
 * Its knobs, read at each call:
 *   FAKE_WEX_ABI=<n>         wex_cuda_helper_abi returns n (default 1)
 *   FAKE_WEX_FEATURES=<n>    wex_cuda_helper_features returns n (default 1|2|4|16|32;
 *                            feature 8, the device batch, only when asked: 63)
 *   FAKE_WEX_HOST_FAIL=1     wex_cuda_decode_to_host returns 1 (CUDA unavailable)
 *   FAKE_WEX_BATCH_FAIL=<i>  the host batch scribbles over record i's dst and reports 1,
 *                            as a CUDA failure in the middle of its copy back would */
#include "../wex_format.h"
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define FAKE_EXPORT __declspec(dllexport)
#else
#define FAKE_EXPORT __attribute__((visibility("default")))
#endif

#define FAKE_MAX_MODELS 16   /* as the helper's WEX_CUDA_NR2_MAX_MODELS */
static WexNr2Model models[FAKE_MAX_MODELS];
static int model_count;

static int knob(const char *name, int fallback) {
    const char *v = getenv(name);
    return v && *v ? atoi(v) : fallback;
}

static const WexNr2Model *fake_model(const unsigned char *record) {
    for (int i = 0; i < model_count; i++)
        if (models[i].id == wex_read_u32_le(record + 4u)) return &models[i];
    return NULL;
}

FAKE_EXPORT unsigned int wex_cuda_helper_abi(void) { return (unsigned int)knob("FAKE_WEX_ABI", 1); }
FAKE_EXPORT unsigned int wex_cuda_helper_features(void) { return (unsigned int)knob("FAKE_WEX_FEATURES", 1 | 2 | 4 | 16 | 32); }

/* 0 registered, or already with the same bytes; -1 invalid, or another blob with the same id; 1 full */
FAKE_EXPORT int wex_cuda_register_model(const unsigned char *blob, size_t bytes) {
    WexNr2Model probe;
    unsigned char *copy;
    if (wex_nr2_model_open(blob, bytes, &probe)) return -1;
    for (int i = 0; i < model_count; i++)
        if (models[i].id == probe.id) return memcmp(models[i].blob, blob, WEX_NR2_MODEL_BYTES) ? -1 : 0;
    copy = model_count < FAKE_MAX_MODELS ? malloc(bytes) : NULL;
    if (!copy) return 1;
    memcpy(copy, blob, bytes);
    if (wex_nr2_model_open(copy, bytes, &models[model_count])) { free(copy); return -1; }
    model_count++;
    return 0;
}

/* record i (host memory) decoded into a fresh slab; NULL and *status when it is not */
static unsigned char *fake_decode(const WexEntry *entry, const unsigned char *record, uint64_t *raw, int *status) {
    const WexNr2Model *model = record ? fake_model(record) : NULL;
    uint64_t aligned;
    unsigned char *slab;
    *status = model ? 0 : 1;   /* a model it was not given: 1, as the helper */
    if (!model) return NULL;
    if (wex_entry_raw_sizes(entry, raw, &aligned) || entry->total_bytes >= aligned) { *status = -1; return NULL; }
    if (!(slab = malloc((size_t)aligned))) { *status = 1; return NULL; }
    memcpy(slab + (aligned - entry->total_bytes), record, entry->total_bytes);
    if (wex_decode_nr2_record(entry, slab, (size_t)aligned, model)) { free(slab); *status = -1; return NULL; }
    return slab;
}

FAKE_EXPORT int wex_cuda_decode_to_host(const WexEntry *entry, unsigned char *slab, size_t capacity) {
    size_t at = wex_record_read_offset(entry);
    uint64_t raw;
    int status;
    unsigned char *out;
    if (knob("FAKE_WEX_HOST_FAIL", 0)) return 1;
    if (!at || at >= capacity) return -1;
    if (!(out = fake_decode(entry, slab + at, &raw, &status))) return status;
    memcpy(slab, out, (size_t)raw);
    free(out);
    return 0;
}

FAKE_EXPORT int wex_cuda_decode_to_device(const WexEntry *entry, const unsigned char *record, int device,
                                          unsigned char *const *scale, unsigned char *const *weight) {
    uint64_t raw;
    int status;
    unsigned char *out = fake_decode(entry, record, &raw, &status);
    (void)device;
    if (!out) return status;
    for (int m = 0; m < 3; m++) {   /* written only once the record decoded */
        memcpy(scale[m], out + entry->scale_offsets[m], (size_t)entry->scale_shapes[m][0] * entry->scale_shapes[m][1]);
        memcpy(weight[m], out + entry->weight_offsets[m], (size_t)entry->weight_shapes[m][0] * entry->weight_shapes[m][1]);
    }
    free(out);
    return 0;
}

FAKE_EXPORT int wex_cuda_decode_batch_to_device(int n, const WexEntry *const *entries, const unsigned char *const *records,
                                                int device, unsigned char *const *scale, unsigned char *const *weight,
                                                int *status) {
    int failed = 0;
    for (int i = 0; i < n; i++) {
        status[i] = wex_cuda_decode_to_device(entries[i], records[i], device, scale + 3 * i, weight + 3 * i);
        failed += status[i] != 0;
    }
    return failed;
}

FAKE_EXPORT int wex_cuda_decode_batch_to_host(int n, const WexEntry *const *entries, const unsigned char *const *records,
                                              int device, unsigned char *const *dst, int *status) {
    const int fail = knob("FAKE_WEX_BATCH_FAIL", -1);
    int failed = 0;
    (void)device;
    for (int i = 0; i < n; i++) {
        uint64_t raw = 0;
        unsigned char *out = fake_decode(entries[i], records[i], &raw, &status[i]);
        if (out) { memcpy(dst[i], out, (size_t)raw); free(out); }
        if (i == fail) {
            wex_entry_raw_sizes(entries[i], &raw, &(uint64_t){0});
            memset(dst[i], 0x5a, (size_t)raw);
            status[i] = 1;
        }
        failed += status[i] != 0;
    }
    return failed;
}

FAKE_EXPORT int wex_cuda_host_register(void *pointer, size_t bytes) { return pointer && bytes ? 0 : -1; }
FAKE_EXPORT int wex_cuda_host_unregister(void *pointer) { return pointer ? 0 : -1; }
