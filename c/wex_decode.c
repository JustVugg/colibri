/* WEX record reconstruction on the CPU: the WNM2 model registry and the
 * decode of one record (wex_decode.h). Wex-WRN2's copy of this file also
 * loads the optional CUDA helper; that part comes with the GPU decode. */
#include "wex_decode.h"
#include "wex4.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

/* WNM2 model registry: a few containers per process at most, never freed, so
 * a found pointer stays valid without holding the lock. */
#define WEX_NR2_MAX_MODELS 16
static pthread_mutex_t wex_nr2_mutex = PTHREAD_MUTEX_INITIALIZER;
static WexNr2Model wex_nr2_models[WEX_NR2_MAX_MODELS];
static int wex_nr2_model_count;

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
            else wex_nr2_model_count++;
        }
    }
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

static const WexNr2Model *wex_nr2_find(uint32_t id) {
    const WexNr2Model *found = NULL;
    int i;
    pthread_mutex_lock(&wex_nr2_mutex);
    for (i = 0; i < wex_nr2_model_count; i++)
        if (wex_nr2_models[i].id == id) {
            found = &wex_nr2_models[i];
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

COLI_WEX_DECODE_API int coli_wex_decode_record(const WexEntry *entry, unsigned char *slab, size_t capacity) {
    const WexNr2Model *model = NULL;
    if (entry && wex_entry_nr_model(entry)) {
        /* the record names its model at bytes 4..7 */
        size_t at = wex_record_read_offset(entry);
        if (!slab || !at || at > capacity || entry->total_bytes < WEX_NR2_HEADER_BYTES ||
            entry->total_bytes > capacity - at) return -1;
        model = wex_nr2_find(wex_read_u32_le(slab + at + 4u));
        if (!model) return -1;
    }
    if (model) return wex_decode_nr2_record(entry, slab, capacity, model);
    return wex_decode_record(entry, slab, capacity);
}
