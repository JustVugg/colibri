#ifndef COLI_GEMMA4_BACKEND_H
#define COLI_GEMMA4_BACKEND_H

#include <stddef.h>
#include <stdint.h>

#include "model.h"

#define COLI_GEMMA4_PATH_MAX 2048
#define COLI_GEMMA4_ERROR_MAX 512

typedef struct {
    uint32_t layer;
    uint32_t expert_count;
    uint32_t model_width;
    uint32_t expert_width;
    uint64_t payload_bytes;
    uint64_t record_stride;
    uint64_t component_bytes[3];
    uint64_t source_offset[3];
    uint64_t source_expert_stride[3];
    uint64_t source_within_expert_offset[3];
    uint64_t scale_offset;
    uint32_t scale_scalar_bytes;
    int has_scale;
    char packed_file[128];
} coli_gemma4_layer;

typedef struct coli_gemma4_cache_slot coli_gemma4_cache_slot;
typedef struct coli_gemma4_prefetch_state coli_gemma4_prefetch_state;

typedef struct {
    uint32_t capacity;
    uint32_t resident;
    uint32_t pinned_capacity;
    uint64_t hits;
    uint64_t pinned_hits;
    uint64_t misses;
    uint64_t preloads;
    uint64_t prefetch_launches;
    uint64_t prefetched_records;
    uint64_t lookahead_launches;
    uint64_t lookahead_records;
    uint64_t lookahead_matches;
    uint64_t lookahead_selected;
    uint64_t evictions;
    uint64_t promotions;
    uint64_t bytes_loaded;
} coli_gemma4_cache_stats;

typedef struct {
    coli_model_config config;
    coli_gemma4_layer *layers;
    uint32_t layer_count;
    coli_gemma4_cache_slot *cache_slots;
    uint32_t cache_capacity;
    uint32_t cache_resident;
    uint32_t cache_pinned_capacity;
    uint32_t *cache_usage;
    uint32_t *cache_heat;
    uint32_t *cache_last;
    uint32_t cache_access_clock;
    uint64_t cache_clock;
    uint64_t cache_hits;
    uint64_t cache_pinned_hits;
    uint64_t cache_misses;
    uint64_t cache_preloads;
    uint64_t cache_prefetch_launches;
    uint64_t cache_prefetched_records;
    uint64_t cache_lookahead_launches;
    uint64_t cache_lookahead_records;
    uint64_t cache_lookahead_matches;
    uint64_t cache_lookahead_selected;
    uint64_t cache_evictions;
    uint64_t cache_promotions;
    uint64_t cache_bytes_loaded;
    coli_gemma4_prefetch_state *prefetch;
    int lookahead_enabled;
    void *cuda;
    char source[COLI_GEMMA4_PATH_MAX];
    char packed_dir[COLI_GEMMA4_PATH_MAX];
    char last_error[COLI_GEMMA4_ERROR_MAX];
} coli_gemma4_backend;

/* Open a g4lab packed-expert directory and validate its manifest. */
int coli_gemma4_open_packed(coli_gemma4_backend *backend, const char *packed_dir);
void coli_gemma4_close(coli_gemma4_backend *backend);
const char *coli_gemma4_last_error(const coli_gemma4_backend *backend);
const coli_gemma4_layer *coli_gemma4_find_layer(
    const coli_gemma4_backend *backend, uint32_t layer);
int coli_gemma4_has_packed_layer(
    const coli_gemma4_backend *backend, const coli_gemma4_layer *layer);

/* Retain exact encoded expert records in a bounded global LRU. Zero disables it. */
int coli_gemma4_cache_configure(coli_gemma4_backend *backend, uint32_t slots);
/* Protect this many slots with Colibri's live LFRU learning policy. */
int coli_gemma4_cache_set_pinned(coli_gemma4_backend *backend, uint32_t slots);
/* Overlap missing-record loads with the model's resident dense branch. */
int coli_gemma4_prefetch_configure(coli_gemma4_backend *backend, int enabled);
int coli_gemma4_lookahead_configure(coli_gemma4_backend *backend, int enabled);
int coli_gemma4_cuda_configure(coli_gemma4_backend *backend, int device);
void coli_gemma4_cache_get_stats(const coli_gemma4_backend *backend,
                                 coli_gemma4_cache_stats *stats);
/* Load/save Colibri-compatible "layer expert count" cumulative usage. */
int coli_gemma4_usage_load(coli_gemma4_backend *backend, const char *path,
                           uint64_t *selections);
int coli_gemma4_usage_save(coli_gemma4_backend *backend, const char *path,
                           uint64_t *selections);

/* Adapt the concrete packed backend to Colibri's placement-independent ABI. */
coli_expert_backend coli_gemma4_expert_backend(coli_gemma4_backend *backend);

/* Public numerical helpers are kept small so unit tests can validate kernels. */
float coli_gemma4_fp16_to_fp32(uint16_t value);
float coli_gemma4_gelu_tanh(float value);
int coli_gemma4_q4_0_matvec(const uint8_t *weights, size_t rows, size_t columns,
                            const float *input, float *output);
int coli_gemma4_q6_k_row(const uint8_t *weights, size_t rows, size_t columns,
                         size_t row, float *output);
int coli_gemma4_q6_k_matvec(const uint8_t *weights, size_t rows, size_t columns,
                            const float *input, float *output);
uint64_t coli_gemma4_checksum_f32(const float *values, size_t count);

#endif
