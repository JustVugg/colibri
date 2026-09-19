#define _CRT_SECURE_NO_WARNINGS
#define _FILE_OFFSET_BITS 64

#include "gemma4_backend.h"
#include "gemma4_cuda.h"

#include <errno.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "json.h"
#include "tier.h"

#ifdef _WIN32
#include <windows.h>
#include <process.h>
#define g4_seek _fseeki64
#define g4_tell _ftelli64
typedef __int64 g4_off_t;
#else
#include <pthread.h>
#include <sys/types.h>
#define g4_seek fseeko
#define g4_tell ftello
typedef off_t g4_off_t;
#endif

enum {
    G4_ROLE_GATE = 0,
    G4_ROLE_UP = 1,
    G4_ROLE_DOWN = 2
};

struct coli_gemma4_cache_slot {
    uint32_t layer;
    uint32_t expert;
    uint8_t *payload;
    size_t payload_capacity;
    uint64_t last_used;
    float scale;
    int valid;
};

struct coli_gemma4_prefetch_state {
    coli_gemma4_backend *backend;
    const coli_gemma4_layer *descriptor;
    uint32_t *expert_ids;
    uint32_t count;
    uint32_t capacity;
    int result;
    int active;
    int lookahead;
#ifdef _WIN32
    HANDLE thread;
#else
    pthread_t thread;
#endif
};

static void g4_prefetch_destroy(coli_gemma4_backend *backend);

static void g4_set_error(coli_gemma4_backend *backend, const char *fmt, ...) {
    va_list args;
    if (!backend) return;
    va_start(args, fmt);
    vsnprintf(backend->last_error, sizeof(backend->last_error), fmt, args);
    va_end(args);
}

static void g4_free_json(jval *value) {
    int i;
    if (!value) return;
    if (value->t == J_OBJ) {
        for (i = 0; i < value->len; ++i) {
            free(value->keys[i]);
            g4_free_json(value->kids[i]);
        }
        free(value->keys);
        free(value->kids);
    } else if (value->t == J_ARR) {
        for (i = 0; i < value->len; ++i) g4_free_json(value->kids[i]);
        free(value->kids);
    } else if (value->t == J_STR) {
        free(value->str);
    }
    free(value);
}

static int g4_copy_string(coli_gemma4_backend *backend, char *destination,
                          size_t capacity, const char *value, const char *label) {
    size_t length;
    if (!value) {
        g4_set_error(backend, "manifest field %s is missing", label);
        return -1;
    }
    length = strlen(value);
    if (length + 1 > capacity) {
        g4_set_error(backend, "manifest field %s is too long", label);
        return -1;
    }
    memcpy(destination, value, length + 1);
    return 0;
}

static int g4_number_u64(coli_gemma4_backend *backend, jval *object,
                         const char *key, uint64_t *result) {
    jval *value = json_get(object, key);
    double number;
    uint64_t converted;
    if (!value || value->t != J_NUM) {
        g4_set_error(backend, "manifest field %s is missing or not numeric", key);
        return -1;
    }
    number = value->num;
    if (!isfinite(number) || number < 0.0 || number > 9007199254740991.0 ||
        floor(number) != number) {
        g4_set_error(backend, "manifest field %s is not an exact non-negative integer", key);
        return -1;
    }
    converted = (uint64_t)number;
    *result = converted;
    return 0;
}

static int g4_number_u32(coli_gemma4_backend *backend, jval *object,
                         const char *key, uint32_t *result) {
    uint64_t value;
    if (g4_number_u64(backend, object, key, &value) != 0) return -1;
    if (value > UINT32_MAX) {
        g4_set_error(backend, "manifest field %s exceeds uint32", key);
        return -1;
    }
    *result = (uint32_t)value;
    return 0;
}

static const char *g4_string(jval *object, const char *key) {
    jval *value = json_get(object, key);
    return value && value->t == J_STR ? value->str : NULL;
}

static int g4_is_absolute_path(const char *path) {
    if (!path || !*path) return 0;
    if (path[0] == '/' || path[0] == '\\') return 1;
    return ((path[0] >= 'A' && path[0] <= 'Z') ||
            (path[0] >= 'a' && path[0] <= 'z')) && path[1] == ':';
}

static int g4_join_path(coli_gemma4_backend *backend, char *destination,
                        size_t capacity, const char *directory, const char *name) {
    size_t directory_length;
    const char *separator;
    int written;
    if (!directory || !name) {
        g4_set_error(backend, "cannot join a null path");
        return -1;
    }
    directory_length = strlen(directory);
    separator = (directory_length && (directory[directory_length - 1] == '/' ||
                                      directory[directory_length - 1] == '\\')) ? "" : "/";
    written = snprintf(destination, capacity, "%s%s%s", directory, separator, name);
    if (written < 0 || (size_t)written >= capacity) {
        g4_set_error(backend, "path is too long: %s/%s", directory, name);
        return -1;
    }
    return 0;
}

static int g4_safe_packed_name(const char *name) {
    if (!name || !*name || strstr(name, "..")) return 0;
    return !strchr(name, '/') && !strchr(name, '\\') && !strchr(name, ':');
}

static char *g4_read_text(coli_gemma4_backend *backend, const char *path) {
    FILE *file;
    g4_off_t length;
    char *text;
    file = fopen(path, "rb");
    if (!file) {
        g4_set_error(backend, "cannot open manifest %s: %s", path, strerror(errno));
        return NULL;
    }
    if (g4_seek(file, 0, SEEK_END) != 0 || (length = g4_tell(file)) < 0 ||
        length > (64LL << 20) || g4_seek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        g4_set_error(backend, "manifest is unreadable or larger than 64 MiB: %s", path);
        return NULL;
    }
    text = (char *)malloc((size_t)length + 1);
    if (!text) {
        fclose(file);
        g4_set_error(backend, "out of memory reading manifest");
        return NULL;
    }
    if (fread(text, 1, (size_t)length, file) != (size_t)length) {
        free(text);
        fclose(file);
        g4_set_error(backend, "short read from manifest %s", path);
        return NULL;
    }
    text[length] = 0;
    fclose(file);
    return text;
}

static uint32_t g4_load_u32(const uint8_t *bytes) {
    return (uint32_t)bytes[0] |
           ((uint32_t)bytes[1] << 8) |
           ((uint32_t)bytes[2] << 16) |
           ((uint32_t)bytes[3] << 24);
}

static uint64_t g4_load_u64(const uint8_t *bytes) {
    return (uint64_t)g4_load_u32(bytes) |
           ((uint64_t)g4_load_u32(bytes + 4) << 32);
}

static int g4_parse_component(coli_gemma4_backend *backend,
                              coli_gemma4_layer *layer, jval *component) {
    const char *role = g4_string(component, "role");
    const char *type_name = g4_string(component, "type_name");
    uint32_t ggml_type;
    uint64_t slice_bytes;
    uint64_t source_offset;
    uint64_t source_expert_stride;
    uint64_t source_within_expert_offset;
    int role_index;
    if (!role || !type_name ||
        g4_number_u32(backend, component, "ggml_type", &ggml_type) != 0 ||
        g4_number_u64(backend, component, "slice_bytes", &slice_bytes) != 0 ||
        g4_number_u64(backend, component, "source_offset", &source_offset) != 0 ||
        g4_number_u64(backend, component, "source_expert_stride",
                      &source_expert_stride) != 0 ||
        g4_number_u64(backend, component, "source_within_expert_offset",
                      &source_within_expert_offset) != 0) {
        return -1;
    }
    if (strcmp(role, "gate") == 0) role_index = G4_ROLE_GATE;
    else if (strcmp(role, "up") == 0) role_index = G4_ROLE_UP;
    else if (strcmp(role, "down") == 0) role_index = G4_ROLE_DOWN;
    else {
        g4_set_error(backend, "unsupported expert component role: %s", role);
        return -1;
    }
    if (ggml_type != 2 || strcmp(type_name, "Q4_0") != 0) {
        g4_set_error(backend, "Gemma packed experts must use Q4_0");
        return -1;
    }
    if (layer->component_bytes[role_index] != 0) {
        g4_set_error(backend, "duplicate %s component in layer %u", role, layer->layer);
        return -1;
    }
    layer->component_bytes[role_index] = slice_bytes;
    layer->source_offset[role_index] = source_offset;
    layer->source_expert_stride[role_index] = source_expert_stride;
    layer->source_within_expert_offset[role_index] = source_within_expert_offset;
    return 0;
}

static void g4_cache_clear(coli_gemma4_backend *backend) {
    uint32_t i;
    if (!backend) return;
    for (i = 0; i < backend->cache_capacity; ++i)
        free(backend->cache_slots[i].payload);
    free(backend->cache_slots);
    free(backend->cache_usage);
    free(backend->cache_heat);
    free(backend->cache_last);
    backend->cache_slots = NULL;
    backend->cache_usage = NULL;
    backend->cache_heat = NULL;
    backend->cache_last = NULL;
    backend->cache_capacity = 0;
    backend->cache_resident = 0;
    backend->cache_pinned_capacity = 0;
    backend->cache_access_clock = 0;
    backend->cache_clock = 0;
    backend->cache_hits = 0;
    backend->cache_pinned_hits = 0;
    backend->cache_misses = 0;
    backend->cache_preloads = 0;
    backend->cache_prefetch_launches = 0;
    backend->cache_prefetched_records = 0;
    backend->cache_lookahead_launches = 0;
    backend->cache_lookahead_records = 0;
    backend->cache_lookahead_matches = 0;
    backend->cache_lookahead_selected = 0;
    backend->cache_evictions = 0;
    backend->cache_promotions = 0;
    backend->cache_bytes_loaded = 0;
}

static int g4_validate_layer(coli_gemma4_backend *backend,
                             const coli_gemma4_layer *layer) {
    uint64_t gate_expected, down_expected, sum;
    if (!layer->expert_count || !layer->model_width || !layer->expert_width ||
        layer->model_width % 32 != 0 || layer->expert_width % 32 != 0) {
        g4_set_error(backend, "layer %u has invalid expert dimensions", layer->layer);
        return -1;
    }
    gate_expected = (uint64_t)layer->expert_width *
                    ((uint64_t)layer->model_width / 32) * 18;
    down_expected = (uint64_t)layer->model_width *
                    ((uint64_t)layer->expert_width / 32) * 18;
    if (layer->component_bytes[G4_ROLE_GATE] != gate_expected ||
        layer->component_bytes[G4_ROLE_UP] != gate_expected ||
        layer->component_bytes[G4_ROLE_DOWN] != down_expected) {
        g4_set_error(backend, "layer %u Q4_0 component sizes do not match %u -> %u -> %u",
                     layer->layer, layer->model_width, layer->expert_width,
                     layer->model_width);
        return -1;
    }
    if (layer->source_within_expert_offset[G4_ROLE_GATE] > UINT64_MAX - gate_expected ||
        layer->source_within_expert_offset[G4_ROLE_UP] > UINT64_MAX - gate_expected ||
        layer->source_within_expert_offset[G4_ROLE_DOWN] > UINT64_MAX - down_expected ||
        layer->source_expert_stride[G4_ROLE_GATE] <
            layer->source_within_expert_offset[G4_ROLE_GATE] + gate_expected ||
        layer->source_expert_stride[G4_ROLE_UP] <
            layer->source_within_expert_offset[G4_ROLE_UP] + gate_expected ||
        layer->source_expert_stride[G4_ROLE_DOWN] <
            layer->source_within_expert_offset[G4_ROLE_DOWN] + down_expected) {
        g4_set_error(backend, "layer %u has invalid GGUF component strides",
                     layer->layer);
        return -1;
    }
    sum = layer->component_bytes[0] + layer->component_bytes[1] +
          layer->component_bytes[2];
    if (sum != layer->payload_bytes || layer->record_stride < layer->payload_bytes ||
        layer->record_stride % 4096 != 0) {
        g4_set_error(backend, "layer %u has inconsistent payload/record sizes", layer->layer);
        return -1;
    }
    if (!g4_safe_packed_name(layer->packed_file)) {
        g4_set_error(backend, "layer %u has an unsafe packed filename", layer->layer);
        return -1;
    }
    return 0;
}

static int g4_parse_layer(coli_gemma4_backend *backend,
                          coli_gemma4_layer *layer, jval *value) {
    jval *components;
    jval *scale;
    const char *layout;
    const char *packed_file;
    uint32_t i;
    memset(layer, 0, sizeof(*layer));
    if (!value || value->t != J_OBJ ||
        g4_number_u32(backend, value, "layer", &layer->layer) != 0 ||
        g4_number_u32(backend, value, "expert_count", &layer->expert_count) != 0 ||
        g4_number_u32(backend, value, "model_width", &layer->model_width) != 0 ||
        g4_number_u32(backend, value, "expert_width", &layer->expert_width) != 0 ||
        g4_number_u64(backend, value, "payload_bytes", &layer->payload_bytes) != 0 ||
        g4_number_u64(backend, value, "record_stride", &layer->record_stride) != 0) {
        return -1;
    }
    layout = g4_string(value, "source_layout");
    packed_file = g4_string(value, "packed_file");
    if (!layout || strcmp(layout, "fused_gate_up") != 0) {
        g4_set_error(backend, "layer %u is not fused_gate_up", layer->layer);
        return -1;
    }
    if (g4_copy_string(backend, layer->packed_file, sizeof(layer->packed_file),
                       packed_file, "packed_file") != 0) return -1;

    scale = json_get(value, "expert_scale");
    if (scale && scale->t == J_OBJ) {
        uint32_t scale_type;
        if (g4_number_u32(backend, scale, "ggml_type", &scale_type) != 0 ||
            g4_number_u64(backend, scale, "source_offset", &layer->scale_offset) != 0 ||
            g4_number_u32(backend, scale, "scalar_bytes", &layer->scale_scalar_bytes) != 0) {
            return -1;
        }
        if (scale_type != 0 || layer->scale_scalar_bytes != 4) {
            g4_set_error(backend, "layer %u expert scale must be F32", layer->layer);
            return -1;
        }
        layer->has_scale = 1;
    } else if (scale && scale->t != J_NULL) {
        g4_set_error(backend, "layer %u expert_scale is malformed", layer->layer);
        return -1;
    }

    components = json_get(value, "components");
    if (!components || components->t != J_ARR || components->len != 3) {
        g4_set_error(backend, "layer %u must contain gate, up, and down components", layer->layer);
        return -1;
    }
    for (i = 0; i < 3; ++i) {
        if (g4_parse_component(backend, layer, components->kids[i]) != 0) return -1;
    }
    return g4_validate_layer(backend, layer);
}

static int g4_validate_unique_layers(coli_gemma4_backend *backend) {
    uint32_t i, j, expected;
    for (i = 0; i < backend->layer_count; ++i) {
        for (j = i + 1; j < backend->layer_count; ++j) {
            if (backend->layers[i].layer == backend->layers[j].layer) {
                g4_set_error(backend, "manifest contains duplicate layer %u",
                             backend->layers[i].layer);
                return -1;
            }
        }
    }
    for (expected = 0; expected < backend->layer_count; ++expected) {
        int found = 0;
        for (i = 0; i < backend->layer_count; ++i)
            if (backend->layers[i].layer == expected) found = 1;
        if (!found) {
            g4_set_error(backend, "manifest is missing routed layer %u", expected);
            return -1;
        }
    }
    return 0;
}

int coli_gemma4_open_packed(coli_gemma4_backend *backend, const char *packed_dir) {
    char manifest_path[COLI_GEMMA4_PATH_MAX];
    char *text = NULL;
    char *arena = NULL;
    jval *root = NULL;
    jval *layers;
    const char *format;
    const char *architecture;
    const char *source;
    uint32_t expert_count, expert_used_count;
    uint32_t i;

    if (!backend || !packed_dir || !*packed_dir) return -1;
    memset(backend, 0, sizeof(*backend));
    if (g4_copy_string(backend, backend->packed_dir, sizeof(backend->packed_dir),
                       packed_dir, "packed directory") != 0 ||
        g4_join_path(backend, manifest_path, sizeof(manifest_path),
                     packed_dir, "manifest.json") != 0) {
        return -1;
    }
    text = g4_read_text(backend, manifest_path);
    if (!text) return -1;
    root = json_parse(text, &arena);
    if (!root || root->t != J_OBJ) {
        g4_set_error(backend, "manifest root is not a JSON object");
        goto fail;
    }
    format = g4_string(root, "format");
    architecture = g4_string(root, "architecture");
    source = g4_string(root, "source");
    if (!format || strcmp(format, "g4lab-expert-manifest-v2") != 0) {
        g4_set_error(backend, "unsupported packed manifest format");
        goto fail;
    }
    if (!architecture || strcmp(architecture, "gemma4") != 0) {
        g4_set_error(backend, "packed manifest architecture is not gemma4");
        goto fail;
    }
    if (g4_number_u32(backend, root, "expert_count", &expert_count) != 0 ||
        g4_number_u32(backend, root, "expert_used_count", &expert_used_count) != 0 ||
        !expert_count || !expert_used_count || expert_used_count > expert_count) {
        if (!backend->last_error[0])
            g4_set_error(backend, "invalid expert count in manifest");
        goto fail;
    }
    if (g4_copy_string(backend, backend->source, sizeof(backend->source),
                       source, "source") != 0) goto fail;
    layers = json_get(root, "layers");
    if (!layers || layers->t != J_ARR || layers->len < 1 || layers->len > 4096) {
        g4_set_error(backend, "manifest layers array is missing or invalid");
        goto fail;
    }
    backend->layers = (coli_gemma4_layer *)calloc((size_t)layers->len,
                                                  sizeof(*backend->layers));
    if (!backend->layers) {
        g4_set_error(backend, "out of memory allocating Gemma layer descriptors");
        goto fail;
    }
    backend->layer_count = (uint32_t)layers->len;
    for (i = 0; i < backend->layer_count; ++i) {
        if (g4_parse_layer(backend, &backend->layers[i], layers->kids[i]) != 0)
            goto fail;
        if (backend->layers[i].expert_count != expert_count) {
            g4_set_error(backend, "layer %u expert count disagrees with manifest",
                         backend->layers[i].layer);
            goto fail;
        }
    }
    if (g4_validate_unique_layers(backend) != 0) goto fail;

    backend->config.n_layer = backend->layer_count;
    backend->config.n_embd = backend->layers[0].model_width;
    backend->config.n_expert = expert_count;
    backend->config.n_expert_used = expert_used_count;
    backend->config.n_expert_ff = backend->layers[0].expert_width;
    for (i = 1; i < backend->layer_count; ++i) {
        if (backend->layers[i].model_width != backend->config.n_embd ||
            backend->layers[i].expert_width != backend->config.n_expert_ff) {
            g4_set_error(backend, "Gemma expert dimensions vary across layers");
            goto fail;
        }
    }
    free(arena);
    g4_free_json(root);
    free(text);
    return 0;

fail:
    free(arena);
    g4_free_json(root);
    free(text);
    free(backend->layers);
    backend->layers = NULL;
    backend->layer_count = 0;
    return -1;
}

void coli_gemma4_close(coli_gemma4_backend *backend) {
    if (!backend) return;
    coli_gemma4_cuda_destroy(backend->cuda);
    g4_prefetch_destroy(backend);
    g4_cache_clear(backend);
    free(backend->layers);
    memset(backend, 0, sizeof(*backend));
}

int coli_gemma4_cache_configure(coli_gemma4_backend *backend, uint32_t slots) {
    coli_gemma4_cache_slot *cache = NULL;
    uint32_t *usage = NULL;
    size_t records = 0;
    if (!backend || !backend->layers) return -1;
#if SIZE_MAX <= UINT32_MAX
    if (slots > SIZE_MAX / sizeof(*cache)) {
        g4_set_error(backend, "expert cache slot count does not fit in memory");
        return -1;
    }
#endif
    if (slots) {
        if (backend->config.n_expert &&
            backend->layer_count > SIZE_MAX / backend->config.n_expert) {
            g4_set_error(backend, "expert usage-map dimensions overflow");
            return -1;
        }
        records = (size_t)backend->layer_count * backend->config.n_expert;
        cache = (coli_gemma4_cache_slot *)calloc((size_t)slots, sizeof(*cache));
        usage = (uint32_t *)calloc(records, sizeof(*usage));
        if (!cache || !usage) {
            free(cache); free(usage);
            g4_set_error(backend, "out of memory allocating expert cache metadata");
            return -1;
        }
    }
    g4_prefetch_destroy(backend);
    g4_cache_clear(backend);
    backend->cache_slots = cache;
    backend->cache_usage = usage;
    backend->cache_capacity = slots;
    return 0;
}

int coli_gemma4_cache_set_pinned(coli_gemma4_backend *backend, uint32_t slots) {
    uint32_t *heat = NULL, *last = NULL;
    size_t records;
    if (!backend || !backend->layers || slots >= backend->cache_capacity) {
        if (backend)
            g4_set_error(backend,
                         "pinned expert slots must be fewer than cache slots");
        return -1;
    }
    if (slots) {
        if (backend->config.n_expert &&
            backend->layer_count > SIZE_MAX / backend->config.n_expert) {
            g4_set_error(backend, "expert heat-map dimensions overflow");
            return -1;
        }
        records = (size_t)backend->layer_count * backend->config.n_expert;
        if (records > INT_MAX) {
            g4_set_error(backend, "expert heat map exceeds LFRU policy limits");
            return -1;
        }
        heat = (uint32_t *)calloc(records, sizeof(*heat));
        last = (uint32_t *)calloc(records, sizeof(*last));
        if (!heat || !last) {
            free(heat); free(last);
            g4_set_error(backend, "out of memory allocating expert heat map");
            return -1;
        }
    }
    free(backend->cache_heat);
    free(backend->cache_last);
    backend->cache_heat = heat;
    backend->cache_last = last;
    backend->cache_pinned_capacity = slots;
    backend->cache_access_clock = 0;
    backend->cache_pinned_hits = 0;
    backend->cache_promotions = 0;
    return 0;
}

void coli_gemma4_cache_get_stats(const coli_gemma4_backend *backend,
                                 coli_gemma4_cache_stats *stats) {
    if (!stats) return;
    memset(stats, 0, sizeof(*stats));
    if (!backend) return;
    stats->capacity = backend->cache_capacity;
    stats->resident = backend->cache_resident;
    stats->pinned_capacity = backend->cache_pinned_capacity;
    stats->hits = backend->cache_hits;
    stats->pinned_hits = backend->cache_pinned_hits;
    stats->misses = backend->cache_misses;
    stats->preloads = backend->cache_preloads;
    stats->prefetch_launches = backend->cache_prefetch_launches;
    stats->prefetched_records = backend->cache_prefetched_records;
    stats->lookahead_launches = backend->cache_lookahead_launches;
    stats->lookahead_records = backend->cache_lookahead_records;
    stats->lookahead_matches = backend->cache_lookahead_matches;
    stats->lookahead_selected = backend->cache_lookahead_selected;
    stats->evictions = backend->cache_evictions;
    stats->promotions = backend->cache_promotions;
    stats->bytes_loaded = backend->cache_bytes_loaded;
}

const char *coli_gemma4_last_error(const coli_gemma4_backend *backend) {
    return backend && backend->last_error[0] ? backend->last_error : "unknown Gemma backend error";
}

const coli_gemma4_layer *coli_gemma4_find_layer(
    const coli_gemma4_backend *backend, uint32_t layer) {
    uint32_t i;
    if (!backend) return NULL;
    for (i = 0; i < backend->layer_count; ++i)
        if (backend->layers[i].layer == layer) return &backend->layers[i];
    return NULL;
}

int coli_gemma4_has_packed_layer(
    const coli_gemma4_backend *backend, const coli_gemma4_layer *layer) {
    char path[COLI_GEMMA4_PATH_MAX];
    FILE *file;
    size_t directory_length;
    const char *separator;
    int written;
    if (!backend || !layer) return 0;
    directory_length = strlen(backend->packed_dir);
    separator = (directory_length &&
                 (backend->packed_dir[directory_length - 1] == '/' ||
                  backend->packed_dir[directory_length - 1] == '\\')) ? "" : "/";
    written = snprintf(path, sizeof(path), "%s%s%s", backend->packed_dir,
                       separator, layer->packed_file);
    if (written < 0 || (size_t)written >= sizeof(path)) return 0;
    file = fopen(path, "rb");
    if (!file) return 0;
    fclose(file);
    return 1;
}

float coli_gemma4_fp16_to_fp32(uint16_t value) {
    uint32_t sign = (uint32_t)(value & 0x8000U) << 16U;
    uint32_t exponent = (value >> 10U) & 0x1FU;
    uint32_t mantissa = value & 0x03FFU;
    uint32_t bits;
    float result;
    if (exponent == 0) {
        if (mantissa == 0) bits = sign;
        else {
            int unbiased = -14;
            while ((mantissa & 0x0400U) == 0) {
                mantissa <<= 1U;
                --unbiased;
            }
            mantissa &= 0x03FFU;
            bits = sign | ((uint32_t)(unbiased + 127) << 23U) | (mantissa << 13U);
        }
    } else if (exponent == 0x1FU) {
        bits = sign | 0x7F800000U | (mantissa << 13U);
    } else {
        bits = sign | ((exponent + 112U) << 23U) | (mantissa << 13U);
    }
    memcpy(&result, &bits, sizeof(result));
    return result;
}

float coli_gemma4_gelu_tanh(float value) {
    const float coefficient = 0.7978845608028654F;
    const float cubic = 0.044715F;
    return 0.5F * value *
           (1.0F + tanhf(coefficient * (value + cubic * value * value * value)));
}

int coli_gemma4_q4_0_matvec(const uint8_t *weights, size_t rows, size_t columns,
                            const float *input, float *output) {
    size_t blocks_per_row, row_bytes;
    long long row_signed;
    if (!weights || !input || !output || !rows || !columns || columns % 32 != 0)
        return -1;
    blocks_per_row = columns / 32;
    row_bytes = blocks_per_row * 18;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (row_signed = 0; row_signed < (long long)rows; ++row_signed) {
        size_t row = (size_t)row_signed;
        const uint8_t *row_data = weights + row * row_bytes;
        float sum = 0.0F;
        size_t block;
        for (block = 0; block < blocks_per_row; ++block) {
            const uint8_t *encoded = row_data + block * 18;
            uint16_t scale_bits = (uint16_t)encoded[0] |
                                  ((uint16_t)encoded[1] << 8);
            float scale = coli_gemma4_fp16_to_fp32(scale_bits);
            const uint8_t *quants = encoded + 2;
            const float *x = input + block * 32;
            float integer_dot = 0.0F;
            size_t index;
            for (index = 0; index < 16; ++index) {
                uint8_t packed = quants[index];
                int low = (int)(packed & 0x0FU) - 8;
                int high = (int)(packed >> 4U) - 8;
                integer_dot += (float)low * x[index];
                integer_dot += (float)high * x[index + 16];
            }
            sum += scale * integer_dot;
        }
        output[row] = sum;
    }
    return 0;
}

static int q6_k_value(const uint8_t *block, size_t half, size_t lane,
                      size_t quarter) {
    const uint8_t *ql = block + half * 64;
    const uint8_t *qh = block + 128 + half * 32;
    uint8_t low;
    uint8_t high;
    if (quarter == 0) {
        low = ql[lane] & 0x0FU;
        high = (qh[lane] >> 0U) & 3U;
    } else if (quarter == 1) {
        low = ql[lane + 32] & 0x0FU;
        high = (qh[lane] >> 2U) & 3U;
    } else if (quarter == 2) {
        low = ql[lane] >> 4U;
        high = (qh[lane] >> 4U) & 3U;
    } else {
        low = ql[lane + 32] >> 4U;
        high = (qh[lane] >> 6U) & 3U;
    }
    return (int)(low | (uint8_t)(high << 4U)) - 32;
}

static int q6_k_scale(const uint8_t *block, size_t index) {
    uint8_t encoded = block[192 + index];
    return encoded < 128U ? (int)encoded : (int)encoded - 256;
}

int coli_gemma4_q6_k_row(const uint8_t *weights, size_t rows, size_t columns,
                         size_t row, float *output) {
    size_t blocks_per_row, block_index;
    if (!weights || !output || !rows || !columns || row >= rows ||
        columns % 256 != 0) return -1;
    blocks_per_row = columns / 256;
    weights += row * blocks_per_row * 210;
    for (block_index = 0; block_index < blocks_per_row; ++block_index) {
        const uint8_t *block = weights + block_index * 210;
        uint16_t scale_bits = (uint16_t)block[208] |
                              ((uint16_t)block[209] << 8U);
        float super_scale = coli_gemma4_fp16_to_fp32(scale_bits);
        size_t half, lane, quarter;
        for (half = 0; half < 2; ++half) {
            for (lane = 0; lane < 32; ++lane) {
                for (quarter = 0; quarter < 4; ++quarter) {
                    size_t within = half * 128 + quarter * 32 + lane;
                    size_t scale_index = half * 8 + quarter * 2 + lane / 16;
                    output[block_index * 256 + within] =
                        super_scale * (float)q6_k_scale(block, scale_index) *
                        (float)q6_k_value(block, half, lane, quarter);
                }
            }
        }
    }
    return 0;
}

int coli_gemma4_q6_k_matvec(const uint8_t *weights, size_t rows, size_t columns,
                            const float *input, float *output) {
    size_t blocks_per_row, row_bytes;
    long long row_signed;
    if (!weights || !input || !output || !rows || !columns ||
        columns % 256 != 0) return -1;
    blocks_per_row = columns / 256;
    row_bytes = blocks_per_row * 210;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (row_signed = 0; row_signed < (long long)rows; ++row_signed) {
        const uint8_t *row_data = weights + (size_t)row_signed * row_bytes;
        float sum = 0.0F;
        size_t block_index;
        for (block_index = 0; block_index < blocks_per_row; ++block_index) {
            const uint8_t *block = row_data + block_index * 210;
            const float *x = input + block_index * 256;
            uint16_t scale_bits = (uint16_t)block[208] |
                                  ((uint16_t)block[209] << 8U);
            float super_scale = coli_gemma4_fp16_to_fp32(scale_bits);
            size_t half, lane, quarter;
            for (half = 0; half < 2; ++half) {
                for (lane = 0; lane < 32; ++lane) {
                    for (quarter = 0; quarter < 4; ++quarter) {
                        size_t within = half * 128 + quarter * 32 + lane;
                        size_t scale_index =
                            half * 8 + quarter * 2 + lane / 16;
                        sum += super_scale *
                               (float)q6_k_scale(block, scale_index) *
                               (float)q6_k_value(block, half, lane, quarter) *
                               x[within];
                    }
                }
            }
        }
        output[(size_t)row_signed] = sum;
    }
    return 0;
}

static int g4_run_payload(coli_gemma4_backend *backend,
                          const coli_gemma4_layer *layer, const uint8_t *payload,
                          const float *input, float scale, float *output) {
    float *gate, *up, *hidden;
    long long index_signed;
    const uint8_t *gate_weights = payload;
    const uint8_t *up_weights = gate_weights + layer->component_bytes[G4_ROLE_GATE];
    const uint8_t *down_weights = up_weights + layer->component_bytes[G4_ROLE_UP];
    if (backend->cuda) {
        if (coli_gemma4_cuda_run(
                backend->cuda, payload, (size_t)layer->payload_bytes,
                (size_t)layer->component_bytes[G4_ROLE_GATE],
                (size_t)layer->component_bytes[G4_ROLE_UP],
                layer->model_width, layer->expert_width, input, scale, output,
                backend->last_error, sizeof(backend->last_error)) != 0)
            return -1;
        return 0;
    }
    gate = (float *)malloc((size_t)layer->expert_width * sizeof(float));
    up = (float *)malloc((size_t)layer->expert_width * sizeof(float));
    hidden = (float *)malloc((size_t)layer->expert_width * sizeof(float));
    if (!gate || !up || !hidden) {
        free(gate); free(up); free(hidden);
        g4_set_error(backend, "out of memory running Gemma expert");
        return -1;
    }
    if (coli_gemma4_q4_0_matvec(gate_weights, layer->expert_width,
                                layer->model_width, input, gate) != 0 ||
        coli_gemma4_q4_0_matvec(up_weights, layer->expert_width,
                                layer->model_width, input, up) != 0) {
        g4_set_error(backend, "invalid Q4_0 gate/up matrix");
        free(gate); free(up); free(hidden);
        return -1;
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (index_signed = 0; index_signed < (long long)layer->expert_width; ++index_signed) {
        size_t index = (size_t)index_signed;
        hidden[index] = coli_gemma4_gelu_tanh(gate[index]) * up[index];
    }
    if (coli_gemma4_q4_0_matvec(down_weights, layer->model_width,
                                layer->expert_width, hidden, output) != 0) {
        g4_set_error(backend, "invalid Q4_0 down matrix");
        free(gate); free(up); free(hidden);
        return -1;
    }
    if (scale != 1.0F) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (index_signed = 0; index_signed < (long long)layer->model_width; ++index_signed)
            output[(size_t)index_signed] *= scale;
    }
    free(gate); free(up); free(hidden);
    return 0;
}

static int g4_read_header_and_payload(coli_gemma4_backend *backend,
                                      const coli_gemma4_layer *layer,
                                      uint32_t expert, uint8_t *payload) {
    char path[COLI_GEMMA4_PATH_MAX];
    uint8_t header[80];
    FILE *file;
    uint64_t header_bytes, payload_bytes, record_stride, offset, required;
    uint64_t component_sum;
    g4_off_t file_size;
    int i;
    if (g4_join_path(backend, path, sizeof(path), backend->packed_dir,
                     layer->packed_file) != 0) return -1;
    file = fopen(path, "rb");
    if (!file) {
        g4_set_error(backend, "cannot open packed layer %s: %s", path, strerror(errno));
        return -1;
    }
    if (fread(header, 1, sizeof(header), file) != sizeof(header) ||
        memcmp(header, "G4EXPK01", 8) != 0) {
        fclose(file);
        g4_set_error(backend, "packed layer header is missing or corrupt: %s", path);
        return -1;
    }
    header_bytes = g4_load_u64(header + 32);
    payload_bytes = g4_load_u64(header + 40);
    record_stride = g4_load_u64(header + 48);
    component_sum = g4_load_u64(header + 56) + g4_load_u64(header + 64) +
                    g4_load_u64(header + 72);
    if (g4_load_u32(header + 8) != 1 ||
        g4_load_u32(header + 12) != layer->layer ||
        g4_load_u32(header + 16) != layer->expert_count ||
        g4_load_u32(header + 20) != 2 ||
        g4_load_u32(header + 24) != 4096 ||
        header_bytes != 4096 ||
        payload_bytes != layer->payload_bytes ||
        record_stride != layer->record_stride ||
        component_sum != payload_bytes) {
        fclose(file);
        g4_set_error(backend, "packed layer header disagrees with manifest: %s", path);
        return -1;
    }
    for (i = 0; i < 3; ++i) {
        if (g4_load_u64(header + 56 + i * 8) != layer->component_bytes[i]) {
            fclose(file);
            g4_set_error(backend, "packed layer component sizes disagree with manifest");
            return -1;
        }
    }
    if (expert >= layer->expert_count ||
        (record_stride &&
         expert > (UINT64_MAX - header_bytes) / record_stride)) {
        fclose(file);
        g4_set_error(backend, "expert %u is outside layer %u", expert, layer->layer);
        return -1;
    }
    offset = header_bytes + (uint64_t)expert * record_stride;
    required = offset + payload_bytes;
    if (required < offset || required > (uint64_t)LLONG_MAX ||
        g4_seek(file, 0, SEEK_END) != 0 || (file_size = g4_tell(file)) < 0 ||
        required > (uint64_t)file_size ||
        g4_seek(file, (g4_off_t)offset, SEEK_SET) != 0 ||
        fread(payload, 1, (size_t)payload_bytes, file) != (size_t)payload_bytes) {
        fclose(file);
        g4_set_error(backend, "packed expert record is truncated: %s", path);
        return -1;
    }
    fclose(file);
    return 0;
}

int coli_gemma4_cuda_configure(coli_gemma4_backend *backend, int device) {
    void *context = NULL;
    if (!backend || !backend->layers || device < 0) return -1;
    if (coli_gemma4_cuda_create(
            &context, device, backend->last_error,
            sizeof(backend->last_error)) != 0) return -1;
    coli_gemma4_cuda_destroy(backend->cuda);
    backend->cuda = context;
    return 0;
}

static int g4_source_path(coli_gemma4_backend *backend, char *destination,
                          size_t capacity) {
    if (g4_is_absolute_path(backend->source))
        return g4_copy_string(backend, destination, capacity,
                              backend->source, "source");
    return g4_join_path(backend, destination, capacity,
                        backend->packed_dir, backend->source);
}

static int g4_read_source_payload(coli_gemma4_backend *backend,
                                  const coli_gemma4_layer *layer,
                                  uint32_t expert, uint8_t *payload) {
    char source_path[COLI_GEMMA4_PATH_MAX];
    FILE *file;
    uint64_t destination_offset = 0;
    int role;
    if (expert >= layer->expert_count) {
        g4_set_error(backend, "expert %u is outside layer %u", expert, layer->layer);
        return -1;
    }
    if (g4_source_path(backend, source_path, sizeof(source_path)) != 0) return -1;
    file = fopen(source_path, "rb");
    if (!file) {
        g4_set_error(backend, "cannot open source GGUF %s: %s",
                     source_path, strerror(errno));
        return -1;
    }
    for (role = 0; role < 3; ++role) {
        uint64_t source;
        uint64_t expert_offset;
        uint64_t bytes = layer->component_bytes[role];
        if (layer->source_expert_stride[role] != 0 &&
            expert > (UINT64_MAX - layer->source_offset[role]) /
                         layer->source_expert_stride[role]) {
            fclose(file);
            g4_set_error(backend, "GGUF expert source offset overflows");
            return -1;
        }
        expert_offset = layer->source_offset[role] +
                        (uint64_t)expert * layer->source_expert_stride[role];
        if (expert_offset > UINT64_MAX - layer->source_within_expert_offset[role]) {
            fclose(file);
            g4_set_error(backend, "GGUF expert component offset overflows");
            return -1;
        }
        source = expert_offset + layer->source_within_expert_offset[role];
        if (source > (uint64_t)LLONG_MAX || bytes > SIZE_MAX ||
            g4_seek(file, (g4_off_t)source, SEEK_SET) != 0 ||
            fread(payload + (size_t)destination_offset, 1, (size_t)bytes, file) !=
                (size_t)bytes) {
            fclose(file);
            g4_set_error(backend,
                         "cannot read layer %u expert %u component %d from %s",
                         layer->layer, expert, role, source_path);
            return -1;
        }
        destination_offset += bytes;
    }
    fclose(file);
    if (destination_offset != layer->payload_bytes) {
        g4_set_error(backend, "GGUF expert components do not fill the payload");
        return -1;
    }
    return 0;
}

static int g4_read_scale(coli_gemma4_backend *backend,
                         const coli_gemma4_layer *layer,
                         uint32_t expert, float *scale) {
    char source_path[COLI_GEMMA4_PATH_MAX];
    uint8_t bytes[4];
    uint64_t offset;
    uint32_t bits;
    FILE *file;
    if (!layer->has_scale) {
        *scale = 1.0F;
        return 0;
    }
    if (expert >= layer->expert_count ||
        expert > (UINT64_MAX - layer->scale_offset) / layer->scale_scalar_bytes) {
        g4_set_error(backend, "expert scale offset overflows");
        return -1;
    }
    if (g4_source_path(backend, source_path, sizeof(source_path)) != 0) return -1;
    offset = layer->scale_offset + (uint64_t)expert * layer->scale_scalar_bytes;
    if (offset > (uint64_t)LLONG_MAX) {
        g4_set_error(backend, "expert scale offset is too large");
        return -1;
    }
    file = fopen(source_path, "rb");
    if (!file || g4_seek(file, (g4_off_t)offset, SEEK_SET) != 0 ||
        fread(bytes, 1, sizeof(bytes), file) != sizeof(bytes)) {
        if (file) fclose(file);
        g4_set_error(backend, "cannot read expert scale from %s", source_path);
        return -1;
    }
    fclose(file);
    bits = g4_load_u32(bytes);
    memcpy(scale, &bits, sizeof(*scale));
    if (!isfinite(*scale)) {
        g4_set_error(backend, "expert scale is not finite");
        return -1;
    }
    return 0;
}

static coli_gemma4_cache_slot *g4_cache_find(coli_gemma4_backend *backend,
                                              uint32_t layer,
                                              uint32_t expert,
                                              uint32_t *slot_index) {
    uint32_t i;
    for (i = 0; i < backend->cache_capacity; ++i) {
        coli_gemma4_cache_slot *slot = backend->cache_slots + i;
        if (slot->valid && slot->layer == layer && slot->expert == expert) {
            if (slot_index) *slot_index = i;
            return slot;
        }
    }
    return NULL;
}

static size_t g4_cache_heat_index(const coli_gemma4_backend *backend,
                                  uint32_t layer, uint32_t expert) {
    return (size_t)layer * backend->config.n_expert + expert;
}

static void g4_cache_record_access(coli_gemma4_backend *backend,
                                   uint32_t layer, uint32_t expert) {
    size_t records, index;
    index = g4_cache_heat_index(backend, layer, expert);
    if (backend->cache_usage && backend->cache_usage[index] != UINT32_MAX)
        ++backend->cache_usage[index];
    if (!backend->cache_pinned_capacity) return;
    records = (size_t)backend->layer_count * backend->config.n_expert;
    if (backend->cache_access_clock == UINT32_MAX) {
        tier_decay(backend->cache_heat, (int)records);
        memset(backend->cache_last, 0, records * sizeof(*backend->cache_last));
        backend->cache_access_clock = 0;
    }
    ++backend->cache_access_clock;
    if (backend->cache_heat[index] != UINT32_MAX)
        ++backend->cache_heat[index];
    backend->cache_last[index] = backend->cache_access_clock;
}

static int g4_cache_pick_pin(const coli_gemma4_backend *backend,
                             uint32_t layer, uint32_t expert,
                             uint32_t *pin_index) {
    uint32_t cold = 0, i;
    uint64_t hot_score, cold_score;
    size_t hot_index, candidate_index;
    if (!backend->cache_pinned_capacity) return 0;
    for (i = 1; i < backend->cache_pinned_capacity; ++i) {
        size_t cold_index;
        if (!backend->cache_slots[i].valid) {
            cold = i;
            break;
        }
        candidate_index = g4_cache_heat_index(
            backend, backend->cache_slots[i].layer,
            backend->cache_slots[i].expert);
        cold_index = g4_cache_heat_index(
            backend, backend->cache_slots[cold].layer,
            backend->cache_slots[cold].expert);
        if (tier_lfru_score(backend->cache_heat[candidate_index],
                            backend->cache_last[candidate_index],
                            backend->cache_access_clock) <
            tier_lfru_score(backend->cache_heat[cold_index],
                            backend->cache_last[cold_index],
                            backend->cache_access_clock)) cold = i;
    }
    if (!backend->cache_slots[cold].valid) {
        *pin_index = cold;
        return 1;
    }
    hot_index = g4_cache_heat_index(backend, layer, expert);
    candidate_index = g4_cache_heat_index(
        backend, backend->cache_slots[cold].layer,
        backend->cache_slots[cold].expert);
    hot_score = tier_lfru_score(backend->cache_heat[hot_index],
                                backend->cache_last[hot_index],
                                backend->cache_access_clock);
    cold_score = tier_lfru_score(backend->cache_heat[candidate_index],
                                 backend->cache_last[candidate_index],
                                 backend->cache_access_clock);
    if (hot_score <= cold_score + (cold_score >> 2) + (4U << 8)) return 0;
    *pin_index = cold;
    return 1;
}

static coli_gemma4_cache_slot *g4_cache_promote_hit(
    coli_gemma4_backend *backend, coli_gemma4_cache_slot *slot,
    uint32_t slot_index) {
    uint32_t pin_index;
    coli_gemma4_cache_slot temporary;
    if (!backend->cache_pinned_capacity) return slot;
    if (slot_index < backend->cache_pinned_capacity) {
        ++backend->cache_pinned_hits;
        return slot;
    }
    if (!g4_cache_pick_pin(backend, slot->layer, slot->expert,
                           &pin_index)) return slot;
    temporary = backend->cache_slots[pin_index];
    backend->cache_slots[pin_index] = *slot;
    *slot = temporary;
    ++backend->cache_promotions;
    return backend->cache_slots + pin_index;
}

static int g4_load_record(coli_gemma4_backend *backend,
                          const coli_gemma4_layer *descriptor,
                          uint32_t expert, uint8_t *payload, float *scale) {
    int read_status;
    if (coli_gemma4_has_packed_layer(backend, descriptor))
        read_status = g4_read_header_and_payload(backend, descriptor,
                                                 expert, payload);
    else
        read_status = g4_read_source_payload(backend, descriptor,
                                             expert, payload);
    if (read_status != 0) return -1;
    return g4_read_scale(backend, descriptor, expert, scale);
}

static coli_gemma4_cache_slot *g4_cache_load(
    coli_gemma4_backend *backend, const coli_gemma4_layer *descriptor,
    uint32_t expert, int access_mode) {
    coli_gemma4_cache_slot *slot;
    uint32_t i, slot_index = 0, pin_index = 0;
    int evicting = 0, promoting = 0;
    slot = g4_cache_find(backend, descriptor->layer, expert, &slot_index);
    if (slot) {
        slot->last_used = ++backend->cache_clock;
        if (access_mode > 0) ++backend->cache_hits;
        return access_mode > 0 ? g4_cache_promote_hit(
                               backend, slot, slot_index) : slot;
    }
    if (access_mode < 0) ++backend->cache_preloads;
    else ++backend->cache_misses;
    if (g4_cache_pick_pin(backend, descriptor->layer, expert, &pin_index)) {
        slot = backend->cache_slots + pin_index;
        promoting = slot->valid;
    } else {
        slot = backend->cache_slots + backend->cache_pinned_capacity;
        for (i = backend->cache_pinned_capacity; i < backend->cache_capacity; ++i) {
            coli_gemma4_cache_slot *candidate = backend->cache_slots + i;
            if (!candidate->valid) {
                slot = candidate;
                break;
            }
            if (candidate->last_used < slot->last_used) slot = candidate;
        }
    }
    evicting = slot->valid;
    slot->valid = 0;
    if (descriptor->payload_bytes > SIZE_MAX) {
        g4_set_error(backend, "expert payload does not fit in memory");
        return NULL;
    }
    if (slot->payload_capacity < (size_t)descriptor->payload_bytes) {
        uint8_t *payload = (uint8_t *)realloc(
            slot->payload, (size_t)descriptor->payload_bytes);
        if (!payload) {
            g4_set_error(backend, "out of memory allocating cached expert record");
            return NULL;
        }
        slot->payload = payload;
        slot->payload_capacity = (size_t)descriptor->payload_bytes;
    }
    if (g4_load_record(backend, descriptor, expert, slot->payload,
                       &slot->scale) != 0) return NULL;
    slot->layer = descriptor->layer;
    slot->expert = expert;
    slot->last_used = ++backend->cache_clock;
    slot->valid = 1;
    if (promoting) ++backend->cache_promotions;
    if (evicting) ++backend->cache_evictions;
    else ++backend->cache_resident;
    backend->cache_bytes_loaded += descriptor->payload_bytes;
    if (descriptor->has_scale)
        backend->cache_bytes_loaded += descriptor->scale_scalar_bytes;
    return slot;
}

#ifdef _WIN32
static unsigned __stdcall g4_prefetch_worker(void *opaque)
#else
static void *g4_prefetch_worker(void *opaque)
#endif
{
    coli_gemma4_prefetch_state *state =
        (coli_gemma4_prefetch_state *)opaque;
    uint32_t i;
    state->result = 0;
    for (i = 0; i < state->count; ++i) {
        if (!g4_cache_load(state->backend, state->descriptor,
                           state->expert_ids[i], state->lookahead ? 0 : 1)) {
            state->result = -1;
            break;
        }
    }
#ifdef _WIN32
    return 0;
#else
    return NULL;
#endif
}

static int g4_prefetch_wait(coli_gemma4_backend *backend) {
    coli_gemma4_prefetch_state *state = backend ? backend->prefetch : NULL;
    if (!state || !state->active) return 0;
#ifdef _WIN32
    if (WaitForSingleObject(state->thread, INFINITE) != WAIT_OBJECT_0) {
        CloseHandle(state->thread);
        state->thread = NULL;
        state->active = 0;
        g4_set_error(backend, "cannot join Gemma expert prefetch worker");
        return -1;
    }
    CloseHandle(state->thread);
    state->thread = NULL;
#else
    if (pthread_join(state->thread, NULL) != 0) {
        state->active = 0;
        g4_set_error(backend, "cannot join Gemma expert prefetch worker");
        return -1;
    }
#endif
    state->active = 0;
    return state->result;
}

static void g4_prefetch_destroy(coli_gemma4_backend *backend) {
    coli_gemma4_prefetch_state *state;
    if (!backend) return;
    state = backend->prefetch;
    backend->lookahead_enabled = 0;
    if (!state) return;
    (void)g4_prefetch_wait(backend);
    free(state->expert_ids);
    free(state);
    backend->prefetch = NULL;
}

int coli_gemma4_prefetch_configure(coli_gemma4_backend *backend, int enabled) {
    coli_gemma4_prefetch_state *state;
    if (!backend || !backend->layers) return -1;
    g4_prefetch_destroy(backend);
    if (!enabled) return 0;
    if (!backend->cache_capacity) {
        g4_set_error(backend, "asynchronous expert prefetch requires a cache");
        return -1;
    }
    state = (coli_gemma4_prefetch_state *)calloc(1, sizeof(*state));
    if (!state) {
        g4_set_error(backend, "out of memory allocating expert prefetch state");
        return -1;
    }
    state->expert_ids = (uint32_t *)malloc(
        (size_t)backend->config.n_expert_used * sizeof(*state->expert_ids));
    if (!state->expert_ids) {
        free(state);
        g4_set_error(backend, "out of memory allocating expert prefetch request");
        return -1;
    }
    state->capacity = backend->config.n_expert_used;
    state->backend = backend;
    backend->prefetch = state;
    return 0;
}

int coli_gemma4_lookahead_configure(coli_gemma4_backend *backend, int enabled) {
    if (!backend) return -1;
    if (enabled && (!backend->prefetch || !backend->cache_capacity)) {
        g4_set_error(backend,
                     "next-layer lookahead requires asynchronous prefetch and a cache");
        return -1;
    }
    backend->lookahead_enabled = enabled != 0;
    return 0;
}

static int g4_prefetch_start(coli_gemma4_backend *backend,
                             const coli_gemma4_layer *descriptor,
                             uint32_t count, int lookahead) {
    coli_gemma4_prefetch_state *state = backend->prefetch;
    if (!state) return 0;
    state->descriptor = descriptor;
    state->count = count;
    state->result = 0;
    state->lookahead = lookahead;
    if (!count) return 0;
#ifdef _WIN32
    state->thread = (HANDLE)_beginthreadex(
        NULL, 0, g4_prefetch_worker, state, 0, NULL);
    if (!state->thread) {
        g4_set_error(backend, "cannot create Gemma expert prefetch worker");
        return -1;
    }
#else
    if (pthread_create(&state->thread, NULL, g4_prefetch_worker, state) != 0) {
        g4_set_error(backend, "cannot create Gemma expert prefetch worker");
        return -1;
    }
#endif
    state->active = 1;
    ++backend->cache_prefetch_launches;
    backend->cache_prefetched_records += count;
    if (lookahead) {
        ++backend->cache_lookahead_launches;
        backend->cache_lookahead_records += count;
    }
    return 0;
}

static void g4_usage_add_total(uint64_t *total, uint32_t value) {
    if (UINT64_MAX - *total < value) *total = UINT64_MAX;
    else *total += value;
}

int coli_gemma4_usage_load(coli_gemma4_backend *backend, const char *path,
                           uint64_t *selections) {
    FILE *file;
    size_t records;
    uint64_t total = 0;
    unsigned layer, expert, count;
    int fields;
    uint32_t pin;
    uint8_t *chosen = NULL;
    if (selections) *selections = 0;
    if (!backend || !backend->cache_usage || !path || !*path) {
        if (backend) g4_set_error(backend, "usage profile requires an active expert cache");
        return -1;
    }
    records = (size_t)backend->layer_count * backend->config.n_expert;
    file = fopen(path, "r");
    if (!file) {
        if (errno == ENOENT) return 0;
        g4_set_error(backend, "cannot open expert usage profile %s: %s",
                     path, strerror(errno));
        return -1;
    }
    memset(backend->cache_usage, 0, records * sizeof(*backend->cache_usage));
    while ((fields = fscanf(file, "%u %u %u", &layer, &expert, &count)) == 3) {
        if (layer < backend->layer_count && expert < backend->config.n_expert) {
            size_t index = g4_cache_heat_index(backend, layer, expert);
            uint32_t old = backend->cache_usage[index];
            backend->cache_usage[index] = UINT32_MAX - old < count ?
                                          UINT32_MAX : old + count;
        }
    }
    if (fields != EOF || ferror(file)) {
        fclose(file);
        g4_set_error(backend, "expert usage profile is malformed: %s", path);
        return -1;
    }
    fclose(file);
    for (size_t index = 0; index < records; ++index)
        g4_usage_add_total(&total, backend->cache_usage[index]);
    if (backend->cache_pinned_capacity && total) {
        chosen = (uint8_t *)calloc(records, 1);
        if (!chosen) {
            g4_set_error(backend, "out of memory ranking expert usage profile");
            return -1;
        }
        for (pin = 0; pin < backend->cache_pinned_capacity; ++pin) {
            size_t best = SIZE_MAX, index;
            uint32_t best_count = 0;
            for (index = 0; index < records; ++index) {
                if (!chosen[index] && backend->cache_usage[index] > best_count) {
                    best = index;
                    best_count = backend->cache_usage[index];
                }
            }
            if (best == SIZE_MAX || !best_count) break;
            chosen[best] = 1;
            layer = (unsigned)(best / backend->config.n_expert);
            expert = (unsigned)(best % backend->config.n_expert);
            {
                const coli_gemma4_layer *descriptor =
                    coli_gemma4_find_layer(backend, layer);
                if (!descriptor || !g4_cache_load(
                        backend, descriptor, expert, -1)) {
                    free(chosen);
                    return -1;
                }
            }
        }
        free(chosen);
    }
    if (selections) *selections = total;
    return 0;
}

int coli_gemma4_usage_save(coli_gemma4_backend *backend, const char *path,
                           uint64_t *selections) {
    char temporary[COLI_GEMMA4_PATH_MAX + 8];
    FILE *file;
    uint64_t total = 0;
    uint32_t layer, expert;
    int written;
    if (selections) *selections = 0;
    if (!backend || !backend->cache_usage || !path || !*path) {
        if (backend) g4_set_error(backend, "usage profile requires an active expert cache");
        return -1;
    }
    written = snprintf(temporary, sizeof(temporary), "%s.tmp", path);
    if (written < 0 || (size_t)written >= sizeof(temporary)) {
        g4_set_error(backend, "expert usage profile path is too long");
        return -1;
    }
    file = fopen(temporary, "w");
    if (!file) {
        g4_set_error(backend, "cannot create expert usage profile %s: %s",
                     temporary, strerror(errno));
        return -1;
    }
    for (layer = 0; layer < backend->layer_count; ++layer) {
        for (expert = 0; expert < backend->config.n_expert; ++expert) {
            uint32_t count = backend->cache_usage[
                g4_cache_heat_index(backend, layer, expert)];
            if (count && fprintf(file, "%u %u %u\n", layer, expert, count) < 0) {
                fclose(file); remove(temporary);
                g4_set_error(backend, "cannot write expert usage profile %s", temporary);
                return -1;
            }
            g4_usage_add_total(&total, count);
        }
    }
    if (fflush(file) != 0 || fclose(file) != 0) {
        remove(temporary);
        g4_set_error(backend, "cannot flush expert usage profile %s", temporary);
        return -1;
    }
#ifdef _WIN32
    if (!MoveFileExA(temporary, path,
                     MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        DWORD code = GetLastError();
        remove(temporary);
        g4_set_error(backend, "cannot replace expert usage profile %s (Windows error %lu)",
                     path, (unsigned long)code);
        return -1;
    }
#else
    if (rename(temporary, path) != 0) {
        remove(temporary);
        g4_set_error(backend, "cannot replace expert usage profile %s: %s",
                     path, strerror(errno));
        return -1;
    }
#endif
    if (selections) *selections = total;
    return 0;
}

static int g4_validate_expert_ids(coli_gemma4_backend *backend,
                                  const coli_gemma4_layer *descriptor,
                                  uint32_t layer,
                                  const uint32_t *expert_ids,
                                  uint32_t count) {
    uint32_t i;
    if (!backend || !descriptor || (count && !expert_ids)) {
        if (backend) g4_set_error(backend, "invalid Gemma prepare_layer request");
        return -1;
    }
    for (i = 0; i < count; ++i) {
        if (expert_ids[i] >= descriptor->expert_count) {
            g4_set_error(backend, "expert %u is outside layer %u", expert_ids[i], layer);
            return -1;
        }
    }
    return 0;
}

static int g4_prepare_layer(void *opaque, uint32_t layer,
                            const uint32_t *expert_ids, uint32_t count) {
    coli_gemma4_backend *backend = (coli_gemma4_backend *)opaque;
    const coli_gemma4_layer *descriptor = coli_gemma4_find_layer(backend, layer);
    uint32_t i;
    if (g4_validate_expert_ids(backend, descriptor, layer,
                               expert_ids, count) != 0) return -1;
    if (backend->prefetch && backend->cache_capacity) {
        uint32_t missing = 0;
        if (backend->prefetch->lookahead &&
            backend->prefetch->descriptor == descriptor) {
            uint32_t selected, predicted;
            backend->cache_lookahead_selected += count;
            for (selected = 0; selected < count; ++selected)
                for (predicted = 0; predicted < backend->prefetch->count;
                     ++predicted)
                    if (expert_ids[selected] ==
                        backend->prefetch->expert_ids[predicted]) {
                        ++backend->cache_lookahead_matches;
                        break;
                    }
        }
        if (g4_prefetch_wait(backend) != 0) return -1;
        if (count > backend->prefetch->capacity) {
            uint32_t *ids = (uint32_t *)realloc(
                backend->prefetch->expert_ids,
                (size_t)count * sizeof(*backend->prefetch->expert_ids));
            if (!ids) {
                g4_set_error(backend,
                             "out of memory growing expert prefetch request");
                return -1;
            }
            backend->prefetch->expert_ids = ids;
            backend->prefetch->capacity = count;
        }
        for (i = 0; i < count; ++i) {
            g4_cache_record_access(backend, layer, expert_ids[i]);
            if (g4_cache_find(backend, layer, expert_ids[i], NULL)) {
                if (!g4_cache_load(backend, descriptor, expert_ids[i], 1))
                    return -1;
            } else {
                backend->prefetch->expert_ids[missing++] = expert_ids[i];
            }
        }
        return g4_prefetch_start(backend, descriptor, missing, 0);
    }
    for (i = 0; i < count; ++i) {
        g4_cache_record_access(backend, layer, expert_ids[i]);
        if (backend->cache_capacity &&
            !g4_cache_load(backend, descriptor, expert_ids[i], 1)) return -1;
    }
    return 0;
}

static int g4_run_experts(void *opaque, uint32_t layer,
                          const uint32_t *expert_ids, const float *weights,
                          uint32_t count, const float *input, float *output) {
    coli_gemma4_backend *backend = (coli_gemma4_backend *)opaque;
    const coli_gemma4_layer *descriptor = coli_gemma4_find_layer(backend, layer);
    uint8_t *payload = NULL;
    float *expert_output = NULL;
    uint32_t selected, i;
    if (!backend || !descriptor || !count || !expert_ids || !weights ||
        !input || !output) {
        if (backend) g4_set_error(backend, "invalid Gemma run_experts request");
        return -1;
    }
    if (g4_validate_expert_ids(backend, descriptor, layer,
                               expert_ids, count) != 0) return -1;
    if (g4_prefetch_wait(backend) != 0) return -1;
    if (descriptor->payload_bytes > SIZE_MAX) {
        g4_set_error(backend, "expert payload does not fit in memory");
        return -1;
    }
    if (!backend->cache_capacity)
        payload = (uint8_t *)malloc((size_t)descriptor->payload_bytes);
    expert_output = (float *)malloc((size_t)descriptor->model_width * sizeof(float));
    if ((!backend->cache_capacity && !payload) || !expert_output) {
        free(payload); free(expert_output);
        g4_set_error(backend, "out of memory allocating expert buffers");
        return -1;
    }
    memset(output, 0, (size_t)descriptor->model_width * sizeof(float));
    for (selected = 0; selected < count; ++selected) {
        float scale;
        const uint8_t *record = payload;
        if (backend->cache_capacity) {
            coli_gemma4_cache_slot *slot = g4_cache_find(
                backend, descriptor->layer, expert_ids[selected], NULL);
            if (!slot) slot = g4_cache_load(
                backend, descriptor, expert_ids[selected], 0);
            if (!slot) {
                free(payload); free(expert_output);
                return -1;
            }
            record = slot->payload;
            scale = slot->scale;
        } else if (g4_load_record(backend, descriptor, expert_ids[selected],
                                  payload, &scale) != 0) {
            free(payload); free(expert_output);
            return -1;
        }
        if (!isfinite(weights[selected]) ||
            g4_run_payload(backend, descriptor, record, input, scale,
                           expert_output) != 0) {
            free(payload); free(expert_output);
            return -1;
        }
        for (i = 0; i < descriptor->model_width; ++i)
            output[i] += weights[selected] * expert_output[i];
    }
    free(payload);
    free(expert_output);
    return 0;
}

static void g4_release_layer(void *opaque, uint32_t layer) {
    coli_gemma4_backend *backend = (coli_gemma4_backend *)opaque;
    (void)layer;
    if (backend) (void)g4_prefetch_wait(backend);
}

static int g4_prefetch_layer(void *opaque, uint32_t layer,
                             const uint32_t *expert_ids, uint32_t count) {
    coli_gemma4_backend *backend = (coli_gemma4_backend *)opaque;
    const coli_gemma4_layer *descriptor = coli_gemma4_find_layer(backend, layer);
    uint32_t i, missing = 0;
    if (!backend || !backend->lookahead_enabled) return 0;
    if (g4_validate_expert_ids(backend, descriptor, layer,
                               expert_ids, count) != 0 ||
        g4_prefetch_wait(backend) != 0) return -1;
    if (count > backend->prefetch->capacity) {
        uint32_t *ids = (uint32_t *)realloc(
            backend->prefetch->expert_ids,
            (size_t)count * sizeof(*backend->prefetch->expert_ids));
        if (!ids) {
            g4_set_error(backend, "out of memory growing lookahead request");
            return -1;
        }
        backend->prefetch->expert_ids = ids;
        backend->prefetch->capacity = count;
    }
    for (i = 0; i < count; ++i)
        if (!g4_cache_find(backend, layer, expert_ids[i], NULL))
            backend->prefetch->expert_ids[missing++] = expert_ids[i];
    return g4_prefetch_start(backend, descriptor, missing, 1);
}

coli_expert_backend coli_gemma4_expert_backend(coli_gemma4_backend *backend) {
    coli_expert_backend result;
    result.ctx = backend;
    result.prepare_layer = g4_prepare_layer;
    result.run_experts = g4_run_experts;
    result.release_layer = g4_release_layer;
    result.prefetch_layer = backend && backend->lookahead_enabled ?
                            g4_prefetch_layer : NULL;
    return result;
}

uint64_t coli_gemma4_checksum_f32(const float *values, size_t count) {
    uint64_t hash = UINT64_C(1469598103934665603);
    size_t i;
    if (!values) return 0;
    for (i = 0; i < count; ++i) {
        uint32_t bits;
        unsigned shift;
        memcpy(&bits, values + i, sizeof(bits));
        for (shift = 0; shift < 32; shift += 8) {
            hash ^= (uint8_t)((bits >> shift) & 0xFFU);
            hash *= UINT64_C(1099511628211);
        }
    }
    return hash;
}
