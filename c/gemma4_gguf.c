#define _CRT_SECURE_NO_WARNINGS
#define _FILE_OFFSET_BITS 64

#include "gemma4_gguf.h"

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define g4_seek _fseeki64
#define g4_tell _ftelli64
typedef __int64 g4_off_t;
#else
#include <sys/types.h>
#define g4_seek fseeko
#define g4_tell ftello
typedef off_t g4_off_t;
#endif

enum {
    GGUF_META_UINT8 = 0,
    GGUF_META_INT8 = 1,
    GGUF_META_UINT16 = 2,
    GGUF_META_INT16 = 3,
    GGUF_META_UINT32 = 4,
    GGUF_META_INT32 = 5,
    GGUF_META_FLOAT32 = 6,
    GGUF_META_BOOL = 7,
    GGUF_META_STRING = 8,
    GGUF_META_ARRAY = 9,
    GGUF_META_UINT64 = 10,
    GGUF_META_INT64 = 11,
    GGUF_META_FLOAT64 = 12
};

static void gguf_error(coli_gemma4_gguf *gguf, const char *format, ...) {
    va_list arguments;
    if (!gguf) return;
    va_start(arguments, format);
    vsnprintf(gguf->last_error, sizeof(gguf->last_error), format, arguments);
    va_end(arguments);
}

static uint32_t load_u32(const uint8_t bytes[4]) {
    return (uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8) |
           ((uint32_t)bytes[2] << 16) | ((uint32_t)bytes[3] << 24);
}

static uint64_t load_u64(const uint8_t bytes[8]) {
    return (uint64_t)load_u32(bytes) | ((uint64_t)load_u32(bytes + 4) << 32);
}

static int read_bytes(coli_gemma4_gguf *gguf, FILE *file,
                      void *destination, size_t bytes) {
    if (bytes && fread(destination, 1, bytes, file) != bytes) {
        gguf_error(gguf, "unexpected end of GGUF file");
        return -1;
    }
    return 0;
}

static int read_u8(coli_gemma4_gguf *gguf, FILE *file, uint8_t *value) {
    return read_bytes(gguf, file, value, 1);
}

static int read_u16(coli_gemma4_gguf *gguf, FILE *file, uint16_t *value) {
    uint8_t bytes[2];
    if (read_bytes(gguf, file, bytes, sizeof(bytes)) != 0) return -1;
    *value = (uint16_t)((uint16_t)bytes[0] | ((uint16_t)bytes[1] << 8));
    return 0;
}

static int read_u32(coli_gemma4_gguf *gguf, FILE *file, uint32_t *value) {
    uint8_t bytes[4];
    if (read_bytes(gguf, file, bytes, sizeof(bytes)) != 0) return -1;
    *value = load_u32(bytes);
    return 0;
}

static int read_u64(coli_gemma4_gguf *gguf, FILE *file, uint64_t *value) {
    uint8_t bytes[8];
    if (read_bytes(gguf, file, bytes, sizeof(bytes)) != 0) return -1;
    *value = load_u64(bytes);
    return 0;
}

static int read_f32(coli_gemma4_gguf *gguf, FILE *file, float *value) {
    uint32_t bits;
    if (read_u32(gguf, file, &bits) != 0) return -1;
    memcpy(value, &bits, sizeof(*value));
    return 0;
}

static int read_f64(coli_gemma4_gguf *gguf, FILE *file, double *value) {
    uint64_t bits;
    if (read_u64(gguf, file, &bits) != 0) return -1;
    memcpy(value, &bits, sizeof(*value));
    return 0;
}

static int skip_bytes(coli_gemma4_gguf *gguf, FILE *file, uint64_t bytes) {
    if (bytes > (uint64_t)LLONG_MAX ||
        g4_seek(file, (g4_off_t)bytes, SEEK_CUR) != 0) {
        gguf_error(gguf, "GGUF metadata skip is outside the file");
        return -1;
    }
    return 0;
}

static int read_string(coli_gemma4_gguf *gguf, FILE *file, char **result) {
    uint64_t length;
    char *value;
    if (read_u64(gguf, file, &length) != 0) return -1;
    if (length > SIZE_MAX - 1 || length > UINT64_C(268435456)) {
        gguf_error(gguf, "GGUF string is unreasonably large");
        return -1;
    }
    value = (char *)malloc((size_t)length + 1);
    if (!value) {
        gguf_error(gguf, "out of memory reading GGUF string");
        return -1;
    }
    if (read_bytes(gguf, file, value, (size_t)length) != 0) {
        free(value);
        return -1;
    }
    value[length] = '\0';
    *result = value;
    return 0;
}

static int copy_string(coli_gemma4_gguf *gguf, char *destination,
                       size_t capacity, const char *source, const char *label) {
    size_t length = strlen(source);
    if (length + 1 > capacity) {
        gguf_error(gguf, "%s is too long", label);
        return -1;
    }
    memcpy(destination, source, length + 1);
    return 0;
}

static int key_is(const char *key, const char *expected) {
    return strcmp(key, expected) == 0;
}

static int store_u64(coli_gemma4_gguf *gguf, const char *key, uint64_t value) {
    if (value > UINT32_MAX) {
        gguf_error(gguf, "metadata %s exceeds uint32", key);
        return -1;
    }
    if (key_is(key, "general.alignment")) gguf->alignment = value;
    else if (key_is(key, "general.sampling.top_k")) gguf->sampling_top_k = (uint32_t)value;
    else if (key_is(key, "gemma4.block_count")) gguf->config.n_layer = (uint32_t)value;
    else if (key_is(key, "gemma4.embedding_length")) gguf->config.n_embd = (uint32_t)value;
    else if (key_is(key, "gemma4.expert_count")) gguf->config.n_expert = (uint32_t)value;
    else if (key_is(key, "gemma4.expert_used_count")) gguf->config.n_expert_used = (uint32_t)value;
    else if (key_is(key, "gemma4.expert_feed_forward_length")) gguf->config.n_expert_ff = (uint32_t)value;
    else if (key_is(key, "gemma4.attention.sliding_window")) gguf->config.sliding_window = (uint32_t)value;
    else if (key_is(key, "gemma4.attention.head_count")) gguf->attention_heads = (uint32_t)value;
    else if (key_is(key, "gemma4.attention.key_length")) gguf->key_length = (uint32_t)value;
    else if (key_is(key, "gemma4.attention.key_length_swa")) gguf->key_length_swa = (uint32_t)value;
    else if (key_is(key, "gemma4.attention.value_length")) gguf->value_length = (uint32_t)value;
    else if (key_is(key, "gemma4.attention.value_length_swa")) gguf->value_length_swa = (uint32_t)value;
    else if (key_is(key, "gemma4.rope.dimension_count")) gguf->rope_dimensions = (uint32_t)value;
    else if (key_is(key, "gemma4.rope.dimension_count_swa")) gguf->rope_dimensions_swa = (uint32_t)value;
    else if (key_is(key, "tokenizer.ggml.bos_token_id")) gguf->tokenizer_bos_id = (uint32_t)value;
    else if (key_is(key, "tokenizer.ggml.eos_token_id")) gguf->tokenizer_eos_id = (uint32_t)value;
    else if (key_is(key, "tokenizer.ggml.unknown_token_id")) gguf->tokenizer_unknown_id = (uint32_t)value;
    else if (key_is(key, "tokenizer.ggml.padding_token_id")) gguf->tokenizer_padding_id = (uint32_t)value;
    else if (key_is(key, "tokenizer.ggml.add_bos_token")) gguf->tokenizer_add_bos = value != 0;
    else if (key_is(key, "clip.has_vision_encoder")) gguf->vision_has_encoder = value != 0;
    else if (key_is(key, "clip.vision.projection_dim")) gguf->vision_projection_dim = (uint32_t)value;
    else if (key_is(key, "clip.vision.image_size")) gguf->vision_image_size = (uint32_t)value;
    else if (key_is(key, "clip.vision.patch_size")) gguf->vision_patch_size = (uint32_t)value;
    else if (key_is(key, "clip.vision.embedding_length")) gguf->vision_embedding_length = (uint32_t)value;
    else if (key_is(key, "clip.vision.feed_forward_length")) gguf->vision_feed_forward_length = (uint32_t)value;
    else if (key_is(key, "clip.vision.block_count")) gguf->vision_block_count = (uint32_t)value;
    else if (key_is(key, "clip.vision.attention.head_count")) gguf->vision_head_count = (uint32_t)value;
    return 0;
}

static void store_f64(coli_gemma4_gguf *gguf, const char *key, double value) {
    if (key_is(key, "gemma4.attention.layer_norm_rms_epsilon")) gguf->rms_epsilon = (float)value;
    else if (key_is(key, "gemma4.rope.freq_base")) gguf->rope_freq_base = (float)value;
    else if (key_is(key, "gemma4.rope.freq_base_swa")) gguf->rope_freq_base_swa = (float)value;
    else if (key_is(key, "gemma4.final_logit_softcapping")) gguf->final_logit_softcap = (float)value;
    else if (key_is(key, "general.sampling.temp")) gguf->sampling_temperature = (float)value;
    else if (key_is(key, "general.sampling.top_p")) gguf->sampling_top_p = (float)value;
    else if (key_is(key, "clip.vision.attention.layer_norm_epsilon")) gguf->vision_epsilon = (float)value;
}

static size_t metadata_scalar_bytes(uint32_t type) {
    switch (type) {
        case GGUF_META_UINT8:
        case GGUF_META_INT8:
        case GGUF_META_BOOL: return 1;
        case GGUF_META_UINT16:
        case GGUF_META_INT16: return 2;
        case GGUF_META_UINT32:
        case GGUF_META_INT32:
        case GGUF_META_FLOAT32: return 4;
        case GGUF_META_UINT64:
        case GGUF_META_INT64:
        case GGUF_META_FLOAT64: return 8;
        default: return 0;
    }
}

static int read_metadata_value(coli_gemma4_gguf *gguf, FILE *file,
                               const char *key, uint32_t type, unsigned depth) {
    uint8_t u8;
    uint16_t u16;
    uint32_t u32;
    uint64_t u64;
    float f32;
    double f64;
    char *text = NULL;
    if (depth > 16) {
        gguf_error(gguf, "GGUF metadata nesting is too deep");
        return -1;
    }
    switch (type) {
        case GGUF_META_UINT8:
        case GGUF_META_INT8:
        case GGUF_META_BOOL:
            if (read_u8(gguf, file, &u8) != 0) return -1;
            return store_u64(gguf, key, u8);
        case GGUF_META_UINT16:
        case GGUF_META_INT16:
            if (read_u16(gguf, file, &u16) != 0) return -1;
            return store_u64(gguf, key, u16);
        case GGUF_META_UINT32:
        case GGUF_META_INT32:
            if (read_u32(gguf, file, &u32) != 0) return -1;
            return store_u64(gguf, key, u32);
        case GGUF_META_FLOAT32:
            if (read_f32(gguf, file, &f32) != 0) return -1;
            store_f64(gguf, key, f32);
            return 0;
        case GGUF_META_UINT64:
        case GGUF_META_INT64:
            if (read_u64(gguf, file, &u64) != 0) return -1;
            return store_u64(gguf, key, u64);
        case GGUF_META_FLOAT64:
            if (read_f64(gguf, file, &f64) != 0) return -1;
            store_f64(gguf, key, f64);
            return 0;
        case GGUF_META_STRING:
            if (read_string(gguf, file, &text) != 0) return -1;
            if ((key_is(key, "general.architecture") &&
                 copy_string(gguf, gguf->architecture, sizeof(gguf->architecture),
                             text, "GGUF architecture") != 0) ||
                (key_is(key, "tokenizer.ggml.model") &&
                 copy_string(gguf, gguf->tokenizer_model,
                             sizeof(gguf->tokenizer_model), text,
                             "GGUF tokenizer model") != 0) ||
                ((key_is(key, "clip.projector_type") ||
                  key_is(key, "clip.vision.projector_type")) &&
                 copy_string(gguf, gguf->projector_type,
                             sizeof(gguf->projector_type), text,
                             "projector type") != 0)) {
                free(text);
                return -1;
            }
            free(text);
            return 0;
        case GGUF_META_ARRAY: {
            uint32_t element_type;
            uint64_t count, index;
            size_t scalar_bytes;
            int capture_kv = key_is(key, "gemma4.attention.head_count_kv");
            int capture_swa = key_is(key, "gemma4.attention.sliding_window_pattern");
            int capture_tokens = key_is(key, "tokenizer.ggml.tokens");
            int capture_merges = key_is(key, "tokenizer.ggml.merges");
            int capture_types = key_is(key, "tokenizer.ggml.token_type");
            int capture_mean = key_is(key, "clip.vision.image_mean");
            int capture_std = key_is(key, "clip.vision.image_std");
            if (read_u32(gguf, file, &element_type) != 0 ||
                read_u64(gguf, file, &count) != 0) return -1;
            if (element_type > GGUF_META_FLOAT64 || count > UINT64_C(1000000000)) {
                gguf_error(gguf, "invalid GGUF metadata array");
                return -1;
            }
            if (capture_kv || capture_swa) {
                if (count > 256 ||
                    (capture_kv && element_type != GGUF_META_INT32) ||
                    (capture_swa && element_type != GGUF_META_BOOL)) {
                    gguf_error(gguf, "invalid Gemma attention metadata array %s", key);
                    return -1;
                }
                for (index = 0; index < count; ++index) {
                    if (capture_kv) {
                        if (read_u32(gguf, file, &u32) != 0) return -1;
                        gguf->head_count_kv[index] = u32;
                    } else {
                        if (read_u8(gguf, file, &u8) != 0 || u8 > 1) return -1;
                        gguf->sliding_window_pattern[index] = u8;
                    }
                }
                return 0;
            }
            if (capture_tokens || capture_merges) {
                char ***strings = capture_tokens ? &gguf->tokenizer_tokens :
                                                   &gguf->tokenizer_merges;
                uint64_t *stored_count = capture_tokens ?
                    &gguf->tokenizer_token_count : &gguf->tokenizer_merge_count;
                if (element_type != GGUF_META_STRING || count > UINT64_C(10000000) ||
                    count > SIZE_MAX / sizeof(**strings)) {
                    gguf_error(gguf, "invalid tokenizer string array %s", key);
                    return -1;
                }
                *strings = (char **)calloc((size_t)count, sizeof(**strings));
                if (count && !*strings) {
                    gguf_error(gguf, "out of memory reading %s", key);
                    return -1;
                }
                *stored_count = count;
                for (index = 0; index < count; ++index)
                    if (read_string(gguf, file, &(*strings)[index]) != 0) return -1;
                return 0;
            }
            if (capture_types) {
                if (element_type != GGUF_META_INT32 || count > UINT64_C(10000000) ||
                    count > SIZE_MAX / sizeof(*gguf->tokenizer_token_types)) {
                    gguf_error(gguf, "invalid tokenizer token-type array");
                    return -1;
                }
                gguf->tokenizer_token_types = (uint32_t *)calloc(
                    (size_t)count, sizeof(*gguf->tokenizer_token_types));
                if (count && !gguf->tokenizer_token_types) {
                    gguf_error(gguf, "out of memory reading tokenizer token types");
                    return -1;
                }
                gguf->tokenizer_token_type_count = count;
                for (index = 0; index < count; ++index)
                    if (read_u32(gguf, file,
                                 &gguf->tokenizer_token_types[index]) != 0) return -1;
                return 0;
            }
            if (capture_mean || capture_std) {
                float *values = capture_mean ? gguf->vision_image_mean :
                                               gguf->vision_image_std;
                if (element_type != GGUF_META_FLOAT32 || count != 3) {
                    gguf_error(gguf, "invalid vision normalization array %s", key);
                    return -1;
                }
                for (index = 0; index < count; ++index)
                    if (read_f32(gguf, file, &values[index]) != 0) return -1;
                return 0;
            }
            scalar_bytes = metadata_scalar_bytes(element_type);
            if (scalar_bytes) {
                if (count > UINT64_MAX / scalar_bytes)
                    return (gguf_error(gguf, "GGUF metadata array overflows"), -1);
                return skip_bytes(gguf, file, count * scalar_bytes);
            }
            for (index = 0; index < count; ++index) {
                if (read_metadata_value(gguf, file, "", element_type, depth + 1) != 0)
                    return -1;
            }
            return 0;
        }
        default:
            gguf_error(gguf, "unknown GGUF metadata type %u", type);
            return -1;
    }
}

static int tensor_nbytes(coli_gemma4_gguf *gguf, coli_gemma4_tensor *tensor) {
    uint64_t elements = 1;
    uint64_t block_elements, block_bytes;
    uint32_t dimension;
    switch (tensor->type) {
        case COLI_GGML_TYPE_F32: block_elements = 1; block_bytes = 4; break;
        case COLI_GGML_TYPE_Q4_0: block_elements = 32; block_bytes = 18; break;
        case COLI_GGML_TYPE_Q6_K: block_elements = 256; block_bytes = 210; break;
        case COLI_GGML_TYPE_BF16: block_elements = 1; block_bytes = 2; break;
        default:
            gguf_error(gguf, "unsupported tensor type %u for %s",
                       tensor->type, tensor->name);
            return -1;
    }
    for (dimension = 0; dimension < tensor->n_dims; ++dimension) {
        if (!tensor->dims[dimension] || elements > UINT64_MAX / tensor->dims[dimension]) {
            gguf_error(gguf, "invalid dimensions for tensor %s", tensor->name);
            return -1;
        }
        elements *= tensor->dims[dimension];
    }
    if (elements % block_elements ||
        elements / block_elements > UINT64_MAX / block_bytes) {
        gguf_error(gguf, "tensor %s does not align to its quantization block", tensor->name);
        return -1;
    }
    tensor->nbytes = elements / block_elements * block_bytes;
    return 0;
}

static int validate_config(coli_gemma4_gguf *gguf) {
    uint32_t layer;
    if (strcmp(gguf->architecture, "clip") == 0) {
        uint32_t channel;
        if (!gguf->vision_has_encoder ||
            strcmp(gguf->projector_type, "gemma4v") != 0 ||
            !gguf->vision_projection_dim || !gguf->vision_image_size ||
            !gguf->vision_patch_size || !gguf->vision_embedding_length ||
            !gguf->vision_feed_forward_length || !gguf->vision_block_count ||
            !gguf->vision_head_count || !isfinite(gguf->vision_epsilon) ||
            gguf->vision_epsilon <= 0.0F) {
            gguf_error(gguf,
                       "missing Gemma 4 vision configuration: encoder=%u projector=%s projection=%u image=%u patch=%u width=%u ff=%u blocks=%u heads=%u epsilon=%.9g",
                       gguf->vision_has_encoder, gguf->projector_type,
                       gguf->vision_projection_dim, gguf->vision_image_size,
                       gguf->vision_patch_size, gguf->vision_embedding_length,
                       gguf->vision_feed_forward_length,
                       gguf->vision_block_count, gguf->vision_head_count,
                       gguf->vision_epsilon);
            return -1;
        }
        for (channel = 0; channel < 3; ++channel)
            if (!isfinite(gguf->vision_image_mean[channel]) ||
                !isfinite(gguf->vision_image_std[channel]) ||
                gguf->vision_image_std[channel] <= 0.0F) {
                gguf_error(gguf, "invalid Gemma 4 vision normalization metadata");
                return -1;
            }
        return 0;
    }
    if (strcmp(gguf->architecture, "gemma4") != 0 ||
        !gguf->config.n_layer || gguf->config.n_layer > 256 ||
        !gguf->config.n_embd || !gguf->config.n_expert ||
        !gguf->config.n_expert_used ||
        gguf->config.n_expert_used > gguf->config.n_expert ||
        !gguf->config.n_expert_ff || !gguf->attention_heads ||
        !isfinite(gguf->rms_epsilon) || gguf->rms_epsilon <= 0.0F) {
        gguf_error(gguf, "GGUF is missing required Gemma 4 configuration");
        return -1;
    }
    for (layer = 0; layer < gguf->config.n_layer; ++layer) {
        if (!gguf->head_count_kv[layer]) {
            gguf_error(gguf, "Gemma layer %u has no KV-head count", layer);
            return -1;
        }
    }
    return 0;
}

int coli_gemma4_gguf_open(coli_gemma4_gguf *gguf, const char *path) {
    FILE *file = NULL;
    char resolved[sizeof(gguf->path)];
    uint8_t magic[4];
    uint64_t metadata_count, index;
    g4_off_t position, size;
    int result = -1;
    if (!gguf || !path) return -1;
    memset(gguf, 0, sizeof(*gguf));
    gguf->alignment = 32;
    gguf->sampling_temperature = 1.0F;
    gguf->sampling_top_p = 1.0F;
    gguf->tokenizer_bos_id = UINT32_MAX;
    gguf->tokenizer_eos_id = UINT32_MAX;
    gguf->tokenizer_unknown_id = UINT32_MAX;
    gguf->tokenizer_padding_id = UINT32_MAX;
#ifdef _WIN32
    if (!_fullpath(resolved, path, sizeof(resolved))) {
#else
    if (!realpath(path, resolved)) {
#endif
        gguf_error(gguf, "cannot resolve GGUF path %s: %s", path, strerror(errno));
        return -1;
    }
    if (copy_string(gguf, gguf->path, sizeof(gguf->path), resolved, "GGUF path") != 0)
        return -1;
    file = fopen(gguf->path, "rb");
    if (!file) {
        gguf_error(gguf, "cannot open GGUF %s: %s", path, strerror(errno));
        return -1;
    }
    if (g4_seek(file, 0, SEEK_END) != 0 || (size = g4_tell(file)) < 0 ||
        g4_seek(file, 0, SEEK_SET) != 0) {
        gguf_error(gguf, "cannot determine GGUF size");
        goto cleanup;
    }
    gguf->file_size = (uint64_t)size;
    if (read_bytes(gguf, file, magic, sizeof(magic)) != 0 ||
        memcmp(magic, "GGUF", 4) != 0 ||
        read_u32(gguf, file, &gguf->version) != 0 ||
        read_u64(gguf, file, &gguf->tensor_count) != 0 ||
        read_u64(gguf, file, &metadata_count) != 0) goto cleanup;
    if (gguf->version < 2 || gguf->version > 3 ||
        gguf->tensor_count > UINT64_C(10000000) ||
        metadata_count > UINT64_C(10000000) ||
        gguf->tensor_count > SIZE_MAX / sizeof(*gguf->tensors)) {
        gguf_error(gguf, "unsupported or unreasonable GGUF header");
        goto cleanup;
    }
    for (index = 0; index < metadata_count; ++index) {
        char *key = NULL;
        uint32_t type;
        if (read_string(gguf, file, &key) != 0 ||
            read_u32(gguf, file, &type) != 0 ||
            read_metadata_value(gguf, file, key ? key : "", type, 0) != 0) {
            free(key);
            goto cleanup;
        }
        free(key);
    }
    gguf->tensors = (coli_gemma4_tensor *)calloc(
        (size_t)gguf->tensor_count, sizeof(*gguf->tensors));
    if (!gguf->tensors) {
        gguf_error(gguf, "out of memory allocating GGUF tensor index");
        goto cleanup;
    }
    for (index = 0; index < gguf->tensor_count; ++index) {
        coli_gemma4_tensor *tensor = &gguf->tensors[index];
        uint32_t dimension;
        if (read_string(gguf, file, &tensor->name) != 0 ||
            read_u32(gguf, file, &tensor->n_dims) != 0 ||
            !tensor->n_dims || tensor->n_dims > COLI_GEMMA4_GGUF_MAX_DIMS) {
            gguf_error(gguf, "invalid GGUF tensor directory entry");
            goto cleanup;
        }
        for (dimension = 0; dimension < tensor->n_dims; ++dimension)
            if (read_u64(gguf, file, &tensor->dims[dimension]) != 0) goto cleanup;
        if (read_u32(gguf, file, &tensor->type) != 0 ||
            read_u64(gguf, file, &tensor->offset) != 0 ||
            tensor_nbytes(gguf, tensor) != 0) goto cleanup;
    }
    position = g4_tell(file);
    if (position < 0 || !gguf->alignment || gguf->alignment % 8 ||
        (uint64_t)position > UINT64_MAX - (gguf->alignment - 1)) {
        gguf_error(gguf, "invalid GGUF alignment or directory size");
        goto cleanup;
    }
    gguf->data_offset = ((uint64_t)position + gguf->alignment - 1) /
                        gguf->alignment * gguf->alignment;
    for (index = 0; index < gguf->tensor_count; ++index) {
        coli_gemma4_tensor *tensor = &gguf->tensors[index];
        if (tensor->offset > UINT64_MAX - gguf->data_offset) {
            gguf_error(gguf, "tensor offset overflows for %s", tensor->name);
            goto cleanup;
        }
        tensor->offset += gguf->data_offset;
        if (tensor->offset > gguf->file_size ||
            tensor->nbytes > gguf->file_size - tensor->offset) {
            gguf_error(gguf, "tensor %s is outside the GGUF", tensor->name);
            goto cleanup;
        }
    }
    if (validate_config(gguf) != 0) goto cleanup;
    if (strcmp(gguf->architecture, "gemma4") == 0) {
        const coli_gemma4_tensor *embedding = coli_gemma4_gguf_find(gguf, "token_embd.weight");
        if (!embedding || embedding->n_dims != 2 ||
            embedding->dims[0] != gguf->config.n_embd ||
            embedding->dims[1] > UINT32_MAX) {
            gguf_error(gguf, "invalid or missing token embedding tensor");
            goto cleanup;
        }
        gguf->config.n_vocab = (uint32_t)embedding->dims[1];
    }
    result = 0;

cleanup:
    fclose(file);
    if (result != 0) {
        char saved[COLI_GEMMA4_GGUF_ERROR_MAX];
        memcpy(saved, gguf->last_error, sizeof(saved));
        coli_gemma4_gguf_close(gguf);
        memcpy(gguf->last_error, saved, sizeof(saved));
    }
    return result;
}

void coli_gemma4_gguf_close(coli_gemma4_gguf *gguf) {
    uint64_t index;
    if (!gguf) return;
    for (index = 0; index < gguf->tensor_count; ++index)
        free(gguf->tensors ? gguf->tensors[index].name : NULL);
    for (index = 0; index < gguf->tokenizer_token_count; ++index)
        free(gguf->tokenizer_tokens ? gguf->tokenizer_tokens[index] : NULL);
    for (index = 0; index < gguf->tokenizer_merge_count; ++index)
        free(gguf->tokenizer_merges ? gguf->tokenizer_merges[index] : NULL);
    free(gguf->tensors);
    free(gguf->tokenizer_tokens);
    free(gguf->tokenizer_merges);
    free(gguf->tokenizer_token_types);
    memset(gguf, 0, sizeof(*gguf));
}

const char *coli_gemma4_gguf_last_error(const coli_gemma4_gguf *gguf) {
    return gguf ? gguf->last_error : "invalid GGUF handle";
}

const coli_gemma4_tensor *coli_gemma4_gguf_find(
    const coli_gemma4_gguf *gguf, const char *name) {
    uint64_t index;
    if (!gguf || !name) return NULL;
    for (index = 0; index < gguf->tensor_count; ++index)
        if (strcmp(gguf->tensors[index].name, name) == 0)
            return &gguf->tensors[index];
    return NULL;
}

int coli_gemma4_gguf_read(const coli_gemma4_gguf *gguf,
                          const coli_gemma4_tensor *tensor,
                          void *destination, size_t bytes) {
    if (!tensor || bytes != tensor->nbytes) return -1;
    return coli_gemma4_gguf_read_slice(
        gguf, tensor, 0, destination, bytes);
}

int coli_gemma4_gguf_read_slice(const coli_gemma4_gguf *gguf,
                                const coli_gemma4_tensor *tensor,
                                uint64_t byte_offset, void *destination,
                                size_t bytes) {
    FILE *file;
    uint64_t offset;
    if (!gguf || !tensor || !destination ||
        byte_offset > tensor->nbytes || bytes > tensor->nbytes - byte_offset ||
        tensor->offset > UINT64_MAX - byte_offset) return -1;
    offset = tensor->offset + byte_offset;
    if (offset > (uint64_t)LLONG_MAX) return -1;
    file = fopen(gguf->path, "rb");
    if (!file) return -1;
    if (g4_seek(file, (g4_off_t)offset, SEEK_SET) != 0 ||
        fread(destination, 1, bytes, file) != bytes) {
        fclose(file);
        return -1;
    }
    return fclose(file) == 0 ? 0 : -1;
}

const char *coli_gemma4_ggml_type_name(uint32_t type) {
    switch (type) {
        case COLI_GGML_TYPE_F32: return "F32";
        case COLI_GGML_TYPE_Q4_0: return "Q4_0";
        case COLI_GGML_TYPE_Q6_K: return "Q6_K";
        case COLI_GGML_TYPE_BF16: return "BF16";
        default: return "UNKNOWN";
    }
}
