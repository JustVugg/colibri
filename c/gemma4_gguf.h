#ifndef COLI_GEMMA4_GGUF_H
#define COLI_GEMMA4_GGUF_H

#include <stddef.h>
#include <stdint.h>

#include "model.h"

#define COLI_GEMMA4_GGUF_MAX_DIMS 8
#define COLI_GEMMA4_GGUF_ERROR_MAX 512

enum {
    COLI_GGML_TYPE_F32 = 0,
    COLI_GGML_TYPE_Q4_0 = 2,
    COLI_GGML_TYPE_Q6_K = 14,
    COLI_GGML_TYPE_BF16 = 30
};

typedef struct {
    char *name;
    uint32_t type;
    uint32_t n_dims;
    uint64_t dims[COLI_GEMMA4_GGUF_MAX_DIMS];
    uint64_t offset;
    uint64_t nbytes;
} coli_gemma4_tensor;

typedef struct {
    coli_model_config config;
    uint32_t version;
    uint64_t alignment;
    uint64_t data_offset;
    uint64_t file_size;
    uint32_t attention_heads;
    uint32_t key_length;
    uint32_t key_length_swa;
    uint32_t value_length;
    uint32_t value_length_swa;
    uint32_t rope_dimensions;
    uint32_t rope_dimensions_swa;
    uint32_t head_count_kv[256];
    uint8_t sliding_window_pattern[256];
    float rms_epsilon;
    float rope_freq_base;
    float rope_freq_base_swa;
    float final_logit_softcap;
    float sampling_temperature;
    float sampling_top_p;
    uint32_t sampling_top_k;
    uint32_t vision_projection_dim;
    uint32_t vision_image_size;
    uint32_t vision_patch_size;
    uint32_t vision_embedding_length;
    uint32_t vision_feed_forward_length;
    uint32_t vision_block_count;
    uint32_t vision_head_count;
    float vision_epsilon;
    float vision_image_mean[3];
    float vision_image_std[3];
    uint8_t vision_has_encoder;
    char **tokenizer_tokens;
    char **tokenizer_merges;
    uint32_t *tokenizer_token_types;
    uint64_t tokenizer_token_count;
    uint64_t tokenizer_merge_count;
    uint64_t tokenizer_token_type_count;
    uint32_t tokenizer_bos_id;
    uint32_t tokenizer_eos_id;
    uint32_t tokenizer_unknown_id;
    uint32_t tokenizer_padding_id;
    uint8_t tokenizer_add_bos;
    coli_gemma4_tensor *tensors;
    uint64_t tensor_count;
    char path[2048];
    char architecture[32];
    char tokenizer_model[32];
    char projector_type[32];
    char last_error[COLI_GEMMA4_GGUF_ERROR_MAX];
} coli_gemma4_gguf;

int coli_gemma4_gguf_open(coli_gemma4_gguf *gguf, const char *path);
void coli_gemma4_gguf_close(coli_gemma4_gguf *gguf);
const char *coli_gemma4_gguf_last_error(const coli_gemma4_gguf *gguf);
const coli_gemma4_tensor *coli_gemma4_gguf_find(
    const coli_gemma4_gguf *gguf, const char *name);
int coli_gemma4_gguf_read(const coli_gemma4_gguf *gguf,
                          const coli_gemma4_tensor *tensor,
                          void *destination, size_t bytes);
int coli_gemma4_gguf_read_slice(const coli_gemma4_gguf *gguf,
                                const coli_gemma4_tensor *tensor,
                                uint64_t byte_offset, void *destination,
                                size_t bytes);
const char *coli_gemma4_ggml_type_name(uint32_t type);

#endif
