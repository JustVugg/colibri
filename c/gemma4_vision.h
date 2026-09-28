#ifndef COLI_GEMMA4_VISION_H
#define COLI_GEMMA4_VISION_H

#include <stddef.h>
#include <stdint.h>

#include "gemma4_gguf.h"

#define COLI_GEMMA4_VISION_ERROR_MAX 512

typedef struct {
    coli_gemma4_gguf gguf;
    uint32_t merge_size;
    uint32_t minimum_tokens;
    uint32_t maximum_tokens;
    uint32_t trace_layer;
    const char *trace_directory;
    char last_error[COLI_GEMMA4_VISION_ERROR_MAX];
} coli_gemma4_vision;

typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t patch_columns;
    uint32_t patch_rows;
    uint32_t token_columns;
    uint32_t token_rows;
    float *pixels;
    size_t value_count;
} coli_gemma4_vision_image;

int coli_gemma4_vision_open(coli_gemma4_vision *vision, const char *path);
void coli_gemma4_vision_close(coli_gemma4_vision *vision);
const char *coli_gemma4_vision_last_error(const coli_gemma4_vision *vision);
int coli_gemma4_vision_target_size(const coli_gemma4_vision *vision,
                                   uint32_t source_width,
                                   uint32_t source_height,
                                   uint32_t *target_width,
                                   uint32_t *target_height,
                                   uint32_t *token_count);
int coli_gemma4_vision_prepare_rgb(const coli_gemma4_vision *vision,
                                   const uint8_t *rgb,
                                   uint32_t source_width,
                                   uint32_t source_height,
                                   coli_gemma4_vision_image *image);
int coli_gemma4_vision_patch_embeddings(
    const coli_gemma4_vision *vision,
    const coli_gemma4_vision_image *image,
    float **embeddings, uint32_t *patch_count);
int coli_gemma4_vision_transform(coli_gemma4_vision *vision,
                                 float *embeddings,
                                 uint32_t patch_count,
                                 uint32_t patch_columns,
                                 uint32_t layer_count);
int coli_gemma4_vision_transform_range(coli_gemma4_vision *vision,
                                       float *embeddings,
                                       uint32_t patch_count,
                                       uint32_t patch_columns,
                                       uint32_t first_layer,
                                       uint32_t layer_count);
void coli_gemma4_vision_trace_layer(coli_gemma4_vision *vision,
                                    uint32_t layer,
                                    const char *directory);
int coli_gemma4_vision_encode(coli_gemma4_vision *vision,
                              const coli_gemma4_vision_image *image,
                              float **embeddings, uint32_t *token_count);
int coli_gemma4_vision_load_ppm(const char *path, uint8_t **rgb,
                                uint32_t *width, uint32_t *height,
                                char *error, size_t error_capacity);
int coli_gemma4_vision_load_image(const char *path, uint8_t **rgb,
                                  uint32_t *width, uint32_t *height,
                                  char *error, size_t error_capacity);
void coli_gemma4_vision_image_close(coli_gemma4_vision_image *image);

#endif
