#define _CRT_SECURE_NO_WARNINGS

#include "../gemma4_model.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FIXTURE "gemma4-router-test.gguf"

static void write_u8(FILE *file, uint8_t value) { fwrite(&value, 1, 1, file); }
static void write_u32(FILE *file, uint32_t value) {
    uint8_t b[4] = {(uint8_t)value, (uint8_t)(value >> 8),
                    (uint8_t)(value >> 16), (uint8_t)(value >> 24)};
    fwrite(b, 1, sizeof(b), file);
}
static void write_u64(FILE *file, uint64_t value) {
    write_u32(file, (uint32_t)value);
    write_u32(file, (uint32_t)(value >> 32));
}
static void write_f32(FILE *file, float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    write_u32(file, bits);
}
static void write_string(FILE *file, const char *value) {
    size_t length = strlen(value);
    write_u64(file, length);
    fwrite(value, 1, length, file);
}
static void write_meta_u32(FILE *file, const char *key, uint32_t value) {
    write_string(file, key); write_u32(file, 4); write_u32(file, value);
}
static void write_meta_f32(FILE *file, const char *key, float value) {
    write_string(file, key); write_u32(file, 6); write_f32(file, value);
}
static void write_tensor(FILE *file, const char *name, uint32_t dims,
                         uint64_t d0, uint64_t d1, uint64_t offset) {
    write_string(file, name); write_u32(file, dims); write_u64(file, d0);
    if (dims == 2) write_u64(file, d1);
    write_u32(file, COLI_GGML_TYPE_F32); write_u64(file, offset);
}

static int make_fixture(void) {
    FILE *file = fopen(FIXTURE, "wb");
    long position;
    uint32_t i;
    const float projection[16] = {
        1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1
    };
    if (!file) return -1;
    fwrite("GGUF", 1, 4, file); write_u32(file, 3); write_u64(file, 4); write_u64(file, 13);
    write_string(file, "general.architecture"); write_u32(file, 8); write_string(file, "gemma4");
    write_meta_u32(file, "general.alignment", 32);
    write_meta_u32(file, "gemma4.block_count", 1);
    write_meta_u32(file, "gemma4.embedding_length", 4);
    write_meta_u32(file, "gemma4.expert_count", 4);
    write_meta_u32(file, "gemma4.expert_used_count", 2);
    write_meta_u32(file, "gemma4.expert_feed_forward_length", 32);
    write_meta_u32(file, "gemma4.attention.sliding_window", 2);
    write_meta_u32(file, "gemma4.attention.head_count", 1);
    write_meta_f32(file, "gemma4.attention.layer_norm_rms_epsilon", 1.0e-6F);
    write_string(file, "gemma4.attention.head_count_kv"); write_u32(file, 9);
    write_u32(file, 5); write_u64(file, 1); write_u32(file, 1);
    write_string(file, "gemma4.attention.sliding_window_pattern"); write_u32(file, 9);
    write_u32(file, 7); write_u64(file, 1); write_u8(file, 1);
    write_meta_u32(file, "gemma4.attention.key_length_swa", 4);
    write_tensor(file, "token_embd.weight", 2, 4, 8, 0);
    write_tensor(file, "blk.0.ffn_gate_inp.scale", 1, 4, 0, 128);
    write_tensor(file, "blk.0.ffn_gate_inp.weight", 2, 4, 4, 144);
    write_tensor(file, "blk.0.ffn_down_exps.scale", 1, 4, 0, 208);
    position = ftell(file);
    if (position < 0) { fclose(file); return -1; }
    while ((position++ % 32) != 0) write_u8(file, 0);
    for (i = 0; i < 32; ++i) write_f32(file, 0.0F);
    for (i = 0; i < 4; ++i) write_f32(file, 1.0F);
    for (i = 0; i < 16; ++i) write_f32(file, projection[i]);
    write_f32(file, 0.5F); write_f32(file, 2.0F);
    write_f32(file, 1.0F); write_f32(file, 1.0F);
    return fclose(file) == 0 ? 0 : -1;
}

static int close_enough(float a, float b) { return fabsf(a - b) < 1.0e-6F; }

int main(void) {
    coli_gemma4_gguf gguf;
    coli_gemma4_router router;
    float hidden[4] = {4,3,2,1}, probabilities[4], weights[2], effective[2];
    float norm_weight[4] = {1,2,3,4}, norm_output[4];
    uint32_t ids[2];
    float norm, score0, score1, expected0;
    int status = 1;
    remove(FIXTURE);
    if (make_fixture() != 0) { fprintf(stderr, "fixture write failed\n"); return 1; }
    if (coli_gemma4_gguf_open(&gguf, FIXTURE) != 0) {
        fprintf(stderr, "GGUF open failed: %s\n", coli_gemma4_gguf_last_error(&gguf));
        goto cleanup;
    }
    if (gguf.tensor_count != 4 || gguf.config.n_layer != 1 ||
        gguf.config.n_embd != 4 || gguf.config.n_vocab != 8 ||
        gguf.head_count_kv[0] != 1 || !gguf.sliding_window_pattern[0]) {
        fprintf(stderr, "GGUF metadata mismatch\n"); goto close_gguf;
    }
    if (coli_gemma4_router_open(&router, &gguf, 0) != 0) {
        fprintf(stderr, "router open failed: %s\n", coli_gemma4_router_last_error(&router));
        goto close_gguf;
    }
    if (coli_gemma4_router_route(&router, hidden, probabilities, ids,
                                 weights, effective) != 0) {
        fprintf(stderr, "router evaluation failed\n"); goto close_router;
    }
    norm = 1.0F / sqrtf((16.0F + 9.0F + 4.0F + 1.0F) / 4.0F + 1.0e-6F);
    score0 = 4.0F * norm * 0.5F;
    score1 = 3.0F * norm * 0.5F;
    expected0 = expf(score0) / (expf(score0) + expf(score1));
    if (ids[0] != 0 || ids[1] != 1 || !close_enough(weights[0], expected0) ||
        !close_enough(weights[1], 1.0F - expected0) ||
        !close_enough(effective[0], weights[0] * 0.5F) ||
        !close_enough(effective[1], weights[1] * 2.0F)) {
        fprintf(stderr, "router top-k or weights mismatch\n"); goto close_router;
    }
    if (coli_gemma4_rmsnorm(hidden, norm_weight, 4, 1.0e-6F,
                            norm_output) != 0 ||
        !close_enough(norm_output[0], 4.0F * norm) ||
        !close_enough(norm_output[1], 6.0F * norm) ||
        !close_enough(norm_output[2], 6.0F * norm) ||
        !close_enough(norm_output[3], 4.0F * norm)) {
        fprintf(stderr, "RMSNorm mismatch\n"); goto close_router;
    }
    puts("Gemma GGUF index and exact router semantics passed");
    status = 0;
close_router:
    coli_gemma4_router_close(&router);
close_gguf:
    coli_gemma4_gguf_close(&gguf);
cleanup:
    remove(FIXTURE);
    return status;
}
