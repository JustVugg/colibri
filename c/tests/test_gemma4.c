#define _CRT_SECURE_NO_WARNINGS

#include "../gemma4_backend.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <direct.h>
#define make_dir(path) _mkdir(path)
#define remove_dir(path) _rmdir(path)
#else
#include <sys/stat.h>
#include <unistd.h>
#define make_dir(path) mkdir(path, 0700)
#define remove_dir(path) rmdir(path)
#endif

#define TEST_DIR "gemma4-test-data"
#define MANIFEST_FILE TEST_DIR "/manifest.json"
#define PACKED_NAME "gemma4-test-layer.g4ex"
#define PACKED_FILE TEST_DIR "/" PACKED_NAME
#define SOURCE_NAME "gemma4-test-source.bin"
#define SOURCE_FILE TEST_DIR "/" SOURCE_NAME
#define USAGE_FILE TEST_DIR "/.gemma4_usage"

static void put_u32(uint8_t *bytes, size_t offset, uint32_t value) {
    bytes[offset + 0] = (uint8_t)value;
    bytes[offset + 1] = (uint8_t)(value >> 8);
    bytes[offset + 2] = (uint8_t)(value >> 16);
    bytes[offset + 3] = (uint8_t)(value >> 24);
}

static void put_u64(uint8_t *bytes, size_t offset, uint64_t value) {
    put_u32(bytes, offset, (uint32_t)value);
    put_u32(bytes, offset + 4, (uint32_t)(value >> 32));
}

static void write_unit_q4_matrix(uint8_t *payload, size_t offset) {
    size_t row, index;
    for (row = 0; row < 32; ++row) {
        uint8_t *block = payload + offset + row * 18;
        block[0] = 0x00;
        block[1] = 0x3c;
        for (index = 0; index < 16; ++index) block[2 + index] = 0x88;
        block[2] = 0x89;
    }
}

static int write_fixture(void) {
    const size_t component_bytes = 32 * 18;
    const size_t payload_bytes = 3 * component_bytes;
    const size_t record_stride = 4096;
    const size_t total_bytes = 4096 + 2 * record_stride;
    const size_t fused_offset = 64;
    const size_t fused_stride = 2 * component_bytes;
    const size_t down_offset = fused_offset + 2 * fused_stride;
    const size_t down_stride = component_bytes;
    const size_t source_bytes = down_offset + 2 * down_stride;
    uint8_t *packed = (uint8_t *)calloc(total_bytes, 1);
    uint8_t *source = (uint8_t *)calloc(source_bytes, 1);
    float scales[2] = {0.75F, 1.25F};
    FILE *file;
    int expert;
    if (!packed || !source) { free(packed); free(source); return -1; }
    memcpy(source + 16, scales, sizeof(scales));
    memcpy(packed, "G4EXPK01", 8);
    put_u32(packed, 8, 1);
    put_u32(packed, 12, 0);
    put_u32(packed, 16, 2);
    put_u32(packed, 20, 2);
    put_u32(packed, 24, 4096);
    put_u64(packed, 32, 4096);
    put_u64(packed, 40, payload_bytes);
    put_u64(packed, 48, record_stride);
    put_u64(packed, 56, component_bytes);
    put_u64(packed, 64, component_bytes);
    put_u64(packed, 72, component_bytes);
    for (expert = 0; expert < 2; ++expert) {
        uint8_t *record = packed + 4096 + (size_t)expert * record_stride;
        write_unit_q4_matrix(record, 0);
        write_unit_q4_matrix(record, component_bytes);
        write_unit_q4_matrix(record, 2 * component_bytes);
        write_unit_q4_matrix(source, fused_offset + (size_t)expert * fused_stride);
        write_unit_q4_matrix(source, fused_offset + (size_t)expert * fused_stride +
                                     component_bytes);
        write_unit_q4_matrix(source, down_offset + (size_t)expert * down_stride);
    }
    file = fopen(PACKED_FILE, "wb");
    if (!file || fwrite(packed, 1, total_bytes, file) != total_bytes) {
        if (file) fclose(file);
        free(packed);
        free(source);
        return -1;
    }
    if (fclose(file) != 0) { free(packed); free(source); return -1; }
    free(packed);

    file = fopen(SOURCE_FILE, "wb");
    if (!file || fwrite(source, 1, source_bytes, file) != source_bytes) {
        if (file) fclose(file);
        free(source);
        return -1;
    }
    free(source);
    if (fclose(file) != 0) return -1;

    file = fopen(MANIFEST_FILE, "wb");
    if (!file) return -1;
    fprintf(file,
        "{\n"
        "  \"format\": \"g4lab-expert-manifest-v2\",\n"
        "  \"source\": \"%s\",\n"
        "  \"architecture\": \"gemma4\",\n"
        "  \"expert_count\": 2,\n"
        "  \"expert_used_count\": 2,\n"
        "  \"layers\": [{\n"
        "    \"layer\": 0, \"expert_count\": 2,\n"
        "    \"source_layout\": \"fused_gate_up\",\n"
        "    \"model_width\": 32, \"expert_width\": 32,\n"
        "    \"payload_bytes\": %zu, \"record_stride\": %zu,\n"
        "    \"packed_file\": \"%s\",\n"
        "    \"expert_scale\": {\"ggml_type\": 0, \"source_offset\": 16, \"scalar_bytes\": 4},\n"
        "    \"components\": [\n"
        "      {\"role\": \"gate\", \"ggml_type\": 2, \"type_name\": \"Q4_0\", \"slice_bytes\": %zu, \"source_offset\": %zu, \"source_expert_stride\": %zu, \"source_within_expert_offset\": 0},\n"
        "      {\"role\": \"up\",   \"ggml_type\": 2, \"type_name\": \"Q4_0\", \"slice_bytes\": %zu, \"source_offset\": %zu, \"source_expert_stride\": %zu, \"source_within_expert_offset\": %zu},\n"
        "      {\"role\": \"down\", \"ggml_type\": 2, \"type_name\": \"Q4_0\", \"slice_bytes\": %zu, \"source_offset\": %zu, \"source_expert_stride\": %zu, \"source_within_expert_offset\": 0}\n"
        "    ]\n"
        "  }]\n"
        "}\n",
        SOURCE_NAME, payload_bytes, record_stride, PACKED_NAME,
        component_bytes, fused_offset, fused_stride,
        component_bytes, fused_offset, fused_stride, component_bytes,
        component_bytes, down_offset, down_stride);
    if (fclose(file) != 0) return -1;
    return 0;
}

static void cleanup_fixture(void) {
    remove(MANIFEST_FILE);
    remove(PACKED_FILE);
    remove(SOURCE_FILE);
    remove(USAGE_FILE);
    remove(USAGE_FILE ".tmp");
    remove_dir(TEST_DIR);
}

static int close_enough(float actual, float expected, float tolerance,
                        const char *label) {
    if (fabsf(actual - expected) > tolerance) {
        fprintf(stderr, "%s: expected %.9g, received %.9g\n",
                label, expected, actual);
        return 0;
    }
    return 1;
}

int main(void) {
    coli_gemma4_backend gemma;
    coli_expert_backend backend;
    const coli_gemma4_layer *layer;
    float input[32] = {0};
    float output[32] = {0};
    uint32_t ids[2] = {0, 1};
    float weights[2] = {0.25F, 0.75F};
    float base, expected;
    coli_gemma4_cache_stats cache_stats;
    uint64_t usage_total = 0;
    int i;
    int status = 1;

    {
        uint8_t q6[2 * 210];
        float q6_input[256];
        float q6_row[256];
        float q6_output[2];
        float q6_sum = 0.0F;
        size_t index;
        memset(q6, 0, sizeof(q6));
        for (index = 0; index < 2; ++index) {
            uint8_t *block = q6 + index * 210;
            memset(block, 0x11, 128);
            memset(block + 128, 0xaa, 64);
            memset(block + 192, index ? 2 : 1, 16);
            block[208] = 0x00;
            block[209] = 0x3c;
        }
        for (index = 0; index < 256; ++index) {
            q6_input[index] = (float)(index + 1);
            q6_sum += q6_input[index];
        }
        if (coli_gemma4_q6_k_row(q6, 2, 256, 1, q6_row) != 0 ||
            coli_gemma4_q6_k_matvec(q6, 2, 256, q6_input, q6_output) != 0) {
            fprintf(stderr, "Q6_K helper failed\n");
            return 1;
        }
        for (index = 0; index < 256; ++index) {
            if (!close_enough(q6_row[index], 2.0F, 0.0F, "Q6_K row"))
                return 1;
        }
        if (!close_enough(q6_output[0], q6_sum, 0.0F, "Q6_K matvec row 0") ||
            !close_enough(q6_output[1], 2.0F * q6_sum, 0.0F,
                          "Q6_K matvec row 1")) return 1;
    }

    cleanup_fixture();
    if (make_dir(TEST_DIR) != 0) {
        fprintf(stderr, "failed to create Gemma test directory\n");
        return 1;
    }
    if (!close_enough(coli_gemma4_fp16_to_fp32(0x3c00), 1.0F, 0.0F,
                      "fp16 one") ||
        !close_enough(coli_gemma4_fp16_to_fp32(0xc000), -2.0F, 0.0F,
                      "fp16 minus two")) {
        return 1;
    }
    if (write_fixture() != 0) {
        fprintf(stderr, "failed to create Gemma test fixture\n");
        goto cleanup;
    }
    if (coli_gemma4_open_packed(&gemma, TEST_DIR) != 0) {
        fprintf(stderr, "open_packed: %s\n", coli_gemma4_last_error(&gemma));
        goto cleanup;
    }
    if (gemma.config.n_layer != 1 || gemma.config.n_embd != 32 ||
        gemma.config.n_expert != 2 || gemma.config.n_expert_used != 2 ||
        gemma.config.n_expert_ff != 32) {
        fprintf(stderr, "parsed Gemma configuration is incorrect\n");
        goto close_backend;
    }
    layer = coli_gemma4_find_layer(&gemma, 0);
    if (!layer || layer->payload_bytes != 1728 || !layer->has_scale ||
        !coli_gemma4_has_packed_layer(&gemma, layer)) {
        fprintf(stderr, "parsed Gemma layer is incorrect\n");
        goto close_backend;
    }
    input[0] = 1.0F;
    backend = coli_gemma4_expert_backend(&gemma);
    if (backend.prepare_layer(backend.ctx, 0, ids, 2) != 0 ||
        backend.run_experts(backend.ctx, 0, ids, weights, 2, input, output) != 0) {
        fprintf(stderr, "run_experts: %s\n", coli_gemma4_last_error(&gemma));
        goto close_backend;
    }
    base = coli_gemma4_gelu_tanh(1.0F);
    expected = base * (0.25F * 0.75F + 0.75F * 1.25F);
    for (i = 0; i < 32; ++i) {
        if (!close_enough(output[i], expected, 1.0e-6F,
                          "weighted expert aggregate"))
            goto close_backend;
    }
    if (remove(PACKED_FILE) != 0 ||
        coli_gemma4_has_packed_layer(&gemma, layer)) {
        fprintf(stderr, "failed to switch fixture to direct GGUF fallback\n");
        goto close_backend;
    }
    memset(output, 0, sizeof(output));
    if (backend.run_experts(backend.ctx, 0, ids, weights, 2, input, output) != 0) {
        fprintf(stderr, "direct GGUF fallback: %s\n",
                coli_gemma4_last_error(&gemma));
        goto close_backend;
    }
    for (i = 0; i < 32; ++i) {
        if (!close_enough(output[i], expected, 1.0e-6F,
                          "direct GGUF expert aggregate"))
            goto close_backend;
    }
    if (coli_gemma4_cache_configure(&gemma, 2) != 0 ||
        backend.prepare_layer(backend.ctx, 0, ids, 2) != 0 ||
        backend.run_experts(backend.ctx, 0, ids, weights, 2, input, output) != 0 ||
        backend.prepare_layer(backend.ctx, 0, ids, 2) != 0 ||
        backend.run_experts(backend.ctx, 0, ids, weights, 2, input, output) != 0) {
        fprintf(stderr, "cached run_experts: %s\n",
                coli_gemma4_last_error(&gemma));
        goto close_backend;
    }
    coli_gemma4_cache_get_stats(&gemma, &cache_stats);
    if (cache_stats.capacity != 2 || cache_stats.resident != 2 ||
        cache_stats.misses != 2 || cache_stats.hits != 2 ||
        cache_stats.evictions != 0 || cache_stats.bytes_loaded != 3464) {
        fprintf(stderr, "expert cache statistics are incorrect\n");
        goto close_backend;
    }
    for (i = 0; i < 32; ++i) {
        if (!close_enough(output[i], expected, 1.0e-6F,
                          "cached expert aggregate"))
            goto close_backend;
    }
    if (coli_gemma4_cache_configure(&gemma, 2) != 0 ||
        coli_gemma4_prefetch_configure(&gemma, 1) != 0 ||
        backend.prepare_layer(backend.ctx, 0, ids, 2) != 0 ||
        backend.run_experts(backend.ctx, 0, ids, weights, 2,
                            input, output) != 0 ||
        backend.prepare_layer(backend.ctx, 0, ids, 2) != 0 ||
        backend.run_experts(backend.ctx, 0, ids, weights, 2,
                            input, output) != 0) {
        fprintf(stderr, "asynchronous expert prefetch failed: %s\n",
                coli_gemma4_last_error(&gemma));
        goto close_backend;
    }
    coli_gemma4_cache_get_stats(&gemma, &cache_stats);
    if (cache_stats.prefetch_launches != 1 ||
        cache_stats.prefetched_records != 2 || cache_stats.misses != 2 ||
        cache_stats.hits != 2) {
        fprintf(stderr, "asynchronous prefetch statistics are incorrect\n");
        goto close_backend;
    }
    for (i = 0; i < 32; ++i) {
        if (!close_enough(output[i], expected, 1.0e-6F,
                          "prefetched expert aggregate"))
            goto close_backend;
    }
    if (coli_gemma4_cache_configure(&gemma, 2) != 0 ||
        coli_gemma4_prefetch_configure(&gemma, 1) != 0 ||
        coli_gemma4_lookahead_configure(&gemma, 1) != 0 ||
        !(backend = coli_gemma4_expert_backend(&gemma)).prefetch_layer ||
        backend.prefetch_layer(backend.ctx, 0, ids, 2) != 0 ||
        backend.prepare_layer(backend.ctx, 0, ids, 2) != 0 ||
        backend.run_experts(backend.ctx, 0, ids, weights, 2,
                            input, output) != 0) {
        fprintf(stderr, "next-layer expert lookahead failed: %s\n",
                coli_gemma4_last_error(&gemma));
        goto close_backend;
    }
    coli_gemma4_cache_get_stats(&gemma, &cache_stats);
    if (cache_stats.lookahead_launches != 1 ||
        cache_stats.lookahead_records != 2 ||
        cache_stats.lookahead_matches != 2 ||
        cache_stats.lookahead_selected != 2 ||
        cache_stats.hits != 2) {
        fprintf(stderr, "next-layer lookahead statistics are incorrect\n");
        goto close_backend;
    }
    {
        FILE *usage = fopen(USAGE_FILE, "w");
        int usage_ok = usage && fprintf(usage, "0 1 20\n0 0 3\n") >= 0;
        if (usage && fclose(usage) != 0) usage_ok = 0;
        if (!usage_ok) {
            fprintf(stderr, "cannot create usage-profile fixture\n");
            goto close_backend;
        }
    }
    if (coli_gemma4_cache_configure(&gemma, 3) != 0 ||
        coli_gemma4_cache_set_pinned(&gemma, 1) != 0 ||
        coli_gemma4_usage_load(&gemma, USAGE_FILE, &usage_total) != 0 ||
        usage_total != 23 ||
        backend.prepare_layer(backend.ctx, 0, ids + 1, 1) != 0 ||
        coli_gemma4_usage_save(&gemma, USAGE_FILE, &usage_total) != 0 ||
        usage_total != 24) {
        fprintf(stderr, "persistent usage profile failed: %s\n",
                coli_gemma4_last_error(&gemma));
        goto close_backend;
    }
    coli_gemma4_cache_get_stats(&gemma, &cache_stats);
    if (cache_stats.resident != 1 || cache_stats.misses != 0 ||
        cache_stats.preloads != 1 ||
        cache_stats.hits != 1 || cache_stats.pinned_hits != 1) {
        fprintf(stderr, "usage-profile seed did not warm the protected slot\n");
        goto close_backend;
    }
    if (coli_gemma4_cache_configure(&gemma, 3) != 0 ||
        coli_gemma4_cache_set_pinned(&gemma, 1) != 0 ||
        coli_gemma4_usage_load(&gemma, USAGE_FILE, &usage_total) != 0 ||
        usage_total != 24 ||
        backend.prepare_layer(backend.ctx, 0, ids + 1, 1) != 0) {
        fprintf(stderr, "saved usage profile cannot warm a new cache: %s\n",
                coli_gemma4_last_error(&gemma));
        goto close_backend;
    }
    coli_gemma4_cache_get_stats(&gemma, &cache_stats);
    if (cache_stats.resident != 1 || cache_stats.misses != 0 ||
        cache_stats.preloads != 1 ||
        cache_stats.hits != 1 || cache_stats.pinned_hits != 1) {
        fprintf(stderr, "reloaded usage profile did not seed hottest expert\n");
        goto close_backend;
    }
    if (coli_gemma4_cache_configure(&gemma, 3) != 0 ||
        coli_gemma4_cache_set_pinned(&gemma, 1) != 0 ||
        backend.prepare_layer(backend.ctx, 0, ids, 1) != 0) {
        fprintf(stderr, "cannot initialize learned expert cache: %s\n",
                coli_gemma4_last_error(&gemma));
        goto close_backend;
    }
    for (i = 0; i < 8; ++i) {
        if (backend.prepare_layer(backend.ctx, 0, ids + 1, 1) != 0) {
            fprintf(stderr, "learned expert cache access failed\n");
            goto close_backend;
        }
    }
    memset(output, 0, sizeof(output));
    if (backend.run_experts(backend.ctx, 0, ids, weights, 2,
                            input, output) != 0) {
        fprintf(stderr, "learned cached run: %s\n",
                coli_gemma4_last_error(&gemma));
        goto close_backend;
    }
    coli_gemma4_cache_get_stats(&gemma, &cache_stats);
    if (cache_stats.capacity != 3 || cache_stats.pinned_capacity != 1 ||
        cache_stats.misses != 2 || cache_stats.hits != 7 ||
        cache_stats.pinned_hits != 2 || cache_stats.promotions != 1) {
        fprintf(stderr, "learned expert cache statistics are incorrect\n");
        goto close_backend;
    }
    for (i = 0; i < 32; ++i) {
        if (!close_enough(output[i], expected, 1.0e-6F,
                          "learned cached expert aggregate"))
            goto close_backend;
    }
    if (coli_gemma4_cache_configure(&gemma, 1) != 0 ||
        backend.prepare_layer(backend.ctx, 0, ids, 2) != 0 ||
        backend.run_experts(backend.ctx, 0, ids, weights, 2, input, output) != 0) {
        fprintf(stderr, "single-slot cached run: %s\n",
                coli_gemma4_last_error(&gemma));
        goto close_backend;
    }
    coli_gemma4_cache_get_stats(&gemma, &cache_stats);
    if (cache_stats.capacity != 1 || cache_stats.resident != 1 ||
        cache_stats.misses != 4 || cache_stats.hits != 0 ||
        cache_stats.evictions != 3) {
        fprintf(stderr, "expert cache eviction statistics are incorrect\n");
        goto close_backend;
    }
    for (i = 0; i < 32; ++i) {
        if (!close_enough(output[i], expected, 1.0e-6F,
                          "evicted expert aggregate"))
            goto close_backend;
    }
    if (coli_gemma4_cache_configure(&gemma, 0) != 0) {
        fprintf(stderr, "failed to disable expert cache\n");
        goto close_backend;
    }
    ids[0] = 2;
    if (backend.prepare_layer(backend.ctx, 0, ids, 1) == 0) {
        fprintf(stderr, "out-of-range expert was accepted\n");
        goto close_backend;
    }
    puts("Gemma records, LRU, pinning, usage persistence, and async prefetch passed");
    status = 0;

close_backend:
    coli_gemma4_close(&gemma);
cleanup:
    cleanup_fixture();
    return status;
}
