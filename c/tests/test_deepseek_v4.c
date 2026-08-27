/* Merged DeepSeek V4 unit tests (GLM-style single harness). */
#include "../deepseek_v4_internal.h"
#include "../compat.h"
#include "../native_quant.h"
#include "../native_quant_fp4_rows16.h"

#include <assert.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* ==== begin test_deepseek_v4_attention_cache.c ==== */
/* umbrella headers */
/* <math.h> */
/* <stdio.h> */
static int test_attention_cache(void) {
    ColiDeepSeekV4AttentionCache *cache = NULL;
    if (coli_v4_attention_cache_create(&cache, 4, 4, 2, 16) != 0) return 1;
    float query[2] = {1, 0};
    float sink[1] = {0};
    float output[2];
    for (int position = 0; position < 4; position++) {
        float window[2] = {(float)position, 1.0f};
        float compressed[2] = {1.5f, 1.0f};
        if (coli_v4_attention_cache_step(cache, output, query, window,
                                         position == 3 ? compressed : NULL,
                                         sink, 1, position, 1.0f) != 0)
            return 1;
    }
    if (!isfinite(output[0]) || !isfinite(output[1]) || output[0] <= 1.5f)
        return 1;
    if (coli_v4_attention_cache_step(cache, output, query,
                                     (float[2]){4, 1}, (float[2]){2, 1},
                                     sink, 1, 4, 1.0f) == 0)
        return 1;
    coli_v4_attention_cache_reset(cache);
    coli_v4_attention_cache_destroy(cache);
    puts("DeepSeek-V4 attention cache tests: ok");
    return 0;
}
/* ==== end test_deepseek_v4_attention_cache.c ==== */

/* ==== begin test_deepseek_v4_config.c ==== */
/* umbrella headers */
/* <stdio.h> */
static int expect_config_rejected(const char *base, const char *needle,
                                  const char *replacement, const char *label) {
    const char *match = strstr(base, needle);
    if (!match) {
        fprintf(stderr, "%s: test replacement target not found\n", label);
        return 1;
    }
    size_t prefix = (size_t)(match - base);
    const char *suffix = match + strlen(needle);
    size_t length = prefix + strlen(replacement) + strlen(suffix) + 1;
    char *mutated = malloc(length);
    if (!mutated) return 1;
    memcpy(mutated, base, prefix);
    strcpy(mutated + prefix, replacement);
    strcpy(mutated + prefix + strlen(replacement), suffix);
    ColiDeepSeekV4Config config;
    char error[256] = {0};
    int accepted =
        coli_v4_config_parse(&config, mutated, error, sizeof(error)) == 0;
    free(mutated);
    if (accepted) {
        fprintf(stderr, "%s: malformed config was accepted\n", label);
        return 1;
    }
    return 0;
}

static int test_config(int argc, char **argv) {
    static const char config_json[] =
        "{\"model_type\":\"deepseek_v4\",\"expert_dtype\":\"fp4\","
        "\"scoring_func\":\"sqrtsoftplus\",\"topk_method\":\"noaux_tc\","
        "\"hidden_size\":128,\"num_hidden_layers\":3,"
        "\"num_attention_heads\":4,\"head_dim\":32,\"q_lora_rank\":64,"
        "\"qk_rope_head_dim\":8,\"o_groups\":2,\"o_lora_rank\":64,"
        "\"sliding_window\":16,\"index_n_heads\":4,\"index_head_dim\":16,"
        "\"index_topk\":8,\"n_routed_experts\":8,\"num_experts_per_tok\":2,"
        "\"n_shared_experts\":1,\"moe_intermediate_size\":32,"
        "\"num_hash_layers\":1,\"num_nextn_predict_layers\":1,"
        "\"dspark_block_size\":5,\"dspark_noise_token_id\":255,"
        "\"dspark_markov_rank\":16,"
        "\"hc_mult\":4,\"hc_sinkhorn_iters\":5,\"vocab_size\":256,"
        "\"max_position_embeddings\":4096,\"rms_norm_eps\":1e-6,"
        "\"hc_eps\":1e-6,\"routed_scaling_factor\":1.5,\"swiglu_limit\":10,"
        "\"rope_theta\":10000,\"compress_rope_theta\":40000,"
        "\"compress_ratios\":[0,4,128,0],"
        "\"rope_scaling\":{\"original_max_position_embeddings\":1024,"
        "\"beta_fast\":32,\"beta_slow\":1,\"factor\":4},"
        "\"quantization_config\":{\"fmt\":\"e4m3\",\"scale_fmt\":\"ue8m0\"}}";
    ColiDeepSeekV4Config config;
    char error[256];
    if (coli_v4_config_parse(&config, config_json, error, sizeof(error)) != 0) {
        fprintf(stderr, "%s\n", error);
        return 1;
    }
    if (config.hidden_size != 128 || config.num_hidden_layers != 3 ||
        config.n_routed_experts != 8 || config.num_experts_per_tok != 2 ||
        config.dspark_block_size != 5 || config.dspark_noise_token_id != 255 ||
        config.dspark_markov_rank != 16 ||
        config.compress_ratio_count != 4 || config.compress_ratios[1] != 4 ||
        config.compress_ratios[2] != 128 || config.rope_factor != 4.0f)
        return 1;
    {
        static const char needle[] = "\"compress_ratios\":[0,4,128,0]";
        static const char replacement[] =
            "\"compress_ratios\":[0,4,128,0,0,0]";
        const char *at = strstr(config_json, needle);
        char extended[sizeof(config_json) + 8];
        if (!at || snprintf(extended, sizeof(extended), "%.*s%s%s",
                            (int)(at - config_json), config_json, replacement,
                            at + strlen(needle)) < 0 ||
            coli_v4_config_parse(&config, extended, error, sizeof(error)) != 0 ||
            config.compress_ratio_count != 6)
            return 1;
    }
    static const struct {
        const char *label;
        const char *needle;
        const char *replacement;
    } malformed_numbers[] = {
        {"integer-nan", "\"hidden_size\":128", "\"hidden_size\":NaN"},
        {"integer-infinity", "\"hidden_size\":128", "\"hidden_size\":1e309"},
        {"integer-fraction", "\"hidden_size\":128", "\"hidden_size\":1.5"},
        {"integer-overflow", "\"hidden_size\":128", "\"hidden_size\":2147483648"},
        {"integer-underflow", "\"hidden_size\":128", "\"hidden_size\":-2147483649"},
        {"float-nan", "\"rms_norm_eps\":1e-6", "\"rms_norm_eps\":NaN"},
        {"float-infinity", "\"rms_norm_eps\":1e-6", "\"rms_norm_eps\":1e309"},
        {"float-overflow", "\"rms_norm_eps\":1e-6", "\"rms_norm_eps\":1e39"},
        {"ratio-nan", "\"compress_ratios\":[0,4,128,0]",
         "\"compress_ratios\":[0,NaN,128,0]"},
        {"ratio-fraction", "\"compress_ratios\":[0,4,128,0]",
         "\"compress_ratios\":[0,1.5,128,0]"},
        {"ratio-overflow", "\"compress_ratios\":[0,4,128,0]",
         "\"compress_ratios\":[0,2147483648,128,0]"},
    };
    for (size_t i = 0;
         i < sizeof(malformed_numbers) / sizeof(malformed_numbers[0]); i++)
        if (expect_config_rejected(
                config_json, malformed_numbers[i].needle,
                malformed_numbers[i].replacement,
                malformed_numbers[i].label))
            return 1;
    if (argc > 1) {
        if (coli_v4_config_load(&config, argv[1], error, sizeof(error)) != 0) {
            fprintf(stderr, "%s\n", error);
            return 1;
        }
        int flash = config.hidden_size == 4096 &&
            config.num_hidden_layers == 43 &&
            config.n_routed_experts == 256 &&
            config.compress_ratio_count >= config.num_hidden_layers &&
            config.compress_ratios[0] == 0;
        int pro = config.hidden_size == 7168 &&
            config.num_hidden_layers == 61 &&
            config.n_routed_experts == 384 &&
            config.compress_ratio_count >= config.num_hidden_layers &&
            config.compress_ratios[0] == 128;
        if ((!flash && !pro) || config.num_experts_per_tok != 6 ||
            config.compress_ratios[2] != 4 ||
            config.compress_ratios[3] != 128 || config.hc_mult != 4)
            return 1;
    }
    puts("DeepSeek-V4 config tests: ok");
    return 0;
}
/* ==== end test_deepseek_v4_config.c ==== */

/* ==== begin test_deepseek_v4_expert.c ==== */
/* umbrella headers */
/* shared */
/* <math.h> */
/* <stdint.h> */
/* <stdio.h> */
/* <string.h> */
static void make_tensor(ColiTensorView *view, uint8_t *data, uint8_t *scales,
                        int rows, int columns, uint8_t code) {
    memset(data, (int)(code | (code << 4)), (size_t)rows * columns / 2);
    memset(scales, 0x7f, (size_t)rows * columns / 32);
    *view = (ColiTensorView){
        COLI_TENSOR_FP4_NATIVE_BLOCK, COLI_SCALE_UE8M0,
        data, scales, (size_t)rows * columns / 2,
        (size_t)rows * columns / 32, rows, columns, 1, 32, NULL
    };
}

static int test_expert(void) {
    enum { DIMENSION = 128, INTERMEDIATE = 128 };
    uint8_t gate_data[INTERMEDIATE * DIMENSION / 2];
    uint8_t up_data[INTERMEDIATE * DIMENSION / 2];
    uint8_t down_data[DIMENSION * INTERMEDIATE / 2];
    uint8_t gate_scale[INTERMEDIATE * DIMENSION / 32];
    uint8_t up_scale[INTERMEDIATE * DIMENSION / 32];
    uint8_t down_scale[DIMENSION * INTERMEDIATE / 32];
    ColiExpertView expert = {0};
    make_tensor(&expert.gate, gate_data, gate_scale,
                INTERMEDIATE, DIMENSION, 1); /* +0.5 */
    make_tensor(&expert.up, up_data, up_scale,
                INTERMEDIATE, DIMENSION, 1);
    make_tensor(&expert.down, down_data, down_scale,
                DIMENSION, INTERMEDIATE, 1);
    float input[DIMENSION], output[DIMENSION];
    for (int index = 0; index < DIMENSION; index++) input[index] = 0.01f;
    if (coli_v4_expert_forward_ref(output, &expert, input, 0.75f, 10.0f) != 0)
        return 1;
    for (int index = 0; index < DIMENSION; index++)
        if (!isfinite(output[index]) || output[index] != output[0]) return 1;
    if (!(output[0] > 0.0f)) return 1;
    puts("DeepSeek-V4 expert tests: ok");
    return 0;
}

/* The production prefill path must use the dormant FP4 batch kernel without
 * moving a single result bit.  Varied packed nibbles, scales, activations and
 * route weights prevent a uniform fixture from hiding an ordering change. */
static int test_expert_batch(void) {
    /* 4096 columns are intentional: the old 32-column-partial batch order
     * matched a 128-column toy by accident but diverged at Flash width. */
    enum { DIMENSION = 4096, INTERMEDIATE = 128, BATCH = 5 };
    static uint8_t gate_data[INTERMEDIATE * DIMENSION / 2];
    static uint8_t up_data[INTERMEDIATE * DIMENSION / 2];
    static uint8_t down_data[DIMENSION * INTERMEDIATE / 2];
    static uint8_t gate_scale[INTERMEDIATE * DIMENSION / 32];
    static uint8_t up_scale[INTERMEDIATE * DIMENSION / 32];
    static uint8_t down_scale[DIMENSION * INTERMEDIATE / 32];
    ColiExpertView expert = {0};
    make_tensor(&expert.gate, gate_data, gate_scale,
                INTERMEDIATE, DIMENSION, 0);
    make_tensor(&expert.up, up_data, up_scale,
                INTERMEDIATE, DIMENSION, 0);
    make_tensor(&expert.down, down_data, down_scale,
                DIMENSION, INTERMEDIATE, 0);
    uint32_t state = 0x1159u;
    uint8_t *data[] = {gate_data, up_data, down_data};
    uint8_t *scales[] = {gate_scale, up_scale, down_scale};
    for (int matrix = 0; matrix < 3; matrix++) {
        for (size_t i = 0; i < sizeof(gate_data); i++) {
            state = state * 1664525u + 1013904223u;
            data[matrix][i] = (uint8_t)(state >> 24);
        }
        for (size_t i = 0; i < sizeof(gate_scale); i++) {
            state = state * 1664525u + 1013904223u;
            scales[matrix][i] = (uint8_t)(125 + ((state >> 24) % 5));
        }
    }
    float inputs[BATCH * DIMENSION];
    float route_weights[BATCH] = {0.125f, 0.5f, 0.75f, 1.0f, 1.375f};
    float scalar[BATCH * DIMENSION], batched[BATCH * DIMENSION];
    for (int item = 0; item < BATCH; item++)
        for (int column = 0; column < DIMENSION; column++)
            inputs[(size_t)item * DIMENSION + column] =
                0.0078125f * (float)(((item + 3) * (column + 5)) % 29 - 14);
    for (int item = 0; item < BATCH; item++)
        if (coli_v4_expert_forward_ref(
                scalar + (size_t)item * DIMENSION, &expert,
                inputs + (size_t)item * DIMENSION, route_weights[item],
                10.0f))
            return 1;
    coli_v4_test_fp4_batch_calls = 0;
    if (coli_v4_expert_forward_batch_ref(
            batched, &expert, inputs, route_weights, BATCH, 10.0f))
        return 1;
    if (coli_v4_test_fp4_batch_calls != 3) {
        fprintf(stderr, "expected 3 FP4 batch matrices, got %llu\n",
                (unsigned long long)coli_v4_test_fp4_batch_calls);
        return 1;
    }
    if (memcmp(scalar, batched, sizeof(scalar)) != 0) {
        for (int i = 0; i < BATCH * DIMENSION; i++)
            if (scalar[i] != batched[i]) {
                fprintf(stderr,
                        "expert batch mismatch item=%d column=%d: %.9g vs %.9g\n",
                        i / DIMENSION, i % DIMENSION,
                        (double)scalar[i], (double)batched[i]);
                break;
            }
        return 1;
    }
    /* The public batch entry keeps batch-one on the dedicated matvec path;
     * protect its exact-output contract alongside the separate perf check. */
    float scalar_matrix[INTERMEDIATE], batch_one[INTERMEDIATE];
    coli_fp4_matvec_rows16_order(
        scalar_matrix, gate_data, gate_scale, inputs, DIMENSION, INTERMEDIATE);
    coli_fp4_matmul_batch_rows16_order(
        batch_one, gate_data, gate_scale, inputs, 1, DIMENSION, INTERMEDIATE);
    if (memcmp(scalar_matrix, batch_one, sizeof(scalar_matrix)) != 0) {
        fputs("FP4 batch-one dispatch diverged from scalar matvec\n", stderr);
        return 1;
    }
    uint64_t calls = coli_v4_test_fp4_batch_calls;
    if (coli_v4_expert_forward_batch_ref(
            batched, &expert, inputs, route_weights, 0, 10.0f) == 0 ||
        coli_v4_test_fp4_batch_calls != calls)
        return 1;
    puts("DeepSeek-V4 expert batch: ok (3 matrices, scalar-exact bits)");
    return 0;
}

/* #1136: a rows16-packed copy and the row-major original of the SAME matrix
 * must produce bit-identical matvec outputs — the reference path accumulates
 * in the rows16 order now, so which kernel an expert takes (cache residency)
 * can no longer move greedy text. Varied nibbles AND varied per-group scales:
 * uniform data would let the old group-partial-sum order pass by accident. */
static int test_rows16_convergence(void) {
#ifndef COLI_FP4_ROWS16_KERNEL
    puts("DeepSeek-V4 rows16 convergence: skipped (no rows16 kernel on this ISA)");
    return 0;
#else
    enum { ROWS = 32, COLUMNS = 128 };
    static uint8_t data[ROWS * COLUMNS / 2], scales[ROWS * COLUMNS / 32];
    static uint8_t packed_data[ROWS * COLUMNS / 2], packed_scales[ROWS * COLUMNS / 32];
    uint32_t state = 0x1136u;
    for (size_t i = 0; i < sizeof(data); i++) {
        state = state * 1664525u + 1013904223u;
        data[i] = (uint8_t)(state >> 24);
    }
    for (size_t i = 0; i < sizeof(scales); i++) {
        state = state * 1664525u + 1013904223u;
        scales[i] = (uint8_t)(120 + ((state >> 24) & 15));   /* e8m0 near 1.0 */
    }
    ColiTensorView source = {
        COLI_TENSOR_FP4_NATIVE_BLOCK, COLI_SCALE_UE8M0,
        data, scales, sizeof(data), sizeof(scales), ROWS, COLUMNS, 1, 32, NULL
    };
    if (coli_fp4_pack_rows16_v10(packed_data, packed_scales, &source) != 0)
        return 1;
    ColiTensorView packed = source;
    packed.data = packed_data; packed.scales = packed_scales;
    packed.block_rows = 16;
    float input[COLUMNS];
    for (int i = 0; i < COLUMNS; i++)
        input[i] = 0.05f * (float)((i * 7) % 13 - 6);
    /* the rows16 kernel qdq's internally; feed the converged core the same
     * qdq'd activation the ref path would use */
    float activation[COLUMNS]; uint8_t activation_scales[COLUMNS / 128];
    if (coli_fp8_activation_qdq_ref(activation, activation_scales,
                                    input, COLUMNS, 128) != 0) return 1;
    float hot[ROWS], cold[ROWS], ref[ROWS];
    if (coli_fp4_matvec_rows16_v10(hot, &packed, input) != 0) return 1;
    coli_fp4_matvec_rows16_order(cold, data, scales, activation, COLUMNS, ROWS);
    if (memcmp(hot, cold, sizeof(hot)) != 0) {
        for (int r = 0; r < ROWS; r++)
            if (hot[r] != cold[r])
                fprintf(stderr, "  row %d: rows16 %.9g vs converged ref %.9g\n",
                        r, (double)hot[r], (double)cold[r]);
        return 1;
    }
    /* and the public entry point routes rows%16==0 through the converged core */
    if (coli_fp4_matvec_ref(ref, &source, input) != 0) return 1;
    if (memcmp(hot, ref, sizeof(hot)) != 0) return 1;
    puts("DeepSeek-V4 rows16 convergence: ok (hot and cold bits identical)");
    return 0;
#endif
}
/* ==== end test_deepseek_v4_expert.c ==== */

/* ==== begin test_deepseek_v4_expert_store.c ==== */
/* _GNU_SOURCE */
/* umbrella headers */
/* shared */
/* <fcntl.h> */
/* <stdint.h> */
/* <stdio.h> */
/* <stdlib.h> */
/* <string.h> */
/* <unistd.h> */
static int write_all(int fd, const void *data, size_t length) {
    const unsigned char *bytes = data;
    while (length) {
        ssize_t count = write(fd, bytes, length);
        if (count <= 0) return -1;
        bytes += count;
        length -= (size_t)count;
    }
    return 0;
}

static int write_fixture_layers(const char *path, int layer_count,
                                int expert_count) {
    static const char *matrix_names[3] = {"w1", "w2", "w3"};
    if (layer_count < 1 || expert_count < 1 ||
        (size_t)layer_count > SIZE_MAX / (size_t)expert_count)
        return -1;
    size_t cells = (size_t)layer_count * expert_count;
    if (cells > (SIZE_MAX - 256) / 1024) return -1;
    size_t header_capacity = 256 + cells * 1024;
    char *header = malloc(header_capacity);
    size_t payload_size = 4 + cells * 51;
    unsigned char *payload = malloc(payload_size);
    if (!header || !payload) {
        free(payload);
        free(header);
        return -1;
    }
    size_t used = (size_t)snprintf(
        header, header_capacity,
        "{\"resident\":{\"dtype\":\"F32\",\"shape\":[1],"
        "\"data_offsets\":[3,7]}");
    for (int layer = 0; layer < layer_count; layer++) {
        for (int expert = 0; expert < expert_count; expert++) {
            size_t cell = (size_t)layer * expert_count + expert;
            size_t scale = cell ? 4 + cell * 51 : 0;
            size_t weight = scale + 3 + (cell ? 0 : 4);
            for (int matrix = 0; matrix < 3; matrix++) {
                int count = snprintf(
                    header + used, header_capacity - used,
                    ",\"layers.%d.ffn.experts.%d.%s.scale\":{"
                    "\"dtype\":\"F8_E8M0\",\"shape\":[1,1],"
                    "\"data_offsets\":[%zu,%zu]}",
                    layer, expert, matrix_names[matrix], scale + matrix,
                    scale + matrix + 1);
                if (count < 0 || (size_t)count >= header_capacity - used) {
                    free(payload); free(header); return -1;
                }
                used += (size_t)count;
            }
            for (int matrix = 0; matrix < 3; matrix++) {
                int count = snprintf(
                    header + used, header_capacity - used,
                    ",\"layers.%d.ffn.experts.%d.%s.weight\":{"
                    "\"dtype\":\"I8\",\"shape\":[1,16],"
                    "\"data_offsets\":[%zu,%zu]}",
                    layer, expert, matrix_names[matrix], weight + matrix * 16,
                    weight + (matrix + 1) * 16);
                if (count < 0 || (size_t)count >= header_capacity - used) {
                    free(payload); free(header); return -1;
                }
                used += (size_t)count;
            }
        }
    }
    if (used + 1 >= header_capacity) {
        free(payload); free(header); return -1;
    }
    header[used++] = '}';
    header[used] = '\0';
    for (size_t i = 0; i < payload_size; i++)
        payload[i] = (unsigned char)i;
    uint64_t header_length = used;
    int fd = open(path, O_CREAT | O_TRUNC | O_WRONLY | COMPAT_O_BINARY, 0600);
    if (fd < 0) {
        free(payload); free(header); return -1;
    }
    int result = write_all(fd, &header_length, sizeof(header_length)) ||
                 write_all(fd, header, (size_t)header_length) ||
                 write_all(fd, payload, payload_size);
    close(fd);
    free(payload);
    free(header);
    return result ? -1 : 0;
}

static int write_fixture_experts(const char *path, int expert_count) {
    return write_fixture_layers(path, 1, expert_count);
}

static int write_fixture(const char *path) {
    return write_fixture_experts(path, 7);
}

static int expect_fixture_expert(const ColiExpertView *view, int expert) {
    int scale = expert ? 4 + expert * 51 : 0;
    int weight = scale + 3 + (expert ? 0 : 4);
    return view->gate.format == COLI_TENSOR_FP4_NATIVE_BLOCK &&
           view->gate.rows == 1 && view->gate.columns == 32 &&
           view->gate.data_bytes == 16 && view->gate.scale_bytes == 1 &&
           ((const unsigned char *)view->gate.scales)[0] ==
               (unsigned char)scale &&
           ((const unsigned char *)view->gate.data)[0] ==
               (unsigned char)weight &&
           ((const unsigned char *)view->down.data)[0] ==
               (unsigned char)(weight + 16) &&
           ((const unsigned char *)view->up.data)[0] ==
               (unsigned char)(weight + 32);
}

static int expect_fixture_expert_at(const ColiExpertView *view, int layer,
                                    int expert, int experts_per_layer) {
    int cell = layer * experts_per_layer + expert;
    int scale = cell ? 4 + cell * 51 : 0;
    int weight = scale + 3 + (cell ? 0 : 4);
    return view->gate.format == COLI_TENSOR_FP4_NATIVE_BLOCK &&
           view->gate.rows == 1 && view->gate.columns == 32 &&
           view->gate.data_bytes == 16 && view->gate.scale_bytes == 1 &&
           ((const unsigned char *)view->gate.scales)[0] ==
               (unsigned char)scale &&
           ((const unsigned char *)view->gate.data)[0] ==
               (unsigned char)weight &&
           ((const unsigned char *)view->down.data)[0] ==
               (unsigned char)(weight + 16) &&
           ((const unsigned char *)view->up.data)[0] ==
               (unsigned char)(weight + 32);
}

enum {
    TEST_READ_SAME_EXPERT,
    TEST_READ_DISTINCT_EXPERTS
};

static pthread_mutex_t expert_hook_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t expert_hook_ready = PTHREAD_COND_INITIALIZER;
static int expert_hook_mode;
static int expert_hook_reads;
static int expert_hook_waits;
static int expert_hook_active;
static int expert_hook_max_active;
static int expert_hook_abort;

static void reset_expert_hooks(int mode) {
    pthread_mutex_lock(&expert_hook_mutex);
    expert_hook_mode = mode;
    expert_hook_reads = 0;
    expert_hook_waits = 0;
    expert_hook_active = 0;
    expert_hook_max_active = 0;
    expert_hook_abort = 0;
    pthread_mutex_unlock(&expert_hook_mutex);
}

static void abort_expert_hooks(void) {
    pthread_mutex_lock(&expert_hook_mutex);
    expert_hook_abort = 1;
    pthread_cond_broadcast(&expert_hook_ready);
    pthread_mutex_unlock(&expert_hook_mutex);
}

static void test_expert_read_hook(ColiExpertKey key) {
    (void)key;
    pthread_mutex_lock(&expert_hook_mutex);
    expert_hook_reads++;
    expert_hook_active++;
    if (expert_hook_active > expert_hook_max_active)
        expert_hook_max_active = expert_hook_active;
    pthread_cond_broadcast(&expert_hook_ready);
    if (expert_hook_mode == TEST_READ_SAME_EXPERT) {
        while (!expert_hook_abort && !expert_hook_waits &&
               expert_hook_reads < 2)
            pthread_cond_wait(&expert_hook_ready, &expert_hook_mutex);
    } else {
        /* A wait here means distinct experts were incorrectly coalesced. */
        while (!expert_hook_abort && expert_hook_reads < 2 &&
               !expert_hook_waits)
            pthread_cond_wait(&expert_hook_ready, &expert_hook_mutex);
    }
    expert_hook_active--;
    pthread_cond_broadcast(&expert_hook_ready);
    pthread_mutex_unlock(&expert_hook_mutex);
}

static void test_expert_wait_hook(ColiExpertKey key) {
    (void)key;
    pthread_mutex_lock(&expert_hook_mutex);
    expert_hook_waits++;
    pthread_cond_broadcast(&expert_hook_ready);
    pthread_mutex_unlock(&expert_hook_mutex);
}

typedef struct {
    ColiExpertStore *store;
    ColiExpertKey key;
    ColiExpertView view;
    int result;
} ExpertLookupJob;

static void *expert_lookup_worker(void *opaque) {
    ExpertLookupJob *job = opaque;
    job->result = coli_expert_lookup(job->store, job->key, &job->view);
    return NULL;
}

static int run_parallel_lookups(ColiExpertStore *store,
                                ColiExpertKey first, ColiExpertKey second,
                                ExpertLookupJob jobs[2]) {
    pthread_t threads[2];
    memset(jobs, 0, 2 * sizeof(*jobs));
    jobs[0].store = jobs[1].store = store;
    jobs[0].key = first;
    jobs[1].key = second;
    coli_v4_test_expert_read_hook = test_expert_read_hook;
    coli_v4_test_expert_wait_hook = test_expert_wait_hook;
    if (pthread_create(&threads[0], NULL, expert_lookup_worker, &jobs[0])) {
        coli_v4_test_expert_read_hook = NULL;
        coli_v4_test_expert_wait_hook = NULL;
        return -1;
    }
    if (pthread_create(&threads[1], NULL, expert_lookup_worker, &jobs[1])) {
        abort_expert_hooks();
        pthread_join(threads[0], NULL);
        if (!jobs[0].result) coli_expert_release(store, &jobs[0].view);
        coli_v4_test_expert_read_hook = NULL;
        coli_v4_test_expert_wait_hook = NULL;
        return -1;
    }
    int join_result = pthread_join(threads[0], NULL) |
                      pthread_join(threads[1], NULL);
    coli_v4_test_expert_read_hook = NULL;
    coli_v4_test_expert_wait_hook = NULL;
    return join_result || jobs[0].result || jobs[1].result ? -1 : 0;
}

static int test_expert_store(void) {
    /* Native MinGW binaries do not resolve the MSYS /tmp mount. */
    char directory[] = "colibri-v4-store-XXXXXX";
    char path[256], error[256];
    setenv("COLI_V4_AUTOPIN", "0", 1);
    setenv("COLI_V4_SAVE_USAGE", "0", 1);
    setenv("COLI_V4_ROWS16", "0", 1);
    if (!mkdtemp(directory)) { perror("mkdtemp"); return 1; }
    snprintf(path, sizeof(path), "%s/model.safetensors", directory);
    if (write_fixture(path) != 0) { perror("write_fixture"); return 1; }

    ColiDeepSeekV4ExpertStoreOptions options = {
        directory, 1, 7, 306, -1, 0
    };
    ColiExpertStore *store = NULL;
    if (coli_deepseek_v4_expert_store_open(&options, &store,
                                            error, sizeof(error)) != 0) {
        fprintf(stderr, "%s\n", error);
        return 1;
    }
    ColiExpertKey key = {0, 0};
    ColiExpertView view;
    if (store->ops->prefetch(store, &key, 1) != 1) {
        fprintf(stderr, "prefetch failed\n"); return 1;
    }
    ExpertLookupJob same[2];
    reset_expert_hooks(TEST_READ_SAME_EXPERT);
    if (run_parallel_lookups(store, key, key, same) != 0) {
        fprintf(stderr, "parallel same-expert lookup failed\n"); return 1;
    }
    if (expert_hook_reads != 1 || expert_hook_waits < 1 ||
        expert_hook_max_active != 1) {
        fprintf(stderr,
                "same-expert load was not single-flight: reads=%d waits=%d active=%d\n",
                expert_hook_reads, expert_hook_waits,
                expert_hook_max_active);
        return 1;
    }
    ColiExpertView *loaded = &same[0].view;
    if (!expect_fixture_expert(loaded, 0))
        { fprintf(stderr, "expert view mismatch: format=%d rows=%lld columns=%lld data=%zu scales=%zu bytes=%u/%u/%u/%u\n",
                  (int)loaded->gate.format, (long long)loaded->gate.rows,
                  (long long)loaded->gate.columns, loaded->gate.data_bytes,
                  loaded->gate.scale_bytes,
                  ((const unsigned char *)loaded->gate.scales)[0],
                  ((const unsigned char *)loaded->gate.data)[0],
                  ((const unsigned char *)loaded->down.data)[0],
                  ((const unsigned char *)loaded->up.data)[0]); return 1; }
    coli_expert_release(store, &same[0].view);
    {
        static const ColiExpertView zero;
        if (memcmp(&same[0].view, &zero, sizeof(same[0].view)) != 0) {
            fprintf(stderr, "release did not clear view\n");
            return 1;
        }
    }
    /* Double release of a cleared view is a no-op. */
    coli_expert_release(store, &same[0].view);
    coli_expert_release(store, &same[1].view);
    /* Invalid key lookup must fail and clear the view. */
    memset(&view, 0x3c, sizeof(view));
    if (coli_expert_lookup(store, (ColiExpertKey){9, 9}, &view) == 0) return 1;
    {
        static const ColiExpertView zero;
        if (memcmp(&view, &zero, sizeof(view)) != 0) {
            fprintf(stderr, "failed lookup did not clear view\n");
            return 1;
        }
    }
    ColiExpertStoreStats stats;
    store->ops->stats(store, &stats);
    if (stats.requests != 2 || stats.hits != 1 || stats.misses != 1 ||
        stats.prefetched != 1 || stats.bytes_read != 51 ||
        stats.resident_bytes != 51 || stats.capacity_bytes != 306)
        { fprintf(stderr, "expert stats mismatch: requests=%llu hits=%llu misses=%llu prefetched=%llu bytes=%llu resident=%llu capacity=%llu\n",
                  (unsigned long long)stats.requests,
                  (unsigned long long)stats.hits,
                  (unsigned long long)stats.misses,
                  (unsigned long long)stats.prefetched,
                  (unsigned long long)stats.bytes_read,
                  (unsigned long long)stats.resident_bytes,
                  (unsigned long long)stats.capacity_bytes); return 1; }

    ExpertLookupJob distinct[2];
    reset_expert_hooks(TEST_READ_DISTINCT_EXPERTS);
    if (run_parallel_lookups(store, (ColiExpertKey){0, 1},
                             (ColiExpertKey){0, 2}, distinct) != 0) {
        fprintf(stderr, "parallel distinct-expert lookup failed\n"); return 1;
    }
    if (expert_hook_reads != 2 || expert_hook_waits != 0 ||
        expert_hook_max_active != 2) {
        fprintf(stderr,
                "distinct expert loads did not overlap: reads=%d waits=%d active=%d\n",
                expert_hook_reads, expert_hook_waits,
                expert_hook_max_active);
        return 1;
    }
    coli_expert_release(store, &distinct[0].view);
    coli_expert_release(store, &distinct[1].view);
    store->ops->stats(store, &stats);
    if (stats.requests != 4 || stats.hits != 1 || stats.misses != 3 ||
        stats.prefetched != 1 || stats.bytes_read != 153 ||
        stats.resident_bytes != 153 || stats.capacity_bytes != 306) {
        fprintf(stderr,
                "concurrent expert stats mismatch: requests=%llu hits=%llu misses=%llu bytes=%llu resident=%llu capacity=%llu\n",
                (unsigned long long)stats.requests,
                (unsigned long long)stats.hits,
                (unsigned long long)stats.misses,
                (unsigned long long)stats.bytes_read,
                (unsigned long long)stats.resident_bytes,
                (unsigned long long)stats.capacity_bytes);
        return 1;
    }

    /* Fill the six-slot cache, make the LRU order deterministic, then force
     * an eviction. The direct index must forget expert 1, retain expert 2,
     * and publish expert 6 in the reused slot. */
    for (int expert = 1; expert <= 2; expert++) {
        ColiExpertView touched;
        if (coli_expert_lookup(store, (ColiExpertKey){0, expert}, &touched) ||
            !expect_fixture_expert(&touched, expert))
            { fprintf(stderr, "indexed cache hit mismatch: expert=%d\n", expert); return 1; }
        coli_expert_release(store, &touched);
    }
    for (int expert = 3; expert <= 5; expert++) {
        ColiExpertView filled;
        if (coli_expert_lookup(store, (ColiExpertKey){0, expert}, &filled) ||
            !expect_fixture_expert(&filled, expert))
            { fprintf(stderr, "cache fill mismatch: expert=%d\n", expert); return 1; }
        coli_expert_release(store, &filled);
    }
    if (coli_expert_lookup(store, key, &view) ||
        !expect_fixture_expert(&view, 0))
        { fprintf(stderr, "failed to refresh expert 0 LRU\n"); return 1; }
    coli_expert_release(store, &view);

    if (coli_expert_lookup(store, (ColiExpertKey){0, 6}, &view) ||
        !expect_fixture_expert(&view, 6))
        { fprintf(stderr, "eviction load mismatch\n"); return 1; }
    coli_expert_release(store, &view);
    if (coli_v4_test_expert_slot_index(store, (ColiExpertKey){0, 1}) != -1 ||
        coli_v4_test_expert_slot_index(store, (ColiExpertKey){0, 2}) < 0 ||
        coli_v4_test_expert_slot_index(store, (ColiExpertKey){0, 6}) < 0) {
        fprintf(stderr, "expert index did not track eviction\n");
        return 1;
    }
    if (coli_expert_lookup(store, (ColiExpertKey){0, 2}, &view) ||
        !expect_fixture_expert(&view, 2))
        { fprintf(stderr, "surviving index entry mismatch\n"); return 1; }
    coli_expert_release(store, &view);
    if (coli_expert_lookup(store, (ColiExpertKey){0, 1}, &view) ||
        !expect_fixture_expert(&view, 1))
        { fprintf(stderr, "reloaded index entry mismatch\n"); return 1; }
    coli_expert_release(store, &view);
    if (coli_v4_test_expert_slot_index(store, (ColiExpertKey){0, 1}) < 0) {
        fprintf(stderr, "reloaded expert was not indexed\n");
        return 1;
    }
    store->ops->stats(store, &stats);
    if (stats.requests != 13 || stats.hits != 5 || stats.misses != 8 ||
        stats.prefetched != 1 || stats.bytes_read != 408 ||
        stats.resident_bytes != 306 || stats.capacity_bytes != 306) {
        fprintf(stderr,
                "indexed eviction stats mismatch: requests=%llu hits=%llu misses=%llu bytes=%llu resident=%llu capacity=%llu\n",
                (unsigned long long)stats.requests,
                (unsigned long long)stats.hits,
                (unsigned long long)stats.misses,
                (unsigned long long)stats.bytes_read,
                (unsigned long long)stats.resident_bytes,
                (unsigned long long)stats.capacity_bytes);
        return 1;
    }
    store->ops->destroy(store);
    unlink(path);
    rmdir(directory);
    puts("DeepSeek-V4 ExpertStore tests: ok");
    return 0;
}

static int lookup_fixture_at(ColiExpertStore *store, int layer, int expert,
                             int experts_per_layer) {
    ColiExpertView view;
    if (coli_expert_lookup(store, (ColiExpertKey){layer, expert}, &view) ||
        !expect_fixture_expert_at(&view, layer, expert, experts_per_layer)) {
        fprintf(stderr, "pooled fixture mismatch: layer=%d expert=%d\n",
                layer, expert);
        return -1;
    }
    coli_expert_release(store, &view);
    return 0;
}

/* Six slots per physical layer cannot retain an eight-expert union.  Pooling
 * two layers provides twelve slots: the first sweep reads each layer-0 expert
 * once, the second sweep must read zero bytes.  The test also proves that a
 * borrowed slot remains indexed after pool close and that switching pool owner
 * does not blindly flush an already-warm expert from another layer. */
static int test_expert_store_prefill_pool(void) {
    enum { LAYERS = 2, EXPERTS = 8, SLOTS_PER_LAYER = 6 };
    char directory[] = "colibri-v4-pool-XXXXXX";
    char path[256], error[256];
    setenv("COLI_V4_AUTOPIN", "0", 1);
    setenv("COLI_V4_SAVE_USAGE", "0", 1);
    setenv("COLI_V4_ROWS16", "0", 1);
    if (!mkdtemp(directory)) { perror("mkdtemp pool"); return 1; }
    snprintf(path, sizeof(path), "%s/model.safetensors", directory);
    if (write_fixture_layers(path, LAYERS, EXPERTS)) {
        perror("write pool fixture");
        rmdir(directory);
        return 1;
    }
    ColiDeepSeekV4ExpertStoreOptions options = {
        directory, LAYERS, EXPERTS,
        (uint64_t)LAYERS * SLOTS_PER_LAYER * 51, -1, UINT64_MAX
    };
    /* Deterministic A/B.  With the ordinary six-slot partition, iterating an
     * eight-expert union in the same order twice is a cyclic 0%%-hit workload:
     * 16 reads.  The pooled twelve-slot capacity must turn the second sweep
     * into eight hits: 8 reads, exactly half the bytes. */
    ColiExpertStore *baseline = NULL;
    if (coli_deepseek_v4_expert_store_open(
            &options, &baseline, error, sizeof(error))) {
        fprintf(stderr, "baseline pool store open failed: %s\n", error);
        unlink(path); rmdir(directory);
        return 1;
    }
    int failed = 0;
    for (int sweep = 0; sweep < 2 && !failed; sweep++)
        for (int expert = 0; expert < EXPERTS; expert++)
            if (lookup_fixture_at(baseline, 0, expert, EXPERTS)) {
                failed = 1;
                break;
            }
    ColiExpertStoreStats baseline_stats;
    baseline->ops->stats(baseline, &baseline_stats);
    baseline->ops->destroy(baseline);
    if (baseline_stats.requests != 16 || baseline_stats.hits != 0 ||
        baseline_stats.misses != 16 || baseline_stats.bytes_read != 16 * 51) {
        fprintf(stderr,
                "prefill baseline mismatch: requests=%llu hits=%llu "
                "misses=%llu bytes=%llu\n",
                (unsigned long long)baseline_stats.requests,
                (unsigned long long)baseline_stats.hits,
                (unsigned long long)baseline_stats.misses,
                (unsigned long long)baseline_stats.bytes_read);
        failed = 1;
    }

    ColiExpertStore *store = NULL;
    if (coli_deepseek_v4_expert_store_open(
            &options, &store, error, sizeof(error))) {
        fprintf(stderr, "pool store open failed: %s\n", error);
        unlink(path); rmdir(directory);
        return 1;
    }
    /* A warm entry owned by the other layer must survive while unused slabs
     * are still available to the pooled layer. */
    failed |= lookup_fixture_at(store, 1, 7, EXPERTS);
    coli_v4_expert_store_prefill_pool(store, 0);
    for (int sweep = 0; sweep < 2 && !failed; sweep++)
        for (int expert = 0; expert < EXPERTS; expert++)
            if (lookup_fixture_at(store, 0, expert, EXPERTS)) {
                failed = 1;
                break;
            }
    ColiExpertStoreStats pooled_sweeps;
    store->ops->stats(store, &pooled_sweeps);
    /* Subtract the one layer-1 warmup read above. */
    uint64_t pooled_requests = pooled_sweeps.requests - 1;
    uint64_t pooled_hits = pooled_sweeps.hits;
    uint64_t pooled_misses = pooled_sweeps.misses - 1;
    uint64_t pooled_bytes = pooled_sweeps.bytes_read - 51;
    if (pooled_requests != 16 || pooled_hits != 8 || pooled_misses != 8 ||
        pooled_bytes != 8 * 51 ||
        baseline_stats.bytes_read != pooled_bytes * 2) {
        fprintf(stderr,
                "prefill pooled A/B mismatch: requests=%llu hits=%llu "
                "misses=%llu bytes=%llu baseline_bytes=%llu\n",
                (unsigned long long)pooled_requests,
                (unsigned long long)pooled_hits,
                (unsigned long long)pooled_misses,
                (unsigned long long)pooled_bytes,
                (unsigned long long)baseline_stats.bytes_read);
        failed = 1;
    }
    int borrowed = coli_v4_test_expert_slot_index(
        store, (ColiExpertKey){0, EXPERTS - 1});
    if (borrowed < SLOTS_PER_LAYER) {
        fprintf(stderr, "prefill did not borrow a slot: index=%d\n", borrowed);
        failed = 1;
    }

    coli_v4_expert_store_prefill_pool(store, 1);
    failed |= lookup_fixture_at(store, 1, 7, EXPERTS);
    coli_v4_expert_store_prefill_pool(store, -1);
    failed |= lookup_fixture_at(store, 0, 7, EXPERTS);
    failed |= lookup_fixture_at(store, 1, 7, EXPERTS);

    ColiExpertStoreStats stats;
    store->ops->stats(store, &stats);
    if (stats.requests != 20 || stats.hits != 11 || stats.misses != 9 ||
        stats.bytes_read != 9 * 51 || stats.resident_bytes != 9 * 51 ||
        stats.capacity_bytes != LAYERS * SLOTS_PER_LAYER * 51) {
        fprintf(stderr,
                "prefill pool stats mismatch: requests=%llu hits=%llu "
                "misses=%llu bytes=%llu resident=%llu capacity=%llu\n",
                (unsigned long long)stats.requests,
                (unsigned long long)stats.hits,
                (unsigned long long)stats.misses,
                (unsigned long long)stats.bytes_read,
                (unsigned long long)stats.resident_bytes,
                (unsigned long long)stats.capacity_bytes);
        failed = 1;
    }

    /* Fill the remaining slabs from layer 1, then force four global victims.
     * Layer 0 started with eight pooled residents: its six-expert decode
     * reserve must survive while layer 1 replaces its own older entries. */
    coli_v4_expert_store_prefill_pool(store, 1);
    for (int expert = 0; expert < 7; expert++)
        failed |= lookup_fixture_at(store, 1, expert, EXPERTS);
    coli_v4_expert_store_prefill_pool(store, -1);
    int layer0_resident = 0;
    for (int expert = 0; expert < EXPERTS; expert++)
        if (coli_v4_test_expert_slot_index(
                store, (ColiExpertKey){0, expert}) >= 0)
            layer0_resident++;
    if (layer0_resident != SLOTS_PER_LAYER) {
        fprintf(stderr, "prefill pool lost decode reserve: layer0=%d\n",
                layer0_resident);
        failed = 1;
    }
    store->ops->stats(store, &stats);
    if (stats.requests != 27 || stats.hits != 11 || stats.misses != 16 ||
        stats.bytes_read != 16 * 51 ||
        stats.resident_bytes != LAYERS * SLOTS_PER_LAYER * 51 ||
        stats.capacity_bytes != LAYERS * SLOTS_PER_LAYER * 51) {
        fprintf(stderr,
                "prefill reserve stats mismatch: requests=%llu hits=%llu "
                "misses=%llu bytes=%llu resident=%llu capacity=%llu\n",
                (unsigned long long)stats.requests,
                (unsigned long long)stats.hits,
                (unsigned long long)stats.misses,
                (unsigned long long)stats.bytes_read,
                (unsigned long long)stats.resident_bytes,
                (unsigned long long)stats.capacity_bytes);
        failed = 1;
    }
    store->ops->destroy(store);
    unlink(path);
    rmdir(directory);
    if (failed) return 1;
    puts("DeepSeek-V4 ExpertStore prefill pool: ok "
         "(A/B bytes 816->408, second sweep=0 reads, warm entries + decode reserve retained)");
    return 0;
}

static int run_expert_miss_scaling_case(const char *directory, int experts,
                                        int slots) {
    char error[256];
    ColiDeepSeekV4ExpertStoreOptions options = {
        directory, 1, experts, (uint64_t)slots * 51, -1, UINT64_MAX
    };
    ColiExpertStore *store = NULL;
    if (coli_deepseek_v4_expert_store_open(&options, &store,
                                            error, sizeof(error))) {
        fprintf(stderr, "scaling store open failed: slots=%d error=%s\n",
                slots, error);
        return 1;
    }
    coli_v4_test_expert_victim_probes = 0;
    for (int expert = 0; expert < slots; expert++) {
        ColiExpertView view;
        if (coli_expert_lookup(store, (ColiExpertKey){0, expert}, &view)) {
            fprintf(stderr, "scaling fill failed: slots=%d expert=%d\n",
                    slots, expert);
            store->ops->destroy(store);
            return 1;
        }
        coli_expert_release(store, &view);
    }
    if (coli_v4_test_expert_victim_probes != 0) {
        fprintf(stderr,
                "empty-slot fill searched residents: slots=%d probes=%llu\n",
                slots, (unsigned long long)coli_v4_test_expert_victim_probes);
        store->ops->destroy(store);
        return 1;
    }

    coli_v4_test_expert_victim_probes = 0;
    ColiExpertView view;
    if (coli_expert_lookup(store, (ColiExpertKey){0, experts - 1}, &view)) {
        fprintf(stderr, "scaling eviction failed: slots=%d\n", slots);
        store->ops->destroy(store);
        return 1;
    }
    uint64_t probes = coli_v4_test_expert_victim_probes;
    coli_expert_release(store, &view);
    ColiExpertStoreStats stats;
    store->ops->stats(store, &stats);
    int failed = probes != 1 || stats.requests != (uint64_t)slots + 1 ||
        stats.hits != 0 || stats.misses != (uint64_t)slots + 1 ||
        stats.bytes_read != ((uint64_t)slots + 1) * 51 ||
        stats.resident_bytes != (uint64_t)slots * 51 ||
        stats.capacity_bytes != (uint64_t)slots * 51;
    if (failed)
        fprintf(stderr,
                "miss scaling mismatch: slots=%d probes=%llu requests=%llu "
                "hits=%llu misses=%llu bytes=%llu resident=%llu capacity=%llu\n",
                slots, (unsigned long long)probes,
                (unsigned long long)stats.requests,
                (unsigned long long)stats.hits,
                (unsigned long long)stats.misses,
                (unsigned long long)stats.bytes_read,
                (unsigned long long)stats.resident_bytes,
                (unsigned long long)stats.capacity_bytes);
    store->ops->destroy(store);
    return failed;
}

static int test_expert_store_miss_scaling(void) {
    char directory[] = "colibri-v4-scaling-XXXXXX";
    char path[256];
    setenv("COLI_V4_AUTOPIN", "0", 1);
    setenv("COLI_V4_SAVE_USAGE", "0", 1);
    setenv("COLI_V4_ROWS16", "0", 1);
    if (!mkdtemp(directory)) { perror("mkdtemp scaling"); return 1; }
    snprintf(path, sizeof(path), "%s/model.safetensors", directory);
    if (write_fixture_experts(path, 256)) {
        perror("write scaling fixture");
        rmdir(directory);
        return 1;
    }
    int result = run_expert_miss_scaling_case(directory, 256, 44) ||
                 run_expert_miss_scaling_case(directory, 256, 104) ||
                 run_expert_miss_scaling_case(directory, 256, 208);
    unlink(path);
    rmdir(directory);
    if (result) return 1;
    puts("DeepSeek-V4 ExpertStore miss scaling: ok "
         "(44/104/208 slots=1 probe each)");
    return 0;
}
/* ==== end test_deepseek_v4_expert_store.c ==== */

/* ==== begin test_deepseek_v4_kv_cache.c ==== */
/* umbrella headers */
/* <stdio.h> */
static int test_kv_cache(void) {
    ColiDeepSeekV4KVCache *cache = NULL;
    if (coli_v4_kv_cache_create(&cache, 4, 4, 2, 16) != 0) return 1;
    float value[2];
    for (int position = 0; position < 6; position++) {
        value[0] = (float)position;
        value[1] = (float)-position;
        if (coli_v4_kv_cache_put_window(cache, position, value) < 0) return 1;
        if ((position + 1) % 4 == 0 &&
            coli_v4_kv_cache_put_compressed(cache, position, value) < 4)
            return 1;
    }
    int indices[8];
    int count = coli_v4_kv_cache_indices(cache, 1, indices, 8);
    if (count != 4 || indices[0] != 0 || indices[1] != 1 ||
        indices[2] != -1 || indices[3] != -1) return 1;
    count = coli_v4_kv_cache_indices(cache, 3, indices, 8);
    if (count != 5 || indices[0] != 0 || indices[3] != 3 || indices[4] != 4)
        return 1;
    count = coli_v4_kv_cache_indices(cache, 5, indices, 8);
    if (count != 5 || indices[0] != 2 || indices[1] != 3 ||
        indices[2] != 0 || indices[3] != 1 || indices[4] != 4)
        return 1;
    const float *values = coli_v4_kv_cache_values(cache);
    if (values[0] != 4.0f || values[2] != 5.0f || values[8] != 3.0f)
        return 1;
    coli_v4_kv_cache_reset(cache);
    if (coli_v4_kv_cache_values(cache)[0] != 0.0f) return 1;
    coli_v4_kv_cache_destroy(cache);
    puts("DeepSeek-V4 KV cache tests: ok");
    return 0;
}
/* ==== end test_deepseek_v4_kv_cache.c ==== */

/* ==== begin test_deepseek_v4_layer.c ==== */
/* <assert.h> */
/* <stdio.h> */
/* <string.h> */
/* umbrella headers */
static const ColiDeepSeekV4TensorSpec *find_spec(
        const ColiDeepSeekV4LayerPlan *plan, const char *suffix) {
    for (size_t i = 0; i < plan->tensor_count; i++)
        if (strstr(plan->tensors[i].name, suffix)) return &plan->tensors[i];
    return NULL;
}

static int test_layer(void) {
    ColiDeepSeekV4Config config = {0};
    config.hidden_size = 4096;
    config.num_hidden_layers = 43;
    config.num_attention_heads = 64;
    config.head_dim = 512;
    config.q_lora_rank = 1024;
    config.o_groups = 8;
    config.o_lora_rank = 1024;
    config.index_n_heads = 64;
    config.index_head_dim = 128;
    config.n_routed_experts = 256;
    config.num_experts_per_tok = 6;
    config.n_shared_experts = 1;
    config.moe_intermediate_size = 2048;
    config.num_hash_layers = 3;
    config.hc_mult = 4;
    config.vocab_size = 129280;
    config.compress_ratio_count = 43;
    config.compress_ratios[2] = 4;
    config.compress_ratios[3] = 128;

    char error[256];
    ColiDeepSeekV4LayerPlan plan;
    assert(coli_v4_layer_plan(&plan, &config, 0, error, sizeof(error)) == 0);
    assert(plan.uses_hash_router && !plan.has_compressor && !plan.has_indexer);
    assert(plan.tensor_count == 29);
    const ColiDeepSeekV4TensorSpec *hash = find_spec(&plan, "tid2eid");
    assert(hash && hash->dtype == COLI_ST_I64);
    assert(hash->shape[0] == 129280 && hash->shape[1] == 6);

    assert(coli_v4_layer_plan(&plan, &config, 2, error, sizeof(error)) == 0);
    assert(plan.uses_hash_router && plan.has_compressor && plan.has_indexer);
    assert(plan.tensor_count == 40);
    const ColiDeepSeekV4TensorSpec *index_q = find_spec(&plan, "indexer.wq_b.weight");
    assert(index_q && index_q->shape[0] == 8192 && index_q->shape[1] == 1024);
    const ColiDeepSeekV4TensorSpec *ape = find_spec(&plan, "attn.compressor.ape");
    assert(ape && ape->shape[0] == 4 && ape->shape[1] == 1024);

    assert(coli_v4_layer_plan(&plan, &config, 3, error, sizeof(error)) == 0);
    assert(!plan.uses_hash_router && plan.has_compressor && !plan.has_indexer);
    assert(plan.tensor_count == 33);
    ape = find_spec(&plan, "attn.compressor.ape");
    assert(ape && ape->shape[0] == 128 && ape->shape[1] == 512);
    assert(find_spec(&plan, "ffn.gate.bias") != NULL);

    /* V4 Pro no longer has hidden_size == one output-attention group's
       input width.  Keep the grouped wo_a layout derived from heads/groups. */
    config.hidden_size = 7168;
    config.num_attention_heads = 128;
    config.o_groups = 16;
    assert(coli_v4_layer_plan(&plan, &config, 0, error, sizeof(error)) == 0);
    const ColiDeepSeekV4TensorSpec *wo_a =
        find_spec(&plan, "attn.wo_a.weight");
    const ColiDeepSeekV4TensorSpec *wo_a_scale =
        find_spec(&plan, "attn.wo_a.scale");
    assert(wo_a && wo_a->shape[0] == 16384 && wo_a->shape[1] == 4096);
    assert(wo_a_scale && wo_a_scale->shape[0] == 128 &&
           wo_a_scale->shape[1] == 32);

    puts("deepseek_v4_layer tests passed");
    return 0;
}
/* ==== end test_deepseek_v4_layer.c ==== */

/* ==== begin test_deepseek_v4_math.c ==== */
/* umbrella headers */
/* <math.h> */
/* <stdio.h> */
/* <string.h> */
static int close_enough(float left, float right, float tolerance) {
    return fabsf(left - right) <= tolerance;
}

static int rope_matches_fixed(const float *actual, const float *expected,
                              size_t count, float tolerance) {
    for (size_t index = 0; index < count; index++)
        if (!close_enough(actual[index], expected[index], tolerance))
            return 0;
    return 1;
}

static int test_math(void) {
    enum { HC = 4, DIMENSION = 3, MIXES = (2 + HC) * HC };
    float input[HC * DIMENSION] = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
    };
    float function[MIXES * HC * DIMENSION];
    float base[MIXES];
    float scale[3] = {1, 1, 1};
    float reduced[DIMENSION], post[HC], comb[HC * HC];
    memset(function, 0, sizeof(function));
    memset(base, 0, sizeof(base));
    if (coli_v4_hc_pre(reduced, post, comb, input, function,
                       scale, base, HC, DIMENSION, 20, 1e-6f, 1e-6f) != 0)
        return 1;
    for (int column = 0; column < DIMENSION; column++) {
        float expected = 0.500001f * (input[column] + input[DIMENSION + column] +
                                     input[2 * DIMENSION + column] +
                                     input[3 * DIMENSION + column]);
        if (!close_enough(reduced[column], expected, 2e-5f)) return 1;
    }
    for (int index = 0; index < HC; index++)
        if (!close_enough(post[index], 1.0f, 1e-6f)) return 1;
    for (int row = 0; row < HC; row++) {
        float row_sum = 0.0f, column_sum = 0.0f;
        for (int column = 0; column < HC; column++) {
            row_sum += comb[row * HC + column];
            column_sum += comb[column * HC + row];
        }
        if (!close_enough(row_sum, 1.0f, 1e-5f) ||
            !close_enough(column_sum, 1.0f, 1e-5f)) return 1;
    }
    float branch[DIMENSION] = {0.25f, -0.5f, 0.75f};
    float expanded[HC * DIMENSION];
    if (coli_v4_hc_post(expanded, branch, input, post, comb,
                        HC, DIMENSION) != 0) return 1;
    for (int copy = 0; copy < HC; copy++)
        for (int column = 0; column < DIMENSION; column++) {
            float average = (input[column] + input[DIMENSION + column] +
                             input[2 * DIMENSION + column] +
                             input[3 * DIMENSION + column]) / 4.0f;
            if (!close_enough(expanded[copy * DIMENSION + column],
                              branch[column] + average, 2e-5f)) return 1;
        }

    float norm_weight[DIMENSION] = {1.0f, 1.5f, 0.5f};
    float normalized[DIMENSION];
    if (coli_v4_rmsnorm(normalized, branch, norm_weight,
                        DIMENSION, 1e-6f) != 0) return 1;
    float inverse_rms = 1.0f / sqrtf(
        (branch[0] * branch[0] + branch[1] * branch[1] +
         branch[2] * branch[2]) / DIMENSION + 1e-6f);
    for (int index = 0; index < DIMENSION; index++)
        if (!close_enough(normalized[index],
                          branch[index] * inverse_rms * norm_weight[index],
                          1e-6f)) return 1;

    float rope_cos[12], rope_sin[12];
    float rope_values[8] = {1, 2, 3, 4, -1, 0.5f, 2, -3};
    float rope_original[8];
    memcpy(rope_original, rope_values, sizeof(rope_values));
    if (coli_v4_rope_precompute(rope_cos, rope_sin, 4, 6, 4,
                                10000.0f, 2.0f, 32, 1) != 0 ||
        coli_v4_rope_apply(rope_values, 2, 4,
                           rope_cos + 2, rope_sin + 2, 0) != 0 ||
        coli_v4_rope_apply(rope_values, 2, 4,
                           rope_cos + 2, rope_sin + 2, 1) != 0)
        return 1;
    for (int index = 0; index < 8; index++)
        if (!close_enough(rope_values[index], rope_original[index], 1e-5f)) return 1;

    static const float reference_cos[] = {
        -0.989992499f, 0.999887526f,
        -0.653643608f, 0.999800026f,
    };
    static const float reference_sin[] = {
        0.141120002f, 0.0149994371f,
        -0.756802499f, 0.0199986659f,
    };
    if (!rope_matches_fixed(rope_cos + 3 * 2, reference_cos,
                            sizeof(reference_cos) / sizeof(*reference_cos),
                            2e-6f) ||
        !rope_matches_fixed(rope_sin + 3 * 2, reference_sin,
                            sizeof(reference_sin) / sizeof(*reference_sin),
                            2e-6f))
        return 1;

    float rope_row_cos[2], rope_row_sin[2];
    if (coli_v4_rope_position(rope_row_cos, rope_row_sin, 4, 3, 4,
                              10000.0f, 2.0f, 32, 1) != 0 ||
        !rope_matches_fixed(rope_row_cos, reference_cos, 2, 2e-6f) ||
        !rope_matches_fixed(rope_row_sin, reference_sin, 2, 2e-6f))
        return 1;

    float rope_range_cos[4], rope_range_sin[4];
    if (coli_v4_rope_precompute_range(
            rope_range_cos, rope_range_sin, 4, 3, 2, 4,
            10000.0f, 2.0f, 32, 1) != 0)
        return 1;
    if (!rope_matches_fixed(rope_range_cos, reference_cos,
                            sizeof(reference_cos) / sizeof(*reference_cos),
                            2e-6f) ||
        !rope_matches_fixed(rope_range_sin, reference_sin,
                            sizeof(reference_sin) / sizeof(*reference_sin),
                            2e-6f))
        return 1;

    static const float yarn_reference_cos[] = {
        0.987353623f, 0.979488254f, 0.966957450f, -0.587573767f,
        -0.521824539f, 0.726297259f, -0.576302052f, 0.954964221f,
        -0.988605082f, -0.159120470f, -0.972486019f, -0.394434780f,
        -0.924126625f, -0.807510197f, 0.966177464f, -0.478773415f,
        -0.893718898f, -0.100252181f, 0.523291290f, 0.823492348f,
        0.942630887f, 0.970277667f, 0.984636068f, 0.992067456f,
        0.995906770f, 0.997888565f, 0.998911023f, 0.999438405f,
        0.999710381f, 0.999850631f, 0.999922991f, 0.999960303f,
    };
    static const float yarn_reference_sin[] = {
        -0.158533379f, 0.201501235f, 0.254937738f, 0.809170604f,
        0.853052855f, 0.687380731f, 0.817236781f, 0.296720892f,
        -0.150532320f, 0.987259150f, -0.232961148f, -0.918923914f,
        0.382086396f, -0.589853585f, 0.257878065f, -0.877938509f,
        0.448627412f, 0.994962037f, 0.852153838f, 0.567327321f,
        0.333836794f, 0.241994366f, 0.174619019f, 0.125706837f,
        0.0903861374f, 0.0649493411f, 0.0466561131f, 0.0335097015f,
        0.0240655378f, 0.0172822978f, 0.0124107366f, 0.00891227461f,
    };
    float yarn_cos[32], yarn_sin[32];
    if (coli_v4_rope_position(yarn_cos, yarn_sin, 64, 1024, 4096,
                              40000.0f, 4.0f, 32, 1) != 0 ||
        !rope_matches_fixed(yarn_cos, yarn_reference_cos, 32, 3e-6f) ||
        !rope_matches_fixed(yarn_sin, yarn_reference_sin, 32, 3e-6f))
        return 1;

    if (coli_v4_rope_precompute(rope_cos, rope_sin, 3, 6, 0,
                                10000.0f, 1.0f, 32, 1) == 0)
        return 1;
    if (coli_v4_rope_precompute_range(
            rope_cos, rope_sin, 4, INT_MAX, 2, 0,
            10000.0f, 1.0f, 32, 1) == 0)
        return 1;

    float hidden[2] = {0.5f, -1.0f};
    float router[8] = {1, 0, 0, 1, -1, 0, 0, -1};
    float bias[4] = {10.0f, 0.0f, 9.0f, 0.0f};
    float route_weights[2];
    int route_indices[2];
    if (coli_v4_route(route_weights, route_indices, hidden, router, bias,
                      NULL, 4, 2, 2, 1.5f) != 0) return 1;
    if (route_indices[0] != 0 || route_indices[1] != 2 ||
        !close_enough(route_weights[0] + route_weights[1], 1.5f, 1e-6f))
        return 1;
    int forced[2] = {3, 1};
    if (coli_v4_route(route_weights, route_indices, hidden, router, NULL,
                      forced, 4, 2, 2, 1.5f) != 0 ||
        route_indices[0] != 3 || route_indices[1] != 1) return 1;

    float gates[3] = {-20.0f, 2.0f, 20.0f};
    float ups[3] = {-20.0f, 3.0f, 20.0f};
    float activated[3];
    if (coli_v4_swiglu(activated, gates, ups, 3, 10.0f) != 0) return 1;
    if (!close_enough(activated[0],
                      -20.0f / (1.0f + expf(20.0f)) * -10.0f, 1e-6f) ||
        !close_enough(activated[1],
                      2.0f / (1.0f + expf(-2.0f)) * 3.0f, 1e-6f) ||
        !close_enough(activated[2],
                      10.0f / (1.0f + expf(-10.0f)) * 10.0f, 1e-5f))
        return 1;
    puts("DeepSeek-V4 math tests: ok");
    return 0;
}
/* ==== end test_deepseek_v4_math.c ==== */

/* ==== begin test_deepseek_v4_prompt.c ==== */
/* umbrella headers */
/* <stdio.h> */
/* <stdlib.h> */
/* <string.h> */
static int expect(const char *user, const char *system,
                  ColiDeepSeekV4PromptMode mode, const char *expected) {
    char *actual = NULL;
    size_t length = 0;
    int result = coli_v4_prompt_build(
        &actual, &length, user, system, mode);
    int failed = result || !actual || strcmp(actual, expected) ||
                 length != strlen(expected);
    free(actual);
    return failed;
}

static int test_prompt(void) {
    if (expect("Hello", NULL, COLI_V4_PROMPT_CHAT,
               "<｜begin▁of▁sentence｜><｜User｜>Hello"
               "<｜Assistant｜></think>") ||
        expect("Hello", "Be concise.", COLI_V4_PROMPT_THINKING,
               "<｜begin▁of▁sentence｜>Be concise.<｜User｜>Hello"
               "<｜Assistant｜><think>") ||
        expect("raw", "ignored", COLI_V4_PROMPT_RAW, "raw"))
        return 1;
    puts("DeepSeek V4 prompt tests: ok");
    return 0;
}
/* ==== end test_deepseek_v4_prompt.c ==== */

/* ==== begin test_deepseek_v4_resource_plan.c ==== */
/* umbrella headers */
/* <stdint.h> */
/* <stdio.h> */
#define MIB UINT64_C(1048576)
#define GIB UINT64_C(1073741824)

static ColiDeepSeekV4ResourceInputs fixture(uint64_t available) {
    ColiDeepSeekV4ResourceInputs input = {
        available, 0, 160 * MIB, 600 * MIB, 13369344,
        43, 6, 256,
    };
    return input;
}

static int test_resource_plan(void) {
    char error[256];
    ColiDeepSeekV4ResourcePlan low, high, capped, auto_24, capped_24;
    ColiDeepSeekV4ResourceInputs input = fixture(8 * GIB);
    if (coli_v4_resource_plan_compute(&low, &input, error, sizeof(error)) ||
        low.slots_per_layer < 6 || low.projected_bytes > 8 * GIB)
        return 1;
    input = fixture(64 * GIB);
    if (coli_v4_resource_plan_compute(&high, &input, error, sizeof(error)) ||
        high.slots_per_layer <= low.slots_per_layer ||
        high.projected_bytes > 64 * GIB)
        return 1;
    input = fixture(64 * GIB);
    input.user_limit_bytes = 8 * GIB;
    if (coli_v4_resource_plan_compute(&capped, &input, error, sizeof(error)) ||
        capped.planner_available_bytes != 8 * GIB ||
        capped.system_reserve_bytes != 0 ||
        capped.slots_per_layer < low.slots_per_layer ||
        capped.projected_bytes > 8 * GIB)
        return 1;
    input = fixture(24 * GIB);
    if (coli_v4_resource_plan_compute(&auto_24, &input, error, sizeof(error)))
        return 1;
    input = fixture(64 * GIB);
    input.user_limit_bytes = 24 * GIB;
    if (coli_v4_resource_plan_compute(
            &capped_24, &input, error, sizeof(error)) ||
        capped_24.system_reserve_bytes != 0 ||
        capped_24.slots_per_layer <= auto_24.slots_per_layer ||
        capped_24.projected_bytes > 24 * GIB)
        return 1;
    input = fixture(4 * GIB);
    if (coli_v4_resource_plan_compute(&capped, &input, error, sizeof(error)) == 0)
        return 1;

    ColiDeepSeekV4ResidentTierPlan tiers;
    ColiDeepSeekV4ResidentTierInputs target_only = {
        40 * GIB, 4 * GIB, 24 * GIB, 12 * GIB,
    };
    if (coli_v4_resident_tier_plan(
            &tiers, &target_only, error, sizeof(error)) ||
        !tiers.dense_resident || tiers.dense_bytes != 24 * GIB)
        return 1;
    target_only.available_bytes = 39 * GIB;
    if (coli_v4_resident_tier_plan(
            &tiers, &target_only, error, sizeof(error)) ||
        tiers.dense_resident || tiers.dense_bytes)
        return 1;
    puts("DeepSeek V4 resource plan tests: ok");
    return 0;
}
/* ==== end test_deepseek_v4_resource_plan.c ==== */

/* ==== begin test_deepseek_v4_sparse_attention.c ==== */
/* umbrella headers */
/* shared */
/* <math.h> */
/* <stdio.h> */
static int close_enough_sparse(float left, float right) {
    return fabsf(left - right) <= 1e-6f;
}

static int test_sparse_attention(void) {
    float query[2] = {1, 0};
    float kv[6] = {1, 0, 0, 1, 1, 1};
    float sink[1] = {0};
    int indices[3] = {0, 2, -1};
    float output[2];
    if (coli_v4_sparse_attention_ref(output, query, kv, sink, indices,
                                     1, 2, 3, 3, 1.0f) != 0)
        return 1;
    float denominator = 2.0f + expf(-1.0f);
    if (!close_enough_sparse(output[0], coli_bf16_round(2.0f / denominator)) ||
        !close_enough_sparse(output[1], coli_bf16_round(1.0f / denominator)))
        return 1;
    int invalid[1] = {3};
    if (coli_v4_sparse_attention_ref(output, query, kv, sink, invalid,
                                     1, 2, 3, 1, 1.0f) == 0)
        return 1;
    puts("DeepSeek-V4 sparse attention tests: ok");
    return 0;
}
/* ==== end test_deepseek_v4_sparse_attention.c ==== */

int main(int argc, char **argv) {
    if (test_attention_cache() != 0) {
        fprintf(stderr, "FAIL: test_attention_cache\n");
        return 1;
    }
    if (test_config(argc, argv) != 0) {
        fprintf(stderr, "FAIL: test_config\n");
        return 1;
    }
    if (test_expert() != 0) {
        fprintf(stderr, "FAIL: test_expert\n");
        return 1;
    }
    if (test_expert_batch() != 0) {
        fprintf(stderr, "FAIL: test_expert_batch\n");
        return 1;
    }
    if (test_rows16_convergence() != 0) {
        fprintf(stderr, "FAIL: test_rows16_convergence\n");
        return 1;
    }
    if (test_expert_store() != 0) {
        fprintf(stderr, "FAIL: test_expert_store\n");
        return 1;
    }
    if (test_expert_store_prefill_pool() != 0) {
        fprintf(stderr, "FAIL: test_expert_store_prefill_pool\n");
        return 1;
    }
    if (test_expert_store_miss_scaling() != 0) {
        fprintf(stderr, "FAIL: test_expert_store_miss_scaling\n");
        return 1;
    }
    if (test_kv_cache() != 0) {
        fprintf(stderr, "FAIL: test_kv_cache\n");
        return 1;
    }
    if (test_layer() != 0) {
        fprintf(stderr, "FAIL: test_layer\n");
        return 1;
    }
    if (test_math() != 0) {
        fprintf(stderr, "FAIL: test_math\n");
        return 1;
    }
    if (test_prompt() != 0) {
        fprintf(stderr, "FAIL: test_prompt\n");
        return 1;
    }
    if (test_resource_plan() != 0) {
        fprintf(stderr, "FAIL: test_resource_plan\n");
        return 1;
    }
    if (test_sparse_attention() != 0) {
        fprintf(stderr, "FAIL: test_sparse_attention\n");
        return 1;
    }
    puts("DeepSeek-V4 tests: ok");
    return 0;
}
