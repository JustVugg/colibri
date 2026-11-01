/* Merged DeepSeek V4 unit tests (GLM-style single harness). */
#include "../deepseek_v4_internal.h"
#include "../compat.h"
#include "../mfu_table.h"
#include "../native_quant.h"
#include "../native_quant_batch.h"
#include "../native_quant_dual.h"
#include "../native_quant_fp4_rows16.h"

#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* mkdtemp() hands out a scratch directory, the engine appends <snap>/.coli_usage
 * to it whenever it saves its expert history, and rmdir() then fails on a
 * directory that is no longer empty.  Nothing was listening to that failure, so
 * a knob that did not arrive left the directory behind and the test still said
 * ok.  Removing the fixture and checking the rmdir turns that back into a
 * failure. */
static int scratch_remove(const char *directory, const char *fixture) {
    unlink(fixture);
    if (rmdir(directory) == 0) return 0;
    fprintf(stderr, "scratch directory %s survived its test (errno=%d): the "
                    "engine wrote into it, so the knobs this test sets did not "
                    "reach getenv()\n", directory, errno);
    return 1;
}

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
        directory, 1, 7, 306, -1, 0, 0, 0
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
    if (scratch_remove(directory, path)) return 1;
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
        (uint64_t)LAYERS * SLOTS_PER_LAYER * 51, -1, UINT64_MAX, 0, 0
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
    failed |= scratch_remove(directory, path);
    if (failed) return 1;
    puts("DeepSeek-V4 ExpertStore prefill pool: ok "
         "(A/B bytes 816->408, second sweep=0 reads, warm entries + decode reserve retained)");
    return 0;
}

static int run_expert_miss_scaling_case(const char *directory, int experts,
                                        int slots) {
    char error[256];
    ColiDeepSeekV4ExpertStoreOptions options = {
        directory, 1, experts, (uint64_t)slots * 51, -1, UINT64_MAX, 0, 0
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
    result |= scratch_remove(directory, path);
    if (result) return 1;
    puts("DeepSeek-V4 ExpertStore miss scaling: ok "
         "(44/104/208 slots=1 probe each)");
    return 0;
}

/* REAP-style layout: per-matrix scale/weight pairs so contiguous_group()
 * fails and build_record() sets per_matrix=1. Payload is padded past 4 KiB
 * so an O_DIRECT window has a real aligned pread. */
static int write_per_matrix_fixture(const char *path, int expert_count) {
    static const char *matrix_names[3] = {"w1", "w2", "w3"};
    if (expert_count < 1) return -1;
    size_t header_capacity = 256 + (size_t)expert_count * 1024;
    char *header = malloc(header_capacity);
    size_t payload_size = 8192;
    unsigned char *payload = malloc(payload_size);
    if (!header || !payload) {
        free(payload);
        free(header);
        return -1;
    }
    size_t used = (size_t)snprintf(header, header_capacity, "{");
    size_t cursor = 0;
    for (int expert = 0; expert < expert_count; expert++) {
        for (int matrix = 0; matrix < 3; matrix++) {
            size_t scale = cursor;
            size_t weight = cursor + 1;
            int count = snprintf(
                header + used, header_capacity - used,
                "%s\"layers.0.ffn.experts.%d.%s.scale\":{"
                "\"dtype\":\"F8_E8M0\",\"shape\":[1,1],"
                "\"data_offsets\":[%zu,%zu]}",
                used > 1 ? "," : "", expert, matrix_names[matrix],
                scale, scale + 1);
            if (count < 0 || (size_t)count >= header_capacity - used) {
                free(payload); free(header); return -1;
            }
            used += (size_t)count;
            count = snprintf(
                header + used, header_capacity - used,
                ",\"layers.0.ffn.experts.%d.%s.weight\":{"
                "\"dtype\":\"I8\",\"shape\":[1,16],"
                "\"data_offsets\":[%zu,%zu]}",
                expert, matrix_names[matrix], weight, weight + 16);
            if (count < 0 || (size_t)count >= header_capacity - used) {
                free(payload); free(header); return -1;
            }
            used += (size_t)count;
            cursor += 17;
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

static int expect_per_matrix_expert(const ColiExpertView *view, int expert) {
    int base = expert * 51;
    return view->gate.format == COLI_TENSOR_FP4_NATIVE_BLOCK &&
           view->gate.rows == 1 && view->gate.columns == 32 &&
           view->gate.data_bytes == 16 && view->gate.scale_bytes == 1 &&
           view->down.data_bytes == 16 && view->up.data_bytes == 16 &&
           ((const unsigned char *)view->gate.scales)[0] ==
               (unsigned char)base &&
           ((const unsigned char *)view->down.scales)[0] ==
               (unsigned char)(base + 17) &&
           ((const unsigned char *)view->up.scales)[0] ==
               (unsigned char)(base + 34) &&
           ((const unsigned char *)view->gate.data)[0] ==
               (unsigned char)(base + 1) &&
           ((const unsigned char *)view->down.data)[0] ==
               (unsigned char)(base + 18) &&
           ((const unsigned char *)view->up.data)[0] ==
               (unsigned char)(base + 35);
}

static int test_expert_store_per_matrix_direct(void) {
    /* Would fail this test: per_matrix always increments v4_direct_fallbacks
     * and never calls v4_read_direct_window even when a direct twin exists. */
    char directory[] = "colibri-v4-reap-XXXXXX";
    char path[256], error[256];
    setenv("COLI_V4_AUTOPIN", "0", 1);
    setenv("COLI_V4_SAVE_USAGE", "0", 1);
    setenv("COLI_V4_ROWS16", "0", 1);
    setenv("COLI_V4_PREWARM", "0", 1);
    unsetenv("COLI_V4_DIRECT");
    if (!mkdtemp(directory)) { perror("mkdtemp per_matrix"); return 1; }
    snprintf(path, sizeof(path), "%s/model.safetensors", directory);
    if (write_per_matrix_fixture(path, 2) != 0) {
        perror("write_per_matrix_fixture");
        rmdir(directory);
        return 1;
    }

    ColiDeepSeekV4ExpertStoreOptions options = {
        directory, 1, 2, 102, -1, 0, 0, 0
    };
    ColiExpertStore *store = NULL;
    if (coli_deepseek_v4_expert_store_open(&options, &store,
                                            error, sizeof(error)) != 0) {
        fprintf(stderr, "%s\n", error);
        unlink(path); rmdir(directory);
        return 1;
    }
    if (coli_v4_test_force_streaming_direct(store) != 0) {
        fprintf(stderr, "per_matrix direct: no streaming-direct fd\n");
        store->ops->destroy(store);
        unlink(path); rmdir(directory);
        return 1;
    }
    coli_v4_test_reset_direct_io_stats();
    ColiExpertView view;
    if (coli_expert_lookup(store, (ColiExpertKey){0, 0}, &view) != 0) {
        fprintf(stderr, "per_matrix lookup failed\n");
        store->ops->destroy(store);
        unlink(path); rmdir(directory);
        return 1;
    }
    if (!expect_per_matrix_expert(&view, 0)) {
        fprintf(stderr,
                "per_matrix slab mismatch: scales=%u/%u/%u weights=%u/%u/%u\n",
                ((const unsigned char *)view.gate.scales)[0],
                ((const unsigned char *)view.down.scales)[0],
                ((const unsigned char *)view.up.scales)[0],
                ((const unsigned char *)view.gate.data)[0],
                ((const unsigned char *)view.down.data)[0],
                ((const unsigned char *)view.up.data)[0]);
        coli_expert_release(store, &view);
        store->ops->destroy(store);
        unlink(path); rmdir(directory);
        return 1;
    }
    coli_expert_release(store, &view);
    uint64_t reads = coli_v4_test_direct_reads();
    uint64_t fallbacks = coli_v4_test_direct_fallbacks();
    store->ops->destroy(store);
    if (scratch_remove(directory, path)) return 1;
    if (reads < 1 || fallbacks != 0) {
        fprintf(stderr,
                "per_matrix direct path unused: reads=%llu fallbacks=%llu\n",
                (unsigned long long)reads, (unsigned long long)fallbacks);
        return 1;
    }
    puts("DeepSeek-V4 ExpertStore per_matrix direct: ok");
    return 0;
}

static int prewarm_resident(ColiExpertStore *store, int layers, int experts) {
    int resident = 0;
    for (int layer = 0; layer < layers; layer++)
        for (int expert = 0; expert < experts; expert++)
            if (coli_v4_test_expert_slot_index(
                    store, (ColiExpertKey){layer, expert}) >= 0)
                resident++;
    return resident;
}

/* #2109: the cache filled before the first request. With slots for every expert,
 * all are read and none counts as a request; with fewer, each layer's most used
 * in the history and no never-used one, and the history saved afterwards is the
 * one loaded; with less free RAM than the plan keeps for the system, the fill
 * stops and the experts it left still load on use. */
static int test_expert_store_prewarm(void) {
    enum { LAYERS = 2, EXPERTS = 8, SLOTS = 6, WIDE = 64 };
    char directory[] = "colibri-v4-prewarm-XXXXXX";
    char path[256], history[300], error[256];
    setenv("COLI_V4_AUTOPIN", "0", 1);
    setenv("COLI_V4_SAVE_USAGE", "0", 1);
    setenv("COLI_V4_ROWS16", "0", 1);
    unsetenv("COLI_V4_PREWARM");
    if (!mkdtemp(directory)) { perror("mkdtemp prewarm"); return 1; }
    snprintf(path, sizeof(path), "%s/model.safetensors", directory);
    snprintf(history, sizeof(history), "%s/.coli_usage", directory);
    if (write_fixture_layers(path, LAYERS, EXPERTS)) {
        perror("write prewarm fixture");
        rmdir(directory);
        return 1;
    }
    int failed = 0;
    ColiExpertStore *store = NULL;
    ColiExpertStoreStats stats;

    ColiDeepSeekV4ExpertStoreOptions every = {
        directory, LAYERS, EXPERTS, (uint64_t)LAYERS * EXPERTS * 51, -1, 0, 0, 0
    };
    if (coli_deepseek_v4_expert_store_open(&every, &store, error, sizeof(error))) {
        fprintf(stderr, "prewarm store open failed: %s\n", error);
        failed = 1;
        goto out;
    }
    if (prewarm_resident(store, LAYERS, EXPERTS) != 0) {
        fprintf(stderr, "prewarm: the store opened warm without being asked\n");
        failed = 1;
    }
    if (coli_v4_expert_store_prewarm(store) != 0 ||
        prewarm_resident(store, LAYERS, EXPERTS) != LAYERS * EXPERTS) {
        fprintf(stderr, "prewarm: %d of %d experts resident\n",
                prewarm_resident(store, LAYERS, EXPERTS), LAYERS * EXPERTS);
        failed = 1;
    }
    store->ops->stats(store, &stats);
    if (stats.requests || stats.hits || stats.misses || stats.bytes_read ||
        stats.resident_bytes != (uint64_t)LAYERS * EXPERTS * 51) {
        fprintf(stderr, "prewarm counted as use: requests=%llu hits=%llu misses=%llu "
                "bytes=%llu resident=%llu\n",
                (unsigned long long)stats.requests, (unsigned long long)stats.hits,
                (unsigned long long)stats.misses, (unsigned long long)stats.bytes_read,
                (unsigned long long)stats.resident_bytes);
        failed = 1;
    }
    for (int layer = 0; layer < LAYERS && !failed; layer++)
        for (int expert = 0; expert < EXPERTS && !failed; expert++)
            if (lookup_fixture_at(store, layer, expert, EXPERTS)) failed = 1;
    /* a second fill is a no-op */
    if (!failed && coli_v4_expert_store_prewarm(store) != 0) failed = 1;
    store->ops->stats(store, &stats);
    if (!failed && (stats.requests != LAYERS * EXPERTS ||
                    stats.hits != LAYERS * EXPERTS || stats.misses || stats.bytes_read)) {
        fprintf(stderr, "after prewarm: requests=%llu hits=%llu misses=%llu bytes=%llu\n",
                (unsigned long long)stats.requests, (unsigned long long)stats.hits,
                (unsigned long long)stats.misses, (unsigned long long)stats.bytes_read);
        failed = 1;
    }
    store->ops->destroy(store);
    store = NULL;
    if (failed) goto out;

    /* six slots a layer: layer 0 has seven used experts (1 is the least), layer 1 two */
    static const int used[][3] = {
        {0, 7, 70}, {0, 6, 60}, {0, 5, 50}, {0, 4, 40}, {0, 3, 30}, {0, 2, 20},
        {0, 1, 5}, {1, 3, 9}, {1, 0, 4},
    };
    enum { USED = (int)(sizeof(used) / sizeof(used[0])) };
    FILE *file = fopen(history, "w");
    if (!file) { perror("prewarm history"); failed = 1; goto out; }
    for (int i = 0; i < USED; i++)
        fprintf(file, "%d %d %d\n", used[i][0], used[i][1], used[i][2]);
    fclose(file);
    setenv("COLI_V4_AUTOPIN", "1", 1);
    setenv("COLI_V4_SAVE_USAGE", "1", 1);
    ColiDeepSeekV4ExpertStoreOptions fewer = {
        directory, LAYERS, EXPERTS, (uint64_t)LAYERS * SLOTS * 51, -1, 0, 0, 0
    };
    if (coli_deepseek_v4_expert_store_open(&fewer, &store, error, sizeof(error))) {
        fprintf(stderr, "prewarm history store open failed: %s\n", error);
        failed = 1;
        goto out;
    }
    coli_v4_expert_store_prewarm(store);
    for (int layer = 0; layer < LAYERS; layer++)
        for (int expert = 0; expert < EXPERTS; expert++) {
            int want = layer == 0 ? expert >= 2 : (expert == 0 || expert == 3);
            int have = coli_v4_test_expert_slot_index(
                store, (ColiExpertKey){layer, expert}) >= 0;
            if (have != want) {
                fprintf(stderr, "prewarm from history: layer %d expert %d %s\n", layer,
                        expert, have ? "read but not among the most used"
                                     : "among the most used but not read");
                failed = 1;
            }
        }
    store->ops->stats(store, &stats);
    if (stats.requests || stats.bytes_read) failed = 1;
    store->ops->destroy(store);   /* saves the history */
    store = NULL;
    setenv("COLI_V4_AUTOPIN", "0", 1);
    setenv("COLI_V4_SAVE_USAGE", "0", 1);
    file = fopen(history, "r");
    int rows = 0;
    if (file) {
        int layer, expert;
        unsigned count;
        char line[128];
        while (fgets(line, sizeof(line), file))
            if (sscanf(line, "%d %d %u", &layer, &expert, &count) == 3 && layer >= 0) {
                int known = 0;
                for (int i = 0; i < USED; i++)
                    if (used[i][0] == layer && used[i][1] == expert &&
                        (unsigned)used[i][2] == count) known = 1;
                if (!known) {
                    fprintf(stderr, "prewarm changed the history: %s", line);
                    failed = 1;
                }
                rows++;
            }
        fclose(file);
    }
    if (rows != USED) {
        fprintf(stderr, "prewarm history saved %d rows of %d\n", rows, USED);
        failed = 1;
    }
    unlink(history);
    if (failed) goto out;

    /* free RAM below what the plan keeps: the fill stops, the rest loads on use */
    if (write_fixture_layers(path, LAYERS, WIDE)) {
        perror("write wide prewarm fixture");
        failed = 1;
        goto out;
    }
    ColiDeepSeekV4ExpertStoreOptions short_of_ram = {
        directory, LAYERS, WIDE, (uint64_t)LAYERS * WIDE * 51, -1, 0, 0,
        (uint64_t)1 << 62
    };
    if (coli_deepseek_v4_expert_store_open(&short_of_ram, &store, error, sizeof(error))) {
        fprintf(stderr, "prewarm wide store open failed: %s\n", error);
        failed = 1;
        goto out;
    }
    coli_v4_expert_store_prewarm(store);
    int resident = prewarm_resident(store, LAYERS, WIDE);
    if (resident < 1 || resident >= LAYERS * WIDE) {
        fprintf(stderr, "prewarm short of RAM: %d of %d experts read, want some but "
                "not all\n", resident, LAYERS * WIDE);
        failed = 1;
    }
    for (int layer = 0; layer < LAYERS && !failed; layer++)
        for (int expert = 0; expert < WIDE && !failed; expert++)
            if (lookup_fixture_at(store, layer, expert, WIDE)) failed = 1;

out:
    if (store) store->ops->destroy(store);
    unlink(history);
    if (scratch_remove(directory, path)) return 1;
    if (failed) return 1;
    puts("DeepSeek-V4 ExpertStore prewarm: ok");
    return 0;
}

#if defined(COLI_VULKAN) || defined(COLI_CUDA)
/* S2's primitive, the stream fills' bypass reader: coli_v4_expert_store_read_private
 * must return bit for bit the bytes a lease returns -- whichever loaded an expert, the
 * same bytes reach the device -- while touching none of the store's counters or slots:
 * a fill through it neither fills nor evicts the cache, and a slot a lease holds
 * survives it. The lease path's own counters then still move as always (the
 * counter-assert), which is what lets an arm compare hit rates. */
static int tensor_bytes_equal(const ColiTensorView *a, const ColiTensorView *b) {
    return a->format == b->format && a->rows == b->rows && a->columns == b->columns &&
           a->block_rows == b->block_rows && a->block_columns == b->block_columns &&
           a->data_bytes == b->data_bytes && a->scale_bytes == b->scale_bytes &&
           memcmp(a->data, b->data, a->data_bytes) == 0 &&
           memcmp(a->scales, b->scales, a->scale_bytes) == 0;
}
static int test_expert_store_read_private(void) {
    char directory[] = "colibri-v4-bypass-XXXXXX";
    char path[256], error[256];
    setenv("COLI_V4_AUTOPIN", "0", 1);
    setenv("COLI_V4_SAVE_USAGE", "0", 1);
    setenv("COLI_V4_ROWS16", "0", 1);
    if (!mkdtemp(directory)) { perror("mkdtemp bypass"); return 1; }
    snprintf(path, sizeof(path), "%s/model.safetensors", directory);
    if (write_fixture(path) != 0) { perror("write_fixture"); return 1; }
    ColiDeepSeekV4ExpertStoreOptions options = { directory, 1, 7, 306, -1, 0, 0, 0 };
    ColiExpertStore *store = NULL;
    unsigned char *stage = NULL;
    int failed = 0;
    if (coli_deepseek_v4_expert_store_open(&options, &store,
                                            error, sizeof(error)) != 0) {
        fprintf(stderr, "%s\n", error);
        failed = 1;
        goto out;
    }
    uint64_t record = coli_v4_expert_store_record_bytes(store);
    if (record != 51) {
        fprintf(stderr, "record bytes %llu, want 51\n", (unsigned long long)record);
        failed = 1;
        goto out;
    }
    if (posix_memalign((void **)&stage, 4096, (size_t)record + 8192u)) {
        fprintf(stderr, "no stage buffer\n");
        failed = 1;
        goto out;
    }
    /* the lease's bytes of expert 3, held so its slot cannot be evicted under the test */
    ColiExpertView leased;
    if (coli_expert_lookup(store, (ColiExpertKey){0, 3}, &leased) != 0 ||
        !expect_fixture_expert(&leased, 3)) {
        fprintf(stderr, "lease of expert 3 failed\n");
        failed = 1;
        goto out;
    }
    ColiExpertStoreStats before, after;
    store->ops->stats(store, &before);
    if (before.requests != 1 || before.misses != 1 || before.resident_bytes != 51) {
        fprintf(stderr, "stats before the bypass: requests=%llu misses=%llu resident=%llu\n",
                (unsigned long long)before.requests, (unsigned long long)before.misses,
                (unsigned long long)before.resident_bytes);
        failed = 1;
        goto out;
    }
    for (int expert = 0; expert < 7 && !failed; expert++) {
        ColiExpertView view;
        if (coli_v4_expert_store_read_private(store, (ColiExpertKey){0, expert},
                                              stage, &view) != 0 ||
            !expect_fixture_expert(&view, expert)) {
            fprintf(stderr, "bypass read of expert %d failed\n", expert);
            failed = 1;
            break;
        }
        if (expert == 3 &&   /* bit for bit the lease's bytes, expert 3's own */
            !(tensor_bytes_equal(&view.gate, &leased.gate) &&
              tensor_bytes_equal(&view.up, &leased.up) &&
              tensor_bytes_equal(&view.down, &leased.down))) {
            fprintf(stderr, "bypass bytes differ from the lease's\n");
            failed = 1;
        }
    }
    store->ops->stats(store, &after);
    if (after.requests != before.requests || after.hits != before.hits ||
        after.misses != before.misses || after.bytes_read != before.bytes_read ||
        after.resident_bytes != before.resident_bytes) {
        fprintf(stderr, "the bypass moved the store: requests %llu->%llu hits %llu->%llu "
                "misses %llu->%llu bytes %llu->%llu resident %llu->%llu\n",
                (unsigned long long)before.requests, (unsigned long long)after.requests,
                (unsigned long long)before.hits, (unsigned long long)after.hits,
                (unsigned long long)before.misses, (unsigned long long)after.misses,
                (unsigned long long)before.bytes_read, (unsigned long long)after.bytes_read,
                (unsigned long long)before.resident_bytes, (unsigned long long)after.resident_bytes);
        failed = 1;
    }
    coli_expert_release(store, &leased);
    /* the counter-assert: the lease path still counts as always, the leased slot
     * survived every bypass read (a hit) and a cold lease still misses */
    ColiExpertView again;
    if (coli_expert_lookup(store, (ColiExpertKey){0, 3}, &again) != 0) {
        fprintf(stderr, "re-lease of expert 3 failed\n");
        failed = 1;
    } else coli_expert_release(store, &again);
    ColiExpertView cold;
    if (coli_expert_lookup(store, (ColiExpertKey){0, 4}, &cold) != 0) {
        fprintf(stderr, "lease of expert 4 failed\n");
        failed = 1;
    } else coli_expert_release(store, &cold);
    store->ops->stats(store, &after);
    if (after.requests != 3 || after.hits != 1 || after.misses != 2 ||
        after.resident_bytes != 102) {
        fprintf(stderr, "stats after the re-leases: requests=%llu hits=%llu misses=%llu "
                "resident=%llu (want 3/1/2/102)\n",
                (unsigned long long)after.requests, (unsigned long long)after.hits,
                (unsigned long long)after.misses, (unsigned long long)after.resident_bytes);
        failed = 1;
    }
out:
    compat_aligned_free(stage);
    if (store) store->ops->destroy(store);
    if (scratch_remove(directory, path)) return 1;
    if (failed) return 1;
    puts("DeepSeek-V4 ExpertStore read_private: ok");
    return 0;
}
#endif /* COLI_VULKAN || COLI_CUDA */

/* S7, the engine-store MFU cache (S-store map §5a/§5b/§5c-seed): the
 * decode-only mfu table beside the store. Four behaviours:
 *   hit_rate_d      -- decode_phase(1) lookups count requests_decode/hits_decode,
 *                      prefill-phase lookups do not;
 *   persistence     -- the fork's rule: <model>/.coli_mfu is saved at destroy iff
 *                      a decode selection was recorded, knob or no knob, and a
 *                      run that recorded nothing leaves no file (default-off
 *                      byte-identity for the file, alongside the suite's other
 *                      default-path tests for the policy itself);
 *   the seed        -- COLI_V4_PREWARM=1 ranks each layer by .coli_mfu when it
 *                      has seen the layer (not by the merged .coli_usage), and
 *                      re-stamps LRU age in reverse rank order so the first
 *                      miss eats the coldest seed, not the hottest;
 *   the freeze      -- COLI_V4_FROZEN_PCT rides the pin machinery: a recency
 *                      storm cannot evict the frozen set, and the same storm
 *                      without the knob evicts exactly those experts. */
static int test_expert_store_mfu(void) {
    enum { LAYERS = 2, EXPERTS = 16, SLOTS = 8, NARROW = 6 };
    char directory[] = "colibri-v4-mfu-XXXXXX";
    char path[256], mfu_path[300], usage_path[300], error[256];
    setenv("COLI_V4_AUTOPIN", "0", 1);
    setenv("COLI_V4_SAVE_USAGE", "0", 1);
    setenv("COLI_V4_ROWS16", "0", 1);
    unsetenv("COLI_V4_PREWARM");
    unsetenv("COLI_V4_FROZEN_PCT");
    if (!mkdtemp(directory)) { perror("mkdtemp mfu"); return 1; }
    snprintf(path, sizeof(path), "%s/model.safetensors", directory);
    snprintf(mfu_path, sizeof(mfu_path), "%s/.coli_mfu", directory);
    snprintf(usage_path, sizeof(usage_path), "%s/.coli_usage", directory);
    if (write_fixture_layers(path, LAYERS, EXPERTS)) {
        perror("write mfu fixture");
        rmdir(directory);
        return 1;
    }
    int failed = 0;
    ColiExpertStore *store = NULL;
    ColiExpertStoreStats stats;
    ColiDeepSeekV4ExpertStoreOptions options = {
        directory, LAYERS, EXPERTS, (uint64_t)LAYERS * SLOTS * 51, -1, 0, 0, 0
    };

    /* -- hit_rate_d counts decode lookups only -- */
    if (coli_deepseek_v4_expert_store_open(&options, &store, error, sizeof(error))) {
        fprintf(stderr, "mfu store open failed: %s\n", error);
        failed = 1;
        goto out;
    }
    coli_v4_expert_store_decode_phase(store, 1);
    for (int i = 0; i < 3 && !failed; i++)
        failed = lookup_fixture_at(store, 0, i % 2, EXPERTS);   /* miss, hit, miss */
    coli_v4_expert_store_decode_phase(store, 0);
    for (int i = 0; i < 2 && !failed; i++)
        failed = lookup_fixture_at(store, 0, 2, EXPERTS);       /* miss, hit */
    store->ops->stats(store, &stats);
    if (stats.requests != 5 || stats.hits != 2 ||
        stats.requests_decode != 3 || stats.hits_decode != 1) {
        fprintf(stderr, "decode-only counters: requests=%llu hits=%llu "
                "requests_decode=%llu hits_decode=%llu (want 5/2/3/1)\n",
                (unsigned long long)stats.requests,
                (unsigned long long)stats.hits,
                (unsigned long long)stats.requests_decode,
                (unsigned long long)stats.hits_decode);
        failed = 1;
    }
    store->ops->destroy(store);
    store = NULL;
    /* nothing was recorded: the fork's rule leaves no file behind */
    FILE *leftover = fopen(mfu_path, "r");
    if (leftover) fclose(leftover);
    if (!failed && leftover) {
        fprintf(stderr, "a run with no recorded selection wrote %s\n", mfu_path);
        failed = 1;
    }
    if (failed) goto out;

    /* -- a recorded run persists, knob or no knob, and reloads -- */
    if (coli_deepseek_v4_expert_store_open(&options, &store, error, sizeof(error))) {
        fprintf(stderr, "mfu record store open failed: %s\n", error);
        failed = 1;
        goto out;
    }
    for (int i = 0; i < 3; i++)
        coli_v4_expert_store_mfu_record(store, (ColiExpertKey){0, 0});
    for (int i = 0; i < 2; i++)
        coli_v4_expert_store_mfu_record(store, (ColiExpertKey){0, 5});
    coli_v4_expert_store_mfu_record(store, (ColiExpertKey){1, 7});
    store->ops->destroy(store);
    store = NULL;
    MfuTable table;
    mfu_init(&table);
    int load_result = mfu_load(&table, mfu_path);
    if (load_result != 0 || table.total_accesses != 6 ||
        mfu_count(&table, 0, 0) != 3 || mfu_count(&table, 0, 5) != 2 ||
        mfu_count(&table, 1, 7) != 1) {
        fprintf(stderr, ".coli_mfu round-trip: load=%d total=%llu "
                "(0,0)=%u (0,5)=%u (1,7)=%u\n",
                load_result,
                (unsigned long long)table.total_accesses,
                mfu_count(&table, 0, 0), mfu_count(&table, 0, 5),
                mfu_count(&table, 1, 7));
        failed = 1;
        goto out;
    }

    /* -- the prewarm seed ranks by .coli_mfu, not the merged usage -- */
    unlink(mfu_path);
    mfu_init(&table);
    table.counts[0 * MFU_EXPERTS_PER_LAYER + 2] = 100;
    table.counts[0 * MFU_EXPERTS_PER_LAYER + 5] = 80;
    table.counts[0 * MFU_EXPERTS_PER_LAYER + 9] = 60;
    table.counts[0 * MFU_EXPERTS_PER_LAYER + 12] = 40;
    table.counts[0 * MFU_EXPERTS_PER_LAYER + 7] = 30;
    table.counts[0 * MFU_EXPERTS_PER_LAYER + 1] = 20;
    table.counts[1 * MFU_EXPERTS_PER_LAYER + 1] = 50;
    table.counts[1 * MFU_EXPERTS_PER_LAYER + 4] = 45;
    if (mfu_save(&table, mfu_path)) { perror("mfu_save seed"); failed = 1; goto out; }
    FILE *usage = fopen(usage_path, "w");   /* a usage history that ranks elsewhere */
    if (!usage) { perror("usage seed"); failed = 1; goto out; }
    for (int expert = 0; expert < NARROW; expert++)
        fprintf(usage, "0 %d %d\n1 %d %d\n", expert, 90 - 10 * expert,
                expert, 90 - 10 * expert);
    fclose(usage);
    setenv("COLI_V4_AUTOPIN", "1", 1);
    setenv("COLI_V4_PREWARM", "1", 1);
    ColiDeepSeekV4ExpertStoreOptions narrow = {
        directory, LAYERS, EXPERTS, (uint64_t)LAYERS * NARROW * 51, -1, 0, 0, 0
    };
    if (coli_deepseek_v4_expert_store_open(&narrow, &store, error, sizeof(error))) {
        fprintf(stderr, "mfu seed store open failed: %s\n", error);
        failed = 1;
        goto out;
    }
    store->ops->stats(store, &stats);
    if (stats.requests) {
        fprintf(stderr, "the mfu-seeded prewarm counted a request\n");
        failed = 1;
    }
    for (int layer = 0; layer < LAYERS && !failed; layer++)
        for (int expert = 0; expert < EXPERTS && !failed; expert++) {
            int want = layer == 0
                ? (expert == 1 || expert == 2 || expert == 5 || expert == 7 ||
                   expert == 9 || expert == 12)
                : (expert == 1 || expert == 4);
            int have = coli_v4_test_expert_slot_index(
                store, (ColiExpertKey){layer, expert}) >= 0;
            if (have != want) {
                fprintf(stderr, "mfu-seeded prewarm: layer %d expert %d %s\n",
                        layer, expert,
                        have ? "read but not in the mfu top"
                             : "in the mfu top but not read");
                failed = 1;
            }
        }
    /* the re-stamp: the first miss eats the coldest seed (1), never the
     * hottest (2) -- without it the fill order leaves expert 2 the oldest */
    if (!failed) failed = lookup_fixture_at(store, 0, 0, EXPERTS);
    if (!failed &&
        (coli_v4_test_expert_slot_index(store, (ColiExpertKey){0, 2}) < 0 ||
         coli_v4_test_expert_slot_index(store, (ColiExpertKey){0, 1}) >= 0)) {
        fprintf(stderr, "the seed re-stamp: expert 2 evicted or expert 1 resident "
                "after one cold miss\n");
        failed = 1;
    }
    store->ops->destroy(store);
    store = NULL;
    setenv("COLI_V4_AUTOPIN", "0", 1);
    unsetenv("COLI_V4_PREWARM");
    unlink(usage_path);
    unlink(mfu_path);
    if (failed) goto out;

    /* -- the frozen section survives a recency storm; without the knob the
     * same storm eats exactly those experts -- */
    mfu_init(&table);
    table.counts[0 * MFU_EXPERTS_PER_LAYER + 3] = 100;
    table.counts[0 * MFU_EXPERTS_PER_LAYER + 5] = 50;
    if (mfu_save(&table, mfu_path)) { perror("mfu_save frozen"); failed = 1; goto out; }
    setenv("COLI_V4_FROZEN_PCT", "100", 1);   /* 8 slots -> min(8, 8-6) = 2 frozen */
    if (coli_deepseek_v4_expert_store_open(&options, &store, error, sizeof(error))) {
        fprintf(stderr, "frozen store open failed: %s\n", error);
        failed = 1;
        goto out;
    }
    /* load every expert once (the frozen ones included), then storm the rest */
    for (int expert = 0; expert < EXPERTS && !failed; expert++)
        failed = lookup_fixture_at(store, 0, expert, EXPERTS);
    for (int pass = 0; pass < 2 && !failed; pass++)
        for (int expert = 0; expert < EXPERTS && !failed; expert++)
            if (expert != 3 && expert != 5)
                failed = lookup_fixture_at(store, 0, expert, EXPERTS);
    if (!failed &&
        (coli_v4_test_expert_slot_index(store, (ColiExpertKey){0, 3}) < 0 ||
         coli_v4_test_expert_slot_index(store, (ColiExpertKey){0, 5}) < 0)) {
        fprintf(stderr, "the recency storm evicted a frozen expert\n");
        failed = 1;
    }
    store->ops->destroy(store);
    store = NULL;
    unsetenv("COLI_V4_FROZEN_PCT");
    if (coli_deepseek_v4_expert_store_open(&options, &store, error, sizeof(error))) {
        fprintf(stderr, "unfrozen store open failed: %s\n", error);
        failed = 1;
        goto out;
    }
    for (int expert = 0; expert < EXPERTS && !failed; expert++)
        failed = lookup_fixture_at(store, 0, expert, EXPERTS);
    for (int pass = 0; pass < 2 && !failed; pass++)
        for (int expert = 0; expert < EXPERTS && !failed; expert++)
            if (expert != 3 && expert != 5)
                failed = lookup_fixture_at(store, 0, expert, EXPERTS);
    if (!failed &&
        (coli_v4_test_expert_slot_index(store, (ColiExpertKey){0, 3}) >= 0 ||
         coli_v4_test_expert_slot_index(store, (ColiExpertKey){0, 5}) >= 0)) {
        fprintf(stderr, "without COLI_V4_FROZEN_PCT the storm kept the mfu-hot "
                "experts: the default is not today's policy\n");
        failed = 1;
    }

out:
    if (store) store->ops->destroy(store);
    unsetenv("COLI_V4_FROZEN_PCT");
    unsetenv("COLI_V4_PREWARM");
    unlink(mfu_path);
    unlink(usage_path);
    if (scratch_remove(directory, path)) return 1;
    if (failed) return 1;
    puts("DeepSeek-V4 ExpertStore mfu: ok");
    return 0;
}

/* ==== begin test_deepseek_v4_pilot.c ==== */
/* COLI_VK_TIER_PILOT's lifecycle (the store unit's decode read-ahead): the
 * non-hash fallback fires nothing, a fired prediction lands and the entry's
 * own lookup hits the warm slot byte-identically, the pilot's reads move no
 * demand book (requests/hits/misses stay zero while the slot turns resident),
 * a misprediction costs a slot and nothing else, an entry that comes while
 * the pilot's read is still on disk coalesces onto it (one disk read, exact
 * bytes -- lookup_hot's single flight), and the depth knob clamps. The reads
 * ride a gate hook so the in-flight leg is deterministic. */
static pthread_mutex_t pilot_gate_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t pilot_gate_cond = PTHREAD_COND_INITIALIZER;
static int pilot_gate_entered, pilot_gate_release, pilot_gate_reads;

static void pilot_read_hook(ColiExpertKey key) {
    (void)key;
    pthread_mutex_lock(&pilot_gate_mutex);
    pilot_gate_entered = 1;
    pilot_gate_reads++;
    pthread_cond_broadcast(&pilot_gate_cond);
    while (!pilot_gate_release)
        pthread_cond_wait(&pilot_gate_cond, &pilot_gate_mutex);
    pthread_mutex_unlock(&pilot_gate_mutex);
}

static int pilot_wait_gate(int milliseconds) {
    struct timespec until;
    clock_gettime(CLOCK_REALTIME, &until);
    until.tv_nsec += (long)milliseconds * 1000000L;
    if (until.tv_nsec >= 1000000000L) {
        until.tv_sec += until.tv_nsec / 1000000000L;
        until.tv_nsec %= 1000000000L;
    }
    pthread_mutex_lock(&pilot_gate_mutex);
    int entered = pilot_gate_entered;
    while (!entered && pthread_cond_timedwait(&pilot_gate_cond,
                                              &pilot_gate_mutex, &until) == 0)
        entered = pilot_gate_entered;
    pthread_mutex_unlock(&pilot_gate_mutex);
    return entered;
}

static int pilot_expect_stats(ColiExpertStore *store, unsigned requests,
                              unsigned hits, unsigned misses,
                              unsigned long long resident) {
    ColiExpertStoreStats stats;
    store->ops->stats(store, &stats);
    if (stats.requests != requests || stats.hits != hits ||
        stats.misses != misses || stats.resident_bytes != resident) {
        fprintf(stderr, "pilot stats mismatch: requests=%llu (want %u), hits=%llu "
                        "(want %u), misses=%llu (want %u), resident=%llu (want %llu)\n",
                (unsigned long long)stats.requests, requests,
                (unsigned long long)stats.hits, hits,
                (unsigned long long)stats.misses, misses,
                (unsigned long long)stats.resident_bytes, resident);
        return 1;
    }
    return 0;
}

typedef struct {
    ColiExpertStore *store;
    ColiExpertKey key;
    ColiExpertView view;
    int result;
} PilotEntryJob;

static void *pilot_entry_lookup(void *opaque) {
    PilotEntryJob *job = opaque;
    job->result = coli_expert_lookup(job->store, job->key, &job->view);
    return NULL;
}

static int test_pilot(void) {
    char directory[] = "colibri-v4-pilot-XXXXXX";
    char path[256], error[256];
    int failed = 0;
    setenv("COLI_V4_AUTOPIN", "0", 1);
    setenv("COLI_V4_SAVE_USAGE", "0", 1);
    setenv("COLI_V4_ROWS16", "0", 1);
    setenv("COLI_VK_TIER_PILOT", "1", 1);
    setenv("COLI_VK_CHAIN_DPROF", "1", 1);
    setenv("COLI_VK_TIER_PILOT_DEPTH", "99", 1);
    coli_v4_test_pilot_env_reset();
    if (v4_pilot_depth() != 8) {   /* clamped to the useful span */
        fprintf(stderr, "pilot depth 99 clamped to %d, want 8\n", v4_pilot_depth());
        return 1;
    }
    setenv("COLI_VK_TIER_PILOT_DEPTH", "0", 1);
    coli_v4_test_pilot_env_reset();
    if (v4_pilot_depth() != 1) {
        fprintf(stderr, "pilot depth 0 clamped to %d, want 1\n", v4_pilot_depth());
        return 1;
    }
    setenv("COLI_VK_TIER_PILOT_DEPTH", "1", 1);
    coli_v4_test_pilot_env_reset();
    if (!mkdtemp(directory)) { perror("mkdtemp"); return 1; }
    snprintf(path, sizeof path, "%s/model.safetensors", directory);
    if (write_fixture(path) != 0) { perror("write_fixture"); return 1; }
    ColiDeepSeekV4ExpertStoreOptions options = { directory, 1, 7, 306, -1, 0, 0, 0 };
    ColiExpertStore *store = NULL;
    if (coli_deepseek_v4_expert_store_open(&options, &store, error,
                                           sizeof error) != 0) {
        fprintf(stderr, "%s\n", error);
        return 1;
    }
    /* token 3 of a 7-expert hash layer selects {2, 5}; the other cells keep
     * out-of-range ids, which the fire must skip without reading */
    int64_t table[8 * 2];
    for (int i = 0; i < 16; i++) table[i] = i / 2 == 3 ? (i % 2 ? 5 : 2) : 9;

    /* a non-hash layer (no tid2eid) fires nothing: the fallback, exact */
    coli_v4_test_pilot_fire(store, 0, 3, 2, NULL, 7, 8);
    coli_v4_test_pilot_quiesce();
    if (pilot_expect_stats(store, 0, 0, 0, 0)) failed = 1;

    /* the prediction lands: two warm reads, zero demand books moved, the
     * slots resident -- then the entry's own lookup hits one, byte-identical */
    coli_v4_test_pilot_fire(store, 0, 3, 2, table, 7, 8);
    coli_v4_test_pilot_quiesce();
    if (pilot_expect_stats(store, 0, 0, 0, 102)) failed = 1;
    if (coli_v4_test_pilot_stat(0) != 2 || coli_v4_test_pilot_stat(3) != 2 ||
        coli_v4_test_pilot_stat(4) != 0) {
        fprintf(stderr, "pilot issued/landed/failed = %llu/%llu/%llu, want 2/2/0\n",
                coli_v4_test_pilot_stat(0), coli_v4_test_pilot_stat(3),
                coli_v4_test_pilot_stat(4));
        failed = 1;
    }
    v4_pilot_classify(0, 2);   /* the preload marker's question, at the entry */
    v4_pilot_classify(0, 5);
    v4_pilot_classify(0, 4);   /* never predicted: a cold read as today */
    if (coli_v4_test_pilot_stat(5) != 2 || coli_v4_test_pilot_stat(7) != 1 ||
        coli_v4_test_pilot_stat(6) != 0) {
        fprintf(stderr, "pilot entry split hit/late/miss = %llu/%llu/%llu, want 2/0/1\n",
                coli_v4_test_pilot_stat(5), coli_v4_test_pilot_stat(6),
                coli_v4_test_pilot_stat(7));
        failed = 1;
    }
    ColiExpertView warm;
    if (coli_expert_lookup(store, (ColiExpertKey){0, 2}, &warm) != 0 ||
        !expect_fixture_expert(&warm, 2)) {
        fprintf(stderr, "the entry's read of a landed prediction is not exact\n");
        failed = 1;
    }
    coli_expert_release(store, &warm);
    if (pilot_expect_stats(store, 1, 1, 0, 102)) failed = 1;   /* the demand read counted, as a hit */

    /* a misprediction: token 5's selections warm, nothing ever demands them
     * -- two slots, no book, no byte of output anywhere */
    int64_t mispredict[16];
    for (int i = 0; i < 16; i++) mispredict[i] = i / 2 == 5 ? (i % 2 ? 6 : 1) : 9;
    coli_v4_test_pilot_fire(store, 0, 5, 2, mispredict, 7, 8);
    coli_v4_test_pilot_quiesce();
    if (pilot_expect_stats(store, 1, 1, 0, 204)) failed = 1;

    /* a re-fire of the resident prediction: no read, no books */
    coli_v4_test_pilot_fire(store, 0, 3, 2, table, 7, 8);
    coli_v4_test_pilot_quiesce();
    if (pilot_expect_stats(store, 1, 1, 0, 204)) failed = 1;

    /* an entry that comes while the pilot's read is on disk: one read, exact
     * bytes -- the single flight, never two slabs */
    int64_t late[2] = {4, 4};
    pthread_mutex_lock(&pilot_gate_mutex);
    pilot_gate_entered = pilot_gate_release = pilot_gate_reads = 0;
    pthread_mutex_unlock(&pilot_gate_mutex);
    coli_v4_test_expert_read_hook = pilot_read_hook;
    coli_v4_test_pilot_fire(store, 0, 1, 1, late, 7, 8);
    if (!pilot_wait_gate(5000)) {
        fprintf(stderr, "the pilot's read never reached the disk hook\n");
        failed = 1;
    }
    if (!failed) {
        v4_pilot_classify(0, 4);   /* still READING: the late class */
        PilotEntryJob job = { store, {0, 4}, {{0}}, 0 };
        pthread_t thread;
        if (pthread_create(&thread, NULL, pilot_entry_lookup, &job)) {
            fprintf(stderr, "could not start the entry's lookup\n");
            failed = 1;
        } else {
            struct timespec pause = {0, 100 * 1000000L};
            nanosleep(&pause, NULL);   /* the entry parks on the load, or serves it */
            pthread_mutex_lock(&pilot_gate_mutex);
            pilot_gate_release = 1;
            pthread_cond_broadcast(&pilot_gate_cond);
            pthread_mutex_unlock(&pilot_gate_mutex);
            pthread_join(thread, NULL);
            if (job.result != 0 || !expect_fixture_expert(&job.view, 4)) {
                fprintf(stderr, "the entry's read over an in-flight pilot is not exact\n");
                failed = 1;
            } else
                coli_expert_release(store, &job.view);
            if (pilot_gate_reads != 1) {
                fprintf(stderr, "the coalesced read did not stay single: %d reads\n",
                        pilot_gate_reads);
                failed = 1;
            }
            if (coli_v4_test_pilot_stat(6) != 1) {
                fprintf(stderr, "the in-flight prediction did not classify late\n");
                failed = 1;
            }
        }
    }
    coli_v4_test_expert_read_hook = NULL;
    pthread_mutex_lock(&pilot_gate_mutex);
    pilot_gate_release = 1;
    pthread_cond_broadcast(&pilot_gate_cond);
    pthread_mutex_unlock(&pilot_gate_mutex);
    coli_v4_test_pilot_quiesce();

    /* unset: fire is a no-op, the lever off costs nothing */
    unsetenv("COLI_VK_TIER_PILOT");
    coli_v4_test_pilot_env_reset();
    coli_v4_test_pilot_fire(store, 0, 3, 2, table, 7, 8);
    coli_v4_test_pilot_quiesce();
    if (pilot_expect_stats(store, 2, 2, 0, 255)) failed = 1;   /* only the late leg's demand read since */
    setenv("COLI_VK_TIER_PILOT", "1", 1);

    coli_v4_test_pilot_quiesce();
    store->ops->destroy(store);   /* zero leases left: the drain held */
    unsetenv("COLI_VK_TIER_PILOT");
    unsetenv("COLI_VK_CHAIN_DPROF");
    unsetenv("COLI_VK_TIER_PILOT_DEPTH");
    coli_v4_test_pilot_env_reset();
    unlink(path);
    if (scratch_remove(directory, path)) return 1;
    if (failed) return 1;
    puts("DeepSeek-V4 pilot read-ahead: ok");
    return 0;
}
/* ==== end test_deepseek_v4_pilot.c ==== */

/* The CUDA expert bank's books (coli_v4_expert_store_touch_union and
 * note_union). The bank reads each routed expert once per layer per prompt,
 * uncounted, and hands the store each chunk's routing (the router's ids, every
 * token's top-k) with one touch before its reads and one note once the chunk
 * is done. The same synthetic layer-major routing runs through two stores:
 * "union" looks up every chunk's union in ascending order, as
 * v4_moe_batch_union does; "bank" touches the routing, reads in routing order
 * only the experts its layer has not read yet (later chunks are mostly, some
 * wholly, bank hits) and notes the routing, as coli_v4_gpu_moe_batch_union
 * does. Two chunks per layer fail after the touch and one read and fall back to
 * the CPU union's counted lookups, unnoted, as a refused bank chunk does. After
 * every chunk the books must be the same (usage, HITS, heat, the request clock
 * and the active pins, through the pin ramp, the repins and a usage decay), and
 * so must the saved .coli_usage; the bank store reads only what its own lookups
 * missed, and a touch or a note reads nothing. */
enum {
    BOOKS_LAYERS = 2, BOOKS_EXPERTS = 24, BOOKS_SLOTS = 14, BOOKS_CHUNKS = 48,
    BOOKS_TOKENS = 4, BOOKS_TOPK = 6, BOOKS_ROUTES = BOOKS_TOKENS * BOOKS_TOPK
};

typedef struct {
    uint64_t usage[BOOKS_EXPERTS];
    unsigned char hit[BOOKS_EXPERTS], heat[BOOKS_EXPERTS];
    uint64_t requests;
    int pins[BOOKS_SLOTS];
    int pin_count;
} BooksSnapshot;

static int books_reads;
static void books_read_hook(ColiExpertKey key) { (void)key; books_reads++; }

static int books_has(const int *ids, int count, int expert) {
    for (int i = 0; i < count; i++)
        if (ids[i] == expert) return 1;
    return 0;
}

/* A chunk's routing as the router returns it: each token's top-6, distinct
 * within the token, in no particular order, so experts repeat across tokens.
 * Skewed, with a hot set that drifts every 12 chunks, so the pins change. */
static void books_routing(int layer, int chunk, int ids[BOOKS_ROUTES]) {
    uint32_t x = 2166136261u ^ (uint32_t)(layer * 977 + chunk * 131);
    for (int token = 0; token < BOOKS_TOKENS; token++) {
        int *top = ids + token * BOOKS_TOPK;
        for (int rank = 0; rank < BOOKS_TOPK; rank++) {
            x = x * 1664525u + 1013904223u;
            int draw = (int)((x >> 8) % 100);
            int expert = draw < 60 ? draw % 6 : 6 + draw % (BOOKS_EXPERTS - 6);
            expert = (expert + chunk / 12 * 5) % BOOKS_EXPERTS;
            while (books_has(top, rank, expert))
                expert = (expert + 1) % BOOKS_EXPERTS;
            top[rank] = expert;
        }
    }
}

/* The CPU union's keys: the routing's members, each once, ascending. */
static int books_union_keys(const int *ids, int *keys) {
    unsigned char used[BOOKS_EXPERTS] = {0};
    int count = 0;
    for (int route = 0; route < BOOKS_ROUTES; route++) used[ids[route]] = 1;
    for (int expert = 0; expert < BOOKS_EXPERTS; expert++)
        if (used[expert]) keys[count++] = expert;
    return count;
}

/* The bank's refill: the routed experts it does not hold yet, in routing
 * order, each once, as coli_v4_gpu_moe_batch_union collects `missing`. */
static int books_missing_keys(const int *ids, const unsigned char *in_bank,
                              int *keys) {
    int count = 0;
    for (int route = 0; route < BOOKS_ROUTES; route++)
        if (!in_bank[ids[route]] && !books_has(keys, count, ids[route]))
            keys[count++] = ids[route];
    return count;
}

/* Looks up `layer`'s `keys` in order: counted, as the CPU union's lookups, or
 * uncounted, as the bank's reads, which the bank then holds (in_bank); at most
 * `limit` of them (-1: all). */
static int books_lookups(ColiExpertStore *store, int layer, const int *keys,
                         int key_count, unsigned char *in_bank, int limit,
                         int counted, int *lookups) {
    for (int k = 0; k < key_count && limit; k++) {
        int expert = keys[k];
        ColiExpertKey key = {layer, expert};
        ColiExpertView view;
        int result = counted
            ? coli_expert_lookup(store, key, &view)
            : coli_v4_expert_store_lookup_uncounted(store, key, &view);
        if (result || !expect_fixture_expert_at(&view, layer, expert,
                                                BOOKS_EXPERTS)) {
            fprintf(stderr, "books: %s lookup %d/%d failed\n",
                    counted ? "union" : "bank", layer, expert);
            if (!result) coli_expert_release(store, &view);
            return 1;
        }
        coli_expert_release(store, &view);
        if (in_bank) in_bank[expert] = 1;
        ++*lookups;
        if (limit > 0) limit--;
    }
    return 0;
}

/* bank_only: the bank's chunks served wholly from VRAM (no read, a note). */
static int run_books_case(const char *directory, int bank,
                          BooksSnapshot *snapshots, int *reads, int *bank_only,
                          ColiExpertStoreStats *stats) {
    char error[256];
    ColiDeepSeekV4ExpertStoreOptions options = {
        directory, BOOKS_LAYERS, BOOKS_EXPERTS,
        (uint64_t)BOOKS_LAYERS * BOOKS_SLOTS * 51, -1, 0, 0, 0
    };
    ColiExpertStore *store = NULL;
    if (coli_deepseek_v4_expert_store_open(&options, &store,
                                            error, sizeof(error))) {
        fprintf(stderr, "books store open failed: %s\n", error);
        return 1;
    }
    int failed = 0, lookups = 0;
    books_reads = 0;
    coli_v4_test_expert_read_hook = books_read_hook;
    for (int layer = 0; layer < BOOKS_LAYERS && !failed; layer++) {
        unsigned char in_bank[BOOKS_EXPERTS] = {0};
        for (int chunk = 0; chunk < BOOKS_CHUNKS && !failed; chunk++) {
            int ids[BOOKS_ROUTES], keys[BOOKS_EXPERTS];
            books_routing(layer, chunk, ids);
            int union_count = books_union_keys(ids, keys);
            if (!bank) {
                failed = books_lookups(store, layer, keys, union_count, NULL,
                                       -1, 1, &lookups);
            } else {
                /* Chunks 5 and 30: the bank refuses the chunk after its first
                 * read, and the CPU union runs it, with no note. */
                int refused = chunk == 5 || chunk == 30;
                int missing[BOOKS_EXPERTS];
                int missing_count = books_missing_keys(ids, in_bank, missing);
                if (!refused && !missing_count) *bank_only += 1;
                int before = books_reads;
                coli_v4_expert_store_touch_union(store, layer, ids,
                                                 BOOKS_ROUTES);
                int touch_reads = books_reads - before;
                failed = books_lookups(store, layer, missing, missing_count,
                                       in_bank, refused ? 1 : -1, 0,
                                       &lookups);
                if (!failed && refused)
                    failed = books_lookups(store, layer, keys, union_count,
                                           NULL, -1, 1, &lookups);
                before = books_reads;
                if (!failed && !refused)
                    coli_v4_expert_store_note_union(store, layer, ids,
                                                    BOOKS_ROUTES);
                if (touch_reads || books_reads != before) {
                    fprintf(stderr, "books: touch_union read %d, note_union "
                                    "%d experts\n", touch_reads,
                            books_reads - before);
                    failed = 1;
                }
            }
            BooksSnapshot *snapshot = &snapshots[layer * BOOKS_CHUNKS + chunk];
            memset(snapshot, 0, sizeof(*snapshot));
            snapshot->pin_count = coli_v4_test_expert_books(
                store, layer, snapshot->usage, snapshot->hit, snapshot->heat,
                &snapshot->requests, snapshot->pins);
            if (snapshot->pin_count < 1 || snapshot->pin_count > BOOKS_SLOTS) {
                fprintf(stderr, "books: pin slots=%d\n", snapshot->pin_count);
                failed = 1;
            }
        }
    }
    coli_v4_test_expert_read_hook = NULL;
    store->ops->stats(store, stats);
    *reads = books_reads;
    if (stats->requests != (uint64_t)lookups ||
        stats->misses != (uint64_t)books_reads) {
        fprintf(stderr, "books: %s requests=%llu (lookups %d) misses=%llu "
                        "(reads %d)\n", bank ? "bank" : "union",
                (unsigned long long)stats->requests, lookups,
                (unsigned long long)stats->misses, books_reads);
        failed = 1;
    }
    store->ops->destroy(store);   /* saves <directory>/.coli_usage */
    return failed;
}

static int books_read_file(const char *path, char *buffer, size_t capacity,
                           size_t *length) {
    FILE *file = fopen(path, "rb");
    if (!file) return -1;
    *length = fread(buffer, 1, capacity, file);
    int result = ferror(file) || !feof(file) ? -1 : 0;
    fclose(file);
    return result;
}

static int test_expert_store_bank_books(void) {
    char directories[2][32] = {"colibri-v4-books-u-XXXXXX",
                               "colibri-v4-books-b-XXXXXX"};
    char paths[2][256], histories[2][256];
    static BooksSnapshot snapshots[2][BOOKS_LAYERS * BOOKS_CHUNKS];
    static char saved[2][1 << 16];
    size_t saved_length[2] = {0, 0};
    int reads[2] = {0, 0}, bank_only = 0;
    ColiExpertStoreStats stats[2];
    setenv("COLI_V4_AUTOPIN", "0", 1);    /* no history in, so none carried over */
    setenv("COLI_V4_SAVE_USAGE", "1", 1);
    setenv("COLI_V4_ROWS16", "0", 1);
    setenv("COLI_V4_PREWARM", "0", 1);
    int failed = 0, made[2] = {0, 0};
    for (int bank = 0; bank < 2 && !failed; bank++) {
        if (!mkdtemp(directories[bank])) {
            perror("mkdtemp books");
            failed = 1;
            break;
        }
        made[bank] = 1;
        snprintf(paths[bank], sizeof(paths[bank]), "%s/model.safetensors",
                 directories[bank]);
        snprintf(histories[bank], sizeof(histories[bank]), "%s/.coli_usage",
                 directories[bank]);
        if (write_fixture_layers(paths[bank], BOOKS_LAYERS, BOOKS_EXPERTS)) {
            perror("write books fixture");
            failed = 1;
            break;
        }
        failed = run_books_case(directories[bank], bank, snapshots[bank],
                                &reads[bank], &bank_only, &stats[bank]);
        if (!failed && books_read_file(histories[bank], saved[bank],
                                       sizeof(saved[bank]),
                                       &saved_length[bank])) {
            fprintf(stderr, "books: no saved history %s\n", histories[bank]);
            failed = 1;
        }
    }
    for (int i = 0; !failed && i < BOOKS_LAYERS * BOOKS_CHUNKS; i++) {
        const BooksSnapshot *u = &snapshots[0][i], *b = &snapshots[1][i];
        if (memcmp(u->usage, b->usage, sizeof(u->usage)) ||
            memcmp(u->hit, b->hit, sizeof(u->hit)) ||
            memcmp(u->heat, b->heat, sizeof(u->heat)) ||
            u->requests != b->requests || u->pin_count != b->pin_count ||
            memcmp(u->pins, b->pins, sizeof(u->pins))) {
            fprintf(stderr, "books differ at layer %d chunk %d: requests "
                            "%llu/%llu pins %d,%d,%d,%d/%d,%d,%d,%d\n",
                    i / BOOKS_CHUNKS, i % BOOKS_CHUNKS,
                    (unsigned long long)u->requests,
                    (unsigned long long)b->requests, u->pins[0], u->pins[1],
                    u->pins[2], u->pins[3], b->pins[0], b->pins[1],
                    b->pins[2], b->pins[3]);
            failed = 1;
        }
    }
    /* The case must reach what it claims to cover: the whole pin ramp, the
     * usage decay at 64 repin intervals (384 requests per layer), and chunks
     * the bank serves without a read, which only the note counts. */
    const BooksSnapshot *last = &snapshots[0][BOOKS_CHUNKS - 1];
    int active = 0;
    for (int rank = 0; rank < last->pin_count; rank++)
        active += last->pins[rank] >= 0;
    if (!failed && (active != last->pin_count || last->requests <= 384 ||
                    reads[1] >= reads[0] || bank_only < 1)) {
        fprintf(stderr, "books: case too weak: pins %d/%d requests=%llu "
                        "reads union=%d bank=%d, %d chunks bank only\n",
                active, last->pin_count, (unsigned long long)last->requests,
                reads[0], reads[1], bank_only);
        failed = 1;
    }
    if (!failed && (!saved_length[0] || saved_length[0] != saved_length[1] ||
                    memcmp(saved[0], saved[1], saved_length[0]))) {
        fprintf(stderr, "books: saved histories differ (%zu vs %zu bytes)\n",
                saved_length[0], saved_length[1]);
        failed = 1;
    }
    for (int bank = 0; bank < 2; bank++) {
        if (!made[bank]) continue;
        unlink(histories[bank]);
        failed |= scratch_remove(directories[bank], paths[bank]);
    }
    setenv("COLI_V4_SAVE_USAGE", "0", 1);
    if (failed) return 1;
    printf("DeepSeek-V4 ExpertStore bank books: ok (%d chunk-layers, %d of "
           "them bank only, %d pins, %llu requests/layer, reads union=%d "
           "bank=%d, .coli_usage %zu bytes identical)\n",
           BOOKS_LAYERS * BOOKS_CHUNKS, bank_only, last->pin_count,
           (unsigned long long)last->requests, reads[0], reads[1],
           saved_length[0]);
    return 0;
}

/* touch_union makes the members the RAM cache holds recent, as a hit would,
 * and counts nothing; note_union counts, and touches only what a lookup's
 * counting touches: the resident pins, when a repin fires. Neither moves
 * anything in or out. Another ExpertStore backend (here the plain base store)
 * keeps no books and still serves the uncounted read. */
enum { RECENCY_EXPERTS = 16 };

/* A one-layer hot store of `slots` slots holding experts 0..slots-1, read
 * uncounted in that order (so 0 is the LRU slot). */
static ColiExpertStore *recency_store(const char *directory, int slots) {
    char error[256];
    ColiDeepSeekV4ExpertStoreOptions options = {
        directory, 1, RECENCY_EXPERTS, (uint64_t)slots * 51, -1, 0, 0, 0
    };
    ColiExpertStore *store = NULL;
    if (coli_deepseek_v4_expert_store_open(&options, &store,
                                            error, sizeof(error))) {
        fprintf(stderr, "recency store open failed: %s\n", error);
        return NULL;
    }
    for (int expert = 0; expert < slots; expert++) {
        ColiExpertView view;
        if (coli_v4_expert_store_lookup_uncounted(
                store, (ColiExpertKey){0, expert}, &view)) {
            fprintf(stderr, "recency fill %d failed\n", expert);
            store->ops->destroy(store);
            return NULL;
        }
        coli_expert_release(store, &view);
    }
    return store;
}

static int recency_resident(ColiExpertStore *store, int expert) {
    return coli_v4_test_expert_slot_index(store, (ColiExpertKey){0, expert})
        >= 0;
}

/* An uncounted read of an expert not in RAM: it evicts the LRU victim. */
static int recency_miss(ColiExpertStore *store, int expert) {
    ColiExpertView view;
    if (coli_v4_expert_store_lookup_uncounted(
            store, (ColiExpertKey){0, expert}, &view))
        return 1;
    coli_expert_release(store, &view);
    return 0;
}

static int test_expert_store_union_recency(void) {
    char directory[] = "colibri-v4-note-XXXXXX";
    char path[256], error[256];
    setenv("COLI_V4_AUTOPIN", "0", 1);
    setenv("COLI_V4_SAVE_USAGE", "0", 1);
    setenv("COLI_V4_ROWS16", "0", 1);
    if (!mkdtemp(directory)) { perror("mkdtemp note"); return 1; }
    snprintf(path, sizeof(path), "%s/model.safetensors", directory);
    if (write_fixture_experts(path, RECENCY_EXPERTS)) {
        perror("write note fixture");
        rmdir(directory);
        return 1;
    }
    int failed = 0, pin_count;
    uint64_t usage[RECENCY_EXPERTS], requests = 0;
    unsigned char hit[RECENCY_EXPERTS], heat[RECENCY_EXPERTS];
    int pins[RECENCY_EXPERTS];

    /* Six slots, so no pins: plain LRU. Note 1 and 7, touch 0 and 7 (7 is
     * not in RAM), each routed more than once and 99 outside the layer: the
     * next miss evicts 1, the LRU slot, and keeps 0. */
    ColiExpertStore *store = recency_store(directory, 6);
    if (!store) {
        failed = 1;
    } else {
        const int noted[] = {7, 1, 99, 7, 1}, touched[] = {7, 0, 7, 0};
        books_reads = 0;
        coli_v4_test_expert_read_hook = books_read_hook;
        coli_v4_expert_store_note_union(store, 0, noted, 5);
        coli_v4_expert_store_touch_union(store, 0, touched, 4);
        int union_reads = books_reads;
        int missed = recency_miss(store, 6);
        coli_v4_test_expert_read_hook = NULL;
        pin_count = coli_v4_test_expert_books(store, 0, usage, hit, heat,
                                              &requests, pins);
        if (missed || union_reads || books_reads != 1 ||
            !recency_resident(store, 0) || recency_resident(store, 1) ||
            !recency_resident(store, 2) || !recency_resident(store, 6) ||
            recency_resident(store, 7) || pin_count != 0 || requests != 2 ||
            usage[0] || usage[1] != 1 || usage[6] || usage[7] != 1 ||
            hit[0] || heat[1] != 1 || !hit[7]) {
            fprintf(stderr, "union recency: missed=%d reads=%d/%d resident "
                            "0:%d 1:%d 2:%d 6:%d 7:%d pins=%d requests=%llu "
                            "usage 0:%llu 1:%llu 7:%llu\n", missed,
                    union_reads, books_reads, recency_resident(store, 0),
                    recency_resident(store, 1), recency_resident(store, 2),
                    recency_resident(store, 6), recency_resident(store, 7),
                    pin_count, (unsigned long long)requests,
                    (unsigned long long)usage[0], (unsigned long long)usage[1],
                    (unsigned long long)usage[7]);
            failed = 1;
        }
        store->ops->destroy(store);
    }

    /* Fourteen slots: eight pins, four active at the ramp's start, a repin
     * every six requests. Noting 0..5 fires one: it pins 0..5 by usage (0..3
     * active) and touches them, as a lookup's counting would, so the next
     * miss evicts 6, the oldest slot, not 4 (or 0, uncounted). */
    store = failed ? NULL : recency_store(directory, 14);
    if (!failed && !store) failed = 1;
    if (store) {
        const int noted[] = {5, 3, 1, 0, 2, 4, 3};
        coli_v4_expert_store_note_union(store, 0, noted, 7);
        int missed = recency_miss(store, 14);
        pin_count = coli_v4_test_expert_books(store, 0, usage, hit, heat,
                                              &requests, pins);
        if (missed || pin_count != 8 || requests != 6 || pins[0] != 0 ||
            pins[3] != 3 || pins[4] != -1 || recency_resident(store, 6) ||
            !recency_resident(store, 0) || !recency_resident(store, 4) ||
            !recency_resident(store, 5) || !recency_resident(store, 14)) {
            fprintf(stderr, "union recency, repin: missed=%d pins=%d "
                            "requests=%llu active 0:%d 3:%d 4:%d resident "
                            "0:%d 4:%d 5:%d 6:%d 14:%d\n", missed, pin_count,
                    (unsigned long long)requests, pins[0], pins[3], pins[4],
                    recency_resident(store, 0), recency_resident(store, 4),
                    recency_resident(store, 5), recency_resident(store, 6),
                    recency_resident(store, 14));
            failed = 1;
        }
        store->ops->destroy(store);
    }

    ColiDeepSeekV4ExpertStoreOptions options = {
        directory, 1, RECENCY_EXPERTS, 6 * 51, -1, 0, 0, 0
    };
    ColiExpertStore *base = NULL;
    if (coli_deepseek_v4_expert_store_open_base(&options, &base,
                                                error, sizeof(error))) {
        fprintf(stderr, "recency base store open failed: %s\n", error);
        failed = 1;
    } else {
        const int routed[] = {0, 3};
        ColiExpertView view;
        coli_v4_expert_store_touch_union(base, 0, routed, 2);
        coli_v4_expert_store_note_union(base, 0, routed, 2);
        coli_v4_expert_store_touch_union(NULL, 0, routed, 2);
        coli_v4_expert_store_note_union(NULL, 0, routed, 2);
        if (coli_v4_test_expert_books(base, 0, usage, hit, heat, &requests,
                                      pins) != -1 ||
            coli_v4_expert_store_lookup_uncounted(
                base, (ColiExpertKey){0, 3}, &view) ||
            !expect_fixture_expert(&view, 3)) {
            fprintf(stderr, "union books on the base store: not a plain "
                            "lookup\n");
            failed = 1;
        } else {
            coli_expert_release(base, &view);
        }
        base->ops->destroy(base);
    }
    failed |= scratch_remove(directory, path);
    if (failed) return 1;
    puts("DeepSeek-V4 ExpertStore union touch/note recency: ok (touched "
         "member kept over the LRU, notes touch only repinned residents, no "
         "reads, base store no-op)");
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

static int test_fp8_grouped_projection(void) {
#ifdef __AVX512F__
    enum { GROUPS = 4, ROWS = 128, COLS = 256 };
    uint8_t packed[GROUPS * ROWS * COLS];
    float scales[GROUPS * 2], inputs[GROUPS * COLS];
    float expected[GROUPS * ROWS], actual[GROUPS * ROWS];
    for (int i = 0; i < GROUPS * ROWS * COLS; i++)
        packed[i] = (uint8_t)((i * 17 % 126) | (i % 3 ? 128 : 0));
    for (int i = 0; i < GROUPS * COLS; i++) inputs[i] = (i % 31 - 15) * 0.039f;
    for (int i = 0; i < GROUPS * 2; i++) scales[i] = (i + 1) * 0.027f;
    ColiTensorView weight = {COLI_TENSOR_FP8_E4M3_BLOCK, COLI_SCALE_F32,
        packed, scales, sizeof(packed), sizeof(scales), GROUPS * ROWS, COLS, 8, 128, NULL};
    for (int group = 0; group < GROUPS; group++) {
        ColiTensorView part = weight;
        part.rows = ROWS; part.data = packed + (size_t)group * ROWS * COLS;
        part.scales = scales + group * 2;
        part.data_bytes = ROWS * COLS; part.scale_bytes = 2 * sizeof(float);
        if (coli_fp8_matmul_batch_ref(expected + group * ROWS, &part,
                                       inputs + group * COLS, 1)) return 1;
    }
    if (coli_fp8_grouped_matvec_ref(actual, &weight, inputs, GROUPS) ||
        memcmp(expected, actual, sizeof(expected)) ||
        !coli_fp8_grouped_matvec_ref(actual, &weight, inputs, 3)) return 1;
#endif
    return 0;
}

static int test_route_bf16_parallel_order(void) {
    enum { EXPERTS = 32, DIM = 128, TOPK = 6 };
    float gate[EXPERTS * DIM], hidden[DIM], bias[EXPERTS];
    uint16_t packed[EXPERTS * DIM];
    float expected[TOPK], actual[TOPK]; int expected_ids[TOPK], actual_ids[TOPK];
    int forced[TOPK] = {5, 1, 7, 19, 5, 31};
    for (int i = 0; i < EXPERTS * DIM; i++) {
        gate[i] = (i % 23 - 11) * 0.03125f;
        uint32_t bits; memcpy(&bits, &gate[i], sizeof(bits)); packed[i] = (uint16_t)(bits >> 16);
    }
    for (int i = 0; i < DIM; i++) hidden[i] = (i % 17 - 8) * 0.01371f;
    for (int i = 0; i < EXPERTS; i++) bias[i] = i * 0.015625f;
    for (int mode = 0; mode < 2; mode++) {
        const int *ids = mode ? forced : NULL;
        if (coli_v4_route(expected, expected_ids, hidden, gate, bias, ids,
                EXPERTS, DIM, TOPK, 1.5f) ||
            coli_v4_route_bf16(actual, actual_ids, hidden, packed, bias, ids,
                EXPERTS, DIM, TOPK, 1.5f) ||
            memcmp(expected_ids, actual_ids, sizeof(expected_ids)) ||
            memcmp(expected, actual, sizeof(expected))) return 1;
    }
    return 0;
}

static int test_sparse_attention_parallel_order(void) {
    enum { HEADS = 7, DIM = 32, KV = 19, TOPK = 17 };
    float query[HEADS * DIM], kv[KV * DIM], sinks[HEADS];
    float serial[HEADS * DIM], parallel[HEADS * DIM];
    int indices[TOPK];
    for (int i = 0; i < HEADS * DIM; i++) query[i] = (i % 11 - 5) * 0.125f;
    for (int i = 0; i < KV * DIM; i++) kv[i] = (i % 13 - 6) * 0.0625f;
    for (int i = 0; i < TOPK; i++) indices[i] = i % 5 ? i : -1;
    for (int head = 0; head < HEADS; head++) {
        sinks[head] = head * 0.25f;
        if (coli_v4_sparse_attention_ref(serial + head * DIM, query + head * DIM,
                kv, sinks + head, indices, 1, DIM, KV, TOPK, 0.125f)) return 1;
    }
    if (coli_v4_sparse_attention_ref(parallel, query, kv, sinks, indices,
            HEADS, DIM, KV, TOPK, 0.125f) ||
        memcmp(serial, parallel, sizeof(serial))) return 1;
    indices[0] = KV;
    if (!coli_v4_sparse_attention_ref(parallel, query, kv, sinks, indices,
            HEADS, DIM, KV, TOPK, 0.125f)) return 1;
    for (int i = 0; i < TOPK; i++) indices[i] = -1;
    if (!coli_v4_sparse_attention_ref(parallel, query, kv, sinks, indices,
            HEADS, DIM, KV, TOPK, 0.125f)) return 1;
    return 0;
}

static int test_fp8_decode_batch_order(void) {
    enum { ROWS = 144, COLS = 256 };
    uint8_t packed[ROWS * COLS];
    float scales[4] = {0.037f, 0.13f, 1.71f, 0.0049f};
    float inputs[2 * COLS], single[ROWS], batched[2 * ROWS];
    for (int i = 0; i < ROWS * COLS; i++)
        packed[i] = (uint8_t)((i * 37 % 126) | (i & 1 ? 128 : 0));
    for (int column = 0; column < COLS; column++)
        inputs[column] = inputs[COLS + column] = (column % 19 - 9) * 0.03125f;
    ColiTensorView weight = {
        COLI_TENSOR_FP8_E4M3_BLOCK, COLI_SCALE_F32,
        packed, scales, sizeof(packed), sizeof(scales),
        ROWS, COLS, 8, 128, NULL
    };
    for (int rows = 136; rows <= ROWS; rows += 8) {
        weight.rows = rows; weight.data_bytes = (size_t)rows * COLS;
        if (coli_fp8_matmul_batch_ref(single, &weight, inputs, 1) ||
        coli_fp8_matmul_batch_ref(batched, &weight, inputs, 2) ||
        memcmp(single, batched, (size_t)rows * sizeof(float)) ||
        memcmp(single, batched + rows, (size_t)rows * sizeof(float))) return 1;
    }
    float reference[136], reference_dual[136], other[ROWS];
    float other_scales[4] = {0.21f, 0.007f, 0.91f, 1.03f};
    ColiTensorView second = weight;
    second.scales = other_scales;
    for (int rows = 136; rows <= ROWS; rows += 8) {
        weight.rows = second.rows = rows;
        weight.data_bytes = second.data_bytes = (size_t)rows * COLS;
        if (coli_fp8_matvec_ref(single, &weight, inputs) ||
            coli_fp8_dual_matvec_ref(batched, other, &weight, &second, inputs) ||
            memcmp(single, batched, (size_t)rows * sizeof(float))) return 1;
        if (rows == 136) {
            memcpy(reference, single, sizeof(reference));
            memcpy(reference_dual, other, sizeof(reference_dual));
        } else if (memcmp(reference, single, sizeof(reference)) ||
                   memcmp(reference_dual, other, sizeof(reference_dual))) return 1;
    }
    puts("DeepSeek-V4 FP8 decode: batch-exact bits");
    return 0;
}

#ifdef COLI_VULKAN
#include "../backend_vulkan.h"
#endif

/* `forkdec`: the forked decode path's store-side seam (COLI_VK_V4_DECODE_PATH=fork,
 * vk-tier-check's seventh leg names this case; the alias seam end-to-end is
 * deepseek-v4-tiny-check-forkdec's token-exact oracle). The fixture keeps every
 * record 16-byte aligned (48 B of scales then 48 B of weights an expert) so the
 * registry alias's descriptor offsets align, and ONE slot a layer so a second
 * expert rewrites the first's slot in place -- the fork's LRU rewrite, the exact
 * hazard the per-token alias model must survive (no content-keyed alias persists,
 * every alias reads the slot's CURRENT bytes). Optional argv[2] is the shader path:
 * given and a device comes up, the registry path itself is asserted (slot in the
 * arena, alias geometry + the wide verdict off the slab's own bytes, free(t)
 * lifetime); without it the case asserts the ladder -- slots fall back to malloc,
 * the store serves byte-identical records either way. */
static int forkdec_write_fixture(const char *path) {
    static const char *matrix_names[3] = {"w1", "w2", "w3"};
    const int experts = 8, per_cell = 3264;   /* 3x64B scales + 3x1KB weights; 6 slots of the 8 */
    size_t header_capacity = 8192;
    char *header = malloc(header_capacity);
    size_t payload_size = 16 + (size_t)experts * per_cell;
    unsigned char *payload = malloc(payload_size);
    if (!header || !payload) { free(payload); free(header); return -1; }
    size_t used = (size_t)snprintf(
        header, header_capacity,
        "{\"resident\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}");
    for (int expert = 0; expert < experts; expert++) {
        size_t cell = 16 + (size_t)expert * per_cell;
        for (int matrix = 0; matrix < 3; matrix++) {
            int count = snprintf(
                header + used, header_capacity - used,
                ",\"layers.0.ffn.experts.%d.%s.scale\":{\"dtype\":\"F8_E8M0\",\"shape\":[1,64],"
                "\"data_offsets\":[%zu,%zu]}",
                expert, matrix_names[matrix], cell + (size_t)matrix * 64,
                cell + (size_t)matrix * 64 + 64);
            if (count < 0 || (size_t)count >= header_capacity - used) {
                free(payload); free(header); return -1; }
            used += (size_t)count;
        }
        for (int matrix = 0; matrix < 3; matrix++) {
            int count = snprintf(
                header + used, header_capacity - used,
                ",\"layers.0.ffn.experts.%d.%s.weight\":{\"dtype\":\"I8\",\"shape\":[1,1024],"
                "\"data_offsets\":[%zu,%zu]}",
                expert, matrix_names[matrix], cell + 192 + (size_t)matrix * 1024,
                cell + 192 + (size_t)matrix * 1024 + 1024);
            if (count < 0 || (size_t)count >= header_capacity - used) {
                free(payload); free(header); return -1; }
            used += (size_t)count;
        }
    }
    if (used + 1 >= header_capacity) { free(payload); free(header); return -1; }
    header[used++] = '}';
    header[used] = '\0';
    for (size_t i = 0; i < payload_size; i++) payload[i] = 0;
    for (int expert = 0; expert < experts; expert++) {
        size_t cell = 16 + (size_t)expert * per_cell;
        for (int m = 0; m < 3; m++)
            for (int b = 0; b < 64; b++) {
                /* ue8m0 exponents: 128..135 (narrow) everywhere but expert 1's gate
                 * scale[0] = 145, past the f16 verdict's 140 line (a wide tensor) */
                payload[cell + (size_t)m * 64 + b] =
                    (expert == 1 && m == 0 && b == 0) ? 145
                    : (unsigned char)(128 + ((expert + m) & 7));
            }
        for (int m = 0; m < 3; m++)
            for (int b = 0; b < 1024; b++)
                payload[cell + 192 + (size_t)m * 1024 + b] =
                    (unsigned char)(expert * 3 + m);   /* every weight byte names its (expert, matrix) */
    }
    uint64_t header_length = used;
    int fd = open(path, O_CREAT | O_TRUNC | O_WRONLY | COMPAT_O_BINARY, 0600);
    if (fd < 0) { free(payload); free(header); return -1; }
    int result = write_all(fd, &header_length, sizeof(header_length)) ||
                 write_all(fd, header, (size_t)header_length) ||
                 write_all(fd, payload, payload_size);
    close(fd);
    free(payload);
    free(header);
    return result ? -1 : 0;
}

static int forkdec_expect_view(const ColiExpertView *view, int expert) {
    const unsigned char *gs = view->gate.scales, *gd = view->gate.data,
                        *dd = view->down.data, *ud = view->up.data;
    /* the record's packed layout: scales [0,192), weights [192, 3264) as w1 w2 w3;
     * every weight byte names its (expert, matrix) -- a stale slot's bytes cannot pass */
    return view->gate.format == COLI_TENSOR_FP4_NATIVE_BLOCK &&
           view->gate.rows == 1 && view->gate.columns == 2048 &&
           view->gate.data_bytes == 1024 && view->gate.scale_bytes == 64 &&
           (size_t)(gd - gs) == 192 && (size_t)(dd - gs) == 192 + 1024 &&
           (size_t)(ud - gs) == 192 + 2048 &&
           gs[0] == (unsigned char)(expert == 1 ? 145 : 128 + (expert & 7)) &&
           gd[0] == (unsigned char)(expert * 3) &&
           dd[0] == (unsigned char)(expert * 3 + 1) &&
           ud[0] == (unsigned char)(expert * 3 + 2);
}

static int test_forkdec(const char *spv) {
    char directory[] = "colibri-v4-forkdec-XXXXXX";
    char path[256], error[256];
    int failed = 0, device = 0;
    setenv("COLI_V4_AUTOPIN", "0", 1);
    setenv("COLI_V4_SAVE_USAGE", "0", 1);
    setenv("COLI_VK_TIER_EXPERTS", "1", 1);
    setenv("COLI_VK_V4_DECODE_PATH", "fork", 1);
#ifdef COLI_VULKAN
    if (spv && *spv) device = coli_vk_init(spv);   /* 0: no device -- the ladder path */
#endif
    if (!mkdtemp(directory)) { perror("mkdtemp"); return 1; }
    snprintf(path, sizeof path, "%s/model.safetensors", directory);
    if (forkdec_write_fixture(path) != 0) { perror("forkdec_write_fixture"); return 1; }
    ColiDeepSeekV4ExpertStoreOptions options = { directory, 1, 8, 6 * 3264, -1, 0, 0, 0 };   /* 6 of the 8 experts: victims exist */
    ColiExpertStore *store = NULL;
    if (coli_deepseek_v4_expert_store_open(&options, &store, error, sizeof error)) {
        fprintf(stderr, "%s\n", error);
        return 1;
    }
    ColiExpertView view;
    const unsigned char *slab = NULL;
    /* the round table: 0..5 fill every slot, 6 takes the LRU victim (expert 0's slot,
     * rewritten in place -- the fork's rewrite model), then 1 (a hit, the wide-tensor
     * alias case) and 0 again (another rewrite, back to the first expert's bytes) */
    static const int rounds[] = {0, 1, 2, 3, 4, 5, 6, 1, 0};
    for (unsigned round = 0; round < sizeof rounds / sizeof rounds[0]; round++) {
        int expert = rounds[round];
        if (coli_expert_lookup(store, (ColiExpertKey){0, expert}, &view)) {
            fprintf(stderr, "forkdec: lookup round %d failed\n", round);
            failed = 1;
            break;
        }
        if (!forkdec_expect_view(&view, expert)) {
            fprintf(stderr, "forkdec: round %d served expert %d's bytes wrong (stale slot rewrite?)\n",
                    round, expert);
            failed = 1;
        }
        size_t bytes = 0;
        const unsigned char *s = coli_v4_expert_store_view_slab(store, &view, &bytes);
        if (!s || bytes != 3264) {
            fprintf(stderr, "forkdec: round %u (expert %d) view_slab refused (bytes=%zu)\n",
                    round, expert, bytes);
            failed = 1;
        } else if (round == 0)
            slab = s;   /* expert 0's slot: round 6 (expert 6) must REWRITE this very one */
        else if (round == 6 && s != slab) {
            fprintf(stderr, "forkdec: expert 6 did not rewrite expert 0's slot in place (%p -> %p)\n",
                    (const void *)slab, (const void *)s);
            failed = 1;
        }
#ifdef COLI_VULKAN
        if (device) {
            ColiVkSlabReg *reg = NULL;
            size_t base = (size_t)-1;
            if (!coli_vk_slabreg_lookup(s, &reg, &base) || base != 0) {
                fprintf(stderr, "forkdec: round %d's slab is not the registry's (base=%zd)\n",
                        round, (ssize_t)base);
                failed = 1;
            } else {
                /* the alias seam's geometry, exactly as v4_vk_alias_src fills it: the
                 * three matrices' woff/soff off the slot's own slab */
                static const int woff[3] = {192, 2240, 1216}, soff[3] = {0, 128, 64};   /* w1 w3 w2 */
                for (int m = 0; m < 3; m++) {
                    ColiVkTensor *t = NULL;
                    if (!coli_vk_slabreg_alias(reg, s, (size_t)woff[m], (size_t)soff[m],
                                               15, 2048, 1, 32, -1, &t)) {
                        fprintf(stderr, "forkdec: round %u (expert %d) matrix %d alias refused\n",
                                round, expert, m);
                        failed = 1;
                        continue;
                    }
                    ColiVkTensorInfo info;
                    if (!coli_vk_tensor_info(t, &info) || info.fmt != 15 || info.I != 2048 ||
                        info.O != 1 || info.rowWords != 256) {
                        fprintf(stderr, "forkdec: round %u (expert %d) matrix %d geometry wrong\n",
                                round, expert, m);
                        failed = 1;
                    }
                    /* the wide verdict: the scan of the slab's own scale bytes -- expert 1's
                     * gate (m=0) is the one past 140, every other tensor narrow */
                    int want_wide = (expert == 1 && m == 0);
                    if (info.wide != want_wide) {
                        fprintf(stderr, "forkdec: round %u (expert %d) matrix %d wide=%d, want %d\n",
                                round, expert, m, (int)info.wide, want_wide);
                        failed = 1;
                    }
                    coli_vk_tensor_free(t);   /* free(t): owns nothing, the slab stays */
                }
                /* a registry tensor's free must leave the slot serving (no handle went away) */
                ColiExpertView again;
                if (coli_expert_lookup(store, (ColiExpertKey){0, expert}, &again) ||
                    !forkdec_expect_view(&again, expert)) {
                    fprintf(stderr, "forkdec: round %u (expert %d) re-lookup after alias frees failed\n",
                            round, expert);
                    failed = 1;
                } else
                    coli_expert_release(store, &again);
            }
        }
#endif
        coli_expert_release(store, &view);
    }
#ifdef COLI_VULKAN
    if (!device)
        fprintf(stderr, "forkdec: no Vulkan device -- the malloc ladder served every round\n");
#endif
    store->ops->destroy(store);   /* zero leases: the registry slabs go back here */
#ifdef COLI_VULKAN
    if (device && slab) {   /* the store's destroy returned the slab's buffer: gone */
        ColiVkSlabReg *reg = NULL;
        if (coli_vk_slabreg_lookup(slab, &reg, NULL)) {
            fprintf(stderr, "forkdec: the destroyed store's slab is still registered\n");
            failed = 1;
        }
    }
#endif
    unlink(path);
    rmdir(directory);
    if (failed) return 1;
    puts("DeepSeek-V4 forkdec: registry slots, in-place rewrites, alias lifetime ok");
    return 0;
}

int main(int argc, char **argv) {
    /* `pilot` runs only the read-ahead case: vk-tier-check's fifth leg names it
     * (the pilot lives in this store unit, deepseek_v4.c -- not in VK_OBJ), so
     * the leg needs this binary's one case without the rest of the suite. */
    if (argc > 1 && !strcmp(argv[1], "pilot")) {
        if (test_pilot() != 0) {
            fprintf(stderr, "FAIL: test_pilot\n");
            return 1;
        }
        return 0;
    }
    /* `forkdec` (argv[2] = the shader path, optional): the forked decode path's
     * store seam, vk-tier-check's seventh leg. */
    if (argc > 1 && !strcmp(argv[1], "forkdec")) {
        if (test_forkdec(argc > 2 ? argv[2] : NULL) != 0) {
            fprintf(stderr, "FAIL: test_forkdec\n");
            return 1;
        }
        return 0;
    }
    if (test_fp8_decode_batch_order() || test_sparse_attention_parallel_order() ||
        test_route_bf16_parallel_order() || test_fp8_grouped_projection()) return 1;
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
    if (test_expert_store_per_matrix_direct() != 0) {
        fprintf(stderr, "FAIL: test_expert_store_per_matrix_direct\n");
        return 1;
    }
    if (test_expert_store_prewarm() != 0) {
        fprintf(stderr, "FAIL: test_expert_store_prewarm\n");
        return 1;
    }
    if (test_expert_store_mfu() != 0) {
        fprintf(stderr, "FAIL: test_expert_store_mfu\n");
        return 1;
    }
    if (test_pilot() != 0) {
        fprintf(stderr, "FAIL: test_pilot\n");
        return 1;
    }
#if defined(COLI_VULKAN) || defined(COLI_CUDA)
    if (test_expert_store_read_private() != 0) {
        fprintf(stderr, "FAIL: test_expert_store_read_private\n");
        return 1;
    }
#endif
    if (test_expert_store_bank_books() != 0) {
        fprintf(stderr, "FAIL: test_expert_store_bank_books\n");
        return 1;
    }
    if (test_expert_store_union_recency() != 0) {
        fprintf(stderr, "FAIL: test_expert_store_union_recency\n");
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
