/* Ownership / lifecycle regression tests for DeepSeek V4 engine + session.
 * Built with -DCOLI_V4_TEST_HOOKS against objects under build/ownership/. */
#include "../deepseek_v4_internal.h"
#include "../compat.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* Stubs for symbols referenced by RUNTIME but unused in these tests. */
int coli_v4_expert_store_open_planned(
    ColiV4Engine *engine,
    const ColiDeepSeekV4Config *config,
    const ColiDeepSeekV4ExpertStoreOptions *options,
    ColiExpertStore **store,
    char *error,
    size_t error_size) {
    (void)engine;
    (void)config;
    (void)options;
    (void)store;
    if (error && error_size)
        snprintf(error, error_size, "expert store stub");
    return -1;
}

void coli_v4_layer_resident_reference_free(
    ColiV4Engine *engine, ColiDeepSeekV4LayerWeights *weights) {
    (void)engine;
    (void)weights;
}

static int write_all(int fd, const void *data, size_t length) {
    const unsigned char *bytes = (const unsigned char *)data;
    while (length) {
        ssize_t count = write(fd, bytes, length);
        if (count <= 0) return -1;
        bytes += count;
        length -= (size_t)count;
    }
    return 0;
}

static int write_config(const char *directory) {
    static const char json[] =
        "{\"model_type\":\"deepseek_v4\",\"expert_dtype\":\"fp4\","
        "\"scoring_func\":\"sqrtsoftplus\",\"topk_method\":\"noaux_tc\","
        "\"hidden_size\":128,\"num_hidden_layers\":1,"
        "\"num_attention_heads\":4,\"head_dim\":32,\"q_lora_rank\":64,"
        "\"qk_rope_head_dim\":8,\"o_groups\":2,\"o_lora_rank\":64,"
        "\"sliding_window\":16,\"index_n_heads\":4,\"index_head_dim\":16,"
        "\"index_topk\":8,\"n_routed_experts\":8,\"num_experts_per_tok\":2,"
        "\"n_shared_experts\":1,\"moe_intermediate_size\":32,"
        "\"num_hash_layers\":1,\"num_nextn_predict_layers\":1,"
        "\"hc_mult\":4,\"hc_sinkhorn_iters\":5,\"vocab_size\":256,"
        "\"max_position_embeddings\":4096,\"rms_norm_eps\":1e-6,"
        "\"hc_eps\":1e-6,\"routed_scaling_factor\":1.5,\"swiglu_limit\":10,"
        "\"rope_theta\":10000,\"compress_rope_theta\":40000,"
        "\"compress_ratios\":[0,0],"
        "\"rope_scaling\":{\"original_max_position_embeddings\":1024,"
        "\"beta_fast\":32,\"beta_slow\":1,\"factor\":4},"
        "\"quantization_config\":{\"fmt\":\"e4m3\",\"scale_fmt\":\"ue8m0\"}}";
    char path[512];
    snprintf(path, sizeof(path), "%s/config.json", directory);
    int fd = open(path, O_CREAT | O_TRUNC | O_WRONLY | COMPAT_O_BINARY, 0600);
    if (fd < 0) return -1;
    int result = write_all(fd, json, sizeof(json) - 1);
    close(fd);
    return result;
}

static int write_safetensors(const char *directory) {
    static const char header[] =
        "{\"dense.fp8\":{\"dtype\":\"F8_E4M3\",\"shape\":[2,2],"
        "\"data_offsets\":[0,4]},"
        "\"expert.fp4\":{\"dtype\":\"I8\",\"shape\":[2,1],"
        "\"data_offsets\":[4,6]},"
        "\"expert.scale\":{\"dtype\":\"F8_E8M0\",\"shape\":[2,1],"
        "\"data_offsets\":[6,8]}}";
    static const unsigned char payload[] = {1, 2, 3, 4, 5, 6, 7, 8};
    char path[512];
    snprintf(path, sizeof(path), "%s/model-00001-of-00001.safetensors",
             directory);
    uint64_t header_length = sizeof(header) - 1;
    int fd = open(path, O_CREAT | O_TRUNC | O_WRONLY | COMPAT_O_BINARY, 0600);
    if (fd < 0) return -1;
    int result = write_all(fd, &header_length, sizeof(header_length)) ||
                 write_all(fd, header, (size_t)header_length) ||
                 write_all(fd, payload, sizeof(payload));
    close(fd);
    return result;
}

static int write_tokenizer(const char *directory) {
    static const char json[] =
        "{\"model\":{\"type\":\"BPE\",\"vocab\":{\"a\":0,\"b\":1,\"ab\":2},"
        "\"merges\":[\"a b\"]},"
        "\"added_tokens\":[{\"id\":3,\"content\":\"<eos>\",\"special\":true}]}";
    char path[512];
    snprintf(path, sizeof(path), "%s/tokenizer.json", directory);
    int fd = open(path, O_CREAT | O_TRUNC | O_WRONLY | COMPAT_O_BINARY, 0600);
    if (fd < 0) return -1;
    int result = write_all(fd, json, sizeof(json) - 1);
    close(fd);
    return result;
}

static int make_fixture(char *directory, size_t directory_size) {
    /* Native MinGW binaries do not resolve the MSYS /tmp mount. */
    char template[] = "colibri-v4-own-XXXXXX";
    if (!mkdtemp(template)) return -1;
    if (strlen(template) + 1 > directory_size) return -1;
    memcpy(directory, template, strlen(template) + 1);
    if (write_config(directory) || write_safetensors(directory) ||
        write_tokenizer(directory))
        return -1;
    return 0;
}

static void cleanup_fixture(const char *directory) {
    char path[512];
    snprintf(path, sizeof(path), "%s/config.json", directory);
    unlink(path);
    snprintf(path, sizeof(path), "%s/model-00001-of-00001.safetensors",
             directory);
    unlink(path);
    snprintf(path, sizeof(path), "%s/tokenizer.json", directory);
    unlink(path);
    rmdir(directory);
}

static int test_index_closed_on_expert_fail(void) {
    char directory[128], error[256];
    if (make_fixture(directory, sizeof(directory))) return 1;

    coli_v4_test_fail_expert_store_open = 1;
    coli_v4_test_skip_expert_store_open = 0;
    coli_v4_test_closed_owned_index = 0;

    ColiV4Engine *engine = NULL;
    ColiV4EngineOpenOptions options = {.target_model_dir = directory};
    int rc = coli_v4_engine_open(&engine, &options, error, sizeof(error));
    coli_v4_test_fail_expert_store_open = 0;
    if (rc == 0 || engine) {
        fprintf(stderr, "expected expert-open failure, got success\n");
        coli_v4_engine_destroy(engine);
        cleanup_fixture(directory);
        return 1;
    }
    if (coli_v4_test_closed_owned_index != 1) {
        fprintf(stderr, "expected owned index close once, got %d\n",
                coli_v4_test_closed_owned_index);
        cleanup_fixture(directory);
        return 1;
    }
    cleanup_fixture(directory);
    puts("ownership: index closed after expert-open failure: ok");
    return 0;
}

static int test_engine_owns_model_path(void) {
    char directory[128], error[256];
    if (make_fixture(directory, sizeof(directory))) return 1;

    char *path = strdup(directory);
    if (!path) return 1;

    coli_v4_test_fail_expert_store_open = 0;
    coli_v4_test_skip_expert_store_open = 1;
    ColiV4Engine *engine = NULL;
    ColiV4EngineOpenOptions options = {.target_model_dir = path};
    if (coli_v4_engine_open(&engine, &options, error, sizeof(error))) {
        fprintf(stderr, "engine open failed: %s\n", error);
        free(path);
        cleanup_fixture(directory);
        return 1;
    }
    free(path);

    const char *owned = coli_v4_engine_target_model_dir(engine);
    if (!owned || strcmp(owned, directory) != 0) {
        fprintf(stderr, "engine lost model path after caller free\n");
        coli_v4_engine_destroy(engine);
        cleanup_fixture(directory);
        return 1;
    }
    if (!coli_v4_engine_config(engine) ||
        coli_v4_engine_config(engine)->hidden_size != 128) {
        fprintf(stderr, "engine config unusable after caller free\n");
        coli_v4_engine_destroy(engine);
        cleanup_fixture(directory);
        return 1;
    }
    coli_v4_engine_destroy(engine);
    coli_v4_test_skip_expert_store_open = 0;
    cleanup_fixture(directory);
    puts("ownership: engine copies model path: ok");
    return 0;
}

static int test_session_lifetime_accounting(void) {
    char directory[128], error[256];
    if (make_fixture(directory, sizeof(directory))) return 1;

    coli_v4_test_skip_expert_store_open = 1;
    ColiV4Engine *engine = NULL;
    ColiV4EngineOpenOptions options = {.target_model_dir = directory};
    if (coli_v4_engine_open(&engine, &options, error, sizeof(error))) {
        fprintf(stderr, "engine open failed: %s\n", error);
        cleanup_fixture(directory);
        return 1;
    }

    ColiV4Session *session = coli_v4_test_session_bare_create(engine);
    if (!session) {
        fprintf(stderr, "bare session create failed\n");
        coli_v4_engine_destroy(engine);
        cleanup_fixture(directory);
        return 1;
    }
    if (engine->active_sessions != 1) {
        fprintf(stderr, "expected active_sessions=1 after attach, got %d\n",
                engine->active_sessions);
        coli_v4_test_session_bare_destroy(session);
        coli_v4_engine_destroy(engine);
        cleanup_fixture(directory);
        return 1;
    }

    coli_v4_test_session_bare_destroy(session);
    if (engine->active_sessions != 0) {
        fprintf(stderr, "expected active_sessions=0 after detach, got %d\n",
                engine->active_sessions);
        coli_v4_engine_destroy(engine);
        cleanup_fixture(directory);
        return 1;
    }
    coli_v4_engine_destroy(engine);
    coli_v4_test_skip_expert_store_open = 0;
    cleanup_fixture(directory);
    puts("ownership: session lifetime accounting: ok");
    return 0;
}

static int test_session_tokenizer_freed(void) {
    char directory[128];
    if (make_fixture(directory, sizeof(directory))) return 1;

    char tokenizer_path[512];
    snprintf(tokenizer_path, sizeof(tokenizer_path), "%s/tokenizer.json",
             directory);

    for (int round = 0; round < 2; round++) {
        /* Mirror coli_v4_session_create / coli_v4_session_destroy tokenizer
         * ownership without pulling the full attention/runtime link set. */
        ColiV4Session session;
        memset(&session, 0, sizeof(session));
        tok_load(&session.tokenizer, tokenizer_path);
        session.tokenizer_ready = 1;

        if (!session.tokenizer_ready || !session.tokenizer.json_root ||
            session.tokenizer.n_ids < 4 || !session.tokenizer.merges.e ||
            tok_id_of(&session.tokenizer, "<eos>") != 3 ||
            hm_get(&session.tokenizer.vocab, "ab", 2) != 2) {
            fprintf(stderr, "tokenizer not owned after load\n");
            cleanup_fixture(directory);
            return 1;
        }

        if (session.tokenizer_ready) {
            tok_free(&session.tokenizer);
            session.tokenizer_ready = 0;
        }
        if (session.tokenizer.json_root || session.tokenizer.merges.e ||
            session.tokenizer.vocab.e || session.tokenizer.id2str) {
            fprintf(stderr, "tokenizer not cleared after tok_free\n");
            cleanup_fixture(directory);
            return 1;
        }
    }

    cleanup_fixture(directory);
    puts("ownership: session tokenizer freed across create/destroy: ok");
    return 0;
}

static void restore_setting(const char *name, char *value) {
    if (value) setenv(name, value, 1);
    else unsetenv(name);
    free(value);
}

static int test_incomplete_full_dspark_profile_not_reserved(void) {
    char directory[128], error[256];
    if (make_fixture(directory, sizeof(directory))) return 1;

    const char *draft = getenv("V4_DRAFT");
    const char *mtp = getenv("V4_MTP");
    char *saved_draft = draft ? strdup(draft) : NULL;
    char *saved_mtp = mtp ? strdup(mtp) : NULL;
    if ((draft && !saved_draft) || (mtp && !saved_mtp) ||
        setenv("V4_DRAFT", "1", 1) || setenv("V4_MTP", "1", 1)) {
        restore_setting("V4_DRAFT", saved_draft);
        restore_setting("V4_MTP", saved_mtp);
        cleanup_fixture(directory);
        return 1;
    }

    coli_v4_test_skip_expert_store_open = 1;
    ColiV4Engine *engine = NULL;
    ColiV4EngineOpenOptions options = {.target_model_dir = directory};
    int result = coli_v4_engine_open(&engine, &options, error, sizeof(error));
    uint64_t reserved = engine
        ? engine->runtime.dspark_reserve_bytes : 0;
    int passed = !result && engine &&
                 reserved == 0;
    if (engine) coli_v4_engine_destroy(engine);
    coli_v4_test_skip_expert_store_open = 0;
    restore_setting("V4_DRAFT", saved_draft);
    restore_setting("V4_MTP", saved_mtp);
    cleanup_fixture(directory);
    if (!passed) {
        fprintf(stderr,
                "incomplete DSpark profile reserved %llu bytes: %s\n",
                (unsigned long long)reserved,
                error);
        return 1;
    }
    puts("ownership: incomplete full DSpark profile reserves no cache: ok");
    return 0;
}

int main(void) {
    if (test_index_closed_on_expert_fail()) return 1;
    if (test_engine_owns_model_path()) return 1;
    if (test_session_lifetime_accounting()) return 1;
    if (test_session_tokenizer_freed()) return 1;
    if (test_incomplete_full_dspark_profile_not_reserved()) return 1;
    puts("DeepSeek-V4 ownership tests: ok");
    return 0;
}
