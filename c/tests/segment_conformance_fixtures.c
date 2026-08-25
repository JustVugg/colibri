#include "segment_conformance_fixtures.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FIXTURE_MAX_STATE_LANES 8u
#define FIXTURE_SNAPSHOT_SIZE 80u
#define FIXTURE_SNAPSHOT_VERSION 1u

typedef struct {
    const ColiSegmentConformanceFixture *fixture;
    uint32_t layer_begin;
    uint32_t layer_end;
    uint32_t context_tokens;
} FixtureEngine;

typedef struct {
    FixtureEngine *engine;
    uint64_t position;
    uint32_t state[FIXTURE_MAX_STATE_LANES];
} FixtureSession;

typedef struct {
    unsigned char bytes[FIXTURE_SNAPSHOT_SIZE];
    size_t length;
    size_t offset;
} FixtureStream;

/* One row per public model family in family_registry.py. The schemas use a
 * fixture/ namespace: they describe the contract state exercised here, never
 * claim wire compatibility with a future real adapter. tiny_generator records
 * the existing real-model oracle that adapter PRs must reuse. OLMoE currently
 * has a real-checkpoint oracle rather than a generated miniature; keeping that
 * fact explicit is preferable to silently labelling a synthetic checkpoint as
 * model coverage. */
static const ColiSegmentConformanceFixture g_fixtures[] = {
    {
        "glm", "GLM-5.2", "fixture/glm-mla-dsa-v1",
        "MLA latent cache + RoPE + DSA indexer + device cache",
        "tools/make_glm_oracle.py",
        COLI_SEGMENT_FIXTURE_MLA_LATENT |
            COLI_SEGMENT_FIXTURE_DSA_INDEXER |
            COLI_SEGMENT_FIXTURE_DEVICE_CACHE,
        4, 8, 5, 64, UINT32_C(0x474c4d35),
    },
    {
        "inkling", "Inkling", "fixture/inkling-kv-ring-conv-v1",
        "global KV + sliding KV ring + convolutional state",
        "tools/make_tiny_inkling.py",
        COLI_SEGMENT_FIXTURE_KV |
            COLI_SEGMENT_FIXTURE_SLIDING_RING |
            COLI_SEGMENT_FIXTURE_CONVOLUTION,
        3, 7, 8, 64, UINT32_C(0x494e4b4c),
    },
    {
        "kimi", "Kimi K3", "fixture/kimi-mla-kda-conv-attnres-v1",
        "MLA + KDA recurrent state + convolution windows + AttnRes",
        "tools/make_kimi_k3_tiny.py",
        COLI_SEGMENT_FIXTURE_MLA_LATENT |
            COLI_SEGMENT_FIXTURE_RECURRENT |
            COLI_SEGMENT_FIXTURE_CONVOLUTION |
            COLI_SEGMENT_FIXTURE_ATTN_RESIDUAL,
        4, 9, 4, 64, UINT32_C(0x4b494d49),
    },
    {
        "olmoe", "OLMoE", "fixture/olmoe-kv-v1",
        "conventional key/value attention cache",
        "tools/make_olmoe_real_oracle.py",
        COLI_SEGMENT_FIXTURE_KV,
        1, 6, 4, 32, UINT32_C(0x4f4c4d4f),
    },
    {
        "qwen36", "Qwen3.6", "fixture/qwen36-kv-deltanet-conv-v1",
        "attention KV + DeltaNet recurrent state + convolution ring",
        "tools/make_qwen36_tiny.py",
        COLI_SEGMENT_FIXTURE_KV |
            COLI_SEGMENT_FIXTURE_RECURRENT |
            COLI_SEGMENT_FIXTURE_CONVOLUTION |
            COLI_SEGMENT_FIXTURE_SLIDING_RING,
        4, 8, 8, 64, UINT32_C(0x5157454e),
    },
    {
        "deepseek_v4", "DeepSeek V4", "fixture/dsv4-mhc-compressed-v1",
        "mHC + window/compressed attention + compressor + indexer",
        "tools/make_deepseek_v4_tiny.py",
        COLI_SEGMENT_FIXTURE_MHC |
            COLI_SEGMENT_FIXTURE_SLIDING_RING |
            COLI_SEGMENT_FIXTURE_COMPRESSOR |
            COLI_SEGMENT_FIXTURE_DSA_INDEXER |
            COLI_SEGMENT_FIXTURE_DEVICE_CACHE,
        5, 10, 4, 64, UINT32_C(0x44535634),
    },
};

static int fail(char *error, size_t error_size, const char *message) {
    if (error && error_size) snprintf(error, error_size, "%s", message);
    return -1;
}

static int fixture_index(const ColiSegmentConformanceFixture *fixture) {
    for (size_t i = 0; i < sizeof(g_fixtures) / sizeof(g_fixtures[0]); i++)
        if (fixture == &g_fixtures[i]) return (int)i;
    return -1;
}

size_t coli_segment_conformance_fixture_count(void) {
    return sizeof(g_fixtures) / sizeof(g_fixtures[0]);
}

const ColiSegmentConformanceFixture *
coli_segment_conformance_fixture_at(size_t index) {
    if (index >= coli_segment_conformance_fixture_count()) return NULL;
    return &g_fixtures[index];
}

static void put_u32le(unsigned char *p, uint32_t value) {
    for (unsigned i = 0; i < 4; i++) p[i] = (unsigned char)(value >> (8u * i));
}

static void put_u64le(unsigned char *p, uint64_t value) {
    for (unsigned i = 0; i < 8; i++) p[i] = (unsigned char)(value >> (8u * i));
}

static uint32_t get_u32le(const unsigned char *p) {
    uint32_t value = 0;
    for (unsigned i = 0; i < 4; i++) value |= (uint32_t)p[i] << (8u * i);
    return value;
}

static uint64_t get_u64le(const unsigned char *p) {
    uint64_t value = 0;
    for (unsigned i = 0; i < 8; i++) value |= (uint64_t)p[i] << (8u * i);
    return value;
}

static uint32_t checksum32(const unsigned char *data, size_t size) {
    uint32_t hash = UINT32_C(2166136261);
    for (size_t i = 0; i < size; i++) {
        hash ^= data[i];
        hash *= UINT32_C(16777619);
    }
    return hash;
}

static uint32_t float_bits(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static int open_fixture(const ColiSegmentConformanceFixture *fixture,
                        void **engine_impl,
                        ColiSegmentCapabilities *capabilities,
                        const ColiSegmentEngineOptions *options,
                        char *error, size_t error_size) {
    if (!fixture || fixture_index(fixture) < 0 ||
        options->layer_end > fixture->num_layers)
        return fail(error, error_size, "fixture range is outside the model");
    FixtureEngine *engine = (FixtureEngine *)calloc(1, sizeof(*engine));
    if (!engine) return fail(error, error_size, "fixture engine allocation failed");
    engine->fixture = fixture;
    engine->layer_begin = options->layer_begin;
    engine->layer_end = options->layer_end;
    engine->context_tokens = options->context_tokens;

    capabilities->abi_version = COLI_SEGMENT_ABI_VERSION;
    capabilities->flags = COLI_SEGMENT_CAP_TOKEN_IDS |
                          COLI_SEGMENT_CAP_SNAPSHOT |
                          COLI_SEGMENT_CAP_RANGE_NATIVE |
                          COLI_SEGMENT_CAP_MULTI_SESSION |
                          COLI_SEGMENT_CAP_CPU;
    snprintf(capabilities->engine_id, sizeof(capabilities->engine_id), "%s",
             fixture->family_id);
    snprintf(capabilities->state_schema, sizeof(capabilities->state_schema),
             "%s", fixture->state_schema);
    snprintf(capabilities->numeric_class,
             sizeof(capabilities->numeric_class),
             "fixture-f32-deterministic-v1");
    capabilities->state_dtype = COLI_SEGMENT_DTYPE_F32;
    capabilities->state_width = fixture->state_width;
    capabilities->max_batch_rows = 4;
    capabilities->max_context_tokens = fixture->max_context_tokens;
    capabilities->num_layers = fixture->num_layers;
    *engine_impl = engine;
    return 0;
}

static void fixture_engine_destroy(void *engine_impl) { free(engine_impl); }

static int fixture_session_create(void *engine_impl, void **session_impl,
                                  const ColiSegmentSessionOptions *options,
                                  char *error, size_t error_size) {
    FixtureEngine *engine = (FixtureEngine *)engine_impl;
    if (!engine || options->context_tokens > engine->context_tokens)
        return fail(error, error_size, "fixture session context is too large");
    FixtureSession *session = (FixtureSession *)calloc(1, sizeof(*session));
    if (!session) return fail(error, error_size, "fixture session allocation failed");
    session->engine = engine;
    *session_impl = session;
    return 0;
}

static void fixture_session_destroy(void *session_impl) { free(session_impl); }

static uint32_t mix_state(uint32_t old, uint32_t input, uint32_t token,
                          uint64_t position, uint32_t lane,
                          const FixtureEngine *engine) {
    uint32_t mixed = old ^ input ^ token ^ (uint32_t)position;
    mixed ^= engine->fixture->fixture_tag + UINT32_C(0x9e3779b9) * (lane + 1);
    mixed ^= engine->layer_begin * UINT32_C(0x045d9f3b);
    mixed ^= engine->layer_end * UINT32_C(0x119de1f3);
    mixed ^= mixed >> 16;
    mixed *= UINT32_C(0x7feb352d);
    mixed ^= mixed >> 15;
    return mixed;
}

static int fixture_session_run(void *session_impl,
                               const ColiSegmentRunRequest *request,
                               char *error, size_t error_size) {
    FixtureSession *session = (FixtureSession *)session_impl;
    const ColiSegmentConformanceFixture *fixture = session->engine->fixture;
    if (request->position != session->position)
        return fail(error, error_size, "fixture run position is not contiguous");

    const float *input = (const float *)request->input;
    float *output = (float *)request->output;
    for (uint32_t row = 0; row < request->rows; row++) {
        uint32_t token = (uint32_t)request->token_ids[row];
        uint64_t position = session->position + row;
        for (uint32_t lane = 0; lane < fixture->state_lanes; lane++) {
            size_t input_index = (size_t)row * fixture->state_width +
                                 lane % fixture->state_width;
            session->state[lane] = mix_state(
                session->state[lane], float_bits(input[input_index]), token,
                position, lane, session->engine);
        }
        for (uint32_t column = 0; column < fixture->state_width; column++) {
            uint32_t state = session->state[column % fixture->state_lanes];
            uint32_t folded = (state >> 8) ^ state;
            float delta = (float)(folded & UINT32_C(0x3ff)) / 1024.0f;
            delta += (float)(session->engine->layer_end -
                             session->engine->layer_begin) / 32.0f;
            output[(size_t)row * fixture->state_width + column] =
                input[(size_t)row * fixture->state_width + column] + delta;
        }
    }
    session->position += request->rows;
    return 0;
}

static void encode_snapshot(const FixtureSession *session,
                            unsigned char bytes[FIXTURE_SNAPSHOT_SIZE]) {
    memset(bytes, 0, FIXTURE_SNAPSHOT_SIZE);
    memcpy(bytes, "CSFX", 4);
    put_u32le(bytes + 4, FIXTURE_SNAPSHOT_VERSION);
    put_u32le(bytes + 8, session->engine->fixture->fixture_tag);
    put_u32le(bytes + 12, session->engine->layer_begin);
    put_u32le(bytes + 16, session->engine->layer_end);
    put_u32le(bytes + 20, session->engine->context_tokens);
    put_u32le(bytes + 24, session->engine->fixture->state_lanes);
    put_u32le(bytes + 28, session->engine->fixture->state_kinds);
    put_u64le(bytes + 32, session->position);
    for (uint32_t i = 0; i < FIXTURE_MAX_STATE_LANES; i++)
        put_u32le(bytes + 40 + (size_t)i * 4, session->state[i]);
    put_u32le(bytes + 72, checksum32(bytes, 72));
    put_u32le(bytes + 76, UINT32_C(0x53454731));
}

static int decode_snapshot(FixtureSession *session,
                           const unsigned char bytes[FIXTURE_SNAPSHOT_SIZE],
                           char *error, size_t error_size) {
    FixtureEngine *engine = session->engine;
    const ColiSegmentConformanceFixture *fixture = engine->fixture;
    uint64_t position = get_u64le(bytes + 32);
    if (memcmp(bytes, "CSFX", 4) != 0 ||
        get_u32le(bytes + 4) != FIXTURE_SNAPSHOT_VERSION ||
        get_u32le(bytes + 8) != fixture->fixture_tag ||
        get_u32le(bytes + 12) != engine->layer_begin ||
        get_u32le(bytes + 16) != engine->layer_end ||
        get_u32le(bytes + 20) != engine->context_tokens ||
        get_u32le(bytes + 24) != fixture->state_lanes ||
        get_u32le(bytes + 28) != fixture->state_kinds ||
        get_u32le(bytes + 72) != checksum32(bytes, 72) ||
        get_u32le(bytes + 76) != UINT32_C(0x53454731) ||
        position > engine->context_tokens)
        return fail(error, error_size, "fixture snapshot identity or checksum mismatch");

    /* Validate everything before mutating the live session: failed restores
     * are transactional and leave the previous state usable. */
    uint32_t restored[FIXTURE_MAX_STATE_LANES];
    for (uint32_t i = 0; i < FIXTURE_MAX_STATE_LANES; i++)
        restored[i] = get_u32le(bytes + 40 + (size_t)i * 4);
    session->position = position;
    memcpy(session->state, restored, sizeof(restored));
    return 0;
}

static int fixture_snapshot(void *session_impl, ColiSegmentWriteFn write_fn,
                            void *write_user_data,
                            char *error, size_t error_size) {
    FixtureSession *session = (FixtureSession *)session_impl;
    unsigned char bytes[FIXTURE_SNAPSHOT_SIZE];
    encode_snapshot(session, bytes);
    /* Multiple writes are intentional: adapters must support streamed output,
     * not rely on a single callback invocation. */
    if (write_fn(write_user_data, bytes, 17) ||
        write_fn(write_user_data, bytes + 17, 29) ||
        write_fn(write_user_data, bytes + 46, FIXTURE_SNAPSHOT_SIZE - 46))
        return fail(error, error_size, "fixture snapshot writer failed");
    return 0;
}

static int fixture_restore(void *session_impl, ColiSegmentReadFn read_fn,
                           void *read_user_data,
                           char *error, size_t error_size) {
    FixtureSession *session = (FixtureSession *)session_impl;
    unsigned char bytes[FIXTURE_SNAPSHOT_SIZE];
    if (read_fn(read_user_data, bytes, 13) ||
        read_fn(read_user_data, bytes + 13, 31) ||
        read_fn(read_user_data, bytes + 44, FIXTURE_SNAPSHOT_SIZE - 44))
        return fail(error, error_size, "fixture snapshot reader failed");
    return decode_snapshot(session, bytes, error, error_size);
}

#define DECLARE_OPEN_WRAPPER(name, index)                                      \
    static int name##_open(void **engine_impl,                                 \
                           ColiSegmentCapabilities *capabilities,              \
                           const ColiSegmentEngineOptions *options,            \
                           char *error, size_t error_size) {                    \
        return open_fixture(&g_fixtures[index], engine_impl, capabilities,      \
                            options, error, error_size);                        \
    }

DECLARE_OPEN_WRAPPER(glm, 0)
DECLARE_OPEN_WRAPPER(inkling, 1)
DECLARE_OPEN_WRAPPER(kimi, 2)
DECLARE_OPEN_WRAPPER(olmoe, 3)
DECLARE_OPEN_WRAPPER(qwen36, 4)
DECLARE_OPEN_WRAPPER(deepseek_v4, 5)

#define FIXTURE_ADAPTER(name)                                                  \
    {                                                                          \
        .struct_size = sizeof(ColiSegmentAdapter),                             \
        .abi_version = COLI_SEGMENT_ABI_VERSION,                               \
        .engine_id = #name,                                                    \
        .engine_open = name##_open,                                            \
        .engine_destroy = fixture_engine_destroy,                              \
        .session_create = fixture_session_create,                              \
        .session_destroy = fixture_session_destroy,                            \
        .session_run = fixture_session_run,                                    \
        .session_snapshot = fixture_snapshot,                                  \
        .session_restore = fixture_restore,                                    \
        .reserved_fn = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL},       \
    }

static const ColiSegmentAdapter g_adapters[] = {
    FIXTURE_ADAPTER(glm),
    FIXTURE_ADAPTER(inkling),
    FIXTURE_ADAPTER(kimi),
    FIXTURE_ADAPTER(olmoe),
    FIXTURE_ADAPTER(qwen36),
    FIXTURE_ADAPTER(deepseek_v4),
};

int coli_segment_conformance_register_fixtures(void) {
    if (sizeof(g_adapters) / sizeof(g_adapters[0]) !=
        coli_segment_conformance_fixture_count())
        return -1;
    for (size_t i = 0; i < coli_segment_conformance_fixture_count(); i++) {
        if (strcmp(g_adapters[i].engine_id, g_fixtures[i].family_id) != 0 ||
            coli_segment_adapter_register(&g_adapters[i]) != 0)
            return -1;
    }
    return 0;
}

static int stream_write(void *user_data, const void *data, size_t size) {
    FixtureStream *stream = (FixtureStream *)user_data;
    if (size > sizeof(stream->bytes) - stream->length) return -1;
    memcpy(stream->bytes + stream->length, data, size);
    stream->length += size;
    return 0;
}

static int stream_read(void *user_data, void *data, size_t size) {
    FixtureStream *stream = (FixtureStream *)user_data;
    if (stream->offset > stream->length || size > stream->length - stream->offset)
        return -1;
    memcpy(data, stream->bytes + stream->offset, size);
    stream->offset += size;
    return 0;
}

static void fill_input(float *input, size_t count, uint32_t salt) {
    for (size_t i = 0; i < count; i++)
        input[i] = (float)((i + 1) * (salt + 3u) % 97u) / 32.0f;
}

static int floats_equal(const float *a, const float *b, size_t count) {
    return memcmp(a, b, count * sizeof(*a)) == 0;
}

#define REQUIRE(condition, message)                                            \
    do {                                                                       \
        if (!(condition)) {                                                    \
            result = fail(error, error_size, message);                         \
            goto cleanup;                                                      \
        }                                                                      \
    } while (0)

int coli_segment_conformance_run_fixture(
    const ColiSegmentConformanceFixture *fixture,
    char *error, size_t error_size) {
    int result = -1;
    int index = fixture_index(fixture);
    ColiSegmentEngine *engine = NULL, *other_range = NULL;
    ColiSegmentSession *first = NULL, *second = NULL, *clean = NULL;
    ColiSegmentSession *wrong_range = NULL;
    float *input = NULL, *out_first = NULL, *out_second = NULL, *out_clean = NULL;
    FixtureStream checkpoint = {{0}, 0, 0};
    if (index < 0) return fail(error, error_size, "unknown conformance fixture");

    ColiSegmentEngineOptions open_options = {
        .struct_size = sizeof(open_options),
        .model_dir = fixture->tiny_generator,
        .layer_begin = 1,
        .layer_end = fixture->num_layers - 1,
        .context_tokens = fixture->max_context_tokens / 2,
    };
    REQUIRE(coli_segment_engine_open(fixture->family_id, &open_options, &engine,
                                     error, error_size) == 0,
            "cannot open fixture segment engine");

    ColiSegmentCapabilities capabilities = {.struct_size = sizeof(capabilities)};
    REQUIRE(coli_segment_engine_capabilities(engine, &capabilities,
                                             error, error_size) == 0,
            "cannot read fixture capabilities");
    REQUIRE(strcmp(capabilities.engine_id, fixture->family_id) == 0,
            "fixture engine identity changed");
    REQUIRE(strcmp(capabilities.state_schema, fixture->state_schema) == 0,
            "fixture state schema changed");
    REQUIRE(capabilities.state_width == fixture->state_width &&
                capabilities.num_layers == fixture->num_layers,
            "fixture geometry changed");
    REQUIRE((capabilities.flags & (COLI_SEGMENT_CAP_TOKEN_IDS |
                                   COLI_SEGMENT_CAP_SNAPSHOT |
                                   COLI_SEGMENT_CAP_RANGE_NATIVE |
                                   COLI_SEGMENT_CAP_MULTI_SESSION |
                                   COLI_SEGMENT_CAP_CPU)) ==
                (COLI_SEGMENT_CAP_TOKEN_IDS | COLI_SEGMENT_CAP_SNAPSHOT |
                 COLI_SEGMENT_CAP_RANGE_NATIVE |
                 COLI_SEGMENT_CAP_MULTI_SESSION | COLI_SEGMENT_CAP_CPU),
            "fixture is missing mandatory Segment capabilities");
    REQUIRE((capabilities.flags & (COLI_SEGMENT_CAP_CUDA |
                                   COLI_SEGMENT_CAP_HIP |
                                   COLI_SEGMENT_CAP_METAL |
                                   COLI_SEGMENT_CAP_VULKAN)) == 0,
            "contract fixture must not claim a real GPU backend");

    ColiSegmentSessionOptions session_options = {
        .struct_size = sizeof(session_options),
        .context_tokens = open_options.context_tokens,
    };
    REQUIRE(coli_segment_session_create(engine, &session_options, &first,
                                        error, error_size) == 0 &&
                coli_segment_session_create(engine, &session_options, &second,
                                            error, error_size) == 0 &&
                coli_segment_session_create(engine, &session_options, &clean,
                                            error, error_size) == 0,
            "cannot create isolated fixture sessions");
    if (coli_segment_engine_close(engine, error, error_size) == 0) {
        /* A broken runtime may have freed it despite the live sessions. Avoid
         * turning the conformance diagnostic into a cleanup use-after-free. */
        engine = NULL;
        result = fail(error, error_size,
                      "engine closed while fixture sessions were alive");
        goto cleanup;
    }

    size_t values = (size_t)2 * fixture->state_width;
    input = (float *)calloc(values, sizeof(*input));
    out_first = (float *)calloc(values, sizeof(*out_first));
    out_second = (float *)calloc(values, sizeof(*out_second));
    out_clean = (float *)calloc(values, sizeof(*out_clean));
    REQUIRE(input && out_first && out_second && out_clean,
            "fixture activation allocation failed");
    fill_input(input, values, fixture->fixture_tag);
    int32_t tokens[2] = {17 + index, 29 + index};
    ColiSegmentRunRequest run = {
        .struct_size = sizeof(run),
        .rows = 2,
        .position = 0,
        .token_ids = tokens,
        .token_count = 2,
        .input = input,
        .input_bytes = values * sizeof(*input),
        .output = out_first,
        .output_bytes = values * sizeof(*out_first),
    };
    REQUIRE(coli_segment_run(first, &run, error, error_size) == 0,
            "first fixture run failed");

    run.output = out_second;
    REQUIRE(coli_segment_run(second, &run, error, error_size) == 0 &&
                floats_equal(out_first, out_second, values),
            "fresh fixture sessions are not deterministic and isolated");

    /* Advance only the first session, proving its state cannot leak into the
     * untouched clean session. */
    run.position = 2;
    run.output = out_first;
    REQUIRE(coli_segment_run(first, &run, error, error_size) == 0,
            "fixture continuation failed");
    run.position = 0;
    run.output = out_clean;
    REQUIRE(coli_segment_run(clean, &run, error, error_size) == 0 &&
                floats_equal(out_second, out_clean, values),
            "fixture session state leaked across conversations");

    REQUIRE(coli_segment_snapshot(first, stream_write, &checkpoint,
                                  error, error_size) == 0 &&
                checkpoint.length == FIXTURE_SNAPSHOT_SIZE,
            "fixture snapshot stream failed");
    checkpoint.offset = 0;
    REQUIRE(coli_segment_restore(second, stream_read, &checkpoint,
                                 error, error_size) == 0 &&
                checkpoint.offset == checkpoint.length,
            "fixture restore stream failed");

    run.position = 4;
    run.output = out_first;
    REQUIRE(coli_segment_run(first, &run, error, error_size) == 0,
            "source run after checkpoint failed");
    run.output = out_second;
    REQUIRE(coli_segment_run(second, &run, error, error_size) == 0 &&
                floats_equal(out_first, out_second, values),
            "restored fixture did not reproduce exact continuation");

    /* A failed restore must be transactional. Snapshot second, corrupt a copy,
     * reject it, then prove the live state remained byte-identical. */
    FixtureStream before_bad = {{0}, 0, 0};
    FixtureStream after_bad = {{0}, 0, 0};
    REQUIRE(coli_segment_snapshot(second, stream_write, &before_bad,
                                  error, error_size) == 0,
            "cannot snapshot before corruption check");
    FixtureStream corrupt = before_bad;
    corrupt.bytes[41] ^= 0x80u;
    corrupt.offset = 0;
    REQUIRE(coli_segment_restore(second, stream_read, &corrupt,
                                 error, error_size) != 0,
            "corrupt fixture snapshot was accepted");
    REQUIRE(coli_segment_snapshot(second, stream_write, &after_bad,
                                  error, error_size) == 0 &&
                before_bad.length == after_bad.length &&
                memcmp(before_bad.bytes, after_bad.bytes,
                       before_bad.length) == 0,
            "failed restore mutated the live fixture session");

    /* The private snapshot identity includes the half-open layer range. */
    ColiSegmentEngineOptions other_options = open_options;
    other_options.layer_begin = 0;
    other_options.layer_end = fixture->num_layers;
    REQUIRE(coli_segment_engine_open(fixture->family_id, &other_options,
                                     &other_range, error, error_size) == 0 &&
                coli_segment_session_create(other_range, &session_options,
                                            &wrong_range,
                                            error, error_size) == 0,
            "cannot open alternate fixture range");
    before_bad.offset = 0;
    REQUIRE(coli_segment_restore(wrong_range, stream_read, &before_bad,
                                 error, error_size) != 0,
            "snapshot crossed incompatible fixture ranges");

    /* Ordering is adapter-owned: reject a gap and preserve state. */
    FixtureStream before_gap = {{0}, 0, 0};
    FixtureStream after_gap = {{0}, 0, 0};
    REQUIRE(coli_segment_snapshot(first, stream_write, &before_gap,
                                  error, error_size) == 0,
            "cannot snapshot before ordering check");
    run.position = 9;
    run.output = out_first;
    REQUIRE(coli_segment_run(first, &run, error, error_size) != 0,
            "fixture accepted a non-contiguous position");
    REQUIRE(coli_segment_snapshot(first, stream_write, &after_gap,
                                  error, error_size) == 0 &&
                before_gap.length == after_gap.length &&
                memcmp(before_gap.bytes, after_gap.bytes,
                       before_gap.length) == 0,
            "rejected fixture run mutated session state");

    result = 0;
cleanup:
    free(input);
    free(out_first);
    free(out_second);
    free(out_clean);
    coli_segment_session_destroy(wrong_range);
    if (other_range) (void)coli_segment_engine_close(other_range, NULL, 0);
    coli_segment_session_destroy(clean);
    coli_segment_session_destroy(second);
    coli_segment_session_destroy(first);
    if (engine) (void)coli_segment_engine_close(engine, NULL, 0);
    return result;
}
