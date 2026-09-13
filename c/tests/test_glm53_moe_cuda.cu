#include "../backend_cuda.h"
#include "../glm53_gpu.h"

#include <cmath>
#include <cfloat>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

static int check(int condition, const char *message) {
    if (!condition) std::fprintf(stderr, "FAIL: %s\n", message);
    return condition;
}

static int compare_output(const char *name, const std::vector<float> &got,
                          const std::vector<float> &want,
                          float max_rel, float min_cosine);

static ColiGpuTensor *tensor(ColiGpuContext *ctx, const float *data,
                             int rows, int columns) {
    ColiGpuTensorDesc desc = {};
    desc.data = data;
    desc.rows = rows;
    desc.columns = columns;
    ColiGpuTensor *out = nullptr;
    return coli_gpu_tensor_create(&out, ctx, &desc) ? out : nullptr;
}

static ColiGpuTensor *quant_tensor(ColiGpuContext *ctx,
                                   const unsigned char *data,
                                   const float *scales,
                                   int rows, int columns) {
    ColiGpuTensorDesc desc = {};
    desc.data = data;
    desc.scales = scales;
    desc.format = 4;
    desc.rows = rows;
    desc.columns = columns;
    desc.group_size = 64;
    ColiGpuTensor *out = nullptr;
    return coli_gpu_tensor_create(&out, ctx, &desc) ? out : nullptr;
}

static void cpu_route(const float *input, const float *weight,
                      const float *bias, int rows, int hidden, int experts,
                      int topk, int normalize, float scale,
                      std::vector<int> *ids, std::vector<float> *weights) {
    ids->assign((size_t)rows * topk, -1);
    weights->assign((size_t)rows * topk, 0.0f);
    std::vector<float> score(experts);
    for (int row = 0; row < rows; ++row) {
        for (int expert = 0; expert < experts; ++expert) {
            float sum = 0.0f;
            for (int column = 0; column < hidden; ++column)
                sum += input[(size_t)row * hidden + column] *
                       weight[(size_t)expert * hidden + column];
            score[expert] = 1.0f / (1.0f + std::exp(-sum));
        }
        float total = 0.0f;
        for (int k = 0; k < topk; ++k) {
            int best = -1;
            float value = -INFINITY;
            for (int expert = 0; expert < experts; ++expert) {
                int used = 0;
                for (int prior = 0; prior < k; ++prior)
                    used |= (*ids)[(size_t)row * topk + prior] == expert;
                float choice = score[expert] + (bias ? bias[expert] : 0.0f);
                if (!used && choice > value) {
                    value = choice;
                    best = expert;
                }
            }
            (*ids)[(size_t)row * topk + k] = best;
            (*weights)[(size_t)row * topk + k] = score[best];
            total += score[best];
        }
        for (int k = 0; k < topk; ++k) {
            float &value = (*weights)[(size_t)row * topk + k];
            if (normalize) value /= total + 1e-20f;
            value *= scale;
        }
    }
}

static int route_case(ColiGpuContext *ctx, int rows, int hidden, int experts,
                      int topk, const std::vector<float> &input,
                      const std::vector<float> &weight,
                      const std::vector<float> &bias,
                      int normalize, float scale,
                      const char *name) {
    ColiGpuRouteConfig config = {};
    config.hidden = hidden;
    config.experts = experts;
    config.topk = topk;
    config.normalize_topk = normalize;
    config.routed_scale = scale;
    ColiGpuRouter *router = nullptr;
    ColiGpuArena *arena = nullptr;
    ColiGpuTensor *rw = tensor(ctx, weight.data(), experts, hidden);
    ColiGpuTensor *rb = tensor(ctx, bias.data(), 1, experts);
    std::vector<int> got_ids((size_t)rows * topk);
    std::vector<float> got_weights((size_t)rows * topk);
    std::vector<int> want_ids;
    std::vector<float> want_weights;
    cpu_route(input.data(), weight.data(), bias.data(), rows, hidden, experts,
              topk, normalize, scale, &want_ids, &want_weights);
    int ok = rw && rb &&
        coli_gpu_arena_create(&arena, ctx,
                              (size_t)rows * hidden * sizeof(float)) &&
        coli_gpu_arena_upload_activation(
            arena, 0, input.data(),
            (size_t)rows * hidden * sizeof(float)) &&
        coli_gpu_router_create(&router, ctx, &config, rows) &&
        coli_gpu_router_run(router, arena, 0, rw, rb, rows) &&
        coli_gpu_router_download(router, got_ids.data(), got_weights.data(),
                                 got_ids.size(), rows);
    if (ok) {
        ok = got_ids == want_ids;
        for (size_t i = 0; ok && i < got_weights.size(); ++i) {
            float error = std::fabs(got_weights[i] - want_weights[i]);
            ok = error <= 3e-4f * (std::fabs(want_weights[i]) + 1e-6f);
        }
        std::string metric_name = std::string("router primitive ") + name;
        ok = ok && compare_output(metric_name.c_str(), got_weights,
                                  want_weights, 3e-4f, 0.99999f);
    }
    std::vector<int> repeated_ids(got_ids.size());
    std::vector<float> repeated_weights(got_weights.size());
    if (ok) {
        ok = coli_gpu_router_run(router, arena, 0, rw, rb, rows) &&
             coli_gpu_router_download(
                 router, repeated_ids.data(), repeated_weights.data(),
                 repeated_ids.size(), rows) &&
             !std::memcmp(got_ids.data(), repeated_ids.data(),
                          got_ids.size() * sizeof(int)) &&
             !std::memcmp(got_weights.data(), repeated_weights.data(),
                          got_weights.size() * sizeof(float));
    }
    if (!ok) std::fprintf(stderr, "FAIL: router oracle %s\n", name);
    coli_gpu_router_destroy(router);
    coli_gpu_arena_destroy(arena);
    coli_gpu_tensor_destroy(rb);
    coli_gpu_tensor_destroy(rw);
    return ok;
}

static int test_router(ColiGpuContext *ctx) {
    {
        std::vector<float> zero(1, 0.0f);
        if (!route_case(ctx, 1, 1, 1, 1, zero, zero, zero,
                        0, 1.0f, "absolute minimum dimensions"))
            return 0;
    }
    {
        const int rows = 1, hidden = 1, experts = 8, topk = 8;
        std::vector<float> input(rows * hidden, 0.0f);
        std::vector<float> weight(experts * hidden, 0.0f);
        std::vector<float> bias(experts, 0.0f);
        if (!route_case(ctx, rows, hidden, experts, topk, input, weight, bias,
                        1, 2.5f, "zero/minimum stable tie"))
            return 0;
    }
    {
        const int rows = 2, hidden = 3, experts = 10, topk = 3;
        std::vector<float> input = {1.0f, -0.5f, 0.25f,
                                    -0.2f, 0.7f, 0.9f};
        std::vector<float> weight((size_t)experts * hidden);
        for (size_t i = 0; i < weight.size(); ++i)
            weight[i] = ((int)(i % 7) - 3) * 0.125f;
        std::vector<float> bias(experts, 0.0f);
        bias[9] = 4.0f;
        bias[8] = 3.0f;
        bias[7] = 2.0f;
        if (!route_case(ctx, rows, hidden, experts, topk, input, weight, bias,
                        0, 0.75f, "bias selects but raw score weights"))
            return 0;
        if (!route_case(ctx, rows, hidden, experts, topk, input, weight, bias,
                        1, 2.5f, "normalization and routed scale"))
            return 0;
    }
    {
        const int rows = 1, hidden = 4096, experts = 288, topk = 8;
        std::vector<float> input(hidden);
        std::vector<float> weight((size_t)experts * hidden);
        std::vector<float> bias(experts);
        for (int i = 0; i < hidden; ++i)
            input[i] = ((i % 19) - 9) * 0.003f;
        for (size_t i = 0; i < weight.size(); ++i)
            weight[i] = ((int)(i % 23) - 11) * 0.0004f;
        for (int i = 0; i < experts; ++i)
            bias[i] = ((i % 13) - 6) * 0.0001f;
        if (!route_case(ctx, rows, hidden, experts, topk, input, weight, bias,
                        1, 2.5f, "production 4096x288 top-8"))
            return 0;
    }
    return 1;
}

static int test_nonfinite_descriptors(ColiGpuContext *ctx) {
    float nan_value = NAN;
    ColiGpuTensorDesc tensor_desc = {};
    tensor_desc.data = &nan_value;
    tensor_desc.rows = 1;
    tensor_desc.columns = 1;
    ColiGpuTensor *bad_tensor = nullptr;
    if (coli_gpu_tensor_create(&bad_tensor, ctx, &tensor_desc) || bad_tensor)
        return 0;
    ColiGpuRouteConfig route = {1, 1, 1, 1, NAN};
    ColiGpuRouter *bad_router = nullptr;
    if (coli_gpu_router_create(&bad_router, ctx, &route, 1) || bad_router)
        return 0;
    ColiGpuExpertCacheConfig cache_config = {};
    cache_config.experts = 1;
    cache_config.slots = 1;
    cache_config.hidden = 1;
    cache_config.intermediate = 1;
    cache_config.group_size = 64;
    cache_config.max_rows = 1;
    cache_config.swiglu_limit = 0.0f;
    ColiGpuExpertCache *bad_cache = nullptr;
    if (coli_gpu_moe_scratch_bytes(&cache_config) != 0 ||
        coli_gpu_expert_cache_create(&bad_cache, ctx, &cache_config) ||
        bad_cache) {
        coli_gpu_expert_cache_destroy(bad_cache);
        std::fprintf(stderr, "FAIL: zero SwiGLU limit accepted\n");
        return 0;
    }
    ColiGpuArena *control_arena = nullptr;
    int control_value = 7;
    ColiGpuTelemetry control_before = {}, control_after = {};
    if (!coli_gpu_arena_create(
            &control_arena, ctx, sizeof(control_value)))
        return 0;
    coli_gpu_context_telemetry(ctx, &control_before);
    if (!coli_gpu_arena_upload(
            control_arena, 0, &control_value, sizeof(control_value)))
        return 0;
    coli_gpu_context_telemetry(ctx, &control_after);
    coli_gpu_arena_destroy(control_arena);
    if (!check(control_after.h2d_copies == control_before.h2d_copies + 1 &&
               control_after.host_activation_h2d_copies ==
                   control_before.host_activation_h2d_copies,
               "control transfer is not activation telemetry"))
        return 0;
    return 1;
}

struct QuantizedExpert {
    std::vector<unsigned char> gate, up, down;
    std::vector<float> gate_scales, up_scales, down_scales;

    QuantizedExpert(int hidden, int intermediate, int seed) {
        gate.resize((size_t)intermediate * ((hidden + 1) / 2));
        up.resize(gate.size());
        down.resize((size_t)hidden * ((intermediate + 1) / 2));
        gate_scales.resize((size_t)intermediate * ((hidden + 63) / 64));
        up_scales.resize(gate_scales.size());
        down_scales.resize((size_t)hidden * ((intermediate + 63) / 64));
        for (size_t i = 0; i < gate.size(); ++i) {
            gate[i] = (unsigned char)((i * 29 + seed * 17) & 255);
            up[i] = (unsigned char)((i * 11 + seed * 31) & 255);
        }
        for (size_t i = 0; i < down.size(); ++i)
            down[i] = (unsigned char)((i * 7 + seed * 13) & 255);
        for (size_t i = 0; i < gate_scales.size(); ++i) {
            gate_scales[i] = 0.006f + (float)((i + seed) % 5) * 0.001f;
            up_scales[i] = 0.005f + (float)((i + 2 * seed) % 7) * 0.001f;
        }
        for (size_t i = 0; i < down_scales.size(); ++i)
            down_scales[i] = 0.004f + (float)((i + 3 * seed) % 6) * 0.001f;
    }

    ColiGpuExpertSource source(int hidden, int intermediate) const {
        ColiGpuExpertSource source = {};
        source.gate = {gate.data(), gate_scales.data(), 4,
                       intermediate, hidden, 64};
        source.up = {up.data(), up_scales.data(), 4,
                     intermediate, hidden, 64};
        source.down = {down.data(), down_scales.data(), 4,
                       hidden, intermediate, 64};
        return source;
    }
};

static int test_cache(ColiGpuContext *ctx) {
    ColiGpuExpertCacheConfig config = {};
    config.experts = 288;
    config.slots = 70;
    config.hidden = 65;
    config.intermediate = 67;
    config.group_size = 64;
    config.max_rows = 9;
    config.swiglu_limit = 1.5f;
    ColiGpuExpertCache *cache = nullptr;
    if (!coli_gpu_expert_cache_create(&cache, ctx, &config)) return 0;
    ColiGpuTelemetry unpublished_before = {}, unpublished_after = {};
    ColiGpuExpertHandle unpublished = {};
    unpublished.slot = 0;
    coli_gpu_context_telemetry(ctx, &unpublished_before);
    if (coli_gpu_expert_cache_validate(cache, &unpublished))
        return 0;
    coli_gpu_context_telemetry(ctx, &unpublished_after);
    if (!check(unpublished_after.unpublished_slot_rejections ==
                   unpublished_before.unpublished_slot_rejections + 1,
               "unpublished slot telemetry is exact"))
        return 0;
    ColiGpuTelemetry before_fill = {};
    coli_gpu_context_telemetry(ctx, &before_fill);
    std::vector<ColiGpuExpertHandle> handles(70);
    for (int expert = 0; expert < 70; ++expert) {
        QuantizedExpert bytes(config.hidden, config.intermediate, expert);
        ColiGpuExpertSource source = bytes.source(
            config.hidden, config.intermediate);
        if (!coli_gpu_expert_cache_upload(
                cache, expert, expert, &source, &handles[expert]))
            return 0;
    }
    ColiGpuTelemetry after_fill = {};
    coli_gpu_context_telemetry(ctx, &after_fill);
    size_t one_upload =
        (size_t)config.intermediate * ((config.hidden + 1) / 2) * 2 +
        (size_t)config.hidden * ((config.intermediate + 1) / 2) +
        ((size_t)config.intermediate * ((config.hidden + 63) / 64) * 2 +
         (size_t)config.hidden * ((config.intermediate + 63) / 64)) *
            sizeof(float);
    if (!check(after_fill.expert_publications -
                    before_fill.expert_publications == 70,
               "cache publishes exactly 70 distinct experts") ||
        !check(after_fill.expert_upload_events -
                    before_fill.expert_upload_events == 70,
               "each upload records exactly one publication event") ||
        !check(after_fill.expert_upload_bytes -
                    before_fill.expert_upload_bytes == one_upload * 70,
               "expert upload-byte telemetry is exact") ||
        !check(after_fill.h2d_copies - before_fill.h2d_copies == 6 * 70,
               "expert upload copy telemetry is exact") ||
        !check(after_fill.h2d_bytes - before_fill.h2d_bytes ==
                    one_upload * 70,
               "expert upload H2D bytes are exact") ||
        !check(after_fill.device_allocations ==
                    before_fill.device_allocations &&
                after_fill.d2h_copies == before_fill.d2h_copies,
               "expert upload has no device allocation or D2H"))
        return 0;
    ColiGpuTelemetry hit_before = {}, hit_after = {};
    ColiGpuExpertHandle hit = {};
    coli_gpu_context_telemetry(ctx, &hit_before);
    if (!coli_gpu_expert_cache_lookup(cache, 69, &hit) ||
        std::memcmp(&hit, &handles[69], sizeof(hit)))
        return 0;
    coli_gpu_context_telemetry(ctx, &hit_after);
    if (!check(hit_after.expert_cache_hits ==
                   hit_before.expert_cache_hits + 1 &&
               hit_after.device_allocations == hit_before.device_allocations &&
               hit_after.h2d_copies == hit_before.h2d_copies &&
               hit_after.d2h_copies == hit_before.d2h_copies,
               "cache hit has exact telemetry and no transfers/allocation"))
        return 0;
    ColiGpuExpertHandle miss = {};
    ColiGpuTelemetry miss_before = {}, miss_after = {};
    coli_gpu_context_telemetry(ctx, &miss_before);
    if (coli_gpu_expert_cache_lookup(cache, 287, &miss)) return 0;
    coli_gpu_context_telemetry(ctx, &miss_after);
    if (!check(miss_after.expert_cache_misses ==
                   miss_before.expert_cache_misses + 1,
               "cache miss telemetry is exact"))
        return 0;

    ColiGpuExpertSlotInfo before = {};
    if (!coli_gpu_expert_cache_slot_info(cache, 0, &before)) return 0;
    QuantizedExpert invalid(config.hidden, config.intermediate, 99);
    invalid.gate_scales[0] = NAN;
    ColiGpuExpertSource invalid_source =
        invalid.source(config.hidden, config.intermediate);
    ColiGpuExpertHandle invalid_handle = {};
    if (coli_gpu_expert_cache_upload(
            cache, 99, 0, &invalid_source, &invalid_handle))
        return 0;
    ColiGpuExpertSlotInfo after_invalid = {};
    if (!coli_gpu_expert_cache_slot_info(cache, 0, &after_invalid) ||
        std::memcmp(&before, &after_invalid, sizeof(before)))
        return 0;
    for (int point = COLI_GPU_EXPERT_FAULT_ALLOCATION;
         point <= COLI_GPU_EXPERT_FAULT_EVENT_RECORD; ++point) {
        QuantizedExpert replacement(config.hidden, config.intermediate,
                                    100 + point);
        ColiGpuExpertSource source = replacement.source(
            config.hidden, config.intermediate);
        ColiGpuExpertHandle rejected = {};
        if (!coli_gpu_expert_cache_inject_fault(
                cache, (ColiGpuExpertFaultPoint)point) ||
            coli_gpu_expert_cache_upload(
                cache, 100 + point, 0, &source, &rejected))
            return 0;
        ColiGpuExpertSlotInfo after = {};
        if (!coli_gpu_expert_cache_slot_info(cache, 0, &after) ||
            std::memcmp(&before, &after, sizeof(before))) {
            std::fprintf(stderr,
                         "FAIL: failed upload changed published slot state\n");
            return 0;
        }
    }
    const size_t copy_bytes[6] = {
        (size_t)config.intermediate * ((config.hidden + 1) / 2),
        (size_t)config.intermediate * ((config.hidden + 63) / 64) *
            sizeof(float),
        (size_t)config.intermediate * ((config.hidden + 1) / 2),
        (size_t)config.intermediate * ((config.hidden + 63) / 64) *
            sizeof(float),
        (size_t)config.hidden * ((config.intermediate + 1) / 2),
        (size_t)config.hidden * ((config.intermediate + 63) / 64) *
            sizeof(float)
    };
    for (int boundary = 1; boundary <= 6; ++boundary) {
        ColiGpuExpertSlotInfo stable = {}, after_failure = {};
        ColiGpuTelemetry fault_before = {}, fault_after = {};
        if (!coli_gpu_expert_cache_slot_info(cache, 0, &stable))
            return 0;
        coli_gpu_context_telemetry(ctx, &fault_before);
        QuantizedExpert partial(config.hidden, config.intermediate,
                                300 + boundary);
        ColiGpuExpertSource partial_source =
            partial.source(config.hidden, config.intermediate);
        ColiGpuExpertHandle rejected = {};
        if (!coli_gpu_expert_cache_inject_fault_at(
                cache, COLI_GPU_EXPERT_FAULT_UPLOAD, boundary) ||
            coli_gpu_expert_cache_upload(
                cache, 210 + boundary, 0, &partial_source, &rejected))
            return 0;
        coli_gpu_context_telemetry(ctx, &fault_after);
        size_t queued_bytes = 0;
        for (int copy = 0; copy < boundary; ++copy)
            queued_bytes += copy_bytes[copy];
        if (!coli_gpu_expert_cache_slot_info(cache, 0, &after_failure) ||
            std::memcmp(&stable, &after_failure, sizeof(stable)) ||
            !check(fault_after.h2d_copies - fault_before.h2d_copies ==
                       (uint64_t)boundary &&
                   fault_after.h2d_bytes - fault_before.h2d_bytes ==
                       queued_bytes &&
                   fault_after.expert_upload_bytes -
                       fault_before.expert_upload_bytes == queued_bytes &&
                   fault_after.expert_upload_events ==
                       fault_before.expert_upload_events &&
                   fault_after.expert_publications ==
                       fault_before.expert_publications,
                   "partial upload accounts exact queued DMA"))
            return 0;
        ColiGpuExpertHandle retry = {};
        if (!coli_gpu_expert_cache_upload(
                cache, 220 + boundary, 0, &partial_source, &retry))
            return 0;
    }
    QuantizedExpert replacement(config.hidden, config.intermediate, 200);
    ColiGpuExpertSource source = replacement.source(
        config.hidden, config.intermediate);
    ColiGpuExpertHandle fresh = {};
    ColiGpuTelemetry eviction_before = {}, eviction_after = {};
    coli_gpu_context_telemetry(ctx, &eviction_before);
    if (!coli_gpu_expert_cache_upload(cache, 200, 0, &source, &fresh) ||
        fresh.generation == handles[0].generation)
        return 0;
    coli_gpu_context_telemetry(ctx, &eviction_after);
    if (!check(eviction_after.expert_cache_evictions ==
                   eviction_before.expert_cache_evictions + 1,
               "cache eviction telemetry is exact"))
        return 0;
    ColiGpuTelemetry first_stale_before = {}, first_stale_after = {};
    coli_gpu_context_telemetry(ctx, &first_stale_before);
    if (coli_gpu_expert_cache_validate(cache, &handles[0]))
        return 0;
    coli_gpu_context_telemetry(ctx, &first_stale_after);
    if (!check(first_stale_after.stale_generation_rejections ==
                   first_stale_before.stale_generation_rejections + 1,
               "evicted handle stale telemetry is exact") ||
        !coli_gpu_expert_cache_validate(cache, &fresh))
        return 0;
    ColiGpuTelemetry category_before = {}, category_after = {};
    coli_gpu_context_telemetry(ctx, &category_before);
    ColiGpuExpertHandle out_of_range = fresh;
    out_of_range.slot = -1;
    ColiGpuExpertHandle wrong_expert = fresh;
    wrong_expert.expert_id++;
    ColiGpuExpertHandle stale = fresh;
    stale.generation--;
    if (coli_gpu_expert_cache_validate(cache, &out_of_range) ||
        coli_gpu_expert_cache_validate(cache, &wrong_expert) ||
        coli_gpu_expert_cache_validate(cache, &stale))
        return 0;
    coli_gpu_context_telemetry(ctx, &category_after);
    if (!check(category_after.expert_handle_range_rejections ==
                   category_before.expert_handle_range_rejections + 1 &&
               category_after.wrong_expert_rejections ==
                   category_before.wrong_expert_rejections + 1 &&
               category_after.stale_generation_rejections ==
                   category_before.stale_generation_rejections + 1 &&
               category_after.unpublished_slot_rejections ==
                   category_before.unpublished_slot_rejections,
               "handle rejection telemetry categories are exact"))
        return 0;

    if (!coli_gpu_expert_cache_test_set_generation(
            cache, 1, UINT64_MAX - 1))
        return 0;
    QuantizedExpert near_wrap(config.hidden, config.intermediate, 201);
    ColiGpuExpertSource near_wrap_source =
        near_wrap.source(config.hidden, config.intermediate);
    ColiGpuExpertHandle maximum = {}, rejected_wrap = {};
    ColiGpuTelemetry exhaustion_before = {}, exhaustion_after = {};
    coli_gpu_context_telemetry(ctx, &exhaustion_before);
    if (!coli_gpu_expert_cache_upload(
            cache, 201, 1, &near_wrap_source, &maximum) ||
        maximum.generation != UINT64_MAX ||
        coli_gpu_expert_cache_upload(
            cache, 202, 1, &near_wrap_source, &rejected_wrap) ||
        coli_gpu_expert_cache_healthy(cache))
        return 0;
    coli_gpu_context_telemetry(ctx, &exhaustion_after);
    if (!check(exhaustion_after.generation_exhaustions ==
                   exhaustion_before.generation_exhaustions + 1 &&
               exhaustion_after.expert_upload_events ==
                   exhaustion_before.expert_upload_events + 1 &&
               exhaustion_after.expert_publications ==
                   exhaustion_before.expert_publications + 1,
               "generation exhaustion telemetry is exact"))
        return 0;
    ColiGpuExpertSlotInfo exhausted = {};
    if (!coli_gpu_expert_cache_slot_info(cache, 1, &exhausted) ||
        exhausted.generation != UINT64_MAX ||
        exhausted.expert_id != maximum.expert_id)
        return 0;
    ColiGpuTelemetry unhealthy_before = {}, unhealthy_after = {};
    coli_gpu_context_telemetry(ctx, &unhealthy_before);
    if (coli_gpu_expert_cache_validate(cache, &maximum))
        return 0;
    coli_gpu_context_telemetry(ctx, &unhealthy_after);
    int ok = check(unhealthy_after.unhealthy_cache_rejections ==
                       unhealthy_before.unhealthy_cache_rejections + 1 &&
                   unhealthy_after.expert_handle_range_rejections ==
                       unhealthy_before.expert_handle_range_rejections,
                   "unhealthy cache rejection telemetry is exact");
    coli_gpu_expert_cache_destroy(cache);
    return ok;
}

static float qvalue(const std::vector<unsigned char> &data,
                    const std::vector<float> &scales,
                    int row, int column, int columns) {
    size_t row_bytes = (size_t)(columns + 1) / 2;
    unsigned char packed = data[(size_t)row * row_bytes + column / 2];
    int nibble = column & 1 ? packed >> 4 : packed & 15;
    int value = nibble - 8;
    int groups = (columns + 63) / 64;
    return value * scales[(size_t)row * groups + column / 64];
}

static void cpu_expert(float *output, const float *input,
                       const QuantizedExpert &expert,
                       int hidden, int intermediate, float limit) {
    std::vector<float> activated(intermediate);
    for (int out = 0; out < intermediate; ++out) {
        float gate = 0.0f, up = 0.0f;
        for (int in = 0; in < hidden; ++in) {
            gate += input[in] * qvalue(expert.gate, expert.gate_scales,
                                       out, in, hidden);
            up += input[in] * qvalue(expert.up, expert.up_scales,
                                     out, in, hidden);
        }
        if (limit > 0.0f) {
            gate = std::fmin(gate, limit);
            up = std::fmin(std::fmax(up, -limit), limit);
        }
        activated[out] = gate / (1.0f + std::exp(-gate)) * up;
    }
    for (int out = 0; out < hidden; ++out) {
        float sum = 0.0f;
        for (int in = 0; in < intermediate; ++in)
            sum += activated[in] *
                   qvalue(expert.down, expert.down_scales,
                          out, in, intermediate);
        output[out] = sum;
    }
}

static int compare_output(const char *name, const std::vector<float> &got,
                          const std::vector<float> &want,
                          float max_rel, float min_cosine) {
    double diff2 = 0.0, got2 = 0.0, want2 = 0.0, dot = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        if (!std::isfinite(got[i]) || !std::isfinite(want[i])) return 0;
        double diff = (double)got[i] - want[i];
        diff2 += diff * diff;
        got2 += (double)got[i] * got[i];
        want2 += (double)want[i] * want[i];
        dot += (double)got[i] * want[i];
    }
    double relative = std::sqrt(diff2 / (want2 + 1e-30));
    double cosine = dot / std::sqrt((got2 + 1e-30) * (want2 + 1e-30));
    std::printf("  %-34s rel_l2 %.3e cosine %.8f\n",
                name, relative, cosine);
    return check(relative <= max_rel && cosine >= min_cosine, name);
}

static int test_moe_site(ColiGpuContext *ctx) {
    const int rows = 9, hidden = 65, intermediate = 67;
    const int experts = 72, topk = 8;
    ColiGpuExpertCacheConfig cache_config = {};
    cache_config.experts = experts;
    cache_config.slots = experts;
    cache_config.hidden = hidden;
    cache_config.intermediate = intermediate;
    cache_config.group_size = 64;
    cache_config.max_rows = rows;
    cache_config.swiglu_limit = 0.35f;
    ColiGpuExpertCache *cache = nullptr;
    if (!coli_gpu_expert_cache_create(&cache, ctx, &cache_config)) return 0;
    std::vector<QuantizedExpert> expert_bytes;
    expert_bytes.reserve(experts);
    std::vector<ColiGpuExpertHandle> by_expert(experts);
    for (int expert = 0; expert < experts; ++expert) {
        expert_bytes.emplace_back(hidden, intermediate, expert + 19);
        ColiGpuExpertSource source =
            expert_bytes.back().source(hidden, intermediate);
        if (!coli_gpu_expert_cache_upload(
                cache, expert, expert, &source, &by_expert[expert]))
            return 0;
    }
    QuantizedExpert shared_bytes(hidden, intermediate, 911);
    ColiGpuTensor *shared_gate = quant_tensor(
        ctx, shared_bytes.gate.data(), shared_bytes.gate_scales.data(),
        intermediate, hidden);
    ColiGpuTensor *shared_up = quant_tensor(
        ctx, shared_bytes.up.data(), shared_bytes.up_scales.data(),
        intermediate, hidden);
    ColiGpuTensor *shared_down = quant_tensor(
        ctx, shared_bytes.down.data(), shared_bytes.down_scales.data(),
        hidden, intermediate);
    ColiGpuMoeSharedWeights shared = {
        shared_gate, shared_up, shared_down
    };

    std::vector<float> input((size_t)rows * hidden, 0.0f);
    for (int row = 0; row < rows; ++row) {
        input[(size_t)row * hidden + row] = 8.0f;
        input[(size_t)row * hidden + 64] = row * 0.3f - 1.0f;
    }
    std::vector<float> route_weight((size_t)experts * hidden, 0.0f);
    std::vector<float> route_bias(experts, 0.0f);
    for (int row = 0; row < rows; ++row)
        for (int k = 0; k < topk; ++k)
            route_weight[(size_t)(row * topk + k) * hidden + row] =
                2.0f - k * 0.05f;
    ColiGpuTensor *rw = tensor(ctx, route_weight.data(), experts, hidden);
    ColiGpuTensor *rb = tensor(ctx, route_bias.data(), 1, experts);
    ColiGpuRouteConfig route_config = {
        hidden, experts, topk, 1, 2.5f
    };
    ColiGpuRouter *router = nullptr;
    size_t input_offset = 0;
    size_t output_offset = ((size_t)rows * hidden * sizeof(float) + 255u) &
                           ~size_t(255u);
    size_t scratch_offset = output_offset +
        (((size_t)rows * hidden * sizeof(float) + 255u) & ~size_t(255u));
    size_t capacity = scratch_offset +
        coli_gpu_moe_scratch_bytes(&cache_config);
    ColiGpuArena *arena = nullptr;
    if (!shared_gate || !shared_up || !shared_down || !rw || !rb ||
        !coli_gpu_router_create(&router, ctx, &route_config, rows) ||
        !capacity || !coli_gpu_arena_create(&arena, ctx, capacity) ||
        !coli_gpu_arena_upload_activation(
            arena, input_offset, input.data(),
            input.size() * sizeof(float)) ||
        !coli_gpu_router_run(router, arena, input_offset, rw, rb, rows))
        return 0;
    std::vector<int> ids((size_t)rows * topk);
    std::vector<float> weights(ids.size());
    if (!coli_gpu_router_download(
            router, ids.data(), weights.data(), ids.size(), rows))
        return 0;
    std::vector<unsigned char> seen(experts, 0);
    std::vector<ColiGpuExpertHandle> selected(ids.size());
    int distinct = 0;
    for (size_t index = 0; index < ids.size(); ++index) {
        selected[index] = by_expert[ids[index]];
        if (!seen[ids[index]]) {
            seen[ids[index]] = 1;
            ++distinct;
        }
    }
    if (!check(distinct > 64, "routing union spans more than 64 experts"))
        return 0;

    int delayed_expert_id = ids[0];
    expert_bytes[delayed_expert_id] =
        QuantizedExpert(hidden, intermediate, 2401);
    std::vector<float> shared_want(input.size()), routed_want(input.size(), 0.0f);
    std::vector<float> combined_want(input.size()), temporary(hidden);
    for (int row = 0; row < rows; ++row) {
        cpu_expert(shared_want.data() + (size_t)row * hidden,
                   input.data() + (size_t)row * hidden, shared_bytes,
                   hidden, intermediate, cache_config.swiglu_limit);
        for (int k = 0; k < topk; ++k) {
            size_t route = (size_t)row * topk + k;
            cpu_expert(temporary.data(), input.data() + (size_t)row * hidden,
                       expert_bytes[ids[route]], hidden, intermediate,
                       cache_config.swiglu_limit);
            for (int column = 0; column < hidden; ++column)
                routed_want[(size_t)row * hidden + column] +=
                    weights[route] * temporary[column];
        }
        for (int column = 0; column < hidden; ++column)
            combined_want[(size_t)row * hidden + column] =
                shared_want[(size_t)row * hidden + column] +
                routed_want[(size_t)row * hidden + column];
    }
    auto run = [&](const char *name, const ColiGpuMoeSharedWeights *shared_arg,
                   ColiGpuRouter *router_arg,
                   const ColiGpuExpertHandle *handles, size_t handle_count,
                   const std::vector<float> &want) {
        std::vector<float> got(input.size());
        if (!coli_gpu_moe_site(
                arena, output_offset, input_offset, scratch_offset,
                router_arg, cache, shared_arg, handles, handle_count, rows) ||
            !coli_gpu_arena_download_activation(
                arena, output_offset, got.data(), got.size() * sizeof(float)))
            return false;
        return compare_output(name, got, want, 1e-3f, 0.9999f) != 0;
    };
    ColiGpuExpertSource delayed_source =
        expert_bytes[delayed_expert_id].source(hidden, intermediate);
    std::vector<float> primitive_got(input.size());
    if (!coli_gpu_expert_primitive(
            arena, output_offset, input_offset, scratch_offset, cache,
            &shared, rows) ||
        !coli_gpu_arena_download_activation(
            arena, output_offset, primitive_got.data(),
            primitive_got.size() * sizeof(float)) ||
        !compare_output("standalone expert primitive", primitive_got,
                        shared_want, 3e-4f, 0.99999f))
        return 0;
    std::vector<float> primitive_nonfinite = input;
    primitive_nonfinite[0] = NAN;
    if (!coli_gpu_arena_upload_activation(
            arena, input_offset, primitive_nonfinite.data(),
            primitive_nonfinite.size() * sizeof(float)) ||
        coli_gpu_expert_primitive(
            arena, output_offset, input_offset, scratch_offset, cache,
            &shared, rows) ||
        !coli_gpu_arena_upload_activation(
            arena, input_offset, input.data(), input.size() * sizeof(float)))
        return 0;
    if (!coli_gpu_expert_cache_test_delay_upload(cache, 200))
        return 0;
    auto upload_start = std::chrono::steady_clock::now();
    ColiGpuExpertHandle delayed_handle = {};
    if (!coli_gpu_expert_cache_upload(
            cache, delayed_expert_id, delayed_expert_id,
            &delayed_source, &delayed_handle))
        return 0;
    auto upload_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - upload_start).count();
    if (!check(upload_ms < 150,
               "upload publication does not synchronize delayed DMA"))
        return 0;
    by_expert[delayed_expert_id] = delayed_handle;
    for (size_t index = 0; index < selected.size(); ++index)
        if (selected[index].expert_id == delayed_expert_id)
            selected[index] = delayed_handle;
    if (!run("immediate upload wait-before-use", nullptr, router,
             selected.data(), selected.size(), routed_want))
        return 0;

    ColiGpuTelemetry before = {}, after = {};
    coli_gpu_context_telemetry(ctx, &before);
    int ok = run("shared-only complete site", &shared, nullptr,
                 nullptr, 0, shared_want) &&
             run("routed-only complete site", nullptr, router,
                 selected.data(), selected.size(), routed_want) &&
             run("combined complete site", &shared, router,
                 selected.data(), selected.size(), combined_want);
    coli_gpu_context_telemetry(ctx, &after);
    ok &= check(after.device_allocations == before.device_allocations,
                "MoE hot path performs no allocation");
    ok &= check(after.host_activation_h2d_copies ==
                    before.host_activation_h2d_copies,
                "MoE performs no activation H2D copy");
    ok &= check(before.host_activation_h2d_copies >= 1,
                "input boundary activation H2D telemetry");
    ok &= check(after.host_activation_d2h_copies >
                    before.host_activation_d2h_copies,
                "output boundary activation D2H telemetry");

    ColiGpuTelemetry site_before = {}, site_after = {};
    coli_gpu_context_telemetry(ctx, &site_before);
    ok &= coli_gpu_moe_site(
        arena, output_offset, input_offset, scratch_offset, router, cache,
        &shared, selected.data(), selected.size(), rows);
    coli_gpu_context_telemetry(ctx, &site_after);
    ok &= check(site_after.host_activation_h2d_copies ==
                    site_before.host_activation_h2d_copies &&
                site_after.host_activation_d2h_copies ==
                    site_before.host_activation_d2h_copies,
                "MoE compute has no host activation copies");
    ok &= check(site_after.device_allocations ==
                    site_before.device_allocations &&
                site_after.h2d_copies == site_before.h2d_copies &&
                site_after.d2h_copies == site_before.d2h_copies,
                "MoE cache-hit path has zero allocation/H2D/D2H");

    std::vector<float> first(input.size()), second(input.size());
    ok &= coli_gpu_moe_site(
        arena, output_offset, input_offset, scratch_offset, router, cache,
        &shared, selected.data(), selected.size(), rows);
    ok &= coli_gpu_arena_download_activation(
        arena, output_offset, first.data(), first.size() * sizeof(float));
    ok &= coli_gpu_moe_site(
        arena, output_offset, input_offset, scratch_offset, router, cache,
        &shared, selected.data(), selected.size(), rows);
    ok &= coli_gpu_arena_download_activation(
        arena, output_offset, second.data(), second.size() * sizeof(float));
    ok &= check(!std::memcmp(first.data(), second.data(),
                            first.size() * sizeof(float)),
                "MoE output is bitwise deterministic");

    ColiGpuExpertSlotInfo publication = {}, unchanged = {};
    coli_gpu_expert_cache_slot_info(cache, 0, &publication);
    auto prelaunch_failure = [&](ColiGpuExpertFaultPoint point,
                                 int occurrence, const char *name) {
        std::vector<float> sentinel(input.size(), 123.25f);
        std::vector<float> observed(input.size());
        if (!coli_gpu_arena_upload_activation(
                arena, output_offset, sentinel.data(),
                sentinel.size() * sizeof(float)))
            return false;
        ColiGpuTelemetry failure_before = {}, failure_after = {};
        coli_gpu_context_telemetry(ctx, &failure_before);
        if (!coli_gpu_expert_cache_inject_fault_at(
                cache, point, occurrence) ||
            coli_gpu_moe_site(
                arena, output_offset, input_offset, scratch_offset,
                router, cache, &shared, selected.data(), selected.size(),
                rows))
            return false;
        coli_gpu_context_telemetry(ctx, &failure_after);
        uint64_t expected_wait_calls =
            point == COLI_GPU_EXPERT_FAULT_EVENT_WAIT
                ? (uint64_t)occurrence
                : (uint64_t)selected.size();
        int telemetry_ok =
            failure_after.expert_event_wait_calls ==
                failure_before.expert_event_wait_calls +
                    expected_wait_calls &&
            failure_after.expert_event_wait_failures ==
                failure_before.expert_event_wait_failures +
                    (point == COLI_GPU_EXPERT_FAULT_EVENT_WAIT ? 1u : 0u) &&
            failure_after.moe_compute_launches ==
                failure_before.moe_compute_launches &&
            failure_after.device_allocations ==
                failure_before.device_allocations &&
            failure_after.h2d_copies == failure_before.h2d_copies &&
            failure_after.d2h_copies == failure_before.d2h_copies &&
            failure_after.host_activation_h2d_copies ==
                failure_before.host_activation_h2d_copies &&
            failure_after.host_activation_d2h_copies ==
                failure_before.host_activation_d2h_copies;
        if (!coli_gpu_arena_download_activation(
                arena, output_offset, observed.data(),
                observed.size() * sizeof(float)))
            return false;
        return check(telemetry_ok &&
                         !std::memcmp(
                             sentinel.data(), observed.data(),
                             sentinel.size() * sizeof(float)),
                     name) != 0;
    };
    ok &= prelaunch_failure(
        COLI_GPU_EXPERT_FAULT_EVENT_WAIT, 2,
        "combined wait failure has no compute/output side effect");
    ok &= prelaunch_failure(
        COLI_GPU_EXPERT_FAULT_LAUNCH, 0,
        "combined launch failure has no compute/output side effect");

    ColiGpuExpertHandle wrong = selected[0];
    selected[0] = by_expert[1];
    ok &= !coli_gpu_moe_site(
        arena, output_offset, input_offset, scratch_offset, router, cache,
        nullptr, selected.data(), selected.size(), rows);
    selected[0] = wrong;
    coli_gpu_expert_cache_slot_info(cache, 0, &unchanged);
    ok &= check(!std::memcmp(&publication, &unchanged, sizeof(publication)),
                "event-wait failure preserves publication");

    QuantizedExpert concurrent_bytes(hidden, intermediate, 2501);
    ColiGpuExpertSource concurrent_source =
        concurrent_bytes.source(hidden, intermediate);
    ColiGpuExpertHandle concurrent_handle = {};
    std::atomic<int> site_result{0};
    std::atomic<int> upload_result{0};
    ok &= coli_gpu_expert_cache_test_hold_snapshot(cache, 1);
    std::thread site_thread([&]() {
        site_result = coli_gpu_moe_site(
            arena, output_offset, input_offset, scratch_offset, router, cache,
            nullptr, selected.data(), selected.size(), rows);
    });
    while (!coli_gpu_expert_cache_test_snapshot_entered(cache))
        std::this_thread::yield();
    std::thread upload_thread([&]() {
        upload_result = coli_gpu_expert_cache_upload(
            cache, delayed_expert_id, delayed_expert_id,
            &concurrent_source, &concurrent_handle);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    ok &= coli_gpu_expert_cache_test_hold_snapshot(cache, 0);
    site_thread.join();
    upload_thread.join();
    std::vector<float> concurrent_got(input.size());
    ok &= site_result.load() && upload_result.load() &&
          coli_gpu_arena_download_activation(
              arena, output_offset, concurrent_got.data(),
              concurrent_got.size() * sizeof(float));
    ok &= compare_output("generation-bound concurrent snapshot",
                         concurrent_got, routed_want, 1e-3f, 0.9999f);

    std::vector<float> bad = input;
    bad[0] = NAN;
    ok &= coli_gpu_arena_upload_activation(
        arena, input_offset, bad.data(), bad.size() * sizeof(float));
    ok &= !coli_gpu_router_run(router, arena, input_offset, rw, rb, rows);

    QuantizedExpert nonfinite_output(hidden, intermediate, 1700);
    for (float &scale : nonfinite_output.down_scales) scale = FLT_MAX;
    ColiGpuExpertSource nonfinite_source =
        nonfinite_output.source(hidden, intermediate);
    ColiGpuExpertHandle overflow_handle = {};
    ok &= coli_gpu_expert_cache_upload(
        cache, 0, 0, &nonfinite_source, &overflow_handle);
    selected[0] = overflow_handle;
    ok &= coli_gpu_arena_upload_activation(
        arena, input_offset, input.data(), input.size() * sizeof(float));
    ok &= coli_gpu_router_run(router, arena, input_offset, rw, rb, rows);
    ok &= !coli_gpu_moe_site(
        arena, output_offset, input_offset, scratch_offset, router, cache,
        nullptr, selected.data(), selected.size(), rows);

    coli_gpu_arena_destroy(arena);
    coli_gpu_router_destroy(router);
    coli_gpu_tensor_destroy(rb);
    coli_gpu_tensor_destroy(rw);
    coli_gpu_tensor_destroy(shared_down);
    coli_gpu_tensor_destroy(shared_up);
    coli_gpu_tensor_destroy(shared_gate);
    coli_gpu_expert_cache_destroy(cache);
    return ok;
}

static ColiGlm53GpuWeightDesc model_weight(
    const std::vector<unsigned char> &data,
    const std::vector<float> &scales, int rows, int columns) {
    ColiGlm53GpuWeightDesc desc = {};
    desc.data = data.data();
    desc.scales = scales.data();
    desc.format = 4;
    desc.rows = rows;
    desc.columns = columns;
    desc.group_size = 64;
    return desc;
}

static int test_model_session_ownership(ColiGpuContext *ctx) {
    const int hidden = 65, intermediate = 67, experts = 288, topk = 8;
    float embedding[65] = {};
    float final_norm[65];
    float lm_head[65] = {};
    for (float &value : final_norm) value = 1.0f;
    std::vector<float> route_weight((size_t)experts * hidden, 0.0f);
    std::vector<float> route_bias(experts, 0.0f);
    QuantizedExpert shared(hidden, intermediate, 1234);
    ColiGlm53GpuMoeLayerDesc layer = {};
    layer.router = route_weight.data();
    layer.correction_bias = route_bias.data();
    layer.shared_gate = model_weight(
        shared.gate, shared.gate_scales, intermediate, hidden);
    layer.shared_up = model_weight(
        shared.up, shared.up_scales, intermediate, hidden);
    layer.shared_down = model_weight(
        shared.down, shared.down_scales, hidden, intermediate);
    ColiGlm53GpuModelDesc desc = {};
    desc.hidden_size = hidden;
    desc.stream_count = 1;
    desc.vocab_size = 1;
    desc.max_prefill_rows = 1;
    desc.max_context_tokens = 1;
    desc.norm_eps = 1e-6f;
    desc.hc_eps = 1e-6f;
    desc.embedding = embedding;
    desc.final_norm = final_norm;
    desc.lm_head = lm_head;
    desc.moe_experts = experts;
    desc.moe_topk = topk;
    desc.moe_intermediate = intermediate;
    desc.moe_cache_slots = topk;
    desc.moe_group_size = 64;
    desc.moe_normalize_topk = 1;
    desc.moe_routed_scale = 2.5f;
    desc.moe_swiglu_limit = 0.35f;
    desc.moe_layers = &layer;
    desc.moe_layer_count = 1;
    ColiGlm53GpuModelDesc invalid_desc = desc;
    invalid_desc.moe_swiglu_limit = 0.0f;
    ColiGlm53GpuModel *invalid_model = nullptr;
    if (coli_glm53_gpu_model_create(&invalid_model, ctx, &invalid_desc) ||
        invalid_model) {
        coli_glm53_gpu_model_destroy(invalid_model);
        std::fprintf(stderr, "FAIL: GLM model accepted zero SwiGLU limit\n");
        return 0;
    }
    ColiGlm53GpuModel *model = nullptr;
    ColiGlm53GpuSession *session = nullptr;
    if (!coli_glm53_gpu_model_create(&model, ctx, &desc) ||
        !coli_glm53_gpu_session_create(&session, model, 1))
        return 0;
    std::vector<ColiGlm53GpuExpertHandle> handles(topk);
    for (int expert = 0; expert < topk; ++expert) {
        QuantizedExpert bytes(hidden, intermediate, 1300 + expert);
        ColiGlm53GpuExpertDesc expert_desc = {};
        expert_desc.gate = model_weight(
            bytes.gate, bytes.gate_scales, intermediate, hidden);
        expert_desc.up = model_weight(
            bytes.up, bytes.up_scales, intermediate, hidden);
        expert_desc.down = model_weight(
            bytes.down, bytes.down_scales, hidden, intermediate);
        if (!coli_glm53_gpu_model_moe_upload_expert(
                model, 0, expert, expert, &expert_desc, &handles[expert]))
            return 0;
    }
    std::vector<float> input(hidden, 0.0f), output(hidden);
    input[0] = 1.0f;
    int selected[topk];
    float weights[topk];
    int ok = coli_glm53_gpu_session_moe_upload_input(
                 session, input.data(), 1) &&
             coli_glm53_gpu_session_moe_route(session, 0, 1) &&
             coli_glm53_gpu_session_moe_selected(
                 session, selected, weights, topk, 1) &&
             coli_glm53_gpu_session_moe_site(
                 session, 0, handles.data(), handles.size(), 1) &&
             coli_glm53_gpu_session_moe_download_output(
                 session, output.data(), 1);
    for (int k = 0; ok && k < topk; ++k)
        ok &= selected[k] == k;
    coli_glm53_gpu_session_destroy(session);
    coli_glm53_gpu_model_destroy(model);
    return check(ok, "GLM model/session owns production 288/top-8 MoE");
}

int main(void) {
    ColiGpuContext *ctx = nullptr;
    if (!coli_gpu_context_create(&ctx, 0)) return 1;
    int ok = test_router(ctx) && test_nonfinite_descriptors(ctx) &&
             test_cache(ctx) && test_moe_site(ctx) &&
             test_model_session_ownership(ctx);
    coli_gpu_context_destroy(ctx);
    if (!check(ok, "device-resident router suite")) return 1;
    std::puts("glm53 moe cuda tests: PASS");
    return 0;
}
