#include "../backend_cuda.h"
#include "../glm53_gpu.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static uint64_t rng_state = 0x243f6a8885a308d3ULL;

static float random_float(float scale = 1.0f) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return scale * ((float)((rng_state >> 40) & 0xffffff) / 8388608.0f - 1.0f);
}

static size_t align256(size_t value) {
    return (value + 255u) & ~size_t(255u);
}

static int compare_vectors(const char *name, const float *got, const float *want,
                           size_t count, float max_rel_l2, float min_cosine) {
    double diff2 = 0.0, want2 = 0.0, got2 = 0.0, dot = 0.0;
    for (size_t i = 0; i < count; ++i) {
        if (!std::isfinite(got[i]) || !std::isfinite(want[i])) {
            std::fprintf(stderr, "%s has non-finite value at %zu\n", name, i);
            return 0;
        }
        double d = (double)got[i] - want[i];
        diff2 += d * d;
        want2 += (double)want[i] * want[i];
        got2 += (double)got[i] * got[i];
        dot += (double)got[i] * want[i];
    }
    double rel = std::sqrt(diff2 / (want2 + 1e-30));
    double cosine = (got2 < 1e-30 && want2 < 1e-30)
        ? 1.0 : dot / std::sqrt((got2 + 1e-30) * (want2 + 1e-30));
    if (!std::isfinite(rel) || !std::isfinite(cosine)) {
        std::fprintf(stderr, "%s produced non-finite metrics\n", name);
        return 0;
    }
    std::printf("  %-28s rel_l2 %.3e cosine %.8f\n", name, rel, cosine);
    if (rel > max_rel_l2 || cosine < min_cosine) {
        std::fprintf(stderr, "%s outside tolerance\n", name);
        return 0;
    }
    return 1;
}

static int test_comparator_rejects_nonfinite(void) {
    const float finite[2] = {1.0f, -2.0f};
    const float nan_value[2] = {1.0f, NAN};
    const float inf_value[2] = {1.0f, INFINITY};
    if (compare_vectors("NaN rejection probe", nan_value, finite, 2,
                        3e-4f, 0.99999f) ||
        compare_vectors("Inf rejection probe", inf_value, finite, 2,
                        3e-4f, 0.99999f)) {
        std::fprintf(stderr, "numeric comparator accepted non-finite output\n");
        return 0;
    }
    return 1;
}

static void cpu_rmsnorm(float *out, const float *input, const float *weight,
                        int rows, int hidden, float eps) {
    for (int row = 0; row < rows; ++row) {
        const float *x = input + (size_t)row * hidden;
        float *y = out + (size_t)row * hidden;
        float square = 0.0f;
        for (int d = 0; d < hidden; ++d) square += x[d] * x[d];
        float inverse = 1.0f / std::sqrt(square / hidden + eps);
        for (int d = 0; d < hidden; ++d) y[d] = x[d] * inverse * weight[d];
    }
}

static void cpu_layernorm(float *out, const float *input, const float *weight,
                          const float *bias, int rows, int hidden, float eps) {
    for (int row = 0; row < rows; ++row) {
        const float *x = input + (size_t)row * hidden;
        float *y = out + (size_t)row * hidden;
        float mean = 0.0f;
        for (int d = 0; d < hidden; ++d) mean += x[d];
        mean /= hidden;
        float variance = 0.0f;
        for (int d = 0; d < hidden; ++d) {
            float centered = x[d] - mean;
            variance += centered * centered;
        }
        variance /= hidden;
        float inverse = 1.0f / std::sqrt(variance + eps);
        for (int d = 0; d < hidden; ++d)
            y[d] = (x[d] - mean) * inverse * weight[d] + bias[d];
    }
}

static float cpu_sigmoid(float value) {
    if (value >= 0.0f) {
        float decay = std::exp(-value);
        return 1.0f / (1.0f + decay);
    }
    float growth = std::exp(value);
    return growth / (1.0f + growth);
}

static int cpu_mhc_site(float *output, float *collapsed, float *normed,
                        const float *residual, const float *branch,
                        const float *fn, const float *scale, const float *base,
                        const float *norm_weight, int rows, int streams,
                        int hidden, float norm_eps, float hc_eps) {
    std::vector<float> post((size_t)rows * streams);
    std::vector<float> comb((size_t)rows * streams * streams);
    std::vector<float> mixes((size_t)(2 + streams) * streams);
    std::vector<float> pre(streams);
    for (int row = 0; row < rows; ++row) {
        const float *input = residual + (size_t)row * streams * hidden;
        float square = 0.0f;
        for (int i = 0; i < streams * hidden; ++i) square += input[i] * input[i];
        float inverse = 1.0f / std::sqrt(square / (streams * hidden) + norm_eps);
        int mix_count = (2 + streams) * streams;
        for (int mix = 0; mix < mix_count; ++mix) {
            float sum = 0.0f;
            for (int i = 0; i < streams * hidden; ++i)
                sum += fn[(size_t)mix * streams * hidden + i] * input[i];
            mixes[mix] = sum * inverse;
        }
        float *row_post = post.data() + (size_t)row * streams;
        float *row_comb = comb.data() + (size_t)row * streams * streams;
        for (int i = 0; i < streams; ++i) {
            pre[i] = cpu_sigmoid(mixes[i] * scale[0] + base[i]) + hc_eps;
            row_post[i] = 2.0f * cpu_sigmoid(
                mixes[streams + i] * scale[1] + base[streams + i]);
        }
        int matrix_offset = 2 * streams;
        for (int r = 0; r < streams; ++r) {
            float maximum = -INFINITY;
            for (int c = 0; c < streams; ++c) {
                int index = matrix_offset + r * streams + c;
                row_comb[r * streams + c] =
                    mixes[index] * scale[2] + base[index];
                maximum = std::fmax(maximum, row_comb[r * streams + c]);
            }
            float sum = 0.0f;
            for (int c = 0; c < streams; ++c) {
                row_comb[r * streams + c] =
                    std::exp(row_comb[r * streams + c] - maximum);
                sum += row_comb[r * streams + c];
            }
            for (int c = 0; c < streams; ++c)
                row_comb[r * streams + c] =
                    row_comb[r * streams + c] / sum + hc_eps;
        }
        for (int c = 0; c < streams; ++c) {
            float sum = 0.0f;
            for (int r = 0; r < streams; ++r)
                sum += row_comb[r * streams + c];
            for (int r = 0; r < streams; ++r)
                row_comb[r * streams + c] /= sum + hc_eps;
        }
        for (int iteration = 1; iteration < 20; ++iteration) {
            for (int r = 0; r < streams; ++r) {
                float sum = 0.0f;
                for (int c = 0; c < streams; ++c)
                    sum += row_comb[r * streams + c];
                for (int c = 0; c < streams; ++c)
                    row_comb[r * streams + c] /= sum + hc_eps;
            }
            for (int c = 0; c < streams; ++c) {
                float sum = 0.0f;
                for (int r = 0; r < streams; ++r)
                    sum += row_comb[r * streams + c];
                for (int r = 0; r < streams; ++r)
                    row_comb[r * streams + c] /= sum + hc_eps;
            }
        }
        for (int d = 0; d < hidden; ++d) {
            float sum = 0.0f;
            for (int stream = 0; stream < streams; ++stream)
                sum += pre[stream] * input[(size_t)stream * hidden + d];
            collapsed[(size_t)row * hidden + d] = sum;
        }
    }
    cpu_rmsnorm(normed, collapsed, norm_weight, rows, hidden, norm_eps);
    for (int row = 0; row < rows; ++row) {
        for (int destination = 0; destination < streams; ++destination)
            for (int d = 0; d < hidden; ++d) {
                float value = 0.0f;
                for (int source = 0; source < streams; ++source)
                    value += comb[((size_t)row * streams + source) * streams +
                                  destination] *
                             residual[((size_t)row * streams + source) * hidden + d];
                value += post[(size_t)row * streams + destination] *
                         branch[(size_t)row * hidden + d];
                output[((size_t)row * streams + destination) * hidden + d] = value;
            }
    }
    return 1;
}

static ColiGpuTensor *upload_f32(ColiGpuContext *ctx, const float *data,
                                 int rows, int columns) {
    ColiGpuTensorDesc desc = {};
    desc.data = data;
    desc.format = 0;
    desc.rows = rows;
    desc.columns = columns;
    ColiGpuTensor *tensor = nullptr;
    return coli_gpu_tensor_create(&tensor, ctx, &desc) ? tensor : nullptr;
}

static int test_norms(ColiGpuContext *ctx) {
    const int rows = 5, hidden = 37;
    const size_t values = (size_t)rows * hidden;
    std::vector<float> input(values), weight(hidden), bias(hidden);
    std::vector<float> got(values), want(values);
    for (size_t i = 0; i < values; ++i) input[i] = random_float(3.0f);
    for (int i = 0; i < hidden; ++i) {
        weight[i] = 1.0f + random_float(0.2f);
        bias[i] = random_float(0.1f);
    }
    ColiGpuTensor *w = upload_f32(ctx, weight.data(), 1, hidden);
    ColiGpuTensor *b = upload_f32(ctx, bias.data(), 1, hidden);
    ColiGpuArena *arena = nullptr;
    size_t input_offset = 0;
    size_t output_offset = align256(values * sizeof(float));
    size_t capacity = output_offset + values * sizeof(float);
    if (!w || !b || !coli_gpu_arena_create(&arena, ctx, capacity)) return 0;
    if (!coli_gpu_arena_upload(arena, input_offset, input.data(), values * sizeof(float)))
        return 0;

    cpu_rmsnorm(want.data(), input.data(), weight.data(), rows, hidden, 1e-6f);
    if (!coli_gpu_rmsnorm(arena, output_offset, input_offset, w, rows, hidden, 1e-6f) ||
        !coli_gpu_arena_download(arena, output_offset, got.data(), values * sizeof(float)) ||
        !compare_vectors("RMSNorm random row tail", got.data(), want.data(), values,
                         3e-4f, 0.99999f))
        return 0;

    std::fill(input.begin(), input.end(), 0.0f);
    std::fill(want.begin(), want.end(), 0.0f);
    if (!coli_gpu_arena_upload(arena, input_offset, input.data(), values * sizeof(float)) ||
        !coli_gpu_rmsnorm(arena, output_offset, input_offset, w, rows, hidden, 1e-6f) ||
        !coli_gpu_arena_download(arena, output_offset, got.data(), values * sizeof(float)) ||
        std::memcmp(got.data(), want.data(), values * sizeof(float)) != 0) {
        std::fprintf(stderr, "RMSNorm zero input mismatch\n");
        return 0;
    }

    for (size_t i = 0; i < values; ++i) input[i] = random_float(2.0f);
    cpu_layernorm(want.data(), input.data(), weight.data(), bias.data(),
                  rows, hidden, 1e-5f);
    if (!coli_gpu_arena_upload(arena, input_offset, input.data(), values * sizeof(float)) ||
        !coli_gpu_layernorm(arena, output_offset, input_offset, w, b,
                            rows, hidden, 1e-5f) ||
        !coli_gpu_arena_download(arena, output_offset, got.data(), values * sizeof(float)) ||
        !compare_vectors("affine LayerNorm", got.data(), want.data(), values,
                         3e-4f, 0.99999f))
        return 0;

    coli_gpu_arena_destroy(arena);
    coli_gpu_tensor_destroy(b);
    coli_gpu_tensor_destroy(w);
    return 1;
}

struct MhcFixture {
    int rows, streams, hidden, mix_count;
    std::vector<float> residual, branch, fn, scale, base, norm;
    std::vector<float> output, collapsed, normed;

    MhcFixture(int row_count, int stream_count, int hidden_size)
        : rows(row_count), streams(stream_count), hidden(hidden_size),
          mix_count((2 + streams) * streams),
          residual((size_t)rows * streams * hidden),
          branch((size_t)rows * hidden),
          fn((size_t)mix_count * streams * hidden),
          scale(3), base(mix_count), norm(hidden),
          output(residual.size()), collapsed((size_t)rows * hidden),
          normed((size_t)rows * hidden) {
        for (float &value : residual) value = random_float(0.8f);
        for (float &value : branch) value = random_float(0.6f);
        for (float &value : fn) value = random_float(0.02f);
        for (float &value : base) value = random_float(0.2f);
        for (float &value : norm) value = 1.0f + random_float(0.1f);
        scale[0] = 0.7f;
        scale[1] = -0.4f;
        scale[2] = 0.3f;
    }
};

struct MhcArenaLayout {
    size_t residual, branch, output, collapsed, normed, post, comb, capacity;

    MhcArenaLayout(int rows, int streams, int hidden) {
        residual = 0;
        branch = align256((size_t)rows * streams * hidden * sizeof(float));
        output = branch + align256((size_t)rows * hidden * sizeof(float));
        collapsed = output + align256((size_t)rows * streams * hidden * sizeof(float));
        normed = collapsed + align256((size_t)rows * hidden * sizeof(float));
        post = normed + align256((size_t)rows * hidden * sizeof(float));
        comb = post + align256((size_t)rows * streams * sizeof(float));
        capacity = comb + (size_t)rows * streams * streams * sizeof(float);
    }
};

static int run_mhc_case(ColiGpuContext *ctx, int rows, int hidden,
                        const char *name, bool zero_input = false) {
    const int streams = 4;
    MhcFixture f(rows, streams, hidden);
    if (zero_input) {
        std::fill(f.residual.begin(), f.residual.end(), 0.0f);
        std::fill(f.branch.begin(), f.branch.end(), 0.0f);
    }
    MhcArenaLayout layout(rows, streams, hidden);
    if (!cpu_mhc_site(f.output.data(), f.collapsed.data(), f.normed.data(),
                      f.residual.data(), f.branch.data(), f.fn.data(),
                      f.scale.data(), f.base.data(), f.norm.data(),
                      rows, streams, hidden, 1e-6f, 1e-6f))
        return 0;

    ColiGpuTensor *fn = upload_f32(ctx, f.fn.data(), f.mix_count,
                                   streams * hidden);
    ColiGpuTensor *scale = upload_f32(ctx, f.scale.data(), 1, 3);
    ColiGpuTensor *base = upload_f32(ctx, f.base.data(), 1, f.mix_count);
    ColiGpuTensor *norm = upload_f32(ctx, f.norm.data(), 1, hidden);
    ColiGpuArena *arena = nullptr;
    if (!fn || !scale || !base || !norm ||
        !coli_gpu_arena_create(&arena, ctx, layout.capacity))
        return 0;
    if (!coli_gpu_arena_upload(arena, layout.residual, f.residual.data(),
                               f.residual.size() * sizeof(float)) ||
        !coli_gpu_arena_upload(arena, layout.branch, f.branch.data(),
                               f.branch.size() * sizeof(float)))
        return 0;

    ColiGpuTelemetry before = {}, after = {};
    coli_gpu_context_telemetry(ctx, &before);
    if (!coli_gpu_mhc_site(arena, layout.output, layout.collapsed, layout.normed,
                           layout.post, layout.comb, layout.residual, layout.branch,
                           fn, scale, base, norm, rows, streams, hidden,
                           1e-6f, 1e-6f))
        return 0;
    coli_gpu_context_telemetry(ctx, &after);
    if (after.h2d_copies != before.h2d_copies ||
        after.d2h_copies != before.d2h_copies ||
        after.device_allocations != before.device_allocations) {
        std::fprintf(stderr, "complete mHC site transferred or allocated\n");
        return 0;
    }

    std::vector<float> got_output(f.output.size()), got_normed(f.normed.size());
    if (!coli_gpu_arena_download(arena, layout.output, got_output.data(),
                                 got_output.size() * sizeof(float)) ||
        !coli_gpu_arena_download(arena, layout.normed, got_normed.data(),
                                 got_normed.size() * sizeof(float)) ||
        !compare_vectors(name, got_output.data(), f.output.data(), f.output.size(),
                         1e-3f, 0.9999f) ||
        !compare_vectors("mHC pre + RMSNorm", got_normed.data(), f.normed.data(),
                         f.normed.size(), 3e-4f, 0.99999f))
        return 0;

    std::vector<float> repeat(got_output.size());
    if (!coli_gpu_mhc_site(arena, layout.output, layout.collapsed, layout.normed,
                           layout.post, layout.comb, layout.residual, layout.branch,
                           fn, scale, base, norm, rows, streams, hidden,
                           1e-6f, 1e-6f) ||
        !coli_gpu_arena_download(arena, layout.output, repeat.data(),
                                 repeat.size() * sizeof(float)) ||
        std::memcmp(repeat.data(), got_output.data(), repeat.size() * sizeof(float)) != 0) {
        std::fprintf(stderr, "%s is not bitwise deterministic\n", name);
        return 0;
    }

    coli_gpu_arena_destroy(arena);
    coli_gpu_tensor_destroy(norm);
    coli_gpu_tensor_destroy(base);
    coli_gpu_tensor_destroy(scale);
    coli_gpu_tensor_destroy(fn);
    return 1;
}

static void cpu_embedding(float *streams_out, const int32_t *tokens,
                          const float *embedding, int rows, int streams,
                          int hidden) {
    for (int row = 0; row < rows; ++row)
        for (int stream = 0; stream < streams; ++stream)
            std::memcpy(streams_out + ((size_t)row * streams + stream) * hidden,
                        embedding + (size_t)tokens[row] * hidden,
                        (size_t)hidden * sizeof(float));
}

static void cpu_output(float *logits, const float *streams_in,
                       const float *norm_weight, const int8_t *head,
                       const float *head_scales, int rows, int streams,
                       int hidden, int vocab, float eps) {
    std::vector<float> collapsed((size_t)rows * hidden);
    std::vector<float> normed((size_t)rows * hidden);
    for (int row = 0; row < rows; ++row)
        for (int d = 0; d < hidden; ++d) {
            float sum = 0.0f;
            for (int stream = 0; stream < streams; ++stream)
                sum += streams_in[((size_t)row * streams + stream) * hidden + d];
            collapsed[(size_t)row * hidden + d] = sum / streams;
        }
    cpu_rmsnorm(normed.data(), collapsed.data(), norm_weight, rows, hidden, eps);
    for (int row = 0; row < rows; ++row)
        for (int token = 0; token < vocab; ++token) {
            float sum = 0.0f;
            for (int d = 0; d < hidden; ++d)
                sum += normed[(size_t)row * hidden + d] *
                       head[(size_t)token * hidden + d] * head_scales[token];
            logits[(size_t)row * vocab + token] = sum;
        }
}

static int test_glm_ownership_pipeline(ColiGpuContext *ctx) {
    const int rows = 5, capacity = 7, streams = 4, hidden = 33, vocab = 29;
    const int mix_count = (2 + streams) * streams;
    std::vector<float> embedding((size_t)vocab * hidden);
    std::vector<float> norm(hidden), fn((size_t)mix_count * streams * hidden);
    std::vector<float> base(mix_count), scale = {0.4f, -0.6f, 0.2f};
    std::vector<int8_t> head((size_t)vocab * hidden);
    std::vector<float> head_scales(vocab);
    int32_t tokens[rows] = {0, 7, 28, 3, 15};
    for (float &value : embedding) value = random_float(0.7f);
    for (float &value : norm) value = 1.0f + random_float(0.1f);
    for (float &value : fn) value = random_float(0.03f);
    for (float &value : base) value = random_float(0.15f);
    for (int8_t &value : head) value = (int8_t)((int)(rng_state = rng_state * 6364136223846793005ULL + 1) % 31 - 15);
    for (float &value : head_scales) value = 0.004f + std::fabs(random_float(0.002f));

    ColiGlm53GpuMhcSiteDesc site = {};
    site.fn = fn.data();
    site.base = base.data();
    site.scale = scale.data();
    site.norm_weight = norm.data();
    ColiGlm53GpuModelDesc desc = {};
    desc.hidden_size = hidden;
    desc.stream_count = streams;
    desc.vocab_size = vocab;
    desc.max_prefill_rows = capacity;
    desc.norm_eps = 1e-6f;
    desc.hc_eps = 1e-6f;
    desc.embedding = embedding.data();
    desc.final_norm = norm.data();
    desc.lm_head = head.data();
    desc.lm_head_scales = head_scales.data();
    desc.lm_head_format = 1;
    desc.sites = &site;
    desc.site_count = 1;

    ColiGlm53GpuModel *model = nullptr;
    ColiGlm53GpuSession *session = nullptr;
    if (!coli_glm53_gpu_model_create(&model, ctx, &desc) ||
        !coli_glm53_gpu_session_create(&session, model, capacity))
        return 0;

    ColiGpuTelemetry before = {}, after_embed = {}, after_site = {}, after_output = {};
    coli_gpu_context_telemetry(ctx, &before);
    if (!coli_glm53_gpu_session_embed(session, tokens, rows)) return 0;
    coli_gpu_context_telemetry(ctx, &after_embed);
    if (after_embed.h2d_copies != before.h2d_copies + 1 ||
        after_embed.h2d_bytes != before.h2d_bytes +
                                     (uint64_t)rows * sizeof(int32_t) ||
        after_embed.d2h_copies != before.d2h_copies ||
        after_embed.d2h_bytes != before.d2h_bytes ||
        after_embed.device_allocations != before.device_allocations) {
        std::fprintf(stderr, "embedding boundary telemetry mismatch\n");
        return 0;
    }
    if (!coli_glm53_gpu_session_core_site(session, 0, rows)) return 0;
    coli_gpu_context_telemetry(ctx, &after_site);
    if (std::memcmp(&after_embed, &after_site, sizeof(after_site)) != 0) {
        std::fprintf(stderr, "core site changed transfer/allocation telemetry\n");
        return 0;
    }

    std::vector<float> logits((size_t)rows * vocab);
    if (!coli_glm53_gpu_session_output(session, logits.data(), rows)) return 0;
    coli_gpu_context_telemetry(ctx, &after_output);
    if (after_output.d2h_copies != after_site.d2h_copies + 1 ||
        after_output.d2h_bytes != after_site.d2h_bytes +
                                      (uint64_t)rows * vocab * sizeof(float) ||
        after_output.h2d_copies != after_site.h2d_copies ||
        after_output.h2d_bytes != after_site.h2d_bytes ||
        after_output.device_allocations != after_site.device_allocations) {
        std::fprintf(stderr, "final-output telemetry mismatch\n");
        return 0;
    }

    std::vector<float> streams_cpu((size_t)rows * streams * hidden);
    std::vector<float> branch((size_t)rows * hidden);
    std::vector<float> next(streams_cpu.size());
    std::vector<float> collapsed((size_t)rows * hidden), normed((size_t)rows * hidden);
    std::vector<float> want_logits(logits.size());
    cpu_embedding(streams_cpu.data(), tokens, embedding.data(), rows, streams, hidden);
    if (!cpu_mhc_site(next.data(), collapsed.data(), normed.data(), streams_cpu.data(),
                      normed.data(), fn.data(), scale.data(), base.data(), norm.data(),
                      rows, streams, hidden, 1e-6f, 1e-6f))
        return 0;
    cpu_output(want_logits.data(), next.data(), norm.data(), head.data(),
               head_scales.data(), rows, streams, hidden, vocab, 1e-6f);
    if (!compare_vectors("embedding->site->LM head", logits.data(), want_logits.data(),
                         logits.size(), 1e-3f, 0.9999f))
        return 0;

    std::vector<float> repeat(logits.size());
    if (!coli_glm53_gpu_session_embed(session, tokens, rows) ||
        !coli_glm53_gpu_session_core_site(session, 0, rows) ||
        !coli_glm53_gpu_session_output(session, repeat.data(), rows) ||
        std::memcmp(repeat.data(), logits.data(), logits.size() * sizeof(float)) != 0) {
        std::fprintf(stderr, "resident pipeline is not bitwise deterministic\n");
        return 0;
    }
    if (coli_glm53_gpu_session_embed(session, tokens, capacity + 1) ||
        coli_glm53_gpu_session_core_site(session, 0, capacity + 1) ||
        coli_glm53_gpu_session_output(session, logits.data(), capacity + 1)) {
        std::fprintf(stderr, "over-capacity row count accepted\n");
        return 0;
    }

    coli_glm53_gpu_session_destroy(session);
    coli_glm53_gpu_model_destroy(model);
    return 1;
}

static float grouped_weight(const std::vector<uint8_t> &packed,
                            const std::vector<float> &scales,
                            int row, int column, int columns, int group_size) {
    size_t row_bytes = (size_t)(columns + 1) / 2;
    uint8_t byte = packed[(size_t)row * row_bytes + column / 2];
    int nibble = (column & 1) ? byte >> 4 : byte & 15;
    int value = nibble - 8;
    int groups = (columns + group_size - 1) / group_size;
    return value * scales[(size_t)row * groups + column / group_size];
}

static int test_collapse_projection_and_grouped_model(ColiGpuContext *ctx) {
    const int rows = 3, streams = 4, hidden = 35, vocab = 11, group_size = 16;
    const int groups = (hidden + group_size - 1) / group_size;
    const size_t row_bytes = (size_t)(hidden + 1) / 2;
    std::vector<float> stream_values((size_t)rows * streams * hidden);
    std::vector<float> collapsed((size_t)rows * hidden);
    std::vector<float> got_collapsed(collapsed.size());
    std::vector<float> projected((size_t)rows * vocab);
    std::vector<float> got_projected(projected.size());
    std::vector<uint8_t> packed_head((size_t)vocab * row_bytes, 0);
    std::vector<float> scales((size_t)vocab * groups);
    for (float &value : stream_values) value = random_float(0.9f);
    for (int row = 0; row < rows; ++row)
        for (int d = 0; d < hidden; ++d) {
            float sum = 0.0f;
            for (int stream = 0; stream < streams; ++stream)
                sum += stream_values[((size_t)row * streams + stream) * hidden + d];
            collapsed[(size_t)row * hidden + d] = sum / streams;
        }
    for (int token = 0; token < vocab; ++token) {
        for (int group = 0; group < groups; ++group)
            scales[(size_t)token * groups + group] =
                0.015f + 0.002f * token + 0.003f * group;
        for (int d = 0; d < hidden; ++d) {
            int signed_nibble = ((token * 5 + d * 3) % 15) - 7;
            int stored = signed_nibble + 8;
            size_t index = (size_t)token * row_bytes + d / 2;
            if (d & 1)
                packed_head[index] |= (uint8_t)((stored & 15) << 4);
            else
                packed_head[index] = (uint8_t)(stored & 15);
        }
    }
    for (int row = 0; row < rows; ++row)
        for (int token = 0; token < vocab; ++token) {
            float sum = 0.0f;
            for (int d = 0; d < hidden; ++d)
                sum += collapsed[(size_t)row * hidden + d] *
                       grouped_weight(packed_head, scales, token, d,
                                      hidden, group_size);
            projected[(size_t)row * vocab + token] = sum;
        }

    ColiGpuTensorDesc head_desc = {};
    head_desc.data = packed_head.data();
    head_desc.scales = scales.data();
    head_desc.format = 4;
    head_desc.rows = vocab;
    head_desc.columns = hidden;
    head_desc.group_size = group_size;
    ColiGpuTensor *head = nullptr;
    ColiGpuArena *arena = nullptr;
    size_t streams_offset = 0;
    size_t collapsed_offset =
        align256(stream_values.size() * sizeof(float));
    size_t projected_offset =
        collapsed_offset + align256(collapsed.size() * sizeof(float));
    size_t arena_bytes = projected_offset + projected.size() * sizeof(float);
    if (!coli_gpu_tensor_create(&head, ctx, &head_desc) ||
        !coli_gpu_arena_create(&arena, ctx, arena_bytes) ||
        !coli_gpu_arena_upload(arena, streams_offset, stream_values.data(),
                               stream_values.size() * sizeof(float)) ||
        !coli_gpu_collapse_streams(arena, collapsed_offset, streams_offset,
                                   rows, streams, hidden) ||
        !coli_gpu_arena_download(arena, collapsed_offset, got_collapsed.data(),
                                 got_collapsed.size() * sizeof(float)) ||
        !compare_vectors("four-stream collapse primitive",
                         got_collapsed.data(), collapsed.data(), collapsed.size(),
                         3e-4f, 0.99999f) ||
        !coli_gpu_projection(arena, projected_offset, collapsed_offset, head,
                             rows, hidden, vocab) ||
        !coli_gpu_arena_download(arena, projected_offset, got_projected.data(),
                                 got_projected.size() * sizeof(float)) ||
        !compare_vectors("grouped-int4 projection primitive",
                         got_projected.data(), projected.data(), projected.size(),
                         3e-4f, 0.99999f))
        return 0;
    coli_gpu_arena_destroy(arena);
    coli_gpu_tensor_destroy(head);

    std::vector<float> embedding((size_t)vocab * hidden);
    std::vector<float> final_norm(hidden);
    int32_t tokens[rows] = {0, 6, 10};
    for (float &value : embedding) value = random_float(0.8f);
    for (float &value : final_norm) value = 1.0f + random_float(0.1f);
    ColiGlm53GpuModelDesc model_desc = {};
    model_desc.hidden_size = hidden;
    model_desc.stream_count = streams;
    model_desc.vocab_size = vocab;
    model_desc.max_prefill_rows = rows;
    model_desc.norm_eps = 1e-6f;
    model_desc.hc_eps = 1e-6f;
    model_desc.embedding = embedding.data();
    model_desc.final_norm = final_norm.data();
    model_desc.lm_head = packed_head.data();
    model_desc.lm_head_scales = scales.data();
    model_desc.lm_head_format = 4;
    model_desc.lm_head_group_size = group_size;

    std::vector<float> model_streams((size_t)rows * streams * hidden);
    std::vector<float> model_collapsed((size_t)rows * hidden);
    std::vector<float> model_normed(model_collapsed.size());
    std::vector<float> want_logits((size_t)rows * vocab);
    std::vector<float> got_logits(want_logits.size());
    cpu_embedding(model_streams.data(), tokens, embedding.data(),
                  rows, streams, hidden);
    for (int row = 0; row < rows; ++row)
        for (int d = 0; d < hidden; ++d) {
            float sum = 0.0f;
            for (int stream = 0; stream < streams; ++stream)
                sum += model_streams[((size_t)row * streams + stream) * hidden + d];
            model_collapsed[(size_t)row * hidden + d] = sum / streams;
        }
    cpu_rmsnorm(model_normed.data(), model_collapsed.data(), final_norm.data(),
                rows, hidden, 1e-6f);
    for (int row = 0; row < rows; ++row)
        for (int token = 0; token < vocab; ++token) {
            float sum = 0.0f;
            for (int d = 0; d < hidden; ++d)
                sum += model_normed[(size_t)row * hidden + d] *
                       grouped_weight(packed_head, scales, token, d,
                                      hidden, group_size);
            want_logits[(size_t)row * vocab + token] = sum;
        }
    ColiGlm53GpuModel *model = nullptr;
    ColiGlm53GpuSession *session = nullptr;
    if (!coli_glm53_gpu_model_create(&model, ctx, &model_desc) ||
        !coli_glm53_gpu_session_create(&session, model, rows) ||
        !coli_glm53_gpu_session_embed(session, tokens, rows) ||
        !coli_glm53_gpu_session_output(session, got_logits.data(), rows) ||
        !compare_vectors("grouped-int4 model output tail",
                         got_logits.data(), want_logits.data(), want_logits.size(),
                         1e-3f, 0.9999f))
        return 0;
    coli_glm53_gpu_session_destroy(session);
    coli_glm53_gpu_model_destroy(model);
    return 1;
}

int main(void) {
    ColiGpuContext *ctx = nullptr;
    if (!coli_gpu_context_create(&ctx, 0)) {
        std::fprintf(stderr, "Task 2 requires a HIP device\n");
        return 1;
    }
    int ok = test_comparator_rejects_nonfinite() &&
             test_norms(ctx) &&
             run_mhc_case(ctx, 1, 1, "mHC zero/minimum shape", true) &&
             run_mhc_case(ctx, 5, 37, "mHC multi-row tail") &&
             run_mhc_case(ctx, 1, 4096, "mHC production geometry") &&
             test_glm_ownership_pipeline(ctx) &&
             test_collapse_projection_and_grouped_model(ctx);
    coli_gpu_context_destroy(ctx);
    if (!ok) return 1;
    std::puts("glm53 resident core HIP: ok");
    return 0;
}
