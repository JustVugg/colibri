#include "../backend_cuda.h"
#include "../glm53_gpu.h"

#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static uint64_t rng_state = 0x6a09e667f3bcc909ULL;

static float random_float(float scale = 1.0f) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return scale * ((float)((rng_state >> 40) & 0xffffff) / 8388608.0f - 1.0f);
}

static size_t align256(size_t value) {
    return (value + 255u) & ~size_t(255u);
}

static float sigmoidf_ref(float value) {
    if (value >= 0.0f) {
        float decay = std::exp(-value);
        return 1.0f / (1.0f + decay);
    }
    float growth = std::exp(value);
    return growth / (1.0f + growth);
}

static int compare_vectors(const char *name, const float *got, const float *want,
                           size_t count, float max_rel_l2, float min_cosine) {
    double diff2 = 0.0, got2 = 0.0, want2 = 0.0, dot = 0.0;
    for (size_t i = 0; i < count; ++i) {
        if (!std::isfinite(got[i]) || !std::isfinite(want[i])) {
            std::fprintf(stderr, "%s has non-finite value at %zu\n", name, i);
            return 0;
        }
        double diff = (double)got[i] - want[i];
        diff2 += diff * diff;
        got2 += (double)got[i] * got[i];
        want2 += (double)want[i] * want[i];
        dot += (double)got[i] * want[i];
    }
    double rel = std::sqrt(diff2 / (want2 + 1e-30));
    double cosine = (got2 < 1e-30 && want2 < 1e-30)
        ? 1.0 : dot / std::sqrt((got2 + 1e-30) * (want2 + 1e-30));
    if (!std::isfinite(rel) || !std::isfinite(cosine) ||
        rel > max_rel_l2 || cosine < min_cosine) {
        std::fprintf(stderr, "%s mismatch: rel_l2 %.3e cosine %.8f\n",
                     name, rel, cosine);
        return 0;
    }
    std::printf("  %-31s rel_l2 %.3e cosine %.8f\n", name, rel, cosine);
    return 1;
}

static void matvec(float *out, const float *weight, const float *input,
                   int rows, int columns) {
    for (int row = 0; row < rows; ++row) {
        float sum = 0.0f;
        for (int column = 0; column < columns; ++column)
            sum += weight[(size_t)row * columns + column] * input[column];
        out[row] = sum;
    }
}

struct CpuState {
    std::vector<float> matrix;
    std::vector<float> window;
};

struct Fixture {
    int rows, hidden, heads, dim, projection, kernel, max_context;
    float recurrent_eps = 1e-6f;
    float output_eps = 1e-6f;
    float gate_lower_bound = -5.0f;
    std::vector<float> input;
    std::vector<float> q, k, v, output;
    std::vector<float> gate_a, gate_b, decay_a, decay_b, beta;
    std::vector<float> conv, dt_bias, a_log, o_norm;

    Fixture(int row_count, int hidden_size, int head_count, int head_dim,
            int kernel_size, int context)
        : rows(row_count), hidden(hidden_size), heads(head_count), dim(head_dim),
          projection(heads * dim), kernel(kernel_size), max_context(context),
          input((size_t)rows * hidden),
          q((size_t)projection * hidden), k(q.size()), v(q.size()),
          output((size_t)hidden * projection),
          gate_a((size_t)dim * hidden), gate_b((size_t)projection * dim),
          decay_a((size_t)dim * hidden), decay_b((size_t)projection * dim),
          beta((size_t)heads * hidden),
          conv((size_t)3 * projection * kernel), dt_bias(projection),
          a_log(heads), o_norm(dim) {}

    void randomize(void) {
        for (float &x : input) x = random_float(0.7f);
        for (float &x : q) x = random_float(0.08f);
        for (float &x : k) x = random_float(0.08f);
        for (float &x : v) x = random_float(0.08f);
        for (float &x : output) x = random_float(0.08f);
        for (float &x : gate_a) x = random_float(0.06f);
        for (float &x : gate_b) x = random_float(0.06f);
        for (float &x : decay_a) x = random_float(0.06f);
        for (float &x : decay_b) x = random_float(0.06f);
        for (float &x : beta) x = random_float(0.06f);
        for (float &x : conv) x = random_float(0.3f);
        for (float &x : dt_bias) x = random_float(0.2f);
        for (float &x : a_log) x = random_float(0.3f);
        for (float &x : o_norm) x = 1.0f + random_float(0.15f);
    }

    void zero(void) {
        std::fill(input.begin(), input.end(), 0.0f);
        std::fill(q.begin(), q.end(), 0.0f);
        std::fill(k.begin(), k.end(), 0.0f);
        std::fill(v.begin(), v.end(), 0.0f);
        std::fill(output.begin(), output.end(), 0.0f);
        std::fill(gate_a.begin(), gate_a.end(), 0.0f);
        std::fill(gate_b.begin(), gate_b.end(), 0.0f);
        std::fill(decay_a.begin(), decay_a.end(), 0.0f);
        std::fill(decay_b.begin(), decay_b.end(), 0.0f);
        std::fill(beta.begin(), beta.end(), 0.0f);
        std::fill(conv.begin(), conv.end(), 0.0f);
        std::fill(dt_bias.begin(), dt_bias.end(), 0.0f);
        std::fill(a_log.begin(), a_log.end(), 0.0f);
        std::fill(o_norm.begin(), o_norm.end(), 0.0f);
    }
};

/* Independent transcription of glm53.c + delta_attention.h. It deliberately
 * does not call either implementation, so operation-order mistakes remain
 * observable. */
static void cpu_site(float *output, CpuState &state, const Fixture &f,
                     const float *input, int rows) {
    const int P = f.projection, H = f.heads, D = f.dim;
    std::vector<float> qkv((size_t)3 * P), decay(P), beta(H), low(D);
    std::vector<float> core(P), memory(D), normed(P), gate(P);
    for (int token = 0; token < rows; ++token) {
        const float *x = input + (size_t)token * f.hidden;
        matvec(qkv.data(), f.q.data(), x, P, f.hidden);
        matvec(qkv.data() + P, f.k.data(), x, P, f.hidden);
        matvec(qkv.data() + 2 * P, f.v.data(), x, P, f.hidden);
        matvec(low.data(), f.decay_a.data(), x, D, f.hidden);
        matvec(decay.data(), f.decay_b.data(), low.data(), P, D);
        for (int h = 0; h < H; ++h)
            for (int d = 0; d < D; ++d) {
                int i = h * D + d;
                decay[i] = f.gate_lower_bound * sigmoidf_ref(
                    std::exp(f.a_log[h]) * (decay[i] + f.dt_bias[i]));
            }
        matvec(beta.data(), f.beta.data(), x, H, f.hidden);
        for (float &value : beta) value = sigmoidf_ref(value);

        for (int channel = 0; channel < 3 * P; ++channel) {
            float *history = state.window.data() + (size_t)channel * f.kernel;
            for (int tap = 0; tap + 1 < f.kernel; ++tap)
                history[tap] = history[tap + 1];
            history[f.kernel - 1] = qkv[channel];
            float sum = 0.0f;
            for (int tap = 0; tap < f.kernel; ++tap)
                sum += f.conv[(size_t)channel * f.kernel + tap] * history[tap];
            qkv[channel] = sum / (1.0f + std::exp(-sum));
        }
        float query_scale = 1.0f / std::sqrt((float)D);
        for (int h = 0; h < H; ++h) {
            float *matrix = state.matrix.data() + (size_t)h * D * D;
            const float *query = qkv.data() + (size_t)h * D;
            const float *key = qkv.data() + P + (size_t)h * D;
            const float *value = qkv.data() + 2 * P + (size_t)h * D;
            float query_square = f.recurrent_eps;
            float key_square = f.recurrent_eps;
            for (int d = 0; d < D; ++d) {
                query_square += query[d] * query[d];
                key_square += key[d] * key[d];
            }
            float query_norm = query_scale / std::sqrt(query_square);
            float key_norm = 1.0f / std::sqrt(key_square);
            std::fill(memory.begin(), memory.end(), 0.0f);
            for (int kd = 0; kd < D; ++kd) {
                float *matrix_row = matrix + (size_t)kd * D;
                float alpha = std::exp(decay[h * D + kd]);
                float scaled_key = key[kd] * key_norm;
                for (int vd = 0; vd < D; ++vd) {
                    matrix_row[vd] *= alpha;
                    memory[vd] += scaled_key * matrix_row[vd];
                }
            }
            for (int vd = 0; vd < D; ++vd) core[h * D + vd] = 0.0f;
            for (int kd = 0; kd < D; ++kd) {
                float *matrix_row = matrix + (size_t)kd * D;
                float scaled_key = key[kd] * key_norm;
                float scaled_query = query[kd] * query_norm;
                for (int vd = 0; vd < D; ++vd) {
                    matrix_row[vd] += scaled_key *
                        (value[vd] - memory[vd]) * beta[h];
                    core[h * D + vd] += scaled_query * matrix_row[vd];
                }
            }
        }
        matvec(low.data(), f.gate_a.data(), x, D, f.hidden);
        matvec(gate.data(), f.gate_b.data(), low.data(), P, D);
        for (int h = 0; h < H; ++h) {
            float square = 0.0f;
            for (int d = 0; d < D; ++d)
                square += core[h * D + d] * core[h * D + d];
            float inverse = 1.0f / std::sqrt(square / D + f.output_eps);
            for (int d = 0; d < D; ++d)
                normed[h * D + d] = core[h * D + d] * inverse *
                    f.o_norm[d] * sigmoidf_ref(gate[h * D + d]);
        }
        matvec(output + (size_t)token * f.hidden, f.output.data(),
               normed.data(), f.hidden, P);
    }
}

static void cpu_recurrent_projected(
    float *output, CpuState &state, std::vector<float> &qkv,
    const std::vector<float> &decay, const std::vector<float> &beta,
    const std::vector<float> &conv, int rows, int heads, int dim,
    int kernel, float norm_eps) {
    const int projection = heads * dim;
    std::vector<float> memory(dim);
    const float query_scale = 1.0f / std::sqrt((float)dim);
    for (int row = 0; row < rows; ++row) {
        for (int head = 0; head < heads; ++head) {
            for (int part = 0; part < 3; ++part)
                for (int d = 0; d < dim; ++d) {
                    int channel = part * projection + head * dim + d;
                    float *history =
                        state.window.data() + (size_t)channel * kernel;
                    for (int tap = 0; tap + 1 < kernel; ++tap)
                        history[tap] = history[tap + 1];
                    float &slot =
                        qkv[((size_t)part * rows + row) * projection +
                            head * dim + d];
                    history[kernel - 1] = slot;
                    float sum = 0.0f;
                    for (int tap = 0; tap < kernel; ++tap)
                        sum += conv[(size_t)channel * kernel + tap] *
                               history[tap];
                    slot = sum / (1.0f + std::exp(-sum));
                }
            const float *query =
                qkv.data() + (size_t)row * projection + head * dim;
            const float *key =
                qkv.data() + ((size_t)rows + row) * projection + head * dim;
            const float *value =
                qkv.data() + ((size_t)2 * rows + row) * projection +
                head * dim;
            float query_square = norm_eps, key_square = norm_eps;
            for (int d = 0; d < dim; ++d) {
                query_square += query[d] * query[d];
                key_square += key[d] * key[d];
            }
            float query_norm = query_scale / std::sqrt(query_square);
            float key_norm = 1.0f / std::sqrt(key_square);
            float *matrix =
                state.matrix.data() + (size_t)head * dim * dim;
            std::fill(memory.begin(), memory.end(), 0.0f);
            for (int kd = 0; kd < dim; ++kd) {
                float *matrix_row = matrix + (size_t)kd * dim;
                float alpha = std::exp(
                    decay[(size_t)row * projection + head * dim + kd]);
                float scaled_key = key[kd] * key_norm;
                for (int vd = 0; vd < dim; ++vd) {
                    matrix_row[vd] *= alpha;
                    memory[vd] += scaled_key * matrix_row[vd];
                }
            }
            float *result =
                output + (size_t)row * projection + head * dim;
            std::fill(result, result + dim, 0.0f);
            for (int kd = 0; kd < dim; ++kd) {
                float *matrix_row = matrix + (size_t)kd * dim;
                float scaled_key = key[kd] * key_norm;
                float scaled_query = query[kd] * query_norm;
                for (int vd = 0; vd < dim; ++vd) {
                    matrix_row[vd] += scaled_key *
                        (value[vd] - memory[vd]) *
                        beta[(size_t)row * heads + head];
                    result[vd] += scaled_query * matrix_row[vd];
                }
            }
        }
    }
}

static ColiGpuTensor *upload_f32(ColiGpuContext *ctx,
                                 const std::vector<float> &data,
                                 int rows, int columns) {
    ColiGpuTensorDesc desc = {};
    desc.data = data.data();
    desc.format = 0;
    desc.rows = rows;
    desc.columns = columns;
    ColiGpuTensor *tensor = nullptr;
    return coli_gpu_tensor_create(&tensor, ctx, &desc) ? tensor : nullptr;
}

struct DeviceWeights {
    ColiGpuKdaWeights weights = {};
    std::vector<ColiGpuTensor *> owned;

    bool create(ColiGpuContext *ctx, const Fixture &f) {
        ColiGpuTensor *items[] = {
            upload_f32(ctx, f.q, f.projection, f.hidden),
            upload_f32(ctx, f.k, f.projection, f.hidden),
            upload_f32(ctx, f.v, f.projection, f.hidden),
            upload_f32(ctx, f.output, f.hidden, f.projection),
            upload_f32(ctx, f.gate_a, f.dim, f.hidden),
            upload_f32(ctx, f.gate_b, f.projection, f.dim),
            upload_f32(ctx, f.decay_a, f.dim, f.hidden),
            upload_f32(ctx, f.decay_b, f.projection, f.dim),
            upload_f32(ctx, f.beta, f.heads, f.hidden),
            upload_f32(ctx, f.conv, 3 * f.projection, f.kernel),
            upload_f32(ctx, f.dt_bias, 1, f.projection),
            upload_f32(ctx, f.a_log, 1, f.heads),
            upload_f32(ctx, f.o_norm, 1, f.dim),
        };
        for (ColiGpuTensor *item : items) {
            if (!item) return false;
            owned.push_back(item);
        }
        weights.q_proj = items[0];
        weights.k_proj = items[1];
        weights.v_proj = items[2];
        weights.o_proj = items[3];
        weights.gate_a_proj = items[4];
        weights.gate_b_proj = items[5];
        weights.decay_a_proj = items[6];
        weights.decay_b_proj = items[7];
        weights.beta_proj = items[8];
        weights.conv = items[9];
        weights.dt_bias = items[10];
        weights.a_log = items[11];
        weights.o_norm = items[12];
        return true;
    }

    ~DeviceWeights() {
        for (ColiGpuTensor *item : owned) coli_gpu_tensor_destroy(item);
    }
};

struct DeviceSite {
    ColiGpuArena *arena = nullptr;
    ColiGpuKdaState *state = nullptr;
    size_t input = 0, output = 0, scratch = 0;
    size_t state_offset = 0, window_offset = 0, capacity = 0;

    bool create(ColiGpuContext *ctx, const Fixture &f, int row_capacity) {
        ColiGpuKdaConfig config = {};
        config.heads = f.heads;
        config.head_dim = f.dim;
        config.kernel = f.kernel;
        config.max_rows = row_capacity;
        config.max_context = f.max_context;
        config.recurrent_norm_eps = f.recurrent_eps;
        config.output_norm_eps = f.output_eps;
        config.gate_lower_bound = f.gate_lower_bound;
        input = 0;
        output = align256((size_t)row_capacity * f.hidden * sizeof(float));
        scratch = output + align256((size_t)row_capacity * f.hidden * sizeof(float));
        state_offset = scratch + align256(
            coli_gpu_kda_scratch_bytes(&config, row_capacity, f.hidden));
        window_offset = state_offset + align256(coli_gpu_kda_state_bytes(&config));
        capacity = window_offset + coli_gpu_kda_window_bytes(&config);
        return coli_gpu_arena_create(&arena, ctx, capacity) &&
            coli_gpu_kda_state_create(&state, ctx, arena, state_offset,
                                      window_offset, &config) &&
            coli_gpu_kda_state_reset(state);
    }

    ~DeviceSite() {
        coli_gpu_kda_state_destroy(state);
        coli_gpu_arena_destroy(arena);
    }
};

static int telemetry_unchanged(const ColiGpuTelemetry &before,
                               const ColiGpuTelemetry &after,
                               const char *where) {
    if (std::memcmp(&before, &after, sizeof(before)) != 0) {
        std::fprintf(stderr, "%s changed allocation/transfer telemetry\n", where);
        return 0;
    }
    return 1;
}

static int inspect_state(ColiGpuKdaState *state, const CpuState &want,
                         const char *label,
                         std::vector<float> *matrix_out = nullptr,
                         std::vector<float> *window_out = nullptr) {
    std::vector<float> matrix(want.matrix.size()), window(want.window.size());
    if (!coli_gpu_kda_state_download(state, matrix.data(), matrix.size(),
                                     window.data(), window.size()) ||
        !compare_vectors(label, matrix.data(), want.matrix.data(), matrix.size(),
                         3e-4f, 0.99999f) ||
        !compare_vectors("KDA convolution windows", window.data(),
                         want.window.data(), window.size(), 3e-4f, 0.99999f))
        return 0;
    if (matrix_out) *matrix_out = matrix;
    if (window_out) *window_out = window;
    return 1;
}

static int run_complete_case(ColiGpuContext *ctx, Fixture &f,
                             const char *name, bool require_zero) {
    DeviceWeights weights;
    DeviceSite token_site, chunk_site;
    if (!weights.create(ctx, f) ||
        !token_site.create(ctx, f, f.rows) ||
        !chunk_site.create(ctx, f, f.rows))
        return 0;
    if (f.max_context > f.rows &&
        coli_gpu_kda_site(token_site.arena, token_site.output,
                          token_site.input, token_site.scratch,
                          token_site.state, &weights.weights,
                          f.rows + 1, 0, f.hidden)) {
        std::fprintf(stderr, "KDA accepted row-capacity violation\n");
        return 0;
    }
    CpuState cpu = {
        std::vector<float>((size_t)f.heads * f.dim * f.dim, 0.0f),
        std::vector<float>((size_t)3 * f.projection * f.kernel, 0.0f)
    };
    std::vector<float> want((size_t)f.rows * f.hidden);
    std::vector<float> token_output(want.size());
    for (int row = 0; row < f.rows; ++row) {
        cpu_site(want.data() + (size_t)row * f.hidden, cpu, f,
                 f.input.data() + (size_t)row * f.hidden, 1);
        if (!coli_gpu_arena_upload(token_site.arena, token_site.input,
                                   f.input.data() + (size_t)row * f.hidden,
                                   (size_t)f.hidden * sizeof(float)))
            return 0;
        ColiGpuTelemetry before = {}, after = {};
        coli_gpu_context_telemetry(ctx, &before);
        if (!coli_gpu_kda_site(token_site.arena, token_site.output,
                               token_site.input, token_site.scratch,
                               token_site.state, &weights.weights,
                               1, row, f.hidden))
            return 0;
        coli_gpu_context_telemetry(ctx, &after);
        if (!telemetry_unchanged(before, after, "single-token KDA site") ||
            !coli_gpu_arena_download(
                token_site.arena, token_site.output,
                token_output.data() + (size_t)row * f.hidden,
                (size_t)f.hidden * sizeof(float)) ||
            !compare_vectors("KDA per-step output",
                             token_output.data() + (size_t)row * f.hidden,
                             want.data() + (size_t)row * f.hidden, f.hidden,
                             1e-3f, 0.9999f) ||
            !inspect_state(token_site.state, cpu, "KDA recurrent state"))
            return 0;
    }
    if (coli_gpu_kda_site(token_site.arena, token_site.output, token_site.input,
                          token_site.scratch, token_site.state, &weights.weights,
                          1, f.max_context, f.hidden) ||
        coli_gpu_kda_site(token_site.arena, token_site.output, token_site.input,
                          token_site.scratch, token_site.state, &weights.weights,
                          1, f.rows - 1, f.hidden)) {
        std::fprintf(stderr, "KDA accepted capacity/position violation\n");
        return 0;
    }

    CpuState chunk_cpu = {
        std::vector<float>(cpu.matrix.size(), 0.0f),
        std::vector<float>(cpu.window.size(), 0.0f)
    };
    std::vector<float> chunk_want(want.size()), chunk_got(want.size());
    std::vector<float> chunk_matrix, chunk_window;
    cpu_site(chunk_want.data(), chunk_cpu, f, f.input.data(), f.rows);
    ColiGpuTelemetry before = {}, after = {};
    if (!coli_gpu_arena_upload(chunk_site.arena, chunk_site.input, f.input.data(),
                               f.input.size() * sizeof(float)))
        return 0;
    coli_gpu_context_telemetry(ctx, &before);
    if (!coli_gpu_kda_site(chunk_site.arena, chunk_site.output, chunk_site.input,
                           chunk_site.scratch, chunk_site.state, &weights.weights,
                           f.rows, 0, f.hidden))
        return 0;
    coli_gpu_context_telemetry(ctx, &after);
    if (!telemetry_unchanged(before, after, "chunked KDA site") ||
        !coli_gpu_arena_download(chunk_site.arena, chunk_site.output,
                                 chunk_got.data(), chunk_got.size() * sizeof(float)) ||
        !compare_vectors(name, chunk_got.data(), chunk_want.data(), chunk_got.size(),
                         1e-3f, 0.9999f) ||
        !compare_vectors("chunked versus token output", chunk_got.data(),
                         token_output.data(), chunk_got.size(), 1e-3f, 0.9999f) ||
        !inspect_state(chunk_site.state, chunk_cpu, "chunked final KDA state",
                       &chunk_matrix, &chunk_window))
        return 0;

    std::vector<float> first_matrix, first_window;
    if (!inspect_state(token_site.state, cpu, "pre-reset KDA state",
                       &first_matrix, &first_window) ||
        !compare_vectors("chunked versus token state", chunk_matrix.data(),
                         first_matrix.data(), first_matrix.size(),
                         3e-4f, 0.99999f) ||
        !compare_vectors("chunked versus token windows", chunk_window.data(),
                         first_window.data(), first_window.size(),
                         3e-4f, 0.99999f) ||
        std::memcmp(chunk_got.data(), token_output.data(),
                    chunk_got.size() * sizeof(float)) ||
        std::memcmp(chunk_matrix.data(), first_matrix.data(),
                    first_matrix.size() * sizeof(float)) ||
        std::memcmp(chunk_window.data(), first_window.data(),
                    first_window.size() * sizeof(float))) {
        std::fprintf(stderr, "chunked/token KDA final state is not bitwise identical\n");
        return 0;
    }
    if (f.max_context > f.rows) {
        int overflow_rows = f.max_context - f.rows + 1;
        if (overflow_rows <= f.rows) {
            ColiGpuTelemetry before_reject = {}, after_reject = {};
            coli_gpu_context_telemetry(ctx, &before_reject);
            int accepted = coli_gpu_kda_site(
                token_site.arena, token_site.output, token_site.input,
                token_site.scratch, token_site.state, &weights.weights,
                overflow_rows, f.rows, f.hidden);
            coli_gpu_context_telemetry(ctx, &after_reject);
            std::vector<float> rejected_matrix(first_matrix.size());
            std::vector<float> rejected_window(first_window.size());
            if (accepted ||
                !telemetry_unchanged(before_reject, after_reject,
                                     "context-overflow rejection") ||
                !coli_gpu_kda_state_download(
                    token_site.state, rejected_matrix.data(),
                    rejected_matrix.size(), rejected_window.data(),
                    rejected_window.size()) ||
                std::memcmp(rejected_matrix.data(), first_matrix.data(),
                            first_matrix.size() * sizeof(float)) ||
                std::memcmp(rejected_window.data(), first_window.data(),
                            first_window.size() * sizeof(float))) {
                std::fprintf(stderr,
                             "context overflow changed KDA state or telemetry\n");
                return 0;
            }
            std::puts("  context overflow rejection is side-effect free");
        }
    }
    if (!coli_gpu_kda_state_reset(token_site.state))
        return 0;
    std::vector<float> replay(token_output.size());
    for (int row = 0; row < f.rows; ++row) {
        if (!coli_gpu_arena_upload(token_site.arena, token_site.input,
                                   f.input.data() + (size_t)row * f.hidden,
                                   (size_t)f.hidden * sizeof(float)) ||
            !coli_gpu_kda_site(token_site.arena, token_site.output,
                               token_site.input, token_site.scratch,
                               token_site.state, &weights.weights,
                               1, row, f.hidden) ||
            !coli_gpu_arena_download(
                token_site.arena, token_site.output,
                replay.data() + (size_t)row * f.hidden,
                (size_t)f.hidden * sizeof(float)))
            return 0;
    }
    std::vector<float> replay_matrix(first_matrix.size());
    std::vector<float> replay_window(first_window.size());
    if (!coli_gpu_kda_state_download(token_site.state, replay_matrix.data(),
                                     replay_matrix.size(), replay_window.data(),
                                     replay_window.size()) ||
        std::memcmp(replay.data(), token_output.data(),
                    replay.size() * sizeof(float)) ||
        std::memcmp(replay_matrix.data(), first_matrix.data(),
                    replay_matrix.size() * sizeof(float)) ||
        std::memcmp(replay_window.data(), first_window.data(),
                    replay_window.size() * sizeof(float))) {
        std::fprintf(stderr, "KDA reset/replay is not bitwise deterministic\n");
        return 0;
    }
    if (require_zero) {
        for (float value : replay)
            if (value != 0.0f) {
                std::fprintf(stderr, "zero KDA fixture produced nonzero output\n");
                return 0;
            }
    }
    return 1;
}

static int test_alias_rejection(ColiGpuContext *ctx) {
    Fixture f(1, 37, 3, 5, 4, 2);
    f.randomize();
    DeviceWeights weights;
    DeviceSite site;
    if (!weights.create(ctx, f) || !site.create(ctx, f, 1) ||
        !coli_gpu_arena_upload(site.arena, site.input, f.input.data(),
                               (size_t)f.hidden * sizeof(float)))
        return 0;
    const size_t state_count = (size_t)f.heads * f.dim * f.dim;
    const size_t window_count = (size_t)3 * f.projection * f.kernel;
    std::vector<float> zero_state(state_count), zero_window(window_count);
    std::vector<float> after_state(state_count), after_window(window_count);

    ColiGpuTelemetry before = {}, after = {};
    coli_gpu_context_telemetry(ctx, &before);
    int scratch_accepted = coli_gpu_kda_site(
        site.arena, site.output, site.input, site.state_offset, site.state,
        &weights.weights, 1, 0, f.hidden);
    coli_gpu_context_telemetry(ctx, &after);
    int scratch_clean = telemetry_unchanged(
        before, after, "scratch/state overlap rejection") &&
        coli_gpu_kda_state_download(
            site.state, after_state.data(), after_state.size(),
            after_window.data(), after_window.size()) &&
        !std::memcmp(after_state.data(), zero_state.data(),
                     state_count * sizeof(float)) &&
        !std::memcmp(after_window.data(), zero_window.data(),
                     window_count * sizeof(float));

    if (!coli_gpu_kda_state_reset(site.state)) return 0;
    coli_gpu_context_telemetry(ctx, &before);
    int output_accepted = coli_gpu_kda_site(
        site.arena, site.window_offset, site.input, site.scratch, site.state,
        &weights.weights, 1, 0, f.hidden);
    coli_gpu_context_telemetry(ctx, &after);
    int output_clean = telemetry_unchanged(
        before, after, "output/window overlap rejection") &&
        coli_gpu_kda_state_download(
            site.state, after_state.data(), after_state.size(),
            after_window.data(), after_window.size()) &&
        !std::memcmp(after_state.data(), zero_state.data(),
                     state_count * sizeof(float)) &&
        !std::memcmp(after_window.data(), zero_window.data(),
                     window_count * sizeof(float));

    if (!coli_gpu_kda_state_reset(site.state)) return 0;
    std::vector<float> qkv((size_t)3 * f.projection);
    std::vector<float> decay(f.projection);
    std::vector<float> beta(f.heads);
    for (float &value : qkv) value = random_float(0.3f);
    for (float &value : decay) value = -0.2f + random_float(0.1f);
    for (float &value : beta) value = 0.5f + random_float(0.1f);
    size_t decay_offset = site.scratch;
    size_t beta_offset =
        decay_offset + align256(decay.size() * sizeof(float));
    if (!coli_gpu_arena_upload(site.arena, site.state_offset, qkv.data(),
                               qkv.size() * sizeof(float)) ||
        !coli_gpu_arena_upload(site.arena, decay_offset, decay.data(),
                               decay.size() * sizeof(float)) ||
        !coli_gpu_arena_upload(site.arena, beta_offset, beta.data(),
                               beta.size() * sizeof(float)) ||
        !coli_gpu_kda_state_download(
            site.state, zero_state.data(), zero_state.size(),
            zero_window.data(), zero_window.size()))
        return 0;
    coli_gpu_context_telemetry(ctx, &before);
    int qkv_accepted = coli_gpu_kda_recurrent(
        site.arena, site.output, site.state_offset, decay_offset, beta_offset,
        site.state, weights.weights.conv, 1, 0);
    coli_gpu_context_telemetry(ctx, &after);
    int qkv_clean = telemetry_unchanged(
        before, after, "qkv/state overlap rejection") &&
        coli_gpu_kda_state_download(
            site.state, after_state.data(), after_state.size(),
            after_window.data(), after_window.size()) &&
        !std::memcmp(after_state.data(), zero_state.data(),
                     state_count * sizeof(float)) &&
        !std::memcmp(after_window.data(), zero_window.data(),
                     window_count * sizeof(float));

    if (scratch_accepted)
        std::fprintf(stderr, "KDA site accepted scratch/state overlap\n");
    if (output_accepted)
        std::fprintf(stderr, "KDA site accepted output/window overlap\n");
    if (qkv_accepted)
        std::fprintf(stderr, "KDA recurrent accepted qkv/state overlap\n");
    if (!scratch_clean || !output_clean || !qkv_clean)
        std::fprintf(stderr, "rejected KDA alias changed persistent state\n");
    if (scratch_accepted || output_accepted || qkv_accepted ||
        !scratch_clean || !output_clean || !qkv_clean)
        return 0;
    std::puts("  KDA alias rejection is side-effect free");
    return 1;
}

static int test_extreme_dimensions_rejected(ColiGpuContext *ctx) {
    ColiGpuKdaConfig product_overflow = {};
    product_overflow.heads = INT_MAX;
    product_overflow.head_dim = 2;
    product_overflow.kernel = 1;
    product_overflow.max_rows = 1;
    product_overflow.max_context = 1;
    product_overflow.recurrent_norm_eps = 1e-6f;
    product_overflow.output_norm_eps = 1e-6f;
    product_overflow.gate_lower_bound = -5.0f;
    ColiGpuKdaConfig row_overflow = product_overflow;
    row_overflow.heads = 2;
    row_overflow.head_dim = 1;
    row_overflow.max_rows = INT_MAX;
    row_overflow.max_context = INT_MAX;
    if (coli_gpu_kda_state_bytes(&product_overflow) ||
        coli_gpu_kda_window_bytes(&product_overflow) ||
        coli_gpu_kda_scratch_bytes(&product_overflow, 1, 1) ||
        coli_gpu_kda_state_bytes(&row_overflow) ||
        coli_gpu_kda_window_bytes(&row_overflow) ||
        coli_gpu_kda_scratch_bytes(&row_overflow, INT_MAX, 1)) {
        std::fprintf(stderr, "KDA accepted overflowing plain dimensions\n");
        return 0;
    }

    float one = 1.0f;
    ColiGlm53GpuKdaLayerDesc layer = {};
    layer.q_proj = &one;
    layer.k_proj = &one;
    layer.v_proj = &one;
    layer.o_proj = &one;
    layer.gate_a_proj = &one;
    layer.gate_b_proj = &one;
    layer.decay_a_proj = &one;
    layer.decay_b_proj = &one;
    layer.beta_proj = &one;
    layer.conv = &one;
    layer.dt_bias = &one;
    layer.a_log = &one;
    layer.o_norm = &one;
    ColiGlm53GpuModelDesc desc = {};
    desc.hidden_size = 1;
    desc.stream_count = 1;
    desc.vocab_size = 1;
    desc.max_prefill_rows = 1;
    desc.max_context_tokens = 1;
    desc.norm_eps = 1e-6f;
    desc.hc_eps = 1e-6f;
    desc.embedding = &one;
    desc.final_norm = &one;
    desc.lm_head = &one;
    desc.lm_head_format = 0;
    desc.kda_heads = INT_MAX;
    desc.kda_head_dim = 2;
    desc.kda_kernel = 1;
    desc.kda_gate_lower_bound = -5.0f;
    desc.kda_layers = &layer;
    desc.kda_layer_count = 1;
    ColiGpuTelemetry before = {}, after = {};
    ColiGlm53GpuModel *model = nullptr;
    coli_gpu_context_telemetry(ctx, &before);
    int accepted = coli_glm53_gpu_model_create(&model, ctx, &desc);
    coli_gpu_context_telemetry(ctx, &after);
    coli_glm53_gpu_model_destroy(model);
    if (accepted || model || std::memcmp(&before, &after, sizeof(before))) {
        std::fprintf(stderr,
                     "GLM KDA model accepted/allocated extreme dimensions\n");
        return 0;
    }
    std::puts("  KDA extreme dimensions rejected before allocation");
    return 1;
}

static int test_production_recurrence(ColiGpuContext *ctx) {
    const int rows = 2, heads = 64, dim = 128, kernel = 4;
    const int projection = heads * dim;
    ColiGpuKdaConfig config = {};
    config.heads = heads;
    config.head_dim = dim;
    config.kernel = kernel;
    config.max_rows = rows;
    config.max_context = rows;
    config.recurrent_norm_eps = 1e-6f;
    config.output_norm_eps = 1e-6f;
    config.gate_lower_bound = -5.0f;
    std::vector<float> qkv((size_t)rows * 3 * projection);
    std::vector<float> decay((size_t)rows * projection);
    std::vector<float> beta((size_t)rows * heads);
    std::vector<float> conv((size_t)3 * projection * kernel);
    for (float &x : qkv) x = random_float(0.4f);
    for (float &x : decay) x = -0.2f + random_float(0.1f);
    for (float &x : beta) x = 0.5f + random_float(0.1f);
    for (float &x : conv) x = random_float(0.3f);
    CpuState oracle = {
        std::vector<float>((size_t)heads * dim * dim, 0.0f),
        std::vector<float>((size_t)3 * projection * kernel, 0.0f)
    };
    std::vector<float> oracle_qkv = qkv;
    std::vector<float> oracle_output((size_t)rows * projection);
    cpu_recurrent_projected(
        oracle_output.data(), oracle, oracle_qkv, decay, beta, conv,
        rows, heads, dim, kernel, config.recurrent_norm_eps);
    ColiGpuTensor *conv_tensor = upload_f32(ctx, conv, 3 * projection, kernel);
    size_t qkv_offset = 0;
    size_t decay_offset = align256(qkv.size() * sizeof(float));
    size_t beta_offset = decay_offset + align256(decay.size() * sizeof(float));
    size_t output_offset = beta_offset + align256(beta.size() * sizeof(float));
    size_t state_offset = output_offset +
        align256((size_t)rows * projection * sizeof(float));
    size_t window_offset = state_offset +
        align256(coli_gpu_kda_state_bytes(&config));
    size_t capacity = window_offset + coli_gpu_kda_window_bytes(&config);
    ColiGpuArena *arena = nullptr;
    ColiGpuKdaState *state = nullptr;
    int ok = conv_tensor &&
        coli_gpu_arena_create(&arena, ctx, capacity) &&
        coli_gpu_kda_state_create(&state, ctx, arena, state_offset,
                                  window_offset, &config) &&
        coli_gpu_kda_state_reset(state) &&
        coli_gpu_arena_upload(arena, qkv_offset, qkv.data(),
                              qkv.size() * sizeof(float)) &&
        coli_gpu_arena_upload(arena, decay_offset, decay.data(),
                              decay.size() * sizeof(float)) &&
        coli_gpu_arena_upload(arena, beta_offset, beta.data(),
                              beta.size() * sizeof(float)) &&
        coli_gpu_kda_recurrent(arena, output_offset, qkv_offset, decay_offset,
                               beta_offset, state, conv_tensor, rows, 0);
    if (ok) {
        std::vector<float> output((size_t)rows * projection);
        ok = coli_gpu_arena_download(arena, output_offset, output.data(),
                                     output.size() * sizeof(float)) &&
            compare_vectors("production recurrence output", output.data(),
                            oracle_output.data(), output.size(),
                            3e-4f, 0.99999f);
        std::vector<float> first_state((size_t)heads * dim * dim);
        std::vector<float> first_window((size_t)3 * projection * kernel);
        ok = ok && coli_gpu_kda_state_download(
            state, first_state.data(), first_state.size(),
            first_window.data(), first_window.size()) &&
            compare_vectors("production recurrent state", first_state.data(),
                            oracle.matrix.data(), first_state.size(),
                            3e-4f, 0.99999f) &&
            compare_vectors("production convolution windows",
                            first_window.data(), oracle.window.data(),
                            first_window.size(), 3e-4f, 0.99999f) &&
            coli_gpu_kda_state_reset(state) &&
            coli_gpu_arena_upload(arena, qkv_offset, qkv.data(),
                                  qkv.size() * sizeof(float)) &&
            coli_gpu_kda_recurrent(arena, output_offset, qkv_offset, decay_offset,
                                   beta_offset, state, conv_tensor, rows, 0);
        std::vector<float> repeat_state(first_state.size());
        std::vector<float> repeat_window(first_window.size());
        ok = ok && coli_gpu_kda_state_download(
            state, repeat_state.data(), repeat_state.size(),
            repeat_window.data(), repeat_window.size()) &&
            !std::memcmp(first_state.data(), repeat_state.data(),
                         first_state.size() * sizeof(float)) &&
            !std::memcmp(first_window.data(), repeat_window.data(),
                         first_window.size() * sizeof(float));
    }
    coli_gpu_kda_state_destroy(state);
    coli_gpu_arena_destroy(arena);
    coli_gpu_tensor_destroy(conv_tensor);
    if (!ok) std::fprintf(stderr, "production KDA recurrence failed\n");
    return ok;
}

static int test_model_session_ownership(ColiGpuContext *ctx) {
    const int layer_count = 34;
    Fixture f(2, 19, 2, 3, 3, 6);
    f.randomize();
    std::vector<float> embedding((size_t)7 * f.hidden);
    std::vector<float> norm(f.hidden, 1.0f);
    std::vector<float> head((size_t)7 * f.hidden);
    for (float &x : embedding) x = random_float(0.3f);
    for (float &x : head) x = random_float(0.1f);
    std::vector<ColiGlm53GpuKdaLayerDesc> layers(layer_count);
    for (ColiGlm53GpuKdaLayerDesc &layer : layers) {
        layer.q_proj = f.q.data();
        layer.k_proj = f.k.data();
        layer.v_proj = f.v.data();
        layer.o_proj = f.output.data();
        layer.gate_a_proj = f.gate_a.data();
        layer.gate_b_proj = f.gate_b.data();
        layer.decay_a_proj = f.decay_a.data();
        layer.decay_b_proj = f.decay_b.data();
        layer.beta_proj = f.beta.data();
        layer.conv = f.conv.data();
        layer.dt_bias = f.dt_bias.data();
        layer.a_log = f.a_log.data();
        layer.o_norm = f.o_norm.data();
    }
    ColiGlm53GpuModelDesc desc = {};
    desc.hidden_size = f.hidden;
    desc.stream_count = 4;
    desc.vocab_size = 7;
    desc.max_prefill_rows = f.rows;
    desc.max_context_tokens = f.max_context;
    desc.norm_eps = f.output_eps;
    desc.hc_eps = 1e-6f;
    desc.embedding = embedding.data();
    desc.final_norm = norm.data();
    desc.lm_head = head.data();
    desc.lm_head_format = 0;
    desc.kda_heads = f.heads;
    desc.kda_head_dim = f.dim;
    desc.kda_kernel = f.kernel;
    desc.kda_gate_lower_bound = f.gate_lower_bound;
    desc.kda_layers = layers.data();
    desc.kda_layer_count = layer_count;
    ColiGlm53GpuModel *model = nullptr;
    ColiGlm53GpuSession *session = nullptr;
    std::vector<float> output((size_t)f.rows * f.hidden);
    CpuState oracle = {
        std::vector<float>((size_t)f.heads * f.dim * f.dim, 0.0f),
        std::vector<float>((size_t)3 * f.projection * f.kernel, 0.0f)
    };
    std::vector<float> oracle_output(output.size());
    cpu_site(oracle_output.data(), oracle, f, f.input.data(), f.rows);
    int ok = coli_glm53_gpu_model_create(&model, ctx, &desc) &&
        coli_glm53_gpu_session_create(&session, model, f.rows) &&
        coli_glm53_gpu_session_kda_upload_input(
            session, f.input.data(), f.rows) &&
        coli_glm53_gpu_session_kda_site(session, 0, f.rows, 0);
    ColiGpuTelemetry before = {}, after = {};
    coli_gpu_context_telemetry(ctx, &before);
    for (int layer = 1; ok && layer < layer_count; ++layer)
        ok = coli_glm53_gpu_session_kda_site(
            session, layer, f.rows, 0);
    coli_gpu_context_telemetry(ctx, &after);
    ok = ok && telemetry_unchanged(before, after, "owned KDA layer") &&
        coli_glm53_gpu_session_kda_download_output(
            session, output.data(), f.rows) &&
        compare_vectors("owned KDA site output", output.data(),
                        oracle_output.data(), output.size(), 1e-3f, 0.9999f);
    for (int layer = 0; ok && layer < layer_count; ++layer) {
        std::vector<float> matrix(oracle.matrix.size());
        std::vector<float> window(oracle.window.size());
        ok = coli_glm53_gpu_session_kda_state_download(
                 session, layer, matrix.data(), matrix.size(),
                 window.data(), window.size()) &&
             compare_vectors("owned per-layer state", matrix.data(),
                             oracle.matrix.data(), matrix.size(),
                             3e-4f, 0.99999f) &&
             compare_vectors("owned per-layer window", window.data(),
                             oracle.window.data(), window.size(),
                             3e-4f, 0.99999f);
    }
    uint64_t allocations_before_reset = after.device_allocations;
    ok = ok && coli_glm53_gpu_session_reset(session);
    coli_gpu_context_telemetry(ctx, &after);
    std::vector<float> zero_matrix(oracle.matrix.size(), 0.0f);
    std::vector<float> zero_window(oracle.window.size(), 0.0f);
    for (int layer = 0; ok && layer < layer_count; ++layer) {
        std::vector<float> matrix(zero_matrix.size());
        std::vector<float> window(zero_window.size());
        ok = coli_glm53_gpu_session_kda_state_download(
                 session, layer, matrix.data(), matrix.size(),
                 window.data(), window.size()) &&
             !std::memcmp(matrix.data(), zero_matrix.data(),
                          matrix.size() * sizeof(float)) &&
             !std::memcmp(window.data(), zero_window.data(),
                          window.size() * sizeof(float));
    }
    ok = ok && after.device_allocations == allocations_before_reset &&
        coli_glm53_gpu_session_kda_upload_input(
            session, f.input.data(), f.rows) &&
        coli_glm53_gpu_session_kda_site(session, 0, f.rows, 0);
    coli_glm53_gpu_session_destroy(session);
    coli_glm53_gpu_model_destroy(model);
    if (!ok) std::fprintf(stderr, "model/session KDA ownership failed\n");
    return ok;
}

int main(void) {
    ColiGpuContext *ctx = nullptr;
    if (!coli_gpu_context_create(&ctx, 0)) {
        std::fprintf(stderr, "Task 3 requires a HIP device\n");
        return 1;
    }
    Fixture zero(1, 1, 1, 1, 1, 1);
    zero.zero();
    Fixture tail(5, 37, 3, 5, 4, 8);
    tail.randomize();
    Fixture hidden4096(1, 4096, 8, 16, 4, 1);
    hidden4096.randomize();
    int ok = 1;
    ok &= test_alias_rejection(ctx);
    ok &= test_extreme_dimensions_rejected(ctx);
    ok &= run_complete_case(ctx, zero, "KDA zero/minimum fixture", true);
    ok &= run_complete_case(ctx, tail, "KDA complete tail fixture", false);
    ok &= run_complete_case(ctx, hidden4096,
                            "KDA hidden-4096 fixture", false);
    ok &= test_production_recurrence(ctx);
    ok &= test_model_session_ownership(ctx);
    coli_gpu_context_destroy(ctx);
    if (!ok) return 1;
    std::puts("glm53 device-resident KDA HIP: ok");
    return 0;
}
