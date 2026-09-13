#include "../backend_cuda.h"
#include "../glm53_gpu.h"
#include "../sparse_index.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static uint64_t rng_state = 0x13198a2e03707344ULL;

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
    std::printf("  %-31s rel_l2 %.3e cosine %.8f\n", name, rel, cosine);
    if (!std::isfinite(rel) || !std::isfinite(cosine) ||
        rel > max_rel_l2 || cosine < min_cosine) {
        std::fprintf(stderr, "%s outside tolerance\n", name);
        return 0;
    }
    return 1;
}

static void matvec(float *out, const std::vector<float> &weight,
                   const float *input, int rows, int columns) {
    for (int row = 0; row < rows; ++row) {
        float sum = 0.0f;
        for (int column = 0; column < columns; ++column)
            sum += weight[(size_t)row * columns + column] * input[column];
        out[row] = sum;
    }
}

static void rms(float *values, const std::vector<float> &weight,
                int count, float eps) {
    float square = 0.0f;
    for (int i = 0; i < count; ++i) square += values[i] * values[i];
    float inverse = 1.0f / std::sqrt(square / count + eps);
    for (int i = 0; i < count; ++i) values[i] *= inverse * weight[i];
}

static void layer_norm(float *values, const std::vector<float> &weight,
                       const std::vector<float> &bias, int count, float eps) {
    float mean = 0.0f;
    for (int i = 0; i < count; ++i) mean += values[i];
    mean /= count;
    float variance = 0.0f;
    for (int i = 0; i < count; ++i) {
        float centered = values[i] - mean;
        variance += centered * centered;
    }
    float inverse = 1.0f / std::sqrt(variance / count + eps);
    for (int i = 0; i < count; ++i)
        values[i] = (values[i] - mean) * inverse * weight[i] + bias[i];
}

struct Fixture {
    ColiGpuMlaConfig config = {};
    std::vector<float> input;
    std::vector<float> qa, qan, qb, kva, kvan, kvbk, kvbv, output;
    std::vector<float> iwq, iwk, iwp, iknw, iknb, ape, gate;

    Fixture(int rows, int hidden, int heads, int qrank, int latent,
            int qk, int value, int index_heads, int index_dim,
            int pool, int topk, int tail, int max_context, int page) {
        config.hidden = hidden;
        config.heads = heads;
        config.q_lora = qrank;
        config.kv_lora = latent;
        config.qk_nope = qk;
        config.qk_rope = 0;
        config.value_dim = value;
        config.index_heads = index_heads;
        config.index_dim = index_dim;
        config.index_pool = pool;
        config.index_topk = topk;
        config.index_select_tail = tail;
        config.max_rows = rows;
        config.max_context = max_context;
        config.page_tokens = page;
        config.rms_norm_eps = 1e-6f;
        config.index_norm_eps = 1e-5f;
        input.resize((size_t)rows * hidden);
        qa.resize((size_t)qrank * hidden);
        qan.resize(qrank);
        qb.resize((size_t)heads * qk * qrank);
        kva.resize((size_t)latent * hidden);
        kvan.resize(latent);
        kvbk.resize((size_t)heads * latent * qk);
        kvbv.resize((size_t)heads * value * latent);
        output.resize((size_t)hidden * heads * value);
        iwq.resize((size_t)index_heads * index_dim * qrank);
        iwk.resize((size_t)index_dim * hidden);
        iwp.resize((size_t)index_heads * hidden);
        iknw.resize(index_dim);
        iknb.resize(index_dim);
        ape.resize((size_t)pool * index_dim);
        gate.resize((size_t)index_dim * hidden);
    }

    void randomize(bool tied_index = false) {
        for (float &x : input) x = random_float(0.7f);
        for (float &x : qa) x = random_float(0.15f);
        for (float &x : qan) x = 1.0f + random_float(0.1f);
        for (float &x : qb) x = random_float(0.12f);
        for (float &x : kva) x = random_float(0.15f);
        for (float &x : kvan) x = 1.0f + random_float(0.1f);
        for (float &x : kvbk) x = random_float(0.13f);
        for (float &x : kvbv) x = random_float(0.13f);
        for (float &x : output) x = random_float(0.11f);
        for (float &x : iwq) x = random_float(0.12f);
        for (float &x : iwk) x = random_float(0.12f);
        for (float &x : iwp) x = tied_index ? 0.0f : random_float(0.12f);
        for (float &x : iknw) x = 1.0f + random_float(0.1f);
        for (float &x : iknb) x = random_float(0.05f);
        for (float &x : ape) x = random_float(0.08f);
        for (float &x : gate) x = random_float(0.12f);
    }

    void zero(void) {
        std::fill(input.begin(), input.end(), 0.0f);
        std::fill(qa.begin(), qa.end(), 0.0f);
        std::fill(qan.begin(), qan.end(), 0.0f);
        std::fill(qb.begin(), qb.end(), 0.0f);
        std::fill(kva.begin(), kva.end(), 0.0f);
        std::fill(kvan.begin(), kvan.end(), 0.0f);
        std::fill(kvbk.begin(), kvbk.end(), 0.0f);
        std::fill(kvbv.begin(), kvbv.end(), 0.0f);
        std::fill(output.begin(), output.end(), 0.0f);
        std::fill(iwq.begin(), iwq.end(), 0.0f);
        std::fill(iwk.begin(), iwk.end(), 0.0f);
        std::fill(iwp.begin(), iwp.end(), 0.0f);
        std::fill(iknw.begin(), iknw.end(), 0.0f);
        std::fill(iknb.begin(), iknb.end(), 0.0f);
        std::fill(ape.begin(), ape.end(), 0.0f);
        std::fill(gate.begin(), gate.end(), 0.0f);
    }
};

struct CpuState {
    int length = 0;
    std::vector<float> latent, keys, gates;

    explicit CpuState(const ColiGpuMlaConfig &c)
        : latent((size_t)c.max_context * c.kv_lora),
          keys((size_t)c.max_context * c.index_dim),
          gates((size_t)c.max_context * c.index_dim) {}
};

/* Projection/norm oracle follows glm53.c, while sparse selection is delegated
 * to the repository helper and attention is evaluated in expanded K/V space.
 * The latter is deliberately independent of the GPU's absorbed-latent form. */
static void cpu_site(std::vector<float> &out, std::vector<int> &selected,
                     CpuState &state, const Fixture &f,
                     const float *input, int rows) {
    const ColiGpuMlaConfig &c = f.config;
    const int width = c.index_select_tail
        ? c.index_topk + c.index_pool - 1 : c.index_topk;
    out.assign((size_t)rows * c.hidden, 0.0f);
    selected.assign((size_t)rows * width, -1);
    std::vector<float> qn(c.q_lora), query((size_t)c.heads * c.qk_nope);
    std::vector<float> iq((size_t)c.index_heads * c.index_dim);
    std::vector<float> head_w(c.index_heads);
    std::vector<float> context((size_t)c.heads * c.value_dim);
    for (int token = 0; token < rows; ++token) {
        const int absolute = state.length;
        const int seen = absolute + 1;
        const float *x = input + (size_t)token * c.hidden;
        matvec(qn.data(), f.qa, x, c.q_lora, c.hidden);
        rms(qn.data(), f.qan, c.q_lora, c.rms_norm_eps);
        matvec(query.data(), f.qb, qn.data(), c.heads * c.qk_nope, c.q_lora);
        float *latent = state.latent.data() + (size_t)absolute * c.kv_lora;
        matvec(latent, f.kva, x, c.kv_lora, c.hidden);
        rms(latent, f.kvan, c.kv_lora, c.rms_norm_eps);
        matvec(iq.data(), f.iwq, qn.data(),
               c.index_heads * c.index_dim, c.q_lora);
        float *key = state.keys.data() + (size_t)absolute * c.index_dim;
        matvec(key, f.iwk, x, c.index_dim, c.hidden);
        layer_norm(key, f.iknw, f.iknb, c.index_dim, c.index_norm_eps);
        matvec(state.gates.data() + (size_t)absolute * c.index_dim,
               f.gate, x, c.index_dim, c.hidden);
        matvec(head_w.data(), f.iwp, x, c.index_heads, c.hidden);
        for (float &value : head_w) value /= std::sqrt((float)c.index_heads);

        int *row = selected.data() + (size_t)token * width;
        std::vector<unsigned char> valid((size_t)seen, 1);
        if (coli_sparse_index_select_range(
                row, iq.data(), state.keys.data(), state.gates.data(),
                head_w.data(), f.ape.data(), valid.data(), seen,
                c.index_heads, c.index_dim, c.index_pool, c.index_topk,
                c.index_select_tail, seen - 1, seen) != 0) {
            std::fprintf(stderr, "CPU sparse-index helper failed\n");
            return;
        }

        std::vector<float> expanded_keys(
            (size_t)seen * c.heads * c.qk_nope);
        std::vector<float> expanded_values(
            (size_t)seen * c.heads * c.value_dim);
        for (int position = 0; position < seen; ++position) {
            const float *cached =
                state.latent.data() + (size_t)position * c.kv_lora;
            for (int h = 0; h < c.heads; ++h) {
                for (int q = 0; q < c.qk_nope; ++q) {
                    float sum = 0.0f;
                    for (int d = 0; d < c.kv_lora; ++d)
                        sum += f.kvbk[
                            ((size_t)h * c.kv_lora + d) * c.qk_nope + q] *
                            cached[d];
                    expanded_keys[
                        ((size_t)position * c.heads + h) * c.qk_nope + q] =
                        sum;
                }
                for (int v = 0; v < c.value_dim; ++v) {
                    float sum = 0.0f;
                    for (int d = 0; d < c.kv_lora; ++d)
                        sum += f.kvbv[
                            ((size_t)h * c.value_dim + v) * c.kv_lora + d] *
                            cached[d];
                    expanded_values[
                        ((size_t)position * c.heads + h) * c.value_dim + v] =
                        sum;
                }
            }
        }
        if (coli_sparse_attention_range(
                context.data(), query.data(), expanded_keys.data(),
                expanded_values.data(), row, seen, width, c.heads,
                c.qk_nope, c.value_dim, seen - 1, seen) != 0) {
            std::fprintf(stderr, "CPU expanded sparse-attention failed\n");
            return;
        }
        matvec(out.data() + (size_t)token * c.hidden, f.output,
               context.data(), c.hidden, c.heads * c.value_dim);
        state.length++;
    }
}

static ColiGpuTensor *upload(ColiGpuContext *ctx,
                             const std::vector<float> &data,
                             int rows, int columns) {
    ColiGpuTensorDesc desc = {};
    desc.data = data.data();
    desc.rows = rows;
    desc.columns = columns;
    ColiGpuTensor *tensor = nullptr;
    return coli_gpu_tensor_create(&tensor, ctx, &desc) ? tensor : nullptr;
}

struct DeviceWeights {
    ColiGpuMlaWeights weights = {};
    std::vector<ColiGpuTensor *> owned;

    bool create(ColiGpuContext *ctx, const Fixture &f) {
        const ColiGpuMlaConfig &c = f.config;
        ColiGpuTensor *items[] = {
            upload(ctx, f.qa, c.q_lora, c.hidden),
            upload(ctx, f.qan, 1, c.q_lora),
            upload(ctx, f.qb, c.heads * c.qk_nope, c.q_lora),
            upload(ctx, f.kva, c.kv_lora, c.hidden),
            upload(ctx, f.kvan, 1, c.kv_lora),
            upload(ctx, f.kvbk, c.heads * c.kv_lora, c.qk_nope),
            upload(ctx, f.kvbv, c.heads * c.value_dim, c.kv_lora),
            upload(ctx, f.output, c.hidden, c.heads * c.value_dim),
            upload(ctx, f.iwq, c.index_heads * c.index_dim, c.q_lora),
            upload(ctx, f.iwk, c.index_dim, c.hidden),
            upload(ctx, f.iwp, c.index_heads, c.hidden),
            upload(ctx, f.iknw, 1, c.index_dim),
            upload(ctx, f.iknb, 1, c.index_dim),
            upload(ctx, f.ape, c.index_pool, c.index_dim),
            upload(ctx, f.gate, c.index_dim, c.hidden),
        };
        for (ColiGpuTensor *item : items) {
            if (!item) return false;
            owned.push_back(item);
        }
        weights.q_a_proj = items[0];
        weights.q_a_norm = items[1];
        weights.q_b_proj = items[2];
        weights.kv_a_proj = items[3];
        weights.kv_a_norm = items[4];
        weights.kv_b_key = items[5];
        weights.kv_b_value = items[6];
        weights.o_proj = items[7];
        weights.index_q_proj = items[8];
        weights.index_k_proj = items[9];
        weights.index_weight_proj = items[10];
        weights.index_key_norm = items[11];
        weights.index_key_bias = items[12];
        weights.index_pool_ape = items[13];
        weights.index_pool_gate = items[14];
        return true;
    }

    ~DeviceWeights() {
        for (ColiGpuTensor *tensor : owned) coli_gpu_tensor_destroy(tensor);
    }
};

struct DeviceSite {
    ColiGpuArena *arena = nullptr;
    ColiGpuMlaState *state = nullptr;
    size_t input = 0, output = 0, scratch = 0, selected = 0, capacity = 0;

    bool create(ColiGpuContext *ctx, const ColiGpuMlaConfig &config) {
        size_t row_bytes = (size_t)config.max_rows * config.hidden * sizeof(float);
        input = 0;
        output = align256(row_bytes);
        scratch = output + align256(row_bytes);
        selected = scratch + align256(coli_gpu_mla_scratch_bytes(&config));
        capacity = selected + coli_gpu_mla_selected_bytes(&config, config.max_rows);
        return capacity > selected &&
            coli_gpu_arena_create(&arena, ctx, capacity) &&
            coli_gpu_mla_state_create(&state, ctx, &config);
    }

    ~DeviceSite() {
        coli_gpu_mla_state_destroy(state);
        coli_gpu_arena_destroy(arena);
    }
};

static int download_state(ColiGpuMlaState *state, const CpuState &want,
                          const ColiGpuMlaConfig &config) {
    size_t latent_count = (size_t)want.length * config.kv_lora;
    size_t index_count = (size_t)want.length * config.index_dim;
    std::vector<float> latent(latent_count), keys(index_count), gates(index_count);
    if (coli_gpu_mla_state_length(state) != want.length ||
        !coli_gpu_mla_state_download(state, latent.data(), latent.size(),
                                     keys.data(), keys.size(),
                                     gates.data(), gates.size()) ||
        !compare_vectors("MLA latent cache", latent.data(), want.latent.data(),
                         latent_count, 3e-4f, 0.99999f) ||
        !compare_vectors("MLA index-key cache", keys.data(), want.keys.data(),
                         index_count, 3e-4f, 0.99999f) ||
        !compare_vectors("MLA index-gate cache", gates.data(), want.gates.data(),
                         index_count, 3e-4f, 0.99999f)) {
        std::fprintf(stderr, "MLA cache snapshot mismatch at length %d\n", want.length);
        return 0;
    }
    return 1;
}

struct StateSnapshot {
    std::vector<float> latent, keys, gates;
};

static int snapshot_state(ColiGpuMlaState *state,
                          const ColiGpuMlaConfig &config,
                          StateSnapshot *snapshot) {
    int length = coli_gpu_mla_state_length(state);
    snapshot->latent.resize((size_t)length * config.kv_lora);
    snapshot->keys.resize((size_t)length * config.index_dim);
    snapshot->gates.resize((size_t)length * config.index_dim);
    return coli_gpu_mla_state_download(
        state, snapshot->latent.data(), snapshot->latent.size(),
        snapshot->keys.data(), snapshot->keys.size(),
        snapshot->gates.data(), snapshot->gates.size());
}

static int snapshots_equal(const StateSnapshot &a, const StateSnapshot &b) {
    return a.latent.size() == b.latent.size() &&
           a.keys.size() == b.keys.size() &&
           a.gates.size() == b.gates.size() &&
           !std::memcmp(a.latent.data(), b.latent.data(),
                        a.latent.size() * sizeof(float)) &&
           !std::memcmp(a.keys.data(), b.keys.data(),
                        a.keys.size() * sizeof(float)) &&
           !std::memcmp(a.gates.data(), b.gates.data(),
                        a.gates.size() * sizeof(float));
}

static int run_case(ColiGpuContext *ctx, Fixture &f, const char *name,
                    bool require_zero, bool check_stable_tie = false,
                    bool require_parallel = false) {
    DeviceWeights weights;
    DeviceSite token, chunk;
    if (!weights.create(ctx, f) ||
        !token.create(ctx, f.config) || !chunk.create(ctx, f.config))
        return 0;
    ColiGpuTelemetry alias_before = {}, alias_after = {};
    coli_gpu_context_telemetry(ctx, &alias_before);
    int alias_accepted = coli_gpu_mla_site(
        token.arena, token.input, token.input, token.scratch,
        token.selected, token.state, &weights.weights, 1, 0);
    coli_gpu_context_telemetry(ctx, &alias_after);
    if (alias_accepted || coli_gpu_mla_state_length(token.state) != 0 ||
        std::memcmp(&alias_before, &alias_after, sizeof(alias_before))) {
        std::fprintf(stderr, "MLA alias rejection changed state or telemetry\n");
        return 0;
    }
    CpuState cpu(f.config), chunk_cpu(f.config);
    std::vector<float> want, token_output(f.input.size());
    std::vector<int> want_indices;
    const int width = f.config.index_select_tail
        ? f.config.index_topk + f.config.index_pool - 1 : f.config.index_topk;
    std::vector<int> token_indices((size_t)f.config.max_rows * width);
    for (int row = 0; row < f.config.max_rows; ++row) {
        std::vector<float> one;
        std::vector<int> one_indices;
        cpu_site(one, one_indices, cpu, f, f.input.data() + (size_t)row * f.config.hidden, 1);
        if (!coli_gpu_arena_upload(
                token.arena, token.input,
                f.input.data() + (size_t)row * f.config.hidden,
                (size_t)f.config.hidden * sizeof(float)))
            return 0;
        ColiGpuTelemetry before = {}, after = {};
        int capacity_before = coli_gpu_mla_state_capacity(token.state);
        coli_gpu_context_telemetry(ctx, &before);
        if (!coli_gpu_mla_site(
                token.arena, token.output, token.input, token.scratch,
                token.selected, token.state, &weights.weights, 1, row))
            return 0;
        coli_gpu_context_telemetry(ctx, &after);
        if (row + 1 <= capacity_before &&
            after.device_allocations != before.device_allocations) {
            std::fprintf(stderr, "MLA allocated within current page capacity\n");
            return 0;
        }
        if (row + 1 > capacity_before) {
            ColiGpuMlaCacheInfo info = {};
            int pages_before =
                (capacity_before + f.config.page_tokens - 1) /
                f.config.page_tokens;
            int max_pages =
                (f.config.max_context + f.config.page_tokens - 1) /
                f.config.page_tokens;
            int expected_pages =
                pages_before > max_pages / 2 ? max_pages : pages_before * 2;
            int expected = expected_pages * f.config.page_tokens;
            if (expected > f.config.max_context) expected = f.config.max_context;
            if (!coli_gpu_mla_state_cache_info(token.state, &info) ||
                info.capacity != expected ||
                info.page_count != expected_pages ||
                info.payload_copy_bytes != 0 ||
                after.device_allocations != before.device_allocations +
                    (uint64_t)(expected_pages - pages_before + 1)) {
                std::fprintf(stderr,
                             "MLA true-page growth was not geometric/atomic\n");
                return 0;
            }
        }
        if (after.h2d_copies != before.h2d_copies ||
            after.d2h_copies != before.d2h_copies) {
            std::fprintf(stderr, "MLA hot path copied an intermediate activation\n");
            return 0;
        }
        std::vector<float> got(f.config.hidden);
        std::vector<int> got_indices(width);
        if (!coli_gpu_arena_download(token.arena, token.output, got.data(),
                                     got.size() * sizeof(float)) ||
            !coli_gpu_arena_download(token.arena, token.selected,
                                     got_indices.data(),
                                     got_indices.size() * sizeof(int)) ||
            !compare_vectors("MLA per-token output", got.data(), one.data(),
                             got.size(), 1e-3f, 0.9999f) ||
            std::memcmp(got_indices.data(), one_indices.data(),
                        got_indices.size() * sizeof(int)) ||
            !download_state(token.state, cpu, f.config))
            return 0;
        std::memcpy(token_output.data() + (size_t)row * f.config.hidden,
                    got.data(), (size_t)f.config.hidden * sizeof(float));
        std::memcpy(token_indices.data() + (size_t)row * width,
                    got_indices.data(), (size_t)width * sizeof(int));
    }
    if (coli_gpu_mla_state_capacity(token.state) < f.config.max_rows ||
        (coli_gpu_mla_state_capacity(token.state) % f.config.page_tokens &&
         coli_gpu_mla_state_capacity(token.state) != f.config.max_context)) {
        std::fprintf(stderr, "MLA geometric page capacity is invalid\n");
        return 0;
    }
    if (check_stable_tie) {
        const int expected_prefix[4] = {0, 1, 2, 3};
        const int boundary_row = 4;
        const int *boundary =
            token_indices.data() + (size_t)boundary_row * width;
        const int *last =
            token_indices.data() + (size_t)(f.config.max_rows - 1) * width;
        if (std::memcmp(boundary, expected_prefix, sizeof(expected_prefix)) ||
            boundary[f.config.index_topk] != boundary_row ||
            std::memcmp(last, expected_prefix, sizeof(expected_prefix)) ||
            last[f.config.index_topk] != f.config.max_rows - 1) {
            std::fprintf(stderr,
                         "stable lower-pool tie or tail expansion mismatch\n");
            return 0;
        }
        std::puts("  stable pool ties and tail expansion verified");
    }

    cpu_site(want, want_indices, chunk_cpu, f, f.input.data(), f.config.max_rows);
    std::vector<float> chunk_output(want.size());
    std::vector<int> chunk_indices(want_indices.size());
    if (!coli_gpu_arena_upload(chunk.arena, chunk.input, f.input.data(),
                               f.input.size() * sizeof(float)) ||
        !coli_gpu_mla_site(chunk.arena, chunk.output, chunk.input, chunk.scratch,
                           chunk.selected, chunk.state, &weights.weights,
                           f.config.max_rows, 0) ||
        !coli_gpu_arena_download(chunk.arena, chunk.output, chunk_output.data(),
                                 chunk_output.size() * sizeof(float)) ||
        !coli_gpu_arena_download(chunk.arena, chunk.selected, chunk_indices.data(),
                                 chunk_indices.size() * sizeof(int)) ||
        !compare_vectors(name, chunk_output.data(), want.data(), want.size(),
                         1e-3f, 0.9999f) ||
        std::memcmp(chunk_indices.data(), want_indices.data(),
                    want_indices.size() * sizeof(int)) ||
        !download_state(chunk.state, chunk_cpu, f.config)) {
        std::fprintf(stderr, "chunked/token MLA equivalence failed\n");
        return 0;
    }
    if (std::memcmp(chunk_output.data(), token_output.data(),
                    token_output.size() * sizeof(float))) {
        size_t at = 0;
        while (at < token_output.size() &&
               !std::memcmp(&chunk_output[at], &token_output[at], sizeof(float)))
            ++at;
        std::fprintf(stderr,
                     "chunked/token MLA output is not bitwise identical at %zu "
                     "(chunk %.9g token %.9g)\n",
                     at, chunk_output[at], token_output[at]);
        return 0;
    }
    if (std::memcmp(chunk_indices.data(), token_indices.data(),
                    token_indices.size() * sizeof(int))) {
        std::fprintf(stderr, "chunked/token MLA indices are not identical\n");
        return 0;
    }
    StateSnapshot first_state, chunk_state;
    if (!snapshot_state(token.state, f.config, &first_state) ||
        !snapshot_state(chunk.state, f.config, &chunk_state) ||
        !snapshots_equal(first_state, chunk_state)) {
        std::fprintf(stderr, "chunked/token MLA caches are not bitwise identical\n");
        return 0;
    }

    ColiGpuTelemetry before_reject = {}, after_reject = {};
    coli_gpu_context_telemetry(ctx, &before_reject);
    if (coli_gpu_mla_site(token.arena, token.output, token.input, token.scratch,
                          token.selected, token.state, &weights.weights,
                          1, f.config.max_context))
        return 0;
    coli_gpu_context_telemetry(ctx, &after_reject);
    if (std::memcmp(&before_reject, &after_reject, sizeof(before_reject)) ||
        coli_gpu_mla_state_length(token.state) != f.config.max_rows) {
        std::fprintf(stderr, "MLA maximum-context rejection changed state\n");
        return 0;
    }

    int retained_capacity = coli_gpu_mla_state_capacity(token.state);
    if (!coli_gpu_mla_state_reset(token.state) ||
        coli_gpu_mla_state_length(token.state) != 0 ||
        coli_gpu_mla_state_capacity(token.state) != retained_capacity)
        return 0;
    std::vector<float> replay(token_output.size());
    std::vector<int> replay_indices(token_indices.size());
    for (int row = 0; row < f.config.max_rows; ++row) {
        if (!coli_gpu_arena_upload(
                token.arena, token.input,
                f.input.data() + (size_t)row * f.config.hidden,
                (size_t)f.config.hidden * sizeof(float)) ||
            !coli_gpu_mla_site(token.arena, token.output, token.input,
                               token.scratch, token.selected, token.state,
                               &weights.weights, 1, row) ||
            !coli_gpu_arena_download(
                token.arena, token.output,
                replay.data() + (size_t)row * f.config.hidden,
                (size_t)f.config.hidden * sizeof(float)) ||
            !coli_gpu_arena_download(
                token.arena, token.selected,
                replay_indices.data() + (size_t)row * width,
                (size_t)width * sizeof(int)))
            return 0;
    }
    if (std::memcmp(replay.data(), token_output.data(),
                    replay.size() * sizeof(float)) ||
        std::memcmp(replay_indices.data(), token_indices.data(),
                    replay_indices.size() * sizeof(int))) {
        std::fprintf(stderr, "MLA reset/replay is not bitwise deterministic\n");
        return 0;
    }
    StateSnapshot replay_state;
    if (!snapshot_state(token.state, f.config, &replay_state) ||
        !snapshots_equal(first_state, replay_state)) {
        std::fprintf(stderr, "MLA reset/replay caches are not bitwise deterministic\n");
        return 0;
    }
    if (require_zero)
        for (float value : replay)
            if (value != 0.0f) return 0;
    if (require_parallel) {
        ColiGpuMlaLaunchInfo launch = {};
        if (!coli_gpu_mla_state_launch_info(chunk.state, &launch) ||
            launch.kernel_launches < 2 || launch.max_grid_blocks < 2 ||
            launch.max_block_threads < 64) {
            std::fprintf(stderr,
                         "production MLA site did not launch parallel work\n");
            return 0;
        }
    }
    return 1;
}

static int test_paged_growth_failure_atomicity(ColiGpuContext *ctx) {
    Fixture f(5, 7, 2, 5, 4, 3, 2, 2, 3, 2, 4, 1, 16, 2);
    f.randomize();
    DeviceWeights weights;
    DeviceSite site;
    if (!weights.create(ctx, f) || !site.create(ctx, f.config) ||
        !coli_gpu_arena_upload(site.arena, site.input, f.input.data(),
                               2u * f.config.hidden * sizeof(float)) ||
        !coli_gpu_mla_site(site.arena, site.output, site.input, site.scratch,
                           site.selected, site.state, &weights.weights, 2, 0))
        return 0;
    StateSnapshot prefix;
    if (!snapshot_state(site.state, f.config, &prefix)) return 0;
    ColiGpuMlaCacheInfo baseline = {};
    if (!coli_gpu_mla_state_cache_info(site.state, &baseline) ||
        baseline.logical_length != 2 || baseline.capacity != 2 ||
        baseline.page_count != 1 || baseline.payload_copy_bytes != 0)
        return 0;

    struct FaultCase {
        ColiGpuMlaFaultPoint point;
        int occurrence;
        const char *name;
    };
    const FaultCase faults[] = {
        {COLI_GPU_MLA_FAULT_TABLE_ALLOC, 0, "table allocation"},
        {COLI_GPU_MLA_FAULT_TABLE_COPY, 0, "table copy"},
        {COLI_GPU_MLA_FAULT_PAGE_ALLOC, 0, "page allocation 0"},
        {COLI_GPU_MLA_FAULT_PAGE_ALLOC, 1, "page allocation 1"},
        {COLI_GPU_MLA_FAULT_PAGE_ALLOC, 2, "page allocation 2"},
        {COLI_GPU_MLA_FAULT_PAGE_PUBLISH, 0, "page publication 0"},
        {COLI_GPU_MLA_FAULT_PAGE_PUBLISH, 1, "page publication 1"},
        {COLI_GPU_MLA_FAULT_PAGE_PUBLISH, 2, "page publication 2"},
        {COLI_GPU_MLA_FAULT_TABLE_PUBLISH, 0, "table publication"},
    };
    if (!coli_gpu_arena_upload(
            site.arena, site.input,
            f.input.data() + 2u * f.config.hidden,
            3u * f.config.hidden * sizeof(float)))
        return 0;
    for (const FaultCase &fault : faults) {
        if (!coli_gpu_mla_state_inject_growth_fault(
                site.state, fault.point, fault.occurrence))
            return 0;
        int accepted = coli_gpu_mla_site(
            site.arena, site.output, site.input, site.scratch, site.selected,
            site.state, &weights.weights, 3, 2);
        ColiGpuMlaCacheInfo after = {};
        StateSnapshot after_prefix;
        if (accepted ||
            !coli_gpu_mla_state_cache_info(site.state, &after) ||
            after.logical_length != baseline.logical_length ||
            after.capacity != baseline.capacity ||
            after.page_count != baseline.page_count ||
            after.page_table_capacity != baseline.page_table_capacity ||
            after.payload_copy_bytes != 0 ||
            !snapshot_state(site.state, f.config, &after_prefix) ||
            !snapshots_equal(prefix, after_prefix) ||
            !coli_gpu_context_healthy(ctx)) {
            std::fprintf(stderr,
                         "paged MLA fault was not atomic at %s\n", fault.name);
            return 0;
        }
    }
    if (!coli_gpu_mla_state_inject_growth_fault(
            site.state, COLI_GPU_MLA_FAULT_NONE, 0) ||
        !coli_gpu_mla_site(site.arena, site.output, site.input, site.scratch,
                           site.selected, site.state, &weights.weights, 3, 2))
        return 0;
    ColiGpuMlaCacheInfo grown = {};
    StateSnapshot full;
    if (!coli_gpu_mla_state_cache_info(site.state, &grown) ||
        grown.logical_length != 5 || grown.capacity != 8 ||
        grown.page_count != 4 || grown.page_table_capacity != 4 ||
        grown.payload_copy_bytes != 0 ||
        !snapshot_state(site.state, f.config, &full) ||
        std::memcmp(full.latent.data(), prefix.latent.data(),
                    prefix.latent.size() * sizeof(float)) ||
        std::memcmp(full.keys.data(), prefix.keys.data(),
                    prefix.keys.size() * sizeof(float)) ||
        std::memcmp(full.gates.data(), prefix.gates.data(),
                    prefix.gates.size() * sizeof(float))) {
        std::fprintf(stderr, "true-page growth did not preserve prefix\n");
        return 0;
    }
    std::puts("  paged growth failure boundaries are atomic");
    return 1;
}

static int test_device_finite_status(ColiGpuContext *ctx) {
    Fixture f(2, 7, 2, 5, 4, 3, 2, 2, 3, 2, 2, 1, 8, 2);
    f.randomize();
    DeviceWeights weights;
    DeviceSite site;
    if (!weights.create(ctx, f) || !site.create(ctx, f.config)) return 0;
    std::vector<float> row(f.input.begin(), f.input.begin() + f.config.hidden);
    const float bad[] = {
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
    };
    for (float value : bad) {
        row[1] = value;
        if (!coli_gpu_arena_upload(site.arena, site.input, row.data(),
                                   row.size() * sizeof(float)))
            return 0;
        ColiGpuTelemetry before = {}, after = {};
        coli_gpu_context_telemetry(ctx, &before);
        int accepted = coli_gpu_mla_site(
            site.arena, site.output, site.input, site.scratch, site.selected,
            site.state, &weights.weights, 1, 0);
        coli_gpu_context_telemetry(ctx, &after);
        if (accepted ||
            coli_gpu_mla_state_status(site.state) !=
                COLI_GPU_MLA_STATUS_NONFINITE_INPUT ||
            coli_gpu_mla_state_length(site.state) != 0 ||
            after.d2h_copies != before.d2h_copies ||
            after.d2h_bytes != before.d2h_bytes ||
            !coli_gpu_context_healthy(ctx)) {
            std::fprintf(stderr,
                         "device non-finite activation was not rejected cleanly\n");
            return 0;
        }
    }

    row.assign(f.input.begin(), f.input.begin() + f.config.hidden);
    if (!coli_gpu_arena_upload(site.arena, site.input, row.data(),
                               row.size() * sizeof(float)) ||
        !coli_gpu_mla_site(site.arena, site.output, site.input, site.scratch,
                           site.selected, site.state, &weights.weights, 1, 0) ||
        !coli_gpu_mla_state_test_corrupt_cache(
            site.state, COLI_GPU_MLA_CACHE_LATENT, 0, 0,
            std::numeric_limits<float>::quiet_NaN()) ||
        !coli_gpu_arena_upload(site.arena, site.input, row.data(),
                               row.size() * sizeof(float)))
        return 0;
    int accepted = coli_gpu_mla_site(
        site.arena, site.output, site.input, site.scratch, site.selected,
        site.state, &weights.weights, 1, 1);
    if (accepted ||
        coli_gpu_mla_state_status(site.state) !=
            COLI_GPU_MLA_STATUS_NONFINITE_CACHE ||
        coli_gpu_mla_state_length(site.state) != 1 ||
        !coli_gpu_context_healthy(ctx) ||
        !coli_gpu_mla_state_reset(site.state) ||
        coli_gpu_mla_state_status(site.state) != COLI_GPU_MLA_STATUS_OK ||
        !coli_gpu_mla_site(site.arena, site.output, site.input, site.scratch,
                           site.selected, site.state, &weights.weights, 1, 0)) {
        std::fprintf(stderr, "cache contamination health/recovery failed\n");
        return 0;
    }
    std::puts("  device finite status rejects input/cache contamination");
    return 1;
}

static int test_long_context_pages(ColiGpuContext *ctx) {
    const int total_rows = 257;
    Fixture f(13, 5, 2, 4, 3, 3, 2, 2, 3, 4, 8, 1,
              total_rows, 16);
    f.input.resize((size_t)total_rows * f.config.hidden);
    f.randomize();
    DeviceWeights weights;
    DeviceSite site;
    if (!weights.create(ctx, f) || !site.create(ctx, f.config)) return 0;
    const int width = coli_sparse_index_width(
        f.config.index_topk, f.config.index_pool,
        f.config.index_select_tail);
    std::vector<float> first_outputs(
        (size_t)total_rows * f.config.hidden);
    std::vector<int> first_indices((size_t)total_rows * width);

    auto execute = [&](bool record) {
        CpuState cpu(f.config);
        int position = 0;
        while (position < total_rows) {
            int rows = total_rows - position;
            if (rows > f.config.max_rows) rows = f.config.max_rows;
            StateSnapshot prefix;
            if (!snapshot_state(site.state, f.config, &prefix)) return false;
            std::vector<float> want;
            std::vector<int> want_indices;
            cpu_site(want, want_indices, cpu, f,
                     f.input.data() + (size_t)position * f.config.hidden,
                     rows);
            if (!coli_gpu_arena_upload(
                    site.arena, site.input,
                    f.input.data() + (size_t)position * f.config.hidden,
                    (size_t)rows * f.config.hidden * sizeof(float)) ||
                !coli_gpu_mla_site(
                    site.arena, site.output, site.input, site.scratch,
                    site.selected, site.state, &weights.weights,
                    rows, position))
                return false;
            std::vector<float> got(want.size());
            std::vector<int> got_indices(want_indices.size());
            if (!coli_gpu_arena_download(
                    site.arena, site.output, got.data(),
                    got.size() * sizeof(float)) ||
                !coli_gpu_arena_download(
                    site.arena, site.selected, got_indices.data(),
                    got_indices.size() * sizeof(int)) ||
                !compare_vectors("MLA long-context output", got.data(),
                                 want.data(), got.size(), 1e-3f, 0.9999f) ||
                std::memcmp(got_indices.data(), want_indices.data(),
                            got_indices.size() * sizeof(int)))
                return false;
            StateSnapshot after;
            ColiGpuMlaCacheInfo info = {};
            if (!snapshot_state(site.state, f.config, &after) ||
                !coli_gpu_mla_state_cache_info(site.state, &info) ||
                info.payload_copy_bytes != 0 ||
                std::memcmp(after.latent.data(), prefix.latent.data(),
                            prefix.latent.size() * sizeof(float)) ||
                std::memcmp(after.keys.data(), prefix.keys.data(),
                            prefix.keys.size() * sizeof(float)) ||
                std::memcmp(after.gates.data(), prefix.gates.data(),
                            prefix.gates.size() * sizeof(float)))
                return false;
            for (int local = 0; local < rows; ++local) {
                int seen = position + local + 1;
                int tail = seen % f.config.index_pool;
                const int *chosen =
                    got_indices.data() + (size_t)local * width;
                for (int j = 0; j < f.config.index_pool - 1; ++j) {
                    int expected = j < tail ? seen - tail + j : -1;
                    if (chosen[f.config.index_topk + j] != expected)
                        return false;
                }
                for (int slot = 0; slot < f.config.index_topk;
                     slot += f.config.index_pool) {
                    if (chosen[slot] < 0) continue;
                    if (chosen[slot] % f.config.index_pool ||
                        chosen[slot] + f.config.index_pool > seen - tail)
                        return false;
                    for (int j = 1; j < f.config.index_pool; ++j)
                        if (chosen[slot + j] != chosen[slot] + j)
                            return false;
                }
            }
            float *saved_out = first_outputs.data() +
                (size_t)position * f.config.hidden;
            int *saved_indices =
                first_indices.data() + (size_t)position * width;
            if (record) {
                std::memcpy(saved_out, got.data(),
                            got.size() * sizeof(float));
                std::memcpy(saved_indices, got_indices.data(),
                            got_indices.size() * sizeof(int));
            } else if (std::memcmp(saved_out, got.data(),
                                   got.size() * sizeof(float)) ||
                       std::memcmp(saved_indices, got_indices.data(),
                                   got_indices.size() * sizeof(int))) {
                return false;
            }
            position += rows;
        }
        return download_state(site.state, cpu, f.config) != 0;
    };

    if (!execute(true)) return 0;
    ColiGpuMlaCacheInfo final_info = {};
    if (!coli_gpu_mla_state_cache_info(site.state, &final_info) ||
        final_info.logical_length != total_rows ||
        final_info.capacity != total_rows ||
        final_info.page_count != 17 ||
        final_info.page_table_capacity != 17 ||
        !coli_gpu_mla_state_reset(site.state) ||
        !execute(false)) {
        std::fprintf(stderr, "long-context page growth/replay failed\n");
        return 0;
    }
    std::puts("  257-token page growth and all pool/tail boundaries verified");
    return 1;
}

static int test_validation_and_production_dims(ColiGpuContext *ctx) {
    Fixture production(1, 8, 1, 1536, 512, 256, 4, 1, 4, 4, 4, 1, 8, 4);
    production.randomize();
    if (!coli_gpu_mla_scratch_bytes(&production.config) ||
        coli_gpu_mla_selected_bytes(&production.config, 1) !=
            7 * sizeof(int))
        return 0;
    ColiGpuMlaConfig invalid = production.config;
    invalid.qk_rope = 2; /* glm53.c explicitly defines GLM-5.3 as NoPE. */
    if (coli_gpu_mla_scratch_bytes(&invalid)) {
        std::fprintf(stderr, "MLA accepted a RoPE layout absent from CPU semantics\n");
        return 0;
    }
    invalid = production.config;
    invalid.rms_norm_eps = NAN;
    ColiGpuMlaState *state = nullptr;
    if (coli_gpu_mla_scratch_bytes(&invalid) ||
        coli_gpu_mla_state_create(&state, ctx, &invalid) || state) {
        std::fprintf(stderr, "MLA accepted non-finite configuration\n");
        return 0;
    }
    float nonfinite_weight[2] = {1.0f, INFINITY};
    ColiGpuTensorDesc tensor_desc = {};
    tensor_desc.data = nonfinite_weight;
    tensor_desc.rows = 1;
    tensor_desc.columns = 2;
    ColiGpuTensor *tensor = nullptr;
    ColiGpuTelemetry before = {}, after = {};
    coli_gpu_context_telemetry(ctx, &before);
    int accepted = coli_gpu_tensor_create(&tensor, ctx, &tensor_desc);
    coli_gpu_context_telemetry(ctx, &after);
    coli_gpu_tensor_destroy(tensor);
    if (accepted || tensor ||
        std::memcmp(&before, &after, sizeof(before))) {
        std::fprintf(stderr, "MLA accepted/allocated non-finite weights\n");
        return 0;
    }
    std::puts("  production MLA dimensions and NoPE contract accepted");
    return 1;
}

static int test_eleven_layer_ownership(ColiGpuContext *ctx) {
    const int layers_count = 11;
    Fixture f(3, 4, 2, 4, 3, 2, 2, 2, 2, 2, 2, 1, 9, 2);
    f.randomize();
    std::vector<float> embedding((size_t)5 * f.config.hidden, 0.1f);
    std::vector<float> norm(f.config.hidden, 1.0f);
    std::vector<float> head((size_t)5 * f.config.hidden, 0.2f);
    std::vector<ColiGlm53GpuMlaLayerDesc> layers(layers_count);
    for (ColiGlm53GpuMlaLayerDesc &layer : layers) {
        layer.q_a_proj = f.qa.data();
        layer.q_a_norm = f.qan.data();
        layer.q_b_proj = f.qb.data();
        layer.kv_a_proj = f.kva.data();
        layer.kv_a_norm = f.kvan.data();
        layer.kv_b_key = f.kvbk.data();
        layer.kv_b_value = f.kvbv.data();
        layer.o_proj = f.output.data();
        layer.index_q_proj = f.iwq.data();
        layer.index_k_proj = f.iwk.data();
        layer.index_weight_proj = f.iwp.data();
        layer.index_key_norm = f.iknw.data();
        layer.index_key_bias = f.iknb.data();
        layer.index_pool_ape = f.ape.data();
        layer.index_pool_gate = f.gate.data();
    }
    ColiGlm53GpuModelDesc desc = {};
    desc.hidden_size = f.config.hidden;
    desc.stream_count = 1;
    desc.vocab_size = 5;
    desc.max_prefill_rows = f.config.max_rows;
    desc.max_context_tokens = f.config.max_context;
    desc.norm_eps = 1e-6f;
    desc.hc_eps = 1e-6f;
    desc.embedding = embedding.data();
    desc.final_norm = norm.data();
    desc.lm_head = head.data();
    desc.lm_head_format = 0;
    desc.mla_heads = f.config.heads;
    desc.mla_q_lora = f.config.q_lora;
    desc.mla_kv_lora = f.config.kv_lora;
    desc.mla_qk_nope = f.config.qk_nope;
    desc.mla_qk_rope = 0;
    desc.mla_value_dim = f.config.value_dim;
    desc.mla_index_heads = f.config.index_heads;
    desc.mla_index_dim = f.config.index_dim;
    desc.mla_index_pool = f.config.index_pool;
    desc.mla_index_topk = f.config.index_topk;
    desc.mla_index_select_tail = f.config.index_select_tail;
    desc.mla_page_tokens = f.config.page_tokens;
    desc.mla_layers = layers.data();
    desc.mla_layer_count = layers_count;
    ColiGlm53GpuModel *model = nullptr;
    ColiGlm53GpuSession *session = nullptr;
    if (!coli_glm53_gpu_model_create(&model, ctx, &desc) ||
        !coli_glm53_gpu_session_create(
            &session, model, f.config.max_rows))
        return 0;

    auto session_snapshot = [&](int layer, StateSnapshot *snapshot) {
        int length = 0, capacity = 0;
        if (!coli_glm53_gpu_session_mla_state_info(
                session, layer, &length, &capacity))
            return false;
        snapshot->latent.resize((size_t)length * f.config.kv_lora);
        snapshot->keys.resize((size_t)length * f.config.index_dim);
        snapshot->gates.resize((size_t)length * f.config.index_dim);
        return coli_glm53_gpu_session_mla_state_download(
            session, layer,
            snapshot->latent.data(), snapshot->latent.size(),
            snapshot->keys.data(), snapshot->keys.size(),
            snapshot->gates.data(), snapshot->gates.size()) != 0;
    };
    const int width = coli_sparse_index_width(
        f.config.index_topk, f.config.index_pool,
        f.config.index_select_tail);
    std::vector<std::vector<float>> layer_inputs(layers_count);
    std::vector<std::vector<float>> layer_outputs(layers_count);
    std::vector<std::vector<int>> layer_indices(layers_count);
    std::vector<StateSnapshot> snapshots(layers_count);
    for (int layer = 0; layer < layers_count; ++layer) {
        int total = 4 + layer % 3;
        layer_inputs[layer].resize((size_t)total * f.config.hidden);
        for (size_t i = 0; i < layer_inputs[layer].size(); ++i)
            layer_inputs[layer][i] =
                0.03f * (float)(layer + 1) +
                0.01f * (float)((int)i - f.config.hidden);
        CpuState cpu(f.config);
        int position = 0;
        while (position < total) {
            int rows = total - position;
            if (rows > f.config.max_rows) rows = f.config.max_rows;
            std::vector<float> want;
            std::vector<int> want_indices;
            cpu_site(
                want, want_indices, cpu, f,
                layer_inputs[layer].data() +
                    (size_t)position * f.config.hidden,
                rows);
            if (!coli_glm53_gpu_session_mla_upload_input(
                    session,
                    layer_inputs[layer].data() +
                        (size_t)position * f.config.hidden,
                    rows) ||
                !coli_glm53_gpu_session_mla_site(
                    session, layer, rows, position))
                return 0;
            std::vector<float> got(want.size());
            std::vector<int> got_indices(want_indices.size());
            if (!coli_glm53_gpu_session_mla_download_output(
                    session, got.data(), got_indices.data(),
                    got_indices.size(), rows) ||
                !compare_vectors("11-layer output", got.data(), want.data(),
                                 got.size(), 1e-3f, 0.9999f) ||
                std::memcmp(got_indices.data(), want_indices.data(),
                            got_indices.size() * sizeof(int)))
                return 0;
            layer_outputs[layer].insert(
                layer_outputs[layer].end(), got.begin(), got.end());
            layer_indices[layer].insert(
                layer_indices[layer].end(),
                got_indices.begin(), got_indices.end());
            position += rows;
        }
        int length = -1, capacity = -1;
        if (!coli_glm53_gpu_session_mla_state_info(
                session, layer, &length, &capacity) ||
            length != total || capacity != (total == 4 ? 4 : 8) ||
            !session_snapshot(layer, &snapshots[layer]) ||
            !compare_vectors(
                "11-layer latent", snapshots[layer].latent.data(),
                cpu.latent.data(), snapshots[layer].latent.size(),
                3e-4f, 0.99999f) ||
            !compare_vectors(
                "11-layer index keys", snapshots[layer].keys.data(),
                cpu.keys.data(), snapshots[layer].keys.size(),
                3e-4f, 0.99999f) ||
            !compare_vectors(
                "11-layer index gates", snapshots[layer].gates.data(),
                cpu.gates.data(), snapshots[layer].gates.size(),
                3e-4f, 0.99999f))
            return 0;
        for (int other = layer + 1; other < layers_count; ++other)
            if (!coli_glm53_gpu_session_mla_state_info(
                    session, other, &length, &capacity) ||
                length != 0 || capacity != f.config.page_tokens)
                return 0;
    }

    if (!coli_glm53_gpu_session_mla_reset_layer(session, 5))
        return 0;
    for (int layer = 0; layer < layers_count; ++layer) {
        int length = -1, capacity = -1;
        if (!coli_glm53_gpu_session_mla_state_info(
                session, layer, &length, &capacity) ||
            length != (layer == 5 ? 0 : 4 + layer % 3) ||
            capacity != (layer % 3 == 0 ? 4 : 8))
            return 0;
        if (layer != 5) {
            StateSnapshot unchanged;
            if (!session_snapshot(layer, &unchanged) ||
                !snapshots_equal(unchanged, snapshots[layer]))
                return 0;
        }
    }

    CpuState replay_cpu(f.config);
    int replay_position = 0;
    std::vector<float> replay_outputs;
    std::vector<int> replay_indices;
    while (replay_position < 6) {
        int rows = 6 - replay_position;
        if (rows > f.config.max_rows) rows = f.config.max_rows;
        std::vector<float> ignored;
        std::vector<int> ignored_indices;
        cpu_site(
            ignored, ignored_indices, replay_cpu, f,
            layer_inputs[5].data() +
                (size_t)replay_position * f.config.hidden,
            rows);
        if (!coli_glm53_gpu_session_mla_upload_input(
                session, layer_inputs[5].data() +
                    (size_t)replay_position * f.config.hidden, rows) ||
            !coli_glm53_gpu_session_mla_site(
                session, 5, rows, replay_position))
            return 0;
        std::vector<float> got((size_t)rows * f.config.hidden);
        std::vector<int> got_indices((size_t)rows * width);
        if (!coli_glm53_gpu_session_mla_download_output(
                session, got.data(), got_indices.data(),
                got_indices.size(), rows))
            return 0;
        replay_outputs.insert(replay_outputs.end(), got.begin(), got.end());
        replay_indices.insert(
            replay_indices.end(), got_indices.begin(), got_indices.end());
        replay_position += rows;
    }
    StateSnapshot replay_state;
    if (std::memcmp(replay_outputs.data(), layer_outputs[5].data(),
                    replay_outputs.size() * sizeof(float)) ||
        std::memcmp(replay_indices.data(), layer_indices[5].data(),
                    replay_indices.size() * sizeof(int)) ||
        !session_snapshot(5, &replay_state) ||
        !snapshots_equal(replay_state, snapshots[5]))
        return 0;

    if (!coli_glm53_gpu_session_reset(session)) return 0;
    for (int layer = 0; layer < layers_count; ++layer) {
        int length = -1, capacity = -1;
        if (!coli_glm53_gpu_session_mla_state_info(
                session, layer, &length, &capacity) ||
            length != 0 || capacity != (layer % 3 == 0 ? 4 : 8))
            return 0;
    }
    coli_glm53_gpu_session_destroy(session);
    coli_glm53_gpu_model_destroy(model);
    std::puts("  all 11 MLA layers grow/reset/replay independently");
    return 1;
}

int main(void) {
    ColiGpuContext *ctx = nullptr;
    if (!coli_gpu_context_create(&ctx, 0)) {
        std::fprintf(stderr, "Task 4 requires a HIP device\n");
        return 1;
    }
    Fixture zero(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1);
    zero.zero();
    Fixture pooled(17, 9, 2, 5, 4, 3, 2, 2, 3, 2, 4, 1, 17, 2);
    pooled.randomize(true);
    Fixture production(
        9, 16, 4, 1536, 512, 256, 8, 4, 16, 4, 8, 1, 9, 4);
    production.randomize();
    int ok = test_validation_and_production_dims(ctx) &&
             test_paged_growth_failure_atomicity(ctx) &&
             test_device_finite_status(ctx) &&
             test_long_context_pages(ctx) &&
             run_case(ctx, zero, "MLA zero/minimum fixture", true) &&
             run_case(ctx, pooled, "MLA pooled-tail oracle", false, true) &&
             run_case(ctx, production,
                      "MLA production representative dims", false, false,
                      true) &&
             test_eleven_layer_ownership(ctx);
    coli_gpu_context_destroy(ctx);
    if (!ok) return 1;
    std::puts("glm53 device-resident MLA/DSA HIP: ok");
    return 0;
}
