#include "../backend_cuda.h"
#include "../glm53_gpu.h"
#include "../sparse_index.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <vector>

static int create_minimal(ColiGpuContext *ctx,
                          ColiGlm53GpuModel **model,
                          ColiGlm53GpuSession **session) {
    static const float embedding[6] = {
        1.0f, 2.0f, 3.0f,
        -1.0f, 0.5f, 2.0f
    };
    static const float norm[3] = {1.0f, 1.0f, 1.0f};
    static const float head[6] = {
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f
    };
    ColiGlm53GpuModelDesc desc;
    memset(&desc, 0, sizeof(desc));
    desc.hidden_size = 3;
    desc.stream_count = 4;
    desc.vocab_size = 2;
    desc.max_prefill_rows = 2;
    desc.max_context_tokens = 4;
    desc.norm_eps = 1e-6f;
    desc.hc_eps = 1e-6f;
    desc.embedding = embedding;
    desc.final_norm = norm;
    desc.lm_head = head;
    return coli_glm53_gpu_model_create(model, ctx, &desc) &&
           coli_glm53_gpu_session_create(session, *model, 4);
}

static int test_full_forward_and_output_boundary(ColiGpuContext *ctx) {
    ColiGlm53GpuModel *model = NULL;
    ColiGlm53GpuSession *session = NULL;
    float logits[2] = {NAN, NAN};
    const int tokens[2] = {0, 1};
    ColiGpuTelemetry before, after;
    coli_gpu_context_telemetry(ctx, &before);
    int ok = create_minimal(ctx, &model, &session) &&
             coli_glm53_gpu_forward(session, tokens, 2, logits);
    coli_gpu_context_telemetry(ctx, &after);
    ok = ok && isfinite(logits[0]) && isfinite(logits[1]) &&
         after.host_activation_h2d_copies == before.host_activation_h2d_copies &&
         after.host_activation_d2h_copies ==
             before.host_activation_d2h_copies + 1;
    coli_glm53_gpu_session_destroy(session);
    coli_glm53_gpu_model_destroy(model);
    if (!ok) fprintf(stderr, "FAIL: full forward/output boundary\n");
    return ok;
}

static int test_offset_binary_grouped_int4_projection(ColiGpuContext *ctx) {
    unsigned char packed[32];
    float scales[1] = {1.0f};
    float input[64], output = NAN;
    memset(packed, 0x88, sizeof(packed)); /* stored zero is level + 8 */
    for (int i = 0; i < 64; ++i) input[i] = 1.0f;
    ColiGpuTensorDesc desc = {
        packed, scales, 4, 1, 64, 64
    };
    ColiGpuTensor *weight = NULL;
    ColiGpuArena *arena = NULL;
    int ok = coli_gpu_tensor_create(&weight, ctx, &desc) &&
        coli_gpu_arena_create(&arena, ctx, 65 * sizeof(float)) &&
        coli_gpu_arena_upload_activation(
            arena, 0, input, sizeof(input)) &&
        coli_gpu_projection(
            arena, 64 * sizeof(float), 0, weight, 1, 64, 1) &&
        coli_gpu_arena_download_activation(
            arena, 64 * sizeof(float), &output, sizeof(output)) &&
        output == 0.0f;
    coli_gpu_arena_destroy(arena);
    coli_gpu_tensor_destroy(weight);
    if (!ok)
        fprintf(stderr,
                "FAIL: grouped int4 offset-binary zero decoded as %.9g\n",
                output);
    return ok;
}

static int test_fault_preserves_output_and_poison_context(
    ColiGpuContext *ctx, ColiGpuFaultPoint point) {
    ColiGlm53GpuModel *model = NULL;
    ColiGlm53GpuSession *session = NULL;
    const int token = 0;
    const float sentinel[2] = {1234.5f, -987.25f};
    float logits[2];
    memcpy(logits, sentinel, sizeof(logits));
    int ok = create_minimal(ctx, &model, &session) &&
             coli_gpu_context_inject_fault(
                 ctx, point, 0) &&
             !coli_glm53_gpu_forward(session, &token, 1, logits) &&
             !memcmp(logits, sentinel, sizeof(logits)) &&
             !coli_gpu_context_healthy(ctx);
    coli_glm53_gpu_session_destroy(session);
    coli_glm53_gpu_model_destroy(model);
    if (!ok) fprintf(stderr, "FAIL: request fault contract\n");
    return ok;
}

static int test_arithmetic_nonfinite_preserves_output_and_poison_context(
    ColiGpuContext *ctx) {
    static const float embedding[1] = {1.0f};
    static const float final_norm[1] = {FLT_MAX};
    static const float head[1] = {FLT_MAX};
    ColiGlm53GpuModelDesc desc = {};
    desc.hidden_size = 1;
    desc.stream_count = 4;
    desc.vocab_size = 1;
    desc.max_prefill_rows = 1;
    desc.max_context_tokens = 1;
    desc.norm_eps = 0.0f;
    desc.hc_eps = 1e-6f;
    desc.embedding = embedding;
    desc.final_norm = final_norm;
    desc.lm_head = head;

    ColiGlm53GpuModel *model = NULL;
    ColiGlm53GpuSession *session = NULL;
    const int token = 0;
    const float sentinel = 2468.5f;
    float logits = sentinel;
    int ok = coli_glm53_gpu_model_create(&model, ctx, &desc) &&
             coli_glm53_gpu_session_create(&session, model, 1) &&
             !coli_glm53_gpu_forward(session, &token, 1, &logits) &&
             !memcmp(&logits, &sentinel, sizeof(logits)) &&
             !coli_gpu_context_healthy(ctx);
    coli_glm53_gpu_session_destroy(session);
    coli_glm53_gpu_model_destroy(model);
    if (!ok)
        fprintf(stderr,
                "FAIL: arithmetic non-finite final logits contract\n");
    return ok;
}

static int test_max_context_fails_without_truncation(ColiGpuContext *ctx) {
    ColiGlm53GpuModel *model = NULL;
    ColiGlm53GpuSession *session = NULL;
    const int tokens[2] = {0, 1};
    float logits[2];
    int ok = create_minimal(ctx, &model, &session) &&
             coli_glm53_gpu_forward(session, tokens, 2, logits) &&
             coli_glm53_gpu_forward(session, tokens, 2, logits);
    const float sentinel[2] = {412.0f, -913.0f};
    memcpy(logits, sentinel, sizeof(logits));
    ok = ok && !coli_glm53_gpu_forward(session, tokens, 1, logits) &&
         !memcmp(logits, sentinel, sizeof(logits)) &&
         coli_gpu_context_healthy(ctx);
    coli_glm53_gpu_session_destroy(session);
    coli_glm53_gpu_model_destroy(model);
    if (!ok) fprintf(stderr, "FAIL: max-context failure contract\n");
    return ok;
}

static int expert_load_calls;

static int load_zero_expert(void *, int, int expert_id, int *slot,
                            ColiGlm53GpuExpertDesc *expert) {
    static const unsigned char packed[2] = {0x88, 0x88};
    static const float scales[2] = {1.0f, 1.0f};
    if (expert_id != 0 || !slot || !expert) return 0;
    memset(expert, 0, sizeof(*expert));
    expert->gate = {packed, scales, 4, 2, 2, 64};
    expert->up = expert->gate;
    expert->down = expert->gate;
    *slot = 0;
    expert_load_calls++;
    return 1;
}

static ColiGlm53GpuWeightDesc f32_weight(
    const float *data, int rows, int columns) {
    ColiGlm53GpuWeightDesc weight = {};
    weight.data = data;
    weight.rows = rows;
    weight.columns = columns;
    return weight;
}

static int compare_oracle(const float *got, const float *want, size_t count) {
    double diff2 = 0.0, want2 = 0.0, got2 = 0.0, dot = 0.0;
    for (size_t i = 0; i < count; ++i) {
        if (!isfinite(got[i]) || !isfinite(want[i])) return 0;
        double difference = (double)got[i] - want[i];
        diff2 += difference * difference;
        want2 += (double)want[i] * want[i];
        got2 += (double)got[i] * got[i];
        dot += (double)got[i] * want[i];
    }
    double relative = sqrt(diff2 / (want2 + 1e-30));
    double cosine = dot / sqrt((got2 + 1e-30) * (want2 + 1e-30));
    printf("  full pipeline rel_l2 %.3e cosine %.8f\n", relative, cosine);
    return relative <= 1e-3 && cosine >= 0.9999;
}

static float ref_sigmoid(float value) {
    if (value >= 0.0f) {
        float decay = expf(-value);
        return 1.0f / (1.0f + decay);
    }
    float growth = expf(value);
    return growth / (1.0f + growth);
}

static void ref_matvec(float *out, const float *weight, const float *input,
                       int rows, int columns) {
    for (int row = 0; row < rows; ++row) {
        float sum = 0.0f;
        for (int column = 0; column < columns; ++column)
            sum += weight[(size_t)row * columns + column] * input[column];
        out[row] = sum;
    }
}

static void ref_rmsnorm(float *out, const float *input, const float *weight,
                        int rows, int hidden, float eps) {
    for (int row = 0; row < rows; ++row) {
        const float *x = input + (size_t)row * hidden;
        float square = 0.0f;
        for (int d = 0; d < hidden; ++d) square += x[d] * x[d];
        float inverse = 1.0f / sqrtf(square / hidden + eps);
        for (int d = 0; d < hidden; ++d)
            out[(size_t)row * hidden + d] =
                x[d] * inverse * weight[d];
    }
}

static void ref_layernorm(float *values, const float *weight,
                          const float *bias, int count, float eps) {
    float mean = 0.0f;
    for (int i = 0; i < count; ++i) mean += values[i];
    mean /= count;
    float variance = 0.0f;
    for (int i = 0; i < count; ++i) {
        float centered = values[i] - mean;
        variance += centered * centered;
    }
    float inverse = 1.0f / sqrtf(variance / count + eps);
    for (int i = 0; i < count; ++i)
        values[i] = (values[i] - mean) * inverse * weight[i] + bias[i];
}

typedef struct {
    std::vector<float> post;
    std::vector<float> comb;
    std::vector<float> normed;
} RefMhc;

static void ref_mhc_pre(RefMhc *cache, const float *residual,
                        const ColiGlm53GpuMhcSiteDesc *site,
                        int rows, int streams, int hidden,
                        float norm_eps, float hc_eps) {
    cache->post.resize((size_t)rows * streams);
    cache->comb.resize((size_t)rows * streams * streams);
    cache->normed.resize((size_t)rows * hidden);
    std::vector<float> collapsed((size_t)rows * hidden);
    std::vector<float> mixes((size_t)(2 + streams) * streams);
    std::vector<float> pre(streams);
    for (int row = 0; row < rows; ++row) {
        const float *input =
            residual + (size_t)row * streams * hidden;
        float square = 0.0f;
        for (int i = 0; i < streams * hidden; ++i)
            square += input[i] * input[i];
        float inverse =
            1.0f / sqrtf(square / (streams * hidden) + norm_eps);
        const int mix_count = (2 + streams) * streams;
        for (int mix = 0; mix < mix_count; ++mix) {
            float sum = 0.0f;
            for (int i = 0; i < streams * hidden; ++i)
                sum += site->fn[(size_t)mix * streams * hidden + i] *
                       input[i];
            mixes[mix] = sum * inverse;
        }
        float *post = cache->post.data() + (size_t)row * streams;
        float *comb =
            cache->comb.data() + (size_t)row * streams * streams;
        for (int stream = 0; stream < streams; ++stream) {
            pre[stream] =
                ref_sigmoid(mixes[stream] * site->scale[0] +
                            site->base[stream]) +
                hc_eps;
            post[stream] =
                2.0f * ref_sigmoid(
                           mixes[streams + stream] * site->scale[1] +
                           site->base[streams + stream]);
        }
        const int matrix_offset = 2 * streams;
        for (int source = 0; source < streams; ++source) {
            float maximum = -INFINITY;
            for (int destination = 0; destination < streams; ++destination) {
                int index =
                    matrix_offset + source * streams + destination;
                comb[source * streams + destination] =
                    mixes[index] * site->scale[2] + site->base[index];
                maximum =
                    fmaxf(maximum, comb[source * streams + destination]);
            }
            float sum = 0.0f;
            for (int destination = 0; destination < streams; ++destination) {
                float *value =
                    &comb[source * streams + destination];
                *value = expf(*value - maximum);
                sum += *value;
            }
            for (int destination = 0; destination < streams; ++destination)
                comb[source * streams + destination] =
                    comb[source * streams + destination] / sum + hc_eps;
        }
        for (int destination = 0; destination < streams; ++destination) {
            float sum = 0.0f;
            for (int source = 0; source < streams; ++source)
                sum += comb[source * streams + destination];
            for (int source = 0; source < streams; ++source)
                comb[source * streams + destination] /= sum + hc_eps;
        }
        for (int iteration = 1; iteration < 20; ++iteration) {
            for (int source = 0; source < streams; ++source) {
                float sum = 0.0f;
                for (int destination = 0; destination < streams;
                     ++destination)
                    sum += comb[source * streams + destination];
                for (int destination = 0; destination < streams;
                     ++destination)
                    comb[source * streams + destination] /= sum + hc_eps;
            }
            for (int destination = 0; destination < streams; ++destination) {
                float sum = 0.0f;
                for (int source = 0; source < streams; ++source)
                    sum += comb[source * streams + destination];
                for (int source = 0; source < streams; ++source)
                    comb[source * streams + destination] /= sum + hc_eps;
            }
        }
        for (int d = 0; d < hidden; ++d) {
            float sum = 0.0f;
            for (int stream = 0; stream < streams; ++stream)
                sum += pre[stream] * input[(size_t)stream * hidden + d];
            collapsed[(size_t)row * hidden + d] = sum;
        }
    }
    ref_rmsnorm(cache->normed.data(), collapsed.data(), site->norm_weight,
                rows, hidden, norm_eps);
}

static void ref_mhc_post(float *output, const float *residual,
                         const float *branch, const RefMhc *cache,
                         int rows, int streams, int hidden) {
    for (int row = 0; row < rows; ++row)
        for (int destination = 0; destination < streams; ++destination)
            for (int d = 0; d < hidden; ++d) {
                float value = 0.0f;
                for (int source = 0; source < streams; ++source)
                    value +=
                        cache->comb[
                            ((size_t)row * streams + source) * streams +
                            destination] *
                        residual[
                            ((size_t)row * streams + source) * hidden + d];
                value += cache->post[(size_t)row * streams + destination] *
                         branch[(size_t)row * hidden + d];
                output[((size_t)row * streams + destination) * hidden + d] =
                    value;
            }
}

typedef struct {
    std::vector<float> matrix;
    std::vector<float> window;
} RefKdaState;

static void ref_kda(float *output, RefKdaState *state,
                    const ColiGlm53GpuKdaLayerDesc *f, const float *input,
                    int rows, int hidden, int heads, int dim, int kernel,
                    float gate_lower_bound, float norm_eps) {
    const int projection = heads * dim;
    std::vector<float> qkv((size_t)3 * projection);
    std::vector<float> low(dim), decay(projection), beta(heads);
    std::vector<float> memory(dim), core(projection), gate(projection);
    std::vector<float> normed(projection);
    for (int token = 0; token < rows; ++token) {
        const float *x = input + (size_t)token * hidden;
        ref_matvec(qkv.data(), f->q_proj, x, projection, hidden);
        ref_matvec(qkv.data() + projection, f->k_proj, x,
                   projection, hidden);
        ref_matvec(qkv.data() + 2 * projection, f->v_proj, x,
                   projection, hidden);
        ref_matvec(low.data(), f->decay_a_proj, x, dim, hidden);
        ref_matvec(decay.data(), f->decay_b_proj, low.data(),
                   projection, dim);
        for (int head = 0; head < heads; ++head)
            for (int d = 0; d < dim; ++d) {
                int index = head * dim + d;
                decay[index] = gate_lower_bound *
                    ref_sigmoid(expf(f->a_log[head]) *
                                (decay[index] + f->dt_bias[index]));
            }
        ref_matvec(beta.data(), f->beta_proj, x, heads, hidden);
        for (float &value : beta) value = ref_sigmoid(value);
        for (int channel = 0; channel < 3 * projection; ++channel) {
            float *history =
                state->window.data() + (size_t)channel * kernel;
            for (int tap = 0; tap + 1 < kernel; ++tap)
                history[tap] = history[tap + 1];
            history[kernel - 1] = qkv[channel];
            float sum = 0.0f;
            for (int tap = 0; tap < kernel; ++tap)
                sum += f->conv[(size_t)channel * kernel + tap] *
                       history[tap];
            qkv[channel] = sum / (1.0f + expf(-sum));
        }
        const float query_scale = 1.0f / sqrtf((float)dim);
        for (int head = 0; head < heads; ++head) {
            float *matrix =
                state->matrix.data() + (size_t)head * dim * dim;
            const float *query = qkv.data() + (size_t)head * dim;
            const float *key =
                qkv.data() + projection + (size_t)head * dim;
            const float *value =
                qkv.data() + 2 * projection + (size_t)head * dim;
            float query_square = 1e-6f, key_square = 1e-6f;
            for (int d = 0; d < dim; ++d) {
                query_square += query[d] * query[d];
                key_square += key[d] * key[d];
            }
            float query_norm = query_scale / sqrtf(query_square);
            float key_norm = 1.0f / sqrtf(key_square);
            std::fill(memory.begin(), memory.end(), 0.0f);
            for (int kd = 0; kd < dim; ++kd) {
                float *matrix_row = matrix + (size_t)kd * dim;
                float alpha = expf(decay[head * dim + kd]);
                float scaled_key = key[kd] * key_norm;
                for (int vd = 0; vd < dim; ++vd) {
                    matrix_row[vd] *= alpha;
                    memory[vd] += scaled_key * matrix_row[vd];
                }
            }
            for (int vd = 0; vd < dim; ++vd)
                core[head * dim + vd] = 0.0f;
            for (int kd = 0; kd < dim; ++kd) {
                float *matrix_row = matrix + (size_t)kd * dim;
                float scaled_key = key[kd] * key_norm;
                float scaled_query = query[kd] * query_norm;
                for (int vd = 0; vd < dim; ++vd) {
                    matrix_row[vd] += scaled_key *
                        (value[vd] - memory[vd]) * beta[head];
                    core[head * dim + vd] +=
                        scaled_query * matrix_row[vd];
                }
            }
        }
        ref_matvec(low.data(), f->gate_a_proj, x, dim, hidden);
        ref_matvec(gate.data(), f->gate_b_proj, low.data(),
                   projection, dim);
        for (int head = 0; head < heads; ++head) {
            float square = 0.0f;
            for (int d = 0; d < dim; ++d)
                square += core[head * dim + d] * core[head * dim + d];
            float inverse = 1.0f / sqrtf(square / dim + norm_eps);
            for (int d = 0; d < dim; ++d)
                normed[head * dim + d] =
                    core[head * dim + d] * inverse * f->o_norm[d] *
                    ref_sigmoid(gate[head * dim + d]);
        }
        ref_matvec(output + (size_t)token * hidden, f->o_proj,
                   normed.data(), hidden, projection);
    }
}

typedef struct {
    int length;
    std::vector<float> latent;
    std::vector<float> keys;
    std::vector<float> gates;
} RefMlaState;

static int ref_mla(float *output, RefMlaState *state,
                   const ColiGlm53GpuMlaLayerDesc *f, const float *input,
                   int rows, int hidden, int heads, int q_lora, int kv_lora,
                   int qk, int value_dim, int index_heads, int index_dim,
                   int pool, int topk, const float *ape,
                   float norm_eps) {
    const int width = topk;
    for (int token = 0; token < rows; ++token) {
        const float *x = input + (size_t)token * hidden;
        std::vector<float> qn(q_lora), query((size_t)heads * qk);
        std::vector<float> iq((size_t)index_heads * index_dim);
        std::vector<float> head_weight(index_heads);
        ref_matvec(qn.data(), f->q_a_proj, x, q_lora, hidden);
        ref_rmsnorm(qn.data(), qn.data(), f->q_a_norm, 1, q_lora, norm_eps);
        ref_matvec(query.data(), f->q_b_proj, qn.data(), heads * qk, q_lora);
        float *latent =
            state->latent.data() + (size_t)state->length * kv_lora;
        ref_matvec(latent, f->kv_a_proj, x, kv_lora, hidden);
        ref_rmsnorm(latent, latent, f->kv_a_norm, 1, kv_lora, norm_eps);
        ref_matvec(iq.data(), f->index_q_proj, qn.data(),
                   index_heads * index_dim, q_lora);
        float *key = state->keys.data() + (size_t)state->length * index_dim;
        ref_matvec(key, f->index_k_proj, x, index_dim, hidden);
        ref_layernorm(key, f->index_key_norm, f->index_key_bias,
                      index_dim, 1e-5f);
        ref_matvec(
            state->gates.data() + (size_t)state->length * index_dim,
            f->index_pool_gate, x, index_dim, hidden);
        ref_matvec(head_weight.data(), f->index_weight_proj, x,
                   index_heads, hidden);
        for (float &weight : head_weight)
            weight /= sqrtf((float)index_heads);
        int seen = ++state->length;
        std::vector<int> selected(width, -1);
        std::vector<unsigned char> valid((size_t)seen, 1);
        if (coli_sparse_index_select_range(
                selected.data(), iq.data(), state->keys.data(),
                state->gates.data(), head_weight.data(), ape, valid.data(),
                seen, index_heads, index_dim, pool, topk, 0,
                seen - 1, seen))
            return 0;
        std::vector<float> expanded_keys((size_t)seen * heads * qk);
        std::vector<float> expanded_values(
            (size_t)seen * heads * value_dim);
        for (int position = 0; position < seen; ++position) {
            const float *cached =
                state->latent.data() + (size_t)position * kv_lora;
            for (int head = 0; head < heads; ++head) {
                for (int q = 0; q < qk; ++q) {
                    float sum = 0.0f;
                    for (int d = 0; d < kv_lora; ++d)
                        sum += f->kv_b_key[
                            ((size_t)head * kv_lora + d) * qk + q] *
                            cached[d];
                    expanded_keys[
                        ((size_t)position * heads + head) * qk + q] = sum;
                }
                for (int v = 0; v < value_dim; ++v) {
                    float sum = 0.0f;
                    for (int d = 0; d < kv_lora; ++d)
                        sum += f->kv_b_value[
                            ((size_t)head * value_dim + v) * kv_lora + d] *
                            cached[d];
                    expanded_values[
                        ((size_t)position * heads + head) * value_dim + v] =
                        sum;
                }
            }
        }
        std::vector<float> context((size_t)heads * value_dim);
        if (coli_sparse_attention_range(
                context.data(), query.data(), expanded_keys.data(),
                expanded_values.data(), selected.data(), seen, width,
                heads, qk, value_dim, seen - 1, seen))
            return 0;
        ref_matvec(output + (size_t)token * hidden, f->o_proj,
                   context.data(), hidden, heads * value_dim);
    }
    return 1;
}

static void ref_dense(float *output, const float *input,
                      const ColiGlm53GpuDenseLayerDesc *f,
                      int rows, int hidden, int intermediate, float limit) {
    std::vector<float> gate(intermediate), up(intermediate);
    for (int row = 0; row < rows; ++row) {
        const float *x = input + (size_t)row * hidden;
        ref_matvec(gate.data(), (const float *)f->gate.data, x,
                   intermediate, hidden);
        ref_matvec(up.data(), (const float *)f->up.data, x,
                   intermediate, hidden);
        for (int i = 0; i < intermediate; ++i) {
            float g = fminf(gate[i], limit);
            float u = fminf(fmaxf(up[i], -limit), limit);
            gate[i] = g / (1.0f + expf(-g)) * u;
        }
        ref_matvec(output + (size_t)row * hidden,
                   (const float *)f->down.data, gate.data(),
                   hidden, intermediate);
    }
}

static float ref_int4(const unsigned char *packed, const float *scales,
                      int row, int column, int columns) {
    size_t row_bytes = (size_t)(columns + 1) / 2;
    unsigned char byte = packed[(size_t)row * row_bytes + column / 2];
    int nibble = column & 1 ? byte >> 4 : byte & 15;
    int value = nibble - 8;
    return value * scales[row];
}

static void ref_expert_f32(float *output, const float *input,
                           const ColiGlm53GpuWeightDesc *gate_weight,
                           const ColiGlm53GpuWeightDesc *up_weight,
                           const ColiGlm53GpuWeightDesc *down_weight,
                           int hidden, int intermediate, float limit) {
    std::vector<float> gate(intermediate), up(intermediate);
    ref_matvec(gate.data(), (const float *)gate_weight->data, input,
               intermediate, hidden);
    ref_matvec(up.data(), (const float *)up_weight->data, input,
               intermediate, hidden);
    for (int i = 0; i < intermediate; ++i) {
        float g = fminf(gate[i], limit);
        float u = fminf(fmaxf(up[i], -limit), limit);
        gate[i] = g / (1.0f + expf(-g)) * u;
    }
    ref_matvec(output, (const float *)down_weight->data, gate.data(),
               hidden, intermediate);
}

static const unsigned char oracle_expert_gate[2] = {0x21, 0x1f};
static const unsigned char oracle_expert_up[2] = {0xe1, 0x12};
static const unsigned char oracle_expert_down[2] = {0x1f, 0x21};
static const float oracle_expert_scales[2] = {0.12f, 0.09f};

static int load_oracle_expert(void *, int pipeline_layer, int expert_id,
                              int *slot, ColiGlm53GpuExpertDesc *expert) {
    if (pipeline_layer != 1 || expert_id != 0 || !slot || !expert) return 0;
    expert->gate =
        {oracle_expert_gate, oracle_expert_scales, 4, 2, 2, 64};
    expert->up =
        {oracle_expert_up, oracle_expert_scales, 4, 2, 2, 64};
    expert->down =
        {oracle_expert_down, oracle_expert_scales, 4, 2, 2, 64};
    *slot = 0;
    return 1;
}

static void ref_expert_int4(float *output, const float *input,
                            const unsigned char *gate_data,
                            const unsigned char *up_data,
                            const unsigned char *down_data,
                            const float *scales,
                            int hidden, int intermediate, float limit) {
    std::vector<float> activated(intermediate);
    for (int out = 0; out < intermediate; ++out) {
        float gate = 0.0f, up = 0.0f;
        for (int in = 0; in < hidden; ++in) {
            gate += input[in] *
                    ref_int4(gate_data, scales, out, in, hidden);
            up += input[in] *
                  ref_int4(up_data, scales, out, in, hidden);
        }
        gate = fminf(gate, limit);
        up = fminf(fmaxf(up, -limit), limit);
        activated[out] = gate / (1.0f + expf(-gate)) * up;
    }
    for (int out = 0; out < hidden; ++out) {
        float sum = 0.0f;
        for (int in = 0; in < intermediate; ++in)
            sum += activated[in] *
                   ref_int4(down_data, scales, out, in, intermediate);
        output[out] = sum;
    }
}

static void ref_moe(float *output, const float *input,
                    const ColiGlm53GpuMoeLayerDesc *f,
                    int rows, int hidden, int intermediate,
                    float routed_scale, float limit) {
    std::vector<float> shared(hidden), routed(hidden);
    for (int row = 0; row < rows; ++row) {
        const float *x = input + (size_t)row * hidden;
        ref_expert_f32(shared.data(), x, &f->shared_gate, &f->shared_up,
                       &f->shared_down, hidden, intermediate, limit);
        ref_expert_int4(routed.data(), x, oracle_expert_gate,
                        oracle_expert_up, oracle_expert_down,
                        oracle_expert_scales, hidden, intermediate, limit);
        float router_score = 0.0f;
        for (int d = 0; d < hidden; ++d)
            router_score += x[d] * f->router[d];
        router_score = ref_sigmoid(router_score);
        float normalized = router_score / (router_score + 1e-20f);
        for (int d = 0; d < hidden; ++d)
            output[(size_t)row * hidden + d] =
                shared[d] + normalized * routed_scale * routed[d];
    }
}

static int test_nontrivial_complete_cpu_oracle(ColiGpuContext *ctx) {
    const int rows = 2, hidden = 2, streams = 4;
    static const float embedding[4] = {0.4f, -0.7f, 0.9f, 0.2f};
    static const float final_norm[2] = {1.1f, 0.85f};
    static const float head[4] = {0.7f, -0.3f, -0.2f, 0.8f};
    std::vector<std::vector<float>> site_fn(
        4, std::vector<float>(192));
    std::vector<std::vector<float>> site_base(
        4, std::vector<float>(24));
    std::vector<std::vector<float>> site_scale(
        4, std::vector<float>(3));
    std::vector<std::vector<float>> site_norm(
        4, std::vector<float>(hidden));
    std::vector<ColiGlm53GpuMhcSiteDesc> sites(4);
    for (int site = 0; site < 4; ++site) {
        for (size_t i = 0; i < site_fn[site].size(); ++i)
            site_fn[site][i] =
                ((i + site) & 1 ? -1.0f : 1.0f) *
                (0.002f + 0.0003f * (float)((i + 3 * site) % 11));
        for (size_t i = 0; i < site_base[site].size(); ++i)
            site_base[site][i] =
                ((i + site) & 1 ? -1.0f : 1.0f) *
                (0.03f + 0.004f * (float)((i + site) % 5));
        site_scale[site] = {
            0.55f + 0.03f * site,
            -0.35f - 0.02f * site,
            0.25f + 0.01f * site
        };
        site_norm[site] = {
            0.9f + 0.04f * site, 1.08f - 0.03f * site
        };
        sites[site] = {
            site_fn[site].data(), site_base[site].data(),
            site_scale[site].data(), site_norm[site].data()
        };
    }

    static const float kq[4] = {0.18f, -0.07f, 0.05f, 0.16f};
    static const float kk[4] = {-0.11f, 0.14f, 0.09f, 0.06f};
    static const float kv[4] = {0.13f, 0.04f, -0.08f, 0.17f};
    static const float ko[4] = {0.22f, -0.05f, 0.07f, 0.19f};
    static const float kga[4] = {0.08f, -0.03f, 0.06f, 0.09f};
    static const float kgb[4] = {0.12f, 0.05f, -0.04f, 0.11f};
    static const float kda[4] = {-0.07f, 0.04f, 0.05f, -0.09f};
    static const float kdb[4] = {0.06f, -0.02f, 0.03f, 0.08f};
    static const float kb[2] = {0.1f, -0.06f};
    static const float kconv[6] =
        {0.7f, -0.5f, 0.6f, 0.4f, -0.3f, 0.8f};
    static const float kdt[2] = {0.03f, -0.02f};
    static const float kalog[1] = {-0.12f};
    static const float konorm[2] = {1.05f, 0.93f};
    ColiGlm53GpuKdaLayerDesc kda_layer = {};
    kda_layer.q_proj = kq;
    kda_layer.k_proj = kk;
    kda_layer.v_proj = kv;
    kda_layer.o_proj = ko;
    kda_layer.gate_a_proj = kga;
    kda_layer.gate_b_proj = kgb;
    kda_layer.decay_a_proj = kda;
    kda_layer.decay_b_proj = kdb;
    kda_layer.beta_proj = kb;
    kda_layer.conv = kconv;
    kda_layer.dt_bias = kdt;
    kda_layer.a_log = kalog;
    kda_layer.o_norm = konorm;

    static const float mqa[4] = {0.16f, -0.04f, 0.07f, 0.13f};
    static const float mqan[2] = {1.04f, 0.96f};
    static const float mqb[4] = {0.11f, 0.03f, -0.05f, 0.14f};
    static const float mkva[4] = {0.12f, -0.08f, 0.04f, 0.15f};
    static const float mkvan[2] = {0.94f, 1.07f};
    static const float mkbk[4] = {0.13f, 0.06f, -0.07f, 0.16f};
    static const float mkbv[4] = {0.17f, -0.03f, 0.05f, 0.12f};
    static const float mo[4] = {0.19f, 0.04f, -0.06f, 0.18f};
    static const float miq[4] = {0.1f, -0.02f, 0.04f, 0.09f};
    static const float mik[4] = {0.08f, 0.03f, -0.05f, 0.11f};
    static const float miw[2] = {0.07f, -0.04f};
    static const float mikn[2] = {1.03f, 0.91f};
    static const float mikb[2] = {0.02f, -0.01f};
    static const float mape[2] = {0.03f, -0.02f};
    static const float mgate[4] = {0.09f, -0.03f, 0.05f, 0.08f};
    ColiGlm53GpuMlaLayerDesc mla_layer = {};
    mla_layer.q_a_proj = mqa;
    mla_layer.q_a_norm = mqan;
    mla_layer.q_b_proj = mqb;
    mla_layer.kv_a_proj = mkva;
    mla_layer.kv_a_norm = mkvan;
    mla_layer.kv_b_key = mkbk;
    mla_layer.kv_b_value = mkbv;
    mla_layer.o_proj = mo;
    mla_layer.index_q_proj = miq;
    mla_layer.index_k_proj = mik;
    mla_layer.index_weight_proj = miw;
    mla_layer.index_key_norm = mikn;
    mla_layer.index_key_bias = mikb;
    mla_layer.index_pool_ape = mape;
    mla_layer.index_pool_gate = mgate;

    static const float dg[4] = {0.2f, -0.08f, 0.06f, 0.17f};
    static const float du[4] = {0.12f, 0.05f, -0.09f, 0.14f};
    static const float dd[4] = {0.18f, -0.04f, 0.07f, 0.15f};
    ColiGlm53GpuDenseLayerDesc dense = {
        f32_weight(dg, 2, 2), f32_weight(du, 2, 2),
        f32_weight(dd, 2, 2)
    };
    static const float router[2] = {0.15f, -0.09f};
    static const float sg[4] = {0.13f, 0.04f, -0.06f, 0.16f};
    static const float su[4] = {0.09f, -0.05f, 0.07f, 0.12f};
    static const float sd[4] = {0.14f, 0.03f, -0.04f, 0.11f};
    ColiGlm53GpuMoeLayerDesc moe = {};
    moe.router = router;
    moe.shared_gate = f32_weight(sg, 2, 2);
    moe.shared_up = f32_weight(su, 2, 2);
    moe.shared_down = f32_weight(sd, 2, 2);
    const ColiGlm53GpuLayerDesc layers[2] = {
        {COLI_GLM53_GPU_ATTN_KDA, 0, COLI_GLM53_GPU_FFN_DENSE, 0, 0, 1},
        {COLI_GLM53_GPU_ATTN_MLA, 0, COLI_GLM53_GPU_FFN_MOE, 0, 2, 3}
    };

    ColiGlm53GpuModelDesc desc = {};
    desc.hidden_size = hidden;
    desc.stream_count = streams;
    desc.vocab_size = 2;
    desc.max_prefill_rows = rows;
    desc.max_context_tokens = rows;
    desc.norm_eps = 1e-6f;
    desc.hc_eps = 1e-6f;
    desc.embedding = embedding;
    desc.final_norm = final_norm;
    desc.lm_head = head;
    desc.sites = sites.data();
    desc.site_count = 4;
    desc.kda_heads = 1;
    desc.kda_head_dim = 2;
    desc.kda_kernel = 1;
    desc.kda_gate_lower_bound = -5.0f;
    desc.kda_layers = &kda_layer;
    desc.kda_layer_count = 1;
    desc.mla_heads = 1;
    desc.mla_q_lora = 2;
    desc.mla_kv_lora = 2;
    desc.mla_qk_nope = 2;
    desc.mla_value_dim = 2;
    desc.mla_index_heads = 1;
    desc.mla_index_dim = 2;
    desc.mla_index_pool = 1;
    desc.mla_index_topk = 1;
    desc.mla_page_tokens = 1;
    desc.mla_layers = &mla_layer;
    desc.mla_layer_count = 1;
    desc.dense_intermediate = 2;
    desc.swiglu_limit = 1.5f;
    desc.dense_layers = &dense;
    desc.dense_layer_count = 1;
    desc.moe_experts = 1;
    desc.moe_topk = 1;
    desc.moe_intermediate = 2;
    desc.moe_cache_slots = 1;
    desc.moe_group_size = 64;
    desc.moe_normalize_topk = 1;
    desc.moe_routed_scale = 0.75f;
    desc.moe_swiglu_limit = 1.5f;
    desc.moe_layers = &moe;
    desc.moe_layer_count = 1;
    desc.layers = layers;
    desc.layer_count = 2;
    desc.expert_loader = load_oracle_expert;

    const int tokens[2] = {0, 1};
    auto cpu_forward = [&](float *last_logits, RefKdaState *kda_out,
                           RefMlaState *mla_out) {
        std::vector<float> current((size_t)rows * streams * hidden);
        for (int row = 0; row < rows; ++row)
            for (int stream = 0; stream < streams; ++stream)
                memcpy(current.data() +
                           ((size_t)row * streams + stream) * hidden,
                       embedding + (size_t)tokens[row] * hidden,
                       (size_t)hidden * sizeof(float));
        std::vector<float> next(current.size());
        std::vector<float> branch((size_t)rows * hidden);
        RefKdaState kda_state = {
            std::vector<float>(4, 0.0f), std::vector<float>(6, 0.0f)
        };
        RefMlaState mla_state = {
            0, std::vector<float>(4, 0.0f),
            std::vector<float>(4, 0.0f), std::vector<float>(4, 0.0f)
        };
        for (int layer = 0; layer < 2; ++layer) {
            RefMhc attention;
            ref_mhc_pre(&attention, current.data(),
                        &sites[layers[layer].attention_site],
                        rows, streams, hidden, 1e-6f, 1e-6f);
            if (layer == 0)
                ref_kda(branch.data(), &kda_state, &kda_layer,
                        attention.normed.data(), rows, hidden, 1, 2, 1,
                        -5.0f, 1e-6f);
            else if (!ref_mla(
                         branch.data(), &mla_state, &mla_layer,
                         attention.normed.data(), rows, hidden, 1, 2, 2,
                         2, 2, 1, 2, 1, 1, mape, 1e-6f))
                return 0;
            ref_mhc_post(next.data(), current.data(), branch.data(),
                         &attention, rows, streams, hidden);
            current.swap(next);

            RefMhc ffn;
            ref_mhc_pre(&ffn, current.data(),
                        &sites[layers[layer].ffn_site],
                        rows, streams, hidden, 1e-6f, 1e-6f);
            if (layer == 0)
                ref_dense(branch.data(), ffn.normed.data(), &dense,
                          rows, hidden, 2, 1.5f);
            else
                ref_moe(branch.data(), ffn.normed.data(), &moe,
                        rows, hidden, 2, 0.75f, 1.5f);
            ref_mhc_post(next.data(), current.data(), branch.data(),
                         &ffn, rows, streams, hidden);
            current.swap(next);
        }
        std::vector<float> collapsed((size_t)rows * hidden);
        std::vector<float> normed((size_t)rows * hidden);
        std::vector<float> logits((size_t)rows * 2);
        for (int row = 0; row < rows; ++row)
            for (int d = 0; d < hidden; ++d) {
                float sum = 0.0f;
                for (int stream = 0; stream < streams; ++stream)
                    sum += current[
                        ((size_t)row * streams + stream) * hidden + d];
                collapsed[(size_t)row * hidden + d] = sum / streams;
            }
        ref_rmsnorm(normed.data(), collapsed.data(), final_norm,
                    rows, hidden, 1e-6f);
        for (int row = 0; row < rows; ++row)
            ref_matvec(logits.data() + (size_t)row * 2, head,
                       normed.data() + (size_t)row * hidden, 2, hidden);
        memcpy(last_logits, logits.data() + 2, 2 * sizeof(float));
        *kda_out = kda_state;
        *mla_out = mla_state;
        return 1;
    };
    float oracle_logits[2];
    RefKdaState kda_state;
    RefMlaState mla_state;
    if (!cpu_forward(oracle_logits, &kda_state, &mla_state)) return 0;

    ColiGlm53GpuModel *model = NULL;
    ColiGlm53GpuSession *session = NULL;
    float got_logits[2];
    float got_matrix[4], got_window[6];
    float got_latent[4], got_keys[4], got_gates[4];
    int ok = coli_glm53_gpu_model_create(&model, ctx, &desc) &&
             coli_glm53_gpu_session_create(&session, model, rows) &&
             coli_glm53_gpu_forward(session, tokens, rows, got_logits) &&
             coli_glm53_gpu_session_kda_state_download(
                 session, 0, got_matrix, 4, got_window, 6) &&
             coli_glm53_gpu_session_mla_state_download(
                 session, 0, got_latent, 4, got_keys, 4, got_gates, 4) &&
             compare_oracle(got_logits, oracle_logits, 2) &&
             compare_oracle(got_matrix, kda_state.matrix.data(), 4) &&
             compare_oracle(got_window, kda_state.window.data(), 6) &&
             compare_oracle(got_latent, mla_state.latent.data(), 4) &&
             compare_oracle(got_keys, mla_state.keys.data(), 4) &&
             compare_oracle(got_gates, mla_state.gates.data(), 4);
    coli_glm53_gpu_session_destroy(session);
    coli_glm53_gpu_model_destroy(model);
    if (!ok) fprintf(stderr, "FAIL: nontrivial complete CPU oracle\n");
    return ok;
}

static int test_45_layer_pipeline(ColiGpuContext *ctx) {
    static const float zero[192] = {0};
    static const float one[2] = {1.0f, 1.0f};
    static const float embedding[4] = {
        1.0f, -0.5f, 0.25f, 0.75f
    };
    static const float head[4] = {
        1.0f, 0.0f, 0.0f, 1.0f
    };
    static const float scale[3] = {0};
    const int layer_count = 45;
    const int kda_count = 34;
    const int mla_count = 11;
    const int dense_count = 3;
    const int moe_count = 42;
    std::vector<ColiGlm53GpuMhcSiteDesc> sites(2 * layer_count);
    for (ColiGlm53GpuMhcSiteDesc &site : sites)
        site = {zero, zero, scale, one};

    ColiGlm53GpuKdaLayerDesc kda_template = {};
    kda_template.q_proj = zero;
    kda_template.k_proj = zero;
    kda_template.v_proj = zero;
    kda_template.o_proj = zero;
    kda_template.gate_a_proj = zero;
    kda_template.gate_b_proj = zero;
    kda_template.decay_a_proj = zero;
    kda_template.decay_b_proj = zero;
    kda_template.beta_proj = zero;
    kda_template.conv = zero;
    kda_template.dt_bias = zero;
    kda_template.a_log = zero;
    kda_template.o_norm = one;
    std::vector<ColiGlm53GpuKdaLayerDesc> kda(
        kda_count, kda_template);

    ColiGlm53GpuMlaLayerDesc mla_template = {};
    mla_template.q_a_proj = zero;
    mla_template.q_a_norm = one;
    mla_template.q_b_proj = zero;
    mla_template.kv_a_proj = zero;
    mla_template.kv_a_norm = one;
    mla_template.kv_b_key = zero;
    mla_template.kv_b_value = zero;
    mla_template.o_proj = zero;
    mla_template.index_q_proj = zero;
    mla_template.index_k_proj = zero;
    mla_template.index_weight_proj = zero;
    mla_template.index_key_norm = one;
    mla_template.index_key_bias = zero;
    mla_template.index_pool_ape = zero;
    mla_template.index_pool_gate = zero;
    std::vector<ColiGlm53GpuMlaLayerDesc> mla(
        mla_count, mla_template);

    ColiGlm53GpuDenseLayerDesc dense_template = {
        f32_weight(zero, 2, 2),
        f32_weight(zero, 2, 2),
        f32_weight(zero, 2, 2)
    };
    std::vector<ColiGlm53GpuDenseLayerDesc> dense(
        dense_count, dense_template);
    ColiGlm53GpuMoeLayerDesc moe_template = {};
    moe_template.router = zero;
    moe_template.shared_gate = f32_weight(zero, 2, 2);
    moe_template.shared_up = f32_weight(zero, 2, 2);
    moe_template.shared_down = f32_weight(zero, 2, 2);
    std::vector<ColiGlm53GpuMoeLayerDesc> moe(moe_count, moe_template);
    std::vector<ColiGlm53GpuLayerDesc> layers(layer_count);
    int kda_at = 0, mla_at = 0, dense_at = 0, moe_at = 0;
    for (int i = 0; i < layer_count; ++i) {
        const int full = i % 4 == 3;
        layers[i].attention_kind =
            full ? COLI_GLM53_GPU_ATTN_MLA : COLI_GLM53_GPU_ATTN_KDA;
        layers[i].attention_index = full ? mla_at++ : kda_at++;
        layers[i].ffn_kind =
            i < 3 ? COLI_GLM53_GPU_FFN_DENSE : COLI_GLM53_GPU_FFN_MOE;
        layers[i].ffn_index = i < 3 ? dense_at++ : moe_at++;
        layers[i].attention_site = 2 * i;
        layers[i].ffn_site = 2 * i + 1;
    }
    if (kda_at != 34 || mla_at != 11 ||
        dense_at != 3 || moe_at != 42)
        return 0;

    ColiGlm53GpuModelDesc desc = {};
    desc.hidden_size = 2;
    desc.stream_count = 4;
    desc.vocab_size = 2;
    desc.max_prefill_rows = 2;
    desc.max_context_tokens = 4;
    desc.norm_eps = 1e-6f;
    desc.hc_eps = 1e-6f;
    desc.embedding = embedding;
    desc.final_norm = one;
    desc.lm_head = head;
    desc.sites = sites.data();
    desc.site_count = (int)sites.size();
    desc.kda_heads = 1;
    desc.kda_head_dim = 2;
    desc.kda_kernel = 1;
    desc.kda_gate_lower_bound = -20.0f;
    desc.kda_layers = kda.data();
    desc.kda_layer_count = (int)kda.size();
    desc.mla_heads = 1;
    desc.mla_q_lora = 2;
    desc.mla_kv_lora = 2;
    desc.mla_qk_nope = 2;
    desc.mla_value_dim = 2;
    desc.mla_index_heads = 1;
    desc.mla_index_dim = 2;
    desc.mla_index_pool = 1;
    desc.mla_index_topk = 1;
    desc.mla_page_tokens = 1;
    desc.mla_layers = mla.data();
    desc.mla_layer_count = (int)mla.size();
    desc.dense_intermediate = 2;
    desc.swiglu_limit = 1.0f;
    desc.dense_layers = dense.data();
    desc.dense_layer_count = (int)dense.size();
    desc.moe_experts = 1;
    desc.moe_topk = 1;
    desc.moe_intermediate = 2;
    desc.moe_cache_slots = 1;
    desc.moe_group_size = 64;
    desc.moe_normalize_topk = 1;
    desc.moe_routed_scale = 1.0f;
    desc.moe_swiglu_limit = 1.0f;
    desc.moe_layers = moe.data();
    desc.moe_layer_count = (int)moe.size();
    desc.layers = layers.data();
    desc.layer_count = (int)layers.size();
    desc.expert_loader = load_zero_expert;

    ColiGlm53GpuModelDesc undersized = desc;
    undersized.moe_experts = 2;
    undersized.moe_topk = 2;
    undersized.moe_cache_slots = 1;
    ColiGlm53GpuModel *invalid = NULL;
    if (coli_glm53_gpu_model_create(&invalid, ctx, &undersized)) {
        coli_glm53_gpu_model_destroy(invalid);
        return 0;
    }

    ColiGlm53GpuModel *model = NULL;
    ColiGlm53GpuSession *session = NULL;
    if (!coli_glm53_gpu_model_create(&model, ctx, &desc) ||
        !coli_glm53_gpu_session_create(&session, model, 4))
        return 0;
    ColiGpuTelemetry before = {}, after = {};
    coli_gpu_context_telemetry(ctx, &before);
    const int prefill[2] = {0, 1};
    const int decode = 0;
    float first_prefill[2], first_decode[2];
    float second_prefill[2], second_decode[2];
    expert_load_calls = 0;
    int ok = coli_glm53_gpu_forward(
                 session, prefill, 2, first_prefill) &&
             coli_glm53_gpu_forward(
                 session, &decode, 1, first_decode);
    float matrix_first[4], window_first[6];
    float latent_first[6], keys_first[6], gates_first[6];
    ok = ok && coli_glm53_gpu_session_kda_state_download(
                   session, 0, matrix_first, 4, window_first, 6) &&
         coli_glm53_gpu_session_mla_state_download(
                   session, 0, latent_first, 6,
                   keys_first, 6, gates_first, 6) &&
         coli_glm53_gpu_session_reset(session) &&
         coli_glm53_gpu_forward(
                   session, prefill, 2, second_prefill) &&
         coli_glm53_gpu_forward(
                   session, &decode, 1, second_decode);
    float matrix_second[4], window_second[6];
    float latent_second[6], keys_second[6], gates_second[6];
    ok = ok && coli_glm53_gpu_session_kda_state_download(
                   session, 0, matrix_second, 4, window_second, 6) &&
         coli_glm53_gpu_session_mla_state_download(
                   session, 0, latent_second, 6,
                   keys_second, 6, gates_second, 6);
    coli_gpu_context_telemetry(ctx, &after);
    float inverse = 1.0f / sqrtf((0.25f * 0.25f + 0.75f * 0.75f) /
                                 2.0f + 1e-6f);
    const float oracle[2] = {0.25f * inverse, 0.75f * inverse};
    ok = ok && compare_oracle(first_prefill, oracle, 2) &&
         !memcmp(first_prefill, second_prefill, sizeof(first_prefill)) &&
         !memcmp(first_decode, second_decode, sizeof(first_decode)) &&
         !memcmp(matrix_first, matrix_second, sizeof(matrix_first)) &&
         !memcmp(window_first, window_second, sizeof(window_first)) &&
         !memcmp(latent_first, latent_second, sizeof(latent_first)) &&
         !memcmp(keys_first, keys_second, sizeof(keys_first)) &&
         !memcmp(gates_first, gates_second, sizeof(gates_first)) &&
         expert_load_calls == 6 * moe_count &&
         after.device_allocations == before.device_allocations &&
         after.host_activation_h2d_copies ==
             before.host_activation_h2d_copies &&
         after.host_activation_d2h_copies ==
             before.host_activation_d2h_copies + 4 &&
         after.expert_upload_bytes > before.expert_upload_bytes;
    coli_glm53_gpu_session_destroy(session);
    coli_glm53_gpu_model_destroy(model);
    if (!ok) fprintf(stderr, "FAIL: 45-layer pipeline/oracle/determinism\n");
    return ok;
}

static int test_probe_fault_matrix(void) {
    const ColiGpuFaultPoint faults[] = {
        COLI_GPU_FAULT_BASE_ALLOCATION,
        COLI_GPU_FAULT_MODEL_ALLOCATION,
        COLI_GPU_FAULT_SESSION_ALLOCATION,
        COLI_GPU_FAULT_TENSOR_UPLOAD,
        COLI_GPU_FAULT_EXPERT_PUBLICATION,
        COLI_GPU_FAULT_KERNEL_LAUNCH,
        COLI_GPU_FAULT_DEVICE_STATUS,
        COLI_GPU_FAULT_STREAM_SYNC
    };
    for (size_t i = 0; i < sizeof(faults) / sizeof(faults[0]); ++i) {
        ColiGpuContext *ctx = NULL;
        if (!coli_gpu_context_create(&ctx, 0) ||
            !coli_gpu_context_inject_fault(ctx, faults[i], 0) ||
            coli_glm53_gpu_pipeline_probe(ctx) ||
            coli_gpu_context_probe(ctx, COLI_GPU_CAP_PIPELINE)) {
            fprintf(stderr, "FAIL: probe fault %d advertised pipeline\n",
                    (int)faults[i]);
            coli_gpu_context_destroy(ctx);
            return 0;
        }
        coli_gpu_context_destroy(ctx);
    }
    return 1;
}

static ColiGpuFaultPoint startup_fault;
static int startup_creates;
static int startup_probes;
static int startup_destroys;

static int startup_context_create(ColiGpuContext **out, int device) {
    startup_creates++;
    if (!coli_gpu_context_create(out, device)) return 0;
    if (!coli_gpu_context_inject_fault(*out, startup_fault, 0)) {
        coli_gpu_context_destroy(*out);
        *out = NULL;
        return 0;
    }
    return 1;
}

static int startup_context_probe(ColiGpuContext *ctx, uint64_t caps) {
    startup_probes++;
    return coli_glm53_gpu_pipeline_probe(ctx) &&
           coli_gpu_context_probe(ctx, caps);
}

static void startup_context_destroy(ColiGpuContext *ctx) {
    startup_destroys++;
    coli_gpu_context_destroy(ctx);
}

static int test_integrated_startup_fault_matrix(void) {
    const ColiGpuFaultPoint faults[] = {
        COLI_GPU_FAULT_BASE_ALLOCATION,
        COLI_GPU_FAULT_MODEL_ALLOCATION,
        COLI_GPU_FAULT_SESSION_ALLOCATION,
        COLI_GPU_FAULT_TENSOR_UPLOAD,
        COLI_GPU_FAULT_EXPERT_PUBLICATION,
        COLI_GPU_FAULT_KERNEL_LAUNCH,
        COLI_GPU_FAULT_DEVICE_STATUS,
        COLI_GPU_FAULT_STREAM_SYNC
    };
    ColiGlm53GpuOps ops = {
        startup_context_create, startup_context_probe,
        startup_context_destroy
    };
    for (size_t fault = 0; fault < sizeof(faults) / sizeof(faults[0]);
         ++fault) {
        for (int forced = 0; forced < 2; ++forced) {
            ColiGlm53GpuBackend backend;
            char err[128] = {};
            startup_fault = faults[fault];
            startup_creates = startup_probes = startup_destroys = 0;
            coli_glm53_gpu_backend_init(&backend);
            int selected = coli_glm53_gpu_backend_select(
                &backend,
                forced ? COLI_GLM53_GPU_MODE_GPU :
                         COLI_GLM53_GPU_MODE_AUTO,
                ops, 0, COLI_GLM53_GPU_REQUIRED_CAPS,
                err, sizeof(err));
            int ok =
                (forced ? !selected : selected) &&
                coli_glm53_gpu_backend_selected(&backend) ==
                    (forced ? COLI_GLM53_BACKEND_UNSELECTED :
                              COLI_GLM53_BACKEND_CPU) &&
                !coli_glm53_gpu_backend_context(&backend) &&
                backend.active_requests == 0 &&
                startup_creates == 1 && startup_probes == 1 &&
                startup_destroys == 1;
            coli_glm53_gpu_backend_destroy(&backend, ops);
            if (!ok) {
                fprintf(stderr,
                        "FAIL: integrated %s startup fault %d "
                        "(select=%d create=%d probe=%d destroy=%d err=%s)\n",
                        forced ? "gpu" : "auto", (int)faults[fault],
                        selected, startup_creates, startup_probes,
                        startup_destroys, err);
                return 0;
            }
        }
    }
    return 1;
}

int main(void) {
    ColiGpuContext *ctx = NULL;
    if (!coli_gpu_context_create(&ctx, 0)) return 1;
    if (!coli_glm53_gpu_pipeline_probe(ctx) ||
        !coli_gpu_context_probe(ctx, COLI_GPU_CAP_PIPELINE) ||
        !test_offset_binary_grouped_int4_projection(ctx) ||
        !test_full_forward_and_output_boundary(ctx) ||
        !test_max_context_fails_without_truncation(ctx) ||
        !test_nontrivial_complete_cpu_oracle(ctx) ||
        !test_45_layer_pipeline(ctx)) {
        coli_gpu_context_destroy(ctx);
        return 1;
    }
    coli_gpu_context_destroy(ctx);

    const ColiGpuFaultPoint runtime_faults[] = {
        COLI_GPU_FAULT_TENSOR_UPLOAD,
        COLI_GPU_FAULT_KERNEL_LAUNCH,
        COLI_GPU_FAULT_DEVICE_STATUS,
        COLI_GPU_FAULT_STREAM_SYNC
    };
    for (size_t i = 0;
         i < sizeof(runtime_faults) / sizeof(runtime_faults[0]); ++i) {
        ctx = NULL;
        if (!coli_gpu_context_create(&ctx, 0) ||
            !coli_glm53_gpu_pipeline_probe(ctx) ||
            !test_fault_preserves_output_and_poison_context(
                ctx, runtime_faults[i])) {
            coli_gpu_context_destroy(ctx);
            return 1;
        }
        coli_gpu_context_destroy(ctx);
    }
    ctx = NULL;
    if (!coli_gpu_context_create(&ctx, 0) ||
        !coli_glm53_gpu_pipeline_probe(ctx) ||
        !test_arithmetic_nonfinite_preserves_output_and_poison_context(ctx)) {
        coli_gpu_context_destroy(ctx);
        return 1;
    }
    coli_gpu_context_destroy(ctx);
    if (!test_probe_fault_matrix() ||
        !test_integrated_startup_fault_matrix())
        return 1;
    puts("glm53 full pipeline: PASS");
    return 0;
}
