#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
/* Amalgamated deepseek_v4.c — GLM-style source; compile with -DCOLI_V4_UNIT_* per object */
/* Umbrella API: deepseek_v4.h (included by units) */

#ifdef COLI_V4_UNIT_ST
/* Shared st.h adapter and V4 tensor materialization helpers. */
#include "deepseek_v4_internal.h"

#include <dirent.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int set_error(char *error, size_t size, const char *format, ...) {
    if (error && size) {
        va_list arguments;
        va_start(arguments, format);
        vsnprintf(error, size, format, arguments);
        va_end(arguments);
    }
    return -1;
}

int coli_st_index_open(ColiSafetensorsIndex **out, const char *directory,
                       char *error, size_t error_size) {
    if (!out || !directory)
        return set_error(error, error_size, "invalid safetensors index arguments");
    *out = NULL;
    DIR *probe = opendir(directory);
    if (!probe)
        return set_error(error, error_size, "cannot open model directory: %s", directory);
    closedir(probe);
    ColiSafetensorsIndex *index = calloc(1, sizeof(*index));
    if (!index)
        return set_error(error, error_size, "out of memory opening: %s", directory);
    st_init(index, directory);
    if (!index->nfd || !index->n) {
        coli_st_index_close(index);
        return set_error(error, error_size, "no safetensors tensors in: %s", directory);
    }
    *out = index;
    return 0;
}

void coli_st_index_close(ColiSafetensorsIndex *index) {
    if (!index) return;
    st_mirror_reset(index);
    for (int i = 0; i < index->n; i++) free(index->t[i].name);
    for (int i = 0; i < index->nfd; i++) {
        if (index->fds[i] >= 0) close(index->fds[i]);
        if (index->dfds[i] >= 0) close(index->dfds[i]);
        free(index->paths[i]);
    }
    for (int i = 0; i < index->fmt_n; i++) {
        free(index->fmt_name[i]);
        free(index->fmt_val[i]);
    }
    free(index->fmt_name);
    free(index->fmt_val);
    free(index->hidx);
    free(index->t);
    free(index);
}

size_t coli_st_tensor_count(const ColiSafetensorsIndex *index) {
    return index ? (size_t)index->n : 0;
}

size_t coli_st_shard_count(const ColiSafetensorsIndex *index) {
    return index ? (size_t)index->nfd : 0;
}

const char *coli_st_shard_path(const ColiSafetensorsIndex *index, int shard) {
    return index && shard >= 0 && shard < index->nfd ? index->paths[shard] : NULL;
}

const ColiSafetensorsTensor *coli_st_find(const ColiSafetensorsIndex *index,
                                         const char *name) {
    return index && name ? st_find((ColiSafetensorsIndex *)index, name) : NULL;
}

int coli_st_tensor_shard(const ColiSafetensorsIndex *index,
                         const ColiSafetensorsTensor *tensor) {
    return index && tensor
        ? st_fidx((ColiSafetensorsIndex *)index, tensor->fd) : -1;
}

int coli_st_read_at(const ColiSafetensorsIndex *index, int shard,
                    uint64_t offset, size_t length, void *destination) {
    if (!index || !destination || shard < 0 || shard >= index->nfd ||
        index->sizes[shard] < 0 || offset > (uint64_t)index->sizes[shard] ||
        length > (uint64_t)index->sizes[shard] - offset)
        return -1;
    unsigned char *output = destination;
    size_t done = 0;
    while (done < length) {
        ssize_t count = pread(index->fds[shard], output + done, length - done,
                              (off_t)(offset + done));
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) return -1;
        done += (size_t)count;
    }
    return 0;
}

int coli_st_streaming_direct_available(const ColiSafetensorsIndex *index,
                                       int shard) {
    if (!index || shard < 0 || shard >= index->nfd) return 0;
    const char *setting = getenv("COLI_V4_DIRECT");
    if (setting && atoi(setting) == 0) return 0;
    return index->dfds[shard] >= 0;
}

int coli_st_read_at_streaming(const ColiSafetensorsIndex *index, int shard,
                              uint64_t offset, size_t length,
                              void *destination) {
    if (!index || !destination || shard < 0 || shard >= index->nfd ||
        index->sizes[shard] < 0 || offset > (uint64_t)index->sizes[shard] ||
        length > (uint64_t)index->sizes[shard] - offset)
        return -1;
    if (!length) return 0;
    if (!coli_st_streaming_direct_available(index, shard))
        return coli_st_read_at(index, shard, offset, length, destination);

    const uint64_t alignment = 4096;
    uint64_t base = offset & ~(alignment - 1);
    uint64_t pad = offset - base;
    if ((uint64_t)length > UINT64_MAX - pad) return -1;
    uint64_t needed = pad + (uint64_t)length;
    if (needed > UINT64_MAX - (alignment - 1) ||
        needed + alignment - 1 > SIZE_MAX) return -1;
    size_t allocation_bytes = (size_t)((needed + alignment - 1) &
                                       ~(alignment - 1));
    /* PORTABILITY: on Windows compat.h maps posix_memalign onto
     * _aligned_malloc, whose blocks MUST be released with compat_aligned_free
     * -- passing one to free() corrupts the CRT heap. compat.h says so at its
     * own definition. Every release below therefore goes through
     * compat_aligned_free, which is plain free() on POSIX. Getting this wrong
     * killed tests/test_deepseek_v4.exe before it printed its first line. */
    unsigned char *bounce = NULL;
    if (posix_memalign((void **)&bounce, (size_t)alignment,
                       allocation_bytes) != 0)
        return coli_st_read_at(index, shard, offset, length, destination);

    uint64_t file_bytes = (uint64_t)index->sizes[shard];
    uint64_t available = file_bytes - base;
    uint64_t direct_bytes = allocation_bytes;
    if (direct_bytes > available)
        direct_bytes = available & ~(alignment - 1);
    size_t done = 0;
    int failed = 0;
    while ((uint64_t)done < direct_bytes) {
        ssize_t count = pread(index->dfds[shard], bounce + done,
                              (size_t)(direct_bytes - done),
                              (off_t)(base + done));
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) { failed = 1; break; }
        done += (size_t)count;
    }
    if (failed) {
        compat_aligned_free(bounce);
        return coli_st_read_at(index, shard, offset, length, destination);
    }
    while ((uint64_t)done < needed) {
        ssize_t count = pread(index->fds[shard], bounce + done,
                              (size_t)(needed - done),
                              (off_t)(base + done));
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) { compat_aligned_free(bounce); return -1; }
        done += (size_t)count;
    }
    memcpy(destination, bounce + pad, length);
    compat_aligned_free(bounce);
    return 0;
}

int coli_st_read_tensor(const ColiSafetensorsIndex *index,
                        const ColiSafetensorsTensor *tensor, void *destination) {
    int shard = coli_st_tensor_shard(index, tensor);
    return tensor && tensor->off >= 0
        ? coli_st_read_at(index, shard, (uint64_t)tensor->off,
                          (size_t)tensor->nbytes, destination) : -1;
}

int coli_st_prefetch_at(const ColiSafetensorsIndex *index, int shard,
                        uint64_t offset, size_t length) {
    if (!index || shard < 0 || shard >= index->nfd ||
        index->sizes[shard] < 0 || offset > (uint64_t)index->sizes[shard] ||
        length > (uint64_t)index->sizes[shard] - offset)
        return -1;
    return posix_fadvise(index->fds[shard], (off_t)offset, (off_t)length,
                         POSIX_FADV_WILLNEED);
}

const char *coli_st_dtype_name(ColiSafetensorsDType dtype) {
    return st_dtype_name(dtype);
}

void coli_owned_tensor_free(ColiOwnedTensor *tensor) {
    if (!tensor) return;
    free(tensor->data_allocation);
    free(tensor->scale_allocation);
    memset(tensor, 0, sizeof(*tensor));
}

int coli_tensor_load_fp8(ColiOwnedTensor *output,
                         const ColiSafetensorsIndex *index,
                         const char *prefix, char *error, size_t error_size) {
    if (!output || !index || !prefix)
        return set_error(error, error_size, "invalid FP8 tensor arguments");
    memset(output, 0, sizeof(*output));
    size_t length = strlen(prefix) + sizeof(".weight");
    char *name = malloc(length);
    if (!name) return set_error(error, error_size, "out of memory building tensor name");
    snprintf(name, length, "%s.weight", prefix);
    const ColiSafetensorsTensor *weight = coli_st_find(index, name);
    snprintf(name, length, "%s.scale", prefix);
    const ColiSafetensorsTensor *scale = coli_st_find(index, name);
    free(name);
    if (!weight || !scale || weight->dtype != COLI_ST_F8_E4M3 ||
        scale->dtype != COLI_ST_F8_E8M0 || weight->rank != 2 || scale->rank != 2 ||
        scale->shape[0] != (weight->shape[0] + 127) / 128 ||
        scale->shape[1] != (weight->shape[1] + 127) / 128)
        return set_error(error, error_size, "invalid native FP8 tensor: %s", prefix);
    output->data_allocation = malloc((size_t)weight->nbytes);
    output->scale_allocation = malloc((size_t)scale->numel * sizeof(float));
    if (!output->data_allocation || !output->scale_allocation) {
        coli_owned_tensor_free(output);
        return set_error(error, error_size, "out of memory loading FP8 tensor: %s", prefix);
    }
    if (coli_st_read_tensor(index, weight, output->data_allocation) != 0 ||
        st_read_scale_f32((ColiSafetensorsIndex *)index, scale->name,
                          output->scale_allocation, scale->numel, 0) != scale->numel) {
        coli_owned_tensor_free(output);
        return set_error(error, error_size, "cannot read FP8 tensor: %s", prefix);
    }
    output->view = (ColiTensorView){
        COLI_TENSOR_FP8_E4M3_BLOCK, COLI_SCALE_F32,
        output->data_allocation, output->scale_allocation,
        (size_t)weight->nbytes, (size_t)scale->numel * sizeof(float),
        weight->shape[0], weight->shape[1], 128, 128
    };
    return 0;
}

void coli_float_tensor_free(ColiFloatTensor *tensor) {
    if (!tensor) return;
    free(tensor->data);
    memset(tensor, 0, sizeof(*tensor));
}

int coli_tensor_load_f32(ColiFloatTensor *output,
                         const ColiSafetensorsIndex *index,
                         const char *name, char *error, size_t error_size) {
    if (!output || !index || !name)
        return set_error(error, error_size, "invalid float tensor arguments");
    memset(output, 0, sizeof(*output));
    const ColiSafetensorsTensor *tensor = coli_st_find(index, name);
    if (!tensor || (tensor->dtype != COLI_ST_F32 && tensor->dtype != COLI_ST_BF16))
        return set_error(error, error_size, "missing BF16/F32 tensor: %s", name);
    void *raw = malloc((size_t)tensor->nbytes);
    output->data = malloc((size_t)tensor->numel * sizeof(*output->data));
    if (!raw || !output->data) {
        free(raw);
        coli_float_tensor_free(output);
        return set_error(error, error_size, "out of memory loading tensor: %s", name);
    }
    if (coli_st_read_tensor(index, tensor, raw) != 0) {
        free(raw);
        coli_float_tensor_free(output);
        return set_error(error, error_size, "cannot read tensor: %s", name);
    }
    if (tensor->dtype == COLI_ST_F32)
        memcpy(output->data, raw, (size_t)tensor->nbytes);
    else {
        const uint16_t *values = raw;
        for (uint64_t i = 0; i < (uint64_t)tensor->numel; i++)
            output->data[i] = coli_bf16_decode(values[i]);
    }
    free(raw);
    output->count = (uint64_t)tensor->numel;
    output->rank = tensor->rank;
    memcpy(output->shape, tensor->shape, sizeof(output->shape));
    return 0;
}
#endif /* COLI_V4_UNIT_ST */


#ifdef COLI_V4_UNIT_LAYER_RESIDENT
/* ######## deepseek_v4_layer_resident.c ######## */
#define coli_v4_layer_load coli_v4_layer_resident_reference_load
#define coli_v4_layer_free coli_v4_layer_resident_reference_free
/* ---- begin include deepseek_v4_layer.c ---- */
#include "deepseek_v4_internal.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int set_error(char *error, size_t size, const char *format, ...) {
    if (error && size) {
        va_list arguments;
        va_start(arguments, format);
        vsnprintf(error, size, format, arguments);
        va_end(arguments);
    }
    return -1;
}

static int add_spec(ColiDeepSeekV4LayerPlan *plan, ColiSafetensorsDType dtype,
                    int rank, const int64_t *shape, const char *suffix,
                    char *error, size_t error_size) {
    if (plan->tensor_count >= COLI_V4_MAX_LAYER_TENSORS)
        return set_error(error, error_size, "too many tensors in layer %d", plan->layer);
    ColiDeepSeekV4TensorSpec *spec = &plan->tensors[plan->tensor_count++];
    int written = snprintf(spec->name, sizeof(spec->name), "layers.%d.%s",
                           plan->layer, suffix);
    if (written < 0 || (size_t)written >= sizeof(spec->name))
        return set_error(error, error_size, "tensor name is too long: %s", suffix);
    spec->dtype = dtype;
    spec->rank = rank;
    memcpy(spec->shape, shape, (size_t)rank * sizeof(*shape));
    return 0;
}

static int add_1d(ColiDeepSeekV4LayerPlan *plan, ColiSafetensorsDType dtype,
                  int64_t d0, const char *name, char *error, size_t size) {
    int64_t shape[] = {d0};
    return add_spec(plan, dtype, 1, shape, name, error, size);
}

static int add_2d(ColiDeepSeekV4LayerPlan *plan, ColiSafetensorsDType dtype,
                  int64_t d0, int64_t d1, const char *name,
                  char *error, size_t size) {
    int64_t shape[] = {d0, d1};
    return add_spec(plan, dtype, 2, shape, name, error, size);
}

static int add_fp8(ColiDeepSeekV4LayerPlan *plan, int64_t rows, int64_t columns,
                   const char *prefix, char *error, size_t size) {
    char name[128];
    snprintf(name, sizeof(name), "%s.weight", prefix);
    if (add_2d(plan, COLI_ST_F8_E4M3, rows, columns, name, error, size) != 0) return -1;
    snprintf(name, sizeof(name), "%s.scale", prefix);
    return add_2d(plan, COLI_ST_F8_E8M0, (rows + 127) / 128,
                  (columns + 127) / 128, name, error, size);
}

#define ADD(call) do { if ((call) != 0) return -1; } while (0)

int coli_v4_layer_plan(ColiDeepSeekV4LayerPlan *plan,
                       const ColiDeepSeekV4Config *config, int layer,
                       char *error, size_t error_size) {
    if (!plan || !config || layer < 0 || layer >= config->num_hidden_layers ||
        layer >= config->compress_ratio_count)
        return set_error(error, error_size, "invalid DeepSeek-V4 layer plan arguments");
    memset(plan, 0, sizeof(*plan));
    plan->layer = layer;
    plan->compression_ratio = config->compress_ratios[layer];
    plan->uses_hash_router = layer < config->num_hash_layers;
    plan->has_compressor = plan->compression_ratio != 0;
    plan->has_indexer = plan->compression_ratio == 4;

    const int64_t hidden = config->hidden_size;
    const int64_t heads = config->num_attention_heads;
    const int64_t head_dim = config->head_dim;
    const int64_t q_rank = config->q_lora_rank;
    if (config->o_groups < 1 || heads % config->o_groups != 0)
        return set_error(error, error_size, "unsupported grouped-output attention dimensions");
    const int64_t o_group_width =
        (heads / config->o_groups) * head_dim;
    const int64_t o_width = (int64_t)config->o_groups * config->o_lora_rank;
    const int64_t experts = config->n_routed_experts;
    const int64_t moe = config->moe_intermediate_size;
    const int64_t hc = config->hc_mult;
    const int64_t hc_params = (2 + hc) * hc;

    ADD(add_1d(plan, COLI_ST_F32, heads, "attn.attn_sink", error, error_size));
    ADD(add_1d(plan, COLI_ST_BF16, head_dim, "attn.kv_norm.weight", error, error_size));
    ADD(add_1d(plan, COLI_ST_BF16, q_rank, "attn.q_norm.weight", error, error_size));
    ADD(add_fp8(plan, head_dim, hidden, "attn.wkv", error, error_size));
    ADD(add_fp8(plan, o_width, o_group_width, "attn.wo_a", error, error_size));
    ADD(add_fp8(plan, hidden, o_width, "attn.wo_b", error, error_size));
    ADD(add_fp8(plan, q_rank, hidden, "attn.wq_a", error, error_size));
    ADD(add_fp8(plan, heads * head_dim, q_rank, "attn.wq_b", error, error_size));
    ADD(add_1d(plan, COLI_ST_BF16, hidden, "attn_norm.weight", error, error_size));

    if (plan->has_compressor) {
        int64_t ratio = plan->compression_ratio;
        int64_t coff = ratio == 4 ? 2 : 1;
        ADD(add_2d(plan, COLI_ST_F32, ratio, coff * head_dim,
                   "attn.compressor.ape", error, error_size));
        ADD(add_1d(plan, COLI_ST_BF16, head_dim,
                   "attn.compressor.norm.weight", error, error_size));
        ADD(add_2d(plan, COLI_ST_BF16, coff * head_dim, hidden,
                   "attn.compressor.wgate.weight", error, error_size));
        ADD(add_2d(plan, COLI_ST_BF16, coff * head_dim, hidden,
                   "attn.compressor.wkv.weight", error, error_size));
    }
    if (plan->has_indexer) {
        int64_t ih = config->index_head_dim;
        int64_t in = config->index_n_heads;
        ADD(add_2d(plan, COLI_ST_F32, 4, 2 * ih,
                   "attn.indexer.compressor.ape", error, error_size));
        ADD(add_1d(plan, COLI_ST_BF16, ih,
                   "attn.indexer.compressor.norm.weight", error, error_size));
        ADD(add_2d(plan, COLI_ST_BF16, 2 * ih, hidden,
                   "attn.indexer.compressor.wgate.weight", error, error_size));
        ADD(add_2d(plan, COLI_ST_BF16, 2 * ih, hidden,
                   "attn.indexer.compressor.wkv.weight", error, error_size));
        ADD(add_2d(plan, COLI_ST_BF16, in, hidden,
                   "attn.indexer.weights_proj.weight", error, error_size));
        ADD(add_fp8(plan, in * ih, q_rank, "attn.indexer.wq_b", error, error_size));
    }

    ADD(add_2d(plan, COLI_ST_BF16, experts, hidden,
               "ffn.gate.weight", error, error_size));
    if (plan->uses_hash_router)
        ADD(add_2d(plan, COLI_ST_I64, config->vocab_size,
                   config->num_experts_per_tok, "ffn.gate.tid2eid", error, error_size));
    else
        ADD(add_1d(plan, COLI_ST_F32, experts, "ffn.gate.bias", error, error_size));
    ADD(add_fp8(plan, moe, hidden, "ffn.shared_experts.w1", error, error_size));
    ADD(add_fp8(plan, hidden, moe, "ffn.shared_experts.w2", error, error_size));
    ADD(add_fp8(plan, moe, hidden, "ffn.shared_experts.w3", error, error_size));
    ADD(add_1d(plan, COLI_ST_BF16, hidden, "ffn_norm.weight", error, error_size));

    ADD(add_1d(plan, COLI_ST_F32, hc_params, "hc_attn_base", error, error_size));
    ADD(add_2d(plan, COLI_ST_F32, hc_params, hc * hidden,
               "hc_attn_fn", error, error_size));
    ADD(add_1d(plan, COLI_ST_F32, 3, "hc_attn_scale", error, error_size));
    ADD(add_1d(plan, COLI_ST_F32, hc_params, "hc_ffn_base", error, error_size));
    ADD(add_2d(plan, COLI_ST_F32, hc_params, hc * hidden,
               "hc_ffn_fn", error, error_size));
    ADD(add_1d(plan, COLI_ST_F32, 3, "hc_ffn_scale", error, error_size));
    return 0;
}

int coli_v4_layer_validate(const ColiDeepSeekV4LayerPlan *plan,
                           const ColiSafetensorsIndex *index,
                           ColiDeepSeekV4LayerStats *stats,
                           char *error, size_t error_size) {
    if (!plan || !index)
        return set_error(error, error_size, "invalid DeepSeek-V4 layer validation arguments");
    ColiDeepSeekV4LayerStats local = {0};
    for (size_t i = 0; i < plan->tensor_count; i++) {
        const ColiDeepSeekV4TensorSpec *spec = &plan->tensors[i];
        const ColiSafetensorsTensor *tensor = coli_st_find(index, spec->name);
        if (!tensor)
            return set_error(error, error_size, "missing tensor: %s", spec->name);
        if (tensor->dtype != spec->dtype || tensor->rank != spec->rank)
            return set_error(error, error_size, "dtype/rank mismatch: %s", spec->name);
        for (int dimension = 0; dimension < spec->rank; dimension++)
            if (tensor->shape[dimension] != spec->shape[dimension])
                return set_error(error, error_size, "shape mismatch: %s", spec->name);
        local.tensor_count++;
        uint64_t resident_bytes = tensor->dtype == COLI_ST_F8_E8M0
            ? (uint64_t)tensor->numel * sizeof(float) : (uint64_t)tensor->nbytes;
        local.total_bytes += resident_bytes;
        switch (tensor->dtype) {
            case COLI_ST_BF16: local.bf16_bytes += tensor->nbytes; break;
            case COLI_ST_F32: local.f32_bytes += tensor->nbytes; break;
            case COLI_ST_F8_E4M3: local.fp8_weight_bytes += tensor->nbytes; break;
            case COLI_ST_F8_E8M0: local.fp8_scale_bytes += resident_bytes; break;
            case COLI_ST_I64: local.i64_bytes += tensor->nbytes; break;
            default: break;
        }
    }
    if (stats) *stats = local;
    return 0;
}

void coli_v4_layer_free(ColiV4Engine *engine,
                        ColiDeepSeekV4LayerWeights *weights) {
    (void)engine;
    if (!weights) return;
    for (size_t i = 0; i < weights->plan.tensor_count; i++) free(weights->data[i]);
    memset(weights, 0, sizeof(*weights));
}

/* AVX2 consumes eight output rows at once.  Interleave those rows by column
 * once when a dense layer becomes resident, instead of gathering eight
 * distant cache lines in every matvec. */
static int v4_fp8_pack_rows8_inplace(unsigned char *data,
                                     int64_t rows, int64_t columns) {
#ifndef __AVX2__
    (void)data; (void)rows; (void)columns;
    return 0;
#else
    if (!data || rows < 8 || rows % 8 || columns < 128 || columns % 128)
        return 0;
    size_t tile_bytes = (size_t)8 * (size_t)columns;
    unsigned char *scratch = malloc(tile_bytes);
    if (!scratch) return -1;
    for (int64_t tile = 0; tile < rows / 8; tile++) {
        unsigned char *target = data + (size_t)tile * tile_bytes;
        memcpy(scratch, target, tile_bytes);
        for (int64_t column = 0; column < columns; column++)
            for (int lane = 0; lane < 8; lane++)
                target[(size_t)column * 8 + lane] =
                    scratch[(size_t)lane * columns + column];
    }
    free(scratch);
    return 1;
#endif
}

int coli_v4_layer_load(ColiV4Engine *engine,
                       ColiDeepSeekV4LayerWeights *weights,
                       const ColiDeepSeekV4Config *config,
                       const ColiSafetensorsIndex *index, int layer,
                       char *error, size_t error_size) {
    (void)engine;
    if (!weights) return set_error(error, error_size, "missing layer weights output");
    memset(weights, 0, sizeof(*weights));
    if (coli_v4_layer_plan(&weights->plan, config, layer, error, error_size) != 0 ||
        coli_v4_layer_validate(&weights->plan, index, &weights->stats,
                               error, error_size) != 0)
        return -1;
    for (size_t i = 0; i < weights->plan.tensor_count; i++) {
        const ColiDeepSeekV4TensorSpec *spec = &weights->plan.tensors[i];
        const ColiSafetensorsTensor *tensor = coli_st_find(index, spec->name);
        size_t resident_bytes = tensor->dtype == COLI_ST_F8_E8M0
            ? (size_t)tensor->numel * sizeof(float) : (size_t)tensor->nbytes;
        weights->data[i] = malloc(resident_bytes);
        if (!weights->data[i]) {
            coli_v4_layer_free(NULL, weights);
            return set_error(error, error_size, "out of memory loading: %s", spec->name);
        }
        int read_failed = tensor->dtype == COLI_ST_F8_E8M0
            ? st_read_scale_f32((ColiSafetensorsIndex *)index, spec->name,
                                weights->data[i], tensor->numel, 0) != tensor->numel
            : coli_st_read_tensor(index, tensor, weights->data[i]) != 0;
        if (read_failed) {
            coli_v4_layer_free(NULL, weights);
            return set_error(error, error_size, "cannot read tensor: %s", spec->name);
        }
        if (spec->dtype == COLI_ST_F8_E4M3 && spec->rank == 2) {
            int packed = v4_fp8_pack_rows8_inplace(
                weights->data[i], spec->shape[0], spec->shape[1]);
            if (packed < 0) {
                coli_v4_layer_free(NULL, weights);
                return set_error(error, error_size,
                                 "out of memory packing FP8 rows8: %s",
                                 spec->name);
            }
            weights->plan.tensors[i].packed_rows8 = packed > 0;
        }
    }
    return 0;
}

const void *coli_v4_layer_data(const ColiDeepSeekV4LayerWeights *weights,
                               const char *name,
                               const ColiDeepSeekV4TensorSpec **spec) {
    if (spec) *spec = NULL;
    if (!weights || !name) return NULL;
    for (size_t i = 0; i < weights->plan.tensor_count; i++) {
        if (strcmp(weights->plan.tensors[i].name, name) == 0) {
            if (spec) *spec = &weights->plan.tensors[i];
            return weights->data[i];
        }
    }
    return NULL;
}
/* ---- end include deepseek_v4_layer.c ---- */

#undef coli_v4_layer_free
#undef coli_v4_layer_load

#include "deepseek_v4_internal.h"

enum { COLI_V4_RESIDENT_MAX_LAYERS_V2 = COLI_V4_RESIDENT_MAX_LAYERS };

static int resident_enabled_v2(ColiV4Engine *engine) {
    return engine && engine->runtime.dense_resident;
}

int coli_v4_layer_load(ColiV4Engine *engine,
                       ColiDeepSeekV4LayerWeights *weights,
                       const ColiDeepSeekV4Config *config,
                       const ColiSafetensorsIndex *index, int layer,
                       char *error, size_t error_size) {
    const ColiDeepSeekV4Config *effective_config =
        engine ? &engine->config : config;
    if (!weights || !effective_config || !index || layer < 0 ||
        layer >= effective_config->num_hidden_layers ||
        layer >= COLI_V4_RESIDENT_MAX_LAYERS_V2) return -1;
    if (!resident_enabled_v2(engine))
        return coli_v4_layer_resident_reference_load(
            NULL, weights, effective_config, index, layer, error, error_size);
    if (engine->dense_resident.index && engine->dense_resident.index != index) {
        if (error && error_size)
            snprintf(error, error_size,
                     "resident V4 dense cache cannot switch model instances");
        return -1;
    }
    engine->dense_resident.index = index;
    if (!engine->dense_resident.ready[layer]) {
        if (coli_v4_layer_resident_reference_load(
                NULL, &engine->dense_resident.layers[layer], effective_config, index,
                layer, error, error_size)) return -1;
        engine->dense_resident.ready[layer] = 1;
        engine->dense_resident.total_bytes +=
            engine->dense_resident.layers[layer].stats.total_bytes;
        if (layer == effective_config->num_hidden_layers - 1)
            fprintf(stderr, "v4_dense_resident layers=%d bytes=%.3fGiB\n",
                    effective_config->num_hidden_layers,
                    engine->dense_resident.total_bytes / 1073741824.0);
    }
    *weights = engine->dense_resident.layers[layer]; return 0;
}

void coli_v4_layer_free(ColiV4Engine *engine,
                        ColiDeepSeekV4LayerWeights *weights) {
    if (!weights) return;
    int layer = weights->plan.layer;
    if (engine && layer >= 0 && layer < COLI_V4_RESIDENT_MAX_LAYERS_V2 &&
        engine->dense_resident.ready[layer] &&
        weights->plan.tensor_count ==
            engine->dense_resident.layers[layer].plan.tensor_count &&
        weights->data[0] == engine->dense_resident.layers[layer].data[0]) {
        memset(weights, 0, sizeof(*weights)); return;
    }
    coli_v4_layer_resident_reference_free(NULL, weights);
}
#endif /* COLI_V4_UNIT_LAYER_RESIDENT */

#ifdef COLI_V4_UNIT_RESOURCE_PLAN
/* ######## deepseek_v4_resource_plan.c ######## */
#include "deepseek_v4_internal.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

#define MIB UINT64_C(1048576)

static int plan_error(char *error, size_t size, const char *format, ...) {
    if (error && size) {
        va_list arguments;
        va_start(arguments, format);
        vsnprintf(error, size, format, arguments);
        va_end(arguments);
    }
    return -1;
}

static int add_u64(uint64_t a, uint64_t b, uint64_t *output) {
    if (UINT64_MAX - a < b) return -1;
    *output = a + b;
    return 0;
}

static int multiply_u64(uint64_t a, uint64_t b, uint64_t *output) {
    if (a && b > UINT64_MAX / a) return -1;
    *output = a * b;
    return 0;
}

uint64_t coli_v4_os_available_memory(void) {
#ifdef _WIN32
    MEMORYSTATUSEX status;
    memset(&status, 0, sizeof(status));
    status.dwLength = sizeof(status);
    return GlobalMemoryStatusEx(&status) ? (uint64_t)status.ullAvailPhys : 0;
#else
    FILE *stream = fopen("/proc/meminfo", "r");
    if (stream) {
        char line[256];
        unsigned long long kib = 0;
        while (fgets(line, sizeof(line), stream))
            if (sscanf(line, "MemAvailable: %llu kB", &kib) == 1) break;
        fclose(stream);
        if (kib) return (uint64_t)kib * 1024;
    }
    long pages = sysconf(_SC_AVPHYS_PAGES), page_size = sysconf(_SC_PAGESIZE);
    if (pages <= 0 || page_size <= 0) return 0;
    return (uint64_t)pages * (uint64_t)page_size;
#endif
}

int coli_v4_resource_plan_compute(
    ColiDeepSeekV4ResourcePlan *plan,
    const ColiDeepSeekV4ResourceInputs *inputs,
    char *error, size_t error_size) {
    if (!plan || !inputs || !inputs->available_bytes ||
        !inputs->maximum_layer_bytes || !inputs->expert_record_bytes ||
        inputs->sparse_layers < 1 || inputs->routed_topk < 1 ||
        inputs->experts_per_layer < inputs->routed_topk)
        return plan_error(error, error_size, "invalid V4 resource-plan inputs");
    memset(plan, 0, sizeof(*plan));
    plan->os_available_bytes = inputs->available_bytes;
    uint64_t available = inputs->available_bytes;
    int explicit_process_limit = inputs->user_limit_bytes &&
        inputs->user_limit_bytes < available;
    if (explicit_process_limit)
        available = inputs->user_limit_bytes;
    plan->planner_available_bytes = available;

    /* A process limit leaves all RAM outside the limit to the OS. Automatic
     * mode starts from MemAvailable and therefore reserves that share here. */
    uint64_t system = explicit_process_limit ? 0 : available / 8;
    if (!explicit_process_limit && system < 512 * MIB) system = 512 * MIB;
    if (system > 4096 * MIB) system = 4096 * MIB;
    plan->system_reserve_bytes = system;

    uint64_t layers_twice;
    if (multiply_u64(inputs->maximum_layer_bytes, 2, &layers_twice) ||
        add_u64(layers_twice, inputs->runtime_other_bytes,
                &plan->runtime_reserve_bytes))
        return plan_error(error, error_size, "V4 runtime reserve overflow");

    uint64_t per_slot;
    if (multiply_u64((uint64_t)inputs->sparse_layers,
                     inputs->expert_record_bytes, &per_slot) ||
        multiply_u64(per_slot, (uint64_t)inputs->routed_topk,
                     &plan->minimum_expert_bytes))
        return plan_error(error, error_size, "V4 expert-cache size overflow");

    uint64_t fixed;
    if (add_u64(system, plan->runtime_reserve_bytes, &fixed) || fixed >= available)
        return plan_error(error, error_size,
                          "available RAM cannot hold V4 runtime reserves");
    uint64_t usable = available - fixed;
    if (usable < plan->minimum_expert_bytes)
        return plan_error(error, error_size,
            "V4 minimum expert cache needs %.2f GiB but only %.2f GiB remains",
            plan->minimum_expert_bytes / 1073741824.0,
            usable / 1073741824.0);

    uint64_t slots = usable / per_slot;
    if (slots > (uint64_t)inputs->experts_per_layer)
        slots = (uint64_t)inputs->experts_per_layer;
    if (slots < (uint64_t)inputs->routed_topk)
        slots = (uint64_t)inputs->routed_topk;
    plan->slots_per_layer = (int)slots;
    if (multiply_u64(per_slot, slots, &plan->expert_cache_bytes) ||
        add_u64(fixed, plan->expert_cache_bytes, &plan->projected_bytes))
        return plan_error(error, error_size, "V4 projected memory overflow");
    if (plan->projected_bytes > available)
        return plan_error(error, error_size, "V4 plan exceeds available RAM");
    return 0;
}

static int resident_tiers_fit(uint64_t available, uint64_t fixed,
                              uint64_t dense, uint64_t minimum_experts) {
    uint64_t total;
    return !add_u64(fixed, dense, &total) &&
           !add_u64(total, minimum_experts, &total) && total <= available;
}

int coli_v4_resident_tier_plan(
    ColiDeepSeekV4ResidentTierPlan *plan,
    const ColiDeepSeekV4ResidentTierInputs *inputs,
    char *error, size_t error_size) {
    if (!plan || !inputs || !inputs->available_bytes ||
        !inputs->dense_bytes || !inputs->minimum_expert_bytes)
        return plan_error(error, error_size,
                          "invalid V4 resident-tier inputs");
    memset(plan, 0, sizeof(*plan));

    if (!resident_tiers_fit(inputs->available_bytes, inputs->fixed_bytes,
                            0, inputs->minimum_expert_bytes))
        return plan_error(error, error_size,
                          "resident V4 tiers leave too little target cache");

    if (resident_tiers_fit(inputs->available_bytes, inputs->fixed_bytes,
                           inputs->dense_bytes,
                           inputs->minimum_expert_bytes)) {
        plan->dense_resident = 1;
        plan->dense_bytes = inputs->dense_bytes;
    }
    return 0;
}
#endif /* COLI_V4_UNIT_RESOURCE_PLAN */

#ifdef COLI_V4_UNIT_HEAD_CACHE
/* ######## deepseek_v4_head_cache.c ######## */
#include "deepseek_v4_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>


static int find_head(const char *model_dir, ColiSafetensorsIndex **index,
                     const ColiSafetensorsTensor **head,
                     char *error, size_t error_size) {
    *index = NULL; *head = NULL;
    if (coli_st_index_open(index, model_dir, error, error_size)) return -1;
    *head = coli_st_find(*index, "head.weight");
    if (!*head || (*head)->dtype != COLI_ST_BF16 || (*head)->rank != 2) {
        snprintf(error, error_size, "missing or invalid BF16 head.weight");
        coli_st_index_close(*index); *index = NULL; return -1;
    }
    return 0;
}

int coli_v4_head_cache_probe(const char *model_dir, uint64_t *bytes,
                             char *error, size_t error_size) {
    ColiSafetensorsIndex *index;
    const ColiSafetensorsTensor *head;
    if (!bytes || find_head(model_dir, &index, &head, error, error_size)) return -1;
    *bytes = head->nbytes;
    coli_st_index_close(index); return 0;
}

int coli_v4_head_cache_load(ColiV4Engine *engine, const char *model_dir,
                            char *error, size_t error_size) {
    if (!engine) {
        snprintf(error, error_size, "head cache requires a V4 engine");
        return -1;
    }
    ColiSafetensorsIndex *index;
    const ColiSafetensorsTensor *head;
    if (find_head(model_dir, &index, &head, error, error_size)) return -1;
    unsigned char *data = malloc((size_t)head->nbytes);
    int shard = coli_st_tensor_shard(index, head);
    if (!data || coli_st_read_at(index, shard, (uint64_t)head->off,
                                 (size_t)head->nbytes, data)) {
        free(data); coli_st_index_close(index);
        snprintf(error, error_size, "cannot load resident BF16 head.weight");
        return -1;
    }
    free(engine->head_cache.data);
    engine->head_cache.data = data;
    engine->head_cache.bytes = head->nbytes;
    engine->head_cache.offset = (uint64_t)head->off;
    engine->head_cache.shard = shard;
    coli_st_index_close(index); return 0;
}

uint64_t coli_v4_head_cache_bytes(const ColiV4Engine *engine) {
    return engine ? engine->head_cache.bytes : 0;
}

const void *coli_v4_head_cache_data(const ColiV4Engine *engine,
                                    int shard, uint64_t offset, size_t length) {
    if (!engine || !engine->head_cache.data || shard != engine->head_cache.shard ||
        offset < engine->head_cache.offset ||
        offset - engine->head_cache.offset > engine->head_cache.bytes ||
        length > engine->head_cache.bytes - (offset - engine->head_cache.offset))
        return NULL;
    return engine->head_cache.data +
           (size_t)(offset - engine->head_cache.offset);
}

int coli_st_read_at_engine(ColiV4Engine *engine,
                           const ColiSafetensorsIndex *index, int shard,
                           uint64_t offset, size_t length, void *destination) {
    if (engine && engine->head_cache.data && destination &&
        shard == engine->head_cache.shard &&
        offset >= engine->head_cache.offset &&
        offset - engine->head_cache.offset <= engine->head_cache.bytes &&
        length <= engine->head_cache.bytes -
                      (offset - engine->head_cache.offset)) {
        memcpy(destination,
               engine->head_cache.data +
                   (size_t)(offset - engine->head_cache.offset),
               length);
        return 0;
    }
    return coli_st_read_at(index, shard, offset, length, destination);
}
#endif /* COLI_V4_UNIT_HEAD_CACHE */

#ifdef COLI_V4_UNIT_EXPERT_STORE_AUTO
/* ######## deepseek_v4_expert_store_auto.c ######## */
/* ---- begin inlined deepseek_v4_expert_store_auto_v5.c ---- */
#include "deepseek_v4_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>


#define MIB UINT64_C(1048576)
#define GIB UINT64_C(1073741824)

static uint64_t expert_record_bytes(const ColiSafetensorsIndex *index) {
    static const char *parts[] = {
        "layers.0.ffn.experts.0.w1.weight", "layers.0.ffn.experts.0.w1.scale",
        "layers.0.ffn.experts.0.w2.weight", "layers.0.ffn.experts.0.w2.scale",
        "layers.0.ffn.experts.0.w3.weight", "layers.0.ffn.experts.0.w3.scale",
    };
    uint64_t total = 0;
    for (size_t i = 0; i < sizeof(parts) / sizeof(parts[0]); i++) {
        const ColiSafetensorsTensor *tensor = coli_st_find(index, parts[i]);
        if (!tensor || tensor->nbytes < 0) return 0;
        uint64_t bytes = (uint64_t)tensor->nbytes;
        if (UINT64_MAX - total < bytes) return 0;
        total += bytes;
    }
    return total;
}

static uint64_t context_bytes(const ColiDeepSeekV4Config *config, int context) {
    uint64_t total = (uint64_t)config->num_hidden_layers *
        config->sliding_window * config->head_dim * sizeof(float);
    for (int layer = 0; layer < config->num_hidden_layers; layer++) {
        int ratio = config->compress_ratios[layer];
        if (!ratio) continue;
        uint64_t compressed = ((uint64_t)context + (uint64_t)ratio - 1) /
                              (uint64_t)ratio;
        total += compressed * config->head_dim * sizeof(float);
        if (ratio == 4)
            total += compressed * config->index_head_dim * sizeof(float);
    }
    return total;
}

static int build_runtime_plan(ColiV4Engine *engine,
                              const ColiDeepSeekV4ExpertStoreOptions *options,
                              ColiDeepSeekV4ResourcePlan *plan,
                              char *error, size_t error_size) {
    ColiDeepSeekV4Config config;
    ColiSafetensorsIndex *index = NULL;
    if (!engine) {
        snprintf(error, error_size, "V4 runtime requires an engine");
        return -1;
    }
    if (coli_v4_config_load(&config, options->model_dir, error, error_size) ||
        coli_st_index_open(&index, options->model_dir, error, error_size))
        return -1;
    uint64_t maximum_layer = 0;
    for (int layer = 0; layer < config.num_hidden_layers; layer++) {
        ColiDeepSeekV4LayerPlan layer_plan;
        ColiDeepSeekV4LayerStats stats;
        if (coli_v4_layer_plan(&layer_plan, &config, layer,
                               error, error_size) ||
            coli_v4_layer_validate(&layer_plan, index, &stats,
                                   error, error_size)) {
            coli_st_index_close(index); return -1;
        }
        if (stats.total_bytes > maximum_layer) maximum_layer = stats.total_bytes;
    }
    uint64_t record = expert_record_bytes(index);
    coli_st_index_close(index);
    if (!record) {
        snprintf(error, error_size, "cannot determine V4 expert record size");
        return -1;
    }
    ColiDeepSeekV4RuntimeOptions *runtime = &engine->runtime;
    int context = runtime->context_tokens;
    if (context > config.max_position_embeddings)
        context = config.max_position_embeddings;
    uint64_t hidden = (uint64_t)64 * config.hc_mult * config.hidden_size *
                      sizeof(float) * 2;
    uint64_t scratch = 512 * MIB;
    uint64_t runtime_other = context_bytes(&config, context) + hidden + scratch;
    if (UINT64_MAX - runtime_other < runtime->dspark_reserve_bytes) {
        snprintf(error, error_size, "V4 DSpark reserve overflow");
        return -1;
    }
    runtime_other += runtime->dspark_reserve_bytes;
    uint64_t available = coli_v4_os_available_memory();
    if (!available) {
        snprintf(error, error_size, "cannot determine OS available memory");
        return -1;
    }
    ColiDeepSeekV4ResourceInputs inputs = {
        available, runtime->memory_limit_bytes, maximum_layer,
        runtime_other, record, config.num_hidden_layers,
        config.num_experts_per_tok, config.n_routed_experts,
    };
    return coli_v4_resource_plan_compute(plan, &inputs, error, error_size);
}

static int v5_dense_inventory(const char *model_dir,
                              uint64_t *bytes,
                              char *error, size_t error_size) {
    ColiDeepSeekV4Config config; ColiSafetensorsIndex *index = NULL;
    if (coli_v4_config_load(&config, model_dir, error, error_size) ||
        coli_st_index_open(&index, model_dir, error, error_size)) return -1;
    uint64_t total = 0;
    for (int layer = 0; layer < config.num_hidden_layers; layer++) {
        ColiDeepSeekV4LayerPlan layer_plan; ColiDeepSeekV4LayerStats stats;
        if (coli_v4_layer_plan(&layer_plan, &config, layer,
                               error, error_size) ||
            coli_v4_layer_validate(&layer_plan, index, &stats,
                                   error, error_size)) {
            coli_st_index_close(index); return -1;
        }
        total += stats.total_bytes;
    }
    coli_st_index_close(index); *bytes = total; return 0;
}

int coli_v4_expert_store_open_planned(
    ColiV4Engine *engine,
    const ColiDeepSeekV4ExpertStoreOptions *options, ColiExpertStore **output,
    char *error, size_t error_size) {
    if (!options || !engine) return -1;
    ColiDeepSeekV4ResourcePlan plan;
    ColiDeepSeekV4RuntimeOptions *runtime = &engine->runtime;
    if (build_runtime_plan(engine, options, &plan, error, error_size)) return -1;
    uint64_t per_slot = plan.expert_cache_bytes /
                        (uint64_t)plan.slots_per_layer;
    uint64_t head_bytes = 0, dense_bytes = 0;
    if (coli_v4_head_cache_probe(options->model_dir, &head_bytes,
                                 error, error_size) ||
        v5_dense_inventory(options->model_dir, &dense_bytes,
                           error, error_size)) return -1;

    uint64_t fixed = plan.system_reserve_bytes + plan.runtime_reserve_bytes;
    ColiDeepSeekV4ResidentTierPlan tiers;
    ColiDeepSeekV4ResidentTierInputs tier_inputs = {
        plan.planner_available_bytes, fixed, dense_bytes,
        plan.minimum_expert_bytes,
    };
    if (coli_v4_resident_tier_plan(&tiers, &tier_inputs,
                                   error, error_size)) return -1;
    dense_bytes = tiers.dense_bytes;
    runtime->dense_resident = tiers.dense_resident;
    if (dense_bytes > plan.planner_available_bytes - fixed) {
        snprintf(error, error_size, "resident V4 tiers exceed available RAM");
        return -1;
    }
    uint64_t safe_payload = plan.planner_available_bytes - fixed -
                            dense_bytes;
    int requested_head = -1;
    int resident_head = safe_payload >= plan.minimum_expert_bytes +
                                      head_bytes + 256 * MIB;
    if (requested_head == 0) resident_head = 0;
    if (requested_head == 1 && !resident_head) {
        snprintf(error, error_size, "resident BF16 head does not fit RAM plan");
        return -1;
    }
    uint64_t cache_limit = safe_payload - (resident_head ? head_bytes : 0);

    if (cache_limit < plan.minimum_expert_bytes) {
        snprintf(error, error_size, "resident tiers leave too little target cache");
        return -1;
    }
    int slots = (int)(cache_limit / per_slot);
    if (slots > plan.slots_per_layer) slots = plan.slots_per_layer;
    if (slots < options->experts_per_layer && slots < 6) slots = 6;
    plan.expert_cache_bytes = (uint64_t)slots * per_slot;
    runtime->target_expert_cache_bytes = plan.expert_cache_bytes;
    plan.projected_bytes = fixed + dense_bytes +
        plan.expert_cache_bytes + (resident_head ? head_bytes : 0);
    if (resident_head && coli_v4_head_cache_load(
            engine, options->model_dir, error, error_size)) return -1;
    fprintf(stderr,
        "ram_tiers available=%.2fGiB dense=%s(%.2fGiB) "
        "target_slots=%d target_cache=%.2fGiB head=%s projected=%.2fGiB\n",
        plan.planner_available_bytes / (double)GIB,
        tiers.dense_resident ? "resident" : "streamed",
        dense_bytes / (double)GIB,
        slots,
        plan.expert_cache_bytes / (double)GIB,
        resident_head ? "resident-bf16" : "streamed-bf16",
        plan.projected_bytes / (double)GIB);
    ColiDeepSeekV4ExpertStoreOptions automatic = *options;
    automatic.cache_bytes = plan.expert_cache_bytes;
    automatic.pin_slots_per_layer = runtime->pin_slots_per_layer;
    automatic.repin_interval = runtime->repin_interval;
    return coli_deepseek_v4_expert_store_open(
        &automatic, output, error, error_size);
}
/* ---- end inlined deepseek_v4_expert_store_auto_v5.c ---- */
#endif /* COLI_V4_UNIT_EXPERT_STORE_AUTO */

#ifdef COLI_V4_UNIT_MATH
/* ######## deepseek_v4_math.c ######## */
#include "deepseek_v4_internal.h"

#include <limits.h>
#include <math.h>
#include <stdlib.h>

static float sigmoidf_stable(float value) {
    if (value >= 0.0f) {
        float decay = expf(-value);
        return 1.0f / (1.0f + decay);
    }
    float growth = expf(value);
    return growth / (1.0f + growth);
}

int coli_v4_hc_split_sinkhorn(float *pre, float *post, float *comb,
                              const float *mixes, const float scale[3],
                              const float *base, int hc, int iterations,
                              float eps) {
    if (!pre || !post || !comb || !mixes || !scale || !base ||
        hc < 1 || iterations < 1 || eps < 0.0f)
        return -1;
    for (int index = 0; index < hc; index++) {
        pre[index] = sigmoidf_stable(
            mixes[index] * scale[0] + base[index]) + eps;
        post[index] = 2.0f * sigmoidf_stable(
            mixes[hc + index] * scale[1] + base[hc + index]);
    }
    int matrix_offset = 2 * hc;
    for (int row = 0; row < hc; row++) {
        float maximum = -INFINITY;
        for (int column = 0; column < hc; column++) {
            int index = matrix_offset + row * hc + column;
            float value = mixes[index] * scale[2] + base[index];
            comb[row * hc + column] = value;
            if (value > maximum) maximum = value;
        }
        float sum = 0.0f;
        for (int column = 0; column < hc; column++) {
            float value = expf(comb[row * hc + column] - maximum);
            comb[row * hc + column] = value;
            sum += value;
        }
        for (int column = 0; column < hc; column++)
            comb[row * hc + column] = comb[row * hc + column] / sum + eps;
    }
    float *sums = malloc((size_t)hc * sizeof(*sums));
    if (!sums) return -1;
    for (int column = 0; column < hc; column++) {
        float sum = 0.0f;
        for (int row = 0; row < hc; row++)
            sum += comb[row * hc + column];
        sums[column] = sum;
    }
    for (int row = 0; row < hc; row++)
        for (int column = 0; column < hc; column++)
            comb[row * hc + column] /= sums[column] + eps;

    for (int iteration = 1; iteration < iterations; iteration++) {
        for (int row = 0; row < hc; row++) {
            float sum = 0.0f;
            for (int column = 0; column < hc; column++)
                sum += comb[row * hc + column];
            sums[row] = sum;
        }
        for (int row = 0; row < hc; row++)
            for (int column = 0; column < hc; column++)
                comb[row * hc + column] /= sums[row] + eps;
        for (int column = 0; column < hc; column++) {
            float sum = 0.0f;
            for (int row = 0; row < hc; row++)
                sum += comb[row * hc + column];
            sums[column] = sum;
        }
        for (int row = 0; row < hc; row++)
            for (int column = 0; column < hc; column++)
                comb[row * hc + column] /= sums[column] + eps;
    }
    free(sums);
    return 0;
}

int coli_v4_hc_pre(float *output, float *post, float *comb,
                   const float *input, const float *hc_fn,
                   const float scale[3], const float *base,
                   int hc, int dimension, int iterations,
                   float norm_eps, float hc_eps) {
    if (!output || !post || !comb || !input || !hc_fn || !scale || !base ||
        hc < 1 || dimension < 1 || norm_eps < 0.0f)
        return -1;
    int flattened = hc * dimension;
    int mix_count = (2 + hc) * hc;
    float mean_square = 0.0f;
    for (int index = 0; index < flattened; index++)
        mean_square += input[index] * input[index];
    float inverse_rms = 1.0f / sqrtf(mean_square / flattened + norm_eps);
    float *mixes = malloc((size_t)mix_count * sizeof(*mixes));
    float *pre = malloc((size_t)hc * sizeof(*pre));
    if (!mixes || !pre) {
        free(mixes);
        free(pre);
        return -1;
    }
    for (int row = 0; row < mix_count; row++) {
        float sum = 0.0f;
        for (int column = 0; column < flattened; column++)
            sum += hc_fn[(size_t)row * flattened + column] * input[column];
        mixes[row] = sum * inverse_rms;
    }
    if (coli_v4_hc_split_sinkhorn(pre, post, comb, mixes, scale, base,
                                  hc, iterations, hc_eps) != 0) {
        free(pre);
        free(mixes);
        return -1;
    }
    for (int column = 0; column < dimension; column++) {
        float sum = 0.0f;
        for (int copy = 0; copy < hc; copy++)
            sum += pre[copy] * input[copy * dimension + column];
        output[column] = sum;
    }
    free(pre);
    free(mixes);
    return 0;
}

int coli_v4_hc_post(float *output, const float *branch,
                    const float *residual, const float *post,
                    const float *comb, int hc, int dimension) {
    if (!output || !branch || !residual || !post || !comb ||
        hc < 1 || dimension < 1)
        return -1;
    for (int destination = 0; destination < hc; destination++) {
        for (int column = 0; column < dimension; column++) {
            float value = 0.0f;
            for (int source = 0; source < hc; source++)
                value += comb[source * hc + destination] *
                         residual[source * dimension + column];
            value += post[destination] * branch[column];
            output[destination * dimension + column] = value;
        }
    }
    return 0;
}

int coli_v4_rmsnorm(float *output, const float *input, const float *weight,
                    int dimension, float eps) {
    if (!output || !input || !weight || dimension < 1 || eps < 0.0f)
        return -1;
    float mean_square = 0.0f;
    for (int index = 0; index < dimension; index++)
        mean_square += input[index] * input[index];
    float inverse_rms = 1.0f / sqrtf(mean_square / dimension + eps);
    for (int index = 0; index < dimension; index++)
        output[index] = input[index] * inverse_rms * weight[index];
    return 0;
}

static void rope_yarn_bounds(int dimension, int original_sequence_length,
                             float base, int beta_fast, int beta_slow,
                             int *low, int *high) {
    *low = 0;
    *high = -1;
    if (original_sequence_length > 0) {
        const float two_pi = 6.2831853071795864769f;
        float denominator = 2.0f * logf(base);
        float low_value = dimension * logf(
            original_sequence_length / (beta_fast * two_pi)) / denominator;
        float high_value = dimension * logf(
            original_sequence_length / (beta_slow * two_pi)) / denominator;
        *low = (int)floorf(low_value);
        *high = (int)ceilf(high_value);
        if (*low < 0) *low = 0;
        if (*high > dimension - 1) *high = dimension - 1;
    }
}

static float rope_frequency(int pair, int dimension,
                            int original_sequence_length, float base,
                            float factor, int low, int high) {
    float frequency = 1.0f / powf(base, (float)(2 * pair) / dimension);
    if (original_sequence_length > 0) {
        float width = high == low ? 0.001f : (float)(high - low);
        float ramp = (pair - low) / width;
        if (ramp < 0.0f) ramp = 0.0f;
        if (ramp > 1.0f) ramp = 1.0f;
        float smooth = 1.0f - ramp;
        frequency = frequency / factor * (1.0f - smooth) + frequency * smooth;
    }
    return frequency;
}

int coli_v4_rope_precompute_range(
    float *cosines, float *sines, int dimension, int start_position,
    int sequence_length, int original_sequence_length, float base,
    float factor, int beta_fast, int beta_slow) {
    if (!cosines || !sines || dimension < 2 || (dimension & 1) ||
        start_position < 0 || sequence_length < 1 ||
        !(base > 1.0f) || !(factor > 0.0f))
        return -1;
    if (start_position > INT_MAX - (sequence_length - 1)) return -1;
    int pairs = dimension / 2;
    int low = 0, high = -1;
    rope_yarn_bounds(dimension, original_sequence_length, base,
                     beta_fast, beta_slow, &low, &high);
    for (int pair = 0; pair < pairs; pair++) {
        float frequency = rope_frequency(pair, dimension,
                                         original_sequence_length, base,
                                         factor, low, high);
        for (int item = 0; item < sequence_length; item++) {
            size_t index = (size_t)item * pairs + pair;
            float angle = (start_position + item) * frequency;
            cosines[index] = cosf(angle);
            sines[index] = sinf(angle);
        }
    }
    return 0;
}

int coli_v4_rope_precompute(float *cosines, float *sines,
                            int dimension, int sequence_length,
                            int original_sequence_length, float base,
                            float factor, int beta_fast, int beta_slow) {
    return coli_v4_rope_precompute_range(
        cosines, sines, dimension, 0, sequence_length,
        original_sequence_length, base, factor, beta_fast, beta_slow);
}

int coli_v4_rope_position(float *cosines, float *sines,
                          int dimension, int position,
                          int original_sequence_length, float base,
                          float factor, int beta_fast, int beta_slow) {
    if (!cosines || !sines || dimension < 2 || (dimension & 1) ||
        position < 0 || !(base > 1.0f) || !(factor > 0.0f))
        return -1;
    int pairs = dimension / 2;
    int low = 0, high = -1;
    rope_yarn_bounds(dimension, original_sequence_length, base,
                     beta_fast, beta_slow, &low, &high);
    for (int pair = 0; pair < pairs; pair++) {
        float frequency = rope_frequency(pair, dimension,
                                         original_sequence_length, base,
                                         factor, low, high);
        float angle = position * frequency;
        cosines[pair] = cosf(angle);
        sines[pair] = sinf(angle);
    }
    return 0;
}

int coli_v4_rope_apply(float *vectors, int vector_count, int dimension,
                       const float *cosines, const float *sines, int inverse) {
    if (!vectors || !cosines || !sines || vector_count < 1 ||
        dimension < 2 || (dimension & 1))
        return -1;
    int pairs = dimension / 2;
    float direction = inverse ? -1.0f : 1.0f;
    for (int vector = 0; vector < vector_count; vector++) {
        for (int pair = 0; pair < pairs; pair++) {
            size_t value_index = (size_t)vector * dimension + 2 * pair;
            size_t frequency_index = (size_t)vector * pairs + pair;
            float real = vectors[value_index];
            float imaginary = vectors[value_index + 1];
            float cosine = cosines[frequency_index];
            float sine = sines[frequency_index] * direction;
            vectors[value_index] = real * cosine - imaginary * sine;
            vectors[value_index + 1] = real * sine + imaginary * cosine;
        }
    }
    return 0;
}

static float softplusf_stable(float value) {
    return fmaxf(value, 0.0f) + log1pf(expf(-fabsf(value)));
}

int coli_v4_route(float *weights, int *indices, const float *hidden,
                  const float *gate, const float *bias,
                  const int *forced_indices, int experts, int dimension,
                  int topk, float route_scale) {
    if (!weights || !indices || !hidden || !gate || experts < 1 ||
        dimension < 1 || topk < 1 || topk > experts)
        return -1;
    float *scores = malloc((size_t)experts * sizeof(*scores));
    float *selection = malloc((size_t)experts * sizeof(*selection));
    unsigned char *selected = calloc((size_t)experts, 1);
    if (!scores || !selection || !selected) {
        free(scores);
        free(selection);
        free(selected);
        return -1;
    }
    for (int expert = 0; expert < experts; expert++) {
        float sum = 0.0f;
        for (int column = 0; column < dimension; column++)
            sum += gate[(size_t)expert * dimension + column] * hidden[column];
        scores[expert] = sqrtf(softplusf_stable(sum));
        selection[expert] = scores[expert] + (bias ? bias[expert] : 0.0f);
    }
    if (forced_indices) {
        for (int rank = 0; rank < topk; rank++) {
            if (forced_indices[rank] < 0 || forced_indices[rank] >= experts) {
                free(selected);
                free(selection);
                free(scores);
                return -1;
            }
            indices[rank] = forced_indices[rank];
        }
    } else {
        for (int rank = 0; rank < topk; rank++) {
            int best = -1;
            for (int expert = 0; expert < experts; expert++) {
                if (!selected[expert] &&
                    (best < 0 || selection[expert] > selection[best]))
                    best = expert;
            }
            indices[rank] = best;
            selected[best] = 1;
        }
    }
    float total = 0.0f;
    for (int rank = 0; rank < topk; rank++)
        total += scores[indices[rank]];
    if (!(total > 0.0f)) {
        free(selected);
        free(selection);
        free(scores);
        return -1;
    }
    for (int rank = 0; rank < topk; rank++)
        weights[rank] = scores[indices[rank]] / total * route_scale;
    free(selected);
    free(selection);
    free(scores);
    return 0;
}

int coli_v4_swiglu(float *output, const float *gate, const float *up,
                   int dimension, float limit) {
    if (!output || !gate || !up || dimension < 1 || limit < 0.0f)
        return -1;
    for (int index = 0; index < dimension; index++) {
        float gate_value = gate[index];
        float up_value = up[index];
        if (limit > 0.0f) {
            gate_value = fminf(gate_value, limit);
            up_value = fmaxf(-limit, fminf(up_value, limit));
        }
        output[index] = gate_value * sigmoidf_stable(gate_value) * up_value;
    }
    return 0;
}
#endif /* COLI_V4_UNIT_MATH */

#ifdef COLI_V4_UNIT_ATTENTION
/* ######## deepseek_v4_attention.c ######## */
#include "deepseek_v4_internal.h"

#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "deepseek_v4_internal.h"
#include "deepseek_v4_internal.h"
#include "deepseek_v4_internal.h"
#include "deepseek_v4_internal.h"
#include "native_quant.h"

static int set_error(char *error, size_t size, const char *format, ...);

struct ColiDeepSeekV4WindowAttentionState {
    int window_size;
    int head_dim;
    int layer;
    int ratio;
    float *kv;
    ColiDeepSeekV4CompressorState *compressor;
    ColiDeepSeekV4Indexer *indexer;
    float *compressed;
    int compressed_count;
    int compressed_capacity;
};

int coli_v4_window_attention_create(ColiDeepSeekV4WindowAttentionState **output,
                                    const ColiDeepSeekV4Config *config) {
    if (!output || !config || config->sliding_window < 1 || config->head_dim < 1)
        return -1;
    *output = calloc(1, sizeof(**output));
    if (!*output) return -1;
    (*output)->window_size = config->sliding_window;
    (*output)->head_dim = config->head_dim;
    (*output)->layer = -1;
    (*output)->kv = calloc((size_t)config->sliding_window * config->head_dim,
                           sizeof(*(*output)->kv));
    if (!(*output)->kv) {
        free(*output);
        *output = NULL;
        return -1;
    }
    return 0;
}

void coli_v4_window_attention_reset(ColiDeepSeekV4WindowAttentionState *state) {
    if (!state) return;
    memset(state->kv, 0,
           (size_t)state->window_size * state->head_dim * sizeof(*state->kv));
    state->compressed_count = 0;
    if (state->compressor) coli_v4_compressor_reset(state->compressor);
    if (state->indexer) coli_v4_indexer_reset(state->indexer);
}

void coli_v4_window_attention_destroy(ColiDeepSeekV4WindowAttentionState *state) {
    if (!state) return;
    coli_v4_indexer_destroy(state->indexer);
    coli_v4_compressor_destroy(state->compressor);
    free(state->compressed);
    free(state->kv);
    free(state);
}

static int prepare_compressed_state(
    ColiDeepSeekV4WindowAttentionState *state,
    const ColiDeepSeekV4LayerWeights *weights,
    const ColiDeepSeekV4Config *config, char *error, size_t error_size) {
    int ratio = weights->plan.compression_ratio;
    if (!ratio) return 0;
    if (state->layer < 0) {
        state->layer = weights->plan.layer;
        state->ratio = ratio;
        state->compressed_capacity = 16;
        state->compressed = calloc((size_t)state->compressed_capacity * state->head_dim,
                                   sizeof(*state->compressed));
        if (!state->compressed || coli_v4_compressor_create(
                &state->compressor, weights, config, error, error_size)) return -1;
        if (ratio == 4 && coli_v4_indexer_create(
                &state->indexer, weights, config, config->max_position_embeddings,
                error, error_size)) return -1;
    } else if (state->layer != weights->plan.layer || state->ratio != ratio) {
        return set_error(error, error_size, "attention state belongs to another layer");
    }
    if (coli_v4_compressor_bind_weights(state->compressor, weights,
                                        error, error_size)) return -1;
    if (state->indexer && coli_v4_indexer_bind_weights(
            state->indexer, weights, error, error_size)) return -1;
    return 0;
}

static int grow_compressed_state(ColiDeepSeekV4WindowAttentionState *state,
                                 char *error, size_t error_size) {
    if (state->compressed_count < state->compressed_capacity) return 0;
    int capacity = state->compressed_capacity * 2;
    float *grown = realloc(state->compressed,
        (size_t)capacity * state->head_dim * sizeof(*grown));
    if (!grown) return set_error(error, error_size, "cannot grow compressed KV cache");
    state->compressed = grown;
    state->compressed_capacity = capacity;
    return 0;
}

static int set_error(char *error, size_t size, const char *format, ...) {
    if (error && size) {
        va_list arguments;
        va_start(arguments, format);
        vsnprintf(error, size, format, arguments);
        va_end(arguments);
    }
    return -1;
}

static const void *layer_data(const ColiDeepSeekV4LayerWeights *weights,
                              const char *suffix,
                              const ColiDeepSeekV4TensorSpec **spec) {
    char name[COLI_V4_MAX_TENSOR_NAME];
    snprintf(name, sizeof(name), "layers.%d.%s", weights->plan.layer, suffix);
    return coli_v4_layer_data(weights, name, spec);
}

static int fp8_view(ColiTensorView *view,
                    const ColiDeepSeekV4LayerWeights *weights,
                    const char *prefix) {
    char suffix[128];
    const ColiDeepSeekV4TensorSpec *weight_spec = NULL, *scale_spec = NULL;
    snprintf(suffix, sizeof(suffix), "%s.weight", prefix);
    const void *data = layer_data(weights, suffix, &weight_spec);
    snprintf(suffix, sizeof(suffix), "%s.scale", prefix);
    const void *scales = layer_data(weights, suffix, &scale_spec);
    if (!data || !scales || !weight_spec || !scale_spec ||
        weight_spec->dtype != COLI_ST_F8_E4M3 ||
        scale_spec->dtype != COLI_ST_F8_E8M0 || weight_spec->rank != 2)
        return -1;
    *view = (ColiTensorView){
        COLI_TENSOR_FP8_E4M3_BLOCK, COLI_SCALE_F32, data, scales,
        (size_t)(weight_spec->shape[0] * weight_spec->shape[1]),
        (size_t)(scale_spec->shape[0] * scale_spec->shape[1]) * sizeof(float),
        weight_spec->shape[0], weight_spec->shape[1],
        weight_spec->packed_rows8 ? 8 : 128, 128
    };
    return 0;
}

static int decode_bf16(float *output, const void *data, size_t count) {
    if (!output || !data) return -1;
    const uint16_t *values = data;
    for (size_t i = 0; i < count; i++) output[i] = coli_bf16_decode(values[i]);
    return 0;
}

static int attention_token_impl(float *output,
                                ColiDeepSeekV4WindowAttentionState *state,
                                const ColiDeepSeekV4LayerWeights *weights,
                                const ColiDeepSeekV4Config *config,
                                const float *input, int position,
                                char *error, size_t error_size) {
    if (!output || !weights || !config || !input || position < 0 ||
        (!state && weights->plan.compression_ratio != 0 && position != 0))
        return set_error(error, error_size, "invalid uncompressed attention arguments");
    int hidden = config->hidden_size;
    int heads = config->num_attention_heads;
    int head_dim = config->head_dim;
    int rope_dim = config->qk_rope_head_dim;
    int q_rank = config->q_lora_rank;
    int groups = config->o_groups;
    int o_rank = config->o_lora_rank;
    if (hidden < 1 || heads < 1 || head_dim < 1 || rope_dim < 2 ||
        rope_dim > head_dim || q_rank < 1 || groups < 1 || heads % groups)
        return set_error(error, error_size, "unsupported attention dimensions");

    ColiTensorView wq_a, wq_b, wkv, wo_a, wo_b;
    if (fp8_view(&wq_a, weights, "attn.wq_a") ||
        fp8_view(&wq_b, weights, "attn.wq_b") ||
        fp8_view(&wkv, weights, "attn.wkv") ||
        fp8_view(&wo_a, weights, "attn.wo_a") ||
        fp8_view(&wo_b, weights, "attn.wo_b"))
        return set_error(error, error_size, "missing native FP8 attention tensor");

    float *qa = calloc((size_t)q_rank, sizeof(*qa));
    float *q = calloc((size_t)heads * head_dim, sizeof(*q));
    float *kv = calloc((size_t)head_dim, sizeof(*kv));
    float *attended = calloc((size_t)heads * head_dim, sizeof(*attended));
    float *oa = calloc((size_t)groups * o_rank, sizeof(*oa));
    float *norm_weight = calloc((size_t)(q_rank > head_dim ? q_rank : head_dim),
                                sizeof(*norm_weight));
    float *cosines = calloc((size_t)rope_dim / 2, sizeof(*cosines));
    float *sines = calloc((size_t)rope_dim / 2, sizeof(*sines));
    int *compressed_indices = NULL;
    int compressed_selected = 0;
    if (!qa || !q || !kv || !attended || !oa || !norm_weight || !cosines || !sines) {
        free(sines); free(cosines); free(norm_weight); free(oa);
        free(attended); free(kv); free(q); free(qa);
        return set_error(error, error_size, "out of memory in attention");
    }

    int result = coli_fp8_matvec_ref(qa, &wq_a, input);
    coli_bf16_round_array(qa, (size_t)q_rank);
    const void *q_norm = layer_data(weights, "attn.q_norm.weight", NULL);
    if (!result && (!q_norm || decode_bf16(norm_weight, q_norm, (size_t)q_rank) ||
                    coli_v4_rmsnorm(qa, qa, norm_weight, q_rank,
                                    config->rms_norm_eps))) result = -1;
    if (!result) coli_bf16_round_array(qa, (size_t)q_rank);
    if (!result && state && weights->plan.compression_ratio) {
        result = prepare_compressed_state(state, weights, config,
                                          error, error_size);
        if (!result && (position + 1) % state->ratio == 0)
            result = grow_compressed_state(state, error, error_size);
        int produced = 0;
        if (!result) result = coli_v4_compressor_step(
            state->compressor,
            state->compressed + (size_t)state->compressed_count * head_dim,
            &produced, input, position, error, error_size);
        if (!result && produced) state->compressed_count++;
        if (!result && state->indexer) {
            compressed_indices = malloc((size_t)config->index_topk *
                                        sizeof(*compressed_indices));
            if (!compressed_indices) result = -1;
            else compressed_selected = coli_v4_indexer_step(
                state->indexer, compressed_indices, config->index_topk,
                qa, input, position, error, error_size);
            if (compressed_selected < 0) result = -1;
        }
    }
    if (!result) result = coli_fp8_matvec_ref(q, &wq_b, qa);
    if (!result) coli_bf16_round_array(q, (size_t)heads * head_dim);
    for (int head = 0; !result && head < heads; head++) {
        float *values = q + (size_t)head * head_dim;
        float mean_square = 0.0f;
        for (int i = 0; i < head_dim; i++) mean_square += values[i] * values[i];
        float scale = 1.0f / sqrtf(mean_square / head_dim + config->rms_norm_eps);
        for (int i = 0; i < head_dim; i++) values[i] = coli_bf16_round(values[i] * scale);
    }

    if (!result) result = coli_fp8_matvec_ref(kv, &wkv, input);
    if (!result) coli_bf16_round_array(kv, (size_t)head_dim);
    const void *kv_norm = layer_data(weights, "attn.kv_norm.weight", NULL);
    if (!result && (!kv_norm || decode_bf16(norm_weight, kv_norm, (size_t)head_dim) ||
                    coli_v4_rmsnorm(kv, kv, norm_weight, head_dim,
                                    config->rms_norm_eps))) result = -1;
    if (!result) coli_bf16_round_array(kv, (size_t)head_dim);

    if (!result) {
        int compressed = weights->plan.compression_ratio != 0;
        if (coli_v4_rope_position(
                cosines, sines, rope_dim, position,
                compressed ? config->original_max_position_embeddings : 0,
                compressed ? config->compress_rope_theta : config->rope_theta,
                config->rope_factor,
                config->rope_beta_fast, config->rope_beta_slow)) result = -1;
    }
    if (!result) {
        for (int head = 0; head < heads; head++) {
            float *rope = q + (size_t)head * head_dim + head_dim - rope_dim;
            coli_v4_rope_apply(rope, 1, rope_dim, cosines, sines, 0);
            coli_bf16_round_array(rope, (size_t)rope_dim);
        }
        float *kv_rope = kv + head_dim - rope_dim;
        coli_v4_rope_apply(kv_rope, 1, rope_dim, cosines, sines, 0);
        coli_bf16_round_array(kv_rope, (size_t)rope_dim);
        size_t nope = (size_t)(head_dim - rope_dim);
        float *qdq = malloc(nope * sizeof(*qdq));
        uint8_t *scales = malloc((nope + 63) / 64);
        if (!qdq || !scales || coli_fp8_activation_qdq_ref(qdq, scales, kv, nope, 64))
            result = -1;
        if (!result) {
            memcpy(kv, qdq, nope * sizeof(*kv));
            coli_bf16_round_array(kv, nope);
        }
        free(scales); free(qdq);
    }

    const float *sinks = layer_data(weights, "attn.attn_sink", NULL);
    if (!result && state) {
        int slot = position % state->window_size;
        memcpy(state->kv + (size_t)slot * head_dim, kv,
               (size_t)head_dim * sizeof(*kv));
        if (!state->indexer) compressed_selected = state->compressed_count;
        int topk = state->window_size + compressed_selected;
        int kv_count = state->window_size + state->compressed_count;
        int *indices = malloc((size_t)topk * sizeof(*indices));
        float *all_kv = state->compressed_count
            ? malloc((size_t)kv_count * head_dim * sizeof(*all_kv)) : NULL;
        if (!indices || (state->compressed_count && !all_kv)) result = -1;
        if (!result) {
            if (position < state->window_size - 1) {
                for (int i = 0; i < state->window_size; i++)
                    indices[i] = i <= position ? i : -1;
            } else {
                int oldest = (position + 1) % state->window_size;
                for (int i = 0; i < state->window_size; i++)
                    indices[i] = (oldest + i) % state->window_size;
            }
            const float *kv_values = state->kv;
            if (state->compressed_count) {
                memcpy(all_kv, state->kv,
                       (size_t)state->window_size * head_dim * sizeof(*all_kv));
                memcpy(all_kv + (size_t)state->window_size * head_dim,
                       state->compressed,
                       (size_t)state->compressed_count * head_dim * sizeof(*all_kv));
                kv_values = all_kv;
            }
            for (int i = 0; i < compressed_selected; i++) {
                int ordinal = state->indexer ? compressed_indices[i] : i;
                indices[state->window_size + i] = state->window_size + ordinal;
            }
            result = coli_v4_sparse_attention_ref(
                attended, q, kv_values, sinks, indices, heads, head_dim,
                kv_count, topk,
                1.0f / sqrtf((float)head_dim));
        }
        free(all_kv);
        free(indices);
    } else for (int head = 0; !result && head < heads; head++) {
        float *query = q + (size_t)head * head_dim;
        float score = 0.0f;
        for (int i = 0; i < head_dim; i++) score += query[i] * kv[i];
        score *= 1.0f / sqrtf((float)head_dim);
        float attention_weight = 1.0f / (1.0f + expf(sinks[head] - score));
        float *head_output = attended + (size_t)head * head_dim;
        for (int i = 0; i < head_dim; i++)
            head_output[i] = coli_bf16_round(kv[i] * attention_weight);
    }
    for (int head = 0; !result && head < heads; head++) {
        float *head_output = attended + (size_t)head * head_dim;
        float *rope = head_output + head_dim - rope_dim;
        coli_v4_rope_apply(rope, 1, rope_dim, cosines, sines, 1);
        coli_bf16_round_array(rope, (size_t)rope_dim);
    }

    int heads_per_group = heads / groups;
    int group_width = heads_per_group * head_dim;
    int scale_columns = (group_width + 127) / 128;
    int scale_rows_per_group = (o_rank + 127) / 128;
    for (int group = 0; !result && group < groups; group++) {
        ColiTensorView group_view = wo_a;
        group_view.rows = o_rank;
        group_view.columns = group_width;
        group_view.data = (const uint8_t *)wo_a.data +
            (size_t)group * o_rank * group_width;
        group_view.scales = (const uint8_t *)wo_a.scales +
            (size_t)group * scale_rows_per_group * scale_columns * sizeof(float);
        group_view.data_bytes = (size_t)o_rank * group_width;
        group_view.scale_bytes =
            (size_t)scale_rows_per_group * scale_columns * sizeof(float);
        result = coli_fp8_matvec_ref(oa + (size_t)group * o_rank, &group_view,
                                     attended + (size_t)group * group_width);
    }
    if (!result) coli_bf16_round_array(oa, (size_t)groups * o_rank);
    if (!result) result = coli_fp8_matvec_ref(output, &wo_b, oa);
    if (!result) coli_bf16_round_array(output, (size_t)hidden);

    free(compressed_indices);
    free(sines); free(cosines); free(norm_weight); free(oa);
    free(attended); free(kv); free(q); free(qa);
    if (result) return set_error(error, error_size, "attention computation failed");
    return 0;
}

int coli_v4_attention_token_ref(float *output,
                                const ColiDeepSeekV4LayerWeights *weights,
                                const ColiDeepSeekV4Config *config,
                                const float *input, int position,
                                char *error, size_t error_size) {
    return attention_token_impl(output, NULL, weights, config, input, position,
                                error, error_size);
}

int coli_v4_attention_window_token_ref(
    float *output, ColiDeepSeekV4WindowAttentionState *state,
    const ColiDeepSeekV4LayerWeights *weights,
    const ColiDeepSeekV4Config *config, const float *input, int position,
    char *error, size_t error_size) {
    return attention_token_impl(output, state, weights, config, input, position,
                                error, error_size);
}
#endif /* COLI_V4_UNIT_ATTENTION */

#ifdef COLI_V4_UNIT_ATTENTION_BATCH
/* ######## deepseek_v4_attention_batch.c ######## */
#define coli_v4_window_attention_create coli_v4_window_attention_batch_create_copy
#define coli_v4_window_attention_reset coli_v4_window_attention_batch_reset_copy
#define coli_v4_window_attention_destroy coli_v4_window_attention_batch_destroy_copy
#define coli_v4_attention_token_ref coli_v4_attention_token_batch_serial_copy
#define coli_v4_attention_window_token_ref coli_v4_attention_window_token_batch_serial_copy
/* ---- begin include deepseek_v4_attention.c ---- */
#include "deepseek_v4_internal.h"

#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "deepseek_v4_internal.h"
#include "deepseek_v4_internal.h"
#include "deepseek_v4_internal.h"
#include "deepseek_v4_internal.h"
#include "native_quant.h"

static int set_error(char *error, size_t size, const char *format, ...);

struct ColiDeepSeekV4WindowAttentionState {
    int window_size;
    int head_dim;
    int layer;
    int ratio;
    float *kv;
    ColiDeepSeekV4CompressorState *compressor;
    ColiDeepSeekV4Indexer *indexer;
    float *compressed;
    int compressed_count;
    int compressed_capacity;
};

int coli_v4_window_attention_create(ColiDeepSeekV4WindowAttentionState **output,
                                    const ColiDeepSeekV4Config *config) {
    if (!output || !config || config->sliding_window < 1 || config->head_dim < 1)
        return -1;
    *output = calloc(1, sizeof(**output));
    if (!*output) return -1;
    (*output)->window_size = config->sliding_window;
    (*output)->head_dim = config->head_dim;
    (*output)->layer = -1;
    (*output)->kv = calloc((size_t)config->sliding_window * config->head_dim,
                           sizeof(*(*output)->kv));
    if (!(*output)->kv) {
        free(*output);
        *output = NULL;
        return -1;
    }
    return 0;
}

void coli_v4_window_attention_reset(ColiDeepSeekV4WindowAttentionState *state) {
    if (!state) return;
    memset(state->kv, 0,
           (size_t)state->window_size * state->head_dim * sizeof(*state->kv));
    state->compressed_count = 0;
    if (state->compressor) coli_v4_compressor_reset(state->compressor);
    if (state->indexer) coli_v4_indexer_reset(state->indexer);
}

void coli_v4_window_attention_destroy(ColiDeepSeekV4WindowAttentionState *state) {
    if (!state) return;
    coli_v4_indexer_destroy(state->indexer);
    coli_v4_compressor_destroy(state->compressor);
    free(state->compressed);
    free(state->kv);
    free(state);
}

static int prepare_compressed_state(
    ColiDeepSeekV4WindowAttentionState *state,
    const ColiDeepSeekV4LayerWeights *weights,
    const ColiDeepSeekV4Config *config, char *error, size_t error_size) {
    int ratio = weights->plan.compression_ratio;
    if (!ratio) return 0;
    if (state->layer < 0) {
        state->layer = weights->plan.layer;
        state->ratio = ratio;
        state->compressed_capacity = 16;
        state->compressed = calloc((size_t)state->compressed_capacity * state->head_dim,
                                   sizeof(*state->compressed));
        if (!state->compressed || coli_v4_compressor_create(
                &state->compressor, weights, config, error, error_size)) return -1;
        if (ratio == 4 && coli_v4_indexer_create(
                &state->indexer, weights, config, config->max_position_embeddings,
                error, error_size)) return -1;
    } else if (state->layer != weights->plan.layer || state->ratio != ratio) {
        return set_error(error, error_size, "attention state belongs to another layer");
    }
    if (coli_v4_compressor_bind_weights(state->compressor, weights,
                                        error, error_size)) return -1;
    if (state->indexer && coli_v4_indexer_bind_weights(
            state->indexer, weights, error, error_size)) return -1;
    return 0;
}

static int grow_compressed_state(ColiDeepSeekV4WindowAttentionState *state,
                                 char *error, size_t error_size) {
    if (state->compressed_count < state->compressed_capacity) return 0;
    int capacity = state->compressed_capacity * 2;
    float *grown = realloc(state->compressed,
        (size_t)capacity * state->head_dim * sizeof(*grown));
    if (!grown) return set_error(error, error_size, "cannot grow compressed KV cache");
    state->compressed = grown;
    state->compressed_capacity = capacity;
    return 0;
}

static int set_error(char *error, size_t size, const char *format, ...) {
    if (error && size) {
        va_list arguments;
        va_start(arguments, format);
        vsnprintf(error, size, format, arguments);
        va_end(arguments);
    }
    return -1;
}

static const void *layer_data(const ColiDeepSeekV4LayerWeights *weights,
                              const char *suffix,
                              const ColiDeepSeekV4TensorSpec **spec) {
    char name[COLI_V4_MAX_TENSOR_NAME];
    snprintf(name, sizeof(name), "layers.%d.%s", weights->plan.layer, suffix);
    return coli_v4_layer_data(weights, name, spec);
}

static int fp8_view(ColiTensorView *view,
                    const ColiDeepSeekV4LayerWeights *weights,
                    const char *prefix) {
    char suffix[128];
    const ColiDeepSeekV4TensorSpec *weight_spec = NULL, *scale_spec = NULL;
    snprintf(suffix, sizeof(suffix), "%s.weight", prefix);
    const void *data = layer_data(weights, suffix, &weight_spec);
    snprintf(suffix, sizeof(suffix), "%s.scale", prefix);
    const void *scales = layer_data(weights, suffix, &scale_spec);
    if (!data || !scales || !weight_spec || !scale_spec ||
        weight_spec->dtype != COLI_ST_F8_E4M3 ||
        scale_spec->dtype != COLI_ST_F8_E8M0 || weight_spec->rank != 2)
        return -1;
    *view = (ColiTensorView){
        COLI_TENSOR_FP8_E4M3_BLOCK, COLI_SCALE_F32, data, scales,
        (size_t)(weight_spec->shape[0] * weight_spec->shape[1]),
        (size_t)(scale_spec->shape[0] * scale_spec->shape[1]) * sizeof(float),
        weight_spec->shape[0], weight_spec->shape[1],
        weight_spec->packed_rows8 ? 8 : 128, 128
    };
    return 0;
}

static int decode_bf16(float *output, const void *data, size_t count) {
    if (!output || !data) return -1;
    const uint16_t *values = data;
    for (size_t i = 0; i < count; i++) output[i] = coli_bf16_decode(values[i]);
    return 0;
}

static int attention_token_impl(float *output,
                                ColiDeepSeekV4WindowAttentionState *state,
                                const ColiDeepSeekV4LayerWeights *weights,
                                const ColiDeepSeekV4Config *config,
                                const float *input, int position,
                                char *error, size_t error_size) {
    if (!output || !weights || !config || !input || position < 0 ||
        (!state && weights->plan.compression_ratio != 0 && position != 0))
        return set_error(error, error_size, "invalid uncompressed attention arguments");
    int hidden = config->hidden_size;
    int heads = config->num_attention_heads;
    int head_dim = config->head_dim;
    int rope_dim = config->qk_rope_head_dim;
    int q_rank = config->q_lora_rank;
    int groups = config->o_groups;
    int o_rank = config->o_lora_rank;
    if (hidden < 1 || heads < 1 || head_dim < 1 || rope_dim < 2 ||
        rope_dim > head_dim || q_rank < 1 || groups < 1 || heads % groups)
        return set_error(error, error_size, "unsupported attention dimensions");

    ColiTensorView wq_a, wq_b, wkv, wo_a, wo_b;
    if (fp8_view(&wq_a, weights, "attn.wq_a") ||
        fp8_view(&wq_b, weights, "attn.wq_b") ||
        fp8_view(&wkv, weights, "attn.wkv") ||
        fp8_view(&wo_a, weights, "attn.wo_a") ||
        fp8_view(&wo_b, weights, "attn.wo_b"))
        return set_error(error, error_size, "missing native FP8 attention tensor");

    float *qa = calloc((size_t)q_rank, sizeof(*qa));
    float *q = calloc((size_t)heads * head_dim, sizeof(*q));
    float *kv = calloc((size_t)head_dim, sizeof(*kv));
    float *attended = calloc((size_t)heads * head_dim, sizeof(*attended));
    float *oa = calloc((size_t)groups * o_rank, sizeof(*oa));
    float *norm_weight = calloc((size_t)(q_rank > head_dim ? q_rank : head_dim),
                                sizeof(*norm_weight));
    float *cosines = calloc((size_t)rope_dim / 2, sizeof(*cosines));
    float *sines = calloc((size_t)rope_dim / 2, sizeof(*sines));
    int *compressed_indices = NULL;
    int compressed_selected = 0;
    if (!qa || !q || !kv || !attended || !oa || !norm_weight || !cosines || !sines) {
        free(sines); free(cosines); free(norm_weight); free(oa);
        free(attended); free(kv); free(q); free(qa);
        return set_error(error, error_size, "out of memory in attention");
    }

    int result = coli_fp8_matvec_ref(qa, &wq_a, input);
    coli_bf16_round_array(qa, (size_t)q_rank);
    const void *q_norm = layer_data(weights, "attn.q_norm.weight", NULL);
    if (!result && (!q_norm || decode_bf16(norm_weight, q_norm, (size_t)q_rank) ||
                    coli_v4_rmsnorm(qa, qa, norm_weight, q_rank,
                                    config->rms_norm_eps))) result = -1;
    if (!result) coli_bf16_round_array(qa, (size_t)q_rank);
    if (!result && state && weights->plan.compression_ratio) {
        result = prepare_compressed_state(state, weights, config,
                                          error, error_size);
        if (!result && (position + 1) % state->ratio == 0)
            result = grow_compressed_state(state, error, error_size);
        int produced = 0;
        if (!result) result = coli_v4_compressor_step(
            state->compressor,
            state->compressed + (size_t)state->compressed_count * head_dim,
            &produced, input, position, error, error_size);
        if (!result && produced) state->compressed_count++;
        if (!result && state->indexer) {
            compressed_indices = malloc((size_t)config->index_topk *
                                        sizeof(*compressed_indices));
            if (!compressed_indices) result = -1;
            else compressed_selected = coli_v4_indexer_step(
                state->indexer, compressed_indices, config->index_topk,
                qa, input, position, error, error_size);
            if (compressed_selected < 0) result = -1;
        }
    }
    if (!result) result = coli_fp8_matvec_ref(q, &wq_b, qa);
    if (!result) coli_bf16_round_array(q, (size_t)heads * head_dim);
    for (int head = 0; !result && head < heads; head++) {
        float *values = q + (size_t)head * head_dim;
        float mean_square = 0.0f;
        for (int i = 0; i < head_dim; i++) mean_square += values[i] * values[i];
        float scale = 1.0f / sqrtf(mean_square / head_dim + config->rms_norm_eps);
        for (int i = 0; i < head_dim; i++) values[i] = coli_bf16_round(values[i] * scale);
    }

    if (!result) result = coli_fp8_matvec_ref(kv, &wkv, input);
    if (!result) coli_bf16_round_array(kv, (size_t)head_dim);
    const void *kv_norm = layer_data(weights, "attn.kv_norm.weight", NULL);
    if (!result && (!kv_norm || decode_bf16(norm_weight, kv_norm, (size_t)head_dim) ||
                    coli_v4_rmsnorm(kv, kv, norm_weight, head_dim,
                                    config->rms_norm_eps))) result = -1;
    if (!result) coli_bf16_round_array(kv, (size_t)head_dim);

    if (!result) {
        int compressed = weights->plan.compression_ratio != 0;
        if (coli_v4_rope_position(
                cosines, sines, rope_dim, position,
                compressed ? config->original_max_position_embeddings : 0,
                compressed ? config->compress_rope_theta : config->rope_theta,
                config->rope_factor,
                config->rope_beta_fast, config->rope_beta_slow)) result = -1;
    }
    if (!result) {
        for (int head = 0; head < heads; head++) {
            float *rope = q + (size_t)head * head_dim + head_dim - rope_dim;
            coli_v4_rope_apply(rope, 1, rope_dim, cosines, sines, 0);
            coli_bf16_round_array(rope, (size_t)rope_dim);
        }
        float *kv_rope = kv + head_dim - rope_dim;
        coli_v4_rope_apply(kv_rope, 1, rope_dim, cosines, sines, 0);
        coli_bf16_round_array(kv_rope, (size_t)rope_dim);
        size_t nope = (size_t)(head_dim - rope_dim);
        float *qdq = malloc(nope * sizeof(*qdq));
        uint8_t *scales = malloc((nope + 63) / 64);
        if (!qdq || !scales || coli_fp8_activation_qdq_ref(qdq, scales, kv, nope, 64))
            result = -1;
        if (!result) {
            memcpy(kv, qdq, nope * sizeof(*kv));
            coli_bf16_round_array(kv, nope);
        }
        free(scales); free(qdq);
    }

    const float *sinks = layer_data(weights, "attn.attn_sink", NULL);
    if (!result && state) {
        int slot = position % state->window_size;
        memcpy(state->kv + (size_t)slot * head_dim, kv,
               (size_t)head_dim * sizeof(*kv));
        if (!state->indexer) compressed_selected = state->compressed_count;
        int topk = state->window_size + compressed_selected;
        int kv_count = state->window_size + state->compressed_count;
        int *indices = malloc((size_t)topk * sizeof(*indices));
        float *all_kv = state->compressed_count
            ? malloc((size_t)kv_count * head_dim * sizeof(*all_kv)) : NULL;
        if (!indices || (state->compressed_count && !all_kv)) result = -1;
        if (!result) {
            if (position < state->window_size - 1) {
                for (int i = 0; i < state->window_size; i++)
                    indices[i] = i <= position ? i : -1;
            } else {
                int oldest = (position + 1) % state->window_size;
                for (int i = 0; i < state->window_size; i++)
                    indices[i] = (oldest + i) % state->window_size;
            }
            const float *kv_values = state->kv;
            if (state->compressed_count) {
                memcpy(all_kv, state->kv,
                       (size_t)state->window_size * head_dim * sizeof(*all_kv));
                memcpy(all_kv + (size_t)state->window_size * head_dim,
                       state->compressed,
                       (size_t)state->compressed_count * head_dim * sizeof(*all_kv));
                kv_values = all_kv;
            }
            for (int i = 0; i < compressed_selected; i++) {
                int ordinal = state->indexer ? compressed_indices[i] : i;
                indices[state->window_size + i] = state->window_size + ordinal;
            }
            result = coli_v4_sparse_attention_ref(
                attended, q, kv_values, sinks, indices, heads, head_dim,
                kv_count, topk,
                1.0f / sqrtf((float)head_dim));
        }
        free(all_kv);
        free(indices);
    } else for (int head = 0; !result && head < heads; head++) {
        float *query = q + (size_t)head * head_dim;
        float score = 0.0f;
        for (int i = 0; i < head_dim; i++) score += query[i] * kv[i];
        score *= 1.0f / sqrtf((float)head_dim);
        float attention_weight = 1.0f / (1.0f + expf(sinks[head] - score));
        float *head_output = attended + (size_t)head * head_dim;
        for (int i = 0; i < head_dim; i++)
            head_output[i] = coli_bf16_round(kv[i] * attention_weight);
    }
    for (int head = 0; !result && head < heads; head++) {
        float *head_output = attended + (size_t)head * head_dim;
        float *rope = head_output + head_dim - rope_dim;
        coli_v4_rope_apply(rope, 1, rope_dim, cosines, sines, 1);
        coli_bf16_round_array(rope, (size_t)rope_dim);
    }

    int heads_per_group = heads / groups;
    int group_width = heads_per_group * head_dim;
    int scale_columns = (group_width + 127) / 128;
    int scale_rows_per_group = (o_rank + 127) / 128;
    for (int group = 0; !result && group < groups; group++) {
        ColiTensorView group_view = wo_a;
        group_view.rows = o_rank;
        group_view.columns = group_width;
        group_view.data = (const uint8_t *)wo_a.data +
            (size_t)group * o_rank * group_width;
        group_view.scales = (const uint8_t *)wo_a.scales +
            (size_t)group * scale_rows_per_group * scale_columns * sizeof(float);
        group_view.data_bytes = (size_t)o_rank * group_width;
        group_view.scale_bytes =
            (size_t)scale_rows_per_group * scale_columns * sizeof(float);
        result = coli_fp8_matvec_ref(oa + (size_t)group * o_rank, &group_view,
                                     attended + (size_t)group * group_width);
    }
    if (!result) coli_bf16_round_array(oa, (size_t)groups * o_rank);
    if (!result) result = coli_fp8_matvec_ref(output, &wo_b, oa);
    if (!result) coli_bf16_round_array(output, (size_t)hidden);

    free(compressed_indices);
    free(sines); free(cosines); free(norm_weight); free(oa);
    free(attended); free(kv); free(q); free(qa);
    if (result) return set_error(error, error_size, "attention computation failed");
    return 0;
}

int coli_v4_attention_token_ref(float *output,
                                const ColiDeepSeekV4LayerWeights *weights,
                                const ColiDeepSeekV4Config *config,
                                const float *input, int position,
                                char *error, size_t error_size) {
    return attention_token_impl(output, NULL, weights, config, input, position,
                                error, error_size);
}

int coli_v4_attention_window_token_ref(
    float *output, ColiDeepSeekV4WindowAttentionState *state,
    const ColiDeepSeekV4LayerWeights *weights,
    const ColiDeepSeekV4Config *config, const float *input, int position,
    char *error, size_t error_size) {
    return attention_token_impl(output, state, weights, config, input, position,
                                error, error_size);
}
/* ---- end include deepseek_v4_attention.c ---- */

#undef coli_v4_window_attention_create
#undef coli_v4_window_attention_reset
#undef coli_v4_window_attention_destroy
#undef coli_v4_attention_token_ref
#undef coli_v4_attention_window_token_ref

#include "deepseek_v4_internal.h"
#include "native_quant_batch.h"

int coli_v4_attention_window_batch_ref(
    float *outputs, ColiDeepSeekV4WindowAttentionState *state,
    const ColiDeepSeekV4LayerWeights *weights,
    const ColiDeepSeekV4Config *config, const float *inputs,
    int start_position, int batch, char *error, size_t error_size) {
    if (!outputs || !state || !weights || !config || !inputs ||
        start_position < 0 || batch < 1 || batch > 64)
        return set_error(error, error_size, "invalid batched attention arguments");
    int hidden = config->hidden_size, heads = config->num_attention_heads;
    int head_dim = config->head_dim, rope_dim = config->qk_rope_head_dim;
    int q_rank = config->q_lora_rank, groups = config->o_groups;
    int o_rank = config->o_lora_rank;
    size_t q_width = (size_t)heads * head_dim;
    size_t oa_width = (size_t)groups * o_rank;

    ColiTensorView wq_a, wq_b, wkv, wo_a, wo_b;
    if (fp8_view(&wq_a, weights, "attn.wq_a") ||
        fp8_view(&wq_b, weights, "attn.wq_b") ||
        fp8_view(&wkv, weights, "attn.wkv") ||
        fp8_view(&wo_a, weights, "attn.wo_a") ||
        fp8_view(&wo_b, weights, "attn.wo_b"))
        return set_error(error, error_size, "missing batched attention tensor");

    float *qa = calloc((size_t)batch * q_rank, sizeof(*qa));
    float *q = calloc((size_t)batch * q_width, sizeof(*q));
    float *kv = calloc((size_t)batch * head_dim, sizeof(*kv));
    float *attended = calloc((size_t)batch * q_width, sizeof(*attended));
    float *oa = calloc((size_t)batch * oa_width, sizeof(*oa));
    float *norm = malloc((size_t)(q_rank > head_dim ? q_rank : head_dim) *
                         sizeof(*norm));
    int *selected_counts = calloc((size_t)batch, sizeof(*selected_counts));
    int *compressed_counts = calloc((size_t)batch, sizeof(*compressed_counts));
    int *compressed_indices = malloc((size_t)batch * config->index_topk *
                                     sizeof(*compressed_indices));
    size_t rope_pairs = (size_t)rope_dim / 2;
    /* The batch consumes only its own positions.  A table from position zero
     * made long-context prefill scale with history that is never read here. */
    float *cosines = malloc((size_t)batch * rope_pairs * sizeof(*cosines));
    float *sines = malloc((size_t)batch * rope_pairs * sizeof(*sines));
    if (!qa || !q || !kv || !attended || !oa || !norm || !selected_counts ||
        !compressed_counts || !compressed_indices || !cosines || !sines) {
        free(sines); free(cosines); free(compressed_indices);
        free(compressed_counts); free(selected_counts); free(norm); free(oa);
        free(attended); free(kv); free(q); free(qa);
        return set_error(error, error_size, "out of memory in batched attention");
    }

    int result = coli_fp8_matmul_batch_ref(qa, &wq_a, inputs, batch);
    if (!result) coli_bf16_round_array(qa, (size_t)batch * q_rank);
    const void *raw_q_norm = layer_data(weights, "attn.q_norm.weight", NULL);
    if (!result && (!raw_q_norm || decode_bf16(norm, raw_q_norm, q_rank))) result = -1;
    for (int item = 0; !result && item < batch; item++) {
        float *item_qa = qa + (size_t)item * q_rank;
        result = coli_v4_rmsnorm(item_qa, item_qa, norm, q_rank,
                                 config->rms_norm_eps);
        if (!result) coli_bf16_round_array(item_qa, (size_t)q_rank);
    }

    for (int item = 0; !result && item < batch; item++) {
        int position = start_position + item;
        if (weights->plan.compression_ratio) {
            result = prepare_compressed_state(state, weights, config,
                                              error, error_size);
            if (!result && (position + 1) % state->ratio == 0)
                result = grow_compressed_state(state, error, error_size);
            int produced = 0;
            if (!result) result = coli_v4_compressor_step(
                state->compressor,
                state->compressed + (size_t)state->compressed_count * head_dim,
                &produced, inputs + (size_t)item * hidden, position,
                error, error_size);
            if (!result && produced) state->compressed_count++;
            compressed_counts[item] = state->compressed_count;
            if (!result && state->indexer) {
                selected_counts[item] = coli_v4_indexer_step(
                    state->indexer,
                    compressed_indices + (size_t)item * config->index_topk,
                    config->index_topk, qa + (size_t)item * q_rank,
                    inputs + (size_t)item * hidden, position,
                    error, error_size);
                if (selected_counts[item] < 0) result = -1;
            } else if (!result) {
                selected_counts[item] = state->compressed_count;
            }
        }
    }

    if (!result) result = coli_fp8_matmul_batch_ref(q, &wq_b, qa, batch);
    if (!result) coli_bf16_round_array(q, (size_t)batch * q_width);
    for (int item = 0; !result && item < batch; item++)
        for (int head = 0; head < heads; head++) {
            float *values = q + (size_t)item * q_width + (size_t)head * head_dim;
            float square = 0.0f;
            for (int i = 0; i < head_dim; i++) square += values[i] * values[i];
            float scale = 1.0f / sqrtf(square / head_dim + config->rms_norm_eps);
            for (int i = 0; i < head_dim; i++)
                values[i] = coli_bf16_round(values[i] * scale);
        }

    if (!result) result = coli_fp8_matmul_batch_ref(kv, &wkv, inputs, batch);
    if (!result) coli_bf16_round_array(kv, (size_t)batch * head_dim);
    const void *raw_kv_norm = layer_data(weights, "attn.kv_norm.weight", NULL);
    if (!result && (!raw_kv_norm || decode_bf16(norm, raw_kv_norm, head_dim))) result = -1;
    for (int item = 0; !result && item < batch; item++) {
        float *item_kv = kv + (size_t)item * head_dim;
        result = coli_v4_rmsnorm(item_kv, item_kv, norm, head_dim,
                                 config->rms_norm_eps);
        if (!result) coli_bf16_round_array(item_kv, (size_t)head_dim);
    }

    int compressed = weights->plan.compression_ratio != 0;
    if (!result) result = coli_v4_rope_precompute_range(
        cosines, sines, rope_dim, start_position, batch,
        compressed ? config->original_max_position_embeddings : 0,
        compressed ? config->compress_rope_theta : config->rope_theta,
        config->rope_factor, config->rope_beta_fast, config->rope_beta_slow);
    for (int item = 0; !result && item < batch; item++) {
        const float *item_cos = cosines + (size_t)item * rope_pairs;
        const float *item_sin = sines + (size_t)item * rope_pairs;
        float *item_q = q + (size_t)item * q_width;
        float *item_kv = kv + (size_t)item * head_dim;
        for (int head = 0; head < heads; head++) {
            float *rope = item_q + (size_t)head * head_dim + head_dim - rope_dim;
            coli_v4_rope_apply(rope, 1, rope_dim, item_cos, item_sin, 0);
            coli_bf16_round_array(rope, (size_t)rope_dim);
        }
        float *kv_rope = item_kv + head_dim - rope_dim;
        coli_v4_rope_apply(kv_rope, 1, rope_dim, item_cos, item_sin, 0);
        coli_bf16_round_array(kv_rope, (size_t)rope_dim);
        size_t nope = (size_t)(head_dim - rope_dim);
        float *qdq = malloc(nope * sizeof(*qdq));
        uint8_t *scales = malloc((nope + 63) / 64);
        if (!qdq || !scales || coli_fp8_activation_qdq_ref(
                qdq, scales, item_kv, nope, 64)) result = -1;
        if (!result) {
            memcpy(item_kv, qdq, nope * sizeof(*item_kv));
            coli_bf16_round_array(item_kv, nope);
        }
        free(scales); free(qdq);
    }

    const float *sinks = layer_data(weights, "attn.attn_sink", NULL);
    for (int item = 0; !result && item < batch; item++) {
        int position = start_position + item;
        float *item_kv = kv + (size_t)item * head_dim;
        float *item_q = q + (size_t)item * q_width;
        float *item_attended = attended + (size_t)item * q_width;
        int slot = position % state->window_size;
        memcpy(state->kv + (size_t)slot * head_dim, item_kv,
               (size_t)head_dim * sizeof(*item_kv));
        int selected = selected_counts[item];
        int compressed_count = compressed_counts[item];
        int topk = state->window_size + selected;
        int kv_count = state->window_size + compressed_count;
        int *indices = malloc((size_t)topk * sizeof(*indices));
        float *all_kv = compressed_count
            ? malloc((size_t)kv_count * head_dim * sizeof(*all_kv)) : NULL;
        if (!indices || (compressed_count && !all_kv)) result = -1;
        if (!result) {
            if (position < state->window_size - 1)
                for (int i = 0; i < state->window_size; i++)
                    indices[i] = i <= position ? i : -1;
            else {
                int oldest = (position + 1) % state->window_size;
                for (int i = 0; i < state->window_size; i++)
                    indices[i] = (oldest + i) % state->window_size;
            }
            const float *values = state->kv;
            if (compressed_count) {
                memcpy(all_kv, state->kv,
                       (size_t)state->window_size * head_dim * sizeof(*all_kv));
                memcpy(all_kv + (size_t)state->window_size * head_dim,
                       state->compressed,
                       (size_t)compressed_count * head_dim * sizeof(*all_kv));
                values = all_kv;
            }
            for (int i = 0; i < selected; i++) {
                int ordinal = state->indexer
                    ? compressed_indices[(size_t)item * config->index_topk + i] : i;
                indices[state->window_size + i] = state->window_size + ordinal;
            }
            result = coli_v4_sparse_attention_ref(
                item_attended, item_q, values, sinks, indices, heads, head_dim,
                kv_count, topk, 1.0f / sqrtf((float)head_dim));
        }
        free(all_kv); free(indices);
        const float *item_cos = cosines + (size_t)item * rope_pairs;
        const float *item_sin = sines + (size_t)item * rope_pairs;
        for (int head = 0; !result && head < heads; head++) {
            float *rope = item_attended + (size_t)head * head_dim +
                          head_dim - rope_dim;
            coli_v4_rope_apply(rope, 1, rope_dim, item_cos, item_sin, 1);
            coli_bf16_round_array(rope, (size_t)rope_dim);
        }
    }

    int heads_per_group = heads / groups;
    int group_width = heads_per_group * head_dim;
    int scale_columns = (group_width + 127) / 128;
    int scale_rows = (o_rank + 127) / 128;
    float *group_inputs = malloc((size_t)batch * group_width * sizeof(*group_inputs));
    float *group_outputs = malloc((size_t)batch * o_rank * sizeof(*group_outputs));
    if (!group_inputs || !group_outputs) result = -1;
    for (int group = 0; !result && group < groups; group++) {
        for (int item = 0; item < batch; item++)
            memcpy(group_inputs + (size_t)item * group_width,
                   attended + (size_t)item * q_width + (size_t)group * group_width,
                   (size_t)group_width * sizeof(*group_inputs));
        ColiTensorView group_view = wo_a;
        group_view.rows = o_rank;
        group_view.columns = group_width;
        group_view.data = (const uint8_t *)wo_a.data +
                          (size_t)group * o_rank * group_width;
        group_view.scales = (const uint8_t *)wo_a.scales +
                            (size_t)group * scale_rows * scale_columns * sizeof(float);
        group_view.data_bytes = (size_t)o_rank * group_width;
        group_view.scale_bytes =
            (size_t)scale_rows * scale_columns * sizeof(float);
        result = coli_fp8_matmul_batch_ref(
            group_outputs, &group_view, group_inputs, batch);
        for (int item = 0; !result && item < batch; item++)
            memcpy(oa + (size_t)item * oa_width + (size_t)group * o_rank,
                   group_outputs + (size_t)item * o_rank,
                   (size_t)o_rank * sizeof(*oa));
    }
    if (!result) coli_bf16_round_array(oa, (size_t)batch * oa_width);
    if (!result) result = coli_fp8_matmul_batch_ref(outputs, &wo_b, oa, batch);
    if (!result) coli_bf16_round_array(outputs, (size_t)batch * hidden);

    free(group_outputs); free(group_inputs);
    free(sines); free(cosines);
    free(compressed_indices); free(compressed_counts); free(selected_counts);
    free(norm); free(oa); free(attended); free(kv); free(q); free(qa);
    return result ? set_error(error, error_size, "batched attention failed") : 0;
}
#endif /* COLI_V4_UNIT_ATTENTION_BATCH */

#ifdef COLI_V4_UNIT_COMPRESSOR
/* ######## deepseek_v4_compressor.c ######## */
#include "deepseek_v4_internal.h"

#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "deepseek_v4_internal.h"
#include "native_quant.h"

struct ColiDeepSeekV4CompressorState {
    const ColiDeepSeekV4LayerWeights *weights;
    const ColiDeepSeekV4Config *config;
    int ratio;
    int layer;
    int hidden;
    int head_dim;
    int projection_dim;
    int state_rows;
    int rope_dim;
    int rotate_fp4;
    char prefix[96];
    float *kv_state;
    float *score_state;
};

static int set_error(char *error, size_t size, const char *format, ...) {
    if (error && size) {
        va_list arguments;
        va_start(arguments, format);
        vsnprintf(error, size, format, arguments);
        va_end(arguments);
    }
    return -1;
}

static const void *layer_value(const ColiDeepSeekV4LayerWeights *weights,
                               const char *suffix) {
    char name[COLI_V4_MAX_TENSOR_NAME];
    snprintf(name, sizeof(name), "layers.%d.%s", weights->plan.layer, suffix);
    return coli_v4_layer_data(weights, name, NULL);
}

int coli_v4_compressor_create(ColiDeepSeekV4CompressorState **output,
                              const ColiDeepSeekV4LayerWeights *weights,
                              const ColiDeepSeekV4Config *config,
                              char *error, size_t error_size) {
    ColiDeepSeekV4CompressorOptions options = {
        "attn.compressor", config ? config->head_dim : 0, 0
    };
    return coli_v4_compressor_create_with_options(
        output, weights, config, &options, error, error_size);
}

int coli_v4_compressor_create_with_options(
    ColiDeepSeekV4CompressorState **output,
    const ColiDeepSeekV4LayerWeights *weights,
    const ColiDeepSeekV4Config *config,
    const ColiDeepSeekV4CompressorOptions *options,
    char *error, size_t error_size) {
    if (!output || !weights || !config || !options || !options->prefix ||
        !options->prefix[0] || options->head_dimension <= 0 ||
        weights->plan.compression_ratio < 1)
        return set_error(error, error_size, "unsupported compressor ratio");
    if (strlen(options->prefix) >= sizeof(((ColiDeepSeekV4CompressorState *)0)->prefix))
        return set_error(error, error_size, "compressor prefix is too long");
    *output = NULL;
    ColiDeepSeekV4CompressorState *state = calloc(1, sizeof(*state));
    if (!state) return set_error(error, error_size, "out of memory creating compressor");
    state->weights = weights;
    state->config = config;
    state->ratio = weights->plan.compression_ratio;
    state->layer = weights->plan.layer;
    state->hidden = config->hidden_size;
    state->head_dim = options->head_dimension;
    state->rotate_fp4 = options->rotate_fp4 != 0;
    memcpy(state->prefix, options->prefix, strlen(options->prefix) + 1);
    int overlap = state->ratio == 4;
    state->projection_dim = (1 + overlap) * state->head_dim;
    state->state_rows = (1 + overlap) * state->ratio;
    state->rope_dim = config->qk_rope_head_dim;
    size_t count = (size_t)state->state_rows * state->projection_dim;
    state->kv_state = calloc(count, sizeof(*state->kv_state));
    state->score_state = malloc(count * sizeof(*state->score_state));
    if (!state->kv_state || !state->score_state) {
        coli_v4_compressor_destroy(state);
        return set_error(error, error_size, "out of memory allocating compressor state");
    }
    for (size_t i = 0; i < count; i++) state->score_state[i] = -INFINITY;
    *output = state;
    return 0;
}

int coli_v4_compressor_bind_weights(ColiDeepSeekV4CompressorState *state,
                                    const ColiDeepSeekV4LayerWeights *weights,
                                    char *error, size_t error_size) {
    if (!state || !weights ||
        weights->plan.layer != state->layer ||
        weights->plan.compression_ratio != state->ratio)
        return set_error(error, error_size, "incompatible compressor weights");
    state->weights = weights;
    return 0;
}

void coli_v4_compressor_reset(ColiDeepSeekV4CompressorState *state) {
    if (!state) return;
    size_t count = (size_t)state->state_rows * state->projection_dim;
    memset(state->kv_state, 0, count * sizeof(*state->kv_state));
    for (size_t i = 0; i < count; i++) state->score_state[i] = -INFINITY;
}

void coli_v4_compressor_destroy(ColiDeepSeekV4CompressorState *state) {
    if (!state) return;
    free(state->score_state);
    free(state->kv_state);
    free(state);
}

int coli_v4_compressor_step(ColiDeepSeekV4CompressorState *state,
                            float *output, int *produced,
                            const float *input, int position,
                            char *error, size_t error_size) {
    if (!state || !produced || !input || position < 0)
        return set_error(error, error_size, "invalid compressor step arguments");
    *produced = 0;
    int slot = position % state->ratio;
    int hidden = state->hidden, dimension = state->head_dim;
    int projection = state->projection_dim;
    int state_row = state->ratio == 4 ? state->ratio + slot : slot;
    char suffix[128];
    snprintf(suffix, sizeof(suffix), "%s.wkv.weight", state->prefix);
    const uint16_t *wkv = layer_value(state->weights, suffix);
    snprintf(suffix, sizeof(suffix), "%s.wgate.weight", state->prefix);
    const uint16_t *wgate = layer_value(state->weights, suffix);
    snprintf(suffix, sizeof(suffix), "%s.ape", state->prefix);
    const float *ape = layer_value(state->weights, suffix);
    if (!wkv || !wgate || !ape)
        return set_error(error, error_size, "missing compressor tensor for %s", state->prefix);
    float *kv_row = state->kv_state + (size_t)state_row * projection;
    float *score_row = state->score_state + (size_t)state_row * projection;
    #pragma omp parallel for
    for (int row = 0; row < projection; row++) {
        float kv_sum = 0.0f, gate_sum = 0.0f;
        const uint16_t *kv_weight = wkv + (size_t)row * hidden;
        const uint16_t *gate_weight = wgate + (size_t)row * hidden;
        for (int column = 0; column < hidden; column++) {
            float value = input[column];
            kv_sum += coli_bf16_decode(kv_weight[column]) * value;
            gate_sum += coli_bf16_decode(gate_weight[column]) * value;
        }
        kv_row[row] = kv_sum;
        score_row[row] = gate_sum + ape[(size_t)slot * projection + row];
    }
    if ((position + 1) % state->ratio != 0) return 0;
    if (!output) return set_error(error, error_size, "compressor output is required");

    #pragma omp parallel for
    for (int column = 0; column < dimension; column++) {
        float maximum = -INFINITY;
        int pool_rows = state->ratio == 4 ? 2 * state->ratio : state->ratio;
        for (int row = 0; row < pool_rows; row++) {
            int source_row = row;
            int source_column = column;
            if (state->ratio == 4 && row >= state->ratio)
                source_column += dimension;
            float score = state->score_state[
                (size_t)source_row * projection + source_column];
            if (score > maximum) maximum = score;
        }
        float total = 0.0f, weighted = 0.0f;
        for (int row = 0; row < pool_rows; row++) {
            int source_column = column;
            if (state->ratio == 4 && row >= state->ratio)
                source_column += dimension;
            float weight = expf(state->score_state[
                (size_t)row * projection + source_column] - maximum);
            total += weight;
            weighted += state->kv_state[
                (size_t)row * projection + source_column] * weight;
        }
        output[column] = weighted / total;
    }
    if (state->ratio == 4) {
        memcpy(state->kv_state,
               state->kv_state + (size_t)state->ratio * projection,
               (size_t)state->ratio * projection * sizeof(*state->kv_state));
        memcpy(state->score_state,
               state->score_state + (size_t)state->ratio * projection,
               (size_t)state->ratio * projection * sizeof(*state->score_state));
    }
    coli_bf16_round_array(output, (size_t)dimension);
    snprintf(suffix, sizeof(suffix), "%s.norm.weight", state->prefix);
    const uint16_t *raw_norm = layer_value(state->weights, suffix);
    float *norm = malloc((size_t)dimension * sizeof(*norm));
    if (!raw_norm || !norm) {
        free(norm);
        return set_error(error, error_size, "missing compressor norm");
    }
    for (int i = 0; i < dimension; i++) norm[i] = coli_bf16_decode(raw_norm[i]);
    coli_v4_rmsnorm(output, output, norm, dimension, state->config->rms_norm_eps);
    coli_bf16_round_array(output, (size_t)dimension);
    free(norm);

    int rope_position = position + 1 - state->ratio;
    int pairs = state->rope_dim / 2;
    float *cosines = malloc((size_t)pairs * sizeof(*cosines));
    float *sines = malloc((size_t)pairs * sizeof(*sines));
    if (!cosines || !sines || coli_v4_rope_position(
            cosines, sines, state->rope_dim, rope_position,
            state->config->original_max_position_embeddings,
            state->config->compress_rope_theta, state->config->rope_factor,
            state->config->rope_beta_fast, state->config->rope_beta_slow)) {
        free(sines); free(cosines);
        return set_error(error, error_size, "cannot create compressor RoPE table");
    }
    float *rope = output + dimension - state->rope_dim;
    coli_v4_rope_apply(rope, 1, state->rope_dim, cosines, sines, 0);
    coli_bf16_round_array(rope, (size_t)state->rope_dim);
    free(sines); free(cosines);

    size_t quantized = state->rotate_fp4
        ? (size_t)dimension : (size_t)(dimension - state->rope_dim);
    size_t block = state->rotate_fp4 ? 32u : 64u;
    float *qdq = malloc(quantized * sizeof(*qdq));
    uint8_t *scales = malloc((quantized + block - 1) / block);
    if (!qdq || !scales) {
        free(scales); free(qdq);
        return set_error(error, error_size, "compressor activation quantization failed");
    }
    int quant_error = 0;
    if (state->rotate_fp4)
        quant_error = coli_hadamard_bf16_ref(output, (size_t)dimension) ||
                      coli_fp4_activation_qdq_ref(qdq, scales, output,
                                                  quantized, block);
    else
        quant_error = coli_fp8_activation_qdq_ref(qdq, scales, output,
                                                  quantized, block);
    if (quant_error) {
        free(scales); free(qdq);
        return set_error(error, error_size, "compressor activation quantization failed");
    }
    memcpy(output, qdq, quantized * sizeof(*output));
    coli_bf16_round_array(output, quantized);
    free(scales); free(qdq);
    *produced = 1;
    return 0;
}
#endif /* COLI_V4_UNIT_COMPRESSOR */

#ifdef COLI_V4_UNIT_INDEXER
/* ######## deepseek_v4_indexer.c ######## */
#include "deepseek_v4_internal.h"

#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "deepseek_v4_internal.h"
#include "deepseek_v4_internal.h"
#include "native_quant.h"

struct ColiDeepSeekV4Indexer {
    const ColiDeepSeekV4LayerWeights *weights;
    const ColiDeepSeekV4Config *config;
    ColiDeepSeekV4CompressorState *compressor;
    int layer;
    int capacity;
    int count;
    float *compressed;
};

typedef struct { float score; int index; } IndexScore;

static int set_error(char *error, size_t size, const char *format, ...) {
    if (error && size) {
        va_list arguments;
        va_start(arguments, format);
        vsnprintf(error, size, format, arguments);
        va_end(arguments);
    }
    return -1;
}

static const void *value(const ColiDeepSeekV4LayerWeights *weights,
                         const char *suffix,
                         const ColiDeepSeekV4TensorSpec **spec) {
    char name[COLI_V4_MAX_TENSOR_NAME];
    snprintf(name, sizeof(name), "layers.%d.%s", weights->plan.layer, suffix);
    return coli_v4_layer_data(weights, name, spec);
}

static int fp8_view(ColiTensorView *view,
                    const ColiDeepSeekV4LayerWeights *weights,
                    const char *prefix) {
    char suffix[128];
    const ColiDeepSeekV4TensorSpec *ws = NULL, *ss = NULL;
    snprintf(suffix, sizeof(suffix), "%s.weight", prefix);
    const void *data = value(weights, suffix, &ws);
    snprintf(suffix, sizeof(suffix), "%s.scale", prefix);
    const void *scales = value(weights, suffix, &ss);
    if (!data || !scales || !ws || !ss || ws->rank != 2 ||
        ws->dtype != COLI_ST_F8_E4M3 || ss->dtype != COLI_ST_F8_E8M0)
        return -1;
    *view = (ColiTensorView){
        COLI_TENSOR_FP8_E4M3_BLOCK, COLI_SCALE_F32, data, scales,
        (size_t)(ws->shape[0] * ws->shape[1]),
        (size_t)(ss->shape[0] * ss->shape[1]) * sizeof(float),
        ws->shape[0], ws->shape[1], ws->packed_rows8 ? 8 : 128, 128
    };
    return 0;
}

static int descending_score(const void *left, const void *right) {
    const IndexScore *a = left, *b = right;
    if (a->score < b->score) return 1;
    if (a->score > b->score) return -1;
    return a->index - b->index;
}

int coli_v4_indexer_create(ColiDeepSeekV4Indexer **output,
                           const ColiDeepSeekV4LayerWeights *weights,
                           const ColiDeepSeekV4Config *config,
                           int max_context, char *error, size_t error_size) {
    if (!output || !weights || !config || !weights->plan.has_indexer ||
        max_context < 4 || config->index_head_dim < 1 ||
        config->index_n_heads < 1)
        return set_error(error, error_size, "invalid indexer options");
    *output = NULL;
    ColiDeepSeekV4Indexer *state = calloc(1, sizeof(*state));
    if (!state) return set_error(error, error_size, "out of memory creating indexer");
    state->weights = weights;
    state->config = config;
    state->layer = weights->plan.layer;
    state->capacity = (max_context + 3) / 4;
    if (state->capacity > 128) state->capacity = 128;
    state->compressed = calloc((size_t)state->capacity * config->index_head_dim,
                               sizeof(*state->compressed));
    ColiDeepSeekV4CompressorOptions options = {
        "attn.indexer.compressor", config->index_head_dim, 1
    };
    if (!state->compressed || coli_v4_compressor_create_with_options(
            &state->compressor, weights, config, &options, error, error_size)) {
        coli_v4_indexer_destroy(state);
        return set_error(error, error_size, "cannot create indexer compressor");
    }
    *output = state;
    return 0;
}

int coli_v4_indexer_bind_weights(ColiDeepSeekV4Indexer *state,
                                 const ColiDeepSeekV4LayerWeights *weights,
                                 char *error, size_t error_size) {
    if (!state || !weights || weights->plan.layer != state->layer ||
        !weights->plan.has_indexer)
        return set_error(error, error_size, "incompatible indexer weights");
    state->weights = weights;
    return coli_v4_compressor_bind_weights(state->compressor, weights,
                                            error, error_size);
}

void coli_v4_indexer_reset(ColiDeepSeekV4Indexer *state) {
    if (!state) return;
    state->count = 0;
    memset(state->compressed, 0,
           (size_t)state->capacity * state->config->index_head_dim * sizeof(float));
    coli_v4_compressor_reset(state->compressor);
}

void coli_v4_indexer_destroy(ColiDeepSeekV4Indexer *state) {
    if (!state) return;
    coli_v4_compressor_destroy(state->compressor);
    free(state->compressed);
    free(state);
}

static int apply_position_rope(float *queries,
                               const ColiDeepSeekV4Config *config,
                               int position) {
    int heads = config->index_n_heads, dimension = config->index_head_dim;
    int rope_dim = config->qk_rope_head_dim, pairs = rope_dim / 2;
    float *cosines = malloc((size_t)pairs * sizeof(*cosines));
    float *sines = malloc((size_t)pairs * sizeof(*sines));
    if (!cosines || !sines || coli_v4_rope_position(
            cosines, sines, rope_dim, position,
            config->original_max_position_embeddings,
            config->compress_rope_theta, config->rope_factor,
            config->rope_beta_fast, config->rope_beta_slow)) {
        free(sines); free(cosines); return -1;
    }
    for (int head = 0; head < heads; head++) {
        float *query = queries + (size_t)head * dimension;
        coli_v4_rope_apply(query + dimension - rope_dim, 1, rope_dim,
                           cosines, sines, 0);
        coli_bf16_round_array(query + dimension - rope_dim, (size_t)rope_dim);
    }
    free(sines); free(cosines);
    return 0;
}

int coli_v4_indexer_step(ColiDeepSeekV4Indexer *state, int *indices,
                         int index_capacity, const float *query_rank,
                         const float *input, int position,
                         char *error, size_t error_size) {
    if (!state || !indices || index_capacity < 1 || !query_rank || !input ||
        position < 0)
        return set_error(error, error_size, "invalid indexer step arguments");
    int dimension = state->config->index_head_dim;
    int heads = state->config->index_n_heads;
    int produced = 0;
    if ((position + 1) % 4 == 0 && state->count >= state->capacity) {
        int next_capacity = state->capacity * 2;
        float *grown = realloc(state->compressed,
            (size_t)next_capacity * dimension * sizeof(*grown));
        if (!grown) return set_error(error, error_size, "cannot grow indexer cache");
        memset(grown + (size_t)state->capacity * dimension, 0,
               (size_t)(next_capacity - state->capacity) * dimension * sizeof(*grown));
        state->compressed = grown;
        state->capacity = next_capacity;
    }
    float *next = state->count < state->capacity
        ? state->compressed + (size_t)state->count * dimension : NULL;
    if (coli_v4_compressor_step(state->compressor, next, &produced, input,
                                position, error, error_size)) return -1;
    if (produced) {
        if (state->count >= state->capacity)
            return set_error(error, error_size, "indexer cache capacity exceeded");
        state->count++;
    }
    if (!state->count) return 0;

    ColiTensorView wq;
    if (fp8_view(&wq, state->weights, "attn.indexer.wq_b"))
        return set_error(error, error_size, "missing indexer query weight");
    float *queries = malloc((size_t)heads * dimension * sizeof(*queries));
    float *head_weights = malloc((size_t)heads * sizeof(*head_weights));
    IndexScore *scores = malloc((size_t)state->count * sizeof(*scores));
    uint8_t *scales = malloc((size_t)dimension / 32);
    float *qdq = malloc((size_t)dimension * sizeof(*qdq));
    const uint16_t *raw_weights = value(
        state->weights, "attn.indexer.weights_proj.weight", NULL);
    if (!queries || !head_weights || !scores || !scales || !qdq || !raw_weights) {
        free(qdq); free(scales); free(scores); free(head_weights); free(queries);
        return set_error(error, error_size, "out of memory scoring indexer");
    }
    int result = coli_fp8_matvec_ref(queries, &wq, query_rank);
    if (!result) coli_bf16_round_array(queries, (size_t)heads * dimension);
    if (!result) result = apply_position_rope(queries, state->config, position);
    for (int head = 0; !result && head < heads; head++) {
        float *query = queries + (size_t)head * dimension;
        result = coli_hadamard_bf16_ref(query, (size_t)dimension);
        if (!result) result = coli_fp4_activation_qdq_ref(
            qdq, scales, query, (size_t)dimension, 32);
        if (!result) {
            memcpy(query, qdq, (size_t)dimension * sizeof(*query));
            coli_bf16_round_array(query, (size_t)dimension);
        }
    }
    float weight_scale = 1.0f / sqrtf((float)(dimension * heads));
    for (int head = 0; !result && head < heads; head++) {
        float sum = 0.0f;
        const uint16_t *row = raw_weights + (size_t)head * state->config->hidden_size;
        for (int column = 0; column < state->config->hidden_size; column++)
            sum += coli_bf16_decode(row[column]) * input[column];
        head_weights[head] = sum * weight_scale;
    }
    for (int candidate = 0; !result && candidate < state->count; candidate++) {
        const float *key = state->compressed + (size_t)candidate * dimension;
        float score = 0.0f;
        for (int head = 0; head < heads; head++) {
            const float *query = queries + (size_t)head * dimension;
            float dot = 0.0f;
            for (int i = 0; i < dimension; i++) dot += query[i] * key[i];
            score += fmaxf(dot, 0.0f) * head_weights[head];
        }
        scores[candidate] = (IndexScore){score, candidate};
    }
    if (!result) qsort(scores, (size_t)state->count, sizeof(*scores), descending_score);
    int selected = state->count;
    if (selected > state->config->index_topk) selected = state->config->index_topk;
    if (selected > index_capacity) selected = index_capacity;
    for (int i = 0; !result && i < selected; i++) indices[i] = scores[i].index;
    free(qdq); free(scales); free(scores); free(head_weights); free(queries);
    return result ? set_error(error, error_size, "indexer scoring failed") : selected;
}

const float *coli_v4_indexer_compressed_values(
    const ColiDeepSeekV4Indexer *state) {
    return state ? state->compressed : NULL;
}

int coli_v4_indexer_compressed_count(const ColiDeepSeekV4Indexer *state) {
    return state ? state->count : 0;
}
#endif /* COLI_V4_UNIT_INDEXER */

#ifdef COLI_V4_UNIT_SPARSE_ATTENTION
/* ######## deepseek_v4_sparse_attention.c ######## */
#include "deepseek_v4_internal.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "native_quant.h"

int coli_v4_sparse_attention_ref(float *output, const float *queries,
                                 const float *kv, const float *sinks,
                                 const int *indices, int heads,
                                 int head_dimension, int kv_count, int topk,
                                 float softmax_scale) {
    if (!output || !queries || !kv || !sinks || !indices || heads < 1 ||
        head_dimension < 1 || kv_count < 1 || topk < 1 || !(softmax_scale > 0.0f))
        return -1;
    float *scores = malloc((size_t)topk * sizeof(*scores));
    if (!scores) return -1;
    for (int head = 0; head < heads; head++) {
        const float *query = queries + (size_t)head * head_dimension;
        float maximum = -INFINITY;
        for (int rank = 0; rank < topk; rank++) {
            int index = indices[rank];
            if (index < 0) {
                scores[rank] = -INFINITY;
                continue;
            }
            if (index >= kv_count) {
                free(scores);
                return -1;
            }
            const float *key = kv + (size_t)index * head_dimension;
            float score = 0.0f;
            for (int column = 0; column < head_dimension; column++)
                score += query[column] * key[column];
            score *= softmax_scale;
            scores[rank] = score;
            if (score > maximum) maximum = score;
        }
        if (!isfinite(maximum)) {
            free(scores);
            return -1;
        }
        float denominator = expf(sinks[head] - maximum);
        float *head_output = output + (size_t)head * head_dimension;
        memset(head_output, 0, (size_t)head_dimension * sizeof(*head_output));
        for (int rank = 0; rank < topk; rank++) {
            if (indices[rank] < 0) continue;
            float probability = expf(scores[rank] - maximum);
            denominator += probability;
            /* TileLang casts the exp fragment to BF16 before value GEMM. */
            probability = coli_bf16_round(probability);
            const float *value = kv + (size_t)indices[rank] * head_dimension;
            for (int column = 0; column < head_dimension; column++)
                head_output[column] += probability * value[column];
        }
        for (int column = 0; column < head_dimension; column++)
            head_output[column] = coli_bf16_round(head_output[column] / denominator);
    }
    free(scores);
    return 0;
}
#endif /* COLI_V4_UNIT_SPARSE_ATTENTION */

#ifdef COLI_V4_UNIT_BLOCK_HYBRID
/* ######## deepseek_v4_block_hybrid.c ######## */
/* Accepted decode pipeline plus batched causal attention for prompt prefill. */
/* ---- begin include deepseek_v4_block_pipeline.c ---- */
/* Windows-native routed-expert I/O pipeline. The original block implementation
 * is retained under serial symbols. Public entry points normally keep three
 * persistent lookup workers so reads N+1..N+3 overlap ordered expert compute.
 * COLI_V4_DISABLE_DUAL_EXPERT_LOADER restores the one-worker pipeline. */
#if !defined(COLI_V4_DISABLE_DUAL_EXPERT_LOADER) && \
    !defined(COLI_V4_DISABLE_PERSISTENT_EXPERT_LOADER) && \
    !defined(COLI_V4_EXPERIMENTAL_SYNC_EXPERT_LOOKUP) && \
    !defined(COLI_V4_EXPERIMENTAL_DUAL_EXPERT_LOADER)
#define COLI_V4_EXPERIMENTAL_DUAL_EXPERT_LOADER
#endif
#define coli_v4_block_token_ref coli_v4_block_token_serial_ref
#define coli_v4_block_window_token_ref coli_v4_block_window_token_serial_ref
/* ---- begin include deepseek_v4_block.c ---- */
#include "deepseek_v4_internal.h"

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "deepseek_v4_internal.h"
#include "deepseek_v4_internal.h"
#include "deepseek_v4_internal.h"
#include "native_quant.h"

static int set_error(char *error, size_t size, const char *format, ...) {
    if (error && size) {
        va_list arguments;
        va_start(arguments, format);
        vsnprintf(error, size, format, arguments);
        va_end(arguments);
    }
    return -1;
}

static const void *value(const ColiDeepSeekV4LayerWeights *weights,
                         const char *suffix,
                         const ColiDeepSeekV4TensorSpec **spec) {
    char name[COLI_V4_MAX_TENSOR_NAME];
    snprintf(name, sizeof(name), "layers.%d.%s", weights->plan.layer, suffix);
    return coli_v4_layer_data(weights, name, spec);
}

static int fp8_view(ColiTensorView *view,
                    const ColiDeepSeekV4LayerWeights *weights,
                    const char *prefix) {
    char name[128];
    const ColiDeepSeekV4TensorSpec *ws = NULL, *ss = NULL;
    snprintf(name, sizeof(name), "%s.weight", prefix);
    const void *data = value(weights, name, &ws);
    snprintf(name, sizeof(name), "%s.scale", prefix);
    const void *scales = value(weights, name, &ss);
    if (!data || !scales || !ws || !ss || ws->rank != 2) return -1;
    *view = (ColiTensorView){
        COLI_TENSOR_FP8_E4M3_BLOCK, COLI_SCALE_F32, data, scales,
        (size_t)(ws->shape[0] * ws->shape[1]),
        (size_t)(ss->shape[0] * ss->shape[1]) * sizeof(float),
        ws->shape[0], ws->shape[1], ws->packed_rows8 ? 8 : 128, 128
    };
    return 0;
}

static void decode_bf16(float *output, const uint16_t *input, size_t count) {
    for (size_t i = 0; i < count; i++) output[i] = coli_bf16_decode(input[i]);
}

static int normalized_hc_pre(float *reduced, float *post, float *comb,
                             float *normalized, const float *input_hc,
                             const ColiDeepSeekV4LayerWeights *weights,
                             const ColiDeepSeekV4Config *config,
                             const char *branch, const char *norm_name) {
    char name[64];
    snprintf(name, sizeof(name), "hc_%s_fn", branch);
    const float *function = value(weights, name, NULL);
    snprintf(name, sizeof(name), "hc_%s_scale", branch);
    const float *scale = value(weights, name, NULL);
    snprintf(name, sizeof(name), "hc_%s_base", branch);
    const float *base = value(weights, name, NULL);
    const uint16_t *raw_norm = value(weights, norm_name, NULL);
    int d = config->hidden_size;
    float *norm = malloc((size_t)d * sizeof(*norm));
    if (!function || !scale || !base || !raw_norm || !norm) {
        free(norm);
        return -1;
    }
    decode_bf16(norm, raw_norm, (size_t)d);
    int result = coli_v4_hc_pre(reduced, post, comb, input_hc, function,
                                scale, base, config->hc_mult, d,
                                config->hc_sinkhorn_iters,
                                config->rms_norm_eps, config->hc_eps);
    if (!result) {
        coli_bf16_round_array(reduced, (size_t)d);
        result = coli_v4_rmsnorm(normalized, reduced, norm, d,
                                 config->rms_norm_eps);
        coli_bf16_round_array(normalized, (size_t)d);
    }
    free(norm);
    return result;
}

static int moe_token(float *output,
                     const ColiDeepSeekV4LayerWeights *weights,
                     const ColiDeepSeekV4Config *config,
                     ColiExpertStore *store, const float *input, int token) {
    int d = config->hidden_size;
    int n = config->n_routed_experts;
    int topk = config->num_experts_per_tok;
    size_t gate_count = (size_t)n * d;
    float *gate = malloc(gate_count * sizeof(*gate));
    float *route_weights = malloc((size_t)topk * sizeof(*route_weights));
    int *indices = malloc((size_t)topk * sizeof(*indices));
    float *expert_output = malloc((size_t)d * sizeof(*expert_output));
    float *shared_output = malloc((size_t)d * sizeof(*shared_output));
    if (!gate || !route_weights || !indices || !expert_output || !shared_output) {
        free(shared_output); free(expert_output); free(indices);
        free(route_weights); free(gate);
        return -1;
    }
    decode_bf16(gate, value(weights, "ffn.gate.weight", NULL), gate_count);
    const int64_t *table = value(weights, "ffn.gate.tid2eid", NULL);
    const float *bias = value(weights, "ffn.gate.bias", NULL);
    int result = token < 0 || token >= config->vocab_size;
    if (!result && weights->plan.uses_hash_router) {
        if (!table) result = -1;
    }
    if (!result && weights->plan.uses_hash_router) {
        for (int i = 0; i < topk; i++)
            indices[i] = (int)table[(size_t)token * topk + i];
    }
    if (!result) result = coli_v4_route(
        route_weights, indices, input, gate, bias,
        weights->plan.uses_hash_router ? indices : NULL,
        n, d, topk, config->routed_scaling_factor);

    ColiTensorView w1, w2, w3;
    if (!result && (fp8_view(&w1, weights, "ffn.shared_experts.w1") ||
                    fp8_view(&w2, weights, "ffn.shared_experts.w2") ||
                    fp8_view(&w3, weights, "ffn.shared_experts.w3"))) result = -1;
    if (!result) result = coli_v4_shared_expert_forward_ref(
        shared_output, &w1, &w2, &w3, input, config->swiglu_limit);
    if (!result) memset(output, 0, (size_t)d * sizeof(*output));
    for (int expert_id = 0; !result && expert_id < n; expert_id++) {
        int rank = -1;
        for (int candidate = 0; candidate < topk; candidate++)
            if (indices[candidate] == expert_id) rank = candidate;
        if (rank < 0) continue;
        ColiExpertView expert;
        if (coli_expert_lookup(store,
                               (ColiExpertKey){weights->plan.layer, expert_id},
                               &expert)) {
            result = -1;
            break;
        }
        result = coli_v4_expert_forward_ref(expert_output, &expert, input,
                                             route_weights[rank],
                                             config->swiglu_limit);
        coli_expert_release(store, &expert);
        if (!result)
            for (int i = 0; i < d; i++) output[i] += expert_output[i];
    }
    if (!result)
        for (int i = 0; i < d; i++)
            output[i] = coli_bf16_round(output[i] + shared_output[i]);
    free(shared_output); free(expert_output); free(indices);
    free(route_weights); free(gate);
    return result;
}

static int block_token_impl(float *output_hc,
                            ColiDeepSeekV4WindowAttentionState *attention,
                            const ColiDeepSeekV4LayerWeights *weights,
                            const ColiDeepSeekV4Config *config,
                            ColiExpertStore *experts,
                            const float *input_hc, int token, int position,
                            char *error, size_t error_size) {
    if (!output_hc || !weights || !config || !experts || !input_hc)
        return set_error(error, error_size, "invalid block arguments");
    int d = config->hidden_size, hc = config->hc_mult;
    size_t hd = (size_t)hc * d;
    float *residual = malloc(hd * sizeof(*residual));
    float *state = malloc(hd * sizeof(*state));
    float *reduced = malloc((size_t)d * sizeof(*reduced));
    float *normalized = malloc((size_t)d * sizeof(*normalized));
    float *branch = malloc((size_t)d * sizeof(*branch));
    float *post = malloc((size_t)hc * sizeof(*post));
    float *comb = malloc((size_t)hc * hc * sizeof(*comb));
    if (!residual || !state || !reduced || !normalized || !branch || !post || !comb) {
        free(comb); free(post); free(branch); free(normalized);
        free(reduced); free(state); free(residual);
        return set_error(error, error_size, "out of memory in block");
    }
    memcpy(residual, input_hc, hd * sizeof(*residual));
    int result = normalized_hc_pre(reduced, post, comb, normalized, input_hc,
                                   weights, config, "attn", "attn_norm.weight");
    if (!result) result = attention
        ? coli_v4_attention_window_token_ref(branch, attention, weights, config,
                                             normalized, position, error, error_size)
        : coli_v4_attention_token_ref(branch, weights, config, normalized,
                                      position, error, error_size);
    if (!result) result = coli_v4_hc_post(state, branch, residual, post, comb, hc, d);
    if (!result) coli_bf16_round_array(state, hd);

    if (!result) memcpy(residual, state, hd * sizeof(*residual));
    if (!result) result = normalized_hc_pre(reduced, post, comb, normalized, state,
                                            weights, config, "ffn", "ffn_norm.weight");
    if (!result) result = moe_token(branch, weights, config, experts, normalized, token);
    if (!result) result = coli_v4_hc_post(output_hc, branch, residual, post, comb, hc, d);
    if (!result) coli_bf16_round_array(output_hc, hd);

    free(comb); free(post); free(branch); free(normalized);
    free(reduced); free(state); free(residual);
    return result ? set_error(error, error_size, "block computation failed") : 0;
}

int coli_v4_block_token_ref(float *output_hc,
                            const ColiDeepSeekV4LayerWeights *weights,
                            const ColiDeepSeekV4Config *config,
                            ColiExpertStore *experts,
                            const float *input_hc, int token, int position,
                            char *error, size_t error_size) {
    return block_token_impl(output_hc, NULL, weights, config, experts, input_hc,
                            token, position, error, error_size);
}

int coli_v4_block_window_token_ref(
    float *output_hc, ColiDeepSeekV4WindowAttentionState *attention,
    const ColiDeepSeekV4LayerWeights *weights,
    const ColiDeepSeekV4Config *config, ColiExpertStore *experts,
    const float *input_hc, int token, int position,
    char *error, size_t error_size) {
    return block_token_impl(output_hc, attention, weights, config, experts,
                            input_hc, token, position, error, error_size);
}
/* ---- end include deepseek_v4_block.c ---- */

#undef coli_v4_block_token_ref
#undef coli_v4_block_window_token_ref

#ifndef COLI_V4_DISABLE_BF16_ROUTE
int coli_v4_route_bf16(float *weights, int *indices, const float *hidden,
                       const uint16_t *gate, const float *bias,
                       const int *forced_indices, int experts, int dimension,
                       int topk, float route_scale);
#endif

typedef struct {
    ColiExpertStore *store;
    ColiExpertKey key;
    ColiExpertView view;
    int result;
} ExpertLoadJob;

#ifdef COLI_V4_EXPERIMENTAL_PREFETCH
static int expert_prefetch_enabled(void) {
    static int enabled = -1;
    if (enabled < 0) {
        const char *text = getenv("COLI_V4_EXPERT_PREFETCH");
        enabled = text && *text && atoi(text) != 0;
    }
    return enabled;
}
#endif


static void *expert_load_worker(void *argument) {
    ExpertLoadJob *job = argument;
    job->result = coli_expert_lookup(job->store, job->key, &job->view);
    return NULL;
}

typedef struct {
    pthread_t thread;
    int active;
#ifdef COLI_V4_EXPERIMENTAL_DUAL_EXPERT_LOADER
    int loader_slot;
#endif
} ExpertLoadHandle;

#ifdef COLI_V4_EXPERIMENTAL_DUAL_EXPERT_LOADER
#ifndef COLI_V4_EXPERT_LOADER_COUNT
#define COLI_V4_EXPERT_LOADER_COUNT 3
#endif
enum { DUAL_EXPERT_LOADER_COUNT = COLI_V4_EXPERT_LOADER_COUNT };

typedef struct {
    pthread_t thread;
    ExpertLoadJob *job;
    int pending;
    int completed;
    int stopping;
    int available;
} DualExpertLoaderSlot;

typedef struct {
    pthread_mutex_t mutex;
    pthread_cond_t ready;
    pthread_cond_t complete;
    pthread_cond_t idle;
    DualExpertLoaderSlot slots[DUAL_EXPERT_LOADER_COUNT];
} DualExpertLoaderPool;

static DualExpertLoaderPool dual_loader_pool = {
    .mutex = PTHREAD_MUTEX_INITIALIZER,
    .ready = PTHREAD_COND_INITIALIZER,
    .complete = PTHREAD_COND_INITIALIZER,
    .idle = PTHREAD_COND_INITIALIZER,
};
static pthread_once_t dual_loader_once = PTHREAD_ONCE_INIT;

static void *dual_expert_loader_worker(void *argument) {
    DualExpertLoaderSlot *slot = argument;
    pthread_mutex_lock(&dual_loader_pool.mutex);
    for (;;) {
        while (!slot->pending && !slot->stopping)
            pthread_cond_wait(&dual_loader_pool.ready,
                              &dual_loader_pool.mutex);
        if (slot->stopping) break;
        ExpertLoadJob *job = slot->job;
        slot->pending = 0;
        pthread_mutex_unlock(&dual_loader_pool.mutex);
        job->result = coli_expert_lookup(job->store, job->key, &job->view);
        pthread_mutex_lock(&dual_loader_pool.mutex);
        slot->completed = 1;
        pthread_cond_broadcast(&dual_loader_pool.complete);
    }
    pthread_mutex_unlock(&dual_loader_pool.mutex);
    return NULL;
}

static void dual_expert_loader_shutdown(void) {
    pthread_mutex_lock(&dual_loader_pool.mutex);
    for (int i = 0; i < DUAL_EXPERT_LOADER_COUNT; i++)
        dual_loader_pool.slots[i].stopping = 1;
    pthread_cond_broadcast(&dual_loader_pool.ready);
    pthread_mutex_unlock(&dual_loader_pool.mutex);
    for (int i = 0; i < DUAL_EXPERT_LOADER_COUNT; i++)
        if (dual_loader_pool.slots[i].available) {
            pthread_join(dual_loader_pool.slots[i].thread, NULL);
            dual_loader_pool.slots[i].available = 0;
        }
}

static void dual_expert_loader_init(void) {
    int available = 0;
    for (int i = 0; i < DUAL_EXPERT_LOADER_COUNT; i++) {
        DualExpertLoaderSlot *slot = &dual_loader_pool.slots[i];
        if (!pthread_create(&slot->thread, NULL,
                            dual_expert_loader_worker, slot)) {
            slot->available = 1;
            available++;
        }
    }
    if (available) atexit(dual_expert_loader_shutdown);
}

static int dual_expert_load_start(ExpertLoadHandle *handle,
                                  ExpertLoadJob *job) {
    pthread_once(&dual_loader_once, dual_expert_loader_init);
    pthread_mutex_lock(&dual_loader_pool.mutex);
    int selected = -1;
    while (selected < 0) {
        int available = 0;
        for (int i = 0; i < DUAL_EXPERT_LOADER_COUNT; i++) {
            DualExpertLoaderSlot *slot = &dual_loader_pool.slots[i];
            if (!slot->available) continue;
            available++;
            if (!slot->job && !slot->pending) { selected = i; break; }
        }
        if (!available ||
            (selected < 0 && available < DUAL_EXPERT_LOADER_COUNT)) {
            pthread_mutex_unlock(&dual_loader_pool.mutex);
            return -1;
        }
        if (selected < 0)
            pthread_cond_wait(&dual_loader_pool.idle,
                              &dual_loader_pool.mutex);
    }
    DualExpertLoaderSlot *slot = &dual_loader_pool.slots[selected];
    slot->job = job;
    slot->completed = 0;
    slot->pending = 1;
    handle->active = 1;
    handle->loader_slot = selected;
    pthread_cond_broadcast(&dual_loader_pool.ready);
    pthread_mutex_unlock(&dual_loader_pool.mutex);
    return 0;
}

static int dual_expert_load_finish(ExpertLoadHandle *handle) {
    if (!handle->active || handle->loader_slot < 0 ||
        handle->loader_slot >= DUAL_EXPERT_LOADER_COUNT) return -1;
    pthread_mutex_lock(&dual_loader_pool.mutex);
    DualExpertLoaderSlot *slot =
        &dual_loader_pool.slots[handle->loader_slot];
    while (!slot->completed)
        pthread_cond_wait(&dual_loader_pool.complete,
                          &dual_loader_pool.mutex);
    slot->job = NULL;
    pthread_cond_broadcast(&dual_loader_pool.idle);
    pthread_mutex_unlock(&dual_loader_pool.mutex);
    handle->active = 0;
    return 0;
}
#endif

#if !defined(COLI_V4_DISABLE_PERSISTENT_EXPERT_LOADER) && \
    !defined(COLI_V4_EXPERIMENTAL_DUAL_EXPERT_LOADER)
typedef struct {
    pthread_mutex_t mutex;
    pthread_cond_t ready;
    pthread_cond_t complete;
    pthread_cond_t idle;
    pthread_t thread;
    ExpertLoadJob *job;
    int pending;
    int completed;
    int stopping;
    int available;
} PersistentExpertLoader;

static PersistentExpertLoader persistent_loader = {
    .mutex = PTHREAD_MUTEX_INITIALIZER,
    .ready = PTHREAD_COND_INITIALIZER,
    .complete = PTHREAD_COND_INITIALIZER,
    .idle = PTHREAD_COND_INITIALIZER,
};
static pthread_once_t persistent_loader_once = PTHREAD_ONCE_INIT;

static void *persistent_expert_loader_worker(void *unused) {
    (void)unused;
    pthread_mutex_lock(&persistent_loader.mutex);
    for (;;) {
        while (!persistent_loader.pending && !persistent_loader.stopping)
            pthread_cond_wait(&persistent_loader.ready,
                              &persistent_loader.mutex);
        if (persistent_loader.stopping) break;
        ExpertLoadJob *job = persistent_loader.job;
        persistent_loader.pending = 0;
        pthread_mutex_unlock(&persistent_loader.mutex);
        job->result = coli_expert_lookup(job->store, job->key, &job->view);
        pthread_mutex_lock(&persistent_loader.mutex);
        persistent_loader.completed = 1;
        pthread_cond_signal(&persistent_loader.complete);
    }
    pthread_mutex_unlock(&persistent_loader.mutex);
    return NULL;
}

static void persistent_expert_loader_shutdown(void) {
    if (!persistent_loader.available) return;
    pthread_mutex_lock(&persistent_loader.mutex);
    persistent_loader.stopping = 1;
    pthread_cond_signal(&persistent_loader.ready);
    pthread_mutex_unlock(&persistent_loader.mutex);
    pthread_join(persistent_loader.thread, NULL);
    persistent_loader.available = 0;
}

static void persistent_expert_loader_init(void) {
    if (!pthread_create(&persistent_loader.thread, NULL,
                        persistent_expert_loader_worker, NULL)) {
        persistent_loader.available = 1;
        atexit(persistent_expert_loader_shutdown);
    }
}
#endif

static int expert_load_start(ExpertLoadHandle *handle, ExpertLoadJob *job) {
#ifdef COLI_V4_EXPERIMENTAL_DUAL_EXPERT_LOADER
    if (!dual_expert_load_start(handle, job)) return 0;
    if (pthread_create(&handle->thread, NULL, expert_load_worker, job))
        return -1;
    handle->loader_slot = -1;
    handle->active = 1;
    return 0;
#elif defined(COLI_V4_EXPERIMENTAL_SYNC_EXPERT_LOOKUP)
    job->result = coli_expert_lookup(job->store, job->key, &job->view);
    handle->active = 1;
    return 0;
#elif !defined(COLI_V4_DISABLE_PERSISTENT_EXPERT_LOADER)
    pthread_once(&persistent_loader_once, persistent_expert_loader_init);
    if (!persistent_loader.available) return -1;
    pthread_mutex_lock(&persistent_loader.mutex);
    while (persistent_loader.job || persistent_loader.pending)
        pthread_cond_wait(&persistent_loader.idle,
                          &persistent_loader.mutex);
    persistent_loader.job = job;
    persistent_loader.completed = 0;
    persistent_loader.pending = 1;
    handle->active = 1;
    pthread_cond_signal(&persistent_loader.ready);
    pthread_mutex_unlock(&persistent_loader.mutex);
    return 0;
#else
    if (pthread_create(&handle->thread, NULL, expert_load_worker, job))
        return -1;
    handle->active = 1;
    return 0;
#endif
}

static int expert_load_finish(ExpertLoadHandle *handle) {
    if (!handle->active) return -1;
#ifdef COLI_V4_EXPERIMENTAL_DUAL_EXPERT_LOADER
    if (handle->loader_slot >= 0)
        return dual_expert_load_finish(handle);
    int result = pthread_join(handle->thread, NULL);
    handle->active = 0;
    return result;
#elif defined(COLI_V4_EXPERIMENTAL_SYNC_EXPERT_LOOKUP)
    handle->active = 0;
    return 0;
#elif !defined(COLI_V4_DISABLE_PERSISTENT_EXPERT_LOADER)
    pthread_mutex_lock(&persistent_loader.mutex);
    while (!persistent_loader.completed)
        pthread_cond_wait(&persistent_loader.complete,
                          &persistent_loader.mutex);
    persistent_loader.job = NULL;
    pthread_cond_broadcast(&persistent_loader.idle);
    pthread_mutex_unlock(&persistent_loader.mutex);
    handle->active = 0;
    return 0;
#else
    int result = pthread_join(handle->thread, NULL);
    handle->active = 0;
    return result;
#endif
}

#ifdef COLI_V4_EXPERIMENTAL_BLOCK_OTHER_PROFILE
enum {
    COLI_V4_BLOCK_PROFILE_MOE_TOTAL = 0,
    COLI_V4_BLOCK_PROFILE_GATE_DECODE = 1,
    COLI_V4_BLOCK_PROFILE_LOADER_START = 2,
    COLI_V4_BLOCK_PROFILE_LOADER_WAIT = 3,
};
double coli_v4_block_profile_now(void);
void coli_v4_block_profile_add(int kind, double seconds);
#endif

static int profiled_expert_load_start(ExpertLoadHandle *handle,
                                      ExpertLoadJob *job) {
#ifdef COLI_V4_EXPERIMENTAL_BLOCK_OTHER_PROFILE
    double began = coli_v4_block_profile_now();
#endif
    int result = expert_load_start(handle, job);
#ifdef COLI_V4_EXPERIMENTAL_BLOCK_OTHER_PROFILE
    coli_v4_block_profile_add(COLI_V4_BLOCK_PROFILE_LOADER_START,
                              coli_v4_block_profile_now() - began);
#endif
    return result;
}

static int profiled_expert_load_finish(ExpertLoadHandle *handle) {
#ifdef COLI_V4_EXPERIMENTAL_BLOCK_OTHER_PROFILE
    double began = coli_v4_block_profile_now();
#endif
    int result = expert_load_finish(handle);
#ifdef COLI_V4_EXPERIMENTAL_BLOCK_OTHER_PROFILE
    coli_v4_block_profile_add(COLI_V4_BLOCK_PROFILE_LOADER_WAIT,
                              coli_v4_block_profile_now() - began);
#endif
    return result;
}

static int moe_token_pipeline(float *output,
                              const ColiDeepSeekV4LayerWeights *weights,
                              const ColiDeepSeekV4Config *config,
                              ColiExpertStore *store,
                              const float *input, int token) {
#ifdef COLI_V4_EXPERIMENTAL_BLOCK_OTHER_PROFILE
    double profile_moe_began = coli_v4_block_profile_now();
#endif
    int d = config->hidden_size;
    int n = config->n_routed_experts;
    int topk = config->num_experts_per_tok;
#ifndef COLI_V4_DISABLE_BF16_ROUTE
    float *gate = NULL;
    const uint16_t *raw_gate = value(weights, "ffn.gate.weight", NULL);
    int missing_gate = !raw_gate;
#else
    size_t gate_count = (size_t)n * d;
    float *gate = malloc(gate_count * sizeof(*gate));
    int missing_gate = !gate;
#endif
    float *route_weights = malloc((size_t)topk * sizeof(*route_weights));
    int *indices = malloc((size_t)topk * sizeof(*indices));
    int *expert_ids = malloc((size_t)topk * sizeof(*expert_ids));
    float *expert_weights = malloc((size_t)topk * sizeof(*expert_weights));
    float *expert_output = malloc((size_t)d * sizeof(*expert_output));
    float *shared_output = malloc((size_t)d * sizeof(*shared_output));
    if (missing_gate || !route_weights || !indices || !expert_ids || !expert_weights ||
        !expert_output || !shared_output) {
        free(shared_output); free(expert_output); free(expert_weights);
        free(expert_ids); free(indices); free(route_weights); free(gate);
        return -1;
    }
#ifdef COLI_V4_DISABLE_BF16_ROUTE
#ifdef COLI_V4_EXPERIMENTAL_BLOCK_OTHER_PROFILE
    double profile_gate_began = coli_v4_block_profile_now();
#endif
    decode_bf16(gate, value(weights, "ffn.gate.weight", NULL), gate_count);
#ifdef COLI_V4_EXPERIMENTAL_BLOCK_OTHER_PROFILE
    coli_v4_block_profile_add(COLI_V4_BLOCK_PROFILE_GATE_DECODE,
                              coli_v4_block_profile_now() - profile_gate_began);
#endif
#endif
    const int64_t *table = value(weights, "ffn.gate.tid2eid", NULL);
    const float *bias = value(weights, "ffn.gate.bias", NULL);
    int result = token < 0 || token >= config->vocab_size;
    if (!result && weights->plan.uses_hash_router && !table) result = -1;
    if (!result && weights->plan.uses_hash_router)
        for (int i = 0; i < topk; i++)
            indices[i] = (int)table[(size_t)token * topk + i];
#ifndef COLI_V4_DISABLE_BF16_ROUTE
    if (!result) result = coli_v4_route_bf16(
        route_weights, indices, input, raw_gate, bias,
        weights->plan.uses_hash_router ? indices : NULL,
        n, d, topk, config->routed_scaling_factor);
#else
    if (!result) result = coli_v4_route(
        route_weights, indices, input, gate, bias,
        weights->plan.uses_hash_router ? indices : NULL,
        n, d, topk, config->routed_scaling_factor);
#endif

    int selected = 0;
    for (int expert_id = 0; !result && expert_id < n; expert_id++) {
        for (int rank = 0; rank < topk; rank++) {
            if (indices[rank] == expert_id) {
                expert_ids[selected] = expert_id;
                expert_weights[selected] = route_weights[rank];
                selected++;
            }
        }
    }
    if (!result && selected != topk) result = -1;

#ifdef COLI_V4_EXPERIMENTAL_PREFETCH
    if (!result && expert_prefetch_enabled() && store->ops->prefetch) {
        ColiExpertKey *keys = malloc((size_t)selected * sizeof(*keys));
        if (keys) {
            for (int i = 0; i < selected; i++)
                keys[i] = (ColiExpertKey){weights->plan.layer, expert_ids[i]};
            store->ops->prefetch(store, keys, (size_t)selected);
            free(keys);
        }
    }
#endif


#ifdef COLI_V4_EXPERIMENTAL_DUAL_EXPERT_LOADER
    ExpertLoadJob jobs[DUAL_EXPERT_LOADER_COUNT] = {{0}};
    ExpertLoadHandle loaders[DUAL_EXPERT_LOADER_COUNT] = {{0}};
    int loader_active[DUAL_EXPERT_LOADER_COUNT] = {0};
    if (!result) {
        int preload = selected < DUAL_EXPERT_LOADER_COUNT
            ? selected : DUAL_EXPERT_LOADER_COUNT;
        for (int i = 0; i < preload; i++) {
            jobs[i].store = store;
            jobs[i].key = (ColiExpertKey){weights->plan.layer, expert_ids[i]};
            jobs[i].result = -1;
            if (profiled_expert_load_start(&loaders[i], &jobs[i]) != 0) {
                result = -1;
                break;
            }
            loader_active[i] = 1;
        }
    }
#else
    ExpertLoadJob job = {0};
    ExpertLoadHandle loader = {0};
    int loader_active = 0;
    if (!result) {
        job.store = store;
        job.key = (ColiExpertKey){weights->plan.layer, expert_ids[0]};
        job.result = -1;
        if (profiled_expert_load_start(&loader, &job) != 0)
            result = -1;
        else
            loader_active = 1;
    }
#endif

    ColiTensorView w1, w2, w3;
    if (!result && (fp8_view(&w1, weights, "ffn.shared_experts.w1") ||
                    fp8_view(&w2, weights, "ffn.shared_experts.w2") ||
                    fp8_view(&w3, weights, "ffn.shared_experts.w3"))) result = -1;
    if (!result) result = coli_v4_shared_expert_forward_ref(
        shared_output, &w1, &w2, &w3, input, config->swiglu_limit);
    if (!result) memset(output, 0, (size_t)d * sizeof(*output));

#ifdef COLI_V4_EXPERIMENTAL_DUAL_EXPERT_LOADER
    for (int current = 0; !result && current < selected; current++) {
        int slot = current % DUAL_EXPERT_LOADER_COUNT;
        if (!loader_active[slot] ||
            profiled_expert_load_finish(&loaders[slot]) != 0) {
            result = -1; break;
        }
        loader_active[slot] = 0;
        if (jobs[slot].result) { result = -1; break; }
        ColiExpertView expert = jobs[slot].view;

        int next = current + DUAL_EXPERT_LOADER_COUNT;
        if (next < selected) {
            memset(&jobs[slot], 0, sizeof(jobs[slot]));
            jobs[slot].store = store;
            jobs[slot].key = (ColiExpertKey){weights->plan.layer,
                                            expert_ids[next]};
            jobs[slot].result = -1;
            if (profiled_expert_load_start(&loaders[slot],
                                           &jobs[slot]) != 0)
                result = -1;
            else
                loader_active[slot] = 1;
        }
        if (!result) result = coli_v4_expert_forward_ref(
            expert_output, &expert, input, expert_weights[current],
            config->swiglu_limit);
        coli_expert_release(store, &expert);
        if (!result)
            for (int i = 0; i < d; i++) output[i] += expert_output[i];
    }
    for (int slot = 0; slot < DUAL_EXPERT_LOADER_COUNT; slot++)
        if (loader_active[slot]) {
            profiled_expert_load_finish(&loaders[slot]);
            if (!jobs[slot].result)
                coli_expert_release(store, &jobs[slot].view);
        }
#else
    for (int current = 0; current < selected && loader_active; current++) {
        if (profiled_expert_load_finish(&loader) != 0) {
            result = -1; loader_active = 0; break;
        }
        loader_active = 0;
        if (job.result) { result = -1; break; }
        ColiExpertView expert = job.view;

        if (current + 1 < selected) {
            memset(&job, 0, sizeof(job));
            job.store = store;
            job.key = (ColiExpertKey){weights->plan.layer,
                                     expert_ids[current + 1]};
            job.result = -1;
            if (profiled_expert_load_start(&loader, &job) != 0)
                result = -1;
            else
                loader_active = 1;
        }
        if (!result) result = coli_v4_expert_forward_ref(
            expert_output, &expert, input, expert_weights[current],
            config->swiglu_limit);
        coli_expert_release(store, &expert);
        if (!result)
            for (int i = 0; i < d; i++) output[i] += expert_output[i];
    }
    if (loader_active) {
        profiled_expert_load_finish(&loader);
        if (!job.result) coli_expert_release(store, &job.view);
    }
#endif
    if (!result)
        for (int i = 0; i < d; i++)
            output[i] = coli_bf16_round(output[i] + shared_output[i]);

    free(shared_output); free(expert_output); free(expert_weights);
    free(expert_ids); free(indices); free(route_weights); free(gate);
#ifdef COLI_V4_EXPERIMENTAL_BLOCK_OTHER_PROFILE
    coli_v4_block_profile_add(COLI_V4_BLOCK_PROFILE_MOE_TOTAL,
                              coli_v4_block_profile_now() - profile_moe_began);
#endif
    return result;
}

static int block_token_pipeline(float *output_hc,
                                ColiDeepSeekV4WindowAttentionState *attention,
                                const ColiDeepSeekV4LayerWeights *weights,
                                const ColiDeepSeekV4Config *config,
                                ColiExpertStore *experts,
                                const float *input_hc, int token, int position,
                                char *error, size_t error_size) {
    if (!output_hc || !weights || !config || !experts || !input_hc)
        return set_error(error, error_size, "invalid block arguments");
    int d = config->hidden_size, hc = config->hc_mult;
    size_t hd = (size_t)hc * d;
    float *residual = malloc(hd * sizeof(*residual));
    float *state = malloc(hd * sizeof(*state));
    float *reduced = malloc((size_t)d * sizeof(*reduced));
    float *normalized = malloc((size_t)d * sizeof(*normalized));
    float *branch = malloc((size_t)d * sizeof(*branch));
    float *post = malloc((size_t)hc * sizeof(*post));
    float *comb = malloc((size_t)hc * hc * sizeof(*comb));
    if (!residual || !state || !reduced || !normalized || !branch ||
        !post || !comb) {
        free(comb); free(post); free(branch); free(normalized);
        free(reduced); free(state); free(residual); return -1;
    }
    memcpy(residual, input_hc, hd * sizeof(*residual));
    int result = normalized_hc_pre(reduced, post, comb, normalized, input_hc,
                                   weights, config, "attn", "attn_norm.weight");
    if (!result) result = attention
        ? coli_v4_attention_window_token_ref(branch, attention, weights, config,
                                             normalized, position, error, error_size)
        : coli_v4_attention_token_ref(branch, weights, config, normalized,
                                      position, error, error_size);
    if (!result) result = coli_v4_hc_post(state, branch, residual,
                                          post, comb, hc, d);
    if (!result) coli_bf16_round_array(state, hd);
    if (!result) memcpy(residual, state, hd * sizeof(*residual));
    if (!result) result = normalized_hc_pre(reduced, post, comb, normalized, state,
                                            weights, config, "ffn",
                                            "ffn_norm.weight");
    if (!result) result = moe_token_pipeline(branch, weights, config, experts,
                                             normalized, token);
    if (!result) result = coli_v4_hc_post(output_hc, branch, residual,
                                          post, comb, hc, d);
    if (!result) coli_bf16_round_array(output_hc, hd);
    free(comb); free(post); free(branch); free(normalized);
    free(reduced); free(state); free(residual);
    return result ? set_error(error, error_size, "block computation failed") : 0;
}

int coli_v4_block_token_ref(float *output_hc,
                            const ColiDeepSeekV4LayerWeights *weights,
                            const ColiDeepSeekV4Config *config,
                            ColiExpertStore *experts,
                            const float *input_hc, int token, int position,
                            char *error, size_t error_size) {
    return block_token_pipeline(output_hc, NULL, weights, config, experts,
                                input_hc, token, position, error, error_size);
}

int coli_v4_block_window_token_ref(
    float *output_hc, ColiDeepSeekV4WindowAttentionState *attention,
    const ColiDeepSeekV4LayerWeights *weights,
    const ColiDeepSeekV4Config *config, ColiExpertStore *experts,
    const float *input_hc, int token, int position,
    char *error, size_t error_size) {
    return block_token_pipeline(output_hc, attention, weights, config, experts,
                                input_hc, token, position, error, error_size);
}
/* ---- end include deepseek_v4_block_pipeline.c ---- */


#include "deepseek_v4_internal.h"
#include "deepseek_v4_internal.h"

int coli_v4_block_window_batch_ref(
    float *outputs_hc, ColiDeepSeekV4WindowAttentionState *attention,
    const ColiDeepSeekV4LayerWeights *weights,
    const ColiDeepSeekV4Config *config, ColiExpertStore *experts,
    const float *inputs_hc, const int *tokens, int start_position, int batch,
    char *error, size_t error_size) {
    if (!outputs_hc || !attention || !weights || !config || !experts ||
        !inputs_hc || !tokens || batch < 1 || batch > 64) return -1;
    int d = config->hidden_size, hc = config->hc_mult;
    size_t hd = (size_t)hc * d;
    float *states = malloc((size_t)batch * hd * sizeof(*states));
    float *normalized = malloc((size_t)batch * d * sizeof(*normalized));
    float *branches = malloc((size_t)batch * d * sizeof(*branches));
    float *posts = malloc((size_t)batch * hc * sizeof(*posts));
    float *combs = malloc((size_t)batch * hc * hc * sizeof(*combs));
    float *reduced = malloc((size_t)d * sizeof(*reduced));
    float *ffn_normalized = malloc((size_t)d * sizeof(*ffn_normalized));
    float *ffn_branch = malloc((size_t)d * sizeof(*ffn_branch));
    float *ffn_post = malloc((size_t)hc * sizeof(*ffn_post));
    float *ffn_comb = malloc((size_t)hc * hc * sizeof(*ffn_comb));
    if (!states || !normalized || !branches || !posts || !combs || !reduced ||
        !ffn_normalized || !ffn_branch || !ffn_post || !ffn_comb) {
        free(ffn_comb); free(ffn_post); free(ffn_branch); free(ffn_normalized);
        free(reduced); free(combs); free(posts); free(branches);
        free(normalized); free(states); return -1;
    }
    int result = 0;
    const char *phase = "attention hyper-connection";
    for (int item = 0; !result && item < batch; item++)
        result = normalized_hc_pre(
            reduced, posts + (size_t)item * hc,
            combs + (size_t)item * hc * hc,
            normalized + (size_t)item * d,
            inputs_hc + (size_t)item * hd,
            weights, config, "attn", "attn_norm.weight");
    phase = "attention";
    if (!result) result = coli_v4_attention_window_batch_ref(
        branches, attention, weights, config, normalized,
        start_position, batch, error, error_size);
    if (!result) phase = "attention post / FFN hyper-connection";
    for (int item = 0; !result && item < batch; item++) {
        float *state = states + (size_t)item * hd;
        result = coli_v4_hc_post(
            state, branches + (size_t)item * d,
            inputs_hc + (size_t)item * hd,
            posts + (size_t)item * hc,
            combs + (size_t)item * hc * hc, hc, d);
        if (!result) coli_bf16_round_array(state, hd);
        if (!result) phase = "FFN hyper-connection";
        if (!result) result = normalized_hc_pre(
            reduced, ffn_post, ffn_comb, ffn_normalized, state,
            weights, config, "ffn", "ffn_norm.weight");
        if (!result) phase = "MoE";
        if (!result) result = moe_token_pipeline(
            ffn_branch, weights, config, experts,
            ffn_normalized, tokens[item]);
        if (!result) result = coli_v4_hc_post(
            outputs_hc + (size_t)item * hd, ffn_branch, state,
            ffn_post, ffn_comb, hc, d);
        if (!result) coli_bf16_round_array(
            outputs_hc + (size_t)item * hd, hd);
    }
    free(ffn_comb); free(ffn_post); free(ffn_branch); free(ffn_normalized);
    free(reduced); free(combs); free(posts); free(branches);
    free(normalized); free(states);
    if (!result) return 0;
    if (error && error_size && error[0]) return -1;
    return set_error(error, error_size, "hybrid batched block failed in %s", phase);
}
#endif /* COLI_V4_UNIT_BLOCK_HYBRID */

#ifdef COLI_V4_UNIT_COMPRESSOR_SNAPSHOT
/* ######## deepseek_v4_compressor_snapshot.c ######## */
#define coli_v4_compressor_create snapshot_copy_compressor_create
#define coli_v4_compressor_create_with_options snapshot_copy_compressor_create_with_options
#define coli_v4_compressor_reset snapshot_copy_compressor_reset
#define coli_v4_compressor_bind_weights snapshot_copy_compressor_bind_weights
#define coli_v4_compressor_destroy snapshot_copy_compressor_destroy
#define coli_v4_compressor_step snapshot_copy_compressor_step
/* ---- begin include deepseek_v4_compressor.c ---- */
#include "deepseek_v4_internal.h"

#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "deepseek_v4_internal.h"
#include "native_quant.h"

struct ColiDeepSeekV4CompressorState {
    const ColiDeepSeekV4LayerWeights *weights;
    const ColiDeepSeekV4Config *config;
    int ratio;
    int layer;
    int hidden;
    int head_dim;
    int projection_dim;
    int state_rows;
    int rope_dim;
    int rotate_fp4;
    char prefix[96];
    float *kv_state;
    float *score_state;
};

static int set_error(char *error, size_t size, const char *format, ...) {
    if (error && size) {
        va_list arguments;
        va_start(arguments, format);
        vsnprintf(error, size, format, arguments);
        va_end(arguments);
    }
    return -1;
}

static const void *layer_value(const ColiDeepSeekV4LayerWeights *weights,
                               const char *suffix) {
    char name[COLI_V4_MAX_TENSOR_NAME];
    snprintf(name, sizeof(name), "layers.%d.%s", weights->plan.layer, suffix);
    return coli_v4_layer_data(weights, name, NULL);
}

int coli_v4_compressor_create(ColiDeepSeekV4CompressorState **output,
                              const ColiDeepSeekV4LayerWeights *weights,
                              const ColiDeepSeekV4Config *config,
                              char *error, size_t error_size) {
    ColiDeepSeekV4CompressorOptions options = {
        "attn.compressor", config ? config->head_dim : 0, 0
    };
    return coli_v4_compressor_create_with_options(
        output, weights, config, &options, error, error_size);
}

int coli_v4_compressor_create_with_options(
    ColiDeepSeekV4CompressorState **output,
    const ColiDeepSeekV4LayerWeights *weights,
    const ColiDeepSeekV4Config *config,
    const ColiDeepSeekV4CompressorOptions *options,
    char *error, size_t error_size) {
    if (!output || !weights || !config || !options || !options->prefix ||
        !options->prefix[0] || options->head_dimension <= 0 ||
        weights->plan.compression_ratio < 1)
        return set_error(error, error_size, "unsupported compressor ratio");
    if (strlen(options->prefix) >= sizeof(((ColiDeepSeekV4CompressorState *)0)->prefix))
        return set_error(error, error_size, "compressor prefix is too long");
    *output = NULL;
    ColiDeepSeekV4CompressorState *state = calloc(1, sizeof(*state));
    if (!state) return set_error(error, error_size, "out of memory creating compressor");
    state->weights = weights;
    state->config = config;
    state->ratio = weights->plan.compression_ratio;
    state->layer = weights->plan.layer;
    state->hidden = config->hidden_size;
    state->head_dim = options->head_dimension;
    state->rotate_fp4 = options->rotate_fp4 != 0;
    memcpy(state->prefix, options->prefix, strlen(options->prefix) + 1);
    int overlap = state->ratio == 4;
    state->projection_dim = (1 + overlap) * state->head_dim;
    state->state_rows = (1 + overlap) * state->ratio;
    state->rope_dim = config->qk_rope_head_dim;
    size_t count = (size_t)state->state_rows * state->projection_dim;
    state->kv_state = calloc(count, sizeof(*state->kv_state));
    state->score_state = malloc(count * sizeof(*state->score_state));
    if (!state->kv_state || !state->score_state) {
        coli_v4_compressor_destroy(state);
        return set_error(error, error_size, "out of memory allocating compressor state");
    }
    for (size_t i = 0; i < count; i++) state->score_state[i] = -INFINITY;
    *output = state;
    return 0;
}

int coli_v4_compressor_bind_weights(ColiDeepSeekV4CompressorState *state,
                                    const ColiDeepSeekV4LayerWeights *weights,
                                    char *error, size_t error_size) {
    if (!state || !weights ||
        weights->plan.layer != state->layer ||
        weights->plan.compression_ratio != state->ratio)
        return set_error(error, error_size, "incompatible compressor weights");
    state->weights = weights;
    return 0;
}

void coli_v4_compressor_reset(ColiDeepSeekV4CompressorState *state) {
    if (!state) return;
    size_t count = (size_t)state->state_rows * state->projection_dim;
    memset(state->kv_state, 0, count * sizeof(*state->kv_state));
    for (size_t i = 0; i < count; i++) state->score_state[i] = -INFINITY;
}

void coli_v4_compressor_destroy(ColiDeepSeekV4CompressorState *state) {
    if (!state) return;
    free(state->score_state);
    free(state->kv_state);
    free(state);
}

int coli_v4_compressor_step(ColiDeepSeekV4CompressorState *state,
                            float *output, int *produced,
                            const float *input, int position,
                            char *error, size_t error_size) {
    if (!state || !produced || !input || position < 0)
        return set_error(error, error_size, "invalid compressor step arguments");
    *produced = 0;
    int slot = position % state->ratio;
    int hidden = state->hidden, dimension = state->head_dim;
    int projection = state->projection_dim;
    int state_row = state->ratio == 4 ? state->ratio + slot : slot;
    char suffix[128];
    snprintf(suffix, sizeof(suffix), "%s.wkv.weight", state->prefix);
    const uint16_t *wkv = layer_value(state->weights, suffix);
    snprintf(suffix, sizeof(suffix), "%s.wgate.weight", state->prefix);
    const uint16_t *wgate = layer_value(state->weights, suffix);
    snprintf(suffix, sizeof(suffix), "%s.ape", state->prefix);
    const float *ape = layer_value(state->weights, suffix);
    if (!wkv || !wgate || !ape)
        return set_error(error, error_size, "missing compressor tensor for %s", state->prefix);
    float *kv_row = state->kv_state + (size_t)state_row * projection;
    float *score_row = state->score_state + (size_t)state_row * projection;
    #pragma omp parallel for
    for (int row = 0; row < projection; row++) {
        float kv_sum = 0.0f, gate_sum = 0.0f;
        const uint16_t *kv_weight = wkv + (size_t)row * hidden;
        const uint16_t *gate_weight = wgate + (size_t)row * hidden;
        for (int column = 0; column < hidden; column++) {
            float value = input[column];
            kv_sum += coli_bf16_decode(kv_weight[column]) * value;
            gate_sum += coli_bf16_decode(gate_weight[column]) * value;
        }
        kv_row[row] = kv_sum;
        score_row[row] = gate_sum + ape[(size_t)slot * projection + row];
    }
    if ((position + 1) % state->ratio != 0) return 0;
    if (!output) return set_error(error, error_size, "compressor output is required");

    #pragma omp parallel for
    for (int column = 0; column < dimension; column++) {
        float maximum = -INFINITY;
        int pool_rows = state->ratio == 4 ? 2 * state->ratio : state->ratio;
        for (int row = 0; row < pool_rows; row++) {
            int source_row = row;
            int source_column = column;
            if (state->ratio == 4 && row >= state->ratio)
                source_column += dimension;
            float score = state->score_state[
                (size_t)source_row * projection + source_column];
            if (score > maximum) maximum = score;
        }
        float total = 0.0f, weighted = 0.0f;
        for (int row = 0; row < pool_rows; row++) {
            int source_column = column;
            if (state->ratio == 4 && row >= state->ratio)
                source_column += dimension;
            float weight = expf(state->score_state[
                (size_t)row * projection + source_column] - maximum);
            total += weight;
            weighted += state->kv_state[
                (size_t)row * projection + source_column] * weight;
        }
        output[column] = weighted / total;
    }
    if (state->ratio == 4) {
        memcpy(state->kv_state,
               state->kv_state + (size_t)state->ratio * projection,
               (size_t)state->ratio * projection * sizeof(*state->kv_state));
        memcpy(state->score_state,
               state->score_state + (size_t)state->ratio * projection,
               (size_t)state->ratio * projection * sizeof(*state->score_state));
    }
    coli_bf16_round_array(output, (size_t)dimension);
    snprintf(suffix, sizeof(suffix), "%s.norm.weight", state->prefix);
    const uint16_t *raw_norm = layer_value(state->weights, suffix);
    float *norm = malloc((size_t)dimension * sizeof(*norm));
    if (!raw_norm || !norm) {
        free(norm);
        return set_error(error, error_size, "missing compressor norm");
    }
    for (int i = 0; i < dimension; i++) norm[i] = coli_bf16_decode(raw_norm[i]);
    coli_v4_rmsnorm(output, output, norm, dimension, state->config->rms_norm_eps);
    coli_bf16_round_array(output, (size_t)dimension);
    free(norm);

    int rope_position = position + 1 - state->ratio;
    int pairs = state->rope_dim / 2;
    float *cosines = malloc((size_t)pairs * sizeof(*cosines));
    float *sines = malloc((size_t)pairs * sizeof(*sines));
    if (!cosines || !sines || coli_v4_rope_position(
            cosines, sines, state->rope_dim, rope_position,
            state->config->original_max_position_embeddings,
            state->config->compress_rope_theta, state->config->rope_factor,
            state->config->rope_beta_fast, state->config->rope_beta_slow)) {
        free(sines); free(cosines);
        return set_error(error, error_size, "cannot create compressor RoPE table");
    }
    float *rope = output + dimension - state->rope_dim;
    coli_v4_rope_apply(rope, 1, state->rope_dim, cosines, sines, 0);
    coli_bf16_round_array(rope, (size_t)state->rope_dim);
    free(sines); free(cosines);

    size_t quantized = state->rotate_fp4
        ? (size_t)dimension : (size_t)(dimension - state->rope_dim);
    size_t block = state->rotate_fp4 ? 32u : 64u;
    float *qdq = malloc(quantized * sizeof(*qdq));
    uint8_t *scales = malloc((quantized + block - 1) / block);
    if (!qdq || !scales) {
        free(scales); free(qdq);
        return set_error(error, error_size, "compressor activation quantization failed");
    }
    int quant_error = 0;
    if (state->rotate_fp4)
        quant_error = coli_hadamard_bf16_ref(output, (size_t)dimension) ||
                      coli_fp4_activation_qdq_ref(qdq, scales, output,
                                                  quantized, block);
    else
        quant_error = coli_fp8_activation_qdq_ref(qdq, scales, output,
                                                  quantized, block);
    if (quant_error) {
        free(scales); free(qdq);
        return set_error(error, error_size, "compressor activation quantization failed");
    }
    memcpy(output, qdq, quantized * sizeof(*output));
    coli_bf16_round_array(output, quantized);
    free(scales); free(qdq);
    *produced = 1;
    return 0;
}
/* ---- end include deepseek_v4_compressor.c ---- */

#undef coli_v4_compressor_step
#undef coli_v4_compressor_destroy
#undef coli_v4_compressor_bind_weights
#undef coli_v4_compressor_reset
#undef coli_v4_compressor_create_with_options
#undef coli_v4_compressor_create

#include "deepseek_v4_internal.h"

struct ColiV4CompressorSnapshot {
    size_t count;
    float *kv_state;
    float *score_state;
};

int coli_v4_compressor_snapshot_create(
    const ColiDeepSeekV4CompressorState *state,
    ColiV4CompressorSnapshot **output) {
    if (!state || !output) return -1;
    *output = calloc(1, sizeof(**output));
    if (!*output) return -1;
    (*output)->count = (size_t)state->state_rows * state->projection_dim;
    (*output)->kv_state = malloc((*output)->count * sizeof(float));
    (*output)->score_state = malloc((*output)->count * sizeof(float));
    if (!(*output)->kv_state || !(*output)->score_state) {
        coli_v4_compressor_snapshot_destroy(*output); *output = NULL; return -1;
    }
    memcpy((*output)->kv_state, state->kv_state,
           (*output)->count * sizeof(float));
    memcpy((*output)->score_state, state->score_state,
           (*output)->count * sizeof(float));
    return 0;
}

int coli_v4_compressor_snapshot_restore(
    ColiDeepSeekV4CompressorState *state,
    const ColiV4CompressorSnapshot *snapshot) {
    if (!state || !snapshot || snapshot->count !=
        (size_t)state->state_rows * state->projection_dim) return -1;
    memcpy(state->kv_state, snapshot->kv_state, snapshot->count * sizeof(float));
    memcpy(state->score_state, snapshot->score_state,
           snapshot->count * sizeof(float));
    return 0;
}

void coli_v4_compressor_snapshot_destroy(ColiV4CompressorSnapshot *snapshot) {
    if (!snapshot) return;
    free(snapshot->score_state); free(snapshot->kv_state); free(snapshot);
}
#endif /* COLI_V4_UNIT_COMPRESSOR_SNAPSHOT */

#ifdef COLI_V4_UNIT_INDEXER_SNAPSHOT
/* ######## deepseek_v4_indexer_snapshot.c ######## */
#define coli_v4_indexer_create snapshot_copy_indexer_create
#define coli_v4_indexer_bind_weights snapshot_copy_indexer_bind_weights
#define coli_v4_indexer_reset snapshot_copy_indexer_reset
#define coli_v4_indexer_destroy snapshot_copy_indexer_destroy
#define coli_v4_indexer_step snapshot_copy_indexer_step
#define coli_v4_indexer_compressed_values snapshot_copy_indexer_values
#define coli_v4_indexer_compressed_count snapshot_copy_indexer_count
/* ---- begin include deepseek_v4_indexer.c ---- */
#include "deepseek_v4_internal.h"

#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "deepseek_v4_internal.h"
#include "deepseek_v4_internal.h"
#include "native_quant.h"

struct ColiDeepSeekV4Indexer {
    const ColiDeepSeekV4LayerWeights *weights;
    const ColiDeepSeekV4Config *config;
    ColiDeepSeekV4CompressorState *compressor;
    int layer;
    int capacity;
    int count;
    float *compressed;
};

typedef struct { float score; int index; } IndexScore;

static int set_error(char *error, size_t size, const char *format, ...) {
    if (error && size) {
        va_list arguments;
        va_start(arguments, format);
        vsnprintf(error, size, format, arguments);
        va_end(arguments);
    }
    return -1;
}

static const void *value(const ColiDeepSeekV4LayerWeights *weights,
                         const char *suffix,
                         const ColiDeepSeekV4TensorSpec **spec) {
    char name[COLI_V4_MAX_TENSOR_NAME];
    snprintf(name, sizeof(name), "layers.%d.%s", weights->plan.layer, suffix);
    return coli_v4_layer_data(weights, name, spec);
}

static int fp8_view(ColiTensorView *view,
                    const ColiDeepSeekV4LayerWeights *weights,
                    const char *prefix) {
    char suffix[128];
    const ColiDeepSeekV4TensorSpec *ws = NULL, *ss = NULL;
    snprintf(suffix, sizeof(suffix), "%s.weight", prefix);
    const void *data = value(weights, suffix, &ws);
    snprintf(suffix, sizeof(suffix), "%s.scale", prefix);
    const void *scales = value(weights, suffix, &ss);
    if (!data || !scales || !ws || !ss || ws->rank != 2 ||
        ws->dtype != COLI_ST_F8_E4M3 || ss->dtype != COLI_ST_F8_E8M0)
        return -1;
    *view = (ColiTensorView){
        COLI_TENSOR_FP8_E4M3_BLOCK, COLI_SCALE_F32, data, scales,
        (size_t)(ws->shape[0] * ws->shape[1]),
        (size_t)(ss->shape[0] * ss->shape[1]) * sizeof(float),
        ws->shape[0], ws->shape[1], ws->packed_rows8 ? 8 : 128, 128
    };
    return 0;
}

static int descending_score(const void *left, const void *right) {
    const IndexScore *a = left, *b = right;
    if (a->score < b->score) return 1;
    if (a->score > b->score) return -1;
    return a->index - b->index;
}

int coli_v4_indexer_create(ColiDeepSeekV4Indexer **output,
                           const ColiDeepSeekV4LayerWeights *weights,
                           const ColiDeepSeekV4Config *config,
                           int max_context, char *error, size_t error_size) {
    if (!output || !weights || !config || !weights->plan.has_indexer ||
        max_context < 4 || config->index_head_dim < 1 ||
        config->index_n_heads < 1)
        return set_error(error, error_size, "invalid indexer options");
    *output = NULL;
    ColiDeepSeekV4Indexer *state = calloc(1, sizeof(*state));
    if (!state) return set_error(error, error_size, "out of memory creating indexer");
    state->weights = weights;
    state->config = config;
    state->layer = weights->plan.layer;
    state->capacity = (max_context + 3) / 4;
    if (state->capacity > 128) state->capacity = 128;
    state->compressed = calloc((size_t)state->capacity * config->index_head_dim,
                               sizeof(*state->compressed));
    ColiDeepSeekV4CompressorOptions options = {
        "attn.indexer.compressor", config->index_head_dim, 1
    };
    if (!state->compressed || coli_v4_compressor_create_with_options(
            &state->compressor, weights, config, &options, error, error_size)) {
        coli_v4_indexer_destroy(state);
        return set_error(error, error_size, "cannot create indexer compressor");
    }
    *output = state;
    return 0;
}

int coli_v4_indexer_bind_weights(ColiDeepSeekV4Indexer *state,
                                 const ColiDeepSeekV4LayerWeights *weights,
                                 char *error, size_t error_size) {
    if (!state || !weights || weights->plan.layer != state->layer ||
        !weights->plan.has_indexer)
        return set_error(error, error_size, "incompatible indexer weights");
    state->weights = weights;
    return coli_v4_compressor_bind_weights(state->compressor, weights,
                                            error, error_size);
}

void coli_v4_indexer_reset(ColiDeepSeekV4Indexer *state) {
    if (!state) return;
    state->count = 0;
    memset(state->compressed, 0,
           (size_t)state->capacity * state->config->index_head_dim * sizeof(float));
    coli_v4_compressor_reset(state->compressor);
}

void coli_v4_indexer_destroy(ColiDeepSeekV4Indexer *state) {
    if (!state) return;
    coli_v4_compressor_destroy(state->compressor);
    free(state->compressed);
    free(state);
}

static int apply_position_rope(float *queries,
                               const ColiDeepSeekV4Config *config,
                               int position) {
    int heads = config->index_n_heads, dimension = config->index_head_dim;
    int rope_dim = config->qk_rope_head_dim, pairs = rope_dim / 2;
    float *cosines = malloc((size_t)pairs * sizeof(*cosines));
    float *sines = malloc((size_t)pairs * sizeof(*sines));
    if (!cosines || !sines || coli_v4_rope_position(
            cosines, sines, rope_dim, position,
            config->original_max_position_embeddings,
            config->compress_rope_theta, config->rope_factor,
            config->rope_beta_fast, config->rope_beta_slow)) {
        free(sines); free(cosines); return -1;
    }
    for (int head = 0; head < heads; head++) {
        float *query = queries + (size_t)head * dimension;
        coli_v4_rope_apply(query + dimension - rope_dim, 1, rope_dim,
                           cosines, sines, 0);
        coli_bf16_round_array(query + dimension - rope_dim, (size_t)rope_dim);
    }
    free(sines); free(cosines);
    return 0;
}

int coli_v4_indexer_step(ColiDeepSeekV4Indexer *state, int *indices,
                         int index_capacity, const float *query_rank,
                         const float *input, int position,
                         char *error, size_t error_size) {
    if (!state || !indices || index_capacity < 1 || !query_rank || !input ||
        position < 0)
        return set_error(error, error_size, "invalid indexer step arguments");
    int dimension = state->config->index_head_dim;
    int heads = state->config->index_n_heads;
    int produced = 0;
    if ((position + 1) % 4 == 0 && state->count >= state->capacity) {
        int next_capacity = state->capacity * 2;
        float *grown = realloc(state->compressed,
            (size_t)next_capacity * dimension * sizeof(*grown));
        if (!grown) return set_error(error, error_size, "cannot grow indexer cache");
        memset(grown + (size_t)state->capacity * dimension, 0,
               (size_t)(next_capacity - state->capacity) * dimension * sizeof(*grown));
        state->compressed = grown;
        state->capacity = next_capacity;
    }
    float *next = state->count < state->capacity
        ? state->compressed + (size_t)state->count * dimension : NULL;
    if (coli_v4_compressor_step(state->compressor, next, &produced, input,
                                position, error, error_size)) return -1;
    if (produced) {
        if (state->count >= state->capacity)
            return set_error(error, error_size, "indexer cache capacity exceeded");
        state->count++;
    }
    if (!state->count) return 0;

    ColiTensorView wq;
    if (fp8_view(&wq, state->weights, "attn.indexer.wq_b"))
        return set_error(error, error_size, "missing indexer query weight");
    float *queries = malloc((size_t)heads * dimension * sizeof(*queries));
    float *head_weights = malloc((size_t)heads * sizeof(*head_weights));
    IndexScore *scores = malloc((size_t)state->count * sizeof(*scores));
    uint8_t *scales = malloc((size_t)dimension / 32);
    float *qdq = malloc((size_t)dimension * sizeof(*qdq));
    const uint16_t *raw_weights = value(
        state->weights, "attn.indexer.weights_proj.weight", NULL);
    if (!queries || !head_weights || !scores || !scales || !qdq || !raw_weights) {
        free(qdq); free(scales); free(scores); free(head_weights); free(queries);
        return set_error(error, error_size, "out of memory scoring indexer");
    }
    int result = coli_fp8_matvec_ref(queries, &wq, query_rank);
    if (!result) coli_bf16_round_array(queries, (size_t)heads * dimension);
    if (!result) result = apply_position_rope(queries, state->config, position);
    for (int head = 0; !result && head < heads; head++) {
        float *query = queries + (size_t)head * dimension;
        result = coli_hadamard_bf16_ref(query, (size_t)dimension);
        if (!result) result = coli_fp4_activation_qdq_ref(
            qdq, scales, query, (size_t)dimension, 32);
        if (!result) {
            memcpy(query, qdq, (size_t)dimension * sizeof(*query));
            coli_bf16_round_array(query, (size_t)dimension);
        }
    }
    float weight_scale = 1.0f / sqrtf((float)(dimension * heads));
    for (int head = 0; !result && head < heads; head++) {
        float sum = 0.0f;
        const uint16_t *row = raw_weights + (size_t)head * state->config->hidden_size;
        for (int column = 0; column < state->config->hidden_size; column++)
            sum += coli_bf16_decode(row[column]) * input[column];
        head_weights[head] = sum * weight_scale;
    }
    for (int candidate = 0; !result && candidate < state->count; candidate++) {
        const float *key = state->compressed + (size_t)candidate * dimension;
        float score = 0.0f;
        for (int head = 0; head < heads; head++) {
            const float *query = queries + (size_t)head * dimension;
            float dot = 0.0f;
            for (int i = 0; i < dimension; i++) dot += query[i] * key[i];
            score += fmaxf(dot, 0.0f) * head_weights[head];
        }
        scores[candidate] = (IndexScore){score, candidate};
    }
    if (!result) qsort(scores, (size_t)state->count, sizeof(*scores), descending_score);
    int selected = state->count;
    if (selected > state->config->index_topk) selected = state->config->index_topk;
    if (selected > index_capacity) selected = index_capacity;
    for (int i = 0; !result && i < selected; i++) indices[i] = scores[i].index;
    free(qdq); free(scales); free(scores); free(head_weights); free(queries);
    return result ? set_error(error, error_size, "indexer scoring failed") : selected;
}

const float *coli_v4_indexer_compressed_values(
    const ColiDeepSeekV4Indexer *state) {
    return state ? state->compressed : NULL;
}

int coli_v4_indexer_compressed_count(const ColiDeepSeekV4Indexer *state) {
    return state ? state->count : 0;
}
/* ---- end include deepseek_v4_indexer.c ---- */

#undef coli_v4_indexer_compressed_count
#undef coli_v4_indexer_compressed_values
#undef coli_v4_indexer_step
#undef coli_v4_indexer_destroy
#undef coli_v4_indexer_reset
#undef coli_v4_indexer_bind_weights
#undef coli_v4_indexer_create

#include "deepseek_v4_internal.h"
#include "deepseek_v4_internal.h"

struct ColiV4IndexerSnapshot {
    int count;
    int head_dim;
    float *compressed;
    ColiV4CompressorSnapshot *compressor;
};

int coli_v4_indexer_snapshot_create(const ColiDeepSeekV4Indexer *state,
                                    ColiV4IndexerSnapshot **output) {
    if (!state || !output || !state->config) return -1;
    *output = calloc(1, sizeof(**output));
    if (!*output) return -1;
    (*output)->count = state->count;
    (*output)->head_dim = state->config->index_head_dim;
    if (state->count) {
        (*output)->compressed = malloc((size_t)state->count *
                                       (*output)->head_dim * sizeof(float));
        if (!(*output)->compressed) {
            coli_v4_indexer_snapshot_destroy(*output); *output = NULL; return -1;
        }
        memcpy((*output)->compressed, state->compressed,
               (size_t)state->count * (*output)->head_dim * sizeof(float));
    }
    if (coli_v4_compressor_snapshot_create(state->compressor,
                                            &(*output)->compressor)) {
        coli_v4_indexer_snapshot_destroy(*output); *output = NULL; return -1;
    }
    return 0;
}

int coli_v4_indexer_snapshot_restore(ColiDeepSeekV4Indexer *state,
                                     const ColiV4IndexerSnapshot *snapshot) {
    if (!state || !snapshot || !state->config ||
        state->config->index_head_dim != snapshot->head_dim ||
        snapshot->count > state->capacity) return -1;
    state->count = snapshot->count;
    if (snapshot->count)
        memcpy(state->compressed, snapshot->compressed,
               (size_t)snapshot->count * snapshot->head_dim * sizeof(float));
    return coli_v4_compressor_snapshot_restore(state->compressor,
                                                snapshot->compressor);
}

void coli_v4_indexer_snapshot_destroy(ColiV4IndexerSnapshot *snapshot) {
    if (!snapshot) return;
    coli_v4_compressor_snapshot_destroy(snapshot->compressor);
    free(snapshot->compressed); free(snapshot);
}
#endif /* COLI_V4_UNIT_INDEXER_SNAPSHOT */

#ifdef COLI_V4_UNIT_ATTENTION_TRANSACTION
/* ######## deepseek_v4_attention_transaction.c ######## */
#define coli_v4_window_attention_create transaction_copy_attention_create
#define coli_v4_window_attention_reset transaction_copy_attention_reset
#define coli_v4_window_attention_destroy transaction_copy_attention_destroy
#define coli_v4_attention_token_ref transaction_copy_attention_token
#define coli_v4_attention_window_token_ref transaction_copy_attention_window_token
/* ---- begin include deepseek_v4_attention.c ---- */
#include "deepseek_v4_internal.h"

#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "deepseek_v4_internal.h"
#include "deepseek_v4_internal.h"
#include "deepseek_v4_internal.h"
#include "deepseek_v4_internal.h"
#include "native_quant.h"

static int set_error(char *error, size_t size, const char *format, ...);

struct ColiDeepSeekV4WindowAttentionState {
    int window_size;
    int head_dim;
    int layer;
    int ratio;
    float *kv;
    ColiDeepSeekV4CompressorState *compressor;
    ColiDeepSeekV4Indexer *indexer;
    float *compressed;
    int compressed_count;
    int compressed_capacity;
};

int coli_v4_window_attention_create(ColiDeepSeekV4WindowAttentionState **output,
                                    const ColiDeepSeekV4Config *config) {
    if (!output || !config || config->sliding_window < 1 || config->head_dim < 1)
        return -1;
    *output = calloc(1, sizeof(**output));
    if (!*output) return -1;
    (*output)->window_size = config->sliding_window;
    (*output)->head_dim = config->head_dim;
    (*output)->layer = -1;
    (*output)->kv = calloc((size_t)config->sliding_window * config->head_dim,
                           sizeof(*(*output)->kv));
    if (!(*output)->kv) {
        free(*output);
        *output = NULL;
        return -1;
    }
    return 0;
}

void coli_v4_window_attention_reset(ColiDeepSeekV4WindowAttentionState *state) {
    if (!state) return;
    memset(state->kv, 0,
           (size_t)state->window_size * state->head_dim * sizeof(*state->kv));
    state->compressed_count = 0;
    if (state->compressor) coli_v4_compressor_reset(state->compressor);
    if (state->indexer) coli_v4_indexer_reset(state->indexer);
}

void coli_v4_window_attention_destroy(ColiDeepSeekV4WindowAttentionState *state) {
    if (!state) return;
    coli_v4_indexer_destroy(state->indexer);
    coli_v4_compressor_destroy(state->compressor);
    free(state->compressed);
    free(state->kv);
    free(state);
}

static int prepare_compressed_state(
    ColiDeepSeekV4WindowAttentionState *state,
    const ColiDeepSeekV4LayerWeights *weights,
    const ColiDeepSeekV4Config *config, char *error, size_t error_size) {
    int ratio = weights->plan.compression_ratio;
    if (!ratio) return 0;
    if (state->layer < 0) {
        state->layer = weights->plan.layer;
        state->ratio = ratio;
        state->compressed_capacity = 16;
        state->compressed = calloc((size_t)state->compressed_capacity * state->head_dim,
                                   sizeof(*state->compressed));
        if (!state->compressed || coli_v4_compressor_create(
                &state->compressor, weights, config, error, error_size)) return -1;
        if (ratio == 4 && coli_v4_indexer_create(
                &state->indexer, weights, config, config->max_position_embeddings,
                error, error_size)) return -1;
    } else if (state->layer != weights->plan.layer || state->ratio != ratio) {
        return set_error(error, error_size, "attention state belongs to another layer");
    }
    if (coli_v4_compressor_bind_weights(state->compressor, weights,
                                        error, error_size)) return -1;
    if (state->indexer && coli_v4_indexer_bind_weights(
            state->indexer, weights, error, error_size)) return -1;
    return 0;
}

static int grow_compressed_state(ColiDeepSeekV4WindowAttentionState *state,
                                 char *error, size_t error_size) {
    if (state->compressed_count < state->compressed_capacity) return 0;
    int capacity = state->compressed_capacity * 2;
    float *grown = realloc(state->compressed,
        (size_t)capacity * state->head_dim * sizeof(*grown));
    if (!grown) return set_error(error, error_size, "cannot grow compressed KV cache");
    state->compressed = grown;
    state->compressed_capacity = capacity;
    return 0;
}

static int set_error(char *error, size_t size, const char *format, ...) {
    if (error && size) {
        va_list arguments;
        va_start(arguments, format);
        vsnprintf(error, size, format, arguments);
        va_end(arguments);
    }
    return -1;
}

static const void *layer_data(const ColiDeepSeekV4LayerWeights *weights,
                              const char *suffix,
                              const ColiDeepSeekV4TensorSpec **spec) {
    char name[COLI_V4_MAX_TENSOR_NAME];
    snprintf(name, sizeof(name), "layers.%d.%s", weights->plan.layer, suffix);
    return coli_v4_layer_data(weights, name, spec);
}

static int fp8_view(ColiTensorView *view,
                    const ColiDeepSeekV4LayerWeights *weights,
                    const char *prefix) {
    char suffix[128];
    const ColiDeepSeekV4TensorSpec *weight_spec = NULL, *scale_spec = NULL;
    snprintf(suffix, sizeof(suffix), "%s.weight", prefix);
    const void *data = layer_data(weights, suffix, &weight_spec);
    snprintf(suffix, sizeof(suffix), "%s.scale", prefix);
    const void *scales = layer_data(weights, suffix, &scale_spec);
    if (!data || !scales || !weight_spec || !scale_spec ||
        weight_spec->dtype != COLI_ST_F8_E4M3 ||
        scale_spec->dtype != COLI_ST_F8_E8M0 || weight_spec->rank != 2)
        return -1;
    *view = (ColiTensorView){
        COLI_TENSOR_FP8_E4M3_BLOCK, COLI_SCALE_F32, data, scales,
        (size_t)(weight_spec->shape[0] * weight_spec->shape[1]),
        (size_t)(scale_spec->shape[0] * scale_spec->shape[1]) * sizeof(float),
        weight_spec->shape[0], weight_spec->shape[1],
        weight_spec->packed_rows8 ? 8 : 128, 128
    };
    return 0;
}

static int decode_bf16(float *output, const void *data, size_t count) {
    if (!output || !data) return -1;
    const uint16_t *values = data;
    for (size_t i = 0; i < count; i++) output[i] = coli_bf16_decode(values[i]);
    return 0;
}

static int attention_token_impl(float *output,
                                ColiDeepSeekV4WindowAttentionState *state,
                                const ColiDeepSeekV4LayerWeights *weights,
                                const ColiDeepSeekV4Config *config,
                                const float *input, int position,
                                char *error, size_t error_size) {
    if (!output || !weights || !config || !input || position < 0 ||
        (!state && weights->plan.compression_ratio != 0 && position != 0))
        return set_error(error, error_size, "invalid uncompressed attention arguments");
    int hidden = config->hidden_size;
    int heads = config->num_attention_heads;
    int head_dim = config->head_dim;
    int rope_dim = config->qk_rope_head_dim;
    int q_rank = config->q_lora_rank;
    int groups = config->o_groups;
    int o_rank = config->o_lora_rank;
    if (hidden < 1 || heads < 1 || head_dim < 1 || rope_dim < 2 ||
        rope_dim > head_dim || q_rank < 1 || groups < 1 || heads % groups)
        return set_error(error, error_size, "unsupported attention dimensions");

    ColiTensorView wq_a, wq_b, wkv, wo_a, wo_b;
    if (fp8_view(&wq_a, weights, "attn.wq_a") ||
        fp8_view(&wq_b, weights, "attn.wq_b") ||
        fp8_view(&wkv, weights, "attn.wkv") ||
        fp8_view(&wo_a, weights, "attn.wo_a") ||
        fp8_view(&wo_b, weights, "attn.wo_b"))
        return set_error(error, error_size, "missing native FP8 attention tensor");

    float *qa = calloc((size_t)q_rank, sizeof(*qa));
    float *q = calloc((size_t)heads * head_dim, sizeof(*q));
    float *kv = calloc((size_t)head_dim, sizeof(*kv));
    float *attended = calloc((size_t)heads * head_dim, sizeof(*attended));
    float *oa = calloc((size_t)groups * o_rank, sizeof(*oa));
    float *norm_weight = calloc((size_t)(q_rank > head_dim ? q_rank : head_dim),
                                sizeof(*norm_weight));
    float *cosines = calloc((size_t)rope_dim / 2, sizeof(*cosines));
    float *sines = calloc((size_t)rope_dim / 2, sizeof(*sines));
    int *compressed_indices = NULL;
    int compressed_selected = 0;
    if (!qa || !q || !kv || !attended || !oa || !norm_weight || !cosines || !sines) {
        free(sines); free(cosines); free(norm_weight); free(oa);
        free(attended); free(kv); free(q); free(qa);
        return set_error(error, error_size, "out of memory in attention");
    }

    int result = coli_fp8_matvec_ref(qa, &wq_a, input);
    coli_bf16_round_array(qa, (size_t)q_rank);
    const void *q_norm = layer_data(weights, "attn.q_norm.weight", NULL);
    if (!result && (!q_norm || decode_bf16(norm_weight, q_norm, (size_t)q_rank) ||
                    coli_v4_rmsnorm(qa, qa, norm_weight, q_rank,
                                    config->rms_norm_eps))) result = -1;
    if (!result) coli_bf16_round_array(qa, (size_t)q_rank);
    if (!result && state && weights->plan.compression_ratio) {
        result = prepare_compressed_state(state, weights, config,
                                          error, error_size);
        if (!result && (position + 1) % state->ratio == 0)
            result = grow_compressed_state(state, error, error_size);
        int produced = 0;
        if (!result) result = coli_v4_compressor_step(
            state->compressor,
            state->compressed + (size_t)state->compressed_count * head_dim,
            &produced, input, position, error, error_size);
        if (!result && produced) state->compressed_count++;
        if (!result && state->indexer) {
            compressed_indices = malloc((size_t)config->index_topk *
                                        sizeof(*compressed_indices));
            if (!compressed_indices) result = -1;
            else compressed_selected = coli_v4_indexer_step(
                state->indexer, compressed_indices, config->index_topk,
                qa, input, position, error, error_size);
            if (compressed_selected < 0) result = -1;
        }
    }
    if (!result) result = coli_fp8_matvec_ref(q, &wq_b, qa);
    if (!result) coli_bf16_round_array(q, (size_t)heads * head_dim);
    for (int head = 0; !result && head < heads; head++) {
        float *values = q + (size_t)head * head_dim;
        float mean_square = 0.0f;
        for (int i = 0; i < head_dim; i++) mean_square += values[i] * values[i];
        float scale = 1.0f / sqrtf(mean_square / head_dim + config->rms_norm_eps);
        for (int i = 0; i < head_dim; i++) values[i] = coli_bf16_round(values[i] * scale);
    }

    if (!result) result = coli_fp8_matvec_ref(kv, &wkv, input);
    if (!result) coli_bf16_round_array(kv, (size_t)head_dim);
    const void *kv_norm = layer_data(weights, "attn.kv_norm.weight", NULL);
    if (!result && (!kv_norm || decode_bf16(norm_weight, kv_norm, (size_t)head_dim) ||
                    coli_v4_rmsnorm(kv, kv, norm_weight, head_dim,
                                    config->rms_norm_eps))) result = -1;
    if (!result) coli_bf16_round_array(kv, (size_t)head_dim);

    if (!result) {
        int compressed = weights->plan.compression_ratio != 0;
        if (coli_v4_rope_position(
                cosines, sines, rope_dim, position,
                compressed ? config->original_max_position_embeddings : 0,
                compressed ? config->compress_rope_theta : config->rope_theta,
                config->rope_factor,
                config->rope_beta_fast, config->rope_beta_slow)) result = -1;
    }
    if (!result) {
        for (int head = 0; head < heads; head++) {
            float *rope = q + (size_t)head * head_dim + head_dim - rope_dim;
            coli_v4_rope_apply(rope, 1, rope_dim, cosines, sines, 0);
            coli_bf16_round_array(rope, (size_t)rope_dim);
        }
        float *kv_rope = kv + head_dim - rope_dim;
        coli_v4_rope_apply(kv_rope, 1, rope_dim, cosines, sines, 0);
        coli_bf16_round_array(kv_rope, (size_t)rope_dim);
        size_t nope = (size_t)(head_dim - rope_dim);
        float *qdq = malloc(nope * sizeof(*qdq));
        uint8_t *scales = malloc((nope + 63) / 64);
        if (!qdq || !scales || coli_fp8_activation_qdq_ref(qdq, scales, kv, nope, 64))
            result = -1;
        if (!result) {
            memcpy(kv, qdq, nope * sizeof(*kv));
            coli_bf16_round_array(kv, nope);
        }
        free(scales); free(qdq);
    }

    const float *sinks = layer_data(weights, "attn.attn_sink", NULL);
    if (!result && state) {
        int slot = position % state->window_size;
        memcpy(state->kv + (size_t)slot * head_dim, kv,
               (size_t)head_dim * sizeof(*kv));
        if (!state->indexer) compressed_selected = state->compressed_count;
        int topk = state->window_size + compressed_selected;
        int kv_count = state->window_size + state->compressed_count;
        int *indices = malloc((size_t)topk * sizeof(*indices));
        float *all_kv = state->compressed_count
            ? malloc((size_t)kv_count * head_dim * sizeof(*all_kv)) : NULL;
        if (!indices || (state->compressed_count && !all_kv)) result = -1;
        if (!result) {
            if (position < state->window_size - 1) {
                for (int i = 0; i < state->window_size; i++)
                    indices[i] = i <= position ? i : -1;
            } else {
                int oldest = (position + 1) % state->window_size;
                for (int i = 0; i < state->window_size; i++)
                    indices[i] = (oldest + i) % state->window_size;
            }
            const float *kv_values = state->kv;
            if (state->compressed_count) {
                memcpy(all_kv, state->kv,
                       (size_t)state->window_size * head_dim * sizeof(*all_kv));
                memcpy(all_kv + (size_t)state->window_size * head_dim,
                       state->compressed,
                       (size_t)state->compressed_count * head_dim * sizeof(*all_kv));
                kv_values = all_kv;
            }
            for (int i = 0; i < compressed_selected; i++) {
                int ordinal = state->indexer ? compressed_indices[i] : i;
                indices[state->window_size + i] = state->window_size + ordinal;
            }
            result = coli_v4_sparse_attention_ref(
                attended, q, kv_values, sinks, indices, heads, head_dim,
                kv_count, topk,
                1.0f / sqrtf((float)head_dim));
        }
        free(all_kv);
        free(indices);
    } else for (int head = 0; !result && head < heads; head++) {
        float *query = q + (size_t)head * head_dim;
        float score = 0.0f;
        for (int i = 0; i < head_dim; i++) score += query[i] * kv[i];
        score *= 1.0f / sqrtf((float)head_dim);
        float attention_weight = 1.0f / (1.0f + expf(sinks[head] - score));
        float *head_output = attended + (size_t)head * head_dim;
        for (int i = 0; i < head_dim; i++)
            head_output[i] = coli_bf16_round(kv[i] * attention_weight);
    }
    for (int head = 0; !result && head < heads; head++) {
        float *head_output = attended + (size_t)head * head_dim;
        float *rope = head_output + head_dim - rope_dim;
        coli_v4_rope_apply(rope, 1, rope_dim, cosines, sines, 1);
        coli_bf16_round_array(rope, (size_t)rope_dim);
    }

    int heads_per_group = heads / groups;
    int group_width = heads_per_group * head_dim;
    int scale_columns = (group_width + 127) / 128;
    int scale_rows_per_group = (o_rank + 127) / 128;
    for (int group = 0; !result && group < groups; group++) {
        ColiTensorView group_view = wo_a;
        group_view.rows = o_rank;
        group_view.columns = group_width;
        group_view.data = (const uint8_t *)wo_a.data +
            (size_t)group * o_rank * group_width;
        group_view.scales = (const uint8_t *)wo_a.scales +
            (size_t)group * scale_rows_per_group * scale_columns * sizeof(float);
        group_view.data_bytes = (size_t)o_rank * group_width;
        group_view.scale_bytes =
            (size_t)scale_rows_per_group * scale_columns * sizeof(float);
        result = coli_fp8_matvec_ref(oa + (size_t)group * o_rank, &group_view,
                                     attended + (size_t)group * group_width);
    }
    if (!result) coli_bf16_round_array(oa, (size_t)groups * o_rank);
    if (!result) result = coli_fp8_matvec_ref(output, &wo_b, oa);
    if (!result) coli_bf16_round_array(output, (size_t)hidden);

    free(compressed_indices);
    free(sines); free(cosines); free(norm_weight); free(oa);
    free(attended); free(kv); free(q); free(qa);
    if (result) return set_error(error, error_size, "attention computation failed");
    return 0;
}

int coli_v4_attention_token_ref(float *output,
                                const ColiDeepSeekV4LayerWeights *weights,
                                const ColiDeepSeekV4Config *config,
                                const float *input, int position,
                                char *error, size_t error_size) {
    return attention_token_impl(output, NULL, weights, config, input, position,
                                error, error_size);
}

int coli_v4_attention_window_token_ref(
    float *output, ColiDeepSeekV4WindowAttentionState *state,
    const ColiDeepSeekV4LayerWeights *weights,
    const ColiDeepSeekV4Config *config, const float *input, int position,
    char *error, size_t error_size) {
    return attention_token_impl(output, state, weights, config, input, position,
                                error, error_size);
}
/* ---- end include deepseek_v4_attention.c ---- */

#undef coli_v4_attention_window_token_ref
#undef coli_v4_attention_token_ref
#undef coli_v4_window_attention_destroy
#undef coli_v4_window_attention_reset
#undef coli_v4_window_attention_create

#include "deepseek_v4_internal.h"
#include "deepseek_v4_internal.h"
#include "deepseek_v4_internal.h"

struct ColiV4AttentionSnapshot {
    int window_size, head_dim, compressed_count;
    float *kv;
    float *compressed;
    ColiV4CompressorSnapshot *compressor;
    ColiV4IndexerSnapshot *indexer;
};

int coli_v4_attention_snapshot_create(
    const ColiDeepSeekV4WindowAttentionState *state,
    ColiV4AttentionSnapshot **output) {
    if (!state || !output) return -1;
    *output = calloc(1, sizeof(**output));
    if (!*output) return -1;
    (*output)->window_size = state->window_size;
    (*output)->head_dim = state->head_dim;
    (*output)->compressed_count = state->compressed_count;
    size_t kv_count = (size_t)state->window_size * state->head_dim;
    (*output)->kv = malloc(kv_count * sizeof(float));
    if (!(*output)->kv) goto failed;
    memcpy((*output)->kv, state->kv, kv_count * sizeof(float));
    if (state->compressed_count) {
        size_t count = (size_t)state->compressed_count * state->head_dim;
        (*output)->compressed = malloc(count * sizeof(float));
        if (!(*output)->compressed) goto failed;
        memcpy((*output)->compressed, state->compressed, count * sizeof(float));
    }
    if (state->compressor && coli_v4_compressor_snapshot_create(
            state->compressor, &(*output)->compressor)) goto failed;
    if (state->indexer && coli_v4_indexer_snapshot_create(
            state->indexer, &(*output)->indexer)) goto failed;
    return 0;
failed:
    coli_v4_attention_snapshot_destroy(*output); *output = NULL; return -1;
}

int coli_v4_attention_snapshot_restore(
    ColiDeepSeekV4WindowAttentionState *state,
    const ColiV4AttentionSnapshot *snapshot) {
    if (!state || !snapshot || state->window_size != snapshot->window_size ||
        state->head_dim != snapshot->head_dim ||
        snapshot->compressed_count > state->compressed_capacity) return -1;
    memcpy(state->kv, snapshot->kv,
           (size_t)state->window_size * state->head_dim * sizeof(float));
    state->compressed_count = snapshot->compressed_count;
    if (snapshot->compressed_count)
        memcpy(state->compressed, snapshot->compressed,
               (size_t)snapshot->compressed_count * state->head_dim * sizeof(float));
    if ((state->compressor != NULL) != (snapshot->compressor != NULL) ||
        (state->indexer != NULL) != (snapshot->indexer != NULL)) return -1;
    if (state->compressor && coli_v4_compressor_snapshot_restore(
            state->compressor, snapshot->compressor)) return -1;
    if (state->indexer && coli_v4_indexer_snapshot_restore(
            state->indexer, snapshot->indexer)) return -1;
    return 0;
}

void coli_v4_attention_snapshot_destroy(ColiV4AttentionSnapshot *snapshot) {
    if (!snapshot) return;
    coli_v4_indexer_snapshot_destroy(snapshot->indexer);
    coli_v4_compressor_snapshot_destroy(snapshot->compressor);
    free(snapshot->compressed); free(snapshot->kv); free(snapshot);
}
#endif /* COLI_V4_UNIT_ATTENTION_TRANSACTION */

#ifdef COLI_V4_UNIT_EXPERT_STORE_HOT_ROWS16
/* ######## deepseek_v4_expert_store_hot_rows16.c ######## */
/* Hot target experts are converted in-place to a 16-row resident layout.
 * Cold experts keep the official row-major FP4 representation. */
#define coli_deepseek_v4_expert_store_open \
    coli_deepseek_v4_expert_store_open_base
/* ---- begin include deepseek_v4_expert_store.c ---- */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "deepseek_v4_internal.h"

#include <assert.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#ifdef COLI_V4_EXPERIMENTAL_PREFETCH_BATCH
int coli_st_prefetch_many(
    const ColiSafetensorsIndex *index, const int *shards,
    const uint64_t *offsets, const size_t *lengths, size_t count);
#endif

enum { V4_W1 = 0, V4_W2 = 1, V4_W3 = 2, V4_MATRIX_COUNT = 3 };

typedef struct {
    const ColiSafetensorsTensor *weight[V4_MATRIX_COUNT];
    const ColiSafetensorsTensor *scale[V4_MATRIX_COUNT];
    int shard;
    uint64_t scale_offset;
    uint64_t scale_bytes;
    uint64_t weight_offset;
    uint64_t weight_bytes;
    uint64_t record_bytes;
} V4ExpertRecord;

typedef struct {
    int expert;
    unsigned references;
    uint64_t used;
    unsigned char *slab;
    int aligned_slab;
} V4ExpertSlot;

typedef struct {
    ColiSafetensorsIndex *index;
    int layers;
    int experts_per_layer;
    int slots_per_layer;
    uint64_t record_bytes;
    V4ExpertRecord *records;
    V4ExpertSlot *slots;
    uint64_t clock;
    unsigned active_leases;
    ColiExpertStoreStats stats;
    pthread_mutex_t mutex;
} V4ExpertStoreState;

static int set_error(char *error, size_t size, const char *format, ...) {
    if (error && size) {
        va_list args;
        va_start(args, format);
        vsnprintf(error, size, format, args);
        va_end(args);
    }
    return -1;
}

static int compare_tensors(const void *left, const void *right) {
    const ColiSafetensorsTensor *const *a = left;
    const ColiSafetensorsTensor *const *b = right;
    return ((*a)->off > (*b)->off) - ((*a)->off < (*b)->off);
}

static int contiguous_group(const ColiSafetensorsTensor *const input[3],
                            const ColiSafetensorsIndex *index,
                            int *shard, uint64_t *offset, uint64_t *bytes) {
    const ColiSafetensorsTensor *parts[3] = {input[0], input[1], input[2]};
    qsort(parts, 3, sizeof(parts[0]), compare_tensors);
    if (parts[0]->fd != parts[1]->fd || parts[1]->fd != parts[2]->fd ||
        parts[0]->off + parts[0]->nbytes != parts[1]->off ||
        parts[1]->off + parts[1]->nbytes != parts[2]->off)
        return -1;
    *shard = coli_st_tensor_shard(index, parts[0]);
    *offset = (uint64_t)parts[0]->off;
    *bytes = (uint64_t)(parts[2]->off + parts[2]->nbytes - parts[0]->off);
    return *shard < 0 ? -1 : 0;
}

static int validate_matrix(const ColiSafetensorsTensor *weight,
                           const ColiSafetensorsTensor *scale) {
    if (!weight || !scale || weight->dtype != COLI_ST_I8 ||
        scale->dtype != COLI_ST_F8_E8M0 || weight->rank != 2 || scale->rank != 2 ||
        weight->shape[0] != scale->shape[0] || weight->shape[1] <= 0 ||
        scale->shape[1] <= 0)
        return -1;
    int64_t logical_columns = weight->shape[1] * 2;
    return scale->shape[1] * 32 == logical_columns ? 0 : -1;
}

static int build_record(V4ExpertStoreState *state, int layer, int expert,
                        V4ExpertRecord *record, char *error, size_t error_size) {
    static const char *matrix_names[V4_MATRIX_COUNT] = {"w1", "w2", "w3"};
    char name[160];
    memset(record, 0, sizeof(*record));
    for (int matrix = 0; matrix < V4_MATRIX_COUNT; matrix++) {
        snprintf(name, sizeof(name), "layers.%d.ffn.experts.%d.%s.weight",
                 layer, expert, matrix_names[matrix]);
        record->weight[matrix] = coli_st_find(state->index, name);
        snprintf(name, sizeof(name), "layers.%d.ffn.experts.%d.%s.scale",
                 layer, expert, matrix_names[matrix]);
        record->scale[matrix] = coli_st_find(state->index, name);
        if (validate_matrix(record->weight[matrix], record->scale[matrix]) != 0)
            return set_error(error, error_size,
                             "invalid native FP4 expert matrix: layer=%d expert=%d %s",
                             layer, expert, matrix_names[matrix]);
    }
    int scale_shard = -1, weight_shard = -1;
    if (contiguous_group(record->scale, state->index,
                         &scale_shard, &record->scale_offset,
                         &record->scale_bytes) != 0 ||
        contiguous_group(record->weight, state->index,
                         &weight_shard, &record->weight_offset,
                         &record->weight_bytes) != 0 || scale_shard != weight_shard)
        return set_error(error, error_size,
                         "expert is not two contiguous ranges: layer=%d expert=%d",
                         layer, expert);
    record->shard = scale_shard;
    record->record_bytes = record->scale_bytes + record->weight_bytes;
    return 0;
}

static V4ExpertRecord *get_record(V4ExpertStoreState *state, ColiExpertKey key) {
    if (key.layer < 0 || key.layer >= state->layers || key.expert < 0 ||
        key.expert >= state->experts_per_layer)
        return NULL;
    return &state->records[(size_t)key.layer * state->experts_per_layer + key.expert];
}

static V4ExpertSlot *layer_slots(V4ExpertStoreState *state, int layer) {
    return state->slots + (size_t)layer * state->slots_per_layer;
}

static void fill_tensor_view(ColiTensorView *view,
                             const V4ExpertRecord *record,
                             const V4ExpertSlot *slot, int matrix) {
    const ColiSafetensorsTensor *weight = record->weight[matrix];
    const ColiSafetensorsTensor *scale = record->scale[matrix];
    memset(view, 0, sizeof(*view));
    view->format = COLI_TENSOR_FP4_NATIVE_BLOCK;
    view->scale_format = COLI_SCALE_UE8M0;
    view->data = slot->slab + record->scale_bytes +
                 ((uint64_t)weight->off - record->weight_offset);
    view->scales = slot->slab + ((uint64_t)scale->off - record->scale_offset);
    view->data_bytes = (size_t)weight->nbytes;
    view->scale_bytes = (size_t)scale->nbytes;
    view->rows = weight->shape[0];
    view->columns = weight->shape[1] * 2;
    view->block_rows = 1;
    view->block_columns = 32;
}

static int lookup(ColiExpertStore *store, ColiExpertKey key,
                  ColiExpertView *view) {
    if (!store || !store->state || !view) {
        if (view) memset(view, 0, sizeof(*view));
        return -1;
    }
    V4ExpertStoreState *state = store->state;
    V4ExpertRecord *record = get_record(state, key);
    if (!record) {
        memset(view, 0, sizeof(*view));
        return -1;
    }
    pthread_mutex_lock(&state->mutex);
    state->stats.requests++;
    V4ExpertSlot *slots = layer_slots(state, key.layer);
    V4ExpertSlot *slot = NULL;
    for (int i = 0; i < state->slots_per_layer; i++) {
        if (slots[i].slab && slots[i].expert == key.expert) {
            slot = &slots[i];
            state->stats.hits++;
            break;
        }
    }
    if (!slot) {
        for (int i = 0; i < state->slots_per_layer; i++) {
            if (!slots[i].references && (!slot || !slots[i].slab ||
                                         (slot->slab && slots[i].used < slot->used)))
                slot = &slots[i];
        }
        if (!slot) {
            pthread_mutex_unlock(&state->mutex);
            memset(view, 0, sizeof(*view));
            return -1;
        }
        if (!slot->slab) {
            slot->slab = malloc((size_t)state->record_bytes);
            if (!slot->slab) {
                pthread_mutex_unlock(&state->mutex);
                memset(view, 0, sizeof(*view));
                return -1;
            }
            state->stats.resident_bytes += state->record_bytes;
        }
        /* A short read must never expose a partially overwritten old slot. */
        slot->expert = -1;
        if (coli_st_read_at_streaming(
                state->index, record->shard, record->scale_offset,
                (size_t)record->scale_bytes, slot->slab) != 0 ||
            coli_st_read_at_streaming(
                state->index, record->shard, record->weight_offset,
                (size_t)record->weight_bytes,
                slot->slab + record->scale_bytes) != 0) {
            pthread_mutex_unlock(&state->mutex);
            memset(view, 0, sizeof(*view));
            return -1;
        }
        slot->expert = key.expert;
        state->stats.misses++;
        state->stats.bytes_read += record->record_bytes;
    }
    slot->references++;
    state->active_leases++;
    slot->used = ++state->clock;
    memset(view, 0, sizeof(*view));
    view->key = key;
    fill_tensor_view(&view->gate, record, slot, V4_W1);
    fill_tensor_view(&view->down, record, slot, V4_W2);
    fill_tensor_view(&view->up, record, slot, V4_W3);
    view->lease = slot;
    pthread_mutex_unlock(&state->mutex);
    return 0;
}

static void release(ColiExpertStore *store, ColiExpertView *view) {
    if (!store || !store->state || !view || !view->lease) {
        if (view) memset(view, 0, sizeof(*view));
        return;
    }
    V4ExpertStoreState *state = store->state;
    V4ExpertSlot *slot = view->lease;
    pthread_mutex_lock(&state->mutex);
    if (slot->references) slot->references--;
    if (state->active_leases) state->active_leases--;
    pthread_mutex_unlock(&state->mutex);
    memset(view, 0, sizeof(*view));
}

static int prefetch(ColiExpertStore *store, const ColiExpertKey *keys,
                    size_t count) {
    if (!store || !store->state || (!keys && count)) return 0;
    V4ExpertStoreState *state = store->state;
    int accepted = 0;
#ifdef COLI_V4_EXPERIMENTAL_PREFETCH_BATCH
    size_t capacity = count * 2, ranges = 0;
    int *shards = malloc(capacity * sizeof(*shards));
    uint64_t *offsets = malloc(capacity * sizeof(*offsets));
    size_t *lengths = malloc(capacity * sizeof(*lengths));
    int candidates = 0;
    if ((!shards || !offsets || !lengths) && capacity) {
        free(lengths); free(offsets); free(shards); return 0;
    }
    pthread_mutex_lock(&state->mutex);
    for (size_t i = 0; i < count; i++) {
        V4ExpertRecord *record = get_record(state, keys[i]);
        if (!record) continue;
        int resident = 0;
        V4ExpertSlot *slots = layer_slots(state, keys[i].layer);
        for (int slot = 0; slot < state->slots_per_layer; slot++)
            if (slots[slot].slab && slots[slot].expert == keys[i].expert) {
                resident = 1; break;
            }
        if (resident) continue;
        shards[ranges] = record->shard;
        offsets[ranges] = record->scale_offset;
        lengths[ranges++] = (size_t)record->scale_bytes;
        shards[ranges] = record->shard;
        offsets[ranges] = record->weight_offset;
        lengths[ranges++] = (size_t)record->weight_bytes;
        candidates++;
    }
    pthread_mutex_unlock(&state->mutex);
    if (candidates && !coli_st_prefetch_many(
            state->index, shards, offsets, lengths, ranges))
        accepted = candidates;
    free(lengths); free(offsets); free(shards);
#else
    for (size_t i = 0; i < count; i++) {
        V4ExpertRecord *record = get_record(state, keys[i]);
        if (!record) continue;
        if (coli_st_prefetch_at(state->index, record->shard, record->scale_offset,
                                (size_t)record->scale_bytes) == 0 &&
            coli_st_prefetch_at(state->index, record->shard, record->weight_offset,
                                (size_t)record->weight_bytes) == 0)
            accepted++;
    }
#endif
    pthread_mutex_lock(&state->mutex);
    state->stats.prefetched += (uint64_t)accepted;
    pthread_mutex_unlock(&state->mutex);
    return accepted;
}

static void stats(const ColiExpertStore *store, ColiExpertStoreStats *output) {
    if (!store || !store->state || !output) return;
    V4ExpertStoreState *state = store->state;
    pthread_mutex_lock(&state->mutex);
    *output = state->stats;
    pthread_mutex_unlock(&state->mutex);
}

static void destroy(ColiExpertStore *store) {
    if (!store) return;
    V4ExpertStoreState *state = store->state;
    if (state) {
        assert(state->active_leases == 0 && "destroy with active expert leases");
        for (int i = 0; i < state->layers * state->slots_per_layer; i++)
            /* aligned_slab means posix_memalign, which on Windows is
             * _aligned_malloc and must not reach free(). */
            if (state->slots[i].aligned_slab)
                compat_aligned_free(state->slots[i].slab);
            else
                free(state->slots[i].slab);
        pthread_mutex_destroy(&state->mutex);
        coli_st_index_close(state->index);
        free(state->records);
        free(state->slots);
        free(state);
    }
    free(store);
}

int coli_deepseek_v4_expert_store_open(
    const ColiDeepSeekV4ExpertStoreOptions *options, ColiExpertStore **output,
    char *error, size_t error_size) {
    static const ColiExpertStoreOps operations = {
        lookup, release, prefetch, stats, destroy
    };
    if (!options || !output || !options->model_dir || options->layers < 1 ||
        options->experts_per_layer < 1 || !options->cache_bytes)
        return set_error(error, error_size, "invalid DeepSeek-V4 ExpertStore options");
    *output = NULL;
    ColiExpertStore *store = calloc(1, sizeof(*store));
    V4ExpertStoreState *state = calloc(1, sizeof(*state));
    if (!store || !state) {
        free(store);
        free(state);
        return set_error(error, error_size, "out of memory creating ExpertStore");
    }
    pthread_mutex_init(&state->mutex, NULL);
    state->layers = options->layers;
    state->experts_per_layer = options->experts_per_layer;
    if (coli_st_index_open(&state->index, options->model_dir, error, error_size) != 0)
        goto fail;
    size_t record_count = (size_t)state->layers * state->experts_per_layer;
    state->records = calloc(record_count, sizeof(*state->records));
    if (!state->records) {
        set_error(error, error_size, "out of memory creating expert manifest");
        goto fail;
    }
    for (int layer = 0; layer < state->layers; layer++) {
        for (int expert = 0; expert < state->experts_per_layer; expert++) {
            V4ExpertRecord *record = &state->records[
                (size_t)layer * state->experts_per_layer + expert];
            if (build_record(state, layer, expert, record, error, error_size) != 0)
                goto fail;
            if (!state->record_bytes) state->record_bytes = record->record_bytes;
            if (record->record_bytes != state->record_bytes) {
                set_error(error, error_size, "non-uniform expert size at layer=%d expert=%d",
                          layer, expert);
                goto fail;
            }
        }
    }
    state->slots_per_layer = (int)(options->cache_bytes /
        ((uint64_t)state->layers * state->record_bytes));
    int minimum_slots = state->experts_per_layer < 6
        ? state->experts_per_layer : 6;
    if (state->slots_per_layer < minimum_slots) {
        set_error(error, error_size,
                  "cache budget cannot hold %d active experts per layer "
                  "(need %llu bytes)", minimum_slots,
                  (unsigned long long)((uint64_t)state->layers * minimum_slots *
                                       state->record_bytes));
        goto fail;
    }
    if (state->slots_per_layer > state->experts_per_layer)
        state->slots_per_layer = state->experts_per_layer;
    state->slots = calloc((size_t)state->layers * state->slots_per_layer,
                          sizeof(*state->slots));
    if (!state->slots) {
        set_error(error, error_size, "out of memory creating expert cache slots");
        goto fail;
    }
    for (int i = 0; i < state->layers * state->slots_per_layer; i++)
        state->slots[i].expert = -1;
    state->stats.capacity_bytes = (uint64_t)state->layers *
                                  state->slots_per_layer * state->record_bytes;
    store->ops = &operations;
    store->state = state;
    *output = store;
    return 0;

fail:
    if (state->slots) free(state->slots);
    free(state->records);
    coli_st_index_close(state->index);
    pthread_mutex_destroy(&state->mutex);
    free(state);
    free(store);
    return -1;
}
/* ---- end include deepseek_v4_expert_store.c ---- */

#undef coli_deepseek_v4_expert_store_open

#include "native_quant_fp4_rows16.h"
#include <limits.h>
#include <time.h>

#include "deepseek_v4_internal.h"

typedef struct V4HotPolicy {
    ColiExpertStore *store;
    int pin_count;
    uint64_t repin_interval;
    uint64_t *usage;
    uint64_t *layer_requests;
    int *pins;
    unsigned char *packed;
    uint64_t packed_slots;
    uint64_t history_total;
    int history_seeded;
    char *history_path;
    struct V4HotPolicy *next;
} V4HotPolicy;

static pthread_mutex_t hot_policies_mutex = PTHREAD_MUTEX_INITIALIZER;
static V4HotPolicy *hot_policies;

static double hot_now(void) {
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return (double)value.tv_sec + value.tv_nsec * 1e-9;
}

static uint32_t hot_engine_id(void) {
    const unsigned char *text = (const unsigned char *)"deepseek_v4";
    uint32_t hash = 2166136261u;
    while (*text) { hash ^= *text++; hash *= 16777619u; }
    return hash;
}

static char *hot_history_path(const char *model_dir) {
    if (!model_dir) return NULL;
    const char suffix[] = "/.coli_usage";
    size_t length = strlen(model_dir) + sizeof(suffix);
    char *path = malloc(length);
    if (path) snprintf(path, length, "%s%s", model_dir, suffix);
    return path;
}

/* Seed the hot-expert ranking from the same sparse text history used by the
 * other engines.  Placement is the only observable effect: expert ids and
 * weights are still selected by the V4 router on every forward. */
static uint64_t hot_usage_load(V4HotPolicy *policy,
                               const V4ExpertStoreState *state,
                               const char *model_dir) {
    if (!policy || !state || !model_dir) return 0;
    char *path = hot_history_path(model_dir);
    if (!path) return 0;
    FILE *stream = fopen(path, "rb");
    if (!stream) { free(path); return 0; }

    int layer = 0, expert = 0, dimensions_seen = 0;
    unsigned count = 0;
    while (fscanf(stream, "%d %d %u", &layer, &expert, &count) == 3) {
        if (layer == -1) {
            dimensions_seen = 1;
            if (expert != state->layers ||
                count != (unsigned)state->experts_per_layer) {
                fprintf(stderr,
                        "v4_autopin ignored=%s dimensions=%d/%u expected=%d/%d\n",
                        path, expert, count, state->layers,
                        state->experts_per_layer);
                fclose(stream); free(path); return 0;
            }
        } else if (layer == -2 &&
                   (expert > 1 || count != hot_engine_id())) {
            fprintf(stderr,
                    "v4_autopin ignored=%s identity=version-%d/id-%u "
                    "expected=version-1/id-%u\n",
                    path, expert, count, hot_engine_id());
            fclose(stream); free(path); return 0;
        }
    }
    if (!dimensions_seen) {
        fclose(stream); free(path); return 0;
    }
    rewind(stream);
    uint64_t total = 0;
    while (fscanf(stream, "%d %d %u", &layer, &expert, &count) == 3) {
        if (layer < 0 || layer >= state->layers || expert < 0 ||
            expert >= state->experts_per_layer || !count) continue;
        uint64_t *slot = &policy->usage[
            (size_t)layer * state->experts_per_layer + expert];
        if (UINT64_MAX - *slot < count) *slot = UINT64_MAX;
        else *slot += count;
        if (UINT64_MAX - total < count) total = UINT64_MAX;
        else total += count;
    }
    fclose(stream);
    if (total)
        fprintf(stderr, "v4_autopin history=%s selections=%llu\n", path,
                (unsigned long long)total);
    free(path);
    return total;
}

static void hot_usage_save(const V4HotPolicy *policy,
                           const V4ExpertStoreState *state) {
    if (!policy || !state || !policy->history_path || !policy->usage) return;
    const char *enabled = getenv("COLI_V4_SAVE_USAGE");
    if (enabled && atoi(enabled) == 0) return;
    size_t tmp_length = strlen(policy->history_path) + sizeof(".tmp");
    char *tmp = malloc(tmp_length);
    if (!tmp) return;
    snprintf(tmp, tmp_length, "%s.tmp", policy->history_path);
    FILE *stream = fopen(tmp, "wb");
    if (!stream) { free(tmp); return; }

    uint64_t total = 0, distinct = 0;
    for (int layer = 0; layer < state->layers; layer++)
        for (int expert = 0; expert < state->experts_per_layer; expert++) {
            uint64_t count = policy->usage[
                (size_t)layer * state->experts_per_layer + expert];
            if (!count) continue;
            total = UINT64_MAX - total < count ? UINT64_MAX : total + count;
            distinct++;
        }
    if (distinct) {
        fprintf(stream, "-1 %d %d\n", state->layers,
                state->experts_per_layer);
        fprintf(stream, "-2 1 %u\n", hot_engine_id());
        for (int layer = 0; layer < state->layers; layer++)
            for (int expert = 0; expert < state->experts_per_layer; expert++) {
                uint64_t count = policy->usage[
                    (size_t)layer * state->experts_per_layer + expert];
                if (!count) continue;
                unsigned stored = count > UINT_MAX ? UINT_MAX : (unsigned)count;
                fprintf(stream, "%d %d %u\n", layer, expert, stored);
            }
    }
    int failed = ferror(stream);
    if (fclose(stream) != 0) failed = 1;
    if (!failed && rename(tmp, policy->history_path) == 0)
        fprintf(stderr,
                "v4_autopin saved=%s selections=%llu distinct=%llu\n",
                policy->history_path, (unsigned long long)total,
                (unsigned long long)distinct);
    else
        fprintf(stderr, "v4_autopin warning=cannot-save-history path=%s\n",
                policy->history_path);
    free(tmp);
}


static V4HotPolicy *hot_find(ColiExpertStore *store) {
    pthread_mutex_lock(&hot_policies_mutex);
    V4HotPolicy *policy = hot_policies;
    while (policy && policy->store != store) policy = policy->next;
    pthread_mutex_unlock(&hot_policies_mutex);
    return policy;
}

#ifndef COLI_V4_PIN_RAMP_REQUESTS
#define COLI_V4_PIN_RAMP_REQUESTS 0
#endif

static int hot_is_pinned(const V4HotPolicy *policy, int layer, int expert) {
    if (!policy || policy->pin_count < 1) return 0;
    int active = policy->pin_count;
#if COLI_V4_PIN_RAMP_REQUESTS > 0
    if (!policy->history_seeded && active > 4) {
        uint64_t grown = policy->layer_requests[layer] /
                         COLI_V4_PIN_RAMP_REQUESTS;
        active = grown >= (uint64_t)(active - 4)
            ? active : 4 + (int)grown;
    }
#endif
    const int *pins = policy->pins + (size_t)layer * policy->pin_count;
    for (int i = 0; i < active; i++)
        if (pins[i] == expert) return 1;
    return 0;
}

static size_t hot_slot_index(const V4ExpertStoreState *state,
                             const V4ExpertSlot *slot) {
    return (size_t)(slot - state->slots);
}

/* Expert records dominate decode I/O.  Read the large FP4 payload directly
 * into the final cache slot, instead of allocating a record-sized bounce
 * buffer and copying it for every miss.  The extra two pages on each slot let
 * an unaligned safetensors range be expanded to an O_DIRECT window safely.
 *
 * FLOCK-packed checkpoints store [scales][weights] contiguously and need one
 * request.  Standard HF checkpoints keep the ranges apart: weights use direct
 * I/O while the much smaller scales use buffered pread.  Any direct-I/O error
 * falls back to the exact buffered path. */
static uint64_t v4_direct_reads;
static uint64_t v4_direct_flock_reads;
static uint64_t v4_direct_payload_bytes;
static uint64_t v4_direct_fallbacks;

static int v4_pread_full_try(int fd, void *destination, size_t length,
                             uint64_t offset) {
    unsigned char *output = destination;
    size_t done = 0;
    while (done < length) {
        ssize_t count = pread(fd, output + done, length - done,
                              (off_t)(offset + done));
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) return -1;
        done += (size_t)count;
    }
    return 0;
}

static int v4_read_direct_window(const V4ExpertStoreState *state, int shard,
                                 unsigned char *slab, uint64_t offset,
                                 size_t length, size_t destination_offset) {
    if (!state || !state->index || shard < 0 || shard >= state->index->nfd ||
        state->index->dfds[shard] < 0) return -1;
    const uint64_t alignment = 4096;
    uint64_t base = offset & ~(alignment - 1);
    size_t pad = (size_t)(offset - base);
    if (length > SIZE_MAX - pad) return -1;
    size_t wanted = pad + length;
    if (wanted > SIZE_MAX - (alignment - 1)) return -1;
    size_t direct_length = (wanted + alignment - 1) & ~(size_t)(alignment - 1);
    uint64_t file_bytes = (uint64_t)state->index->sizes[shard];
    if (base > file_bytes) return -1;
    uint64_t available = file_bytes - base;
    if ((uint64_t)direct_length > available)
        direct_length = (size_t)(available & ~(alignment - 1));
    if (direct_length && v4_pread_full_try(
            state->index->dfds[shard], slab, direct_length, base)) return -1;
    if (direct_length < wanted && v4_pread_full_try(
            state->index->fds[shard], slab + direct_length,
            wanted - direct_length, base + direct_length)) return -1;
    memmove(slab + destination_offset, slab + pad, length);
    return 0;
}

static int v4_read_expert_record(V4ExpertStoreState *state,
                                 const V4ExpertRecord *record,
                                 V4ExpertSlot *slot) {
    int direct_available = slot->aligned_slab &&
        coli_st_streaming_direct_available(state->index, record->shard);
    if (direct_available &&
        record->scale_offset + record->scale_bytes == record->weight_offset &&
        !v4_read_direct_window(state, record->shard, slot->slab,
                               record->scale_offset,
                               (size_t)record->record_bytes, 0)) {
        __atomic_fetch_add(&v4_direct_reads, UINT64_C(1), __ATOMIC_RELAXED);
        __atomic_fetch_add(&v4_direct_flock_reads, UINT64_C(1),
                           __ATOMIC_RELAXED);
        __atomic_fetch_add(&v4_direct_payload_bytes, record->record_bytes,
                           __ATOMIC_RELAXED);
        return 0;
    }

    int weight_direct = direct_available && !v4_read_direct_window(
        state, record->shard, slot->slab, record->weight_offset,
        (size_t)record->weight_bytes, (size_t)record->scale_bytes);
    if (weight_direct) {
        __atomic_fetch_add(&v4_direct_reads, UINT64_C(1), __ATOMIC_RELAXED);
        __atomic_fetch_add(&v4_direct_payload_bytes, record->weight_bytes,
                           __ATOMIC_RELAXED);
    } else {
        if (direct_available)
            __atomic_fetch_add(&v4_direct_fallbacks, UINT64_C(1),
                               __ATOMIC_RELAXED);
        if (coli_st_read_at(state->index, record->shard,
                            record->weight_offset,
                            (size_t)record->weight_bytes,
                            slot->slab + record->scale_bytes)) return -1;
    }
    return coli_st_read_at(state->index, record->shard, record->scale_offset,
                           (size_t)record->scale_bytes, slot->slab);
}

static void hot_fill_view(ColiTensorView *view,
                          const V4ExpertRecord *record,
                          const V4ExpertSlot *slot, int matrix,
                          const V4HotPolicy *policy,
                          const V4ExpertStoreState *state) {
    fill_tensor_view(view, record, slot, matrix);
#ifdef COLI_FP4_ROWS16_KERNEL
    if (policy->packed[hot_slot_index(state, slot)]) view->block_rows = 16;
#endif
}

static int hot_pack_matrix(ColiTensorView *view, unsigned char *scratch) {
#ifndef COLI_FP4_ROWS16_KERNEL
    (void)view; (void)scratch; return -1;
#else
    unsigned char *packed_scales = scratch + view->data_bytes;
    if (coli_fp4_pack_rows16_v10(scratch, packed_scales, view)) return -1;
    memcpy((void *)view->data, scratch, view->data_bytes);
    memcpy((void *)view->scales, packed_scales, view->scale_bytes);
    return 0;
#endif
}

/* state->mutex is held and the slot has at least one reference. */
static int hot_pack_slot_locked(V4HotPolicy *policy,
                                V4ExpertStoreState *state,
                                const V4ExpertRecord *record,
                                V4ExpertSlot *slot) {
#ifndef COLI_FP4_ROWS16_KERNEL
    (void)policy; (void)state; (void)record; (void)slot; return -1;
#else
    size_t slot_index = hot_slot_index(state, slot);
    if (policy->packed[slot_index]) return 0;
    ColiTensorView gate, down, up;
    fill_tensor_view(&gate, record, slot, V4_W1);
    fill_tensor_view(&down, record, slot, V4_W2);
    fill_tensor_view(&up, record, slot, V4_W3);
    const ColiTensorView *views[3] = {&gate, &down, &up};
    size_t scratch_size = 0;
    for (int i = 0; i < 3; i++) {
        size_t needed = views[i]->data_bytes + views[i]->scale_bytes;
        if (needed > scratch_size) scratch_size = needed;
    }
    unsigned char *scratch = malloc(scratch_size);
    if (!scratch) return -1;
    int result = hot_pack_matrix(&gate, scratch) ||
                 hot_pack_matrix(&down, scratch) ||
                 hot_pack_matrix(&up, scratch);
    free(scratch);
    if (!result) {
        policy->packed[slot_index] = 1;
        policy->packed_slots++;
    }
    return result;
#endif
}

static void hot_repin_locked(V4HotPolicy *policy, V4ExpertStoreState *state,
                             int layer) {
    if (!policy || policy->pin_count < 1) return;
    uint64_t *usage = policy->usage +
        (size_t)layer * state->experts_per_layer;
    int *pins = policy->pins + (size_t)layer * policy->pin_count;
    for (int rank = 0; rank < policy->pin_count; rank++) {
        int best = -1;
        for (int expert = 0; expert < state->experts_per_layer; expert++) {
            int already = 0;
            for (int prior = 0; prior < rank; prior++)
                if (pins[prior] == expert) { already = 1; break; }
            if (!already && usage[expert] &&
                (best < 0 || usage[expert] > usage[best] ||
                 (usage[expert] == usage[best] && expert < best)))
                best = expert;
        }
        pins[rank] = best;
    }
    V4ExpertSlot *slots = layer_slots(state, layer);
    for (int i = 0; i < state->slots_per_layer; i++)
        if (slots[i].slab && slots[i].expert >= 0 &&
            hot_is_pinned(policy, layer, slots[i].expert))
            slots[i].used = ++state->clock;
    uint64_t decay_interval = policy->repin_interval * 64;
    if (decay_interval && policy->layer_requests[layer] &&
        policy->layer_requests[layer] % decay_interval == 0)
        for (int expert = 0; expert < state->experts_per_layer; expert++)
            usage[expert] = (usage[expert] + 1) / 2;
}

static int lookup_hot(ColiExpertStore *store, ColiExpertKey key,
                      ColiExpertView *view) {
    if (!store || !store->state || !view) {
        if (view) memset(view, 0, sizeof(*view));
        return -1;
    }
    V4ExpertStoreState *state = store->state;
    V4ExpertRecord *record = get_record(state, key);
    V4HotPolicy *policy = hot_find(store);
    if (!record || !policy) {
        memset(view, 0, sizeof(*view));
        return -1;
    }
    pthread_mutex_lock(&state->mutex);
    state->stats.requests++;
    policy->usage[(size_t)key.layer * state->experts_per_layer + key.expert]++;
    uint64_t layer_requests = ++policy->layer_requests[key.layer];
    if (policy->repin_interval &&
        layer_requests % policy->repin_interval == 0)
        hot_repin_locked(policy, state, key.layer);

    V4ExpertSlot *slots = layer_slots(state, key.layer);
    V4ExpertSlot *slot = NULL;
    for (int i = 0; i < state->slots_per_layer; i++) {
        if (slots[i].slab && slots[i].expert == key.expert) {
            slot = &slots[i]; slot->references++;
            state->active_leases++;
            slot->used = ++state->clock; state->stats.hits++;
            if (hot_is_pinned(policy, key.layer, key.expert))
                hot_pack_slot_locked(policy, state, record, slot);
            memset(view, 0, sizeof(*view)); view->key = key;
            hot_fill_view(&view->gate, record, slot, V4_W1, policy, state);
            hot_fill_view(&view->down, record, slot, V4_W2, policy, state);
            hot_fill_view(&view->up, record, slot, V4_W3, policy, state);
            view->lease = slot;
            pthread_mutex_unlock(&state->mutex); return 0;
        }
    }
    for (int i = 0; i < state->slots_per_layer; i++)
        if (!slots[i].references && !slots[i].slab) { slot = &slots[i]; break; }
    if (!slot)
        for (int i = 0; i < state->slots_per_layer; i++)
            if (!slots[i].references &&
                !hot_is_pinned(policy, key.layer, slots[i].expert) &&
                (!slot || slots[i].used < slot->used)) slot = &slots[i];
    if (!slot)
        for (int i = 0; i < state->slots_per_layer; i++)
            if (!slots[i].references && (!slot || slots[i].used < slot->used))
                slot = &slots[i];
    if (!slot) {
        pthread_mutex_unlock(&state->mutex);
        memset(view, 0, sizeof(*view));
        return -1;
    }
    if (!slot->slab) {
        size_t capacity = (size_t)state->record_bytes + 8192u;
        if (posix_memalign((void **)&slot->slab, 4096, capacity)) {
            slot->slab = NULL;
            pthread_mutex_unlock(&state->mutex);
            memset(view, 0, sizeof(*view));
            return -1;
        }
        slot->aligned_slab = 1;
        state->stats.resident_bytes += state->record_bytes;
    }
    policy->packed[hot_slot_index(state, slot)] = 0;
    slot->expert = -1; slot->references = 1;
    state->active_leases++;
    slot->used = ++state->clock;
    pthread_mutex_unlock(&state->mutex);
    int read_result = v4_read_expert_record(state, record, slot);
    pthread_mutex_lock(&state->mutex);
    if (read_result) {
        slot->references = 0; slot->expert = -1;
        if (state->active_leases) state->active_leases--;
        pthread_mutex_unlock(&state->mutex);
        memset(view, 0, sizeof(*view));
        return -1;
    }
    slot->expert = key.expert; slot->used = ++state->clock;
    state->stats.misses++; state->stats.bytes_read += record->record_bytes;
    if (hot_is_pinned(policy, key.layer, key.expert))
        hot_pack_slot_locked(policy, state, record, slot);
    memset(view, 0, sizeof(*view)); view->key = key;
    hot_fill_view(&view->gate, record, slot, V4_W1, policy, state);
    hot_fill_view(&view->down, record, slot, V4_W2, policy, state);
    hot_fill_view(&view->up, record, slot, V4_W3, policy, state);
    view->lease = slot;
    pthread_mutex_unlock(&state->mutex); return 0;
}

static void destroy_hot(ColiExpertStore *store) {
    pthread_mutex_lock(&hot_policies_mutex);
    V4HotPolicy **link = &hot_policies;
    while (*link && (*link)->store != store) link = &(*link)->next;
    V4HotPolicy *policy = *link;
    if (policy) *link = policy->next;
    pthread_mutex_unlock(&hot_policies_mutex);
    if (policy) {
        hot_usage_save(policy, store ? store->state : NULL);
        fprintf(stderr, "v4_rows16 packed_slots=%llu\n",
                (unsigned long long)policy->packed_slots);
        fprintf(stderr,
                "v4_direct reads=%llu flock_reads=%llu fallbacks=%llu "
                "payload_bytes=%llu\n",
                (unsigned long long)__atomic_load_n(
                    &v4_direct_reads, __ATOMIC_RELAXED),
                (unsigned long long)__atomic_load_n(
                    &v4_direct_flock_reads, __ATOMIC_RELAXED),
                (unsigned long long)__atomic_load_n(
                    &v4_direct_fallbacks, __ATOMIC_RELAXED),
                (unsigned long long)__atomic_load_n(
                    &v4_direct_payload_bytes, __ATOMIC_RELAXED));
        free(policy->history_path);
        free(policy->packed); free(policy->pins);
        free(policy->layer_requests); free(policy->usage); free(policy);
    }
    destroy(store);
}

static int hot_prewarm_history(V4HotPolicy *policy,
                               V4ExpertStoreState *state) {
    if (!policy || !state || !policy->history_seeded ||
        policy->pin_count < 1) return 0;
    size_t capacity = (size_t)state->layers * policy->pin_count;
    ColiExpertKey *keys = malloc(capacity * sizeof(*keys));
    if (!keys) return -1;
    size_t count = 0;
    for (int layer = 0; layer < state->layers; layer++) {
        hot_repin_locked(policy, state, layer);
        const int *pins = policy->pins + (size_t)layer * policy->pin_count;
        for (int rank = 0; rank < policy->pin_count; rank++)
            if (pins[rank] >= 0)
                keys[count++] = (ColiExpertKey){layer, pins[rank]};
    }

    double began = hot_now();
    int warmed = 0;
    uint64_t repin_interval = policy->repin_interval;
    policy->repin_interval = 0;
    #pragma omp parallel for schedule(dynamic, 1) reduction(+:warmed)
    for (size_t i = 0; i < count; i++) {
        ColiExpertView view;
        if (!lookup_hot(policy->store, keys[i], &view)) {
            release(policy->store, &view);
            warmed++;
        }
    }
    policy->repin_interval = repin_interval;
    free(keys);

    /* Warmup traffic must not dilute request hit-rate telemetry. */
    pthread_mutex_lock(&state->mutex);
    uint64_t resident = state->stats.resident_bytes;
    uint64_t capacity_bytes = state->stats.capacity_bytes;
    memset(&state->stats, 0, sizeof(state->stats));
    state->stats.resident_bytes = resident;
    state->stats.capacity_bytes = capacity_bytes;
    memset(policy->layer_requests, 0,
           (size_t)state->layers * sizeof(*policy->layer_requests));
    pthread_mutex_unlock(&state->mutex);
    fprintf(stderr,
            "v4_autopin warmed=%d requested=%zu bytes=%.3fGiB time=%.3fs\n",
            warmed, count, resident / 1073741824.0, hot_now() - began);
    return warmed == (int)count ? 0 : -1;
}

#ifndef COLI_V4_ROWS16_STORE_OPEN
#define COLI_V4_ROWS16_STORE_OPEN coli_deepseek_v4_expert_store_open
#endif

int COLI_V4_ROWS16_STORE_OPEN(
    const ColiDeepSeekV4ExpertStoreOptions *options, ColiExpertStore **output,
    char *error, size_t error_size) {
    static const ColiExpertStoreOps hot_operations = {
        lookup_hot, release, prefetch, stats, destroy_hot
    };
    int result = coli_deepseek_v4_expert_store_open_base(
        options, output, error, error_size);
    if (result) return result;
    V4ExpertStoreState *state = (*output)->state;
    int direct_io = state->layers > 0 && state->experts_per_layer > 0 &&
        coli_st_streaming_direct_available(
            state->index, state->records[0].shard);
    fprintf(stderr, "v4_ssd_io mode=%s fallback=buffered-pread\n",
            direct_io ? "direct-aligned" : "buffered-pread");
    int minimum_slots = state->experts_per_layer < 6
        ? state->experts_per_layer : 6;
    int maximum_pins = state->slots_per_layer - minimum_slots;
#ifndef COLI_V4_MAX_PIN_SLOTS_PER_LAYER
#define COLI_V4_MAX_PIN_SLOTS_PER_LAYER 4
#endif
    if (maximum_pins > COLI_V4_MAX_PIN_SLOTS_PER_LAYER)
        maximum_pins = COLI_V4_MAX_PIN_SLOTS_PER_LAYER;
    int pin_requested = options->pin_slots_per_layer;
    /* -1 / 0 => implementation default (use maximum_pins). */
    uint64_t requested = pin_requested > 0
        ? (uint64_t)pin_requested
        : (uint64_t)(maximum_pins > 0 ? maximum_pins : 0);
    int pin_count = requested > (uint64_t)maximum_pins
        ? maximum_pins : (int)requested;
    if (pin_count < 0) pin_count = 0;
    V4HotPolicy *policy = calloc(1, sizeof(*policy));
    size_t records = (size_t)state->layers * state->experts_per_layer;
    size_t pins = (size_t)state->layers * (pin_count ? pin_count : 1);
    size_t slots = (size_t)state->layers * state->slots_per_layer;
    if (policy) policy->usage = calloc(records, sizeof(*policy->usage));
    if (policy) policy->layer_requests = calloc(
        (size_t)state->layers, sizeof(*policy->layer_requests));
    if (policy) policy->pins = malloc(pins * sizeof(*policy->pins));
    if (policy) policy->packed = calloc(slots, sizeof(*policy->packed));
    if (!policy || !policy->usage || !policy->layer_requests ||
        !policy->pins || !policy->packed) {
        free(policy ? policy->packed : NULL); free(policy ? policy->pins : NULL);
        free(policy ? policy->layer_requests : NULL);
        free(policy ? policy->usage : NULL); free(policy);
        destroy(*output); *output = NULL;
        return set_error(error, error_size, "out of memory creating hot policy");
    }
    for (size_t i = 0; i < pins; i++) policy->pins[i] = -1;
    policy->store = *output; policy->pin_count = pin_count;
    policy->history_path = hot_history_path(options->model_dir);
    policy->repin_interval = options->repin_interval
        ? options->repin_interval : (uint64_t)minimum_slots;
    if (!policy->repin_interval) policy->repin_interval = 1;
    const char *autopin = getenv("COLI_V4_AUTOPIN");
    if (!autopin || atoi(autopin) != 0) {
        policy->history_total = hot_usage_load(policy, state,
                                                options->model_dir);
        /* A tiny history is less predictive than the adaptive LRU. */
        policy->history_seeded = policy->history_total >= 5000;
        if (policy->history_seeded)
            for (int layer = 0; layer < state->layers; layer++)
                hot_repin_locked(policy, state, layer);
    }
    pthread_mutex_lock(&hot_policies_mutex);
    policy->next = hot_policies; hot_policies = policy;
    pthread_mutex_unlock(&hot_policies_mutex);
    (*output)->ops = &hot_operations;
    fprintf(stderr,
            "v4_hot_policy pin_slots_per_layer=%d repin_interval=%llu "
            "mode=resident-ram rows16=hot-pins\n", pin_count,
            (unsigned long long)policy->repin_interval);
    const char *prewarm = getenv("COLI_V4_PREWARM");
    if (policy->history_seeded && prewarm && atoi(prewarm) != 0 &&
        hot_prewarm_history(policy, state))
        fprintf(stderr, "v4_autopin warning=partial-warmup; continuing\n");
    return 0;
}
#endif /* COLI_V4_UNIT_EXPERT_STORE_HOT_ROWS16 */

#ifdef COLI_V4_UNIT_EXPERT_ROWS16
/* ######## deepseek_v4_expert_rows16.c ######## */
#define coli_v4_expert_forward_ref coli_v4_expert_forward_v17_fallback
/* ---- begin include deepseek_v4_expert_dual.c ---- */
#include "deepseek_v4_internal.h"

#include <stdlib.h>

#include "deepseek_v4_internal.h"
#include "native_quant.h"
#include "native_quant_dual.h"

int coli_v4_expert_forward_ref(float *output, const ColiExpertView *expert,
                               const float *input, float route_weight,
                               float swiglu_limit) {
    if (!output || !expert || !input || swiglu_limit < 0.0f ||
        expert->gate.rows != expert->up.rows ||
        expert->gate.columns != expert->up.columns ||
        expert->down.columns != expert->gate.rows ||
        expert->down.rows != expert->gate.columns) return -1;
    size_t intermediate = (size_t)expert->gate.rows;
    size_t output_size = (size_t)expert->down.rows;
    float *gate = malloc(intermediate * sizeof(*gate));
    float *up = malloc(intermediate * sizeof(*up));
    float *activated = malloc(intermediate * sizeof(*activated));
    if (!gate || !up || !activated) {
        free(activated); free(up); free(gate); return -1;
    }
    int result = coli_fp4_dual_matvec_ref(
        gate, up, &expert->gate, &expert->up, input);
    if (!result) {
        coli_bf16_round_array(gate, intermediate);
        coli_bf16_round_array(up, intermediate);
        result = coli_v4_swiglu(activated, gate, up,
                                (int)intermediate, swiglu_limit);
    }
    if (!result) {
        for (size_t i = 0; i < intermediate; i++)
            activated[i] = coli_bf16_round(activated[i] * route_weight);
        result = coli_fp4_matvec_ref(output, &expert->down, activated);
    }
    if (!result) coli_bf16_round_array(output, output_size);
    free(activated); free(up); free(gate);
    return result ? -1 : 0;
}

int coli_v4_shared_expert_forward_ref(float *output,
                                      const ColiTensorView *gate_weight,
                                      const ColiTensorView *down_weight,
                                      const ColiTensorView *up_weight,
                                      const float *input,
                                      float swiglu_limit) {
    if (!output || !gate_weight || !down_weight || !up_weight || !input ||
        gate_weight->format != COLI_TENSOR_FP8_E4M3_BLOCK ||
        down_weight->format != COLI_TENSOR_FP8_E4M3_BLOCK ||
        up_weight->format != COLI_TENSOR_FP8_E4M3_BLOCK ||
        gate_weight->rows != up_weight->rows ||
        gate_weight->columns != up_weight->columns ||
        down_weight->columns != gate_weight->rows ||
        down_weight->rows != gate_weight->columns) return -1;
    size_t intermediate = (size_t)gate_weight->rows;
    size_t output_size = (size_t)down_weight->rows;
    float *gate = malloc(intermediate * sizeof(*gate));
    float *up = malloc(intermediate * sizeof(*up));
    float *activated = malloc(intermediate * sizeof(*activated));
    if (!gate || !up || !activated) {
        free(activated); free(up); free(gate); return -1;
    }
    int result = coli_fp8_dual_matvec_ref(
        gate, up, gate_weight, up_weight, input);
    if (!result) {
        coli_bf16_round_array(gate, intermediate);
        coli_bf16_round_array(up, intermediate);
        result = coli_v4_swiglu(activated, gate, up,
                                (int)intermediate, swiglu_limit);
    }
    if (!result) {
        coli_bf16_round_array(activated, intermediate);
        result = coli_fp8_matvec_ref(output, down_weight, activated);
    }
    if (!result) coli_bf16_round_array(output, output_size);
    free(activated); free(up); free(gate);
    return result ? -1 : 0;
}
/* ---- end include deepseek_v4_expert_dual.c ---- */

#undef coli_v4_expert_forward_ref

#include "native_quant_fp4_rows16.h"


int coli_v4_expert_forward_ref(float *output, const ColiExpertView *expert,
                               const float *input, float route_weight,
                               float swiglu_limit) {
#ifndef COLI_FP4_ROWS16_KERNEL
    return coli_v4_expert_forward_v17_fallback(
        output, expert, input, route_weight, swiglu_limit);
#else
    if (!expert || expert->gate.block_rows != 16 ||
        expert->down.block_rows != 16 || expert->up.block_rows != 16)
        return coli_v4_expert_forward_v17_fallback(
            output, expert, input, route_weight, swiglu_limit);
    if (!output || !input || swiglu_limit < 0.0f) return -1;
    size_t intermediate = (size_t)expert->gate.rows;
    size_t output_size = (size_t)expert->down.rows;
    float *gate = malloc(intermediate * sizeof(*gate));
    float *up = malloc(intermediate * sizeof(*up));
    float *activated = malloc(intermediate * sizeof(*activated));
    if (!gate || !up || !activated) {
        free(activated); free(up); free(gate); return -1;
    }
    int result = coli_fp4_dual_matvec_rows16_v10(
        gate, up, &expert->gate, &expert->up, input);
    if (!result) {
        coli_bf16_round_array(gate, intermediate);
        coli_bf16_round_array(up, intermediate);
        result = coli_v4_swiglu(activated, gate, up,
                                (int)intermediate, swiglu_limit);
    }
    if (!result) {
        for (size_t i = 0; i < intermediate; i++)
            activated[i] = coli_bf16_round(activated[i] * route_weight);
        result = coli_fp4_matvec_rows16_v10(
            output, &expert->down, activated);
    }
    if (!result) coli_bf16_round_array(output, output_size);
    free(activated); free(up); free(gate);
    return result ? -1 : 0;
#endif
}
#endif /* COLI_V4_UNIT_EXPERT_ROWS16 */

#ifdef COLI_V4_UNIT_ROUTE_BF16
/* ######## deepseek_v4_route_bf16.c ######## */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static float route_bf16_decode(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float output;
    memcpy(&output, &bits, sizeof(output));
    return output;
}

static float route_softplus(float value) {
    return fmaxf(value, 0.0f) + log1pf(expf(-fabsf(value)));
}

int coli_v4_route_bf16(float *weights, int *indices, const float *hidden,
                       const uint16_t *gate, const float *bias,
                       const int *forced_indices, int experts, int dimension,
                       int topk, float route_scale) {
    if (!weights || !indices || !hidden || !gate || experts < 1 ||
        dimension < 1 || topk < 1 || topk > experts) return -1;
    float *scores = malloc((size_t)experts * sizeof(*scores));
    float *selection = malloc((size_t)experts * sizeof(*selection));
    unsigned char *selected = calloc((size_t)experts, 1);
    if (!scores || !selection || !selected) {
        free(selected); free(selection); free(scores); return -1;
    }
    for (int expert = 0; expert < experts; expert++) {
        float sum = 0.0f;
        const uint16_t *row = gate + (size_t)expert * dimension;
        for (int column = 0; column < dimension; column++)
            sum += route_bf16_decode(row[column]) * hidden[column];
        scores[expert] = sqrtf(route_softplus(sum));
        selection[expert] = scores[expert] + (bias ? bias[expert] : 0.0f);
    }
    if (forced_indices) {
        for (int rank = 0; rank < topk; rank++) {
            if (forced_indices[rank] < 0 || forced_indices[rank] >= experts) {
                free(selected); free(selection); free(scores); return -1;
            }
            indices[rank] = forced_indices[rank];
        }
    } else {
        for (int rank = 0; rank < topk; rank++) {
            int best = -1;
            for (int expert = 0; expert < experts; expert++)
                if (!selected[expert] &&
                    (best < 0 || selection[expert] > selection[best]))
                    best = expert;
            indices[rank] = best;
            selected[best] = 1;
        }
    }
    float total = 0.0f;
    for (int rank = 0; rank < topk; rank++)
        total += scores[indices[rank]];
    if (!(total > 0.0f)) {
        free(selected); free(selection); free(scores); return -1;
    }
    for (int rank = 0; rank < topk; rank++)
        weights[rank] = scores[indices[rank]] / total * route_scale;
    free(selected); free(selection); free(scores);
    return 0;
}
#endif /* COLI_V4_UNIT_ROUTE_BF16 */

#ifdef COLI_V4_UNIT_RUNTIME
/* ######## deepseek_v4_runtime.c / engine ######## */
#include "deepseek_v4_internal.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Provided by LAYER_RESIDENT. */
void coli_v4_layer_resident_reference_free(ColiV4Engine *engine,
                                           ColiDeepSeekV4LayerWeights *weights);

#ifdef COLI_V4_TEST_HOOKS
int coli_v4_test_fail_expert_store_open = 0;
int coli_v4_test_skip_expert_store_open = 0;
int coli_v4_test_closed_owned_index = 0;
#endif

void coli_v4_engine_attach_session(ColiV4Engine *engine) {
    if (engine) engine->active_sessions++;
}

void coli_v4_engine_detach_session(ColiV4Engine *engine) {
    if (!engine) return;
    assert(engine->active_sessions > 0);
    engine->active_sessions--;
}

int coli_v4_full_dspark_wanted;

double coli_v4_dspark_cache_gb(void) {
    const char *setting = getenv("V4_MTP_GB");
    double value = setting ? atof(setting) : 0.45;
    if (value < 0.15) value = 0.15;
    /* The target cache is reduced by the same reservation, but keep a hard
     * ceiling so a typo cannot turn a lazy drafter into an OOM request. */
    if (value > 4.0) value = 4.0;
    return value;
}

static int v4_dspark_full_wanted(
    const ColiV4EngineOpenOptions *options) {
    if (!options || options->no_dspark) return 0;
    const char *mtp = getenv("V4_MTP");
    const char *draft = getenv("V4_DRAFT");
    return mtp && atoi(mtp) != 0 && draft && atoi(draft) > 0;
}

/* The native full drafter is a three-stage DSpark profile.  The ordinary
 * Flash checkpoint exposes only the standard MTP tensors, while the official
 * DSpark supplement keeps num_nextn_predict_layers=1 and adds these markers.
 * The full loader remains the authority for detailed dtype/shape validation;
 * this cheap probe only decides whether its cache should enter the plan. */
enum {
    V4_DSPARK_PROFILE_BLOCK = 5,
    V4_DSPARK_PROFILE_RANK = 256,
};

static int v4_dspark_full_profile_present(
    const ColiSafetensorsIndex *index,
    const ColiDeepSeekV4Config *config) {
    if (!index || !config) return 0;

    /* Converted checkpoints may omit these optional fields, but an explicit
     * incompatible profile must not reserve memory for this fixed loader. */
    if ((config->dspark_block_size > 0 &&
         config->dspark_block_size != V4_DSPARK_PROFILE_BLOCK) ||
        (config->dspark_markov_rank > 0 &&
         config->dspark_markov_rank != V4_DSPARK_PROFILE_RANK))
        return 0;

    /* Cover the entry projection, middle and final stages, and the distinctive
     * output heads. v4_ds_load_all() performs the complete validation later. */
    static const char *profile_tensors[] = {
        "mtp.0.main_proj.weight", "mtp.0.main_proj.scale",
        "mtp.1.attn.wq_a.weight", "mtp.1.attn.wq_a.scale",
        "mtp.2.attn.wq_a.weight", "mtp.2.attn.wq_a.scale",
        "mtp.2.confidence_head.proj.weight",
        "mtp.2.markov_head.markov_w1.weight",
        "mtp.2.markov_head.markov_w2.weight",
    };
    for (size_t item = 0;
         item < sizeof(profile_tensors) / sizeof(profile_tensors[0]); item++)
        if (!coli_st_find(index, profile_tensors[item])) return 0;
    return 1;
}

static uint64_t v4_dspark_full_reserve_bytes(void) {
    double cache = coli_v4_dspark_cache_gb() * 1e9;
    /* The released Flash checkpoint has ~0.56 GiB of resident MTP dense,
     * projection, Markov and norm tensors.  Add head/scratch/tap margin. */
    double total = cache + 768.0 * 1024.0 * 1024.0;
    return total >= (double)UINT64_MAX ? UINT64_MAX : (uint64_t)total;
}

/* Lightweight fallback: keep only the official low-rank Markov head resident
 * when explicitly requested without the complete three-stage drafter. */
static int v4_dspark_markov_probe(ColiV4Engine *engine,
                                  const ColiSafetensorsTensor **w1_out,
                                  const ColiSafetensorsTensor **w2_out) {
    if (!engine || !engine->target_index || !w1_out || !w2_out) return -1;
    *w1_out = NULL;
    *w2_out = NULL;
    char name[160];
    for (int stage = 0; stage < 16; stage++) {
        snprintf(name, sizeof(name),
                 "mtp.%d.markov_head.markov_w1.weight", stage);
        const ColiSafetensorsTensor *w1 = coli_st_find(engine->target_index,
                                                       name);
        snprintf(name, sizeof(name),
                 "mtp.%d.markov_head.markov_w2.weight", stage);
        const ColiSafetensorsTensor *w2 = coli_st_find(engine->target_index,
                                                       name);
        if (!w1 || !w2) continue;
        int rank = engine->config.dspark_markov_rank;
        if (w1->dtype != COLI_ST_BF16 || w2->dtype != COLI_ST_BF16 ||
            w1->rank != 2 || w2->rank != 2 || rank < 1 ||
            w1->shape[0] != engine->config.vocab_size ||
            w2->shape[0] != engine->config.vocab_size ||
            w1->shape[1] != rank || w2->shape[1] != rank ||
            w1->nbytes <= 0 || w2->nbytes <= 0)
            return -1;
        engine->dspark.rank = rank;
        /* A Markov-only draft is intentionally conservative.  Two tokens cap
         * rollback cost on prose where the low-rank bigram is uncertain; a
         * benchmarker may raise it up to the checkpoint's trained block. */
        engine->dspark.block_size = 2;
        const char *block_setting = getenv("COLI_V4_MARKOV_BLOCK");
        if (block_setting && atoi(block_setting) > 1)
            engine->dspark.block_size = atoi(block_setting);
        if (engine->dspark.block_size > engine->config.dspark_block_size)
            engine->dspark.block_size = engine->config.dspark_block_size;
        if (engine->dspark.block_size > 8) engine->dspark.block_size = 8;
        if (engine->dspark.block_size < 2) return -1;
        engine->dspark.stage = stage;
        engine->dspark.bytes = (uint64_t)w1->nbytes + (uint64_t)w2->nbytes;
        *w1_out = w1;
        *w2_out = w2;
        return 0;
    }
    return -1;
}

static int v4_dspark_markov_wanted(const ColiV4EngineOpenOptions *options) {
    if (!options || options->no_dspark) return 0;
    const char *setting = getenv("COLI_V4_MARKOV_SPEC");
    /* Opt-in while real-hardware acceptance is being measured. */
    return setting && atoi(setting) != 0;
}

static int v4_dspark_markov_load(ColiV4Engine *engine,
                                 const ColiSafetensorsTensor *w1,
                                 const ColiSafetensorsTensor *w2) {
    if (!engine || !w1 || !w2) return -1;
    engine->dspark.markov_w1 = malloc((size_t)w1->nbytes);
    engine->dspark.markov_w2 = malloc((size_t)w2->nbytes);
    if (!engine->dspark.markov_w1 || !engine->dspark.markov_w2 ||
        coli_st_read_tensor(engine->target_index, w1,
                            engine->dspark.markov_w1) ||
        coli_st_read_tensor(engine->target_index, w2,
                            engine->dspark.markov_w2)) {
        free(engine->dspark.markov_w2);
        free(engine->dspark.markov_w1);
        engine->dspark.markov_w2 = NULL;
        engine->dspark.markov_w1 = NULL;
        engine->dspark.bytes = 0;
        return -1;
    }
    engine->dspark.enabled = 1;
    fprintf(stderr,
            "v4_dspark mode=markov-draft stage=%d block=%d rank=%d "
            "resident=%.3fGiB verification=exact-target\n",
            engine->dspark.stage, engine->dspark.block_size,
            engine->dspark.rank, engine->dspark.bytes / 1073741824.0);
    return 0;
}

#ifdef COLI_V4_TEST_HOOKS
ColiV4Session *coli_v4_test_session_bare_create(ColiV4Engine *engine) {
    if (!engine) return NULL;
    ColiV4Session *session = calloc(1, sizeof(*session));
    if (!session) return NULL;
    session->engine = engine;
    coli_v4_engine_attach_session(engine);
    return session;
}

void coli_v4_test_session_bare_destroy(ColiV4Session *session) {
    if (!session) return;
    if (session->engine) {
        coli_v4_engine_detach_session(session->engine);
        session->engine = NULL;
    }
    free(session);
}

#endif

const ColiDeepSeekV4Config *coli_v4_engine_config(const ColiV4Engine *engine) {
    return engine ? &engine->config : NULL;
}

ColiSafetensorsIndex *coli_v4_engine_target_index(ColiV4Engine *engine) {
    return engine ? engine->target_index : NULL;
}

ColiExpertStore *coli_v4_engine_expert_store(ColiV4Engine *engine) {
    return engine ? engine->experts : NULL;
}

void coli_v4_engine_memory_summary(const ColiV4Engine *engine,
                                   ColiV4EngineMemorySummary *summary) {
    if (!summary) return;
    if (!engine) {
        memset(summary, 0, sizeof(*summary));
        return;
    }
    *summary = engine->summary;
}

const char *coli_v4_engine_target_model_dir(const ColiV4Engine *engine) {
    return engine ? engine->runtime.target_model_dir : NULL;
}

void coli_v4_engine_destroy(ColiV4Engine *engine) {
    if (!engine) return;
    assert(engine->active_sessions == 0 &&
           "destroy engine while sessions are still alive");

    for (int layer = 0; layer < COLI_V4_RESIDENT_MAX_LAYERS; layer++) {
        if (!engine->dense_resident.ready[layer]) continue;
        coli_v4_layer_resident_reference_free(
            NULL, &engine->dense_resident.layers[layer]);
        engine->dense_resident.ready[layer] = 0;
    }
    engine->dense_resident.index = NULL;
    engine->dense_resident.total_bytes = 0;

    free(engine->dspark.markov_w2);
    free(engine->dspark.markov_w1);
    engine->dspark.markov_w2 = NULL;
    engine->dspark.markov_w1 = NULL;
    engine->dspark.enabled = 0;
    free(engine->head_cache.data);
    engine->head_cache.data = NULL;
    if (engine->owns_experts && engine->experts && engine->experts->ops &&
        engine->experts->ops->destroy)
        engine->experts->ops->destroy(engine->experts);
    engine->experts = NULL;
    if (engine->owns_index && engine->target_index) {
#ifdef COLI_V4_TEST_HOOKS
        coli_v4_test_closed_owned_index++;
#endif
        coli_st_index_close(engine->target_index);
    }
    engine->target_index = NULL;
    engine->runtime.target_model_dir = NULL;
    free(engine->owned_target_model_dir);
    engine->owned_target_model_dir = NULL;
    free(engine);
}

int coli_v4_engine_open(ColiV4Engine **output,
                        const ColiV4EngineOpenOptions *options,
                        char *error, size_t error_size) {
    if (!output || !options || !options->target_model_dir) {
        if (error && error_size)
            snprintf(error, error_size, "invalid V4 engine open options");
        return -1;
    }
    *output = NULL;
    ColiV4Engine *engine = calloc(1, sizeof(*engine));
    if (!engine) {
        if (error && error_size)
            snprintf(error, error_size, "out of memory creating V4 engine");
        return -1;
    }

    engine->owned_target_model_dir = strdup(options->target_model_dir);
    if (!engine->owned_target_model_dir) {
        if (error && error_size)
            snprintf(error, error_size, "out of memory copying model directory");
        goto fail;
    }
    engine->runtime.target_model_dir = engine->owned_target_model_dir;
    engine->runtime.memory_limit_bytes = options->memory_limit_bytes;
    engine->runtime.context_tokens =
        options->context_tokens > 0 ? options->context_tokens : 4096;
    engine->runtime.repin_interval = options->repin_interval;
    engine->runtime.pin_slots_per_layer = options->pin_slots_per_layer;

    /* Set owns_* immediately after each acquire so destroy cleans partial opens. */
    if (coli_v4_config_load(&engine->config, engine->runtime.target_model_dir,
                            error, error_size))
        goto fail;
    if (coli_st_index_open(&engine->target_index,
                           engine->runtime.target_model_dir, error,
                           error_size))
        goto fail;
    engine->owns_index = 1;
    const ColiSafetensorsTensor *dspark_w1 = NULL, *dspark_w2 = NULL;
    int requested_full_dspark = v4_dspark_full_wanted(options);
    int want_full_dspark = requested_full_dspark &&
                           v4_dspark_full_profile_present(
                               engine->target_index, &engine->config);
    if (requested_full_dspark && !want_full_dspark)
        fprintf(stderr,
                "v4_dspark warning=unsupported-checkpoint "
                "expected_full_profile=3stage actual_mtp_layers=%d; "
                "continuing-target-only\n",
                engine->config.num_nextn_predict_layers);
    int want_dspark = !want_full_dspark && v4_dspark_markov_wanted(options);
    if (want_dspark && v4_dspark_markov_probe(engine, &dspark_w1,
                                               &dspark_w2)) {
        fprintf(stderr,
                "v4_dspark warning=compatible-markov-head-not-found; "
                "continuing-target-only\n");
        want_dspark = 0;
    }
    engine->runtime.dspark_reserve_bytes = want_full_dspark
        ? v4_dspark_full_reserve_bytes()
        : (want_dspark ? engine->dspark.bytes : 0);
    coli_v4_full_dspark_wanted = want_full_dspark;
    if (want_full_dspark)
        fprintf(stderr,
                "v4_dspark mode=full-3stage cache=%.2fGB reserve=%.2fGiB "
                "load=lazy verification=exact-target\n",
                coli_v4_dspark_cache_gb(),
                engine->runtime.dspark_reserve_bytes / 1073741824.0);
#ifdef COLI_V4_TEST_HOOKS
    if (coli_v4_test_fail_expert_store_open) {
        if (error && error_size)
            snprintf(error, error_size, "forced expert store open failure");
        goto fail;
    }
    if (coli_v4_test_skip_expert_store_open) {
        engine->summary.dense_resident = engine->runtime.dense_resident;
        engine->summary.head_resident = engine->head_cache.data != NULL;
        engine->summary.expert_cache_bytes =
            engine->runtime.target_expert_cache_bytes;
        *output = engine;
        return 0;
    }
#endif
    if (coli_v4_expert_store_open_planned(
            engine,
            &(ColiDeepSeekV4ExpertStoreOptions){
                engine->runtime.target_model_dir,
                engine->config.num_hidden_layers,
                engine->config.n_routed_experts,
                4ULL << 30,
                engine->runtime.pin_slots_per_layer,
                engine->runtime.repin_interval},
            &engine->experts, error, error_size))
        goto fail;
    engine->owns_experts = 1;
    if (want_dspark && v4_dspark_markov_load(engine, dspark_w1, dspark_w2))
        fprintf(stderr,
                "v4_dspark warning=cannot-load-markov-head; "
                "continuing-target-only\n");
    engine->summary.dense_resident = engine->runtime.dense_resident;
    engine->summary.head_resident = engine->head_cache.data != NULL;
    engine->summary.expert_cache_bytes =
        engine->runtime.target_expert_cache_bytes;
    *output = engine;
    return 0;

fail:
    coli_v4_engine_destroy(engine);
    return -1;
}
#endif /* COLI_V4_UNIT_RUNTIME */

#ifdef COLI_V4_UNIT_PROMPT
/* ######## deepseek_v4_prompt.c ######## */
#include "deepseek_v4_internal.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static const char v4_bos[] = "<｜begin▁of▁sentence｜>";
static const char v4_user[] = "<｜User｜>";
static const char v4_assistant[] = "<｜Assistant｜>";

static int add_size(size_t *total, size_t value) {
    if (SIZE_MAX - *total < value) return -1;
    *total += value;
    return 0;
}

int coli_v4_prompt_build(char **output, size_t *output_length,
                         const char *user_message, const char *system_message,
                         ColiDeepSeekV4PromptMode mode) {
    if (!output || !user_message || mode < COLI_V4_PROMPT_CHAT ||
        mode > COLI_V4_PROMPT_RAW) return -1;
    *output = NULL;
    if (output_length) *output_length = 0;
    const char *system = system_message ? system_message : "";
    if (mode == COLI_V4_PROMPT_RAW) {
        size_t length = strlen(user_message);
        char *copy = malloc(length + 1);
        if (!copy) return -1;
        memcpy(copy, user_message, length + 1);
        *output = copy;
        if (output_length) *output_length = length;
        return 0;
    }
    const char *thinking = mode == COLI_V4_PROMPT_THINKING
        ? "<think>" : "</think>";
    size_t length = 0;
    if (add_size(&length, strlen(v4_bos)) ||
        add_size(&length, strlen(system)) ||
        add_size(&length, strlen(v4_user)) ||
        add_size(&length, strlen(user_message)) ||
        add_size(&length, strlen(v4_assistant)) ||
        add_size(&length, strlen(thinking)) || length == SIZE_MAX)
        return -1;
    char *prompt = malloc(length + 1);
    if (!prompt) return -1;
    char *at = prompt;
#define APPEND(part) do { \
    size_t count = strlen(part); memcpy(at, part, count); at += count; \
} while (0)
    APPEND(v4_bos);
    APPEND(system);
    APPEND(v4_user);
    APPEND(user_message);
    APPEND(v4_assistant);
    APPEND(thinking);
#undef APPEND
    *at = '\0';
    *output = prompt;
    if (output_length) *output_length = length;
    return 0;
}
#endif /* COLI_V4_UNIT_PROMPT */

#ifdef COLI_V4_UNIT_GENERATE_STATS
/* ######## tools/deepseek_v4_generate_stats.c ######## */
#define COLI_V4_GENERATE_MAIN coli_v4_generate_stats_legacy_main
#define COLI_V4_GENERATE_HELPERS_ONLY
#define spec_print spec_print_diagnostic_legacy
/* Target-only generation helpers. */
#include <time.h>
#if defined(__APPLE__) || defined(__linux__) || defined(__FreeBSD__)
#include <sys/resource.h>                         /* getrusage/RUSAGE_SELF for v4_serve_rss_gb;
                                                   * on Windows compat.h supplies the shim. */
#endif

#define main coli_v4_first_token_legacy_main
/* ---- begin include tools/deepseek_v4_first_token.c ---- */
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef __AVX2__
#include <immintrin.h>
#endif

#include "deepseek_v4_internal.h"
#include "deepseek_v4_internal.h"
#include "deepseek_v4_internal.h"
#include "deepseek_v4_internal.h"
#include "deepseek_v4_internal.h"
#include "json.h"
#include "native_quant.h"
#include "tok.h"

static int load_embedding(float *state, const ColiSafetensorsIndex *index,
                          const ColiDeepSeekV4Config *config, int token) {
    const ColiSafetensorsTensor *embed = coli_st_find(index, "embed.weight");
    int d = config->hidden_size, hc = config->hc_mult;
    int shard = coli_st_tensor_shard(index, embed);
    uint16_t *row = malloc((size_t)d * sizeof(*row));
    if (!embed || embed->dtype != COLI_ST_BF16 || !row || token < 0 ||
        token >= config->vocab_size ||
        coli_st_read_at(index, shard,
                        (uint64_t)embed->off + (uint64_t)token * d * sizeof(*row),
                        (size_t)d * sizeof(*row), row)) {
        free(row);
        return -1;
    }
    for (int copy = 0; copy < hc; copy++)
        for (int i = 0; i < d; i++)
            state[(size_t)copy * d + i] = coli_bf16_decode(row[i]);
    free(row);
    return 0;
}

static int final_hidden(float *output, const float *state,
                        const ColiSafetensorsIndex *index,
                        const ColiDeepSeekV4Config *config,
                        char *error, size_t error_size) {
    ColiFloatTensor function = {0}, base = {0}, scale = {0}, norm = {0};
    if (coli_tensor_load_f32(&function, index, "hc_head_fn", error, error_size) ||
        coli_tensor_load_f32(&base, index, "hc_head_base", error, error_size) ||
        coli_tensor_load_f32(&scale, index, "hc_head_scale", error, error_size) ||
        coli_tensor_load_f32(&norm, index, "norm.weight", error, error_size))
        return -1;
    int d = config->hidden_size, hc = config->hc_mult;
    int flattened = hc * d;
    float square = 0.0f;
    for (int i = 0; i < flattened; i++) square += state[i] * state[i];
    float inverse_rms = 1.0f / sqrtf(square / flattened + config->rms_norm_eps);
    float pre[16];
    if (hc > 16) return -1;
    for (int copy = 0; copy < hc; copy++) {
        float mix = 0.0f;
        for (int i = 0; i < flattened; i++)
            mix += function.data[(size_t)copy * flattened + i] * state[i];
        mix *= inverse_rms;
        float z = mix * scale.data[0] + base.data[copy];
        float sigmoid = z >= 0.0f
            ? 1.0f / (1.0f + expf(-z))
            : expf(z) / (1.0f + expf(z));
        pre[copy] = sigmoid + config->hc_eps;
    }
    for (int i = 0; i < d; i++) {
        float value = 0.0f;
        for (int copy = 0; copy < hc; copy++)
            value += pre[copy] * state[(size_t)copy * d + i];
        output[i] = coli_bf16_round(value);
    }
    coli_v4_rmsnorm(output, output, norm.data, d, config->rms_norm_eps);
    coli_bf16_round_array(output, (size_t)d);
    coli_float_tensor_free(&norm);
    coli_float_tensor_free(&scale);
    coli_float_tensor_free(&base);
    coli_float_tensor_free(&function);
    return 0;
}

static float head_bf16_dot(const uint16_t *weight, const float *hidden,
                           int dimension) {
    float sum = 0.0f;
    int column = 0;
#ifdef __AVX2__
    for (; column + 8 <= dimension; column += 8) {
        float products[8];
        __m128i packed = _mm_loadu_si128(
            (const __m128i *)(weight + column));
        __m256i bits = _mm256_slli_epi32(
            _mm256_cvtepu16_epi32(packed), 16);
        _mm256_storeu_ps(products, _mm256_mul_ps(
            _mm256_castsi256_ps(bits),
            _mm256_loadu_ps(hidden + column)));
        sum += products[0];
        sum += products[1];
        sum += products[2];
        sum += products[3];
        sum += products[4];
        sum += products[5];
        sum += products[6];
        sum += products[7];
    }
#endif
    for (; column < dimension; column++)
        sum += coli_bf16_decode(weight[column]) * hidden[column];
    return sum;
}

static int head_argmax(ColiV4Engine *engine, const float *hidden,
                       const ColiSafetensorsIndex *index,
                       const ColiDeepSeekV4Config *config,
                       int *best_token, float *best_logit) {
    const ColiSafetensorsTensor *head = coli_st_find(index, "head.weight");
    int d = config->hidden_size, vocab = config->vocab_size;
    if (!head || head->dtype != COLI_ST_BF16 || d < 1 || vocab < 1)
        return -1;
    int shard = coli_st_tensor_shard(index, head);
    size_t resident_bytes = (size_t)vocab * (size_t)d * sizeof(uint16_t);
    const uint16_t *resident = coli_v4_head_cache_data(
        engine, shard, (uint64_t)head->off, resident_bytes);

    /* The normal V4 memory plan keeps the BF16 head resident.  Compute all
     * rows in one OpenMP team directly from that allocation: the old tiled
     * path copied the complete ~1 GiB head and created ~2,000 teams per token.
     * Each row retains the same scalar accumulation order and the final scan
     * retains vocabulary order, so logits/tie-breaking do not change. */
    if (resident) {
        float *scores = malloc((size_t)vocab * sizeof(*scores));
        if (!scores) return -1;
        #pragma omp parallel for schedule(static)
        for (int row = 0; row < vocab; row++) {
            const uint16_t *weight = resident + (size_t)row * d;
            scores[row] = head_bf16_dot(weight, hidden, d);
        }
        int winner = -1;
        float maximum = -FLT_MAX;
        for (int row = 0; row < vocab; row++)
            if (scores[row] > maximum) {
                maximum = scores[row];
                winner = row;
            }
        free(scores);
        *best_token = winner;
        *best_logit = maximum;
        return winner < 0 ? -1 : 0;
    }

    /* Low-memory fallback: stream small row tiles exactly as before. */
    enum { ROWS = 64 };
    uint16_t *raw = malloc((size_t)ROWS * d * sizeof(*raw));
    float *scores = malloc((size_t)ROWS * sizeof(*scores));
    if (!raw || !scores) {
        free(scores); free(raw);
        return -1;
    }
    int winner = -1;
    float maximum = -FLT_MAX;
    for (int start = 0; start < vocab; start += ROWS) {
        int rows = vocab - start < ROWS ? vocab - start : ROWS;
        size_t bytes = (size_t)rows * d * sizeof(*raw);
        if (coli_st_read_at_engine(
                engine, index, shard,
                (uint64_t)head->off + (uint64_t)start * d * sizeof(*raw),
                bytes, raw)) {
            free(scores); free(raw);
            return -1;
        }
        #pragma omp parallel for
        for (int row = 0; row < rows; row++) {
            const uint16_t *weight = raw + (size_t)row * d;
            scores[row] = head_bf16_dot(weight, hidden, d);
        }
        for (int row = 0; row < rows; row++)
            if (scores[row] > maximum) {
                maximum = scores[row];
                winner = start + row;
            }
    }
    free(scores); free(raw);
    *best_token = winner;
    *best_logit = maximum;
    return winner < 0 ? -1 : 0;
}

static int head_argmax_batch(ColiV4Engine *engine, const float *hidden,
                             const ColiSafetensorsIndex *index,
                             const ColiDeepSeekV4Config *config, int batch,
                             int *best_tokens, float *best_logits) {
    if (!engine || !hidden || !index || !config || batch < 1 ||
        !best_tokens || !best_logits) return -1;
    if (batch == 1)
        return head_argmax(engine, hidden, index, config, best_tokens,
                           best_logits);
    const ColiSafetensorsTensor *head = coli_st_find(index, "head.weight");
    int d = config->hidden_size, vocab = config->vocab_size;
    if (!head || head->dtype != COLI_ST_BF16) return -1;
    int shard = coli_st_tensor_shard(index, head);
    const uint16_t *resident = coli_v4_head_cache_data(
        engine, shard, (uint64_t)head->off,
        (size_t)vocab * d * sizeof(uint16_t));
    if (!resident) {
        for (int item = 0; item < batch; item++)
            if (head_argmax(engine, hidden + (size_t)item * d, index, config,
                            &best_tokens[item], &best_logits[item])) return -1;
        return 0;
    }
    float *scores = malloc((size_t)vocab * batch * sizeof(*scores));
    if (!scores) return -1;
    #pragma omp parallel for schedule(static)
    for (int row = 0; row < vocab; row++) {
        const uint16_t *weight = resident + (size_t)row * d;
        for (int item = 0; item < batch; item++)
            scores[(size_t)item * vocab + row] = head_bf16_dot(
                weight, hidden + (size_t)item * d, d);
    }
    for (int item = 0; item < batch; item++) {
        int winner = -1;
        float maximum = -FLT_MAX;
        const float *item_scores = scores + (size_t)item * vocab;
        for (int row = 0; row < vocab; row++)
            if (item_scores[row] > maximum) {
                maximum = item_scores[row];
                winner = row;
            }
        best_tokens[item] = winner;
        best_logits[item] = maximum;
    }
    free(scores);
    return 0;
}

static int dspark_markov_argmax(const ColiV4Engine *engine, int token,
                                int *best_token) {
    if (!engine || !engine->dspark.enabled || !best_token || token < 0 ||
        token >= engine->config.vocab_size) return -1;
    int rank = engine->dspark.rank, vocab = engine->config.vocab_size;
    float *embedding = malloc((size_t)rank * sizeof(*embedding));
    float *scores = malloc((size_t)vocab * sizeof(*scores));
    if (!embedding || !scores) {
        free(scores); free(embedding); return -1;
    }
    const uint16_t *source = engine->dspark.markov_w1 + (size_t)token * rank;
    for (int column = 0; column < rank; column++)
        embedding[column] = coli_bf16_decode(source[column]);
    #pragma omp parallel for schedule(static)
    for (int row = 0; row < vocab; row++)
        scores[row] = head_bf16_dot(
            engine->dspark.markov_w2 + (size_t)row * rank,
            embedding, rank);
    int winner = -1;
    float maximum = -FLT_MAX;
    for (int row = 0; row < vocab; row++)
        if (scores[row] > maximum) {
            maximum = scores[row];
            winner = row;
        }
    free(scores); free(embedding);
    *best_token = winner;
    return winner < 0 ? -1 : 0;
}

static int has_sentence_end(const char *text, int length) {
    for (int i = 0; i < length; i++) {
        unsigned char value = (unsigned char)text[i];
        if (value == '.' || value == '!' || value == '?' || value == '\n') return 1;
        if (i + 2 < length && value == 0xe3 &&
            (unsigned char)text[i + 1] == 0x80 &&
            (unsigned char)text[i + 2] == 0x82) return 1; /* 。 */
        if (i + 2 < length && value == 0xef &&
            (unsigned char)text[i + 1] == 0xbc &&
            ((unsigned char)text[i + 2] == 0x81 ||
             (unsigned char)text[i + 2] == 0x9f)) return 1; /* ！？ */
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 3 || argc > 6) {
        fprintf(stderr, "usage: %s MODEL_DIR INPUT_TOKEN_ID [TOKEN_COUNT]\n"
                        "       %s MODEL_DIR --prompt TEXT [MAX_NEW_TOKENS] [--stop-sentence]\n",
                argv[0], argv[0]);
        return 2;
    }
    int text_mode = !strcmp(argv[2], "--prompt");
    if (text_mode && argc < 4) return 2;
    int stop_sentence = text_mode && argc == 6 &&
                        !strcmp(argv[5], "--stop-sentence");
    if (text_mode && argc == 6 && !stop_sentence) return 2;
    int input_token = text_mode ? -1 : atoi(argv[2]);
    int token_count = text_mode ? (argc == 5 ? atoi(argv[4]) : 32)
                                : (argc == 4 ? atoi(argv[3]) : 1);
    if (token_count < 1) return 2;
    char error[512] = {0};
    ColiDeepSeekV4Config config;
    ColiSafetensorsIndex *index = NULL;
    ColiExpertStore *experts = NULL;
    if (coli_v4_config_load(&config, argv[1], error, sizeof(error)) ||
        coli_st_index_open(&index, argv[1], error, sizeof(error)) ||
        coli_deepseek_v4_expert_store_open(
            &(ColiDeepSeekV4ExpertStoreOptions){
                argv[1], config.num_hidden_layers, config.n_routed_experts,
                UINT64_C(4) * 1024 * 1024 * 1024, -1, 0,
            }, &experts, error, sizeof(error))) {
        fprintf(stderr, "%s\n", error);
        return 1;
    }
    Tok tokenizer;
    int *prompt_ids = NULL, prompt_count = 0;
    int *generated_ids = NULL, generated_count = 0;
    if (text_mode) {
        char tokenizer_path[4096];
        snprintf(tokenizer_path, sizeof(tokenizer_path), "%s/tokenizer.json", argv[1]);
        tok_load(&tokenizer, tokenizer_path);
        int prompt_capacity = (int)strlen(argv[3]) + 16;
        prompt_ids = malloc((size_t)prompt_capacity * sizeof(*prompt_ids));
        generated_ids = malloc((size_t)token_count * sizeof(*generated_ids));
        if (!prompt_ids || !generated_ids) return 1;
        prompt_count = tok_encode(&tokenizer, argv[3], (int)strlen(argv[3]),
                                  prompt_ids, prompt_capacity);
        if (prompt_count < 1) {
            fprintf(stderr, "prompt produced no tokens\n"); return 1;
        }
        fprintf(stderr, "prompt_tokens=%d max_new_tokens=%d eos_token=1\n",
                prompt_count, token_count);
    }
    size_t state_count = (size_t)config.hc_mult * config.hidden_size;
    float *state = malloc(state_count * sizeof(*state));
    float *next = malloc(state_count * sizeof(*next));
    float *hidden = malloc((size_t)config.hidden_size * sizeof(*hidden));
    ColiDeepSeekV4WindowAttentionState **attention = calloc(
        (size_t)config.num_hidden_layers, sizeof(*attention));
    if (!state || !next || !hidden || !attention) return 1;
    for (int layer = 0; layer < config.num_hidden_layers; layer++)
        if (coli_v4_window_attention_create(&attention[layer], &config)) return 1;

    int current_token = text_mode ? prompt_ids[0] : input_token;
    int total_steps = text_mode ? prompt_count + token_count - 1 : token_count;
    for (int position = 0; position < total_steps; position++) {
        if (text_mode && position < prompt_count)
            current_token = prompt_ids[position];
        if (load_embedding(state, index, &config, current_token)) return 1;
        for (int layer_id = 0; layer_id < config.num_hidden_layers; layer_id++) {
            ColiDeepSeekV4LayerWeights layer;
            if (coli_v4_layer_load(NULL, &layer, &config, index, layer_id,
                                   error, sizeof(error)) ||
                coli_v4_block_window_token_ref(
                    next, attention[layer_id], &layer, &config, experts, state,
                    current_token, position, error, sizeof(error))) {
                fprintf(stderr, "position %d layer %d: %s\n",
                        position, layer_id, error);
                return 1;
            }
            coli_v4_layer_free(NULL, &layer);
            float *swap = state; state = next; next = swap;
        }
        fprintf(stderr, "position %d/%d complete (%d layers)\n", position,
                total_steps - 1, config.num_hidden_layers);
        if (final_hidden(hidden, state, index, &config, error, sizeof(error))) {
            fprintf(stderr, "final hidden: %s\n", error);
            return 1;
        }
        int output_token;
        float output_logit;
        if (head_argmax(NULL, hidden, index, &config, &output_token, &output_logit)) {
            fprintf(stderr, "lm_head failed\n");
            return 1;
        }
        if (!text_mode) {
            printf("position=%d input_token=%d output_token=%d logit=%.9g\n",
                   position, current_token, output_token, output_logit);
        } else if (position >= prompt_count - 1) {
            generated_ids[generated_count++] = output_token;
            char piece[1024];
            int piece_length = tok_decode(&tokenizer, &output_token, 1,
                                          piece, (int)sizeof(piece) - 1);
            printf("generated=%d position=%d token=%d logit=%.9g piece=",
                   generated_count, position, output_token, output_logit);
            fwrite(piece, 1, (size_t)piece_length, stdout);
            fputc('\n', stdout);
            fflush(stdout);
            if (output_token == 1) break;
            if (stop_sentence && has_sentence_end(piece, piece_length)) break;
        }
        current_token = output_token;
    }
    ColiExpertStoreStats stats;
    experts->ops->stats(experts, &stats);
    printf("summary tokens=%d expert_reads=%llu bytes=%llu\n",
           text_mode ? generated_count : token_count,
           (unsigned long long)stats.misses,
           (unsigned long long)stats.bytes_read);

    if (text_mode) {
        size_t text_capacity = (size_t)generated_count * 256 + 1;
        char *text = malloc(text_capacity);
        if (!text) return 1;
        int decode_count = generated_count;
        if (decode_count && generated_ids[decode_count - 1] == 1) decode_count--;
        int text_length = tok_decode(&tokenizer, generated_ids, decode_count,
                                     text, (int)text_capacity - 1);
        printf("generated_text=");
        fwrite(text, 1, (size_t)text_length, stdout);
        fputc('\n', stdout);
        printf("completed_text=");
        fwrite(argv[3], 1, strlen(argv[3]), stdout);
        fwrite(text, 1, (size_t)text_length, stdout);
        fputc('\n', stdout);
        free(text);
    }
    for (int layer = 0; layer < config.num_hidden_layers; layer++)
        coli_v4_window_attention_destroy(attention[layer]);
    free(attention);
    free(generated_ids); free(prompt_ids);
    free(hidden); free(next); free(state);
    experts->ops->destroy(experts);
    coli_st_index_close(index);
    return 0;
}
/* ---- end include tools/deepseek_v4_first_token.c ---- */

#undef main

#include "deepseek_v4_internal.h"
static double spec_now(void) {
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return (double)value.tv_sec + value.tv_nsec * 1e-9;
}

static int spec_sentence_end(const char *text, int length) {
    for (int i = 0; i < length; i++) {
        unsigned char c = (unsigned char)text[i];
        if (c == '.' || c == '!' || c == '?' || c == '\n') return 1;
        if (i + 2 < length && c == 0xe3 &&
            (unsigned char)text[i + 1] == 0x80 &&
            (unsigned char)text[i + 2] == 0x82) return 1;
        if (i + 2 < length && c == 0xef &&
            (unsigned char)text[i + 1] == 0xbc &&
            ((unsigned char)text[i + 2] == 0x81 ||
             (unsigned char)text[i + 2] == 0x9f)) return 1;
    }
    return 0;
}

/* DSpark conditions on the target model's final three layer hiddens.  Keep a
 * short ring only when the full drafter is enabled; target-only runs pay no
 * allocation and no per-layer reduction. */
#define V4_MAINH_TARGETS 3
#define V4_MAINH_RING 128
static float *g_mainh_ring;
static int64_t g_mainh_abs[V4_MAINH_RING];
static int g_mainh_dim;

static void v4_mainh_tap(const ColiDeepSeekV4Config *config, int layer_id,
                         const float *row, int64_t position) {
    if (!coli_v4_full_dspark_wanted || !config || !row || position < 0) return;
    int first = config->num_hidden_layers - V4_MAINH_TARGETS;
    int target = layer_id - first;
    if (target < 0 || target >= V4_MAINH_TARGETS) return;
    int dimension = config->hidden_size;
    int copies = config->hc_mult;
    if (!g_mainh_ring) {
        g_mainh_ring = calloc(
            (size_t)V4_MAINH_RING * V4_MAINH_TARGETS * dimension,
            sizeof(float));
        if (!g_mainh_ring) return;
        for (int item = 0; item < V4_MAINH_RING; item++)
            g_mainh_abs[item] = -1;
        g_mainh_dim = dimension;
    }
    if (g_mainh_dim != dimension) return;
    int ring = (int)(position % V4_MAINH_RING);
    float *slot = g_mainh_ring +
        ((size_t)ring * V4_MAINH_TARGETS + target) * dimension;
    for (int column = 0; column < dimension; column++) {
        float sum = 0.0f;
        for (int copy = 0; copy < copies; copy++)
            sum += row[(size_t)copy * dimension + column];
        slot[column] = coli_bf16_round(sum / copies);
    }
    if (target == V4_MAINH_TARGETS - 1) g_mainh_abs[ring] = position;
}

static const float *v4_mainh_get(int64_t position) {
    if (!g_mainh_ring || position < 0) return NULL;
    int slot = (int)(position % V4_MAINH_RING);
    if (g_mainh_abs[slot] != position) return NULL;
    return g_mainh_ring +
        (size_t)slot * V4_MAINH_TARGETS * g_mainh_dim;
}

#include "deepseek_v4_dspark.inc"

static int target_batch(ColiV4Engine *engine, float **state_ptr, float **next_ptr,
                        ColiDeepSeekV4WindowAttentionState **attention,
                        const ColiSafetensorsIndex *index,
                        const ColiDeepSeekV4Config *config,
                        ColiExpertStore *experts, const int *tokens,
                        int start, int batch, char *error, size_t error_size) {
    if (!state_ptr || !next_ptr || !*state_ptr || !*next_ptr || !attention ||
        !index || !config || !experts || !tokens || start < 0 || batch < 1) {
        if (error && error_size)
            snprintf(error, error_size, "invalid target batch arguments");
        return -1;
    }
    float *state = *state_ptr, *next = *next_ptr;
    size_t hd = (size_t)config->hc_mult * config->hidden_size;
    for (int layer_id = 0; layer_id < config->num_hidden_layers; layer_id++) {
        ColiDeepSeekV4LayerWeights layer;
        if (coli_v4_layer_load(engine, &layer, config, index, layer_id,
                               error, error_size)) return -1;
        int result = 0;
        for (int offset = 0; !result && offset < batch; offset += 64) {
            int chunk = batch - offset;
            if (chunk > 64) chunk = 64;
            result = coli_v4_block_window_batch_ref(
                next + (size_t)offset * hd, attention[layer_id],
                &layer, config, experts, state + (size_t)offset * hd,
                tokens + offset, start + offset, chunk, error, error_size);
            if (result && error && error_size && !error[0])
                snprintf(error, error_size,
                         "target prefill failed layer=%d offset=%d batch=%d",
                         layer_id, offset, chunk);
        }
        coli_v4_layer_free(engine, &layer);
        if (result) return -1;
        float *swap = state; state = next; next = swap;
        for (int item = 0; item < batch; item++)
            v4_mainh_tap(config, layer_id, state + (size_t)item * hd,
                         (int64_t)start + item);
    }
    *state_ptr = state;
    *next_ptr = next;
    return 0;
}

static int target_token(ColiV4Engine *engine, float **state_ptr, float **next_ptr,
                        ColiDeepSeekV4WindowAttentionState **attention,
                        const ColiSafetensorsIndex *index,
                        const ColiDeepSeekV4Config *config,
                        ColiExpertStore *experts, int token, int position,
                        char *error, size_t error_size) {
    float *state = *state_ptr, *next = *next_ptr;
    if (load_embedding(state, index, config, token)) return -1;
    for (int layer_id = 0; layer_id < config->num_hidden_layers; layer_id++) {
        ColiDeepSeekV4LayerWeights layer;
        if (coli_v4_layer_load(engine, &layer, config, index, layer_id,
                               error, error_size)) return -1;
        int result = coli_v4_block_window_token_ref(
            next, attention[layer_id], &layer, config, experts,
            state, token, position, error, error_size);
        coli_v4_layer_free(engine, &layer);
        if (result) return -1;
        float *swap = state; state = next; next = swap;
        v4_mainh_tap(config, layer_id, state, position);
    }
    *state_ptr = state;
    *next_ptr = next;
    return 0;
}

static ColiV4AttentionSnapshot **spec_attention_save(
    ColiDeepSeekV4WindowAttentionState **attention, int layers) {
    if (!attention || layers < 1) return NULL;
    ColiV4AttentionSnapshot **snapshots = calloc(
        (size_t)layers, sizeof(*snapshots));
    if (!snapshots) return NULL;
    for (int layer = 0; layer < layers; layer++)
        if (coli_v4_attention_snapshot_create(attention[layer],
                                               &snapshots[layer])) {
            for (int item = 0; item < layers; item++)
                coli_v4_attention_snapshot_destroy(snapshots[item]);
            free(snapshots);
            return NULL;
        }
    return snapshots;
}

static int spec_attention_restore(
    ColiDeepSeekV4WindowAttentionState **attention,
    ColiV4AttentionSnapshot **snapshots, int layers) {
    if (!attention || !snapshots) return -1;
    for (int layer = 0; layer < layers; layer++)
        if (coli_v4_attention_snapshot_restore(attention[layer],
                                               snapshots[layer])) return -1;
    return 0;
}

static void spec_attention_free(ColiV4AttentionSnapshot **snapshots,
                                int layers) {
    if (!snapshots) return;
    for (int layer = 0; layer < layers; layer++)
        coli_v4_attention_snapshot_destroy(snapshots[layer]);
    free(snapshots);
}

#undef spec_print
#undef COLI_V4_GENERATE_HELPERS_ONLY
#undef COLI_V4_GENERATE_MAIN

#include "deepseek_v4_internal.h"
#include "deepseek_v4_internal.h"

static ColiExpertStoreStats stats_subtract(ColiExpertStoreStats end,
                                           ColiExpertStoreStats begin) {
    ColiExpertStoreStats delta = end;
    delta.requests -= begin.requests; delta.hits -= begin.hits;
    delta.misses -= begin.misses; delta.prefetched -= begin.prefetched;
    delta.prefetch_hits -= begin.prefetch_hits;
    delta.bytes_read -= begin.bytes_read;
    return delta;
}

static double stats_hit_rate(ColiExpertStoreStats stats) {
    return stats.requests ? 100.0 * stats.hits / stats.requests : 0.0;
}

#ifdef COLI_V4_EXPERIMENTAL_STATE_HASH
static uint64_t state_hash_v70(const float *values, size_t count) {
    const unsigned char *bytes = (const unsigned char *)values;
    uint64_t hash = UINT64_C(1469598103934665603);
    for (size_t i = 0; i < count * sizeof(float); i++) {
        hash ^= bytes[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}
#endif

typedef struct {
    const char *model_dir;
    const char *prompt;
    const char *prompt_file;
    const char *system_prompt;
    const char *oracle_path;
    const char *record_oracle_path;
    int max_new_tokens;
    int teacher_forcing;
    int greedy;
    int stop_sentence;
    int no_dspark;
    double memory_gib;
    ColiDeepSeekV4PromptMode prompt_mode;
} V4CliOptions;

static void v4_cli_usage(FILE *stream, const char *program) {
    fprintf(stream,
        "usage: %s MODEL PROMPT [options]\n"
        "       %s MODEL --prompt-file FILE [options]\n"
        "       %s MODEL --oracle FILE [--teacher-forcing N] [--greedy N] [options]\n"
        "  --max-tokens N       maximum generated tokens (default: 128)\n"
        "  --memory-gb GiB      cap this process; otherwise use available RAM\n"
        "  --prompt-file PATH   read UTF-8 prompt from file (avoids argv encoding issues)\n"
        "  --system TEXT        optional system message\n"
        "  --thinking           enable the official V4 thinking prefix\n"
        "  --raw-prompt         bypass the default V4 chat template\n"
        "  --stop-sentence      stop after the first sentence terminator\n"
        "  --no-dspark          disable verified speculative drafting\n"
        "  --oracle FILE        validate against an oracle JSON fixture\n"
        "  --teacher-forcing N  oracle: compare top-1 on N prompt positions\n"
        "  --greedy N           oracle: compare N greedy continuation tokens\n"
        "  --record-oracle FILE write greedy tokens + tf_pred to JSON\n",
        program, program, program);
}

static char *v4_read_prompt_file(const char *path, char *error, size_t error_size) {
    FILE *stream = fopen(path, "rb");
    if (!stream) {
        snprintf(error, error_size, "cannot open prompt file: %s", path);
        return NULL;
    }
    if (fseek(stream, 0, SEEK_END)) {
        fclose(stream);
        snprintf(error, error_size, "cannot seek prompt file: %s", path);
        return NULL;
    }
    long length = ftell(stream);
    if (length < 0 || fseek(stream, 0, SEEK_SET)) {
        fclose(stream);
        snprintf(error, error_size, "cannot size prompt file: %s", path);
        return NULL;
    }
    char *text = malloc((size_t)length + 1);
    if (!text) {
        fclose(stream);
        snprintf(error, error_size, "out of memory reading prompt file");
        return NULL;
    }
    size_t read = fread(text, 1, (size_t)length, stream);
    fclose(stream);
    if (read != (size_t)length) {
        free(text);
        snprintf(error, error_size, "cannot read prompt file: %s", path);
        return NULL;
    }
    while (read > 0 && (text[read - 1] == '\n' || text[read - 1] == '\r'))
        read--;
    text[read] = 0;
    return text;
}

static int v4_cli_positive_int(const char *text, int *output) {
    char *end = NULL;
    long value = strtol(text, &end, 10);
    if (!text[0] || end == text || *end || value < 1 || value > 1048576)
        return -1;
    *output = (int)value;
    return 0;
}

static int v4_cli_memory(const char *text, double *output) {
    char *end = NULL;
    double value = strtod(text, &end);
    if (!text[0] || end == text || *end || value <= 0.0 || value > 1048576.0)
        return -1;
    *output = value;
    return 0;
}

static int v4_cli_parse(int argc, char **argv, V4CliOptions *options) {
    if (!options || argc < 3) return -1;
    memset(options, 0, sizeof(*options));
    options->model_dir = argv[1];
    options->max_new_tokens = 128;
    options->prompt_mode = COLI_V4_PROMPT_CHAT;
    int argi = 2;
    if (argv[2][0] != '-') {
        options->prompt = argv[2];
        argi = 3;
    }
    for (int i = argi; i < argc; i++) {
        const char *option = argv[i];
        if (!strcmp(option, "--max-tokens")) {
            if (++i == argc ||
                v4_cli_positive_int(argv[i], &options->max_new_tokens))
                return -1;
        } else if (!strcmp(option, "--memory-gb")) {
            if (++i == argc || v4_cli_memory(argv[i], &options->memory_gib))
                return -1;
        } else if (!strcmp(option, "--prompt-file")) {
            if (++i == argc || !argv[i][0]) return -1;
            options->prompt_file = argv[i];
        } else if (!strcmp(option, "--system")) {
            if (++i == argc) return -1;
            options->system_prompt = argv[i];
        } else if (!strcmp(option, "--thinking")) {
            if (options->prompt_mode == COLI_V4_PROMPT_RAW) return -1;
            options->prompt_mode = COLI_V4_PROMPT_THINKING;
        } else if (!strcmp(option, "--raw-prompt")) {
            if (options->prompt_mode == COLI_V4_PROMPT_THINKING) return -1;
            options->prompt_mode = COLI_V4_PROMPT_RAW;
        } else if (!strcmp(option, "--stop-sentence")) {
            options->stop_sentence = 1;
        } else if (!strcmp(option, "--no-dspark")) {
            options->no_dspark = 1;
        } else if (!strcmp(option, "--oracle")) {
            if (++i == argc || !argv[i][0]) return -1;
            options->oracle_path = argv[i];
        } else if (!strcmp(option, "--record-oracle")) {
            if (++i == argc || !argv[i][0]) return -1;
            options->record_oracle_path = argv[i];
        } else if (!strcmp(option, "--teacher-forcing")) {
            if (++i == argc ||
                v4_cli_positive_int(argv[i], &options->teacher_forcing))
                return -1;
        } else if (!strcmp(option, "--greedy")) {
            if (++i == argc ||
                v4_cli_positive_int(argv[i], &options->greedy))
                return -1;
        } else {
            return -1;
        }
    }
    if (options->oracle_path) {
        if (options->prompt || options->prompt_file ||
            options->record_oracle_path) return -1;
        if (!options->teacher_forcing) options->teacher_forcing = 32;
        if (!options->greedy) options->greedy = 20;
        options->no_dspark = 1;
    } else if (options->prompt_file) {
        if (options->prompt) return -1;
    } else if (!options->prompt) {
        return -1;
    }
    return 0;
}

static int *v4_oracle_read_ids(jval *root, const char *key, int *count) {
    jval *array = json_get(root, key);
    if (!array || array->t != J_ARR || array->len < 1) return NULL;
    int *ids = malloc((size_t)array->len * sizeof(*ids));
    if (!ids) return NULL;
    for (int i = 0; i < array->len; i++) {
        if (!array->kids[i] || array->kids[i]->t != J_NUM) {
            free(ids);
            return NULL;
        }
        ids[i] = (int)array->kids[i]->num;
    }
    *count = array->len;
    return ids;
}

static int v4_oracle_write_json(const char *path, const char *source,
                                const char *model_dir, const char *prompt,
                                const int *prompt_ids, int prompt_count,
                                const int *full_ids, int full_count,
                                const int *tf_pred, int tf_count) {
    FILE *out = fopen(path, "wb");
    if (!out) return -1;
    fprintf(out,
            "{\n  \"source\": \"%s\",\n  \"model\": \"%s\",\n  \"prompt\": ",
            source, model_dir);
    fputc('"', out);
    for (const char *p = prompt ? prompt : ""; *p; p++) {
        unsigned char c = (unsigned char)*p;
        if (c == '\\' || c == '"') fputc('\\', out);
        if (c == '\n') { fputs("\\n", out); continue; }
        if (c == '\r') { fputs("\\r", out); continue; }
        if (c == '\t') { fputs("\\t", out); continue; }
        fputc(c, out);
    }
    fputs("\",\n  \"comparison\": {\n"
          "    \"top1_token\": \"exact\",\n"
          "    \"logits\": \"not required for coli-self fixtures\"\n"
          "  },\n  \"prompt_ids\": [", out);
    for (int i = 0; i < prompt_count; i++)
        fprintf(out, "%s%d", i ? ", " : "", prompt_ids[i]);
    fputs("],\n  \"full_ids\": [", out);
    for (int i = 0; i < full_count; i++)
        fprintf(out, "%s%d", i ? ", " : "", full_ids[i]);
    fputs("],\n  \"tf_pred\": [", out);
    for (int i = 0; i < tf_count; i++)
        fprintf(out, "%s%d", i ? ", " : "", tf_pred[i]);
    fputs("]\n}\n", out);
    fclose(out);
    return 0;
}

static void v4_attention_free(ColiDeepSeekV4WindowAttentionState **attention,
                              int layers) {
    if (!attention) return;
    for (int layer = 0; layer < layers; layer++)
        coli_v4_window_attention_destroy(attention[layer]);
    free(attention);
}

#include <assert.h>

static void session_free_buffers(ColiV4Session *session) {
    if (!session) return;
    free(session->text); session->text = NULL; session->text_length = 0;
    free(session->hidden); session->hidden = NULL;
    free(session->next); session->next = NULL;
    free(session->state); session->state = NULL;
    free(session->generated); session->generated = NULL;
    free(session->prompt_ids); session->prompt_ids = NULL;
}

static void session_free_attention(ColiV4Session *session) {
    if (!session || !session->attention) return;
    v4_attention_free(session->attention, session->config.num_hidden_layers);
    session->attention = NULL;
}

void coli_v4_session_destroy(ColiV4Session *session) {
    if (!session) return;
    kv_prefix_free(&session->fed);
    session_free_attention(session);
    session_free_buffers(session);
    if (session->tokenizer_ready) {
        tok_free(&session->tokenizer);
        session->tokenizer_ready = 0;
    }
    if (session->engine) {
        coli_v4_engine_detach_session(session->engine);
        session->engine = NULL;
    }
    free(session);
}

int coli_v4_session_create(ColiV4Session **output, ColiV4Engine *engine,
                           const ColiV4SessionCreateOptions *options,
                           char *error, size_t error_size) {
    if (!output || !engine) {
        if (error && error_size)
            snprintf(error, error_size, "invalid V4 session create arguments");
        return -1;
    }
    *output = NULL;
    ColiV4Session *session = calloc(1, sizeof(*session));
    if (!session) {
        if (error && error_size)
            snprintf(error, error_size, "out of memory creating V4 session");
        return -1;
    }
    session->engine = engine;
    coli_v4_engine_attach_session(engine);
    session->config = *coli_v4_engine_config(engine);
    session->max_prompt_tokens =
        options && options->max_prompt_tokens > 0 ? options->max_prompt_tokens
                                                  : 512;
    session->max_new_tokens_cap =
        options && options->max_new_tokens_cap > 0 ? options->max_new_tokens_cap
                                                   : 512;

    const char *model_dir = coli_v4_engine_target_model_dir(engine);
    char tokenizer_path[4096];
    if (!model_dir) {
        coli_v4_session_destroy(session);
        if (error && error_size)
            snprintf(error, error_size, "engine has no target model directory");
        return -1;
    }
    snprintf(tokenizer_path, sizeof(tokenizer_path), "%s/tokenizer.json",
             model_dir);
    tok_load(&session->tokenizer, tokenizer_path);
    session->tokenizer_ready = 1;

    session->attention = calloc((size_t)session->config.num_hidden_layers,
                                sizeof(*session->attention));
    if (!session->attention) {
        coli_v4_session_destroy(session);
        if (error && error_size)
            snprintf(error, error_size, "out of memory allocating attention");
        return -1;
    }
    for (int layer = 0; layer < session->config.num_hidden_layers; layer++) {
        if (coli_v4_window_attention_create(&session->attention[layer],
                                            &session->config)) {
            coli_v4_session_destroy(session);
            if (error && error_size)
                snprintf(error, error_size, "cannot create attention layer %d",
                         layer);
            return -1;
        }
    }

    size_t hd = (size_t)session->config.hc_mult * session->config.hidden_size;
    size_t slots = (size_t)session->max_prompt_tokens;
    session->state = malloc(slots * hd * sizeof(float));
    session->next = malloc(slots * hd * sizeof(float));
    session->hidden = malloc((size_t)session->config.hidden_size * sizeof(float));
    session->prompt_ids =
        malloc((size_t)(session->max_prompt_tokens + 16) * sizeof(int));
    session->generated =
        malloc((size_t)(session->max_new_tokens_cap + 64) * sizeof(int));
    if (!session->state || !session->next || !session->hidden ||
        !session->prompt_ids || !session->generated) {
        coli_v4_session_destroy(session);
        if (error && error_size)
            snprintf(error, error_size, "out of memory allocating session buffers");
        return -1;
    }
    /* Holds prompt and generated ids together: the next request's prompt
     * contains both, so both have to match for the state to be reusable.
     * A failure here is not an error — kv_prefix_alloc leaves the record empty,
     * kv_prefix_reuse then returns 0, and every request prefills in full, which
     * is exactly the behaviour before this change. */
    (void)kv_prefix_alloc(&session->fed,
                          session->max_prompt_tokens +
                          session->max_new_tokens_cap + 64);
    *output = session;
    return 0;
}

int coli_v4_session_generated_text(const ColiV4Session *session,
                                   char *buffer, size_t buffer_size,
                                   size_t *out_length) {
    if (!session || !buffer || buffer_size == 0) return -1;
    size_t copy = session->text_length;
    if (copy >= buffer_size) copy = buffer_size - 1;
    if (session->text && copy)
        memcpy(buffer, session->text, copy);
    buffer[copy] = 0;
    if (out_length) *out_length = copy;
    return 0;
}

static int session_emit_token(ColiV4Session *session,
                              ColiV4SessionTokenFn on_token, void *user_data,
                              int token, float logit, int position, int ordinal,
                              int stop_at_sentence) {
    if (on_token) {
        if (on_token(user_data, token, logit, position, ordinal)) return 1;
    } else {
        char piece[1024];
        int length = tok_decode(&session->tokenizer, &token, 1, piece,
                                (int)sizeof(piece) - 1);
        if (length > 0) fwrite(piece, 1, (size_t)length, stdout);
        fflush(stdout);
        if (stop_at_sentence && spec_sentence_end(piece, length)) return 1;
    }
    return token == 1;
}

static int v4_context_token(const int *prompt, int prompt_count,
                            const int *generated, int generated_count,
                            int position) {
    if (position < 0 || position >= prompt_count + generated_count) return -1;
    return position < prompt_count ? prompt[position]
                                   : generated[position - prompt_count];
}

/* Prompt-lookup drafting costs no model I/O.  Match the context tail against
 * its most recent earlier 3-gram/2-gram occurrence and propose the following
 * tokens.  Every proposal is still checked by the exact target batch below. */
static int v4_ngram_draft(const int *prompt, int prompt_count,
                          const int *generated, int generated_count,
                          int *output, int maximum) {
    int count = prompt_count + generated_count;
    if (!prompt || !generated || !output || maximum < 1) return 0;
    for (int gram = 3; gram >= 2; gram--) {
        if (count < gram + 1) continue;
        int tail = count - gram;
        for (int start = count - gram - 1; start >= 0; start--) {
            int matches = 1;
            for (int item = 0; item < gram; item++)
                if (v4_context_token(prompt, prompt_count, generated,
                                     generated_count, start + item) !=
                    v4_context_token(prompt, prompt_count, generated,
                                     generated_count, tail + item)) {
                    matches = 0;
                    break;
                }
            if (!matches) continue;
            int from = start + gram;
            int take = count - from;
            if (take > maximum) take = maximum;
            if (take < 1) break;
            for (int item = 0; item < take; item++)
                output[item] = v4_context_token(
                    prompt, prompt_count, generated, generated_count,
                    from + item);
            return take;
        }
    }
    return 0;
}

int coli_v4_session_generate(ColiV4Session *session,
                             const char *prompt, size_t prompt_length,
                             const ColiV4SessionGenerateOptions *options,
                             ColiV4SessionTokenFn on_token, void *user_data,
                             ColiV4SessionGenerateStats *stats_out,
                             char *error, size_t error_size) {
    if (!session || !session->engine || !prompt || !options ||
        options->max_new_tokens < 1) {
        if (error && error_size)
            snprintf(error, error_size, "invalid V4 session generate arguments");
        return -1;
    }
    if (stats_out) memset(stats_out, 0, sizeof(*stats_out));
    free(session->text);
    session->text = NULL;
    session->text_length = 0;
    session->prompt_count = 0;
    session->generated_count = 0;
    session->prefix_reused = 0;
    session->spec_attempts = 0;
    session->spec_drafted = 0;
    session->spec_accepted = 0;
    session->spec_disabled = 0;

    int max_new = options->max_new_tokens;
    if (max_new > session->max_new_tokens_cap)
        max_new = session->max_new_tokens_cap;
    int prompt_capacity = session->max_prompt_tokens + 16;
    int prompt_count = tok_encode(&session->tokenizer, prompt, prompt_length,
                                  session->prompt_ids, prompt_capacity);
    if (prompt_count < 1 || prompt_count > session->max_prompt_tokens) {
        if (error && error_size)
            snprintf(error, error_size,
                     "V4 prompt must encode to between 1 and %d tokens",
                     session->max_prompt_tokens);
        return -1;
    }
    session->prompt_count = prompt_count;

    ColiV4Engine *engine = session->engine;
    const ColiDeepSeekV4Config *config = &session->config;
    ColiSafetensorsIndex *index = coli_v4_engine_target_index(engine);
    ColiExpertStore *experts = coli_v4_engine_expert_store(engine);
    float *state = session->state;
    float *next = session->next;
    float *hidden = session->hidden;
    ColiDeepSeekV4WindowAttentionState **attention = session->attention;
    int *generated = session->generated;
    size_t hd = (size_t)config->hc_mult * config->hidden_size;

    /* ---------------------------------------------------------------------
     * KV PREFIX REUSE.
     *
     * The window attention state is not truncatable to an arbitrary position:
     * the sliding window is a ring and the compressor carries recurrent
     * kv_state/score_state rather than per-position rows. What it *can* do is
     * keep going. So the reusable case is the exact one a conversation
     * produces: turn N+1's prompt begins with every id turn N fed, prompt and
     * reply alike, and only the tail is new.
     *
     * kv_prefix_reuse returns 0 unless the recorded ids are a strict prefix of
     * this prompt, so a divergent or shorter prompt falls back to a full reset
     * and prefill, and the reuse path always has at least one token left to
     * feed. Nothing about the math changes: positions stay absolute and the
     * tail is prefilled at start=reuse, so the logits are the ones a cold run
     * would have produced.
     * ------------------------------------------------------------------- */
    int reuse = kv_prefix_reuse(&session->fed, session->prompt_ids, prompt_count);
    if (!reuse) {
        for (int layer = 0; layer < config->num_hidden_layers; layer++)
            coli_v4_window_attention_reset(session->attention[layer]);
        if (coli_v4_full_dspark_wanted) v4_ds_reset_history();
    }
    session->prefix_reused = reuse;
    if (reuse && getenv("V4_PREFIX_LOG"))
        fprintf(stderr, "[PREFIX] reusing %d of %d prompt tokens\n",
                reuse, prompt_count);

    int fresh = prompt_count - reuse;
    for (int item = 0; item < fresh; item++)
        if (load_embedding(state + (size_t)item * hd, index, config,
                           session->prompt_ids[reuse + item])) {
            /* The state now matches neither the old ids nor the new ones. */
            kv_prefix_taint(&session->fed);
            if (error && error_size)
                snprintf(error, error_size, "cannot load embedding");
            return -1;
        }

    double setup_done = spec_now();
    if (target_batch(engine, &state, &next, attention, index, config, experts,
                     session->prompt_ids + reuse, reuse, fresh,
                     error, error_size)) {
        kv_prefix_taint(&session->fed);
        return -1;
    }
    session->state = state;
    session->next = next;
    /* The batch holds only the fresh tail, so the final row is at fresh-1
     * even though its absolute position is prompt_count-1. */
    const float *last = state + (size_t)(fresh - 1) * hd;
    int current = 0;
    float current_logit = 0.0f;
    if (final_hidden(hidden, last, index, config, error, error_size) ||
        head_argmax(engine, hidden, index, config, &current, &current_logit)) {
        kv_prefix_taint(&session->fed);
        return -1;
    }
    /* The prompt is in the attention state from here on; record it before the
     * decode loop so a failure mid-generation still leaves fed describing what
     * was actually fed. */
    kv_prefix_record(&session->fed, session->prompt_ids + reuse, reuse, fresh);
    session->fed.len = prompt_count;
    int generated_count = 0;
    int last_processed = prompt_count - 1;
    generated[generated_count++] = current;
    int done = session_emit_token(session, on_token, user_data, current,
                                  current_logit, last_processed,
                                  generated_count, options->stop_at_sentence);
    double first_at = spec_now();

    int draft_limit = getenv("V4_DRAFT") ? atoi(getenv("V4_DRAFT")) : 0;
    if (draft_limit < 0) draft_limit = 0;
    if (draft_limit > 24) draft_limit = 24;
    int markov_disabled = 0;
    int mtp_disabled = 0;
    int full_mtp_ready = 0;

    while (!done && generated_count < max_new) {
        int remaining = max_new - generated_count;
        if (!options->no_dspark && !session->spec_disabled && remaining >= 3) {
            int inputs[25] = {0}, drafts[24] = {0};
            int predictions[25] = {0};
            float logits[25] = {0};
            int room = remaining - 1; /* verification emits one exact fallback */
            int proposals = draft_limit < room ? draft_limit : room;
            const char *ngram_env = getenv("V4_NGRAM");
            int ngram_enabled = !ngram_env || atoi(ngram_env) != 0;
            if (proposals > 1 && ngram_enabled)
                proposals = v4_ngram_draft(
                    session->prompt_ids, prompt_count, generated,
                    generated_count, drafts, proposals);
            else
                proposals = 0;
            int using_full_mtp = 0;
            int using_markov = 0;
            int using_ngram = proposals > 1;
            inputs[0] = current;
            int draft_ready = proposals > 1;
            if (!draft_ready && coli_v4_full_dspark_wanted && full_mtp_ready &&
                !mtp_disabled) {
                int mtp_min = getenv("V4_MTP_MIN")
                    ? atoi(getenv("V4_MTP_MIN")) : 3;
                if (mtp_min < 1) mtp_min = 1;
                int maximum = draft_limit < room ? draft_limit : room;
                int mtp_max = getenv("V4_MTP_DRAFT")
                    ? atoi(getenv("V4_MTP_DRAFT")) : 3;
                if (mtp_max < 1) mtp_max = 1;
                if (maximum > mtp_max) maximum = mtp_max;
                if (maximum >= mtp_min) {
                    proposals = v4_dspark_draft(
                        engine, index, config, current, last_processed + 1,
                        drafts, maximum);
                    draft_ready = proposals > 0;
                    using_full_mtp = draft_ready;
                    if (!draft_ready) mtp_disabled = 1;
                }
            } else if (!draft_ready && engine->dspark.enabled &&
                       !markov_disabled) {
                proposals = engine->dspark.block_size;
                if (proposals > room) proposals = room;
                draft_ready = proposals > 1;
                using_markov = draft_ready;
                for (int item = 0; draft_ready && item < proposals; item++) {
                    if (dspark_markov_argmax(engine, inputs[item],
                                             &drafts[item])) {
                        draft_ready = 0;
                        break;
                    }
                    if (item + 1 < proposals) inputs[item + 1] = drafts[item];
                }
            }
            if (!draft_ready) {
                if (using_markov) markov_disabled = 1;
            } else {
                int batch = proposals + 1;
                for (int item = 1; item < batch; item++)
                    inputs[item] = drafts[item - 1];
                ColiV4AttentionSnapshot **snapshots = spec_attention_save(
                    attention, config->num_hidden_layers);
                if (!snapshots) {
                    session->spec_disabled = 1;
                } else {
                    int old_last = last_processed;
                    for (int item = 0; item < batch; item++)
                        if (load_embedding(state + (size_t)item * hd, index,
                                           config, inputs[item])) {
                            spec_attention_free(snapshots,
                                                config->num_hidden_layers);
                            kv_prefix_taint(&session->fed);
                            if (error && error_size)
                                snprintf(error, error_size,
                                         "cannot load speculative embedding");
                            return -1;
                        }
                    if (target_batch(engine, &state, &next, attention, index,
                                     config, experts, inputs, old_last + 1,
                                     batch, error, error_size)) {
                        (void)spec_attention_restore(
                            attention, snapshots, config->num_hidden_layers);
                        spec_attention_free(snapshots,
                                            config->num_hidden_layers);
                        kv_prefix_taint(&session->fed);
                        return -1;
                    }
                    float *batch_hidden = malloc(
                        (size_t)batch * config->hidden_size * sizeof(float));
                    int heads_ok = batch_hidden != NULL;
                    for (int item = 0; heads_ok && item < batch; item++)
                        if (final_hidden(
                                batch_hidden +
                                    (size_t)item * config->hidden_size,
                                state + (size_t)item * hd, index, config,
                                error, error_size)) heads_ok = 0;
                    if (heads_ok && head_argmax_batch(
                            engine, batch_hidden, index, config, batch,
                            predictions, logits)) heads_ok = 0;
                    free(batch_hidden);
                    if (!heads_ok) {
                        (void)spec_attention_restore(
                            attention, snapshots, config->num_hidden_layers);
                        spec_attention_free(snapshots,
                                            config->num_hidden_layers);
                        kv_prefix_taint(&session->fed);
                        if (error && error_size && !error[0])
                            snprintf(error, error_size,
                                     "speculative target head failed");
                        return -1;
                    }

                    int accepted = 0;
                    while (accepted < proposals &&
                           predictions[accepted] == drafts[accepted])
                        accepted++;
                    session->spec_attempts++;
                    session->spec_drafted += (uint64_t)proposals;
                    session->spec_accepted += (uint64_t)accepted;
                    if (using_full_mtp)
                        v4_dspark_feedback(proposals, accepted);
                    /* With recurrent compressed attention, a rejected suffix
                     * cannot be truncated in place: the accepted prefix must
                     * be replayed.  Real chat measured 10/24 accepted and
                     * 495 s for 14 visible tokens, so stop full MTP after the
                     * first non-perfect block unless explicitly benchmarking
                     * it.  Exact prompt lookup remains available. */
                    if (using_full_mtp && accepted < proposals) {
                        const char *keep = getenv("V4_MTP_PARTIAL_KEEP");
                        if (!keep || atoi(keep) == 0) mtp_disabled = 1;
                    }
                    if (using_ngram && accepted < proposals) {
                        const char *keep = getenv("V4_NGRAM_PARTIAL_KEEP");
                        if (!keep || atoi(keep) == 0)
                            session->spec_disabled = 1;
                    }
                    int available_outputs = accepted + 1;
                    int retained = 0;
                    for (int item = 0;
                         item < available_outputs && !done &&
                         generated_count < max_new; item++) {
                        current = predictions[item];
                        current_logit = logits[item];
                        generated[generated_count++] = current;
                        retained++;
                        done = session_emit_token(
                            session, on_token, user_data, current,
                            current_logit, old_last + 1 + item,
                            generated_count, options->stop_at_sentence);
                    }

                    /* A mismatch in the first row (or an early callback stop)
                     * leaves unverified draft inputs in the batched KV state.
                     * Restore the exact snapshot and replay only inputs that
                     * really correspond to emitted outputs. */
                    if (retained < batch) {
                        if (spec_attention_restore(
                                attention, snapshots,
                                config->num_hidden_layers)) {
                            spec_attention_free(
                                snapshots, config->num_hidden_layers);
                            kv_prefix_taint(&session->fed);
                            if (error && error_size)
                                snprintf(error, error_size,
                                         "cannot restore speculative KV state");
                            return -1;
                        }
                        if (coli_v4_full_dspark_wanted)
                            v4_ds_invalidate_from(old_last + 1);
                        for (int item = 0; item < retained; item++)
                            if (load_embedding(state + (size_t)item * hd,
                                               index, config, inputs[item])) {
                                spec_attention_free(
                                    snapshots, config->num_hidden_layers);
                                kv_prefix_taint(&session->fed);
                                if (error && error_size)
                                    snprintf(error, error_size,
                                             "cannot replay speculative input");
                                return -1;
                            }
                        if (retained > 0 && target_batch(
                                engine, &state, &next, attention, index,
                                config, experts, inputs, old_last + 1,
                                retained, error, error_size)) {
                            spec_attention_free(
                                snapshots, config->num_hidden_layers);
                            kv_prefix_taint(&session->fed);
                            return -1;
                        }
                    }
                    spec_attention_free(snapshots,
                                        config->num_hidden_layers);
                    if (retained > 0) {
                        kv_prefix_record(&session->fed, inputs, old_last + 1,
                                         retained);
                        last_processed = old_last + retained;
                        full_mtp_ready = 1;
                    }
                    session->state = state;
                    session->next = next;
                    const char *keep = getenv("COLI_V4_MARKOV_KEEP");
                    if (using_markov && accepted == 0 &&
                        (!keep || atoi(keep) == 0)) markov_disabled = 1;
                    continue;
                }
            }
        }
        int position = last_processed + 1;
        if (target_token(engine, &state, &next, attention, index, config, experts,
                         current, position, error, error_size)) {
            kv_prefix_taint(&session->fed);
            return -1;
        }
        /* `current` is now in the attention state at `position`. Record it here,
         * before head_argmax overwrites it: the token generated last is never
         * fed, so recording after the loop would claim one token too many. */
        kv_prefix_record(&session->fed, &current, position, 1);
        session->state = state;
        session->next = next;
        if (final_hidden(hidden, state, index, config, error, error_size) ||
            head_argmax(engine, hidden, index, config, &current, &current_logit)) {
            kv_prefix_taint(&session->fed);
            return -1;
        }
        last_processed = position;
        generated[generated_count++] = current;
        done = session_emit_token(session, on_token, user_data, current,
                                  current_logit, last_processed,
                                  generated_count,
                                  options->stop_at_sentence);
        full_mtp_ready = 1;
    }
    if (coli_v4_full_dspark_wanted) v4_dspark_report();
    double ended = spec_now();
    session->state = state;
    session->next = next;
    session->generated_count = generated_count;

    size_t text_capacity = (size_t)generated_count * 256 + 1;
    session->text = malloc(text_capacity);
    if (session->text) {
        int text_count = generated_count;
        if (text_count && generated[text_count - 1] == 1) text_count--;
        session->text_length = tok_decode(&session->tokenizer, generated,
                                          text_count, session->text,
                                          (int)text_capacity - 1);
    }
    if (!on_token) {
        fputc('\n', stdout);
        fflush(stdout);
    }
    if (stats_out) {
        stats_out->prompt_tokens = prompt_count;
        stats_out->generated_tokens = generated_count;
        stats_out->eos_stopped = done && generated_count > 0 &&
                                 generated[generated_count - 1] == 1;
        stats_out->time_to_first_token_sec = first_at - setup_done;
        stats_out->decode_sec = ended - first_at;
        stats_out->speculative_drafted = session->spec_drafted;
        stats_out->speculative_accepted = session->spec_accepted;
    }
    if (session->spec_attempts)
        fprintf(stderr,
                "v4_dspark attempts=%llu drafted=%llu accepted=%llu "
                "acceptance=%.1f%% adaptive_disabled=%d\n",
                (unsigned long long)session->spec_attempts,
                (unsigned long long)session->spec_drafted,
                (unsigned long long)session->spec_accepted,
                session->spec_drafted
                    ? 100.0 * session->spec_accepted / session->spec_drafted
                    : 0.0,
                session->spec_disabled);
    return 0;
}

static int v4_oracle_teacher_forcing(
        const int *full_ids, int full_count, const int *expected, int expect_count,
        ColiDeepSeekV4WindowAttentionState **attention,
        const ColiSafetensorsIndex *index, const ColiDeepSeekV4Config *config,
        ColiExpertStore *experts, char *error, size_t error_size,
        int *matched_out) {
    size_t hd = (size_t)config->hc_mult * config->hidden_size;
    float *state = malloc((size_t)full_count * hd * sizeof(float));
    float *next = malloc((size_t)full_count * hd * sizeof(float));
    float *hidden = malloc((size_t)config->hidden_size * sizeof(float));
    if (!state || !next || !hidden) {
        free(state); free(next); free(hidden);
        return -1;
    }
    for (int item = 0; item < full_count; item++)
        if (load_embedding(state + (size_t)item * hd, index, config,
                           full_ids[item])) {
            free(state); free(next); free(hidden);
            return -1;
        }
    if (target_batch(NULL, &state, &next, attention, index, config, experts,
                     full_ids, 0, full_count, error, error_size)) {
        free(state); free(next); free(hidden);
        return -1;
    }
    int limit = expect_count < full_count ? expect_count : full_count;
    int matched = 0;
    for (int pos = 0; pos < limit; pos++) {
        int pred = -1;
        float logit = 0.0f;
        if (final_hidden(hidden, state + (size_t)pos * hd, index, config,
                         error, error_size) ||
            head_argmax(NULL, hidden, index, config, &pred, &logit)) {
            free(state); free(next); free(hidden);
            return -1;
        }
        if (pred == expected[pos]) matched++;
        else
            fprintf(stderr,
                    "[ORACLE] TF mismatch pos=%d expected=%d got=%d logit=%.6g\n",
                    pos, expected[pos], pred, logit);
    }
    free(state); free(next); free(hidden);
    *matched_out = matched;
    return 0;
}

static int v4_oracle_greedy_from_prompt(
        const int *prompt_ids, int prompt_count, int *generated, int max_new,
        ColiDeepSeekV4WindowAttentionState **attention,
        const ColiSafetensorsIndex *index, const ColiDeepSeekV4Config *config,
        ColiExpertStore *experts, char *error, size_t error_size) {
    size_t hd = (size_t)config->hc_mult * config->hidden_size;
    float *state = malloc((size_t)prompt_count * hd * sizeof(float));
    float *next = malloc((size_t)prompt_count * hd * sizeof(float));
    float *hidden = malloc((size_t)config->hidden_size * sizeof(float));
    if (!state || !next || !hidden) {
        free(state); free(next); free(hidden);
        return -1;
    }
    for (int item = 0; item < prompt_count; item++)
        if (load_embedding(state + (size_t)item * hd, index, config,
                           prompt_ids[item])) {
            free(state); free(next); free(hidden);
            return -1;
        }
    if (target_batch(NULL, &state, &next, attention, index, config, experts,
                     prompt_ids, 0, prompt_count, error, error_size)) {
        free(state); free(next); free(hidden);
        return -1;
    }
    int current = -1;
    float logit = 0.0f;
    if (final_hidden(hidden, state + (size_t)(prompt_count - 1) * hd,
                     index, config, error, error_size) ||
        head_argmax(NULL, hidden, index, config, &current, &logit)) {
        free(state); free(next); free(hidden);
        return -1;
    }
    int count = 0;
    generated[count++] = current;
    int position = prompt_count;
    while (count < max_new && current != 1) {
        if (target_token(NULL, &state, &next, attention, index, config, experts,
                         current, position, error, error_size) ||
            final_hidden(hidden, state, index, config, error, error_size) ||
            head_argmax(NULL, hidden, index, config, &current, &logit)) {
            free(state); free(next); free(hidden);
            return -1;
        }
        generated[count++] = current;
        position++;
    }
    free(state); free(next); free(hidden);
    return count;
}

static int spec_print(Tok *tokenizer, int token, float logit,
                      int position, int ordinal, int stop_sentence) {
    (void)logit; (void)position; (void)ordinal;
    if (token == 1) return 1;
    char piece[1024];
    int length = tok_decode(tokenizer, &token, 1, piece, sizeof(piece) - 1);
    if (length > 0) fwrite(piece, 1, (size_t)length, stdout);
    fflush(stdout);
    return stop_sentence && spec_sentence_end(piece, length);
}

static void v4_generate_cleanup(
    ColiV4Session *session,
    char *prompt_storage,
    ColiV4Engine *engine,
    ColiDeepSeekV4WindowAttentionState **attention,
    int layers,
    int *prompt_ids,
    int *generated,
    float *state,
    float *next,
    float *hidden,
    char *text,
    int *full_ids,
    int *tf_pred,
    float *tf_state,
    float *tf_next,
    float *tf_hidden)
{
    /* Session owns attention/runner/buffers on the normal path. Clear any
     * aliases into the session before destroying it to avoid double-free. */
    if (session) {
        prompt_ids = NULL;
        generated = NULL;
        attention = NULL;
        state = NULL;
        next = NULL;
        hidden = NULL;
        text = NULL;
        coli_v4_session_destroy(session);
    }
    free(tf_hidden);
    free(tf_next);
    free(tf_state);
    free(tf_pred);
    free(full_ids);
    free(text);
    free(hidden);
    free(next);
    free(state);
    free(generated);
    free(prompt_ids);
    v4_attention_free(attention, layers);
    coli_v4_engine_destroy(engine);
    free(prompt_storage);
}

typedef struct {
    char id[64];
    char *prompt;
    int prompt_bytes;
    int max_tokens;
    float temperature;
    float top_p;
    int extension_bytes;
} V4ServeRequest;

typedef struct {
    ColiV4Session *session;
    const char *request_id;
    int cancelled;
} V4ServeStream;

static double v4_serve_rss_gb(void) {
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage)) return 0.0;
#ifdef __APPLE__
    return usage.ru_maxrss / (1024.0 * 1024.0 * 1024.0);
#else
    return usage.ru_maxrss / (1024.0 * 1024.0);
#endif
}

static int v4_serve_read_request(V4ServeRequest *request,
                                 const char *active_id) {
    char line[512], command[16], id[64];
    if (!fgets(line, sizeof(line), stdin)) return -1;
    if (sscanf(line, "%15s %63s", command, id) < 2) return 0;
    if (!strcmp(command, "CANCEL") || !strcmp(command, "STOP"))
        return active_id && !strcmp(active_id, id);
    if (strcmp(command, "SUBMIT")) return 0;

    int slot = 0, prompt_bytes = 0, max_tokens = 0, extension_bytes = 0;
    float temperature = 0.0f, top_p = 1.0f;
    int fields = sscanf(line, "%*s %*s %d %d %d %f %f %d",
                        &slot, &prompt_bytes, &max_tokens,
                        &temperature, &top_p, &extension_bytes);
    if (fields < 5 || slot != 0 || prompt_bytes < 0 ||
        prompt_bytes > (1 << 24) || max_tokens < 1 ||
        extension_bytes < 0 || extension_bytes > (1 << 24)) {
        printf("ERROR %s bad submit header\n", id);
        fflush(stdout);
        return 0;
    }
    size_t total = (size_t)prompt_bytes + (size_t)extension_bytes;
    char *payload = malloc(total + 1);
    if (!payload) {
        printf("ERROR %s out of memory\n", id);
        fflush(stdout);
        return 0;
    }
    if (fread(payload, 1, total, stdin) != total) {
        free(payload);
        return -1;
    }
    (void)fgetc(stdin);
    payload[prompt_bytes] = 0;
    memset(request, 0, sizeof(*request));
    snprintf(request->id, sizeof(request->id), "%s", id);
    request->prompt = payload;
    request->prompt_bytes = prompt_bytes;
    request->max_tokens = max_tokens;
    request->temperature = temperature;
    request->top_p = top_p;
    request->extension_bytes = extension_bytes;
    return 2;
}

static void v4_serve_data(const char *id, const char *data, int bytes) {
    if (bytes <= 0) return;
    printf("DATA %s %d\n", id, bytes);
    fwrite(data, 1, (size_t)bytes, stdout);
    fputc('\n', stdout);
    fflush(stdout);
}

static int v4_serve_token(void *user_data, int token, float logit,
                          int position, int ordinal) {
    (void)logit;
    (void)position;
    (void)ordinal;
    V4ServeStream *stream = user_data;
    if (token != 1) {
        char piece[1024];
        int bytes = tok_decode(&stream->session->tokenizer, &token, 1,
                               piece, (int)sizeof(piece) - 1);
        v4_serve_data(stream->request_id, piece, bytes);
    }
    while (coli_stdin_readable()) {
        V4ServeRequest queued = {0};
        int result = v4_serve_read_request(&queued, stream->request_id);
        if (result < 0 || result == 1) {
            stream->cancelled = 1;
            free(queued.prompt);
            return 1;
        }
        if (result == 2) {
            printf("ERROR %s engine busy\n", queued.id);
            fflush(stdout);
            free(queued.prompt);
        }
    }
    return 0;
}

static void v4_serve_error(const char *id, const char *message) {
    char clean[512];
    snprintf(clean, sizeof(clean), "%s", message && *message ? message : "engine request failed");
    for (char *p = clean; *p; p++)
        if (*p == '\r' || *p == '\n') *p = ' ';
    printf("ERROR %s %s\n", id, clean);
    fflush(stdout);
}

static void v4_serve_one(ColiV4Engine *engine, ColiV4Session *session,
                         V4ServeRequest *request) {
    if (request->extension_bytes) {
        v4_serve_error(request->id, "unsupported request extension");
        return;
    }
    if (request->temperature != 0.0f)
        fprintf(stderr, "[V4] temperature %.3g ignored; target engine is greedy\n",
                request->temperature);
    if (request->top_p != 1.0f)
        fprintf(stderr, "[V4] top_p %.3g ignored; target engine is greedy\n",
                request->top_p);

    int prompt_count = tok_encode(&session->tokenizer, request->prompt,
                                  request->prompt_bytes, session->prompt_ids,
                                  session->max_prompt_tokens + 16);
    int context = engine->runtime.context_tokens;
    if (prompt_count < 1 || prompt_count > session->max_prompt_tokens ||
        prompt_count + request->max_tokens > context) {
        char message[256];
        snprintf(message, sizeof(message),
                 "CONTEXT_EXCEEDED prompt_tokens=%d requested=%d capacity=%d",
                 prompt_count, request->max_tokens, context);
        v4_serve_error(request->id, message);
        return;
    }
    printf("ACCEPT %s %d\n", request->id, prompt_count);
    fflush(stdout);

    ColiExpertStoreStats before = {0}, after = {0};
    if (engine->experts && engine->experts->ops && engine->experts->ops->stats)
        engine->experts->ops->stats(engine->experts, &before);
    V4ServeStream stream = {session, request->id, 0};
    ColiV4SessionGenerateStats stats = {0};
    char error[512] = {0};
    double started = spec_now();
    int result = coli_v4_session_generate(
        session, request->prompt, (size_t)request->prompt_bytes,
        &(ColiV4SessionGenerateOptions){
            .max_new_tokens = request->max_tokens,
            .stop_at_sentence = 0,
            .no_dspark = 0,
        },
        v4_serve_token, &stream, &stats, error, sizeof(error));
    double elapsed = spec_now() - started;
    if (result) {
        v4_serve_error(request->id, error);
        return;
    }
    if (engine->experts && engine->experts->ops && engine->experts->ops->stats)
        engine->experts->ops->stats(engine->experts, &after);
    uint64_t hits = after.hits - before.hits;
    uint64_t misses = after.misses - before.misses;
    double hit_rate = hits + misses ? 100.0 * hits / (hits + misses) : 0.0;
    int completion = stats.generated_tokens - (stats.eos_stopped ? 1 : 0);
    if (completion < 0) completion = 0;
    int length_limited = !stream.cancelled && !stats.eos_stopped &&
                         stats.generated_tokens >= request->max_tokens;
    double decode = stats.decode_sec > 0.0 ? stats.decode_sec : elapsed;
    /* Trailing field: prompt tokens served from the previous turn's attention
     * state instead of being prefilled again. Appended rather than inserted --
     * openai_server.py accepts `len(fields) >= 7`, so an older reader ignores
     * it and a newer one can report it. */
    printf("DONE %s STAT %d %.3f %.1f %.2f %d %d %d\n",
           request->id, completion,
           decode > 0.0 ? completion / decode : 0.0,
           hit_rate, v4_serve_rss_gb(), stats.prompt_tokens, length_limited,
           session->prefix_reused);
    fflush(stdout);
}

static int v4_serve_main(void) {
    const char *model_dir = getenv("SNAP");
    if (!model_dir || !*model_dir) {
        fprintf(stderr, "set SNAP=<DeepSeek V4 model directory>\n");
        return 1;
    }
    int context = getenv("CTX") ? atoi(getenv("CTX")) : 4096;
    int max_tokens = getenv("NGEN") ? atoi(getenv("NGEN")) : 1024;
    if (context < 2) context = 4096;
    if (max_tokens < 1) max_tokens = 1024;
    char error[512] = {0};
    ColiV4Engine *engine = NULL;
    ColiV4Session *session = NULL;
    ColiV4EngineOpenOptions open_options = {
        .target_model_dir = model_dir,
        .context_tokens = context,
        .pin_slots_per_layer = -1,
        .no_dspark = 0,
    };
    const char *ram = getenv("RAM_GB");
    if (ram && atof(ram) > 0.0)
        open_options.memory_limit_bytes =
            (uint64_t)(atof(ram) * 1073741824.0);
    if (coli_v4_engine_open(&engine, &open_options, error, sizeof(error))) {
        fprintf(stderr, "%s\n", error);
        return 1;
    }
    context = engine->runtime.context_tokens;
    if (coli_v4_session_create(
            &session, engine,
            &(ColiV4SessionCreateOptions){
                .max_prompt_tokens = context,
                .max_new_tokens_cap = max_tokens,
            },
            error, sizeof(error))) {
        fprintf(stderr, "%s\n", error);
        coli_v4_engine_destroy(engine);
        return 1;
    }

    coli_serve_binary_mode();
    setvbuf(stdin, NULL, _IONBF, 0);
    fputs("\x01\x01READY\x01\x01\n", stdout);
    printf("STAT 0 0.0 0.0 %.2f 0 0\n", v4_serve_rss_gb());
    fflush(stdout);
    for (;;) {
        V4ServeRequest request = {0};
        int result;
        do result = v4_serve_read_request(&request, NULL); while (result == 0);
        if (result < 0) break;
        if (result == 2) {
            v4_serve_one(engine, session, &request);
            free(request.prompt);
        }
    }
    coli_v4_session_destroy(session);
    coli_v4_engine_destroy(engine);
    return 0;
}


#ifndef COLI_V4_SKIP_GENERATE_MAIN
int main(int argc, char **argv) {
    if (getenv("SERVE") && getenv("SERVE")[0] == '1')
        return v4_serve_main();
    double process_started = spec_now();
    int result = 1;
    V4CliOptions cli;
    if (v4_cli_parse(argc, argv, &cli)) {
        v4_cli_usage(stderr, argc ? argv[0] : "deepseek-v4");
        return 2;
    }
    int max_new = cli.max_new_tokens;
    int stop_sentence = cli.stop_sentence;

    char error[512] = {0}, tokenizer_path[4096];
    char *prompt_storage = NULL;
    ColiV4Engine *engine = NULL;
    ColiV4Session *session = NULL;
    Tok tokenizer;
    memset(&tokenizer, 0, sizeof(tokenizer));
    ColiDeepSeekV4Config config;
    memset(&config, 0, sizeof(config));
    ColiSafetensorsIndex *index = NULL;
    ColiExpertStore *experts = NULL;
    ColiDeepSeekV4WindowAttentionState **attention = NULL;
    int *prompt_ids = NULL;
    int *generated = NULL;
    float *state = NULL, *next = NULL, *hidden = NULL;
    char *text = NULL;
    int *full_ids = NULL, *tf_pred = NULL;
    float *tf_state = NULL, *tf_next = NULL, *tf_hidden = NULL;
    int layers = 0;
    if (cli.prompt_file) {
        prompt_storage = v4_read_prompt_file(cli.prompt_file, error, sizeof(error));
        if (!prompt_storage) {
            fprintf(stderr, "%s\n", error);
            goto cleanup;
        }
        cli.prompt = prompt_storage;
    }
    {
        ColiV4EngineOpenOptions open_opts = {
            .target_model_dir = cli.model_dir,
            .no_dspark = cli.no_dspark,
            .pin_slots_per_layer = -1,
        };
        if (cli.memory_gib > 0.0)
            open_opts.memory_limit_bytes =
                (uint64_t)(cli.memory_gib * 1073741824.0);
        if (coli_v4_engine_open(&engine, &open_opts, error, sizeof(error))) {
            fprintf(stderr, "%s\n", error);
            goto cleanup;
        }
    }
    config = *coli_v4_engine_config(engine);
    index = coli_v4_engine_target_index(engine);
    experts = coli_v4_engine_expert_store(engine);
    layers = config.num_hidden_layers;
    snprintf(tokenizer_path, sizeof(tokenizer_path), "%s/tokenizer.json",
             cli.model_dir);
    tok_load(&tokenizer, tokenizer_path);

    if (cli.oracle_path) {
        FILE *oracle_file = fopen(cli.oracle_path, "rb");
        if (!oracle_file) { perror(cli.oracle_path); goto cleanup; }
        fseek(oracle_file, 0, SEEK_END);
        long oracle_bytes = ftell(oracle_file);
        fseek(oracle_file, 0, SEEK_SET);
        char *oracle_text = malloc((size_t)oracle_bytes + 1);
        if (!oracle_text ||
            fread(oracle_text, 1, (size_t)oracle_bytes, oracle_file) !=
                (size_t)oracle_bytes) {
            fclose(oracle_file);
            free(oracle_text);
            goto cleanup;
        }
        oracle_text[oracle_bytes] = 0;
        fclose(oracle_file);
        char *arena = NULL;
        jval *root = json_parse(oracle_text, &arena);
        free(oracle_text);
        int prompt_count = 0, full_count = 0, tf_count = 0;
        prompt_ids = v4_oracle_read_ids(root, "prompt_ids", &prompt_count);
        full_ids = v4_oracle_read_ids(root, "full_ids", &full_count);
        tf_pred = v4_oracle_read_ids(root, "tf_pred", &tf_count);
        json_free(root);
        free(arena);
        if (!prompt_ids || !full_ids || !tf_pred ||
            prompt_count < 1 || full_count <= prompt_count ||
            tf_count < 1) {
            fprintf(stderr, "invalid oracle fixture: %s\n", cli.oracle_path);
            goto cleanup;
        }
        attention = calloc((size_t)config.num_hidden_layers, sizeof(*attention));
        if (!attention) goto cleanup;
        for (int layer = 0; layer < config.num_hidden_layers; layer++)
            if (coli_v4_window_attention_create(&attention[layer], &config))
                goto cleanup;

        int tf_limit = cli.teacher_forcing;
        if (tf_limit > tf_count) tf_limit = tf_count;
        if (tf_limit > full_count) tf_limit = full_count;
        int tf_matched = 0;
        if (v4_oracle_teacher_forcing(full_ids, full_count, tf_pred, tf_limit,
                                      attention, index, &config, experts,
                                      error, sizeof(error), &tf_matched)) {
            fprintf(stderr, "%s\n", error);
            goto cleanup;
        }
        printf("PREFILL (teacher-forcing) C vs oracle: %d/%d positions\n",
               tf_matched, tf_limit);

        for (int layer = 0; layer < config.num_hidden_layers; layer++)
            coli_v4_window_attention_reset(attention[layer]);
        int greedy_limit = cli.greedy;
        generated = malloc((size_t)(greedy_limit + 8) * sizeof(int));
        int got = v4_oracle_greedy_from_prompt(
            prompt_ids, prompt_count, generated, greedy_limit, attention,
            index, &config, experts, error, sizeof(error));
        if (got < 0) {
            fprintf(stderr, "%s\n", error);
            goto cleanup;
        }
        int greedy_matched = 0;
        int continue_count = full_count - prompt_count;
        int expected_count = continue_count < greedy_limit
            ? continue_count : greedy_limit;
        int compare = got < expected_count ? got : expected_count;
        for (int i = 0; i < compare; i++) {
            int expected = full_ids[prompt_count + i];
            if (generated[i] == expected) greedy_matched++;
            else
                fprintf(stderr,
                        "[ORACLE] greedy mismatch i=%d expected=%d got=%d\n",
                        i, expected, generated[i]);
        }
        int greedy_exact_length = got == expected_count;
        if (!greedy_exact_length)
            fprintf(stderr,
                    "[ORACLE] greedy length mismatch expected=%d got=%d "
                    "reference=%d requested=%d\n",
                    expected_count, got, continue_count, greedy_limit);
        printf("GREEDY C vs oracle: %d/%d tokens\n",
               greedy_matched, expected_count);
        free(full_ids); free(tf_pred);
        full_ids = NULL; tf_pred = NULL;
        (void)process_started;
        result = (tf_matched == tf_limit && greedy_exact_length &&
                  greedy_matched == expected_count) ? 0 : 1;
        goto cleanup;
    }

    char *prompt = NULL;
    size_t prompt_length = 0;
    if (coli_v4_prompt_build(&prompt, &prompt_length, cli.prompt,
                             cli.system_prompt, cli.prompt_mode) ||
        prompt_length > INT_MAX - 16) {
        free(prompt);
        fprintf(stderr, "cannot build DeepSeek V4 prompt\n");
        goto cleanup;
    }
    int target_only = !engine->dspark.enabled || cli.no_dspark;
    if (cli.no_dspark && engine->dspark.enabled)
        fprintf(stderr, "note: speculative drafting disabled by CLI\n");
    fprintf(stderr, "v4_cli mode=%s memory=%s target_only=%d\n",
            cli.prompt_mode == COLI_V4_PROMPT_RAW ? "raw" :
            cli.prompt_mode == COLI_V4_PROMPT_THINKING ? "thinking" : "chat",
            cli.memory_gib > 0.0 ? "limited" : "auto", target_only);

    int session_context = engine->runtime.context_tokens;
    ColiV4SessionCreateOptions session_opts = {
        .max_prompt_tokens = session_context,
        .max_new_tokens_cap = session_context,
    };
    if (coli_v4_session_create(&session, engine, &session_opts, error,
                               sizeof(error))) {
        free(prompt);
        fprintf(stderr, "%s\n", error);
        goto cleanup;
    }
    ColiV4SessionGenerateOptions gen_opts = {
        .max_new_tokens = max_new,
        .stop_at_sentence = stop_sentence,
        .no_dspark = cli.no_dspark,
    };
    ColiV4SessionGenerateStats gen_stats;
    memset(&gen_stats, 0, sizeof(gen_stats));
    /* clock_gettime inline rather than a helper: the amalgamated build compiles
     * this file once per COLI_V4_UNIT_*, and hot_now() is static to another one. */
    struct timespec tune_a, tune_b;
    clock_gettime(CLOCK_MONOTONIC, &tune_a);
    if (coli_v4_session_generate(session, prompt, prompt_length, &gen_opts,
                                 NULL, NULL, &gen_stats, error,
                                 sizeof(error))) {
        free(prompt);
        fprintf(stderr, "%s\n", error);
        goto cleanup;
    }
    free(prompt);
    prompt = NULL;

    char out_text[65536];
    size_t out_len = 0;
    coli_v4_session_generated_text(session, out_text, sizeof(out_text),
                                   &out_len);
    ColiExpertStoreStats stats_end = {0};
    experts->ops->stats(experts, &stats_end);
    fprintf(stderr, "v4_tokens prompt=%d generated=%d total=%d "
           "expert_requests=%llu hits=%llu misses=%llu hit_rate=%.3f "
           "bytes=%llu target_only=%d\n",
           gen_stats.prompt_tokens, gen_stats.generated_tokens,
           gen_stats.prompt_tokens + gen_stats.generated_tokens,
           (unsigned long long)stats_end.requests,
           (unsigned long long)stats_end.hits,
           (unsigned long long)stats_end.misses, stats_hit_rate(stats_end),
           (unsigned long long)stats_end.bytes_read, target_only);
    /* One line, every engine, one format: `coli tune` sweeps scheduling knobs and
     * needs tokens-and-elapsed to compare candidates. Before this only colibri
     * emitted a parseable throughput line (REPLAY decode), so the tuner was
     * GLM-only and bannered the right model while launching the wrong engine
     * (#898). stdout, which is what autotune captures -- the v4_* diagnostics
     * above go to stderr. Tokens and seconds rather than tok/s: the ratio is
     * derived by the caller at full precision (#852). */
    clock_gettime(CLOCK_MONOTONIC, &tune_b);
    printf("TUNE decode: %d tokens in %.3fs\n", gen_stats.generated_tokens,
           (double)(tune_b.tv_sec - tune_a.tv_sec) +
           (tune_b.tv_nsec - tune_a.tv_nsec) * 1e-9);
    fflush(stdout);
    fprintf(stderr, "generated_text=");
    if (out_len) fwrite(out_text, 1, out_len, stderr);
    fprintf(stderr, "\ntiming time_to_first_token=%.3fs after_first=%.3fs\n",
           gen_stats.time_to_first_token_sec, gen_stats.decode_sec);

    /* Alias session buffers for optional record-oracle path.
     * cleanup must destroy the session and must not free these aliases. */
    prompt_ids = session->prompt_ids;
    generated = session->generated;
    attention = session->attention;
    layers = session->config.num_hidden_layers;
    int prompt_count = session->prompt_count;
    int generated_count = session->generated_count;

    if (cli.record_oracle_path) {
        int full_count = prompt_count + generated_count;
        full_ids = malloc((size_t)full_count * sizeof(int));
        tf_pred = malloc((size_t)full_count * sizeof(int));
        if (!full_ids || !tf_pred) goto cleanup;
        memcpy(full_ids, prompt_ids, (size_t)prompt_count * sizeof(int));
        memcpy(full_ids + prompt_count, generated,
               (size_t)generated_count * sizeof(int));
        for (int layer = 0; layer < config.num_hidden_layers; layer++)
            coli_v4_window_attention_reset(attention[layer]);
        /* Rebuild tf_pred for the fixture (argmax at each position). */
        size_t hd_tf = (size_t)config.hc_mult * config.hidden_size;
        tf_state = malloc((size_t)full_count * hd_tf * sizeof(float));
        tf_next = malloc((size_t)full_count * hd_tf * sizeof(float));
        tf_hidden = malloc((size_t)config.hidden_size * sizeof(float));
        if (!tf_state || !tf_next || !tf_hidden) goto cleanup;
        for (int item = 0; item < full_count; item++)
            if (load_embedding(tf_state + (size_t)item * hd_tf, index, &config,
                               full_ids[item])) goto cleanup;
        if (target_batch(engine, &tf_state, &tf_next, attention, index, &config,
                         experts, full_ids, 0, full_count, error, sizeof(error))) {
            fprintf(stderr, "%s\n", error); goto cleanup;
        }
        for (int pos = 0; pos < full_count; pos++) {
            float logit = 0.0f;
            if (final_hidden(tf_hidden, tf_state + (size_t)pos * hd_tf,
                             index, &config, error, sizeof(error)) ||
                head_argmax(engine, tf_hidden, index, &config, &tf_pred[pos],
                            &logit))
                goto cleanup;
        }
        free(tf_state); free(tf_next); free(tf_hidden);
        tf_state = NULL; tf_next = NULL; tf_hidden = NULL;
        /* Chat-template prompt tokens need not be model-greedy; only score
         * the continuation window that record actually generated. */
        int tf_matched = 0, tf_total = generated_count;
        for (int i = 0; i < generated_count; i++) {
            int pos = prompt_count - 1 + i;
            if (pos >= 0 && pos < full_count - 1 &&
                tf_pred[pos] == full_ids[pos + 1])
                tf_matched++;
        }
        if (v4_oracle_write_json(cli.record_oracle_path, "coli-self",
                                 cli.model_dir, cli.prompt,
                                 prompt_ids, prompt_count,
                                 full_ids, full_count,
                                 tf_pred, full_count)) {
            fprintf(stderr, "cannot write oracle %s\n", cli.record_oracle_path);
            goto cleanup;
        }
        fprintf(stderr,
                "wrote oracle %s (source=coli-self, "
                "continuation_self_check=%d/%d)\n",
                cli.record_oracle_path, tf_matched, tf_total);
        free(full_ids); free(tf_pred);
        full_ids = NULL; tf_pred = NULL;
    }
    result = 0;
cleanup:
    v4_generate_cleanup(session, prompt_storage, engine, attention,
                        layers, prompt_ids, generated, state, next, hidden,
                        text, full_ids, tf_pred, tf_state,
                        tf_next, tf_hidden);
    return result;
}
#endif /* !COLI_V4_SKIP_GENERATE_MAIN */

#endif /* COLI_V4_UNIT_GENERATE_STATS */

#ifdef COLI_V4_UNIT_KV_CACHE
/* ######## deepseek_v4_kv_cache.c ######## */
#include "deepseek_v4_internal.h"

#include <stdlib.h>
#include <string.h>

struct ColiDeepSeekV4KVCache {
    int window_size;
    int compression_ratio;
    int head_dimension;
    int compressed_capacity;
    float *values;
};

int coli_v4_kv_cache_create(ColiDeepSeekV4KVCache **output,
                            int window_size, int compression_ratio,
                            int head_dimension, int max_context) {
    if (!output || window_size < 1 || compression_ratio < 1 ||
        head_dimension < 1 || max_context < 1)
        return -1;
    *output = NULL;
    ColiDeepSeekV4KVCache *cache = calloc(1, sizeof(*cache));
    if (!cache) return -1;
    cache->window_size = window_size;
    cache->compression_ratio = compression_ratio;
    cache->head_dimension = head_dimension;
    cache->compressed_capacity = max_context / compression_ratio;
    if (cache->compressed_capacity < 1) cache->compressed_capacity = 1;
    size_t count = (size_t)(window_size + cache->compressed_capacity) * head_dimension;
    cache->values = calloc(count, sizeof(*cache->values));
    if (!cache->values) {
        free(cache);
        return -1;
    }
    *output = cache;
    return 0;
}

void coli_v4_kv_cache_reset(ColiDeepSeekV4KVCache *cache) {
    if (!cache) return;
    size_t count = (size_t)(cache->window_size + cache->compressed_capacity) *
                   cache->head_dimension;
    memset(cache->values, 0, count * sizeof(*cache->values));
}

void coli_v4_kv_cache_destroy(ColiDeepSeekV4KVCache *cache) {
    if (!cache) return;
    free(cache->values);
    free(cache);
}

int coli_v4_kv_cache_put_window(ColiDeepSeekV4KVCache *cache,
                                int position, const float *kv) {
    if (!cache || !kv || position < 0) return -1;
    int slot = position % cache->window_size;
    memcpy(cache->values + (size_t)slot * cache->head_dimension, kv,
           (size_t)cache->head_dimension * sizeof(*kv));
    return slot;
}

int coli_v4_kv_cache_put_compressed(ColiDeepSeekV4KVCache *cache,
                                    int position, const float *kv) {
    if (!cache || !kv || position < 0 ||
        (position + 1) % cache->compression_ratio != 0)
        return -1;
    int slot = (position + 1) / cache->compression_ratio - 1;
    if (slot < 0 || slot >= cache->compressed_capacity) return -1;
    int combined = cache->window_size + slot;
    memcpy(cache->values + (size_t)combined * cache->head_dimension, kv,
           (size_t)cache->head_dimension * sizeof(*kv));
    return combined;
}

int coli_v4_kv_cache_indices(const ColiDeepSeekV4KVCache *cache,
                             int position, int *indices, size_t capacity) {
    if (!cache || !indices || position < 0) return -1;
    int compressed = (position + 1) / cache->compression_ratio;
    if (compressed > cache->compressed_capacity) return -1;
    size_t required = (size_t)cache->window_size + compressed;
    if (capacity < required) return -1;
    if (position < cache->window_size - 1) {
        for (int i = 0; i < cache->window_size; i++)
            indices[i] = i <= position ? i : -1;
    } else {
        int oldest = (position + 1) % cache->window_size;
        for (int i = 0; i < cache->window_size; i++)
            indices[i] = (oldest + i) % cache->window_size;
    }
    for (int i = 0; i < compressed; i++)
        indices[cache->window_size + i] = cache->window_size + i;
    return (int)required;
}

const float *coli_v4_kv_cache_values(const ColiDeepSeekV4KVCache *cache) {
    return cache ? cache->values : NULL;
}

int coli_v4_kv_cache_value_count(const ColiDeepSeekV4KVCache *cache) {
    return cache ? cache->window_size + cache->compressed_capacity : 0;
}
#endif /* COLI_V4_UNIT_KV_CACHE */

#ifdef COLI_V4_UNIT_ATTENTION_CACHE
/* ######## deepseek_v4_attention_cache.c ######## */
#include "deepseek_v4_internal.h"

#include <stdlib.h>

#include "deepseek_v4_internal.h"
#include "deepseek_v4_internal.h"

struct ColiDeepSeekV4AttentionCache {
    ColiDeepSeekV4KVCache *kv;
    int window_size;
    int compression_ratio;
    int head_dimension;
    int compressed_capacity;
};

int coli_v4_attention_cache_create(ColiDeepSeekV4AttentionCache **output,
                                   int window_size, int compression_ratio,
                                   int head_dimension, int max_context) {
    if (!output) return -1;
    *output = NULL;
    ColiDeepSeekV4AttentionCache *cache = calloc(1, sizeof(*cache));
    if (!cache) return -1;
    cache->window_size = window_size;
    cache->compression_ratio = compression_ratio;
    cache->head_dimension = head_dimension;
    cache->compressed_capacity = max_context / compression_ratio;
    if (cache->compressed_capacity < 1) cache->compressed_capacity = 1;
    if (coli_v4_kv_cache_create(&cache->kv, window_size, compression_ratio,
                                head_dimension, max_context) != 0) {
        free(cache);
        return -1;
    }
    *output = cache;
    return 0;
}

void coli_v4_attention_cache_reset(ColiDeepSeekV4AttentionCache *cache) {
    if (cache) coli_v4_kv_cache_reset(cache->kv);
}

void coli_v4_attention_cache_destroy(ColiDeepSeekV4AttentionCache *cache) {
    if (!cache) return;
    coli_v4_kv_cache_destroy(cache->kv);
    free(cache);
}

int coli_v4_attention_cache_step(ColiDeepSeekV4AttentionCache *cache,
                                 float *output, const float *query,
                                 const float *window_kv,
                                 const float *compressed_kv,
                                 const float *sinks, int heads,
                                 int position, float softmax_scale) {
    if (!cache || !output || !query || !window_kv || !sinks || heads < 1 ||
        position < 0 || position / cache->compression_ratio >= cache->compressed_capacity)
        return -1;
    int boundary = (position + 1) % cache->compression_ratio == 0;
    if (boundary != (compressed_kv != NULL)) return -1;
    if (coli_v4_kv_cache_put_window(cache->kv, position, window_kv) < 0)
        return -1;
    if (compressed_kv &&
        coli_v4_kv_cache_put_compressed(cache->kv, position, compressed_kv) < 0)
        return -1;
    size_t capacity = (size_t)cache->window_size + cache->compressed_capacity;
    int *indices = malloc(capacity * sizeof(*indices));
    if (!indices) return -1;
    int topk = coli_v4_kv_cache_indices(cache->kv, position, indices, capacity);
    int result = topk < 0 ? -1 : coli_v4_sparse_attention_ref(
        output, query, coli_v4_kv_cache_values(cache->kv), sinks, indices,
        heads, cache->head_dimension, coli_v4_kv_cache_value_count(cache->kv),
        topk, softmax_scale);
    free(indices);
    return result;
}
#endif /* COLI_V4_UNIT_ATTENTION_CACHE */

#ifdef COLI_V4_UNIT_EXPERT
/* ######## deepseek_v4_expert.c ######## */
#include "deepseek_v4_internal.h"

#include <stdlib.h>

#include "deepseek_v4_internal.h"
#include "native_quant.h"

int coli_v4_expert_forward_ref(float *output, const ColiExpertView *expert,
                               const float *input, float route_weight,
                               float swiglu_limit) {
    if (!output || !expert || !input || swiglu_limit < 0.0f ||
        expert->gate.rows != expert->up.rows ||
        expert->gate.columns != expert->up.columns ||
        expert->down.columns != expert->gate.rows ||
        expert->down.rows != expert->gate.columns)
        return -1;
    size_t intermediate = (size_t)expert->gate.rows;
    size_t output_size = (size_t)expert->down.rows;
    float *gate = malloc(intermediate * sizeof(*gate));
    float *up = malloc(intermediate * sizeof(*up));
    float *activated = malloc(intermediate * sizeof(*activated));
    if (!gate || !up || !activated) {
        free(gate);
        free(up);
        free(activated);
        return -1;
    }
    int result = coli_fp4_matvec_ref(gate, &expert->gate, input) ||
                 coli_fp4_matvec_ref(up, &expert->up, input);
    if (!result) {
        coli_bf16_round_array(gate, intermediate);
        coli_bf16_round_array(up, intermediate);
        result = coli_v4_swiglu(activated, gate, up,
                                (int)intermediate, swiglu_limit);
    }
    if (!result) {
        for (size_t index = 0; index < intermediate; index++)
            activated[index] = coli_bf16_round(
                activated[index] * route_weight);
        result = coli_fp4_matvec_ref(output, &expert->down, activated);
    }
    if (!result) coli_bf16_round_array(output, output_size);
    free(activated);
    free(up);
    free(gate);
    return result ? -1 : 0;
}

int coli_v4_shared_expert_forward_ref(float *output,
                                      const ColiTensorView *gate_weight,
                                      const ColiTensorView *down_weight,
                                      const ColiTensorView *up_weight,
                                      const float *input,
                                      float swiglu_limit) {
    if (!output || !gate_weight || !down_weight || !up_weight || !input ||
        gate_weight->format != COLI_TENSOR_FP8_E4M3_BLOCK ||
        down_weight->format != COLI_TENSOR_FP8_E4M3_BLOCK ||
        up_weight->format != COLI_TENSOR_FP8_E4M3_BLOCK ||
        gate_weight->rows != up_weight->rows ||
        gate_weight->columns != up_weight->columns ||
        down_weight->columns != gate_weight->rows ||
        down_weight->rows != gate_weight->columns)
        return -1;
    size_t intermediate = (size_t)gate_weight->rows;
    size_t output_size = (size_t)down_weight->rows;
    float *gate = malloc(intermediate * sizeof(*gate));
    float *up = malloc(intermediate * sizeof(*up));
    float *activated = malloc(intermediate * sizeof(*activated));
    if (!gate || !up || !activated) {
        free(activated); free(up); free(gate);
        return -1;
    }
    int result = coli_fp8_matvec_ref(gate, gate_weight, input) ||
                 coli_fp8_matvec_ref(up, up_weight, input);
    if (!result) {
        coli_bf16_round_array(gate, intermediate);
        coli_bf16_round_array(up, intermediate);
        result = coli_v4_swiglu(activated, gate, up,
                                (int)intermediate, swiglu_limit);
    }
    if (!result) {
        coli_bf16_round_array(activated, intermediate);
        result = coli_fp8_matvec_ref(output, down_weight, activated);
    }
    if (!result) coli_bf16_round_array(output, output_size);
    free(activated); free(up); free(gate);
    return result ? -1 : 0;
}
#endif /* COLI_V4_UNIT_EXPERT */

#ifdef COLI_V4_UNIT_EXPERT_STORE
/* ######## deepseek_v4_expert_store.c ######## */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "deepseek_v4_internal.h"

#include <assert.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#ifdef COLI_V4_EXPERIMENTAL_PREFETCH_BATCH
int coli_st_prefetch_many(
    const ColiSafetensorsIndex *index, const int *shards,
    const uint64_t *offsets, const size_t *lengths, size_t count);
#endif

enum { V4_W1 = 0, V4_W2 = 1, V4_W3 = 2, V4_MATRIX_COUNT = 3 };

typedef struct {
    const ColiSafetensorsTensor *weight[V4_MATRIX_COUNT];
    const ColiSafetensorsTensor *scale[V4_MATRIX_COUNT];
    int shard;
    uint64_t scale_offset;
    uint64_t scale_bytes;
    uint64_t weight_offset;
    uint64_t weight_bytes;
    uint64_t record_bytes;
} V4ExpertRecord;

typedef struct {
    int expert;
    unsigned references;
    uint64_t used;
    unsigned char *slab;
} V4ExpertSlot;

typedef struct {
    ColiSafetensorsIndex *index;
    int layers;
    int experts_per_layer;
    int slots_per_layer;
    uint64_t record_bytes;
    V4ExpertRecord *records;
    V4ExpertSlot *slots;
    uint64_t clock;
    unsigned active_leases;
    ColiExpertStoreStats stats;
    pthread_mutex_t mutex;
} V4ExpertStoreState;

static int set_error(char *error, size_t size, const char *format, ...) {
    if (error && size) {
        va_list args;
        va_start(args, format);
        vsnprintf(error, size, format, args);
        va_end(args);
    }
    return -1;
}

static int compare_tensors(const void *left, const void *right) {
    const ColiSafetensorsTensor *const *a = left;
    const ColiSafetensorsTensor *const *b = right;
    return ((*a)->off > (*b)->off) - ((*a)->off < (*b)->off);
}

static int contiguous_group(const ColiSafetensorsTensor *const input[3],
                            const ColiSafetensorsIndex *index,
                            int *shard, uint64_t *offset, uint64_t *bytes) {
    const ColiSafetensorsTensor *parts[3] = {input[0], input[1], input[2]};
    qsort(parts, 3, sizeof(parts[0]), compare_tensors);
    if (parts[0]->fd != parts[1]->fd || parts[1]->fd != parts[2]->fd ||
        parts[0]->off + parts[0]->nbytes != parts[1]->off ||
        parts[1]->off + parts[1]->nbytes != parts[2]->off)
        return -1;
    *shard = coli_st_tensor_shard(index, parts[0]);
    *offset = (uint64_t)parts[0]->off;
    *bytes = (uint64_t)(parts[2]->off + parts[2]->nbytes - parts[0]->off);
    return *shard < 0 ? -1 : 0;
}

static int validate_matrix(const ColiSafetensorsTensor *weight,
                           const ColiSafetensorsTensor *scale) {
    if (!weight || !scale || weight->dtype != COLI_ST_I8 ||
        scale->dtype != COLI_ST_F8_E8M0 || weight->rank != 2 || scale->rank != 2 ||
        weight->shape[0] != scale->shape[0] || weight->shape[1] <= 0 ||
        scale->shape[1] <= 0)
        return -1;
    int64_t logical_columns = weight->shape[1] * 2;
    return scale->shape[1] * 32 == logical_columns ? 0 : -1;
}

static int build_record(V4ExpertStoreState *state, int layer, int expert,
                        V4ExpertRecord *record, char *error, size_t error_size) {
    static const char *matrix_names[V4_MATRIX_COUNT] = {"w1", "w2", "w3"};
    char name[160];
    memset(record, 0, sizeof(*record));
    for (int matrix = 0; matrix < V4_MATRIX_COUNT; matrix++) {
        snprintf(name, sizeof(name), "layers.%d.ffn.experts.%d.%s.weight",
                 layer, expert, matrix_names[matrix]);
        record->weight[matrix] = coli_st_find(state->index, name);
        snprintf(name, sizeof(name), "layers.%d.ffn.experts.%d.%s.scale",
                 layer, expert, matrix_names[matrix]);
        record->scale[matrix] = coli_st_find(state->index, name);
        if (validate_matrix(record->weight[matrix], record->scale[matrix]) != 0)
            return set_error(error, error_size,
                             "invalid native FP4 expert matrix: layer=%d expert=%d %s",
                             layer, expert, matrix_names[matrix]);
    }
    int scale_shard = -1, weight_shard = -1;
    if (contiguous_group(record->scale, state->index,
                         &scale_shard, &record->scale_offset,
                         &record->scale_bytes) != 0 ||
        contiguous_group(record->weight, state->index,
                         &weight_shard, &record->weight_offset,
                         &record->weight_bytes) != 0 || scale_shard != weight_shard)
        return set_error(error, error_size,
                         "expert is not two contiguous ranges: layer=%d expert=%d",
                         layer, expert);
    record->shard = scale_shard;
    record->record_bytes = record->scale_bytes + record->weight_bytes;
    return 0;
}

static V4ExpertRecord *get_record(V4ExpertStoreState *state, ColiExpertKey key) {
    if (key.layer < 0 || key.layer >= state->layers || key.expert < 0 ||
        key.expert >= state->experts_per_layer)
        return NULL;
    return &state->records[(size_t)key.layer * state->experts_per_layer + key.expert];
}

static V4ExpertSlot *layer_slots(V4ExpertStoreState *state, int layer) {
    return state->slots + (size_t)layer * state->slots_per_layer;
}

static void fill_tensor_view(ColiTensorView *view,
                             const V4ExpertRecord *record,
                             const V4ExpertSlot *slot, int matrix) {
    const ColiSafetensorsTensor *weight = record->weight[matrix];
    const ColiSafetensorsTensor *scale = record->scale[matrix];
    memset(view, 0, sizeof(*view));
    view->format = COLI_TENSOR_FP4_NATIVE_BLOCK;
    view->scale_format = COLI_SCALE_UE8M0;
    view->data = slot->slab + record->scale_bytes +
                 ((uint64_t)weight->off - record->weight_offset);
    view->scales = slot->slab + ((uint64_t)scale->off - record->scale_offset);
    view->data_bytes = (size_t)weight->nbytes;
    view->scale_bytes = (size_t)scale->nbytes;
    view->rows = weight->shape[0];
    view->columns = weight->shape[1] * 2;
    view->block_rows = 1;
    view->block_columns = 32;
}

static int lookup(ColiExpertStore *store, ColiExpertKey key,
                  ColiExpertView *view) {
    if (!store || !store->state || !view) {
        if (view) memset(view, 0, sizeof(*view));
        return -1;
    }
    V4ExpertStoreState *state = store->state;
    V4ExpertRecord *record = get_record(state, key);
    if (!record) {
        memset(view, 0, sizeof(*view));
        return -1;
    }
    pthread_mutex_lock(&state->mutex);
    state->stats.requests++;
    V4ExpertSlot *slots = layer_slots(state, key.layer);
    V4ExpertSlot *slot = NULL;
    for (int i = 0; i < state->slots_per_layer; i++) {
        if (slots[i].slab && slots[i].expert == key.expert) {
            slot = &slots[i];
            state->stats.hits++;
            break;
        }
    }
    if (!slot) {
        for (int i = 0; i < state->slots_per_layer; i++) {
            if (!slots[i].references && (!slot || !slots[i].slab ||
                                         (slot->slab && slots[i].used < slot->used)))
                slot = &slots[i];
        }
        if (!slot) {
            pthread_mutex_unlock(&state->mutex);
            memset(view, 0, sizeof(*view));
            return -1;
        }
        if (!slot->slab) {
            slot->slab = malloc((size_t)state->record_bytes);
            if (!slot->slab) {
                pthread_mutex_unlock(&state->mutex);
                memset(view, 0, sizeof(*view));
                return -1;
            }
            state->stats.resident_bytes += state->record_bytes;
        }
        /* A short read must never expose a partially overwritten old slot. */
        slot->expert = -1;
        if (coli_st_read_at_streaming(
                state->index, record->shard, record->scale_offset,
                (size_t)record->scale_bytes, slot->slab) != 0 ||
            coli_st_read_at_streaming(
                state->index, record->shard, record->weight_offset,
                (size_t)record->weight_bytes,
                slot->slab + record->scale_bytes) != 0) {
            pthread_mutex_unlock(&state->mutex);
            memset(view, 0, sizeof(*view));
            return -1;
        }
        slot->expert = key.expert;
        state->stats.misses++;
        state->stats.bytes_read += record->record_bytes;
    }
    slot->references++;
    state->active_leases++;
    slot->used = ++state->clock;
    memset(view, 0, sizeof(*view));
    view->key = key;
    fill_tensor_view(&view->gate, record, slot, V4_W1);
    fill_tensor_view(&view->down, record, slot, V4_W2);
    fill_tensor_view(&view->up, record, slot, V4_W3);
    view->lease = slot;
    pthread_mutex_unlock(&state->mutex);
    return 0;
}

static void release(ColiExpertStore *store, ColiExpertView *view) {
    if (!store || !store->state || !view || !view->lease) {
        if (view) memset(view, 0, sizeof(*view));
        return;
    }
    V4ExpertStoreState *state = store->state;
    V4ExpertSlot *slot = view->lease;
    pthread_mutex_lock(&state->mutex);
    if (slot->references) slot->references--;
    if (state->active_leases) state->active_leases--;
    pthread_mutex_unlock(&state->mutex);
    memset(view, 0, sizeof(*view));
}

static int prefetch(ColiExpertStore *store, const ColiExpertKey *keys,
                    size_t count) {
    if (!store || !store->state || (!keys && count)) return 0;
    V4ExpertStoreState *state = store->state;
    int accepted = 0;
#ifdef COLI_V4_EXPERIMENTAL_PREFETCH_BATCH
    size_t capacity = count * 2, ranges = 0;
    int *shards = malloc(capacity * sizeof(*shards));
    uint64_t *offsets = malloc(capacity * sizeof(*offsets));
    size_t *lengths = malloc(capacity * sizeof(*lengths));
    int candidates = 0;
    if ((!shards || !offsets || !lengths) && capacity) {
        free(lengths); free(offsets); free(shards); return 0;
    }
    pthread_mutex_lock(&state->mutex);
    for (size_t i = 0; i < count; i++) {
        V4ExpertRecord *record = get_record(state, keys[i]);
        if (!record) continue;
        int resident = 0;
        V4ExpertSlot *slots = layer_slots(state, keys[i].layer);
        for (int slot = 0; slot < state->slots_per_layer; slot++)
            if (slots[slot].slab && slots[slot].expert == keys[i].expert) {
                resident = 1; break;
            }
        if (resident) continue;
        shards[ranges] = record->shard;
        offsets[ranges] = record->scale_offset;
        lengths[ranges++] = (size_t)record->scale_bytes;
        shards[ranges] = record->shard;
        offsets[ranges] = record->weight_offset;
        lengths[ranges++] = (size_t)record->weight_bytes;
        candidates++;
    }
    pthread_mutex_unlock(&state->mutex);
    if (candidates && !coli_st_prefetch_many(
            state->index, shards, offsets, lengths, ranges))
        accepted = candidates;
    free(lengths); free(offsets); free(shards);
#else
    for (size_t i = 0; i < count; i++) {
        V4ExpertRecord *record = get_record(state, keys[i]);
        if (!record) continue;
        if (coli_st_prefetch_at(state->index, record->shard, record->scale_offset,
                                (size_t)record->scale_bytes) == 0 &&
            coli_st_prefetch_at(state->index, record->shard, record->weight_offset,
                                (size_t)record->weight_bytes) == 0)
            accepted++;
    }
#endif
    pthread_mutex_lock(&state->mutex);
    state->stats.prefetched += (uint64_t)accepted;
    pthread_mutex_unlock(&state->mutex);
    return accepted;
}

static void stats(const ColiExpertStore *store, ColiExpertStoreStats *output) {
    if (!store || !store->state || !output) return;
    V4ExpertStoreState *state = store->state;
    pthread_mutex_lock(&state->mutex);
    *output = state->stats;
    pthread_mutex_unlock(&state->mutex);
}

static void destroy(ColiExpertStore *store) {
    if (!store) return;
    V4ExpertStoreState *state = store->state;
    if (state) {
        assert(state->active_leases == 0 && "destroy with active expert leases");
        for (int i = 0; i < state->layers * state->slots_per_layer; i++)
            free(state->slots[i].slab);   /* this store allocates slabs with
                                           * malloc only; the aligned path and
                                           * its compat_aligned_free live in the
                                           * hot rows16 store above. */
        pthread_mutex_destroy(&state->mutex);
        coli_st_index_close(state->index);
        free(state->records);
        free(state->slots);
        free(state);
    }
    free(store);
}

int coli_deepseek_v4_expert_store_open(
    const ColiDeepSeekV4ExpertStoreOptions *options, ColiExpertStore **output,
    char *error, size_t error_size) {
    static const ColiExpertStoreOps operations = {
        lookup, release, prefetch, stats, destroy
    };
    if (!options || !output || !options->model_dir || options->layers < 1 ||
        options->experts_per_layer < 1 || !options->cache_bytes)
        return set_error(error, error_size, "invalid DeepSeek-V4 ExpertStore options");
    *output = NULL;
    ColiExpertStore *store = calloc(1, sizeof(*store));
    V4ExpertStoreState *state = calloc(1, sizeof(*state));
    if (!store || !state) {
        free(store);
        free(state);
        return set_error(error, error_size, "out of memory creating ExpertStore");
    }
    pthread_mutex_init(&state->mutex, NULL);
    state->layers = options->layers;
    state->experts_per_layer = options->experts_per_layer;
    if (coli_st_index_open(&state->index, options->model_dir, error, error_size) != 0)
        goto fail;
    size_t record_count = (size_t)state->layers * state->experts_per_layer;
    state->records = calloc(record_count, sizeof(*state->records));
    if (!state->records) {
        set_error(error, error_size, "out of memory creating expert manifest");
        goto fail;
    }
    for (int layer = 0; layer < state->layers; layer++) {
        for (int expert = 0; expert < state->experts_per_layer; expert++) {
            V4ExpertRecord *record = &state->records[
                (size_t)layer * state->experts_per_layer + expert];
            if (build_record(state, layer, expert, record, error, error_size) != 0)
                goto fail;
            if (!state->record_bytes) state->record_bytes = record->record_bytes;
            if (record->record_bytes != state->record_bytes) {
                set_error(error, error_size, "non-uniform expert size at layer=%d expert=%d",
                          layer, expert);
                goto fail;
            }
        }
    }
    state->slots_per_layer = (int)(options->cache_bytes /
        ((uint64_t)state->layers * state->record_bytes));
    int minimum_slots = state->experts_per_layer < 6
        ? state->experts_per_layer : 6;
    if (state->slots_per_layer < minimum_slots) {
        set_error(error, error_size,
                  "cache budget cannot hold %d active experts per layer "
                  "(need %llu bytes)", minimum_slots,
                  (unsigned long long)((uint64_t)state->layers * minimum_slots *
                                       state->record_bytes));
        goto fail;
    }
    if (state->slots_per_layer > state->experts_per_layer)
        state->slots_per_layer = state->experts_per_layer;
    state->slots = calloc((size_t)state->layers * state->slots_per_layer,
                          sizeof(*state->slots));
    if (!state->slots) {
        set_error(error, error_size, "out of memory creating expert cache slots");
        goto fail;
    }
    for (int i = 0; i < state->layers * state->slots_per_layer; i++)
        state->slots[i].expert = -1;
    state->stats.capacity_bytes = (uint64_t)state->layers *
                                  state->slots_per_layer * state->record_bytes;
    store->ops = &operations;
    store->state = state;
    *output = store;
    return 0;

fail:
    if (state->slots) free(state->slots);
    free(state->records);
    coli_st_index_close(state->index);
    pthread_mutex_destroy(&state->mutex);
    free(state);
    free(store);
    return -1;
}
#endif /* COLI_V4_UNIT_EXPERT_STORE */

#ifdef COLI_V4_UNIT_LAYER
/* ######## deepseek_v4_layer.c ######## */
#include "deepseek_v4_internal.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int set_error(char *error, size_t size, const char *format, ...) {
    if (error && size) {
        va_list arguments;
        va_start(arguments, format);
        vsnprintf(error, size, format, arguments);
        va_end(arguments);
    }
    return -1;
}

static int add_spec(ColiDeepSeekV4LayerPlan *plan, ColiSafetensorsDType dtype,
                    int rank, const int64_t *shape, const char *suffix,
                    char *error, size_t error_size) {
    if (plan->tensor_count >= COLI_V4_MAX_LAYER_TENSORS)
        return set_error(error, error_size, "too many tensors in layer %d", plan->layer);
    ColiDeepSeekV4TensorSpec *spec = &plan->tensors[plan->tensor_count++];
    int written = snprintf(spec->name, sizeof(spec->name), "layers.%d.%s",
                           plan->layer, suffix);
    if (written < 0 || (size_t)written >= sizeof(spec->name))
        return set_error(error, error_size, "tensor name is too long: %s", suffix);
    spec->dtype = dtype;
    spec->rank = rank;
    memcpy(spec->shape, shape, (size_t)rank * sizeof(*shape));
    return 0;
}

static int add_1d(ColiDeepSeekV4LayerPlan *plan, ColiSafetensorsDType dtype,
                  int64_t d0, const char *name, char *error, size_t size) {
    int64_t shape[] = {d0};
    return add_spec(plan, dtype, 1, shape, name, error, size);
}

static int add_2d(ColiDeepSeekV4LayerPlan *plan, ColiSafetensorsDType dtype,
                  int64_t d0, int64_t d1, const char *name,
                  char *error, size_t size) {
    int64_t shape[] = {d0, d1};
    return add_spec(plan, dtype, 2, shape, name, error, size);
}

static int add_fp8(ColiDeepSeekV4LayerPlan *plan, int64_t rows, int64_t columns,
                   const char *prefix, char *error, size_t size) {
    char name[128];
    snprintf(name, sizeof(name), "%s.weight", prefix);
    if (add_2d(plan, COLI_ST_F8_E4M3, rows, columns, name, error, size) != 0) return -1;
    snprintf(name, sizeof(name), "%s.scale", prefix);
    return add_2d(plan, COLI_ST_F8_E8M0, (rows + 127) / 128,
                  (columns + 127) / 128, name, error, size);
}

#define ADD(call) do { if ((call) != 0) return -1; } while (0)

int coli_v4_layer_plan(ColiDeepSeekV4LayerPlan *plan,
                       const ColiDeepSeekV4Config *config, int layer,
                       char *error, size_t error_size) {
    if (!plan || !config || layer < 0 || layer >= config->num_hidden_layers ||
        layer >= config->compress_ratio_count)
        return set_error(error, error_size, "invalid DeepSeek-V4 layer plan arguments");
    memset(plan, 0, sizeof(*plan));
    plan->layer = layer;
    plan->compression_ratio = config->compress_ratios[layer];
    plan->uses_hash_router = layer < config->num_hash_layers;
    plan->has_compressor = plan->compression_ratio != 0;
    plan->has_indexer = plan->compression_ratio == 4;

    const int64_t hidden = config->hidden_size;
    const int64_t heads = config->num_attention_heads;
    const int64_t head_dim = config->head_dim;
    const int64_t q_rank = config->q_lora_rank;
    if (config->o_groups < 1 || heads % config->o_groups != 0)
        return set_error(error, error_size, "unsupported grouped-output attention dimensions");
    const int64_t o_group_width =
        (heads / config->o_groups) * head_dim;
    const int64_t o_width = (int64_t)config->o_groups * config->o_lora_rank;
    const int64_t experts = config->n_routed_experts;
    const int64_t moe = config->moe_intermediate_size;
    const int64_t hc = config->hc_mult;
    const int64_t hc_params = (2 + hc) * hc;

    ADD(add_1d(plan, COLI_ST_F32, heads, "attn.attn_sink", error, error_size));
    ADD(add_1d(plan, COLI_ST_BF16, head_dim, "attn.kv_norm.weight", error, error_size));
    ADD(add_1d(plan, COLI_ST_BF16, q_rank, "attn.q_norm.weight", error, error_size));
    ADD(add_fp8(plan, head_dim, hidden, "attn.wkv", error, error_size));
    ADD(add_fp8(plan, o_width, o_group_width, "attn.wo_a", error, error_size));
    ADD(add_fp8(plan, hidden, o_width, "attn.wo_b", error, error_size));
    ADD(add_fp8(plan, q_rank, hidden, "attn.wq_a", error, error_size));
    ADD(add_fp8(plan, heads * head_dim, q_rank, "attn.wq_b", error, error_size));
    ADD(add_1d(plan, COLI_ST_BF16, hidden, "attn_norm.weight", error, error_size));

    if (plan->has_compressor) {
        int64_t ratio = plan->compression_ratio;
        int64_t coff = ratio == 4 ? 2 : 1;
        ADD(add_2d(plan, COLI_ST_F32, ratio, coff * head_dim,
                   "attn.compressor.ape", error, error_size));
        ADD(add_1d(plan, COLI_ST_BF16, head_dim,
                   "attn.compressor.norm.weight", error, error_size));
        ADD(add_2d(plan, COLI_ST_BF16, coff * head_dim, hidden,
                   "attn.compressor.wgate.weight", error, error_size));
        ADD(add_2d(plan, COLI_ST_BF16, coff * head_dim, hidden,
                   "attn.compressor.wkv.weight", error, error_size));
    }
    if (plan->has_indexer) {
        int64_t ih = config->index_head_dim;
        int64_t in = config->index_n_heads;
        ADD(add_2d(plan, COLI_ST_F32, 4, 2 * ih,
                   "attn.indexer.compressor.ape", error, error_size));
        ADD(add_1d(plan, COLI_ST_BF16, ih,
                   "attn.indexer.compressor.norm.weight", error, error_size));
        ADD(add_2d(plan, COLI_ST_BF16, 2 * ih, hidden,
                   "attn.indexer.compressor.wgate.weight", error, error_size));
        ADD(add_2d(plan, COLI_ST_BF16, 2 * ih, hidden,
                   "attn.indexer.compressor.wkv.weight", error, error_size));
        ADD(add_2d(plan, COLI_ST_BF16, in, hidden,
                   "attn.indexer.weights_proj.weight", error, error_size));
        ADD(add_fp8(plan, in * ih, q_rank, "attn.indexer.wq_b", error, error_size));
    }

    ADD(add_2d(plan, COLI_ST_BF16, experts, hidden,
               "ffn.gate.weight", error, error_size));
    if (plan->uses_hash_router)
        ADD(add_2d(plan, COLI_ST_I64, config->vocab_size,
                   config->num_experts_per_tok, "ffn.gate.tid2eid", error, error_size));
    else
        ADD(add_1d(plan, COLI_ST_F32, experts, "ffn.gate.bias", error, error_size));
    ADD(add_fp8(plan, moe, hidden, "ffn.shared_experts.w1", error, error_size));
    ADD(add_fp8(plan, hidden, moe, "ffn.shared_experts.w2", error, error_size));
    ADD(add_fp8(plan, moe, hidden, "ffn.shared_experts.w3", error, error_size));
    ADD(add_1d(plan, COLI_ST_BF16, hidden, "ffn_norm.weight", error, error_size));

    ADD(add_1d(plan, COLI_ST_F32, hc_params, "hc_attn_base", error, error_size));
    ADD(add_2d(plan, COLI_ST_F32, hc_params, hc * hidden,
               "hc_attn_fn", error, error_size));
    ADD(add_1d(plan, COLI_ST_F32, 3, "hc_attn_scale", error, error_size));
    ADD(add_1d(plan, COLI_ST_F32, hc_params, "hc_ffn_base", error, error_size));
    ADD(add_2d(plan, COLI_ST_F32, hc_params, hc * hidden,
               "hc_ffn_fn", error, error_size));
    ADD(add_1d(plan, COLI_ST_F32, 3, "hc_ffn_scale", error, error_size));
    return 0;
}

int coli_v4_layer_validate(const ColiDeepSeekV4LayerPlan *plan,
                           const ColiSafetensorsIndex *index,
                           ColiDeepSeekV4LayerStats *stats,
                           char *error, size_t error_size) {
    if (!plan || !index)
        return set_error(error, error_size, "invalid DeepSeek-V4 layer validation arguments");
    ColiDeepSeekV4LayerStats local = {0};
    for (size_t i = 0; i < plan->tensor_count; i++) {
        const ColiDeepSeekV4TensorSpec *spec = &plan->tensors[i];
        const ColiSafetensorsTensor *tensor = coli_st_find(index, spec->name);
        if (!tensor)
            return set_error(error, error_size, "missing tensor: %s", spec->name);
        if (tensor->dtype != spec->dtype || tensor->rank != spec->rank)
            return set_error(error, error_size, "dtype/rank mismatch: %s", spec->name);
        for (int dimension = 0; dimension < spec->rank; dimension++)
            if (tensor->shape[dimension] != spec->shape[dimension])
                return set_error(error, error_size, "shape mismatch: %s", spec->name);
        local.tensor_count++;
        uint64_t resident_bytes = tensor->dtype == COLI_ST_F8_E8M0
            ? (uint64_t)tensor->numel * sizeof(float) : (uint64_t)tensor->nbytes;
        local.total_bytes += resident_bytes;
        switch (tensor->dtype) {
            case COLI_ST_BF16: local.bf16_bytes += tensor->nbytes; break;
            case COLI_ST_F32: local.f32_bytes += tensor->nbytes; break;
            case COLI_ST_F8_E4M3: local.fp8_weight_bytes += tensor->nbytes; break;
            case COLI_ST_F8_E8M0: local.fp8_scale_bytes += resident_bytes; break;
            case COLI_ST_I64: local.i64_bytes += tensor->nbytes; break;
            default: break;
        }
    }
    if (stats) *stats = local;
    return 0;
}

void coli_v4_layer_free(ColiV4Engine *engine,
                        ColiDeepSeekV4LayerWeights *weights) {
    (void)engine;
    if (!weights) return;
    for (size_t i = 0; i < weights->plan.tensor_count; i++) free(weights->data[i]);
    memset(weights, 0, sizeof(*weights));
}

int coli_v4_layer_load(ColiV4Engine *engine,
                       ColiDeepSeekV4LayerWeights *weights,
                       const ColiDeepSeekV4Config *config,
                       const ColiSafetensorsIndex *index, int layer,
                       char *error, size_t error_size) {
    (void)engine;
    if (!weights) return set_error(error, error_size, "missing layer weights output");
    memset(weights, 0, sizeof(*weights));
    if (coli_v4_layer_plan(&weights->plan, config, layer, error, error_size) != 0 ||
        coli_v4_layer_validate(&weights->plan, index, &weights->stats,
                               error, error_size) != 0)
        return -1;
    for (size_t i = 0; i < weights->plan.tensor_count; i++) {
        const ColiDeepSeekV4TensorSpec *spec = &weights->plan.tensors[i];
        const ColiSafetensorsTensor *tensor = coli_st_find(index, spec->name);
        size_t resident_bytes = tensor->dtype == COLI_ST_F8_E8M0
            ? (size_t)tensor->numel * sizeof(float) : (size_t)tensor->nbytes;
        weights->data[i] = malloc(resident_bytes);
        if (!weights->data[i]) {
            coli_v4_layer_free(NULL, weights);
            return set_error(error, error_size, "out of memory loading: %s", spec->name);
        }
        int read_failed = tensor->dtype == COLI_ST_F8_E8M0
            ? st_read_scale_f32((ColiSafetensorsIndex *)index, spec->name,
                                weights->data[i], tensor->numel, 0) != tensor->numel
            : coli_st_read_tensor(index, tensor, weights->data[i]) != 0;
        if (read_failed) {
            coli_v4_layer_free(NULL, weights);
            return set_error(error, error_size, "cannot read tensor: %s", spec->name);
        }
    }
    return 0;
}

const void *coli_v4_layer_data(const ColiDeepSeekV4LayerWeights *weights,
                               const char *name,
                               const ColiDeepSeekV4TensorSpec **spec) {
    if (spec) *spec = NULL;
    if (!weights || !name) return NULL;
    for (size_t i = 0; i < weights->plan.tensor_count; i++) {
        if (strcmp(weights->plan.tensors[i].name, name) == 0) {
            if (spec) *spec = &weights->plan.tensors[i];
            return weights->data[i];
        }
    }
    return NULL;
}
#endif /* COLI_V4_UNIT_LAYER */

#ifdef COLI_V4_UNIT_CONFIG
/* ######## deepseek_v4_config.c ######## */
#include "deepseek_v4_internal.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "json.h"

static int set_error(char *error, size_t size, const char *format, ...) {
    if (error && size) {
        va_list arguments;
        va_start(arguments, format);
        vsnprintf(error, size, format, arguments);
        va_end(arguments);
    }
    return -1;
}

static int json_int_value(const jval *value, int *output) {
    if (!value || value->t != J_NUM || !isfinite(value->num) ||
        floor(value->num) != value->num ||
        value->num < (double)INT_MIN || value->num > (double)INT_MAX)
        return -1;
    *output = (int)value->num;
    return 0;
}

static int required_int(jval *root, const char *name, int *output,
                        char *error, size_t error_size) {
    if (json_int_value(json_get(root, name), output) != 0)
        return set_error(error, error_size, "invalid integer config field: %s", name);
    return 0;
}

static int optional_int(jval *root, const char *name, int *output,
                        char *error, size_t error_size) {
    jval *value = json_get(root, name);
    if (!value) return 0;
    if (json_int_value(value, output) != 0)
        return set_error(error, error_size,
                         "invalid integer config field: %s", name);
    return 0;
}

static int required_float(jval *root, const char *name, float *output,
                          char *error, size_t error_size) {
    jval *value = json_get(root, name);
    if (!value || value->t != J_NUM || !isfinite(value->num) ||
        fabs(value->num) > (double)FLT_MAX)
        return set_error(error, error_size, "invalid numeric config field: %s", name);
    *output = (float)value->num;
    return 0;
}

static int require_string(jval *root, const char *name, const char *expected,
                          char *error, size_t error_size) {
    jval *value = json_get(root, name);
    if (!value || value->t != J_STR || strcmp(value->str, expected))
        return set_error(error, error_size, "unsupported %s (expected %s)",
                         name, expected);
    return 0;
}

int coli_v4_config_parse(ColiDeepSeekV4Config *config, const char *json,
                         char *error, size_t error_size) {
    if (!config || !json)
        return set_error(error, error_size, "invalid DeepSeek-V4 config arguments");
    memset(config, 0, sizeof(*config));
    char *arena = NULL;
    jval *root = json_parse(json, &arena);
    if (!root || root->t != J_OBJ) {
        json_free(root);
        free(arena);
        return set_error(error, error_size, "DeepSeek-V4 config is not an object");
    }
    int failed =
        require_string(root, "model_type", "deepseek_v4", error, error_size) ||
        require_string(root, "expert_dtype", "fp4", error, error_size) ||
        require_string(root, "scoring_func", "sqrtsoftplus", error, error_size) ||
        require_string(root, "topk_method", "noaux_tc", error, error_size) ||
        required_int(root, "hidden_size", &config->hidden_size, error, error_size) ||
        required_int(root, "num_hidden_layers", &config->num_hidden_layers, error, error_size) ||
        required_int(root, "num_attention_heads", &config->num_attention_heads, error, error_size) ||
        required_int(root, "head_dim", &config->head_dim, error, error_size) ||
        required_int(root, "q_lora_rank", &config->q_lora_rank, error, error_size) ||
        required_int(root, "qk_rope_head_dim", &config->qk_rope_head_dim, error, error_size) ||
        required_int(root, "o_groups", &config->o_groups, error, error_size) ||
        required_int(root, "o_lora_rank", &config->o_lora_rank, error, error_size) ||
        required_int(root, "sliding_window", &config->sliding_window, error, error_size) ||
        required_int(root, "index_n_heads", &config->index_n_heads, error, error_size) ||
        required_int(root, "index_head_dim", &config->index_head_dim, error, error_size) ||
        required_int(root, "index_topk", &config->index_topk, error, error_size) ||
        required_int(root, "n_routed_experts", &config->n_routed_experts, error, error_size) ||
        required_int(root, "num_experts_per_tok", &config->num_experts_per_tok, error, error_size) ||
        required_int(root, "n_shared_experts", &config->n_shared_experts, error, error_size) ||
        required_int(root, "moe_intermediate_size", &config->moe_intermediate_size, error, error_size) ||
        required_int(root, "num_hash_layers", &config->num_hash_layers, error, error_size) ||
        required_int(root, "num_nextn_predict_layers", &config->num_nextn_predict_layers, error, error_size) ||
        required_int(root, "hc_mult", &config->hc_mult, error, error_size) ||
        required_int(root, "hc_sinkhorn_iters", &config->hc_sinkhorn_iters, error, error_size) ||
        required_int(root, "vocab_size", &config->vocab_size, error, error_size) ||
        required_int(root, "max_position_embeddings", &config->max_position_embeddings, error, error_size) ||
        required_float(root, "rms_norm_eps", &config->rms_norm_eps, error, error_size) ||
        required_float(root, "hc_eps", &config->hc_eps, error, error_size) ||
        required_float(root, "routed_scaling_factor", &config->routed_scaling_factor, error, error_size) ||
        required_float(root, "swiglu_limit", &config->swiglu_limit, error, error_size) ||
        required_float(root, "rope_theta", &config->rope_theta, error, error_size) ||
        required_float(root, "compress_rope_theta", &config->compress_rope_theta, error, error_size);
    if (failed) {
        json_free(root);
        free(arena);
        return -1;
    }
    if (optional_int(root, "dspark_block_size", &config->dspark_block_size,
                     error, error_size) ||
        optional_int(root, "dspark_noise_token_id",
                     &config->dspark_noise_token_id, error, error_size) ||
        optional_int(root, "dspark_markov_rank",
                     &config->dspark_markov_rank, error, error_size)) {
        json_free(root);
        free(arena);
        return -1;
    }
    jval *rope = json_get(root, "rope_scaling");
    if (!rope || rope->t != J_OBJ ||
        required_int(rope, "original_max_position_embeddings",
                     &config->original_max_position_embeddings, error, error_size) ||
        required_int(rope, "beta_fast", &config->rope_beta_fast, error, error_size) ||
        required_int(rope, "beta_slow", &config->rope_beta_slow, error, error_size) ||
        required_float(rope, "factor", &config->rope_factor, error, error_size)) {
        json_free(root);
        free(arena);
        return -1;
    }
    jval *ratios = json_get(root, "compress_ratios");
    if (!ratios || ratios->t != J_ARR || ratios->len < 1 ||
        ratios->len > COLI_V4_MAX_LAYERS) {
        json_free(root);
        free(arena);
        return set_error(error, error_size, "invalid compress_ratios");
    }
    config->compress_ratio_count = ratios->len;
    for (int index = 0; index < ratios->len; index++) {
        if (json_int_value(ratios->kids[index],
                           &config->compress_ratios[index]) != 0) {
            json_free(root);
            free(arena);
            return set_error(error, error_size, "invalid compress ratio");
        }
    }
    jval *quantization = json_get(root, "quantization_config");
    if (!quantization || quantization->t != J_OBJ ||
        require_string(quantization, "fmt", "e4m3", error, error_size) ||
        require_string(quantization, "scale_fmt", "ue8m0", error, error_size)) {
        json_free(root);
        free(arena);
        return -1;
    }
    if (config->hidden_size < 1 || config->num_hidden_layers < 1 ||
        config->num_attention_heads < 1 || config->n_routed_experts < 1 ||
        config->num_experts_per_tok < 1 ||
        config->num_experts_per_tok > config->n_routed_experts ||
        config->n_shared_experts != 1 || config->hc_mult < 1 ||
        config->compress_ratio_count < config->num_hidden_layers) {
        json_free(root);
        free(arena);
        return set_error(error, error_size, "inconsistent DeepSeek-V4 config dimensions");
    }
    json_free(root);
    free(arena);
    return 0;
}

int coli_v4_config_load(ColiDeepSeekV4Config *config, const char *model_dir,
                        char *error, size_t error_size) {
    if (!config || !model_dir)
        return set_error(error, error_size, "invalid DeepSeek-V4 config path");
    size_t path_length = strlen(model_dir) + sizeof("/config.json");
    char *path = malloc(path_length);
    if (!path) return set_error(error, error_size, "out of memory building config path");
    snprintf(path, path_length, "%s/config.json", model_dir);
    FILE *stream = fopen(path, "rb");
    if (!stream) {
        int result = set_error(error, error_size, "cannot open %s", path);
        free(path);
        return result;
    }
    fseek(stream, 0, SEEK_END);
    long length = ftell(stream);
    rewind(stream);
    if (length < 1) {
        fclose(stream);
        free(path);
        return set_error(error, error_size, "empty config: %s", model_dir);
    }
    char *text = malloc((size_t)length + 1);
    if (!text || fread(text, 1, (size_t)length, stream) != (size_t)length) {
        free(text);
        fclose(stream);
        free(path);
        return set_error(error, error_size, "cannot read config: %s", model_dir);
    }
    text[length] = 0;
    fclose(stream);
    int result = coli_v4_config_parse(config, text, error, error_size);
    free(text);
    free(path);
    return result;
}
#endif /* COLI_V4_UNIT_CONFIG */


#ifdef COLI_V4_UNIT_NATIVE_QUANT
/*
 * The former native_quant.c, native_quant_dual.c, native_quant_batch.c,
 * and native_quant_fp4_rows16.c implementations are folded into this engine.
 * The private native_quant_parallel.c and native_quant_batch_avx512.c kernels
 * duplicated shared quant.h FP8 work and were removed during the fold; the
 * units below dispatch through the shared matmul_fp8 implementation instead.
 */
#include "native_quant.h"
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#endif
#include "quant.h"
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#ifdef __AVX2__
#include <immintrin.h>
#endif

float coli_e8m0_decode(uint8_t value) {
    if (value == 0xff) return NAN;
    return ldexpf(1.0f, (int)value - 127);
}

float coli_e2m1_decode(uint8_t nibble) {
    static const float values[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
        0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
    };
    return values[nibble & 15];
}

float coli_e4m3fn_decode(uint8_t value) {
    int sign = value >> 7;
    int exponent = (value >> 3) & 15;
    int mantissa = value & 7;
    if (exponent == 15 && mantissa == 7) return NAN;
    float number;
    if (!exponent)
        number = ldexpf((float)mantissa, -9);
    else
        number = ldexpf(1.0f + (float)mantissa / 8.0f, exponent - 7);
    return sign ? -number : number;
}

uint8_t coli_e4m3fn_encode(float value) {
    if (isnan(value)) return 0x7f;
    int negative = signbit(value) != 0;
    float magnitude = fabsf(value);
    if (!magnitude) return negative ? 0x80 : 0;
    if (magnitude >= 448.0f) return (uint8_t)((negative ? 0x80 : 0) | 0x7e);

    uint8_t best = 0;
#if FLT_RADIX == 2 && FLT_MANT_DIG == 24 && FLT_MAX_EXP == 128
    /* Exact binary32 round-to-nearest-even conversion in constant time. */
    if (magnitude < 0.015625f) {
        float scaled = magnitude * 512.0f;
        uint8_t rounded = (uint8_t)scaled;
        float fraction = scaled - rounded;
        if (fraction > 0.5f || (fraction == 0.5f && (rounded & 1)))
            rounded++;
        best = rounded;
    } else {
        uint32_t bits;
        memcpy(&bits, &magnitude, sizeof(bits));
        int exponent = (int)((bits >> 23) & 0xff) - 127;
        uint32_t significand = 0x800000u | (bits & 0x7fffffu);
        uint32_t rounded = significand >> 20;
        uint32_t remainder = significand & 0xfffffu;
        if (remainder > 0x80000u ||
            (remainder == 0x80000u && (rounded & 1u)))
            rounded++;
        if (rounded == 16u) {
            rounded = 8u;
            exponent++;
        }
        best = (uint8_t)((exponent + 7) * 8 + (int)rounded - 8);
    }
#else
    float best_distance = FLT_MAX;
    for (uint8_t code = 0; code <= 0x7e; code++) {
        float candidate = coli_e4m3fn_decode(code);
        float distance = fabsf(candidate - magnitude);
        if (distance < best_distance ||
            (distance == best_distance && !(code & 1) && (best & 1))) {
            best = code;
            best_distance = distance;
        }
    }
#endif
    return (uint8_t)(best | (negative ? 0x80 : 0));
}

float coli_bf16_round(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7f800000u) != 0x7f800000u) {
        uint32_t tie = (bits >> 16) & 1u;
        bits += 0x7fffu + tie;
    }
    bits &= 0xffff0000u;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

float coli_bf16_decode(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float output;
    memcpy(&output, &bits, sizeof(output));
    return output;
}

void coli_bf16_round_array(float *values, size_t count) {
    if (!values) return;
    for (size_t index = 0; index < count; index++)
        values[index] = coli_bf16_round(values[index]);
}

static int ceil_log2_positive(float value) {
    int exponent;
    float fraction = frexpf(value, &exponent);
    return fraction == 0.5f ? exponent - 1 : exponent;
}

int coli_fp8_activation_qdq_ref(float *output, uint8_t *scales,
                                const float *input, size_t length,
                                size_t block_size) {
    if (!output || !scales || !input || !length || !block_size)
        return -1;
    for (size_t base = 0; base < length; base += block_size) {
        size_t count = length - base < block_size ? length - base : block_size;
        float maximum = 0.0f;
        for (size_t i = 0; i < count; i++)
            maximum = fmaxf(maximum, fabsf(input[base + i]));
        maximum = fmaxf(maximum, 1e-4f);
        int scale_exponent = ceil_log2_positive(maximum / 448.0f);
        if (scale_exponent < -127) scale_exponent = -127;
        if (scale_exponent > 127) scale_exponent = 127;
        uint8_t encoded_scale = (uint8_t)(scale_exponent + 127);
        float scale = coli_e8m0_decode(encoded_scale);
        scales[base / block_size] = encoded_scale;
        for (size_t i = 0; i < count; i++) {
            float normalized = fmaxf(-448.0f,
                                     fminf(448.0f, input[base + i] / scale));
            output[base + i] = coli_e4m3fn_decode(
                coli_e4m3fn_encode(normalized)) * scale;
        }
    }
    return 0;
}

int coli_fp4_activation_qdq_ref(float *output, uint8_t *scales,
                                const float *input, size_t length,
                                size_t block_size) {
    if (!output || !scales || !input || !length || !block_size)
        return -1;
    for (size_t base = 0; base < length; base += block_size) {
        size_t count = length - base < block_size ? length - base : block_size;
        float maximum = 0.0f;
        for (size_t i = 0; i < count; i++)
            maximum = fmaxf(maximum, fabsf(input[base + i]));
        maximum = fmaxf(maximum, 6.0f * ldexpf(1.0f, -126));
        int exponent = ceil_log2_positive(maximum / 6.0f);
        if (exponent < -127) exponent = -127;
        if (exponent > 127) exponent = 127;
        scales[base / block_size] = (uint8_t)(exponent + 127);
        float scale = coli_e8m0_decode(scales[base / block_size]);
        for (size_t i = 0; i < count; i++) {
            float value = fmaxf(-6.0f, fminf(6.0f, input[base + i] / scale));
            int best = 0;
            float distance = fabsf(value - coli_e2m1_decode(0));
            for (int code = 1; code < 16; code++) {
                float candidate = fabsf(value - coli_e2m1_decode((uint8_t)code));
                if (candidate < distance) {
                    distance = candidate;
                    best = code;
                }
            }
            output[base + i] = coli_e2m1_decode((uint8_t)best) * scale;
        }
    }
    return 0;
}

int coli_hadamard_bf16_ref(float *values, size_t length) {
    if (!values || !length || (length & (length - 1))) return -1;
    for (size_t width = 1; width < length; width *= 2)
        for (size_t base = 0; base < length; base += 2 * width)
            for (size_t i = 0; i < width; i++) {
                float left = values[base + i];
                float right = values[base + width + i];
                values[base + i] = left + right;
                values[base + width + i] = left - right;
            }
    float scale = 1.0f / sqrtf((float)length);
    for (size_t i = 0; i < length; i++)
        values[i] = coli_bf16_round(values[i] * scale);
    return 0;
}

int coli_fp4_matvec_ref(float *output, const ColiTensorView *weight,
                        const float *input) {
    if (!output || !weight || !input ||
        weight->format != COLI_TENSOR_FP4_NATIVE_BLOCK ||
        weight->scale_format != COLI_SCALE_UE8M0 ||
        !weight->data || !weight->scales || weight->rows < 1 ||
        weight->columns < 1 || weight->columns % 128 ||
        weight->block_rows != 1 || weight->block_columns != 32)
        return -1;
    size_t rows = (size_t)weight->rows;
    size_t columns = (size_t)weight->columns;
    size_t packed_stride = columns / 2;
    size_t scale_stride = columns / 32;
    if (weight->data_bytes != rows * packed_stride ||
        weight->scale_bytes != rows * scale_stride)
        return -1;
    float *activation = malloc(columns * sizeof(*activation));
    uint8_t *activation_scales = malloc(columns / 128);
    if (!activation || !activation_scales) {
        free(activation);
        free(activation_scales);
        return -1;
    }
    if (coli_fp8_activation_qdq_ref(activation, activation_scales,
                                    input, columns, 128) != 0) {
        free(activation);
        free(activation_scales);
        return -1;
    }
    matmul_mxfp4(output, activation, weight->data, weight->scales,
                 1, (int)columns, (int)rows);
    free(activation_scales);
    free(activation);
    return 0;
}

int coli_fp8_matvec_ref(float *output, const ColiTensorView *weight,
                        const float *input) {
    if (!output || !weight || !input ||
        weight->format != COLI_TENSOR_FP8_E4M3_BLOCK ||
        weight->scale_format != COLI_SCALE_F32 ||
        !weight->data || !weight->scales || weight->rows < 1 ||
        weight->columns < 1 || weight->columns % 128 ||
        (weight->block_rows != 128 && weight->block_rows != 8) ||
        weight->block_columns != 128)
        return -1;
    size_t rows = (size_t)weight->rows;
    size_t columns = (size_t)weight->columns;
    size_t scale_rows = (rows + 127) / 128;
    size_t scale_columns = columns / 128;
    if (weight->data_bytes != rows * columns ||
        weight->scale_bytes != scale_rows * scale_columns * sizeof(float))
        return -1;
    float *activation = malloc(columns * sizeof(*activation));
    uint8_t *activation_scales = malloc(scale_columns);
    if (!activation || !activation_scales) {
        free(activation);
        free(activation_scales);
        return -1;
    }
    if (coli_fp8_activation_qdq_ref(activation, activation_scales,
                                    input, columns, 128) != 0) {
        free(activation);
        free(activation_scales);
        return -1;
    }
#ifdef __AVX2__
    if (weight->block_rows == 8) {
        if (rows % 8) {
            free(activation_scales); free(activation); return -1;
        }
        float fp8[256];
        for (int code = 0; code < 256; code++)
            fp8[code] = coli_e4m3fn_decode((uint8_t)code);
        const uint8_t *data = weight->data;
        const float *scales = weight->scales;
        #pragma omp parallel for schedule(static)
        for (int64_t tile = 0; tile < weight->rows / 8; tile++) {
            __m256 sum = _mm256_setzero_ps();
            size_t scale_row = ((size_t)tile * 8) / 128;
            for (size_t base = 0; base < columns; base += 128) {
                __m256 scale = _mm256_set1_ps(
                    scales[scale_row * scale_columns + base / 128]);
                for (size_t offset = 0; offset < 128; offset++) {
                    size_t column = base + offset;
                    __m128i bytes = _mm_loadl_epi64((const __m128i *)(data +
                        ((size_t)tile * columns + column) * 8));
                    __m256i codes = _mm256_cvtepu8_epi32(bytes);
                    __m256 values = _mm256_i32gather_ps(fp8, codes, 4);
                    __m256 x = _mm256_set1_ps(activation[column]);
                    sum = _mm256_add_ps(sum, _mm256_mul_ps(
                        _mm256_mul_ps(x, values), scale));
                }
            }
            _mm256_storeu_ps(output + (size_t)tile * 8, sum);
        }
        free(activation_scales); free(activation); return 0;
    }
#endif
    matmul_fp8(output, activation, weight->data, weight->scales,
               1, (int)columns, (int)rows);
    free(activation_scales);
    free(activation);
    return 0;
}
#endif /* COLI_V4_UNIT_NATIVE_QUANT */

#ifdef COLI_V4_UNIT_NATIVE_QUANT_DUAL
/* Folded into the DeepSeek V4 engine translation units. */
#include "native_quant_dual.h"

#include <stdint.h>
#include <stdlib.h>
#ifdef __AVX2__
#include <immintrin.h>
#endif

#include "native_quant.h"
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#endif
#include "quant.h"
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

static int dual_same_shape(const ColiTensorView *a, const ColiTensorView *b) {
    return a && b && a->rows == b->rows && a->columns == b->columns &&
           a->block_rows == b->block_rows &&
           a->block_columns == b->block_columns;
}

int coli_fp4_dual_matvec_ref(float *output_a, float *output_b,
                             const ColiTensorView *a,
                             const ColiTensorView *b,
                             const float *input) {
    if (!output_a || !output_b || !input || !dual_same_shape(a, b) ||
        a->format != COLI_TENSOR_FP4_NATIVE_BLOCK ||
        b->format != COLI_TENSOR_FP4_NATIVE_BLOCK ||
        a->scale_format != COLI_SCALE_UE8M0 ||
        b->scale_format != COLI_SCALE_UE8M0 || !a->data || !b->data ||
        !a->scales || !b->scales || a->rows < 1 || a->columns < 1 ||
        a->columns % 128 || a->block_rows != 1 || a->block_columns != 32)
        return -1;
    size_t rows = (size_t)a->rows, columns = (size_t)a->columns;
    size_t packed_stride = columns / 2, scale_stride = columns / 32;
    if (a->data_bytes != rows * packed_stride ||
        b->data_bytes != rows * packed_stride ||
        a->scale_bytes != rows * scale_stride ||
        b->scale_bytes != rows * scale_stride) return -1;
    float *activation = malloc(columns * sizeof(*activation));
    uint8_t *activation_scales = malloc(columns / 128);
    if (!activation || !activation_scales) {
        free(activation_scales); free(activation); return -1;
    }
    if (coli_fp8_activation_qdq_ref(activation, activation_scales,
                                    input, columns, 128) != 0) {
        free(activation_scales); free(activation); return -1;
    }
    matmul_mxfp4(output_a, activation, a->data, a->scales,
                 1, (int)columns, (int)rows);
    matmul_mxfp4(output_b, activation, b->data, b->scales,
                 1, (int)columns, (int)rows);
    free(activation_scales); free(activation); return 0;
}

int coli_fp8_dual_matvec_ref(float *output_a, float *output_b,
                             const ColiTensorView *a,
                             const ColiTensorView *b,
                             const float *input) {
    if (!output_a || !output_b || !input || !dual_same_shape(a, b) ||
        a->format != COLI_TENSOR_FP8_E4M3_BLOCK ||
        b->format != COLI_TENSOR_FP8_E4M3_BLOCK ||
        a->scale_format != COLI_SCALE_F32 ||
        b->scale_format != COLI_SCALE_F32 || !a->data || !b->data ||
        !a->scales || !b->scales || a->rows < 1 || a->columns < 1 ||
        a->columns % 128 ||
        (a->block_rows != 128 && a->block_rows != 8) ||
        a->block_columns != 128) return -1;
    size_t rows = (size_t)a->rows, columns = (size_t)a->columns;
    size_t scale_rows = (rows + 127) / 128, scale_columns = columns / 128;
    if (a->data_bytes != rows * columns || b->data_bytes != rows * columns ||
        a->scale_bytes != scale_rows * scale_columns * sizeof(float) ||
        b->scale_bytes != scale_rows * scale_columns * sizeof(float)) return -1;
    float *activation = malloc(columns * sizeof(*activation));
    uint8_t *activation_scales = malloc(scale_columns);
    if (!activation || !activation_scales) {
        free(activation_scales); free(activation); return -1;
    }
    if (coli_fp8_activation_qdq_ref(activation, activation_scales,
                                    input, columns, 128) != 0) {
        free(activation_scales); free(activation); return -1;
    }
#ifdef __AVX2__
    if (a->block_rows == 8) {
        if (rows % 8) {
            free(activation_scales); free(activation); return -1;
        }
        float fp8[256];
        for (int code = 0; code < 256; code++)
            fp8[code] = coli_e4m3fn_decode((uint8_t)code);
        const uint8_t *data_a = a->data, *data_b = b->data;
        const float *scales_a = a->scales, *scales_b = b->scales;
        #pragma omp parallel for schedule(static)
        for (int64_t tile = 0; tile < a->rows / 8; tile++) {
            __m256 sum_a = _mm256_setzero_ps();
            __m256 sum_b = _mm256_setzero_ps();
            size_t scale_row = ((size_t)tile * 8) / 128;
            for (size_t base = 0; base < columns; base += 128) {
                size_t scale_index = scale_row * scale_columns + base / 128;
                __m256 scale_a = _mm256_set1_ps(scales_a[scale_index]);
                __m256 scale_b = _mm256_set1_ps(scales_b[scale_index]);
                for (size_t offset = 0; offset < 128; offset++) {
                    size_t column = base + offset;
                    size_t packed = ((size_t)tile * columns + column) * 8;
                    __m256i codes_a = _mm256_cvtepu8_epi32(_mm_loadl_epi64(
                        (const __m128i *)(data_a + packed)));
                    __m256i codes_b = _mm256_cvtepu8_epi32(_mm_loadl_epi64(
                        (const __m128i *)(data_b + packed)));
                    __m256 values_a = _mm256_i32gather_ps(fp8, codes_a, 4);
                    __m256 values_b = _mm256_i32gather_ps(fp8, codes_b, 4);
                    __m256 x = _mm256_set1_ps(activation[column]);
                    sum_a = _mm256_add_ps(sum_a, _mm256_mul_ps(
                        _mm256_mul_ps(x, values_a), scale_a));
                    sum_b = _mm256_add_ps(sum_b, _mm256_mul_ps(
                        _mm256_mul_ps(x, values_b), scale_b));
                }
            }
            _mm256_storeu_ps(output_a + (size_t)tile * 8, sum_a);
            _mm256_storeu_ps(output_b + (size_t)tile * 8, sum_b);
        }
        free(activation_scales); free(activation); return 0;
    }
#endif
    matmul_fp8(output_a, activation, a->data, a->scales,
               1, (int)columns, (int)rows);
    matmul_fp8(output_b, activation, b->data, b->scales,
               1, (int)columns, (int)rows);
    free(activation_scales); free(activation); return 0;
}
#endif /* COLI_V4_UNIT_NATIVE_QUANT_DUAL */

#ifdef COLI_V4_UNIT_NATIVE_QUANT_BATCH
/* Folded into the DeepSeek V4 engine translation units. */
#include "native_quant_batch.h"

#include <stdint.h>
#include <stdlib.h>
#ifdef __AVX2__
#include <immintrin.h>
#endif

#include "native_quant.h"
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#endif
#include "quant.h"
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

int coli_fp8_matmul_batch_ref(float *outputs, const ColiTensorView *weight,
                              const float *inputs, int batch) {
    if (!outputs || !weight || !inputs || batch < 1 || batch > 64 ||
        weight->format != COLI_TENSOR_FP8_E4M3_BLOCK ||
        weight->scale_format != COLI_SCALE_F32 ||
        !weight->data || !weight->scales || weight->rows < 1 ||
        weight->columns < 1 || weight->columns % 128 ||
        (weight->block_rows != 128 && weight->block_rows != 8) ||
        weight->block_columns != 128) return -1;
    size_t rows = (size_t)weight->rows, columns = (size_t)weight->columns;
    size_t scale_rows = (rows + 127) / 128;
    size_t scale_columns = columns / 128;
    if (weight->data_bytes != rows * columns ||
        weight->scale_bytes != scale_rows * scale_columns * sizeof(float)) return -1;
    float *activations = malloc((size_t)batch * columns * sizeof(*activations));
    uint8_t *activation_scales = malloc((size_t)batch * scale_columns);
    if (!activations || !activation_scales) {
        free(activation_scales); free(activations); return -1;
    }
    for (int item = 0; item < batch; item++)
        if (coli_fp8_activation_qdq_ref(
                activations + (size_t)item * columns,
                activation_scales + (size_t)item * scale_columns,
                inputs + (size_t)item * columns, columns, 128) != 0) {
            free(activation_scales); free(activations); return -1;
        }
#ifdef __AVX2__
    if (weight->block_rows == 8) {
        if (rows % 8) {
            free(activation_scales); free(activations); return -1;
        }
        float fp8[256];
        for (int code = 0; code < 256; code++)
            fp8[code] = coli_e4m3fn_decode((uint8_t)code);
        const uint8_t *data = weight->data;
        const float *scales = weight->scales;
        #pragma omp parallel for schedule(static)
        for (int64_t tile = 0; tile < weight->rows / 8; tile++) {
            __m256 sums[64];
            for (int item = 0; item < batch; item++)
                sums[item] = _mm256_setzero_ps();
            size_t scale_row = ((size_t)tile * 8) / 128;
            for (size_t base = 0; base < columns; base += 128) {
                __m256 scale = _mm256_set1_ps(
                    scales[scale_row * scale_columns + base / 128]);
                for (size_t offset = 0; offset < 128; offset++) {
                    size_t column = base + offset;
                    __m128i bytes = _mm_loadl_epi64((const __m128i *)(data +
                        ((size_t)tile * columns + column) * 8));
                    __m256i codes = _mm256_cvtepu8_epi32(bytes);
                    __m256 values = _mm256_mul_ps(
                        _mm256_i32gather_ps(fp8, codes, 4), scale);
                    for (int item = 0; item < batch; item++) {
                        __m256 x = _mm256_set1_ps(
                            activations[(size_t)item * columns + column]);
                        sums[item] = _mm256_add_ps(
                            sums[item], _mm256_mul_ps(x, values));
                    }
                }
            }
            for (int item = 0; item < batch; item++)
                _mm256_storeu_ps(outputs + (size_t)item * rows +
                                 (size_t)tile * 8, sums[item]);
        }
        free(activation_scales); free(activations); return 0;
    }
#endif
    matmul_fp8(outputs, activations, weight->data, weight->scales,
               batch, (int)columns, (int)rows);
    free(activation_scales); free(activations); return 0;
}

int coli_fp4_matmul_batch_ref(float *outputs, const ColiTensorView *weight,
                              const float *inputs, int batch) {
    if (!outputs || !weight || !inputs || batch < 1 || batch > 64 ||
        weight->format != COLI_TENSOR_FP4_NATIVE_BLOCK ||
        weight->scale_format != COLI_SCALE_UE8M0 ||
        !weight->data || !weight->scales || weight->rows < 1 ||
        weight->columns < 1 || weight->columns % 128 ||
        weight->block_rows != 1 || weight->block_columns != 32) return -1;
    size_t rows = (size_t)weight->rows, columns = (size_t)weight->columns;
    size_t packed_stride = columns / 2, scale_stride = columns / 32;
    if (weight->data_bytes != rows * packed_stride ||
        weight->scale_bytes != rows * scale_stride) return -1;
    float *activations = malloc((size_t)batch * columns * sizeof(*activations));
    uint8_t *activation_scales = malloc((size_t)batch * columns / 128);
    if (!activations || !activation_scales) {
        free(activation_scales); free(activations); return -1;
    }
    for (int item = 0; item < batch; item++)
        if (coli_fp8_activation_qdq_ref(
                activations + (size_t)item * columns,
                activation_scales + (size_t)item * columns / 128,
                inputs + (size_t)item * columns, columns, 128) != 0) {
            free(activation_scales); free(activations); return -1;
        }
    matmul_mxfp4(outputs, activations, weight->data, weight->scales,
                 batch, (int)columns, (int)rows);
    free(activation_scales); free(activations); return 0;
}
#endif /* COLI_V4_UNIT_NATIVE_QUANT_BATCH */

#ifdef COLI_V4_UNIT_NATIVE_QUANT_ROWS16
/* Folded into the DeepSeek V4 engine translation units. */
#include "native_quant_fp4_rows16.h"

/* TODO(upstream-fmt7-rows16): quant.h now owns the canonical fmt=7 MXFP4
 * decoder. This private rows16 repack/fused kernel remains only because the
 * shared layer has no resident-cache rows16 layout yet. Migrate and delete it
 * when that performance API lands upstream. */

#if defined(__AVX512F__) || defined(__AVX2__)
#include <immintrin.h>
#elif defined(__aarch64__)
#include <arm_neon.h>
#endif
#include <stdint.h>
#include <stdlib.h>

static int source_valid(const ColiTensorView *weight) {
    if (!weight || weight->format != COLI_TENSOR_FP4_NATIVE_BLOCK ||
        weight->scale_format != COLI_SCALE_UE8M0 || !weight->data ||
        !weight->scales || weight->rows < 1 || weight->rows % 16 ||
        weight->columns < 1 || weight->columns % 128 ||
        weight->block_rows != 1 || weight->block_columns != 32) return 0;
    size_t rows = (size_t)weight->rows, columns = (size_t)weight->columns;
    return weight->data_bytes == rows * columns / 2 &&
           weight->scale_bytes == rows * columns / 32;
}

static int packed_valid(const ColiTensorView *weight) {
    if (!weight || weight->format != COLI_TENSOR_FP4_NATIVE_BLOCK ||
        weight->scale_format != COLI_SCALE_UE8M0 || !weight->data ||
        !weight->scales || weight->rows < 1 || weight->rows % 16 ||
        weight->columns < 1 || weight->columns % 128 ||
        weight->block_rows != 16 || weight->block_columns != 32) return 0;
    size_t rows = (size_t)weight->rows, columns = (size_t)weight->columns;
    return weight->data_bytes == rows * columns / 2 &&
           weight->scale_bytes == rows * columns / 32;
}

int coli_fp4_pack_rows16_v10(unsigned char *packed_data,
                             unsigned char *packed_scales,
                             const ColiTensorView *source) {
    if (!packed_data || !packed_scales || !source_valid(source)) return -1;
    const unsigned char *data = source->data;
    const unsigned char *scales = source->scales;
    size_t rows = (size_t)source->rows, columns = (size_t)source->columns;
    size_t data_stride = columns / 2, scale_stride = columns / 32;
    for (size_t row = 0; row < rows; row++) {
        size_t tile = row / 16, lane = row % 16;
        for (size_t column = 0; column < data_stride; column++)
            packed_data[(tile * data_stride + column) * 16 + lane] =
                data[row * data_stride + column];
        for (size_t column = 0; column < scale_stride; column++)
            packed_scales[(tile * scale_stride + column) * 16 + lane] =
                scales[row * scale_stride + column];
    }
    return 0;
}

#ifdef __AVX512F__
static void decode_tables(__m512 *fp4_table, float e8[256]) {
    float fp4[16];
    for (int i = 0; i < 16; i++) fp4[i] = coli_e2m1_decode((uint8_t)i);
    for (int i = 0; i < 256; i++) e8[i] = coli_e8m0_decode((uint8_t)i);
    *fp4_table = _mm512_loadu_ps(fp4);
}

static __m512 decode_fp4_rows16(const unsigned char *data, int high,
                                __m512 fp4_table) {
    __m512i codes = _mm512_cvtepu8_epi32(
        _mm_loadu_si128((const __m128i *)data));
    codes = high ? _mm512_srli_epi32(codes, 4)
                 : _mm512_and_si512(codes, _mm512_set1_epi32(15));
    return _mm512_permutexvar_ps(codes, fp4_table);
}

static __m512 decode_scales_rows16(const unsigned char *scales,
                                   const float e8[256]) {
    __m512i codes = _mm512_cvtepu8_epi32(
        _mm_loadu_si128((const __m128i *)scales));
    return _mm512_i32gather_ps(codes, e8, 4);
}
#elif defined(__AVX2__)
typedef struct Avx2Rows16Tables {
    float fp4[16];
    float e8[256];
} Avx2Rows16Tables;

static void avx2_rows16_tables(Avx2Rows16Tables *tables) {
    for (int i = 0; i < 16; i++)
        tables->fp4[i] = coli_e2m1_decode((uint8_t)i);
    for (int i = 0; i < 256; i++)
        tables->e8[i] = coli_e8m0_decode((uint8_t)i);
}

static inline void avx2_decode_rows16(__m256 values[2],
                                      const unsigned char *data, int high,
                                      const Avx2Rows16Tables *tables) {
    __m128i bytes = _mm_loadu_si128((const __m128i *)data);
    __m256i lo = _mm256_cvtepu8_epi32(bytes);
    __m256i hi = _mm256_cvtepu8_epi32(_mm_srli_si128(bytes, 8));
    __m256i mask = _mm256_set1_epi32(15);
    if (high) {
        lo = _mm256_srli_epi32(lo, 4);
        hi = _mm256_srli_epi32(hi, 4);
    } else {
        lo = _mm256_and_si256(lo, mask);
        hi = _mm256_and_si256(hi, mask);
    }
    values[0] = _mm256_i32gather_ps(tables->fp4, lo, 4);
    values[1] = _mm256_i32gather_ps(tables->fp4, hi, 4);
}

static inline void avx2_decode_scales(__m256 values[2],
                                      const unsigned char *codes,
                                      const Avx2Rows16Tables *tables) {
    __m128i bytes = _mm_loadu_si128((const __m128i *)codes);
    __m256i lo = _mm256_cvtepu8_epi32(bytes);
    __m256i hi = _mm256_cvtepu8_epi32(_mm_srli_si128(bytes, 8));
    values[0] = _mm256_i32gather_ps(tables->e8, lo, 4);
    values[1] = _mm256_i32gather_ps(tables->e8, hi, 4);
}
#elif defined(__aarch64__)
/* The 16 packed rows map onto four float32x4 accumulators. Per-row work is
 * (activation * value) * scale added once per column, columns ascending —
 * the exact operation sequence of the scalar reference and of the AVX-512
 * kernel, so all three produce bit-identical rows. No fused multiply-add. */

typedef struct NeonRows16Tables {
    /* The 16-value E2M1 table is exactly 64 bytes, so one four-register TBL
     * can gather whole floats: spread[group] replicates each of four code
     * lanes into four byte positions and byte_offsets walks float bytes. */
    uint8x16x4_t fp4_bytes;
    uint8x16_t spread[4];
    uint8x16_t byte_offsets;
    float e8[256];
} NeonRows16Tables;

static void neon_rows16_tables(NeonRows16Tables *tables) {
    float fp4[16];
    for (int i = 0; i < 16; i++) fp4[i] = coli_e2m1_decode((uint8_t)i);
    for (int i = 0; i < 256; i++) tables->e8[i] = coli_e8m0_decode((uint8_t)i);
    const unsigned char *bytes = (const unsigned char *)fp4;
    for (int group = 0; group < 4; group++)
        tables->fp4_bytes.val[group] = vld1q_u8(bytes + 16 * group);
    for (int group = 0; group < 4; group++) {
        uint8_t lanes[16];
        for (int byte = 0; byte < 16; byte++)
            lanes[byte] = (uint8_t)(4 * group + byte / 4);
        tables->spread[group] = vld1q_u8(lanes);
    }
    uint8_t offsets[16];
    for (int byte = 0; byte < 16; byte++) offsets[byte] = (uint8_t)(byte % 4);
    tables->byte_offsets = vld1q_u8(offsets);
}

static inline void neon_rows16_block_scales(float32x4_t scales[4],
                                            const unsigned char *codes,
                                            const NeonRows16Tables *tables) {
    float decoded[16];
    for (int lane = 0; lane < 16; lane++)
        decoded[lane] = tables->e8[codes[lane]];
    for (int group = 0; group < 4; group++)
        scales[group] = vld1q_f32(decoded + 4 * group);
}

static inline void neon_rows16_accumulate(float32x4_t sums[4],
                                          uint8x16_t codes, float activation,
                                          const float32x4_t scales[4],
                                          const NeonRows16Tables *tables) {
    uint8x16_t byte_base = vshlq_n_u8(codes, 2);
    float32x4_t x = vdupq_n_f32(activation);
    for (int group = 0; group < 4; group++) {
        uint8x16_t gather = vaddq_u8(
            vqtbl1q_u8(byte_base, tables->spread[group]),
            tables->byte_offsets);
        float32x4_t values =
            vreinterpretq_f32_u8(vqtbl4q_u8(tables->fp4_bytes, gather));
        sums[group] = vaddq_f32(
            sums[group], vmulq_f32(vmulq_f32(x, values), scales[group]));
    }
}
#endif

int coli_fp4_matvec_rows16_v10(float *output,
                               const ColiTensorView *weight,
                               const float *input) {
#if !defined(COLI_FP4_ROWS16_KERNEL)
    (void)output; (void)weight; (void)input; return -1;
#elif defined(__AVX512F__)
    if (!output || !input || !packed_valid(weight)) return -1;
    size_t columns = (size_t)weight->columns;
    size_t data_stride = columns / 2, scale_stride = columns / 32;
    float *activation = malloc(columns * sizeof(*activation));
    uint8_t *activation_scales = malloc(columns / 128);
    if (!activation || !activation_scales) {
        free(activation_scales); free(activation); return -1;
    }
    if (coli_fp8_activation_qdq_ref(activation, activation_scales,
                                    input, columns, 128)) {
        free(activation_scales); free(activation); return -1;
    }
    __m512 fp4_table; float e8[256]; decode_tables(&fp4_table, e8);
    const unsigned char *data = weight->data, *scales = weight->scales;
    #pragma omp parallel for schedule(static)
    for (int64_t tile = 0; tile < weight->rows / 16; tile++) {
        __m512 sum = _mm512_setzero_ps();
        for (size_t base = 0; base < columns; base += 32) {
            __m512 scale = decode_scales_rows16(
                scales + ((size_t)tile * scale_stride + base / 32) * 16, e8);
            for (size_t offset = 0; offset < 32; offset++) {
                size_t column = base + offset;
                const unsigned char *codes = data +
                    ((size_t)tile * data_stride + column / 2) * 16;
                __m512 values = decode_fp4_rows16(codes, column & 1, fp4_table);
                __m512 product = _mm512_mul_ps(
                    _mm512_mul_ps(_mm512_set1_ps(activation[column]), values),
                    scale);
                sum = _mm512_add_ps(sum, product);
            }
        }
        _mm512_storeu_ps(output + (size_t)tile * 16, sum);
    }
    free(activation_scales); free(activation); return 0;
#elif defined(__AVX2__)
    if (!output || !input || !packed_valid(weight)) return -1;
    size_t columns = (size_t)weight->columns;
    size_t data_stride = columns / 2, scale_stride = columns / 32;
    float *activation = malloc(columns * sizeof(*activation));
    uint8_t *activation_scales = malloc(columns / 128);
    if (!activation || !activation_scales) {
        free(activation_scales); free(activation); return -1;
    }
    if (coli_fp8_activation_qdq_ref(activation, activation_scales,
                                    input, columns, 128)) {
        free(activation_scales); free(activation); return -1;
    }
    Avx2Rows16Tables tables; avx2_rows16_tables(&tables);
    const unsigned char *data = weight->data, *scales = weight->scales;
    #pragma omp parallel for schedule(static)
    for (int64_t tile = 0; tile < weight->rows / 16; tile++) {
        __m256 sum[2] = {_mm256_setzero_ps(), _mm256_setzero_ps()};
        for (size_t base = 0; base < columns; base += 32) {
            __m256 scale[2];
            avx2_decode_scales(
                scale,
                scales + ((size_t)tile * scale_stride + base / 32) * 16,
                &tables);
            for (size_t offset = 0; offset < 32; offset++) {
                size_t column = base + offset;
                const unsigned char *codes = data +
                    ((size_t)tile * data_stride + column / 2) * 16;
                __m256 values[2];
                avx2_decode_rows16(values, codes, column & 1, &tables);
                __m256 x = _mm256_set1_ps(activation[column]);
                for (int half = 0; half < 2; half++)
                    sum[half] = _mm256_add_ps(sum[half], _mm256_mul_ps(
                        _mm256_mul_ps(x, values[half]), scale[half]));
            }
        }
        _mm256_storeu_ps(output + (size_t)tile * 16, sum[0]);
        _mm256_storeu_ps(output + (size_t)tile * 16 + 8, sum[1]);
    }
    free(activation_scales); free(activation); return 0;
#else /* __aarch64__ */
    if (!output || !input || !packed_valid(weight)) return -1;
    size_t columns = (size_t)weight->columns;
    size_t data_stride = columns / 2, scale_stride = columns / 32;
    float *activation = malloc(columns * sizeof(*activation));
    uint8_t *activation_scales = malloc(columns / 128);
    if (!activation || !activation_scales) {
        free(activation_scales); free(activation); return -1;
    }
    if (coli_fp8_activation_qdq_ref(activation, activation_scales,
                                    input, columns, 128)) {
        free(activation_scales); free(activation); return -1;
    }
    NeonRows16Tables tables;
    neon_rows16_tables(&tables);
    const unsigned char *data = weight->data, *scales = weight->scales;
    #pragma omp parallel for schedule(static)
    for (int64_t tile = 0; tile < weight->rows / 16; tile++) {
        float32x4_t sums[4] = {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
                               vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};
        for (size_t base = 0; base < columns; base += 32) {
            float32x4_t block_scales[4];
            neon_rows16_block_scales(
                block_scales,
                scales + ((size_t)tile * scale_stride + base / 32) * 16,
                &tables);
            for (size_t offset = 0; offset < 32; offset += 2) {
                const unsigned char *codes = data +
                    ((size_t)tile * data_stride + (base + offset) / 2) * 16;
                uint8x16_t bytes = vld1q_u8(codes);
                neon_rows16_accumulate(
                    sums, vandq_u8(bytes, vdupq_n_u8(15)),
                    activation[base + offset], block_scales, &tables);
                neon_rows16_accumulate(
                    sums, vshrq_n_u8(bytes, 4),
                    activation[base + offset + 1], block_scales, &tables);
            }
        }
        for (int group = 0; group < 4; group++)
            vst1q_f32(output + (size_t)tile * 16 + 4 * group, sums[group]);
    }
    free(activation_scales); free(activation); return 0;
#endif
}

int coli_fp4_dual_matvec_rows16_v10(float *output_a, float *output_b,
                                    const ColiTensorView *a,
                                    const ColiTensorView *b,
                                    const float *input) {
#if !defined(COLI_FP4_ROWS16_KERNEL)
    (void)output_a; (void)output_b; (void)a; (void)b; (void)input; return -1;
#elif defined(__AVX512F__)
    if (!output_a || !output_b || !input || !packed_valid(a) ||
        !packed_valid(b) || a->rows != b->rows ||
        a->columns != b->columns) return -1;
    size_t columns = (size_t)a->columns;
    size_t data_stride = columns / 2, scale_stride = columns / 32;
    float *activation = malloc(columns * sizeof(*activation));
    uint8_t *activation_scales = malloc(columns / 128);
    if (!activation || !activation_scales) {
        free(activation_scales); free(activation); return -1;
    }
    if (coli_fp8_activation_qdq_ref(activation, activation_scales,
                                    input, columns, 128)) {
        free(activation_scales); free(activation); return -1;
    }
    __m512 fp4_table; float e8[256]; decode_tables(&fp4_table, e8);
    const unsigned char *data_a = a->data, *data_b = b->data;
    const unsigned char *scales_a = a->scales, *scales_b = b->scales;
    #pragma omp parallel for schedule(static)
    for (int64_t tile = 0; tile < a->rows / 16; tile++) {
        __m512 sum_a = _mm512_setzero_ps(), sum_b = _mm512_setzero_ps();
        for (size_t base = 0; base < columns; base += 32) {
            const unsigned char *scale_offset_a = scales_a +
                ((size_t)tile * scale_stride + base / 32) * 16;
            const unsigned char *scale_offset_b = scales_b +
                ((size_t)tile * scale_stride + base / 32) * 16;
            __m512 scale_a = decode_scales_rows16(scale_offset_a, e8);
            __m512 scale_b = decode_scales_rows16(scale_offset_b, e8);
            for (size_t offset = 0; offset < 32; offset++) {
                size_t column = base + offset;
                size_t packed_offset =
                    ((size_t)tile * data_stride + column / 2) * 16;
                __m512 values_a = decode_fp4_rows16(
                    data_a + packed_offset, column & 1, fp4_table);
                __m512 values_b = decode_fp4_rows16(
                    data_b + packed_offset, column & 1, fp4_table);
                __m512 x = _mm512_set1_ps(activation[column]);
                sum_a = _mm512_add_ps(sum_a, _mm512_mul_ps(
                    _mm512_mul_ps(x, values_a), scale_a));
                sum_b = _mm512_add_ps(sum_b, _mm512_mul_ps(
                    _mm512_mul_ps(x, values_b), scale_b));
            }
        }
        _mm512_storeu_ps(output_a + (size_t)tile * 16, sum_a);
        _mm512_storeu_ps(output_b + (size_t)tile * 16, sum_b);
    }
    free(activation_scales); free(activation); return 0;
#elif defined(__AVX2__)
    if (!output_a || !output_b || !input || !packed_valid(a) ||
        !packed_valid(b) || a->rows != b->rows ||
        a->columns != b->columns) return -1;
    size_t columns = (size_t)a->columns;
    size_t data_stride = columns / 2, scale_stride = columns / 32;
    float *activation = malloc(columns * sizeof(*activation));
    uint8_t *activation_scales = malloc(columns / 128);
    if (!activation || !activation_scales) {
        free(activation_scales); free(activation); return -1;
    }
    if (coli_fp8_activation_qdq_ref(activation, activation_scales,
                                    input, columns, 128)) {
        free(activation_scales); free(activation); return -1;
    }
    Avx2Rows16Tables tables; avx2_rows16_tables(&tables);
    const unsigned char *data_a = a->data, *data_b = b->data;
    const unsigned char *scales_a = a->scales, *scales_b = b->scales;
    #pragma omp parallel for schedule(static)
    for (int64_t tile = 0; tile < a->rows / 16; tile++) {
        __m256 sum_a[2] = {_mm256_setzero_ps(), _mm256_setzero_ps()};
        __m256 sum_b[2] = {_mm256_setzero_ps(), _mm256_setzero_ps()};
        for (size_t base = 0; base < columns; base += 32) {
            size_t scale_offset =
                ((size_t)tile * scale_stride + base / 32) * 16;
            __m256 scale_a[2], scale_b[2];
            avx2_decode_scales(scale_a, scales_a + scale_offset, &tables);
            avx2_decode_scales(scale_b, scales_b + scale_offset, &tables);
            for (size_t offset = 0; offset < 32; offset++) {
                size_t column = base + offset;
                size_t packed_offset =
                    ((size_t)tile * data_stride + column / 2) * 16;
                __m256 values_a[2], values_b[2];
                avx2_decode_rows16(values_a, data_a + packed_offset,
                                   column & 1, &tables);
                avx2_decode_rows16(values_b, data_b + packed_offset,
                                   column & 1, &tables);
                __m256 x = _mm256_set1_ps(activation[column]);
                for (int half = 0; half < 2; half++) {
                    sum_a[half] = _mm256_add_ps(sum_a[half], _mm256_mul_ps(
                        _mm256_mul_ps(x, values_a[half]), scale_a[half]));
                    sum_b[half] = _mm256_add_ps(sum_b[half], _mm256_mul_ps(
                        _mm256_mul_ps(x, values_b[half]), scale_b[half]));
                }
            }
        }
        _mm256_storeu_ps(output_a + (size_t)tile * 16, sum_a[0]);
        _mm256_storeu_ps(output_a + (size_t)tile * 16 + 8, sum_a[1]);
        _mm256_storeu_ps(output_b + (size_t)tile * 16, sum_b[0]);
        _mm256_storeu_ps(output_b + (size_t)tile * 16 + 8, sum_b[1]);
    }
    free(activation_scales); free(activation); return 0;
#else /* __aarch64__ */
    if (!output_a || !output_b || !input || !packed_valid(a) ||
        !packed_valid(b) || a->rows != b->rows ||
        a->columns != b->columns) return -1;
    size_t columns = (size_t)a->columns;
    size_t data_stride = columns / 2, scale_stride = columns / 32;
    float *activation = malloc(columns * sizeof(*activation));
    uint8_t *activation_scales = malloc(columns / 128);
    if (!activation || !activation_scales) {
        free(activation_scales); free(activation); return -1;
    }
    if (coli_fp8_activation_qdq_ref(activation, activation_scales,
                                    input, columns, 128)) {
        free(activation_scales); free(activation); return -1;
    }
    NeonRows16Tables tables;
    neon_rows16_tables(&tables);
    const unsigned char *data_a = a->data, *data_b = b->data;
    const unsigned char *scales_a = a->scales, *scales_b = b->scales;
    #pragma omp parallel for schedule(static)
    for (int64_t tile = 0; tile < a->rows / 16; tile++) {
        float32x4_t sums_a[4] = {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
                                 vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};
        float32x4_t sums_b[4] = {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
                                 vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};
        for (size_t base = 0; base < columns; base += 32) {
            size_t scale_offset =
                ((size_t)tile * scale_stride + base / 32) * 16;
            float32x4_t block_scales_a[4], block_scales_b[4];
            neon_rows16_block_scales(block_scales_a, scales_a + scale_offset,
                                     &tables);
            neon_rows16_block_scales(block_scales_b, scales_b + scale_offset,
                                     &tables);
            for (size_t offset = 0; offset < 32; offset += 2) {
                size_t packed_offset =
                    ((size_t)tile * data_stride + (base + offset) / 2) * 16;
                uint8x16_t bytes_a = vld1q_u8(data_a + packed_offset);
                uint8x16_t bytes_b = vld1q_u8(data_b + packed_offset);
                neon_rows16_accumulate(
                    sums_a, vandq_u8(bytes_a, vdupq_n_u8(15)),
                    activation[base + offset], block_scales_a, &tables);
                neon_rows16_accumulate(
                    sums_b, vandq_u8(bytes_b, vdupq_n_u8(15)),
                    activation[base + offset], block_scales_b, &tables);
                neon_rows16_accumulate(
                    sums_a, vshrq_n_u8(bytes_a, 4),
                    activation[base + offset + 1], block_scales_a, &tables);
                neon_rows16_accumulate(
                    sums_b, vshrq_n_u8(bytes_b, 4),
                    activation[base + offset + 1], block_scales_b, &tables);
            }
        }
        for (int group = 0; group < 4; group++) {
            vst1q_f32(output_a + (size_t)tile * 16 + 4 * group,
                      sums_a[group]);
            vst1q_f32(output_b + (size_t)tile * 16 + 4 * group,
                      sums_b[group]);
        }
    }
    free(activation_scales); free(activation); return 0;
#endif
}
#endif /* COLI_V4_UNIT_NATIVE_QUANT_ROWS16 */
