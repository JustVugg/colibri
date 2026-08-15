/* ================= Colibri DSpark-SSD: complete native drafter =============
 *
 * This is an original C implementation of the architecture described by the
 * DeepSeek-V4 checkpoint's own inference/model.py.  It intentionally does not
 * depend on another inference runtime.  The invariants that matter here are:
 *
 *   - all three mtp.* transformer stages are evaluated;
 *   - the five draft positions attend bidirectionally inside their block;
 *   - mtp.2's HC head, Markov logit bias and confidence head are used;
 *   - routed MTP experts live in a separate bounded LRU, so drafting cannot
 *     evict the target model's learned expert cache;
 *   - candidates are never committed without exact target verification.
 *
 * DSpark is opt-in while the real-checkpoint A/B gate is being calibrated:
 * V4_DRAFT=5 V4_MTP=1 V4_MTP_DRAFT=3.  V4_MTP_GB is a TOTAL cache budget
 * across the three stages, and V4_MTP_MISS caps cold expert records loaded by
 * one draft round.  Prompt lookup may still use all five draft slots; the
 * trained MTP path defaults to three to avoid rejected-suffix replay.
 * ========================================================================== */

#define V4_DSPARK_STAGES 3
#define V4_DSPARK_BLOCK  5
#define V4_DSPARK_WIN    128
#define V4_DSPARK_RANK   256

typedef struct {
    int id;
    uint8_t *slab;
    uint64_t used;
} V4DSparkSlot;

typedef struct {
    int ready;
    int stage;

    /* FP8 weights stay E4M3.  Expand the tiny UE8M0 sidecars once at load:
     * the current native kernel deliberately has one branch-free F32-scale
     * hot path (the target model uses the same representation). */
    uint8_t *wq_a_w, *wq_b_w, *wkv_w, *wo_a_w, *wo_b_w;
    uint8_t *sh1_w, *sh2_w, *sh3_w;
    float *wq_a_s, *wq_b_s, *wkv_s, *wo_a_s, *wo_b_s;
    float *sh1_s, *sh2_s, *sh3_s;

    ColiFloatTensor q_norm, kv_norm, attn_norm, ffn_norm, attn_sink;
    ColiFloatTensor hc_attn_fn, hc_attn_base, hc_attn_scale;
    ColiFloatTensor hc_ffn_fn,  hc_ffn_base,  hc_ffn_scale;
    ColiFloatTensor gate_w, gate_b;

    /* Context injection KV is independent for every MTP stage. */
    float *kv;                         /* [window][head_dim] */
    int64_t kv_abs[V4_DSPARK_WIN];

    /* FP4 expert cache.  One slab is scales+payload for w1,w2,w3. */
    V4DSparkSlot *slots;
    int nslots, cap_slots;
    int64_t slab_bytes;
    uint64_t clock;
    const ColiSafetensorsTensor *ew[256][3], *es[256][3];
    int64_t hits, misses, aborted;
} V4DSparkStage;

static V4DSparkStage g_v4ds[V4_DSPARK_STAGES];

static struct {
    int ready;
    uint8_t *main_proj_w;
    float *main_proj_s;
    int64_t main_proj_rows, main_proj_cols;
    ColiFloatTensor main_norm;

    ColiFloatTensor head_fn, head_base, head_scale, norm;
    ColiFloatTensor confidence_proj;
    uint16_t *markov_w1, *markov_w2;   /* [vocab][256], BF16 */
    size_t markov_w1_bytes, markov_w2_bytes;

    float *main_x;                     /* projected context ring */
    int64_t main_x_abs[V4_DSPARK_WIN];

    uint64_t rounds, drafted, accepted;
    double accept_ewma[V4_DSPARK_BLOCK];
} g_v4ds_core;

static void v4_ds_bf16(float *values, size_t count) {
    coli_bf16_round_array(values, count);
}

static void v4_ds_rmsnorm(float *out, const float *in, const float *weight,
                          int n, float eps) {
    float square = 0.0f;
    for (int i = 0; i < n; i++) square += in[i] * in[i];
    float scale = 1.0f / sqrtf(square / n + eps);
    for (int i = 0; i < n; i++) out[i] = in[i] * scale * weight[i];
    v4_ds_bf16(out, (size_t)n);
}

static int v4_ds_read_raw(const ColiSafetensorsIndex *index, const char *name,
                          int dtype, void **output, size_t *bytes) {
    const ColiSafetensorsTensor *tensor = coli_st_find(index, name);
    if (!tensor || tensor->dtype != dtype || !tensor->nbytes) return -1;
    void *data = malloc((size_t)tensor->nbytes);
    if (!data || coli_st_read_tensor(index, tensor, data)) {
        free(data);
        return -1;
    }
    *output = data;
    if (bytes) *bytes = (size_t)tensor->nbytes;
    return 0;
}

/* DSpark's resident FP8 matrices use the same column-major rows8 tile consumed
 * by the AVX2 target kernel.  Packing is local to this translation unit because
 * the split build keeps the target layer loader in a different object. */
static int v4_ds_pack_rows8(uint8_t *data, int64_t rows, int64_t columns) {
#ifndef __AVX2__
    (void)data; (void)rows; (void)columns;
    return 0;
#else
    if (!data || rows < 8 || rows % 8 || columns < 128 || columns % 128)
        return -1;
    size_t tile_bytes = (size_t)8 * (size_t)columns;
    uint8_t *scratch = malloc(tile_bytes);
    if (!scratch) return -1;
    for (int64_t tile = 0; tile < rows / 8; tile++) {
        uint8_t *target = data + (size_t)tile * tile_bytes;
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

static int v4_ds_load_fp8(const ColiSafetensorsIndex *index,
                          const char *weight_name, const char *scale_name,
                          uint8_t **weight, float **scale) {
    const ColiSafetensorsTensor *weight_tensor = coli_st_find(index, weight_name);
    const ColiSafetensorsTensor *scale_tensor = coli_st_find(index, scale_name);
    size_t wb = 0;
    if (!weight_tensor || weight_tensor->dtype != COLI_ST_F8_E4M3 ||
        weight_tensor->rank != 2 ||
        weight_tensor->nbytes != weight_tensor->shape[0] *
                                  weight_tensor->shape[1] ||
        !scale_tensor || scale_tensor->rank != 2 ||
        scale_tensor->dtype != COLI_ST_F8_E8M0 ||
        scale_tensor->shape[0] != (weight_tensor->shape[0] + 127) / 128 ||
        scale_tensor->shape[1] != (weight_tensor->shape[1] + 127) / 128 ||
        scale_tensor->numel < 1)
        return -1;
    if (v4_ds_read_raw(index, weight_name, COLI_ST_F8_E4M3,
                       (void **)weight, &wb))
        return -1;
    *scale = malloc((size_t)scale_tensor->numel * sizeof(**scale));
    if (!*scale ||
        st_read_scale_f32((ColiSafetensorsIndex *)index, scale_name, *scale,
                          scale_tensor->numel, 0) != scale_tensor->numel) {
        free(*weight);
        free(*scale);
        *weight = NULL;
        *scale = NULL;
        return -1;
    }
    if (v4_ds_pack_rows8(*weight, weight_tensor->shape[0],
                         weight_tensor->shape[1]) < 0) {
        free(*weight);
        free(*scale);
        *weight = NULL;
        *scale = NULL;
        return -1;
    }
    return wb ? 0 : -1;
}

static int v4_ds_name(char *out, size_t size, int stage, const char *suffix) {
    int n = snprintf(out, size, "mtp.%d.%s", stage, suffix);
    return n < 0 || (size_t)n >= size ? -1 : 0;
}

static int v4_ds_load_fp8_suffix(const ColiSafetensorsIndex *index, int stage,
                                 const char *stem,
                                 uint8_t **weight, float **scale) {
    char wn[160], sn[160];
    if (snprintf(wn, sizeof(wn), "mtp.%d.%s.weight", stage, stem) >= (int)sizeof(wn) ||
        snprintf(sn, sizeof(sn), "mtp.%d.%s.scale", stage, stem) >= (int)sizeof(sn))
        return -1;
    return v4_ds_load_fp8(index, wn, sn, weight, scale);
}

/* SEC (GHSA-9gjf): every F32 sidecar loaded here is later indexed linearly by
 * config-derived dimensions, but coli_tensor_load_f32 does no shape accounting —
 * an undersized snapshot tensor became a heap OOB read. `expected` is the numel
 * the consumer will index; require at least that many (-1 = skip). `<` (not `!=`)
 * so a legitimately padded checkpoint is never false-rejected. */
static int v4_ds_load_f32_suffix(ColiFloatTensor *output,
                                 const ColiSafetensorsIndex *index, int stage,
                                 const char *suffix, int64_t expected) {
    char name[160], error[256];
    if (v4_ds_name(name, sizeof(name), stage, suffix)) return -1;
    if (coli_tensor_load_f32(output, index, name, error, sizeof(error))) return -1;
    if (expected >= 0 && (int64_t)output->count < expected) {
        fprintf(stderr, "[MTP] shape mismatch: %s (numel %llu, need >= %lld)\n",
                name, (unsigned long long)output->count, (long long)expected);
        coli_float_tensor_free(output);
        return -1;
    }
    return 0;
}

static void v4_ds_fp8_view(ColiTensorView *view,
                           const uint8_t *weight, const float *scale,
                           int64_t rows, int64_t columns) {
    memset(view, 0, sizeof(*view));
    view->format = COLI_TENSOR_FP8_E4M3_BLOCK;
    view->scale_format = COLI_SCALE_F32;
    view->data = weight;
    view->scales = scale;
    view->data_bytes = (size_t)(rows * columns);
    view->scale_bytes = (size_t)(((rows + 127) / 128) *
                                ((columns + 127) / 128)) * sizeof(float);
    view->rows = rows;
    view->columns = columns;
#ifdef __AVX2__
    view->block_rows = 8;
#else
    view->block_rows = 128;
#endif
    view->block_columns = 128;
}

/* Official FP8 linears quantize activations to E4M3/UE8M0 first.  Reusing the
 * engine's native reference kernel is both more faithful and considerably
 * faster than the scalar, unquantized prototype. */
static int v4_ds_mm(float *out, const uint8_t *weight, const float *scale,
                    int64_t rows, int64_t columns, const float *input) {
    ColiTensorView view;
    v4_ds_fp8_view(&view, weight, scale, rows, columns);
    if (coli_fp8_matvec_ref(out, &view, input)) return -1;
    v4_ds_bf16(out, (size_t)rows);
    return 0;
}

static int v4_ds_core_load(const ColiSafetensorsIndex *index,
                           const ColiDeepSeekV4Config *config) {
    if (g_v4ds_core.ready) return g_v4ds_core.ready > 0;
    g_v4ds_core.ready = -1;
    if (config->hc_mult < 1 || config->hc_mult > 16 ||
        config->n_routed_experts < 1 || config->n_routed_experts > 256 ||
        config->num_experts_per_tok < 1 ||
        config->num_experts_per_tok > 16)
        return 0;

    const ColiSafetensorsTensor *mp =
        coli_st_find(index, "mtp.0.main_proj.weight");
    const ColiSafetensorsTensor *ms =
        coli_st_find(index, "mtp.0.main_proj.scale");
    if (!mp || !ms || mp->dtype != COLI_ST_F8_E4M3 ||
        ms->dtype != COLI_ST_F8_E8M0 || mp->rank != 2)
        return 0;
    g_v4ds_core.main_proj_rows = config->hidden_size;
    g_v4ds_core.main_proj_cols =
        (int64_t)config->hidden_size * V4_MAINH_TARGETS;
    if (mp->shape[0] != g_v4ds_core.main_proj_rows ||
        mp->shape[1] != g_v4ds_core.main_proj_cols ||
        v4_ds_load_fp8(index, "mtp.0.main_proj.weight",
                       "mtp.0.main_proj.scale",
                       &g_v4ds_core.main_proj_w,
                       &g_v4ds_core.main_proj_s) ||
        v4_ds_load_f32_suffix(&g_v4ds_core.main_norm, index, 0,
                              "main_norm.weight", config->hidden_size) ||
        v4_ds_load_f32_suffix(&g_v4ds_core.head_fn, index, 2,
                              "hc_head_fn",
                              (int64_t)config->hc_mult * config->hc_mult *
                              config->hidden_size) ||
        v4_ds_load_f32_suffix(&g_v4ds_core.head_base, index, 2,
                              "hc_head_base", config->hc_mult) ||
        v4_ds_load_f32_suffix(&g_v4ds_core.head_scale, index, 2,
                              "hc_head_scale", 1) ||
        v4_ds_load_f32_suffix(&g_v4ds_core.norm, index, 2,
                              "norm.weight", config->hidden_size) ||
        v4_ds_load_f32_suffix(&g_v4ds_core.confidence_proj, index, 2,
                              "confidence_head.proj.weight",
                              (int64_t)config->hidden_size + V4_DSPARK_RANK) ||
        v4_ds_read_raw(index, "mtp.2.markov_head.markov_w1.weight",
                       COLI_ST_BF16, (void **)&g_v4ds_core.markov_w1,
                       &g_v4ds_core.markov_w1_bytes) ||
        v4_ds_read_raw(index, "mtp.2.markov_head.markov_w2.weight",
                       COLI_ST_BF16, (void **)&g_v4ds_core.markov_w2,
                       &g_v4ds_core.markov_w2_bytes))
        return 0;

    size_t markov_bytes = (size_t)config->vocab_size * V4_DSPARK_RANK *
                          sizeof(uint16_t);
    if (g_v4ds_core.markov_w1_bytes != markov_bytes ||
        g_v4ds_core.markov_w2_bytes != markov_bytes)
        return 0;

    g_v4ds_core.main_x = calloc((size_t)V4_DSPARK_WIN * config->hidden_size,
                                sizeof(float));
    if (!g_v4ds_core.main_x) return 0;
    for (int i = 0; i < V4_DSPARK_WIN; i++)
        g_v4ds_core.main_x_abs[i] = -1;
    for (int i = 0; i < V4_DSPARK_BLOCK; i++)
        g_v4ds_core.accept_ewma[i] = 0.5;

    g_v4ds_core.ready = 1;
    fprintf(stderr,
            "[MTP] DSpark core ready: stages=%d block=%d main_proj=%lldx%lld "
            "markov_rank=%d\n",
            V4_DSPARK_STAGES, V4_DSPARK_BLOCK,
            (long long)g_v4ds_core.main_proj_rows,
            (long long)g_v4ds_core.main_proj_cols, V4_DSPARK_RANK);
    return 1;
}

static int v4_ds_stage_load(const ColiSafetensorsIndex *index,
                            const ColiDeepSeekV4Config *config, int stage_id) {
    V4DSparkStage *stage = &g_v4ds[stage_id];
    if (stage->ready) return stage->ready > 0;
    stage->ready = -1;
    stage->stage = stage_id;

#define V4_DS_LOAD_FP8(field, stem) \
    do { if (v4_ds_load_fp8_suffix(index, stage_id, stem, \
                                   &stage->field##_w, &stage->field##_s)) \
             return 0; } while (0)
    V4_DS_LOAD_FP8(wq_a, "attn.wq_a");
    V4_DS_LOAD_FP8(wq_b, "attn.wq_b");
    V4_DS_LOAD_FP8(wkv,  "attn.wkv");
    V4_DS_LOAD_FP8(wo_a, "attn.wo_a");
    V4_DS_LOAD_FP8(wo_b, "attn.wo_b");
    V4_DS_LOAD_FP8(sh1,  "ffn.shared_experts.w1");
    V4_DS_LOAD_FP8(sh2,  "ffn.shared_experts.w2");
    V4_DS_LOAD_FP8(sh3,  "ffn.shared_experts.w3");
#undef V4_DS_LOAD_FP8

#define V4_DS_LOAD_F32(field, suffix, expected) \
    do { if (v4_ds_load_f32_suffix(&stage->field, index, stage_id, suffix, \
                                   expected)) \
             return 0; } while (0)
    {   int64_t mixes = (int64_t)(2 + config->hc_mult) * config->hc_mult;
        int64_t flat = (int64_t)config->hc_mult * config->hidden_size;
        V4_DS_LOAD_F32(q_norm,       "attn.q_norm.weight", config->q_lora_rank);
        V4_DS_LOAD_F32(kv_norm,      "attn.kv_norm.weight", config->head_dim);
        V4_DS_LOAD_F32(attn_norm,    "attn_norm.weight", config->hidden_size);
        V4_DS_LOAD_F32(ffn_norm,     "ffn_norm.weight", config->hidden_size);
        V4_DS_LOAD_F32(attn_sink,    "attn.attn_sink", config->num_attention_heads);
        V4_DS_LOAD_F32(hc_attn_fn,   "hc_attn_fn", mixes * flat);
        V4_DS_LOAD_F32(hc_attn_base, "hc_attn_base", mixes);
        V4_DS_LOAD_F32(hc_attn_scale,"hc_attn_scale", 3);
        V4_DS_LOAD_F32(hc_ffn_fn,    "hc_ffn_fn", mixes * flat);
        V4_DS_LOAD_F32(hc_ffn_base,  "hc_ffn_base", mixes);
        V4_DS_LOAD_F32(hc_ffn_scale, "hc_ffn_scale", 3);
        V4_DS_LOAD_F32(gate_w,       "ffn.gate.weight",
                       (int64_t)config->n_routed_experts * config->hidden_size);
        V4_DS_LOAD_F32(gate_b,       "ffn.gate.bias", config->n_routed_experts);
    }
#undef V4_DS_LOAD_F32

    static const char *matrices[3] = {"w1", "w2", "w3"};
    /* SEC (GHSA-8p69): slab_bytes must cover the LARGEST expert, not expert 0.
     * Each slot later copies whichever expert is routed to it, using that
     * expert's own declared nbytes — so sizing from expert 0 alone let a crafted
     * checkpoint (expert 0 tiny, another expert large) overflow the slab with
     * file-controlled bytes (heap OOB write). Real MTP checkpoints have uniform
     * expert shapes, so max == expert 0 and behaviour is unchanged. */
    stage->slab_bytes = 0;
    for (int expert = 0; expert < config->n_routed_experts; expert++) {
        int64_t expert_bytes = 0;
        for (int matrix = 0; matrix < 3; matrix++) {
            char name[192];
            snprintf(name, sizeof(name), "mtp.%d.ffn.experts.%d.%s.weight",
                     stage_id, expert, matrices[matrix]);
            stage->ew[expert][matrix] = coli_st_find(index, name);
            snprintf(name, sizeof(name), "mtp.%d.ffn.experts.%d.%s.scale",
                     stage_id, expert, matrices[matrix]);
            stage->es[expert][matrix] = coli_st_find(index, name);
            if (!stage->ew[expert][matrix] || !stage->es[expert][matrix])
                return 0;
            expert_bytes += stage->ew[expert][matrix]->nbytes +
                            stage->es[expert][matrix]->nbytes;
        }
        if (expert_bytes > stage->slab_bytes) stage->slab_bytes = expert_bytes;
    }
    if (stage->slab_bytes < 1) return 0;

    /* V4_MTP_GB is intentionally total, not per stage. */
    double total_gb = coli_v4_dspark_cache_gb();
    double stage_bytes = total_gb * 1e9 / V4_DSPARK_STAGES;
    stage->cap_slots = (int)(stage_bytes / (double)stage->slab_bytes);
    if (stage->cap_slots < 4) stage->cap_slots = 4;
    if (stage->cap_slots > config->n_routed_experts)
        stage->cap_slots = config->n_routed_experts;
    stage->slots = calloc((size_t)stage->cap_slots, sizeof(*stage->slots));
    stage->kv = calloc((size_t)V4_DSPARK_WIN * config->head_dim, sizeof(float));
    if (!stage->slots || !stage->kv) return 0;
    for (int i = 0; i < stage->cap_slots; i++) stage->slots[i].id = -1;
    for (int i = 0; i < V4_DSPARK_WIN; i++) stage->kv_abs[i] = -1;

    stage->ready = 1;
    fprintf(stderr,
            "[MTP] stage %d ready: expert_cache=%d/%d %.2fGB\n",
            stage_id, stage->cap_slots, config->n_routed_experts,
            stage->cap_slots * (double)stage->slab_bytes / 1e9);
    return 1;
}

static int v4_ds_load_all(const ColiSafetensorsIndex *index,
                          const ColiDeepSeekV4Config *config) {
    if (!v4_ds_core_load(index, config)) return 0;
    for (int stage = 0; stage < V4_DSPARK_STAGES; stage++)
        if (!v4_ds_stage_load(index, config, stage)) return 0;
    return 1;
}

static int v4_ds_expert(const ColiSafetensorsIndex *index,
                        V4DSparkStage *stage, int expert,
                        ColiExpertView *view, int *miss_budget) {
    int slot = -1;
    int lru = 0;
    for (int i = 0; i < stage->nslots; i++) {
        if (stage->slots[i].id == expert) {
            slot = i;
            stage->hits++;
            break;
        }
        if (stage->slots[i].used < stage->slots[lru].used) lru = i;
    }
    if (slot < 0) {
        if (*miss_budget <= 0) {
            stage->aborted++;
            return 0;
        }
        (*miss_budget)--;
        stage->misses++;
        slot = stage->nslots < stage->cap_slots ? stage->nslots++ : lru;
        if (!stage->slots[slot].slab) {
            stage->slots[slot].slab = malloc((size_t)stage->slab_bytes);
            if (!stage->slots[slot].slab) return 0;
        }
        stage->slots[slot].id = -1;
        uint8_t *cursor = stage->slots[slot].slab;
        uint8_t *slab_end = cursor + stage->slab_bytes;
        for (int matrix = 0; matrix < 3; matrix++) {
            const ColiSafetensorsTensor *scale = stage->es[expert][matrix];
            const ColiSafetensorsTensor *weight = stage->ew[expert][matrix];
            /* SEC (GHSA-8p69): defence in depth — never write past the slab even
             * if the max-sizing above is ever defeated (shape drift within a
             * stage). Fail closed rather than overflow. */
            if (scale->nbytes < 0 || weight->nbytes < 0 ||
                (size_t)(slab_end - cursor) < (size_t)scale->nbytes)
                return 0;
            if (coli_st_read_tensor(index, scale, cursor))
                return 0;
            cursor += scale->nbytes;
            if ((size_t)(slab_end - cursor) < (size_t)weight->nbytes)
                return 0;
            int shard = coli_st_tensor_shard(index, weight);
            if (shard < 0 || weight->off < 0 ||
                coli_st_read_at_streaming(
                    index, shard, (uint64_t)weight->off,
                    (size_t)weight->nbytes, cursor))
                return 0;
            cursor += weight->nbytes;
        }
        stage->slots[slot].id = expert;
    }
    stage->slots[slot].used = ++stage->clock;

    uint8_t *cursor = stage->slots[slot].slab;
    ColiTensorView *views[3] = {&view->gate, &view->down, &view->up};
    for (int matrix = 0; matrix < 3; matrix++) {
        const ColiSafetensorsTensor *scale = stage->es[expert][matrix];
        const ColiSafetensorsTensor *weight = stage->ew[expert][matrix];
        ColiTensorView *tensor = views[matrix];
        memset(tensor, 0, sizeof(*tensor));
        tensor->format = COLI_TENSOR_FP4_NATIVE_BLOCK;
        tensor->scale_format = COLI_SCALE_UE8M0;
        tensor->scales = cursor;
        cursor += scale->nbytes;
        tensor->data = cursor;
        cursor += weight->nbytes;
        tensor->scale_bytes = (size_t)scale->nbytes;
        tensor->data_bytes = (size_t)weight->nbytes;
        tensor->rows = weight->shape[0];
        tensor->columns = weight->shape[1] * 2;
        tensor->block_rows = 1;
        tensor->block_columns = 32;
    }
    view->lease = NULL;
    return 1;
}

/* Hyper-Connections: exact ordering of the checkpoint's split-Sinkhorn. */
static void v4_ds_sinkhorn(const float *mix, const float *scale,
                           const float *base, int hc, int iterations,
                           float eps, float *pre, float *post, float *comb) {
    for (int j = 0; j < hc; j++) {
        float z = mix[j] * scale[0] + base[j];
        pre[j] = 1.0f / (1.0f + expf(-z)) + eps;
        z = mix[j + hc] * scale[1] + base[j + hc];
        post[j] = 2.0f / (1.0f + expf(-z));
    }
    for (int j = 0; j < hc; j++) {
        float maximum = -FLT_MAX;
        for (int k = 0; k < hc; k++) {
            float value = mix[2 * hc + j * hc + k] * scale[2] +
                          base[2 * hc + j * hc + k];
            comb[j * hc + k] = value;
            if (value > maximum) maximum = value;
        }
        float sum = 0.0f;
        for (int k = 0; k < hc; k++) {
            comb[j * hc + k] = expf(comb[j * hc + k] - maximum);
            sum += comb[j * hc + k];
        }
        for (int k = 0; k < hc; k++)
            comb[j * hc + k] = comb[j * hc + k] / sum + eps;
    }

    /* Initial column normalisation, then (row,column) x (iters-1). */
    for (int k = 0; k < hc; k++) {
        float sum = 0.0f;
        for (int j = 0; j < hc; j++) sum += comb[j * hc + k];
        for (int j = 0; j < hc; j++) comb[j * hc + k] /= sum + eps;
    }
    for (int iteration = 0; iteration < iterations - 1; iteration++) {
        for (int j = 0; j < hc; j++) {
            float sum = 0.0f;
            for (int k = 0; k < hc; k++) sum += comb[j * hc + k];
            for (int k = 0; k < hc; k++) comb[j * hc + k] /= sum + eps;
        }
        for (int k = 0; k < hc; k++) {
            float sum = 0.0f;
            for (int j = 0; j < hc; j++) sum += comb[j * hc + k];
            for (int j = 0; j < hc; j++) comb[j * hc + k] /= sum + eps;
        }
    }
}

static void v4_ds_hc_pre(float *out, const float *state,
                         const ColiFloatTensor *function,
                         const ColiFloatTensor *base,
                         const ColiFloatTensor *scale,
                         const ColiDeepSeekV4Config *config,
                         float *post, float *comb) {
    int hc = config->hc_mult;
    int hidden = config->hidden_size;
    int flat = hc * hidden;
    int mixes = (2 + hc) * hc;
    float square = 0.0f;
    for (int i = 0; i < flat; i++) square += state[i] * state[i];
    float inverse_rms =
        1.0f / sqrtf(square / flat + config->rms_norm_eps);
    float mixed[64], pre[16];
    for (int row = 0; row < mixes; row++) {
        float sum = 0.0f;
        const float *weight = function->data + (size_t)row * flat;
        for (int i = 0; i < flat; i++) sum += weight[i] * state[i];
        mixed[row] = sum * inverse_rms;
    }
    v4_ds_sinkhorn(mixed, scale->data, base->data, hc,
                   config->hc_sinkhorn_iters, config->hc_eps,
                   pre, post, comb);
    for (int i = 0; i < hidden; i++) {
        float sum = 0.0f;
        for (int copy = 0; copy < hc; copy++)
            sum += pre[copy] * state[(size_t)copy * hidden + i];
        out[i] = coli_bf16_round(sum);
    }
}

static void v4_ds_hc_post(float *out, const float *branch,
                          const float *residual, const float *post,
                          const float *comb,
                          const ColiDeepSeekV4Config *config) {
    int hc = config->hc_mult;
    int hidden = config->hidden_size;
    for (int dst = 0; dst < hc; dst++) {
        for (int i = 0; i < hidden; i++) {
            float sum = post[dst] * branch[i];
            for (int src = 0; src < hc; src++)
                sum += comb[src * hc + dst] *
                       residual[(size_t)src * hidden + i];
            out[(size_t)dst * hidden + i] = coli_bf16_round(sum);
        }
    }
}

static int v4_ds_main_x(float *out, int64_t position,
                        const ColiDeepSeekV4Config *config) {
    if (position < 0 || !g_v4ds_core.main_x) return 0;
    int slot = (int)(position % V4_DSPARK_WIN);
    float *cached = g_v4ds_core.main_x +
                    (size_t)slot * config->hidden_size;
    if (g_v4ds_core.main_x_abs[slot] == position) {
        memcpy(out, cached, (size_t)config->hidden_size * sizeof(float));
        return 1;
    }
    const float *main_hidden = v4_mainh_get(position);
    if (!main_hidden) return 0;
    if (v4_ds_mm(cached, g_v4ds_core.main_proj_w,
                 g_v4ds_core.main_proj_s,
                 g_v4ds_core.main_proj_rows,
                 g_v4ds_core.main_proj_cols, main_hidden))
        return 0;
    v4_ds_rmsnorm(cached, cached, g_v4ds_core.main_norm.data,
                  config->hidden_size, config->rms_norm_eps);
    g_v4ds_core.main_x_abs[slot] = position;
    memcpy(out, cached, (size_t)config->hidden_size * sizeof(float));
    return 1;
}

/* Speculative target batches can write hidden taps for a rejected suffix.
 * Purge them before refeeding the accepted prefix; otherwise future drafts
 * would condition on a history that never happened. */
static void v4_ds_invalidate_from(int64_t position) {
    for (int slot = 0; slot < V4_MAINH_RING; slot++) {
        if (g_mainh_abs[slot] >= position) g_mainh_abs[slot] = -1;
    }
    for (int slot = 0; slot < V4_DSPARK_WIN; slot++) {
        if (g_v4ds_core.main_x_abs[slot] >= position)
            g_v4ds_core.main_x_abs[slot] = -1;
        for (int stage = 0; stage < V4_DSPARK_STAGES; stage++)
            if (g_v4ds[stage].kv_abs[slot] >= position)
                g_v4ds[stage].kv_abs[slot] = -1;
    }
}

static void v4_ds_reset_history(void) {
    v4_ds_invalidate_from(0);
}

/* DSpark QAT keeps the RoPE tail in BF16 and simulates FP8 only for the
 * non-positional 448 dimensions, in blocks of 64. */
static int v4_ds_kv_qdq(float *kv, int head_dim, int rope_dim) {
    int nope = head_dim - rope_dim;
    uint8_t scales[16];
    if (nope < 1 || nope > 1024 ||
        coli_fp8_activation_qdq_ref(kv, scales, kv, (size_t)nope, 64))
        return -1;
    v4_ds_bf16(kv, (size_t)head_dim);
    return 0;
}

static int v4_ds_backfill_context(const ColiDeepSeekV4Config *config,
                                  int64_t position,
                                  const float *cosines, const float *sines) {
    int hidden = config->hidden_size;
    int head_dim = config->head_dim;
    int rope_dim = config->qk_rope_head_dim;
    int pairs = rope_dim / 2;
    float *main_x = malloc((size_t)hidden * sizeof(float));
    float *kv = malloc((size_t)head_dim * sizeof(float));
    if (!main_x || !kv) {
        free(kv); free(main_x); return -1;
    }
    int64_t first = position > V4_DSPARK_WIN
        ? position - V4_DSPARK_WIN : 0;
    for (int64_t absolute = first; absolute < position; absolute++) {
        int slot = (int)(absolute % V4_DSPARK_WIN);
        int needed = 0;
        for (int stage = 0; stage < V4_DSPARK_STAGES; stage++)
            if (g_v4ds[stage].kv_abs[slot] != absolute) needed = 1;
        if (!needed || !v4_ds_main_x(main_x, absolute, config)) continue;
        for (int stage_id = 0; stage_id < V4_DSPARK_STAGES; stage_id++) {
            V4DSparkStage *stage = &g_v4ds[stage_id];
            if (stage->kv_abs[slot] == absolute) continue;
            if (v4_ds_mm(kv, stage->wkv_w, stage->wkv_s,
                         head_dim, hidden, main_x)) {
                free(kv); free(main_x); return -1;
            }
            v4_ds_rmsnorm(kv, kv, stage->kv_norm.data, head_dim,
                          config->rms_norm_eps);
            coli_v4_rope_apply(kv + head_dim - rope_dim, 1, rope_dim,
                               cosines + (size_t)absolute * pairs,
                               sines + (size_t)absolute * pairs, 0);
            v4_ds_bf16(kv + head_dim - rope_dim, (size_t)rope_dim);
            if (v4_ds_kv_qdq(kv, head_dim, rope_dim)) {
                free(kv); free(main_x); return -1;
            }
            memcpy(stage->kv + (size_t)slot * head_dim, kv,
                   (size_t)head_dim * sizeof(float));
            stage->kv_abs[slot] = absolute;
        }
    }
    free(kv);
    free(main_x);
    return 0;
}

static int v4_ds_attention(float *out, V4DSparkStage *stage,
                           const float *input, int64_t query_position,
                           const float *block_kv, int block_count,
                           const ColiDeepSeekV4Config *config,
                           const float *cosines, const float *sines) {
    int hidden = config->hidden_size;
    int head_dim = config->head_dim;
    int rope_dim = config->qk_rope_head_dim;
    int heads = config->num_attention_heads;
    int groups = config->o_groups;
    int q_rank = config->q_lora_rank;
    int o_rank = config->o_lora_rank;
    int pairs = rope_dim / 2;

    float *qa = malloc((size_t)q_rank * sizeof(float));
    float *query = malloc((size_t)heads * head_dim * sizeof(float));
    float *attended = calloc((size_t)heads * head_dim, sizeof(float));
    float *oa = malloc((size_t)groups * o_rank * sizeof(float));
    if (!qa || !query || !attended || !oa) {
        free(oa); free(attended); free(query); free(qa); return -1;
    }
    if (v4_ds_mm(qa, stage->wq_a_w, stage->wq_a_s,
                 q_rank, hidden, input))
        goto fail;
    v4_ds_rmsnorm(qa, qa, stage->q_norm.data, q_rank,
                  config->rms_norm_eps);
    if (v4_ds_mm(query, stage->wq_b_w, stage->wq_b_s,
                 (int64_t)heads * head_dim, q_rank, qa))
        goto fail;
    for (int head = 0; head < heads; head++) {
        float *row = query + (size_t)head * head_dim;
        float square = 0.0f;
        for (int d = 0; d < head_dim; d++) square += row[d] * row[d];
        float inverse = 1.0f /
            sqrtf(square / head_dim + config->rms_norm_eps);
        for (int d = 0; d < head_dim; d++)
            row[d] = coli_bf16_round(row[d] * inverse);
        coli_v4_rope_apply(row + head_dim - rope_dim, 1, rope_dim,
                           cosines + (size_t)query_position * pairs,
                           sines + (size_t)query_position * pairs, 0);
        v4_ds_bf16(row + head_dim - rope_dim, (size_t)rope_dim);
    }

    float softmax_scale = 1.0f / sqrtf((float)head_dim);
    for (int head = 0; head < heads; head++) {
        const float *q = query + (size_t)head * head_dim;
        const float *keys[V4_DSPARK_WIN + V4_DSPARK_BLOCK];
        float scores[V4_DSPARK_WIN + V4_DSPARK_BLOCK];
        int count = 0;
        float maximum = stage->attn_sink.data[head];
        for (int slot = 0; slot < V4_DSPARK_WIN; slot++) {
            if (stage->kv_abs[slot] < 0 ||
                stage->kv_abs[slot] >= query_position)
                continue;
            keys[count] = stage->kv + (size_t)slot * head_dim;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) dot += q[d] * keys[count][d];
            scores[count] = dot * softmax_scale;
            if (scores[count] > maximum) maximum = scores[count];
            count++;
        }
        /* The parallel backbone is deliberately bidirectional in-block. */
        for (int item = 0; item < block_count; item++) {
            keys[count] = block_kv + (size_t)item * head_dim;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) dot += q[d] * keys[count][d];
            scores[count] = dot * softmax_scale;
            if (scores[count] > maximum) maximum = scores[count];
            count++;
        }
        float denominator = expf(stage->attn_sink.data[head] - maximum);
        for (int item = 0; item < count; item++) {
            scores[item] = expf(scores[item] - maximum);
            denominator += scores[item];
        }
        float *result = attended + (size_t)head * head_dim;
        for (int item = 0; item < count; item++) {
            float probability = scores[item] / denominator;
            for (int d = 0; d < head_dim; d++)
                result[d] += probability * keys[item][d];
        }
        v4_ds_bf16(result, (size_t)head_dim);
        coli_v4_rope_apply(result + head_dim - rope_dim, 1, rope_dim,
                           cosines + (size_t)query_position * pairs,
                           sines + (size_t)query_position * pairs, 1);
        v4_ds_bf16(result + head_dim - rope_dim, (size_t)rope_dim);
    }

    int heads_per_group = heads / groups;
    int group_width = heads_per_group * head_dim;
    int scale_columns = (group_width + 127) / 128;
    int scale_rows = (o_rank + 127) / 128;
    for (int group = 0; group < groups; group++) {
        ColiTensorView view;
        v4_ds_fp8_view(&view,
            stage->wo_a_w + (size_t)group * o_rank * group_width,
            stage->wo_a_s + (size_t)group * scale_rows * scale_columns,
            o_rank, group_width);
        if (coli_fp8_matvec_ref(oa + (size_t)group * o_rank, &view,
                                attended + (size_t)group * group_width))
            goto fail;
        v4_ds_bf16(oa + (size_t)group * o_rank, (size_t)o_rank);
    }
    if (v4_ds_mm(out, stage->wo_b_w, stage->wo_b_s,
                 hidden, (int64_t)groups * o_rank, oa))
        goto fail;
    free(oa); free(attended); free(query); free(qa);
    return 0;

fail:
    free(oa); free(attended); free(query); free(qa);
    return -1;
}

static int v4_ds_moe(float *out, V4DSparkStage *stage,
                     const float *input,
                     const ColiSafetensorsIndex *index,
                     const ColiDeepSeekV4Config *config,
                     int *miss_budget) {
    int hidden = config->hidden_size;
    int intermediate = config->moe_intermediate_size;
    int topk = config->num_experts_per_tok;
    float weights[16];
    int experts[16];
    if (topk > 16 || coli_v4_route(weights, experts, input,
            stage->gate_w.data, stage->gate_b.data, NULL,
            config->n_routed_experts, hidden, topk,
            config->routed_scaling_factor))
        return -1;

    ColiTensorView shared_gate, shared_down, shared_up;
    v4_ds_fp8_view(&shared_gate, stage->sh1_w, stage->sh1_s,
                   intermediate, hidden);
    v4_ds_fp8_view(&shared_down, stage->sh2_w, stage->sh2_s,
                   hidden, intermediate);
    v4_ds_fp8_view(&shared_up, stage->sh3_w, stage->sh3_s,
                   intermediate, hidden);
    if (coli_v4_shared_expert_forward_ref(out, &shared_gate, &shared_down,
                                          &shared_up, input,
                                          config->swiglu_limit))
        return -1;

    float *branch = malloc((size_t)hidden * sizeof(float));
    if (!branch) return -1;
    for (int rank = 0; rank < topk; rank++) {
        ColiExpertView expert;
        if (!v4_ds_expert(index, stage, experts[rank], &expert, miss_budget)) {
            free(branch);
            return 1;                   /* bounded cold-start abort */
        }
        if (coli_v4_expert_forward_ref(branch, &expert, input,
                                       weights[rank],
                                       config->swiglu_limit)) {
            free(branch);
            return -1;
        }
        for (int i = 0; i < hidden; i++) out[i] += branch[i];
    }
    v4_ds_bf16(out, (size_t)hidden);
    free(branch);
    return 0;
}

/* One complete MTP transformer stage over all five positions.  Attention is
 * split into prepare/all-KV then query phases so every position can see the
 * entire block, exactly as the parallel reference does. */
static int v4_ds_stage_forward(float *state, V4DSparkStage *stage,
                               int64_t first_position,
                               const ColiSafetensorsIndex *index,
                               const ColiDeepSeekV4Config *config,
                               const float *cosines, const float *sines,
                               int *miss_budget) {
    int hidden = config->hidden_size;
    int hc = config->hc_mult;
    int head_dim = config->head_dim;
    int rope_dim = config->qk_rope_head_dim;
    int pairs = rope_dim / 2;
    size_t flat = (size_t)hc * hidden;

    float *residual = malloc((size_t)V4_DSPARK_BLOCK * flat * sizeof(float));
    float *normed = malloc((size_t)V4_DSPARK_BLOCK * hidden * sizeof(float));
    float *block_kv = malloc((size_t)V4_DSPARK_BLOCK * head_dim * sizeof(float));
    float *posts = malloc((size_t)V4_DSPARK_BLOCK * hc * sizeof(float));
    float *combs = malloc((size_t)V4_DSPARK_BLOCK * hc * hc * sizeof(float));
    float *branch = malloc((size_t)config->num_attention_heads * head_dim *
                           sizeof(float));
    if (!residual || !normed || !block_kv || !posts || !combs || !branch) {
        free(branch); free(combs); free(posts); free(block_kv);
        free(normed); free(residual); return -1;
    }

    memcpy(residual, state,
           (size_t)V4_DSPARK_BLOCK * flat * sizeof(float));
    for (int item = 0; item < V4_DSPARK_BLOCK; item++) {
        float *x = state + (size_t)item * flat;
        float *xn = normed + (size_t)item * hidden;
        float *post = posts + (size_t)item * hc;
        float *comb = combs + (size_t)item * hc * hc;
        v4_ds_hc_pre(xn, x, &stage->hc_attn_fn,
                     &stage->hc_attn_base, &stage->hc_attn_scale,
                     config, post, comb);
        v4_ds_rmsnorm(xn, xn, stage->attn_norm.data, hidden,
                      config->rms_norm_eps);
        float *kv = block_kv + (size_t)item * head_dim;
        if (v4_ds_mm(kv, stage->wkv_w, stage->wkv_s,
                     head_dim, hidden, xn))
            goto fail;
        v4_ds_rmsnorm(kv, kv, stage->kv_norm.data, head_dim,
                      config->rms_norm_eps);
        int64_t absolute = first_position + item;
        coli_v4_rope_apply(kv + head_dim - rope_dim, 1, rope_dim,
                           cosines + (size_t)absolute * pairs,
                           sines + (size_t)absolute * pairs, 0);
        v4_ds_bf16(kv + head_dim - rope_dim, (size_t)rope_dim);
        if (v4_ds_kv_qdq(kv, head_dim, rope_dim)) goto fail;
    }
    for (int item = 0; item < V4_DSPARK_BLOCK; item++) {
        float *x = state + (size_t)item * flat;
        const float *xn = normed + (size_t)item * hidden;
        if (v4_ds_attention(branch, stage, xn, first_position + item,
                            block_kv, V4_DSPARK_BLOCK, config,
                            cosines, sines))
            goto fail;
        v4_ds_hc_post(x, branch, residual + (size_t)item * flat,
                      posts + (size_t)item * hc,
                      combs + (size_t)item * hc * hc, config);
    }

    memcpy(residual, state,
           (size_t)V4_DSPARK_BLOCK * flat * sizeof(float));
    for (int item = 0; item < V4_DSPARK_BLOCK; item++) {
        float *x = state + (size_t)item * flat;
        float *xn = normed + (size_t)item * hidden;
        float *post = posts + (size_t)item * hc;
        float *comb = combs + (size_t)item * hc * hc;
        v4_ds_hc_pre(xn, x, &stage->hc_ffn_fn,
                     &stage->hc_ffn_base, &stage->hc_ffn_scale,
                     config, post, comb);
        v4_ds_rmsnorm(xn, xn, stage->ffn_norm.data, hidden,
                      config->rms_norm_eps);
        int result = v4_ds_moe(branch, stage, xn, index, config, miss_budget);
        if (result < 0) goto fail;
        if (result > 0) {
            free(branch); free(combs); free(posts); free(block_kv);
            free(normed); free(residual); return 1;
        }
        v4_ds_hc_post(x, branch, residual + (size_t)item * flat,
                      post, comb, config);
    }

    free(branch); free(combs); free(posts); free(block_kv);
    free(normed); free(residual);
    return 0;

fail:
    free(branch); free(combs); free(posts); free(block_kv);
    free(normed); free(residual);
    return -1;
}

static int v4_ds_final_hidden(float *out, const float *state,
                              const ColiDeepSeekV4Config *config) {
    int hidden = config->hidden_size;
    int hc = config->hc_mult;
    int flat = hidden * hc;
    float square = 0.0f;
    for (int i = 0; i < flat; i++) square += state[i] * state[i];
    float inverse = 1.0f /
        sqrtf(square / flat + config->rms_norm_eps);
    float pre[16];
    for (int copy = 0; copy < hc; copy++) {
        float mix = 0.0f;
        const float *weight = g_v4ds_core.head_fn.data +
                              (size_t)copy * flat;
        for (int i = 0; i < flat; i++) mix += weight[i] * state[i];
        float z = mix * inverse * g_v4ds_core.head_scale.data[0] +
                  g_v4ds_core.head_base.data[copy];
        pre[copy] = 1.0f / (1.0f + expf(-z)) + config->hc_eps;
    }
    for (int i = 0; i < hidden; i++) {
        float sum = 0.0f;
        for (int copy = 0; copy < hc; copy++)
            sum += pre[copy] * state[(size_t)copy * hidden + i];
        out[i] = coli_bf16_round(sum);
    }
    v4_ds_rmsnorm(out, out, g_v4ds_core.norm.data, hidden,
                  config->rms_norm_eps);
    return 0;
}

static void v4_ds_markov_embed(float out[V4_DSPARK_RANK], int token) {
    const uint16_t *row = g_v4ds_core.markov_w1 +
                          (size_t)token * V4_DSPARK_RANK;
    for (int i = 0; i < V4_DSPARK_RANK; i++) out[i] = coli_bf16_decode(row[i]);
}

/* Compute all backbone logits in one vocabulary sweep so head.weight is read
 * once for the five positions.  The small Markov correction remains serial,
 * which is the defining semi-autoregressive part of DSpark. */
static int v4_ds_heads(ColiV4Engine *engine,
                       const ColiSafetensorsIndex *index,
                       const ColiDeepSeekV4Config *config,
                       const float *state, int anchor_token,
                       int *draft, float *confidence) {
    const ColiSafetensorsTensor *head = coli_st_find(index, "head.weight");
    if (!head || head->dtype != COLI_ST_BF16) return -1;
    int head_shard = coli_st_tensor_shard(index, head);
    if (head_shard < 0 || head->off < 0) return -1;
    uint64_t head_offset = (uint64_t)head->off;
    int hidden = config->hidden_size;
    int vocab = config->vocab_size;
    int hc = config->hc_mult;
    size_t flat = (size_t)hc * hidden;
    float *hiddens = malloc((size_t)V4_DSPARK_BLOCK * hidden * sizeof(float));
    float *logits = malloc((size_t)V4_DSPARK_BLOCK * vocab * sizeof(float));
    if (!hiddens || !logits) {
        free(logits); free(hiddens); return -1;
    }
    for (int item = 0; item < V4_DSPARK_BLOCK; item++)
        if (v4_ds_final_hidden(hiddens + (size_t)item * hidden,
                               state + (size_t)item * flat, config)) {
            free(logits); free(hiddens); return -1;
        }

    const uint16_t *resident = coli_v4_head_cache_data(
        engine, head_shard, head_offset, (size_t)head->nbytes);
    enum { ROWS = 64 };
    uint16_t *raw = resident ? NULL :
        malloc((size_t)ROWS * hidden * sizeof(uint16_t));
    if (!resident && !raw) {
        free(logits); free(hiddens); return -1;
    }
    for (int start = 0; start < vocab; start += ROWS) {
        int rows = vocab - start < ROWS ? vocab - start : ROWS;
        const uint16_t *weights = resident
            ? resident + (size_t)start * hidden : raw;
        if (!resident) {
            if (coli_st_read_at_engine(engine, index, head_shard,
                    head_offset + (uint64_t)start * hidden * sizeof(uint16_t),
                    (size_t)rows * hidden * sizeof(uint16_t), raw)) {
                free(raw); free(logits); free(hiddens); return -1;
            }
            weights = raw;
        }
        #pragma omp parallel for
        for (int row = 0; row < rows; row++) {
            const uint16_t *weight = weights + (size_t)row * hidden;
            float sums[V4_DSPARK_BLOCK] = {0};
            for (int d = 0; d < hidden; d++) {
                float w = coli_bf16_decode(weight[d]);
                for (int item = 0; item < V4_DSPARK_BLOCK; item++)
                    sums[item] += w * hiddens[(size_t)item * hidden + d];
            }
            for (int item = 0; item < V4_DSPARK_BLOCK; item++)
                logits[(size_t)item * vocab + start + row] = sums[item];
        }
    }
    free(raw);

    int previous = anchor_token;
    for (int item = 0; item < V4_DSPARK_BLOCK; item++) {
        float embedding[V4_DSPARK_RANK];
        v4_ds_markov_embed(embedding, previous);
        int winner = -1;
        float maximum = -FLT_MAX;
        #pragma omp parallel
        {
            int local_winner = -1;
            float local_maximum = -FLT_MAX;
            #pragma omp for nowait
            for (int token = 0; token < vocab; token++) {
                const uint16_t *weight = g_v4ds_core.markov_w2 +
                    (size_t)token * V4_DSPARK_RANK;
                float score = logits[(size_t)item * vocab + token];
                for (int rank = 0; rank < V4_DSPARK_RANK; rank++)
                    score += coli_bf16_decode(weight[rank]) * embedding[rank];
                if (score > local_maximum) {
                    local_maximum = score;
                    local_winner = token;
                }
            }
            #pragma omp critical
            if (local_maximum > maximum) {
                maximum = local_maximum;
                winner = local_winner;
            }
        }
        if (winner < 0) {
            free(logits); free(hiddens); return -1;
        }
        draft[item] = winner;
        previous = winner;

        float score = 0.0f;
        const float *hidden_row = hiddens + (size_t)item * hidden;
        for (int d = 0; d < hidden; d++)
            score += g_v4ds_core.confidence_proj.data[d] * hidden_row[d];
        for (int rank = 0; rank < V4_DSPARK_RANK; rank++)
            score += g_v4ds_core.confidence_proj.data[hidden + rank] *
                     embedding[rank];
        confidence[item] = score >= 0.0f
            ? 1.0f / (1.0f + expf(-score))
            : expf(score) / (1.0f + expf(score));
    }
    free(logits);
    free(hiddens);
    return 0;
}

static int v4_dspark_draft(ColiV4Engine *engine,
                           const ColiSafetensorsIndex *index,
                           const ColiDeepSeekV4Config *config,
                           int anchor_token, int64_t first_position,
                           int *draft, int max_out) {
    if (!engine || !index || !config || !draft || max_out < 1) return 0;
    double began = spec_now();
    if (!v4_ds_load_all(index, config)) return 0;
    double loaded = spec_now();

    int hidden = config->hidden_size;
    int hc = config->hc_mult;
    int rope_dim = config->qk_rope_head_dim;
    int pairs = rope_dim / 2;
    int64_t table_rows = first_position + V4_DSPARK_BLOCK;
    if (table_rows < V4_DSPARK_BLOCK || table_rows > INT_MAX) return 0;
    float *cosines = malloc((size_t)table_rows * pairs * sizeof(float));
    float *sines = malloc((size_t)table_rows * pairs * sizeof(float));
    float *state = malloc((size_t)V4_DSPARK_BLOCK * hc * hidden *
                          sizeof(float));
    if (!cosines || !sines || !state) {
        free(state); free(sines); free(cosines); return 0;
    }

    /* MTP stages have compress_ratio=0, hence base RoPE without YaRN. */
    if (coli_v4_rope_precompute(cosines, sines, rope_dim, (int)table_rows,
            0, config->rope_theta, config->rope_factor,
            config->rope_beta_fast, config->rope_beta_slow) ||
        v4_ds_backfill_context(config, first_position, cosines, sines)) {
        free(state); free(sines); free(cosines); return 0;
    }

    int noise_token = getenv("V4_MTP_NOISE")
        ? atoi(getenv("V4_MTP_NOISE")) : 128799;
    size_t flat = (size_t)hc * hidden;
    for (int item = 0; item < V4_DSPARK_BLOCK; item++) {
        int token = item == 0 ? anchor_token : noise_token;
        if (load_embedding(state + (size_t)item * flat,
                           index, config, token)) {
            free(state); free(sines); free(cosines); return 0;
        }
    }

    /* A cold first block can route to at most 5*6*3 distinct records.  The
     * cache normally makes subsequent rounds zero-I/O; users can lower this
     * cap to warm gradually on slower or memory-constrained machines. */
    int miss_budget = getenv("V4_MTP_MISS")
        ? atoi(getenv("V4_MTP_MISS")) : 96;
    if (miss_budget < 0) miss_budget = 0;
    double prepared = spec_now();
    double stage_seconds[V4_DSPARK_STAGES] = {0};
    for (int stage_id = 0; stage_id < V4_DSPARK_STAGES; stage_id++) {
        double stage_began = spec_now();
        int result = v4_ds_stage_forward(state, &g_v4ds[stage_id], first_position,
                                         index, config, cosines, sines,
                                         &miss_budget);
        stage_seconds[stage_id] = spec_now() - stage_began;
        if (result != 0) {
            free(state); free(sines); free(cosines); return 0;
        }
    }

    int candidates[V4_DSPARK_BLOCK];
    float confidence[V4_DSPARK_BLOCK];
    if (v4_ds_heads(engine, index, config, state, anchor_token,
                    candidates, confidence)) {
        free(state); free(sines); free(cosines); return 0;
    }
    free(state);
    free(sines);
    free(cosines);
    double headed = spec_now();

    int count = max_out < V4_DSPARK_BLOCK ? max_out : V4_DSPARK_BLOCK;
    double threshold = getenv("V4_MTP_CONF")
        ? atof(getenv("V4_MTP_CONF")) : 0.55;
    if (threshold > 0.0) {
        for (int item = 0; item < count; item++) {
            /* Blend the trained confidence with observed survival on this
             * exact machine/workload.  SSD verification is kept only while
             * its estimated prefix survival clears the configured price. */
            double survival = 0.70 * confidence[item] +
                              0.30 * g_v4ds_core.accept_ewma[item];
            if (survival < threshold) {
                count = item;
                break;
            }
        }
    }
    if (count < 1) return 0;
    memcpy(draft, candidates, (size_t)count * sizeof(int));
    g_v4ds_core.rounds++;
    g_v4ds_core.drafted += (uint64_t)count;

    if (getenv("V4_MTP_LOG") && atoi(getenv("V4_MTP_LOG"))) {
        fprintf(stderr,
                "[MTP] draft n=%d conf=%.3f,%.3f,%.3f,%.3f,%.3f "
                "cold_left=%d time=load:%.3f prep:%.3f "
                "stages:%.3f/%.3f/%.3f head:%.3f total:%.3f\n",
                count, confidence[0], confidence[1], confidence[2],
                confidence[3], confidence[4], miss_budget,
                loaded - began, prepared - loaded,
                stage_seconds[0], stage_seconds[1], stage_seconds[2],
                headed - prepared - stage_seconds[0] - stage_seconds[1] -
                    stage_seconds[2],
                headed - began);
    }
    return count;
}

static void v4_dspark_feedback(int drafted, int accepted) {
    if (drafted < 0) return;
    if (accepted < 0) accepted = 0;
    if (accepted > drafted) accepted = drafted;
    g_v4ds_core.accepted += (uint64_t)accepted;
    for (int item = 0; item < drafted && item < V4_DSPARK_BLOCK; item++) {
        double sample = item < accepted ? 1.0 : 0.0;
        g_v4ds_core.accept_ewma[item] =
            0.90 * g_v4ds_core.accept_ewma[item] + 0.10 * sample;
    }
}

static void v4_dspark_report(void) {
    if (!g_v4ds_core.rounds) return;
    fprintf(stderr,
            "[MTP] rounds=%llu drafted=%llu accepted=%llu rate=%.1f%% "
            "ewma=%.2f,%.2f,%.2f,%.2f,%.2f\n",
            (unsigned long long)g_v4ds_core.rounds,
            (unsigned long long)g_v4ds_core.drafted,
            (unsigned long long)g_v4ds_core.accepted,
            g_v4ds_core.drafted
                ? 100.0 * g_v4ds_core.accepted / g_v4ds_core.drafted : 0.0,
            g_v4ds_core.accept_ewma[0], g_v4ds_core.accept_ewma[1],
            g_v4ds_core.accept_ewma[2], g_v4ds_core.accept_ewma[3],
            g_v4ds_core.accept_ewma[4]);
    for (int stage_id = 0; stage_id < V4_DSPARK_STAGES; stage_id++) {
        V4DSparkStage *stage = &g_v4ds[stage_id];
        fprintf(stderr,
                "[MTP] stage=%d cache=%d/%d hits=%lld misses=%lld aborts=%lld\n",
                stage_id, stage->nslots, stage->cap_slots,
                (long long)stage->hits, (long long)stage->misses,
                (long long)stage->aborted);
    }
}
