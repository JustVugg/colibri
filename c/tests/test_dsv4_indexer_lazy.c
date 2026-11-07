/* Exercise production attention dispatch with real CPU math and injected GPU
 * success/failure. No CUDA initialization or model files are needed. */
#define coli_v4_indexer_select_batch observed_select
#define coli_v4_gpu_kv_cache_sync injected_sync
#define coli_v4_gpu_sparse_attention_batch_cached injected_cached
#define coli_v4_gpu_sparse_attention_batch injected_slab
#define COLI_V4_UNIT_ATTENTION_BATCH
#include "../deepseek_v4.c"
#undef coli_v4_indexer_select_batch
#undef coli_v4_gpu_kv_cache_sync
#undef coli_v4_gpu_sparse_attention_batch_cached
#undef coli_v4_gpu_sparse_attention_batch
#include <assert.h>

int coli_v4_indexer_select_batch(ColiDeepSeekV4Indexer *, int *, int,
    const float *, const float *, int, int, const int *, int *, char *, size_t);
static int selections, gpu_mode, reordered_full_set;
int observed_select(ColiDeepSeekV4Indexer *s, int *ids, int cap,
    const float *q, const float *x, int pos, int n, const int *counts,
    int *selected, char *error, size_t size) {
    selections++;
    int rc = coli_v4_indexer_select_batch(s, ids, cap, q, x, pos, n, counts,
                                         selected, error, size);
    if (!rc) for (int t = 0; t < n; t++) if (selected[t] == counts[t])
        for (int k = 0; k < selected[t]; k++)
            reordered_full_set |= ids[t * cap + k] != k;
    return rc;
}
int injected_sync(const ColiDeepSeekV4LayerWeights *w, const float *ring,
    int window, int dim, int pos, const float *compressed, int count) {
    return 1;
}
int injected_cached(const ColiDeepSeekV4LayerWeights *w, float *out,
    const float *q, const float *chunk, int pos, const float *sinks,
    const int *meta, int base, int count, int heads, int dim, int n) {
    if (gpu_mode) return -1;
    for (int i = 0; i < n * heads * dim; i++) out[i] = .125f;
    return 0;
}
int injected_slab(const ColiDeepSeekV4LayerWeights *w, float *out,
    const float *q, const float *values, const float *sinks, const int *meta,
    int rows, int base, int heads, int dim, int n) {
    if (gpu_mode == 2) return -1;
    for (int i = 0; i < n * heads * dim; i++) out[i] = .125f;
    return 0;
}

static void tensor(ColiDeepSeekV4LayerWeights *w, const char *suffix,
    ColiSafetensorsDType dtype, int rows, int cols, int fill) {
    int k = (int)w->plan.tensor_count++;
    ColiDeepSeekV4TensorSpec *t = &w->plan.tensors[k];
    snprintf(t->name, sizeof(t->name), "layers.0.%s", suffix);
    t->dtype = dtype; t->rank = cols ? 2 : 1; t->shape[0] = rows; t->shape[1] = cols;
    size_t count = (size_t)rows * (cols ? cols : 1);
    size_t width = dtype == COLI_ST_BF16 ? 2 : dtype == COLI_ST_F8_E4M3 ? 1 : 4;
    w->data[k] = calloc(count, width); assert(w->data[k]);
    if (dtype == COLI_ST_F8_E4M3) {
        for (int i = 0; i < rows; i++) ((uint8_t *)w->data[k])[i * cols + i % cols] = 0x38;
    } else if (dtype == COLI_ST_BF16) {
        for (size_t i = 0; i < count; i++) ((uint16_t *)w->data[k])[i] = fill ? 0x3f80 : (0x3a00 + (i * 13) % 512) | (i % 3 == 0 ? 0x8000 : 0);
    } else {
        for (size_t i = 0; i < count; i++) ((float *)w->data[k])[i] = fill ? 1.f : 0.f;
    }
}
static void fp8(ColiDeepSeekV4LayerWeights *w, const char *prefix) {
    char s[128]; snprintf(s, sizeof(s), "%s.weight", prefix);
    tensor(w, s, COLI_ST_F8_E4M3, 128, 128, 0);
    snprintf(s, sizeof(s), "%s.scale", prefix);
    tensor(w, s, COLI_ST_F8_E8M0, 1, 1, 1);
}
static void weights(ColiDeepSeekV4LayerWeights *w) {
    w->plan.compression_ratio = 4; w->plan.has_indexer = 1;
    const char *fp[] = {"attn.wq_a", "attn.wq_b", "attn.wkv", "attn.wo_a", "attn.wo_b", "attn.indexer.wq_b"};
    for (int i = 0; i < 6; i++) fp8(w, fp[i]);
    tensor(w, "attn.q_norm.weight", COLI_ST_BF16, 128, 0, 1);
    tensor(w, "attn.kv_norm.weight", COLI_ST_BF16, 128, 0, 1);
    tensor(w, "attn.attn_sink", COLI_ST_F32, 1, 0, 0);
    tensor(w, "attn.indexer.weights_proj.weight", COLI_ST_BF16, 1, 128, 0);
    const char *prefix[] = {"attn.compressor", "attn.indexer.compressor"};
    for (int i = 0; i < 2; i++) {
        char s[128]; snprintf(s, sizeof(s), "%s.ape", prefix[i]);
        tensor(w, s, COLI_ST_F32, 4, 256, 0);
        snprintf(s, sizeof(s), "%s.norm.weight", prefix[i]);
        tensor(w, s, COLI_ST_BF16, 128, 0, 1);
        snprintf(s, sizeof(s), "%s.wkv.weight", prefix[i]);
        tensor(w, s, COLI_ST_BF16, 256, 128, 0);
        snprintf(s, sizeof(s), "%s.wgate.weight", prefix[i]);
        tensor(w, s, COLI_ST_BF16, 256, 128, 0);
    }
}
static void check(int n, int topk, int mode) {
    ColiDeepSeekV4Config c = {0};
    c.hidden_size = c.head_dim = c.q_lora_rank = c.o_lora_rank = c.index_head_dim = 128;
    c.num_attention_heads = c.o_groups = c.index_n_heads = 1;
    c.index_topk = topk; c.sliding_window = 8; c.max_position_embeddings = 128;
    c.qk_rope_head_dim = 64; c.rms_norm_eps = 1e-6f;
    c.rope_theta = c.compress_rope_theta = 10000; c.rope_factor = 1;
    c.original_max_position_embeddings = 4096; c.rope_beta_fast = 32; c.rope_beta_slow = 1;
    ColiDeepSeekV4LayerWeights w = {0}; weights(&w);
    float x[12 * 128], a[12 * 128], b[12 * 128]; char error[256] = {0};
    for (int i = 0; i < 12 * 128; i++) x[i] = (i % 37 - 18) * .03125f;
    gpu_mode = mode;
    for (int lazy = 0; lazy < 2; lazy++) {
        setenv("DSV4_CUDA_INDEXER_LAZY", lazy ? "1" : "0", 1);
        ColiDeepSeekV4WindowAttentionState *s = NULL;
        assert(!coli_v4_window_attention_batch_create_copy(&s, &c));
        assert(!coli_v4_attention_window_batch_ref(b, s, &w, &c, x, 0, 12, error, sizeof(error)));
        selections = 0;
        assert(!coli_v4_attention_window_batch_ref(lazy ? b : a, s, &w, &c, x, 12, n, error, sizeof(error)));
        int full = (12 + n) / 4 <= topk;
        assert(selections == (lazy && full && mode != 2 ? 0 : 1));
        if (lazy) assert(!memcmp(a, b, (size_t)n * 128 * sizeof(float)));
        coli_v4_window_attention_batch_destroy_copy(s);
    }
    for (size_t i = 0; i < w.plan.tensor_count; i++) free(w.data[i]);
    printf("batch=%d topk=%d gpu-mode=%d: exact output and selection dispatch\n", n, topk, mode);
}
int main(void) {
    setenv("COLI_CUDA_ATTN_BATCH", "1", 1); setenv("V4_UNIFIED_DECODE", "1", 1);
    setenv("V4_IDX_IDENTITY", "0", 1);
    for (int mode = 0; mode < 3; mode++) for (int topk = 2; topk <= 4; topk++) {
        check(1, topk, mode); check(6, topk, mode);
    }
    assert(reordered_full_set);
}
