#define COLI_V4_UNIT_ATTENTION_BATCH
#include "../deepseek_v4.c"
#include <assert.h>

static void add_compressor(ColiDeepSeekV4LayerWeights *w, const char *prefix,
                           int ratio, int dim) {
    int n = (int)w->plan.tensor_count;
    snprintf(w->plan.tensors[n].name, COLI_V4_MAX_TENSOR_NAME,
             "layers.0.%s.ape", prefix);
    w->data[n] = calloc((size_t)ratio * (ratio == 4 ? 2 : 1) * dim, sizeof(float));
    snprintf(w->plan.tensors[n + 1].name, COLI_V4_MAX_TENSOR_NAME,
             "layers.0.%s.norm.weight", prefix);
    uint16_t *norm = malloc((size_t)dim * sizeof(*norm));
    for (int i = 0; i < dim; i++) norm[i] = 0x3f80;
    w->data[n + 1] = norm;
    w->plan.tensor_count += 2;
}

static void advance(ColiDeepSeekV4WindowAttentionState *s, int position,
    const float *x, const float *kv, const float *comp, const float *gate,
    const float *idx, const float *idx_gate) {
    char error[256] = {0};
    if (s->compressor) {
        int produced = 0;
        assert(!grow_compressed_state(s, error, sizeof(error)));
        assert(!coli_v4_compressor_advance(s->compressor,
            s->compressed + (size_t)s->compressed_count * s->head_dim,
            &produced, comp, gate, position, error, sizeof(error)));
        s->compressed_count += produced;
    }
    if (s->indexer)
        assert(coli_v4_indexer_advance(s->indexer, x, position, idx, idx_gate,
                                      error, sizeof(error)) >= 0);
    memcpy(s->kv + (size_t)(position % s->window_size) * s->head_dim,
           kv, (size_t)s->head_dim * sizeof(float));
}

static void equal_state(ColiDeepSeekV4WindowAttentionState *a,
                        ColiDeepSeekV4WindowAttentionState *b) {
    ColiV4AttentionSnapshot *sa = NULL, *sb = NULL;
    assert(!coli_v4_attention_snapshot_create(a, &sa));
    assert(!coli_v4_attention_snapshot_create(b, &sb));
    FILE *fa = tmpfile(), *fb = tmpfile(); assert(fa && fb);
    assert(!coli_v4_attention_snapshot_write(sa, fa));
    assert(!coli_v4_attention_snapshot_write(sb, fb));
    assert(ftell(fa) == ftell(fb)); rewind(fa); rewind(fb);
    int ca, cb;
    do { ca = fgetc(fa); cb = fgetc(fb); assert(ca == cb); } while (ca != EOF);
    fclose(fa); fclose(fb);
    coli_v4_attention_snapshot_destroy(sa); coli_v4_attention_snapshot_destroy(sb);
}

static void check(int ratio, int start) {
    enum { N = 6, H = 128, D = 128, P = 256 };
    ColiDeepSeekV4Config c = {0};
    c.hidden_size = H; c.head_dim = D; c.index_head_dim = D;
    c.index_n_heads = 1; c.sliding_window = 8; c.max_position_embeddings = 512;
    c.qk_rope_head_dim = 64; c.rms_norm_eps = 1e-6f;
    c.compress_rope_theta = 10000; c.rope_factor = 1;
    c.original_max_position_embeddings = 4096; c.rope_beta_fast = 32; c.rope_beta_slow = 1;
    ColiDeepSeekV4LayerWeights w = {0};
    w.plan.compression_ratio = ratio; w.plan.has_indexer = ratio == 4;
    if (ratio) add_compressor(&w, "attn.compressor", ratio, D);
    if (ratio == 4) add_compressor(&w, "attn.indexer.compressor", 4, D);
    ColiDeepSeekV4WindowAttentionState *a = NULL, *b = NULL;
    char error[256] = {0};
    assert(!coli_v4_window_attention_batch_create_copy(&a, &c));
    assert(!coli_v4_window_attention_batch_create_copy(&b, &c));
    assert(!prepare_compressed_state(a, &w, &c, error, sizeof(error)));
    assert(!prepare_compressed_state(b, &w, &c, error, sizeof(error)));
    int cr = ratio ? (ratio == 4 ? 2 : 1) * D : 0, ir = ratio == 4 ? P : 0;
    float x[N * H], kv[N * D], comp[N * P], gate[N * P], idx[N * P], ig[N * P];
    for (int i = 0; i < N * H; i++) x[i] = (i % 29 - 14) * .01f;
    for (int i = 0; i < N * D; i++) kv[i] = (i % 37 - 18) * .02f;
    for (int i = 0; i < N * P; i++) {
        comp[i] = (i % 43 - 21) * .07f; gate[i] = (i % 19 - 9) * .03f;
        idx[i] = (i % 31 - 15) * .05f; ig[i] = (i % 23 - 11) * .01f;
    }
    for (int pos = 0; pos < start; pos++) advance(a, pos, x, kv, comp, gate, idx, ig);
    ColiV4AttentionSnapshot *base = NULL;
    assert(!coli_v4_attention_snapshot_create(a, &base));
    assert(!coli_v4_attention_trial_begin(a, start, N));
    v4_attention_trial_record(a, &c, x, kv, start, 2, comp, gate, cr, idx, ig, ir);
    assert(!coli_v4_attention_trial_ready(a));
    v4_attention_trial_record(a, &c, x + 2 * H, kv + 2 * D, start + 2, N - 2,
        comp + 2 * cr, gate + 2 * cr, cr, idx + 2 * ir, ig + 2 * ir, ir);
    assert(coli_v4_attention_trial_ready(a));
    for (int kept = 0; kept <= N; kept++) {
        assert(!coli_v4_attention_snapshot_restore(a, base));
        for (int i = 0; i < N; i++) advance(a, start + i, x + i * H, kv + i * D,
            comp + i * cr, gate + i * cr, idx + i * ir, ig + i * ir);
        assert(!coli_v4_attention_snapshot_restore(a, base));
        assert(!coli_v4_attention_trial_retain(a, &w, &c, kept, error, sizeof(error)));
        assert(!coli_v4_attention_snapshot_restore(b, base));
        for (int i = 0; i < kept; i++) advance(b, start + i, x + i * H, kv + i * D,
            comp + i * cr, gate + i * cr, idx + i * ir, ig + i * ir);
        equal_state(a, b);
    }
    /* The journal owns its values, independent of reused attention scratch. */
    memset(kv, 0, sizeof(kv)); memset(comp, 0, sizeof(comp));
    assert(!coli_v4_attention_snapshot_restore(a, base));
    assert(!coli_v4_attention_trial_retain(a, &w, &c, N, error, sizeof(error)));
    equal_state(a, b);
    assert(coli_v4_attention_trial_retain(a, &w, &c, N + 1, error, sizeof(error)) < 0);
    assert(!coli_v4_attention_trial_begin(a, start, N));
    v4_attention_trial_record(a, &c, x, kv, start + 1, N, comp, gate, cr, idx, ig, ir);
    assert(!coli_v4_attention_trial_ready(a));
    if (ratio) {
        assert(!coli_v4_attention_trial_begin(a, start, N));
        v4_attention_trial_record(a, &c, x, kv, start, N, NULL, NULL, cr, idx, ig, ir);
        assert(!coli_v4_attention_trial_ready(a));
    }
    coli_v4_attention_snapshot_destroy(base);
    coli_v4_window_attention_batch_destroy_copy(a); coli_v4_window_attention_batch_destroy_copy(b);
    for (size_t i = 0; i < w.plan.tensor_count; i++) free(w.data[i]);
    printf("ratio=%d start=%d: all retained lengths, split chunks, ring wrap, fallback OK\n", ratio, start);
}
int main(void) {
    check(0, 7); check(4, 2); check(4, 7); check(128, 126); check(128, 255);
}
