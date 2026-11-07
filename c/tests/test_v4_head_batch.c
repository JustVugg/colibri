/* Exercise the real speculative head dispatch, verification and CPU fallback. */
#include <pthread.h>
#define COLI_V4_GPU_TIER
#define COLI_V4_UNIT_GENERATE_STATS
#define COLI_V4_SKIP_GENERATE_MAIN
#include "../deepseek_v4.c"
#include <assert.h>

static uint16_t test_weights[32 * 8];
static ColiSafetensorsTensor test_head = {.dtype = COLI_ST_BF16};
static int gpu_available, corrupt_scores, corrupt_id, score_calls, compact_calls;
float coli_bf16_decode(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result; memcpy(&result, &bits, sizeof(result)); return result;
}
const ColiSafetensorsTensor *coli_st_find(const ColiSafetensorsIndex *index,const char *name) {
    (void)index; assert(!strcmp(name, "head.weight")); return &test_head;
}
int coli_st_tensor_shard(const ColiSafetensorsIndex *index,const ColiSafetensorsTensor *tensor) {
    (void)index; assert(tensor == &test_head); return 0;
}
const void *coli_v4_head_cache_data(const ColiV4Engine *engine,int shard,uint64_t offset,size_t length) {
    (void)engine; assert(!shard && !offset && length == sizeof(test_weights)); return test_weights;
}
int coli_st_read_at_engine(ColiV4Engine *engine,const ColiSafetensorsIndex *index,
                            int shard,uint64_t offset,size_t length,void *out) {
    (void)engine; (void)index; (void)shard; (void)offset; (void)length; (void)out;
    assert(0); return -1;
}
int coli_v4_gpu_head_scores(ColiV4Engine *engine,const float *input,float *scores) {
    (void)engine; (void)input; (void)scores; return -1;
}
int coli_v4_gpu_head_batch(ColiV4Engine *engine,const float *input,int batch,
                           float *scores,int *ids,float *values) {
    (void)engine;
    if (!gpu_available) return -1;
    if (scores) score_calls++; else compact_calls++;
    for (int item = 0; item < batch; item++) {
        float row[32];
        for (int i = 0; i < 32; i++)
            row[i] = head_bf16_dot(test_weights + i * 8, input + item * 8, 8);
        if (scores) memcpy(scores + item * 32, row, sizeof(row));
        else {
            (void)head_scores_argmax(row, 32, ids + item, values + item);
            if (corrupt_id) ids[item]++;
        }
    }
    if (scores && corrupt_scores) scores[31] += 1.f;
    return 0;
}

int main(void) {
    ColiV4Engine engine = {0}; ColiSafetensorsIndex index = {0};
    ColiDeepSeekV4Config config = {.hidden_size = 8, .vocab_size = 32};
    float input[3 * 8], values[3]; int ids[3];
    for (int i = 0; i < 3 * 8; i++) input[i] = 1.f;
    for (int i = 0; i < 2 * 8; i++) test_weights[i] = 0x3f80;
    unsetenv("DSV4_HEAD_VERIFY");
    assert(!head_argmax_batch(&engine, input, &index, &config, 3, ids, values));
    assert(g_v4_prof_head_s > 0 && !compact_calls);
    for (int i = 0; i < 3; i++) assert(ids[i] == 0 && values[i] == 8.f);
    gpu_available = 1;
    assert(!head_argmax_batch(&engine, input, &index, &config, 3, ids, values));
    assert(compact_calls == 1 && !score_calls);
    setenv("DSV4_HEAD_VERIFY", "1", 1);
    assert(!head_argmax_batch(&engine, input, &index, &config, 3, ids, values));
    assert(compact_calls == 2 && score_calls == 1);
    corrupt_scores = 1;
    assert(head_argmax_batch(&engine, input, &index, &config, 3, ids, values) == -1);
    corrupt_scores = 0; corrupt_id = 1;
    assert(head_argmax_batch(&engine, input, &index, &config, 3, ids, values) == -1);
    corrupt_id = 0; gpu_available = 0;
    assert(!head_argmax_batch(&engine, input, &index, &config, 3, ids, values));
    for (int i = 0; i < 3 * 8; i++) input[i] = NAN;
    assert(head_argmax_batch(&engine, input, &index, &config, 3, ids, values) == -1);
    gpu_available = 1; unsetenv("DSV4_HEAD_VERIFY");
    assert(head_argmax_batch(&engine, input, &index, &config, 3, ids, values) == -1);
    puts("test_v4_head_batch: ok");
    return 0;
}
