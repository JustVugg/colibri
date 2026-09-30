/* Exercise the production GPU unit with a tiny backend, no CUDA toolkit. */
#include <time.h>
#include <pthread.h>
#define COLI_V4_GPU_TIER
#define COLI_V4_UNIT_GPU
#include "../deepseek_v4.c"
#include <assert.h>

struct Dsv4CudaTensor { int device; };
struct Dsv4CudaExpertSet { int device; };
struct Dsv4CudaActivation { int device; long long elements; };
static int activations_created, activations_freed;
Dsv4CudaActivation *dsv4_cuda_activation_create(int device, long long elements) {
    Dsv4CudaActivation *a = malloc(sizeof(*a)); assert(a);
    a->device = device; a->elements = elements; activations_created++; return a;
}
void dsv4_cuda_activation_free(Dsv4CudaActivation *a) {
    if (a) { activations_freed++; free(a); }
}
static void test_batch_scratch(void) {
    Dsv4CudaActivation *a = v4_gpu_batch_scratch(V4_BATCH_FP8_IN, 0, 128);
    Dsv4CudaActivation *b = v4_gpu_batch_scratch(V4_BATCH_FP8_IN, 5, 256);
    assert(a && b && a != b && a->device == 0 && b->device == 5);
    assert(v4_gpu_batch_scratch(V4_BATCH_FP8_IN, 0, 64) == a);
    assert(activations_created == 2 && activations_freed == 0);
    assert(v4_gpu_batch_scratch(V4_BATCH_FP8_IN, 0, 512)->elements == 512);
    assert(activations_created == 3 && activations_freed == 1);
    assert(v4_gpu_batch_scratch(V4_BATCH_FP8_IN, 5, 256) == b);
    v4_gpu_batch_scratch_release();
    assert(activations_created == activations_freed && !v4_batch_devices);
}
static int sets_created, sets_freed;
void dsv4_cuda_expert_set_free(Dsv4CudaExpertSet *set) {
    if (set) { sets_freed++; free(set); }
}
static int initialized, shutdowns, live_tensors, free_queries[16], drained[16];
static int upload_fail_after = -1, low_memory_device = -1;
int dsv4_cuda_backend_arch_ok(int device) { return device >= 0 && device < 6; }
const char *dsv4_cuda_backend_name(void) { return "test"; }
int dsv4_cuda_init(const int *devices, int count) {
    for (int i = 0; i < count; i++)
        if (devices[i] < 0 || devices[i] >= 6) return 0;
    initialized = count;
    assert(count == 1 || devices[0] == 5);
    return 1;
}
void dsv4_cuda_shutdown(void) { shutdowns++; }
int dsv4_cuda_device_unified(int device) { (void)device; return 0; }
long long dsv4_cuda_mem_free_mb(int device) { free_queries[device]++; return device == low_memory_device ? 0 : 32000; }
int dsv4_cuda_stream_drain(int device) { drained[device]++; return device != 3; }
void dsv4_cuda_tensor_free(Dsv4CudaTensor *t) { if (t) { live_tensors--; free(t); } }
int dsv4_cuda_upload_fp4(Dsv4CudaTensor **t, const uint8_t *w,
                          const uint8_t *scale, int rows, int cols, int device) {
    (void)w; (void)scale; (void)rows; (void)cols;
    if (upload_fail_after == 0) return 0;
    if (upload_fail_after > 0) upload_fail_after--;
    *t = malloc(sizeof(**t));
    assert(*t);
    (*t)->device = device;
    live_tensors++;
    return 1;
}
int dsv4_cuda_tensor_refill_fp4(Dsv4CudaTensor *t, const uint8_t *w,
                                 const uint8_t *scale, int rows, int cols, int sync) {
    (void)t; (void)w; (void)scale; (void)rows; (void)cols; (void)sync;
    return 1;
}

static int head_upload_ok = 1, head_scores_ok = 1, head_uploads;
long long dsv4_cuda_tensor_bytes(const Dsv4CudaTensor *t) { return t ? 512 : 0; }
int dsv4_cuda_upload_head_exact(Dsv4CudaTensor **t,const uint16_t *w,int rows,int cols,int device) {
    assert(w && rows == 32 && cols == 8);
    head_uploads++;
    if (!head_upload_ok) return 0;
    *t = malloc(sizeof(**t)); assert(*t); (*t)->device = device; live_tensors++;
    return 1;
}
int dsv4_cuda_head_scores_exact(Dsv4CudaTensor *t,const float *input,float *scores) {
    assert(t && input && scores);
    if (head_scores_ok) scores[0] = input[0];
    return head_scores_ok;
}
static int head_batch_calls;
int dsv4_cuda_head_scores_batch_exact(Dsv4CudaTensor *t,const float *input,int batch,float *scores) {
    assert(t && input && scores && batch == 3);
    head_batch_calls++;
    return head_scores_ok;
}
int dsv4_cuda_head_argmax_batch_exact(Dsv4CudaTensor *t,const float *input,int batch,int *ids,float *values) {
    assert(t && input && ids && values && batch == 3);
    head_batch_calls++;
    return head_scores_ok;
}
static void test_head(void) {
#if defined(__AVX2__)
    ColiV4Engine *engine = calloc(1, sizeof(*engine)); assert(engine);
    engine->gpu.enabled = 1;
    engine->config.vocab_size = 32; engine->config.hidden_size = 8;
    uint16_t weights[256] = {0};
    engine->head_cache.data = (unsigned char *)weights; engine->head_cache.bytes = sizeof(weights);
    setenv("DSV4_CUDA_HEAD", "0", 1);
    coli_v4_gpu_head_upload(engine); assert(!engine->gpu.head && !head_uploads);
    unsetenv("DSV4_CUDA_HEAD"); low_memory_device = 0;
    coli_v4_gpu_head_upload(engine); assert(!engine->gpu.head && !head_uploads);
    low_memory_device = -1; head_upload_ok = 0;
    coli_v4_gpu_head_upload(engine); assert(!engine->gpu.head && head_uploads == 1);
    head_upload_ok = 1;
    coli_v4_gpu_head_upload(engine); assert(engine->gpu.head && live_tensors == 1);
    assert(engine->gpu.uploaded_bytes == 512);
    coli_v4_gpu_head_upload(engine); assert(head_uploads == 2);
    float input = 7, scores = -1;
    assert(!coli_v4_gpu_head_scores(engine, &input, &scores) && scores == input);
    int ids[3]; float values[3];
    unsetenv("DSV4_CUDA_HEAD_BATCH");
    assert(!coli_v4_gpu_head_batch(engine, &input, 3, NULL, ids, values));
    setenv("DSV4_CUDA_HEAD_BATCH", "0", 1);
    assert(coli_v4_gpu_head_batch(engine, &input, 3, NULL, ids, values) == -1);
    assert(head_batch_calls == 1);
    setenv("DSV4_CUDA_HEAD_BATCH", "1", 1);
    assert(!coli_v4_gpu_head_batch(engine, &input, 3, NULL, ids, values));
    assert(!coli_v4_gpu_head_batch(engine, &input, 3, &scores, NULL, NULL));
    assert(coli_v4_gpu_head_batch(engine, &input, 0, &scores, NULL, NULL) == -1);
    assert(coli_v4_gpu_head_batch(engine, &input, 129, &scores, NULL, NULL) == -1);
    assert(head_batch_calls == 3);
    head_scores_ok = 0;
    assert(coli_v4_gpu_head_scores(engine, &input, &scores) == -1);
    assert(coli_v4_gpu_head_batch(engine, &input, 3, NULL, ids, values) == -1);
    int before = shutdowns;
    coli_v4_gpu_engine_close(engine);
    assert(!live_tensors && !engine->gpu.head && !engine->gpu.uploaded_bytes && shutdowns == before + 1);
    assert(coli_v4_gpu_head_scores(engine, &input, &scores) == -1);
    assert(coli_v4_gpu_head_batch(engine, &input, 3, NULL, ids, values) == -1);
    unsetenv("DSV4_CUDA_HEAD_BATCH");
    coli_v4_gpu_engine_close(engine); assert(shutdowns == before + 1);
    free(engine);
#endif
}

static int wo_calls, wo_ok = 1;
int dsv4_cuda_wo_decode(Dsv4CudaTensor *a, Dsv4CudaTensor *b, int groups,
                        float *output, const float *input) {
    assert(a && b && groups == 8);
    wo_calls++;
    if (wo_ok) *output = *input;
    return wo_ok;
}
static void test_wo_decode(void) {
    Dsv4CudaTensor a = {0}, b = {0};
    ColiTensorView wa = {.gpu = &a}, wb = {.gpu = &b};
    float input = 7, output = -1;
    unsetenv("DSV4_CUDA_WO_DECODE");
    assert(!coli_v4_gpu_wo_decode(&wa, &wb, &output, &input, 8));
    assert(wo_calls == 1 && output == input);
    setenv("DSV4_CUDA_WO_DECODE", "0", 1);
    assert(coli_v4_gpu_wo_decode(&wa, &wb, &output, &input, 8) == -1);
    assert(wo_calls == 1);
    setenv("DSV4_CUDA_WO_DECODE", "1", 1);
    wo_ok = 0;
    assert(coli_v4_gpu_wo_decode(&wa, &wb, &output, &input, 8) == -1);
    assert(wo_calls == 2);
    wa.gpu = NULL;
    assert(coli_v4_gpu_wo_decode(&wa, &wb, &output, &input, 8) == -1);
    wa.gpu = &a; wb.gpu = NULL;
    assert(coli_v4_gpu_wo_decode(&wa, &wb, &output, &input, 8) == -1);
    assert(wo_calls == 2);
    unsetenv("DSV4_CUDA_WO_DECODE");
}

static int moe_calls, moe_ok = 1;
static V4GpuExpertMirrorCache *locked_cache;
int dsv4_cuda_moe(Dsv4CudaTensor *const *gate, Dsv4CudaTensor *const *up,
    Dsv4CudaTensor *const *down, const float *weights, int count,
    Dsv4CudaTensor *sg, Dsv4CudaTensor *su, Dsv4CudaTensor *sd,
    float limit, float *y, const float *x) {
    (void)limit;
    assert(count == 2 && weights[0] == 0.25f && weights[1] == 0.75f);
    assert(sg == su && su == sd && sg->device == gate[0]->device);
    assert(pthread_mutex_trylock(&locked_cache->mutex) != 0);
    for (int i = 0; i < count; i++) {
        assert(gate[i]->device == sg->device);
        assert(up[i]->device == sg->device && down[i]->device == sg->device);
    }
    moe_calls++;
    if (moe_ok) y[0] = x[0];
    return moe_ok;
}

static void check_resident(ColiExpertStore *store, int layer, int device) {
    locked_cache = v4_gpu_expert_cache(store, layer);
    struct Dsv4CudaTensor shared = {device};
    int ids[] = {0, 1};
    float weights[] = {0.25f, 0.75f}, input = 7, output = -1;
    int before = moe_calls;
    assert(coli_v4_gpu_moe_resident(store, layer, ids, weights, 2,
        &shared, &shared, &shared, 0, &output, &input) == 1);
    assert(output == input && moe_calls == before + 1);
    ids[1] = 255;
    output = -1;
    assert(coli_v4_gpu_moe_resident(store, layer, ids, weights, 2,
        &shared, &shared, &shared, 0, &output, &input) == 0);
    assert(output == -1 && moe_calls == before + 1);
    ids[1] = 1;
    assert(coli_v4_gpu_moe_resident(store, layer, ids, weights, 2,
        NULL, &shared, &shared, 0, &output, &input) == 0);
    V4GpuExpertMirror *incomplete = NULL;
    for (int i = 0; i < locked_cache->count; i++)
        if (locked_cache->entries[i].layer == layer && locked_cache->entries[i].expert == 1)
            incomplete = &locked_cache->entries[i];
    assert(incomplete);
    Dsv4CudaTensor *saved = incomplete->down;
    incomplete->down = NULL;
    assert(coli_v4_gpu_moe_resident(store, layer, ids, weights, 2,
        &shared, &shared, &shared, 0, &output, &input) == 0);
    incomplete->down = saved;
    assert(moe_calls == before + 1);
    moe_ok = 0;
    assert(coli_v4_gpu_moe_resident(store, layer, ids, weights, 2,
        &shared, &shared, &shared, 0, &output, &input) == 0);
    assert(output == -1 && moe_calls == before + 2);
    moe_ok = 1;
    assert(pthread_mutex_trylock(&locked_cache->mutex) == 0);
    pthread_mutex_unlock(&locked_cache->mutex);
}

static int lookups, releases, active_leases, lookup_fail_after = -1;
static int mock_lookup(ColiExpertStore *store, ColiExpertKey key, ColiExpertView *view) {
    (void)store;
    if (lookup_fail_after == 0) return -1;
    if (lookup_fail_after > 0) lookup_fail_after--;
    static uint8_t bytes[512];
    memset(view, 0, sizeof(*view));
    view->key = key;
    view->lease = bytes;
    view->gate.data = view->up.data = view->down.data = bytes;
    view->gate.scales = view->up.scales = view->down.scales = bytes;
    view->gate.rows = view->up.rows = view->down.rows = 32;
    view->gate.columns = view->up.columns = view->down.columns = 32;
    view->gate.block_rows = view->up.block_rows = view->down.block_rows = 1;
    lookups++; active_leases++;
    return 0;
}
static void mock_release(ColiExpertStore *store, ColiExpertView *view) {
    (void)store;
    assert(view->lease && active_leases == 1);
    releases++; active_leases--;
}
int coli_v4_layer_load(ColiV4Engine *engine, ColiDeepSeekV4LayerWeights *weights,
    const ColiDeepSeekV4Config *config, const ColiSafetensorsIndex *index,
    int layer, char *error, size_t size) {
    (void)engine; (void)config; (void)index; (void)error; (void)size;
    memset(weights, 0, sizeof(*weights));
    weights->plan.layer = layer;
    return 0;
}
static int shared_missing;
void *coli_v4_layer_gpu(const ColiDeepSeekV4LayerWeights *weights, const char *prefix) {
    (void)weights; (void)prefix;
    static Dsv4CudaTensor shared;
    return shared_missing ? NULL : &shared;
}
static void test_preload(void) {
    ColiV4Engine *engine = calloc(1, sizeof(*engine));
    const ColiExpertStoreOps ops = {.lookup = mock_lookup, .release = mock_release};
    ColiExpertStore store = {.ops = &ops};
    engine->experts = &store;
    engine->runtime.dense_resident = 1;
    engine->config.num_hidden_layers = 6;
    engine->config.n_routed_experts = 2;
    engine->config.hidden_size = engine->config.moe_intermediate_size = 32;
    engine->gpu.enabled = 1;
    engine->gpu.device_count = 2;
    engine->gpu.devices[0] = 5; engine->gpu.devices[1] = 3;
    store.gpu = v4_gpu_expert_mirrors_create_devices(engine->gpu.devices, 2, 6);
    void *original = store.gpu;
    char error[256];
    shared_missing = 1;
    unsetenv("DSV4_CUDA_RESIDENT_EXPERTS");
    assert(coli_v4_gpu_experts_init(engine, error, sizeof(error)) == 0);
    assert(store.gpu == original && !live_tensors && !error[0]);
    setenv("DSV4_CUDA_RESIDENT_EXPERTS", "1", 1);
    assert(coli_v4_gpu_experts_init(engine, error, sizeof(error)) == -1);
    assert(store.gpu == original && !live_tensors && error[0]);
    setenv("DSV4_CUDA_RESIDENT_EXPERTS", "0", 1);
    int calls = lookups;
    assert(coli_v4_gpu_experts_init(engine, error, sizeof(error)) == 0);
    assert(lookups == calls && store.gpu == original);
    unsetenv("DSV4_CUDA_RESIDENT_EXPERTS");
    shared_missing = 0;
    low_memory_device = 3;
    assert(coli_v4_gpu_experts_preload(engine, error, sizeof(error)) == 1);
    assert(store.gpu == original && lookups == 0 && !live_tensors);
    low_memory_device = -1;
    upload_fail_after = 20; /* fail on the second device after publishing none */
    assert(coli_v4_gpu_experts_preload(engine, error, sizeof(error)) == -1);
    assert(store.gpu == original && !active_leases && !live_tensors && releases == lookups);
    setenv("DSV4_CUDA_RESIDENT_EXPERTS", "auto", 1);
    upload_fail_after = 20;
    assert(coli_v4_gpu_experts_init(engine, error, sizeof(error)) == 0);
    assert(store.gpu == original && !live_tensors && !error[0]);
    setenv("DSV4_CUDA_RESIDENT_EXPERTS", "1", 1);
    upload_fail_after = 20;
    assert(coli_v4_gpu_experts_init(engine, error, sizeof(error)) == -1);
    assert(store.gpu == original && !live_tensors && error[0]);
    unsetenv("DSV4_CUDA_RESIDENT_EXPERTS");
    upload_fail_after = 20;
    assert(coli_v4_gpu_experts_init(engine, error, sizeof(error)) == 0);
    assert(store.gpu == original && !live_tensors && !error[0]);
    upload_fail_after = -1;
    lookup_fail_after = 7;
    assert(coli_v4_gpu_experts_preload(engine, error, sizeof(error)) == -1);
    assert(store.gpu == original && !active_leases && !live_tensors && releases == lookups);
    lookup_fail_after = -1;
    int before = lookups;
    assert(coli_v4_gpu_experts_preload(engine, error, sizeof(error)) == 0);
    assert(store.gpu != original && lookups == before + 12 && releases == lookups);
    assert(live_tensors == 36 && !active_leases);
    for (int layer = 0; layer < 6; layer++) {
        locked_cache = v4_gpu_expert_cache(&store, layer);
        assert(coli_v4_gpu_experts_resident(&store, layer));
        assert(v4_gpu_expert_find(locked_cache, layer, -1) == NULL);
        assert(v4_gpu_expert_find(locked_cache, layer, 2) == NULL);
        assert(v4_gpu_expert_find(locked_cache, 6, 0) == NULL);
        for (int expert = 0; expert < 2; expert++) {
            V4GpuExpertMirror *entry = v4_gpu_expert_find(locked_cache, layer, expert);
            assert(entry->layer == layer && entry->expert == expert);
            assert(entry->gate->device == (layer < 3 ? 5 : 3));
        }
        int ids[] = {1, 0}; float w[] = {.25f, .75f}, x = 7, y = 0;
        Dsv4CudaTensor shared = {layer < 3 ? 5 : 3};
        assert(coli_v4_gpu_moe_resident(&store, layer, ids, w, 2,
            &shared, &shared, &shared, 0, &y, &x) == 1 && y == x);
        moe_ok = 0;
        assert(coli_v4_gpu_moe_resident(&store, layer, ids, w, 2,
            &shared, &shared, &shared, 0, &y, &x) == -1);
        moe_ok = 1;
        ids[1] = 2;
        assert(coli_v4_gpu_moe_resident(&store, layer, ids, w, 2,
            &shared, &shared, &shared, 0, &y, &x) == -1);
    }
    assert(lookups == before + 12); /* compute never reacquired host weights */
    v4_gpu_expert_mirrors_free(store.gpu);
    assert(!live_tensors);
    free(engine);
}

Dsv4CudaExpertSet *dsv4_cuda_expert_set_create(Dsv4CudaTensor *const *g,
    Dsv4CudaTensor *const *u,Dsv4CudaTensor *const *d,int count,
    Dsv4CudaTensor *sg,Dsv4CudaTensor *su,Dsv4CudaTensor *sd) {
    (void)sg; (void)su; (void)sd;
    assert(count == 256);
    for (int i=0;i<count;i++) assert(g[i] && u[i] && d[i]);
    Dsv4CudaExpertSet *set = malloc(sizeof(*set));
    assert(set); set->device = g[0]->device; sets_created++;
    return set;
}
static int bad_hash, route_ok=1, route_calls;
const void *coli_v4_layer_data(const ColiDeepSeekV4LayerWeights *weights,
    const char *name, const ColiDeepSeekV4TensorSpec **spec) {
    (void)weights; (void)name; (void)spec;
    static int64_t map[12];
    for(int i=0;i<12;i++)map[i]=i%6;
    if(bad_hash)map[0]=256;
    return map;
}
int dsv4_cuda_resident_route_moe(Dsv4CudaExpertSet *set,Dsv4CudaTensor *gate,
    Dsv4CudaTensor *bias,const int *fixed,float scale,float limit,float *out,const float *in) {
    (void)gate; (void)bias; (void)scale; (void)limit;
    assert(set->device==0);
    if(fixed)for(int k=0;k<6;k++)assert(fixed[k]==k);
    route_calls++;
    if(route_ok)*out=*in;
    return route_ok;
}
int dsv4_cuda_resident_route_moe_batch(Dsv4CudaExpertSet *set,Dsv4CudaTensor *gate,
    Dsv4CudaTensor *bias,const int *fixed,float scale,float limit,float *out,const float *in,int tokens) {
    int ok = 1;
    for (int t = 0; t < tokens; t++)
        ok &= dsv4_cuda_resident_route_moe(set,gate,bias,fixed ? fixed + t*6 : NULL,
                                          scale,limit,out+t,in+t);
    return ok;
}
static void test_resident_route(void) {
    V4GpuExpertMirrorCache *cache=v4_gpu_expert_mirrors_create_capacity(0,256);
    cache->first_layer=7; cache->end_layer=8; cache->experts_per_layer=256;
    ColiExpertStore store={.gpu=cache};
    ColiDeepSeekV4LayerWeights weights={0}; weights.plan.layer=7;
    ColiDeepSeekV4Config config={0}; config.num_experts_per_tok=6; config.vocab_size=2;
    for(int e=0;e<256;e++) {
        V4GpuExpertMirror *entry=&cache->entries[cache->count++];
        entry->layer=7; entry->expert=e;
        entry->gate=calloc(1,sizeof(Dsv4CudaTensor));
        entry->up=calloc(1,sizeof(Dsv4CudaTensor));
        entry->down=calloc(1,sizeof(Dsv4CudaTensor));
        assert(entry->gate&&entry->up&&entry->down); live_tensors+=3;
    }
    float in=7,out=0;
    setenv("DSV4_CUDA_RESIDENT_ROUTE","0",1);
    assert(!coli_v4_gpu_resident_route(&out,&weights,&config,&store,&in,0));
    assert(!sets_created && !route_calls);
    unsetenv("DSV4_CUDA_RESIDENT_ROUTE"); /* resident routing defaults on */
    assert(coli_v4_gpu_resident_route(&out,&weights,&config,&store,&in,0)==1 && out==in);
    weights.plan.uses_hash_router=1;
    assert(coli_v4_gpu_resident_route(&out,&weights,&config,&store,&in,1)==1);
    assert(sets_created==1 && route_calls==2);
    bad_hash=1;
    assert(coli_v4_gpu_resident_route(&out,&weights,&config,&store,&in,0)==-1);
    bad_hash=0;
    assert(coli_v4_gpu_resident_route(&out,&weights,&config,&store,&in,2)==-1);
    int batch_tokens[2] = {0, 1};
    float batch_input[2] = {3, 7}, batch_output[2] = {0};
    assert(v4_gpu_resident_route_batch(batch_output, &weights, &config, &store,
                                      batch_input, batch_tokens, 2) == 1);
    assert(batch_output[0] == 3 && batch_output[1] == 7);
    route_ok=0;
    assert(coli_v4_gpu_resident_route(&out,&weights,&config,&store,&in,0)==0);
    v4_gpu_expert_mirrors_free(cache);
    assert(sets_created==sets_freed && !live_tensors);
    unsetenv("DSV4_CUDA_RESIDENT_ROUTE");
}

static unsigned char expected_dense[8 * 128];
static int dense_uploads, dense_fail;
int dsv4_cuda_upload_fp8(Dsv4CudaTensor **t, const uint8_t *w, const uint8_t *scale,
                         int rows, int cols, int device) {
    assert(rows == 8 && cols == 128 && scale[0] == 125);
    assert(!memcmp(w, expected_dense, sizeof(expected_dense)));
    dense_uploads++;
    if (dense_fail) return 0;
    *t = malloc(sizeof(**t)); assert(*t); (*t)->device = device; live_tensors++;
    return 1;
}
static void test_draft_cache(void) {
    ColiV4Engine *engine = calloc(1, sizeof(*engine)); assert(engine);
    engine->gpu.enabled = 1; engine->gpu.device_count = 6;
    for (int i = 0; i < 6; i++) engine->gpu.devices[i] = i;
    unsigned char packed[8 * 128], copy[8 * 128];
    for (int r = 0; r < 8; r++) for (int c = 0; c < 128; c++) {
        expected_dense[r * 128 + c] = (r * 17 + c) & 255;
        packed[c * 8 + r] = expected_dense[r * 128 + c];
    }
    memcpy(copy, packed, sizeof(copy));
    float scale = .25f;
    ColiTensorView view = {.format=COLI_TENSOR_FP8_E4M3_BLOCK, .scale_format=COLI_SCALE_F32,
        .data=packed, .scales=&scale, .rows=8, .columns=128, .block_rows=8};
    assert(!coli_v4_gpu_dspark_dense_attach(engine, &view));
    assert(view.gpu && dense_uploads == 1 && live_tensors == 1);
    void *first = view.gpu;
    view.gpu = NULL;
    assert(!coli_v4_gpu_dspark_dense_attach(engine, &view) && view.gpu == first);
    assert(dense_uploads == 1);
    view.data = copy; view.gpu = NULL; dense_fail = 1;
    assert(coli_v4_gpu_dspark_dense_attach(engine, &view) < 0 && !view.gpu);
    assert(coli_v4_gpu_dspark_dense_attach(engine, &view) < 0 && dense_uploads == 2);
    dense_fail = 0;
    setenv("V4_MTP_GPU_DENSE", "1", 1); setenv("V4_MTP_GPU_MIRRORS", "25", 1);
    assert(!coli_v4_gpu_dspark_mirrors_ensure(engine));
    V4GpuExpertMirrorCache *cache = engine->gpu.dspark_mirrors;
    for (int stage = 0; stage < 3; stage++, cache = cache->next) {
        assert(cache && cache->device == stage + 1 && cache->first_layer == stage && cache->end_layer == stage + 1);
        assert(cache->capacity == (stage == 0 ? 9 : 8));
    }
    assert(!cache);
    coli_v4_gpu_engine_close(engine);
    assert(!live_tensors && !engine->gpu.dspark_dense && !engine->gpu.dspark_mirrors);
    free(engine); unsetenv("V4_MTP_GPU_DENSE"); unsetenv("V4_MTP_GPU_MIRRORS");
}

int main(void) {
    ColiV4Engine *engine = calloc(1, sizeof(*engine));
    ColiExpertStore store = {0};
    engine->config.num_hidden_layers = 43;
    engine->experts = &store;
    setenv("DSV4_CUDA", "1", 1);
    setenv("DSV4_CUDA_DEVICE", "0", 1);
    setenv("DSV4_CUDA_DEVICES", "5,3,1,4,2,0", 1);
    setenv("DSV4_CUDA_EXPERT_MIRRORS", "8", 1);
    assert(coli_v4_gpu_engine_open(engine) == 0);
    assert(initialized == 6 && engine->gpu.device == 5);
    assert(v4_gpu_expert_cache(&store, -1) == NULL);
    assert(v4_gpu_expert_cache(&store, 43) == NULL);
    uint8_t data[4] = {0};
    for (int layer = 0; layer < 43; layer++) {
        int owner = coli_v4_gpu_layer_owner(layer, 6, 43);
        for (int expert = 0; expert < 2; expert++) {
            ColiExpertView view = {0};
            view.key = (ColiExpertKey){layer, expert};
            view.gate.data = view.up.data = view.down.data = data;
            view.gate.scales = view.up.scales = view.down.scales = data;
            view.gate.rows = view.up.rows = view.down.rows = 2;
            view.gate.columns = view.up.columns = view.down.columns = 2;
            view.gate.block_rows = view.up.block_rows = view.down.block_rows = 1;
            assert(coli_v4_gpu_expert_peek(&store, &view) == -1);
            assert(coli_v4_gpu_expert_attach_async(&store, &view) == 0);
            assert(((Dsv4CudaTensor *)view.gate.gpu)->device == engine->gpu.devices[owner]);
            assert(coli_v4_gpu_expert_peek(&store, &view) == 0);
        }
        check_resident(&store, layer, engine->gpu.devices[owner]);
    }
    assert(coli_v4_gpu_expert_drain(&store) == -1);
    for (int i = 0; i < 6; i++) {
        assert(free_queries[i] > 0);
        assert(drained[i] == 1); /* drain all devices even after one fails */
    }
    coli_v4_gpu_engine_close(engine);
    assert(!live_tensors && !store.gpu && shutdowns == 1);
    setenv("DSV4_CUDA_DEVICES", "0,0", 1);
    assert(coli_v4_gpu_engine_open(engine) == -1);
    setenv("DSV4_CUDA_DEVICES", "0,99", 1);
    assert(coli_v4_gpu_engine_open(engine) == -1);
    assert(!store.gpu && shutdowns == 1);
    unsetenv("DSV4_CUDA_DEVICES");
    setenv("DSV4_CUDA_DEVICE", "not-a-number", 1);
    assert(coli_v4_gpu_engine_open(engine) == 0);
    assert(engine->gpu.enabled && engine->gpu.device == 0);
    coli_v4_gpu_engine_close(engine);
    setenv("DSV4_CUDA_DEVICE", "99", 1);
    assert(coli_v4_gpu_engine_open(engine) == 0);
    assert(!engine->gpu.enabled && !store.gpu);
    unsetenv("DSV4_CUDA_DEVICE");
    free(engine);
    test_preload();
    test_resident_route();
    test_wo_decode();
    test_head();
    test_draft_cache();
    test_batch_scratch();
    puts("test_v4_gpu_placement: ok");
    return 0;
}
