/* Exercise the production GPU unit with a tiny backend, no CUDA toolkit. */
#include <time.h>
#include <pthread.h>
#define COLI_V4_GPU_TIER
#define COLI_V4_UNIT_GPU
#include "../deepseek_v4.c"
#include <assert.h>

struct Dsv4CudaTensor { int device; };
static int initialized, shutdowns, live_tensors, free_queries[16], drained[16];
int dsv4_cuda_backend_arch_ok(int device) { return device >= 0 && device < 6; }
const char *dsv4_cuda_backend_name(void) { return "test"; }
int dsv4_cuda_init(const int *devices, int count) {
    for (int i = 0; i < count; i++)
        if (devices[i] < 0 || devices[i] >= 6) return 0;
    initialized = count;
    assert(devices[0] == 5);
    return 1;
}
void dsv4_cuda_shutdown(void) { shutdowns++; }
int dsv4_cuda_device_unified(int device) { (void)device; return 0; }
long long dsv4_cuda_mem_free_mb(int device) { free_queries[device]++; return 32000; }
int dsv4_cuda_stream_drain(int device) { drained[device]++; return device != 3; }
void dsv4_cuda_tensor_free(Dsv4CudaTensor *t) { if (t) { live_tensors--; free(t); } }
int dsv4_cuda_upload_fp4(Dsv4CudaTensor **t, const uint8_t *w,
                          const uint8_t *scale, int rows, int cols, int device) {
    (void)w; (void)scale; (void)rows; (void)cols;
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
    free(engine);
    puts("test_v4_gpu_placement: ok");
    return 0;
}
