#include "backend_cuda.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

struct ColiCudaTensor {
    void *weights;
    float *scales;
    size_t weight_bytes;
    int fmt, I, O, device;
    int tracked;
};

typedef struct {
    int device;
    float *x, *y;
    size_t x_cap, y_cap;
    size_t tensor_count, tensor_bytes;
    /* Fase 2: pool FFN fusa — uno stream per device, staging host pinned e buffer
     * device dimensionati alla begin (mai riallocati con lavoro in volo). dm0/dm1
     * (intermedi gate/up) sono condivisi tra gli expert del blocco: lo stream
     * serializza i kernel, quindi il riuso a offset 0 e' sicuro. */
    cudaStream_t stream;
    float *hx, *hy;                 /* host pinned: input staging / output */
    size_t hx_cap, hy_cap;
    float *dx, *dy, *dm0, *dm1;     /* device: input, output, intermedi [rows,I] */
    size_t dx_cap, dy_cap, dm0_cap, dm1_cap;
    long long ffn_off;              /* righe gia' impegnate nel blocco corrente */
    int ffn_D, ffn_I, ffn_err, ffn_open;
} DeviceContext;

static DeviceContext g_ctx[COLI_CUDA_MAX_DEVICES];
static int g_nctx;

static int cuda_ok(cudaError_t err, const char *what) {
    if (err == cudaSuccess) return 1;
    std::fprintf(stderr, "[CUDA] %s: %s\n", what, cudaGetErrorString(err));
    return 0;
}

static DeviceContext *find_ctx(int device) {
    for (int i = 0; i < g_nctx; i++) if (g_ctx[i].device == device) return &g_ctx[i];
    return nullptr;
}

static int select_ctx(DeviceContext *ctx) {
    return ctx && cuda_ok(cudaSetDevice(ctx->device), "select device");
}

static size_t row_bytes(int fmt, int I) {
    if (fmt == 0) return (size_t)I * sizeof(float);
    if (fmt == 1) return (size_t)I;
    if (fmt == 2) return (size_t)(I + 1) / 2;
    if (fmt == 3) return (size_t)(I + 3) / 4;
    return 0;
}

__device__ static float weight_at(const void *weights, int fmt, size_t row, int i) {
    const uint8_t *base = static_cast<const uint8_t *>(weights) + row;
    if (fmt == 0) return reinterpret_cast<const float *>(base)[i];
    if (fmt == 1) return static_cast<float>(reinterpret_cast<const int8_t *>(base)[i]);
    const uint8_t *q = base;
    if (fmt == 2) {
        uint8_t v = q[i >> 1];
        return static_cast<float>(((i & 1) ? (v >> 4) : (v & 15)) - 8);
    }
    uint8_t v = q[i >> 2];
    return static_cast<float>(((v >> ((i & 3) * 2)) & 3) - 2);
}

__global__ static void quant_matmul(float *y, const float *x, const void *weights,
                                    const float *scales, int fmt, int S, int I, int O,
                                    size_t rb) {
    int o = blockIdx.x;
    int s = blockIdx.y;
    float sum = 0.0f;
    size_t row = (size_t)o * rb;
    const float *xs = x + (size_t)s * I;
    for (int i = threadIdx.x; i < I; i += blockDim.x)
        sum += xs[i] * weight_at(weights, fmt, row, i);

    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int n = blockDim.x >> 1; n; n >>= 1) {
        if (threadIdx.x < n) partial[threadIdx.x] += partial[threadIdx.x + n];
        __syncthreads();
    }
    if (!threadIdx.x)
        y[(size_t)s * O + o] = partial[0] * (fmt ? scales[o] : 1.0f);
}

static int reserve(float **ptr, size_t *cap, size_t bytes) {
    if (*cap >= bytes) return 1;
    if (*ptr) cudaFree(*ptr);
    *ptr = nullptr;
    *cap = 0;
    if (!cuda_ok(cudaMalloc(ptr, bytes), "scratch allocation")) return 0;
    *cap = bytes;
    return 1;
}

static int reserve_pinned(float **ptr, size_t *cap, size_t bytes) {
    if (*cap >= bytes) return 1;
    if (*ptr) cudaFreeHost(*ptr);
    *ptr = nullptr;
    *cap = 0;
    if (!cuda_ok(cudaHostAlloc(ptr, bytes, cudaHostAllocDefault), "pinned allocation")) return 0;
    *cap = bytes;
    return 1;
}

__global__ static void silu_mul(float *g, const float *u, long long n) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i < n) { float v = g[i]; g[i] = v / (1.0f + expf(-v)) * u[i]; }
}

extern "C" int coli_cuda_ffn_begin(int device, long long rows, int D, int I) {
    DeviceContext *ctx = find_ctx(device);
    if (rows < 1 || D < 1 || I < 1 || !select_ctx(ctx)) return 0;
    if (!ctx->stream && !cuda_ok(cudaStreamCreate(&ctx->stream), "stream create")) return 0;
    /* gli out del blocco precedente puntano in hy: sincronizza PRIMA di riallocare */
    if (!cuda_ok(cudaStreamSynchronize(ctx->stream), "ffn begin sync")) return 0;
    size_t xb = (size_t)rows * D * sizeof(float), mb = (size_t)rows * I * sizeof(float);
    if (!reserve_pinned(&ctx->hx, &ctx->hx_cap, xb) || !reserve_pinned(&ctx->hy, &ctx->hy_cap, xb) ||
        !reserve(&ctx->dx, &ctx->dx_cap, xb) || !reserve(&ctx->dy, &ctx->dy_cap, xb) ||
        !reserve(&ctx->dm0, &ctx->dm0_cap, mb) || !reserve(&ctx->dm1, &ctx->dm1_cap, mb)) return 0;
    ctx->ffn_off = 0; ctx->ffn_D = D; ctx->ffn_I = I;
    ctx->ffn_err = 0; ctx->ffn_open = 1;
    return 1;
}

extern "C" int coli_cuda_ffn_enqueue(ColiCudaTensor *gate, ColiCudaTensor *up, ColiCudaTensor *down,
                                     const float *x, int nr, const float **out_host) {
    if (!gate || !up || !down || !x || !out_host || nr < 1) return 0;
    if (gate->device != up->device || up->device != down->device) return 0;
    DeviceContext *ctx = find_ctx(gate->device);
    if (!ctx || !ctx->ffn_open) return 0;
    int D = ctx->ffn_D, I = ctx->ffn_I;
    if (gate->I != D || gate->O != I || up->I != D || up->O != I || down->I != I || down->O != D) return 0;
    size_t need = ((size_t)ctx->ffn_off + nr) * D * sizeof(float);
    if (need > ctx->hx_cap || need > ctx->hy_cap) return 0;   /* begin sottodimensionata */
    if (!select_ctx(ctx)) return 0;
    long long off = ctx->ffn_off;
    float *hx = ctx->hx + off * D, *hy = ctx->hy + off * D;
    float *dx = ctx->dx + off * D, *dy = ctx->dy + off * D;
    size_t xb = (size_t)nr * D * sizeof(float);
    memcpy(hx, x, xb);                            /* il chiamante riusa x subito dopo */
    cudaMemcpyAsync(dx, hx, xb, cudaMemcpyHostToDevice, ctx->stream);
    quant_matmul<<<dim3((unsigned)I, (unsigned)nr), 256, 0, ctx->stream>>>(
        ctx->dm0, dx, gate->weights, gate->scales, gate->fmt, nr, D, I, row_bytes(gate->fmt, D));
    quant_matmul<<<dim3((unsigned)I, (unsigned)nr), 256, 0, ctx->stream>>>(
        ctx->dm1, dx, up->weights, up->scales, up->fmt, nr, D, I, row_bytes(up->fmt, D));
    long long n = (long long)nr * I;
    silu_mul<<<(unsigned)((n + 255) / 256), 256, 0, ctx->stream>>>(ctx->dm0, ctx->dm1, n);
    quant_matmul<<<dim3((unsigned)D, (unsigned)nr), 256, 0, ctx->stream>>>(
        dy, ctx->dm0, down->weights, down->scales, down->fmt, nr, I, D, row_bytes(down->fmt, I));
    cudaMemcpyAsync(hy, dy, xb, cudaMemcpyDeviceToHost, ctx->stream);
    if (cudaGetLastError() != cudaSuccess) ctx->ffn_err = 1;
    *out_host = hy;
    ctx->ffn_off += nr;
    return 1;
}

extern "C" int coli_cuda_ffn_sync(int device) {
    DeviceContext *ctx = find_ctx(device);
    if (!ctx || !ctx->ffn_open || !select_ctx(ctx)) return 0;
    ctx->ffn_open = 0;
    if (!cuda_ok(cudaStreamSynchronize(ctx->stream), "ffn sync")) return 0;
    return !ctx->ffn_err;
}

extern "C" int coli_cuda_init(const int *devices, int count) {
    int available = 0;
    if (!devices || count < 1 || count > COLI_CUDA_MAX_DEVICES) return 0;
    if (!cuda_ok(cudaGetDeviceCount(&available), "device discovery")) return 0;
    g_nctx = 0;
    for (int i = 0; i < count; i++) {
        int device = devices[i];
        if (device < 0 || device >= available) {
            std::fprintf(stderr, "[CUDA] invalid device %d (available: 0..%d)\n", device, available - 1);
            g_nctx = 0;
            return 0;
        }
        if (find_ctx(device)) {
            std::fprintf(stderr, "[CUDA] duplicate device %d\n", device);
            g_nctx = 0;
            return 0;
        }
        DeviceContext *ctx = &g_ctx[g_nctx];
        *ctx = {};
        ctx->device = device;
        if (!select_ctx(ctx)) { g_nctx = 0; return 0; }
        cudaDeviceProp prop{};
        if (!cuda_ok(cudaGetDeviceProperties(&prop, device), "device properties")) { g_nctx = 0; return 0; }
        g_nctx++;
        std::fprintf(stderr, "[CUDA] device %d: %s, %.1f GB VRAM, sm_%d%d\n",
                     device, prop.name, prop.totalGlobalMem / 1e9, prop.major, prop.minor);
    }
    return 1;
}

extern "C" void coli_cuda_shutdown(void) {
    for (int i = 0; i < g_nctx; i++) {
        DeviceContext *ctx = &g_ctx[i];
        if (!select_ctx(ctx)) continue;
        if (ctx->stream) { cudaStreamSynchronize(ctx->stream); cudaStreamDestroy(ctx->stream); }
        if (ctx->x) cudaFree(ctx->x);
        if (ctx->y) cudaFree(ctx->y);
        if (ctx->dx) cudaFree(ctx->dx);
        if (ctx->dy) cudaFree(ctx->dy);
        if (ctx->dm0) cudaFree(ctx->dm0);
        if (ctx->dm1) cudaFree(ctx->dm1);
        if (ctx->hx) cudaFreeHost(ctx->hx);
        if (ctx->hy) cudaFreeHost(ctx->hy);
        *ctx = {};
    }
    g_nctx = 0;
}

extern "C" int coli_cuda_device_count(void) { return g_nctx; }

extern "C" int coli_cuda_device_at(int index) {
    return index >= 0 && index < g_nctx ? g_ctx[index].device : -1;
}

extern "C" int coli_cuda_mem_info(int device, size_t *free_bytes, size_t *total_bytes) {
    DeviceContext *ctx = find_ctx(device);
    if (!free_bytes || !total_bytes || !select_ctx(ctx)) return 0;
    return cuda_ok(cudaMemGetInfo(free_bytes, total_bytes), "memory info");
}

extern "C" void coli_cuda_stats(int device, size_t *tensor_count, size_t *tensor_bytes) {
    size_t count = 0, bytes = 0;
    for (int i = 0; i < g_nctx; i++) if (device < 0 || g_ctx[i].device == device) {
        count += g_ctx[i].tensor_count;
        bytes += g_ctx[i].tensor_bytes;
    }
    if (tensor_count) *tensor_count = count;
    if (tensor_bytes) *tensor_bytes = bytes;
}

extern "C" int coli_cuda_tensor_upload(ColiCudaTensor **tensor,
                                        const void *weights, const float *scales,
                                        int fmt, int I, int O, int device) {
    DeviceContext *ctx = find_ctx(device);
    if (!tensor || I < 1 || O < 1 || !select_ctx(ctx)) return 0;
    size_t rb = row_bytes(fmt, I);
    if (!rb) return 0;
    if (*tensor) {
        /* Riuso di una copia gia' residente: i dati host possono essere NULL
         * (tier VRAM senza backing RAM — la copia host e' stata liberata). */
        ColiCudaTensor *t = *tensor;
        return t->fmt == fmt && t->I == I && t->O == O && t->device == device;
    }
    if (!weights || (fmt && !scales)) return 0;  /* il PRIMO upload richiede i dati host */
    ColiCudaTensor *t = static_cast<ColiCudaTensor *>(std::calloc(1, sizeof(*t)));
    if (!t) return 0;
    t->fmt = fmt; t->I = I; t->O = O; t->device = device; t->weight_bytes = rb * (size_t)O;
    if (!cuda_ok(cudaMalloc(&t->weights, t->weight_bytes), "tensor allocation") ||
        !cuda_ok(cudaMemcpy(t->weights, weights, t->weight_bytes, cudaMemcpyHostToDevice), "tensor upload")) {
        coli_cuda_tensor_free(t);
        return 0;
    }
    if (fmt) {
        if (!cuda_ok(cudaMalloc(&t->scales, (size_t)O * sizeof(float)), "scale allocation") ||
            !cuda_ok(cudaMemcpy(t->scales, scales, (size_t)O * sizeof(float), cudaMemcpyHostToDevice), "scale upload")) {
            coli_cuda_tensor_free(t);
            return 0;
        }
    }
    t->tracked = 1;
    ctx->tensor_count++;
    ctx->tensor_bytes += t->weight_bytes + (fmt ? (size_t)O * sizeof(float) : 0);
    *tensor = t;
    return 1;
}

extern "C" int coli_cuda_matmul(ColiCudaTensor **tensor,
                                 float *y, const float *x,
                                 const void *weights, const float *scales,
                                 int fmt, int S, int I, int O, int device) {
    if (S < 1 || !coli_cuda_tensor_upload(tensor, weights, scales, fmt, I, O, device)) return 0;
    ColiCudaTensor *t = *tensor;
    DeviceContext *ctx = find_ctx(t->device);
    if (!select_ctx(ctx)) return 0;
    size_t rb = row_bytes(fmt, I);
    size_t xb = (size_t)S * I * sizeof(float), yb = (size_t)S * O * sizeof(float);
    if (!reserve(&ctx->x, &ctx->x_cap, xb) || !reserve(&ctx->y, &ctx->y_cap, yb)) return 0;
    if (!cuda_ok(cudaMemcpy(ctx->x, x, xb, cudaMemcpyHostToDevice), "input upload")) return 0;
    dim3 grid((unsigned)O, (unsigned)S);
    quant_matmul<<<grid, 256>>>(ctx->y, ctx->x, t->weights, t->scales, fmt, S, I, O, rb);
    if (!cuda_ok(cudaGetLastError(), "matmul launch") ||
        !cuda_ok(cudaMemcpy(y, ctx->y, yb, cudaMemcpyDeviceToHost), "output download")) return 0;
    return 1;
}

extern "C" void coli_cuda_tensor_free(ColiCudaTensor *tensor) {
    if (!tensor) return;
    DeviceContext *ctx = find_ctx(tensor->device);
    if (ctx) select_ctx(ctx);
    if (tensor->tracked && ctx) {
        size_t bytes = tensor->weight_bytes + (tensor->fmt ? (size_t)tensor->O * sizeof(float) : 0);
        if (ctx->tensor_count) ctx->tensor_count--;
        if (ctx->tensor_bytes >= bytes) ctx->tensor_bytes -= bytes;
    }
    if (tensor->weights) cudaFree(tensor->weights);
    if (tensor->scales) cudaFree(tensor->scales);
    std::free(tensor);
}

extern "C" size_t coli_cuda_tensor_bytes(const ColiCudaTensor *tensor) {
    return tensor ? tensor->weight_bytes + (tensor->fmt ? (size_t)tensor->O * sizeof(float) : 0) : 0;
}

extern "C" int coli_cuda_tensor_device(const ColiCudaTensor *tensor) {
    return tensor ? tensor->device : -1;
}
