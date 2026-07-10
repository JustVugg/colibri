#include "backend_cuda.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

struct ColiCudaTensor {
    void *weights;
    float *scales;
    size_t weight_bytes;
    int fmt, I, O;
    int tracked;
};

static int g_ready;
static int g_device;
static float *g_x;
static float *g_y;
static size_t g_x_cap;
static size_t g_y_cap;
static size_t g_tensor_count;
static size_t g_tensor_bytes;

static int cuda_ok(cudaError_t err, const char *what) {
    if (err == cudaSuccess) return 1;
    std::fprintf(stderr, "[CUDA] %s: %s\n", what, cudaGetErrorString(err));
    return 0;
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

extern "C" int coli_cuda_init(int device) {
    int count = 0;
    if (!cuda_ok(cudaGetDeviceCount(&count), "device discovery")) return 0;
    if (device < 0 || device >= count) {
        std::fprintf(stderr, "[CUDA] invalid device %d (available: 0..%d)\n", device, count - 1);
        return 0;
    }
    if (!cuda_ok(cudaSetDevice(device), "select device")) return 0;
    cudaDeviceProp prop{};
    if (!cuda_ok(cudaGetDeviceProperties(&prop, device), "device properties")) return 0;
    g_device = device;
    g_ready = 1;
    std::fprintf(stderr, "[CUDA] device %d: %s, %.1f GB VRAM, sm_%d%d\n",
                 device, prop.name, prop.totalGlobalMem / 1e9, prop.major, prop.minor);
    return 1;
}

extern "C" void coli_cuda_shutdown(void) {
    if (!g_ready) return;
    cudaSetDevice(g_device);
    if (g_x) cudaFree(g_x);
    if (g_y) cudaFree(g_y);
    g_x = g_y = nullptr;
    g_x_cap = g_y_cap = 0;
    g_ready = 0;
}

extern "C" int coli_cuda_mem_info(size_t *free_bytes, size_t *total_bytes) {
    if (!g_ready || !free_bytes || !total_bytes) return 0;
    return cuda_ok(cudaMemGetInfo(free_bytes, total_bytes), "memory info");
}

extern "C" void coli_cuda_stats(size_t *tensor_count, size_t *tensor_bytes) {
    if (tensor_count) *tensor_count = g_tensor_count;
    if (tensor_bytes) *tensor_bytes = g_tensor_bytes;
}

extern "C" int coli_cuda_tensor_upload(ColiCudaTensor **tensor,
                                        const void *weights, const float *scales,
                                        int fmt, int I, int O) {
    if (!g_ready || !tensor || !weights || I < 1 || O < 1) return 0;
    size_t rb = row_bytes(fmt, I);
    if (!rb || (fmt && !scales)) return 0;
    if (*tensor) {
        ColiCudaTensor *t = *tensor;
        return t->fmt == fmt && t->I == I && t->O == O;
    }
    ColiCudaTensor *t = static_cast<ColiCudaTensor *>(std::calloc(1, sizeof(*t)));
    if (!t) return 0;
    t->fmt = fmt; t->I = I; t->O = O; t->weight_bytes = rb * (size_t)O;
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
    g_tensor_count++;
    g_tensor_bytes += t->weight_bytes + (fmt ? (size_t)O * sizeof(float) : 0);
    *tensor = t;
    return 1;
}

extern "C" int coli_cuda_matmul(ColiCudaTensor **tensor,
                                 float *y, const float *x,
                                 const void *weights, const float *scales,
                                 int fmt, int S, int I, int O) {
    if (S < 1 || !coli_cuda_tensor_upload(tensor, weights, scales, fmt, I, O)) return 0;
    size_t rb = row_bytes(fmt, I);
    ColiCudaTensor *t = *tensor;
    size_t xb = (size_t)S * I * sizeof(float), yb = (size_t)S * O * sizeof(float);
    if (!reserve(&g_x, &g_x_cap, xb) || !reserve(&g_y, &g_y_cap, yb)) return 0;
    if (!cuda_ok(cudaMemcpy(g_x, x, xb, cudaMemcpyHostToDevice), "input upload")) return 0;
    dim3 grid((unsigned)O, (unsigned)S);
    quant_matmul<<<grid, 256>>>(g_y, g_x, t->weights, t->scales, fmt, S, I, O, rb);
    if (!cuda_ok(cudaGetLastError(), "matmul launch") ||
        !cuda_ok(cudaMemcpy(y, g_y, yb, cudaMemcpyDeviceToHost), "output download")) return 0;
    return 1;
}

extern "C" void coli_cuda_tensor_free(ColiCudaTensor *tensor) {
    if (!tensor) return;
    if (tensor->tracked) {
        size_t bytes = tensor->weight_bytes + (tensor->fmt ? (size_t)tensor->O * sizeof(float) : 0);
        if (g_tensor_count) g_tensor_count--;
        if (g_tensor_bytes >= bytes) g_tensor_bytes -= bytes;
    }
    if (tensor->weights) cudaFree(tensor->weights);
    if (tensor->scales) cudaFree(tensor->scales);
    std::free(tensor);
}

extern "C" size_t coli_cuda_tensor_bytes(const ColiCudaTensor *tensor) {
    return tensor ? tensor->weight_bytes + (tensor->fmt ? (size_t)tensor->O * sizeof(float) : 0) : 0;
}
