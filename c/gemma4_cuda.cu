#include "gemma4_cuda.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

struct gemma4_cuda_context {
    int device;
    uint8_t *payload;
    float *input;
    float *gate;
    float *up;
    float *hidden;
    float *output;
    size_t payload_capacity;
    uint32_t model_capacity;
    uint32_t expert_capacity;
};

static void cuda_error(char *error, size_t capacity, const char *operation,
                       cudaError_t result) {
    if (error && capacity)
        std::snprintf(error, capacity, "%s: %s", operation,
                      cudaGetErrorString(result));
}

__device__ static float fp16_value(const uint8_t *bytes) {
    __half_raw raw;
    raw.x = static_cast<unsigned short>(bytes[0]) |
            static_cast<unsigned short>(bytes[1] << 8U);
    return __half2float(raw);
}

__global__ static void q4_0_matvec(const uint8_t *weights,
                                   uint32_t rows, uint32_t columns,
                                   const float *input, float multiplier,
                                   float *output) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const uint32_t blocks = columns / 32U;
    const uint8_t *row_data = weights + static_cast<size_t>(row) * blocks * 18U;
    float sum = 0.0F;
    for (uint32_t block = 0; block < blocks; ++block) {
        const uint8_t *encoded = row_data + static_cast<size_t>(block) * 18U;
        const float scale = fp16_value(encoded);
        const float *x = input + static_cast<size_t>(block) * 32U;
        float dot = 0.0F;
        for (uint32_t lane = 0; lane < 16U; ++lane) {
            const uint8_t packed = encoded[2U + lane];
            dot += static_cast<float>(static_cast<int>(packed & 15U) - 8) *
                   x[lane];
            dot += static_cast<float>(static_cast<int>(packed >> 4U) - 8) *
                   x[lane + 16U];
        }
        sum += scale * dot;
    }
    output[row] = multiplier * sum;
}

__global__ static void gated_gelu(const float *gate, const float *up,
                                  uint32_t count, float *hidden) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float value = gate[index];
    const float activated = 0.5F * value *
        (1.0F + tanhf(0.7978845608028654F *
                      (value + 0.044715F * value * value * value)));
    hidden[index] = activated * up[index];
}

static void release_buffers(gemma4_cuda_context *context) {
    if (!context) return;
    cudaFree(context->payload); cudaFree(context->input);
    cudaFree(context->gate); cudaFree(context->up);
    cudaFree(context->hidden); cudaFree(context->output);
    context->payload = nullptr; context->input = nullptr;
    context->gate = nullptr; context->up = nullptr;
    context->hidden = nullptr; context->output = nullptr;
    context->payload_capacity = 0;
    context->model_capacity = 0;
    context->expert_capacity = 0;
}

static cudaError_t reserve(gemma4_cuda_context *context,
                           size_t payload_bytes, uint32_t model_width,
                           uint32_t expert_width) {
    if (context->payload_capacity >= payload_bytes &&
        context->model_capacity >= model_width &&
        context->expert_capacity >= expert_width) return cudaSuccess;
    release_buffers(context);
    cudaError_t result = cudaMalloc(&context->payload, payload_bytes);
    if (result == cudaSuccess)
        result = cudaMalloc(&context->input,
                            static_cast<size_t>(model_width) * sizeof(float));
    if (result == cudaSuccess)
        result = cudaMalloc(&context->gate,
                            static_cast<size_t>(expert_width) * sizeof(float));
    if (result == cudaSuccess)
        result = cudaMalloc(&context->up,
                            static_cast<size_t>(expert_width) * sizeof(float));
    if (result == cudaSuccess)
        result = cudaMalloc(&context->hidden,
                            static_cast<size_t>(expert_width) * sizeof(float));
    if (result == cudaSuccess)
        result = cudaMalloc(&context->output,
                            static_cast<size_t>(model_width) * sizeof(float));
    if (result != cudaSuccess) {
        release_buffers(context);
        return result;
    }
    context->payload_capacity = payload_bytes;
    context->model_capacity = model_width;
    context->expert_capacity = expert_width;
    return cudaSuccess;
}

extern "C" int coli_gemma4_cuda_create(void **opaque, int device,
                                        char *error, size_t error_capacity) {
    if (!opaque || device < 0) return -1;
    *opaque = nullptr;
    int count = 0;
    cudaError_t result = cudaGetDeviceCount(&count);
    if (result != cudaSuccess || device >= count) {
        cuda_error(error, error_capacity, "cannot select CUDA device", result);
        return -1;
    }
    result = cudaSetDevice(device);
    if (result != cudaSuccess) {
        cuda_error(error, error_capacity, "cannot activate CUDA device", result);
        return -1;
    }
    gemma4_cuda_context *context =
        static_cast<gemma4_cuda_context *>(std::calloc(1, sizeof(*context)));
    if (!context) return -1;
    context->device = device;
    *opaque = context;
    return 0;
}

extern "C" void coli_gemma4_cuda_destroy(void *opaque) {
    gemma4_cuda_context *context = static_cast<gemma4_cuda_context *>(opaque);
    if (!context) return;
    cudaSetDevice(context->device);
    release_buffers(context);
    std::free(context);
}

extern "C" int coli_gemma4_cuda_run(
    void *opaque, const uint8_t *payload, size_t payload_bytes,
    size_t gate_bytes, size_t up_bytes, uint32_t model_width,
    uint32_t expert_width, const float *input, float scale, float *output,
    char *error, size_t error_capacity) {
    gemma4_cuda_context *context = static_cast<gemma4_cuda_context *>(opaque);
    if (!context || !payload || !input || !output || !model_width ||
        !expert_width || model_width % 32U || expert_width % 32U ||
        gate_bytes + up_bytes > payload_bytes) return -1;
    cudaError_t result = cudaSetDevice(context->device);
    if (result == cudaSuccess)
        result = reserve(context, payload_bytes, model_width, expert_width);
    if (result == cudaSuccess)
        result = cudaMemcpy(context->payload, payload, payload_bytes,
                            cudaMemcpyHostToDevice);
    if (result == cudaSuccess)
        result = cudaMemcpy(context->input, input,
                            static_cast<size_t>(model_width) * sizeof(float),
                            cudaMemcpyHostToDevice);
    if (result != cudaSuccess) {
        cuda_error(error, error_capacity, "CUDA expert upload failed", result);
        return -1;
    }
    const uint32_t threads = 128;
    q4_0_matvec<<<(expert_width + threads - 1U) / threads, threads>>>(
        context->payload, expert_width, model_width, context->input, 1.0F,
        context->gate);
    q4_0_matvec<<<(expert_width + threads - 1U) / threads, threads>>>(
        context->payload + gate_bytes, expert_width, model_width,
        context->input, 1.0F, context->up);
    gated_gelu<<<(expert_width + threads - 1U) / threads, threads>>>(
        context->gate, context->up, expert_width, context->hidden);
    q4_0_matvec<<<(model_width + threads - 1U) / threads, threads>>>(
        context->payload + gate_bytes + up_bytes, model_width, expert_width,
        context->hidden, scale, context->output);
    result = cudaGetLastError();
    if (result == cudaSuccess)
        result = cudaMemcpy(output, context->output,
                            static_cast<size_t>(model_width) * sizeof(float),
                            cudaMemcpyDeviceToHost);
    if (result != cudaSuccess) {
        cuda_error(error, error_capacity, "CUDA expert execution failed", result);
        return -1;
    }
    return 0;
}
