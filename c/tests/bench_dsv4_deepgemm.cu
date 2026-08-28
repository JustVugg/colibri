#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#include <deep_gemm/impls/sm120_fp8_fp4_gemm_1d1d.cuh>

using namespace deep_gemm;

static void check(cudaError_t e, const char *what) {
    if (e != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        exit(1);
    }
}

static void check_driver(CUresult e, const char *what) {
    if (e != CUDA_SUCCESS) {
        const char *message = nullptr;
        cuGetErrorString(e, &message);
        fprintf(stderr, "%s: %s\n", what, message ? message : "CUDA driver error");
        exit(1);
    }
}

static CUtensorMap map2d(CUtensorMapDataType type, void *ptr, unsigned long long inner,
                         unsigned long long outer, unsigned long long stride_bytes,
                         unsigned box_inner, unsigned box_outer, CUtensorMapSwizzle swizzle) {
    CUtensorMap map{};
    const cuuint64_t dims[2] = {inner, outer};
    const cuuint64_t strides[1] = {stride_bytes};
    const cuuint32_t box[2] = {box_inner, box_outer};
    const cuuint32_t elem[2] = {1, 1};
    check_driver(cuTensorMapEncodeTiled(&map, type, 2, ptr, dims, strides, box, elem,
                                       CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
                                       CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                                       CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
                 "encode TMA map");
    return map;
}

using Output = cutlass::bfloat16_t;
using Epilogue = epilogue::transform::EpilogueIdentity;

__global__ void fill_layout(int *layout) {
    int i = threadIdx.x;
    if (i < 96) layout[i] = i / 16;
}

int main() {
    constexpr int M = 96, N = 4096, K = 4096, G = 256;
    uint8_t *a, *b;
    int *sfa, *sfb, *layout;
    Output *d;
    check(cudaMalloc(&a, (size_t)M * K), "allocate A");
    check(cudaMalloc(&b, (size_t)G * N * K / 2), "allocate B");
    check(cudaMalloc(&sfa, (size_t)M * (K / 128 / 4) * sizeof(int)), "allocate SFA");
    check(cudaMalloc(&sfb, (size_t)G * N * (K / 32 / 4) * sizeof(int)), "allocate SFB");
    check(cudaMalloc(&d, (size_t)M * N * sizeof(*d)), "allocate D");
    check(cudaMalloc(&layout, M * sizeof(int)), "allocate grouped layout");
    check(cudaMemset(a, 0x38, (size_t)M * K), "initialize A to FP8 one");
    check(cudaMemset(b, 0x11, (size_t)G * N * K / 2), "initialize B to FP4 half");
    check(cudaMemset(sfa, 0x7f, (size_t)M * (K / 128 / 4) * sizeof(int)), "initialize SFA");
    check(cudaMemset(sfb, 0x7f, (size_t)G * N * (K / 32 / 4) * sizeof(int)), "initialize SFB");
    fill_layout<<<1, 128>>>(layout);

    CUtensorMap ma = map2d(CU_TENSOR_MAP_DATA_TYPE_UINT8, a, K, M, K, 128, 128,
                           CU_TENSOR_MAP_SWIZZLE_128B);
    CUtensorMap mb = map2d(CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B, b, K, (unsigned long long)G * N,
                           K / 2, 128, 64, CU_TENSOR_MAP_SWIZZLE_128B);
    CUtensorMap msa = map2d(CU_TENSOR_MAP_DATA_TYPE_INT32, sfa, M, K / 128 / 4,
                            (unsigned long long)M * sizeof(int), 128, 1,
                            CU_TENSOR_MAP_SWIZZLE_NONE);
    CUtensorMap msb = map2d(CU_TENSOR_MAP_DATA_TYPE_INT32, sfb, N, (unsigned long long)G * K / 32 / 4,
                            (unsigned long long)N * sizeof(int), 64, 1,
                            CU_TENSOR_MAP_SWIZZLE_NONE);
    CUtensorMap md = map2d(CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, d, N, M,
                           (unsigned long long)N * sizeof(*d), 64, 128,
                           CU_TENSOR_MAP_SWIZZLE_128B);

    auto kernel = sm120_fp8_fp4_gemm_1d1d_impl<
        0, N, K, 128, 32, G, 128, 64, 128, 128, 128, 128, 3, 128, 256, 170,
        GemmType::MGroupedContiguous, false, Output, Epilogue, false, true, false,
        true, false, 128, 1>;
    check(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 92672),
          "set DeepGEMM shared memory");

    auto launch = [&] {
        kernel<<<170, 384, 92672>>>(d, nullptr, reinterpret_cast<__nv_fp8_e4m3 *>(a),
                                    reinterpret_cast<__nv_fp8_e4m3 *>(b), layout, nullptr,
                                    nullptr, M, N, K, N, 0, 0, ma, mb, msa, msb, md);
    };
    for (int i = 0; i < 10; ++i) launch();
    check(cudaDeviceSynchronize(), "DeepGEMM warmup");
    __nv_bfloat16 first;
    check(cudaMemcpy(&first, d, sizeof(first), cudaMemcpyDeviceToHost), "read correctness sample");
    float got = __bfloat162float(first);
    if (got != 2048.f) {
        fprintf(stderr, "DeepGEMM mismatch: got %.9g, expected 2048\n", got);
        return 1;
    }
    cudaEvent_t begin, end;
    check(cudaEventCreate(&begin), "create begin event");
    check(cudaEventCreate(&end), "create end event");
    check(cudaEventRecord(begin), "record begin event");
    for (int i = 0; i < 100; ++i) launch();
    check(cudaEventRecord(end), "record end event");
    check(cudaEventSynchronize(end), "wait benchmark");
    float ms = 0;
    check(cudaEventElapsedTime(&ms, begin, end), "read benchmark time");
    check(cudaGetLastError(), "DeepGEMM launch");
    printf("DeepGEMM SM120 M=%d N=%d K=%d G=%d: %.4f ms/launch, sample=%.0f\n",
           M, N, K, G, ms / 100, got);
    return 0;
}
