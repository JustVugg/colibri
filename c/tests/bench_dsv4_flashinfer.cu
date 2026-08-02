#include <cuda_runtime.h>
#include <flashinfer/gemm/group_gemm_mxfp4_groupwise_sm120.cuh>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <vector>

namespace flashinfer::group_gemm {
INSTANTIATE_GROUP_GEMM_MXFP4_GROUPWISE_SM120(
    128, 32, 128, true, cutlass::float_e4m3_t, cutlass::float_e2m1_t,
    cutlass::float_ue8m0_t, cutlass::float_ue8m0_t, cutlass::bfloat16_t,
    fp8, fp4, ue8m0, ue8m0, bf16)
}

using namespace flashinfer::group_gemm;

static void check(cudaError_t e, const char *where) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", where, cudaGetErrorString(e));
        std::exit(1);
    }
}

int main(int argc, char **argv) {
    int groups = argc > 1 ? std::atoi(argv[1]) : 6;
    int n = argc > 2 ? std::atoi(argv[2]) : 2048;
    int k = argc > 3 ? std::atoi(argv[3]) : 4096;
    constexpr int rows = 4;
    int total_rows = groups * rows;
    constexpr size_t int_workspace_bytes = 8 << 20;
    constexpr size_t float_workspace_bytes = 64 << 20;
    cutlass::float_e4m3_t *a;
    cutlass::float_e2m1_t *b;
    cutlass::float_ue8m0_t *as, *bs;
    cutlass::bfloat16_t *out;
    int *indptr;
    void *int_workspace, *float_workspace;
    check(cudaMalloc(&a, (size_t)total_rows * k), "allocate activation");
    check(cudaMalloc(&b, (size_t)groups * n * k / 2), "allocate weight");
    check(cudaMalloc(&as, (size_t)groups * 128 * (k / 32)), "allocate activation scales");
    check(cudaMalloc(&bs, (size_t)groups * n * (k / 32)), "allocate weight scales");
    check(cudaMalloc(&out, (size_t)total_rows * n * sizeof(*out)), "allocate output");
    check(cudaMalloc(&indptr, (groups + 1) * sizeof(*indptr)), "allocate indptr");
    check(cudaMalloc(&int_workspace, int_workspace_bytes), "allocate integer workspace");
    check(cudaMalloc(&float_workspace, float_workspace_bytes), "allocate CUTLASS workspace");
    check(cudaMemset(a, 0x38, (size_t)total_rows * k), "initialize unit FP8 activation");
    check(cudaMemset(b, 0x22, (size_t)groups * n * k / 2), "initialize unit FP4 weight");
    check(cudaMemset(as, 127, (size_t)groups * 128 * (k / 32)), "initialize activation scales");
    check(cudaMemset(bs, 127, (size_t)groups * n * (k / 32)), "initialize weight scales");
    std::vector<int> host_indptr(groups + 1);
    for (int i = 0; i <= groups; ++i) host_indptr[i] = i * rows;
    check(cudaMemcpy(indptr, host_indptr.data(), (groups + 1) * sizeof(*indptr), cudaMemcpyHostToDevice),
          "upload indptr");
    auto run = [&] {
        check(CutlassMXFP4GroupwiseScaledGroupGEMMSM120<
                  128, 32, 128, true, cutlass::float_e4m3_t, cutlass::float_e2m1_t,
                  cutlass::float_ue8m0_t, cutlass::float_ue8m0_t, cutlass::bfloat16_t>(
                  int_workspace, int_workspace_bytes, float_workspace, float_workspace_bytes,
                  a, b, as, bs, out, indptr, n, k, groups, nullptr, 0),
              "launch grouped MXFP4 GEMM");
    };
    for (int i = 0; i < 10; ++i) run();
    check(cudaDeviceSynchronize(), "warmup");
    cutlass::bfloat16_t first;
    check(cudaMemcpy(&first, out, sizeof(first), cudaMemcpyDeviceToHost), "download correctness sample");
    if (std::abs(float(first) - k) > k / 100.f) {
        std::fprintf(stderr, "wrong MXFP4 result: got %.3f expected %d\n", float(first), k);
        return 2;
    }
    cudaEvent_t begin, end;
    check(cudaEventCreate(&begin), "create begin event");
    check(cudaEventCreate(&end), "create end event");
    check(cudaEventRecord(begin), "record begin");
    for (int i = 0; i < 100; ++i) run();
    check(cudaEventRecord(end), "record end");
    check(cudaEventSynchronize(end), "benchmark");
    float elapsed_ms = 0;
    check(cudaEventElapsedTime(&elapsed_ms, begin, end), "elapsed time");
    std::printf("flashinfer_sm120_mxfp4 groups=%d n=%d k=%d %.3f us/call\n",
                groups, n, k, elapsed_ms * 10.f);
}
