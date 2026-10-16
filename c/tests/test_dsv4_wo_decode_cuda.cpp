#include "../backend_cuda_dsv4.h"
#include <cuda_runtime.h>
#include <cassert>
#include <bit>
#include <cstdio>
#include <cstring>
#include <vector>

static void round_bf16(std::vector<float> &values) {
    for (float &value : values) {
        uint32_t bits = std::bit_cast<uint32_t>(value);
        if ((bits & 0x7f800000u) != 0x7f800000u)
            bits += 0x7fffu + ((bits >> 16) & 1u);
        bits &= 0xffff0000u;
        value = std::bit_cast<float>(bits);
    }
}

int main() {
    int devices[] = {0, 5};
    assert(dsv4_cuda_init(devices, 2));
    assert(!dsv4_cuda_wo_decode(nullptr, nullptr, 1, nullptr, nullptr));
    for (int device : devices) for (int groups : {1, 8}) for (int format : {8, 9}) {
        const int rank = groups == 1 ? 128 : 1024;
        const int width = groups == 1 ? 256 : 4096;
        const int rows = groups * rank, hidden = groups == 1 ? 256 : 4096;
        std::vector<uint8_t> a((size_t)rows * width), b((size_t)hidden * rows);
        std::vector<uint8_t> as(rows / 128 * (width / 128)), bs(hidden / 128 * (rows / 128));
        unsigned seed = 12345;
        auto next = [&]() { seed = 1664525 * seed + 1013904223; return seed; };
        for (auto &v : a) v = (next() >> 16) % 120 | (next() & 128);
        for (auto &v : b) v = (next() >> 16) % 120 | (next() & 128);
        for (auto &v : as) v = 117 + (next() >> 16) % 5;
        for (auto &v : bs) v = 117 + (next() >> 16) % 5;
        std::vector<float> x(groups * width), middle(rows), expected(hidden), actual(hidden), quantized(hidden);
        Dsv4CudaTensor *wa = nullptr, *wb = nullptr, *peer = nullptr;
        auto upload = format == 9 ? dsv4_cuda_upload_fp8_bf16 : dsv4_cuda_upload_fp8;
        assert(upload(&wa, a.data(), as.data(), rows, width, device));
        assert(dsv4_cuda_upload_fp8(&wb, b.data(), bs.data(), hidden, rows, device));
        // Generic upload uses the default stream; matvec uses a nonblocking stream.
        assert(cudaDeviceSynchronize() == cudaSuccess);
        assert(!dsv4_cuda_wo_decode(wa, wb, 0, actual.data(), x.data()));
        assert(!dsv4_cuda_wo_decode(wa, wb, rows + 1, actual.data(), x.data()));
        assert(!dsv4_cuda_wo_decode(wa, wa, groups, actual.data(), x.data()));
        assert(!dsv4_cuda_wo_decode(wa, wb, groups, nullptr, x.data()));
        assert(dsv4_cuda_upload_fp8(&peer, b.data(), bs.data(), hidden, rows, device == 0 ? 5 : 0));
        assert(cudaDeviceSynchronize() == cudaSuccess);
        assert(!dsv4_cuda_wo_decode(wa, peer, groups, actual.data(), x.data()));
        for (int trial = 0; trial < 3; trial++) {
            for (auto &v : x) v = ((int)((next() >> 8) % 200001) - 100000) / 100003.f;
            assert(dsv4_cuda_matvec_grouped(wa, middle.data(), x.data(), groups));
            round_bf16(middle);
            assert(dsv4_cuda_matvec(wb, expected.data(), middle.data()));
            round_bf16(expected);
            assert(dsv4_cuda_wo_decode(wa, wb, groups, actual.data(), x.data()));
            assert(std::memcmp(expected.data(), actual.data(), hidden * sizeof(float)) == 0);
            // Catch accidental reuse of the older extra-FP8-quantization path.
            if (format == 9) {
                assert(dsv4_cuda_wo(wa, wb, groups, quantized.data(), x.data()));
                assert(expected != quantized);
            }
        }
        printf("device=%d groups=%d format=%d width=%d rank=%d exact=3/3\n", device, groups, format, width, rank);
        dsv4_cuda_tensor_free(wa); dsv4_cuda_tensor_free(wb); dsv4_cuda_tensor_free(peer);
    }
    dsv4_cuda_shutdown();
}
