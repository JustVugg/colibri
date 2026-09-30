#include "../backend_cuda_dsv4.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static float bf16(float x) {
    uint32_t bits; std::memcpy(&bits, &x, 4);
    bits = (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000;
    std::memcpy(&x, &bits, 4); return x;
}
static void reference(float *out, const float *q, const float *past,
    const int64_t *positions, const float *block, const float *sinks,
    int64_t position, int pn, int bn, int heads, int dim) {
    for (int h = 0; h < heads; h++) {
        std::vector<const float *> keys;
        for (int i = 0; i < pn; i++)
            if (positions[i] >= 0 && positions[i] < position) keys.push_back(past + i * dim);
        for (int i = 0; i < bn; i++) keys.push_back(block + i * dim);
        std::vector<float> scores;
        float maximum = sinks[h];
        for (auto k : keys) {
            float dot = 0;
            for (int d = 0; d < dim; d++) { volatile float product = q[h * dim + d] * k[d]; dot += product; }
            scores.push_back(dot / std::sqrt(float(dim)));
            maximum = std::max(maximum, scores.back());
        }
        float denominator = std::exp(sinks[h] - maximum);
        for (auto &s : scores) { s = std::exp(s - maximum); denominator += s; }
        for (int d = 0; d < dim; d++) {
            float sum = 0;
            for (size_t i = 0; i < keys.size(); i++) {
                volatile float product = (scores[i] / denominator) * keys[i][d]; sum += product;
            }
            out[h * dim + d] = bf16(sum);
        }
    }
}
int main() {
    int devices[] = {0, 5}; assert(dsv4_cuda_init(devices, 2));
    double worst = 0;
    for (int device : devices) for (int dim : {32, 128, 512})
    for (int heads : {1, 64}) for (int pn : {0, 8, 128}) {
        constexpr int bn = 5;
        std::vector<float> q(heads * dim), past(pn * dim), block(bn * dim), sinks(heads), want(heads * dim), got(heads * dim);
        std::vector<int64_t> positions(pn);
        for (int i = 0; i < heads * dim; i++) q[i] = (i % 97 - 48) * .021f;
        for (int i = 0; i < pn; i++) {
            positions[i] = i % 3 == 0 ? -1 : (i % 11);
            for (int d = 0; d < dim; d++) past[i * dim + d] = positions[i] >= 0 && positions[i] < 7
                ? ((i * 17 + d) % 41 - 20) * .023f : NAN;
        }
        for (int i = 0; i < bn * dim; i++) block[i] = ((i * 13) % 59 - 29) * .037f;
        for (int h = 0; h < heads; h++) sinks[h] = h % 3 == 0 ? 30.f : -.1f * h;
        reference(want.data(), q.data(), past.data(), positions.data(), block.data(), sinks.data(), 7, pn, bn, heads, dim);
        assert(dsv4_cuda_dspark_attention(device, got.data(), q.data(), past.data(), positions.data(), block.data(), sinks.data(), 7, pn, bn, heads, dim));
        double error = 0, norm = 0;
        for (size_t i = 0; i < got.size(); i++) {
            assert(std::isfinite(got[i]));
            assert(std::fabs(got[i] - want[i]) <= 1e-5f + .008f * std::fabs(want[i]));
            error += double(got[i] - want[i]) * (got[i] - want[i]); norm += double(want[i]) * want[i];
        }
        double relative = std::sqrt(error / std::max(norm, 1e-30));
        assert(relative < .002); worst = std::max(worst, relative);
        assert(!dsv4_cuda_dspark_attention(device, got.data(), q.data(), past.data(), positions.data(), block.data(), sinks.data(), 7, 129, bn, heads, dim));
        assert(!dsv4_cuda_dspark_attention(device, got.data(), q.data(), past.data(), positions.data(), block.data(), sinks.data(), 7, pn, 0, heads, dim));
    }
    dsv4_cuda_shutdown();
    printf("DSpark attention: two devices, masked invalid/future slots, bidirectional block, sink; worst relative L2 %.9g\n", worst);
}
