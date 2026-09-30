#include "../backend_cuda_dsv4_quant.cuh"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

/* Freeze the previous implementation as an independent numerical oracle. */
__host__ __device__ float legacy_decode(unsigned char b) {
    int e = (b >> 3) & 15, m = b & 7;
    float v = !e ? ldexpf((float)m, -9)
                : e == 15 ? (m == 7 ? NAN : ldexpf(1.f + m / 8.f, 8))
                          : ldexpf(1.f + m / 8.f, e - 7);
    return b & 128 ? -v : v;
}

__device__ unsigned char legacy_encode(float x) {
    int best = 0;
    float distance = INFINITY;
    for (int v = 0; v < 255; v++) {
        float d = fabsf(legacy_decode(v) - x);
        if (d < distance || (d == distance && !(v & 1) && (best & 1))) {
            distance = d;
            best = v;
        }
    }
    return best;
}

__global__ void check(const float *values, int count, unsigned *failures) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 256) {
        float old = legacy_decode(i), now = e4m3(i);
        if (!(isnan(old) && isnan(now)) && __float_as_uint(old) != __float_as_uint(now))
            atomicAdd(failures, 1u);
    }
    if (i < 16) {
        const float table[] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
        float old = i & 8 ? -table[i & 7] : table[i & 7];
        if (__float_as_uint(old) != __float_as_uint(f4(i))) atomicAdd(failures, 1u);
    }
    if (i < count) {
        unsigned char expected = legacy_encode(values[i]);
        if (expected != e4m3_code(values[i]) ||
            __float_as_uint(legacy_decode(expected)) != __float_as_uint(e4m3_round(values[i])))
            atomicAdd(failures, 1u);
    }
}

int main() {
    std::vector<float> values = {0.f, -0.f, INFINITY, -INFINITY, NAN};
    for (int i = 0; i < 127; i++) {
        float value = legacy_decode(i);
        values.push_back(value);
        values.push_back(-value);
        if (i == 126) continue;
        float midpoint = (value + legacy_decode(i + 1)) / 2.f;
        for (float x : {nextafterf(midpoint, -INFINITY), midpoint,
                        nextafterf(midpoint, INFINITY)}) {
            values.push_back(x);
            values.push_back(-x);
        }
    }
    /* Every finite BF16 value in the callers' clamped domain. */
    for (unsigned bits = 0; bits < 65536; bits++) {
        unsigned raw = bits << 16;
        float value;
        memcpy(&value, &raw, sizeof(value));
        if (std::isfinite(value) && fabsf(value) <= 448.f) values.push_back(value);
    }
    for (int device : {0, 5}) {
        assert(cudaSetDevice(device) == cudaSuccess);
        float *input;
        unsigned *failures, result = 0;
        assert(cudaMalloc(&input, values.size() * sizeof(float)) == cudaSuccess);
        assert(cudaMalloc(&failures, sizeof(unsigned)) == cudaSuccess);
        assert(cudaMemcpy(input, values.data(), values.size() * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);
        assert(cudaMemset(failures, 0, sizeof(unsigned)) == cudaSuccess);
        check<<<(values.size() + 255) / 256, 256>>>(input, values.size(), failures);
        assert(cudaGetLastError() == cudaSuccess);
        assert(cudaMemcpy(&result, failures, sizeof(result), cudaMemcpyDeviceToHost) == cudaSuccess);
        printf("device=%d cases=%zu mismatches=%u\n", device, values.size(), result);
        assert(result == 0);
        assert(cudaFree(input) == cudaSuccess);
        assert(cudaFree(failures) == cudaSuccess);
    }
}
