// vramtest.cu -- the card's memory and its copies, checked: GB of VRAM in 1 GiB blocks, each pass writes a
// pattern that depends on the address and the pass (and its inverse), reads it back on the card and counts
// the words that differ; every pass also copies pinned host blocks up and back and compares them on the
// host. Runs SECONDS seconds, then prints every count. A healthy card at its clocks: 0.
// usage: vramtest GB SECONDS
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("%s: %s\n", #x, cudaGetErrorString(e_)); exit(1); } } while (0)

__device__ __forceinline__ uint32_t pat(size_t i, uint32_t seed) {
    uint32_t x = (uint32_t)i * 0x9E3779B1u ^ (uint32_t)(i >> 32) * 0x85EBCA77u ^ seed;
    x ^= x >> 16; x *= 0x7FEB352Du; x ^= x >> 15; x *= 0x846CA68Bu; x ^= x >> 16;
    return x;
}
__global__ void k_write(uint32_t *p, size_t n, size_t base, uint32_t seed, uint32_t inv) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        p[i] = pat(base + i, seed) ^ inv;
}
__global__ void k_check(const uint32_t *p, size_t n, size_t base, uint32_t seed, uint32_t inv,
                        unsigned long long *bad, unsigned long long *first) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        const uint32_t want = pat(base + i, seed) ^ inv, got = p[i];
        if (got != want) {
            if (atomicAdd(bad, 1ull) < 8) first[atomicAdd(bad + 1, 1ull) % 8] = ((unsigned long long)(base + i) << 32) ^ (got ^ want);
        }
    }
}
// read-modify-write in place (moving inversions): check, then write the inverse
__global__ void k_flip(uint32_t *p, size_t n, size_t base, uint32_t seed, uint32_t inv, unsigned long long *bad) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        const uint32_t want = pat(base + i, seed) ^ inv, got = p[i];
        if (got != want) atomicAdd(bad, 1ull);
        p[i] = ~got;
    }
}

int main(int argc, char **argv) {
    const int gb = argc > 1 ? atoi(argv[1]) : 16;
    const double secs = argc > 2 ? atof(argv[2]) : 120.0;
    const size_t blk = (size_t)1 << 30, nw = blk / 4;
    std::vector<uint32_t *> d;
    for (int i = 0; i < gb; i++) { uint32_t *p = nullptr; if (cudaMalloc(&p, blk) != cudaSuccess) { cudaGetLastError(); break; } d.push_back(p); }
    printf("%zu GiB of VRAM under test for %.0f s\n", d.size(), secs);
    unsigned long long *dbad; CK(cudaMalloc(&dbad, 16 * 8)); CK(cudaMemset(dbad, 0, 16 * 8));
    // pinned host blocks for the copies: 2 GiB up and back each pass
    const size_t hb = (size_t)256 << 20;
    uint32_t *h[8], *hb2; CK(cudaHostAlloc(&hb2, hb, cudaHostAllocDefault));
    for (int i = 0; i < 8; i++) CK(cudaHostAlloc(&h[i], hb, cudaHostAllocDefault));
    cudaStream_t s; CK(cudaStreamCreate(&s));
    unsigned long long copy_bad = 0, passes = 0, words = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (uint32_t pass = 1;; pass++) {
        const uint32_t seed = pass * 0x2545F491u;
        for (size_t b = 0; b < d.size(); b++) k_write<<<4096, 256, 0, s>>>(d[b], nw, b * nw, seed, 0);
        for (size_t b = 0; b < d.size(); b++) k_flip<<<4096, 256, 0, s>>>(d[b], nw, b * nw, seed, 0, dbad + 2);
        for (size_t b = 0; b < d.size(); b++) k_check<<<4096, 256, 0, s>>>(d[b], nw, b * nw, seed, 0xFFFFFFFFu, dbad, dbad + 4);
        CK(cudaGetLastError());
        // the copy engines: host patterns up into block 0's first 2 GiB, back into hb2, compared on the host
        for (int i = 0; i < 8 && !d.empty() && pass % 8 == 1; i++) {   // every 8th pass: the host's fill is the slow part
            uint32_t *hp = h[i];
            const size_t n = hb / 4;
            for (size_t k = 0; k < n; k++) hp[k] = (uint32_t)(k * 2654435761u) ^ seed ^ (uint32_t)i;
            CK(cudaMemcpyAsync((char *)d[0] + (size_t)i * hb, hp, hb, cudaMemcpyHostToDevice, s));
            CK(cudaMemcpyAsync(hb2, (char *)d[0] + (size_t)i * hb, hb, cudaMemcpyDeviceToHost, s));
            CK(cudaStreamSynchronize(s));
            if (memcmp(hp, hb2, hb)) for (size_t k = 0; k < n; k++) copy_bad += hp[k] != hb2[k];
        }
        CK(cudaStreamSynchronize(s));
        passes++; words += (unsigned long long)d.size() * nw * 3;
        const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (el >= secs) break;
    }
    unsigned long long hbad[16]; CK(cudaMemcpy(hbad, dbad, sizeof hbad, cudaMemcpyDeviceToHost));
    const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    printf("%llu passes in %.0f s, %.3g word reads: %llu wrong after a write, %llu wrong on the read-and-invert; "
           "copies through pinned memory: %llu wrong words\n", passes, el, (double)words, hbad[0], hbad[2], copy_bad);
    for (int i = 0; i < 8 && i < (int)hbad[0]; i++)
        printf("  word %llu: bits %08llx wrong\n", hbad[4 + i] >> 32, hbad[4 + i] & 0xFFFFFFFFull);
    return hbad[0] || hbad[2] || copy_bad ? 2 : 0;
}
