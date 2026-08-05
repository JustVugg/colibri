// GPU decode microbenchmark for analyze_split_entropy.py's practical codec.
// Build: nvcc -O3 -arch=native -o /tmp/bench_split benchmark_split_entropy.cu
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define CUDA_OK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    exit(1); }} while (0)

static constexpr int TILE = 4096;
static constexpr int THREADS = 256;

__device__ __forceinline__ unsigned get_bits(
        const uint8_t *p, unsigned bit, unsigned width) {
    if (!width) return 0;
    uint64_t word = 0;
    memcpy(&word, p + (bit >> 3), sizeof(word));
    return (word >> (bit & 7)) & ((1u << width) - 1);
}

template<int BASE_BITS>
__global__ void decode(const uint8_t *__restrict__ base,
                       const uint8_t *__restrict__ residual,
                       const uint32_t *__restrict__ residual_offsets,
                       uint8_t *__restrict__ out, size_t weights,
                       const uint8_t *__restrict__ group_width,
                       const uint8_t *__restrict__ group_symbol) {
    __shared__ uint16_t prefix[TILE];
    const size_t tile = blockIdx.x;
    const size_t begin = tile * TILE;
    if (begin >= weights) return;

    for (int i = threadIdx.x; i < TILE; i += blockDim.x) {
        size_t index = begin + i;
        unsigned group = 0;
        if (index < weights)
            group = get_bits(base + tile * (TILE * BASE_BITS / 8),
                             i * BASE_BITS, BASE_BITS);
        prefix[i] = index < weights ? group_width[group] : 0;
    }
    __syncthreads();

    // Blelloch exclusive scan; max tile residual is 16384 bits.
    for (int stride = 1; stride < TILE; stride <<= 1) {
        int i = (threadIdx.x + 1) * stride * 2 - 1;
        for (; i < TILE; i += blockDim.x * stride * 2)
            prefix[i] += prefix[i - stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) prefix[TILE - 1] = 0;
    __syncthreads();
    for (int stride = TILE >> 1; stride; stride >>= 1) {
        int i = (threadIdx.x + 1) * stride * 2 - 1;
        for (; i < TILE; i += blockDim.x * stride * 2) {
            uint16_t left = prefix[i - stride];
            prefix[i - stride] = prefix[i];
            prefix[i] += left;
        }
        __syncthreads();
    }

    const uint8_t *tile_residual = residual + residual_offsets[tile];
    for (int i = threadIdx.x; i < TILE; i += blockDim.x) {
        size_t index = begin + i;
        if (index >= weights) continue;
        unsigned group = get_bits(base + tile * (TILE * BASE_BITS / 8),
                                  i * BASE_BITS, BASE_BITS);
        unsigned local = get_bits(tile_residual, prefix[i],
                                  group_width[group]);
        out[index] = group_symbol[group * 16 + local];
    }
}

static void put_bits(std::vector<uint8_t>& dst, size_t bit,
                     unsigned width, unsigned value) {
    for (unsigned i = 0; i < width; ++i)
        if (value & (1u << i))
            dst[(bit + i) >> 3] |= 1u << ((bit + i) & 7);
}

template<int BASE_BITS>
static void run(const std::vector<uint8_t>& symbols,
                const uint8_t *symbol_group, const uint8_t *symbol_local,
                const std::vector<uint8_t>& widths,
                const std::vector<uint8_t>& group_symbols) {
    const size_t weights = symbols.size();
    const size_t tiles = (weights + TILE - 1) / TILE;
    std::vector<uint8_t> base(tiles * TILE * BASE_BITS / 8);
    std::vector<uint8_t> residual;
    std::vector<uint32_t> offsets(tiles + 1);
    residual.reserve(weights / 2);
    for (size_t tile = 0; tile < tiles; ++tile) {
        offsets[tile] = residual.size();
        size_t residual_bit = 0;
        std::vector<uint8_t> chunk(TILE / 2 + 8);
        for (int i = 0; i < TILE; ++i) {
            size_t index = tile * TILE + i;
            uint8_t symbol = index < weights ? symbols[index] : 0;
            unsigned group = symbol_group[symbol];
            put_bits(base, tile * TILE * BASE_BITS + i * BASE_BITS,
                     BASE_BITS, group);
            if (index < weights) {
                put_bits(chunk, residual_bit, widths[group],
                         symbol_local[symbol]);
                residual_bit += widths[group];
            }
        }
        size_t bytes = (residual_bit + 7) / 8;
        residual.insert(residual.end(), chunk.begin(), chunk.begin() + bytes);
    }
    offsets[tiles] = residual.size();
    residual.resize(residual.size() + 8);

    uint8_t *d_base, *d_residual, *d_out, *d_width, *d_symbols;
    uint32_t *d_offsets;
    CUDA_OK(cudaMalloc(&d_base, base.size()));
    CUDA_OK(cudaMalloc(&d_residual, residual.size()));
    CUDA_OK(cudaMalloc(&d_offsets, offsets.size() * sizeof(uint32_t)));
    CUDA_OK(cudaMalloc(&d_out, weights));
    CUDA_OK(cudaMalloc(&d_width, widths.size()));
    CUDA_OK(cudaMalloc(&d_symbols, group_symbols.size()));
    CUDA_OK(cudaMemcpy(d_base, base.data(), base.size(), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_residual, residual.data(), residual.size(),
                       cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_offsets, offsets.data(),
                       offsets.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_width, widths.data(), widths.size(),
                       cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_symbols, group_symbols.data(), group_symbols.size(),
                       cudaMemcpyHostToDevice));

    decode<BASE_BITS><<<tiles, THREADS>>>(d_base, d_residual, d_offsets, d_out,
                                          weights, d_width, d_symbols);
    CUDA_OK(cudaDeviceSynchronize());
    std::vector<uint8_t> output(weights);
    CUDA_OK(cudaMemcpy(output.data(), d_out, weights, cudaMemcpyDeviceToHost));
    if (output != symbols) {
        fprintf(stderr, "base%d roundtrip mismatch\n", BASE_BITS);
        exit(2);
    }

    cudaEvent_t start, stop;
    CUDA_OK(cudaEventCreate(&start));
    CUDA_OK(cudaEventCreate(&stop));
    constexpr int rounds = 100;
    CUDA_OK(cudaEventRecord(start));
    for (int i = 0; i < rounds; ++i)
        decode<BASE_BITS><<<tiles, THREADS>>>(d_base, d_residual, d_offsets,
                                              d_out, weights, d_width, d_symbols);
    CUDA_OK(cudaEventRecord(stop));
    CUDA_OK(cudaEventSynchronize(stop));
    float ms;
    CUDA_OK(cudaEventElapsedTime(&ms, start, stop));
    double seconds = ms / 1e3 / rounds;
    printf("base%d: base %.3f bits + residual %.3f bits = %.3f bits/weight"
           " | exact roundtrip | decode %.1f Gweight/s"
           " | raw-INT4 equivalent %.1f GB/s",
           BASE_BITS, double(BASE_BITS),
           residual.size() * 8.0 / weights,
           BASE_BITS + residual.size() * 8.0 / weights,
           weights / seconds / 1e9, weights / 2.0 / seconds / 1e9);

    CUDA_OK(cudaHostRegister(residual.data(), residual.size(),
                             cudaHostRegisterDefault));
    CUDA_OK(cudaEventRecord(start));
    for (int i = 0; i < rounds; ++i) {
        CUDA_OK(cudaMemcpyAsync(d_residual, residual.data(), residual.size(),
                                cudaMemcpyHostToDevice));
        decode<BASE_BITS><<<tiles, THREADS>>>(d_base, d_residual, d_offsets,
                                              d_out, weights, d_width, d_symbols);
    }
    CUDA_OK(cudaEventRecord(stop));
    CUDA_OK(cudaEventSynchronize(stop));
    CUDA_OK(cudaEventElapsedTime(&ms, start, stop));
    seconds = ms / 1e3 / rounds;
    printf(" | H2D+decode %.1f Gweight/s (residual %.1f GB/s,"
           " raw-equivalent %.1f GB/s)\n",
           weights / seconds / 1e9,
           (residual.size() - 8) / seconds / 1e9,
           weights / 2.0 / seconds / 1e9);
    CUDA_OK(cudaHostUnregister(residual.data()));

    cudaFree(d_base); cudaFree(d_residual); cudaFree(d_offsets);
    cudaFree(d_out); cudaFree(d_width); cudaFree(d_symbols);
}

int main() {
    const size_t weights = 1ull << 26;
    const double p[16] = {
        0, .0003231, .001374, .005613, .02026, .05401, .1164, .1861,
        .2314, .1861, .1164, .05399, .02025, .005611, .001373, .0003231};
    std::discrete_distribution<int> distribution(p, p + 16);
    std::mt19937 rng(52);
    std::vector<uint8_t> symbols(weights);
    for (auto& symbol : symbols) symbol = distribution(rng);

    // Descending-probability order: 8,7,9,6,10,5,11,4,12,3,13,2,14,1,15,0.
    const uint8_t order[16] = {8,7,9,6,10,5,11,4,12,3,13,2,14,1,15,0};
    for (int bits = 1; bits <= 2; ++bits) {
        const std::vector<int> sizes = bits == 1
            ? std::vector<int>{4, 12} : std::vector<int>{1, 2, 2, 11};
        std::vector<uint8_t> widths, table(sizes.size() * 16);
        uint8_t group[16] = {}, local[16] = {};
        int cursor = 0;
        for (int g = 0; g < (int)sizes.size(); ++g) {
            widths.push_back(sizes[g] == 1 ? 0 :
                             sizes[g] <= 2 ? 1 :
                             sizes[g] <= 4 ? 2 : 4);
            for (int i = 0; i < sizes[g]; ++i) {
                uint8_t symbol = order[cursor++];
                group[symbol] = g; local[symbol] = i;
                table[g * 16 + i] = symbol;
            }
        }
        if (bits == 1) run<1>(symbols, group, local, widths, table);
        else run<2>(symbols, group, local, widths, table);
    }
}
