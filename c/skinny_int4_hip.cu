/* skinny_int4_hip.cu — bandwidth-optimal int4 W4A16 skinny GEMV for gfx1100.
 *
 * Extracted and reduced from the vLLM ROCm fork
 * (csrc/rocm/skinny_gemms_int4.cu, wvSplitK_int4_hf_sml_): half activations,
 * per-channel scale (GROUP_SIZE=0), symmetric uint4b8, no zero points, no bias,
 * "small" case where the activation tile fits in LDS. Colibri's resident int4
 * expert decode hits exactly this case (S small, K in {2048,6144}).
 *
 * Weights arrive as colibri fmt=2 (signed two's-complement s4 after the upload
 * XOR), packed [O, K/2] sequential nibbles. coli_hip_int4_repack() undoes the
 * XOR (back to uint4b8) and permutes nibbles into the layout this kernel's
 * decode scramble expects (see repack for the exact permutation).
 *
 * Only compiled under HIP=1 (hipcc). Empty under nvcc.
 */
#ifdef COLI_HIP
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstdlib>
#include <type_traits>

#if defined(__GFX11__) || defined(__GFX12__)
  #define __HIP__GFX1X__
#endif

#define LDS_SIZE (64 * 1024)

template <typename T>
__device__ __forceinline__ T loadnt(T* addr) {
  return __builtin_nontemporal_load(addr);
}

__device__ inline unsigned int min__(uint32_t a, uint32_t b) { return min(a, b); }

#define DOT2C(V0, V2, V3)                                                     \
  V0 = __builtin_amdgcn_fdot2(*((half2*)(&(V2))), *((half2*)(&(V3))), V0, false);

#if defined(__HIP__GFX1X__)
  #define REDUCE_SUM_DPP_WAVE32(val)                                        \
    do {                                                                    \
      val += __builtin_amdgcn_mov_dpp(val, 0x118, 0xf, 0xf, 1); /* shr8  */ \
      val += __builtin_amdgcn_mov_dpp(val, 0x114, 0xf, 0xf, 1); /* shr4  */ \
      val += __builtin_amdgcn_mov_dpp(val, 0x112, 0xf, 0xf, 1); /* shr2  */ \
      val += __builtin_amdgcn_mov_dpp(val, 0x111, 0xf, 0xf, 1); /* shr1  */ \
      val += __shfl_xor(val, 16);                                           \
    } while (0)
#endif

// K-elements per thread per step. Weights: 2 int4/byte, [M, K/2] bytes.
#if defined(__HIP__GFX1X__)
template <int THRDS, int YTILE, int WvPrGrp, int A_CHUNK, int UNRL, int N>
__global__ void __launch_bounds__(WvPrGrp* THRDS)
    wv_int4_sml(const int K, const int M, const uint8_t* B_packed,
                const half* __restrict__ A, const half* scale, half* C,
                const int _WvPrGrp, const int CuCount) {
  constexpr int max_lds_len = LDS_SIZE / 2;
  const int K_packed = K / 2;

  union bigTypeA {
    half h[A_CHUNK];
    float f[A_CHUNK / 2];
  };
  union bigTypeW {
    uint8_t b[A_CHUNK / 2];
    uint32_t u32[A_CHUNK / 8];
    float f[A_CHUNK / 8];
  };

  __shared__ half s[max_lds_len];

  for (uint32_t k = 0; k < min__(K * N, max_lds_len);
       k += THRDS * WvPrGrp * A_CHUNK) {
    uint32_t k_in = k + ((threadIdx.y * THRDS + threadIdx.x) * A_CHUNK);
    if (k_in >= min__(K * N, max_lds_len)) break;
    *((bigTypeA*)(&s[k_in])) = *((bigTypeA*)(&A[k_in]));
  }
  __syncthreads();

  if (threadIdx.y >= _WvPrGrp) return;

  uint32_t m = (blockIdx.x * _WvPrGrp + (threadIdx.y % _WvPrGrp)) * YTILE;
  float sum[N][YTILE];

  while (m < M) {
    for (int i = 0; i < YTILE; i++)
      for (int n = 0; n < N; n++) sum[n][i] = 0;

    bigTypeA bigA[N][UNRL];
    bigTypeW bigB[YTILE][UNRL];

    for (uint32_t k1 = 0; k1 < K; k1 += THRDS * A_CHUNK * UNRL) {
#pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + threadIdx.x * A_CHUNK;
        if (k_ >= (uint32_t)K) break;
        const uint8_t* B_ = &B_packed[(m + 0) * K_packed + k_ / 2];
        for (int y = 0; y < YTILE; y++) {
          const float* src = (const float*)(&B_[y * K_packed]);
#pragma unroll
          for (int i = 0; i < A_CHUNK / 8; i++)
            bigB[y][k2].f[i] = loadnt((float*)&src[i]);
        }
      }
#pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + threadIdx.x * A_CHUNK;
        if (k_ >= (uint32_t)K) break;
        for (int n = 0; n < N; n++)
          bigA[n][k2] = *((const bigTypeA*)(&(s[k_ + K * n])));
      }
#pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + threadIdx.x * A_CHUNK;
        if (k_ >= (uint32_t)K) break;
#pragma unroll
        for (uint32_t n = 0; n < N; n++) {
#pragma unroll
          for (int y = 0; y < YTILE; y++) {
            bigTypeA cvtB;
            constexpr uint32_t FP16_MAGIC = 0x64006400u;
            constexpr uint32_t BIAS_LO = 0x64086408u;  // symmetric: bake -8
            constexpr uint32_t SCALE16 = 0x2C002C00u;
            constexpr uint32_t BIAS_HI = 0xD480D480u;
#pragma unroll
            for (uint32_t w = 0; w < A_CHUNK / 8; w++) {
              uint32_t qa = bigB[y][k2].u32[w];
              uint32_t lo0 = (qa & 0x000F000Fu) | FP16_MAGIC;
              uint32_t hi0 = (qa & 0x00F000F0u) | FP16_MAGIC;
              qa >>= 8;
              uint32_t lo1 = (qa & 0x000F000Fu) | FP16_MAGIC;
              uint32_t hi1 = (qa & 0x00F000F0u) | FP16_MAGIC;
              *(half2*)&cvtB.f[w * 4 + 0] =
                  __hsub2(*(half2*)&lo0, *(const half2*)&BIAS_LO);
              *(half2*)&cvtB.f[w * 4 + 1] = __hfma2(
                  *(half2*)&hi0, *(const half2*)&SCALE16, *(const half2*)&BIAS_HI);
              *(half2*)&cvtB.f[w * 4 + 2] =
                  __hsub2(*(half2*)&lo1, *(const half2*)&BIAS_LO);
              *(half2*)&cvtB.f[w * 4 + 3] = __hfma2(
                  *(half2*)&hi1, *(const half2*)&SCALE16, *(const half2*)&BIAS_HI);
            }
#pragma unroll
            for (uint32_t b = 0; b < A_CHUNK / 2; b++)
              DOT2C(sum[n][y], bigA[n][k2].f[b], cvtB.f[b])
          }
        }
      }
    }

    for (int n = 0; n < N; n++)
      for (int y = 0; y < YTILE; y++) REDUCE_SUM_DPP_WAVE32(sum[n][y]);

    if (threadIdx.x == (THRDS - 1)) {
      for (int n = 0; n < N; n++)
        for (int i = 0; i < YTILE; i++) {
          sum[n][i] *= __half2float(scale[m + i]);
          C[m + i + n * M] = __float2half(sum[n][i]);
        }
    }
    m += CuCount * _WvPrGrp * YTILE;
  }
}
#else   // host pass / non-GFX1X: stub so the launcher still links
template <int THRDS, int YTILE, int WvPrGrp, int A_CHUNK, int UNRL, int N>
__global__ void wv_int4_sml(const int, const int, const uint8_t*, const half*,
                            const half*, half*, const int, const int) {}
#endif

// ---- repack: colibri fmt=2 (signed s4, [O,K/2] sequential) -> kernel layout ----
// Per group of 8 columns (4 src bytes -> 4 dst bytes). Undo the upload XOR 0x88
// (back to uint4b8) then place nibbles so the decode scramble [0,4,1,5,2,6,3,7]
// lands natural column i at logical position i.
__global__ static void repack_int4(uint8_t* dst, const uint8_t* src, int O,
                                   int K_packed) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;  // group index
  int groups = K_packed / 4;                                   // 8 cols / group
  size_t total = (size_t)O * groups;
  if (idx >= total) return;
  size_t base = (idx / groups) * K_packed + (idx % groups) * 4;
  uint8_t sb0 = src[base + 0] ^ 0x88, sb1 = src[base + 1] ^ 0x88;
  uint8_t sb2 = src[base + 2] ^ 0x88, sb3 = src[base + 3] ^ 0x88;
  dst[base + 0] = (sb0 & 0x0F) | ((sb1 & 0x0F) << 4);
  dst[base + 1] = (sb2 & 0x0F) | ((sb3 & 0x0F) << 4);
  dst[base + 2] = (sb0 >> 4) | ((sb1 >> 4) << 4);
  dst[base + 3] = (sb2 >> 4) | ((sb3 >> 4) << 4);
}

__global__ static void f32_to_f16(half* dst, const float* src, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = __float2half(src[i]);
}
__global__ static void f16_to_f32(float* dst, const half* src, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = __half2float(src[i]);
}

// ponytail: tuning knob. Default YTILE=4,UNRL=2 (gfx1100 sml winner per the
// vLLM handoff — its sYT heuristic is miscalibrated for 96 CUs). Override with
// COLI_INT4_YTILE / COLI_INT4_UNRL to sweep.
static int env_int(const char* k) { const char* e = getenv(k); return e ? atoi(e) : 0; }

#define LAUNCH_SML(YT, UN, NN)                                              \
  wv_int4_sml<32, YT, 16, 16, UN, NN><<<grid, block, 0, stream>>>(         \
      K, M, Wpacked, A, scale, C, wvprgrp, cu_count)

// Returns 1 if launched, 0 if the shape is unsupported (caller falls back).
extern "C" int coli_hip_int4_gemv(half* C, const half* A, const uint8_t* Wpacked,
                                  const half* scale, int Nout /*M*/, int K,
                                  int Mrows /*N=S*/, int cu_count,
                                  hipStream_t stream) {
  const int M = Nout, N = Mrows;
  const int max_lds_len = LDS_SIZE / 2;
  if (K % 16 != 0 || (long)K * N > max_lds_len || N < 1 || N > 8) return 0;

  int YT = env_int("COLI_INT4_YTILE"), UN = env_int("COLI_INT4_UNRL");
  if (YT != 1 && YT != 2 && YT != 4) YT = 4;
  if (UN != 1 && UN != 2 && UN != 4) UN = 2;
  if (M % YT != 0) { YT = 1; UN = 4; }        // YTILE must divide M
  if (M % YT != 0) return 0;

  dim3 grid(cu_count);
  dim3 block(32, 16);
  int wvprgrp = 16;

#define DISPATCH_N(YT_, UN_)          \
  switch (N) {                        \
    case 1: LAUNCH_SML(YT_, UN_, 1); break; \
    case 2: LAUNCH_SML(YT_, UN_, 2); break; \
    case 3: LAUNCH_SML(YT_, UN_, 3); break; \
    case 4: LAUNCH_SML(YT_, UN_, 4); break; \
    case 5: LAUNCH_SML(YT_, UN_, 5); break; \
    case 6: LAUNCH_SML(YT_, UN_, 6); break; \
    case 7: LAUNCH_SML(YT_, UN_, 7); break; \
    default: LAUNCH_SML(YT_, UN_, 8); break; \
  }
  if (YT == 4 && UN == 2) { DISPATCH_N(4, 2) }
  else if (YT == 4 && UN == 1) { DISPATCH_N(4, 1) }
  else if (YT == 4 && UN == 4) { DISPATCH_N(4, 4) }
  else if (YT == 2 && UN == 2) { DISPATCH_N(2, 2) }
  else if (YT == 2 && UN == 1) { DISPATCH_N(2, 1) }
  else if (YT == 2 && UN == 4) { DISPATCH_N(2, 4) }
  else if (YT == 1 && UN == 1) { DISPATCH_N(1, 1) }
  else if (YT == 1 && UN == 2) { DISPATCH_N(1, 2) }
  else { DISPATCH_N(1, 4) }
  return 1;
}

// Repack colibri signed-s4 weights [O,K/2] -> kernel layout, in place-compatible
// (dst may differ from src). K must be a multiple of 16.
extern "C" void coli_hip_int4_repack(uint8_t* dst, const uint8_t* src, int O,
                                     int K, hipStream_t stream) {
  int K_packed = K / 2;
  size_t groups = (size_t)O * (K_packed / 4);
  repack_int4<<<(unsigned)((groups + 255) / 256), 256, 0, stream>>>(dst, src, O,
                                                                    K_packed);
}
extern "C" void coli_hip_f32_to_f16(half* dst, const float* src, size_t n,
                                    hipStream_t stream) {
  f32_to_f16<<<(unsigned)((n + 255) / 256), 256, 0, stream>>>(dst, src, n);
}
extern "C" void coli_hip_f16_to_f32(float* dst, const half* src, size_t n,
                                    hipStream_t stream) {
  f16_to_f32<<<(unsigned)((n + 255) / 256), 256, 0, stream>>>(dst, src, n);
}

// ---- WMMA batched int4 GEMM (RDNA3 matrix cores), for rows>8 ----
// C[rows,Nout] = A[rows,K] @ dequant(W)^T. W is colibri signed-s4 [Nout,K/2]
// (after the upload XOR); the per-channel fp32 scale is applied at store time,
// so dequant is just nibble->fp16 of a small signed int (exact).
//
// One wave owns one 16x16 output tile (out-tile blockIdx.x*WV+wave, row-tile
// blockIdx.y) -> parallelism scales with both output width and rows, which is
// what keeps the GPU busy (this kernel is occupancy/latency-bound, not
// bandwidth-bound). When that grid is still too small (narrow output at low
// rows) K is split across blockIdx.z (grid.z=G): each slice atomic-adds its
// partial into an fp32 accumulator, scaled in a second pass. WV waves per block
// share the LDS-staged activation; BK amortizes the barrier over several WMMA-K.
#if defined(__HIP__GFX1X__)
#include <rocwmma/rocwmma.hpp>
namespace rw = rocwmma;

template <int WV, int BK>
__global__ void __launch_bounds__(WV * 32)
    wmma_int4(const int Nout, const int K, const int rows, const int Kchunk,
              const uint8_t* __restrict__ W, const half* __restrict__ A,
              const float* __restrict__ scale, float* __restrict__ Cf,
              half* __restrict__ C) {
  const int wave = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int nbase = (blockIdx.x * WV + wave) * 16;
  const int row0 = blockIdx.y * 16;
  const int Kp = K / 2;
  const int kstart = blockIdx.z * Kchunk;
  const int kend = min(kstart + Kchunk, K);
  __shared__ rw::float16_t lA[2][16 * BK];
  __shared__ rw::float16_t lB[2][WV][BK * 16];
  __shared__ float lC[WV][16 * 16];

  constexpr int APT = (16 * BK + WV * 32 - 1) / (WV * 32);  // A elems / thread
  constexpr int BPT = (BK * 16 + 31) / 32;                  // B elems / lane
  rw::float16_t rA[APT], rB[BPT];

  rw::fragment<rw::accumulator, 16, 16, 16, float> acc;
  rw::fill_fragment(acc, 0.0f);

  // Read one K-tile (A load + int4 weight dequant) into registers.
#define WMMA_LOAD_REG(k0)                                                     \
  do {                                                                        \
    for (int j = 0; j < APT; j++) {                                           \
      int idx = threadIdx.x + j * WV * 32;                                    \
      if (idx < 16 * BK) {                                                    \
        int m = idx / BK, k = idx % BK, row = row0 + m;                       \
        rA[j] = (row < rows) ? (rw::float16_t)A[(size_t)row * K + (k0) + k]   \
                             : (rw::float16_t)0.0f;                           \
      }                                                                       \
    }                                                                         \
    for (int j = 0; j < BPT; j++) {                                           \
      int idx = lane + j * 32;                                                \
      if (idx < BK * 16) {                                                    \
        int k = idx >> 4, n = idx & 15, out = nbase + n, kk = (k0) + k;       \
        uint8_t byte = W[(size_t)out * Kp + (kk >> 1)];                       \
        int n4 = (kk & 1) ? (byte >> 4) : (byte & 15);                        \
        rB[j] = (rw::float16_t)(float)((n4 & 8) ? n4 - 16 : n4);              \
      }                                                                       \
    }                                                                         \
  } while (0)
#define WMMA_STORE_REG(buf)                                                   \
  do {                                                                        \
    for (int j = 0; j < APT; j++) {                                           \
      int idx = threadIdx.x + j * WV * 32;                                    \
      if (idx < 16 * BK) lA[buf][idx] = rA[j];                                \
    }                                                                         \
    for (int j = 0; j < BPT; j++) {                                           \
      int idx = lane + j * 32;                                                \
      if (idx < BK * 16) lB[buf][wave][idx] = rB[j];                          \
    }                                                                         \
  } while (0)

  const int nsteps = (kend - kstart + BK - 1) / BK;
  WMMA_LOAD_REG(kstart);        // prologue: tile 0 -> registers -> buf 0
  WMMA_STORE_REG(0);
  __syncthreads();

  for (int s = 0; s < nsteps; s++) {
    int cur = s & 1;
    if (s + 1 < nsteps) WMMA_LOAD_REG(kstart + (s + 1) * BK);  // prefetch next
#pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {                      // compute current
      rw::fragment<rw::matrix_a, 16, 16, 16, rw::float16_t, rw::row_major> af;
      rw::fragment<rw::matrix_b, 16, 16, 16, rw::float16_t, rw::row_major> bf;
      rw::load_matrix_sync(af, lA[cur] + kk, BK);
      rw::load_matrix_sync(bf, lB[cur][wave] + kk * 16, 16);
      rw::mma_sync(acc, af, bf, acc);
    }
    if (s + 1 < nsteps) WMMA_STORE_REG((s + 1) & 1);          // stash next
    __syncthreads();
  }
#undef WMMA_LOAD_REG
#undef WMMA_STORE_REG

  rw::store_matrix_sync(lC[wave], acc, 16, rw::mem_row_major);
  __syncthreads();
  for (int idx = lane; idx < 256; idx += 32) {
    int m = idx >> 4, n = idx & 15, row = row0 + m;
    if (row < rows) {
      if (Cf) atomicAdd(&Cf[(size_t)row * Nout + nbase + n], lC[wave][idx]);
      else C[(size_t)row * Nout + nbase + n] = (half)(lC[wave][idx] * scale[nbase + n]);
    }
  }
}

__global__ static void wmma_scale_cols(half* C, const float* Cf,
                                       const float* scale, size_t n, int Nout) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) C[i] = (half)(Cf[i] * scale[i - (i / Nout) * Nout]);
}
#else
template <int WV, int BK>
__global__ void wmma_int4(const int, const int, const int, const int,
                          const uint8_t*, const half*, const float*, float*, half*) {}
__global__ static void wmma_scale_cols(half*, const float*, const float*, size_t, int) {}
#endif

// fp32 split-K accumulator scratch (kernel-launch-serialized per stream).
static float* g_wmma_cf = nullptr;
static size_t g_wmma_cf_cap = 0;
static float* wmma_scratch(size_t n) {
  if (n > g_wmma_cf_cap) {
    if (g_wmma_cf) hipFree(g_wmma_cf);
    if (hipMalloc(&g_wmma_cf, n * sizeof(float)) != hipSuccess) { g_wmma_cf_cap = 0; g_wmma_cf = nullptr; return nullptr; }
    g_wmma_cf_cap = n;
  }
  return g_wmma_cf;
}

// Returns 1 if launched, 0 if the shape is unsupported (caller falls back).
extern "C" int coli_hip_int4_wmma(half* C, const half* A, const uint8_t* W,
                                  const float* scale, int Nout, int K, int rows,
                                  hipStream_t stream) {
  constexpr int WV = 4, BK = 32;
  if (K % BK || Nout % (16 * WV) || rows < 1) return 0;
  const int base = Nout / (16 * WV);
  const int rowtiles = (rows + 15) / 16;
  int blocks0 = base * rowtiles;
  int G = blocks0 >= 256 ? 1 : (256 + blocks0 - 1) / blocks0;
  int maxG = K / BK; if (G > maxG) G = maxG; if (G < 1) G = 1;
  int Kchunk = ((K / G + BK - 1) / BK) * BK;
  G = (K + Kchunk - 1) / Kchunk;
  dim3 grid((unsigned)base, (unsigned)rowtiles, (unsigned)G), block(WV * 32);
  if (G == 1) {
    wmma_int4<WV, BK><<<grid, block, 0, stream>>>(Nout, K, rows, Kchunk, W, A, scale, nullptr, C);
  } else {
    size_t nn = (size_t)rows * Nout;
    float* Cf = wmma_scratch(nn);
    if (!Cf) return 0;
    hipMemsetAsync(Cf, 0, nn * sizeof(float), stream);
    wmma_int4<WV, BK><<<grid, block, 0, stream>>>(Nout, K, rows, Kchunk, W, A, scale, Cf, nullptr);
    wmma_scale_cols<<<(unsigned)((nn + 255) / 256), 256, 0, stream>>>(C, Cf, scale, nn, Nout);
  }
  return 1;
}

#endif  // COLI_HIP
