/* cuda_f8_mma.cuh -- tensor-core fmt=8 (block-FP8) grouped experts, COLI_CUDA_F8_MMA=1.
 *
 * Derived from TensorFold (https://github.com/ashhart/TensorFold, Apache-2.0,
 * Copyright 2026 TensorFold contributors; see NOTICE and THIRD_PARTY_NOTICES.md):
 *   zig/kernels/cuda/fp8_experts.cu   fp8_expert_prompt_kernel's scheme: e4m3 pairs widened
 *                                     to bf16 in registers, bf16 mma.sync m16n8k16, a part
 *                                     per 128-k scale block folded into the accumulator
 *   zig/kernels/cuda/common/fp_pairs.cuh   fp8pair
 *   zig/kernels/cuda/common/qmm_frag.cuh   mma
 *   zig/kernels/cuda/common/cp_async.cuh   cp16, cp16z, cp_commit, cp_wait
 *   zig/kernels/cuda/experts.cuh           comp, the swiglu epilogue's shape
 * MODIFIED for colibri, 2026-11-17:
 *   - reads colibri's fmt=8 layout as uploaded (row-major e4m3 [O][K], f32 scales
 *     [O/128][K/128]) through a cp.async tile instead of TensorFold's lane-order repack
 *     (f8_mma.h says why that is the same permutation); 128 output columns a block, so a
 *     block's scale is one value per 128-k step;
 *   - colibri's GroupDesc table (expert c's rows contiguous at offset) instead of
 *     items/counts/members; one launch for gate+up (silu(g)*u in the epilogue, f32 like
 *     grouped_hidden_f8w_dual, then stored as bf16 for the down kernel), one for down (f32);
 *   - activations staged from a bf16 copy of the f32 input (f8m_to_bf16);
 *   - no SwiGLU clamp, no bf16 rounding of g and u before silu (colibri's f32 epilogue),
 *     no native e4m3x2 cvt path (the product builds plain sm_120, not the family target).
 *
 * Included by backend_cuda.cu after GroupDesc. CUDA only (PTX asm); the bodies need
 * sm_80 (bf16 mma.sync, cp.async, cvt.rn.bf16x2) and trap below it -- the host gate
 * (f8m_eligible) never launches them there. */
#ifndef COLI_CUDA_F8_MMA_CUH
#define COLI_CUDA_F8_MMA_CUH

#include "f8_mma.h"

namespace f8mma {

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define F8M_SM80 1
#else
#define F8M_SM80 0
#endif

__device__ __forceinline__ uint32_t smem_u32(const void *p) { return (uint32_t)__cvta_generic_to_shared(p); }

__device__ __forceinline__ void cp16(void *dst, const void *src) {
#if F8M_SM80
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_u32(dst)), "l"(src) : "memory");
#endif
}
/* n = 0 zero-fills the 16 bytes (rows past the expert's own) */
__device__ __forceinline__ void cp16z(void *dst, const void *src, int n) {
#if F8M_SM80
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_u32(dst)), "l"(src), "r"(n) : "memory");
#endif
}
__device__ __forceinline__ void cp_commit() {
#if F8M_SM80
    asm volatile("cp.async.commit_group;\n" ::: "memory");
#endif
}
template <int N> __device__ __forceinline__ void cp_wait() {
#if F8M_SM80
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
#endif
}

/* two e4m3 codes (low two bytes of v) as a bf16x2 pair: f8m_pair_bits * 2^120, -0 addend */
__device__ __forceinline__ uint32_t fp8pair(uint32_t v) {
    uint32_t r = 0;
#if F8M_SM80
    asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(r) : "r"(f8m_pair_bits(v)), "r"(0x7B807B80u), "r"(0x80008000u));
#endif
    return r;
}

/* lo to the low half, RN */
__device__ __forceinline__ uint32_t bf16x2(float lo, float hi) {
    uint32_t r = 0;
#if F8M_SM80
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(r) : "f"(hi), "f"(lo));
#endif
    return r;
}

__device__ __forceinline__ void mma(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                    uint32_t b0, uint32_t b1) {
#if F8M_SM80
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

__device__ __forceinline__ uint32_t comp(const uint4 &v, int c) { return c == 0 ? v.x : c == 1 ? v.y : c == 2 ? v.z : v.w; }

} // namespace f8mma

/* f32 rows -> bf16 rows (RN), two values a thread; n2 = values / 2 */
__global__ static void f8m_to_bf16(uint32_t *dst, const float *src, size_t n2) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n2) { float2 v = ((const float2 *)src)[i]; dst[i] = f8mma::bf16x2(v.x, v.y); }
}

/* One block: expert blockIdx.y, output columns [128*blockIdx.x, +128) (one scale-block
 * row; warp w owns 32 of them, four n8 tiles), rows [16*RT*blockIdx.z, +16*RT) of that
 * expert. K walks in 64-byte stages, double-buffered through cp.async: the block's 128
 * weight rows x 64 bytes per matrix (rows of the fmt=8 layout as uploaded, 16-B chunks,
 * coalesced), and 16*RT activation rows x 64 bf16. Per 128-k scale block the MMAs fill
 * `part` from zero, then acc += part * scale (fmaf): TensorFold's convention, f32 across
 * blocks where the LUT kernels use double (the tolerance in tests/test_f8_mma_cuda.cu
 * accounts for both).
 * M = 2: gate (d.g) and up (d.u) against x [.][K=D], O = I; h = silu(g)*u, f32, stored as
 *        bf16 to hb [total][I].
 * M = 1: down (d.d) against hb [.][K=I], O = D; f32 to y [total][D]. */
template <int M, int RT>
__global__ static void __launch_bounds__(F8M_WARPS * 32)
f8m_group(const uint16_t *__restrict__ xb, const GroupDesc *__restrict__ desc, int K, int O,
          float *__restrict__ y, uint32_t *__restrict__ hb) {
#if F8M_SM80
    constexpr int BM = 16 * RT;
    __shared__ __align__(16) uint8_t sw[2][M][F8M_COLS * F8M_KSTEP];
    __shared__ __align__(16) uint16_t sx[2][BM * F8M_XSTRIDE];
    const GroupDesc d = desc[blockIdx.y];
    const int r0 = blockIdx.z * BM;
    if (r0 >= d.rows) return;   /* whole block, before any barrier */
    const int nb0 = blockIdx.x * F8M_COLS;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gq = lane >> 2, t = lane & 3;
    /* matrix 0 / 1 picked by select, not an indexed local array (that one went to the
     * stack: 16 bytes of frame in ptxas -v) */
    const uint8_t *w0 = (const uint8_t *)(M == 2 ? d.g : d.d), *w1 = (const uint8_t *)d.u;
    const float *s0 = M == 2 ? d.gs : d.ds, *s1 = d.us;
    const int nrows = d.rows - r0 < BM ? d.rows - r0 : BM;
    const uint16_t *xr0 = xb + (size_t)(d.offset + r0) * K;
    const int nsteps = K / F8M_KSTEP;

    auto stage = [&](int s) {
        const int buf = s & 1, k0 = s * F8M_KSTEP;
        for (int i = threadIdx.x; i < M * F8M_COLS * 4; i += F8M_WARPS * 32) {
            const int m = i / (F8M_COLS * 4), row = (i >> 2) & (F8M_COLS - 1), ch = i & 3;
            f8mma::cp16(&sw[buf][m][f8m_wsh(row, ch)], (m ? w1 : w0) + (size_t)(nb0 + row) * K + k0 + 16 * ch);
        }
        for (int i = threadIdx.x; i < BM * 8; i += F8M_WARPS * 32) {
            const int row = i >> 3, ch = i & 7, in = row < nrows;
            f8mma::cp16z(&sx[buf][row * F8M_XSTRIDE + 8 * ch], xr0 + (size_t)(in ? row : 0) * K + k0 + 8 * ch,
                         in ? 16 : 0);
        }
    };

    float acc[M][RT][4][4], part[M][RT][4][4];
#pragma unroll
    for (int m = 0; m < M; m++)
#pragma unroll
        for (int r = 0; r < RT; r++)
#pragma unroll
            for (int j = 0; j < 4; j++)
#pragma unroll
                for (int i = 0; i < 4; i++) acc[m][r][j][i] = part[m][r][j][i] = 0.f;

    stage(0);
    f8mma::cp_commit();
    for (int s = 0; s < nsteps; s++) {
        if (s + 1 < nsteps) stage(s + 1);
        f8mma::cp_commit();   /* possibly empty: keeps wait_group's count uniform */
        f8mma::cp_wait<1>();
        __syncthreads();
        const int buf = s & 1;
        if (!(s & 1)) {
#pragma unroll
            for (int m = 0; m < M; m++)
#pragma unroll
                for (int r = 0; r < RT; r++)
#pragma unroll
                    for (int j = 0; j < 4; j++)
#pragma unroll
                        for (int i = 0; i < 4; i++) part[m][r][j][i] = 0.f;
        }
        uint4 wv[M][4];
#pragma unroll
        for (int m = 0; m < M; m++)
#pragma unroll
            for (int j = 0; j < 4; j++)
                wv[m][j] = *(const uint4 *)&sw[buf][m][f8m_wsh(32 * warp + 8 * j + gq, t)];
#pragma unroll
        for (int qq = 0; qq < 2; qq++) {
            uint4 xa[RT], xc[RT];   /* rows gq and gq+8 of each m16 tile */
#pragma unroll
            for (int r = 0; r < RT; r++) {
                xa[r] = *(const uint4 *)&sx[buf][f8m_xsh(16 * r + gq, t, qq)];
                xc[r] = *(const uint4 *)&sx[buf][f8m_xsh(16 * r + gq + 8, t, qq)];
            }
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const int q = 2 * qq + h;
#pragma unroll
                for (int m = 0; m < M; m++)
#pragma unroll
                    for (int j = 0; j < 4; j++) {
                        const uint32_t word = f8mma::comp(wv[m][j], q);
                        const uint32_t b0 = f8mma::fp8pair(word), b1 = f8mma::fp8pair(word >> 16);
#pragma unroll
                        for (int r = 0; r < RT; r++)
                            f8mma::mma(part[m][r][j], h ? xa[r].z : xa[r].x, h ? xc[r].z : xc[r].x,
                                       h ? xa[r].w : xa[r].y, h ? xc[r].w : xc[r].y, b0, b1);
                    }
            }
        }
        if (s & 1) {
#pragma unroll
            for (int m = 0; m < M; m++) {
                const float sc = __ldg((m ? s1 : s0) + f8m_scale_at(nb0, s >> 1, K));
#pragma unroll
                for (int r = 0; r < RT; r++)
#pragma unroll
                    for (int j = 0; j < 4; j++)
#pragma unroll
                        for (int i = 0; i < 4; i++) acc[m][r][j][i] = fmaf(part[m][r][j][i], sc, acc[m][r][j][i]);
            }
        }
        __syncthreads();   /* the next iteration's stage(s+2) overwrites this buffer */
    }
    f8mma::cp_wait<0>();

#pragma unroll
    for (int r = 0; r < RT; r++)
#pragma unroll
        for (int hh = 0; hh < 2; hh++) {
            const int row = 16 * r + f8m_c_row(gq, 2 * hh);
            if (row >= nrows) continue;
            const size_t zr = (size_t)(d.offset + r0 + row) * O;
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const int col = nb0 + f8m_c_col(warp, j, t, 0);
                const float a0 = acc[0][r][j][2 * hh], a1 = acc[0][r][j][2 * hh + 1];
                if (M == 2) {
                    /* grouped_hidden_f8w_dual's epilogue, f32 */
                    const float u0 = acc[M - 1][r][j][2 * hh], u1 = acc[M - 1][r][j][2 * hh + 1];
                    const float h0 = (a0 / (1.f + expf(-a0))) * u0, h1 = (a1 / (1.f + expf(-a1))) * u1;
                    hb[(zr + col) >> 1] = f8mma::bf16x2(h0, h1);
                } else {
                    *(float2 *)(y + zr + col) = make_float2(a0, a1);
                }
            }
        }
#else
    (void)xb; (void)desc; (void)K; (void)O; (void)y; (void)hb;
    __trap();   /* below sm_80: f8m_eligible keeps the host from launching this */
#endif
}

#endif
