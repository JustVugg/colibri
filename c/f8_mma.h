/* f8_mma.h -- the host-checkable half of the fmt=8 tensor-core expert path
 * (COLI_CUDA_F8_MMA, backend_cuda.cu; kernels in cuda_f8_mma.cuh).
 *
 * Plain C, no CUDA headers: backend_cuda.cu includes it for the kernels and the
 * dispatch gate, tests/test_f8_mma_idx.c for a CPU model of the same index maps
 * (fragment layout, staged-tile offsets, bank conflicts, the gate), which runs
 * in the CPU suite on a box with no GPU.
 *
 * Derived from TensorFold (Apache-2.0, Copyright 2026 TensorFold contributors, see
 * NOTICE and THIRD_PARTY_NOTICES.md), MODIFIED for colibri 2026-11-17 (TensorFold
 * has no such header: these are its kernels' index expressions, named, plus
 * colibri's dispatch gate).
 * The fragment scheme is TensorFold's fp8_experts.cu: one lane's 32-bit weight word holds four consecutive
 * k bytes of one output row, and the A operand uses the SAME permutation of k
 * within a 16-k MMA step, so the dot is unchanged. TensorFold repacks the
 * weights into lane order at load time (Fp8Experts.packOne); its pack is a pure
 * permutation of 4-byte words of the row-major [O][K] matrix, so here the
 * kernels read colibri's existing fmt=8 layout (e4m3 [O][K] + f32 scales
 * [ceil(O/128)][ceil(K/128)]) through a cp.async stage instead: no repack, no
 * second VRAM copy, no change to the tier's byte accounting. */
#ifndef COLI_F8_MMA_H
#define COLI_F8_MMA_H

#include <stdint.h>
#include <stdlib.h>

#if defined(__CUDACC__)
#define F8M_FN __host__ __device__ __forceinline__
#else
#define F8M_FN static inline
#endif

#define F8M_KSTEP   64    /* k bytes one pipeline stage holds (two per 128-k scale block) */
#define F8M_WARPS   4     /* warps a block, each 32 output columns (four n8 tiles) */
#define F8M_COLS    128   /* output columns a block: exactly one scale-block row */
#define F8M_XSTRIDE 72    /* bf16 a staged activation row: 144 B, so a 16-B lane load
                           * of rows r and r+1 lands on disjoint banks (9 chunks a row) */
#define F8M_DEFAULT_MIN_ROWS 8

/* Physical k (0..63, within one stage) of element e (0,1) of B register b (0 = b0, 1 = b1)
 * for lane t = lane&3 at k16 sub-step q (0..3). The A operand uses the same map: a0/a1
 * hold (b=0), a2/a3 (b=1). mma.m16n8k16's logical k 2t+e lands on 16t+4q+e, logical
 * 2t+8+e on 16t+4q+2+e: sub-step q takes bytes 4q..4q+3 of each lane's 16-byte chunk. */
F8M_FN int f8m_k(int t, int q, int b, int e) { return 16 * t + 4 * q + 2 * b + e; }

/* bf16 offset, in a staged activation tile, of the 16-byte A load of row `row`, lane t,
 * half qq (0: sub-steps 0,1; 1: sub-steps 2,3) */
F8M_FN int f8m_xsh(int row, int t, int qq) { return row * F8M_XSTRIDE + 16 * t + 8 * qq; }
/* byte offset, in a staged weight tile (128 rows x 64 B), of lane t's 16 bytes of row `row` */
F8M_FN int f8m_wsh(int row, int t) { return row * F8M_KSTEP + 16 * t; }
/* output coordinates of accumulator element i (0..3) of n8 tile j, lane (gq,t), warp w */
F8M_FN int f8m_c_row(int gq, int i) { return gq + 8 * (i >> 1); }
F8M_FN int f8m_c_col(int w, int j, int t, int i) { return 32 * w + 8 * j + 2 * t + (i & 1); }
/* the fmt=8 scale of output block row n/128, input block kb (absorb_scale's geometry) */
F8M_FN size_t f8m_scale_at(int n, int kb, int K) { return (size_t)(n >> 7) * (size_t)((K + 127) >> 7) + (size_t)kb; }

/* TensorFold's fp8pair, first half: the low two e4m3 bytes of v moved into the two bf16
 * halves (sign to bit 15, exponent+mantissa to bits 4..10). Read as bf16 that is the
 * code times 2^-120; the kernel multiplies by 2^120 (0x7B80) with a bf16x2 fma, exact
 * for every finite code, subnormals included. NaN codes (0x7F, 0xFF) come out as +-480,
 * NOT NaN: the LUT kernels propagate a NaN byte, this path does not. */
F8M_FN uint32_t f8m_pair_bits(uint32_t v) {
    return ((v & 0x7Fu) << 4) | ((v & 0x80u) << 8) | ((v & 0x7F00u) << 12) | ((v & 0x8000u) << 16);
}

/* Row tiles (m16) a block takes: the dual gate/up kernel holds two accumulator sets per
 * tile, so it stops at two (32 rows), the down kernel at four (64 rows). */
F8M_FN int f8m_row_tiles(int M, int max_rows) {
    if (max_rows <= 16) return 1;
    if (M == 2 || max_rows <= 32) return 2;
    return 4;
}
F8M_FN int f8m_row_blocks(int M, int max_rows) {
    int bm = 16 * f8m_row_tiles(M, max_rows);
    return (max_rows + bm - 1) / bm;
}

/* Strict parse like f8_warp_mode: unset, empty or non-numeric selects the default. */
static inline int f8m_parse(const char *e, int dflt) {
    if (!e || !*e) return dflt;
    char *end; long v = strtol(e, &end, 10);
    return *end ? dflt : (int)v;
}

/* Every condition the dispatch checks, in one place. on: COLI_CUDA_F8_MMA; min_rows:
 * COLI_CUDA_F8_MMA_MIN_ROWS; cc: compute capability x10 (bf16 mma.sync, cp.async: sm_80);
 * D, I: the group's dims, both used as K, so both must be whole 128-k scale blocks;
 * max_rows: the largest member's rows (the m16 tile is per expert, so a call of many
 * one-row experts gains nothing); pin: a speculation's verify that must give a decode
 * step's bits (coli_cuda_expert_group_pinned); aligned: every weight base 16-B aligned
 * (cp.async). */
F8M_FN int f8m_eligible(int on, int min_rows, int cc, int D, int I, int max_rows, int pin, int aligned) {
    return on == 1 && cc >= 80 && !pin && aligned && D >= 128 && I >= 128 && D % 128 == 0 && I % 128 == 0 &&
           max_rows >= (min_rows < 1 ? 1 : min_rows);
}

#endif
