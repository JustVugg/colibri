/* Kimi K3 on a CUDA device, beside coli_cuda.dll's dense matrices: a prompt step's routed
 * experts (K3_GPU_EXPERTS), in the CPU's arithmetic to the bit -- the same integer sums of
 * four products a lane (dp4a), the same scale and fused add a group, the same lane sums,
 * SiTU through k3_exact.h's exp_f and tanhf, the same int8 quantization for down -- so an
 * expert computed here gives the bytes kimi_k3.c's k3_wave gives, and a step may split its
 * experts between the two in any way.
 *
 * One host thread drives the device (the engine's feeder); k3cu_host_alloc may come from any.
 * Every call returns 1 (or a count) on success, 0 or -1 on failure. */
#ifndef COLIBRI_BACKEND_CUDA_K3_H
#define COLIBRI_BACKEND_CUDA_K3_H
#include <stddef.h>
#include <stdint.h>
#if defined(_WIN32) && defined(K3CU_BUILDING_DLL)
#define K3CU_API __declspec(dllexport)
#else
#define K3CU_API
#endif
#ifdef __cplusplus
extern "C" {
#endif
K3CU_API int k3cu_init(int device);
K3CU_API size_t k3cu_free_bytes(void);
/* Pinned host memory the device writes to directly; NULL when there is none. */
K3CU_API void *k3cu_host_alloc(size_t bytes);
K3CU_API void k3cu_host_free(void *p);
/* Tests: the device's exp_f (kind 0) or tanhf (kind 1) of the floats whose bits are
 * first .. first+count-1, their bits into out. */
K3CU_API int k3cu_exact_eval(int kind, uint32_t first, uint32_t count, uint32_t *out);
/* A prompt step's routed experts.
 * begin: the step's n rows quantized as mxfp4_qx does (zq [n][LT], zs [n][LT/32]), copied
 *   up before it returns; LT and MI (multiples of 32), SiTU's b1 and b2; an expert's bytes
 *   (ebytes) laid out as the engine's slots hold them: off[] = w1p, w1s, w2p, w2s, w3p, w3s;
 *   Y [npairs][LT], pinned (k3cu_host_alloc): the experts' outputs land there.
 * expert: an expert's bytes (any host memory, free again once it returns), its r rows
 *   (indices into begin's rows), its outputs into Y's rows pair0 .. pair0+r-1; tag names it
 *   to poll. Waits while every device slot is busy.
 * poll: the tags of the experts whose outputs are complete since the last poll, up to cap;
 *   their count, -1 on a device error.
 * end: waits for every expert of the step; 1 when all of them completed. */
K3CU_API int k3cu_moe_begin(const int8_t *zq, const float *zs, int n, int LT, int MI, float b1, float b2,
                            size_t ebytes, const size_t *off, float *Y, int npairs);
K3CU_API int k3cu_moe_expert(const uint8_t *bytes, const int *rows, int r, int pair0, int tag);
K3CU_API int k3cu_moe_poll(int *tags, int cap);
/* wait: until the oldest expert in flight is complete; 1, or 0 when none is in flight. */
K3CU_API int k3cu_moe_wait(void);
K3CU_API int k3cu_moe_end(void);
/* y [S][O] = x [S][I] . W^T for a dense matrix the device does not hold, in the CPU's
 * arithmetic: fmt 4 int4 groups of gs (quant.h's matmul_i4_grouped; scale [O][I/gs]), fmt 1
 * int8 rows (matmul_q; a scale a row). Host memory in and out; 0 when it cannot (the CPU's). */
K3CU_API int k3cu_matmul(int fmt, float *y, const float *x, const void *w, const float *scale,
                         int S, int I, int O, int gs);
K3CU_API void k3cu_report(void);
K3CU_API void k3cu_shutdown(void);
#ifdef __cplusplus
}
#endif
#endif
