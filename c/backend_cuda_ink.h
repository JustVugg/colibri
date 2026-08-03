/* Minimal CUDA backend for inkling.c: bf16 resident weights live in VRAM,
 * matmuls run on-device. Motivation is bandwidth, not FLOPs: decode reads
 * ~35 GB of bf16 residents per token, which caps CPU decode at the DDR5
 * bandwidth (~1.2 s/token); the GPU's VRAM feeds the same reads at ~12x.
 * Deliberately tiny API — expert streaming, attention and everything else
 * stay on the CPU in this phase. */
#ifndef BACKEND_CUDA_INK_H
#define BACKEND_CUDA_INK_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

int ink_cuda_init(int dev); /* 0 = ok */
size_t ink_cuda_free_bytes(void);
void *ink_cuda_upload(const void *h, size_t n); /* NULL = OOM/error */
/* y[S,O] = x[S,I] @ W^T, W = device bf16 [O,I]; x,y host f32. 0 = ok */
int ink_cuda_matmul_bf16(float *y, const float *x, const void *W, int S, int I, int O);

#ifdef __cplusplus
}
#endif
#endif
