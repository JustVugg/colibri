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

int    ink_cuda_init(int dev);                 /* 0 = ok */
size_t ink_cuda_free_bytes(void);
void  *ink_cuda_upload(const void *h, size_t n);   /* NULL = OOM/error */
/* an uploaded matrix's bytes back to the host (0 = ok), and its memory freed: Inkling-2's CUDA chain
 * moves the matrices to the main backend's tensors (inkling2_cuchain.h) */
int    ink_cuda_download(void *h, const void *d, size_t n);
void   ink_cuda_free(void *d);
/* y[S,O] = x[S,I] @ W^T, W = device bf16 [O,I]; x,y host f32. 0 = ok */
int    ink_cuda_matmul_bf16(float *y, const float *x, const void *W, int S, int I, int O);

/* The same six as a table, the one name coli_cuda.dll exports for them (Windows, CUDA_DLL=1):
 * backend_loader.c's ink_cuda_* forward through it, and a DLL without it leaves inkling.exe on
 * Vulkan or the CPU. */
typedef struct {
    int    (*init)(int dev);
    size_t (*free_bytes)(void);
    void  *(*upload)(const void *h, size_t n);
    int    (*download)(void *h, const void *d, size_t n);
    void   (*free)(void *d);
    int    (*matmul_bf16)(float *y, const float *x, const void *W, int S, int I, int O);
} ColiInkCudaOps;
#if defined(_WIN32) && defined(COLI_CUDA_BUILDING_DLL)
__declspec(dllexport)
#endif
const ColiInkCudaOps *coli_cuda_ink_ops(void);

#ifdef __cplusplus
}
#endif
#endif
