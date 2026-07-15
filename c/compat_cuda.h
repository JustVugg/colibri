/* compat_cuda.h — Runtime loader shim for coli_cuda.dll on Windows.
 *
 * On Linux/macOS the CUDA backend is compile-time linked (backend_cuda.o
 * produced by nvcc directly from the .cu). On Windows the CPU engine
 * builds with MinGW-w64 GCC, but nvcc requires MSVC as its host compiler,
 * so backend_cuda.cu is built as a separate MSVC-produced DLL and
 * loaded at runtime via LoadLibrary + GetProcAddress. C linkage is
 * stable across the MSVC/MinGW x64 ABI, so the boundary is safe.
 *
 * MUST be included AFTER "backend_cuda.h" in the same translation unit.
 * The backend_cuda.h declarations must be visible first so the macros
 * below (which redirect the CALL SITE only) do not accidentally rewrite
 * the function declarations themselves.
 *
 * On Linux/macOS this header is a no-op.
 */
#ifndef COMPAT_CUDA_H
#define COMPAT_CUDA_H

#if defined(_WIN32) && defined(COLI_CUDA)

#include "backend_cuda.h"
#include <stddef.h>

/* Function-pointer types matching each coli_cuda_* signature. */
typedef int    (*fn_coli_cuda_init)         (const int *devices, int count);
typedef void   (*fn_coli_cuda_shutdown)     (void);
typedef int    (*fn_coli_cuda_device_count) (void);
typedef int    (*fn_coli_cuda_device_at)    (int index);
typedef int    (*fn_coli_cuda_mem_info)     (int device, size_t *free_bytes, size_t *total_bytes);
typedef void   (*fn_coli_cuda_stats)        (int device, size_t *tensor_count, size_t *tensor_bytes);
typedef int    (*fn_coli_cuda_tensor_upload)(ColiCudaTensor **tensor, const void *weights,
                                             const float *scales, int fmt, int I, int O, int device);
typedef int    (*fn_coli_cuda_matmul)       (ColiCudaTensor **tensor, float *y, const float *x,
                                             const void *weights, const float *scales,
                                             int fmt, int S, int I, int O, int device);
typedef void   (*fn_coli_cuda_tensor_free)  (ColiCudaTensor *tensor);
typedef size_t (*fn_coli_cuda_tensor_bytes) (const ColiCudaTensor *tensor);
typedef int    (*fn_coli_cuda_tensor_device)(const ColiCudaTensor *tensor);

extern fn_coli_cuda_init          p_coli_cuda_init;
extern fn_coli_cuda_shutdown      p_coli_cuda_shutdown;
extern fn_coli_cuda_device_count  p_coli_cuda_device_count;
extern fn_coli_cuda_device_at     p_coli_cuda_device_at;
extern fn_coli_cuda_mem_info      p_coli_cuda_mem_info;
extern fn_coli_cuda_stats         p_coli_cuda_stats;
extern fn_coli_cuda_tensor_upload p_coli_cuda_tensor_upload;
extern fn_coli_cuda_matmul        p_coli_cuda_matmul;
extern fn_coli_cuda_tensor_free   p_coli_cuda_tensor_free;
extern fn_coli_cuda_tensor_bytes  p_coli_cuda_tensor_bytes;
extern fn_coli_cuda_tensor_device p_coli_cuda_tensor_device;

/* Redirect the 11 call sites. Macros affect only function-call syntax; the
 * prototype declarations already parsed from backend_cuda.h above are not
 * touched. C function pointers accept plain f(a,b) syntax — no (*f)(a,b). */
#define coli_cuda_init          p_coli_cuda_init
#define coli_cuda_shutdown      p_coli_cuda_shutdown
#define coli_cuda_device_count  p_coli_cuda_device_count
#define coli_cuda_device_at     p_coli_cuda_device_at
#define coli_cuda_mem_info      p_coli_cuda_mem_info
#define coli_cuda_stats         p_coli_cuda_stats
#define coli_cuda_tensor_upload p_coli_cuda_tensor_upload
#define coli_cuda_matmul        p_coli_cuda_matmul
#define coli_cuda_tensor_free   p_coli_cuda_tensor_free
#define coli_cuda_tensor_bytes  p_coli_cuda_tensor_bytes
#define coli_cuda_tensor_device p_coli_cuda_tensor_device

/* Load coli_cuda.dll (from cwd or PATH) and resolve all 11 exports.
 * Returns 1 on success, 0 on failure (with an explanatory line on stderr).
 * Must be called ONCE before the first coli_cuda_* call. Idempotent: a
 * successful load is cached; a failed load is not retried. */
int compat_cuda_load(void);

#endif /* _WIN32 && COLI_CUDA */

#endif /* COMPAT_CUDA_H */
