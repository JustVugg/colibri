#ifndef COLIBRI_BACKEND_METAL_H
#define COLIBRI_BACKEND_METAL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Apple-GPU (Metal) backend for colibrì. Apple Silicon has one GPU and unified
 * memory, so there is no device list and no host<->device copy: resident weights
 * are read zero-copy from the RAM they already occupy. The shader is compiled at
 * runtime (newLibraryWithSource:), so no Xcode / offline metal compiler is needed.
 */

/* Opaque, persistent GPU handle for one resident quantized tensor. */
typedef struct ColiMetalTensor ColiMetalTensor;

/* Returns 1 if a Metal device is available and pipelines compiled, else 0. */
int  coli_metal_init(void);
void coli_metal_shutdown(void);
int  coli_metal_available(void);
/* Bytes of unified memory in use by wrapped tensors, and their count. */
void coli_metal_stats(size_t *tensor_count, size_t *tensor_bytes);
int  coli_metal_mem_info(size_t *used_bytes, size_t *total_bytes);

/*
 * y[S,O] = (x[S,I] @ W[O,I]^T) * scale[o].
 * fmt matches QT in glm.c: 0=f32, 1=int8, 2=int4(packed), 3=int2(packed).
 * The first successful call wraps W and its row scales in GPU-visible buffers;
 * later calls reuse them (weights are assumed stable at the same address).
 * Returns 1 on success, 0 if Metal is unavailable or fmt is invalid.
 */
int coli_metal_matmul(ColiMetalTensor **tensor,
                      float *y, const float *x,
                      const void *weights, const float *scales,
                      int fmt, int S, int I, int O);

void   coli_metal_tensor_free(ColiMetalTensor *tensor);
size_t coli_metal_tensor_bytes(const ColiMetalTensor *tensor);

/*
 * Register a page-aligned host allocation (expert slab / scale slab) so the batched
 * MoE path can read it zero-copy: the backend wraps it once in an MTLBuffer
 * (newBufferWithBytesNoCopy) and resolves any pointer inside [base,base+len) to a GPU
 * address. Call after (re)allocating a slab; call unregister before freeing it.
 * base must be aligned to 16384 (Apple page) and len a multiple of it.
 */
void coli_metal_register(void *base, size_t len);
void coli_metal_unregister(void *base);

/*
 * Batched routed-expert SwiGLU for one MoE block, in ONE command buffer.
 * For each expert e in [0,nb): computes hh_e[nr_e, D] = down( silu(gate(xg_e)) * up(xg_e) )
 * and scatter-adds rw * hh_e into out. All experts share the command buffer so the
 * ~150us Metal launch latency is paid once per block, not per matmul.
 *
 *  D           = hidden size, Iinter = moe intermediate size
 *  g/u/d[e]    = pointers to expert e's gate/up/down quantized weights (in RAM slabs)
 *  gs/us/ds[e] = pointers to expert e's per-row scales
 *  fmt         = quant format (shared across experts)
 *  xg          = packed activations [total_rows, D]; xoff[e] = row offset of expert e
 *  nr[e]       = rows for expert e; rows[]/rw[] map packed rows back to out positions
 *  out         = [S, D] accumulate target
 * Returns 1 on success, 0 to signal the caller to fall back to the CPU path.
 */
int coli_metal_moe_block(int nb, int D, int Iinter, int fmt,
                         const void *const *g, const void *const *u, const void *const *d,
                         const float *const *gs, const float *const *us, const float *const *ds,
                         const float *xg, const int *xoff, const int *nr,
                         const int *rows, const float *rw,
                         float *out, int S);

#ifdef __cplusplus
}
#endif

#endif
