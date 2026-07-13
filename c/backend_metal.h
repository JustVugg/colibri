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
void coli_metal_spin_start(void);   /* COLI_METAL_SPIN=1 keep-alive experiment */
void coli_metal_spin_stop(void);
void coli_metal_register(void *base, size_t len);
void coli_metal_unregister(void *base);

/*
 * Fused decode (S=1) attention for one layer, entirely on the GPU in one command buffer:
 * q_a -> rmsnorm -> q_b -> RoPE ; kv_a -> latent rmsnorm@pos + krot RoPE@pos (cache write) ;
 * MLA absorption core ; o_proj. Weights (q_a/q_b/kv_a/kv_b/o) and the Lc/Rc caches must be
 * registered (page-aligned) for zero-copy resolve. GLM-5.2 dims compiled in. Handles st0==0
 * full-range only. Returns 1 on success, 0 to signal CPU fallback.
 */
/*
 * Full decode layer in ONE command buffer: in_ln -> attention -> residual -> post_ln ->
 * shared expert -> router+top-K (exact phase-A semantics). x updated in place; nrm_out
 * is the expert input; sh_out the shared-expert output; idx/w/keff the routing.
 * Returns 0 -> caller runs the whole layer on the CPU path.
 */
int coli_metal_layer_decode(float *x,
    const float *in_ln, const float *post_ln,
    const void *qa_w, const float *qa_s, int qa_fmt, const float *qa_ln,
    const void *qb_w, const float *qb_s, int qb_fmt,
    const void *kva_w, const float *kva_s, int kva_fmt, const float *kva_ln,
    const void *kvb_w, const float *kvb_s, int kvb_fmt,
    const void *o_w, const float *o_s, int o_fmt,
    const void *shg_w, const float *shg_s, int shg_fmt,
    const void *shu_w, const float *shu_s, int shu_fmt,
    const void *shd_w, const float *shd_s, int shd_fmt,
    const float *router_w, const float *router_bias,
    int E, int K, int Ksel, float topp, int normk, float rscale,
    float *Lc, float *Rc, int S, int pos_base, int st0,
    float eps, float theta, float ascale,
    float *inrm_out, float *nrm_out, float *sh_out, int *idx_out, float *w_out, int *keff_out);

int coli_metal_gemm(float *y, const float *x, const void *weights, const float *scales,
                    int fmt, int S, int I, int O);   /* large-batch sync GEMM; 0 -> CPU */
void coli_metal_attn_counts(uint64_t *ok, double *wall, double *kernel);
void coli_metal_attn_lat(double *ksched, double *gsched);
int coli_metal_attn_decode(const float *x,
    const void *qa_w, const float *qa_s, int qa_fmt, const float *qa_ln,
    const void *qb_w, const float *qb_s, int qb_fmt,
    const void *kva_w, const float *kva_s, int kva_fmt, const float *kva_ln,
    const void *kvb_w, const float *kvb_s, int kvb_fmt,
    const void *o_w, const float *o_s, int o_fmt,
    float *Lc, float *Rc, int S, int pos_base, int st0, float eps, float theta, float ascale, float *out);

/* Diagnostics: GPU blocks executed, CPU-fallback blocks, experts run on GPU. */
void coli_metal_moe_counts(uint64_t *ok, uint64_t *fb, uint64_t *experts);
void coli_metal_moe_times(double *setup, double *gpu, double *scatter);
double coli_metal_moe_kernel_time(void);

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

/*
 * Async two-phase variant: begin encodes+commits the block (own scratch, no wait) and
 * returns a handle, so the CPU can load missed experts from disk WHILE the GPU computes
 * the resident ones; end waits, checks for GPU faults, scatter-adds into out, and frees
 * the handle. begin returns NULL (nothing submitted) on unresolved slab / bad fmt / R==0;
 * end returns 0 on GPU fault (caller redoes those experts on CPU).
 */
typedef struct ColiMetalMoeHandle ColiMetalMoeHandle;
ColiMetalMoeHandle* coli_metal_moe_block_begin(int nb, int D, int Iinter, int fmt,
                         const void *const *g, const void *const *u, const void *const *d,
                         const float *const *gs, const float *const *us, const float *const *ds,
                         const float *xg, const int *xoff, const int *nr,
                         const int *rows, const float *rw);
int coli_metal_moe_block_end(ColiMetalMoeHandle *h, float *out);

#ifdef __cplusplus
}
#endif

#endif
