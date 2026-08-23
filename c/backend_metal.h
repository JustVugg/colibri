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

/* ---- Standalone elementwise / normalization ops ---------------------------------- */
/*
 * RMSNorm: out[nrows*D] = rmsnorm(x[nrows*D], w[D], eps).
 * Each of nrows independent rows of width D. w is shared (1 copy, [D]).
 * Returns 1 on success, 0 if Metal unavailable.
 */
int coli_metal_rmsnorm(float *x, const float *w, int nrows, int D, float eps);

/*
 * Residual add: y[i] += a[i] for i in [0, n).
 * Returns 1 on success, 0 if Metal unavailable.
 */
int coli_metal_add(float *y, const float *a, size_t n);

/*
 * SiLU gate-multiply: g[i] = silu(g[i]) * u[i] for i in [0, n).
 * Destroy-u variant: u is consumed (gate * up fusion for SwiGLU).
 * Returns 1 on success, 0 if Metal unavailable.
 */
int coli_metal_silu_mul(float *g, const float *u, size_t n);

/*
 * y[S,O] = (x[S,I] @ W[O,I]^T) * scale[o]. fmt=4 (grouped int4) instead folds a
 * PER-GROUP scale into the accumulation -- see the shader comment in backend_metal.mm.
 * fmt=8 (fp8 passthrough -- see colibri.c) likewise folds its per-128x128-block
 * scale into the accumulation.
 * fmt matches QT in colibri.c: 0=f32, 1=int8, 2=int4(packed), 3=int2(packed),
 * 4=int4(packed, same layout as 2)+per-group scale (gs = group size along I; scale
 * array is [O, ceil(I/gs)] floats instead of [O]),
 * 8=fp8-e4m3 (one raw byte/element, same layout as fmt=1) + per-128x128-
 * block scale (scale array is [ceil(O/128),ceil(I/128)] floats; no group-size
 * parameter needed -- the block is a fixed 128x128, not caller-configurable).
 * gs is ignored for fmt!=4 (pass 0).
 * The first successful call wraps W and its scales in GPU-visible buffers (sized from
 * fmt+gs); later calls reuse them (weights are assumed stable at the same address).
 * Returns 1 on success, 0 if Metal is unavailable or fmt is invalid.
 */
int coli_metal_matmul(ColiMetalTensor **tensor,
                      float *y, const float *x,
                      const void *weights, const float *scales,
                      int fmt, int S, int I, int O, int gs);

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
    const void *qa_w, const float *qa_s, int qa_fmt, int qa_gs, const float *qa_ln,
    const void *qb_w, const float *qb_s, int qb_fmt, int qb_gs,
    const void *kva_w, const float *kva_s, int kva_fmt, int kva_gs, const float *kva_ln,
    const void *kvb_w, const float *kvb_s, int kvb_fmt, int kvb_gs,
    const void *o_w, const float *o_s, int o_fmt, int o_gs,
    const void *shg_w, const float *shg_s, int shg_fmt, int shg_gs,
    const void *shu_w, const float *shu_s, int shu_fmt, int shu_gs,
    const void *shd_w, const float *shd_s, int shd_fmt, int shd_gs,
    const float *router_w, const float *router_bias,
    int E, int K, int Ksel, float topp, int normk, float rscale,
    float *Lc, float *Rc, int S, int pos_base, int st0,
    float eps, float theta, float ascale,
    float *inrm_out, float *nrm_out, float *sh_out, int *idx_out, float *w_out, int *keff_out);

int coli_metal_gemm(float *y, const float *x, const void *weights, const float *scales,
                    int fmt, int S, int I, int O, int gs);   /* large-batch sync GEMM; 0 -> CPU. gs: fmt=4 group size (0 otherwise) */
/* Parallel top-8 expert selection (r_top8_par): run ONE top-8 selection kernel standalone
 * on host arrays — par=0 the serial r_top8, par=1 the parallel exact-match replica gated
 * in the engine by COLI_RTOP8 (default ON; COLI_RTOP8=0 opts out to the serial kernel).
 * Exists so the metal-test suite (and any battery probe) can prove serial/parallel
 * equivalence on the ENGINE build's own compiled shaders, not just in the bench tool.
 * sig[S*E], bias[E], idx[S*K], w[S*K], keff[S].
 * Expert-count generality: the parallel kernel's blocked-lane design (ch[8]/32-lane
 * threadgroup) is validated correct for arbitrary E<=256, including non-multiples of the
 * 32-lane width and small E (see metal-test's E=24/E=168/E=256 cases — 168 is the REAP
 * expert-pruned package width from #428/#426). For E>256 (out of contract) this function
 * transparently falls back to the serial kernel even when par=1 is requested, and the
 * same automatic fallback is wired into the engine dispatch site — "par" is a request,
 * never a guarantee, so no caller can reach the unguarded parallel path out of contract.
 * Returns 1 on success, 0 if Metal is unavailable. */
int coli_metal_rtop8(int par, const float *sig, const float *bias, int S, int E, int K,
                     int Ksel, float topp, int normk, float rscale,
                     int *idx, float *w, int *keff);
void coli_metal_attn_counts(uint64_t *ok, double *wall, double *kernel);
void coli_metal_attn_lat(double *ksched, double *gsched);
int coli_metal_attn_decode(const float *x,
    const void *qa_w, const float *qa_s, int qa_fmt, int qa_gs, const float *qa_ln,
    const void *qb_w, const float *qb_s, int qb_fmt, int qb_gs,
    const void *kva_w, const float *kva_s, int kva_fmt, int kva_gs, const float *kva_ln,
    const void *kvb_w, const float *kvb_s, int kvb_fmt, int kvb_gs,
    const void *o_w, const float *o_s, int o_fmt, int o_gs,
    float *Lc, float *Rc, int S, int pos_base, int st0, float eps, float theta, float ascale, float *out);

/*
 * Write S KV-cache rows on GPU. out = Lc + pos_base*kv_lora gets rmsnorm(raw L, kv_a_ln, eps)
 * and Rc + pos_base*qk_rope gets rope_interleave(raw R, pos_base+s, theta). Mirrors the
 * CPU cache write: rmsnorm(Lc[pos], Lc[pos], kv_a_ln, kv_lora, eps) + rope_interleave(Rc[pos], pos, cfg).
 * L_raw/R_raw are [S, kv_lora] / [S, qk_rope]. Returns 1 on success, 0 if Metal unavailable. */
int coli_metal_kv_write(float *Lc, float *Rc, const float *L_raw, const float *R_raw,
    const float *kv_a_ln, int S, int pos_base, int kv_lora, int qk_rope, float eps, float theta);

/*
 * Zero cache rows Lc[from*kv_lora .. (to-1)*kv_lora] and Rc[from*qk_rope .. (to-1)*qk_rope]
 * on GPU. Covered by dedicated kernel. Returns 1 on success, 0 if Metal unavailable. */
int coli_metal_kv_clear(float *Lc, float *Rc, int from, int to, int kv_lora, int qk_rope);

/* Diagnostics: GPU blocks executed, CPU-fallback blocks, experts run on GPU. */
void coli_metal_moe_counts(uint64_t *ok, uint64_t *fb, uint64_t *experts);
void coli_metal_moe_times(double *setup, double *gpu, double *scatter);
double coli_metal_moe_kernel_time(void);
/* E5 (COLI_METAL_RESSET=1): returns 1 when the queue-attached residency set is active and
 * writes the cumulative seconds moe_submit spent committing pending set adds -- a cost that
 * sits OUTSIDE the setup/gpu/scatter breakdown above. Returns 0 (and writes 0) when off. */
int coli_metal_resset_stats(double *flush_s);

/*
 * Batched routed-expert SwiGLU for one MoE block, in ONE command buffer.
 * For each expert e in [0,nb): computes hh_e[nr_e, D] = down( silu(gate(xg_e)) * up(xg_e) )
 * and scatter-adds rw * hh_e into out. All experts share the command buffer so the
 * ~150us Metal launch latency is paid once per block, not per matmul.
 *
 *  D           = hidden size, Iinter = moe intermediate size
 *  g/u/d[e]    = pointers to expert e's gate/up/down quantized weights (in RAM slabs)
 *  gs/us/ds[e] = pointers to expert e's per-row (fmt=1/2) or per-group (fmt=4) scales
 *  fmt         = quant format (shared across experts): 1=int8, 2=int4 per-row, 4=int4
 *                grouped. qgs is the fmt=4 group size (ignored, pass 0, for fmt!=4).
 *  qgs         = fmt=4 group size shared across experts in this block (0 for fmt!=4)
 *  xg          = packed activations [total_rows, D]; xoff[e] = row offset of expert e
 *  nr[e]       = rows for expert e; rows[]/rw[] map packed rows back to out positions
 *  out         = [S, D] accumulate target
 * Returns 1 on success, 0 to signal the caller to fall back to the CPU path.
 */
int coli_metal_moe_block(int nb, int D, int Iinter, int fmt, int qgs,
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
ColiMetalMoeHandle* coli_metal_moe_block_begin(int nb, int D, int Iinter, int fmt, int qgs,
                         const void *const *g, const void *const *u, const void *const *d,
                         const float *const *gs, const float *const *us, const float *const *ds,
                         const float *xg, const int *xoff, const int *nr,
                         const int *rows, const float *rw);
int coli_metal_moe_block_end(ColiMetalMoeHandle *h, float *out);

/*
 * KDA state recurrence for one token (decode: C=1). Replaces the serial per-head
 * sweep in kda_forward(): decay state by alpha, accumulate kS, expand by kn*vt,
 * merge output with qn. Each thread handles one element of [H * hd] output space;
 * alpha/kn/qn/alpha are precomputed per-head arrays.
 *   S          — in-place state `[H * hd * hd]` (row-major: [h][kk][vv])
 *   qn         — L2-normalized, gated Q        `[H * hd]`
 *   kn         — L2-normalized K               `[H * hd]`
 *   vh         — gated V                        `[H * hd]`
 *   alpha      — decay factors                  `[H * hd]`
 *   beta       — mixing factor                  `[H]`
 *   oh         — output accumulator             `[H * hd]` (initialized to 0 by kernel)
 *   H, hd     — number of heads, head dimension
 * Returns 1 on success, 0 if Metal unavailable.
 */
int coli_metal_kda_state(float *S, const float *qn, const float *kn, const float *vh,
                           const float *alpha, const float *beta, float *oh,
                           int H, int hd);

/*
 * KDA Conv1d + SiLU gate for one projection (q, k, or v). Replaces the OMP-parallel
 * depthwise conv loop in kda_forward(). The conv window is stateful across tokens:
 *   conv_win[d*K]   — [P, K] shift register, oldest first, newest at K-1
 *   vec[d]          — [P] raw projected value (saved to window[K-1], overwritten with gated result)
 *   taps[d*K]       — [P, K] depthwise convolution taps (constant weights)
 * One kernel dispatch per projection. Threadgroup of 256 threads, one thread per dimension.
 * Returns 1 on success, 0 if Metal unavailable.
 */
int coli_metal_kda_conv_silu(float *conv_win, float *vec, const float *taps, int P, int K);

/*
 * KDA L2 normalization: in-place normalization of Q and K per head.
 *   q[h*hd:(h+1)*hd] *= 1/sqrt(sum(q^2) + eps) * qscale
 *   k[h*hd:(h+1)*hd] *= 1/sqrt(sum(k^2) + eps)
 *   eps = 1e-6, qscale = 1/sqrt(hd).
 * One kernel dispatch for all H heads. Returns 1 on success, 0 if Metal unavailable.
 */
int coli_metal_kda_l2_norm(float *q, float *k, int H, int hd, float qscale);

/*
 * Fused KDA token step: conv_silu(q,k,v) + l2_norm(q,k) + state recurrence,
 * all encoded into a single MTLCommandBuffer. This eliminates per-kernel
 * command-buffer creation/commit/wait overhead (~5 × 150us → one 150us).
 *   win_q, win_k, win_v  — [P*K] conv windows (stateful across tokens)
 *   qt, kt                — [P] = [H*hd] raw projected (become L2-normalized q/k in-place)
 *   tv                    — [P] = [H*hd] raw projected (becomes SiLU-gated V in-place)
 *   taps_q, taps_k, taps_v — [P*K] depthwise taps (constant)
 *   S                     — [H*hd*hd] in-place attention state
 *   alpha                 — [H*hd] precomputed decay factors (CPU from rgt+dt+A)
 *   beta                  — [H] precomputed mixing factors (CPU from braw)
 *   oh                    — [H*hd] output accumulator (written by kernel, read by CPU)
 *   P, K, H, hd           — projection dim, kernel width, heads, head dim
 * Returns 1 on success, 0 if Metal unavailable. Caller must zero oh[] before call.
 */
int coli_metal_kda_fused_token(
    float *win_q, float *qt,
    float *win_k, float *kt,
    float *win_v, float *tv,
    const float *taps_q, const float *taps_k, const float *taps_v,
    float *S,
    const float *alpha, const float *beta,
    float *oh,
    int P, int K, int H, int hd);

#ifdef __cplusplus
}
#endif

#endif
