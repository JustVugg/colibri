#ifndef COLIBRI_BACKEND_CUDA_H
#define COLIBRI_BACKEND_CUDA_H

#include <stddef.h>
#include <stdint.h>

/* COLI_CUDA_DLLEXPORT marks functions exported from coli_cuda.dll on Windows.
 * Define COLI_CUDA_BUILDING_DLL when compiling the .cu into the DLL (so the
 * functions are __declspec(dllexport)); the host loader does NOT include this
 * header's declarations — it resolves symbols at runtime via GetProcAddress. */
#if defined(_WIN32) && defined(COLI_CUDA_BUILDING_DLL)
#define COLI_CUDA_DLLEXPORT __declspec(dllexport)
#else
#define COLI_CUDA_DLLEXPORT
#endif


#ifdef __cplusplus
extern "C" {
#endif

#define COLI_CUDA_MAX_DEVICES 16

/* Opaque, persistent device copy of one resident quantized tensor. */
typedef struct ColiCudaTensor ColiCudaTensor;

/* Devices are CUDA ordinals, not positions in the input list. */
COLI_CUDA_DLLEXPORT int coli_cuda_init(const int *devices, int count);
COLI_CUDA_DLLEXPORT void coli_cuda_shutdown(void);
COLI_CUDA_DLLEXPORT int coli_cuda_device_count(void);
COLI_CUDA_DLLEXPORT int coli_cuda_device_at(int index);
COLI_CUDA_DLLEXPORT int coli_cuda_mem_info(int device, size_t *free_bytes, size_t *total_bytes);
COLI_CUDA_DLLEXPORT int coli_cuda_device_integrated(int device);
/* device < 0 returns aggregate statistics for all configured devices. */
COLI_CUDA_DLLEXPORT void coli_cuda_stats(int device, size_t *tensor_count, size_t *tensor_bytes);
COLI_CUDA_DLLEXPORT void coli_cuda_group_stats(uint64_t *calls, uint64_t *experts, uint64_t *rows,
                           double *h2d_ms, double *kernel_ms, double *d2h_ms);
/* Per-device form of coli_cuda_group_stats; unknown devices return zeros. */
COLI_CUDA_DLLEXPORT void coli_cuda_group_stats_device(
    int device, uint64_t *calls, uint64_t *experts, uint64_t *rows,
    double *h2d_ms, double *kernel_ms, double *d2h_ms);

/* Publish the E8 codebook (quant.h's e8_grid, 256x4 bytes) to every configured
 * device. Must be called after coli_cuda_init and before any fmt=6 upload; the
 * backend keeps no copy of the table so it cannot drift from the CPU decoder. */
COLI_CUDA_DLLEXPORT int coli_cuda_e8_set_grid(const void *grid);

/* Publish the fmt=8 e4m3 decode table (quant.h's E4M3_LUT, 256 f32) the same
 * way. Must be called after coli_cuda_init; fmt=8 uploads are refused until it
 * succeeds, because kernels would decode against a zero-initialized table. */
COLI_CUDA_DLLEXPORT int coli_cuda_fp8_set_lut(const float *lut);

/* Upload without executing, so capacity failures happen during model startup. */
COLI_CUDA_DLLEXPORT int coli_cuda_tensor_upload_g(ColiCudaTensor **tensor,
        const void *weights, const float *scales,
        int fmt, int I, int O, int device, int gs);
COLI_CUDA_DLLEXPORT int coli_cuda_tensor_upload(ColiCudaTensor **tensor,
                            const void *weights, const float *scales,
                            int fmt, int I, int O, int device);
#ifdef COLI_ANS
/* Experimental Linux-only GPU-resident entropy tier. The archive remains in
 * VRAM and is decoded into per-device scratch immediately before a grouped
 * expert launch. */
COLI_CUDA_DLLEXPORT int coli_cuda_tensor_upload_compressed(ColiCudaTensor **tensor,
                            const void *weights, const float *scales,
                            int fmt, int I, int O, int device);
#endif

/*
 * y[S,O] = x[S,I] @ W[O,I]^T.
 * fmt matches QT in glm.c: 0=f32, 1=int8, 2=int4, 3=int2, 4=grouped int4.
 * gs is the group size for fmt=4 (0 for all other formats).
 * The first successful call uploads W and its scales; later calls reuse it.
 * Returns 1 on success and 0 when CUDA is not initialized or the format is invalid.
 */
/* y[S,O] = x[S,I] @ dequant_mxfp4(W[O,I])^T, fmt=7 (OCP microscaling FP4).
 * q4 is [O, ceil(I/2)] e2m1 nibbles, e8s is [O, ceil(I/32)] ue8m0 exponents --
 * BYTES, not floats, which is why this is not folded into coli_cuda_matmul.
 * Stateless: weights are uploaded per call, matching the streaming expert tier
 * Kimi K3 uses. Returns 1 on success, 0 (y untouched) to fall back to CPU. */
COLI_CUDA_DLLEXPORT int coli_cuda_matmul_mxfp4(float *y, const float *x,
                                               const unsigned char *q4,
                                               const unsigned char *e8s,
                                               int S, int I, int O);

COLI_CUDA_DLLEXPORT int coli_cuda_matmul(ColiCudaTensor **tensor,
                     float *y, const float *x,
                     const void *weights, const float *scales,
                     int fmt, int S, int I, int O, int device, int gs);

/* Fused expert pipeline: y = down(silu(gate(x)) * up(x)).  All three tensors
 * must already be resident on one device.  Activations cross PCIe once in
 * each direction instead of once per matrix. */
COLI_CUDA_DLLEXPORT int coli_cuda_expert_mlp(ColiCudaTensor *gate, ColiCudaTensor *up,
                         ColiCudaTensor *down, float *y, const float *x, int S);

/* Prefill-oriented shared expert path.  INT4 weights stay packed in global
 * memory, activations are converted to FP16 per tile, and Tensor Cores
 * accumulate into FP32.  Unlike COLI_CUDA_TC_INT4 this does not quantize the
 * activation to INT4. */
COLI_CUDA_DLLEXPORT int coli_cuda_shared_mlp_w4a16(ColiCudaTensor *gate, ColiCudaTensor *up,
                               ColiCudaTensor *down, float *y,
                               const float *x, int S);

/* Packed group of same-shaped experts. Inputs and outputs contain sum(rows)
 * consecutive [D] rows in call order. */
/* Async issue/take split of the group call below (Inc.4): issue launches on the
 * device stream and returns; take syncs and returns the pinned result rows (valid
 * until the next issue on that device). Small totals only (<=8 rows); one
 * outstanding issue per device. */
COLI_CUDA_DLLEXPORT int coli_cuda_expert_group_issue(ColiCudaTensor *const *gates,
                               ColiCudaTensor *const *ups,
                               ColiCudaTensor *const *downs,
                               const int *rows, int count, const float *x);
COLI_CUDA_DLLEXPORT const float *coli_cuda_expert_group_take(int device);

COLI_CUDA_DLLEXPORT int coli_cuda_expert_group(ColiCudaTensor *const *gates,
                           ColiCudaTensor *const *ups,
                           ColiCudaTensor *const *downs,
                           const int *rows, int count,
                           float *y, const float *x);

/* Decode-only MLA weight-absorption core for one token. kv_b is [H*(Q+V),K]. */
COLI_CUDA_DLLEXPORT int coli_cuda_attention_absorb(ColiCudaTensor *kv_b,float *ctx,const float *q,
                               const float *latent,const float *rope,int H,int Q,
                               int R,int V,int K,int T,float attention_scale);

/* Causal MLA absorption for S contiguous rows from one sequence.  The KV
 * arrays contain T rows ending at the final query; query s attends T-S+s+1
 * rows.  One transfer and one launch replace S host round-trips. */
COLI_CUDA_DLLEXPORT int coli_cuda_attention_absorb_batch(ColiCudaTensor *kv_b,float *ctx,const float *q,
                                     const float *latent,const float *rope,int S,
                                     int H,int Q,int R,int V,int K,int T,
                                     float attention_scale);

/* Same attention batch followed immediately by resident o_proj on the same
 * device.  Only the final [S,D] tensor crosses back to the host. */
COLI_CUDA_DLLEXPORT int coli_cuda_attention_project_batch(ColiCudaTensor *kv_b,ColiCudaTensor *o_proj,
                                      float *out,const float *q,const float *latent,
                                      const float *rope,int S,int H,int Q,int R,
                                      int V,int K,int T,float attention_scale);

COLI_CUDA_DLLEXPORT int coli_cuda_attention_project_ragged(ColiCudaTensor *kv_b,ColiCudaTensor *o_proj,
        float *out,const float *q,const void *const *keys,
        const float *const *latent,const float *const *rope,
        const int *lengths,int S,int H,int Q,int R,int V,int K,int max_t,float attention_scale);

COLI_CUDA_DLLEXPORT void coli_cuda_tensor_free(ColiCudaTensor *tensor);
COLI_CUDA_DLLEXPORT size_t coli_cuda_tensor_bytes(const ColiCudaTensor *tensor);
COLI_CUDA_DLLEXPORT int coli_cuda_tensor_device(const ColiCudaTensor *tensor);

/* Replace a resident tensor's contents without reallocating its device slot. */
COLI_CUDA_DLLEXPORT int coli_cuda_tensor_update(ColiCudaTensor *tensor,
                            const void *weights, const float *scales);

/* ---- resident-pipeline primitives (Inc.0): device-pointer entry points ---- */
COLI_CUDA_DLLEXPORT float *coli_cuda_pipe_scratch(int device,int slot,size_t bytes);
COLI_CUDA_DLLEXPORT void *coli_cuda_pipe_alloc(int device,size_t bytes);
COLI_CUDA_DLLEXPORT void coli_cuda_pipe_free(int device,void *p);
COLI_CUDA_DLLEXPORT int coli_cuda_pipe_upload(int device,void *dst,const void *src,size_t bytes);
COLI_CUDA_DLLEXPORT int coli_cuda_pipe_download(int device,const void *src,void *dst,size_t bytes);
COLI_CUDA_DLLEXPORT int coli_cuda_pipe_rmsnorm(int device,float *y_dev,const float *x_dev,
                           const float *w_dev,int S,int D,float eps);
COLI_CUDA_DLLEXPORT int coli_cuda_pipe_rope(int device,float *v_dev,const int *pos_dev,int rows,
                        int stride,int offset,int R,int heads,float theta);
COLI_CUDA_DLLEXPORT int coli_cuda_pipe_silu_mul(int device,float *gate_dev,const float *up_dev,size_t n);
COLI_CUDA_DLLEXPORT int coli_cuda_pipe_add(int device,float *x_dev,const float *t_dev,size_t n);
COLI_CUDA_DLLEXPORT int coli_cuda_pipe_rows_add(int device,float *x_dev,const float *partial_dev,
                            const int *rows_dev,int nrows,int D);
COLI_CUDA_DLLEXPORT int coli_cuda_pipe_gemm(ColiCudaTensor *t,float *y_dev,const float *x_dev,int S);
COLI_CUDA_DLLEXPORT int coli_cuda_pipe_rmsnorm_s(int device,float *y_dev,const float *x_dev,
                             const float *w_dev,int S,int D,float eps,
                             int xstride,int ystride);
COLI_CUDA_DLLEXPORT int coli_cuda_pipe_rope_base(int device,float *v_dev,int pos_base,int rows,
                             int stride,int offset,int R,int heads,float theta);
COLI_CUDA_DLLEXPORT int coli_cuda_expert_group_resident_issue(ColiCudaTensor *const *gates,
        ColiCudaTensor *const *ups, ColiCudaTensor *const *downs,
        const float *weights, int count,
        int home_device, const float *x_src_dev, float *partial_slot_dev);
COLI_CUDA_DLLEXPORT int coli_cuda_expert_group_resident_take(int home_device,const int *devices,
        int n_issued,float *slots_dev,float *acc_dev,int D);
COLI_CUDA_DLLEXPORT int coli_cuda_pipe_router(int device,const float *x_dev,
        const void *rw_dev,const void *rb_dev,int D,int E,int Ksel,
        float topp,int norm_topk,float routed_scale,
        int *idx_host,float *w_host,int *keff_host);
COLI_CUDA_DLLEXPORT int coli_cuda_pipe_copy2d(int device,float *dst,int dpitch,const float *src,
                          int spitch,int width,int height);
COLI_CUDA_DLLEXPORT int coli_cuda_attention_project_batch_dev(ColiCudaTensor *kv_b,ColiCudaTensor *o_proj,
        float *out,const float *q_dev,const float *latent_dev,const float *rope_dev,
        int S,int H,int Q,int R,int V,int K,int T,float scale);
COLI_CUDA_DLLEXPORT int coli_cuda_attention_absorb_batch_dev(ColiCudaTensor *kv_b_shard,float *ctx_dev,
        const float *q_dev,const float *latent_dev,const float *rope_dev,
        int S,int H,int Q,int R,int V,int K,int T,float scale);
COLI_CUDA_DLLEXPORT int coli_cuda_attention_absorb_kvdev(ColiCudaTensor *kv_b,float *ctx,const float *q,
        const float *latent_dev,const float *rope_dev,int H,int Q,int R,int V,int K,int T,
        float scale);
COLI_CUDA_DLLEXPORT int coli_cuda_pipe_peer_copy(int dst_dev,float *dst,int src_dev,
                             const float *src,size_t bytes);
COLI_CUDA_DLLEXPORT int coli_cuda_attention_project_batch_dev_out(ColiCudaTensor *kv_b,ColiCudaTensor *o_proj,
        float *out_dev,const float *q_dev,const float *latent_dev,const float *rope_dev,
        int S,int H,int Q,int R,int V,int K,int T,float scale);
COLI_CUDA_DLLEXPORT int coli_cuda_pipe_sync(int device);

#ifdef __cplusplus
}
#endif

#endif
