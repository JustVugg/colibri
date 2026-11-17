/* Metal kernels for the Qwen3.8 engine, as plain C.
 *
 * They live in their own library (metal/qwen38.metal, built into
 * qwen38_metal.o) but run on qwen36's device and in qwen36's command batch:
 * q36_tensor holds their data, a q38 dispatch joins the encoder q36 dispatches
 * use, in the order they are called, and q36_gpu_flush/q36_gpu_sync commit and
 * wait for both. Every wrapper returns 1 when its work is enqueued and 0 when
 * it declines (a bad shape or size; nothing is enqueued). No wrapper syncs. */
#ifndef QWEN38_GPU_H
#define QWEN38_GPU_H
#include <stddef.h>
#include "qwen36_gpu.h"
#ifdef __cplusplus
extern "C" {
#endif

/* 1 ready, 0 not: runs q36_gpu_init() if needed, then compiles the qwen38
 * library on the same device (32-lane SIMD groups required). */
int q38_gpu_init(void);

/* y[i] = x[i] for n floats. Plumbing test only. */
int q38_copy(q36_tensor *y, q36_tensor *x, size_t n);

/* y (rows x out) = x (rows x in) . w^T. w is out x in raw BF16 (uint16, the GGUF
 * layout, read in place), x and y f32. 0 (nothing enqueued) if in % 8 != 0 or a
 * size does not fit. Exact mode sums in a fixed order; fast uses simd_sum. */
int q38_dot_bf16(q36_tensor *y, q36_tensor *x, q36_tensor *w, int rows, int in, int out);

/* Hyper-connection element-wise ops (the CPU's q38_gr_read / q38_gr_apply; fc = the stream
 * count C as a float). All return 0, enqueueing nothing, for a bad shape or size.
 * y[i] = silu(a[i] / fc), n floats. */
int q38_hc_low(q36_tensor *y, q36_tensor *a, int n, float fc);
/* y[r][d] = sum_k sigmoid(a[r][k][d]) * b[r][k][d] / fc; a, b are rows x (C*D), y rows x D. */
int q38_hc_mix(q36_tensor *y, q36_tensor *a, q36_tensor *b, int rows, int D, int C, float fc);
/* y[i] = 2 * sigmoid(a[i] / fc), n floats. */
int q38_hc_inj(q36_tensor *y, q36_tensor *a, int n, float fc);
/* hyper[r][k][d] += inject[r][k] * block[r][d]; hyper rows x (C*D), inject rows x C, block rows x D. */
int q38_hc_apply(q36_tensor *hyper, q36_tensor *inject, q36_tensor *block, int rows, int D, int C);
/* y[r*C+b] = rms0(x[r*C+b]) over D, weight row b of the C*D-long w: x * rsqrt(mean(x^2)+eps) * (1+w[b*D+j])
 * (q38_rms0 per stream, as q38_gr_read). x, y rows x (C*D). Same fixed-order arithmetic in both modes. */
int q38_hc_norm(q36_tensor *y, q36_tensor *x, q36_tensor *w, int rows, int D, int C, float eps);

/* The per-layer n-gram embedding (PLE) after its key and value projections: qwen38_core.h's
 * q38_ple from the per-stream gate on, and shaders/chain_ple.comp on Vulkan. W is the
 * hyper-connection width C*H (cfg hc_width), not H: keys, hyper, gated, normv and out are
 * S x W (C streams of H each), value is S x H (one row shared by every stream). The ring
 * is W x SL with SL = (CK-1)*NG <= 32; snap is snap_rows x W x SL. */
typedef struct { int S, C, W, H, CK, NG; int snap_rows; float eps; int accumulate; } q38_ple_shape;
/* gate, per row s and stream k (slices k*H of keys/hyper/gated/normv and of each norm):
 *   dot = rms0(keys[s][k], norm_key) . rms0(hyper[s][k], norm_query) / sqrt(H);
 *   g = sigmoid(copysign(sqrt(max(|dot|, 1e-6)), dot)); gated[s][k] = g * value[s];
 *   normv[s][k] = rms0(gated[s][k], norm_conv). prm = norm_key[W] | norm_query[W] | norm_conv[W].
 * One 256-thread group a (s, k), fixed-order sums: exact and fast are the same kernel. */
int q38_ple_gate(q36_tensor *gated, q36_tensor *normv, q36_tensor *keys, q36_tensor *hyper,
                 q36_tensor *value, q36_tensor *prm, const q38_ple_shape *s);
/* conv, one thread per channel d < W, rows s in order (conv is W x CK):
 *   a = conv[d*CK+CK-1]*normv[s][d] + sum_{k<CK-1} conv[d*CK+k]*ring[d*SL+k*NG];
 *   v = gated[s][d] + silu(a); out[s][d] = v (accumulate 0, the CPU's out) or out[s][d] += v
 *   (accumulate 1, Vulkan's hyper += form); then ring[d] shifts left by one and its last
 *   entry takes normv[s][d]; for s < snap_rows the whole ring is copied to snap[s].
 * snap may be NULL when snap_rows is 0. */
int q38_ple_conv(q36_tensor *out, q36_tensor *ring, q36_tensor *snap, q36_tensor *gated,
                 q36_tensor *normv, q36_tensor *conv, const q38_ple_shape *s);

#ifdef __cplusplus
}
#endif
#endif
