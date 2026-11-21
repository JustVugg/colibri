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

/* QSA, the sparse attention of Qwen3.8's indexer layers: qwen38_core.h's q38_attention and
 * shaders/chain_qsa.comp (pool, score, select) / chain_attn.comp (attention over the list).
 * rope(x, pos) is q38_rope: the first `rotary` dims, pairs (i, i + rotary/2), angle
 * pos / theta^(2i/rotary) (the base powf'd on the host, as the CPU does). rms0 is q38_rms0:
 * x * rsqrt(mean(x^2) + eps) * (1 + w). Row s of a step is at position pos_base + s and sees
 * nblk(s) = (pos_base + s + 1) / R complete blocks of R index keys. */
typedef struct { int IQ, ID, R, budget, rotary; float theta, eps; } q38_qsa_shape;
/* pool blocks [b0, b0+nb): pk[b] = rope(rms0(mean_{r<R} ikraw[b*R+r], idx_kn), pos = b*R).
 * ikraw is [positions][ID] (at least (b0+nb)*R rows), pk [blocks][ID]. ID <= 256. One
 * 256-thread group a block, fixed-order sums: exact and fast are the same kernel. */
int q38_qsa_pool(q36_tensor *pk, q36_tensor *ikraw, q36_tensor *idx_kn, int b0, int nb, const q38_qsa_shape *s);
/* score: sc[s*nbmax + b] = sum_h max(0, dot(q[s][h], pk[b])) / sqrt(ID) for b < nblk(s); the
 * rest of the row is not written. q is [S][IQ][ID], already rms0'd and roped (the indexer
 * queries); nbmax >= nblk(S-1). One thread a (s, b), dots in d order, both modes. */
int q38_qsa_score(q36_tensor *sc, q36_tensor *q, q36_tensor *pk, int S, int pos_base, int nbmax, const q38_qsa_shape *s);
/* select per row (sel row s at s*sel_row): take = min(nblk, budget/R) blocks by score desc,
 * ties -> lower block (q38_block_desc); sel[s] = [count, positions...]: each taken block's R
 * positions in rank order, then the tail nblk*R .. pos; count = -1 when take == nblk (every
 * position 0..pos, contiguous). Scores must be >= 0 and not NaN (their bits are ranked).
 * One 256-thread group a row: a radix select of the cutoff, then a bitonic sort of the taken
 * blocks, so the list is the CPU's exactly. budget/R <= 2048; sel_row >= budget/R*R + R. */
int q38_qsa_select(q36_tensor *sel, q36_tensor *sc, int S, int pos_base, int nbmax, int sel_row, const q38_qsa_shape *s);
/* attention over the selection, per row s and query head h (KV head h / (heads/kv_heads)):
 * softmax_j(q[s][h] . K[sel_j] / sqrt(D)) . V[sel_j], then *= sigmoid(gate[s][h]). q (already
 * rms0'd and roped), gate and out are [S][heads][D]; kc, vc [kv_heads][cap][D] floats, or
 * halves when kv_half. A count of -1 means positions 0..pos_base+s. Online softmax over tiles
 * of 256 positions in list order, fixed-order sums: exact and fast are the same kernel.
 * D <= 256. (pos_base and cap are not in the plan's signature; the contiguous rows and the
 * cache layout need them.) */
int q38_attn_sel(q36_tensor *out, q36_tensor *q, q36_tensor *gate, q36_tensor *kc, q36_tensor *vc,
                 q36_tensor *sel, int S, int pos_base, int heads, int kv_heads, int D, int cap, int sel_row,
                 int kv_half);

/* Attention glue (qwen38_core.h's q38_attention, 3460-3504 and 3568-3592). Element-wise, no
 * reductions but idx_prep's rms0 (fixed order): exact and fast are the same kernels. */
/* x is rows x heads x D, in place: q38_rope at position pos_base + r (row r), the NeoX half
 * layout: dims i < rotary/2 pair with i + rotary/2, angle pos / base_i with base_i =
 * powf(theta, 2i/rotary) on the host and the division rounded as the CPU's (q38_rope_angle);
 * dims >= rotary unchanged. rotary even, <= D, <= 256. */
int q38_rope_half(q36_tensor *x, int rows, int heads, int D, int rotary, int pos_base, float theta);
/* The q_proj output qraw is rows x heads x 2D, each head [q (D) | gate (D)] (the CPU reads q at
 * qp + (s*heads + h)*2D and the gate D after it): q and gate get rows x heads x D. */
int q38_qgate_split(q36_tensor *q, q36_tensor *gate, q36_tensor *qraw, int rows, int heads, int D);
/* k, v rows x kv_heads x D (k already rms0'd and roped) into the caches [kv_heads][cap][D] at
 * position pos_base + r. half = 0: floats; half = 1: halves (kc, vc 2 bytes an entry), the CPU's
 * kv16_round (clamped to +-65504, then rounded to nearest even), as q38_attn_sel's kv_half reads.
 * pos_base + rows <= cap. */
int q38_kv_append(q36_tensor *kc, q36_tensor *vc, q36_tensor *k, q36_tensor *v, int rows, int pos_base,
                  int kv_heads, int D, int cap, int half);
/* The indexer's per-row prep from its projection ip, rows x (IQ+1)*ID ([IQ query heads | one key]):
 * q[r][h] = rope(rms0(ip[r][h], idx_qn), pos) for h < IQ (rope over s->rotary dims, s->theta),
 * and ikc[pos] = ip[r][IQ], the raw key (no norm, no rope), pos = pos_base + r < cap. q is rows x
 * IQ x ID, ikc cap x ID. One 256-thread group a (row, head); ID <= 256. */
int q38_idx_prep(q36_tensor *q, q36_tensor *ikc, q36_tensor *ip, q36_tensor *idx_qn, int rows, int pos_base,
                 int cap, const q38_qsa_shape *s);

#ifdef __cplusplus
}
#endif
#endif
