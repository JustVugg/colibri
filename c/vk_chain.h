/* vk_chain.h -- a layer's dense chain on the Vulkan device, recorded into one submission.
 *
 * Why: the engines' dense part on Vulkan was one synchronous coli_vk_matmul per matrix
 * (about 726 submits and host round trips per Qwen3.8 decode token), slower on an
 * integrated GPU than the CPU. Here an engine records a whole layer -- norms,
 * projections, RoPE, attention over a KV cache that lives on the device, the Gated
 * DeltaNet recurrence with its state on the device, gates, the router logits, the
 * shared expert, the residual add -- into one command buffer, and the residual stream
 * stays on the device from one layer to the next. Only what the CPU needs crosses:
 * the rows the CPU's routed experts read and the router logits, then the routed sum
 * coming back.
 *
 * The pieces:
 *   - buffers (VkcBuf): VKC_DEV the device's own (state, scratch), VKC_UP written by the
 *     host and read by the device, VKC_DOWN written by the device and read by the host.
 *     They are sub-allocated from a few large memory blocks per kind (vk_alloc.h), so a
 *     submit references a handful of allocations, not hundreds.
 *   - recording: vkc_begin opens a command buffer (a ring of frames, each with its own
 *     fence and descriptor pool); every op records the barrier it needs (a buffer read or
 *     written after an earlier write, or written after a read, since the last barrier);
 *     vkc_submit(wait) sends it. Submissions run in order on the backend's main queue, a
 *     frame's first barrier orders it after everything submitted before it.
 *   - ops: vkc_matmul over the resident tensors the engines already upload
 *     (coli_vk_tensor_ensure), the GEMV per row or, from the backend's threshold, the
 *     fp32 tiled GEMM; and the chain's shaders (shaders/chain_*.comp), each documented
 *     at its top: chain_norm, chain_rope, chain_attn, chain_dnconv, chain_dnrec,
 *     chain_ew, chain_qsa, chain_ple. Offsets and strides are in floats.
 *   - multi-head latent attention: the MLA ops and the layer op below (chain_mla,
 *     chain_hgemv, chain_dsa, its k-pooled modes included), Kimi Delta Attention
 *     (chain_kda), manifold-constrained hyper-connections (chain_mhc) and Kimi K3's
 *     attention residuals and SiTU-GLU (chain_ares): loaded beside the others but optional.
 *   - a KV cache past the device's budget (chain_kvs, vkc_kvs_*; vk_kvsplit.h drives it):
 *     the attention over the part of the cache on the device, merged with the host's
 *     part; optional too.
 *
 * Threading: the engine thread only (the main queue is the backend's, used from the
 * same thread by coli_vk_matmul; the expert tier submits on its own queue).
 * A failed fence wait marks the device lost (coli_vk_mark_lost): vkc_lost() says so,
 * and every later call returns 0.
 *
 * tests/test_vk_chain.c checks every op against a CPU reference (make vk-chain-check). */
#ifndef COLI_VK_CHAIN_H
#define COLI_VK_CHAIN_H
#include <stddef.h>
#include <stdint.h>
#include "backend_vulkan.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct VkcBuf VkcBuf;
#define VKC_DEV  0
#define VKC_UP   1
#define VKC_DOWN 2
#define VKC_HOST 3   /* host memory read in place (vkc_host): bound read-only, never written */

int  vkc_init(void);          /* after coli_vk_init; 1 = the chain's pipelines are up */
int  vkc_ready(void);
int  vkc_lost(void);
void vkc_shutdown(void);      /* before coli_vk_shutdown (register it with atexit after it) */
/* The device the calls go to: 0 the primary, 1 COLI_VK_DEV2's (coli_vk_core_dev), each
 * with a context of its own that vkc_init opens and vkc_shutdown closes while it is
 * current. A buffer belongs to the device it was made on: bound or copied on the other
 * the op fails; vkc_free takes it back on its own device. Returns the previous device. */
int  vkc_device(int d);
int  vkc_device_now(void);
void vkc_shutdown_all(void);   /* both devices' contexts: what an engine registers with atexit */

VkcBuf *vkc_buf(size_t bytes, int kind);                 /* zero-filled; NULL when out of memory */
void    vkc_free(VkcBuf *b);                             /* waits for the frames that may read it */
int     vkc_reserve(VkcBuf **b, size_t bytes, int kind); /* at least `bytes`; growing drops the contents */
void   *vkc_ptr(const VkcBuf *b);                        /* host mapping (VKC_UP, VKC_DOWN; VKC_DEV when host-visible) */
size_t  vkc_bytes(const VkcBuf *b);
/* The pages around a host range as a buffer the device reads in place (coli_vk_host_buffer:
 * a device that shares the RAM, without staged uploads); *off: ptr's byte offset in it.
 * NULL when the device cannot. vkc_free releases it after the frames that read it. */
VkcBuf *vkc_host(const void *ptr, size_t bytes, size_t *off);

/* recording */
int  vkc_begin(void);
int  vkc_submit(int wait);
int  vkc_finish(void);        /* wait for every submitted frame */

/* transfers, recorded into the open frame (offsets and counts in floats) */
int  vkc_copy(VkcBuf *dst, size_t doff, VkcBuf *src, size_t soff, size_t n);
int  vkc_zero(VkcBuf *dst, size_t off, size_t n);
/* one copy command over n regions (a KV row per head, say) */
typedef struct { size_t dst, src, n; } VkcRegion;
int  vkc_copy_regions(VkcBuf *dst, VkcBuf *src, const VkcRegion *r, int n);
int  vkc_write(VkcBuf *dst, size_t off, const void *src, size_t bytes);   /* through the frame's staging */
/* synchronous: finishes what is in flight, copies, waits (the open frame, if any, is submitted first) */
int  vkc_read(VkcBuf *src, size_t off, void *dst, size_t bytes);

/* y[S][O] = x[S][I] @ W^T for a resident tensor (its fmt, I, O); x and y at float offsets */
int  vkc_matmul(ColiVkTensor *t, VkcBuf *x, size_t xo, VkcBuf *y, size_t yo, int S);
/* Rows from which vkc_matmul takes the tiled GEMM: -1 = the backend's rule (S >= 2 and
 * S*O >= 4096), 0 = never (an MTP verify, whose rows must get a decode step's bits). */
void vkc_gemm_rows(int rows);

/* chain_norm.comp */
typedef struct { int nseg, D, per_row, x_off, x_row, x_seg, y_off, y_row, y_seg, w_off, w_mod, flags; float eps, post; } VkcNorm;
#define VKC_NORM_ADD1 1
#define VKC_NORM_NOW  2
#define VKC_NORM_L2   4
int  vkc_norm(VkcBuf *x, VkcBuf *w, VkcBuf *y, const VkcNorm *p);
/* chain_rope.comp */
typedef struct { int nseg, per_row, x_off, x_row, x_seg, half_, cs_off, cs_row; } VkcRope;
int  vkc_rope(VkcBuf *x, VkcBuf *cs, const VkcRope *p);
/* chain_attn.comp (k_off/v_off: where the layer's cache starts in kc/vc) */
typedef struct { int S, H, KVH, hd, pos_base, cap, q_off, q_row, q_seg, g_off, g_row, g_seg, has_gate,
                 o_off, o_row, sel_off, sel_row; float scale; int k_off, v_off; } VkcAttn;
int  vkc_attn(VkcBuf *q, VkcBuf *kc, VkcBuf *vc, VkcBuf *o, VkcBuf *gate, VkcBuf *sel, const VkcAttn *p);
/* The same core with what MiMo adds (vkc_attn is this with every field below zero):
 * win a sliding window of that many positions (row s sees max(0, pos - win + 1)..pos);
 * ring the cache a ring of that many rows (position t in row t % ring); vd V's head dim
 * (0 = hd; also the output's per-head stride), at most 256; kv_pm position-major rows
 * (K[(row*KVH + kvh)*hd + d], V[(row*KVH + kvh)*vd + d]) instead of head-major; sink a
 * sink logit per head at snk[sink_off + h], which joins the softmax denominator only. */
typedef struct { VkcAttn a; int win, ring, vd, kv_pm, sink, sink_off; } VkcAttnW;
int  vkc_attn_w(VkcBuf *q, VkcBuf *kc, VkcBuf *vc, VkcBuf *o, VkcBuf *gate, VkcBuf *sel, VkcBuf *snk,
                const VkcAttnW *p);
/* From this many rows (COLI_VK_ATTN_BLOCK, 16; 0 = never) vkc_attn and vkc_attn_w take
 * chain_attnb.comp: a workgroup per KV head and block of rows, each K and V row read once
 * per block instead of once per (head, row); below it chain_attn as before. */
int  vkc_attn_block_rows(void);
/* chain_attnb's part mode (vk_kvsplit.h, the host's part on the device): each row's causal
 * positions before its share of a split cache, [first, (pos/B - anchor)*B), from a cache
 * in the host's head-major layout (p->a.cap positions a KV head), the value sum
 * unnormalized at o_off + s*o_row + h*vd and (m, l) at st_off + (s*H + h)*2. Blocked for
 * any S, its tiles at multiples of their size: a row's bits do not depend on the step
 * around it. No lists, sink or ring; 0 without the blocked pipeline. */
int  vkc_attn_part(VkcBuf *q, VkcBuf *kc, VkcBuf *vc, VkcBuf *o, const VkcAttnW *p, int blk, int anchor, int st_off);
/* The same over chunks of `ch` positions (a multiple of 16), nz of them, chunk z's part at
 * o_off + z*o_z + s*o_row + h*vd and (m, l) at st_off + z*st_z + (s*H + h)*2 (vkc_kvs_join
 * joins them): a decode row's positions over many workgroups. */
int  vkc_attn_part_chunks(VkcBuf *q, VkcBuf *kc, VkcBuf *vc, VkcBuf *o, const VkcAttnW *p, int blk, int anchor, int st_off,
                          int ch, int nz, int o_z, int st_z);
/* chain_kvs.comp mode 5: the chunks of a part joined in order (vkc_attn_part_chunks): segment
 * g of d floats, chunk z's sum at a_off + z*a_z + g*d and (m, l) at sa_off + z*sa_z + 2g of
 * parts, into out at o_off + g*d and so_off + 2g. */
typedef struct { int n, d, nz, a_off, a_z, sa_off, sa_z, o_off, so_off; } VkcKvsJoin;
int  vkc_kvs_join(VkcBuf *parts, VkcBuf *out, const VkcKvsJoin *p);
int  vkc_attn_flash_rows(void);   /* COLI_VK_CHAIN_FLASH: rows from which chain_attn_flash runs (0 = never) */
long long vkc_attn_slice_budget(void);   /* COLI_VK_ATTN_SLICE below: rows x positions x width a submission (0 = never) */
/* The attention ops (vkc_attn, vkc_attn_w, vkc_mla_core, vkc_relattn) whose rows x positions
 * x heads x head dim pass COLI_VK_ATTN_SLICE (2^32; 0 = never) record their rows in slices,
 * each ending its frame (submitted, not waited for) and the next in a new one: no single
 * submission of a big prompt chunk runs for seconds (a driver resets a job past its
 * timeout: amdgpu 10 s, Windows 2 s). The rows' arithmetic does not change. */
/* chain_dnconv.comp */
typedef struct { int S, CD, CK, in_off, in_row, out_off, out_row, snap_row, order, w_off, ring_off, snap_off; } VkcDnConv;
int  vkc_dnconv(VkcBuf *in, VkcBuf *w, VkcBuf *ring, VkcBuf *out, VkcBuf *snap, const VkcDnConv *p);
/* chain_dnrec.comp (KD a specialization constant, VD <= 128) */
typedef struct { int S, VH, KH, VD, Ktot, cv_off, cv_row, b_off, b_row, a_off, a_row, z_off, z_row,
                 y_off, y_row, snap_row, flags; float eps, qscale; int st_off, snap_off, prm_off; } VkcDnRec;
int  vkc_dnrec(int KD, VkcBuf *cv, VkcBuf *ab, VkcBuf *z, VkcBuf *st, VkcBuf *prm, VkcBuf *y, VkcBuf *snap, const VkcDnRec *p);
/* chain_ew.comp */
#define VKC_EW_ADD      0
#define VKC_EW_COMBINE  1
#define VKC_EW_SWIGLU   2
#define VKC_EW_HC_LOW   3
#define VKC_EW_HC_MIX   4
#define VKC_EW_HC_INJ   5
#define VKC_EW_HC_APPLY 6
#define VKC_EW_SCALE    7
#define VKC_EW_GATE_ADD 8
typedef struct { int op, n, D, C, flags, e_row, y_off, a_off, b_off, c_off, e_off; float fc; } VkcEw;
int  vkc_ew(VkcBuf *y, VkcBuf *a, VkcBuf *b, VkcBuf *c, VkcBuf *e, const VkcEw *p);
/* chain_qsa.comp (mode 0: nb block keys from b0; mode 1: S rows' selections) */
typedef struct { int mode, ID, R, b0, half_, S, pos_base, budget, IQ, q_off, q_row, nbmax, sel_row; float eps;
                 int src_off, w_off, pk_off, nb; } VkcQsa;
int  vkc_qsa(VkcBuf *src, VkcBuf *w, VkcBuf *pk, VkcBuf *cs, VkcBuf *sc, VkcBuf *sel, const VkcQsa *p);
/* chain_ple.comp (mode 0: the gate over S*C (row, stream) pairs; mode 1: the convolution) */
typedef struct { int mode, S, C, H, CK, NG, keys_off, hyp_off, val_off, snap_row, snap_off; float eps;
                 int prm_off, conv_off, ring_off; } VkcPle;
int  vkc_ple(VkcBuf *keys, VkcBuf *hyp, VkcBuf *val, VkcBuf *prm, VkcBuf *gated, VkcBuf *normv,
             VkcBuf *conv, VkcBuf *ring, const VkcPle *p);

/* ---- multi-head latent attention (MLA) -------------------------------------------
 * The attention of GLM-5.2, GLM-5.3, DeepSeek V3-style and Kimi K3's MLA layers, for S
 * rows (one at decode, a prompt chunk at prefill), with the KV cache on the device. Per
 * head h the query has Q no-position floats and R rotated ones; the cache holds, per
 * position, the normalized latent (K floats, kv_lora) and the rotated shared key (R).
 * Weight absorption: the no-position query enters the latent space once per head
 * (qa = W_k^T q_nope), the scores are qa . latent + q_rot . rope_key, and the value
 * rows apply once to the softmax-weighted latent (ctx = W_v clat). Nothing assumes a
 * model's shapes: H, Q, R (0 = NoPE), V, K, q_lora (0 = no q latent), the RoPE style,
 * the softmax scale (YaRN's mscale^2 included by the caller) and the cos/sin table
 * (YaRN's frequencies and mscale included by the caller) are all the caller's.
 *
 * Two levels. The ops (shaders/chain_mla.comp, chain_hgemv.comp, chain_dsa.comp):
 *   vkc_mla_core   the attention core over the cache: scores, online softmax, clat;
 *                  the causal range or a selection list (a DSA indexer's)
 *   vkc_mla_hgemv  per-head block matmuls of a resident tensor (the absorbed query
 *                  from kv_b's key rows, transposed, or from a [H*K x Q] key matrix;
 *                  the value rows; an optional sigmoid gate on the values)
 *   vkc_mla_rope   RoPE from a host cos/sin table, rotate-half or interleaved pairs in
 *                  and halves out, in place or into a cache row
 *   vkc_mla_lnorm  LayerNorm with weight and bias (the indexer's key norm)
 *   vkc_dsa_select a token-level DSA indexer's scores and top-k, in the CPU's order
 * and the layer op, vkc_mla_qkv + vkc_mla_attn (or vkc_mla for both): q_a, its norm,
 * q_b (or q straight from the hidden rows), kv_a, the latent norm, RoPE, the new rows
 * into the cache (and a copy for the host's), then the absorbed core, the values, the
 * gate and o_proj. Between the two an engine records what reads the projections (a
 * DSA indexer reads the normalized q latent, s->qa).
 * The cache is the caller's (VkcMlaCache): positions [kv_start, pos_base) must be there
 * before a step from pos_base (the engine keeps a watermark, as for the GQA caches);
 * the step writes [pos_base, pos_base + S).
 * Limits: K <= 1024, R <= 128 and even, Q <= 1024 for kv_b's transposed absorption,
 * IH <= 64 and IH*ID <= 4096 for the indexer. The ops return 0 (nothing recorded) when
 * the MLA shaders are missing or a limit is passed; vkc_mla_ready() says whether the
 * shaders are there. */
int vkc_mla_ready(void);

/* chain_mla.comp mode 0: clat[s][h] (o_off + s*o_row + h*o_seg) from the absorbed query
 * qabs (qa_off + s*qa_row + h*qa_seg, K floats) and the rotated query (qr_off + ..., R),
 * over lat (lat_off + t*lat_row) and rope (rope_off + t*rope_row) at the positions
 * kv_start..pos_base+s or sel's list (sel_row 0: none; sel[sel_off + s*sel_row] = count,
 * -1 the causal range, then the positions, a negative one skipped). rope may be NULL with R 0. */
typedef struct { int S, H, K, R, pos_base, kv_start, qa_off, qa_row, qa_seg, qr_off, qr_row, qr_seg,
                 lat_off, lat_row, rope_off, rope_row, sel_off, sel_row, o_off, o_row, o_seg; float scale; } VkcMlaCore;
int vkc_mla_core(VkcBuf *qabs, VkcBuf *qr, VkcBuf *lat, VkcBuf *rope, VkcBuf *sel, VkcBuf *clat, const VkcMlaCore *p);
/* chain_mla.comp modes 1 and 2: nseg segments, segment g at row g / per_row, index
 * g % per_row: x_off + row*x_row + j*x_seg (y likewise; y may be x).
 * RoPE: the first rd floats rotate with the cos/sin pairs at cs_off + row*cs_row, the
 * rest of seg_len is copied when y is elsewhere. LayerNorm: seg_len floats, weight at
 * w_off and bias at b_off (has_b) in prm. */
#define VKC_ROPE_HALF        0   /* a = x[i], b = x[i + rd/2] */
#define VKC_ROPE_INTERLEAVED 1   /* a = x[2i], b = x[2i + 1], written to i and i + rd/2 */
typedef struct { int nseg, per_row, rd, seg_len, style, x_off, x_row, x_seg, y_off, y_row, y_seg,
                 cs_off, cs_row, w_off, b_off, has_b; float eps; } VkcMlaRow;
int vkc_mla_rope(VkcBuf *x, VkcBuf *cs, VkcBuf *y, const VkcMlaRow *p);
int vkc_mla_lnorm(VkcBuf *x, VkcBuf *prm, VkcBuf *y, const VkcMlaRow *p);
/* chain_hgemv.comp: head h's block is t's rows h*hstride + hoff + [0, n).
 * trans 0: y[s][h][o] = W[row o] . x[s][h][0..I), o < n (has_gate: times
 *          sigmoid(gate[g_off + s*g_row + h*n + o]));
 * trans 1: y[s][h][i] = sum_d W[row d][i] * x[s][h][d], d < n, i < I.
 * x[s][h] at x_off + s*x_row + h*x_seg, y[s][h] at y_off + s*y_row + h*y_seg. */
typedef struct { int trans, S, H, n, hstride, hoff, x_off, x_row, x_seg, y_off, y_row, y_seg, g_off, g_row, has_gate; } VkcHgemv;
int vkc_mla_hgemv(ColiVkTensor *t, VkcBuf *x, VkcBuf *y, VkcBuf *gate, const VkcHgemv *p);
/* chain_dsa.comp mode 0: row s (position pos_base + s) scores its pos_base+s+1 positions
 * as (sum_h [d_h > 0] w_h d_h) * wscale, d_h = (q_h . k_t) * qscale (q at q_off +
 * s*q_row, IH x ID; w at w_off + s*w_row; k_t at k_off + t*k_row), into sc[s*sc_row + t],
 * and writes sel[s*sel_row] = keep, then the positions: those above the keep-th largest
 * score in position order, then those equal to it in position order. With no more
 * positions than topk (and force 0), sel[s*sel_row] = -1: every position. */
typedef struct { int S, pos_base, IH, ID, topk, force, q_off, q_row, w_off, w_row, k_off, k_row, sc_row, sel_row;
                 float qscale, wscale; } VkcDsa;
int vkc_dsa_select(VkcBuf *iq, VkcBuf *hw, VkcBuf *keys, VkcBuf *sc, VkcBuf *sel, const VkcDsa *p);

/* chain_dsa.comp with k-pooling (GLM-5.3): mode 1, the pooled keys of the np complete
 * pools from p0 on, each the per-channel softmax mixture (gate logits plus ape, at
 * ape_off in prm, [pool][ID]) of its members' keys, into pk[pk_off + p*ID]; mode 2, row
 * s's selection: the complete pools visible to position pos_base + s scored
 * sum_h [dot > 0] (w_h / wdiv) * dot * scale (dot = q_h . pk[p]), the top topk/pool in
 * rank order filling `pool` slots each, the incomplete tail from slot topk (tail), -1
 * elsewhere; sel[s*sel_row] = topk (+ pool - 1 with the tail). sc: [S][sc_row] scores. */
typedef struct { int np, pool, p0, ID, g_off, g_row, ape_off, pk_off, k_off, k_row; } VkcDsaPool;
int vkc_dsa_pool_keys(VkcBuf *keys, VkcBuf *gates, VkcBuf *prm, VkcBuf *pk, const VkcDsaPool *p);
typedef struct { int S, pos_base, IH, ID, topk, pool, q_off, q_row, w_off, w_row, pk_off, tail, sc_row, sel_row;
                 float wdiv, scale; } VkcDsaPick;
int vkc_dsa_pool_select(VkcBuf *iq, VkcBuf *hw, VkcBuf *pk, VkcBuf *sc, VkcBuf *sel, const VkcDsaPick *p);

/* The layer op. Tensors are the resident copies the per-matrix path uploads
 * (coli_vk_tensor_ensure), in any format the backend holds. */
typedef struct {
    int H, Q, R, V, K;            /* heads; per head qk_nope, qk_rope (0 = NoPE), v_head; kv_lora */
    int D, q_lora;                /* hidden; q_lora 0: no q latent, q_b reads the hidden rows */
    float eps;                    /* the latent RMSNorms' eps */
    float scale;                  /* the softmax scale */
    int rope_style;               /* VKC_ROPE_HALF or VKC_ROPE_INTERLEAVED */
    ColiVkTensor *q_a;            /* [q_lora x D]; NULL with q_lora 0 */
    ColiVkTensor *q_b;            /* [H*(Q+R) x (q_lora or D)] */
    ColiVkTensor *kv_a;           /* [K+R x D]: the latent, then the shared key's rotated part */
    ColiVkTensor *kv_b;           /* [H*(Q+V) x K], per head Q key rows then V value rows; or: */
    ColiVkTensor *k_abs, *v_abs;  /*   [H*K x Q] (W_k^T per head) and [H*V x K] */
    ColiVkTensor *o;              /* [D x H*V]; NULL: the context stays in s->ctx */
    VkcBuf *prm;                  /* the norm weights: q latent at q_norm (unused with q_lora 0), */
    size_t q_norm, kv_norm;       /*   kv latent at kv_norm (floats) */
} VkcMla;
typedef struct { VkcBuf *qa, *q, *kv, *qabs, *clat, *ctx; int rows; } VkcMlaScratch;
typedef struct { VkcBuf *lat, *rope; int cap; } VkcMlaCache;   /* [cap][K] and [cap][R] (rope NULL with R 0) */
/* scratch for `rows` rows (grows, never shrinks); free releases it */
int  vkc_mla_scratch(VkcMlaScratch *s, const VkcMla *m, int rows);
void vkc_mla_scratch_free(VkcMlaScratch *s);
/* x: S rows of D floats at x_off (the layer's normalized input). cs: the cos/sin pairs
 * of the S positions, R floats a row (NULL with R 0). Leaves s->qa = the normalized q
 * latent [S][q_lora] and s->q = the queries [S][H*(Q+R)], rotated; writes the cache's
 * rows [pos_base, pos_base+S) and, when down is given, copies them to down at down_off
 * (S*K latent floats, then S*R rope floats) for the host's cache. */
int vkc_mla_qkv(const VkcMla *m, VkcMlaScratch *s, VkcBuf *x, size_t x_off, int S, int pos_base,
                VkcBuf *cs, VkcMlaCache *c, VkcBuf *down, size_t down_off);
/* The core over positions kv_start..pos_base+s (or sel's lists, sel_row > 0), the values
 * (times sigmoid(gate[gate_off + s*H*V + h*V + v]) when gate is given) into s->ctx, then
 * out[out_off + s*D] = o(ctx) when m->o and out are given. */
int vkc_mla_attn(const VkcMla *m, VkcMlaScratch *s, int S, int pos_base, int kv_start, VkcMlaCache *c,
                 VkcBuf *sel, size_t sel_off, int sel_row, VkcBuf *gate, size_t gate_off, VkcBuf *out, size_t out_off);
/* The same for rows of different sequences (a multiplexed decode step): row s attends its
 * own cache c[s] over positions 0..pos[s] (or sel's list at sel_off + s*sel_row; sel's
 * row count -1 is that causal range); the projections take every row at once, the core
 * one row at a time. */
int vkc_mla_attn_rows(const VkcMla *m, VkcMlaScratch *s, int S, const int *pos, VkcMlaCache *const *c,
                      VkcBuf *sel, size_t sel_off, int sel_row, VkcBuf *out, size_t out_off);
/* both, the causal range */
int vkc_mla(const VkcMla *m, VkcMlaScratch *s, VkcBuf *x, size_t x_off, int S, int pos_base, int kv_start,
            VkcBuf *cs, VkcMlaCache *c, VkcBuf *out, size_t out_off);

/* ---- Kimi Delta Attention (chain_kda.comp) --------------------------------------------
 * delta_attention.h's coli_kda_step for S rows in order, the state and the short
 * convolution's window on the device (GLM-5.3's linear layers; Kimi K3's KDA).
 * vkc_kda_conv: C = 3P channels (q, k, v), the input of part c / P at in_off + part*in_part
 *   + s*in_row + c % P, the window [C][K] at win_off (the last K inputs, oldest first, as
 *   the CPU keeps it), taps [C][K] at w_off; out[out_off + s*out_row + c] = silu(sum).
 * vkc_kda_rec: one workgroup per head, the key dim KD a specialization (<= 256), VD <= 128;
 *   m = the convolution's output rows (q at h*KD, k at P + h*KD, v at 2P + h*VD); f, b, g
 *   the raw decay, beta and output-gate projections; prm at prm_off: A_log[H], dt[P],
 *   norm[VD]; alpha = exp(lb * sigmoid(exp(A_log) * (f + dt))), beta = sigmoid(b), q and k
 *   l2-normalized with neps inside the root; y = RMSNorm(o, eps) * norm * sigmoid(g).
 *   The state [H][KD][VD] at st_off.
 * vkc_kda_rec_flags: the same with flags (0 is vkc_kda_rec, GLM-5.3's arithmetic);
 *   Kimi K3's (kimi_k3.c's kda_forward) sets both:
 *   VKC_KDA_EXP_A   prm holds exp(A_log) itself, used as it is
 *   VKC_KDA_K3      the l2 sums take neps after the squares, q is normalized, then
 *                   scaled, and the update is k * ((v - mem) * beta) */
int vkc_kda_ready(void);
typedef struct { int S, C, K, P, in_off, in_row, in_part, out_off, out_row, w_off, win_off; } VkcKdaConv;
int vkc_kda_conv(VkcBuf *in, VkcBuf *w, VkcBuf *win, VkcBuf *out, const VkcKdaConv *p);
#define VKC_KDA_EXP_A 1
#define VKC_KDA_K3    2
typedef struct { int S, H, VD, P, m_off, m_row, f_off, f_row, b_off, b_row, g_off, g_row, y_off, y_row, st_off, prm_off;
                 float lb, neps, eps; } VkcKdaRec;
int vkc_kda_rec(int KD, VkcBuf *m, VkcBuf *f, VkcBuf *b, VkcBuf *g, VkcBuf *prm, VkcBuf *st, VkcBuf *y, const VkcKdaRec *p);
int vkc_kda_rec_flags(int KD, VkcBuf *m, VkcBuf *f, VkcBuf *b, VkcBuf *g, VkcBuf *prm, VkcBuf *st, VkcBuf *y,
                      const VkcKdaRec *p, int flags);

/* ---- manifold-constrained hyper-connections (chain_mhc.comp) ------------------------
 * hyper_connections.h for S rows of H <= 8 streams of D floats ([S][H*D] at x_off,
 * x_row apart), GLM-5.3's (and DeepSeek V4's) residual:
 *   VKC_MHC_SPLIT     m: the raw mix products hc_fn . x ([S][(2+H)*H], from a matmul),
 *                     scaled by the rows' 1/rms (eps); pre, post and the Sinkhorn-projected
 *                     comb (iters, hc_eps; scale[3] and base at prm_off in prm) into
 *                     hp[hp_off + s*hp_row]: H pre, H post, H*H comb
 *   VKC_MHC_COLLAPSE  y[s][d] = sum_i pre[i] x[s][i*D + d]
 *   VKC_MHC_POST      y[s][j*D + d] = sum_i comb[i*H + j] x[s][i*D + d] + post[j] m[s][d]
 *   VKC_MHC_MEAN      y[s][d] = (sum_i x[s][i*D + d]) / H
 *   VKC_SWIGLU_CLAMP  y[i] = silu(min(x[i], lim)) * clamp(m[i], -lim, lim), i < n */
#define VKC_MHC_SPLIT    0
#define VKC_MHC_COLLAPSE 1
#define VKC_MHC_POST     2
#define VKC_MHC_MEAN     3
#define VKC_SWIGLU_CLAMP 4
int vkc_mhc_ready(void);
typedef struct { int S, H, D, iters, x_off, x_row, m_off, m_row, hp_off, hp_row, y_off, y_row, prm_off, n;
                 float eps, hc_eps, lim; } VkcMhc;
int vkc_mhc(int mode, VkcBuf *x, VkcBuf *m, VkcBuf *hp, VkcBuf *prm, VkcBuf *y, const VkcMhc *p);
/* Inkling's ops, their pipelines made on first use (an engine checks *_ready at setup:
 * a build without the shader keeps every other op, and that engine's chain off).
 * chain_sconv.comp (mode 0: the depthwise causal short convolution, residual inside, in
 * place, its ring carried; mode 1: x *= fc over n floats; mode 2: x /= fc) */
typedef struct { int mode, S, C, CK, x_off, x_row, w_off, ring_off, n; float fc; } VkcSconv;
int  vkc_sconv_ready(void);
int  vkc_sconv(VkcBuf *x, VkcBuf *w, VkcBuf *ring, const VkcSconv *p);
/* chain_relattn.comp: attention with a relative-position bias bank, a per-row scale
 * tau and a sliding window over a ring cache, the step's own rows read from kvs */
typedef struct { int S, H, KVH, hd, pos_base, cap, window, ext, d_rel;
                 int q_off, q_row, o_off, o_row, k_off, v_off, ks_off, vs_off, kv_row;
                 int r_off, r_row, relp_off, tau_off; float scale; } VkcRelAttn;
int  vkc_relattn_ready(void);
int  vkc_relattn(VkcBuf *q, VkcBuf *kc, VkcBuf *vc, VkcBuf *o, VkcBuf *kvs, VkcBuf *r, VkcBuf *relp, VkcBuf *tau,
                 const VkcRelAttn *p);

/* ---- attention residuals and SiTU-GLU (chain_ares.comp) ----------------------------
 * Kimi K3's residual stream (AttnRes) and its activation, for S rows:
 *   vkc_ares_mix  kimi_k3.c's res_mix: row s mixes the nb block snapshots ([nb][D] at
 *                 b_off + s*b_row) and the running prefix (x_off + s*x_row) by the
 *                 softmax of (v . w) / sqrt(mean(v^2) + eps), w at w_off in prm, into
 *                 y_off + s*y_row; nb <= 15, y apart from x and blk
 *   vkc_situ      y[i] = b1*tanh(g/b1)*sigmoid(g)*b2*tanh(u/b2), i < n, in the CPU's order */
int vkc_ares_ready(void);
typedef struct { int S, D, nb, x_off, x_row, b_off, b_row, w_off, y_off, y_row; float eps; } VkcAres;
int vkc_ares_mix(VkcBuf *x, VkcBuf *blk, VkcBuf *prm, VkcBuf *y, const VkcAres *p);
typedef struct { int n, g_off, u_off, y_off; float b1, b2; } VkcSitu;
int vkc_situ(VkcBuf *g, VkcBuf *u, VkcBuf *y, const VkcSitu *p);

/* ---- DeepSeek V4.1 Flash and DeepSeek V4 attention (chain_dsv4.comp) -----------------
 * The model is MQA over one KV row per position (the same row is key and value): a
 * sliding window of raw rows, plus compressed rows (a compressor pools `ratio`
 * positions into one) that a DSA indexer picks per query. Optional like the MLA ops:
 * vkc_dsv4_ready() says whether the shader is there; a missing one turns off these ops.
 *   vkc_dsv4_attn     sparse attention with a sink over a per-row list (sparse_attn.h):
 *                     entry e < 0 skipped, e < nwin the window ring's row e, else the
 *                     compressed row e - nwin; scores (q . k) * scale, the sink in the
 *                     denominator only, the value sum and the denominator in list order.
 *                     flags 1: the weights round to bf16 before the value sum and the
 *                     output to bf16 (DeepSeek V4). Limits: hd <= 1024, cnt <= 3072.
 *   vkc_dsv4_rope     RoPE on interleaved pairs in place, the first rd floats of each
 *                     segment, (cos, sin) pairs from a host table; inverse negates the sine
 *                     (the attention output's un-rotation). flags 1: round to bf16.
 *   vkc_dsv4_compress the compressor's rolling group for S rows in order: each row's kv
 *                     and score rows (P floats; score + ape[slot] when ape_off >= 0) into
 *                     the ring row pos % ratio (ratio + that with overlap), and at each
 *                     completed group the per-channel softmax pooling of the ring rows
 *                     (overlap: the first ratio rows read channel d, the next ratio rows
 *                     channel D + d; the second half then moves to the first) into
 *                     out[pos / ratio]. ring: (1 + overlap) * ratio kv rows of P at
 *                     ring_off, then as many score rows.
 *   vkc_dsv4_score    the indexer's scores: row s (position pos_base + s) scores columns
 *                     j < lens = (pos + 1) / ratio (and mask[mask_off + s*mask_row + j] != 0
 *                     when mask_row > 0) as sum_h [dot > 0] dot * hw_h * wscale, dot = q_h .
 *                     key[j]; every other j < width scores -inf; into sc[sc_off + s*sc_row
 *                     + j]. IH <= 4096; IH <= 64 and IH*ID <= 4096 stage the queries in
 *                     shared memory, larger ones read them from memory (the same sums).
 *   vkc_dsv4_cand     the candidate blocks: per row, each block's best score, the block of
 *                     column lens - 1 pinned, then the best min(topb, blocks) blocks (the
 *                     lower on a tie), mask 1 on their columns. At most 4096 blocks.
 *   vkc_dsv4_topk     per row the min(topk, finite scores) largest, ties to the lower
 *                     column, as base + j into the list, ascending (order 0) or by rank
 *                     (order 1); -1 in the rest of the topk slots. topk <= 4096 for order 1.
 *   vkc_dsv4_engram   the engram gate: per (stream c, row s) the stream gains
 *                     sigmoid(signed sqrt(x . (qw * kw * key) * rms(x)^-1 * rms(key)^-1 /
 *                     sqrt(D))) * value; kv[s] = H keys of D, then the value.
 * Offsets and strides in floats; lists and masks are int32 buffers. */
int vkc_dsv4_ready(void);
typedef struct { int S, H, hd, cnt, l_off, l_row, nwin, w_off, c_off, q_off, q_row, o_off, o_row, sink_off, flags;
                 float scale; } VkcDsAttn;
int vkc_dsv4_attn(VkcBuf *q, VkcBuf *win, VkcBuf *cmp, VkcBuf *list, VkcBuf *prm, VkcBuf *out, const VkcDsAttn *p);
typedef struct { int nseg, per_row, rd, x_off, x_row, x_seg, cs_off, cs_row, inverse, flags; } VkcDsRope;
int vkc_dsv4_rope(VkcBuf *x, VkcBuf *cs, const VkcDsRope *p);
typedef struct { int S, pos_base, ratio, P, D, overlap, kv_off, kv_row, sc_off, sc_row, ape_off, out_off, out_row,
                 ring_off; } VkcDsComp;
int vkc_dsv4_compress(VkcBuf *kv, VkcBuf *sc, VkcBuf *ring, VkcBuf *prm, VkcBuf *out, const VkcDsComp *p);
typedef struct { int S, pos_base, ratio, IH, ID, width, q_off, q_row, w_off, w_row, k_off, k_row, mask_row, sc_row;
                 float wscale; int sc_off, mask_off; } VkcDsScore;
int vkc_dsv4_score(VkcBuf *iq, VkcBuf *hw, VkcBuf *keys, VkcBuf *mask, VkcBuf *sc, const VkcDsScore *p);
typedef struct { int S, pos_base, ratio, width, block, topb, sc_row, mask_row; } VkcDsCand;
int vkc_dsv4_cand(VkcBuf *sc, VkcBuf *mask, const VkcDsCand *p);
typedef struct { int S, width, topk, sc_row, l_off, l_row, base, order; } VkcDsTopk;
int vkc_dsv4_topk(VkcBuf *sc, VkcBuf *list, const VkcDsTopk *p);
typedef struct { int S, H, D, kv_off, kv_row, x_off, x_row, qw_off, kw_off; float eps; } VkcDsEngram;
int vkc_dsv4_engram(VkcBuf *kv, VkcBuf *prm, VkcBuf *x, const VkcDsEngram *p);
/* DeepSeek V4's roundings, each the engine's C bit for bit (chain_dsv4.comp modes 7, 8):
 *   vkc_dsv4_round    nseg segments of len floats (segment g at row g / per_row, index
 *                     g % per_row: x_off + row*x_row + j*x_seg, y likewise) from x into y
 *                     (y may be x: in place), as `kind`:
 *                       VKC_DS_BF16      to bf16, nearest even (coli_bf16_round)
 *                       VKC_DS_E4M3      E4M3 per block of `block` values with one
 *                                        power-of-two scale (coli_fp8_activation_qdq_ref)
 *                       VKC_DS_E2M1      E2M1 per block (coli_fp4_activation_qdq_ref)
 *                       VKC_DS_HADAMARD  the Hadamard transform times hscale (the host's
 *                                        1 / sqrtf(len)), to bf16 (coli_hadamard_bf16_ref)
 *                     flags 1: the E4M3 or E2M1 result to bf16 after. Limits: block <= 256;
 *                     the Hadamard's len a power of two up to 4096.
 *   vkc_dsv4_swiglu   y[y_off + i] = bf16(g * sigmoid(g) * u), g = bf16(a[a_off + i]) and
 *                     u = bf16(b[b_off + i]) clamped by lim when lim > 0, i < n (the shared
 *                     expert's activation between its roundings; coli_v4_swiglu). */
#define VKC_DS_BF16     0
#define VKC_DS_E4M3     1
#define VKC_DS_E2M1     2
#define VKC_DS_HADAMARD 3
typedef struct { int kind, nseg, per_row, len, block, flags, x_off, x_row, x_seg, y_off, y_row, y_seg; float hscale; } VkcDsRound;
int vkc_dsv4_round(VkcBuf *x, VkcBuf *y, const VkcDsRound *p);
typedef struct { int n, a_off, b_off, y_off; float lim; } VkcDsSwiglu;
int vkc_dsv4_swiglu(VkcBuf *a, VkcBuf *b, VkcBuf *y, const VkcDsSwiglu *p);

/* Big prefill chunks: the prompt rows an engine's chain runs per chunk. COLI_VK_CHAIN_ROWS=n
 * takes n (1..65535). Unset (or "auto"): the most rows, up to COLI_VK_CHAIN_ROWS_MAX
 * (8192), whose buffers take at most half of the device memory free at the engine's first
 * call (VK_EXT_memory_budget, else a quarter of the device-local heap; capped by
 * available host RAM), row_bytes a row (the engine's chain
 * scratch plus the routed experts' outputs for that row); a multiple of 256 rows below
 * the cap; below 256 keep the rows that fit, at least one. Decided once per engine
 * and printed as a [VK] line. */
int vkc_chunk_rows(const char *engine, size_t row_bytes);
/* 1 when the chunk is the budget's (COLI_VK_CHAIN_ROWS unset or "auto"): an engine that
 * blocks a prompt on the host before the chain (MiMo's MIMO_CHUNK, DeepSeek V4's
 * V4_PREFILL_CHUNK) then hands the chain blocks of the chain's chunk; with
 * COLI_VK_CHAIN_ROWS set it keeps its own blocks, as before. */
int vkc_chunk_auto(void);

/* ---- a partial chain: the first N layers on the device (docs/vulkan.md, "A partial chain")
 * When the dense layers do not all fit the device, the chain takes a contiguous prefix of
 * N layers (their matrices, the residual and their state, as for every layer before) and
 * the CPU runs the other L - N layers and the head; the residual (and an engine's per-token
 * stream state) moves to the host once per forward, after layer N - 1. N = L is the full
 * chain; N = 0 is the chain off, nothing uploaded, every byte left to the expert tier.
 *
 * vkc_fit decides N at startup, before any upload and before the RAM plan sizes the expert
 * cache, from the device's free memory (coli_vk_free_bytes) less:
 *   - the reserve, COLI_VK_TIER_RESERVE_GB (1 GiB): what the tier leaves for scratch, KV
 *     mirrors and the driver;
 *   - fixed: the engine's fixed_bytes (its parameter buffers, the scratch of one prompt
 *     chunk of vkc_fit_rows(..) rows, buffers every layer shares) plus the pools' own
 *     granularity (vkc_fit_pools: a weight block, a block of each chain pool, the frames'
 *     staging);
 *   - each layer's bytes, layer_bytes[i]: its matrices (vkc_fit_tensor each) and its state
 *     at its starting size (vkc_fit_buf each; the KV split covers the growth);
 *   - tail_bytes: what the engine puts on the device only with every layer there (the head,
 *     a final norm, matrices the per-matrix path would upload lazily).
 * The rule: N = L when fixed + every layer fits; the tail goes up too (fit->tail) when it
 * fits beside them; else N = the most layers from layer 0 with fixed + their bytes within
 * the room, the tail on the CPU. COLI_VK_CHAIN_LAYERS=n forces N (capped at L; 0 = the chain
 * off; the tail goes up when n = L). When everything fits the result is the full chain, as
 * before. Two lines on stderr:
 *   [VK] <engine> chain fit: free F B, reserve R B, fixed X B (the engine's E B, the pools' P B), tail T B, layers b0 b1 .. B, matrices m0 m1 .. B
 *   [VK] <engine> chain: N of L layers on the device (S), L-N on the CPU (free F, reserve R)
 * (the second with "(COLI_VK_CHAIN_LAYERS=n)" when forced; with N = 0 it says the chain stays
 * off). matrix_bytes[i] (NULL: not printed) is the payload of layer i's matrices as
 * coli_vk_mem_info counts it (coli_vk_tensor_payload each), for vkc_fit_placed's check.
 * fit->per and fit->mat are copies the fit keeps. Returns N.
 *
 * vkc_fit_shrink: a layer that did not fully reach the device during setup (a driver
 * refusing what the budget promised, an upload that failed): the engine freed everything
 * of that layer (and of the layers after it, if any were placed) and continues with N = n,
 * the layers before it. The line, in the same format:
 *   [VK] <engine> chain: n of L layers on the device (S), L-n on the CPU (layer n did not reach the device: why; what it had placed was freed)
 * vkc_fit_mark: the engine finished layer i's setup (COLI_VK_STAGED_FAULT's point count is
 * noted for the placed line, so a test can aim a fault inside a given layer).
 * vkc_fit_placed: after setup, before the tier sizes itself:
 *   [VK] <engine> chain: N of L layers placed: M B of matrices on the device (the fit counted M' B for these layers), chain buffers C B, device memory held H B[; COLI_VK_STAGED_FAULT's point reached c0,c1,.. times by the end of each layer]
 * vkc_fit_partial: 1 when something the full chain would place stays on the CPU (N < L, or
 * the tail kept off): the engine's per-matrix path must then not upload matrices that are
 * not on the device already (the CPU layers and the tail stay on the CPU).
 * vkc_layer_free: frees a layer's tensors (coli_vk_tensor_free) and chain buffers (vkc_free)
 * after the frames that may read them, setting each pointer to NULL. */
typedef struct {
    int L, n;                   /* the model's layers; the first n run on the device */
    int forced, tail;           /* COLI_VK_CHAIN_LAYERS decided; the tail goes on the device */
    size_t free_b, reserve, fixed, engine_fixed, pools, tail_b, used_b;   /* used_b: the n layers' bytes */
    size_t *per, *mat;          /* each layer's bytes and its matrices' payload */
    unsigned long long *marks;  /* COLI_VK_STAGED_FAULT's count at the end of each layer's setup */
} VkcFit;
int    vkc_fit(const char *engine, int L, const size_t *layer_bytes, const size_t *matrix_bytes,
               size_t fixed_bytes, size_t tail_bytes, VkcFit *fit);
void   vkc_fit_shrink(const char *engine, VkcFit *fit, int n, const char *why);
void   vkc_fit_mark(VkcFit *fit, int layer);
void   vkc_fit_placed(const char *engine, const VkcFit *fit);
int    vkc_fit_partial(const VkcFit *fit);
/* COLI_VK_CHAIN_ROWS when set to a number, else def: the rows the fit's scratch is for */
int    vkc_fit_rows(int def);
size_t vkc_fit_tensor(int fmt, int I, int O, int gs);   /* a resident tensor's device bytes, its two ranges aligned */
size_t vkc_fit_buf(size_t bytes);                       /* a chain buffer's (vkc_buf) device bytes */
size_t vkc_fit_reserve(void);                           /* COLI_VK_TIER_RESERVE_GB in bytes */
size_t vkc_fit_pools(void);                             /* the pools' granularity the fit adds to fixed */
void   vkc_layer_free(ColiVkTensor **const *t, int nt, VkcBuf **const *b, int nb);

/* ---- a KV cache split between the device and the host (chain_kvs.comp) ----------------
 * Past the device's budget a layer's cache keeps `ns` blocks of B positions on the
 * device (the recent window, and where an engine selects positions, the blocks read
 * most); the rest stays in the host's RAM, which holds the whole cache anyway. The
 * device attends over its blocks and writes a partial result (the unnormalized value
 * sum and the softmax statistics m, l), the host attends over the rest (vk_kvsplit.h),
 * and vkc_kvs_merge joins the two through their statistics: the full attention up to
 * the order of the sums. The residency is an int buffer `tab`: bt[nblk] at bt_off
 * (block -> slot, -1 in the host's RAM only; bit 30 set: a block pinned by the reads of
 * a selection), position t in device row slot*B + t % B. The partition of a row depends
 * on its position only (the same bits however a forward is cut into steps): the device
 * takes the blocks pos/B - anchor .. pos/B (all of them on the device while a step
 * holds the row) and, from a list, the pinned blocks; the host the rest. Optional like
 * the MLA ops: vkc_kvs_ready() says whether the shader is there.
 *   vkc_kvs_attn   grouped-query attention (chain_attn's) over the device's rows: K/V
 *                  head-major (kvh*rows + r)*hd or position-major (kv_pm), a window, a
 *                  sink (it joins the device's part), a selection list (entries outside
 *                  the device skipped). VKC_KVS_FIN: every visible position is on the
 *                  device, the normalized output (VKC_KVS_GATE: times sigmoid(gate)) at
 *                  o_off + s*o_row + h*vd as vkc_attn writes it; else the value sum there
 *                  and (m, l) at st_off + (s*H + h)*2. hd, vd <= 256.
 *   vkc_kvs_mla    the MLA core (vkc_mla_core's) over the device's rows: the latent at
 *                  lat_off + r*lat_row, the rope key at rope_off + r*rope_row; outputs as
 *                  above with o_seg per head. K <= 1024, R <= 128.
 *   vkc_kvs_rel    Inkling's attention (vkc_relattn's) over a global layer's device rows.
 *   vkc_kvs_ds     DeepSeek's sparse attention (vkc_dsv4_attn's) over its window rows and
 *                  the compressed rows on the device.
 *   vkc_kvs_merge  segment g = (s = g / H, h) of d floats: the device's part (dev at
 *                  a_off + s*a_row + h*a_seg, (m, l) at sa_off + 2g) and the host's
 *                  (cpu, packed at c_off + g*d, (m, l) at sc_off + 2g) into out at o_off
 *                  + s*o_row + h*o_seg, VKC_KVS_GATE: times sigmoid(gate[g_off + s*g_row
 *                  + h*g_seg + i]). */
int vkc_kvs_ready(void);
#define VKC_KVS_FIN  1
#define VKC_KVS_GATE 2
#define VKC_KVS_COLD 32   /* vkc_kvs_attn/_mla/_rel: the host's positions, from a buffer in the host's layout (row t = position t) */
typedef struct { int S, H, KVH, hd, vd, pos_base, rows, q_off, q_row, q_seg, sel_off, sel_row; float scale;
                 int k_off, v_off, win, kv_pm, sink, sink_off, B, ns, bt_off, nblk, anchor, o_off, o_row, st_off, flags,
                 g_off, g_row, g_seg; } VkcKvsAttn;
int vkc_kvs_attn(VkcBuf *q, VkcBuf *kc, VkcBuf *vc, VkcBuf *out, VkcBuf *gate, VkcBuf *sel, VkcBuf *snk, VkcBuf *tab,
                 const VkcKvsAttn *p);
typedef struct { int S, H, K, R, pos_base, kv_start, qa_off, qa_row, qa_seg, qr_off, qr_row, qr_seg, lat_off, lat_row,
                 rope_off, rope_row, sel_off, sel_row, o_off, o_row, o_seg; float scale;
                 int B, ns, bt_off, nblk, anchor, flags, st_off; } VkcKvsMla;
int vkc_kvs_mla(VkcBuf *qa, VkcBuf *qr, VkcBuf *lat, VkcBuf *rope, VkcBuf *sel, VkcBuf *out, VkcBuf *tab, const VkcKvsMla *p);
typedef struct { int n, H, d, a_off, a_row, a_seg, sa_off, c_off, sc_off, o_off, o_row, o_seg, flags, g_off, g_row, g_seg; } VkcKvsMerge;
#define VKC_KVS_BF16 4   /* vkc_kvs_merge: the result rounded to bf16 (DeepSeek V4) */
int vkc_kvs_merge(VkcBuf *dev, VkcBuf *cpu, VkcBuf *gate, VkcBuf *out, const VkcKvsMerge *p);
/* Inkling's attention over the device's rows of a global layer (chain_relattn's
 * arithmetic, causal): score = tau[s] * (q . k * scale + bias(pos - t)), the bias mixed
 * from r (r_off + s*r_row + h*d_rel) and relp (relp_off, [d_rel][ext]) for distances
 * below ext, tau[s] at tau_off + s; K/V head-major (kvh*rows + r)*hd; outputs as
 * vkc_kvs_attn's (VKC_KVS_FIN: normalized at o_off + s*o_row + h*hd). hd <= 256,
 * d_rel <= 64. */
typedef struct { int S, H, KVH, hd, pos_base, rows, q_off, q_row; float scale; int k_off, v_off, ext, d_rel, r_off, r_row,
                 relp_off, tau_off, B, bt_off, nblk, anchor, o_off, o_row, st_off, flags; } VkcKvsRel;
int vkc_kvs_rel(VkcBuf *q, VkcBuf *kc, VkcBuf *vc, VkcBuf *out, VkcBuf *r, VkcBuf *tau, VkcBuf *relp, VkcBuf *tab,
                const VkcKvsRel *p);
/* DeepSeek's sparse attention with a sink over the device's rows (vkc_dsv4_attn's
 * arithmetic): row s's list (l_off + s*l_row, cnt entries) of window rows (e < nwin,
 * win at w_off + e*hd) and compressed rows c = e - nwin (in a slot: cmp at c_off +
 * (slot*B + c % B)*hd; a compressed row in the host's RAM only is skipped), the sink at
 * sink_off + h of prm. VKC_KVS_DSFIN: every listed row is on the device, out = the
 * attention at o_off + s*o_row + h*hd with vkc_dsv4_attn's bits; else the part (its max
 * over the sink too). VKC_KVS_V4: the weights and the output rounded to bf16 as
 * DeepSeek V4. hd <= 1024. */
#define VKC_KVS_V4    1
#define VKC_KVS_DSFIN 8
typedef struct { int S, H, hd, cnt, l_off, l_row, nwin, w_off, c_off, q_off, q_row, sink_off, flags; float scale;
                 int B, ns, bt_off, nblk, o_off, o_row, st_off; } VkcKvsDs;
int vkc_kvs_ds(VkcBuf *q, VkcBuf *win, VkcBuf *cmp, VkcBuf *out, VkcBuf *prm, VkcBuf *list, VkcBuf *tab, const VkcKvsDs *p);
/* Complete sparse attention with selected cold rows staged from canonical RAM.
 * List entries -2-j address cold row j; -1 still means skipped. This keeps the
 * original list order and, for V4, one maximum before rounding weights to bf16.
 * V4 requires DSFIN: independently rounded partials cannot be merged equivalently. */
#define VKC_KVS_DSCOLD 16
int vkc_kvs_ds_cold(VkcBuf *q, VkcBuf *win, VkcBuf *cmp, VkcBuf *cold, VkcBuf *out,
                    VkcBuf *prm, VkcBuf *list, VkcBuf *tab, const VkcKvsDs *p);
/* Frames by serial: vkc_serial() is the serial of the frame vkc_submit sent last;
 * vkc_wait_serial(s) waits until that frame has completed while later ones run on
 * (1; 0 on a lost device). The split's host part starts once the frame holding the
 * queries is through, while the device runs its own part in the next frame. */
unsigned long long vkc_serial(void);
int vkc_wait_serial(unsigned long long serial);
/* ---- the decision engines' encoders (chain_enc.comp) ---------------------------------
 * Laya's ModernBERT and decision head (laya.c) and GLiNER2.5-Decide's DeBERTa-v3
 * (gliner_decide.c) as one recorded forward: no cache, no state, every row of a request
 * at once, so the attention reads the step's own q, k and v and a row sees a range of
 * rows (its sequence, a window inside it) instead of a causal prefix. Made on first use
 * like inkling's ops: vkc_enc_ready() says whether the shader is there. Offsets and
 * strides in floats; rng, ridx, pos and idx are int32 buffers. The shader's header has
 * each op's arithmetic.
 *   vkc_enc_norm   LayerNorm of a (+ b with VKC_ENC_ADD; the sum written back to a with
 *                  VKC_ENC_SUM) by w at w_off and the bias at bias_off (-1 none) in prm,
 *                  into y; D <= 4096
 *   vkc_enc_bias   y = act(y + bias) in place over n = rows*O (b_off -1: no bias)
 *   vkc_enc_geglu  y = act(u + bu) * (g + bg), u and g the halves of a 2I-wide row
 *   vkc_enc_rope   rotate-half RoPE in place on the query and key heads (part_off apart),
 *                  cos and sin from a table at row pos[r]
 *   vkc_enc_addrow y += a table row picked per row by idx[r]
 *   vkc_enc_attn   attention over each row's range rng[2r]..rng[2r+1] (inclusive), with
 *                  DeBERTa's two relative terms (VKC_ENC_C2P, VKC_ENC_P2C: r = ridx[pos[i]
 *                  - pos[j] + rc] - r0, c2p[h][i][r] at c_off in c2p, p2c[h][j][r] at p_off
 *                  in p2c, nr buckets a row); hd <= 128
 *   vkc_enc_rel    those products: y[y_off + (h*S + i)*nr + t] = x_i,h . p[r0 + t],h for
 *                  every head, row and bucket t < nr (x at x_off + i*x_row + h*hd, p at
 *                  p_off + r*p_row + h*hd: the relative embeddings through the layer's key
 *                  or query projection) */
int vkc_enc_ready(void);
#define VKC_ENC_ADD 1
#define VKC_ENC_SUM 2
#define VKC_ENC_ACT_NONE      0
#define VKC_ENC_ACT_GELU      1   /* erf */
#define VKC_ENC_ACT_GELU_TANH 2
#define VKC_ENC_ACT_RELU      3
#define VKC_ENC_C2P 1
#define VKC_ENC_P2C 2
typedef struct { int mode, rows, D, a_off, a_row, b_off, b_row, y_off, y_row, w_off, bias_off, flags; float eps; } VkcEncNorm;
int vkc_enc_norm(VkcBuf *a, VkcBuf *b, VkcBuf *prm, VkcBuf *y, const VkcEncNorm *p);
typedef struct { int mode, n, O, y_off, y_row, b_off, act; } VkcEncBias;
int vkc_enc_bias(VkcBuf *y, VkcBuf *prm, const VkcEncBias *p);
typedef struct { int mode, n, I, a_off, a_row, y_off, y_row, b_off, act; } VkcEncGeglu;
int vkc_enc_geglu(VkcBuf *a, VkcBuf *prm, VkcBuf *y, const VkcEncGeglu *p);
typedef struct { int mode, n, H, hd, x_off, x_row, part_off, cos_off, sin_off; } VkcEncRope;
int vkc_enc_rope(VkcBuf *x, VkcBuf *tab, VkcBuf *pos, const VkcEncRope *p);
typedef struct { int mode, n, D, y_off, y_row, t_off; } VkcEncAddRow;
int vkc_enc_addrow(VkcBuf *y, VkcBuf *tab, VkcBuf *idx, const VkcEncAddRow *p);
typedef struct { int mode, S, H, hd, q_off, k_off, v_off, kv_row, o_off, o_row, flags, c_off, p_off, nr, rc;
                 float scale; int r0; } VkcEncAttn;
int vkc_enc_attn(VkcBuf *qkv, VkcBuf *c2p, VkcBuf *p2c, VkcBuf *o, VkcBuf *rng, VkcBuf *ridx, VkcBuf *pos,
                 const VkcEncAttn *p);
typedef struct { int mode, S, H, hd, x_off, x_row, p_off, p_row, y_off, nr, r0; } VkcEncRel;
int vkc_enc_rel(VkcBuf *x, VkcBuf *pt, VkcBuf *y, const VkcEncRel *p);

/* chain_attn_full.comp: a diffusion step's attention (Qwen-Image's DiT). S query rows, each
 * over all T key rows, H heads of hd (32, 64 or 128) floats, q, k and v token-major in one
 * buffer: q[q_off + i*q_row + h*hd + d], k and v at k_off and v_off + t*kv_row + h*hd + d,
 * the output at o_off + i*o_row + h*hd + d; the softmax's scale. 64 query rows of a head a
 * workgroup, the keys in tiles of 64, the softmax online in float. Made on first use. */
typedef struct { int S, T, H, hd, q_off, q_row, k_off, v_off, kv_row, o_off, o_row; float scale; } VkcAttnFull;
int vkc_attn_full(VkcBuf *qkv, VkcBuf *o, const VkcAttnFull *p);
int vkc_attn_full_ready(void);
/* The same on the matrix units (chain_attn_coop.comp, cooperative matrices at subgroups of
 * 32 or 64, hd 64 or 128): Q, K, V and P rounded to f16, the result the f32 one's to about
 * 1e-3. 0: not on this device (nothing recorded). */
int vkc_attn_full_coop(VkcBuf *qkv, VkcBuf *o, const VkcAttnFull *p);
int vkc_attn_full_coop_ready(int hd);

/* chain_vae.comp: Qwen-Image's VAE decoder (qwenimage_vae_vk.h), the two steps of its
 * convolutions that are not a GEMM; maps channel-last [H][W][C]. mode 0: the 3x3 taps of
 * output rows y0.. (n = pixels x 9C floats at o_off, [kx][ky][c] a pixel, zero outside the
 * Ho x Wo map; up 1: x is H x W, nearest-upsampled 2x); mode 1: o[o_off + p*Co + c] +=
 * the DupUp3D shortcut's input channel tab[c*4 + sub-pixel] of the half-size map x (dupC
 * channels), n = pixels x Co. Made on first use. */
typedef struct { int mode, C, H, W, up, Ho, Wo, y0, n, x_off, o_off, Co, dupC; } VkcVae;
int vkc_vae(VkcBuf *x, VkcBuf *o, VkcBuf *tab, const VkcVae *p);
int vkc_vae_ready(void);

/* counters, for the engines' [VK] lines */
typedef struct {
    unsigned long long frames, waits, ops, matmuls, gemms, barriers, bytes_up, bytes_down;
    double wait_ms;               /* host time blocked in fence waits */
    size_t dev_bytes;             /* live chain buffers */
    unsigned long long attn_blocked;   /* attention calls through chain_attnb (blocks of rows) */
    unsigned long long attn_flash;     /* attention calls through chain_attn_flash (matrix units) */
    unsigned long long attn_slices;    /* extra submissions that cut a long attention (COLI_VK_ATTN_SLICE) */
    unsigned long long tile_gemms;     /* prompt matmuls through chain_gemm.comp (matrix units) */
} VkcStats;
void vkc_stats(VkcStats *st);
/* COLI_VK_CHAIN_PROF=1: one stderr line of device time per kind of op */
void vkc_prof_print(void);

#ifdef __cplusplus
}
#endif
#endif
