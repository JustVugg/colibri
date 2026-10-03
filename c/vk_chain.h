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

int  vkc_init(void);          /* after coli_vk_init; 1 = the chain's pipelines are up */
int  vkc_ready(void);
int  vkc_lost(void);
void vkc_shutdown(void);      /* before coli_vk_shutdown (register it with atexit after it) */

VkcBuf *vkc_buf(size_t bytes, int kind);                 /* zero-filled; NULL when out of memory */
void    vkc_free(VkcBuf *b);                             /* waits for the frames that may read it */
int     vkc_reserve(VkcBuf **b, size_t bytes, int kind); /* at least `bytes`; growing drops the contents */
void   *vkc_ptr(const VkcBuf *b);                        /* host mapping (VKC_UP, VKC_DOWN; VKC_DEV when host-visible) */
size_t  vkc_bytes(const VkcBuf *b);

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

/* counters, for the engines' [VK] lines */
typedef struct {
    unsigned long long frames, waits, ops, matmuls, gemms, barriers, bytes_up, bytes_down;
    double wait_ms;               /* host time blocked in fence waits */
    size_t dev_bytes;             /* live chain buffers */
} VkcStats;
void vkc_stats(VkcStats *st);
/* COLI_VK_CHAIN_PROF=1: one stderr line of device time per kind of op */
void vkc_prof_print(void);

#ifdef __cplusplus
}
#endif
#endif
