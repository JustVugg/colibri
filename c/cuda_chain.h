/* cuda_chain.h -- a layer's dense chain on a CUDA device, issued on one stream.
 *
 * The CUDA twin of vk_chain.h. With the dense trunk in VRAM one coli_cuda_matmul at a
 * time, every matrix is a host round trip (the input up, the kernel, the output down,
 * a synchronization): on Qwen3.6-35B about seventy per decode token, the attention
 * core, the norms, the router and the shared expert still on the CPU between them.
 * Here an engine issues a whole layer -- norms, projections, RoPE, attention over a
 * KV cache that lives on the device, the Gated DeltaNet recurrence with its state on
 * the device, gates, the router logits, the shared expert, the residual add -- on one
 * stream, and the residual stream stays on the device from one layer to the next.
 * Only what the CPU needs crosses: the rows the CPU's routed experts read and the
 * router logits, then the routed sum coming back.
 *
 * The pieces, shaped as vk_chain.h's so an engine wires both chains the same way:
 *   - buffers (CcBuf): CC_DEV the device's own (state, scratch); CC_UP written by the
 *     host and read by the device; CC_DOWN written by the device and read by the host.
 *     UP and DOWN are page-locked host memory mapped into the device's address space:
 *     the host writes or reads them in place (cc_ptr), the device reads or writes them
 *     over the bus, as the Vulkan chain's host-visible buffers.
 *   - recording: cc_begin opens a frame; every op goes to the device's stream in order
 *     (a stream is the barrier Vulkan records by hand); cc_submit(wait) ends it with an
 *     event, waited for or not. Frames run in order on the one stream.
 *   - ops: cc_matmul over the resident tensors the engines already upload
 *     (coli_cuda_tensor_upload: the backend's GEMV / GEMM kernel on the chain's
 *     stream), and the chain's kernels, each the arithmetic of the shader of the same
 *     name in shaders/chain_*.comp: cc_norm, cc_rope, cc_attn, cc_dnconv, cc_dnrec,
 *     cc_ew. Offsets and strides are in floats; the parameter structs are field for
 *     field the Vkc ones.
 *
 * The backend exports the ops as one table (coli_cuda_chain_ops): one symbol for the
 * Windows loader, versioned by its size, so a DLL that predates the chain simply has
 * none and the engine stays on the per-matrix path. The cc_* calls below go through
 * the table.
 *
 * Threading: the engine thread only. The expert tier's groups run on the backend's own
 * stream of the device and overlap the chain's frames. A CUDA error inside a frame
 * marks the device lost (cc_lost() says so, every later call returns 0): the engine
 * then rebuilds its state on the CPU, as with the Vulkan chain.
 *
 * tests/test_cuda_chain.cu checks every op against a CPU reference on a real device;
 * tests/qwen36_fake_cuda.h carries a host-side table of the same ops for the engine
 * tests. */
#ifndef COLI_CUDA_CHAIN_H
#define COLI_CUDA_CHAIN_H
#include <stddef.h>
#include <string.h>
#include "backend_cuda.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CcBuf CcBuf;
#define CC_DEV  0
#define CC_UP   1
#define CC_DOWN 2

/* chain_norm.comp */
typedef struct { int nseg, D, per_row, x_off, x_row, x_seg, y_off, y_row, y_seg, w_off, w_mod, flags; float eps, post; } CcNorm;
#define CC_NORM_ADD1 1
#define CC_NORM_NOW  2
#define CC_NORM_L2   4
#define CC_NORM_PRESCALE 8   /* scale by w, then normalise (Muse Spark; cc_muse_ready) */
/* chain_rope.comp */
typedef struct { int nseg, per_row, x_off, x_row, x_seg, half_, cs_off, cs_row; } CcRope;
/* chain_attn.comp (k_off/v_off: where the layer's cache starts in kc/vc; the cache
 * head-major, K[(kvh*cap + t)*hd + d]; sel_row > 0: a selection list per row) */
typedef struct { int S, H, KVH, hd, pos_base, cap, q_off, q_row, q_seg, g_off, g_row, g_seg, has_gate,
                 o_off, o_row, sel_off, sel_row; float scale; int k_off, v_off; } CcAttn;
/* The same with Flint's two (vk_chain.h VkcAttnW's win and sink): a sliding window of win
 * positions (row s sees max(0, pos - win + 1)..pos; 0 = all), and with sink a logit per head
 * at snk[sink_off + h] that joins the softmax's denominator only. No selection list. */
typedef struct { CcAttn a; int win, sink, sink_off; } CcAttnW;
/* chain_dnconv.comp */
typedef struct { int S, CD, CK, in_off, in_row, out_off, out_row, snap_row, order, w_off, ring_off, snap_off; } CcDnConv;
/* chain_dnrec.comp (KD a template parameter: 4, 8, 16, 32, 64, 128 or 256; VD <= 128). flags: 1 the gated
 * norm's gate a sigmoid (Qwen3.8), 2 the parameter A itself (a GGUF's, -exp(A_log)), 4 the value
 * heads tiled (a GGUF's: value head h reads key head h % KH) */
typedef struct { int S, VH, KH, VD, Ktot, cv_off, cv_row, b_off, b_row, a_off, a_row, z_off, z_row,
                 y_off, y_row, snap_row, flags; float eps, qscale; int st_off, snap_off, prm_off; } CcDnRec;
/* chain_ew.comp */
#define CC_EW_ADD      0
#define CC_EW_COMBINE  1
#define CC_EW_SWIGLU   2
#define CC_EW_HC_LOW   3
#define CC_EW_HC_MIX   4
#define CC_EW_HC_INJ   5
#define CC_EW_HC_APPLY 6
#define CC_EW_SCALE    7
#define CC_EW_GATE_ADD 8
#define CC_EW_SIGMUL   9    /* y = a * sigmoid(b) (cc_muse_ready) */
#define CC_EW_RESGATE  10   /* y = e[col] * a + c[col] * b (cc_muse_ready) */
#define CC_EW_F16      11   /* y = a clamped to +-65504, rounded to the nearest float16, as kv16.h (cc_kv16_ready) */
typedef struct { int op, n, D, C, flags, e_row, y_off, a_off, b_off, c_off, e_off; float fc; } CcEw;
/* chain_qsa.comp (mode 0: nb block keys from b0; mode 1: S rows' selections) */
typedef struct { int mode, ID, R, b0, half_, S, pos_base, budget, IQ, q_off, q_row, nbmax, sel_row; float eps;
                 int src_off, w_off, pk_off, nb; } CcQsa;
/* chain_ple.comp (mode 0: the gate over S*C (row, stream) pairs; mode 1: the convolution) */
typedef struct { int mode, S, C, H, CK, NG, keys_off, hyp_off, val_off, snap_row, snap_off; float eps;
                 int prm_off, conv_off, ring_off; } CcPle;
/* chain_route.comp (vk_chain.h's VkcRoute, field for field): every MoE engine's routing
 * for S rows of E logits (l_off + s*l_row): kind 0 softmax, 1 sigmoid, 2 sqrt(softplus)
 * of logit / temp; the bias (b_off) added for the pick only with has_bias 1, to the
 * logit with 2; with n_group > 1 the topk_group best groups' experts only; the K best
 * picks, the lower expert on a tie, in rank order into ids (i_off + s*i_row, -1 for a
 * slot no finite pick fills), their scores into w (w_off + s*w_row), divided by their
 * sum plus norm_eps with norm 1 (when that is above 0) or norm 2 (from K > 1 only),
 * times scale. ids holds ints. */
#define CC_ROUTE_EMAX 1024
#define CC_ROUTE_KMAX 32
typedef struct {
    int S, E, K, kind;
    int l_off, l_row, b_off, has_bias;
    int i_off, i_row, w_off, w_row;
    int n_group, topk_group, norm;
    float temp, scale, norm_eps;
} CcRoute;
/* A routed expert as a device holds it (the CUDA tier's resident tensors: their weights
 * and scales, coli_cuda_tensor_dev); g NULL: not on the device (the host computes it).
 * planar 1: int4 per 64 as the engine keeps it in RAM (expert_ffn.h's planar blocks: in each
 * 64 weights' 32 bytes, byte k holds weight k in its low nibble and k + 32 in its high one,
 * both v + 8), as a prompt chunk stages it from the tier's RAM cache (cc_fetch; its GEMMs
 * only); 0: the backend's own layout (signed nibbles, weight 2j low and 2j + 1 high).
 * host 1: the matrices are in page-locked host memory the device reads over the bus (an
 * engine's host pool, a GGUF's experts: g, u and d in one span from g); a decode step
 * copies its such experts into the staging cc_moe_stage gave first, where there is one. */
typedef struct { const void *g, *u, *d; const float *gs, *us, *ds; int planar, host; } CcExpert;
#define CC_MOE_STAGE_MAP 256   /* the staging's map of a step's slots to its places, bytes before them */
/* A layer's routed experts on the device for S rows (cc_moe): x (x_off + s*x_row, D
 * floats) routed to ids and w as cc_route writes them (i_off + s*i_row, w_off + s*w_row,
 * K each); expert e is entry t_off + e of the table (E of them), its matrices of format fmt (1 int8
 * rows, 2 int4 rows, 4 int4 with a scale per gs inputs, 8 fp8-e4m3 with a scale per
 * 128 x 128 block: the tier's formats; 100 + t a GGUF's ggml type t, fmt_d down's, the types
 * with cuda_bd.cuh's integer step), D and I multiples of 32 up to CC_MOE_DIM_MAX. hid holds S*K rows of
 * I floats (silu(gate) * up, act 0, the only one) and then S*K rows of D (down's
 * output). parts 1: the experts the table holds; parts 2: y (y_off + s*y_row) = the
 * host's row (hy at h_off + s*D, h_off >= 0; else 0) plus each slot the table holds,
 * w * down's output, in rank order (the CPU tier's order: its misses, then the device's
 * results); 3 both. A slot whose id is -1 or whose entry is not on the device adds
 * nothing. */
#define CC_MOE_DIM_MAX 12032   /* D and I: an input row and the fp8 table in 48 KiB of shared memory */
typedef struct {
    int S, K, D, I, fmt, gs, act, parts;
    int t_off, x_off, x_row, i_off, i_row, w_off, w_row, y_off, y_row, h_off;
    int E;   /* the layer's experts: a step of more than CC_MOE_FRAME_ROWS rows sorts its slots by expert */
    int planar;   /* the table may hold CcExpert.planar entries: refused where the GEMMs cannot read them */
    int fmt_d;    /* fmt >= 100 (a GGUF's blocks, ggml type fmt - 100 for gate and up): down's,
                     0 = fmt's; a matrix's rows as the file has them, padded to words */
} CcMoe;
/* the ggml types cc_moe takes for a GGUF's experts (CcMoe.fmt >= 100: cuda_bd.cuh's integer step) */
static inline int cc_moe_block_type(int t) {
    return t == 8 || t == 10 || t == 11 || t == 12 || t == 13 || t == 14 || t == 16 || t == 17 || t == 18 || t == 19 || t == 20 ||
           t == 21 || t == 22 || t == 23 || t == 29 || t == 42;
}
/* More rows than this (a prompt chunk): the slots sorted by expert and each expert's rows
 * through the tensor cores' GEMM where the device has it (gate and up, then down); hid
 * then also holds the sort's integers, cc_moe_hid_floats in all. */
#define CC_MOE_FRAME_ROWS 8
static inline size_t cc_moe_hid_floats(int S, int K, int D, int I, int E) {
    size_t A = (size_t)S * K, n = A * (size_t)(I + D);
    if (S > CC_MOE_FRAME_ROWS) n += 3 * (size_t)E + 2 + 2 * ((A + 63) / 64 + (size_t)(E < (int)A ? E : (int)A)) + 2 * A;
    return n;
}
/* one copy over n regions (a KV row per head, say); offsets and counts in floats */
typedef struct { size_t dst, src, n; } CcRegion;
/* a fetch's region: bytes of page-locked host memory at src into a DEV buffer at byte dst */
typedef struct { const void *src; size_t dst, bytes; } CcFetch;
/* chain_enc.comp: the decision engines' encoders (decide_cuda.h), every row of a request at
 * once with no cache and no state. Field for field vk_chain.h's VkcEnc* (its comment has each
 * op's arithmetic): enc_norm a LayerNorm (+ b with CC_ENC_ADD, the sum written back to a with
 * CC_ENC_SUM; D <= 4096), enc_bias y = act(y + bias) in place, enc_geglu act(u + bu) * (g + bg),
 * enc_rope rotate-half RoPE on the query and key heads from a table row pos[r], enc_addrow a
 * table row per row, enc_attn each row over its range of rows with DeBERTa's two relative
 * terms (hd <= 128), enc_rel those terms' products. rng, ridx, pos and idx: int32 buffers. */
#define CC_ENC_ADD 1
#define CC_ENC_SUM 2
#define CC_ENC_ACT_NONE      0
#define CC_ENC_ACT_GELU      1   /* erf */
#define CC_ENC_ACT_GELU_TANH 2
#define CC_ENC_ACT_RELU      3
#define CC_ENC_C2P 1
#define CC_ENC_P2C 2
typedef struct { int mode, rows, D, a_off, a_row, b_off, b_row, y_off, y_row, w_off, bias_off, flags; float eps; } CcEncNorm;
typedef struct { int mode, n, O, y_off, y_row, b_off, act; } CcEncBias;
typedef struct { int mode, n, I, a_off, a_row, y_off, y_row, b_off, act; } CcEncGeglu;
typedef struct { int mode, n, H, hd, x_off, x_row, part_off, cos_off, sin_off; } CcEncRope;
typedef struct { int mode, n, D, y_off, y_row, t_off; } CcEncAddRow;
typedef struct { int mode, S, H, hd, q_off, k_off, v_off, kv_row, o_off, o_row, flags, c_off, p_off, nr, rc;
                 float scale; int r0; } CcEncAttn;
typedef struct { int mode, S, H, hd, x_off, x_row, p_off, p_row, y_off, nr, r0; } CcEncRel;
/* counters, for the engines' [chain] lines */
typedef struct {
    unsigned long long frames, waits, ops, matmuls, bytes_up, bytes_down;
    double wait_ms;      /* host time blocked in event waits */
    size_t dev_bytes;    /* live chain buffers on the device */
} CcStats;

/* The table the backend exports. `size` is the backend's sizeof: a field past it is
 * not there (an older DLL), and the engine must not call it. */
typedef struct ColiCudaChainOps {
    size_t size;
    /* after coli_cuda_init, for a configured device; 1 = the chain is up on it. Each
     * device has a context of its own; cc_device makes one current (returns the
     * previous), cc_device_now says which. shutdown closes every context. */
    int  (*init)(int device);
    int  (*ready)(void);
    int  (*lost)(void);
    void (*shutdown)(void);
    int  (*device)(int d);
    int  (*device_now)(void);
    CcBuf *(*buf)(size_t bytes, int kind);                 /* zero-filled; NULL when out of memory */
    void   (*free)(CcBuf *b);                              /* waits for the frames that may read it */
    int    (*reserve)(CcBuf **b, size_t bytes, int kind);  /* at least `bytes`; growing drops the contents */
    void  *(*ptr)(const CcBuf *b);                         /* host mapping (CC_UP, CC_DOWN) */
    size_t (*bytes)(const CcBuf *b);
    int  (*begin)(void);
    int  (*submit)(int wait);
    int  (*finish)(void);                                  /* wait for every submitted frame */
    int  (*copy)(CcBuf *dst, size_t doff, CcBuf *src, size_t soff, size_t n);
    int  (*zero)(CcBuf *dst, size_t off, size_t n);
    int  (*copy_regions)(CcBuf *dst, CcBuf *src, const CcRegion *r, int n);
    int  (*write)(CcBuf *dst, size_t off, const void *src, size_t bytes);   /* into the open frame */
    int  (*read)(CcBuf *src, size_t off, void *dst, size_t bytes);          /* synchronous */
    /* y[S][O] = x[S][I] @ W^T for a resident tensor (its fmt, I, O); x and y at float offsets */
    int  (*matmul)(ColiCudaTensor *t, CcBuf *x, size_t xo, CcBuf *y, size_t yo, int S);
    int  (*norm)(CcBuf *x, CcBuf *w, CcBuf *y, const CcNorm *p);
    int  (*rope)(CcBuf *x, CcBuf *cs, const CcRope *p);
    int  (*attn)(CcBuf *q, CcBuf *kc, CcBuf *vc, CcBuf *o, CcBuf *gate, CcBuf *sel, const CcAttn *p);
    int  (*dnconv)(CcBuf *in, CcBuf *w, CcBuf *ring, CcBuf *out, CcBuf *snap, const CcDnConv *p);
    int  (*dnrec)(int KD, CcBuf *cv, CcBuf *ab, CcBuf *z, CcBuf *st, CcBuf *prm, CcBuf *y, CcBuf *snap, const CcDnRec *p);
    int  (*ew)(CcBuf *y, CcBuf *a, CcBuf *b, CcBuf *c, CcBuf *e, const CcEw *p);
    void (*stats)(CcStats *st);
    /* Qwen3.8's two: the QSA indexer's block keys and selections, the PLE gate and convolution */
    int  (*qsa)(CcBuf *src, CcBuf *w, CcBuf *pk, CcBuf *cs, CcBuf *sc, CcBuf *sel, const CcQsa *p);
    int  (*ple)(CcBuf *keys, CcBuf *hyp, CcBuf *val, CcBuf *prm, CcBuf *gated, CcBuf *normv, CcBuf *conv, CcBuf *ring, const CcPle *p);
    /* The routed experts in the frame (optional: a table that ends before them has none).
     * tab: CcExpert entries (written with write, offsets in floats as ever: an entry is
     * sizeof(CcExpert) / 4 of them). signal: once the frame's earlier ops are done, word
     * `word` of a DOWN buffer holds v (the host polls it); wait_host: the frame's later
     * ops wait until word `word` of an UP buffer is at least v (the host writes it). A
     * hybrid step: route, signal, the device's experts, wait_host, the sum with the
     * host's rows; the host computes its experts between the two. */
    int  (*route)(CcBuf *logits, CcBuf *bias, CcBuf *ids, CcBuf *w, const CcRoute *p);
    int  (*moe)(CcBuf *tab, CcBuf *x, CcBuf *ids, CcBuf *w, CcBuf *hid, CcBuf *y, CcBuf *hy, const CcMoe *p);
    int  (*signal)(CcBuf *flag, size_t word, unsigned v);
    int  (*wait_host)(CcBuf *flag, size_t word, unsigned v);
    /* a resident tensor's device copy, for a CcExpert: its weights, scales and the scales'
     * group size (fmt 4); 0 for none */
    int  (*tensor_dev)(const ColiCudaTensor *t, const void **w, const float **sc, int *gs);
    /* Staging (optional): a DEV buffer filled from page-locked host memory on the device's
     * copy stream, beside the frame's ops (a prompt chunk's experts from the tier's RAM
     * cache). fetch: n regions into dst, once the frame's ops recorded before the last
     * fetch_done are done (the buffer's last reader); fetch_wait: the frame's later ops wait
     * for the last fetch's copies; fetch_done: the buffer is free again once the frame's
     * earlier ops are done. dev_addr: a buffer's device address, for a CcExpert (an UP or DOWN
     * buffer's: where the device reads the host's pages over the bus; NULL from a backend
     * before the host pool). */
    int  (*fetch)(CcBuf *dst, const CcFetch *r, int n);
    int  (*fetch_wait)(void);
    int  (*fetch_done)(void);
    const void *(*dev_addr)(const CcBuf *b);
    /* moe_stage: a DEV buffer the context's cc_moe copies a decode step's host experts
     * (CcExpert.host) into before its kernels read them, slots of slot_bytes after a
     * 256-byte map; NULL: none (they read the host's memory where it is). A GGUF's experts
     * (fmt >= 100) only. */
    int  (*moe_stage)(CcBuf *b, size_t slot_bytes, int slots);
    /* host_up: bytes of page-locked host memory at src to the device address dst on the copy
     * stream, once the frame's ops recorded so far are done; *ticket marks it. host_up_done:
     * 1 once the copies up to ticket landed (an engine's experts moved into its slots) */
    int  (*host_up)(void *dst, const void *src, size_t bytes, unsigned *ticket);
    int  (*host_up_done)(unsigned ticket);
    /* attn_w: cc_attn with a sliding window and sinks (CcAttnW) */
    int  (*attn_w)(CcBuf *q, CcBuf *kc, CcBuf *vc, CcBuf *o, CcBuf *gate, CcBuf *snk, const CcAttnW *p);
    /* muse: 1 when the norm takes CC_NORM_PRESCALE and ew CC_EW_SIGMUL and CC_EW_RESGATE (an
     * older backend's kernels would skip them without a word) */
    int  (*muse)(void);
    /* attn_wr: attn_w over a cache that is a ring of `ring` rows (position t in row t % ring,
     * p->a.cap == ring): a sliding layer's (qwen36 Flint, Muse Spark), whose rows must hold
     * the step's rows and their windows (ring >= win + S - 1); 0: attn_w */
    int  (*attn_wr)(CcBuf *q, CcBuf *kc, CcBuf *vc, CcBuf *o, CcBuf *gate, CcBuf *snk, const CcAttnW *p, int ring);
    /* kv16: 1 when ew takes CC_EW_F16 (an older backend's kernel would skip it without a word) */
    int  (*kv16)(void);
    /* rope_f16: rope on x, then each of a segment's seg values of x, and of v at x's offsets,
     * rounded to float16 as ew's CC_EW_F16 rounds them: a float16 KV cache's new K and V rows
     * in one launch, the bits of rope and two ew (qwen36_cuda_chain.h); absent: those three */
    int  (*rope_f16)(CcBuf *x, CcBuf *v, CcBuf *cs, const CcRope *p, int seg);
    /* pick: the argmax (lowest index on a tie) of n logits in `logits` into word 0 of `out`
     * (an int) and its softmax probability (terms within 20 of the maximum) into word 1 (a
     * float): an MTP head's draft token and confidence without copying the vocabulary down */
    int  (*pick)(CcBuf *logits, CcBuf *out, int n);
    /* matmul_rows: y[0..n) = the rows ids[0..n) of a resident tensor (fmt 0 or 1, one input
     * row x[I]); an MTP head's adaptive draft vocabulary scored from the model's lm_head */
    int  (*matmul_rows)(ColiCudaTensor *t, CcBuf *ids, CcBuf *x, CcBuf *y, int n);
    /* A float16 KV cache (half the memory and half the reads of a float32 one): kvh 1 when
     * the next three take one (0 on HIP); attn_h and attn_wh are attn and attn_wr over caches
     * of halves (kc, vc and their offsets in halves); copy_regions_h copies float32 regions of
     * src (offsets in floats) into such a cache (offsets in halves), each value rounded to the
     * nearest float16 */
    int  (*kvh)(void);
    int  (*attn_h)(CcBuf *q, CcBuf *kc, CcBuf *vc, CcBuf *o, CcBuf *gate, CcBuf *sel, const CcAttn *p);
    int  (*attn_wh)(CcBuf *q, CcBuf *kc, CcBuf *vc, CcBuf *o, CcBuf *gate, CcBuf *snk, const CcAttnW *p, int ring);
    int  (*copy_regions_h)(CcBuf *dst, CcBuf *src, const CcRegion *r, int n);
    /* the decision engines' encoders (CcEnc* above) */
    int  (*enc_norm)(CcBuf *a, CcBuf *b, CcBuf *prm, CcBuf *y, const CcEncNorm *p);
    int  (*enc_bias)(CcBuf *y, CcBuf *prm, const CcEncBias *p);
    int  (*enc_geglu)(CcBuf *a, CcBuf *prm, CcBuf *y, const CcEncGeglu *p);
    int  (*enc_rope)(CcBuf *x, CcBuf *tab, CcBuf *pos, const CcEncRope *p);
    int  (*enc_addrow)(CcBuf *y, CcBuf *tab, CcBuf *idx, const CcEncAddRow *p);
    int  (*enc_attn)(CcBuf *qkv, CcBuf *c2p, CcBuf *p2c, CcBuf *o, CcBuf *rng, CcBuf *ridx, CcBuf *pos, const CcEncAttn *p);
    int  (*enc_rel)(CcBuf *x, CcBuf *pt, CcBuf *y, const CcEncRel *p);
} ColiCudaChainOps;
/* the table's size without the optional ops: a backend at least this big has the chain */
#define CC_OPS_BASE_SIZE (offsetof(ColiCudaChainOps, route))

/* The backend's table; NULL from a backend without the chain. Optional in the DLL
 * loader, like every entry point added after the first release. */
COLI_CUDA_DLLEXPORT const ColiCudaChainOps *coli_cuda_chain_ops(void);

/* ---- the engine's calls, through the table ----------------------------------- */
#ifndef COLI_CUDA_CHAIN_NO_WRAPPERS
static inline const ColiCudaChainOps *cc_ops(void) {
    static const ColiCudaChainOps *t; static int asked;
    if (!asked) { asked = 1; t = coli_cuda_chain_ops(); if (t && t->size < CC_OPS_BASE_SIZE) t = NULL; }
    return t;
}
static inline int cc_available(void) { return cc_ops() != NULL; }
static inline int  cc_init(int device) { const ColiCudaChainOps *t = cc_ops(); return t ? t->init(device) : 0; }
static inline int  cc_ready(void) { const ColiCudaChainOps *t = cc_ops(); return t ? t->ready() : 0; }
static inline int  cc_lost(void) { const ColiCudaChainOps *t = cc_ops(); return t ? t->lost() : 1; }
static inline void cc_shutdown(void) { const ColiCudaChainOps *t = cc_ops(); if (t) t->shutdown(); }
static inline int  cc_device(int d) { const ColiCudaChainOps *t = cc_ops(); return t ? t->device(d) : -1; }
static inline int  cc_device_now(void) { const ColiCudaChainOps *t = cc_ops(); return t ? t->device_now() : -1; }
static inline CcBuf *cc_buf(size_t bytes, int kind) { const ColiCudaChainOps *t = cc_ops(); return t ? t->buf(bytes, kind) : NULL; }
static inline void   cc_free(CcBuf *b) { const ColiCudaChainOps *t = cc_ops(); if (t && b) t->free(b); }
static inline int    cc_reserve(CcBuf **b, size_t bytes, int kind) { const ColiCudaChainOps *t = cc_ops(); return t ? t->reserve(b, bytes, kind) : 0; }
static inline void  *cc_ptr(const CcBuf *b) { const ColiCudaChainOps *t = cc_ops(); return t && b ? t->ptr(b) : NULL; }
static inline size_t cc_bytes(const CcBuf *b) { const ColiCudaChainOps *t = cc_ops(); return t && b ? t->bytes(b) : 0; }
static inline int cc_begin(void) { const ColiCudaChainOps *t = cc_ops(); return t ? t->begin() : 0; }
static inline int cc_submit(int wait) { const ColiCudaChainOps *t = cc_ops(); return t ? t->submit(wait) : 0; }
static inline int cc_finish(void) { const ColiCudaChainOps *t = cc_ops(); return t ? t->finish() : 0; }
static inline int cc_copy(CcBuf *dst, size_t doff, CcBuf *src, size_t soff, size_t n) { const ColiCudaChainOps *t = cc_ops(); return t ? t->copy(dst, doff, src, soff, n) : 0; }
static inline int cc_zero(CcBuf *dst, size_t off, size_t n) { const ColiCudaChainOps *t = cc_ops(); return t ? t->zero(dst, off, n) : 0; }
static inline int cc_copy_regions(CcBuf *dst, CcBuf *src, const CcRegion *r, int n) { const ColiCudaChainOps *t = cc_ops(); return t ? t->copy_regions(dst, src, r, n) : 0; }
static inline int cc_write(CcBuf *dst, size_t off, const void *src, size_t bytes) { const ColiCudaChainOps *t = cc_ops(); return t ? t->write(dst, off, src, bytes) : 0; }
static inline int cc_read(CcBuf *src, size_t off, void *dst, size_t bytes) { const ColiCudaChainOps *t = cc_ops(); return t ? t->read(src, off, dst, bytes) : 0; }
static inline int cc_matmul(ColiCudaTensor *w, CcBuf *x, size_t xo, CcBuf *y, size_t yo, int S) { const ColiCudaChainOps *t = cc_ops(); return t ? t->matmul(w, x, xo, y, yo, S) : 0; }
static inline int cc_norm(CcBuf *x, CcBuf *w, CcBuf *y, const CcNorm *p) { const ColiCudaChainOps *t = cc_ops(); return t ? t->norm(x, w, y, p) : 0; }
static inline int cc_rope(CcBuf *x, CcBuf *cs, const CcRope *p) { const ColiCudaChainOps *t = cc_ops(); return t ? t->rope(x, cs, p) : 0; }
static inline int cc_attn(CcBuf *q, CcBuf *kc, CcBuf *vc, CcBuf *o, CcBuf *gate, CcBuf *sel, const CcAttn *p) { const ColiCudaChainOps *t = cc_ops(); return t ? t->attn(q, kc, vc, o, gate, sel, p) : 0; }
static inline int cc_dnconv(CcBuf *in, CcBuf *w, CcBuf *ring, CcBuf *out, CcBuf *snap, const CcDnConv *p) { const ColiCudaChainOps *t = cc_ops(); return t ? t->dnconv(in, w, ring, out, snap, p) : 0; }
static inline int cc_dnrec(int KD, CcBuf *cv, CcBuf *ab, CcBuf *z, CcBuf *st, CcBuf *prm, CcBuf *y, CcBuf *snap, const CcDnRec *p) { const ColiCudaChainOps *t = cc_ops(); return t ? t->dnrec(KD, cv, ab, z, st, prm, y, snap, p) : 0; }
static inline int cc_ew(CcBuf *y, CcBuf *a, CcBuf *b, CcBuf *c, CcBuf *e, const CcEw *p) { const ColiCudaChainOps *t = cc_ops(); return t ? t->ew(y, a, b, c, e, p) : 0; }
static inline void cc_stats(CcStats *st) { const ColiCudaChainOps *t = cc_ops(); if (t) t->stats(st); else memset(st, 0, sizeof *st); }
static inline int cc_qsa(CcBuf *src, CcBuf *w, CcBuf *pk, CcBuf *cs, CcBuf *sc, CcBuf *sel, const CcQsa *p) { const ColiCudaChainOps *t = cc_ops(); return t ? t->qsa(src, w, pk, cs, sc, sel, p) : 0; }
static inline int cc_ple(CcBuf *keys, CcBuf *hyp, CcBuf *val, CcBuf *prm, CcBuf *gated, CcBuf *normv, CcBuf *conv, CcBuf *ring, const CcPle *p) { const ColiCudaChainOps *t = cc_ops(); return t ? t->ple(keys, hyp, val, prm, gated, normv, conv, ring, p) : 0; }
/* the routed experts in the frame: a backend whose table has them */
static inline int cc_moe_ready(void) { const ColiCudaChainOps *t = cc_ops(); return t && t->size >= offsetof(ColiCudaChainOps, tensor_dev) + sizeof t->tensor_dev; }
static inline int cc_route(CcBuf *l, CcBuf *b, CcBuf *ids, CcBuf *w, const CcRoute *p) { return cc_moe_ready() ? cc_ops()->route(l, b, ids, w, p) : 0; }
static inline int cc_moe(CcBuf *tab, CcBuf *x, CcBuf *ids, CcBuf *w, CcBuf *hid, CcBuf *y, CcBuf *hy, const CcMoe *p) { return cc_moe_ready() ? cc_ops()->moe(tab, x, ids, w, hid, y, hy, p) : 0; }
static inline int cc_signal(CcBuf *flag, size_t word, unsigned v) { return cc_moe_ready() ? cc_ops()->signal(flag, word, v) : 0; }
static inline int cc_wait_host(CcBuf *flag, size_t word, unsigned v) { return cc_moe_ready() ? cc_ops()->wait_host(flag, word, v) : 0; }
static inline int cc_tensor_dev(const ColiCudaTensor *w, const void **p, const float **sc, int *gs) { return cc_moe_ready() ? cc_ops()->tensor_dev(w, p, sc, gs) : 0; }
static inline int cc_fetch_ready(void) { const ColiCudaChainOps *t = cc_ops(); return t && t->size >= offsetof(ColiCudaChainOps, dev_addr) + sizeof t->dev_addr; }
static inline int cc_fetch(CcBuf *dst, const CcFetch *r, int n) { return cc_fetch_ready() ? cc_ops()->fetch(dst, r, n) : 0; }
static inline int cc_fetch_wait(void) { return cc_fetch_ready() ? cc_ops()->fetch_wait() : 0; }
static inline int cc_fetch_done(void) { return cc_fetch_ready() ? cc_ops()->fetch_done() : 0; }
static inline const void *cc_dev_addr(const CcBuf *b) { return cc_fetch_ready() ? cc_ops()->dev_addr(b) : NULL; }
static inline int cc_host_up_ready(void) { const ColiCudaChainOps *t = cc_ops(); return t && t->size >= offsetof(ColiCudaChainOps, host_up_done) + sizeof t->host_up_done; }
static inline int cc_host_up(void *dst, const void *src, size_t bytes, unsigned *ticket) { return cc_host_up_ready() ? cc_ops()->host_up(dst, src, bytes, ticket) : 0; }
static inline int cc_host_up_done(unsigned ticket) { return cc_host_up_ready() ? cc_ops()->host_up_done(ticket) : 0; }
static inline int cc_moe_stage_ready(void) { const ColiCudaChainOps *t = cc_ops(); return t && t->size >= offsetof(ColiCudaChainOps, moe_stage) + sizeof t->moe_stage; }
static inline int cc_moe_stage(CcBuf *b, size_t slot_bytes, int slots) { return cc_moe_stage_ready() ? cc_ops()->moe_stage(b, slot_bytes, slots) : 0; }
static inline int cc_attn_w_ready(void) { const ColiCudaChainOps *t = cc_ops(); return t && t->size >= offsetof(ColiCudaChainOps, attn_w) + sizeof t->attn_w; }
static inline int cc_attn_w(CcBuf *q, CcBuf *kc, CcBuf *vc, CcBuf *o, CcBuf *gate, CcBuf *snk, const CcAttnW *p) {
    return cc_attn_w_ready() ? cc_ops()->attn_w(q, kc, vc, o, gate, snk, p) : 0;
}
static inline int cc_attn_wr_ready(void) { const ColiCudaChainOps *t = cc_ops(); return t && t->size >= offsetof(ColiCudaChainOps, attn_wr) + sizeof t->attn_wr; }
static inline int cc_attn_wr(CcBuf *q, CcBuf *kc, CcBuf *vc, CcBuf *o, CcBuf *gate, CcBuf *snk, const CcAttnW *p, int ring) {
    return cc_attn_wr_ready() ? cc_ops()->attn_wr(q, kc, vc, o, gate, snk, p, ring) : 0;
}
/* a float16 KV cache's ops (kvh): 0 from an older backend or HIP */
static inline int cc_kvh_ready(void) {
    const ColiCudaChainOps *t = cc_ops();
    return t && t->size >= offsetof(ColiCudaChainOps, copy_regions_h) + sizeof t->copy_regions_h && t->kvh();
}
static inline int cc_attn_h(CcBuf *q, CcBuf *kc, CcBuf *vc, CcBuf *o, CcBuf *gate, CcBuf *sel, const CcAttn *p) {
    return cc_kvh_ready() ? cc_ops()->attn_h(q, kc, vc, o, gate, sel, p) : 0;
}
static inline int cc_attn_wh(CcBuf *q, CcBuf *kc, CcBuf *vc, CcBuf *o, CcBuf *gate, CcBuf *snk, const CcAttnW *p, int ring) {
    return cc_kvh_ready() ? cc_ops()->attn_wh(q, kc, vc, o, gate, snk, p, ring) : 0;
}
static inline int cc_copy_regions_h(CcBuf *dst, CcBuf *src, const CcRegion *r, int n) {
    return cc_kvh_ready() ? cc_ops()->copy_regions_h(dst, src, r, n) : 0;
}
static inline int cc_kv16_ready(void) {
    const ColiCudaChainOps *t = cc_ops();
    return t && t->size >= offsetof(ColiCudaChainOps, kv16) + sizeof t->kv16 && t->kv16 && t->kv16();
}
static inline int cc_rope_f16_ready(void) {
    const ColiCudaChainOps *t = cc_ops();
    return t && t->size >= offsetof(ColiCudaChainOps, rope_f16) + sizeof t->rope_f16 && t->rope_f16;
}
static inline int cc_rope_f16(CcBuf *x, CcBuf *v, CcBuf *cs, const CcRope *p, int seg) {
    return cc_rope_f16_ready() ? cc_ops()->rope_f16(x, v, cs, p, seg) : 0;
}
static inline int cc_muse_ready(void) {
    const ColiCudaChainOps *t = cc_ops();
    return t && t->size >= offsetof(ColiCudaChainOps, muse) + sizeof t->muse && t->muse && t->muse();
}
static inline int cc_pick_ready(void) {
    const ColiCudaChainOps *t = cc_ops();
    return t && t->size >= offsetof(ColiCudaChainOps, pick) + sizeof t->pick && t->pick;
}
static inline int cc_pick(CcBuf *logits, CcBuf *out, int n) {
    return cc_pick_ready() ? cc_ops()->pick(logits, out, n) : 0;
}
static inline int cc_matmul_rows_ready(void) {
    const ColiCudaChainOps *t = cc_ops();
    return t && t->size >= offsetof(ColiCudaChainOps, matmul_rows) + sizeof t->matmul_rows && t->matmul_rows;
}
static inline int cc_matmul_rows(ColiCudaTensor *w, CcBuf *ids, CcBuf *x, CcBuf *y, int n) {
    return cc_matmul_rows_ready() ? cc_ops()->matmul_rows(w, ids, x, y, n) : 0;
}
static inline int cc_enc_ready(void) {
    const ColiCudaChainOps *t = cc_ops();
    return t && t->size >= offsetof(ColiCudaChainOps, enc_rel) + sizeof t->enc_rel && t->enc_rel;
}
static inline int cc_enc_norm(CcBuf *a, CcBuf *b, CcBuf *prm, CcBuf *y, const CcEncNorm *p) {
    return cc_enc_ready() ? cc_ops()->enc_norm(a, b, prm, y, p) : 0;
}
static inline int cc_enc_bias(CcBuf *y, CcBuf *prm, const CcEncBias *p) { return cc_enc_ready() ? cc_ops()->enc_bias(y, prm, p) : 0; }
static inline int cc_enc_geglu(CcBuf *a, CcBuf *prm, CcBuf *y, const CcEncGeglu *p) {
    return cc_enc_ready() ? cc_ops()->enc_geglu(a, prm, y, p) : 0;
}
static inline int cc_enc_rope(CcBuf *x, CcBuf *tab, CcBuf *pos, const CcEncRope *p) {
    return cc_enc_ready() ? cc_ops()->enc_rope(x, tab, pos, p) : 0;
}
static inline int cc_enc_addrow(CcBuf *y, CcBuf *tab, CcBuf *idx, const CcEncAddRow *p) {
    return cc_enc_ready() ? cc_ops()->enc_addrow(y, tab, idx, p) : 0;
}
static inline int cc_enc_attn(CcBuf *qkv, CcBuf *c2p, CcBuf *p2c, CcBuf *o, CcBuf *rng, CcBuf *ridx, CcBuf *pos, const CcEncAttn *p) {
    return cc_enc_ready() ? cc_ops()->enc_attn(qkv, c2p, p2c, o, rng, ridx, pos, p) : 0;
}
static inline int cc_enc_rel(CcBuf *x, CcBuf *pt, CcBuf *y, const CcEncRel *p) { return cc_enc_ready() ? cc_ops()->enc_rel(x, pt, y, p) : 0; }
#endif

#ifdef __cplusplus
}
#endif
#endif
