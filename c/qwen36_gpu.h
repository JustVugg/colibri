/* The Metal side of the Qwen3.6 engine, as the C code sees it.
 *
 * qwen36.c owns the model and the order of operations; qwen36_metal.m owns
 * only Metal objects. Tensors live in shared (unified) memory and stay on the
 * GPU across a whole forward pass: the C code reads them back only where it
 * must (router choices, logits).
 *
 * Operations are encoded into one command batch, in order. q36_gpu_flush()
 * starts the GPU on what was encoded so far; q36_gpu_sync() also waits for
 * it. Nothing written by the GPU is valid on the CPU before a sync, and a
 * tensor the GPU may still read must not be rewritten by the CPU. */
#ifndef QWEN36_GPU_H
#define QWEN36_GPU_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct q36_tensor q36_tensor;
typedef struct q36_map q36_map;

int q36_gpu_init(void);          /* 1 ready, 0 no usable Metal device */
/* Fast kernels for prompt batches and decode sum in another order, so their
 * results differ from the exact ones in the last bits. QWEN36_METAL_EXACT=1 (read
 * by q36_gpu_init) or q36_gpu_set_exact(1) keeps the exact kernels everywhere. */
void q36_gpu_set_exact(int on);
int q36_gpu_is_exact(void);
void q36_gpu_cleanup(void);
const char *q36_gpu_device_name(void);   /* the initialized device's name, "none" before */
/* Without q36_gpu_init: the system's default Metal device ("" when there is none),
 * and whether the kernels compile on it, with nothing else required of the device
 * (tests/test_qwen36_kernels.c on GitHub's Apple Paravirtual device). Single-threaded,
 * like the rest of this layer except q36_tensor_new and q36_tensor_free. */
const char *q36_gpu_default_device_name(void);
int q36_gpu_compile_check(void);

q36_tensor *q36_tensor_new(size_t bytes);    /* zero filled */
q36_tensor *q36_tensor_view(q36_tensor *base, size_t offset, size_t bytes);
void q36_tensor_free(q36_tensor *t);
void *q36_tensor_data(q36_tensor *t);
size_t q36_tensor_bytes(const q36_tensor *t);
q36_tensor *q36_tensor_upload(const void *data, size_t bytes);   /* new + copy */

/* Read-only file mapping (page aligned), seen by the GPU without a copy.
 * Views are served from reusable windows of 64 MiB, and a view larger than that
 * gets a window of its own: the GPU keeps only the windows a batch uses
 * resident, never the whole file. The mapping must outlive the q36_map and all
 * of its views. */
q36_map *q36_map_new(const void *base, size_t bytes);
q36_tensor *q36_map_view(q36_map *m, size_t offset, size_t bytes);
void q36_map_free(q36_map *m);

int q36_gpu_flush(void);
int q36_gpu_sync(void);
/* q36_gpu_flush, and q36_gpu_wait_mark(mark) then waits until the GPU has done
 * what was encoded before the mark, while it runs what is encoded after: the
 * CPU can read those results meanwhile. The mark is a ticket (0 on failure);
 * up to four can be waited for in any order. */
int q36_gpu_mark(void);
int q36_gpu_wait_mark(int mark);
/* Between q36_gpu_concurrent(1) and (0) the operations encoded may run at the
 * same time: they must not depend on each other, except across a
 * q36_gpu_barrier(), after which everything encoded before is done. Outside a
 * region every operation sees the results of the ones before it. */
void q36_gpu_concurrent(int on);
void q36_gpu_barrier(void);
/* q36_gpu_flush, and the GPU time of the batch in ms is added to *ms at the
 * next sync (COLI_TIMERS). */
int q36_gpu_flush_timed(double *ms);
/* Keeps the GPU clocks up for `seconds` from now, 0 stops: the graph renews
 * it during prefill and at every decode step (see q36_keepalive). */
void q36_gpu_keepalive(double seconds);
uint64_t q36_gpu_keepalive_runs(void);   /* keep-alive dispatches completed so far */

/* Shape of one layer's stateful core (rows = tokens in the batch). */
typedef struct {
    int rows, hidden;
    int q_heads, kv_heads, head_dim, q_head_dim, rotary;    /* attention */
    int v_heads, k_heads, k_dim, v_dim, conv_kernel, conv_dim; /* DeltaNet */
    int pos, cap;               /* first position of the batch, KV capacity */
    int snap;                   /* DeltaNet: copy the state after this many rows (0: none) */
} q36_shape;

/* Dense trunk. plain=0: Qwen gamma (1+w); plain=1: w. */
int q36_rmsnorm(q36_tensor *y, q36_tensor *x, q36_tensor *w, int rows, int dim, float eps, int plain);
int q36_dot_i8(q36_tensor *y, q36_tensor *x, q36_tensor *q, q36_tensor *sc, int rows, int in, int out);
/* One part of every head of a projection, in a fast prompt batch: output o
 * takes weight row (o / group) * stride + o % group, group a multiple of 32
 * (the query halves of heads that hold a query and a gate: group = head size,
 * stride = twice that). The matrix kernel only: 0, with nothing encoded, in
 * exact mode, for 8 rows or fewer, or for shapes it does not take. */
int q36_dot_i8_heads(q36_tensor *y, q36_tensor *x, q36_tensor *q, q36_tensor *sc, int rows, int in, int out, int group,
                     int stride);
/* The first out columns of a fast prompt product, into rows ldy wide: the CPU
 * computes the others. The matrix kernel only, 0 with nothing encoded where it
 * does not apply, as q36_dot_i8_heads. */
int q36_dot_i8_cols(q36_tensor *y, int ldy, q36_tensor *x, q36_tensor *q, q36_tensor *sc, int rows, int in, int out);
int q36_gate(q36_tensor *y, q36_tensor *x, q36_tensor *z, int n, int silu);
int q36_add(q36_tensor *y, q36_tensor *x, q36_tensor *r, int n);

/* Gated DeltaNet. ring: conv history; state: high then low plane, or the
 * high plane only where q36_dn_high_only: fast mode, and a shape its kernels
 * take (the exact kernels need the low plane). */
int q36_dn_high_only(const q36_shape *s);
int q36_dn_aux(q36_tensor *decay, q36_tensor *beta, q36_tensor *x, q36_tensor *a, q36_tensor *b,
               q36_tensor *alog, q36_tensor *dt, const q36_shape *s);
/* With s->snap, conv and delta also write the conv history and the state as
 * they are after that many rows into snap (same layout as ring and state). */
int q36_dn_conv(q36_tensor *y, q36_tensor *ring, q36_tensor *x, q36_tensor *w, q36_tensor *snap,
                const q36_shape *s);
int q36_dn_l2(q36_tensor *y, q36_tensor *x, const q36_shape *s);
int q36_dn_delta(q36_tensor *out, q36_tensor *state, q36_tensor *qk, q36_tensor *conv,
                 q36_tensor *decay, q36_tensor *beta, q36_tensor *snap, const q36_shape *s);

/* Gated attention. rot: rows x rotary/2 (cos, sin) pairs. */
int q36_attn_split(q36_tensor *query, q36_tensor *gate, q36_tensor *key, q36_tensor *q, q36_tensor *k,
                   const q36_shape *s);
int q36_attn_rope(q36_tensor *x, q36_tensor *rot, int heads, const q36_shape *s);
/* The KV cache holds floats, or halves (2 bytes) where q36_kv_half: fast mode
 * and a shape the fast attention kernels take for every batch (eight query
 * heads per KV head). The kernels tell the two apart by the cache's size. */
int q36_kv_half(const q36_shape *s);
int q36_attn_kv(q36_tensor *kc, q36_tensor *vc, q36_tensor *k, q36_tensor *v, const q36_shape *s);
/* scores: scratch of min(rows, Q36_ATTN_CHUNK) * q_heads * cap floats, then as
 * many rows times q_heads times ceil(cap / 32) floats (at least Q36_ATTN_CHUNK *
 * q_heads), or 3 * q_heads * cap for one row, then Q36_ATTN_CHUNK * q_heads. Exact
 * decode runs three parallel steps with the same result. Fast GQA decode can
 * use matrix QK/PV and parallel reductions when geometry and scratch fit
 * (q_heads * (cap + ceil((pos + 1) / 256) * head_dim) floats); it is the only
 * decode a half cache takes, so with less scratch q36_attn returns 0 there;
 * prompt rows run Q36_ATTN_CHUNK at a time on the matrix units, as matrices of
 * 32 query vectors where a KV head serves eight query heads. These fast paths
 * change the sum order. NULL keeps one SIMD group per row and head (float
 * caches only). */
#define Q36_ATTN_CHUNK 64
int q36_attn(q36_tensor *out, q36_tensor *q, q36_tensor *kc, q36_tensor *vc, q36_tensor *scores, const q36_shape *s);

/* MoE. ids are int32. A NULL shared gate weight means a factor of 1. */
/* Router projections are ordered in exact mode; fast mode uses the dense
 * vector/matrix kernels. Softmax and top-k rules are the same in both modes. */
int q36_router_dot(q36_tensor *y, q36_tensor *x, q36_tensor *q, q36_tensor *sc, int rows, int in, int out);
/* Decode: two routers of one row each (the layer's and the next layer's guess) in one dispatch, as two calls of
 * q36_router_dot would compute them. */
int q36_router_dot2(q36_tensor *y, q36_tensor *x, q36_tensor *q, q36_tensor *sc,
                    q36_tensor *y2, q36_tensor *x2, q36_tensor *q2, q36_tensor *sc2, int in, int out);
/* Shared projections use router_dot in exact mode and q36_dot_i8's fast
 * vector or matrix kernels otherwise. */
int q36_shared_dot(q36_tensor *y, q36_tensor *x, q36_tensor *q, q36_tensor *sc, int rows, int in, int out);
/* softmax(logits + bias) per row; logits keeps the sum (bias added in place). */
int q36_router_softmax(q36_tensor *prob, q36_tensor *logits, q36_tensor *bias, int rows, int experts);
/* With fewer eligible experts than topk (a NaN, or the groups), the missing
 * entries get id -1 and weight -1e30, and the others are not renormalized: the
 * caller decides (the CPU engine degrades instead). */
int q36_router_select(q36_tensor *ids, q36_tensor *weights, q36_tensor *prob, int rows, int experts,
                      int topk, int groups, int topk_groups);
int q36_silu_mul(q36_tensor *h, q36_tensor *g, q36_tensor *u, int n);
int q36_shared_gate(q36_tensor *gate, q36_tensor *x, q36_tensor *w, int rows, int hidden);
int q36_scale_rows(q36_tensor *x, q36_tensor *s, int rows, int hidden);

/* Routed experts, by expert: hidden and inter multiples of 64, topk at most 8
 * (other shapes are refused). A slot holds planar int4 gate|up|down and, from
 * byte `scales`, their float scales. The uint32 entries pairs[first..first+n)
 * each name a slot (bits 28..30) and a choice row*topk+k (bits 0..27).
 * h: rows*topk x inter, y: rows*topk x hidden. expert_sum then writes, per
 * row, residual + sum_k weight_k * y_k + shared, k in the router's order. */
int q36_expert_gate_up(q36_tensor *h, q36_tensor *x, q36_tensor *pairs, int first, int n,
                       q36_tensor *const slots[8], int rows, int hidden, int inter, int topk, size_t scales);
int q36_expert_down(q36_tensor *y, q36_tensor *h, q36_tensor *pairs, int first, int n,
                    q36_tensor *const slots[8], int rows, int hidden, int inter, int topk, size_t scales);
int q36_expert_sum(q36_tensor *out, q36_tensor *y, q36_tensor *weights, q36_tensor *shared,
                   q36_tensor *residual, int rows, int hidden, int topk);

#ifdef __cplusplus
}
#endif
#endif
