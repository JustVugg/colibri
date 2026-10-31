/* qwen36_tier.h -- optional CUDA VRAM expert tier for the qwen36 engine.
 *
 * Applies colibri's placement concept ("route -> place -> overlap -> learn")
 * one level up from the GLM disk tier: experts live in RAM, the *hot* ones
 * are promoted into DEVICE_LOCAL VRAM across one or more GPUs and computed
 * there via the existing CUDA backend (backend_cuda.cu, expert-group API).
 *
 *  - Every expert has one home device (eid % n_gpus), no duplicates.
 *  - Routing heat decides who earns VRAM (LFRU semantics from tier.h, with
 *    hysteresis); a warmstart pre-fills the budget before the first token,
 *    ordered by a persisted heat table (HEAT_FILE) when available.
 *  - Uploads run on a background thread through staging copies; decode never
 *    blocks on placement. A VRAM miss falls back to the CPU int8 path and
 *    overlaps with the in-flight GPU groups.
 *
 * Enable with COLI_CUDA=1 [COLI_GPUS=0,1] [CUDA_EXPERT_GB=<G>|auto]
 * [HEAT_FILE=<path>] [QT_NO_WARMSTART=1]. Compiled only when the build sets
 * -DCOLI_CUDA (CUDA=1); otherwise the inline stubs below keep the engine
 * CPU-only with zero overhead. */
#ifndef QWEN36_TIER_H
#define QWEN36_TIER_H
#include <stdint.h>

#ifdef COLI_CUDA

/* Init after model load. Returns 1 when the tier is active.
 * cap_experts_per_layer >= n_experts (full RAM residency): the tier keeps raw
 * pointers into the expert slots, which are never evicted. Any smaller cap:
 * copy mode, as for fp8 below -- the slots are recycled, so the tier copies an
 * expert while the engine holds it (qt_note) and keeps no pointer
 * (qt_streaming() says which). */
/* expert_is_int4: 1 = pesi int4 impacchettati (fmt=4), 0 = int8 (fmt=1), -1 =
 * no expert format the tier takes (the mixed layout): the devices hold the
 * dense trunk only. Il chiamante lo determina dalla TAGLIA SU DISCO, non da
 * meta.ebits, che su qualche container mente (cfr. il rilevamento in qwen36.c).
 * The tier is also trunk-only for int8 experts with grouped scales and for
 * COLI_PLACE experts=cpu; qt_experts_on() is 0 then. */
/* Before qt_init: the int4 bytes qt_note hands over are expert_ffn.h's planar
 * blocks (the engine's RAM layout when its shared kernel runs the CPU's
 * experts), not the container's signed pairs; staging converts them. */
void qt_source_planar(int on);
/* R4 role split: park the dense-i8 lm_head on its own CUDA device
 * (COLI_LMHEAD_GPU=<dev>). One GEMV per token, at token end — outside the
 * per-layer latency chain — so a slower second card can host it without
 * pacing the expert stream. qt_init places no experts on that device. */
int  qt_lmhead_init(const int8_t *q, const float *sc, int I, int O);
int  qt_lmhead_matmul(float *y, const float *x, int I, int O);

/* ---- placement table (R4) ------------------------------------------------
 * Every movable piece of the forward pass can be pinned to the CPU or to a
 * specific CUDA device, so configurations can be A/B'd instead of argued
 * about. One variable, not one per component:
 *
 *   COLI_PLACE="experts=0,lmhead=0,dnproj=1,dnout=1,attnproj=cpu"
 *
 * Target is `cpu` or a CUDA ordinal. A component may also be split across
 * cards by layer count, joined with '+' so it cannot be confused with the
 * component separator:
 *
 *   COLI_PLACE="dnproj=0:15+1:15"   first 15 DeltaNet layers on dev 0, rest on 1
 *
 * Unnamed components keep their default (CPU; experts keep following
 * COLI_GPUS). Splitting matters because the DeltaNet projections hang
 * serially in the layer chain anyway -- a slower card delays only its own
 * layers, never the whole stream, which is what sank asymmetric EXPERT
 * placement. `layer` is the model layer index; pass 0 for whole-model
 * components like lmhead. */
#define QT_PLACE_CPU (-1)
int  qt_place_of(const char *component, int layer);
/* Automatic placement (COLI_PLACE unset or "auto"; "off" disables). The
 * engine offers each trunk component with its byte size BEFORE qt_init --
 * "lmhead" once (layer 0), "dnproj" per DeltaNet layer -- and qt_init decides
 * by bytes saved per token per byte of VRAM, pricing displaced experts by
 * heat. The decision is what qt_place_of() then returns, and the placed
 * bytes come out of that device's expert budget. Sizes only; the tensors
 * follow through qt_lmhead_init / qt_dnproj_init as before. */
void qt_trunk_offer(const char *component, int layer, size_t bytes);
/* The automatic placement is a prediction; the engine measures it at startup
 * (one GEMV both ways, qwen36.c trunk_probe_gpu_wins) and withdraws the whole
 * trunk when the GPU loses, giving the bytes back to the expert budget. Only
 * the automatic placement can be withdrawn; a COLI_PLACE list stands. */
int  qt_place_is_auto(void);
void qt_trunk_withdraw(const char *why);
/* The experts back to the CPU as well (the same probe; #2211), when the allowance is
 * automatic (CUDA_EXPERT_GB unset or auto). Before the warm start: nothing resident yet. */
void qt_experts_withdraw(const char *why);
/* Bytes the engine will put on the device after the tier sizes itself (the CUDA chain's own
 * router, b|a and gate rows), kept out of the automatic allowance. Call before qt_init. */
void qt_reserve(size_t bytes);
void qt_reserve_short(size_t bytes, size_t experts);   /* out of the expert budget only when fewer than `experts` fit */
/* A GGUF's arena slots at each layer's own size (one device; not for a chain that reads the arena
 * as slots of one size, qt_chain_arena). Call before qt_init_gguf. */
void qt_gguf_exact_slots(int on);

/* DeltaNet input projections, qkv ++ z fused into one resident tensor per
 * layer: one GEMV instead of two, and the engine's qkv/z buffers are laid out
 * contiguously so the result needs no split copy. */
int  qt_dnproj_init(int layer, const int8_t *q, const float *sc,
                    int I, int O, int device);
int  qt_dnproj_matmul(int layer, float *y, const float *x, int I, int O);
int  qt_dnproj_ready(int layer);
int  qt_dnproj_matmul_batch(int layer, float *y, const float *x, int S, int I, int O);

/* The whole DeltaNet layer on the device (backend coli_cuda_dn_*): after the
 * input projection (qt_dnproj_init) and the out_proj (a qt_dense handle) have
 * both landed on the same device, qt_dn_gpu_init puts the layer's conv ring,
 * recurrent state and norm/conv weights there too; qt_dn_gpu_step then runs a
 * decode token end to end on the card, host in, host out. The host copy of
 * the state stays the engine's, moved with qt_dn_gpu_set/get_state whenever
 * the CPU path (prefill, snapshot, reset) needs it. A failing step turns the
 * layer off, like every other GPU path here, and the engine continues on the
 * CPU from the state it can still get back. */
int  qt_dn_gpu_init(int layer, int vh, int vk, int kdim, int vdim, int conv_dim, int convk, int hidden,
                    const float *conv_w, const float *norm_w, float eps, int dnout_handle_plus1);
int  qt_dn_gpu_ready(int layer);
/* the same from three dense handles (in_proj qkv, in_proj z, out_proj), all on one
 * device -- qwen38's trunk items; gate_sigmoid: the gated norm's gate (1 = sigmoid(z)) */
int  qt_dn_gpu_init_dense(int layer, int vh, int vk, int kdim, int vdim, int conv_dim, int convk, int hidden,
                          const float *conv_w, const float *norm_w, float eps, int gate_sigmoid,
                          int proj_handle_plus1, int projz_handle_plus1, int dnout_handle_plus1);
int  qt_dn_gpu_set_state(int layer, const float *ring, const float *rec);   /* NULL = zero */
int  qt_dn_gpu_get_state(int layer, float *ring, float *rec);
int  qt_dn_gpu_step(int layer, const float *x, float *out, const float *egh, const float *beta);
/* Generic resident dense matrix (int8 per-row, one GEMV per call), addressed
 * by a handle: the Qwen3.8 trunk uses this for every matrix it places. Offer
 * the size with qt_trunk_offer(name, layer, bytes) before qt_init, ask
 * qt_place_of(name, layer) after it, then hand the quantized bytes here.
 * Returns the handle (>= 0) or -1 (stays on the CPU). */
int  qt_dense_init(const int8_t *q, const float *sc, int I, int O, int device);
/* The same three with the tensor's format: 1 int8 rows (scales sc), 100 + ggml type a GGUF's
 * blocks as the file has them (sc NULL: the backend's block formats, coli_cuda_block_elems). */
int  qt_dense_init_fmt(const void *w, const float *sc, int fmt, int I, int O, int device);
int  qt_lmhead_init_fmt(const void *w, const float *sc, int fmt, int I, int O);
int  qt_dnproj_init_fmt(int layer, const void *w, const float *sc, int fmt, int I, int O, int device);
int  qt_dense_matmul(int handle, float *y, const float *x, int I, int O);
/* Row-major x[S,I] -> y[S,O], using the same resident int8 tensor. */
int  qt_dense_matmul_batch(int handle, float *y, const float *x, int S, int I, int O);
int  qt_dense_count(void);
/* The resident tensors behind the handles, for the dense chain (qwen36_cuda_chain.h):
 * the backend's object and the CUDA ordinal it lives on; NULL when nothing is placed. */
struct ColiCudaTensor *qt_dense_tensor(int handle, int *dev);
struct ColiCudaTensor *qt_dnproj_tensor(int layer, int *dev);
struct ColiCudaTensor *qt_lmhead_tensor(int *dev);

/* fp8 streaming mode (Qwen3.8): experts arrive as e4m3 bytes with 128x128
 * block scales and do NOT all fit in RAM. cap may be smaller than n_experts;
 * the tier copies what it uploads inside the qt_note call and keeps no
 * pointer into the engine's slot. e4m3_lut is quant.h's E4M3_LUT, published
 * to the backend so fmt=8 uploads are accepted. */
/* Init returns 0 without changing an active tier. Shut down before reinit;
 * callers must serialize init/shutdown with new work. */
int  qt_init_fp8(int n_layers, int n_experts, int hidden, int inter,
                 int cap_experts_per_layer, int topk, const float *e4m3_lut);
int  qt_init(int n_layers, int n_experts, int hidden, int inter,
             int cap_experts_per_layer, int topk, int expert_gs,
             int expert_is_int4);
/* A GGUF's experts as the file has them (fmt 100 + ggml type: the backend's block formats,
 * coli_cuda_block_elems): gate and up of type tg[l], down of td[l] in layer l, one gate (or
 * up) matrix gu_bytes[l] bytes and one down d_bytes[l]; qt_note hands over the file's rows
 * (g4, u4, d4; no scales); copy 1: those rows live only for the call (copy mode, as below
 * full residency). The residents compute in the dense chain's frame (qt_chain_layer_format)
 * and in the tier's expert groups. */
int  qt_init_gguf(int n_layers, int n_experts, int hidden, int inter,
                  int cap_experts_per_layer, int topk, const int *tg, const int *td,
                  const size_t *gu_bytes, const size_t *d_bytes, int copy);
int  qt_ready(void);
int  qt_streaming(void);   /* copy mode: no pointer into the engine's slots is kept */
int  qt_experts_on(void);  /* some device has room for experts (not trunk-only) */
int  qt_is_resident(int layer, int eid);
void qt_shutdown(void);

/* Call for each routed expert the CPU computes (pointers to the RAM slot:
 * packed int4, planar int4 after qt_source_planar(1), or, on an int8
 * container, the live int8 weights -- tier fmt=1 is accepted since #1334; see
 * also #1391 for the decode-path offer). Counts its heat (qt_issue counts the
 * hits') and may enqueue a background upload; in copy mode the bytes are
 * copied before this returns. */
void qt_note(int layer, int eid,
             const uint8_t *g4, const uint8_t *u4, const uint8_t *d4,
             const float *gs, const float *us, const float *ds);

/* Launch the GPU groups for the resident subset of the K selected experts
 * (async, all devices in parallel). Returns a bitmask of the k handled by
 * the GPU. Compute the misses on the CPU, then call qt_take(). */
uint32_t qt_issue(int layer, const int *eids, int K, const float *x);

/* Collect all GPU results and accumulate val[k]*y_k into out[hidden].
 * Returns 0 on collection failure, leaving out unchanged. The caller must
 * stop inference: experts selected by qt_issue were not computed on CPU. */
int qt_take(uint32_t mask, const float *val, int K, float *out);
/* qt_take with Muse Spark's norm on every expert's row first: out += w * row*s / rms(row*s),
 * eps inside the root (the CPU path's products, in its order) */
int qt_take_scaled(uint32_t mask, const float *val, int K, float *out, const float *s, float eps);

/* Warmstart: plan the full fill set (heat order, budget reserved), then any
 * number of loader threads may call qt_note_planned per planned expert. */
int  qt_plan_fill(int *layers, int *eids, int max);
/* A layer that routes nothing (Flint's dense layers): qt_plan_fill skips it. */
void qt_layer_off(int layer);
void qt_note_planned(int layer, int eid,
             const uint8_t *g4, const uint8_t *u4, const uint8_t *d4,
             const float *gs, const float *us, const float *ds);
int  qt_fill_next(int *layer, int *eid);
void qt_note_block(int layer, int eid,
             const uint8_t *g4, const uint8_t *u4, const uint8_t *d4,
             const float *gs, const float *us, const float *ds);
void qt_fill_wait(void);   /* blocks until every enqueued upload is resident (not merely dequeued) */

/* Re-plan the resident set from the prompt's own routing. counts[] are this
 * prompt's routing counts per expert: for layer >= 0 one layer's ne counts
 * (the engine calls this after each prefill layer's routing, so the swaps of
 * layer L upload while layers L+1.. still compute), for layer < 0 the whole
 * nl*ne table. Residents the prompt never routed to are swapped, budget-
 * neutral, for the prompt's most-routed non-residents of the same layer (or
 * device, for the whole-table form), through the victim-first swap the LFRU
 * tick uses, while the newcomer's count strictly exceeds the victim's. As
 * many swaps as the upload queue takes start at once, the rest are pending
 * and drain on later calls and, a few per token, on the decode ticks: nothing
 * blocks. Returns the pairs planned by this call. */
int  qt_replan(int layer, const uint32_t *counts, int max_swaps);
/* Experts the tier evicted since the last call (LFRU and re-plan swaps), as
 * (layer, eid) pairs, oldest first; the engine rebuilds their CPU copies ahead
 * of the miss path. Bounded ring: what overflows is simply not reported, the
 * miss path rebuilds lazily as before. Returns the count written. */
int  qt_evicted_take(int *layers, int *eids, int max);
/* Snapshot the hit/miss counters; qt_stats then also reports the hit rate
 * from this point on (the engine marks the prefill/decode boundary). */
void qt_stats_mark(void);

/* One telemetry block on stderr: residency, hits/misses, uploads per device. */
void qt_stats(void);

/* The dense chain's routed experts (the engines' *_cuda_chain.h: cc_route, cc_moe): the
 * chain computes the residents on its device by address and the host the rest.
 *   qt_chain_epoch     moves with every change of residency: the chain's table is stale
 *   qt_chain_resident  expert eid of layer, resident on CUDA ordinal device: its tensors
 *   qt_chain_hold      1 before a forward that reads the table, 0 after: no swap writes
 *                      into a resident while one is held (an eviction clears the flag at
 *                      once and moves the epoch, its buffers wait)
 *   qt_chain_count     the books qt_issue keeps for K routed ids of one row (-1 skipped):
 *                      the heat of the residents, hits and misses, the LFRU tick at layer 0
 *   qt_chain_format    the residents' format as the backend holds them (1 int8, 2 int4,
 *                      4 int4 per gs inputs, 8 fp8-e4m3, 100 a GGUF's types by layer) and gs
 *   qt_chain_layer_format  layer's gate/up and down formats (100 + ggml type for a GGUF)
 *   qt_chain_devices   the tier's device count
 *   qt_chain_history   1 when the warm start's set came from a history (HEAT_FILE, or an
 *                      engine's qt_heat_order); 0: the experts in index order */
unsigned qt_chain_epoch(void);
int  qt_chain_resident(int layer, int eid, int device, struct ColiCudaTensor **g, struct ColiCudaTensor **u, struct ColiCudaTensor **d);
void qt_chain_hold(int on);
void qt_chain_count(int layer, const int *eids, int K);
int  qt_chain_format(int *fmt, int *gs);
int  qt_chain_layer_format(int layer, int *fmt, int *fmt_d);
/* a GGUF's arena for a chain that moves the experts itself (qwen36_tier.c's comment) */
int  qt_chain_arena(void **base, size_t *slot, size_t *off_u, size_t *off_d, int *nslots);
int  qt_chain_slot(int layer, int eid);
void qt_chain_own(void);
/* copy mode: 1 when a qt_note of the expert now would copy it (convert the bytes only then);
 * qt_note_heat counts a pass of one the engine does not hand over */
/* the warm start's order from an engine's expert history (counts[l][e], nl x ne) */
void qt_heat_order(uint32_t *const *counts, int nl, int ne);
int  qt_note_wanted(int layer, int eid);
void qt_note_heat(int layer, int eid);
int  qt_chain_devices(void);
int  qt_chain_history(void);
/* The chain's promotions (qwen36_tier.c's comment): after a decode-sized step, up to max of its
 * misses into the buffers of the coldest residents by this conversation's routes -- the pairs,
 * the newcomer's RAM bytes and the victim's tensors, for the chain to copy; qt_chain_promote_done
 * once a pair's copy landed (ok) or failed; the changes of residency since the chain last asked
 * (slot indices; *overflow: read every entry again); how many promotions landed so far */
typedef struct {
    int ci, vi;                                   /* newcomer and victim: layer * ne + eid */
    const uint8_t *g, *u, *d; const float *gs, *us, *ds;   /* the newcomer's RAM copy */
    struct ColiCudaTensor *tg, *tu, *td;          /* the victim's device tensors */
    size_t mb, sb_gu, sb_d;                       /* a matrix's bytes; the gate/up and down scales' bytes */
} QtPromo;
int  qt_chain_promote(int max, QtPromo *out);
void qt_chain_promote_stage(void *dst, const QtPromo *p);
void qt_chain_promote_done(int ci, int vi, int ok);
int  qt_chain_changes(int *idx, int max, int *overflow);
uint64_t qt_chain_promoted(void);
/* The host pool (qwen36_tier.c): qt_note_host page-locks a non-resident expert's RAM copy (its
 * planar blocks and scales, as the engine's slot holds them) for the device to copy a prompt
 * chunk's experts from; qt_chain_host the copy's two blocks, the three matrices and the three
 * scale arrays (1 when registered); qt_chain_host_count how many; qt_chain_host_follow makes
 * the pool the non-residents again after experts moved (1 when it changed). */
void qt_note_host(int layer, int eid, const uint8_t *g4, const uint8_t *u4, const uint8_t *d4,
                  const float *gs, const float *us, const float *ds);
int  qt_chain_host(int layer, int eid, const void **w, size_t *wb, const float **sc, size_t *sb);
int  qt_chain_host_count(void);
int  qt_chain_host_follow(int *joined, int *left);

#else /* !COLI_CUDA: inline stubs, engine stays CPU-only */

static inline int  qt_init(int a,int b,int c,int d,int e,int f,int g,int h){(void)h;(void)a;(void)b;(void)c;(void)d;(void)e;(void)f;(void)g;return 0;}
static inline int  qt_init_fp8(int a,int b,int c,int d,int e,int f,const float*g){(void)a;(void)b;(void)c;(void)d;(void)e;(void)f;(void)g;return 0;}
static inline int  qt_init_gguf(int a,int b,int c,int d,int e,int f,const int*g,const int*h,const size_t*i,const size_t*j,int k){(void)a;(void)b;(void)c;(void)d;(void)e;(void)f;(void)g;(void)h;(void)i;(void)j;(void)k;return 0;}
static inline int  qt_lmhead_init(const int8_t*a,const float*b,int c,int d){(void)a;(void)b;(void)c;(void)d;return 0;}
static inline int  qt_lmhead_matmul(float*a,const float*b,int c,int d){(void)a;(void)b;(void)c;(void)d;return 0;}
#define QT_PLACE_CPU (-1)
static inline int  qt_place_of(const char*a,int b){(void)a;(void)b;return QT_PLACE_CPU;}
static inline void qt_trunk_offer(const char*a,int b,size_t c){(void)a;(void)b;(void)c;}
static inline int  qt_place_is_auto(void){return 0;}
static inline void qt_trunk_withdraw(const char*a){(void)a;}
static inline void qt_experts_withdraw(const char*a){(void)a;}
static inline void qt_reserve(size_t b){(void)b;}
static inline void qt_reserve_short(size_t b,size_t e){(void)b;(void)e;}
static inline void qt_gguf_exact_slots(int on){(void)on;}
static inline int  qt_dnproj_init(int a,const int8_t*b,const float*c,int d,int e,int f){(void)a;(void)b;(void)c;(void)d;(void)e;(void)f;return 0;}
static inline int  qt_dnproj_matmul(int a,float*b,const float*c,int d,int e){(void)a;(void)b;(void)c;(void)d;(void)e;return 0;}
static inline int  qt_dnproj_ready(int a){(void)a;return 0;}
static inline int  qt_dnproj_matmul_batch(int a,float*b,const float*c,int d,int e,int f){(void)a;(void)b;(void)c;(void)d;(void)e;(void)f;return 0;}
static inline int  qt_dn_gpu_init(int a,int b,int c,int d,int e,int f,int g,int i,const float*j,const float*k,float l,int m){(void)a;(void)b;(void)c;(void)d;(void)e;(void)f;(void)g;(void)i;(void)j;(void)k;(void)l;(void)m;return 0;}
static inline int  qt_dn_gpu_ready(int a){(void)a;return 0;}
static inline int  qt_dn_gpu_init_dense(int a,int b,int c,int d,int e,int f,int g,int i,const float*j,const float*k,float l,int m,int n,int o,int p){(void)a;(void)b;(void)c;(void)d;(void)e;(void)f;(void)g;(void)i;(void)j;(void)k;(void)l;(void)m;(void)n;(void)o;(void)p;return 0;}
static inline int  qt_dn_gpu_set_state(int a,const float*b,const float*c){(void)a;(void)b;(void)c;return 0;}
static inline int  qt_dn_gpu_get_state(int a,float*b,float*c){(void)a;(void)b;(void)c;return 0;}
static inline int  qt_dn_gpu_step(int a,const float*b,float*c,const float*d,const float*e){(void)a;(void)b;(void)c;(void)d;(void)e;return 0;}
static inline int  qt_dense_init(const int8_t*a,const float*b,int c,int d,int e){(void)a;(void)b;(void)c;(void)d;(void)e;return -1;}
static inline int  qt_dense_init_fmt(const void*a,const float*b,int c,int d,int e,int f){(void)a;(void)b;(void)c;(void)d;(void)e;(void)f;return -1;}
static inline int  qt_lmhead_init_fmt(const void*a,const float*b,int c,int d,int e){(void)a;(void)b;(void)c;(void)d;(void)e;return 0;}
static inline int  qt_dnproj_init_fmt(int a,const void*b,const float*c,int d,int e,int f,int g){(void)a;(void)b;(void)c;(void)d;(void)e;(void)f;(void)g;return 0;}
static inline int  qt_dense_matmul(int a,float*b,const float*c,int d,int e){(void)a;(void)b;(void)c;(void)d;(void)e;return 0;}
static inline int  qt_dense_matmul_batch(int a,float*b,const float*c,int d,int e,int f){(void)a;(void)b;(void)c;(void)d;(void)e;(void)f;return 0;}
static inline int  qt_dense_count(void){return 0;}
static inline void *qt_dense_tensor(int a,int*b){(void)a;(void)b;return NULL;}
static inline void *qt_dnproj_tensor(int a,int*b){(void)a;(void)b;return NULL;}
static inline void *qt_lmhead_tensor(int*b){(void)b;return NULL;}
static inline int  qt_ready(void){return 0;}
static inline unsigned qt_chain_epoch(void){return 0;}
struct ColiCudaTensor;
static inline int  qt_chain_resident(int a,int b,int c,struct ColiCudaTensor**g,struct ColiCudaTensor**u,struct ColiCudaTensor**d){(void)a;(void)b;(void)c;(void)g;(void)u;(void)d;return 0;}
static inline void qt_chain_hold(int a){(void)a;}
static inline void qt_chain_count(int a,const int*b,int c){(void)a;(void)b;(void)c;}
static inline int  qt_chain_format(int*a,int*b){(void)a;(void)b;return 0;}
static inline int  qt_chain_layer_format(int a,int*b,int*c){(void)a;(void)b;(void)c;return 0;}
static inline int  qt_chain_arena(void**a,size_t*b,size_t*c,size_t*d,int*e){(void)a;(void)b;(void)c;(void)d;(void)e;return 0;}
static inline int  qt_chain_slot(int a,int b){(void)a;(void)b;return -1;}
static inline void qt_chain_own(void){}
static inline void qt_heat_order(uint32_t*const*a,int b,int c){(void)a;(void)b;(void)c;}
static inline int  qt_note_wanted(int a,int b){(void)a;(void)b;return 0;}
static inline void qt_note_heat(int a,int b){(void)a;(void)b;}
static inline int  qt_chain_devices(void){return 0;}
static inline int  qt_chain_history(void){return 0;}
typedef struct { int ci, vi; const uint8_t *g, *u, *d; const float *gs, *us, *ds; struct ColiCudaTensor *tg, *tu, *td; size_t mb, sb_gu, sb_d; } QtPromo;
static inline int  qt_chain_promote(int a,QtPromo*b){(void)a;(void)b;return 0;}
static inline void qt_chain_promote_stage(void*a,const QtPromo*b){(void)a;(void)b;}
static inline void qt_chain_promote_done(int a,int b,int c){(void)a;(void)b;(void)c;}
static inline int  qt_chain_changes(int*a,int b,int*c){(void)a;(void)b;*c=0;return 0;}
static inline uint64_t qt_chain_promoted(void){return 0;}
static inline void qt_note_host(int l,int e,const uint8_t*g,const uint8_t*u,const uint8_t*d,const float*a,const float*b,const float*c){(void)l;(void)e;(void)g;(void)u;(void)d;(void)a;(void)b;(void)c;}
static inline int  qt_chain_host(int l,int e,const void**w,size_t*wb,const float**sc,size_t*sb){(void)l;(void)e;(void)w;(void)wb;(void)sc;(void)sb;return 0;}
static inline int  qt_chain_host_count(void){return 0;}
static inline int  qt_chain_host_follow(int*j,int*l){*j=*l=0;return 0;}
static inline int  qt_streaming(void){return 0;}
static inline int  qt_experts_on(void){return 0;}
static inline void qt_source_planar(int a){(void)a;}
static inline int  qt_is_resident(int a,int b){(void)a;(void)b;return 0;}
static inline void qt_shutdown(void){}
static inline void qt_note(int a,int b,const uint8_t*c,const uint8_t*d,const uint8_t*e,const float*f,const float*g,const float*h){(void)a;(void)b;(void)c;(void)d;(void)e;(void)f;(void)g;(void)h;}
static inline uint32_t qt_issue(int a,const int*b,int c,const float*d){(void)a;(void)b;(void)c;(void)d;return 0;}
static inline int qt_take(uint32_t a,const float*b,int c,float*d){(void)b;(void)c;(void)d;return a==0;}
static inline int qt_take_scaled(uint32_t a,const float*b,int c,float*d,const float*s,float e){(void)b;(void)c;(void)d;(void)s;(void)e;return a==0;}
static inline int  qt_plan_fill(int*a,int*b,int c){(void)a;(void)b;(void)c;return 0;}
static inline void qt_layer_off(int a){(void)a;}
static inline void qt_note_planned(int a,int b,const uint8_t*c,const uint8_t*d,const uint8_t*e,const float*f,const float*g,const float*h){(void)a;(void)b;(void)c;(void)d;(void)e;(void)f;(void)g;(void)h;}
static inline int  qt_fill_next(int*a,int*b){(void)a;(void)b;return 0;}
static inline void qt_note_block(int a,int b,const uint8_t*c,const uint8_t*d,const uint8_t*e,const float*f,const float*g,const float*h){(void)a;(void)b;(void)c;(void)d;(void)e;(void)f;(void)g;(void)h;}
static inline void qt_fill_wait(void){}
static inline int  qt_replan(int l,const uint32_t*a,int b){(void)l;(void)a;(void)b;return 0;}
static inline int  qt_evicted_take(int*a,int*b,int c){(void)a;(void)b;(void)c;return 0;}
static inline void qt_stats_mark(void){}
static inline void qt_stats(void){}

#endif /* COLI_CUDA */
#endif /* QWEN36_TIER_H */
