/* vk_tier.h -- the Vulkan routed-expert tier shared by the MoE engines.
 *
 * What it does, for a GPU a Vulkan driver can see (COLI_VULKAN=1):
 *   - holds a cache of routed experts in device memory, under a budget taken from
 *     VK_EXT_memory_budget (minus a reserve and the dense trunk the engine will
 *     place there), capped by COLI_VK_TIER_GB;
 *   - fills it at startup from the expert history (.coli_usage, route_trace.h) and
 *     keeps adapting while the user chats: an expert the CPU computes is a candidate,
 *     admitted when there is room or when it is hotter than the coldest resident by
 *     tier.h's LFRU margin, which is then evicted; uploads run on a thread of their
 *     own and never hold up a layer;
 *   - computes, per layer step, every resident expert the routing picked as ONE
 *     asynchronous submit (backend_vulkan.c's expert batch) while the engine computes
 *     the rest on the CPU, then hands back one output row per (row, rank) so the
 *     engine adds them in its own routing order;
 *   - on an integrated GPU, whose device memory is the CPU's RAM, defaults to a
 *     budget that leaves the RAM the engine's expert cache will need;
 *   - accounts device / CPU hits, uploads, evictions and how much device time the
 *     CPU work hid, in one "[VK] tier <engine>" line per run and serve turn;
 *   - with COLI_VK_DEV2=auto|<index>, holds experts on a second device too: the
 *     hottest on the primary device, the next ones on the second (its budget: its free
 *     memory less COLI_VK_RESERVE2_GB, at most COLI_VK_EXPERTS2 experts when set), each
 *     step one batch per device, both in flight at once. Nothing changes for the
 *     engine: the calls below are the same with one device or two;
 *   - keeps the RAM and the VRAM from holding the same experts when RAM is short
 *     (COLI_VK_TIER_EXCLUSIVE, on unless 0): an engine's RAM expert cache that must
 *     evict gives up first a slot whose expert the device holds (vkt_ram_first), and
 *     the prefetchers that read experts into RAM skip what the device holds. With RAM
 *     to spare nothing is evicted and nothing changes.
 * With COLI_VULKAN unset, COLI_VK_TIER=0, a CUDA tier active, or a build without
 * VK=1 (the inline stubs below), nothing here runs and the engine is unchanged.
 *
 * ---- integrating an engine ----------------------------------------------------
 * 0. Open the device saying whether the tier will be tried, so the dense matrices
 *    get their default place (on a device that shares the CPU's RAM they stay on
 *    the CPU while the tier is on; COLI_VK_DENSE decides when set):
 *
 *        ready = coli_vk_init_env_tier("glm53", vkt_wanted() && E > 0 && !cuda_tier);
 *        dense = coli_vk_dense();          // the engine's dense hook checks this
 *        ...                               // step 1
 *        if (ready && !vkt_ready() && !dense)
 *            dense = coli_vk_dense_decide("glm53", 0, 1);   // no tier after all
 *
 * 1. Describe the experts once, after the model and the dense trunk are loaded and
 *    after the engine's history is read (rt_load), before the first token:
 *
 *        VktConfig c = {.engine = "glm53", .layers = L, .experts = E,
 *                       .hidden = H, .inter = F, .topk = K,
 *                       .gate_up = {VKT_SRC_I4U_PAIRS_GS, 64},   // how RAM holds them
 *                       .down    = {VKT_SRC_I4U_PAIRS_GS, 64},
 *                       .act = VKT_ACT_SWIGLU, .act_limit = swiglu_limit,
 *                       .max_rows = prefill_rows * K,
 *                       .ram_reserve = bytes the RAM expert cache may still grow by,
 *                       .dense_bytes = bytes of dense weights the engine will put
 *                                      on the device after this call};
 *        atexit(coli_vk_shutdown);       // runs last: the device goes before the
 *        if (vkt_init(&c, rt_counts_all()))   // drivers unload (a driver can still
 *            atexit(vkt_shutdown);       // be compiling the tier's pipelines), and
 *                                        // whether or not the tier starts: vkt_init
 *                                        // makes the expert batch's pipelines
 *                                        // before it can refuse (no room)
 *
 *    .in_ram (optional) lets the tier hand experts the CPU holds in RAM back to the
 *    CPU when the device is the slower side of a step (an integrated GPU at its
 *    floor clock): give it whenever the engine can answer cheaply.
 *
 *    The heat table is [layers][experts] counts (route_trace.h's rows; NULL rows are
 *    layers without experts). Pick .gate_up/.down from the table below; gate and up
 *    must share one kind. A layer the tier should not serve (an MTP layer in another
 *    format) is simply never passed to vkt_issue: index layers 0..layers-1 only.
 *
 * 2. Warm start (recommended: with a history the hottest go first, and the rest of the
 *    device is filled either way, COLI_VK_TIER_FILL): the tier plans, the
 *    engine reads each planned expert from disk (any number of threads) and hands
 *    it over, then publishes on the engine thread:
 *
 *        int n = vkt_plan(pl, pe, max);
 *        #pragma omp parallel for
 *        for (i < n) { read expert (pl[i], pe[i]) into a private buffer;
 *                      vkt_put(pl[i], pe[i], &src); }
 *        vkt_put_done();
 *
 * 3. Every MoE layer step, rows x[S][hidden] routed to idx[S][K] (rank order, -1 =
 *    unused slot):
 *
 *        uint8_t taken[S*K];  const float *dev[S*K];
 *        int n = vkt_issue(layer, x, S, K, idx, taken);   // returns at once
 *        for every (s,k) with !taken: compute on the CPU into its own buffer,
 *            and when the expert's bytes are in RAM: vkt_note(layer, e, &src);
 *        shared experts, other CPU work ...
 *        if (n && !vkt_join(dev)) compute the taken (s,k) on the CPU too (device lost);
 *        for s: for k in rank order: out[s] += w[s][k] * (taken ? dev[s*K+k] : cpu[s][k]);
 *
 *    vkt_issue returns how many (s,k) the device took; join only when it took any.
 *
 *    Add in rank order, device or not: then the sum's order never depends on which
 *    experts happened to be resident (it is the CPU run's order, the values differ
 *    by the device's summation order only). Never call the dense Vulkan matmuls
 *    between vkt_issue and vkt_join from a parallel region (backend rule), and
 *    keep one step in flight: join before the next layer's issue.
 *
 * 4. Report: vkt_report("run" or "turn", ram_hits, disk_loads) where the engine
 *    prints its "[VK] <engine>" line; EMAP's tier bits: vkt_resident(l, e) -> 2.
 *
 * 5. Streaming (optional, for prompts; vk_tier.c, "streaming"): give .load and .release
 *    (and .load_batch when the engine reads several experts in parallel), which hand
 *    the tier an expert's bytes as the CPU path would get them. Then
 *      - block a prompt step with B = vkt_step_rows(S, block): the whole step when the
 *        tier streams, so each expert's rows meet in one GEMM;
 *      - vkt_issue takes, besides the resident experts, every cold expert the rule
 *        streams (it loads them through .load, uploads them into its staging slots and
 *        computes them as sub-batches); taken[] says which pairs, as before, and the
 *        join hands back their rows;
 *      - optionally, between submitting the frame that computes a layer's routing and
 *        waiting for it: vkt_stream_prefetch(layer, S) loads the experts that layer will
 *        likely stream while the device works.
 *    COLI_VK_TIER_STREAM=0 turns it off: the tier behaves as without the hooks.
 *
 *    COLI_VK_TIER_STREAM_RING=1 (with .ring_batch given; default off = the inline
 *    prefetch above, unchanged) moves the fills onto a worker thread, the fork's ring
 *    discipline: the prefetch call for layer L joins the ring's job for L (the take) and
 *    hands the worker the NEXT layer's predicted set (T.pred, the last big step's rows
 *    per expert at that layer -- where no prediction exists yet, a cold start, nothing
 *    is handed over and the engine fills inline as before); the worker reads the experts
 *    through .ring_batch and uploads them into its half of the streaming slots while the
 *    engine computes layer L, the step joins nothing it does not need, and every miss
 *    (an unpredicted expert, a refused read, a failed worker) takes the inline fill at
 *    the step. COLI_VK_TIER_STREAM_SYNC=1's deterministic drain includes the ring: every
 *    quiescent point joins the worker too, so what is in the slots at a step depends on
 *    the routing alone, never on thread timing.
 *
 *    COLI_VK_TIER_STREAM_RING_COLD=1 (needs the ring; default off = the pred-gated rule
 *    above stands) is the fork's literal cold shape for the cold branch: a next layer
 *    with no prediction behind it -- the prompt's first chunk, the layer's first visit --
 *    is planned whole instead of handed to the inline fill, so the job covers the next
 *    layer's whole non-resident set ahead (the same in-RAM exclusion, the same st_per
 *    cap; the below-rule filter drops, for there is no prediction to scale at the coming
 *    step's S). Where a prediction exists the plan is the prediction's, unchanged.
 *
 * Source kinds (VktSrc) and what the device holds -- the conversion runs on the
 * uploader thread, nothing changes in the engine's RAM:
 *   VKT_SRC_I8_ROW        int8, f32 per row                      -> fmt 1
 *   VKT_SRC_I8_GS         int8, f32 per (row, gs)                -> fmt 13
 *   VKT_SRC_I8_AS_I4_ROW  int8 holding int4 values [-8,7], per row -> fmt 2 (repacked: half the bytes)
 *   VKT_SRC_I8_AS_I4_GS   same, f32 per (row, gs)                -> fmt 4
 *   VKT_SRC_I4S_PAIRS_ROW signed int4, byte j = elements 2j (low), 2j+1, per row -> fmt 2
 *   VKT_SRC_I4S_PAIRS_GS  same, per (row, gs)                    -> fmt 4
 *   VKT_SRC_I4U_PAIRS_ROW int4 as v+8, pairs, per row            -> fmt 2
 *   VKT_SRC_I4U_PAIRS_GS  int4 as v+8, pairs, per (row, gs)      -> fmt 4 (GLM, glm53)
 *   VKT_SRC_I4U_PLANAR64  expert_ffn.h planar blocks of 64, v+8, f32 per 64 -> fmt 4 gs 64
 *   VKT_SRC_I3_G64        int3-g64 planes                        -> fmt 5
 *   VKT_SRC_MXFP4_F32     e2m1 pairs, f32 per gs                 -> fmt 7
 *   VKT_SRC_MXFP4_E8M0    e2m1 pairs, ue8m0 byte per gs          -> fmt 15, the bytes packed
 *                         as the store holds them (kimi_k3, mimo, deepseek_v4/_v41; fmt 7
 *                         on the CUDA tier, whose ops table has no fmt 15 yet)
 *   VKT_SRC_FP8_GS        e4m3, f32 per (row, gs)                -> fmt 12
 *   VKT_SRC_FP8_BLOCK     e4m3, f32 per gs x gs block            -> fmt 12 (qwen38 native)
 *   VKT_SRC_BF16, VKT_SRC_F32                                    -> fmt 11, 10
 *   VKT_SRC_GGML + t      ggml block type t, rows as the GGUF stores them -> fmt 100 + t
 *                         (no scales: they are in the blocks; shaders/blockdec.glsl)
 *   VKT_SRC_LB_I2G32, _I2G16, _I3G16  colibri's int2-g32 / int2-g16 / int3-g16 codes, and
 *                         as "scales" per matrix its group scales (I/32 bytes a row) then
 *                         its f32 row scales                -> fmt 200, 201, 202
 *   VKT_SRC_LB_VQ2        vq2's index bytes, as "scales" its 4-bit scales (I/64 bytes a
 *                         row) then its f32 row scales; coli_vk_set_vq2_books first -> fmt 203
 * Activations (VktConfig.act): VKT_ACT_SWIGLU silu(g)*u, act_limit > 0 clamps the
 * gate from above and up to [-limit, limit] first (GLM-5.3, DeepSeek V4/V4.1);
 * VKT_ACT_SITU a*tanh(g/a)*sigmoid(g) * b*tanh(u/b) with act_a, act_b (Kimi K3);
 * VKT_ACT_SWIGLU_V4 DeepSeek V4's expert with its CPU kernel's roundings: gate and
 * up to bf16, the clamped SwiGLU, times the route weight (vkt_issue_w) and to bf16,
 * then E4M3 and back per 128 inputs before down. The engine hands in x already
 * rounded to E4M3 per 128 (as its kernel rounds it), adds a device row with no
 * weight of its own (the weight is in it) and rounds the row to bf16 first, as
 * its CPU expert rounds its output. */
#ifndef COLI_VK_TIER_H
#define COLI_VK_TIER_H
#include <stdint.h>
#include <stddef.h>
#include "vk_policy.h"   /* the default-on levers' env resolution (the engine's store unit reads it through this header) */

typedef enum {
    VKT_SRC_NONE = 0,
    VKT_SRC_I8_ROW, VKT_SRC_I8_GS, VKT_SRC_I8_AS_I4_ROW, VKT_SRC_I8_AS_I4_GS,
    VKT_SRC_I4S_PAIRS_ROW, VKT_SRC_I4S_PAIRS_GS, VKT_SRC_I4U_PAIRS_ROW, VKT_SRC_I4U_PAIRS_GS,
    VKT_SRC_I4U_PLANAR64, VKT_SRC_I3_G64, VKT_SRC_MXFP4_F32, VKT_SRC_MXFP4_E8M0,
    VKT_SRC_FP8_GS, VKT_SRC_FP8_BLOCK, VKT_SRC_BF16, VKT_SRC_F32,
    VKT_SRC_LB_I2G32, VKT_SRC_LB_I2G16, VKT_SRC_LB_I3G16, VKT_SRC_LB_VQ2,
    VKT_SRC_GGML = 64                   /* + the ggml type: VKT_SRC_GGML + t, t < 43 */
} VktSrc;
typedef struct { VktSrc kind; int gs; } VktFmt;     /* gs: group (block) size, 0 = per row */

#define VKT_ACT_SWIGLU 0
#define VKT_ACT_SITU   1
#define VKT_ACT_SWIGLU_V4 2

/* One expert as it sits in RAM: codes and scales of gate, up, down (float scales,
 * or ue8m0 bytes for VKT_SRC_MXFP4_E8M0; NULL for bf16/f32). Read during the call
 * only: the tier copies what it keeps. */
typedef struct {
    const void *g, *u, *d;
    const void *gs, *us, *ds;
} VktExpertSrc;

/* One aliased selection for the compute-site group (COLI_VK_TIER_EXPERTS, CS-1b): the
 * store slot's own slab (imported whole, refcounted by pointer: one
 * VK_EXT_external_memory_host import serves every record it holds) and the three
 * matrices' records in it -- code and scale byte offsets, input and output widths,
 * the format (15 the MXFP4 codes with packed ue8m0 scales, 7 their f32-scale twin)
 * and its group size. The engine fills it at the token's finish site from the view
 * the loader just read; rows are the native layout (a rows16-packed or foreign-layout
 * view never reaches the tier -- the engine refuses it and that expert computes on
 * the CPU, the refusal ladder). */
typedef struct {
    const void *slab;
    size_t bytes;
    size_t woff[3], soff[3];   /* gate, up, down */
    int I[3], O[3];
    int fmt, gs;
} VktAliasSrc;

typedef struct {
    const char *engine;            /* names the [VK] tier lines */
    int layers, experts, hidden, inter, topk;
    VktFmt gate_up, down;
    int act; float act_limit, act_a, act_b;
    int max_rows;                  /* most (row, expert) assignments one vkt_issue carries */
    size_t ram_reserve;            /* bytes the engine's RAM caches may still grow by */
    /* Optional: the host pool (system RAM the device reads) comes out of the RAM those caches
     * were sized from. When the pool cannot hold every expert the device leaves, the CPU still
     * computes the others from the caches, and the two would ask the same RAM twice (an OOM
     * kill on a model bigger than RAM). ram_given: once the pool is sized, the engine lowers
     * its caches by the bytes it is given, to no less than ram_floor (what a step needs).
     * Without it the pool leaves ram_reserve to the caches and takes what is left. */
    size_t ram_floor;
    void (*ram_given)(void *ctx, size_t bytes);   /* with ram_ctx */
    /* Optional: the RAM the engine still allocates once it serves (the KV cache at the served
     * context, each conversation's, and the attention's scratch). The free RAM the host pool
     * is sized from does not show it yet; the pool leaves it. Without it the pool took the
     * room of GLM-5.3's KV cache at 131072 tokens, and the cache's allocation then failed at
     * start (#2314). */
    size_t ram_later;
    size_t dense_bytes;            /* dense weights the engine puts on the device after vkt_init */
    size_t chain_bytes;            /* what the dense chain allocates at its first forward, after the
                                      tier has filled the device (its KV mirror, a prompt chunk): left free */
    /* The engine's RAM plan, when it keeps one (0 = it keeps none, today's behaviour):
     * ram_budget is what it resolved (--memory-gb / RAM_GB / MemAvailable), ram_spare
     * what that plan leaves unclaimed -- the most the tier's host pool may take. */
    int    ram_planned;
    size_t ram_budget;
    size_t ram_spare;
    /* Optional: is this expert in the engine's RAM cache right now? With it the tier
     * balances a step: when the device keeps the CPU waiting, resident experts the
     * CPU also holds in RAM go back to the CPU (never ones it would read from disk),
     * and return to the device when it finishes early. Called on the engine thread. */
    int (*in_ram)(void *ctx, int layer, int eid);
    void *ram_ctx;
    /* Optional: at most this many experts resident whatever the budget holds (0 = no
     * cap); for an engine whose users already size its device tier in experts. */
    int max_experts;
    /* Optional, for streaming (a big prefill step's cold experts computed on the device,
     * see "streaming" below): get an expert's bytes as the engine's CPU path would (its
     * RAM cache, else the disk), valid until release(ctx, *h). 1 = *src filled; 0 = not
     * to be had (the CPU computes it). Called on the engine thread, from vkt_issue and
     * vkt_stream_prefetch, one expert at a time. Without it the tier never streams. */
    int  (*load)(void *ctx, int layer, int eid, VktExpertSrc *src, void **h);
    void (*release)(void *ctx, void *h);
    void *load_ctx;
    /* Optional, with load: n experts at once (an engine whose reads run in parallel),
     * srcs[i] and h[i] as load's, each released on its own; returns how many it loaded,
     * the first ones in order. A partial batch is consumed and released before the
     * next batch; 0 falls back to load for the next expert. */
    int  (*load_batch)(void *ctx, int layer, const int *eids, int n, VktExpertSrc *srcs, void **h);
    /* Optional, with load: the stream ring's own reader (COLI_VK_TIER_STREAM_RING), the
     * group shape of load_batch called on the ring's worker thread instead of the
     * engine thread -- straight from the engine's store, off its leases and caches, so
     * it must take no lock the engine's own loads take (a bypass/pread reader). Only a
     * leading prefix need be filled (the first ones in order), released by release(ctx,
     * h) on the same thread; an expert it does not fill stays to the engine's inline
     * fill at the step. Without it the ring never arms. */
    int  (*ring_batch)(void *ctx, int layer, const int *eids, int n, VktExpertSrc *srcs, void **h);
    /* Optional: layers past `layers` (an MTP head's) whose experts the engine holds in another
     * form (extra_gate_up, extra_down), with the same geometry: index them layers..
     * layers+extra_layers-1 in every call. They never stream (an MTP head's steps are a few
     * rows; streaming's slots are in the main form) and have no history (heat rows: the
     * main layers only). 0: none. */
    int extra_layers;
    VktFmt extra_gate_up, extra_down;
    /* Optional: 1 = the extra layers' pool holds all their experts when they take at most a
     * fifth of the budget, and the warm start puts them up first (an MTP head whose layer the
     * dense chain runs on the device, its routed experts in the frame: its steps are serial
     * with every forward, a few rows each); 0 = their layers' share of the budget. */
    int extra_whole;
    /* Optional: the main layers' forms when they differ by layer (a GGUF whose layers mix
     * block types, as published ones do): layer_gate_up[l], layer_down[l] for l < layers,
     * at most VKT_FORMS - 1 distinct pairs, gate_up/down among them or not; NULL = every
     * layer gate_up/down. The arrays are read during vkt_init only. */
    const VktFmt *layer_gate_up, *layer_down;
    /* Optional, COLI_VK_TIER_DECODE_GROW: the engine's decode-only routing counts,
     * [layers][experts] like `heat` (the .coli_mfu class -- prefill never counts there),
     * the rank that decides which experts the tier's decode-time growth installs as
     * decode runs. NULL = no such table and the growth never arms. Read during vkt_init
     * only (the tier keeps its own copy). */
    uint32_t *const *decode_heat;
    /* Optional: 1 = the dense chain copies a prompt chunk's host pool experts into the
     * device's memory a layer at a time (vk_dmoe.h): when the host pool will hold experts,
     * room for a layer's share of them comes out of the budget (vkt_stage_room). */
    int stage;
} VktConfig;
/* The forms a tier holds: form 1 the extra layers', so VKT_FORMS - 1 for the main layers.
 * unsloth's flint UD-Q2_K_XL mixes 14 (gate, down) pairs of block types over its layers. */
#define VKT_FORMS 16
#define VKT_MAX_LAYERS 512


#ifdef COLI_VULKAN
/* 1 unless COLI_VK_TIER=0: whether an engine with routed experts will try the tier
 * (for coli_vk_init_env_tier, before the engine knows whether vkt_init succeeds). */
int  vkt_wanted(void);
/* 1 = the tier is on. heat: [layers][experts] routing counts, or NULL. */
int  vkt_init(const VktConfig *cfg, uint32_t *const *heat);
int  vkt_ready(void);
/* 1 = the compute site's arm (COLI_VK_TIER_EXPERTS) is on and this tier serves it:
 * the engine wires its finish-site group seam only then, so an unarmed run keeps
 * today's issue/join path whole. */
int  vkt_cs_armed(void);
void vkt_shutdown(void);
int  vkt_plan(int *layers, int *eids, int max);
/* The warm start's line ("[VK] tier <engine>: warm start, N experts in Xs (...)"): what
 * came from the history and what filled the rest of the device (vkt_plan). */
void vkt_warm_report(const char *engine, int n, double seconds);
int  vkt_put(int layer, int eid, const VktExpertSrc *src);   /* any thread */
void vkt_put_done(void);
void vkt_note(int layer, int eid, const VktExpertSrc *src);
int  vkt_issue(int layer, const float *x, int S, int K, const int *idx, uint8_t *taken);
/* vkt_issue with the route weights w[S*K] (beside idx), for VKT_ACT_SWIGLU_V4, which
 * applies them on the device; the other activations ignore them. */
int  vkt_issue_w(int layer, const float *x, int S, int K, const int *idx, const float *w, uint8_t *taken);
/* 1 when vkt_note would take this expert now (not on the device or on its way, the
 * promotion rate not spent, room or a colder resident to displace): for an engine
 * whose RAM form must be converted before vkt_note can read it, so it converts only
 * the experts the tier will take. No side effects. */
int  vkt_wants(int layer, int eid);
/* COLI_VK_TIER_EXPERTS (CS-1b): the compute site at the token's finish. After a
 * vkt_issue that deferred the step (every valid selection marked taken, nothing
 * submitted), builds ONE expert-batch group -- the residents' cached handles beside
 * aliased store slabs (VktAliasSrc) -- and submits it on the tier's xb queue, outside
 * the chain's frame ring. On return taken[i] names the sites (1 the device computed
 * it); a selection whose alias or the batch itself refused keeps taken 0 and computes
 * on the CPU -- degrade, never drop. 0: nothing flew, every selection the CPU's. */
int  vkt_issue_group(int layer, const float *x, int S, int K, const int *idx, const float *w,
                     const VktAliasSrc *src, uint8_t *taken);
int  vkt_join(const float **rows);
/* vkt_join; for a CS group that carried the device-sum tail (COLI_VK_TIER_CS_DEVSUM),
 * *sum = its ONE merged row (each row bf16 rounded, added in rank order -- the finish
 * site's host merge, bit-for-bit) beside the rows, which stay valid; *sum = NULL on
 * every other join. */
int  vkt_join_merged(const float **rows, const float **sum);
/* vkt_join; for a big step that ran wholly on the device, *sum = its S routed rows (the
 * rank-order sum of w[s*K+k] * expert(s, k) as an fma chain from 0, the engines' loop),
 * summed on the device; else *sum = NULL. w as vkt_issue_w's (NULL = 1). */
int vkt_join_sum(const float **rows, const float *w, const float **sum);
int  vkt_resident(int layer, int eid);
/* The CS defer's own taken rule, read-only: would a decode issue mark (layer, eid)
 * taken with no read (a resident's cached ColiVkExpert handle)? The pilot
 * (COLI_VK_TIER_PILOT) skips those -- a warm read of one is pure disk traffic. */
int  vkt_cs_served(int layer, int eid);
/* The next vkt_issue starts a forward (promotion rate, decay, eviction candidates).
 * Optional: the tier tells forwards apart by the layer index going back, which a model
 * with a single MoE layer never does; such an engine calls this at its first MoE layer. */
void vkt_begin_forward(void);
void vkt_report(const char *scope, unsigned long long ram_hits, unsigned long long disk_loads);
/* Sizing helpers for engines: bytes one expert takes on the device in a source format. */
size_t vkt_expert_bytes(int hidden, int inter, VktFmt gate_up, VktFmt down);
/* Streaming: how many rows an engine hands one vkt_issue for a step of S rows whose
 * blocks are otherwise `block` rows: S when the tier streams steps of that size (the
 * whole step, so each expert's rows meet in one GEMM), else min(S, block). */
int  vkt_step_rows(int S, int block);
/* Streaming, optional: layer's experts the next step will likely stream, loaded and
 * uploaded now, while the device runs the layer's attention (the engine calls this
 * after submitting that frame and before waiting for it). The prediction is the
 * layer's routing in the previous big step (else the history); an expert it misses
 * is streamed after the routing, one it adds costs an upload. Returns how many. */
int  vkt_stream_prefetch(int layer, int S);
/* How many devices hold experts: 0 (the tier is off), 1, or 2 (COLI_VK_DEV2's too). */
int  vkt_devices(void);
/* The layers the tier serves: VktConfig.layers plus the extra layers it took (an MTP
 * head's, when their form has a device form); 0 when it is off. */
int  vkt_layers(void);
/* Exclusive RAM/VRAM: 1 when the device holds this expert and the tier wants the RAM
 * copy given up first (COLI_VK_TIER_EXCLUSIVE unset or not 0). An engine's RAM cache
 * asks it while it picks the slot to evict (prefer such a slot to its LRU choice) and
 * its prefetchers before reading an expert; vkt_ram_gave() counts a slot given up so.
 * Any thread (the prefetch workers too): it reads the slot table without the engine. */
int  vkt_ram_first(int layer, int eid);
void vkt_ram_gave(void);
/* The routed experts on the device (vk_chain.h, vkc_moe): an engine whose dense chain
 * routes and computes a layer's experts there, with no host step, for the layers whose
 * experts the tier holds on the primary device.
 *   vkt_epoch       changes whenever a resident comes or goes (the engine thread's
 *                   quiescent points: a vkt_issue, vkt_join, vkt_note, vkt_put_done,
 *                   vkt_dev_count or vkt_settle); residents stay put between two of them
 *   vkt_dev_table   layer's E entries for vkc_moe (COLI_VK_XB_ENTRY words each, a zero
 *                   entry where the expert is not resident there); returns how many are,
 *                   -1 when the device cannot read them by address
 *   vkt_dev_count   the books for S rows of K experts the device computed (idx as
 *                   vkt_issue's): heat, routed and served, as a vkt_issue that took them;
 *                   after the frame, with nothing in flight: where the host pool holds
 *                   every expert, the routed ones it served move up to the device's
 *                   memory and the coldest there go back to their copy in the pool
 *   vkt_settle      at a frame's start, nothing in flight: the experts that moved up
 *                   since take their entries (the table read next has them)
 *   vkt_dev_changes the slots (layer * experts + expert) whose entries the epochs after
 *                   `since` changed, oldest first; -1 when the tier cannot say (more than
 *                   max, or its log starts later): read the tables whole
 *   vkt_dev_entry   one expert's entry as vkt_dev_table gives it (and, ent_hybrid and hx
 *                   given, as vkt_dev_table_hybrid does; host: in the host pool): 1 resident
 *                   on the primary device, 0 not, -1 an error */
unsigned vkt_epoch(void);
int  vkt_dev_table(int layer, uint32_t *entries);
void vkt_dev_count(int layer, int S, int K, const int *idx);
void vkt_settle(void);
/* The store's slot-free path (an expert store's destroy, deepseek_v4.c) hands each
 * slab here before its pages are freed: the CS alias cache's keepalive holds on that
 * address drop and the backend's import dedup goes blind to it, so a later allocation
 * at the SAME address -- which the geometry key alone cannot tell apart -- misses and
 * imports the new pages fresh (the VA-reuse hazard's guard). Cheap and a no-op when
 * the cache holds nothing for the address. */
void vkt_slab_forget(const void *slab);
/* The forked decode path (COLI_VK_V4_DECODE_PATH=fork, gap-wave3): the expert store's
 * slot slabs route through the backend's persistent slab registry (arena blocks of
 * mapped HOST_COHERENT system RAM, one STORAGE VkBuffer a slot) so the CS alias is a
 * calloc against the registry, not a per-token host-page import. vkt_forkdec_wanted
 * reads the arm (COLI_VK_V4_DECODE_PATH=fork AND COLI_VK_TIER_EXPERTS=1/2 -- the
 * compute site is the alias's consumer; anything else: one stderr line, ignored).
 * vkt_slabreg_slot is the alloc (0: this slot keeps the malloc/import path, the
 * ladder), vkt_slabreg_slot_free the destroy-time return (after vkt_slab_forget),
 * vkt_forkdec_store_line the store's one engagement line (planned slots and slab
 * bytes, the registry's own cap line said the rest). */
int  vkt_forkdec_wanted(void);
int  vkt_slabreg_slot(size_t bytes, void **host, void **reg);
void vkt_slabreg_slot_free(void *reg, void *host);
void vkt_forkdec_store_line(int slots, size_t slab_bytes);
int  vkt_dev_changes(unsigned since, int *slots, int max);
/* The VRAM bank's refill (COLI_VK_TIER_BANK_LAYER, vk_tier.c "the VRAM bank"): at the
 * banked layer's routed step, post-router and pre-issue -- the engine's tier seam --
 * with idx the step's routed expert ids (rank order, -1 a free slot, any count). The
 * missing set -- routed, not valid yet, not tried this tenure -- loads through
 * VktConfig.load_batch (the bypass group reader where the engine armed it) and uploads
 * sequential into the bank's places, each expert at most once a tenure; a failure is
 * never re-read until the layer switches, the CPU serves it. A step of another layer is
 * not the bank's: the env's layer holds the places for the run.
 * vkt_bank_switch moves them to another layer (the fork's switch discipline: the bitmap
 * clears, the old layer's device copies free at a quiescent point, the new layer's
 * places queue) -- the rotation seam S3's layer-outer loop drives; no engine calls it
 * in S1. No-op without the tier or the bank (the default): nothing arms, nothing
 * changes. */
void vkt_bank_fill(int layer, const int *idx, int n);
void vkt_bank_switch(int layer);
/* The dense chain's hybrid steps (vk_dmoe.h): an expert of the host pool the CPU can
 * compute in place (a GGUF block type: its rows as the file has them, in system RAM) gets
 * a zero entry, so the device skips it, and its rows in hx[e]; every other expert's entry
 * as vkt_dev_table gives it, its hx[e] zero. The count is vkt_dev_table's. */
/* tg, td: a ggml type (>= 0) or -fmt, colibri's int8 and int4 device formats (fmt 1, 13, 2, 4)
 * with their scales sg/su/sd -- f32 (a row's, or a group's of gsgu/gsd columns), fmt 15's the
 * packed ue8m0 bytes, one a group (host_scale_row strides the two apart); rgu, rd: bytes a row */
typedef struct { const uint8_t *g, *u, *d; int tg, td; size_t rgu, rd; const float *sg, *su, *sd; int gsgu, gsd; } VktHostExpert;
int  vkt_dev_table_hybrid(int layer, uint32_t *entries, VktHostExpert *hx);
int  vkt_dev_entry(int layer, int eid, uint32_t *ent, uint32_t *ent_hybrid, VktHostExpert *hx, uint8_t *host);
/* The CPU's form of np pairs of the host pool's experts (the hybrid steps: vk_dmoe.h's, and
 * the tier's own batch steps): pair p is hx[p] on the row xr[p] (D floats), its output into
 * out + p * D. SwiGLU, the gate clamped from above at limit and up to [-limit, limit] first
 * when limit > 0; every matmul as the CPU computes a GGUF's (ggml_quants.h: int8 activations
 * and integer dots, with GQ_EXACT=1 the float reference), each output its own dot, so how the
 * nt threads split them moves no bit (nt 0: COLI_VK_HYBRID_THREADS, else OpenMP's). colibri's
 * int8 and int4 formats likewise: int8 activations per 32 against the codes, each block's
 * activation scale times the row's or group's (GQ_EXACT=1: the dequantized rows in doubles).
 * The pairs
 * of one expert (the same hx[p]->g) go together, its rows read once for all of them, and a
 * pair's bits are the same as on its own (ggml_quants.h's kernels take each activation row
 * apart). gu (np * 2 * I floats) and hh (np * I) its scratch.
 * 0 = a row it could not quantize, or no memory. */
int  vkt_host_pairs(const VktHostExpert *const *hx, const float *const *xr, int np, int D, int I,
                    float limit, int nt, float *gu, float *hh, float *out);
/* vkt_host_pairs for DeepSeek V4's expert (VKT_ACT_SWIGLU_V4): w[p] pair p's route weight,
 * applied between gate/up and down with the device's roundings (bf16, then E4M3 per 128),
 * so out[p] carries it, as the device's rows do. */
int  vkt_host_pairs_w(const VktHostExpert *const *hx, const float *const *xr, int np, int D, int I,
                      float limit, int nt, float *gu, float *hh, float *out, const float *w);
/* whether the tier keeps experts in its host pool (system RAM the device reads over PCIe) */
int  vkt_host_pool_on(void);
/* the device memory the budget left for a prompt chunk's staging (VktConfig.stage), 0 none */
size_t vkt_stage_room(void);
/* host[e]: 1 where layer `layer`'s expert e is in vkt_dev_table's entries from the host pool
 * (a prompt chunk stages those into VRAM: vk_dmoe.h); returns how many, -1 when off */
int  vkt_dev_host(int layer, uint8_t *host);
/* score[e]: the tier's rank of layer `layer`'s expert e (its heat and recency: the order it
 * keeps experts in, the higher the hotter); returns 0, -1 when off */
int  vkt_score(int layer, uint64_t *score);
/* The host pool's room in bytes: with the engine's RAM plan, what it leaves unclaimed
 * (never above the budget itself); without one, what is available less the engine's
 * cache growth (hold) and 4 GiB. Pure arithmetic, for tests; never above avail. */
size_t vkt_host_room(size_t avail, size_t budget, size_t spare, int planned, size_t hold);
/* Device memory back to the engine (a dense chain's KV mirror growing with the context): at
 * least `bytes` of the tier's pool on the primary device emptied and released, the coldest
 * blocks first, their residents back to their host pool copies (the CPU computes them) or
 * out of the tier; the tier then holds that much less. Between steps, on the engine's
 * thread. The bytes released, 0 when none could be. */
size_t vkt_give_back(size_t bytes);
#else
static inline int vkt_host_pool_on(void){return 0;}
static inline size_t vkt_stage_room(void){return 0;}
static inline int vkt_dev_host(int l,uint8_t *h){(void)l;(void)h;return -1;}
static inline int vkt_score(int l,uint64_t *s){(void)l;(void)s;return -1;}
static inline size_t vkt_host_room(size_t avail, size_t budget, size_t spare, int planned, size_t hold){
    (void)avail;(void)budget;(void)spare;(void)planned;(void)hold;return 0;}
static inline size_t vkt_give_back(size_t b){(void)b;return 0;}

typedef struct { const uint8_t *g, *u, *d; int tg, td; size_t rgu, rd; const float *sg, *su, *sd; int gsgu, gsd; } VktHostExpert;   /* rgu, rd: bytes a row; sg/su/sd: f32 scales, fmt 15's ue8m0 bytes */
static inline int vkt_dev_table_hybrid(int l,uint32_t *e,VktHostExpert *h){(void)l;(void)e;(void)h;return -1;}
static inline int vkt_dev_entry(int l,int e,uint32_t *t,uint32_t *th,VktHostExpert *h,uint8_t *o){(void)l;(void)e;(void)t;(void)th;(void)h;(void)o;return -1;}
static inline int vkt_host_pairs(const VktHostExpert *const *h,const float *const *x,int n,int D,int I,float l,int t,float *g,float *u,float *o){(void)h;(void)x;(void)n;(void)D;(void)I;(void)l;(void)t;(void)g;(void)u;(void)o;return 0;}
static inline int vkt_dev_changes(unsigned s,int *c,int m){(void)s;(void)c;(void)m;return -1;}
static inline int  vkt_wanted(void){return 0;}
static inline int  vkt_init(const VktConfig *c, uint32_t *const *h){(void)c;(void)h;return 0;}
static inline int  vkt_ready(void){return 0;}
static inline void vkt_shutdown(void){}
static inline int  vkt_plan(int *l,int *e,int m){(void)l;(void)e;(void)m;return 0;}
static inline void vkt_warm_report(const char *e,int n,double s){(void)e;(void)n;(void)s;}
static inline int  vkt_put(int l,int e,const VktExpertSrc *s){(void)l;(void)e;(void)s;return 0;}
static inline void vkt_put_done(void){}
static inline void vkt_note(int l,int e,const VktExpertSrc *s){(void)l;(void)e;(void)s;}
static inline int  vkt_issue(int l,const float *x,int S,int K,const int *i,uint8_t *t){(void)l;(void)x;(void)S;(void)K;(void)i;(void)t;return 0;}
static inline int  vkt_issue_w(int l,const float *x,int S,int K,const int *i,const float *w,uint8_t *t){(void)l;(void)x;(void)S;(void)K;(void)i;(void)w;(void)t;return 0;}
static inline int  vkt_wants(int l,int e){(void)l;(void)e;return 0;}
static inline int  vkt_join(const float **r){(void)r;return 0;}
static inline int  vkt_join_merged(const float **r, const float **s){(void)r;*s=0;return 0;}
static inline int  vkt_join_sum(const float **r, const float *w, const float **s){(void)r;(void)w;*s=0;return 0;}
static inline int  vkt_resident(int l,int e){(void)l;(void)e;return 0;}
static inline int  vkt_cs_served(int l,int e){(void)l;(void)e;return 0;}
static inline void vkt_begin_forward(void){}
static inline void vkt_report(const char *s,unsigned long long r,unsigned long long d){(void)s;(void)r;(void)d;}
static inline size_t vkt_expert_bytes(int h,int i,VktFmt a,VktFmt b){(void)h;(void)i;(void)a;(void)b;return 0;}
static inline int  vkt_step_rows(int S,int b){return S<b?S:b;}
static inline int  vkt_stream_prefetch(int l,int S){(void)l;(void)S;return 0;}
static inline int  vkt_devices(void){return 0;}
static inline int  vkt_layers(void){return 0;}
static inline int  vkt_ram_first(int l,int e){(void)l;(void)e;return 0;}
static inline void vkt_ram_gave(void){}
static inline unsigned vkt_epoch(void){return 0;}
static inline int  vkt_dev_table(int l,uint32_t *e){(void)l;(void)e;return -1;}
static inline void vkt_dev_count(int l,int S,int K,const int *i){(void)l;(void)S;(void)K;(void)i;}
static inline void vkt_settle(void){}
static inline void vkt_slab_forget(const void *s){(void)s;}
static inline int  vkt_forkdec_wanted(void){return 0;}
static inline int  vkt_slabreg_slot(size_t b, void **h, void **r){(void)b;(void)h;(void)r;return 0;}
static inline void vkt_slabreg_slot_free(void *r, void *h){(void)r;(void)h;}
static inline void vkt_forkdec_store_line(int s, size_t b){(void)s;(void)b;}
static inline void vkt_bank_fill(int l,const int *i,int n){(void)l;(void)i;(void)n;}
static inline void vkt_bank_switch(int l){(void)l;}
#endif

#endif
