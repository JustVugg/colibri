#ifndef COLIBRI_ENGINE_H
#define COLIBRI_ENGINE_H
/* Shared engine types: what the engine core and the per-arch modules (arch_*.c)
 * both need, so an arch module can be compiled as a standalone .c file.
 *
 * ORDERING: include this AFTER st.h (shards), backend_vulkan.h (ColiVkTensor)
 * and, under COLI_CUDA, backend_cuda.h
 * (ColiCudaTensor, COLI_CUDA_MAX_DEVICES) — Layer/Model reference those types.
 *
 * Cfg/Layer hold what the engine core needs for the registered families; a new
 * arch adds its fields here alongside its descriptor, and they stay 0 for the
 * archs that don't use them. */
#include <stdint.h>
#include <stdio.h>

typedef struct {
    int hidden, n_layers, n_heads, n_experts, topk, moe_inter, dense_inter;
    int first_dense, q_lora, kv_lora, qk_nope, qk_rope, qk_head, v_head, n_shared, vocab;
    int n_group, topk_group, norm_topk;
    int stop_ids[8], n_stop;                     /* eos_token_id dal config (GLM-5.2 ne ha 3!) */
    int index_topk, index_nh, index_hd;          /* DSA lightning indexer */
    int8_t idx_type[128];                        /* per layer: 1=full (calcola), 0=shared (riusa) */
    float eps, theta, attn_scale, routed_scale;
} Cfg;

/* tensore [O,I] in uno di tre formati:
 *   fmt=0 F32   -> qf
 *   fmt=1 INT8  -> q8 (1 byte/param) + scala per riga
 *   fmt=2 INT4  -> q4 (2 valori per byte, impacchettati) + scala per riga
 * INT4 e' cio' che fa stare la densa residente nei 15 GB (0.5 byte/param). */
/* fmt: 0 F32, 1 INT8, 2 INT4 (2/byte), 3 INT2 (4/byte), 4 INT4-GROUPED, 5 INT3-G64,
 * 6 E8/IQ3 lattice, 8 FP8-E4M3 (native, passthrough -- see quant.h). fmt=7 is
 * MXFP4 (Kimi K3 Vulkan tier, #676/#705, backend_vulkan.c) -- claimed upstream,
 * never a QT format here, deliberately absent from this struct's dispatch.
 * fmt=8 is a PUBLIC ordinal: it developed under the PRIVATE ORDINAL BLOCK
 * convention below as fmt=100, graduated to fmt=7 when the maintainer assigned
 * that ordinal on #524, and was renumbered to 8 after #705 merged claiming 7
 * for MXFP4 while this PR was still open (see the convention comment below).
 * q4 ospita int4/int2/int3 packed. fmt=4 (grouped int4, #242): per-row nibbles + one f32
 * scale per group of `gs` inputs (s has O*ceil(I/gs) entries).
 * fmt=6 (E8/IQ3 lattice, #452): 98B per 256 weights = 3.0625 bits/weight, grid
 * indices + parity-packed signs + sub-scales + fp16 super-scale, ALL inside q4 —
 * `s` is unused for this format (see quant.h E8_* and tools/iq3_pack.py).
 * fmt=5 (int3, per-GROUP scales, group=64, see quant.h I3_*): values in [-4,3] stored per
 * 64-input group as 24 bytes = 16B low plane (2 bits/val, int2 layout) + 8B high plane
 * (1 bit/val), plus ONE f32 scale PER GROUP (s has O*ceil(I/64) entries, not O). 3.5
 * bits/weight effective — the quality/size sweet spot measured in the #132 ablation.
 * fmt=8 (native FP8-e4m3 passthrough, resident/quality-core tier -- see quant.h's
 * FP8_BLOCK/e4m3_decode/matmul_fp8): q8 holds O*I raw e4m3 bytes, ONE byte per weight,
 * byte-identical layout to fmt=1's weight bytes (q4 is unused/NULL, same as fmt=1) --
 * disambiguated from fmt=1 (and, at small [O,I], from fmt=6 -- see below) purely by
 * scale geometry, see qt_resolve_fmt's "THE DESIGN LANDMINE" comment further down.
 * s holds ONE f32 scale PER 128x128 BLOCK, ceil(O/128)*ceil(I/128) entries total
 * (row-major: block-row-major then block-col), NOT O and NOT O*ceil(I/gs) --
 * qt_bytes()/qt_scale_bytes() below are the authoritative byte-count formulas. The
 * scale ENCODING is itself a declared PROPERTY of this format, not a hardcoded
 * constant -- f32 (4 bytes/block, what's above) is the value THIS build
 * implements; qt_resolve_fmt's "SCALE ENCODING IS A DECLARED PROPERTY" comment
 * documents why (a DeepSeek-V4 checkpoint ships this identical weight geometry
 * with a UE8M0 scale encoding instead) and how an unimplemented encoding is
 * recognized and refused by name rather than silently misread.
 * gs is unused (0) for fmt=8, same as fmt 1/2/3. */
/* ---- PRIVATE ORDINAL BLOCK CONVENTION ------------------------------------
 * fmt values 0-8 are upstream-assigned, public, stable ordinals -- do not
 * reuse or renumber them (fmt=8 is the newest member -- see "renumbered"
 * above). fmt values 100+ remain this repo's PRIVATE/EXPERIMENTAL
 * block for any OTHER in-flight format proposal: ordinals a branch mints for
 * itself during development so it can't collide with a number upstream claims
 * out from under it. That collision has now bitten this SAME format twice:
 * first when #465's E8/IQ3 proposal claimed its original private number,
 * fmt=6, upstream while this branch was still developing against it (forcing
 * the re-mint to fmt=100 -- see upstream_contribution/FORMATS_registry_draft.md
 * for the incident that prompted the rule); and again when #705 merged MXFP4
 * as fmt=7 while this PR sat open already holding the maintainer-assigned
 * ordinal 7 from #524, forcing the 7 -> 8 renumber recorded here. Ordinals
 * are only settled by MERGE into dev, not by assignment on an open PR --
 * docs/FORMATS.md (PR 2 of this pair) is the registry meant to make the
 * next claim visible before it lands.
 *
 * A PRIVATE-BLOCK ordinal is an internal enum value only -- qt_resolve_fmt
 * (below) infers format purely from byte arithmetic; the container on disk
 * carries no format ordinal at all. (A self-describing container stamp that
 * would persist a format's NAME, not its ordinal, is a follow-up proposal --
 * see qt_resolve_fmt's own note on where that plumbing would attach -- not
 * present in this build.) Nothing outside this binary's own compiled code
 * ever observes a 100+ number, so renumbering one later (e.g. when a format
 * is upstreamed and assigned a real public ordinal, as has now happened twice
 * for fmt=8) is a pure find-and-replace with zero on-disk or cross-version
 * compatibility impact.
 *
 * Rule for adding a new format to this branch or a future one: claim the next
 * unused 100+ integer, never a number already claimed upstream (check dev AND
 * open PRs before picking one -- dev alone was not enough to prevent either of
 * fmt=8's two renumbers) or by another in-flight private format. Never ship a
 * 100+ ordinal as a public default/committed-upstream value -- the real
 * ordinal is only settled when the format MERGES into dev, exactly as fmt=8's
 * two renumbers demonstrate. */
typedef struct {
    int fmt; float *qf; int8_t *q8; uint8_t *q4; float *s; int O, I, gs;  /* gs=group size (0=per-row, 128=grouped) */
#ifdef COLI_CUDA
    ColiCudaTensor *cuda;
#endif
#ifdef COLI_VULKAN
    ColiVkTensor *vk; int vk_eligible;   /* resident on the Vulkan expert tier */
#endif
    int cuda_eligible, cuda_failed, cuda_device;  /* resident tensor, never a reused expert slot */
} QT;

typedef struct Layer {
    float *in_ln, *post_ln;
    /* MLA (densa, quantizzata) */
    QT q_a, q_b, kv_a, kv_b, o; float *q_a_ln, *kv_a_ln;
#ifdef COLI_CUDA
    ColiCudaTensor *kv_b_shard[COLI_CUDA_MAX_DEVICES];
    int shard_h0[COLI_CUDA_MAX_DEVICES],shard_hn[COLI_CUDA_MAX_DEVICES],n_kv_b_shard;
    int shared_w4a16_failed;
#endif
    int sparse;
    /* dense mlp (sparse==0) */
    QT gate_proj, up_proj, down_proj;
    /* moe (sparse==1) */
    float *router, *router_bias;                 /* router f32 (sensibile) */
#ifdef COLI_CUDA
    void *router_cuda, *router_bias_cuda;        /* device router (#431 PR-A), lazy-uploaded */
    int router_cuda_bad;                         /* upload failed once: stay on the CPU router */
#endif
    QT sh_gate, sh_up, sh_down;                  /* shared expert */
} Layer;

/* slot di un expert: pesi quantizzati + scale. Nel container pre-quantizzato g/u/d sono
 * VISTE dentro `slab` (una sola pread coalescente); nel fallback hanno buffer propri.
 * slab_cap/fslab_cap: capienza allocata — gli slot ws[] sono riusati TRA layer e gli
 * expert non hanno tutti la stessa taglia (layer MTP int8 = 2x i layer int4). */
typedef struct { int eid; QT g,u,d; uint8_t *slab; float *fslab;
                 int64_t slab_cap, fslab_cap; uint64_t used;
                 /* pin-arena backing (#419): when set, slab/fslab are interior
                  * slices of a per-layer arena and must never be free()d —
                  * expert_host_release detaches them, expert_host_ensure
                  * re-attaches. NULL for every individually-allocated slot. */
                 uint8_t *aslab; float *afslab; } ESlot;

typedef struct KVState {
    float **Lc, **Rc, **Ic;
    int *kv_start, max_t;
    int disk_nrec;
    char disk_path[2048];
    FILE *disk_fp;       /* kept-open handle: fopen once, fwrite per turn, fclose at exit (#4) */
    uint8_t *disk_buf;   /* staging buffer: one contiguous record per position (#1) */
    int64_t disk_buf_cap;
} KVState;

typedef struct {
    KVState *kv;
    int token, pos;
} DecodeRow;

typedef struct Model {
    Cfg c; shards S;
    int ebits, dbits;                            /* bit expert / bit densa */
    QT embed, lm_head; float *final_norm;
    Layer *L;
    /* KV-cache MLA COMPRESSA: per token si tiene solo il latente normato [kv_lora] e
     * k_rot [qk_rope] (576 vs 32768 valori/token). k_nope e value si ricostruiscono al
     * volo con kv_b. E' cio' che rende gestibile il contesto su 15 GB (64 teste, no GQA). */
    float **Lc, **Rc; int max_t;                 /* alias della KVState attiva */
    int *kv_start;                               /* prima pos valida nella KV del layer (MTP: parziale) */
    KVState *kv;
    ESlot **ecache; int *ecn; int ecap;          /* LRU expert per-layer */
    float **kv_dev_L, **kv_dev_R; int *kv_dev_valid; /* ombra KV su device (decode) */
    float **ln_dev;                              /* in_ln/post_ln cached on device: [layer*2+{0,1}] (Inc.4) */
#ifdef COLI_VULKAN
    int *vk_kv_valid;                            /* righe [0,v) specchiate nella cache KV Vulkan */
#endif
    ESlot ws[64];                                /* working set del layer corrente (load paralleli) */
    ESlot **pin; int *npin;                      /* HOT-STORE: expert pinnati in RAM (mai evicted) */
    uint32_t **eusage;                           /* contatori persistenti (per STATS/PIN) */
    uint32_t **eheat;                            /* calore recente per promotion/demotion live */
    uint32_t **elast, eaccess_clock;              /* recency per LFRU session-local */
    /* DISK-CLASS: PRIVATE recency state, read only by expert_classify(). Private --
     * not the real elast/eaccess_clock -- kept fully separate so DISK-CLASS's bookkeeping
     * can never read from or write into stock eviction state: every DISK-CLASS write lives
     * inside its own need_classify/dc_on gate, so "byte-identical with PROF=0" is provable
     * by construction instead of by argument. (Historical note: when this was first written,
     * the Metal pre-routed FASE A path (g_pre_idx) never bumped the real elast/eaccess_clock
     * -- on Metal decode the real clock froze at end of prefill, so REPIN's LRU tie-breaker
     * ran on stale recency for the rest of the run. That was an upstream defect; it has since
     * been reported and fixed (#417, cfcc742) -- FASE A now bumps the real clock too. The
     * private clock is retained anyway: separation from stock state is the stronger property,
     * independent of whether the real clock is correct.) elast_dc/eaccess_clock_dc tick in
     * BOTH FASE A paths, under the same need_classify gate, at the same rate the real clock
     * ticks on the CPU path (one per selected (position,expert)) -- so the
     * COLI_DISKCLASS_WINDOW window keeps its meaning in every mode. elast_pre snapshots
     * elast_dc just BEFORE this call's own bump (see the touched[] guard in FASE A) --
     * classifying against the live array would read the bump routing just made a few lines
     * above the load that needed it, so a giant cold prefill burst would score every expert
     * "just accessed" and get called warm. Recency alone (not eheat's access COUNT): a count
     * never decays, so an expert hot early in a long session would keep reading "warm" long
     * after it dropped out of the working set. Same shape/allocation as elast; NULL for dense
     * layers. */
    uint32_t **elast_dc, **elast_pre, eaccess_clock_dc;
    /* DSA lightning indexer (attivo solo se i pesi out-idx-* sono presenti) */
    int has_dsa;
    QT *ix_wq, *ix_wk, *ix_wp;                   /* per layer FULL: wq_b, wk, weights_proj */
    float **ix_knw, **ix_knb;                    /* k_norm (LayerNorm, eps 1e-6) */
    float **Ic;                                  /* alias KVState: cache indexer [max_t*hd] */
    int *dsa_sel, *dsa_nsel; int dsa_scap;       /* selezione per posizione del batch corrente */
    /* testa MTP (layer n_layers, stile DeepSeek-V3): draft nativi ad alta acceptance */
    int has_mtp; Layer mtpL; QT eh_proj;
    float *enorm, *hnorm, *mtp_norm;
    float *hlast, *h_all;                        /* hidden pre-norm: ultima pos / tutte le pos batch */
    uint64_t mtp_prop, mtp_acc;                  /* statistica acceptance */
    int **eroute; int *enr;                      /* metodo C: routing dell'ULTIMO token per layer */
    uint64_t eclock, hits, miss, ereq;
    uint64_t hit_pin, hit_ecache;                /* split di hits per tier (#336): pin vs LRU ecache */
    uint64_t hit_vk;                             /* VK VRAM tier hits (registry-served, no RAM load) */
    uint64_t gpu_expert_calls; int gpu_expert_count; int64_t gpu_expert_bytes;
    uint64_t n_fw, n_emit;                       /* metodo E: forward di decode / token emessi */
    uint64_t route_slots, route_swaps;            /* CACHE_ROUTE: slots chosen / substituted vs true top-K */
    uint64_t route_agree_hit, route_agree_tot;    /* ROUTE_AGREE: |chosen ∩ true top-K| / K */
    double route_kl_sum; uint64_t route_kl_n;     /* mean KL(true||chosen) on gate mass */
    double t_ewait, t_emm, t_ecpu, t_egpu, t_route, t_p2p, t_attn, t_kvb, t_head;
    uint64_t n_p2p;                              /* P0 execution profile: tier split + residual hops */
    uint64_t cpu_expert_rows; int64_t cpu_expert_bytes;
                                                 /* profiling: dove va il tempo (wall del
                                                  * thread di compute; il servizio disco
                                                  * overlappato vive in g_edisk_ns) */
    double t_aproj,t_acore,t_aout;                     /* attention breakdown */
    int64_t resident_bytes;
    /* DISK_SPLIT=1: split dei DISK LOAD (miss LRU -> expert_load) per contesto e per tipo
     * di layer. ld_ctx: 0=main/verify/prefill, 1=dentro mtp_draft, 2=dentro mtp_absorb. */
    int ld_ctx;
    uint64_t miss_draft, miss_absorb;            /* miss in moe() per contesto */
    uint64_t ld_mtp, ld_main;                    /* expert_load per tipo layer (MTP int8 vs main int4) */
    uint64_t bytes_mtp, bytes_main;              /* byte letti da disco per tipo layer */
} Model;

/* Per-architecture forward-pass ops. The engine dispatches the two
 * arch-divergent stages — attention and the MoE route+expert stage — through
 * this vtable, selected from the ModelArch metadata at load time
 * (model_ops_bind in colibri.c).
 *
 * Only the canonical per-layer dispatch (layer_forward_rows) goes through this
 * vtable. The GLM-only fused fast paths (Metal full-layer, CUDA pipe_layer_sparse)
 * call the built-ins directly on purpose: they are gated by GLM-specific config
 * (D==6144, n_heads==64, MLA dims) that a non-GLM arch never satisfies. */
typedef struct ModelOps {
    void (*attention_rows)(Model *m, Layer *l, int layer, float *x, int S, int pos_base,
                           KVState *const *kvs, const int *positions, float *out);
    void (*moe)(Model *m, Layer *l, int layer, float *x, int S, float *out, int with_shared);
} ModelOps;

#endif
