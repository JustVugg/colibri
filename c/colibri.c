/* Motore GLM-5.2 (architettura glm_moe_dsa) in C puro.
 * Stadio B: replica fedele del forward di transformers (modeling_glm_moe_dsa.py):
 *   - attenzione MLA (q/kv-LoRA, RoPE interleaved parziale)
 *   - router sigmoid + noaux_tc (n_group=1) con routed_scaling_factor
 *   - shared expert + expert routed in streaming dal disco (per-expert)
 *   - primi first_k_dense_replace layer densi
 * Il DSA indexer e' un NO-OP per seq <= index_topk (seleziona tutte le key): qui si usa
 * attenzione causale densa -> output identico all'oracolo su prompt corti.
 *
 * QUANTIZZAZIONE: gli expert (streaming) e la parte DENSA residente (attenzione, lm_head,
 * embed, mlp densa, shared expert) sono tenuti in int8 per-riga + scala (dequant-on-use).
 * E' cio' che fa entrare GLM-5.2 nei 15 GB: ~17B param residenti a int4 ~= 8.7 GB.
 * Norme/router/bias restano f32 (piccoli e sensibili).
 *
 * Validazione: stessi token id di ref_glm.json (oracolo transformers, c/tools/make_glm_oracle.py).
 *   build: make glm   run: SNAP=./glm_tiny ./glm <cap> <expert_bits> <dense_bits>
 *   TF=1 -> teacher-forcing (valida il prefill su tutta la sequenza)
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <math.h>
#include <time.h>
#include <limits.h>
#include <pthread.h>                              /* thread I/O del PILOTA */
#include <stdatomic.h>                            /* PIPE ready-flags/job queue + PILOT_REAL cross-layer handshake */
#include <sched.h>                                /* sched_yield: PIPE spin / PILOT barrier */
#include <unistd.h>
#if defined(__APPLE__) || defined(__linux__) || defined(__FreeBSD__)
#include <sys/select.h>                             /* select() serve-loop polling (#68); not on native MinGW */
#include <sys/socket.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#endif
#if defined(__APPLE__) || defined(__linux__) || defined(__FreeBSD__)
#include <sys/resource.h>
#include <sys/mman.h>                             /* mlock: inchioda le pagine in RAM / wire pages into RAM */
#ifdef __linux__
#include <sys/syscall.h>                          /* COLI_NUMA: mbind degli slab expert / expert-slab interleave */
#include <malloc.h>                               /* Needed to actually free memory if we go over our RAM budget */
#endif
#include <sys/stat.h>                             /* fstat per mmap degli shard (COLI_MMAP) */
#include <signal.h>                               /* SIGINT = stop morbido del turno in serve mode */
#elif defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <psapi.h>                                /* Required for GetProcessMemoryInfo */
#endif
#ifdef __linux__
#include <sys/vfs.h>                              /* statfs: real fs-type check for the 9p warning (below) */
#endif
#if defined(_WIN32) && (defined(__x86_64__) || defined(__i386__))
#include <cpuid.h>                                /* hwinfo_emit: CPU brand string senza /proc */
#endif
#include "st.h"
#ifdef __linux__
#include "uring.h"
#endif
#include "tok.h"
#include "tier.h"
#include "grammar.h"                              /* metodo F: draft grammaticali (#48) */
#include "abl.h"                                   /* per-expert causal-ablation harness — inert unless g_abl.mode set (ABLATE_SCORE=<manifest>) */
#include "schema_gbnf.h"                          /* SCHEMA=: JSON-Schema -> GBNF for method F */
#include "decode_batch.h"
#include "route_trace.h"                           /* ROUTE_TRACE + .coli_usage, engine-agnostic (#700) */
#include "kv_fp8.h"                               /* KV8=1: cache latente in fp8 e4m3 + scala per-riga */
#include "kv_tq.h"                                /* KV_TQ=3|4: cache latente PolarQuant (rot+polare) */
#ifdef _OPENMP
#include <omp.h>                                  /* scratch per-thread nell'attention */
#else
/* Shims for a build without an OpenMP runtime (Apple clang without Homebrew
 * libomp is the common case; the Makefile warns and carries on). Every omp_*
 * the engine calls needs one, or the fallback only appears to work: the CPU
 * build compiled because it happens not to reach omp_in_parallel(), while
 * METAL=1 does and failed. */
static inline int omp_get_max_threads(void){ return 1; }
static inline int omp_get_thread_num(void){ return 0; }
static inline int omp_in_parallel(void){ return 0; }     /* no runtime: never inside a team */
static inline void omp_set_num_threads(int n){ (void)n; }
#endif
#include "omp_tune.h"
#ifdef COLI_SEGMENT_ADAPTER
#include "segment_runtime.h"
#include "segment_adapters.h"
#include "segment_adapter_internal.h"
#endif
#ifdef COLI_EDGE_ADAPTER
#include "edge_runtime.h"
#include "edge_adapters.h"
#include "edge_tok_internal.h"
#endif
#ifdef COLI_CUDA
#include "backend_cuda.h"
#endif
#ifdef COLI_VULKAN
#include "backend_vulkan.h"
#endif
/* Declared unconditionally (not just under COLI_METAL): on a non-Metal build they just sit
 * at 0 forever, which is the correct value there (no Metal backend => never enabled, and
 * COLI_METAL_MOE_EXACT is only parsed under COLI_METAL). Kept outside the #ifdef so
 * portable code — the shared fused-path format predicate metal_fused_layer_fmt_miss and
 * metal_fmt_gate_notice below — can read "is Metal active" and the MOE-exact mode without
 * needing COLI_METAL (and the backend_metal.h / Metal framework link it would drag in) on
 * every platform's test build. */
static int g_metal_enabled;
static int g_moe_exact=0;     /* COLI_METAL_MOE_EXACT: exact (ungrouped) MoE kernels only */
/* COLI_SLAB_SHRINK=0 disables the #856 slab shrink at runtime.
 *
 * The shrink stops a wide slab migrating into a narrow row through the
 * ws[]<->LRU swap, which is what lets the cap be priced per row. It is also the
 * only new allocator work in that change: a slot alternating between the int8
 * MTP row and the int4 main rows reallocates instead of staying wide, and at
 * ~19-38 MB glibc serves that with mmap/munmap plus page faults on first touch.
 * #869 stated that cost was unmeasured; #885 then measured five runs on an
 * int4-g64 container at 1.12/0.98/1.11/0.97/1.05 -- spread 0.15, bimodal --
 * against <=0.02 on every other arm of the same box.
 *
 * This exists so that hypothesis can be A/B'd without a rebuild. Setting it to 0
 * restores grow-only and re-introduces the migration, so the per-row cap becomes
 * optimistic again: a diagnostic, not a supported configuration. */
static int g_slab_shrink = 1;
#ifdef COLI_METAL
#include "backend_metal.h"
/* No <omp.h> here: the guarded include above already provides it under _OPENMP
 * and shims omp_get_max_threads/omp_get_thread_num without it. Including it
 * unconditionally defeated that fallback and made METAL=1 unbuildable on a
 * stock macOS -- exactly the platform this backend targets -- even though the
 * Makefile advertises the single-threaded path ("libomp not found: building
 * single-threaded"). */
/* g_metal_enabled is declared unconditionally above (#587) so portable code can
 * read it; the duplicate that used to sit here was a tentative definition the
 * linker merged, hence no warning and no symptom. */
static int g_metal_gemm_min=16;   /* COLI_METAL_GEMM_MIN: min rows to send a matmul_qt GEMM to GPU */
/* g_moe_exact moved next to g_metal_enabled above, outside the #ifdef: the shared
 * fused-path format predicate (metal_fused_layer_fmt_miss) consumes it and must
 * compile on every platform's test build. */
/* output dello shared expert gia' calcolato su GPU (solo Metal layer-CB) */
static const float *g_pre_sh;
#endif
/* routing precalcolata dalla GPU (Metal layer CB o device router CUDA, #431):
 * moe() la usa e salta la FASE A. NULL = router su CPU. */
static const int *g_pre_idx; static const float *g_pre_w; static const int *g_pre_keff;
#if !defined(_WIN32)
typedef struct { int fd; char host[128]; int port; } ClusterWorker;
static ClusterWorker g_cluster_workers[16];
static int g_cluster_n;
#endif
#ifdef __APPLE__
#include <mach/mach.h>                            /* host_statistics64: MemAvailable di macOS */
#endif

/* #379: set in main() when the F_NOCACHE storage probe (coli_ssd_probe_cached,
 * defined below) measured the model volume as fast (Metal + darwin only).
 * Stays 0 on every other platform/build, so every read of it below is a no-op
 * there -- see coli_resolve_cap() and cap_for_ram()'s CAP_RAISE default. */
static int g_ssd_fast;

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
    int planar;  /* K1: fmt=2 con nibble a PIANI (vedi quant.h) — i kernel *_i4p; 0 = coppie classiche */
#ifdef COLI_CUDA
    ColiCudaTensor *cuda;
#endif
#ifdef COLI_VULKAN
    ColiVkTensor *vk; int vk_eligible;   /* resident on the Vulkan expert tier */
#endif
    int cuda_eligible, cuda_device;   /* resident tensor, never a reused expert slot */
    /* #767: the row count of the smallest call that has failed on this tensor, or 0 if
     * none has. A CUDA failure here is almost always a scratch cudaMalloc under memory
     * pressure, and that pressure scales with S -- so it is a property of the CALL, not
     * of the tensor. Recording S instead of a boolean lets a narrower call retry: a
     * prefill chunk that OOMs at S=512 does not condemn decode at S=1. A genuine device
     * fault fails at S=1, records 1, and is never retried -- the old behaviour, reached
     * as a special case of the general rule rather than as a separate one. */
    int cuda_fail_s;
} QT;
static int64_t qt_bytes(const QT *t){    /* byte residenti del tensore */
    int64_t n=(int64_t)t->O*t->I;
    if(t->fmt==0) return n*4;
    if(t->fmt==1) return n + (int64_t)t->O*4;
    if(t->fmt==3) return (int64_t)t->O*((t->I+3)/4) + (int64_t)t->O*4;
    if(t->fmt==4){ /* int4 grouped: packed nibbles + O*ceil(I/gs) scales */
        int ng=(t->I+t->gs-1)/t->gs;
        return (int64_t)t->O*((t->I+1)/2) + (int64_t)t->O*ng*4; }
    if(t->fmt==5){ /* int3-g64: 24B/group weights + one f32 scale per group (I3_* in quant.h,
                    * included below — keep the arithmetic literal here) */
        int64_t ng=((int64_t)t->I+63)/64;
        return (int64_t)t->O*ng*24 + (int64_t)t->O*ng*4; }
    if(t->fmt==6)  /* E8/IQ3: 98B per 256 weights, scales in-block, .qs is a 4-byte tag */
        return (int64_t)t->O*(((int64_t)t->I+255)/256)*98 + 4;
    if(t->fmt==8){ /* fp8-e4m3 passthrough: O*I raw e4m3 bytes (n, byte-identical layout
                    * to fmt=1's weight bytes) + one f32 scale per 128x128 block
                    * (FP8_BLOCK in quant.h, included below qt_bytes -- keep the
                    * arithmetic literal here, same discipline as fmt=5's comment
                    * above). Missing this branch would fall through to the fmt=2
                    * default below (packed-nibble formula, ~half the real weight
                    * bytes) and undercount a resident fp8 tensor's byte footprint --
                    * feeds AUTOPIN/RAM-budget math, so this branch is load-bearing
                    * from day one, not a later fix (contrast fmt=6 above, upstream's
                    * own #452 review round 1 caught this exact bug shape for E8). */
        int64_t nblkO=((int64_t)t->O+127)/128, nblkI=((int64_t)t->I+127)/128;
        return n + nblkO*nblkI*4; }
    return (int64_t)t->O*((t->I+1)/2) + (int64_t)t->O*4;  /* fmt=2 int4 per-row */
}
/* scale-array byte count only, format-aware -- split out of qt_bytes() because
 * qt_wire_mmap/qt_unwire_mmap mlock the weight and scale ranges as TWO SEPARATE
 * regions (separate allocations, not one contiguous buffer: qt_from_disk always
 * qalloc's t->q8/t->q4 and t->s independently) and need the scale byte count on
 * its own to split qt_bytes(t) into a weight half and a scale half. Hardcoding
 * scale_b=O*4 at those two call sites -- i.e. assuming PER-ROW scale for every
 * format -- is right for fmt 0/1/2/3 but wrong for fmt=4 (grouped,
 * O*ceil(I/gs) scales), fmt=5 (int3-g64, O*ceil(I/64) scales), fmt=6 (E8/IQ3,
 * a FIXED 4-byte tag, see below), and fmt=8 (per-128x128-block,
 * ceil(O/128)*ceil(I/128) scales): any tensor in one of those four formats
 * reaching qt_wire_mmap/qt_unwire_mmap would mlock/munlock the wrong byte
 * ranges on BOTH halves (weight_b = qt_bytes(t)-scale_b shifts too). This is
 * not purely a dormant fmt=8-only fix: fmt=4 is the routed-expert
 * "int4-g64" format (qt_resolve_fmt assigns it to ESlot g/u/d the same as
 * any other tensor, see expert_load_impl), so any COLI_MMAP +
 * mem_should_wire() session pinning fmt=4 experts mlocks the wrong ranges
 * without this. UNVERIFIED at runtime (no model run performed -- static/
 * arithmetic fix only, see the report).
 *
 * REVIEW FINDING (maintainer, #528): this function existed correctly (the
 * fmt=8 branch below is right) but neither qt_wire_mmap nor qt_unwire_mmap
 * actually called it -- both independently hardcoded scale_b=(int64_t)t->O*4
 * inline, so the "fix" this comment describes was never applied at either
 * call site, and the resulting -Wunused-function warning on this then-dead
 * function was suppressed by -Wno-unused-function in CFLAGS rather than
 * caught. qt_wire_split() below is now the ONE place both call sites get
 * weight_b/scale_b from -- see it and its call sites for the actual fix.
 *
 * FIX ROUND (audit finding, SHOULD-FIX): this comment used to ALSO claim
 * fmt=6 (E8/IQ3) was "deliberately NOT covered" because "t->fmt==6 tensors
 * never reach qt_wire_mmap/qt_unwire_mmap's mlock path in this build
 * (int3-g64/E8 stay CPU-side, see qt_cuda_upload -- MMAP wiring is a
 * CUDA/Metal-adjacent memory concern this build doesn't extend to them)" --
 * that conflated two UNRELATED mechanisms: qt_cuda_upload decides GPU-VRAM
 * upload eligibility (fmt=5/6 genuinely excluded there, no CUDA kernel for
 * either -- see qt_cuda_upload's own `fmt==5||fmt==6` guard), while
 * qt_wire_mmap/qt_unwire_mmap mlock HOST RAM pages under COLI_MMAP, an
 * entirely CPU-side concern. Staying CPU-side (no CUDA kernel) is exactly
 * what makes a tensor a CANDIDATE for RAM wiring, not exempt from it.
 * expert_load_impl assigns fmt=6 to ESlot g/u/d exactly like fmt=4/5 (same
 * qt_resolve_fmt call site, three lines above the fmt=4 comment), and
 * pin_wire calls qt_wire_mmap/qt_unwire_mmap on every pinned ESlot's g/u/d
 * unconditionally, with no format filter -- so a pinned E8/IQ3 expert under
 * COLI_MMAP + mem_should_wire() DOES reach here. Before the fmt=6 branch
 * below, the O*4 fallback would have returned a scale_b wildly larger than
 * the tensor's real scale allocation (a FIXED 4 bytes, qsalloc(1) in
 * qt_from_disk's fmt==6 branch -- confirmed against qt_bytes' own `+4`
 * literal and qt_from_disk's st_read_f32_cap cardinality switch, both of
 * which already treat fmt=6's scale as exactly 1 float, not O floats), then
 * mlock/munlock'd that many bytes starting at t->s -- past the end of a
 * 4-byte allocation. UNVERIFIED at runtime (same static/arithmetic-only
 * caveat as the fmt=4/5/8 fix above -- no model run performed). */
static int64_t qt_scale_bytes(const QT *t){
    if(t->fmt==4){ int ng=(t->I+t->gs-1)/t->gs; return (int64_t)t->O*ng*4; }
    if(t->fmt==5){ int64_t ng=((int64_t)t->I+63)/64; return (int64_t)t->O*ng*4; }
    if(t->fmt==6) return 4;   /* E8/IQ3: FIXED 4-byte tag (qsalloc(1)), not O*4 -- see this
                              * function's own header comment for why this is reachable and
                              * load-bearing, not dead code. */
    if(t->fmt==8){ int64_t nblkO=((int64_t)t->O+127)/128, nblkI=((int64_t)t->I+127)/128; return nblkO*nblkI*4; }
    return (int64_t)t->O*4;   /* fmt 0 (t->s NULL, guarded by callers)/1/2/3: per-row. */
}
/* qt_wire_mmap/qt_unwire_mmap's shared weight/scale byte-range split -- the ONE
 * place that computes it, so the two sites can never independently drift back
 * to a hardcoded per-row-only formula (that drift -- qt_scale_bytes() existing
 * but unused at both sites -- was the maintainer's #528 finding, see the
 * comment above qt_scale_bytes()). Directly unit-tested (test_fp8_load.c) so a
 * regression here, not just at a call site, fails the suite. */
static void qt_wire_split(const QT *t, int64_t *weight_b, int64_t *scale_b){
    *scale_b = qt_scale_bytes(t);
    *weight_b = qt_bytes(t) - *scale_b;
}

typedef struct {
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
                 unsigned in_flight; /* async GPU readers borrowing this slot */
                 /* pin-arena backing (#419): when set, slab/fslab are interior
                  * slices of a per-layer arena and must never be free()d —
                  * expert_host_release detaches them, expert_host_ensure
                  * re-attaches. NULL for every individually-allocated slot. */
                 uint8_t *aslab; float *afslab; } ESlot;

static void eslot_acquire(ESlot *s){ __atomic_add_fetch(&s->in_flight,1,__ATOMIC_ACQ_REL); }
static void eslot_release(ESlot *s){
    unsigned old=__atomic_fetch_sub(&s->in_flight,1,__ATOMIC_ACQ_REL);
    if(!old){ fprintf(stderr,"[CUDA] ESlot reference underflow\n"); abort(); }
}
static int eslot_busy(const ESlot *s){ return __atomic_load_n(&s->in_flight,__ATOMIC_ACQUIRE)!=0; }
static void eslots_acquire(ESlot **slots,int n){ for(int i=0;i<n;i++) eslot_acquire(slots[i]); }
static void eslots_release(ESlot **slots,int n){ for(int i=0;i<n;i++) eslot_release(slots[i]); }
/* Victim per una riga piena (#1034): uno slot svuotato da rss_guard (eid=-1,
 * slab=NULL) e' riusabile SOLO finche' gli slab vivi della riga stanno sotto
 * ecap — riusarlo rialloca uno slab, quindi e' crescita, non eviction. Le
 * prenotazioni in volo (eid<-1) contano come vive: stanno per possederne uno.
 * EN: reusing a slab-less slot re-allocates, so it only counts as eviction
 * EN: while the row's live-slab count is under ecap; else pick a slab owner. */
static int eslot_lru_victim(ESlot *slots,int n,int ecap){
    int lru=-1, empty=-1, live=0;
    for(int i=0;i<n;i++){
        ESlot *s=&slots[i];
        if(s->slab || s->eid<-1) live++;
        if(eslot_busy(s) || s->eid<-1) continue;
        if(!s->slab){ if(s->eid==-1 && empty<0) empty=i; continue; }
        if(s->eid==-1) return i;              /* slot libero che possiede ancora lo slab */
        if(lru<0 || s->used<slots[lru].used) lru=i;
    }
    if(empty>=0 && live<ecap) return empty;   /* sotto il tetto: meglio il vuoto che sfrattare */
    return lru;
}

typedef struct {
    float **Lc, **Rc, **Ic;
    uint8_t **Lc8, **Rc8;                        /* KV8: righe latenti fp8 e4m3 (Lc/Rc restano NULL) */
    float **Lsc, **Rsc;                          /* KV8: scala amax/448 per riga (per token, per layer) */
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

typedef struct {
    Cfg c; shards S;
    int ebits, dbits;                            /* bit expert / bit densa */
    QT embed, lm_head; float *final_norm;
    Layer *L;
    /* KV-cache MLA COMPRESSA: per token si tiene solo il latente normato [kv_lora] e
     * k_rot [qk_rope] (576 vs 32768 valori/token). k_nope e value si ricostruiscono al
     * volo con kv_b. E' cio' che rende gestibile il contesto su 15 GB (64 teste, no GQA). */
    float **Lc, **Rc; int max_t;                 /* alias della KVState attiva */
    uint8_t **Lc8, **Rc8; float **Lsc, **Rsc;    /* alias KV8 (fp8 + scale) della KVState attiva */
    int *kv_start;                               /* prima pos valida nella KV del layer (MTP: parziale) */
    KVState *kv;
    ESlot **ecache; int *ecn; int ecap;          /* LRU expert per-layer */
    int **ecache_slot_by_expert;                 /* eid -> LRU slot (resident or PILOT reservation) */
    float **kv_dev_L, **kv_dev_R; int *kv_dev_valid; /* ombra KV su device (decode) */
    float **ln_dev;                              /* in_ln/post_ln cached on device: [layer*2+{0,1}] (Inc.4) */
#ifdef COLI_VULKAN
    int *vk_kv_valid;                            /* righe [0,v) specchiate nella cache KV Vulkan */
#endif
    ESlot ws[64];                                /* working set del layer corrente (load paralleli) */
    ESlot **pin; int *npin;                      /* HOT-STORE: expert pinnati in RAM (mai evicted) */
    int **pin_slot_by_expert;                    /* eid -> hot-store slot, -1 if absent */
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

#include "quant.h"

/* Runtime policy knobs used only by the GLM engine.  Keep them here rather
 * than in quant.h: that header is shared by standalone kernel tests and
 * sibling engines, where translation-unit-local copies are unused and trip
 * -Wunused-variable. */
static int g_idot=1;
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
static int g_i4s=1;
#elif defined(__VSX__)
static int g_i4s=1;
#elif defined(__AVX512VNNI__) && defined(__AVX512BW__)
static int g_i4s=1;   /* AVX-512 VNNI: come SDOT, l'IDOT int4 conviene anche a S=1. Misurato su
                       * 2x Xeon 8370C (48 core, GLM-5.2 int4 tutto residente, TEMP=0 DRAFT=0,
                       * 256 token): 3.65 -> 3.85 tok/s (+5.5%), expert-matmul 67.8 -> 89.5 GB/s.
                       * EN: with AVX-512 VNNI, like SDOT, int4 IDOT pays at S=1 too. Measured on
                       * a 2-socket Ice Lake (config above): +5.5% end-to-end greedy decode. */
#else
static int g_i4s=2;
#endif
static int g_xexp=0;  /* XEXP=1 (opt-in): S==1 decode, all-resident int4 block -> ONE OpenMP
                       * region across all experts of the batch-union block instead of ~2
                       * fork/joins per expert. Engages only with the int4-IDOT S=1 family
                       * (g_i4s<=1) and off the speculation window (spec_pinned): output is
                       * byte-identical to that family (same dot_i4i8 per row, same silu,
                       * same requant, same accumulation order into out). Measured on a
                       * 2-socket Ice Lake 48C (GLM-5.2 int4 fully resident, TEMP=0 DRAFT=0,
                       * 256 tok greedy, ABAB 3 prompts x 2 reps): 4.20 -> 4.68 tok/s
                       * (+11.6% mean, worst prompt +11.3%), expert-matmul effective
                       * 89.5 -> 131.9 GB/s. A similar restructuring was NEUTRAL/negative on
                       * a 24-core box (docs/experiments/glm52-6x5090-2026-07-12.md) - hence
                       * opt-in; measure on your host. */
static int g_no_fused_pair=0;
static int g_spec_pin=1;
static int g_spec_live=0;
static inline int spec_pinned(void){ return g_spec_pin && g_spec_live; }

static void matmul_qt_ex(float *y, const float *x, QT *w, int S, int allow_idot);
static void matmul_qt(float *y, const float *x, QT *w, int S){ matmul_qt_ex(y,x,w,S,1); }

/* fmt=4 fused gate+up (defined later, after the quant kernels) */
static void matmul_i4_grouped_pair(float *yg, float *yu, const float *x,
                                    const uint8_t *qg, const float *sg,
                                    const uint8_t *qu, const float *su,
                                    int S, int I, int O, int gs);

static void expert_gate_up(float *g,float *u,const float *x,QT *wg,QT *wu,int S){
    if(!g_no_fused_pair&&!spec_pinned()&&S==1&&wg->fmt==2&&!wg->planar&&!wu->planar&&wu->fmt==2&&wg->I==wu->I&&wg->O==wu->O)
        matmul_i4_pair(g,u,x,wg->q4,wg->s,wu->q4,wu->s,wg->I,wg->O);
    else if(!g_no_fused_pair&&S==1&&wg->fmt==4&&!wg->planar&&!wu->planar&&wu->fmt==4&&wg->I==wu->I&&wg->O==wu->O&&wg->gs==wu->gs)
        matmul_i4_grouped_pair(g,u,x,wg->q4,wg->s,wu->q4,wu->s,S,wg->I,wg->O,wg->gs);
    else { matmul_qt(g,x,wg,S); matmul_qt(u,x,wu,S); }
}

static int g_repin;
static uint64_t g_last_repin;
#ifdef COLI_VULKAN
static int g_vulkan;          /* COLI_VULKAN=1: compute routed experts on the Vulkan tier */
static int g_vk_budget;       /* COLI_VK_EXPERTS: pinned VK expert tier size (0 = tier off) */
static int g_vk_resident;     /* how many experts currently VK-resident */
/* Pinned VK expert tier: top heat-ranked routed experts uploaded ONCE at startup into a
 * registry keyed by (layer,eid), decoupled from the RAM cache slots — stable residency
 * (no LRU churn, no uploads on the decode critical path), mirroring the HIP VRAM tier. */
static struct ColiVkTensor **g_vk_reg;   /* [(layer*E+eid)*3 + {gate,up,down}] */
static int g_vk_reg_n, g_vk_reg_E;
static inline struct ColiVkTensor **vk_reg_at(int layer,int eid){
    return g_vk_reg ? &g_vk_reg[((size_t)layer*g_vk_reg_E+eid)*3] : NULL;
}
static int g_vk_reg_NL;                  /* registry layer bound (n_layers, MTP excluded) */
/* Will the VK tier serve this expert at decode? (prefetch/pilot can skip its I/O.) */
static inline int vk_reg_served(int layer,int eid){
    if(!g_vk_reg || layer<0 || layer>=g_vk_reg_NL || eid<0 || eid>=g_vk_reg_E) return 0;
    return g_vk_reg[((size_t)layer*g_vk_reg_E+eid)*3] != NULL;
}
static int g_vk_dense;        /* COLI_VK_DENSE=1: run the resident dense matmuls (attention
                               * projections + shared expert) on Vulkan too */
static int g_vk_budget2;      /* COLI_VK_EXPERTS2: dev2 expert-tier cap (with COLI_VK_DEV2) */
static int g_vk_reg_n2;       /* experts resident on the dev2 tier */
/* Resolve the main shader path (#523): COLI_VK_SHADERS may be the qmatmul.spv file itself
 * OR a directory containing it; unset, look alongside the binary (<exedir>/shaders/, the
 * build layout) before the historical CWD-relative fallback, so launching from outside c/
 * works. The other shaders load as siblings of the returned path (backend derive_*). */
static const char *vk_resolve_spv(char *buf, size_t n){
    const char *env = getenv("COLI_VK_SHADERS");
    struct stat st;
    if(env && *env){
        if(!stat(env,&st) && S_ISDIR(st.st_mode)){ snprintf(buf,n,"%s/qmatmul.spv",env); return buf; }
        return env;
    }
#ifdef __linux__
    ssize_t k = readlink("/proc/self/exe", buf, n-1);
    if(k > 0){
        buf[k] = 0;
        char *sl = strrchr(buf, '/');
        if(sl && (size_t)(sl+1-buf) + sizeof("shaders/qmatmul.spv") <= n){
            strcpy(sl+1, "shaders/qmatmul.spv");
            if(!stat(buf,&st)) return buf;
        }
    }
#endif
    return "shaders/qmatmul.spv";
}
/* PROF anatomy of the VK expert block (master-thread accumulated in moe(), printed by
 * profile_print): where a decode block's wall goes besides t_ecpu/t_ewait/t_egpu. */
static double g_vkb_cls, g_vkb_issue, g_vkb_acc, g_vkb_wrk, g_vkb_join;
static int64_t g_vkb_blocks, g_vkb_nvk, g_vkb_nvk2, g_vkb_ncpu;
static int g_vk_attn;         /* COLI_VK_ATTN=1: run the MLA absorb attention core on Vulkan
                               * (mirrors COLI_CUDA_ATTN; KV latent cache mirrored on-device) */
#endif
#ifdef COLI_CUDA
static int g_cuda_enabled;
static double g_cuda_expert_gb;
static int g_cuda_expert_auto;
static int g_cuda_dense;
static int g_cuda_release_host;
static double g_cuda_reserve_gb;   /* CUDA_RESERVE_GB: VRAM headroom kept free of expert tier (default 2 GB) */
static int g_cuda_raw_experts=-1;   /* experimental ANS tier: keep this global hot prefix raw */
static int g_cuda_devices[COLI_CUDA_MAX_DEVICES], g_cuda_ndev, g_cuda_rr;
static int64_t g_cuda_dense_projected[COLI_CUDA_MAX_DEVICES];
static void qt_cuda_reset(QT *t){
    if(t->cuda){ coli_cuda_tensor_free(t->cuda); t->cuda=NULL; }
    t->cuda_fail_s=0;
}
/* #687: a resident tensor that fails to upload falls back to CPU silently. The
 * per-tensor line below is easy to lose in a busy log, and the end state -- GPU fully
 * allocated, ~0% useful -- reads as healthy. Count them and say it once, loudly,
 * naming the knob that actually fixes it.
 *
 * Since #767 the fallback is per-width rather than permanent (see cuda_fail_s), so a
 * tensor counted here may recover on its own at a narrower S. The count is still worth
 * shouting about: it means the tier left too little VRAM for the prompt being run. */
static int g_cuda_disabled_n;
static void cuda_disabled_note(void){
    enum { CUDA_DISABLED_LOUD_AT = 8 };
    if(++g_cuda_disabled_n != CUDA_DISABLED_LOUD_AT) return;
    fprintf(stderr,
        "[CUDA] ***** %d resident tensors have fallen back to CPU *****\n"
        "[CUDA] The GPU still holds its expert tier, but the dense/attention path is\n"
        "[CUDA] running on CPU at this prompt width: expect at or below CPU speed while\n"
        "[CUDA] the card stays allocated. They are retried at narrower S, so decode may\n"
        "[CUDA] recover even when prefill does not. The usual cause is the expert tier\n"
        "[CUDA] claiming all VRAM and leaving nothing for the lazily-uploaded tensors.\n"
        "[CUDA] Fix: set an explicit CUDA_EXPERT_GB below the auto value (leave room\n"
        "[CUDA] per device for dense + attention + workspace). See issue #687.\n",
        CUDA_DISABLED_LOUD_AT);
}
static int g_cuda_e8_ready;   /* codebook published to the devices (see cuda_boot) */
static int g_cuda_fp8_ready;  /* e4m3 LUT published to the devices (see cuda_boot) */
#endif
#ifdef COLI_VULKAN
/* Drop a QT's Vulkan-resident copy (slot reused for a different expert). */
static void qt_vk_reset(QT *t){
    if(t->vk){ coli_vk_tensor_free(t->vk); t->vk=NULL; }
    t->vk_eligible=0;
}
/* Dense matmul on the Vulkan tier: y[S,O] = x[S,I] @ dequant(t)^T. Uploads the resident
 * int8/int4/int3-g64 weight once (t->vk), then reuses it. Returns 0 (caller runs the CPU
 * matmul) when VK-dense is off, the format is unsupported, or upload/compute fails. */
#define VK_FMT_OK(t) ((t)->fmt==1||(t)->fmt==2||(t)->fmt==5||((t)->fmt==4&&(t)->gs>=8&&(t)->gs%8==0))
static int vk_matmul_qt(QT *t, float *y, const float *x, int S){
    if(!g_vk_dense || !VK_FMT_OK(t)) return 0;
    const void *w = t->fmt==1 ? (const void*)t->q8 : (const void*)t->q4;
    return coli_vk_matmul(&t->vk, y, x, w, t->s, t->fmt, S, t->I, t->O, t->gs);
}
/* Two same-input resident matmuls in one submit (q_a + kv_a read the same x). */
static int vk_matmul_pair_qt(QT *a, float *ya, QT *b, float *yb, const float *x, int S){
    if(!g_vk_dense || a->fmt!=b->fmt || !VK_FMT_OK(a) || a->gs!=b->gs || a->I!=b->I) return 0;
    const void *wa = a->fmt==1 ? (const void*)a->q8 : (const void*)a->q4;
    const void *wb = b->fmt==1 ? (const void*)b->q8 : (const void*)b->q4;
    return coli_vk_matmul_pair(&a->vk, ya, wa, a->s, a->O,
                               &b->vk, yb, wb, b->s, b->O, a->fmt, x, S, a->I, a->gs);
}
#endif
#ifdef COLI_CUDA
static int qt_cuda_upload(QT *t){
    if(t->fmt==5) return 0;   /* int3-g64: no CUDA kernel yet — tensor stays CPU-side */
    if(t->fmt==6 && !g_cuda_e8_ready) return 0;   /* E8 without its codebook would decode garbage */
    if(t->fmt==8 && !g_cuda_fp8_ready) return 0;  /* same idiom: fp8 without its e4m3 LUT stays
                                                   * CPU-side (an old DLL without the symbol
                                                   * leaves the flag 0, exactly like fmt=6) */
    const void *weights = t->fmt==0 ? (const void*)t->qf
                        : (t->fmt==1||t->fmt==8) ? (const void*)t->q8 : (const void*)t->q4;
    if(t->fmt==4)   /* grouped int4 (#334): scales are [O, ceil(I/gs)] — the plain
                     * upload would truncate them to O floats and the group kernels
                     * would read garbage. An old DLL without the _g symbol returns 0
                     * and the tensor simply stays CPU-side. */
        return coli_cuda_tensor_upload_g(&t->cuda,weights,t->s,t->fmt,t->I,t->O,t->cuda_device,t->gs);
    return coli_cuda_tensor_upload(&t->cuda,weights,t->s,t->fmt,t->I,t->O,t->cuda_device);
}
#ifdef COLI_ANS
static int qt_cuda_upload_compressed(QT *t){
    if(t->fmt!=2) return 0;
    return coli_cuda_tensor_upload_compressed(&t->cuda,t->q4,t->s,t->fmt,t->I,t->O,t->cuda_device);
}
#endif
static int qt_cuda_update(QT *t){
    const void *weights=t->fmt==0?(const void*)t->qf:
                        (t->fmt==1||t->fmt==8)?(const void*)t->q8:(const void*)t->q4;
    return coli_cuda_tensor_update(t->cuda,weights,t->s);
}
static double g_ovl_issue,g_ovl_cpu,g_ovl_take,g_ovl_mark; /* Inc.4 overlap-window split (OVL report) */
static void cuda_stats_print(void){
    size_t n=0,b=0; coli_cuda_stats(-1,&n,&b);
    fprintf(stderr,"[CUDA] resident set: %zu tensors, %.2f GB VRAM\n",n,b/1e9);
    /* #687: say it again at the end -- by now the per-tensor lines are thousands of
     * log lines back, and this is the number that explains a CPU-speed "GPU" run. */
    if(g_cuda_disabled_n) fprintf(stderr,
        "[CUDA] %d tensors fell back to CPU after failed uploads at least once. Lower "
        "CUDA_EXPERT_GB to leave room for them (#687, #767).\n", g_cuda_disabled_n);
    if(g_cuda_ndev>1) for(int i=0;i<g_cuda_ndev;i++){
        coli_cuda_stats(g_cuda_devices[i],&n,&b);
        fprintf(stderr,"[CUDA]   device %d: %zu tensors, %.2f GB\n",g_cuda_devices[i],n,b/1e9);
    }
    uint64_t calls=0,experts=0,rows=0; double h2d=0,kernel=0,d2h=0;
    coli_cuda_group_stats(&calls,&experts,&rows,&h2d,&kernel,&d2h);
    if(calls) fprintf(stderr,"[CUDA] expert groups: %llu call, %llu expert, %llu righe "
        "(%.2f expert/call)%s\n",(unsigned long long)calls,(unsigned long long)experts,
        (unsigned long long)rows,(double)experts/calls,
        getenv("COLI_CUDA_PROFILE")?"; timing sotto":"");
    if(calls&&g_cuda_ndev>1) for(int i=0;i<g_cuda_ndev;i++){
        uint64_t dc=0,de=0,dr=0;
        coli_cuda_group_stats_device(g_cuda_devices[i],&dc,&de,&dr,NULL,NULL,NULL);
        fprintf(stderr,"[CUDA]   device %d groups: %llu call, %llu expert, %llu rows "
            "(%.2f expert/call)\n",g_cuda_devices[i],(unsigned long long)dc,
            (unsigned long long)de,(unsigned long long)dr,dc?(double)de/dc:0.0);
    }
    if(calls&&getenv("COLI_CUDA_PROFILE")) fprintf(stderr,
        "[CUDA] expert groups timing: H2D %.1f ms | kernel %.1f ms | D2H %.1f ms\n",h2d,kernel,d2h);
    if(g_ovl_issue+g_ovl_cpu+g_ovl_take>0) fprintf(stderr,
        "[CUDA] overlap window: pack+issue %.2fs | cpu-rows %.2fs | take(sync+acc) %.2fs\n",
        g_ovl_issue,g_ovl_cpu,g_ovl_take);
}
static int parse_cuda_devices(const char *list, int *out){
    if(!list||!*list) return 0;
    int n=0; const char *p=list;
    while(*p){
        char *end=NULL; long v=strtol(p,&end,10);
        if(end==p||v<0||v>INT_MAX||n>=COLI_CUDA_MAX_DEVICES) return 0;
        for(int i=0;i<n;i++) if(out[i]==(int)v) return 0;
        out[n++]=(int)v; p=end;
        while(*p==' '||*p=='\t') p++;
        if(!*p) break;
        if(*p++!=',') return 0;
        while(*p==' '||*p=='\t') p++;
        if(!*p) return 0;
    }
    return n;
}
#endif
static double now_s(void){
#ifdef _WIN32
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart / (double)freq.QuadPart;
#else
    struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9;
#endif
}
static double rss_gb(void){
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS_EX pmc = {0}; pmc.cb = sizeof(pmc);
    if(GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc)))
        return pmc.WorkingSetSize / (1024.0 * 1024.0 * 1024.0);
    return 0;
#else
    struct rusage r; getrusage(RUSAGE_SELF,&r);
#ifdef __APPLE__
    return r.ru_maxrss/(1024.0*1024.0*1024.0);   /* macOS: ru_maxrss in BYTE */
#else
    return r.ru_maxrss/(1024.0*1024.0);          /* Linux: in KB */
#endif
#endif
}
#ifdef __linux__
static double current_rss_gb(void) {
  FILE *f = fopen("/proc/self/status", "r");
  static int announced_proc_failure;
  if (!f) {
    if(!announced_proc_failure) {
      announced_proc_failure=1;
      fprintf(stderr, "[RSS] failed to open /proc/self/status, cannot measure current memory usage. Falling back to peak memory usage measurement.\n");
    }
    return rss_gb();  /*Return peak memory usage if we can't measure current usage*/
  }

  char line[256];
  unsigned long long kb = 0;

  while (fgets(line, sizeof(line), f)) {
    if (strncmp(line, "VmRSS:", 6) == 0) {
      if (sscanf(line + 6, "%llu", &kb) == 1) {
        fclose(f);
        return (double)kb / (1024.0 * 1024.0);
      }
    }
  }

  fclose(f);
  if(!announced_proc_failure) {
    announced_proc_failure=1;
    fprintf(stderr, "[RSS] failed to find VmRSS in /proc/self/status, cannot measure current memory usage. Falling back to peak memory usage measurement.\n");
  }
  return rss_gb();   /*Return peak memory usage if we can't measure current usage*/
}
#endif
/* ---- PROF=1: opt-in performance profile ----------------------------------
 * Records per-forward decode latency and expert-file bytes fetched, then
 * reports percentiles, I/O totals, phase shares and a tuning verdict next to
 * the existing PROFILE line. Additive only: with PROF unset the output of
 * every mode stays byte-identical. */
static int g_prof=0;
static _Atomic int64_t g_prof_io;                /* bytes pread()/faulted from expert files */
/* Disk service: wall time inside expert_load on whichever thread runs the read
 * (PIPE I/O workers, OMP loaders, the speculative pilot). It overlaps compute,
 * so it is NOT a wall-time phase — the stall the compute thread actually felt
 * is m->t_ewait. Thread-seconds, so it can exceed wall time under parallel
 * reads; wait << service means overlap/parallelism is hiding the reads,
 * wait ~ service means the loads block the compute thread. */
static _Atomic int64_t g_edisk_ns;
static double edisk_s(void){ return atomic_load_explicit(&g_edisk_ns,memory_order_relaxed)*1e-9; }
/* DISK-CLASS (PROF=1): per-load cold/warm classification against the engine's own
 * recency state. Instrumentation only -- it never changes which fd serves a read (see
 * expert_classify() and its call site in expert_load_impl; the fd choice expression is
 * untouched by this feature). COLI_DISKCLASS_WINDOW is the recency window in ticks of the
 * PRIVATE clock (m->eaccess_clock_dc -- NOT the real eaccess_clock; see elast_dc in Model
 * for why DISK-CLASS keeps its own clock instead of reading the real one): one tick per
 * selected (position,expert) in FASE A while classification is active, the same per-token
 * rate the real clock has on the CPU path, so the window's meaning is unchanged. At or
 * under the window = warm; 0 (default, unset) derives it from topk*n_layers*8 once the
 * model config is known (main(), right after model_init) -- roughly "seen in the last ~8
 * tokens", generous on purpose (conservative-toward-warm: a load the page cache could have
 * served that gets labeled cold overstates the cold class, the bucket this line exists to
 * size). */
static uint32_t g_direct_heat_ticks=0;
static int g_direct_heat_explicit=0;    /* 1 if COLI_DISKCLASS_WINDOW was set (skip the auto-derive) */
#define DC_COLD 0
#define DC_WARM 1
static _Atomic uint64_t g_dc_n[2];              /* [DC_COLD]/[DC_WARM]: loads classified */
static _Atomic int64_t  g_dc_bytes[2];          /* bytes read (weights + scales, matches g_prof_io) */
static _Atomic int64_t  g_dc_ns[2];             /* wall ns spent reading (thread-seconds, like g_edisk_ns) */
static _Atomic uint64_t g_dc_direct_n[2];       /* subset of the above ACTUALLY served by the uncached fd */
/* Busy-wall per class + combined: how much WALL time had >=1 classified load of the
 * class in flight (thread-seconds / busy-wall = average concurrency; bytes / busy-wall
 * = aggregate GB/s the disk actually delivered for that class -- the quantity the
 * thread-second numbers alone can't answer: N slow overlapped reads can beat N fast
 * serial ones in aggregate, and only wall-denominated rates see it). Transition scheme:
 * 0->1 records a start, 1->0 accumulates (now - start). One dedicated mutex serializes
 * the transition bookkeeping -- two short lock/unlock pairs per load against ms-scale
 * reads; a CAS scheme would save nothing measurable and be harder to audit
 * (correctness over cleverness). Only COMPLETED intervals are in the accumulators: an
 * interval still open at report time is not counted (bounded by one read's duration --
 * noise at report granularity). */
static pthread_mutex_t g_dc_wall_mx=PTHREAD_MUTEX_INITIALIZER;
static int g_dc_inflight[2], g_dc_inflight_all;              /* guarded by g_dc_wall_mx */
static double g_dc_wall_open[2], g_dc_wall_open_all;         /* start of the open interval */
static int64_t g_dc_wall_ns[2], g_dc_wall_all_ns;            /* completed busy-wall ns */
static void dc_wall_enter(int cls, double now){
    pthread_mutex_lock(&g_dc_wall_mx);
    if(g_dc_inflight[cls]++==0) g_dc_wall_open[cls]=now;
    if(g_dc_inflight_all++==0)  g_dc_wall_open_all=now;
    pthread_mutex_unlock(&g_dc_wall_mx);
}
static void dc_wall_exit(int cls, double now){
    pthread_mutex_lock(&g_dc_wall_mx);
    if(--g_dc_inflight[cls]==0) g_dc_wall_ns[cls]+=(int64_t)((now-g_dc_wall_open[cls])*1e9);
    if(--g_dc_inflight_all==0)  g_dc_wall_all_ns +=(int64_t)((now-g_dc_wall_open_all)*1e9);
    pthread_mutex_unlock(&g_dc_wall_mx);
}
static int dc_needed(void);                                  /* fwd: defined with the classifier (needs g_prof) */
static void dc_wall_read(int64_t out[2], int64_t *all){      /* mutex-consistent snapshot for the report */
    if(!dc_needed()){ out[0]=out[1]=0; *all=0; return; }     /* off for the whole process => accumulators are
                                                              * provably zero (only dc_wall_exit writes them,
                                                              * only under dc_on): skip the lock, zero work.
                                                              * Needed because prof_base runs unconditionally
                                                              * at some call sites ("cheap enough to always"). */
    pthread_mutex_lock(&g_dc_wall_mx);
    out[0]=g_dc_wall_ns[0]; out[1]=g_dc_wall_ns[1]; *all=g_dc_wall_all_ns;
    pthread_mutex_unlock(&g_dc_wall_mx);
}
#define PROF_LAT_CAP 32768
static double g_prof_lat[PROF_LAT_CAP];          /* per-forward decode wall clock (ring) */
static uint64_t g_prof_nlat;                     /* forwards recorded (monotonic) */
static void prof_lat(double s){ g_prof_lat[g_prof_nlat++ % PROF_LAT_CAP]=s; }
/* snapshot for windowed reports (serve mode: one report per turn) */
typedef struct {
    double edisk,ewait,emm,ecpu,egpu,route,p2p,attn,head;
    int64_t io,cpu_bytes; uint64_t hits,miss,ereq,n_fw,n_emit,nlat,n_p2p,cpu_rows;
    uint64_t hit_pin,hit_ecache;
    uint64_t dc_n[2], dc_direct_n[2]; int64_t dc_bytes[2], dc_ns[2]; /* DISK-CLASS */
    int64_t dc_wall_ns[2], dc_wall_all_ns;       /* busy-wall (per class + combined) */
} ProfBase;
static void prof_base(Model *m, ProfBase *b){
    b->edisk=edisk_s(); b->ewait=m->t_ewait; b->emm=m->t_emm;
    b->ecpu=m->t_ecpu; b->egpu=m->t_egpu; b->route=m->t_route; b->p2p=m->t_p2p;
    b->attn=m->t_attn; b->head=m->t_head;
    b->io=atomic_load_explicit(&g_prof_io,memory_order_relaxed);
    b->hits=m->hits; b->miss=m->miss; b->ereq=m->ereq;
    b->hit_pin=m->hit_pin; b->hit_ecache=m->hit_ecache;
    b->n_fw=m->n_fw; b->n_emit=m->n_emit; b->nlat=g_prof_nlat; b->n_p2p=m->n_p2p;
    b->cpu_bytes=m->cpu_expert_bytes;b->cpu_rows=m->cpu_expert_rows;
    for(int i=0;i<2;i++){
        b->dc_n[i]=atomic_load_explicit(&g_dc_n[i],memory_order_relaxed);
        b->dc_bytes[i]=atomic_load_explicit(&g_dc_bytes[i],memory_order_relaxed);
        b->dc_ns[i]=atomic_load_explicit(&g_dc_ns[i],memory_order_relaxed);
        b->dc_direct_n[i]=atomic_load_explicit(&g_dc_direct_n[i],memory_order_relaxed);
    }
    dc_wall_read(b->dc_wall_ns,&b->dc_wall_all_ns);
}

static float *falloc(int64_t n){
    /* guardia anti-wrap (report PR #25): n assurdo da file modello ostili non deve
     * diventare una malloc piccola. Niente calloc: il memset nel percorso caldo costa. */
    if(n<0 || (uint64_t)n > SIZE_MAX/sizeof(float)){ fprintf(stderr,"falloc: n=%lld is out of range\n",(long long)n); exit(1); }
    float *p=malloc((size_t)n*sizeof(float)); if(!p){fprintf(stderr,"OOM\n");exit(1);} return p; }

/* Come falloc, per i buffer non-float del percorso caldo. moe() e' la funzione piu'
 * chiamata del motore e girare al soffitto di RAM e' la premessa del progetto: e'
 * esattamente la condizione in cui malloc torna NULL. Un deref di NULL li' e' un
 * segfault a meta' generazione senza diagnostica, indistinguibile da un bug vero
 * quando l'utente lo riporta. exit(1) con messaggio e' la stessa convenzione di
 * falloc e dello scratch xexp. */
static void *xalloc(size_t n, const char *what){
    if(n==0) n=1;
    void *p=malloc(n);
    if(!p){ fprintf(stderr,"OOM: %s (%zu byte)\n",what,n); exit(1); }
    return p;
}
static void *xzalloc(size_t n, const char *what){
    if(n==0) n=1;
    void *p=calloc(n,1);
    if(!p){ fprintf(stderr,"OOM: %s (%zu byte)\n",what,n); exit(1); }
    return p;
}



/* Fused gate+up for grouped int4 (fmt=4): computes both yg[S,O] and yu[S,O] from
 * the same x[S,I], reading x once instead of twice — saves ~33% of expert-matmul time at decode.
 * The per-group scale logic matches matmul_i4_grouped exactly. */
static void matmul_i4_grouped_pair(float *yg, float *yu, const float *x,
                                    const uint8_t *qg, const float *sg,
                                    const uint8_t *qu, const float *su,
                                    int S, int I, int O, int gs){
    int rb=(I+1)/2; int ng=(I+gs-1)/gs;
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){
        const uint8_t *wg=qg+(int64_t)o*rb; const uint8_t *wu2=qu+(int64_t)o*rb;
        const float *sgl=sg+(int64_t)o*ng;   const float *sul=su+(int64_t)o*ng;
        for(int s=0;s<S;s++){
            const float *xs=x+(int64_t)s*I;
            float ag=0, au=0;
            for(int g=0; g*gs<I; g++){
                int base=g*gs; int glen=gs; if(base+glen>I) glen=I-base;
                float scg=sgl[g], scu=sul[g];
                int i=base;
#ifdef __AVX2__
                const __m128i m4=_mm_set1_epi8(0x0F); const __m256i b8=_mm256_set1_epi32(8);
                __m256 accg=_mm256_setzero_ps(), accu=_mm256_setzero_ps();
                for(; i+16<=base+glen; i+=16){
                    __m128i byg=_mm_loadl_epi64((const __m128i*)(wg+(i>>1)));
                    __m128i log=_mm_and_si128(byg,m4),hig=_mm_and_si128(_mm_srli_epi16(byg,4),m4);
                    __m128i nibg=_mm_unpacklo_epi8(log,hig);
                    __m256 w0g=_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(nibg),b8));
                    __m256 w1g=_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_srli_si128(nibg,8)),b8));
                    accg=_mm256_fmadd_ps(_mm256_loadu_ps(xs+i),   w0g, accg);
                    accg=_mm256_fmadd_ps(_mm256_loadu_ps(xs+i+8), w1g, accg);
                    __m128i byu=_mm_loadl_epi64((const __m128i*)(wu2+(i>>1)));
                    __m128i lou=_mm_and_si128(byu,m4),hiu=_mm_and_si128(_mm_srli_epi16(byu,4),m4);
                    __m128i nibu=_mm_unpacklo_epi8(lou,hiu);
                    __m256 w0u=_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(nibu),b8));
                    __m256 w1u=_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_srli_si128(nibu,8)),b8));
                    accu=_mm256_fmadd_ps(_mm256_loadu_ps(xs+i),   w0u, accu);
                    accu=_mm256_fmadd_ps(_mm256_loadu_ps(xs+i+8), w1u, accu);
                }
                /* Same pinning as matmul_i4_grouped in quant.h: contraction
                 * here is the compiler's choice, so the result was too. */
                ag=fmaf(hsum256(accg),scg,ag); au=fmaf(hsum256(accu),scu,au);
#endif
                for(; i+1<base+glen; i+=2){
                    uint8_t bg=wg[i>>1], bu=wu2[i>>1];
                    ag+=(xs[i]*(float)((int)(bg&0xF)-8)+xs[i+1]*(float)((int)(bg>>4)-8))*scg;
                    au+=(xs[i]*(float)((int)(bu&0xF)-8)+xs[i+1]*(float)((int)(bu>>4)-8))*scu;
                }
                if(i<base+glen){
                    uint8_t bg=wg[i>>1], bu=wu2[i>>1];
                    ag+=xs[i]*(float)((int)(bg&0xF)-8)*scg;
                    au+=xs[i]*(float)((int)(bu&0xF)-8)*scu;
                }
            }
            yg[(int64_t)s*O+o]=ag; yu[(int64_t)s*O+o]=au;
        }
    }
}

/* ---- K1: layout int4 a piani, gate e repack (vedi quant.h) -----------------
 * Planarizziamo SOLO i tensori MoE (slab expert + trio MLP denso + shared):
 * i loro consumatori sono esattamente matmul_qt/expert_gate_up, tutti
 * planar-aware. kv_b (letto raw dal path DSA fuso), embedding (dequant
 * per-token) e attenzione restano a coppie PER COSTRUZIONE. Off su build GPU
 * (i backend leggono q4 a coppie), sotto XEXP (dot_i4i8 sulle slab) e su
 * build AVX-512F (il ramo f32 a 512 bit accumula in altro ordine).
 * EN: K1 planar gate. MoE tensors only; GPU builds, XEXP and AVX-512F builds
 * keep the classic pair layout. PLANAR=0 is the kill switch. */
static int g_planar=-1;
static int planar_on(void){
    if(g_planar<0){
#if defined(COLI_CUDA)||defined(COLI_METAL)||defined(COLI_VULKAN)
        g_planar=0;
#elif defined(__AVX512F__)&&defined(__AVX512BW__)
        g_planar=0;
#elif !defined(__AVX2__)
        g_planar=0;   /* matmul_i4p non ha (ancora) un ramo NEON: su ARM il path
                       * f32 planare degraderebbe a scalare E cambierebbe l'ordine
                       * di accumulo rispetto al gemello NEON a coppie — il claim
                       * bit-identico vale solo dove i gemelli esistono entrambi.
                       * EN: no NEON arm in matmul_i4p yet — planar stays off on
                       * non-AVX2 builds until one lands with matching order. */
#else
        const char *e=getenv("PLANAR"); g_planar=!(e&&*e=='0');
        const char *xe=getenv("XEXP"); if(xe&&*xe=='1') g_planar=0;
#endif
    }
    return g_planar;
}
static _Atomic long g_planar_n;   /* tensori planarizzati (telemetria una-tantum) */
/* K1b: IDOT a gruppi per fmt=4 (gs%64==0) — OPT-IN, cambia le numeriche
 * (attivazioni int8): resta 0 finche' l'ablazione non benedice un default.
 * EN: opt-in grouped IDOT for fmt=4; int8 activations, awaiting ablation. */
static int g_idot_gs=-1;
static int idot_gs_on(void){
    if(g_idot_gs<0){ const char *e=getenv("IDOT_GS"); g_idot_gs=(e&&atoi(e))?1:0;
        if(g_idot_gs&&!planar_on()) g_idot_gs=0;   /* richiede la famiglia planare */
        if(g_idot_gs) fprintf(stderr,"[K1b] grouped planar IDOT active for gs64 tensors (opt-in)\n"); }
    return g_idot_gs;
}
static void qt_planarize(QT *t){
    if(!planar_on()||t->planar||!t->q4) return;
    if(t->fmt==4){
        if(!idot_gs_on()||t->gs<64||t->gs%64) return;   /* K1b: solo su opt-in */
        planarize_i4(t->q4,t->O,t->I); t->planar=1; return;
    }
    if(t->fmt!=2) return;
    planarize_i4(t->q4,t->O,t->I); t->planar=1;
    if(atomic_fetch_add_explicit(&g_planar_n,1,memory_order_relaxed)==0)
        fprintf(stderr,"[K1] planar int4 layout active (PLANAR=0 disables)\n");
}

/* ---- quantizzazione attivazioni sollevata a livello layer ---------------------------
 * moe() quantizza x UNA volta per layer e raccoglie le righe int8 accanto al gather
 * f32; il ramo IDOT di matmul_qt_ex le riusa quando (x,S,I) coincidono ESATTAMENTE
 * col contesto. Prima lo stesso vettore veniva riquantizzato da ogni matmul che lo
 * consumava (~16x per layer: 8 expert x gate+up), in seriale sul thread master.
 * Bit-identico: qrow_i8 e' pura, quantizzare la riga originale o la sua copia
 * raccolta produce gli stessi byte; l'ordine per-riga non cambia. Ogni altro
 * percorso (CUDA/Metal/Vulkan, fmt 0/3/4/5/6/8, attenzione con allow_idot=0, la
 * copia ruotata fmt=6) ignora il contesto e resta identico.
 * EN: layer-level hoisted activation quantization. moe() quantizes x once and
 * gathers int8 rows next to the f32 gather; the IDOT branch reuses them on an
 * exact (x,S,I) pointer/shape match. Bit-identical (qrow_i8 is a pure function
 * of the row bytes). Every other path ignores the context and is unchanged. */
static struct { const float *x; int S, I; const int8_t *xq; const float *sx; const int32_t *xsum; } g_pq;
static void pq_build(const float *x, int S, int D, int8_t *xq, float *sx){
    /* righe indipendenti -> parallelizzare sulle righe e' bit-identico; il loop
     * DENTRO qrow_i8 resta scalare (stesso ordine di lrintf di sempre).
     * EN: rows are independent, so parallelizing across rows is bit-identical;
     * the per-row loop inside qrow_i8 stays scalar (same lrintf order). */
    #pragma omp parallel for schedule(static) if(S>=8)
    for(int s=0;s<S;s++) sx[s]=qrow_i8(x+(int64_t)s*D, xq+(int64_t)s*D, D);
}
/* somma int32 per riga degli int8 gia' quantizzati: il termine -8*sum(x) del
 * dot planare unsigned (K1). EN: per-row int32 sums of the quantized rows. */
static void pq_build_xsum(const int8_t *xq, int S, int D, int32_t *xsum){
    #pragma omp parallel for schedule(static) if(S>=8)
    for(int s=0;s<S;s++){ int32_t t=0; const int8_t *r=xq+(int64_t)s*D;
        for(int i=0;i<D;i++) t+=r[i]; xsum[s]=t; }
}
/* Il gate/up di questo expert passerebbe dall'IDOT di matmul_qt_ex? Specchia
 * l'ordine di dispatch di expert_gate_up: a S==1 la fused pair f32 (fmt=2)
 * preempta matmul_qt, quindi li' il contesto non servirebbe a nessuno.
 * EN: would this expert's gate/up take matmul_qt_ex's IDOT branch? Mirrors
 * expert_gate_up's dispatch order: at S==1 the f32 fused pair (fmt=2) preempts
 * matmul_qt entirely, so the context would go unused there. */
static inline int pq_want(const QT *g,const QT *u,int nr){
    if(!g_idot) return 0;
    if(nr==1 && !g_no_fused_pair && !spec_pinned() &&
       g->fmt==2 && u->fmt==2 && g->I==u->I && g->O==u->O) return 0;
    int a = g->fmt==1 || (g->fmt==2 && (spec_pinned() ? g_i4s<=1 : nr>=g_i4s)) || (g->fmt==4 && g->planar);
    int b = u->fmt==1 || (u->fmt==2 && (spec_pinned() ? g_i4s<=1 : nr>=g_i4s)) || (u->fmt==4 && u->planar);
    return a || b;
}

/* allow_idot=0: forza il kernel int4/int8 ESATTO (attivazioni f32). Serve alle proiezioni di
 * attenzione: sono sensibili alla quantizzazione int8 delle attivazioni dell'IDOT. Misurato su
 * GLM-5.2 int4, 1023 token, log-lik -5040.33 (esatto) -> -5160.47 (IDOT) = +0.117 nat/token,
 * ~+12% perplexity. Gli altri matmul del prefill (o_proj, kv_b, expert) tengono l'IDOT.
 * EN: allow_idot=0 forces the EXACT int4/int8 kernel (f32 activations). The attention
 * projections need it: IDOT's int8 activation quantization costs +0.117 nats/token there
 * (~+12% perplexity), measured. Every other prefill matmul keeps IDOT as before. */
static void matmul_qt_ex(float *y, const float *x, QT *w, int S, int allow_idot){
#ifdef COLI_METAL
    /* fmt=8 (fp8 passthrough) deliberately absent from this allowlist, same as fmt=5/6:
     * the S>=g_metal_gemm_min batched prefill GEMM path (coli_metal_gemm) only knows
     * fmt 1/2/4 in this build, so it fails CLOSED to the CPU branch below (matmul_fp8)
     * rather than being silently misread. coli_metal_gemm() also carries its own
     * internal fmt!=1&&fmt!=2&&fmt!=4 guard, so this is belt-and-braces. */
    if(g_metal_enabled && S>=g_metal_gemm_min && !spec_pinned() && (w->fmt==1||w->fmt==2||(w->fmt==4&&!g_moe_exact)) && !omp_in_parallel()){
        const void *wp = w->fmt==1 ? (const void*)w->q8 : (const void*)w->q4;
        if(coli_metal_gemm(y,x,wp,w->s,w->fmt,S,w->I,w->O,w->gs)) return;
    }
#endif
#ifdef COLI_CUDA
    /* fmt=5 excluded like fmt=6-without-codebook: the CUDA backend has no int3
     * kernel. fmt=8 flows once its e4m3 LUT is published (g_cuda_fp8_ready,
     * cuda_boot): quant_matmul dispatches it by format, deriving the 128x128
     * block-scale geometry from the dims alone. Without the LUT (old DLL) it is
     * excluded up front — same silent-and-immediate idiom as fmt=5. */
    if(g_cuda_enabled && w->cuda_eligible && (!w->cuda_fail_s || S < w->cuda_fail_s) &&
       w->fmt!=5 && (w->fmt!=8 || g_cuda_fp8_ready) && !omp_in_parallel()){
        const void *weights = w->fmt==0 ? (const void*)w->qf
                            : (w->fmt==1||w->fmt==8) ? (const void*)w->q8 : (const void*)w->q4;
        if(coli_cuda_matmul(&w->cuda,y,x,weights,w->s,w->fmt,S,w->I,w->O,w->cuda_device,w->gs)){
            w->cuda_fail_s=0;          /* it fits again: forget the width that did not */
            return;
        }
        w->cuda_fail_s = S;
        if(g_cuda_disabled_n < 8)      /* keep the detail for the first few, then the summary */
            fprintf(stderr,"[CUDA] tensor [%d,%d] on device %d fell back to CPU at S=%d; "
                "narrower calls will still try the GPU\n", w->O,w->I,w->cuda_device,S);
        cuda_disabled_note();
    }
#endif
    if(w->fmt==0){ matmul(y,x,w->qf,S,w->I,w->O); return; }
    if(w->fmt==4){
        if(w->planar){
            /* K1b: quantizza le attivazioni (o riusa il contesto g_pq) e calcola le
             * somme per (riga, gruppo) — il termine -8*sum del dot unsigned.
             * EN: grouped planar IDOT — int8 activations + per-(row,group) sums. */
            int I=w->I, ng=(I+w->gs-1)/w->gs; int8_t *xq; float *sx;
            if(S<0 || I<0 || (size_t)S>SIZE_MAX/(size_t)(I?I:1)){ fprintf(stderr,"matmul_qt: shape overflow\n"); exit(1); }
            if(g_pq.x==x && g_pq.S==S && g_pq.I==I){ xq=(int8_t*)g_pq.xq; sx=(float*)g_pq.sx; }
            else {
                quant_scratch((size_t)S*I,(size_t)S,&xq,&sx);
                for(int s=0;s<S;s++) sx[s]=qrow_i8(x+(int64_t)s*I, xq+(int64_t)s*I, I);
            }
            static _Thread_local int32_t *xsg; static _Thread_local size_t xsg_cap;
            if((size_t)S*ng>xsg_cap){ free(xsg); xsg=malloc((size_t)S*ng*sizeof(int32_t));
                xsg_cap=xsg?(size_t)S*ng:0; if(!xsg){ fprintf(stderr,"OOM xsg\n"); exit(1); } }
            for(int s=0;s<S;s++) for(int g=0;g<ng;g++){
                int base=g*w->gs, end=base+w->gs; if(end>I) end=I;
                int32_t t2=0; const int8_t *r2=xq+(int64_t)s*I;
                for(int i=base;i<end;i++) t2+=r2[i];
                xsg[(int64_t)s*ng+g]=t2;
            }
            matmul_i4p_grouped_idot(y,xq,sx,xsg,w->q4,w->s,S,I,w->O,w->gs);
            return;
        }
        matmul_i4_grouped(y,x,w->q4,w->s,S,w->I,w->O,w->gs); return;
    }
    if(w->fmt==6){ matmul_e8(y,x,w->q4,NULL,S,w->I,w->O); return; }   /* scales live in-block */
    if(allow_idot && g_idot && (w->fmt==1 || (w->fmt==2 && (spec_pinned() ? g_i4s<=1 : S>=g_i4s)))){
        int I=w->I; int8_t *xq; float *sx;
        if(g_pq.x==x && g_pq.S==S && g_pq.I==I){
            /* righe gia' quantizzate una volta dal layer (vedi g_pq sopra)
             * EN: rows already quantized once at layer level (see g_pq above) */
            xq=(int8_t*)g_pq.xq; sx=(float*)g_pq.sx;
        } else {
            if(S<0 || I<0 || (size_t)S>SIZE_MAX/(size_t)(I?I:1)){ fprintf(stderr,"matmul_qt: shape overflow\n"); exit(1); }
            quant_scratch((size_t)S*I,(size_t)S,&xq,&sx);
            for(int s=0;s<S;s++) sx[s]=qrow_i8(x+(int64_t)s*I, xq+(int64_t)s*I, I);
        }
        if(w->fmt==1) matmul_q_idot(y,xq,sx,w->q8,w->s,S,I,w->O);
        else if(w->planar){
            /* K1: kernel planare unsigned; il termine -8*sum(x) arriva dal
             * contesto (g_pq, gia' raccolto per riga) o si calcola qui.
             * EN: planar unsigned kernel; -8*sum(x) from g_pq or computed. */
            const int32_t *xs32 = (g_pq.x==x && g_pq.S==S && g_pq.I==I) ? g_pq.xsum : NULL;
            static _Thread_local int32_t *xsb; static _Thread_local size_t xsb_cap;
            if(!xs32){
                if((size_t)S>xsb_cap){ free(xsb); xsb=malloc((size_t)S*sizeof(int32_t));
                    xsb_cap=xsb?(size_t)S:0;
                    if(!xsb){ fprintf(stderr,"OOM planar xsum\n"); exit(1); } }
                for(int s2=0;s2<S;s2++){ int32_t t2=0; const int8_t *r2=xq+(int64_t)s2*I;
                    for(int i2=0;i2<I;i2++) t2+=r2[i2]; xsb[s2]=t2; }
                xs32=xsb;
            }
            matmul_i4p_idot(y,xq,sx,xs32,w->q4,w->s,S,I,w->O);
        }
        else matmul_i4_idot(y,xq,sx,w->q4,w->s,S,I,w->O);
        return;
    }
    if(w->fmt==1) matmul_q(y,x,w->q8,w->s,S,w->I,w->O);
    else if(w->fmt==3) matmul_i2(y,x,w->q4,w->s,S,w->I,w->O);
    else if(w->fmt==5) matmul_i3(y,x,w->q4,w->s,S,w->I,w->O);
    else if(w->fmt==8) matmul_fp8(y,x,(const uint8_t*)w->q8,w->s,S,w->I,w->O);
    else if(w->planar) matmul_i4p(y,x,w->q4,w->s,S,w->I,w->O);
    else matmul_i4(y,x,w->q4,w->s,S,w->I,w->O);
}

static int g_nopack=0;   /* NOPACK=1 -> tiene i valori <=4bit in contenitore int8 (per validare il packing) */
static int g_drop=0;     /* DROP=1 -> scarta le pagine expart dopo l'uso. Default 0: le lascia in
                          * page-cache (buff/cache, NON RSS) come L2 gratuito -> sfrutta lo
                          * sbilanciamento del routing MoE (pochi expert "caldi" riusati). */
static int g_prefetch=0; /* PREFETCH=1 -> riabilita il WILLNEED cross-layer (metodo C). Default
                          * OFF: i load VERI in parallelo lo hanno reso superfluo, e sotto
                          * pressione di memoria il readahead speculativo veniva rievictato. */
static int g_direct=0;   /* DIRECT=1 -> O_DIRECT sugli slab expert. Default OFF: su questo host
                          * (VHDX su NVMe DRAM-less, latenza serializzata ~60ms/req) il buffered
                          * liscio e' risultato il migliore; su NVMe veri DIRECT=1 rende di piu'. */
static float g_temp=-1;  /* COLI_TEMP: temperatura di sampling sui TOKEN. <0 = auto (1.0 in chat/
                          * testo, 0=greedy in validazione). 0 = greedy puro. */

/* #509: COLI_TEMP e' il canale primario; TEMP resta come alias legacy ma SOLO se
 * interamente numerica. $TEMP e' la directory temporanea per ROCm (comgr/MIOpen) e
 * per Windows: un valore tipo 0.6 fa fallire l'init HIP, e sulle build native
 * Windows atof("C:\...\Temp")==0 avrebbe silenziosamente forzato greedy sempre. */
static float temp_from_env(const char *coli_temp, const char *temp){
    if(coli_temp) return (float)atof(coli_temp);
    if(temp && *temp){ char *tend; double tv=strtod(temp,&tend);
                       if(tend!=temp && *tend=='\0') return (float)tv; }
    return -1.f;
}
static float g_nuc=0.95f;/* NUCLEUS: top-p sul vocabolario (default dal generation_config GLM-5.2) */
static int g_topk=0;     /* TOPK=n -> usa n expert/token invece di config (ricerca: meno disco) */
static float g_topp=0;   /* TOPP=p (0..1) -> top-p adattivo: tieni gli expert fino a peso cumulato p */
static Abl g_abl = {0};  /* per-expert causal-ablation config (per item). mode==0 -> the three moe()
                          * hooks below are inert and output stays byte-identical to upstream.
                          * Set only on the ABLATE_SCORE teacher-forced path (research instrument). */
static int g_expert_budget=0; /* EXPERT_BUDGET=N -> cap distinct experts loaded per layer across the
                               * batch-union. Reduces disk I/O on cold/low-RAM hosts by dropping the
                               * lowest-gate-weight experts from the cross-position union. MoE-Spec
                               * (arXiv 2602.16052): top-32 of 64 capture 93% routing weight. */
static int64_t g_budget_dropped=0; /* total experts dropped by EXPERT_BUDGET across all layers */
static int64_t g_budget_rescued=0; /* experts re-kept because a position would have been left with zero */
/* CACHE_ROUTE (paper 2412.00099 max-rank): opt-in only. Keep true top-J always;
 * fill remaining slots preferring pin∪LRU experts ranked within top-M (or mass ROUTE_P). */
static int g_cache_route=0;
static int g_route_j=2;      /* ROUTE_J: sacred top ranks (always take, even uncached) */
static int g_route_m=12;     /* ROUTE_M: max-rank window for cache-preferring fill */
static float g_route_p=0;    /* ROUTE_P: if >0, choose M from cumulative router mass instead */
static float g_route_alpha=1.f; /* ROUTE_ALPHA: scale gate mass of CACHE_ROUTE substitutes before renorm (1=off) */
static int g_route_agree=0;  /* ROUTE_AGREE=1: footer overlap% + mean KL vs true top-K */
static int expert_is_resident(Model *m, int layer, int eid); /* pin∪LRU; defined near pilot */
static int g_spec=1;     /* metodo C: SPEC=0 disabilita il prefetch speculativo cross-layer */
static int g_draft=0;    /* metodo E: DRAFT=n token auto-speculati per forward via n-gram lookup
                          * (0=off). LOSSLESS: verifica = output identico al greedy. Default OFF:
                          * misurato sul run reale (2026-07-03) acceptance ~5% -> ogni draft
                          * rifiutato paga comunque i suoi expert dal disco = ~3x piu' lento.
                          * Opt-in (DRAFT=4) per testi ripetitivi dove l'acceptance e' alta. */
/* metodo F (#48): GRAMMAR=<file.gbnf> -> terza sorgente di draft, la grammatica stessa.
 * Nei workload a output vincolato (JSON/NDJSON, function calling) i byte FORZATI dalla
 * grammatica (chiavi, punteggiatura, valori enum) sono draft gratuiti ad acceptance ~1:
 * nessuna testa, nessuna lookup table, e si aggancia anche dove la testa MTP int4 non
 * parte (#8). MAI un vincolo sul sampling: solo proposte, la verifica batch-union
 * decide — grammatica sbagliata = draft rifiutati, output identico.
 * GRAMMAR_DRAFT=n (default 24) limita i token forzati per forward. */
/* ROUTE_TRACE=<path> (per-position top-K routing, for offline co-activation analysis)
 * and the .coli_usage history both live in route_trace.h now, so every engine writes
 * the same bytes. Measurement only; zero effect on computation. */
/* COUPLE=<.coli_pairs>: coupling-scored cross-layer prefetch. The routing of layer L
 * strongly constrains the routing of L+1/L+2 (measured: median co-activation lift 1.8x
 * over independence, p99 40x, and the structure TRANSFERS across workloads — it is a
 * property of the model, not the session). An offline table (tools/route_pairs.py,
 * built from ROUTE_TRACE dumps) maps (layer, expert) -> top co-activated experts of the
 * next layer(s); after FASE A routing we score candidates by summing counts over the
 * position's routed set and enqueue the top COUPLE_K non-resident ones into the SAME
 * pilot ring (worker, residency re-check, safety invariants unchanged). Unlike PILOT,
 * no router matmul is needed — prediction is a table lookup on ids the layer just
 * produced. Hints only: a wrong prediction costs bandwidth, never output. */
#define CP_M 16
static int g_couple=0, g_couple_k=8, g_couple_d=1;
static int16_t *cp_pred=NULL;    /* [(L*2+(dL-1))*E + e]*CP_M + j -> target id (-1 none) */
static float   *cp_cnt=NULL;
static long g_cp_enq=0;
/* All grammar-forced-draft state in one struct so it can become per-request
 * in the multiplexed server. Fields (same semantics as the former globals):
 * on = grammar loaded and walker alive; armed = lazy start from the first byte
 * accepted at the root (skips preambles); max = forced-span cap per forward;
 * prop/acc = proposed/accepted forced-draft counters. */
typedef struct {
    Grammar gram;
    GrState st;
    Tok *T;
    int on, armed, max;
    uint64_t prop, acc;
    char *src;   /* #7: owned copy of the grammar/schema text that built `gram`, used to
                  * skip recompiling when a slot's next request sends the identical schema.
                  * NULL unless a per-request grammar compiled successfully. */
} GrDraft;
/* NO initializer: GrDraft is ~107 KB (Grammar's 1024 static rules + the walker), and any
 * initializer — even `={.max=24}` — moves the whole struct from .bss into .data, writing
 * 106,848 bytes of mostly zeros into every binary we ship (#527: the v1.1.0 Windows exe
 * grew a 108 KB near-zero-entropy .data blob, which is also exactly the shape an antivirus
 * ML heuristic reads as an unpacking buffer). Static storage is zero-initialized by the C
 * standard; `max` is set in grammar_setup(), which runs before any read of it. */
static GrDraft g_grd;             /* process-level instance: PROMPT mode + run_serve keep using this */
static void couple_prefetch(Model *m, int layer, const int *idx, int Ke);
static int g_looka=0;    /* LOOKA=1: misura (solo contatori, zero effetti) quanto il routing MoE
                          * e' predicibile IN ANTICIPO — la domanda che decide se un prefetch
                          * pilotato dal router puo' riempire i tempi morti del disco.
                          * [0] token precedente, stesso layer (cio' che usa gia' SPEC/PREFETCH)
                          * [1] ingresso del layer -> routing dello STESSO layer (salta l'attention)
                          * [2] post-attention del layer L -> routing di L+1 (un residuo MoE e
                          *     un'attention di anticipo: il punto dove il prefetch avrebbe
                          *     un intero giro di disco per lavorare in ombra). */
static int64_t la_hit[4], la_tot[4];  /* [0]=prev, [1]=skip-attn, [2]=PILOT, [3]=two-step */
static int la_pred[3][130][16]; static signed char la_val[3][130];
static int g_pilot=0;    /* PILOT=1: prefetch pilotato dal router (vedi pilot_prefetch) */
static int g_pilot_k=8;  /* PILOT_K=k: prefetcha solo le prime k predizioni per posizione */
static int g_disk_split=0; /* DISK_SPLIT=1: contatori che spezzano i DISK LOAD (miss LRU) in
                          * draft MTP / absorb / verify-main e in layer MTP (int8) vs main
                          * (int4), con i byte letti. Default OFF: a flag spento gli atomic
                          * non vengono MAI toccati (zero overhead), le righe extra di stats
                          * non vengono stampate. Solo misura: nessun effetto sull'output. */

#include "sample.h"
/* KV-cache quantization tier flags — defined here (before kv_persist.h) so the .coli_kv
 * disk format can see them; the full rationale comments live at their original site below. */
static int g_kv8=0;                             /* KV8=1: fp8 e4m3 latent KV + per-row scale */
static int g_kv8_gs=0;                          /* KV8_GS=<n>: one scale per n latent elements
                                                 * instead of per row (FlashMLA uses 128 on the
                                                 * 512-dim latent). 0 = per-row (unchanged). */
static int g_tq=0, g_tq_bits=4, g_tq_codec=1;   /* KV_TQ: codec 1=rotated int4 (default), 0=PolarQuant */
#include "kv_persist.h"
#include "telemetry.h"

/* Aligned allocator for dense QT weights/scales: under METAL, page-align + register so the
 * GPU reads them zero-copy (no upload duplicate). Plain malloc otherwise. */
/* ---- COLI_NUMA=1 (#82): interleave the expert slabs across NUMA nodes ----
 * On multi-socket hosts first-touch parks nearly the whole pin+LRU on the loader
 * thread's node (measured: node0 766MB free / node1 idle), and every far-socket
 * core then streams weights over the interconnect. Interleaving ONLY the expert
 * slabs recruits all memory controllers: +7%/-14% expert-matmul on 2 sockets,
 * +40% on a 4-socket (#82). Blanket `numactl --interleave=all` is NOT equivalent:
 * it also interleaves the CUDA pinned staging buffers and cost a 4-socket GPU host
 * 10x (#82) — hence per-region mbind here and nothing else. Raw syscall, no libnuma
 * dependency. Linux-only, silent no-op elsewhere or on single-node hosts.
 *
 * VMA discipline (#419): every mbind carries its own memory policy, so a bound
 * region cannot merge with its neighbours — measured ~2 VMAs per slab, with or
 * without MPOL_MF_MOVE. Per-slab binds on a PIN_GB=all load (19,456 experts x
 * slab+fslab) cross the default vm.max_map_count=65530 and posix_memalign dies
 * with terabytes free. So the bulk (the pinned hot-store) is bound as ONE arena
 * per layer (see pin_load), and per-slab mbind remains only for the bounded
 * allocations: dense qalloc, the LRU ecache, and GPU-tier staging. No flag:
 * every bind here lands before the pread that first-touches the pages, so
 * there is nothing to migrate. */
#ifdef __linux__
static int g_numa_nodes=0;      /* only touched under __linux__; off-Linux NUMA is a no-op */
static int g_numa_skip_bind=0;  /* raised around the GPU-prefix pin load: those slabs are
                                 * upload staging, freed right after — binding them buys
                                 * nothing and costs ~2 transient VMAs each (#419) */
#endif
static void numa_slab_bind(void *p, size_t n){
#ifdef __linux__
    if(g_numa_nodes<2 || g_numa_skip_bind || !p || !n) return;
    unsigned long mask=(1UL<<g_numa_nodes)-1;
    uintptr_t a=(uintptr_t)p & ~(uintptr_t)4095;
    size_t len=(((uintptr_t)p+n+4095) & ~(uintptr_t)4095) - a;
    syscall(SYS_mbind,a,len,3/*MPOL_INTERLEAVE*/,&mask,
            (unsigned long)(g_numa_nodes+1),0);
#else
    (void)p;(void)n;
#endif
}
static void numa_init(void){
#ifdef __linux__
    if(!getenv("COLI_NUMA")||!atoi(getenv("COLI_NUMA"))) return;
    for(int i=0;i<64;i++){ char pth[64]; snprintf(pth,sizeof(pth),"/sys/devices/system/node/node%d",i);
        struct stat st; if(stat(pth,&st)) break; g_numa_nodes=i+1; }
    if(g_numa_nodes<2){ fprintf(stderr,"[NUMA] single node: COLI_NUMA ignored\n"); return; }
    /* Probe mbind once so a constrained container degrades with a message
     * instead of silently losing the interleave. The probe page must be
     * page-aligned (mbind rejects unaligned addresses with EINVAL) and only
     * errno==EPERM disables — any other failure keeps NUMA on. */
    { void *pg=NULL;
      if(!posix_memalign(&pg,4096,4096)){
          unsigned long mask=(1UL<<g_numa_nodes)-1; errno=0;
          long rc=syscall(SYS_mbind,pg,4096,3/*MPOL_INTERLEAVE*/,&mask,
                          (unsigned long)(g_numa_nodes+1),0);
          int eperm = rc<0 && errno==EPERM;
          free(pg);
          if(eperm){
              fprintf(stderr,"[NUMA] mbind not permitted (EPERM) — COLI_NUMA disabled\n");
              g_numa_nodes=1; return;
          }
      }
    }
    fprintf(stderr,"[NUMA] expert slabs interleaved across %d nodes\n",g_numa_nodes);
#endif
}

static void *qalloc(size_t n){
#ifdef COLI_METAL
    if(g_metal_enabled){ void *p; size_t r=(n+16383)&~(size_t)16383;
        if(posix_memalign(&p,16384,r)){fprintf(stderr,"OOM qalloc\n");exit(1);}
        coli_metal_register(p,r); return p; }
#endif
    void *p=malloc(n);
    if(n>=(size_t)1<<20) numa_slab_bind(p,n);      /* resident dense weights too (#82: attention/shared stream from RAM every token) */
    return p;
}
static float *qsalloc(int O){ return (float*)qalloc((size_t)O*sizeof(float)); }
static int g_pilot_real=0;/* PILOT_REAL=1: il pilota fa LOAD VERI cross-layer dentro ecache[L+1]
                          * (non il semplice WILLNEED). Implica PILOT=1. Default OFF: hint-only. */
static int g_pilot_two=0; /* PILOT_TWO=1: two-step prefetch — before running L+1's router,
                          * approximate MoE(L) using only the shared expert (resident, no disk)
                          * and add it to the state. Trades 3 small matmuls for +2.3% recall. */
/* Handshake main<->pilota per il load-vero cross-layer. Invariante di sicurezza in DUE parti:
 *  1) Percorso MATMUL (moe): il pilota scrive SOLO ecache[layer] con layer > g_cur_moe_layer;
 *     il matmul in moe() legge SOLO ecache[layer]==g_cur_moe_layer, e la barriera a inizio moe()
 *     aspetta l'eventuale load in volo su QUEL layer. Quindi NESSUNO slot mezzo-caricato viene
 *     mai matmul-ato: il matmul e il pilota non toccano mai lo stesso layer contemporaneamente.
 *  2) Percorso SCAN (pilot_prefetch, anch'esso sul MAIN): la scansione di residenza gira sul
 *     layer FUTURO (lnext = layer corrente + 1), esattamente il layer che il pilota sta scrivendo
 *     -> QUI i due thread toccano davvero la stessa ecache. Percio' quella scansione prende
 *     g_pilot_mx (lo stesso lock del worker): letture e pubblicazione degli slot sono serializzate,
 *     niente torn read di ecn[]/eid. Il pilota non altera MAI il valore di un expert, solo QUALE
 *     expert e' residente: con un load andato a buon fine l'output resta byte-identico all'OFF. */
static pthread_mutex_t g_pilot_mx=PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_pilot_cv=PTHREAD_COND_INITIALIZER;
static _Atomic int g_cur_moe_layer=-1;   /* massimo layer moe in cui il MAIN e' entrato (per forward) */
static int g_pilot_inflight[256];        /* protected by g_pilot_mx; URING can load a layer concurrently */
static int g_pilot_nw=1;                 /* PILOT_WORKERS: blocking-path pilot threads (SPMC ring). 1 = today's behaviour (byte-identical). Only the non-URING PILOT_REAL path fans out; URING already batches to QD>1. */
static int g_pilot_evict_guard=1;/* PILOT_EVICT_GUARD=0 -> old plain-LRU eviction (A/B). Default ON:
                          * a speculative pilot load may evict a RESIDENT expert only if the predicted
                          * expert is historically HOTTER than the victim (same LFRU hysteresis as
                          * tier_pick_lfru); else it drops the speculation rather than thrash a warm
                          * demand-loaded expert. Cache placement only -> output byte-identical. (#441/#474) */
static _Atomic long g_pilot_loads=0;     /* load cross-layer VERI completati (banda spesa) */
static _Atomic long g_pilot_drops=0;     /* predizioni scartate perche' il main possiede gia' il layer */
/* format from `bits`: >=16 f32, 5..8 int8, 4 int4-packed, 3 int3-g64 (group scales), <=2 int2 */
static void qt_alloc(QT *t, int O, int I, int bits){
    t->O=O; t->I=I; t->gs=0; t->qf=NULL; t->q8=NULL; t->q4=NULL; t->s=NULL; t->planar=0;
    if(bits>=16){ t->fmt=0; t->qf=falloc((int64_t)O*I); }
    else if(bits>=5 || g_nopack){ t->fmt=1; t->q8=qalloc((int64_t)O*I); t->s=qsalloc(O); }
    else if(bits>=4){ t->fmt=2; t->q4=qalloc((int64_t)O*((I+1)/2)); t->s=qsalloc(O); }
    else if(bits==3){ t->fmt=5; t->q4=qalloc((int64_t)O*i3_rowbytes(I));
                      t->s=(float*)qalloc((size_t)O*i3_groups(I)*sizeof(float)); }
    else { t->fmt=3; t->q4=qalloc((int64_t)O*((I+3)/4)); t->s=qsalloc(O); }
}
static void qt_fill(QT *t, const float *w, int bits){
    if(t->fmt==0) memcpy(t->qf, w, (int64_t)t->O*t->I*sizeof(float));
    else if(t->fmt==1) quantize_rows(w, t->q8, t->s, t->O, t->I, bits);
    else if(t->fmt==3) pack_int2(w, t->q4, t->s, t->O, t->I, bits);
    else if(t->fmt==5) pack_int3_g64(w, t->q4, t->s, t->O, t->I);
    else pack_int4(w, t->q4, t->s, t->O, t->I, bits);
}

static void rmsnorm(float *out, const float *x, const float *w, int D, float eps){
    double ms=0; for(int i=0;i<D;i++) ms+=(double)x[i]*x[i];
    float r=1.f/sqrtf((float)(ms/D)+eps); for(int i=0;i<D;i++) out[i]=x[i]*r*w[i];
}
/* LayerNorm classica (media+varianza, weight+bias) — usata dal k_norm dell'indexer DSA */
static void layernorm(float *v, const float *w, const float *b, int n, float eps){
    double mu=0; for(int i=0;i<n;i++) mu+=v[i]; mu/=n;
    double var=0; for(int i=0;i<n;i++){ double d=v[i]-mu; var+=d*d; } var/=n;
    float r=1.f/sqrtf((float)var+eps);
    for(int i=0;i<n;i++) v[i]=((float)(v[i]-mu))*r*w[i]+b[i];
}
static void softmax(float *x,int n){ float m=-1e30f; for(int i=0;i<n;i++) if(x[i]>m)m=x[i];
    float s=0; for(int i=0;i<n;i++){x[i]=expf(x[i]-m);s+=x[i];} for(int i=0;i<n;i++) x[i]/=s; }
static inline float sigmoidf(float x){ return 1.f/(1.f+expf(-x)); }
static inline float siluf(float x){ return x/(1.f+expf(-x)); }

/* FFN di un expert MoE (routed): gate+up -> silu(gate)*up -> down. Un solo helper per
 * i ~6 siti inline di moe() e (piu' avanti) il worker distribuito, che la chiama con i
 * propri buffer e i tensori dell'expert — nessun locale di moe(). xg arriva GIA' ruotata
 * (Q^T x) dal chiamante via E8_XE (#452: una copia ruotata per chiamata, MAI per expert).
 * La rotazione dell'input del down e' per-expert e vive qui: `d->fmt==6` la applica
 * SEMPRE — e' la forma CANONICA, pura funzione del fmt del down-tensor. Perche' resta
 * byte-identica al vecchio inline: (a) i fallback device-lost Vulkan ricaricano solo
 * expert che erano in registry (vk_reg_at), e vk_registry_fill ammette solo fmt 2/4/5 —
 * li' d->fmt!=6 e la rotazione e' un no-op; (b) l'oracolo e' single-format, e una
 * conversione normale single-pass produce un modello single-format: le due copie che
 * POSSONO vedere fmt=6 lo fanno solo su un container mixed-format, che il tooling
 * normale non emette mai.
 * Le DUE copie che saltavano la rotazione e potevano davvero vedere fmt=6 — la CPU-share
 * del blocco Vulkan e il fallback CUDA early-issued — ora seguono il canonico: su un
 * container mixed-format (fmt per-tensor da qt_resolve_fmt, nessuna uniformita' tra
 * expert, vedi il commento MB_BUILD) e' la correzione di un bug latente, NON un no-op. */
static void expert_ffn(float *hh, float *gg, float *uu, const float *xg, QT *g, QT *u, QT *d, int nr, int I){
    expert_gate_up(gg,uu,xg,g,u,nr);
    /* K3: silu*up in parallelo — per-elemento, nessun ordine condiviso: gli
     * stessi bit in qualunque schedulazione. Sotto soglia resta seriale (il
     * fork/join costerebbe piu' del loop).
     * EN: elementwise silu*up, parallel above a size threshold; per-element
     * math has no shared accumulation order, so bits are unchanged. */
    #pragma omp parallel for schedule(static) if((int64_t)nr*I>=16384)
    for(int64_t z=0;z<(int64_t)nr*I;z++) gg[z]=siluf(gg[z])*uu[z];
    if(d->fmt==6) e8_rot_rows(gg,nr,I);   /* down input is per-expert — rotate here */
#if !defined(COLI_CUDA)&&!defined(COLI_METAL)&&!defined(COLI_VULKAN)
    /* K3: la meta' mancante di #1071 — la 0.1 sollevo' la quantizzazione della
     * x di gate/up; la h del down restava SERIALE dentro matmul_qt_ex, una
     * volta per expert. Se il down prendera' l'IDOT, quantizza h QUI (righe in
     * parallelo, stessa qrow_i8: byte identici) e pubblica il contesto g_pq
     * che matmul_qt_ex gia' consuma. Azzerato subito dopo: gg viene riscritto
     * dal prossimo expert e un match stantio leggerebbe spazzatura.
     * EN: the down-side half of #1071. If the down matmul will take the IDOT
     * branch, quantize h here (rows in parallel, same qrow_i8 -> identical
     * bytes) and publish the g_pq context matmul_qt_ex already consumes;
     * cleared right after so no stale match can outlive this call. */
    if(g_idot && (d->fmt==1 || (d->fmt==2 && (spec_pinned() ? g_i4s<=1 : nr>=g_i4s)) || (d->fmt==4 && d->planar))){
        static _Thread_local int8_t *hq; static _Thread_local size_t hq_cap;
        static _Thread_local float *hsx; static _Thread_local int32_t *hxs;
        static _Thread_local size_t hrow_cap;
        if((size_t)nr*I>hq_cap){ free(hq); hq=malloc((size_t)nr*I);
            hq_cap=hq?(size_t)nr*I:0; }
        if((size_t)nr>hrow_cap){ free(hsx); free(hxs);
            hsx=malloc((size_t)nr*sizeof(float)); hxs=malloc((size_t)nr*sizeof(int32_t));
            hrow_cap=(hsx&&hxs)?(size_t)nr:0; }
        if(hq && hrow_cap>=(size_t)nr){
            /* copie locali PRIMA della regione: dentro l'omp parallel ogni
             * worker risolve la PROPRIA _Thread_local (NULL), non quella del
             * master — referenziarle direttamente li' segfaulta.
             * EN: locals before the region — TLS variables are re-resolved
             * per thread inside omp parallel; workers would see NULL. */
            int8_t *hq_=hq; float *hsx_=hsx; int32_t *hxs_=hxs;
            #pragma omp parallel for schedule(static) if(nr>=4)
            for(int r=0;r<nr;r++){
                hsx_[r]=qrow_i8(gg+(int64_t)r*I, hq_+(int64_t)r*I, I);
                int32_t t=0; const int8_t *row=hq_+(int64_t)r*I;
                for(int i=0;i<I;i++) t+=row[i];
                hxs_[r]=t;
            }
            g_pq.x=gg; g_pq.S=nr; g_pq.I=I; g_pq.xq=hq; g_pq.sx=hsx; g_pq.xsum=hxs;
            matmul_qt(hh, gg, d, nr);
            g_pq.x=NULL;
            return;
        }
    }
#endif
    matmul_qt(hh, gg, d, nr);
}

/* RoPE interleaved su un vettore di dimensione qk_rope a posizione pos */
static void rope_interleave(float *v, int pos, const Cfg *c){
    int half = c->qk_rope/2;
    /* Validate against the fixed buffers (in[256], cache cs/sn[128] -> qk_rope<=256).
     * Abort cleanly instead of smashing the stack. (GLM-5.2 qk_rope=64.) (#183) */
    if(c->qk_rope > 256){ fprintf(stderr,"qk_rope=%d exceeds rope_interleave buffer (256)\n",c->qk_rope); exit(1); }
    /* inv[j]=powf(theta,-2j/qk_rope) depends only on the model constants theta/qk,
     * never on pos — but the pos-keyed refresh below re-ran `half` powf() calls every
     * position (32 on GLM qk_rope=64), ~11M powf for a 4k prompt x 90 layers. Cache inv[]
     * separately, keyed on (qk,theta), and recompute only cs/sn per position. Byte-
     * identical: `ang = pos*inv[j]` is the same float product against the same inv value
     * the old code formed inline, so cosf/sinf receive the identical argument. (#80) */
    typedef struct { int pos,qk,valid,inv_valid; float theta,inv_theta,inv[128],cs[128],sn[128]; } RopeCache;
    static _Thread_local RopeCache cache;
    float in[256]; memcpy(in,v,c->qk_rope*sizeof(float));
    if(!cache.inv_valid||cache.qk!=c->qk_rope||cache.inv_theta!=c->theta){
        for(int j=0;j<half;j++) cache.inv[j]=powf(c->theta,-2.0f*j/c->qk_rope);
        cache.qk=c->qk_rope; cache.inv_theta=c->theta; cache.inv_valid=1;
        cache.valid=0;   /* inv[] changed -> force cs/sn refresh below */
    }
    if(!cache.valid||cache.pos!=pos||cache.qk!=c->qk_rope||cache.theta!=c->theta){
        for(int j=0;j<half;j++){
            float ang=pos*cache.inv[j];
            cache.cs[j]=cosf(ang); cache.sn[j]=sinf(ang);
        }
        cache.pos=pos; cache.qk=c->qk_rope; cache.theta=c->theta; cache.valid=1;
    }
    for(int j=0;j<half;j++){
        float cs=cache.cs[j],sn=cache.sn[j];
        float a=in[2*j], b=in[2*j+1];
        v[j]      = a*cs - b*sn;
        v[half+j] = b*cs + a*sn;
    }
}

/* ---------- config ---------- */
/* SEC-9: bounded slurp for untrusted config/oracle JSON. config.json arrives from
 * unverified mirrors (see qt_check_fmt threat model); an unbounded ftell->malloc
 * gave a hostile file a load-time OOM or, on malloc failure, a NULL deref via
 * b[got]=0. Cap the size, NULL-check the alloc, require a full read. Returns a
 * malloc'd NUL-terminated buffer, or NULL on any failure. Mirrors tok.h tk_read_file. */
#define CFG_MAX_BYTES (256ll<<20)   /* config/oracle JSON is KB-MB in practice */
static char* cfg_slurp(const char *path){
    FILE *f=fopen(path,"rb"); if(!f) return NULL;
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    if(n<0 || (long long)n>CFG_MAX_BYTES){ fclose(f); return NULL; }
    char *b=malloc((size_t)n+1); if(!b){ fclose(f); return NULL; }
    size_t got=fread(b,1,(size_t)n,f); fclose(f);
    if((long)got!=n){ free(b); return NULL; }
    b[got]=0; return b;
}
static jval* cfg_root(const char *snap, char **arena){
    char p[2048]; snprintf(p,sizeof(p),"%s/config.json",snap);
    FILE *f=fopen(p,"rb"); if(!f){perror(p);exit(1);}
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    /* SEC: config.json arriva dalla dir modello non fidata. Limita la dimensione
     * (un file ostile enorme = OOM al load) e controlla la malloc: senza il NULL
     * check, b[got]=0 su malloc fallita era un NULL-deref. */
    if(n<0 || n>(256L<<20)){ fprintf(stderr,"%s: size %ld out of range (0..256 MB)\n",p,n); exit(1); }
    char *b=malloc((size_t)n+1); if(!b){ fprintf(stderr,"OOM reading %s (%ld bytes)\n",p,n); exit(1); }
    size_t got=fread(b,1,(size_t)n,f); b[got]=0; fclose(f);
    if((long)got!=n) fprintf(stderr,"warning: short read on %s (%ld of %ld)\n",p,(long)got,n);
    return json_parse(b,arena);
}
static int gi(jval*r,const char*k){ jval*v=json_get(r,k); return v?(int)v->num:0; }
static void load_cfg(Cfg *c, const char *snap){
    char *ar=NULL; jval *r=cfg_root(snap,&ar);
    c->hidden=gi(r,"hidden_size"); c->n_layers=gi(r,"num_hidden_layers");
    c->n_heads=gi(r,"num_attention_heads"); c->n_experts=gi(r,"n_routed_experts");
    c->topk=gi(r,"num_experts_per_tok"); c->moe_inter=gi(r,"moe_intermediate_size");
    c->dense_inter=gi(r,"intermediate_size"); c->first_dense=gi(r,"first_k_dense_replace");
    c->q_lora=gi(r,"q_lora_rank"); c->kv_lora=gi(r,"kv_lora_rank");
    c->qk_nope=gi(r,"qk_nope_head_dim"); c->qk_rope=gi(r,"qk_rope_head_dim");
    c->v_head=gi(r,"v_head_dim"); c->n_shared=gi(r,"n_shared_experts"); c->vocab=gi(r,"vocab_size");
    c->n_group=gi(r,"n_group"); c->topk_group=gi(r,"topk_group");
    jval *nt=json_get(r,"norm_topk_prob"); c->norm_topk=(nt&&nt->t==J_BOOL)?nt->boolean:0;
    jval *ep=json_get(r,"rms_norm_eps"); c->eps=ep?(float)ep->num:1e-5f;
    jval *rs=json_get(r,"routed_scaling_factor"); c->routed_scale=rs?(float)rs->num:1.f;
    jval *rp=json_get(r,"rope_parameters"); jval *th=rp?json_get(rp,"rope_theta"):NULL;
    c->theta = th?(float)th->num:10000.f;
    /* token di stop: GLM-5.2 ne ha TRE (endoftext, user, observation). Fermarsi solo sul
     * primo = generare spazzatura invisibile dopo la fine del turno (5-10x token sprecati). */
    c->n_stop=0;
    jval *eo=json_get(r,"eos_token_id");
    if(eo){ if(eo->t==J_NUM) c->stop_ids[c->n_stop++]=(int)eo->num;
            else if(eo->t==J_ARR) for(int i=0;i<eo->len && c->n_stop<8;i++)
                c->stop_ids[c->n_stop++]=(int)eo->kids[i]->num; }
    /* generation_config.json e' il file AUTOREVOLE per la generazione secondo HuggingFace:
     * config.json ne porta spesso una copia legacy o parziale. Un tool di conversione che
     * rigenera un config.json ridotto lascia il motore fermo su MENO stop del dovuto, e i
     * token di controllo che restano finiscono stampati in chat come testo (woolcoxm, #298:
     * "the stop token being printed to chat", verificato sui token id). Unione dei due:
     * uno stop in piu' non fa danno, uno in meno si' -- e chi converte i pesi non siamo noi.
     * EN: generation_config.json is HF's authority for generation; config.json often carries
     * a partial legacy copy. Union both -- an extra stop is harmless, a missing one is not. */
    { char gp[2100]; snprintf(gp,sizeof(gp),"%s/generation_config.json",snap);
      FILE *gf=fopen(gp,"rb");                  /* assente = nessun problema: e' opzionale */
      if(gf){
        fseek(gf,0,SEEK_END); long gn=ftell(gf); fseek(gf,0,SEEK_SET);
        char *gb = (gn>0 && gn<=(256L<<20)) ? malloc((size_t)gn+1) : NULL;   /* SEC: cap + NULL check */
        if(gb){
            size_t gg=fread(gb,1,(size_t)gn,gf); gb[gg]=0;
            char *ga=NULL; jval *gr=json_parse(gb,&ga);
            jval *ge=gr?json_get(gr,"eos_token_id"):NULL;
            if(ge){
                int add[8], na=0;
                if(ge->t==J_NUM) add[na++]=(int)ge->num;
                else if(ge->t==J_ARR) for(int i=0;i<ge->len && na<8;i++) add[na++]=(int)ge->kids[i]->num;
                for(int i=0;i<na && c->n_stop<8;i++){
                    int dup=0; for(int j=0;j<c->n_stop;j++) if(c->stop_ids[j]==add[i]) dup=1;
                    if(!dup) c->stop_ids[c->n_stop++]=add[i];
                }
            }
            free(ga); free(gb);
        }
        fclose(gf);
      } }
    /* DSA lightning indexer: parametri + tipo per-layer (lista esplicita o formula freq/offset) */
    c->index_topk=gi(r,"index_topk"); c->index_nh=gi(r,"index_n_heads"); c->index_hd=gi(r,"index_head_dim");
    { jval *it=json_get(r,"indexer_types");
      int freq=gi(r,"index_topk_freq"); if(freq<1) freq=1;
      jval *of=json_get(r,"index_skip_topk_offset"); int off=of?(int)of->num:2;
      for(int i=0;i<c->n_layers && i<128;i++){
          if(it && it->t==J_ARR && i<it->len && it->kids[i]->str)
              c->idx_type[i] = !strcmp(it->kids[i]->str,"full");
          else { int v=i-off+1; if(v<0) v=0; c->idx_type[i] = (v%freq)==0; }
      } }
    c->qk_head=c->qk_nope+c->qk_rope;
    c->attn_scale = 1.f / sqrtf((float)c->qk_head);
    if(c->n_group!=1){ fprintf(stderr,"this engine requires n_group=1 (GLM-5.2)\n"); exit(1); }
    /* VALIDAZIONE (report PR #25): il config.json arriva da mirror non fidati — dimensioni
     * ostili non devono superare questo punto. Un solo choke point protegge ogni alloc a valle. */
    #define CKR(name,v,lo,hi) if((v)<(lo)||(v)>(hi)){ \
        fprintf(stderr,"config: %s=%d is outside [%d,%d]\n",name,(int)(v),(int)(lo),(int)(hi)); exit(1); }
    CKR("hidden_size",c->hidden,1,1<<20)         CKR("num_hidden_layers",c->n_layers,1,128)
    CKR("num_attention_heads",c->n_heads,1,1024) CKR("n_routed_experts",c->n_experts,1,4096)
    CKR("num_experts_per_tok",c->topk,1,64)      CKR("moe_intermediate_size",c->moe_inter,1,1<<20)
    CKR("intermediate_size",c->dense_inter,1,1<<24) CKR("first_k_dense_replace",c->first_dense,0,c->n_layers)
    CKR("q_lora_rank",c->q_lora,0,1<<20)         CKR("kv_lora_rank",c->kv_lora,1,1<<20)
    CKR("qk_nope_head_dim",c->qk_nope,1,1<<16)   CKR("qk_rope_head_dim",c->qk_rope,1,1<<16)
    CKR("v_head_dim",c->v_head,1,1<<16)          CKR("n_shared_experts",c->n_shared,0,64)
    CKR("vocab_size",c->vocab,1,1<<24)           CKR("index_topk",c->index_topk,0,1<<20)
    CKR("index_n_heads",c->index_nh,0,1024)      CKR("index_head_dim",c->index_hd,0,1<<16)
    if(c->topk>c->n_experts){
        fprintf(stderr,"config: num_experts_per_tok=%d exceeds n_routed_experts=%d\n",
                c->topk,c->n_experts); exit(1); }
    #undef CKR
    free(ar);
}

/* Derive the fmt=4 group size from the scale-array byte count. A grouped-int4
 * tensor stores ceil(I/gs) f32 scales per output row, so:
 *     ns_bytes == O * ceil(I/gs) * 4   =>   gs == I * 4 / (ns_bytes/O - ... )
 * We probe candidate group sizes (must be a multiple of 16, the AVX2 vector
 * width the grouped kernel requires) from finest to coarsest and return the
 * first whose predicted scale-array size matches ns_bytes. Returns 0 if no
 * candidate fits (then it's plain per-row int4, fmt=2, not grouped).
 * Data-driven: g64/g128/g256 all just work; adding a size means listing it. */
static int detect_group_size(int O, int I, int64_t ns){
    if(O<=0 || ns<=(int64_t)O*4 || I<=0) return 0;   /* not grouped */
    /* ns/O is the per-row scale bytes; groups = (ns/O)/4; gs = ceil(I/groups).
     * Probe from small gs (finest granularity) upward so the most granular
     * match wins — that's what we want, since finer groups are unambiguous. */
    static const int cands[]={16,32,48,64,96,128,192,256};
    for(int ci=0; ci<(int)(sizeof(cands)/sizeof(cands[0])); ci++){
        int gs=cands[ci];
        if(gs>I) break;
        int ng=(I+gs-1)/gs;
        if(ns==(int64_t)O*ng*4) return gs;
    }
    return 0;
}

/* FORMAT NAME <-> internal fmt-int table. The NAME is the public identity a
 * container or a Feature Request advertises; the int on the right is this
 * build's internal enum value (colibri.c's QT.fmt / qt_resolve_fmt's return
 * value), never itself persisted to a container -- a container's __metadata__
 * stamp (below) carries the NAME, never the number, matching docs/FORMATS.md's
 * own registry. Covers every format qt_resolve_fmt can return, not just the
 * one this branch's tool stamps: a single-entry table could only ever
 * exercise the "unrecognized name" refusal path, never a genuine
 * recognized-but-different-format mismatch. "e8-iq3-lattice" is this build's
 * own placeholder name for upstream fmt=6 (#465 never stamped containers --
 * dev has no metadata-stamp feature of its own) -- listed so
 * qt_fmt_by_name/qt_name_by_fmt are total over every value qt_resolve_fmt can
 * return, matching this comment's own claim; a real name for fmt=6 belongs to
 * whoever upstreams a stamp for it. "fp8-e4m3-b128" (fmt=8) names the WEIGHT
 * geometry only, not a specific scale encoding -- a tensor stamped
 * "fp8-e4m3-b128" whose scale sidecar carries UE8M0 bytes still refuses (see
 * qt_resolve_fmt's "SCALE ENCODING IS A DECLARED PROPERTY" comment): the stamp
 * confirms the WEIGHT format, it does not grant this build a decoder it
 * doesn't have. */
static const struct { const char *name; int fmt; } FMT_NAMES[] = {
    { "f32",           0 },
    { "int8-row",      1 },
    { "int4-row",      2 },
    { "int2-row",      3 },
    { "int4-grouped",  4 },
    { "int3-g64",      5 },
    { "e8-iq3-lattice", 6 },
    { "fp8-e4m3-b128", 8 },
};
#define N_FMT_NAMES (int)(sizeof(FMT_NAMES)/sizeof(FMT_NAMES[0]))

static int qt_fmt_by_name(const char *name){
    for(int i=0;i<N_FMT_NAMES;i++) if(!strcmp(FMT_NAMES[i].name,name)) return FMT_NAMES[i].fmt;
    return -1;   /* unrecognized name -- never a valid qt_resolve_fmt() return value */
}
static const char *qt_name_by_fmt(int fmt){
    for(int i=0;i<N_FMT_NAMES;i++) if(FMT_NAMES[i].fmt==fmt) return FMT_NAMES[i].name;
    return NULL; /* fmt has no registered public name yet (e.g. upstream 0-5) */
}

/* SEC: risolve e VALIDA il formato quantizzato di un tensore [O,I] letto da un
 * container non fidato (mirror). L'inferenza precedente (`?1:?2:3`) cadeva su
 * int2 per QUALSIASI conteggio byte non riconosciuto: un peso troppo corto
 * diventava un int2 valido e il matmul leggeva oltre il buffer (O*I nibble a
 * 4/byte). Qui i byte del peso devono corrispondere a un layout noto e i byte
 * della scala alla cardinalita' attesa (O per-row, O*ng per-gruppo) — altrimenti
 * si termina invece di sforare. Ritorna fmt (1/2/3/4/5/6/8) e scrive *gs.
 * `stamped_name`: the tensor's __metadata__ format-NAME stamp if the caller
 * looked one up (st_fmt_stamp, st.h), else NULL -- used ONLY to break a
 * genuine byte-count collision (see THE DESIGN LANDMINE and the SECOND
 * DESIGN LANDMINE below); every other decision in this function is
 * byte-arithmetic alone, unchanged by whether a stamp is present. A stamp
 * can never grant this build a decoder it doesn't have: the UE8M0
 * scale-encoding refusals below stay refusals even when a stamp names
 * "fp8-e4m3-b128" correctly, because that name confirms the WEIGHT format,
 * not a scale encoding this build can read. Routed-expert callers
 * (expert_load_impl and friends) always pass NULL: this branch's repack
 * tool never stamps routed experts, so there is nothing for those paths to
 * consult. */
static int qt_resolve_fmt(const char *name, int O, int I, int64_t nb, int64_t ns, int *gs, const char *stamped_name){
    int64_t exp_i8=(int64_t)O*I, exp_i4=(int64_t)O*((I+1)/2), exp_i2=(int64_t)O*((I+3)/4);
    int64_t exp_i3=(int64_t)O*i3_rowbytes(I);   /* int3-g64 (fmt=5): 24B per 64-input group */
    /* fmt=6 (E8/IQ3, #452): scales live inside the 98B super-blocks, so the .qs
     * convention is kept with a single-float tag — ns==4 is the discriminator
     * (every other format carries at least O floats of real scales).
     *
     * SECOND DESIGN LANDMINE: e8_rowbytes(I) = ceil(I/256)*98 is the constant 98
     * for every I in (0,256], so this check's nb==O*e8_rowbytes(I) collapses to
     * nb==O*98 -- which is ALSO fp8-e4m3-b128's raw-byte weight count (O*I) at
     * the ONE value of I where I itself equals 98 (solving 98k==I for I in
     * ((k-1)*256,k*256] only admits k=1, I=98). At that same I=98 a handful of
     * fp8 scale-array shapes ALSO carry exactly ns==4 bytes, same as this
     * check's own tag, and each is a genuine collision candidate this block
     * must catch before the unconditional `return 6` below:
     *   - a SINGLE-BLOCK fp8 tensor (O<=128, fp8_nblk(O)*fp8_nblk(I)==1) with
     *     f32 block scales: ns==1*4==4.
     *   - a FOUR-BLOCK fp8 tensor (fp8_nblk(O)*fp8_nblk(I)==4, e.g. O in
     *     (384,512] at this I) with UE8M0 (1 byte/block) scales: ns==4*1==4 --
     *     the SAME arithmetic collision the fmt=1-vs-fmt=8 landmine below has
     *     with UE8M0, just against fmt=6's tag instead of fmt=1's per-row
     *     count; see that landmine's own comment for the general shape.
     *   - at O==1 specifically, fmt=1's own per-row tag (O*4) is also 4.
     * A stamp naming exactly one live candidate resolves the ambiguity --
     * EXCEPT a "fp8-e4m3-b128" stamp on the UE8M0-shaped candidate: the stamp
     * confirms the WEIGHT format, not a scale encoding this build can decode,
     * so that specific case still refuses (same "recognized but not
     * implemented" discipline as the landmine below). Everything else
     * (unstamped, or a stamp that doesn't resolve) refuses rather than let
     * dev's own unconditional `return 6` silently misread an fp8-e4m3-b128
     * (or, at O==1, plain int8) tensor as E8/IQ3-lattice-decoded garbage with
     * no error. No real GLM tensor has these shapes; the discipline exists
     * for untrusted containers. */
    if(ns==4 && nb==(int64_t)O*e8_rowbytes(I)){
        int fp8_blk_f32_also   = (nb==(int64_t)O*I) && (fp8_nblk(O)*fp8_nblk(I)==1);
        int fp8_blk_ue8m0_also = (nb==(int64_t)O*I) && (fp8_nblk(O)*fp8_nblk(I)==4);
        int i8_row_also        = (nb==(int64_t)O*I) && (O==1);   /* ns==O*4==4 iff O==1 */
        if(fp8_blk_f32_also || fp8_blk_ue8m0_also || i8_row_also){
            int sf = stamped_name ? qt_fmt_by_name(stamped_name) : -1;
            if(sf==6){ *gs=0; return 6; }
            if(sf==8 && fp8_blk_f32_also){ *gs=0; return 8; }
            if(sf==1 && i8_row_also){ *gs=0; return 1; }
            /* sf==8 with only fp8_blk_ue8m0_also true falls through here too --
             * see the comment above this block. */
            fprintf(stderr,"%s: [%d,%d] byte layout (nb=%lld ns=%lld) matches E8/IQ3 "
                "(fmt=6, 4-byte tag)%s%s%s; refusing rather than guessing (untrusted "
                "container, fmt=6 collision at I=98)%s\n",
                name,O,I,(long long)nb,(long long)ns,
                fp8_blk_f32_also   ? " AND per-128x128-block FP8 f32 scales (fmt=8, single block)" : "",
                fp8_blk_ue8m0_also ? " AND per-128x128-block FP8 ue8m0 scales (fmt=8, 4 blocks, recognized-not-implemented)" : "",
                i8_row_also        ? " AND plain int8 per-row (fmt=1, O=1)" : "",
                stamped_name ? " -- metadata stamp present but names a format/encoding that doesn't resolve the ambiguity"
                             : "");
            exit(1);
        }
        *gs=0; return 6;
    }
    /* Row formats take precedence: for tiny I the int3-g64 byte count can coincide with
     * a row layout (e.g. [O,48]: ceil(48/2)=24=1*24). For real tensor shapes the counts
     * are distinct, and the weight bytes — not the scale size — are the int3 tag, because
     * int3-g64 and grouped-int4-at-gs=64 carry the SAME scale cardinality O*ceil(I/64). */
    int fmt = (nb==exp_i8)?1 : (nb==exp_i4)?2 : (nb==exp_i2)?3 : (nb==exp_i3)?5 : 0;
    if(!fmt){
        fprintf(stderr,"%s: quantized weight is %lld bytes — no int8/int4/int2/int3-g64/fp8 layout for [%d,%d], refusing (untrusted container)\n",
                name,(long long)nb,O,I); exit(1); }
    *gs=0;
    if(fmt==2){ int g=detect_group_size(O,I,ns); if(g>0){ fmt=4; *gs=g; } }
    /* fmt=1 vs fmt=8 (native FP8-e4m3 passthrough): THE DESIGN LANDMINE. Weight
     * bytes are IDENTICAL (O*I raw bytes, both matched exp_i8 above) -- fmt=1
     * and fmt=8 can only be told apart by the SCALE array's geometry: fmt=1 is
     * per-row (ns==O*4 bytes); fmt=8 is per-128x128-BLOCK, ns==ceil(O/128)*
     * ceil(I/128)*4 bytes for THIS build's implemented f32 scale encoding. For
     * most real shapes these two byte counts are distinct and the match is
     * unambiguous. But for small O (<=128) and/or small I, ceil(O/128)*
     * ceil(I/128) can equal O exactly (e.g. O=1,I<=128 -> 1*1=1=O; O=2,I in
     * [129,256] -> 1*2=2=O; O=256,I in (16256,16384] -> 2*128=256=O) -- a
     * tensor whose scale array satisfies BOTH conventions at once.
     *
     * REVIEW FINDING (maintainer, #528): this collision is not a theoretical
     * corner case. GLM-5.2's own self_attn.o_proj.weight loads as
     * [D,H*v_head] = [6144,16384]: nblkO=ceil(6144/128)=48,
     * nblkI=ceil(16384/128)=128, nblkO*nblkI=6144==O -- a real, pre-existing,
     * VALID int8-row o_proj tensor hits this exact byte-count collision on
     * every GLM-5.2 checkpoint this engine has ever loaded. The general
     * family is any int8 tensor where O==ceil(O/128)*ceil(I/128) (see the
     * CENSUS SCAN, tools/fp8_collision_census.py, for the complete
     * enumerated set over this repo's own containers). An earlier revision
     * of this function refused unconditionally here -- exit(1) at load time
     * on an ordinary, already-shipping, valid model -- which is a strictly
     * worse failure mode than the misread it was guarding against.
     *
     * INVERSION, governing the UNSTAMPED case: an ambiguous shape with no
     * resolving stamp resolves to fmt=1 -- the incumbent, already-on-disk,
     * decodable format -- instead of refusing. This is sound because the
     * WRITER side (tools/repack_fp8_passthrough.py's _check_geometry) now
     * refuses to EMIT an fmt=8 container at any shape satisfying this same
     * collision predicate: no genuine fmt=8 tensor this engine's own tooling
     * can produce will ever reach this branch unstamped, so resolving to
     * fmt=1 here is not a guess against a live fmt=8 candidate -- it is the
     * only remaining candidate. THE STAMPED CASE IS UNCHANGED by this
     * inversion: a metadata stamp present and naming exactly one of the two
     * colliding candidates (int8-row or fp8-e4m3-b128) still resolves via
     * that stamp through the existing TRUST-VERIFY-REFUSE path below (the
     * maintainer verified that path is sound on #524/#529) -- a THIRD-PARTY
     * fmt=8 container can still declare itself at this same colliding shape
     * and be believed, exactly as before. A stamp naming anything else
     * (unrecognized, or a format that isn't one of the two candidates) does
     * NOT resolve the ambiguity and refuses, same as it did before this
     * inversion -- the inversion changes ONLY what an absent stamp does here.
     *
     * SCALE ENCODING IS A DECLARED PROPERTY, not a hardcoded constant: f32 (4
     * bytes/block, above) is what THIS build implements, but it is not the
     * only encoding real fp8-e4m3-b128 weight geometry ships with. DeepSeek-V4
     * ships the SAME weight layout (FP8 E4M3, 128x128 blocks) with UE8M0
     * scales instead -- one byte per block, a power-of-two exponent, dtype
     * F8_E8M0 (maintainer finding, colibri #524) -- so a fp8-e4m3-b128
     * container's scale sidecar can legitimately be ceil(O/128)*ceil(I/128)*1
     * bytes, not *4. That byte count is a REAL, distinct signature (never
     * equal to the f32 block-scale count above for any O,I>=1, since one is
     * exactly 4x the other and 4n==n only at n==0) -- recognized here rather
     * than silently misread as a truncated or corrupt f32 scale array. This
     * build does not implement UE8M0 decode: recognizing the signature and
     * refusing BY NAME (rather than falling through to the generic
     * "wrong byte count" refusal below, or worse, matching it against the
     * wrong candidate) is the whole point -- a future decoder lands into this
     * seam rather than a near-duplicate format. Checked for collision against
     * every OTHER format's ns arithmetic reachable from this nb==O*I branch
     * (fmt=1 and fmt=8-f32, both above): the byte counts are realistically
     * distinct (nblkO*nblkI is orders of magnitude smaller than O*4 for any
     * real GLM-sized matrix), but NOT categorically distinct -- the same
     * small-O regime that makes the fmt=1-vs-fmt=8-f32 collision above
     * possible also makes a fmt=1-vs-fmt=8-ue8m0 collision possible (e.g.
     * O=1, I in (384,512]: nblkO=1, nblkI=4, ue8m0 ns=4=O*4=fmt=1's own
     * per-row count). A stamp naming "int8-row" resolves that specific
     * corner to fmt=1 (the stamp confirms it really is plain int8, a format
     * this build CAN decode); a stamp naming "fp8-e4m3-b128" does NOT resolve
     * it, for the same "recognized but not implemented" reason the clean
     * (non-colliding) ue8m0 case below refuses regardless of any stamp.
     * The colliding FAMILY is exactly: nb==O*I && ns==O*4 &&
     * ceil(O/128)*ceil(I/128)==4*O (is_row && is_blk_ue8m0 below; is_blk can
     * never co-hold since nblk==O and nblk==4*O are disjoint for O>=1).
     * At I<=16384 (nblkI<=128) membership forces O<=32; every GLM-5.2
     * resident/routed role has O>=576 (kv_a's kv_lora+qk_rope is the
     * smallest -- tools/fp8_collision_census.py enumerates the roles, and
     * the census over the real checkpoint was run in this PR's rev5 round),
     * so no GLM-5.2 tensor is a member -- the discipline exists for
     * untrusted containers. tests/test_fp8_load.c's Part A3b pins the
     * family's polarity at member shapes [1,400], [2,1024], [129,33000].
     * See also the SECOND DESIGN LANDMINE above for the analogous
     * ue8m0-vs-fmt=6 corner this same signature can hit. */
    if(fmt==1){
        int64_t nblkO=fp8_nblk(O), nblkI=fp8_nblk(I);
        int64_t ns_row=(int64_t)O*4, ns_blk=nblkO*nblkI*4, ns_blk_ue8m0=nblkO*nblkI;
        int is_row=(ns==ns_row), is_blk=(ns==ns_blk), is_blk_ue8m0=(ns==ns_blk_ue8m0);
        int sf = stamped_name ? qt_fmt_by_name(stamped_name) : -1;
        /* THE DESIGN LANDMINE, INVERTED for the UNSTAMPED case (maintainer
         * review, #528): is_row&&is_blk with no resolving stamp used to
         * exit(1) unconditionally -- see this function's "REVIEW
         * FINDING"/"INVERSION" comment above for why that was wrong. The
         * STAMPED sub-case (sf==1 || sf==8) is untouched by the inversion:
         * it already resolved via TRUST-VERIFY-REFUSE before this revision
         * and still does. Only the "no stamp, or a stamp that names neither
         * candidate" branch changes: it used to exit(1); it now falls
         * through to fmt=1 (already the value coming into this block, so no
         * explicit assignment is needed -- the `is_blk && !is_row` check
         * further down evaluates false whenever is_row is true). A stamp
         * that names some OTHER, unrelated format still refuses -- the
         * inversion applies only to the genuinely stamp-less case. */
        if(is_row && is_blk){
            if(sf==1 || sf==8){
                fmt = sf;
            } else if(stamped_name){
                fprintf(stderr,"%s: [%d,%d] scale array is %lld bytes — matches BOTH per-row "
                    "int8 (fmt=1) and per-128x128-block FP8 (fmt=8) scale geometry; refusing "
                    "rather than guessing (untrusted container, THE DESIGN LANDMINE) -- metadata "
                    "stamp present but names a format that doesn't resolve the ambiguity\n",
                    name,O,I,(long long)ns);
                exit(1);
            }
            /* else (no stamp at all): falls through to fmt=1, the INVERSION. */
        } else if(is_blk_ue8m0){
            if(sf==1 && is_row){
                fmt = 1;   /* stamp confirms this is genuinely plain int8, not an
                            * unimplemented-encoding fp8 tensor -- safe to resolve. */
            } else {
                fprintf(stderr,"%s: [%d,%d] fp8-e4m3-b128 with ue8m0 scales recognized but not "
                    "implemented; only f32 block scales are supported in this build (nb=%lld "
                    "bytes matches raw e4m3 weight bytes, ns=%lld bytes matches %lld blocks x "
                    "1 byte/block)%s%s -- refusing rather than misreading the sidecar "
                    "(untrusted container)\n",
                    name,O,I,(long long)nb,(long long)ns,(long long)(nblkO*nblkI),
                    is_row ? " -- scale array ALSO matches per-row int8 (fmt=1)" : "",
                    stamped_name ? " -- a metadata stamp cannot grant this build a decoder it doesn't have"
                                 : "");
                exit(1);
            }
        } else if(is_blk && !is_row) fmt=8;
    }
    int64_t exp_scale = (fmt==4)? (int64_t)O*((I+*gs-1)/(*gs))
                      : (fmt==5)? (int64_t)O*i3_groups(I)
                      : (fmt==8)? fp8_nblk(O)*fp8_nblk(I)
                      : (int64_t)O;   /* in FLOAT */
    if(ns != exp_scale*4){
        fprintf(stderr,"%s: scale array is %lld bytes — expected %lld for [%d,%d] fmt=%d, refusing (untrusted container)\n",
                name,(long long)ns,(long long)(exp_scale*4),O,I,fmt); exit(1); }
    return fmt;
}

/* TRUST-VERIFY-REFUSE: if `stamped` (the tensor's __metadata__ format-NAME
 * stamp, or NULL if none -- see st_fmt_stamp/st_fmt_stamp_ingest in st.h)
 * is present, verify it agrees with `fmt` (qt_resolve_fmt's byte-arithmetic
 * result) and refuse loudly on disagreement -- same "untrusted container"
 * discipline qt_resolve_fmt applies throughout. A stamp naming a format this
 * build doesn't recognize is ALSO a refusal: silently accepting an
 * unrecognized name would be indistinguishable from missing a real mismatch,
 * and "refuse rather than guess" is the whole point of this function's
 * design. No stamp at all is NOT an error -- the container simply predates
 * this feature (or was never stamped by a stamping tool), and byte-arithmetic
 * inference alone decides, exactly as before this function existed: zero
 * behavior change for unstamped containers.
 *
 * Called from qt_from_disk right after qt_resolve_fmt -- the resident-tensor
 * load path this branch's repack tool actually stamps. Deliberately NOT
 * threaded into the routed-expert loader paths (expert_load_impl and
 * friends, which call qt_resolve_fmt separately for g/u/d slab layout): this
 * branch's tools/repack_fp8_passthrough.py never stamps routed experts (kind
 * "x" is explicitly excluded, see that tool's module docstring), so there is
 * no stamp for those paths to verify yet -- adding the plumbing there now
 * would be framework-building ahead of any container that needs it, which
 * this reference implementation deliberately avoids. */
static void qt_verify_fmt_stamp(const char *name, const char *stamped, int fmt){
    if(!stamped) return;                       /* unstamped: infer exactly as today */
    int stamped_fmt = qt_fmt_by_name(stamped);
    if(stamped_fmt == fmt) return;              /* agree: silent pass-through, loads normally */
    if(stamped_fmt < 0){
        fprintf(stderr,
            "%s: metadata stamp names format '%s', which this build does not recognize "
            "(byte-arithmetic inference says fmt=%d) -- refusing (untrusted container, "
            "unrecognized stamp name)\n", name, stamped, fmt);
        exit(1);
    }
    const char *inferred_name = qt_name_by_fmt(fmt);
    fprintf(stderr,
        "%s: metadata stamp says format '%s' but byte-arithmetic inference says fmt=%d "
        "(%s) -- refusing (untrusted container, stamp/inference mismatch)\n",
        name, stamped, fmt, inferred_name ? inferred_name : "no registered name");
    exit(1);
}

/* costruisce un QT [O,I] dal disco in `t` (buffer riusabili tra chiamate).
 *  - se esiste `name.qs`: pesi GIA' quantizzati nel container (U8 qdata + F32 scala) -> letti diretti
 *  - altrimenti: tensore pieno (f32/bf16) -> quantizzato a runtime a `bits` (oracolo tiny / pesi pieni)
 * drop=1 -> fadvise DONTNEED (streaming expert). */
static void qt_from_disk(Model *m, const char *name, int O, int I, int bits, int drop, QT *t){
    t->planar=0;   /* K1: il flag non sopravvive mai a un refill */
    char sn[300]; snprintf(sn,sizeof(sn),"%s.qs",name);
    if(st_has(&m->S,sn)){
        int64_t nb=st_nbytes(&m->S,name);
        int64_t ns=st_nbytes(&m->S,sn);   /* scale bytes (F32) */
        /* fmt=4 int4-grouped: byte int4 ma scala > O*4 — gs deriva dalla scala.
         * qt_resolve_fmt valida entrambi i conteggi contro [O,I] e termina se
         * non fidati (SEC). */
        int gs=0;
        const char *stamped = st_fmt_stamp(&m->S,name);   /* NULL if unstamped */
        int fmt = qt_resolve_fmt(name,O,I,nb,ns,&gs,stamped);
        qt_verify_fmt_stamp(name,stamped,fmt);   /* TRUST-VERIFY-REFUSE: no-op if unstamped */
        if(fmt==1){ if(t->fmt!=1||!t->q8){ t->fmt=1; t->O=O; t->I=I; t->gs=0; t->q8=qalloc(nb); t->s=qsalloc(O); } st_read_raw(&m->S,name,t->q8,drop); }
        else if(fmt==4){ int ng=(I+gs-1)/gs;
            /* METAL: t->s must be page-aligned + coli_metal_register'd like every other
             * fmt's scale buffer (qsalloc, used by fmt 1/2/3 just below/above), or
             * bind_gemv/coli_metal_gemm's resolve(t->s,...) can never find it and the
             * whole GPU dispatch silently CPU-falls-back (safe, but defeats the point of
             * wiring fmt=4 into the shader at all). falloc() here was a plain malloc --
             * O*ng floats never registered -- found while tracing the per-row-scale path
             * this stage extends to per-group; fixed alongside it.
             * Trade-off (this site is NOT #ifdef COLI_METAL): on a CPU-only build this
             * swaps falloc's checked-exit overflow/OOM guard for qalloc's unchecked
             * malloc -- same hole qsalloc already has for fmt 1/2/3, so consistency,
             * not a new class; noted rather than silent. fmt=5's group scales (just
             * below) still use falloc and so remain Metal-inert -- pre-existing,
             * inconsistent after this change, a candidate for whoever wires fmt=5. */
            if(t->fmt!=4||!t->q4){ t->fmt=4; t->O=O; t->I=I; t->gs=gs; t->q4=qalloc(nb); t->s=(float*)qalloc((size_t)O*(size_t)ng*sizeof(float)); }
            st_read_raw(&m->S,name,t->q4,drop); }
        else if(fmt==5){ int64_t ng=i3_groups(I);   /* int3-g64: 24B/group weights + O*ng group scales */
            if(t->fmt!=5||!t->q4){ t->fmt=5; t->O=O; t->I=I; t->gs=0; t->q4=qalloc(nb); t->s=falloc((int64_t)O*ng); }
            st_read_raw(&m->S,name,t->q4,drop); }
        else if(fmt==6){   /* E8/IQ3: everything in-block, .qs is the 4-byte tag */
            if(t->fmt!=6||!t->q4){ t->fmt=6; t->O=O; t->I=I; t->gs=0; t->q4=qalloc(nb); t->s=qsalloc(1); }
            st_read_raw(&m->S,name,t->q4,drop); }
        else if(fmt==8){ int64_t nblk=fp8_nblk(O)*fp8_nblk(I);   /* fp8-e4m3: raw bytes (q8, like fmt=1)
            * + one f32 scale per 128x128 block. BOTH weights and scale via qalloc, not falloc,
            * from day one -- the GPU-visibility lesson fmt=4 and fmt=5 each had to learn
            * separately (see review round 1 audit history in the report). */
            if(t->fmt!=8||!t->q8){ t->fmt=8; t->O=O; t->I=I; t->gs=0; t->q8=qalloc(nb); t->s=(float*)qalloc((size_t)nblk*sizeof(float)); }
            st_read_raw(&m->S,name,t->q8,drop); }
        else      { if(t->fmt!=fmt||!t->q4){ t->fmt=fmt; t->O=O; t->I=I; t->gs=0; t->q4=qalloc(nb); t->s=qsalloc(O); } st_read_raw(&m->S,name,t->q4,drop); }
        /* cap MUST match the scale cardinality qt_resolve_fmt already validated and
         * the falloc above actually reserved, per format: grouped-int4 (fmt=4) keeps
         * O*ceil(I/gs) scales, int3-g64 (fmt=5) keeps O*i3_groups(I), E8/IQ3 (fmt=6)
         * keeps the 1-float tag, and fp8-e4m3 (fmt=8) keeps ceil(O/128)*ceil(I/128)
         * block scales; everything else is per-row O. Using the per-row bound for a
         * grouped/blocked format would reject a legitimate container (fmt=5 regressed
         * exactly that way). */
        /* fmt=8 goes through st_read_scale_f32: the block-scale GEOMETRY is the
         * same in a container we repacked (f32 scales) and in a native fp8
         * checkpoint (UE8M0, one byte per block) -- same shape, same meaning,
         * same multiply. Only the encoding of the number differs. The reader
         * accepts either and always yields f32, so matmul_fp8 stays a single
         * implementation with no branch in the hot loop.
         *
         * Every OTHER format still goes through st_read_f32_cap exactly as
         * before: fmt 0/1/2/4/5/6 are byte-for-byte unchanged, and an f32-scaled
         * fmt=8 container behaves identically too (st_read_scale_f32 dispatches
         * to st_read_f32 for dtype F32, the same call it made before). */
        if(fmt==8) st_read_scale_f32(&m->S,sn,t->s,fp8_nblk(O)*fp8_nblk(I),drop);
        else st_read_f32_cap(&m->S,sn,t->s,
                        fmt==4 ? (int64_t)O*((I+gs-1)/gs) :
                        fmt==5 ? (int64_t)O*i3_groups(I)  :
                        fmt==6 ? (int64_t)1               : (int64_t)O, drop);
    } else {
        if(!t->qf && !t->q8 && !t->q4) qt_alloc(t,O,I,bits);
        if(t->fmt==0) st_read_f32_cap(&m->S,name,t->qf,(int64_t)O*I,drop);
        else { float *tmp=falloc((int64_t)O*I); st_read_f32_cap(&m->S,name,tmp,(int64_t)O*I,drop); qt_fill(t,tmp,bits); free(tmp); }
    }
}
static QT qt_load(Model *m, const char *name, int O, int I, int bits){
    QT t; memset(&t,0,sizeof(t)); qt_from_disk(m,name,O,I,bits,0,&t);
#ifdef COLI_CUDA
    if(g_cuda_enabled&&g_cuda_dense){
        t.cuda_eligible=1;
        int slot=g_cuda_rr++%g_cuda_ndev; t.cuda_device=g_cuda_devices[slot];
        g_cuda_dense_projected[slot]+=qt_bytes(&t);
    }
#endif
    return t;
}
static float *ld(Model *m, const char *name){   /* tensore 1D f32 residente (norme/bias) */
    int64_t n=st_numel(&m->S,name); if(n<0) st_die_missing(&m->S,name);
    float *p=(float*)qalloc((size_t)n*sizeof(float));   /* registrato per la GPU sotto METAL */
    st_read_f32(&m->S,name,p,0); return p;
}
#ifdef COLI_CUDA
static void qt_cuda_colocate(QT *dst,const QT *src){
    if(!g_cuda_enabled||!g_cuda_dense||!dst->cuda_eligible||!src->cuda_eligible||
       dst->cuda_device==src->cuda_device)return;
    int old=-1,now=-1;for(int i=0;i<g_cuda_ndev;i++){
        if(g_cuda_devices[i]==dst->cuda_device)old=i;if(g_cuda_devices[i]==src->cuda_device)now=i;
    }
    if(old>=0)g_cuda_dense_projected[old]-=qt_bytes(dst);
    if(now>=0)g_cuda_dense_projected[now]+=qt_bytes(dst);
    dst->cuda_device=src->cuda_device;
}
static void layer_cuda_shard_kvb(Layer *l,int H,int Q,int V){
    if(!g_cuda_enabled||!g_cuda_dense||g_cuda_ndev<2||l->kv_b.fmt==0)return;
    int rb=l->kv_b.fmt==1?l->kv_b.I:
           (l->kv_b.fmt==2||l->kv_b.fmt==4)?(l->kv_b.I+1)/2:(l->kv_b.I+3)/4;
    const uint8_t *weights=l->kv_b.fmt==1?(const uint8_t*)l->kv_b.q8:l->kv_b.q4;
    for(int d=0,h0=0;d<g_cuda_ndev;d++){
        int hn=H/g_cuda_ndev+(d<H%g_cuda_ndev),rows=hn*(Q+V);
        const void *part=weights+(int64_t)h0*(Q+V)*rb;
        const float *scale=l->kv_b.s+(int64_t)h0*(Q+V)*(l->kv_b.gs>0?(l->kv_b.I+l->kv_b.gs-1)/l->kv_b.gs:1);
        if(!coli_cuda_tensor_upload_g(&l->kv_b_shard[d],part,scale,l->kv_b.fmt,l->kv_b.I,rows,g_cuda_devices[d],l->kv_b.gs))return;
        l->shard_h0[d]=h0;l->shard_hn[d]=hn;l->n_kv_b_shard++;h0+=hn;
    }
    int old=-1;for(int i=0;i<g_cuda_ndev;i++)if(g_cuda_devices[i]==l->kv_b.cuda_device)old=i;
    if(old>=0)g_cuda_dense_projected[old]-=qt_bytes(&l->kv_b);
    l->kv_b.cuda_eligible=0;
}
#endif

/* One bit per weight tensor the fused Metal decode kernels bind, as reported by
 * metal_fused_layer_fmt_miss() (defined next to metal_fused_fmt_ok below — the single
 * per-layer format predicate both fused gates AND metal_fmt_gate_notice consume). The
 * two masks name exactly which of the 8 each fused entry binds. */
enum {
    METAL_FUSED_KV_B    = 1<<0,   /* the gates' own kv_b term: fmt==2, or fmt==4 with g_moe_exact off */
    METAL_FUSED_Q_A     = 1<<1,   /* the rest: metal_fused_fmt_ok() allowlist */
    METAL_FUSED_Q_B     = 1<<2,
    METAL_FUSED_KV_A    = 1<<3,
    METAL_FUSED_O       = 1<<4,
    METAL_FUSED_SH_GATE = 1<<5,   /* sh_*: checked on sparse layers only — dense layers */
    METAL_FUSED_SH_UP   = 1<<6,   /* never load them (fmt stays 0) and never reach the */
    METAL_FUSED_SH_DOWN = 1<<7,   /* fused layer CB, so a miss there would be noise */
};
#define METAL_FUSED_ATTN_TENSORS  (METAL_FUSED_KV_B|METAL_FUSED_Q_A|METAL_FUSED_Q_B|METAL_FUSED_KV_A|METAL_FUSED_O)
#define METAL_FUSED_LAYER_TENSORS (METAL_FUSED_ATTN_TENSORS|METAL_FUSED_SH_GATE|METAL_FUSED_SH_UP|METAL_FUSED_SH_DOWN)
static unsigned metal_fused_layer_fmt_miss(const Layer *l);

/* FUSED-METAL FORMAT-GATE NOTICE (#kvb, widened): kernel contract — both fused Metal
 * decode entries (attention_rows' coli_metal_attn_decode and layer_forward_rows'
 * coli_metal_layer_decode) gate per layer on metal_fused_layer_fmt_miss() over the weight
 * tensors they bind: kv_b_proj on the gates' own two-format+mode term (int4 per-row
 * fmt==2 always; grouped fmt==4 only while g_moe_exact is off — #587's grouped-int4 kv_b
 * is mode-gated), the rest on the metal_fused_fmt_ok() allowlist (fmt 1/2/3/4). Any miss
 * silently drops that layer off the fused path onto the CPU implementation, with no
 * signal anywhere. v1-class mixed-precision containers can mint any of these tensors at
 * an off-allowlist format, so print it once per offending tensor KIND (never per layer —
 * bounded at 8 lines) from model_init, after all layer tensors are resolved. The
 * condition is the gates' own predicate, so notice and gates cannot drift apart again
 * (the kv_b-only notice this replaces re-typed the gate condition and had already gone
 * stale: it missed the `!g_moe_exact` term, staying silent on fmt=4 kv_b under
 * COLI_METAL_MOE_EXACT=1 while both gates closed). NOTICE ONLY — no gate/behavior change
 * here. Main layers only (m->L): the MTP head (m->mtpL) is deliberately excluded by
 * ratified decision — common containers ship it at INT8 by design, so its kv_b closes
 * the fused attention gate on the MTP row on every ordinary load (the gate DOES evaluate
 * the MTP head; that CPU fallback is carried, correct behavior) and a line about it here
 * would be noise on every load, not signal.
 * FORMAT-ONLY SEMANTICS: a line here names a FORMAT (or format-mode) obstacle and
 * nothing else — it does not claim the fused path would otherwise engage. The gates
 * carry further runtime preconditions (GLM-5.2 dims, batch shape/S, ragged-mux guard,
 * expert config) that this notice deliberately does not duplicate: those are properties
 * of the call, not of the container, and re-stating them here would recreate exactly the
 * duplicated-condition drift the shared predicate exists to kill. On a container/arch
 * the fused path could never serve anyway, an off-allowlist tensor still gets its line —
 * the format is still worth curing before it bites on a config that DOES reach the
 * gates. sh_* lines count over the SPARSE-layer population (the only layers that load
 * them); every other kind counts over all main layers. */
static void metal_fmt_gate_notice(Model *m){
    Cfg *c=&m->c;
    if(!g_metal_enabled) return;
    enum { NK=8 };
    int nbad[NK]={0}, ffmt[NK]={-1,-1,-1,-1,-1,-1,-1,-1}, nsparse=0;
    for(int i=0;i<c->n_layers;i++){
        Layer *l=&m->L[i];
        if(l->sparse) nsparse++;
        unsigned miss=metal_fused_layer_fmt_miss(l);
        if(!miss) continue;
        const int f[NK]={ l->kv_b.fmt, l->q_a.fmt, l->q_b.fmt, l->kv_a.fmt,
                          l->o.fmt, l->sh_gate.fmt, l->sh_up.fmt, l->sh_down.fmt };
        for(int k=0;k<NK;k++) if(miss>>k & 1u){ nbad[k]++; if(ffmt[k]<0) ffmt[k]=f[k]; }
    }
    /* The remedy is MODE-AWARE because the plain suggestion is circular under MOE-exact:
     * the converter's --group-size defaults to 64 (community-validated, see
     * c/tools/convert_fp8_to_int4.py), so `--kvb-bits 4` alone mints GROUPED int4 =
     * fmt=4 — which is exactly what g_moe_exact keeps off the fused path. Under the
     * mode, point at the two real cures: an ungrouped requant (--group-size 0 -> fmt=2)
     * or, for a kv_b that is already fmt=4, unsetting COLI_METAL_MOE_EXACT. */
    if(nbad[0]) fprintf(stderr,
        "[METAL] kv_b_proj is fmt=%d on %d/%d layer%s: the fused Metal attention path "
        "serves kv_b only as int4, per-row (fmt=2) or grouped (fmt=4; grouped is served "
        "only while COLI_METAL_MOE_EXACT is off), so those layers run decode attention "
        "on the CPU absorb path instead. %s\n",
        ffmt[0], nbad[0], c->n_layers, nbad[0]==1?"":"s",
        g_moe_exact
          ? "Requantize kv_b to ungrouped int4 (--kvb-bits 4 --group-size 0) to re-enable "
            "the fused path; for a grouped-int4 (fmt=4) kv_b, unsetting COLI_METAL_MOE_EXACT "
            "also re-enables it without requantizing."
          : "Requantize kv_b to int4 (--kvb-bits 4) to re-enable the fused path.");
    static const char *const kind_name[NK]={ "kv_b_proj" /* [0] has its own line above */,
        "q_a_proj","q_b_proj","kv_a_proj_with_mqa","o_proj",
        "shared_experts.gate_proj","shared_experts.up_proj","shared_experts.down_proj" };
    for(int k=1;k<NK;k++) if(nbad[k]) fprintf(stderr,
        "[METAL] %s is fmt=%d on %d/%d %slayer%s: outside the fused-path allowlist "
        "(fmt 1/2/3/4), so the fused Metal decode kernels that bind this tensor skip "
        "those layers and they run on the CPU path instead.\n",
        kind_name[k], ffmt[k], nbad[k],
        k>=5?nsparse:c->n_layers,        /* sh_* (k>=5): only sparse layers load them */
        k>=5?"sparse ":"", nbad[k]==1?"":"s");
}

static void model_init_range(Model *m, const char *snap, int cap,
                             int ebits, int dbits, int layer_begin,
                             int layer_end, int load_boundaries, int load_mtp,
                             int init_telemetry){
    memset(m,0,sizeof(*m)); m->ebits=ebits; m->dbits=dbits;
    load_cfg(&m->c,snap);
    { const char *xd=getenv("COLI_MODEL_DIRS");        /* SPLIT: model shards spread across N drives */
      st_init_multi(&m->S,snap,(xd&&*xd)?xd:NULL); }
    Cfg *c=&m->c; char nm[256]; int H=c->n_heads, D=c->hidden;
    if(layer_end==0) layer_end=c->n_layers;
    if(layer_begin<0||layer_begin>=layer_end||layer_end>c->n_layers){
        fprintf(stderr,"invalid GLM layer range [%d,%d) for %d layers\n",
                layer_begin,layer_end,c->n_layers); exit(1); }
    /* embed e lm_head sono il confine I/O: tenerli ad alta precisione (come i quant dynamic
     * reali). A bf16 ~1.9GB su GLM reale: trascurabile. dbits>=8 -> qui f32; piu' basso -> dbits. */
    int io_bits = dbits>=8 ? 16 : dbits;
    if(load_boundaries){
        m->embed   = qt_load(m,"model.embed_tokens.weight", c->vocab, D, io_bits);
        m->lm_head = qt_load(m,"lm_head.weight", c->vocab, D, io_bits);
        m->final_norm = ld(m,"model.norm.weight");
    }
    m->L=calloc(c->n_layers,sizeof(Layer));
    int NR=c->n_layers+1;                        /* +1: riga del layer MTP */
    m->ecap=cap; m->ecache=calloc(NR,sizeof(ESlot*)); m->ecn=calloc(NR,sizeof(int));
    m->ecache_slot_by_expert=calloc(NR,sizeof(int*));
    m->kv_dev_L=calloc(NR,sizeof(float*)); m->kv_dev_R=calloc(NR,sizeof(float*));
    m->kv_dev_valid=calloc(NR,sizeof(int));
#ifdef COLI_VULKAN
    m->vk_kv_valid=calloc(NR,sizeof(int));
#endif
    m->eroute=calloc(NR,sizeof(int*)); m->enr=calloc(NR,sizeof(int));
    m->pin=calloc(NR,sizeof(ESlot*)); m->npin=calloc(NR,sizeof(int));
    m->pin_slot_by_expert=calloc(NR,sizeof(int*));
    if(!m->ecache_slot_by_expert||!m->pin_slot_by_expert){ fprintf(stderr,"OOM expert cache index\n"); exit(1); }
    for(int i=0;i<NR;i++){
        m->ecache_slot_by_expert[i]=malloc((size_t)c->n_experts*sizeof(int));
        m->pin_slot_by_expert[i]=malloc((size_t)c->n_experts*sizeof(int));
        if(!m->ecache_slot_by_expert[i]||!m->pin_slot_by_expert[i]){ fprintf(stderr,"OOM expert cache index\n"); exit(1); }
        for(int e=0;e<c->n_experts;e++){
            m->ecache_slot_by_expert[i][e]=-1;
            m->pin_slot_by_expert[i][e]=-1;
        }
    }
    if(init_telemetry){
        rt_init("glm_moe_dsa",c->n_layers,c->n_experts); /* owns counters + format */
        m->eusage=rt_counts_all();                       /* bump sites keep their shape */
    }else{
        /* Segment engines may coexist in one process. Do not let their
         * telemetry aliases overwrite route_trace.h's process-global owner. */
        m->eusage=calloc((size_t)NR,sizeof(*m->eusage));
    }
    m->eheat=calloc(NR,sizeof(uint32_t*));
    m->elast=calloc(NR,sizeof(uint32_t*));
    m->elast_dc=calloc(NR,sizeof(uint32_t*)); m->elast_pre=calloc(NR,sizeof(uint32_t*));
    m->kv=calloc(1,sizeof(KVState));
    m->kv_start=m->kv->kv_start=calloc(NR,sizeof(int));
    for(int i=layer_begin;i<layer_end;i++){
        Layer *l=&m->L[i];
        #define P(s) (snprintf(nm,sizeof(nm),"model.layers.%d." s,i),nm)
        l->in_ln=ld(m,P("input_layernorm.weight"));
        l->post_ln=ld(m,P("post_attention_layernorm.weight"));
        l->q_a   = qt_load(m,P("self_attn.q_a_proj.weight"), c->q_lora, D, dbits);
        l->q_a_ln= ld(m,P("self_attn.q_a_layernorm.weight"));
        l->q_b   = qt_load(m,P("self_attn.q_b_proj.weight"), H*c->qk_head, c->q_lora, dbits);
        l->kv_a  = qt_load(m,P("self_attn.kv_a_proj_with_mqa.weight"), c->kv_lora+c->qk_rope, D, dbits);
        l->kv_a_ln= ld(m,P("self_attn.kv_a_layernorm.weight"));
        l->kv_b  = qt_load(m,P("self_attn.kv_b_proj.weight"), H*(c->qk_nope+c->v_head), c->kv_lora, dbits);
        l->o     = qt_load(m,P("self_attn.o_proj.weight"), D, H*c->v_head, dbits);
#ifdef COLI_CUDA
        qt_cuda_colocate(&l->o,&l->kv_b);
        qt_cuda_colocate(&l->q_a,&l->kv_b);   /* PIPE: intera catena attention sulla */
        qt_cuda_colocate(&l->q_b,&l->kv_b);   /* stessa scheda / whole attention chain */
        qt_cuda_colocate(&l->kv_a,&l->kv_b);  /* on the layer home device */
        if(getenv("COLI_CUDA_ATTN_SHARD")&&atoi(getenv("COLI_CUDA_ATTN_SHARD")))
            layer_cuda_shard_kvb(l,H,c->qk_nope,c->v_head);
#endif
        l->sparse = (i >= c->first_dense);
        if(!l->sparse){
            l->gate_proj = qt_load(m,P("mlp.gate_proj.weight"), c->dense_inter, D, dbits);
            l->up_proj   = qt_load(m,P("mlp.up_proj.weight"),   c->dense_inter, D, dbits);
            l->down_proj = qt_load(m,P("mlp.down_proj.weight"), D, c->dense_inter, dbits);
            qt_planarize(&l->gate_proj); qt_planarize(&l->up_proj); qt_planarize(&l->down_proj);   /* K1 */
        } else {
            l->router=ld(m,P("mlp.gate.weight"));
            l->router_bias=ld(m,P("mlp.gate.e_score_correction_bias"));
            int sI=c->moe_inter*c->n_shared;
            l->sh_gate = qt_load(m,P("mlp.shared_experts.gate_proj.weight"), sI, D, dbits);
            l->sh_up   = qt_load(m,P("mlp.shared_experts.up_proj.weight"),   sI, D, dbits);
            l->sh_down = qt_load(m,P("mlp.shared_experts.down_proj.weight"), D, sI, dbits);
            qt_planarize(&l->sh_gate); qt_planarize(&l->sh_up); qt_planarize(&l->sh_down);   /* K1 */
#ifdef COLI_CUDA
            qt_cuda_colocate(&l->sh_gate,&l->kv_b);  /* PIPE2: shared chain on the layer home device */
            qt_cuda_colocate(&l->sh_up,&l->sh_gate);
            qt_cuda_colocate(&l->sh_down,&l->sh_gate);
#endif
            m->ecache[i]=calloc(cap,sizeof(ESlot));
            m->eroute[i]=calloc(c->topk,sizeof(int));      /* metodo C: ultimo routing del layer */
            m->eheat[i]=calloc(c->n_experts,sizeof(uint32_t));
            m->elast[i]=calloc(c->n_experts,sizeof(uint32_t));
            m->elast_dc[i]=calloc(c->n_experts,sizeof(uint32_t));
            m->elast_pre[i]=calloc(c->n_experts,sizeof(uint32_t));
        }
        #undef P
    }
    metal_fmt_gate_notice(m);                     /* once per load, after all layer tensors resolved */
    /* testa MTP (layer n_layers): presente solo se convertita con --mtp */
    if(load_mtp){
        /* MTP attiva SOLO se il set e' COMPLETO (i tensori vivono su 3 shard: durante la
         * conversione parziale ne esiste solo una parte). MTP=0 la disabilita comunque. */
        const char *req[]={"eh_proj.weight","enorm.weight","hnorm.weight","shared_head.norm.weight",
            "input_layernorm.weight","post_attention_layernorm.weight",
            "self_attn.q_a_proj.weight","self_attn.q_b_proj.weight","self_attn.kv_a_proj_with_mqa.weight",
            "self_attn.kv_b_proj.weight","self_attn.o_proj.weight","mlp.gate.weight",
            "mlp.shared_experts.gate_proj.weight","mlp.shared_experts.down_proj.weight",
            "mlp.experts.0.gate_proj.weight"};
        char mn[256]; m->has_mtp=1;
        for(unsigned q=0;q<sizeof(req)/sizeof(req[0]);q++){
            snprintf(mn,sizeof(mn),"model.layers.%d.%s",c->n_layers,req[q]);
            if(!st_has(&m->S,mn)){ m->has_mtp=0; break; }
        }
        /* probe the LAST expert by index, not a fixed 255: REAP-pruned
         * checkpoints have n_routed_experts < 256 and the MTP set stays complete,
         * so a hardcoded expert.255 would spuriously report has_mtp=0 on them. */
        snprintf(mn,sizeof(mn),"model.layers.%d.mlp.experts.%d.down_proj.weight",c->n_layers,c->n_experts-1);
        if(!st_has(&m->S,mn)) m->has_mtp=0;
        if(getenv("MTP") && atoi(getenv("MTP"))==0) m->has_mtp=0;
        if(m->has_mtp){
            int i=c->n_layers; Layer *l=&m->mtpL;
            #define PM(s) (snprintf(nm,sizeof(nm),"model.layers.%d." s,i),nm)
            l->in_ln=ld(m,PM("input_layernorm.weight"));
            l->post_ln=ld(m,PM("post_attention_layernorm.weight"));
            l->q_a   = qt_load(m,PM("self_attn.q_a_proj.weight"), c->q_lora, D, dbits);
            l->q_a_ln= ld(m,PM("self_attn.q_a_layernorm.weight"));
            l->q_b   = qt_load(m,PM("self_attn.q_b_proj.weight"), H*c->qk_head, c->q_lora, dbits);
            l->kv_a  = qt_load(m,PM("self_attn.kv_a_proj_with_mqa.weight"), c->kv_lora+c->qk_rope, D, dbits);
            l->kv_a_ln= ld(m,PM("self_attn.kv_a_layernorm.weight"));
            l->kv_b  = qt_load(m,PM("self_attn.kv_b_proj.weight"), H*(c->qk_nope+c->v_head), c->kv_lora, dbits);
            l->o     = qt_load(m,PM("self_attn.o_proj.weight"), D, H*c->v_head, dbits);
            l->sparse=1;
            l->router=ld(m,PM("mlp.gate.weight"));
            l->router_bias=ld(m,PM("mlp.gate.e_score_correction_bias"));
            int sI=c->moe_inter*c->n_shared;
            l->sh_gate = qt_load(m,PM("mlp.shared_experts.gate_proj.weight"), sI, D, dbits);
            l->sh_up   = qt_load(m,PM("mlp.shared_experts.up_proj.weight"),   sI, D, dbits);
            l->sh_down = qt_load(m,PM("mlp.shared_experts.down_proj.weight"), D, sI, dbits);
            m->eh_proj = qt_load(m,PM("eh_proj.weight"), D, 2*D, dbits);
            m->enorm=ld(m,PM("enorm.weight")); m->hnorm=ld(m,PM("hnorm.weight"));
            m->mtp_norm=ld(m,PM("shared_head.norm.weight"));
            m->ecache[i]=calloc(cap,sizeof(ESlot));
            m->eroute[i]=calloc(c->topk,sizeof(int));
            m->eheat[i]=calloc(c->n_experts,sizeof(uint32_t));
            m->elast[i]=calloc(c->n_experts,sizeof(uint32_t));
            m->elast_dc[i]=calloc(c->n_experts,sizeof(uint32_t));
            m->elast_pre[i]=calloc(c->n_experts,sizeof(uint32_t));
            m->kv_start[i]=-1;                    /* KV MTP: parte dalla prima posizione di decode */
            #undef PM
        }
    }
    /* DSA lightning indexer: attivo SOLO se i pesi (conversione --indexer) ci sono per
     * TUTTI i layer full. Auto-rilevamento come per MTP: niente flag, niente passi extra. */
    {
        m->has_dsa = (c->index_topk>0 && c->index_nh>0 && c->index_hd>0 && c->index_hd<=256);
        char inm[300];
        for(int i=layer_begin;i<layer_end && m->has_dsa;i++){
            if(!c->idx_type[i]) continue;
            snprintf(inm,sizeof(inm),"model.layers.%d.self_attn.indexer.wq_b.weight",i);
            if(!st_has(&m->S,inm)) m->has_dsa=0;
        }
        if(getenv("DSA") && atoi(getenv("DSA"))==0) m->has_dsa=0;
        if(m->has_dsa){
            m->ix_wq=calloc(c->n_layers,sizeof(QT)); m->ix_wk=calloc(c->n_layers,sizeof(QT));
            m->ix_wp=calloc(c->n_layers,sizeof(QT));
            m->ix_knw=calloc(c->n_layers,sizeof(float*)); m->ix_knb=calloc(c->n_layers,sizeof(float*));
            for(int i=layer_begin;i<layer_end;i++){
                if(!c->idx_type[i]) continue;
                #define PI(s) (snprintf(nm,sizeof(nm),"model.layers.%d.self_attn.indexer." s,i),nm)
                m->ix_wq[i]=qt_load(m,PI("wq_b.weight"), c->index_nh*c->index_hd, c->q_lora, dbits);
                m->ix_wk[i]=qt_load(m,PI("wk.weight"), c->index_hd, D, dbits);
                m->ix_wp[i]=qt_load(m,PI("weights_proj.weight"), c->index_nh, D, dbits);
                m->ix_knw[i]=ld(m,PI("k_norm.weight")); m->ix_knb[i]=ld(m,PI("k_norm.bias"));
                #undef PI
            }
            fprintf(stderr,"[DSA] indexer active: top-%d sparse attention beyond %d context tokens\n",
                c->index_topk, c->index_topk);
        }
    }
    if(load_boundaries){
        m->hlast=falloc(D); m->h_all=falloc((int64_t)512*D);
    }

    /* byte della parte DENSA residente (embed+lm_head+attn+mlp densa+shared+norme) */
    int64_t rb=load_boundaries?(qt_bytes(&m->embed)+qt_bytes(&m->lm_head)):0;
    for(int i=layer_begin;i<layer_end;i++){ Layer *l=&m->L[i];
        rb+=qt_bytes(&l->q_a)+qt_bytes(&l->q_b)+qt_bytes(&l->kv_a)+qt_bytes(&l->kv_b)+qt_bytes(&l->o);
        if(!l->sparse) rb+=qt_bytes(&l->gate_proj)+qt_bytes(&l->up_proj)+qt_bytes(&l->down_proj);
        else rb+=qt_bytes(&l->sh_gate)+qt_bytes(&l->sh_up)+qt_bytes(&l->sh_down);
    }
    if(m->has_mtp){ Layer *l=&m->mtpL;
        rb+=qt_bytes(&l->q_a)+qt_bytes(&l->q_b)+qt_bytes(&l->kv_a)+qt_bytes(&l->kv_b)+qt_bytes(&l->o);
        rb+=qt_bytes(&l->sh_gate)+qt_bytes(&l->sh_up)+qt_bytes(&l->sh_down)+qt_bytes(&m->eh_proj);
    }
    if(m->has_dsa) for(int i=layer_begin;i<layer_end;i++) if(c->idx_type[i])
        rb+=qt_bytes(&m->ix_wq[i])+qt_bytes(&m->ix_wk[i])+qt_bytes(&m->ix_wp[i]);
    /* Dense layers, and the MTP row when the model has none, do not route and so get no
     * counter row -- the exact shape eusage had before route_trace.h owned it. rt_init
     * cannot know this: sparsity is settled while the layers are built, above. A row is
     * the load admission rule, so dropping these is what keeps a history record naming a
     * dense layer from being absorbed and written back out. */
    if(init_telemetry){
        for(int i=0;i<c->n_layers;i++)
            if(i<layer_begin||i>=layer_end||!m->L[i].sparse) rt_drop_row(i);
        if(!m->has_mtp) rt_drop_row(c->n_layers);
    }
    m->resident_bytes=rb;
}

static void model_init(Model *m, const char *snap, int cap,
                       int ebits, int dbits){
    model_init_range(m,snap,cap,ebits,dbits,0,0,1,1,1);
}

/* embed: dequantizza la riga del token (scala per-riga) in x[hidden] */
static void embed_row(Model *m, int tok, float *x){
    int D=m->c.hidden; QT *e=&m->embed;
    if(tok<0 || tok>=e->O){ memset(x,0,(size_t)D*sizeof(float)); return; }   /* #SEC-5: out-of-range token id -> zero row, never OOB */
    if(e->fmt==0){ memcpy(x, e->qf+(int64_t)tok*D, D*sizeof(float)); return; }
    if(e->fmt==4){ /* grouped int4: per-group scale (embed/lm_head at io_bits, usually fmt 0/1) */
        const uint8_t *q=e->q4+(int64_t)tok*((D+1)/2); int gs=e->gs,ng=(D+gs-1)/gs;
        const float *scl=e->s+(int64_t)tok*ng;
        for(int g=0;g*gs<D;g++){ int base=g*gs,glen=gs; if(base+glen>D)glen=D-base; float s=scl[g];
            for(int i=base;i+1<base+glen;i+=2){ uint8_t byte=q[i>>1]; x[i]=(float)((int)(byte&0xF)-8)*s;
                x[i+1]=(float)((int)(byte>>4)-8)*s; }
            if(glen&1){ uint8_t byte=q[(base+glen-1)>>1]; x[base+glen-1]=(float)((int)(byte&0xF)-8)*s; } }
        return; }
    if(e->fmt==1){ const int8_t *q=e->q8+(int64_t)tok*D; float s=e->s[tok];
        for(int i=0;i<D;i++) x[i]=(float)q[i]*s; return; }
    if(e->fmt==2){ const uint8_t *q=e->q4+(int64_t)tok*((D+1)/2); float s=e->s[tok];   /* int4 */
        for(int i=0;i<D;i+=2){ uint8_t byte=q[i>>1]; x[i]=(float)((int)(byte&0xF)-8)*s;
            if(i+1<D) x[i+1]=(float)((int)(byte>>4)-8)*s; }
        return; }
    if(e->fmt==5){ const uint8_t *q=e->q4+(int64_t)tok*i3_rowbytes(D);   /* int3-g64 */
        const float *sr=e->s+(int64_t)tok*i3_groups(D); int64_t ng=i3_groups(D);
        for(int64_t g=0; g<ng; g++){ const uint8_t *lo=q+g*I3_GBYTES, *hi=lo+16;
            int base=(int)(g*I3_GROUP), n=D-base<I3_GROUP?D-base:I3_GROUP;
            for(int k=0;k<n;k++){ unsigned u=((lo[k>>2]>>((k&3)*2))&3)|(((hi[k>>3]>>(k&7))&1)<<2);
                x[base+k]=(float)((int)u-4)*sr[g]; } }
        return; }
    const uint8_t *q=e->q4+(int64_t)tok*((D+3)/4); float s=e->s[tok];   /* int2 */
    for(int i=0;i<D;i++){ uint8_t byte=q[i>>2]; int sh=(i&3)*2; x[i]=(float)((int)((byte>>sh)&3)-2)*s; }
}

/* COLI_MMAP=1: gli expert diventano VISTE dentro mmap dei file safetensors (niente pread,
 * niente slab, niente copia: la page cache del kernel E' la cache). Le mappe sono
 * registrate con Metal (newBufferWithBytesNoCopy su pagine file-backed, come llama.cpp),
 * quindi la GPU legge gli stessi byte. Fallback allo slab path su disallineamento. */
static int g_mmap=0;
static struct { int fd; void *base; size_t len; } g_maps[512]; static int g_nmaps;
static pthread_mutex_t g_map_mtx = PTHREAD_MUTEX_INITIALIZER;   /* expert_load e' OMP-parallel */
/* forward decls: mem_should_wire/mem_wire live near pin_wire() further down, but
 * qt_wire_mmap() (also further down, used by pin_wire()'s COLI_MMAP path) needs
 * them declared before its own definition. Real mlock-ing of mmap'd pinned
 * experts happens there, not in expert_load() -- see qt_wire_mmap() for why. */
static int mem_should_wire(void);
static int mem_wire(void *addr, size_t len);
static void qt_unwire_mmap(QT *t);   /* def. presso pin_wire / defined near pin_wire */
static int64_t g_mmap_wired=0; static long g_mmap_wire_failed=0;
static void *map_of_fd(int fd){
    pthread_mutex_lock(&g_map_mtx);
    for(int i=0;i<g_nmaps;i++) if(g_maps[i].fd==fd){ void *b=g_maps[i].base; pthread_mutex_unlock(&g_map_mtx); return b; }
    void *base=NULL;
#if defined(__APPLE__) || defined(__linux__) || defined(__FreeBSD__)
    struct stat st;
    if(g_nmaps<512 && fstat(fd,&st)==0){
        size_t len=((size_t)st.st_size+16383)&~(size_t)16383;
        void *p=mmap(NULL,len,PROT_READ,MAP_SHARED,fd,0);
        if(p!=MAP_FAILED){
            base=p; g_maps[g_nmaps].fd=fd; g_maps[g_nmaps].base=p; g_maps[g_nmaps].len=len; g_nmaps++;
#ifdef COLI_METAL
            if(g_metal_enabled) coli_metal_register(p,len);
#endif
        }
    }
#endif
    pthread_mutex_unlock(&g_map_mtx);
    return base;
}

/* ==================== MULTI-SSD: N model copies, N drives ====================
 * COLI_MODEL_MIRROR=<dir>[;<dir>...] registers additional read-only copies of
 * the model on other drives; expert reads split across all copies according to
 * COLI_DISK_WEIGHTS=<primary>,<mirror>[,<mirror2>...] (relative bandwidth;
 * without the env it is measured at startup with the engine's own access
 * pattern). Cold decode is disk-bound (~11 GB/token): N NVMe drives reading in
 * parallel add up. */
#define MIR_REPS (1+ST_MAX_MIR)        /* replicas incl. the primary */
static const char *g_mirror_dir=NULL;  /* COLI_MODEL_MIRROR / SNAP_MIRROR (dir list) */
static int g_mirror=0;                 /* 1 = mirror active (at least one shard accepted) */
static int g_mir_nrep=1;               /* replicas incl. the primary */
static int g_mir_cut[MIR_REPS]={256};  /* cumulative hash cuts of 256: replica r serves h in [cut[r-1],cut[r]) */
static _Atomic int64_t g_mir_bytes[MIR_REPS]; /* bytes served per drive: [0] primary, [r] mirror r */
static _Atomic int64_t g_mir_nread[MIR_REPS];

/* replica of one expert: DETERMINISTIC hash of (layer,eid). Determinism is a
 * requirement, not a style choice: the readahead/PILOT WILLNEED and the demand
 * pread must hit the same fd/page-cache, and in buffered mode an expert must
 * never be cached twice (one copy per drive). */
static inline int expert_route(int layer,int eid){
    if(!g_mirror) return 0;
    uint32_t h=(uint32_t)layer*2654435761u ^ (uint32_t)eid*0x9E3779B9u;
    h^=h>>16; h*=0x45d9f3bu; h^=h>>16;
    int hv=(int)(h&255), r=0;
    while(hv>=g_mir_cut[r]) r++;       /* cut[nrep-1]==256 terminates the scan */
    return r;
}

/* buffered fd of the replica, falling back to the primary if the file is not mirrored */
static inline int rep_bfd(shards *S,int fd,int rep){
    int r=st_fd_rep(S,fd,rep); return r<0?fd:r;
}

/* MULTI-SSD striped demand read (COLI_MIR_STRIPE=0 disables): split the one
 * coalesced O_DIRECT expert pread into disjoint 4K-aligned chunks, one per
 * replica that holds the shard, read in parallel. A cold ~19 MB expert read is
 * single-thread latency-bound (~4 GB/s on one NVMe) and it is the felt wait of
 * every demand miss; N drives reading stripes of the SAME expert cut it ~N-fold.
 * O_DIRECT only: no page cache involved, so striping cannot double-cache an
 * expert across drives (the buffered path keeps whole-expert replica routing).
 * Chunk 0 starts on the ROUTED replica so the hash still spreads first-chunk
 * load evenly. Any short stripe fails the whole attempt (caller falls back to
 * the single-replica read). */
static int g_mir_stripe=1;
typedef struct { int fd; char *buf; int64_t len, off; ssize_t r; } MirStripe;
static void *mir_stripe_worker(void *a){
    MirStripe *st=(MirStripe*)a;
    st->r = pread(st->fd, st->buf, (size_t)st->len, st->off);
    return NULL;
}
/* Returns total bytes read (== len) on success, -1 to make the caller fall back. */
static int64_t mir_pread_striped(shards *S,int fd,int rep,char *buf,int64_t len,int64_t base){
    if(!g_mir_stripe || g_mir_nrep<2 || len < (4<<20)) return -1;
    int sfd[MIR_REPS], srep[MIR_REPS], nsf=0;
    for(int r=0;r<g_mir_nrep && r<MIR_REPS;r++){
        int f=st_direct_fd_rep(S,fd,r);
        if(f>=0){ sfd[nsf]=f; srep[nsf]=r; nsf++; }
    }
    if(nsf<2) return -1;
    int64_t chunk=((len+nsf-1)/nsf + 4095) & ~4095LL;
    MirStripe st[MIR_REPS]; pthread_t th[MIR_REPS]; int nth=0, ns=0;
    for(int i=0;i<nsf;i++){
        int64_t o=(int64_t)i*chunk; if(o>=len) break;
        int k=(rep+i)%nsf;                    /* chunk 0 on the routed replica */
        st[ns]=(MirStripe){sfd[k], buf+o, len-o<chunk?len-o:chunk, base+o, -1};
        ns++;
    }
    for(int i=1;i<ns;i++)
        if(pthread_create(&th[nth],NULL,mir_stripe_worker,&st[i])==0) nth++;
        else st[i].r=pread(st[i].fd,st[i].buf,(size_t)st[i].len,st[i].off);
    st[0].r = pread(st[0].fd, st[0].buf, (size_t)st[0].len, st[0].off);
    for(int i=0;i<nth;i++) pthread_join(th[i],NULL);
    for(int i=0;i<ns;i++) if(st[i].r!=st[i].len) return -1;
    for(int i=0;i<ns;i++){
        int k=(rep+i)%nsf;
        atomic_fetch_add_explicit(&g_mir_bytes[srep[k]],st[i].len,memory_order_relaxed);
        atomic_fetch_add_explicit(&g_mir_nread[srep[k]],1,memory_order_relaxed);
    }
    return len;
}

static int pread_full(int fd, void *buf, int64_t n, int64_t off, const char *tag);

/* pread on the chosen replica with fallback to the primary on error/short-read:
 * an unreadable sector (or an unmount) of the mirror must never kill the process
 * when the primary can serve the same bytes. Delegates to pread_full so the
 * mirror path inherits the short-read/EINTR loop and honest reporting (#236).
 * Accounts bytes per drive. Returns 0 = ok, -1 = real error/EOF (like pread_full). */
static ssize_t mir_pread(shards *S,int fd,int rep,void *buf,int64_t n,int64_t off,const char *tag){
    int rfd = st_fd_rep(S,fd,rep);
    int used = (rep && rfd>=0) ? rep : 0;
    if(rfd<0) rfd=fd;
    int rc=pread_full(rfd,buf,n,off,tag);
    if(rc && used){
        static _Atomic int warned;
        if(!atomic_exchange(&warned,1))
            fprintf(stderr,"[MIRROR] read error on the mirror copy — falling back to the primary drive\n");
        used=0; rc=pread_full(fd,buf,n,off,tag);
    }
    if(!rc){ atomic_fetch_add_explicit(&g_mir_bytes[used],n,memory_order_relaxed);
             atomic_fetch_add_explicit(&g_mir_nread[used],1,memory_order_relaxed); }
    return rc;
}

/* carica un expert nello slot. Container pre-quantizzato: le 3 matrici sono contigue nel
 * file -> UNA pread coalescente da ~19 MB dentro `slab` (+ le scale in fslab); i QT sono
 * viste dentro lo slab (zero copie). Fallback per modelli non quantizzati (oracolo tiny).
 * THREAD-SAFE su slot distinti (pread posizionale, st_find read-only). */
/* Load one expert's weights into slot `s`. Returns 0 on success, -1 on failure.
 * fatal=1 (all main / on-demand / REPIN / pin callers): preserve the original
 * exit-on-error contract byte-for-byte — any missing tensor, OOM, short read or
 * pread error aborts the process. fatal=0 (speculative pilot only): the same
 * errors instead abandon the load and return -1 without touching s->eid, so a
 * mispredicted cross-layer prefetch can never kill the server. */
/* pread completo: gestisce le short-read (POSIX le ammette su file regolari
 * sotto pressione di memoria) e le EINTR, e riporta un errore ONESTO. perror
 * stampava "Success" quando pread ritorna un conteggio corto invece di -1
 * (errno resta 0 dalla syscall precedente) -> messaggio fuorviante nel path
 * score/bench (#236). Ritorna 0 = ok, -1 = errore reale o EOF. */
static int pread_full(int fd, void *buf, int64_t n, int64_t off, const char *tag){
    char *p=buf; int64_t got=0;
    while(got<n){
        ssize_t r=pread(fd, p+got, (size_t)(n-got), off+got);
        if(r<0){ if(errno==EINTR) continue;
#ifdef _WIN32
            fprintf(stderr,"%s: %s (off %lld, %lld/%lld bytes, WinErr=%lu)\n",tag,strerror(errno),
                    (long long)off,(long long)got,(long long)n,(unsigned long)compat_pread_lasterr);
#else
            fprintf(stderr,"%s: %s (off %lld, %lld/%lld bytes)\n",tag,strerror(errno),
                    (long long)off,(long long)got,(long long)n);
#endif
            return -1; }
        if(r==0){ fprintf(stderr,"%s: short read at EOF (off %lld, %lld/%lld bytes) — truncated shard?\n",
                    tag,(long long)off,(long long)got,(long long)n); return -1; }
        got+=r;
    }
    return 0;
}
/* DISK-CLASS: is classification active right now? Single point of truth shared by
 * moe()'s pre-bump snapshot (FASE A, below) and expert_load_impl's classify-and-read --
 * they must agree, or expert_load_impl reads a snapshot moe() never bothered to write.
 * PROF=1 is the only trigger: this feature is measurement-only, the verdict never picks
 * an fd (kept as one function anyway so a future policy consumer cannot drift out of
 * sync with the snapshot writer by construction). */
static int dc_needed(void){ return g_prof; }
/* DISK-CLASS: cold/warm verdict for ONE demand load, read from the pre-bump snapshot
 * (Model's elast_pre) so routing's OWN bump for THIS call can't contaminate the read --
 * prefill is one giant moe() call where every newly-seen expert gets its `last` bumped
 * a few lines above the load that made it "new"; classifying off the live, post-bump
 * array would call the whole cold burst warm. Ages against the PRIVATE clock
 * (eaccess_clock_dc, see its declaration in Model), NEVER the real eaccess_clock: kept
 * separate by design (see elast_dc in Model for why DISK-CLASS keeps its own clock
 * instead of reading the real one -- before #417/cfcc742 the real one also froze on the
 * Metal pre-routed decode path, which would have made every prefill-touched expert warm
 * forever and everything else cold forever; that's fixed upstream now, but DISK-CLASS
 * still doesn't read the real clock, for the isolation property, not the freeze).
 * Conservative-toward-warm on the one genuinely ambiguous input (missing snapshot --
 * defensive only, elast_pre is allocated everywhere elast is); a demonstrable first-ever
 * access (last_pre==0) is not a judgment call, it stays cold regardless of that bias. */
static int expert_classify(Model *m, int layer, int eid){
    if(!m->elast_pre || !m->elast_pre[layer]) return DC_WARM;  /* no snapshot: label as the safe class */
    uint32_t last_pre=m->elast_pre[layer][eid];
    if(last_pre==0) return DC_COLD;                             /* never touched before this call: certain cold */
    uint32_t age=m->eaccess_clock_dc-last_pre;                  /* ticks since last access, PRE this call's bump */
    return age>g_direct_heat_ticks ? DC_COLD : DC_WARM;         /* '>' not '>=': ties lean warm */
}
static int expert_load_impl(Model *m, int layer, int eid, ESlot *s, int fatal, int demand){
#ifdef COLI_CUDA
    /* A live REPIN may reuse a GPU-enabled pinned slot for a different expert.
     * Keep its tier assignment, but invalidate the old device weights. */
    if(s->eid!=eid){ qt_cuda_reset(&s->g); qt_cuda_reset(&s->u); qt_cuda_reset(&s->d); }
#endif
#ifdef COLI_VULKAN
    /* Slot reused for a different expert: free the stale VK-resident weights so the new
     * expert re-uploads instead of computing with the old expert's tensors. */
    if(s->eid!=eid){
        if(s->g.vk_eligible) g_vk_resident--;
        qt_vk_reset(&s->g); qt_vk_reset(&s->u); qt_vk_reset(&s->d);
    }
#endif
    Cfg *c=&m->c; int I=c->moe_inter, D=c->hidden, b=m->ebits;
    /* suf as a bounded char[][16] (not const char*) lets GCC prove the %s in the
     * nm[k]/qn snprintfs can't overflow: worst key is "model.layers.<i>.mlp.experts.<i>.down_proj.weight"
     * = 66 bytes incl NUL, well under nm[288] and qn[320]. See #484. */
    char nm[3][288]; const char suf[3][16]={"gate_proj","up_proj","down_proj"};
    for(int k=0;k<3;k++) snprintf(nm[k],sizeof(nm[k]),"model.layers.%d.mlp.experts.%d.%s.weight",layer,eid,suf[k]);
    char qn[320]; snprintf(qn,sizeof(qn),"%s.qs",nm[0]);
    if(!st_has(&m->S,qn)){                       /* fallback: tensori pieni, quantizza a runtime.
                                                  * Reachable ONLY for unquantized models (no .qs);
                                                  * GLM always has .qs, so the pilot never hits it. */
        qt_from_disk(m,nm[0],I,D,b,g_drop,&s->g);
        qt_from_disk(m,nm[1],I,D,b,g_drop,&s->u);
        qt_from_disk(m,nm[2],D,I,b,g_drop,&s->d);
        qt_planarize(&s->g); qt_planarize(&s->u); qt_planarize(&s->d);   /* K1 */
        atomic_fetch_add_explicit(&g_prof_io,
            st_nbytes(&m->S,nm[0])+st_nbytes(&m->S,nm[1])+st_nbytes(&m->S,nm[2]),memory_order_relaxed);
        s->eid=eid; return 0;
    }
    st_tensor *tw[3], *tq[3];
    for(int k=0;k<3;k++){
        tw[k]=st_find(&m->S,nm[k]);
        /* strnlen+memcpy come al load fmt-aware (#484): gcc -Wformat-truncation
         * vede nm[k] a indice variabile come l'intero nm[3][] e segnala 863>320 */
        { size_t n=strnlen(nm[k],sizeof(nm[k])); memcpy(qn,nm[k],n); memcpy(qn+n,".qs",4); }
        tq[k]=st_find(&m->S,qn);
        if(!tw[k]||!tq[k]){ if(fatal) st_die_missing(&m->S,nm[k]);   /* #586: diagnose, don't just name it */
                            fprintf(stderr,"missing %s\n",nm[k]); return -1; }
    }
    if(g_disk_split){ /* split load/byte per tipo layer; atomici: expert_load gira anche su OMP/pipe/pilot */
        int64_t tb=0; for(int k=0;k<3;k++) tb+=tw[k]->nbytes+tq[k]->nbytes;
        if(layer==c->n_layers){ __atomic_add_fetch(&m->ld_mtp,1,__ATOMIC_RELAXED);
                                __atomic_add_fetch(&m->bytes_mtp,(uint64_t)tb,__ATOMIC_RELAXED); }
        else                  { __atomic_add_fetch(&m->ld_main,1,__ATOMIC_RELAXED);
                                __atomic_add_fetch(&m->bytes_main,(uint64_t)tb,__ATOMIC_RELAXED); }
    }
    int rep=expert_route(layer,eid);             /* DUAL-SSD: this expert's replica */
    if(rep && st_fd_rep(&m->S,tw[0]->fd,rep)<0) rep=0; /* shard not in that mirror (partial) */
    if(g_mmap){
        void *bw[3],*bq[3]; int okm=1;
        for(int k=0;k<3;k++){
            bw[k]=map_of_fd(rep_bfd(&m->S,tw[k]->fd,rep)); bq[k]=map_of_fd(rep_bfd(&m->S,tq[k]->fd,rep));
            if(!bw[k]||!bq[k]||((tw[k]->off)&3)||((tq[k]->off)&3)) okm=0;
        }
        if(okm){
            QT *qt[3]={&s->g,&s->u,&s->d}; int OO[3]={I,I,D}, II[3]={D,D,I};
            for(int k=0;k<3;k++){
                int64_t nb=tw[k]->nbytes;
                int gs=0;
                int fmt=qt_resolve_fmt(tw[k]->name,OO[k],II[k],nb,tq[k]->nbytes,&gs,NULL);   /* routed expert: never stamped */
                qt[k]->fmt=fmt; qt[k]->O=OO[k]; qt[k]->I=II[k]; qt[k]->gs=gs; qt[k]->qf=NULL; qt[k]->planar=0;   /* K1: byte nella MAPPA (read-only/shared): MAI planarizzare */
                qt[k]->q8=(int8_t*)((char*)bw[k]+tw[k]->off); qt[k]->q4=(uint8_t*)((char*)bw[k]+tw[k]->off);
                qt[k]->s=(float*)((char*)bq[k]+tq[k]->off);
            }
            /* CPU pre-touch: fault the pages in HERE (cheap, parallel, overlapped with the
             * resident-experts GPU submit) so the GPU never demand-faults file-backed pages
             * (measured catastrophic). madvise starts async readahead, the touch guarantees
             * residency. This is pread's I/O without the copy and without the slab. */
            for(int k=0;k<3;k++){
                char *p=(char*)bw[k]+tw[k]->off; size_t n=(size_t)tw[k]->nbytes;
#if defined(__APPLE__) || defined(__linux__) || defined(__FreeBSD__)
                madvise((void*)((uintptr_t)p & ~16383UL), n+16384, MADV_WILLNEED);
#endif
                volatile char acc=0;
                for(size_t i=0;i<n;i+=4096) acc+=p[i];
                acc+=p[n-1]; (void)acc;
                char *q=(char*)bq[k]+tq[k]->off; size_t nq=(size_t)tq[k]->nbytes;
                for(size_t i=0;i<nq;i+=4096) acc+=q[i];
                /* mlock deliberately NOT done here: this fires for every expert_load call,
                 * including the transient VRAM-staging pass in pin_load (host copy loaded,
                 * uploaded to GPU, then "released" via expert_host_release -- which only
                 * knows how to munlock s->slab, always NULL under mmap, so wiring here would
                 * leak locked pages for every GPU-tier expert). See pin_wire() below: it wires
                 * the final resident set only, after GPU release has already nulled out the
                 * pointers for anything that isn't genuinely RAM-tier. */
                atomic_fetch_add_explicit(&g_prof_io,(int64_t)(n+nq),memory_order_relaxed);
                atomic_fetch_add_explicit(&g_mir_bytes[rep],tw[k]->nbytes+tq[k]->nbytes,memory_order_relaxed);
            }
            atomic_fetch_add_explicit(&g_mir_nread[rep],1,memory_order_relaxed);
            s->eid=eid; return 0;
        }
    }
    int64_t wtot=tw[0]->nbytes+tw[1]->nbytes+tw[2]->nbytes;
    int64_t ftot=(tq[0]->nbytes+tq[1]->nbytes+tq[2]->nbytes)/4;
    /* rialloca se lo slot (riusato tra layer) e' troppo piccolo per QUESTO expert:
     * pread oltre la mappatura = short-read o CORRUZIONE silenziosa dei vicini
     *
     * ...and reallocate when it is too LARGE, which is #856. Slabs do not stay in
     * the row that grew them: the LRU promotion swaps ws[q] with a cache slot
     * (search "promozione LRU"), so a ws[] slot widened to hold an int8 MTP expert
     * comes back on the next token, loads a narrow int4 routed expert without
     * resizing, and is then swapped into a MAIN layer's cache -- still carrying the
     * MTP width. Up to 64 slabs per token bleed that way, so given enough tokens
     * every cache slot in the model costs the widest width in the container.
     *
     * That is why cap_for_ram() charged every row the widest width: the pessimism
     * was the true asymptote, not paranoia. It also halved the cache on GLM-5.2
     * (154 -> 77 slots per row, measured in #856). Shrinking here removes the bleed,
     * which is what lets the cap be computed from each row's REAL width.
     *
     * Hysteresis, not exact fit: alignment rounding alone makes slab_cap exceed the
     * request under COLI_METAL, and a shrink for a few KB would churn the allocator
     * on every miss. 25% is relative so it scales with the model; the floor only has
     * to clear that 16 KB rounding. Experts within a row share a shape, so the only
     * thing that crosses this threshold is a slab that came from a DIFFERENT row --
     * which is exactly the migration being stopped.
     *
     * Never arena slices (aslab): those are interior pointers into one per-layer
     * allocation and must not be freed. They are already per-layer-width, so they
     * have nothing to shrink. */
    int64_t want=wtot+8192;
    int64_t hyst=want/4; if(hyst<(1<<16)) hyst=1<<16;   /* 64 KB clears the 16 KB METAL rounding */
    if(!s->slab || want > s->slab_cap || (g_slab_shrink && !s->aslab && s->slab_cap > want+hyst)){
#ifdef COLI_METAL
        /* page-align + zero-copy wrap: the GPU reads this slab in place (unified memory) */
        if(s->slab && g_metal_enabled) coli_metal_unregister(s->slab);
        compat_aligned_free(s->slab);
        size_t need=((size_t)wtot+8192+16383)&~(size_t)16383;
        if(posix_memalign((void**)&s->slab,16384,need)){fprintf(stderr,"OOM slab\n"); if(fatal) exit(1); s->slab=NULL; s->slab_cap=0; return -1;}
        s->slab_cap=need;
        if(g_metal_enabled) coli_metal_register(s->slab,need);
#else
        compat_aligned_free(s->slab);
        if(posix_memalign((void**)&s->slab,4096,wtot+8192)){fprintf(stderr,"OOM slab\n"); if(fatal) exit(1); s->slab=NULL; s->slab_cap=0; return -1;}
        s->slab_cap=wtot+8192;
        numa_slab_bind(s->slab,(size_t)s->slab_cap);
#endif
    }
    /* The scales migrate with their weights, so they shrink on the same rule (#856).
     * fslab_cap counts FLOATS, so the hysteresis is in floats too. */
    int64_t fhyst=ftot/4; if(fhyst<(1<<14)) fhyst=1<<14;   /* floats, so 64 KB again */
    if(!s->fslab || ftot > s->fslab_cap || (g_slab_shrink && !s->afslab && s->fslab_cap > ftot+fhyst)){
#ifdef COLI_METAL
        /* page-align + register: the GPU reads the scales in place (unified memory).
         * Honours `fatal` exactly like the CPU arm below — a speculative pilot load
         * that hits OOM must unwind into a clean hidden slot, never exit(). */
        if(s->fslab && g_metal_enabled) coli_metal_unregister(s->fslab);
        free(s->fslab);
        size_t fb=(((size_t)ftot*sizeof(float))+16383)&~(size_t)16383;
        if(ftot<0 || (uint64_t)ftot > SIZE_MAX/sizeof(float) ||
           posix_memalign((void**)&s->fslab,16384,fb)){
            fprintf(stderr,"OOM fslab\n"); if(fatal) exit(1);
            /* unregister BEFORE freeing -- a stale g_slabs entry would let resolve() hand
             * the GPU a pointer into freed memory (and under COLI_METAL_RESSET=1, leave the
             * buffer a permanent residency-set member over it). Ported from e4/metal-heap
             * validator fix 6753225; pre-existing gap on main/dev. */
            if(s->slab && g_metal_enabled) coli_metal_unregister(s->slab);
            compat_aligned_free(s->slab); s->slab=NULL; s->slab_cap=0;  /* clean, hidden slot (eid stays -1) */
            s->fslab=NULL; s->fslab_cap=0; return -1;
        }
        s->fslab_cap=ftot;
        if(g_metal_enabled) coli_metal_register(s->fslab,fb);
#else
        free(s->fslab);
        if(fatal){ s->fslab=falloc(ftot); }          /* main path: byte-identical exit-on-OOM */
        else {                                        /* speculative pilot: checked alloc, never exit() */
            /* replicate falloc's anti-wrap guard + malloc (no zeroing/alignment) */
            if(ftot<0 || (uint64_t)ftot > SIZE_MAX/sizeof(float) ||
               !(s->fslab=malloc((size_t)ftot*sizeof(float)))){
                fprintf(stderr,"OOM fslab\n");
                compat_aligned_free(s->slab); s->slab=NULL; s->slab_cap=0; /* leave a clean, hidden slot (eid stays -1) */
                s->fslab=NULL; s->fslab_cap=0; return -1;
            }
        }
        s->fslab_cap=ftot;
        numa_slab_bind(s->fslab,(size_t)ftot*sizeof(float));
#endif
    }
    /* DISK-CLASS: classify before the reads; computed unconditionally at dc_on sites so
     * the timer (dc_t0) brackets exactly the read work, matching what the GB/s in the
     * DISK-CLASS line describes. dc_on gates ALL of it off demand=0 call sites
     * (pilot/repin/pin -- never classified, see the call sites) and off PROF=0 runs
     * (dc_needed()) -- zero cost, zero behavior change there. The fd choice below is
     * NOT influenced by the verdict: this is measurement only. */
    int dc_on = demand && dc_needed();
    int dc_cls = dc_on ? expert_classify(m,layer,eid) : DC_WARM;
    double dc_t0 = dc_on ? now_s() : 0;
    if(dc_on) dc_wall_enter(dc_cls,dc_t0);        /* busy-wall open; EVERY exit path below must pair it */
    int ord[3]={0,1,2};                          /* ordina per offset nel file */
    for(int a=0;a<3;a++) for(int bb=a+1;bb<3;bb++) if(tw[ord[bb]]->off<tw[ord[a]]->off){ int t=ord[a]; ord[a]=ord[bb]; ord[bb]=t; }
    int contig = tw[ord[0]]->fd==tw[ord[1]]->fd && tw[ord[1]]->fd==tw[ord[2]]->fd
              && tw[ord[0]]->off+tw[ord[0]]->nbytes==tw[ord[1]]->off
              && tw[ord[1]]->off+tw[ord[1]]->nbytes==tw[ord[2]]->off;
    int64_t pos[3]; int done=0, dc_direct=0;
    if(contig){
        int64_t off0=tw[ord[0]]->off;
        int dfd = g_direct ? st_direct_fd_rep(&m->S, tw[ord[0]]->fd, rep) : -1;
        if(dfd>=0){                              /* O_DIRECT: offset/len allineati a 4K */
            int64_t base=off0 & ~4095LL, need=(off0-base)+wtot;
            int64_t len=(need+4095)&~4095LL;
            /* multi-SSD: stripe the read across the replicas (accounts internally);
             * any failure falls back to the whole read on the routed replica */
            ssize_t r=(ssize_t)mir_pread_striped(&m->S,tw[ord[0]]->fd,rep,(char*)s->slab,len,base);
            if(r<need){
                r=pread(dfd, s->slab, len, base);
                if(r>=need){
                    atomic_fetch_add_explicit(&g_mir_bytes[rep],(int64_t)r,memory_order_relaxed);
                    atomic_fetch_add_explicit(&g_mir_nread[rep],1,memory_order_relaxed);
                }
            }
            if(r>=need){
                pos[ord[0]]=off0-base; pos[ord[1]]=pos[ord[0]]+tw[ord[0]]->nbytes;
                pos[ord[2]]=pos[ord[1]]+tw[ord[1]]->nbytes; done=1; dc_direct=1;
            }
        }
        if(!done){                               /* fallback bufferizzato */
            if(mir_pread(&m->S, tw[ord[0]]->fd, rep, s->slab, wtot, off0, "pread expert")){ if(fatal) exit(1);
                if(dc_on) dc_wall_exit(dc_cls,now_s());   /* pair the enter on the non-fatal unwind */
                return -1; }
            pos[ord[0]]=0; pos[ord[1]]=tw[ord[0]]->nbytes; pos[ord[2]]=tw[ord[0]]->nbytes+tw[ord[1]]->nbytes; done=1;
        }
    }
    if(!done){                                   /* non contigui: 3 pread bufferizzate */
        int64_t o=0;
        for(int a=0;a<3;a++){ int k=ord[a];
            if(mir_pread(&m->S, tw[k]->fd, rep, s->slab+o, tw[k]->nbytes, tw[k]->off, "pread expert")){ if(fatal) exit(1);
                if(dc_on) dc_wall_exit(dc_cls,now_s());   /* pair the enter on the non-fatal unwind */
                return -1; }
            pos[k]=o; o+=tw[k]->nbytes; }
    }
    float *fp[3]; int64_t fo=0;                  /* scale (piccole) */
    for(int k=0;k<3;k++){
        if(mir_pread(&m->S, tq[k]->fd, rep, (char*)(s->fslab+fo), tq[k]->nbytes, tq[k]->off, "pread qs")){ if(fatal) exit(1);
            if(dc_on) dc_wall_exit(dc_cls,now_s());       /* pair the enter on the non-fatal unwind */
            return -1; }
        fp[k]=s->fslab+fo; fo+=tq[k]->nbytes/4; }
    atomic_fetch_add_explicit(&g_prof_io,wtot+fo*4,memory_order_relaxed);
    if(dc_on){                                    /* DISK-CLASS accounting, see dc_needed() */
        double dc_t1=now_s();                     /* one clock read for thread-ns AND the wall exit */
        int64_t bytes=wtot+fo*4;
        atomic_fetch_add_explicit(&g_dc_n[dc_cls],1,memory_order_relaxed);
        atomic_fetch_add_explicit(&g_dc_bytes[dc_cls],bytes,memory_order_relaxed);
        atomic_fetch_add_explicit(&g_dc_ns[dc_cls],(int64_t)((dc_t1-dc_t0)*1e9),memory_order_relaxed);
        dc_wall_exit(dc_cls,dc_t1);
        if(dc_direct)                             /* which fd ACTUALLY served this class */
            atomic_fetch_add_explicit(&g_dc_direct_n[dc_cls],1,memory_order_relaxed);
    }
    if(g_drop){                                  /* scarta subito le pagine: evita che la page
                                                  * cache in pressione strangoli il throughput.
                                                  * The drop targets the fd of the replica READ. */
        posix_fadvise(rep_bfd(&m->S,tw[ord[0]]->fd,rep), tw[ord[0]]->off, wtot, POSIX_FADV_DONTNEED);
        for(int k=0;k<3;k++) posix_fadvise(rep_bfd(&m->S,tq[k]->fd,rep), tq[k]->off, tq[k]->nbytes, POSIX_FADV_DONTNEED);
    }
    QT *qt[3]={&s->g,&s->u,&s->d}; int OO[3]={I,I,D}, II[3]={D,D,I};
    for(int k=0;k<3;k++){
        int64_t nb=tw[k]->nbytes;
        int gs=0;
        int fmt=qt_resolve_fmt(tw[k]->name,OO[k],II[k],nb,tq[k]->nbytes,&gs,NULL);   /* routed expert: never stamped */
        qt[k]->fmt=fmt; qt[k]->O=OO[k]; qt[k]->I=II[k]; qt[k]->gs=gs; qt[k]->qf=NULL;
        qt[k]->q8=(int8_t*)(s->slab+pos[k]); qt[k]->q4=s->slab+pos[k]; qt[k]->s=fp[k];
        /* K1: slab di proprieta' (pread, riscritta a ogni fill) -> planarizza.
         * Il reset dentro qt_planarize non basta: il flag del fill PRECEDENTE
         * va azzerato PRIMA, perche' i byte appena letti sono a coppie.
         * EN: slab-owned bytes -> planarize; clear the stale flag first. */
        qt[k]->planar=0; qt_planarize(qt[k]);
    }
    s->eid=eid; return 0;
}
/* Every expert read goes through here: time the whole load (pread/fault +
 * bookkeeping) on the thread that runs it, into the disk-service counter. */
static int expert_load(Model *m, int layer, int eid, ESlot *s, int fatal, int demand){
    /* `demand` marks a routing-driven demand-load (moe()'s PIPE/OMP miss path, where the
     * pre-bump elast_pre snapshot moe() just wrote is valid) -- pass 0 from anywhere else
     * (pilot speculative loads, repin, startup PIN loading): those never run through THIS
     * call's own FASE A, so the snapshot either doesn't apply or was never written for
     * them, and DISK-CLASS deliberately leaves them unclassified -- see expert_classify()'s
     * call site. */
    double t0=now_s();
    int rc=expert_load_impl(m,layer,eid,s,fatal,demand);
    atomic_fetch_add_explicit(&g_edisk_ns,(int64_t)((now_s()-t0)*1e9),memory_order_relaxed);
    return rc;
}

#if !defined(_WIN32)
/* Expert-worker protocol. Headers use network-order u32 values; activation
 * bytes remain raw little-endian f32, matching the native engine ABI. One
 * request contains the routed batch-union for a layer. */
#define COLI_CLUSTER_MAGIC "COLIEX01"
#define COLI_CLUSTER_VERSION 1u
static int cluster_io(int fd, void *buf, size_t n, int write_mode){
    char *p=(char*)buf;
    while(n){
        ssize_t r=write_mode?send(fd,p,n,0):recv(fd,p,n,MSG_WAITALL);
        if(r<=0){ if(r<0&&errno==EINTR) continue; return -1; }
        p+=r; n-=(size_t)r;
    }
    return 0;
}
static int cluster_u32(int fd, uint32_t *v, int write_mode){
    uint32_t x=write_mode?htonl(*v):0;
    if(cluster_io(fd,write_mode?(void*)&x:(void*)v,sizeof(x),write_mode)) return -1;
    if(!write_mode) *v=ntohl(*v);
    return 0;
}
static int cluster_connect_one(const char *spec, ClusterWorker *out){
    char copy[256]; strncpy(copy,spec,sizeof(copy)-1); copy[sizeof(copy)-1]=0;
    char *colon=strrchr(copy,':'); if(!colon||colon==copy||!colon[1]) return -1;
    *colon=0; int port=atoi(colon+1); if(port<1||port>65535) return -1;
    char portbuf[16]; snprintf(portbuf,sizeof(portbuf),"%d",port);
    struct addrinfo hint={0},*ai=NULL; hint.ai_socktype=SOCK_STREAM;
    if(getaddrinfo(copy,portbuf,&hint,&ai)!=0) return -1;
    int fd=-1;
    for(struct addrinfo *p=ai;p;p=p->ai_next){
        fd=socket(p->ai_family,p->ai_socktype,p->ai_protocol);
        if(fd<0) continue;
        if(connect(fd,p->ai_addr,p->ai_addrlen)==0) break;
        close(fd); fd=-1;
    }
    freeaddrinfo(ai); if(fd<0) return -1;
    out->fd=fd;
    size_t hostlen=strlen(copy); if(hostlen>=sizeof(out->host)) hostlen=sizeof(out->host)-1;
    memcpy(out->host,copy,hostlen); out->host[hostlen]=0;
    out->port=port; return 0;
}
static void cluster_close_all(void){
    for(int i=0;i<g_cluster_n;i++) if(g_cluster_workers[i].fd>=0) close(g_cluster_workers[i].fd);
    g_cluster_n=0;
}
static void cluster_init(void){
    const char *list=getenv("CLUSTER_WORKERS"); if(!list||!*list) return;
    char *copy=strdup(list),*save=NULL;
    for(char *tok=strtok_r(copy,",",&save);tok&&g_cluster_n<16;tok=strtok_r(NULL,",",&save)){
        while(*tok==' '||*tok=='\t') tok++;
        if(!cluster_connect_one(tok,&g_cluster_workers[g_cluster_n])) g_cluster_n++;
        else fprintf(stderr,"[CLUSTER] cannot connect to expert worker %s\n",tok);
    }
    free(copy);
    if(g_cluster_n<1){ fprintf(stderr,"[CLUSTER] no expert workers reachable\n"); exit(1); }
    fprintf(stderr,"[CLUSTER] coordinator connected to %d expert worker(s)\n",g_cluster_n);
}
typedef struct { int eid,nr; int *rows; float *weights,*inputs; } ClusterItem;
static int cluster_item(const int *idxs,const float *ws,const int *keff,int K,int S,
                        int eid,ClusterItem *it,int D,const float *x){
    it->eid=eid; it->nr=0;
    for(int s=0;s<S;s++) for(int k=0;k<keff[s];k++)
        if(idxs[(int64_t)s*K+k]==eid){ it->nr++; break; }
    if(!it->nr) return 0;
    it->rows=malloc((size_t)it->nr*sizeof(int));
    it->weights=malloc((size_t)it->nr*sizeof(float));
    it->inputs=falloc((int64_t)it->nr*D); int r=0;
    for(int s=0;s<S;s++) for(int k=0;k<keff[s];k++) if(idxs[(int64_t)s*K+k]==eid){
        it->rows[r]=s; it->weights[r]=ws[(int64_t)s*K+k];
        memcpy(it->inputs+(int64_t)r*D,x+(int64_t)s*D,(size_t)D*sizeof(float)); r++; break;
    }
    return 1;
}
static void cluster_item_free(ClusterItem *it){ free(it->rows); free(it->weights); free(it->inputs); memset(it,0,sizeof(*it)); }
static void cluster_moe_batch(Model *m,int layer,float *x,int S,float *out,
                              const int *idxs,const float *ws,const int *keff,int K,
                              const int *uniq,int base,int nb){
    int D=m->c.hidden;
    for(int wi=0;wi<g_cluster_n;wi++){
        ClusterItem items[64]; memset(items,0,sizeof(items)); int n=0;
        for(int j=0;j<nb;j++){
            int eid=uniq[base+j];
            if((eid+layer)%g_cluster_n!=wi) continue;
            if(n<64 && cluster_item(idxs,ws,keff,K,S,eid,&items[n],D,x)) n++;
        }
        if(!n) continue;
        ClusterWorker *w=&g_cluster_workers[wi]; char magic[8]; uint32_t v;
        if(cluster_io(w->fd,(void*)COLI_CLUSTER_MAGIC,8,1)) goto fail;
        v=COLI_CLUSTER_VERSION; if(cluster_u32(w->fd,&v,1)) goto fail;
        v=(uint32_t)layer; if(cluster_u32(w->fd,&v,1)) goto fail;
        v=(uint32_t)D; if(cluster_u32(w->fd,&v,1)) goto fail;
        v=(uint32_t)m->c.moe_inter; if(cluster_u32(w->fd,&v,1)) goto fail;
        v=(uint32_t)n; if(cluster_u32(w->fd,&v,1)) goto fail;
        for(int j=0;j<n;j++){
            v=(uint32_t)items[j].eid; if(cluster_u32(w->fd,&v,1)) goto fail;
            v=(uint32_t)items[j].nr; if(cluster_u32(w->fd,&v,1)) goto fail;
            if(cluster_io(w->fd,items[j].inputs,(size_t)items[j].nr*D*sizeof(float),1)) goto fail;
        }
        if(cluster_io(w->fd,magic,8,0)||memcmp(magic,COLI_CLUSTER_MAGIC,8)) goto fail;
        if(cluster_u32(w->fd,&v,0)||v!=COLI_CLUSTER_VERSION) goto fail;
        if(cluster_u32(w->fd,&v,0)||v!=0) goto fail;
        if(cluster_u32(w->fd,&v,0)||v!=(uint32_t)n) goto fail;
        for(int j=0;j<n;j++){
            uint32_t eid,nr;
            if(cluster_u32(w->fd,&eid,0)||cluster_u32(w->fd,&nr,0) ||
               eid!=(uint32_t)items[j].eid || nr!=(uint32_t)items[j].nr) goto fail;
            float *y=falloc((int64_t)nr*D);
            if(cluster_io(w->fd,y,(size_t)nr*D*sizeof(float),0)){ free(y); goto fail; }
            for(uint32_t r=0;r<nr;r++){ float *dst=out+(int64_t)items[j].rows[r]*D;
                float wt=items[j].weights[r]; for(int d=0;d<D;d++) dst[d]+=wt*y[(int64_t)r*D+d]; }
            free(y);
        }
        for(int j=0;j<n;j++) cluster_item_free(&items[j]);
        continue;
fail:
        for(int j=0;j<n;j++) cluster_item_free(&items[j]);
        fprintf(stderr,"[CLUSTER] expert worker %s:%d failed during layer %d batch\n",w->host,w->port,layer);
        exit(1);
    }
}
typedef struct { int eid,nr; float *inputs; } ClusterRequestItem;
/* Mirror expert_load_impl's resolution so the worker can NAME the tensor it can't
 * resolve instead of dying inside a fatal path (st_die_missing / st_read_* exit(1)).
 * Returns 1 and fills `out` with the first missing expert tensor — quantized: a
 * gate/up/down weight or its .qs sidecar; unquantized: the full weight — else 0. */
static int worker_expert_missing_tensor(Model *m,int layer,int eid,char *out,size_t outsz){
    static const char *suf[3]={"gate_proj","up_proj","down_proj"};
    char nm[288], qn[320];
    snprintf(nm,sizeof(nm),"model.layers.%d.mlp.experts.%d.gate_proj.weight",layer,eid);
    snprintf(qn,sizeof(qn),"%s.qs",nm);
    if(!st_has(&m->S,qn)){                       /* unquantized: full weights only */
        for(int k=0;k<3;k++){
            snprintf(nm,sizeof(nm),"model.layers.%d.mlp.experts.%d.%s.weight",layer,eid,suf[k]);
            if(!st_has(&m->S,nm)){ snprintf(out,outsz,"%s",nm); return 1; }
        }
        return 0;
    }
    for(int k=0;k<3;k++){                        /* quantized: weight + .qs sidecar */
        snprintf(nm,sizeof(nm),"model.layers.%d.mlp.experts.%d.%s.weight",layer,eid,suf[k]);
        snprintf(qn,sizeof(qn),"%s.qs",nm);
        if(!st_has(&m->S,nm)||!st_has(&m->S,qn)){ snprintf(out,outsz,"%s",nm); return 1; }
    }
    return 0;
}
static int cluster_worker_run(const char *snap,int port,int ebits,int dbits){
    Model m; memset(&m,0,sizeof(m)); m.ebits=ebits; m.dbits=dbits; load_cfg(&m.c,snap);
    { const char *xd=getenv("COLI_MODEL_DIRS");        /* SPLIT: expert shards spread across N drives */
      st_init_multi(&m.S,snap,(xd&&*xd)?xd:NULL); }
    int nr_layers=m.c.n_layers+1; ESlot *cache=calloc((size_t)nr_layers,sizeof(ESlot));
    for(int i=0;i<nr_layers;i++) cache[i].eid=-1;
    int fd=socket(AF_INET,SOCK_STREAM,0); if(fd<0){perror("cluster worker socket");return 1;}
    int yes=1; setsockopt(fd,SOL_SOCKET,SO_REUSEADDR,&yes,sizeof(yes));
    struct sockaddr_in addr={0}; addr.sin_family=AF_INET; addr.sin_addr.s_addr=htonl(INADDR_ANY); addr.sin_port=htons((uint16_t)port);
    if(bind(fd,(struct sockaddr*)&addr,sizeof(addr))||listen(fd,4)){perror("cluster worker bind/listen");return 1;}
    fprintf(stderr,"[CLUSTER] expert worker listening on 0.0.0.0:%d (disk-backed, cache=%d/layer)\n",port,nr_layers);
    for(;;){
        int cfd=accept(fd,NULL,NULL); if(cfd<0){if(errno==EINTR)continue;break;}
        for(;;){
            char magic[8]; uint32_t v,layer,D,I,n;
            if(cluster_io(cfd,magic,8,0)) break;
            if(memcmp(magic,COLI_CLUSTER_MAGIC,8)||cluster_u32(cfd,&v,0)||v!=COLI_CLUSTER_VERSION||
               cluster_u32(cfd,&layer,0)||cluster_u32(cfd,&D,0)||cluster_u32(cfd,&I,0)||cluster_u32(cfd,&n,0)||
               D!=(uint32_t)m.c.hidden||I!=(uint32_t)m.c.moe_inter||layer>=(uint32_t)nr_layers||n<1||n>64){
                close(cfd); cfd=-1; break;
            }
            ClusterRequestItem *items=calloc(n,sizeof(*items)); int bad=0;
            for(uint32_t j=0;j<n;j++){
                uint32_t eid,nr;
                if(cluster_u32(cfd,&eid,0)||cluster_u32(cfd,&nr,0)||eid>=(uint32_t)m.c.n_experts||nr<1||nr>65536){bad=1;break;}
                items[j].eid=(int)eid; items[j].nr=(int)nr; items[j].inputs=falloc((int64_t)nr*D);
                if(cluster_io(cfd,items[j].inputs,(size_t)nr*D*sizeof(float),0)){bad=1;break;}
            }
            if(bad){ for(uint32_t j=0;j<n;j++)free(items[j].inputs); free(items); close(cfd); cfd=-1; break; }
            v=COLI_CLUSTER_VERSION; if(cluster_io(cfd,(void*)COLI_CLUSTER_MAGIC,8,1)||cluster_u32(cfd,&v,1)) break;
            v=0; if(cluster_u32(cfd,&v,1)) break; v=n; if(cluster_u32(cfd,&v,1)) break;
            ESlot *slot=&cache[layer];
            for(uint32_t j=0;j<n;j++){
                if(slot->eid!=items[j].eid || !slot->slab){
                    char miss[288];
                    if(worker_expert_missing_tensor(&m,(int)layer,items[j].eid,miss,sizeof(miss))){
                        fprintf(stderr,"[CLUSTER] worker cannot resolve expert tensor %s (layer %d, expert %d)\n",
                                miss,(int)layer,items[j].eid);
                        bad=1; break;
                    }
                    if(expert_load(&m,(int)layer,items[j].eid,slot,0,0)){bad=1;break;}
                }
                int rows=items[j].nr; float *g=falloc((int64_t)rows*I),*u=falloc((int64_t)rows*I),*y=falloc((int64_t)rows*D);
                /* fmt=6: the gate/up input is per-item here (never reused after this
                 * expert), so rotate it in place — the same Q^T x that moe() applies
                 * once per layer via E8_XE. The down-input rotation lives in expert_ffn. */
                if(slot->g.fmt==6) e8_rot_rows(items[j].inputs,rows,D);
                expert_ffn(y,g,u,items[j].inputs,&slot->g,&slot->u,&slot->d,rows,I);
                v=(uint32_t)items[j].eid; if(cluster_u32(cfd,&v,1)){bad=1;free(g);free(u);free(y);break;}
                v=(uint32_t)rows; if(cluster_u32(cfd,&v,1)||cluster_io(cfd,y,(size_t)rows*D*sizeof(float),1)){bad=1;free(g);free(u);free(y);break;}
                free(g);free(u);free(y);
            }
            for(uint32_t j=0;j<n;j++)free(items[j].inputs); free(items);
            if(bad) break;
        }
        if(cfd>=0)close(cfd);
    }
    close(fd); return 0;
}
#endif

#ifdef __linux__
/* io_uring expert batches.  One owner prepares all reads for a block, submits
 * them in one syscall, and reaps CQEs on demand.  The kernel, rather than a set
 * of blocking pthreads, owns the I/O concurrency. */
#define URING_LOAD_MAX 64
#define URING_REQ_MAX  512
typedef struct {
    int load, expect;
    int fd, primary_fd, rep;
    void *buf;
    int64_t off;
} UringRead;
typedef struct {
    Model *m; ESlot *s; int layer,eid,fatal;
    st_tensor *tw[3],*tq[3]; int64_t pos[3];
    int pending,done,finalized,error;
} UringLoad;
typedef struct {
    ColiUring ring;
    UringLoad load[URING_LOAD_MAX];
    UringRead req[URING_REQ_MAX];
    int nload,nreq,started;
} UringBatch;
static UringBatch g_ub_pipe, g_ub_pilot;

static int uring_batch_init(UringBatch *b){
    if(b->started) return 0;
    if(coli_uring_init(&b->ring,URING_REQ_MAX)) return -1;
    b->started=1; return 0;
}
static void uring_batch_reset(UringBatch *b){
    b->nload=0; b->nreq=0;
}
static int uring_load_error(UringLoad *l,int err,const char *what){
    l->error=err?err:EIO; l->done=1;
    if(l->fatal){ errno=l->error; perror(what); exit(1); }
    return -1;
}
static int uring_add_read(UringBatch *b,int li,int fd,int primary_fd,int rep,
                          void *buf,size_t len,int64_t off,size_t expect){
    if(b->nreq>=URING_REQ_MAX || expect>INT_MAX){ errno=E2BIG; return -1; }
    if(rep<0 || rep>=MIR_REPS){ errno=EINVAL; return -1; }
    int ri=b->nreq++;
    b->req[ri]=(UringRead){li,(int)expect,fd,primary_fd,rep,buf,off};
    if(coli_uring_prep_read(&b->ring,fd,buf,len,off,(uint64_t)ri+1)) return -1;
    b->load[li].pending++;
    return 0;
}
/* Buffered replica read with per-shard fallback.  Keep the actual source in
 * UringRead: completion accounting, DROP and runtime error fallback must all
 * describe the fd that was really submitted, not recompute a route later. */
static int uring_add_rep_read(UringBatch *b,int li,shards *S,int primary_fd,
                              int rep,void *buf,size_t len,int64_t off,
                              size_t expect){
    int fd=st_fd_rep(S,primary_fd,rep);
    int used=(rep && fd>=0)?rep:0;
    if(fd<0) fd=primary_fd;
    return uring_add_read(b,li,fd,primary_fd,used,buf,len,off,expect);
}
/* Returns the load index. URING is intentionally a quantized streaming path;
 * unsupported layouts fail instead of silently dropping back to pread. */
static int uring_load_add(UringBatch *b,Model *m,int layer,int eid,ESlot *s,int fatal){
    if(b->nload>=URING_LOAD_MAX){ errno=E2BIG; return -1; }
    int li=b->nload++;
    UringLoad *l=&b->load[li]; memset(l,0,sizeof(*l));
    l->m=m; l->s=s; l->layer=layer; l->eid=eid; l->fatal=fatal;
    char nm[3][288],qn[320]; const char suf[3][16]={"gate_proj","up_proj","down_proj"};  /* bounded suf: see #484 */
    for(int k=0;k<3;k++) snprintf(nm[k],sizeof(nm[k]),"model.layers.%d.mlp.experts.%d.%s.weight",layer,eid,suf[k]);
    snprintf(qn,sizeof(qn),"%s.qs",nm[0]);
    if(g_mmap || !st_has(&m->S,qn))
        return uring_load_error(l,ENOTSUP,"URING requires quantized expert tensors"),li;
#ifdef COLI_CUDA
    if(s->eid!=eid){ qt_cuda_reset(&s->g); qt_cuda_reset(&s->u); qt_cuda_reset(&s->d); }
#endif
    for(int k=0;k<3;k++){
        l->tw[k]=st_find(&m->S,nm[k]);
        size_t n=strnlen(nm[k],sizeof(nm[k]));
        if(n+3>=sizeof(qn)) return uring_load_error(l,ENAMETOOLONG,"io_uring expert metadata"),li;
        memcpy(qn,nm[k],n); memcpy(qn+n,".qs",4); l->tq[k]=st_find(&m->S,qn);
        if(!l->tw[k]||!l->tq[k]) return uring_load_error(l,ENOENT,"io_uring expert metadata"),li;
    }
    int64_t wtot=l->tw[0]->nbytes+l->tw[1]->nbytes+l->tw[2]->nbytes;
    int64_t ftot=(l->tq[0]->nbytes+l->tq[1]->nbytes+l->tq[2]->nbytes)/4;
    if(wtot<=0 || ftot<=0) return uring_load_error(l,EINVAL,"io_uring expert size"),li;
    /* Same shrink rule as the pread path (#856): a slot that once held a wider
     * expert must not carry that width into a narrower row, or the per-row cap
     * cap_for_ram() computes stops being true. Same hysteresis, same arena guard. */
    int64_t want=wtot+8192, hyst=want/4; if(hyst<(1<<16)) hyst=1<<16;
    int64_t fhyst=ftot/4; if(fhyst<(1<<14)) fhyst=1<<14;   /* floats, so 64 KB again */
    if(!s->slab || want>s->slab_cap || (g_slab_shrink && !s->aslab && s->slab_cap > want+hyst)){
#ifdef COLI_METAL
        if(s->slab&&g_metal_enabled) coli_metal_unregister(s->slab);
        compat_aligned_free(s->slab);
        size_t need=((size_t)wtot+8192+16383)&~(size_t)16383;
        if(posix_memalign((void**)&s->slab,16384,need)){
            s->slab=NULL; s->slab_cap=0; return uring_load_error(l,ENOMEM,"io_uring expert slab"),li; }
        s->slab_cap=need; if(g_metal_enabled) coli_metal_register(s->slab,need);
#else
        compat_aligned_free(s->slab);
        if(posix_memalign((void**)&s->slab,4096,(size_t)wtot+8192)){
            s->slab=NULL; s->slab_cap=0; return uring_load_error(l,ENOMEM,"io_uring expert slab"),li; }
        s->slab_cap=wtot+8192;
#endif
    }
    if(!s->fslab || ftot>s->fslab_cap || (g_slab_shrink && !s->afslab && s->fslab_cap > ftot+fhyst)){
#ifdef COLI_METAL
        if(s->fslab&&g_metal_enabled) coli_metal_unregister(s->fslab);
        free(s->fslab); size_t fb=(((size_t)ftot*sizeof(float))+16383)&~(size_t)16383;
        if(posix_memalign((void**)&s->fslab,16384,fb)){
            s->fslab=NULL; s->fslab_cap=0; return uring_load_error(l,ENOMEM,"io_uring expert scales"),li; }
        s->fslab_cap=ftot; if(g_metal_enabled) coli_metal_register(s->fslab,fb);
#else
        free(s->fslab); s->fslab=malloc((size_t)ftot*sizeof(float));
        if(!s->fslab){ s->fslab_cap=0; return uring_load_error(l,ENOMEM,"io_uring expert scales"),li; }
        s->fslab_cap=ftot;
#endif
    }
    /* DUAL-SSD (#1165): pick this expert's replica exactly as the blocking
     * path does. The uring path used the primary fd unconditionally, so
     * URING=1 read every expert from drive 0 and COLI_MODEL_MIRROR bought
     * nothing -- visible as an idle mirror in iostat while URING=0 lit it up.
     * Same deterministic hash of (layer,eid), so an expert always comes from
     * the same drive and PILOT's readahead lands where the demand read will.
     * Partial mirrors fall back to the primary per shard, as at the blocking
     * site. */
    int rep=expert_route(layer,eid);
    if(rep && st_fd_rep(&m->S,l->tw[0]->fd,rep)<0) rep=0;
    int ord[3]={0,1,2};
    for(int a=0;a<3;a++) for(int z=a+1;z<3;z++) if(l->tw[ord[z]]->off<l->tw[ord[a]]->off){int t=ord[a];ord[a]=ord[z];ord[z]=t;}
    int contig=l->tw[ord[0]]->fd==l->tw[ord[1]]->fd && l->tw[ord[1]]->fd==l->tw[ord[2]]->fd
        && l->tw[ord[0]]->off+l->tw[ord[0]]->nbytes==l->tw[ord[1]]->off
        && l->tw[ord[1]]->off+l->tw[ord[1]]->nbytes==l->tw[ord[2]]->off;
    if(contig){
        int64_t off0=l->tw[ord[0]]->off;
        int dfd=g_direct?st_direct_fd_rep(&m->S,l->tw[ord[0]]->fd,rep):-1;
        if(dfd>=0){
            int64_t base=off0&~4095LL,need=(off0-base)+wtot,len=(need+4095)&~4095LL;
            l->pos[ord[0]]=off0-base; l->pos[ord[1]]=l->pos[ord[0]]+l->tw[ord[0]]->nbytes;
            l->pos[ord[2]]=l->pos[ord[1]]+l->tw[ord[1]]->nbytes;
            if(uring_add_read(b,li,dfd,l->tw[ord[0]]->fd,rep,
                              s->slab,(size_t)len,base,(size_t)need))
                return uring_load_error(l,errno,"io_uring direct expert read"),li;
        }else{
            l->pos[ord[0]]=0; l->pos[ord[1]]=l->tw[ord[0]]->nbytes;
            l->pos[ord[2]]=l->pos[ord[1]]+l->tw[ord[1]]->nbytes;
            if(uring_add_rep_read(b,li,&m->S,l->tw[ord[0]]->fd,rep,
                                  s->slab,(size_t)wtot,off0,(size_t)wtot))
                return uring_load_error(l,errno,"io_uring expert read"),li;
        }
    }else{
        int64_t o=0;
        for(int a=0;a<3;a++){ int k=ord[a]; l->pos[k]=o;
            if(uring_add_rep_read(b,li,&m->S,l->tw[k]->fd,rep,
                                  s->slab+o,(size_t)l->tw[k]->nbytes,
                                  l->tw[k]->off,(size_t)l->tw[k]->nbytes))
                return uring_load_error(l,errno,"io_uring expert read"),li;
            o+=l->tw[k]->nbytes;
        }
    }
    int64_t fo=0;
    for(int k=0;k<3;k++){
        if(uring_add_rep_read(b,li,&m->S,l->tq[k]->fd,rep,
                              s->fslab+fo,(size_t)l->tq[k]->nbytes,
                              l->tq[k]->off,(size_t)l->tq[k]->nbytes))
            return uring_load_error(l,errno,"io_uring expert scale read"),li;
        fo+=l->tq[k]->nbytes/4;
    }
    return li;
}
static void uring_reap(UringBatch *b){
    struct io_uring_cqe cqe;
    while(coli_uring_peek(&b->ring,&cqe)){
        if(!cqe.user_data || cqe.user_data>(uint64_t)b->nreq) continue;
        UringRead *r=&b->req[cqe.user_data-1]; UringLoad *l=&b->load[r->load];
        int served=cqe.res>=r->expect;
        if(!served && r->rep){
            /* Match mir_pread's availability contract. A missing replica file
             * is handled before submission; an I/O error or short completion
             * discovered here gets one synchronous retry on the primary. This
             * path is exceptional, so preserving inference is more important
             * than retaining queue depth while a mirror is degraded. */
            static _Atomic int warned;
            if(!atomic_exchange(&warned,1))
                fprintf(stderr,"[MIRROR] io_uring read error on the mirror copy — falling back to the primary drive\n");
            if(!pread_full(r->primary_fd,r->buf,r->expect,r->off,
                           "io_uring mirror fallback")){
                r->fd=r->primary_fd; r->rep=0; cqe.res=r->expect; served=1;
            }
        }
        if(served){
            atomic_fetch_add_explicit(&g_mir_bytes[r->rep],cqe.res,memory_order_relaxed);
            atomic_fetch_add_explicit(&g_mir_nread[r->rep],1,memory_order_relaxed);
        }else if(!l->error) l->error=cqe.res<0?-cqe.res:EIO;
        if(l->pending>0) l->pending--;
        if(l->pending==0) l->done=1;
    }
}
static int uring_submit_batch(UringBatch *b){
    if(coli_uring_enter(&b->ring,0)<0) return -1;
    uring_reap(b); return 0;
}
static int uring_wait_load(UringBatch *b,int li){
    UringLoad *l=&b->load[li];
    while(!l->done){
        uring_reap(b); if(l->done) break;
        if(coli_uring_enter(&b->ring,1)<0) return uring_load_error(l,errno,"io_uring wait");
    }
    return l->error?-1:0;
}
static int uring_finalize_load(UringBatch *b,int li,int publish_eid){
    UringLoad *l=&b->load[li]; ESlot *s=l->s;
    if(l->finalized) return 0;
    if(uring_wait_load(b,li)<0){ errno=l->error; if(l->fatal){perror("io_uring expert completion");exit(1);} return -1; }
    if(g_drop){
        /* Each request remembers its final source (mirror, per-shard primary,
         * or primary after an error retry). Dropping tensor metadata's primary
         * fd here left the pages actually read from a mirror resident forever. */
        for(int ri=0;ri<b->nreq;ri++) if(b->req[ri].load==li)
            posix_fadvise(b->req[ri].fd,b->req[ri].off,
                          (off_t)b->req[ri].expect,POSIX_FADV_DONTNEED);
    }
    Cfg *c=&l->m->c; int I=c->moe_inter,D=c->hidden; float *fp[3]; int64_t fo=0;
    QT *qt[3]={&s->g,&s->u,&s->d}; int OO[3]={I,I,D},II[3]={D,D,I};
    for(int k=0;k<3;k++){
        fp[k]=s->fslab+fo; fo+=l->tq[k]->nbytes/4;
        int64_t nb=l->tw[k]->nbytes;
        /* qt_resolve_fmt like the other two expert paths: the raw ?1:?2:3 inference here
         * missed grouped int4 (fmt=4, gs never set) and would mis-tag int3-g64 as int2. */
        int gs=0;
        int fmt=qt_resolve_fmt(l->tw[k]->name,OO[k],II[k],nb,l->tq[k]->nbytes,&gs,NULL);   /* routed expert: never stamped */
        qt[k]->fmt=fmt; qt[k]->O=OO[k]; qt[k]->I=II[k]; qt[k]->gs=gs; qt[k]->qf=NULL;
        qt[k]->q8=(int8_t*)(s->slab+l->pos[k]); qt[k]->q4=s->slab+l->pos[k]; qt[k]->s=fp[k];
    }
    atomic_fetch_add_explicit(&g_prof_io,
        l->tw[0]->nbytes+l->tw[1]->nbytes+l->tw[2]->nbytes+fo*4,
        memory_order_relaxed);
    if(publish_eid) s->eid=l->eid;
    l->finalized=1; return 0;
}
static int uring_wait_all(UringBatch *b){
    for(int i=0;i<b->nload;i++) if(uring_wait_load(b,i)<0) return -1;
    return 0;
}
#endif

/* ============================ PIPE: load ‖ matmul ============================
 * Overlap NVMe expert-weight loads with expert matmul. A small persistent pool
 * of I/O worker pthreads runs the misses' pread (expert_load) into distinct
 * ws[] slabs and sets a per-slot `ready` flag; the MAIN thread walks the block's
 * experts in order, waiting on ready[q] only for the expert it needs right now,
 * and does all matmul_qt on itself (matmul_qt parallelises internally via OpenMP
 * and checks !omp_in_parallel() for GPU dispatch — so it must stay off the omp
 * team and off these I/O threads).
 *
 * Cross-generation safety is provided by a single generation-tagged, lock-free
 * cursor `cur = (gen<<8) | index`. The main thread is the sole writer of `gen`
 * (monotonic bump, so no ABA); workers grab jobs by CAS-advancing the low 8-bit
 * index. THE INVARIANT: a worker reads eids[i]/layer only AFTER its winning CAS,
 * and that CAS's comparand carries the generation — so if `cur`'s gen advanced
 * (a new batch was published), the CAS fails and the worker re-reads, seeing the
 * new generation. A straggler preempted anywhere (wake gap, post-cursor) can
 * therefore NEVER grab a wrong-generation job or read torn batch state: its
 * first act is a gen-checked CAS. dispatch publishes all batch state with
 * relaxed stores and then RELEASE-stores `cur`; each worker ACQUIRE-loads `cur`,
 * so the ready[] reset + eids[]/njobs/layer are visible before any worker acts.
 * The per-expert pipe_wait(ready[q]) in the matmul loop makes every grabbed job
 * complete before the block ends, so no grab outlives its generation — which is
 * why the old `active` counter AND the end-of-block drain barrier are gone (both
 * were redundant with those per-slot waits + the gen-tagged cursor). The mutex/
 * condvar exist ONLY to park/wake idle workers, never for correctness. Gated
 * behind PIPE=1; OFF => the original blocking-load + serial-matmul path runs
 * byte-identically. */
static int g_pipe=0;      /* PIPE=1: async expert-load pipeline. Default ON for Windows
                           * (parsed in main: getenv("PIPE")?:1 on _WIN32, :0 elsewhere).
                           * Keeps expert pread off the forward-pass thread so loads overlap
                           * the matmul. PIPE=0 opts back into the blocking serial path. */
static int g_pipe_nw=8;   /* PIPE_WORKERS=n: I/O worker threads (disk-parallel reads) */
static int g_uring=0;     /* URING=1: Linux io_uring load/completion backend; implies PIPE */
static int g_pipe_block=0;/* COLI_PIPE_BLOCK=1: pipe_wait blocca su una condvar invece dello
                           * spin sched_yield (default OFF = spin byte-identico). EN: a yield
                           * storm on the main thread fights the OpenMP team for cycles during
                           * multi-ms loads; the condvar wake costs ~5us against reads that
                           * cost 0.5-3ms (#159). Pthread pool only: the URING backend has no
                           * waiter spin to replace. */
/* PIPE_WORKERS>0 esplicito nell'env implica PIPE=1: dimensionare il pool
 * dichiara l'intento di usarlo (una campagna intera l'ha impostato con la
 * pipe spenta senza accorgersene). EN: fires ONLY when PIPE is unset in the
 * env AND the platform default left the pipe off (on _WIN32 it already
 * defaults to 1) AND PIPE_WORKERS parses positive — the internal default of
 * 8 does not count, PIPE_WORKERS=0/empty does not, and an explicit PIPE=0
 * always wins. */
static int pipe_workers_imply_pipe(const char *pipe_env, const char *pw_env, int pipe_now){
    return !pipe_now && !pipe_env && pw_env && atoi(pw_env)>0;
}
typedef struct {
    _Atomic uint64_t cur;                         /* (gen<<8)|index; gen main-only, index 0..njobs (≤64) */
    _Atomic int njobs;                            /* current batch job count */
    _Atomic int eids[64];                         /* current batch expert ids */
    _Atomic int layer;                            /* current batch layer */
    _Atomic int ready[64];                        /* per-slot load-done flag */
    pthread_mutex_t mx; pthread_cond_t cv;        /* ONLY for parking/waking idle workers */
    pthread_cond_t cv_done;                       /* COLI_PIPE_BLOCK: signals ready[] transitions */
    Model *m;
    pthread_t th[16]; int nw; int started;
} PipePool;
static PipePool g_pp;

static void *pipe_worker(void *arg){
    (void)arg; PipePool *p=&g_pp; uint64_t seen=0;
    for(;;){
        pthread_mutex_lock(&p->mx);
        while((atomic_load_explicit(&p->cur,memory_order_relaxed)>>8)==seen)
            pthread_cond_wait(&p->cv,&p->mx);
        pthread_mutex_unlock(&p->mx);
        for(;;){
            uint64_t c=atomic_load_explicit(&p->cur,memory_order_acquire);
            seen=c>>8;
            uint32_t i=(uint32_t)(c & 0xFF);
            if(i >= (uint32_t)atomic_load_explicit(&p->njobs,memory_order_relaxed))
                break;                                /* batch drained → re-park */
            if(atomic_compare_exchange_weak_explicit(&p->cur,&c,c+1,
                    memory_order_acq_rel,memory_order_relaxed)){
                int L  =atomic_load_explicit(&p->layer,memory_order_relaxed);
                int eid=atomic_load_explicit(&p->eids[i],memory_order_relaxed); /* AFTER winning CAS */
                expert_load(p->m,L,eid,&p->m->ws[i],1,1);  /* needed-now load: fatal on I/O error (matches serial path); demand=1: this IS moe()'s own miss path */
                atomic_store_explicit(&p->ready[i],1,memory_order_release);
                if(g_pipe_block){                     /* wake a main thread parked in pipe_wait */
                    pthread_mutex_lock(&p->mx);
                    pthread_cond_broadcast(&p->cv_done);
                    pthread_mutex_unlock(&p->mx);
                }
            }
            /* CAS failed → another worker advanced index (or gen advanced): re-loop */
        }
    }
    return NULL;
}
static void pipe_init(Model *m){
    if(g_pp.started) return;
#ifdef __linux__
    if(g_uring){
        if(uring_batch_init(&g_ub_pipe)){ perror("URING=1 io_uring_setup"); exit(1); }
        g_pp.m=m; g_pp.started=1; return;
    }
#endif
    g_pp.m=m; g_pp.nw=g_pipe_nw; if(g_pp.nw>16) g_pp.nw=16; if(g_pp.nw<1) g_pp.nw=1;
    atomic_store(&g_pp.cur,0); atomic_store(&g_pp.njobs,0);
    pthread_mutex_init(&g_pp.mx,NULL); pthread_cond_init(&g_pp.cv,NULL);
    pthread_cond_init(&g_pp.cv_done,NULL);
    for(int i=0;i<g_pp.nw;i++) pthread_create(&g_pp.th[i],NULL,pipe_worker,NULL);
    g_pp.started=1;
}
/* enqueue `njobs` loads (slots ws[0..njobs)); returns immediately, workers run ahead.
 * Order is load-bearing: write all batch state RELAXED, then RELEASE-store cur to
 * publish it, then wake parked workers. */
static void pipe_dispatch(Model *m,int layer,const int *eids,int njobs){
#ifdef __linux__
    if(g_uring){
        uring_batch_reset(&g_ub_pipe);
        for(int q=0;q<njobs;q++){
            int li=uring_load_add(&g_ub_pipe,m,layer,eids[q],&m->ws[q],1);
            if(li!=q){ fprintf(stderr,"URING: expert batch overflow\n"); exit(1); }
        }
        if(uring_submit_batch(&g_ub_pipe)){ perror("URING: submit"); exit(1); }
        return;
    }
#endif
    g_pp.m=m;
    atomic_store_explicit(&g_pp.njobs,njobs,memory_order_relaxed);
    atomic_store_explicit(&g_pp.layer,layer,memory_order_relaxed);
    for(int q=0;q<njobs;q++) atomic_store_explicit(&g_pp.eids[q],eids[q],memory_order_relaxed);
    for(int q=0;q<njobs;q++) atomic_store_explicit(&g_pp.ready[q],0,memory_order_relaxed); /* reset BEFORE publish */
    uint64_t g=(atomic_load_explicit(&g_pp.cur,memory_order_relaxed)>>8)+1;
    atomic_store_explicit(&g_pp.cur,(g<<8),memory_order_release);                          /* PUBLISH */
    pthread_mutex_lock(&g_pp.mx); pthread_cond_broadcast(&g_pp.cv); pthread_mutex_unlock(&g_pp.mx);
}
/* Non-blocking probe of a pipe slot's load-done flag — an ORDERING hint only (the
 * VK block computes ready experts first): callers still pipe_wait() before touching
 * the slab. Under URING completion happens inside finalize, there is no flag to
 * peek — report not-ready and let the wait do the work. */
static inline int pipe_ready(int q){
#ifdef __linux__
    if(g_uring) return 0;
#endif
    return atomic_load_explicit(&g_pp.ready[q],memory_order_acquire)!=0;
}
static inline void pipe_wait(int q){
#ifdef __linux__
    if(g_uring){
        if(uring_finalize_load(&g_ub_pipe,q,1)){ perror("URING: expert load"); exit(1); }
        return;
    }
#endif
    if(g_pipe_block){
        /* Fast path senza lock; poi ri-verifica SOTTO il lock prima di ogni
         * wait. EN: the worker stores ready (release) BEFORE it takes mx to
         * broadcast, so a set flag can never be missed (no lost wakeup). */
        if(atomic_load_explicit(&g_pp.ready[q],memory_order_acquire)) return;
        pthread_mutex_lock(&g_pp.mx);
        while(!atomic_load_explicit(&g_pp.ready[q],memory_order_acquire))
            pthread_cond_wait(&g_pp.cv_done,&g_pp.mx);
        pthread_mutex_unlock(&g_pp.mx);
        return;
    }
    while(!atomic_load_explicit(&g_pp.ready[q],memory_order_acquire)) sched_yield();
}

#ifdef COLI_CUDA
static void expert_host_release(Model *m, ESlot *s){
    if(!s->slab&&!s->fslab) return;
#if defined(__APPLE__) || defined(__linux__) || defined(__FreeBSD__)
    if(s->slab) munlock(s->slab,(size_t)s->slab_cap);
    if(s->fslab) munlock(s->fslab,(size_t)s->fslab_cap*sizeof(float));
#elif defined(_WIN32)
    if(s->slab) compat_munlock(s->slab,(size_t)s->slab_cap);
    if(s->fslab) compat_munlock(s->fslab,(size_t)s->fslab_cap*sizeof(float));
#endif
    int64_t bytes=qt_bytes(&s->g)+qt_bytes(&s->u)+qt_bytes(&s->d);
    /* slab is posix_memalign'd: on Windows that is _aligned_malloc, and plain
     * free() corrupts the CRT heap (0xC0000374) — same bug the compat.h audit
     * fixed at the original expert_load site. fslab is plain malloc/falloc
     * on the CPU path, so its free() stays plain (Metal path frees it before
     * re-alloc and never reaches here with an aligned fslab on _WIN32). */
    if(s->aslab){
#ifdef __linux__
        /* The arena owns the virtual address, not the resident pages.  A GPU
         * slot no longer needs its host copy after the synchronous H2D upload.
         * Keeping those anonymous pages resident made every live REPIN pass
         * grow RSS by one swapped batch (~300 MB at 16 experts).  DONTNEED
         * preserves the reusable slice while returning its pages to Linux;
         * expert_host_ensure reloads them before the slot can run on CPU. */
        madvise(s->slab,(size_t)s->slab_cap,MADV_DONTNEED);
        madvise(s->fslab,(size_t)s->fslab_cap*sizeof(float),MADV_DONTNEED);
#endif
        s->slab=NULL; s->fslab=NULL;               /* detach, keep caps/arena ownership */
    }
    else { compat_aligned_free(s->slab); free(s->fslab); s->slab=NULL; s->fslab=NULL; s->slab_cap=s->fslab_cap=0; }
    QT *q[3]={&s->g,&s->u,&s->d};
    for(int k=0;k<3;k++){ q[k]->qf=NULL; q[k]->q8=NULL; q[k]->q4=NULL; q[k]->s=NULL; }
    m->resident_bytes-=bytes; if(m->resident_bytes<0) m->resident_bytes=0;
}
static void expert_host_ensure(Model *m, int layer, ESlot *s){
    if(s->slab) return;
    if(s->aslab){ s->slab=s->aslab; s->fslab=s->afslab; }  /* re-attach the arena slice; caps survived release */
    /* re-materializing a GPU-resident expert's host copy, not a routing miss: demand=0 */
    expert_load(m,layer,s->eid,s,1,0);                     /* rebuild the QT views (release NULLed them) + reload */
}
#endif

/* prefetch asincrono dei pesi di un expert (e delle sue scale .qs): avvia il readahead
 * cosi' le letture sincrone successive trovano la page-cache calda.
 * Sotto g_direct i PESI vengono letti con O_DIRECT (bypassa la page-cache, vedi
 * expert_load): il WILLNEED su di essi scalda pagine che la lettura di domanda non
 * consuma -> readahead sprecato sul disco, la risorsa piu' scarsa nello streaming.
 * Le scale .qs restano SEMPRE bufferizzate (pread sul fd normale), quindi il loro
 * WILLNEED resta utile anche con DIRECT=1. fadvise e' solo consultivo: saltarlo non
 * cambia mai l'output (bit-identico), riduce solo I/O sprecato.
 * The readahead targets the SAME replica that will serve the pread (expert_route
 * is deterministic).
 * EN: under O_DIRECT the weights bypass the page cache, so their WILLNEED is wasted;
 * the .qs scales are always buffered, so keep theirs. Advisory hint -> output-preserving. */
static void expert_prefetch(Model *m, int layer, int eid){
    char nm[300]; int rep=expert_route(layer,eid);
    const char *suf[3]={"gate_proj.weight","up_proj.weight","down_proj.weight"};
    for(int k=0;k<3;k++){
        snprintf(nm,sizeof(nm),"model.layers.%d.mlp.experts.%d.%s",layer,eid,suf[k]);
        if(!g_direct) st_prefetch_rep(&m->S,nm,rep);
        char qs[320]; snprintf(qs,sizeof(qs),"%s.qs",nm); st_prefetch_rep(&m->S,qs,rep);
    }
}

/* ---- helper per l'ABSORPTION: accesso per-riga ai QT quantizzati ---- */
/* acc[0..I) += coef * W[row,:] (dequant al volo) */
static void qt_addrow(const QT *t, int row, float coef, float *acc){
    int I=t->I;
    if(t->fmt==0){ const float *w=t->qf+(int64_t)row*I; for(int i=0;i<I;i++) acc[i]+=coef*w[i]; return; }
    /* fmt=4 PRIMA del calcolo di c: s[] e' [O,ng] per-gruppo, s[row] sarebbe la scala
     * sbagliata. Senza questo ramo il fall-through int2 decodificava i nibble int4 come
     * coppie di valori a 2 bit — lo stesso bug di #298 sui kernel absorb CUDA, lato CPU.
     * EN: fmt=4 BEFORE computing c: s[] is [O,ng] per-group, s[row] would be the wrong
     * scale. Without this branch the int2 fall-through decoded int4 nibbles as pairs of
     * 2-bit values — the same bug #298 fixed in the CUDA absorb kernels, CPU side. */
    if(t->fmt==4){ const uint8_t *w=t->q4+(int64_t)row*((I+1)/2);
        int gs=t->gs, ng=(I+gs-1)/gs; const float *scl=t->s+(int64_t)row*ng;
        for(int i=0;i+1<I;i+=2){ uint8_t b=w[i>>1];
            acc[i]  +=coef*scl[i/gs]    *((int)(b&0xF)-8);
            acc[i+1]+=coef*scl[(i+1)/gs]*((int)(b>>4)-8); }
        if(I&1){ uint8_t b=w[I>>1]; acc[I-1]+=coef*scl[(I-1)/gs]*((int)(b&0xF)-8); } return; }
    /* fmt=5 likewise before c: int3-g64 scales are per-GROUP [O,ng], not s[row] */
    if(t->fmt==5){ const uint8_t *w=t->q4+(int64_t)row*i3_rowbytes(I);
        const float *sr=t->s+(int64_t)row*i3_groups(I); int64_t ng=i3_groups(I);
        for(int64_t g=0; g<ng; g++){ const uint8_t *lo=w+g*I3_GBYTES, *hi=lo+16;
            float cg=coef*sr[g]; int base=(int)(g*I3_GROUP), n=I-base<I3_GROUP?I-base:I3_GROUP;
            for(int k=0;k<n;k++){ unsigned u=((lo[k>>2]>>((k&3)*2))&3)|(((hi[k>>3]>>(k&7))&1)<<2);
                acc[base+k]+=cg*(float)((int)u-4); } }
        return; }
    /* GUARD (fix round 2, engine defect -- clean-room conformance trial found a real
     * SIGSEGV, reproduced): fmt 0/4/5 already returned above; everything below this
     * point assumes a PER-ROW scale (t->s[row]) followed by fmt=1 (int8, explicit
     * branch), fmt=2 (int4 packed, explicit branch), or the tail's own IMPLICIT fmt=3
     * (int2 packed, the final unconditional block) -- there was no guard stopping any
     * OTHER fmt from reaching here. fmt=6 (E8/IQ3): t->s is a FIXED 4-byte tag (ONE
     * float total, qsalloc(1) in qt_from_disk), not O floats -- t->s[row] for row>0 is
     * a heap OVERREAD, and the untouched fall-through then misreads t->q4's real E8
     * lattice bytes as int2-packed data (same bug SHAPE as #298's CUDA absorb-kernel
     * fix, and the same one this file's own fmt=4/5 branches above were added to
     * dodge -- fmt=6 was simply missed). fmt=8 (fp8-e4m3-b128): t->s holds
     * ceil(O/128)*ceil(I/128) per-block floats, not O -- t->s[row] overreads for
     * row>=nblk (e.g. a [130,130] tensor has nblk=4, so every row past 3 already reads
     * out of bounds), AND t->q4 is NULL for fmt=8 (raw bytes live in t->q8 instead,
     * same convention as fmt=1 -- see the QT struct comment), so the fall-through's
     * `t->q4+(int64_t)row*((I+3)/4)` dereferences NULL-plus-offset: SIGSEGV,
     * reproduced (see the report's proof-of-bite transcript). Refuse loudly instead --
     * this function has no byte-count context of its own to validate against (it only
     * ever sees an already-resolved QT), so "unsupported fmt" is the only check
     * available, same "refuse rather than misread" discipline qt_resolve_fmt applies
     * at load time. */
    if(t->fmt!=1 && t->fmt!=2 && t->fmt!=3){
        fprintf(stderr,"qt_addrow: unsupported fmt=%d for the per-row-scale absorb path "
            "(only fmt 1/2/3 reach this point; fmt 0/4/5 are handled above and return "
            "before it) -- refusing rather than misread t->s[row]/t->q4\n", t->fmt);
        exit(1);
    }
    float c=coef*t->s[row];
    if(t->fmt==1){ const int8_t *w=t->q8+(int64_t)row*I; for(int i=0;i<I;i++) acc[i]+=c*(float)w[i]; return; }
    if(t->fmt==2){ const uint8_t *w=t->q4+(int64_t)row*((I+1)/2);
#if defined(__AVX512F__) && defined(__AVX512BW__)
        axpy_i4f_avx512(w,c,acc,I); return;   /* bit-identical: one fma per element */
#endif
        for(int i=0;i+1<I;i+=2){ uint8_t b=w[i>>1]; acc[i]+=c*((int)(b&0xF)-8); acc[i+1]+=c*((int)(b>>4)-8); }
        if(I&1){ uint8_t b=w[I>>1]; acc[I-1]+=c*((int)(b&0xF)-8); } return; }
    const uint8_t *w=t->q4+(int64_t)row*((I+3)/4);
    for(int i=0;i<I;i++){ uint8_t b=w[i>>2]; acc[i]+=c*((int)((b>>((i&3)*2))&3)-2); }
}
/* y[0..n) = W[r0+j,:]·x  (matvec su una FETTA di righe del QT) */
static void qt_matvec_rows(const QT *t, int r0, int n, const float *x, float *y){
    int I=t->I;
    for(int j=0;j<n;j++){ int row=r0+j; double a=0;
        if(t->fmt==0){ const float *w=t->qf+(int64_t)row*I; for(int i=0;i<I;i++) a+=(double)w[i]*x[i]; }
        else if(t->fmt==4){ /* grouped int4: per-group scale */
            const uint8_t *w=t->q4+(int64_t)row*((I+1)/2); int gs=t->gs,ng=(I+gs-1)/gs;
            const float *scl=t->s+(int64_t)row*ng;
            for(int g=0;g*gs<I;g++){ int base=g*gs,glen=gs; if(base+glen>I)glen=I-base; float sc=scl[g]; float acc=0;
                for(int i=base;i+1<base+glen;i+=2){ uint8_t b=w[i>>1]; acc+=((int)(b&0xF)-8)*x[i]+((int)(b>>4)-8)*x[i+1]; }
                if(glen&1){ uint8_t b=w[(base+glen-1)>>1]; acc+=((int)(b&0xF)-8)*x[base+glen-1]; }
                a+=acc*sc; } }
        else if(t->fmt==1){ const int8_t *w=t->q8+(int64_t)row*I; float s=t->s[row];
            float acc=0; for(int i=0;i<I;i++) acc+=(float)w[i]*x[i]; a=acc*s; }
        else if(t->fmt==2){ const uint8_t *w=t->q4+(int64_t)row*((I+1)/2); float s=t->s[row]; float acc=0;
#if defined(__AVX512F__) && defined(__AVX512BW__)
            /* same accumulation-order tradeoff (and gate) as matmul_i4's I4_ACC512 path */
            if(g_i4_acc512 && !(I&31)){ y[j]=dot_i4f_avx512(w,x,I)*s; continue; }
#endif
            for(int i=0;i+1<I;i+=2){ uint8_t b=w[i>>1]; acc+=((int)(b&0xF)-8)*x[i]+((int)(b>>4)-8)*x[i+1]; }
            if(I&1){ uint8_t b=w[I>>1]; acc+=((int)(b&0xF)-8)*x[I-1]; } a=acc*s; }
        else if(t->fmt==4){ /* per-gruppo, come matmul_i4_grouped / per-group, as matmul_i4_grouped */
            const uint8_t *w=t->q4+(int64_t)row*((I+1)/2);
            int gs=t->gs, ng=(I+gs-1)/gs; const float *scl=t->s+(int64_t)row*ng;
            for(int g=0; g*gs<I; g++){ int base=g*gs, end=base+gs>I?I:base+gs; float acc=0;
                for(int i=base;i<end;i++){ uint8_t b=w[i>>1];
                    acc+=(float)((i&1)?((int)(b>>4)-8):((int)(b&0xF)-8))*x[i]; }
                a+=(double)acc*scl[g]; } }
        else if(t->fmt==5){ const uint8_t *w=t->q4+(int64_t)row*i3_rowbytes(I);
            const float *sr=t->s+(int64_t)row*i3_groups(I); int64_t ng=i3_groups(I);
            for(int64_t g=0; g<ng; g++){ const uint8_t *lo=w+g*I3_GBYTES, *hi=lo+16;
                int base=(int)(g*I3_GROUP), n=I-base<I3_GROUP?I-base:I3_GROUP; float acc=0;
                for(int k=0;k<n;k++){ unsigned u=((lo[k>>2]>>((k&3)*2))&3)|(((hi[k>>3]>>(k&7))&1)<<2);
                    acc+=(float)((int)u-4)*x[base+k]; }
                a+=(double)(acc*sr[g]); } }
        /* fmt=3 (int2 packed, per-row scale) is the only fmt this final arm legitimately
         * handles -- fmt 0/4/5 matched above, fmt 1/2 have their own explicit branches
         * above too. Same GUARD and same reasoning as qt_addrow's (fix round 2, engine
         * defect): fmt=6's t->s is a fixed 4-byte tag (t->s[row] overreads for row>0),
         * fmt=8's t->s holds per-128x128-block floats (t->s[row] overreads for
         * row>=nblk) and t->q4 is NULL for fmt=8 -- both would have silently misread or
         * crashed here exactly like qt_addrow did before its own fix; refuse instead. */
        else if(t->fmt==3){ const uint8_t *w=t->q4+(int64_t)row*((I+3)/4); float s=t->s[row]; float acc=0;
            for(int i=0;i<I;i++){ uint8_t b=w[i>>2]; acc+=((int)((b>>((i&3)*2))&3)-2)*x[i]; } a=acc*s; }
        else {
            fprintf(stderr,"qt_matvec_rows: unsupported fmt=%d for the per-row-scale absorb "
                "path (only fmt 0/1/2/3/4/5 are handled) -- refusing rather than misread "
                "t->s[row]/t->q4\n", t->fmt);
            exit(1);
        }
        y[j]=(float)a;
    }
}
static int g_absorb=-1;
static int g_metal_prefill=0; /* default 0: S>4 prefill attention stays on the CPU (bit-exact). COLI_METAL_PREFILL=1 opts it onto the GPU (~4x, near-tie divergence — see docs/metal.md, #622) */
/* KV8=1: cache latente Lc/Rc in fp8 e4m3 + scala f32 per riga (~4x meno RAM del f32).
 * CPU-only in this PR — sui percorsi CUDA/Metal che leggono righe f32 si spegne da
 * solo (guardie !g_kv8), e forza COLI_CUDA_PIPE=0 (il pipe-prefill legge righe f32).
 * I kernel nativi CUDA/Metal arrivano nei follow-up con hardware owner. */
/* g_kv8 defined before the kv_persist.h include (above). */
/* KV_TQ=3|4: tier TurboQuant/PolarQuant (rotazione Hadamard randomizzata +
 * trasformata polare ricorsiva; raggio = norma L2 nella scala per-riga, solo gli
 * angoli nei byte). CPU-only, come KV8: si spegne dove i percorsi leggono righe
 * f32. Mutuamente esclusivo con KV8. Riusa Lc8/Rc8 (byte impacchettati, righe di
 * coli_kvq_row_bytes) + Lsc/Rsc (raggio f32 per riga). g_tq_bits = livello-1 (3 o 4). */
/* g_tq, g_tq_bits, g_tq_codec defined before the kv_persist.h include (above). */
#ifdef COLI_CUDA
static int g_cuda_pipe=0;   /* COLI_CUDA_PIPE=1: prefill attention chain resident on the layer home device */
static int g_cuda_router=0; /* COLI_CUDA_ROUTER=1 (#431 PR-A): router on the layer home device at decode */
static int g_cuda_resid=0;  /* COLI_CUDA_RESID=1 (#431 PR-C0): expert-group results stay on device */
/* pipe -> moe handoff for the resident group path (set per layer by
 * pipe_layer_sparse, consumed by moe()'s group dispatch, cleared after take) */
static int g_pres_home=-1;                /* home device, -1 = path off        */
static const float *g_pres_xsrc;          /* nrm_d on the home device          */
static float *g_pres_slots;               /* [ndev][D] partial slots on home   */
static int g_pres_used[COLI_CUDA_MAX_DEVICES], g_pres_nused;
static ESlot *g_pres_refs[COLI_CUDA_MAX_DEVICES*64];
static int g_pres_nrefs;
#endif   /* ABSORB: -1 auto (decode S<=4), 0 mai, 1 sempre (test) */
static int g_dsa_force=0; /* DSA_FORCE=1: selezione sempre attiva (test: top-min(k,T)=denso) */
static int cmp_fdesc(const void *a,const void *b){
    float x=*(const float*)a, y=*(const float*)b; return x<y?1:x>y?-1:0; }

/* PARTIAL SELECT (quickselect, Hoare partition, DESCending). After this call the k
 * LARGEST elements of a[0..n) are in a[0..k) in unspecified order; the (k+1)-th and
 * beyond are untouched-or-smaller. O(n) average, O(n^2) pathological (mitigated by
 * median-of-three below) — and unlike a full qsort it never orders more than needed.
 *
 * Why this exists (#356): the DSA top-keep in attention_rows previously full-qsorted
 * all nk context scores (O(nk log nk)) per layer per token just to read ONE value --
 * the keep-th largest (the threshold). quickselect finds that pivot in O(nk) average,
 * and the position-order scans that build dst[] are unchanged, so the kept set is
 * bit-identical. Mirrors the sampling-side fix in #335 (heap partial-select there).
 *
 * NOT a stable partition: callers must derive the threshold and then re-scan the
 * ORIGINAL array (the DSA code does exactly this) rather than reading a[0..k). */
static void partial_select_desc(float *a, int n, int k){
    if(k<=0) return;
    if(k>=n) return;                 /* nothing to partition: all kept */
    int lo=0, hi=n-1;
    while(lo<hi){
        /* median-of-three pivot to dodge the O(n^2) path on sorted/reverse input */
        int mid=lo+((hi-lo)>>1);
        if(a[mid]>a[lo]){ float t=a[lo]; a[lo]=a[mid]; a[mid]=t; }
        if(a[hi]>a[lo]){  float t=a[lo]; a[lo]=a[hi];  a[hi]=t;  }
        if(a[mid]>a[hi]){ float t=a[hi]; a[hi]=a[mid]; a[mid]=t; }
        float piv=a[hi];
        int i=lo, j=hi;
        for(;;){
            while(a[i]>piv) i++;     /* desc: large values go left */
            while(j>lo && a[j]<piv) j--;
            if(i>=j) break;
            float t=a[i]; a[i]=a[j]; a[j]=t; i++; if(i>j) break; j--;
        }
        /* partition point: a[lo..i) are all >= piv, a[i..hi] are all <= piv */
        if(k<=i-1) hi=i-1;          /* the k-th largest is in the left partition */
        else       lo=i;            /* it's in the right partition */
    }
}

/* attenzione MLA con KV-cache compressa, su token nuovi x[S,hidden], pos_base = pos del primo */
/* kvs/pos describe a ragged decode batch: each row may belong to a different
 * sequence.  NULL keeps the original contiguous, currently-bound KV path. */
#ifdef COLI_CUDA
/* Ombra KV su device per il DECODE: righe [0,upto) valide sulla scheda di kv_b.
 * L'host resta canonico; l'ombra si riallinea in blocco quando resta indietro e
 * viene invalidata da kv_bind / dalla riscrittura di righe gia' specchiate. */
static int kv_dev_sync(Model *m, Layer *l, int layer, int upto){
    Cfg *c=&m->c; int kvl=c->kv_lora, R=c->qk_rope, dev=l->kv_b.cuda_device;
    if(upto>m->max_t) return 0;
    if(!m->kv_dev_L[layer]){
        m->kv_dev_L[layer]=(float*)coli_cuda_pipe_alloc(dev,(size_t)m->max_t*kvl*4);
        m->kv_dev_R[layer]=(float*)coli_cuda_pipe_alloc(dev,(size_t)m->max_t*R*4);
        m->kv_dev_valid[layer]=0;
        if(!m->kv_dev_L[layer]||!m->kv_dev_R[layer]) return 0;
    }
    int v=m->kv_dev_valid[layer];
    if(v<upto){
        if(!coli_cuda_pipe_upload(dev,m->kv_dev_L[layer]+(size_t)v*kvl,
            coli_kv_row(m->kv->Lc[layer],v,kvl),(size_t)(upto-v)*kvl*4)||
           !coli_cuda_pipe_upload(dev,m->kv_dev_R[layer]+(size_t)v*R,
            coli_kv_row(m->kv->Rc[layer],v,R),(size_t)(upto-v)*R*4)) return 0;
        m->kv_dev_valid[layer]=upto;
    }
    return 1;
}

/* Inc.1a — catena attention residente sul device del layer / attention chain
 * resident on the layer home device. Proiezioni q/kv, norme, RoPE, batch
 * attention e o_proj girano sulla scheda di kv_b; scaricano solo out [S,D],
 * i nuovi record KV [S,kvl+R] e nulla altro. Ritorna 0 su qualsiasi errore:
 * il chiamante riesegue il percorso CPU (idempotente). */
static int attn_pipe_prefill(Model *m, Layer *l, int layer, const float *x, int x_is_dev,
                             int S, int pos_base, float *out, float *out_dev){
    Cfg *c=&m->c; int H=c->n_heads, D=c->hidden, qh=c->qk_head;
    int kvl=c->kv_lora, R=c->qk_rope, ql=c->q_lora;
    int dev=l->kv_b.cuda_device;
    if(l->q_a.cuda_device!=dev||l->q_b.cuda_device!=dev||
       l->kv_a.cuda_device!=dev||l->o.cuda_device!=dev) return 0;
    int st0=m->kv_start[layer], T=pos_base+S-st0, old=pos_base-st0;
    if(T<S||T>8192) return 0;
    double t0=now_s();
    size_t xb=(size_t)S*D*4, qrb=(size_t)S*ql*4, qb=(size_t)S*H*qh*4;
    size_t cb=(size_t)S*(kvl+R)*4, lb=(size_t)T*kvl*4, rb=(size_t)T*R*4;
    float *chost=NULL; int ok=0;
    /* scratch persistenti (slot fissi per device): zero churn di cudaMalloc */
    float *xd =x_is_dev?(float*)x:coli_cuda_pipe_scratch(dev,0,xb);
    float *qrd=coli_cuda_pipe_scratch(dev,1,qrb);
    float *qd =coli_cuda_pipe_scratch(dev,2,qb),  *cd =coli_cuda_pipe_scratch(dev,3,cb);
    float *ld_=coli_cuda_pipe_scratch(dev,4,lb),  *rd =coli_cuda_pipe_scratch(dev,5,rb);
    float *w1 =coli_cuda_pipe_scratch(dev,6,(size_t)ql*4);
    float *w2 =coli_cuda_pipe_scratch(dev,7,(size_t)kvl*4);
    chost=(float*)malloc(cb);
    if(!xd||!qrd||!qd||!cd||!ld_||!rd||!w1||!w2||!chost) goto done;
    if((!x_is_dev&&!coli_cuda_pipe_upload(dev,xd,x,xb))||
       !coli_cuda_pipe_upload(dev,w1,l->q_a_ln,(size_t)ql*4)||
       !coli_cuda_pipe_upload(dev,w2,l->kv_a_ln,(size_t)kvl*4)) goto done;
    /* proiezioni + norme + rope, tutto sul device */
    if(!coli_cuda_pipe_gemm(l->q_a.cuda,qrd,xd,S)) goto done;
    if(!coli_cuda_pipe_rmsnorm(dev,qrd,qrd,w1,S,ql,c->eps)) goto done;
    if(!coli_cuda_pipe_gemm(l->q_b.cuda,qd,qrd,S)) goto done;
    if(!coli_cuda_pipe_rope_base(dev,qd,pos_base,S*H,qh,c->qk_nope,R,H,c->theta)) goto done;
    if(!coli_cuda_pipe_gemm(l->kv_a.cuda,cd,xd,S)) goto done;
    if(!coli_cuda_pipe_rmsnorm_s(dev,cd,cd,w2,S,kvl,c->eps,kvl+R,kvl+R)) goto done;
    if(!coli_cuda_pipe_rope_base(dev,cd,pos_base,S,kvl+R,kvl,R,1,c->theta)) goto done;
    /* cache latente [T,kvl] + rot [T,R] contigue: righe vecchie da host, nuove da cd */
    if(old>0){
        if(!coli_cuda_pipe_upload(dev,ld_,coli_kv_row(m->Lc[layer],st0,kvl),(size_t)old*kvl*4)||
           !coli_cuda_pipe_upload(dev,rd,coli_kv_row(m->Rc[layer],st0,R),(size_t)old*R*4)) goto done;
    }
    if(!coli_cuda_pipe_copy2d(dev,ld_+(size_t)old*kvl,kvl,cd,kvl+R,kvl,S)) goto done;
    if(!coli_cuda_pipe_copy2d(dev,rd+(size_t)old*R,R,cd+kvl,kvl+R,R,S)) goto done;
    /* KV host resta canonica: scarica i record nuovi (gia' normati+ropati) */
    if(!coli_cuda_pipe_download(dev,cd,chost,cb)) goto done;
    for(int s=0;s<S;s++){
        memcpy(coli_kv_row(m->Lc[layer],pos_base+s,kvl),chost+(size_t)s*(kvl+R),kvl*4);
        memcpy(coli_kv_row(m->Rc[layer],pos_base+s,R),chost+(size_t)s*(kvl+R)+kvl,R*4);
    }
    if(m->kv_dev_valid[layer]>pos_base) m->kv_dev_valid[layer]=pos_base;
    m->t_aproj+=now_s()-t0; t0=now_s();
#ifdef COLI_CUDA
    /* Negativo (2026-07-13): P2P a stella dal device di casa serializza ~95MB/layer
     * sul suo link PCIe — attention 26->41-44s. Resta opt-in per topologie NVLink. */
    if(out_dev && l->n_kv_b_shard>1 &&
       getenv("COLI_CUDA_PIPE_SHARD") && atoi(getenv("COLI_CUDA_PIPE_SHARD"))){
        /* head-shard nel pipeline: q gia' sul device di casa. Per ogni scheda:
         * slice di q (repack strided->contiguo), broadcast latent+rope via P2P,
         * score parallelo sui rispettivi head, ctx slice riportata a casa e
         * ricomposta, poi o_proj residente. */
        int n=l->n_kv_b_shard, vh=c->v_head;
        size_t ctxb=(size_t)S*H*vh*4;
        size_t stage_one=(size_t)S*H*(size_t)(c->qk_head>vh?c->qk_head:vh)*4;
        float *ctx_full=coli_cuda_pipe_scratch(dev,16,ctxb);
        float *stage=coli_cuda_pipe_scratch(dev,17,stage_one*n);
        int ok_sh=(ctx_full&&stage)?1:0;
        if(ok_sh){
            #pragma omp parallel for schedule(static) reduction(&:ok_sh)
            for(int d2=0;d2<n;d2++){
                int hn=l->shard_hn[d2], h0=l->shard_h0[d2];
                int sdev=coli_cuda_tensor_device(l->kv_b_shard[d2]);
                float *st=stage+(size_t)d2*(stage_one/4);
                size_t qsb=(size_t)S*hn*qh*4, csb=(size_t)S*hn*vh*4;
                float *qs_r=coli_cuda_pipe_scratch(sdev,18,qsb);
                float *ld_r=coli_cuda_pipe_scratch(sdev,19,(size_t)T*kvl*4);
                float *rr_r=coli_cuda_pipe_scratch(sdev,20,(size_t)T*R*4);
                float *cx_r=coli_cuda_pipe_scratch(sdev,21,csb);
                int okd=qs_r&&ld_r&&rr_r&&cx_r;
                /* slice di q: [S,H,qh] -> [S,hn,qh] contigua sul device di casa */
                okd=okd&&coli_cuda_pipe_copy2d(dev,st,hn*qh,qd+(size_t)h0*qh,H*qh,hn*qh,S);
                okd=okd&&coli_cuda_pipe_peer_copy(sdev,qs_r,dev,st,qsb);
                okd=okd&&coli_cuda_pipe_peer_copy(sdev,ld_r,dev,ld_,(size_t)T*kvl*4);
                okd=okd&&coli_cuda_pipe_peer_copy(sdev,rr_r,dev,rd,(size_t)T*R*4);
                okd=okd&&coli_cuda_attention_absorb_batch_dev(l->kv_b_shard[d2],cx_r,qs_r,ld_r,rr_r,
                        S,hn,c->qk_nope,R,vh,kvl,T,c->attn_scale);
                okd=okd&&coli_cuda_pipe_peer_copy(dev,st,sdev,cx_r,csb);
                okd=okd&&coli_cuda_pipe_copy2d(dev,ctx_full+(size_t)h0*vh,H*vh,st,hn*vh,hn*vh,S);
                ok_sh&=okd;
            }
        }
        if(ok_sh){
            ok=coli_cuda_pipe_gemm(l->o.cuda,out_dev,ctx_full,S)&&coli_cuda_pipe_sync(dev);
        } else ok=0;
        if(!ok)
            ok=coli_cuda_attention_project_batch_dev_out(l->kv_b.cuda,l->o.cuda,out_dev,qd,ld_,rd,
                S,H,c->qk_nope,R,c->v_head,kvl,T,c->attn_scale);
    } else
#endif
    ok=out_dev?coli_cuda_attention_project_batch_dev_out(l->kv_b.cuda,l->o.cuda,out_dev,qd,ld_,rd,
            S,H,c->qk_nope,R,c->v_head,kvl,T,c->attn_scale)
              :coli_cuda_attention_project_batch_dev(l->kv_b.cuda,l->o.cuda,out,qd,ld_,rd,
            S,H,c->qk_nope,R,c->v_head,kvl,T,c->attn_scale);
    m->t_acore+=now_s()-t0;
done:
    free(chost);                              /* gli scratch device restano al contesto */
    return ok;
}

/* Decode-scale prefix only: upload x once and reuse it for q_a and kv_a. Keep
 * the stock CPU RMSNorm between q_a and q_b so greedy output stays exact, then
 * return Q/comp to the stock attention path. */
static int attn_pipe_prefix(Model *m,Layer *l,const float *x,int S,float *QR,float *Q,float *comp){
    Cfg *c=&m->c; int D=c->hidden,ql=c->q_lora,kvl=c->kv_lora,R=c->qk_rope;
    int dev=l->q_a.cuda_device;
    if(S<1||S>4||dev<0||l->q_b.cuda_device!=dev||l->kv_a.cuda_device!=dev) return 0;
    size_t xb=(size_t)S*D*4,qrb=(size_t)S*ql*4;
    size_t qb=(size_t)S*c->n_heads*c->qk_head*4,cb=(size_t)S*(kvl+R)*4;
    float *xd=coli_cuda_pipe_scratch(dev,0,xb);
    float *qrd=coli_cuda_pipe_scratch(dev,1,qrb);
    float *qd=coli_cuda_pipe_scratch(dev,2,qb);
    float *cd=coli_cuda_pipe_scratch(dev,3,cb);
    if(!xd||!qrd||!qd||!cd) return 0;
    if(!coli_cuda_pipe_upload(dev,xd,x,xb)||
       !coli_cuda_pipe_gemm(l->q_a.cuda,qrd,xd,S)||
       !coli_cuda_pipe_download(dev,qrd,QR,qrb)) return 0;
    for(int s=0;s<S;s++) rmsnorm(QR+(int64_t)s*ql,QR+(int64_t)s*ql,l->q_a_ln,ql,c->eps);
    if(!coli_cuda_pipe_upload(dev,qrd,QR,qrb)||
       !coli_cuda_pipe_gemm(l->q_b.cuda,qd,qrd,S)||
       !coli_cuda_pipe_gemm(l->kv_a.cuda,cd,xd,S)||
       !coli_cuda_pipe_download(dev,qd,Q,qb)||
       !coli_cuda_pipe_download(dev,cd,comp,cb)) return 0;
    return 1;
}
#endif

/* POSITIVE allowlist for the two fused Metal decode gates below
 * (attention_rows / layer_forward_rows): only the mm_gemv shader's explicit
 * INTEGER-format branches (fmt 1/2/3/4) may enter the fused path -- the
 * exact set that was BOTH admitted by the old negative guard AND computed
 * correctly. Anything else falls back to the CPU path (correct, slower).
 * INVARIANT this predicate names: a NEGATIVE guard (the previous
 * `fmt!=8` form) is fail-OPEN for every format it doesn't name --
 * bind_gemv (backend_metal.mm) has no fmt allowlist of its own, and the
 * mm_gemv shader's dispatch chain ends in an unconditional f32 fallback
 * branch, so a resident fmt=5 or fmt=6 tensor slipping through would have
 * its REAL weight bytes (they live in q4, which is exactly what the WP_()
 * macro hands the kernel) silently misread as f32 garbage, with no error.
 * This form is fail-CLOSED: a future format is excluded until someone
 * proves it on the fused path and adds it here. fmt=3 (int2) is included
 * because it was legal-and-correct here before this change (explicit
 * mm_gemv branch, real q4 bytes) -- excluding it would be this change's
 * only regression. fmt=0 (f32) stays excluded: it never actually fused --
 * WP_() hands the shader q4, which is NULL for fmt=0 (weights live in qf)
 * -- so admitting it would be dead permissiveness, not a preserved
 * behavior. fmt=5/6 (misread hazard), fmt=8 (NULL-q4 variant), and every
 * future format remain excluded.
 * Defined unconditionally (not #ifdef COLI_METAL) so the CPU test build
 * can unit-test the truth table (tests/test_fp8_load.c Part F). */
static inline int metal_fused_fmt_ok(int fmt){ return fmt==1 || fmt==2 || fmt==3 || fmt==4; }

/* Per-layer companion to metal_fused_fmt_ok: ONE predicate over all the weight tensors
 * the fused decode kernels bind, returning a bitmask of offending tensors
 * (METAL_FUSED_*, 0 == fully fused-eligible). Both fused gates test it against the mask
 * of tensors their kernel actually binds (METAL_FUSED_ATTN_TENSORS /
 * METAL_FUSED_LAYER_TENSORS), and metal_fmt_gate_notice (model_init) reports every set
 * bit — the notice consumes the gates' own condition, so the two can never drift apart
 * again (the previous kv_b-only notice re-typed the gate condition and was already
 * stale: it omitted the `!g_moe_exact` term below). kv_b's term is the gates' own:
 * fmt==2 (int4 per-row, its absorb kernel's native format) always serves; fmt==4
 * (grouped int4, #587) serves only while g_moe_exact is off — COLI_METAL_MOE_EXACT=1
 * keeps grouped kv_b off the fused path. sh_gate/sh_up/sh_down are checked on sparse
 * layers only: dense layers never load them (fmt stays 0) and layer_forward_rows' fused
 * path already requires l->sparse, so flagging them there would be pure false positive.
 * Reads only the layer's fmt/sparse fields and the g_moe_exact mode flag — no side
 * effects. */
static unsigned metal_fused_layer_fmt_miss(const Layer *l){
    unsigned miss=0;
    if(!(l->kv_b.fmt==2 || (l->kv_b.fmt==4 && !g_moe_exact))) miss|=METAL_FUSED_KV_B;
    if(!metal_fused_fmt_ok(l->q_a.fmt))     miss|=METAL_FUSED_Q_A;
    if(!metal_fused_fmt_ok(l->q_b.fmt))     miss|=METAL_FUSED_Q_B;
    if(!metal_fused_fmt_ok(l->kv_a.fmt))    miss|=METAL_FUSED_KV_A;
    if(!metal_fused_fmt_ok(l->o.fmt))       miss|=METAL_FUSED_O;
    if(l->sparse){
        if(!metal_fused_fmt_ok(l->sh_gate.fmt)) miss|=METAL_FUSED_SH_GATE;
        if(!metal_fused_fmt_ok(l->sh_up.fmt))   miss|=METAL_FUSED_SH_UP;
        if(!metal_fused_fmt_ok(l->sh_down.fmt)) miss|=METAL_FUSED_SH_DOWN;
    }
    return miss;
}

/* #1151 follow-up: the flash/gather kvb arms under quantized KV.
 * kv_lc_rows_f32 dequantizes latent rows [t0, t0+n) into an f32 staging buffer
 * (the same per-row codecs the one-shot path uses); the roped-key reads
 * inline the one-shot path's three-way branch so each representation keeps
 * its exact accumulation order. */
static void kv_lc_rows_f32(Model *m, int layer, int64_t t0, int64_t n, float *dst, Cfg *c){
    if(g_tq){
        int lbb=coli_kvq_row_bytes(c->kv_lora,g_tq_bits,g_tq_codec);
        for(int64_t t=t0;t<t0+n;t++)
            coli_kvq_dequant_row(coli_kv_row8(m->Lc8[layer],t,lbb), m->Lsc[layer][t],
                                 dst+(t-t0)*c->kv_lora, c->kv_lora, g_tq_bits, g_tq_codec);
    } else {
        for(int64_t t=t0;t<t0+n;t++)
            coli_kv8_dequant_row(coli_kv_row8(m->Lc8[layer],t,c->kv_lora), m->Lsc[layer][t],
                                 dst+(t-t0)*c->kv_lora, c->kv_lora);
    }
}
static void attention_rows(Model *m, Layer *l, int layer, float *x, int S, int pos_base,
                           KVState *const *kvs, const int *positions, float *out){
    Cfg *c=&m->c; int H=c->n_heads, D=c->hidden, qh=c->qk_head, vh=c->v_head;
    int kvb_dim=H*(c->qk_nope+vh), Tk=pos_base+S;
    double ta0=now_s();
#ifdef COLI_METAL
    /* Fused decode attention on GPU: whole layer in one command buffer (keeps the GPU hot).
     * S<=4 absorption path with st0==0, DSA selection inactive, and GLM-5.2 int4 dims.
     * RAGGED GUARD (!kvs): the kernel takes ONE Lc/Rc pair and ONE pos_base — it assumes
     * row s is token pos_base+s of the SAME sequence. The batched mux decode
     * (step_decode_batch) passes per-row kvs[]/positions[] with pos_base=0, so the kernel
     * would rope every row at position 0 and attend over a 1-token window of the wrong
     * cache -> greedy decode hits EOS at token 2 (mux answers truncated to 1 token).
     * Ragged rows take the CPU absorb path below, which reads kvs[s]/positions[s].
     * QUANT GUARD (!g_kv8&&!g_tq): the fused kernel reads f32 Lc/Rc rows, which are not
     * even allocated under KV8/KV_TQ (byte caches + per-row scales instead). Quantized
     * KV takes the CPU absorb path below; the fp8/TQ Metal kernels are the follow-up PR.
     * metal_fused_layer_fmt_miss & METAL_FUSED_ATTN_TENSORS: kv_b on its
     * two-format+mode term (fmt==2, or fmt==4 with g_moe_exact off) plus the
     * POSITIVE allowlist (fmt 1/2/3/4) over q_a/q_b/kv_a/o -- see the shared
     * predicate's own comment above attention_rows. The allowlist checks were
     * previously negative fmt!=8 guards -- fail-OPEN for everything they
     * didn't name: for fmt=8 specifically the WP_() macro two lines below
     * picks q8 only for fmt==1, else q4 -- and q4 is NULL/unallocated for
     * fmt=8 (same convention as fmt=1, see the QT struct comment), so an
     * fmt=8 tensor reaching coli_metal_attn_decode would hand the kernel a
     * NULL weight pointer; for fmt=5/6 the hazard is WORSE -- their real
     * weight bytes live in q4, so WP_() hands the shader a valid pointer and
     * mm_gemv's terminal f32 fallback branch silently misreads them as f32
     * garbage. The allowlist fails CLOSED to the CPU path below for all of
     * them (matmul_qt_ex there dispatches every format correctly, incl. fmt=8
     * via matmul_fp8); wiring fmt=8 through bind_gemv/coli_metal_attn_decode
     * stays a deferred follow-up (see this PR's GPU-path note). kv_b's
     * stricter term lives inside the same predicate for an unrelated reason
     * (its absorb kernel is int4-only, grouped served only outside MOE-exact
     * mode); the four allowlist checks are the same discipline extended to
     * the tensors that flow through the shared per-fmt shader. */
    if(g_metal_enabled && !kvs && !g_kv8 && !g_tq && g_absorb!=0 && (S<=4 || g_metal_prefill) && m->kv_start[layer]==0
       && D==6144 && H==64 && c->q_lora==2048 && c->kv_lora==512 && c->qk_nope==192
       && c->qk_rope==64 && vh==256
       && !(metal_fused_layer_fmt_miss(l) & METAL_FUSED_ATTN_TENSORS)){
        int sel_active = m->has_dsa && layer<c->n_layers && c->idx_type[layer] && (pos_base+S) > c->index_topk;
        if(!sel_active){
            if(m->has_dsa && layer<c->n_layers && c->idx_type[layer]){   /* index keys for future selection */
                for(int s=0;s<S;s++){ int pos=pos_base+s; float *kd=m->Ic[layer]+(int64_t)pos*c->index_hd;
                    matmul_qt(kd, x+(int64_t)s*D, &m->ix_wk[layer], 1);
                    layernorm(kd, m->ix_knw[layer], m->ix_knb[layer], c->index_hd, 1e-6f);
                    rope_interleave(kd, pos, c); }
            }
            #define WP_(q) ((q).fmt==1?(const void*)(q).q8:(const void*)(q).q4)
            int ok = coli_metal_attn_decode(x,
                WP_(l->q_a), l->q_a.s, l->q_a.fmt, l->q_a.gs, l->q_a_ln,
                WP_(l->q_b), l->q_b.s, l->q_b.fmt, l->q_b.gs,
                WP_(l->kv_a), l->kv_a.s, l->kv_a.fmt, l->kv_a.gs, l->kv_a_ln,
                WP_(l->kv_b), l->kv_b.s, l->kv_b.fmt, l->kv_b.gs,
                WP_(l->o), l->o.s, l->o.fmt, l->o.gs,
                m->Lc[layer], m->Rc[layer], S, pos_base, m->kv_start[layer], c->eps, c->theta, c->attn_scale, out);
            #undef WP_
            if(ok){ m->t_attn += now_s()-ta0; return; }
        }
    }
#endif
    float *ctx=falloc((int64_t)S*H*vh);
    float *Q=falloc((int64_t)S*H*qh);                  /* query (roped) dei token nuovi */
    int cw=c->kv_lora+c->qk_rope;
    float *QR=falloc((int64_t)S*c->q_lora), *comp=falloc((int64_t)S*cw);
    /* 1) query roped + latente normato e k_rot roped -> in cache.
     * QR tiene il residuo q_a per TUTTE le posizioni: serve anche all'indexer DSA.
     *
     * BATCH-ROWS: le tre proiezioni girano su tutte le S righe in un colpo solo, come gia' fa
     * o_proj (matmul_qt(...,S) sotto) e come fa moe() con la batch-union. Una riga per volta
     * il peso veniva ri-letto per OGNI token; a S righe si legge una volta sola.
     * matmul_qt_ex(...,0): restano sul kernel int4 ESATTO. Con l'IDOT (che il gate S>=g_i4s
     * abiliterebbe da solo appena S>1) il prefill sarebbe molto piu' veloce ma la qualita'
     * cala: -5040.33 -> -5158.68 di log-lik su 1023 token (~+12% perplexity). Il batch da
     * solo e' bit-identical all'originale; il kernel no. Vedi issue.
     * EN: batch the three projections over all S rows, like o_proj below and moe()'s
     * batch-union. matmul_qt_ex(...,0) keeps them on the EXACT int4 kernel: letting S>1 pull
     * them into IDOT is much faster but costs ~12% perplexity (measured). Batching alone is
     * bit-identical to upstream; the kernel switch is not. */
    int pipe_done=0,prefix_done=0;
#ifdef COLI_CUDA
    if(g_cuda_pipe&&!kvs&&S>=8&&layer<c->n_layers&&g_cuda_enabled&&c->kv_lora<=512&&
       !(m->has_dsa&&pos_base+S>c->index_topk)&&
       l->q_a.cuda_eligible&&l->q_b.cuda_eligible&&l->kv_a.cuda_eligible&&
       l->kv_b.cuda_eligible&&l->o.cuda_eligible&&
       qt_cuda_upload(&l->q_a)&&qt_cuda_upload(&l->q_b)&&qt_cuda_upload(&l->kv_a)&&
       qt_cuda_upload(&l->kv_b)&&qt_cuda_upload(&l->o))
        pipe_done=attn_pipe_prefill(m,l,layer,x,0,S,pos_base,out,NULL);
    if(!pipe_done&&getenv("COLI_CUDA_ATTN_PREFIX")&&atoi(getenv("COLI_CUDA_ATTN_PREFIX"))&&
       !kvs&&S<=4&&g_cuda_enabled&&
       (!m->has_dsa||pos_base+S<=c->index_topk)&&
       l->q_a.cuda_eligible&&l->q_b.cuda_eligible&&l->kv_a.cuda_eligible&&
       qt_cuda_upload(&l->q_a)&&qt_cuda_upload(&l->q_b)&&qt_cuda_upload(&l->kv_a))
        prefix_done=attn_pipe_prefix(m,l,x,S,QR,Q,comp);
#endif
    if(!pipe_done&&!prefix_done){
        int vk_qp=0; (void)vk_qp;
#ifdef COLI_VULKAN
        /* Single-submit q-prep chain: [q_a+kv_a] -> rmsnorm(q latent, on-GPU) -> q_b.
         * One fence where the split path pays three (the CPU norm forced the extra
         * roundtrips). Q and comp are written directly; QR is skipped entirely. */
        static int g_vk_qprep=-1;
        if(g_vk_qprep<0) g_vk_qprep=getenv("COLI_VK_QPREP")?atoi(getenv("COLI_VK_QPREP")):1;
        float *dbgQ=NULL,*dbgC=NULL;
        if(g_vk_qprep==2){ dbgQ=falloc((int64_t)S*l->q_b.O); dbgC=falloc((int64_t)S*l->kv_a.O); }
        if(g_vk_qprep && g_vk_dense && l->q_a.fmt==l->kv_a.fmt && l->q_a.fmt==l->q_b.fmt && VK_FMT_OK(&l->q_a)
           && l->q_a.gs==l->kv_a.gs && l->q_a.gs==l->q_b.gs)
            vk_qp=coli_vk_attn_qprep(layer,
                &l->q_a.vk, l->q_a.fmt==1?(const void*)l->q_a.q8:(const void*)l->q_a.q4, l->q_a.s, l->q_a.O,
                &l->kv_a.vk, l->kv_a.fmt==1?(const void*)l->kv_a.q8:(const void*)l->kv_a.q4, l->kv_a.s, l->kv_a.O,
                &l->q_b.vk, l->q_b.fmt==1?(const void*)l->q_b.q8:(const void*)l->q_b.q4, l->q_b.s, l->q_b.O,
                l->q_a.fmt, l->q_a.gs, l->q_a_ln, c->eps, x, S, l->q_a.I,
                g_vk_qprep==2?dbgQ:Q, g_vk_qprep==2?dbgC:comp,
                g_vk_qprep==2?NULL:QR);   /* the DSA indexer reads the NORMED latent from QR */
        int qp_ran=vk_qp;
        if(g_vk_qprep==2 && vk_qp) vk_qp=0;   /* verify mode: chain ran, split path is authoritative */
        if(!vk_qp){
#endif
        int vk_pair=0; (void)vk_pair;
#ifdef COLI_VULKAN
        /* q_a and kv_a read the SAME x: one fused submit for the pair (one x upload,
         * one fence) instead of two full roundtrips. Fallback = the single calls. */
        vk_pair=vk_matmul_pair_qt(&l->q_a, QR, &l->kv_a, comp, x, S);
        if(!vk_pair)
        if(!vk_matmul_qt(&l->q_a, QR, x, S))
#endif
        matmul_qt_ex(QR, x, &l->q_a, S, 0);
        for(int s=0;s<S;s++){ float *qr=QR+(int64_t)s*c->q_lora;
            rmsnorm(qr, qr, l->q_a_ln, c->q_lora, c->eps); }         /* q_b legge il residuo NORMATO */
#ifdef COLI_VULKAN
        if(!vk_matmul_qt(&l->q_b, Q, QR, S))
#endif
        matmul_qt_ex(Q, QR, &l->q_b, S, 0);
#ifdef COLI_VULKAN
        if(!vk_pair)
        if(!vk_matmul_qt(&l->kv_a, comp, x, S))
#endif
        matmul_qt_ex(comp, x, &l->kv_a, S, 0);
#ifdef COLI_VULKAN
        }
        if(dbgQ){ float mq=0,mc=0;
            for(int64_t i2=0;i2<(int64_t)S*l->q_b.O;i2++){ float d=fabsf(dbgQ[i2]-Q[i2])/(fabsf(Q[i2])+1e-3f); if(d>mq) mq=d; }
            for(int64_t i2=0;i2<(int64_t)S*l->kv_a.O;i2++){ float d=fabsf(dbgC[i2]-comp[i2])/(fabsf(comp[i2])+1e-3f); if(d>mc) mc=d; }
            fprintf(stderr,"[QPREP-VERIFY] layer %d S=%d ran=%d maxrel q %.4g kv %.4g\n",layer,S,qp_ran,mq,mc);
            free(dbgQ); free(dbgC); }
#endif
    }
    /* Every iteration writes a DISTINCT KV row (its own `pos`) and rope_interleave's
     * cache is _Thread_local, so the positions are independent -> parallelizing is
     * byte-identical regardless of order. Only the CUDA/VULKAN shadow-shrink writes
     * touch shared state (m->kv_dev_valid/vk_kv_valid), so the pragma is gated to
     * CPU-only builds where those blocks compile out; on GPU builds this stays serial.
     * (dev's structure supersedes the PR's pre-loop shrink hoist: same goal, and the
     * KV8/KV_TQ producers below ride the same parallel loop — ~576 libm encodes per
     * row per layer under KV8 want the OMP pool awake during prefill.) */
    if(!pipe_done){
#if !defined(COLI_CUDA) && !defined(COLI_VULKAN)
    #pragma omp parallel for schedule(static) if(S > 1)
#endif
    for(int s=0;s<S;s++){
        KVState *ks=kvs?kvs[s]:m->kv;
        int pos=positions?positions[s]:pos_base+s;
        float *qfull=Q+(int64_t)s*H*qh;
        for(int h=0;h<H;h++) rope_interleave(qfull+(int64_t)h*qh+c->qk_nope, pos, c);
        const float *cs=comp+(int64_t)s*cw;
#ifdef COLI_CUDA
        if(ks==m->kv&&m->kv_dev_valid&&layer<=c->n_layers&&m->kv_dev_valid[layer]>pos)
            m->kv_dev_valid[layer]=pos;              /* riga riscritta: l'ombra si accorcia */
#endif
#ifdef COLI_VULKAN
        if(ks==m->kv&&m->vk_kv_valid&&layer<=c->n_layers&&m->vk_kv_valid[layer]>pos)
            m->vk_kv_valid[layer]=pos;               /* riga riscritta: la cache VK si accorcia */
#endif
        if(g_tq){
            /* KV_TQ: stessa norma+rope del produttore, poi PolarQuant. Il raggio
             * (norma L2) va nella scala per-riga (Lsc/Rsc); solo gli angoli nei byte. */
            float *Ls=comp+(int64_t)s*cw, *Rs=Ls+c->kv_lora;
            rmsnorm(Ls, Ls, l->kv_a_ln, c->kv_lora, c->eps);
            rope_interleave(Rs, pos, c);
            ks->Lsc[layer][pos]=coli_kvq_quant_row(Ls, coli_kv_row8(ks->Lc8[layer],pos,coli_kvq_row_bytes(c->kv_lora,g_tq_bits,g_tq_codec)), c->kv_lora, g_tq_bits, g_tq_codec);
            ks->Rsc[layer][pos]=coli_kvq_quant_row(Rs, coli_kv_row8(ks->Rc8[layer],pos,coli_kvq_row_bytes(c->qk_rope,g_tq_bits,g_tq_codec)), c->qk_rope, g_tq_bits, g_tq_codec);
        } else if(g_kv8){
            /* KV8: norma+rope sul residuo di comp (scratch, mai riletto), poi
             * quantizza riga+scala. E' IL produttore caldo: ogni token, ogni layer. */
            float *Ls=comp+(int64_t)s*cw, *Rs=Ls+c->kv_lora;
            rmsnorm(Ls, Ls, l->kv_a_ln, c->kv_lora, c->eps);      /* latente normato */
            rope_interleave(Rs, pos, c);                          /* k_rot roped */
            coli_kv8_quant_row_gs(Ls, coli_kv_row8(ks->Lc8[layer],pos,c->kv_lora),
                                  ks->Lsc[layer]+(int64_t)pos*coli_kv8_nscale(c->kv_lora,g_kv8_gs),
                                  c->kv_lora, g_kv8_gs);
            ks->Rsc[layer][pos]=coli_kv8_quant_row(Rs, coli_kv_row8(ks->Rc8[layer],pos,c->qk_rope), c->qk_rope);
        } else {
            float *Ldst=coli_kv_row(ks->Lc[layer],pos,c->kv_lora);
            float *Rdst=coli_kv_row(ks->Rc[layer],pos,c->qk_rope);
            memcpy(Ldst, cs, c->kv_lora*sizeof(float));
            rmsnorm(Ldst, Ldst, l->kv_a_ln, c->kv_lora, c->eps);  /* latente normato */
            memcpy(Rdst, cs+c->kv_lora, c->qk_rope*sizeof(float));
            rope_interleave(Rdst, pos, c);                        /* k_rot roped, condiviso fra teste */
        }
    }
    }
    /* ---- DSA lightning indexer ----
     * Layer FULL: k_idx dei token nuovi in cache + selezione top-k per query (riusata
     * dai layer SHARED successivi). Selezione attiva solo con contesto > index_topk
     * (o DSA_FORCE=1 per il test: selezionare TUTTO deve dare l'output denso esatto). */
    const int *dsel=NULL, *dnsel=NULL; int dtopk=0;
    if(m->has_dsa && layer<c->n_layers && ((!kvs && m->kv_start[layer]==0) || kvs)){
        int nh=c->index_nh, hd=c->index_hd; dtopk=c->index_topk;
        if(c->idx_type[layer]){
            /* BATCH-ROWS, come le proiezioni di attenzione sopra: ix_wk (D x index_hd) veniva
             * ri-letto per OGNI token. matmul_qt_ex(...,0) lo tiene sul kernel int4 ESATTO:
             * il batch da solo supererebbe il gate S>=g_i4s e cambierebbe la quantizzazione
             * delle attivazioni. Cosi' l'output resta bit-identical.
             * EN: batch ix_wk over all S rows like the attention projections; allow_idot=0
             * keeps it on the exact int4 kernel so the result stays bit-identical. */
            float *KD=falloc((int64_t)S*hd);
            matmul_qt_ex(KD, x, &m->ix_wk[layer], S, 0);
            for(int s=0;s<S;s++){
                KVState *ks=kvs?kvs[s]:m->kv;
                int pos=positions?positions[s]:pos_base+s;
                float *kd=coli_kv_row(ks->Ic[layer],pos,hd);
                memcpy(kd, KD+(int64_t)s*hd, (size_t)hd*sizeof(float));
                layernorm(kd, m->ix_knw[layer], m->ix_knb[layer], hd, 1e-6f);
                rope_interleave(kd, pos, c);                 /* primi qk_rope dim, interleaved */
            }
            free(KD);
            if((int64_t)S*dtopk > m->dsa_scap){
                free(m->dsa_sel); free(m->dsa_nsel);
                m->dsa_scap=(int64_t)S*dtopk;
                m->dsa_sel=malloc((size_t)m->dsa_scap*sizeof(int));
                m->dsa_nsel=malloc((size_t)S*sizeof(int));
            }
            /* if(S>1): at decode S is 1, so this region has exactly ONE iteration --
             * and OpenMP still forks and joins the whole team to run it. That is
             * pure overhead on the hottest path there is: once per layer, per
             * token, whether or not the body does any work (both early `continue`
             * branches below are taken on short contexts, and the fork happens
             * anyway). The clause makes the single-iteration case run inline;
             * S>1 prefill is untouched, and the results are identical either way. */
            #pragma omp parallel for schedule(dynamic,1) if(S > 1)
            for(int s=0;s<S;s++){
                KVState *ks=kvs?kvs[s]:m->kv;
                int pos=positions?positions[s]:pos_base+s, nk=pos+1;
                if(ks->kv_start[layer]!=0){ m->dsa_nsel[s]=0; continue; }
                if(nk<=dtopk && !g_dsa_force){ m->dsa_nsel[s]=0; continue; }
                int keep = nk<dtopk ? nk : dtopk;
                float *qi=falloc((int64_t)nh*hd);
                matmul_qt(qi, QR+(int64_t)s*c->q_lora, &m->ix_wq[layer], 1);
                for(int h=0;h<nh;h++) rope_interleave(qi+(int64_t)h*hd, pos, c);
                float *w32=falloc(nh);
                matmul_qt(w32, x+(int64_t)s*D, &m->ix_wp[layer], 1);
                float wsc=1.f/sqrtf((float)nh), rs=1.f/sqrtf((float)hd);
                float *isc=falloc(nk);
                for(int t=0;t<nk;t++){
                    const float *kt=coli_kv_row(ks->Ic[layer],t,hd);
                    float a=0;
                    for(int h=0;h<nh;h++){ const float *qhp=qi+(int64_t)h*hd;
                        float d0=0; for(int i=0;i<hd;i++) d0+=qhp[i]*kt[i];
                        d0*=rs; if(d0>0) a+=w32[h]*d0;       /* ReLU sullo score, poi peso */
                    }
                    isc[t]=a*wsc;
                }
                /* top-keep: threshold via PARTIAL SELECT (#356), poi scan in ordine di posizione.
                 * Era un qsort completo su nk (O(nk log nk)); quickselect estrae solo il
                 * keep-esimo valore piu' grande in O(nk) medio. La soglia (= min del blocco
                 * dei keep maggiori) e' identica a tmp[keep-1] del vecchio qsort, quindi i
                 * due scan qui sotto costruiscono dst[] bit-identical. */
                float *tmp=falloc(nk); memcpy(tmp,isc,nk*sizeof(float));
                partial_select_desc(tmp,nk,keep);
                float thr=tmp[0]; for(int t=1;t<keep;t++) if(tmp[t]<thr) thr=tmp[t];
                int *dst=m->dsa_sel+(int64_t)s*dtopk, nd=0;
                for(int t=0;t<nk && nd<keep;t++) if(isc[t]>thr) dst[nd++]=t;
                for(int t=0;t<nk && nd<keep;t++) if(isc[t]==thr) dst[nd++]=t;
                m->dsa_nsel[s]=nd;
                free(qi); free(w32); free(isc); free(tmp);
            }
        }
        if(m->dsa_nsel){ dsel=m->dsa_sel; dnsel=m->dsa_nsel; }
    }
    /* WEIGHT ABSORPTION (DeepSeek): per S piccoli (decode/verifica MTP) NON si ricostruisce
     * k/v per ogni token del contesto. Per linearita':
     *   q·k_nope_t = (W_K^hT q_nope)·L_t      ctx^h = W_V^h (Σ_t a_t L_t)
     * costo per step ~O(T·kv_lora) invece di O(T·H·(nope+vh)) del matmul kvb_all. */
    if(pipe_done){
        free(ctx); free(Q); free(QR); free(comp);
        m->t_attn += now_s()-ta0;
        return;
    }
    int cuda_absorb=0;
#ifdef COLI_CUDA
    /* QUANT GUARD (!g_kv8&&!g_tq): the CUDA absorb kernels read f32 Lc/Rc rows, which are
     * not allocated under KV8/KV_TQ. Quantized KV takes the CPU consumer below (which decodes
     * the byte cache natively); the fp8/int4 CUDA kernels are the follow-up PR. */
    cuda_absorb=layer<c->n_layers&&!kvs&&g_cuda_enabled&&getenv("COLI_CUDA_ATTN")&&
                atoi(getenv("COLI_CUDA_ATTN"))&&c->kv_lora<=512&&!g_kv8&&!g_tq;
    /* Non-silent: say so ONCE so a CUDA operator isn't puzzled by CPU-bound attention. */
    if((g_kv8||g_tq) && g_cuda_enabled && getenv("COLI_CUDA_ATTN") && atoi(getenv("COLI_CUDA_ATTN"))){
        static int kvq_cuda_noted=0;
        if(!kvq_cuda_noted){ kvq_cuda_noted=1;
            fprintf(stderr,"[%s] no CUDA attention kernels for quantized KV yet: "
                "attention uses the CPU consumer (native byte-cache decode). Follow-up PR.\n",
                g_tq?"KV_TQ":"KV8"); }
    }
#endif
    int absorb = kvs || g_absorb==1 || (g_absorb<0 && S<=4) || cuda_absorb;
    if(absorb && c->kv_lora<=512){
        m->t_aproj+=now_s()-ta0; double tac=now_s();
        int kvl=c->kv_lora, r0v=c->qk_nope;      /* offset righe V dentro il blocco di testa */
        /* Punteggi per-thread sul HEAP. Il cap DEVE essere il massimo nt effettivo del
         * batch, non Tk+1: Tk=pos_base+S vale solo quando pos==pos_base+s. Il percorso
         * batched (step_decode_batch da run_serve_mux) passa positions[] e kv_start
         * per-slot, quindi nt=pos+1-st0 puo' superare Tk+1 -> heap-buffer-overflow su
         * sc[jj]. Si conta esattamente come il loop sotto. */
        int64_t sc_cap = 1;
        for(int s=0;s<S;s++){
            KVState *ks=kvs?kvs[s]:m->kv;
            int pos=positions?positions[s]:pos_base+s;
            int st0=ks->kv_start[layer];
            int ns=(dnsel && dnsel[s]>0)?dnsel[s]:0;      /* DSA: top-k, altrimenti range pieno */
            int64_t nt = ns ? (int64_t)ns : (int64_t)pos+1-st0;
            if(nt>sc_cap) sc_cap=nt;
        }
        float *sc_all = falloc((int64_t)omp_get_max_threads()*sc_cap);
        int cuda_core=0,cuda_projected=0;
#ifdef COLI_CUDA
        /* Ragged batched attention (multi-slot serve) is f32-only: under KV8/KV_TQ the
         * f32 rows this branch reads (kvs[s]->Lc/Rc) are not even allocated; quantized
         * ragged rows take the CPU ragged path instead, which decodes the byte cache
         * natively. A quantized ragged gather is follow-up-PR work. */
        if(kvs&&!g_kv8&&!g_tq&&g_cuda_enabled&&getenv("COLI_CUDA_ATTN")&&atoi(getenv("COLI_CUDA_ATTN"))&&
           !dnsel&&l->kv_b.cuda_eligible&&l->o.cuda_eligible&&
           qt_cuda_upload(&l->kv_b)&&qt_cuda_upload(&l->o)){
            const float **rl=malloc((size_t)S*sizeof(*rl)),**rr=malloc((size_t)S*sizeof(*rr));
            const void **rk=malloc((size_t)S*sizeof(*rk));
            int *rn=malloc((size_t)S*sizeof(*rn)); int mt=0;
            if(rk&&rl&&rr&&rn){
                for(int s=0;s<S;s++){
                    int pos=positions[s],st0=kvs[s]->kv_start[layer]; rn[s]=pos+1-st0;
                    rk[s]=kvs[s];
                    rl[s]=coli_kv_row(kvs[s]->Lc[layer],st0,kvl);
                    rr[s]=coli_kv_row(kvs[s]->Rc[layer],st0,c->qk_rope);
                    if(rn[s]>mt)mt=rn[s];
                }
                cuda_core=cuda_projected=coli_cuda_attention_project_ragged(l->kv_b.cuda,l->o.cuda,
                    out,Q,rk,rl,rr,rn,S,H,c->qk_nope,c->qk_rope,vh,kvl,mt,c->attn_scale);
            }
            free(rk);free(rl);free(rr);free(rn);
        } else if(cuda_absorb&&l->n_kv_b_shard>1){
            int n=l->n_kv_b_shard,st0=m->kv_start[layer],nt=pos_base+S-st0,ok=1;
            float *qs=falloc((int64_t)S*H*qh),*cs=falloc((int64_t)S*H*vh);
            for(int d=0;d<n;d++)for(int s=0;s<S;s++)memcpy(
                qs+(int64_t)l->shard_h0[d]*S*qh+(int64_t)s*l->shard_hn[d]*qh,
                Q+((int64_t)s*H+l->shard_h0[d])*qh,(size_t)l->shard_hn[d]*qh*sizeof(float));
            #pragma omp parallel for schedule(static) reduction(&:ok)
            for(int d=0;d<n;d++)ok&=coli_cuda_attention_absorb_batch(l->kv_b_shard[d],
                cs+(int64_t)l->shard_h0[d]*S*vh,qs+(int64_t)l->shard_h0[d]*S*qh,
                coli_kv_row(m->Lc[layer],st0,kvl),coli_kv_row(m->Rc[layer],st0,c->qk_rope),
                S,l->shard_hn[d],c->qk_nope,c->qk_rope,vh,kvl,nt,c->attn_scale);
            if(ok)for(int d=0;d<n;d++)for(int s=0;s<S;s++)memcpy(
                ctx+((int64_t)s*H+l->shard_h0[d])*vh,
                cs+(int64_t)l->shard_h0[d]*S*vh+(int64_t)s*l->shard_hn[d]*vh,
                (size_t)l->shard_hn[d]*vh*sizeof(float));
            free(qs);free(cs);cuda_core=ok;
        } else if(cuda_absorb&&l->kv_b.cuda_eligible&&l->o.cuda_eligible&&
           qt_cuda_upload(&l->kv_b)&&qt_cuda_upload(&l->o)){
            int st0=m->kv_start[layer],nt=pos_base+S-st0;
            cuda_core=cuda_projected=coli_cuda_attention_project_batch(l->kv_b.cuda,l->o.cuda,out,Q,
                coli_kv_row(m->Lc[layer],st0,kvl),coli_kv_row(m->Rc[layer],st0,c->qk_rope),
                S,H,c->qk_nope,c->qk_rope,vh,kvl,nt,c->attn_scale);
        } else if(S<=4&&g_cuda_enabled&&!g_kv8&&!g_tq&&getenv("COLI_CUDA_ATTN")&&atoi(getenv("COLI_CUDA_ATTN"))&&
           l->kv_b.cuda_eligible&&qt_cuda_upload(&l->kv_b)){   /* quant guard: absorb/kvdev read f32 Lc/Rc rows */
            cuda_core=1;
            for(int s=0;s<S&&cuda_core;s++){
                KVState *ks=kvs?kvs[s]:m->kv;int pos=positions?positions[s]:pos_base+s;
                int st0=ks->kv_start[layer],nt=pos+1-st0;
                if(dnsel&&dnsel[s]>0){
                    /* Sparse decode used to force the CPU path here: the CUDA
                     * absorb kernel only scans contiguous rows and cannot take
                     * the DSA gather list. COLI_DSA_GATHER=1: gather the
                     * selected top-k rows into a compact staging pair and hand
                     * the dense kernel that instead. Host KV stays canonical —
                     * only ~index_topk rows cross PCIe per (layer, token), so
                     * the device never needs the full context resident (the
                     * first stone of KV tiering). Same row set as the CPU loop
                     * below; results differ only by kernel-family FP order,
                     * the same class of divergence the dense CUDA absorb
                     * already has vs the CPU path (#510). Verified: with
                     * DSA_FORCE=1 (identity selection) output is byte-identical
                     * to the dense CUDA path. */
                    static int dsag=-1;
                    if(dsag<0) dsag=getenv("COLI_DSA_GATHER")?atoi(getenv("COLI_DSA_GATHER")):0;
                    if(!dsag){cuda_core=0;break;}
                    int ns=dnsel[s]; const int *tl=dsel+(int64_t)s*dtopk;
                    float *gl=falloc((int64_t)ns*kvl), *gr=falloc((int64_t)ns*c->qk_rope);
                    for(int jj=0;jj<ns;jj++){
                        memcpy(gl+(int64_t)jj*kvl, coli_kv_row(ks->Lc[layer],tl[jj],kvl), (size_t)kvl*sizeof(float));
                        memcpy(gr+(int64_t)jj*c->qk_rope, coli_kv_row(ks->Rc[layer],tl[jj],c->qk_rope), (size_t)c->qk_rope*sizeof(float));
                    }
                    cuda_core=coli_cuda_attention_absorb(l->kv_b.cuda,ctx+(int64_t)s*H*vh,
                        Q+(int64_t)s*H*qh,gl,gr,H,c->qk_nope,c->qk_rope,vh,kvl,ns,c->attn_scale);
                    free(gl);free(gr);
                    if(!cuda_core)break;    /* CPU fallback recomputes every row: idempotent */
                    continue;
                }
                cuda_core=0;
                if(g_cuda_pipe&&ks==m->kv&&layer<c->n_layers&&kv_dev_sync(m,l,layer,pos+1))
                    cuda_core=coli_cuda_attention_absorb_kvdev(l->kv_b.cuda,ctx+(int64_t)s*H*vh,
                        Q+(int64_t)s*H*qh,m->kv_dev_L[layer]+(size_t)st0*kvl,
                        m->kv_dev_R[layer]+(size_t)st0*c->qk_rope,H,c->qk_nope,c->qk_rope,
                        vh,kvl,nt,c->attn_scale);
                if(!cuda_core)
                    cuda_core=coli_cuda_attention_absorb(l->kv_b.cuda,ctx+(int64_t)s*H*vh,
                        Q+(int64_t)s*H*qh,coli_kv_row(ks->Lc[layer],st0,kvl),
                        coli_kv_row(ks->Rc[layer],st0,c->qk_rope),H,c->qk_nope,c->qk_rope,
                        vh,kvl,nt,c->attn_scale);
            }
        }
#endif
        int vk_core=0, vk_projected=0;
#ifdef COLI_VULKAN
        /* Vulkan MLA absorb core (COLI_VK_ATTN=1): scores/softmax/latent/value rows for all
         * S x H in ONE submit per layer, reading the persistent on-device KV mirror (rows
         * appended incrementally; vk_kv_valid = watermark, invalidated like the CUDA shadow
         * on rewrite/rebind/resize). Falls back to CPU on DSA top-k selection, ragged KV,
         * the MTP layer, or any backend failure — output identical either way. */
        if(!cuda_core&&g_vk_attn&&!kvs&&!positions&&S<=4&&layer<c->n_layers&&
           VK_FMT_OK(&l->kv_b)&&kvl<=512&&c->qk_nope<=256&&c->qk_rope<=64&&
           m->vk_kv_valid&&m->Lc[layer]&&m->Rc[layer]){   /* f32 KV only: a quantized-KV cache
                                                * (upstream #399 KV8/TQ leaves Lc/Rc NULL)
                                                * falls back to the CPU path until the VK
                                                * mirror learns fp8 */
            int dsa_on=0; if(dnsel) for(int s=0;s<S;s++) if(dnsel[s]>0) dsa_on=1;
            int st0=m->kv_start[layer], T=pos_base+S;
            if(!dsa_on&&T<=m->max_t&&coli_vk_kv_ensure(layer,m->max_t,kvl,c->qk_rope)){
                int ok=1;
                for(int t=m->vk_kv_valid[layer];t<T&&ok;t++)
                    ok=coli_vk_kv_row(layer,t,coli_kv_row(m->Lc[layer],t,kvl),
                                      coli_kv_row(m->Rc[layer],t,c->qk_rope));
                if(ok){
                    m->vk_kv_valid[layer]=T;
                    const void *kw=l->kv_b.fmt==1?(const void*)l->kv_b.q8:(const void*)l->kv_b.q4;
                    /* fused absorb + o-projection: ctx never leaves the device */
                    if(VK_FMT_OK(&l->o)&&
                       coli_vk_attention_absorb_project(&l->kv_b.vk,kw,l->kv_b.s,l->kv_b.fmt,l->kv_b.gs,
                            &l->o.vk,l->o.fmt==1?(const void*)l->o.q8:(const void*)l->o.q4,
                            l->o.s,l->o.fmt,l->o.gs,out,Q,layer,S,H,c->qk_nope,c->qk_rope,
                            vh,kvl,st0,T,c->attn_scale,D))
                        vk_core=vk_projected=1;
                    if(!vk_core)
                        vk_core=coli_vk_attention_absorb(&l->kv_b.vk,kw,
                            l->kv_b.s,l->kv_b.fmt,l->kv_b.gs,ctx,Q,layer,S,H,c->qk_nope,c->qk_rope,
                            vh,kvl,st0,T,c->attn_scale);
                }
            }
        }
#endif
        if(!cuda_core&&!vk_core){
        /* Causal rows grow with s; round-robin pairs keep that work balanced. */
        #pragma omp parallel for collapse(2) schedule(static,1)
        for(int s=0;s<S;s++) for(int h=0;h<H;h++){
            KVState *ks=kvs?kvs[s]:m->kv;
            int pos=positions?positions[s]:pos_base+s;
            const float *qp=Q+(int64_t)s*H*qh+(int64_t)h*qh;
            const float *qr=qp+c->qk_nope;
            int rbase=h*(c->qk_nope+vh);
            float qabs[512]; memset(qabs,0,kvl*sizeof(float));
            for(int d=0;d<c->qk_nope;d++) qt_addrow(&l->kv_b, rbase+d, qp[d], qabs);
            float *sc = sc_all + (int64_t)omp_get_thread_num()*sc_cap;
            int st0=ks->kv_start[layer];
            int ns = (dnsel && dnsel[s]>0) ? dnsel[s] : 0;    /* DSA: lista top-k o range pieno */
            const int *tlist = ns ? dsel+(int64_t)s*dtopk : NULL;
            int nt = ns ? ns : pos+1-st0;
            /* codec-1 rotated-int4: rotate the query ONCE per head (q.x_hat == rotate(q).c, the
             * same orthogonality identity the Metal/CUDA native kernels use) and dot the packed
             * nibbles directly — instead of f32-dequantizing every shared latent row per head
             * (2H-redundant). std = radius/sqrt(n) rides Lsc/Rsc. codec-0 keeps the dequant path. */
            int tq1 = (g_tq && g_tq_codec==1);
            float qtl[512], qtr[512];
            float invsnL = tq1 ? 1.f/sqrtf((float)kvl) : 0.f, invsnR = tq1 ? 1.f/sqrtf((float)c->qk_rope) : 0.f;
            int lrb1=0, rrb1=0;
            if(tq1){ coli_tq_rotate(qabs,qtl,kvl,COLI_TQ_SEED); coli_tq_rotate(qr,qtr,c->qk_rope,COLI_TQ_SEED);
                     lrb1=coli_q4_row_bytes(kvl); rrb1=coli_q4_row_bytes(c->qk_rope); }
            for(int jj=0;jj<nt;jj++){ int t = tlist ? tlist[jj] : st0+jj;
                float a=0;
                if(tq1){
                    const uint8_t *Lt=coli_kv_row8(ks->Lc8[layer],t,lrb1);
                    const uint8_t *Rt=coli_kv_row8(ks->Rc8[layer],t,rrb1);
                    float al=0, ar=0;
                    for(int i=0;i<kvl;i++){ int cc=(Lt[i>>1]>>((i&1)*4))&0xF; al+=qtl[i]*coli_q4_lev[cc]; }
                    for(int d=0;d<c->qk_rope;d++){ int cc=(Rt[d>>1]>>((d&1)*4))&0xF; ar+=qtr[d]*coli_q4_lev[cc]; }
                    a = al*ks->Lsc[layer][t]*invsnL + ar*ks->Rsc[layer][t]*invsnR;
                } else if(g_tq){
                    /* PolarQuant (codec 0): ricostruisci la riga latente + rope a f32, poi il
                     * dot diretto (il raggio e' gia' nel dequant, niente scala esterna). */
                    float Lf[512], Rf[512];
                    coli_kvq_dequant_row(coli_kv_row8(ks->Lc8[layer],t,coli_kvq_row_bytes(kvl,g_tq_bits,g_tq_codec)), ks->Lsc[layer][t], Lf, kvl, g_tq_bits, g_tq_codec);
                    coli_kvq_dequant_row(coli_kv_row8(ks->Rc8[layer],t,coli_kvq_row_bytes(c->qk_rope,g_tq_bits,g_tq_codec)), ks->Rsc[layer][t], Rf, c->qk_rope, g_tq_bits, g_tq_codec);
                    for(int i=0;i<kvl;i++) a+=qabs[i]*Lf[i];
                    for(int d=0;d<c->qk_rope;d++) a+=qr[d]*Rf[d];
                } else if(g_kv8){
                    /* LUT-dequant inline nel dot; la scala per-riga esce dalla
                     * somma: score = Lsc·Σ q·lut[b] + Rsc·Σ qr·lut[b].
                     * KV8_GS: per-group scales — partial dot per group, each
                     * scaled before the sum (same math, tighter grid). */
                    const uint8_t *Lt=coli_kv_row8(ks->Lc8[layer],t,kvl);
                    const uint8_t *kr=coli_kv_row8(ks->Rc8[layer],t,c->qk_rope);
                    int nsl=coli_kv8_nscale(kvl,g_kv8_gs);
                    const float *Ls_t=ks->Lsc[layer]+(int64_t)t*nsl;
                    float ar=0; a=0;
                    if(g_kv8_gs){
                        for(int g=0,k2=0;g<kvl;g+=g_kv8_gs,k2++){
                            int m=kvl-g<g_kv8_gs?kvl-g:g_kv8_gs; float al=0;
                            for(int i=0;i<m;i++) al+=qabs[g+i]*coli_fp8_lut[Lt[g+i]];
                            a+=al*Ls_t[k2];
                        }
                    } else {
                        float al=0;
                        for(int i=0;i<kvl;i++) al+=qabs[i]*coli_fp8_lut[Lt[i]];
                        a=al*Ls_t[0];
                    }
                    for(int d=0;d<c->qk_rope;d++) ar+=qr[d]*coli_fp8_lut[kr[d]];
                    a+=ar*ks->Rsc[layer][t];
                } else {
                const float *Lt=coli_kv_row(ks->Lc[layer],t,kvl);
                const float *kr=coli_kv_row(ks->Rc[layer],t,c->qk_rope);
                /* MLA-absorb score: dot(qabs, Lt) + dot(qr, kr). #442: the qabs·Lt
                 * reduction is the hot f32 dot at this site (kvl=512 on GLM-5.2,
                 * runs nt times per (s,h), grows with context). SIMD-ify under
                 * AVX2 (8-lane fmadd + hsum256, same shape as matmul_q in quant.h)
                 * and NEON, with a scalar tail for the remainder. Reassociation
                 * is accepted here — softmax downstream softens the rounding flip. */
                int i=0;
#if defined(__AVX2__)
                __m256 acc=_mm256_setzero_ps();
                for(;i+8<=kvl;i+=8)
                    acc=_mm256_fmadd_ps(_mm256_loadu_ps(qabs+i), _mm256_loadu_ps(Lt+i), acc);
                a=hsum256(acc);
#elif defined(__ARM_NEON)
                float32x4_t ac0=vdupq_n_f32(0), ac1=vdupq_n_f32(0);
                for(;i+8<=kvl;i+=8){
                    ac0=vfmaq_f32(ac0, vld1q_f32(qabs+i),   vld1q_f32(Lt+i));
                    ac1=vfmaq_f32(ac1, vld1q_f32(qabs+i+4), vld1q_f32(Lt+i+4)); }
                a=vaddvq_f32(vaddq_f32(ac0,ac1));
#endif
                for(;i<kvl;i++) a+=qabs[i]*Lt[i];
                for(int d=0;d<c->qk_rope;d++) a+=qr[d]*kr[d];
                }
                sc[jj]=a*c->attn_scale;
            }
            softmax(sc,nt);
            float clat[512]; memset(clat,0,kvl*sizeof(float));
            for(int jj=0;jj<nt;jj++){ int t = tlist ? tlist[jj] : st0+jj;
                if(tq1){
                    /* accumulate acc = sum_t w_t*std_t*lev[L[t]] in the rotated basis; unrotate
                     * ONCE after the loop (context = unrotate(sum_t w_t c_t)). */
                    const uint8_t *Lt=coli_kv_row8(ks->Lc8[layer],t,lrb1);
                    float a=sc[jj]*ks->Lsc[layer][t]*invsnL;
                    for(int i=0;i<kvl;i++){ int cc=(Lt[i>>1]>>((i&1)*4))&0xF; clat[i]+=a*coli_q4_lev[cc]; }
                } else if(g_tq){
                    float Lf[512];
                    coli_kvq_dequant_row(coli_kv_row8(ks->Lc8[layer],t,coli_kvq_row_bytes(kvl,g_tq_bits,g_tq_codec)), ks->Lsc[layer][t], Lf, kvl, g_tq_bits, g_tq_codec);
                    float a=sc[jj];                          /* raggio gia' nel dequant */
                    for(int i=0;i<kvl;i++) clat[i]+=a*Lf[i];
                } else if(g_kv8){
                    const uint8_t *Lt=coli_kv_row8(ks->Lc8[layer],t,kvl);
                    int nsl=coli_kv8_nscale(kvl,g_kv8_gs);
                    const float *Ls_t=ks->Lsc[layer]+(int64_t)t*nsl;
                    if(g_kv8_gs){
                        for(int g=0,k2=0;g<kvl;g+=g_kv8_gs,k2++){
                            int m=kvl-g<g_kv8_gs?kvl-g:g_kv8_gs;
                            float a=sc[jj]*Ls_t[k2];
                            for(int i=0;i<m;i++) clat[g+i]+=a*coli_fp8_lut[Lt[g+i]];
                        }
                    } else {
                        float a=sc[jj]*Ls_t[0];             /* la scala si fonde nel peso */
                        for(int i=0;i<kvl;i++) clat[i]+=a*coli_fp8_lut[Lt[i]];
                    }
                } else {
                const float *Lt=coli_kv_row(ks->Lc[layer],t,kvl);
                /* MLA-absorb value mix: clat += sc[jj] * Lt (AXPY over kvl).
                 * #442: SIMD-ified — each lane writes back independently so there
                 * is no reassociation here (strictly bit-identical to scalar). */
                float a=sc[jj]; int i=0;
#if defined(__AVX2__)
                __m256 va=_mm256_set1_ps(a);
                for(;i+8<=kvl;i+=8){
                    __m256 cl=_mm256_loadu_ps(clat+i), lt=_mm256_loadu_ps(Lt+i);
                    _mm256_storeu_ps(clat+i, _mm256_fmadd_ps(va, lt, cl));
                }
#elif defined(__ARM_NEON)
                float32x4_t va=vdupq_n_f32(a);
                for(;i+8<=kvl;i+=8){
                    vst1q_f32(clat+i,   vfmaq_f32(vld1q_f32(clat+i),   va, vld1q_f32(Lt+i)));
                    vst1q_f32(clat+i+4, vfmaq_f32(vld1q_f32(clat+i+4), va, vld1q_f32(Lt+i+4)));
                }
#endif
                for(;i<kvl;i++) clat[i]+=a*Lt[i];
                } }
            if(tq1) coli_tq_unrotate(clat,kvl,COLI_TQ_SEED);   /* rotated basis -> latent */
            qt_matvec_rows(&l->kv_b, rbase+r0v, vh, clat, ctx+((int64_t)s*H+h)*vh);
        }
        }
        m->t_acore+=now_s()-tac; double tao=now_s();
        if(!cuda_projected&&!vk_projected){
#ifdef COLI_VULKAN
            if(!vk_matmul_qt(&l->o, out, ctx, S))
#endif
            matmul_qt(out, ctx, &l->o, S);
        } m->t_aout+=now_s()-tao;
        free(ctx); free(Q); free(QR); free(comp); free(sc_all);
        m->t_attn += now_s()-ta0;
        return;
    }
    /* 2) ricostruzione di k_nope+value per TUTTI i token 0..Tk-1 (un solo matmul su kv_b) */
    m->t_aproj+=now_s()-ta0; double tk0=now_s();
    int stL=m->kv_start[layer];
    /* #768 (cause 2): the one-shot kvb_all buffer is Tk*H*(qk_nope+v_head) floats —
     * 30.1 GB at ctx 262144 — and cap_for_ram reserves it PERMANENTLY even though it
     * only exists during prefill; that reservation is what starves the expert cache
     * at long context (5.46 -> 0.86 tok/s). Above KVB_FLASH_MB the reconstruction is
     * tiled instead: rebuild kv_b for KVB_TILE_MB worth of tokens at a time and fold
     * scores/values through an online (flash-style) softmax. Same rebuild total (one
     * matmul pass over the context), same t-order for scores and values; only the
     * softmax normalisation is applied incrementally, so the output can differ from
     * the one-shot path by rounding — the same kernel-family divergence class as the
     * CUDA/Metal attention arms (#510). Peak transient drops from Tk*kvb_dim*4 to the
     * tile plus 2 floats per (row,head). DSA rows (dnsel) keep the one-shot path: the
     * top-keep list is scanned in its two-band order and is not tileable by t-range.
     * KVB_FLASH_MB=0 disables tiling entirely; KVB_FLASH=1 forces it at any size. */
    int dsa_any=0; if(dnsel) for(int s=0;s<S && !dsa_any;s++) if(dnsel[s]>0) dsa_any=1;
    int64_t kvb_rows=(int64_t)Tk-stL, kvb_need=kvb_rows*kvb_dim*4;
    int64_t flash_mb=getenv("KVB_FLASH_MB")?atoll(getenv("KVB_FLASH_MB")):2048;
    int use_flash=!dsa_any && flash_mb>0 && kvb_need>flash_mb*1048576;
    if(getenv("KVB_FLASH")) use_flash=!dsa_any && atoi(getenv("KVB_FLASH"))!=0;
    if(use_flash){
        int64_t tile_mb=getenv("KVB_TILE_MB")?atoll(getenv("KVB_TILE_MB")):512;
        int64_t tile=tile_mb*1048576/((int64_t)kvb_dim*4);
        if(tile<256) tile=256; if(tile>kvb_rows) tile=kvb_rows;
        static int said; if(!said){ said=1;
            fprintf(stderr,"[ATTN] kvb reconstruction tiled (#768): %.1f GB one-shot buffer "
                "-> %.0f MB tiles, online softmax (KVB_FLASH_MB=0 restores one-shot)\n",
                kvb_need/1e9, tile*(double)kvb_dim*4/1048576.0);
        }
        m->t_kvb += now_s()-tk0;
        float *kvb_tile=falloc(tile*kvb_dim);
        /* quantized KV: dequantize each tile's latent rows into an f32 staging
         * buffer for the kv_b matmul — same codecs as the one-shot path, tile-
         * sized instead of context-sized (#1151 follow-up) */
        float *Lf_tile=(g_kv8||g_tq)?falloc(tile*c->kv_lora):NULL;
        float *ml=falloc((int64_t)S*H*2);              /* running (max, sum) per (row, head) */
        double tac=now_s(), kvb_acc=0;
        #pragma omp parallel for collapse(2) schedule(static)
        for(int s=0;s<S;s++) for(int h=0;h<H;h++){
            ml[((int64_t)s*H+h)*2]=-1e30f; ml[((int64_t)s*H+h)*2+1]=0.f;
            float *cx=ctx+((int64_t)s*H+h)*vh; for(int d=0;d<vh;d++) cx[d]=0;
        }
        for(int64_t t0=stL; t0<Tk; t0+=tile){
            int64_t tn=Tk-t0<tile?Tk-t0:tile;
            double tk1=now_s();
            if(Lf_tile){
                kv_lc_rows_f32(m,layer,t0,tn,Lf_tile,c);
                matmul_qt(kvb_tile, Lf_tile, &l->kv_b, (int)tn);
            } else
                matmul_qt(kvb_tile, m->Lc[layer]+t0*c->kv_lora, &l->kv_b, (int)tn);
            kvb_acc+=now_s()-tk1;
            #pragma omp parallel for collapse(2) schedule(static,1)
            for(int s=0;s<S;s++) for(int h=0;h<H;h++){
                int pos=pos_base+s;
                int64_t nt=(int64_t)pos+1-stL;         /* causal length of this row */
                if(t0-stL>=nt) continue;
                int64_t jn=nt-(t0-stL); if(jn>tn) jn=tn;
                const float *qp=Q+(int64_t)s*H*qh+(int64_t)h*qh, *qr=qp+c->qk_nope;
                float *st=ml+((int64_t)s*H+h)*2, *cx=ctx+((int64_t)s*H+h)*vh;
                float mrun=st[0], lrun=st[1];
                for(int64_t jj=0;jj<jn;jj++){
                    const float *kn=kvb_tile+jj*kvb_dim+(int64_t)h*(c->qk_nope+vh);
                    float a=0; for(int d=0;d<c->qk_nope;d++) a+=qp[d]*kn[d];
                    if(g_tq){
                        float Rf[512];
                        coli_kvq_dequant_row(coli_kv_row8(m->Rc8[layer],t0+jj,coli_kvq_row_bytes(c->qk_rope,g_tq_bits,g_tq_codec)),
                                             m->Rsc[layer][t0+jj], Rf, c->qk_rope, g_tq_bits, g_tq_codec);
                        for(int d=0;d<c->qk_rope;d++) a+=qr[d]*Rf[d];
                    } else if(g_kv8){
                        const uint8_t *kr=coli_kv_row8(m->Rc8[layer],t0+jj,c->qk_rope);
                        float ar=0; for(int d=0;d<c->qk_rope;d++) ar+=qr[d]*coli_fp8_lut[kr[d]];
                        a+=ar*m->Rsc[layer][t0+jj];
                    } else {
                        const float *kr=m->Rc[layer]+(t0+jj)*c->qk_rope;
                        for(int d=0;d<c->qk_rope;d++) a+=qr[d]*kr[d];
                    }
                    a*=c->attn_scale;
                    const float *vv=kvb_tile+jj*kvb_dim+(int64_t)h*(c->qk_nope+vh)+c->qk_nope;
                    if(a>mrun){
                        float rs=expf(mrun-a);
                        lrun*=rs; for(int d=0;d<vh;d++) cx[d]*=rs;
                        mrun=a;
                    }
                    float w=expf(a-mrun);
                    lrun+=w; for(int d=0;d<vh;d++) cx[d]+=w*vv[d];
                }
                st[0]=mrun; st[1]=lrun;
            }
        }
        #pragma omp parallel for collapse(2) schedule(static)
        for(int s=0;s<S;s++) for(int h=0;h<H;h++){
            float lrun=ml[((int64_t)s*H+h)*2+1];
            float *cx=ctx+((int64_t)s*H+h)*vh;
            if(lrun>0) for(int d=0;d<vh;d++) cx[d]/=lrun;
        }
        free(kvb_tile); free(Lf_tile); free(ml);
        m->t_kvb+=kvb_acc; m->t_acore+=now_s()-tac-kvb_acc; double tao=now_s();
        matmul_qt(out, ctx, &l->o, S); m->t_aout+=now_s()-tao;
        free(ctx); free(Q); free(QR); free(comp);
        m->t_attn += now_s()-ta0;
        return;
    }
    /* #768 follow-up (DSA): the loops below read only the top-keep rows, yet the
     * one-shot matmul rebuilt ALL Tk rows — the same 30 GB paid for a top-k read.
     * Past the same ceiling, rebuild only the UNION of selected rows: each row's
     * rebuild is independent and the read order (jj along each tlist) is unchanged,
     * so this is bit-identical to the one-shot path. Rows with ns==0 scan their full
     * causal range and force those rows into the union; if the union does not at
     * least halve the buffer, fall back to one-shot (gather overhead without the
     * memory win). kvb_map: row t -> compact index, -1 = not rebuilt (never read). */
    float *kvb_all=NULL; int32_t *kvb_map=NULL;
    if(dsa_any && flash_mb>0 && kvb_need>flash_mb*1048576){
        uint8_t *want=calloc((size_t)kvb_rows,1);
        if(want){
            for(int s=0;s<S;s++){
                int ns=dnsel[s];
                if(ns>0){ const int *tl=dsel+(int64_t)s*dtopk;
                    for(int j=0;j<ns;j++){ int64_t r=(int64_t)tl[j]-stL;
                        if(r>=0&&r<kvb_rows) want[r]=1; } }
                else { int64_t upto=(int64_t)pos_base+s+1-stL;
                    if(upto>kvb_rows) upto=kvb_rows;
                    if(upto>0) memset(want,1,(size_t)upto); }
            }
            int64_t un=0; for(int64_t r=0;r<kvb_rows;r++) if(want[r]) un++;
            if(un>0 && un<kvb_rows/2){
                kvb_map=malloc((size_t)kvb_rows*sizeof(int32_t));
                if(kvb_map){
                    float *Lg=falloc(un*c->kv_lora); int64_t w=0;
                    for(int64_t r=0;r<kvb_rows;r++){
                        if(want[r]){
                            if(g_kv8||g_tq)              /* quantized latent: dequant the row (#1151 follow-up) */
                                kv_lc_rows_f32(m,layer,(int64_t)stL+r,1,Lg+w*c->kv_lora,c);
                            else
                                memcpy(Lg+w*c->kv_lora,
                                    m->Lc[layer]+((int64_t)stL+r)*c->kv_lora,
                                    (size_t)c->kv_lora*sizeof(float));
                            kvb_map[r]=(int32_t)w; w++; }
                        else kvb_map[r]=-1;
                    }
                    kvb_all=falloc(un*kvb_dim);
                    matmul_qt(kvb_all, Lg, &l->kv_b, (int)un);
                    free(Lg);
                    static int said_dsa; if(!said_dsa){ said_dsa=1;
                        fprintf(stderr,"[ATTN] DSA kvb rebuild gathered (#768): %lld of %lld rows "
                            "(%.1f -> %.1f GB)\n",(long long)un,(long long)kvb_rows,
                            kvb_need/1e9, un*(double)kvb_dim*4/1e9); }
                }
            }
            free(want);
        }
    }
    if(!kvb_map){
        kvb_all=falloc((int64_t)Tk*kvb_dim);
        if(g_tq){
            /* PolarQuant: ricostruisci il latente a f32 (kv_b vuole righe float). */
            float *Lf=falloc((int64_t)(Tk-stL)*c->kv_lora);
            int lbb=coli_kvq_row_bytes(c->kv_lora,g_tq_bits,g_tq_codec);
            for(int t=stL;t<Tk;t++)
                coli_kvq_dequant_row(coli_kv_row8(m->Lc8[layer],t,lbb), m->Lsc[layer][t],
                                    Lf+(int64_t)(t-stL)*c->kv_lora, c->kv_lora, g_tq_bits, g_tq_codec);
            matmul_qt(kvb_all+(int64_t)stL*kvb_dim, Lf, &l->kv_b, Tk-stL);
            free(Lf);
        } else if(g_kv8){
            /* staging f32 del latente dequantizzato: kv_b vuole righe float. Il buffer
             * [Tk-stL,kvl] e' rumore rispetto a kvb_all [Tk,H*(nope+vh)] gia' allocato. */
            float *Lf=falloc((int64_t)(Tk-stL)*c->kv_lora);
            { int nsl=coli_kv8_nscale(c->kv_lora,g_kv8_gs);
              for(int t=stL;t<Tk;t++)
                coli_kv8_dequant_row_gs(coli_kv_row8(m->Lc8[layer],t,c->kv_lora),
                                        m->Lsc[layer]+(int64_t)t*nsl,
                                        Lf+(int64_t)(t-stL)*c->kv_lora, c->kv_lora, g_kv8_gs); }
            matmul_qt(kvb_all+(int64_t)stL*kvb_dim, Lf, &l->kv_b, Tk-stL);
            free(Lf);
        } else
            matmul_qt(kvb_all+(int64_t)stL*kvb_dim, m->Lc[layer]+(int64_t)stL*c->kv_lora, &l->kv_b, Tk-stL);
    }
    m->t_kvb += now_s()-tk0;
    /* 3) attenzione causale: score = q_pass·k_nope + q_rot·k_rot
     * (punteggi sul heap, per-thread: vedi il commento nel ramo absorb) */
    int64_t sc_cap = Tk - stL;
    float *sc_all = falloc((int64_t)omp_get_max_threads()*sc_cap);
    double tac=now_s();
    /* Causal rows grow with s; round-robin pairs keep that work balanced. */
    #pragma omp parallel for collapse(2) schedule(static,1)
    for(int s=0;s<S;s++) for(int h=0;h<H;h++){
        int pos=pos_base+s;
        const float *qp=Q+(int64_t)s*H*qh+(int64_t)h*qh;          /* [qk_nope | qk_rope] */
        const float *qr=qp+c->qk_nope;
        float *sc = sc_all + (int64_t)omp_get_thread_num()*sc_cap;
        int st0=m->kv_start[layer];
        int ns = (dnsel && dnsel[s]>0) ? dnsel[s] : 0;        /* DSA: lista top-k o range pieno */
        const int *tlist = ns ? dsel+(int64_t)s*dtopk : NULL;
        int nt = ns ? ns : pos+1-st0;
        for(int jj=0;jj<nt;jj++){ int t = tlist ? tlist[jj] : st0+jj;
            int64_t ti = kvb_map ? (int64_t)kvb_map[t-stL] : (int64_t)t;   /* gathered vs absolute row */
            const float *kn=kvb_all+ti*kvb_dim+(int64_t)h*(c->qk_nope+vh);
            float a=0; for(int d=0;d<c->qk_nope;d++) a+=qp[d]*kn[d];
            if(g_tq){
                float Rf[512];
                coli_kvq_dequant_row(coli_kv_row8(m->Rc8[layer],t,coli_kvq_row_bytes(c->qk_rope,g_tq_bits,g_tq_codec)), m->Rsc[layer][t], Rf, c->qk_rope, g_tq_bits, g_tq_codec);
                for(int d=0;d<c->qk_rope;d++) a+=qr[d]*Rf[d];
            } else if(g_kv8){
                const uint8_t *kr=coli_kv_row8(m->Rc8[layer],t,c->qk_rope);
                float ar=0; for(int d=0;d<c->qk_rope;d++) ar+=qr[d]*coli_fp8_lut[kr[d]];
                a+=ar*m->Rsc[layer][t];
            } else {
                const float *kr=m->Rc[layer]+(int64_t)t*c->qk_rope;
                for(int d=0;d<c->qk_rope;d++) a+=qr[d]*kr[d];
            }
            sc[jj]=a*c->attn_scale;
        }
        softmax(sc,nt);
        float *cx=ctx+((int64_t)s*H+h)*vh; for(int d=0;d<vh;d++) cx[d]=0;
        for(int jj=0;jj<nt;jj++){ int t = tlist ? tlist[jj] : st0+jj;
            int64_t ti = kvb_map ? (int64_t)kvb_map[t-stL] : (int64_t)t;
            const float *vv=kvb_all+ti*kvb_dim+(int64_t)h*(c->qk_nope+vh)+c->qk_nope;
            float a=sc[jj]; for(int d=0;d<vh;d++) cx[d]+=a*vv[d]; }
    }
    m->t_acore+=now_s()-tac; double tao=now_s();
    matmul_qt(out, ctx, &l->o, S); m->t_aout+=now_s()-tao;
    free(ctx); free(Q); free(QR); free(comp); free(kvb_all); free(kvb_map); free(sc_all);
    m->t_attn += now_s()-ta0;
}

static void attention(Model *m, Layer *l, int layer, float *x, int S, int pos_base, float *out){
    attention_rows(m,l,layer,x,S,pos_base,NULL,NULL,out);
}

/* MoE GLM su x[S,hidden] -> out (router sigmoid/noaux_tc, n_group=1, + shared expert).
 * BATCH-UNION: per S>1 (prefill, verifica MTP) ogni expert UNICO del batch viene caricato
 * una volta sola e moltiplicato per tutte le posizioni che lo usano (pesi letti 1 volta);
 * lo shared expert e' un unico matmul a S righe. Per posizione l'accumulo resta
 * nell'ordine (routed nel loro ordine di union, poi shared). */
#ifdef COLI_VULKAN
/* dev2 group issue on a worker thread: the Polaris/x4 submit path costs ~0.8ms per
 * dev2-active block serialized on the main thread (measured 3.2s/decode vs 0.15s
 * dev0-only) while the 580's exec itself finishes early (take-wait 0.1s) — off-thread
 * it overlaps dev0's issue + the CPU expert share. All Vulkan state it touches is
 * G2-private, and moe() joins before take2, preserving the one-in-flight invariant. */
typedef struct { ColiVkTensor **g,**u,**d; const int *rows; int n; const float *x; int rc; double dt; } Vk2Iss;
static void *vk2_issue_worker(void *p){
    Vk2Iss *j=(Vk2Iss*)p;
    double t0=now_s();
    j->rc = coli_vk_expert_group_issue2(j->g,j->u,j->d,j->rows,j->n,j->x);
    j->dt = now_s()-t0;
    return NULL;
}
#endif

#ifdef COLI_CACHE_INDEX_TEST
static uint64_t g_glm_slot_index_probes;
#endif

/* The hot store is immutable during a forward pass; REPIN updates this map on
 * the main thread between passes.  Validate the slot identity so corrupt or
 * stale bookkeeping degrades to a miss rather than serving wrong weights. */
static ESlot *pin_indexed(Model *m,int layer,int eid){
    if(layer<0||layer>m->c.n_layers||eid<0||eid>=m->c.n_experts||
       !m->pin_slot_by_expert||!m->pin_slot_by_expert[layer]) return NULL;
#ifdef COLI_CACHE_INDEX_TEST
    g_glm_slot_index_probes++;
#endif
    int i=m->pin_slot_by_expert[layer][eid];
    if(i<0||i>=m->npin[layer]||!m->pin[layer]||m->pin[layer][i].eid!=eid) return NULL;
    return &m->pin[layer][i];
}

static void pin_unindex(Model *m,int layer,ESlot *s){
    if(!m->pin_slot_by_expert||!m->pin_slot_by_expert[layer]||!m->pin[layer]) return;
    int eid=s->eid, i=(int)(s-m->pin[layer]);
    if(eid>=0&&eid<m->c.n_experts&&m->pin_slot_by_expert[layer][eid]==i)
        m->pin_slot_by_expert[layer][eid]=-1;
}

static void pin_index(Model *m,int layer,ESlot *s){
    if(!m->pin_slot_by_expert||!m->pin_slot_by_expert[layer]||!m->pin[layer]) return;
    int eid=s->eid;
    if(eid>=0&&eid<m->c.n_experts)
        m->pin_slot_by_expert[layer][eid]=(int)(s-m->pin[layer]);
}

/* include_reserved is used only by PILOT de-duplication.  Runtime accesses to
 * a future layer hold g_pilot_mx; the current MoE layer is frozen by the
 * existing g_cur_moe_layer barrier before lock-free decode lookups begin. */
static ESlot *ecache_indexed(Model *m,int layer,int eid,int include_reserved){
    if(layer<0||layer>m->c.n_layers||eid<0||eid>=m->c.n_experts||
       !m->ecache_slot_by_expert||!m->ecache_slot_by_expert[layer]) return NULL;
#ifdef COLI_CACHE_INDEX_TEST
    g_glm_slot_index_probes++;
#endif
    int i=m->ecache_slot_by_expert[layer][eid];
    if(i<0||i>=m->ecn[layer]||!m->ecache[layer]) return NULL;
    int actual=m->ecache[layer][i].eid;
    if(actual!=eid && (!include_reserved||actual!=-(eid+2))) return NULL;
    return &m->ecache[layer][i];
}

static int ecache_key(const ESlot *s){
    if(s->eid>=0) return s->eid;
    if(s->eid<=-2) return -(s->eid+2);             /* PILOT reservation -(eid+2) */
    return -1;
}

static void ecache_unindex(Model *m,int layer,ESlot *s){
    int eid=ecache_key(s), i=(int)(s-m->ecache[layer]);
    if(eid>=0&&eid<m->c.n_experts&&m->ecache_slot_by_expert&&
       m->ecache_slot_by_expert[layer]&&m->ecache_slot_by_expert[layer][eid]==i)
        m->ecache_slot_by_expert[layer][eid]=-1;
}

static void ecache_publish(Model *m,int layer,ESlot *s,int eid){
    ecache_unindex(m,layer,s);
    s->eid=eid;
    if(eid>=0&&eid<m->c.n_experts&&m->ecache_slot_by_expert&&
       m->ecache_slot_by_expert[layer])
        m->ecache_slot_by_expert[layer][eid]=(int)(s-m->ecache[layer]);
}

static void ecache_reserve(Model *m,int layer,ESlot *s,int eid){
    ecache_unindex(m,layer,s);
    s->eid=-(eid+2);
    if(eid>=0&&eid<m->c.n_experts&&m->ecache_slot_by_expert&&
       m->ecache_slot_by_expert[layer])
        m->ecache_slot_by_expert[layer][eid]=(int)(s-m->ecache[layer]);
}

static void ecache_hide(Model *m,int layer,ESlot *s){
    ecache_unindex(m,layer,s);
    s->eid=-1;
}

/* pin ∪ LRU residency probe (used by CACHE_ROUTE max-rank fill). */
static int expert_is_resident(Model *m, int layer, int eid){
    return pin_indexed(m,layer,eid)!=NULL || ecache_indexed(m,layer,eid,0)!=NULL;
}

/* I loop di selezione top-K partono da best=-1 e lo usano come indice appena il
 * giro finisce. Se OGNI punteggio candidato e' non finito nessun confronto riesce
 * (NaN>bv e' falso) e best resta -1: logit[-1] e' una lettura fuori range, e
 * idx[]=-1 si propaga a eusage/eheat/elast (tre scritture heap a indice -1) e al
 * VLA seen[] (underflow di stack). Non e' teorico: c/tests/test_logit_nan.c
 * documenta NaN/Inf nei logit da un tile expert corrotto o da un overflow fp al
 * confine di eviction, e ha indurito solo il lato sampling — il router e' a monte.
 * Qui degradiamo a una scelta deterministica e avvisiamo una volta sola. */
static int router_best_or_fallback(int best, int kk, int E, int layer){
    if(best>=0) return best;
    static int warned;
    if(!warned){ warned=1;
        fprintf(stderr,"[router] logits non finiti al layer %d: selezione degradata\n",layer); }
    return kk<E ? kk : 0;
}

#ifdef COLI_METAL
/* Rotate the Metal-staged gate/up input rows for fmt=6 (Q^T x). Duplicate rows
 * (same source s) are copied from the first rotated instance: O(p^2) scan, p<=65. */
static void metal_stage_rot_e8(float *mxg, const int *mrows, int p, int D){
    for(int i=0;i<p;i++){
        int j=-1; for(int k=0;k<i;k++) if(mrows[k]==mrows[i]){ j=k; break; }
        if(j>=0) memcpy(mxg+(int64_t)i*D, mxg+(int64_t)j*D, (size_t)D*sizeof(float));
        else e8_rot_rows(mxg+(int64_t)i*D, 1, D);
    }
}
#endif
/* moe()'s MB_BUILD subset builder (review F2): coli_metal_moe_block(_begin) take ONE
 * fmt/qgs for an entire GPU batch of experts (+ optionally the fused shared expert), so
 * a candidate member at (fmt,gs) is only safe to fold into a batch already established
 * at (est_fmt,est_gs) if it can't disagree about layout: fmt=4 (grouped int4) is the
 * only format with a group size to disagree ON, so this is a no-op (always compatible)
 * for est_fmt!=4 (nothing established yet, or the batch isn't grouped) or fmt!=4 (the
 * candidate itself isn't grouped -- its own fmt mismatch is a separate, pre-existing gap
 * this guard does not cover, see MB_BUILD's comment). Pulled out of MB_BUILD as its own
 * function so it's independently testable (see tests/test_moe_gs_guard.c). */
static int mb_gs_compat(int est_fmt, int est_gs, int fmt, int gs){
    return !(est_fmt==4 && fmt==4 && gs!=est_gs);
}

static void moe(Model *m, Layer *l, int layer, float *x, int S, float *out, int with_shared){
    if(g_pilot_real){   /* barriera cross-layer: prendi possesso di QUESTO layer e aspetta
                         * l'eventuale load-pilota in volo sullo stesso layer (dopodiche' il
                         * worker droppa ogni nuovo load <= layer -> ecache[layer] e' stabile
                         * per tutto il resolve/matmul/promozione qui sotto). */
        pthread_mutex_lock(&g_pilot_mx);
        atomic_store_explicit(&g_cur_moe_layer,layer,memory_order_release);
        while(layer>=0 && layer<256 && g_pilot_inflight[layer]>0)
            pthread_cond_wait(&g_pilot_cv,&g_pilot_mx);
        pthread_mutex_unlock(&g_pilot_mx);
    }
    Cfg *c=&m->c; int D=c->hidden, E=c->n_experts, K=c->topk, I=c->moe_inter;
    /* DISK-CLASS: does THIS call need the pre-bump recency snapshot? Must agree with
     * dc_needed() in expert_load_impl -- that's what reads what this writes. touched[]
     * makes the write once-per-call: an expert routed by more than one position in a big
     * batch (prefill's S) must snapshot the state from BEFORE this call started, not from
     * an earlier position's bump within the SAME call (which would reintroduce the
     * same-call contamination one position later). Unconditional VLA like the FASE B
     * `seen[E]` below -- E is small, cost is noise. */
    int need_classify = dc_needed();
    unsigned char touched[E]; if(need_classify) memset(touched,0,(size_t)E);
    float *choice=falloc(E);
    int sI=c->moe_inter*c->n_shared;
    /* Rank buffer for CACHE_ROUTE max-rank selection (up to all E experts). */
    int *rank_buf=NULL; float *rank_w=NULL;
    int do_cache_route = g_cache_route && E>0 && K>0;
    int rank_cap = do_cache_route ? (g_route_m>K?g_route_m:K) : 0;
    if(rank_cap>E) rank_cap=E;
    if(do_cache_route){
        rank_buf=malloc((size_t)rank_cap*sizeof(int));
        rank_w=malloc((size_t)rank_cap*sizeof(float));
        if(!rank_buf||!rank_w){ free(rank_buf); free(rank_w); rank_buf=NULL; rank_w=NULL; do_cache_route=0; }
    }
    /* ---- FASE A: routing di tutte le S posizioni ---- */
    double route_t0=g_prof?now_s():0;
    int *idxs=xalloc((size_t)S*K*sizeof(int),"moe idxs"); float *ws=xalloc((size_t)S*K*sizeof(float),"moe ws");
    int *keff=xalloc((size_t)S*sizeof(int),"moe keff");
    /* router in UN matmul batch: stessa matematica, via le S chiamate S=1 */
    float *logits_all=falloc((int64_t)S*E);
    int pre_routed=0; (void)pre_routed;
    /* pre-routed shortcut: Metal layer-CB o device router CUDA (#431) — stessa
     * contabilita' (#417: recency clock incluso), la selezione arriva dalla GPU */
    if(g_pre_idx){                               /* routing gia' calcolata dal layer CB (GPU) */
        memcpy(idxs,g_pre_idx,(size_t)S*K*sizeof(int));
        memcpy(ws,g_pre_w,(size_t)S*K*sizeof(float));
        memcpy(keff,g_pre_keff,(size_t)S*sizeof(int));
        /* Stessa classe di difetto dal lato GPU: un risultato di routing malformato
         * (keff oltre K, id expert fuori range) indicizzerebbe gli stessi array
         * eusage/eheat/elast e il VLA seen[] di FASE B. Meglio sanificare qui una
         * volta che fidarsi del device. */
        for(int s=0;s<S;s++){
            if(keff[s]<0) keff[s]=0;
            if(keff[s]>K) keff[s]=K;
            for(int kk=0;kk<keff[s];kk++){
                int e=idxs[(int64_t)s*K+kk];
                if(e<0||e>=E){
                    static int warned_pre;
                    if(!warned_pre){ warned_pre=1;
                        fprintf(stderr,"[router] indice expert %d fuori range al layer %d: azzerato\n",e,layer); }
                    idxs[(int64_t)s*K+kk]=0;
                }
            }
        }
        for(int s=0;s<S;s++){
            m->ereq+=keff[s];
            for(int kk=0;kk<keff[s];kk++){
                if(m->eusage&&m->eusage[layer])
                    m->eusage[layer][idxs[(int64_t)s*K+kk]]++;
                ehit_mark(m,layer,idxs[(int64_t)s*K+kk]);
                if(need_classify){                /* DISK-CLASS private recency -- snapshot BEFORE this call's own bump,
                                                   * then tick. This path also bumps the REAL elast/eaccess_clock a few
                                                   * lines below (#417/cfcc742 fixed the once-missing bump on Metal
                                                   * decode) -- DISK-CLASS's own clock stays independent regardless of
                                                   * that fix, see elast_dc in Model. */
                    int e=idxs[(int64_t)s*K+kk];
                    if(!touched[e]){ m->elast_pre[layer][e]=m->elast_dc[layer][e]; touched[e]=1; }
                    m->elast_dc[layer][e]=++m->eaccess_clock_dc;
                }
                if(m->eheat[layer][idxs[(int64_t)s*K+kk]]<UINT32_MAX) m->eheat[layer][idxs[(int64_t)s*K+kk]]++;
                /* #417: la scorciatoia GPU-prerouted deve far avanzare l'orologio di recency
                 * come il percorso router completo (riga ~3055), altrimenti elast/eaccess_clock
                 * si congelano a fine prefill e il tie-breaker LFRU di REPIN gira su punteggi
                 * stantii durante il decode su Metal. */
                m->elast[layer][idxs[(int64_t)s*K+kk]]=++m->eaccess_clock;
            }
            for(int d=0;d<D;d++) out[(int64_t)s*D+d]=0;
        }
        pre_routed=1;
    }
    if(!pre_routed) matmul(logits_all, x, l->router, S, D, E);
    if(!pre_routed)
    for(int s=0;s<S;s++){
        float *logit=logits_all+(int64_t)s*E;
        for(int e=0;e<E;e++){ logit[e]=sigmoidf(logit[e]); choice[e]=logit[e]+l->router_bias[e]; }
        int *idx=idxs+(int64_t)s*K; float *w=ws+(int64_t)s*K;
        int Ksel = g_topk>0 ? (g_topk<K?g_topk:K) : K;
        if(do_cache_route){
            /* Full ranking of top rank_cap experts by choice (bias-augmented). */
            int Mwin=rank_cap;
            if(g_route_p>0.f && g_route_p<1.f){
                /* Cumulative-mass variant: grow M until mass covers ROUTE_P. */
                int Mmax=g_route_m>Ksel*4?g_route_m:Ksel*4; if(Mmax>E) Mmax=E; if(Mmax>rank_cap) Mmax=rank_cap;
                for(int kk=0;kk<Mmax;kk++){ int best=-1; float bv=-1e30f;
                    for(int e=0;e<E;e++){ int tk=0; for(int j=0;j<kk;j++) if(rank_buf[j]==e){tk=1;break;}
                        if(!tk && choice[e]>bv){bv=choice[e];best=e;} }
                    best=router_best_or_fallback(best,kk,E,layer);
                    rank_buf[kk]=best; rank_w[kk]=logit[best];
                }
                float tot=1e-20f; for(int kk=0;kk<Mmax;kk++) tot+=rank_w[kk]>0?rank_w[kk]:0;
                float cum=0; Mwin=Ksel;
                for(int kk=0;kk<Mmax;kk++){ cum+=rank_w[kk]>0?rank_w[kk]:0;
                    if(cum>=g_route_p*tot){ Mwin=kk+1; break; } Mwin=kk+1; }
                if(Mwin<Ksel) Mwin=Ksel;
            } else {
                for(int kk=0;kk<Mwin;kk++){ int best=-1; float bv=-1e30f;
                    for(int e=0;e<E;e++){ int tk=0; for(int j=0;j<kk;j++) if(rank_buf[j]==e){tk=1;break;}
                        if(!tk && choice[e]>bv){bv=choice[e];best=e;} }
                    best=router_best_or_fallback(best,kk,E,layer);
                    rank_buf[kk]=best; rank_w[kk]=logit[best];
                }
            }
            int J=g_route_j; if(J<0) J=0; if(J>Ksel) J=Ksel;
            int chosen=0;
            /* Always take true top-J (even if uncached). */
            for(int kk=0;kk<J && chosen<Ksel;kk++){
                idx[chosen]=rank_buf[kk]; w[chosen]=rank_w[kk]; chosen++;
            }
            /* Remaining slots: prefer resident experts within top-Mwin. */
            for(int r=J;r<Mwin && chosen<Ksel;r++){
                int e=rank_buf[r]; int already=0;
                for(int j=0;j<chosen;j++) if(idx[j]==e){already=1;break;}
                if(already) continue;
                if(expert_is_resident(m,layer,e)){
                    idx[chosen]=e; w[chosen]=rank_w[r]; chosen++;
                }
            }
            /* Fill remainder from true ranking order. */
            for(int r=0;r<Mwin && chosen<Ksel;r++){
                int e=rank_buf[r]; int already=0;
                for(int j=0;j<chosen;j++) if(idx[j]==e){already=1;break;}
                if(already) continue;
                idx[chosen]=e; w[chosen]=rank_w[r]; chosen++;
            }
            /* Swap accounting vs true top-Ksel (rank_buf[0..Ksel)). */
            m->route_slots+=(uint64_t)Ksel;
            for(int kk=0;kk<Ksel;kk++){
                int e=idx[kk], in_true=0;
                for(int t=0;t<Ksel;t++) if(rank_buf[t]==e){in_true=1;break;}
                if(!in_true) m->route_swaps++;
            }
            /* Pad if somehow short (shouldn't happen). */
            while(chosen<Ksel){ idx[chosen]=rank_buf[chosen]; w[chosen]=rank_w[chosen]; chosen++; }
            /* ROUTE_ALPHA: down-weight substituted experts' gate mass before renorm. */
            if(g_route_alpha>0.f && g_route_alpha<1.f){
                for(int kk=0;kk<Ksel;kk++){
                    int e=idx[kk], in_true=0;
                    for(int t=0;t<Ksel;t++) if(rank_buf[t]==e){in_true=1;break;}
                    if(!in_true) w[kk]*=g_route_alpha;
                }
            }
            /* ROUTE_AGREE: overlap + KL(true top-K mass || chosen mass). */
            if(g_route_agree || g_cache_route){
                int ov=0;
                for(int kk=0;kk<Ksel;kk++){
                    for(int t=0;t<Ksel;t++) if(idx[kk]==rank_buf[t]){ ov++; break; }
                }
                m->route_agree_hit+=(uint64_t)ov;
                m->route_agree_tot+=(uint64_t)Ksel;
                float tsum=1e-20f, csum=1e-20f;
                for(int t=0;t<Ksel;t++) tsum+=rank_w[t]>0?rank_w[t]:0;
                for(int kk=0;kk<Ksel;kk++) csum+=w[kk]>0?w[kk]:0;
                double kl=0;
                for(int t=0;t<Ksel;t++){
                    double pt=(rank_w[t]>0?rank_w[t]:0)/tsum;
                    if(pt<=0) continue;
                    double pc=1e-12;
                    for(int kk=0;kk<Ksel;kk++) if(idx[kk]==rank_buf[t]){
                        pc=(w[kk]>0?w[kk]:0)/csum; break; }
                    kl+=pt*log(pt/pc);
                }
                m->route_kl_sum+=kl; m->route_kl_n++;
            }
        } else {
            for(int kk=0;kk<Ksel;kk++){ int best=-1; float bv=-1e30f;
                for(int e=0;e<E;e++){ int tk=0; for(int j=0;j<kk;j++) if(idx[j]==e){tk=1;break;}
                    if(!tk && choice[e]>bv){bv=choice[e];best=e;} }
                best=router_best_or_fallback(best,kk,E,layer);
                idx[kk]=best; w[kk]=logit[best];
            }
            if(g_route_agree){
                m->route_agree_hit+=(uint64_t)Ksel;
                m->route_agree_tot+=(uint64_t)Ksel;
                m->route_kl_sum+=0; m->route_kl_n++;
            }
        }
        int Ke=Ksel;
        if(g_topp>0 && g_topp<1.f){
            for(int a=1;a<Ksel;a++){ int ii=idx[a]; float ww=w[a]; int b=a-1;
                while(b>=0 && w[b]<ww){ w[b+1]=w[b]; idx[b+1]=idx[b]; b--; } w[b+1]=ww; idx[b+1]=ii; }
            float tot=1e-20f; for(int kk=0;kk<Ksel;kk++) tot+=w[kk];
            float cum=0; for(int kk=0;kk<Ksel;kk++){ cum+=w[kk]; if(cum>=g_topp*tot){ Ke=kk+1; break; } }
        }
        /* ===== CAUSAL ABLATION (FASE A) — inert unless g_abl.mode set this item.
         * Both hooks act on the token's SELECTED set BEFORE the eusage/eheat/elast
         * counters, norm_topk, routed_scale, and ROUTE_TRACE below, so the load
         * counters, renormalised weights, and any route trace all reflect the
         * post-ablation routing (route-around: the ablated cell truly did not fire;
         * module-swap: the substitute fired at the original routed weight). */
        if(g_abl.mode==2){                 /* route-around: drop ablated cells, survivors renormalise below */
            int wr=0;
            for(int kk=0;kk<Ke;kk++){
                if(abl_route_around(&g_abl, layer, idx[kk])) continue;
                idx[wr]=idx[kk]; w[wr]=w[kk]; wr++;
            }
            Ke=wr;
        } else if(g_abl.mode==3){          /* module-swap: remap the slot's expert id, keep its weight */
            for(int kk=0;kk<Ke;kk++){
                int to=abl_swap_target(&g_abl, layer, idx[kk]);
                if(to>=0) idx[kk]=to;
            }
        }
        keff[s]=Ke; m->ereq+=Ke;
        for(int kk=0;kk<Ke;kk++){
            if(m->eusage&&m->eusage[layer]) m->eusage[layer][idx[kk]]++;
            ehit_mark(m,layer,idx[kk]);
            if(need_classify){                    /* DISK-CLASS private recency -- snapshot BEFORE this call's own bump,
                                                   * then tick (same rate as the real clock below: one per (s,kk)) */
                if(!touched[idx[kk]]){ m->elast_pre[layer][idx[kk]]=m->elast_dc[layer][idx[kk]]; touched[idx[kk]]=1; }
                m->elast_dc[layer][idx[kk]]=++m->eaccess_clock_dc;
            }
            if(m->eheat[layer][idx[kk]]<UINT32_MAX) m->eheat[layer][idx[kk]]++;
            m->elast[layer][idx[kk]]=++m->eaccess_clock;
        }
        if(c->norm_topk){ float sm=0; for(int kk=0;kk<Ke;kk++) sm+=w[kk]; sm+=1e-20f; for(int kk=0;kk<Ke;kk++) w[kk]/=sm; }
        for(int kk=0;kk<Ke;kk++) w[kk]*=c->routed_scale;
        rt_trace(layer,s,idx,w,Ke);           /* ROUTE_TRACE: one line per (position, layer) */
        for(int d=0;d<D;d++) out[(int64_t)s*D+d]=0;
    }
    free(rank_buf); free(rank_w);
    if(g_prof)m->t_route+=now_s()-route_t0;
    rt_trace_end();
    if(g_couple && cp_pred && S<=8)
        for(int s2=0;s2<S;s2++) couple_prefetch(m,layer,idxs+(int64_t)s2*K,keff[s2]);
    if(g_looka && S==1 && layer<c->n_layers){
        int Ke=keff[0];
        if(m->enr[layer]>0){                       /* [0] vs routing del token precedente */
            for(int kk=0;kk<Ke;kk++) for(int z=0;z<m->enr[layer];z++)
                if(m->eroute[layer][z]==idxs[kk]){ la_hit[0]++; break; }
            la_tot[0]+=Ke;
        }
        for(int kind=0;kind<3;kind++) if(la_val[kind][layer]){   /* score all prediction kinds */
            for(int kk=0;kk<Ke;kk++) for(int z=0;z<K;z++)
                if(la_pred[kind][layer][z]==idxs[kk]){ la_hit[1+kind]++; break; }
            la_tot[1+kind]+=Ke; la_val[kind][layer]=0;
        }
    }
    m->enr[layer]=keff[S-1]; for(int kk=0;kk<keff[S-1];kk++) m->eroute[layer][kk]=idxs[(int64_t)(S-1)*K+kk];
    /* ---- FASE B: union degli expert del batch ---- */
    int *uniq=xalloc((size_t)E*sizeof(int),"moe uniq"); int nu=0;
    unsigned char seen[E]; memset(seen,0,(size_t)E);
    for(int s=0;s<S;s++) for(int kk=0;kk<keff[s];kk++){
        int e=idxs[(int64_t)s*K+kk];
        if(!seen[e]){ seen[e]=1; uniq[nu++]=e; }
    }
    /* EXPERT_BUDGET: cap distinct experts per layer to reduce disk I/O on cold/low-RAM
     * hosts. MISS-AWARE: always keep cache hits (pin/LRU — they're free, no disk I/O),
     * only drop from misses. From the misses, keep the highest-aggregate-gate-weight
     * ones up to the budget; drop the rest from idxs[] so they're never loaded.
     * (MoE-Spec arXiv 2602.16052: top-32 of 64 capture 93% routing weight.)
     * Complementary to TOPP (per-position) — this trims cross-position.
     * DECODE-ONLY (S<=4, incl. MTP verify): during prefill S=prompt_len the batch
     * union nu is 30-100+ experts and capping to 4-8 drops 80-90% of them, each with
     * non-trivial gate weight -> corrupted prefill hidden state -> wrong KV cache ->
     * repetitive garbage decode. The budget is only safe token-by-token, where the
     * prefill KV cache is already correct. (woolcoxm, #292.) */
    if(g_expert_budget>0 && S<=4 && nu>g_expert_budget){
        /* compute aggregate gate weight per unique expert */
        float *wsum=falloc(nu); for(int j=0;j<nu;j++) wsum[j]=0;
        for(int s=0;s<S;s++) for(int kk=0;kk<keff[s];kk++){
            int e=idxs[(int64_t)s*K+kk];
            for(int j=0;j<nu;j++) if(uniq[j]==e){ wsum[j]+=ws[(int64_t)s*K+kk]; break; }
        }
        /* residency pre-scan: which experts are already in pin or ecache (hits)? */
        unsigned char *is_hit=xzalloc((size_t)nu,"moe is_hit"); int nhits=0;
        for(int j=0;j<nu;j++){ int eid=uniq[j];
            int found=expert_is_resident(m,layer,eid);
            if(found){ is_hit[j]=1; nhits++; }
        }
        /* budget for misses = total budget - hits already kept (min 0) */
        int miss_budget = g_expert_budget - nhits; if(miss_budget<0) miss_budget=0;
        /* mark which unique experts to keep (1) or drop (0): keep all hits, fill rest
         * with top-weight misses up to miss_budget */
        unsigned char *keep=xzalloc((size_t)nu,"moe keep"); int nkeep=0;
        for(int j=0;j<nu;j++) if(is_hit[j]){ keep[j]=1; nkeep++; }
        for(int rank=0;rank<miss_budget;rank++){
            int best=-1; float bv=-1e30f;
            for(int j=0;j<nu;j++) if(!keep[j] && wsum[j]>bv){ bv=wsum[j]; best=j; }
            if(best<0) break; keep[best]=1; nkeep++;
        }
        /* build a lookup: for each expert id, is it kept? (reuse seen[]) */
        memset(seen,0,(size_t)E);
        for(int j=0;j<nu;j++) if(keep[j]) seen[uniq[j]]=1;
        /* Nessuna posizione puo' restare a ZERO routed expert. keep[] parte da tutti
         * i cache hit, quindi con nhits>=g_expert_budget il miss_budget e' 0 e una
         * posizione il cui top-K e' fatto solo di miss viene compattata a w==0 (il
         * guard `w>0` qui sotto lo ammette). Quel token riceverebbe solo lo shared
         * expert, e lo stato nascosto sbagliato finisce nella KV cache avvelenando
         * ogni token successivo — la stessa modalita' di guasto descritta sopra per
         * il prefill (#292), raggiunta da un'altra direzione. Bastano S=2, K=1,
         * EXPERT_BUDGET=1 e due miss diversi. Ripeschiamo il miss col peso di gate
         * piu' alto della posizione e lo contiamo in STATS. */
        for(int s=0;s<S;s++){
            int alive=0;
            for(int kk=0;kk<keff[s] && !alive;kk++) if(seen[idxs[(int64_t)s*K+kk]]) alive=1;
            if(alive || keff[s]<=0) continue;
            int be=-1; float bw=-1e30f;
            for(int kk=0;kk<keff[s];kk++){
                float wv=ws[(int64_t)s*K+kk];
                if(wv>bw){ bw=wv; be=idxs[(int64_t)s*K+kk]; }
            }
            if(be<0) be=idxs[(int64_t)s*K];        /* pesi tutti non finiti: primo della lista */
            seen[be]=1; g_budget_rescued++;
            for(int j=0;j<nu;j++) if(uniq[j]==be && !keep[j]){ keep[j]=1; nkeep++; break; }
        }
        int dropped=nu-nkeep; g_budget_dropped+=dropped;
        /* remove dropped experts from each position's routing list */
        for(int s=0;s<S;s++){
            int w=0; float sold=0, snew=0;
            for(int kk=0;kk<keff[s];kk++){
                int e=idxs[(int64_t)s*K+kk]; float wv=ws[(int64_t)s*K+kk];
                sold+=wv;
                if(seen[e]){ idxs[(int64_t)s*K+w]=e; ws[(int64_t)s*K+w]=wv; snew+=wv; w++; }
            }
            if(w<keff[s]){
                keff[s]=w;
                /* renormalize remaining weights per position */
                if(c->norm_topk && w>0){
                    /* sm porta gia' routed_scale (applicato in FASE A): la divisione lo
                     * cancella e il multiply qui sotto lo riapplica — non e' doppio scaling */
                    float sm=0; for(int kk=0;kk<w;kk++) sm+=ws[(int64_t)s*K+kk]; sm+=1e-20f;
                    for(int kk=0;kk<w;kk++) ws[(int64_t)s*K+kk]/=sm;
                    for(int kk=0;kk<w;kk++) ws[(int64_t)s*K+kk]*=c->routed_scale;
                } else if(w>0 && snew>1e-20f && sold>snew){
                    /* Senza norm_topk i pesi NON sono normalizzati: scartare expert
                     * ridurrebbe la magnitudine del contributo routed di tutta la massa
                     * di gate buttata via, cioe' il budget cambierebbe la matematica del
                     * layer invece di risparmiare solo I/O. Riscaliamo per old/new (il
                     * rapporto e' invariante a routed_scale) cosi' la magnitudine resta.
                     * GLM-5.2 usa norm_topk=1, quindi questo tocca solo le altre config MoE. */
                    float sc=sold/snew;
                    for(int kk=0;kk<w;kk++) ws[(int64_t)s*K+kk]*=sc;
                }
            }
        }
        /* compact uniq[] to kept experts only */
        int nu2=0;
        for(int j=0;j<nu;j++) if(keep[j]) uniq[nu2++]=uniq[j];
        nu=nu2;
        free(wsum); free(is_hit); free(keep);
    }
    /* ---- FASE C/D: risolvi (pin/cache/disco) e calcola, a blocchi di 64 unici ---- */
    float *xg=falloc((int64_t)S*D), *gg=falloc((int64_t)S*I), *uu=falloc((int64_t)S*I), *hh=falloc((int64_t)S*D);
    /* quantizzazione sollevata (g_pq): materializzata al PRIMO expert che prenderebbe
     * l'IDOT — mai per fmt 0/3/4/5/6/8, mai sui percorsi GPU, costo zero se inutile.
     * EN: hoisted activation quantization, materialized on the FIRST IDOT-eligible
     * expert; zero cost when no expert would take the IDOT branch. */
    int8_t *xq_all=NULL, *xqg=NULL; float *sx_all=NULL, *sxg=NULL; int pq_ready=0;
    int32_t *xsum_all=NULL, *xsumg=NULL;   /* K1: termine -8*sum(x) per il dot planare */
    g_pq.x=NULL;      /* mai fidarsi di un contesto di un moe() precedente / never trust a stale context */
    float *xe=NULL;   /* fmt=6: x under the rotation Q^T, built once per call — all routed
                       * experts of the layer share it (the placement rule in quant.h) */
    /* Materialise xe on first use. Every site that feeds a routed expert's gate/up
     * input — CPU, the per-expert GPU call, and all three group packing paths —
     * must go through here: doing the transform per expert instead of per layer
     * costs ~11 ms against ~1.4 ms on GLM dims (#452). */
    #define E8_XE(e) ((e)->g.fmt==6 ? (xe ? xe : (xe=falloc((int64_t)S*D), \
        memcpy(xe,x,(size_t)S*D*sizeof(float)), e8_rot_rows(xe,S,D), xe)) : x)
    int *rows=xalloc((size_t)S*sizeof(int),"moe rows"); float *rw=xalloc((size_t)S*sizeof(float),"moe rw");
#ifdef COLI_CUDA
    /* PIPE Inc.1b: il batch-union del prefill passa dai gruppi GPU — prima di
     * questo, 9343 expert in VRAM restavano INUTILIZZATI durante il prefill
     * (misurato: 81s di expert-matmul tutto su CPU, GPU groups 21ms totali). */
    int group_enabled = S<=64 || (g_cuda_pipe && S<=4096);
    float *group_x=group_enabled?falloc((int64_t)S*K*D):NULL;
    float *group_y=group_enabled?falloc((int64_t)S*K*D):NULL;
    int *group_row=group_enabled?xalloc((size_t)64*S*sizeof(int),"moe group_row"):NULL;
    float *group_weight=group_enabled?xalloc((size_t)64*S*sizeof(float),"moe group_weight"):NULL;
#endif
    int vk_active = 0; (void)vk_active;
#ifdef COLI_VULKAN
    vk_active = g_vulkan && (g_vk_reg_n+g_vk_reg_n2)>0 && !omp_in_parallel() && S<=4;   /* empty
                                                * registry (tier off / no usage history) = normal CPU expert loop */
    float *vk_xh = vk_active?falloc((int64_t)S*K*D):NULL;
    float *vk_yh = vk_active?falloc((int64_t)S*K*D):NULL;
    int vk2_on = vk_active && g_vk_reg_n2>0;    /* dev2 tier live: second async group */
    float *vk_xh2 = vk2_on?falloc((int64_t)S*K*D):NULL;
    float *vk_yh2 = vk2_on?falloc((int64_t)S*K*D):NULL;
#endif
    int shared_on_gpu=0; (void)shared_on_gpu;   /* set by the Metal path when Phase E was fused */
    for(int base=0;base<nu;base+=64){
        int nb = nu-base<64 ? nu-base : 64;
#if !defined(_WIN32)
        if(g_cluster_n){
            cluster_moe_batch(m,layer,x,S,out,idxs,ws,keff,K,uniq,base,nb);
            continue;
        }
#endif
        ESlot *use[64]; int missk[64]; int qof[64]; int nmiss=0;
#ifdef COLI_VULKAN
        int vk_hit[64]={0};
#endif
        for(int j=0;j<nb;j++){ int eid=uniq[base+j]; use[j]=NULL; qof[j]=-1;
#ifdef COLI_VULKAN
            /* VK VRAM tier first: registry-served experts need NO RAM slot and NO disk
             * load (the whole point) — and skipping the LRU recency bump lets them age
             * out of the RAM cache, freeing capacity for the CPU-served experts. */
            if(vk_active && layer<c->n_layers){
                ColiVkTensor **rg=vk_reg_at(layer,eid);
                if(rg && rg[0]){ vk_hit[j]=1; m->hits++; m->hit_vk++; continue; }
            }
#endif
            use[j]=pin_indexed(m,layer,eid);
            if(use[j]){ m->hits++; m->hit_pin++; }
            if(!use[j]){
                use[j]=ecache_indexed(m,layer,eid,0);
                if(use[j]){ m->hits++; m->hit_ecache++; use[j]->used=(uint64_t)__atomic_add_fetch(&m->eclock,1,__ATOMIC_RELAXED); }
            }
            if(!use[j]){ qof[j]=nmiss; use[j]=&m->ws[nmiss]; missk[nmiss++]=j; m->miss++;
                if(g_disk_split){ if(m->ld_ctx==1) m->miss_draft++; else if(m->ld_ctx==2) m->miss_absorb++; } }
        }
        int metal_done=0;
#ifdef COLI_METAL
        /* GPU/disk OVERLAP: submit the RESIDENT experts (pin/LRU hits, + shared expert on
         * the first block) to the GPU BEFORE loading the missed experts from disk, so the
         * preads run while the GPU computes; the missed subset follows in a second submit.
         * Per-subset CPU fallback on unresolved slab / bad fmt / GPU fault. */
        /* fmt=6 stores W@Q, so the staged gate/up input must be rotated (Q^T x) before
         * upload. Rotate each distinct source row once, copy to duplicates — a decode
         * batch's rows all share s, so this is one FWHT per token in practice. The down
         * input is rotated on-GPU by moe_fwht inside moe_submit. */
        int is_miss[64]={0}; ColiMetalMoeHandle *mh=NULL;
        int cpu_res=1, cpu_miss=1, mh_shared=0, nbb=0, Rtot=0, mfmt=-1, mgs=0, mgs_ok=1, sh_in=0;
        const void *MG[65],*MU[65],*MD[65]; const float *MGS[65],*MUS[65],*MDS[65];
        int xoffb[65],nrb[65];
        float *mxg=NULL; int *mrows=NULL; float *mrw=NULL;
        /* subset builder: experts with is_miss==WANTMISS (+ shared expert when TRY_SH).
         * mgs_ok (review F2): moe_submit takes ONE fmt/qgs for the WHOLE batch, so a
         * fmt=4 subset needs every member's group size to agree with the first expert's
         * (mgs) -- fmt equality alone (already checked below for the shared expert) isn't
         * enough once fmt=4 has a group size to disagree on. Routed experts get the same
         * guard against each other (first-expert-gs consistency): a v1-class mixed-
         * precision container could in principle mint routed experts at different gs
         * within one layer (qt_resolve_fmt derives fmt/gs per-tensor from the file, with
         * no uniformity enforced across experts at load time), even though a normal
         * single-pass conversion never would. mgs_ok=0 does NOT drop the mismatched
         * expert's rows: it leaves nbb's bookkeeping untouched and instead suppresses the
         * GPU submit call at the two call sites below, which leaves cpu_res/cpu_miss at
         * their initial 1 -- the same state a genuine GPU submission failure leaves them
         * in, so the existing CPU fallback loop (metal_done false) redoes this whole
         * nb-expert block correctly regardless of the mismatch. (Not checked, and out of
         * scope for this guard: per-expert fmt agreement across g/u/d, and gs agreement
         * across sh_gate/sh_up/sh_down or across a single expert's own g/u/d -- see
         * WORKER_REPORT UNCERTAINTIES.) */
        #define MB_BUILD(WANTMISS, TRY_SH) do{ \
            nbb=0; Rtot=0; mfmt=-1; mgs=0; mgs_ok=1; sh_in=0; \
            for(int j=0;j<nb;j++){ if(is_miss[j]!=(WANTMISS)) continue; \
                int eid=uniq[base+j]; ESlot *e=use[j]; int cnt=0; \
                for(int s=0;s<S;s++) for(int kk=0;kk<keff[s];kk++) \
                    if(idxs[(int64_t)s*K+kk]==eid){ cnt++; break; } \
                if(!cnt) continue; \
                if(mfmt<0){ mfmt=e->g.fmt; mgs=e->g.gs; } \
                else if(!mb_gs_compat(mfmt,mgs,e->g.fmt,e->g.gs)) mgs_ok=0; \
                MG[nbb]=e->g.fmt==1?(const void*)e->g.q8:(const void*)e->g.q4; \
                MU[nbb]=e->u.fmt==1?(const void*)e->u.q8:(const void*)e->u.q4; \
                MD[nbb]=e->d.fmt==1?(const void*)e->d.q8:(const void*)e->d.q4; \
                MGS[nbb]=e->g.s; MUS[nbb]=e->u.s; MDS[nbb]=e->d.s; \
                xoffb[nbb]=Rtot; nrb[nbb]=cnt; Rtot+=cnt; nbb++; \
            } \
            if(TRY_SH){ int shf = mfmt<0 ? l->sh_gate.fmt : mfmt; \
                if(c->n_shared==1 && sI==I && l->sh_gate.fmt==shf && l->sh_up.fmt==shf && l->sh_down.fmt==shf \
                   && mb_gs_compat(mfmt,mgs,shf,l->sh_gate.gs)){ \
                    if(mfmt<0){ mfmt=shf; mgs=l->sh_gate.gs; } \
                    MG[nbb]=shf==1?(const void*)l->sh_gate.q8:(const void*)l->sh_gate.q4; \
                    MU[nbb]=shf==1?(const void*)l->sh_up.q8  :(const void*)l->sh_up.q4; \
                    MD[nbb]=shf==1?(const void*)l->sh_down.q8:(const void*)l->sh_down.q4; \
                    MGS[nbb]=l->sh_gate.s; MUS[nbb]=l->sh_up.s; MDS[nbb]=l->sh_down.s; \
                    xoffb[nbb]=Rtot; nrb[nbb]=S; Rtot+=S; nbb++; sh_in=1; } } \
            int p=0; \
            for(int j=0;j<nb;j++){ if(is_miss[j]!=(WANTMISS)) continue; int eid=uniq[base+j]; \
                for(int s=0;s<S;s++) for(int kk=0;kk<keff[s];kk++) \
                    if(idxs[(int64_t)s*K+kk]==eid){ \
                        memcpy(mxg+(int64_t)p*D, x+(int64_t)s*D, D*sizeof(float)); \
                        mrows[p]=s; mrw[p]=ws[(int64_t)s*K+kk]; p++; break; } } \
            if(sh_in) for(int s=0;s<S;s++){ \
                memcpy(mxg+(int64_t)p*D, x+(int64_t)s*D, D*sizeof(float)); \
                mrows[p]=s; mrw[p]=1.0f; p++; } \
        }while(0)
        if(g_metal_enabled){
            for(int q=0;q<nmiss;q++) is_miss[missk[q]]=1;
            mxg=falloc((int64_t)(nb+1)*S*D);
            mrows=xalloc((size_t)(nb+1)*S*sizeof(int),"moe mrows"); mrw=xalloc((size_t)(nb+1)*S*sizeof(float),"moe mrw");
            MB_BUILD(0, base==0 && !g_pre_sh);
            if(nbb>0 && mgs_ok){
                double t0=now_s();
                if(mfmt==6) metal_stage_rot_e8(mxg,mrows,Rtot,D);
                mh=coli_metal_moe_block_begin(nbb,D,I,mfmt,mgs,MG,MU,MD,MGS,MUS,MDS,mxg,xoffb,nrb,mrows,mrw);
                m->t_emm += now_s()-t0;
                if(mh){ cpu_res=0; mh_shared=sh_in; }
            } else if(!nbb) cpu_res=0;   /* nbb==0: nothing in this subset. nbb>0 && !mgs_ok
                                          * (F2): gs-heterogeneous fmt=4 subset -- leave
                                          * cpu_res=1 so the CPU loop below redoes it. */
        }
#endif
        /* Expert loads run HERE, after the resident-experts GPU submit above: under METAL the
         * preads overlap the GPU compute (that submit is async). With METAL off the submit block
         * is a no-op / compiled out, so this sits exactly where dev put it and CPU behaviour is
         * unchanged. */
        if(nmiss){
            if(g_pipe){                            /* PIPE: launch loads async, matmul overlaps them */
                if(!g_pp.started) pipe_init(m);
                double t0=now_s();
                int eids[64]; for(int q=0;q<nmiss;q++) eids[q]=uniq[base+missk[q]];
                pipe_dispatch(m,layer,eids,nmiss);
                m->t_ewait += now_s()-t0;           /* dispatch only; the reads overlap matmul and
                                                     * are timed as service inside expert_load */
            } else { double t0=now_s();             /* ORIGINALE: blocking parallel load */
                #pragma omp parallel for schedule(dynamic,1)
                for(int q=0;q<nmiss;q++) expert_load(m,layer,uniq[base+missk[q]],&m->ws[q],1,1);   /* demand=1: this IS the miss path */
                m->t_ewait += now_s()-t0; }         /* compute thread blocked for the whole load */
        }
        /* I/O ASINCRONO: readahead (WILLNEED) del blocco SUCCESSIVO mentre calcoliamo
         * questo — il kernel legge in background, le pread dopo trovano cache calda */
        if(base+64<nu){
            int nb2 = nu-(base+64)<64 ? nu-(base+64) : 64;
            for(int j=0;j<nb2;j++){ int eid=uniq[base+64+j]; int found=0;
#ifdef COLI_VULKAN
                if(vk_active && vk_reg_served(layer,eid)) found=1;   /* VK-tier-served at decode */
#endif
                if(!found) found=expert_is_resident(m,layer,eid);
                if(!found) expert_prefetch(m,layer,eid);
            }
        }
#ifdef COLI_CUDA
        ESlot *group_e[64]; int group_n[64]; int ngroup=0;
        /* Inc.4 overlap stash: pass-1 packing kept for the take phase after the CPU loop */
        ESlot *eg_e[64]; int eg_n[64], eg_row[64][4], eg_npg=0; float eg_w[64][4];
        int dev_nc0[COLI_CUDA_MAX_DEVICES], dev_off0[COLI_CUDA_MAX_DEVICES],
            dev_total0[COLI_CUDA_MAX_DEVICES], dev_which0[COLI_CUDA_MAX_DEVICES][64];
        memset(dev_nc0,0,sizeof(dev_nc0)); (void)eg_npg; (void)dev_total0; (void)dev_off0;
#endif
#ifdef COLI_METAL
        if(g_metal_enabled){
            /* PIPE drain. Two reasons this barrier is mandatory here, and not optional:
             *  1) MB_BUILD(1) hands the missed experts' slabs straight to the GPU — a slot still
             *     being pread by an I/O worker would be matmul-ed half-loaded.
             *  2) PIPE's only drain barrier is the per-expert pipe_wait() in the CPU matmul loop
             *     below, which metal_done SKIPS ENTIRELY. Without this, a still-writing worker
             *     would race the end-of-block LRU swap that recycles ws[].
             * pipe_wait() is an idempotent spin on ready[q], so the per-expert waits below stay
             * correct (and free) when a subset falls back to the CPU. */
            if(g_pipe && nmiss){ double tw=now_s();
                for(int q=0;q<nmiss;q++) pipe_wait(q);
                m->t_ewait += now_s()-tw; }
            MB_BUILD(1, 0);                                   /* missed experts, now loaded */
            if(nbb>0 && mgs_ok){
                double t0=now_s();
                if(mfmt==6) metal_stage_rot_e8(mxg,mrows,Rtot,D);
                if(coli_metal_moe_block(nbb,D,I,mfmt,mgs,MG,MU,MD,MGS,MUS,MDS,mxg,xoffb,nrb,mrows,mrw,out,S)) cpu_miss=0;
                m->t_emm += now_s()-t0;
            } else if(!nbb) cpu_miss=0;   /* see the resident-subset call site above */
            if(mh){ double t0=now_s();
                if(coli_metal_moe_block_end(mh,out)){ if(mh_shared) shared_on_gpu=1; }
                else cpu_res=1;
                m->t_emm += now_s()-t0; mh=NULL; }
            metal_done = (!cpu_res && !cpu_miss);
            free(mxg); free(mrows); free(mrw);
        }
        #undef MB_BUILD
#endif
#ifdef COLI_CUDA
        /* Inc.4 pass 1: collect the VRAM-resident experts' groups and ISSUE them async
         * BEFORE the CPU loop below, so the GPU computes its share while the CPU works
         * through the RAM-tier/miss rows — t_emm becomes max(cpu, gpu) instead of the
         * sum. Only resident experts are collected (misses are never cuda_eligible), so
         * no pipe_wait is needed here; the CPU loop keeps its own waits. Any issue
         * failure drops the layer back to the collect-in-loop + sync-group path. */
        int early_issued=0, done_j[64]={0};
        double early_issue_start=0;
        {
            static int g_group_async2=-1;
            if(g_group_async2<0) g_group_async2=getenv("COLI_GROUP_ASYNC")?atoi(getenv("COLI_GROUP_ASYNC")):0;
            if(!metal_done && g_group_async2 && group_enabled && S<=4 && g_cuda_enabled &&
               g_cuda_ndev>0 && !omp_in_parallel()){
                ESlot *pg_e[64]; int pg_n[64], pg_j[64], npg=0;
                int prow[64][4]; float pw[64][4];
                for(int j=0;j<nb;j++){ ESlot *e=use[j]; int eid=uniq[base+j];
                    if(!(e->g.cuda_eligible&&e->u.cuda_eligible&&e->d.cuda_eligible)) continue;
                    int nr=0;
                    for(int s=0;s<S && nr<4;s++) for(int kk=0;kk<keff[s];kk++)
                        if(idxs[(int64_t)s*K+kk]==eid){ prow[npg][nr]=s; pw[npg][nr]=ws[(int64_t)s*K+kk]; nr++; break; }
                    if(!nr) continue;
                    pg_e[npg]=e; pg_n[npg]=nr; pg_j[npg]=j; npg++;
                }
                if(npg){
                    /* pack per device exactly like the sync path below */
                    ColiCudaTensor *pd_g[COLI_CUDA_MAX_DEVICES][64],*pd_u[COLI_CUDA_MAX_DEVICES][64],*pd_d[COLI_CUDA_MAX_DEVICES][64];
                    int pd_rows[COLI_CUDA_MAX_DEVICES][64],pd_which[COLI_CUDA_MAX_DEVICES][64];
                    int pd_nc[COLI_CUDA_MAX_DEVICES]={0},pd_total[COLI_CUDA_MAX_DEVICES]={0},pd_off[COLI_CUDA_MAX_DEVICES]={0};
                    for(int di=0;di<g_cuda_ndev;di++) for(int q=0;q<npg;q++)
                        if(pg_e[q]->g.cuda_device==g_cuda_devices[di]) pd_total[di]+=pg_n[q];
                    for(int di=1;di<g_cuda_ndev;di++) pd_off[di]=pd_off[di-1]+pd_total[di-1];
                    for(int di=0;di<g_cuda_ndev;di++){
                        int cursor=0,device=g_cuda_devices[di];
                        for(int q=0;q<npg;q++) if(pg_e[q]->g.cuda_device==device){
                            int nc=pd_nc[di]++; ESlot *e=pg_e[q];
                            pd_g[di][nc]=e->g.cuda; pd_u[di][nc]=e->u.cuda; pd_d[di][nc]=e->d.cuda;
                            pd_rows[di][nc]=pg_n[q]; pd_which[di][nc]=q;
                            const float *xsrc=E8_XE(e);      /* fmt=6 feeds the rotated copy */
                            for(int r=0;r<pg_n[q];r++) memcpy(group_x+(int64_t)(pd_off[di]+cursor+r)*D,
                                xsrc+(int64_t)prow[q][r]*D,D*sizeof(float));
                            cursor+=pg_n[q];
                        }
                    }
                    double tg0=now_s();
                    int all=1, issued[COLI_CUDA_MAX_DEVICES]={0};
                    for(int di=0;di<g_cuda_ndev && all;di++) if(pd_nc[di]){
                        ESlot *refs[64]; for(int q=0;q<pd_nc[di];q++) refs[q]=pg_e[pd_which[di][q]];
                        eslots_acquire(refs,pd_nc[di]);
                        all=issued[di]=coli_cuda_expert_group_issue(pd_g[di],pd_u[di],pd_d[di],
                                pd_rows[di],pd_nc[di],group_x+(int64_t)pd_off[di]*D);
                        if(!issued[di]) eslots_release(refs,pd_nc[di]);
                    }
                    g_ovl_issue+=now_s()-tg0;
                    if(all){
                        static int announced2;
                        if(!announced2){ announced2=1; fprintf(stderr,"[CUDA] expert group overlap active\n"); }
                        early_issued=1; early_issue_start=tg0; g_ovl_mark=now_s();
                        for(int q=0;q<npg;q++) done_j[pg_j[q]]=1;
                        /* stash packing for the take phase */
                        for(int di=0;di<g_cuda_ndev;di++){ dev_nc0[di]=pd_nc[di]; dev_off0[di]=pd_off[di]; dev_total0[di]=pd_total[di];
                            for(int q=0;q<pd_nc[di];q++) dev_which0[di][q]=pd_which[di][q]; }
                        for(int q=0;q<npg;q++){ eg_e[q]=pg_e[q]; eg_n[q]=pg_n[q];
                            for(int r=0;r<pg_n[q];r++){ eg_row[q][r]=prow[q][r]; eg_w[q][r]=pw[q][r]; } }
                        eg_npg=npg;
                        m->t_emm+=now_s()-tg0;
                        for(int q=0;q<npg;q++){                    /* bookkeeping normally done in the loop */
                            m->gpu_expert_calls++;
                        }
                    } else {
                        for(int di=0;di<g_cuda_ndev;di++) if(issued[di]){
                            coli_cuda_expert_group_take(g_cuda_devices[di]);
                            for(int q=0;q<pd_nc[di];q++) eslot_release(pg_e[pd_which[di][q]]);
                        }
                    }
                }
            }
        }
#endif
        /* ---- XEXP=1: one parallel region across ALL experts of the block (S==1, all
         * resident, all int4, IDOT S=1 family active). The default path opens ~2 OpenMP
         * regions per expert (16/layer at topk=8) and each worker touches only ~200 KB
         * per region; on wide multi-socket hosts the fork/join cadence and short streams
         * cap the expert loop well below DRAM bandwidth. Here: phase 1 computes gate+up
         * row-chunks (dot_i4i8 per row, same function as matmul_i4_idot S=1) and applies
         * silu*up on the chunk; a per-expert pass requantizes the intermediate; phase 2
         * computes down-projection row-chunks. One region, two internal barriers, per
         * block. Byte-identical to the stock g_i4s<=1 + COLI_NO_FUSED_PAIR path (verified
         * on GLM-5.2 int4: identical 256-token greedy output, and on the int4 tiny
         * oracle). Gated off the speculation window like every S-dependent kernel switch. */
        int xexp_done=0;
        if(g_xexp && !metal_done && !vk_active && S==1 && !nmiss && !spec_pinned() && g_idot && g_i4s<=1
#ifdef COLI_CUDA
           && !g_cuda_enabled
#endif
          ){
            int allq4=1;
            for(int j=0;j<nb;j++){ ESlot *e=use[j];
                if(e->g.fmt!=2||e->u.fmt!=2||e->d.fmt!=2||e->g.I!=D||e->g.O!=I||e->d.I!=I||e->d.O!=D){ allq4=0; break; } }
            if(allq4){
                double t0=now_s();
                float wj[64]; int okw=1;
                for(int j=0;j<nb;j++){ int eid=uniq[base+j]; wj[j]=0; int f=0;
                    for(int kk=0;kk<keff[0];kk++) if(idxs[kk]==eid){ wj[j]=ws[kk]; f=1; break; }
                    if(!f) okw=0; }
                if(okw){
                    int rbD=(D+1)/2, rbI=(I+1)/2;
                    int8_t *xq8=malloc((size_t)D + (size_t)nb*I);
                    float *GG=falloc((int64_t)nb*I), *UU=falloc((int64_t)nb*I), *HH=falloc((int64_t)nb*D);
                    float gsc[64];
                    if(!xq8){ fprintf(stderr,"OOM xexp scratch\n"); exit(1); }
                    int8_t *GQ=xq8+D;
                    float sx0=qrow_i8(x, xq8, D);
                    const int C1=12, C2=12;           /* chunks per expert per phase */
                    int r1=(I+C1-1)/C1, r2=(D+C2-1)/C2;
                    #pragma omp parallel
                    {
                        #pragma omp for schedule(dynamic,1)
                        for(int it=0; it<nb*C1; it++){ int j=it/C1, c0=(it%C1)*r1;
                            int c1=c0+r1<I?c0+r1:I; ESlot *e=use[j];
                            const uint8_t *qg=e->g.q4, *qu=e->u.q4;
                            float *gj=GG+(int64_t)j*I, *uj=UU+(int64_t)j*I;
                            for(int o=c0;o<c1;o++) gj[o]=(float)dot_i4i8(qg+(int64_t)o*rbD,xq8,D)*e->g.s[o]*sx0;
                            for(int o=c0;o<c1;o++) uj[o]=(float)dot_i4i8(qu+(int64_t)o*rbD,xq8,D)*e->u.s[o]*sx0;
                            for(int o=c0;o<c1;o++) gj[o]=siluf(gj[o])*uj[o];
                        }                              /* implicit barrier */
                        #pragma omp for schedule(static)
                        for(int j=0;j<nb;j++) gsc[j]=qrow_i8(GG+(int64_t)j*I, GQ+(int64_t)j*I, I);
                        #pragma omp for schedule(dynamic,1)
                        for(int it=0; it<nb*C2; it++){ int j=it/C2, c0=(it%C2)*r2;
                            int c1=c0+r2<D?c0+r2:D; ESlot *e=use[j];
                            const uint8_t *qd=e->d.q4; const int8_t *gq=GQ+(int64_t)j*I;
                            float *hj=HH+(int64_t)j*D;
                            for(int o=c0;o<c1;o++) hj[o]=(float)dot_i4i8(qd+(int64_t)o*rbI,gq,I)*e->d.s[o]*gsc[j];
                        }
                    }
                    for(int j=0;j<nb;j++){ float w=wj[j], *hj=HH+(int64_t)j*D;
                        for(int d=0;d<D;d++) out[d]+=w*hj[d]; }
                    double dt=now_s()-t0; m->t_emm+=dt;
                    if(g_prof){ m->t_ecpu+=dt; m->cpu_expert_rows+=(uint64_t)nb;
                        for(int j=0;j<nb;j++) m->cpu_expert_bytes+=qt_bytes(&use[j]->g)+qt_bytes(&use[j]->u)+qt_bytes(&use[j]->d); }
                    free(xq8); free(GG); free(UU); free(HH);
                    xexp_done=1;
                }
            }
        }
#ifdef COLI_VULKAN
        /* Vulkan expert tier (COLI_VULKAN=1): upload routed int4/int3-g64 experts to the GPU once
         * (capped by COLI_VK_EXPERTS), then compute the resident ones as one batched
         * coli_vk_expert_group (fused gate+up+silu -> down, on-device); the rest + misses
         * fall back to the CPU below. Decode-only (S<=4). Measured ~35% faster than ROCm. */
        if(vk_active && !metal_done){
            /* Pinned+async VK expert tier. Pass 1 partitions by REGISTRY residency (the
             * heat-pinned startup uploads — no uploads, no loads here); pass 2 ISSUES the
             * GPU batch async and computes the CPU share while it runs; pass 3 drains the
             * pipe + RAM-loads the GPU-side slots (cache invariants: every dispatched slot
             * waited, every use[] slot loaded before the end-of-block LRU swap); pass 4
             * takes the GPU results (fallback: recompute those rows on the CPU). */
            ColiVkTensor *vg[64],*vu[64],*vd[64]; int veid[64]; int vrows[64], voff[64], vrmap[64*4]; float vwmap[64*4];
            ColiVkTensor *vg2[64],*vu2[64],*vd2[64]; int veid2[64]; int vrows2[64], voff2[64], vrmap2[64*4]; float vwmap2[64*4];
            ESlot *ce[64]; int cnr[64], crmap[64*4]; float cwmap[64*4]; int cqof[64];
            int nvk=0, vtot=0, nvk2=0, vtot2=0, ncpu=0;
            double t0=now_s();
            for(int j=0;j<nb;j++){ int eid=uniq[base+j];
                int nr=0;
                for(int s=0;s<S;s++) for(int kk=0;kk<keff[s];kk++)
                    if(idxs[(int64_t)s*K+kk]==eid){ rows[nr]=s; rw[nr]=ws[(int64_t)s*K+kk]; nr++; break; }
                if(!nr){ if(g_pipe && qof[j]>=0){ double tw=now_s(); pipe_wait(qof[j]); m->t_ewait += now_s()-tw; } continue; }
                if(vk_hit[j]){          /* registry-served: no RAM slot, no disk load */
                    ColiVkTensor **reg=vk_reg_at(layer,eid);
                    if(vk2_on && coli_vk_tensor_dev(reg[0])==1){   /* dev2 tier expert */
                        voff2[nvk2]=vtot2;
                        for(int r=0;r<nr;r++){ memcpy(vk_xh2+(int64_t)(vtot2+r)*D, x+(int64_t)rows[r]*D, D*sizeof(float));
                            vrmap2[nvk2*S+r]=rows[r]; vwmap2[nvk2*S+r]=rw[r]; }
                        vg2[nvk2]=reg[0]; vu2[nvk2]=reg[1]; vd2[nvk2]=reg[2]; veid2[nvk2]=eid; vrows2[nvk2]=nr; vtot2+=nr; nvk2++;
                    } else {
                        voff[nvk]=vtot;
                        for(int r=0;r<nr;r++){ memcpy(vk_xh+(int64_t)(vtot+r)*D, x+(int64_t)rows[r]*D, D*sizeof(float));
                            vrmap[nvk*S+r]=rows[r]; vwmap[nvk*S+r]=rw[r]; }
                        vg[nvk]=reg[0]; vu[nvk]=reg[1]; vd[nvk]=reg[2]; veid[nvk]=eid; vrows[nvk]=nr; vtot+=nr; nvk++;
                    }
                } else {
                    ce[ncpu]=use[j]; cnr[ncpu]=nr; cqof[ncpu]=qof[j];
                    for(int r=0;r<nr;r++){ crmap[ncpu*S+r]=rows[r]; cwmap[ncpu*S+r]=rw[r]; }
                    ncpu++;
                }
            }
            if(g_prof){ g_vkb_cls+=now_s()-t0; g_vkb_blocks++; g_vkb_nvk+=nvk+nvk2; g_vkb_nvk2+=nvk2; g_vkb_ncpu+=ncpu; }
            double t_iss0=now_s();
            /* issue the SLOWER device first so it gets the longest overlap window;
             * dev2's submit runs on a worker thread (joined before take2 below) so its
             * per-block cost overlaps dev0 issue + the CPU share instead of serializing */
            Vk2Iss iss2 = { vg2, vu2, vd2, vrows2, nvk2, vk_xh2, 0, 0 };
            pthread_t iss2_th; int iss2_threaded=0;
            if(nvk2>0){
                if(pthread_create(&iss2_th,NULL,vk2_issue_worker,&iss2)==0) iss2_threaded=1;
                else iss2.rc = coli_vk_expert_group_issue2(vg2,vu2,vd2,vrows2,nvk2,vk_xh2);
            }
            int vk_issued = nvk>0 && coli_vk_expert_group_issue(vg,vu,vd,vrows,nvk,vk_xh);
            if(g_prof) g_vkb_issue+=now_s()-t_iss0;
            /* CPU share stays SERIAL over experts with row-parallel kernels: a decode
             * block leaves only ~5-6 CPU experts (the tier absorbs the hot head), fewer
             * than the OMP pool, so one-task-per-expert ran each expert single-threaded
             * at ~2.5 GB/s while cores idled — measured -23% vs this structure (A/B
             * 2026-07-20); the kernels' internal parallel-for already uses every core.
             * What IS reordered: pipe-ready and cache-resident experts run FIRST, so a
             * still-loading expert gets its I/O hidden behind their matmuls instead of
             * head-of-line blocking the block. The class is an ordering HINT only —
             * the body still pipe_waits/loads every entry, so a stale probe costs
             * nothing but order. t_ecpu counts kernel time only (the default path's
             * meaning); the waits land in t_ewait. */
            {
                uint8_t vcls[64];
                for(int c2=0;c2<ncpu;c2++){
                    if(g_pipe && cqof[c2]>=0) vcls[c2] = pipe_ready(cqof[c2]) ? 0 : 2;   /* loaded : in flight */
                    else                      vcls[c2] = ce[c2]->slab       ? 0 : 1;   /* resident : sync miss */
                }
                int vord[64], no=0;
                for(uint8_t k2=0;k2<3;k2++) for(int c2=0;c2<ncpu;c2++) if(vcls[c2]==k2) vord[no++]=c2;
                for(int oi=0;oi<no;oi++){ int c2=vord[oi]; ESlot *e=ce[c2]; int nr=cnr[c2];
                    if(g_pipe && cqof[c2]>=0){ double tw=now_s(); pipe_wait(cqof[c2]); m->t_ewait += now_s()-tw; }
                    if(!e->slab) expert_load(m,layer,e->eid,e,1,1);   /* demand=1: moe miss path (FASE A snapshot valid) */
                    for(int r=0;r<nr;r++) memcpy(xg+(int64_t)r*D, x+(int64_t)crmap[c2*S+r]*D, D*sizeof(float));
                    double te0=now_s();
                    expert_ffn(hh,gg,uu,xg,&e->g,&e->u,&e->d,nr,I);
                    for(int r=0;r<nr;r++){ float *os=out+(int64_t)crmap[c2*S+r]*D, wgt=cwmap[c2*S+r], *hr=hh+(int64_t)r*D;
                        for(int d=0;d<D;d++) os[d]+=wgt*hr[d]; }
                    if(g_prof){ m->t_ecpu+=now_s()-te0;
                        m->cpu_expert_bytes+=qt_bytes(&e->g)+qt_bytes(&e->u)+qt_bytes(&e->d);
                        m->cpu_expert_rows+=(uint64_t)nr; }
                }
            }
            double t_take0=now_s();
            int vk_ok = vk_issued && coli_vk_expert_group_take(vk_yh);
            if(g_prof) m->t_egpu+=now_s()-t_take0;
            for(int c2=0;c2<nvk;c2++){ int nr=vrows[c2];
                if(vk_ok){ int o=voff[c2];
                    for(int r=0;r<nr;r++){ float *os=out+(int64_t)vrmap[c2*S+r]*D, wgt=vwmap[c2*S+r], *src=vk_yh+(int64_t)(o+r)*D;
                        for(int d=0;d<D;d++) os[d]+=wgt*src[d]; }
                } else {   /* issue/take failed (device lost): load + recompute on the CPU */
                    ESlot *e=&m->ws[nmiss<63?nmiss:63];
                    if(e->eid!=veid[c2] || !e->slab) expert_load(m,layer,veid[c2],e,1,0);   /* device-lost recovery: DISK-CLASS leaves it unclassified */
                    for(int r=0;r<nr;r++) memcpy(xg+(int64_t)r*D, x+(int64_t)vrmap[c2*S+r]*D, D*sizeof(float));
                    expert_ffn(hh,gg,uu,xg,&e->g,&e->u,&e->d,nr,I);
                    for(int r=0;r<nr;r++){ float *os=out+(int64_t)vrmap[c2*S+r]*D, wgt=vwmap[c2*S+r], *hr=hh+(int64_t)r*D;
                        for(int d=0;d<D;d++) os[d]+=wgt*hr[d]; }
                }
            }
            /* dev2 group: taken AFTER dev0's accumulate so the slower card gets the
             * extra overlap; same per-expert CPU recompute fallback on failure. */
            if(iss2_threaded){ double t_j0=now_s(); pthread_join(iss2_th,NULL);
                if(g_prof){ g_vkb_join+=now_s()-t_j0; g_vkb_wrk+=iss2.dt; } }
            int vk2_ok = iss2.rc && coli_vk_expert_group_take2(vk_yh2);
            for(int c2=0;c2<nvk2;c2++){ int nr=vrows2[c2];
                if(vk2_ok){ int o=voff2[c2];
                    for(int r=0;r<nr;r++){ float *os=out+(int64_t)vrmap2[c2*S+r]*D, wgt=vwmap2[c2*S+r], *src=vk_yh2+(int64_t)(o+r)*D;
                        for(int d=0;d<D;d++) os[d]+=wgt*src[d]; }
                } else {
                    ESlot *e=&m->ws[nmiss<63?nmiss:63];
                    if(e->eid!=veid2[c2] || !e->slab) expert_load(m,layer,veid2[c2],e,1,0);
                    for(int r=0;r<nr;r++) memcpy(xg+(int64_t)r*D, x+(int64_t)vrmap2[c2*S+r]*D, D*sizeof(float));
                    expert_ffn(hh,gg,uu,xg,&e->g,&e->u,&e->d,nr,I);
                    for(int r=0;r<nr;r++){ float *os=out+(int64_t)vrmap2[c2*S+r]*D, wgt=vwmap2[c2*S+r], *hr=hh+(int64_t)r*D;
                        for(int d=0;d<D;d++) os[d]+=wgt*hr[d]; }
                }
            }
            double dt=now_s()-t0; m->t_emm+=dt;
            if(g_prof) g_vkb_acc+=now_s()-t_take0;   /* take-wait (t_egpu) + result accumulate */
        }
#endif
        if(!metal_done && !xexp_done && !vk_active)
        for(int j=0;j<nb;j++){ int eid=uniq[base+j]; ESlot *e=use[j];
#ifdef COLI_CUDA
            if(early_issued && done_j[j]) continue;    /* computing on the GPU right now */
#endif
            /* Drain this miss's async load BEFORE the nr==0 early-exit below: every
             * dispatched slot must be waited before the end-of-block LRU swap can reuse
             * its ws[] slab, so correctness does not depend on the nr>=1 routing invariant.
             * Stays ABOVE the METAL skip: a subset that fell back to the CPU still needs its
             * slot drained here, and under METAL the block-level drain above already ran (this
             * spin is then a no-op). */
            if(g_pipe && qof[j]>=0){ double tw=now_s(); pipe_wait(qof[j]); m->t_ewait += now_s()-tw; }
#ifdef COLI_METAL
            /* skip the subsets already computed on GPU */
            if(g_metal_enabled && ((is_miss[j] && !cpu_miss) || (!is_miss[j] && !cpu_res))) continue;
#endif
            int nr=0;                                 /* righe (posizioni) che usano questo expert */
            for(int s=0;s<S;s++) for(int kk=0;kk<keff[s];kk++)
                if(idxs[(int64_t)s*K+kk]==eid){ rows[nr]=s; rw[nr]=ws[(int64_t)s*K+kk]; nr++; break; }
            if(!nr) continue;
            /* CAUSAL ABLATION (FASE C, contribution mode): this expert's routing,
             * counters, and any route trace were already recorded in FASE A; its
             * weighted output is now replaced by ZERO -> skip its matmul+accumulate.
             * CPU path only (the ablation harness runs the CPU forward). */
            if(g_abl.mode==1 && abl_zero_contrib(&g_abl, layer, eid)) continue;
#ifdef COLI_CUDA
            if(g_cuda_enabled && e->g.cuda_eligible) m->gpu_expert_calls++;
            if(group_enabled && g_cuda_enabled && e->g.cuda_eligible && e->u.cuda_eligible && e->d.cuda_eligible &&
               !omp_in_parallel()){
                group_e[ngroup]=e; group_n[ngroup]=nr;
                for(int r=0;r<nr;r++){ group_row[(int64_t)ngroup*S+r]=rows[r]; group_weight[(int64_t)ngroup*S+r]=rw[r]; }
                ngroup++; continue;
            }
#endif
            const float *xsrc=E8_XE(e);
            for(int r=0;r<nr;r++) memcpy(xg+(int64_t)r*D, xsrc+(int64_t)rows[r]*D, D*sizeof(float));
            double t0=now_s();
#ifdef COLI_CUDA
            if(!group_enabled && g_cuda_enabled && e->g.cuda_eligible && e->u.cuda_eligible &&
               e->d.cuda_eligible && !omp_in_parallel() &&
               coli_cuda_expert_mlp(e->g.cuda,e->u.cuda,e->d.cuda,hh,xg,nr)){
                for(int r=0;r<nr;r++){ float *os=out+(int64_t)rows[r]*D,wgt=rw[r],*hr=hh+(int64_t)r*D;
                    for(int d=0;d<D;d++) os[d]+=wgt*hr[d]; }
                double dt=now_s()-t0;m->t_emm+=dt;if(g_prof)m->t_egpu+=dt;continue;
            }
            if(!e->slab) expert_host_ensure(m,layer,e);
#endif
            /* quantizzazione sollevata: raccogli le righe int8 accanto al gather f32 e
             * pubblica il contesto per il ramo IDOT di matmul_qt_ex. Solo quando l'input
             * e' x originale (mai la copia ruotata fmt=6) e gate/up lo consumerebbero.
             * EN: hoisted quantization — gather the int8 rows next to the f32 gather and
             * publish the context for matmul_qt_ex's IDOT branch. Only when the input is
             * the original x (never the fmt=6 rotated copy) and gate/up would take it. */
            g_pq.x=NULL;
            if(xsrc==x && pq_want(&e->g,&e->u,nr)){
                if(!pq_ready){
                    if(!xq_all){
                        xq_all=xalloc((size_t)S*D,"moe xq"); sx_all=xalloc((size_t)S*sizeof(float),"moe sx");
                        xqg   =xalloc((size_t)S*D,"moe xqg"); sxg  =xalloc((size_t)S*sizeof(float),"moe sxg");
                        xsum_all=xalloc((size_t)S*sizeof(int32_t),"moe xsum");
                        xsumg   =xalloc((size_t)S*sizeof(int32_t),"moe xsumg");
                    }
                    pq_build(x,S,D,xq_all,sx_all); pq_build_xsum(xq_all,S,D,xsum_all); pq_ready=1;
                }
                for(int r=0;r<nr;r++){ memcpy(xqg+(int64_t)r*D, xq_all+(int64_t)rows[r]*D,(size_t)D); sxg[r]=sx_all[rows[r]]; xsumg[r]=xsum_all[rows[r]]; }
                g_pq.x=xg; g_pq.S=nr; g_pq.I=D; g_pq.xq=xqg; g_pq.sx=sxg; g_pq.xsum=xsumg;
            }
            expert_ffn(hh,gg,uu,xg,&e->g,&e->u,&e->d,nr,I);
            for(int r=0;r<nr;r++){ float *os=out+(int64_t)rows[r]*D, wgt=rw[r], *hr=hh+(int64_t)r*D;
                for(int d=0;d<D;d++) os[d]+=wgt*hr[d]; }
            double dt=now_s()-t0;m->t_emm+=dt;if(g_prof){m->t_ecpu+=dt;
                m->cpu_expert_bytes+=qt_bytes(&e->g)+qt_bytes(&e->u)+qt_bytes(&e->d);
                m->cpu_expert_rows+=(uint64_t)nr;}
        }
#ifdef COLI_CUDA
        /* Inc.4 take phase: the CPU loop above ran while the GPU computed the issued
         * groups — collect them now. A failed device recomputes its experts on the CPU
         * (expert_host_ensure reloads slabs released by CUDA_RELEASE_HOST). */
        if(early_issued){
            double tg1=now_s();
            g_ovl_cpu+=tg1-g_ovl_mark;               /* CPU-row window between issue and take */
            for(int di=0;di<g_cuda_ndev;di++) if(dev_nc0[di]){
                const float *hy=coli_cuda_expert_group_take(g_cuda_devices[di]);
                for(int q=0;q<dev_nc0[di];q++) eslot_release(eg_e[dev_which0[di][q]]);
                int cur=0;
                for(int q=0;q<dev_nc0[di];q++){
                    int gi=dev_which0[di][q], nr=eg_n[gi];
                    if(hy){
                        for(int r=0;r<nr;r++){ float *os=out+(int64_t)eg_row[gi][r]*D; float wgt=eg_w[gi][r];
                            const float *hr=hy+(int64_t)(cur+r)*D;
                            for(int d=0;d<D;d++) os[d]+=wgt*hr[d]; }
                    } else {
                        ESlot *e=eg_e[gi];
                        for(int r=0;r<nr;r++) memcpy(xg+(int64_t)r*D,x+(int64_t)eg_row[gi][r]*D,D*sizeof(float));
                        expert_host_ensure(m,layer,e);
                        expert_ffn(hh,gg,uu,xg,&e->g,&e->u,&e->d,nr,I);
                        for(int r=0;r<nr;r++){ float *os=out+(int64_t)eg_row[gi][r]*D; float wgt=eg_w[gi][r];
                            for(int d=0;d<D;d++) os[d]+=wgt*hh[(int64_t)r*D+d]; }
                    }
                    cur+=nr;
                }
            }
            m->t_emm+=now_s()-tg1; g_ovl_take+=now_s()-tg1;
            if(g_prof&&early_issue_start>0) m->t_egpu+=now_s()-early_issue_start;
        }
        ColiCudaTensor *dev_g[COLI_CUDA_MAX_DEVICES][64],*dev_u[COLI_CUDA_MAX_DEVICES][64];
        ColiCudaTensor *dev_d[COLI_CUDA_MAX_DEVICES][64];
        int dev_rows[COLI_CUDA_MAX_DEVICES][64],dev_which[COLI_CUDA_MAX_DEVICES][64];
        int dev_nc[COLI_CUDA_MAX_DEVICES]={0},dev_total[COLI_CUDA_MAX_DEVICES]={0};
        int dev_off[COLI_CUDA_MAX_DEVICES]={0},dev_ok[COLI_CUDA_MAX_DEVICES]={0};
        double dev_time[COLI_CUDA_MAX_DEVICES]={0};
        for(int di=0;di<g_cuda_ndev;di++) for(int q=0;q<ngroup;q++)
            if(group_e[q]->g.cuda_device==g_cuda_devices[di]) dev_total[di]+=group_n[q];
        for(int di=1;di<g_cuda_ndev;di++) dev_off[di]=dev_off[di-1]+dev_total[di-1];
        for(int di=0;di<g_cuda_ndev;di++){
            int cursor=0,device=g_cuda_devices[di];
            for(int q=0;q<ngroup;q++) if(group_e[q]->g.cuda_device==device){
                int nc=dev_nc[di]++; ESlot *e=group_e[q];
                dev_g[di][nc]=e->g.cuda; dev_u[di][nc]=e->u.cuda; dev_d[di][nc]=e->d.cuda;
                dev_rows[di][nc]=group_n[q]; dev_which[di][nc]=q;
                const float *xsrc=E8_XE(e);                  /* fmt=6 feeds the rotated copy */
                for(int r=0;r<group_n[q];r++) memcpy(group_x+(int64_t)(dev_off[di]+cursor+r)*D,
                    xsrc+(int64_t)group_row[(int64_t)q*S+r]*D,D*sizeof(float));
                cursor+=group_n[q];
            }
        }
        double tg=now_s();
        /* Inc.4: at decode scale, issue every device's group WITHOUT syncing, then take
         * them all — one stream sync per device per layer instead of a full staged
         * round-trip per call (measured: ~70% of the sync call is host-side wait).
         * Any issue failure drains what was issued and the whole layer falls back to
         * the sync path below, which recomputes from group_x (idempotent). */
        int async_done=0;
        static int g_group_async=-1;
        if(g_group_async<0) g_group_async=getenv("COLI_GROUP_ASYNC")?atoi(getenv("COLI_GROUP_ASYNC")):0;
        if(g_group_async && S<=4 && g_cuda_ndev>0){
            int issued[COLI_CUDA_MAX_DEVICES]={0}, all=1;
            for(int di=0;di<g_cuda_ndev && all;di++) if(dev_nc[di]){
                ESlot *refs[64]; for(int q=0;q<dev_nc[di];q++) refs[q]=group_e[dev_which[di][q]];
                eslots_acquire(refs,dev_nc[di]);
                all=issued[di]=coli_cuda_expert_group_issue(dev_g[di],dev_u[di],dev_d[di],
                        dev_rows[di],dev_nc[di],group_x+(int64_t)dev_off[di]*D);
                if(!issued[di]) eslots_release(refs,dev_nc[di]);
            }
            if(all){
                static int announced;
                if(!announced){ announced=1; fprintf(stderr,"[CUDA] expert group async path active\n"); }
                async_done=1;
                for(int di=0;di<g_cuda_ndev;di++) if(dev_nc[di]){
                    const float *hy=coli_cuda_expert_group_take(g_cuda_devices[di]);
                    for(int q=0;q<dev_nc[di];q++) eslot_release(group_e[dev_which[di][q]]);
                    if(hy){ dev_ok[di]=1;
                        memcpy(group_y+(int64_t)dev_off[di]*D,hy,(size_t)dev_total[di]*D*sizeof(float)); }
                    else dev_ok[di]=0;      /* per-device sync failure: CPU fallback below */
                }
                if(g_prof) m->t_egpu+=now_s()-tg;
            } else for(int di=0;di<g_cuda_ndev;di++) if(issued[di]){
                coli_cuda_expert_group_take(g_cuda_devices[di]);
                for(int q=0;q<dev_nc[di];q++) eslot_release(group_e[dev_which[di][q]]);
            }
        }
        if(!async_done){
            /* resident path (#431 PR-C0): issue the group on its device with the input
             * P2P'd from the home device and the weighted partial pushed back there —
             * no host bytes, no per-device sync; take() runs after moe() returns.
             * Issue is serial (it only queues async work); failure falls back to the
             * synchronous host-round-trip call below, per device. */
            if(g_pres_home>=0 && S==1){
                for(int di=0;di<g_cuda_ndev;di++) if(dev_nc[di]){
                    float wbuf[64];
                    for(int q=0;q<dev_nc[di];q++) wbuf[q]=group_weight[(int64_t)dev_which[di][q]*S];
                    ESlot *refs[64]; for(int q=0;q<dev_nc[di];q++) refs[q]=group_e[dev_which[di][q]];
                    eslots_acquire(refs,dev_nc[di]);
                    if(coli_cuda_expert_group_resident_issue(dev_g[di],dev_u[di],dev_d[di],wbuf,dev_nc[di],
                            g_pres_home,g_pres_xsrc,g_pres_slots+(int64_t)g_pres_nused*D)){
                        g_pres_used[g_pres_nused++]=g_cuda_devices[di];
                        for(int q=0;q<dev_nc[di];q++) g_pres_refs[g_pres_nrefs++]=refs[q];
                        dev_ok[di]=2;                       /* handled on device: skip host collection */
                    } else eslots_release(refs,dev_nc[di]);
                }
            }
        }
        /* dev_ok==0 is the real "needs the synchronous host round-trip" condition:
         * never attempted, or a per-device async take() that failed. The old guard
         * was !=2, which let dev_ok==1 through — a device whose ASYNC group had
         * SUCCEEDED was recomputed here and overwrote group_y with identical values.
         * Output stayed correct, so it went unnoticed; the symptom was simply that
         * COLI_GROUP_ASYNC never showed a speedup while doubling GPU expert work. */
        #pragma omp parallel for if(g_cuda_ndev>1) schedule(static)
        for(int di=0;di<g_cuda_ndev;di++) if(dev_nc[di]&&dev_ok[di]==0){
            double td=g_prof?now_s():0;
            dev_ok[di]=coli_cuda_expert_group_pinned(dev_g[di],dev_u[di],dev_d[di],
                dev_rows[di],dev_nc[di],group_y+(int64_t)dev_off[di]*D,
                group_x+(int64_t)dev_off[di]*D,spec_pinned());
            if(g_prof)dev_time[di]=now_s()-td;
        }
        for(int di=0;di<g_cuda_ndev;di++){
            if(dev_ok[di]==2) continue;               /* results live on the home device (#431 PR-C0) */
            int off=dev_off[di];
            for(int q=0;q<dev_nc[di];q++){
                int gi=dev_which[di][q],nr=group_n[gi]; ESlot *e=group_e[gi];
                if(!dev_ok[di]){
                    const float *xsrc=E8_XE(e);          /* fmt=6 feeds the rotated copy */
                    for(int r=0;r<nr;r++) memcpy(xg+(int64_t)r*D,xsrc+(int64_t)group_row[(int64_t)gi*S+r]*D,D*sizeof(float));
                    double tc=g_prof?now_s():0;
                    if(!coli_cuda_expert_mlp(e->g.cuda,e->u.cuda,e->d.cuda,hh,xg,nr)){
                        expert_host_ensure(m,layer,e);
                        expert_ffn(hh,gg,uu,xg,&e->g,&e->u,&e->d,nr,I);
                        if(g_prof){m->cpu_expert_bytes+=qt_bytes(&e->g)+qt_bytes(&e->u)+qt_bytes(&e->d);
                            m->cpu_expert_rows+=(uint64_t)nr;}
                    }
                    if(g_prof)m->t_ecpu+=now_s()-tc;
                }
                float *src=dev_ok[di]?group_y+(int64_t)off*D:hh;
                for(int r=0;r<nr;r++){ float *os=out+(int64_t)group_row[(int64_t)gi*S+r]*D,wgt=group_weight[(int64_t)gi*S+r];
                    for(int d=0;d<D;d++) os[d]+=wgt*src[(int64_t)r*D+d]; }
                off+=nr;
            }
        }
        if(g_prof){double mx=0;for(int di=0;di<g_cuda_ndev;di++)if(dev_time[di]>mx)mx=dev_time[di];m->t_egpu+=mx;}
        m->t_emm+=now_s()-tg;
#endif
        /* No drain barrier: the per-expert pipe_wait(qof[j]) above (issued for every
         * dispatched miss slot, before the nr==0 skip) already waited on all ws[] loads
         * for this block, so they are complete before the LRU swap — and the gen-tagged
         * cursor keeps any still-spinning worker off a wrong-generation slot. */
        { ESlot *Sl=m->ecache[layer]; int *nn=&m->ecn[layer];   /* promozione LRU (swap buffer) */
          int promo = nmiss<m->ecap ? nmiss : m->ecap;
          for(int a=0;a<promo;a++){ int q=nmiss-1-a; ESlot *dst;
              if(*nn<m->ecap) dst=&Sl[(*nn)++];
              else { int lru=eslot_lru_victim(Sl,*nn,m->ecap);
                     if(lru<0){ static int warned;
                         if(!warned){ warned=1; fprintf(stderr,"[CUDA] no reusable LRU expert slot (in flight or cap reached); skipping cache promotion\n"); }
                         continue; }
                     dst=&Sl[lru]; }
              ecache_unindex(m,layer,dst);
              ESlot tmp=*dst; *dst=m->ws[q]; m->ws[q]=tmp;
              ecache_publish(m,layer,dst,dst->eid);
              dst->used=(uint64_t)__atomic_add_fetch(&m->eclock,1,__ATOMIC_RELAXED); }
        }
    }
    /* ---- FASE E: shared expert (PIPE2: gia' sul device; Metal CB: gia' sommata) ---- */
    if(!with_shared) goto shared_done;
    {
    float *sg=NULL,*su=NULL;int shared_cuda=0;
#ifdef COLI_METAL
    if(g_pre_sh){ for(int64_t z=0;z<(int64_t)S*D;z++) out[z]+=g_pre_sh[z]; shared_on_gpu=1; }
    if(shared_on_gpu) shared_cuda=2;             /* gia' sommato in out: salta calcolo e add */
#endif
#ifdef COLI_CUDA
    int shared_min=getenv("COLI_CUDA_SHARED_W4A16_MIN_ROWS")?
        atoi(getenv("COLI_CUDA_SHARED_W4A16_MIN_ROWS")):32;
    if(shared_min<16)shared_min=16;
    if(shared_cuda==0&&!spec_pinned()&&S>=shared_min&&!l->shared_w4a16_failed&&!omp_in_parallel()&&g_cuda_enabled&&
       l->sh_gate.fmt==2&&l->sh_up.fmt==2&&l->sh_down.fmt==2&&
       getenv("COLI_CUDA_SHARED_W4A16")&&atoi(getenv("COLI_CUDA_SHARED_W4A16"))&&
       qt_cuda_upload(&l->sh_gate)&&qt_cuda_upload(&l->sh_up)&&qt_cuda_upload(&l->sh_down)){
        shared_cuda=coli_cuda_shared_mlp_w4a16(l->sh_gate.cuda,l->sh_up.cuda,
                                               l->sh_down.cuda,hh,x,S);
        if(!shared_cuda)l->shared_w4a16_failed=1;
    }
#endif
    if(!shared_cuda){
#ifdef COLI_VULKAN
        /* Whole shared expert as ONE fused submit (gate+up+silu -> down, hidden on-device)
         * via the expert-group primitive with count=1 — replaces 3 separate VK matmuls
         * (x read once, 1 fence instead of 3). Falls through to the per-matmul chain. */
        int fsh=l->sh_gate.fmt;
        if(g_vk_dense && !omp_in_parallel() && (fsh==1||fsh==2||fsh==5) &&
           l->sh_up.fmt==fsh && l->sh_down.fmt==fsh){
            #define SW_(t) ((t).fmt==1?(const void*)(t).q8:(const void*)(t).q4)
            if(coli_vk_tensor_ensure(&l->sh_gate.vk,SW_(l->sh_gate),l->sh_gate.s,fsh,D,sI,l->sh_gate.gs)&&
               coli_vk_tensor_ensure(&l->sh_up.vk,  SW_(l->sh_up),  l->sh_up.s,  fsh,D,sI,l->sh_up.gs)&&
               coli_vk_tensor_ensure(&l->sh_down.vk,SW_(l->sh_down),l->sh_down.s,fsh,sI,D,l->sh_down.gs)){
                ColiVkTensor *vg=l->sh_gate.vk,*vu=l->sh_up.vk,*vd=l->sh_down.vk;
                int rows1[1]={S};
                if(coli_vk_expert_group(&vg,&vu,&vd,rows1,1,hh,x)) shared_cuda=1;
            }
            #undef SW_
        }
        if(!shared_cuda){
#endif
        sg=falloc((int64_t)S*sI);su=falloc((int64_t)S*sI);
#ifdef COLI_VULKAN
        if(!vk_matmul_qt(&l->sh_gate, sg, x, S))
#endif
        matmul_qt(sg, x, &l->sh_gate, S);
#ifdef COLI_VULKAN
        if(!vk_matmul_qt(&l->sh_up, su, x, S))
#endif
        matmul_qt(su, x, &l->sh_up,   S);
        for(int64_t z=0;z<(int64_t)S*sI;z++) sg[z]=siluf(sg[z])*su[z];
#ifdef COLI_VULKAN
        if(!vk_matmul_qt(&l->sh_down, hh, sg, S))
#endif
        matmul_qt(hh, sg, &l->sh_down, S);
#ifdef COLI_VULKAN
        }
#endif
    }
    if(shared_cuda!=2) for(int64_t z=0;z<(int64_t)S*D;z++) out[z]+=hh[z];
    free(sg); free(su);
    }
shared_done:
    free(logits_all); free(choice); free(idxs); free(ws); free(keff); free(uniq);
    g_pq.x=NULL;   /* xg sta per essere liberato: nessun contesto deve sopravvivergli
                    * EN: xg is about to be freed — no context may outlive it */
    free(xq_all); free(sx_all); free(xqg); free(sxg); free(xsum_all); free(xsumg);
    free(xg); free(gg); free(uu); free(hh); free(rows); free(rw); free(xe);
    #undef E8_XE
#ifdef COLI_CUDA
    free(group_x);free(group_y);
    free(group_row); free(group_weight);
#endif
#ifdef COLI_VULKAN
    free(vk_xh); free(vk_yh);
#endif
}

static void dense_mlp(Layer *l, float *x, int S, int D, int I, float *out){
    float *g=falloc((int64_t)S*I), *u=falloc((int64_t)S*I);
    matmul_qt(g, x, &l->gate_proj, S);
    matmul_qt(u, x, &l->up_proj,   S);
    for(int64_t i=0;i<(int64_t)S*I;i++) g[i]=siluf(g[i])*u[i];
    matmul_qt(out, g, &l->down_proj, S);
    free(g); free(u);
}

/* LOOKA: predice il top-K del router del layer `target` dallo stato h (residual stream),
 * usando la STESSA pipeline del routing vero (post_ln -> router -> sigmoid+bias, top-K).
 * kind 0 = stesso layer saltando l'attention
 * kind 1 = layer successivo (PILOT: stale state, 75.8% recall)
 * kind 2 = two-step: approximate L's shared expert output, add to state, THEN predict L+1.
 *   The shared expert is resident (part of dense model), so this adds 3 small matmuls
 *   but no disk I/O. The corrected state includes the dominant part of MoE(L) that the
 *   stale PILOT prediction is missing. */
static void la_predict(Model *m, int target, const float *h, int kind){
    Cfg *c=&m->c; Layer *l=&m->L[target]; int D=c->hidden, E=c->n_experts, K=c->topk;
    float *nrm=falloc(D), *ch=falloc(E);

    if(kind==2){
        /* Two-step: h is L's post-attention state (pre-MoE). We want to predict L+1's
         * routing. The real L+1 router sees h + MoE(L). We approximate MoE(L) by
         * computing ONLY the shared expert (resident, no disk) on the post_ln-normalized
         * state, then add it to h before running L+1's router.
         *
         * target = L+1, so the layer we need the shared expert from is L = target-1. */
        int src_layer = target - 1;
        if(src_layer < 0 || src_layer >= c->n_layers || !m->L[src_layer].sparse
           || c->n_shared <= 0 || c->moe_inter <= 0){
            la_val[2][target] = 0; free(nrm); free(ch); return;
        }
        Layer *sl = &m->L[src_layer];
        int sI = c->moe_inter * c->n_shared;
        float *snrm = falloc(D), *sg = falloc(sI), *su = falloc(sI);
        float *sout = falloc(D), *hc = falloc(D);
        rmsnorm(snrm, h, sl->post_ln, D, c->eps);
        matmul_qt(sg, snrm, &sl->sh_gate, 1);
        matmul_qt(su, snrm, &sl->sh_up,   1);
        for(int i=0;i<sI;i++) sg[i] = siluf(sg[i]) * su[i];
        matmul_qt(sout, sg, &sl->sh_down, 1);
        for(int i=0;i<D;i++) hc[i] = h[i] + sout[i];
        rmsnorm(nrm, hc, l->post_ln, D, c->eps);
        free(snrm); free(sg); free(su); free(sout); free(hc);
        matmul(ch, nrm, l->router, 1, D, E);
        for(int e=0;e<E;e++) ch[e] = sigmoidf(ch[e]) + l->router_bias[e];
        int *pred = la_pred[2][target];
        for(int kk=0;kk<K;kk++){ int best=-1; float bv=-1e30f;
            for(int e=0;e<E;e++){ int tk=0; for(int j=0;j<kk;j++) if(pred[j]==e){tk=1;break;}
                if(!tk && ch[e]>bv){bv=ch[e];best=e;} }
            pred[kk]=best; }
        la_val[2][target]=1;
        free(nrm); free(ch);
        return;
    }

    /* Baseline kinds 0 and 1: pure router on the given state */
    rmsnorm(nrm,h,l->post_ln,D,c->eps);
    matmul(ch,nrm,l->router,1,D,E);
    for(int e=0;e<E;e++) ch[e]=sigmoidf(ch[e])+l->router_bias[e];
    int *pred=la_pred[kind][target];
    for(int kk=0;kk<K;kk++){ int best=-1; float bv=-1e30f;
        for(int e=0;e<E;e++){ int tk=0; for(int j=0;j<kk;j++) if(pred[j]==e){tk=1;break;}
            if(!tk && ch[e]>bv){bv=ch[e];best=e;} }
        pred[kk]=best; }
    la_val[kind][target]=1;
    free(nrm); free(ch);
}

/* PILOTA: prefetch guidato dal router. Predice il top-K del layer L+1 dallo stato
 * post-attention di L (recall misurato 71.6% su GLM-5.2, vs 41.3% del token precedente)
 * e lancia il WILLNEED degli expert mancanti MENTRE il MoE di L legge i suoi: il disco
 * lavora nei tempi morti del calcolo invece di aspettare il routing vero. Con MTP attiva
 * predice per TUTTE le posizioni del draft: la speculazione pilota anche l'I/O.
 * PILOT_K limita alle prime k predizioni (la testa del ranking e' piu' affidabile
 * della coda: meno banda sprecata sulle predizioni sbagliate).
 *
 * I WILLNEED partono da un THREAD I/O dedicato: con la coda disco satura la submit
 * del fadvise BLOCCA (~0.5ms x 169k chiamate = +92s/48 token, misurato) — inline
 * il pilota costava piu' di quanto rendesse. Ring lock-free 1P/1C; pieno = scarta
 * (un hint perso non e' un errore). */
static struct { _Atomic int l,e; } pilot_q[4096];  /* payload atomico (relaxed): la claim SPMC legge speculativamente prima della CAS e scarta se perde -> senza _Atomic sarebbe una data race C11 col produttore. int e' sempre lock-free: stessa size/align. */
static volatile unsigned pilot_w=0, pilot_r=0;
static Model *pilot_m=NULL;
/* PILOT_REAL: load VERO dell'expert predetto dentro la LRU del layer FUTURO. Vedi
 * l'invariante di sicurezza accanto a g_pilot_real. Il pread (lento) gira FUORI dal lock;
 * il lock protegge solo la scelta/pubblicazione dello slot e l'handshake col main. */
static void pilot_realload(Model *m, int layer, int eid){
    pthread_mutex_lock(&g_pilot_mx);
    if(layer<0 || layer>=256 || layer <= atomic_load_explicit(&g_cur_moe_layer,memory_order_acquire)){
        atomic_fetch_add_explicit(&g_pilot_drops,1,memory_order_relaxed);   /* fuori range (come il ramo URING) o main gia' su questo layer */
        pthread_mutex_unlock(&g_pilot_mx); return;
    }
    /* gia' residente (pin o ecache), oppure gia' prenotato? */
#ifdef COLI_VULKAN
    if(vk_reg_served(layer,eid)){ pthread_mutex_unlock(&g_pilot_mx); return; }   /* VK-tier-served: no load */
#endif
    if(pin_indexed(m,layer,eid)){ pthread_mutex_unlock(&g_pilot_mx); return; }
    ESlot *Sl=m->ecache[layer]; int nn=m->ecn[layer];   /* dedup contro residenti E prenotazioni in volo -(eid+2) */
    if(ecache_indexed(m,layer,eid,1)){ pthread_mutex_unlock(&g_pilot_mx); return; }
    /* SPMC (PILOT_WORKERS>1): scegli lo slot sotto lock e MARCALO prenotato prima di
     * rilasciarlo, cosi' gli altri worker non lo scelgono come vittima ne' ricaricano
     * lo stesso eid. Stesso schema del ramo URING (prenotazione visibile -(eid+2),
     * ecn bumpato subito, scan-vittima che salta le prenotazioni). */
    int slot,isnew=0;
    if(nn<m->ecap){ slot=nn; isnew=1; m->ecn[layer]=nn+1; }   /* cresci: pubblica subito lo slot (marcato prenotato) */
    else {
        slot=eslot_lru_victim(Sl,nn,m->ecap);           /* riusa libero-con-slab, poi LRU; slot svuotati solo sotto ecap (#1034) */
        if(slot<0){ atomic_fetch_add_explicit(&g_pilot_drops,1,memory_order_relaxed);
                    pthread_mutex_unlock(&g_pilot_mx); return; }   /* tutti in volo, o cap raggiunto */
        /* LFRU eviction guard (#441, narrowed by #497 — folded into the SPMC selection):
         * protect the victim only when genuinely WARM (>=2 demand accesses) AND clearly
         * hotter than the speculation by tier_pick_lfru's 25%+4-freq hysteresis; the
         * un-narrowed #474 test dropped ~all speculations on a full cache (#490).
         * Skips free slots (eid==-1). Cache placement only -> output unchanged. */
        if(g_pilot_evict_guard && m->eheat && m->elast && Sl[slot].eid>=0){
            int vid=Sl[slot].eid; uint32_t vh=m->eheat[layer][vid];
            if(vh>=2){
                uint64_t vs=tier_lfru_score(vh,m->elast[layer][vid],m->eaccess_clock);
                uint64_t cs=tier_lfru_score(m->eheat[layer][eid],m->elast[layer][eid],m->eaccess_clock);
                if(vs+(vs>>2)+(4u<<8)>cs){ atomic_fetch_add_explicit(&g_pilot_drops,1,memory_order_relaxed);
                                            pthread_mutex_unlock(&g_pilot_mx); return; } } }
    }
    ESlot *dst=&Sl[slot];
    ecache_reserve(m,layer,dst,eid);                    /* prenotazione VISIBILE: dedup + vittima degli altri worker la vedono */
    (void)isnew;
    dst->used=(uint64_t)-1;                             /* sentinella "in carica": mai vittima LRU finche' expert_load non pubblica l'eid reale
                                                         * (chiude la finestra eid-reale/used-vecchio: uno snapshot di eclock si sarebbe potuto
                                                         * far superare da altri load durante il pread). used fresco ristampato al successo. */
    g_pilot_inflight[layer]++;
    pthread_mutex_unlock(&g_pilot_mx);

    int rc=expert_load(m,layer,eid,dst,0,0);            /* pread VERO — fuori dal lock, concorrente fra worker (come i PIPE demand); fatal=0: una speculazione fallita NON uccide il server; demand=0: speculative, never classified. Al successo expert_load setta dst->eid=eid. */

    pthread_mutex_lock(&g_pilot_mx);
    if(rc==0){
        ecache_publish(m,layer,dst,eid);
        dst->used=(uint64_t)__atomic_add_fetch(&m->eclock,1,__ATOMIC_RELAXED);  /* eid gia' reale (expert_load); timbra used fresco */
        atomic_fetch_add_explicit(&g_pilot_loads,1,memory_order_relaxed);
    } else {
        /* load fallito: libera la prenotazione. LO SLOT VA ANCHE RIPORTATO A used=0:
         * la prenotazione lo aveva marcato used=(uint64_t)-1 ("in carica", mai vittima),
         * e lasciarlo cosi' lo rendeva invisibile agli hit ma ULTIMO in ogni scan LRU —
         * mai evictable: ogni speculazione fallita (disco lento, errore I/O transitorio)
         * sottraeva ~19MB di cache in modo permanente e silenzioso. used=0 e' la stessa
         * convenzione "slot vergine" di rss_guard: diventa la PRIMA vittima, non l'ultima. */
        ecache_hide(m,layer,dst);
        dst->used=0;
        atomic_fetch_add_explicit(&g_pilot_drops,1,memory_order_relaxed);
    }
    g_pilot_inflight[layer]--;
    pthread_cond_broadcast(&g_pilot_cv);
    pthread_mutex_unlock(&g_pilot_mx);
    if(rc!=0)                                            /* mai swallow silenzioso: logga (una riga) e prosegui */
        fprintf(stderr,"[PILOT] load speculativo abbandonato: layer %d expert %d (I/O error/short read) — nessun impatto sull'output\n",layer,eid);
}
#ifdef __linux__
typedef struct { int layer,eid,li; ESlot *dst; } PilotUringDone;
static void pilot_uring_batch(Model *m){
    PilotUringDone done[URING_LOAD_MAX]; int nd=0;
    uring_batch_reset(&g_ub_pilot);
    unsigned r=__atomic_load_n(&pilot_r,__ATOMIC_ACQUIRE);
    unsigned w=__atomic_load_n(&pilot_w,__ATOMIC_ACQUIRE);
    while(r!=w && nd<URING_LOAD_MAX){
        int layer=atomic_load_explicit(&pilot_q[r&4095].l,memory_order_relaxed),eid=atomic_load_explicit(&pilot_q[r&4095].e,memory_order_relaxed); r++;
        if(layer<0 || layer>=256){ atomic_fetch_add_explicit(&g_pilot_drops,1,memory_order_relaxed); continue; }
        pthread_mutex_lock(&g_pilot_mx);
        if(layer<=atomic_load_explicit(&g_cur_moe_layer,memory_order_acquire)){
            atomic_fetch_add_explicit(&g_pilot_drops,1,memory_order_relaxed);
            pthread_mutex_unlock(&g_pilot_mx); continue;
        }
        int found=0;
#ifdef COLI_VULKAN
        if(vk_reg_served(layer,eid)) found=1;               /* VK-tier-served: no load */
#endif
        if(!found&&pin_indexed(m,layer,eid)) found=1;
        ESlot *Sl=m->ecache[layer]; int nn=m->ecn[layer];
        if(!found&&ecache_indexed(m,layer,eid,1)) found=1;
        if(found){ pthread_mutex_unlock(&g_pilot_mx); continue; }
        int slot;
        if(nn<m->ecap){ slot=nn; m->ecn[layer]=nn+1; }
        else slot=eslot_lru_victim(Sl,nn,m->ecap);    /* riusa libero-con-slab, poi LRU; slot svuotati solo sotto ecap (#1034) */
        if(slot<0){ atomic_fetch_add_explicit(&g_pilot_drops,1,memory_order_relaxed); pthread_mutex_unlock(&g_pilot_mx); continue; }
        /* LFRU eviction guard (#441, narrowed by #497): protect only a genuinely WARM
         * resident (>=2 accesses) that is clearly hotter (see pilot_realload) */
        if(g_pilot_evict_guard && m->eheat && m->elast && Sl[slot].eid>=0){
            int vid=Sl[slot].eid; uint32_t vh=m->eheat[layer][vid];
            if(vh>=2){
                uint64_t vs=tier_lfru_score(vh,m->elast[layer][vid],m->eaccess_clock);
                uint64_t cs=tier_lfru_score(m->eheat[layer][eid],m->elast[layer][eid],m->eaccess_clock);
                if(vs+(vs>>2)+(4u<<8)>cs){ atomic_fetch_add_explicit(&g_pilot_drops,1,memory_order_relaxed);
                                            pthread_mutex_unlock(&g_pilot_mx); continue; } } }
        ESlot *dst=&Sl[slot];
        ecache_reserve(m,layer,dst,eid);           /* visible reservation; never considered resident/evictable */
        g_pilot_inflight[layer]++;
        pthread_mutex_unlock(&g_pilot_mx);

        int li=uring_load_add(&g_ub_pilot,m,layer,eid,dst,0);
        if(li<0){
            pthread_mutex_lock(&g_pilot_mx); ecache_hide(m,layer,dst); g_pilot_inflight[layer]--;
            pthread_cond_broadcast(&g_pilot_cv); pthread_mutex_unlock(&g_pilot_mx);
            atomic_fetch_add_explicit(&g_pilot_drops,1,memory_order_relaxed); continue;
        }
        done[nd++]=(PilotUringDone){layer,eid,li,dst};
    }
    __atomic_store_n(&pilot_r,r,__ATOMIC_RELEASE);
    if(!nd) return;
    if(uring_submit_batch(&g_ub_pilot)<0){
        int err=errno;
        for(int i=0;i<g_ub_pilot.nload;i++){
            g_ub_pilot.load[i].error=err; g_ub_pilot.load[i].done=1;
        }
    }
    for(int i=0;i<nd;i++){
        PilotUringDone *d=&done[i];
        int rc=uring_finalize_load(&g_ub_pilot,d->li,0);
        pthread_mutex_lock(&g_pilot_mx);
        if(rc==0){
            ecache_publish(m,d->layer,d->dst,d->eid);
            d->dst->used=(uint64_t)__atomic_add_fetch(&m->eclock,1,__ATOMIC_RELAXED);
            atomic_fetch_add_explicit(&g_pilot_loads,1,memory_order_relaxed);
        }else{
            ecache_hide(m,d->layer,d->dst);
            atomic_fetch_add_explicit(&g_pilot_drops,1,memory_order_relaxed);
        }
        g_pilot_inflight[d->layer]--;
        pthread_cond_broadcast(&g_pilot_cv);
        pthread_mutex_unlock(&g_pilot_mx);
        if(rc) fprintf(stderr,"[PILOT/URING] load speculativo abbandonato: layer %d expert %d: %s\n",
                       d->layer,d->eid,strerror(g_ub_pilot.load[d->li].error));
    }
}
#endif
/* SPMC ring claim: each of N pilot workers grabs a UNIQUE ring index via CAS, never
 * advancing past pilot_w. Returns 1 and *out=index, or 0 if the ring is empty. The
 * producer stays single (main thread, only pilot_w) — this only splits the consumer. */
static int pilot_ring_claim(int *out_l, int *out_e){
    for(;;){
        unsigned r=__atomic_load_n(&pilot_r,__ATOMIC_ACQUIRE);
        unsigned w=__atomic_load_n(&pilot_w,__ATOMIC_ACQUIRE);
        if(r==w) return 0;                              /* empty */
        /* Read the payload BEFORE committing the claim. While pilot_r==r the slot at
         * index r cannot be overwritten (the producer's w-r<4096 guard keeps pilot_w
         * below r+4096). If the CAS succeeds, pilot_r was r the whole time -> the read
         * is valid. If it fails, another worker advanced pilot_r; we discard and retry
         * (a torn read there is thrown away, never used). */
        int l=atomic_load_explicit(&pilot_q[r&4095].l,memory_order_relaxed), e=atomic_load_explicit(&pilot_q[r&4095].e,memory_order_relaxed);
        if(__atomic_compare_exchange_n(&pilot_r,&r,r+1,/*weak=*/1,
                                       __ATOMIC_ACQ_REL,__ATOMIC_ACQUIRE)){
            *out_l=l; *out_e=e; return 1;                /* claimed exactly one item, payload valid */
        }
        /* lost the CAS race to another worker -> retry */
    }
}
static void *pilot_worker(void *arg){
    (void)arg;
    for(;;){
#ifdef __linux__
        if(g_pilot_real && g_uring){                    /* URING drains a whole batch itself -> single worker (nw==1) */
            unsigned r=__atomic_load_n(&pilot_r,__ATOMIC_ACQUIRE);
            unsigned w=__atomic_load_n(&pilot_w,__ATOMIC_ACQUIRE);
            if(r==w){ usleep(200); continue; }
            pilot_uring_batch(pilot_m);
            continue;
        }
#endif
        int l,e;                                        /* blocking / hint path: SPMC, one item per worker */
        if(!pilot_ring_claim(&l,&e)){ usleep(200); continue; }
        if(g_pilot_real) pilot_realload(pilot_m, l, e); /* QD=N: N concurrent preads instead of 1 */
        else             expert_prefetch(pilot_m, l, e);
    }
    return NULL;
}
/* Spawn the pilot worker(s) ONCE, from the MAIN thread (first pilot_prefetch/couple_prefetch).
 * Only the blocking PILOT_REAL path fans out to g_pilot_nw threads; the URING path already
 * batches to QD>1 and hint-only is cheap fadvise, so both keep a single worker. */
static void pilot_spawn(Model *m){
    if(pilot_m) return;                                 /* pilot_m is the once-guard; only the main thread calls this */
    pilot_m=m;
    int uring_active=0;
#ifdef __linux__
    uring_active=g_uring;
#endif
    /* INVARIANTE: con URING attivo nw DEVE restare 1 — pilot_uring_batch e' un
     * consumatore SINGOLO (avanza pilot_r con uno store semplice, non la CAS di
     * pilot_ring_claim); due drainer URING concorrenti corromperebbero pilot_r.
     * Il ramo blocking (pilot_ring_claim, CAS) e' invece SPMC-safe a N. */
    int nw=(g_pilot_real && !uring_active)?g_pilot_nw:1;
    if(nw<1) nw=1; if(nw>16) nw=16;
    for(int i=0;i<nw;i++){ pthread_t t; pthread_create(&t,NULL,pilot_worker,NULL); }
}
/* parse .coli_pairs (see tools/route_pairs.py): "COLIPAIRS 1 <n>" then
 * "<L> <dL> <e> f:c f:c ..." lines. Needs c->n_experts/n_layers -> called post-init. */
static void couple_load(Model *m, const char *path){
    Cfg *c=&m->c; int E=c->n_experts, NL=c->n_layers;
    FILE *f=fopen(path,"rb");
    if(!f){ fprintf(stderr,"[COUPLE] cannot open %s\n",path); return; }
    char magic[16]; int ver=0; long n=0;
    if(fscanf(f,"%15s %d %ld",magic,&ver,&n)!=3 || strcmp(magic,"COLIPAIRS") || ver!=1){
        fprintf(stderr,"[COUPLE] %s: bad header\n",path); fclose(f); return; }
    size_t cells=(size_t)NL*2*E*CP_M;
    cp_pred=malloc(cells*sizeof(int16_t)); cp_cnt=calloc(cells,sizeof(float));
    if(!cp_pred||!cp_cnt){ fprintf(stderr,"[COUPLE] OOM\n"); free(cp_pred); free(cp_cnt); cp_pred=NULL; fclose(f); return; }
    for(size_t i=0;i<cells;i++) cp_pred[i]=-1;
    long used=0;
    char *ln=NULL; size_t lcap=0;
    while(getline(&ln,&lcap,f)>0){          /* line-based: a malformed line cannot eat the next */
        char *p=ln; int L,dL,e; int nc=0;
        if(sscanf(p,"%d %d %d%n",&L,&dL,&e,&nc)!=3) continue;
        p+=nc;
        if(L<0||L>=NL||(dL!=1&&dL!=2)||e<0||e>=E) continue;
        size_t base=((size_t)(L*2+(dL-1))*E+e)*CP_M;
        int j=0;
        while(j<CP_M){
            int fe; float fc;
            if(sscanf(p," %d:%f%n",&fe,&fc,&nc)!=2) break;
            p+=nc;
            if(fe>=0&&fe<E){ cp_pred[base+j]=(int16_t)fe; cp_cnt[base+j]=fc; j++; }
        }
        if(j) used++;
    }
    free(ln);
    fclose(f);
    g_couple=1;
    fprintf(stderr,"[COUPLE] %s: %ld conditioning entries, K=%d depth=%d\n",path,used,g_couple_k,g_couple_d);
}
/* score + enqueue: called from moe() after FASE A with the position's routed set */
static void couple_prefetch(Model *m, int layer, const int *idx, int Ke){
    Cfg *c=&m->c; int E=c->n_experts;
    if(E>512) return;
    pilot_spawn(m);
    for(int dL=1; dL<=g_couple_d; dL++){
        int lt=layer+dL;
        if(lt>=c->n_layers || !m->L[lt].sparse) continue;
        float sc[512]; memset(sc,0,(size_t)E*sizeof(float));
        for(int kk=0;kk<Ke;kk++){
            size_t base=((size_t)(layer*2+(dL-1))*E+idx[kk])*CP_M;
            for(int j=0;j<CP_M && cp_pred[base+j]>=0;j++) sc[cp_pred[base+j]]+=cp_cnt[base+j];
        }
        for(int kk=0;kk<g_couple_k;kk++){
            int best=-1; float bv=0;
            for(int e=0;e<E;e++) if(sc[e]>bv){bv=sc[e];best=e;}
            if(best<0) break;
            sc[best]=0;
            int found=0;                            /* residency scan, same locking as pilot */
            pthread_mutex_lock(&g_pilot_mx);
#ifdef COLI_VULKAN
            if(vk_reg_served(lt,best)) found=1;             /* VK-tier-served: no load */
#endif
            if(!found&&pin_indexed(m,lt,best)) found=1;
            if(!found&&ecache_indexed(m,lt,best,1)) found=1;
            pthread_mutex_unlock(&g_pilot_mx);
            if(!found){
                unsigned w=__atomic_load_n(&pilot_w,__ATOMIC_RELAXED);
                if(w-__atomic_load_n(&pilot_r,__ATOMIC_ACQUIRE)<4096){
                    atomic_store_explicit(&pilot_q[w&4095].l,lt,memory_order_relaxed); atomic_store_explicit(&pilot_q[w&4095].e,best,memory_order_relaxed);
                    __atomic_store_n(&pilot_w,w+1,__ATOMIC_RELEASE);
                    g_cp_enq++;
                }
            }
        }
    }
}
static void pilot_prefetch(Model *m, int lnext, const float *x, int S){
    Cfg *c=&m->c; Layer *l=&m->L[lnext]; int D=c->hidden, E=c->n_experts;
    int K = g_pilot_k<c->topk ? g_pilot_k : c->topk;
    pilot_spawn(m);
    float *nrm=falloc(D), *ch=falloc(E);
    /* Two-step workspace (allocated once, reused across positions) */
    float *snrm=NULL, *sg=NULL, *su=NULL, *sout=NULL, *hc=NULL;
    int src_layer = lnext - 1;
    int sI = 0;
    int can_two = g_pilot_two && src_layer>=0 && src_layer<c->n_layers
                  && m->L[src_layer].sparse && c->n_shared>0 && c->moe_inter>0;
    if(can_two){
        sI = c->moe_inter * c->n_shared;
        snrm=falloc(D); sg=falloc(sI); su=falloc(sI); sout=falloc(D); hc=falloc(D);
    }
    for(int s=0;s<S;s++){
        const float *xs = x+(int64_t)s*D;
        if(can_two){
            /* Two-step: approximate MoE(src_layer) via shared expert only (resident, no disk),
             * then run lnext's router on the corrected state. */
            Layer *sl = &m->L[src_layer];
            rmsnorm(snrm, xs, sl->post_ln, D, c->eps);
            matmul_qt(sg, snrm, &sl->sh_gate, 1);
            matmul_qt(su, snrm, &sl->sh_up,   1);
            for(int i=0;i<sI;i++) sg[i] = siluf(sg[i]) * su[i];
            matmul_qt(sout, sg, &sl->sh_down, 1);
            for(int i=0;i<D;i++) hc[i] = xs[i] + sout[i];
            rmsnorm(nrm, hc, l->post_ln, D, c->eps);
        } else {
            rmsnorm(nrm, xs, l->post_ln, D, c->eps);
        }
        matmul(ch, nrm, l->router, 1, D, E);
        for(int e=0;e<E;e++) ch[e]=sigmoidf(ch[e])+l->router_bias[e];
        for(int kk=0;kk<K;kk++){
            int best=0; for(int e=1;e<E;e++) if(ch[e]>ch[best]) best=e;
            ch[best]=-2e30f;
            /* Residency scan of the FUTURE layer lnext under g_pilot_mx: with
             * PILOT_REAL=1 the pilot worker mutates ecache[lnext]/ecn[lnext]
             * concurrently, so read them under the same lock (Option A). Decide
             * under the lock, then enqueue AFTER unlocking — the pilot_q ring is
             * lock-free (pilot_w/pilot_r atomics, not g_pilot_mx) so there is no
             * re-entrant double-lock, and the worker re-checks residency under the
             * lock anyway, making a racing redundant enqueue harmless. */
            int found=0;
            pthread_mutex_lock(&g_pilot_mx);
#ifdef COLI_VULKAN
            if(vk_reg_served(lnext,best)) found=1;          /* VK-tier-served: no load */
#endif
            if(!found&&pin_indexed(m,lnext,best)) found=1;
            if(!found&&ecache_indexed(m,lnext,best,1)) found=1;
            pthread_mutex_unlock(&g_pilot_mx);
            if(!found){
                unsigned w=__atomic_load_n(&pilot_w,__ATOMIC_RELAXED);
                if(w-__atomic_load_n(&pilot_r,__ATOMIC_ACQUIRE)<4096){
                    atomic_store_explicit(&pilot_q[w&4095].l,lnext,memory_order_relaxed); atomic_store_explicit(&pilot_q[w&4095].e,best,memory_order_relaxed);
                    __atomic_store_n(&pilot_w,w+1,__ATOMIC_RELEASE);
                }
            }
        }
    }
    free(nrm); free(ch);
    if(can_two){ free(snrm); free(sg); free(su); free(sout); free(hc); }
}

/* forward di UN layer (usato dai 78 principali e dal layer MTP) */
#ifdef COLI_CUDA
/* Inc.2a — intero layer SPARSO residente sul device del layer. x_dev entra e resta;
 * lasciano il device solo: nrm post-attention (router + expert CPU + gather dei
 * gruppi), i nuovi record KV, e la nrm pre-attention sui layer con indexer DSA.
 * Ritorna 0 su errore: il chiamante ripristina lo snapshot e rifa' il layer su CPU. */
static int pipe_layer_sparse(Model *m, Layer *l, int li, float *x_dev, int S, int pos_base,
                             float *nrm_host, float *out_host){
    Cfg *c=&m->c; int D=c->hidden, dev=l->kv_b.cuda_device;
    int sI=c->moe_inter*c->n_shared;
    size_t xb=(size_t)S*D*4;
    if(!l->sh_gate.cuda_eligible||!l->sh_up.cuda_eligible||!l->sh_down.cuda_eligible||
       !qt_cuda_upload(&l->sh_gate)||!qt_cuda_upload(&l->sh_up)||!qt_cuda_upload(&l->sh_down)||
       l->sh_gate.cuda_device!=dev||l->sh_up.cuda_device!=dev||l->sh_down.cuda_device!=dev) return 0;
    /* Inc.4: the layernorm weights are constants — upload once per layer and keep them
     * on the layer's device, instead of two synchronous 24 KB uploads per layer per
     * token (152 sync H2D/token measured on the profile). */
    if(!m->ln_dev) m->ln_dev=calloc((size_t)(c->n_layers+1)*2,sizeof(float*));
    float *w_in=m->ln_dev[(size_t)li*2], *w_post=m->ln_dev[(size_t)li*2+1];
    if(!w_in){
        w_in=coli_cuda_pipe_alloc(dev,(size_t)D*4);
        if(!w_in||!coli_cuda_pipe_upload(dev,w_in,l->in_ln,(size_t)D*4)) return 0;
        m->ln_dev[(size_t)li*2]=w_in;
    }
    if(!w_post){
        w_post=coli_cuda_pipe_alloc(dev,(size_t)D*4);
        if(!w_post||!coli_cuda_pipe_upload(dev,w_post,l->post_ln,(size_t)D*4)) return 0;
        m->ln_dev[(size_t)li*2+1]=w_post;
    }
    float *nrm_d=coli_cuda_pipe_scratch(dev,10,xb);
    float *y_d  =coli_cuda_pipe_scratch(dev,11,xb);
    float *sg_d =coli_cuda_pipe_scratch(dev,12,(size_t)S*sI*4);
    float *su_d =coli_cuda_pipe_scratch(dev,13,(size_t)S*sI*4);
    float *snap =coli_cuda_pipe_scratch(dev,14,xb);
    if(!nrm_d||!y_d||!sg_d||!su_d||!snap) return 0;
    if(!coli_cuda_pipe_peer_copy(dev,snap,dev,x_dev,xb)) return 0;   /* snapshot per il fallback */
    double ta=now_s();
    if(!coli_cuda_pipe_rmsnorm(dev,nrm_d,x_dev,w_in,S,D,c->eps)) return 0;
    /* DSA: i layer con indexer FULL cachano k_idx dalla nrm pre-attention (CPU, piccolo) */
    if(m->has_dsa && li<c->n_layers && m->kv_start[li]==0 && c->idx_type[li]){
        if(!coli_cuda_pipe_download(dev,nrm_d,nrm_host,xb)) return 0;
        int nh=c->index_nh, hd=c->index_hd; (void)nh;
        for(int s=0;s<S;s++){
            int pos=pos_base+s;
            float *kd=coli_kv_row(m->kv->Ic[li],pos,hd);
            matmul_qt(kd, nrm_host+(int64_t)s*D, &m->ix_wk[li], 1);
            layernorm(kd, m->ix_knw[li], m->ix_knb[li], hd, 1e-6f);
            rope_interleave(kd, pos, c);
        }
    }
    if(!attn_pipe_prefill(m,l,li,nrm_d,1,S,pos_base,NULL,y_d)) return 0;
    if(!coli_cuda_pipe_add(dev,x_dev,y_d,(size_t)S*D)) return 0;         /* prima mutazione */
    if(!coli_cuda_pipe_rmsnorm(dev,nrm_d,x_dev,w_post,S,D,c->eps)) return 0;
    /* device router (#431 PR-A): route THIS row on the home device while the
     * stream is still hot, then hand the selection to moe() through the same
     * pre-routed shortcut the Metal layer-CB uses. Any failure (upload, launch,
     * feature gate) falls back to the CPU router inside moe() — byte-identical
     * behaviour, just slower. Gated to the plain routing path: CACHE_ROUTE /
     * ROUTE_P / ROUTE_TRACE keep the CPU ranking they need. */
    static int lr_idx[64]; static float lr_w[64]; static int lr_keff[1];
    int dev_routed=0;
    if(g_cuda_router && S==1 && !g_cache_route && g_route_p<=0.f && !rt_tracing()
       && c->n_experts<=4096 && c->topk<=64 && !l->router_cuda_bad){
        int E=c->n_experts, K=c->topk;
        int Ksel = g_topk>0 ? (g_topk<K?g_topk:K) : K;
        float tp = (g_topp>0 && g_topp<1.f) ? g_topp : 0.f;
        if(!l->router_cuda){
            void *rw=coli_cuda_pipe_alloc(dev,(size_t)E*D*4);
            void *rb=coli_cuda_pipe_alloc(dev,(size_t)E*4);
            if(rw&&rb&&coli_cuda_pipe_upload(dev,rw,l->router,(size_t)E*D*4)
                    &&coli_cuda_pipe_upload(dev,rb,l->router_bias,(size_t)E*4)){
                l->router_cuda=rw; l->router_bias_cuda=rb;
            } else {
                if(rw)coli_cuda_pipe_free(dev,rw); if(rb)coli_cuda_pipe_free(dev,rb);
                l->router_cuda_bad=1;
            }
        }
        if(l->router_cuda &&
           coli_cuda_pipe_router(dev,nrm_d,l->router_cuda,l->router_bias_cuda,
                                 D,E,Ksel,tp,c->norm_topk,c->routed_scale,
                                 lr_idx,lr_w,lr_keff))
            dev_routed=1;
    }
    if(!coli_cuda_pipe_download(dev,nrm_d,nrm_host,xb)) return 0;
    m->t_attn+=now_s()-ta;
    /* OVERLAP: issue the shared expert on the GPU BEFORE moe() runs on the CPU.
     * The shared expert reads nrm_d (valid after the download above) and writes its
     * residual into x_dev (async). While the GPU computes this, the CPU enters moe()
     * for routing + expert disk loads + matmul — ~50ms of work that previously left
     * the GPU idle. The shared expert (~0.5ms) finishes early in that window.
     *
     * After moe(), the routed-expert result is uploaded (sync pipe_upload) and added
     * to x_dev (async). Both residual adds (shared + routed) are ordered on the same
     * stream — the next layer's pipe_rmsnorm reads x_dev after both complete.
     *
     * No pipe_sync at the end: the next layer's pipe_download (sync cudaMemcpy)
     * provides the implicit sync point. The fallback path (caller downloads x_dev)
     * also uses pipe_download which syncs. This lets GPU work chain across layers
     * without a per-layer stall.
     *
     * Profiling: moe() self-times its own t_emm (routed expert matmul). We time only
     * the GPU work that moe() does NOT cover: the shared-expert dispatch and the
     * routed-expert upload+add. Previously a single outer span wrapped everything
     * including moe(), double-counting the routed-expert time and driving the
     * profile's "other" bucket negative (#292). */
    double te=now_s();
    if(!coli_cuda_pipe_gemm(l->sh_gate.cuda,sg_d,nrm_d,S)) return 0;
    if(!coli_cuda_pipe_gemm(l->sh_up.cuda,su_d,nrm_d,S)) return 0;
    if(!coli_cuda_pipe_silu_mul(dev,sg_d,su_d,(size_t)S*sI)) return 0;
    if(!coli_cuda_pipe_gemm(l->sh_down.cuda,y_d,sg_d,S)) return 0;
    if(!coli_cuda_pipe_add(dev,x_dev,y_d,(size_t)S*D)) return 0;  /* shared residual (async) */
    m->t_emm += now_s()-te;                                       /* shared-expert GPU dispatch only */
    /* expert routed su CPU/gruppi GPU come oggi (shared saltata: la fa il device) */
    if(dev_routed){ g_pre_idx=lr_idx; g_pre_w=lr_w; g_pre_keff=lr_keff; }
    float *res_acc=NULL;
    if(g_cuda_resid && S==1){
        g_pres_slots=coli_cuda_pipe_scratch(dev,25,(size_t)g_cuda_ndev*D*sizeof(float));
        res_acc     =coli_cuda_pipe_scratch(dev,26,(size_t)D*sizeof(float));
        if(g_pres_slots && res_acc){ g_pres_home=dev; g_pres_xsrc=nrm_d; g_pres_nused=0; g_pres_nrefs=0; }
        else { g_pres_slots=NULL; res_acc=NULL; }
    }
    moe(m,l,li,nrm_host,S,out_host,0);                            /* self-times its own t_emm */
    if(dev_routed){ g_pre_idx=NULL; g_pre_w=NULL; g_pre_keff=NULL; }
    if(g_pres_home>=0){
        int nused=g_pres_nused; g_pres_home=-1; g_pres_xsrc=NULL;
        if(nused>0){
            /* partials are in flight toward our slots; take() orders the legacy
             * stream behind every issue event and reduces in issue order. A
             * failure here would silently drop routed experts — that is a wrong
             * answer, not a slow one, so it is fatal by design. */
            int take_ok=coli_cuda_expert_group_resident_take(dev,g_pres_used,nused,g_pres_slots,res_acc,D);
            eslots_release(g_pres_refs,g_pres_nrefs); g_pres_nrefs=0;
            if(!take_ok || !coli_cuda_pipe_add(dev,x_dev,res_acc,(size_t)D)){
                fprintf(stderr,"[CUDA] resident expert take failed — refusing to drop routed experts\n");
                exit(1);
            }
        }
    }
    te=now_s();
    if(!coli_cuda_pipe_upload(dev,y_d,out_host,xb)) return 0;     /* sync: waits for moe */
    if(!coli_cuda_pipe_add(dev,x_dev,y_d,(size_t)S*D)) return 0;  /* routed residual (async) */
    m->t_emm += now_s()-te;                                       /* routed-expert upload + add only */
    return 1;
}
#endif

static void layer_forward_rows(Model *m, Layer *l, int li, float *x, int S, int pos_base,
                               KVState *const *kvs, const int *positions, float *nrm, float *tmp){
    Cfg *c=&m->c; int D=c->hidden;
    if(g_spec && g_prefetch && l->sparse && m->enr[li]>0)
        for(int z=0;z<m->enr[li];z++){
            int pe=m->eroute[li][z];
#ifdef COLI_VULKAN
            /* VK-tier-served experts never need host bytes at decode — their pages are
             * never demand-read, so on a RAM-tight box this WILLNEED would re-fetch the
             * same hot experts from disk every time cache pressure evicts them (#523).
             * S>4 = prefill batches, where the tier doesn't serve and the hint stays. */
            if(S<=4 && vk_reg_served(li,pe)) continue;
#endif
            expert_prefetch(m,li,pe);
        }
    if(g_looka && S==1 && li<c->n_layers && l->sparse) la_predict(m,li,x,0);
#ifdef COLI_METAL
    /* FULL-LAYER CB: in_ln + attention + residuo + post_ln + shared expert + router/top-K
     * in un solo submit GPU; la CPU legge il routing e fa solo resolve/disk/expert-CB.
     * Fallback: qualsiasi condizione mancante -> percorso CPU intero qui sotto.
     * !kvs: ragged mux rows (per-row KV/position) are not expressible in this kernel's
     * single Lc/Rc + pos_base contract — see the matching guard in attention_rows.
     * QUANT GUARD (!g_kv8&&!g_tq): the fused layer kernel reads f32 Lc/Rc rows (not
     * allocated under KV8/KV_TQ); quantized KV falls to the CPU path below.
     * metal_fused_layer_fmt_miss & METAL_FUSED_LAYER_TENSORS: kv_b on its
     * two-format+mode term (fmt==2, or fmt==4 with g_moe_exact off) plus the
     * POSITIVE allowlist (fmt 1/2/3/4) over q_a/q_b/kv_a/o/sh_gate/sh_up/sh_down,
     * same fail-closed discipline as attention_rows, extended to the
     * shared-expert MLP this fused kernel also covers. The mm_gemv shader these
     * bind_gemv calls dispatch through has a real fmt==8 branch (this PR's own
     * Metal kernel commit), but the WP_() macro below picks q8 only for fmt==1
     * and q4 (NULL/unallocated for fmt=8) otherwise, so none of these seven
     * bind_gemv-routed tensors can safely carry fmt=8 through this fused path
     * yet -- and a fmt=5/6 tensor's REAL q4 bytes would be silently misread by
     * the shader's terminal f32 fallback (see the predicate's comment). CPU
     * below dispatches every format correctly, incl. fmt=8 via
     * matmul_qt_ex/matmul_fp8. Fixing WP_() and wiring fmt=8 through bind_gemv
     * is the same deferred follow-up noted in attention_rows, not done in this
     * round. */
    if(g_metal_enabled && !kvs && !g_kv8 && !g_tq && S<=4 && li<c->n_layers && l->sparse
       && (g_absorb==1||(g_absorb<0&&S<=4)) && m->kv_start[li]==0
       && D==6144 && c->n_heads==64 && c->q_lora==2048 && c->kv_lora==512
       && c->qk_nope==192 && c->qk_rope==64 && c->v_head==256
       && c->n_experts==256 && c->topk==8 && c->n_shared==1 && c->moe_inter==2048
       && !(metal_fused_layer_fmt_miss(l) & METAL_FUSED_LAYER_TENSORS)){
        int sel_active = m->has_dsa && c->idx_type[li] && (pos_base+S) > c->index_topk;
        if(!sel_active){
            static float *linrm,*lnrm,*lsh,*lw; static int *lidx,*lkeff;
            if(!linrm){ linrm=falloc(4*(int64_t)D); lnrm=falloc(4*(int64_t)D); lsh=falloc(4*(int64_t)D);
                        lidx=malloc(4*8*sizeof(int)); lw=malloc(4*8*sizeof(float)); lkeff=malloc(4*sizeof(int)); }
            int Ksel = g_topk>0 ? (g_topk<8?g_topk:8) : 8;
            float tp = (g_topp>0 && g_topp<1.f) ? g_topp : 0.f;
            double ta0=now_s();
            #define WP_(q) ((q).fmt==1?(const void*)(q).q8:(const void*)(q).q4)
            int ok = coli_metal_layer_decode(x, l->in_ln, l->post_ln,
                WP_(l->q_a), l->q_a.s, l->q_a.fmt, l->q_a.gs, l->q_a_ln,
                WP_(l->q_b), l->q_b.s, l->q_b.fmt, l->q_b.gs,
                WP_(l->kv_a), l->kv_a.s, l->kv_a.fmt, l->kv_a.gs, l->kv_a_ln,
                WP_(l->kv_b), l->kv_b.s, l->kv_b.fmt, l->kv_b.gs,
                WP_(l->o), l->o.s, l->o.fmt, l->o.gs,
                WP_(l->sh_gate), l->sh_gate.s, l->sh_gate.fmt, l->sh_gate.gs,
                WP_(l->sh_up),   l->sh_up.s,   l->sh_up.fmt,   l->sh_up.gs,
                WP_(l->sh_down), l->sh_down.s, l->sh_down.fmt, l->sh_down.gs,
                l->router, l->router_bias,
                c->n_experts, c->topk, Ksel, tp, c->norm_topk, c->routed_scale,
                m->Lc[li], m->Rc[li], S, pos_base, m->kv_start[li],
                c->eps, c->theta, c->attn_scale,
                linrm, lnrm, lsh, lidx, lw, lkeff);
            #undef WP_
            if(ok){
                m->t_attn += now_s()-ta0;
                if(m->has_dsa && c->idx_type[li]){            /* index key per selezioni future */
                    for(int s=0;s<S;s++){ int pos=pos_base+s;
                        float *kd=m->Ic[li]+(int64_t)pos*c->index_hd;
                        matmul_qt(kd, linrm+(int64_t)s*D, &m->ix_wk[li], 1);
                        layernorm(kd, m->ix_knw[li], m->ix_knb[li], c->index_hd, 1e-6f);
                        rope_interleave(kd, pos, c);
                    }
                }
                if(g_pilot && S<=8 && li+1<c->n_layers && m->L[li+1].sparse) pilot_prefetch(m,li+1,x,S);
                if(g_looka && S==1 && li+1<c->n_layers && m->L[li+1].sparse){
                    la_predict(m,li+1,x,1);
                    la_predict(m,li+1,x,2);
                }
                g_pre_idx=lidx; g_pre_w=lw; g_pre_keff=lkeff; g_pre_sh=lsh;
                moe(m,l,li,lnrm,S,tmp,1);
                g_pre_idx=NULL; g_pre_w=NULL; g_pre_keff=NULL; g_pre_sh=NULL;
                for(int64_t j=0;j<(int64_t)S*D;j++) x[j]+=tmp[j];
                return;
            }
        }
    }
#endif
    for(int s=0;s<S;s++) rmsnorm(nrm+(int64_t)s*D, x+(int64_t)s*D, l->in_ln, D, c->eps);
    attention_rows(m,l,li,nrm,S,pos_base,kvs,positions,tmp);
    for(int64_t j=0;j<(int64_t)S*D;j++) x[j]+=tmp[j];
    if(g_pilot && S<=8 && li+1<c->n_layers && m->L[li+1].sparse) pilot_prefetch(m,li+1,x,S);
    if(g_looka && S==1 && li+1<c->n_layers && m->L[li+1].sparse){
        la_predict(m,li+1,x,1);  /* baseline: stale-state PILOT */
        la_predict(m,li+1,x,2);  /* two-step: shared-expert-corrected prediction */
    }
    for(int s=0;s<S;s++) rmsnorm(nrm+(int64_t)s*D, x+(int64_t)s*D, l->post_ln, D, c->eps);
    if(l->sparse) moe(m,l,li,nrm,S,tmp,1); else dense_mlp(l,nrm,S,D,c->dense_inter,tmp);
    for(int64_t j=0;j<(int64_t)S*D;j++) x[j]+=tmp[j];
}
static void layer_forward(Model *m, Layer *l, int li, float *x, int S, int pos_base, float *nrm, float *tmp){
    layer_forward_rows(m,l,li,x,S,pos_base,NULL,NULL,nrm,tmp);
}
static void layers_forward_rows_range(Model *m, float *x, int S, int pos_base,
                                      KVState *const *kvs,
                                      const int *positions,
                                      int layer_begin, int layer_end){
    Cfg *c=&m->c; int D=c->hidden;
    if(g_pilot_real){   /* nuovo forward: il possesso-layer riparte da -1 (i layer si rifanno da 0) */
        pthread_mutex_lock(&g_pilot_mx);
        atomic_store_explicit(&g_cur_moe_layer,-1,memory_order_release);
        pthread_mutex_unlock(&g_pilot_mx);
    }
    float *nrm=falloc((int64_t)S*D), *tmp=falloc((int64_t)S*D);
#ifdef COLI_CUDA
    /* PIPE2 (Inc.2a): il residuo resta sul device del layer, saltando tra le schede
     * ai confini di layer. x host diventa STALE finche' la residenza e' attiva.
     *
     * S threshold is device-count-dependent (#273): on a single GPU the resident
     * stream wins at S=1 (evicts the CPU round-trips that dominate small-batch
     * decode — +49% on a 5070 Ti). With layers sharded across multiple GPUs each
     * resident forward crosses P2P per layer group, and at one token per forward
     * those hops don't amortize — A/B on 6x5090 showed S=1 is a wash there. So:
     * single-GPU engages at S=1, multi-GPU keeps the original S>=8 prefill gate.
     * COLI_CUDA_PIPE_S_MIN overrides for anyone who wants to measure. */
    float *x_dev=NULL; int x_dev_on=-1;
    size_t xb=(size_t)S*(size_t)D*4;
    int pipe_s_min = getenv("COLI_CUDA_PIPE_S_MIN") ? atoi(getenv("COLI_CUDA_PIPE_S_MIN"))
                                                     : (g_cuda_ndev<=1 ? 1 : 8);
    int pipe2 = g_cuda_pipe>=2 && !kvs && S>=pipe_s_min && g_cuda_enabled && c->kv_lora<=512 &&
                !(m->has_dsa && pos_base+S>c->index_topk);
#endif
    double tl0=now_s();
    for(int i=layer_begin;i<layer_end;i++){
        /* progresso su stderr per i batch grossi (prefill): il primo byte di risposta
         * puo' arrivare dopo MINUTI di streaming — al buio sembra un blocco. */
        if(S>=8 && (i%4==0 || i==layer_end-1))
            fprintf(stderr,"[prefill] layer %d/%d · %d token · +%.2fs\n", i+1, c->n_layers, S, now_s()-tl0);
#ifdef COLI_CUDA
        Layer *l=&m->L[i];
        if(pipe2 && l->sparse && i<c->n_layers &&
           l->q_a.cuda_eligible&&l->q_b.cuda_eligible&&l->kv_a.cuda_eligible&&
           l->kv_b.cuda_eligible&&l->o.cuda_eligible&&
           qt_cuda_upload(&l->q_a)&&qt_cuda_upload(&l->q_b)&&qt_cuda_upload(&l->kv_a)&&
           qt_cuda_upload(&l->kv_b)&&qt_cuda_upload(&l->o)&&
           l->q_a.cuda_device==l->kv_b.cuda_device&&l->q_b.cuda_device==l->kv_b.cuda_device&&
           l->kv_a.cuda_device==l->kv_b.cuda_device&&l->o.cuda_device==l->kv_b.cuda_device){
            int dev=l->kv_b.cuda_device, ok=1;
            float *dst=coli_cuda_pipe_scratch(dev,15,xb);
            /* Ogni uscita dalla residenza deve riportare il residuo su host PRIMA di
             * lasciare il layer al percorso CPU: x host e' STALE finche' x_dev_on>=0,
             * quindi scaricare e' l'unico modo di non calcolare su uno stato vecchio. */
            if(!dst){
                /* niente scratch su questo device: si cade sul percorso CPU qui sotto */
                if(x_dev_on>=0){ coli_cuda_pipe_download(x_dev_on,x_dev,x,xb); x_dev_on=-1; }
            } else {
                if(x_dev_on<0) ok=coli_cuda_pipe_upload(dev,dst,x,xb);
                else if(x_dev_on!=dev){
                    double tp=g_prof?now_s():0;
                    ok=coli_cuda_pipe_peer_copy(dev,dst,x_dev_on,x_dev,xb);
                    if(g_prof){m->t_p2p+=now_s()-tp;m->n_p2p++;}
                }
                else dst=x_dev;
                if(ok){
                    x_dev=dst; x_dev_on=dev;
                    if(pipe_layer_sparse(m,l,i,x_dev,S,pos_base,nrm,tmp)) continue;
                    /* fallback: snapshot -> host, layer rifatto sul percorso CPU */
                    coli_cuda_pipe_peer_copy(dev,x_dev,dev,coli_cuda_pipe_scratch(dev,14,xb),xb);
                    coli_cuda_pipe_download(dev,x_dev,x,xb);
                    x_dev_on=-1;
                }else{
                    /* upload o peer-copy fallita (P2P disabilitato, rifiuto del driver,
                     * NVLink giu'). Se la residenza era attiva il dato autorevole e'
                     * ancora sul device PRECEDENTE, non in dst: va scaricato da li'.
                     * Con x_dev_on<0 ha fallito l'upload e x host e' gia' autorevole. */
                    if(x_dev_on>=0) coli_cuda_pipe_download(x_dev_on,x_dev,x,xb);
                    x_dev_on=-1;
                }
            }
        } else if(x_dev_on>=0){                 /* layer fuori pipe: il residuo torna a casa */
            coli_cuda_pipe_download(x_dev_on,x_dev,x,xb);
            x_dev_on=-1;
        }
#endif
        layer_forward_rows(m,&m->L[i],i,x,S,pos_base,kvs,positions,nrm,tmp);
    }
#ifdef COLI_CUDA
    if(x_dev_on>=0) coli_cuda_pipe_download(x_dev_on,x_dev,x,xb);
#endif
    free(nrm); free(tmp);
}
static void layers_forward_rows(Model *m, float *x, int S, int pos_base,
                                KVState *const *kvs, const int *positions){
    layers_forward_rows_range(m,x,S,pos_base,kvs,positions,0,m->c.n_layers);
}
static void layers_forward(Model *m, float *x, int S, int pos_base){
    layers_forward_rows(m,x,S,pos_base,NULL,NULL);
}
static void layers_forward_range(Model *m, float *x, int S, int pos_base,
                                 int layer_begin, int layer_end){
    layers_forward_rows_range(m,x,S,pos_base,NULL,NULL,layer_begin,layer_end);
}

static void kv_alloc(Model *m, int max_t){
    Cfg *c=&m->c;
    KVState *k=m->kv;
#ifdef COLI_CUDA
    if(m->kv_dev_L) for(int i=0;i<c->n_layers+1;i++){    /* dimensioni cambiate: ombra da rifare */
        if(m->kv_dev_L[i]){ coli_cuda_pipe_free(m->L[i<c->n_layers?i:0].kv_b.cuda_device,m->kv_dev_L[i]); m->kv_dev_L[i]=NULL; }
        if(m->kv_dev_R[i]){ coli_cuda_pipe_free(m->L[i<c->n_layers?i:0].kv_b.cuda_device,m->kv_dev_R[i]); m->kv_dev_R[i]=NULL; }
        m->kv_dev_valid[i]=0;
    }
#endif
#ifdef COLI_VULKAN
    if(g_vulkan&&m->vk_kv_valid){                        /* dimensioni cambiate: cache VK da rifare */
        coli_vk_kv_reset();
        for(int i=0;i<c->n_layers+1;i++) m->vk_kv_valid[i]=0;
    }
#endif
    if(k->Lc){ for(int i=0;i<c->n_layers+1;i++){
#ifdef COLI_METAL
        if(g_metal_enabled){ coli_metal_unregister(k->Lc[i]); coli_metal_unregister(k->Rc[i]); }
#endif
        free(k->Lc[i]); free(k->Rc[i]); } free(k->Lc); free(k->Rc); }
    if(k->Lc8){ for(int i=0;i<c->n_layers+1;i++){
        free(k->Lc8[i]); free(k->Rc8[i]);
        free(k->Lsc[i]); free(k->Rsc[i]); }
        free(k->Lc8); free(k->Rc8); free(k->Lsc); free(k->Rsc);
        k->Lc8=k->Rc8=NULL; k->Lsc=k->Rsc=NULL; }
    if(k->Ic){ for(int i=0;i<c->n_layers;i++) free(k->Ic[i]); free(k->Ic); k->Ic=NULL; }
    if(m->has_dsa){
        k->Ic=calloc(c->n_layers,sizeof(float*));
        for(int i=0;i<c->n_layers;i++) if(c->idx_type[i]) k->Ic[i]=falloc((int64_t)max_t*c->index_hd);
    }
    k->max_t=max_t;
    int NR=c->n_layers+1;                        /* riga extra: KV del layer MTP */
    k->Lc=calloc(NR,sizeof(float*)); k->Rc=calloc(NR,sizeof(float*));
    if(g_kv8 || g_tq){
        /* KV8: byte fp8 + una scala f32 per riga. KV_TQ: byte polari impacchettati +
         * raggio f32 per riga. In entrambi i casi Lc/Rc restano NULL e le righe hanno
         * larghezza in BYTE lb/rb — KV8: kv_lora/qk_rope (1 byte/valore); TQ:
         * coli_tq_row_bytes (< kv_lora, solo gli angoli). Lsc/Rsc reggono scala|raggio. */
        int lb = g_tq ? coli_kvq_row_bytes(c->kv_lora,g_tq_bits,g_tq_codec) : c->kv_lora;
        int rb = g_tq ? coli_kvq_row_bytes(c->qk_rope,g_tq_bits,g_tq_codec) : c->qk_rope;
        if(g_kv8) coli_fp8_lut_init();
        k->Lc8=calloc(NR,sizeof(uint8_t*)); k->Rc8=calloc(NR,sizeof(uint8_t*));
        k->Lsc=calloc(NR,sizeof(float*));   k->Rsc=calloc(NR,sizeof(float*));
        for(int i=0;i<NR;i++){
            k->Lc8[i]=malloc((size_t)max_t*lb);
            k->Rc8[i]=malloc((size_t)max_t*rb);
            k->Lsc[i]=falloc((int64_t)max_t*(g_kv8?coli_kv8_nscale(c->kv_lora,g_kv8_gs):1));
            k->Rsc[i]=falloc(max_t);
            if(!k->Lc8[i]||!k->Rc8[i]){fprintf(stderr,"OOM kv8\n");exit(1);}
        }
    } else
    for(int i=0;i<NR;i++){ k->Lc[i]=falloc((int64_t)max_t*c->kv_lora);
        k->Rc[i]=falloc((int64_t)max_t*c->qk_rope);
#ifdef COLI_METAL
        /* page-align + register Lc/Rc for zero-copy GPU attention. falloc isn't 16K-aligned,
         * so re-allocate aligned and register the exact byte length. */
        if(g_metal_enabled){
            size_t lb=(((size_t)max_t*c->kv_lora*sizeof(float))+16383)&~(size_t)16383;
            size_t rb=(((size_t)max_t*c->qk_rope*sizeof(float))+16383)&~(size_t)16383;
            free(k->Lc[i]); free(k->Rc[i]); void *lp,*rp;
            if(posix_memalign(&lp,16384,lb)||posix_memalign(&rp,16384,rb)){fprintf(stderr,"OOM kv\n");exit(1);}
            k->Lc[i]=lp; k->Rc[i]=rp;
            coli_metal_register(k->Lc[i],lb); coli_metal_register(k->Rc[i],rb);
        }
#endif
    }
    m->Lc=k->Lc; m->Rc=k->Rc; m->Ic=k->Ic; m->max_t=k->max_t; m->kv_start=k->kv_start;
    m->Lc8=k->Lc8; m->Rc8=k->Rc8; m->Lsc=k->Lsc; m->Rsc=k->Rsc;
}

static void kv_bind(Model *m, KVState *k){
    if(m->kv!=k && m->kv_dev_valid)                 /* ombra legata al KVState corrente */
        for(int i=0;i<m->c.n_layers+1;i++) m->kv_dev_valid[i]=0;
#ifdef COLI_VULKAN
    if(m->kv!=k && m->vk_kv_valid)                  /* la cache VK specchia la KV corrente */
        for(int i=0;i<m->c.n_layers+1;i++) m->vk_kv_valid[i]=0;
#endif
    m->kv=k; m->Lc=k->Lc; m->Rc=k->Rc; m->Ic=k->Ic;
    m->Lc8=k->Lc8; m->Rc8=k->Rc8; m->Lsc=k->Lsc; m->Rsc=k->Rsc;
    m->max_t=k->max_t; m->kv_start=k->kv_start;
}

static void mtp_absorb(Model *m, const int *next_ids, const float *x, int S, int pos_base);
static float *step(Model *m, const int *ids, int S, int pos_base){
    Cfg *c=&m->c; int D=c->hidden;
    /* Chunked prefill (COLI_PREFILL_CHUNK=N): run a long prompt through the
     * layers in N-token slices. KV rows are position-addressed, so a slice at
     * pos_base+done is exactly the serve path's incremental suffix prefill —
     * every S-scaled activation buffer here and inside attention/moe shrinks
     * from prompt-sized to chunk-sized, and a serve scheduler gains a natural
     * yield point between slices. Leading slices only write KV; logits come
     * from the final slice's fall-through. Skipped under an active MTP draft:
     * mtp_absorb consumes cross-position pairs a chunk boundary would split.
     * Verified: 1,683-token prompt at N=256 produces byte-identical greedy
     * output to the single-shot path. */
    static int chunk=-1;
    if(chunk<0) chunk=getenv("COLI_PREFILL_CHUNK")?atoi(getenv("COLI_PREFILL_CHUNK")):0;
    if(chunk>0 && S>chunk && !(m->has_mtp && g_draft>0)){
        int done=0;
        while(S-done>chunk){
            float *cx=falloc((int64_t)chunk*D);
            for(int s=0;s<chunk;s++) embed_row(m, ids[done+s], cx+(int64_t)s*D);
            layers_forward(m,cx,chunk,pos_base+done);
            free(cx);
            done+=chunk;
        }
        ids+=done; S-=done; pos_base+=done;
    }
    float *x=falloc((int64_t)S*D);
    for(int s=0;s<S;s++) embed_row(m, ids[s], x+(int64_t)s*D);
    layers_forward(m,x,S,pos_base);
    if(m->hlast) memcpy(m->hlast, x+(int64_t)(S-1)*D, D*sizeof(float));
    if(m->has_mtp && S>=2 && g_draft>0) mtp_absorb(m, ids+1, x, S-1, pos_base);
    float *last=falloc(D); rmsnorm(last, x+(int64_t)(S-1)*D, m->final_norm, D, c->eps);
    double th0=now_s();
    float *logit=falloc(c->vocab); matmul_qt(logit,last,&m->lm_head,1);
    m->t_head += now_s()-th0;
    free(x); free(last); return logit;
}

/* come step(), ma ritorna i logits di TUTTE le S posizioni [S,vocab] (per la verifica spec) */
static float *step_all(Model *m, const int *ids, int S, int pos_base){
    Cfg *c=&m->c; int D=c->hidden;
    float *x=falloc((int64_t)S*D);
    for(int s=0;s<S;s++) embed_row(m, ids[s], x+(int64_t)s*D);
    layers_forward(m,x,S,pos_base);
    if(m->h_all) memcpy(m->h_all, x, (int64_t)S*D*sizeof(float));   /* hidden di TUTTE le pos (S<=512) */
    if(m->hlast) memcpy(m->hlast, x+(int64_t)(S-1)*D, D*sizeof(float));
    float *lo=falloc((int64_t)S*c->vocab), *row=falloc(D);
    for(int s=0;s<S;s++){ rmsnorm(row, x+(int64_t)s*D, m->final_norm, D, c->eps);
        matmul_qt(lo+(int64_t)s*c->vocab, row, &m->lm_head, 1); }
    free(x); free(row); return lo;
}

/* One decode token from each independent sequence, evaluated as a single MoE
 * batch.  Prefill and speculative batches retain their contiguous-KV path. */
static float *step_decode_batch(Model *m, const DecodeRow *rows, int S){
    Cfg *c=&m->c; int D=c->hidden;
    /* Ragged KV currently uses MLA absorption; the stack kernel is sized to 512. */
    if(!rows || S<1 || S>512 || c->kv_lora>512) return NULL;
    KVState *kvs[512]; int positions[512];
    float *x=falloc((int64_t)S*D);
    for(int s=0;s<S;s++){
        if(!rows[s].kv || !rows[s].kv->Lc || !rows[s].kv->Rc || !rows[s].kv->kv_start ||
           rows[s].token<0 || rows[s].token>=c->vocab ||
           rows[s].pos<0 || rows[s].pos>=rows[s].kv->max_t){
            free(x); return NULL;
        }
        for(int l=0;l<c->n_layers;l++){
            if(((g_kv8||g_tq) ? (!rows[s].kv->Lc8 || !rows[s].kv->Rc8 ||
                         !rows[s].kv->Lc8[l] || !rows[s].kv->Rc8[l] ||
                         !rows[s].kv->Lsc[l] || !rows[s].kv->Rsc[l])
                      : (!rows[s].kv->Lc[l] || !rows[s].kv->Rc[l])) ||
               rows[s].kv->kv_start[l]<0 || rows[s].kv->kv_start[l]>rows[s].pos ||
               (m->has_dsa && c->idx_type[l] &&
                (!rows[s].kv->Ic || !rows[s].kv->Ic[l]))){ free(x); return NULL; }
        }
        for(int p=0;p<s;p++) if(rows[p].kv==rows[s].kv){ free(x); return NULL; }
        kvs[s]=rows[s].kv; positions[s]=rows[s].pos;
        embed_row(m,rows[s].token,x+(int64_t)s*D);
    }
    layers_forward_rows(m,x,S,0,kvs,positions);
    float *norm=falloc((int64_t)S*D);
    for(int s=0;s<S;s++)
        rmsnorm(norm+(int64_t)s*D,x+(int64_t)s*D,m->final_norm,D,c->eps);
    double th0=now_s();
    float *logit=falloc((int64_t)S*c->vocab);
    matmul_qt(logit,norm,&m->lm_head,S);
    m->t_head+=now_s()-th0;
    free(x); free(norm);
    return logit;
}

/* METODO E — prompt-lookup: cerca l'occorrenza piu' recente dell'ultimo bigramma nel
 * contesto e propone i token che la seguirono. Zero pesi extra, zero costo: e' solo
 * un'ipotesi che il modello verifichera'. */
/* ---- CORPUS DRAFTS (REST-style external draft source) ----------------------
 * COLI_DRAFT_CORPUS=<file>  whitespace-separated token ids of past generations
 *                           (build one with TOKENS=1, which already dumps them)
 * COLI_CORPUS_K=n           proposal depth, default 8 (cap 48: batch[64] in spec_decode)
 * Proposals come from the longest suffix of the live context that also occurs in
 * the corpus; the engine's existing verification accepts only what it would have
 * generated anyway, so output stays byte-identical (lossless by construction).
 * Unlike MTP (1 token/forward) a corpus hit proposes a whole span at once. */
static int *g_corp=NULL; static long g_corp_n=0; static int g_corp_k=0;
static uint64_t g_corp_prop=0, g_corp_acc=0;
static void corpus_load(void){
    const char *p=getenv("COLI_DRAFT_CORPUS");
    if(!p||!*p) return;
    FILE *f=fopen(p,"rb");
    if(!f){ fprintf(stderr,"[CORPUS] cannot open %s\n",p); return; }
    long cap=1<<16; g_corp=malloc((size_t)cap*sizeof(int)); g_corp_n=0;
    int v;
    while(g_corp && fscanf(f,"%d",&v)==1){
        if(g_corp_n>=cap){ cap*=2; int *n2=realloc(g_corp,(size_t)cap*sizeof(int));
            if(!n2){ free(g_corp); g_corp=NULL; break; } g_corp=n2; }
        g_corp[g_corp_n++]=v;
    }
    fclose(f);
    if(!g_corp){ g_corp_n=0; fprintf(stderr,"[CORPUS] OOM\n"); return; }
    g_corp_k=getenv("COLI_CORPUS_K")?atoi(getenv("COLI_CORPUS_K")):8;
    if(g_corp_k<1) g_corp_k=1;
    if(g_corp_k>48) g_corp_k=48;
    fprintf(stderr,"[CORPUS] %ld ids from %s (draft depth %d)\n",g_corp_n,p,g_corp_k);
}
static int corpus_draft(const int *ctx, int nctx, int *out, int k, int minn, int maxn){
    /* il minimo utile e' un match di `minn` piu' ALMENO un token da proporre:
     * sotto quella soglia non esiste proposta possibile (#test_corpus_draft) */
    if(!g_corp||g_corp_n<(long)minn+1||nctx<minn||k<1) return 0;
    for(int n=maxn;n>=minn;n--){
        if(n>nctx) continue;
        const int *suf=ctx+nctx-n;
        for(long i=g_corp_n-n-1;i>=0;i--){
            int ok=1;
            for(int j=0;j<n;j++) if(g_corp[i+j]!=suf[j]||g_corp[i+j]<0){ ok=0; break; }
            if(!ok) continue;
            int g=0;
            for(int j=0;j<k && i+n+j<g_corp_n && g_corp[i+n+j]>=0;j++) out[g++]=g_corp[i+n+j];
            if(g>0) return g;
        }
    }
    return 0;
}
static int ngram_draft(const int *ids, int len, int G, int *draft){
    if(len<4 || G<1) return 0;
    int a=ids[len-2], b=ids[len-1];
    for(int i=len-3;i>=1;i--)
        if(ids[i-1]==a && ids[i]==b){
            int n=0; for(int j=i+1;j<len && n<G;j++) draft[n++]=ids[j];
            return n;
        }
    return 0;
}

/* METODO MTP: propone fino a G draft con la testa multi-token nativa di GLM-5.2.
 * Input: next_tok (appena emesso, posizione kv) e hlast (hidden pre-norm della pos kv-1).
 * Catena DeepSeek-V3: h' = Layer78( eh_proj[ enorm(emb(tok)) ; hnorm(h) ] ),
 * draft = argmax(lm_head(shared_head.norm(h'))). La KV del layer MTP vive alla riga n_layers
 * ed e' valida da kv_start (niente prefill: finestra di solo-decode, basta per il draft). */
static int mtp_argmax(const float *lo, int V){
    int b=0; float bv=lo[0]; for(int i=1;i<V;i++) if(lo[i]>bv){bv=lo[i];b=i;} return b;
}
static int mtp_draft(Model *m, int next_tok, int kv, int G, int *draft){
    Cfg *c=&m->c; int D=c->hidden, li=c->n_layers;
    int p=kv-1; if(p<0||G<1) return 0;
    if(m->kv_start[li]<0 || m->kv_start[li]>p) m->kv_start[li]=p;
    float *x=falloc(D), *cat=falloc(2*D), *hx=falloc(D), *nrm=falloc(D), *tmp=falloc(D);
    float *row=falloc(D), *logit=falloc(c->vocab), *h=falloc(D);
    memcpy(h, m->hlast, D*sizeof(float));
    int tok=next_tok, n=0;
    m->ld_ctx=1;                                 /* DISK_SPLIT: i load da qui sono draft-path */
    int prenorm = getenv("MTP_PRENORM")!=NULL;
    for(int g=0; g<G; g++){
        int pos=p+g; if(pos+2>=m->max_t) break;
        embed_row(m, tok, x);
        rmsnorm(x, x, m->enorm, D, c->eps);
        if(g==0 && !prenorm) rmsnorm(h, h, m->final_norm, D, c->eps);  /* h vero: post model.norm */
        rmsnorm(h, h, m->hnorm, D, c->eps);
        if(getenv("MTP_SWAP")){ memcpy(cat, h, D*sizeof(float)); memcpy(cat+D, x, D*sizeof(float)); }
        else { memcpy(cat, x, D*sizeof(float)); memcpy(cat+D, h, D*sizeof(float)); }
        matmul_qt(hx, cat, &m->eh_proj, 1);
        double n_eh=0; for(int d=0;d<D;d++) n_eh+=hx[d]*hx[d];
        int dbg = getenv("MTP_DEBUG") && atoi(getenv("MTP_DEBUG"))>=2;
        int t_pre=-1;
        if(dbg){ rmsnorm(row, hx, m->mtp_norm, D, c->eps); matmul_qt(logit, row, &m->lm_head, 1);
                 t_pre=mtp_argmax(logit, c->vocab); }
        layer_forward(m, &m->mtpL, li, hx, 1, pos, nrm, tmp);
        double n_post=0; for(int d=0;d<D;d++) n_post+=hx[d]*hx[d];
        rmsnorm(row, hx, m->mtp_norm, D, c->eps);
        matmul_qt(logit, row, &m->lm_head, 1);
        int t2=mtp_argmax(logit, c->vocab);
        if(dbg) fprintf(stderr,"[mtp2] pos=%d in_tok=%d ||eh||=%.1f ||post||=%.1f pre_blk=%d post_blk=%d\n",
                        pos, tok, sqrt(n_eh), sqrt(n_post), t_pre, t2);
        draft[n++]=t2; tok=t2; memcpy(h, hx, D*sizeof(float));
    }
    m->ld_ctx=0;
    free(x); free(cat); free(hx); free(nrm); free(tmp); free(row); free(logit); free(h);
    return n;
}
/* assorbe nella KV della testa MTP le coppie VERIFICATE (emb(token@pos+1), h_vero@pos):
 * next_ids[i] = token alla posizione pos_base+i+1; x[i] = hidden VERO a pos_base+i.
 * Un solo passaggio batch del layer MTP (il batch-union rende economici gli expert). */
static void mtp_absorb(Model *m, const int *next_ids, const float *x, int S, int pos_base){
    if(!m->has_mtp || S<1) return;
    Cfg *c=&m->c; int D=c->hidden, li=c->n_layers;
    if(m->kv_start[li]<0 || m->kv_start[li]>pos_base) m->kv_start[li]=pos_base;
    float *hx=falloc((int64_t)S*D), *cat=falloc(2*D), *e=falloc(D), *hn=falloc(D), *hf=falloc(D);
    int prenorm = getenv("MTP_PRENORM")!=NULL;
    for(int i=0;i<S;i++){
        embed_row(m,next_ids[i],e);
        rmsnorm(e,e,m->enorm,D,c->eps);
        if(prenorm) rmsnorm(hn,x+(int64_t)i*D,m->hnorm,D,c->eps);
        else { rmsnorm(hf,x+(int64_t)i*D,m->final_norm,D,c->eps);   /* vLLM: h POST model.norm */
               rmsnorm(hn,hf,m->hnorm,D,c->eps); }
        if(getenv("MTP_SWAP")){ memcpy(cat,hn,D*sizeof(float)); memcpy(cat+D,e,D*sizeof(float)); }
        else { memcpy(cat,e,D*sizeof(float)); memcpy(cat+D,hn,D*sizeof(float)); }
        matmul_qt(hx+(int64_t)i*D, cat, &m->eh_proj, 1);
    }
    float *nrm=falloc((int64_t)S*D), *tmp=falloc((int64_t)S*D);
    m->ld_ctx=2;                                 /* DISK_SPLIT: load del layer MTP in absorb */
    layer_forward(m,&m->mtpL,li,hx,S,pos_base,nrm,tmp);
    m->ld_ctx=0;
    free(hx); free(cat); free(e); free(hn); free(hf); free(nrm); free(tmp);
}


static void repin_pass_limit(Model *m,int limit);
static void repin_pass(Model *m){ repin_pass_limit(m,16); }

/* ---- METODO F: draft grammaticale (#48) ----
 * gr_feed consuma i byte di ogni token EMESSO e tiene il walker in sync con l'output;
 * grammar_draft propone lo span FORZATO successivo (un solo byte legale per posizione)
 * gia' tokenizzato. Il confine di tokenizzazione non e' garantito coincidere con quello
 * del modello: la verifica assorbe la differenza (al peggio l'ultimo draft e' rifiutato). */
/* Compile grammar TEXT into g: raw GBNF, or — if the first non-space byte is '{'
 * — a JSON-Schema compiled via schema_gbnf.h. Takes ownership of txt (always
 * freed). Fail-soft: returns -1 with g->on=0, engine runs without a grammar. */
static int grammar_setup_text(GrDraft *g, Tok *T, char *txt, const char *label){
    const char *p=txt; while(*p==' '||*p=='\t'||*p=='\n'||*p=='\r') p++;
    if(*p=='{'){
        char serr[160];
        char *gbnf=schema_to_gbnf(txt,serr,sizeof serr);
        free(txt);
        if(!gbnf){ fprintf(stderr,"[SCHEMA] %s: %s (running without grammar)\n",label,serr); return -1; }
        txt=gbnf;
    }
    if(gr_parse(&g->gram,txt)){ fprintf(stderr,"[GRAMMAR] %s: %s\n",label,g->gram.err); free(txt); return -1; }
    free(txt);
    gr_state_init(&g->st,&g->gram);
    if(!g->st.alive){ fprintf(stderr,"[GRAMMAR] %s: grammar cannot be evaluated (left recursion?)\n",label); return -1; }
    if(g->max<1) g->max=24;
    if(g->max>48) g->max=48;
    g->T=T; g->on=1; g->armed=0;
    fprintf(stderr,"[GRAMMAR] %s: %d rules, forced span capped at %d tokens/forward\n",label,g->gram.n,g->max);
    return 0;
}
/* Release a per-request grammar so the slot can host the next request (keeps max). */
static void grammar_teardown(GrDraft *g){
    if(g->gram.n) gr_free(&g->gram);
    free(g->src);                        /* #7: release the cached schema text (NULL-safe) */
    int max=g->max; memset(g,0,sizeof(*g)); g->max=max;
}
static void grammar_setup(GrDraft *g, Tok *T){
    g->max=24;                    /* was a static initializer; see g_grd's declaration (#527) */
    /* GRAMMAR=<file.gbnf> takes precedence; SCHEMA=<file.json> compiles a JSON-Schema
     * to GBNF. Both fail soft: the engine runs without a grammar, output unchanged. */
    const char *gf=getenv("GRAMMAR");
    const char *sf=(gf&&*gf)?NULL:getenv("SCHEMA");
    if((!gf||!*gf)&&(!sf||!*sf)) return;
    const char *path=(gf&&*gf)?gf:sf;
    FILE *f=fopen(path,"rb");
    if(!f){ fprintf(stderr,"[GRAMMAR] cannot open %s\n",path); return; }
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    char *txt=malloc((size_t)n+1);
    if(!txt || fread(txt,1,(size_t)n,f)!=(size_t)n){
        fprintf(stderr,"[GRAMMAR] failed to read %s\n",path); fclose(f); free(txt); return; }
    fclose(f); txt[n]=0;
    if(getenv("GRAMMAR_DRAFT")) g->max=atoi(getenv("GRAMMAR_DRAFT"));
    grammar_setup_text(g,T,txt,path);
}
/* stato pulito all'inizio di ogni RISPOSTA (non tra i \x02MORE, che continuano) */
static void grammar_reset(GrDraft *g){
    if(!g->on) return;
    gr_state_init(&g->st,&g->gram); g->armed=0;
    if(!g->st.alive) g->on=0;
}
/* consuma i byte di un token emesso. Preambolo (prima dell'arming): ignorato.
 * Desync dopo l'arming: si riarma in attesa del prossimo inizio valido — al peggio
 * i draft vengono rifiutati dalla verifica, l'output non cambia MAI. */
static void gr_feed(GrDraft *g, int t){
    if(!g->on||!g->T) return;
    char b[64]; int n=tok_decode(g->T,&t,1,b,63);
    for(int i=0;i<n;i++){
        int r=gr_accept(&g->st,(unsigned char)b[i]);
        if(r==1){ g->armed=1; continue; }
        if(r<0){ g->on=0; return; }                   /* walker spento: fine dei draft */
        if(!g->armed) continue;                       /* preambolo: aspetta l'inizio */
        gr_state_init(&g->st,&g->gram); g->armed=0;   /* desync: riparti dalla radice */
        if(!g->st.alive){ g->on=0; return; }
        if(gr_accept(&g->st,(unsigned char)b[i])==1) g->armed=1;
    }
}
/* propone lo span forzato come token (max cap); 0 se la grammatica dirama qui */
static int grammar_draft(GrDraft *g, int *draft, int cap){
    if(!g->on||!g->armed||!g->T||cap<1) return 0;
    if(g->prop>=32 && g->acc*2<g->prop){              /* guardia adattiva, come per MTP:
        acceptance sotto il 50% = tokenizzazione fuori asse, meglio spegnersi */
        g->on=0;
        fprintf(stderr,"[GRAMMAR] %.0f%% acceptance after %llu proposals: grammar drafts disabled\n",
            100.0*g->acc/g->prop,(unsigned long long)g->prop);
        return 0;
    }
    char fb[512]; int nb=gr_forced(&g->st,fb,(int)sizeof fb-1);
    if(nb<=0) return 0;
    int nt=tok_encode(g->T,fb,nb,draft,cap);          /* renamed local: 'g' is now the state param */
    return nt>0?nt:0;
}

/* ---- SAMPLING (temperatura + nucleus) con verifica speculativa LOSSLESS ----
 * Il draft (MTP/n-gram) e' DETERMINISTICO (argmax della testa): q = massa puntuale.
 * Rejection sampling di Leviathan: accetta il draft x_d con prob p(x_d); al rifiuto
 * ricampiona da p con x_d azzerato e rinormalizzato. La distribuzione risultante e'
 * ESATTAMENTE p: la speculazione resta invisibile all'output anche col sampling. */

/* decode greedy con SELF-SPECULATION n-gram: LOSSLESS (output identico al greedy puro).
 * Ogni forward verifica fino a g_draft token proposti dal contesto: i token accettati
 * costano UNA sola passata sui pesi -> disco e banda RAM ammortizzati su piu' token.
 * all: storia token (capacita' >= kv+n_new+g_draft+2), kv = token gia' in KV.
 * logit = logits della posizione kv-1 (dal prefill); viene liberato qui.
 * emit(tok,ud) per ogni token emesso. Ritorna i token emessi; *kv_out = nuova kv. */
/* STOP MORBIDO (serve/chat): SIGINT chiude il turno CORRENTE per la stessa via
 * del tetto NGEN (stats, usage_save, KV append, sentinella END tutti normali)
 * invece di uccidere il motore; :more puo' continuare la risposta interrotta.
 * Il flag e' armato solo nei serve-loop (intr_install): nei run one-shot e in
 * validazione SIGINT resta il default (morte immediata). Solo POSIX: su
 * Windows il comportamento di Ctrl-C non cambia.
 * EN: soft stop (serve/chat): SIGINT ends the CURRENT turn through the same
 * path as the NGEN cap — stats/usage/KV/END sentinel all normal — instead of
 * killing the engine; :more can continue the interrupted answer. Armed only
 * in the serve loops; one-shot runs keep default SIGINT. POSIX only. */
static volatile sig_atomic_t g_intr=0;
/* #810: SIGINT and SIGTERM mean different things and cannot share a flag. SIGINT is a
 * SOFT stop -- it ends the current turn and the serve loop clears g_intr and keeps
 * serving. SIGTERM is a request to shut down, so it needs a flag the loop does NOT
 * clear. It sets g_intr too, so the in-flight turn unwinds through the ordinary path
 * (stats, usage_save, KV append, END sentinel) instead of being torn down. */
static volatile sig_atomic_t g_shutdown=0;
#if defined(__APPLE__) || defined(__linux__) || defined(__FreeBSD__)
static void intr_sig(int s){ (void)s; g_intr=1; }
static void term_sig(int s){ (void)s; g_intr=1; g_shutdown=1; }
static void intr_install(void){
    struct sigaction sa; memset(&sa,0,sizeof(sa));
    sa.sa_handler=intr_sig; sigemptyset(&sa.sa_mask);
    sa.sa_flags=SA_RESTART;              /* getline/pread non devono vedere EINTR */
    sigaction(SIGINT,&sa,NULL);
    /* SIGTERM deliberately WITHOUT SA_RESTART. The serve loop blocks in getline()
     * waiting for the next request, and SA_RESTART would silently resume that read
     * after the handler ran -- the flag would be set and never looked at again, so
     * `systemctl stop` would hang until its TimeoutStopSec expired into SIGKILL.
     * Without it, getline() returns -1/EINTR and the loop exits through its normal
     * end. The original reason for SA_RESTART was "getline/pread must not see EINTR",
     * and the pread half no longer applies: every read path retries EINTR itself
     * (st.h st_pread_full, the mirror loop here, uring.h). getline is the only caller
     * that needed protecting, and it is precisely the one a shutdown must interrupt.
     * select() in run_serve_mux is unaffected: it tests `> 0`, so an EINTR return is
     * just "no input this round" and the next iteration sees the flag. */
    sa.sa_handler=term_sig;
    sa.sa_flags=0;
    sigaction(SIGTERM,&sa,NULL);
}
#else
static void intr_install(void){}
#endif
/* #678: mid-turn STOP/CANCEL for the single-slot speculative serve path. With
 * KV_SLOTS=1 the whole turn runs inside ONE spec_decode call, so run_serve_mux's
 * stdin poll never runs mid-turn and a server-sent STOP (raised when its stop
 * filter matches, e.g. a role marker) sat unread until max_tokens. These flags
 * are raised by mux_ctl_poll (called from the mux emit callback, once per emitted
 * token) and checked in spec_decode next to g_intr. They are only ever set while
 * a mux spec turn is running and are reset at each turn boundary, so every other
 * spec_decode caller (chat, run, oracle) sees them permanently 0. */
static volatile sig_atomic_t g_mux_stop=0, g_mux_cancel=0;
/* emit callback contract (U7a): `lo` is the vocab-sized logit row the token
 * was picked or verified from -- live only for the duration of the call.
 * Accepted DRAFT tokens get their verification row, so the per-token numeric
 * channel has no gap on the speculative path (they bypass every pick_tok
 * call site in the mux loop). Callers that don't need it ignore it. */
static int spec_decode(Model *m, int *all, int kv, int n_new, int eos, float *logit,
                       void (*emit)(int,const float*,void*), void *ud, int *kv_out, float **logit_out){
    Cfg *c=&m->c; int V=c->vocab; int emitted=0, done=0;
    int draft[64]; if(g_draft>63) g_draft=63;
    int carry_ban=-1;                    /* token rifiutato dalla verifica: escluso dal resample */
    /* #163: draft del modello attivi -> pin della famiglia di kernel per draft+verifica.
     * EN: model drafts live -> pin the kernel family for draft+verify forwards. */
    g_spec_live = (g_draft>0);
    if(spec_pinned() && m->has_mtp){ static int once=0; if(!once){ once=1;
        fprintf(stderr,"[SPEC_PIN] draft+verify pinned to the S=1 kernel family: int4=%s int8=%s (#163; SPEC_PIN=0 for A/B)\n",
            (g_idot&&g_i4s<=1)?"idot":"exact", g_idot?"idot":"exact"); } }
    /* guardia MTP morbida (#163): finestra di 24 proposte, pausa e ri-arma invece del
     * latch permanente — una regressione transitoria non spegne MTP per tutta la sessione.
     * EN: soft MTP guard (#163): 24-proposal window, pause and re-arm instead of the
     * permanent latch — a transient collapse no longer kills MTP for the whole session. */
    enum { GUARD_PAUSE_TOKENS = 256 };
    int gd_min_pct=getenv("COLI_MTP_GUARD_PCT")?atoi(getenv("COLI_MTP_GUARD_PCT")):70;
    int gd_window=getenv("COLI_MTP_GUARD_WINDOW")?atoi(getenv("COLI_MTP_GUARD_WINDOW")):24;
    if(gd_min_pct<0)gd_min_pct=0;if(gd_min_pct>100)gd_min_pct=100;
    if(gd_window<4)gd_window=4;if(gd_window>256)gd_window=256;
    uint64_t gd_prop0=m->mtp_prop, gd_acc0=m->mtp_acc; int gd_pause=0;
    uint64_t cp_prop0=g_corp_prop, cp_acc0=g_corp_acc; int cp_pause=0;
    while(emitted<n_new && !done && !g_intr && !g_mux_stop && !g_mux_cancel){
        /* g_intr / g_mux_*: stessa uscita del tetto n_new (#678) */
        int next=pick_tok(logit,V,carry_ban); carry_ban=-1;
        if((eos>=0 && next==eos) || is_stop(next)){ free(logit); logit=NULL; break; }
        emit(next,logit,ud); free(logit); logit=NULL; all[kv]=next; emitted++; m->n_emit++;
        gr_feed(&g_grd,next);                           /* il walker segue l'output emesso */
        /* One-shot generation does not need logits or KV for the last token.
         * Stateful callers do: their kv_out becomes chat history, feeds MORE,
         * and is persisted to .coli_kv.  Let those callers take the normal
         * g=0 step_all path below so the final token is really committed; just
         * incrementing kv would serialize an uninitialized KV row. */
        if(emitted>=n_new && !kv_out) break;
        int g = 0, gsrc = 0;                            /* sorgente: 1=grammatica 2=MTP/n-gram */
        if(g_grd.on){                                   /* metodo F: prima la grammatica — dove
                                                         * forza, l'acceptance e' ~1 (#48) */
            g=grammar_draft(&g_grd,draft,g_grd.max);
            if(g>0) gsrc=1;
        }
        if(!g && g_corp && g_corp_k>0){                 /* corpus: uno span intero per forward
                                                         * dove il contesto ricalca il congelato */
            /* Guard di accettazione (stessa forma di quello MTP sopra, soglia diversa).
             * Un draft di corpus RIFIUTATO costa piu' di uno MTP: la verifica batcha
             * 1+K righe e in un MoE ogni riga attiva i propri expert, quindi lo spreco
             * scala con la profondita'. Misurato su GLM-5.2/H200: 90% acceptance =
             * +22% decode, 19% = -25% (il break-even sta circa a meta'). Sotto
             * COLI_CORPUS_MINACC (default 50%) la fonte si mette in pausa. */
            if(cp_pause>0){ cp_pause--; if(!cp_pause){ cp_prop0=g_corp_prop; cp_acc0=g_corp_acc; } }
            else {
                uint64_t pw=g_corp_prop-cp_prop0, aw=g_corp_acc-cp_acc0;
                int minacc=getenv("COLI_CORPUS_MINACC")?atoi(getenv("COLI_CORPUS_MINACC")):50;
                if(minacc<0) minacc=0; if(minacc>100) minacc=100;
                if(pw>=24 && aw*100 < pw*(uint64_t)minacc){
                    fprintf(stderr,"[CORPUS] %.0f%% acceptance over the last %llu proposals (<%d%%): "
                            "drafts paused for %d tokens\n",
                            100.0*aw/pw,(unsigned long long)pw,minacc,(int)GUARD_PAUSE_TOKENS);
                    cp_pause=GUARD_PAUSE_TOKENS;
                }
            }
            if(!cp_pause){
                g=corpus_draft(all,kv+1,draft,g_corp_k,3,8);
                if(g>0){ gsrc=3; g_corp_prop+=(uint64_t)g; }
            }
        }
        if(!g && g_draft>0 && m->has_mtp){
            /* pausa adattiva: draft che non vengono mai accettati = solo tassa disco,
             * ma il vecchio g_draft=0 era permanente. EN: adaptive pause; the old
             * g_draft=0 latch was permanent. */
            if(gd_pause>0){ gd_pause--; if(!gd_pause){ gd_prop0=m->mtp_prop; gd_acc0=m->mtp_acc; } }
            else if(m->mtp_prop-gd_prop0>=(uint64_t)gd_window &&
                    (m->mtp_acc-gd_acc0)*100 < (m->mtp_prop-gd_prop0)*(uint64_t)gd_min_pct){
                fprintf(stderr,"[MTP] %.0f%% acceptance over the last %llu proposals: drafts paused for %d tokens\n",
                    100.0*(m->mtp_acc-gd_acc0)/(m->mtp_prop-gd_prop0),
                    (unsigned long long)(m->mtp_prop-gd_prop0), (int)GUARD_PAUSE_TOKENS);
                gd_pause=GUARD_PAUSE_TOKENS;
            }
        }
        if(!g && g_draft>0 && !(m->has_mtp && gd_pause>0)){
            if(m->has_mtp){ g=mtp_draft(m,next,kv,g_draft,draft); m->mtp_prop+=g; if(g)gsrc=2; }
            else { g=ngram_draft(all,kv+1,g_draft,draft); if(g)gsrc=2; }
        }
        if(g>n_new-emitted) g=n_new-emitted;
        if(kv+1+g+1>m->max_t) g=m->max_t-kv-2;
        if(g<0) g=0;
        if(gsrc==1) g_grd.prop+=(uint64_t)g;
        int S=1+g; int batch[64]; batch[0]=next; memcpy(batch+1,draft,g*sizeof(int));
        double tf0=g_prof?now_s():0;
        float *lo=step_all(m,batch,S,kv); m->n_fw++;
        if(g_prof) prof_lat(now_s()-tf0);
        int k=0;                                        /* verifica: accetta finche' coincide */
        if(g>0 && getenv("MTP_DEBUG")){ int veri=argmax_v(lo,V);
            fprintf(stderr,"[mtpdbg] draft0=%d verified=%d %s\n", draft[0], veri, draft[0]==veri?"HIT":"miss"); }
        while(k<g && emitted<n_new){
            int accept;
            if(g_temp<=0) accept = (argmax_v(lo+(int64_t)k*V,V)==draft[k]);
            else { dist_build(lo+(int64_t)k*V,V);          /* rejection sampling: p(draft) */
                   accept = (rndu() < g_pbuf[draft[k]]); }
            if(!accept){ if(g_temp>0) carry_ban=draft[k]; break; }
            if((eos>=0 && draft[k]==eos) || is_stop(draft[k])){ done=1; break; }
            emit(draft[k],lo+(int64_t)k*V,ud); all[kv+1+k]=draft[k]; emitted++; m->n_emit++;
            gr_feed(&g_grd,draft[k]); k++;
        }
        if(gsrc==1) g_grd.acc+=(uint64_t)k;
        else if(gsrc==2 && m->has_mtp) m->mtp_acc+=k;
        else if(gsrc==3) g_corp_acc+=(uint64_t)k;
        if(m->has_mtp && k>=1) mtp_absorb(m, all+kv+1, m->h_all, k, kv);   /* KV MTP in sync coi verificati */
        /* hlast deve corrispondere all'ultima posizione ACCETTATA (kv+k), non a fine batch */
        if(m->h_all && k<S-1) memcpy(m->hlast, m->h_all+(int64_t)k*m->c.hidden, m->c.hidden*sizeof(float));
        kv += 1+k;                                      /* KV oltre kv e' stantia: verra' sovrascritta */
        logit=falloc(V); memcpy(logit, lo+(int64_t)k*V, V*sizeof(float)); free(lo);
        repin_pass(m);                                  /* safe point: all device work is synchronized */
    }
    g_spec_live = 0;                     /* prefill/decode successivi: gate normali / next prefill: normal gates */
    /* logit_out (mux chunking): hand the continuation logits to the caller.
     * NULL here means the loop exited right after emitting a token that was
     * never forwarded — it sits at all[kv], and the caller must forward it
     * before the next chunk. */
    if(logit_out) *logit_out=logit;
    else if(logit) free(logit);
    if(kv_out) *kv_out=kv;
    return emitted;
}

/* emit callback: accumula in un array (validazione) */
typedef struct { int *dst; int n; } EmitStore;
static void emit_store(int t, const float *lo, void *ud){ (void)lo; EmitStore *e=(EmitStore*)ud; e->dst[e->n++]=t; }
/* emit callback: detokenizza e stampa in streaming (chat/run), con heartbeat */
typedef struct { Tok *T; Model *m; double t0; int count; int quiet; } EmitStream;
static void emit_stream(int t, const float *lo, void *ud){
    (void)lo;
    EmitStream *e=(EmitStream*)ud; char dec[64];
    int dn=tok_decode(e->T,&t,1,dec,63); dec[dn]=0; fputs(dec,stdout); fflush(stdout);
    if(!e->quiet && ++e->count%16==0){ double tt=e->m->hits+e->m->miss;
        if(g_cache_route && e->m->route_slots){
            double swap=100.0*e->m->route_swaps/e->m->route_slots;
            fprintf(stderr,"\n[t=%d  RSS %.2f GB  hit %.0f%%  swap %.0f%%  %.2f tok/s  %.2f tok/fw]\n", e->count,
                rss_gb(), tt?100.0*e->m->hits/tt:0.0, swap, e->count/(now_s()-e->t0),
                e->m->n_fw?(double)e->m->n_emit/e->m->n_fw:1.0);
        } else {
            fprintf(stderr,"\n[t=%d  RSS %.2f GB  hit %.0f%%  %.2f tok/s  %.2f tok/fw]\n", e->count,
                rss_gb(), tt?100.0*e->m->hits/tt:0.0, e->count/(now_s()-e->t0),
                e->m->n_fw?(double)e->m->n_emit/e->m->n_fw:1.0);
        }
    }
}

/* teacher-forcing: un solo forward su ids[S], argmax per posizione in pred[S] */
/* DEBUG_LOGITS=1: on a teacher-forcing mismatch, dump the top-5 logits, the top1-top2 margin,
 * and where the expected/got tokens land. Makes a near-tie divergence (e.g. the Metal prefill
 * GEMM accumulation-order drift, #622) visible as the tiny gap it is, rather than a bare token
 * mismatch. Stderr, opt-in, only on a mismatch — normal runs are byte-for-byte unchanged. */
static void dump_top5_logits(int pos, const float *lo, int V, int expected, int got){
    int idx[5]; float val[5];
    for(int k=0;k<5;k++){ idx[k]=-1; val[k]=-INFINITY; }
    for(int i=0;i<V;i++){ float v=lo[i];
        for(int k=0;k<5;k++) if(v>val[k]){
            for(int j=4;j>k;j--){ val[j]=val[j-1]; idx[j]=idx[j-1]; }
            val[k]=v; idx[k]=i; break; } }
    double gap = (idx[1]>=0) ? (double)(val[0]-val[1]) : 0.0;
    fprintf(stderr,"[LOGITS] pos=%d top1-top2 gap=%.3e | expected=%d (%.5f)  got=%d (%.5f) | top5:",
            pos, gap, expected, (expected>=0&&expected<V)?(double)lo[expected]:0.0,
            got, (got>=0&&got<V)?(double)lo[got]:0.0);
    for(int k=0;k<5&&idx[k]>=0;k++) fprintf(stderr," %d:%.5f", idx[k], (double)val[k]);
    fprintf(stderr,"\n");
}
static void forward_all(Model *m, const int *ids, int S, int *pred, const int *ref){
    Cfg *c=&m->c; int D=c->hidden;
    int dbg = ref && getenv("DEBUG_LOGITS");
    kv_alloc(m,S);
    float *x=falloc((int64_t)S*D);
    for(int s=0;s<S;s++) embed_row(m, ids[s], x+(int64_t)s*D);
    layers_forward(m,x,S,0);
    float *lo=falloc(c->vocab);
    float *row=falloc(D);
    for(int s=0;s<S;s++){
        rmsnorm(row, x+(int64_t)s*D, m->final_norm, D, c->eps);   /* heap row (#183) */
        matmul_qt(lo, row, &m->lm_head, 1);
        int best=0; float bv=lo[0]; for(int i=1;i<c->vocab;i++) if(lo[i]>bv){bv=lo[i];best=i;}
        pred[s]=best;
        if(dbg && pred[s]!=ref[s]) dump_top5_logits(s, lo, c->vocab, ref[s], pred[s]);
    }
    free(x); free(lo); free(row);
}

/* log-prob (log-softmax) del token target dato il vettore di logit; *am=1 se e' l'argmax */
/* modalita' SCORING per i benchmark (stile lm-eval, log-likelihood):
 * input: file con righe "<ctxlen> <contlen> <id0> .. <id_{T-1}>"  (T=ctxlen+contlen)
 * output: riga "<logprob_continuazione> <contlen> <greedy 0/1>" per richiesta.
 * Un solo forward per richiesta (teacher-forcing): niente generazione -> fattibile a bassa velocita'. */
static void run_score(Model *m, const char *snap, const char *path){
    Cfg *c=&m->c; int D=c->hidden;
    /* prefisso GLM (#108): il modello vede [gMASK]<sop> in testa a OGNI sequenza di training —
     * scorare stream nudi e' out-of-distribution e deprime/distorce i punteggi. Se il config
     * dice glm* gli id dei due token vengono chiesti al tokenizer.json dello snapshot (per
     * GLM-5.2: 154822,154824 — mai fidarsi di costanti cablate, il vocabolario cambia tra
     * release) e anteposti al CONTESTO delle richieste che non li hanno gia'; chi arriva GIA'
     * prefissato (eval_glm.py post-#194) passa INTATTO. SCORE_PREFIX=0 -> comportamento nudo. */
    int pfx[2]={-1,-1}, pfx_on=0;
    if(!getenv("SCORE_PREFIX")||atoi(getenv("SCORE_PREFIX"))){
        char *ar=NULL; jval *r=cfg_root(snap,&ar);
        jval *mt=json_get(r,"model_type");
        if(mt_is_glm(mt?mt->str:NULL)){
            char tkp[2048]; snprintf(tkp,sizeof(tkp),"%s/tokenizer.json",snap);
            Tok T; tok_load(&T,tkp);
            pfx[0]=tok_id_of(&T,"[gMASK]"); pfx[1]=tok_id_of(&T,"<sop>");
            if(pfx[0]>=0&&pfx[1]>=0){ pfx_on=1;
                fprintf(stderr,"[SCORE] GLM snapshot: prepending [gMASK]<sop> (ids %d,%d) to unprefixed requests — disable with SCORE_PREFIX=0\n",pfx[0],pfx[1]);
            } else fprintf(stderr,"[SCORE] GLM config but tokenizer has no [gMASK]/<sop>: prefix OFF\n");
        }
        free(ar);
    }
    FILE *f=fopen(path,"rb"); if(!f){perror(path);exit(1);}
    int maxT=1; { char *ln=NULL; size_t cp=0;
        while(getline(&ln,&cp,f)>0){ int a,b; if(sscanf(ln,"%d %d",&a,&b)==2 && a+b>maxT) maxT=a+b; }
        free(ln); }
    if(pfx_on) maxT+=2;   /* le richieste senza prefisso crescono di 2 token */
    kv_alloc(m,maxT);
    float *x=falloc((int64_t)maxT*D), *lo=falloc(c->vocab), *row=falloc(D);
    int *ids=malloc(maxT*sizeof(int));
    rewind(f); char *ln=NULL; size_t cp=0; int nreq=0; double t0=now_s();
    while(getline(&ln,&cp,f)>0){
        char *p=ln; int ctxlen=strtol(p,&p,10), contlen=strtol(p,&p,10), T=ctxlen+contlen;
        if(T<=0||ctxlen<1){ printf("0 0 0\n"); fflush(stdout); continue; }
        for(int i=0;i<T;i++) ids[i]=strtol(p,&p,10);
        if(pfx_on && !(T>=2 && ids[0]==pfx[0] && ids[1]==pfx[1])){   /* gia' prefissato -> intatto */
            memmove(ids+2,ids,(size_t)T*sizeof(int));
            ids[0]=pfx[0]; ids[1]=pfx[1]; ctxlen+=2; T+=2;
        }
        for(int s=0;s<T;s++) embed_row(m, ids[s], x+(int64_t)s*D);
        layers_forward(m,x,T,0);
        double lp=0; int greedy=1;
        for(int pos=ctxlen-1; pos<T-1; pos++){
            rmsnorm(row, x+(int64_t)pos*D, m->final_norm, D, c->eps);
            matmul_qt(lo,row,&m->lm_head,1);
            int am; lp += logprob_target(lo,c->vocab,ids[pos+1],&am); if(!am) greedy=0;
        }
        printf("%.6f %d %d\n", lp, contlen, greedy); fflush(stdout);
        if(++nreq%5==0) fprintf(stderr,"[score %d req | %.1fs | RSS %.2f GB | hit %.0f%%]\n",
            nreq, now_s()-t0, rss_gb(), (m->hits+m->miss)?100.0*m->hits/(m->hits+m->miss):0.0);
    }
    free(ln); free(ids); free(x); free(lo); free(row); fclose(f);
}

/* ===========================================================================
 * CAUSAL-ABLATION teacher-forced path.  Reached via ABLATE_SCORE=<manifest>;
 * writes per-target-position final-logit summaries to ABLATE_OUT=<file> (JSONL).
 * For each item it configures g_abl (mode + ablated (layer,expert) cells), runs
 * ONE teacher-forced prefill (moe() applies the ablation), then reads out the
 * FINAL logits at every target position -- the causal effect on the model's real
 * output, not a logit lens.  If ROUTE_TRACE=<path> is also set, moe() dumps the
 * POST-ablation router trace for free (the host correlates by manifest order).
 *
 * Manifest, one item per line, whitespace ints:
 *     item  T  n_prompt  mode  ncells  (L E A){ncells}  t_0 .. t_{T-1}
 *   item    = manifest item id (host joins to (corpus_item, condition, cell))
 *   T       = seq length (prompt + teacher-forced target)
 *   n_prompt= # prompt tokens; targets are positions [n_prompt, T)
 *   mode    = 0 baseline | 1 contribution | 2 route-around | 3 module-swap
 *   ncells  = # ablated cells (0 for baseline)
 *   L E A   = layer, expert, swap-target (A=-1 unless mode 3), one triple per cell
 *   t_*     = token ids (host pre-tokenised, prefix included if the model needs it)
 * abl_reset() before each item makes the ablation PER-ITEM (an item's spec can
 * never leak into the next). NLL/margin/correctness are exact; top-K is for a
 * paired approximate next-token KL on the host. */
#define ABL_LOGIT_TOPK 32
static void run_ablate_score(Model *m, const char *path){
    Cfg *c=&m->c; int D=c->hidden, V=c->vocab;
    FILE *f=fopen(path,"rb"); if(!f){perror(path);exit(1);}
    const char *outp=getenv("ABLATE_OUT");
    FILE *of = outp ? fopen(outp,"wb") : NULL;
    if(outp && !of){ fprintf(stderr,"[ablate] cannot open ABLATE_OUT=%s\n",outp); }
    if(of) fprintf(of,"{\"t\":\"hdr\",\"schema\":\"coli-ablate/1\",\"vocab\":%d,\"topk\":%d}\n",V,ABL_LOGIT_TOPK);
    int maxT=1; { char *ln=NULL; size_t cp=0;
        while(getline(&ln,&cp,f)>0){ long id,T; char *e;
            id=strtol(ln,&e,10); if(e==ln) continue; T=strtol(e,&e,10);
            if(T>maxT) maxT=(int)T; (void)id; }
        free(ln); }
    kv_alloc(m,maxT);
    float *x=falloc((int64_t)maxT*D), *lo=falloc(V), *row=falloc(D);
    int *ids=malloc((size_t)maxT*sizeof(int));
    int tk_id[ABL_LOGIT_TOPK]; float tk_val[ABL_LOGIT_TOPK];
    rewind(f); char *ln=NULL; size_t cp=0; int nreq=0; double t0=now_s();
    while(getline(&ln,&cp,f)>0){
        char *p=ln, *e;
        long item=strtol(p,&e,10); if(e==p) continue; p=e;                 /* blank line */
        long T=strtol(p,&e,10);  if(e==p){ fprintf(stderr,"[ablate] bad T\n"); continue; } p=e;
        long np=strtol(p,&e,10); if(e==p){ fprintf(stderr,"[ablate] bad n_prompt\n"); continue; } p=e;
        long mode=strtol(p,&e,10); if(e==p){ fprintf(stderr,"[ablate] bad mode\n"); continue; } p=e;
        long nc=strtol(p,&e,10);  if(e==p){ fprintf(stderr,"[ablate] bad ncells\n"); continue; } p=e;
        int Ls[ABL_MAX_CELLS], Es[ABL_MAX_CELLS], As[ABL_MAX_CELLS];
        int bad=0; long ncc = nc<0?0:(nc>ABL_MAX_CELLS?ABL_MAX_CELLS:nc);
        for(long i=0;i<nc;i++){                                            /* read every triple; keep first ABL_MAX_CELLS */
            long L=strtol(p,&e,10); if(e==p){bad=1;break;} p=e;
            long E=strtol(p,&e,10); if(e==p){bad=1;break;} p=e;
            long A=strtol(p,&e,10); if(e==p){bad=1;break;} p=e;
            if(i<ncc){ Ls[i]=(int)L; Es[i]=(int)E; As[i]=(int)A; }
        }
        bad = bad || (T<1 || T>maxT || np<0 || np>T || mode<0 || mode>3);
        for(long i=0;i<T && !bad;i++){ long v=strtol(p,&e,10); if(e==p){bad=1;break;} p=e;
            if(v<0 || v>=V) bad=1; else ids[i]=(int)v; }
        if(bad){ fprintf(stderr,"[ablate] ERR item %ld (bad field/token)\n",item); continue; }
        /* PER-ITEM RESET then configure this item's ablation (no cross-item leak). */
        abl_reset(&g_abl);
        abl_set_item(&g_abl, (int)mode, Ls, Es, (mode==3?As:NULL), (int)ncc);
        if(of){
            fprintf(of,"{\"t\":\"ah\",\"item\":%ld,\"mode\":%ld,\"ncells\":%ld,\"T\":%ld,\"n_prompt\":%ld,\"cells\":[",
                    item, mode, ncc, T, np);
            for(int i=0;i<(int)ncc;i++) fprintf(of,"%s[%d,%d,%d]", i?",":"", Ls[i],Es[i],As[i]);
            fprintf(of,"]}\n");
        }
        for(int s=0;s<T;s++) embed_row(m, ids[s], x+(int64_t)s*D);
        layers_forward(m,x,(int)T,0);                                      /* ONE prefill; moe() applies g_abl */
        /* FINAL-logit read-out at every target position pos in [np-1, T-1). */
        if(of) for(long pos=(np>0?np-1:0); pos<T-1; pos++){
            rmsnorm(row, x+(int64_t)pos*D, m->final_norm, D, c->eps);
            matmul_qt(lo, row, &m->lm_head, 1);
            int gold=ids[pos+1];
            float mx=lo[0]; int am=0;
            for(int i=1;i<V;i++){ if(lo[i]>mx){mx=lo[i];am=i;} }
            double se=0; for(int i=0;i<V;i++) se+=exp((double)lo[i]-mx);
            double logZ=(double)mx+log(se);
            double gnll=logZ-(double)lo[gold];                             /* -log p(gold) */
            float molo=-1e30f; for(int i=0;i<V;i++){ if(i!=gold && lo[i]>molo) molo=lo[i]; }
            float mgn=lo[gold]-molo;                                       /* gold-vs-best-competitor logit margin */
            for(int k=0;k<ABL_LOGIT_TOPK;k++){ tk_id[k]=-1; tk_val[k]=-1e30f; }
            for(int i=0;i<V;i++){ float v=lo[i];
                int mn=0; for(int k=1;k<ABL_LOGIT_TOPK;k++) if(tk_val[k]<tk_val[mn]) mn=k;
                if(v>tk_val[mn]){ tk_val[mn]=v; tk_id[mn]=i; } }
            fprintf(of,"{\"t\":\"lg\",\"item\":%ld,\"pos\":%ld,\"gold\":%d,\"nll\":%.6f,"
                       "\"glogit\":%.6g,\"molo\":%.6g,\"mgn\":%.6g,\"am\":%d,\"amlogit\":%.6g,"
                       "\"logZ\":%.6f,\"corr\":%d,\"tk\":[",
                    item, pos, gold, gnll, (double)lo[gold], (double)molo, (double)mgn,
                    am, (double)mx, logZ, (am==gold)?1:0);
            for(int k=0;k<ABL_LOGIT_TOPK;k++) fprintf(of,"%s[%d,%.5g]", k?",":"", tk_id[k], (double)tk_val[k]);
            fprintf(of,"]}\n");
        }
        if(of) fflush(of);
        if(++nreq%8==0) fprintf(stderr,"[ablate %d item | %.1fs | RSS %.2f GB | hit %.0f%%]\n",
            nreq, now_s()-t0, rss_gb(), (m->hits+m->miss)?100.0*m->hits/(m->hits+m->miss):0.0);
    }
    abl_reset(&g_abl);                                                     /* leave the engine in the OFF state */
    if(of){ fflush(of); fclose(of); }
    free(ln); free(ids); free(x); free(lo); free(row); fclose(f);
}

static void generate(Model *m, const int *prompt, int np, int n_new, int *out){
    kv_alloc(m,np+n_new+g_draft+2);
    for(int i=0;i<np;i++) out[i]=prompt[i];
    float *logit=step(m,prompt,np,0);
    EmitStore es={out+np,0};
    spec_decode(m,out,np,n_new,-1,logit,emit_store,&es,NULL,NULL);
}

static void profile_print(Model *m, double elapsed){
    double accounted=m->t_ewait+m->t_emm+m->t_attn+m->t_head;
    printf("PROFILE: expert-disk %.3fs service / %.3fs wait | expert-matmul %.3fs | attention %.3fs "
           "(including kvb %.3fs) | lm_head %.3fs | other %.3fs\n",
        edisk_s(),m->t_ewait,m->t_emm,m->t_attn,m->t_kvb,m->t_head,elapsed-accounted);
    printf("ATTENTION: projection/RoPE %.3fs | score-softmax-value %.3fs | output projection %.3fs\n",
        m->t_aproj,m->t_acore,m->t_aout);
    if(g_prof)printf("P0-EXEC: routed CPU %.3fs / %.2f GB/s (%llu row) | routed GPU critical %.3fs | router %.3fs | residual P2P %.3fs / %llu hop | orchestration %.3fs\n",
        m->t_ecpu,m->t_ecpu>0?m->cpu_expert_bytes/1e9/m->t_ecpu:0.0,
        (unsigned long long)m->cpu_expert_rows,m->t_egpu,m->t_route,m->t_p2p,(unsigned long long)m->n_p2p,
        elapsed-m->t_ewait-m->t_emm-m->t_attn-m->t_head-m->t_route-m->t_p2p>0?
        elapsed-m->t_ewait-m->t_emm-m->t_attn-m->t_head-m->t_route-m->t_p2p:0);
#ifdef COLI_VULKAN
    if(g_prof && g_vkb_blocks)
        printf("VK-BLOCK: %lld blocks | avg nvk %.2f (dev2 %.2f) / ncpu %.2f | classify %.3fs | issue %.3fs | take+acc %.3fs (take-wait %.3fs) | d2-iss wrk %.3fs join %.3fs | cpu exec %.3fs wait %.3fs\n",
            (long long)g_vkb_blocks,(double)g_vkb_nvk/g_vkb_blocks,(double)g_vkb_nvk2/g_vkb_blocks,(double)g_vkb_ncpu/g_vkb_blocks,
            g_vkb_cls,g_vkb_issue,g_vkb_acc,m->t_egpu,g_vkb_wrk,g_vkb_join,m->t_ecpu,m->t_ewait);
#endif
    if(g_mirror){
        double br[MIR_REPS]={0}, bt=0;
        for(int r=0;r<g_mir_nrep;r++){ br[r]=atomic_load_explicit(&g_mir_bytes[r],memory_order_relaxed)/1e9; bt+=br[r]; }
        printf("MIRROR: primary %.2f GB (%lld reads)",
            br[0],(long long)atomic_load_explicit(&g_mir_nread[0],memory_order_relaxed));
        for(int r=1;r<g_mir_nrep;r++)
            printf(" | mirror%d %.2f GB (%lld reads)",
                r,br[r],(long long)atomic_load_explicit(&g_mir_nread[r],memory_order_relaxed));
        printf(" — %.0f%% of expert bytes from the mirrors\n", bt>0?100.0*(bt-br[0])/bt:0.0);
    }
#ifdef COLI_METAL
    if(g_metal_enabled){ uint64_t ok=0,fb=0,ex=0; double su=0,gp=0,sc=0;
        coli_metal_moe_counts(&ok,&fb,&ex); coli_metal_moe_times(&su,&gp,&sc);
        { uint64_t aok=0; double aw=0,ak=0; coli_metal_attn_counts(&aok,&aw,&ak);
          if(aok){ double ks=0,gs=0; coli_metal_attn_lat(&ks,&gs);
          printf("METAL-ATTN: layer GPU %llu | gpu-wall %.2fs (kernel %.2fs | cpu-sched %.2fs gpu-sched %.2fs)\n",(unsigned long long)aok,aw,ak,ks,gs); } }
        printf("METAL: blocchi GPU %llu | fallback CPU %llu | expert su GPU %llu | setup %.2fs gpu-wall %.2fs (kernel %.2fs) scatter %.2fs\n",
               (unsigned long long)ok,(unsigned long long)fb,(unsigned long long)ex,su,gp,coli_metal_moe_kernel_time(),sc);
        { double rsf=0; if(coli_metal_resset_stats(&rsf))   /* E5: printed only when the gate is on */
            printf("METAL-RESSET: flush %.2fs (residency-set commit in moe_submit, outside setup/gpu-wall)\n",rsf); } }
#endif
}

static void profile_reset(Model *m){
    m->t_ewait=m->t_emm=m->t_attn=m->t_kvb=m->t_head=0;
    m->t_ecpu=m->t_egpu=m->t_route=m->t_p2p=0;m->n_p2p=0;
    m->cpu_expert_bytes=0;m->cpu_expert_rows=0;
#ifdef COLI_VULKAN
    g_vkb_cls=g_vkb_issue=g_vkb_acc=g_vkb_wrk=g_vkb_join=0; g_vkb_blocks=g_vkb_nvk=g_vkb_nvk2=g_vkb_ncpu=0;
#endif
    m->t_aproj=m->t_acore=m->t_aout=0;
    atomic_store_explicit(&g_edisk_ns,0,memory_order_relaxed);
}

/* PROF=1 report: forward-latency percentiles, expert I/O totals, phase shares
 * of wall time, and a plain-language verdict naming the knob most likely to
 * move tok/s on THIS machine with THIS config. `b` marks the window start
 * (serve mode reports per turn; batch modes snapshot right after reset). */
static int prof_cmp_d(const void *a,const void *b){
    double x=*(const double*)a, y=*(const double*)b; return (x>y)-(x<y); }
static void prof_report(Model *m, const ProfBase *b, double elapsed, int tokens, FILE *f){
    Cfg *c=&m->c; if(elapsed<1e-9) elapsed=1e-9;
    uint64_t nw=g_prof_nlat-b->nlat; if(nw>PROF_LAT_CAP) nw=PROF_LAT_CAP;  /* ring keeps the tail */
    uint64_t nfw=m->n_fw-b->n_fw, nem=m->n_emit-b->n_emit;
    if(nw){
        double *v=malloc((size_t)nw*sizeof(double));
        if(v){
            for(uint64_t i=0;i<nw;i++) v[i]=g_prof_lat[(g_prof_nlat-nw+i)%PROF_LAT_CAP];
            qsort(v,(size_t)nw,sizeof(double),prof_cmp_d);
            double p50=v[(nw-1)/2], p90=v[(uint64_t)((nw-1)*0.90)], p99=v[(uint64_t)((nw-1)*0.99)], mx=v[nw-1];
            fprintf(f,"[PROF] decode forwards: %llu | latency p50 %.1f ms | p90 %.1f ms | p99 %.1f ms | max %.1f ms | %.2f tok/forward\n",
                (unsigned long long)nfw,p50*1e3,p90*1e3,p99*1e3,mx*1e3,nfw?(double)nem/nfw:0.0);
            if(nw>=32 && p99>3*p50)
                fprintf(f,"[PROF] tail: p99 is %.1fx p50 — the slow forwards are cold-cache expert loads; "
                          "a warm-up turn or a pinned hot-store (PIN / AUTOPIN history) shrinks them\n",p99/p50);
            free(v);
        }
    }
    int64_t io=atomic_load_explicit(&g_prof_io,memory_order_relaxed)-b->io;
    uint64_t dh=m->hits-b->hits, dm=m->miss-b->miss, dq=m->ereq-b->ereq;
    double hitp=(dh+dm)?100.0*dh/(dh+dm):100.0;
    /* Per-row widths (#856): "resident experts: N (X GB)" is the line people quote
     * when they compare the engine's own accounting against free RAM. */
    int pinned=0,lru=0; double pin_b=0,lru_b=0;
    for(int i=0;i<=c->n_layers;i++){
        double w=(double)expert_bytes_row(m,i,m->ebits);
        if(m->npin){ pinned+=m->npin[i]; pin_b+=(double)m->npin[i]*w; }
        if(m->ecn){ lru+=m->ecn[i];      lru_b+=(double)m->ecn[i]*w; }
    }
    double io_w=m->t_ewait-b->ewait;    /* stall the compute thread felt */
    double io_svc=edisk_s()-b->edisk;   /* read service on the loading threads (overlaps compute) */
    uint64_t dhp=m->hit_pin-b->hit_pin, dhe=m->hit_ecache-b->hit_ecache;   /* split #336 */
    fprintf(f,"[PROF] expert I/O: %.3f GB fetched (%.1f MB/token, %.2f GB/s over the run%s) | "
              "hit %.1f%% (%llu pin + %llu lru / %llu load) | %.1f loads/token | %.1fs read service / %.1fs felt wait\n",
        io/1e9, tokens>0?io/1e6/tokens:0.0, io/1e9/elapsed,
        g_mmap?"; COLI_MMAP=1: page cache may serve part":"",
        hitp,(unsigned long long)dhp,(unsigned long long)dhe,(unsigned long long)dm, tokens>0?(double)dq/tokens:0.0,
        io_svc,io_w);
    /* DISK-CLASS: per-load cold/warm classification vs. which fd ACTUALLY served it.
     * Three per-class rates, labeled to keep the units unambiguous (ambiguous units
     * mislead -- measured lesson): GB/s-thread = bytes / thread-seconds (per-read
     * service rate, same convention as the read-service line above); GB/s-wall =
     * bytes / busy-wall (aggregate rate the disk actually delivered while >=1 load of
     * the class was in flight); avg-conc = thread-seconds / busy-wall (mean overlap
     * depth). disk-busy = combined busy-wall (either class in flight) as a share of
     * the profile window. */
    {
        uint64_t dcn[2], dcdn[2]; int64_t dcbytes[2], dcns[2], dcwall[2], wnow[2], dcwall_all, wall_now;
        dc_wall_read(wnow,&wall_now);
        for(int i=0;i<2;i++){
            dcn[i]=atomic_load_explicit(&g_dc_n[i],memory_order_relaxed)-b->dc_n[i];
            dcbytes[i]=atomic_load_explicit(&g_dc_bytes[i],memory_order_relaxed)-b->dc_bytes[i];
            dcns[i]=atomic_load_explicit(&g_dc_ns[i],memory_order_relaxed)-b->dc_ns[i];
            dcdn[i]=atomic_load_explicit(&g_dc_direct_n[i],memory_order_relaxed)-b->dc_direct_n[i];
            dcwall[i]=wnow[i]-b->dc_wall_ns[i];
        }
        dcwall_all=wall_now-b->dc_wall_all_ns;
        if(dcn[DC_COLD]+dcn[DC_WARM]){
            double ct=dcns[DC_COLD]>0?(double)dcbytes[DC_COLD]/dcns[DC_COLD]:0.0;      /* bytes/ns = GB/s */
            double wt=dcns[DC_WARM]>0?(double)dcbytes[DC_WARM]/dcns[DC_WARM]:0.0;
            double cw=dcwall[DC_COLD]>0?(double)dcbytes[DC_COLD]/dcwall[DC_COLD]:0.0;
            double ww=dcwall[DC_WARM]>0?(double)dcbytes[DC_WARM]/dcwall[DC_WARM]:0.0;
            double cc=dcwall[DC_COLD]>0?(double)dcns[DC_COLD]/dcwall[DC_COLD]:0.0;
            double wc=dcwall[DC_WARM]>0?(double)dcns[DC_WARM]/dcwall[DC_WARM]:0.0;
            fprintf(f,"[PROF] DISK-CLASS (recency split): cold %llu x %.2f GB @ %.2f GB/s-thread, wall %.1fs @ %.2f GB/s-wall, avg-conc %.1f (direct %llu/%llu) | "
                      "warm %llu x %.2f GB @ %.2f GB/s-thread, wall %.1fs @ %.2f GB/s-wall, avg-conc %.1f (direct %llu/%llu) | "
                      "disk-busy %.1fs (%.0f%% of window)\n",
                (unsigned long long)dcn[DC_COLD],dcbytes[DC_COLD]/1e9,ct,dcwall[DC_COLD]/1e9,cw,cc,
                (unsigned long long)dcdn[DC_COLD],(unsigned long long)dcn[DC_COLD],
                (unsigned long long)dcn[DC_WARM],dcbytes[DC_WARM]/1e9,wt,dcwall[DC_WARM]/1e9,ww,wc,
                (unsigned long long)dcdn[DC_WARM],(unsigned long long)dcn[DC_WARM],
                dcwall_all/1e9,100.0*(dcwall_all/1e9)/elapsed);
        }
    }
    fprintf(f,"[PROF] resident experts: %d pinned (%.1f GB) + %d in LRU (%.1f GB, cap %d/layer)\n",
        pinned,pin_b/1e9,lru,lru_b/1e9,m->ecap);
    double emm=m->t_emm-b->emm, ecpu=m->t_ecpu-b->ecpu, egpu=m->t_egpu-b->egpu;
    double route=m->t_route-b->route,p2p=m->t_p2p-b->p2p;
    uint64_t np2p=m->n_p2p-b->n_p2p;
    int64_t cpu_bytes=m->cpu_expert_bytes-b->cpu_bytes;
    uint64_t cpu_rows=m->cpu_expert_rows-b->cpu_rows;
    double attn=m->t_attn-b->attn, head=m->t_head-b->head;
    double other=elapsed-io_w-emm-attn-head-route-p2p; if(other<0) other=0;
    double f_io=io_w/elapsed, f_emm=emm/elapsed, f_attn=attn/elapsed;
    fprintf(f,"[PROF] time shares: expert-I/O %.0f%% | expert-matmul %.0f%% | attention %.0f%% | lm_head %.0f%% | other %.0f%%\n",
        100*f_io,100*f_emm,100*f_attn,100*head/elapsed,100*other/elapsed);
    double slow=ecpu>egpu?ecpu:egpu,fast=ecpu<egpu?ecpu:egpu;
    fprintf(f,"[PROF] P0 execution: routed CPU %.3fs / %.2f GB/s (%llu row) | routed GPU critical %.3fs | tier straggler %.2fx | "
              "router %.3fs | residual P2P %.3fs (%llu hop, %.3f ms/hop) | orchestration %.3fs\n",
        ecpu,ecpu>0?cpu_bytes/1e9/ecpu:0.0,(unsigned long long)cpu_rows,
        egpu,fast>1e-9?slow/fast:0.0,route,p2p,(unsigned long long)np2p,
        np2p?p2p*1e3/np2p:0.0,other);
    if(f_io>=0.30){
        fprintf(f,"[PROF] verdict: I/O-bound — %.0f%% of the time waits on expert reads (hit %.0f%%).",100*f_io,hitp);
        if(hitp<90) fprintf(f," More cache is the lever: raise RAM_GB (or add RAM).");
        else fprintf(f," The cache is already warm — the routed working set streams from disk; a faster disk or a bigger pinned tier (PIN_GB) is the lever.");
        if(!g_pipe) fprintf(f," Try PIPE=1 (overlap reads with matmul).");
        if(!g_direct) fprintf(f," On NVMe try DIRECT=1.");
        fprintf(f,"\n");
    } else if(f_emm>=0.40){
        fprintf(f,"[PROF] verdict: compute-bound in expert matmuls (%.0f%%) — more cores/threads help; keep IDOT=1, or move hot experts to a GPU tier (COLI_CUDA / COLI_METAL).%s\n",
            100*f_emm, g_mmap?" Note: with COLI_MMAP=1 page-fault I/O is accounted inside matmul.":"");
    } else if(f_attn>=0.35){
        fprintf(f,"[PROF] verdict: attention-bound (%.0f%%) — context length is the cost (DSA %s). A lower CTX helps if the workload allows.\n",
            100*f_attn, m->has_dsa?"on":"not available for this model");
    } else {
        fprintf(f,"[PROF] verdict: balanced — no phase dominates (I/O %.0f%%, matmul %.0f%%, attention %.0f%%); this config is a reasonable fit for this machine.\n",
            100*f_io,100*f_emm,100*f_attn);
    }
}

/* Fixed-token decode benchmark: prefill all but the prompt's last token, then
 * replay the oracle sequence one token at a time. CPU and CUDA therefore see
 * identical hidden-state inputs even if their argmax predictions differ. */
static void run_replay(Model *m, const int *full, int nfull, int np){
    if(np<2||nfull<=np){ fprintf(stderr,"REPLAY requires a non-empty prompt and continuation\n"); return; }
    kv_alloc(m,nfull+2);
    float *logit=step(m,full,np-1,0); free(logit);
    m->hits=m->miss=m->ereq=m->gpu_expert_calls=0; m->hit_pin=m->hit_ecache=0; m->hit_vk=0;
    profile_reset(m);
    ProfBase pb; prof_base(m,&pb);
    for(int r=0;r<MIR_REPS;r++){ atomic_store(&g_mir_bytes[r],0); atomic_store(&g_mir_nread[r],0); }
    double t0=now_s(); int steps=0;
    for(int i=np-1;i<nfull-1;i++){
        double tf0=g_prof?now_s():0;
        logit=step(m,full+i,1,i); free(logit); steps++;
        if(g_prof){ prof_lat(now_s()-tf0); m->n_fw++; m->n_emit++; }
    }
    double dt=now_s()-t0, tot=m->hits+m->miss;
    printf("REPLAY decode: %d tokens in %.3fs | %.2f tok/s | expert hit %.1f%%\n",
        steps,dt,steps/dt,tot?100.0*m->hits/tot:0.0);
    if(g_cp_enq) printf("couple: %ld cross-layer prefetch hints enqueued\n",g_cp_enq);
    profile_print(m,dt);
    if(g_prof) prof_report(m,&pb,dt,steps,stdout);
#ifdef COLI_CUDA
    if(m->gpu_expert_count) printf("CUDA expert tier: %d resident experts (%.2f GB) | %llu calls served from VRAM\n",
        m->gpu_expert_count,m->gpu_expert_bytes/1e9,(unsigned long long)m->gpu_expert_calls);
    if(g_cuda_enabled) cuda_stats_print();
#endif
}

/* generazione reale: tokenizza PROMPT, prefill + decode greedy con stop su EOS,
 * detokenizza e stampa il testo in streaming. */
static void run_text(Model *m, const char *snap, const char *prompt, int ngen){
    Cfg *c=&m->c; char tkp[2048]; snprintf(tkp,sizeof(tkp),"%s/tokenizer.json",snap);
    Tok T; tok_load(&T,tkp);
    int eos=tok_id_of(&T,"<|endoftext|>");
    stops_arm_tok(&m->c, eos, &T);
    grammar_setup(&g_grd,&T);                   /* metodo F: GRAMMAR=file.gbnf (#48) */
    if(g_temp<0) g_temp=0.7f;            /* auto: 0.7, NON l'1.0 ufficiale — la coda della
                                          * distribuzione int4 e' rumore di quantizzazione */
    int cap=(int)strlen(prompt)+16; int *pids=malloc((cap+2)*sizeof(int));
    int np=tok_encode(&T,prompt,(int)strlen(prompt),pids,cap);
    if(np<1){ fprintf(stderr,"prompt is empty after tokenization\n"); return; }
    /* GLM prefix (#108): every GLM training sequence starts [gMASK]<sop>; without it the
     * model is out-of-distribution and generates garbage. Serve/SCORE modes prepend it;
     * run_text must too. CHAT_TEMPLATE=0 disables (raw prompt, for debugging/non-GLM).
     * A prompt that already starts with the prefix passes through intact. */
    int templ=getenv("CHAT_TEMPLATE")?atoi(getenv("CHAT_TEMPLATE")):1;
    if(templ){
        int gmask=tok_id_of(&T,"[gMASK]"), sop=tok_id_of(&T,"<sop>");
        if(gmask>=0 && sop>=0 && (np<2 || pids[0]!=gmask || pids[1]!=sop)){
            memmove(pids+2,pids,np*sizeof(int));
            pids[0]=gmask; pids[1]=sop; np+=2;
        }
    }
    printf("prompt: %d tokens | generating up to %d (EOS stop=%d) | n-gram draft=%d\n", np, ngen, eos, g_draft);
    fputs(prompt,stdout); fflush(stdout);
    kv_alloc(m, np+ngen+g_draft+2);
    int *all=malloc((np+ngen+g_draft+2)*sizeof(int)); memcpy(all,pids,np*sizeof(int));
    double prefill_t=now_s();
    float *logit=step(m,pids,np,0);
    if(g_repin>0){
        m->n_emit=(uint64_t)g_repin;
        int limit=32;
#ifdef COLI_CUDA
        if(m->gpu_expert_count) limit=m->c.n_layers;
#endif
        repin_pass_limit(m,limit);                  /* prompt routing seeds every GPU layer */
    }
    prefill_t=now_s()-prefill_t;
    printf("PROFILO PREFILL (%.2fs):\n",prefill_t); profile_print(m,prefill_t);
    m->hits=m->miss=m->ereq=m->gpu_expert_calls=0; m->hit_pin=m->hit_ecache=0; m->hit_vk=0;
    m->n_emit=m->n_fw=0;
    g_last_repin=0;
    profile_reset(m);
    ProfBase pb; prof_base(m,&pb);
    double t=now_s();
    EmitStream es={&T,m,t,0,0};
    grammar_reset(&g_grd);
    int produced=spec_decode(m,all,np,ngen,eos,logit,emit_stream,&es,NULL,NULL);
    double dt=now_s()-t;
    double tot=m->hits+m->miss;
    int nsp=0; for(int i=0;i<c->n_layers;i++) if(m->L[i].sparse) nsp++;
    printf("\n---\nprefill %d tokens in %.2fs | decode %d tokens in %.2fs (%.2f tok/s) | "
           "expert hit rate %.1f%% (pin %.1f%% + lru %.1f%%%s) | RSS %.2f GB",     /* split #336 (+VK VRAM tier) */
        np,prefill_t,produced,dt,produced/dt,tot?100.0*m->hits/tot:0.0,
        tot?100.0*m->hit_pin/tot:0.0, tot?100.0*m->hit_ecache/tot:0.0,
        m->hit_vk?({ static char vkb[40]; snprintf(vkb,sizeof(vkb)," + vk %.1f%%",tot?100.0*m->hit_vk/tot:0.0); vkb; }):"",
        rss_gb());
    if(g_cache_route && m->route_slots)
        printf(" | swap %.1f%% (%llu/%llu)",
            100.0*m->route_swaps/m->route_slots,
            (unsigned long long)m->route_swaps,(unsigned long long)m->route_slots);
    if(m->route_agree_tot)
        printf(" | route_agree %.1f%% | route_kl %.4f",
            100.0*m->route_agree_hit/m->route_agree_tot,
            m->route_kl_n?m->route_kl_sum/(double)m->route_kl_n:0.0);
    printf("\n");
    printf("experts loaded/token: %.1f (per-layer %.2f across %d; baseline topk=%d) | TOPK=%d TOPP=%.2f",
        produced?(double)m->ereq/produced:0.0, (produced&&nsp)?(double)m->ereq/produced/nsp:0.0, nsp, c->topk, g_topk, g_topp);
    if(g_cache_route) printf(" | CACHE_ROUTE J=%d M=%d P=%.2f alpha=%.2f", g_route_j, g_route_m, g_route_p, g_route_alpha);
    if(g_expert_budget){
        printf(" | EXPERT_BUDGET=%d (dropped %lld experts, ~%.1f GB I/O saved)", g_expert_budget, (long long)g_budget_dropped, g_budget_dropped*18.9e6/1e9);
        if(g_budget_rescued) printf(" [%lld rescued: budget too tight, position would have had 0 routed experts]", (long long)g_budget_rescued);
    }
    printf("\n");
    printf("speculation: %.2f tokens/forward (%llu forwards per %llu tokens) | MTP acceptance %.0f%% (%llu/%llu)\n",
        m->n_fw?(double)m->n_emit/m->n_fw:1.0, (unsigned long long)m->n_fw, (unsigned long long)m->n_emit,
        m->mtp_prop?100.0*m->mtp_acc/m->mtp_prop:0.0, (unsigned long long)m->mtp_acc, (unsigned long long)m->mtp_prop);
    if(g_corp_prop) printf("corpus drafts: %.0f%% acceptance (%llu/%llu proposed from %ld frozen ids)\n",
        100.0*g_corp_acc/g_corp_prop, (unsigned long long)g_corp_acc,
        (unsigned long long)g_corp_prop, g_corp_n);
    if(g_cp_enq) printf("couple: %ld cross-layer prefetch hints enqueued\n", g_cp_enq);
    if(g_grd.prop) printf("grammar: %.0f%% acceptance (%llu/%llu forced drafts)\n",
        100.0*g_grd.acc/g_grd.prop, (unsigned long long)g_grd.acc, (unsigned long long)g_grd.prop);
    if(g_disk_split) printf("disk-load split: draft %llu + absorb %llu + verify/main %llu misses | "
           "MTP-layer %llu loads %.2f GB | main-layers %llu loads %.2f GB (MTP %.1f%% of bytes)\n",
        (unsigned long long)m->miss_draft, (unsigned long long)m->miss_absorb,
        (unsigned long long)(m->miss - m->miss_draft - m->miss_absorb),
        (unsigned long long)m->ld_mtp, m->bytes_mtp/1e9,
        (unsigned long long)m->ld_main, m->bytes_main/1e9,
        (m->bytes_mtp+m->bytes_main)?100.0*m->bytes_mtp/(m->bytes_mtp+m->bytes_main):0.0);
#ifdef COLI_CUDA
    if(m->gpu_expert_count) printf("CUDA expert tier: %d resident experts (%.2f GB) | %llu calls served from VRAM\n",
        m->gpu_expert_count,m->gpu_expert_bytes/1e9,(unsigned long long)m->gpu_expert_calls);
    if(g_cuda_enabled) cuda_stats_print();
#endif
    profile_print(m,dt);
    if(g_prof) prof_report(m,&pb,dt,produced,stdout);
    if(g_pilot_real) printf("PILOT_REAL: %ld load cross-layer completati, %ld scartati (main gia' sul layer) | PILOT_K=%d\n",
        (long)atomic_load_explicit(&g_pilot_loads,memory_order_relaxed),
        (long)atomic_load_explicit(&g_pilot_drops,memory_order_relaxed), g_pilot_k);
    if(g_pilot_two) printf("PILOT_TWO: two-step shared-expert-corrected prefetch active (3 extra matmuls/prediction)\n");
    if(g_looka){
        const char *nm[4]={"previous token (=SPEC prefetch)","layer input, skip attention","next layer (PILOT, stale)","next layer (two-step, shared-expert)"};
        printf("LOOKAHEAD routing — recall of true experts in predicted top-8:\n");
        for(int i=0;i<4;i++) printf("  %-42s %5.1f%%  (%lld/%lld)\n", nm[i],
            la_tot[i]?100.0*la_hit[i]/la_tot[i]:0.0, (long long)la_hit[i], (long long)la_tot[i]);
    }
    /* TOKENS=1: dump the generated token ids (newline-separated) to stderr,
     * for exact A/B comparison across decode paths (e.g. resident vs CPU).
     * The ids are all[np .. np+produced-1]. */
    if(getenv("TOKENS") && atoi(getenv("TOKENS"))){
        fprintf(stderr,"[PROMPT_TOKENS] %d:",np);
        for(int i=0;i<np;i++) fprintf(stderr," %d",all[i]);
        fprintf(stderr,"\n");
        fprintf(stderr,"[TOKENS] %d generated:",produced);
        for(int i=np;i<np+produced;i++) fprintf(stderr," %d",all[i]);
        fprintf(stderr,"\n");
    }
    free(pids); free(all);
    usage_save(m);
}

/* modalita' SERVE (per la CLI 'coli'): carica il modello UNA volta, poi CHAT conversazionale.
 * KV-cache PERSISTENTE tra i turni: la storia resta in cache, si fa il prefill solo dei
 * token NUOVI -> il modello RICORDA la conversazione e non ri-processa il passato (lossless,
 * piu' umano, piu' veloce). Template chat GLM con token speciali (CHAT_TEMPLATE=0 -> grezzo).
 * Protocollo: "\x01\x01" "READY" "\x01\x01\n" dopo il load; risposta in streaming; "\x01\x01" "END" "\x01\x01\n" a fine turno.
 * ":reset" (riga "\x02RESET") azzera la memoria. EOF -> esce. */
/* ---- RFC: RE-PIN A CALDO / LIVE RE-PIN (opt-in, REPIN=n, default OFF) ----
 * Upstream fa AUTOPIN allo START (dalla storia .coli_usage). Questo aggiunge un re-pin
 * TRA I TURNI: nel punto sicuro dopo la risposta scambia i pin peggiori con i non-pinnati
 * piu' caldi, cosi' l'hot-store insegue il carico VIVO senza un profilo a parte. Isteresi
 * 25% (+4) contro il ping-pong; max 4 scambi/passata (~20 MB di disco l'uno). Una heat
 * map separata decade a ogni passata: la storia persistente .coli_usage resta intatta.
 * EN: upstream AUTOPINs at START (from .coli_usage). This adds a between-turns re-pin: at
 * the safe point after the reply, swap the worst pins for the hottest unpinned, so the
 * hot-store tracks the LIVE workload without a separate profile. 25% (+4) hysteresis vs
 * ping-pong; max 4 swaps/pass (~20 MB disk each). A separate decaying heat map keeps
 * persistent .coli_usage intact while adapting to the current workload. */
typedef struct { long gain; int l, slot, eid, gpu_swap; } RepinCand;
static int repin_pick(Model *m, RepinCand *out, int maxc){
    Cfg *c=&m->c; int nb=0;
    for(int l=0;l<c->n_layers;l++){
        if(!m->npin || m->npin[l]<1 || !m->eheat[l]) continue;
#ifdef COLI_CUDA
        int cold=-1,hot=-1;
        for(int z=0;z<m->npin[l];z++){
            ESlot *s=&m->pin[l][z]; uint32_t heat=m->eheat[l][s->eid];
            if(s->g.cuda_eligible){
                if(cold<0||heat<m->eheat[l][m->pin[l][cold].eid]) cold=z;
            }else if(hot<0||heat>m->eheat[l][m->pin[l][hot].eid]) hot=z;
        }
        if(cold>=0&&hot>=0){
            uint32_t ch=m->eheat[l][m->pin[l][cold].eid],hh=m->eheat[l][m->pin[l][hot].eid];
            if(hh>ch+1){
                RepinCand v={(long)hh-(long)ch,l,cold,m->pin[l][hot].eid,1};
                if(nb<maxc) out[nb++]=v;
                else { int w=0; for(int b=1;b<maxc;b++) if(out[b].gain<out[w].gain)w=b;
                       if(v.gain>out[w].gain)out[w]=v; }
                continue;
            }
        }
#endif
        ESlot *P=m->pin[l]; int ids[4096], zp, eu; long g;
        int np=m->npin[l]; if(np>4096) np=4096;
        for(int z=0;z<np;z++) ids[z]=P[z].eid;
        if(!tier_pick_lfru(m->eheat[l],m->elast[l],m->eaccess_clock,
                           c->n_experts,ids,np,&zp,&eu,&g)) continue;
        if(nb<maxc){ out[nb]=(RepinCand){g,l,zp,eu,0}; nb++; }
        else { int w=0; for(int b=1;b<maxc;b++) if(out[b].gain<out[w].gain) w=b;
               if(g>out[w].gain) out[w]=(RepinCand){g,l,zp,eu,0}; }
    }
    return nb;
}
/* ---- RSS GUARD (#403) -----------------------------------------------------
 * La proiezione di cap_for_ram e' una STIMA: sul GB10 (#403) le generazioni
 * lunghe l'hanno sforata di ~40 GB (proiettato 74.4, reale 115.6 -> 3 kill del
 * kernel). La run D dell'issue prova che un cap piu' basso CONTIENE la crescita:
 * questa guardia lo fa da sola, sull'RSS MISURATO invece che sul proiettato.
 * Al safe point (stessa sede di repin: nessun moe in volo), ogni ~16 token
 * emessi: se l'RSS supera il budget, svuota gli slot LRU meno usati e abbassa
 * ecap perche' non ricrescano. Gli slab sono >128KB (mmap'd da glibc): la free
 * restituisce le pagine al kernel subito, quindi l'RSS scende davvero.
 * Lo slot NON viene compattato via: resta al suo posto con eid=-1/used=0 (primo
 * candidato al riuso), perche' con PILOT_REAL il worker tiene puntatori dentro
 * ecache[] durante i suoi pread e uno spostamento li invaliderebbe; per lo
 * stesso motivo gli slot eid<0 (riservati/in caricamento) non si toccano e la
 * selezione avviene sotto g_pilot_mx. resident_bytes resta invariato: gli slot
 * LRU non sono mai contati li' (solo pin e densa).
 * EN: evict = free the slab in place (eid=-1, used=0, never compact: PILOT_REAL
 * EN: holds pointers into ecache[] across its preads), skip eid<0 reservations,
 * EN: select under g_pilot_mx. RSS_GUARD_GB=<gb> forces an explicit ceiling. */
static double g_ram_budget_gb=0;              /* budget risolto, scritto da cap_for_ram */
static uint64_t g_rssg_last=0;
static void rss_guard(Model *m){
    double lim = getenv("RSS_GUARD_GB") ? atof(getenv("RSS_GUARD_GB")) : g_ram_budget_gb;
    if(lim<=0) return;
    if(m->n_emit - g_rssg_last < 16) return;
    g_rssg_last = m->n_emit;
#ifdef __linux__
    double rss=current_rss_gb();
#else
    double rss=rss_gb();
#endif
    if(rss <= lim*1.02+0.3) return;                       /* tolleranza: 2% + 300MB */
    Cfg *c=&m->c;
    int64_t need=(int64_t)((rss-lim)*1e9), freed=0; int dropped=0;
    for(int pass=0; pass<8 && freed<need; pass++){
        for(int l=0; l<=c->n_layers && freed<need; l++){
            if(!m->ecache || !m->ecache[l]) continue;
            pthread_mutex_lock(&g_pilot_mx);
            int nn=m->ecn[l], lru=-1;
            for(int z=0;z<nn;z++){                        /* solo slot pubblicati e con slab */
                ESlot *cand=&m->ecache[l][z];
                if(cand->eid<0 || !cand->slab || eslot_busy(cand)) continue;
                if(lru<0 || cand->used<m->ecache[l][lru].used) lru=z;
            }
            if(lru<0){ pthread_mutex_unlock(&g_pilot_mx); continue; }
            ESlot *s=&m->ecache[l][lru];
            ecache_hide(m,l,s);                            /* nascosto: nessun hit/evict altrui */
            int64_t sb=s->slab_cap + s->fslab_cap*4;
#ifdef COLI_METAL
            if(s->slab && g_metal_enabled) coli_metal_unregister(s->slab);
#endif
            /* La free resta SOTTO g_pilot_mx. Sbloccando prima, lo slot e' visibile come
             * {eid=-1, slab ancora valido}: il pilota (pilot_realload) riusa per primo gli
             * slot eid==-1, quindi puo' prenderlo e fare pread dentro lo slab MENTRE lo
             * liberiamo -> use-after-free / double-free. Slab valido e slot riusabile
             * devono restare mutuamente esclusivi finche' il puntatore non e' NULL. */
            compat_aligned_free(s->slab); free(s->fslab);
            s->slab=NULL; s->fslab=NULL; s->slab_cap=s->fslab_cap=0;
            QT *q[3]={&s->g,&s->u,&s->d};
            for(int k=0;k<3;k++){ q[k]->qf=NULL; q[k]->q8=NULL; q[k]->q4=NULL; q[k]->s=NULL; }
            s->used=0;                                    /* primo candidato al riuso */
            pthread_mutex_unlock(&g_pilot_mx);
            freed += sb; dropped++;
        }
        if(m->ecap>2) m->ecap--;                           /* il tetto scende: niente ricrescita */
    }
    if(dropped)
        fprintf(stderr,"[RAM-GUARD] RSS %.1f GB over the %.1f GB budget (#403): "
                       "dropped %d cached experts, cap -> %d\n", rss, lim, dropped, m->ecap);
#ifdef __linux__
    malloc_trim(1024);
#endif
}
static void repin_pass_limit(Model *m,int limit){
    rss_guard(m);                     /* #403: il budget si fa rispettare sull'RSS MISURATO */
    if(g_repin<=0) return;
    if(m->n_emit - g_last_repin < (uint64_t)g_repin) return;
    g_last_repin = m->n_emit;
    double pass_t0=now_s(); int gpu_swaps=0;
    RepinCand cd[130];
    if(limit<1) limit=1; if(limit>130) limit=130;
    int nb=repin_pick(m,cd,limit);
#ifdef COLI_CUDA
    /* Cold GPU slots have no host backing. Restore all demoted experts in
     * parallel first; serial 20 MB reads made a 32-slot adaptation pass cost
     * ~0.7 s on the six-GPU host. */
    #pragma omp parallel for schedule(dynamic,1)
    for(int b=0;b<nb;b++) if(cd[b].gpu_swap){
        ESlot *s=&m->pin[cd[b].l][cd[b].slot];
        expert_host_ensure(m,cd[b].l,s);
    }
    for(int b=0;b<nb;b++) if(cd[b].gpu_swap){
        ESlot *s=&m->pin[cd[b].l][cd[b].slot];
        m->resident_bytes+=qt_bytes(&s->g)+qt_bytes(&s->u)+qt_bytes(&s->d);
    }
#endif
    for(int b=0;b<nb;b++){
        ESlot *s=&m->pin[cd[b].l][cd[b].slot];
        int old=s->eid;
        uint32_t old_heat=m->eheat[cd[b].l][old], new_heat=m->eheat[cd[b].l][cd[b].eid];
#ifdef COLI_CUDA
        if(cd[b].gpu_swap){
            ESlot *hot=pin_indexed(m,cd[b].l,cd[b].eid);
            if(!hot||hot->g.cuda_eligible) continue;
            double t0=now_s();
            QT *cq[3]={&s->g,&s->u,&s->d},*hq[3]={&hot->g,&hot->u,&hot->d};
            int ok=1;
            for(int k=0;k<3;k++){
                hq[k]->cuda=cq[k]->cuda; cq[k]->cuda=NULL;
                hq[k]->cuda_device=cq[k]->cuda_device;
                hq[k]->cuda_eligible=1; cq[k]->cuda_eligible=0;
                if(!qt_cuda_update(hq[k])) ok=0;
            }
            if(!ok){ fprintf(stderr,"[REPIN] refresh VRAM fallito\n"); exit(1); }
            /* promoted expert now computes from VRAM: drop its host mlock
             * (mmap path; no-op otherwise) or every swap leaks locked pages */
            qt_unwire_mmap(&hot->g); qt_unwire_mmap(&hot->u); qt_unwire_mmap(&hot->d);
            if(g_cuda_release_host) expert_host_release(m,hot);
            gpu_swaps++;
            if(getenv("REPIN_VERBOSE")) fprintf(stderr,
                "[REPIN] VRAM layer %d: esce/out %d (heat=%u) <- entra/in %d "
                "(heat=%u) in %.0f ms\n",cd[b].l,old,old_heat,cd[b].eid,new_heat,(now_s()-t0)*1e3);
            continue;
        }
        int gpu=s->g.cuda_eligible;
        int64_t old_gpu=gpu ? (int64_t)coli_cuda_tensor_bytes(s->g.cuda)
                             +(int64_t)coli_cuda_tensor_bytes(s->u.cuda)
                             +(int64_t)coli_cuda_tensor_bytes(s->d.cuda) : 0;
#endif
        double t0=now_s();
        pin_unindex(m,cd[b].l,s);
        expert_load(m,cd[b].l,cd[b].eid,s,1,0);     /* disk -> RAM, same resident slot; demand=0: repin, never classified */
        pin_index(m,cd[b].l,s);
        const char *tier="RAM";
#ifdef COLI_CUDA
        if(gpu){                                  /* refresh the same VRAM slot now, not lazily */
            if(qt_cuda_upload(&s->g) && qt_cuda_upload(&s->u) && qt_cuda_upload(&s->d)){
                int64_t now_gpu=(int64_t)coli_cuda_tensor_bytes(s->g.cuda)
                               +(int64_t)coli_cuda_tensor_bytes(s->u.cuda)
                               +(int64_t)coli_cuda_tensor_bytes(s->d.cuda);
                m->gpu_expert_bytes+=now_gpu-old_gpu; tier="VRAM";
                if(g_cuda_release_host) expert_host_release(m,s);
            } else {
                qt_cuda_reset(&s->g); qt_cuda_reset(&s->u); qt_cuda_reset(&s->d);
                s->g.cuda_eligible=s->u.cuda_eligible=s->d.cuda_eligible=0;
                m->gpu_expert_count--; m->gpu_expert_bytes-=old_gpu;
                fprintf(stderr,"[REPIN] VRAM upload failed; slot downgraded to RAM\n");
            }
        }
#endif
        fprintf(stderr,"[REPIN] %s layer %d: evict %d (heat=%u) <- admit %d (heat=%u) in %.0f ms\n",
            tier,cd[b].l,old,old_heat,cd[b].eid,new_heat,(now_s()-t0)*1e3);
    }
    if(gpu_swaps) fprintf(stderr,"[REPIN] VRAM: %d expert scambiati/swapped in %.0f ms\n",
        gpu_swaps,(now_s()-pass_t0)*1e3);
    for(int l=0;l<m->c.n_layers;l++) if(m->eheat[l]) tier_decay(m->eheat[l],m->c.n_experts);
}
/* ---- KV SU DISCO: la conversazione si riapre CALDA (KVSAVE=0 disattiva) ----
 * Il re-prefill di una chat riaperta costa ore su questo disco; la KV compressa MLA
 * costa ~182 KB/token. File <SNAP>/.coli_kv append-only: header (magic + dimensioni +
 * nrec) e un record per posizione [tok i32][Lc+Rc dei 78 layer][Ic DSA]. A fine turno
 * si appendono SOLO le posizioni nuove e si riscrive nrec per ultimo: un crash a meta'
 * append lascia nrec vecchio = file coerente. La riga KV del layer MTP non si salva:
 * al resume kv_start=-1 e la finestra di draft riparte da sola. */

typedef struct { KVState kv; int *hist, len, first; } ServeCtx;
static double kv_pool_bytes(Model *m, int max_ctx);

static void serve_ctx_init(Model *m, ServeCtx *s, const char *snap, int slot, int maxctx){
    s->kv.kv_start=calloc(m->c.n_layers+1,sizeof(int));
    if(m->has_mtp) s->kv.kv_start[m->c.n_layers]=-1;
    kv_bind(m,&s->kv); kv_alloc(m,maxctx);
    s->hist=malloc(maxctx*sizeof(int));
    if(!s->hist){ fprintf(stderr,"OOM serve_ctx_init hist\n"); exit(1); }
    s->first=1;
    if(slot==0) snprintf(s->kv.disk_path,sizeof(s->kv.disk_path),"%s/.coli_kv",snap);
    else snprintf(s->kv.disk_path,sizeof(s->kv.disk_path),"%s/.coli_kv.%d",snap,slot);
    s->len=kv_disk_load(m,s->hist,maxctx); if(s->len>0) s->first=0;
}

static void serve_ctx_free(Model *m, ServeCtx *s){
    KVState *k=&s->kv; int NR=m->c.n_layers+1;
    if(k->disk_fp){ fclose(k->disk_fp); k->disk_fp=NULL; }
    free(k->disk_buf); k->disk_buf=NULL;
    if(k->Lc) for(int i=0;i<NR;i++){ free(k->Lc[i]); free(k->Rc[i]); }
    if(k->Lc8) for(int i=0;i<NR;i++){ free(k->Lc8[i]); free(k->Rc8[i]);
        free(k->Lsc[i]); free(k->Rsc[i]); }
    if(k->Ic) for(int i=0;i<m->c.n_layers;i++) free(k->Ic[i]);
    free(k->Lc); free(k->Rc); free(k->Lc8); free(k->Rc8); free(k->Lsc); free(k->Rsc);
    free(k->Ic); free(k->kv_start); free(s->hist);
}

typedef struct {
    int active, pending, emitted, maximum, prompt_tokens, length_limited;
    int spec;                            /* single-slot speculation (#492): decode runs through
                                            spec_decode chunks instead of the shared batch */
    float *spec_logit;                   /* continuation logits between chunks; NULL = the last
                                            emitted token sits at hist[len], not yet forwarded */
    unsigned long long id;
    int logprobs;                        /* per-token numeric channel (U7a): requested
                                            top-k count from SUBMIT logprobs=k; 0 = off */
    float temp, top_p;
    double started;
    uint64_t hits0, miss0;
    ProfBase pb;                         /* phase-time window start (same convention as hits0):
                                            feeds the PROF protocol line and the PROF=1 report */
} ServeReq;

/* Numeric tail shared by opted-in DATA and ECHO frames (U7a): writes
 * " <lp> <k> [<tid> <tlp>]*k" into dst. lp = log-softmax of `token` over the
 * logit row; the top-k table selects by logit (identical order to selecting
 * by log-probability) and is UNSORTED, same as run_ablate_score's tk output.
 * The token's own entry, when it appears in the table, is the SAME double as
 * lp printed through the same format -- the bit-identity the harness's
 * is_greedy check depends on. lo==NULL (an echo's position 0: nothing to
 * condition on) writes " nan 0"; the server maps that to OpenAI's null.
 * k is capped by COLI_SUBMIT_TOPK_MAX (=32, run_ablate_score's ceiling). */
static int logprob_tail(char *dst, size_t cap, const float *lo, int V, int token, int topk){
    int tk_id[COLI_SUBMIT_TOPK_MAX]; float tk_val[COLI_SUBMIT_TOPK_MAX];
    int w;
    if(!lo || token<0 || token>=V) return snprintf(dst,cap," nan 0");
    if(topk>COLI_SUBMIT_TOPK_MAX) topk=COLI_SUBMIT_TOPK_MAX;
    if(topk>V) topk=V;
    float mx=lo[0]; for(int i=1;i<V;i++) if(lo[i]>mx) mx=lo[i];
    double se=0; for(int i=0;i<V;i++) se+=exp((double)lo[i]-mx);
    double logZ=(double)mx+log(se);
    for(int k=0;k<topk;k++){ tk_id[k]=-1; tk_val[k]=-1e30f; }
    for(int i=0;i<V;i++){ float v=lo[i];
        int mn=0; for(int k=1;k<topk;k++) if(tk_val[k]<tk_val[mn]) mn=k;
        if(v>tk_val[mn]){ tk_val[mn]=v; tk_id[mn]=i; } }
    w=snprintf(dst,cap," %.6f %d",(double)lo[token]-logZ,topk);
    for(int k=0;k<topk && w>0 && (size_t)w<cap;k++)
        w+=snprintf(dst+w,cap-(size_t)w," %d %.6f",tk_id[k],(double)tk_val[k]-logZ);
    return w;
}

/* One generated token -> one DATA frame. Opted-in requests (SUBMIT logprobs=k)
 * carry the numeric channel in the header: "DATA <id> <n> <lp> <k> [tid tlp]*k";
 * opt-out requests keep the exact legacy 3-field frame, byte for byte -- an old
 * server (whose dispatcher hard-fails on unknown framing) can only ever be
 * paired with requests that never opt in, so it never sees the extended form. */
static void mux_data(Tok *T, unsigned long long id, int token,
                     const float *lo, int V, int topk){
    char out[256]; int n=tok_decode(T,&token,1,out,sizeof(out));
    if(topk>0 && lo){
        char tail[1024]; logprob_tail(tail,sizeof(tail),lo,V,token,topk);
        printf("DATA %llu %d%s\n",id,n,tail);
    } else
        printf("DATA %llu %d\n",id,n);
    if(n>0) fwrite(out,1,(size_t)n,stdout); putchar('\n');
    fflush(stdout);
}

/* Echoed-prompt read-out frame (U7a, opt-in only): "ECHO <id> <n> <pos> <lp>
 * <k> [tid tlp]*k" + payload framed exactly like DATA (n bytes + '\n').
 * One frame per prompt position, in position order, BEFORE any DATA frame;
 * position 0 carries " nan 0" (nothing to condition on). Never emitted unless
 * the request set SUBMIT logprobs=k, so an old server can never receive one. */
static void mux_echo(Tok *T, unsigned long long id, int pos, int token,
                     const float *lo, int V, int topk){
    char out[256]; int n=tok_decode(T,&token,1,out,sizeof(out));
    char tail[1024]; logprob_tail(tail,sizeof(tail),lo,V,token,topk);
    printf("ECHO %llu %d %d%s\n",id,n,pos,tail);
    if(n>0) fwrite(out,1,(size_t)n,stdout); putchar('\n');
    fflush(stdout);
}

/* Opt-in prefill read-out (U7a, SUBMIT logprobs=k): one full-prompt forward
 * whose per-position hidden states are read out through lm_head -- one ECHO
 * frame per prompt position. Structurally run_score's loop, but live inside
 * the serve path and gated per request instead of a launch-env-gated
 * exit-early mode. Cost: one vocab-sized lm_head matmul per prompt position
 * (P x vocab, P = prompt length), paid ONLY by requests that opted in; the
 * opt-out path keeps step()'s single last-position lm_head untouched.
 * Returns the last position's logits (step()'s contract) as the generation
 * continuation. The caller resets the slot to len 0 first: the read-out
 * needs logits at EVERY position, so this path takes no cached-prefix skip
 * and no cross-slot KV adoption (KV rows [0,nt) are rewritten in full). */
static float *mux_prefill_echo(Model *m, Tok *T, unsigned long long id,
                               const int *ids, int nt, int topk){
    Cfg *c=&m->c; int D=c->hidden, V=c->vocab;
    float *x=falloc((int64_t)nt*D);
    for(int s=0;s<nt;s++) embed_row(m, ids[s], x+(int64_t)s*D);
    layers_forward(m,x,nt,0);
    if(m->hlast) memcpy(m->hlast, x+(int64_t)(nt-1)*D, D*sizeof(float));
    if(m->has_mtp && nt>=2 && g_draft>0) mtp_absorb(m, ids+1, x, nt-1, 0);  /* same as step() */
    float *lo=falloc(V), *row=falloc(D);
    mux_echo(T,id,0,ids[0],NULL,V,0);
    double th0=now_s();
    for(int pos=1; pos<nt; pos++){
        rmsnorm(row, x+(int64_t)(pos-1)*D, m->final_norm, D, c->eps);
        matmul_qt(lo, row, &m->lm_head, 1);
        mux_echo(T,id,pos,ids[pos],lo,V,topk);
    }
    rmsnorm(row, x+(int64_t)(nt-1)*D, m->final_norm, D, c->eps);
    matmul_qt(lo, row, &m->lm_head, 1);
    m->t_head += now_s()-th0;
    free(x); free(row);
    return lo;                           /* last position: the generation continuation */
}

/* #678: non-blocking stdin poll while a single-slot spec turn is running. Consumes
 * pending STOP/CANCEL lines (raising g_mux_stop/g_mux_cancel for the active id) so
 * the server's stop takes effect within ~1 token instead of after max_tokens. The
 * framed protocol stays in sync: STOP/CANCEL are single lines; a SUBMIT cannot
 * legally arrive while the only slot is busy, but if one does its payload is
 * drained and it is refused with SLOT_BUSY exactly as mux_submit would. On stdin
 * types the platform cannot poll (e.g. Windows non-pipe stdin), ready stays 0 and
 * behaviour falls back to the pre-#678 end-of-turn handling. */
static void mux_ctl_poll(unsigned long long id){
    for(;;){
        int ready=0;
#if defined(__APPLE__) || defined(__linux__) || defined(__FreeBSD__)
        fd_set rfds; FD_ZERO(&rfds); FD_SET(STDIN_FILENO,&rfds);
        struct timeval tv={0,0};
        ready = select(STDIN_FILENO+1,&rfds,NULL,NULL,&tv)>0 && FD_ISSET(STDIN_FILENO,&rfds);
#elif defined(_WIN32)
        HANDLE ih=(HANDLE)_get_osfhandle(_fileno(stdin));
        DWORD avail=0;
        ready=(PeekNamedPipe(ih,NULL,0,NULL,&avail,NULL) && avail>0)?1:0;
#endif
        if(!ready) return;
        char *line=NULL; size_t cap=0; ssize_t nr=getline(&line,&cap,stdin);
        if(nr<0){ free(line); return; }
        if(nr && line[nr-1]=='\n') line[--nr]=0;
        unsigned long long cid=0; char tail;
        if(!strncmp(line,"STOP ",5) && sscanf(line+5,"%llu %c",&cid,&tail)==1 && cid){
            if(cid==id) g_mux_stop=1;
            else { printf("ERROR %llu NOT_FOUND\n",cid); fflush(stdout); }
        }else if(!strncmp(line,"CANCEL ",7) && sscanf(line+7,"%llu %c",&cid,&tail)==1 && cid){
            if(cid==id) g_mux_cancel=1;
            else { printf("ERROR %llu NOT_FOUND\n",cid); fflush(stdout); }
        }else{
            ColiSubmit sub;
            if(coli_submit_parse(line,&sub)){
                long long left=(long long)sub.bytes+(long long)sub.gbytes;   /* drain payload */
                char buf[4096];
                while(left>0){
                    size_t take = left>(long long)sizeof(buf) ? sizeof(buf) : (size_t)left;
                    size_t got=fread(buf,1,take,stdin); if(!got) break; left-=(long long)got;
                }
                int delim=fgetc(stdin); (void)delim;                          /* trailing '\n' */
                printf("ERROR %llu SLOT_BUSY\n",sub.id); fflush(stdout);
            }else{
                printf("ERROR 0 BAD_REQUEST\n"); fflush(stdout);
            }
        }
        free(line);
    }
}

/* emit callback for the single-slot speculative path: stream straight to the mux
 * protocol. `lo` is the row the token was picked/verified from (spec_decode's
 * emit contract) -- accepted draft tokens included, so opted-in requests get a
 * numeric value on EVERY generated token, with no speculative-path gap. */
typedef struct { Tok *T; unsigned long long id; int V, logprobs; } MuxEmit;
static void mux_spec_emit(int t, const float *lo, void *ud){
    MuxEmit *e=(MuxEmit*)ud; mux_data(e->T,e->id,t,lo,e->V,e->logprobs);
    mux_ctl_poll(e->id);                 /* #678: honor STOP/CANCEL within ~1 token */
}

static void mux_done(Model *m, ServeCtx *sc, ServeReq *r){
    double dt=now_s()-r->started; if(dt<1e-6) dt=1e-6;
    double dh=(double)(m->hits-r->hits0), dm=(double)(m->miss-r->miss0);
    hwinfo_emit(m);
    usage_save(m);                       /* la cache che impara non deve aspettare l'uscita */
    tiers_emit(m);
    emap_emit(m);
    hits_emit(m);
    /* PROF: per-turn phase timings for the dashboard profiling page —
     * "PROF <wall_s> <prompt> <completion> <edisk> <ewait> <emm> <attn> <head> <n_fw>".
     * edisk = disk service (expert_load wall on the reading threads, overlaps
     * compute); ewait = the stall the compute thread felt — only ewait belongs
     * in a wall-time breakdown. With KV_SLOTS>1 concurrent slots share the
     * batched forwards, so the shares describe the whole engine over the
     * window, not the single request (same convention as the STAT hit% below). */
    printf("PROF %.3f %d %d %.3f %.3f %.3f %.3f %.3f %llu\n",dt,
           r->prompt_tokens,r->emitted,
           edisk_s()-r->pb.edisk,m->t_ewait-r->pb.ewait,m->t_emm-r->pb.emm,
           m->t_attn-r->pb.attn,m->t_head-r->pb.head,
           (unsigned long long)(m->n_fw-r->pb.n_fw));
    printf("DONE %llu STAT %d %.2f %.1f %.2f %d %d\n",r->id,r->emitted,
           r->emitted/dt,(dh+dm)>0?100.0*dh/(dh+dm):0.0,rss_gb(),
           r->prompt_tokens,r->length_limited);
    fflush(stdout); kv_bind(m,&sc->kv); kv_disk_append(m,sc->hist,sc->len);
    /* PROF window = this request's lifetime; with KV_SLOTS>1 concurrent slots
     * share the batched forwards, so the shares describe the engine, not the
     * single request (same convention as the STAT hit%% above). */
    if(g_prof) prof_report(m,&r->pb,dt,r->emitted,stderr);
    if(r->spec_logit){ free(r->spec_logit); r->spec_logit=NULL; }
    r->spec=0;
    r->active=0;
}

/* Read and prefill one request. Returns -1 on EOF, 0 for a rejected frame and
 * 1 for an accepted request. Prefill deliberately remains serial: continuous
 * batching starts at decode, where every active slot contributes one row. */
static int mux_submit(Model *m, Tok *T, ServeCtx *ctx, ServeReq *req, GrDraft *grd,
                      int nctx, int maxctx, int eos){
    char *line=NULL; size_t cap=0; ssize_t nr=getline(&line,&cap,stdin);
    if(nr<0){ free(line); return -1; }
    if(nr && line[nr-1]=='\n') line[--nr]=0;
    if(!strncmp(line,"STOP ",5)){
        unsigned long long id=0; char tail;
        if(sscanf(line+5,"%llu %c",&id,&tail)!=1 || id==0){
            printf("ERROR 0 BAD_REQUEST\n"); fflush(stdout); free(line); return 0;
        }
        for(int i=0;i<nctx;i++) if(req[i].active && req[i].id==id){
            mux_done(m,&ctx[i],&req[i]); free(line); return 0;
        }
        printf("ERROR %llu NOT_FOUND\n",id); fflush(stdout); free(line); return 0;
    }
    if(!strncmp(line,"CANCEL ",7)){
        unsigned long long id=0; char tail;
        if(sscanf(line+7,"%llu %c",&id,&tail)!=1 || id==0){
            printf("ERROR 0 BAD_REQUEST\n"); fflush(stdout); free(line); return 0;
        }
        for(int i=0;i<nctx;i++) if(req[i].active && req[i].id==id){
            req[i].active=0; kv_bind(m,&ctx[i].kv);
            if(req[i].spec_logit){ free(req[i].spec_logit); req[i].spec_logit=NULL; }
            req[i].spec=0;
            kv_disk_append(m,ctx[i].hist,ctx[i].len);
            printf("ERROR %llu CANCELLED\n",id); fflush(stdout); free(line); return 0;
        }
        printf("ERROR %llu NOT_FOUND\n",id); fflush(stdout); free(line); return 0;
    }
    ColiSubmit sub; int valid=coli_submit_parse(line,&sub);
    if(!valid){ printf("ERROR 0 BAD_REQUEST\n"); fflush(stdout); free(line); return 0; }
    char *raw=malloc((size_t)sub.bytes+1);
    if(!raw){ fprintf(stderr,"OOM multiplex payload\n"); exit(1); }
    if(fread(raw,1,(size_t)sub.bytes,stdin)!=(size_t)sub.bytes){ free(raw); free(line); return -1; }
    char *gtxt=NULL;                     /* optional per-request grammar/schema text */
    if(sub.gbytes){
        gtxt=malloc((size_t)sub.gbytes+1);
        if(!gtxt){ fprintf(stderr,"OOM multiplex payload\n"); exit(1); }
        if(fread(gtxt,1,(size_t)sub.gbytes,stdin)!=(size_t)sub.gbytes){
            free(gtxt); free(raw); free(line); return -1; }
        gtxt[sub.gbytes]=0;
    }
    int delim=fgetc(stdin);
    if(delim!='\n'){
        printf("ERROR %llu BAD_FRAME\n",sub.id); fflush(stdout);
        free(gtxt); free(raw); free(line); return -1;
    }
    raw[sub.bytes]=0;
    if(sub.slot>=nctx || memchr(raw,0,(size_t)sub.bytes)){
        printf("ERROR %llu BAD_REQUEST\n",sub.id); fflush(stdout); free(gtxt); free(raw); free(line); return 0;
    }
    if(req[sub.slot].active){
        printf("ERROR %llu SLOT_BUSY\n",sub.id); fflush(stdout); free(gtxt); free(raw); free(line); return 0;
    }
    for(int i=0;i<nctx;i++) if(req[i].active && req[i].id==sub.id){
        printf("ERROR %llu DUPLICATE_ID\n",sub.id); fflush(stdout); free(gtxt); free(raw); free(line); return 0;
    }
    /* #7: if this slot's last request compiled the *same* schema/grammar text, skip the
     * schema->GBNF->parse->state-init pipeline (measurable for tool-loop agents that resend
     * one schema every turn) and just reset the walker -- which is exactly how
     * grammar_setup_text finishes. Byte-identical to the teardown+recompile path either way. */
    GrDraft *gd=&grd[sub.slot];
    if(gtxt && gd->on && gd->src && !strcmp(gd->src,gtxt)){
        grammar_reset(gd);               /* fresh walker state, same compiled grammar */
        free(gtxt);
    } else {
        grammar_teardown(gd);            /* new request owns the slot's grammar state */
        if(gtxt){
            size_t glen=(size_t)sub.gbytes;
            char *keep=malloc(glen+1);   /* setup_text frees gtxt; keep a copy as the cache key */
            if(keep){ memcpy(keep,gtxt,glen); keep[glen]=0; }
            if(grammar_setup_text(gd,T,gtxt,"request")==0) gd->src=keep;  /* fail-soft; owns gtxt */
            else free(keep);
        }
    }
    ServeCtx *sc=&ctx[sub.slot]; kv_bind(m,&sc->kv);
    int *tmp=malloc(maxctx*sizeof(int));
    if(!tmp){ fprintf(stderr,"OOM mux_submit tmp\n"); free(raw); free(line); exit(1); }
    /* Encode with one token of headroom (maxctx-1, buffer is maxctx) purely to DETECT overflow:
     * tok_encode stops dead at its cap and returns no signal that it ran out, so encoding at the
     * real limit silently truncates an over-long prompt to its first maxctx-2 tokens. That is the
     * #401 field failure: a coding client's system prompt alone exceeded CTX=4096, the tail --
     * the tool instructions and the actual user turn -- was dropped, the model saw a prompt cut
     * mid-markup and emitted a bare "<". Retries were identical because the surviving *head*
     * never changes when a client appends to the end (hence "prefill 0" on every retry).
     * Refuse loudly instead; the gateway turns this into a 400 context_length_exceeded. */
    int nt;
    if(sub.tok_ids){
        /* Pre-tokenized intake (SUBMIT ids=1, U7a): the payload is ASCII token
         * ids fed straight into the same tmp[] buffer the text path fills --
         * identical embedding/position machinery downstream, no tok_encode, no
         * detokenize/re-encode round trip. Same one-token headroom contract as
         * the text arm above (coli_ids_parse reports the cap on overflow). */
        nt=coli_ids_parse(raw,(size_t)sub.bytes,tmp,maxctx-1,m->c.vocab);
        if(nt<0){
            free(tmp); free(raw); free(line);
            printf("ERROR %llu BAD_REQUEST\n",sub.id); fflush(stdout); return 0;
        }
    } else
        nt=tok_encode(T,raw,(int)sub.bytes,tmp,maxctx-1);
    free(raw); free(line);
    if(nt<1){ free(tmp); printf("ERROR %llu EMPTY_PROMPT\n",sub.id); fflush(stdout); return 0; }
    if(nt>maxctx-2){
        free(tmp);
        fprintf(stderr,"[API] prompt does not fit: >=%d token, context is %d (CTX=%d). "
                       "Raise it, e.g. CTX=32768 -- coding clients send large system prompts.\n",
                nt,maxctx-2,maxctx);
        printf("ERROR %llu CONTEXT_EXCEEDED %d %d\n",sub.id,nt,maxctx-2);
        fflush(stdout); return 0;
    }
    /* #597: the submission is valid (encoded, non-empty, fits the context). Announce it before
     * prefill so the gateway commits the streaming 200 only now. Every failure above returned
     * first with an ERROR, so a request yields exactly one of ACCEPT or an early ERROR -- which
     * lets a CONTEXT_EXCEEDED become a clean HTTP 400 instead of a broken already-200 stream. */
    printf("ACCEPT %llu %d\n",sub.id,nt); fflush(stdout);
    /* Echo read-out (U7a): needs logits at EVERY prompt position, so the whole
     * prompt re-prefills from position 0 -- no cached-prefix skip and no
     * cross-slot adoption below (either would leave positions with no logits). */
    int echo = sub.logprobs>0;
    int prefix=0;
    if(!echo) while(prefix<sc->len && prefix<nt && sc->hist[prefix]==tmp[prefix]) prefix++;
    if(prefix<sc->len){ sc->len=prefix; if(m->has_mtp) m->kv_start[m->c.n_layers]=-1;
        kv_disk_truncate(m,sc->len); }
    /* Cross-slot prefix adoption (COLI_KV_SHARE=1) — RadixAttention's benefit
     * at memcpy cost: if another slot's history shares a longer prefix with
     * this prompt (shared system prompt, agent loop), copy its KV rows instead
     * of recomputing them. MLA rows are position-addressed and self-contained
     * (the same per-position serialization .coli_kv relies on), so a raw row
     * copy is exact — the adopted prefill is skipped, not approximated.
     * Copying beats aliasing here: step_decode_batch rejects two rows sharing
     * one KVState, and a copy keeps every slot's lifetime independent.
     * Measured (6x5090, 675-token shared prefix): slot TTFT 50.1s -> 1.7s,
     * generated tokens identical to the full-prefill run. */
    static int kvshare=-1;
    if(kvshare<0) kvshare=getenv("COLI_KV_SHARE")?atoi(getenv("COLI_KV_SHARE")):0;
    if(kvshare && !echo && nctx>1 && prefix>=sc->len){
        int best=-1, blen=sc->len;
        for(int i=0;i<nctx;i++){
            if(i==sub.slot) continue;
            ServeCtx *dc2=&ctx[i];
            int lim=dc2->len<nt?dc2->len:nt, p=0;
            while(p<lim && dc2->hist[p]==tmp[p]) p++;
            if(p>blen){ blen=p; best=i; }
        }
        if(best>=0){
            ServeCtx *dn=&ctx[best]; Cfg *cc=&m->c; int NR=cc->n_layers+1;
            int from=sc->len, n=blen-from;
            for(int l=0;l<NR;l++){
                if(sc->kv.Lc[l]&&dn->kv.Lc[l])
                    memcpy(coli_kv_row(sc->kv.Lc[l],from,cc->kv_lora),
                           coli_kv_row(dn->kv.Lc[l],from,cc->kv_lora),
                           (size_t)n*cc->kv_lora*sizeof(float));
                if(sc->kv.Rc[l]&&dn->kv.Rc[l])
                    memcpy(coli_kv_row(sc->kv.Rc[l],from,cc->qk_rope),
                           coli_kv_row(dn->kv.Rc[l],from,cc->qk_rope),
                           (size_t)n*cc->qk_rope*sizeof(float));
                if(sc->kv.Ic&&dn->kv.Ic&&sc->kv.Ic[l]&&dn->kv.Ic[l])
                    memcpy(coli_kv_row(sc->kv.Ic[l],from,cc->index_hd),
                           coli_kv_row(dn->kv.Ic[l],from,cc->index_hd),
                           (size_t)n*cc->index_hd*sizeof(float));
            }
            memcpy(sc->hist+from,tmp+from,(size_t)n*sizeof(int));
            sc->len=blen;
            if(m->has_mtp) m->kv_start[cc->n_layers]=-1;   /* MTP rows are per-slot decode state */
            fprintf(stderr,"[API] KV cross-slot adopt: slot %d took rows [%d,%d) from slot %d\n",
                    sub.slot,from,blen,best);
        }
    }
    int add=nt-sc->len;
    if(add>0) memcpy(sc->hist+sc->len,tmp+sc->len,(size_t)add*sizeof(int));
    fprintf(stderr,"[API] KV slot %d prefix %d/%d token, prefill %d\n",sub.slot,sc->len,nt,add);
    free(tmp);
    float *logit = echo ? mux_prefill_echo(m,T,sub.id,sc->hist,nt,sub.logprobs)
                        : add>0 ? step(m,sc->hist+sc->len,add,sc->len)
                                : step(m,sc->hist+sc->len-1,1,sc->len-1);
    sc->len+=add; sc->first=0;
    ServeReq *r=&req[sub.slot]; memset(r,0,sizeof(*r));
    r->id=sub.id; r->maximum=sub.max_tokens; r->temp=sub.temperature; r->top_p=sub.top_p;
    r->logprobs=sub.logprobs;
    r->prompt_tokens=nt; r->started=now_s(); r->hits0=m->hits; r->miss0=m->miss;
    prof_base(m,&r->pb);                 /* a few loads: cheap enough to always track */
    /* Clamp to the KV room WITHOUT flagging: length_limited must mean "the
     * limit is what stopped us", not "the request asked for more than the
     * context could ever hold" — clients that default max_tokens to the
     * context size would otherwise see finish_reason=length on every
     * naturally-completed generation. The flag is set at the actual
     * emitted>=maximum stops below instead. */
    int room=maxctx-sc->len-1; if(r->maximum>room) r->maximum=room;
    g_temp=r->temp; g_nuc=r->top_p;
    /* Single-slot speculation (#492/#358): with one KV slot there is no ragged
     * batch — the decode is a single contiguous sequence, exactly the regime
     * spec_decode already serves in chat/run/run_serve. Hand the request to the
     * scheduler's chunked spec path instead of the pending-token cycle; grammar
     * requests keep the forced-draft path (its acceptance is ~1 where it fires,
     * and the two draft sources would fight over the same forward). */
    if(g_draft>0 && nctx==1 && !grd[sub.slot].on){
        if(r->maximum<=0){ free(logit); mux_done(m,sc,r); return 1; }
        r->spec=1; r->spec_logit=logit; r->active=1;
        return 1;
    }
    int next=pick_tok(logit,m->c.vocab,-1);
    if(r->maximum<=0){ free(logit); r->length_limited=1; mux_done(m,sc,r); return 1; }   /* no room at all */
    if(next==eos || is_stop(next)){ free(logit); mux_done(m,sc,r); return 1; }
    r->pending=next; r->emitted=1; r->active=1; sc->hist[sc->len]=next; m->n_emit++;
    if(grd[sub.slot].on){ grammar_reset(&grd[sub.slot]); gr_feed(&grd[sub.slot],next); }
    mux_data(T,r->id,next,logit,m->c.vocab,r->logprobs);
    free(logit);
    if(r->emitted>=r->maximum){ r->length_limited=1; mux_done(m,sc,r); }
    return 1;
}

static void run_serve_mux(Model *m, const char *snap){
    char tkp[2048]; snprintf(tkp,sizeof(tkp),"%s/tokenizer.json",snap);
    Tok T; tok_load(&T,tkp); int eos=tok_id_of(&T,"<|endoftext|>"); stops_arm_tok(&m->c,eos,&T);
    int maxctx=getenv("CTX")?atoi(getenv("CTX")):4096;
    int nctx=getenv("KV_SLOTS")?atoi(getenv("KV_SLOTS")):1;
    if(nctx<1||nctx>512){fprintf(stderr,"KV_SLOTS must be between 1 and 512\n");exit(2);}
    /* MTP/n-gram speculation is not ragged-safe across KV slots, so multi-slot
     * serve keeps one scheduler owning every forward (g_draft=0). At KV_SLOTS=1
     * there IS no ragged batch — decode is one contiguous sequence, the exact
     * regime spec_decode already serves in chat/run/run_serve — so the engine's
     * resolved draft setting stays live (#492/#358). Grammar-forced drafts stay
     * mux-safe on every slot count, as before. */
    if(nctx>1) g_draft=0;
    else if(g_draft>0)
        fprintf(stderr,"[MTP] single-slot serve: speculation active (draft=%d)\n",g_draft);
    g_kvsave=getenv("KVSAVE")?atoi(getenv("KVSAVE")):1;
    KVState *initial=m->kv; free(initial->kv_start); free(initial);
    ServeCtx *ctx=calloc(nctx,sizeof(*ctx)); ServeReq *req=calloc(nctx,sizeof(*req));
    GrDraft *grd=calloc(nctx,sizeof(*grd));   /* per-slot request grammars (SUBMIT 7th field) */
    for(int i=0;i<nctx;i++){ serve_ctx_init(m,&ctx[i],snap,i,maxctx); grd[i].max=24; }
#ifdef _WIN32
    /* Same byte-exact protocol as run_serve: in TEXT mode the CRT collapses CRLF in
     * fread() payloads (waits forever for the missing bytes) and expands LF on the
     * way out (corrupting the READY/STAT sentinels). BINARY on both ends. (#195) */
    _setmode(_fileno(stdin),  _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
    setvbuf(stdout, NULL, _IONBF, 0);
#endif
    setvbuf(stdin,NULL,_IONBF,0);
    intr_install();                      /* Ctrl-C = chiudi i turni in volo, non il processo */
    printf("\x01\x01READY\x01\x01\nSTAT 0 0.00 0.0 %.2f\n",rss_gb()); fflush(stdout);
    hwinfo_emit(m);
    tiers_emit(m);
    emap_emit(m);
    int eof=0;
    for(;;){
        if(g_intr){ g_intr=0;            /* stop morbido: ogni request attiva finisce ORA per la
                                          * via normale di mux_done (DONE+stats+KV coerenti) */
            for(int i=0;i<nctx;i++) if(req[i].active) mux_done(m,&ctx[i],&req[i]);
        }
        /* #810: SIGTERM. Ordered AFTER the mux_done sweep above on purpose -- the
         * in-flight requests finish through their normal path first, then the loop
         * ends. With no active request select() blocks with a NULL timeout, so the
         * EINTR from an un-restarted SIGTERM is what wakes us to reach this line. */
        if(g_shutdown) break;
        int active=0; for(int i=0;i<nctx;i++) active+=req[i].active;
        /* Poll stdin for available input without blocking. On POSIX this is
         * select(); on Windows, select() on a pipe handle routes to winsock
         * and always returns -1 (SOCKET_ERROR), so the batch loop could never
         * accept a request (#139). PeekNamedPipe on the stdin OS handle is
         * the Windows equivalent: it reports bytes available without reading. */
        int ready=0;
        if(!eof){
#if defined(__APPLE__) || defined(__linux__) ||	defined(__FreeBSD__)
            fd_set rfds; FD_ZERO(&rfds); FD_SET(STDIN_FILENO,&rfds);
            struct timeval tv={0,0}, *ptv=active?&tv:NULL;
            ready=select(STDIN_FILENO+1,&rfds,NULL,NULL,ptv);
            if(ready>0 && FD_ISSET(STDIN_FILENO,&rfds))
#elif defined(_WIN32)
            HANDLE ih=(HANDLE)_get_osfhandle(_fileno(stdin));
            DWORD avail=0;
            /* Anonymous pipes are NOT waitable objects: WaitForSingleObject on them is
             * undefined (always-signaled or WAIT_FAILED), and PeekNamedPipe fails on
             * file/console handles — the old gate never dispatched (#195). New rule:
             * idle -> block in getline() inside mux_submit (same semantics as the
             * POSIX select(NULL)); active -> poll the pipe with PeekNamedPipe, and on
             * non-pipe stdin just defer submits until the batch finishes. */
            if(eof) ready=0;
            else if(!active) ready=1;
            else ready=(PeekNamedPipe(ih,NULL,0,NULL,&avail,NULL) && avail>0)?1:0;
            if(ready)
#endif
                if(mux_submit(m,&T,ctx,req,grd,nctx,maxctx,eos)<0) eof=1;
        }
        active=0; for(int i=0;i<nctx;i++) active+=req[i].active;
        if(!active){ if(eof) break; continue; }
        DecodeRow rows[512]; int slots[512], S=0;
        for(int i=0;i<nctx;i++) if(req[i].active){
            ServeCtx *sc=&ctx[i]; ServeReq *r=&req[i]; GrDraft *gd=&grd[i];
            if(r->spec){
                /* Single-slot speculative decode (#492): KV_SLOTS=1 has no ragged
                 * batch, so the whole turn runs through spec_decode in one call —
                 * the exact contract run_serve (non-mux) already uses. No chunk
                 * splicing (that would re-enter spec_decode mid-turn and double
                 * the boundary token); Ctrl-C interrupts through g_intr inside
                 * spec_decode, and a server STOP/CANCEL through g_mux_stop /
                 * g_mux_cancel raised by mux_ctl_poll in the emit callback (#678).
                 * r->spec_logit holds the prefill continuation from mux_submit. */
                kv_bind(m,&sc->kv);
                g_temp=r->temp; g_nuc=r->top_p;
                float *lg=r->spec_logit; r->spec_logit=NULL;   /* spec_decode takes ownership */
                MuxEmit ud={&T,r->id,m->c.vocab,r->logprobs};
                g_mux_stop=0; g_mux_cancel=0;                  /* fresh per turn (#678) */
                int prod=spec_decode(m,sc->hist,sc->len,r->maximum-r->emitted,eos,lg,
                                     mux_spec_emit,&ud,&sc->len,NULL);
                r->emitted+=prod;
                if(g_mux_cancel){
                    /* mirror of mux_submit's CANCEL epilogue: no DONE frame, the
                     * dispatcher expects ERROR <id> CANCELLED. KV history is kept
                     * consistent exactly like the between-turns path. */
                    g_mux_cancel=0; g_mux_stop=0;
                    r->spec=0; r->active=0;
                    kv_bind(m,&sc->kv);
                    kv_disk_append(m,sc->hist,sc->len);
                    printf("ERROR %llu CANCELLED\n",r->id); fflush(stdout);
                    continue;
                }
                g_mux_stop=0;                /* STOP or natural end: same DONE path */
                mux_done(m,sc,r);
                continue;                    /* whole turn handled outside the shared batch */
            }
            /* grammar-forced drafts (greedy requests only: verification under sampling
             * needs rejection resampling, out of scope here). The slot leaves the shared
             * batch for one forward and runs the single-sequence verify path. */
            int draft[49], k=0;
            if(gd->on && r->temp==0) k=grammar_draft(gd,draft,gd->max);
            if(k>0 && sc->len+1+k<maxctx-1){
                if(sc->len+1+k>=(int)sc->kv.max_t) k=(int)sc->kv.max_t-sc->len-2;
                if(k<1){ rows[S]=(DecodeRow){&sc->kv,r->pending,sc->len}; slots[S++]=i; continue; }
                kv_bind(m,&sc->kv);
                int seq[50]; seq[0]=r->pending;
                memcpy(seq+1,draft,(size_t)k*sizeof(int));
                float *lo=step_all(m,seq,1+k,sc->len); m->n_fw++;
                gd->prop+=(uint64_t)k;
                int done=0;
                for(int j=0;j<=k && !done;j++){
                    g_temp=r->temp; g_nuc=r->top_p;
                    int next=pick_tok(lo+(int64_t)j*m->c.vocab,m->c.vocab,-1);
                    sc->len++;                       /* seq[j] joins the committed history */
                    if(next==eos || is_stop(next)){ mux_done(m,sc,r); done=1; break; }
                    r->pending=next; sc->hist[sc->len]=next; r->emitted++; m->n_emit++;
                    if(gd->on) gr_feed(gd,next);
                    mux_data(&T,r->id,next,lo+(int64_t)j*m->c.vocab,m->c.vocab,r->logprobs);
                    if(r->emitted>=r->maximum){ r->length_limited=1; mux_done(m,sc,r); done=1; break; }
                    if(j<k){
                        if(next!=draft[j]) break;    /* rejected: seq[j+1..] stale, overwritten next forward */
                        gd->acc++;
                    }
                }
                free(lo);
                continue;                            /* handled outside the shared batch */
            }
            rows[S]=(DecodeRow){&sc->kv,r->pending,sc->len}; slots[S++]=i;
        }
        if(S==0) continue;               /* every active slot drafted this round */
        double tf0=g_prof?now_s():0;
        float *lo=step_decode_batch(m,rows,S); if(!lo){fprintf(stderr,"decode batch failed\n");break;}
        m->n_fw++;
        if(g_prof) prof_lat(now_s()-tf0);
        for(int s=0;s<S;s++){
            int i=slots[s]; ServeCtx *sc=&ctx[i]; ServeReq *r=&req[i];
            sc->len++; g_temp=r->temp; g_nuc=r->top_p;
            int next=pick_tok(lo+(int64_t)s*m->c.vocab,m->c.vocab,-1);
            if(next==eos || is_stop(next)){mux_done(m,sc,r);continue;}
            r->pending=next; sc->hist[sc->len]=next; r->emitted++; m->n_emit++;
            if(grd[i].on) gr_feed(&grd[i],next);   /* walker stays in sync when not drafting */
            mux_data(&T,r->id,next,lo+(int64_t)s*m->c.vocab,m->c.vocab,r->logprobs);
            if(r->emitted>=r->maximum){ r->length_limited=1; mux_done(m,sc,r); }
        }
        free(lo);
    }
    usage_save(m);
    for(int i=0;i<nctx;i++){ serve_ctx_free(m,&ctx[i]); grammar_teardown(&grd[i]); }
    free(ctx); free(req); free(grd);
    m->kv=NULL; m->Lc=m->Rc=m->Ic=NULL; m->Lc8=m->Rc8=NULL; m->Lsc=m->Rsc=NULL; m->kv_start=NULL; m->max_t=0;
}

static void run_serve(Model *m, const char *snap){
    /* Serve mode speaks a byte protocol over BOTH stdout and stdin:
     *   stdout: \x01\x01READY\x01\x01\n, STAT lines, \x01\x01END\x01\x01\n
     *   stdin:  text lines plus \x02RESET / \x02MORE control bytes.
     * 'coli' matches the sentinels with endswith() and a "^STAT ..." regex,
     * so they must arrive byte-exact (LF, no CR). On Windows the CRT opens
     * both handles in TEXT mode: stdout translates '\n'->'\r\n' (so the READY
     * sentinel never matches and chat hangs at ~10 GB resident), and stdin
     * translates '\r\n'->'\n' and rejects writes of raw bytes with EINVAL,
     * breaking the control protocol. Put BOTH handles in BINARY mode so the
     * protocol bytes are exact in both directions. No-op on Linux/macOS. */
#ifdef _WIN32
    _setmode(_fileno(stdin),  _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
    setvbuf(stdout, NULL, _IONBF, 0);
#endif
    double t_serve0=now_s();             /* PROF: wall base for the exit-time profile_print */
    char tkp[2048]; snprintf(tkp,sizeof(tkp),"%s/tokenizer.json",snap);
    Tok T; tok_load(&T,tkp);
    int eos=tok_id_of(&T,"<|endoftext|>");
    stops_arm_tok(&m->c, eos, &T);
    grammar_setup(&g_grd,&T);                   /* metodo F: GRAMMAR=file.gbnf (#48) */
    if(g_temp<0) g_temp=0.7f;            /* auto: 0.7, NON l'1.0 ufficiale — la coda della
                                          * distribuzione int4 e' rumore di quantizzazione */
    int ngen=getenv("NGEN")?atoi(getenv("NGEN")):256;
    int maxctx=getenv("CTX")?atoi(getenv("CTX")):4096;
    int templ=getenv("CHAT_TEMPLATE")?atoi(getenv("CHAT_TEMPLATE")):1;
    g_kvsave = getenv("KVSAVE")?atoi(getenv("KVSAVE")):1;
    int nctx=getenv("KV_SLOTS")?atoi(getenv("KV_SLOTS")):1;
    if(nctx<1||nctx>16){ fprintf(stderr,"KV_SLOTS must be between 1 and 16\n"); exit(2); }
    KVState *initial=m->kv; free(initial->kv_start); free(initial);
    ServeCtx *ctx=calloc(nctx,sizeof(ServeCtx));
    for(int i=0;i<nctx;i++) serve_ctx_init(m,&ctx[i],snap,i,maxctx);
    int active=0; ServeCtx *sc=&ctx[0]; kv_bind(m,&sc->kv);
    fprintf(stderr,"[KV] context slots: %d x %d tokens, projected pool %.2f GB\n",
        nctx,maxctx,kv_pool_bytes(m,maxctx)/1e9);
    #define hist  (sc->hist)
    #define len   (sc->len)
    #define first (sc->first)
    char *line=NULL; size_t cap=0; ssize_t nr; char *buf=malloc(1<<16);
    intr_install();                      /* Ctrl-C = fine turno, non fine processo */
    printf("\x01\x01" "READY" "\x01\x01\n"); printf("STAT 0 0.00 0.0 %.2f\n", rss_gb()); fflush(stdout);
    tiers_emit(m);
    while(!g_shutdown && (nr=getline(&line,&cap,stdin))>0){
        g_intr=0;                        /* interruzioni arrivate tra i turni: stantie */
        if(nr>0 && line[nr-1]=='\n') line[--nr]=0;
        if(!strcmp(line,"\x02RESET")){ len=0; first=1; if(m->has_mtp) m->kv_start[m->c.n_layers]=-1;
            kv_disk_reset(m);
            printf("\x01\x01" "END" "\x01\x01\n"); printf("STAT 0 0.00 0.0 %.2f\n", rss_gb()); fflush(stdout); continue; }
        if(!strcmp(line,"\x02MORE")){                /* continua la risposta troncata da NGEN:
            la storia e' gia' in KV, basta ri-forwardare l'ULTIMO token per riavere i logits */
            if(len<1){ printf("\x01\x01" "END" "\x01\x01\n"); printf("STAT 0 0.00 0.0 %.2f\n", rss_gb()); fflush(stdout); continue; }
            int cur=ngen; if(len+cur+g_draft+2>=maxctx) cur=maxctx-len-g_draft-2;
            uint64_t h0=m->hits, ms0=m->miss; double tt0=now_s();
            ProfBase pb; if(g_prof) prof_base(m,&pb);
            float *logit=step(m,hist+len-1,1,len-1);
            EmitStream es={&T,m,now_s(),0,1};
            int prod=0;
            if(cur>0) prod=spec_decode(m,hist,len,cur,eos,logit,emit_stream,&es,&len,NULL);
            else free(logit);
            double tdt=now_s()-tt0; if(tdt<1e-6) tdt=1e-6;
            double dh=(double)(m->hits-h0), dm=(double)(m->miss-ms0);
            printf("\n\x01\x01" "END" "\x01\x01\n");
            printf("STAT %d %.2f %.1f %.2f\n", prod, prod/tdt, (dh+dm)>0?100.0*dh/(dh+dm):0.0, rss_gb());
            fflush(stdout);
            if(g_prof) prof_report(m,&pb,tdt,prod,stderr);   /* per-turn window; stdout is the framed protocol */
            kv_disk_append(m,hist,len); repin_pass(m); continue; }   /* RFC: re-pin a caldo tra i turni / live re-pin between turns */
        if(nr<1){ printf("\x01\x01" "END" "\x01\x01\n"); printf("STAT 0 0.00 0.0 %.2f\n", rss_gb()); fflush(stdout); continue; }
        /* API mode: an exact, length-prefixed prompt. Unlike the interactive
         * line protocol this accepts newlines. The tokenized prompt is matched
         * against hist so the common KV prefix survives stateless HTTP turns.
         * Per-request generation controls follow the byte count:
         *   \x02PROMPT <bytes> <max_tokens> <temperature> <top_p> [kv_slot]\n<prompt>\n */
        char *raw=NULL, *input=line;
        int input_n=(int)nr, raw_mode=0, req_ngen=ngen, prompt_tokens=0;
        float base_temp=g_temp, base_nuc=g_nuc;
        if(!strncmp(line,"\x02PROMPT ",8)){
            unsigned long long nb=0; double rt=0, rp=0; int slot=0;
            int nf=sscanf(line+8,"%llu %d %lf %lf %d",&nb,&req_ngen,&rt,&rp,&slot);
            if(nf<4 || nb>(16u<<20) || req_ngen<1 || rt<0 || rt>2 || rp<=0 || rp>1 ||
               slot<0 || slot>=nctx){
                printf("\x01\x01" "END" "\x01\x01\n"); printf("STAT 0 0.00 0.0 %.2f 0 0\n",rss_gb()); fflush(stdout); continue;
            }
            active=slot; sc=&ctx[active]; kv_bind(m,&sc->kv);
            raw=malloc((size_t)nb+1); if(!raw){fprintf(stderr,"OOM raw prompt\n");exit(1);}
            if(fread(raw,1,(size_t)nb,stdin)!=(size_t)nb){free(raw);break;}
            int delim=fgetc(stdin); if(delim!='\n' && delim!=EOF) ungetc(delim,stdin);
            if(memchr(raw,0,(size_t)nb)){free(raw); printf("\x01\x01" "END" "\x01\x01\n");
                printf("STAT 0 0.00 0.0 %.2f 0 0\n",rss_gb()); fflush(stdout); continue;}
            raw[nb]=0; input=raw; input_n=(int)nb; raw_mode=1;
            if(req_ngen>ngen) req_ngen=ngen;
            g_temp=(float)rt; g_nuc=(float)rp;
        } else { active=0; sc=&ctx[0]; kv_bind(m,&sc->kv); }
        int bl=0, k=0;                           /* costruisce/tokenizza il turno */
        /* template UFFICIALE GLM-5.2 (chat_template.jinja): niente \n dopo i ruoli, e dopo
         * <|assistant|> serve SEMPRE il blocco think — <think></think> lo DISATTIVA (nothink):
         * col template sbagliato il modello farfuglia e non emette mai lo stop. THINK=1 lo abilita. */
        const char *tk = getenv("THINK")&&atoi(getenv("THINK"))? "<think>" : "<think></think>";
        if(raw_mode){
            int *tmp=malloc(maxctx*sizeof(int)); if(!tmp){fprintf(stderr,"OOM raw tokens\n");exit(1);}
            prompt_tokens=tok_encode(&T,input,input_n,tmp,maxctx-8-g_draft);
            int old_len=len, prefix=0;
            while(prefix<old_len && prefix<prompt_tokens && hist[prefix]==tmp[prefix]) prefix++;
            if(prefix<old_len){
                len=prefix;
                if(m->has_mtp) m->kv_start[m->c.n_layers]=-1;
                kv_disk_truncate(m,len);           /* il prossimo append sovrascrive solo la coda */
            }
            k=prompt_tokens-len;
            if(k>0) memcpy(hist+len,tmp+len,k*sizeof(int));
            fprintf(stderr,"[API] KV slot %d prefix %d/%d token, prefill %d\n",
                active,len,prompt_tokens,k);
            free(tmp);
        } else {
            if(templ){ if(first) bl+=snprintf(buf+bl,(1<<16)-bl,"[gMASK]<sop>");
                       bl+=snprintf(buf+bl,(1<<16)-bl,"<|user|>%s<|assistant|>%s",input,tk); }
            else bl+=snprintf(buf+bl,(1<<16)-bl,"%s",input);
            k=tok_encode(&T,buf,bl,hist+len,maxctx-len); prompt_tokens=k;
            if(len+k+8+g_draft>=maxctx){ len=0; first=1; kv_disk_reset(m);
                bl=0; if(templ){ bl+=snprintf(buf+bl,(1<<16)-bl,"[gMASK]<sop><|user|>%s<|assistant|>%s",input,tk); }
                else bl+=snprintf(buf+bl,(1<<16)-bl,"%s",input);
                k=tok_encode(&T,buf,bl,hist,maxctx); if(k>maxctx-8-g_draft) k=maxctx-8-g_draft;
                prompt_tokens=k;
            }
        }
        if(prompt_tokens<1){ free(raw); g_temp=base_temp; g_nuc=base_nuc;
            printf("\x01\x01" "END" "\x01\x01\n"); printf("STAT 0 0.00 0.0 %.2f 0 0\n", rss_gb()); fflush(stdout); continue; }
        first=0;
        int cur=req_ngen; if(len+k+cur+g_draft+2>=maxctx) cur=maxctx-len-k-g_draft-2;
        uint64_t h0=m->hits, ms0=m->miss;
        uint64_t rs0=m->route_slots, rw0=m->route_swaps;
        uint64_t agh0=m->route_agree_hit, agt0=m->route_agree_tot;
        uint64_t kln0=m->route_kl_n; double kls0=m->route_kl_sum;
        double tt0=now_s();
        ProfBase pb; if(g_prof) prof_base(m,&pb);
        float *logit;
        if(k>0){ logit=step(m,hist+len,k,len); len+=k; }
        else logit=step(m,hist+len-1,1,len-1);   /* prompt identico/prefisso: rigenera i logits */
        EmitStream es={&T,m,now_s(),0,1};
        int prod=0;
        grammar_reset(&g_grd);                         /* nuova risposta = nuovo documento (MORE invece continua) */
        if(cur>0) prod=spec_decode(m,hist,len,cur,eos,logit,emit_stream,&es,&len,NULL);
        else free(logit);
        double tdt=now_s()-tt0; if(tdt<1e-6) tdt=1e-6;
        double dh=(double)(m->hits-h0), dm=(double)(m->miss-ms0);
        uint64_t rslots=m->route_slots-rs0, rswaps=m->route_swaps-rw0;
        double swap_pct=rslots?100.0*rswaps/rslots:0.0;
        uint64_t ag_hit=m->route_agree_hit-agh0, ag_tot=m->route_agree_tot-agt0;
        uint64_t kl_n=m->route_kl_n-kln0; double kl_sum=m->route_kl_sum-kls0;
        double agree_pct=ag_tot?100.0*ag_hit/ag_tot:100.0;
        double kl_mean=kl_n?kl_sum/(double)kl_n:0.0;
        printf("%s\x01\x01" "END" "\x01\x01\n",raw_mode?"":"\n");
        printf("STAT %d %.2f %.1f %.2f %d %d", prod, prod/tdt,
            (dh+dm)>0?100.0*dh/(dh+dm):0.0, rss_gb(), prompt_tokens, prod>=cur);
        if(g_cache_route || rslots || ag_tot)
            printf(" swap_pct=%.1f route_swaps=%llu route_slots=%llu"
                   " route_agree=%.1f route_kl=%.4f",
                swap_pct,(unsigned long long)rswaps,(unsigned long long)rslots,
                agree_pct,kl_mean);
        printf("\n");
        fflush(stdout);
        if(g_prof) prof_report(m,&pb,tdt,prod,stderr);   /* per-turn window; stdout is the framed protocol */
        free(raw); g_temp=base_temp; g_nuc=base_nuc;
        usage_save(m);                   /* la cache che impara: storia aggiornata a ogni turno */
        kv_disk_append(m,hist,len);      /* KV su disco: il prossimo avvio riparte da qui */
        repin_pass(m);                   /* safe request boundary: adapt session-local hot tier */
    }
    free(line); free(buf);
    usage_save(m);
    /* PROF=1 only: the cumulative backend counters (METAL:/METAL-ATTN:/MIRROR:) were
     * unreachable in serve mode — profile_print only ran on the oracle/generate exit
     * paths, so a served session could never show GPU-vs-fallback truth. stdin has hit
     * EOF here: the last END/STAT frame is already out, so these stdout lines can no
     * longer interleave with protocol a client is parsing. */
    if(g_prof) profile_print(m, now_s()-t_serve0);
    #undef hist
    #undef len
    #undef first
    for(int i=0;i<nctx;i++) serve_ctx_free(m,&ctx[i]);
    free(ctx); m->kv=NULL; m->Lc=m->Rc=m->Ic=NULL; m->Lc8=m->Rc8=NULL; m->Lsc=m->Rsc=NULL; m->kv_start=NULL; m->max_t=0;
}

static int *read_arr(jval*o,const char*k,int*n){
    jval*a=json_get(o,k);
    if(!a){ *n=0; return NULL; }
    int*r=malloc(a->len*sizeof(int));
    if(!r){ fprintf(stderr,"OOM read_arr\n"); exit(1); }
    for(int i=0;i<a->len;i++) r[i]=(int)a->kids[i]->num; *n=a->len; return r; }

/* telemetry, stats, usage persistence — moved to telemetry.h */

#ifdef COLI_VULKAN
/* Pinned VK expert tier fill: upload the top-COLI_VK_EXPERTS heat-ranked routed experts
 * ONCE at startup (mirrors the HIP VRAM tier: stable residency, zero uploads at decode).
 * Ranking comes from the persistent usage history; already-pinned RAM slots feed the
 * upload directly, the rest stream through one transient slot and are freed after. */
typedef struct { uint32_t u; int layer, eid; } VkCand;
static int vk_cand_cmp(const void *a, const void *b){
    uint32_t ua=((const VkCand*)a)->u, ub=((const VkCand*)b)->u;
    return ua<ub ? 1 : ua>ub ? -1 : 0;
}
/* Upload the VK dense working set (attention projections, absorb kv_b, o-proj,
 * shared expert) at startup, BEFORE the tier fill. These ~8 GB otherwise
 * allocate lazily on the first forwards — AFTER the tier has filled to its
 * budget — and on a 16 GB card the late arrivals overflow to GTT (measured
 * 2.1 GB spilled with a 320-expert int4 tier; every per-token attention
 * submit then pays PCIe latency: absorb 1.69 → 2.26 ms/call, +34%). Claiming
 * the dense set first lets the budget-capped tier fill see the true remainder
 * and self-size to what actually fits in device memory. */
static void vk_dense_preload(Model *m){
    Cfg *c=&m->c;
    if(!g_vulkan || !g_vk_dense) return;
    double t0=now_s(); int64_t bytes=0; int nt=0, full=0;
    for(int i=0;i<=c->n_layers && !full;i++){
        Layer *l = i<c->n_layers ? &m->L[i] : &m->mtpL;
        QT *ts[]={&l->q_a,&l->q_b,&l->kv_a,&l->kv_b,&l->o,&l->sh_gate,&l->sh_up,&l->sh_down};
        for(size_t k=0;k<sizeof(ts)/sizeof(ts[0]);k++){
            QT *t=ts[k];
            if(t->vk || !VK_FMT_OK(t) || t->I<=0 || t->O<=0) continue;
            const void *w = t->fmt==1 ? (const void*)t->q8 : (const void*)t->q4;
            if(!w || !t->s) continue;
            if(!coli_vk_tensor_ensure(&t->vk,w,t->s,t->fmt,t->I,t->O,t->gs)){
                fprintf(stderr,"[VK] dense preload: VRAM full at layer %d — remaining tensors stay lazy\n",i);
                full=1; break;
            }
            bytes+=coli_vk_tensor_bytes(t->vk); nt++;
        }
    }
    if(nt) fprintf(stderr,"[VK] dense preloaded: %d tensors, %.2f GB VRAM in %.1fs\n",
                   nt,bytes/1e9,now_s()-t0);
}

static void vk_registry_fill(Model *m){
    Cfg *c=&m->c; int E=c->n_experts, NL=c->n_layers;
    if(!g_vulkan || g_vk_budget<=0) return;
    int64_t nz=0;
    for(int i=0;i<NL;i++) if(m->eusage[i]) for(int e=0;e<E;e++) if(m->eusage[i][e]) nz++;
    if(!nz){ fprintf(stderr,"[VK] expert tier: no usage history yet — tier empty this run "
                     "(it seeds from %s as you use the model)\n", g_usage_path); return; }
    VkCand *cand=malloc((size_t)nz*sizeof(VkCand)); if(!cand) return;
    int64_t n=0;
    for(int i=0;i<NL;i++) if(m->eusage[i]) for(int e=0;e<E;e++)
        if(m->eusage[i][e]) cand[n++]=(VkCand){m->eusage[i][e],i,e};
    qsort(cand,(size_t)n,sizeof(VkCand),vk_cand_cmp);
    g_vk_reg=calloc((size_t)NL*E*3,sizeof(*g_vk_reg)); g_vk_reg_E=E; g_vk_reg_NL=NL;
    if(!g_vk_reg){ free(cand); return; }
    ESlot tmp; memset(&tmp,0,sizeof(tmp)); tmp.eid=-1;
    double t0=now_s(); int64_t bytes=0; int tried=0, loadfail=0;
    /* Pressure-proofing: tier weights are the EVICTABLE class (0.4 vs scratch/KV 1.0,
     * dense 0.75) so an oversubscribed heap sheds cold experts instead of thrashing
     * the per-token attention submits; and the fill STOPS while the device-local
     * budget still holds COLI_VK_RESERVE_GB (default 3) for the lazily-allocated
     * dense weights + KV mirror + staging (measured ~1.7 GB at 4k ctx, growing with
     * max_t). Without the budget extension the count cap alone applies, as before. */
    double vkr_reserve = getenv("COLI_VK_RESERVE_GB")?atof(getenv("COLI_VK_RESERVE_GB")):3.0;
    int vkr_stopped=0; double vkr_used=0, vkr_budget=0;
    int64_t i2=0;
    coli_vk_alloc_priority(0.4f);
    for(;i2<n && g_vk_reg_n<g_vk_budget;i2++){
        if(vkr_reserve>0 && (g_vk_reg_n&7)==0 && coli_vk_mem_budget(&vkr_used,&vkr_budget)
           && vkr_budget-vkr_used < vkr_reserve){ vkr_stopped=1; break; }
        int layer=cand[i2].layer, eid=cand[i2].eid; tried++;
        ESlot *src=pin_indexed(m,layer,eid);
        if(src&&!src->slab) src=NULL;
        if(!src){ if(expert_load(m,layer,eid,&tmp,0,0)!=0){   /* 0 = success (impl convention); demand=0: startup tier fill */
                if(++loadfail<4) fprintf(stderr,"[VK] tier fill: expert_load(%d,%d) failed\n",layer,eid);
                if(loadfail>=64) break;                     /* disk trouble: stop burning time */
                continue; } src=&tmp; }
        int xf=src->g.fmt;   /* int4 (2), grouped int4 (4), int3-g64 (5) tiers; gate/up must
                              * share fmt for the fused gate_up shader, down may differ */
        if((xf!=2&&xf!=4&&xf!=5)||src->u.fmt!=xf||src->u.gs!=src->g.gs
           ||(src->d.fmt!=2&&src->d.fmt!=4&&src->d.fmt!=5)
           ||(xf==4&&(src->g.gs<8||src->g.gs%8))||(src->d.fmt==4&&(src->d.gs<8||src->d.gs%8))){
            if(tried<4) fprintf(stderr,"[VK] tier fill: (%d,%d) fmt %d/%d/%d not int4/int3-g64 (src=%s)\n",
                layer,eid,src->g.fmt,src->u.fmt,src->d.fmt,src==&tmp?"load":"pin");
            continue; }
        ColiVkTensor **slot=vk_reg_at(layer,eid);
        if(!coli_vk_tensor_ensure(&slot[0],src->g.q4,src->g.s,xf,c->hidden,c->moe_inter,src->g.gs)||
           !coli_vk_tensor_ensure(&slot[1],src->u.q4,src->u.s,xf,c->hidden,c->moe_inter,src->u.gs)||
           !coli_vk_tensor_ensure(&slot[2],src->d.q4,src->d.s,src->d.fmt,c->moe_inter,c->hidden,src->d.gs)){
            if(slot[0]){coli_vk_tensor_free(slot[0]);slot[0]=NULL;}
            if(slot[1]){coli_vk_tensor_free(slot[1]);slot[1]=NULL;}
            fprintf(stderr,"[VK] expert tier: VRAM full after %d experts\n",g_vk_reg_n);
            break;
        }
        bytes+=coli_vk_tensor_bytes(slot[0])+coli_vk_tensor_bytes(slot[1])+coli_vk_tensor_bytes(slot[2]);
        g_vk_reg_n++;
    }
    coli_vk_alloc_priority(0.75f);               /* back to the dense/default class */
    fprintf(stderr,"[VK] expert tier: %d hot experts resident (%.2f GB VRAM, %.1fs, top-%d of history)\n",
            g_vk_reg_n,bytes/1e9,now_s()-t0,tried);
    if(vkr_stopped)
        fprintf(stderr,"[VK] expert tier: budget stop at %d experts — %.1f of %.1f GB device-local used, %.1f GB reserved (COLI_VK_RESERVE_GB)\n",
                g_vk_reg_n,vkr_used,vkr_budget,vkr_reserve);
    /* DEV2 tier: continue down the heat ranking onto the second GPU (COLI_VK_DEV2),
     * starting at the candidate dev0 stopped on. Same fmt gates, its own budget. */
    if(g_vk_budget2>0 && coli_vk_dev2_available()){
        double t20=now_s(); int64_t bytes2=0; int tried2=0;
        double vkr2_reserve = getenv("COLI_VK_RESERVE2_GB")?atof(getenv("COLI_VK_RESERVE2_GB")):0.5;
        int vkr2_stopped=0; double u2=0,b2=0;
        for(;i2<n && g_vk_reg_n2<g_vk_budget2;i2++){
            if(vkr2_reserve>0 && (g_vk_reg_n2&7)==0 && coli_vk_mem_budget2(&u2,&b2)
               && b2-u2 < vkr2_reserve){ vkr2_stopped=1; break; }
            int layer=cand[i2].layer, eid=cand[i2].eid; tried2++;
            ESlot *src=pin_indexed(m,layer,eid);
            if(src&&!src->slab) src=NULL;
            if(!src){ if(expert_load(m,layer,eid,&tmp,0,0)!=0){
                    if(++loadfail>=64) break;
                    continue; } src=&tmp; }
            int xf=src->g.fmt;
            if((xf!=2&&xf!=4&&xf!=5)||src->u.fmt!=xf||src->u.gs!=src->g.gs
               ||(src->d.fmt!=2&&src->d.fmt!=4&&src->d.fmt!=5)
               ||(xf==4&&(src->g.gs<8||src->g.gs%8))||(src->d.fmt==4&&(src->d.gs<8||src->d.gs%8)))
                continue;
            ColiVkTensor **slot=vk_reg_at(layer,eid);
            if(!coli_vk_tensor_ensure2(&slot[0],src->g.q4,src->g.s,xf,c->hidden,c->moe_inter,src->g.gs)||
               !coli_vk_tensor_ensure2(&slot[1],src->u.q4,src->u.s,xf,c->hidden,c->moe_inter,src->u.gs)||
               !coli_vk_tensor_ensure2(&slot[2],src->d.q4,src->d.s,src->d.fmt,c->moe_inter,c->hidden,src->d.gs)){
                if(slot[0]){coli_vk_tensor_free(slot[0]);slot[0]=NULL;}
                if(slot[1]){coli_vk_tensor_free(slot[1]);slot[1]=NULL;}
                fprintf(stderr,"[VK] dev2 tier: VRAM full after %d experts\n",g_vk_reg_n2);
                break;
            }
            bytes2+=coli_vk_tensor_bytes(slot[0])+coli_vk_tensor_bytes(slot[1])+coli_vk_tensor_bytes(slot[2]);
            g_vk_reg_n2++;
        }
        fprintf(stderr,"[VK] dev2 tier: %d experts resident (%.2f GB VRAM, %.1fs, next-%d of history)\n",
                g_vk_reg_n2,bytes2/1e9,now_s()-t20,tried2);
        if(vkr2_stopped)
            fprintf(stderr,"[VK] dev2 tier: budget stop at %d experts — %.1f of %.1f GB device-local used, %.1f GB reserved (COLI_VK_RESERVE2_GB)\n",
                    g_vk_reg_n2,u2,b2,vkr2_reserve);
    }
    if(tmp.slab){ compat_aligned_free(tmp.slab); free(tmp.fslab); }
    free(cand);
}
#endif

/* HOT-STORE ("il redis del colibri'"): carica in RAM, UNA VOLTA e per sempre, i top expert
 * per frequenza d'uso misurata (file STATS di un run precedente), entro un budget in GB.
 * Ogni hit evita una lettura dal disco lento. */
/* MLOCK: inchioda in RAM fisica gli expert pinnati cosi' il compressore di memoria di
 * macOS non li comprime/evacua (visto: RSS reale < residente previsto -> "hit" lenti).
 * -1 = auto (ON su macOS dove serve e RLIMIT_MEMLOCK e' permissivo; OFF altrove, dove
 * il limite e' spesso minuscolo e va alzato a mano), 0 = off, 1 = force.
 * EN: MLOCK: wire pinned experts into physical RAM so macOS's memory compressor can't
 * compress/evict them (we saw actual RSS < intended resident -> slow "hits"). -1 = auto
 * (ON on macOS where it matters and RLIMIT_MEMLOCK is permissive; OFF elsewhere, where the
 * limit is often tiny and must be raised by hand), 0 = off, 1 = force. */
static int g_mlock=-1;
static int mem_should_wire(void){
    if(g_mlock>=0) return g_mlock;
#if defined(__APPLE__)
    return 1;                                     /* macOS: default ON */
#else
    return 0;                                     /* Linux/altri: opt-in via MLOCK=1 / opt-in */
#endif
}
/* Inchioda [addr,addr+len) in RAM fisica. No-op fuori da POSIX (Windows ecc.).
 * EN: wire [addr,addr+len) into physical RAM. No-op off POSIX (Windows, etc.). */
static int mem_wire(void *addr, size_t len){
#if defined(__APPLE__) || defined(__linux__) || defined(__FreeBSD__)
    return mlock(addr, len);
#elif defined(_WIN32)
    return compat_mlock(addr, len);   /* VirtualLock + working-set growth */
#else
    (void)addr; (void)len; return 0;
#endif
}
/* Inchioda tutti gli slab degli expert pinnati (pesi + scale). Non fatale se fallisce.
 * EN: wire all pinned-expert slabs (weights + scales). Non-fatal on failure. */
/* mlock a single mmap'd QT's weight + scale ranges. Skips VRAM-tier QTs
 * (cuda_eligible): their compute runs from device memory, so wiring the host
 * mmap range would pin ~137 GB of never-touched file pages. NOTE the q8/q4
 * NULL check alone is NOT enough here: expert_host_release() early-returns
 * for mmap experts (no slab) without nulling the host pointers, so GPU-tier
 * slots keep live-looking q8/q4 forever -- that was the bug that wired 363 GB
 * instead of 231 GB and starved the kernel into page-cache thrashing.
 * wired/failed are accumulated into the caller's counters. */
/* undo qt_wire_mmap for one QT: used when a REPIN gpu_swap promotes a wired
 * RAM-tier expert into VRAM -- without this every promotion leaks its locked
 * host range and the dead-weight lock re-grows over a long session. */
static void qt_unwire_mmap(QT *t){
    if(!g_mmap || !mem_should_wire()) return;
    if(!t->q8 && !t->q4) return;
    int64_t weight_b, scale_b; qt_wire_split(t,&weight_b,&scale_b);
    void *wp=t->q8?(void*)t->q8:(void*)t->q4;
#if defined(__APPLE__) || defined(__linux__) || defined(__FreeBSD__)
    if(weight_b>0 && !munlock(wp,(size_t)weight_b)) g_mmap_wired-=weight_b;
    if(t->s && scale_b>0 && !munlock(t->s,(size_t)scale_b)) g_mmap_wired-=scale_b;
#elif defined(_WIN32)
    if(weight_b>0 && !compat_munlock(wp,(size_t)weight_b)) g_mmap_wired-=weight_b;
    if(t->s && scale_b>0 && !compat_munlock(t->s,(size_t)scale_b)) g_mmap_wired-=scale_b;
#endif
}
static void qt_wire_mmap(QT *t, int64_t *wired, long *failed){
    if(!t->q8 && !t->q4) return;
    if(t->cuda_eligible) return;   /* resident in VRAM; host range is dead weight */
    int64_t weight_b, scale_b; qt_wire_split(t,&weight_b,&scale_b);
    void *wp=t->q8?(void*)t->q8:(void*)t->q4;
    if(weight_b>0){ if(mem_wire(wp,(size_t)weight_b)==0) *wired+=weight_b; else (*failed)++; }
    if(t->s && scale_b>0){ if(mem_wire(t->s,(size_t)scale_b)==0) *wired+=scale_b; else (*failed)++; }
}
static void pin_wire(Model *m){
    if(!mem_should_wire()) return;
    if(g_mmap){
        /* Wire the FINAL resident set only, after pin_load's GPU-upload pass
         * has already run -- qt_wire_mmap() skips cuda_eligible (VRAM-tier)
         * slots, so only the genuinely RAM-tier experts get locked. */
        Cfg *c=&m->c; double t0=now_s();
        for(int i=0;i<c->n_layers;i++) for(int z=0;z<m->npin[i];z++){
            ESlot *s=&m->pin[i][z];
            qt_wire_mmap(&s->g,&g_mmap_wired,&g_mmap_wire_failed);
            qt_wire_mmap(&s->u,&g_mmap_wired,&g_mmap_wire_failed);
            qt_wire_mmap(&s->d,&g_mmap_wired,&g_mmap_wire_failed);
        }
        fprintf(stderr,"[PIN] mlock (mmap): %.1f GB wired in physical RAM%s in %.0fs\n",
            g_mmap_wired/1e9, g_mmap_wire_failed?" (some allocations failed -- raise: ulimit -l unlimited)":"", now_s()-t0);
        return;
    }
    Cfg *c=&m->c; double t0=now_s(); int64_t wired=0; long failed=0;
    for(int i=0;i<c->n_layers;i++) for(int z=0;z<m->npin[i];z++){
        ESlot *s=&m->pin[i][z];
        if(s->slab){  if(mem_wire(s->slab, s->slab_cap)==0) wired+=s->slab_cap; else failed++; }
        if(s->fslab){ size_t fl=(size_t)s->fslab_cap*sizeof(float);
                      if(mem_wire(s->fslab, fl)==0) wired+=fl; else failed++; }
    }
    if(failed)
        fprintf(stderr,"[PIN] mlock: %.1f GB wired, %ld allocations failed "
            "(raise the limit: ulimit -l unlimited) in %.0fs\n", wired/1e9, failed, now_s()-t0);
    else
        fprintf(stderr,"[PIN] mlock: %.1f GB wired in physical RAM "
            "(no compression) in %.0fs\n", wired/1e9, now_s()-t0);
}

/* DUAL-SSD: measure one replica's read bandwidth with the engine's own access
 * pattern (parallel ~19 MB reads, O_DIRECT twin when available — buffered would
 * measure the page cache, see iobench's #86 caveat). Reads the largest shard at
 * deterministic spread-out offsets; ~150 MB per drive, a few hundred ms total. */
static double mirror_probe_bw(shards *S,int rep){
    int big=-1; int64_t bsz=0;
    for(int i=0;i<S->nfd;i++){
        if(rep && S->mfds[rep-1][i]<0) continue;
        int64_t sz=lseek(rep?S->mfds[rep-1][i]:S->fds[i],0,SEEK_END);
        if(sz>bsz){ bsz=sz; big=i; }
    }
    const int64_t blk=19ll<<20; const int NB=8;
    if(big<0 || bsz<blk*(NB+1)) return 0;
    int dfd = rep? S->mdfds[rep-1][big] : S->dfds[big];
    int fd  = dfd>=0? dfd : (rep? S->mfds[rep-1][big] : S->fds[big]);
    if(dfd<0) fprintf(stderr,"[MIRROR] no O_DIRECT on replica %d: the probe may read the page cache — "
                             "set COLI_DISK_WEIGHTS for an accurate split\n", rep);
    double t0=now_s(); int64_t tot=0;
    #pragma omp parallel for schedule(dynamic,1) reduction(+:tot)
    for(int i=0;i<NB;i++){
        void *buf;
        if(!posix_memalign(&buf,4096,blk)){
            int64_t off=(((bsz-blk)/NB)*i) & ~4095ll;   /* deterministic, spread across the file */
            ssize_t r=pread(fd,buf,blk,off);
            if(r>0) tot+=r;
            compat_aligned_free(buf);
        }
    }
    double dt=now_s()-t0;
    return (dt>0 && tot>0)? tot/1e9/dt : 0;
}

/* MULTI-SSD setup: register every mirror copy listed in COLI_MODEL_MIRROR
 * (';' or ',' separated dirs), derive the read split from
 * COLI_DISK_WEIGHTS=<primary>,<mirror>[,<mirror2>...] or from a startup
 * bandwidth probe. Runs after model_init (needs the shard index) and BEFORE
 * any pin/autopin load, so the OMP-parallel pin warmup streams from all drives. */
static void mirror_setup(Model *m){
    if(!g_mirror_dir) return;
    const char *snap=getenv("SNAP"); if(!snap||!*snap) snap=getenv("COLI_MODEL");
    st_mirror_reset(&m->S);
    int nrep=1;
    {   char buf[4096]; snprintf(buf,sizeof(buf),"%s",g_mirror_dir);
        char *p=buf;
        while(p && *p){
            char *sep=p; while(*sep && *sep!=';' && *sep!=',') sep++;
            int last=(*sep==0); *sep=0;
            while(*p==' ') p++;
            size_t plen=strlen(p); while(plen>0 && p[plen-1]==' ') p[--plen]=0;
            if(*p){
                if(snap && !strcmp(snap,p))
                    fprintf(stderr,"[MIRROR] %s equals the model dir — ignored\n",p);
                else if(nrep>=MIR_REPS)
                    fprintf(stderr,"[MIRROR] %s: too many mirrors (max %d) — ignored\n",p,ST_MAX_MIR);
                else {
                    int nf=st_mirror_add(&m->S,p);
                    if(nf<=0)
                        fprintf(stderr,"[MIRROR] %s: no usable shard (missing or divergent copy) — skipped\n",p);
                    else {
                        fprintf(stderr,"[MIRROR] %s: %d/%d shards (replica %d)\n",p,nf,m->S.nfd,nrep);
                        nrep++;
                    }
                }
            }
            p=last?NULL:sep+1;
        }
    }
    if(nrep<2){
        fprintf(stderr,"[MIRROR] no usable mirror — running on the primary drive only\n");
        return;
    }
    g_mirror=1; g_mir_nrep=nrep;
    double wt[MIR_REPS]; int have=0;
    const char *w=getenv("COLI_DISK_WEIGHTS"); const char *how="COLI_DISK_WEIGHTS";
    if(w && *w){
        char wb[256]; snprintf(wb,sizeof(wb),"%s",w); int wn=0, bad=0;
        for(char *tok=strtok(wb,", "); tok; tok=strtok(NULL,", ")){
            double v=atof(tok);
            if(v<=0 || wn>=MIR_REPS){ bad=1; break; }
            wt[wn++]=v;
        }
        if(!bad && wn==nrep) have=1;
        else fprintf(stderr,"[MIRROR] invalid COLI_DISK_WEIGHTS '%s' (want %d positive comma-separated "
                            "weights, e.g. 9,3) — probing instead\n",w,nrep);
    }
    if(!have){
        have=1; how="measured";
        for(int r=0;r<nrep;r++){ wt[r]=mirror_probe_bw(&m->S,r); if(wt[r]<=0) have=0; }
        if(have){
            fprintf(stderr,"[MIRROR] probe:");
            for(int r=0;r<nrep;r++) fprintf(stderr,"%s %s %.2f GB/s",r?" |":"",r?"mirror":"primary",wt[r]);
            fprintf(stderr,"\n");
        } else { for(int r=0;r<nrep;r++) wt[r]=1; how="fallback 1:1 (probe failed)"; }
    }
    double W=0; for(int r=0;r<nrep;r++) W+=wt[r];
    int acc=0; double cum=0;
    for(int r=0;r<nrep;r++){
        cum+=wt[r];
        int c=(int)(256.0*cum/W+0.5);
        if(c<=acc) c=acc+1;            /* every replica keeps a non-empty slice */
        if(c>256) c=256;
        g_mir_cut[r]=c; acc=c;
    }
    g_mir_cut[nrep-1]=256;             /* the last replica absorbs rounding */
    fprintf(stderr,"[MIRROR] %d drives | read split",nrep);
    for(int r=0;r<nrep;r++){ int lo=r?g_mir_cut[r-1]:0;
        fprintf(stderr,"%s %.0f%%",r?" /":"",100.0*(g_mir_cut[r]-lo)/256); }
    fprintf(stderr," (%s)\n",how);
}

typedef struct { int l,e; uint32_t c; } PinRec;
static int pin_rec_cmp(const void *a,const void *b){
    const PinRec *x=a,*y=b; return x->c<y->c?1:x->c>y->c?-1:0;
}

/* A pin budget buys a popularity-ranked PREFIX, but mixed-width containers do
 * not have one honest bytes-per-expert divisor.  GLM-5.2's routed rows are int4
 * while its MTP row is int8: pricing every candidate at the widest row made a
 * 51.1 GB budget buy the same 1,352 experts in both g64 and E8 containers even
 * though their routed experts cost only 21.2 and 14.6 MB (#885).
 *
 * Keep the ranking semantics -- never skip a hot wide candidate to admit a
 * colder narrow one -- and stop immediately before the first candidate that
 * would cross the byte ceiling.  `from` lets CUDA_RELEASE_HOST exclude the
 * transient VRAM prefix before spending the disjoint RAM budget. */
static double pin_range_bytes(Model *m, const PinRec *r, int from, int to){
    double used=0.0;
    for(int a=from;a<to;a++) used+=(double)expert_bytes_row(m,r[a].l,m->ebits);
    return used;
}
static int pin_count_for_budget(Model *m, const PinRec *r, int from, int n,
                                double budget_b){
    double used=0.0; int count=0;
    if(budget_b<=0.0 || from>=n) return 0;
    for(int a=from;a<n;a++){
        double need=(double)expert_bytes_row(m,r[a].l,m->ebits);
        if(need<=0.0 || used+need>budget_b) break;
        used+=need; count++;
    }
    return count;
}

#ifdef __linux__
/* #419: bind the pinned hot-store as ONE arena per layer instead of one mbind
 * per slab. Per-slab policies cost ~2 unmergeable VMAs each; a PIN_GB=all load
 * (19,456 experts x slab+fslab) crosses the default vm.max_map_count=65530 and
 * posix_memalign dies with terabytes free. Experts of one layer share a tensor
 * shape, so a layer's pins pack into two arenas (weights + scales) at a fixed
 * stride: 2 binds and a handful of VMAs per layer instead of ~500. Slices are
 * pre-attached to the slots (slab_cap covers expert_load's realloc check, so
 * its alloc branch never fires); aslab marks arena ownership for the
 * release/ensure paths. Arena-OOM just leaves the slots on the individual path. */
static void pin_arena_bind(Model *m, PinRec *r, int *slot_of, int from, int to){
    if(g_numa_nodes<2 || g_mmap || from>=to) return;
    Cfg *c=&m->c; int NR=c->n_layers+1;
    int *cnt=calloc((size_t)NR,sizeof(int)); int *first=malloc((size_t)NR*sizeof(int));
    if(!cnt||!first){ free(cnt); free(first); return; }
    for(int i=0;i<NR;i++) first[i]=-1;
    for(int a=from;a<to;a++){ if(first[r[a].l]<0) first[r[a].l]=a; cnt[r[a].l]++; }
    const char suf[3][16]={"gate_proj","up_proj","down_proj"};  /* bounded suf: see #484 */
    for(int l=0;l<NR;l++){
        if(cnt[l]<2) continue;
        int64_t wtot=0, qtot=0; int ok=1;
        for(int k=0;k<3 && ok;k++){
            char nm[288],qn[320];
            snprintf(nm,sizeof nm,"model.layers.%d.mlp.experts.%d.%s.weight",l,r[first[l]].e,suf[k]);
            snprintf(qn,sizeof qn,"%s.qs",nm);
            st_tensor *tw=st_find(&m->S,nm), *tq=st_find(&m->S,qn);
            if(!tw||!tq) ok=0; else { wtot+=tw->nbytes; qtot+=tq->nbytes; }
        }
        if(!ok) continue;                     /* unquantized fallback: individual allocs */
        size_t ws=((size_t)wtot+8192+4095)&~(size_t)4095;
        size_t fs=((size_t)(qtot/4)*sizeof(float)+4095)&~(size_t)4095;
        uint8_t *aw=NULL; float *af=NULL;
        if(posix_memalign((void**)&aw,4096,(size_t)cnt[l]*ws)) continue;
        if(posix_memalign((void**)&af,4096,(size_t)cnt[l]*fs)){ free(aw); continue; }
        numa_slab_bind(aw,(size_t)cnt[l]*ws);
        numa_slab_bind(af,(size_t)cnt[l]*fs);
        int i=0;
        for(int a=from;a<to;a++){
            if(r[a].l!=l) continue;
            ESlot *s=&m->pin[l][slot_of[a]];
            s->slab=aw+(size_t)i*ws;   s->slab_cap=(int64_t)ws;   s->aslab=s->slab;
            s->fslab=(float*)((uint8_t*)af+(size_t)i*fs);
            s->fslab_cap=(int64_t)(fs/sizeof(float));             s->afslab=s->fslab;
            i++;
        }
    }
    free(cnt); free(first);
}
#endif
static double expert_avail(Model *m, double ram_gb, int ebits, int max_ctx);  /* def. sotto */
static double g_mem_avail_boot;   /* def. sotto (#653: corretta qui su GPU integrate) */
static double coli_clamp_ram_gb(double ram_gb, double mem_avail_gb, int overcommit);  /* def. sotto (#759) */
/* Admission test unchanged; it just runs from route_trace.h's reader now, so PIN=<file>
 * accepts every history layout an engine can write instead of only sparse text (#700). */
typedef struct { Model *m; PinRec *r; int *n, cap; unsigned char *seen; } PinCollect;
static int pin_collect_cb(int l, int e, uint32_t cnt, void *ud){
    PinCollect *p=(PinCollect*)ud; Model *m=p->m; Cfg *c=&m->c;
    if(*p->n>=p->cap) return 0;
    int ok = l>=0 && e>=0 && e<c->n_experts &&
             ((l<c->n_layers && m->L[l].sparse) || (l==c->n_layers && m->has_mtp));
    int64_t key=(int64_t)l*c->n_experts+e;
    if(ok&&!p->seen[key]){ p->r[(*p->n)++]=(PinRec){l,e,cnt}; p->seen[key]=1; return 1; }
    return 0;
}
/* trusted=1 only when the user typed this path. An auto-discovered history has not been
 * vouched for by anyone, so it stays subject to the identity check like any other. */
static void pin_load(Model *m, const char *statspath, double gb, int trusted){
    { FILE *probe=fopen(statspath,"rb");        /* keep the same message on a bad path */
      if(!probe){ perror(statspath); return; } fclose(probe); }
    Cfg *c=&m->c; int cap=(c->n_layers+1)*c->n_experts;
    PinRec *r=malloc((size_t)cap*sizeof(PinRec)); int n=0;
    unsigned char *seen=calloc((size_t)(c->n_layers+1)*c->n_experts,1);
    /* A named file is what the identity refusal tells the user to pass, so honouring it
     * here is what makes that message true. Dimensions and format version still apply:
     * those refusals never offered a way past them. */
    { PinCollect pc={m,r,&n,cap,seen}; rt_read_ex(statspath,pin_collect_cb,&pc,trusted); }
    int fill=getenv("PIN_FILL")?atoi(getenv("PIN_FILL")):0;
#ifdef COLI_CUDA
    if(!getenv("PIN_FILL")&&g_cuda_release_host) fill=1;
#endif
    if(fill) for(int li=0;li<=c->n_layers;li++){
        int sparse=(li<c->n_layers&&m->L[li].sparse)||(li==c->n_layers&&m->has_mtp);
        if(sparse) for(int ei=0;ei<c->n_experts;ei++) if(!seen[(int64_t)li*c->n_experts+ei])
            r[n++]=(PinRec){li,ei,0};
    }
    free(seen);
    qsort(r,(size_t)n,sizeof(*r),pin_rec_cmp);
    /* PIN_GB=all (#80): NON "tutti" alla lettera. Pinnare l'intero set ignora il
     * budget --ram e fa OOM-kill del kernel a meta' generazione (#229: host 92 GB
     * ucciso con --ram 78, anon-rss 89 GB). Clampa a quanti expert entrano nel
     * budget RAM, come AUTOPIN; il pin aggiorna resident_bytes, quindi cap_for_ram
     * dopo restringe la LRU di conseguenza (nessun doppio conteggio). */
    double pin_budget_b;
    if(gb<0){
        double ram_env=getenv("RAM_GB")?atof(getenv("RAM_GB")):0.0;
        /* #759: same clamp as cap_for_ram; here the snapshot is still the
         * uncorrected boot value (#653 runs later in this function), so on
         * unified-memory hosts this only rejects budgets larger than physical
         * RAM at boot -- never smaller than what the tier will leave behind. */
        ram_env=coli_clamp_ram_gb(ram_env,g_mem_avail_boot,
            getenv("COLI_RAM_OVERCOMMIT")?atoi(getenv("COLI_RAM_OVERCOMMIT")):0);
        int est_ctx=getenv("CTX")?atoi(getenv("CTX")):4096;   /* stesso default del call site */
        double avail=expert_avail(m,ram_env,m->ebits,est_ctx);
        pin_budget_b=avail>0?avail:0.0;
    } else pin_budget_b=gb*1e9;
    int cpu_from=0;
#ifdef COLI_CUDA
    int64_t eb=expert_bytes_probe(m,m->ebits);  /* shared ws / CUDA staging ceiling */
    /* The VRAM budget must be known BEFORE npin is finalized: with
     * CUDA_RELEASE_HOST the VRAM-ranked prefix's host slabs are freed right
     * after upload, so those slots must NOT consume the RAM pin budget.
     * Before this, a 6-GPU host lost its top ~9k ranked experts from the RAM
     * count and pinned only the leftovers (measured: 9,280 VRAM + 1,721 RAM
     * on a box whose RAM fits ~11k — the cold tail then paid disk forever). */
    double remaining[COLI_CUDA_MAX_DEVICES]={0}, placed_b[COLI_CUDA_MAX_DEVICES]={0};
    double placed_w[COLI_CUDA_MAX_DEVICES]={0};
    int load_balance=getenv("CUDA_EXPERT_LOAD_BALANCE")?
                     atoi(getenv("CUDA_EXPERT_LOAD_BALANCE")):0;
    int placed_n[COLI_CUDA_MAX_DEVICES]={0}, gpu_prefix=0, prefix_est=0;
    double budget=g_cuda_expert_gb*1e9, safe_total=0;
    if(g_cuda_enabled&&(g_cuda_expert_gb>0||g_cuda_expert_auto)) for(int i=0;i<g_cuda_ndev;i++){
        size_t free_b=0,total_b=0;
        if(coli_cuda_mem_info(g_cuda_devices[i],&free_b,&total_b)){
            remaining[i]=(double)free_b-(double)g_cuda_dense_projected[i]-g_cuda_reserve_gb*1e9;
            if(remaining[i]<0) remaining[i]=0; safe_total+=remaining[i];
        }
    }
    /* auto: fill to measured headroom (safe_total). An explicit CUDA_EXPERT_GB is
     * honored as-is even when it exceeds headroom; per-expert upload failure below
     * (remaining[best]=0, continue to next expert) degrades gracefully rather
     * than OOM-ing. Previously both paths were clamped, which silently capped the
     * tier under CUDA_DENSE=1 regardless of the configured budget (#491). */
    if(g_cuda_expert_auto) budget=safe_total;
    if(g_cuda_enabled&&g_cuda_release_host&&budget>0){
        prefix_est=(int)(budget/eb)+g_cuda_ndev;
#ifdef COLI_ANS
        if(g_cuda_raw_experts>=0){
            int raw=g_cuda_raw_experts;
            if((double)raw*eb>budget) raw=(int)(budget/eb);
            prefix_est=raw+(int)((budget-(double)raw*eb)/(0.80*eb))+g_cuda_ndev;
        }
#endif
        if(prefix_est>n) prefix_est=n;
        cpu_from=prefix_est;                    /* prefix RAM is returned after upload */
    }
#endif
    int npin=cpu_from+pin_count_for_budget(m,r,cpu_from,n,pin_budget_b);
    if(npin>n) npin=n;
    if(npin<1){ free(r); return; }
    int *cnt_l=calloc(c->n_layers+1,sizeof(int));   /* +1: riga MTP */
    for(int a=0;a<npin;a++) cnt_l[r[a].l]++;
    for(int i=0;i<=c->n_layers;i++) if(cnt_l[i]) m->pin[i]=calloc(cnt_l[i],sizeof(ESlot));
    int *slot_of=malloc((size_t)npin*sizeof(int)), *next=calloc(c->n_layers+1,sizeof(int));
    for(int a=0;a<npin;a++) slot_of[a]=next[r[a].l]++;
    for(int i=0;i<=c->n_layers;i++) m->npin[i]=cnt_l[i];
    double t0=now_s(), pin_host_released=0.0;
#ifdef COLI_CUDA
    if(prefix_est>0){ gpu_prefix=prefix_est; if(gpu_prefix>npin) gpu_prefix=npin; }
#else
    int gpu_prefix=0;
#endif
#ifdef __linux__
    /* CPU-resident pins only: the GPU prefix stays on individual allocs (its
     * host backing is released after upload; arena slices are never freed). */
    pin_arena_bind(m,r,slot_of,gpu_prefix,npin);
#endif
    /* Load the VRAM-ranked prefix first.  Once uploaded its host backing is
     * released before the disjoint RAM-ranked suffix is allocated. */
#ifdef __linux__
    if(gpu_prefix>0) g_numa_skip_bind=1;   /* prefix slabs = transient upload staging: don't bind (#419) */
#endif
    int pre_n = gpu_prefix?gpu_prefix:npin;   /* quanti esperti nella fase "prefisso" */
    /* #730: quanti caricarne per volta. Il default e' "tutti", che riproduce esattamente
     * il comportamento precedente. Sotto CUDA_RELEASE_HOST i slab host del prefisso sono
     * staging TRANSITORIO -- il commento di budget qui sopra lo dice: "prefix RAM is
     * returned after upload" -- ma erano transitori in AGGREGATO: si caricava tutto il
     * prefisso in RAM host e solo dopo si liberava un esperto alla volta. Il picco di RSS
     * host era quindi l'intero CUDA_EXPERT_GB, e su un host con piu' VRAM che RAM (96 GB
     * di VRAM contro 64 di RAM, #730) l'OOM arriva prima che il primo release parta.
     * A round, il picco diventa stage*eb e il carico parallelo mantiene la sua banda. */
    int stage = pre_n;
#ifdef COLI_CUDA
    if(g_cuda_enabled && g_cuda_release_host && gpu_prefix>0 && budget>0){
        /* Tetto di staging: il piu' piccolo tra 4 GB e un ottavo del budget del tier,
         * mai meno di un esperto (altrimenti il ciclo non avanza) e mai piu' del
         * prefisso. Un ottavo mantiene i round abbastanza grandi da tenere occupati
         * i thread del carico parallelo; il tetto assoluto protegge chi ha poca RAM
         * e un budget enorme, che e' esattamente il caso segnalato. */
        double cap = 4e9; if(budget/8.0 < cap) cap = budget/8.0;
        int st = (int)(cap/eb);
        if(st < 1) st = 1;
        if(st < stage) stage = st;
        if(stage < pre_n)
            fprintf(stderr,"[CUDA] tier staging: %d experts per round (%.1f GB host peak) "
                           "instead of %d at once (%.1f GB) — CUDA_RELEASE_HOST frees each "
                           "round before the next (#730)\n",
                    stage, stage*eb/1e9, pre_n, pre_n*eb/1e9);
    }
#endif
    for(int base=0; base<pre_n; base+=stage){
    int hi = base+stage; if(hi>pre_n) hi=pre_n;
    #pragma omp parallel for schedule(dynamic,1)
    for(int a=base;a<hi;a++)
        expert_load(m,r[a].l,r[a].e,&m->pin[r[a].l][slot_of[a]],1,0);   /* startup pin load; demand=0, never classified */
    m->resident_bytes+=(int64_t)pin_range_bytes(m,r,base,hi);
#ifdef COLI_CUDA
    if(g_cuda_enabled && budget>0){
        for(int a=base;a<hi && m->gpu_expert_bytes<budget;a++){
            int li=r[a].l;
            { ESlot *s=&m->pin[li][slot_of[a]];
                int64_t need=qt_bytes(&s->g)+qt_bytes(&s->u)+qt_bytes(&s->d);
                int compress=0;
#ifdef COLI_ANS
                compress=g_cuda_raw_experts>=0&&a>=g_cuda_raw_experts;
#endif
                int64_t projected=compress?need*80/100:need;
                if(m->gpu_expert_bytes+projected>budget) break;
                int tried[COLI_CUDA_MAX_DEVICES]={0}, placed=0;
                for(int attempt=0;attempt<g_cuda_ndev && !placed;attempt++){
                    int best=-1;
                    for(int i=0;i<g_cuda_ndev;i++) if(!tried[i] && remaining[i]>=projected &&
                        (best<0||
                         (load_balance && (placed_w[i]<placed_w[best] ||
                           (placed_w[i]==placed_w[best]&&placed_b[i]<placed_b[best])))||
                         (!load_balance&&placed_b[i]<placed_b[best]))) best=i;
                    if(best<0) break;
                    tried[best]=1;
                    s->g.cuda_device=s->u.cuda_device=s->d.cuda_device=g_cuda_devices[best];
                    s->g.cuda_eligible=s->u.cuda_eligible=s->d.cuda_eligible=1;
                    int uploaded=
#ifdef COLI_ANS
                        compress ? (qt_cuda_upload_compressed(&s->g)&&qt_cuda_upload_compressed(&s->u)&&
                                    qt_cuda_upload_compressed(&s->d)) :
#endif
                        (qt_cuda_upload(&s->g) && qt_cuda_upload(&s->u) && qt_cuda_upload(&s->d));
#ifdef COLI_ANS
                    if(compress&&!uploaded){
                        fprintf(stderr,"[ANS] sidecar load failed; refusing partial VRAM placement\n");
                        exit(2);
                    }
#endif
                    if(uploaded){
                        int64_t actual=(int64_t)coli_cuda_tensor_bytes(s->g.cuda)
                                      +(int64_t)coli_cuda_tensor_bytes(s->u.cuda)
                                      +(int64_t)coli_cuda_tensor_bytes(s->d.cuda);
                        m->gpu_expert_count++; m->gpu_expert_bytes+=actual;
                        remaining[best]-=actual; placed_b[best]+=actual; placed_n[best]++;
                        placed_w[best]+=(double)r[a].c;
                        if(g_cuda_release_host){ expert_host_release(m,s); pin_host_released+=(double)need; }
                        placed=1;
                    } else {
                        qt_cuda_reset(&s->g); qt_cuda_reset(&s->u); qt_cuda_reset(&s->d);
                        s->g.cuda_eligible=s->u.cuda_eligible=s->d.cuda_eligible=0;
                        remaining[best]=0;             /* device rejected its projected capacity */
                    }
                }
            }
        }
    }
#endif
    }   /* fine del ciclo a round (#730) */
#ifdef __linux__
    g_numa_skip_bind=0;
#endif
#ifdef COLI_CUDA
    if(g_cuda_enabled && budget>0){
        fprintf(stderr,"[CUDA] hot expert tier: %d/%d experts, VRAM %.2f GB (budget %.1f GB%s, reserve %.1f GB)\n",
            m->gpu_expert_count,npin,m->gpu_expert_bytes/1e9,
            g_cuda_expert_auto?safe_total/1e9:budget/1e9,
            g_cuda_expert_auto?", auto":"",
            g_cuda_reserve_gb);
        if(load_balance) fprintf(stderr,
            "[CUDA] expert device assignment: frequency-load balanced (experimental)\n");
        for(int i=0;i<g_cuda_ndev;i++) fprintf(stderr,"[CUDA]   device %d: %d experts, %.2f GB\n",
            g_cuda_devices[i],placed_n[i],placed_b[i]/1e9);
        /* #653: on integrated / unified-memory GPUs (Grace-Blackwell GB10, Jetson) the
         * expert tier and the host RAM cache draw from ONE physical pool, so the boot
         * MemAvailable snapshot -- captured before this tier was placed -- over-counts
         * free RAM by exactly the tier size. The auto RAM budget (cap_for_ram) and the
         * overcommit guard both read g_mem_avail_boot, so correcting it here fixes both:
         * the budget shrinks by the placed bytes and no longer over-commits into an
         * OOM-kill during prefill. Discrete GPUs have a separate VRAM pool -> integrated
         * is 0 and this is a no-op, leaving their behaviour unchanged. */
        if(g_cuda_ndev>0 && m->gpu_expert_bytes>0 && g_mem_avail_boot>0 &&
           coli_cuda_device_integrated(g_cuda_devices[0])){
            double tier_gb = m->gpu_expert_bytes/1e9;
            g_mem_avail_boot -= tier_gb;
            if(g_mem_avail_boot < 1.0) g_mem_avail_boot = 1.0;
            fprintf(stderr,"[CUDA] integrated/unified memory: expert tier shares physical RAM; "
                "RAM budget snapshot reduced by %.2f GB -> MemAvailable=%.1f GB (#653)\n",
                tier_gb, g_mem_avail_boot);
        }
    }
#endif
    if(gpu_prefix>0&&gpu_prefix<npin){
        #pragma omp parallel for schedule(dynamic,1)
        for(int a=gpu_prefix;a<npin;a++)
            expert_load(m,r[a].l,r[a].e,&m->pin[r[a].l][slot_of[a]],1,0);   /* startup pin load; demand=0, never classified */
        m->resident_bytes+=(int64_t)pin_range_bytes(m,r,gpu_prefix,npin);
    }
    /* Every startup load has completed. Publish the immutable hot-store map
     * once; the session-local REPIN path maintains individual entries later. */
    for(int a=0;a<npin;a++){
        int l=r[a].l;
        pin_index(m,l,&m->pin[l][slot_of[a]]);
    }
    double warm_b=pin_range_bytes(m,r,0,npin)-pin_host_released;
    if(warm_b<0.0) warm_b=0.0;
    fprintf(stderr,"[PIN] placement: %d VRAM + %d RAM expert (%.1f GB warm) in %.0fs da %s\n",
        m->gpu_expert_count,npin-m->gpu_expert_count,warm_b/1e9,now_s()-t0,statspath);
    pin_wire(m);                                   /* inchioda in RAM (no compressione) / wire in RAM (no compression) */
    free(r); free(cnt_l); free(slot_of); free(next);
}

static double g_mem_avail_boot=0;   /* MemAvailable all'avvio, prima di caricare il modello */
/* RAM disponibile ADESSO (GB): e' il tetto vero, non il totale. Linux: MemAvailable
 * da /proc/meminfo. macOS: pagine free+inactive+purgeable da host_statistics64
 * (stessa semantica: recuperabili senza swap). Senza questo ramo il fallback
 * "assumo 8 GB" castrava la cache expert proprio sulle macchine con piu' RAM. */
static double mem_available_gb(void){
#ifdef __APPLE__
    mach_msg_type_number_t cnt=HOST_VM_INFO64_COUNT;
    vm_statistics64_data_t vm;
    if(host_statistics64(mach_host_self(),HOST_VM_INFO64,(host_info64_t)&vm,&cnt)!=KERN_SUCCESS) return 0;
    return ((double)vm.free_count+(double)vm.inactive_count+(double)vm.purgeable_count)
           * (double)sysconf(_SC_PAGESIZE) / 1e9;
#elif defined(_WIN32)
    double total, avail;
    compat_meminfo(&total, &avail);
    return avail;
#else
    FILE *f=fopen("/proc/meminfo","r"); if(!f) return 0;
    char ln[256]; double kb=0;
    while(fgets(ln,sizeof(ln),f)) if(sscanf(ln,"MemAvailable: %lf",&kb)==1) break;
    fclose(f); return kb/1e6;
#endif
}

static int kv_slot_count(void){
    if(!getenv("SERVE")) return 1;
    return getenv("KV_SLOTS")?atoi(getenv("KV_SLOTS")):1;
}

static double kv_pool_bytes(Model *m, int max_ctx){
    /* KV8: 1 byte/valore + 8 B di scale per token/layer, non 4 B/valore. E' questo
     * conto che governa expert_avail e cap_for_ram: con KV8 il clamp PIN recupera
     * ~35 GB/slot a 256k e KV_SLOTS=2 smette di demolire gli expert su disco. */
    Cfg *c=&m->c;
    double one=(double)(c->n_layers+1)*max_ctx*
        (g_tq ? (double)(coli_kvq_row_bytes(c->kv_lora,g_tq_bits,g_tq_codec)+coli_kvq_row_bytes(c->qk_rope,g_tq_bits,g_tq_codec))+8.0
         : g_kv8 ? (double)(c->kv_lora+c->qk_rope)+8.0
         : (c->kv_lora+c->qk_rope)*4.0);
    if(m->has_dsa) for(int i=0;i<c->n_layers;i++) if(c->idx_type[i])
        one+=(double)max_ctx*c->index_hd*4.0;
    int slots=kv_slot_count(); if(slots<1||slots>16) slots=1;
    return one*slots;
}


/* byte disponibili per gli expert (pin + LRU) nel budget — specchio del conto di cap_for_ram */
static double expert_avail(Model *m, double ram_gb, int ebits, int max_ctx){
    Cfg *c=&m->c; int64_t eb=expert_bytes_probe(m,ebits);
    if(ram_gb<=0){ ram_gb=g_mem_avail_boot*0.88; if(ram_gb<4) ram_gb=8; }
    double ws_b = (g_expert_budget>0 && g_expert_budget<64) ? (double)(g_expert_budget+4)*(double)eb : 64.0*(double)eb;
    double slack = 1.2e9 + 2.5e9 + ws_b
        + kv_pool_bytes(m,max_ctx)
        + (double)max_ctx*c->n_heads*(c->qk_nope+c->v_head)*4.0;
    return ram_gb*1e9 - (double)m->resident_bytes - slack;
}

/* Automatic history pinning and the adaptive LRU share the expert RAM budget.
 * Preserve the LRU capacity affordable before pinning, up to the requested
 * cap; explicit PIN/PIN_GB settings bypass this policy and remain authoritative. */
static double autopin_preserve_lru(double planned_pin, double expert_available,
                                   double lru_reserve){
    if(planned_pin<=0.0 || expert_available<=lru_reserve) return 0.0;
    double max_pin=expert_available-lru_reserve;
    return planned_pin<max_pin?planned_pin:max_pin;
}

static double autopin_lru_reserve(double expert_available, double bytes_per_slot,
                                  int requested_cap, int *preserved_cap){
    int cap=0;
    if(expert_available>0.0 && bytes_per_slot>0.0 && requested_cap>0){
        int affordable=(int)(expert_available/bytes_per_slot);
        cap=requested_cap<affordable?requested_cap:affordable;
    }
    if(preserved_cap) *preserved_cap=cap;
    return (double)cap*bytes_per_slot;
}

/* #856: nsp*widest -> the sum of each row's REAL width. The "nsp+=2" was an
 * approximation of one wide MTP row while every OTHER row was charged the narrow
 * width; once the probe started returning the widest for all of them the two
 * compounded, and the autopin reserve inherited the same halving as the cap. */
static double expert_cache_bytes_per_slot(Model *m, int ebits){
    return expert_cache_row_bytes(m,ebits);
}



/* #759: an explicit RAM_GB used to be honored literally even when it named more
 * physical memory than the machine actually has. On unified-memory hosts the
 * #653 correction shrinks the boot snapshot by the VRAM expert tier, so the gap
 * between the requested budget and reality is easy to hit; CAP_RAISE then scaled
 * the LRU against that phantom budget and the kernel OOM-killed mid-prefill.
 * Pure function (no I/O, no globals) so tests can drive it directly, same
 * contract style as coli_resolve_cap(). */
static double coli_clamp_ram_gb(double ram_gb, double mem_avail_gb, int overcommit){
    if(!overcommit && ram_gb>0 && mem_avail_gb>0 && ram_gb>mem_avail_gb)
        return mem_avail_gb;
    return ram_gb;
}

/* clampa la cache expert a un budget RAM (GB): cap t.c. residente + cache + slack <= budget.
 * ram_gb<=0 -> budget AUTO = 88% della RAM disponibile adesso (lascia respiro a OS+wrapper:
 * sforare = OOM-kill del kernel a meta' generazione, molto peggio di una cache piu' piccola). */
static void cap_for_ram(Model *m, double ram_gb, int ebits, int max_ctx){
    Cfg *c=&m->c; int nsp=0; for(int i=0;i<c->n_layers;i++) if(m->L[i].sparse) nsp++;
    if(m->has_mtp) nsp++;                        /* la riga MTP e' una riga; la sua LARGHEZZA
                                                  * la porta row_b, non piu' un "conta doppio" */
    /* eb = the WIDEST width in the container: correct for ws[], which is shared across
     * rows, and only for that. row_b = the sum of the widths the rows really hold,
     * which is what one LRU slot per row costs (#856). */
    int64_t eb=expert_bytes_probe(m,ebits);
    double row_b=expert_cache_row_bytes(m,ebits);
    int auto_b = ram_gb<=0;
    if(auto_b){ ram_gb = g_mem_avail_boot*0.88;   /* misurata PRIMA del load: il residente gia'
                                                   * allocato viene sottratto sotto, non due volte */
        if(ram_gb<4){ fprintf(stderr,"[RAM] MemAvailable is unreadable or too low; assuming 8 GB\n"); ram_gb=8; } }
    else{
        /* #759: an explicit budget larger than what is really available (the
         * #653-corrected snapshot on unified memory) would only ever be caught
         * by the OOM killer, after CAP_RAISE had scaled the LRU up against it. */
        int oc = getenv("COLI_RAM_OVERCOMMIT")?atoi(getenv("COLI_RAM_OVERCOMMIT")):0;
        double clamped = coli_clamp_ram_gb(ram_gb,g_mem_avail_boot,oc);
        if(clamped<ram_gb){
            fprintf(stderr,"[RAM_GB=%.1f] clamped to %.1f GB: that is the MemAvailable this run "
                "can actually use (#653 snapshot; COLI_RAM_OVERCOMMIT=1 keeps the literal budget)\n",
                ram_gb,clamped);
            ram_gb=clamped;
        }
    }
    g_ram_budget_gb = ram_gb;                    /* #403: la RSS-guard usa il budget RISOLTO */
    /* slack ONESTO, non forfettario (l'OOM del 2026-07-04 veniva da qui):
     *  ws[64] slab del working-set (si materializzano TUTTI nel prefill batch-union),
     *  KV cache a max_ctx, kvb_all della ricostruzione k/v in attention,
     *  attivazioni+logits+overhead ~1.2 GB */
    double ws_b  = 64.0*(double)eb;
    /* Under EXPERT_BUDGET, the block-of-64 working set is capped at budget experts
     * per layer — only ws[0..budget-1] are populated, not all 64. The 64×eb reserve
     * overcounts by 16x at budget=4, starving the LRU cache (cap 3 instead of 4).
     * Cap=4 matches budget=4, eliminating LRU thrashing that causes excessive disk
     * re-reads. Clamp ws_b to the actual budget (min 8 for non-budgeted / prefill). */
    if(g_expert_budget>0 && g_expert_budget<64) ws_b = (double)(g_expert_budget+4) * (double)eb;
    double kv_b  = kv_pool_bytes(m,max_ctx);
    double kvb_b = (double)max_ctx*c->n_heads*(c->qk_nope+c->v_head)*4.0;
    /* #768: above KVB_FLASH_MB attention_rows tiles the reconstruction instead of
     * materialising kvb_all, so the transient to reserve is the flash ceiling plus
     * one tile and the per-(row,head) running state — not 30 GB at ctx 262144.
     * The reserve mirrors the trigger in attention_rows: same env, same default. */
    { int64_t flash_mb=getenv("KVB_FLASH_MB")?atoll(getenv("KVB_FLASH_MB")):2048;
      int64_t tile_mb =getenv("KVB_TILE_MB") ?atoll(getenv("KVB_TILE_MB")) :512;
      double capped = (double)flash_mb*1048576.0 + (double)tile_mb*1048576.0
                    + (double)max_ctx*c->n_heads*2*4.0;
      if(flash_mb>0 && kvb_b>capped) kvb_b = capped; }
    /* RISERVA PAGE-CACHE (misurato 2026-07-06 su Linux): strangolarla fa crollare
     * le pread buffered da ~800 a ~180 MB/s — gli ultimi GB di LRU rendono MENO di
     * quanto costino in banda disco persa. 2.5 GB restano SEMPRE al kernel.
     * NOTE: tested removing this under Windows+DIRECT (it should be dead weight when
     * O_DIRECT bypasses the buffer cache). Result: cap went 4->5 but RSS hit 24 GB
     * on a 32 GB machine, causing memory pressure that DROPPED the hit rate (73%->57%)
     * and slowed decode (1.03->0.83 tok/s). The reserve is a legitimate safety margin
     * for OS + CUDA + file metadata, not just buffered pread throughput. Keep it. */
    double pc_b  = 2.5e9;
    double slack = 1.2e9 + pc_b + ws_b + kv_b + kvb_b;
    double avail = ram_gb*1e9 - (double)m->resident_bytes - slack;
    int capmax = (avail>0 && row_b>0) ? (int)(avail/row_b) : 0;
    int floored = capmax<1;   /* il budget non regge nemmeno UNO slot per layer */
    if(capmax<1) capmax=1;
    /* Il floor a 1 e' una bugia comoda: con avail negativo capmax sarebbe 0, cioe'
     * "non ci sto nel tuo budget". Alzarlo a 1 e proseguire trasforma "non ci sto"
     * in "sforo" -- ed e' esattamente l'OOM-kill a meta' generazione che questa
     * funzione esiste per evitare. Il kernel uccide con SIGKILL: nessun errore,
     * nessun log, il motore muore muto (issue #305). Dirlo, e fermarsi se il picco
     * non entra nemmeno nella RAM realmente disponibile misurata all'avvio. */
    if(floored){
        double peak = (double)m->resident_bytes + (double)capmax*row_b + slack;
        /* The width is printed because it is the one term nobody can check from
         * outside: every projection here divides by it, so a wrong width is
         * indistinguishable from a wrong budget in this message (#766). Report the
         * per-row AVERAGE, since that is now the divisor -- printing the widest while
         * dividing by something else is how #856 stayed invisible for a release. */
        fprintf(stderr,"[RAM_GB=%.1f%s] WARNING: cap=1 is the floor, projected peak %.1f GB is "
            "%.1f GB OVER the budget (resident %.1f GB + reserve %.1f GB, expert %.1f MB avg/row).%s\n",
            ram_gb,auto_b?" auto":"",peak/1e9,(peak-ram_gb*1e9)/1e9,
            m->resident_bytes/1e9,slack/1e9,row_b/(nsp>0?nsp:1)/1e6,
            getenv("PIN_GB")?" PIN_GB is inflating the resident set: lower it or drop it.":"");
#ifdef COLI_CUDA
        /* #686: on a single GPU that also holds host copies of the VRAM tier, the
         * thing inflating the resident set is often those copies rather than
         * PIN_GB itself -- name the knob instead of leaving the user to find it. */
        if(g_cuda_enabled && g_cuda_ndev==1 && !g_cuda_release_host &&
           (g_cuda_expert_gb>0||g_cuda_expert_auto))
            fprintf(stderr,"[RAM] the VRAM expert tier still has host copies in RAM "
                "(CUDA_RELEASE_HOST=0 on a single GPU): CUDA_RELEASE_HOST=1 frees them for "
                "the RAM tier and is what this topology usually wants (#686).\n");
#endif
        if(g_mem_avail_boot>0 && peak > g_mem_avail_boot*1e9 &&
           !(getenv("COLI_RAM_OVERCOMMIT") && atoi(getenv("COLI_RAM_OVERCOMMIT")))){
            fprintf(stderr,"[RAM] refusing to start: that peak also exceeds the %.1f GB actually "
                "available on this machine, so the kernel would OOM-kill this run mid-generation.\n"
                "[RAM] lower PIN_GB, lower the context, or raise the RAM budget if the box really has it "
                "(COLI_RAM_OVERCOMMIT=1 overrides this check).\n", g_mem_avail_boot);
            exit(2);
        }
    }
    if(capmax < m->ecap){
        fprintf(stderr,"[RAM_GB=%.1f%s] resident %.1f GB + reserve %.1f GB (ws %.1f, KV %dx%d %.1f, kvb %.1f), "
            "experts %.1f MB avg x %d layers -> cap lowered %d->%d (projected peak %.1f GB)\n",
            ram_gb,auto_b?" auto":"",m->resident_bytes/1e9,slack/1e9,ws_b/1e9,
            kv_slot_count(),max_ctx,kv_b/1e9,kvb_b/1e9,
            row_b/(nsp>0?nsp:1)/1e6, nsp, m->ecap, capmax,
            (m->resident_bytes + (double)capmax*row_b + slack)/1e9);
        m->ecap=capmax;
    } else {
        /* AUTO-RAISE (issue #12): il budget consente PIU' cache di quella chiesta.
         * Senza questo, una macchina da 128 GB girava con la LRU di una da 16
         * (cap=8 di default in coli): hit 23-28% con decine di GB inutilizzati.
         * Tetto a n_experts: oltre, ogni layer avrebbe slot che non puo' riempire.
         * CAP_RAISE=0 ripristina il comportamento fisso.
         * #379: on Metal + darwin + fast SSD, raising the cache back up is the
         * same anti-pattern the platform cap default exists to avoid, so the
         * default flips to off there. An explicit CAP_RAISE always wins either
         * way -- this only changes what happens when nobody set it.
         * CURRENT-STATE CALIBRATION (#379): same revisit trigger and
         * measurement stamp as the cap-1 default in coli_resolve_cap(). */
        int raise_on = getenv("CAP_RAISE")?atoi(getenv("CAP_RAISE")):(g_ssd_fast?0:1);
        int newcap = capmax>c->n_experts ? c->n_experts : capmax;
        if(raise_on && newcap>m->ecap){
            for(int i=0;i<=c->n_layers;i++) if(m->ecache[i]){
                m->ecache[i]=realloc(m->ecache[i],(size_t)newcap*sizeof(ESlot));
                memset(m->ecache[i]+m->ecap,0,(size_t)(newcap-m->ecap)*sizeof(ESlot));
            }
            fprintf(stderr,"[RAM_GB=%.1f%s] cap raised %d->%d: budget allows it "
                "(projected peak %.1f GB; set CAP_RAISE=0 to disable)\n",
                ram_gb, auto_b?" auto":"", m->ecap, newcap,
                (m->resident_bytes + (double)newcap*row_b + slack)/1e9);
            m->ecap=newcap;
        } else
            fprintf(stderr,"[RAM_GB=%.1f%s] cap=%d ok (projected peak %.1f GB)\n", ram_gb, auto_b?" auto":"", m->ecap,
                (m->resident_bytes + (double)m->ecap*row_b + slack)/1e9);
    }
}

/* The user's generation prompt. COLI_PROMPT is honored on every platform; a bare
 * PROMPT is honored too, EXCEPT on Windows, where cmd.exe always exports its own
 * PROMPT template (default "$P$G", the thing that draws "C:\...>") into the child's
 * environment. That is a shell UI string, not a prompt: taking it would send the
 * engine into text-generation mode (needing a tokenizer) instead of the oracle
 * self-test, and would "generate" from "$P$G". So on Windows a PROMPT carrying
 * cmd's $-metacodes is ignored; set COLI_PROMPT to pass a real prompt from cmd. */
static const char *coli_user_prompt(void){
    const char *p = getenv_utf8("COLI_PROMPT");
    if(p) return p;
    p = getenv_utf8("PROMPT");
#ifdef _WIN32
    if(p) for(const char *q=p; q[0]; q++)
        if(q[0]=='$' && q[1] && strchr("ABCDEFGHLNPQSTV_+|$", q[1]&~0x20)){ p=NULL; break; }
#endif
    return p;
}

/* PROF=1 startup header: one self-describing block so a saved log answers
 * "what machine, what config" when comparing runs after changing RAM_GB,
 * knobs, or moving to another host. */
static void prof_config(Model *m, double ram_env, int est_ctx){
    Cfg *c=&m->c;
    char cpu[256]; int cores; double rt,ra;
    hw_probe(cpu,sizeof(cpu),&cores,&rt,&ra);
    const char *backend="CPU";
#ifdef COLI_CUDA
    if(g_cuda_enabled) backend="CUDA";
#endif
#ifdef COLI_METAL
    if(g_metal_enabled) backend="Metal";
#endif
    int nsp=0; for(int i=0;i<c->n_layers;i++) if(m->L[i].sparse) nsp++;
    /* #856: the cache figure comes from the same per-row sum the cap is computed
     * from. Printing a projection derived differently from the cap is how the
     * halving stayed invisible through a whole release. */
    int pinned=0; double pin_b=0;
    for(int i=0;i<=c->n_layers;i++) if(m->npin){
        pinned+=m->npin[i];
        pin_b+=(double)m->npin[i]*(double)expert_bytes_row(m,i,m->ebits);
    }
    (void)nsp;
    fprintf(stderr,"[PROF] machine: %s | %d cores (%d omp threads) | RAM %.1f GB total, %.1f GB available | backend %s\n",
        cpu[0]?cpu:"unknown CPU",cores,omp_get_max_threads(),rt,ra,backend);
    fprintf(stderr,"[PROF] config: RAM_GB=%s%.1f CTX=%d | expert cache cap %d/layer (up to %.1f GB) | pinned %d (%.1f GB) | "
        "DRAFT=%d PIPE=%d DIRECT=%d MMAP=%d IDOT=%d DSA=%s PILOT=%d CACHE_ROUTE=%d\n",
        ram_env<=0?"auto ":"",ram_env<=0?g_mem_avail_boot*0.88:ram_env,est_ctx,
        m->ecap,(double)m->ecap*expert_cache_row_bytes(m,m->ebits)/1e9,pinned,pin_b/1e9,
        g_draft,g_pipe,g_direct,g_mmap,g_idot,
        (m->has_dsa&&c->index_topk)?"on":"off",g_pilot,g_cache_route);
}

/* ---- #379: honest Metal-cache storage probe (S1) --------------------------
 * Metal's per-buffer residency cost makes the engine's own expert LRU cache
 * anti-productive on fast NVMe: the OS page cache already streams cold
 * experts cheaper than Metal can re-pin them (maintainer-endorsed root cause,
 * issue #379). "Fast" has to be MEASURED, not assumed -- buffered reads lie
 * (page-cache hits read 23-97 GB/s on the reference box; the real F_NOCACHE
 * number was 14 GB/s), so this probes the actual model volume the same way
 * compat_open_direct()/iobench.c's __APPLE__ branch do: open + F_NOCACHE,
 * random 16K-aligned pread, no readahead. F_NOCACHE alone is NOT enough: it
 * bypasses the cache for NEW pages but still serves already-resident ones, so
 * a warm shard measured 22.1 GB/s of RAM and latched it as "disk". The probe
 * therefore steers its reads to ranges mincore reports cold, and refuses to
 * conclude anything when the file offers too few cold bytes (veto below).
 * Gating: the whole block builds only for the one consumer (COLI_METAL on
 * __APPLE__), so the default build carries no orphaned probe code; tests
 * define COLI_SSD_PROBE_TEST to compile it anywhere -- the decision layer
 * (grammar parser, cold-tile steering, veto) is pure portable C by design,
 * exercised by tests/test_ssd_probe.c on every platform. */
#if (defined(COLI_METAL) && defined(__APPLE__)) || defined(COLI_SSD_PROBE_TEST)
#define COLI_SSD_PROBE_BLK   (4u*1024*1024)     /* 4 MB reads, 16K-aligned offsets */
#define COLI_SSD_PROBE_MAXB  (300*1024*1024)    /* "a few hundred MB", capped on fast media */
#define COLI_SSD_PROBE_SECS  0.35               /* wall-clock budget, well under the ~1s target */
#define COLI_SSD_PROBE_COLD_FLOOR (64ll*1024*1024) /* fewer cold bytes than this = contaminated */
#define COLI_SSD_CACHE_MAX   64                 /* longest well-formed .coli_ssd, in bytes */

/* Why a measurement was refused (contaminated-class: nothing cached, one
 * honest stderr line naming the ACTUAL condition, retry on a later start).
 * The reasons differ in what the user should do about them -- "wait for a
 * cold start" vs "finish the download" vs "nothing, this shard can never
 * host a trustworthy probe" -- so the message must not lump them. */
enum {
    COLI_SSD_VETO_NONE = 0,
    COLI_SSD_VETO_RESIDENT,                     /* too few cold pages: page cache would answer */
    COLI_SSD_VETO_SMALL,                        /* shard offers < COLD_FLOOR of whole windows, ever */
    COLI_SSD_VETO_SPARSE                        /* under-allocated: holes read as VFS zero-fill */
};

/* Under-allocation gate (#386 round 2, F1): steering prefers never-resident
 * pages, and in a sparse or still-downloading file those are HOLES -- pread
 * of a hole is VFS zero-fill at RAM speed (a 10 GB shard with 100 MB of real
 * data measured 73 GB/s and latched it as disk bandwidth). st_blocks*512 is
 * what actually sits on disk; require it to cover the file size within 12.5%
 * slack (HFS+/APFS metadata rounding, tail blocks). Same conservative
 * polarity as the residency veto: not fully allocated -> not measurable. */
static int coli_ssd_probe_underallocated(long long allocated, long long size){
    return allocated < size - size/8;
}

/* digits [ "." digits ] -- returns chars consumed, 0 = no match. The ONLY
 * number shape the cache grammar admits (see coli_ssd_cache_parse). */
static size_t coli_ssd_scan_number(const char *p, size_t len){
    size_t i=0;
    while(i<len && p[i]>='0' && p[i]<='9') i++;
    if(!i) return 0;
    if(i<len && p[i]=='.'){
        size_t j=i+1;
        while(j<len && p[j]>='0' && p[j]<='9') j++;
        if(j==i+1) return 0;                     /* "5." is garbage */
        i=j;
    }
    return i;
}

/* Strict .coli_ssd grammar -- mirrored byte-for-byte by resource_plan.py's
 * parse_ssd_cache() (keep the two in lockstep; test_ssd_probe.c and
 * test_resource_plan.py chew the same vector file,
 * tests/fixtures/ssd_cache_vectors.txt):
 *   v2:     "v2 <gbs> <st_dev>"   single spaces, at most one trailing \n
 *   legacy: "<gbs>"               the pre-fix format, never trusted (re-probed)
 * where <gbs> = digits["."digits], 0 < gbs < 1000, and <st_dev> = 1..20
 * digits fitting unsigned 64-bit. Total length <= COLI_SSD_CACHE_MAX bytes,
 * no NULs, nothing else. Everything outside the grammar is garbage and
 * re-probes: strtod permissiveness ("inf", "nan", "0x8", "1e99", leading
 * whitespace/signs) is deliberately out -- it let hand-edited or corrupt
 * caches masquerade as measurements.
 * Returns 2 = v2 (gbs+stdev filled), 1 = legacy (gbs filled), 0 = garbage. */
static int coli_ssd_cache_parse(const char *buf, size_t len, double *gbs_out,
                                unsigned long long *stdev_out){
    if(!len || len>COLI_SSD_CACHE_MAX || memchr(buf,0,len)) return 0;
    if(buf[len-1]=='\n') len--;                  /* at most one trailing newline */
    int v2 = len>=3 && !memcmp(buf,"v2 ",3);
    const char *p = buf + (v2?3:0);
    size_t left = len - (v2?3:0);
    size_t n = coli_ssd_scan_number(p,left);
    if(!n) return 0;
    char num[COLI_SSD_CACHE_MAX+1];
    memcpy(num,p,n); num[n]=0;
    double gbs=strtod(num,NULL);                 /* shape already vetted above */
    if(!(gbs>0 && gbs<1000)) return 0;
    p+=n; left-=n;
    if(!v2){
        if(left) return 0;
        *gbs_out=gbs;
        return 1;
    }
    if(!left || *p!=' ') return 0;
    p++; left--;
    size_t d=0;
    while(d<left && p[d]>='0' && p[d]<='9') d++;
    if(!d || d!=left || d>20) return 0;
    char dev[21];
    memcpy(dev,p,d); dev[d]=0;
    errno=0;
    unsigned long long sd=strtoull(dev,NULL,10);
    if(errno) return 0;                          /* 20 nines overflows u64: garbage */
    *gbs_out=gbs; *stdev_out=sd;
    return 2;
}

/* The single writer of the grammar above: "v2 %.3f %llu\n". %.3f of a value
 * in (0,1000) is always digits.digits, so what this writes always re-parses
 * -- the round-trip is pinned by test_ssd_probe.c. */
static int coli_ssd_cache_format(char *buf, size_t bufsz, double gbs,
                                 unsigned long long stdev){
    int n=snprintf(buf,bufsz,"v2 %.3f %llu\n",gbs,stdev);
    return (n>0 && (size_t)n<bufsz) ? n : -1;
}

/* Probe steering (#386 fix round): pure, so tests inject residency vectors --
 * no mincore, no files. vec has one byte per page_sz-sized page, mincore
 * convention (bit 0 set = resident in RAM). The file is tiled into
 * non-overlapping COLI_SSD_PROBE_BLK windows; a window qualifies only when
 * EVERY page in it is non-resident, because one warm page lets the page cache
 * serve part of a "disk" read (a fully-resident shard measured 22.1 GB/s
 * through F_NOCACHE and latched the fast-SSD default on unknown hardware).
 * Emits qualifying window start offsets into tiles[] (caller sizes it at
 * file_size/BLK+1 entries) and returns the count; *cold_bytes_out gets
 * count*BLK, the bytes the probe can trust. */
static size_t coli_ssd_probe_cold_tiles(const unsigned char *vec, size_t npages,
                                        size_t page_sz, long long *tiles,
                                        long long *cold_bytes_out){
    size_t count=0;
    if(page_sz && page_sz<=COLI_SSD_PROBE_BLK){
        size_t per=(COLI_SSD_PROBE_BLK+page_sz-1)/page_sz;
        for(size_t t=0; t+per<=npages; t+=per){
            size_t i=0;
            while(i<per && !(vec[t+i]&1)) i++;
            if(i==per) tiles[count++]=(long long)t*(long long)page_sz;
        }
    }
    if(cold_bytes_out) *cold_bytes_out=(long long)count*COLI_SSD_PROBE_BLK;
    return count;
}

/* Contamination veto (#386 fix round): with fewer than COLI_SSD_PROBE_COLD_FLOOR
 * verified-cold bytes to read, any measurement would be mostly RAM. An
 * untrustworthy measurement behaves as SLOW: no number, no cache write, the
 * conservative default holds, and the probe retries on a colder startup. */
static int coli_ssd_probe_vetoed(long long cold_bytes){
    return cold_bytes < COLI_SSD_PROBE_COLD_FLOOR;
}

#ifdef __APPLE__
/* Path of the LARGEST *.safetensors shard in snap_dir into out, or out[0]=0
 * if none. Any shard's bandwidth is representative, but the biggest one
 * offers the most cold 4 MB windows -- a stray small shard picked by readdir
 * order could starve the probe below the 64 MB floor forever (#386 round 2,
 * F4) even though a full-size neighbor sits right next to it. */
static void coli_ssd_probe_pick_shard(const char *snap_dir, char *out, size_t outsz){
    out[0]=0;
    DIR *d=opendir(snap_dir);
    if(!d) return;
    struct dirent *e;
    long long best=-1;
    while((e=readdir(d))){
        const char *dot=strrchr(e->d_name,'.');
        if(!dot || strcmp(dot,".safetensors")) continue;
        char cand[1280];
        snprintf(cand,sizeof(cand),"%s/%s",snap_dir,e->d_name);
        struct stat st;
        if(stat(cand,&st)==0 && (long long)st.st_size>best){
            best=(long long)st.st_size;
            snprintf(out,outsz,"%s",cand);
        }
    }
    closedir(d);
}

/* F_NOCACHE random-read probe on one real shard, steered to verified-cold
 * ranges. Returns measured GB/s, or -1 if the probe could not run (no shard,
 * open/map failure, file too small) or was vetoed as contaminated -- callers
 * treat -1 as "not fast", never as an error to surface; *veto_out gets the
 * COLI_SSD_VETO_* reason so the caller can say honestly why and skip the
 * cache write. Veto order: SPARSE (holes would measure as zero-fill) before
 * SMALL (can never offer the floor) before RESIDENT (cold today, maybe).
 * The mmap is address space only (nothing is faulted in): it exists so
 * mincore can fill the residency vector. */
static double coli_ssd_probe_raw(const char *snap_dir, int *veto_out){
    if(veto_out) *veto_out=COLI_SSD_VETO_NONE;
    char path[1280];
    coli_ssd_probe_pick_shard(snap_dir,path,sizeof(path));
    if(!path[0]) return -1;
    int fd=open(path,O_RDONLY);
    if(fd<0) return -1;
    fcntl(fd,F_NOCACHE,1);
    fcntl(fd,F_RDAHEAD,0);                       /* random access, not sequential streaming */
    struct stat st;
    if(fstat(fd,&st)!=0 || st.st_size<(off_t)(COLI_SSD_PROBE_BLK*2)){ close(fd); return -1; }
    if(coli_ssd_probe_underallocated((long long)st.st_blocks*512,(long long)st.st_size)){
        if(veto_out) *veto_out=COLI_SSD_VETO_SPARSE;
        close(fd);
        return -1;
    }
    if(((long long)st.st_size/COLI_SSD_PROBE_BLK)*COLI_SSD_PROBE_BLK<COLI_SSD_PROBE_COLD_FLOOR){
        if(veto_out) *veto_out=COLI_SSD_VETO_SMALL;   /* even fully cold it cannot reach the floor */
        close(fd);
        return -1;
    }
    size_t page_sz=(size_t)getpagesize();
    size_t npages=((size_t)st.st_size+page_sz-1)/page_sz;
    unsigned char *vec=malloc(npages);
    long long *tiles=malloc(((size_t)st.st_size/COLI_SSD_PROBE_BLK+1)*sizeof *tiles);
    void *map=(vec&&tiles)?mmap(NULL,(size_t)st.st_size,PROT_READ,MAP_SHARED,fd,0):MAP_FAILED;
    double gbs=-1;
    if(map!=MAP_FAILED && mincore(map,(size_t)st.st_size,(char*)vec)==0){
        long long cold_bytes=0;
        size_t ntiles=coli_ssd_probe_cold_tiles(vec,npages,page_sz,tiles,&cold_bytes);
        if(ntiles==0 || coli_ssd_probe_vetoed(cold_bytes)){
            /* ntiles==0 is implied by the floor today, but the modulo below
             * must NEVER be reachable with 0 on its own merits (#386 r2, F7) */
            if(veto_out) *veto_out=COLI_SSD_VETO_RESIDENT;
        }else{
            void *buf=NULL;
            if(posix_memalign(&buf,16384,COLI_SSD_PROBE_BLK)!=0) buf=NULL;
            if(buf){
                unsigned seed=0xC0117125u;
                double t0=now_s(); long long total=0;
                while(now_s()-t0<COLI_SSD_PROBE_SECS && total<COLI_SSD_PROBE_MAXB){
                    long long off=tiles[(size_t)rand_r(&seed)%ntiles];
                    ssize_t got=pread(fd,buf,COLI_SSD_PROBE_BLK,off);
                    if(got<=0) break;
                    total+=got;
                }
                double dt=now_s()-t0;
                if(total>0 && dt>0) gbs=(double)total/1e9/dt;
                free(buf);
            }
        }
    }
    if(map!=MAP_FAILED) munmap(map,(size_t)st.st_size);
    free(tiles); free(vec);
    close(fd);
    return gbs;
}

/* Cache the measurement in <snap>/.coli_ssd so every startup after the first
 * reads a file instead of re-probing. v2 records the volume's identity
 * ("v2 <gbs> <st_dev>"): a cache is trusted ONLY when it parses under the
 * strict grammar above AND its st_dev matches the model dir's current device
 * -- a dir rsync'd to another volume (external drive, RAM disk) carries a
 * measurement of the WRONG hardware and must re-probe. Legacy bare-number
 * caches (written before steering existed, so possibly warm-contaminated)
 * re-probe once and upgrade to v2. A vetoed run writes nothing and says why
 * on stderr; the probe retries naturally on the next, colder, startup. */
static double coli_ssd_probe_cached(const char *snap_dir){
    char cpath[1290];
    snprintf(cpath,sizeof(cpath),"%s/.coli_ssd",snap_dir);
    struct stat dst;
    unsigned long long cur_dev = stat(snap_dir,&dst)==0 ? (unsigned long long)dst.st_dev : 0;
    FILE *f=fopen(cpath,"rb");                   /* exact bytes: never let a text-mode
                                                  * layer eat the \r the grammar rejects */
    if(f){
        char buf[COLI_SSD_CACHE_MAX+2];
        size_t n=fread(buf,1,sizeof(buf),f);
        fclose(f);
        double cached=0; unsigned long long sdev=0;
        if(coli_ssd_cache_parse(buf,n,&cached,&sdev)==2 && sdev==cur_dev)
            return cached;
        /* v2 from another volume, legacy, or garbage: fall through, re-measure */
    }
    int veto=COLI_SSD_VETO_NONE;
    double gbs=coli_ssd_probe_raw(snap_dir,&veto);
    if(veto!=COLI_SSD_VETO_NONE){
        /* one honest line per deferral, naming the condition that actually
         * fired (#386 r2, F4): "resident" advice (wait for a cold start) is
         * wrong advice for a partial download or a permanently tiny shard. */
        const char *why =
            veto==COLI_SSD_VETO_SPARSE ? "the shard is not fully allocated on disk (partial "
                                         "download?), holes would measure as RAM-speed zero-fill" :
            veto==COLI_SSD_VETO_SMALL  ? "the largest shard offers under 64 MB of probe windows, "
                                         "too small for a trustworthy measurement" :
                                         "the shard is already page-cache resident (<64 MB cold), "
                                         "a read now would measure RAM, not the disk";
        fprintf(stderr,"METAL: storage probe deferred -- %s; keeping the conservative cache "
                "default, will re-check on a later start\n",why);
        return -1;
    }
    if(gbs<0) return -1;                         /* could not run at all: no shard, open failure */
    if(!(gbs>0 && gbs<1000)){
        /* a number the cache grammar itself would refuse (0, >=1000 GB/s,
         * non-finite) is no more trustworthy for THIS run than for the next
         * one (#386 r2, F6): treat it as contaminated, not as "fast". */
        fprintf(stderr,"METAL: storage probe measured an implausible %g GB/s -- ignoring the "
                "measurement, keeping the conservative cache default\n",gbs);
        return -1;
    }
    char line[COLI_SSD_CACHE_MAX+1];
    if(coli_ssd_cache_format(line,sizeof(line),gbs,cur_dev)>0){
        /* mkstemp (unique name, O_EXCL) + rename: the model dir is shared (a
         * concurrent serve + run pair may both probe a virgin dir) so a torn
         * write must never survive, and a fixed, predictable temp name could
         * be pre-planted as a symlink -- mkstemp cannot follow one. Two
         * racing probers both measure under mutual contention: a LOW reading,
         * the safe direction (the platform default just stays off). */
        char tpath[1310];
        snprintf(tpath,sizeof(tpath),"%s.XXXXXX",cpath);
        int cache_errno=0;
        int tfd=mkstemp(tpath);
        if(tfd<0) cache_errno=errno;
        else{
            size_t len=strlen(line);
            mode_t um=umask(0); umask(um);       /* read it back; umask() has no getter */
            int ok = write(tfd,line,len)==(ssize_t)len;
            ok = fchmod(tfd,(mode_t)(0644&~um))==0 && ok;  /* mkstemp forces 0600; honor the umask instead */
            if(!ok) cache_errno=errno;
            close(tfd);
            if(ok && rename(tpath,cpath)!=0){ cache_errno=errno; ok=0; }
            if(!ok) unlink(tpath);
        }
        if(cache_errno)
            /* the measurement still governs THIS run; only persistence failed
             * (read-only model dir, quota, ...) -- say so once (#386 r2, F5) */
            fprintf(stderr,"METAL: could not cache the storage probe in %s (%s) -- "
                    "will re-measure on every start until it is writable\n",
                    cpath,strerror(cache_errno));
    }
    return gbs;
}
#endif /* __APPLE__ */
#endif /* (COLI_METAL && __APPLE__) || COLI_SSD_PROBE_TEST */

/* Expert-cache-cap precedence (#379, S2): explicit CLI positional > explicit
 * CAP env > platform default (Metal + darwin + fast SSD) > historic default.
 * `cli_given` distinguishes a bare invocation (no positional at all -> the
 * engine's historic 64, byte-identical to the old argc<=1 fallback) from the
 * coli wrapper's "0 = auto" sentinel (-> historic 8, what coli always forced
 * before). cap=0 as an explicit request for zero cache slots was undocumented
 * and unused repo-wide, so repurposing it as the sentinel is safe. Pure
 * function (no I/O, no globals) so precedence is unit-testable in isolation
 * from the probe and from argv/getenv. */
static int coli_resolve_cap(int cli_given, int cli, int env, int platform_fast, int *explicit_out){
    int is_explicit = (cli_given && cli!=0) || (env!=0);
    if(explicit_out) *explicit_out=is_explicit;
    if(cli_given && cli) return cli;
    if(env) return env;
    /* CURRENT-STATE CALIBRATION (#379; measured 2026-07, macOS 26.5, M5 Max,
     * engine base caa49f7): cap 1 encodes that stack's per-buffer residency
     * cost. Revisit when residency handling changes (the heap/residency-set
     * work in #379, or MTLTensor-based expert storage): with heap-backed
     * slabs cap 16 already beats cap 1. The machinery above and below is
     * durable; only this constant changes. */
    if(platform_fast) return 1;
    return cli_given ? 8 : 64;   /* sentinel 0 -> coli's historic 8; bare ./glm -> historic 64 */
}

/* Una variabile di STATO e' attiva se e' impostata E non vale 0/false/off/no.
 * Volutamente diverso dai kill-switch, che restano presence-based. */
static int coli_env_on(const char *name)
{
    const char *v = getenv(name);
    if(!v || !*v) return 0;
    return !(strcmp(v,"0")==0 || strcmp(v,"false")==0 ||
             strcmp(v,"off")==0 || strcmp(v,"no")==0);
}

#ifndef COLIBRI_NO_MAIN
int main(int argc, char **argv){
    /* ---- Permanent OpenMP hot-thread tuning. The per-expert matmul regions are
     * tiny and back-to-back; with the default passive wait policy libgomp parks
     * the worker team between regions and the re-wake latency dominates. Keeping
     * the threads hot (active spin) collapses that overhead — measured matmul
     * time 66.9s -> 20.9s on the Zen5 build, with no change to numerical output.
     *
     * libgomp reads the OMP_ / GOMP_ vars in a CONSTRUCTOR that runs before
     * main(), so setenv() here and continuing would be too late (verified:
     * setenv-in-main is ignored by the already-initialised runtime). Instead, on
     * first entry seed the winning defaults — respecting anything the user
     * already set (overwrite=0) — then re-exec self once so a fresh libgomp
     * constructor picks them up. The COLI_OMP_TUNED sentinel guards the exec so
     * we re-exec at most once. Fully overridable: any explicit OMP_/GOMP_ env the
     * user sets wins (overwrite=0), pre-setting COLI_OMP_TUNED=1 skips the
     * re-exec entirely (runs with whatever policy the environment already has),
     * and COLI_NO_OMP_TUNE=1 is a documented kill-switch that disables the whole
     * re-exec + tuning path (distinct from the internal COLI_OMP_TUNED sentinel).
     *
     * Must remain the FIRST statement in main(): argv is passed verbatim to execv(). */
    /* COLI_OMP_TUNED e COLI_NO_OMP_TUNE sono KILL-SWITCH: presence-based e' voluto
     * (c/coli lo documenta: "impostarla a qualsiasi valore, anche 0, disattiva").
     * COLI_CUDA e COLI_METAL invece sono STATO, e uno 0 significa "niente GPU":
     * testarne la presenza faceva saltare il tuning proprio a chi la GPU non ce
     * l'ha. Ed e' raggiungibile, non teorico -- `coli` stesso scrive
     * e["COLI_CUDA"]="0" (c/coli:281 e :360) quando il piano non usa CUDA, quindi
     * su un host Linux CPU-only un `coli run --auto-tier --gpu none` documentato
     * perdeva silenziosamente il tuning che un `./colibri` nudo riceve. Grazie a
     * @fredchu per averlo letto nel sorgente e detto chiaramente di non averlo
     * misurato (#707). */
    if(!getenv("COLI_OMP_TUNED") && !getenv("COLI_NO_OMP_TUNE") &&
       !coli_env_on("COLI_CUDA") && !coli_env_on("COLI_METAL")){
        /* #707: the spin-wait knobs are measured HARMFUL on macOS / Apple
         * Silicon with LLVM libomp. Reporter (M1 Max, 32 GB, GLM-5.2 int4):
         * OMP_WAIT_POLICY=active alone +122% decode, KMP_BLOCKTIME=200 alone
         * +115%. Reproduced on an M3 16 GB with the OLMoE engine (same libomp
         * behaviour, smaller scale): wait-only 2.23 tok/s vs ~3.4 baseline
         * (-34%), block-only 2.54 (-25%). The Linux/FreeBSD re-exec below
         * never runs on Darwin (libomp's constructor already read the
         * environment), so these setenvs are inert today -- but keep them OFF
         * explicitly so a future Darwin exec path cannot apply a measured
         * regression. The OMP_PROC_BIND/OMP_DYNAMIC pair is noise-level and
         * stays for all platforms. */
#ifndef __APPLE__
        setenv("OMP_WAIT_POLICY","active",0);  /* keep the team hot across the tiny per-expert matmul regions */
        setenv("GOMP_SPINCOUNT","200000",0);   /* spin briefly, then yield so long disk waits don't burn a core */
        /* LLVM libomp (clang builds: FreeBSD cc, some Linux setups) does not
         * read GOMP_*: with OMP_WAIT_POLICY=active it sets KMP_BLOCKTIME=infinite,
         * so the idle team SPINS FOREVER once generation ends — a serve-mode engine
         * parked on stdin burns ~100% x nthreads (#341, measured 3000% on FreeBSD).
         * 200 ms of blocktime keeps the team hot across back-to-back expert matmuls
         * and lets it sleep at the prompt. libgomp ignores KMP_*; overwrite=0 keeps
         * the user's own setting authoritative. */
        setenv("KMP_BLOCKTIME","200",0);
#else
        fprintf(stderr,"[OMP] hot-thread tuning skipped on macOS (#707): spin-wait knobs "
                       "measured slower on Apple Silicon (COLI_NO_OMP_TUNE=1 to skip)\n");
#endif
        setenv("OMP_PROC_BIND","close",0);     /* pack the team onto adjacent cores for cache locality */
        setenv("OMP_DYNAMIC","FALSE",0);       /* fixed team size: no per-region thread-count churn */
        setenv("COLI_OMP_TUNED","1",1);
#ifdef __linux__
        fprintf(stderr,"[OMP] hot-thread tuning: re-exec once (COLI_NO_OMP_TUNE=1 to skip)\n");
        /* #471: execv PRESERVES the CPU affinity mask. If the user exported
         * OMP_PROC_BIND/OMP_PLACES, libgomp's constructor already bound THIS thread to
         * place 0 (one core's SMT siblings) before main() ran; the re-exec'd image would
         * inherit that 1-core mask, enumerate OMP_PLACES=cores inside it, and jail the
         * whole team on one core (measured ~20x slowdown). Reset to all online CPUs so
         * the fresh libgomp binds from the full set — the user's OMP_* env still wins. */
        /* CPU_SETSIZE is only exposed when _GNU_SOURCE was defined before the first
         * system header. The standalone engine build defines it at the top of this
         * file so the reset is active where it matters; test TUs that #include this
         * file after <assert.h>/<math.h> get it too late, so guard for them (a test
         * never re-execs — skipping the reset there is harmless). */
#ifdef CPU_SETSIZE
        { cpu_set_t all; CPU_ZERO(&all);
          long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
          if(ncpu > CPU_SETSIZE) ncpu = CPU_SETSIZE;
          for(long i = 0; i < ncpu; i++) CPU_SET((int)i, &all);
          if(sched_setaffinity(0, sizeof(all), &all) != 0)
              perror("[OMP] sched_setaffinity pre-reexec (continuing)"); }
#endif
        execv("/proc/self/exe", argv);         /* returns only on failure -> fall through and run untuned */
        perror("[OMP] execv self-reexec failed, running untuned");
#endif
#ifdef __FreeBSD__
        fprintf(stderr,"[OMP] hot-thread tuning: re-exec once (COLI_NO_OMP_TUNE=1 to skip)\n");
        execv("/proc/curproc/file", argv);         /* returns only on failure -> fall through and run untuned */
        perror("[OMP] execv self-reexec failed, running untuned");
#endif
    }
    /* #718: the hot-team block above tunes wake latency but historically left
     * GLM at libgomp's logical-CPU default.  Memory-bound quantized matmuls can
     * collapse when SMT siblings share each core, measured 2.3x on a 5950X.
     * The shared helper already protects kimi_k3/olmoe: apply its independent
     * physical-core sizing here too.  This must stay after the possible re-exec;
     * omp_set_num_threads() is a runtime API and needs no second exec. */
    coli_omp_tune_threads("colibri");
#ifdef _WIN32
    _setmode(fileno(stdout), O_BINARY);
#endif
#if defined(__AVX512F__) && defined(__AVX512BW__)
    if(getenv("I4_ACC512")) g_i4_acc512=atoi(getenv("I4_ACC512"))!=0;
    if(getenv("I4_ACC512_TEST")){
        if(!i4_acc512_selftest()) return 1;
        puts("AVX512 i4 selftest: ok"); return 0;
    }
    if(getenv("I3_AVX512")) g_i3_avx512=atoi(getenv("I3_AVX512"))!=0;
    if(getenv("I3_AVX512_TEST")){
        if(!i3_avx512_selftest()) return 1;
        puts("AVX512 i3 selftest: ok"); return 0;
    }
#endif
    const char *snap=getenv("SNAP");
    if(!snap){ coli_print_launcher_help("GLM-5.2"); return 1; }
    g_nopack = getenv("NOPACK")?1:0;
    g_drop = getenv("DROP")?1:0;
    g_prefetch = getenv("PREFETCH")?atoi(getenv("PREFETCH")):0;
    g_mmap = getenv("COLI_MMAP")?atoi(getenv("COLI_MMAP")):0;
    if(g_mmap) fprintf(stderr,"[MMAP] expert = viste zero-copy nei file (page cache = cache)\n");
    numa_init();                                       /* COLI_NUMA=1: expert-slab interleave (#82) */
    g_topk = getenv("TOPK")?atoi(getenv("TOPK")):0;
    g_topp = getenv("TOPP")?atof(getenv("TOPP")):0;
    /* EXPERT_BUDGET e' sotto quarantena: la finestra operativa e' misurata VUOTA.
     * @bokiko su tre host (#303) e riprodotto qui su un 25 GB / WSL:
     *   - hellaswag 30% a budget=8 contro 90% a budget spento (25% = il caso);
     *   - a budget=4 il decode e' rumore ("The **1...: s2151:");
     *   - accettazione MTP 0%: quali expert sopravvivono al cap dipende dalla
     *     residenza in cache al momento del forward, quindi draft e verify NON
     *     calcolano la stessa funzione -- la stessa invariante che #294 ha appena
     *     stabilito, violata via stato di cache invece che via scelta del kernel;
     *   - 0.13 tok/s contro 0.30 di baseline, con 14.66 expert caricati per layer
     *     contro topk=8: il cap fa piu' I/O di quello che dice di risparmiare, e la
     *     riga "~N GB I/O saved" conta esperti scartati, non byte non letti.
     * Resta compilato e sviluppabile (EXPERT_BUDGET_EXPERIMENTAL=1) perche' l'idea
     * -- MoE-Spec, arXiv 2602.16052 -- non e' sbagliata: e' l'implementazione che
     * finora non ha un punto in cui sia insieme piu' veloce e corretta. Riaccenderlo
     * di default richiede una misura di qualita' accanto a quella di velocita'. */
    g_expert_budget = getenv("EXPERT_BUDGET")?atoi(getenv("EXPERT_BUDGET")):0;
    if(g_expert_budget>0 && !getenv("EXPERT_BUDGET_EXPERIMENTAL")){
        fprintf(stderr,"[EXPERT_BUDGET] ignored: measured empty operating window (issue #303).\n"
            "[EXPERT_BUDGET] every tested setting is either no faster or no longer coherent:\n"
            "[EXPERT_BUDGET]   budget=8 -> hellaswag 30%% (90%% with it off) | budget=4 -> decode is noise\n"
            "[EXPERT_BUDGET]   MTP acceptance 0%% (the cap breaks the draft/verify contract, #294)\n"
            "[EXPERT_BUDGET]   0.13 tok/s vs 0.30 baseline, loading 14.7 experts/layer vs topk=8\n"
            "[EXPERT_BUDGET] set EXPERT_BUDGET_EXPERIMENTAL=1 to run it anyway (expect garbage).\n");
        g_expert_budget=0;
    }
    g_cache_route = getenv("CACHE_ROUTE")?atoi(getenv("CACHE_ROUTE")):0;
    g_route_j = getenv("ROUTE_J")?atoi(getenv("ROUTE_J")):2;
    g_route_m = getenv("ROUTE_M")?atoi(getenv("ROUTE_M")):12;
    g_route_p = getenv("ROUTE_P")?atof(getenv("ROUTE_P")):0;
    g_route_alpha = getenv("ROUTE_ALPHA")?atof(getenv("ROUTE_ALPHA")):1.f;
    g_route_agree = getenv("ROUTE_AGREE")?atoi(getenv("ROUTE_AGREE")):0;
    if(g_route_j<0) g_route_j=0;
    if(g_route_m<1) g_route_m=1;
    if(g_route_m>4096) g_route_m=4096;
    if(g_route_alpha<=0.f) g_route_alpha=1.f;
    if(g_route_alpha>1.f) g_route_alpha=1.f;
    if(g_cache_route)
        fprintf(stderr,"[CACHE_ROUTE] on J=%d M=%d P=%.2f alpha=%.2f (pin∪LRU prefer; never default)\n",
                g_route_j,g_route_m,g_route_p,g_route_alpha);
    if(g_route_agree)
        fprintf(stderr,"[ROUTE_AGREE] telemetry on (overlap%% + mean KL vs true top-K)\n");
    /* Auto-enable agree telemetry when CACHE_ROUTE is on (cheap quality leading indicator). */
    if(g_cache_route && !getenv("ROUTE_AGREE")) g_route_agree=1;
    const char *policy=getenv("COLI_POLICY"); if(!policy) policy="quality";
    int experimental=!strcmp(policy,"experimental-fast");
    if(strcmp(policy,"quality")&&strcmp(policy,"balanced")&&!experimental){
        fprintf(stderr,"COLI_POLICY non valida: quality, balanced o experimental-fast\n"); return 2;
    }
    if(!experimental&&(g_topk>0||g_topp>0)){
        fprintf(stderr,"[policy] --topp/--topk drop low-weight experts (~1.6x fewer reads, small quality cost)\n");
    }
    g_mlock  = getenv("MLOCK")?atoi(getenv("MLOCK")):-1;   /* -1 auto (ON macOS), 0 off, 1 force / auto (ON macOS), 0 off, 1 force */
    g_spec = getenv("SPEC")?atoi(getenv("SPEC")):1;
    g_draft = getenv("DRAFT")?atoi(getenv("DRAFT")):-1;
    g_no_fused_pair = getenv("COLI_NO_FUSED_PAIR")?atoi(getenv("COLI_NO_FUSED_PAIR")):0;   /* -1 = auto: 3 se MTP, 0 senza */
    g_looka = getenv("LOOKA")?atoi(getenv("LOOKA")):0;    /* 1 = misura predicibilita' routing */
    g_pilot = getenv("PILOT")?atoi(getenv("PILOT")):0;    /* 1 = prefetch pilotato dal router */
    g_pilot_real = getenv("PILOT_REAL")?atoi(getenv("PILOT_REAL")):0; /* default OFF: load VERI cross-layer (value-preserving prefetch); PILOT_REAL=1 opta in */
    if(g_pilot_real) g_pilot=1;                           /* PILOT_REAL implica il pilota attivo */
    g_pilot_two = getenv("PILOT_TWO")?atoi(getenv("PILOT_TWO")):0; /* 1 = two-step: shared-expert-corrected router prediction (+2.3% recall, 3 extra matmuls) */
    if(g_pilot_two) g_pilot=1;                            /* PILOT_TWO implies PILOT active */
    /* Default K: hint-only PILOT keeps 8 (WILLNEED hints are free, no eviction).
     * Under PILOT_REAL the speculative loads are REAL and create LRU eviction
     * pressure, so at ~28% mispredict a large K thrashes the cache — default to 6
     * (best-measured this session) unless the user set PILOT_K explicitly. */
    g_pilot_k = getenv("PILOT_K")?atoi(getenv("PILOT_K")):(g_pilot_real?6:8);
    if(g_pilot_k<1) g_pilot_k=1;
    /* PILOT_WORKERS: blocking-path pilot threads (SPMC ring). Default 1 = today's
     * behaviour, byte-identical. >1 raises NVMe queue depth on the non-URING
     * PILOT_REAL path (the URING path already batches). Clamp [1,16]. */
    g_pilot_nw = getenv("PILOT_WORKERS")?atoi(getenv("PILOT_WORKERS")):1;
    if(g_pilot_nw<1) g_pilot_nw=1; if(g_pilot_nw>16) g_pilot_nw=16;
    g_pilot_evict_guard = getenv("PILOT_EVICT_GUARD")?atoi(getenv("PILOT_EVICT_GUARD")):1; /* 0 = old LRU eviction (A/B) */
    g_disk_split = getenv("DISK_SPLIT")?atoi(getenv("DISK_SPLIT")):0; /* 1 = split dei disk load nelle stats */
    g_pipe = getenv("PIPE")?atoi(getenv("PIPE")):
#ifdef _WIN32
        1                        /* default ON: overlap expert load ‖ matmul (byte-identical; reorders I/O). PIPE=0 opts out */
#else
        0
#endif
        ;
    if(pipe_workers_imply_pipe(getenv("PIPE"),getenv("PIPE_WORKERS"),g_pipe)){
        g_pipe=1;
        fprintf(stderr,"[PIPE] PIPE_WORKERS is set — enabling the async pipe (PIPE=1 implied; set PIPE=0 to override)\n");
    }
    g_pipe_nw = getenv("PIPE_WORKERS")?atoi(getenv("PIPE_WORKERS")):8; /* I/O worker threads */
    if(g_pipe_nw<1) g_pipe_nw=1;
    g_pipe_block = getenv("COLI_PIPE_BLOCK")?atoi(getenv("COLI_PIPE_BLOCK")):0; /* blocking pipe_wait (default: spin) */
    g_direct = getenv("DIRECT")?atoi(getenv("DIRECT")):0;
    g_slab_shrink = getenv("COLI_SLAB_SHRINK") ? atoi(getenv("COLI_SLAB_SHRINK")) : 1;
    if(!g_slab_shrink)
        fprintf(stderr,"[SLAB] shrink disabled (COLI_SLAB_SHRINK=0): reused slots keep the widest "
                       "width they have held, so the per-row cap is optimistic again. Diagnostic "
                       "for #885 -- not a supported configuration.\n");
    { const char *dh=getenv("COLI_DISKCLASS_WINDOW");        /* DISK-CLASS recency window, see its declaration */
      if(dh){ g_direct_heat_ticks=(uint32_t)strtoul(dh,NULL,10); g_direct_heat_explicit=1; } }
    g_uring = getenv("URING")?atoi(getenv("URING")):0;
    if(g_uring){
#ifdef __linux__
        if(g_mmap){ fprintf(stderr,"URING=1 is incompatible with COLI_MMAP=1\n"); return 2; }
        g_pipe=1;
        if(uring_batch_init(&g_ub_pipe) || (g_pilot_real&&uring_batch_init(&g_ub_pilot))){
            fprintf(stderr,"URING=1: io_uring_setup failed: %s\n",strerror(errno)); return 2;
        }
        unsigned uw=(unsigned)(g_pipe_nw>64?64:g_pipe_nw);
        if(coli_uring_set_workers(&g_ub_pipe.ring,uw) ||
           (g_pilot_real&&coli_uring_set_workers(&g_ub_pilot.ring,uw)))
            fprintf(stderr,"[URING] warning: cannot set io-wq workers=%u: %s\n",uw,strerror(errno));
        fprintf(stderr,"[URING] queued expert I/O active (depth=%d, workers=%u, %s%s)\n",URING_REQ_MAX,uw,
                g_direct?"DIRECT=1":"buffered",g_pilot_real?", batched PILOT_REAL":"");
        if(!g_direct) fprintf(stderr,"[URING] cold NVMe: DIRECT=1 avoids page-cache copy/readahead bottlenecks\n");
#else
        fprintf(stderr,"URING=1 is supported only on Linux\n"); return 2;
#endif
    }
    if(getenv("COLI_MIR_STRIPE")) g_mir_stripe=atoi(getenv("COLI_MIR_STRIPE"));
    g_mirror_dir = getenv("COLI_MODEL_MIRROR");            /* MULTI-SSD: replica model copies */
    if(!g_mirror_dir||!*g_mirror_dir) g_mirror_dir = getenv("SNAP_MIRROR");
    if(g_mirror_dir&&!*g_mirror_dir) g_mirror_dir = NULL;
    g_idot = getenv("IDOT")?atoi(getenv("IDOT")):1;        /* 0 = kernel f32 esatti (A/B) */
    g_spec_pin = getenv("SPEC_PIN")?atoi(getenv("SPEC_PIN")):1; /* #163: 0 = gate S-dipendenti storici / legacy S-dependent gates */
    corpus_load();                                       /* COLI_DRAFT_CORPUS: external draft source */
    rt_trace_open();                     /* same place as before, so the log order is identical */
    g_repin = getenv("REPIN")?atoi(getenv("REPIN")):0;     /* RFC: re-pin ogni n token emessi (0=off) / live re-pin every n emitted tokens (0=off) */
    g_absorb = getenv("ABSORB")?atoi(getenv("ABSORB")):-1; /* -1 auto: assorbita per S<=4 */
    g_metal_prefill = getenv("COLI_METAL_PREFILL")?atoi(getenv("COLI_METAL_PREFILL")):0; /* default 0: S>4 attention on CPU (bit-exact); =1 opt-in GPU prefill */
    g_dsa_force = getenv("DSA_FORCE")?atoi(getenv("DSA_FORCE")):0;
    /* matmul_qt documenta la soglia int4-IDOT come "configurabile con I4S" ma il getenv non
     * c'era: la variabile non aveva alcun effetto. I4S=<n> -> IDOT int4 solo per S>=n.
     * EN: matmul_qt documents the int4 IDOT threshold as "configurable via I4S", but the
     * getenv was missing, so the knob did nothing. I4S=<n> -> int4 IDOT only for S>=n. */
    if(getenv("I4S")) g_i4s=atoi(getenv("I4S"));
    if(getenv("XEXP")) g_xexp=atoi(getenv("XEXP"))!=0;
    g_temp = temp_from_env(getenv("COLI_TEMP"),getenv("TEMP")); /* -1 = auto (1.0 chat/testo, greedy altrove) */
    g_nuc  = getenv("NUCLEUS")?atof(getenv("NUCLEUS")):0.90f;  /* piu' stretto dell'ufficiale 0.95: la coda int4 e' rumore */
    if(getenv("SEED")) g_rng = (uint64_t)atoll(getenv("SEED"))*0x9E3779B97F4A7C15ULL+1;
    else { struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); g_rng ^= (uint64_t)ts.tv_nsec<<20 ^ (uint64_t)getpid(); }
    if(g_draft>63) g_draft=63;                             /* -1 = auto, risolto dopo model_init */
    /* cap itself is resolved below, once g_metal_enabled and the SSD probe (both
     * needed for the platform default) are known -- see coli_resolve_cap(). */
    int cap_given = argc>1;
    int cap_arg = cap_given?atoi(argv[1]):0;
    int cap_env = getenv("CAP")?atoi(getenv("CAP")):0;
    int ebits= argc>2?atoi(argv[2]):8;
    int dbits= argc>3?atoi(argv[3]):ebits;
#if !defined(_WIN32)
    if(getenv("EXPERT_WORKER")){
        int port=getenv("CLUSTER_WORKER_PORT")?atoi(getenv("CLUSTER_WORKER_PORT")):9100;
        if(port<1||port>65535){fprintf(stderr,"CLUSTER_WORKER_PORT must be 1..65535\n");return 2;}
        return cluster_worker_run(snap,port,ebits,dbits);
    }
#else
    if(getenv("EXPERT_WORKER")){
        fprintf(stderr,"[CLUSTER] expert workers are not supported on Windows yet\n"); return 2;
    }
#endif
    int kv_limit=(getenv("SERVE_BATCH")&&atoi(getenv("SERVE_BATCH")))?512:16;
    if(getenv("SERVE") && (kv_slot_count()<1 || kv_slot_count()>kv_limit)){
        fprintf(stderr,"KV_SLOTS must be between 1 and %d\n",kv_limit); return 2;
    }
#ifdef COLI_CUDA
    if(getenv("COLI_CUDA") && atoi(getenv("COLI_CUDA"))){
        const char *one=getenv("COLI_GPU"), *many=getenv("COLI_GPUS");
        if(one&&many){ fprintf(stderr,"use COLI_GPU or COLI_GPUS, not both\n"); return 2; }
        if(many) g_cuda_ndev=parse_cuda_devices(many,g_cuda_devices);
        else if(one) g_cuda_ndev=parse_cuda_devices(one,g_cuda_devices);
        else { g_cuda_ndev=1; g_cuda_devices[0]=0; }
        if(g_cuda_ndev<1){ fprintf(stderr,"invalid COLI_GPUS: use a list such as 0,1,2\n"); return 2; }
        g_cuda_enabled=coli_cuda_init(g_cuda_devices,g_cuda_ndev);
        if(!g_cuda_enabled){ fprintf(stderr,"[CUDA] requested backend is unavailable\n"); return 2; }
        /* fmt=6 decodes against quant.h's codebook; publish it to every device so
         * the backend never keeps a second copy that could drift (#452). An older
         * DLL without the symbol leaves this 0 and fmt=6 tensors stay CPU-side. */
        g_cuda_e8_ready=coli_cuda_e8_set_grid(e8_grid);
        /* fmt=8 decodes against quant.h's E4M3_LUT — same arrangement. */
        g_cuda_fp8_ready=coli_cuda_fp8_set_lut(E4M3_LUT);
    }
    g_cuda_dense=getenv("CUDA_DENSE")?atoi(getenv("CUDA_DENSE")):0;
#endif
#ifdef COLI_VULKAN
    if(getenv("COLI_VULKAN") && atoi(getenv("COLI_VULKAN"))){
        char spvbuf[512]; const char *spv = vk_resolve_spv(spvbuf, sizeof(spvbuf));
        g_vulkan = coli_vk_init(spv);
        if(!g_vulkan){ fprintf(stderr,"[VK] Vulkan backend unavailable (tried %s; need libvulkan + "
                               "the compiled shaders — point COLI_VK_SHADERS at the shader directory "
                               "or the qmatmul.spv file, or run `make VK=1` to build them)\n", spv); return 2; }
        /* 320 = sweep optimum on a 16 GB card (256-384 measured flat, 320 best median;
         * ~6 GB tier + ~8 GB dense leaves headroom for the long-context KV mirror). */
        g_vk_budget = getenv("COLI_VK_EXPERTS") ? atoi(getenv("COLI_VK_EXPERTS")) : 320;
        g_vk_dense = getenv("COLI_VK_DENSE") ? atoi(getenv("COLI_VK_DENSE")) : 0;
        g_vk_attn = getenv("COLI_VK_ATTN") ? atoi(getenv("COLI_VK_ATTN")) : 0;
        fprintf(stderr,"[VK] expert tier active: routed quantized experts on the GPU (budget %d)%s%s\n",
                g_vk_budget, g_vk_dense ? " + dense projections + shared expert" : "",
                g_vk_attn ? " + absorb attention core" : "");
        /* COLI_VK_DEV2=auto|<index>: bring up a SECOND GPU for tier experts only
         * (e.g. an RX 580 beside the primary card). The dev2 tier fills with the
         * next heat-ranked experts after dev0's budget stop, capped by
         * COLI_VK_EXPERTS2 and its own VRAM budget (COLI_VK_RESERVE2_GB). */
        { const char *d2 = getenv("COLI_VK_DEV2");
          if(d2 && *d2){
              int idx = strcmp(d2,"auto") ? atoi(d2) : -1;
              if(coli_vk_init_dev2(spv, idx))
                  g_vk_budget2 = getenv("COLI_VK_EXPERTS2") ? atoi(getenv("COLI_VK_EXPERTS2")) : 512;
          } }
    }
#endif
#ifdef COLI_CUDA
    g_cuda_pipe=getenv("COLI_CUDA_PIPE")?atoi(getenv("COLI_CUDA_PIPE")):0;
    g_cuda_router=getenv("COLI_CUDA_ROUTER")?atoi(getenv("COLI_CUDA_ROUTER")):0;
    g_cuda_resid=getenv("COLI_CUDA_RESID")?atoi(getenv("COLI_CUDA_RESID")):0;
    const char *cuda_expert=getenv("CUDA_EXPERT_GB");
    g_cuda_expert_auto=cuda_expert&&!strcmp(cuda_expert,"auto");
    g_cuda_expert_gb=cuda_expert&&!g_cuda_expert_auto?atof(cuda_expert):0;
    g_cuda_reserve_gb=getenv("CUDA_RESERVE_GB")?atof(getenv("CUDA_RESERVE_GB")):2.0;
#ifdef COLI_ANS
    g_cuda_raw_experts=getenv("CUDA_RAW_EXPERTS")?atoi(getenv("CUDA_RAW_EXPERTS")):-1;
    if(g_cuda_raw_experts>=0&&(!getenv("COLI_ANS_SIDECAR")||!*getenv("COLI_ANS_SIDECAR"))){
        fprintf(stderr,"COLI_ANS_SIDECAR is required when CUDA_RAW_EXPERTS is set\n");
        return 1;
    }
    if(g_cuda_raw_experts>=0&&g_repin){
        fprintf(stderr,"REPIN is incompatible with the fixed-order ANS sidecar\n");
        return 1;
    }
#endif
    if(!getenv("REPIN")&&g_cuda_expert_auto&&getenv("PIN_GB")&&
       !strcmp(getenv("PIN_GB"),"all")) g_repin=16;
#ifdef COLI_ANS
    if(g_cuda_raw_experts>=0) g_repin=0;
#endif
    /* CUDA_RELEASE_HOST default: ndev>1 was chosen when the host copy was the
     * multi-GPU re-upload path. On a SINGLE GPU asked to fill RAM as well
     * (PIN_GB=all, or a PIN_GB large enough that the two tiers compete), that
     * default keeps a full host copy of every VRAM-tier expert and the RAM tier
     * starves: measured on 1x H200 + 235 GB, 9,297 vs 14,951 resident experts and
     * 1.10 vs 4.18-5.37 tok/s -- a 3.8x decode difference from one default (#686).
     * The host copy is provably redundant there: releasing it still reloads from
     * disk if CUDA later fails. So: keep ndev>1 exactly as before, and add the
     * single-GPU + large-PIN_GB case. An explicit CUDA_RELEASE_HOST always wins. */
    if(getenv("CUDA_RELEASE_HOST")) g_cuda_release_host=atoi(getenv("CUDA_RELEASE_HOST"));
    else if(g_cuda_ndev>1)          g_cuda_release_host=1;          /* unchanged */
    else if(g_cuda_enabled && (g_cuda_expert_gb>0||g_cuda_expert_auto)){
        const char *pg=getenv("PIN_GB");
        /* "large PIN_GB" = all, or >= the VRAM tier itself. Under CUDA_EXPERT_GB=auto
         * the tier is sized to the whole card, so any explicit PIN_GB qualifies. */
        if(pg&&*pg){
            if(!strcmp(pg,"all")) g_cuda_release_host=1;
            else { double p=atof(pg);
                   if(p>0 && (g_cuda_expert_auto || p>=g_cuda_expert_gb)) g_cuda_release_host=1; }
        }
        if(g_cuda_release_host)
            fprintf(stderr,"[CUDA] single-GPU with a large PIN_GB: releasing host copies of the "
                           "VRAM tier so the RAM tier can use that memory (#686; "
                           "CUDA_RELEASE_HOST=0 keeps them)\n");
    }
    if((getenv("COLI_GPU")||getenv("COLI_GPUS"))&&!g_cuda_enabled){ fprintf(stderr,"COLI_GPU(S) requires COLI_CUDA=1\n"); return 2; }
    if(g_cuda_dense&&!g_cuda_enabled){ fprintf(stderr,"CUDA_DENSE requires COLI_CUDA=1\n"); return 2; }
    if((g_cuda_expert_gb>0||g_cuda_expert_auto) && !g_cuda_enabled){ fprintf(stderr,"CUDA_EXPERT_GB requires COLI_CUDA=1\n"); return 2; }
    if(g_cuda_enabled) fprintf(stderr,"[CUDA] mode: routed experts%s%s\n",
        g_cuda_dense?" + resident dense tensors":" only (resident dense on CPU)",
        g_cuda_release_host?"; VRAM experts without host backing":"");
#else
    if((getenv("COLI_CUDA") && atoi(getenv("COLI_CUDA"))) ||
       getenv("COLI_GPU") || getenv("COLI_GPUS") ||
       (getenv("CUDA_DENSE") && atoi(getenv("CUDA_DENSE"))) ||
        (getenv("CUDA_EXPERT_GB") &&
        (!strcmp(getenv("CUDA_EXPERT_GB"),"auto")||atof(getenv("CUDA_EXPERT_GB"))>0))){
        fprintf(stderr,"CUDA was requested, but this binary is CPU-only; rebuild with: make CUDA=1\n");
        return 2;
    }
#endif
#ifdef COLI_METAL
    if(getenv("COLI_METAL") && atoi(getenv("COLI_METAL"))){
        g_metal_enabled = coli_metal_init();
        if(!g_metal_enabled){ fprintf(stderr,"[METAL] backend requested but not available\n"); return 2; }
        fprintf(stderr,"[METAL] mode: batched routed experts on GPU (unified-memory zero-copy)\n");
        if(getenv("COLI_METAL_SPIN") && atoi(getenv("COLI_METAL_SPIN"))){ coli_metal_spin_start(); fprintf(stderr,"[METAL] keep-alive spinner ON\n"); }
        if(getenv("COLI_METAL_GEMM_MIN")) g_metal_gemm_min=atoi(getenv("COLI_METAL_GEMM_MIN"));
        { const char *e=getenv("COLI_METAL_MOE_EXACT"); g_moe_exact=(e&&e[0]&&e[0]!='0'); }
    }
#else
    if(getenv("COLI_METAL") && atoi(getenv("COLI_METAL"))){
        fprintf(stderr,"METAL was requested, but this binary has no Metal backend; rebuild with: make METAL=1\n");
        return 2;
    }
#endif
    /* #379 (S2/S3): probe the model volume once Metal's on/off state is known,
     * cache the result in the model dir, and let it set the cap/CAP_RAISE
     * defaults -- never an explicit --cap/CAP/CAP_RAISE, per coli_resolve_cap()
     * above. Silent when the probe ran but storage was slow (defaults unchanged);
     * one stderr line when the platform default actually engages, one when a
     * contaminated (page-cache-warm) measurement is vetoed and deferred. */
    double coli_ssd_gbs = -1;
#if defined(COLI_METAL) && defined(__APPLE__)
    if(g_metal_enabled){
        coli_ssd_gbs = coli_ssd_probe_cached(snap);
        double ssd_fast_gbs = getenv("COLI_SSD_FAST_GBS")?atof(getenv("COLI_SSD_FAST_GBS")):4.0;
        g_ssd_fast = (coli_ssd_gbs>=ssd_fast_gbs);
    }
#endif
    int cap_explicit=0;
    int cap = coli_resolve_cap(cap_given, cap_arg, cap_env, g_ssd_fast, &cap_explicit);
    if(g_ssd_fast && !cap_explicit)
        fprintf(stderr,"METAL: fast SSD (%.1f GB/s) — page cache favored, expert cache minimal (cap 1); override with --cap\n", coli_ssd_gbs);
    /* KV8=1: KV-cache latente in fp8 e4m3. CPU-only in questo PR: i percorsi CUDA/Metal
     * che leggono righe f32 si spengono da soli (guardie), e COLI_CUDA_PIPE va disattivato
     * (il pipe-prefill legge righe f32). I kernel nativi arrivano nei follow-up. */
    g_kv8 = getenv("KV8")?atoi(getenv("KV8")):0;
    if(g_kv8){
#ifdef COLI_CUDA
        if(g_cuda_pipe){
            fprintf(stderr,"[KV8] COLI_CUDA_PIPE reads f32 KV rows; pipe disabled under KV8\n");
            g_cuda_pipe=0;
        }
#endif
        coli_fp8_lut_init();
        { const char *gsv=getenv("KV8_GS"); g_kv8_gs = gsv?atoi(gsv):0; if(g_kv8_gs<0) g_kv8_gs=0; }
        fprintf(stderr,"[KV8] latent KV cache in fp8 e4m3 + %s scale (~3.9x less KV RAM)\n",
                g_kv8_gs?"per-group":"per-row");
        if(g_kv8_gs && (!getenv("KVSAVE")||atoi(getenv("KVSAVE")))){
            fprintf(stderr,"[KV8] KV8_GS has no .coli_kv format yet: KV persistence disabled for this run\n");
            setenv("KVSAVE","0",1);
        }
    }
    /* KV_TQ=3|4: tier TurboQuant/PolarQuant (mutuamente esclusivo con KV8). CPU-only,
     * come KV8: si spegne dove i percorsi leggono righe f32. */
    { int tqv = getenv("KV_TQ")?atoi(getenv("KV_TQ")):0;
      if(tqv){
        if(g_kv8){ fprintf(stderr,"[KV_TQ] KV8 and KV_TQ are mutually exclusive; KV_TQ wins (KV8 off)\n"); g_kv8=0; }
        /* KV_TQ=1 used to clamp UP to 2, silently handing "just turn it on" the
         * most aggressive, lowest-quality tier. 4 is the recommended one, so a
         * bare/underspecified value lands there instead; >6 still clamps down to
         * the header's grid range. */
        if(tqv<2){ if(tqv!=4) fprintf(stderr,"[KV_TQ] KV_TQ=%d is below the 2..6 grid; using the recommended 4-bit tier\n",tqv);
                   tqv=4; }
        if(tqv>6) tqv=6;
        g_tq=1; g_tq_bits=tqv;
        /* rotated int4 is a fixed 4-bit codec (best 4-bit for MLA); other bit widths and
         * KV_TQ_POLAR=1 use PolarQuant (variable bits, paper-faithful). */
        g_tq_codec = (getenv("KV_TQ_POLAR") || g_tq_bits!=4) ? 0 : 1;
#ifdef COLI_CUDA
        if(g_cuda_pipe){ fprintf(stderr,"[KV_TQ] COLI_CUDA_PIPE reads f32 KV rows; pipe disabled under KV_TQ\n"); g_cuda_pipe=0; }
#endif
        fprintf(stderr,"[KV_TQ] latent KV in %s (randomized-Hadamard rotation; radius = per-row scale)\n",
            g_tq_codec ? "rotated int4 + Lloyd codebook (4-bit)" : "PolarQuant recursive-polar");
      }
    }
    /* ebits/dbits are the compute (dequant) width the idot kernels run at, not
     * the stored weight format: a fmt=4 grouped-int4 container still computes at
     * 8-bit here. Label it as compute so the banner is not misread as a storage
     * claim (#1183). */
    printf("== GLM C engine (glm_moe_dsa), cache=%d experts/layer | compute experts@%d-bit dense@%d-bit | idot: " IDOT_KERNEL " ==\n", cap, ebits, dbits);
    g_mem_avail_boot = mem_available_gb();
#if !defined(_WIN32)
    if(getenv("CLUSTER_WORKERS") && *getenv("CLUSTER_WORKERS")){
        cluster_init();
        atexit(cluster_close_all);
    }
#endif
    Model m; double t0=now_s(); model_init(&m,snap,cap,ebits,dbits);
    /* KV_TQ requires power-of-two row widths: both codecs rotate through a
     * radix-2 FWHT, and coli_kvq_quant_row returns an inert radius 0 for any
     * other width. On a model whose kv_lora/qk_rope are not powers of two that
     * would quantize EVERY latent row to zero and generate confident garbage
     * with no diagnostic -- the exact silent-misread failure the .coli_kv tier
     * magic exists to prevent. Refuse instead. GLM-5.2 (512/64) is unaffected;
     * this only fires on a model shape the codec cannot represent. */
    if(g_tq){
        int kl=m.c.kv_lora, kr=m.c.qk_rope;
        if(kl<2||kr<2||(kl&(kl-1))||(kr&(kr-1))){
            fprintf(stderr,"[KV_TQ] this model's latent rows are kv_lora=%d qk_rope=%d, but the "
                "rotation needs power-of-two widths (>=2): every row would quantize to zero. "
                "Refusing to run quantized -- unset KV_TQ (or use KV8=1, which has no width "
                "constraint).\n", kl, kr);
            return 2;
        }
        /* The attention consumers stage dequantized/rotated rows in 512-wide
         * stack buffers (qtl/qtr, Lf/Rf); a wider power-of-two row would pass
         * the check above and overflow them. Same refuse-don't-corrupt rule. */
        if(kl>512||kr>512){
            fprintf(stderr,"[KV_TQ] this model's latent rows are kv_lora=%d qk_rope=%d, but the "
                "quantized-KV attention path stages rows in 512-wide buffers. Refusing to run "
                "quantized -- unset KV_TQ (or use KV8=1, which has no width constraint).\n", kl, kr);
            return 2;
        }
    }
    if(!g_direct_heat_explicit){                     /* COLI_DISKCLASS_WINDOW default, needs m.c (topk/n_layers) */
        /* CURRENT-STATE CALIBRATION: the "8" multiplier (recency window ~= the last 8
         * tokens' worth of routing) is a measured-config constant, not a derived truth.
         * Coordinates: measured 2026-07, macOS 26.5, M5 Max 128 GB, base caa49f7,
         * GLM-5.2 int4 (topk=8, n_layers=78). On that setup it splits the classes
         * cleanly (decode cold share ~71% of classified bytes, 1061.85/1491.49 GB);
         * other models, page-cache pressures, or disks may want a different window --
         * COLI_DISKCLASS_WINDOW overrides, and the DISK-CLASS line itself is the
         * tuning instrument. */
        uint32_t nl=(uint32_t)(m.c.n_layers>0?m.c.n_layers:1), k=(uint32_t)(m.c.topk>0?m.c.topk:1);
        g_direct_heat_ticks = k*nl*8u;               /* ~last 8 tokens' worth of routing, see the declaration */
        if(!g_direct_heat_ticks) g_direct_heat_ticks=1;
    }
    if(g_draft<0){
#ifdef COLI_CUDA
        /* MTP is disabled under CUDA by default: cold (streaming) experts still
         * run on the CPU, where the S==1 fused-pair kernel and the S>=2 IDOT
         * kernel diverge in FP accumulation order, collapsing draft acceptance
         * (#163). GPU-resident experts have no divergence, but the cold subset
         * always exists on a single 16 GB card. COLI_CUDA_MTP=1 opts in for
         * users who want to test speculation under CUDA — the #163 thread shows
         * acceptance can still reach 30-50% even with the cold-expert mismatch.
         * See #292 for the diagnostic sweep that identified this. */
        int cuda_mtp = getenv("COLI_CUDA_MTP") ? atoi(getenv("COLI_CUDA_MTP")) : 0;
        /* Auto depth = 1, not 3. A GLM-5.2 744B sweep (DRAFT=0/1/2/3, streaming and
         * fully-resident) showed single-token speculation is the only depth that pays:
         * acceptance ~85% at depth 1 vs ~44-62% at 2-3, and every extra draft token
         * both costs verify compute and (when streaming) faults experts that evict the
         * LRU working set. Depth 1 was the fastest MTP setting in every measured
         * configuration; 2-3 never beat it anywhere. DRAFT=n still forces any depth. */
        g_draft = (m.has_mtp && (!g_cuda_enabled || cuda_mtp)) ? 1 : 0;
#else
        g_draft = m.has_mtp ? 1 : 0;
#endif
    }
    if(getenv("DSA_TOPK")) m.c.index_topk=atoi(getenv("DSA_TOPK"));   /* override per test */
    /* Il path MUX (SERVE_BATCH=1, cioe' `coli serve`) forza g_draft=0 sotto —
     * la speculazione non e' ragged-safe nel batch multi-slot. Segnalarlo QUI,
     * altrimenti "MTP active (draft=8)" mentirebbe: il messaggio e' stampato
     * prima della scelta del path (run_serve_mux, sotto), e con DRAFT=8 diceva
     * "active" per poi disabilitarlo in silenzio (#358, LordMZTE). */
    /* Multi-slot mux only: at KV_SLOTS=1 there is no ragged batch and the mux
     * keeps the resolved draft setting live (#492 single-slot speculation). */
    int mux_slots = getenv("KV_SLOTS") ? atoi(getenv("KV_SLOTS")) : 1;
    int mux_will_disable_mtp = getenv("SERVE") && getenv("SERVE_BATCH") &&
                               atoi(getenv("SERVE_BATCH")) && mux_slots>1;
    int eff_draft = mux_will_disable_mtp ? 0 : g_draft;
    printf("loaded in %.2fs | resident dense: %.2f MB | layers=%d experts=%d | MTP %s (draft=%d)\n",
           now_s()-t0, m.resident_bytes/(1024.0*1024.0), m.c.n_layers, m.c.n_experts,
           m.has_mtp?(mux_will_disable_mtp?"DISABLED (multiplexed serve)":"ACTIVE"):"absent", eff_draft);
    /* anche su stderr: e' il canale che le UI (coli) mostrano all'utente */
    if(mux_will_disable_mtp && m.has_mtp)
        fprintf(stderr,"[MTP] disabled in multiplexed serve (SERVE_BATCH=1, KV_SLOTS>1): speculation is "
                       "not ragged-safe across KV slots. Single-slot serve (KV_SLOTS=1) keeps MTP.\n");
    else
        fprintf(stderr,"[MTP] %s (draft=%d)\n", m.has_mtp?"active: native speculative decoding":"absent", eff_draft);
#ifdef __linux__
    {   /* Only warn for a GENUINE 9p mount (WSL Windows drives, magic 0x01021997), where
         * fadvise is a no-op. The old check was `snap` starting with "/mnt/", which
         * false-positives on native-Linux ZFS/ext4/xfs/NFS mounts that also live under /mnt. */
        struct statfs sfb;
        if(statfs(snap,&sfb)==0 && (unsigned long)sfb.f_type==0x01021997UL)
            fprintf(stderr,"WARNING: the model is on %s (9p/Windows filesystem; fadvise is ineffective).\n"
                           "         Keep it on a native Linux fs (ext4/xfs/zfs) for memory efficiency and speed.\n", snap);
    }
#endif
    /* DUAL-SSD: register the mirror copy BEFORE any pin/autopin load, so the
     * OMP-parallel pin warmup already streams from both drives. */
    mirror_setup(&m);
    /* HOT-STORE: PIN=<statsfile> [PIN_GB=g] -> top expert per frequenza fissi in RAM.
     * Va PRIMA di cap_for_ram: i pinnati contano nel residente. */
    if(getenv("PIN")){
        const char *pin=getenv("PIN"); char pauto[2100];
        int pin_named=strcmp(pin,"auto")!=0;   /* typed by the user, vs auto-discovered */
        if(!strcmp(pin,"auto")){
            /* PIN=auto: la storia VIVA <SNAP>/.coli_usage (appesa a ogni turno) batte il
             * profilo congelato stats.txt — il pin di ogni riavvio riflette il carico reale
             * accumulato, non il prompt di bootstrap. Fallback stats.txt per una dir vergine;
             * nessuno dei due -> nessun pin (AUTOPIN piu' sotto resta escluso: PIN e' settato).
             * EN: prefer the live usage history over the frozen one-shot profile, so each
             * reload's pin placement follows the accumulated real workload. */
            snprintf(pauto,sizeof(pauto),"%s/.coli_usage",snap);
            FILE *pf=fopen(pauto,"rb"); long psz=0;
            if(pf){ fseek(pf,0,SEEK_END); psz=ftell(pf); fclose(pf); }
            if(psz<=0){ snprintf(pauto,sizeof(pauto),"%s/stats.txt",snap);
                pf=fopen(pauto,"rb"); psz=0;
                if(pf){ fseek(pf,0,SEEK_END); psz=ftell(pf); fclose(pf); } }
            if(psz>0){ pin=pauto; fprintf(stderr,"[PIN] auto: seeding from %s\n",pauto); }
            else { pin=NULL; fprintf(stderr,"[PIN] auto: no .coli_usage or stats.txt in %s yet (no pin this run)\n",snap); }
        }
        if(pin){
            const char *pin_gb=getenv("PIN_GB");
            pin_load(&m,pin,pin_gb&&!strcmp(pin_gb,"all")?-1.0:pin_gb?atof(pin_gb):10.0,pin_named);   /* PIN_GB=all (#80) */
        }
    }
#if defined(COLI_CUDA) && defined(COLI_ANS)
    if(getenv("COLI_ANS_PACK")&&atoi(getenv("COLI_ANS_PACK"))){
        fprintf(stderr,"[ANS] sidecar packing complete; exiting before inference\n");
        coli_cuda_shutdown();
        return 0;
    }
#endif
    if(getenv("COUPLE")&&*getenv("COUPLE")){    /* coupling-scored cross-layer prefetch (#176) */
        g_couple_k=getenv("COUPLE_K")?atoi(getenv("COUPLE_K")):8;
        if(g_couple_k<1)g_couple_k=1; if(g_couple_k>32)g_couple_k=32;
        g_couple_d=getenv("COUPLE_D")?atoi(getenv("COUPLE_D")):1;
        if(g_couple_d<1)g_couple_d=1; if(g_couple_d>2)g_couple_d=2;
        couple_load(&m, getenv("COUPLE"));
    }
    /* CACHE CHE IMPARA: l'uso degli expert si accumula in <SNAP>/.coli_usage tra le sessioni;
     * all'avvio i piu' usati vengono auto-pinnati in RAM (meta' del budget expert: il pin
     * conosce la TUA storia, la LRU si adatta alla sessione). AUTOPIN=0 disattiva. */
    { double ram_env = getenv("RAM_GB")?atof(getenv("RAM_GB")):0.0;
      int est_ctx = getenv("CTX")?atoi(getenv("CTX")):4096;   /* stesso default di run_serve */
      snprintf(g_usage_path,sizeof(g_usage_path),"%s/.coli_usage",snap);
      int64_t hist = usage_load(&m,g_usage_path);
      if(hist>0) fprintf(stderr,"[USAGE] expert history: %lld selections (%s)\n",(long long)hist,g_usage_path);
      int autopin = getenv("AUTOPIN")?atoi(getenv("AUTOPIN")):1;
      if(!getenv("PIN") && autopin && hist>=5000){
          /* quota pin proporzionale alla FIDUCIA nella storia: con pochi dati il pin
           * sbaglia expert e ruba slot alla LRU adattiva; a regime (>=200k selezioni,
           * qualche ora di chat) arriva a meta' del budget expert. */
          double conf = (double)hist/200000.0; if(conf>1) conf=1;
          double expert_available=expert_avail(&m,ram_env,ebits,est_ctx);
          double planned_pin=expert_available*0.5*conf;
          int preserved_cap=0;
          double lru_reserve=autopin_lru_reserve(
              expert_available,expert_cache_bytes_per_slot(&m,ebits),
              m.ecap,&preserved_cap);
          double pin_bytes=autopin_preserve_lru(
              planned_pin,expert_available,lru_reserve);
          /* Print every term, every time, capped or not.
           *
           * This line used to appear ONLY when the clamp bit (pin_bytes <
           * planned_pin), which makes "the reserve was zero" and "the reserve
           * did not bind" indistinguishable from the outside -- and those are
           * the two hypotheses anyone debugging placement is choosing between.
           * #885 took fourteen instrumented runs on a 128 GB host to establish
           * numbers this line already had in registers. @mohamedmastouri2000-boop
           * asked for exactly this, twice, before anyone acted on it.
           *
           * budget  = what expert_avail() left after resident + slack
           * plan    = 0.5 * budget * confidence(history), the autopin quota
           * reserve = cap * per-row width, held back for the adaptive LRU (#815)
           * max_pin = budget - reserve, the ceiling the plan is clamped to
           *
           * A reserve near the budget means the pin is being starved by a
           * reservation the LRU may never claim, since the LRU is demand-filled
           * and only allocates on a miss. That is visible here at a glance. */
          fprintf(stderr,"[PIN] auto: budget %.1f GB | plan %.1f GB (conf %.2f, %lld selections) | "
              "LRU reserve %.1f GB (cap %d/layer x %.1f MB/row-set) | max_pin %.1f GB -> pinning %.1f GB%s\n",
              expert_available/1e9, planned_pin/1e9, conf, (long long)hist,
              lru_reserve/1e9, preserved_cap, expert_cache_bytes_per_slot(&m,ebits)/1e6,
              (expert_available-lru_reserve)/1e9, pin_bytes/1e9,
              pin_bytes+1.0<planned_pin ? "  [CAPPED by the LRU reserve]" : "");
          double pin_gb=pin_bytes/1e9;
          if(pin_gb>=0.5) pin_load(&m, g_usage_path, pin_gb, 0);   /* auto-discovered: not trusted */
      }
      /* SEMPRE: senza clamp la LRU cresce fino a cap*76 layer = decine di GB -> OOM-kill.
       * RAM_GB assente o <=0 = budget automatico da MemAvailable. */
      cap_for_ram(&m, ram_env, ebits, est_ctx);
      g_prof = getenv("PROF")?atoi(getenv("PROF")):0;   /* PROF=1: opt-in performance profile */
      if(g_prof) prof_config(&m, ram_env, est_ctx); }
#ifdef COLI_VULKAN
    vk_dense_preload(&m);   /* dense claims VRAM first — the tier fill sizes to the remainder */
    vk_registry_fill(&m);   /* pinned VK expert tier: needs the usage history loaded above */
#endif
    const char *stats=getenv("STATS");   /* STATS=<file> -> istogramma uso expert a fine run */

    /* CAUSAL ABLATION: ABLATE_SCORE=<manifest> -> teacher-forced per-expert
     * ablation sweep with per-target-position final-logit read-out (ABLATE_OUT=
     * <file>). Precedes SCORE. Optional ROUTE_TRACE=<file> records the
     * post-ablation router trace. */
    if(getenv("ABLATE_SCORE")){ run_ablate_score(&m, getenv("ABLATE_SCORE")); if(stats) stats_dump(&m,stats); return 0; }

    /* modo scoring per benchmark: SCORE=<requests.txt> -> log-likelihood per riga */
    if(getenv("SCORE")){ run_score(&m, snap, getenv("SCORE")); if(stats) stats_dump(&m,stats); return 0; }

    /* modo serve persistente per la CLI 'coli': SERVE=1 */
    if(getenv("SERVE")){
        if(getenv("SERVE_BATCH") && atoi(getenv("SERVE_BATCH"))) run_serve_mux(&m,snap);
        else run_serve(&m,snap);
        if(stats) stats_dump(&m,stats); return 0;
    }

    /* modo testo reale: PROMPT="..." [NGEN=n] -> tokenizza, genera, detokenizza */
    const char *user_prompt = coli_user_prompt();   /* ignores cmd.exe's PROMPT template (#271) */
    if(user_prompt){
        int ngen=getenv("NGEN")?atoi(getenv("NGEN")):64;
        run_text(&m, snap, user_prompt, ngen);
        if(stats) stats_dump(&m,stats);
        return 0;
    }

    /* altrimenti: validazione contro l'oracolo (ref_glm.json) */
    const char *refpath=getenv("REF")?getenv("REF"):"ref_glm.json";
    char *b=cfg_slurp(refpath);
    if(!b){ fprintf(stderr,"%s: cannot read oracle file (missing, unreadable, short, or > %lld bytes)\n",refpath,(long long)CFG_MAX_BYTES); return 1; }
    char *ar=NULL; jval *ref=json_parse(b,&ar);
    int np=0,nfull=0; int *prompt=read_arr(ref,"prompt_ids",&np); int *full=read_arr(ref,"full_ids",&nfull);
    if(!prompt||!full||np<1||nfull<np){ fprintf(stderr,"ref file missing prompt_ids/full_ids or empty\n"); return 1; }
    int n_new=nfull-np;
    /* L'oracolo (ref_glm.json in repo) e' del modello TINY: contro il 744B da' 0/20
     * garantito su OGNI piattaforma (prompt-token tiny = spazzatura per il modello vero).
     * Non e' un bug del motore — vedi #76. */
    { int maxid=0; for(int i=0;i<nfull;i++) if(full[i]>maxid) maxid=full[i];
      if(m.c.vocab>1000 && maxid<1000 && !getenv("REF_FORCE")){
        fprintf(stderr,
          "ERROR: no PROMPT given, so this is oracle self-test mode — but ref_glm.json is the TINY\n"
          "       model's oracle (max token %d) and your model's vocab is %d. Nothing to validate here.\n"
          "         Engine self-test:  SNAP=./glm_tiny TF=1 ./glm 64 16 16      (expect ~30-32/32; FP near-ties are toolchain-dependent)\n"
          "         Real generation:   PROMPT=\"Hello\" NGEN=32 SNAP=<model> ./glm 64\n"
          "         or:                python coli chat --model <model>\n"
          "         REF_FORCE=1 to run the comparison anyway (meaningless).\n"
          "  --- IT ---\n"
          "  Nessun PROMPT: modo auto-validazione, ma ref_glm.json e' l'oracolo del modello TINY\n"
          "  (token max %d, il tuo vocab e' %d). Usa PROMPT=... per generare davvero (vedi sopra).\n",
          maxid, m.c.vocab, maxid, m.c.vocab);
        return 1;
      } }

    if(getenv("REPLAY")){
        run_replay(&m,full,nfull,np);
        if(stats) stats_dump(&m,stats);
        return 0;
    }

    if(getenv("TF")){
        int *tf=read_arr(ref,"tf_pred",&(int){0});
        int *pred=malloc(nfull*sizeof(int)); double tt=now_s();
        forward_all(&m, full, nfull, pred, tf); double tdt=now_s()-tt;
        int ok=0; for(int i=0;i<nfull;i++){
            if(pred[i]==tf[i]) ok++;
            else fprintf(stderr,"[ORACLE] mismatch pos=%d expected=%d got=%d\n",i,tf[i],pred[i]);
        }
        printf("PREFILL (teacher-forcing) C vs oracle: %d/%d positions | %.1f pos/s\n",
            ok,nfull,nfull/tdt);
        if(ok<nfull) fprintf(stderr,
            "[ORACLE] %d/%d mismatches — run: TF=1 DEBUG_LOGITS=1 for top-5 logit dump\n",
            nfull-ok,nfull);
        profile_print(&m,tdt);
#ifdef COLI_CUDA
        if(g_cuda_enabled) cuda_stats_print();
#endif
        return 0;
    }
    int *out=malloc((np+n_new)*sizeof(int));
    ProfBase pb; prof_base(&m,&pb);
    double t=now_s(); generate(&m,prompt,np,n_new,out); double dt=now_s()-t;
    int match=0;
    printf("\nReference (oracle): "); for(int i=np;i<nfull;i++) printf("%d ", full[i]);
    printf("\nGLM C engine      : "); for(int i=np;i<nfull;i++){ printf("%d ", out[i]); if(out[i]==full[i])match++; }
    printf("\nMatching tokens: %d/%d\n", match, n_new);
    double tot=m.hits+m.miss;
    printf("N-gram speculation (DRAFT=%d): %.2f tokens/forward (%llu forwards per %llu tokens)\n",
        g_draft, m.n_fw?(double)m.n_emit/m.n_fw:1.0, (unsigned long long)m.n_fw, (unsigned long long)m.n_emit);
    printf("Expert cache hit rate: %.1f%% (%llu pin + %llu lru / %llu miss) | RSS: %.2f GB | %.1f tok/s\n",
           tot?100.0*m.hits/tot:0.0, (unsigned long long)m.hit_pin, (unsigned long long)m.hit_ecache,
           (unsigned long long)m.miss, rss_gb(), n_new/dt);
    profile_print(&m,dt);
    if(g_prof) prof_report(&m,&pb,dt,n_new,stdout);
#ifdef COLI_CUDA
    if(m.gpu_expert_count) printf("CUDA expert tier: %d resident experts (%.2f GB) | %llu calls served from VRAM\n",
        m.gpu_expert_count,m.gpu_expert_bytes/1e9,(unsigned long long)m.gpu_expert_calls);
    if(g_cuda_enabled) cuda_stats_print();
#endif
    if(g_looka){
        const char *nm[4]={"previous token (=SPEC prefetch)","layer input, skip attention","next layer (PILOT, stale)","next layer (two-step, shared-expert)"};
        printf("LOOKAHEAD routing — recall of true experts in predicted top-8:\n");
        for(int i=0;i<4;i++) printf("  %-42s %5.1f%%  (%lld/%lld)\n", nm[i],
            la_tot[i]?100.0*la_hit[i]/la_tot[i]:0.0, (long long)la_hit[i], (long long)la_tot[i]);
    }
    if(stats) stats_dump(&m,stats);
    return 0;
}
#endif /* COLIBRI_NO_MAIN */

#ifdef COLI_SEGMENT_ADAPTER
/* ---------- engine-owned Segment adapter ------------------------------ */

typedef struct {
    Model model;
    KVState *base_kv;
    uint32_t layer_begin, layer_end, context_tokens;
    pthread_mutex_t run_lock;
} GlmSegmentEngine;

typedef struct {
    GlmSegmentEngine *engine;
    KVState kv;
    uint32_t context_tokens, position;
} GlmSegmentSession;

static void glm_segment_qt_destroy(QT *tensor) {
    if (!tensor) return;
    free(tensor->qf); free(tensor->q8); free(tensor->q4); free(tensor->s);
    memset(tensor, 0, sizeof(*tensor));
}

static void glm_segment_layer_destroy(Layer *layer) {
    if (!layer) return;
    free(layer->in_ln); free(layer->post_ln);
    free(layer->q_a_ln); free(layer->kv_a_ln);
    glm_segment_qt_destroy(&layer->q_a);
    glm_segment_qt_destroy(&layer->q_b);
    glm_segment_qt_destroy(&layer->kv_a);
    glm_segment_qt_destroy(&layer->kv_b);
    glm_segment_qt_destroy(&layer->o);
    if (layer->sparse) {
        free(layer->router); free(layer->router_bias);
        glm_segment_qt_destroy(&layer->sh_gate);
        glm_segment_qt_destroy(&layer->sh_up);
        glm_segment_qt_destroy(&layer->sh_down);
    } else {
        glm_segment_qt_destroy(&layer->gate_proj);
        glm_segment_qt_destroy(&layer->up_proj);
        glm_segment_qt_destroy(&layer->down_proj);
    }
}

static void glm_segment_eslot_destroy(ESlot *slot) {
    if (!slot) return;
    if (slot->slab || slot->fslab || slot->aslab) {
        if (slot->aslab) {
            /* Segment adapters never build pin arenas, but avoid freeing an
             * interior pointer if a future CPU tier attaches one. */
            slot->slab = NULL; slot->fslab = NULL;
        } else {
            compat_aligned_free(slot->slab);
            free(slot->fslab);
        }
    } else {
        glm_segment_qt_destroy(&slot->g);
        glm_segment_qt_destroy(&slot->u);
        glm_segment_qt_destroy(&slot->d);
    }
    memset(slot, 0, sizeof(*slot));
}

static void glm_segment_kv_destroy(KVState *state, int num_layers) {
    if (!state) return;
    int rows = num_layers + 1;
    if (state->Lc || state->Rc)
        for (int layer = 0; layer < rows; layer++) {
            free(state->Lc ? state->Lc[layer] : NULL);
            free(state->Rc ? state->Rc[layer] : NULL);
        }
    if (state->Lc8 || state->Rc8 || state->Lsc || state->Rsc)
        for (int layer = 0; layer < rows; layer++) {
            free(state->Lc8 ? state->Lc8[layer] : NULL);
            free(state->Rc8 ? state->Rc8[layer] : NULL);
            free(state->Lsc ? state->Lsc[layer] : NULL);
            free(state->Rsc ? state->Rsc[layer] : NULL);
        }
    if (state->Ic)
        for (int layer = 0; layer < num_layers; layer++) free(state->Ic[layer]);
    free(state->Lc); free(state->Rc); free(state->Ic);
    free(state->Lc8); free(state->Rc8);
    free(state->Lsc); free(state->Rsc); free(state->kv_start);
    if (state->disk_fp) fclose(state->disk_fp);
    free(state->disk_buf);
    memset(state, 0, sizeof(*state));
}

static void glm_segment_model_destroy(GlmSegmentEngine *engine) {
    if (!engine) return;
    Model *model = &engine->model;
    int rows = model->c.n_layers + 1;
    for (uint32_t layer = engine->layer_begin; layer < engine->layer_end;
         layer++) {
        glm_segment_layer_destroy(&model->L[layer]);
        if (model->ecache && model->ecache[layer])
            for (int slot = 0; slot < model->ecap; slot++)
                glm_segment_eslot_destroy(&model->ecache[layer][slot]);
        if (model->pin && model->pin[layer])
            for (int slot = 0; slot < model->npin[layer]; slot++)
                glm_segment_eslot_destroy(&model->pin[layer][slot]);
        free(model->ecache ? model->ecache[layer] : NULL);
        free(model->pin ? model->pin[layer] : NULL);
        free(model->eroute ? model->eroute[layer] : NULL);
        free(model->eheat ? model->eheat[layer] : NULL);
        free(model->elast ? model->elast[layer] : NULL);
        free(model->elast_dc ? model->elast_dc[layer] : NULL);
        free(model->elast_pre ? model->elast_pre[layer] : NULL);
    }
    for (size_t slot = 0; slot < sizeof(model->ws) / sizeof(model->ws[0]);
         slot++) glm_segment_eslot_destroy(&model->ws[slot]);
    for (int layer = 0; layer < rows; layer++) {
        free(model->ecache_slot_by_expert
                 ? model->ecache_slot_by_expert[layer] : NULL);
        free(model->pin_slot_by_expert
                 ? model->pin_slot_by_expert[layer] : NULL);
    }
    if (model->has_dsa) {
        for (uint32_t layer = engine->layer_begin; layer < engine->layer_end;
             layer++) if (model->c.idx_type[layer]) {
            glm_segment_qt_destroy(&model->ix_wq[layer]);
            glm_segment_qt_destroy(&model->ix_wk[layer]);
            glm_segment_qt_destroy(&model->ix_wp[layer]);
            free(model->ix_knw[layer]); free(model->ix_knb[layer]);
        }
    }
    glm_segment_qt_destroy(&model->embed);
    glm_segment_qt_destroy(&model->lm_head);
    free(model->final_norm); free(model->hlast); free(model->h_all);
    free(model->ix_wq); free(model->ix_wk); free(model->ix_wp);
    free(model->ix_knw); free(model->ix_knb);
    free(model->dsa_sel); free(model->dsa_nsel);
    glm_segment_kv_destroy(engine->base_kv, model->c.n_layers);
    free(engine->base_kv);
    free(model->ecache); free(model->ecn);
    free(model->ecache_slot_by_expert);
    free(model->pin); free(model->npin); free(model->pin_slot_by_expert);
    free(model->eheat); free(model->elast);
    free(model->elast_dc); free(model->elast_pre);
    free(model->eroute); free(model->enr);
    free(model->eusage);
    free(model->kv_dev_L); free(model->kv_dev_R); free(model->kv_dev_valid);
    free(model->ln_dev);
#ifdef COLI_VULKAN
    free(model->vk_kv_valid);
#endif
    free(model->L);
    st_destroy(&model->S);
    model->kv = NULL;
}

static int glm_segment_engine_open(
    void **engine_impl, ColiSegmentCapabilities *capabilities,
    const ColiSegmentEngineOptions *options, char *error, size_t error_size) {
    if (!engine_impl || !capabilities || !options)
        return coli_segment_adapter_error(error, error_size,
                                           "invalid GLM Segment open");
    *engine_impl = NULL;
    if (options->backend_mask &&
        (options->backend_mask & ~COLI_SEGMENT_CAP_CPU))
        return coli_segment_adapter_error(error, error_size,
                                           "GLM Segment currently supports CPU");
    Cfg config;
    memset(&config, 0, sizeof(config));
    load_cfg(&config, options->model_dir);
    if (options->layer_end > (uint32_t)config.n_layers)
        return coli_segment_adapter_error(error, error_size,
                                           "GLM Segment range exceeds model");
    int ebits = getenv("GLM_SEGMENT_EBITS")
        ? atoi(getenv("GLM_SEGMENT_EBITS")) : 8;
    int dbits = getenv("GLM_SEGMENT_DBITS")
        ? atoi(getenv("GLM_SEGMENT_DBITS")) : ebits;
    if (ebits < 2 || ebits > 16 || dbits < 2 || dbits > 16)
        return coli_segment_adapter_error(error, error_size,
                                           "GLM Segment bit widths must be 2..16");
    int sparse_layers = 0;
    for (uint32_t layer = options->layer_begin; layer < options->layer_end;
         layer++) sparse_layers += layer >= (uint32_t)config.first_dense;
    int cap = 8;
    if (options->memory_limit_bytes && sparse_layers) {
        uint64_t values = 3u * (uint64_t)config.moe_inter * config.hidden;
        uint64_t weight_bytes = ebits >= 16 ? values * sizeof(float)
            : ebits >= 5 ? values : ebits >= 4 ? (values + 1u) / 2u
            : ebits == 3 ? (values * 3u + 7u) / 8u
            : (values + 3u) / 4u;
        uint64_t scale_bytes =
            (uint64_t)(2 * config.moe_inter + config.hidden) * sizeof(float);
        uint64_t slot_bytes = weight_bytes + scale_bytes;
        uint64_t slots = slot_bytes
            ? options->memory_limit_bytes / slot_bytes /
              (uint64_t)sparse_layers : 0;
        cap = slots > (uint64_t)config.n_experts
            ? config.n_experts : (int)slots;
        if (cap < 1) cap = 1;
    }

    GlmSegmentEngine *engine = calloc(1, sizeof(*engine));
    if (!engine)
        return coli_segment_adapter_error(error, error_size,
                                           "out of memory opening GLM Segment");
    engine->layer_begin = options->layer_begin;
    engine->layer_end = options->layer_end;
    engine->context_tokens = options->context_tokens;
    if (pthread_mutex_init(&engine->run_lock, NULL)) {
        free(engine);
        return coli_segment_adapter_error(error, error_size,
                                           "cannot initialize GLM Segment lock");
    }
    model_init_range(&engine->model, options->model_dir, cap, ebits, dbits,
                     (int)options->layer_begin, (int)options->layer_end,
                     0, 0, 0);
    engine->base_kv = engine->model.kv;

    memset(capabilities, 0, sizeof(*capabilities));
    capabilities->struct_size = sizeof(*capabilities);
    capabilities->abi_version = COLI_SEGMENT_ABI_VERSION;
    capabilities->flags = COLI_SEGMENT_CAP_SNAPSHOT |
                          COLI_SEGMENT_CAP_RANGE_NATIVE |
                          COLI_SEGMENT_CAP_MULTI_SESSION |
                          COLI_SEGMENT_CAP_CPU;
    coli_segment_capability_string(capabilities->engine_id,
                                   sizeof(capabilities->engine_id), "glm");
    coli_segment_capability_string(capabilities->state_schema,
                                   sizeof(capabilities->state_schema),
                                   "glm/mla-rope-dsa-f32-v1");
    snprintf(capabilities->numeric_class,
             sizeof(capabilities->numeric_class),
             "glm/e%d-d%d-dsa%d-kvf32/cpu-v1",
             ebits, dbits, engine->model.has_dsa != 0);
    capabilities->state_dtype = COLI_SEGMENT_DTYPE_F32;
    capabilities->state_width = (uint32_t)config.hidden;
    capabilities->max_batch_rows = 512;
    capabilities->max_context_tokens = options->context_tokens;
    capabilities->num_layers = (uint32_t)config.n_layers;
    *engine_impl = engine;
    return 0;
}

static void glm_segment_engine_destroy(void *engine_impl) {
    GlmSegmentEngine *engine = (GlmSegmentEngine *)engine_impl;
    if (!engine) return;
    glm_segment_model_destroy(engine);
    pthread_mutex_destroy(&engine->run_lock);
    free(engine);
}

static void glm_segment_session_free(GlmSegmentSession *session) {
    if (!session) return;
    if (session->engine)
        glm_segment_kv_destroy(&session->kv,
                               session->engine->model.c.n_layers);
    free(session);
}

static int glm_segment_session_create(
    void *engine_impl, void **session_impl,
    const ColiSegmentSessionOptions *options, char *error, size_t error_size) {
    GlmSegmentEngine *engine = (GlmSegmentEngine *)engine_impl;
    if (!engine || !session_impl || !options)
        return coli_segment_adapter_error(error, error_size,
                                           "invalid GLM Segment session");
    *session_impl = NULL;
    GlmSegmentSession *session = calloc(1, sizeof(*session));
    if (!session)
        return coli_segment_adapter_error(error, error_size,
                                           "out of memory creating GLM session");
    session->engine = engine;
    session->context_tokens = options->context_tokens;
    int rows = engine->model.c.n_layers + 1;
    session->kv.Lc = calloc((size_t)rows, sizeof(*session->kv.Lc));
    session->kv.Rc = calloc((size_t)rows, sizeof(*session->kv.Rc));
    session->kv.kv_start = calloc((size_t)rows,
                                  sizeof(*session->kv.kv_start));
    if (engine->model.has_dsa)
        session->kv.Ic = calloc((size_t)engine->model.c.n_layers,
                                sizeof(*session->kv.Ic));
    if (!session->kv.Lc || !session->kv.Rc || !session->kv.kv_start ||
        (engine->model.has_dsa && !session->kv.Ic)) goto oom;
    session->kv.max_t = (int)options->context_tokens;
    uint64_t state_bytes = 0;
    Cfg *config = &engine->model.c;
    for (uint32_t layer = engine->layer_begin; layer < engine->layer_end;
         layer++) {
        size_t cells;
        if (coli_segment_size_mul(options->context_tokens,
                                  (size_t)config->kv_lora, &cells)) goto oom;
        session->kv.Lc[layer] = calloc(cells, sizeof(float));
        state_bytes += (uint64_t)cells * sizeof(float);
        if (coli_segment_size_mul(options->context_tokens,
                                  (size_t)config->qk_rope, &cells)) goto oom;
        session->kv.Rc[layer] = calloc(cells, sizeof(float));
        state_bytes += (uint64_t)cells * sizeof(float);
        if (engine->model.has_dsa && config->idx_type[layer]) {
            if (coli_segment_size_mul(options->context_tokens,
                                      (size_t)config->index_hd,
                                      &cells)) goto oom;
            session->kv.Ic[layer] = calloc(cells, sizeof(float));
            state_bytes += (uint64_t)cells * sizeof(float);
        }
        if (!session->kv.Lc[layer] || !session->kv.Rc[layer] ||
            (engine->model.has_dsa && config->idx_type[layer] &&
             !session->kv.Ic[layer])) goto oom;
    }
    if (options->memory_limit_bytes &&
        state_bytes > options->memory_limit_bytes) {
        glm_segment_session_free(session);
        return coli_segment_adapter_error(error, error_size,
                                           "GLM session state exceeds memory limit");
    }
    *session_impl = session;
    return 0;

oom:
    glm_segment_session_free(session);
    return coli_segment_adapter_error(error, error_size,
                                       "out of memory allocating GLM state");
}

static void glm_segment_session_destroy(void *session_impl) {
    glm_segment_session_free((GlmSegmentSession *)session_impl);
}

static int glm_segment_session_run(void *session_impl,
                                   const ColiSegmentRunRequest *request,
                                   char *error, size_t error_size) {
    GlmSegmentSession *session = (GlmSegmentSession *)session_impl;
    if (!session || !request || request->position != session->position)
        return coli_segment_adapter_error(
            error, error_size, "GLM Segment requires contiguous positions");
    if (request->should_cancel &&
        request->should_cancel(request->cancel_user_data))
        return coli_segment_adapter_error(error, error_size,
                                           "GLM Segment run cancelled");
    GlmSegmentEngine *engine = session->engine;
    if (request->output != request->input)
        memcpy(request->output, request->input, request->input_bytes);
    pthread_mutex_lock(&engine->run_lock);
    kv_bind(&engine->model, &session->kv);
    layers_forward_range(&engine->model, (float *)request->output,
                         (int)request->rows, (int)request->position,
                         (int)engine->layer_begin, (int)engine->layer_end);
    kv_bind(&engine->model, engine->base_kv);
    pthread_mutex_unlock(&engine->run_lock);
    session->position += request->rows;
    return 0;
}

static int glm_segment_spans(
    GlmSegmentSession *session, uint32_t position,
    ColiSegmentStateSpan **spans_output, size_t *count_output,
    char *error, size_t error_size) {
    size_t capacity = (size_t)(session->engine->layer_end -
                               session->engine->layer_begin) * 3u;
    ColiSegmentStateSpan *spans = capacity
        ? calloc(capacity, sizeof(*spans)) : NULL;
    if (capacity && !spans)
        return coli_segment_adapter_error(error, error_size,
                                           "out of memory describing GLM state");
    Cfg *config = &session->engine->model.c;
    size_t count = 0;
    for (uint32_t layer = session->engine->layer_begin;
         layer < session->engine->layer_end; layer++) {
        spans[count++] = (ColiSegmentStateSpan){
            session->kv.Lc[layer],
            (size_t)position * config->kv_lora * sizeof(float)};
        spans[count++] = (ColiSegmentStateSpan){
            session->kv.Rc[layer],
            (size_t)position * config->qk_rope * sizeof(float)};
        if (session->engine->model.has_dsa && config->idx_type[layer])
            spans[count++] = (ColiSegmentStateSpan){
                session->kv.Ic[layer],
                (size_t)position * config->index_hd * sizeof(float)};
    }
    *spans_output = spans; *count_output = count;
    return 0;
}

static int glm_segment_session_snapshot(
    void *session_impl, ColiSegmentWriteFn write_fn, void *write_user_data,
    char *error, size_t error_size) {
    GlmSegmentSession *session = (GlmSegmentSession *)session_impl;
    ColiSegmentStateSpan *spans = NULL;
    size_t count = 0, payload_bytes;
    if (!session || glm_segment_spans(session, session->position, &spans,
                                      &count, error, error_size))
        return -1;
    if (coli_segment_spans_size(spans, count, &payload_bytes)) {
        free(spans);
        return coli_segment_adapter_error(error, error_size,
                                           "GLM snapshot size overflow");
    }
    ColiSegmentSnapshotHeader header;
    coli_segment_snapshot_header_init(
        &header, "glm", session->engine->layer_begin,
        session->engine->layer_end, session->context_tokens, session->position,
        payload_bytes, coli_segment_spans_hash(spans, count));
    int result = coli_segment_stream_write(
        write_fn, write_user_data, &header, sizeof(header), error, error_size);
    if (!result)
        result = coli_segment_spans_write(spans, count, write_fn,
                                          write_user_data, error, error_size);
    free(spans);
    return result;
}

static int glm_segment_session_restore(
    void *session_impl, ColiSegmentReadFn read_fn, void *read_user_data,
    char *error, size_t error_size) {
    GlmSegmentSession *session = (GlmSegmentSession *)session_impl;
    ColiSegmentSnapshotHeader header;
    if (!session || coli_segment_stream_read(read_fn, read_user_data, &header,
                                             sizeof(header), error, error_size))
        return -1;
    ColiSegmentStateSpan *spans = NULL;
    size_t count = 0, payload_bytes;
    if (glm_segment_spans(session, header.position, &spans, &count,
                          error, error_size)) return -1;
    if (coli_segment_spans_size(spans, count, &payload_bytes) ||
        coli_segment_snapshot_header_valid(
            &header, "glm", session->engine->layer_begin,
            session->engine->layer_end, session->context_tokens,
            payload_bytes, error, error_size)) {
        free(spans); return -1;
    }
    int result = coli_segment_spans_restore(
        spans, count, header.payload_hash, read_fn, read_user_data,
        error, error_size);
    free(spans);
    if (!result) session->position = header.position;
    return result;
}

static const ColiSegmentAdapter glm_segment_adapter = {
    sizeof(ColiSegmentAdapter), COLI_SEGMENT_ABI_VERSION, "glm",
    glm_segment_engine_open, glm_segment_engine_destroy,
    glm_segment_session_create, glm_segment_session_destroy,
    glm_segment_session_run, glm_segment_session_snapshot,
    glm_segment_session_restore, {0}
};

int coli_glm_segment_adapter_register(void) {
    return coli_segment_adapter_register(&glm_segment_adapter);
}
#endif /* COLI_SEGMENT_ADAPTER */

#ifdef COLI_EDGE_ADAPTER
/* ---------- engine-owned model Edge adapter --------------------------- */

typedef struct {
    Model model;
    Tok tokenizer;
} GlmEdgeEngine;

static void glm_edge_qt_destroy(QT *tensor) {
    if (!tensor) return;
    free(tensor->qf); free(tensor->q8); free(tensor->q4); free(tensor->s);
    memset(tensor, 0, sizeof(*tensor));
}

static void glm_edge_engine_destroy(void *engine_impl) {
    GlmEdgeEngine *engine = (GlmEdgeEngine *)engine_impl;
    if (!engine) return;
    glm_edge_qt_destroy(&engine->model.embed);
    glm_edge_qt_destroy(&engine->model.lm_head);
    free(engine->model.final_norm);
    st_destroy(&engine->model.S);
    tok_free(&engine->tokenizer);
    free(engine);
}

static int glm_edge_has_dsa(Model *model) {
    Cfg *config = &model->c;
    int enabled = config->index_topk > 0 && config->index_nh > 0 &&
                  config->index_hd > 0 && config->index_hd <= 256;
    char name[320];
    for (int layer = 0; enabled && layer < config->n_layers; layer++) {
        if (!config->idx_type[layer]) continue;
        snprintf(name, sizeof(name),
                 "model.layers.%d.self_attn.indexer.wq_b.weight", layer);
        if (!st_has(&model->S, name)) enabled = 0;
    }
    if (getenv("DSA") && atoi(getenv("DSA")) == 0) enabled = 0;
    return enabled;
}

static int glm_edge_engine_open(
    void **engine_impl, ColiEdgeCapabilities *capabilities,
    const ColiEdgeEngineOptions *options, char *error, size_t error_size) {
    if (!engine_impl || !capabilities || !options)
        return coli_edge_adapter_error(error, error_size,
                                       "invalid GLM Edge open");
    *engine_impl = NULL;
    if (options->backend_mask &&
        (options->backend_mask & ~COLI_EDGE_CAP_CPU))
        return coli_edge_adapter_error(error, error_size,
                                       "GLM Edge supports CPU only");
    int ebits = getenv("GLM_SEGMENT_EBITS")
        ? atoi(getenv("GLM_SEGMENT_EBITS")) : 8;
    int dbits = getenv("GLM_SEGMENT_DBITS")
        ? atoi(getenv("GLM_SEGMENT_DBITS")) : ebits;
    if (ebits < 2 || ebits > 16 || dbits < 2 || dbits > 16)
        return coli_edge_adapter_error(error, error_size,
                                       "GLM Edge bit widths must be 2..16");
    GlmEdgeEngine *engine = calloc(1, sizeof(*engine));
    if (!engine)
        return coli_edge_adapter_error(error, error_size,
                                       "out of memory opening GLM Edge");
    Model *model = &engine->model;
    model->ebits = ebits; model->dbits = dbits;
    load_cfg(&model->c, options->model_dir);
    const char *model_dirs = getenv("COLI_MODEL_DIRS");
    st_init_multi(&model->S, options->model_dir,
                  model_dirs && *model_dirs ? model_dirs : NULL);
    int io_bits = dbits >= 8 ? 16 : dbits;
    model->embed = qt_load(model, "model.embed_tokens.weight",
                           model->c.vocab, model->c.hidden, io_bits);
    model->lm_head = qt_load(model, "lm_head.weight",
                             model->c.vocab, model->c.hidden, io_bits);
    model->final_norm = ld(model, "model.norm.weight");
    char tokenizer_path[4096];
    snprintf(tokenizer_path, sizeof(tokenizer_path), "%s/tokenizer.json",
             options->model_dir);
    tok_load(&engine->tokenizer, tokenizer_path);

    int has_dsa = glm_edge_has_dsa(model);
    uint64_t resident = (uint64_t)qt_bytes(&model->embed) +
                        (uint64_t)qt_bytes(&model->lm_head) +
                        (uint64_t)model->c.hidden * sizeof(float);
    if (options->memory_limit_bytes && resident > options->memory_limit_bytes) {
        glm_edge_engine_destroy(engine);
        return coli_edge_adapter_error(error, error_size,
                                       "GLM Edge exceeds memory limit");
    }
    memset(capabilities, 0, sizeof(*capabilities));
    capabilities->struct_size = sizeof(*capabilities);
    capabilities->abi_version = COLI_EDGE_ABI_VERSION;
    capabilities->flags = COLI_EDGE_CAP_TOKENIZE |
                          COLI_EDGE_CAP_DETOKENIZE |
                          COLI_EDGE_CAP_GREEDY | COLI_EDGE_CAP_LOGITS |
                          COLI_EDGE_CAP_CPU;
    coli_edge_capability_string(capabilities->engine_id,
                                sizeof(capabilities->engine_id), "glm");
    coli_edge_capability_string(capabilities->state_schema,
                                sizeof(capabilities->state_schema),
                                "glm/mla-rope-dsa-f32-v1");
    snprintf(capabilities->numeric_class,
             sizeof(capabilities->numeric_class),
             "glm/e%d-d%d-dsa%d-kvf32/cpu-v1", ebits, dbits, has_dsa);
    coli_edge_capability_string(capabilities->tokenizer_class,
                                sizeof(capabilities->tokenizer_class),
                                "glm/cl100k-byte-bpe-v1");
    capabilities->state_dtype = COLI_EDGE_DTYPE_F32;
    capabilities->state_width = (uint32_t)model->c.hidden;
    capabilities->vocab_size = (uint32_t)model->c.vocab;
    capabilities->max_batch_rows = 512;
    capabilities->max_context_tokens = UINT32_MAX;
    capabilities->num_layers = (uint32_t)model->c.n_layers;
    capabilities->bos_token_id = -1;
    capabilities->eos_token_id = -1;
    capabilities->resident_bytes = resident;
    *engine_impl = engine;
    return 0;
}

static int glm_edge_tokenize(
    void *engine_impl, const char *text, size_t text_bytes,
    int32_t *token_ids, size_t token_capacity, size_t *token_count,
    char *error, size_t error_size) {
    GlmEdgeEngine *engine = (GlmEdgeEngine *)engine_impl;
    return coli_edge_tok_tokenize(&engine->tokenizer, text, text_bytes,
                                  token_ids, token_capacity, token_count,
                                  error, error_size);
}

static int glm_edge_detokenize(
    void *engine_impl, const int32_t *token_ids, size_t token_count,
    char *text, size_t text_capacity, size_t *text_bytes,
    char *error, size_t error_size) {
    GlmEdgeEngine *engine = (GlmEdgeEngine *)engine_impl;
    return coli_edge_tok_detokenize(&engine->tokenizer, token_ids, token_count,
                                    text, text_capacity, text_bytes,
                                    error, error_size);
}

static int glm_edge_embed(void *engine_impl,
                          const ColiEdgeEmbedRequest *request,
                          char *error, size_t error_size) {
    GlmEdgeEngine *engine = (GlmEdgeEngine *)engine_impl;
    int hidden = engine->model.c.hidden;
    float *output = (float *)request->output;
    for (uint32_t row = 0; row < request->rows; row++) {
        int token = request->token_ids[row];
        if (token < 0 || token >= engine->model.c.vocab)
            return coli_edge_adapter_error(error, error_size,
                                           "GLM token ID is out of range");
        embed_row(&engine->model, token, output + (size_t)row * hidden);
    }
    return 0;
}

static int glm_edge_select(void *engine_impl,
                           const ColiEdgeSelectRequest *request,
                           char *error, size_t error_size) {
    GlmEdgeEngine *engine = (GlmEdgeEngine *)engine_impl;
    Cfg *config = &engine->model.c;
    float *normalized = falloc(config->hidden);
    float *logits = falloc(config->vocab);
    const float *input = (const float *)request->input;
    for (uint32_t row = 0; row < request->rows; row++) {
        if (request->should_cancel &&
            request->should_cancel(request->cancel_user_data)) {
            free(logits); free(normalized);
            return coli_edge_adapter_error(error, error_size,
                                           "GLM Edge selection cancelled");
        }
        rmsnorm(normalized, input + (size_t)row * config->hidden,
                engine->model.final_norm, config->hidden, config->eps);
        matmul_qt(logits, normalized, &engine->model.lm_head, 1);
        if (coli_edge_argmax(logits, (uint32_t)config->vocab,
                            &request->token_ids[row],
                            request->scores ? &request->scores[row] : NULL)) {
            free(logits); free(normalized);
            return coli_edge_adapter_error(error, error_size,
                                           "GLM Edge head failed");
        }
    }
    free(logits); free(normalized);
    return 0;
}

static int glm_edge_logits(void *engine_impl,
                           const ColiEdgeLogitsRequest *request,
                           char *error, size_t error_size) {
    GlmEdgeEngine *engine = (GlmEdgeEngine *)engine_impl;
    Cfg *config = &engine->model.c;
    float *normalized = falloc(config->hidden);
    const float *input = (const float *)request->input;
    for (uint32_t row = 0; row < request->rows; row++) {
        if (request->should_cancel &&
            request->should_cancel(request->cancel_user_data)) {
            free(normalized);
            return coli_edge_adapter_error(error, error_size,
                                           "GLM Edge logits cancelled");
        }
        rmsnorm(normalized, input + (size_t)row * config->hidden,
                engine->model.final_norm, config->hidden, config->eps);
        matmul_qt(request->logits + (size_t)row * config->vocab,
                  normalized, &engine->model.lm_head, 1);
    }
    free(normalized);
    return 0;
}

static const ColiEdgeAdapter glm_edge_adapter = {
    sizeof(ColiEdgeAdapter), COLI_EDGE_ABI_VERSION, "glm",
    glm_edge_engine_open, glm_edge_engine_destroy,
    glm_edge_tokenize, glm_edge_detokenize,
    glm_edge_embed, glm_edge_select, glm_edge_logits, {0}
};

int coli_glm_edge_adapter_register(void) {
    return coli_edge_adapter_register(&glm_edge_adapter);
}
#endif /* COLI_EDGE_ADAPTER */
