/* coli_sycl.h — what every SYCL translation unit of the backend shares.
 *
 * The SYCL backend is the CUDA backend's migrated fork (backend_cuda.dp.cpp, cuda_chain.dp.cpp,
 * cu_xb.dp.cpp, by dpct over backend_cuda.cu, cuda_chain.cu and cu_xb.cu: tools/dpct-migrate.sh)
 * compiled by icpx for Intel GPUs. It exports the same C ABI the engine links against with
 * -DCOLI_CUDA (backend_cuda.h's coli_cuda_*, cuda_chain.h's ColiCudaChainOps, cu_tier_ops.h's
 * CuTierOps or NULL); the engine never includes this header. The units include it for the four
 * things the migration cannot give them:
 *
 *   1. the device profile: which Intel GPU this is, read from the device at run time (never from a
 *      compile-time target), and the decisions the profile makes: the sub-group the warp-32-style
 *      kernels run at, whether the matrix engines (XMX) are used, their shapes;
 *   2. per-device state: an in-order queue per device (the backend's "stream"), the context, and a
 *      persistent scratch arena in place of cudaMallocAsync/cudaFreeAsync at the decode sites
 *      (gemv_i8_w4's split-K partials, the dnrec split's q/k rows);
 *   3. the sub-group dispatch: a kernel body written for SG lanes is instantiated at 16 and 32
 *      with the required sub-group size, and the launch picks one from the profile;
 *   4. the XMX GEMM family the prompt path calls in place of the NVIDIA tensor-core kernels
 *      (hcg_*, tcg_*, f32t_*, compiled out of the migration's input by COLI_GPU_PORTABLE_ONLY,
 *      backend_gpu_compat.h): a dense GEMM over the tier's int8 / int4-g64 rows and the routed
 *      experts of a prompt chunk, each behind a guard that says whether the device takes it.
 *
 * Everything here is declared extern and implemented by the units named at each declaration;
 * coli_sycl_stubs.cpp holds weak definitions that return "unsupported" so a unit links alone
 * while the real ones are written. C++17, SYCL 2020 (oneAPI 2025.3). Nothing in this header
 * touches a device at include time.
 *
 * Targets, by detection (the engine's COLI_GPUS ordinals are positions in the GPU list this
 * header enumerates, Level Zero devices in PCI order):
 *   Data Center GPU Max 1100 / 1550 (Ponte Vecchio, intel_gpu_pvc): sub-groups 16 and 32, 448 EUs
 *     (one tile of the 1100), 128 KiB SLM a work-group, 48 GB HBM, XMX;
 *   Arc Pro B60 / B70, Arc B580 / B570 (Battlemage, intel_gpu_bmg_g21 / g31, Xe2): 16 and 32 with
 *     16 native, 24 / 32 GB, XMX;
 *   Lunar Lake (intel_gpu_lnl_m), Panther Lake (intel_gpu_ptl_*), Arrow Lake (arl_h), Alchemist
 *     (acm_g10..12, "dg2"), and architectures this header does not know: the generic profile from
 *     the queried properties.
 * Decisions: the warp-32-style kernels exist at sub-group 16 and 32; the profile takes 32 on PVC
 * and 16 on Xe2 and everything else; COLI_SYCL_SG=16|32 overrides it, COLI_SYCL_PROFILE=<name>
 * forces a named profile (pvc, bmg, generic, ...; coli_sycl_profile.h), COLI_SYCL_XMX=0|1 forces the matrix engines
 * off or on. XMX use is decided by the device's matrix_combinations query, never by its name. */
#ifndef COLI_SYCL_H
#define COLI_SYCL_H

#include <sycl/sycl.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include "coli_sycl_host_sync.h"

#include "cuda_chain.h"   /* CcExpert, CcMoe: the routed experts' table and a step's layout (the grouped GEMM) */

/* ---- 1. the device profile ------------------------------------------------------------ */

/* The profile's types, the derivation rules (sg_default, has_xmx, the name, the environment
 * overrides) and the names are coli_sycl_profile.h: plain C++, no SYCL, so a host test checks
 * the decisions without a device (tests/test_sycl_profile.cpp). This unit adds the reading. */
#include "coli_sycl_profile.h"

/* Reads the profile of a device: the architecture query first, then every property of
 * coli_sycl_props, then the decisions (sg_default, has_xmx) and the environment overrides
 * (coli_sycl_profile_derive). No kernel runs. Host only (coli_sycl_profile.cpp). An unknown
 * architecture (the runtime's architecture::unknown, or a value with no row on an Intel device)
 * yields the generic profile: sg_default 16 unless 32 is the only size offered, XMX from the
 * query alone, which lists no shape for an architecture the runtime cannot name, so off in
 * practice until a runtime that knows the part. */
extern coli_sycl_profile coli_sycl_profile_detect(const sycl::device &d);
/* The profile of an open device (coli_sycl_open); `device` is the engine's ordinal. A device that
 * is not open yields the generic profile of nothing (arch NONE, every count 0) rather than a
 * fault, so a caller may print it. */
extern const coli_sycl_profile &coli_sycl_profile_for(int device);
/* The architecture listed GPU `ordinal` would have once open, read without opening it (NONE past the
 * list): for decisions taken before coli_sycl_open (coli_cuda_tuned_default before an engine's init). */
extern coli_sycl_arch coli_sycl_listed_arch(int ordinal);
/* Every GPU's profile on stdout, in the engine's ordinal order (the list of section 2), with
 * the platform and driver each runs on: what COLI_SYCL_LIST=1 prints before the first entry
 * point of section 2 exits 0 (the first-contact check of sycl/README.md). C linkage, so an
 * engine or a test written in C may call it. */
extern "C" void coli_sycl_print_profiles(void);

/* ---- 2. per-device state -------------------------------------------------------------- */

/* The GPUs this process can use, in PCI order (ext::intel pci_address; Level Zero first, the
 * OpenCL view of the same cards is not listed twice): the ordinals the engine's COLI_GPUS /
 * COLI_GPU name, before any is opened. 0 when the runtime finds no GPU. Host only. */
extern int coli_sycl_available_device_count(void);
/* Opens the devices of `devices` (ordinals, as coli_cuda_init takes them): a context and an
 * in-order queue per device (the backend's stream), a second in-order queue for the chain's copies
 * (cuda_chain.cu's c->cs), the profile. Repeating the same list keeps everything; a different list
 * while devices are open returns 0, as coli_cuda_init documents. 1 on success. */
extern int coli_sycl_open(const int *devices, int count);
extern void coli_sycl_close(void);
extern int coli_sycl_is_open(int device);
/* the device, context and queues of an open device (a fault to ask for one that is not open:
 * check coli_sycl_is_open where the ordinal comes from outside) */
extern sycl::device &coli_sycl_device(int device);
extern sycl::context &coli_sycl_context(int device);
extern sycl::queue &coli_sycl_queue(int device);        /* the compute queue, in order */
extern sycl::queue &coli_sycl_copy_queue(int device);   /* the copy queue, in order */
/* A queue's device ordinal (the engine's), -1 if it is none of the open devices' queues: what the
 * migrated units use where the CUDA code read the current device of a stream. */
extern int coli_sycl_queue_device(const sycl::queue &q);

/* The scratch arena: a persistent, grow-only device allocation per device and slot, returned
 * for `bytes` or more. It replaces cudaMallocAsync / cudaFreeAsync on the decode path (the
 * stream-ordered allocator has no SYCL form, and a device allocation per GEMV costs tens of
 * microseconds through Level Zero): the kernels of one in-order queue run in order, so a kernel
 * writing partial sums into the arena and the kernel summing them cannot overlap a later user.
 * A grow (with headroom: half again the request) never waits on the queues, which inside a chain
 * frame could sit behind a cc_wait_host kernel whose host side comes after the frame: the old
 * block is retired and freed by coli_sycl_scratch_reap, below, once the chain has seen the queue
 * pass the frames, or at close. A grow after a slot's first allocation is logged (an allocation
 * in the token loop; the GEMV partials are sized for every row count their path takes at the
 * first use, so a verify after a decode never regrows). `slot` keeps two users that are in
 * flight in the same queue apart (0 the GEMV partials, 1 the dnrec split's rows, 2 spare, 3 the
 * XMX family's staging copy); NULL when the allocation fails or the device is not open (the
 * callers then fall back, as they do when cudaMallocAsync fails). */
extern void *coli_sycl_scratch(int device, size_t bytes, int slot = 0);
extern size_t coli_sycl_scratch_bytes(int device, int slot = 0);   /* what the slot holds now (0 none) */
/* A block out of use at the next frame's end: the arena's grow retires its old block this way,
 * bd_xq_scratch and the XMX unit's private copy theirs. Freed by coli_sycl_scratch_reap, which
 * the chain calls where it knows the queue passed the frames (cc_submit with a wait, cc_finish,
 * its shutdown) and coli_sycl_close calls after its waits. coli_sycl_retire takes a block used on
 * an open device's own queues (1: those the frame waits cover) and refuses any other queue's
 * (0: cu_xb's own, a test's; the caller waits on that queue and frees as before). Eight retired
 * blocks without a reap make the ninth grow wait, logged. */
extern void coli_sycl_scratch_reap(int device);
extern int coli_sycl_retire(const sycl::queue &q, void *p, size_t bytes);
/* 1 once the device's queues delivered an asynchronous error to their handler (a command the
 * runtime reports as failed and hands to the handler at a wait_and_throw, never throws to the
 * caller): the twin of a caught exception, read by the chain's live / cc_try and the backend's
 * fault_injected. It polls the queues (throw_asynchronous: a lock and a list check), so an error
 * collected since the last wait counts at the next call. */
extern int coli_sycl_lost(int device);
/* Host tasks capture this state before submit. Its methods never touch SYCL or
 * the global device table; shared ownership outlives queue/context teardown.
 * Marking a fatal chain error cancels the same state as the async handler. */
extern std::shared_ptr<coli_sycl_host_sync> coli_sycl_host_state(int device);
extern void coli_sycl_mark_lost(int device);
/* The device's memory, cudaMemGetInfo's answer: the total from the device; the free figure from
 * the device's free_memory query where it has one (ext_intel_free_memory: Level Zero with
 * ZES_ENABLE_SYSMAN=1 in the environment before the driver starts, which the GPU list sets), else
 * the total minus what this backend holds (the arena, and every unit's ledger below: the
 * backend's resident tensors, the chain's buffers, cu_xb's pools), since the engine budgets on
 * this figure (the tier's allowance, the chain's prompt chunk, the staging guard) and a "free =
 * total" on a card the tier filled would plan it twice. `queried` (may be NULL) says which.
 * 0 when the device is not open or the query failed. */
extern int coli_sycl_mem_info(int device, size_t *free_bytes, size_t *total_bytes, int *queried);
typedef size_t (*coli_sycl_ledger_fn)(int device);   /* a unit's device bytes on `device` */
extern void coli_sycl_ledger_add(coli_sycl_ledger_fn fn);   /* once per function; a repeat is ignored */

/* ---- 3. the sub-group dispatch -------------------------------------------------------- */

/* The warp-32-style kernels (a warp per output row, lane = id & 31, warp = id >> 5, width-32
 * shuffles, 8 warps a 256-thread block) are written once as a body over SG, the lanes of a
 * sub-group, and instantiated twice: at 16 and at 32, each with the required sub-group size
 * the compiler must honour (the shuffles are then sub-group operations of exactly SG lanes).
 * The profile picks which one runs. The body is a callable
 *
 *     [=](sycl::nd_item<3> it, coli_sycl_sg<SG>, uint8_t *slm) { ... }
 *
 * for example a generic lambda that reads SG as decltype(tag)::value and calls the migrated
 * kernel function instantiated at that SG. slm is the launch's dynamic local memory (the
 * dpct_local pointer of the migrated kernels), nullptr when slm_bytes is 0. A kernel at SG 16
 * has twice the warps a block: everything that counted 8 warps of 32 counts (block / SG). */
template <int SG> struct coli_sycl_sg { static constexpr int value = SG; };

template <int SG, typename Body>
struct coli_sycl_sg_kernel {
    Body body;
    sycl::local_accessor<uint8_t, 1> slm;
    [[sycl::reqd_sub_group_size(SG)]] void operator()(sycl::nd_item<3> it) const {
        uint8_t *p = slm.size() > 1 ? slm.template get_multi_ptr<sycl::access::decorated::no>().get() : nullptr;
        body(it, coli_sycl_sg<SG>{}, p);
    }
};
/* Launches `body` over grid x block work-items at sub-group size `sg` (16 or 32: the profile's
 * sg_default, or a kernel's own choice where the profile offers it), with slm_bytes of dynamic
 * local memory. A size the device does not offer is the caller's error: check the profile. */
template <typename Body>
inline sycl::event coli_sycl_launch_sg(sycl::queue &q, int sg, sycl::range<3> grid, sycl::range<3> block,
                                       size_t slm_bytes, Body body) {
    return q.submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint8_t, 1> slm(sycl::range<1>(slm_bytes ? slm_bytes : 1), cgh);
        sycl::nd_range<3> r(grid * block, block);
        if (sg == 32) cgh.parallel_for(r, coli_sycl_sg_kernel<32, Body>{body, slm});
        else cgh.parallel_for(r, coli_sycl_sg_kernel<16, Body>{body, slm});
    });
}
/* the same choice as a statement, for a migrated launch site that stays a parallel_for with a
 * literal attribute: COLI_SYCL_SG_SWITCH(p.sg_default, X) expands X(32) or X(16) */
#define COLI_SYCL_SG_SWITCH(sg, X) do { if ((sg) == 32) { X(32); } else { X(16); } } while (0)
/* the width the migrated shuffles take (dpct::experimental::shift_sub_group_left and the others
 * carry a logical sub-group size): the kernel's SG, never a literal 32 */
#define COLI_SYCL_SHFL_WIDTH(SG) (SG)

/* ---- 4. the XMX GEMM family (prompt rows) ------------------------------------------------ */

/* Whether the profile's matrix engines take operands of `dt` (from matrix_combinations and
 * COLI_SYCL_XMX): the guard in front of every call below, and of the chain's hcg_ok / tcg_ok
 * twins (coli_sycl_xmx_gemm_ok). The GEMMs below do not need it: on a device without the
 * f16 shape (or COLI_SYCL_XMX=0) they run the same staging through a plain tile of f32 FMAs, so
 * coli_sycl_xmx_gemm_ok answers for the shape and format alone and a caller that would rather
 * keep the migrated warp and block kernels on such a device asks this first. */
extern bool coli_sycl_xmx_supported(const coli_sycl_profile &p, coli_sycl_dtype dt);

/* (a) The dense GEMM of a prompt chunk: C[M][N] (f32, row stride ldc) = A[M][K] x W, where W is the
 * tier's resident matrix of N rows of K inputs (fmt 1: int8 rows, a scale a row; fmt 4: int4
 * nibbles, a scale per `gs` inputs, ng = ceil(K / gs) scales a row, the nibbles already signed by
 * offset_to_signed_s4 at upload; fmt 2: int4 rows, a scale a row), as backend_cuda's tensors
 * hold it (weights, scales). A is f32 (a_f16 0, stride lda floats) or f16 (a_f16 1, stride lda
 * halves) activations; the kernel stages A as f16 and the weights as f16 tiles (int8 and int4
 * are exact in f16, the scales applied after the product, hcg_dense_k's arithmetic) on the
 * f16 8x16x16 shape, or as s8 on the int8 8x16x32 shape where a_s8 asks for it and the device
 * has it (the activations quantized per row, moe's quantize_s4_rows style; not bit-identical).
 * K a multiple of 64 and 4-byte aligned rows (joint_matrix's load restriction: stride x size a
 * multiple of 8). Submits on q and returns 1; 0 means not taken (an unsupported shape, format,
 * alignment or device) and nothing was submitted: the caller launches its fallback. Never
 * blocks (the activations' copy is the arena's slot 3 on an open device's queue, grown with
 * headroom and the old block retired, as coli_sycl_scratch does; a foreign queue's private copy
 * waits on that queue). The int8 shape (a_s8) is taken for fmt 1 and 2
 * only (a row's scale on each side; fmt 4's scale per group needs the f16 staging). Implemented
 * by coli_sycl_xmx.cpp; COLI_SYCL_XMX_TM=16 takes the compound 16x16x16 shape where listed. */
struct coli_sycl_gemm {
    float *C; long ldc;
    const void *A; long lda; int a_f16; int a_s8;   /* a_s8 1: take the int8 shape where available */
    const void *W; const float *scales; int fmt; int gs;
    int M, K, N;
};
extern int coli_sycl_xmx_gemm(const coli_sycl_profile &p, sycl::queue &q, const coli_sycl_gemm &g);
/* the gate and up pair with the GLU epilogue, one pass over A (hcg2_dense_k's): H[M][N] =
 * silu(A x Wg) * (A x Wu); Wg, Wu of one format and shape */
extern int coli_sycl_xmx_gemm_glu(const coli_sycl_profile &p, sycl::queue &q, const coli_sycl_gemm &gate,
                                  const void *Wu, const float *scales_u);
/* whether (a) would take a tensor of this shape and format on this profile, without submitting:
 * the chain's hcg_ok / tcg_ok twin and coli_cuda_planar_rows's answer */
extern int coli_sycl_xmx_gemm_ok(const coli_sycl_profile &p, int fmt, int gs, int K, int N);

/* (b) The routed experts of a prompt chunk, the slots sorted by expert (backend_cuda's moe_sort_*
 * kernels: the integers after hid's floats). Expert e owns slots order[base[e] .. base[e + 1]);
 * slot a is row a / K of x (x_off + s * x_row, D floats) with the chain's weight w[a], and row a
 * of hid. Tile t of T is expert te[t]'s rows tr[t] .. tr[t] + TR of its slot list (ntile tiles
 * are live, written by the sort; TR is 64). Phase 0 (gate and up, silu(g) * u, CcMoe.act 0, the only act):
 * hid[a][0..I) = silu(x[s] . gate_e) * (x[s] . up_e); phase 1 (down): the rows after them,
 * hid[A + a][0..D) = hid[a] . down_e. The weights are the table's CcExpert entries from t_off
 * (g, u, d and their scales gs, us, ds; planar 1: an int4-g64 matrix in the host pool's planar
 * layout, which the f16 staging reads directly as hcg_moe_*_k does; host 1 entries are read
 * where they are, which works for USM host memory only: the backend keeps any other host
 * matrix out of the table it hands over). The rank-order sum into y stays moe_rows_sum_k's. Formats as (a).
 * 1 submitted, 0 not taken (nothing submitted: the caller takes moe_gu_k / moe_dn_k). */
struct coli_sycl_moe_sort {   /* backend_cuda's MoeSort, device pointers */
    const int *order, *base, *te, *tr, *ntile;
    int T, TR;                /* tiles the sort may produce, rows a tile (64) */
};
extern int coli_sycl_xmx_moe(const coli_sycl_profile &p, sycl::queue &q, const CcExpert *tab, const CcMoe &m,
                             const coli_sycl_moe_sort &sort, const float *x, float *hid, int phase);
extern int coli_sycl_xmx_moe_ok(const coli_sycl_profile &p, const CcMoe &m);

/* (c) A prompt chunk's causal attention on the matrix engines, in the place of cuda_chain.cu's
 * tensor-core kernels (cc_attn_fk, cc_attn_f2: WMMA and mma.sync, compiled out of the migration;
 * cuda_chain.dp.cpp's cc_attn_xmx asks here). The rows' queries at q (CcAttn's offsets: q_off +
 * s * q_row + h * q_seg, hd floats) against the cache kc / vc (head-major, K[(kvh * cap + t) * hd
 * + d] from k_off / v_off, floats or halves: kv_f16), into o (o_off + s * o_row + h * hd) with the
 * gate (has_gate: times the sigmoid of gate[g_off + s * g_row + h * g_seg + d]); key t valid for
 * row s iff t <= pos_base + s and, with a window of win positions, t > pos_base + s - win (0: no
 * window); a sink logit a head (snk[sink_off + h], NULL none) in the denominator only; a ring of
 * `ring` rows (key t in row t % ring, 0 none); a selection list a row (sel, CcAttn.sel_row: the
 * count, -1 none, then ascending positions; NULL none). G the query heads a kv head.
 * cc_attn_k's arithmetic on f16 operands (not bit-identical to it). Submits on q and returns 1;
 * 0 not taken, nothing submitted (the chain keeps cc_attn_k / cc_attn_gk). The _ok twin answers
 * for the shape without submitting: what the chain's decode split keys on (cc_attn_dec_on).
 * Implemented by coli_sycl_xmx.cpp. */
struct coli_sycl_attn {
    const float *q; const void *kc, *vc; int kv_f16;
    float *o; const float *gate;
    CcAttn p; int G;
    int win; const float *snk; int sink_off; int ring;
    const int *sel;
};
extern int coli_sycl_xmx_attn(const coli_sycl_profile &p, sycl::queue &q, const coli_sycl_attn &a);
extern int coli_sycl_xmx_attn_ok(const coli_sycl_profile &p, int hd, int G, int kv_f16, int sel);
/* The same chunk through oneDNN's fused attention (coli_sycl_dnnl.cpp): only in a SYCL_DNNL=1 build,
 * only with COLI_SYCL_DNNL_ATTN=1 (_on), a float16 cache, head sizes 64, 128 and 256, no selection
 * list, window, sinks or ring; returns 1 when submitted, 0 (nothing written) otherwise, as
 * coli_sycl_xmx_attn does. _release drops what it holds for a queue (the chain's shutdown). */
extern int coli_sycl_dnnl_attn_on(void);
extern int coli_sycl_dnnl_attn(const coli_sycl_profile &p, sycl::queue &q, const coli_sycl_attn &a);
extern void coli_sycl_dnnl_release(sycl::queue &q);

/* ---- 5. frame profiling (COLI_SYCL_FRAME_PROF) ------------------------------------------ */

/* Per-command device time on the compute queue from event profiling, off by default (the hooks
 * cost one flag test). COLI_SYCL_FRAME_PROF=1 makes the queues with enable_profiling (as
 * COLI_SYCL_PROFILING=1 does) and the launch sites that pass their event through
 * coli_sycl_fprof_ev (the chain's CC_K / CC_KSG and its copies, the backend's COLI_SG_LAUNCH and
 * its plain launches on the decode path) hand it here, unnamed; the chain names the commands it
 * just submitted (coli_sycl_fprof_name: the op's kind, a matmul's shape); once a frame has been
 * waited for, coli_sycl_fprof_frame reads every event's command_start / command_end and books the
 * frame: per kind the count and device time, per frame the span (first start to last end), the
 * busy time (the durations' sum), the idle between consecutive commands (the device's gaps on an
 * in-order queue) and the host's submit / wait / total times. Frames are bucketed by their
 * command count (a decode token's frames share one), and coli_sycl_fprof_report prints the modal
 * bucket's table once, at the chain's shutdown; =2 also dumps one frame command by command (kind,
 * start, duration, the gap before it). The chain's thread alone; the events are read after the
 * wait, so nothing here waits on the device. Reading ~900 events a frame costs host time after
 * the wait: the wall a token under this env is not the production wall (run without it for that),
 * the device times are. A prompt chunk's matrix-engine GEMMs (coli_sycl_xmx.cpp) are not hooked:
 * a prompt frame's books are incomplete, a decode frame's are whole. */
extern int coli_sycl_fprof_flag;                          /* the env was set: the hooks are live */
extern void coli_sycl_fprof_push(const sycl::event &e);   /* a command of the compute queue, unnamed yet */
extern void coli_sycl_fprof_name(const char *kind);       /* names every command pushed since the last name */
extern void coli_sycl_fprof_frame(double host_ms, double submit_ms, double wait_ms);   /* the frame was waited for */
extern void coli_sycl_fprof_report(void);                 /* once (a repeat is ignored) */
inline const sycl::event &coli_sycl_fprof_ev(const sycl::event &e) { if (coli_sycl_fprof_flag) coli_sycl_fprof_push(e); return e; }

#endif /* COLI_SYCL_H */
