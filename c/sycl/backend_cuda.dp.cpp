/* backend_cuda.dp.cpp -- the CUDA backend (backend_cuda.cu) on Intel GPUs through SYCL: dpct's
 * migration (c/sycl/MIGRATION.md) with the hand edits that make it a backend: the device profile
 * and queues of coli_sycl.h in place of the CUDA runtime's devices and streams, the sub-group
 * dispatch of the warp-32-style kernels (each at sub-group 16 and 32), the scratch arena in place
 * of the stream-ordered allocator, the matrix engines' GEMMs (coli_sycl_xmx.cpp) in place of the
 * tensor-core kernels, which are gone. The C ABI (backend_cuda.h) is the CUDA build's. */
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>   /* dim3, atomic_fetch_add, byte_level_permute, dp4a: the helpers the migration kept */
#include "backend_cuda.h"
#define COLI_CUDA_CHAIN_NO_WRAPPERS
#include "cuda_chain.h"   /* CcExpert, CcMoe: the chain's routed experts (coli_cuda_moe_async) */
#include "fp8_format.h"   /* FP8_BLOCK: the shared fmt=8 scale-block edge (see that header) */

#include "backend_gpu_compat.h"
#include "coli_sycl.h"    /* the profile, the per-device queues and arena, the SG dispatch, the XMX GEMMs */
#include "coli_sycl_matmul_plan.h"
#include "fp8_host_silu.h"
#include "backend_sycl.h"  /* the error protocol, the shuffles, the SG launch: shared with cuda_bd.dp.hpp and cu_xb */
#include "cuda_bd.dp.hpp" /* a GGUF's block formats: fmt 100 + ggml type (blockdec_host.h's table) */

static sycl::queue &dq(void);   /* the current device's queue, where the legacy stream's work goes (below) */

static_assert(FP8_BLOCK == 128, "fmt=8 on-disk containers carry ceil(dim/128)-edged scale "
              "grids (mint tool, docs/FORMATS.md); FP8_BLOCK is container format, not a "
              "tunable -- an edit here is a format change");

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <chrono>
#include <atomic>
#include <mutex>
#include <string>
#include <vector>

#include <fcntl.h>
#include <unistd.h>
#include <cmath>




struct RaggedKVEntry {
    const void *key;
    const float *host_l,*host_r;
    float **latent_pages,**rope_pages;
    int length,page_count,K,R;
};

#ifndef COLI_KV_PAGE_TOKENS
#define COLI_KV_PAGE_TOKENS 64
#endif

static void ragged_kv_clear(RaggedKVEntry *e) {
    for (int i=0;i<e->page_count;i++) {
        if (e->latent_pages[i])
            sycl::free(e->latent_pages[i], dq());
        if (e->rope_pages[i])
            sycl::free(e->rope_pages[i], dq());
    }
    std::free(e->latent_pages);
    std::free(e->rope_pages);
    e->latent_pages=e->rope_pages=nullptr;
    e->length=e->page_count=0;
}

struct ColiCudaTensor {
    void *weights;
    float *scales;
    size_t weight_bytes;
    int fmt, I, O, device;
    int gs;                    /* quant group size; 0 = per-row scales (#334) */
    int ng;                    /* number of scale groups per row = ceil(I/gs) for fmt=4 */
    int row_words;             /* fmt >= 100 (a GGUF's blocks): a row's 32-bit words, zero-padded */
    size_t scale_count;        /* scale elements: ue8m0 bytes for fmt=7, floats otherwise */
    int tracked;
    int weights_owned;
    int scales_borrowed;       /* the scales are in memory the caller owns too (an arena's slot, coli_cuda_tensor_at_g) */
    RaggedKVEntry ragged[512];
    int ragged_count;
};

static size_t tensor_scale_bytes(const ColiCudaTensor *t) {
    if (!t->fmt || t->fmt == 6) return 0;
    return t->scale_count * (t->fmt == 7 ? sizeof(uint8_t) : sizeof(float));
}

typedef struct dpct_type_677673 {
    int device;
    int compute_major,compute_minor;
    float *x, *y, *gate, *up;
    size_t x_cap, y_cap, gate_cap, up_cap;
    /* Streaming MXFP4 weights are refreshed on every call; only storage is reused. */
    void *mxfp4_weights, *mxfp4_scales;
    size_t mxfp4_weights_cap, mxfp4_scales_cap;
    void *mxfp4_expert_weights, *mxfp4_expert_scales;
    size_t mxfp4_expert_weights_cap, mxfp4_expert_scales_cap;
    /* Staging of the resident dense matvec (coli_cuda_matmul), apart from
     * x/y: the expert group (coli_cuda_expert_group_issue) runs on ctx->stream
     * asynchronously while the engine's thread keeps computing -- qwen38's
     * shared expert and the Qwen3.8 trunk go through coli_cuda_matmul between
     * qt_issue and qt_take. Sharing x/y there overwrote the group's input and
     * output mid-flight (silent, output-only corruption, no CUDA error). */
    float *dx, *dy; size_t dx_cap, dy_cap;
    uint8_t *qx; float *qscale;
    size_t qx_cap, qscale_cap;
    float *host_x,*host_y,*host_kv,*host_gate,*host_up;
    size_t host_x_cap,host_y_cap,host_kv_cap,host_gate_cap,host_up_cap;
    float *host_desc; size_t host_desc_cap;   /* COLI_SYCL_F8_ASYNC_WAITS: an issued group's descriptors, page-locked */
    float *aq,*al,*ar,*ac; size_t aq_cap,al_cap,ar_cap,ac_cap;
    float *pipe_buf[27]; size_t pipe_cap[27];   /* scratch persistenti del resident pipeline */
    sycl::queue * stream;
    sycl::event * ev_done;
        int ev_done_ok; /* resident-group issue completion (#431 PR-C0) */
    void *group_desc; size_t group_desc_cap;
    size_t tensor_count, tensor_bytes;
    int group_pending; size_t group_pending_bytes;   /* async expert-group in flight (Inc.4) */
    void *moe_tab; size_t moe_tab_cap; float *moe_host; size_t moe_host_cap;   /* group_moe_launch's table and row ids */
    bd_u32 *bdtab;          /* blockdec_host.h's table, made with the first block-format tensor */
    size_t smem_optin;      /* shared bytes a block may ask for (the block GEMVs past 48 KiB) */
    int sms;                /* multiprocessors (f32_gemm_launch's tile): Xe-cores here */
    int sg;                 /* the sub-group the warp-32-style kernels run at: the profile's sg_default, 16 or 32 */
    int wpb;                /* sub-groups ("warps") a block of 256 threads: 256 / sg */
    int has_fp64;           /* the kernels with double sums (fmt 8's warp form, DeltaNet, the pipe's rmsnorm) run */
    uint8_t *e8_grid;       /* quant.h's E8 codebook, 256 x 4 bytes (coli_cuda_e8_set_grid): CUDA's __constant__ c_e8_grid */
    float *e4m3;            /* quant.h's E4M3_LUT, 256 floats (coli_cuda_fp8_set_lut): CUDA's c_e4m3 */
    char *moe_stage; size_t moe_stage_slot; int moe_stage_n;   /* coli_cuda_moe_stage: a decode step's host experts' copies */
} DeviceContext;

typedef struct dpct_type_958039 {
    const void *g,*u,*d; const float *gs,*us,*ds;
    int gf,uf,df,rows,offset;
    int ggs,ugs,dgs;      /* per-tensor quant group size; 0 = per-row scales (#334 fmt=4) */
} GroupDesc;

static DeviceContext g_ctx[COLI_CUDA_MAX_DEVICES];
static int g_nctx;
static uint64_t g_group_calls,g_group_experts,g_group_rows;
static double g_group_h2d_ms,g_group_kernel_ms,g_group_d2h_ms;
static uint64_t g_device_group_calls[COLI_CUDA_MAX_DEVICES];
static uint64_t g_device_group_experts[COLI_CUDA_MAX_DEVICES];
static uint64_t g_device_group_rows[COLI_CUDA_MAX_DEVICES];
static double g_device_group_h2d_ms[COLI_CUDA_MAX_DEVICES];
static double g_device_group_kernel_ms[COLI_CUDA_MAX_DEVICES];
static double g_device_group_d2h_ms[COLI_CUDA_MAX_DEVICES];
static std::mutex g_group_stats_mu;

/* ---- the host-side protocol, where the CUDA runtime's was ----------------------------------
 *   - errors: a SYCL call that fails throws. COLI_SYCL_TRY(expr) runs one in a try block and gives
 *     the code cuda_ok takes (0 success, 1 a failure, logged); every entry point is a function-try-
 *     block (dpct's), whose handler sycl_caught logs the exception and returns the failure the ABI
 *     documents (0, NULL, nothing); both are backend_sycl.h's. Either marks the backend lost
 *     (g_sycl_lost), and so does an asynchronous error the runtime hands to a queue's handler
 *     instead of throwing (coli_sycl_lost, polled by fault_injected): fault_injected then refuses
 *     every compute entry, as under COLI_GPU_FAIL_AFTER, and the engine keeps its CPU paths (the
 *     chain has its own cc_lost). cuda_ok(0, "... launch")
 *     after a submit is the CUDA cudaGetLastError check: a submit that fails throws into the
 *     handler, so the check itself always passes.
 *   - the current device (cudaSetDevice) is select_ctx's thread-local ordinal, as it was per thread
 *     in the CUDA runtime, and the legacy default stream of that device, dq(), is its in-order
 *     compute queue (coli_sycl_queue): the allocations, the synchronous uploads and the resident
 *     matmuls' round trips go through it as they went through stream 0. ctx->stream, the
 *     non-blocking stream the expert groups ran on, is the same queue: one in-order queue a device
 *     orders everything, at the price of the overlap the two CUDA streams had between a group in
 *     flight and a resident matmul. The queues the chain passes in (void *stream) are its own. */
static int cuda_ok(int err, const char *what) {
    if (err == 0) return 1;
    /* cudaGetErrorString's words: "out of memory" for an allocation the device refused
     * (COLI_SYCL_NOMEM, backend_sycl.h), else a call that threw (its text logged by sycl_caught) */
    std::fprintf(stderr, "[CUDA] %s: %s\n", what, err == COLI_SYCL_NOMEM ? "out of memory" : "failed");
    return 0;
}

static DeviceContext *find_ctx(int device) {
    for (int i = 0; i < g_nctx; i++) if (g_ctx[i].device == device) return &g_ctx[i];
    return nullptr;
}

/* cudaSetDevice on every call doubles expert-matmul time on 2 GPUs when the
 * serial expert loop alternates devices (measured on RTX 5090 + 4090: 14.3s
 * -> 25.4s per 32 tokens). The current device is per-thread in the CUDA
 * runtime, so a thread-local cache skips the redundant switches. Here it is
 * the ordinal dq() and cur_ctx() read: there is nothing to switch. */
static thread_local int g_current_device = -1;

static int select_ctx(DeviceContext *ctx) {
    if (!ctx || !coli_sycl_is_open(ctx->device)) return 0;
    g_current_device = ctx->device;
    return 1;
}
/* the current device's queue (where the legacy stream's work goes), and its context: the device
 * select_ctx chose on this thread, else the first open one (coli_cuda_tensor_vram asks before
 * any entry point selected: CUDA's device 0 default) */
static int cur_device(void) {
    return g_current_device >= 0 ? g_current_device : g_nctx ? g_ctx[0].device : -1;
}
static sycl::queue &dq(void) {
    int d = cur_device();
    if (d < 0 || !coli_sycl_is_open(d)) { std::fprintf(stderr, "[CUDA] no device is open (coli_cuda_init first)\n"); abort(); }
    return coli_sycl_queue(d);
}
static DeviceContext *cur_ctx(void) {
    DeviceContext *c = find_ctx(cur_device());
    return c ? c : &g_ctx[0];
}
/* cudaEventRecord / cudaEventDestroy: an event is a barrier's, on the queue (an in-order queue:
 * every submit before it) */
static inline void sync_barrier(sycl::event *e, sycl::queue *q) { *e = q->ext_oneapi_submit_barrier(); }
static inline void destroy_event(sycl::event *e) { delete e; }
/* the device's memory as the runtime reports it (cudaMemGetInfo): coli_sycl_mem_info's figure,
 * the device's own free count where it has the query (ext_intel_free_memory: ZES_ENABLE_SYSMAN=1,
 * set before the runtime starts), else the total less the backend's books (this unit's resident
 * tensors, the chain's buffers, cu_xb's pools, the arena), since the engine budgets on it (the
 * tier's allowance, the chain's prompt chunk) and "free = total" on a filled card planned it
 * twice. coli_cuda_alloc_footprint's probe then measures nothing (its blocks are not booked) and
 * keeps the logical size, the behaviour on a device without the query. */
static int dev_mem_info(int device, size_t *free_bytes, size_t *total_bytes) {
    return coli_sycl_mem_info(device, free_bytes, total_bytes, nullptr);
}
/* this unit's books for coli_sycl_mem_info: a device's resident tensors */
static size_t backend_booked_bytes(int device) {
    size_t bytes = 0;
    coli_cuda_stats(device, nullptr, &bytes);
    return bytes;
}
/* cudaMemcpyPeerAsync: a plain memcpy on the queue where both devices are in one SYCL context
 * (USM device memory of a context is accessible to its devices); each device opened by
 * coli_sycl_open has a context of its own, so a copy between two of them is refused (1, an
 * error the caller reports) until the GLM pipe asks for a shared context. */
static int peer_copy(sycl::queue *q, int dst_dev, void *dst, int src_dev, const void *src, size_t bytes, int wait) {
    if (dst_dev != src_dev && !(coli_sycl_is_open(dst_dev) && coli_sycl_is_open(src_dev) &&
                                coli_sycl_context(dst_dev) == coli_sycl_context(src_dev))) {
        std::fprintf(stderr, "[CUDA] peer copy device %d -> %d: the devices are in different SYCL contexts (unsupported)\n", src_dev, dst_dev);
        return 1;
    }
    return COLI_SYCL_TRY(sycl::event ev = q->memcpy(dst, src, bytes); if (wait) ev.wait());
}
/* the matrix engines as the family would run them: the profile has the f16 8x16x16 shape (and
 * COLI_SYCL_XMX is not 0). Without it the family would run its plain f32 tile over the same
 * staging (coli_sycl_xmx_gemm_ok answers for the shape and format alone); the migrated warp and
 * block kernels are kept instead, the arithmetic of a CUDA device without tensor cores and what
 * COLI_SYCL_XMX=0 promises to bring back */
static int xmx_engines(const coli_sycl_profile &p) { return coli_sycl_xmx_supported(p, COLI_SYCL_DT_F16) ? 1 : 0; }
/* the matrix engines' dense GEMM takes this tensor's shape and format on this device
 * (coli_sycl_xmx_gemm_ok); and a one-line note at the first use of each prompt-GEMM site */
static int xmx_dense_ok(const DeviceContext *ctx, int fmt, int gs, int K, int N) {
    if (!ctx) return 0;
    const coli_sycl_profile &p = coli_sycl_profile_for(ctx->device);
    return xmx_engines(p) && coli_sycl_xmx_gemm_ok(p, fmt, gs, K, N);
}
/* Activation quantization changes prompt numerics. Opt in independently of XMX;
 * decode/verify keep their GEMV arithmetic. Bound the worst signed int8 dot so
 * the s32 accumulator cannot overflow (K * 128 * 127 <= INT32_MAX). */
static int xmx_dense_s8(const DeviceContext *ctx, int fmt, int K) {
    static int on = -1;
    if (on < 0) { const char *e = getenv("COLI_SYCL_XMX_S8"); on = e && atoi(e) == 1; }
    const coli_sycl_profile &p = coli_sycl_profile_for(ctx->device);
    return on && fmt == 1 && K <= 131071 && xmx_engines(p) &&
           (p.xmx_shapes & COLI_SYCL_XMX_INT8);
}
static void xmx_note(const char *site, int taken) {
    static const char *seen[8]; static int n;
    for (int i = 0; i < n; i++) if (!strcmp(seen[i], site)) return;
    if (n < 8) seen[n++] = site;
    std::fprintf(stderr, "[SYCL] %s: %s\n", site,
                 taken ? "XMX joint_matrix GEMM (coli_sycl_xmx)" : "portable block kernels (no XMX GEMM for this device, shape or format)");
}
#define XMX_BK 64   /* the K step of coli_sycl_xmx_gemm / _moe: K a multiple of 64 (coli_sycl.h) */

/* fmt=6 (E8/IQ3) geometry, mirroring quant.h. A super-block packs 256 weights
 * into 98 bytes: 64 codebook indices, 8 words of (4x7 signs + 4-bit sub-scale),
 * and one fp16 super-scale. Scales live INSIDE the block, so fmt=6 tensors carry
 * no separate scale array (#452). */
#define COLI_E8_QK      256
#define COLI_E8_SUB      32
#define COLI_E8_BBYTES   98

static size_t row_bytes(int fmt, int I) {
    if (fmt == 0) return (size_t)I * sizeof(float);
    if (fmt == 1) return (size_t)I;
    if (fmt == 2 || fmt == 4) return (size_t)(I + 1) / 2;   /* fmt=4: same packed int4 */
    if (fmt == 3) return (size_t)(I + 3) / 4;
    if (fmt == 4) return (size_t)(I + 1) / 2;   /* grouped int4: nibbles like fmt 2 */
    if (fmt == 7) return (size_t)(I + 1) / 2;   /* MXFP4: e2m1 nibbles, 2 per byte */
    if (fmt == 6) return (size_t)(((int64_t)I + COLI_E8_QK - 1) / COLI_E8_QK) * COLI_E8_BBYTES;
    if (fmt == 8) return (size_t)I;             /* fp8-e4m3: raw bytes, layout of fmt=1 */
    return 0;
}

/* The E8 codebook (256 x 4 bytes) and the fmt=8 e4m3 decode table (256 floats), each uploaded
 * once per device from quant.h's e8_grid and E4M3_LUT so the tables have a single source of truth
 * and cannot drift from the CPU decoder: CUDA's __constant__ symbols are a device array a context
 * here (DeviceContext.e8_grid, .e4m3: zeroed at init, the kernels take them as plain pointers).
 * Uploads of fmt=8 tensors are refused until the LUT is published (g_fp8_lut_ready): a kernel
 * reading the zero-initialized table would compute silent zeros, the exact failure mode this
 * format's dispatch work exists to prevent. */
static int g_fp8_lut_ready;

/* A super-block is 98 bytes, so nothing inside it is guaranteed 4- or 2-byte
 * aligned: assemble the words byte-wise instead of dereferencing. */
__dpct_inline__ uint32_t e8_ld_u32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1]<<8) | ((uint32_t)p[2]<<16) | ((uint32_t)p[3]<<24);
}
/* Mirrors e8_fp16_to_f32 rather than calling __half2float, so the two decoders
 * cannot disagree on subnormals. */
__dpct_inline__ float e8_fp16(const uint8_t *p) {
    uint16_t h = (uint16_t)p[0] | ((uint16_t)p[1]<<8);
    uint32_t sign=(uint32_t)(h&0x8000)<<16, exp=(h>>10)&0x1F, man=h&0x3FF, bits;
    if (!exp)         bits = man ? (sign|((127u-15u+1u-1u)<<23)|(man<<13)) : sign;
    else if (exp==31) bits = sign|0x7F800000u|(man<<13);
    else              bits = sign|((exp+112u)<<23)|(man<<13);
    float f; memcpy(&f,&bits,4); return f;
}
/* Expand one 32-weight sub-block; mirrors e8_expand_sub in quant.h. */
__dpct_inline__ void
e8_expand_sub_dev(const uint8_t *blk, int ib, float d, float *out,
                  const uint8_t *c_e8_grid) {
    uint32_t word = e8_ld_u32(blk + COLI_E8_QK/4 + ib*4);
    float db = d * (0.5f + (float)((word>>28)&0xF)) * 0.5f;
    const uint8_t *idx = blk + ib*8;
    for (int l=0;l<4;l++){
        uint32_t seven=(word>>(7*l))&0x7F;
        const uint8_t *g0=c_e8_grid+4*idx[l*2+0], *g1=c_e8_grid+4*idx[l*2+1];
        int par=0;
        for (int j=0;j<8;j++){
            int neg = j<7 ? (int)((seven>>j)&1) : 0;
            if (j<7) par^=neg; else neg=par;        /* odd parity closes the block */
            float mag = (j<4 ? (float)g0[j] : (float)g1[j-4]) * 0.5f;
            out[l*8+j] = neg ? -mag*db : mag*db;
        }
    }
}

/* ---- MXFP4 (OCP microscaling FP4), fmt=7 -----------------------------------
 * Same layout the CPU path decodes in quant.h's matmul_mxfp4, and the same two
 * tricks, so the two agree bit for bit:
 *
 *   packed [O, I/2]  u8 — e2m1 nibbles, LOW nibble = even column, bit3 = sign,
 *                         bits 0..2 index {0,.5,1,1.5,2,3,4,6}
 *   scales [O, I/32] u8 — ue8m0 exponent per 32-column group, w = v * 2^(s-127)
 *
 * The exponent is decoded as a bit pattern rather than exp2f: (uint32)s << 23
 * reinterpreted as float IS 2^(s-127) for s in [1,254], and reproduces the CPU
 * path's documented edge behaviour exactly -- s=0 gives +0 and s=255 gives +inf
 * on both sides. Using exp2f here would agree for the normal range and diverge
 * at the ends, which is precisely where a silent mismatch would hide.
 *
 * The LUT holds DOUBLED values so every entry is an exact small integer; the
 * compensating 0.5f rides along in mx4_scale_dev, as it does on the CPU. */
/* Decoded arithmetically rather than from a __constant__ table: a file-scope
 * __constant__ array with static linkage is initialised per translation unit,
 * and this kernel is also compiled into the HIP build and the DLL, where that
 * silently yields garbage. The magnitude is 2^(exp-1) * 0.5 for exp in 1..3 and
 * 0 for exp 0, which is exactly the OCP e2m1 table {0,.5,1,1.5,2,3,4,6}. */
static inline float mx4_decode(int n) {
    int mant = n & 1, exp = (n >> 1) & 3;
    float mag = exp ? sycl::ldexp(1.0f + 0.5f * (float)mant, exp - 1)
                    : 0.5f * (float)mant;
    return (n & 8) ? -mag : mag;
}

static inline float mx4_scale_dev(uint8_t s) {
    union { uint32_t u; float f; } b;
    b.u = static_cast<uint32_t>(s) << 23;
    return b.f;
}

/* e2m1 nibble at column i of a packed row. */
static inline float mx4_weight_at(const uint8_t *q, int i) {
    uint8_t v = q[i >> 1];
    return mx4_decode((i & 1) ? (v >> 4) : (v & 15));
}

/* Generic per-element weight decode for the kernels that need one (the absorb
 * kernels and the generic grouped-expert path). fmt=3 (int2) is now an EXPLICIT
 * branch and the fall-through is a refusal.
 *
 * It used to be the other way round: int2 was the fall-through, so every format
 * this function does not decode -- fmt=5 (int3-g64), fmt=6 (E8/IQ3), and
 * anything added later -- was read as 2-bit values and returned numbers.
 * (fmt=8 was in that misread set too, then refused, until it gained its own
 * explicit branch below for the absorb path.) Meanwhile the CPU functions
 * doing the same job on the same tensor, qt_addrow and qt_matvec_rows
 * (colibri.c), both exit(1) naming the function and the fmt. Two backends,
 * identical unsupported input, one refusing and one fabricating: that
 * asymmetry is the defect, independent of any particular format's arrival.
 *
 * WHY __trap() AND NOT A DIAGNOSTIC. This is device code inside a running
 * kernel; there is no stderr to name the tensor on and no way to unwind. __trap
 * aborts the kernel and poisons the context, so the next cuda_ok() call on the
 * host reports a failure instead of the caller consuming fabricated values --
 * the same "stop rather than misread" outcome as the CPU's exit(1), reached the
 * only way device code can reach it.
 *
 * IT IS A BACKSTOP, NOT THE PRIMARY GATE -- and the gates above it are several
 * DIFFERENT checks, not one. Naming them exactly, because "every launcher checks
 * coli_cuda_weight_at_supported" would be false and a reader will verify it:
 *   - UPLOAD is the widest gate. coli_cuda_tensor_upload{,_g} refuse when
 *     row_bytes(fmt,I) == 0, which is every fmt this file has no row stride for --
 *     fmt=5 (int3-g64), every negative fmt, every unknown fmt. Those can never
 *     become a ColiCudaTensor at all, so the uploadable set is {0,1,2,3,4,6,7,8}.
 *   - quant_matmul dispatches 6, 7, 4 and 8 in explicit branches of its own, so
 *     the generic else that calls this function sees only 0/1/2/3 out of that set.
 *   - the generic grouped-expert path refuses gf>3 || uf>3 || df>3 on the host.
 *   - the absorb call sites (the other caller of this function) are the ones
 *     gated by coli_cuda_weight_at_supported, via absorb_fmt_ok below -- one
 *     caller, not "every launcher".
 * Each of those is sufficient on its own for the sites it covers; this trap
 * exists because none of them is a property of THIS function. Reaching this line
 * means a launch site got past its own gate, which is a bug in that gate. */
static float weight_at(const void *weights, int fmt, size_t row, int i,
                       float const *c_e4m3) {
    const uint8_t *base = static_cast<const uint8_t *>(weights) + row;
    if (fmt == 0) return reinterpret_cast<const float *>(base)[i];
    if (fmt == 1) return static_cast<float>(reinterpret_cast<const int8_t *>(base)[i]);
    /* fmt=8 (fp8-e4m3): raw byte, same layout as fmt=1 (row_bytes(8,I)==I), decoded
     * through the shared c_e4m3 LUT (same table quant_matmul's fmt==8 branch reads,
     * uploaded once by coli_cuda_fp8_set_lut). Callers gate on the LUT being live
     * before a fmt=8 tensor ever reaches this function (coli_cuda_tensor_upload
     * refuses the upload otherwise), so the table is always populated here. Returns
     * the decoded WEIGHT only, unscaled -- absorb_scale below applies the
     * per-128x128-block scale, exactly like every other quantized fmt returns
     * unscaled through this function. */
    if (fmt == 8) return c_e4m3[base[i]];
    const uint8_t *q = base;
    if (fmt == 2 || fmt == 4) {                               /* fmt=4: same nibble layout */
        uint8_t v = q[i >> 1];
        int n=(i&1)?(v>>4):(v&15); return static_cast<float>(n&8?n-16:n);
    }
    if (fmt == 3) {                                           /* int2 */
        uint8_t v = q[i >> 2];
        return static_cast<float>(((v >> ((i & 3) * 2)) & 3) - 2);
    }
    assert(0);
    return 0.0f;   /* not reached: __trap() does not return */
}

/* Scale for output `row`, input element `k`. Three layouts reach this: fmt=4
 * (grouped int4) stores ng scales per row at scales[row*ng + k/gs]; fmt=8
 * (fp8-e4m3) stores one scale per 128x128 BLOCK, block-row-major, and is
 * handled by its own branch below; every OTHER quantized format has one scale
 * per row at scales[row]. Mirrors quant_matmul's fmt==4 branch so the
 * attention absorb kernels apply per-group scales instead of the per-row
 * (fmt=2) semantic that crashed #298's g64 kv_b. */
static float absorb_scale(const float *wscale, int fmt, int gs, int ng, int row, int k) {
    if (!fmt) return 1.f;
    if (fmt == 8) {
        /* fp8-e4m3: one f32 scale per 128x128 BLOCK, block-row-major
         * ([ceil(O/128), ceil(I/128)]), exactly quant_matmul's fmt==8 indexing
         * (scl[i >> 7] on a scale row selected by o >> 7) and matmul_fp8's CPU
         * reference (quant.h). `ng` here is coli_cuda_tensor_upload's t->ng,
         * which for fmt=8 is set to ceil(I/128) specifically (not the fmt=4
         * group count) -- see the upload-time assignment there. `gs` is unused
         * for fmt=8 (always 0, only fmt=4 sets it), so the block edge is the
         * fixed FP8_BLOCK constant (fp8_format.h, shared with the CPU side),
         * not a caller-supplied group size. Rounding note: the GEOMETRY here
         * matches quant_matmul_f8w/matmul_fp8, but their fp8 accumulation
         * convention (f32 partial per block, scale once per partial, double
         * across blocks) is NOT carried into the absorb kernels -- they apply
         * the scale per element into a float accumulator, matching their own
         * fmt=4 arm's long-standing behavior; CPU-vs-CUDA absorb divergence
         * is an accepted, documented class (#510). */
        int rowBlk = row / FP8_BLOCK, colBlk = k / FP8_BLOCK;
        return wscale[(size_t)rowBlk * ng + colBlk];
    }
    if (fmt != 4) return wscale[row];
    int g = k / gs; if (g >= ng) g = ng - 1;   /* tail of the last (partial) group */
    return wscale[(size_t)row * ng + g];
}

static void offset_to_signed_s4(uint8_t *q,size_t n){
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    size_t i = (size_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
               item_ct1.get_local_id(2);
        if (i < n) q[i] ^= 0x88;
}

/* ---- fmt=8 (fp8-e4m3) warp decode/accumulate helpers -----------------------
 * The fmt=8 dot products are compared against the CPU reference (quant.h
 * matmul_fp8) by the cross-tier parity tests, so this TU must never be built
 * with fast-math: nvcc's --use_fast_math implies -ftz=true, which flushes the
 * scale*subnormal contributions the denormal-scale test pins down. (-ftz has
 * no macro of its own to test; the Makefile pins -ftz=false on the nvcc line.) */
#if defined(__FAST_MATH__)   /* icpx defaults to -fp-model=fast: build with -fp-model=precise */
#error "backend_cuda.dp.cpp: fmt=8 kernels forbid fast-math builds (FTZ breaks CPU parity)"
#endif

/* Fixed-order 5-shuffle reduce over a 32-lane logical warp. Width pinned to 32
 * so a HIP wave64 device does not fold two output rows into one reduction. */

/* Decode one e4m3 byte. Default: a __shared__ copy of c_e4m3 — per-thread
 * data-dependent indices are the __constant__ cache's documented worst case
 * (accesses to different addresses within a warp serialize), while shared
 * memory takes them at full rate. HW=1 (COLI_CUDA_F8_WARP=2, CUDA-only): the
 * cuda_fp8.h route. Every finite e4m3 value is exactly representable in f16,
 * so the two paths should agree bit for bit — but the on-device 256-value
 * sweep is the authority, and the LUT stays the default until the sweep
 * certifies the cvt path on the target silicon. */
template <int HW> __dpct_inline__ float f8_dec(const float *slut, uint8_t b) {
    return slut[b];
}

/* One 128-column scale block, one warp: 32 lanes x 4 bytes is the exact
 * FP8_BLOCK fit. Weights are decoded once and reused across ns activation
 * rows (weights are the traffic; activations come from cache), then each
 * row's per-lane products are tree-reduced in fixed order — the block partial
 * lands in LANE 0 of gp[]/up_[] (other lanes hold partials). vec picks the
 * LOAD width only (uchar4/float4 on aligned full blocks, guarded bytes on
 * tails and unaligned bases — e.g. zero-copy views: correct, slower); both
 * load paths feed ONE shared product/accumulate sequence below, and the
 * vec/byte parity test asserts their bit-equality on identical data. Tail
 * padding contributes +0.0f terms, which can at most flip a -0.0 partial to
 * +0.0. DUAL folds a second weight row (gate+up) against the same
 * activations without re-reading x. The per-row arithmetic never depends on
 * ns or s, so S-tiling cannot reorder a dot (B5). */
/* SG: the sub-group's lanes. At 32 a lane holds one 4-byte chunk of the block, the CUDA warp's
 * arithmetic exactly; at 16 two (CPL), a lane's two dots added before the shuffle tree. */
template <int HW, int DUAL, int SG>
__dpct_inline__ void f8w_block(const float *slut, const uint8_t *gr,
                               const uint8_t *ur, const float *xs, int xstride,
                               int ns, int base, int len, int vec, float *gp,
                               float *up_) {
    constexpr int CPL = 32 / SG;   /* 4-byte chunks a lane */
    int lane = sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2) & (SG - 1),
        full = vec && len == 128;
    float gw[CPL][4],uw[CPL][4]={};
    for(int c=0;c<CPL;c++){
        int i0=base+(lane+c*SG)*4;
        if(full){
            sycl::uchar4 q = *(const sycl::uchar4 *)(gr + i0);
            gw[c][0] = f8_dec<HW>(slut, q.x()); gw[c][1] = f8_dec<HW>(slut, q.y());
            gw[c][2] = f8_dec<HW>(slut, q.z()); gw[c][3] = f8_dec<HW>(slut, q.w());
            if (DUAL) { sycl::uchar4 r = *(const sycl::uchar4 *)(ur + i0);
                uw[c][0] = f8_dec<HW>(slut, r.x()); uw[c][1] = f8_dec<HW>(slut, r.y());
                uw[c][2]=f8_dec<HW>(slut,r.z());uw[c][3]=f8_dec<HW>(slut,r.w());}
        }else for(int k=0;k<4;k++){
            int in=i0+k<base+len;
            gw[c][k]=in?f8_dec<HW>(slut,gr[i0+k]):0.f;
            if(DUAL)uw[c][k]=in?f8_dec<HW>(slut,ur[i0+k]):0.f;
        }
    }
    for(int s=0;s<ns;s++){
        const float *xr=xs+(size_t)s*xstride;
        float g=0,u=0;
        for(int c=0;c<CPL;c++){
            int i0=base+(lane+c*SG)*4;
            float xv[4];
            if(full){
                sycl::float4 xf = *(const sycl::float4 *)(xr + i0);
                xv[0] = xf.x(); xv[1] = xf.y(); xv[2] = xf.z(); xv[3] = xf.w();
            }else for(int k=0;k<4;k++) xv[k]=i0+k<base+len?xr[i0+k]:0.f;
            for(int k=0;k<4;k++){g+=xv[k]*gw[c][k];if(DUAL)u+=xv[k]*uw[c][k];}
        }
        for(int off=SG/2;off;off>>=1){g+=shfl_down(g,off);if(DUAL)u+=shfl_down(u,off);}
        gp[s]=g;if(DUAL)up_[s]=u;
    }
}

/* Diagnostic counterpart of q38_matmul_fp8_vec's full 128-column block.
 * Eight FMA chains and hsum256's ((a0+a4)+(a2+a6))+((a1+a5)+(a3+a7)).
 * All SG16/32 lanes participate in each shuffle; only the first eight load
 * and accumulate. Partial blocks retain the original kernels at dispatch. */
template <int DUAL, int SG>
__dpct_inline__ void f8cpu_block(const float *lut, const uint8_t *gr,
        const uint8_t *ur, const float *xs, int stride, int ns, int base,
        float *gp, float *up) {
    static_assert(SG==16 || SG==32,"supported subgroup width");
    int lane=sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2)&(SG-1);
    for(int s=0;s<ns;s++) {
        const float *xr=xs+(size_t)s*stride;
        float g=0.f,u=0.f;
        if(lane<8)for(int i=lane;i<128;i+=8) {
            int k=base+i;float x=xr[k];
            g=sycl::fma(lut[gr[k]],x,g);
            if constexpr(DUAL)u=sycl::fma(lut[ur[k]],x,u);
        }
        for(int off=4;off;off>>=1) {
            g+=shfl_down(g,off);
            if constexpr(DUAL)u+=shfl_down(u,off);
        }
        gp[s]=g;if constexpr(DUAL)up[s]=u;
    }
}

/* One subgroup/work-group per (output,row), with ascending double block sum.
 * This deliberately isolates CPU arithmetic order, not a performance default. */
template <int SG>
static void quant_matmul_f8cpu(float *y,const float *x,const void *weights,
        const float *scales,int I,int O,const float *lut) {
    auto item=sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int lane=item.get_local_id(2),o=item.get_group(2),s=item.get_group(1);
    auto &local_lut=*sycl::ext::oneapi::group_local_memory_for_overwrite<float[256]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    for(int i=lane;i<256;i+=SG)local_lut[i]=lut[i];
    item.barrier(sycl::access::fence_space::local_space);
    const uint8_t *w=(const uint8_t*)weights+(size_t)o*I;
    const float *xr=x+(size_t)s*I,*sc=scales+(size_t)(o/128)*(I/128);
    double sum=0;
    for(int bi=0;bi<I/128;bi++) {
        float p;f8cpu_block<0,SG>(local_lut,w,nullptr,xr,0,1,bi*128,&p,nullptr);
        if(!lane)sum+=(double)p*sc[bi];
    }
    if(!lane)y[(size_t)s*O+o]=(float)sum;
}

/* Dense fmt=8 rework: quant_matmul's fmt=8 branch as its OWN kernel, so
 * COLI_CUDA_F8_WARP=0 restores the fully original dense behavior too (the
 * host picks in quant_matmul_launch). Same grid/block contract as
 * quant_matmul: one 256-thread block per (o,s). Accumulation mirrors the CPU
 * reference: f32 partial per 128-block (f8w_block), the scale applied ONCE
 * per partial, double across blocks. Warps stride the block axis; the
 * cross-warp double sum runs in fixed warp order, so the reduction order is
 * a pure function of the dims. Decode reads a shared copy of c_e4m3
 * (data-dependent __constant__ indices serialize); the cvt candidate stays
 * grouped-path-only until certified. NaN bytes decode to NaN and propagate,
 * same policy as the CPU path. */
template <int SG>
static void quant_matmul_f8w(float *y,const float *x,const void *weights,
                                        const float *scales,int S,int I,int O,
                                        float const *c_e4m3){
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int o = item_ct1.get_group(2), s = item_ct1.get_group(1);
    const float *xs=x+(size_t)s*I;
    auto &slut =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[256]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &dsum =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<double[32]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    for (int i = item_ct1.get_local_id(2); i < 256;
         i += item_ct1.get_local_range(2)) slut[i] = c_e4m3[i];
    item_ct1.barrier(sycl::access::fence_space::local_space);
    const uint8_t *wrow=(const uint8_t*)weights+(size_t)o*I;
    const float *scl=scales+(size_t)(o>>7)*(size_t)((I+127)>>7);
    int warp = item_ct1.get_local_id(2) / SG,
        nw = item_ct1.get_local_range(2) / SG, nblk = (I + 127) >> 7;
    int vec=!(I&3)&&!((size_t)wrow&3)&&!((size_t)xs&15);
    double a=0;
    for(int bi=warp;bi<nblk;bi+=nw){
        int base=bi<<7,len=I-base<128?I-base:128;
        float p;
        f8w_block<0,0,SG>(slut,wrow,nullptr,xs,0,1,base,len,vec,&p,nullptr);
        if (!(item_ct1.get_local_id(2) & (SG - 1))) a += (double)p * scl[bi];
    }
    if (!(item_ct1.get_local_id(2) & (SG - 1))) dsum[warp] = a;
    item_ct1.barrier(sycl::access::fence_space::local_space);
    if (!item_ct1.get_local_id(2)) {
        for(int w=1;w<nw;w++) a+=dsum[w];
        y[(size_t)s*O+o]=(float)a;
    }
    (void)S;
}

static void quant_matmul(float *y, const float *x, const void *weights,
                         const float *scales, int fmt, int S, int I, int O,
                         size_t rb, int gs, int ng,
                         const uint8_t *c_e8_grid,
                         float const *c_e4m3) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int o = item_ct1.get_group(2);
    int s = item_ct1.get_group(1);
    float sum = 0.0f;
    size_t row = (size_t)o * rb;
    const float *xs = x + (size_t)s * I;
    if (fmt == 6) {
        /* E8/IQ3: decode is per 32-weight sub-block, so threads stride over
         * sub-blocks rather than elements -- expanding once per 32 weights
         * instead of redoing the word/parity work for every element. */
        const uint8_t *wrow = static_cast<const uint8_t *>(weights) + row;
        int nsub = (I + COLI_E8_SUB - 1) / COLI_E8_SUB;
        for (int sb = item_ct1.get_local_id(2); sb < nsub;
             sb += item_ct1.get_local_range(2)) {
            const uint8_t *blk = wrow + (size_t)(sb / (COLI_E8_QK/COLI_E8_SUB)) * COLI_E8_BBYTES;
            float w[COLI_E8_SUB];
            e8_expand_sub_dev(blk, sb % (COLI_E8_QK / COLI_E8_SUB),
                              e8_fp16(blk + 96), w, c_e8_grid);
            int off = sb*COLI_E8_SUB, n = I-off < COLI_E8_SUB ? I-off : COLI_E8_SUB;
            for (int k=0;k<n;k++) sum += xs[off+k]*w[k];
        }
    } else if (fmt == 7) {
        /* MXFP4: one ue8m0 exponent per 32 columns. The SCALAR CPU reference
         * (quant.h matmul_mxfp4's scalar loop -- its AVX2 sibling applies the
         * scale per 8-lane FMA instead and disagrees with it at s=255)
         * accumulates each 32-group UNSCALED and multiplies the group
         * subtotal by the scale once. For scales in [0,254] -- every value a
         * real checkpoint contains -- the scale is a finite power of two,
         * scaling each element or the subtotal is exact either way, so
         * threads stride over columns exactly as before and outputs stay
         * bit-identical with prior builds. s=255 decodes to +inf (the
         * documented edge), and there the application point is visible:
         * per-element scaling turns every zero product into 0*inf = NaN,
         * where the scalar reference's finite-subtotal-times-inf keeps the
         * group's sign. Rows carrying a 255 scale byte therefore take the
         * per-group path mirroring that scalar loop; the row's +-inf/NaN
         * classification does not depend on summation order unless several
         * large-finite (s~254) group results overflow only in aggregate --
         * out-of-domain either way -- so the tree reduce below needs no
         * change. */
        const uint8_t *wrow = static_cast<const uint8_t *>(weights) + row;
        const uint8_t *scl = reinterpret_cast<const uint8_t *>(scales) + (size_t)o * ng;
        auto &scale_inf =
            *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(
                sycl::ext::oneapi::this_work_item::get_work_group<3>());
        if (!item_ct1.get_local_id(2)) scale_inf = 0;
        item_ct1.barrier(sycl::access::fence_space::local_space);
        for (int g = item_ct1.get_local_id(2); g < ng;
             g += item_ct1.get_local_range(2))
            if (scl[g] == 255) scale_inf = 1;
        item_ct1.barrier(sycl::access::fence_space::local_space);
        if (!scale_inf) {
            for (int i = item_ct1.get_local_id(2); i < I;
                 i += item_ct1.get_local_range(2)) {
                int g = i >> 5;
                if (g >= ng) g = ng - 1;
                sum += xs[i] * mx4_weight_at(wrow, i) * mx4_scale_dev(scl[g]);
            }
        } else {
            for (int g = item_ct1.get_local_id(2); g < ng;
                 g += item_ct1.get_local_range(2)) {
                /* same element->group mapping as the clamp above: the last
                 * group takes every remaining column, a group past the data
                 * contributes nothing */
                int base = g << 5, end = base + 32;
                if (g == ng - 1 || end > I) end = I;
                if (base >= end) continue;
                float ga = 0.0f;
                for (int i = base; i < end; i++)
                    ga += xs[i] * mx4_weight_at(wrow, i);
                sum += ga * mx4_scale_dev(scl[g]);
            }
        }
    } else if (fmt == 4) {
        /* Grouped int4: one f32 scale per gs elements along I (ng groups per row).
         * Scale layout: scales[o*ng + g]. Each thread strides through I, applying
         * the appropriate group scale as it crosses group boundaries. This matches
         * the CPU matmul_i4_grouped accumulation exactly. */
        const float *scl = scales + (size_t)o * ng;
        for (int i = item_ct1.get_local_id(2); i < I;
             i += item_ct1.get_local_range(2)) {
            int g = i / gs;
            if (g >= ng) g = ng - 1;  /* tail elements in the last (partial) group */
            sum += xs[i] * weight_at(weights, fmt, row, i, c_e4m3) * scl[g];
        }
    } else if (fmt == 8) {
        /* fp8-e4m3 (matmul_fp8): one byte per weight (layout of fmt=1), one f32
         * scale per 128x128 BLOCK of [O,I] — scales[(o/128)*ceil(I/128) + i/128].
         * The block edge is a fixed property of the format (FP8_BLOCK), so the
         * geometry derives from I alone and gs/ng are ignored: this branch is
         * correct no matter which call site launched it. NaN bytes decode to NaN
         * through the LUT and propagate, same policy as the CPU path. This is
         * the ORIGINAL dense path, kept for COLI_CUDA_F8_WARP=0; the default
         * routes fmt=8 to quant_matmul_f8w instead (quant_matmul_launch). */
        const uint8_t *wrow = static_cast<const uint8_t *>(weights) + row;
        const float *scl = scales + (size_t)(o / FP8_BLOCK) * (size_t)((I + FP8_BLOCK - 1) / FP8_BLOCK);
        for (int i = item_ct1.get_local_id(2); i < I;
             i += item_ct1.get_local_range(2))
            sum += xs[i] * c_e4m3[wrow[i]] * scl[i / FP8_BLOCK];
    } else {
        for (int i = item_ct1.get_local_id(2); i < I;
             i += item_ct1.get_local_range(2))
            sum += xs[i] * weight_at(weights, fmt, row, i, c_e4m3);
    }

    auto &partial =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[256]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    partial[item_ct1.get_local_id(2)] = sum;
    item_ct1.barrier(sycl::access::fence_space::local_space);
    for (int n = item_ct1.get_local_range(2) >> 1; n; n >>= 1) {
        if (item_ct1.get_local_id(2) < n) partial[item_ct1.get_local_id(2)] +=
            partial[item_ct1.get_local_id(2) + n];
        item_ct1.barrier(sycl::access::fence_space::local_space);
    }
    if (!item_ct1.get_local_id(2))
        /* fmt 4/6/7/8 already applied their scaling inside the loop: 4 and 7 are
         * per-group (one scale per gs / per 32 columns), 6 carries it in the
         * block header, 8 reads a per-128 block scale alongside the weights.
         * Only the per-row formats get the trailing multiply --
         * and for fmt=7 `scales` points at ue8m0 BYTES, so reading it as float
         * here does not merely double-scale, it reads garbage. */
        y[(size_t)s * O + o] = (fmt && fmt != 4 && fmt != 6 && fmt != 7 && fmt != 8) ? partial[0] * scales[o] : partial[0];
}

/* fmt=6 activation rotation, y = Q^T x for Q = D*H/sqrt(n) (#452). One block per
 * row; the power-of-two block is staged in shared memory, capping n at 4096
 * floats -- which covers every block GLM produces (6144 -> 2048+4096, 1536 ->
 * 512+1024). The sign stream is regenerated in-kernel from the same xorshift64*
 * that quant.h's e8_signs uses, so no rotation data is stored or uploaded.
 *
 * Placement note: all routed experts of a layer share one gate/up input, so that
 * rotation belongs to the CALLER (once per layer). This kernel exists for the
 * down projection, whose input is the per-expert silu(gate)*up product and so
 * cannot be shared -- mirroring colibri.c's split at moe(). */
static void e8_rot_rows_kernel(float *rows, int dim, int off, int n,
                               uint8_t *dpct_local){
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto sh = (float *)dpct_local;
    auto &sbits =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<uint8_t[4096 / 8]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    if (!item_ct1.get_local_id(2)) {
        uint64_t s = 417u + (uint64_t)n;
        for (int i=0;i<(n+7)/8;i++){
            s^=s>>12; s^=s<<25; s^=s>>27;
            sbits[i] = (uint8_t)((s*2685821657736338717ULL)>>56);
        }
    }
    item_ct1.barrier();
    float *row = rows + (size_t)item_ct1.get_group(2) * dim + off;
    for (int i = item_ct1.get_local_id(2); i < n;
         i += item_ct1.get_local_range(2)) {
        float v=row[i];
        sh[i] = (sbits[i>>3]>>(i&7)&1) ? -v : v;
    }
    item_ct1.barrier();
    for (int len=1;len<n;len<<=1){
        for (int j = item_ct1.get_local_id(2); j < n / 2;
             j += item_ct1.get_local_range(2)) {
            int i = (j/len)*(len<<1) + (j%len);
            float u=sh[i], v=sh[i+len];
            sh[i]=u+v; sh[i+len]=u-v;
        }
        item_ct1.barrier();
    }
    float sc = sycl::rsqrt((float)n);
    for (int i = item_ct1.get_local_id(2); i < n;
         i += item_ct1.get_local_range(2)) row[i] = sh[i] * sc;
}

/* Rotate nr rows in place, tiling non-power-of-two dims block-diagonally exactly
 * as e8_rot_rows does. Returns 0 if a block exceeds the shared-memory cap. */
static int e8_rot_rows_dev(float *rows, int nr, int dim,
                           sycl::queue * stream) {
    int off = 0;
    while (off < dim) {
        int rem = dim-off, b = rem & (-rem);
        while (b > 4096) b >>= 1;
        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range<1>((size_t)b * sizeof(float)), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, (unsigned)nr) *
                                      sycl::range<3>(1, 1, 256),
                                  sycl::range<3>(1, 1, 256)),
                [=](sycl::nd_item<3> item_ct1) {
                    e8_rot_rows_kernel(
                        rows, dim, off, b,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                });
        });
        off += b;
    }
    return 1;
}

/* Optional arithmetic diagnostic: match q38_silu's stable sigmoid expression.
 * Double exp is rounded to float before the float denominator; division is
 * rounded through double to avoid the native float reciprocal approximation.
 * This is not a claim that device exp reproduces every host libm expf bit. */
static float cpu_form_silu_mul(float g, float u) {
    float z=(float)sycl::exp((double)(g>=0.f ? -g : g));
    float den=1.f+z;
    float sig=(float)((g>=0.f ? 1.0 : (double)z)/(double)den);
    float hidden=g*sig;
    return hidden*u;
}
static void silu_cpu_mul(float *gate,const float *up,size_t n) {
    auto item=sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    size_t i=item.get_global_id(2);
    if(i<n)gate[i]=cpu_form_silu_mul(gate[i],up[i]);
}

static void silu_mul(float *gate, const float *up, size_t n) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    size_t i = (size_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
               item_ct1.get_local_id(2);
    if (i < n) {
        float v = gate[i];
        gate[i] = (v / (1.0f + sycl::native::exp(-v))) * up[i];
    }
}

static void situ_mul(float *gate, const float *up, size_t n, float b1, float b2) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    size_t i = (size_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
               item_ct1.get_local_id(2);
    if (i < n) {
        float g = gate[i], u = up[i];
        gate[i] = b1 * sycl::tanh(g / b1) *
                  (1.f / (1.f + sycl::native::exp(-g))) * b2 *
                  sycl::tanh(u / b2);
    }
}

static void grouped_hidden(float *y, const float *x, const GroupDesc *desc,
                           int I, int D, int which, float const *c_e4m3) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int o = item_ct1.get_group(2), s = item_ct1.get_group(1),
        c = item_ct1.get_group(0);
        GroupDesc d = desc[c];
    if(s>=d.rows) return;
    const void *w=which?d.u:d.g; const float *sc=which?d.us:d.gs; int fmt=which?d.uf:d.gf;
    size_t rb=row_bytes(fmt,D),row=(size_t)o*rb; const float *xs=x+(size_t)(d.offset+s)*D;
    float sum = 0; for (int i = item_ct1.get_local_id(2); i < D;
                        i += item_ct1.get_local_range(2)) sum +=
                   xs[i] * weight_at(w, fmt, row, i, c_e4m3);
    auto &p = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[256]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
        p[item_ct1.get_local_id(2)] = sum; item_ct1.barrier();
    for (int n = 128; n; n >>= 1) {
        if (item_ct1.get_local_id(2) < n) p[item_ct1.get_local_id(2)] +=
        p[item_ct1.get_local_id(2) + n];
        item_ct1.barrier();
    }
    if (!item_ct1.get_local_id(2)) y[(size_t)(d.offset + s) * I + o] =
        p[0] * (fmt ? sc[o] : 1.f);
}

static void grouped_down(float *y,const float *x,const GroupDesc *desc,int D,int I,
                         float const *c_e4m3){
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int o = item_ct1.get_group(2), s = item_ct1.get_group(1),
        c = item_ct1.get_group(0);
        GroupDesc d = desc[c];
    if(s>=d.rows) return;
    size_t rb=row_bytes(d.df,I),row=(size_t)o*rb; const float *xs=x+(size_t)(d.offset+s)*I;
    float sum = 0; for (int i = item_ct1.get_local_id(2); i < I;
                        i += item_ct1.get_local_range(2)) sum +=
                   xs[i] * weight_at(d.d, d.df, row, i, c_e4m3);
    auto &p = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[256]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
        p[item_ct1.get_local_id(2)] = sum; item_ct1.barrier();
    for (int n = 128; n; n >>= 1) {
        if (item_ct1.get_local_id(2) < n) p[item_ct1.get_local_id(2)] +=
        p[item_ct1.get_local_id(2) + n];
        item_ct1.barrier();
    }
    if (!item_ct1.get_local_id(2)) y[(size_t)(d.offset + s) * D + o] =
        p[0] * (d.df ? d.ds[o] : 1.f);
}

/* Native fmt=6 expert groups.  One block owns one (expert,row,output) and
 * expands each 32-weight E8 sub-block once for both gate/up projections. */
static void
grouped_hidden_e8_dual(float *gate, const float *x, const GroupDesc *desc,
                       int I, int D,
                       const uint8_t *c_e8_grid) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int o = item_ct1.get_group(2), s = item_ct1.get_group(1),
        c = item_ct1.get_group(0);
        GroupDesc d = desc[c]; if (s >= d.rows) return;
    size_t rb=row_bytes(6,D);
    const uint8_t *gr=(const uint8_t*)d.g+(size_t)o*rb;
    const uint8_t *ur=(const uint8_t*)d.u+(size_t)o*rb;
    const float *xs=x+(size_t)(d.offset+s)*D;float ga=0,ua=0;
    int nsub=(D+COLI_E8_SUB-1)/COLI_E8_SUB;
    for (int sb = item_ct1.get_local_id(2); sb < nsub;
         sb += item_ct1.get_local_range(2)) {
        int ib=sb%(COLI_E8_QK/COLI_E8_SUB);
        const uint8_t *gb=gr+(size_t)(sb/(COLI_E8_QK/COLI_E8_SUB))*COLI_E8_BBYTES;
        const uint8_t *ub=ur+(size_t)(sb/(COLI_E8_QK/COLI_E8_SUB))*COLI_E8_BBYTES;
        float gw[COLI_E8_SUB],uw[COLI_E8_SUB];
        e8_expand_sub_dev(gb, ib, e8_fp16(gb + 96), gw, c_e8_grid);
        e8_expand_sub_dev(ub, ib, e8_fp16(ub + 96), uw, c_e8_grid);
        int off=sb*COLI_E8_SUB,n=D-off<COLI_E8_SUB?D-off:COLI_E8_SUB;
        for(int k=0;k<n;k++){float v=xs[off+k];ga+=v*gw[k];ua+=v*uw[k];}
    }
    auto &gp = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[256]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &up = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[256]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
        gp[item_ct1.get_local_id(2)] = ga; up[item_ct1.get_local_id(2)] = ua;
        item_ct1.barrier();
    for (int n = 128; n; n >>= 1) {
        if (item_ct1.get_local_id(2) < n) {
            gp[item_ct1.get_local_id(2)] += gp[item_ct1.get_local_id(2) + n];
            up[item_ct1.get_local_id(2)] += up[item_ct1.get_local_id(2) + n];
        } item_ct1.barrier();
    }
    if (!item_ct1.get_local_id(2)) {
        float g = gp[0]; gate[(size_t)(d.offset + s) * I + o] =
                             (g / (1.f + sycl::native::exp(-g))) * up[0];
    }
}

static void
grouped_down_e8(float *y, const float *x, const GroupDesc *desc, int D, int I,
                const uint8_t *c_e8_grid) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int o = item_ct1.get_group(2), s = item_ct1.get_group(1),
        c = item_ct1.get_group(0);
        GroupDesc d = desc[c]; if (s >= d.rows) return;
    size_t rb=row_bytes(6,I);const uint8_t *wr=(const uint8_t*)d.d+(size_t)o*rb;
    const float *xs=x+(size_t)(d.offset+s)*I;float sum=0;
    int nsub=(I+COLI_E8_SUB-1)/COLI_E8_SUB;
    for (int sb = item_ct1.get_local_id(2); sb < nsub;
         sb += item_ct1.get_local_range(2)) {
        int ib=sb%(COLI_E8_QK/COLI_E8_SUB);
        const uint8_t *blk=wr+(size_t)(sb/(COLI_E8_QK/COLI_E8_SUB))*COLI_E8_BBYTES;
        float w[COLI_E8_SUB];
            e8_expand_sub_dev(blk, ib, e8_fp16(blk + 96), w, c_e8_grid);
        int off=sb*COLI_E8_SUB,n=I-off<COLI_E8_SUB?I-off:COLI_E8_SUB;
        for(int k=0;k<n;k++)sum+=xs[off+k]*w[k];
    }
    auto &p = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[256]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
        p[item_ct1.get_local_id(2)] = sum; item_ct1.barrier();
    for (int n = 128; n; n >>= 1) {
        if (item_ct1.get_local_id(2) < n) p[item_ct1.get_local_id(2)] +=
        p[item_ct1.get_local_id(2) + n];
        item_ct1.barrier();
    }
    if (!item_ct1.get_local_id(2)) y[(size_t)(d.offset + s) * D + o] = p[0];
}

static void unpack_s4(uint8_t v,float *lo,float *hi){
    int a=v&15,b=v>>4; *lo=(float)(a&8?a-16:a); *hi=(float)(b&8?b-16:b);
}

/* Exact low-row W4A32 path. It consumes each packed weight byte once instead
 * of routing both nibbles through weight_at(), preserving FP32 activations. */
static void grouped_hidden_w4(float *y, const float *x, const GroupDesc *desc,
                              int I, int D, int which) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int o = item_ct1.get_group(2), s = item_ct1.get_group(1),
        c = item_ct1.get_group(0);
        GroupDesc d = desc[c]; if (s >= d.rows) return;
    const uint8_t *w=(const uint8_t*)(which?d.u:d.g);const float *sc=which?d.us:d.gs;
    const uint8_t *row=w+(size_t)o*((D+1)/2);const float *xs=x+(size_t)(d.offset+s)*D;
    float sum = 0; for (int b = item_ct1.get_local_id(2); b < (D + 1) / 2;
                        b += item_ct1.get_local_range(2)) {
        float a, z; unpack_s4(row[b], &a, &z);
        int i=b*2;sum+=xs[i]*a;if(i+1<D)sum+=xs[i+1]*z;}
    auto &p = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[256]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
        p[item_ct1.get_local_id(2)] = sum; item_ct1.barrier();
    for (int n = 128; n; n >>= 1) {
        if (item_ct1.get_local_id(2) < n) p[item_ct1.get_local_id(2)] +=
        p[item_ct1.get_local_id(2) + n];
        item_ct1.barrier();
    }
    if (!item_ct1.get_local_id(2)) y[(size_t)(d.offset + s) * I + o] =
        p[0] * sc[o];
}

static void grouped_hidden_w4_dual(float *gate, float *up, const float *x,
                                   const GroupDesc *desc, int I, int D) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int o = item_ct1.get_group(2), s = item_ct1.get_group(1),
        c = item_ct1.get_group(0);
        GroupDesc d = desc[c]; if (s >= d.rows) return;
    const uint8_t *gr=(const uint8_t*)d.g+(size_t)o*((D+1)/2);
    const uint8_t *ur=(const uint8_t*)d.u+(size_t)o*((D+1)/2);
    const float *xs=x+(size_t)(d.offset+s)*D;float ga=0,ua=0;
    for (int b = item_ct1.get_local_id(2); b < (D + 1) / 2;
         b += item_ct1.get_local_range(2)) {
        float g0, g1, u0, u1; unpack_s4(gr[b], &g0, &g1);
        unpack_s4(ur[b], &u0, &u1);
        int i=b*2;ga+=xs[i]*g0;ua+=xs[i]*u0;if(i+1<D){ga+=xs[i+1]*g1;ua+=xs[i+1]*u1;}}
    auto &gp = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[256]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &upv =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[256]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
        gp[item_ct1.get_local_id(2)] = ga; upv[item_ct1.get_local_id(2)] = ua;
        item_ct1.barrier();
    for (int n = 128; n; n >>= 1) {
        if (item_ct1.get_local_id(2) < n) {
            gp[item_ct1.get_local_id(2)] += gp[item_ct1.get_local_id(2) + n];
            upv[item_ct1.get_local_id(2)] += upv[item_ct1.get_local_id(2) + n];
        } item_ct1.barrier();
    }
    /* Fused epilogue: silu(gate)*up lands here instead of a third kernel —
     * the exact silu_mul expression on the exact same inputs, so bit-identical,
     * and the up[] round-trip through global memory disappears. up stays a
     * param so the launch sites keep their signature. */
    if (!item_ct1.get_local_id(2)) {
        size_t z = (size_t)(d.offset + s) * I + o;
        float g=gp[0]*d.gs[o],u=upv[0]*d.us[o];
        gate[z]=(g/(1.0f+sycl::native::exp(-g)))*u;(void)up;}
}

static void grouped_down_w4(float *y, const float *x, const GroupDesc *desc,
                            int D, int I) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int o = item_ct1.get_group(2), s = item_ct1.get_group(1),
        c = item_ct1.get_group(0);
        GroupDesc d = desc[c]; if (s >= d.rows) return;
    const uint8_t *row=(const uint8_t*)d.d+(size_t)o*((I+1)/2);
    const float *xs=x+(size_t)(d.offset+s)*I;float sum=0;
    for (int b = item_ct1.get_local_id(2); b < (I + 1) / 2;
         b += item_ct1.get_local_range(2)) {
        float a, z; unpack_s4(row[b], &a, &z);
        int i=b*2;sum+=xs[i]*a;if(i+1<I)sum+=xs[i+1]*z;}
    auto &p = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[256]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
        p[item_ct1.get_local_id(2)] = sum; item_ct1.barrier();
    for (int n = 128; n; n >>= 1) {
        if (item_ct1.get_local_id(2) < n) p[item_ct1.get_local_id(2)] +=
        p[item_ct1.get_local_id(2) + n];
        item_ct1.barrier();
    }
    if (!item_ct1.get_local_id(2)) y[(size_t)(d.offset + s) * D + o] =
        p[0] * d.ds[o];
}

/* fmt=4 grouped-int4 variants (#334): identical structure to the w4 kernels,
 * but the scale varies along the input dimension — sc[o*ng + i/gs], applied
 * per element inside the accumulation (gs is even, so a packed byte never
 * straddles a group). gs<=0 degrades to per-row (ng=1), so mixed fmt2/fmt4
 * groups run correctly through this one kernel family. */
static void grouped_hidden_g4_dual(float *gate, float *up, const float *x,
                                   const GroupDesc *desc, int I, int D) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int o = item_ct1.get_group(2), s = item_ct1.get_group(1),
        c = item_ct1.get_group(0);
        GroupDesc d = desc[c]; if (s >= d.rows) return;
    const uint8_t *gr=(const uint8_t*)d.g+(size_t)o*((D+1)/2);
    const uint8_t *ur=(const uint8_t*)d.u+(size_t)o*((D+1)/2);
    int ggs=d.ggs>0?d.ggs:D, ugs=d.ugs>0?d.ugs:D;
    const float *gsc=d.gs+(size_t)o*(size_t)((D+ggs-1)/ggs);
    const float *usc=d.us+(size_t)o*(size_t)((D+ugs-1)/ugs);
    const float *xs=x+(size_t)(d.offset+s)*D;float ga=0,ua=0;
    for (int b = item_ct1.get_local_id(2); b < (D + 1) / 2;
         b += item_ct1.get_local_range(2)) {
        float g0, g1, u0, u1; unpack_s4(gr[b], &g0, &g1);
        unpack_s4(ur[b], &u0, &u1);
        int i=b*2;float gv=gsc[i/ggs],uv=usc[i/ugs];
        ga+=xs[i]*g0*gv;ua+=xs[i]*u0*uv;
        if(i+1<D){ga+=xs[i+1]*g1*gv;ua+=xs[i+1]*u1*uv;}}
    auto &gp = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[256]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &upv =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[256]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
        gp[item_ct1.get_local_id(2)] = ga; upv[item_ct1.get_local_id(2)] = ua;
        item_ct1.barrier();
    for (int n = 128; n; n >>= 1) {
        if (item_ct1.get_local_id(2) < n) {
            gp[item_ct1.get_local_id(2)] += gp[item_ct1.get_local_id(2) + n];
            upv[item_ct1.get_local_id(2)] += upv[item_ct1.get_local_id(2) + n];
        } item_ct1.barrier();
    }
    /* same epilogue fusion as the w4 dual above (per-group scales already
     * applied inside the accumulation, so silu runs on the raw sums) */
    if (!item_ct1.get_local_id(2)) {
        size_t z = (size_t)(d.offset + s) * I + o;
        float g=gp[0],u=upv[0];
        gate[z]=(g/(1.0f+sycl::native::exp(-g)))*u;(void)up;}
}
static void grouped_down_g4(float *y, const float *x, const GroupDesc *desc,
                            int D, int I) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int o = item_ct1.get_group(2), s = item_ct1.get_group(1),
        c = item_ct1.get_group(0);
        GroupDesc d = desc[c]; if (s >= d.rows) return;
    const uint8_t *row=(const uint8_t*)d.d+(size_t)o*((I+1)/2);
    int dgs=d.dgs>0?d.dgs:I;
    const float *dsc=d.ds+(size_t)o*(size_t)((I+dgs-1)/dgs);
    const float *xs=x+(size_t)(d.offset+s)*I;float sum=0;
    for (int b = item_ct1.get_local_id(2); b < (I + 1) / 2;
         b += item_ct1.get_local_range(2)) {
        float a, z; unpack_s4(row[b], &a, &z);
        int i=b*2;float sv=dsc[i/dgs];
        sum+=xs[i]*a*sv;if(i+1<I)sum+=xs[i+1]*z*sv;}
    auto &p = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[256]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
        p[item_ct1.get_local_id(2)] = sum; item_ct1.barrier();
    for (int n = 128; n; n >>= 1) {
        if (item_ct1.get_local_id(2) < n) p[item_ct1.get_local_id(2)] +=
        p[item_ct1.get_local_id(2) + n];
        item_ct1.barrier();
    }
    if (!item_ct1.get_local_id(2)) y[(size_t)(d.offset + s) * D + o] = p[0];
}

/* fmt=8 fp8-e4m3 variants: same structure as the g4 kernels, but one byte per
 * weight (decoded through c_e4m3) and the scale is per 128x128 BLOCK of the
 * member's [O,I] matrix — sc[(o/128)*ceil(I/128) + i/128]. The block edge is a
 * property of the format, so the geometry derives from the dims alone; these
 * kernels require every member to be fmt=8 (no ride-along: a per-row member's
 * scales are [O], which this indexing would read out of bounds). */
static void grouped_hidden_f8_dual(float *gate, float *up, const float *x,
                                   const GroupDesc *desc, int I, int D,
                                   float const *c_e4m3) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int o = item_ct1.get_group(2), s = item_ct1.get_group(1),
        c = item_ct1.get_group(0);
        GroupDesc d = desc[c]; if (s >= d.rows) return;
    const uint8_t *gr=(const uint8_t*)d.g+(size_t)o*D;
    const uint8_t *ur=(const uint8_t*)d.u+(size_t)o*D;
    int nblk=(D+127)>>7;
    const float *gsc=d.gs+(size_t)(o>>7)*nblk;
    const float *usc=d.us+(size_t)(o>>7)*nblk;
    const float *xs=x+(size_t)(d.offset+s)*D;float ga=0,ua=0;
    for (int i = item_ct1.get_local_id(2); i < D;
         i += item_ct1.get_local_range(2)) {
        float xv = xs[i]; int b = i >> 7;
        ga+=xv*c_e4m3[gr[i]]*gsc[b];ua+=xv*c_e4m3[ur[i]]*usc[b];}
    auto &gp = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[256]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &upv =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[256]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
        gp[item_ct1.get_local_id(2)] = ga; upv[item_ct1.get_local_id(2)] = ua;
        item_ct1.barrier();
    for (int n = 128; n; n >>= 1) {
        if (item_ct1.get_local_id(2) < n) {
            gp[item_ct1.get_local_id(2)] += gp[item_ct1.get_local_id(2) + n];
            upv[item_ct1.get_local_id(2)] += upv[item_ct1.get_local_id(2) + n];
        } item_ct1.barrier();
    }
    /* same fused epilogue as the w4/g4 duals: scales applied in the
     * accumulation, silu(gate)*up lands in gate[], up[] is never written */
    if (!item_ct1.get_local_id(2)) {
        size_t z = (size_t)(d.offset + s) * I + o;
        float g=gp[0],u=upv[0];
        gate[z]=(g/(1.0f+sycl::native::exp(-g)))*u;(void)up;}
}
static void grouped_down_f8(float *y, const float *x, const GroupDesc *desc,
                            int D, int I, float const *c_e4m3) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int o = item_ct1.get_group(2), s = item_ct1.get_group(1),
        c = item_ct1.get_group(0);
        GroupDesc d = desc[c]; if (s >= d.rows) return;
    const uint8_t *row=(const uint8_t*)d.d+(size_t)o*I;
    int nblk=(I+127)>>7;
    const float *dsc=d.ds+(size_t)(o>>7)*nblk;
    const float *xs=x+(size_t)(d.offset+s)*I;float sum=0;
    for (int i = item_ct1.get_local_id(2); i < I;
         i += item_ct1.get_local_range(2)) sum +=
        xs[i] * c_e4m3[row[i]] * dsc[i >> 7];
    auto &p = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[256]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
        p[item_ct1.get_local_id(2)] = sum; item_ct1.barrier();
    for (int n = 128; n; n >>= 1) {
        if (item_ct1.get_local_id(2) < n) p[item_ct1.get_local_id(2)] +=
        p[item_ct1.get_local_id(2) + n];
        item_ct1.barrier();
    }
    if (!item_ct1.get_local_id(2)) y[(size_t)(d.offset + s) * D + o] = p[0];
}

/* fmt=8 warp rework (COLI_CUDA_F8_WARP, default on): one WARP owns one
 * (expert,output-row) pair and walks the row block-major through f8w_block —
 * weights stream once per tile of up to 4 activation rows instead of once per
 * (o,s) 256-thread block, and the accumulation is the CPU reference's (f32
 * block partial, scale once per block, double across blocks) instead of the
 * old flat per-element-scale f32 sum. Same descriptor surface as the old
 * kernels; grid (ceil(O/8), count): 8 warps per 256-thread block, rows looped
 * in-kernel. The old kernels stay compiled and selectable (=0) as the field
 * escape hatch. */
template <int HW, int SG, bool CPU_ORDER=false, bool CPU_SILU=false, bool RAW=false>
static void grouped_hidden_f8w_dual(float *gate, float *up, const float *x,
                                    const GroupDesc *desc, int I, int D,
                                    float const *c_e4m3) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &slut = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[256]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    for (int i = item_ct1.get_local_id(2); i < 256;
         i += item_ct1.get_local_range(2)) slut[i] = c_e4m3[i];
    item_ct1.barrier();
    int warp = item_ct1.get_local_id(2) / SG,
        lane = item_ct1.get_local_id(2) & (SG - 1);
    int o = item_ct1.get_group(2) * (item_ct1.get_local_range(2) / SG) + warp,
        c = item_ct1.get_group(1);
        GroupDesc d = desc[c];
    if(o>=I)return;
    const uint8_t *gr=(const uint8_t*)d.g+(size_t)o*D;
    const uint8_t *ur=(const uint8_t*)d.u+(size_t)o*D;
    int nblk=(D+127)>>7;
    const float *gsc=d.gs+(size_t)(o>>7)*nblk;
    const float *usc=d.us+(size_t)(o>>7)*nblk;
    int vec=!(D&3)&&!(((size_t)gr|(size_t)ur)&3)&&!((size_t)x&15);
    for(int s0=0;s0<d.rows;s0+=4){
        int ns=d.rows-s0<4?d.rows-s0:4;
        const float *xs=x+(size_t)(d.offset+s0)*D;
        double ga[4]={0,0,0,0},ua[4]={0,0,0,0};
        for(int bi=0;bi<nblk;bi++){
            int base=bi<<7,len=D-base<128?D-base:128;
            float gp[4],up_[4];
            if constexpr(CPU_ORDER)f8cpu_block<1,SG>(slut,gr,ur,xs,D,ns,base,gp,up_);
            else f8w_block<HW,1,SG>(slut,gr,ur,xs,D,ns,base,len,vec,gp,up_);
            if(!lane){float sg=gsc[bi],su=usc[bi];
                for(int s=0;s<ns;s++){ga[s]+=(double)gp[s]*sg;ua[s]+=(double)up_[s]*su;}}
        }
        /* same fused epilogue as the old dual: silu(gate)*up lands in gate[],
         * up[] is never written */
        if(!lane)for(int s=0;s<ns;s++){
            float g=(float)ga[s],u=(float)ua[s];
            if constexpr(RAW) {
                gate[(size_t)(d.offset+s0+s)*I+o]=g;
                up[(size_t)(d.offset+s0+s)*I+o]=u;
            } else if constexpr(CPU_SILU)
                gate[(size_t)(d.offset+s0+s)*I+o]=cpu_form_silu_mul(g,u);
            else
                gate[(size_t)(d.offset + s0 + s) * I + o] =
                    (g / (1.f + sycl::native::exp(-g))) * u;
        }
    }
    (void)up;
}
template <int HW, int SG, bool CPU_ORDER=false>
static void grouped_down_f8w(float *y, const float *x, const GroupDesc *desc,
                             int D, int I, float const *c_e4m3) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &slut = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[256]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    for (int i = item_ct1.get_local_id(2); i < 256;
         i += item_ct1.get_local_range(2)) slut[i] = c_e4m3[i];
    item_ct1.barrier();
    int warp = item_ct1.get_local_id(2) / SG,
        lane = item_ct1.get_local_id(2) & (SG - 1);
    int o = item_ct1.get_group(2) * (item_ct1.get_local_range(2) / SG) + warp,
        c = item_ct1.get_group(1);
        GroupDesc d = desc[c];
    if(o>=D)return;
    const uint8_t *row=(const uint8_t*)d.d+(size_t)o*I;
    int nblk=(I+127)>>7;
    const float *dsc=d.ds+(size_t)(o>>7)*nblk;
    int vec=!(I&3)&&!((size_t)row&3)&&!((size_t)x&15);
    for(int s0=0;s0<d.rows;s0+=4){
        int ns=d.rows-s0<4?d.rows-s0:4;
        const float *xs=x+(size_t)(d.offset+s0)*I;
        double a[4]={0,0,0,0};
        for(int bi=0;bi<nblk;bi++){
            int base=bi<<7,len=I-base<128?I-base:128;
            float p[4];
            if constexpr(CPU_ORDER)f8cpu_block<0,SG>(slut,row,nullptr,xs,I,ns,base,p,nullptr);
            else f8w_block<HW,0,SG>(slut,row,nullptr,xs,I,ns,base,len,vec,p,nullptr);
            if(!lane){float sd=dsc[bi];
                for(int s=0;s<ns;s++)a[s]+=(double)p[s]*sd;}
        }
        if(!lane)for(int s=0;s<ns;s++)
            y[(size_t)(d.offset+s0+s)*D+o]=(float)a[s];
    }
}

/* COLI_SYCL_F8_CPU_ORDER=2: the CPU order's arithmetic (=1, f8cpu_block) with every lane at work. A sub-group's
 * lanes in groups of eight, a group one output row (32 rows a block of 256 at either width); lane c of a group is
 * the CPU's FMA chain c of a 128-block (elements c, c+8, ..., c+120, in that order), its 16 weights decoded once
 * for the rows of a pass (up to 4, as =1). The group's tree is f8cpu_block's shifts by 4, 2, 1, which at the
 * group's first lane read only the group's own eight lanes: hsum256's ((a0+a4)+(a2+a6))+((a1+a5)+(a3+a7)). That
 * lane sums the blocks in double in ascending order, the block scale applied once. Every (output,row) gets =1's
 * operations in =1's order, so =1's bits. D and I whole 128-blocks (the dispatch's CPU-order condition). */
template <int DUAL>
__dpct_inline__ void f8c2_block(const float *lut, const uint8_t *gr, const uint8_t *ur, const float *xs,
                                int stride, int ns, int base, int c, bool act, float *gp, float *up) {
    float gw[16], uw[16];
    if (act)
        for (int t = 0; t < 16; t++) {
            const int k = base + c + 8 * t;
            gw[t] = lut[gr[k]];
            if constexpr (DUAL) uw[t] = lut[ur[k]];
        }
    for (int s = 0; s < ns; s++) {
        float g = 0.f, u = 0.f;
        if (act) {
            const float *xr = xs + (size_t)s * stride + base + c;
            for (int t = 0; t < 16; t++) {
                const float xv = xr[8 * t];
                g = sycl::fma(gw[t], xv, g);
                if constexpr (DUAL) u = sycl::fma(uw[t], xv, u);
            }
        }
        for (int off = 4; off; off >>= 1) {   /* every lane of the sub-group, as the shuffles need */
            g += shfl_down(g, off);
            if constexpr (DUAL) u += shfl_down(u, off);
        }
        gp[s] = g;
        if constexpr (DUAL) up[s] = u;
    }
}
/* COLI_SYCL_F8_CPU_ORDER=3: =1's operations with a decode step's critical path cut. One sub-group an output row, as
 * =1; its lanes in G = SG/8 groups of eight, group k taking the row's blocks k, k+G, k+2G, ... at the same time; lane c
 * of a group is the CPU's FMA chain c of its block (16 weights decoded once for the rows of a pass), the tree
 * f8cpu_block's (at a group's first lane, only the group's lanes). Lane 0 takes an iteration's G block results from the
 * groups' first lanes (select_from_group, every lane taking part) and adds them to its double sum in ascending block
 * order, the block scale applied once: =1's sum in =1's order, a quarter of =1's serial walk at SG 32. */
template <int SG, int DUAL>
__dpct_inline__ void f8c3_row(const float *lut, const uint8_t *gr, const uint8_t *ur, const float *gsc,
                              const float *usc, const float *xs, int stride, int ns, int nblk, double *ga, double *ua) {
    constexpr int G = SG / 8;
    auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
    const int lane = (int)sg.get_local_linear_id(), grp = lane >> 3, c = lane & 7;
    for (int it = 0; it * G < nblk; it++) {
        const int bi = it * G + grp;
        const bool act = bi < nblk;
        const int base = (act ? bi : 0) << 7;
        float gw[16], uw[16];
        if (act)
            for (int t = 0; t < 16; t++) {
                const int k = base + c + 8 * t;
                gw[t] = lut[gr[k]];
                if constexpr (DUAL) uw[t] = lut[ur[k]];
            }
        for (int s = 0; s < ns; s++) {
            float g = 0.f, u = 0.f;
            if (act) {
                const float *xr = xs + (size_t)s * stride + base + c;
                for (int t = 0; t < 16; t++) {
                    const float xv = xr[8 * t];
                    g = sycl::fma(gw[t], xv, g);
                    if constexpr (DUAL) u = sycl::fma(uw[t], xv, u);
                }
            }
            for (int off = 4; off; off >>= 1) {
                g += shfl_down(g, off);
                if constexpr (DUAL) u += shfl_down(u, off);
            }
            for (int k = 0; k < G; k++) {   /* the iteration's blocks, ascending, into lane 0's double sums */
                const float pg = sycl::select_from_group(sg, g, k * 8);
                float pu = 0.f;
                if constexpr (DUAL) pu = sycl::select_from_group(sg, u, k * 8);
                const int b = it * G + k;
                if (lane == 0 && b < nblk) {
                    ga[s] += (double)pg * gsc[b];
                    if constexpr (DUAL) ua[s] += (double)pu * usc[b];
                }
            }
        }
    }
}
template <int SG, bool CPU_SILU=false, bool RAW=false>
static void grouped_hidden_f8c3_dual(float *gate, float *up, const float *x, const GroupDesc *desc,
                                     int I, int D, float const *c_e4m3) {
    static_assert(SG == 16 || SG == 32, "supported subgroup width");
    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto &slut = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[256]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    for (int i = item.get_local_id(2); i < 256; i += item.get_local_range(2)) slut[i] = c_e4m3[i];
    item.barrier();
    const int warp = item.get_local_id(2) / SG, lane = item.get_local_id(2) & (SG - 1);
    const int o = item.get_group(2) * (item.get_local_range(2) / SG) + warp;
    const GroupDesc d = desc[item.get_group(1)];
    if (o >= I) return;   /* a whole sub-group's row past the end: uniform */
    const uint8_t *gr = (const uint8_t *)d.g + (size_t)o * D, *ur = (const uint8_t *)d.u + (size_t)o * D;
    const int nblk = D >> 7;
    const float *gsc = d.gs + (size_t)(o >> 7) * nblk, *usc = d.us + (size_t)(o >> 7) * nblk;
    for (int s0 = 0; s0 < d.rows; s0 += 4) {
        const int ns = d.rows - s0 < 4 ? d.rows - s0 : 4;
        const float *xs = x + (size_t)(d.offset + s0) * D;
        double ga[4] = {0, 0, 0, 0}, ua[4] = {0, 0, 0, 0};
        f8c3_row<SG, 1>(slut, gr, ur, gsc, usc, xs, D, ns, nblk, ga, ua);
        if (!lane)   /* the epilogue of grouped_hidden_f8w_dual */
            for (int s = 0; s < ns; s++) {
                const float g = (float)ga[s], u = (float)ua[s];
                const size_t at = (size_t)(d.offset + s0 + s) * I + o;
                if constexpr (RAW) { gate[at] = g; up[at] = u; }
                else if constexpr (CPU_SILU) gate[at] = cpu_form_silu_mul(g, u);
                else gate[at] = (g / (1.f + sycl::native::exp(-g))) * u;
            }
    }
    (void)up;
}
template <int SG>
static void grouped_down_f8c3(float *y, const float *x, const GroupDesc *desc, int D, int I,
                              float const *c_e4m3) {
    static_assert(SG == 16 || SG == 32, "supported subgroup width");
    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto &slut = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[256]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    for (int i = item.get_local_id(2); i < 256; i += item.get_local_range(2)) slut[i] = c_e4m3[i];
    item.barrier();
    const int warp = item.get_local_id(2) / SG, lane = item.get_local_id(2) & (SG - 1);
    const int o = item.get_group(2) * (item.get_local_range(2) / SG) + warp;
    const GroupDesc d = desc[item.get_group(1)];
    if (o >= D) return;
    const uint8_t *row = (const uint8_t *)d.d + (size_t)o * I;
    const int nblk = I >> 7;
    const float *dsc = d.ds + (size_t)(o >> 7) * nblk;
    for (int s0 = 0; s0 < d.rows; s0 += 4) {
        const int ns = d.rows - s0 < 4 ? d.rows - s0 : 4;
        const float *xs = x + (size_t)(d.offset + s0) * I;
        double a[4] = {0, 0, 0, 0}, unused[4];
        f8c3_row<SG, 0>(slut, row, nullptr, dsc, nullptr, xs, I, ns, nblk, a, unused);
        if (!lane)
            for (int s = 0; s < ns; s++) y[(size_t)(d.offset + s0 + s) * D + o] = (float)a[s];
    }
}
template <int SG, bool CPU_SILU=false, bool RAW=false>
static void grouped_hidden_f8c2_dual(float *gate, float *up, const float *x, const GroupDesc *desc,
                                     int I, int D, float const *c_e4m3) {
    static_assert(SG == 16 || SG == 32, "supported subgroup width");
    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto &slut = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[256]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    for (int i = item.get_local_id(2); i < 256; i += item.get_local_range(2)) slut[i] = c_e4m3[i];
    item.barrier();
    const int tid = item.get_local_id(2), c = tid & 7, rows_wg = item.get_local_range(2) / 8;
    const int o = item.get_group(2) * rows_wg + (tid >> 3);
    const int o0 = item.get_group(2) * rows_wg + ((tid & ~(SG - 1)) >> 3);   /* the sub-group's first row */
    const GroupDesc d = desc[item.get_group(1)];
    if (o0 >= I) return;   /* the whole sub-group past the end: uniform */
    const bool act = o < I;
    const int oo = act ? o : 0, nblk = D >> 7;
    const uint8_t *gr = (const uint8_t *)d.g + (size_t)oo * D, *ur = (const uint8_t *)d.u + (size_t)oo * D;
    const float *gsc = d.gs + (size_t)(oo >> 7) * nblk, *usc = d.us + (size_t)(oo >> 7) * nblk;
    for (int s0 = 0; s0 < d.rows; s0 += 4) {
        const int ns = d.rows - s0 < 4 ? d.rows - s0 : 4;
        const float *xs = x + (size_t)(d.offset + s0) * D;
        double ga[4] = {0, 0, 0, 0}, ua[4] = {0, 0, 0, 0};
        for (int bi = 0; bi < nblk; bi++) {
            float gp[4], up_[4];
            f8c2_block<1>(slut, gr, ur, xs, D, ns, bi << 7, c, act, gp, up_);
            if (act && !c) {
                const float sg = gsc[bi], su = usc[bi];
                for (int s = 0; s < ns; s++) { ga[s] += (double)gp[s] * sg; ua[s] += (double)up_[s] * su; }
            }
        }
        if (act && !c)   /* the epilogue of grouped_hidden_f8w_dual */
            for (int s = 0; s < ns; s++) {
                const float g = (float)ga[s], u = (float)ua[s];
                const size_t at = (size_t)(d.offset + s0 + s) * I + o;
                if constexpr (RAW) { gate[at] = g; up[at] = u; }
                else if constexpr (CPU_SILU) gate[at] = cpu_form_silu_mul(g, u);
                else gate[at] = (g / (1.f + sycl::native::exp(-g))) * u;
            }
    }
    (void)up;
}
template <int SG>
static void grouped_down_f8c2(float *y, const float *x, const GroupDesc *desc, int D, int I,
                              float const *c_e4m3) {
    static_assert(SG == 16 || SG == 32, "supported subgroup width");
    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto &slut = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[256]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    for (int i = item.get_local_id(2); i < 256; i += item.get_local_range(2)) slut[i] = c_e4m3[i];
    item.barrier();
    const int tid = item.get_local_id(2), c = tid & 7, rows_wg = item.get_local_range(2) / 8;
    const int o = item.get_group(2) * rows_wg + (tid >> 3);
    const int o0 = item.get_group(2) * rows_wg + ((tid & ~(SG - 1)) >> 3);
    const GroupDesc d = desc[item.get_group(1)];
    if (o0 >= D) return;
    const bool act = o < D;
    const int oo = act ? o : 0, nblk = I >> 7;
    const uint8_t *row = (const uint8_t *)d.d + (size_t)oo * I;
    const float *dsc = d.ds + (size_t)(oo >> 7) * nblk;
    for (int s0 = 0; s0 < d.rows; s0 += 4) {
        const int ns = d.rows - s0 < 4 ? d.rows - s0 : 4;
        const float *xs = x + (size_t)(d.offset + s0) * I;
        double a[4] = {0, 0, 0, 0};
        for (int bi = 0; bi < nblk; bi++) {
            float p[4];
            f8c2_block<0>(slut, row, nullptr, xs, I, ns, bi << 7, c, act, p, nullptr);
            if (act && !c) { const float sd = dsc[bi]; for (int s = 0; s < ns; s++) a[s] += (double)p[s] * sd; }
        }
        if (act && !c)
            for (int s = 0; s < ns; s++) y[(size_t)(d.offset + s0 + s) * D + o] = (float)a[s];
    }
}

static void attention_absorb_kernel(float *ctx, const float *q,
                                    const float *latent, const float *rope,
                                    const void *weights, const float *wscale,
                                    int fmt, int H, int Q, int R, int V, int K,
                                    int T, float scale, int gs, int ng,
                                    uint8_t *dpct_local, float const *c_e4m3) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int h = item_ct1.get_group(2), tid = item_ct1.get_local_id(2),
        rbase = h * (Q + V);
        auto sm = (float *)dpct_local;
    float *qa=sm,*cl=qa+K,*scores=cl+K;
    for (int k = tid; k < K; k += item_ct1.get_local_range(2)) {
        float a = 0; for (int d = 0; d < Q; d++)
        a+=q[(size_t)h*(Q+R)+d]*weight_at(weights,fmt,(size_t)(rbase+d)*row_bytes(fmt,K),k, c_e4m3)*absorb_scale(wscale,fmt,gs,ng,rbase+d,k);qa[k]=a;}
    item_ct1.barrier(sycl::access::fence_space::local_space);
    for (int t = tid; t < T; t += item_ct1.get_local_range(2)) {
        float a = 0;
        const float *lt = latent + (size_t)t * K, *rt = rope + (size_t)t * R;
        for(int k=0;k<K;k++)a+=qa[k]*lt[k];for(int d=0;d<R;d++)a+=q[(size_t)h*(Q+R)+Q+d]*rt[d];scores[t]=a*scale;}
    item_ct1.barrier(sycl::access::fence_space::local_space);
    if (!tid) {
        float mx = scores[0];
        for (int t = 1; t < T; t++) mx = sycl::fmax(mx, scores[t]); float z = 0;
        for(int t=0;t<T;t++){scores[t]=sycl::native::exp(scores[t]-mx);z+=scores[t];}for(int t=0;t<T;t++)scores[t]/=z;}
    item_ct1.barrier(sycl::access::fence_space::local_space);
    for (int k = tid; k < K; k += item_ct1.get_local_range(2)) {
        float a = 0;
        for (int t = 0; t < T; t++) a += scores[t] * latent[(size_t)t * K + k];
        cl[k] = a;
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);
    for (int v = tid; v < V; v += item_ct1.get_local_range(2)) {
        int row = rbase + Q + v; float a = 0; size_t rb = row_bytes(fmt, K);
        for(int k=0;k<K;k++)a+=cl[k]*weight_at(weights,fmt,(size_t)row*rb,k, c_e4m3)*absorb_scale(wscale,fmt,gs,ng,row,k);ctx[(size_t)h*V+v]=a;}
}

static void attention_absorb_batch_kernel(
    float *ctx, const float *q, const float *latent, const float *rope,
    const void *weights, const float *wscale, int fmt, int S, int H, int Q,
    int R, int V, int K, int T, float scale, int gs, int ng,
    uint8_t *dpct_local, float const *c_e4m3) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int s = item_ct1.get_group(1), h = item_ct1.get_group(2),
        tid = item_ct1.get_local_id(2), nt = T - S + s + 1, rbase = h * (Q + V);
    if(s>=S||nt<1)return;
    auto sm = (float *)dpct_local;
        float *qa = sm, *cl = qa + K, *scores = cl + K, *red = scores + T;
    const float *qs=q+((size_t)s*H+h)*(Q+R);
    for (int k = tid; k < K; k += item_ct1.get_local_range(2)) {
        float a = 0; for (int d = 0; d < Q; d++)
        a+=qs[d]*weight_at(weights,fmt,(size_t)(rbase+d)*row_bytes(fmt,K),k, c_e4m3)*
          absorb_scale(wscale,fmt,gs,ng,rbase+d,k);qa[k]=a;}
    item_ct1.barrier(sycl::access::fence_space::local_space);
    for (int t = tid; t < nt; t += item_ct1.get_local_range(2)) {
        float a = 0; const float *lt = latent + (size_t)t * K;
        const float *rt=rope+(size_t)t*R;for(int k=0;k<K;k++)a+=qa[k]*lt[k];
        for(int d=0;d<R;d++)a+=qs[Q+d]*rt[d];scores[t]=a*scale;}
    item_ct1.barrier(sycl::access::fence_space::local_space);
    float local = -3.402823466e+38F;
        for (int t = tid; t < nt; t += item_ct1.get_local_range(2)) local =
            sycl::fmax(local, scores[t]);
    red[tid] = local; item_ct1.barrier(sycl::access::fence_space::local_space);
    for (int n = item_ct1.get_local_range(2) >> 1; n; n >>= 1) {
        if (tid < n) red[tid] = sycl::fmax(red[tid], red[tid + n]);
        item_ct1.barrier(sycl::access::fence_space::local_space);
    }
    float mx = red[0]; local = 0; for (int t = tid; t < nt;
                                       t += item_ct1.get_local_range(2)) {
        float e = sycl::native::exp(scores[t] - mx); scores[t] = e; local += e;
    }
    item_ct1.barrier(
        sycl::access::fence_space::local_space); /* every warp must read red[0]
                                                    above before red[] is reused
                                                    below */
    red[tid] = local; item_ct1.barrier(sycl::access::fence_space::local_space);
    for (int n = item_ct1.get_local_range(2) >> 1; n; n >>= 1) {
        if (tid < n) red[tid] += red[tid + n];
        item_ct1.barrier(sycl::access::fence_space::local_space);
    }
    float inv = 1.f / red[0];
        for (int t = tid; t < nt; t += item_ct1.get_local_range(2)) scores[t] *=
        inv;
    item_ct1.barrier(sycl::access::fence_space::local_space);
    for (int k = tid; k < K; k += item_ct1.get_local_range(2)) {
        float a = 0; for (int t = 0; t < nt; t++)
        a+=scores[t]*latent[(size_t)t*K+k];cl[k]=a;}
    item_ct1.barrier(sycl::access::fence_space::local_space);
    for (int v = tid; v < V; v += item_ct1.get_local_range(2)) {
        int row = rbase + Q + v; float a = 0; size_t rb = row_bytes(fmt, K);
        for (int k = 0; k < K; k++) a +=
            cl[k] * weight_at(weights, fmt, (size_t)row * rb, k, c_e4m3) *
            absorb_scale(wscale, fmt, gs, ng, row, k);
        ctx[((size_t)s*H+h)*V+v]=a;}
}

/* Independent device-resident KV sequence per row. lengths selects the valid
 * prefix; latent/rope point at paged caches updated by the host wrapper. */
static void attention_absorb_ragged_kernel(
    float *ctx, const float *q, const float *const *latent,
    const float *const *rope, const int *lengths, const void *weights,
    const float *wscale, int fmt, int S, int H, int Q, int R, int V, int K,
    int T, int page_stride, float scale, int gs, int ng, uint8_t *dpct_local,
    float const *c_e4m3) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int s = item_ct1.get_group(1), h = item_ct1.get_group(2),
        tid = item_ct1.get_local_id(2), nt = lengths[s], rbase = h * (Q + V);
    if(s>=S||nt<1||nt>T)return;
    auto sm = (float *)dpct_local;
        float *qa = sm, *cl = qa + K, *scores = cl + K, *red = scores + T;
    const float *qs=q+((size_t)s*H+h)*(Q+R);
    for (int k = tid; k < K; k += item_ct1.get_local_range(2)) {
        float a = 0; for (int d = 0; d < Q; d++)
        a+=qs[d]*weight_at(weights,fmt,(size_t)(rbase+d)*row_bytes(fmt,K),k, c_e4m3)*
          absorb_scale(wscale,fmt,gs,ng,rbase+d,k);qa[k]=a;}
    item_ct1.barrier();
    for (int t = tid; t < nt; t += item_ct1.get_local_range(2)) {
        float a = 0;
        int pg = t / COLI_KV_PAGE_TOKENS, pt = t % COLI_KV_PAGE_TOKENS;
        const float *lt=latent[(size_t)s*page_stride+pg]+(size_t)pt*K;
        const float *rt=rope[(size_t)s*page_stride+pg]+(size_t)pt*R;for(int k=0;k<K;k++)a+=qa[k]*lt[k];
        for(int d=0;d<R;d++)a+=qs[Q+d]*rt[d];scores[t]=a*scale;}
    item_ct1.barrier();
    float local = -3.402823466e+38F;
        for (int t = tid; t < nt; t += item_ct1.get_local_range(2)) local =
            sycl::fmax(local, scores[t]);
    red[tid] = local; item_ct1.barrier();
    for (int n = item_ct1.get_local_range(2) >> 1; n; n >>= 1) {
        if (tid < n) red[tid] = sycl::fmax(red[tid], red[tid + n]);
        item_ct1.barrier();
    }
    float mx = red[0]; local = 0; for (int t = tid; t < nt;
                                       t += item_ct1.get_local_range(2)) {
        float e = sycl::native::exp(scores[t] - mx); scores[t] = e; local += e;
    }
    item_ct1.barrier(); /* every warp must read red[0] above before red[] is
                           reused below */
    red[tid] = local; item_ct1.barrier();
    for (int n = item_ct1.get_local_range(2) >> 1; n; n >>= 1) {
        if (tid < n) red[tid] += red[tid + n]; item_ct1.barrier();
    }
    float inv = 1.f / red[0];
        for (int t = tid; t < nt; t += item_ct1.get_local_range(2)) scores[t] *=
        inv;
    item_ct1.barrier();
    for (int k = tid; k < K; k += item_ct1.get_local_range(2)) {
        float a = 0; for (int t = 0; t < nt; t++) {
        const float *lt=latent[(size_t)s*page_stride+t/COLI_KV_PAGE_TOKENS]+
                        (size_t)(t%COLI_KV_PAGE_TOKENS)*K;
        a+=scores[t]*lt[k];}cl[k]=a;}
    item_ct1.barrier();
    for (int v = tid; v < V; v += item_ct1.get_local_range(2)) {
        int row = rbase + Q + v; float a = 0; size_t rb = row_bytes(fmt, K);
        for (int k = 0; k < K; k++) a +=
            cl[k] * weight_at(weights, fmt, (size_t)row * rb, k, c_e4m3) *
            absorb_scale(wscale, fmt, gs, ng, row, k);
        ctx[((size_t)s*H+h)*V+v]=a;}
}

static void ragged_kv_append(float *const *latent,float *const *rope,
        const float *packed,const int *old_len,const int *add,const int *offset,
        int K,int R,int page_stride){
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int s = item_ct1.get_group(2), n = add[s], base = offset[s];
    for(int t=0;t<n;t++){
        int pos=old_len[s]+t,pg=pos/COLI_KV_PAGE_TOKENS,pt=pos%COLI_KV_PAGE_TOKENS;
        float *lp=latent[(size_t)s*page_stride+pg]+(size_t)pt*K;
        float *rp=rope[(size_t)s*page_stride+pg]+(size_t)pt*R;
        for (int k = item_ct1.get_local_id(2); k < K;
             k += item_ct1.get_local_range(2))
            lp[k] = packed[base + (size_t)t * K + k];
        for (int r = item_ct1.get_local_id(2); r < R;
             r += item_ct1.get_local_range(2))
            rp[r] = packed[base + (size_t)n * K + (size_t)t * R + r];
    }
}

static int reserve(float **ptr, size_t *cap, size_t bytes) try {
    if (*cap >= bytes) return 1;
    if (*ptr) sycl::free(*ptr, dq());
    *ptr = nullptr;
    *cap = 0;
    if (!cuda_ok(COLI_SYCL_ALLOC(*ptr, (float *)sycl::malloc_device(
                                      bytes, dq()), bytes),
                 "scratch allocation")) return 0;
    *cap = bytes;
    return 1;
}
catch (sycl::exception const &exc) { return sycl_caught(exc, "reserve"); }

static int reserve_bytes(void **ptr, size_t *cap, size_t bytes) try {
    if (*cap >= bytes) return 1;
        if (*ptr) sycl::free(*ptr, dq());
        *ptr = nullptr; *cap = 0;
    if (!cuda_ok(COLI_SYCL_ALLOC(*ptr, (void *)sycl::malloc_device(
                                      bytes, dq()), bytes),
                 "descriptor allocation")) return 0;
        *cap = bytes; return 1;
}
catch (sycl::exception const &exc) { return sycl_caught(exc, "reserve_bytes"); }

static int reserve_pinned(float **ptr, size_t *cap, size_t bytes) try {
    if (*cap >= bytes) return 1;
        if (*ptr) sycl::free(*ptr, dq()); *ptr = nullptr;
        *cap = 0;
    if (!cuda_ok(COLI_SYCL_ALLOC(*ptr, (float *)sycl::malloc_host(
                                      bytes, dq()), bytes),
                 "pinned staging allocation")) return 0;
        *cap = bytes; return 1;
}
catch (sycl::exception const &exc) { return sycl_caught(exc, "reserve_pinned"); }

static int prepare_group_weights(DeviceContext *,ColiCudaTensor *const *,
        ColiCudaTensor *const *,ColiCudaTensor *const *,int,GroupDesc *){return 1;}

/* Publish quant.h's E8 codebook to every configured device. __constant__ memory
 * is per-device, so this walks the contexts; the engine calls it once after init
 * rather than the backend carrying a second copy of the table that could drift
 * from the CPU decoder's (#452). Safe to call before any fmt=6 upload only. */
extern "C" int coli_cuda_e8_set_grid(const void *grid) try {
    if (!grid || g_nctx < 1) return 0;
    for (int i = 0; i < g_nctx; i++) {
        if (!select_ctx(&g_ctx[i])) return 0;
        if (!cuda_ok(COLI_SYCL_TRY(dq().memcpy(g_ctx[i].e8_grid, grid, 256 * 4).wait()),
                     "E8 codebook upload"))
            return 0;
    }
    return 1;
}
catch (sycl::exception const &exc) { return sycl_caught(exc, "coli_cuda_e8_set_grid"); }

/* Publish quant.h's E4M3_LUT the same way — one source of truth for the fmt=8
 * decode on CPU and GPU. Until this succeeds, fmt=8 uploads are refused. */
extern "C" int coli_cuda_fp8_set_lut(const float *lut) try {
    if (!lut || g_nctx < 1) return 0;
    for (int i = 0; i < g_nctx; i++) {
        if (!select_ctx(&g_ctx[i])) return 0;
        if (!cuda_ok(COLI_SYCL_TRY(dq().memcpy(g_ctx[i].e4m3, lut, 256 * sizeof(float)).wait()),
                     "e4m3 LUT upload"))
            return 0;
    }
    g_fp8_lut_ready = 1;
    return 1;
}
catch (sycl::exception const &exc) { return sycl_caught(exc, "coli_cuda_fp8_set_lut"); }

extern "C" int coli_cuda_init(const int *devices, int count) try {
    if (!devices || count < 1 || count > COLI_CUDA_MAX_DEVICES) return 0;
    int available = coli_sycl_available_device_count();   /* the GPUs, in PCI order (coli_sycl.h) */
    /* Validate the whole list before creating resources or replacing state. */
    for (int i = 0; i < count; i++) {
        if (devices[i] < 0 || devices[i] >= available) {
            std::fprintf(stderr, "[CUDA] invalid device %d (available: 0..%d)\n", devices[i], available - 1);
            return 0;
        }
        for (int j = 0; j < i; j++) if (devices[j] == devices[i]) {
            std::fprintf(stderr, "[CUDA] duplicate device %d\n", devices[i]);
            return 0;
        }
    }
    if (g_nctx) {
        /* Same decision as before, routed through the shared predicate in
         * backend_cuda.h so a host-side test can pin it without nvcc; the
         * return value is unchanged (1 for the same set, 0 otherwise). */
        int live[COLI_CUDA_MAX_DEVICES];
        for (int i = 0; i < g_nctx; i++) live[i] = g_ctx[i].device;
        int d = coli_cuda_init_disposition(g_nctx, count, devices, live);
        if (d == COLI_CUDA_INIT_REFUSE)
            std::fprintf(stderr, "[CUDA] device list change requires shutdown first\n");
        return d == COLI_CUDA_INIT_ACCEPT;
    }
    /* the devices: a context, two in-order queues and the profile each, read at run time
     * (coli_sycl_profile.cpp), under the engine's ordinals */
    if (!coli_sycl_open(devices, count)) return 0;
    g_sycl_lost = 0;
    coli_sycl_ledger_add(backend_booked_bytes);
    for (int i = 0; i < count; i++) {
        int device = devices[i];
        DeviceContext *ctx = &g_ctx[g_nctx];
        *ctx = {};
        ctx->device = device;
        if (!select_ctx(ctx)) { coli_cuda_shutdown(); return 0; }
        const coli_sycl_profile &p = coli_sycl_profile_for(device);
        /* no compute capability: every >= 7 / 8 gate of the CUDA code (its tensor-core paths) stays
         * off; the matrix engines are gated by the profile's matrix_combinations query instead */
        ctx->compute_major = ctx->compute_minor = 0;
        ctx->sms = p.eu_count >= 8 ? p.eu_count / 8 : 1;   /* Xe-cores of 8 EUs: what the grid heuristics tuned to SMs count */
        ctx->smem_optin = p.slm_bytes ? p.slm_bytes : 48 * 1024;   /* the device's SLM a work-group: no opt-in exists */
        ctx->sg = p.sg_default == 32 ? 32 : 16;
        ctx->wpb = 256 / ctx->sg;
        ctx->has_fp64 = coli_sycl_device(device).has(sycl::aspect::fp64);
        ctx->stream = &coli_sycl_queue(device);
        g_nctx++;
        if (!cuda_ok(COLI_SYCL_ALLOC(ctx->e8_grid, sycl::malloc_device<uint8_t>(256 * 4, dq()), 256 * 4), "E8 codebook table") ||
            !cuda_ok(COLI_SYCL_ALLOC(ctx->e4m3, sycl::malloc_device<float>(256, dq()), 256), "e4m3 table") ||
            !cuda_ok(COLI_SYCL_TRY(dq().memset(ctx->e8_grid, 0, 256 * 4).wait()), "E8 codebook table zero") ||
            !cuda_ok(COLI_SYCL_TRY(dq().memset(ctx->e4m3, 0, 256 * sizeof(float)).wait()), "e4m3 table zero")) {
            coli_cuda_shutdown(); return 0;
        }
        /* the line the engine and its tools look for ("[CUDA] device N:", tools/efficiency.py; no
         * compute capability: sm_00), then the profile's: the architecture, the sub-group the warp
         * kernels run at and the sizes offered, the EUs, the SLM a work-group may take, the matrix
         * engines and their shapes, the profile's name (COLI_SYCL_PROFILE, COLI_SYCL_SG, COLI_SYCL_XMX) */
        std::fprintf(stderr, "[CUDA] device %d: %s, %.1f GB VRAM, sm_%d%d\n",
                     device, p.device_name, p.global_mem_bytes / 1e9, ctx->compute_major, ctx->compute_minor);
        char sizes[64] = "", shapes[80] = "";
        for (int k = 0; k < p.sg_count; k++)
            std::snprintf(sizes + strlen(sizes), sizeof sizes - strlen(sizes), "%s%d", k ? "," : "", p.sg_sizes[k]);
        if (p.has_xmx) {
            static const struct { unsigned f; const char *s; } sh[] = {
                {COLI_SYCL_XMX_F16, "f16"}, {COLI_SYCL_XMX_BF16, "bf16"}, {COLI_SYCL_XMX_INT8, "int8"},
                {COLI_SYCL_XMX_TF32, "tf32"}, {COLI_SYCL_XMX_F16_16, "f16x16"}};
            for (size_t k = 0; k < sizeof sh / sizeof sh[0]; k++) if (p.xmx_shapes & sh[k].f)
                std::snprintf(shapes + strlen(shapes), sizeof shapes - strlen(shapes), "%s%s", *shapes ? "," : ":", sh[k].s);
        }
        std::fprintf(stderr, "[SYCL] device %d: %s arch=%s sg=%d/%s eu=%d slm=%zuKB xmx=%s%s mem=%.1fGB profile=%s\n",
                     device, p.device_name, coli_sycl_arch_name(p.arch), ctx->sg, *sizes ? sizes : "?", p.eu_count,
                     p.slm_bytes / 1024, p.has_xmx ? "yes" : "no", shapes, p.global_mem_bytes / 1e9, p.name);
    }
    return 1;
}
catch (sycl::exception const &exc) { return sycl_caught(exc, "coli_cuda_init"); }

extern "C" int coli_cuda_available_device_count(void) {
    return coli_sycl_available_device_count();
}

extern "C" void coli_cuda_shutdown(void) {
    for (int i = 0; i < g_nctx; i++) {
        DeviceContext *ctx = &g_ctx[i];
        if (!select_ctx(ctx)) continue;
        /* cudaFree synchronised the device before releasing; sycl::free releases at once, and
         * the engine reaches here with a group possibly in flight (qwen36_tier's shutdown with a
         * group open) and the uploader's last copy queued, writing host_y and reading the
         * staging: both queues drained before anything they touch goes (the chain's shutdown,
         * before this by atexit's order, does the same) */
        try { coli_sycl_queue(ctx->device).wait(); coli_sycl_copy_queue(ctx->device).wait(); } catch (...) {}
        if (ctx->bdtab) sycl::free(ctx->bdtab, dq());
        if (ctx->mxfp4_weights)
            sycl::free(ctx->mxfp4_weights, dq());
        if (ctx->mxfp4_scales)
            sycl::free(ctx->mxfp4_scales, dq());
        if (ctx->mxfp4_expert_weights) sycl::free(
            ctx->mxfp4_expert_weights, dq());
        if (ctx->mxfp4_expert_scales) sycl::free(
            ctx->mxfp4_expert_scales, dq());
        if (ctx->x) sycl::free(ctx->x, dq());
        if (ctx->y) sycl::free(ctx->y, dq());
        if (ctx->dx) sycl::free(ctx->dx, dq());
        if (ctx->dy) sycl::free(ctx->dy, dq());
        if (ctx->gate) sycl::free(ctx->gate, dq());
        if (ctx->up) sycl::free(ctx->up, dq());
        if (ctx->qx) sycl::free(ctx->qx, dq());
        if (ctx->qscale)
            sycl::free(ctx->qscale, dq());
        if (ctx->aq) sycl::free(ctx->aq, dq());
            if (ctx->al) sycl::free(ctx->al, dq());
            if (ctx->ar) sycl::free(ctx->ar, dq());
            if (ctx->ac) sycl::free(ctx->ac, dq());
        for (int b = 0; b < 27; b++) if (ctx->pipe_buf[b])
            sycl::free(ctx->pipe_buf[b], dq());
        if (ctx->host_x) sycl::free(ctx->host_x, dq());
        if (ctx->host_y) sycl::free(ctx->host_y, dq());
        if (ctx->host_kv) sycl::free(ctx->host_kv, dq());
        if (ctx->host_gate) sycl::free(ctx->host_gate, dq());
        if (ctx->host_up) sycl::free(ctx->host_up, dq());
        if (ctx->host_desc) sycl::free(ctx->host_desc, dq());
        if (ctx->e8_grid) sycl::free(ctx->e8_grid, dq());
        if (ctx->e4m3) sycl::free(ctx->e4m3, dq());
        ctx->e8_grid = nullptr; ctx->e4m3 = nullptr;
        if (ctx->group_desc)
            sycl::free(ctx->group_desc, dq());
        if (ctx->moe_tab)
            sycl::free(ctx->moe_tab, dq());
        if (ctx->moe_host)
            sycl::free(ctx->moe_host, dq());
        ctx->mxfp4_weights = ctx->mxfp4_scales = nullptr;
        ctx->mxfp4_weights_cap = ctx->mxfp4_scales_cap = 0;
        ctx->mxfp4_expert_weights = ctx->mxfp4_expert_scales = nullptr;
        ctx->mxfp4_expert_weights_cap = ctx->mxfp4_expert_scales_cap = 0;
        ctx->x = ctx->y = ctx->gate = ctx->up = nullptr;
        ctx->dx = ctx->dy = nullptr; ctx->dx_cap = ctx->dy_cap = 0;
        ctx->qx=nullptr; ctx->qscale=nullptr;
        ctx->aq=ctx->al=ctx->ar=ctx->ac=nullptr;
        ctx->host_x = ctx->host_y = ctx->host_kv = nullptr;
        ctx->host_gate = ctx->host_up = nullptr;
        ctx->host_desc = nullptr; ctx->host_desc_cap = 0;
            ctx->stream = nullptr;   /* the queue is coli_sycl_open's, closed below */
        ctx->x_cap = ctx->y_cap = ctx->gate_cap = ctx->up_cap = 0;
        ctx->qx_cap=ctx->qscale_cap=0;
        ctx->aq_cap=ctx->al_cap=ctx->ar_cap=ctx->ac_cap=0;
        ctx->host_x_cap=ctx->host_y_cap=ctx->host_kv_cap=0;
        ctx->host_gate_cap=ctx->host_up_cap=0;
        ctx->group_desc=nullptr; ctx->group_desc_cap=0;
        ctx->moe_tab=nullptr; ctx->moe_tab_cap=0; ctx->moe_host=nullptr; ctx->moe_host_cap=0;
    }
    g_nctx = 0;
    g_current_device = -1;
    coli_sycl_close();   /* the contexts and queues (the chain's too: its shutdown runs before this, atexit's order) */
    /* g_fp8_lut_ready is PROCESS-WIDE while the e4m3 table (c_e4m3, a
     * __constant__ device symbol whose lifetime is the CUDA primary context,
     * not this file's host-side DeviceContext structs) is PER-DEVICE. A later
     * coli_cuda_init may select a device the previous span never published to;
     * without this reset the upload gate (g_fp8_lut_ready, checked in
     * coli_cuda_tensor_upload) would still be satisfied from the PREVIOUS
     * boot and admit fmt=8 tensors whose kernels there decode against an
     * unwritten (zero) table: silent all-zero weights, the exact
     * fabricated-numbers failure mode the format gates exist to refuse.
     * Reset so every boot must publish its own LUT (coli_cuda_fp8_set_lut)
     * before any fmt=8 upload. Shutdown is the ONLY site that needs to clear
     * the flag: coli_cuda_init refuses a re-init that names a different
     * device set (it returns early while g_nctx is non-zero, leaving the
     * existing contexts and their published table untouched), so the device
     * set can only WIDEN by passing through here first. */
    g_fp8_lut_ready = 0;
}

extern "C" int coli_cuda_device_count(void) { return g_nctx; }

extern "C" int coli_cuda_device_at(int index) {
    return index >= 0 && index < g_nctx ? g_ctx[index].device : -1;
}

extern "C" int coli_cuda_mem_info(int device, size_t *free_bytes,
                                  size_t *total_bytes) try {
    DeviceContext *ctx = find_ctx(device);
    if (!free_bytes || !total_bytes || !select_ctx(ctx)) return 0;
    return cuda_ok(!dev_mem_info(device, free_bytes, total_bytes), "memory info");
}
catch (sycl::exception const &exc) { return sycl_caught(exc, "coli_cuda_mem_info"); }

/* #653: 1 when the device shares physical memory with the host (Grace-Blackwell /
 * GB10, Jetson, integrated GPUs). On these the expert tier and the RAM cache draw
 * from the same pool, so the RAM budget must account for the tier; on a discrete GPU
 * VRAM is a separate pool and this returns 0. */
extern "C" int coli_cuda_device_integrated(int device) {
    /* the profile's host_unified_memory (lnl, ptl, arl, mtl); 0 for a device that is not open
     * (the engine asks after coli_cuda_init) */
    return coli_sycl_profile_for(device).is_integrated ? 1 : 0;
}

extern "C" void coli_cuda_stats(int device, size_t *tensor_count, size_t *tensor_bytes) {
    size_t count = 0, bytes = 0;
    for (int i = 0; i < g_nctx; i++) if (device < 0 || g_ctx[i].device == device) {
        count += g_ctx[i].tensor_count;
        bytes += g_ctx[i].tensor_bytes;
    }
    if (tensor_count) *tensor_count = count;
    if (tensor_bytes) *tensor_bytes = bytes;
}

extern "C" void coli_cuda_group_stats(uint64_t *calls, uint64_t *experts, uint64_t *rows,
                                        double *h2d_ms, double *kernel_ms, double *d2h_ms) {
    if(calls) *calls=g_group_calls; if(experts) *experts=g_group_experts; if(rows) *rows=g_group_rows;
    if(h2d_ms) *h2d_ms=g_group_h2d_ms; if(kernel_ms) *kernel_ms=g_group_kernel_ms;
    if(d2h_ms) *d2h_ms=g_group_d2h_ms;
}

extern "C" void coli_cuda_group_stats_device(
    int device, uint64_t *calls, uint64_t *experts, uint64_t *rows,
    double *h2d_ms, double *kernel_ms, double *d2h_ms) {
    std::lock_guard<std::mutex> lock(g_group_stats_mu);
    int index=-1;
    for(int i=0;i<g_nctx;i++) if(g_ctx[i].device==device){ index=i; break; }
    if(calls) *calls=index<0?0:g_device_group_calls[index];
    if(experts) *experts=index<0?0:g_device_group_experts[index];
    if(rows) *rows=index<0?0:g_device_group_rows[index];
    if(h2d_ms) *h2d_ms=index<0?0:g_device_group_h2d_ms[index];
    if(kernel_ms) *kernel_ms=index<0?0:g_device_group_kernel_ms[index];
    if(d2h_ms) *d2h_ms=index<0?0:g_device_group_d2h_ms[index];
}

/* ---- a GGUF's block formats (cuda_bd.cuh): fmt 100 + ggml type, the rows as the file
 * stores them, each padded to whole words; no scale array (the scales are in the blocks).
 * The decoders' table (blockdec_host.h) goes to a device with its first such tensor. */
static std::mutex g_bdtab_mx;
static const bd_u32 *bd_table(DeviceContext *ctx) try {
    std::lock_guard<std::mutex> lk(g_bdtab_mx);
    if (ctx->bdtab) return ctx->bdtab;
    std::vector<uint32_t> t(BDT_WORDS);
    bdtab_fill(t.data(), NULL);
    bd_u32 *d = NULL;
    if (!cuda_ok(COLI_SYCL_ALLOC(d, (bd_u32 *)sycl::malloc_device(
                         (size_t)BDT_WORDS * 4, dq()), (size_t)BDT_WORDS * 4),
                 "block table allocation")) return NULL;
    if (!cuda_ok(
            COLI_SYCL_TRY(dq()
                                 .memcpy(d, t.data(), (size_t)BDT_WORDS * 4)
                                 .wait()),
            "block table upload")) {
        sycl::free(d, dq()); return NULL;
    }
    ctx->bdtab = d;
    return d;
}
catch (sycl::exception const &exc) { sycl_caught(exc, "bd_table"); return NULL; }
/* the file's bytes of a row of I inputs, 0 when the format or the width is not one cuda_bd.cuh takes */
static size_t bd_file_row_bytes(int fmt, int I) {
    if (!bd_fmt_ok(fmt) || I < 32 || I % 32 || I % BD_BLK[fmt - 100]) return 0;
    return (size_t)I / BD_BLK[fmt - 100] * BD_BYTES[fmt - 100];
}
/* A resident tensor's copies done on the device before the caller may use it: a cudaMemcpy
 * from pageable host memory returns once the bytes are staged, its DMA (and a conversion
 * kernel after it) still on the legacy stream, which the chain's non-blocking stream does
 * not wait for. A kernel there read a just-uploaded f32 matrix's later rows unwritten. */
static int upload_settled(const char *what) try {
    return cuda_ok(COLI_SYCL_TRY(dq().wait()), what);
}
catch (sycl::exception const &exc) { return sycl_caught(exc, "upload_settled"); }
/* O rows of rb bytes from the host into rows of row_words words; zero_pad: the padding zeroed
 * first (a new tensor's; a refill keeps what the upload wrote there) */
static int bd_rows_copy(ColiCudaTensor *t, const void *weights,
                        int zero_pad) try {
    size_t rb = bd_file_row_bytes(t->fmt, t->I), pitch = (size_t)t->row_words * 4;
    if (pitch == rb) return cuda_ok(
        COLI_SYCL_TRY(dq()
                             .memcpy(t->weights, weights, t->weight_bytes)
                             .wait()),
        "block tensor upload") &&
        upload_settled("block tensor upload");
    return (!zero_pad ||
            cuda_ok(COLI_SYCL_TRY(dq()
                                         .memset(t->weights, 0, t->weight_bytes)
                                         .wait()),
                    "block tensor padding")) &&
           cuda_ok(COLI_SYCL_TRY(dq().ext_oneapi_memcpy2d(t->weights, pitch, weights, rb, rb, (size_t)t->O).wait()),
                   "block tensor upload") &&
           upload_settled("block tensor upload");
}
catch (sycl::exception const &exc) { return sycl_caught(exc, "bd_rows_copy"); }
static int bd_tensor_upload(ColiCudaTensor **tensor, const void *weights,
                            int fmt, int I, int O, int device) try {
    DeviceContext *ctx = find_ctx(device);
    size_t rb = bd_file_row_bytes(fmt, I);
    if (!weights || !rb || O < 1 || !select_ctx(ctx) || !bd_table(ctx)) return 0;
    ColiCudaTensor *t = static_cast<ColiCudaTensor *>(std::calloc(1, sizeof(*t)));
    if (!t) return 0;
    t->fmt = fmt; t->I = I; t->O = O; t->device = device; t->ng = 1;
    t->row_words = (int)((rb + 3) / 4);
    t->weight_bytes = (size_t)t->row_words * 4 * (size_t)O;
    if (!cuda_ok(
            COLI_SYCL_ALLOC(t->weights, (void *)sycl::malloc_device(
                                 t->weight_bytes, dq()), t->weight_bytes),
            "tensor allocation")) {
        coli_cuda_tensor_free(t); return 0;
    }
    t->weights_owned = 1;
    if (!bd_rows_copy(t, weights, 1)) { coli_cuda_tensor_free(t); return 0; }
    t->tracked = 1;
    ctx->tensor_count++;
    ctx->tensor_bytes += t->weight_bytes;
    *tensor = t;
    return 1;
}
catch (sycl::exception const &exc) { return sycl_caught(exc, "bd_tensor_upload"); }
/* y[S, O] = x[S, I] W^T for a block-format tensor, on stream st (device pointers): every row count
 * on the GEMV kernels, each row a decode step's bits (a prompt chunk's rows went through the
 * tensor cores' f16 tiles, hcg's and hcm's; the matrix engines' family takes the tier's int8 and
 * int4 rows alone, so a GGUF's prompt rows stay on the GEMV here) */
#define BD_GEMV_MAX_S 8
static int bd_tensor_gemv(DeviceContext *ctx, ColiCudaTensor *t, float *y,
                          const float *x, int S, sycl::queue * st) {
    const bd_u32 *tab = bd_table(ctx);
    if (!tab || ((uintptr_t)x & 15)) return 0;
    /* the float form for every type and every row count: COLI_CUDA_BD_FLOAT=1 (tests), or
     * GQ_EXACT=1, where the CPU's GGUF matmuls are the float reference (as cu_xb.cu's) */
    static int exact = -1;
    if (exact < 0) {
        const char *e = std::getenv("COLI_CUDA_BD_FLOAT"), *q = std::getenv("GQ_EXACT");
        exact = (e && *e == '1') || (q && *q == '1');
    }
    if (!bd_gemv_launch(y, x, (const bd_u32 *)t->weights, tab, t->fmt, t->I, t->O, t->row_words, S, ctx->smem_optin, exact, ctx->sg, st)) return 0;
    return cuda_ok(0, "block gemv launch");
}
/* a block-format tensor in device memory the caller owns (an arena's slot of cap bytes):
 * coli_cuda_tensor_free leaves the memory, and its bytes are counted where it was allocated */
extern "C" int coli_cuda_tensor_upload_at(ColiCudaTensor **tensor, void *dev, size_t cap, const void *weights,
                                          int fmt, int I, int O, int device) {
    if (!tensor || *tensor || !dev) return 0;
    DeviceContext *ctx = find_ctx(device);
    size_t rb = bd_file_row_bytes(fmt, I);
    if (!weights || !rb || O < 1 || !select_ctx(ctx) || !bd_table(ctx)) return 0;
    size_t rw = (rb + 3) / 4;
    if (rw * 4 * (size_t)O > cap) return 0;
    ColiCudaTensor *t = static_cast<ColiCudaTensor *>(std::calloc(1, sizeof(*t)));
    if (!t) return 0;
    t->fmt = fmt; t->I = I; t->O = O; t->device = device; t->ng = 1;
    t->row_words = (int)rw; t->weight_bytes = rw * 4 * (size_t)O;
    t->weights = dev; t->weights_owned = 0;
    if (!bd_rows_copy(t, weights, 1)) { std::free(t); return 0; }
    *tensor = t;
    return 1;
}
/* Page-locked host memory on the device's context (cudaHostAlloc's place; host USM): a copy from it to
 * the device is the device's own. A copy from pageable memory passes through a copy of the bytes on the
 * calling thread first: the tier's warm start spent 2.58 s of host time in 62 119 copy calls for 18.7 GB,
 * about 7 GB/s whatever a call's size (the campaign's H4). NULL when the device refuses it. */
extern "C" void *coli_cuda_host_alloc(int device, size_t bytes) try {
    DeviceContext *ctx = find_ctx(device);
    if (!bytes || !select_ctx(ctx)) return NULL;
    void *p = NULL;
    if (!cuda_ok(COLI_SYCL_ALLOC(p, (void *)sycl::malloc_host(bytes, dq()), bytes), "host staging allocation")) return NULL;
    return p;
}
catch (sycl::exception const &exc) { sycl_caught(exc, "coli_cuda_host_alloc"); return NULL; }
extern "C" void coli_cuda_host_free(int device, void *p) {
    DeviceContext *ctx = find_ctx(device);
    if (!p || !select_ctx(ctx)) return;
    sycl::free(p, dq());
}
/* An expert of the tier's native int4 format in device memory the caller owns (an arena's slot): the
 * host's nw bytes of packed nibbles, its matrices back to back as the staging holds them, to dev_w in
 * ONE copy; the ns floats of their grouped scales to dev_s in one more; the nibbles made signed by one
 * launch over the whole range; and one wait for the three. coli_cuda_tensor_upload_g takes two device
 * allocations, two waited copies, a launch and a wait for each of the expert's three matrices: a warm
 * start of 10 240 experts was 61 916 allocations and 62 000 copies of 512 and 64 KiB, each waited
 * for, on a link that moves a 25 MB copy at 43 GB/s and those at 9 (MIGRATION, the campaign's H4).
 * The queue is in order (one a device), so the launch reads what the copy before it wrote and the
 * last event's end is everyone's. The tensors over the slot come from coli_cuda_tensor_at_g. */
extern "C" int coli_cuda_slot_fill_s4(int device, void *dev_w, const void *w, size_t nw,
                                      float *dev_s, const float *s, size_t ns) try {
    DeviceContext *ctx = find_ctx(device);
    if (!dev_w || !w || !nw || !dev_s || !s || !ns || !select_ctx(ctx)) return 0;
    sycl::queue &q = dq();
    q.memcpy(dev_w, w, nw);
    uint8_t *nib = (uint8_t *)dev_w; const size_t n = nw;
    q.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, (unsigned)((n + 255) / 256)) * sycl::range<3>(1, 1, 256),
                                     sycl::range<3>(1, 1, 256)),
                   [=](sycl::nd_item<3>) { offset_to_signed_s4(nib, n); });
    q.memcpy(dev_s, s, ns * sizeof(float)).wait_and_throw();
    return 1;
}
catch (sycl::exception const &exc) { return sycl_caught(exc, "coli_cuda_slot_fill_s4"); }
/* a tensor of that format (fmt 4, gs elements a scale group) over device memory the caller owns and
 * has filled: no allocation and no copy here; coli_cuda_tensor_free leaves both ranges, and their
 * bytes are counted where they were allocated (as coli_cuda_tensor_upload_at's) */
extern "C" int coli_cuda_tensor_at_g(ColiCudaTensor **tensor, void *dev_w, float *dev_s,
                                     int I, int O, int device, int gs) {
    if (!tensor || *tensor || !dev_w || !dev_s || I < 1 || O < 1 || gs < 1) return 0;
    DeviceContext *ctx = find_ctx(device);
    size_t rb = row_bytes(4, I);
    if (!rb || !select_ctx(ctx)) return 0;
    ColiCudaTensor *t = static_cast<ColiCudaTensor *>(std::calloc(1, sizeof(*t)));
    if (!t) return 0;
    t->fmt = 4; t->I = I; t->O = O; t->device = device; t->weight_bytes = rb * (size_t)O;
    t->gs = gs; t->ng = (I + gs - 1) / gs; t->scale_count = (size_t)O * (size_t)t->ng;
    t->weights = dev_w; t->weights_owned = 0;
    t->scales = dev_s; t->scales_borrowed = 1;
    *tensor = t;
    return 1;
}
/* A matrix of the fp8-e4m3 format over device memory the caller owns and has filled (an arena's slot:
 * gate | up | down | the three block-scale tables, one coli_cuda_pipe_upload). The fields are the ones
 * coli_cuda_tensor_upload(fmt 8) sets, so the group kernels' descriptors (GroupDesc: weights, scales)
 * address the slot as they address an allocation a matrix. No allocation, no copy; coli_cuda_tensor_free
 * leaves both ranges (weights_owned 0, scales_borrowed 1); not tracked. */
extern "C" int coli_cuda_tensor_at_f8(ColiCudaTensor **tensor, void *dev_w, float *dev_s,
                                      int I, int O, int device) {
    if (!tensor || *tensor || !dev_w || !dev_s || I < 1 || O < 1) return 0;
    if (((uintptr_t)dev_w & 3) || ((uintptr_t)dev_s & 3)) return 0;   /* the word loads and the floats */
    if (!coli_cuda_fp8_gate_admits(8, g_fp8_lut_ready)) return 0;
    DeviceContext *ctx = find_ctx(device);
    size_t rb = row_bytes(8, I);
    if (!rb || !select_ctx(ctx)) return 0;
    ColiCudaTensor *t = static_cast<ColiCudaTensor *>(std::calloc(1, sizeof(*t)));
    if (!t) return 0;
    t->fmt = 8; t->I = I; t->O = O; t->device = device; t->weight_bytes = rb * (size_t)O;
    t->gs = 0; t->ng = (int)fp8_nblk(I); t->scale_count = (size_t)fp8_nblk(O) * (size_t)t->ng;
    t->weights = dev_w; t->weights_owned = 0;
    t->scales = dev_s; t->scales_borrowed = 1;
    *tensor = t;
    return 1;
}
extern "C" int coli_cuda_block_elems(int fmt) { return bd_file_row_bytes(fmt, 256) && fmt >= 100 ? (int)BD_BLK[fmt - 100] : 0; }

/* group size for the NEXT upload on this thread (fmt=4): routed through a
 * thread_local so the widely-wired upload signature (and the Windows DLL ABI)
 * stays untouched. pin_load uploads in parallel, hence thread_local. */
static thread_local int g_upload_gs = 0;
extern "C" int coli_cuda_tensor_upload_g(ColiCudaTensor **tensor,
                                         const void *weights, const float *scales,
                                         int fmt, int I, int O, int device, int gs);
extern "C" int coli_cuda_tensor_upload(ColiCudaTensor **tensor,
                                       const void *weights, const float *scales,
                                       int fmt, int I, int O, int device) try {
    if (!tensor) return 0;
    if (*tensor) {
        /* Cached device copy: usable even when the caller's host pointers are
         * gone. CUDA_RELEASE_HOST slots null their host pointers after upload,
         * and with the old order (!weights checked first) every later matmul
         * on such a slot failed here — the GPU tier silently never computed
         * for host-released slab experts. */
        ColiCudaTensor *t = *tensor;
        int want_gs = (fmt==4 && g_upload_gs>0) ? g_upload_gs : 0;
        return t->fmt == fmt && t->I == I && t->O == O && t->device == device && t->gs == want_gs;
    }
    if (fmt >= 100) return bd_tensor_upload(tensor, weights, fmt, I, O, device);
    DeviceContext *ctx = find_ctx(device);
    if (!weights || I < 1 || O < 1 || !select_ctx(ctx)) return 0;
    size_t rb = row_bytes(fmt, I);
    /* fmt=6 keeps its scales inside each 98-byte block, so it is the one
     * quantized format that legitimately arrives with scales == NULL. */
    if (!rb || (fmt && fmt != 6 && !scales)) return 0;
    /* kernels would read a zero LUT; shared predicate, pinned by
     * tests/test_cuda_lut_gate.c without a CUDA toolchain */
    if (!coli_cuda_fp8_gate_admits(fmt, g_fp8_lut_ready)) return 0;
    ColiCudaTensor *t = static_cast<ColiCudaTensor *>(std::calloc(1, sizeof(*t)));
    if (!t) return 0;
    t->fmt = fmt; t->I = I; t->O = O; t->device = device; t->weight_bytes = rb * (size_t)O;
    t->gs = (fmt==4 && g_upload_gs>0) ? g_upload_gs : 0;
    t->ng = t->gs ? (I + t->gs - 1) / t->gs : 1;
    t->scale_count = t->gs ? (size_t)O * (size_t)t->ng : (size_t)O;
    if (fmt == 7) {
        t->ng = (I + 31) / 32;
        t->scale_count = (size_t)O * t->ng;
    }
    if (fmt == 8) {   /* per-block scales: [ceil(O/FP8_BLOCK), ceil(I/FP8_BLOCK)] (fp8_format.h) */
        t->ng = (int)fp8_nblk(I);
        t->scale_count = (size_t)fp8_nblk(O) * (size_t)t->ng;
    }
    if (!cuda_ok(
            COLI_SYCL_ALLOC(t->weights, (void *)sycl::malloc_device(
                                 t->weight_bytes, dq()), t->weight_bytes),
            "tensor allocation")) {
        coli_cuda_tensor_free(t);
        return 0;
    }
    /* Ownership is a fact of the allocation, not the copy: set it BEFORE the
     * memcpy, or a failed H2D upload frees the tensor while weights_owned is
     * still 0 and free()'s ownership gate leaks the device buffer. */
    t->weights_owned=1;
    if (!cuda_ok(
            COLI_SYCL_TRY(dq()
                                 .memcpy(t->weights, weights, t->weight_bytes)
                                 .wait()),
            "tensor upload")) {
        coli_cuda_tensor_free(t);
        return 0;
    }
    if(fmt==2||fmt==4){ /* same nibble layout: offset-binary -> signed in place */
        dq().submit([&](sycl::handler &cgh) {
            auto t_weights_ct0 = (uint8_t *)t->weights;
            auto t_weight_bytes_ct1 = t->weight_bytes;

            cgh.parallel_for(
                sycl::nd_range<3>(
                    sycl::range<3>(1, 1,
                                   (unsigned)((t->weight_bytes + 255) / 256)) *
                        sycl::range<3>(1, 1, 256),
                    sycl::range<3>(1, 1, 256)),
                [=](sycl::nd_item<3> item_ct1) {
                    offset_to_signed_s4(t_weights_ct0, t_weight_bytes_ct1);
                });
        });
        }
    if (fmt && fmt != 6) {
        if (!cuda_ok(COLI_SYCL_ALLOC(t->scales, (float *)sycl::malloc_device(
                                          tensor_scale_bytes(t),
                                          dq()), tensor_scale_bytes(t)),
                     "scale allocation") ||
            !cuda_ok(COLI_SYCL_TRY(
                         dq()
                             .memcpy(t->scales, scales, tensor_scale_bytes(t))
                             .wait()),
                     "scale upload")) {
            coli_cuda_tensor_free(t);
            return 0;
        }
    }
    if (!upload_settled("tensor upload")) { coli_cuda_tensor_free(t); return 0; }
    if (fmt == 6) t->scale_count = 0;      /* in-block scales: nothing separate to track */
    t->tracked = 1;
    ctx->tensor_count++;
    ctx->tensor_bytes += t->weight_bytes + tensor_scale_bytes(t);
    *tensor = t;
    return 1;
}
catch (sycl::exception const &exc) { return sycl_caught(exc, "coli_cuda_tensor_upload"); }
extern "C" int coli_cuda_tensor_upload_g(ColiCudaTensor **tensor,
                                         const void *weights, const float *scales,
                                         int fmt, int I, int O, int device, int gs){
    g_upload_gs = gs>0 ? gs : 0;
    int r = coli_cuda_tensor_upload(tensor, weights, scales, fmt, I, O, device);
    g_upload_gs = 0;
    return r;
}

extern "C" int coli_cuda_tensor_overwrite(ColiCudaTensor *t,
                                          const void *weights,
                                          const float *scales) try {
    /* a block-format tensor: its own buffer or an arena's slot (coli_cuda_tensor_upload_at) */
    if (t && weights && t->weights && t->fmt >= 100) { DeviceContext *c = find_ctx(t->device); return select_ctx(c) && bd_rows_copy(t, weights, 0); }
    if (!t || !weights || !t->weights || !t->weights_owned) return 0;
    if (t->fmt && t->fmt != 6 && (!scales || !t->scales)) return 0;
    DeviceContext *ctx = find_ctx(t->device);
    if (!select_ctx(ctx)) return 0;
    /* Same steps as the upload, into the buffers already there. A failure
     * half-way leaves the tensor's content undefined: the caller frees it. */
    if (!cuda_ok(
            COLI_SYCL_TRY(dq()
                                 .memcpy(t->weights, weights, t->weight_bytes)
                                 .wait()),
            "tensor overwrite")) return 0;
    if (t->fmt==2||t->fmt==4) {
        dq().submit([&](sycl::handler &cgh) {
            auto t_weights_ct0 = (uint8_t *)t->weights;
            auto t_weight_bytes_ct1 = t->weight_bytes;

            cgh.parallel_for(
                sycl::nd_range<3>(
                    sycl::range<3>(1, 1,
                                   (unsigned)((t->weight_bytes + 255) / 256)) *
                        sycl::range<3>(1, 1, 256),
                    sycl::range<3>(1, 1, 256)),
                [=](sycl::nd_item<3> item_ct1) {
                    offset_to_signed_s4(t_weights_ct0, t_weight_bytes_ct1);
                });
        });
    }
    if (t->fmt && t->fmt != 6 &&
        !cuda_ok(COLI_SYCL_TRY(
                     dq()
                         .memcpy(t->scales, scales, tensor_scale_bytes(t))
                         .wait()),
                 "scale overwrite")) return 0;
    return upload_settled("tensor overwrite");
}
catch (sycl::exception const &exc) { return sycl_caught(exc, "coli_cuda_tensor_overwrite"); }


extern "C" int coli_cuda_tensor_update(ColiCudaTensor *tensor,
                                       const void *weights,
                                       const float *scales) try {
    if (tensor && weights && tensor->fmt >= 100) { DeviceContext *c = find_ctx(tensor->device); return select_ctx(c) && bd_rows_copy(tensor, weights, 0); }
    if (!tensor || !weights || (tensor->fmt && tensor->fmt != 6 && !scales)) return 0;
    DeviceContext *ctx=find_ctx(tensor->device);
    if (!select_ctx(ctx)) return 0;
    if (!cuda_ok(COLI_SYCL_TRY(
                     dq()
                         .memcpy(tensor->weights, weights, tensor->weight_bytes)
                         .wait()),
                 "tensor refresh")) return 0;
    if(tensor->fmt==2||tensor->fmt==4){
        dq().submit([&](sycl::handler &cgh) {
            auto tensor_weights_ct0 = (uint8_t *)tensor->weights;
            auto tensor_weight_bytes_ct1 = tensor->weight_bytes;

            cgh.parallel_for(
                sycl::nd_range<3>(
                    sycl::range<3>(
                        1, 1, (unsigned)((tensor->weight_bytes + 255) / 256)) *
                        sycl::range<3>(1, 1, 256),
                    sycl::range<3>(1, 1, 256)),
                [=](sycl::nd_item<3> item_ct1) {
                    offset_to_signed_s4(tensor_weights_ct0,
                                        tensor_weight_bytes_ct1);
                });
        });
    }
    /* fmt=6 stores scales in-block; fmt=7 stores byte exponents separately. */
    return !tensor->fmt || tensor->fmt == 6 ||
           cuda_ok(COLI_SYCL_TRY(dq()
                                        .memcpy(tensor->scales, scales,
                                                tensor_scale_bytes(tensor))
                                        .wait()),
                   "scale refresh");
}
catch (sycl::exception const &exc) { return sycl_caught(exc, "coli_cuda_tensor_update"); }

/* Test hook: COLI_GPU_FAIL_AFTER=N makes every GPU COMPUTE entry point report
 * failure after N successful calls (N=0: every call fails), exercising the
 * engine's CPU fallbacks and host-rematerialization end-to-end without real
 * hardware faults. Uploads/queries are not gated. Unset: no effect. */
static long g_gpu_calls;
static int fault_injected(void) {
    const char *fa = std::getenv("COLI_GPU_FAIL_AFTER");
    if (!g_sycl_lost)   /* an asynchronous error a queue's handler took: the same loss as a thrown one */
        for (int i = 0; i < g_nctx; i++) if (coli_sycl_lost(g_ctx[i].device)) { g_sycl_lost = 1; break; }
    return g_sycl_lost || (fa && g_gpu_calls++ >= std::atol(fa));
}

/* COLI_CUDA_F8_WARP mode, re-read per dispatch like its siblings. Strict
 * parse: a non-numeric value selects the DEFAULT, not atoi's silent 0.
 * Default 1 (warp kernels) on CUDA; 0 (original kernels) on HIP — the warp
 * kernels' wave64 width-32 shuffle sub-grouping has never been validated on
 * AMD silicon, and a scoring instrument must not default onto an untested
 * reduction. Opt in explicitly with =1/=2 once hip-test certifies it. */
#define COLI_F8_DEFAULT 1
static int f8_warp_mode(void) {
    const char *e = std::getenv("COLI_CUDA_F8_WARP");
    if (!e || !*e) return COLI_F8_DEFAULT;
    char *end; long v = std::strtol(e, &end, 10);
    return *end ? COLI_F8_DEFAULT : (int)v;
}

/* Opt-in numerical experiment; default and partial-width dispatch stay intact.
 * COLI_CUDA_F8_WARP=0 remains the original-family escape hatch. */
/* 1: the CPU order's kernels (f8cpu_block); 2: the same operations, every lane at work, a group of 8 lanes an output
 * row (grouped_*_f8c2); 3: the same operations, an output row's blocks four at a time (grouped_*_f8c3); 4: =3's gate/up
 * kernel with =2's down kernel (each =1's bits); for the expert groups (the dense path keeps =1's kernel); else 0 */
static int f8_cpu_order_mode(void) {
    const char *e=std::getenv("COLI_SYCL_F8_CPU_ORDER");
    return !e || !e[0] || e[1] ? 0 : e[0]=='1' ? 1 : e[0]=='2' ? 2 : e[0]=='3' ? 3 : e[0]=='4' ? 4 : 0;
}
static bool f8_cpu_order_on(void) { return f8_cpu_order_mode()>0; }
static bool cpu_silu_on(void) {
    const char *e=std::getenv("COLI_SYCL_SILU_CPU");
    return e && !std::strcmp(e,"1");
}
static void cpu_silu_note(int site) {
    // Host logging only; these atomics are not passed to the device.
    static std::atomic_flag mlp=ATOMIC_FLAG_INIT,group=ATOMIC_FLAG_INIT,pipe=ATOMIC_FLAG_INIT;
    auto &seen=site==0?mlp:(site==1?group:pipe);
    if(!seen.test_and_set(std::memory_order_relaxed))
        std::fprintf(stderr,"[SYCL] CPU-form SiLU: %s, double exp rounded to float, stable sigmoid, rounded division\n",
                     site==0?"fp8 mlp":(site==1?"fp8 group":"pipe"));
}
static void f8_cpu_order_note(bool grouped,int I,int O,int sg) {
    // Host-side logging only; these atomics are never visible to a kernel.
    static std::atomic_flag dense=ATOMIC_FLAG_INIT,group=ATOMIC_FLAG_INIT;
    auto &seen=grouped?group:dense;
    if(!seen.test_and_set(std::memory_order_relaxed))
        std::fprintf(stderr,"[SYCL] fp8 CPU order: %s I=%d O=%d SG=%d, 8 FMA lanes, full128 blocks, ascending double sum\n",
                     grouped?"group":"dense",I,O,sg);
}

/* One launch site for the dense matvec so fmt=8 honors the same toggle as
 * f8_group_launch: mode 0 runs the original quant_matmul branch (fully
 * original behavior), anything else the warp/shared-LUT rework. */
static int quant_matmul_launch(float *y, const float *x, const void *w,
        const float *sc, int fmt, int S, int I, int O, size_t rb, int gs, int ng,
        sycl::queue *stream = nullptr, bool warp_fp8 = true) {
    DeviceContext *ctx = cur_ctx();
    sycl::queue &q = stream ? *stream : dq();
    const int step = coli_sycl_mm_rows(S, O,
        coli_sycl_mm_override(std::getenv("COLI_SYCL_MM_ROWS")));
    if (!step || I < 1) return 0;
    /* oneAPI's default id-query contract bounds the GLOBAL LINEAR size to
     * INT_MAX. GLM S338/O28672 would launch 2,480,930,816 work-items. Rebase
     * complete rows; each output keeps exactly the original reduction tree.
     * No global kernel offset, new device kernel or relaxed compiler limit. */
    for (int base = 0; base < S; ) {
        const int rows = S - base < step ? S - base : step;
        float *yb = y + (size_t)base * O;
        const float *xb = x + (size_t)base * I;
        dpct::dim3 grid((unsigned)O, (unsigned)rows);
        if (fmt == 8 && warp_fp8 && f8_warp_mode() && ctx->has_fp64)   /* the warp form sums its blocks in double */
        {
            const float *c_e4m3_ptr_ct1 = ctx->e4m3;
            if(!(I&127) && f8_cpu_order_on()) {
                f8_cpu_order_note(false,I,O,ctx->sg);
                COLI_SG_LAUNCH(q, ctx->sg, grid, ctx->sg, 0,
                    quant_matmul_f8cpu<SG>(yb,xb,w,sc,I,O,c_e4m3_ptr_ct1));
            } else {
                COLI_SG_LAUNCH(q, ctx->sg, grid, 256, 0, quant_matmul_f8w<SG>(yb, xb, w, sc, rows, I, O, c_e4m3_ptr_ct1));
            }
        } else {

            coli_sycl_fprof_ev(q.submit([&](sycl::handler &cgh) {
                const float *c_e4m3_ptr_ct1 = cur_ctx()->e4m3;

                const uint8_t *c_e8_grid_acc_ct1 = cur_ctx()->e8_grid;

                cgh.parallel_for(sycl::nd_range<3>(grid * sycl::range<3>(1, 1, 256),
                                                   sycl::range<3>(1, 1, 256)),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     quant_matmul(yb, xb, w, sc, fmt, rows, I, O, rb, gs,
                                                  ng, c_e8_grid_acc_ct1,
                                                  c_e4m3_ptr_ct1);
                                 });
            }));
        }
        base += rows;
    }
    return 1;
}

extern "C" int coli_cuda_matmul(ColiCudaTensor **tensor, float *y,
                                const float *x, const void *weights,
                                const float *scales, int fmt, int S, int I,
                                int O, int device, int gs) try {
    if (fault_injected()) return 0;
    /* fmt=4 carries [O, ceil(I/gs)] scales: without the group size the plain
     * upload truncates the buffer to O floats and quant_matmul divides by
     * gs==0. Callers must come through the gs>0 path (upload_g) or stay on
     * the CPU (#298, #334). */
    if (fmt == 4 && gs <= 0) return 0;
    if (S < 1) return 0;
    if (gs > 0) { if (!coli_cuda_tensor_upload_g(tensor, weights, scales, fmt, I, O, device, gs)) return 0; }
    else        { if (!coli_cuda_tensor_upload(tensor, weights, scales, fmt, I, O, device)) return 0; }
    ColiCudaTensor *t = *tensor;
    DeviceContext *ctx = find_ctx(t->device);
    if (!select_ctx(ctx)) return 0;
    size_t rb = row_bytes(fmt, I);
    size_t xb = (size_t)S * I * sizeof(float), yb = (size_t)S * O * sizeof(float);
    if (!reserve(&ctx->dx, &ctx->dx_cap, xb) || !reserve(&ctx->dy, &ctx->dy_cap, yb)) return 0;
    if (!cuda_ok(COLI_SYCL_TRY(
                     dq().memcpy(ctx->dx, x, xb).wait()),
                 "input upload")) return 0;
    if (fmt >= 100) {
        if (!bd_tensor_gemv(ctx, t, ctx->dy, ctx->dx, S,
                            &dq())) return 0;
    } else if (!quant_matmul_launch(ctx->dy, ctx->dx, t->weights, t->scales, fmt, S,
                                    I, O, rb, t->gs, t->ng)) return 0;
    if (!cuda_ok(0, "matmul launch") ||
        !cuda_ok(COLI_SYCL_TRY(
                     dq().memcpy(y, ctx->dy, yb).wait()),
                 "output download")) return 0;
    return 1;
}
catch (sycl::exception const &exc) { return sycl_caught(exc, "coli_cuda_matmul"); }

/* coli_cuda_matmul_i8act (Qwen3.8's Q38_TRUNK_GPU_I8): a resident fmt=1 tensor times int8 activations the caller
 * quantized (one scale a row). One sub-group a (output row, activation row): the lanes take 4-byte words of the row
 * and of the activation, every product summed in int32 -- exact, so the order does not matter -- and the lane sums
 * meet in an integer reduction; lane 0 writes ((float)dot * scale[o]) * sx[s], the CPU's matmul_q_idot to the bit
 * (an int converted once, two single multiplies, nothing to fuse). I a multiple of 4 (checked) and rows 4-byte
 * aligned (an int8 tensor's rows are I bytes apart); |a product| <= 128 * 128, so a row up to 131 071 wide stays
 * inside int32. */
template <int SG>
static void gemv_i8act_kernel(float *y, const int8_t *xq, const float *sx, const int8_t *w, const float *sc,
                              int S, int I, int O) {
    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
    const int warp = item.get_local_id(2) / SG, lane = item.get_local_id(2) & (SG - 1);
    const int o = item.get_group(2) * (item.get_local_range(2) / SG) + warp;
    if (o >= O) return;   /* a sub-group's row past the end: uniform */
    const int words = I >> 2;
    const int32_t *wr = reinterpret_cast<const int32_t *>(w + (size_t)o * I);
    const float so = sc[o];
    for (int s = 0; s < S; s++) {
        const int32_t *xr = reinterpret_cast<const int32_t *>(xq + (size_t)s * I);
        int acc = 0;
        for (int k = lane; k < words; k += SG) {
            const int32_t a = wr[k], b = xr[k];
            acc += (int)(int8_t)(a & 0xff) * (int)(int8_t)(b & 0xff)
                 + (int)(int8_t)((a >> 8) & 0xff) * (int)(int8_t)((b >> 8) & 0xff)
                 + (int)(int8_t)((a >> 16) & 0xff) * (int)(int8_t)((b >> 16) & 0xff)
                 + (int)(int8_t)((a >> 24) & 0xff) * (int)(int8_t)((b >> 24) & 0xff);
        }
        acc = sycl::reduce_over_group(sg, acc, sycl::plus<int>());
        if (lane == 0) y[(size_t)s * O + o] = ((float)acc * so) * sx[s];
    }
}
extern "C" int coli_cuda_matmul_i8act(ColiCudaTensor *t, float *y, const signed char *xq, const float *sx,
                                      int S, int I, int O) try {
    if (fault_injected()) return 0;
    if (!t || !y || !xq || !sx || S < 1 || t->fmt != 1 || t->I != I || t->O != O || (I & 3) || I > 131071 ||
        !t->weights || !t->scales)
        return 0;   /* I > 131 071: a dot could leave int32 */
    DeviceContext *ctx = find_ctx(t->device);
    if (!select_ctx(ctx)) return 0;
    /* the activations and their scales in one device block: int8 rows, then the floats 16-byte aligned */
    const size_t xb = (size_t)S * I, sb = (size_t)S * sizeof(float), soff = (xb + 15) & ~(size_t)15;
    const size_t yb = (size_t)S * O * sizeof(float);
    if (!reserve(&ctx->dx, &ctx->dx_cap, soff + sb) || !reserve(&ctx->dy, &ctx->dy_cap, yb)) return 0;
    char *dxb = reinterpret_cast<char *>(ctx->dx);
    if (!cuda_ok(COLI_SYCL_TRY(dq().memcpy(dxb, xq, xb)), "i8act activation upload") ||
        !cuda_ok(COLI_SYCL_TRY(dq().memcpy(dxb + soff, sx, sb)), "i8act scale upload")) { try { dq().wait(); } catch (...) {} return 0; }
    const int8_t *dxq = reinterpret_cast<const int8_t *>(dxb);
    const float *dsx = reinterpret_cast<const float *>(dxb + soff);
    float *dy = ctx->dy;
    const int8_t *dw = static_cast<const int8_t *>(t->weights);
    const float *dsc = t->scales;
    const int wpb = ctx->wpb;   /* output rows a block: a sub-group each */
    dpct::dim3 grid((unsigned)((O + wpb - 1) / wpb), 1u);
    COLI_SG_LAUNCH(dq(), ctx->sg, grid, 256, 0, gemv_i8act_kernel<SG>(dy, dxq, dsx, dw, dsc, S, I, O));
    if (!cuda_ok(0, "i8act launch") ||
        !cuda_ok(COLI_SYCL_TRY(dq().memcpy(y, dy, yb).wait()), "i8act output download")) return 0;
    return 1;
}
catch (sycl::exception const &exc) {
    try { dq().wait(); } catch (...) {}   /* the caller frees xq and sx when this returns: no copy of them in flight */
    return sycl_caught(exc, "coli_cuda_matmul_i8act");
}

/* coli_cuda_tuned_default (backend_cuda.h): a switch's measured default from the devices' profiles
 * (coli_sycl_tuned_default), only where every device gives the same value; the first use of each switch that
 * takes a measured value says so on stderr. The devices: the opened ones; before coli_cuda_init (an engine
 * deciding its defaults before it opens anything), the ones it will open -- COLI_GPUS's ordinals, else every
 * listed GPU -- read without opening them (coli_sycl_listed_arch). */
static coli_sycl_arch tuned_arch_at(int k, const int *pre, int npre) {
    return g_nctx > 0 ? coli_sycl_profile_for(g_ctx[k].device).arch : k < npre ? coli_sycl_listed_arch(pre[k]) : COLI_SYCL_ARCH_NONE;
}
extern "C" int coli_cuda_tuned_default(const char *knob, int generic) {
    if (!knob) return generic;
    const char *pd = std::getenv("COLI_PLATFORM_DEFAULTS");
    if (pd && !std::strcmp(pd, "0")) return generic;   /* asked first: the opt-out lists no GPU and starts no runtime */
    int pre[COLI_CUDA_MAX_DEVICES], n = g_nctx;
    if (n <= 0) {
        n = 0;
        const char *gl = std::getenv("COLI_GPUS");
        if (gl && *gl) {
            for (const char *s = gl; *s && n < COLI_CUDA_MAX_DEVICES;) {
                char *end; long v = std::strtol(s, &end, 10);
                if (end == s) return generic;            /* not a list of ordinals: decide nothing */
                pre[n++] = (int)v;
                s = *end == ',' ? end + 1 : end;
                if (*end && *end != ',') return generic;
            }
        } else {
            const int cnt = coli_sycl_available_device_count();
            for (int i = 0; i < cnt && n < COLI_CUDA_MAX_DEVICES; i++) pre[n++] = i;
        }
        if (n <= 0) return generic;
    }
    coli_sycl_profile p;
    std::memset(&p, 0, sizeof p);
    p.arch = tuned_arch_at(0, pre, n);
    const coli_sycl_arch arch0 = p.arch;
    const int v = coli_sycl_tuned_default(p, pd, knob, generic);
    for (int i = 1; i < n; i++) {
        p.arch = tuned_arch_at(i, pre, n);
        if (coli_sycl_tuned_default(p, pd, knob, generic) != v) return generic;   /* devices of different architectures */
    }
    if (v != generic) {
        static std::mutex said_mu;
        static std::vector<std::string> said;
        std::lock_guard<std::mutex> lk(said_mu);
        bool seen = false;
        for (const auto &s : said) if (s == knob) { seen = true; break; }
        if (!seen) {
            said.emplace_back(knob);
            std::fprintf(stderr, "[SYCL] %s unset: %d, the value measured on this architecture (%s); "
                         "COLI_PLATFORM_DEFAULTS=0 keeps the code's %d\n", knob, v, coli_sycl_arch_name(arch0), generic);
        }
    }
    return v;
}

/* coli_cuda_vision_attention (backend_cuda.h): a vision tower's attention, which this backend does not offer
 * yet. 0 leaves out untouched and the attention to the CPU, as the loader answers for a backend without
 * the symbol; the engines link this backend directly, so the symbol must exist. */
extern "C" int coli_cuda_vision_attention(int /*device*/, float * /*out*/, const float * /*qkv*/, int /*n*/,
                                          int /*heads*/, int /*D*/, float /*scale*/) {
    return 0;
}

/* ---- gated delta layer on the device (see backend_cuda.h) ---------------- */
#define DN_MAX_HEADS 64
struct DnGates { float egh[DN_MAX_HEADS]; float beta[DN_MAX_HEADS]; };
struct ColiCudaDn {
    int device, vh, vk, kdim, vdim, conv_dim, convk, hidden;
    float eps; int gate_sigmoid;
    float *ring, *rec, *conv_w, *norm_w;              /* resident */
    float *x, *qkvz, *conv_out, *outr, *out;          /* scratch for one token */
};

/* one thread per channel: causal depthwise conv over the ring + this token,
 * SiLU, then the ring advances (drop oldest, append this token's value) */
void dn_conv_kernel(const float *__restrict__ qkv, float *__restrict__ ring,
                               const float *__restrict__ w, float *__restrict__ conv_out,
                               int conv_dim, int convk) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int cc = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
             item_ct1.get_local_id(2);
    if (cc >= conv_dim) return;
    const float *wc = w + (size_t)cc * convk;
    float *rg = ring + (size_t)cc * (convk - 1);
    float acc = 0.f;
    for (int kk = 0; kk < convk - 1; kk++) acc += wc[kk] * rg[kk];
    float cur = qkv[cc];
    acc += wc[convk - 1] * cur;
    conv_out[cc] = acc / (1.f + sycl::native::exp(-acc));
    for (int kk = 0; kk < convk - 2; kk++) rg[kk] = rg[kk + 1];
    rg[convk - 2] = cur;
}

template <int SG>
static double dn_block_sum(double v, double *sh) {
    /* blockDim <= 256 (kdim, vdim <= 256); sub-group shuffle then one pass over the sub-groups */
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    for (int o = SG / 2; o > 0; o >>= 1) v += shfl_down(v, o);
    int lane = item_ct1.get_local_id(2) & (SG - 1),
        wid = item_ct1.get_local_id(2) / SG;
    item_ct1.barrier();
    if (lane == 0) sh[wid] = v;
    item_ct1.barrier();
    double r = 0.0;
    int nw = (item_ct1.get_local_range(2) + SG - 1) / SG;
    for (int i = 0; i < nw; i++) r += sh[i];
    return r;
}

/* one block per value head; thread t owns column t of the state (vdim) and
 * element t of q/k (kdim). Same arithmetic as the CPU deltanet(): norms in
 * double with eps inside the sqrt, S = S*egh + k delta^T with delta read from
 * the decayed state, out = q^T S over the updated state, then the gated
 * RMSNorm with silu(z). */
template <int SG>
void dn_head_kernel(const float *__restrict__ conv_out,
                    const float *__restrict__ z, float *__restrict__ S,
                    float *__restrict__ outr, const float *__restrict__ normw,
                    DnGates g, int rep, int kdim, int vdim, int key_dim_tot,
                    float scale, float eps, int gate_sigmoid) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &q = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[256]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &k = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[256]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &delta =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[256]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &red =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<double[32]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    int h = item_ct1.get_group(2), t = item_ct1.get_local_id(2);
    int kh = h / rep;
    const float *qs = conv_out + (size_t)kh * kdim;
    const float *ks = conv_out + key_dim_tot + (size_t)kh * kdim;
    const float *vs = conv_out + 2 * (size_t)key_dim_tot + (size_t)h * vdim;
    float qv = t < kdim ? qs[t] : 0.f, kv_ = t < kdim ? ks[t] : 0.f;
    double sq = dn_block_sum<SG>((double)qv * qv, red);
    double sk = dn_block_sum<SG>((double)kv_ * kv_, red);
    double nq = sycl::sqrt(sq + 1e-6), nk = sycl::sqrt(sk + 1e-6);
    if (t < kdim) { q[t] = (float)((double)qv / nq * scale); k[t] = (float)((double)kv_ / nk); }
    item_ct1.barrier();
    float egh = g.egh[h], beta = g.beta[h];
    float *Sh = S + (size_t)h * kdim * vdim;
    float o = 0.f;
    if (t < vdim) {
        float kvsum = 0.f;
        for (int kk = 0; kk < kdim; kk++) kvsum += k[kk] * (Sh[(size_t)kk * vdim + t] * egh);
        delta[t] = (vs[t] - kvsum) * beta;
        for (int kk = 0; kk < kdim; kk++) {
            float s = Sh[(size_t)kk * vdim + t] * egh + k[kk] * delta[t];
            Sh[(size_t)kk * vdim + t] = s;
            o += q[kk] * s;
        }
    }
    double ms = dn_block_sum<SG>((double)o * o, red);
    if (t < vdim) {
        float r = 1.f / sycl::sqrt((float)(ms / vdim) + eps);
        float zz = z[(size_t)h * vdim + t];
        float gate = gate_sigmoid ? 1.f / (1.f + sycl::native::exp(-zz))
                                  : zz / (1.f + sycl::native::exp(-zz));
        outr[(size_t)h * vdim + t] = (o * r * normw[t]) * gate;
    }
}

extern "C" ColiCudaDn *coli_cuda_dn_create(int device, int vh, int vk, int kdim,
                                           int vdim, int conv_dim, int convk,
                                           int hidden, const float *conv_w,
                                           const float *norm_w, float eps,
                                           int gate_sigmoid) try {
    if (vh < 1 || vh > DN_MAX_HEADS || vk < 1 || vh % vk || kdim < 1 || kdim > 256 || vdim < 1 || vdim > 256 ||
        convk < 2 || conv_dim != 2 * vk * kdim + vh * vdim || hidden < 1 || !conv_w || !norm_w) return NULL;
    DeviceContext *ctx = find_ctx(device);
    if (!select_ctx(ctx)) return NULL;
    ColiCudaDn *d = static_cast<ColiCudaDn *>(std::calloc(1, sizeof(*d)));
    if (!d) return NULL;
    d->device = device; d->vh = vh; d->vk = vk; d->kdim = kdim; d->vdim = vdim;
    d->conv_dim = conv_dim; d->convk = convk; d->hidden = hidden; d->eps = eps; d->gate_sigmoid = gate_sigmoid ? 1 : 0;
    size_t ring_b = (size_t)conv_dim * (convk - 1) * sizeof(float), rec_b = (size_t)vh * kdim * vdim * sizeof(float);
    size_t proj = (size_t)conv_dim + (size_t)vh * vdim;
    int ok =
        cuda_ok(COLI_SYCL_ALLOC(d->ring, (float *)sycl::malloc_device(
                                     ring_b, dq()), ring_b),
                "dn ring") &&
        cuda_ok(COLI_SYCL_ALLOC(d->rec, (float *)sycl::malloc_device(
                                     rec_b, dq()), rec_b),
                "dn state") &&
        cuda_ok(COLI_SYCL_ALLOC(d->conv_w, sycl::malloc_device<float>(
                        (size_t)conv_dim * convk, dq()), (size_t)conv_dim * convk),
                "dn conv weights") &&
        cuda_ok(COLI_SYCL_ALLOC(d->norm_w, sycl::malloc_device<float>(
                                     (size_t)vdim, dq()), (size_t)vdim),
                "dn norm weights") &&
        cuda_ok(
            COLI_SYCL_ALLOC(d->x, sycl::malloc_device<float>(
                                 (size_t)hidden, dq()), (size_t)hidden),
            "dn x") &&
        cuda_ok(COLI_SYCL_ALLOC(d->qkvz, sycl::malloc_device<float>(
                                     proj, dq()), proj),
                "dn qkvz") &&
        cuda_ok(
            COLI_SYCL_ALLOC(d->conv_out, sycl::malloc_device<float>(
                                 (size_t)conv_dim, dq()), (size_t)conv_dim),
            "dn conv out") &&
        cuda_ok(COLI_SYCL_ALLOC(d->outr, sycl::malloc_device<float>(
                        (size_t)vh * vdim, dq()), (size_t)vh * vdim),
                "dn outr") &&
        cuda_ok(
            COLI_SYCL_ALLOC(d->out, sycl::malloc_device<float>(
                                 (size_t)hidden, dq()), (size_t)hidden),
            "dn out") &&
        cuda_ok(COLI_SYCL_TRY(
                    dq()
                        .memcpy(d->conv_w, conv_w,
                                (size_t)conv_dim * convk * sizeof(float))
                        .wait()),
                "dn conv upload") &&
        cuda_ok(COLI_SYCL_TRY(
                    dq()
                        .memcpy(d->norm_w, norm_w, (size_t)vdim * sizeof(float))
                        .wait()),
                "dn norm upload") &&
        cuda_ok(
            COLI_SYCL_TRY(
                dq().memset(d->ring, 0, ring_b).wait()),
            "dn ring zero") &&
        cuda_ok(COLI_SYCL_TRY(
                    dq().memset(d->rec, 0, rec_b).wait()),
                "dn state zero");
    if (!ok) { coli_cuda_dn_free(d); return NULL; }
    return d;
}
catch (sycl::exception const &exc) { sycl_caught(exc, "coli_cuda_dn_create"); return NULL; }
extern "C" void coli_cuda_dn_free(ColiCudaDn *d) {
    if (!d) return;
    DeviceContext *ctx = find_ctx(d->device);
    if (ctx) select_ctx(ctx);
    float *bufs[] = { d->ring, d->rec, d->conv_w, d->norm_w, d->x, d->qkvz, d->conv_out, d->outr, d->out };
    for (float *b : bufs) if (b) sycl::free(b, dq());
    std::free(d);
}
extern "C" int coli_cuda_dn_set_state(ColiCudaDn *d, const float *ring,
                                      const float *rec) try {
    if (!d) return 0;
    if (!select_ctx(find_ctx(d->device))) return 0;
    size_t ring_b = (size_t)d->conv_dim * (d->convk - 1) * sizeof(float), rec_b = (size_t)d->vh * d->kdim * d->vdim * sizeof(float);
    int ok = ring ? cuda_ok(COLI_SYCL_TRY(dq()
                                                 .memcpy(d->ring, ring, ring_b)
                                                 .wait()),
                            "dn ring set")
                  : cuda_ok(COLI_SYCL_TRY(dq()
                                                 .memset(d->ring, 0, ring_b)
                                                 .wait()),
                            "dn ring zero");
    ok = ok && (rec ? cuda_ok(COLI_SYCL_TRY(dq()
                                                   .memcpy(d->rec, rec, rec_b)
                                                   .wait()),
                              "dn state set")
                    : cuda_ok(COLI_SYCL_TRY(dq()
                                                   .memset(d->rec, 0, rec_b)
                                                   .wait()),
                              "dn state zero"));
    return ok;
}
catch (sycl::exception const &exc) { return sycl_caught(exc, "coli_cuda_dn_set_state"); }
extern "C" int coli_cuda_dn_get_state(ColiCudaDn *d, float *ring,
                                      float *rec) try {
    if (!d || !ring || !rec) return 0;
    if (!select_ctx(find_ctx(d->device))) return 0;
    size_t ring_b = (size_t)d->conv_dim * (d->convk - 1) * sizeof(float), rec_b = (size_t)d->vh * d->kdim * d->vdim * sizeof(float);
    return cuda_ok(COLI_SYCL_TRY(dq()
                                        .memcpy(ring, d->ring, ring_b)
                                        .wait()),
                   "dn ring get") &&
           cuda_ok(COLI_SYCL_TRY(dq()
                                        .memcpy(rec, d->rec, rec_b)
                                        .wait()),
                   "dn state get");
}
catch (sycl::exception const &exc) { return sycl_caught(exc, "coli_cuda_dn_get_state"); }
extern "C" int coli_cuda_dn_step(ColiCudaDn *d, ColiCudaTensor *proj,
                                 ColiCudaTensor *projz, ColiCudaTensor *outp,
                                 const float *x, float *out, const float *egh,
                                 const float *beta) try {
    if (fault_injected()) return 0;
    if (!d || !proj || !outp || !x || !out || !egh || !beta) return 0;
    size_t proj_dim = (size_t)d->conv_dim + (size_t)d->vh * d->vdim, value_dim = (size_t)d->vh * d->vdim;
    size_t want_o = projz ? (size_t)d->conv_dim : proj_dim;
    if (proj->fmt != 1 || proj->I != d->hidden || (size_t)proj->O != want_o || proj->device != d->device ||
        outp->fmt != 1 || (size_t)outp->I != value_dim || outp->O != d->hidden || outp->device != d->device) return 0;
    if (projz && (projz->fmt != 1 || projz->I != d->hidden || (size_t)projz->O != value_dim || projz->device != d->device)) return 0;
    if (!select_ctx(find_ctx(d->device))) return 0;
    if (!cur_ctx()->has_fp64) return 0;   /* dn_head_kernel's norms are double */
    DnGates g;
    for (int h = 0; h < d->vh; h++) { g.egh[h] = egh[h]; g.beta[h] = beta[h]; }
    if (!cuda_ok(COLI_SYCL_TRY(
                     dq()
                         .memcpy(d->x, x, (size_t)d->hidden * sizeof(float))
                         .wait()),
                 "dn x upload")) return 0;
    if (!quant_matmul_launch(d->qkvz, d->x, proj->weights, proj->scales, 1, 1, d->hidden, (int)want_o,
                             row_bytes(1, d->hidden), proj->gs, proj->ng)) return 0;
    if (projz)
        if (!quant_matmul_launch(d->qkvz + d->conv_dim, d->x, projz->weights, projz->scales, 1, 1, d->hidden, (int)value_dim,
                                 row_bytes(1, d->hidden), projz->gs, projz->ng)) return 0;
    dq().submit([&](sycl::handler &cgh) {
        const float *d_qkvz_ct0 = d->qkvz;
        auto d_ring_ct1 = d->ring;
        const float *d_conv_w_ct2 = d->conv_w;
        auto d_conv_out_ct3 = d->conv_out;
        auto d_conv_dim_ct4 = d->conv_dim;
        auto d_convk_ct5 = d->convk;

        cgh.parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, (unsigned)((d->conv_dim + 255) / 256)) *
                    sycl::range<3>(1, 1, 256),
                sycl::range<3>(1, 1, 256)),
            [=](sycl::nd_item<3> item_ct1) {
                dn_conv_kernel(d_qkvz_ct0, d_ring_ct1, d_conv_w_ct2,
                               d_conv_out_ct3, d_conv_dim_ct4, d_convk_ct5);
            });
    });
    DeviceContext *ctx = cur_ctx();
    int threads = d->kdim > d->vdim ? d->kdim : d->vdim;
    threads = ((threads + ctx->sg - 1) / ctx->sg) * ctx->sg;
    {
        const float *d_conv_out_ct0 = d->conv_out;
        const float *d_qkvz_d_conv_dim_ct1 = d->qkvz + d->conv_dim;
        auto d_rec_ct2 = d->rec;
        auto d_outr_ct3 = d->outr;
        const float *d_norm_w_ct4 = d->norm_w;
        auto d_vh_d_vk_ct6 = d->vh / d->vk;
        auto d_kdim_ct7 = d->kdim;
        auto d_vdim_ct8 = d->vdim;
        auto d_vk_d_kdim_ct9 = d->vk * d->kdim;
        auto sqrtf_float_d_kdim_ct10 = 1.f / sqrtf((float)d->kdim);
        auto d_eps_ct11 = d->eps;
        auto d_gate_sigmoid_ct12 = d->gate_sigmoid;
        COLI_SG_LAUNCH(dq(), ctx->sg, sycl::range<3>(1, 1, (unsigned)d->vh), threads, 0,
                       dn_head_kernel<SG>(d_conv_out_ct0, d_qkvz_d_conv_dim_ct1, d_rec_ct2, d_outr_ct3, d_norm_w_ct4, g,
                                          d_vh_d_vk_ct6, d_kdim_ct7, d_vdim_ct8, d_vk_d_kdim_ct9, sqrtf_float_d_kdim_ct10,
                                          d_eps_ct11, d_gate_sigmoid_ct12));
    }
    if (!quant_matmul_launch(d->out, d->outr, outp->weights, outp->scales, 1, 1, (int)value_dim, d->hidden,
                             row_bytes(1, (int)value_dim), outp->gs, outp->ng)) return 0;
    return cuda_ok(0, "dn step launch") &&
           cuda_ok(
               COLI_SYCL_TRY(
                   dq()
                       .memcpy(out, d->out, (size_t)d->hidden * sizeof(float))
                       .wait()),
               "dn out download");
}
catch (sycl::exception const &exc) { return sycl_caught(exc, "coli_cuda_dn_step"); }

/* MXFP4 matmul, stateless. Separate from coli_cuda_matmul on purpose: that one
 * takes scales as const float* and caches an uploaded tensor, while MXFP4
 * scales are ue8m0 BYTES -- passing them through the float* parameter would
 * compile and silently reinterpret the buffer. Kimi K3's routed experts stream
 * (a fill-once tier at decode), so there is nothing to cache here anyway; the
 * weights go up with the call.
 *
 * Returns 0 and leaves y untouched on any failure, which is the contract the
 * engine's GPU paths already use to fall back to CPU. */
extern "C" int coli_cuda_matmul_mxfp4(float *y, const float *x,
                                      const uint8_t *q4, const uint8_t *e8s,
                                      int S, int I, int O) try {
    if (fault_injected()) return 0;
    if (S < 1 || I < 1 || O < 1 || !y || !x || !q4 || !e8s) return 0;
    DeviceContext *ctx = find_ctx(0);
    if (!select_ctx(ctx)) return 0;

    size_t rb = (size_t)(I + 1) / 2, ng = (size_t)(I + 31) / 32;
    size_t wb = (size_t)O * rb, sb = (size_t)O * ng;
    size_t xb = (size_t)S * I * sizeof(float), yb = (size_t)S * O * sizeof(float);

    if (!reserve_bytes(&ctx->mxfp4_weights, &ctx->mxfp4_weights_cap, wb) ||
        !reserve_bytes(&ctx->mxfp4_scales, &ctx->mxfp4_scales_cap, sb)) return 0;
    uint8_t *dw = static_cast<uint8_t *>(ctx->mxfp4_weights);
    uint8_t *ds = static_cast<uint8_t *>(ctx->mxfp4_scales);

    int ok =
        reserve(&ctx->x, &ctx->x_cap, xb) &&
        reserve(&ctx->y, &ctx->y_cap, yb) &&
        cuda_ok(COLI_SYCL_TRY(
                    dq().memcpy(dw, q4, wb).wait()),
                "mxfp4 weight upload") &&
        cuda_ok(COLI_SYCL_TRY(
                    dq().memcpy(ds, e8s, sb).wait()),
                "mxfp4 scale upload") &&
        cuda_ok(COLI_SYCL_TRY(
                    dq().memcpy(ctx->x, x, xb).wait()),
                "mxfp4 input upload");
    if (ok) {
        dpct::dim3 grid((unsigned)O, (unsigned)S);
        {

            dq().submit([&](sycl::handler &cgh) {
                const float *c_e4m3_ptr_ct1 = cur_ctx()->e4m3;

                const uint8_t *c_e8_grid_acc_ct1 = cur_ctx()->e8_grid;

                auto ctx_y_ct0 = ctx->y;
                const float *ctx_x_ct1 = ctx->x;

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * sycl::range<3>(1, 1, 256),
                                      sycl::range<3>(1, 1, 256)),
                    [=](sycl::nd_item<3> item_ct1) {
                        quant_matmul(ctx_y_ct0, ctx_x_ct1, dw,
                                     reinterpret_cast<const float *>(ds), 7, S,
                                     I, O, rb, 32, (int)ng, c_e8_grid_acc_ct1,
                                     c_e4m3_ptr_ct1);
                    });
            });
        }
        ok = cuda_ok(0, "mxfp4 launch") &&
             cuda_ok(
                 COLI_SYCL_TRY(
                     dq().memcpy(y, ctx->y, yb).wait()),
                 "mxfp4 output download");
    }
    return ok;
}
catch (sycl::exception const &exc) { return sycl_caught(exc, "coli_cuda_matmul_mxfp4"); }

/* Reuse one weight/scale staging allocation across the three projections.
 * Default-stream copies are ordered after the previous projection's reads. */
static int mxfp4_project(float *y, const float *x, uint8_t *dw, uint8_t *ds,
                         const uint8_t *w, const uint8_t *sc, int S, int I,
                         int O) try {
    size_t rb = ((size_t)I + 1) / 2, ng = ((size_t)I + 31) / 32;
    if (!cuda_ok(COLI_SYCL_TRY(dq()
                                      .memcpy(dw, w, (size_t)O * rb)
                                      .wait()),
                 "expert weight upload") ||
        !cuda_ok(COLI_SYCL_TRY(dq()
                                      .memcpy(ds, sc, (size_t)O * ng)
                                      .wait()),
                 "expert scale upload")) return 0;
    {

        dq().submit([&](sycl::handler &cgh) {
            const float *c_e4m3_ptr_ct1 = cur_ctx()->e4m3;

            const uint8_t *c_e8_grid_acc_ct1 = cur_ctx()->e8_grid;

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, S, O) *
                                      sycl::range<3>(1, 1, 256),
                                  sycl::range<3>(1, 1, 256)),
                [=](sycl::nd_item<3> item_ct1) {
                    quant_matmul(y, x, dw, reinterpret_cast<const float *>(ds),
                                 7, S, I, O, rb, 32, (int)ng, c_e8_grid_acc_ct1,
                                 c_e4m3_ptr_ct1);
                });
        });
    }
    return cuda_ok(0, "MXFP4 expert projection");
}
catch (sycl::exception const &exc) { return sycl_caught(exc, "mxfp4_project"); }

extern "C" int
coli_cuda_expert_mxfp4(float *y, const float *x, const unsigned char *gate_w,
                       const unsigned char *gate_s, const unsigned char *up_w,
                       const unsigned char *up_s, const unsigned char *down_w,
                       const unsigned char *down_s, int S, int D, int I,
                       float b1, float b2) try {
    if (fault_injected() || !x || !y || !gate_w || !gate_s || !up_w || !up_s ||
        !down_w || !down_s || S < 1 || S > 65535 || D < 1 || I < 1 ||
        !(b1 > 0.f) || !(b2 > 0.f) || !std::isfinite(b1) || !std::isfinite(b2)) return 0;
    DeviceContext *ctx = find_ctx(0);
    if (!select_ctx(ctx)) return 0;
    size_t xb = (size_t)S * D * sizeof(float), ib = (size_t)S * I * sizeof(float);
    if (!reserve(&ctx->x, &ctx->x_cap, xb) || !reserve(&ctx->y, &ctx->y_cap, xb) ||
        !reserve(&ctx->gate, &ctx->gate_cap, ib) || !reserve(&ctx->up, &ctx->up_cap, ib)) return 0;
    size_t gw = (size_t)I * (((size_t)D + 1) / 2), dwb = (size_t)D * (((size_t)I + 1) / 2);
    size_t gs = (size_t)I * (((size_t)D + 31) / 32), dsb = (size_t)D * (((size_t)I + 31) / 32);
    /* Grow to the largest projection seen, then reuse across routed experts.
     * Slot identity is irrelevant: every call refreshes all weight bytes. */
    if (!reserve_bytes(&ctx->mxfp4_expert_weights, &ctx->mxfp4_expert_weights_cap, gw > dwb ? gw : dwb) ||
        !reserve_bytes(&ctx->mxfp4_expert_scales, &ctx->mxfp4_expert_scales_cap, gs > dsb ? gs : dsb)) return 0;
    uint8_t *dw = static_cast<uint8_t *>(ctx->mxfp4_expert_weights);
    uint8_t *ds = static_cast<uint8_t *>(ctx->mxfp4_expert_scales);
    int ok =
        cuda_ok(COLI_SYCL_TRY(
                    dq().memcpy(ctx->x, x, xb).wait()),
                "expert input upload") &&
        mxfp4_project(ctx->gate, ctx->x, dw, ds, gate_w, gate_s, S, D, I) &&
        mxfp4_project(ctx->up, ctx->x, dw, ds, up_w, up_s, S, D, I);
    if (ok) {
        size_t n = (size_t)S * I;
        dq().submit([&](sycl::handler &cgh) {
            auto ctx_gate_ct0 = ctx->gate;
            const float *ctx_up_ct1 = ctx->up;

            cgh.parallel_for(
                sycl::nd_range<3>(
                    sycl::range<3>(1, 1, (unsigned)((n + 255) / 256)) *
                        sycl::range<3>(1, 1, 256),
                    sycl::range<3>(1, 1, 256)),
                [=](sycl::nd_item<3> item_ct1) {
                    situ_mul(ctx_gate_ct0, ctx_up_ct1, n, b1, b2);
                });
        });
        ok =
            cuda_ok(0, "SiTU-GLU launch") &&
            mxfp4_project(ctx->y, ctx->gate, dw, ds, down_w, down_s, S, I, D) &&
            cuda_ok(
                COLI_SYCL_TRY(
                    dq().memcpy(y, ctx->y, xb).wait()),
                "expert output download");
    }
    return ok;
}
catch (sycl::exception const &exc) { return sycl_caught(exc, "coli_cuda_expert_mxfp4"); }

extern "C" int coli_cuda_expert_mlp(ColiCudaTensor *gate, ColiCudaTensor *up,
                                    ColiCudaTensor *down, float *y,
                                    const float *x, int S) try {
    if (fault_injected()) return 0;
    /* same reason as coli_cuda_matmul: fmt=4 without recorded group info would
     * misread the scales (and divide by gs==0 in the kernel). */
    if (gate && ((gate->fmt == 4 && gate->gs <= 0) ||
                 (up && up->fmt == 4 && up->gs <= 0) ||
                 (down && down->fmt == 4 && down->gs <= 0))) return 0;
    if ((gate && gate->fmt >= 100) || (up && up->fmt >= 100) || (down && down->fmt >= 100)) return 0;   /* the block formats: the chain's GEMVs only */
    if (!gate || !up || !down || !x || !y || S < 1 ||
        gate->device != up->device || gate->device != down->device ||
        gate->I != up->I || gate->O != up->O ||
        down->I != gate->O || down->O != gate->I) return 0;
    DeviceContext *ctx = find_ctx(gate->device);
    if (!select_ctx(ctx)) return 0;
    int D = gate->I, I = gate->O;
    size_t xb=(size_t)S*D*sizeof(float), ib=(size_t)S*I*sizeof(float);
    size_t yb=(size_t)S*D*sizeof(float);
    if (!reserve(&ctx->x,&ctx->x_cap,xb) || !reserve(&ctx->y,&ctx->y_cap,yb) ||
        !reserve(&ctx->gate,&ctx->gate_cap,ib) || !reserve(&ctx->up,&ctx->up_cap,ib)) return 0;
    if (!cuda_ok(COLI_SYCL_TRY(
                     dq().memcpy(ctx->x, x, xb).wait()),
                 "expert input upload")) return 0;
    if (!quant_matmul_launch(ctx->gate,ctx->x,gate->weights,gate->scales,
        gate->fmt,S,D,I,row_bytes(gate->fmt,D),gate->gs,gate->ng)) return 0;
    if (!quant_matmul_launch(ctx->up,ctx->x,up->weights,up->scales,
        up->fmt,S,D,I,row_bytes(up->fmt,D),up->gs,up->ng)) return 0;
    size_t n=(size_t)S*I;
    const bool cpu_silu=gate->fmt==8 && up->fmt==8 && down->fmt==8 &&
        !(D&127) && !(I&127) && ctx->has_fp64 && f8_warp_mode() && cpu_silu_on();
    if(cpu_silu) {
        cpu_silu_note(0);
        float *g=ctx->gate;const float *u=ctx->up;
        dq().parallel_for(sycl::nd_range<3>(sycl::range<3>(1,1,((n+255)/256)*256),
                                          sycl::range<3>(1,1,256)),
                         [=](sycl::nd_item<3>){silu_cpu_mul(g,u,n);});
    } else dq().submit([&](sycl::handler &cgh) {
        auto ctx_gate_ct0 = ctx->gate;
        const float *ctx_up_ct1 = ctx->up;

        cgh.parallel_for(sycl::nd_range<3>(
                             sycl::range<3>(1, 1, (unsigned)((n + 255) / 256)) *
                                 sycl::range<3>(1, 1, 256),
                             sycl::range<3>(1, 1, 256)),
                         [=](sycl::nd_item<3> item_ct1) {
                             silu_mul(ctx_gate_ct0, ctx_up_ct1, n);
                         });
    });
    /* fmt=6: the down projection stores W@Q, so its input needs Q^T applied. This
     * one is per-expert (the silu product is not shared), unlike the gate/up input
     * rotation, which the caller does once per layer -- same split as moe(). */
    if (down->fmt == 6 &&
        !e8_rot_rows_dev(ctx->gate, S, I,
                         &dq())) return 0;
    if (!quant_matmul_launch(ctx->y,ctx->gate,down->weights,down->scales,
        down->fmt,S,I,D,row_bytes(down->fmt,I),down->gs,down->ng)) return 0;
    if (!cuda_ok(0, "expert MLP launch") ||
        !cuda_ok(COLI_SYCL_TRY(
                     dq().memcpy(y, ctx->y, yb).wait()),
                 "expert output download")) return 0;
    return 1;
}
catch (sycl::exception const &exc) { return sycl_caught(exc, "coli_cuda_expert_mlp"); }

/* The shared expert of a prompt chunk, int4 rows (fmt 2) at f16 activations: the gate and up pair
 * with the silu epilogue, then the down, through the matrix engines' GEMMs (coli_sycl_xmx_gemm_glu,
 * coli_sycl_xmx_gemm: the f16 joint_matrix tiles of coli_sycl_xmx.cpp, where the W4A16 WMMA kernels
 * were); 0 where the profile has no f16 shape or the family declines the widths (the caller's CPU
 * path, as on a device without tensor cores). */
extern "C" int coli_cuda_shared_mlp_w4a16(ColiCudaTensor *gate,
                                          ColiCudaTensor *up,
                                          ColiCudaTensor *down, float *y,
                                          const float *x, int S) try {
    if (fault_injected()) return 0;
    if(!gate||!up||!down||!x||!y||S<1||gate->fmt!=2||up->fmt!=2||down->fmt!=2||
       gate->device!=up->device||gate->device!=down->device||gate->I!=up->I||
       gate->O!=up->O||down->I!=gate->O||down->O!=gate->I)return 0;
    DeviceContext *ctx=find_ctx(gate->device);if(!select_ctx(ctx))return 0;
    int D=gate->I,I=gate->O;size_t xb=(size_t)S*D*sizeof(float),ib=(size_t)S*I*sizeof(float);
    if(!xmx_dense_ok(ctx,2,0,D,I)||!xmx_dense_ok(ctx,2,0,I,D)){ xmx_note("shared expert (prompt rows)", 0); return 0; }
    xmx_note("shared expert (prompt rows)", 1);
    if(!reserve(&ctx->x,&ctx->x_cap,xb)||!reserve(&ctx->gate,&ctx->gate_cap,ib)||
       !reserve(&ctx->up,&ctx->up_cap,ib)||!reserve(&ctx->y,&ctx->y_cap,xb)||
       !reserve_pinned(&ctx->host_x,&ctx->host_x_cap,xb)||
       !reserve_pinned(&ctx->host_y,&ctx->host_y_cap,xb))return 0;
    std::memcpy(ctx->host_x,x,xb);
    if (!cuda_ok(COLI_SYCL_TRY(ctx->stream->memcpy(ctx->x, ctx->host_x, xb)),
                 "shared w4a16 input upload")) return 0;
    const coli_sycl_profile &prof = coli_sycl_profile_for(gate->device);
    coli_sycl_gemm gg = {ctx->gate, I, ctx->x, D, 0, 0, gate->weights, gate->scales, 2, 0, S, D, I};   /* H = silu(x Wg) * (x Wu) */
    coli_sycl_gemm gd = {ctx->y, D, ctx->gate, I, 0, 0, down->weights, down->scales, 2, 0, S, I, D};   /* y = H Wd */
    if (!coli_sycl_xmx_gemm_glu(prof, *ctx->stream, gg, up->weights, up->scales) ||
        !coli_sycl_xmx_gemm(prof, *ctx->stream, gd)) return 0;
    if (!cuda_ok(0, "shared w4a16 launch") ||
        !cuda_ok(COLI_SYCL_TRY(ctx->stream->memcpy(ctx->host_y, ctx->y, xb)),
                 "shared w4a16 output download") ||
        !cuda_ok(COLI_SYCL_TRY(ctx->stream->wait()),
                 "shared w4a16 synchronize")) return 0;
    std::memcpy(y,ctx->host_y,xb);
    return 1;
}
catch (sycl::exception const &exc) { return sycl_caught(exc, "coli_cuda_shared_mlp_w4a16"); }

/* Single launch site for the fmt=8 group kernels, shared by the sync and the
 * issue/take dispatches so both stay one-line call sites (rebase-tolerant
 * against #935's e8_group_launch refactor of the adjacent branch).
 * COLI_CUDA_F8_WARP (docs/ENVIRONMENT.md, parsed by f8_warp_mode): 1 = warp
 * kernels (the CUDA default; HIP defaults to 0), 0 = the original per-(o,s)
 * kernels (field escape hatch, dense path included via quant_matmul_launch),
 * 2 = warp kernels decoding through cuda_fp8.h — a real cvt instruction only
 * on sm_89+, the header's bit-manip emulation below that, and plain =1
 * behavior where cuda_fp8.h is absent (HIP); experimental until the
 * 256-value sweep certifies it bit-identical on the target silicon. */
/* COLI_SYCL_F8_ASYNC_WAITS=1 (default 0): fewer host waits around an FP8 group on the in-order queue.
 * An issued group's descriptors go up from a page-locked copy without a wait (the stack array they
 * came from is not read after the call); the host SiLU's gate and up downloads share one wait; the
 * hidden upload is not waited for (the down kernel after it on the same queue reads it, and nothing
 * writes the host buffer before the group's take). The kernels and the host loop are the same, so
 * the bits are. */
/* A failure after a copy that was not waited for: the queue drained before the caller can reuse the host
 * buffers the copy reads or writes (host_desc, host_gate, host_x); errors there are already reported. */
static void f8_async_drain(DeviceContext *ctx) {
    try { ctx->stream->wait(); } catch (...) {}
}
static bool f8_async_waits_on(void) {
    static int v=-1;
    if(v<0){const char *e=std::getenv("COLI_SYCL_F8_ASYNC_WAITS");v=e&&e[0]=='1'&&!e[1];}
    return v!=0;
}
static bool f8_group_launch(DeviceContext *ctx,GroupDesc *dev,int I,int D,
                            int max_rows,int count,int total){
    int mode=f8_warp_mode();
    if(mode&&ctx->has_fp64){   /* the warp kernels sum their blocks in double */
        const bool cpu_order=!(I&127) && !(D&127) && f8_cpu_order_on();
        const bool cpu_silu=!(I&127) && !(D&127) && cpu_silu_on();
        const char *host_env=std::getenv("COLI_SYCL_F8_HOST_SILU");
        const bool host_silu=cpu_order&&host_env&&!std::strcmp(host_env,"1");
        size_t hidden_bytes=0;
        if(host_silu){
            if(total<1||I<1||(size_t)total>SIZE_MAX/(size_t)I/sizeof(float))return false;
            hidden_bytes=(size_t)total*I*sizeof(float);
            if(!reserve_pinned(&ctx->host_gate,&ctx->host_gate_cap,hidden_bytes)||
               !reserve_pinned(&ctx->host_up,&ctx->host_up_cap,hidden_bytes))return false;
            static std::atomic_flag noted=ATOMIC_FLAG_INIT; /* host logging only */
            if(!noted.test_and_set(std::memory_order_relaxed))
                std::fprintf(stderr,"[SYCL] fp8 host SiLU: GPU gate/up and down, host expf, synchronized copies\n");
        }
        if(cpu_order)f8_cpu_order_note(true,D,I,ctx->sg);
        if(cpu_silu&&!host_silu)cpu_silu_note(1);
        const int cmode=cpu_order ? f8_cpu_order_mode() : 0;
        const bool cpu2=cmode==2;               /* =2: every lane at work, 32 output rows a block */
        const bool cpu3=cmode==3 || cmode==4;   /* =3: a row's blocks four at a time, =1's grid (=4: its gate/up) */
        const bool down2=cmode==2 || cmode==4;  /* the down kernel of =2 (=4 too) */
        if(cpu3){
            static std::atomic_flag noted3=ATOMIC_FLAG_INIT; /* host logging only */
            if(!noted3.test_and_set(std::memory_order_relaxed))
                std::fprintf(stderr,"[SYCL] fp8 CPU order %d: the same FMA chains, tree and double sums; gate/up an output row's blocks %d at a time, down %s\n",
                             cmode,ctx->sg/8,down2?"a group of 8 lanes an output row (=2's)":"the same as gate/up");
        }
        if(cpu2){
            static std::atomic_flag noted2=ATOMIC_FLAG_INIT; /* host logging only */
            if(!noted2.test_and_set(std::memory_order_relaxed))
                std::fprintf(stderr,"[SYCL] fp8 CPU order 2: the same FMA chains, tree and double sums, a group of 8 lanes an output row\n");
        }
        const int wpb=ctx->wpb;   /* output rows a block: a sub-group each */
        dpct::dim3 hg((unsigned)((I + wpb - 1) / wpb), (unsigned)count),
            og((unsigned)((D + wpb - 1) / wpb), (unsigned)count);
        dpct::dim3 hg2((unsigned)((I + 31) / 32), (unsigned)count),
            og2((unsigned)((D + 31) / 32), (unsigned)count);
        const float *c_e4m3_ptr_ct1 = ctx->e4m3;
        {
            auto ctx_gate_ct0 = ctx->gate;
            auto ctx_up_ct1 = ctx->up;
            const float *ctx_x_ct2 = ctx->x;
            if(host_silu && cpu3) {
                COLI_SG_LAUNCH(*ctx->stream, ctx->sg, hg, 256, 0,
                    grouped_hidden_f8c3_dual<SG,false,true>(ctx_gate_ct0,ctx_up_ct1,ctx_x_ct2,dev,I,D,c_e4m3_ptr_ct1));
            } else if(host_silu && cpu2) {
                COLI_SG_LAUNCH(*ctx->stream, ctx->sg, hg2, 256, 0,
                    grouped_hidden_f8c2_dual<SG,false,true>(ctx_gate_ct0,ctx_up_ct1,ctx_x_ct2,dev,I,D,c_e4m3_ptr_ct1));
            } else if(host_silu) {
                COLI_SG_LAUNCH(*ctx->stream, ctx->sg, hg, 256, 0,
                    grouped_hidden_f8w_dual<0,SG,true,false,true>(ctx_gate_ct0,ctx_up_ct1,ctx_x_ct2,dev,I,D,c_e4m3_ptr_ct1));
            } else if(cpu_silu && cpu3) {
                COLI_SG_LAUNCH(*ctx->stream, ctx->sg, hg, 256, 0,
                    grouped_hidden_f8c3_dual<SG,true>(ctx_gate_ct0,ctx_up_ct1,ctx_x_ct2,dev,I,D,c_e4m3_ptr_ct1));
            } else if(cpu_silu && cpu2) {
                COLI_SG_LAUNCH(*ctx->stream, ctx->sg, hg2, 256, 0,
                    grouped_hidden_f8c2_dual<SG,true>(ctx_gate_ct0,ctx_up_ct1,ctx_x_ct2,dev,I,D,c_e4m3_ptr_ct1));
            } else if(cpu_silu && cpu_order) {
                COLI_SG_LAUNCH(*ctx->stream, ctx->sg, hg, 256, 0,
                    grouped_hidden_f8w_dual<0,SG,true,true>(ctx_gate_ct0,ctx_up_ct1,ctx_x_ct2,dev,I,D,c_e4m3_ptr_ct1));
            } else if(cpu_silu) {
                COLI_SG_LAUNCH(*ctx->stream, ctx->sg, hg, 256, 0,
                    grouped_hidden_f8w_dual<0,SG,false,true>(ctx_gate_ct0,ctx_up_ct1,ctx_x_ct2,dev,I,D,c_e4m3_ptr_ct1));
            } else if(cpu3) {
                COLI_SG_LAUNCH(*ctx->stream, ctx->sg, hg, 256, 0,
                    grouped_hidden_f8c3_dual<SG>(ctx_gate_ct0,ctx_up_ct1,ctx_x_ct2,dev,I,D,c_e4m3_ptr_ct1));
            } else if(cpu2) {
                COLI_SG_LAUNCH(*ctx->stream, ctx->sg, hg2, 256, 0,
                    grouped_hidden_f8c2_dual<SG>(ctx_gate_ct0,ctx_up_ct1,ctx_x_ct2,dev,I,D,c_e4m3_ptr_ct1));
            } else if(cpu_order) {
                COLI_SG_LAUNCH(*ctx->stream, ctx->sg, hg, 256, 0,
                    grouped_hidden_f8w_dual<0,SG,true>(ctx_gate_ct0,ctx_up_ct1,ctx_x_ct2,dev,I,D,c_e4m3_ptr_ct1));
            } else {
                COLI_SG_LAUNCH(*ctx->stream, ctx->sg, hg, 256, 0,
                               grouped_hidden_f8w_dual<0, SG>(ctx_gate_ct0, ctx_up_ct1, ctx_x_ct2, dev, I, D, c_e4m3_ptr_ct1));
            }
        }
        if(host_silu){
            /* Explicit copies and waits keep the GPU from accessing host data
             * while the CPU transforms it; no device atomics on host memory. */
            const bool aw=f8_async_waits_on();
            if(aw){   /* both downloads queued, one wait (in order: the up copy ends after the gate copy) */
                if(!cuda_ok(COLI_SYCL_TRY(ctx->stream->memcpy(ctx->host_gate,ctx->gate,hidden_bytes)),"fp8 host SiLU gate download")||
                   !cuda_ok(COLI_SYCL_TRY(ctx->stream->memcpy(ctx->host_up,ctx->up,hidden_bytes).wait()),"fp8 host SiLU up download")){
                    f8_async_drain(ctx); return false;   /* the gate download may still be writing host_gate */
                }
            } else if(!cuda_ok(COLI_SYCL_TRY(ctx->stream->memcpy(ctx->host_gate,ctx->gate,hidden_bytes).wait()),"fp8 host SiLU gate download")||
               !cuda_ok(COLI_SYCL_TRY(ctx->stream->memcpy(ctx->host_up,ctx->up,hidden_bytes).wait()),"fp8 host SiLU up download"))return false;
            for(size_t j=0;j<(size_t)total*I;j++)ctx->host_gate[j]=coli_fp8_host_silu_mul(ctx->host_gate[j],ctx->host_up[j]);
            if(aw){   /* not waited: the down kernel after it reads it in queue order */
                if(!cuda_ok(COLI_SYCL_TRY(ctx->stream->memcpy(ctx->gate,ctx->host_gate,hidden_bytes)),"fp8 host SiLU hidden upload")){
                    f8_async_drain(ctx); return false;
                }
            } else if(!cuda_ok(COLI_SYCL_TRY(ctx->stream->memcpy(ctx->gate,ctx->host_gate,hidden_bytes).wait()),"fp8 host SiLU hidden upload"))return false;
        }
        {
            auto ctx_y_ct0 = ctx->y;
            const float *ctx_gate_ct1 = ctx->gate;
            if(cpu3 && !down2) {
                COLI_SG_LAUNCH(*ctx->stream, ctx->sg, og, 256, 0,
                    grouped_down_f8c3<SG>(ctx_y_ct0,ctx_gate_ct1,dev,D,I,c_e4m3_ptr_ct1));
            } else if(down2) {
                COLI_SG_LAUNCH(*ctx->stream, ctx->sg, og2, 256, 0,
                    grouped_down_f8c2<SG>(ctx_y_ct0,ctx_gate_ct1,dev,D,I,c_e4m3_ptr_ct1));
            } else if(cpu_order) {
                COLI_SG_LAUNCH(*ctx->stream, ctx->sg, og, 256, 0,
                    grouped_down_f8w<0,SG,true>(ctx_y_ct0,ctx_gate_ct1,dev,D,I,c_e4m3_ptr_ct1));
            } else {
                COLI_SG_LAUNCH(*ctx->stream, ctx->sg, og, 256, 0,
                               grouped_down_f8w<0, SG>(ctx_y_ct0, ctx_gate_ct1, dev, D, I, c_e4m3_ptr_ct1));
            }
        }
        return true;
    }
    dpct::dim3 hg((unsigned)I, (unsigned)max_rows, (unsigned)count),
        og((unsigned)D, (unsigned)max_rows, (unsigned)count);
    {

        ctx->stream->submit([&](sycl::handler &cgh) {
            const float *c_e4m3_ptr_ct1 = cur_ctx()->e4m3;

            auto ctx_gate_ct0 = ctx->gate;
            auto ctx_up_ct1 = ctx->up;
            const float *ctx_x_ct2 = ctx->x;

            cgh.parallel_for(sycl::nd_range<3>(hg * sycl::range<3>(1, 1, 256),
                                               sycl::range<3>(1, 1, 256)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 grouped_hidden_f8_dual(
                                     ctx_gate_ct0, ctx_up_ct1, ctx_x_ct2, dev,
                                     I, D, c_e4m3_ptr_ct1);
                             });
        });
    }
    {

        ctx->stream->submit([&](sycl::handler &cgh) {
            const float *c_e4m3_ptr_ct1 = cur_ctx()->e4m3;

            auto ctx_y_ct0 = ctx->y;
            const float *ctx_gate_ct1 = ctx->gate;

            cgh.parallel_for(sycl::nd_range<3>(og * sycl::range<3>(1, 1, 256),
                                               sycl::range<3>(1, 1, 256)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 grouped_down_f8(ctx_y_ct0, ctx_gate_ct1, dev,
                                                 D, I, c_e4m3_ptr_ct1);
                             });
        });
    }
    return true;
}

/* The kernels the CUDA chain runs a decode step's routed experts on (coli_cuda_moe_async: a
 * block of output rows of one (row, slot) pair a launch, its input row in shared memory,
 * 16-byte weight loads), for a small group whose experts share one format: every row of the
 * group a pair of its own (S rows, K 1), the experts a table of their own. The grouped
 * kernels further down take a block of 256 threads per output row and a byte a thread a
 * step: GLM-5.2's eight int4-g64 experts of 6144 x 2048 took 1165 us a step on an RTX PRO
 * 4000 with the round trip, these 537 without it. The sync and the async group take it alike,
 * so their bits stay one path's. COLI_CUDA_GROUP_MOE=0 keeps the grouped kernels. */
extern "C" int coli_cuda_moe_async(int device, const CcExpert *tab, const float *x, const int *ids, const float *w,
                                   float *hid, float *y, const float *hy, const CcMoe *p, void *stream);
static int group_moe_ok(const GroupDesc *host, int count, int total, int D, int I) {
    static int on = -1;
    if (on < 0) { const char *e = getenv("COLI_CUDA_GROUP_MOE"); on = !(e && *e == '0'); }
    if (!on || count < 1 || D < 32 || I < 32 || D % 32 || I % 32 || D > CC_MOE_DIM_MAX || I > CC_MOE_DIM_MAX) return 0;
    int f = host[0].gf, gs = host[0].ggs, fd = host[0].df;
    if (f >= 100) {   /* a GGUF's blocks: gate and up of one type, down of one, both with an integer step */
        if (fd < 100 || f >= 143 || fd >= 143 || !bdi_has(f - 100) || !bdi_has(fd - 100)) return 0;
        for (int c = 0; c < count; c++) if (host[c].gf != f || host[c].uf != f || host[c].df != fd) return 0;
        return 1;
    }
    if (!(f == 1 || f == 2 || (f == 4 && gs >= 16 && gs % 16 == 0))) return 0;
    for (int c = 0; c < count; c++)
        if (host[c].gf != f || host[c].uf != f || host[c].df != f ||
            (f == 4 && (host[c].ggs != gs || host[c].ugs != gs || host[c].dgs != gs))) return 0;
    return 1;
}
static uint64_t g_group_moe_calls;   /* groups group_moe_launch took (tests/test_grouped_g4_cuda.cu) */
/* the group's x already in ctx->x (its rows expert by expert); y into ctx->y the same way */
static int group_moe_launch(DeviceContext *ctx, int device,
                            const GroupDesc *host, int count, const int *rows,
                            int total, int D, int I, int pin) try {
    /* the table, then the rows' experts, then a weight of 1 a row: the chain's own sum of a
     * row's slots in rank order (K 1) puts each row's down output in its place, a prompt
     * chunk's from the tensor cores' sorted rows (more than CC_MOE_FRAME_ROWS rows) too */
    size_t tb = (size_t)count * sizeof(CcExpert), nb = tb + (size_t)total * (sizeof(int) + sizeof(float));
    if (!reserve_bytes(&ctx->moe_tab, &ctx->moe_tab_cap, nb) || !reserve_pinned(&ctx->moe_host, &ctx->moe_host_cap, nb) ||
        !reserve(&ctx->gate, &ctx->gate_cap, cc_moe_hid_floats(total, 1, D, I, pin ? 0 : count) * sizeof(float))) return 0;
    CcExpert *t = (CcExpert *)ctx->moe_host;
    int *ids = (int *)((char *)ctx->moe_host + tb);
    float *w1 = (float *)(ids + total);
    for (int c = 0, r = 0; c < count; c++) {
        t[c].g = host[c].g; t[c].u = host[c].u; t[c].d = host[c].d;
        t[c].gs = host[c].gs; t[c].us = host[c].us; t[c].ds = host[c].ds; t[c].planar = 0; t[c].host = 0;
        for (int k = 0; k < rows[c]; k++) { w1[r] = 1.f; ids[r++] = c; }
    }
    if (!cuda_ok(COLI_SYCL_TRY(
                     ctx->stream->memcpy(ctx->moe_tab, ctx->moe_host, nb)),
                 "expert group table")) return 0;
    CcMoe mp; std::memset(&mp, 0, sizeof mp);
    mp.S = total; mp.K = 1; mp.D = D; mp.I = I; mp.fmt = host[0].gf; mp.gs = host[0].gf == 4 ? host[0].ggs : 0;
    if (host[0].gf >= 100) mp.fmt_d = host[0].df;   /* a GGUF's down: its own type */
    /* pin (a speculation's verify, coli_cuda_expert_group_pinned): the warp kernels at any rows
     * (E 0), whose rows are a decode step's bits; else past CC_MOE_FRAME_ROWS the tensor cores */
    mp.parts = 3; mp.x_row = D; mp.i_row = 1; mp.w_row = 1; mp.y_row = D; mp.h_off = -1; mp.E = pin ? 0 : count;
    const int *dids = (const int *)((char *)ctx->moe_tab + tb);
    if (!coli_cuda_moe_async(device, (const CcExpert *)ctx->moe_tab, ctx->x, dids, (const float *)(dids + total),
                             ctx->gate, ctx->y, nullptr, &mp, (void *)ctx->stream)) return 0;
    g_group_moe_calls++;
    return 1;
}
catch (sycl::exception const &exc) { return sycl_caught(exc, "group_moe_launch"); }
static int expert_group_impl(ColiCudaTensor *const *gates,
                             ColiCudaTensor *const *ups,
                             ColiCudaTensor *const *downs, const int *rows,
                             int count, float *y, const float *x,
                             int pin_small_batch) try {
    if (fault_injected()) return 0;
    if (!gates || !ups || !downs || !rows || !x || !y || count < 1) return 0;
    ColiCudaTensor *first=gates[0];
    if (!first) return 0;
    int device=first->device,D=first->I,I=first->O,total=0,max_rows=0;
    GroupDesc host[64]; if(count>64) return 0;
    int all_s4=1,all_q4=1,any_g4=0,any_e8=0,all_e8=1,any_f8=0,all_f8=1;
    for(int c=0;c<count;c++){
        ColiCudaTensor *g=gates[c],*u=ups[c],*d=downs[c];
        if(!g||!u||!d||rows[c]<1||g->device!=device||u->device!=device||d->device!=device||
           g->I!=D||u->I!=D||g->O!=I||u->O!=I||d->I!=I||d->O!=D) return 0;
        host[c]={g->weights,u->weights,d->weights,g->scales,u->scales,d->scales,
                 g->fmt,u->fmt,d->fmt,rows[c],total,
                 g->gs,u->gs,d->gs};
        all_s4&=g->fmt==2&&u->fmt==2&&d->fmt==2;
        all_q4&=(g->fmt==2||g->fmt==4)&&(u->fmt==2||u->fmt==4)&&(d->fmt==2||d->fmt==4)&&
                !(g->gs&1)&&!(u->gs&1)&&!(d->gs&1);   /* even gs: a packed byte never straddles groups */
        any_g4|=g->fmt==4||u->fmt==4||d->fmt==4;
        any_e8|=g->fmt==6||u->fmt==6||d->fmt==6;
        all_e8&=g->fmt==6&&u->fmt==6&&d->fmt==6;
        any_f8|=g->fmt==8||u->fmt==8||d->fmt==8;
        all_f8&=g->fmt==8&&u->fmt==8&&d->fmt==8;
        total+=rows[c]; if(rows[c]>max_rows) max_rows=rows[c];
    }
    /* Mixed E8/FP8 groups cannot use a homogeneous grouped kernel. */
    if((any_e8&&!all_e8)||(any_f8&&!all_f8)){
        int off=0;
        for(int c=0;c<count;c++){
            if(!coli_cuda_expert_mlp(gates[c],ups[c],downs[c],
                    y+(size_t)off*D,x+(size_t)off*D,rows[c])) return 0;
            off+=rows[c];
        }
        { std::lock_guard<std::mutex> lock(g_group_stats_mu);
          g_group_calls++; g_group_experts+=(uint64_t)count; g_group_rows+=(uint64_t)total; }
        return 1;
    }
    DeviceContext *ctx=find_ctx(device); if(!select_ctx(ctx)) return 0;
    if(!prepare_group_weights(ctx,gates,ups,downs,count,host)) return 0;
    size_t xb=(size_t)total*D*sizeof(float), ib=(size_t)total*I*sizeof(float);
    if(!reserve(&ctx->x,&ctx->x_cap,xb)||!reserve(&ctx->y,&ctx->y_cap,xb)||
       !reserve(&ctx->gate,&ctx->gate_cap,ib)||!reserve(&ctx->up,&ctx->up_cap,ib)||
       !reserve_bytes(&ctx->group_desc,&ctx->group_desc_cap,(size_t)count*sizeof(GroupDesc))) return 0;
    int async=!getenv("COLI_CUDA_ASYNC")||atoi(getenv("COLI_CUDA_ASYNC"));
    if(async&&(!reserve_pinned(&ctx->host_x,&ctx->host_x_cap,xb)||
               !reserve_pinned(&ctx->host_y,&ctx->host_y_cap,xb)))return 0;
    int copy_desc =
        async ? COLI_SYCL_TRY(ctx->stream->memcpy(
                    ctx->group_desc, host, (size_t)count * sizeof(GroupDesc)).wait())
              : COLI_SYCL_TRY(dq()
                                     .memcpy(ctx->group_desc, host,
                                             (size_t)count * sizeof(GroupDesc))
                                     .wait());
    if(!cuda_ok(copy_desc,"expert group descriptors"))return 0;
    int profile=getenv("COLI_CUDA_PROFILE")&&atoi(getenv("COLI_CUDA_PROFILE"));
    sycl::event * ev[4] = {};
    if (profile) for (int i = 0; i < 4;
                      i++) if (!cuda_ok(COLI_SYCL_TRY(ev[i] =
                                                             new sycl::event()),
                                        "profile event")) {
        for(int j=0;j<i;j++) destroy_event(ev[j]); profile=0; break; }   /* (#B8) don't leak the events already created */
    if (profile) sync_barrier(ev[0], ctx->stream);
    if(async)std::memcpy(ctx->host_x,x,xb);
    int copy_x =
        async ? COLI_SYCL_TRY(ctx->stream->memcpy(ctx->x, ctx->host_x, xb))
              : COLI_SYCL_TRY(
                    dq().memcpy(ctx->x, x, xb).wait());
    if(!cuda_ok(copy_x,"expert group input upload")) return 0;
    if (profile) sync_barrier(ev[1], ctx->stream);
    GroupDesc *dev=(GroupDesc*)ctx->group_desc;
    if(all_e8){
        dpct::dim3 hg((unsigned)I, (unsigned)max_rows, (unsigned)count),
            og((unsigned)D, (unsigned)max_rows, (unsigned)count);
        {

            ctx->stream->submit([&](sycl::handler &cgh) {
                const uint8_t *c_e8_grid_acc_ct1 = cur_ctx()->e8_grid;

                auto ctx_gate_ct0 = ctx->gate;
                const float *ctx_x_ct1 = ctx->x;

                cgh.parallel_for(
                    sycl::nd_range<3>(hg * sycl::range<3>(1, 1, 256),
                                      sycl::range<3>(1, 1, 256)),
                    [=](sycl::nd_item<3> item_ct1) {
                        grouped_hidden_e8_dual(ctx_gate_ct0, ctx_x_ct1, dev, I,
                                               D, c_e8_grid_acc_ct1);
                    });
            });
        }
        if(!e8_rot_rows_dev(ctx->gate,total,I,ctx->stream))return 0;
        {

            ctx->stream->submit([&](sycl::handler &cgh) {
                const uint8_t *c_e8_grid_acc_ct1 = cur_ctx()->e8_grid;

                auto ctx_y_ct0 = ctx->y;
                const float *ctx_gate_ct1 = ctx->gate;

                cgh.parallel_for(
                    sycl::nd_range<3>(og * sycl::range<3>(1, 1, 256),
                                      sycl::range<3>(1, 1, 256)),
                    [=](sycl::nd_item<3> item_ct1) {
                        grouped_down_e8(ctx_y_ct0, ctx_gate_ct1, dev, D, I,
                                        c_e8_grid_acc_ct1);
                    });
            });
        }
    }else if(all_f8){
        /* fp8-e4m3 groups: silu fused in the dual epilogue, like the w4/g4 duals. */
        if(!f8_group_launch(ctx,dev,I,D,max_rows,count,total)){
            for(int i=0;i<4;i++)destroy_event(ev[i]);
            return 0;
        }
    }else if(!pin_small_batch&&all_s4&&xmx_dense_ok(ctx,2,0,D,I)&&xmx_dense_ok(ctx,2,0,I,D)&&
             !(getenv("COLI_CUDA_TC_W4A16")&&*getenv("COLI_CUDA_TC_W4A16")=='0')&&
             [&]{ int tc16_min=getenv("COLI_CUDA_TC_W4A16_MIN")?atoi(getenv("COLI_CUDA_TC_W4A16_MIN")):16;
                  for(int c=0;c<count;c++) if(rows[c]>=tc16_min) return 1;
                  return 0; }()){
        /* At least one expert has enough rows for a matrix-engine tile: its gate and up pair with
         * the silu epilogue, then its down, through coli_sycl_xmx_gemm_glu / coli_sycl_xmx_gemm (the
         * f16 joint_matrix GEMMs of coli_sycl_xmx.cpp, in place of the W4A16 WMMA kernels), an expert
         * at a time inside the stream; an expert below the threshold keeps the block kernels. Groups
         * where EVERY expert is below the threshold (decode: r=1) fall through
         * to the grouped-W4 path below — 3 launches for the whole group instead
         * of 4 per expert (#431: the launch flood measured at ~981 micro-kernels
         * per token came from decode riding this branch's per-expert fallback).
         * COLI_CUDA_TC_W4A16=0 keeps the block kernels for every row count. */
        xmx_note("expert group (prompt rows)", 1);
        const coli_sycl_profile &prof = coli_sycl_profile_for(device);
        int tc16_min=getenv("COLI_CUDA_TC_W4A16_MIN")?atoi(getenv("COLI_CUDA_TC_W4A16_MIN")):16;
        int off16=0;
        for(int c=0;c<count;c++){
            int r=rows[c];
            float *g16=ctx->gate+(size_t)off16*I,*u16=ctx->up+(size_t)off16*I;
            float *x16=ctx->x+(size_t)off16*D,*y16=ctx->y+(size_t)off16*D;
            if(r>=tc16_min){
                coli_sycl_gemm gg = {g16, I, x16, D, 0, 0, host[c].g, host[c].gs, 2, 0, r, D, I};   /* silu(x Wg) * (x Wu) into g16 */
                coli_sycl_gemm gd = {y16, D, g16, I, 0, 0, host[c].d, host[c].ds, 2, 0, r, I, D};
                if(!coli_sycl_xmx_gemm_glu(prof, *ctx->stream, gg, host[c].u, host[c].us) ||
                   !coli_sycl_xmx_gemm(prof, *ctx->stream, gd)) return 0;
            }else{
                /* piccoli batch: tile TC quasi vuoti + overhead di lancio — il
                 * kernel naive per-elemento resta piu' veloce (misurato in decode) */
                {

                    ctx->stream->submit([&](sycl::handler &cgh) {
                            const float *c_e4m3_ptr_ct1 = cur_ctx()->e4m3;

                            const uint8_t *c_e8_grid_acc_ct1 = cur_ctx()->e8_grid;

                            auto host_c_g_ct2 = host[c].g;
                            auto host_c_gs_ct3 = host[c].gs;
                            auto host_c_gf_ct4 = host[c].gf;
                            auto row_bytes_host_c_gf_D_ct8 =
                                row_bytes(host[c].gf, D);

                            cgh.parallel_for(
                                sycl::nd_range<3>(sycl::range<3>(1, (unsigned)r,
                                                                 (unsigned)I) *
                                                      sycl::range<3>(1, 1, 256),
                                                  sycl::range<3>(1, 1, 256)),
                                [=](sycl::nd_item<3> item_ct1) {
                                    quant_matmul(
                                        g16, x16, host_c_g_ct2, host_c_gs_ct3,
                                        host_c_gf_ct4, r, D, I,
                                        row_bytes_host_c_gf_D_ct8, 0, 1,
                                        c_e8_grid_acc_ct1, c_e4m3_ptr_ct1);
                                });
                        });
                }
                {

                    ctx->stream->submit([&](sycl::handler &cgh) {
                            const float *c_e4m3_ptr_ct1 = cur_ctx()->e4m3;

                            const uint8_t *c_e8_grid_acc_ct1 = cur_ctx()->e8_grid;

                            auto host_c_u_ct2 = host[c].u;
                            auto host_c_us_ct3 = host[c].us;
                            auto host_c_uf_ct4 = host[c].uf;
                            auto row_bytes_host_c_uf_D_ct8 =
                                row_bytes(host[c].uf, D);

                            cgh.parallel_for(
                                sycl::nd_range<3>(sycl::range<3>(1, (unsigned)r,
                                                                 (unsigned)I) *
                                                      sycl::range<3>(1, 1, 256),
                                                  sycl::range<3>(1, 1, 256)),
                                [=](sycl::nd_item<3> item_ct1) {
                                    quant_matmul(
                                        u16, x16, host_c_u_ct2, host_c_us_ct3,
                                        host_c_uf_ct4, r, D, I,
                                        row_bytes_host_c_uf_D_ct8, 0, 1,
                                        c_e8_grid_acc_ct1, c_e4m3_ptr_ct1);
                                });
                        });
                }
                ctx->stream->submit([&](sycl::handler &cgh) {
                    auto size_t_r_I_ct2 = (size_t)r * I;

                    cgh.parallel_for(
                        sycl::nd_range<3>(
                            sycl::range<3>(
                                1, 1, (unsigned)(((size_t)r * I + 255) / 256)) *
                                sycl::range<3>(1, 1, 256),
                            sycl::range<3>(1, 1, 256)),
                        [=](sycl::nd_item<3> item_ct1) {
                            silu_mul(g16, u16, size_t_r_I_ct2);
                        });
                });
                {

                    ctx->stream->submit([&](sycl::handler &cgh) {
                            const float *c_e4m3_ptr_ct1 = cur_ctx()->e4m3;

                            const uint8_t *c_e8_grid_acc_ct1 = cur_ctx()->e8_grid;

                            auto host_c_d_ct2 = host[c].d;
                            auto host_c_ds_ct3 = host[c].ds;
                            auto host_c_df_ct4 = host[c].df;
                            auto row_bytes_host_c_df_I_ct8 =
                                row_bytes(host[c].df, I);

                            cgh.parallel_for(
                                sycl::nd_range<3>(sycl::range<3>(1, (unsigned)r,
                                                                 (unsigned)D) *
                                                      sycl::range<3>(1, 1, 256),
                                                  sycl::range<3>(1, 1, 256)),
                                [=](sycl::nd_item<3> item_ct1) {
                                    quant_matmul(
                                        y16, g16, host_c_d_ct2, host_c_ds_ct3,
                                        host_c_df_ct4, r, I, D,
                                        row_bytes_host_c_df_I_ct8, 0, 1,
                                        c_e8_grid_acc_ct1, c_e4m3_ptr_ct1);
                                });
                        });
                }
            }
            off16+=r;
        }
    }else if(group_moe_ok(host,count,total,D,I)){
        if(!group_moe_launch(ctx,device,host,count,rows,total,D,I,pin_small_batch))return 0;
    }else if(all_s4&&(!getenv("COLI_CUDA_W4_PACKED")||atoi(getenv("COLI_CUDA_W4_PACKED")))){
        dpct::dim3 hg((unsigned)I, (unsigned)max_rows, (unsigned)count),
            og((unsigned)D, (unsigned)max_rows, (unsigned)count);
        int dual=!getenv("COLI_CUDA_DUAL_PROJ")||atoi(getenv("COLI_CUDA_DUAL_PROJ"));
            if (dual)ctx->stream->submit([&](sycl::handler &cgh) {
                    auto ctx_gate_ct0 = ctx->gate;
                    auto ctx_up_ct1 = ctx->up;
                    const float *ctx_x_ct2 = ctx->x;

                    cgh.parallel_for(
                        sycl::nd_range<3>(hg * sycl::range<3>(1, 1, 256),
                                          sycl::range<3>(1, 1, 256)),
                        [=](sycl::nd_item<3> item_ct1) {
                            grouped_hidden_w4_dual(ctx_gate_ct0, ctx_up_ct1,
                                                   ctx_x_ct2, dev, I, D);
                        });
                });
        else{   /* non-dual path has no fused epilogue: silu stays a kernel here */
            ctx->stream->submit([&](sycl::handler &cgh) {
                auto ctx_gate_ct0 = ctx->gate;
                const float *ctx_x_ct1 = ctx->x;

                cgh.parallel_for(
                    sycl::nd_range<3>(hg * sycl::range<3>(1, 1, 256),
                                      sycl::range<3>(1, 1, 256)),
                    [=](sycl::nd_item<3> item_ct1) {
                        grouped_hidden_w4(ctx_gate_ct0, ctx_x_ct1, dev, I, D,
                                          0);
                    });
            });
            ctx->stream->submit([&](sycl::handler &cgh) {
                auto ctx_up_ct0 = ctx->up;
                const float *ctx_x_ct1 = ctx->x;

                cgh.parallel_for(
                    sycl::nd_range<3>(hg * sycl::range<3>(1, 1, 256),
                                      sycl::range<3>(1, 1, 256)),
                    [=](sycl::nd_item<3> item_ct1) {
                        grouped_hidden_w4(ctx_up_ct0, ctx_x_ct1, dev, I, D, 1);
                    });
            });
            ctx->stream->submit([&](sycl::handler &cgh) {
                auto ctx_gate_ct0 = ctx->gate;
                const float *ctx_up_ct1 = ctx->up;
                auto size_t_total_I_ct2 = (size_t)total * I;

                cgh.parallel_for(
                    sycl::nd_range<3>(
                        sycl::range<3>(
                            1, 1, (unsigned)(((size_t)total * I + 255) / 256)) *
                            sycl::range<3>(1, 1, 256),
                        sycl::range<3>(1, 1, 256)),
                    [=](sycl::nd_item<3> item_ct1) {
                        silu_mul(ctx_gate_ct0, ctx_up_ct1, size_t_total_I_ct2);
                    });
            });
        }
        ctx->stream->submit([&](sycl::handler &cgh) {
            auto ctx_y_ct0 = ctx->y;
            const float *ctx_gate_ct1 = ctx->gate;

            cgh.parallel_for(sycl::nd_range<3>(og * sycl::range<3>(1, 1, 256),
                                               sycl::range<3>(1, 1, 256)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 grouped_down_w4(ctx_y_ct0, ctx_gate_ct1, dev,
                                                 D, I);
                             });
        });
    }else if(all_q4&&any_g4){
        /* grouped-int4 (fmt=4) present: per-group scales (#334). fmt=2 members
         * ride along as the ng=1 special case. silu fused in the dual epilogue. */
        dpct::dim3 hg((unsigned)I, (unsigned)max_rows, (unsigned)count),
            og((unsigned)D, (unsigned)max_rows, (unsigned)count);
        ctx->stream->submit([&](sycl::handler &cgh) {
            auto ctx_gate_ct0 = ctx->gate;
            auto ctx_up_ct1 = ctx->up;
            const float *ctx_x_ct2 = ctx->x;

            cgh.parallel_for(sycl::nd_range<3>(hg * sycl::range<3>(1, 1, 256),
                                               sycl::range<3>(1, 1, 256)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 grouped_hidden_g4_dual(ctx_gate_ct0,
                                                        ctx_up_ct1, ctx_x_ct2,
                                                        dev, I, D);
                             });
        });
        ctx->stream->submit([&](sycl::handler &cgh) {
            auto ctx_y_ct0 = ctx->y;
            const float *ctx_gate_ct1 = ctx->gate;

            cgh.parallel_for(sycl::nd_range<3>(og * sycl::range<3>(1, 1, 256),
                                               sycl::range<3>(1, 1, 256)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 grouped_down_g4(ctx_y_ct0, ctx_gate_ct1, dev,
                                                 D, I);
                             });
        });
    }else{
        /* generic path decodes fmt 0/1/2/3 only — refuse everything else rather
         * than whitelist known offenders: a fmt=4 group that slipped the gates
         * above (odd gs) must NOT be silently decoded as int2 (#334), and any
         * group/block-scaled format that gains CUDA tensors later (fmt=5, fmt=8)
         * carries scale geometry this kernel's epilogue does not apply.
         * STRICTER than weight_at's own admissible set on purpose: fmt=4 belongs
         * on the g4 path above, and returning 0 here keeps the CORRECT CPU
         * fallback, which is the outcome we want — weight_at's device-side
         * __trap backstop is for a launch that got past a gate like this one,
         * not a substitute for having the gate. */
        for(int c=0;c<count;c++)
            if(host[c].gf>3||host[c].uf>3||host[c].df>3) return 0;
        dpct::dim3 hg((unsigned)I, (unsigned)max_rows, (unsigned)count),
            og((unsigned)D, (unsigned)max_rows, (unsigned)count);
        {

            ctx->stream->submit([&](sycl::handler &cgh) {
                const float *c_e4m3_ptr_ct1 = cur_ctx()->e4m3;

                auto ctx_gate_ct0 = ctx->gate;
                const float *ctx_x_ct1 = ctx->x;

                cgh.parallel_for(
                    sycl::nd_range<3>(hg * sycl::range<3>(1, 1, 256),
                                      sycl::range<3>(1, 1, 256)),
                    [=](sycl::nd_item<3> item_ct1) {
                        grouped_hidden(ctx_gate_ct0, ctx_x_ct1, dev, I, D, 0,
                                       c_e4m3_ptr_ct1);
                    });
            });
        }
        {

            ctx->stream->submit([&](sycl::handler &cgh) {
                const float *c_e4m3_ptr_ct1 = cur_ctx()->e4m3;

                auto ctx_up_ct0 = ctx->up;
                const float *ctx_x_ct1 = ctx->x;

                cgh.parallel_for(
                    sycl::nd_range<3>(hg * sycl::range<3>(1, 1, 256),
                                      sycl::range<3>(1, 1, 256)),
                    [=](sycl::nd_item<3> item_ct1) {
                        grouped_hidden(ctx_up_ct0, ctx_x_ct1, dev, I, D, 1,
                                       c_e4m3_ptr_ct1);
                    });
            });
        }
        ctx->stream->submit([&](sycl::handler &cgh) {
            auto ctx_gate_ct0 = ctx->gate;
            const float *ctx_up_ct1 = ctx->up;
            auto size_t_total_I_ct2 = (size_t)total * I;

            cgh.parallel_for(
                sycl::nd_range<3>(
                    sycl::range<3>(
                        1, 1, (unsigned)(((size_t)total * I + 255) / 256)) *
                        sycl::range<3>(1, 1, 256),
                    sycl::range<3>(1, 1, 256)),
                [=](sycl::nd_item<3> item_ct1) {
                    silu_mul(ctx_gate_ct0, ctx_up_ct1, size_t_total_I_ct2);
                });
        });
        {

            ctx->stream->submit([&](sycl::handler &cgh) {
                const float *c_e4m3_ptr_ct1 = cur_ctx()->e4m3;

                auto ctx_y_ct0 = ctx->y;
                const float *ctx_gate_ct1 = ctx->gate;

                cgh.parallel_for(
                    sycl::nd_range<3>(og * sycl::range<3>(1, 1, 256),
                                      sycl::range<3>(1, 1, 256)),
                    [=](sycl::nd_item<3> item_ct1) {
                        grouped_down(ctx_y_ct0, ctx_gate_ct1, dev, D, I,
                                     c_e4m3_ptr_ct1);
                    });
            });
        }
    }
    if (profile) sync_barrier(ev[2], ctx->stream);
    if (!async && !cuda_ok(COLI_SYCL_TRY(ctx->stream->wait()),
                           "expert group synchronize")) return 0;
    int copy_y =
        async ? COLI_SYCL_TRY(ctx->stream->memcpy(ctx->host_y, ctx->y, xb))
              : COLI_SYCL_TRY(
                    dq().memcpy(y, ctx->y, xb).wait());
    if (!cuda_ok(0, "expert group launch") ||
        !cuda_ok(copy_y, "expert group output download")) return 0;
    if (async) {
        if (!cuda_ok(COLI_SYCL_TRY(ctx->stream->wait()),
                     "expert group synchronize")) return 0;
        std::memcpy(y,ctx->host_y,xb);}
    if(profile){
        sync_barrier(ev[3], ctx->stream); ev[3]->wait_and_throw();
            float a = 0, b = 0, c = 0;
        try {   /* event timestamps need profiling queues (COLI_SYCL_PROFILING=1): else the times stay 0 */
            auto ms = [](sycl::event *from, sycl::event *to) {
                return (to->get_profiling_info<sycl::info::event_profiling::command_end>() -
                        from->get_profiling_info<sycl::info::event_profiling::command_start>()) / 1000000.0f;
            };
            a = ms(ev[0], ev[1]); b = ms(ev[1], ev[2]); c = ms(ev[2], ev[3]);
        } catch (sycl::exception const &) {}
        { std::lock_guard<std::mutex> lock(g_group_stats_mu);
          int index=(int)(ctx-g_ctx);
          g_group_h2d_ms+=a; g_group_kernel_ms+=b; g_group_d2h_ms+=c;
          g_device_group_h2d_ms[index]+=a;
          g_device_group_kernel_ms[index]+=b;
          g_device_group_d2h_ms[index]+=c; }
        for (int i = 0; i < 4; i++) destroy_event(ev[i]);
    }
    { std::lock_guard<std::mutex> lock(g_group_stats_mu);
      int index=(int)(ctx-g_ctx);
      g_group_calls++; g_group_experts+=(uint64_t)count; g_group_rows+=(uint64_t)total;
      g_device_group_calls[index]++; g_device_group_experts[index]+=(uint64_t)count;
      g_device_group_rows[index]+=(uint64_t)total; }
    return 1;
}
catch (sycl::exception const &exc) { return sycl_caught(exc, "expert_group_impl"); }

extern "C" int coli_cuda_expert_group(ColiCudaTensor *const *gates,
                                        ColiCudaTensor *const *ups,
                                        ColiCudaTensor *const *downs,
                                        const int *rows, int count,
                                        float *y, const float *x) {
    return expert_group_impl(gates,ups,downs,rows,count,y,x,0);
}

extern "C" int coli_cuda_expert_group_pinned(ColiCudaTensor *const *gates,
                                               ColiCudaTensor *const *ups,
                                               ColiCudaTensor *const *downs,
                                               const int *rows, int count,
                                               float *y, const float *x,
                                               int pin_small_batch) {
    return expert_group_impl(gates,ups,downs,rows,count,y,x,pin_small_batch);
}

/* ---- Async expert group (Inc.4): issue/take split of coli_cuda_expert_group ----
 * The measured cost of the sync call at decode is ~0.45 ms/call of HOST-side wait
 * (stream sync + staging), vs ~0.18 ms of actual GPU work — 70% tax, paid ~5x per
 * layer because a token's 8 experts scatter across devices. issue() stages and
 * launches on the device stream and returns immediately; take() syncs and hands
 * back the pinned result rows. One issue may be outstanding per device; moe()
 * takes at each layer end, which also orders the next layer's reuse of the ctx
 * scratch buffers. Small batches only (decode/spec): bigger totals keep the sync
 * path with its TC variants. Numerics are the sync path's small-batch kernels,
 * so greedy output is byte-identical by construction. */
extern "C" int coli_cuda_expert_group_issue(ColiCudaTensor *const *gates,
                                            ColiCudaTensor *const *ups,
                                            ColiCudaTensor *const *downs,
                                            const int *rows, int count,
                                            const float *x) try {
    if (!gates || !ups || !downs || !rows || !x || count < 1 || count > 64) return 0;
    ColiCudaTensor *first=gates[0];
    if (!first) return 0;
    int device=first->device,D=first->I,I=first->O,total=0,max_rows=0,all_s4=1,any_e8=0,all_e8=1,
        all_q4=1,any_g4=0,any_f8=0,all_f8=1;
    GroupDesc host[64];
    for(int c=0;c<count;c++){
        ColiCudaTensor *g=gates[c],*u=ups[c],*d=downs[c];
        if(!g||!u||!d||rows[c]<1||g->device!=device||u->device!=device||d->device!=device||
           g->I!=D||u->I!=D||g->O!=I||u->O!=I||d->I!=I||d->O!=D) return 0;
        host[c]={g->weights,u->weights,d->weights,g->scales,u->scales,d->scales,
                 g->fmt,u->fmt,d->fmt,rows[c],total,
                 g->gs,u->gs,d->gs};
        all_s4&=g->fmt==2&&u->fmt==2&&d->fmt==2;
        any_e8|=g->fmt==6||u->fmt==6||d->fmt==6;
        all_e8&=g->fmt==6&&u->fmt==6&&d->fmt==6;
        all_q4&=(g->fmt==2||g->fmt==4)&&(u->fmt==2||u->fmt==4)&&(d->fmt==2||d->fmt==4)&&
                !(g->gs&1)&&!(u->gs&1)&&!(d->gs&1);   /* even gs: a packed byte never straddles groups */
        any_g4|=g->fmt==4||u->fmt==4||d->fmt==4;
        any_f8|=g->fmt==8||u->fmt==8||d->fmt==8;
        all_f8&=g->fmt==8&&u->fmt==8&&d->fmt==8;
        total+=rows[c]; if(rows[c]>max_rows) max_rows=rows[c];
    }
    if(any_e8&&!all_e8) return 0;
    if(any_f8&&!all_f8) return 0;   /* mixed FP8: no homogeneous kernel, sync path has the per-expert loop */
    /* Decode scale: the qwen36 tier issues one group per device per token, up to its
     * QT_MAX_ROWS (32) rows, i.e. a model's top-k when every routed expert is resident
     * on one card (Qwen3.8 routes 10). Above 32 the caller keeps the CPU path. */
    if(total>32) return 0;
    /* COLI_SYCL_F8_ASYNC_WAITS: no issue after a lost device (as the sync path), whose unwaited copies may be
     * in flight */
    const bool aw=all_f8 && f8_async_waits_on();
    if(aw && fault_injected()) return 0;
    DeviceContext *ctx=find_ctx(device); if(!ctx||ctx->group_pending||!select_ctx(ctx)) return 0;
    if(!prepare_group_weights(ctx,gates,ups,downs,count,host)) return 0;
    size_t xb=(size_t)total*D*sizeof(float), ib=(size_t)total*I*sizeof(float);
    if(!reserve(&ctx->x,&ctx->x_cap,xb)||!reserve(&ctx->y,&ctx->y_cap,xb)||
       !reserve(&ctx->gate,&ctx->gate_cap,ib)||!reserve(&ctx->up,&ctx->up_cap,ib)||
       !reserve_bytes(&ctx->group_desc,&ctx->group_desc_cap,(size_t)count*sizeof(GroupDesc))||
       !reserve_pinned(&ctx->host_x,&ctx->host_x_cap,xb)||
       !reserve_pinned(&ctx->host_y,&ctx->host_y_cap,xb)) return 0;
    std::memcpy(ctx->host_x,x,xb);
    const size_t descb=(size_t)count*sizeof(GroupDesc);
    bool unwaited=false;   /* a failure past this copy drains the queue first (f8_async_drain) */
    if (aw && reserve_pinned(&ctx->host_desc,&ctx->host_desc_cap,descb)) {
        /* from a page-locked copy, not waited (the take waits before the next issue reuses it) */
        std::memcpy(ctx->host_desc, host, descb);
        unwaited=true;
        if (!cuda_ok(COLI_SYCL_TRY(ctx->stream->memcpy(ctx->group_desc, ctx->host_desc, descb)),
                     "expert group issue descriptors")) { f8_async_drain(ctx); return 0; }
    } else if (!cuda_ok(COLI_SYCL_TRY(ctx->stream->memcpy(
                     ctx->group_desc, host, descb).wait()),
                 "expert group issue descriptors")) return 0;
    if (!cuda_ok(COLI_SYCL_TRY(ctx->stream->memcpy(ctx->x, ctx->host_x, xb)),
                 "expert group issue upload")) { if(unwaited) f8_async_drain(ctx); return 0; }
    if(all_e8){
        GroupDesc *dev=(GroupDesc*)ctx->group_desc;
        dpct::dim3 hg((unsigned)I, (unsigned)max_rows, (unsigned)count);
        dpct::dim3 og((unsigned)D, (unsigned)max_rows, (unsigned)count);
        {

            ctx->stream->submit([&](sycl::handler &cgh) {
                const uint8_t *c_e8_grid_acc_ct1 = cur_ctx()->e8_grid;

                auto ctx_gate_ct0 = ctx->gate;
                const float *ctx_x_ct1 = ctx->x;

                cgh.parallel_for(
                    sycl::nd_range<3>(hg * sycl::range<3>(1, 1, 256),
                                      sycl::range<3>(1, 1, 256)),
                    [=](sycl::nd_item<3> item_ct1) {
                        grouped_hidden_e8_dual(ctx_gate_ct0, ctx_x_ct1, dev, I,
                                               D, c_e8_grid_acc_ct1);
                    });
            });
        }
        if(!e8_rot_rows_dev(ctx->gate,total,I,ctx->stream))return 0;
        {

            ctx->stream->submit([&](sycl::handler &cgh) {
                const uint8_t *c_e8_grid_acc_ct1 = cur_ctx()->e8_grid;

                auto ctx_y_ct0 = ctx->y;
                const float *ctx_gate_ct1 = ctx->gate;

                cgh.parallel_for(
                    sycl::nd_range<3>(og * sycl::range<3>(1, 1, 256),
                                      sycl::range<3>(1, 1, 256)),
                    [=](sycl::nd_item<3> item_ct1) {
                        grouped_down_e8(ctx_y_ct0, ctx_gate_ct1, dev, D, I,
                                        c_e8_grid_acc_ct1);
                    });
            });
        }
    }else if(all_f8){
        /* fp8-e4m3 groups on the async decode path: same launch helper as the
         * sync dispatch, silu fused in the dual epilogue. */
        if(!f8_group_launch(ctx,(GroupDesc*)ctx->group_desc,I,D,max_rows,count,total)){ if(unwaited) f8_async_drain(ctx); return 0; }
    }else if(group_moe_ok(host,count,total,D,I)){
        /* pin 1: the warp kernels at any row count, so 9..32 rows give the bits of the
         * same experts issued 8 rows at a time (the tensor-core family past
         * CC_MOE_FRAME_ROWS is for prompt chunks, not a decode step). */
        if(!group_moe_launch(ctx,device,host,count,rows,total,D,I,1))return 0;
    }else if(all_s4&&(!getenv("COLI_CUDA_W4_PACKED")||atoi(getenv("COLI_CUDA_W4_PACKED")))){
        GroupDesc *dev=(GroupDesc*)ctx->group_desc;
        dpct::dim3 hg((unsigned)I, (unsigned)max_rows, (unsigned)count);
        dpct::dim3 og((unsigned)D, (unsigned)max_rows, (unsigned)count);
        int dual=!getenv("COLI_CUDA_DUAL_PROJ")||atoi(getenv("COLI_CUDA_DUAL_PROJ"));
            if (dual)ctx->stream->submit([&](sycl::handler &cgh) {
                    auto ctx_gate_ct0 = ctx->gate;
                    auto ctx_up_ct1 = ctx->up;
                    const float *ctx_x_ct2 = ctx->x;

                    cgh.parallel_for(
                        sycl::nd_range<3>(hg * sycl::range<3>(1, 1, 256),
                                          sycl::range<3>(1, 1, 256)),
                        [=](sycl::nd_item<3> item_ct1) {
                            grouped_hidden_w4_dual(ctx_gate_ct0, ctx_up_ct1,
                                                   ctx_x_ct2, dev, I, D);
                        });
                });
        else {
            ctx->stream->submit([&](sycl::handler &cgh) {
                auto ctx_gate_ct0 = ctx->gate;
                const float *ctx_x_ct1 = ctx->x;

                cgh.parallel_for(
                    sycl::nd_range<3>(hg * sycl::range<3>(1, 1, 256),
                                      sycl::range<3>(1, 1, 256)),
                    [=](sycl::nd_item<3> item_ct1) {
                        grouped_hidden_w4(ctx_gate_ct0, ctx_x_ct1, dev, I, D,
                                          0);
                    });
            });
            ctx->stream->submit([&](sycl::handler &cgh) {
                auto ctx_up_ct0 = ctx->up;
                const float *ctx_x_ct1 = ctx->x;

                cgh.parallel_for(
                    sycl::nd_range<3>(hg * sycl::range<3>(1, 1, 256),
                                      sycl::range<3>(1, 1, 256)),
                    [=](sycl::nd_item<3> item_ct1) {
                        grouped_hidden_w4(ctx_up_ct0, ctx_x_ct1, dev, I, D, 1);
                    });
            });
            ctx->stream->submit([&](sycl::handler &cgh) {
                auto ctx_gate_ct0 = ctx->gate;
                const float *ctx_up_ct1 = ctx->up;
                auto size_t_total_I_ct2 = (size_t)total * I;

                cgh.parallel_for(
                    sycl::nd_range<3>(
                        sycl::range<3>(
                            1, 1, (unsigned)(((size_t)total * I + 255) / 256)) *
                            sycl::range<3>(1, 1, 256),
                        sycl::range<3>(1, 1, 256)),
                    [=](sycl::nd_item<3> item_ct1) {
                        silu_mul(ctx_gate_ct0, ctx_up_ct1, size_t_total_I_ct2);
                    });
            });
        }
        ctx->stream->submit([&](sycl::handler &cgh) {
            auto ctx_y_ct0 = ctx->y;
            const float *ctx_gate_ct1 = ctx->gate;

            cgh.parallel_for(sycl::nd_range<3>(og * sycl::range<3>(1, 1, 256),
                                               sycl::range<3>(1, 1, 256)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 grouped_down_w4(ctx_y_ct0, ctx_gate_ct1, dev,
                                                 D, I);
                             });
        });
    } else if(all_q4&&any_g4){
        /* grouped int4 (fmt=4) present in the async decode path: per-group
         * scales via the #334 kernels (fmt=2 members ride along as ng=1). The
         * previous fallback ran quant_matmul with gs=0,ng=1, which silently
         * applied one per-row scale to a grouped container -> wrong output. */
        GroupDesc *dev=(GroupDesc*)ctx->group_desc;
        dpct::dim3 hg((unsigned)I, (unsigned)max_rows, (unsigned)count);
        dpct::dim3 og((unsigned)D, (unsigned)max_rows, (unsigned)count);
        /* silu is fused in the dual kernel's epilogue (like the sync path):
         * an extra silu_mul here would re-apply it against the never-written
         * ctx->up buffer. */
        ctx->stream->submit([&](sycl::handler &cgh) {
            auto ctx_gate_ct0 = ctx->gate;
            auto ctx_up_ct1 = ctx->up;
            const float *ctx_x_ct2 = ctx->x;

            cgh.parallel_for(sycl::nd_range<3>(hg * sycl::range<3>(1, 1, 256),
                                               sycl::range<3>(1, 1, 256)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 grouped_hidden_g4_dual(ctx_gate_ct0,
                                                        ctx_up_ct1, ctx_x_ct2,
                                                        dev, I, D);
                             });
        });
        ctx->stream->submit([&](sycl::handler &cgh) {
            auto ctx_y_ct0 = ctx->y;
            const float *ctx_gate_ct1 = ctx->gate;

            cgh.parallel_for(sycl::nd_range<3>(og * sycl::range<3>(1, 1, 256),
                                               sycl::range<3>(1, 1, 256)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 grouped_down_g4(ctx_y_ct0, ctx_gate_ct1, dev,
                                                 D, I);
                             });
        });
    } else {
        /* Fallback runs quant_matmul with gs=0,ng=1 — per-row-scale semantics.
         * That is only correct for fmt 0/1/2/3: refuse group/block-scaled
         * members (fmt=4 with odd gs today; fmt=5/8 if they ever gain CUDA
         * tensors) instead of silently mis-scaling them, mirroring the sync
         * path's refusal (#334). fmt=6 cannot reach here (any_e8 gates above). */
        for(int c=0;c<count;c++)
            if(host[c].gf>3||host[c].uf>3||host[c].df>3) return 0;
        for(int c=0;c<count;c++){
        int r=rows[c];
        float *g16=ctx->gate+(size_t)host[c].offset*I,*u16=ctx->up+(size_t)host[c].offset*I;
        float *x16=ctx->x+(size_t)host[c].offset*D,*y16=ctx->y+(size_t)host[c].offset*D;
            {

                ctx->stream->submit([&](sycl::handler &cgh) {
                    const float *c_e4m3_ptr_ct1 = cur_ctx()->e4m3;

                    const uint8_t *c_e8_grid_acc_ct1 = cur_ctx()->e8_grid;

                    auto host_c_g_ct2 = host[c].g;
                    auto host_c_gs_ct3 = host[c].gs;
                    auto host_c_gf_ct4 = host[c].gf;
                    auto row_bytes_host_c_gf_D_ct8 = row_bytes(host[c].gf, D);

                    cgh.parallel_for(
                        sycl::nd_range<3>(
                            sycl::range<3>(1, (unsigned)r, (unsigned)I) *
                                sycl::range<3>(1, 1, 256),
                            sycl::range<3>(1, 1, 256)),
                        [=](sycl::nd_item<3> item_ct1) {
                            quant_matmul(g16, x16, host_c_g_ct2, host_c_gs_ct3,
                                         host_c_gf_ct4, r, D, I,
                                         row_bytes_host_c_gf_D_ct8, 0, 1,
                                         c_e8_grid_acc_ct1, c_e4m3_ptr_ct1);
                        });
                });
            }
            {

                ctx->stream->submit([&](sycl::handler &cgh) {
                    const float *c_e4m3_ptr_ct1 = cur_ctx()->e4m3;

                    const uint8_t *c_e8_grid_acc_ct1 = cur_ctx()->e8_grid;

                    auto host_c_u_ct2 = host[c].u;
                    auto host_c_us_ct3 = host[c].us;
                    auto host_c_uf_ct4 = host[c].uf;
                    auto row_bytes_host_c_uf_D_ct8 = row_bytes(host[c].uf, D);

                    cgh.parallel_for(
                        sycl::nd_range<3>(
                            sycl::range<3>(1, (unsigned)r, (unsigned)I) *
                                sycl::range<3>(1, 1, 256),
                            sycl::range<3>(1, 1, 256)),
                        [=](sycl::nd_item<3> item_ct1) {
                            quant_matmul(u16, x16, host_c_u_ct2, host_c_us_ct3,
                                         host_c_uf_ct4, r, D, I,
                                         row_bytes_host_c_uf_D_ct8, 0, 1,
                                         c_e8_grid_acc_ct1, c_e4m3_ptr_ct1);
                        });
                });
            }
            ctx->stream->submit([&](sycl::handler &cgh) {
                auto size_t_r_I_ct2 = (size_t)r * I;

                cgh.parallel_for(
                    sycl::nd_range<3>(
                        sycl::range<3>(
                            1, 1, (unsigned)(((size_t)r * I + 255) / 256)) *
                            sycl::range<3>(1, 1, 256),
                        sycl::range<3>(1, 1, 256)),
                    [=](sycl::nd_item<3> item_ct1) {
                        silu_mul(g16, u16, size_t_r_I_ct2);
                    });
            });
            {

                ctx->stream->submit([&](sycl::handler &cgh) {
                    const float *c_e4m3_ptr_ct1 = cur_ctx()->e4m3;

                    const uint8_t *c_e8_grid_acc_ct1 = cur_ctx()->e8_grid;

                    auto host_c_d_ct2 = host[c].d;
                    auto host_c_ds_ct3 = host[c].ds;
                    auto host_c_df_ct4 = host[c].df;
                    auto row_bytes_host_c_df_I_ct8 = row_bytes(host[c].df, I);

                    cgh.parallel_for(
                        sycl::nd_range<3>(
                            sycl::range<3>(1, (unsigned)r, (unsigned)D) *
                                sycl::range<3>(1, 1, 256),
                            sycl::range<3>(1, 1, 256)),
                        [=](sycl::nd_item<3> item_ct1) {
                            quant_matmul(y16, g16, host_c_d_ct2, host_c_ds_ct3,
                                         host_c_df_ct4, r, I, D,
                                         row_bytes_host_c_df_I_ct8, 0, 1,
                                         c_e8_grid_acc_ct1, c_e4m3_ptr_ct1);
                        });
                });
            }
    }}
    if (!cuda_ok(0, "expert group issue launch") ||
        !cuda_ok(COLI_SYCL_TRY(ctx->stream->memcpy(ctx->host_y, ctx->y, xb)),
                 "expert group issue download")) { if(unwaited) f8_async_drain(ctx); return 0; }
    ctx->group_pending=1; ctx->group_pending_bytes=xb;
    { std::lock_guard<std::mutex> lock(g_group_stats_mu);
      int index=(int)(ctx-g_ctx);
      g_group_calls++; g_group_experts+=(uint64_t)count; g_group_rows+=(uint64_t)total;
      g_device_group_calls[index]++; g_device_group_experts[index]+=(uint64_t)count;
      g_device_group_rows[index]+=(uint64_t)total; }
    return 1;
}
catch (sycl::exception const &exc) { return sycl_caught(exc, "coli_cuda_expert_group_issue"); }

extern "C" const float *coli_cuda_expert_group_take(int device) try {
    DeviceContext *ctx=find_ctx(device);
    if(!ctx||!ctx->group_pending) return nullptr;
    ctx->group_pending=0;
    if(!select_ctx(ctx)) return nullptr;
    if (!cuda_ok(COLI_SYCL_TRY(ctx->stream->wait()),
                 "expert group take")) return nullptr;
    return ctx->host_y;
}
catch (sycl::exception const &exc) { sycl_caught(exc, "coli_cuda_expert_group_take"); return NULL; }

/* The absorb kernels decode `w` through weight_at + absorb_scale, which know
 * per-row scales, fmt=4 group scales, and fmt=8 per-128x128-block scales.
 * Refuse anything else (fmt=5/6/7) rather than mis-decode it — the caller
 * keeps its CPU attention path. (`proj` tensors are exempt: they run through
 * quant_matmul, which dispatches every format it uploads.) fmt=8 support
 * funnels through this one predicate for all the absorb host wrappers below,
 * so none of them needed a separate change.
 *
 * The admissible set is weight_at's own, taken from the shared predicate rather
 * than restated as an inequality: this gate and weight_at's device-side
 * backstop must not be able to drift apart, and the old `fmt <= 4` also
 * admitted NEGATIVE fmt values, which weight_at would then have fallen through
 * on. A fmt=8 tensor implies a live e4m3 LUT (upload refuses it otherwise --
 * see the predicate's caveat note in backend_cuda.h), so no extra gate is
 * needed here. */
static int absorb_fmt_ok(const ColiCudaTensor *w){
    return w && coli_cuda_weight_at_supported(w->fmt);
}

extern "C" int coli_cuda_attention_absorb(ColiCudaTensor *w, float *ctx,
                                          const float *q, const float *latent,
                                          const float *rope, int H, int Q,
                                          int R, int V, int K, int T,
                                          float scale) try {
    if (fault_injected()) return 0;
    if(!absorb_fmt_ok(w)||!ctx||!q||!latent||!rope||H<1||Q<1||R<1||V<1||K<1||K>512||T<1||T>4096||
       w->I!=K||w->O!=H*(Q+V))return 0;
    DeviceContext *dc=find_ctx(w->device);if(!select_ctx(dc))return 0;
    size_t qb=(size_t)H*(Q+R)*sizeof(float),lb=(size_t)T*K*sizeof(float);
    size_t rb=(size_t)T*R*sizeof(float),cb=(size_t)H*V*sizeof(float);
    if(!reserve(&dc->aq,&dc->aq_cap,qb)||!reserve(&dc->al,&dc->al_cap,lb)||
       !reserve(&dc->ar,&dc->ar_cap,rb)||!reserve(&dc->ac,&dc->ac_cap,cb))return 0;
    if (!cuda_ok(COLI_SYCL_TRY(dc->stream->memcpy(dc->aq, q, qb)),
                 "attention q upload") ||
        !cuda_ok(COLI_SYCL_TRY(dc->stream->memcpy(dc->al, latent, lb)),
                 "attention latent upload") ||
        !cuda_ok(COLI_SYCL_TRY(dc->stream->memcpy(dc->ar, rope, rb)),
                 "attention rope upload")) return 0;
    size_t shared = (size_t)(2 * K + T) * sizeof(float);
    {

        dc->stream->submit([&](sycl::handler &cgh) {
            const float *c_e4m3_ptr_ct1 = cur_ctx()->e4m3;

            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range<1>(shared), cgh);

            auto dc_ac_ct0 = dc->ac;
            const float *dc_aq_ct1 = dc->aq;
            const float *dc_al_ct2 = dc->al;
            const float *dc_ar_ct3 = dc->ar;
            const void *w_weights_ct4 = w->weights;
            const float *w_scales_ct5 = w->scales;
            auto w_fmt_ct6 = w->fmt;
            auto w_gs_ct14 = w->gs;
            auto w_ng_ct15 = w->ng;

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, H) *
                                      sycl::range<3>(1, 1, 256),
                                  sycl::range<3>(1, 1, 256)),
                [=](sycl::nd_item<3> item_ct1) {
                    attention_absorb_kernel(
                        dc_ac_ct0, dc_aq_ct1, dc_al_ct2, dc_ar_ct3,
                        w_weights_ct4, w_scales_ct5, w_fmt_ct6, H, Q, R, V, K,
                        T, scale, w_gs_ct14, w_ng_ct15,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get(),
                        c_e4m3_ptr_ct1);
                });
        });
    }
    if (!cuda_ok(0, "attention absorb launch") ||
        !cuda_ok(COLI_SYCL_TRY(dc->stream->memcpy(ctx, dc->ac, cb)),
                 "attention context download") ||
        !cuda_ok(COLI_SYCL_TRY(dc->stream->wait()),
                 "attention synchronize")) return 0;
    return 1;
}
catch (sycl::exception const &exc) { return sycl_caught(exc, "coli_cuda_attention_absorb"); }

static int attention_absorb_batch_run(ColiCudaTensor *w, ColiCudaTensor *proj,
                                      float *out, const float *q,
                                      const float *latent, const float *rope,
                                      int S, int H, int Q, int R, int V, int K,
                                      int T, float scale) try {
    if(!absorb_fmt_ok(w)||!out||!q||!latent||!rope||S<1||H<1||Q<1||R<1||V<1||K<1||K>512||
       T<S||T>8192||w->I!=K||w->O!=H*(Q+V))return 0;
    if(proj&&(proj->device!=w->device||proj->I!=H*V))return 0;
    DeviceContext *dc=find_ctx(w->device);if(!select_ctx(dc))return 0;
    size_t qb=(size_t)S*H*(Q+R)*sizeof(float),lb=(size_t)T*K*sizeof(float);
    size_t rb=(size_t)T*R*sizeof(float),cb=(size_t)S*H*V*sizeof(float);
    if(!reserve(&dc->aq,&dc->aq_cap,qb)||!reserve(&dc->al,&dc->al_cap,lb)||
       !reserve(&dc->ar,&dc->ar_cap,rb)||!reserve(&dc->ac,&dc->ac_cap,cb))return 0;
    if (!cuda_ok(COLI_SYCL_TRY(dc->stream->memcpy(dc->aq, q, qb)),
                 "attention batch q upload") ||
        !cuda_ok(COLI_SYCL_TRY(dc->stream->memcpy(dc->al, latent, lb)),
                 "attention batch latent upload") ||
        !cuda_ok(COLI_SYCL_TRY(dc->stream->memcpy(dc->ar, rope, rb)),
                 "attention batch rope upload")) return 0;
    size_t shared = (size_t)(2 * K + T + 256) * sizeof(float);
    {

        dc->stream->submit([&](sycl::handler &cgh) {
            const float *c_e4m3_ptr_ct1 = cur_ctx()->e4m3;

            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range<1>(shared), cgh);

            auto dc_ac_ct0 = dc->ac;
            const float *dc_aq_ct1 = dc->aq;
            const float *dc_al_ct2 = dc->al;
            const float *dc_ar_ct3 = dc->ar;
            const void *w_weights_ct4 = w->weights;
            const float *w_scales_ct5 = w->scales;
            auto w_fmt_ct6 = w->fmt;
            auto w_gs_ct15 = w->gs;
            auto w_ng_ct16 = w->ng;

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, S, H) *
                                      sycl::range<3>(1, 1, 256),
                                  sycl::range<3>(1, 1, 256)),
                [=](sycl::nd_item<3> item_ct1) {
                    attention_absorb_batch_kernel(
                        dc_ac_ct0, dc_aq_ct1, dc_al_ct2, dc_ar_ct3,
                        w_weights_ct4, w_scales_ct5, w_fmt_ct6, S, H, Q, R, V,
                        K, T, scale, w_gs_ct15, w_ng_ct16,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get(),
                        c_e4m3_ptr_ct1);
                });
        });
    }
    if (!cuda_ok(0, "attention batch launch")) return 0;
    const float *src=dc->ac;size_t ob=cb;
    if(proj){
        ob=(size_t)S*proj->O*sizeof(float);if(!reserve(&dc->y,&dc->y_cap,ob))return 0;
        {

            dc->stream->submit([&](sycl::handler &cgh) {
                const float *c_e4m3_ptr_ct1 = cur_ctx()->e4m3;

                const uint8_t *c_e8_grid_acc_ct1 = cur_ctx()->e8_grid;

                auto dc_y_ct0 = dc->y;
                const float *dc_ac_ct1 = dc->ac;
                const void *proj_weights_ct2 = proj->weights;
                const float *proj_scales_ct3 = proj->scales;
                auto proj_fmt_ct4 = proj->fmt;
                auto proj_I_ct6 = proj->I;
                auto proj_O_ct7 = proj->O;
                auto row_bytes_proj_fmt_proj_I_ct8 =
                    row_bytes(proj->fmt, proj->I);
                auto proj_gs_ct9 = proj->gs;
                auto proj_ng_ct10 = proj->ng;

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, S, proj->O) *
                                          sycl::range<3>(1, 1, 256),
                                      sycl::range<3>(1, 1, 256)),
                    [=](sycl::nd_item<3> item_ct1) {
                        quant_matmul(dc_y_ct0, dc_ac_ct1, proj_weights_ct2,
                                     proj_scales_ct3, proj_fmt_ct4, S,
                                     proj_I_ct6, proj_O_ct7,
                                     row_bytes_proj_fmt_proj_I_ct8, proj_gs_ct9,
                                     proj_ng_ct10, c_e8_grid_acc_ct1,
                                     c_e4m3_ptr_ct1);
                    });
            });
        }
        if (!cuda_ok(0, "attention o_proj launch")) return 0; src = dc->y;
    }
    if (!cuda_ok(COLI_SYCL_TRY(dc->stream->memcpy(out, src, ob)),
                 proj ? "attention projected output download"
                      : "attention batch context download") ||
        !cuda_ok(COLI_SYCL_TRY(dc->stream->wait()),
                 "attention batch synchronize")) return 0;
    return 1;
}
catch (sycl::exception const &exc) { return sycl_caught(exc, "attention_absorb_batch_run"); }

extern "C" int coli_cuda_attention_absorb_batch(ColiCudaTensor *w,float *ctx,const float *q,
        const float *latent,const float *rope,int S,int H,int Q,int R,int V,int K,int T,
        float scale){
    if (fault_injected()) return 0;
    return attention_absorb_batch_run(w,nullptr,ctx,q,latent,rope,S,H,Q,R,V,K,T,scale);
}

extern "C" int coli_cuda_attention_project_batch(ColiCudaTensor *w,ColiCudaTensor *proj,
        float *out,const float *q,const float *latent,const float *rope,int S,int H,int Q,
        int R,int V,int K,int T,float scale){
    if (fault_injected()) return 0;
    return attention_absorb_batch_run(w,proj,out,q,latent,rope,S,H,Q,R,V,K,T,scale);
}

extern "C" int coli_cuda_attention_project_ragged(
    ColiCudaTensor *w, ColiCudaTensor *proj, float *out, const float *q,
    const void *const *keys, const float *const *latent,
    const float *const *rope, const int *lengths, int S, int H, int Q, int R,
    int V, int K, int T, float scale) try {
    if(!absorb_fmt_ok(w)||!proj||!out||!q||!keys||!latent||!rope||!lengths||S<1||S>512||T<1||T>8192||
       H<1||Q<1||R<1||V<1||K<1||K>512||w->I!=K||w->O!=H*(Q+V)||
       proj->device!=w->device||proj->I!=H*V)return 0;
    DeviceContext *dc=find_ctx(w->device);
    if(!select_ctx(dc))return 0;
    int *old=(int*)std::malloc((size_t)S*sizeof(*old));
    int *add=(int*)std::malloc((size_t)S*sizeof(*add));
    int *off=(int*)std::malloc((size_t)S*sizeof(*off));int packed_n=0;
    if(!old||!add||!off){std::free(old);std::free(add);std::free(off);return 0;}
    int page_stride=0;
    for(int s=0;s<S;s++){
        if(!keys[s]||lengths[s]<1||lengths[s]>T){std::free(old);std::free(add);std::free(off);return 0;}
        RaggedKVEntry *e=nullptr;
        for(int i=0;i<w->ragged_count;i++)if(w->ragged[i].key==keys[s]){e=&w->ragged[i];break;}
        if(!e){
            if(w->ragged_count>=512){std::free(old);std::free(add);std::free(off);return 0;}
            e=&w->ragged[w->ragged_count++];std::memset(e,0,sizeof(*e));e->key=keys[s];
        }
        if(e->K!=K||e->R!=R||e->host_l!=latent[s]||e->host_r!=rope[s]||lengths[s]<e->length){
            ragged_kv_clear(e);
            e->K=K;e->R=R;e->host_l=latent[s];e->host_r=rope[s];
        }
        int need=(lengths[s]+COLI_KV_PAGE_TOKENS-1)/COLI_KV_PAGE_TOKENS;
        if(need>e->page_count){
            float **nl=(float**)std::calloc((size_t)need,sizeof(*nl));
            float **nr=(float**)std::calloc((size_t)need,sizeof(*nr));
            if(!nl||!nr){std::free(nl);std::free(nr);std::free(old);std::free(add);std::free(off);return 0;}
            for(int i=0;i<e->page_count;i++){nl[i]=e->latent_pages[i];nr[i]=e->rope_pages[i];}
            int made=e->page_count;
            for(;made<need;made++){
                if (!cuda_ok(
                        COLI_SYCL_ALLOC(nl[made], sycl::malloc_device<float>(
                                             (size_t)COLI_KV_PAGE_TOKENS * K,
                                             dq()), (size_t)COLI_KV_PAGE_TOKENS * K),
                        "ragged KV latent page") ||
                    !cuda_ok(
                        COLI_SYCL_ALLOC(nr[made], sycl::malloc_device<float>(
                                             (size_t)COLI_KV_PAGE_TOKENS * R,
                                             dq()), (size_t)COLI_KV_PAGE_TOKENS * R),
                        "ragged KV rope page")) break;
            }
            if(made<need){
                if (nl[made])
                    sycl::free(nl[made], dq());
                if (nr[made])
                    sycl::free(nr[made], dq());
                for (int i = e->page_count; i < made; i++) {
                    sycl::free(nl[i], dq());
                    sycl::free(nr[i], dq());
                }
                std::free(nl);std::free(nr);std::free(old);std::free(add);std::free(off);return 0;
            }
            std::free(e->latent_pages);std::free(e->rope_pages);
            e->latent_pages=nl;e->rope_pages=nr;e->page_count=need;
        }
        if(e->page_count>page_stride)page_stride=e->page_count;
        old[s]=e->length;add[s]=lengths[s]-e->length;
        off[s]=packed_n;packed_n+=add[s]*(K+R);
    }
    size_t table_n=(size_t)S*page_stride;
    float **dl=(float**)std::calloc(table_n,sizeof(*dl));
    float **dr=(float**)std::calloc(table_n,sizeof(*dr));
    if(!dl||!dr){std::free(dl);std::free(dr);std::free(old);std::free(add);std::free(off);return 0;}
    for(int s=0;s<S;s++)for(int i=0;i<w->ragged_count;i++)if(w->ragged[i].key==keys[s]){
        for(int p=0;p<w->ragged[i].page_count;p++){
            dl[(size_t)s*page_stride+p]=w->ragged[i].latent_pages[p];
            dr[(size_t)s*page_stride+p]=w->ragged[i].rope_pages[p];
        }
        break;
    }
    size_t qb=(size_t)S*H*(Q+R)*sizeof(float);
    size_t cb=(size_t)S*H*V*sizeof(float),ob=(size_t)S*proj->O*sizeof(float);
    size_t pb=(size_t)packed_n*sizeof(float);
    size_t desc=2*table_n*sizeof(float*)+(size_t)S*4*sizeof(int);
    int ok=reserve(&dc->aq,&dc->aq_cap,qb)&&reserve(&dc->ac,&dc->ac_cap,cb)&&
           reserve(&dc->y,&dc->y_cap,ob)&&reserve_bytes(&dc->group_desc,&dc->group_desc_cap,desc)&&
           (!pb||(reserve(&dc->al,&dc->al_cap,pb)&&reserve_pinned(&dc->host_kv,&dc->host_kv_cap,pb)));
    char *db=(char*)dc->group_desc;float **ddl=(float**)db,**ddr=ddl+table_n;
    int *dn=(int*)(ddr+table_n),*dold=dn+S,*dadd=dold+S,*doff=dadd+S;
    if(ok&&pb){
        for(int s=0;s<S;s++)if(add[s]){
            float *p=dc->host_kv+off[s];
            std::memcpy(p,latent[s]+(size_t)old[s]*K,(size_t)add[s]*K*sizeof(float));
            std::memcpy(p+(size_t)add[s]*K,rope[s]+(size_t)old[s]*R,(size_t)add[s]*R*sizeof(float));
        }
        ok = cuda_ok(
            COLI_SYCL_TRY(dc->stream->memcpy(dc->al, dc->host_kv, pb)),
            "ragged KV append upload");
    }
    if (ok) ok =
        cuda_ok(COLI_SYCL_TRY(dc->stream->memcpy(dc->aq, q, qb)),
                "ragged q upload") &&
        cuda_ok(COLI_SYCL_TRY(
                    dc->stream->memcpy(ddl, dl, table_n * sizeof(float *)).wait()),
                "ragged latent page table") &&
        cuda_ok(COLI_SYCL_TRY(
                    dc->stream->memcpy(ddr, dr, table_n * sizeof(float *)).wait()),
                "ragged rope page table") &&
        cuda_ok(COLI_SYCL_TRY(
                    dc->stream->memcpy(dn, lengths, (size_t)S * sizeof(int)).wait()),
                "ragged lengths upload") &&
        cuda_ok(COLI_SYCL_TRY(
                    dc->stream->memcpy(dold, old, (size_t)S * sizeof(int)).wait()),
                "ragged old lengths") &&
        cuda_ok(COLI_SYCL_TRY(
                    dc->stream->memcpy(dadd, add, (size_t)S * sizeof(int)).wait()),
                "ragged append lengths") &&
        cuda_ok(COLI_SYCL_TRY(
                    dc->stream->memcpy(doff, off, (size_t)S * sizeof(int)).wait()),
                "ragged append offsets");
        if (ok && pb)dc->stream->submit([&](sycl::handler &cgh) {
                const float *dc_al_ct2 = dc->al;

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, 1, S) *
                                          sycl::range<3>(1, 1, 256),
                                      sycl::range<3>(1, 1, 256)),
                    [=](sycl::nd_item<3> item_ct1) {
                        ragged_kv_append(ddl, ddr, dc_al_ct2, dold, dadd, doff,
                                         K, R, page_stride);
                    });
            });
    if(ok)for(int s=0;s<S;s++){
        for(int i=0;i<w->ragged_count;i++)if(w->ragged[i].key==keys[s]){w->ragged[i].length=lengths[s];break;}
    }
    std::free(dl);std::free(dr);std::free(old);std::free(add);std::free(off);if(!ok)return 0;
    size_t shared = (size_t)(2 * K + T + 256) * sizeof(float);
    {

        dc->stream->submit([&](sycl::handler &cgh) {
            const float *c_e4m3_ptr_ct1 = cur_ctx()->e4m3;

            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range<1>(shared), cgh);

            auto dc_ac_ct0 = dc->ac;
            const float *dc_aq_ct1 = dc->aq;
            const void *w_weights_ct5 = w->weights;
            const float *w_scales_ct6 = w->scales;
            auto w_fmt_ct7 = w->fmt;
            auto w_gs_ct17 = w->gs;
            auto w_ng_ct18 = w->ng;

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, S, H) *
                                      sycl::range<3>(1, 1, 256),
                                  sycl::range<3>(1, 1, 256)),
                [=](sycl::nd_item<3> item_ct1) {
                    attention_absorb_ragged_kernel(
                        dc_ac_ct0, dc_aq_ct1, ddl, ddr, dn, w_weights_ct5,
                        w_scales_ct6, w_fmt_ct7, S, H, Q, R, V, K, T,
                        page_stride, scale, w_gs_ct17, w_ng_ct18,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get(),
                        c_e4m3_ptr_ct1);
                });
        });
    }
    {

        dc->stream->submit([&](sycl::handler &cgh) {
            const float *c_e4m3_ptr_ct1 = cur_ctx()->e4m3;

            const uint8_t *c_e8_grid_acc_ct1 = cur_ctx()->e8_grid;

            auto dc_y_ct0 = dc->y;
            const float *dc_ac_ct1 = dc->ac;
            const void *proj_weights_ct2 = proj->weights;
            const float *proj_scales_ct3 = proj->scales;
            auto proj_fmt_ct4 = proj->fmt;
            auto proj_I_ct6 = proj->I;
            auto proj_O_ct7 = proj->O;
            auto row_bytes_proj_fmt_proj_I_ct8 = row_bytes(proj->fmt, proj->I);
            auto proj_gs_ct9 = proj->gs;
            auto proj_ng_ct10 = proj->ng;

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, S, proj->O) *
                                      sycl::range<3>(1, 1, 256),
                                  sycl::range<3>(1, 1, 256)),
                [=](sycl::nd_item<3> item_ct1) {
                    quant_matmul(dc_y_ct0, dc_ac_ct1, proj_weights_ct2,
                                 proj_scales_ct3, proj_fmt_ct4, S, proj_I_ct6,
                                 proj_O_ct7, row_bytes_proj_fmt_proj_I_ct8,
                                 proj_gs_ct9, proj_ng_ct10, c_e8_grid_acc_ct1,
                                 c_e4m3_ptr_ct1);
                });
        });
    }
    return cuda_ok(0, "ragged attention launch") &&
           cuda_ok(COLI_SYCL_TRY(dc->stream->memcpy(out, dc->y, ob)),
                   "ragged output download") &&
           cuda_ok(COLI_SYCL_TRY(dc->stream->wait()),
                   "ragged attention synchronize");
}
catch (sycl::exception const &exc) { return sycl_caught(exc, "coli_cuda_attention_project_ragged"); }

extern "C" void coli_cuda_tensor_free(ColiCudaTensor *tensor) {
    /* sycl::free releases at once where cudaFree synchronised the device first: a free under a
     * kernel still reading the tensor is kept off by the engine's issue_open / chain_hold
     * ordering (qwen36_tier.c: a victim's free never meets an in-flight group or forward), which
     * this relies on where the CUDA line did not have to */
    if (!tensor) return;
    DeviceContext *ctx = find_ctx(tensor->device);
    if (ctx) select_ctx(ctx);
    if (tensor->tracked && ctx) {
        /* Charge and release the same format-specific scale storage. */
        size_t storage_bytes =
            tensor->weight_bytes;
        size_t bytes = storage_bytes +
            tensor_scale_bytes(tensor);
        if (ctx->tensor_count) ctx->tensor_count--;
        if (ctx->tensor_bytes >= bytes) ctx->tensor_bytes -= bytes;
    }
    if (tensor->weights && tensor->weights_owned)
        sycl::free(tensor->weights, dq());
    if (tensor->scales && !tensor->scales_borrowed)
        sycl::free(tensor->scales, dq());
    for(int i=0;i<tensor->ragged_count;i++)ragged_kv_clear(&tensor->ragged[i]);
    std::free(tensor);
}

extern "C" size_t coli_cuda_tensor_bytes(const ColiCudaTensor *tensor) {
    if (!tensor) return 0;
    /* Logical size uses the same scale layout as upload and free. */
    size_t storage_bytes =
        tensor->weight_bytes;
    return storage_bytes +
        tensor_scale_bytes(tensor);
}

/* What a cudaMalloc of `bytes` actually takes off the card.
 *
 * coli_cuda_tensor_bytes() above is the LOGICAL size and must stay that way -
 * it mirrors upload and free so the three cannot drift. This is a different
 * question: the allocator rounds a request up, and nothing was charging the
 * difference to the expert budget.
 *
 * Measured on an RTX 3090 (sm_86, CUDA 13.1):
 *
 *     request              actual      overhead
 *     0.75 MiB          1.00 MiB        +0.25     <- int4-g64 scale array
 *     1.00 MiB          1.00 MiB        +0.00
 *     1.00 MiB + 1 B    2.00 MiB        +1.00
 *     1.50 MiB          2.00 MiB        +0.50
 *     2.00 MiB          2.00 MiB        +0.00
 *     2.00 MiB + 1 B    4.00 MiB        +2.00
 *     6.00 MiB          6.00 MiB        +0.00     <- int4-g64 weight tensor
 *
 * A GLM-5.2 int4-g64 expert is three 6 MiB weight arrays, which are already on
 * a boundary and cost nothing extra, and three 0.75 MiB scale arrays, each
 * padded to 1 MiB. 3 x 0.25 = 0.75 MiB per expert unaccounted - which is the
 * 0.741 MiB/expert (sigma 0.019) measured independently on H100 and H200 in
 * #687, from the other direction.
 *
 * PROBED, not modelled. The table above is not a rule this hardcodes: the
 * rounding is a driver and architecture property, and a formula fitted to one
 * card would be silently wrong on the next. Probed once per distinct size
 * and cached, so the cost does not scale with the tier.
 *
 * cudaMemGetInfo is the obvious alternative and it does not survive contact
 * with the numbers: it is O(live allocations), measured 0.52 us empty and
 * 68.2 us with 12,000 live, so charging it per expert would be quadratic
 * across a tier that places thousands.
 *
 * Returns `bytes` unchanged if the probe cannot run. Under-reporting is the
 * pre-existing behaviour and degrades to exactly what this replaced; refusing
 * to size at all would be worse.
 */
#define COLI_CUDA_FOOTPRINT_CACHE 16
static struct { size_t req, real; } g_fp_cache[COLI_CUDA_FOOTPRINT_CACHE];
static int g_fp_cache_n = 0;

/* The probe MUST allocate several buffers and divide, not one and measure it.
 *
 * A single cudaMalloc reserves a whole 2 MiB VMM page, so one allocation of
 * anything smaller reads as a flat 2 MiB and the answer is the page size
 * rather than the per-allocation cost. Later allocations suballocate from
 * pages already reserved, so the real amortised figure only appears once
 * enough of them are live to fill a page. Measured both ways on sm_86, same
 * 786,432-byte request:
 *
 *     1 allocation    2.00 MiB   <- the page, not the allocation
 *     64 allocations  1.00 MiB   <- the truth, and what the tier will pay
 *
 * The single-shot version of this function shipped in an earlier draft and its
 * own test caught it: it over-reported every expert by 3x and would have
 * shrunk the tier far more than the bug it fixes.
 */
#define COLI_CUDA_FOOTPRINT_PROBE_BYTES (24u << 20)   /* transient probe budget */
#define COLI_CUDA_FOOTPRINT_PROBE_MAX   64

extern "C" size_t coli_cuda_alloc_footprint(size_t bytes) try {
    if (!bytes) return 0;
    for (int i = 0; i < g_fp_cache_n; i++)
        if (g_fp_cache[i].req == bytes) return g_fp_cache[i].real;

    /* Enough allocations to cross a page boundary, capped so the probe never
     * takes a meaningful bite out of a card that is about to be filled. */
    int want = (int)(COLI_CUDA_FOOTPRINT_PROBE_BYTES / bytes);
    if (want < 8) want = 8;
    if (want > COLI_CUDA_FOOTPRINT_PROBE_MAX) want = COLI_CUDA_FOOTPRINT_PROBE_MAX;

    size_t free_before = 0, free_after = 0, total = 0, real = bytes;
    void *p[COLI_CUDA_FOOTPRINT_PROBE_MAX];
    int made = 0;
    if (g_nctx > 0 && dev_mem_info(cur_device(), &free_before, &total)) {
        for (int i = 0; i < want; i++) {
            if (COLI_SYCL_ALLOC(p[i], (void *)sycl::malloc_device(
                                     bytes, dq()), bytes) !=
                0) break;
            made++;
        }
        if (made > 0 && dev_mem_info(cur_device(), &free_after, &total) &&
            free_before > free_after) {
            size_t avg = (free_before - free_after) / (size_t)made;
            /* Only ever round UP. A concurrent free elsewhere on the device
             * can make the delta read small, and charging less than the
             * logical size is the failure this exists to fix. */
            if (avg > real) real = avg;
        }
        for (int i = 0; i < made; i++)
            sycl::free(p[i], dq());
    }
    /* Cache even a failed probe: `real` is then the logical size, which is the
     * pre-existing behaviour, and retrying per expert on a card that just
     * refused 8 allocations would be the worst possible time to try again. */
    if (g_fp_cache_n < COLI_CUDA_FOOTPRINT_CACHE)
        g_fp_cache[g_fp_cache_n++] = { bytes, real };
    return real;
}
catch (sycl::exception const &exc) { sycl_caught(exc, "coli_cuda_alloc_footprint"); return bytes; }

/* Same split as coli_cuda_tensor_bytes, but per ALLOCATION rather than summed
 * first: the weights and the scales are two separate cudaMallocs and each is
 * rounded on its own. Summing then rounding once would miss the scale array's
 * padding entirely, which is the whole term. */
extern "C" size_t coli_cuda_tensor_vram(const ColiCudaTensor *tensor) {
    if (!tensor) return 0;
    size_t storage_bytes =
        tensor->weight_bytes;
    size_t total = coli_cuda_alloc_footprint(storage_bytes);
    if (tensor->fmt && tensor->fmt != 6)
        total += coli_cuda_alloc_footprint(tensor_scale_bytes(tensor));
    return total;
}

extern "C" int coli_cuda_tensor_device(const ColiCudaTensor *tensor) {
    return tensor ? tensor->device : -1;
}
extern "C" int coli_cuda_tensor_shape(const ColiCudaTensor *t, int *fmt, int *I, int *O) {
    if (!t) return 0;
    if (fmt) *fmt = t->fmt;
    if (I) *I = t->I;
    if (O) *O = t->O;
    return 1;
}
/* The qwen36 tier's host pool pinned its RAM cache for the device to read over the bus
 * (cudaHostRegister). SYCL registers no existing host memory (USM host memory is the runtime's
 * own allocation), so the pool is off: 0, and the tier keeps its copies. */
extern "C" int coli_cuda_host_register(void *p, size_t bytes) {
    static int said;
    if (!p || !bytes) return 0;
    if (!said) { said = 1; std::fprintf(stderr, "[SYCL] host pool: cudaHostRegister has no SYCL form (USM host memory is the runtime's): the tier's host pool is off\n"); }
    return 0;
}
extern "C" void coli_cuda_host_unregister(void *p) { (void)p; }
/* whether the chain may upload planar int4-g64 experts (expert_ffn.h's blocks), which only the
 * matrix engines' grouped GEMM reads (coli_sycl_xmx_moe): where the profile takes the f16 shape
 * for the gate / up widths and the down's */
extern "C" int coli_cuda_planar_rows(int device, int D, int I) {
    const coli_sycl_profile &p = coli_sycl_profile_for(device);
    return D > 0 && I > 0 && D % XMX_BK == 0 && I % XMX_BK == 0 && xmx_engines(p) &&
           coli_sycl_xmx_gemm_ok(p, 4, 64, D, I) && coli_sycl_xmx_gemm_ok(p, 4, 64, I, D);
}
/* A hyper-connection read's up projection and its mix in one pass was hcm_mix_k's (f16 mma.sync
 * over a GGUF's blocks): not here. -1, not taken: the chain's matmul and its CC_EW_HC_MIX do it. */
extern "C" int coli_cuda_tensor_gemm_hcmix_async(ColiCudaTensor *t, float *mixed, const float *x, const float *hn, int S, int H, int C,
                                                 float fc, void *stream) {
    (void)t; (void)mixed; (void)x; (void)hn; (void)S; (void)H; (void)C; (void)fc; (void)stream;
    return -1;
}

/* A decode GEMV for int8 rows (fmt 1, one activation row): a warp per output row,
 * each lane 16 bytes of the row per step (I a multiple of 16), the activation in
 * shared memory, a shuffle reduction, the row's scale. quant_matmul above spends a
 * block of 256 threads on one row and reaches a quarter of the bus; this reads the
 * trunk at the bus's pace, which is what the dense chain's decode token is made of
 * (about 1.9 GB of rows on Qwen3.6-35B). The same sums as quant_matmul up to the
 * order of the float additions. */
/* output rows a block of 256 threads: a sub-group ("warp") each, so 256 / SG (ctx->wpb on the host);
 * CUDA's GEMV_I8_ROWS 8 at warp 32 */
#define GEMV_I8_MAX_S 8  /* activation rows it takes: a verify's rows get a decode step's bits, as
                            the engines ask (q36_spec_step) */
#define GEMV_I8_NR 4     /* a verify's rows a block takes, every weight load shared among them */
/* NR 1: a decode step's row, a block a row as ever. NR GEMV_I8_NR: nr rows of x a block (from
 * the host: what 48 KiB of shared memory holds, up to NR), every 16-byte load of the weight
 * row shared among them; each row's sum is the NR 1 kernel's operations in their order, so
 * its bits are a decode step's (tests/test_gemv_rows_cuda.cu). Before, a verify's rows were
 * a block each and read every weight once a row. */
/* x in shared memory as float4s with one of padding after every eight (gx4): a lane's 16
 * values are four float4s and the next lane's start 72 bytes on, where 16 floats a lane apart
 * put the lanes of a load on the same banks. The values and the order of the sums are those
 * of plain floats: the layout moves no bit. A row past what 48 KiB holds padded keeps them
 * unpadded (PAD 0). */
static inline int gx4(int k) { return k + (k >> 3); }
/* XG reads one activation row directly from global memory (L1), with the same lane order.
 * NR>1 keeps staging: a verify and a decode must differ only in where x is read. */
template <int NR, int PAD, int XG, int SG>
static void gemv_i8_rows(float *__restrict__ y, const float *__restrict__ x,
                         const int8_t *__restrict__ w,
                         const float *__restrict__ sc, int I, int O, int S,
                         int nr, uint8_t *dpct_local) {
    static_assert(!XG || NR == 1, "global x is a one-row variant");
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto xs4 = (sycl::float4 *)dpct_local;
    int s0 = NR == 1 ? (int)item_ct1.get_group(1)
                     : (int)item_ct1.get_group(1) * nr,
        nrow = NR == 1 ? 1 : sycl::min(nr, S - s0);
    int I4 = I / 4, XR = PAD ? gx4(I4) + 1 : I4;
    x += (size_t)s0 * I; y += (size_t)s0 * O;
    const sycl::float4 *x4 = reinterpret_cast<const sycl::float4 *>(x);
    if constexpr (!XG) {
        for (int k = item_ct1.get_local_id(2); k < nrow * I4; k += 256) {
            int r = NR == 1 ? 0 : k / I4, kk = NR == 1 ? k : k - r * I4;
            xs4[r * XR + (PAD ? gx4(kk) : kk)] = x4[(size_t)r * I4 + kk];
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);
    }
    int lane = item_ct1.get_local_id(2) & (SG - 1),
        warp = item_ct1.get_local_id(2) / SG;
    int o = item_ct1.get_group(2) * (256 / SG) + warp;
    if (o >= O) return;
    const int8_t *row = w + (size_t)o * I;
    float acc[NR];
    #pragma unroll
    for (int r = 0; r < NR; r++) acc[r] = 0.f;
    for (int i0 = lane * 16; i0 < I; i0 += SG * 16) {
        sycl::int4 v = *reinterpret_cast<const sycl::int4 *>(row + i0);
        const int8_t *b = reinterpret_cast<const int8_t *>(&v);
        float wf[16];
        #pragma unroll
        for (int k = 0; k < 16; k++) wf[k] = (float)b[k];
        int k4 = i0 / 4;
        #pragma unroll
        for (int r = 0; r < NR; r++) if (r < nrow) {
            sycl::float4 q[4];
#pragma unroll
            for (int j = 0; j < 4; j++) {
                if constexpr (XG) q[j] = x4[k4 + j];
                else q[j] = xs4[r * XR + (PAD ? gx4(k4 + j) : k4 + j)];
            }
            const float *xv = reinterpret_cast<const float *>(q);
            #pragma unroll
            for (int k = 0; k < 16; k++) acc[r] += xv[k] * wf[k];
        }
    }
    #pragma unroll
    for (int r = 0; r < NR; r++) if (r < nrow) {
        float a = acc[r];
        for (int off = SG / 2; off > 0; off >>= 1) a += shfl_down(a, off);
        if (lane == 0) y[(size_t)r * O + o] = a * sc[o];
    }
}
/* gemv_i8_rows' arithmetic, four outputs a warp: a lane's slice of each row of x (shared memory)
 * loaded once for the four weight rows it meets, where gemv_i8_rows loaded it again for every
 * output (at a verify's four rows the loads of x, not the weights, set the time: a 10240 x 320
 * matrix 15 us, 218 GB/s, on an RTX PRO 4000). Each output's sums in gemv_i8_rows' lanes and
 * order, so its bits. Optionally over a split of the inputs (kz splits of kc, blockIdx.y): the
 * partial sums into part[(z * S + s) * O + o], gemv_i8_sk_sum adding them in order and scaling;
 * kz 1: y scaled here. The split follows the matrix's shape alone (gemv_i8_sk_splits), so a
 * verify's rows keep a decode step's bits. */
#define GEMV_I8_OPW 4
/* PAD 0 (COLI_CUDA_GEMV_XPAD=0, rows of several activation rows only): x in shared memory without the
 * padding float4s, as gemv_i8_rows' PAD 0. The padding spreads a load's lanes over CUDA's banks; on an
 * Intel device it only makes the block's local memory larger, and four rows of 2048 inputs then ask for
 * 36.1 KiB where 32 KiB would hold them (measured on a Max 1100: four rows a block take twice the time
 * of three; g6-evidence/m4-kv-slots). The values and the order of every sum are the padded kernel's. */
template <int NR, int XG, int SG, int PAD = 1>
static void gemv_i8_w4(float *__restrict__ y, float *__restrict__ part,
                       const float *__restrict__ x,
                       const int8_t *__restrict__ w,
                       const float *__restrict__ sc, int I, int O, int S,
                       int nr, int kc, uint8_t *dpct_local) {
    static_assert(!XG || NR == 1, "global x is only a one-row GEMV");
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto xs4 = (sycl::float4 *)dpct_local;
    const int z = (int)item_ct1.get_group(1), k0 = z * kc,
              kn = sycl::min(kc, I - k0), s0 = (int)item_ct1.get_group(0) * nr,
              nrow = sycl::min(nr, S - s0);
    const int K4 = kn / 4, XR = PAD ? gx4(K4) + 1 : K4;
    const auto *x4 = reinterpret_cast<const sycl::float4 *>(x + (size_t)s0 * I + k0);
    if constexpr (!XG) {
        for (int k = item_ct1.get_local_id(2); k < nrow * K4; k += 256) {
            int r = k / K4, kk = k - r * K4;
            xs4[r * XR + (PAD ? gx4(kk) : kk)] = reinterpret_cast<const sycl::float4 *>(
                x + (size_t)(s0 + r) * I + k0)[kk];
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);
    }
    int lane = item_ct1.get_local_id(2) & (SG - 1),
        warp = item_ct1.get_local_id(2) / SG,
        o0 = ((int)item_ct1.get_group(2) * (256 / SG) + warp) * GEMV_I8_OPW;
    if (o0 >= O) return;
    const int8_t *row[GEMV_I8_OPW];
    #pragma unroll
    for (int q = 0; q < GEMV_I8_OPW; q++) row[q] = w + (size_t)(o0 + q < O ? o0 + q : o0) * I + k0;
    float acc[GEMV_I8_OPW][NR];
    #pragma unroll
    for (int q = 0; q < GEMV_I8_OPW; q++)
        #pragma unroll
        for (int r = 0; r < NR; r++) acc[q][r] = 0.f;
    for (int i0 = lane * 16; i0 < kn; i0 += SG * 16) {
        const int k4 = i0 / 4;
        sycl::float4 xq[NR][4];
#pragma unroll
        for (int r = 0; r < NR; r++) if (r < nrow) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                if constexpr (XG) xq[r][j] = x4[k4 + j];
                else xq[r][j] = xs4[r * XR + (PAD ? gx4(k4 + j) : k4 + j)];
            }
        }
        #pragma unroll
        for (int q = 0; q < GEMV_I8_OPW; q++) {
            sycl::int4 v = *reinterpret_cast<const sycl::int4 *>(row[q] + i0);
            const int8_t *b = reinterpret_cast<const int8_t *>(&v);
            float wf[16];
            #pragma unroll
            for (int k = 0; k < 16; k++) wf[k] = (float)b[k];
            #pragma unroll
            for (int r = 0; r < NR; r++) if (r < nrow) {
                const float *xv = reinterpret_cast<const float *>(xq[r]);
                #pragma unroll
                for (int k = 0; k < 16; k++) acc[q][r] += xv[k] * wf[k];
            }
        }
    }
    #pragma unroll
    for (int q = 0; q < GEMV_I8_OPW; q++)
        #pragma unroll
        for (int r = 0; r < NR; r++) if (r < nrow) {
            float a = acc[q][r];
            for (int off = SG / 2; off > 0; off >>= 1) a += shfl_down(a, off);
            const int o = o0 + q;
            if (lane == 0 && o < O) {
                if (part) part[((size_t)z * S + s0 + r) * O + o] = a;
                else y[(size_t)(s0 + r) * O + o] = a * sc[o];
            }
        }
}
static void gemv_i8_sk_sum(float *__restrict__ y, const float *__restrict__ part, const float *__restrict__ sc, int O, int S, int ks) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int i = (int)(item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2));
    if (i >= S * O) return;
    float a = part[i];
    for (int z = 1; z < ks; z++) a += part[(size_t)z * S * O + i];
    y[i] = a * sc[i % O];
}
/* the splits of an int8 matrix's inputs (gemv_i8_w4): until its blocks (of 32 outputs) fill the
 * multiprocessors twice, whole steps of 512, from 2048 inputs (Qwen3.8's hyper-connection read:
 * 10240 inputs, 320 outputs, 10 blocks). COLI_CUDA_GEMV_SK=0: none. */
static int gemv_i8_sk_splits(const DeviceContext *ctx, int I, unsigned bx) {
    static int on = -1;
    if (on < 0) { const char *e = getenv("COLI_CUDA_GEMV_SK"); on = !(e && *e == '0'); }
    if (!on || bx >= 2u * (unsigned)ctx->sms || I < 2048 || I % 512) return 1;
    int ks = (int)((2u * (unsigned)ctx->sms + bx - 1) / bx);
    if (ks > I / 512) ks = I / 512;
    return ks < 2 ? 1 : ks;
}
/* Parsed once per knob; an unsupported width is checked against each device below. */
static int kernel_sg_env(const char *name) {
    const char *e = getenv(name);
    if (!e || !*e) return 0;
    if (!strcmp(e, "16")) return 16;
    if (!strcmp(e, "32")) return 32;
    fprintf(stderr, "[SYCL] %s=%s ignored (requires 16 or 32)\n", name, e);
    return 0;
}
/* Shape only, never S: verify rows must use the same reduction tree as one-row calls. */
static int gemv_i8_sg(const DeviceContext *ctx, int O, int I) {
    (void)I;
    static const int requested = kernel_sg_env("COLI_CUDA_GEMV_SG");
    const auto &pr = coli_sycl_profile_for(ctx->device);
    if (requested) {
        if (coli_sycl_sg_offered(pr, requested)) return requested;
        static bool warned = false;
        if (!warned) { fprintf(stderr, "[SYCL] COLI_CUDA_GEMV_SG=%d ignored (not offered)\n", requested); warned = true; }
    }
    return pr.arch == COLI_SYCL_ARCH_PVC && O >= 1024 && O <= 2048 && coli_sycl_sg_offered(pr, 16) ? 16 : ctx->sg;
}
/* W4 is decided per architecture for every S, including its shape-based split-K. */
static int gemv_i8_w4_on(const DeviceContext *ctx) {
    static const int setting = [] { const char *e = getenv("COLI_CUDA_GEMV_W4"); return e && *e ? *e != '0' : -1; }();
    /* PLAN6: verify rows regressed >1.5x and lookup is used. Keep W4 for every S. */
    return setting >= 0 ? setting : 1;
}
static int gemv_i8_xs_on(void) {
    static const int on = [] { const char *e = getenv("COLI_CUDA_GEMV_XS"); return e && *e == '1'; }();
    return on;
}
/* COLI_CUDA_GEMV_XPAD: how the kernels that take several activation rows a block (gemv_i8_w4, gemv_i8_rows)
 * keep x in local memory. 1 (the code's default): padded, as before; unset on a Data Center GPU Max: 2, the
 * value measured there (coli_cuda_tuned_default). 0: unpadded for every such block. 2: unpadded only
 * for a block of GEMV_I8_NR rows of an unsplit matrix whose padded rows pass 32 KiB and whose unpadded rows do
 * not (four rows of 2048 inputs: 36.1 KiB against 32.0) -- the case measured to cost a block twice its time on
 * a Max 1100, where three-row blocks and the split 2048 x 4096 matrix measured a little slower unpadded. Never
 * where the padded row would not fit the cap: the call then keeps the kernel family it had, as a one-row call
 * of the same matrix does. A one-row call is not touched (it reads x from global memory). */
static int gemv_i8_xpad_mode(void) {
    static const int mode = [] {
        const char *e = getenv("COLI_CUDA_GEMV_XPAD");
        return e && *e ? atoi(e) : coli_cuda_tuned_default("COLI_CUDA_GEMV_XPAD", 1);
    }();
    return mode;
}
#define GEMV_I8_SLOT ((size_t)32768)   /* the local memory a block of four unpadded rows of 2048 inputs takes */
/* COLI_CUDA_GEMV_NR=1: a verify's rows a block each, as before */
static int gemv_i8_nr_on(void) {
    static int on = -1;
    if (on < 0) { const char *e = getenv("COLI_CUDA_GEMV_NR"); on = !(e && *e == '1'); }
    return on;
}

/* The dense chain's matmul (cuda_chain.cu): the resident tensor's kernel on the
 * chain's own stream, device pointers in and out, nothing synchronized here. */
/* A prompt chunk's rows of an f32 matrix (fmt 0: a router, DeltaNet's b and a rows): quant_matmul
 * gives every (output, row) pair a block that reads the whole weight row, which took a GGUF's
 * router (256 x 2048) 6 ms a layer for 2781 rows on an RTX PRO 4000. Here a block computes a
 * BM x 64 tile of y, K in steps of 64 through shared memory, each thread BM/16 x 4 outputs, in
 * f32 fused multiply-adds: the same products as quant_matmul, summed in another order. BM 16
 * where tiles of 64 rows would leave the device's multiprocessors without two blocks each
 * (the router at 489 rows: 32 tiles). */
#define F32G_N 64
#define F32G_K 64
template <int BM>
static void f32_gemm_k(float *__restrict__ y, const float *__restrict__ x,
                       const float *__restrict__ w, int S, int I, int O) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr int RT = BM / 16;   /* a thread's rows */
    auto &xs = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[F32G_K][BM + 4]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &ws = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[F32G_K][F32G_N + 4]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    int tx = item_ct1.get_local_id(2) & 15, ty = item_ct1.get_local_id(2) >> 4,
        r0 = (int)item_ct1.get_group(1) * BM,
        o0 = (int)item_ct1.get_group(2) * F32G_N;
    float acc[RT][4];
    #pragma unroll
    for (int i = 0; i < RT; i++) for (int j = 0; j < 4; j++) acc[i][j] = 0.f;
    for (int k0 = 0; k0 < I; k0 += F32G_K) {
        for (int t = item_ct1.get_local_id(2); t < BM * F32G_K;
             t += 256) { /* a row's 64 consecutive inputs, then the next row */
            int r = t / F32G_K, k = t % F32G_K, gk = k0 + k;
            xs[k][r] = r0 + r < S && gk < I ? x[(size_t)(r0 + r) * I + gk] : 0.f;
        }
        for (int t = item_ct1.get_local_id(2); t < F32G_N * F32G_K; t += 256) {
            int r = t / F32G_K, k = t % F32G_K, gk = k0 + k;
            ws[k][r] = o0 + r < O && gk < I ? w[(size_t)(o0 + r) * I + gk] : 0.f;
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);
#pragma unroll
        for (int k = 0; k < F32G_K; k++) {
            float a[RT], b[4];
            #pragma unroll
            for (int i = 0; i < RT; i++) a[i] = xs[k][ty * RT + i];
            #pragma unroll
            for (int j = 0; j < 4; j++) b[j] = ws[k][tx * 4 + j];
            #pragma unroll
            for (int i = 0; i < RT; i++) for (int j = 0; j < 4; j++)
                acc[i][j] = sycl::fma(a[i], b[j], acc[i][j]);
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);
    }
    #pragma unroll
    for (int i = 0; i < RT; i++) for (int j = 0; j < 4; j++) {
        int r = r0 + ty * RT + i, o = o0 + tx * 4 + j;
        if (r < S && o < O) y[(size_t)r * O + o] = acc[i][j];
    }
}
/* A decode step's rows (up to GEMV_I8_MAX_S) of an f32 matrix of few outputs and long rows
 * (Qwen3.8's hyper-connection injection, 4 x 10240; DeltaNet's b and a rows, 48 x 2560):
 * quant_matmul took the injection 18.8 us on an RTX PRO 4000, four blocks with one load at a
 * time in flight a thread (its loop reads through weight_at's format switch). This is its
 * arithmetic for fmt 0 in its order (each thread's fused multiply-adds at a stride of 256,
 * then the same tree), the bits it gives, with eight loads in flight a thread. */
static void f32_gemv_k(float *__restrict__ y, const float *__restrict__ x,
                                                          const float *__restrict__ w, int I, int O) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto &partial =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[256]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    int o = (int)item_ct1.get_group(2), s = (int)item_ct1.get_group(1),
        i = item_ct1.get_local_id(2);
    const float *wr = w + (size_t)o * I, *xr = x + (size_t)s * I;
    float sum = 0.f;
    for (; i + 7 * 256 < I; i += 8 * 256) {
        float a[8], b[8];
        #pragma unroll
        for (int k = 0; k < 8; k++) { a[k] = xr[i + k * 256]; b[k] = wr[i + k * 256]; }
        #pragma unroll
        for (int k = 0; k < 8; k++) sum = sycl::fma(a[k], b[k], sum);
    }
    for (; i < I; i += 256)
        sum = sycl::fma((float)(xr[i]), (float)(wr[i]), sum);
    partial[item_ct1.get_local_id(2)] = sum;
    item_ct1.barrier(sycl::access::fence_space::local_space);
    for (int n = 128; n; n >>= 1) {
        if (item_ct1.get_local_id(2) < n) partial[item_ct1.get_local_id(2)] +=
            partial[item_ct1.get_local_id(2) + n];
        item_ct1.barrier(sycl::access::fence_space::local_space);
    }
    if (!item_ct1.get_local_id(2)) y[(size_t)s * O + o] = partial[0];
}
static int f32_gemm_launch(DeviceContext *ctx, float *y, const float *x,
                           const float *w, int S, int I, int O,
                           sycl::queue * st) {
    unsigned gx = (unsigned)((O + F32G_N - 1) / F32G_N), big = (unsigned)((S + 63) / 64);
    if ((long)gx * big >= 2L * (ctx->sms > 0 ? ctx->sms : 1))
        coli_sycl_fprof_ev(st->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, big, gx) *
                                               sycl::range<3>(1, 1, 256),
                                           sycl::range<3>(1, 1, 256)),
                         [=](sycl::nd_item<3> item_ct1) {
                             f32_gemm_k<64>(y, x, w, S, I, O);
                         }));
        else coli_sycl_fprof_ev(st->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, (unsigned)((S + 15) / 16), gx) *
                                  sycl::range<3>(1, 1, 256),
                              sycl::range<3>(1, 1, 256)),
            [=](sycl::nd_item<3> item_ct1) {
                f32_gemm_k<16>(y, x, w, S, I, O);
            }));
    return cuda_ok(0, "chain f32 gemm launch");
}
/* y[S, O] = silu(x Wg^T) * (x Wu^T) for two GGUF block tensors of one type and shape: a decode
 * step's rows (BD_GEMV_MAX_S at most) in one GEMV pass (bd_gemv_i8_k's GLU: the bits of the two
 * GEMVs and CC_EW_SWIGLU, three kernels into one). A prompt chunk's rows went through hcm_dense_k
 * (f16 mma.sync over the blocks), which has no form here: 0, not taken (as for another type, or
 * GQ_EXACT), the caller's two GEMMs and its CC_EW_SWIGLU do it. COLI_CUDA_GLU=0 never. */
extern "C" int coli_cuda_tensor_glu_async(ColiCudaTensor *tg, ColiCudaTensor *tu, float *y_dev, const float *x_dev, int S, void *stream) {
    static int dec = -1;
    if (dec < 0) {
        const char *e = getenv("COLI_CUDA_GLU"), *f = getenv("COLI_CUDA_BD_FLOAT"), *q = getenv("GQ_EXACT");
        dec = !(e && *e == '0') && !(f && *f == '1') && !(q && *q == '1');
    }
    if (fault_injected() || !tg || !tu || !y_dev || !x_dev || S < 1 || tg->fmt < 100 || ((uintptr_t)x_dev & 15)) return 0;
    if (S <= BD_GEMV_MAX_S) {
        if (!dec || tu->fmt != tg->fmt || tu->I != tg->I || tu->O != tg->O || tu->row_words != tg->row_words || tu->device != tg->device) return 0;
        DeviceContext *bc = find_ctx(tg->device);
        const bd_u32 *tab = select_ctx(bc) ? bd_table(bc) : nullptr;
        if (!tab ||
            !bd_glu_launch(y_dev, x_dev, (const bd_u32 *)tg->weights,
                           (const bd_u32 *)tu->weights, tab, tg->fmt, tg->I,
                           tg->O, tg->row_words, S, bc->smem_optin, bc->sg,
                           (sycl::queue *)stream)) return 0;
        return cuda_ok(0, "block glu launch");
    }
    return 0;
}
/* y[j] = row ids[j] of a block-format tensor against one row of x, each the bits the whole GEMV
 * gives that row (a draft's vocabulary, cuda_chain.cu's cc_matmul_rows); 0 where not taken (another
 * format, GQ_EXACT's float form) */
extern "C" int coli_cuda_tensor_gemv_rows_async(ColiCudaTensor *t, const int *ids_dev, float *y_dev, const float *x_dev, int n, void *stream) {
    if (fault_injected() || !t || !ids_dev || !y_dev || !x_dev || n < 1 || t->fmt < 100 || ((uintptr_t)x_dev & 15)) return 0;
    static int exact = -1;
    if (exact < 0) {
        const char *e = std::getenv("COLI_CUDA_BD_FLOAT"), *q = std::getenv("GQ_EXACT");
        exact = (e && *e == '1') || (q && *q == '1');
    }
    if (exact) return 0;
    DeviceContext *ctx = find_ctx(t->device);
    const bd_u32 *tab = select_ctx(ctx) ? bd_table(ctx) : nullptr;
    if (!tab || !bd_gemv_rows_launch(y_dev, x_dev, (const bd_u32 *)t->weights,
                                     tab, ids_dev, t->fmt, t->I, t->O,
                                     t->row_words, n, ctx->smem_optin, ctx->sg,
                                     (sycl::queue *)stream)) return 0;
    return cuda_ok(0, "block rows gemv launch");
}
/* f32 rows with few outputs for a prompt chunk's rows (Qwen3.8's gated residuals: inject, 4
 * outputs of 10,240 inputs): quant_matmul's grid read a row once for every output, 105 ms of a
 * 4180-token prompt on an RTX PRO 4000. A block a row reads it once for all of them, each output's
 * per-thread sum and tree quant_matmul's (fmt 0), so the same bits. */
#define F32_FEW_MAX 8
static void f32_few_k(float *__restrict__ y, const float *__restrict__ x, const float *__restrict__ w, int I, int O) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int s = item_ct1.get_group(2);
    const float *xs = x + (size_t)s * I;
    float sum[F32_FEW_MAX];
    #pragma unroll
    for (int o = 0; o < F32_FEW_MAX; o++) sum[o] = 0.0f;
    for (int i = item_ct1.get_local_id(2); i < I;
         i += item_ct1.get_local_range(2)) {
        const float xv = xs[i];
        #pragma unroll
        for (int o = 0; o < F32_FEW_MAX; o++) if (o < O) sum[o] += xv * w[(size_t)o * I + i];
    }
    auto &partial = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[F32_FEW_MAX][256]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
#pragma unroll
    for (int o = 0; o < F32_FEW_MAX; o++) if (o < O)
        partial[o][item_ct1.get_local_id(2)] = sum[o];
    item_ct1.barrier(sycl::access::fence_space::local_space);
    for (int n = item_ct1.get_local_range(2) >> 1; n; n >>= 1) {
        if (item_ct1.get_local_id(2) < n)
#pragma unroll
            for (int o = 0; o < F32_FEW_MAX; o++) if (o < O)
                partial[o][item_ct1.get_local_id(2)] +=
                partial[o][item_ct1.get_local_id(2) + n];
        item_ct1.barrier(sycl::access::fence_space::local_space);
    }
    if ((int)item_ct1.get_local_id(2) < O)
        y[(size_t)s * O + item_ct1.get_local_id(2)] =
            partial[item_ct1.get_local_id(2)][0];
}

extern "C" int coli_cuda_tensor_gemm_async(ColiCudaTensor *t, float *y_dev,
                                           const float *x_dev, int S,
                                           void *stream) try {
    if (fault_injected()) return 0;
    if (!t || !y_dev || !x_dev || S < 1) return 0;
    if (t->fmt == 4 && t->gs <= 0) return 0;
    DeviceContext *ctx = find_ctx(t->device);
    if (!select_ctx(ctx)) return 0;
    sycl::queue * st = (sycl::queue *)stream;
    if (t->fmt >= 100) return bd_tensor_gemv(ctx, t, y_dev, x_dev, S, st);   /* a GGUF's blocks: cuda_bd.cuh */
    if (t->fmt == 1 && S <= GEMV_I8_MAX_S && t->I % 16 == 0 && t->I <= 12288 && ((uintptr_t)x_dev & 15) == 0) {   /* x in float4 loads */
        const int sg = gemv_i8_sg(ctx, t->O, t->I), wpb = 256 / sg;
        unsigned bx = (unsigned)((t->O + wpb - 1) / wpb);
        const int8_t *w = (const int8_t *)t->weights;
        const float *sc = t->scales;
        const int I = t->I, O = t->O;
        size_t prow = (size_t)(gx4(t->I / 4) + 1) * 16;   /* a padded row's bytes */
        /* the rows' local memory: 48 KiB as on CUDA (what a block asked for without an opt-in),
         * within the device's SLM (64 KiB on Xe2, 128 on PVC: never less than 48) */
        const size_t cap = ctx->smem_optin < 48 * 1024 ? ctx->smem_optin : 48 * 1024;
        /* a verify's rows share each weight load where the matrix has the blocks to fill the
         * device without its rows and shared memory holds them padded */
        if (gemv_i8_w4_on(ctx) && t->I >= 2048 && (size_t)t->O * t->I >= ((size_t)1 << 21)) {   /* long rows: four outputs a warp
                                                                               * (gemv_i8_w4), the inputs split where the outputs are
                                                                               * few; a smaller matrix's blocks are too few for it
                                                                               * (512 x 2560: 3.1 -> 4.0 us) */
            const unsigned bw = (unsigned)((t->O + wpb * GEMV_I8_OPW - 1) / (wpb * GEMV_I8_OPW));
            const int ks = gemv_i8_sk_splits(ctx, t->I, bw);
            int kc = ks > 1 ? ((t->I + ks - 1) / ks + 511) / 512 * 512 : t->I, kz = (t->I + kc - 1) / kc;
            int sr = S < GEMV_I8_NR ? S : GEMV_I8_NR;
            const size_t xrow_p = (size_t)(gx4(kc / 4) + 1) * 16, xrow_u = (size_t)kc * 4;
            const int xmode = gemv_i8_xpad_mode();
            /* S 1: no local memory, or the padded one-row kernel, as before; a padded row past the cap: the other family, as before */
            const int xpad = !(S > 1 && xrow_p <= cap &&
                               (xmode == 0 || (xmode == 2 && sr == GEMV_I8_NR && kz == 1 &&
                                               sr * xrow_p > GEMV_I8_SLOT && sr * xrow_u <= GEMV_I8_SLOT)));
            size_t xrow = xpad ? xrow_p : xrow_u;
            while (sr > 1 && sr * xrow > cap) sr--;
            /* the split's partial sums in the device's scratch arena (coli_sycl_scratch, slot 0: grown
             * once and kept, where cudaMallocAsync took and gave back a block a call): the kernels of
             * the in-order queue keep the sum after the partials. Asked for GEMV_I8_MAX_S rows, every
             * row count this path takes, so a verify's rows after a decode never regrow it (a larger
             * matrix does, retiring the old block: no wait in the frame). The arena refusing keeps
             * the unsplit kernel, as a failed cudaMallocAsync did. */
            float *part = NULL;
            if (kz > 1) part = (float *)coli_sycl_scratch(t->device, (size_t)kz * GEMV_I8_MAX_S * t->O * sizeof(float), 0);
            if (sr * xrow <= cap && (kz == 1 || part)) {
                dpct::dim3 g(bw, (unsigned)kz, (unsigned)((S + sr - 1) / sr));
                if (S == 1 && !gemv_i8_xs_on()) COLI_SG_LAUNCH(*st, sg, g, 256, 0,
                                            gemv_i8_w4<1, 1, SG>(y_dev, part, x_dev, w, sc, I, O, S, 1, kc, dpct_local));
                else if (sr == 1 && xpad) COLI_SG_LAUNCH(*st, sg, g, 256, xrow,
                                            gemv_i8_w4<1, 0, SG>(y_dev, part, x_dev, w, sc, I, O, S, 1, kc, dpct_local));
                else if (sr == 1) COLI_SG_LAUNCH(*st, sg, g, 256, xrow,   /* several rows, one a block: xrow is the unpadded row */
                                            gemv_i8_w4<1, 0, SG, 0>(y_dev, part, x_dev, w, sc, I, O, S, 1, kc, dpct_local));
                else if (xpad) COLI_SG_LAUNCH(*st, sg, g, 256, sr * xrow,
                                    gemv_i8_w4<GEMV_I8_NR, 0, SG>(y_dev, part, x_dev, w, sc, I, O, S, sr, kc, dpct_local));
                else COLI_SG_LAUNCH(*st, sg, g, 256, sr * xrow,
                                    gemv_i8_w4<GEMV_I8_NR, 0, SG, 0>(y_dev, part, x_dev, w, sc, I, O, S, sr, kc, dpct_local));
                if (part)
                    coli_sycl_fprof_ev(st->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, (unsigned)((S * t->O + 255) / 256)) * sycl::range<3>(1, 1, 256),
                                                       sycl::range<3>(1, 1, 256)),
                                     [=](sycl::nd_item<3>) { gemv_i8_sk_sum(y_dev, (const float *)part, sc, O, S, kz); }));
                return cuda_ok(0, "chain gemv launch (four outputs a warp)");
            }
        }
        int nr = S > 1 && bx >= 64 && gemv_i8_nr_on() ? (S < GEMV_I8_NR ? S : GEMV_I8_NR) : 1;
        const size_t urow = (size_t)t->I * sizeof(float);   /* an unpadded row's bytes (COLI_CUDA_GEMV_XPAD=0|2) */
        const int rmode = gemv_i8_xpad_mode();
        const int upad = nr > 1 && prow <= cap &&
                         (rmode == 0 || (rmode == 2 && nr == GEMV_I8_NR && nr * prow > GEMV_I8_SLOT && nr * urow <= GEMV_I8_SLOT));
        while (nr > 1 && nr * (upad ? urow : prow) > cap) nr--;
        if (S == 1 && !gemv_i8_xs_on())
            COLI_SG_LAUNCH(*st, sg, sycl::range<3>(1, 1, bx), 256, 0,
                           gemv_i8_rows<1, 1, 1, SG>(y_dev, x_dev, w, sc, I, O, S, 1, dpct_local));
        else if (nr > 1 && upad)
            COLI_SG_LAUNCH(*st, sg, sycl::range<3>(1, (unsigned)((S + nr - 1) / nr), bx), 256, nr * urow,
                           gemv_i8_rows<GEMV_I8_NR, 0, 0, SG>(y_dev, x_dev, w, sc, I, O, S, nr, dpct_local));
        else if (nr > 1)
            COLI_SG_LAUNCH(*st, sg, sycl::range<3>(1, (unsigned)((S + nr - 1) / nr), bx), 256, nr * prow,
                           gemv_i8_rows<GEMV_I8_NR, 1, 0, SG>(y_dev, x_dev, w, sc, I, O, S, nr, dpct_local));
        else if (prow <= cap)
            COLI_SG_LAUNCH(*st, sg, sycl::range<3>(1, (unsigned)S, bx), 256, prow,
                           gemv_i8_rows<1, 1, 0, SG>(y_dev, x_dev, w, sc, I, O, S, 1, dpct_local));
        else if ((size_t)t->I * sizeof(float) <= ctx->smem_optin)   /* unpadded: 48 KiB at I 12288, within every known SLM */
            COLI_SG_LAUNCH(*st, sg, sycl::range<3>(1, (unsigned)S, bx), 256, (size_t)I * sizeof(float),
                           gemv_i8_rows<1, 0, 0, SG>(y_dev, x_dev, w, sc, I, O, S, 1, dpct_local));
        else return 0;
        return cuda_ok(0, "chain gemv launch");
    }
    /* a prompt chunk's rows of int8 and int4 rows: the matrix engines' GEMM (coli_sycl_xmx_gemm, the f16
     * joint_matrix tiles of coli_sycl_xmx.cpp, where hcg_dense_k's and tcg_dense_k's tensor-core tiles
     * were), where the profile has the shape and the family takes the format; else the block kernel
     * below, as on a device without them */
    if (S > GEMV_I8_MAX_S && t->I % XMX_BK == 0 && ((uintptr_t)x_dev & 15) == 0 &&
        (t->fmt == 1 || t->fmt == 2 || (t->fmt == 4 && t->gs >= 16 && t->gs % 16 == 0))) {
        if (xmx_dense_ok(ctx, t->fmt, t->gs, t->I, t->O)) {
            coli_sycl_gemm g = {y_dev, t->O, x_dev, t->I, 0, 0, t->weights, t->scales, t->fmt, t->gs, S, t->I, t->O};
            g.a_s8 = xmx_dense_s8(ctx, t->fmt, t->I);
            if (coli_sycl_xmx_gemm(coli_sycl_profile_for(t->device), *st, g)) {
                xmx_note("dense gemm (prompt rows)", 1);
                if (g.a_s8) xmx_note("dense int8 prompt activations (rowwise s8)", 1);
                return 1;
            }
        }
        xmx_note("dense gemm (prompt rows)", 0);
    }
    if (t->fmt == 0 && S > GEMV_I8_MAX_S && t->O >= 16)   /* a decode step's rows keep quant_matmul's bits, as a single gate row does */
        return f32_gemm_launch(ctx, y_dev, x_dev, (const float *)t->weights, S, t->I, t->O, st);
    if (t->fmt == 0 && S <= GEMV_I8_MAX_S && t->O <= 512 && t->I >= 1024) {   /* few outputs, long rows: quant_matmul's bits */
        coli_sycl_fprof_ev(st->submit([&](sycl::handler &cgh) {
            auto t_weights_ct2 = (const float *)t->weights;
            auto t_I_ct3 = t->I;
            auto t_O_ct4 = t->O;

            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, (unsigned)S,
                                                              (unsigned)t->O) *
                                                   sycl::range<3>(1, 1, 256),
                                               sycl::range<3>(1, 1, 256)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 f32_gemv_k(y_dev, x_dev, t_weights_ct2,
                                            t_I_ct3, t_O_ct4);
                             });
        }));
        return cuda_ok(0, "chain f32 gemv launch");
    }
    if (t->fmt == 0 && S > GEMV_I8_MAX_S && t->O <= F32_FEW_MAX) {   /* few outputs, a chunk's rows: a block a row */
        coli_sycl_fprof_ev(st->submit([&](sycl::handler &cgh) {
            auto t_weights_ct2 = (const float *)t->weights;
            auto t_I_ct3 = t->I;
            auto t_O_ct4 = t->O;

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, (unsigned)S) *
                                      sycl::range<3>(1, 1, 256),
                                  sycl::range<3>(1, 1, 256)),
                [=](sycl::nd_item<3> item_ct1) {
                    f32_few_k(y_dev, x_dev, t_weights_ct2, t_I_ct3, t_O_ct4);
                });
        }));
        return cuda_ok(0, "chain f32 few-outputs launch");
    }
    if (!quant_matmul_launch(y_dev, x_dev, t->weights, t->scales, t->fmt, S,
                             t->I, t->O, row_bytes(t->fmt, t->I), t->gs, t->ng, st)) return 0;
    return cuda_ok(0, "chain gemm launch");
}
catch (sycl::exception const &exc) { return sycl_caught(exc, "coli_cuda_tensor_gemm_async"); }

/* A resident tensor's device copy for the dense chain's routed experts (cuda_chain.cu's
 * table of CcExpert): its weights, its scales and the scales' group size (fmt 4). */
extern "C" int coli_cuda_tensor_dev(const ColiCudaTensor *t, const void **w, const float **sc, int *gs) {
    if (!t || !t->weights) return 0;
    if (w) *w = t->weights;
    if (sc) *sc = t->scales;
    if (gs) *gs = t->gs;
    return 1;
}

/* This file's kernels were one CUDA module the runtime loaded lazily at the first launch
 * (cc_preload asked for it before the first request). The AOT images here (-device pvc,bmg) are
 * in the binary and the Level Zero module is built at the first launch of each kernel; nothing
 * to ask for: 1. */
extern "C" int coli_cuda_load_module(void) {
    return 1;
}

/* The dense chain's routed experts (cuda_chain.cu: cc_moe; cuda_chain.h: CcMoe): every
 * (row, slot) pair of a decode step whose expert the table holds, gate and up, then down,
 * each a warp per output row (MOE_RPW rows a warp, eight warps a block, the pair's input
 * row in shared memory, a shuffle reduction), then each row's sum in rank order. The
 * formats are the CUDA tier's: 1 int8 rows and 2 int4 rows (a scale per row), 4 int4
 * with a scale per gs inputs (gs a multiple of 16: a lane's 16 weights sit in one
 * group), 8 fp8-e4m3 with a scale per 128 x 128 block (a warp's 128 weights a step sit
 * in one), its values from a shared copy of c_e4m3. The sums are the CPU's up to the
 * order of the float additions. */
#define MOE_RPW 2
#define MOE_ROWS(wpb) ((wpb) * MOE_RPW)   /* output rows a block: a sub-group ("warp") MOE_RPW of them */
#define MOE_DIM_MAX CC_MOE_DIM_MAX
/* row o's dot with xs (n inputs), this lane's share; FMT 2/4 scaled by group here, 1/2/8
 * by row or block where noted */
template <int FMT, int SG>
__dpct_inline__ float
moe_lane_dot(const uint8_t *__restrict__ row, const float *__restrict__ sc,
             int gs, int ng, int o, int n, const float *xs, const float *slut) {
    int lane =
        sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2) &
        (SG - 1);
    float acc = 0.f;
    if (FMT == 1) {
        for (int i0 = lane * 16; i0 < n; i0 += SG * 16) {
            sycl::int4 v = *reinterpret_cast<const sycl::int4 *>(row + i0);
            const int8_t *b = reinterpret_cast<const int8_t *>(&v);
            #pragma unroll
            for (int k = 0; k < 16; k++) acc += xs[i0 + k] * (float)b[k];
        }
    } else if (FMT == 2 || FMT == 4) {
        for (int i0 = lane * 16; i0 < n; i0 += SG * 16) {
            sycl::uint2 v =
                *reinterpret_cast<const sycl::uint2 *>(row + i0 / 2);
            float part = 0.f;
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                unsigned bb = ((k < 4 ? v.x() : v.y()) >> (8 * (k & 3))) & 255u;
                int lo = (int)(bb & 15u), hi = (int)(bb >> 4);
                part += xs[i0 + 2 * k] * (float)(lo & 8 ? lo - 16 : lo) + xs[i0 + 2 * k + 1] * (float)(hi & 8 ? hi - 16 : hi);
            }
            acc += FMT == 4 ? part * sc[(size_t)o * ng + i0 / gs] : part;
        }
    } else if (FMT == 8) {
        int nb = (n + 127) >> 7;
        for (int b0 = 0; b0 < n; b0 += 128) {
            float part = 0.f;
            /* a lane's 4-byte chunks of the 128-column block: one at SG 32 (the CUDA warp's), two at 16 */
            for (int i0 = b0 + lane * 4; i0 < b0 + 128 && i0 < n; i0 += SG * 4) {
                sycl::uchar4 q =
                    *reinterpret_cast<const sycl::uchar4 *>(row + i0);
                part += xs[i0] * slut[q.x()] + xs[i0 + 1] * slut[q.y()] +
                        xs[i0 + 2] * slut[q.z()] + xs[i0 + 3] * slut[q.w()];
            }
            acc += part * sc[(size_t)(o >> 7) * nb + (b0 >> 7)];
        }
    }
    return acc;
}
template <int SG>
__dpct_inline__ float moe_warp_sum(float v) {
    for (int off = SG / 2; off > 0; off >>= 1) v += shfl_down(v, off);
    return v;
}
template <int FMT>
__dpct_inline__ size_t moe_row_bytes(int n) {
    return FMT == 2 || FMT == 4 ? (size_t)n / 2 : (size_t)n;
}

/* gate and up for pair a = blockIdx.y (row a / K, slot a % K): hid[a][o] = silu(g) * u */
template <int FMT, int SG>
static void moe_gu_k(const CcExpert *__restrict__ tab, const float *__restrict__ x,
        const int *__restrict__ ids, float *__restrict__ hid, CcMoe p,
        uint8_t *dpct_local, float const *c_e4m3) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto sm = (float *)dpct_local;
    int a = item_ct1.get_group(1), s = a / p.K, k = a - s * p.K;
    int e = ids[p.i_off + s * p.i_row + k];
    if (e < 0) return;
    CcExpert ex = tab[p.t_off + e];
    if (!ex.g) return;
    float *xs = sm, *slut = sm + p.D;
    for (int i = item_ct1.get_local_id(2); i < p.D; i += 256)
        xs[i] = x[p.x_off + (size_t)s * p.x_row + i];
    if (FMT == 8) slut[item_ct1.get_local_id(2)] =
        c_e4m3[item_ct1.get_local_id(2)];
    item_ct1.barrier();
    int lane = item_ct1.get_local_id(2) & (SG - 1),
        warp = item_ct1.get_local_id(2) / SG,
        ng = FMT == 4 ? (p.D + p.gs - 1) / p.gs : 1;
    size_t rb = moe_row_bytes<FMT>(p.D);
    for (int r = 0; r < MOE_RPW; r++) {
        int o = item_ct1.get_group(2) * MOE_ROWS(256 / SG) + r * (256 / SG) + warp;
        if (o >= p.I) break;
        float g = moe_warp_sum<SG>(moe_lane_dot<FMT, SG>((const uint8_t *)ex.g + o * rb, ex.gs, p.gs, ng, o, p.D, xs, slut));
        float u = moe_warp_sum<SG>(moe_lane_dot<FMT, SG>((const uint8_t *)ex.u + o * rb, ex.us, p.gs, ng, o, p.D, xs, slut));
        if (lane == 0) {
            if (FMT == 1 || FMT == 2) { g *= ex.gs[o]; u *= ex.us[o]; }
            hid[(size_t)a * p.I + o] = (g / (1.f + sycl::native::exp(-g))) * u;
        }
    }
}
/* down for pair a: hid[S*K + a][o] (D outputs) from hid[a] (I inputs) */
template <int FMT, int SG>
static void moe_dn_k(const CcExpert *__restrict__ tab,
                     const int *__restrict__ ids, float *__restrict__ hid,
                     CcMoe p, uint8_t *dpct_local, float const *c_e4m3) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto sm = (float *)dpct_local;
    int a = item_ct1.get_group(1), s = a / p.K, k = a - s * p.K;
    int e = ids[p.i_off + s * p.i_row + k];
    if (e < 0) return;
    CcExpert ex = tab[p.t_off + e];
    if (!ex.g) return;
    float *xs = sm, *slut = sm + p.I;
    for (int i = item_ct1.get_local_id(2); i < p.I; i += 256)
        xs[i] = hid[(size_t)a * p.I + i];
    if (FMT == 8) slut[item_ct1.get_local_id(2)] =
        c_e4m3[item_ct1.get_local_id(2)];
    item_ct1.barrier();
    int lane = item_ct1.get_local_id(2) & (SG - 1),
        warp = item_ct1.get_local_id(2) / SG,
        ng = FMT == 4 ? (p.I + p.gs - 1) / p.gs : 1;
    size_t rb = moe_row_bytes<FMT>(p.I), dn = (size_t)p.S * p.K * p.I;
    for (int r = 0; r < MOE_RPW; r++) {
        int o = item_ct1.get_group(2) * MOE_ROWS(256 / SG) + r * (256 / SG) + warp;
        if (o >= p.D) break;
        float v = moe_warp_sum<SG>(moe_lane_dot<FMT, SG>((const uint8_t *)ex.d + o * rb, ex.ds, p.gs, ng, o, p.I, xs, slut));
        if (lane == 0) {
            if (FMT == 1 || FMT == 2) v *= ex.ds[o];
            hid[dn + (size_t)a * p.D + o] = v;
        }
    }
}
/* y[s] = the host's row (or 0), then each held slot's w * down, in rank order */
static void moe_sum_k(const CcExpert *__restrict__ tab,
                      const int *__restrict__ ids, const float *__restrict__ w,
                      const float *__restrict__ hid,
                      const float *__restrict__ hy, float *__restrict__ y,
                      CcMoe p) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int s = item_ct1.get_group(1),
        d = item_ct1.get_group(2) * 256 + item_ct1.get_local_id(2);
    if (d >= p.D) return;
    size_t dn = (size_t)(p.lay_S > 0 ? p.lay_S : p.S) * p.K * p.I;   /* the layout's: lay_S rows (CcMoe) */
    const int hs = p.lay_s0 + s;
    float v = p.h_off >= 0 ? hy[p.h_off + (size_t)s * p.D + d] : 0.f;
    /* eight slots a pass: their ids, then their table entries, then their rows, each a batch
     * of independent loads (slot by slot they were three dependent loads each); the sum keeps
     * the slots' order */
    for (int k0 = 0; k0 < p.K; k0 += 8) {
        int e[8];
        bool on[8];
        float wk[8], hk[8];
        #pragma unroll
        for (int j = 0; j < 8; j++) e[j] = k0 + j < p.K ? ids[p.i_off + s * p.i_row + k0 + j] : -1;
        #pragma unroll
        for (int j = 0; j < 8; j++) on[j] = e[j] >= 0 && tab[p.t_off + e[j]].g;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            wk[j] = on[j] ? w[p.w_off + s * p.w_row + k0 + j] : 0.f;
            hk[j] = on[j] ? hid[dn + (size_t)(hs * p.K + k0 + j) * p.D + d] : 0.f;
        }
        #pragma unroll
        for (int j = 0; j < 8; j++) if (on[j]) v += wk[j] * hk[j];
    }
    y[p.y_off + (size_t)s * p.y_row + d] = v;
}
/* A prompt chunk's slots by expert (S > CC_MOE_FRAME_ROWS): counts, offsets and the
 * GEMMs' tiles (64 rows of one expert each), each slot's place. The integers sit in hid
 * after the floats: cnt[E] base[E + 1] cur[E] ntile[1] te[T] tr[T] order[A] posof[A]. */
struct MoeSort { int *cnt, *base, *cur, *ntile, *te, *tr, *order, *posof; int T, TR; };   /* TR: a tile's rows, 64 or 128 (hcm_moe's) */
static MoeSort moe_sort_of(float *hid, const CcMoe *p) {
    size_t A = (size_t)(p->lay_S > 0 ? p->lay_S : p->S) * p->K;
    int *q = reinterpret_cast<int *>(hid + A * (size_t)(p->I + p->D));
    MoeSort m;
    m.T = (int)((A + 63) / 64 + (size_t)(p->E < (int)A ? p->E : (int)A)); m.TR = 64;
    m.cnt = q; m.base = q + p->E; m.cur = m.base + p->E + 1; m.ntile = m.cur + p->E;
    m.te = m.ntile + 1; m.tr = m.te + m.T; m.order = m.tr + m.T; m.posof = m.order + A;
    return m;
}
__dpct_inline__ int moe_slot_expert(const CcExpert *tab, const int *ids,
                                    const CcMoe &p, int a) {
    int s = a / p.K, k = a - s * p.K, e = ids[p.i_off + s * p.i_row + k];
    return e >= 0 && e < p.E && tab[p.t_off + e].g ? e : -1;
}
static void moe_sort_count_k(const CcExpert *__restrict__ tab, const int *__restrict__ ids, MoeSort m, CcMoe p) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int a = item_ct1.get_group(2) * 256 + item_ct1.get_local_id(2);
    if (a >= p.S * p.K) return;
    int e = moe_slot_expert(tab, ids, p, a);
    if (e >= 0)
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            &m.cnt[e], 1);
}
static void moe_sort_scan_k(MoeSort m, CcMoe p) {
    if (sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2) !=
        0) return;
    int b = 0, t = 0;
    for (int e = 0; e < p.E; e++) {
        m.base[e] = b; m.cur[e] = 0;
        for (int r = 0; r < m.cnt[e] && t < m.T; r += m.TR) { m.te[t] = e; m.tr[t] = r; t++; }
        b += m.cnt[e];
    }
    m.base[p.E] = b; *m.ntile = t;
}
static void moe_sort_place_k(const CcExpert *__restrict__ tab, const int *__restrict__ ids, MoeSort m, CcMoe p) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int a = item_ct1.get_group(2) * 256 + item_ct1.get_local_id(2);
    if (a >= p.S * p.K) return;
    int e = moe_slot_expert(tab, ids, p, a);
    if (e < 0) { m.posof[a] = -1; return; }
    int pos =
        m.base[e] +
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            &m.cur[e], 1);
    m.order[pos] = a; m.posof[a] = pos;
}
/* y[s] = the host's row (or 0), then each held slot's w * down, in rank order (its place) */
static void moe_rows_sum_k(const float *__restrict__ w, const float *__restrict__ hid, const float *__restrict__ hy,
                                      float *__restrict__ y, MoeSort m, CcMoe p) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int s = item_ct1.get_group(1),
        d = item_ct1.get_group(2) * 256 + item_ct1.get_local_id(2);
    if (d >= p.D) return;
    const float *yd = hid + (size_t)(p.lay_S > 0 ? p.lay_S : p.S) * p.K * p.I;   /* the layout's (CcMoe.lay_S) */
    float v = p.h_off >= 0 ? hy[p.h_off + (size_t)s * p.D + d] : 0.f;
    for (int k = 0; k < p.K; k++) {
        int pos = m.posof[(p.lay_s0 + s) * p.K + k];
        if (pos >= 0) v += w[p.w_off + s * p.w_row + k] * yd[(size_t)pos * p.D + d];
    }
    y[p.y_off + (size_t)s * p.y_row + d] = v;
}
/* A decode step's routed experts in a dense GEMV's layout (vk_chain.c's chain_moe_gv.comp
 * on Vulkan): the input row staged once in shared memory with a float4 of pad every eight
 * (a lane's 16-byte step reads consecutive float4s, the next lane's start nine on: no bank
 * conflict), each output row lpr lanes of 16-byte weight steps, gate and up in the same loop,
 * the lanes joined by shuffles. moe_gu_k and moe_dn_k took a warp a row with 8-byte steps
 * and lanes 16 floats apart: GLM-5.2's eight int4-g64 experts (6144 x 2048) 537 us a step on
 * an RTX PRO 4000. Formats 1, 2 and 4 (the CUDA tier's signed nibbles, weight 2j low), rows
 * 16-byte aligned (cudaMalloc'd tensors), the staged row within 48 KiB; else moe_launch. */
#define MOE_GV_SMEM (49152 - 4096)   /* the staged row's dynamic shared memory: the slots' table beside it */
static inline int moe_gv_lanes(int fmt, int n, int sg) {   /* about eight 16-byte steps a lane, 1 to the sub-group */
    int steps = n / (fmt == 1 ? 16 : 32), l = 1;
    while (l < sg && l * 2 * 8 <= steps) l *= 2;
    return l;
}
static inline size_t moe_gv_smem(int n) {
    return ((size_t)n / 4 + (size_t)n / 32 + 1) * sizeof(sycl::float4);
}
__dpct_inline__ int mgx(int k) { return k + (k >> 3); }
__dpct_inline__ float mg_s4(unsigned w, int i) {
    return (float)((int)(w << (28 - 4 * i)) >> 28);
}
__dpct_inline__ float mg_s8(unsigned w, int i) {
    return (float)((int)(w << (24 - 8 * i)) >> 24);
}
__dpct_inline__ float mg_dot4w(unsigned w, const sycl::float4 a,
                               const sycl::float4 b) {
    return a.x() * mg_s4(w, 0) + a.y() * mg_s4(w, 1) + a.z() * mg_s4(w, 2) +
           a.w() * mg_s4(w, 3) + b.x() * mg_s4(w, 4) + b.y() * mg_s4(w, 5) +
           b.z() * mg_s4(w, 6) + b.w() * mg_s4(w, 7);
}
__dpct_inline__ float mg_dot8w(unsigned w, const sycl::float4 a) {
    return a.x() * mg_s8(w, 0) + a.y() * mg_s8(w, 1) + a.z() * mg_s8(w, 2) +
           a.w() * mg_s8(w, 3);
}
/* one 16-byte step q of a row against the staged row: 16 int8 or 32 int4 weights */
template <int FMT>
__dpct_inline__ float mg_step(sycl::uint4 v, int q, const sycl::float4 *xs) {
    if (FMT == 1) {
        int b = q * 4;
        return mg_dot8w(v.x(), xs[mgx(b)]) + mg_dot8w(v.y(), xs[mgx(b + 1)]) +
               mg_dot8w(v.z(), xs[mgx(b + 2)]) +
               mg_dot8w(v.w(), xs[mgx(b + 3)]);
    }
    int b = q * 8;
    return mg_dot4w(v.x(), xs[mgx(b)], xs[mgx(b + 1)]) +
           mg_dot4w(v.y(), xs[mgx(b + 2)], xs[mgx(b + 3)]) +
           mg_dot4w(v.z(), xs[mgx(b + 4)], xs[mgx(b + 5)]) +
           mg_dot4w(v.w(), xs[mgx(b + 6)], xs[mgx(b + 7)]);
}
__dpct_inline__ float mg_lanes_sum(float v, int lpr) {
    for (int off = lpr >> 1; off > 0; off >>= 1) v += shfl_xor(v, off);
    return v;
}
/* gate and up: block (b, s) stages row s once and takes units u = k * I + o (slot k's row o),
 * a unit lpr lanes; hid[s*K + k][o] = silu(gate . x) * (up . x) */
template <int FMT, int SG>
static void moe_gv_gu_k(const CcExpert *__restrict__ tab,
                        const float *__restrict__ x,
                        const int *__restrict__ ids, float *__restrict__ hid,
                        CcMoe p, int lpr, uint8_t *dpct_local) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto mg_xs = (sycl::float4 *)dpct_local;
    sycl::float4 *xs = mg_xs;
    auto &ke = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        CcExpert[CC_ROUTE_KMAX]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &kl = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        int[CC_ROUTE_KMAX]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    int s = item_ct1.get_group(1), tid = item_ct1.get_local_id(2);
    size_t xb = (size_t)p.x_off + (size_t)s * p.x_row;
    for (int i = tid; i < p.D / 4; i += 256) xs[mgx(i)] = sycl::float4(
        x[xb + 4 * i], x[xb + 4 * i + 1], x[xb + 4 * i + 2], x[xb + 4 * i + 3]);
    if (tid < p.K) {
        int e = ids[p.i_off + s * p.i_row + tid];
        int on = e >= 0 && tab[p.t_off + e].g != nullptr;
        kl[tid] = on;
        if (on) ke[tid] = tab[p.t_off + e];
    }
    item_ct1.barrier();
    int lane = tid & (SG - 1), warp = tid / SG, rps = SG / lpr, sub = lane / lpr, ql = lane % lpr;
    int per = FMT == 1 ? 16 : 32, nq = p.D / per, ng = FMT == 4 ? (p.D + p.gs - 1) / p.gs : 1;
    int NU = p.K * p.I, stride = (int)item_ct1.get_group_range(2) * (256 / SG) * rps;
    for (int u0 = ((int)item_ct1.get_group(2) * (256 / SG) + warp) * rps; u0 < NU;
         u0 += stride) { /* uniform in the warp */
        int u = sycl::min(u0 + sub, NU - 1), k = u / p.I, o = u - k * p.I;
        bool live = u0 + sub < NU && kl[k];
        float g = 0.f, uu = 0.f;
        if (live) {
            const sycl::uint4 *wg =
                                  (const sycl::uint4 *)ke[k].g + (size_t)o * nq,
                              *wu =
                                  (const sycl::uint4 *)ke[k].u + (size_t)o * nq;
            for (int q = ql; q < nq; q += lpr) {
                sycl::uint4 vg = wg[q], vu = wu[q];
                if (FMT == 4) {
                    int gi = o * ng + (q * 32) / p.gs;
                    g += mg_step<FMT>(vg, q, xs) * ke[k].gs[gi]; uu += mg_step<FMT>(vu, q, xs) * ke[k].us[gi];
                } else { g += mg_step<FMT>(vg, q, xs); uu += mg_step<FMT>(vu, q, xs); }
            }
        }
        g = mg_lanes_sum(g, lpr); uu = mg_lanes_sum(uu, lpr);
        if (ql == 0 && live) {
            if (FMT != 4) { g *= ke[k].gs[o]; uu *= ke[k].us[o]; }
            hid[((size_t)s * p.K + k) * p.I + o] =
                (g / (1.f + sycl::native::exp(-g))) * uu;
        }
    }
}
/* down: block (pair j, tile) stages pair j's hidden row and takes the tile's rows, a row
 * lpr lanes: hid[S*K + j][o] = down_e[o] . hid[j] */
template <int FMT, int SG>
static void moe_gv_dn_k(const CcExpert *__restrict__ tab,
                        const int *__restrict__ ids, float *__restrict__ hid,
                        CcMoe p, int lpr, int tiles, uint8_t *dpct_local) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto mg_xs = (sycl::float4 *)dpct_local;
    sycl::float4 *xs = mg_xs;
    int wgi = (int)(item_ct1.get_group(2) +
                    item_ct1.get_group(1) * item_ct1.get_group_range(2)),
        j = wgi / tiles, tile = wgi - j * tiles;
    if (j >= p.S * p.K) return;
    int s = j / p.K, k = j - s * p.K, e = ids[p.i_off + s * p.i_row + k];
    if (e < 0) return;
    CcExpert ex = tab[p.t_off + e];
    if (!ex.g) return;
    int tid = item_ct1.get_local_id(2);
    const float *hr = hid + (size_t)j * p.I;
    for (int i = tid; i < p.I / 4; i += 256) xs[mgx(i)] =
        sycl::float4(hr[4 * i], hr[4 * i + 1], hr[4 * i + 2], hr[4 * i + 3]);
    item_ct1.barrier();
    int lane = tid & (SG - 1), warp = tid / SG, rps = SG / lpr, sub = lane / lpr, ql = lane % lpr;
    int per = FMT == 1 ? 16 : 32, nq = p.I / per, ng = FMT == 4 ? (p.I + p.gs - 1) / p.gs : 1;
    int per_tile = (p.D + tiles - 1) / tiles, lo = tile * per_tile,
        hi = sycl::min(lo + per_tile, p.D);
    size_t dn = (size_t)p.S * p.K * p.I;
    for (int o0 = lo + warp * rps; o0 < hi; o0 += (256 / SG) * rps) {   /* uniform in the warp */
        int o = o0 + sub;
        bool live = o < hi;
        int oc = sycl::min(o, p.D - 1);
        float v = 0.f;
        if (live) {
            const sycl::uint4 *wd = (const sycl::uint4 *)ex.d + (size_t)oc * nq;
            for (int q = ql; q < nq; q += lpr) {
                float d = mg_step<FMT>(wd[q], q, xs);
                v += FMT == 4 ? d * ex.ds[oc * ng + (q * 32) / p.gs] : d;
            }
        }
        v = mg_lanes_sum(v, lpr);
        if (ql == 0 && live) hid[dn + (size_t)j * p.D + o] = FMT != 4 ? v * ex.ds[oc] : v;
    }
}
static int moe_gv_sg(const DeviceContext *ctx) {
    static const int requested = kernel_sg_env("COLI_CUDA_MOE_SG");
    const auto &pr = coli_sycl_profile_for(ctx->device);
    if (requested) {
        if (coli_sycl_sg_offered(pr, requested)) return requested;
        static bool warned = false;
        if (!warned) { fprintf(stderr, "[SYCL] COLI_CUDA_MOE_SG=%d ignored (not offered)\n", requested); warned = true; }
    }
    return pr.arch == COLI_SYCL_ARCH_PVC && coli_sycl_sg_offered(pr, 16) ? 16 : ctx->sg;
}
template <int FMT>
static int moe_gv_launch(const CcExpert *tab, const float *x, const int *ids,
                         float *hid, const CcMoe *p, sycl::queue * st) {
    const DeviceContext *ctx = cur_ctx();
    const int sg = moe_gv_sg(ctx), wpb = 256 / sg;
    int l0 = moe_gv_lanes(FMT, p->D, sg), l1 = moe_gv_lanes(FMT, p->I, sg);
    int upw = wpb * (sg / l0), nwg = (p->K * p->I + upw - 1) / upw;   /* units a block */
    if (nwg > 1024) nwg = 1024;
    {
        auto p_ct4 = *p;
        COLI_SG_LAUNCH(*st, sg, sycl::range<3>(1, (unsigned)p->S, (unsigned)nwg), 256, moe_gv_smem(p->D),
                       moe_gv_gu_k<FMT, SG>(tab, x, ids, hid, p_ct4, l0, dpct_local));
    }
    int rpp = wpb * (sg / l1), tiles = (p->D + rpp - 1) / rpp;   /* a down tile: one pass of the block */
    uint64_t nb = (uint64_t)p->S * p->K * tiles;
    unsigned gx = nb < 65535 ? (unsigned)nb : 65535u, gy = (unsigned)((nb + gx - 1) / gx);
    {
        auto p_ct3 = *p;
        COLI_SG_LAUNCH(*st, sg, sycl::range<3>(1, gy, gx), 256, moe_gv_smem(p->I),
                       moe_gv_dn_k<FMT, SG>(tab, ids, hid, p_ct3, l1, tiles, dpct_local));
    }
    return cuda_ok(0, "chain moe gv launch");
}
/* A GGUF's routed experts (CcMoe.fmt >= 100: gate and up of ggml type TG, down of type TD),
 * in the CPU's integer form (cuda_bd.cuh): gate and up as moe_gv_gu_k takes them, block
 * (b, s) quantizing row s of x once and taking units u = k * I + o (slot k's row o) over lpr
 * lanes; down as moe_gv_dn_k, block (pair j, tile) quantizing pair j's hidden row. A row of
 * the matrices is rw words (the file's row padded). */
/* A host pool's experts (CcExpert.host: page-locked host memory the device reads over the
 * bus) through these kernels' word loads came at a third of the bus's rate (Qwen3.8 IQ2_S
 * gate/up, an RTX PRO 4000 on PCIe 4.0 x8: 3.3 GB/s). With a staging (coli_cuda_moe_stage)
 * a decode step's distinct such experts, in slot order, are first copied there in 16-byte
 * loads (bd_stage_k), and the kernels read them there: the map at the staging's start says
 * slot j's place (-1: none, the entry as it is). Up to BD_STAGE_SLOTS slots (S * K) a step;
 * an expert's span (g to the end of down, rounded up to 16 bytes) must fit a place. */
#define BD_STAGE_SLOTS (CC_MOE_STAGE_MAP / 4)
static inline void bd_stage_at(CcExpert *x, const char *stg, size_t sslot, int pc) {
    const char *g0 = (const char *)x->g, *b = stg + CC_MOE_STAGE_MAP + (size_t)pc * sslot;
    x->g = b; x->u = b + ((const char *)x->u - g0); x->d = b + ((const char *)x->d - g0);
}
static void bd_stage_k(const CcExpert *__restrict__ tab,
                       const int *__restrict__ ids, CcMoe p,
                       char *__restrict__ stg, size_t sslot, int nslot,
                       size_t gub, size_t dnb) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto &eid = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        int[BD_STAGE_SLOTS]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &fo = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        int[BD_STAGE_SLOTS]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &pl = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        int[BD_STAGE_SLOTS]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &src = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        const char *[BD_STAGE_SLOTS]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &len = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        size_t[BD_STAGE_SLOTS]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    int n = p.S * p.K, j = item_ct1.get_local_id(2);
    if (j < n) {   /* each slot's expert, whether the bus serves it, its span */
        int s = j / p.K, e = ids[p.i_off + s * p.i_row + (j - s * p.K)];
        int h = 0;
        if (e >= 0) {
            CcExpert x = tab[p.t_off + e];
            const char *g = (const char *)x.g, *u = (const char *)x.u, *d = (const char *)x.d;
            size_t sp = 0;
            if (x.host && g && u >= g && d >= g) {
                size_t a = (size_t)(u - g) + gub, b = (size_t)(d - g) + dnb;
                sp = ((a > b ? a : b) + 15) / 16 * 16;
            }
            h = sp && sp <= sslot;
            src[j] = g; len[j] = sp;
        }
        eid[j] = h ? e : -1;
    }
    item_ct1.barrier();
    if (j < n) {   /* the slot an expert first takes */
        int f = j;
        if (eid[j] >= 0) for (int q = 0; q < j; q++) if (eid[q] == eid[j]) { f = q; break; }
        fo[j] = f;
    }
    item_ct1.barrier();
    if (j < n) {   /* the place: the distinct ones before it, in slot order */
        int r = -1;
        if (eid[j] >= 0) {
            int f = fo[j];
            r = 0;
            for (int q = 0; q < f; q++) r += eid[q] >= 0 && fo[q] == q;
            if (r >= nslot) r = -1;
        }
        pl[j] = r;
        if (item_ct1.get_group(2) == 0 &&
            item_ct1.get_group(1) == 0)((int *)stg)[j] = r;
    }
    item_ct1.barrier();
    int place = (int)item_ct1.get_group(1), from = -1;
    for (int q = 0; q < n; q++) if (pl[q] == place && fo[q] == q) { from = q; break; }
    if (from < 0) return;
    const sycl::uint4 *a = (const sycl::uint4 *)src[from];
    sycl::uint4 *b =
        (sycl::uint4 *)(stg + CC_MOE_STAGE_MAP + (size_t)place * sslot);
    size_t nv = len[from] / 16, step = (size_t)item_ct1.get_group_range(2) *
                                       item_ct1.get_local_range(2);
    for (size_t v =
             (size_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
             item_ct1.get_local_id(2);
         v < nv; v += 4 * step) { /* four loads in flight a thread */
        sycl::uint4 r0 = a[v], r1, r2, r3;
        if (v + step < nv) r1 = a[v + step];
        if (v + 2 * step < nv) r2 = a[v + 2 * step];
        if (v + 3 * step < nv) r3 = a[v + 3 * step];
        b[v] = r0;
        if (v + step < nv) b[v + step] = r1;
        if (v + 2 * step < nv) b[v + 2 * step] = r2;
        if (v + 3 * step < nv) b[v + 3 * step] = r3;
    }
}
template <int TG, int SG>
static void
moe_bd_gu_k(const CcExpert *__restrict__ tab, const float *__restrict__ x,
            const int *__restrict__ ids, float *__restrict__ hid, CcMoe p,
            int lpr, int rw, const bd_u32 *__restrict__ btab,
            const char *__restrict__ stg, size_t sslot, uint8_t *dpct_local) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto bd_sm4 = (sycl::uint4 *)dpct_local;
    auto &ke = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        CcExpert[CC_ROUTE_KMAX]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &kl = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        int[CC_ROUTE_KMAX]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    constexpr int TW = bd_tab_words(TG);
    BdTabs tb = bd_tables_fill<TG>(reinterpret_cast<bd_u32 *>(bd_sm4), btab);
    int s = item_ct1.get_group(1), tid = item_ct1.get_local_id(2),
        nb = p.D / 32, qr = bd_xq_row(p.D);
    sycl::uint4 *xq = bd_sm4 + TW / 4;
    float *dx = reinterpret_cast<float *>(xq + qr);
    int *xs = reinterpret_cast<int *>(dx + nb);
    bdx_quant(x + p.x_off + (size_t)s * p.x_row, 1, p.D, xq, dx, xs);
    if (tid < p.K) {
        int e = ids[p.i_off + s * p.i_row + tid];
        int on = e >= 0 && tab[p.t_off + e].g != nullptr;
        kl[tid] = on;
        if (on) {
            CcExpert ex = tab[p.t_off + e];
            int pc = stg ? ((const int *)stg)[s * p.K + tid] : -1;
            if (pc >= 0) bd_stage_at(&ex, stg, sslot, pc);
            ke[tid] = ex;
        }
    }
    item_ct1.barrier();
    BdX X = {xq, dx, xs, qr, nb};
    int lane = tid & (SG - 1), warp = tid / SG, rps = SG / lpr, sub = lane / lpr, ql = lane % lpr, nq = p.D / 32;
    int NU = p.K * p.I, stride = (int)item_ct1.get_group_range(2) * (256 / SG) * rps;
    for (int u0 = ((int)item_ct1.get_group(2) * (256 / SG) + warp) * rps; u0 < NU;
         u0 += stride) { /* uniform in the warp */
        int u = sycl::min(u0 + sub, NU - 1), k = u / p.I, o = u - k * p.I;
        bool live = u0 + sub < NU && kl[k];
        float g = 0.f, uu = 0.f;
        if (live) {
            const bd_u32 *wg = (const bd_u32 *)ke[k].g + (size_t)o * rw, *wu = (const bd_u32 *)ke[k].u + (size_t)o * rw;
            for (int q = ql; q < nq; q += lpr) { bdi_step<TG, 1>(wg, (bd_u32)q, X, 1, tb, &g); bdi_step<TG, 1>(wu, (bd_u32)q, X, 1, tb, &uu); }
        }
        g = mg_lanes_sum(g, lpr); uu = mg_lanes_sum(uu, lpr);
        if (ql == 0 && live) hid[((size_t)s * p.K + k) * p.I + o] =
            (g / (1.f + sycl::native::exp(-g))) * uu;
    }
}

template <int TD, int SG>
static void
moe_bd_dn_k(const CcExpert *__restrict__ tab, const int *__restrict__ ids,
            float *__restrict__ hid, CcMoe p, int lpr, int tiles, int rw,
            const bd_u32 *__restrict__ btab, const char *__restrict__ stg,
            size_t sslot, uint8_t *dpct_local) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto bd_sm4 = (sycl::uint4 *)dpct_local;
    int wgi = (int)(item_ct1.get_group(2) +
                    item_ct1.get_group(1) * item_ct1.get_group_range(2)),
        j = wgi / tiles, tile = wgi - j * tiles;
    if (j >= p.S * p.K) return;
    int s = j / p.K, k = j - s * p.K, e = ids[p.i_off + s * p.i_row + k];
    if (e < 0) return;
    CcExpert ex = tab[p.t_off + e];
    if (!ex.g) return;
    int pc = stg ? ((const int *)stg)[j] : -1;
    if (pc >= 0) bd_stage_at(&ex, stg, sslot, pc);
    constexpr int TW = bd_tab_words(TD);
    BdTabs tb = bd_tables_fill<TD>(reinterpret_cast<bd_u32 *>(bd_sm4), btab);
    int nb = p.I / 32, qr = bd_xq_row(p.I);
    sycl::uint4 *xq = bd_sm4 + TW / 4;
    float *dx = reinterpret_cast<float *>(xq + qr);
    int *xs = reinterpret_cast<int *>(dx + nb);
    bdx_quant(hid + (size_t)j * p.I, 1, p.I, xq, dx, xs);
    item_ct1.barrier();
    BdX X = {xq, dx, xs, qr, nb};
    int lane = item_ct1.get_local_id(2) & (SG - 1),
        warp = item_ct1.get_local_id(2) / SG, rps = SG / lpr, sub = lane / lpr,
        ql = lane % lpr, nq = p.I / 32;
    int per_tile = (p.D + tiles - 1) / tiles, lo = tile * per_tile,
        hi = sycl::min(lo + per_tile, p.D);
    size_t dn = (size_t)p.S * p.K * p.I;
    for (int o0 = lo + warp * rps; o0 < hi; o0 += (256 / SG) * rps) {   /* uniform in the warp */
        int o = o0 + sub;
        bool live = o < hi;
        float v = 0.f;
        if (live) {
            const bd_u32 *wd = (const bd_u32 *)ex.d + (size_t)o * rw;
            for (int q = ql; q < nq; q += lpr) bdi_step<TD, 1>(wd, (bd_u32)q, X, 1, tb, &v);
        }
        v = mg_lanes_sum(v, lpr);
        if (ql == 0 && live) hid[dn + (size_t)j * p.D + o] = v;
    }
}

template <int TG>
static int moe_bd_gu_launch(const CcExpert *tab, const float *x, const int *ids,
                            float *hid, const CcMoe *p, const bd_u32 *btab,
                            size_t smem_max, int sg, const char *stg, size_t sslot,
                            sycl::queue * st) {
    size_t sm = (size_t)bd_tab_words(TG) * 4 + bd_x_bytes(p->D, 1);
    if (sm > smem_max) return 0;   /* the device's SLM: no opt-in past it exists */
    int lpr = bd_lpr(p->D, sg), upw = 256 / lpr, nwg = (p->K * p->I + upw - 1) / upw;   /* units a block */
    if (nwg > 1024) nwg = 1024;
    int rw = (int)((size_t)p->D / BD_BLK[TG] * BD_BYTES[TG] + 3) / 4;
    auto p_ct4 = *p;
    COLI_SG_LAUNCH(*st, sg, sycl::range<3>(1, (unsigned)p->S, (unsigned)nwg), 256, sm,
                   moe_bd_gu_k<TG, SG>(tab, x, ids, hid, p_ct4, lpr, rw, btab, stg, sslot, dpct_local));
    return 1;
}
template <int TD>
static int moe_bd_dn_launch(const CcExpert *tab, const int *ids, float *hid,
                            const CcMoe *p, const bd_u32 *btab, size_t smem_max, int sg,
                            const char *stg, size_t sslot, sycl::queue * st) {
    size_t sm = (size_t)bd_tab_words(TD) * 4 + bd_x_bytes(p->I, 1);
    if (sm > smem_max) return 0;
    int lpr = bd_lpr(p->I, sg), rpp = 256 / lpr, tiles = (p->D + rpp - 1) / rpp;   /* a down tile: one pass of the block */
    uint64_t nb = (uint64_t)p->S * p->K * tiles;
    unsigned gx = nb < 65535 ? (unsigned)nb : 65535u, gy = (unsigned)((nb + gx - 1) / gx);
    int rw = (int)((size_t)p->I / BD_BLK[TD] * BD_BYTES[TD] + 3) / 4;
    auto p_ct3 = *p;
    COLI_SG_LAUNCH(*st, sg, sycl::range<3>(1, gy, gx), 256, sm,
                   moe_bd_dn_k<TD, SG>(tab, ids, hid, p_ct3, lpr, tiles, rw, btab, stg, sslot, dpct_local));
    return 1;
}
#define BD_MOE_TYPES(X) X(6) X(7) X(8) X(10) X(11) X(12) X(13) X(14) X(16) X(17) X(18) X(19) X(20) X(21) X(22) X(23) X(29) X(42)
/* 1 = both types have an integer step and the widths whole blocks of them */
static int moe_bd_ok(const CcMoe *p) {
    int tg = p->fmt - 100, td = (p->fmt_d ? p->fmt_d : p->fmt) - 100;
    return tg >= 0 && tg < 43 && td >= 0 && td < 43 && bd_fmt_ok(p->fmt) && bd_fmt_ok(td + 100) && bdi_has(tg) && bdi_has(td) &&
           p->D % BD_BLK[tg] == 0 && p->I % BD_BLK[td] == 0 && p->K <= CC_ROUTE_KMAX && !p->planar;
}
static int moe_bd_launch(DeviceContext *ctx, const CcExpert *tab,
                         const float *x, const int *ids, float *hid,
                         const CcMoe *p, sycl::queue * st) {
    const bd_u32 *btab = bd_table(ctx);
    if (!btab || ((uintptr_t)(x + p->x_off) & 15) || p->x_row % 4) return 0;
    int ok = 0, tg = p->fmt - 100, td = (p->fmt_d ? p->fmt_d : p->fmt) - 100;
    const char *stg = nullptr;
    size_t sslot = 0;
    if (ctx->moe_stage && p->S * p->K <= BD_STAGE_SLOTS) {   /* the step's host experts to the staging first */
        size_t gub = (size_t)p->I * (((size_t)p->D / BD_BLK[tg] * BD_BYTES[tg] + 3) / 4 * 4);
        size_t dnb = (size_t)p->D * (((size_t)p->I / BD_BLK[td] * BD_BYTES[td] + 3) / 4 * 4);
        stg = ctx->moe_stage; sslot = ctx->moe_stage_slot;
        st->submit([&](sycl::handler &cgh) {
            CcMoe p_ct2 = *p;
            auto ctx_moe_stage_ct3 = ctx->moe_stage;
            auto ctx_moe_stage_n_ct5 = ctx->moe_stage_n;

            cgh.parallel_for(
                sycl::nd_range<3>(
                    sycl::range<3>(1, (unsigned)ctx->moe_stage_n, 48u) *
                        sycl::range<3>(1, 1, 256),
                    sycl::range<3>(1, 1, 256)),
                [=](sycl::nd_item<3> item_ct1) {
                    bd_stage_k(tab, ids, p_ct2, ctx_moe_stage_ct3, sslot,
                               ctx_moe_stage_n_ct5, gub, dnb);
                });
        });
    }
    switch (tg) {
#define BD_GU_CASE(t) case t: ok = moe_bd_gu_launch<t>(tab, x, ids, hid, p, btab, ctx->smem_optin, ctx->sg, stg, sslot, st); break;
        BD_MOE_TYPES(BD_GU_CASE)
#undef BD_GU_CASE
        default: return 0;
    }
    if (!ok) return 0;
    switch (td) {
#define BD_DN_CASE(t) case t: ok = moe_bd_dn_launch<t>(tab, ids, hid, p, btab, ctx->smem_optin, ctx->sg, stg, sslot, st); break;
        BD_MOE_TYPES(BD_DN_CASE)
#undef BD_DN_CASE
        default: return 0;
    }
    return ok && cuda_ok(0, "chain moe block launch");
}
static int moe_gv_ok(const CcMoe *p) {
    static int on = -1;
    if (on < 0) { const char *e = getenv("COLI_CUDA_MOE_GV"); on = !(e && *e == '0'); }
    return on && (p->fmt == 1 || p->fmt == 2 || (p->fmt == 4 && p->gs % 32 == 0)) && p->K <= CC_ROUTE_KMAX &&
           moe_gv_smem(p->D) <= MOE_GV_SMEM && moe_gv_smem(p->I) <= MOE_GV_SMEM;
}

template <int FMT>
static int moe_launch(const CcExpert *tab, const float *x, const int *ids,
                      float *hid, const CcMoe *p, sycl::queue * st) {
    const DeviceContext *ctx = cur_ctx();
    const int rows = MOE_ROWS(ctx->wpb);
    size_t smg = ((size_t)p->D + (FMT == 8 ? 256 : 0)) * sizeof(float),
           smd = ((size_t)p->I + (FMT == 8 ? 256 : 0)) * sizeof(float);
    if (smg > ctx->smem_optin || smd > ctx->smem_optin) return 0;   /* an input row and the fp8 table in the SLM */
    dpct::dim3 gg((unsigned)((p->I + rows - 1) / rows), (unsigned)(p->S * p->K)),
        gd((unsigned)((p->D + rows - 1) / rows), (unsigned)(p->S * p->K));
    const float *c_e4m3_ptr_ct1 = ctx->e4m3;
    {
        auto p_ct4 = *p;
        COLI_SG_LAUNCH(*st, ctx->sg, gg, 256, smg, moe_gu_k<FMT, SG>(tab, x, ids, hid, p_ct4, dpct_local, c_e4m3_ptr_ct1));
    }
    {
        auto p_ct3 = *p;
        COLI_SG_LAUNCH(*st, ctx->sg, gd, 256, smd, moe_dn_k<FMT, SG>(tab, ids, hid, p_ct3, dpct_local, c_e4m3_ptr_ct1));
    }
    return cuda_ok(0, "chain moe launch");
}
/* A prompt chunk's slots sorted by expert (moe_sort_*: counts, offsets, the tiles of 64 rows of one
 * expert each, each slot's place), then the matrix engines' grouped GEMM over the tiles
 * (coli_sycl_xmx_moe: gate and up with the silu epilogue, then down), the rows' sums in rank order
 * by moe_rows_sum_k after (parts & 2). The caller decided on this layout by coli_sycl_xmx_moe_ok,
 * a pure function of the call, so a later call with parts 2 alone finds the sort where it is. */
static int moe_xmx_launch(const coli_sycl_profile &prof, const CcExpert *tab, const float *x, const int *ids,
                          float *hid, MoeSort m, const CcMoe *p, sycl::queue *st) {
    size_t A = (size_t)p->S * p->K;
    unsigned ga = (unsigned)((A + 255) / 256);
    if (!cuda_ok(COLI_SYCL_TRY(st->memset(m.cnt, 0, (size_t)p->E * sizeof(int))), "chain moe sort")) return 0;
    {
        CcMoe pc = *p;
        st->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, ga) * sycl::range<3>(1, 1, 256), sycl::range<3>(1, 1, 256)),
                         [=](sycl::nd_item<3>) { moe_sort_count_k(tab, ids, m, pc); });
        st->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
                         [=](sycl::nd_item<3>) { moe_sort_scan_k(m, pc); });
        st->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, ga) * sycl::range<3>(1, 1, 256), sycl::range<3>(1, 1, 256)),
                         [=](sycl::nd_item<3>) { moe_sort_place_k(tab, ids, m, pc); });
    }
    coli_sycl_moe_sort s = {m.order, m.base, m.te, m.tr, m.ntile, m.T, m.TR};
    if (!coli_sycl_xmx_moe(prof, *st, tab, *p, s, x, hid, 0) || !coli_sycl_xmx_moe(prof, *st, tab, *p, s, x, hid, 1)) return 0;
    return cuda_ok(0, "chain moe rows launch");
}
/* the staging a decode step's host experts go to (moe_bd_launch): stage NULL, none */
extern "C" int coli_cuda_moe_stage(int device, void *stage, size_t slot_bytes, int slots) {
    DeviceContext *ctx = find_ctx(device);
    if (!ctx || (stage && (slots < 1 || slot_bytes % 16))) return 0;
    ctx->moe_stage = (char *)stage; ctx->moe_stage_slot = stage ? slot_bytes : 0;
    ctx->moe_stage_n = stage ? (slots < BD_STAGE_SLOTS ? slots : BD_STAGE_SLOTS) : 0;
    return 1;
}
extern "C" int coli_cuda_moe_async(int device, const CcExpert *tab, const float *x, const int *ids, const float *w,
                                   float *hid, float *y, const float *hy, const CcMoe *p, void *stream) {
    if (fault_injected()) return 0;
    if (!tab || !ids || !hid || !p || p->S < 1 || p->K < 1 || p->D < 32 || p->I < 32 || p->D % 32 || p->I % 32 || p->act) return 0;
    const int LS = p->lay_S > 0 ? p->lay_S : p->S;   /* the rows hid's layout is for: a part 1's, its rows' sums' (CcMoe.lay_S) */
    if (p->lay_S < 0 || p->lay_s0 < 0 || p->lay_s0 > LS - p->S || ((p->parts & 1) && (LS != p->S || p->lay_s0))) return 0;
    if (p->fmt >= 100) {   /* a GGUF's experts: every row on the GEMV kernels (a prompt chunk's went through the
                            * f16 tiles of hcg / hcm; the matrix engines' family takes the tier's rows alone) */
        if (!moe_bd_ok(p) || p->D > MOE_DIM_MAX || p->I > MOE_DIM_MAX) return 0;
        DeviceContext *bc = find_ctx(device);
        if (!select_ctx(bc)) return 0;
        sycl::queue * bst = (sycl::queue *)stream;
        if ((p->parts & 1) && (!x || !moe_bd_launch(bc, tab, x, ids, hid, p, bst))) return 0;
        if (p->parts & 2) {
            if (!w || !y || (p->h_off >= 0 && !hy)) return 0;
            dpct::dim3 g((unsigned)((p->D + 255) / 256), (unsigned)p->S);
                coli_sycl_fprof_ev(bst->submit([&](sycl::handler &cgh) {
                    CcMoe p_ct6 = *p;

                    cgh.parallel_for(
                        sycl::nd_range<3>(g * sycl::range<3>(1, 1, 256),
                                          sycl::range<3>(1, 1, 256)),
                        [=](sycl::nd_item<3> item_ct1) {
                            moe_sum_k(tab, ids, w, hid, hy, y, p_ct6);
                        });
                }));
            if (!cuda_ok(0, "chain moe sum launch")) return 0;
        }
        return 1;
    }
    if (p->fmt == 4 && (p->gs < 16 || p->gs % 16)) return 0;
    if (p->fmt != 1 && p->fmt != 2 && p->fmt != 4 && p->fmt != 8) return 0;
    if (p->D > MOE_DIM_MAX || p->I > MOE_DIM_MAX) return 0;   /* an input row in 48 KiB of shared memory, the fp8 table beside it */
    DeviceContext *ctx = find_ctx(device);
    if (!select_ctx(ctx)) return 0;
    sycl::queue * st = (sycl::queue *)stream;
    /* a prompt chunk's rows through the matrix engines (sorted by expert: coli_sycl_xmx_moe, the f16
     * joint_matrix tiles where hcg's and tcg's tensor-core tiles were), where the profile has the
     * shape and the family takes the format and the widths (whole steps of 64); else, and for a
     * decode step, the warp kernels per (row, slot) */
    const coli_sycl_profile &prof = coli_sycl_profile_for(device);
    int rows = LS > CC_MOE_FRAME_ROWS && p->E > 0 && p->E <= 4096 && p->D % XMX_BK == 0 && p->I % XMX_BK == 0 &&
               (p->fmt == 1 || p->fmt == 2 || (p->fmt == 4 && p->gs >= 16 && p->gs % 16 == 0)) &&
               (!x || ((uintptr_t)(x + p->x_off) & 15) == 0) && p->x_row % 4 == 0 && xmx_engines(prof) && coli_sycl_xmx_moe_ok(prof, *p);
    if (LS > CC_MOE_FRAME_ROWS && (p->parts & 1)) xmx_note("routed experts (prompt rows)", rows);
    if (p->planar && !(rows && p->fmt == 4 && p->gs == 64)) return 0;   /* planar entries: only the matrix engines' GEMM reads them */
    MoeSort m = moe_sort_of(hid, p);
    if (p->parts & 1) {
        if (!x) return 0;
        int ok = rows ? moe_xmx_launch(prof, tab, x, ids, hid, m, p, st)
               : moe_gv_ok(p) ? (p->fmt == 1 ? moe_gv_launch<1>(tab, x, ids, hid, p, st) : p->fmt == 2 ? moe_gv_launch<2>(tab, x, ids, hid, p, st)
                                                : moe_gv_launch<4>(tab, x, ids, hid, p, st))
               : p->fmt == 1 ? moe_launch<1>(tab, x, ids, hid, p, st) : p->fmt == 2 ? moe_launch<2>(tab, x, ids, hid, p, st)
               : p->fmt == 4 ? moe_launch<4>(tab, x, ids, hid, p, st) : moe_launch<8>(tab, x, ids, hid, p, st);
        if (!ok) return 0;
    }
    if (p->parts & 2) {
        if (!w || !y || (p->h_off >= 0 && !hy)) return 0;
        dpct::dim3 g((unsigned)((p->D + 255) / 256), (unsigned)p->S);
            if (rows) coli_sycl_fprof_ev(st->submit([&](sycl::handler &cgh) {
                CcMoe p_ct5 = *p;

                cgh.parallel_for(
                    sycl::nd_range<3>(g * sycl::range<3>(1, 1, 256),
                                      sycl::range<3>(1, 1, 256)),
                    [=](sycl::nd_item<3> item_ct1) {
                        moe_rows_sum_k(w, hid, hy, y, m, p_ct5);
                    });
            }));
            else coli_sycl_fprof_ev(st->submit([&](sycl::handler &cgh) {
                CcMoe p_ct6 = *p;

                cgh.parallel_for(
                    sycl::nd_range<3>(g * sycl::range<3>(1, 1, 256),
                                      sycl::range<3>(1, 1, 256)),
                    [=](sycl::nd_item<3> item_ct1) {
                        moe_sum_k(tab, ids, w, hid, hy, y, p_ct6);
                    });
            }));
        if (!cuda_ok(0, "chain moe sum launch")) return 0;
    }
    return 1;
}

/* A prompt's rows of a layer's routed experts in one call, more than a chunk's: hid holds the gate's
 * and up's outputs of every slot and then down's for a slice of cols outputs (cc_moe_hid_floats with
 * cols for D), each slice's sums going into y before the next slice's down. Those of a GGUF's types
 * hcm_moe's f16 tiles take, parts 3, no host rows (h_off < 0), cols a multiple of 128; the sums
 * are moe_rows_sum_k's, so each row of y has the bits a call of all D outputs gives it. 1 done, -1 not
 * taken (another type, few rows: the caller's chunks), 0 a failure. */
extern "C" int coli_cuda_moe_cols_async(int device, const CcExpert *tab, const float *x, const int *ids, const float *w,
                                        float *hid, float *y, const CcMoe *p, int cols, void *stream) {
    /* hcm_moe's slices (a GGUF's experts through f16 mma.sync, a layer's rows in one call) have no
     * form here: -1, not taken, the caller's chunks go through coli_cuda_moe_async */
    (void)device; (void)tab; (void)x; (void)ids; (void)w; (void)hid; (void)y; (void)p; (void)cols; (void)stream;
    return -1;
}

/* ==== resident-pipeline primitives (Inc.0, 2026-07-13) ====
 * Device-side building blocks so the residual stream can stay on the layer's
 * home device across a whole layer. Control flow stays on CPU; only the data
 * plane lives here. All entry points take DEVICE pointers (no transfers) —
 * the caller owns staging via the pipe buffer API below. */

static void pipe_rmsnorm_rows(float *y,const float *x,const float *w,
                                         int D,float eps,int xstride,int ystride){
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const float *xr = x + (size_t)item_ct1.get_group(2) * xstride;
        float *yr = y + (size_t)item_ct1.get_group(2) * ystride;
    auto &sh =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<double[256]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    double a = 0; for (int i = item_ct1.get_local_id(2); i < D;
                       i += item_ct1.get_local_range(2)) {
        double v = xr[i]; a += v * v;
    }
    sh[item_ct1.get_local_id(2)] = a;
        item_ct1.barrier(sycl::access::fence_space::local_space);
    for (int s = item_ct1.get_local_range(2) / 2; s > 0; s >>= 1) {
        if (item_ct1.get_local_id(2) < s) sh[item_ct1.get_local_id(2)] +=
        sh[item_ct1.get_local_id(2) + s];
        item_ct1.barrier(sycl::access::fence_space::local_space);
    }
    float r = sycl::rsqrt((float)(sh[0] / D) + eps);
    for (int i = item_ct1.get_local_id(2); i < D;
         i += item_ct1.get_local_range(2)) yr[i] = xr[i] * r * w[i];
}

/* RoPE interleaved, identical math to glm.c rope_interleave. One block per row;
 * row layout: v + row*stride + offset holds R floats. pos index = row/heads
 * (heads=1 for k_rot rows, heads=H for [S,H,qh] query rows). */
static void pipe_rope_rows(float *v,const int *pos,int pos_base,int stride,
                                      int offset,int R,int heads,float theta){
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    float *p = v + (size_t)item_ct1.get_group(2) * stride + offset;
    int half = R / 2,
        ps = pos ? pos[item_ct1.get_group(2) / heads]
                 : pos_base + (int)(item_ct1.get_group(2) / heads);
    auto &in = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[256]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    for (int j = item_ct1.get_local_id(2); j < R;
         j += item_ct1.get_local_range(2)) in[j] = p[j];
    item_ct1.barrier();
    for (int j = item_ct1.get_local_id(2); j < half;
         j += item_ct1.get_local_range(2)) {
        float inv = sycl::pow(theta, -2.0f * j / R);
        float ang = ps * inv, cs = sycl::cos(ang), sn = sycl::sin(ang);
        float a=in[2*j], b=in[2*j+1];
        p[j]=a*cs-b*sn; p[half+j]=b*cs+a*sn;
    }
}

static void pipe_add_n(float *x,const float *t,size_t n){
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    size_t i = (size_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
               item_ct1.get_local_id(2);
    if(i<n) x[i]+=t[i];
}

/* Fixed-order partial merge: block b adds partial row b into x row rows[b].
 * Target rows are unique by construction (CPU pre-sums per token), so no
 * atomics — the 9.20.7 lesson. */
static void pipe_rows_add(float *x,const float *partial,const int *rows,
                                     int D){
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    float *xr = x + (size_t)rows[item_ct1.get_group(2)] * D;
    const float *pr = partial + (size_t)item_ct1.get_group(2) * D;
    for (int i = item_ct1.get_local_id(2); i < D;
         i += item_ct1.get_local_range(2)) xr[i] += pr[i];
}

/* scratch persistente per (device,slot): cresce e resta — niente cudaMalloc/Free
 * per layer (78 x ~10 alloc/richiesta erano puro churn). */
extern "C" float *coli_cuda_pipe_scratch(int device,int slot,size_t bytes){
    DeviceContext *ctx=find_ctx(device);
    if(slot<0||slot>=27||!select_ctx(ctx)) return NULL;
    if(!reserve(&ctx->pipe_buf[slot],&ctx->pipe_cap[slot],bytes)) return NULL;
    return ctx->pipe_buf[slot];
}
extern "C" void *coli_cuda_pipe_alloc(int device, size_t bytes) try {
    DeviceContext *ctx=find_ctx(device); if(!select_ctx(ctx)) return NULL;
    void *p=NULL;
    if (!cuda_ok(COLI_SYCL_ALLOC(p, (void *)sycl::malloc_device(
                                      bytes, dq()), bytes),
                 "pipe alloc")) return NULL;
    return p;
}
catch (sycl::exception const &exc) { sycl_caught(exc, "coli_cuda_pipe_alloc"); return NULL; }
extern "C" void coli_cuda_pipe_free(int device,void *p){
    DeviceContext *ctx=find_ctx(device); if(!p||!select_ctx(ctx)) return;
    sycl::free(p, dq());
}
extern "C" int coli_cuda_pipe_upload(int device, void *dst, const void *src,
                                     size_t bytes) try {
    DeviceContext *ctx=find_ctx(device); if(!select_ctx(ctx)) return 0;
    return cuda_ok(
        COLI_SYCL_TRY(
            dq().memcpy(dst, src, bytes).wait()),
        "pipe upload");
}
catch (sycl::exception const &exc) { return sycl_caught(exc, "coli_cuda_pipe_upload"); }
extern "C" int coli_cuda_pipe_download(int device, const void *src, void *dst,
                                       size_t bytes) try {
    DeviceContext *ctx=find_ctx(device); if(!select_ctx(ctx)) return 0;
    return cuda_ok(
        COLI_SYCL_TRY(
            dq().memcpy(dst, src, bytes).wait()),
        "pipe download");
}
catch (sycl::exception const &exc) { return sycl_caught(exc, "coli_cuda_pipe_download"); }
extern "C" int coli_cuda_pipe_rmsnorm(int device,float *y_dev,const float *x_dev,
                                      const float *w_dev,int S,int D,float eps){
    if (fault_injected()) return 0;
    DeviceContext *ctx=find_ctx(device);
    if(S<1||D<1||!select_ctx(ctx)||!ctx->has_fp64) return 0;   /* the sums are double */
    {
        dq().parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, S) *
                                  sycl::range<3>(1, 1, 256),
                              sycl::range<3>(1, 1, 256)),
            [=](sycl::nd_item<3> item_ct1) {
                pipe_rmsnorm_rows(y_dev, x_dev, w_dev, D, eps, D, D);
            });
    }
    return cuda_ok(0, "pipe rmsnorm");
}
extern "C" int coli_cuda_pipe_rmsnorm_s(int device,float *y_dev,const float *x_dev,
                                        const float *w_dev,int S,int D,float eps,
                                        int xstride,int ystride){
    if (fault_injected()) return 0;
    DeviceContext *ctx=find_ctx(device);
    if(S<1||D<1||xstride<D||ystride<D||!select_ctx(ctx)||!ctx->has_fp64) return 0;
    {
        dq().parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, S) *
                                  sycl::range<3>(1, 1, 256),
                              sycl::range<3>(1, 1, 256)),
            [=](sycl::nd_item<3> item_ct1) {
                pipe_rmsnorm_rows(y_dev, x_dev, w_dev, D, eps, xstride,
                                  ystride);
            });
    }
    return cuda_ok(0, "pipe rmsnorm strided");
}
extern "C" int coli_cuda_pipe_rope(int device,float *v_dev,const int *pos_dev,
                                   int rows,int stride,int offset,int R,int heads,
                                   float theta){
    if (fault_injected()) return 0;
    DeviceContext *ctx=find_ctx(device);
    if(rows<1||R<2||R>256||heads<1||!select_ctx(ctx)) return 0;
    dq().parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, rows) *
                              sycl::range<3>(1, 1, 128),
                          sycl::range<3>(1, 1, 128)),
        [=](sycl::nd_item<3> item_ct1) {
            pipe_rope_rows(v_dev, pos_dev, 0, stride, offset, R, heads, theta);
        });
    return cuda_ok(0, "pipe rope");
}
extern "C" int coli_cuda_pipe_rope_base(int device,float *v_dev,int pos_base,int rows,
                                        int stride,int offset,int R,int heads,float theta){
    if (fault_injected()) return 0;
    DeviceContext *ctx=find_ctx(device);
    if(rows<1||R<2||R>256||heads<1||!select_ctx(ctx)) return 0;
    dq().parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, rows) *
                              sycl::range<3>(1, 1, 128),
                          sycl::range<3>(1, 1, 128)),
        [=](sycl::nd_item<3> item_ct1) {
            pipe_rope_rows(v_dev, NULL, pos_base, stride, offset, R, heads,
                           theta);
        });
    return cuda_ok(0, "pipe rope base");
}
/* ---- device router (#431 PR-A) -------------------------------------------
 * Router for one decode row, entirely on the layer's home device: logits GEMV
 * (E x D, tiny) + sigmoid, bias-augmented top-K selection, route-level TOPP
 * truncation, norm_topk and routed_scale — a float-faithful clone of moe()'s
 * plain routing path (colibri.c FASE A). Selection runs single-thread so the
 * argmax order, tie-breaking (strict >, lowest index wins) and weight math
 * match the CPU reference exactly; only the dot/expf rounding can differ,
 * which is the documented kernel-family divergence class (#100/#163).
 * Results are packed [idx[K] | w[K] | keff] in one scratch buffer and read
 * back with a single tiny D2H. */
void pipe_router_logits(const float *__restrict__ x,
                                   const float *__restrict__ W,
                                   const float *__restrict__ bias,
                                   int D, float *logit, float *choice){
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int e = item_ct1.get_group(2);
    const float *w = W + (size_t)e*D;
    float acc = 0.f;
    for (int i = item_ct1.get_local_id(2); i < D;
         i += item_ct1.get_local_range(2)) acc += x[i] * w[i];
    auto &sh = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[128]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    sh[item_ct1.get_local_id(2)] = acc;
        item_ct1.barrier(sycl::access::fence_space::local_space);
    for (int s = item_ct1.get_local_range(2) >> 1; s > 0; s >>= 1) {
        if (item_ct1.get_local_id(2) < s) sh[item_ct1.get_local_id(2)] +=
            sh[item_ct1.get_local_id(2) + s];
        item_ct1.barrier(sycl::access::fence_space::local_space);
    }
    if (!item_ct1.get_local_id(2)) {
        float lg = 1.f / (1.f + sycl::native::exp(-sh[0]));
        logit[e]=lg; choice[e]=lg+bias[e];
    }
}
void pipe_router_select(const float *__restrict__ logit,
                                   const float *__restrict__ choice, int E,
                                   int Ksel, float topp, int norm_topk,
                                   float routed_scale, char *out){
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    if (item_ct1.get_local_id(2) || item_ct1.get_group(2)) return;
    int   *idx = (int*)out;
    float *w   = (float*)(out + Ksel*sizeof(int));
    int   *keff= (int*)(out + Ksel*(sizeof(int)+sizeof(float)));
    for(int kk=0;kk<Ksel;kk++){
        int best=-1; float bv=-1e30f;
        for(int e=0;e<E;e++){ int tk=0; for(int j=0;j<kk;j++) if(idx[j]==e){tk=1;break;}
            if(!tk && choice[e]>bv){bv=choice[e];best=e;} }
        idx[kk]=best; w[kk]=logit[best];
    }
    int Ke=Ksel;
    if(topp>0.f && topp<1.f){
        for(int a=1;a<Ksel;a++){ int ii=idx[a]; float ww=w[a]; int b=a-1;
            while(b>=0 && w[b]<ww){ w[b+1]=w[b]; idx[b+1]=idx[b]; b--; } w[b+1]=ww; idx[b+1]=ii; }
        float tot=1e-20f; for(int kk=0;kk<Ksel;kk++) tot+=w[kk];
        float cum=0.f; for(int kk=0;kk<Ksel;kk++){ cum+=w[kk]; if(cum>=topp*tot){ Ke=kk+1; break; } }
    }
    if(norm_topk){ float sm=0.f; for(int kk=0;kk<Ke;kk++) sm+=w[kk]; sm+=1e-20f;
                   for(int kk=0;kk<Ke;kk++) w[kk]/=sm; }
    for(int kk=0;kk<Ke;kk++) w[kk]*=routed_scale;
    *keff=Ke;
}
extern "C" int coli_cuda_pipe_router(int device, const float *x_dev,
                                     const void *rw_dev, const void *rb_dev,
                                     int D, int E, int Ksel, float topp,
                                     int norm_topk, float routed_scale,
                                     int *idx_host, float *w_host,
                                     int *keff_host) try {
    DeviceContext *ctx=find_ctx(device);
    if(!x_dev||!rw_dev||!rb_dev||D<1||E<1||E>4096||Ksel<1||Ksel>64||!select_ctx(ctx)) return 0;
    size_t pack=(size_t)Ksel*(sizeof(int)+sizeof(float))+sizeof(int);
    float *logit=coli_cuda_pipe_scratch(device,22,(size_t)E*sizeof(float));
    float *chc  =coli_cuda_pipe_scratch(device,23,(size_t)E*sizeof(float));
    char  *out  =(char*)coli_cuda_pipe_scratch(device,24,pack);
    if(!logit||!chc||!out) return 0;
    dq().parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, E) * sycl::range<3>(1, 1, 128),
                          sycl::range<3>(1, 1, 128)),
        [=](sycl::nd_item<3> item_ct1) {
            pipe_router_logits(x_dev, (const float *)rw_dev,
                               (const float *)rb_dev, D, logit, chc);
        });
    dq().parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
        [=](sycl::nd_item<3> item_ct1) {
            pipe_router_select(logit, chc, E, Ksel, topp, norm_topk,
                               routed_scale, out);
        });
    if (!cuda_ok(0, "pipe router launch")) return 0;
    char buf[64*(sizeof(int)+sizeof(float))+sizeof(int)];
    if (!cuda_ok(COLI_SYCL_TRY(
                     dq().memcpy(buf, out, pack).wait()),
                 "pipe router readback")) return 0;
    memcpy(idx_host,buf,(size_t)Ksel*sizeof(int));
    memcpy(w_host,buf+Ksel*sizeof(int),(size_t)Ksel*sizeof(float));
    memcpy(keff_host,buf+Ksel*(sizeof(int)+sizeof(float)),sizeof(int));
    return 1;
}
catch (sycl::exception const &exc) { return sycl_caught(exc, "coli_cuda_pipe_router"); }
/* ---- resident expert-group accumulation (#431 PR-C0) ----------------------
 * Decode-time (S=1) expert groups without the host round-trip: the input row
 * is P2P'd from the layer's home device, the group runs through the grouped-W4
 * kernels on its own stream, the down-projection outputs are weighted and
 * reduced ON DEVICE (fixed expert order), and the device's partial sum is
 * peer-pushed into a per-issue slot on the home device. take() makes the home
 * legacy stream wait on every issue event and reduces the slots in issue order
 * — deterministic, no atomics, no host bytes. The CPU tier overlaps with all
 * of it exactly as before. */
static void bcast_row(float *dst,const float *src,int count,int D){
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    for (int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                 item_ct1.get_local_id(2);
         i < D;
         i += item_ct1.get_group_range(2) * item_ct1.get_local_range(2)) {
        float v=src[i];
        for(int c=0;c<count;c++) dst[(size_t)c*D+i]=v;
    }
}
static void weighted_sum_rows(float *out,const float *y,const float *w,
                                         int count,int D){
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    for (int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                 item_ct1.get_local_id(2);
         i < D;
         i += item_ct1.get_group_range(2) * item_ct1.get_local_range(2)) {
        float acc=0.f;
        for(int c=0;c<count;c++) acc+=w[c]*y[(size_t)c*D+i];   /* fixed order */
        out[i]=acc;
    }
}
static void sum_slots(float *dst,const float *slots,int n,int D){
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    for (int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                 item_ct1.get_local_id(2);
         i < D;
         i += item_ct1.get_group_range(2) * item_ct1.get_local_range(2)) {
        float acc=0.f;
        for(int s=0;s<n;s++) acc+=slots[(size_t)s*D+i];        /* issue order */
        dst[i]=acc;
    }
}
extern "C" int coli_cuda_expert_group_resident_issue(
    ColiCudaTensor *const *gates, ColiCudaTensor *const *ups,
    ColiCudaTensor *const *downs, const float *weights, int count,
    int home_device, const float *x_src_dev, float *partial_slot_dev) try {
    if(!gates||!ups||!downs||!weights||count<1||count>64||!x_src_dev||!partial_slot_dev) return 0;
    ColiCudaTensor *first=gates[0]; if(!first) return 0;
    for(int i=0;i<count;i++)   /* the block formats: the chain's GEMVs only */
        if((gates[i]&&gates[i]->fmt>=100)||(ups[i]&&ups[i]->fmt>=100)||(downs[i]&&downs[i]->fmt>=100)) return 0;
    int device=first->device,D=first->I,I=first->O;
    GroupDesc host[64];
    int total=0,all_s4=1;
    for(int c=0;c<count;c++){
        ColiCudaTensor *g=gates[c],*u=ups[c],*d=downs[c];
        if(!g||!u||!d||g->device!=device||u->device!=device||d->device!=device||
           g->I!=D||u->I!=D||g->O!=I||u->O!=I||d->I!=I||d->O!=D) return 0;
        host[c]={g->weights,u->weights,d->weights,g->scales,u->scales,d->scales,
                 g->fmt,u->fmt,d->fmt,1,total,
                 g->gs,u->gs,d->gs};
        all_s4&=g->fmt==2&&u->fmt==2&&d->fmt==2;
        total++;
    }
    if(!all_s4) return 0;                       /* resident path: per-row int4 only */
    DeviceContext *ctx=find_ctx(device); if(!select_ctx(ctx)) return 0;
    if(!prepare_group_weights(ctx,gates,ups,downs,count,host)) return 0;
    if(!ctx->ev_done_ok){
        if (!cuda_ok(COLI_SYCL_TRY(ctx->ev_done = new sycl::event()),
                     "resident group event")) return 0;
        ctx->ev_done_ok=1;
    }
    /* size for the 64-expert cap, not for `count`: reserve() reallocs on growth,
     * and a realloc here could free a buffer the PREVIOUS layer's still-queued
     * async work on this stream reads. Fixed caps make re-issue realloc-free. */
    size_t xb=(size_t)64*D*sizeof(float), ib=(size_t)64*I*sizeof(float);
    if(!reserve(&ctx->x,&ctx->x_cap,xb)||!reserve(&ctx->y,&ctx->y_cap,xb)||
       !reserve(&ctx->gate,&ctx->gate_cap,ib)||!reserve(&ctx->up,&ctx->up_cap,ib)||
       !reserve(&ctx->ac,&ctx->ac_cap,(size_t)(D+64)*sizeof(float))||
       !reserve_bytes(&ctx->group_desc,&ctx->group_desc_cap,(size_t)64*sizeof(GroupDesc)))
        return 0;
    float *w_dev=ctx->ac+D, *partial_local=ctx->ac;
    if (!cuda_ok(COLI_SYCL_TRY(ctx->stream->memcpy(
                     ctx->group_desc, host, (size_t)count * sizeof(GroupDesc)).wait()),
                 "resident group desc") ||
        !cuda_ok(COLI_SYCL_TRY(ctx->stream->memcpy(
                     w_dev, weights, (size_t)count * sizeof(float)).wait()),
                 "resident group weights"))
        return 0;
    /* input row: P2P from the home device. The caller guarantees x_src_dev is
     * materialized (the pre-moe nrm download already synced the home stream). */
    if (!cuda_ok(peer_copy(ctx->stream, device, ctx->x, home_device, x_src_dev, (size_t)D * sizeof(float), 0),
                 "resident group x p2p"))
        return 0;
    ctx->stream->submit([&](sycl::handler &cgh) {
        auto ctx_x_ct0 = ctx->x;
        const float *ctx_x_ct1 = ctx->x;

        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, 64) *
                                               sycl::range<3>(1, 1, 256),
                                           sycl::range<3>(1, 1, 256)),
                         [=](sycl::nd_item<3> item_ct1) {
                             bcast_row(ctx_x_ct0, ctx_x_ct1, count, D);
                         });
    }); /* row 0 -> rows 1..count-1 (in-place safe: row 0 rewritten with itself)
         */
    GroupDesc *dev=(GroupDesc*)ctx->group_desc;
    dpct::dim3 hg((unsigned)I, 1, (unsigned)count),
        og((unsigned)D, 1, (unsigned)count);
    ctx->stream->submit([&](sycl::handler &cgh) {
        auto ctx_gate_ct0 = ctx->gate;
        auto ctx_up_ct1 = ctx->up;
        const float *ctx_x_ct2 = ctx->x;

        cgh.parallel_for(sycl::nd_range<3>(hg * sycl::range<3>(1, 1, 256),
                                           sycl::range<3>(1, 1, 256)),
                         [=](sycl::nd_item<3> item_ct1) {
                             grouped_hidden_w4_dual(ctx_gate_ct0, ctx_up_ct1,
                                                    ctx_x_ct2, dev, I, D);
                         });
    }); /* silu fused in epilogue */
    ctx->stream->submit([&](sycl::handler &cgh) {
        auto ctx_y_ct0 = ctx->y;
        const float *ctx_gate_ct1 = ctx->gate;

        cgh.parallel_for(sycl::nd_range<3>(og * sycl::range<3>(1, 1, 256),
                                           sycl::range<3>(1, 1, 256)),
                         [=](sycl::nd_item<3> item_ct1) {
                             grouped_down_w4(ctx_y_ct0, ctx_gate_ct1, dev, D,
                                             I);
                         });
    });
    ctx->stream->submit([&](sycl::handler &cgh) {
        const float *ctx_y_ct1 = ctx->y;

        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, 48) *
                                               sycl::range<3>(1, 1, 256),
                                           sycl::range<3>(1, 1, 256)),
                         [=](sycl::nd_item<3> item_ct1) {
                             weighted_sum_rows(partial_local, ctx_y_ct1, w_dev,
                                               count, D);
                         });
    });
    if (!cuda_ok(peer_copy(ctx->stream, home_device, partial_slot_dev, device, partial_local, (size_t)D * sizeof(float), 0),
                 "resident partial p2p"))
        return 0;
    if (!cuda_ok(
            COLI_SYCL_TRY(sync_barrier(ctx->ev_done, ctx->stream)),
            "resident event record")) return 0;
    return cuda_ok(0, "resident group launch");
}
catch (sycl::exception const &exc) { return sycl_caught(exc, "coli_cuda_expert_group_resident_issue"); }
extern "C" int coli_cuda_expert_group_resident_take(int home_device,
                                                    const int *devices,
                                                    int n_issued,
                                                    float *slots_dev,
                                                    float *acc_dev, int D) try {
    if(n_issued<1||!slots_dev||!acc_dev||D<1) return 0;
    DeviceContext *home=find_ctx(home_device); if(!select_ctx(home)) return 0;
    for(int i=0;i<n_issued;i++){
        DeviceContext *src=find_ctx(devices[i]);
        if(!src||!src->ev_done_ok) return 0;
        if (!cuda_ok(COLI_SYCL_TRY(
                         dq().ext_oneapi_submit_barrier(
                             {*src->ev_done})),
                     "resident take wait")) return 0;
    }
    dq().parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, 48) * sycl::range<3>(1, 1, 256),
                          sycl::range<3>(1, 1, 256)),
        [=](sycl::nd_item<3> item_ct1) {
            sum_slots(acc_dev, slots_dev, n_issued, D);
        }); /* legacy stream: ordered with pipe_* */
    return cuda_ok(0, "resident take reduce");
}
catch (sycl::exception const &exc) { return sycl_caught(exc, "coli_cuda_expert_group_resident_take"); }
extern "C" int coli_cuda_pipe_copy2d(int device, float *dst, int dpitch,
                                     const float *src, int spitch, int width,
                                     int height) try {
    DeviceContext *ctx=find_ctx(device); if(!select_ctx(ctx)) return 0;
    return cuda_ok(COLI_SYCL_TRY(dq().ext_oneapi_memcpy2d(dst, (size_t)dpitch * 4, src, (size_t)spitch * 4,
                                                           (size_t)width * 4, height).wait()),
                   "pipe copy2d");
}
catch (sycl::exception const &exc) { return sycl_caught(exc, "coli_cuda_pipe_copy2d"); }
/* attention batch + fused o_proj with DEVICE-resident q/latent/rope: the whole
 * upstream projection chain stayed on this device, so nothing is uploaded here.
 * Only the final [S,O] projection is downloaded to host. */
extern "C" int coli_cuda_attention_project_batch_dev(
    ColiCudaTensor *w, ColiCudaTensor *proj, float *out, const float *q_dev,
    const float *latent_dev, const float *rope_dev, int S, int H, int Q, int R,
    int V, int K, int T, float scale) try {
    if (fault_injected()) return 0;
    if(!absorb_fmt_ok(w)||!proj||!out||!q_dev||!latent_dev||!rope_dev||S<1||H<1||Q<1||R<1||V<1||
       K<1||K>512||T<S||T>8192||w->I!=K||w->O!=H*(Q+V)||
       proj->device!=w->device||proj->I!=H*V)return 0;
    DeviceContext *dc=find_ctx(w->device);if(!select_ctx(dc))return 0;
    size_t cb=(size_t)S*H*V*sizeof(float);
    if(!reserve(&dc->ac,&dc->ac_cap,cb))return 0;
    size_t shared = (size_t)(2 * K + T + 256) * sizeof(float);
    {

        dc->stream->submit([&](sycl::handler &cgh) {
            const float *c_e4m3_ptr_ct1 = cur_ctx()->e4m3;

            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range<1>(shared), cgh);

            auto dc_ac_ct0 = dc->ac;
            const void *w_weights_ct4 = w->weights;
            const float *w_scales_ct5 = w->scales;
            auto w_fmt_ct6 = w->fmt;
            auto w_gs_ct15 = w->gs;
            auto w_ng_ct16 = w->ng;

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, S, H) *
                                      sycl::range<3>(1, 1, 256),
                                  sycl::range<3>(1, 1, 256)),
                [=](sycl::nd_item<3> item_ct1) {
                    attention_absorb_batch_kernel(
                        dc_ac_ct0, q_dev, latent_dev, rope_dev, w_weights_ct4,
                        w_scales_ct5, w_fmt_ct6, S, H, Q, R, V, K, T, scale,
                        w_gs_ct15, w_ng_ct16,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get(),
                        c_e4m3_ptr_ct1);
                });
        });
    }
    if (!cuda_ok(0, "pipe attention launch")) return 0;
    size_t ob=(size_t)S*proj->O*sizeof(float);
    if(!reserve(&dc->y,&dc->y_cap,ob))return 0;
    {

        dc->stream->submit([&](sycl::handler &cgh) {
            const float *c_e4m3_ptr_ct1 = cur_ctx()->e4m3;

            const uint8_t *c_e8_grid_acc_ct1 = cur_ctx()->e8_grid;

            auto dc_y_ct0 = dc->y;
            const float *dc_ac_ct1 = dc->ac;
            const void *proj_weights_ct2 = proj->weights;
            const float *proj_scales_ct3 = proj->scales;
            auto proj_fmt_ct4 = proj->fmt;
            auto proj_I_ct6 = proj->I;
            auto proj_O_ct7 = proj->O;
            auto row_bytes_proj_fmt_proj_I_ct8 = row_bytes(proj->fmt, proj->I);
            auto proj_gs_ct9 = proj->gs;
            auto proj_ng_ct10 = proj->ng;

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, S, proj->O) *
                                      sycl::range<3>(1, 1, 256),
                                  sycl::range<3>(1, 1, 256)),
                [=](sycl::nd_item<3> item_ct1) {
                    quant_matmul(dc_y_ct0, dc_ac_ct1, proj_weights_ct2,
                                 proj_scales_ct3, proj_fmt_ct4, S, proj_I_ct6,
                                 proj_O_ct7, row_bytes_proj_fmt_proj_I_ct8,
                                 proj_gs_ct9, proj_ng_ct10, c_e8_grid_acc_ct1,
                                 c_e4m3_ptr_ct1);
                });
        });
    }
    if (!cuda_ok(0, "pipe o_proj launch")) return 0;
    if (!cuda_ok(COLI_SYCL_TRY(dc->stream->memcpy(out, dc->y, ob)),
                 "pipe attention download") ||
        !cuda_ok(COLI_SYCL_TRY(dc->stream->wait()),
                 "pipe attention sync")) return 0;
    return 1;
}
catch (sycl::exception const &exc) { return sycl_caught(exc, "coli_cuda_attention_project_batch_dev"); }
extern "C" int coli_cuda_pipe_silu_mul(int device,float *gate_dev,const float *up_dev,
                                       size_t n){
    if (fault_injected()) return 0;
    DeviceContext *ctx=find_ctx(device); if(!n||!select_ctx(ctx)) return 0;
    if(ctx->has_fp64 && f8_warp_mode() && cpu_silu_on()) {
        cpu_silu_note(2);
        dq().parallel_for(sycl::nd_range<3>(sycl::range<3>(1,1,((n+255)/256)*256),
                                          sycl::range<3>(1,1,256)),
                         [=](sycl::nd_item<3>){silu_cpu_mul(gate_dev,up_dev,n);});
    } else dq().parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, (unsigned)((n + 255) / 256)) *
                              sycl::range<3>(1, 1, 256),
                          sycl::range<3>(1, 1, 256)),
        [=](sycl::nd_item<3> item_ct1) {
            silu_mul(gate_dev, up_dev, n);
        });
    return cuda_ok(0, "pipe silu mul");
}
extern "C" int coli_cuda_pipe_add(int device,float *x_dev,const float *t_dev,size_t n){
    if (fault_injected()) return 0;
    DeviceContext *ctx=find_ctx(device); if(!n||!select_ctx(ctx)) return 0;
    dq().parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, (unsigned)((n + 255) / 256)) *
                              sycl::range<3>(1, 1, 256),
                          sycl::range<3>(1, 1, 256)),
        [=](sycl::nd_item<3> item_ct1) {
            pipe_add_n(x_dev, t_dev, n);
        });
    return cuda_ok(0, "pipe add");
}
extern "C" int coli_cuda_pipe_rows_add(int device,float *x_dev,const float *partial_dev,
                                       const int *rows_dev,int nrows,int D){
    if (fault_injected()) return 0;
    DeviceContext *ctx=find_ctx(device); if(nrows<1||D<1||!select_ctx(ctx)) return 0;
    dq().parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nrows) *
                              sycl::range<3>(1, 1, 256),
                          sycl::range<3>(1, 1, 256)),
        [=](sycl::nd_item<3> item_ct1) {
            pipe_rows_add(x_dev, partial_dev, rows_dev, D);
        });
    return cuda_ok(0, "pipe rows add");
}
/* GEMM with device-resident activations: same quant_matmul kernel as
 * coli_cuda_matmul, zero host transfers. */
extern "C" int coli_cuda_pipe_gemm(ColiCudaTensor *t,float *y_dev,const float *x_dev,
                                   int S) try {
    if (fault_injected()) return 0;
    if(!t||!y_dev||!x_dev||S<1) return 0;
    DeviceContext *ctx=find_ctx(t->device); if(!select_ctx(ctx)) return 0;
    if (t->fmt >= 100) return bd_tensor_gemv(ctx, t, y_dev, x_dev, S, &dq());
    /* Preserve this entry point's original FP8 reduction (not the warp form). */
    if (!quant_matmul_launch(y_dev, x_dev, t->weights, t->scales, t->fmt, S,
                             t->I, t->O, row_bytes(t->fmt, t->I), t->gs, t->ng,
                             &dq(), false)) return 0;
    return cuda_ok(0, "pipe gemm");
}
catch (sycl::exception const &exc) { return sycl_caught(exc, "coli_cuda_pipe_gemm"); }
/* copia diretta scheda->scheda (P2P se disponibile, altrimenti staging driver) */
extern "C" int coli_cuda_pipe_peer_copy(int dst_dev, float *dst, int src_dev,
                                        const float *src, size_t bytes) try {
    if(!dst||!src) return 0;
    if(dst_dev==src_dev){ DeviceContext *c=find_ctx(dst_dev); if(!select_ctx(c)) return 0;
        return cuda_ok(COLI_SYCL_TRY(dq().memcpy(dst, src, bytes)),"pipe intra copy"); }
    DeviceContext *c=find_ctx(dst_dev); if(!select_ctx(c)) return 0;
    return cuda_ok(peer_copy(&dq(), dst_dev, dst, src_dev, src, bytes, 1), "pipe peer copy");
}
catch (sycl::exception const &exc) { return sycl_caught(exc, "coli_cuda_pipe_peer_copy"); }
/* come attention_project_batch_dev ma l'uscita di o_proj RESTA sul device (out_dev). */
extern "C" int coli_cuda_attention_project_batch_dev_out(
    ColiCudaTensor *w, ColiCudaTensor *proj, float *out_dev, const float *q_dev,
    const float *latent_dev, const float *rope_dev, int S, int H, int Q, int R,
    int V, int K, int T, float scale) try {
    if (fault_injected()) return 0;
    if(!absorb_fmt_ok(w)||!proj||!out_dev||!q_dev||!latent_dev||!rope_dev||S<1||H<1||Q<1||R<1||V<1||
       K<1||K>512||T<S||T>8192||w->I!=K||w->O!=H*(Q+V)||
       proj->device!=w->device||proj->I!=H*V)return 0;
    DeviceContext *dc=find_ctx(w->device);if(!select_ctx(dc))return 0;
    size_t cb=(size_t)S*H*V*sizeof(float);
    if(!reserve(&dc->ac,&dc->ac_cap,cb))return 0;
    size_t shared = (size_t)(2 * K + T + 256) * sizeof(float);
    {

        dc->stream->submit([&](sycl::handler &cgh) {
            const float *c_e4m3_ptr_ct1 = cur_ctx()->e4m3;

            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range<1>(shared), cgh);

            auto dc_ac_ct0 = dc->ac;
            const void *w_weights_ct4 = w->weights;
            const float *w_scales_ct5 = w->scales;
            auto w_fmt_ct6 = w->fmt;
            auto w_gs_ct15 = w->gs;
            auto w_ng_ct16 = w->ng;

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, S, H) *
                                      sycl::range<3>(1, 1, 256),
                                  sycl::range<3>(1, 1, 256)),
                [=](sycl::nd_item<3> item_ct1) {
                    attention_absorb_batch_kernel(
                        dc_ac_ct0, q_dev, latent_dev, rope_dev, w_weights_ct4,
                        w_scales_ct5, w_fmt_ct6, S, H, Q, R, V, K, T, scale,
                        w_gs_ct15, w_ng_ct16,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get(),
                        c_e4m3_ptr_ct1);
                });
        });
    }
    if (!cuda_ok(0, "pipe attention launch (dev out)")) return 0;
    {

        dc->stream->submit([&](sycl::handler &cgh) {
            const float *c_e4m3_ptr_ct1 = cur_ctx()->e4m3;

            const uint8_t *c_e8_grid_acc_ct1 = cur_ctx()->e8_grid;

            const float *dc_ac_ct1 = dc->ac;
            const void *proj_weights_ct2 = proj->weights;
            const float *proj_scales_ct3 = proj->scales;
            auto proj_fmt_ct4 = proj->fmt;
            auto proj_I_ct6 = proj->I;
            auto proj_O_ct7 = proj->O;
            auto row_bytes_proj_fmt_proj_I_ct8 = row_bytes(proj->fmt, proj->I);
            auto proj_gs_ct9 = proj->gs;
            auto proj_ng_ct10 = proj->ng;

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, S, proj->O) *
                                      sycl::range<3>(1, 1, 256),
                                  sycl::range<3>(1, 1, 256)),
                [=](sycl::nd_item<3> item_ct1) {
                    quant_matmul(out_dev, dc_ac_ct1, proj_weights_ct2,
                                 proj_scales_ct3, proj_fmt_ct4, S, proj_I_ct6,
                                 proj_O_ct7, row_bytes_proj_fmt_proj_I_ct8,
                                 proj_gs_ct9, proj_ng_ct10, c_e8_grid_acc_ct1,
                                 c_e4m3_ptr_ct1);
                });
        });
    }
    if (!cuda_ok(0, "pipe o_proj launch (dev out)")) return 0;
    return cuda_ok(COLI_SYCL_TRY(dc->stream->wait()),
                   "pipe attention sync (dev out)");
}
catch (sycl::exception const &exc) { return sycl_caught(exc, "coli_cuda_attention_project_batch_dev_out"); }
/* absorb batch con TUTTO su device (q/latent/rope gia' residenti sulla scheda
 * dello shard, ctx resta sul device): il cuore della attention head-shardata
 * dentro il pipeline. Nessun trasferimento host. */
extern "C" int coli_cuda_attention_absorb_batch_dev(
    ColiCudaTensor *w, float *ctx_dev, const float *q_dev,
    const float *latent_dev, const float *rope_dev, int S, int H, int Q, int R,
    int V, int K, int T, float scale) try {
    if (fault_injected()) return 0;
    if(!absorb_fmt_ok(w)||!ctx_dev||!q_dev||!latent_dev||!rope_dev||S<1||H<1||Q<1||R<1||V<1||
       K<1||K>512||T<S||T>8192||w->I!=K||w->O!=H*(Q+V))return 0;
    DeviceContext *dc=find_ctx(w->device);if(!select_ctx(dc))return 0;
    size_t shared = (size_t)(2 * K + T + 256) * sizeof(float);
    {

        dc->stream->submit([&](sycl::handler &cgh) {
            const float *c_e4m3_ptr_ct1 = cur_ctx()->e4m3;

            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range<1>(shared), cgh);

            const void *w_weights_ct4 = w->weights;
            const float *w_scales_ct5 = w->scales;
            auto w_fmt_ct6 = w->fmt;
            auto w_gs_ct15 = w->gs;
            auto w_ng_ct16 = w->ng;

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, S, H) *
                                      sycl::range<3>(1, 1, 256),
                                  sycl::range<3>(1, 1, 256)),
                [=](sycl::nd_item<3> item_ct1) {
                    attention_absorb_batch_kernel(
                        ctx_dev, q_dev, latent_dev, rope_dev, w_weights_ct4,
                        w_scales_ct5, w_fmt_ct6, S, H, Q, R, V, K, T, scale,
                        w_gs_ct15, w_ng_ct16,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get(),
                        c_e4m3_ptr_ct1);
                });
        });
    }
    if (!cuda_ok(0, "pipe shard attention launch")) return 0;
    return cuda_ok(COLI_SYCL_TRY(dc->stream->wait()),
                   "pipe shard attention sync");
}
catch (sycl::exception const &exc) { return sycl_caught(exc, "coli_cuda_attention_absorb_batch_dev"); }
/* absorb per il DECODE con KV gia' residente: carica solo q (poche KB),
 * latent/rope arrivano dall'ombra device. ctx torna a host (S piccolo). */
extern "C" int coli_cuda_attention_absorb_kvdev(ColiCudaTensor *w, float *ctx,
                                                const float *q,
                                                const float *latent_dev,
                                                const float *rope_dev, int H,
                                                int Q, int R, int V, int K,
                                                int T, float scale) try {
    if (fault_injected()) return 0;
    if(!absorb_fmt_ok(w)||!ctx||!q||!latent_dev||!rope_dev||H<1||Q<1||R<1||V<1||K<1||K>512||T<1||T>8192||
       w->I!=K||w->O!=H*(Q+V))return 0;
    DeviceContext *dc=find_ctx(w->device);if(!select_ctx(dc))return 0;
    size_t qb=(size_t)H*(Q+R)*sizeof(float),cb=(size_t)H*V*sizeof(float);
    if(!reserve(&dc->aq,&dc->aq_cap,qb)||!reserve(&dc->ac,&dc->ac_cap,cb))return 0;
    if (!cuda_ok(COLI_SYCL_TRY(dc->stream->memcpy(dc->aq, q, qb)),
                 "kvdev q upload")) return 0;
    size_t shared = (size_t)(2 * K + T + 256) * sizeof(float);
    {

        dc->stream->submit([&](sycl::handler &cgh) {
            const float *c_e4m3_ptr_ct1 = cur_ctx()->e4m3;

            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range<1>(shared), cgh);

            auto dc_ac_ct0 = dc->ac;
            const float *dc_aq_ct1 = dc->aq;
            const void *w_weights_ct4 = w->weights;
            const float *w_scales_ct5 = w->scales;
            auto w_fmt_ct6 = w->fmt;
            auto w_gs_ct15 = w->gs;
            auto w_ng_ct16 = w->ng;

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, H) *
                                      sycl::range<3>(1, 1, 256),
                                  sycl::range<3>(1, 1, 256)),
                [=](sycl::nd_item<3> item_ct1) {
                    attention_absorb_batch_kernel(
                        dc_ac_ct0, dc_aq_ct1, latent_dev, rope_dev,
                        w_weights_ct4, w_scales_ct5, w_fmt_ct6, 1, H, Q, R, V,
                        K, T, scale, w_gs_ct15, w_ng_ct16,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get(),
                        c_e4m3_ptr_ct1);
                });
        });
    }
    if (!cuda_ok(0, "kvdev absorb launch") ||
        !cuda_ok(COLI_SYCL_TRY(dc->stream->memcpy(ctx, dc->ac, cb)),
                 "kvdev ctx download") ||
        !cuda_ok(COLI_SYCL_TRY(dc->stream->wait()),
                 "kvdev absorb sync")) return 0;
    return 1;
}
catch (sycl::exception const &exc) { return sycl_caught(exc, "coli_cuda_attention_absorb_kvdev"); }
extern "C" int coli_cuda_pipe_sync(int device) try {
    DeviceContext *ctx=find_ctx(device); if(!select_ctx(ctx)) return 0;
    return cuda_ok(COLI_SYCL_TRY(dq().wait_and_throw(); coli_sycl_copy_queue(ctx->device).wait_and_throw()),
                   "pipe sync");
}
catch (sycl::exception const &exc) { return sycl_caught(exc, "coli_cuda_pipe_sync"); }
