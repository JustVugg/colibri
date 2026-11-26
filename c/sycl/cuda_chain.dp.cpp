/* cuda_chain.dp.cpp -- the dense chain on an Intel GPU: cuda_chain.h's table, in SYCL.
 *
 * cuda_chain.cu's migration (c/sycl/tools/dpct-migrate.sh, then by hand: MIGRATION.md). The
 * kernels are the .cu's, each the arithmetic of the shader of the same name in
 * shaders/chain_*.comp ported line for line (the sums in the same order, the same sentinels),
 * so the two chains give the same bits for the same bits in and tests/test_vk_chain.c's
 * references hold here too. The warp-32-style ones (a warp a row, lane = id & 31, width-32
 * shuffles, eight warps a block of 256) are templates over SG, a sub-group's lanes, built at
 * 16 and at 32 and launched at the device's (coli_sycl.h's profile: 32 on Ponte Vecchio, 16 on
 * Xe2; COLI_SYCL_SG overrides); their lane arithmetic is SG's, a block's warps block / SG. The
 * matmul is the backend's own GEMV / GEMM kernel (coli_cuda_tensor_gemm_async) on the chain's
 * queue.
 *
 * One context per device: the device's in-order queue (coli_sycl_queue: the stream, the frames'
 * order), a second for the staging copies (coli_sycl_copy_queue, cuda_chain.cu's c->cs), an
 * event per frame, the lost flag. UP and DOWN buffers are host USM (sycl::malloc_host): the
 * kernels read or write them over the bus (plain loads and stores only: a device atomic on
 * host memory is an access violation on these cards, so the hybrid steps' flag words are
 * written and polled by host tasks on the queue, cc_signal / cc_wait_host), the host through
 * cc_ptr, nothing copied, and the device pointer is the host pointer. A sycl::exception in any call marks the context lost, and
 * so does an asynchronous error the runtime delivers to the queue's handler instead of throwing
 * (coli_sycl_lost, read at every call and after every try): every later call returns 0 and the
 * engine rebuilds on the CPU, as with the Vulkan chain.
 * What the CUDA file had and this one has not: the tensor-core attention (cc_attn_fk and
 * cc_attn_f2, WMMA and mma.sync; the matrix engines' twin is coli_sycl.h's coli_sycl_xmx_attn,
 * asked where the profile has it), programmatic dependent launch, the batched host copies,
 * the lazy module load (AOT: nothing to load). */
#include <sycl/sycl.hpp>
#include "backend_cuda.h"
#define COLI_CUDA_CHAIN_NO_WRAPPERS
#include "cuda_chain.h"
#include "coli_sycl.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <exception>
#include <new>
#include <sched.h>
#include <time.h>
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

/* Two of backend_cuda.dp.cpp's, for this file alone (the same object set, never the host's: not
 * in backend_cuda.h, not in the loader): a resident tensor's format and shape, and its GEMV /
 * GEMM on a queue the caller owns (a sycl::queue *, NULL = the device's) with device pointers,
 * no transfer, no synchronization. */
extern "C" int coli_cuda_tensor_shape(const ColiCudaTensor *t, int *fmt, int *I, int *O);
extern "C" int coli_cuda_tensor_gemm_async(ColiCudaTensor *t, float *y_dev, const float *x_dev, int S, void *stream);
extern "C" int coli_cuda_tensor_gemm_grp_async(ColiCudaTensor *const *t, int ng, float *y_dev, int y_row, const float *x_dev,
                                               int x_row, int x_grp, int S, void *stream);
extern "C" int coli_cuda_tensor_glu_async(ColiCudaTensor *tg, ColiCudaTensor *tu, float *y_dev, const float *x_dev, int S, void *stream);
extern "C" int coli_cuda_tensor_gemv_rows_async(ColiCudaTensor *t, const int *ids_dev, float *y_dev, const float *x_dev, int n, void *stream);
/* and the routed experts' kernels (cc_moe), on the same terms, and a resident tensor's device copy */
extern "C" int coli_cuda_tensor_dev(const ColiCudaTensor *t, const void **w, const float **sc, int *gs);
extern "C" int coli_cuda_tensor_gemm_hcmix_async(ColiCudaTensor *t, float *mixed, const float *x, const float *hn, int S, int H, int C,
                                                 float fc, void *stream);
extern "C" int coli_cuda_moe_cols_async(int device, const CcExpert *tab, const float *x, const int *ids, const float *w,
                                        float *hid, float *y, const CcMoe *p, int cols, void *stream);
extern "C" int coli_cuda_moe_stage(int device, void *stage, size_t slot_bytes, int slots);
extern "C" int coli_cuda_moe_async(int device, const CcExpert *tab, const float *x, const int *ids, const float *w,
                                   float *hid, float *y, const float *hy, const CcMoe *p, void *stream);

struct CcBuf { int kind, dev; size_t bytes; float *d; void *h; };

namespace {
#define CC_UP_MARKS 1024   /* host_up's marks in flight (three a promoted expert) */
#define CC_STAGE_N 32      /* cc_write's staging slots in flight */
/* the context's events, SYCL objects: made with the context (new), not in the static table */
struct CcSync {
    sycl::event ev;                       /* the frame's end (cc_submit) */
    sycl::event ev_fetched, ev_free;      /* staging (cc_fetch): the copies' and the buffer's last reader's */
    sycl::event up_ev[CC_UP_MARKS];       /* host_up: the copies' marks, a ring on the copy queue */
    sycl::event up_after;                 /* the frame's point a host_up waits for */
    sycl::event stg_ev[CC_STAGE_N];       /* cc_write: the copy that last read each staging slot */
    sycl::event mark;                     /* cc_mark: the point of the open frame the host asks about (cc_reached) */
    int mark_set = 0;
};
struct CcCtx {
    int device, up, lost, in_frame;
    int sg;                               /* the sub-group the warp-32-style kernels run at (the profile's) */
    int sg_dn, sg_attn;                    /* per-kernel widths; the route and norms keep sg */
    int sg_dnp;                            /* cc_dnrec_split_k's (a prompt chunk's rows): COLI_CUDA_DNREC_PROMPT_SG, else 16 on Max cards, else sg */
    size_t slm;                           /* the local memory a work-group may take (the profile's) */
    int fp64;                             /* the device has doubles (the encoders' LayerNorm sums) */
    sycl::queue *stream;                  /* the device's in-order queue: the frames' order */
    sycl::queue *cs;                      /* the copy queue (staging, host_up), in order too */
    CcSync *sy; int ev_pending;
    unsigned up_seq, up_made;
    void *stg[CC_STAGE_N]; size_t stg_cap[CC_STAGE_N]; unsigned stg_next;   /* cc_write's staging ring */
    float *adec; size_t adec_bytes;       /* cc_attn_dec_k's blocks */
    /* the scratch a resize retired: freed once a later frame has been waited for (a queue wait
     * here could wait on a cc_wait_host kernel whose host side comes after this frame) */
    float *adec_old[4]; int adec_n_old;
};
CcCtx g_cc[COLI_CUDA_MAX_DEVICES];
int g_ncc, g_cur = -1;   /* g_cur: index into g_cc of the current context */
CcStats g_st;

double cc_now_ms(void) {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}
CcCtx *cur(void) { return g_cur >= 0 ? &g_cc[g_cur] : nullptr; }
CcCtx *ctx_of(int device) { for (int i = 0; i < g_ncc; i++) if (g_cc[i].device == device) return &g_cc[i]; return nullptr; }
/* The device check (COLI_SYCL_TEST_CPU: logical devices on the CPU device, tests only; nothing otherwise). With two
 * devices an engine runs some layers' frames on each (deepseek_v41_cuchain.h, V41_CHAIN_SPLIT), and every buffer,
 * table and expert a frame's op reads or writes must be that device's: on cards another device's address is not
 * mapped and the kernel faults (a GPU page fault on the second card, seen once), while on the CPU device, where every
 * logical device's memory is the host's, the same mistake computes the right answer. So here each op says whose
 * its arguments are before it records anything: a buffer by the device it was made on (and, asked of the runtime, the
 * context that owns its memory), a raw device pointer by the owning context. A foreign one aborts with the op's name
 * and the argument. */
int cc_own_which(const void *ptr) {   /* the chain device whose context owns ptr, -1 none of them */
    for (int i = 0; i < g_ncc; i++) {
        if (!g_cc[i].up || !g_cc[i].stream) continue;
        try { if (sycl::get_pointer_type(ptr, g_cc[i].stream->get_context()) != sycl::usm::alloc::unknown) return g_cc[i].device; }
        catch (...) {}
    }
    return -1;
}
void cc_own_ptr(CcCtx *c, const char *op, const char *what, const void *ptr) {
    if (!c || !ptr || !c->stream) return;
    int mine = 0;
    try { mine = sycl::get_pointer_type(ptr, c->stream->get_context()) != sycl::usm::alloc::unknown; } catch (...) { mine = 0; }
    if (mine) return;
    std::fprintf(stderr, "[SYCL] device check (COLI_SYCL_TEST_CPU): %s: %s (%p) is not device %d's memory, where this frame runs "
                         "(it is device %d's)\n", op, what, ptr, c->device, cc_own_which(ptr));
    std::fflush(stderr);
    std::abort();
}
void cc_own_check(CcCtx *c, const char *op, const char *names, const CcBuf *const *b, int n) {
    if (!c) return;
    for (int k = 0; k < n; k++) {
        if (!b[k]) continue;
        if (b[k]->dev != c->device) {
            std::fprintf(stderr, "[SYCL] device check (COLI_SYCL_TEST_CPU): %s: buffer argument %d of (%s) was made on device %d, "
                                 "this frame runs on device %d\n", op, k + 1, names, b[k]->dev, c->device);
            std::fflush(stderr);
            std::abort();
        }
        cc_own_ptr(c, op, "a buffer's memory", b[k]->d);
    }
}
#define CC_OWN(c, op, ...) do { if (coli_sycl_test_devices()) { const CcBuf *const own_[] = { __VA_ARGS__ }; \
        cc_own_check((c), op, #__VA_ARGS__, own_, (int)(sizeof own_ / sizeof *own_)); } } while (0)
/* a layer's table of experts (cc_moe's): every entry's tensors and scales this device's (E entries from t_off) */
void cc_own_tensor(CcCtx *c, const char *op, const ColiCudaTensor *t) {
    if (!coli_sycl_test_devices() || !c || !t || coli_cuda_tensor_device(t) == c->device) return;
    std::fprintf(stderr, "[SYCL] device check (COLI_SYCL_TEST_CPU): %s: the matrix is on device %d, this frame runs on device %d\n",
                 op, coli_cuda_tensor_device(t), c->device);
    std::fflush(stderr);
    std::abort();
}
void cc_own_table(CcCtx *c, const char *op, const CcBuf *tab, int t_off, int E) {
    if (!coli_sycl_test_devices() || !c || !tab || E < 1 || t_off < 0 || ((size_t)t_off + (size_t)E) * sizeof(CcExpert) > tab->bytes) return;
    const CcExpert *t = (const CcExpert *)tab->d + t_off;   /* the CPU device: its memory is the host's */
    for (int e = 0; e < E; e++) {
        const void *q[6] = { t[e].g, t[e].u, t[e].d, t[e].gs, t[e].us, t[e].ds };
        static const char *const nm[6] = { "an expert's gate weights", "an expert's up weights", "an expert's down weights",
                                           "an expert's gate scales", "an expert's up scales", "an expert's down scales" };
        for (int k = 0; k < 6; k++) if (q[k]) cc_own_ptr(c, op, nm[k], q[k]);
    }
}
/* every call: the context, up and not lost (the queues carry their device: no cudaSetDevice) */
int cc_fail(CcCtx *c, const char *what, const char *why);
/* An asynchronous error (a command the runtime reports as failed, delivered to the queue's
 * handler, never thrown to the caller) sets the device's flag (coli_sycl_lost): read at every
 * call and after every try, so it loses the context exactly as a thrown exception does */
int cc_async_lost(CcCtx *c, const char *what) {
    if (!c || c->lost || !coli_sycl_lost(c->device)) return 0;
    cc_fail(c, what, "an asynchronous error on the device's queue");
    return 1;
}
CcCtx *live(void) {
    CcCtx *c = cur();
    if (c && c->up) cc_async_lost(c, "call");
    return c && c->up && !c->lost ? c : nullptr;
}
/* A failure on the device loses the context to the chain, as a CUDA error did: cc_fail logs and
 * marks it; cc_try runs a call that may throw (a submit, a wait) and returns 1, or fails the
 * context and returns 0; launched is what a launch site returns, the context still there. */
void cc_mark_lost(CcCtx *c) {
    if (!c) return;
    c->lost = 1;
    coli_sycl_mark_lost(c->device);  // releases host waits without a queue call
}
int cc_fail(CcCtx *c, const char *what, const char *why) {
    std::fprintf(stderr, "[SYCL] chain: %s: %s -- the device is lost to the chain\n", what, why);
    cc_mark_lost(c);
    return 0;
}
template <typename F> int cc_try(CcCtx *c, const char *what, F f) {
    try { f(); }
    catch (const sycl::exception &e) { return cc_fail(c, what, e.what()); }
    catch (const std::exception &e) { return cc_fail(c, what, e.what()); }
    return cc_async_lost(c, what) ? 0 : 1;
}
int launched(CcCtx *c, const char *what) {
    g_st.ops++;
    if (coli_sycl_fprof_flag) coli_sycl_fprof_name(what);   /* the frame profile: the commands just submitted are this op's */
    return c && !c->lost;
}
/* the frame profile's host clock (coli_sycl.h section 5): a frame runs from the first cc_begin
 * after a wait to the wait that covers it (cc_submit with a wait, cc_finish); its submit time
 * ends at the last cc_submit's barrier */
double g_fp_t0, g_fp_tsub; int g_fp_open;
void cc_fprof_begin(void) {
    if (!coli_sycl_fprof_flag || g_fp_open) return;
    g_fp_open = 1; g_fp_t0 = cc_now_ms(); g_fp_tsub = 0;
}
void cc_fprof_end(void) {
    if (!coli_sycl_fprof_flag || !g_fp_open) return;
    const double now = cc_now_ms(), sub = g_fp_tsub > 0 ? g_fp_tsub : now;
    g_fp_open = 0;
    coli_sycl_fprof_frame(now - g_fp_t0, sub - g_fp_t0, now - sub);
}
/* memory on the context's device (host 1: host USM, page-locked and mapped): NULL when there is no
 * room, the context kept, as cudaMalloc's failure was the caller's to size down; a release never throws */
void *cc_alloc(CcCtx *c, size_t bytes, int host) {
    try { return host ? sycl::malloc_host(bytes, *c->stream) : sycl::malloc_device(bytes, *c->stream); }
    catch (const sycl::exception &) { return nullptr; }
}
void cc_release(CcCtx *c, void *p) {
    if (!p) return;
    try { sycl::free(p, *c->stream); } catch (const sycl::exception &) {}
}
/* the retired attention scratches, after the queue was seen to pass the frames that read them */
void cc_adec_reap(CcCtx *c) {
    for (int i = 0; i < c->adec_n_old; i++) cc_release(c, c->adec_old[i]);
    c->adec_n_old = 0;
}
unsigned grid1(size_t n, unsigned block) { size_t g = (n + block - 1) / block; return g > 0x7fffffffu ? 0x7fffffffu : (unsigned)g; }
const float *fp(const CcBuf *b) { return b ? b->d : nullptr; }
float *fpw(CcBuf *b) { return b ? b->d : nullptr; }
/* dim3(x, y, z) as a range: SYCL's slowest index first, so x is the last (the kernels read
 * blockIdx.x as get_group(2), threadIdx.x as get_local_id(2)) */
sycl::range<3> dim3(unsigned x, unsigned y = 1, unsigned z = 1) { return sycl::range<3>(z, y, x); }
}

/* A launch: `grid` blocks of `block` work-items on the chain's queue, the statement a kernel call
 * over values it captures by copy (a buffer's pointer taken before, never the CcBuf: the host's).
 * A submit that throws loses the context (cc_try); launched reads that after. CC_KSG is the same
 * at the context's sub-group size for the warp-32-style kernels: SG names it in the statement
 * (coli_sycl_launch_sg builds the kernel at 16 and at 32 and runs the one the profile chose),
 * slm_ the launch's dynamic local memory (slm bytes, the extern __shared__ of the .cu). */
#define CC_K(c, grid, block, ...) cc_try((c), "launch", [&] { \
        coli_sycl_fprof_ev((c)->stream->parallel_for(sycl::nd_range<3>((grid) * sycl::range<3>(1, 1, (block)), sycl::range<3>(1, 1, (block))), \
                                  [=](sycl::nd_item<3>) { __VA_ARGS__; })); })
#define CC_KSG_AT(c, sg_, grid, block, slm, ...) cc_try((c), "launch", [&] { \
        coli_sycl_fprof_ev(coli_sycl_launch_sg(*(c)->stream, (sg_), (grid), sycl::range<3>(1, 1, (block)), (slm), \
                            [=](sycl::nd_item<3>, auto tag, uint8_t *slm_) { \
                                constexpr int SG = decltype(tag)::value; (void)SG; (void)slm_; __VA_ARGS__; })); })
#define CC_KSG(c, grid, block, slm, ...) CC_KSG_AT(c, (c)->sg, grid, block, slm, __VA_ARGS__)

/* ---- device helpers ---------------------------------------------------------------- */
/* the shuffle (__shfl_xor_sync's place): across this sub-group's lanes, SG of them */
#define cc_shfl_xor(v, o) sycl::permute_group_by_xor(sycl::ext::oneapi::this_work_item::get_sub_group(), (v), (o))
/* a warp's max, sum and min over its SG lanes, every lane the result (chain_route.comp's subgroup ops) */
template <int SG> inline float cc_wmax(float v) { for (int o = SG / 2; o > 0; o >>= 1) v = sycl::fmax(v, cc_shfl_xor(v, o)); return v; }
template <int SG> inline float cc_wsum(float v) { for (int o = SG / 2; o > 0; o >>= 1) v += cc_shfl_xor(v, o); return v; }
template <int SG> inline int cc_wmin(int v) { for (int o = SG / 2; o > 0; o >>= 1) v = sycl::min(v, cc_shfl_xor(v, o)); return v; }
/* atomicAdd on a word of local memory (cc_qsa_k's histogram) */
inline void cc_local_add(unsigned &w, unsigned v) {
    sycl::atomic_ref<unsigned, sycl::memory_order::relaxed, sycl::memory_scope::work_group, sycl::access::address_space::local_space> a(w);
    a.fetch_add(v);
}

/* ---- kernels: shaders/chain_*.comp, line for line --------------------------- */
/* The norms' 1 / sqrt(sum / n + eps) (n 0: the sum as it is, the L2 norm's flags & 4), one body
 * for cc_norm_k and cc_addnorm_k, never inlined. The .cu wrote 1.f / sqrtf(...) in both and nvcc
 * gave both the same IEEE division and square root. Here the device compiler (IGC) is free with
 * fp32 division and sqrt (OpenCL's 2.5 / 3 ulp: nothing in the build asks for correctly rounded
 * ones) and lowered the same expression two ways on the first card: in cc_addnorm_k it matched
 * 1 / sqrt(x) and emitted one math.rsqt, in cc_norm_k, where the two branches of flags & 4 meet
 * before the division, math.sqt then math.inv -- a different rounding of r in some rows, and
 * addnorm's output off by an ulp from norm's (the test's "x and its norm bit for bit"). An
 * explicit rsqrt (what the matched kernel computed anyway; within the hardware's ulp, as the
 * folded form was) in a subroutine IGC compiles once, out of any caller's context (checked in
 * the AOT ISA: one call, one math.inv, one math.rsqt in every kernel), is the same bits in both.
 * MIGRATION.md section 13. */
__attribute__((noinline)) static float cc_rstd(float ss, float n, float eps) {
    const float ms = n > 0.f ? ss / n : ss;
    return sycl::rsqrt(ms + eps);
}
/* NV: the elements a work-item holds in registers (D <= 256 NV). 8 is the kernel as it was (up to 2048 a row, longer
 * rows through the loops); 20 takes a row of up to 5120 the same way (DeepSeek V4.1's two FFN/attention input norms a
 * layer, D 5120: 25 us a launch through the loops against 6 us for the short rows, by a frame profile) -- the values,
 * the order of the sum and the products are the loops', so the bits are. COLI_SYCL_NORM_NV=8: as before. */
template <int NV>
static void cc_norm_k(const float *__restrict__ x, const float *__restrict__ w,
                      float *__restrict__ y, CcNorm p) {
    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto &red =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[256]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    int g = it.get_group(2);
    if (g >= p.nseg) return;
    int row = g / p.per_row, j = g - row * p.per_row;
    int xb = p.x_off + row * p.x_row + j * p.x_seg;
    int yb = p.y_off + row * p.y_row + j * p.y_seg;
    int wb = p.w_off + (p.w_mod > 0 ? (j % p.w_mod) * p.D : 0);
    int tid = it.get_local_id(2);
    float acc = 0.f;
    const bool pre = (p.flags & 8) != 0, wt = pre || !(p.flags & 2);
    /* up to 2048 a row: x and the weights in registers, loaded together before the sum (the
     * weights used to be read after it, a second trip to memory on the step's path); the same
     * values in the same order, so the same bits */
    const bool fit = p.D <= NV * 256;
    float xv[NV], wv8[NV];
    if (fit) {
        #pragma unroll
        for (int k = 0; k < NV; k++) {
            const int i = tid + 256 * k;
            xv[k] = i < p.D ? x[xb + i] : 0.f;
            wv8[k] = wt && i < p.D ? w[wb + i] : 0.f;
        }
        #pragma unroll
        for (int k = 0; k < NV; k++)
            if (tid + 256 * k < p.D) { float v = pre ? xv[k] * wv8[k] : xv[k]; acc += v * v; }
    } else
        for (int i = tid; i < p.D; i += 256) { float v = pre ? x[xb + i] * w[wb + i] : x[xb + i]; acc += v * v; }
    red[tid] = acc;
    it.barrier(sycl::access::fence_space::local_space);
    for (int off = 128; off > 0; off >>= 1) {
        if (tid < off) red[tid] += red[tid + off];
        it.barrier(sycl::access::fence_space::local_space);
    }
    float r = cc_rstd(red[0], (p.flags & 4) ? 0.f : (float)p.D, p.eps);   /* the .cu's 1.f / sqrtf, see cc_rstd */
    if (fit) {
        #pragma unroll
        for (int k = 0; k < NV; k++) {
            const int i = tid + 256 * k;
            if (i >= p.D) continue;
            float wv = (p.flags & 2) ? 1.f : ((p.flags & 1) ? 1.f + wv8[k] : wv8[k]);
            y[yb + i] = xv[k] * (pre ? wv8[k] : wv) * r * p.post;
        }
    } else
        for (int i = tid; i < p.D; i += 256) {
            float wv = (p.flags & 2) ? 1.f : ((p.flags & 1) ? 1.f + w[wb + i] : w[wb + i]);
            y[yb + i] = x[xb + i] * (pre ? w[wb + i] : wv) * r * p.post;
        }
}
/* cc_addnorm: a block a row, a thread elements tid + 256 k as cc_norm_k's loops take them and its
 * tree, so each sum, scale and product is cc_norm_k's and the add cc_ew_k's (the products and the
 * add spelled out: an fma would round once where the kernels apart round twice). NV: D <= 256 NV */
template <int NV>
static void cc_addnorm_k(float *__restrict__ x, const float *__restrict__ t,
                         const float *__restrict__ w, float *__restrict__ y,
                         CcAddNorm p) {
    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto &red =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[256]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int row = it.get_group(2), tid = it.get_local_id(2),
              D = p.D;
    float *xr = x + p.x_off + (size_t)row * D, *yr = y + p.y_off + (size_t)row * D;
    const float *tr = t + p.t_off + (size_t)row * D;
    /* every load first: t, x and both weights in one round trip to memory, not four in a row
     * (a decode step's 2048 values a block: the trips were most of the kernel's time) */
    float v[NV], xv[NV], wp[NV], wn[NV];
    #pragma unroll
    for (int k = 0; k < NV; k++) {
        const int i = tid + 256 * k;
        v[k] = i < D ? tr[i] : 0.f;
        xv[k] = i < D ? xr[i] : 0.f;
        wp[k] = (p.flags & 1) && i < D ? w[p.wp_off + i] : 0.f;
        wn[k] = i < D ? w[p.w_off + i] : 0.f;
    }
    float acc = 0.f;
    #pragma unroll
    for (int k = 0; k < NV; k++) {
        const int i = tid + 256 * k;
        if ((p.flags & 1) && i < D) acc += v[k] * v[k];
    }
    if (p.flags & 1) {
        red[tid] = acc;
        it.barrier();
        for (int off = 128; off > 0; off >>= 1) {
            if (tid < off) red[tid] += red[tid + off];
            it.barrier();
        }
        const float r = cc_rstd(red[0], (float)D, p.eps);   /* cc_norm_k's r, the same subroutine */
        it.barrier();
#pragma unroll
        for (int k = 0; k < NV; k++) {
            const int i = tid + 256 * k;
            if (i < D) v[k] = ((v[k] * (1.f + wp[k])) * r) * 1.f;   /* the .cu's nested __fmul_rn: ((v * (1 + w)) * r) * 1 */
        }
    }
    acc = 0.f;
    #pragma unroll
    for (int k = 0; k < NV; k++) {
        const int i = tid + 256 * k;
        if (i < D) { v[k] = xv[k] + v[k]; xr[i] = v[k]; acc += v[k] * v[k]; }
    }
    red[tid] = acc;
    it.barrier();
    for (int off = 128; off > 0; off >>= 1) {
        if (tid < off) red[tid] += red[tid + off];
        it.barrier();
    }
    const float r = cc_rstd(red[0], (float)D, p.eps);
#pragma unroll
    for (int k = 0; k < NV; k++) {
        const int i = tid + 256 * k;
        if (i < D) yr[i] = ((v[k] * (1.f + wn[k])) * r) * 1.f;
    }
}
/* kv16.h's kv16_round_bits line for line: clamped, the nearest float16 (ew's CC_EW_F16) */
static inline float cc_f16_round(float x) {
    if (x > 65504.f) x = 65504.f; else if (x < -65504.f) x = -65504.f;
    unsigned u = sycl::bit_cast<unsigned int>(x), s = u & 0x80000000u,
             m = u & 0x7fffffffu;
    if (m > 0x7f800000u) u = u | 0x400000u;
    else if (m >= 0x38800000u) u = s | ((m + 0xfffu + ((m >> 13) & 1u)) & 0xffffe000u);
    else {
        unsigned e = m >> 23, mm = (m & 0x7fffffu) | 0x800000u;
        if (e < 102u) u = s;
        else {
            unsigned sh = 126u - e, n = (mm + (1u << (sh - 1u)) - 1u + ((mm >> sh) & 1u)) >> sh;
            u = s |
                sycl::bit_cast<unsigned int>((float)n * 5.9604644775390625e-8f);
        }
    }
    return sycl::bit_cast<float>(u);
}
/* one rotate-half pair, its roundings spelled out: a * c - b * s and b * c + a * s leave the
 * compiler free to fuse either product, and it fused a different one once the pair moved here.
 * These are the ones nvcc picked for cc_rope_k on sm_120 before, for cc_rope_k and cc_rope_f16_k
 * alike, so their bits agree on every target and, on sm_120, are the rope's of before */
static inline void cc_rope_pair(float a, float b, float c, float s, float *ra, float *rb) {
    *ra = sycl::fma(a, c, -b * s);
    *rb = sycl::fma(a, s, b * c);
}
static void cc_rope_k(float *__restrict__ x, const float *__restrict__ cs, CcRope p) {
    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    long long gid =
        (long long)it.get_group(2) * it.get_local_range(2) +
        it.get_local_id(2);
    if (gid >= (long long)p.nseg * p.half_) return;
    int seg = (int)(gid / p.half_), j = (int)(gid - (long long)seg * p.half_);
    int row = seg / p.per_row, hh = seg - row * p.per_row;
    int base = p.x_off + row * p.x_row + hh * p.x_seg;
    int cb = p.cs_off + row * p.cs_row + 2 * j;
    cc_rope_pair(x[base + j], x[base + j + p.half_], cs[cb], cs[cb + 1], &x[base + j], &x[base + j + p.half_]);
}
/* cc_rope_k, then each of a segment's seg values rounded to float16 (cc_f16_round), and the
 * same values of v (at x's offsets) too: a float16 KV cache's new K and V rows in one launch
 * where the rope and two ew took three. A thread a value: below half_ its pair, from 2 * half_
 * the values past the rotation. */
static void cc_rope_f16_k(float *__restrict__ x, float *__restrict__ v, const float *__restrict__ cs, CcRope p, int seg) {
    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    long long gid =
        (long long)it.get_group(2) * it.get_local_range(2) +
        it.get_local_id(2);
    if (gid >= (long long)p.nseg * seg) return;
    int sg = (int)(gid / seg), j = (int)(gid - (long long)sg * seg);
    int row = sg / p.per_row, hh = sg - row * p.per_row;
    int base = p.x_off + row * p.x_row + hh * p.x_seg;
    if (j < p.half_) {
        int cb = p.cs_off + row * p.cs_row + 2 * j;
        float ra, rb;
        cc_rope_pair(x[base + j], x[base + j + p.half_], cs[cb], cs[cb + 1], &ra, &rb);
        x[base + j] = cc_f16_round(ra);
        x[base + j + p.half_] = cc_f16_round(rb);
    } else if (j >= 2 * p.half_) x[base + j] = cc_f16_round(x[base + j]);
    v[base + j] = cc_f16_round(v[base + j]);
}
/* A cache's elements as floats: float32, or float16 (attn_h: a float16 KV cache holds the
 * numbers a float16 host cache holds, so the attention reads the same values). cc_kv4: four
 * of them at an offset that is a multiple of four. */
inline float cc_kvf(const float *p, size_t i) { return p[i]; }
inline sycl::float4 cc_kv4(const float *p, size_t i) { return *reinterpret_cast<const sycl::float4 *>(p + i); }
inline float cc_kvf(const sycl::half *p, size_t i) { return (float)p[i]; }
inline sycl::float4 cc_kv4(const sycl::half *p, size_t i) {
    const sycl::half4 h = *reinterpret_cast<const sycl::half4 *>(p + i);   /* eight bytes, aligned as the offset is */
    return h.convert<float, sycl::rounding_mode::automatic>();
}
/* cc_kv2: two of them at an even offset (one load) */
inline sycl::float2 cc_kv2(const float *p, size_t i) { return *reinterpret_cast<const sycl::float2 *>(p + i); }
inline sycl::float2 cc_kv2(const sycl::half *p, size_t i) {
    const sycl::vec<sycl::half, 2> h = *reinterpret_cast<const sycl::vec<sycl::half, 2> *>(p + i);   /* four bytes, aligned as the offset is */
    return h.convert<float, sycl::rounding_mode::automatic>();
}
/* cc_kv8: eight of them at an offset that is a multiple of eight (one load a work item) */
inline sycl::vec<float, 8> cc_kv8(const float *p, size_t i) { return *reinterpret_cast<const sycl::vec<float, 8> *>(p + i); }
inline sycl::vec<float, 8> cc_kv8(const sycl::half *p, size_t i) {
    const sycl::vec<sycl::half, 8> h = *reinterpret_cast<const sycl::vec<sycl::half, 8> *>(p + i);   /* sixteen bytes, aligned as the offset is */
    return h.convert<float, sycl::rounding_mode::automatic>();
}
template <typename KT> static const KT *kvp(CcBuf *b) { return b ? reinterpret_cast<const KT *>(b->d) : nullptr; }
/* one block of 128 per (head, row); K and V rows read once per tile of 128 positions;
 * hd <= 256: two dims per thread */
/* win > 0: a sliding window, the row's last win positions; snk: a sink logit per head at
 * snk[sink_off + h], in the denominator only (CcAttnW); ring > 0: the cache a ring of that
 * many rows, position t in row t % ring (a sliding layer's, cc_attn_wr) */
template <typename KT = float>
static void cc_attn_k(const float *__restrict__ q, const KT *__restrict__ kc,
                      const KT *__restrict__ vc, float *__restrict__ o,
                      const float *__restrict__ gt, const int *__restrict__ sel,
                      CcAttn p, int win = 0,
                      const float *__restrict__ snk = nullptr, int sink_off = 0,
                      int ring = 0) {
    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto &qs = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[256]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &pe = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[128]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &pt = *sycl::ext::oneapi::group_local_memory_for_overwrite<int[128]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &red =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[128]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    int h = it.get_group(2), s = it.get_group(1),
        tid = it.get_local_id(2);
    int kvh = h / (p.H / p.KVH);
    int qb = p.q_off + s * p.q_row + h * p.q_seg;
    for (int d = tid; d < p.hd; d += 128) qs[d] = q[qb + d];
    int n = p.pos_base + s + 1;
    bool list = false;
    int lb = 0;
    if (p.sel_row > 0) {
        lb = p.sel_off + s * p.sel_row;
        int c = sel[lb];
        if (c >= 0) { n = c; list = true; }
    }
    it.barrier(sycl::access::fence_space::local_space);
    float m = -3.0e38f, l = 0.f, acc0 = 0.f, acc1 = 0.f;
    int kbase = kvh * p.cap;
    int first = win > 0 && !list && n > win ? n - win : 0;
    for (int t0 = first; t0 < n; t0 += 128) {
        int i = t0 + tid;
        float sv = -3.0e38f;
        if (i < n) {
            int t = list ? sel[lb + 1 + i] : i;
            if (ring) t %= ring;
            pt[tid] = t;
            int kb = p.k_off + (kbase + t) * p.hd;
            float a = 0.f;
            for (int d = 0; d < p.hd; d++) a += qs[d] * cc_kvf(kc, kb + d);
            sv = a * p.scale;
        }
        red[tid] = sv;
        it.barrier(sycl::access::fence_space::local_space);
        for (int off = 64; off > 0; off >>= 1) {
            if (tid < off) red[tid] = sycl::fmax(red[tid], red[tid + off]);
            it.barrier(sycl::access::fence_space::local_space);
        }
        float mn = sycl::fmax(m, red[0]);
        it.barrier(sycl::access::fence_space::local_space);
        float e = i < n ? sycl::native::exp(sv - mn) : 0.f;
        pe[tid] = e;
        red[tid] = e;
        it.barrier(sycl::access::fence_space::local_space);
        for (int off = 64; off > 0; off >>= 1) {
            if (tid < off) red[tid] += red[tid + off];
            it.barrier(sycl::access::fence_space::local_space);
        }
        float corr = sycl::native::exp(m - mn);
        l = l * corr + red[0];
        acc0 *= corr; acc1 *= corr;
        int cnt = sycl::min(128, n - t0);
        int d0 = tid, d1 = tid + 128;
        for (int j = 0; j < cnt; j++) {
            int vb = p.v_off + (kbase + pt[j]) * p.hd;
            float w = pe[j];
            if (d0 < p.hd) acc0 += w * cc_kvf(vc, vb + d0);
            if (d1 < p.hd) acc1 += w * cc_kvf(vc, vb + d1);
        }
        m = mn;
        it.barrier(sycl::access::fence_space::local_space);
    }
    if (snk) {   /* the sink: one more term of the denominator, with no value */
        float sv = snk[sink_off + h], mn = sycl::fmax(m, sv),
              corr = sycl::native::exp(m - mn);
        l = l * corr + sycl::native::exp(sv - mn);
        acc0 *= corr; acc1 *= corr;
    }
    float inv = l > 0.f ? 1.f / l : 0.f;
    int ob = p.o_off + s * p.o_row + h * p.hd;
    int gb = p.g_off + s * p.g_row + h * p.g_seg;
    for (int k = 0; k < 2; k++) {
        int d = tid + 128 * k;
        if (d >= p.hd) break;
        float v = (k == 0 ? acc0 : acc1) * inv;
        if (p.has_gate) v *= 1.f / (1.f + sycl::native::exp(-gt[gb + d]));
        o[ob + d] = v;
    }
}
/* cc_attn_k for a prompt chunk's rows with a selection list (QSA), a block per (kv head, row):
 * the G query heads of a kv head read the row's selected keys and values once for all of them,
 * where a block per query head read them G times (Qwen3.8: 12, the keys from L2 every time).
 * Each head's arithmetic is cc_attn_k's, in its order (the dot over d, the two trees, the
 * values in position order), so its bits are the same. Grid: x = kv heads, y = rows. */
template <int G, typename KT = float>
static void cc_attn_gk(const float *__restrict__ q, const KT *__restrict__ kc,
                       const KT *__restrict__ vc, float *__restrict__ o,
                       const float *__restrict__ gt,
                       const int *__restrict__ sel, CcAttn p) {
    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto &qs =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[G][256]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &pe =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[G][128]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &pt = *sycl::ext::oneapi::group_local_memory_for_overwrite<int[128]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    /* a tile's 128 selected keys, 32 of their elements at a time, in the cache's type with a pad word
     * (a thread's row reads no bank another of its warp reads); the reductions' red after them */
    constexpr int KS = sizeof(KT) == 2 ? 34 : 33;
    constexpr size_t KSB = (size_t)128 * KS * sizeof(KT), RDB = (size_t)G * 128 * sizeof(float);
    auto &ksm = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        unsigned char[KSB > RDB ? KSB : RDB]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    KT (*Ks)[KS] = reinterpret_cast<KT (*)[KS]>(ksm);
    float (*red)[128] = reinterpret_cast<float (*)[128]>(ksm);
    const int kvh = it.get_group(2), s = it.get_group(1),
              tid = it.get_local_id(2), h0 = kvh * G;
#pragma unroll
    for (int g = 0; g < G; g++) {
        const int qb = p.q_off + s * p.q_row + (h0 + g) * p.q_seg;
        for (int d = tid; d < p.hd; d += 128) qs[g][d] = q[qb + d];
    }
    int n = p.pos_base + s + 1;
    bool list = false;
    int lb = 0;
    if (p.sel_row > 0) {
        lb = p.sel_off + s * p.sel_row;
        int c = sel[lb];
        if (c >= 0) { n = c; list = true; }
    }
    it.barrier(sycl::access::fence_space::local_space);
    float m[G], l[G], acc0[G], acc1[G];
    #pragma unroll
    for (int g = 0; g < G; g++) { m[g] = -3.0e38f; l[g] = 0.f; acc0[g] = 0.f; acc1[g] = 0.f; }
    const int kbase = kvh * p.cap;
    for (int t0 = 0; t0 < n; t0 += 128) {
        const int i = t0 + tid;
        float sv[G];
        #pragma unroll
        for (int g = 0; g < G; g++) sv[g] = -3.0e38f;
        if (i < n) pt[tid] = list ? sel[lb + 1 + i] : i;
        it.barrier(sycl::access::fence_space::local_space);
        /* each thread its position's dot over d in order (cc_attn_k's), the keys staged in Ks: a thread
         * reading its own key's 256 elements in turn left the warps' loads 32 lines apart, the kernel
         * bound by L1 (82% of it, the multiprocessors at 43%: Qwen3.8's prompt on an RTX PRO 4000) */
        const int cnt = sycl::min(128, n - t0),
                  vec = (p.hd & 7) == 0 && (p.k_off & 7) == 0;
        float a[G];
        #pragma unroll
        for (int g = 0; g < G; g++) a[g] = 0.f;
        for (int d0 = 0; d0 < p.hd; d0 += 32) {
            #pragma unroll
            for (int pass = 0; pass < 4; pass++) {   /* four threads a key, eight elements each */
                const int r = (tid >> 2) + 32 * pass, e0 = 8 * (tid & 3);
                if (r >= cnt) continue;
                const size_t kb = (size_t)p.k_off + (size_t)(kbase + pt[r]) * p.hd + d0 + e0;
                if (vec && d0 + e0 + 8 <= p.hd) {
                    if (sizeof(KT) == 2) {
                        const sycl::uint4 v =
                            *reinterpret_cast<const sycl::uint4 *>(kc + kb);
                        unsigned *dst = reinterpret_cast<unsigned *>(&Ks[r][e0]);
                        dst[0] = v.x(); dst[1] = v.y(); dst[2] = v.z();
                            dst[3] = v.w();
                    } else {
                        const sycl::float4
                            v0 = *reinterpret_cast<const sycl::float4 *>(kc +
                                                                         kb),
                            v1 = *reinterpret_cast<const sycl::float4 *>(
                                kc + kb + 4);
                        float *dst = reinterpret_cast<float *>(&Ks[r][e0]);
                        dst[0] = v0.x(); dst[1] = v0.y(); dst[2] = v0.z();
                            dst[3] = v0.w(); dst[4] = v1.x(); dst[5] = v1.y();
                            dst[6] = v1.z(); dst[7] = v1.w();
                    }
                } else {
                    for (int e = 0; e < 8 && d0 + e0 + e < p.hd; e++) Ks[r][e0 + e] = kc[kb + e];
                }
            }
            it.barrier(sycl::access::fence_space::local_space);
            if (i < n) {
                const int dn = p.hd - d0 < 32 ? p.hd - d0 : 32;
                for (int d = 0; d < dn; d++) {
                    const float kv = cc_kvf(Ks[tid], d);
                    #pragma unroll
                    for (int g = 0; g < G; g++) a[g] += qs[g][d0 + d] * kv;
                }
            }
            it.barrier(sycl::access::fence_space::local_space);
        }
        if (i < n) {
            #pragma unroll
            for (int g = 0; g < G; g++) sv[g] = a[g] * p.scale;
        }
        #pragma unroll
        for (int g = 0; g < G; g++) red[g][tid] = sv[g];
        it.barrier(sycl::access::fence_space::local_space);
        for (int off = 64; off > 0; off >>= 1) {
            if (tid < off) {
                #pragma unroll
                for (int g = 0; g < G; g++)
                    red[g][tid] = sycl::fmax(red[g][tid], red[g][tid + off]);
            }
            it.barrier(sycl::access::fence_space::local_space);
        }
        float mn[G];
        #pragma unroll
        for (int g = 0; g < G; g++) mn[g] = sycl::fmax(m[g], red[g][0]);
        it.barrier(sycl::access::fence_space::local_space);
#pragma unroll
        for (int g = 0; g < G; g++) {
            const float e = i < n ? sycl::native::exp(sv[g] - mn[g]) : 0.f;
            pe[g][tid] = e;
            red[g][tid] = e;
        }
        it.barrier(sycl::access::fence_space::local_space);
        for (int off = 64; off > 0; off >>= 1) {
            if (tid < off) {
                #pragma unroll
                for (int g = 0; g < G; g++) red[g][tid] += red[g][tid + off];
            }
            it.barrier(sycl::access::fence_space::local_space);
        }
        #pragma unroll
        for (int g = 0; g < G; g++) {
            const float corr = sycl::native::exp(m[g] - mn[g]);
            l[g] = l[g] * corr + red[g][0];
            acc0[g] *= corr;
            acc1[g] *= corr;
        }
        const int d0 = tid, d1 = tid + 128;   /* cnt: the tile's positions, above */
        for (int j = 0; j < cnt; j++) {
            const int vb = p.v_off + (kbase + pt[j]) * p.hd;
            const float v0 = d0 < p.hd ? cc_kvf(vc, vb + d0) : 0.f, v1 = d1 < p.hd ? cc_kvf(vc, vb + d1) : 0.f;
            #pragma unroll
            for (int g = 0; g < G; g++) {
                const float w = pe[g][j];
                if (d0 < p.hd) acc0[g] += w * v0;
                if (d1 < p.hd) acc1[g] += w * v1;
            }
        }
        #pragma unroll
        for (int g = 0; g < G; g++) m[g] = mn[g];
        it.barrier(sycl::access::fence_space::local_space);
    }
    #pragma unroll
    for (int g = 0; g < G; g++) {
        const float inv = l[g] > 0.f ? 1.f / l[g] : 0.f;
        const int h = h0 + g, ob = p.o_off + s * p.o_row + h * p.hd, gb = p.g_off + s * p.g_row + h * p.g_seg;
        for (int k = 0; k < 2; k++) {
            const int d = tid + 128 * k;
            if (d >= p.hd) break;
            float v = (k == 0 ? acc0[g] : acc1[g]) * inv;
            if (p.has_gate) v *= 1.f / (1.f + sycl::native::exp(-gt[gb + d]));
            o[ob + d] = v;
        }
    }
}
/* cc_attn_gk<8> for plain causal rows (no selection list); a work item reads its own key eight elements a load from the cache into registers, with no staging in local memory; the queries eight a load; a position's eight probabilities in one load; the same sums in the same order as cc_attn_k, so the same bits; hd and k_off multiples of eight. */

/* VP 1 (COLI_CUDA_ATTN_GROUPED_VEC=2; hd and v_off even): a work item's two output dimensions are the
 * ADJACENT ones 2 tid and 2 tid + 1, their values one two-element load a position, where VP 0 takes tid and
 * tid + 128 with a load each. A dimension's sum is the same terms in the same order whichever work item
 * holds it, so the bits are the same. The value loop was 8.4 ms of the kernel's 10.3 at 2 048 rows, and its
 * time the number of loads from the cache (the campaign's C1: 256 a tile a work item against the dot
 * products' 32). */
template <typename KT = float, int VP = 0>
static void cc_attn_g8v(const float *__restrict__ q, const KT *__restrict__ kc,
                        const KT *__restrict__ vc, float *__restrict__ o,
                        const float *__restrict__ gt, CcAttn p) {
    constexpr int G = 8;
    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int kvh = it.get_group(2), s = it.get_group(1),
        tid = it.get_local_id(2), h0 = kvh * G;
    auto &qs = *sycl::ext::oneapi::group_local_memory_for_overwrite<sycl::vec<float, 8>[G][32]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &pe = *sycl::ext::oneapi::group_local_memory_for_overwrite<sycl::vec<float, 8>[128]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &red = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[G][128]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());

    #pragma unroll
    for (int g = 0; g < G; g++) {
        const int qb = p.q_off + s * p.q_row + (h0 + g) * p.q_seg;
        float *qf = reinterpret_cast<float *>(&qs[g][0]);
        for (int d = tid; d < p.hd; d += 128) qf[d] = q[qb + d];
    }

    // No selection list: n becomes pos_base + s + 1
    const int n = p.pos_base + s + 1;

    it.barrier(sycl::access::fence_space::local_space);
    float m[G], l[G], acc0[G], acc1[G];
    #pragma unroll
    for (int g = 0; g < G; g++) { m[g] = -3.0e38f; l[g] = 0.f; acc0[g] = 0.f; acc1[g] = 0.f; }

    const int kbase = kvh * p.cap;
    for (int t0 = 0; t0 < n; t0 += 128) {
        const int i = t0 + tid;
        float sv[G];
        #pragma unroll
        for (int g = 0; g < G; g++) sv[g] = -3.0e38f;

        if (i < n) {
            /* this position's dot over d in order (cc_attn_k's), eight elements of its key a load */
            float a[G];
            #pragma unroll
            for (int g = 0; g < G; g++) a[g] = 0.f;
            const size_t kb = (size_t)p.k_off + (size_t)(kbase + i) * p.hd;
            for (int d0 = 0; d0 < p.hd; d0 += 8) {
                const sycl::vec<float, 8> kv = cc_kv8(kc, kb + d0);
                #pragma unroll
                for (int g = 0; g < G; g++) {
                    const sycl::vec<float, 8> qv = qs[g][d0 >> 3];
                    #pragma unroll
                    for (int e = 0; e < 8; e++) a[g] += qv[e] * kv[e];
                }
            }
            #pragma unroll
            for (int g = 0; g < G; g++) sv[g] = a[g] * p.scale;
        }

        #pragma unroll
        for (int g = 0; g < G; g++) red[g][tid] = sv[g];
        it.barrier(sycl::access::fence_space::local_space);
        for (int off = 64; off > 0; off >>= 1) {
            if (tid < off) {
                #pragma unroll
                for (int g = 0; g < G; g++)
                    red[g][tid] = sycl::fmax(red[g][tid], red[g][tid + off]);
            }
            it.barrier(sycl::access::fence_space::local_space);
        }
        float mn[G];
        #pragma unroll
        for (int g = 0; g < G; g++) mn[g] = sycl::fmax(m[g], red[g][0]);
        it.barrier(sycl::access::fence_space::local_space);
sycl::vec<float, 8> ev;
        #pragma unroll
        for (int g = 0; g < G; g++) {
            const float e = i < n ? sycl::native::exp(sv[g] - mn[g]) : 0.f;
            ev[g] = e;
            red[g][tid] = e;
        }
        pe[tid] = ev;
        it.barrier(sycl::access::fence_space::local_space);
        for (int off = 64; off > 0; off >>= 1) {
            if (tid < off) {
                #pragma unroll
                for (int g = 0; g < G; g++) red[g][tid] += red[g][tid + off];
            }
            it.barrier(sycl::access::fence_space::local_space);
        }
        #pragma unroll
        for (int g = 0; g < G; g++) {
            const float corr = sycl::native::exp(m[g] - mn[g]);
            l[g] = l[g] * corr + red[g][0];
            acc0[g] *= corr;
            acc1[g] *= corr;
        }
        const int d0 = VP ? 2 * tid : tid, d1 = VP ? 2 * tid + 1 : tid + 128;   /* this work item's two output dimensions */

        // Value loop with eight-element probability loads
        const int cnt = sycl::min(128, n - t0);
        for (int j = 0; j < cnt; j++) {
            const int vb = p.v_off + (kbase + t0 + j) * p.hd;
            float v0 = 0.f, v1 = 0.f;
            if (VP) {   /* both in one load: d1 = d0 + 1 < hd with hd even */
                if (d0 < p.hd) { const sycl::float2 vv = cc_kv2(vc, (size_t)vb + d0); v0 = vv[0]; v1 = vv[1]; }
            } else {
                v0 = d0 < p.hd ? cc_kvf(vc, vb + d0) : 0.f; v1 = d1 < p.hd ? cc_kvf(vc, vb + d1) : 0.f;
            }
            const sycl::vec<float, 8> w = pe[j];
            #pragma unroll
            for (int g = 0; g < G; g++) {
                if (d0 < p.hd) acc0[g] += w[g] * v0;
                if (d1 < p.hd) acc1[g] += w[g] * v1;
            }
        }

        #pragma unroll
        for (int g = 0; g < G; g++) m[g] = mn[g];
        it.barrier(sycl::access::fence_space::local_space);
    }
    #pragma unroll
    for (int g = 0; g < G; g++) {
        const float inv = l[g] > 0.f ? 1.f / l[g] : 0.f;
        const int h = h0 + g, ob = p.o_off + s * p.o_row + h * p.hd, gb = p.g_off + s * p.g_row + h * p.g_seg;
        for (int k = 0; k < 2; k++) {
            const int d = VP ? 2 * tid + k : tid + 128 * k;
            if (d >= p.hd) break;
            float v = (k == 0 ? acc0[g] : acc1[g]) * inv;
            if (p.has_gate) v *= 1.f / (1.f + sycl::native::exp(-gt[gb + d]));
            o[ob + d] = v;
        }
    }
}
/* A prompt chunk's attention on the matrix engines, in the tensor-core kernels' place (cc_attn_fk
 * and cc_attn_f2 of cuda_chain.cu: WMMA and mma.sync, which have no SYCL form): coli_sycl.h's
 * coli_sycl_xmx_attn, a joint_matrix kernel where the profile lists the f16 shape, declined (0,
 * nothing submitted) elsewhere -- the rows then take cc_attn_k or cc_attn_gk, as on a device
 * without tensor cores. COLI_CUDA_CHAIN_FLASH=0 never asks. Not bit-identical to cc_attn_k (f16
 * operands). The _ok twin answers without submitting, which is what the decode split keys on. */
static int cc_attn_xmx_ok(const CcCtx *c, const CcAttn *p, int G, int kv_f16, int sel) {
    return coli_sycl_xmx_attn_ok(coli_sycl_profile_for(c->device), p->hd, G, kv_f16, sel);
}
template <typename KT>
static int cc_attn_xmx(CcCtx *c, CcBuf *q, CcBuf *kc, CcBuf *vc, CcBuf *o, CcBuf *gate, const CcAttn *p, int G,
                       int win, const float *sk, int so, int ring, const int *sel) {
    CC_OWN(cur(), "cc_attn_xmx", q, kc, vc, o, gate);
    coli_sycl_attn a;
    a.q = fp(q); a.kc = kvp<KT>(kc); a.vc = kvp<KT>(vc); a.kv_f16 = sizeof(KT) == 2; a.o = fpw(o); a.gate = fp(gate);
    a.p = *p; a.G = G; a.win = win; a.snk = sk; a.sink_off = so; a.ring = ring; a.sel = sel;
    int r = 0;
    if (!cc_try(c, "attn (matrix engines)", [&] { r = coli_sycl_xmx_attn(coli_sycl_profile_for(c->device), *c->stream, a); })) return 0;
    return r;
}
/* whether a decode's rows split over their positions (cc_attn_dec_k), COLI_CUDA_ATTN_DEC: 0 never;
 * 1 only where a prompt chunk takes the matrix engines (the CUDA rule: elsewhere a row's bits do
 * not depend on how a forward is cut into steps, and a prefix resumed answers as the prompt
 * recomputed does); else always, the default here (a block a (head, row) walking every position
 * leaves the card idle, and the engines' attention is not there on every part) */
static int cc_attn_dec_on(int xmx) {
    static int dec = -1;
    if (dec < 0) { const char *e = getenv("COLI_CUDA_ATTN_DEC"); dec = e && (*e == '0' || *e == '1') ? *e - '0' : 2; }
    return dec == 2 || (dec == 1 && xmx);
}
/* one thread per channel, the rows in order, the channel's ring in registers */
static void cc_dnconv_k(const float *__restrict__ xin, const float *__restrict__ w, float *__restrict__ ring,
                                   float *__restrict__ o, float *__restrict__ snap, CcDnConv p) {
    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    long long c =
        (long long)it.get_group(2) * it.get_local_range(2) +
        it.get_local_id(2);
    if (c >= p.CD) return;
    int nh = p.CK - 1;
    float hist[8];
    for (int k = 0; k < 8; k++) hist[k] = k < nh ? ring[p.ring_off + c * nh + k] : 0.f;
    int wb = p.w_off + (int)c * p.CK;
    float wl = w[wb + p.CK - 1];
    for (int s = 0; s < p.S; s++) {
        long long ib = p.in_off + (long long)s * p.in_row;   /* order 2 (LFM3): the row is B | C | x */
        float cur = p.order == 2 ? xin[ib + c] * xin[ib + c + 2LL * p.CD] : xin[ib + c];
        float acc;
        if (p.order != 1) {
            acc = 0.f;
            for (int k = 0; k < 8; k++) if (k < nh) acc += w[wb + k] * hist[k];
            acc += wl * cur;
        } else {
            acc = wl * cur;
            for (int k = 0; k < 8; k++) if (k < nh) acc += w[wb + k] * hist[k];
        }
        o[p.out_off + s * p.out_row + c] =
            p.order == 2 ? xin[ib + c + p.CD] * acc
                         : acc / (1.f + sycl::native::exp(-acc));
        for (int k = 0; k < 7; k++) if (k + 1 < nh) hist[k] = hist[k + 1];
        for (int k = 0; k < 8; k++) if (k == nh - 1) hist[k] = cur;
        if (s == p.snap_row)
            for (int k = 0; k < 8; k++) if (k < nh) snap[p.snap_off + c * nh + k] = hist[k];
    }
    for (int k = 0; k < 8; k++) if (k < nh) ring[p.ring_off + c * nh + k] = hist[k];
}
/* a prompt's rows apart: one thread per row and channel, the channel's rows before the chunk from
 * its ring (the same sums in the same order as cc_dnconv_k's); cc_dnconv_ring_k then moves the
 * ring and the snapshot past the chunk. One thread per channel took the chunk's rows one after
 * another: LFM3's 2048 channels were 32 blocks, 0.23 ms a 512-row layer on an RTX PRO 4000. */
static float cc_dnconv_val(const float *__restrict__ xin, const float *__restrict__ ring, long long c, int r, int nh, const CcDnConv &p) {
    if (r < 0) return ring[p.ring_off + c * nh + nh + r];   /* the ring's last nh, the newest last */
    long long ib = p.in_off + (long long)r * p.in_row;
    return p.order == 2 ? xin[ib + c] * xin[ib + c + 2LL * p.CD] : xin[ib + c];
}
static void cc_dnconv_rows_k(const float *__restrict__ xin, const float *__restrict__ w, const float *__restrict__ ring,
                                        float *__restrict__ o, CcDnConv p) {
    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    long long c =
        (long long)it.get_group(2) * it.get_local_range(2) +
        it.get_local_id(2);
    int s = it.get_group(1);
    if (c >= p.CD) return;
    int nh = p.CK - 1, wb = p.w_off + (int)c * p.CK;
    float cur = cc_dnconv_val(xin, ring, c, s, nh, p), wl = w[wb + p.CK - 1], acc;
    if (p.order != 1) {
        acc = 0.f;
        for (int k = 0; k < 8; k++) if (k < nh) acc += w[wb + k] * cc_dnconv_val(xin, ring, c, s - nh + k, nh, p);
        acc += wl * cur;
    } else {
        acc = wl * cur;
        for (int k = 0; k < 8; k++) if (k < nh) acc += w[wb + k] * cc_dnconv_val(xin, ring, c, s - nh + k, nh, p);
    }
    long long ib = p.in_off + (long long)s * p.in_row;
    o[p.out_off + (long long)s * p.out_row + c] =
        p.order == 2 ? xin[ib + c + p.CD] * acc
                     : acc / (1.f + sycl::native::exp(-acc));
}
static void cc_dnconv_ring_k(const float *__restrict__ xin, float *__restrict__ ring, float *__restrict__ snap, CcDnConv p) {
    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    long long c =
        (long long)it.get_group(2) * it.get_local_range(2) +
        it.get_local_id(2);
    if (c >= p.CD) return;
    int nh = p.CK - 1;
    float h[8];
    if (p.snap_row >= 0 && p.snap_row < p.S)   /* the ring after row snap_row */
        for (int k = 0; k < 8; k++) if (k < nh) snap[p.snap_off + c * nh + k] = cc_dnconv_val(xin, ring, c, p.snap_row + 1 - nh + k, nh, p);
    for (int k = 0; k < 8; k++) if (k < nh) h[k] = cc_dnconv_val(xin, ring, c, p.S - nh + k, nh, p);
    for (int k = 0; k < 8; k++) if (k < nh) ring[p.ring_off + c * nh + k] = h[k];
}
static float cc_softplus(float x) {
    if (x > 20.f) return x;
    float e = sycl::native::exp(x);
    return e < 1e-4f ? e - 0.5f * e * e : sycl::log(1.f + e);
}
/* one block of 128 per value head, thread v's column of the state in registers */
template <int KD>
static void cc_dnrec_k(const float *__restrict__ cv,
                       const float *__restrict__ ab,
                       const float *__restrict__ z, float *__restrict__ st,
                       const float *__restrict__ prm, float *__restrict__ y,
                       float *__restrict__ snap, CcDnRec p) {
    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto &qn = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[KD]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &kn = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[KD]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &red =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[128]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &red2 =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[128]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    int h = it.get_group(2), v = it.get_local_id(2);
    bool own = v < p.VD;
    int kh = (p.flags & 4) ? h % p.KH : h / (p.VH / p.KH);   /* flags 4: a GGUF's value heads, tiled */
    float col[KD];
    int sb = p.st_off + h * KD * p.VD, nb = p.snap_off + h * KD * p.VD;
    #pragma unroll
    for (int k = 0; k < KD; k++) col[k] = own ? st[sb + k * p.VD + v] : 0.f;
    float alog = prm[p.prm_off + h], dtb = prm[p.prm_off + p.VH + h];
    float nw = own ? prm[p.prm_off + 2 * p.VH + v] : 0.f;
    for (int s = 0; s < p.S; s++) {
        int cb = p.cv_off + s * p.cv_row;
        float qa = 0.f, ka = 0.f;
        for (int k = v; k < KD; k += 128) {
            float qv = cv[cb + kh * KD + k], kv = cv[cb + p.Ktot + kh * KD + k];
            qn[k] = qv; kn[k] = kv;
            qa += qv * qv; ka += kv * kv;
        }
        red[v] = qa; red2[v] = ka;
        it.barrier();
        for (int off = 64; off > 0; off >>= 1) {
            if (v < off) { red[v] += red[v + off]; red2[v] += red2[v + off]; }
            it.barrier();
        }
        float qsc = 1.f / sycl::sqrt(red[0] + 1e-6f) * p.qscale;
        float ksc = 1.f / sycl::sqrt(red2[0] + 1e-6f);
        it.barrier();
        for (int k = v; k < KD; k += 128) { qn[k] *= qsc; kn[k] *= ksc; }
        it.barrier();
        float bv = ab[p.b_off + s * p.b_row + h], av = ab[p.a_off + s * p.a_row + h];
        float beta = 1.f / (1.f + sycl::native::exp(-bv));
        float decay = sycl::native::exp(
            ((p.flags & 2) ? alog : -sycl::native::exp(alog)) *
            cc_softplus(av + dtb)); /* flags 2: a GGUF's A itself */
        float o = 0.f;
        if (own) {
            float vin = cv[cb + 2 * p.Ktot + h * p.VD + v];
            float kvs = 0.f;
            #pragma unroll
            for (int k = 0; k < KD; k++) { col[k] *= decay; kvs += kn[k] * col[k]; }
            float delta = (vin - kvs) * beta;
            #pragma unroll
            for (int k = 0; k < KD; k++) col[k] += kn[k] * delta;
            #pragma unroll
            for (int k = 0; k < KD; k++) o += qn[k] * col[k];
        }
        red[v] = o * o;
        it.barrier();
        for (int off = 64; off > 0; off >>= 1) {
            if (v < off) red[v] += red[v + off];
            it.barrier();
        }
        float r = 1.f / sycl::sqrt(red[0] / (float)p.VD + p.eps);
        if (own) {
            float zv = z[p.z_off + s * p.z_row + h * p.VD + v];
            float g = (p.flags & 1) ? 1.f / (1.f + sycl::native::exp(-zv))
                                    : zv / (1.f + sycl::native::exp(-zv));
            y[p.y_off + s * p.y_row + h * p.VD + v] = o * r * nw * g;
            if (s == p.snap_row) {
                #pragma unroll
                for (int k = 0; k < KD; k++) snap[nb + k * p.VD + v] = col[k];
            }
        }
        it.barrier();
    }
    if (own) {
        #pragma unroll
        for (int k = 0; k < KD; k++) st[sb + k * p.VD + v] = col[k];
    }
}
#define CC_DNREC_ROWS 8   /* rows from which cc_dnrec takes cc_dnrec_rows_k (COLI_CUDA_DNREC_ROWS=0 never) */
/* A prompt chunk's rows (cc_dnrec from CC_DNREC_ROWS rows): cc_dnrec_k's arithmetic, R
 * rows at a time. cc_dnrec_k takes about eighteen barriers a row (the tree sums of the q, k
 * and output norms); here, per block: the q and k norms, beta, decay and v in parallel (a
 * warp a vector), the rows in order with no barrier (each thread's column in registers,
 * its dots over the block's shared q and k), then each row's gated RMSNorm (a warp a row):
 * three barriers a block. The norms' sums run in a warp's order, the state's as before. SG: the
 * warp's lanes, 128 / SG warps a block. */
template <int KD, int R, int SG>
static void
cc_dnrec_rows_k(const float *__restrict__ cv, const float *__restrict__ ab,
                const float *__restrict__ z, float *__restrict__ st,
                const float *__restrict__ prm, float *__restrict__ y,
                float *__restrict__ snap, CcDnRec p) {
    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto &qs =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[R][KD]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &ks =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[R][KD]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &vs =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[R][128]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &os =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[R][128]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &bs = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[R]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &ds = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[R]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    constexpr int W = 128 / SG;   /* the block's warps */
    int h = it.get_group(2), v = it.get_local_id(2), wp = v / SG, lane = v & (SG - 1);
    bool own = v < p.VD;
    int kh = (p.flags & 4) ? h % p.KH : h / (p.VH / p.KH);   /* flags 4: a GGUF's value heads, tiled */
    float col[KD];
    int sb = p.st_off + h * KD * p.VD, nb = p.snap_off + h * KD * p.VD;
    #pragma unroll
    for (int k = 0; k < KD; k++) col[k] = own ? st[sb + k * p.VD + v] : 0.f;
    float alog = prm[p.prm_off + h], dtb = prm[p.prm_off + p.VH + h];
    for (int s0 = 0; s0 < p.S; s0 += R) {
        int nr = sycl::min(R, p.S - s0);
        for (int w = wp; w < 2 * nr; w += W) {
            int r = w >> 1, isk = w & 1, cb = p.cv_off + (s0 + r) * p.cv_row + isk * p.Ktot + kh * KD;
            float a = 0.f;
            for (int k = lane; k < KD; k += SG) { float x = cv[cb + k]; a += x * x; }
            #pragma unroll
            for (int m = SG / 2; m > 0; m >>= 1) a += cc_shfl_xor(a, m);
            float sc = 1.f / sycl::sqrt(a + 1e-6f) * (isk ? 1.f : p.qscale);
            for (int k = lane; k < KD; k += SG) { float x = cv[cb + k] * sc; if (isk) ks[r][k] = x; else qs[r][k] = x; }
        }
        for (int i = v; i < nr * p.VD; i += 128) {
            int r = i / p.VD, c = i - r * p.VD;
            vs[r][c] = cv[p.cv_off + (s0 + r) * p.cv_row + 2 * p.Ktot + h * p.VD + c];
        }
        if (v < nr) {
            int s = s0 + v;
            float bv = ab[p.b_off + s * p.b_row + h], av = ab[p.a_off + s * p.a_row + h];
            bs[v] = 1.f / (1.f + sycl::native::exp(-bv));
            ds[v] = sycl::native::exp(
                ((p.flags & 2) ? alog : -sycl::native::exp(alog)) *
                cc_softplus(av + dtb));
        }
        it.barrier();
        if (own)
            for (int r = 0; r < nr; r++) {
                float decay = ds[r], kvs = 0.f, o = 0.f;
                #pragma unroll
                for (int k = 0; k < KD; k++) { col[k] *= decay; kvs += ks[r][k] * col[k]; }
                float delta = (vs[r][v] - kvs) * bs[r];
                #pragma unroll
                for (int k = 0; k < KD; k++) col[k] += ks[r][k] * delta;
                #pragma unroll
                for (int k = 0; k < KD; k++) o += qs[r][k] * col[k];
                os[r][v] = o;
                if (s0 + r == p.snap_row) {
                    #pragma unroll
                    for (int k = 0; k < KD; k++) snap[nb + k * p.VD + v] = col[k];
                }
            }
        it.barrier();
        for (int r = wp; r < nr; r += W) {
            int s = s0 + r;
            float oo = 0.f;
            for (int c = lane; c < p.VD; c += SG) oo += os[r][c] * os[r][c];
            #pragma unroll
            for (int m = SG / 2; m > 0; m >>= 1) oo += cc_shfl_xor(oo, m);
            float rr = 1.f / sycl::sqrt(oo / (float)p.VD + p.eps);
            for (int c = lane; c < p.VD; c += SG) {
                float zv = z[p.z_off + s * p.z_row + h * p.VD + c];
                float g = (p.flags & 1) ? 1.f / (1.f + sycl::native::exp(-zv))
                                        : zv / (1.f + sycl::native::exp(-zv));
                y[p.y_off + s * p.y_row + h * p.VD + c] = os[r][c] * rr * prm[p.prm_off + 2 * p.VH + c] * g;
            }
        }
        it.barrier();
    }
    if (own) {
        #pragma unroll
        for (int k = 0; k < KD; k++) st[sb + k * p.VD + v] = col[k];
    }
}
/* cc_dnrec_split_k's inputs for every row at once, the device full: each row's q and k of every key
 * head l2-normalised (q times qscale) into qk[row][q, k][KH][KD], a warp a vector (cc_dnrec_rows_k's
 * sums), and each (row, head)'s beta and decay into bd[row][VH][2]. nvb: the vectors' blocks, 128 / SG
 * vectors each */
template <int SG>
static void cc_dnprep_k(const float *__restrict__ cv, const float *__restrict__ ab,
        const float *__restrict__ prm, float *__restrict__ qk, float *__restrict__ bd, int KD, int nvb, CcDnRec p) {
    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int lane = it.get_local_id(2) & (SG - 1);
    if ((int)it.get_group(2) < nvb) {
        const int g = it.get_group(2) * (128 / SG) + it.get_local_id(2) / SG;
        if (g >= p.S * 2 * p.KH) return;
        const int s = g / (2 * p.KH), rem = g - s * 2 * p.KH, isk = rem / p.KH, kh = rem - isk * p.KH;
        const int cb = p.cv_off + s * p.cv_row + isk * p.Ktot + kh * KD;
        float a = 0.f;
        for (int k = lane; k < KD; k += SG) { float x = cv[cb + k]; a += x * x; }
        #pragma unroll
        for (int m = SG / 2; m > 0; m >>= 1) a += cc_shfl_xor(a, m);
        const float sc = 1.f / sycl::sqrt(a + 1e-6f) * (isk ? 1.f : p.qscale);
        float *o = qk + (size_t)g * KD;
        for (int k = lane; k < KD; k += SG) o[k] = cv[cb + k] * sc;
        return;
    }
    const int i =
        ((int)it.get_group(2) - nvb) * 128 + it.get_local_id(2);
    if (i >= p.S * p.VH) return;
    const int s = i / p.VH, h = i - s * p.VH;
    const float alog = prm[p.prm_off + h], dtb = prm[p.prm_off + p.VH + h];
    const float bv = ab[p.b_off + s * p.b_row + h], av = ab[p.a_off + s * p.a_row + h];
    bd[2 * i] = 1.f / (1.f + sycl::native::exp(-bv));
    bd[2 * i + 1] =
        sycl::native::exp(((p.flags & 2) ? alog : -sycl::native::exp(alog)) *
                          cc_softplus(av + dtb));
}
/* A prompt chunk's rows with the state spread over the device and in registers: cc_dnrec_rows_k gives
 * a value head one block of 128 (a column of the state a thread, every row's KD products in turn
 * against q and k in shared memory, and one block on each of 32 multiprocessors). Here P threads
 * share C columns, each the KD / P products k = P i + part of the C columns: one read of k[i] (or
 * q[i]) serves C products, and the parts join by shuffles that leave every lane the same sum. A block
 * of 128 threads takes 128 / P x C of a head's columns: VH x VD / (128 / P x C) blocks. The rows'
 * q, k, beta and decay come from cc_dnprep_k, R rows at a time, the next R loaded into registers
 * while these R run. The rows' outputs go to y as they are; cc_dnnorm_k takes each head's gated
 * RMSNorm after (it needs the head's every column). */
template <int KD, int R, int P, int C>
static void
cc_dnrec_split_k(const float *__restrict__ cv, const float *__restrict__ qk,
                 const float *__restrict__ bd, float *__restrict__ st,
                 float *__restrict__ y, float *__restrict__ snap, CcDnRec p) {
    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr int KQ = KD / P, CB = 128 / P * C, QF = 2 * R * KD / 4 / 128, VF = R * CB / 4 / 128 > 0 ? R * CB / 4 / 128 : 1;
    static_assert(2 * R * KD % 512 == 0, "q and k: whole float4s a thread");
    auto &qk_s = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[2][R][2][KD]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>()); /* [buffer][row][q,
                                                                    k][KD] */
    auto &vs =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[2][R][CB]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &bs =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[2][R]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &ds =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[2][R]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int h = it.get_group(2), c0 = it.get_group(1) * CB,
              t = it.get_local_id(2);
    const int grp = t / P, qt = t % P, v0 = c0 + grp * C;   /* the thread's columns v0 .. v0 + C - 1 and part */
    const int kh = (p.flags & 4) ? h % p.KH : h / (p.VH / p.KH);
    const int sb = p.st_off + h * KD * p.VD, nb = p.snap_off + h * KD * p.VD;
    sycl::float4 rq[QF], rv[VF]; float rb = 0.f, rd = 0.f;
    /* the R rows from s0 into registers (rows past S as zeros) */
    auto fetch = [&](int s0) {
        #pragma unroll
        for (int j = 0; j < QF; j++) {
            const int f = t + 128 * j, r = f / (KD / 2), w = f % (KD / 2), isk = w / (KD / 4), k4 = w % (KD / 4);
            rq[j] = s0 + r < p.S
                        ? reinterpret_cast<const sycl::float4 *>(
                              qk + (((size_t)(s0 + r) * 2 + isk) * p.KH + kh) *
                                       KD)[k4]
                        : sycl::float4(0.f, 0.f, 0.f, 0.f);
        }
        #pragma unroll
        for (int j = 0; j < VF; j++) {
            const int f = t + 128 * j, r = f / (CB / 4), c4 = f % (CB / 4);
            sycl::float4 z = sycl::float4(0.f, 0.f, 0.f, 0.f);
            if (f < R * CB / 4 && s0 + r < p.S) {
                const float *src = cv + p.cv_off + (size_t)(s0 + r) * p.cv_row + 2 * p.Ktot + h * p.VD + c0 + 4 * c4;
                if (c0 + 4 * c4 + 3 < p.VD) z =
                    sycl::float4(src[0], src[1], src[2], src[3]);
                else {
                    float e[4] = {0.f, 0.f, 0.f, 0.f};
                    for (int u = 0; u < 4; u++) if (c0 + 4 * c4 + u < p.VD)
                        e[u] = src[u];
                    z = sycl::float4(e[0], e[1], e[2], e[3]);
                }
            }
            rv[j] = z;
        }
        if (t < R && s0 + t < p.S) { rb = bd[2 * ((size_t)(s0 + t) * p.VH + h)]; rd = bd[2 * ((size_t)(s0 + t) * p.VH + h) + 1]; }
    };
    auto store = [&](int b) {
        #pragma unroll
        for (int j = 0; j < QF; j++) {
            const int f = t + 128 * j, r = f / (KD / 2), w = f % (KD / 2), isk = w / (KD / 4), k4 = w % (KD / 4);
            reinterpret_cast<sycl::float4 *>(qk_s[b][r][isk])[k4] = rq[j];
        }
        #pragma unroll
        for (int j = 0; j < VF; j++) {
            const int f = t + 128 * j, r = f / (CB / 4), c4 = f % (CB / 4);
            if (f < R * CB / 4) reinterpret_cast<sycl::float4 *>(vs[b][r])[c4] =
                rv[j];
        }
        if (t < R) { bs[b][t] = rb; ds[b][t] = rd; }
    };
    float col[C][KQ];
    #pragma unroll
    for (int c = 0; c < C; c++)
        #pragma unroll
        for (int i = 0; i < KQ; i++) col[c][i] = v0 + c < p.VD ? st[sb + (P * i + qt) * p.VD + v0 + c] : 0.f;
    fetch(0); store(0);
    it.barrier();
    for (int s0 = 0, b = 0; s0 < p.S; s0 += R, b ^= 1) {
        const int nr = sycl::min(R, p.S - s0);
        if (s0 + R < p.S) fetch(s0 + R);   /* in flight while these rows run */
        for (int r = 0; r < nr; r++) {
            const float decay = ds[b][r], beta = bs[b][r];
            const float *qs = qk_s[b][r][0], *ks = qk_s[b][r][1];
            float kv[C], o[C];
            #pragma unroll
            for (int c = 0; c < C; c++) kv[c] = 0.f;
            #pragma unroll
            for (int i = 0; i < KQ; i++) {
                const float kk = ks[P * i + qt];
                #pragma unroll
                for (int c = 0; c < C; c++) { col[c][i] *= decay; kv[c] += kk * col[c][i]; }
            }
            #pragma unroll
            for (int c = 0; c < C; c++) {
                #pragma unroll
                for (int m = 1; m < P; m <<= 1) kv[c] += cc_shfl_xor(kv[c], m);
                kv[c] = (vs[b][r][grp * C + c] - kv[c]) * beta;   /* the column's delta */
                o[c] = 0.f;
            }
            #pragma unroll
            for (int i = 0; i < KQ; i++) {
                const float kk = ks[P * i + qt], qq = qs[P * i + qt];
                #pragma unroll
                for (int c = 0; c < C; c++) { col[c][i] += kk * kv[c]; o[c] += qq * col[c][i]; }
            }
            #pragma unroll
            for (int c = 0; c < C; c++) {
                #pragma unroll
                for (int m = 1; m < P; m <<= 1) o[c] += cc_shfl_xor(o[c], m);
                if (qt == 0 && v0 + c < p.VD) y[p.y_off + (s0 + r) * p.y_row + h * p.VD + v0 + c] = o[c];
            }
            if (s0 + r == p.snap_row)
                #pragma unroll
                for (int c = 0; c < C; c++)
                    #pragma unroll
                    for (int i = 0; i < KQ; i++) if (v0 + c < p.VD) snap[nb + (P * i + qt) * p.VD + v0 + c] = col[c][i];
        }
        if (s0 + R < p.S) store(b ^ 1);   /* the other buffer: nobody reads it this round */
        it.barrier();
    }
    #pragma unroll
    for (int c = 0; c < C; c++)
        #pragma unroll
        for (int i = 0; i < KQ; i++) if (v0 + c < p.VD) st[sb + (P * i + qt) * p.VD + v0 + c] = col[c][i];
}
/* cc_dnrec_split_k's rows: each (row, head)'s gated RMSNorm in place, a warp each (128 / SG a block),
 * cc_dnrec_rows_k's sums */
template <int SG>
static void cc_dnnorm_k(const float *__restrict__ z, const float *__restrict__ prm,
                                                           float *__restrict__ y, CcDnRec p) {
    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int g = it.get_group(2) * (128 / SG) + it.get_local_id(2) / SG, lane = it.get_local_id(2) & (SG - 1);
    if (g >= p.S * p.VH) return;
    const int s = g / p.VH, h = g - s * p.VH;
    float *yr = y + p.y_off + s * p.y_row + h * p.VD;
    float oo = 0.f;
    for (int c = lane; c < p.VD; c += SG) oo += yr[c] * yr[c];
    #pragma unroll
    for (int m = SG / 2; m > 0; m >>= 1) oo += cc_shfl_xor(oo, m);
    const float rr = 1.f / sycl::sqrt(oo / (float)p.VD + p.eps);
    for (int c = lane; c < p.VD; c += SG) {
        float zv = z[p.z_off + s * p.z_row + h * p.VD + c];
        float gt = (p.flags & 1) ? 1.f / (1.f + sycl::native::exp(-zv))
                                 : zv / (1.f + sycl::native::exp(-zv));
        yr[c] = yr[c] * rr * prm[p.prm_off + 2 * p.VH + c] * gt;
    }
}
static float cc_sig(float x) { return 1.f / (1.f + sycl::native::exp(-x)); }
/* a score's order as an unsigned key: a larger float, a larger key (equal floats, equal keys but
 * for -0 and +0, and a selection's scores are sums of positive products: never -0) */
static inline unsigned cc_qsa_key(float v) {
    unsigned u = sycl::bit_cast<unsigned int>(v);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
/* x's exclusive prefix over a block of 128 threads in buf (128 ints), *tot their sum */
static int cc_scan128(int x, int *buf, int *tot) {
    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int tid = it.get_local_id(2);
    buf[tid] = x;
    it.barrier();
    for (int o = 1; o < 128; o <<= 1) {
        const int v = tid >= o ? buf[tid - o] : 0;
        it.barrier();
        buf[tid] += v;
        it.barrier();
    }
    const int incl = buf[tid];
    *tot = buf[127];
    it.barrier();
    return incl - x;
}
/* mode 1's scores, cc_qsa's first launch: a thread a block of keys of a row, over a grid of the rows' blocks
 * (x) and the rows (y), the dot over d in the CPU's order. One block of 128 threads a row scored its 1100 blocks
 * of 4400 positions in 197 us (Qwen3.8-Flash-Next on an RTX PRO 4000, a decode step's 1-4 rows: as many SMs) */
static void cc_qsa_score_k(const float *__restrict__ src, const float *__restrict__ pk,
                                                             float *__restrict__ sc, CcQsa p) {
    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int s = it.get_group(1),
              b = it.get_group(2) * 128 + it.get_local_id(2);
    const int visible = p.pos_base + s + 1, blocks = visible / p.R,
              take = sycl::min(blocks, p.budget / p.R);
    if (take >= blocks || b >= blocks) return;
    const int qb = p.q_off + s * p.q_row;
    float score = 0.f;
    for (int h = 0; h < p.IQ; h++) {
        float a = 0.f;
        for (int d = 0; d < p.ID; d++) a += src[qb + h * p.ID + d] * pk[p.pk_off + b * p.ID + d];
        if (a > 0.f) score += a;
    }
    sc[s * 2 * p.nbmax + b] = score / sycl::sqrt((float)p.ID);
}
/* cc_qsa_score_k's scores for a prompt chunk's rows in tiles: 64 rows by 64 blocks of keys a block
 * of 256 threads, each thread 4 rows by 4 blocks, the rows' index queries and the blocks' keys staged
 * 32 dims at a time. Each score is cc_qsa_score_k's: a head's dot over d in order from 0, its
 * positive part added head by head, so the same bits and the same selections. cc_qsa_score_k's
 * thread a (row, block) read its block's 128 dims alone, a chunk of 2048 rows 2 ms a layer on an
 * RTX PRO 4000. Grid: x = the rows' most blocks / 64, y = rows / 64. */
#define CC_QST 64
static void cc_qsa_score_t(const float *__restrict__ src,
                           const float *__restrict__ pk, float *__restrict__ sc,
                           CcQsa p) {
    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto &qs = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[32][CC_QST + 1]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &ks = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[32][CC_QST + 1]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>()); /* d by row, d
                                                                    by block */
    const int tid = it.get_local_id(2), tr = tid >> 4,
              tb = tid & 15; /* the thread's rows tr + 16 i, blocks tb + 16 j */
    const int s0 = it.get_group(1) * CC_QST,
              b0 = it.get_group(2) * CC_QST;
    float score[4][4];
    #pragma unroll
    for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) score[i][j] = 0.f;
    for (int h = 0; h < p.IQ; h++) {
        float a[4][4];
        #pragma unroll
        for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) a[i][j] = 0.f;
        for (int d0 = 0; d0 < p.ID; d0 += 32) {
            for (int e = tid; e < 32 * CC_QST; e += 256) {
                const int r = e >> 5, d = e & 31, s = s0 + r, b = b0 + r, dd = d0 + d;
                qs[d][r] = s < p.S && dd < p.ID ? src[p.q_off + s * p.q_row + h * p.ID + dd] : 0.f;
                ks[d][r] = dd < p.ID ? pk[p.pk_off + (size_t)b * p.ID + dd] : 0.f;
            }
            it.barrier(sycl::access::fence_space::local_space);
            const int dn = p.ID - d0 < 32 ? p.ID - d0 : 32;
            for (int d = 0; d < dn; d++) {
                float qv[4], kv[4];
                #pragma unroll
                for (int i = 0; i < 4; i++) { qv[i] = qs[d][tr + 16 * i]; kv[i] = ks[d][tb + 16 * i]; }
                #pragma unroll
                for (int i = 0; i < 4; i++)
                    #pragma unroll
                    for (int j = 0; j < 4; j++) a[i][j] += qv[i] * kv[j];
            }
            it.barrier(sycl::access::fence_space::local_space);
        }
        #pragma unroll
        for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) if (a[i][j] > 0.f) score[i][j] += a[i][j];
    }
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        const int s = s0 + tr + 16 * i;
        if (s >= p.S) continue;
        const int visible = p.pos_base + s + 1, blocks = visible / p.R,
                  take = sycl::min(blocks, p.budget / p.R);
        if (take >= blocks) continue;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            const int b = b0 + tb + 16 * j;
            if (b < blocks) sc[s * 2 * p.nbmax + b] =
                score[i][j] / sycl::sqrt((float)p.ID);
        }
    }
}
/* chain_qsa.comp: mode 0 one block of 128 per block key; mode 1 one block per query row (its scores
 * cc_qsa_score_k's) */
static void cc_qsa_k(const float *__restrict__ src, const float *__restrict__ w,
                     float *__restrict__ pk, const float *__restrict__ cs,
                     float *__restrict__ sc, int *__restrict__ sel, CcQsa p) {
    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto &pool =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[256]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &red =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[128]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    int tid = it.get_local_id(2);
    if (p.mode == 0) {
        int b = p.b0 + it.get_group(2);
        float ss = 0.f;
        for (int d = tid; d < p.ID; d += 128) {
            float v = 0.f;
            for (int r = 0; r < p.R; r++) v += src[p.src_off + (b * p.R + r) * p.ID + d] / (float)p.R;
            pool[d] = v;
            ss += v * v;
        }
        red[tid] = ss;
        it.barrier(sycl::access::fence_space::local_space);
        for (int off = 64; off > 0; off >>= 1) {
            if (tid < off) red[tid] += red[tid + off];
            it.barrier(sycl::access::fence_space::local_space);
        }
        float rr = 1.f / sycl::sqrt(red[0] / (float)p.ID + p.eps);
        it.barrier(sycl::access::fence_space::local_space);
        for (int d = tid; d < p.ID; d += 128) pool[d] = pool[d] * rr * (1.f + w[p.w_off + d]);
        it.barrier(sycl::access::fence_space::local_space);
        int cb = it.get_group(2) * 2 * p.half_;
        for (int d = tid; d < p.ID; d += 128) {
            float v = pool[d];
            if (d < p.half_) {
                float c = cs[cb + 2 * d], sn = cs[cb + 2 * d + 1];
                v = pool[d] * c - pool[d + p.half_] * sn;
            } else if (d < 2 * p.half_) {
                int j = d - p.half_;
                float c = cs[cb + 2 * j], sn = cs[cb + 2 * j + 1];
                v = pool[d] * c + pool[j] * sn;
            }
            pk[p.pk_off + b * p.ID + d] = v;
        }
        return;
    }
    int s = it.get_group(2);
    int visible = p.pos_base + s + 1, blocks = visible / p.R;
    int take = sycl::min(blocks, p.budget / p.R);
    int lb = s * p.sel_row;
    if (take >= blocks) {
        if (tid == 0) sel[lb] = -1;
        return;
    }
    int sb = s * 2 * p.nbmax;
    /* The blocks of rank < take (a higher score first, of equal ones the lower block), as the CPU
     * ranks them. Each block counted its rank against every other and one thread wrote the list:
     * O(blocks^2) on 128 threads, 298 us a call at 4400 positions (1100 blocks, Qwen3.8-Flash-Next
     * on an RTX PRO 4000), 7 ms of a 31 ms decode step over its 13 QSA layers, and four times
     * that a doubling of the context. Here the take-th score's key by a radix select over its
     * four bytes, then, in the blocks' order, every block above it and of those equal to it the
     * first by index, their positions written by prefix sums: the same list. */
    auto &hist =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<unsigned[256]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_pre =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<unsigned>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_rem = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_eq = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_out = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &sbuf = *sycl::ext::oneapi::group_local_memory_for_overwrite<int[128]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    unsigned pre = 0u, msk = 0u;
    int rem = take;
    for (int sh = 24; sh >= 0; sh -= 8) {
        for (int i = tid; i < 256; i += 128) hist[i] = 0u;
        it.barrier();
        for (int b = tid; b < blocks; b += 128) {
            const unsigned k = cc_qsa_key(sc[sb + b]);
            if ((k & msk) == pre) cc_local_add(hist[(k >> sh) & 255u], 1u);
        }
        it.barrier();
        if (tid == 0) {   /* the byte where the count from the top reaches rem */
            int acc = 0, d = 255;
            for (; d > 0 && acc + (int)hist[d] < rem; d--) acc += (int)hist[d];
            s_pre = pre | ((unsigned)d << sh); s_rem = rem - acc;
        }
        it.barrier();
        pre = s_pre; rem = s_rem; msk |= 255u << sh;
        it.barrier();
    }
    if (tid == 0) { s_eq = 0; s_out = 0; }   /* pre: the take-th key; rem: how many of that key to take */
    it.barrier();
    for (int base = 0; base < blocks; base += 128) {
        const int b = base + tid;
        const unsigned k = b < blocks ? cc_qsa_key(sc[sb + b]) : 0u;
        const int eq = b < blocks && k == pre;
        int te, to;
        const int eb = cc_scan128(eq, sbuf, &te);
        const int tk = b < blocks && (k > pre || (eq && s_eq + eb < rem));
        const int ob = cc_scan128(tk, sbuf, &to);
        const int o0 = s_out;
        if (tk) for (int r = 0; r < p.R; r++) sel[lb + 1 + o0 + ob * p.R + r] = b * p.R + r;
        it.barrier();
        if (tid == 0) { s_eq += te; s_out += to * p.R; }
        it.barrier();
    }
    const int n0 = s_out, t0 = blocks * p.R;
    for (int t = t0 + tid; t < visible; t += 128) sel[lb + 1 + n0 + t - t0] = t;
    if (tid == 0) sel[lb] = n0 + visible - t0;
}
/* chain_ple.comp: mode 0 one block of 256 per (row, stream); mode 1 one thread per channel */
static void cc_ple_reduce2(float *r1, float *r2, int tid) {
    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    it.barrier();
    for (int off = 128; off > 0; off >>= 1) {
        if (tid < off) { r1[tid] += r1[tid + off]; r2[tid] += r2[tid + off]; }
        it.barrier();
    }
}
static void cc_ple_k(const float *__restrict__ keys, float *__restrict__ hyp,
                     const float *__restrict__ val,
                     const float *__restrict__ prm, float *__restrict__ gated,
                     float *__restrict__ normv, const float *__restrict__ conv,
                     float *__restrict__ ring, CcPle p) {
    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto &r1 = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[256]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &r2 = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[256]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    int W = p.C * p.H, tid = it.get_local_id(2);
    if (p.mode == 0) {
        int g = it.get_group(2), s = g / p.C, k = g - s * p.C;
        int kb = p.keys_off + s * W + k * p.H, hb = p.hyp_off + s * W + k * p.H, vb = p.val_off + s * p.H;
        int wk = p.prm_off + k * p.H, wq = p.prm_off + W + k * p.H, wc = p.prm_off + 2 * W + k * p.H;
        float a = 0.f, b = 0.f;
        for (int d = tid; d < p.H; d += 256) { float u = keys[kb + d], v = hyp[hb + d]; a += u * u; b += v * v; }
        r1[tid] = a; r2[tid] = b;
        cc_ple_reduce2(r1, r2, tid);
        float rk = 1.f / sycl::sqrt(r1[0] / (float)p.H + p.eps),
              rq = 1.f / sycl::sqrt(r2[0] / (float)p.H + p.eps);
        it.barrier();
        float dt = 0.f;
        for (int d = tid; d < p.H; d += 256)
            dt += (keys[kb + d] * rk * (1.f + prm[wk + d])) * (hyp[hb + d] * rq * (1.f + prm[wq + d]));
        r1[tid] = dt; r2[tid] = 0.f;
        cc_ple_reduce2(r1, r2, tid);
        float dot = r1[0] / sycl::sqrt((float)p.H);
        float shaped = (dot > 0.f   ? 1.f
                        : dot < 0.f ? -1.f
                                    : 0.f) *
                       sycl::sqrt(sycl::fmax(sycl::fabs(dot), 1e-6f));
        if (dot == 0.f) shaped = sycl::sqrt(1e-6f); /* copysign(x, +0) is +x */
        float gt = cc_sig(shaped);
        it.barrier();
        float c3 = 0.f;
        for (int d = tid; d < p.H; d += 256) {
            float u = gt * val[vb + d];
            gated[s * W + k * p.H + d] = u;
            c3 += u * u;
        }
        r1[tid] = c3; r2[tid] = 0.f;
        cc_ple_reduce2(r1, r2, tid);
        float rc = 1.f / sycl::sqrt(r1[0] / (float)p.H + p.eps);
        for (int d = tid; d < p.H; d += 256)
            normv[s * W + k * p.H + d] = gated[s * W + k * p.H + d] * rc * (1.f + prm[wc + d]);
        return;
    }
    long long d =
        (long long)it.get_group(2) * it.get_local_range(2) +
        it.get_local_id(2);
    if (d >= W) return;
    int SL = (p.CK - 1) * p.NG;
    float rg[32];
    for (int t = 0; t < 32; t++) rg[t] = t < SL ? ring[p.ring_off + d * SL + t] : 0.f;
    int cw = p.conv_off + (int)d * p.CK;
    float wl = conv[cw + p.CK - 1];
    for (int s = 0; s < p.S; s++) {
        float nv = normv[s * W + d];
        float a = wl * nv;
        for (int t = 0; t < 31; t++)
            if (t < p.CK - 1) a += conv[cw + t] * rg[t * p.NG < 32 ? t * p.NG : 0];
        hyp[p.hyp_off + s * W + d] += gated[s * W + d] + a * cc_sig(a);
        for (int t = 0; t < 31; t++) if (t + 1 < SL) rg[t] = rg[t + 1];
        for (int t = 0; t < 32; t++) if (t == SL - 1) rg[t] = nv;
        if (s == p.snap_row)
            for (int t = 0; t < 32; t++) if (t < SL) ring[p.snap_off + d * SL + t] = rg[t];
    }
    for (int t = 0; t < 32; t++) if (t < SL) ring[p.ring_off + d * SL + t] = rg[t];
}
static void cc_ew_k(float *__restrict__ y, const float *__restrict__ a, const float *__restrict__ b,
                               const float *__restrict__ c, const float *__restrict__ e, CcEw p) {
    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    long long i =
        (long long)it.get_group(2) * it.get_local_range(2) +
        it.get_local_id(2);
    if (i >= p.n) return;
    if (p.op == 0) {
        y[p.y_off + i] = a[p.a_off + i] + b[p.b_off + i];
    } else if (p.op == 1) {
        int r = (int)(i / p.D), d = (int)(i - (long long)r * p.D);
        float t = (p.flags & 1) ? b[p.b_off + r * p.D + d] : 0.f;
        if (p.flags & 2) {
            float g = (p.flags & 4) ? cc_sig(e[p.e_off + r * p.e_row]) : 1.f;
            t = t + g * c[p.c_off + r * p.D + d];
        }
        y[p.y_off + i] = (p.flags & 8) ? t : a[p.a_off + i] + t;
    } else if (p.op == 2) {
        float g = a[p.a_off + i];
        y[p.y_off + i] = (g / (1.f + sycl::native::exp(-g))) * b[p.b_off + i];
    } else if (p.op == 3) {
        float v = a[p.a_off + i] / p.fc;
        y[p.y_off + i] = v * cc_sig(v);
    } else if (p.op == 4) {
        int r = (int)(i / p.D), d = (int)(i - (long long)r * p.D), W = p.C * p.D;
        float v = 0.f;
        for (int k = 0; k < p.C; k++) {
            int j = r * W + k * p.D + d;
            v += cc_sig(a[p.a_off + j]) * b[p.b_off + j];
        }
        y[p.y_off + i] = v / p.fc;
    } else if (p.op == 5) {
        y[p.y_off + i] = 2.f * cc_sig(a[p.a_off + i] / p.fc);
    } else if (p.op == 6) {
        int W = p.C * p.D;
        int r = (int)(i / W), rem = (int)(i - (long long)r * W), k = rem / p.D, d = rem - k * p.D;
        y[p.y_off + i] += a[p.a_off + r * p.C + k] * b[p.b_off + r * p.D + d];
    } else if (p.op == 7) {
        y[p.y_off + i] = a[p.a_off + i] * p.fc;
    } else if (p.op == 8) {
        y[p.y_off + i] = a[p.a_off + i] + e[p.e_off + (int)(i % p.D)] * b[p.b_off + i];
    } else if (p.op == 9) {
        y[p.y_off + i] = a[p.a_off + i] * cc_sig(b[p.b_off + i]);
    } else if (p.op == 10) {
        int d = (int)(i % p.D);
        y[p.y_off + i] = e[p.e_off + d] * a[p.a_off + i] + c[p.c_off + d] * b[p.b_off + i];
    } else if (p.op == 11) {   /* kv16.h's kv16_round_bits line for line (cc_f16_round) */
        y[p.y_off + i] = cc_f16_round(a[p.a_off + i]);
    } else if (p.op == 12) {   /* LFM3's head gate: one logit a head of D values */
        y[p.y_off + i] = a[p.a_off + i] * cc_sig(b[p.b_off + i / p.D]);
    }
}

/* chain_route.comp: one block per row, its 256 / SG warps the shader's subgroups (the row's max
 * and sum: each warp's, then theirs in warp order) */
template <int SG>
static void cc_route_k(const float *__restrict__ l,
                       const float *__restrict__ bias, int *__restrict__ ids,
                       float *__restrict__ w, CcRoute p) {
    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const float NEG = -3.402823466e38f;
    auto &sc =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[1024]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &pk =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[1024]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &rv = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[256]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &gv = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[64]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &wv = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[32]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &keepg = *sycl::ext::oneapi::group_local_memory_for_overwrite<int[64]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &chosen =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<int[32]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &keepi =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<int[256]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    int s = it.get_group(2);
    if (s >= p.S) return;
    int tid = it.get_local_id(2), lane = tid & (SG - 1), sg = tid / SG, nsg = 256 / SG;
    int lb = p.l_off + s * p.l_row;
    auto logit = [&](int e) { float v = l[lb + e]; if (p.has_bias == 2) v += bias[p.b_off + e]; return v / p.temp; };
    if (p.kind == 0) {
        float m = NEG;
        for (int e = tid; e < p.E; e += 256) m = sycl::fmax(m, logit(e));
        m = cc_wmax<SG>(m);
        if (lane == 0) rv[sg] = m;
        it.barrier(sycl::access::fence_space::local_space);
        float mx = NEG;
        for (int g = 0; g < nsg; g++) mx = sycl::fmax(mx, rv[g]);
        it.barrier(sycl::access::fence_space::local_space);
        float acc = 0.f;
        for (int e = tid; e < p.E; e += 256) {
            float v = sycl::native::exp(logit(e) - mx); sc[e] = v; acc += v;
        }
        acc = cc_wsum<SG>(acc);
        if (lane == 0) rv[sg] = acc;
        it.barrier(sycl::access::fence_space::local_space);
        float sum = 0.f;
        for (int g = 0; g < nsg; g++) sum += rv[g];
        it.barrier(sycl::access::fence_space::local_space);
        for (int e = tid; e < p.E; e += 256) sc[e] = sc[e] / sum;
    } else {
        for (int e = tid; e < p.E; e += 256) {
            float v = logit(e);
            sc[e] =
                p.kind == 1
                    ? 1.f / (1.f + sycl::native::exp(-v))
                    : sycl::sqrt(
                          v > 20.f ? v : sycl::log(1.f + sycl::native::exp(v)));
        }
    }
    it.barrier(sycl::access::fence_space::local_space);
    for (int e = tid; e < p.E; e += 256) {
        float v = sc[e];
        if (p.has_bias == 1) v += bias[p.b_off + e];
        pk[e] = sycl::isnan(v) ? NEG : v;
    }
    it.barrier(sycl::access::fence_space::local_space);
    if (p.n_group > 1) {
        int per = p.E / p.n_group;
        if (tid < p.n_group) {
            float b1 = NEG, b2 = NEG;
            for (int e = tid * per; e < tid * per + per; e++) { float v = pk[e]; if (v > b1) { b2 = b1; b1 = v; } else if (v > b2) b2 = v; }
            gv[tid] = b1 + b2; keepg[tid] = 0;
        }
        it.barrier(sycl::access::fence_space::local_space);
        if (tid == 0)
            for (int k = 0; k < p.topk_group; k++) {
                int bg = -1; float bv = NEG;
                for (int g = 0; g < p.n_group; g++) if (keepg[g] == 0 && gv[g] > bv) { bv = gv[g]; bg = g; }
                if (bg < 0) break;
                keepg[bg] = 1;
            }
        it.barrier(sycl::access::fence_space::local_space);
        for (int e = tid; e < p.E;
             e +=
             256) if (keepg[sycl::min(sycl::max(e / per, 0), p.n_group - 1)] ==
                      0) pk[e] = NEG;
        it.barrier(sycl::access::fence_space::local_space);
    }
    float mine[4];
    #pragma unroll
    for (int j = 0; j < 4; j++) { int e = tid + 256 * j; mine[j] = e < p.E ? pk[e] : NEG; }
    for (int k = 0; k < p.K; k++) {
        float bv = NEG; int be = 0x7fffffff;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int e = tid + 256 * j;
            if (mine[j] > bv || (mine[j] == bv && mine[j] > NEG && e < be)) { bv = mine[j]; be = e; }
        }
        float m = cc_wmax<SG>(bv);
        int me = cc_wmin<SG>(bv == m && m > NEG ? be : 0x7fffffff);
        int h = (k & 1) * 128;
        if (lane == 0) { rv[h + sg] = m; keepi[h + sg] = me; }
        it.barrier(sycl::access::fence_space::local_space);
        float b = NEG; int c = -1;
        for (int g = 0; g < nsg; g++) {
            float u = rv[h + g]; int ie = keepi[h + g];
            if (u > NEG && ie != 0x7fffffff && (u > b || (u == b && ie < c))) { b = u; c = ie; }
        }
        if (tid == 0) chosen[k] = c;
        #pragma unroll
        for (int j = 0; j < 4; j++) if (c == tid + 256 * j) mine[j] = NEG;
    }
    it.barrier(sycl::access::fence_space::local_space);
    int b = tid < 32 ? chosen[tid] : -1;
    if (tid < 32) wv[tid] = tid < p.K && b >= 0 ? sc[b] : 0.f;
    it.barrier(sycl::access::fence_space::local_space);
    float sum = 0.f;
    for (int k = 0; k < p.K; k++) sum += wv[k];
    float div = sum + p.norm_eps;
    bool norm = (p.norm == 1 || (p.norm == 2 && p.K > 1)) && div > 0.f;
    if (tid < p.K) {
        float x = wv[tid];
        if (norm) x = x / div;
        ids[p.i_off + s * p.i_row + tid] = b;
        w[p.w_off + s * p.w_row + tid] = x * p.scale;
    }
}
/* The hybrid steps' flags (cc_signal, cc_wait_host): host tasks on the chain's in-order queue,
 * not kernels. The .cu's cc_signal_k / cc_wait_k ran on the device, a store to and a spin on a
 * word of mapped host memory; the first migration made them a system-scope atomic_ref on the
 * host-USM word, and on Ponte Vecchio a device atomic on host memory over PCIe is an
 * AtomicAccessViolation: i915 logs "Atomic access violation @ ... / Catastrophic memory error in
 * context", bans the context, and the runtime aborts the process (the first card, the chain
 * test's "signal, wait_host"; MIGRATION.md section 13). Battlemage sits on the same bus with the
 * same rule, so there is no device form of this at all: a host task is the only implementation.
 * A host task runs on the runtime's own thread once every earlier command of the in-order
 * queue is complete (the step's rows the kernels wrote to DOWN memory are there by then: their
 * events completed), and every later command of the queue is held until it returns. So signal
 * = store v into the DOWN word (the host polls it, as before), wait_host = poll the UP word
 * until the host has written v or the context is lost, then return: the frame's
 * later ops go on. The words are host USM (sycl::malloc_host, cc_buf): a host store is an
 * ordinary cacheable write that the device's later reads see over PCIe, as the mapped pinned
 * memory of the .cu; nothing is copied. The task takes only the host state's short flag-access lock and calls nothing of the chain:
 * the engine's writer thread may submit on either queue (host_up, write) meanwhile, and the
 * runtime's own locks are not held while a host task's body runs. The poll spins briefly
 * (a host that is already there) and then naps 20 us between loads, so the host's experts
 * keep their cores during the milliseconds they take. A kernel spinning on host memory would
 * also have sat under the driver's hang check (heartbeat 0 on the i915 backport): no kernel. */
/* The shared host state serializes flag accesses with cancellation. A canceled
 * task never touches the flag, even if it starts after error cleanup freed it.
 * No SYCL queue API is called inside the host task (including coli_sycl_lost). */
/* one block of 256, two passes: the argmax of n logits (lowest index on a tie) into word 0
 * of out and its softmax probability (terms within 20 of the maximum) into word 1; a block a
 * row, the row's n logits and two words past the block before's (cc_pick_rows) */
template <int SG>
static void cc_pick_k(const float *__restrict__ lg, void *__restrict__ out, int n) {
    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    lg += (size_t)it.get_group(2) * n;
        out = (char *)out + (size_t)it.get_group(2) * 8;
    constexpr int W = 256 / SG;   /* the block's warps */
    auto &rv = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[W]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &ri = *sycl::ext::oneapi::group_local_memory_for_overwrite<int[W]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    int tid = it.get_local_id(2), lane = tid & (SG - 1), sg = tid / SG;
    int be = tid < n ? tid : 0x7fffffff;
    float bv = tid < n ? lg[tid] : -INFINITY;
    for (int i = tid + 256; i < n; i += 256) {
        float v = lg[i];
        if (v > bv) { bv = v; be = i; }
    }
    float m = cc_wmax<SG>(bv);
    int me = cc_wmin<SG>(bv == m ? be : 0x7fffffff);
    if (lane == 0) { rv[sg] = m; ri[sg] = me; }
    it.barrier(sycl::access::fence_space::local_space);
    float mx = rv[0]; int best = ri[0];
    for (int g = 1; g < W; g++)
        if (rv[g] > mx || (rv[g] == mx && ri[g] < best)) { mx = rv[g]; best = ri[g]; }
    float z = 0.f;
    for (int i = tid; i < n; i += 256) {
        float d = lg[i] - mx;
        if (d > -20.f) z += sycl::native::exp(d);
    }
    z = cc_wsum<SG>(z);
    it.barrier(sycl::access::fence_space::local_space);
    if (lane == 0) rv[sg] = z;
    it.barrier(sycl::access::fence_space::local_space);
    if (tid == 0) {
        float tot = 0.f;
        for (int g = 0; g < W; g++) tot += rv[g];
        ((int *)out)[0] = best == 0x7fffffff ? 0 : best;
        ((float *)out)[1] = tot > 0.f ? 1.f / tot : 0.f;
    }
}
/* backend_cuda.cu's gx4_idx: every 8 float4s padded by 1 so the warps' 4-way shared loads
 * land in distinct banks (a warp a row, 256 / SG rows a block). cc_matmul_rows_i8_k has gemv_i8_rows<1, 1>'s layout for a row list
 * ids[0..n), but sums 16 products before adding them: its bits can differ (drafts only). */
static inline int cc_gx4(int k) { return k + (k >> 3); }
template <int PAD, int SG>
static void
cc_matmul_rows_i8_k(float *__restrict__ y, const float *__restrict__ x,
                    const int8_t *__restrict__ w, const float *__restrict__ sc,
                    const int *__restrict__ ids, int I, int O, int n,
                    uint8_t *slm) {
    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto sx = (sycl::float4 *)slm;
    int tid = it.get_local_id(2), warp = tid / SG, lane = tid & (SG - 1), I4 = I >> 2, I16 = I >> 4;
    const sycl::float4 *x4 = reinterpret_cast<const sycl::float4 *>(x);
    for (int i = tid; i < I4; i += 256) sx[PAD ? cc_gx4(i) : i] = x4[i];
    it.barrier(sycl::access::fence_space::local_space);
    int j = (int)it.get_group(2) * (256 / SG) + warp;
    if (j >= n) return;
    int o = ids[j];
    if (o < 0 || o >= O) { if (lane == 0) y[j] = -INFINITY; return; }
    const sycl::int4 *wr =
        reinterpret_cast<const sycl::int4 *>(w + (size_t)o * I);
    float acc = 0.f;
    for (int i = lane; i < I16; i += SG) {
        sycl::int4 v = wr[i];
        const int8_t *q = reinterpret_cast<const int8_t *>(&v);
        int k = i * 4;
        sycl::float4 x0 = sx[PAD ? cc_gx4(k) : k],
                     x1 = sx[PAD ? cc_gx4(k + 1) : k + 1];
        sycl::float4 x2 = sx[PAD ? cc_gx4(k + 2) : k + 2],
                     x3 = sx[PAD ? cc_gx4(k + 3) : k + 3];
        acc += (float)q[0] * x0.x() + (float)q[1] * x0.y() +
               (float)q[2] * x0.z() + (float)q[3] * x0.w() +
               (float)q[4] * x1.x() + (float)q[5] * x1.y() +
               (float)q[6] * x1.z() + (float)q[7] * x1.w() +
               (float)q[8] * x2.x() + (float)q[9] * x2.y() +
               (float)q[10] * x2.z() + (float)q[11] * x2.w() +
               (float)q[12] * x3.x() + (float)q[13] * x3.y() +
               (float)q[14] * x3.z() + (float)q[15] * x3.w();
    }
    acc = cc_wsum<SG>(acc);
    if (lane == 0) y[j] = acc * sc[o];
}
template <int SG>
static void cc_matmul_rows_k(float *__restrict__ y, const float *__restrict__ x,
        const void *__restrict__ w, const float *__restrict__ sc, const int *__restrict__ ids, int fmt, int I, int O, int n) {
    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int warp = it.get_local_id(2) / SG, lane = it.get_local_id(2) & (SG - 1), j = (int)it.get_group(2) * (256 / SG) + warp;
    if (j >= n) return;
    int o = ids[j];
    if (o < 0 || o >= O) { if (lane == 0) y[j] = -INFINITY; return; }
    float acc = 0.f;
    if (fmt == 1) {
        const int8_t *wr = (const int8_t *)w + (size_t)o * I;
        for (int i = lane; i < I; i += SG) acc += x[i] * (float)wr[i];
        acc = cc_wsum<SG>(acc);
        if (lane == 0) y[j] = acc * sc[o];
    } else {
        const float *wr = (const float *)w + (size_t)o * I;
        for (int i = lane; i < I; i += SG) acc += x[i] * wr[i];
        acc = cc_wsum<SG>(acc);
        if (lane == 0) y[j] = acc;
    }
}

/* ---- contexts ------------------------------------------------------------------- */
/* the chain's books for coli_sycl_mem_info (where the device has no free-memory query): its live
 * device buffers, one count for every device (g_st is not per device: an estimate on a box with
 * several) */
static size_t cc_booked_bytes(int device) { (void)device; return g_st.dev_bytes; }
/* Read at context init, never in a device lambda. Unsupported overrides keep the profile's choice. Unset: the
 * sub-group measured best on the devices' architecture where the profile has a row and every opened device agrees
 * (coli_cuda_tuned_default, which says it: COLI_CUDA_DNREC_PROMPT_SG 16 on Data Center GPU Max cards) and this
 * device offers it, else `fallback`. */
static int cc_kernel_sg(const coli_sycl_profile &pr, int fallback, const char *name) {
    const char *e = getenv(name);
    if (!e || !*e) {
        const int t = coli_cuda_tuned_default(name, fallback);   /* its line names the context's own as the code's */
        return t != fallback && coli_sycl_sg_offered(pr, t) ? t : fallback;
    }
    const int sg = !strcmp(e, "16") ? 16 : !strcmp(e, "32") ? 32 : 0;
    if (sg && coli_sycl_sg_offered(pr, sg)) return sg;
    std::fprintf(stderr, "[SYCL] %s=%s ignored (requires an offered sub-group of 16 or 32); using %d\n", name, e, fallback);
    return fallback;
}
static int cc_init(int device) {
    CcCtx *c = ctx_of(device);
    if (c && c->up) { g_cur = (int)(c - g_cc); return !c->lost; }
    if (!c) {
        if (g_ncc >= COLI_CUDA_MAX_DEVICES) return 0;
        c = &g_cc[g_ncc];
        std::memset(c, 0, sizeof *c);
        c->device = device;
    }
    if (!coli_sycl_is_open(device)) return 0;   /* coli_cuda_init opens the devices (coli_sycl_open) */
    const coli_sycl_profile &pr = coli_sycl_profile_for(device);
    c->stream = &coli_sycl_queue(device);
    c->cs = &coli_sycl_copy_queue(device);
    c->sg = pr.sg_default == 32 ? 32 : 16;
    const int pvc_sg = pr.arch == COLI_SYCL_ARCH_PVC && coli_sycl_sg_offered(pr, 16) ? 16 : c->sg;
    c->sg_dn = cc_kernel_sg(pr, pvc_sg, "COLI_CUDA_DNREC_SG");
    c->sg_attn = cc_kernel_sg(pr, pvc_sg, "COLI_CUDA_ATTN_SG");
    c->sg_dnp = cc_kernel_sg(pr, c->sg, "COLI_CUDA_DNREC_PROMPT_SG");   /* cc_dnrec_split_k alone (a prompt chunk's rows) */
    c->slm = pr.slm_bytes ? pr.slm_bytes : 48 * 1024;
    if (!cc_try(nullptr, "init", [&] { c->sy = new CcSync; c->fp64 = coli_sycl_device(device).has(sycl::aspect::fp64); })) return 0;
    c->up = 1; c->lost = 0; c->in_frame = 0; c->ev_pending = 0;
    coli_sycl_ledger_add(cc_booked_bytes);
    if (c == &g_cc[g_ncc]) g_ncc++;
    g_cur = (int)(c - g_cc);
    return 1;
}
static int cc_ready(void) { return live() != nullptr; }
/* Hybrid engine host loops poll lost() while waiting for their DOWN word.
 * Collect asynchronous errors here too, on the caller thread, never in a task. */
static int cc_lost_(void) { return live() == nullptr; }
static void cc_shutdown(void) {
    coli_sycl_fprof_report();   /* COLI_SYCL_FRAME_PROF: the books, once */
    for (int i = 0; i < g_ncc; i++) {
        CcCtx *c = &g_cc[i];
        if (!c->up) continue;
        if (coli_sycl_is_open(c->device)) {   /* a device closed under the chain has nothing left to wait for or free */
            if (!c->lost) (void)cc_try(c, "shutdown", [&] { c->stream->wait_and_throw(); c->cs->wait_and_throw(); });
            cc_release(c, c->adec);
            cc_adec_reap(c);
            coli_sycl_scratch_reap(c->device);
            for (int j = 0; j < CC_STAGE_N; j++) cc_release(c, c->stg[j]);
        }
        delete c->sy;
        c->sy = nullptr; c->up_made = 0; c->up_seq = 0;
        c->adec = nullptr; c->adec_bytes = 0; c->adec_n_old = 0;
        std::memset(c->stg, 0, sizeof c->stg); std::memset(c->stg_cap, 0, sizeof c->stg_cap); c->stg_next = 0;
        c->up = 0;
    }
    g_ncc = 0; g_cur = -1;
}
static int cc_device(int d) {
    int was = g_cur >= 0 ? g_cc[g_cur].device : -1;
    CcCtx *c = ctx_of(d);
    if (c) g_cur = (int)(c - g_cc);
    return was;
}
static int cc_device_now(void) { return g_cur >= 0 ? g_cc[g_cur].device : -1; }

/* ---- buffers ---------------------------------------------------------------------- */
static CcBuf *cc_buf(size_t bytes, int kind) {
    CcCtx *c = live();
    if (!c || kind < 0 || kind > 2) return nullptr;
    if (!bytes) bytes = 4;
    CcBuf *b = (CcBuf *)std::calloc(1, sizeof *b);
    if (!b) return nullptr;
    b->kind = kind; b->dev = c->device; b->bytes = bytes;
    if (kind == CC_DEV) {
        /* zeroed on the chain's queue, in order with the frames (a fill elsewhere could land
         * after a first write) */
        b->d = (float *)cc_alloc(c, bytes, 0);
        if (!b->d || !cc_try(c, "buffer", [&] { c->stream->memset(b->d, 0, bytes); })) {
            cc_release(c, b->d);
            std::free(b);
            return nullptr;
        }
        g_st.dev_bytes += bytes;
    } else {
        /* host USM: page-locked and mapped into the device by nature (cudaHostAlloc's flags),
         * the device pointer the host's */
        b->h = cc_alloc(c, bytes, 1);
        if (!b->h) { std::free(b); return nullptr; }
        b->d = (float *)b->h;
        std::memset(b->h, 0, bytes);
    }
    return b;
}
static void cc_free(CcBuf *b) {
    if (!b) return;
    CcCtx *c = ctx_of(b->dev);
    if (c && c->up && coli_sycl_is_open(c->device)) {
        /* the frames that may read it, and the copies that may write it */
        if (!c->lost) (void)cc_try(c, "free", [&] { c->stream->wait_and_throw(); c->cs->wait_and_throw(); });
        if (b->kind == CC_DEV) { cc_release(c, b->d); g_st.dev_bytes -= b->bytes; }
        else cc_release(c, b->h);
    }
    std::free(b);
}
static int cc_reserve(CcBuf **b, size_t bytes, int kind) {
    if (*b) CC_OWN(cur(), "cc_reserve", *b);   /* a bank's buffer asked for on the other device */
    if (*b && (*b)->bytes >= bytes && (*b)->kind == kind) return 1;
    CcBuf *n = cc_buf(bytes, kind);
    if (!n) return 0;
    cc_free(*b);
    *b = n;
    return 1;
}
static void *cc_ptr(const CcBuf *b) { return b ? b->h : nullptr; }
static size_t cc_bytes(const CcBuf *b) { return b ? b->bytes : 0; }

/* ---- frames ----------------------------------------------------------------------- */
static int cc_begin(void) {
    CcCtx *c = live();
    if (!c) return 0;
    c->in_frame = 1;
    g_st.frames++;
    cc_fprof_begin();
    return 1;
}
static int cc_submit(int wait) {
    CcCtx *c = live();
    if (!c) return 0;
    c->in_frame = 0;
    /* the frame's end: a barrier on the in-order queue, complete once every op before it is */
    if (!cc_try(c, "frame event", [&] { c->sy->ev = c->stream->ext_oneapi_submit_barrier(); })) return 0;
    if (coli_sycl_fprof_flag) { coli_sycl_fprof_ev(c->sy->ev); coli_sycl_fprof_name("barrier"); g_fp_tsub = cc_now_ms(); }
    c->ev_pending = 1;
    if (!wait) return 1;
    double t0 = cc_now_ms();
    int ok = cc_try(c, "frame wait", [&] { c->sy->ev.wait_and_throw(); });
    g_st.waits++; g_st.wait_ms += cc_now_ms() - t0;
    c->ev_pending = 0;
    if (ok) { cc_adec_reap(c); coli_sycl_scratch_reap(c->device); }   /* the queue passed the frames: the retired blocks go */
    if (ok) cc_fprof_end();
    return ok;
}
static int cc_finish(void) {
    CcCtx *c = live();
    if (!c) return 0;
    double t0 = cc_now_ms();
    int ok = cc_try(c, "finish", [&] { c->stream->wait_and_throw(); c->cs->wait_and_throw(); });
    g_st.waits++; g_st.wait_ms += cc_now_ms() - t0;
    c->ev_pending = 0;
    if (ok) { cc_adec_reap(c); coli_sycl_scratch_reap(c->device); }   /* the queue passed the frames: the retired blocks go */
    if (ok) cc_fprof_end();
    return ok;
}

/* ---- transfers ------------------------------------------------------------------ */
/* A copy of up to CC_COPY_K_MAX bytes by a kernel, mapped host memory on either side or none:
 * cudaMemcpyAsync takes a copy engine, and a frame's small copies waited there behind the
 * megabytes of the experts the chain promotes (cc_host_up) on another stream (Qwen3.8 on an RTX
 * PRO 4000: a draft's first kernel up to 12.5 ms late). The same floats; COLI_CUDA_COPY_K=0 keeps
 * the engine. Regions of a copy the same way (cc_regions_k), up to CC_REG_K a launch. */
#define CC_COPY_K_MAX ((size_t)1 << 20)
static void cc_copy_k(float *__restrict__ d, const float *__restrict__ s, long long n) {
    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    long long i = (long long)it.get_group(2) * it.get_local_range(2) + it.get_local_id(2),
              st = (long long)it.get_group_range(2) * it.get_local_range(2);
    if ((((uintptr_t)d | (uintptr_t)s) & 15) == 0) {
        long long n4 = n / 4;
        for (long long k = i; k < n4; k += st) reinterpret_cast<sycl::float4 *>(d)[k] = reinterpret_cast<const sycl::float4 *>(s)[k];
        for (long long k = n4 * 4 + i; k < n; k += st) d[k] = s[k];
    } else
        for (long long k = i; k < n; k += st) d[k] = s[k];
}
/* the regions a launch takes (the list is a kernel argument: under a kilobyte, which every
 * runtime passes whole) */
#define CC_REG_K 64
struct CcRegK { unsigned n, d[CC_REG_K], s[CC_REG_K], len[CC_REG_K]; };   /* offsets and counts in floats */
static void cc_regions_k(float *__restrict__ d, const float *__restrict__ s, CcRegK r) {
    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const unsigned i = it.get_group(2);
    if (i >= r.n) return;
    for (unsigned j = it.get_local_id(2); j < r.len[i]; j += it.get_local_range(2)) d[r.d[i] + j] = s[r.s[i] + j];
}
static int cc_copy_k_on(void) {
    static int on = -1;
    if (on < 0) { const char *e = getenv("COLI_CUDA_COPY_K"); on = !(e && *e == '0'); }
    return on;
}
static int cc_copy(CcBuf *dst, size_t doff, CcBuf *src, size_t soff, size_t n) {
    CcCtx *c = live();
    CC_OWN(c, "cc_copy", dst, src);
    if (!c || !dst || !src) return 0;
    if (!n) return 1;
    if ((doff + n) * 4 > dst->bytes || (soff + n) * 4 > src->bytes) return 0;
    /* mapped host memory is addressed by its device pointer on both sides */
    g_st.ops++;
    if (dst->kind != CC_DEV) g_st.bytes_down += n * 4;
    if (src->kind != CC_DEV) g_st.bytes_up += n * 4;
    float *d = dst->d + doff; const float *s = src->d + soff;
    if (n * 4 <= CC_COPY_K_MAX && cc_copy_k_on()) {
        size_t blocks = (n / 4 + 255) / 256;
        if (blocks < 1) blocks = 1;
        if (blocks > 256) blocks = 256;
        const long long nn = (long long)n;
        CC_K(c, dim3((unsigned)blocks), 256, cc_copy_k(d, s, nn));
        return launched(c, "copy (kernel)");
    }
    const int ok = cc_try(c, "copy", [&] { coli_sycl_fprof_ev(c->stream->memcpy(d, s, n * 4)); });
    if (coli_sycl_fprof_flag) coli_sycl_fprof_name("copy (memcpy)");
    return ok;
}
static int cc_zero(CcBuf *dst, size_t off, size_t n) {
    CcCtx *c = live();
    CC_OWN(c, "cc_zero", dst);
    if (!c || !dst) return 0;
    if (!n) return 1;
    if ((off + n) * 4 > dst->bytes) return 0;
    g_st.ops++;
    const int ok = cc_try(c, "zero", [&] { coli_sycl_fprof_ev(c->stream->memset(dst->d + off, 0, n * 4)); });
    if (coli_sycl_fprof_flag) coli_sycl_fprof_name("zero (memset)");
    return ok;
}
/* Regions on a grid (vk_chain.c's cp2_grid): region i = (q, h) = (i / per, i % per) copies len
 * floats from src[s0 + q*sq + h*sh ..] to dst[d0 + q*dq + h*dh ..]. A step's K and V rows into
 * the head-major cache are such a grid, a region per row and head: unlike vkCmdCopyBuffer, the
 * fallback below issues a cudaMemcpyAsync per region (41366 calls and 521 ms of host time on a
 * 1024-token prompt, 40 calls a decode token on Qwen3.6-35B), so any grid of two or more regions
 * takes one kernel launch. */
struct CcGrid { long long d0, s0, dh, sh, dq, sq; int len, per, n; };
static void cc_copy_grid_k(float *__restrict__ d, const float *__restrict__ s, CcGrid g) {
    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    long long gid = (long long)it.get_group(2) * it.get_local_range(2) + it.get_local_id(2);
    if (gid >= (long long)g.n * g.len) return;
    long long i = gid / g.len, j = gid - i * g.len, q = i / g.per, h = i - q * g.per;
    d[g.d0 + q * g.dq + h * g.dh + j] = s[g.s0 + q * g.sq + h * g.sh + j];
}
static int cc_grid_of(const CcRegion *r, int n, CcGrid *g) {
    if (n < 2 || !r[0].n || r[0].n > 0x7fffffff) return 0;
    long long L = (long long)r[0].n, d0 = (long long)r[0].dst, s0 = (long long)r[0].src;
    long long dh = (long long)r[1].dst - d0, sh = (long long)r[1].src - s0;
    int per = 1;
    while (per < n && (long long)r[per].dst - (long long)r[per - 1].dst == dh && (long long)r[per].src - (long long)r[per - 1].src == sh) per++;
    if (n % per) return 0;
    long long Q = n / per, dq = Q > 1 ? (long long)r[per].dst - d0 : 0, sq = Q > 1 ? (long long)r[per].src - s0 : 0;
    for (int i = 0; i < n; i++) {
        long long q = i / per, h = i % per;
        if ((long long)r[i].n != L || (long long)r[i].dst != d0 + q * dq + h * dh || (long long)r[i].src != s0 + q * sq + h * sh) return 0;
    }
    *g = CcGrid{d0, s0, dh, sh, dq, sq, (int)L, per, n};
    return 1;
}
/* How many of the regions from r[0] make one grid, in whole rows of it. A list that is not one
 * grid is often a few: the rows a step writes into a ring break where the ring wraps (on Flint
 * each sliding layer's prompt step wrote 2 x 1538 regions a cudaMemcpyAsync each, 862400 calls
 * and about 15 s of host time in a 16K prompt on an RTX PRO 4000), and each side is a grid. */
static int cc_grid_run(const CcRegion *r, int n) {
    if (n < 2) return n;
    long long L = (long long)r[0].n, d0 = (long long)r[0].dst, s0 = (long long)r[0].src;
    long long dh = (long long)r[1].dst - d0, sh = (long long)r[1].src - s0;
    int per = 1;
    while (per < n && (long long)r[per].n == L && (long long)r[per].dst - (long long)r[per - 1].dst == dh &&
           (long long)r[per].src - (long long)r[per - 1].src == sh) per++;
    if (per == n) return n;
    long long dq = (long long)r[per].dst - d0, sq = (long long)r[per].src - s0;
    int Q = 1;
    for (; (Q + 1) * per <= n; Q++) {
        int ok = 1;
        for (int h = 0; h < per && ok; h++) {
            const CcRegion *x = &r[Q * per + h];
            ok = (long long)x->n == L && (long long)x->dst == d0 + Q * dq + h * dh && (long long)x->src == s0 + Q * sq + h * sh;
        }
        if (!ok) break;
    }
    return Q * per;
}
static int cc_copy_regions(CcBuf *dst, CcBuf *src, const CcRegion *r, int n) {
    CcCtx *c = live();
    CC_OWN(c, "cc_copy_regions", dst, src);
    if (!c || !dst || !src || (n > 0 && !r)) return 0;
    for (int i = 0; i < n; i++)
        if ((r[i].dst + r[i].n) * 4 > dst->bytes || (r[i].src + r[i].n) * 4 > src->bytes) return 0;
    float *d = dst->d; const float *s = src->d;
    int small = n > 0 && cc_copy_k_on();
    for (int i = 0; i < n && small; i++) small = r[i].n * 4 <= CC_COPY_K_MAX && r[i].dst + r[i].n < (1ull << 32) && r[i].src + r[i].n < (1ull << 32);
    if (small && !(n >= 2 && dst != src && cc_grid_run(r, n) == n)) {   /* scattered regions: a block each, by kernel */
        for (int i = 0; i < n; i += CC_REG_K) {
            CcRegK k; k.n = (unsigned)(n - i < CC_REG_K ? n - i : CC_REG_K);
            for (unsigned j = 0; j < k.n; j++) { k.d[j] = (unsigned)r[i + j].dst; k.s[j] = (unsigned)r[i + j].src; k.len[j] = (unsigned)r[i + j].n; }
            CC_K(c, dim3(k.n), 128, cc_regions_k(d, s, k));
            if (!launched(c, "copy regions (kernel)")) return 0;
        }
        g_st.ops++;
        return 1;
    }
    CcGrid g;
    const int grids = dst != src && dst->d && src->d;
    for (int i = 0; i < n; ) {
        int k = grids ? cc_grid_run(r + i, n - i) : 1;
        if (k >= 2 && cc_grid_of(r + i, k, &g)) {   /* one kernel for each grid */
            long long tot = (long long)g.n * g.len;
            CC_K(c, dim3(grid1((size_t)tot, 256)), 256, cc_copy_grid_k(d, s, g));
            if (!launched(c, "copy regions (grid)")) return 0;
            i += k;
            continue;
        }
        if (r[i].n && !cc_try(c, "copy regions", [&] { coli_sycl_fprof_ev(c->stream->memcpy(d + r[i].dst, s + r[i].src, r[i].n * 4)); })) return 0;
        if (coli_sycl_fprof_flag) coli_sycl_fprof_name("copy regions (memcpy)");
        i++;
    }
    g_st.ops++;
    return 1;
}
/* A DEV buffer takes the bytes through the queue, from a staging slot of host USM the bytes go
 * to first: the caller may reuse its memory when the call returns, as it could after
 * cudaMemcpyAsync from pageable memory (a SYCL copy reads its source when it runs). CC_STAGE_N
 * slots in a ring, each grown to its largest write, a slot reused once the copy that last read it
 * is done (a write that gets no slot goes the slow way: the copy waited for). A host-mapped
 * buffer is written in place: the caller orders that against the frames that read it, as with
 * cc_ptr. */
static int cc_write(CcBuf *dst, size_t off, const void *src, size_t bytes) {
    CcCtx *c = live();
    CC_OWN(c, "cc_write", dst);
    if (!c || !dst || !src) return 0;
    if (!bytes) return 1;
    if (off * 4 + bytes > dst->bytes) return 0;
    g_st.bytes_up += bytes; g_st.ops++;
    if (dst->kind != CC_DEV) { std::memcpy((char *)dst->h + off * 4, src, bytes); return 1; }
    float *d = dst->d + off;
    const unsigned i = c->stg_next++ % CC_STAGE_N;
    if (!cc_try(c, "write (slot)", [&] { c->sy->stg_ev[i].wait_and_throw(); })) return 0;
    if (c->stg_cap[i] < bytes) {
        cc_release(c, c->stg[i]);
        c->stg[i] = cc_alloc(c, bytes, 1); c->stg_cap[i] = c->stg[i] ? bytes : 0;
        if (!c->stg[i]) return cc_try(c, "write", [&] { c->stream->memcpy(d, src, bytes).wait_and_throw(); });
    }
    std::memcpy(c->stg[i], src, bytes);
    const void *s = c->stg[i];
    const int ok = cc_try(c, "write", [&] { c->sy->stg_ev[i] = c->stream->memcpy(d, s, bytes); coli_sycl_fprof_ev(c->sy->stg_ev[i]); });
    if (coli_sycl_fprof_flag) coli_sycl_fprof_name("write (memcpy)");
    return ok;
}
static int cc_read(CcBuf *src, size_t off, void *dst, size_t bytes) {
    CcCtx *c = live();
    CC_OWN(c, "cc_read", src);
    if (!c || !src || !dst) return 0;
    if (!bytes) return 1;
    if (off * 4 + bytes > src->bytes) return 0;
    if (!cc_finish()) return 0;
    g_st.bytes_down += bytes;
    if (src->kind != CC_DEV) { std::memcpy(dst, (const char *)src->h + off * 4, bytes); return 1; }
    const float *s = src->d + off;
    return cc_try(c, "read", [&] { c->stream->memcpy(dst, s, bytes).wait_and_throw(); });
}

/* ---- ops --------------------------------------------------------------------------- */
static int cc_matmul(ColiCudaTensor *t, CcBuf *x, size_t xo, CcBuf *y, size_t yo, int S) {
    CcCtx *c = live();
    CC_OWN(c, "cc_matmul", x, y);
    cc_own_tensor(c, "cc_matmul", t);
    if (!c || !t || !x || !y || S < 1) return 0;
    int fmt, I, O;
    if (!coli_cuda_tensor_shape(t, &fmt, &I, &O) || coli_cuda_tensor_device(t) != c->device) return 0;
    if ((xo + (size_t)S * I) * 4 > x->bytes || (yo + (size_t)S * O) * 4 > y->bytes) return 0;
    g_st.matmuls++; g_st.ops++;
    if (!coli_cuda_tensor_gemm_async(t, y->d + yo, x->d + xo, S, (void *)c->stream)) { cc_mark_lost(c); return 0; }
    if (coli_sycl_fprof_flag) {   /* the backend's launches, named by the matrix: shape and format */
        char k[48];
        std::snprintf(k, sizeof k, "matmul %dx%d fmt%d", O, I, fmt);
        coli_sycl_fprof_name(k);
    }
    return 1;
}
static int cc_matmul_grp(ColiCudaTensor *const *t, int n, CcBuf *x, size_t xo, int x_row, int x_grp, CcBuf *y, size_t yo, int y_row, int S) {
    CcCtx *c = live();
    CC_OWN(c, "cc_matmul_grp", x, y);
    if (!c || !t || n < 1 || !x || !y || S < 1) return 0;
    int fmt, I, O;
    if (!t[0] || !coli_cuda_tensor_shape(t[0], &fmt, &I, &O) || coli_cuda_tensor_device(t[0]) != c->device) return -1;
    if ((xo + (size_t)(S - 1) * x_row + (size_t)(n - 1) * x_grp + I) * 4 > x->bytes ||
        (yo + (size_t)(S - 1) * y_row + (size_t)n * O) * 4 > y->bytes) return 0;
    const int r = coli_cuda_tensor_gemm_grp_async(t, n, y->d + yo, y_row, x->d + xo, x_row, x_grp, S, (void *)c->stream);
    if (r == 0) { cc_mark_lost(c); return 0; }
    if (r < 0) return -1;
    g_st.matmuls++; g_st.ops++;
    if (coli_sycl_fprof_flag) {
        char k[64];
        std::snprintf(k, sizeof k, "matmul %d x %dx%d fmt%d (grouped)", n, O, I, fmt);
        coli_sycl_fprof_name(k);
    }
    return 1;
}
static int cc_addnorm(CcBuf *x, CcBuf *t, CcBuf *w, CcBuf *y, const CcAddNorm *p) {
    CcCtx *c = live();
    CC_OWN(c, "cc_addnorm", x, t, w, y);
    if (!c || !x || !t || !w || !y || !p || p->rows < 1 || p->D < 1) return 0;
    static int on = -1;
    if (on < 0) { const char *e = getenv("COLI_CUDA_ADDNORM"); on = !(e && *e == '0'); }
    if (!on || p->D > 256 * 32) return -1;
    if ((size_t)(p->x_off + (size_t)p->rows * p->D) * 4 > x->bytes || (size_t)(p->t_off + (size_t)p->rows * p->D) * 4 > t->bytes ||
        (size_t)(p->y_off + (size_t)p->rows * p->D) * 4 > y->bytes) return 0;
    const int nv = (p->D + 255) / 256;
    float *xp = fpw(x), *yp = fpw(y); const float *tp = fp(t), *wp = fp(w); const CcAddNorm pp = *p;
    const sycl::range<3> g = dim3((unsigned)p->rows);
    #define CC_AN(NV_) CC_K(c, g, 256, cc_addnorm_k<NV_>(xp, tp, wp, yp, pp))
    if (nv <= 8) CC_AN(8); else if (nv <= 16) CC_AN(16); else if (nv <= 24) CC_AN(24); else CC_AN(32);
    #undef CC_AN
    return launched(c, "addnorm");
}
static int cc_matmul_glu(ColiCudaTensor *wg, ColiCudaTensor *wu, CcBuf *x, CcBuf *y, int S) {
    CcCtx *c = live();
    CC_OWN(c, "cc_matmul_glu", x, y);
    if (!c || !wg || !wu || !x || !y || S < 1) return 0;
    int fmt, I, O, fu, Iu, Ou;
    if (!coli_cuda_tensor_shape(wg, &fmt, &I, &O) || !coli_cuda_tensor_shape(wu, &fu, &Iu, &Ou) ||
        coli_cuda_tensor_device(wg) != c->device || coli_cuda_tensor_device(wu) != c->device) return 0;
    if (fu != fmt || Iu != I || Ou != O) return -1;
    if ((size_t)S * I * 4 > x->bytes || (size_t)S * O * 4 > y->bytes) return 0;
    if (!coli_cuda_tensor_glu_async(wg, wu, y->d, x->d, S, (void *)c->stream)) return -1;   /* the two matmuls then */
    g_st.matmuls += 2; g_st.ops++;
    if (coli_sycl_fprof_flag) coli_sycl_fprof_name("matmul_glu");
    return 1;
}
static int cc_norm_nv20(void) {   /* read at each launch: the test runs both kernels in one process */
    const char *e = getenv("COLI_SYCL_NORM_NV");
    return !(e && atoi(e) == 8);
}
static int cc_norm(CcBuf *x, CcBuf *w, CcBuf *y, const CcNorm *p) {
    CcCtx *c = live();
    CC_OWN(c, "cc_norm", x, w, y);
    if (!c || !x || !y || !p || p->nseg < 1 || p->D < 1 || p->per_row < 1) return 0;
    if (!(p->flags & CC_NORM_NOW) && !w) return 0;
    const float *xp = fp(x), *wp = fp(w); float *yp = fpw(y); const CcNorm pp = *p;
    if (p->D > 8 * 256 && p->D <= 20 * 256 && cc_norm_nv20()) CC_K(c, dim3(grid1((size_t)p->nseg, 1)), 256, cc_norm_k<20>(xp, wp, yp, pp));
    else CC_K(c, dim3(grid1((size_t)p->nseg, 1)), 256, cc_norm_k<8>(xp, wp, yp, pp));
    return launched(c, "norm");
}
static int cc_rope(CcBuf *x, CcBuf *cs, const CcRope *p) {
    CcCtx *c = live();
    CC_OWN(c, "cc_rope", x, cs);
    if (!c || !x || !cs || !p || p->nseg < 1 || p->half_ < 1 || p->per_row < 1) return 0;
    float *xp = fpw(x); const float *cp = fp(cs); const CcRope pp = *p;
    CC_K(c, dim3(grid1((size_t)p->nseg * p->half_, 64)), 64, cc_rope_k(xp, cp, pp));
    return launched(c, "rope");
}
static int cc_rope_f16(CcBuf *x, CcBuf *v, CcBuf *cs, const CcRope *p, int seg) {
    CcCtx *c = live();
    CC_OWN(c, "cc_rope_f16", x, v, cs);
    if (!c || !x || !v || !cs || !p || p->nseg < 1 || p->half_ < 1 || p->per_row < 1 || seg < 2 * p->half_) return 0;
    float *xp = fpw(x), *vp = fpw(v); const float *cp = fp(cs); const CcRope pp = *p;
    CC_K(c, dim3(grid1((size_t)p->nseg * seg, 128)), 128, cc_rope_f16_k(xp, vp, cp, pp, seg));
    return launched(c, "rope_f16");
}
/* A decode's attention split over its positions (cc_attn: up to 4 rows, no selection
 * list, from CC_ATTN_DEC_MIN positions; chain_attn_dec.comp's way): cc_attn_k gives a
 * (head, row) one block that walks every position in turn, a decode step's few rows
 * leaving the card all but idle. Mode 0: a block a (head, block of L positions, row):
 * the block's scores (a warp a position, its lanes over the head dim), their max m, sum l
 * of exp(score - m) and acc[d] = sum exp(score - m) V[t][d] into the scratch; mode 1: a
 * block a (head, row): the blocks joined, acc / l, the gate. COLI_CUDA_ATTN_DEC=0 never. */
#define CC_ATTN_DEC_MIN 128
/* the positions a block of cc_attn_dec_k takes for a row of n positions over H heads, as a
 * decode step of that row alone chooses it: 64, doubled to 256 at most while there would be
 * more than 1024 blocks */
/* COLI_CUDA_ATTN_DEC_L: the first block length (64; 16 / 32 give a short context more blocks on a wide
 * device), COLI_CUDA_ATTN_DEC_MIN: the positions from which plain rows split (1 by default;
 * =128 restores the old threshold). Selection-list rows keep CC_ATTN_DEC_MIN and 64-position blocks. */
static inline int cc_attn_dec_len0(void) {
    static int v = -1;
    if (v < 0) { const char *e = getenv("COLI_CUDA_ATTN_DEC_L"); v = e && *e ? atoi(e) : 64; if (v < 16 || v > 256 || (v & (v - 1))) v = 64; }
    return v;
}
static inline int cc_attn_dec_min(void) {
    static int v = -1;
    if (v < 0) { const char *e = getenv("COLI_CUDA_ATTN_DEC_MIN"); v = e && *e ? atoi(e) : 1; if (v < 1) v = 1; }
    return v;
}
static inline int cc_attn_dec_len(int n, int H) {   /* H's high half: the first block length (the host's hc carries it; 0 = 64) */
    int L = H >> 16; if (!L) L = 64; H &= 0xffff;
    int nb = (n + L - 1) / L;
    while (L < 256 && H * nb > 1024) { L *= 2; nb = (n + L - 1) / L; }
    return L;
}
#define CC_ATTN_DEC_ROWS 8   /* a decode step's or a verify's rows (every engine's verify) */
template <typename KT, int SG>
static void cc_attn_dec_k(
    const float *__restrict__ q, const KT *__restrict__ kc,
    const KT *__restrict__ vc, float *__restrict__ o,
    const float *__restrict__ gt, float *__restrict__ xs, CcAttn p, int mode,
    int nb, int win = 0, const float *__restrict__ snk = nullptr,
    int sink_off = 0, int ring = 0,
    int hc = 0, /* hc: the blocks' rule's heads (cc_attn_dec_len), 0 = H */
    const int *__restrict__ sel =
        nullptr) { /* sel: the rows' selection lists (CcAttn.sel_row), as
                      cc_attn_k reads them */
    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto &qs = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[256]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &sc = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[256]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &pt = *sycl::ext::oneapi::group_local_memory_for_overwrite<int[256]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    constexpr int W = 256 / SG;   /* the block's warps */
    auto &red = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[W]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    int tid = it.get_local_id(2), wp = tid / SG, lane = tid & (SG - 1), h = it.get_group(2), hd = p.hd, stride = hd + 2;
    if ((hc & 0xffff) < 1) hc |= p.H;   /* the low half: the rule's heads (0 = H); the high half the first block length */
    if (mode == 0) {
        int b = it.get_group(1), s = it.get_group(0),
            kvh = h / (p.H / p.KVH);
        int n = p.pos_base + s + 1, lb = 0;
        bool list = false;
        if (sel && p.sel_row > 0) { lb = p.sel_off + s * p.sel_row; int c = sel[lb]; if (c >= 0) { n = c; list = true; } }
        int first = !list && win > 0 && n > win ? n - win : 0;   /* a window: its last win positions */
        int L = cc_attn_dec_len(n - first, hc), t0 = first + b * L,
            cnt = sycl::min(L, n - t0);
        size_t xo = ((size_t)(s * p.H + h) * nb + b) * stride;
        if (cnt <= 0) {   /* past the row's positions: an empty block */
            if (tid == 0) { xs[xo] = -3.0e38f; xs[xo + 1] = 0.f; }
            for (int d = tid; d < hd; d += 256) xs[xo + 2 + d] = 0.f;
            return;
        }
        int qb = p.q_off + s * p.q_row + h * p.q_seg;
        for (int d = tid; d < hd; d += 256) qs[d] = q[qb + d];
        it.barrier();
        int kbase = kvh * p.cap;
        for (int j = tid; j < cnt; j += 256) { int t = list ? sel[lb + 1 + t0 + j] : t0 + j; pt[j] = ring ? t % ring : t; }
        it.barrier();
        for (int j = wp; j < cnt; j += W) {
            const KT *kr = kc + p.k_off + (size_t)(kbase + pt[j]) * hd;
            float a = 0.f;
            for (int d = lane; d < hd; d += SG) a += qs[d] * cc_kvf(kr, d);
            for (int m = SG / 2; m > 0; m >>= 1) a += cc_shfl_xor(a, m);
            if (lane == 0) sc[j] = a * p.scale;
        }
        it.barrier();
        float mx = -3.0e38f;
        for (int j = tid; j < cnt; j += 256) mx = sycl::fmax(mx, sc[j]);
        for (int m = SG / 2; m > 0; m >>= 1) mx = sycl::fmax(mx, cc_shfl_xor(mx, m));
        if (lane == 0) red[wp] = mx;
        it.barrier();
        float M = red[0];
        for (int g = 1; g < W; g++) M = sycl::fmax(M, red[g]);
        it.barrier();
        float e = 0.f;
        for (int j = tid; j < cnt; j += 256) {
            float w = sycl::native::exp(sc[j] - M); sc[j] = w; e += w;
        }
        for (int m = SG / 2; m > 0; m >>= 1) e += cc_shfl_xor(e, m);
        if (lane == 0) red[wp] = e;
        it.barrier();
        float l = 0.f;
        for (int g = 0; g < W; g++) l += red[g];
        for (int d = tid; d < hd; d += 256) {
            float acc = 0.f;
            for (int j = 0; j < cnt; j++) acc += sc[j] * cc_kvf(vc, p.v_off + (size_t)(kbase + pt[j]) * hd + d);
            xs[xo + 2 + d] = acc;
        }
        if (tid == 0) { xs[xo] = M; xs[xo + 1] = l; }
        return;
    }
    /* the row's blocks joined: their max M (in any order: the same bits) with the sink's
     * logit; each block's weight w = exp(m_b - M) once and l_b * w, 128 blocks at a time in
     * sc (not a weight a value, each thread's loads one after the other); the sums over the
     * blocks in block order, l's by thread 0 from sc, the values' loads four at a time, so the
     * bits are a weight a value's (chain_attn_dec.comp's mode 1). hd <= 256 (qs) */
    int s = it.get_group(1), n = p.pos_base + s + 1;
    bool list = false;
    if (sel && p.sel_row > 0) { int c = sel[p.sel_off + s * p.sel_row]; if (c >= 0) { n = c; list = true; } }
    int first = !list && win > 0 && n > win ? n - win : 0;
    int L = cc_attn_dec_len(n - first, hc), nbr = (n - first + L - 1) / L;
    size_t xo = (size_t)(s * p.H + h) * nb * stride;
    float mm = -3.0e38f;
    for (int b = tid; b < nbr; b += 256)
        mm = sycl::fmax(mm, xs[xo + (size_t)b * stride]);
    for (int m = SG / 2; m > 0; m >>= 1) mm = sycl::fmax(mm, cc_shfl_xor(mm, m));
    if (lane == 0) red[wp] = mm;
    it.barrier();
    float M = snk ? snk[sink_off + h] : -3.0e38f;   /* the sink joins the max and the sum, with no value */
    for (int g = 0; g < W; g++) M = sycl::fmax(M, red[g]);
    float Lt = snk ? sycl::native::exp(snk[sink_off + h] - M) : 0.f, v = 0.f;
    for (int b0 = 0; b0 < nbr; b0 += 128) {
        int nc = sycl::min(128, nbr - b0);
        it.barrier(); /* red, then the last chunk's weights, read */
        if (tid < nc) {
            size_t xb = xo + (size_t)(b0 + tid) * stride;
            float w = sycl::native::exp(xs[xb] - M);
            sc[tid] = w;
            sc[128 + tid] = xs[xb + 1] * w;
        }
        it.barrier();
        if (tid == 0) for (int b = 0; b < nc; b++) Lt += sc[128 + b];
        if (tid < hd) {
            const float *xv = xs + xo + (size_t)b0 * stride + 2 + tid;
            int b = 0;
            for (; b + 4 <= nc; b += 4) {
                float x0 = xv[(size_t)b * stride], x1 = xv[(size_t)(b + 1) * stride], x2 = xv[(size_t)(b + 2) * stride], x3 = xv[(size_t)(b + 3) * stride];
                v += x0 * sc[b]; v += x1 * sc[b + 1]; v += x2 * sc[b + 2]; v += x3 * sc[b + 3];
            }
            for (; b < nc; b++) v += xv[(size_t)b * stride] * sc[b];
        }
    }
    if (tid == 0) red[0] = Lt;
    it.barrier();
    Lt = red[0];
    float inv = Lt > 0.f ? 1.f / Lt : 0.f;
    int ob = p.o_off + s * p.o_row + h * hd, gb = p.g_off + s * p.g_row + h * p.g_seg;
    if (tid < hd) {
        v *= inv;
        if (p.has_gate) v *= 1.f / (1.f + sycl::native::exp(-gt[gb + tid]));
        o[ob + tid] = v;
    }
}
/* cc_attn_dec_k's mode 0 for the G query heads of a KV head at once: a block a (KV head, block
 * of positions, row), each K and V row read once for the G heads instead of once a head, four
 * floats a load (chain_attn_dec.comp's mode 2). Qwen3.6's eight heads a KV head after an 18K
 * prompt read 74 MB of rows a layer eight times. The scores a thread or two a position; V a float4
 * column a thread, the positions over 256 / (hd / 4) groups of threads, their sums added group by group.
 * The blocks' rule counts four blocks a KV head (hc); the join is cc_attn_dec_k's mode 1 with
 * that rule. Dynamic shared memory: the heads' q rows and scores, G * (hd + 256) floats. SG: the warp's
 * lanes (256 / SG warps a block, 64 / SG float4s of a K row a lane). */
template <int G, typename KT, int SG>
static void cc_attn_decg_k(const float *__restrict__ q,
                           const KT *__restrict__ kc, const KT *__restrict__ vc,
                           float *__restrict__ xs, CcAttn p, int nb, int win,
                           int ring, int hc, int tpp, uint8_t *slm) {
    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr int W = 256 / SG;   /* the block's warps */
    auto cc_decg_sm = (sycl::float4 *)slm;
    auto &red =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[G * W]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int hd = p.hd, stride = hd + 2, NV = hd / 4, NG = 256 / NV;
    sycl::float4 *qv =
        cc_decg_sm; /* G * NV: the heads' q rows, then the groups' V sums */
    float *sc = (float *)(cc_decg_sm + G * NV);    /* G * 256 */
    int tid = it.get_local_id(2), wp = tid / SG, lane = tid & (SG - 1), kvh = it.get_group(2), b = it.get_group(1), s = it.get_group(0);
    int n = p.pos_base + s + 1, first = win > 0 && n > win ? n - win : 0;
    int L = cc_attn_dec_len(n - first, hc), t0 = first + b * L,
        cnt = sycl::min(L, n - t0);
    size_t xo0 = ((size_t)(s * p.H + kvh * G) * nb + b) * stride, xg = (size_t)nb * stride;   /* head kvh * G + g at xo0 + g * xg */
    if (cnt <= 0) {
        if (tid < G) { xs[xo0 + tid * xg] = -3.0e38f; xs[xo0 + tid * xg + 1] = 0.f; }
        for (int i = tid; i < G * hd; i += 256) xs[xo0 + (i / hd) * xg + 2 + i % hd] = 0.f;
        return;
    }
    int qb = p.q_off + s * p.q_row + kvh * G * p.q_seg;
    for (int i = tid; i < G * NV; i += 256) {
        int g = i / NV, c = i - g * NV;
        const float *x = q + qb + g * p.q_seg + 4 * c;
        qv[i] = sycl::float4(x[0], x[1], x[2], x[3]);
    }
    it.barrier();
    int kbase = kvh * p.cap;
    /* the scores. tpp 0: a K row over a warp's lanes, 64 / SG float4s a lane, the heads' sums by
     * shuffles; tpp 1, 2 or 4: that many threads a position (adjacent lanes, a float4 each in
     * turn along the K row) against the G q rows, their sums joined across the tpp lanes, the max
     * on the way (chain_attn_dec.comp's mode 2), the loop's trips the block's so every lane meets
     * the shuffle. cc_decg_tpp chooses */
    float M[G], e[G];
#pragma unroll
    for (int g = 0; g < G; g++) M[g] = -3.0e38f;
    if (tpp == 0) {
        for (int j = wp; j < cnt; j += W) {
            size_t kr = p.k_off + (size_t)(kbase + (ring ? (t0 + j) % ring : t0 + j)) * hd;
            sycl::float4 kk[64 / SG];
#pragma unroll
            for (int i = 0; i < 64 / SG; i++) {
                int c = lane + SG * i;
                kk[i] = c < NV ? cc_kv4(kc, kr + 4 * c)
                               : sycl::float4(0.f, 0.f, 0.f, 0.f);
            }
            float a[G];
#pragma unroll
            for (int g = 0; g < G; g++) a[g] = 0.f;
#pragma unroll
            for (int i = 0; i < 64 / SG; i++) {
                int c = lane + SG * i;
                if (c < NV)
#pragma unroll
                    for (int g = 0; g < G; g++) {
                        sycl::float4 x = qv[g * NV + c];
                        a[g] += x.x() * kk[i].x() + x.y() * kk[i].y() +
                                x.z() * kk[i].z() + x.w() * kk[i].w();
                    }
            }
#pragma unroll
            for (int g = 0; g < G; g++) {
                float x = a[g];
                for (int m = SG / 2; m > 0; m >>= 1) x += cc_shfl_xor(x, m);
                if (lane == 0) sc[g * 256 + j] = x * p.scale;
            }
        }
        it.barrier();
        for (int j = tid; j < cnt; j += 256)
#pragma unroll
            for (int g = 0; g < G; g++)
                M[g] = sycl::fmax(M[g], sc[g * 256 + j]);
    } else {
        int part = tid % tpp;
        for (int j0 = 0; j0 < cnt; j0 += 256 / tpp) {
            int j = j0 + tid / tpp;
            float a[G];
#pragma unroll
            for (int g = 0; g < G; g++) a[g] = 0.f;
            if (j < cnt) {
                size_t kr = p.k_off + (size_t)(kbase + (ring ? (t0 + j) % ring : t0 + j)) * hd;
#pragma unroll 4
                for (int c = part; c < NV; c += tpp) {
                    sycl::float4 kk = cc_kv4(kc, kr + 4 * c);
#pragma unroll
                    for (int g = 0; g < G; g++) {
                        sycl::float4 x = qv[g * NV + c];
                        a[g] += x.x() * kk.x() + x.y() * kk.y() +
                                x.z() * kk.z() + x.w() * kk.w();
                    }
                }
            }
            for (int m = 1; m < tpp; m <<= 1)
#pragma unroll
                for (int g = 0; g < G; g++) a[g] += cc_shfl_xor(a[g], m);
            if (j < cnt)
#pragma unroll
                for (int g = 0; g < G; g++) {
                    float x = a[g] * p.scale;
                    if (part == 0) sc[g * 256 + j] = x;
                    M[g] = sycl::fmax(M[g], x);
                }
        }
    }
#pragma unroll
    for (int g = 0; g < G; g++) {
        float x = M[g];
        for (int m = SG / 2; m > 0; m >>= 1) x = sycl::fmax(x, cc_shfl_xor(x, m));
        if (lane == 0) red[g * W + wp] = x;
    }
    it.barrier();
#pragma unroll
    for (int g = 0; g < G; g++) {
        M[g] = red[g * W];
        for (int k = 1; k < W; k++) M[g] = sycl::fmax(M[g], red[g * W + k]);
    }
    it.barrier();
#pragma unroll
    for (int g = 0; g < G; g++) e[g] = 0.f;
    for (int j = tid; j < cnt; j += 256)
#pragma unroll
        for (int g = 0; g < G; g++) {
            float w = sycl::native::exp(sc[g * 256 + j] - M[g]);
            sc[g * 256 + j] = w;
            e[g] += w;
        }
#pragma unroll
    for (int g = 0; g < G; g++) {
        float x = e[g];
        for (int m = SG / 2; m > 0; m >>= 1) x += cc_shfl_xor(x, m);
        if (lane == 0) red[g * W + wp] = x;
    }
    it.barrier();
    int c = tid % NV, grp = tid / NV;
    sycl::float4 acc[G];
#pragma unroll
    for (int g = 0; g < G; g++) acc[g] = sycl::float4(0.f, 0.f, 0.f, 0.f);
    /* four positions' loads in flight, then their sums in position order (the bits of one at a time) */
    int j = grp;
    for (; j + 3 * NG < cnt; j += 4 * NG) {
        sycl::float4 v[4];
#pragma unroll
        for (int u = 0; u < 4; u++) { int t = t0 + j + u * NG; v[u] = cc_kv4(vc, p.v_off + (size_t)(kbase + (ring ? t % ring : t)) * hd + 4 * c); }
#pragma unroll
        for (int u = 0; u < 4; u++)
#pragma unroll
            for (int g = 0; g < G; g++) {
                float w = sc[g * 256 + j + u * NG];
                acc[g].x() += w * v[u].x(); acc[g].y() += w * v[u].y();
                    acc[g].z() += w * v[u].z(); acc[g].w() += w * v[u].w();
            }
    }
    for (; j < cnt; j += NG) {
        sycl::float4 v = cc_kv4(
            vc, p.v_off +
                    (size_t)(kbase + (ring ? (t0 + j) % ring : t0 + j)) * hd +
                    4 * c);
#pragma unroll
        for (int g = 0; g < G; g++) {
            float w = sc[g * 256 + j];
            acc[g].x() += w * v.x(); acc[g].y() += w * v.y();
                acc[g].z() += w * v.z(); acc[g].w() += w * v.w();
        }
    }
    for (int k = NG - 1; k >= 1; k--) {   /* the groups' sums into group 0's, the last group first */
        if (grp == k)
#pragma unroll
            for (int g = 0; g < G; g++) {
                sycl::float4 x = qv[g * NV + c];
                qv[g * NV + c] =
                    k == NG - 1
                        ? acc[g]
                        : sycl::float4(x.x() + acc[g].x(), x.y() + acc[g].y(),
                                       x.z() + acc[g].z(), x.w() + acc[g].w());
            }
        it.barrier();
    }
    if (grp == 0)
#pragma unroll
        for (int g = 0; g < G; g++) {
            sycl::float4 v = acc[g];
            if (NG > 1) {
                sycl::float4 x = qv[g * NV + c];
                v = sycl::float4(v.x() + x.x(), v.y() + x.y(), v.z() + x.z(),
                                 v.w() + x.w());
            }
            float *o = xs + xo0 + g * xg + 2 + 4 * c;
            o[0] = v.x(); o[1] = v.y(); o[2] = v.z(); o[3] = v.w();
        }
    if (tid < G) {
        float l = 0.f;
        for (int k = 0; k < W; k++) l += red[tid * W + k];
#pragma unroll
        for (int g = 0; g < G; g++) if (g == tid) { xs[xo0 + g * xg] = M[g]; xs[xo0 + g * xg + 1] = l; }   /* M[] by a constant index */
    }
}
/* the query heads a KV head cc_attn_decg_k takes (0: none, cc_attn_dec_k a head a block);
 * COLI_CUDA_ATTN_DEC_GQA=0 never */
static int cc_decg_heads(int G, int hd) {
    static int on = -1;
    if (on < 0) { const char *e = getenv("COLI_CUDA_ATTN_DEC_GQA"); on = !(e && *e == '0'); }
    return on && hd >= 4 && hd <= 256 && hd % 4 == 0 && (G == 2 || G == 4 || G == 8 || G == 16) ? G : 0;
}
/* cc_attn_decg_k's scores: two threads a position, COLI_CUDA_ATTN_DEC_TPP (0, 1, 2 or 4) sets it.
 * One layer after 18K positions in float16 on an RTX PRO 4000 (tests/test_cuda_chain.cu's
 * COLI_CUDA_CHAIN_BENCH=adec), at 0 / 1 / 2 / 4: Qwen3.6's 16/2 heads of 256 115.0 / 128.1 /
 * 108.6 / 108.5 us, Flint's 32/4 of 128 140.5 / 146.2 / 122.1 / 121.9 */
static int cc_decg_tpp(void) {
    static int v = -1;
    if (v < 0) { const char *e = getenv("COLI_CUDA_ATTN_DEC_TPP"); v = e && (*e == '0' || *e == '1' || *e == '2' || *e == '4') ? *e - '0' : 2; }
    return v;
}
template <typename KT>
static void cc_decg_launch(CcCtx *c, int G, const float *q, const KT *kc, const KT *vc, float *xs, const CcAttn &p, int nb,
                           int win, int ring, int hc) {
    const sycl::range<3> g = dim3((unsigned)p.KVH, (unsigned)nb, (unsigned)p.S);
    const size_t sm = (size_t)G * (p.hd + 256) * sizeof(float);   /* the heads' q rows and scores */
    const int tpp = cc_decg_tpp();
    const CcAttn pp = p;
    switch (G) {
    case 2: CC_KSG_AT(c, c->sg_attn, g, 256, sm, cc_attn_decg_k<2, KT, SG>(q, kc, vc, xs, pp, nb, win, ring, hc, tpp, slm_)); break;
    case 4: CC_KSG_AT(c, c->sg_attn, g, 256, sm, cc_attn_decg_k<4, KT, SG>(q, kc, vc, xs, pp, nb, win, ring, hc, tpp, slm_)); break;
    case 8: CC_KSG_AT(c, c->sg_attn, g, 256, sm, cc_attn_decg_k<8, KT, SG>(q, kc, vc, xs, pp, nb, win, ring, hc, tpp, slm_)); break;
    default: CC_KSG_AT(c, c->sg_attn, g, 256, sm, cc_attn_decg_k<16, KT, SG>(q, kc, vc, xs, pp, nb, win, ring, hc, tpp, slm_)); break;
    }
}
/* cc_attn_dec_k's scratch for S rows of H heads in nb blocks of hd + 2 floats (the chain's own,
 * kept and grown rarely: a resize retires the old block to cc_adec_reap rather than freeing it
 * under the frames that may still read it, and never waits on the queue, where a cc_wait_host
 * kernel whose host side comes after this frame could be; coli_sycl_scratch's grow retires the
 * same way now, and its blocks are reaped with these at the frame's end) */
static int cc_adec_fit(CcCtx *c, size_t need) {
    if (need <= c->adec_bytes) return c->adec != nullptr;
    if (c->adec) {   /* rare: the context past what it was */
        if (c->adec_n_old == 4) {
            if (!cc_try(c, "attn scratch", [&] { c->stream->wait_and_throw(); })) return 0;
            cc_adec_reap(c);
        }
        c->adec_old[c->adec_n_old++] = c->adec;
    }
    c->adec = nullptr; c->adec_bytes = 0;
    size_t want = need + need / 2;
    c->adec = (float *)cc_alloc(c, want, 0);
    if (c->adec) c->adec_bytes = want;
    return c->adec != nullptr;
}
template <typename KT>
static int cc_attn_t(CcBuf *q, CcBuf *kc, CcBuf *vc, CcBuf *o, CcBuf *gate, CcBuf *sel, const CcAttn *p) {
    CcCtx *c = live();
    CC_OWN(c, "cc_attn_t", q, kc, vc, o, gate, sel);
    if (!c || !q || !kc || !vc || !o || !p || p->S < 1 || p->H < 1 || p->KVH < 1 || p->H % p->KVH || p->hd < 1 || p->hd > 256) return 0;
    if (p->has_gate && !gate) return 0;
    if (p->sel_row > 0 && !sel) return 0;
    const float *qp = fp(q), *gp = fp(gate); const KT *kp = kvp<KT>(kc), *vp = kvp<KT>(vc); float *op = fpw(o);
    const int *sp = sel ? (const int *)sel->d : nullptr;
    /* a prompt chunk's plain causal attention on the matrix engines (cc_attn_xmx), from
     * COLI_CUDA_CHAIN_FLASH rows (16; 0 = never), no selection list */
    static int fl = -1;
    if (fl < 0) { const char *e = getenv("COLI_CUDA_CHAIN_FLASH"); fl = e && *e ? atoi(e) : 16; if (fl < 0) fl = 0; }
    int G = p->H / p->KVH;
    const int xmx = fl > 0 && cc_attn_xmx_ok(c, p, G, sizeof(KT) == 2, 0);
    if (xmx && p->S >= fl && p->sel_row <= 0) {
        if (cc_attn_xmx<KT>(c, q, kc, vc, o, gate, p, G, 0, nullptr, 0, 0, nullptr)) return launched(c, "attn (matrix engines)");
        if (c->lost) return 0;
    }
    /* a decode's or a verify's rows over many positions: split over them (cc_attn_dec_on) */
    int last = p->pos_base + p->S;
    int fk = cc_attn_dec_on(xmx);
    /* a selection list's rows too (Qwen3.8's QSA: 24 heads of a verify's rows, cc_attn_k's block
     * a (head, row) walking up to its budget's positions, 104-144 us a layer on an RTX PRO 4000);
     * their bits then are not those the same rows get in a prompt chunk (cc_attn_gk, cc_attn_k's).
     * COLI_CUDA_ATTN_DEC_SEL=0 keeps cc_attn_k for them (read each call) */
    int lists = 0;
    if (p->sel_row > 0) { const char *e = getenv("COLI_CUDA_ATTN_DEC_SEL"); lists = cc_attn_dec_on(1) && !(e && *e == '0') && sp; }
    const int mn = p->sel_row > 0 ? CC_ATTN_DEC_MIN : cc_attn_dec_min();
    if ((p->sel_row <= 0 ? fk : lists) && p->S <= CC_ATTN_DEC_ROWS && last >= mn) {
        /* each row as a decode step of it alone: its own blocks of positions (cc_attn_dec_len),
         * and the rows whose positions are fewer than mn through cc_attn_k */
        int k = mn - (p->pos_base + 1);
        if (k > 0) {
            CcAttn a = *p; a.S = k;
            CC_K(c, dim3((unsigned)p->H, (unsigned)k), 128, cc_attn_k<KT>(qp, kp, vp, op, gp, sp, a, 0, nullptr, 0, 0));
            if (!launched(c, "attn")) return 0;
            CcAttn b = *p; b.S -= k; b.pos_base += k; b.q_off += k * p->q_row; b.o_off += k * p->o_row; b.g_off += k * p->g_row;
            if (b.sel_row > 0) b.sel_off += k * p->sel_row;
            return cc_attn_t<KT>(q, kc, vc, o, gate, sel, &b);
        }
        /* a KV head's query heads a block where cc_attn_decg_k takes them (offsets of whole float4s):
         * its rule counts four blocks a KV head */
        int gq = p->sel_row <= 0 && p->k_off % 4 == 0 && p->v_off % 4 == 0 ? cc_decg_heads(G, p->hd) : 0, hcr = gq ? 4 * p->KVH : p->H, nb = 1;
        /* the rule's heads, the first block length in the high half (cc_attn_dec_len); a selection list's rows keep 64:
         * their block count below assumes it (COLI_CUDA_ATTN_DEC_L=16 failed their op check, run 6 of perf m-kernels) */
        const int l0 = p->sel_row > 0 ? 64 : cc_attn_dec_len0(), hc = hcr | (l0 << 16);
        for (int r = 0; r < p->S; r++) {
            int n = p->pos_base + r + 1, L = cc_attn_dec_len(n, hc), m = (n + L - 1) / L;
            if (p->sel_row > 0) {   /* a list's length is the device's: the most blocks any length up to the row's can take
                                     * (a row without a list, -1, its causal positions: m above) */
                int nl = n < p->sel_row - 1 ? n : p->sel_row - 1, F = 1024 / hcr > 1 ? 1024 / hcr : 1, a = (nl + 63) / 64;
                int ml = a < F ? a : F;
                if ((nl + 255) / 256 > ml) ml = (nl + 255) / 256;
                if (ml > m) m = ml;
            }
            if (m > nb) nb = m;
        }
        if (cc_adec_fit(c, (size_t)p->S * p->H * nb * (p->hd + 2) * sizeof(float))) {
            float *xs = c->adec; const CcAttn pp = *p; const int *lp = lists ? sp : nullptr;
            const int hc0 = l0 << 16;   /* read on the host: the launch's body runs on the device */
            if (gq) cc_decg_launch(c, gq, qp, kp, vp, xs, *p, nb, 0, 0, hc);
            else CC_KSG_AT(c, c->sg_attn, dim3((unsigned)p->H, (unsigned)nb, (unsigned)p->S), 256, 0,
                        cc_attn_dec_k<KT, SG>(qp, kp, vp, op, gp, xs, pp, 0, nb, 0, nullptr, 0, 0, hc0, lp));
            CC_KSG_AT(c, c->sg_attn, dim3((unsigned)p->H, (unsigned)p->S), 256, 0,
                   cc_attn_dec_k<KT, SG>(qp, kp, vp, op, gp, xs, pp, 1, nb, 0, nullptr, 0, 0, hc, lp));
            return launched(c, "attn (positions in blocks)");
        }
        if (c->lost) return 0;
    }
    /* a prompt chunk's rows with a selection list on the matrix engines (head dim 256), from
     * COLI_CUDA_CHAIN_FLASH rows: the lists' positions ascending, as cc_qsa writes them.
     * cc_attn_gk took 15.9 ms of a 2048-row chunk of Qwen3.8's on an RTX PRO 4000 (CUDA cores, f32).
     * COLI_CUDA_ATTN_SEL_TC=0 keeps cc_attn_gk below. Not bit-identical to it (f16 operands) */
    int stc = 1;   /* read each call, as COLI_CUDA_ATTN_DEC_SEL */
    { const char *e = getenv("COLI_CUDA_ATTN_SEL_TC"); stc = !(e && *e == '0'); }
    if (stc && fl > 0 && p->S >= fl && p->sel_row > 0 && p->hd == 256 && sp && cc_attn_xmx_ok(c, p, G, sizeof(KT) == 2, 1)) {
        if (cc_attn_xmx<KT>(c, q, kc, vc, o, gate, p, G, 0, nullptr, 0, 0, sp)) return launched(c, "attn (matrix engines, a selection list)");
        if (c->lost) return 0;
    }
    /* a prompt chunk's rows with a selection list: a block per (kv head, row), cc_attn_gk, from
     * COLI_CUDA_ATTN_GROUPED rows (32; 0 = never) */
    static int gr = -1;
    if (gr < 0) { const char *e = getenv("COLI_CUDA_ATTN_GROUPED"); gr = e && *e ? atoi(e) : 32; if (gr < 0) gr = 0; }
    /* plain rows (no selection list) through cc_attn_gk too, from COLI_CUDA_ATTN_GROUPED_PLAIN rows
     * (0 = never, the code's default; unset on a Data Center GPU Max: 32, measured there --
     * coli_cuda_tuned_default): the same bits as cc_attn_k, a KV head's keys and values read once
     * for its query heads */
    static int gpl = -1;
    if (gpl < 0) {
        const char *e = getenv("COLI_CUDA_ATTN_GROUPED_PLAIN");
        gpl = e && *e ? atoi(e) : coli_cuda_tuned_default("COLI_CUDA_ATTN_GROUPED_PLAIN", 0);
        if (gpl < 0) gpl = 0;
    }
    const CcAttn pp = *p;
    const int grouped = p->sel_row > 0 ? (gr > 0 && p->S >= gr) : (gpl > 0 && p->S >= gpl);
    if (grouped && p->hd <= 256) {
        const sycl::range<3> gg = dim3((unsigned)p->KVH, (unsigned)p->S);
        /* plain rows of eight query heads a kv head, whole eights: the kernel with the vector loads, from
         * COLI_CUDA_ATTN_GROUPED_VEC=1 (0, the code's default: cc_attn_gk; unset on a Data Center GPU Max: 2,
         * measured there -- coli_cuda_tuned_default); the same bits */
        static int gv = -1;
        if (gv < 0) {
            const char *e = getenv("COLI_CUDA_ATTN_GROUPED_VEC");
            gv = e && *e ? atoi(e) : coli_cuda_tuned_default("COLI_CUDA_ATTN_GROUPED_VEC", 0);
        }
        if (gv > 0 && p->sel_row <= 0 && G == 8 && (p->hd & 7) == 0 && (p->k_off & 7) == 0) {
            /* =2: the values two adjacent dimensions a load as well (v_off even); the same bits */
            if (gv >= 2 && (p->v_off & 1) == 0) {
                CC_K(c, gg, 128, (cc_attn_g8v<KT, 1>(qp, kp, vp, op, gp, pp)));
                return launched(c, "attn (a kv head's query heads a block, vector loads, value pairs)");
            }
            CC_K(c, gg, 128, cc_attn_g8v<KT>(qp, kp, vp, op, gp, pp));
            return launched(c, "attn (a kv head's query heads a block, vector loads)");
        }
#define CC_ATTN_G(GG) case GG: CC_K(c, gg, 128, cc_attn_gk<GG, KT>(qp, kp, vp, op, gp, sp, pp)); \
                      return launched(c, "attn (a kv head's query heads a block)");
        switch (G) { CC_ATTN_G(2) CC_ATTN_G(4) CC_ATTN_G(6) CC_ATTN_G(8) CC_ATTN_G(12) CC_ATTN_G(16) default: break; }
#undef CC_ATTN_G
    }
    CC_K(c, dim3((unsigned)p->H, (unsigned)p->S), 128, cc_attn_k<KT>(qp, kp, vp, op, gp, sp, pp, 0, nullptr, 0, 0));
    return launched(c, "attn");
}
static int cc_attn(CcBuf *q, CcBuf *kc, CcBuf *vc, CcBuf *o, CcBuf *gate, CcBuf *sel, const CcAttn *p) {
    CC_OWN(cur(), "cc_attn", q, kc, vc, o, gate, sel);
    return cc_attn_t<float>(q, kc, vc, o, gate, sel, p);
}
static int cc_dnconv(CcBuf *in, CcBuf *w, CcBuf *ring, CcBuf *out, CcBuf *snap, const CcDnConv *p) {
    CcCtx *c = live();
    CC_OWN(c, "cc_dnconv", in, w, ring, out, snap);
    if (!c || !in || !w || !ring || !out || !p || p->S < 1 || p->CD < 1 || p->CK < 2 || p->CK > 9) return 0;
    if (p->snap_row >= 0 && !snap) return 0;
    const float *ip = fp(in), *wp = fp(w); float *rp = fpw(ring), *op = fpw(out), *sp = fpw(snap); const CcDnConv pp = *p;
    if (p->S > 1 && p->S < 65536) {   /* a prompt's rows apart, then the ring past them */
        const float *rc = rp;
        CC_K(c, dim3((unsigned)((p->CD + 127) / 128), (unsigned)p->S), 128, cc_dnconv_rows_k(ip, wp, rc, op, pp));
        CC_K(c, dim3(grid1((size_t)p->CD, 128)), 128, cc_dnconv_ring_k(ip, rp, sp, pp));
        return launched(c, "dnconv");
    }
    CC_K(c, dim3(grid1((size_t)p->CD, 64)), 64, cc_dnconv_k(ip, wp, rp, op, sp, pp));
    return launched(c, "dnconv");
}
static int cc_dnrec(int KD, CcBuf *cv, CcBuf *ab, CcBuf *z, CcBuf *st, CcBuf *prm, CcBuf *y, CcBuf *snap, const CcDnRec *p) {
    CcCtx *c = live();
    CC_OWN(c, "cc_dnrec", cv, ab, z, st, prm, y, snap);
    if (!c || !cv || !ab || !z || !st || !prm || !y || !p || p->S < 1 || p->VH < 1 || p->KH < 1 || p->VH % p->KH ||
        p->VD < 1 || p->VD > 128) return 0;
    if (p->snap_row >= 0 && !snap) return 0;
    const float *a = fp(cv), *b = fp(ab), *zz = fp(z), *pr = fp(prm);
    float *s = fpw(st), *yy = fpw(y), *sn = fpw(snap);
    const CcDnRec pp = *p;
    const sycl::range<3> gh = dim3((unsigned)p->VH);
    static int rows_on = -1;
    if (rows_on < 0) { const char *e = getenv("COLI_CUDA_DNREC_ROWS"); rows_on = !(e && *e == '0'); }
    static int split_on = -1;
    if (split_on < 0) { const char *e = getenv("COLI_CUDA_DNREC_SPLIT"); split_on = !(e && *e == '0'); }
    /* COLI_CUDA_DNREC_SPLIT=0: cc_dnrec_rows_k. COLI_CUDA_DNREC_CFG=PC (tests): P threads over C columns */
    static const int cfg_override = [] { const char *e = getenv("COLI_CUDA_DNREC_CFG"); return e ? atoi(e) : -1; }();
    const int cfg = cfg_override >= 0 ? cfg_override :
        (coli_sycl_profile_for(c->device).arch == COLI_SYCL_ARCH_PVC ? 162 : 84);
    /* COLI_CUDA_DNREC_MIN: the rows from which the prompt's kernels are taken (CC_DNREC_ROWS; 1 takes them for
     * a decode row too: perf m-kernels measured cc_dnrec_k's one block a head against them, MIGRATION 14) */
    static int rows_min = -1;
    if (rows_min < 0) { const char *e = getenv("COLI_CUDA_DNREC_MIN"); rows_min = e && *e ? atoi(e) : CC_DNREC_ROWS; if (rows_min < 1) rows_min = 1; }
    if (rows_on && split_on && p->S >= rows_min && (KD == 64 || KD == 128) && p->VD % 4 == 0 &&
        p->cv_row % 4 == 0 && p->cv_off % 4 == 0 && p->Ktot % 4 == 0) {
        /* the split kernel's own sub-group (COLI_CUDA_DNREC_PROMPT_SG=16|32; unset: 16 on Data Center GPU Max cards,
         * cc_kernel_sg, else the context's, as before). It
         * holds C x KD / P state columns a lane in registers, and its sums are the P lanes' shuffles whatever the
         * sub-group, so the width changes its code and not its bits; cc_dnprep_k and cc_dnnorm_k sum over a whole
         * sub-group and keep the context's */
        const int sgp = c->sg_dnp;
        const int cf = cfg / 10 > sgp ? 84 : cfg;   /* the P lanes of a column join by shuffles: within a sub-group */
        const int P = cf / 10, C = cf % 10, cb = 128 / (P ? P : 1) * C;   /* a block's columns */
        const size_t nq = (size_t)p->S * 2 * p->KH * KD, nbd = (size_t)p->S * p->VH * 2;
        /* the rows' q, k, beta and decay: the device's arena (coli_sycl_scratch, slot 1), persistent;
         * a larger chunk grows it with headroom, the old block retired until the frame's end (no
         * wait inside the frame); the rows kernel below when it has no room */
        float *qk = (float *)coli_sycl_scratch(c->device, (nq + nbd) * sizeof(float), 1);
        if (qk) {
            const int W = 128 / c->sg, nvb = (p->S * 2 * p->KH + W - 1) / W, nbb = (p->S * p->VH + 127) / 128;   /* a warp a vector */
            float *bd = qk + nq;
            CC_KSG(c, dim3((unsigned)(nvb + nbb)), 128, 0, cc_dnprep_k<SG>(a, b, pr, qk, bd, KD, nvb, pp));
            const sycl::range<3> g = dim3((unsigned)p->VH, (unsigned)((p->VD + cb - 1) / cb));
            const float *qkc = qk, *bdc = bd;
            #define CC_DS(P_, C_) do { if (KD == 64) CC_KSG_AT(c, sgp, g, 128, 0, cc_dnrec_split_k<64, 16, P_, C_>(a, qkc, bdc, s, yy, sn, pp)); \
                                       else CC_KSG_AT(c, sgp, g, 128, 0, cc_dnrec_split_k<128, 16, P_, C_>(a, qkc, bdc, s, yy, sn, pp)); } while (0)
            switch (cf) {
            case 81: CC_DS(8, 1); break;   case 82: CC_DS(8, 2); break;   case 84: CC_DS(8, 4); break;
            case 162: CC_DS(16, 2); break; case 41: CC_DS(4, 1); break;
            case 42: CC_DS(4, 2); break;   default: CC_DS(8, 4); break;
            }
            #undef CC_DS
            if (!launched(c, "dnrec split")) return 0;
            CC_KSG(c, dim3((unsigned)((p->S * p->VH + W - 1) / W)), 128, 0, cc_dnnorm_k<SG>(zz, pr, yy, pp));
            return launched(c, "dnrec norm");
        }
    }
    if (rows_on && p->S >= rows_min) {   /* a prompt chunk's rows, a block at a time */
        switch (KD) {
        case 4:   CC_KSG(c, gh, 128, 0, cc_dnrec_rows_k<4, 16, SG>(a, b, zz, s, pr, yy, sn, pp)); return launched(c, "dnrec rows");
        case 8:   CC_KSG(c, gh, 128, 0, cc_dnrec_rows_k<8, 16, SG>(a, b, zz, s, pr, yy, sn, pp)); return launched(c, "dnrec rows");
        case 16:  CC_KSG(c, gh, 128, 0, cc_dnrec_rows_k<16, 16, SG>(a, b, zz, s, pr, yy, sn, pp)); return launched(c, "dnrec rows");
        case 32:  CC_KSG(c, gh, 128, 0, cc_dnrec_rows_k<32, 16, SG>(a, b, zz, s, pr, yy, sn, pp)); return launched(c, "dnrec rows");
        case 64:  CC_KSG(c, gh, 128, 0, cc_dnrec_rows_k<64, 16, SG>(a, b, zz, s, pr, yy, sn, pp)); return launched(c, "dnrec rows");
        case 128: CC_KSG(c, gh, 128, 0, cc_dnrec_rows_k<128, 16, SG>(a, b, zz, s, pr, yy, sn, pp)); return launched(c, "dnrec rows");
        case 256: CC_KSG(c, gh, 128, 0, cc_dnrec_rows_k<256, 8, SG>(a, b, zz, s, pr, yy, sn, pp)); return launched(c, "dnrec rows");
        default: return 0;
        }
    }
    /* Few rows (decode or verify) use the same row kernel at the recurrence's own width.
     * Prompt dispatch above is unchanged; DEC=0 restores cc_dnrec_k for these rows. */
    static int dec_on = -1;
    if (dec_on < 0) { const char *e = getenv("COLI_CUDA_DNREC_DEC"); dec_on = !(e && *e == '0'); }
    if (rows_on && dec_on && p->S < rows_min) {
        switch (KD) {
        case 4:   CC_KSG_AT(c, c->sg_dn, gh, 128, 0, cc_dnrec_rows_k<4, 16, SG>(a, b, zz, s, pr, yy, sn, pp)); break;
        case 8:   CC_KSG_AT(c, c->sg_dn, gh, 128, 0, cc_dnrec_rows_k<8, 16, SG>(a, b, zz, s, pr, yy, sn, pp)); break;
        case 16:  CC_KSG_AT(c, c->sg_dn, gh, 128, 0, cc_dnrec_rows_k<16, 16, SG>(a, b, zz, s, pr, yy, sn, pp)); break;
        case 32:  CC_KSG_AT(c, c->sg_dn, gh, 128, 0, cc_dnrec_rows_k<32, 16, SG>(a, b, zz, s, pr, yy, sn, pp)); break;
        case 64:  CC_KSG_AT(c, c->sg_dn, gh, 128, 0, cc_dnrec_rows_k<64, 16, SG>(a, b, zz, s, pr, yy, sn, pp)); break;
        case 128: CC_KSG_AT(c, c->sg_dn, gh, 128, 0, cc_dnrec_rows_k<128, 16, SG>(a, b, zz, s, pr, yy, sn, pp)); break;
        case 256: CC_KSG_AT(c, c->sg_dn, gh, 128, 0, cc_dnrec_rows_k<256, 8, SG>(a, b, zz, s, pr, yy, sn, pp)); break;
        default: return 0;
        }
        return launched(c, "dnrec rows");
    }
    switch (KD) {
    case 4:   CC_K(c, gh, 128, cc_dnrec_k<4>(a, b, zz, s, pr, yy, sn, pp)); break;
    case 8:   CC_K(c, gh, 128, cc_dnrec_k<8>(a, b, zz, s, pr, yy, sn, pp)); break;
    case 16:  CC_K(c, gh, 128, cc_dnrec_k<16>(a, b, zz, s, pr, yy, sn, pp)); break;
    case 32:  CC_K(c, gh, 128, cc_dnrec_k<32>(a, b, zz, s, pr, yy, sn, pp)); break;
    case 64:  CC_K(c, gh, 128, cc_dnrec_k<64>(a, b, zz, s, pr, yy, sn, pp)); break;
    case 128: CC_K(c, gh, 128, cc_dnrec_k<128>(a, b, zz, s, pr, yy, sn, pp)); break;
    case 256: CC_K(c, gh, 128, cc_dnrec_k<256>(a, b, zz, s, pr, yy, sn, pp)); break;
    default: return 0;
    }
    return launched(c, "dnrec");
}
static int cc_ew(CcBuf *y, CcBuf *a, CcBuf *b, CcBuf *cb, CcBuf *e, const CcEw *p) {
    CcCtx *c = live();
    CC_OWN(c, "cc_ew", y, a, b, cb, e);
    if (!c || !y || !p || p->n < 1) return 0;
    switch (p->op) {
    case CC_EW_ADD: case CC_EW_SWIGLU: if (!a || !b) return 0; break;
    case CC_EW_COMBINE: if (!(p->flags & 8) && !a) return 0; if ((p->flags & 1) && !b) return 0;
                        if ((p->flags & 2) && !cb) return 0; if ((p->flags & 4) && !e) return 0; if (p->D < 1) return 0; break;
    case CC_EW_HC_LOW: case CC_EW_HC_INJ: case CC_EW_SCALE: case CC_EW_F16: if (!a) return 0; break;
    case CC_EW_HC_MIX: case CC_EW_HC_APPLY: if (!a || !b || p->D < 1 || p->C < 1) return 0; break;
    case CC_EW_GATE_ADD: if (!a || !b || !e || p->D < 1) return 0; break;
    /* Muse Spark's attention gate and residual gate: the kernel has had them since the
     * chain took Muse (cc_muse), but this switch refused them, so its every forward fell
     * back to the host's step (tests/test_cuda_ew_ops_source.py) */
    case CC_EW_SIGMUL: if (!a || !b) return 0; break;
    case CC_EW_RESGATE: if (!a || !b || !cb || !e || p->D < 1) return 0; break;
    case CC_EW_HEADGATE: if (!a || !b || p->D < 1) return 0; break;   /* LFM3's gate a head (cc_lfm3) */
    default: return 0;
    }
    float *yp = fpw(y); const float *ap = fp(a), *bp = fp(b), *cp = fp(cb), *ep = fp(e); const CcEw pp = *p;
    CC_K(c, dim3(grid1((size_t)p->n, 256)), 256, cc_ew_k(yp, ap, bp, cp, ep, pp));
    return launched(c, p->op == CC_EW_COMBINE ? "ew combine" : p->op == CC_EW_SWIGLU ? "ew swiglu" : p->op == CC_EW_ADD ? "ew add" : "ew");
}
static int cc_qsa(CcBuf *src, CcBuf *w, CcBuf *pk, CcBuf *cs, CcBuf *sc, CcBuf *sel, const CcQsa *p) {
    CcCtx *c = live();
    CC_OWN(c, "cc_qsa", src, w, pk, cs, sc, sel);
    if (!c || !src || !pk || !p || p->ID < 1 || p->ID > 256 || p->R < 1) return 0;
    const float *srcp = fp(src), *wp = fp(w), *csp = fp(cs); float *pkp = fpw(pk), *scp = fpw(sc); const CcQsa pp = *p;
    if (p->mode == 0) {
        if (!w || !cs || p->nb < 1 || 2 * p->half_ > p->ID) return 0;
        CC_K(c, dim3((unsigned)p->nb), 128, cc_qsa_k(srcp, wp, pkp, csp, nullptr, nullptr, pp));
    } else if (p->mode == 1) {
        if (!sc || !sel || p->S < 1 || p->IQ < 1 || p->nbmax < 1 || p->sel_row < 1) return 0;
        const int mb = (p->pos_base + p->S) / p->R;   /* the last row's blocks, the most */
        int st = 1;   /* COLI_CUDA_QSA_TILES=0: a thread a (row, block) for every chunk (read each call) */
        { const char *e = getenv("COLI_CUDA_QSA_TILES"); st = !(e && *e == '0'); }
        const float *pkc = pkp;
        /* the pooled keys the tiles read past the last row's blocks are the buffer's (nbmax of them) */
        if (st && p->S >= 32 && mb > 0 && (size_t)((mb + CC_QST - 1) / CC_QST * CC_QST) * p->ID * 4 <= pk->bytes - (size_t)p->pk_off * 4)
            CC_K(c, dim3((unsigned)((mb + CC_QST - 1) / CC_QST), (unsigned)((p->S + CC_QST - 1) / CC_QST)), 256,
                 cc_qsa_score_t(srcp, pkc, scp, pp));
        else
            CC_K(c, dim3((unsigned)(((mb > 0 ? mb : 1) + 127) / 128), (unsigned)p->S), 128, cc_qsa_score_k(srcp, pkc, scp, pp));
        int *selp = (int *)sel->d;
        CC_K(c, dim3((unsigned)p->S), 128, cc_qsa_k(srcp, nullptr, pkp, nullptr, scp, selp, pp));
    } else return 0;
    return launched(c, "qsa");
}
static int cc_ple(CcBuf *keys, CcBuf *hyp, CcBuf *val, CcBuf *prm, CcBuf *gated, CcBuf *normv, CcBuf *conv, CcBuf *ring, const CcPle *p) {
    CcCtx *c = live();
    CC_OWN(c, "cc_ple", keys, hyp, val, prm, gated, normv, conv, ring);
    if (!c || !hyp || !gated || !normv || !p || p->S < 1 || p->C < 1 || p->H < 1) return 0;
    const float *kp = fp(keys), *vp = fp(val), *pp_ = fp(prm), *cp = fp(conv);
    float *hp = fpw(hyp), *gp = fpw(gated), *np = fpw(normv), *rp = fpw(ring); const CcPle pp = *p;
    if (p->mode == 0) {
        if (!keys || !val || !prm) return 0;
        CC_K(c, dim3((unsigned)(p->S * p->C)), 256, cc_ple_k(kp, hp, vp, pp_, gp, np, nullptr, nullptr, pp));
    } else if (p->mode == 1) {
        if (!conv || !ring || p->CK < 2 || (p->CK - 1) * p->NG > 32) return 0;
        CC_K(c, dim3(grid1((size_t)p->C * p->H, 256)), 256, cc_ple_k(nullptr, hp, nullptr, nullptr, gp, np, cp, rp, pp));
    } else return 0;
    return launched(c, "ple");
}
static int cc_route(CcBuf *l, CcBuf *bias, CcBuf *ids, CcBuf *w, const CcRoute *p) {
    CcCtx *c = live();
    CC_OWN(c, "cc_route", l, bias, ids, w);
    if (!c || !l || !ids || !w || !p || p->S < 1 || p->E < 1 || p->E > CC_ROUTE_EMAX || p->K < 1 || p->K > CC_ROUTE_KMAX ||
        p->K > p->E || p->kind < 0 || p->kind > 2 || p->has_bias < 0 || p->has_bias > 2 || (p->has_bias && !bias) || !(p->temp > 0.f) ||
        (p->n_group > 1 && (p->n_group > 64 || p->E % p->n_group || p->topk_group < 1 || p->topk_group > p->n_group))) return 0;
    if ((size_t)(p->l_off + (p->S - 1) * p->l_row + p->E) * 4 > l->bytes || (size_t)(p->i_off + (p->S - 1) * p->i_row + p->K) * 4 > ids->bytes ||
        (size_t)(p->w_off + (p->S - 1) * p->w_row + p->K) * 4 > w->bytes) return 0;
    const float *lp = fp(l), *bp = fp(bias); int *ip = (int *)ids->d; float *wp = fpw(w); const CcRoute pp = *p;
    CC_KSG(c, dim3((unsigned)p->S), 256, 0, cc_route_k<SG>(lp, bp, ip, wp, pp));
    return launched(c, "route");
}
static int cc_moe(CcBuf *tab, CcBuf *x, CcBuf *ids, CcBuf *w, CcBuf *hid, CcBuf *y, CcBuf *hy, const CcMoe *p) {
    CcCtx *c = live();
    CC_OWN(c, "cc_moe", tab, x, ids, w, hid, y, hy);
    if (p) cc_own_table(c, "cc_moe", tab, p->t_off, p->E);
    if (!c || !tab || !ids || !hid || !p || p->parts < 1 || p->parts > 3 || p->S < 1 || p->K < 1) return 0;
    if ((p->parts & 1) && !x) return 0;
    if ((p->parts & 2) && (!w || !y || (p->h_off >= 0 && !hy))) return 0;
    const int LS = p->lay_S > 0 ? p->lay_S : p->S;
    if (p->lay_S < 0 || p->lay_s0 < 0 || p->lay_s0 > LS - p->S || ((p->parts & 1) && (LS != p->S || p->lay_s0))) return 0;
    if (cc_moe_hid_floats(LS, p->K, p->D, p->I, p->E) * 4 > hid->bytes || (LS > CC_MOE_FRAME_ROWS && p->E < 1)) return 0;
    g_st.ops++;
    if (!coli_cuda_moe_async(c->device, (const CcExpert *)tab->d, fp(x), (const int *)ids->d, fp(w), fpw(hid), fpw(y), fp(hy), p,
                             (void *)c->stream)) { cc_mark_lost(c); return 0; }
    if (coli_sycl_fprof_flag) coli_sycl_fprof_name(p->parts == 1 ? "moe gate+up, down" : p->parts == 2 ? "moe sum" : "moe gate+up, down, sum");
    return 1;
}
/* a hyper-connection read's up projection and mix (coli_cuda_tensor_gemm_hcmix_async): -1 not taken */
static int cc_matmul_hcmix(ColiCudaTensor *t, CcBuf *x, size_t x_off, CcBuf *hn, CcBuf *y, int n, int H, int C, float fc) {
    CcCtx *c = live();
    CC_OWN(c, "cc_matmul_hcmix", x, hn, y);
    if (!c || !t || !x || !hn || !y || n < 1 || H < 1 || C < 1) return -1;
    int fmt = 0, I = 0, O = 0;
    if (!coli_cuda_tensor_shape(t, &fmt, &I, &O) || coli_cuda_tensor_device(t) != c->device || O != C * H ||
        (x_off + (size_t)n * I) * 4 > x->bytes || (size_t)n * O * 4 > hn->bytes || (size_t)n * H * 4 > y->bytes) return -1;
    const int r = coli_cuda_tensor_gemm_hcmix_async(t, fpw(y), fp(x) + x_off, fp(hn), n, H, C, fc, (void *)c->stream);
    if (r > 0) { g_st.ops++; g_st.matmuls++; return 1; }
    if (r == 0) cc_mark_lost(c);
    return r;
}
/* moe's parts 3 with down and the sums by slices of cols outputs (coli_cuda_moe_cols_async): -1 not
 * taken, the device as it was */
static int cc_moe_cols(CcBuf *tab, CcBuf *x, CcBuf *ids, CcBuf *w, CcBuf *hid, CcBuf *y, const CcMoe *p, int cols) {
    CcCtx *c = live();
    CC_OWN(c, "cc_moe_cols", tab, x, ids, w, hid, y);
    if (!c || !tab || !x || !ids || !w || !hid || !y || !p || p->parts != 3 || p->S < 1 || p->K < 1 || p->lay_S || p->lay_s0 ||
        p->h_off >= 0 || cols < 128 || cols % 128) return -1;
    if (cc_moe_hid_floats(p->S, p->K, cols < p->D ? cols : p->D, p->I, p->E) * 4 > hid->bytes ||
        (size_t)(p->x_off + (size_t)(p->S - 1) * p->x_row + p->D) * 4 > x->bytes ||
        (size_t)(p->y_off + (size_t)(p->S - 1) * p->y_row + p->D) * 4 > y->bytes) return -1;
    const int r = coli_cuda_moe_cols_async(c->device, (const CcExpert *)tab->d, fp(x), (const int *)ids->d, fp(w), fpw(hid), fpw(y), p,
                                           cols, (void *)c->stream);
    if (r > 0) { g_st.ops++; return 1; }
    if (r == 0) cc_mark_lost(c);
    return r;
}
/* Host tasks, not kernels: see the host synchronization contract above. */
static int cc_signal(CcBuf *flag, size_t word, unsigned v) {
    CcCtx *c = live();
    CC_OWN(c, "cc_signal", flag);
    if (!c || !flag || flag->kind != CC_DOWN || (word + 1) * 4 > flag->bytes) return 0;
    unsigned *f = (unsigned *)flag->d + word;
    const auto state = coli_sycl_host_state(c->device);
    (void)cc_try(c, "signal", [&] { coli_sycl_fprof_ev(c->stream->submit([=](sycl::handler &h) { h.host_task([=] { state->signal(f, v); }); })); });
    return launched(c, "signal");
}
static int cc_wait_host(CcBuf *flag, size_t word, unsigned v) {
    CcCtx *c = live();
    CC_OWN(c, "cc_wait_host", flag);
    if (!c || !flag || flag->kind != CC_UP || (word + 1) * 4 > flag->bytes) return 0;
    const unsigned *f = (const unsigned *)flag->d + word;
    const auto state = coli_sycl_host_state(c->device);
    (void)cc_try(c, "wait host", [&] { coli_sycl_fprof_ev(c->stream->submit([=](sycl::handler &h) { h.host_task([=] { state->wait(f, v); }); })); });
    return launched(c, "wait host");
}
/* mark / reached: the hybrid step's "the routes are down" without a host task. cc_signal's host task holds every
 * later command of the in-order queue until the runtime has seen the routes' kernel end, run the task and released them:
 * the card sat idle 20-100 us a layer between the router and the resident experts (by a frame profile), and the host
 * learned of the routes only through that hop. A mark is a barrier event at this point of the frame -- complete once
 * every earlier command is, exactly what the host task waited for -- and holds nothing: the frame's later commands
 * follow at once, and the host asks the event's status (reached: 1 complete, 0 not yet, -1 lost or no mark). No device
 * access to host memory beyond the kernels' own stores, no atomics, no kernel that waits: the contract above stands.
 * The rows the kernels before the mark wrote to DOWN memory are there when it is complete (their events completed). */
static int cc_mark(void) {
    CcCtx *c = live();
    if (!c) return 0;
    if (!cc_try(c, "mark", [&] { c->sy->mark = c->stream->ext_oneapi_submit_barrier(); })) return 0;
    if (coli_sycl_fprof_flag) { coli_sycl_fprof_ev(c->sy->mark); coli_sycl_fprof_name("mark"); }
    c->sy->mark_set = 1;
    return 1;
}
static int cc_reached(void) {
    CcCtx *c = live();
    if (!c || !c->sy->mark_set) return -1;
    int done = 0;
    if (!cc_try(c, "mark status", [&] {
            done = c->sy->mark.get_info<sycl::info::event::command_execution_status>() == sycl::info::event_command_status::complete; }))
        return -1;
    return done;
}
static int cc_moe_stage_op(CcBuf *b, size_t slot_bytes, int slots) {
    CcCtx *c = live();
    CC_OWN(c, "cc_moe_stage_op", b);
    if (!c || (b && (b->kind != CC_DEV || slots < 1 || CC_MOE_STAGE_MAP + (size_t)slots * slot_bytes > b->bytes))) return 0;
    return coli_cuda_moe_stage(c->device, b ? (void *)b->d : nullptr, b ? slot_bytes : 0, b ? slots : 0);
}
static void cc_stats(CcStats *st) { if (st) *st = g_st; }

/* ---- staging: copies from page-locked host memory beside the frame ---------------- */
/* The copy queue (c->cs, the device's second in-order queue) and its events are the context's
 * from cc_init: cudaStreamCreate's and cudaEventCreate's lazy twins are nothing here. A barrier
 * submitted with an event is cudaStreamWaitEvent, one submitted alone cudaEventRecord (its event
 * completes once everything before it on the in-order queue has); an event never recorded is
 * complete, so the first fetch starts at once. */
static int cc_fetch(CcBuf *dst, const CcFetch *r, int n) {
    CcCtx *c = live();
    CC_OWN(c, "cc_fetch", dst);
    if (!c || !dst || dst->kind != CC_DEV || n < 0 || (n && !r)) return 0;
    for (int i = 0; i < n; i++) if (!r[i].src || r[i].dst > dst->bytes || r[i].bytes > dst->bytes - r[i].dst) return 0;
    if (!cc_try(c, "fetch's reader", [&] { c->cs->ext_oneapi_submit_barrier({c->sy->ev_free}); })) return 0;
    for (int i = 0; i < n; i++) {
        if (!r[i].bytes) continue;
        if (!cc_try(c, "fetch", [&] { c->cs->memcpy((char *)dst->d + r[i].dst, r[i].src, r[i].bytes); })) return 0;
        g_st.bytes_up += r[i].bytes;
    }
    g_st.ops++;
    return cc_try(c, "fetch event", [&] { c->sy->ev_fetched = c->cs->ext_oneapi_submit_barrier(); });
}
static int cc_fetch_wait(void) {
    CcCtx *c = live();
    if (!c) return 0;
    g_st.ops++;
    return cc_try(c, "fetch wait", [&] { c->stream->ext_oneapi_submit_barrier({c->sy->ev_fetched}); });
}
static int cc_fetch_done(void) {
    CcCtx *c = live();
    if (!c) return 0;
    return cc_try(c, "fetch done", [&] { c->sy->ev_free = c->stream->ext_oneapi_submit_barrier(); });
}
/* host_up: bytes of page-locked host memory to the device on the copy queue, once the frame's ops
 * recorded so far are done (up_after, a barrier on the chain's queue the copy queue waits for);
 * the copy's own event is its mark in the ring, which host_up_done asks about by ticket */
static int cc_host_up(void *dst, const void *src, size_t bytes, unsigned *ticket) {
    CcCtx *c = live();
    if (!c || !dst || !src || !ticket) return 0;
    c->up_made = 1;
    unsigned t = ++c->up_seq;
    sycl::event &ev = c->sy->up_ev[t % CC_UP_MARKS];
    if (t > CC_UP_MARKS && !cc_try(c, "host up ring", [&] { ev.wait_and_throw(); })) return 0;   /* the mark it reuses landed */
    if (!cc_try(c, "host up", [&] {
            c->sy->up_after = c->stream->ext_oneapi_submit_barrier();
            c->cs->ext_oneapi_submit_barrier({c->sy->up_after});
            ev = c->cs->memcpy(dst, src, bytes); })) return 0;
    g_st.bytes_up += bytes;
    *ticket = t;
    return 1;
}
/* host_up for n copies: one wait for the frame, the copies one after the other on the copy queue
 * (cudaMemcpyBatchAsync has no SYCL form), one mark (the last copy's event) */
static int cc_host_up_n(void *const *dst, const void *const *src, const size_t *bytes, int n, unsigned *ticket) {
    CcCtx *c = live();
    if (!c || !dst || !src || !bytes || n < 1 || !ticket) return 0;
    for (int i = 0; i < n; i++) if (!dst[i] || !src[i]) return 0;
    c->up_made = 1;
    unsigned t = ++c->up_seq;
    sycl::event &ev = c->sy->up_ev[t % CC_UP_MARKS];
    if (t > CC_UP_MARKS && !cc_try(c, "host up ring", [&] { ev.wait_and_throw(); })) return 0;   /* the mark it reuses landed */
    if (!cc_try(c, "host up", [&] {
            c->sy->up_after = c->stream->ext_oneapi_submit_barrier();
            c->cs->ext_oneapi_submit_barrier({c->sy->up_after});
            for (int i = 0; i < n; i++) ev = c->cs->memcpy(dst[i], src[i], bytes[i]); })) return 0;
    for (int i = 0; i < n; i++) g_st.bytes_up += bytes[i];
    *ticket = t;
    return 1;
}
static int cc_host_up_done(unsigned ticket) {
    CcCtx *c = live();
    if (!c || !c->up_made || !ticket || ticket > c->up_seq) return 0;
    if (c->up_seq - ticket >= CC_UP_MARKS) return 1;   /* its mark reused: waited for then */
    int done = 0;
    if (!cc_try(c, "host up done", [&] {
            done = c->sy->up_ev[ticket % CC_UP_MARKS].get_info<sycl::info::event::command_execution_status>() ==
                   sycl::info::event_command_status::complete; })) return 0;
    return done;
}
static const void *cc_dev_addr(const CcBuf *b) { return b ? (const void *)b->d : nullptr; }   /* a mapped UP/DOWN buffer's: where the device reads it */
/* cc_attn with a window and sinks (Flint), the same three forms cc_attn picks: a prompt chunk
 * on the matrix engines (cc_attn_xmx), a decode's or a verify's rows split over their positions
 * (cc_attn_dec_k, as cc_attn_dec_on says), else cc_attn_k; each row's positions the window's,
 * the sink in each softmax's denominator. */
template <typename KT>
static int cc_attn_w2_t(CcBuf *q, CcBuf *kc, CcBuf *vc, CcBuf *o, CcBuf *gate, CcBuf *snk, const CcAttnW *w, int ring) {
    CcCtx *c = live();
    CC_OWN(c, "cc_attn_w2_t", q, kc, vc, o, gate, snk);
    if (!w) return 0;
    const CcAttn *p = &w->a;
    if (!c || !q || !kc || !vc || !o || p->S < 1 || p->H < 1 || p->KVH < 1 || p->H % p->KVH || p->hd < 1 || p->hd > 256 ||
        p->sel_row > 0 || w->win < 0 || (w->sink && !snk) || (p->has_gate && !gate) || ring < 0 ||
        (ring && (ring != p->cap || w->win < 1 || ring < w->win + p->S - 1))) return 0;   /* a ring holds the step's rows and their windows */
    if (!w->win && !w->sink) return cc_attn_t<KT>(q, kc, vc, o, gate, nullptr, p);
    const float *sk = w->sink ? fp(snk) : nullptr;
    const int win = w->win, so = w->sink_off;
    const float *qp = fp(q), *gp = fp(gate); const KT *kp = kvp<KT>(kc), *vp = kvp<KT>(vc); float *op = fpw(o);
    static int fl = -1;
    if (fl < 0) { const char *e = getenv("COLI_CUDA_CHAIN_FLASH"); fl = e && *e ? atoi(e) : 16; if (fl < 0) fl = 0; }
    int G = p->H / p->KVH;
    const int xmx = fl > 0 && cc_attn_xmx_ok(c, p, G, sizeof(KT) == 2, 0);
    if (xmx && p->S >= fl) {
        if (cc_attn_xmx<KT>(c, q, kc, vc, o, gate, p, G, win, sk, so, ring, nullptr)) return launched(c, "attn (matrix engines, window, sinks)");
        if (c->lost) return 0;
    }
    const int fk = cc_attn_dec_on(xmx);
    /* a row's positions: the window's at most */
    #define CC_WPOS(r) (win > 0 && p->pos_base + (r) + 1 > win ? win : p->pos_base + (r) + 1)
    if (fk && p->S <= CC_ATTN_DEC_ROWS && CC_WPOS(p->S - 1) >= CC_ATTN_DEC_MIN) {
        int k = 0;   /* the rows with fewer than CC_ATTN_DEC_MIN positions: cc_attn_k, as in cc_attn */
        while (k < p->S && CC_WPOS(k) < CC_ATTN_DEC_MIN) k++;
        if (k > 0) {
            CcAttn a = *p; a.S = k;
            CC_K(c, dim3((unsigned)p->H, (unsigned)k), 128, cc_attn_k<KT>(qp, kp, vp, op, gp, nullptr, a, win, sk, so, ring));
            if (!launched(c, "attn (window, sinks)")) return 0;
            CcAttnW b = *w; b.a.S -= k; b.a.pos_base += k; b.a.q_off += k * p->q_row; b.a.o_off += k * p->o_row; b.a.g_off += k * p->g_row;
            return cc_attn_w2_t<KT>(q, kc, vc, o, gate, snk, &b, ring);
        }
        int gq = p->k_off % 4 == 0 && p->v_off % 4 == 0 ? cc_decg_heads(G, p->hd) : 0, hc = (gq ? 4 * p->KVH : p->H) | (cc_attn_dec_len0() << 16), nb = 1;   /* as in cc_attn */
        for (int r = 0; r < p->S; r++) {
            int n = CC_WPOS(r), L = cc_attn_dec_len(n, hc), m = (n + L - 1) / L;
            if (m > nb) nb = m;
        }
        if (cc_adec_fit(c, (size_t)p->S * p->H * nb * (p->hd + 2) * sizeof(float))) {
            float *xs = c->adec; const CcAttn pp = *p;
            const int hc0 = cc_attn_dec_len0() << 16;   /* read on the host: the launch's body runs on the device */
            if (gq) cc_decg_launch(c, gq, qp, kp, vp, xs, *p, nb, win, ring, hc);
            else CC_KSG_AT(c, c->sg_attn, dim3((unsigned)p->H, (unsigned)nb, (unsigned)p->S), 256, 0,
                        cc_attn_dec_k<KT, SG>(qp, kp, vp, op, gp, xs, pp, 0, nb, win, sk, so, ring, hc0, nullptr));
            CC_KSG_AT(c, c->sg_attn, dim3((unsigned)p->H, (unsigned)p->S), 256, 0,
                   cc_attn_dec_k<KT, SG>(qp, kp, vp, op, gp, xs, pp, 1, nb, win, sk, so, ring, hc, nullptr));
            return launched(c, "attn (positions in blocks, window, sinks)");
        }
        if (c->lost) return 0;
    }
    #undef CC_WPOS
    const CcAttn pp = *p;
    CC_K(c, dim3((unsigned)p->H, (unsigned)p->S), 128, cc_attn_k<KT>(qp, kp, vp, op, gp, nullptr, pp, win, sk, so, ring));
    return launched(c, "attn (window, sinks)");
}
static int cc_attn_w2(CcBuf *q, CcBuf *kc, CcBuf *vc, CcBuf *o, CcBuf *gate, CcBuf *snk, const CcAttnW *w, int ring) {
    CC_OWN(cur(), "cc_attn_w2", q, kc, vc, o, gate, snk);
    return cc_attn_w2_t<float>(q, kc, vc, o, gate, snk, w, ring);
}
static int cc_attn_w(CcBuf *q, CcBuf *kc, CcBuf *vc, CcBuf *o, CcBuf *gate, CcBuf *snk, const CcAttnW *w) {
    CC_OWN(cur(), "cc_attn_w", q, kc, vc, o, gate, snk);
    return cc_attn_w2(q, kc, vc, o, gate, snk, w, 0);
}

static int cc_muse(void) { return 1; }   /* the norm's PRESCALE, ew's SIGMUL and RESGATE */
static int cc_kv16(void) { return 1; }   /* ew's F16 */
/* ---- several conversations' decode rows in one launch (dnconv_mx, dnrec_mx) -----------------------
 * A step of n conversations (KV_SLOTS) launched cc_dnconv_k and cc_dnrec_rows_k once a conversation,
 * one after another: on a Max 1100 with eight conversations 480 of a step's 1557 commands and 8.7 ms
 * of its 46 (g6-evidence/m4-kv-slots), each using a small part of the device. Here work-group (.., j)
 * runs the same kernel body for conversation j: its rows are j rows on in the step's buffers, its ring
 * or state its own allocation, taken from a table in the kernel's arguments. Nothing else changes, so
 * a conversation's bits are those of its own launch (the chain test compares them). */
struct CcMxP { float *p[CC_MX_N]; };
static void cc_dnconv_mx_k(const float *__restrict__ xin, const float *__restrict__ w, CcMxP ring,
                           float *__restrict__ o, CcDnConv p) {
    const int j = (int)sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_group(1);
    p.in_off += j * p.in_row; p.out_off += j * p.out_row;
    cc_dnconv_k(xin, w, ring.p[j], o, nullptr, p);
}
template <int KD, int R, int SG>
static void cc_dnrec_mx_k(const float *__restrict__ cv, const float *__restrict__ ab, const float *__restrict__ z,
                          CcMxP st, const float *__restrict__ prm, float *__restrict__ y, CcDnRec p) {
    const int j = (int)sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_group(1);
    p.cv_off += j * p.cv_row; p.b_off += j * p.b_row; p.a_off += j * p.a_row; p.z_off += j * p.z_row; p.y_off += j * p.y_row;
    cc_dnrec_rows_k<KD, R, SG>(cv, ab, z, st.p[j], prm, y, nullptr, p);
}
/* the buffers of n conversations as a kernel's table: all there, all different (two rows on one state would race) */
static int cc_mx_table(CcBuf *const *b, int n, CcMxP *t) {
    if (coli_sycl_test_devices() && b && n > 0) cc_own_check(cur(), "cc_mx_table", "b[0..n)", b, n);
    for (int j = 0; j < CC_MX_N; j++) t->p[j] = nullptr;
    for (int j = 0; j < n; j++) {
        if (!b[j] || !(t->p[j] = fpw(b[j]))) return 0;
        for (int i = 0; i < j; i++) if (t->p[i] == t->p[j]) return 0;
    }
    return 1;
}
static int cc_dnconv_mx(CcBuf *in, CcBuf *w, CcBuf *const *ring, int n, CcBuf *out, const CcDnConv *p) {
    CcCtx *c = live();
    CC_OWN(c, "cc_dnconv_mx", in, w, out);
    if (!c || !in || !w || !ring || !out || !p) return 0;
    /* what cc_dnconv gives cc_dnconv_k: one row, no snapshot */
    if (n < 2 || n > CC_MX_N || p->S != 1 || p->snap_row >= 0 || p->CD < 1 || p->CK < 2 || p->CK > 9) return -1;
    CcMxP rt;
    if (!cc_mx_table(ring, n, &rt)) return -1;
    const float *ip = fp(in), *wp = fp(w); float *op = fpw(out); const CcDnConv pp = *p;
    const sycl::range<3> g = dim3(grid1((size_t)p->CD, 64), (unsigned)n);
    CC_K(c, g, 64, cc_dnconv_mx_k(ip, wp, rt, op, pp));
    return launched(c, "dnconv (conversations)") ? 1 : 0;
}
static int cc_dnrec_mx(int KD, CcBuf *cv, CcBuf *ab, CcBuf *z, CcBuf *const *st, int n, CcBuf *prm, CcBuf *y, const CcDnRec *p) {
    CcCtx *c = live();
    CC_OWN(c, "cc_dnrec_mx", cv, ab, z, prm, y);
    if (!c || !cv || !ab || !z || !st || !prm || !y || !p) return 0;
    if (n < 2 || n > CC_MX_N || p->S != 1 || p->snap_row >= 0 || p->VH < 1 || p->KH < 1 || p->VH % p->KH ||
        p->VD < 1 || p->VD > 128) return -1;
    /* only where cc_dnrec gives one row to cc_dnrec_rows_k (its default: COLI_CUDA_DNREC_ROWS and _DEC on,
     * COLI_CUDA_DNREC_MIN above one row); any other choice keeps the caller's n calls */
    static const int rows_dec = [] {
        const char *r = getenv("COLI_CUDA_DNREC_ROWS"), *d = getenv("COLI_CUDA_DNREC_DEC"), *m = getenv("COLI_CUDA_DNREC_MIN");
        return !(r && *r == '0') && !(d && *d == '0') && (m && *m ? atoi(m) : CC_DNREC_ROWS) > 1;
    }();
    if (!rows_dec) return -1;
    CcMxP t;
    if (!cc_mx_table(st, n, &t)) return -1;
    const float *a = fp(cv), *b = fp(ab), *zz = fp(z), *pr = fp(prm); float *yy = fpw(y); const CcDnRec pp = *p;
    const sycl::range<3> g = dim3((unsigned)p->VH, (unsigned)n);
    switch (KD) {
    case 4:   CC_KSG_AT(c, c->sg_dn, g, 128, 0, cc_dnrec_mx_k<4, 16, SG>(a, b, zz, t, pr, yy, pp)); break;
    case 8:   CC_KSG_AT(c, c->sg_dn, g, 128, 0, cc_dnrec_mx_k<8, 16, SG>(a, b, zz, t, pr, yy, pp)); break;
    case 16:  CC_KSG_AT(c, c->sg_dn, g, 128, 0, cc_dnrec_mx_k<16, 16, SG>(a, b, zz, t, pr, yy, pp)); break;
    case 32:  CC_KSG_AT(c, c->sg_dn, g, 128, 0, cc_dnrec_mx_k<32, 16, SG>(a, b, zz, t, pr, yy, pp)); break;
    case 64:  CC_KSG_AT(c, c->sg_dn, g, 128, 0, cc_dnrec_mx_k<64, 16, SG>(a, b, zz, t, pr, yy, pp)); break;
    case 128: CC_KSG_AT(c, c->sg_dn, g, 128, 0, cc_dnrec_mx_k<128, 16, SG>(a, b, zz, t, pr, yy, pp)); break;
    case 256: CC_KSG_AT(c, c->sg_dn, g, 128, 0, cc_dnrec_mx_k<256, 8, SG>(a, b, zz, t, pr, yy, pp)); break;
    default: return -1;
    }
    return launched(c, "dnrec rows (conversations)") ? 1 : 0;
}
static int cc_lfm3(void) { return 1; }   /* dnconv's order 2, ew's HEADGATE */
static int cc_pick(CcBuf *logits, CcBuf *out, int n) {
    CcCtx *c = live();
    CC_OWN(c, "cc_pick", logits, out);
    if (!c || !logits || !out || n < 1 || (size_t)n * 4 > logits->bytes || out->bytes < 8) return 0;
    const float *lg = fp(logits); void *op = out->d;
    CC_KSG(c, dim3(1), 256, 0, cc_pick_k<SG>(lg, op, n));
    return launched(c, "pick");
}
static int cc_pick_rows(CcBuf *logits, size_t off, CcBuf *out, int n, int rows) {
    CcCtx *c = live();
    CC_OWN(c, "cc_pick_rows", logits, out);
    if (!c || !logits || !out || n < 1 || rows < 1 || (off + (size_t)rows * n) * 4 > logits->bytes || out->bytes < (size_t)rows * 8) return 0;
    const float *lg = fp(logits) + off; void *op = out->d;
    CC_KSG(c, dim3((unsigned)rows), 256, 0, cc_pick_k<SG>(lg, op, n));
    return launched(c, "pick rows");
}
static int cc_matmul_rows(ColiCudaTensor *t, CcBuf *ids, CcBuf *x, CcBuf *y, int n) {
    CcCtx *c = live();
    CC_OWN(c, "cc_matmul_rows", ids, x, y);
    if (!c || !t || !ids || !x || !y || n < 1) return 0;
    int fmt, I, O; const void *w = nullptr; const float *sc = nullptr;
    if (!coli_cuda_tensor_shape(t, &fmt, &I, &O) || coli_cuda_tensor_device(t) != c->device || (fmt != 0 && fmt != 1 && fmt < 100)) return 0;
    if ((size_t)I * 4 > x->bytes || (size_t)n * 4 > ids->bytes || (size_t)n * 4 > y->bytes) return 0;
    if (fmt >= 100) {   /* a GGUF's blocks: the whole GEMV's bits for each listed row (backend_cuda.cu) */
        g_st.matmuls++;
        if (!coli_cuda_tensor_gemv_rows_async(t, (const int *)ids->d, fpw(y), fp(x), n, (void *)c->stream)) return 0;
        return launched(c, "matmul_rows");
    }
    if (!coli_cuda_tensor_dev(t, &w, &sc, nullptr) || (fmt == 1 && !sc)) return 0;
    g_st.matmuls++;
    const unsigned grid = grid1((size_t)n, 256 / c->sg);   /* a warp a row */
    float *yp = fpw(y); const float *xp = fp(x); const int *ip = (const int *)ids->d;
    if (fmt == 1 && I % 16 == 0 && I <= 12288) {
        const int8_t *w8 = (const int8_t *)w;
        size_t smem = (size_t)(I / 4 + I / 32 + 1) * sizeof(sycl::float4);   /* x as float4s, padded (cc_gx4) */
        if (smem <= c->slm) CC_KSG(c, dim3(grid), 256, smem, cc_matmul_rows_i8_k<1, SG>(yp, xp, w8, sc, ip, I, O, n, slm_));
        else CC_KSG(c, dim3(grid), 256, (size_t)I * sizeof(float), cc_matmul_rows_i8_k<0, SG>(yp, xp, w8, sc, ip, I, O, n, slm_));
    } else {
        CC_KSG(c, dim3(grid), 256, 0, cc_matmul_rows_k<SG>(yp, xp, w, sc, ip, fmt, I, O, n));
    }
    return launched(c, "matmul_rows");
}
/* copy_regions_h: float32 regions of src (offsets and lengths in floats) into a float16 cache
 * (in halves), each value rounded to the nearest float16: a step's K and V rows into the
 * mirrors of a float16 KV cache, whose values rope_f16 has already rounded */
static void cc_copy_grid_h_k(sycl::half *__restrict__ d,
                             const float *__restrict__ s, CcGrid g) {
    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    long long gid =
        (long long)it.get_group(2) * it.get_local_range(2) +
        it.get_local_id(2);
    if (gid >= (long long)g.n * g.len) return;
    long long i = gid / g.len, j = gid - i * g.len, q = i / g.per, h = i - q * g.per;
    d[g.d0 + q * g.dq + h * g.dh + j] =
        sycl::vec<float, 1>(s[g.s0 + q * g.sq + h * g.sh + j])
            .convert<sycl::half, sycl::rounding_mode::rte>()[0];
}
static int cc_copy_regions_h(CcBuf *dst, CcBuf *src, const CcRegion *r, int n) {
    CcCtx *c = live();
    CC_OWN(c, "cc_copy_regions_h", dst, src);
    if (!c || !dst || !src || !dst->d || !src->d || dst == src || (n > 0 && !r)) return 0;
    for (int i = 0; i < n; i++)
        if ((r[i].dst + r[i].n) * 2 > dst->bytes || (r[i].src + r[i].n) * 4 > src->bytes) return 0;
    sycl::half *d = reinterpret_cast<sycl::half *>(dst->d); const float *s = src->d;
    for (int i = 0; i < n; ) {
        int k = cc_grid_run(r + i, n - i);
        CcGrid g;
        if (!(k >= 2 && cc_grid_of(r + i, k, &g))) { k = 1; g = CcGrid{(long long)r[i].dst, (long long)r[i].src, 0, 0, 0, 0, (int)r[i].n, 1, 1}; }
        long long tot = (long long)g.n * g.len;
        if (tot > 0) {
            CC_K(c, dim3(grid1((size_t)tot, 256)), 256, cc_copy_grid_h_k(d, s, g));
            if (!launched(c, "copy regions (float16)")) return 0;
        }
        i += k;
    }
    g_st.ops++;
    return 1;
}
/* attn_h, attn_wh: attn and attn_wr over a float16 cache (its offsets in halves) */
static int cc_attn_h(CcBuf *q, CcBuf *kc, CcBuf *vc, CcBuf *o, CcBuf *gate, CcBuf *sel, const CcAttn *p) {
    CC_OWN(cur(), "cc_attn_h", q, kc, vc, o, gate, sel);
    return cc_attn_t<sycl::half>(q, kc, vc, o, gate, sel, p);
}
static int cc_attn_wh(CcBuf *q, CcBuf *kc, CcBuf *vc, CcBuf *o, CcBuf *gate, CcBuf *snk, const CcAttnW *w, int ring) {
    CC_OWN(cur(), "cc_attn_wh", q, kc, vc, o, gate, snk);
    return cc_attn_w2_t<sycl::half>(q, kc, vc, o, gate, snk, w, ring);
}
static int cc_kvh(void) { return 1; }
/* ---- several conversations' decode rows in one launch: their attention and their cache rows --------
 * (attn_h_mx, copy_regions_h_mx: float16 caches, COLI_CUDA_MUX_BATCH's 4 and 8). As dnconv_mx and
 * dnrec_mx above: the one-row kernels' bodies as they are, the row's conversation from a table in the
 * kernel's arguments.
 * The attention's kernels take rows already (a verify's: group 0 in the blocks, group 1 in the join)
 * and read of a row, besides its offsets, only n = p.pos_base + s + 1 and kbase = kvh * p.cap: the
 * wrappers give row s its own position (p.pos_base = pos[s] - s, so n is pos[s] + 1), its cap and its
 * caches. A row's block length is cc_attn_dec_len of its own n, and nb (the longest row's blocks) only
 * places the blocks in the scratch: a shorter row's further blocks are written empty and its join reads
 * its own nbr alone. So each row is cc_attn_t<half>'s decode step of that row (the chain test compares). */
static_assert(CC_MX_ATTN_N <= CC_ATTN_DEC_ROWS && CC_MX_ATTN_N <= CC_MX_N, "attn_h_mx: a decode step's rows");
struct CcMxKV { const sycl::half *kc[CC_MX_ATTN_N], *vc[CC_MX_ATTN_N]; int pos[CC_MX_ATTN_N], cap[CC_MX_ATTN_N]; };
template <int G, int SG>
static void cc_attn_decg_mx_k(const float *__restrict__ q, CcMxKV t, float *__restrict__ xs, CcAttn p, int nb,
                              int win, int ring, int hc, int tpp, uint8_t *slm) {
    const int s = (int)sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_group(0);
    p.pos_base = t.pos[s] - s; p.cap = t.cap[s];
    cc_attn_decg_k<G, sycl::half, SG>(q, t.kc[s], t.vc[s], xs, p, nb, win, ring, hc, tpp, slm);
}
/* mode 0: the blocks a head (the row in group 0); mode 1: the join (the row in group 1) */
template <int SG>
static void cc_attn_dec_mx_k(const float *__restrict__ q, CcMxKV t, float *__restrict__ o, const float *__restrict__ gt,
                             float *__restrict__ xs, CcAttn p, int mode, int nb, int hc, const int *__restrict__ sel) {
    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int s = (int)(mode == 0 ? it.get_group(0) : it.get_group(1));
    p.pos_base = t.pos[s] - s; p.cap = t.cap[s];
    cc_attn_dec_k<sycl::half, SG>(q, t.kc[s], t.vc[s], o, gt, xs, p, mode, nb, 0, nullptr, 0, 0, hc, sel);
}
static int cc_attn_h_mx(CcBuf *q, CcBuf *const *kc, CcBuf *const *vc, const int *pos, const int *cap, int n, CcBuf *o,
                        CcBuf *gate, const CcAttn *p) {
    CcCtx *c = live();
    CC_OWN(c, "cc_attn_h_mx", q, o, gate);
    if (!c || !q || !kc || !vc || !pos || !cap || !o || !p) return 0;
    if (n < 2 || n > CC_MX_ATTN_N || p->S != 1 || p->sel_row > 0 || p->H < 1 || p->KVH < 1 || p->H % p->KVH ||
        p->hd < 1 || p->hd > 256 || (p->has_gate && !gate)) return -1;
    /* cc_attn_t's choice for one row with no list: not the matrix engines (a row only at
     * COLI_CUDA_CHAIN_FLASH=1), the split over positions on (COLI_CUDA_ATTN_DEC), every row at or past
     * COLI_CUDA_ATTN_DEC_MIN positions; any other choice keeps the caller's n calls */
    static const int fl = [] { const char *e = getenv("COLI_CUDA_CHAIN_FLASH"); const int v = e && *e ? atoi(e) : 16; return v < 0 ? 0 : v; }();
    const int G = p->H / p->KVH, xmx = fl > 0 && cc_attn_xmx_ok(c, p, G, 1, 0);
    if ((xmx && fl <= 1) || !cc_attn_dec_on(xmx)) return -1;
    const int mn = cc_attn_dec_min();
    CcMxKV t;
    for (int j = 0; j < CC_MX_ATTN_N; j++) { t.kc[j] = t.vc[j] = nullptr; t.pos[j] = t.cap[j] = 0; }
    for (int j = 0; j < n; j++) {
        if (!kc[j] || !vc[j] || !(t.kc[j] = kvp<sycl::half>(kc[j])) || !(t.vc[j] = kvp<sycl::half>(vc[j]))) return -1;
        if (pos[j] < 0 || pos[j] + 1 < mn || cap[j] <= pos[j]) return -1;
        const size_t ne = (size_t)p->KVH * cap[j] * p->hd;   /* the cache holds cap rows a KV head */
        if (((size_t)p->k_off + ne) * 2 > kc[j]->bytes || ((size_t)p->v_off + ne) * 2 > vc[j]->bytes) return -1;
        t.pos[j] = pos[j]; t.cap[j] = cap[j];
    }
    /* as cc_attn_t: a KV head's query heads a block where cc_attn_decg_k takes them, the rule's heads and
     * first block length, nb the most blocks any row takes */
    const int gq = p->k_off % 4 == 0 && p->v_off % 4 == 0 ? cc_decg_heads(G, p->hd) : 0, hcr = gq ? 4 * p->KVH : p->H;
    const int l0 = cc_attn_dec_len0(), hc = hcr | (l0 << 16), hc0 = l0 << 16;
    int nb = 1;
    for (int j = 0; j < n; j++) {
        const int nn = pos[j] + 1, L = cc_attn_dec_len(nn, hc), m = (nn + L - 1) / L;
        if (m > nb) nb = m;
    }
    if (!cc_adec_fit(c, (size_t)n * p->H * nb * (p->hd + 2) * sizeof(float))) return c->lost ? 0 : -1;
    float *xs = c->adec; const float *qp = fp(q), *gp = fp(gate); float *op = fpw(o); const CcAttn pp = *p;
    const int *lp = nullptr;   /* no list: a kernel argument, as cc_attn_t's lp */
    if (gq) {
        const sycl::range<3> g = dim3((unsigned)p->KVH, (unsigned)nb, (unsigned)n);
        const size_t sm = (size_t)gq * (p->hd + 256) * sizeof(float);   /* cc_decg_launch's: the heads' q rows and scores */
        const int tpp = cc_decg_tpp();
        int win = 0, ring = 0;   /* kernel arguments, as cc_decg_launch's */
        switch (gq) {
        case 2: CC_KSG_AT(c, c->sg_attn, g, 256, sm, cc_attn_decg_mx_k<2, SG>(qp, t, xs, pp, nb, win, ring, hc, tpp, slm_)); break;
        case 4: CC_KSG_AT(c, c->sg_attn, g, 256, sm, cc_attn_decg_mx_k<4, SG>(qp, t, xs, pp, nb, win, ring, hc, tpp, slm_)); break;
        case 8: CC_KSG_AT(c, c->sg_attn, g, 256, sm, cc_attn_decg_mx_k<8, SG>(qp, t, xs, pp, nb, win, ring, hc, tpp, slm_)); break;
        default: CC_KSG_AT(c, c->sg_attn, g, 256, sm, cc_attn_decg_mx_k<16, SG>(qp, t, xs, pp, nb, win, ring, hc, tpp, slm_)); break;
        }
    } else CC_KSG_AT(c, c->sg_attn, dim3((unsigned)p->H, (unsigned)nb, (unsigned)n), 256, 0,
                     cc_attn_dec_mx_k<SG>(qp, t, op, gp, xs, pp, 0, nb, hc0, lp));
    CC_KSG_AT(c, c->sg_attn, dim3((unsigned)p->H, (unsigned)n), 256, 0,
              cc_attn_dec_mx_k<SG>(qp, t, op, gp, xs, pp, 1, nb, hc, lp));
    return launched(c, "attn (positions in blocks, conversations)") ? 1 : 0;
}
/* copy_regions_h_mx: work-group (.., j) runs cc_copy_grid_h_k on conversation j's cache and grid. The
 * grids share their shape (len, per, n, the source's stride); each has its own first element in the
 * cache and in the source and its own stride in the cache (its cap) */
struct CcMxG { sycl::half *d[CC_MX_N]; long long d0[CC_MX_N], dh[CC_MX_N], s0[CC_MX_N]; CcGrid g; };
static void cc_copy_grid_h_mx_k(CcMxG t, const float *__restrict__ s) {
    const int j = (int)sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_group(1);
    CcGrid g = t.g; g.d0 = t.d0[j]; g.dh = t.dh[j]; g.s0 = t.s0[j];
    cc_copy_grid_h_k(t.d[j], s, g);
}
static int cc_copy_regions_h_mx(CcBuf *const *dst, int n, CcBuf *src, const CcRegion *r, int per) {
    CcCtx *c = live();
    CC_OWN(c, "cc_copy_regions_h_mx", src);
    if (!c || !dst || !src || !src->d || !r) return 0;
    if (n < 2 || n > CC_MX_N || per < 1) return -1;
    CcMxG t;
    for (int j = 0; j < CC_MX_N; j++) { t.d[j] = nullptr; t.d0[j] = t.dh[j] = t.s0[j] = 0; }
    t.g = CcGrid{0, 0, 0, 0, 0, 0, 0, 1, 0};
    for (int j = 0; j < n; j++) {
        const CcRegion *rj = r + (size_t)j * per;
        if (!dst[j] || !dst[j]->d || dst[j] == src) return -1;
        for (int i = 0; i < j; i++) if (dst[i]->d == dst[j]->d) return -1;   /* two rows into one cache: the caller's order */
        for (int h = 0; h < per; h++)
            if ((rj[h].dst + rj[h].n) * 2 > dst[j]->bytes || (rj[h].src + rj[h].n) * 4 > src->bytes) return -1;
        /* what cc_copy_regions_h launches for these regions, where that is one kernel: one grid of them
         * all, or one region alone */
        CcGrid g;
        if (per >= 2) { if (cc_grid_run(rj, per) != per || !cc_grid_of(rj, per, &g)) return -1; }
        else g = CcGrid{(long long)rj[0].dst, (long long)rj[0].src, 0, 0, 0, 0, (int)rj[0].n, 1, 1};
        if (j == 0) t.g = g;
        else if (g.len != t.g.len || g.per != t.g.per || g.n != t.g.n || g.sh != t.g.sh || g.dq != t.g.dq || g.sq != t.g.sq) return -1;
        t.d[j] = reinterpret_cast<sycl::half *>(dst[j]->d); t.d0[j] = g.d0; t.dh[j] = g.dh; t.s0[j] = g.s0;
    }
    const long long tot = (long long)t.g.n * t.g.len;
    if (tot <= 0) return -1;
    const float *s = src->d;
    CC_K(c, dim3(grid1((size_t)tot, 256), (unsigned)n), 256, cc_copy_grid_h_mx_k(t, s));
    return launched(c, "copy regions (float16, conversations)") ? 1 : 0;
}
/* ---- the decision engines' encoders: shaders/chain_enc.comp ------------------------
 * Its six modes as kernels, the shader's arithmetic but for two choices that keep a row
 * nearer the engines' own: the LayerNorm's mean and variance are summed in double, as
 * laya.c and gliner_decide.c sum them, and the GELU is erff's (the shader fits its own
 * erfc, GLSL having none). */
static inline float cc_enc_act(float x, int k) {
    if (k == 1) return 0.5f * x * (1.0f + sycl::erf(x * 0.70710678118654752f));
    if (k == 2) return 0.5f * x *
        (1.0f + sycl::tanh(0.7978845608028654f * (x + 0.044715f * x * x * x)));
    if (k == 3) return sycl::fmax(x, 0.0f);
    return x;
}
static inline double cc_enc_sum256(double v, double *red, int tid) {
    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    red[tid] = v;
    it.barrier();
    for (int off = 128; off > 0; off >>= 1) {
        if (tid < off) red[tid] += red[tid + off];
        it.barrier();
    }
    double s = red[0];
    it.barrier();
    return s;
}
/* mode 0: a block a row, its values in registers before any write (y may be a or b) */
static void cc_enc_norm_k(float *a, const float *b, const float *prm, float *y,
                          CcEncNorm p) {
    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto &red =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<double[256]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    int r = it.get_group(2), tid = it.get_local_id(2);
    if (r >= p.rows) return;
    const size_t ab = (size_t)p.a_off + (size_t)r * p.a_row, bb = (size_t)p.b_off + (size_t)r * p.b_row;
    float t[16];
    double s = 0.0;
    for (int k = 0; k < 16; k++) {
        int i = tid + 256 * k;
        t[k] = 0.0f;
        if (i < p.D) {
            float v = a[ab + i];
            if (p.flags & CC_ENC_ADD) v = v + b[bb + i];
            t[k] = v;
            s += v;
        }
    }
    double mean = cc_enc_sum256(s, red, tid) / p.D, q = 0.0;
    for (int k = 0; k < 16; k++) { int i = tid + 256 * k; if (i < p.D) { double c = (double)t[k] - mean; q += c * c; } }
    double var = cc_enc_sum256(q, red, tid) / p.D;
    float rstd = (float)(1.0 / sycl::sqrt(var + (double)p.eps));
    const size_t yb = (size_t)p.y_off + (size_t)r * p.y_row;
    for (int k = 0; k < 16; k++) {
        int i = tid + 256 * k;
        if (i >= p.D) continue;
        if (p.flags & CC_ENC_SUM) a[ab + i] = t[k];
        float o = (float)((double)t[k] - mean) * rstd * prm[p.w_off + i];
        if (p.bias_off >= 0) o = o + prm[p.bias_off + i];
        y[yb + i] = o;
    }
}
/* mode 1 */
static void cc_enc_bias_k(float *y, const float *prm, CcEncBias p) {
    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    long long i =
        (long long)it.get_group(2) * 256 + it.get_local_id(2);
    if (i >= p.n) return;
    int r = (int)(i / p.O), o = (int)(i - (long long)r * p.O);
    size_t at = (size_t)p.y_off + (size_t)r * p.y_row + o;
    float v = y[at];
    if (p.b_off >= 0) v = v + prm[p.b_off + o];
    y[at] = cc_enc_act(v, p.act);
}
/* mode 2 */
static void cc_enc_geglu_k(const float *a, const float *prm, float *y, CcEncGeglu p) {
    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    long long i =
        (long long)it.get_group(2) * 256 + it.get_local_id(2);
    if (i >= p.n) return;
    int r = (int)(i / p.I), j = (int)(i - (long long)r * p.I);
    size_t ab = (size_t)p.a_off + (size_t)r * p.a_row;
    float u = a[ab + j], g = a[ab + p.I + j];
    if (p.b_off >= 0) { u = u + prm[p.b_off + j]; g = g + prm[p.b_off + p.I + j]; }
    y[(size_t)p.y_off + (size_t)r * p.y_row + j] = cc_enc_act(u, p.act) * g;
}
/* mode 3: a thread a (row, query or key, head, pair) */
static void cc_enc_rope_k(float *x, const float *tab, const int *pos, CcEncRope p) {
    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    long long i =
        (long long)it.get_group(2) * 256 + it.get_local_id(2);
    if (i >= p.n) return;
    int half = p.hd / 2, per_row = 2 * p.H * half;
    int r = (int)(i / per_row), rem = (int)(i - (long long)r * per_row);
    int part = rem / (p.H * half), rem2 = rem - part * p.H * half, hh = rem2 / half, k = rem2 - hh * half;
    size_t vb = (size_t)p.x_off + (size_t)r * p.x_row + (size_t)part * p.part_off + (size_t)hh * p.hd;
    size_t ps = (size_t)pos[r] * half + k;
    float c = tab[p.cos_off + ps], sn = tab[p.sin_off + ps];
    float x0 = x[vb + k], x1 = x[vb + k + half];
    x[vb + k] = x0 * c - x1 * sn;
    x[vb + k + half] = x1 * c + x0 * sn;
}
/* mode 4 */
static void cc_enc_addrow_k(float *y, const float *tab, const int *idx, CcEncAddRow p) {
    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    long long i =
        (long long)it.get_group(2) * 256 + it.get_local_id(2);
    if (i >= p.n) return;
    int r = (int)(i / p.D), d = (int)(i - (long long)r * p.D);
    y[(size_t)p.y_off + (size_t)r * p.y_row + d] += tab[(size_t)p.t_off + (size_t)idx[r] * p.D + d];
}
/* mode 5: a block a (head, 16 rows), the union of their ranges in tiles of 32 rows, the
 * softmax online (a running max and sum a row) */
#define CC_EBQ 16
#define CC_EBK 32
#define CC_EHD 128
static void cc_enc_attn_k(const float *qkv, const float *c2p, const float *p2c,
                          float *o, const int *rng, const int *ridx,
                          const int *pos, CcEncAttn p) {
    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto &qs = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[CC_EBQ * CC_EHD]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &kv = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[CC_EBK * (CC_EHD + 1)]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &sc = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[CC_EBQ * CC_EBK]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &mrow =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[CC_EBQ]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &lrow =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[CC_EBQ]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &crow =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[CC_EBQ]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &rlo =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<int[CC_EBQ]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &rhi =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<int[CC_EBQ]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &rpos =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<int[CC_EBQ]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &klo = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &khi = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int S = p.S, hd = p.hd, HDP = CC_EHD + 1, rel = p.flags & (CC_ENC_C2P | CC_ENC_P2C);
    int h = it.get_group(2), b0 = it.get_group(1) * CC_EBQ,
        tid = it.get_local_id(2);
    int nrow = S - b0 < CC_EBQ ? S - b0 : CC_EBQ;
    for (int e = tid; e < nrow * hd; e += 256) {
        int i = e / hd, d = e - i * hd;
        qs[i * CC_EHD + d] = qkv[(size_t)p.q_off + (size_t)(b0 + i) * p.kv_row + (size_t)h * hd + d];
    }
    if (tid < CC_EBQ) {
        if (tid < nrow) {
            rlo[tid] = rng[2 * (b0 + tid)]; rhi[tid] = rng[2 * (b0 + tid) + 1];
            rpos[tid] = rel ? pos[b0 + tid] : 0;
        } else { rlo[tid] = 1; rhi[tid] = 0; rpos[tid] = 0; }
        mrow[tid] = -3.0e38f; lrow[tid] = 0.0f;
    }
    it.barrier(sycl::access::fence_space::local_space);
    if (tid == 0) {
        int lo = 0x7fffffff, hi = -1;
        for (int i = 0; i < nrow; i++) if (rlo[i] <= rhi[i]) {
            lo = sycl::min(lo, rlo[i]); hi = sycl::max(hi, rhi[i]);
        }
        klo = lo; khi = hi;
    }
    it.barrier(sycl::access::fence_space::local_space);
    int pi = tid >> 4, pd = tid & 15;   /* the PV step: row pi, dims pd + 16k */
    float acc[8];
    for (int k = 0; k < 8; k++) acc[k] = 0.0f;
    for (int j0 = klo; j0 <= khi; j0 += CC_EBK) {
        int nk = khi - j0 + 1 < CC_EBK ? khi - j0 + 1 : CC_EBK;
        for (int e = tid; e < nk * hd; e += 256) {
            int jj = e / hd, d = e - jj * hd;
            kv[jj * HDP + d] = qkv[(size_t)p.k_off + (size_t)(j0 + jj) * p.kv_row + (size_t)h * hd + d];
        }
        it.barrier(sycl::access::fence_space::local_space);
        for (int u = 0; u < 2; u++) {
            int pr = tid + 256 * u, i = pr / CC_EBK, jj = pr - i * CC_EBK, j = j0 + jj;
            float sv = 0.0f;
            if (i < nrow && jj < nk && j >= rlo[i] && j <= rhi[i]) {
                float sum = 0.0f;
                for (int d = 0; d < hd; d++) sum += qs[i * CC_EHD + d] * kv[jj * HDP + d];
                if (rel) {
                    int rr = ridx[rpos[i] - pos[j] + p.rc] - p.r0;
                    if (p.flags & CC_ENC_C2P) sum += c2p[(size_t)p.c_off + ((size_t)h * S + b0 + i) * p.nr + rr];
                    if (p.flags & CC_ENC_P2C) sum += p2c[(size_t)p.p_off + ((size_t)h * S + j) * p.nr + rr];
                }
                sv = sum * p.scale;
            }
            sc[i * CC_EBK + jj] = sv;
        }
        it.barrier(sycl::access::fence_space::local_space);
        if (tid < CC_EBQ) {
            int i = tid;
            float m = mrow[i], mn = m;
            int jlo = rlo[i] - j0 > 0 ? rlo[i] - j0 : 0, jhi = rhi[i] - j0 < nk - 1 ? rhi[i] - j0 : nk - 1;
            for (int jj = jlo; jj <= jhi; jj++)
                mn = sycl::fmax(mn, sc[i * CC_EBK + jj]);
            float l = 0.0f;
            for (int jj = 0; jj < CC_EBK; jj++) {
                float e = jj >= jlo && jj <= jhi
                              ? sycl::native::exp(sc[i * CC_EBK + jj] - mn)
                              : 0.0f;
                sc[i * CC_EBK + jj] = e;
                l += e;
            }
            float cr = sycl::native::exp(m - mn);
            crow[i] = cr;
            lrow[i] = lrow[i] * cr + l;
            mrow[i] = mn;
        }
        it.barrier(sycl::access::fence_space::local_space);
        for (int e = tid; e < nk * hd; e += 256) {
            int jj = e / hd, d = e - jj * hd;
            kv[jj * HDP + d] = qkv[(size_t)p.v_off + (size_t)(j0 + jj) * p.kv_row + (size_t)h * hd + d];
        }
        it.barrier(sycl::access::fence_space::local_space);
        if (pi < nrow) {
            float cr = crow[pi];
            for (int k = 0; k < 8; k++) {
                int d = pd + 16 * k;
                if (d >= hd) break;
                float t = acc[k] * cr;
                for (int jj = 0; jj < nk; jj++) t += sc[pi * CC_EBK + jj] * kv[jj * HDP + d];
                acc[k] = t;
            }
        }
        it.barrier(sycl::access::fence_space::local_space);
    }
    if (pi < nrow) {
        float l = lrow[pi], inv = l > 0.0f ? 1.0f / l : 0.0f;
        for (int k = 0; k < 8; k++) {
            int d = pd + 16 * k;
            if (d >= hd) break;
            o[(size_t)p.o_off + (size_t)(b0 + pi) * p.o_row + (size_t)h * hd + d] = acc[k] * inv;
        }
    }
}
/* mode 6: a block 16 rows by 64 buckets of one head, the inputs 32 at a time */
#define CC_EBR 64
#define CC_EDC 32
static void cc_enc_rel_k(const float *x, const float *pt, float *y, CcEncRel p) {
    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto &xs = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[CC_EBQ * (CC_EDC + 1)]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &ps = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[CC_EBR * (CC_EDC + 1)]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    int t0 = it.get_group(2) * CC_EBR,
        i0 = it.get_group(1) * CC_EBQ, h = it.get_group(0);
    int tid = it.get_local_id(2), tt = tid & 63,
        iq = tid >> 6; /* bucket tt, rows 4iq .. 4iq+3 */
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int d0 = 0; d0 < p.hd; d0 += CC_EDC) {
        int dn = p.hd - d0 < CC_EDC ? p.hd - d0 : CC_EDC;
        for (int e = tid; e < CC_EBQ * CC_EDC; e += 256) {
            int i = e / CC_EDC, d = e - i * CC_EDC;
            xs[i * (CC_EDC + 1) + d] = i0 + i < p.S && d < dn ? x[(size_t)p.x_off + (size_t)(i0 + i) * p.x_row + (size_t)h * p.hd + d0 + d] : 0.0f;
        }
        for (int e = tid; e < CC_EBR * CC_EDC; e += 256) {
            int t = e / CC_EDC, d = e - t * CC_EDC;
            ps[t * (CC_EDC + 1) + d] = t0 + t < p.nr && d < dn ? pt[(size_t)p.p_off + (size_t)(p.r0 + t0 + t) * p.p_row + (size_t)h * p.hd + d0 + d] : 0.0f;
        }
        it.barrier(sycl::access::fence_space::local_space);
        for (int d = 0; d < dn; d++) {
            float pv = ps[tt * (CC_EDC + 1) + d];
            for (int k = 0; k < 4; k++) acc[k] += xs[(4 * iq + k) * (CC_EDC + 1) + d] * pv;
        }
        it.barrier(sycl::access::fence_space::local_space);
    }
    if (t0 + tt < p.nr)
        for (int k = 0; k < 4; k++) {
            int i = i0 + 4 * iq + k;
            if (i < p.S) y[(size_t)p.y_off + ((size_t)h * p.S + i) * p.nr + t0 + tt] = acc[k];
        }
}
static int cc_enc_norm(CcBuf *a, CcBuf *b, CcBuf *prm, CcBuf *y, const CcEncNorm *p) {
    CcCtx *c = live();
    CC_OWN(c, "cc_enc_norm", a, b, prm, y);
    if (!c || !a || !prm || !y || !p || p->rows < 1 || p->D < 1 || p->D > 4096 || ((p->flags & CC_ENC_ADD) && !b)) return 0;
    if (!c->fp64) return 0;   /* the sums are doubles, as the engines' (laya.c, gliner_decide.c) */
    float *ap = fpw(a), *yp = fpw(y); const float *bp = fp(b), *pp_ = fp(prm); const CcEncNorm pp = *p;
    CC_K(c, dim3((unsigned)p->rows), 256, cc_enc_norm_k(ap, bp, pp_, yp, pp));
    return launched(c, "encoder norm");
}
static int cc_enc_bias(CcBuf *y, CcBuf *prm, const CcEncBias *p) {
    CcCtx *c = live();
    CC_OWN(c, "cc_enc_bias", y, prm);
    if (!c || !y || !p || p->n < 1 || p->O < 1 || (p->b_off >= 0 && !prm)) return 0;
    float *yp = fpw(y); const float *pp_ = fp(prm); const CcEncBias pp = *p;
    CC_K(c, dim3(grid1((size_t)p->n, 256)), 256, cc_enc_bias_k(yp, pp_, pp));
    return launched(c, "encoder bias");
}
static int cc_enc_geglu(CcBuf *a, CcBuf *prm, CcBuf *y, const CcEncGeglu *p) {
    CcCtx *c = live();
    CC_OWN(c, "cc_enc_geglu", a, prm, y);
    if (!c || !a || !y || !p || p->n < 1 || p->I < 1 || (p->b_off >= 0 && !prm)) return 0;
    const float *ap = fp(a), *pp_ = fp(prm); float *yp = fpw(y); const CcEncGeglu pp = *p;
    CC_K(c, dim3(grid1((size_t)p->n, 256)), 256, cc_enc_geglu_k(ap, pp_, yp, pp));
    return launched(c, "encoder geglu");
}
static int cc_enc_rope(CcBuf *x, CcBuf *tab, CcBuf *pos, const CcEncRope *p) {
    CcCtx *c = live();
    CC_OWN(c, "cc_enc_rope", x, tab, pos);
    if (!c || !x || !tab || !pos || !p || p->n < 1 || p->H < 1 || p->hd < 2) return 0;
    float *xp = fpw(x); const float *tp = fp(tab); const int *psp = (const int *)pos->d; const CcEncRope pp = *p;
    CC_K(c, dim3(grid1((size_t)p->n, 256)), 256, cc_enc_rope_k(xp, tp, psp, pp));
    return launched(c, "encoder rope");
}
static int cc_enc_addrow(CcBuf *y, CcBuf *tab, CcBuf *idx, const CcEncAddRow *p) {
    CcCtx *c = live();
    CC_OWN(c, "cc_enc_addrow", y, tab, idx);
    if (!c || !y || !tab || !idx || !p || p->n < 1 || p->D < 1) return 0;
    float *yp = fpw(y); const float *tp = fp(tab); const int *ip = (const int *)idx->d; const CcEncAddRow pp = *p;
    CC_K(c, dim3(grid1((size_t)p->n, 256)), 256, cc_enc_addrow_k(yp, tp, ip, pp));
    return launched(c, "encoder add row");
}
static int cc_enc_attn(CcBuf *qkv, CcBuf *c2p, CcBuf *p2c, CcBuf *o, CcBuf *rng, CcBuf *ridx, CcBuf *pos, const CcEncAttn *p) {
    CcCtx *c = live();
    CC_OWN(c, "cc_enc_attn", qkv, c2p, p2c, o, rng, ridx, pos);
    int rel = p ? p->flags & (CC_ENC_C2P | CC_ENC_P2C) : 0;
    if (!c || !qkv || !o || !rng || !p || p->S < 1 || p->H < 1 || p->hd < 1 || p->hd > CC_EHD || (p->S + CC_EBQ - 1) / CC_EBQ > 65535 ||
        (rel && (!ridx || !pos)) || ((p->flags & CC_ENC_C2P) && !c2p) || ((p->flags & CC_ENC_P2C) && !p2c)) return 0;
    const float *qp = fp(qkv), *cp = fp(c2p), *pcp = fp(p2c); float *op = fpw(o);
    const int *rp = (const int *)rng->d, *rip = ridx ? (const int *)ridx->d : nullptr, *psp = pos ? (const int *)pos->d : nullptr;
    const CcEncAttn pp = *p;
    CC_K(c, dim3((unsigned)p->H, (unsigned)((p->S + CC_EBQ - 1) / CC_EBQ)), 256, cc_enc_attn_k(qp, cp, pcp, op, rp, rip, psp, pp));
    return launched(c, "encoder attention");
}
static int cc_enc_rel(CcBuf *x, CcBuf *pt, CcBuf *y, const CcEncRel *p) {
    CcCtx *c = live();
    CC_OWN(c, "cc_enc_rel", x, pt, y);
    if (!c || !x || !pt || !y || !p || p->S < 1 || p->H < 1 || p->hd < 1 || p->nr < 1 || p->H > 65535 ||
        (p->S + CC_EBQ - 1) / CC_EBQ > 65535) return 0;
    const float *xp = fp(x), *ptp = fp(pt); float *yp = fpw(y); const CcEncRel pp = *p;
    CC_K(c, dim3((unsigned)((p->nr + CC_EBR - 1) / CC_EBR), (unsigned)((p->S + CC_EBQ - 1) / CC_EBQ), (unsigned)p->H), 256,
         cc_enc_rel_k(xp, ptp, yp, pp));
    return launched(c, "encoder relative products");
}
/* this file's kernels and the backend's loaded on the chain's device now: the code is built ahead
 * of time for the device (AOT), the runtime loads a module at the first launch from it; nothing
 * to ask for, so 1 */
static int cc_preload(void) {
    return live() ? 1 : 0;
}
#include "cuda_dsv4c.dp.hpp"   /* DeepSeek V4.1's layer ops (cuda_dsv4c.cuh's port) */
static const ColiCudaChainOps g_cc_ops = {
    sizeof(ColiCudaChainOps),
    cc_init, cc_ready, cc_lost_, cc_shutdown, cc_device, cc_device_now,
    cc_buf, cc_free, cc_reserve, cc_ptr, cc_bytes,
    cc_begin, cc_submit, cc_finish,
    cc_copy, cc_zero, cc_copy_regions, cc_write, cc_read,
    cc_matmul, cc_norm, cc_rope, cc_attn, cc_dnconv, cc_dnrec, cc_ew,
    cc_stats,
    cc_qsa, cc_ple,
    cc_route, cc_moe, cc_signal, cc_wait_host, coli_cuda_tensor_dev,
    cc_fetch, cc_fetch_wait, cc_fetch_done, cc_dev_addr,
    cc_moe_stage_op, cc_host_up, cc_host_up_done,
    cc_attn_w,
    cc_muse,
    cc_attn_w2,
    cc_kv16,
    cc_rope_f16,
    cc_pick,
    cc_matmul_rows,
    cc_kvh,
    cc_attn_h,
    cc_attn_wh,
    cc_copy_regions_h,
    cc_enc_norm, cc_enc_bias, cc_enc_geglu, cc_enc_rope, cc_enc_addrow, cc_enc_attn, cc_enc_rel,
    cc_lfm3,
    cc_matmul_glu,
    cc_addnorm,
    cc_pick_rows,
    cc_host_up_n,
    cc_preload,
    cc_moe_cols,
    cc_matmul_hcmix,
    cc_mhc, cc_ds_attn, cc_ds_rope, cc_ds_compress, cc_ds_score, cc_ds_cand, cc_ds_topk, cc_ds_engram,
    /* upstream's operations for other models (GLM-5.3, Kimi K3, DFlash2, Inkling) and the banked
     * fetch: not in this module; the engine finds them empty and keeps its other path */
    nullptr /* kda_conv */, nullptr /* kda_rec */, nullptr /* mla_core */, nullptr /* mla_rope */,
    nullptr /* mla_lnorm */, nullptr /* mla_hgemv */, nullptr /* dsa_pool_keys */, nullptr /* dsa_pool_select */,
    nullptr /* dsa_select */, nullptr /* mla_hgemv_q8 */, nullptr /* ares_mix */, nullptr /* situ */,
    nullptr /* dflash */, nullptr /* dfconv */, nullptr /* dftopk */, nullptr /* dftap */,
    nullptr /* wait_host_if */, nullptr /* sconv */, nullptr /* relattn */, nullptr /* ink2_mix */,
    nullptr /* ink2_clip */, nullptr /* ink2_dconv */, nullptr /* fetch_at */, nullptr /* fetch_wait_at */,
    nullptr /* fetch_done_at */,
    cc_dnconv_mx, cc_dnrec_mx,
    cc_attn_h_mx, cc_copy_regions_h_mx,
    cc_matmul_grp,
    cc_ds_attn_rows,
    cc_moe_tail,
    cc_mark, cc_reached,
};
extern "C" const ColiCudaChainOps *coli_cuda_chain_ops(void) { return &g_cc_ops; }
