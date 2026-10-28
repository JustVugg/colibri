/* cuda_chain.cu -- the dense chain on a CUDA device: cuda_chain.h's table.
 *
 * Buffers, frames and the chain's kernels. Each kernel is the arithmetic of the
 * shader of the same name in shaders/chain_*.comp, ported line for line (the sums in
 * the same order, the same sentinels), so the two chains give the same bits for the
 * same bits in and tests/test_vk_chain.c's references hold here too. The matmul is
 * the backend's own GEMV / GEMM kernel (coli_cuda_tensor_gemm_async) on the chain's
 * stream.
 *
 * One context per CUDA device: a non-blocking stream (the backend's synchronous
 * cudaMemcpy on the legacy default stream must not serialize the chain behind the
 * expert tier's work), one event per frame, the lost flag. UP and DOWN buffers are
 * page-locked host memory mapped into the device (cudaHostAlloc, cudaHostAllocMapped):
 * the kernels read or write them over the bus, the host through cc_ptr, nothing
 * copied. A CUDA error in any call marks the context lost: every later call returns
 * 0 and the engine rebuilds on the CPU, as with the Vulkan chain. */
#include "backend_cuda.h"
#include "backend_gpu_compat.h"
#define COLI_CUDA_CHAIN_NO_WRAPPERS
#include "cuda_chain.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>

/* Two of backend_cuda.cu's, for this file alone (the same DLL / object set, never the
 * host's: not in backend_cuda.h, not in the loader): a resident tensor's format and
 * shape, and its GEMV / GEMM on a stream the caller owns (a cudaStream_t, NULL = the
 * device's default) with device pointers, no transfer, no synchronization. */
extern "C" int coli_cuda_tensor_shape(const ColiCudaTensor *t, int *fmt, int *I, int *O);
extern "C" int coli_cuda_tensor_gemm_async(ColiCudaTensor *t, float *y_dev, const float *x_dev, int S, void *stream);
/* and the routed experts' kernels (cc_moe), on the same terms, and a resident tensor's device copy */
extern "C" int coli_cuda_tensor_dev(const ColiCudaTensor *t, const void **w, const float **sc, int *gs);
extern "C" int coli_cuda_moe_stage(int device, void *stage, size_t slot_bytes, int slots);
extern "C" int coli_cuda_moe_async(int device, const CcExpert *tab, const float *x, const int *ids, const float *w,
                                   float *hid, float *y, const float *hy, const CcMoe *p, void *stream);

struct CcBuf { int kind, dev; size_t bytes; float *d; void *h; };

namespace {
#define CC_UP_MARKS 1024   /* host_up's marks in flight (three a promoted expert) */
struct CcCtx {
    int device, up, lost, in_frame;
    int major;   /* the device's compute capability: cc_attn_fk's tensor cores from 7 */
    cudaStream_t stream;
    cudaEvent_t ev; int ev_ok, ev_pending;
    /* staging (cc_fetch): the copy stream, its copies' event and the buffer's last reader's */
    cudaStream_t cs; cudaEvent_t ev_fetched, ev_free; int cs_ok;
    /* host_up: the copies' marks, a ring of CC_UP_MARKS events on the copy stream */
    cudaEvent_t up_ev[CC_UP_MARKS]; unsigned up_seq, up_made; cudaEvent_t up_after;
    float *adec; size_t adec_bytes;   /* cc_attn_dec_k's blocks */
    /* the scratch a resize retired: freed once a later frame has been waited for (a stream
     * sync here could wait on a cc_wait_host kernel whose host side comes after this frame) */
    float *adec_old[4]; int adec_n_old;
};
CcCtx g_cc[COLI_CUDA_MAX_DEVICES];
/* the retired attention scratches, after the stream was seen to pass the frames that read them */
static void cc_adec_reap(CcCtx *c) {
    for (int i = 0; i < c->adec_n_old; i++) cudaFree(c->adec_old[i]);
    c->adec_n_old = 0;
}
int g_ncc, g_cur = -1;   /* g_cur: index into g_cc of the current context */
CcStats g_st;

double cc_now_ms(void) {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}
CcCtx *cur(void) { return g_cur >= 0 ? &g_cc[g_cur] : nullptr; }
CcCtx *ctx_of(int device) { for (int i = 0; i < g_ncc; i++) if (g_cc[i].device == device) return &g_cc[i]; return nullptr; }
/* every call: the context's device current (the backend's other calls switch it) */
CcCtx *live(void) {
    CcCtx *c = cur();
    if (!c || !c->up || c->lost) return nullptr;
    if (cudaSetDevice(c->device) != cudaSuccess) { c->lost = 1; (void)cudaGetLastError(); return nullptr; }
    return c;
}
int cc_ok(CcCtx *c, cudaError_t e, const char *what) {
    if (e == cudaSuccess) return 1;
    std::fprintf(stderr, "[CUDA] chain: %s: %s -- the device is lost to the chain\n", what, cudaGetErrorString(e));
    (void)cudaGetLastError();
    if (c) c->lost = 1;
    return 0;
}
int launched(CcCtx *c, const char *what) { g_st.ops++; return cc_ok(c, cudaGetLastError(), what); }
unsigned grid1(size_t n, unsigned block) { size_t g = (n + block - 1) / block; return g > 0x7fffffffu ? 0x7fffffffu : (unsigned)g; }
const float *fp(const CcBuf *b) { return b ? b->d : nullptr; }
float *fpw(CcBuf *b) { return b ? b->d : nullptr; }
}

/* ---- kernels: shaders/chain_*.comp, line for line --------------------------- */
__global__ static void cc_norm_k(const float *__restrict__ x, const float *__restrict__ w, float *__restrict__ y, CcNorm p) {
    __shared__ float red[256];
    int g = blockIdx.x;
    if (g >= p.nseg) return;
    int row = g / p.per_row, j = g - row * p.per_row;
    int xb = p.x_off + row * p.x_row + j * p.x_seg;
    int yb = p.y_off + row * p.y_row + j * p.y_seg;
    int wb = p.w_off + (p.w_mod > 0 ? (j % p.w_mod) * p.D : 0);
    int tid = threadIdx.x;
    float acc = 0.f;
    const bool pre = (p.flags & 8) != 0;
    for (int i = tid; i < p.D; i += 256) { float v = pre ? x[xb + i] * w[wb + i] : x[xb + i]; acc += v * v; }
    red[tid] = acc;
    __syncthreads();
    for (int off = 128; off > 0; off >>= 1) {
        if (tid < off) red[tid] += red[tid + off];
        __syncthreads();
    }
    float r = (p.flags & 4) ? 1.f / sqrtf(red[0] + p.eps) : 1.f / sqrtf(red[0] / (float)p.D + p.eps);
    for (int i = tid; i < p.D; i += 256) {
        float wv = (p.flags & 2) ? 1.f : ((p.flags & 1) ? 1.f + w[wb + i] : w[wb + i]);
        y[yb + i] = x[xb + i] * (pre ? w[wb + i] : wv) * r * p.post;
    }
}
/* kv16.h's kv16_round_bits line for line: clamped, the nearest float16 (ew's CC_EW_F16) */
__device__ static inline float cc_f16_round(float x) {
    if (x > 65504.f) x = 65504.f; else if (x < -65504.f) x = -65504.f;
    unsigned u = __float_as_uint(x), s = u & 0x80000000u, m = u & 0x7fffffffu;
    if (m > 0x7f800000u) u = u | 0x400000u;
    else if (m >= 0x38800000u) u = s | ((m + 0xfffu + ((m >> 13) & 1u)) & 0xffffe000u);
    else {
        unsigned e = m >> 23, mm = (m & 0x7fffffu) | 0x800000u;
        if (e < 102u) u = s;
        else {
            unsigned sh = 126u - e, n = (mm + (1u << (sh - 1u)) - 1u + ((mm >> sh) & 1u)) >> sh;
            u = s | __float_as_uint(__fmul_rn((float)n, 5.9604644775390625e-8f));
        }
    }
    return __uint_as_float(u);
}
/* one rotate-half pair, its roundings spelled out: a * c - b * s and b * c + a * s leave the
 * compiler free to fuse either product, and it fused a different one once the pair moved here.
 * These are the ones nvcc picked for cc_rope_k on sm_120 before, for cc_rope_k and cc_rope_f16_k
 * alike, so their bits agree on every target and, on sm_120, are the rope's of before */
__device__ static inline void cc_rope_pair(float a, float b, float c, float s, float *ra, float *rb) {
    *ra = __fmaf_rn(a, c, -__fmul_rn(b, s));
    *rb = __fmaf_rn(a, s, __fmul_rn(b, c));
}
__global__ static void cc_rope_k(float *__restrict__ x, const float *__restrict__ cs, CcRope p) {
    long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
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
__global__ static void cc_rope_f16_k(float *__restrict__ x, float *__restrict__ v, const float *__restrict__ cs, CcRope p, int seg) {
    long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
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
/* one block of 128 per (head, row); K and V rows read once per tile of 128 positions;
 * hd <= 256: two dims per thread */
/* win > 0: a sliding window, the row's last win positions; snk: a sink logit per head at
 * snk[sink_off + h], in the denominator only (CcAttnW); ring > 0: the cache a ring of that
 * many rows, position t in row t % ring (a sliding layer's, cc_attn_wr) */
__global__ static void __launch_bounds__(128) cc_attn_k(const float *__restrict__ q, const float *__restrict__ kc,
        const float *__restrict__ vc, float *__restrict__ o, const float *__restrict__ gt, const int *__restrict__ sel, CcAttn p,
        int win = 0, const float *__restrict__ snk = nullptr, int sink_off = 0, int ring = 0) {
    __shared__ float qs[256];
    __shared__ float pe[128];
    __shared__ int   pt[128];
    __shared__ float red[128];
    int h = blockIdx.x, s = blockIdx.y, tid = threadIdx.x;
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
    __syncthreads();
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
            for (int d = 0; d < p.hd; d++) a += qs[d] * kc[kb + d];
            sv = a * p.scale;
        }
        red[tid] = sv;
        __syncthreads();
        for (int off = 64; off > 0; off >>= 1) {
            if (tid < off) red[tid] = fmaxf(red[tid], red[tid + off]);
            __syncthreads();
        }
        float mn = fmaxf(m, red[0]);
        __syncthreads();
        float e = i < n ? expf(sv - mn) : 0.f;
        pe[tid] = e;
        red[tid] = e;
        __syncthreads();
        for (int off = 64; off > 0; off >>= 1) {
            if (tid < off) red[tid] += red[tid + off];
            __syncthreads();
        }
        float corr = expf(m - mn);
        l = l * corr + red[0];
        acc0 *= corr; acc1 *= corr;
        int cnt = min(128, n - t0);
        int d0 = tid, d1 = tid + 128;
        for (int j = 0; j < cnt; j++) {
            int vb = p.v_off + (kbase + pt[j]) * p.hd;
            float w = pe[j];
            if (d0 < p.hd) acc0 += w * vc[vb + d0];
            if (d1 < p.hd) acc1 += w * vc[vb + d1];
        }
        m = mn;
        __syncthreads();
    }
    if (snk) {   /* the sink: one more term of the denominator, with no value */
        float sv = snk[sink_off + h], mn = fmaxf(m, sv), corr = expf(m - mn);
        l = l * corr + expf(sv - mn);
        acc0 *= corr; acc1 *= corr;
    }
    float inv = l > 0.f ? 1.f / l : 0.f;
    int ob = p.o_off + s * p.o_row + h * p.hd;
    int gb = p.g_off + s * p.g_row + h * p.g_seg;
    for (int k = 0; k < 2; k++) {
        int d = tid + 128 * k;
        if (d >= p.hd) break;
        float v = (k == 0 ? acc0 : acc1) * inv;
        if (p.has_gate) v *= 1.f / (1.f + expf(-gt[gb + d]));
        o[ob + d] = v;
    }
}
/* A prompt chunk's plain causal attention on the tensor cores (cc_attn with no selection
 * list, from COLI_CUDA_CHAIN_FLASH rows): chain_attn_flash.comp's way, WMMA f16 x f16 ->
 * f32. A block of 128 (four warps) takes 16 query rows: TPW = 16 / GQ tokens x the GQ query
 * heads of one kv head, so each K and V block is staged once for all of them. Two passes
 * over the keys in blocks of 64: the first finds each row's max m and sum l, the second adds
 * exp(s - m) * V into the accumulators, divided by l at the end (no rescaling: the output
 * stays in the fragments). Q is scaled before its f16 rounding and kept in registers; K and
 * V are rounded to f16 as they are staged, row-major with a stride of HD + 8. Key t is
 * valid for row r iff t <= pos_base + token(r). Not bit-identical to cc_attn_k (f16 operands).
 * Grid: x = ceil(S / TPW) token groups, y = kv heads. */
#define CC_FA_BC 64
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700 && !defined(__HIP__) && !defined(__HIP_PLATFORM_AMD__)
#define CC_FA_BODY 1
#else
#define CC_FA_BODY 0
#endif
#if CC_FA_BODY
/* cc_attn_fk<HD, true>: a key block of K or V (keys k0 .. k0 + 63) into skv 16 bytes a load. Each thread starts a
 * group of float4 loads (16 at head dim 256) before it converts and stores any: the scalar loop waits on each 4-byte
 * load as it uses it, with one warp a scheduler to hide the wait (on a T4, the last chunk of an 8192-token prompt:
 * two thirds of the warps' samples on the two conversions after those loads). The same layout and rounding as
 * the scalar loop: the same bits. The groups one after the other, not unrolled: Q's fragments already hold 128
 * registers. Needs src + off 16-byte aligned (cc_attn checks). */
template<int HD>
__device__ static inline void cc_fa_stage4(__half *skv, const float *__restrict__ src, int off, int g, int cap, int k0,
                                           int nt, int ring, int tid) {
    constexpr int LD = HD + 8, NV = CC_FA_BC * HD / 4 / 128, NB = NV < 16 ? NV : 16;   /* a thread's float4s, a group */
    static_assert(HD % 4 == 0 && NV % NB == 0, "whole float4s and whole groups");
    #pragma unroll 1
    for (int b0 = 0; b0 < NV; b0 += NB) {
        float4 r[NB];
        #pragma unroll
        for (int i = 0; i < NB; i++) {
            int e = (tid + 128 * (b0 + i)) * 4, kl = e / HD, d = e - kl * HD, key = k0 + kl;
            r[i] = key < nt ? __ldg(reinterpret_cast<const float4 *>(src + off + (g * cap + (ring ? key % ring : key)) * HD + d))
                            : make_float4(0.f, 0.f, 0.f, 0.f);
        }
        #pragma unroll
        for (int i = 0; i < NB; i++) {
            int e = (tid + 128 * (b0 + i)) * 4, kl = e / HD, d = e - kl * HD;
            __half2 lo = __floats2half2_rn(r[i].x, r[i].y), hi = __floats2half2_rn(r[i].z, r[i].w);
            *reinterpret_cast<uint2 *>(skv + kl * LD + d) = make_uint2(*reinterpret_cast<unsigned *>(&lo), *reinterpret_cast<unsigned *>(&hi));
        }
    }
}
#endif
template<int HD, bool F4 = false>   /* F4: K and V staged 16 bytes a load (cc_fa_stage4) */
__global__ static void __launch_bounds__(128) cc_attn_fk(const float *__restrict__ q, const float *__restrict__ kc,
        const float *__restrict__ vc, float *__restrict__ o, const float *__restrict__ gt, CcAttn p, int GQ,
        int win = 0, const float *__restrict__ snk = nullptr, int sink_off = 0, int ring = 0) {   /* CcAttnW's window and sinks, a ring */
#if CC_FA_BODY
    using namespace nvcuda;
    constexpr int LD = HD + 8, NDT = HD / 16, NDS = NDT / 4, LS = CC_FA_BC + 4, LP = CC_FA_BC + 8;
    __shared__ __align__(32) __half skv[CC_FA_BC * LD];     /* K, then V, of a key block; Q first; the output last */
    __shared__ __align__(32) float ss[16 * LS];              /* the block's scores, row r key c at r * LS + c */
    __shared__ __align__(32) __half sp[16 * LP];             /* exp(s - m) */
    __shared__ float sm[16], sl[16];
    int tid = threadIdx.x, warp = tid >> 5;
    int TPW = 16 / GQ, g = blockIdx.y, t0 = blockIdx.x * TPW, S = p.S;
    int tlast = min(t0 + TPW, S) - 1, nt = p.pos_base + S;
    int nkb = (p.pos_base + tlast) / CC_FA_BC + 1;
    /* a window: the blocks before the first row's first position hold no key of any row here */
    int kb_lo = win > 0 && p.pos_base + t0 - win + 1 > 0 ? (p.pos_base + t0 - win + 1) / CC_FA_BC : 0;
    /* Q, scaled, into the fragments (staged through skv) */
    for (int e = tid; e < 16 * HD; e += 128) {
        int r = e / HD, d = e - r * HD, tk = t0 + r / GQ, h = g * GQ + r % GQ;
        float v = tk < S ? q[p.q_off + tk * p.q_row + h * p.q_seg + d] * p.scale : 0.f;
        skv[r * LD + d] = __float2half(v);
    }
    __syncthreads();
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> A[NDT];
    #pragma unroll
    for (int j = 0; j < NDT; j++) wmma::load_matrix_sync(A[j], skv + j * 16, LD);
    __syncthreads();
    /* the softmax's layout: row r = tid / 8, keys c = (tid % 8) * 8 .. + 7 */
    int row = tid >> 3, cl = (tid & 7) * 8;
    int qpos = p.pos_base + min(t0 + row / GQ, S - 1);
    float m = -3.0e38f, l = 0.f;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> Bk;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> Sc;
    for (int pass = 0; pass < 2; pass++) {
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> Oa[NDS > 0 ? NDS : 1];
        #pragma unroll
        for (int u = 0; u < NDS; u++) wmma::fill_fragment(Oa[u], 0.f);
        if (pass == 1) {
            if (snk) {   /* the head's sink: one more term of the denominator, with no value */
                float sv = snk[sink_off + g * GQ + row % GQ], mn = fmaxf(m, sv);
                l = l * __expf(m - mn) + __expf(sv - mn);
                m = mn;
            }
            if ((tid & 7) == 0) { sm[row] = m; sl[row] = l; }
            __syncthreads();
            m = sm[row];
        }
        for (int kb = kb_lo; kb < nkb; kb++) {
            int k0 = kb * CC_FA_BC;
            if (F4) cc_fa_stage4<HD>(skv, kc, p.k_off, g, p.cap, k0, nt, ring, tid);
            else for (int e = tid; e < CC_FA_BC * HD; e += 128) {   /* K of keys k0 .. k0 + 63, zero past nt */
                int kl = e / HD, d = e - kl * HD, key = k0 + kl;
                float v = key < nt ? kc[p.k_off + (g * p.cap + (ring ? key % ring : key)) * HD + d] : 0.f;
                skv[kl * LD + d] = __float2half(v);
            }
            __syncthreads();
            wmma::fill_fragment(Sc, 0.f);
            #pragma unroll
            for (int j = 0; j < NDT; j++) {   /* warp w: keys k0 + 16w .. + 15 */
                wmma::load_matrix_sync(Bk, skv + (warp * 16) * LD + j * 16, LD);
                wmma::mma_sync(Sc, A[j], Bk, Sc);
            }
            wmma::store_matrix_sync(ss + warp * 16, Sc, LS, wmma::mem_row_major);
            __syncthreads();
            if (pass == 0) {
                float sv[8], bm = -3.0e38f;
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    int key = k0 + cl + i;
                    sv[i] = key <= qpos && (win <= 0 || key > qpos - win) ? ss[row * LS + cl + i] : -3.0e38f;
                    bm = fmaxf(bm, sv[i]);
                }
                for (int off = 4; off > 0; off >>= 1) bm = fmaxf(bm, __shfl_xor_sync(0xffffffffu, bm, off, 8));
                float mn = fmaxf(m, bm), part = 0.f;
                #pragma unroll
                for (int i = 0; i < 8; i++) part += sv[i] > -1e38f ? __expf(sv[i] - mn) : 0.f;
                for (int off = 4; off > 0; off >>= 1) part += __shfl_xor_sync(0xffffffffu, part, off, 8);
                l = l * __expf(m - mn) + part;
                m = mn;
                __syncthreads();   /* skv and ss are rewritten next block */
                continue;
            }
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int key = k0 + cl + i;
                float sv = ss[row * LS + cl + i];
                sp[row * LP + cl + i] = __float2half(key <= qpos && (win <= 0 || key > qpos - win) ? __expf(sv - m) : 0.f);
            }
            __syncthreads();   /* P complete, K consumed */
            if (F4) cc_fa_stage4<HD>(skv, vc, p.v_off, g, p.cap, k0, nt, ring, tid);
            else for (int e = tid; e < CC_FA_BC * HD; e += 128) {   /* V of the same keys */
                int kl = e / HD, d = e - kl * HD, key = k0 + kl;
                float v = key < nt ? vc[p.v_off + (g * p.cap + (ring ? key % ring : key)) * HD + d] : 0.f;
                skv[kl * LD + d] = __float2half(v);
            }
            __syncthreads();
            #pragma unroll
            for (int kt = 0; kt < 4; kt++) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> Pa;
                wmma::load_matrix_sync(Pa, sp + kt * 16, LP);
                #pragma unroll
                for (int u = 0; u < NDS; u++) {   /* warp w: dims (w * NDS + u) * 16 .. + 15 */
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> Bv;
                    wmma::load_matrix_sync(Bv, skv + (kt * 16) * LD + (warp * NDS + u) * 16, LD);
                    wmma::mma_sync(Oa[u], Pa, Bv, Oa[u]);
                }
            }
            __syncthreads();
        }
        if (pass == 0) continue;
        /* out: rows map to (token, head); through skv as floats, divided by l, the gate */
        float *so = reinterpret_cast<float *>(skv);
        #pragma unroll
        for (int u = 0; u < NDS; u++) wmma::store_matrix_sync(so + (warp * NDS + u) * 16, Oa[u], HD + 4, wmma::mem_row_major);
        __syncthreads();
        for (int e = tid; e < 16 * HD; e += 128) {
            int r = e / HD, d = e - r * HD, tk = t0 + r / GQ, h = g * GQ + r % GQ;
            if (tk >= S) continue;
            float lv = sl[r], v = lv > 0.f ? so[r * (HD + 4) + d] / lv : 0.f;
            if (p.has_gate) v *= 1.f / (1.f + __expf(-gt[p.g_off + tk * p.g_row + h * p.g_seg + d]));
            o[p.o_off + tk * p.o_row + h * HD + d] = v;
        }
    }
#endif
}
/* one thread per channel, the rows in order, the channel's ring in registers */
__global__ static void cc_dnconv_k(const float *__restrict__ xin, const float *__restrict__ w, float *__restrict__ ring,
                                   float *__restrict__ o, float *__restrict__ snap, CcDnConv p) {
    long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= p.CD) return;
    int nh = p.CK - 1;
    float hist[8];
    for (int k = 0; k < 8; k++) hist[k] = k < nh ? ring[p.ring_off + c * nh + k] : 0.f;
    int wb = p.w_off + (int)c * p.CK;
    float wl = w[wb + p.CK - 1];
    for (int s = 0; s < p.S; s++) {
        float cur = xin[p.in_off + s * p.in_row + c];
        float acc;
        if (p.order == 0) {
            acc = 0.f;
            for (int k = 0; k < 8; k++) if (k < nh) acc += w[wb + k] * hist[k];
            acc += wl * cur;
        } else {
            acc = wl * cur;
            for (int k = 0; k < 8; k++) if (k < nh) acc += w[wb + k] * hist[k];
        }
        o[p.out_off + s * p.out_row + c] = acc / (1.f + expf(-acc));
        for (int k = 0; k < 7; k++) if (k + 1 < nh) hist[k] = hist[k + 1];
        for (int k = 0; k < 8; k++) if (k == nh - 1) hist[k] = cur;
        if (s == p.snap_row)
            for (int k = 0; k < 8; k++) if (k < nh) snap[p.snap_off + c * nh + k] = hist[k];
    }
    for (int k = 0; k < 8; k++) if (k < nh) ring[p.ring_off + c * nh + k] = hist[k];
}
__device__ static float cc_softplus(float x) {
    if (x > 20.f) return x;
    float e = expf(x);
    return e < 1e-4f ? e - 0.5f * e * e : logf(1.f + e);
}
/* one block of 128 per value head, thread v's column of the state in registers */
template <int KD>
__global__ static void __launch_bounds__(128) cc_dnrec_k(const float *__restrict__ cv, const float *__restrict__ ab,
        const float *__restrict__ z, float *__restrict__ st, const float *__restrict__ prm, float *__restrict__ y,
        float *__restrict__ snap, CcDnRec p) {
    __shared__ float qn[KD];
    __shared__ float kn[KD];
    __shared__ float red[128];
    __shared__ float red2[128];
    int h = blockIdx.x, v = threadIdx.x;
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
        __syncthreads();
        for (int off = 64; off > 0; off >>= 1) {
            if (v < off) { red[v] += red[v + off]; red2[v] += red2[v + off]; }
            __syncthreads();
        }
        float qsc = 1.f / sqrtf(red[0] + 1e-6f) * p.qscale;
        float ksc = 1.f / sqrtf(red2[0] + 1e-6f);
        __syncthreads();
        for (int k = v; k < KD; k += 128) { qn[k] *= qsc; kn[k] *= ksc; }
        __syncthreads();
        float bv = ab[p.b_off + s * p.b_row + h], av = ab[p.a_off + s * p.a_row + h];
        float beta = 1.f / (1.f + expf(-bv));
        float decay = expf(((p.flags & 2) ? alog : -expf(alog)) * cc_softplus(av + dtb));   /* flags 2: a GGUF's A itself */
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
        __syncthreads();
        for (int off = 64; off > 0; off >>= 1) {
            if (v < off) red[v] += red[v + off];
            __syncthreads();
        }
        float r = 1.f / sqrtf(red[0] / (float)p.VD + p.eps);
        if (own) {
            float zv = z[p.z_off + s * p.z_row + h * p.VD + v];
            float g = (p.flags & 1) ? 1.f / (1.f + expf(-zv)) : zv / (1.f + expf(-zv));
            y[p.y_off + s * p.y_row + h * p.VD + v] = o * r * nw * g;
            if (s == p.snap_row) {
                #pragma unroll
                for (int k = 0; k < KD; k++) snap[nb + k * p.VD + v] = col[k];
            }
        }
        __syncthreads();
    }
    if (own) {
        #pragma unroll
        for (int k = 0; k < KD; k++) st[sb + k * p.VD + v] = col[k];
    }
}
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP__)
#define cc_shfl_xor(v, o) __shfl_xor((v), (o), 32)   /* 32-lane groups on a wave64 device too */
#else
#define cc_shfl_xor(v, o) __shfl_xor_sync(0xffffffffu, (v), (o), 32)
#endif
#define CC_DNREC_ROWS 8   /* rows from which cc_dnrec takes cc_dnrec_rows_k (COLI_CUDA_DNREC_ROWS=0 never) */
/* A prompt chunk's rows (cc_dnrec from CC_DNREC_ROWS rows): cc_dnrec_k's arithmetic, R
 * rows at a time. cc_dnrec_k takes about eighteen barriers a row (the tree sums of the q, k
 * and output norms); here, per block: the q and k norms, beta, decay and v in parallel (a
 * warp a vector), the rows in order with no barrier (each thread's column in registers,
 * its dots over the block's shared q and k), then each row's gated RMSNorm (a warp a row):
 * three barriers a block. The norms' sums run in a warp's order, the state's as before. */
template <int KD, int R>
__global__ static void __launch_bounds__(128) cc_dnrec_rows_k(const float *__restrict__ cv, const float *__restrict__ ab,
        const float *__restrict__ z, float *__restrict__ st, const float *__restrict__ prm, float *__restrict__ y,
        float *__restrict__ snap, CcDnRec p) {
    __shared__ float qs[R][KD];
    __shared__ float ks[R][KD];
    __shared__ float vs[R][128];
    __shared__ float os[R][128];
    __shared__ float bs[R], ds[R];
    int h = blockIdx.x, v = threadIdx.x, wp = v >> 5, lane = v & 31;
    bool own = v < p.VD;
    int kh = (p.flags & 4) ? h % p.KH : h / (p.VH / p.KH);   /* flags 4: a GGUF's value heads, tiled */
    float col[KD];
    int sb = p.st_off + h * KD * p.VD, nb = p.snap_off + h * KD * p.VD;
    #pragma unroll
    for (int k = 0; k < KD; k++) col[k] = own ? st[sb + k * p.VD + v] : 0.f;
    float alog = prm[p.prm_off + h], dtb = prm[p.prm_off + p.VH + h];
    for (int s0 = 0; s0 < p.S; s0 += R) {
        int nr = min(R, p.S - s0);
        for (int w = wp; w < 2 * nr; w += 4) {
            int r = w >> 1, isk = w & 1, cb = p.cv_off + (s0 + r) * p.cv_row + isk * p.Ktot + kh * KD;
            float a = 0.f;
            for (int k = lane; k < KD; k += 32) { float x = cv[cb + k]; a += x * x; }
            #pragma unroll
            for (int m = 16; m > 0; m >>= 1) a += cc_shfl_xor(a, m);
            float sc = 1.f / sqrtf(a + 1e-6f) * (isk ? 1.f : p.qscale);
            for (int k = lane; k < KD; k += 32) { float x = cv[cb + k] * sc; if (isk) ks[r][k] = x; else qs[r][k] = x; }
        }
        for (int i = v; i < nr * p.VD; i += 128) {
            int r = i / p.VD, c = i - r * p.VD;
            vs[r][c] = cv[p.cv_off + (s0 + r) * p.cv_row + 2 * p.Ktot + h * p.VD + c];
        }
        if (v < nr) {
            int s = s0 + v;
            float bv = ab[p.b_off + s * p.b_row + h], av = ab[p.a_off + s * p.a_row + h];
            bs[v] = 1.f / (1.f + expf(-bv));
            ds[v] = expf(((p.flags & 2) ? alog : -expf(alog)) * cc_softplus(av + dtb));
        }
        __syncthreads();
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
        __syncthreads();
        for (int r = wp; r < nr; r += 4) {
            int s = s0 + r;
            float oo = 0.f;
            for (int c = lane; c < p.VD; c += 32) oo += os[r][c] * os[r][c];
            #pragma unroll
            for (int m = 16; m > 0; m >>= 1) oo += cc_shfl_xor(oo, m);
            float rr = 1.f / sqrtf(oo / (float)p.VD + p.eps);
            for (int c = lane; c < p.VD; c += 32) {
                float zv = z[p.z_off + s * p.z_row + h * p.VD + c];
                float g = (p.flags & 1) ? 1.f / (1.f + expf(-zv)) : zv / (1.f + expf(-zv));
                y[p.y_off + s * p.y_row + h * p.VD + c] = os[r][c] * rr * prm[p.prm_off + 2 * p.VH + c] * g;
            }
        }
        __syncthreads();
    }
    if (own) {
        #pragma unroll
        for (int k = 0; k < KD; k++) st[sb + k * p.VD + v] = col[k];
    }
}
__device__ static float cc_sig(float x) { return 1.f / (1.f + expf(-x)); }
/* chain_qsa.comp: mode 0 one block of 128 per block key; mode 1 one block per query row */
__global__ static void __launch_bounds__(128) cc_qsa_k(const float *__restrict__ src, const float *__restrict__ w,
        float *__restrict__ pk, const float *__restrict__ cs, float *__restrict__ sc, int *__restrict__ sel, CcQsa p) {
    __shared__ float pool[256];
    __shared__ float red[128];
    int tid = threadIdx.x;
    if (p.mode == 0) {
        int b = p.b0 + blockIdx.x;
        float ss = 0.f;
        for (int d = tid; d < p.ID; d += 128) {
            float v = 0.f;
            for (int r = 0; r < p.R; r++) v += src[p.src_off + (b * p.R + r) * p.ID + d] / (float)p.R;
            pool[d] = v;
            ss += v * v;
        }
        red[tid] = ss;
        __syncthreads();
        for (int off = 64; off > 0; off >>= 1) {
            if (tid < off) red[tid] += red[tid + off];
            __syncthreads();
        }
        float rr = 1.f / sqrtf(red[0] / (float)p.ID + p.eps);
        __syncthreads();
        for (int d = tid; d < p.ID; d += 128) pool[d] = pool[d] * rr * (1.f + w[p.w_off + d]);
        __syncthreads();
        int cb = blockIdx.x * 2 * p.half_;
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
    int s = blockIdx.x;
    int visible = p.pos_base + s + 1, blocks = visible / p.R;
    int take = min(blocks, p.budget / p.R);
    int lb = s * p.sel_row;
    if (take >= blocks) {
        if (tid == 0) sel[lb] = -1;
        return;
    }
    int sb = s * 2 * p.nbmax;
    int qb = p.q_off + s * p.q_row;
    for (int b = tid; b < blocks; b += 128) {
        float score = 0.f;
        for (int h = 0; h < p.IQ; h++) {
            float a = 0.f;
            for (int d = 0; d < p.ID; d++) a += src[qb + h * p.ID + d] * pk[p.pk_off + b * p.ID + d];
            if (a > 0.f) score += a;
        }
        sc[sb + b] = score / sqrtf((float)p.ID);
    }
    __threadfence_block();
    __syncthreads();
    for (int b = tid; b < blocks; b += 128) {
        float v = sc[sb + b];
        int rank = 0;
        for (int b2 = 0; b2 < blocks; b2++) {
            float v2 = sc[sb + b2];
            if (v2 > v || (v2 == v && b2 < b)) rank++;
        }
        sc[sb + p.nbmax + b] = rank < take ? 1.f : 0.f;
    }
    __threadfence_block();
    __syncthreads();
    if (tid == 0) {
        int n = 0;
        for (int b = 0; b < blocks; b++)
            if (sc[sb + p.nbmax + b] != 0.f)
                for (int r = 0; r < p.R; r++) sel[lb + 1 + n++] = b * p.R + r;
        for (int t = blocks * p.R; t < visible; t++) sel[lb + 1 + n++] = t;
        sel[lb] = n;
    }
}
/* chain_ple.comp: mode 0 one block of 256 per (row, stream); mode 1 one thread per channel */
__device__ static void cc_ple_reduce2(float *r1, float *r2, int tid) {
    __syncthreads();
    for (int off = 128; off > 0; off >>= 1) {
        if (tid < off) { r1[tid] += r1[tid + off]; r2[tid] += r2[tid + off]; }
        __syncthreads();
    }
}
__global__ static void __launch_bounds__(256) cc_ple_k(const float *__restrict__ keys, float *__restrict__ hyp,
        const float *__restrict__ val, const float *__restrict__ prm, float *__restrict__ gated, float *__restrict__ normv,
        const float *__restrict__ conv, float *__restrict__ ring, CcPle p) {
    __shared__ float r1[256];
    __shared__ float r2[256];
    int W = p.C * p.H, tid = threadIdx.x;
    if (p.mode == 0) {
        int g = blockIdx.x, s = g / p.C, k = g - s * p.C;
        int kb = p.keys_off + s * W + k * p.H, hb = p.hyp_off + s * W + k * p.H, vb = p.val_off + s * p.H;
        int wk = p.prm_off + k * p.H, wq = p.prm_off + W + k * p.H, wc = p.prm_off + 2 * W + k * p.H;
        float a = 0.f, b = 0.f;
        for (int d = tid; d < p.H; d += 256) { float u = keys[kb + d], v = hyp[hb + d]; a += u * u; b += v * v; }
        r1[tid] = a; r2[tid] = b;
        cc_ple_reduce2(r1, r2, tid);
        float rk = 1.f / sqrtf(r1[0] / (float)p.H + p.eps), rq = 1.f / sqrtf(r2[0] / (float)p.H + p.eps);
        __syncthreads();
        float dt = 0.f;
        for (int d = tid; d < p.H; d += 256)
            dt += (keys[kb + d] * rk * (1.f + prm[wk + d])) * (hyp[hb + d] * rq * (1.f + prm[wq + d]));
        r1[tid] = dt; r2[tid] = 0.f;
        cc_ple_reduce2(r1, r2, tid);
        float dot = r1[0] / sqrtf((float)p.H);
        float shaped = (dot > 0.f ? 1.f : dot < 0.f ? -1.f : 0.f) * sqrtf(fmaxf(fabsf(dot), 1e-6f));
        if (dot == 0.f) shaped = sqrtf(1e-6f);   /* copysign(x, +0) is +x */
        float gt = cc_sig(shaped);
        __syncthreads();
        float c3 = 0.f;
        for (int d = tid; d < p.H; d += 256) {
            float u = gt * val[vb + d];
            gated[s * W + k * p.H + d] = u;
            c3 += u * u;
        }
        r1[tid] = c3; r2[tid] = 0.f;
        cc_ple_reduce2(r1, r2, tid);
        float rc = 1.f / sqrtf(r1[0] / (float)p.H + p.eps);
        for (int d = tid; d < p.H; d += 256)
            normv[s * W + k * p.H + d] = gated[s * W + k * p.H + d] * rc * (1.f + prm[wc + d]);
        return;
    }
    long long d = (long long)blockIdx.x * blockDim.x + threadIdx.x;
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
__global__ static void cc_ew_k(float *__restrict__ y, const float *__restrict__ a, const float *__restrict__ b,
                               const float *__restrict__ c, const float *__restrict__ e, CcEw p) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
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
        y[p.y_off + i] = (g / (1.f + expf(-g))) * b[p.b_off + i];
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
    }
}

/* chain_route.comp: one block per row, its 8 warps the shader's subgroups (the row's max
 * and sum: each warp's, then theirs in warp order) */
__device__ __forceinline__ float cc_wmax(float v) { for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, cc_shfl_xor(v, o)); return v; }
__device__ __forceinline__ float cc_wsum(float v) { for (int o = 16; o > 0; o >>= 1) v += cc_shfl_xor(v, o); return v; }
__device__ __forceinline__ int cc_wmin(int v) { for (int o = 16; o > 0; o >>= 1) v = min(v, cc_shfl_xor(v, o)); return v; }
__global__ static void __launch_bounds__(256) cc_route_k(const float *__restrict__ l, const float *__restrict__ bias,
                                                         int *__restrict__ ids, float *__restrict__ w, CcRoute p) {
    const float NEG = -3.402823466e38f;
    __shared__ float sc[1024], pk[1024], rv[256], gv[64], wv[32];
    __shared__ int keepg[64], chosen[32], keepi[256];
    int s = blockIdx.x;
    if (s >= p.S) return;
    int tid = threadIdx.x, lane = tid & 31, sg = tid >> 5, nsg = 8;
    int lb = p.l_off + s * p.l_row;
    auto logit = [&](int e) { float v = l[lb + e]; if (p.has_bias == 2) v += bias[p.b_off + e]; return v / p.temp; };
    if (p.kind == 0) {
        float m = NEG;
        for (int e = tid; e < p.E; e += 256) m = fmaxf(m, logit(e));
        m = cc_wmax(m);
        if (lane == 0) rv[sg] = m;
        __syncthreads();
        float mx = NEG;
        for (int g = 0; g < nsg; g++) mx = fmaxf(mx, rv[g]);
        __syncthreads();
        float acc = 0.f;
        for (int e = tid; e < p.E; e += 256) { float v = expf(logit(e) - mx); sc[e] = v; acc += v; }
        acc = cc_wsum(acc);
        if (lane == 0) rv[sg] = acc;
        __syncthreads();
        float sum = 0.f;
        for (int g = 0; g < nsg; g++) sum += rv[g];
        __syncthreads();
        for (int e = tid; e < p.E; e += 256) sc[e] = sc[e] / sum;
    } else {
        for (int e = tid; e < p.E; e += 256) {
            float v = logit(e);
            sc[e] = p.kind == 1 ? 1.f / (1.f + expf(-v)) : sqrtf(v > 20.f ? v : logf(1.f + expf(v)));
        }
    }
    __syncthreads();
    for (int e = tid; e < p.E; e += 256) {
        float v = sc[e];
        if (p.has_bias == 1) v += bias[p.b_off + e];
        pk[e] = isnan(v) ? NEG : v;
    }
    __syncthreads();
    if (p.n_group > 1) {
        int per = p.E / p.n_group;
        if (tid < p.n_group) {
            float b1 = NEG, b2 = NEG;
            for (int e = tid * per; e < tid * per + per; e++) { float v = pk[e]; if (v > b1) { b2 = b1; b1 = v; } else if (v > b2) b2 = v; }
            gv[tid] = b1 + b2; keepg[tid] = 0;
        }
        __syncthreads();
        if (tid == 0)
            for (int k = 0; k < p.topk_group; k++) {
                int bg = -1; float bv = NEG;
                for (int g = 0; g < p.n_group; g++) if (keepg[g] == 0 && gv[g] > bv) { bv = gv[g]; bg = g; }
                if (bg < 0) break;
                keepg[bg] = 1;
            }
        __syncthreads();
        for (int e = tid; e < p.E; e += 256) if (keepg[min(max(e / per, 0), p.n_group - 1)] == 0) pk[e] = NEG;
        __syncthreads();
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
        float m = cc_wmax(bv);
        int me = cc_wmin(bv == m && m > NEG ? be : 0x7fffffff);
        int h = (k & 1) * 128;
        if (lane == 0) { rv[h + sg] = m; keepi[h + sg] = me; }
        __syncthreads();
        float b = NEG; int c = -1;
        for (int g = 0; g < nsg; g++) {
            float u = rv[h + g]; int ie = keepi[h + g];
            if (u > NEG && ie != 0x7fffffff && (u > b || (u == b && ie < c))) { b = u; c = ie; }
        }
        if (tid == 0) chosen[k] = c;
        #pragma unroll
        for (int j = 0; j < 4; j++) if (c == tid + 256 * j) mine[j] = NEG;
    }
    __syncthreads();
    int b = tid < 32 ? chosen[tid] : -1;
    if (tid < 32) wv[tid] = tid < p.K && b >= 0 ? sc[b] : 0.f;
    __syncthreads();
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
/* the hybrid steps' flags: the device's word in host memory once the stream gets here
 * (a system-wide fence first: the step's rows the host reads are there before it), and
 * the stream held until the host's word reaches v */
__global__ static void cc_signal_k(volatile unsigned *f, unsigned v) { __threadfence_system(); *f = v; __threadfence_system(); }
__global__ static void cc_wait_k(volatile unsigned *f, unsigned v) { while (*f < v) { } __threadfence_system(); }
/* one block of 256, two passes: the argmax of n logits (lowest index on a tie) into word 0
 * of out and its softmax probability (terms within 20 of the maximum) into word 1 */
__global__ static void __launch_bounds__(256) cc_pick_k(const float *__restrict__ lg, void *__restrict__ out, int n) {
    __shared__ float rv[8];
    __shared__ int ri[8];
    int tid = threadIdx.x, lane = tid & 31, sg = tid >> 5;
    int be = tid < n ? tid : 0x7fffffff;
    float bv = tid < n ? lg[tid] : -INFINITY;
    for (int i = tid + 256; i < n; i += 256) {
        float v = lg[i];
        if (v > bv) { bv = v; be = i; }
    }
    float m = cc_wmax(bv);
    int me = cc_wmin(bv == m ? be : 0x7fffffff);
    if (lane == 0) { rv[sg] = m; ri[sg] = me; }
    __syncthreads();
    float mx = rv[0]; int best = ri[0];
    for (int g = 1; g < 8; g++)
        if (rv[g] > mx || (rv[g] == mx && ri[g] < best)) { mx = rv[g]; best = ri[g]; }
    float z = 0.f;
    for (int i = tid; i < n; i += 256) {
        float d = lg[i] - mx;
        if (d > -20.f) z += expf(d);
    }
    z = cc_wsum(z);
    __syncthreads();
    if (lane == 0) rv[sg] = z;
    __syncthreads();
    if (tid == 0) {
        float tot = 0.f;
        for (int g = 0; g < 8; g++) tot += rv[g];
        ((int *)out)[0] = best == 0x7fffffff ? 0 : best;
        ((float *)out)[1] = tot > 0.f ? 1.f / tot : 0.f;
    }
}
/* backend_cuda.cu's gx4_idx: every 8 float4s padded by 1 so the 8 warps' 4-way shared loads
 * land in distinct banks. cc_matmul_rows_i8_k has gemv_i8_rows<1, 1>'s layout for a row list
 * ids[0..n), but sums 16 products before adding them: its bits can differ (drafts only). */
__device__ static inline int cc_gx4(int k) { return k + (k >> 3); }
template <int PAD>
__global__ static void __launch_bounds__(256) cc_matmul_rows_i8_k(float *__restrict__ y, const float *__restrict__ x,
        const int8_t *__restrict__ w, const float *__restrict__ sc, const int *__restrict__ ids, int I, int O, int n) {
    extern __shared__ float4 sx[];
    int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, I4 = I >> 2, I16 = I >> 4;
    const float4 *x4 = reinterpret_cast<const float4 *>(x);
    for (int i = tid; i < I4; i += 256) sx[PAD ? cc_gx4(i) : i] = x4[i];
    __syncthreads();
    int j = (int)blockIdx.x * 8 + warp;
    if (j >= n) return;
    int o = ids[j];
    if (o < 0 || o >= O) { if (lane == 0) y[j] = -INFINITY; return; }
    const int4 *wr = reinterpret_cast<const int4 *>(w + (size_t)o * I);
    float acc = 0.f;
    for (int i = lane; i < I16; i += 32) {
        int4 v = wr[i];
        const int8_t *q = reinterpret_cast<const int8_t *>(&v);
        int k = i * 4;
        float4 x0 = sx[PAD ? cc_gx4(k) : k], x1 = sx[PAD ? cc_gx4(k + 1) : k + 1];
        float4 x2 = sx[PAD ? cc_gx4(k + 2) : k + 2], x3 = sx[PAD ? cc_gx4(k + 3) : k + 3];
        acc += (float)q[0] * x0.x + (float)q[1] * x0.y + (float)q[2] * x0.z + (float)q[3] * x0.w
             + (float)q[4] * x1.x + (float)q[5] * x1.y + (float)q[6] * x1.z + (float)q[7] * x1.w
             + (float)q[8] * x2.x + (float)q[9] * x2.y + (float)q[10] * x2.z + (float)q[11] * x2.w
             + (float)q[12] * x3.x + (float)q[13] * x3.y + (float)q[14] * x3.z + (float)q[15] * x3.w;
    }
    acc = cc_wsum(acc);
    if (lane == 0) y[j] = acc * sc[o];
}
__global__ static void __launch_bounds__(256) cc_matmul_rows_k(float *__restrict__ y, const float *__restrict__ x,
        const void *__restrict__ w, const float *__restrict__ sc, const int *__restrict__ ids, int fmt, int I, int O, int n) {
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, j = (int)blockIdx.x * 8 + warp;
    if (j >= n) return;
    int o = ids[j];
    if (o < 0 || o >= O) { if (lane == 0) y[j] = -INFINITY; return; }
    float acc = 0.f;
    if (fmt == 1) {
        const int8_t *wr = (const int8_t *)w + (size_t)o * I;
        for (int i = lane; i < I; i += 32) acc += x[i] * (float)wr[i];
        acc = cc_wsum(acc);
        if (lane == 0) y[j] = acc * sc[o];
    } else {
        const float *wr = (const float *)w + (size_t)o * I;
        for (int i = lane; i < I; i += 32) acc += x[i] * wr[i];
        acc = cc_wsum(acc);
        if (lane == 0) y[j] = acc;
    }
}

/* ---- contexts ------------------------------------------------------------------- */
static int cc_init(int device) {
    CcCtx *c = ctx_of(device);
    if (c && c->up) { g_cur = (int)(c - g_cc); return !c->lost; }
    if (!c) {
        if (g_ncc >= COLI_CUDA_MAX_DEVICES) return 0;
        c = &g_cc[g_ncc];
        std::memset(c, 0, sizeof *c);
        c->device = device;
    }
    if (cudaSetDevice(device) != cudaSuccess) { (void)cudaGetLastError(); return 0; }
    if (!cc_ok(nullptr, cudaStreamCreateWithFlags(&c->stream, cudaStreamNonBlocking), "stream")) return 0;
    if (!cc_ok(nullptr, cudaEventCreateWithFlags(&c->ev, cudaEventDisableTiming), "event")) { cudaStreamDestroy(c->stream); return 0; }
    c->ev_ok = 1; c->up = 1; c->lost = 0; c->in_frame = 0; c->ev_pending = 0;
#if !defined(__HIP__) && !defined(__HIP_PLATFORM_AMD__)   /* the flash attention's tensor cores (CUDA only) */
    if (cudaDeviceGetAttribute(&c->major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess) { c->major = 0; (void)cudaGetLastError(); }
#else
    c->major = 0;
#endif
    if (c == &g_cc[g_ncc]) g_ncc++;
    g_cur = (int)(c - g_cc);
    return 1;
}
static int cc_ready(void) { CcCtx *c = cur(); return c && c->up && !c->lost; }
static int cc_lost_(void) { CcCtx *c = cur(); return !c || !c->up || c->lost; }
static void cc_shutdown(void) {
    for (int i = 0; i < g_ncc; i++) {
        CcCtx *c = &g_cc[i];
        if (!c->up) continue;
        if (cudaSetDevice(c->device) == cudaSuccess) {
            if (!c->lost) cudaStreamSynchronize(c->stream);
            if (c->ev_ok) cudaEventDestroy(c->ev);
            cudaStreamDestroy(c->stream);
            if (c->cs_ok) {
                if (!c->lost) cudaStreamSynchronize(c->cs);
                cudaEventDestroy(c->ev_fetched); cudaEventDestroy(c->ev_free); cudaStreamDestroy(c->cs);
            }
            if (c->up_made) { for (int i = 0; i < CC_UP_MARKS; i++) cudaEventDestroy(c->up_ev[i]); cudaEventDestroy(c->up_after); }
            c->up_made = 0; c->up_seq = 0;
            if (c->adec) cudaFree(c->adec);
            cc_adec_reap(c);
        }
        c->adec = nullptr; c->adec_bytes = 0;
        (void)cudaGetLastError();
        c->up = 0; c->ev_ok = 0; c->cs_ok = 0;
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
        /* zeroed on the chain's stream: a cudaMemset runs on the legacy default stream, which
         * a non-blocking stream does not wait for, and landed after a first write now and then */
        if (cudaMalloc((void **)&b->d, bytes) != cudaSuccess || cudaMemsetAsync(b->d, 0, bytes, c->stream) != cudaSuccess) {
            (void)cudaGetLastError();
            if (b->d) cudaFree(b->d);
            std::free(b);
            return nullptr;
        }
        g_st.dev_bytes += bytes;
    } else {
        unsigned flags = cudaHostAllocMapped | (kind == CC_UP ? cudaHostAllocWriteCombined : 0u);
        void *dp = nullptr;
        if (cudaHostAlloc(&b->h, bytes, flags) != cudaSuccess || cudaHostGetDevicePointer(&dp, b->h, 0) != cudaSuccess) {
            (void)cudaGetLastError();
            if (b->h) cudaFreeHost(b->h);
            std::free(b);
            return nullptr;
        }
        b->d = (float *)dp;
        std::memset(b->h, 0, bytes);
    }
    return b;
}
static void cc_free(CcBuf *b) {
    if (!b) return;
    CcCtx *c = ctx_of(b->dev);
    if (c && c->up && cudaSetDevice(c->device) == cudaSuccess) {
        if (!c->lost) cudaStreamSynchronize(c->stream);   /* the frames that may read it */
        if (!c->lost && c->cs_ok) cudaStreamSynchronize(c->cs);   /* and the copies that may write it */
        if (b->kind == CC_DEV) { cudaFree(b->d); g_st.dev_bytes -= b->bytes; }
        else cudaFreeHost(b->h);
    }
    (void)cudaGetLastError();
    std::free(b);
}
static int cc_reserve(CcBuf **b, size_t bytes, int kind) {
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
    return 1;
}
static int cc_submit(int wait) {
    CcCtx *c = live();
    if (!c) return 0;
    c->in_frame = 0;
    if (!cc_ok(c, cudaEventRecord(c->ev, c->stream), "frame event")) return 0;
    c->ev_pending = 1;
    if (!wait) return 1;
    double t0 = cc_now_ms();
    int ok = cc_ok(c, cudaEventSynchronize(c->ev), "frame wait");
    g_st.waits++; g_st.wait_ms += cc_now_ms() - t0;
    c->ev_pending = 0;
    if (ok) cc_adec_reap(c);
    return ok;
}
static int cc_finish(void) {
    CcCtx *c = live();
    if (!c) return 0;
    double t0 = cc_now_ms();
    int ok = cc_ok(c, cudaStreamSynchronize(c->stream), "finish");
    if (ok && c->cs_ok) ok = cc_ok(c, cudaStreamSynchronize(c->cs), "finish copies");
    g_st.waits++; g_st.wait_ms += cc_now_ms() - t0;
    c->ev_pending = 0;
    if (ok) cc_adec_reap(c);
    return ok;
}

/* ---- transfers ------------------------------------------------------------------ */
static int cc_copy(CcBuf *dst, size_t doff, CcBuf *src, size_t soff, size_t n) {
    CcCtx *c = live();
    if (!c || !dst || !src) return 0;
    if (!n) return 1;
    if ((doff + n) * 4 > dst->bytes || (soff + n) * 4 > src->bytes) return 0;
    /* mapped host memory is addressed by its device pointer on both sides */
    g_st.ops++;
    if (dst->kind != CC_DEV) g_st.bytes_down += n * 4;
    if (src->kind != CC_DEV) g_st.bytes_up += n * 4;
    return cc_ok(c, cudaMemcpyAsync(dst->d + doff, src->d + soff, n * 4, cudaMemcpyDefault, c->stream), "copy");
}
static int cc_zero(CcBuf *dst, size_t off, size_t n) {
    CcCtx *c = live();
    if (!c || !dst) return 0;
    if (!n) return 1;
    if ((off + n) * 4 > dst->bytes) return 0;
    g_st.ops++;
    return cc_ok(c, cudaMemsetAsync(dst->d + off, 0, n * 4, c->stream), "zero");
}
/* Regions on a grid (vk_chain.c's cp2_grid): region i = (q, h) = (i / per, i % per) copies len
 * floats from src[s0 + q*sq + h*sh ..] to dst[d0 + q*dq + h*dh ..]. A step's K and V rows into
 * the head-major cache are such a grid, a region per row and head: unlike vkCmdCopyBuffer, the
 * fallback below issues a cudaMemcpyAsync per region (41366 calls and 521 ms of host time on a
 * 1024-token prompt, 40 calls a decode token on Qwen3.6-35B), so any grid of two or more regions
 * takes one kernel launch. */
struct CcGrid { long long d0, s0, dh, sh, dq, sq; int len, per, n; };
__global__ static void cc_copy_grid_k(float *__restrict__ d, const float *__restrict__ s, CcGrid g) {
    long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
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
static int cc_copy_regions(CcBuf *dst, CcBuf *src, const CcRegion *r, int n) {
    CcCtx *c = live();
    if (!c || !dst || !src || (n > 0 && !r)) return 0;
    for (int i = 0; i < n; i++)
        if ((r[i].dst + r[i].n) * 4 > dst->bytes || (r[i].src + r[i].n) * 4 > src->bytes) return 0;
    CcGrid g;
    if (dst != src && dst->d && src->d && cc_grid_of(r, n, &g)) {   /* one kernel for the grid */
        long long tot = (long long)g.n * g.len;
        cc_copy_grid_k<<<grid1((size_t)tot, 256), 256, 0, c->stream>>>(dst->d, src->d, g);
        return launched(c, "copy regions (grid)");
    }
    for (int i = 0; i < n; i++) {
        if (!r[i].n) continue;
        if (!cc_ok(c, cudaMemcpyAsync(dst->d + r[i].dst, src->d + r[i].src, r[i].n * 4, cudaMemcpyDefault, c->stream), "copy regions"))
            return 0;
    }
    g_st.ops++;
    return 1;
}
/* A DEV buffer takes the bytes through the stream (a pageable source is staged before
 * the call returns, so the caller may reuse it). A host-mapped buffer is written in
 * place: the caller orders that against the frames that read it, as with cc_ptr. */
static int cc_write(CcBuf *dst, size_t off, const void *src, size_t bytes) {
    CcCtx *c = live();
    if (!c || !dst || !src) return 0;
    if (!bytes) return 1;
    if (off * 4 + bytes > dst->bytes) return 0;
    g_st.bytes_up += bytes; g_st.ops++;
    if (dst->kind != CC_DEV) { std::memcpy((char *)dst->h + off * 4, src, bytes); return 1; }
    return cc_ok(c, cudaMemcpyAsync(dst->d + off, src, bytes, cudaMemcpyHostToDevice, c->stream), "write");
}
static int cc_read(CcBuf *src, size_t off, void *dst, size_t bytes) {
    CcCtx *c = live();
    if (!c || !src || !dst) return 0;
    if (!bytes) return 1;
    if (off * 4 + bytes > src->bytes) return 0;
    if (!cc_finish()) return 0;
    g_st.bytes_down += bytes;
    if (src->kind != CC_DEV) { std::memcpy(dst, (const char *)src->h + off * 4, bytes); return 1; }
    return cc_ok(c, cudaMemcpy(dst, src->d + off, bytes, cudaMemcpyDeviceToHost), "read");
}

/* ---- ops --------------------------------------------------------------------------- */
static int cc_matmul(ColiCudaTensor *t, CcBuf *x, size_t xo, CcBuf *y, size_t yo, int S) {
    CcCtx *c = live();
    if (!c || !t || !x || !y || S < 1) return 0;
    int fmt, I, O;
    if (!coli_cuda_tensor_shape(t, &fmt, &I, &O) || coli_cuda_tensor_device(t) != c->device) return 0;
    if ((xo + (size_t)S * I) * 4 > x->bytes || (yo + (size_t)S * O) * 4 > y->bytes) return 0;
    g_st.matmuls++; g_st.ops++;
    if (!coli_cuda_tensor_gemm_async(t, y->d + yo, x->d + xo, S, (void *)c->stream)) { c->lost = 1; return 0; }
    return 1;
}
static int cc_norm(CcBuf *x, CcBuf *w, CcBuf *y, const CcNorm *p) {
    CcCtx *c = live();
    if (!c || !x || !y || !p || p->nseg < 1 || p->D < 1 || p->per_row < 1) return 0;
    if (!(p->flags & CC_NORM_NOW) && !w) return 0;
    cc_norm_k<<<grid1((size_t)p->nseg, 1), 256, 0, c->stream>>>(fp(x), fp(w), fpw(y), *p);
    return launched(c, "norm");
}
static int cc_rope(CcBuf *x, CcBuf *cs, const CcRope *p) {
    CcCtx *c = live();
    if (!c || !x || !cs || !p || p->nseg < 1 || p->half_ < 1 || p->per_row < 1) return 0;
    cc_rope_k<<<grid1((size_t)p->nseg * p->half_, 64), 64, 0, c->stream>>>(fpw(x), fp(cs), *p);
    return launched(c, "rope");
}
static int cc_rope_f16(CcBuf *x, CcBuf *v, CcBuf *cs, const CcRope *p, int seg) {
    CcCtx *c = live();
    if (!c || !x || !v || !cs || !p || p->nseg < 1 || p->half_ < 1 || p->per_row < 1 || seg < 2 * p->half_) return 0;
    cc_rope_f16_k<<<grid1((size_t)p->nseg * seg, 128), 128, 0, c->stream>>>(fpw(x), fpw(v), fp(cs), *p, seg);
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
__host__ __device__ static inline int cc_attn_dec_len(int n, int H) {
    int L = 64, nb = (n + L - 1) / L;
    while (L < 256 && H * nb > 1024) { L *= 2; nb = (n + L - 1) / L; }
    return L;
}
#define CC_ATTN_DEC_ROWS 8   /* a decode step's or a verify's rows (every engine's verify) */
__global__ static void __launch_bounds__(256) cc_attn_dec_k(const float *__restrict__ q, const float *__restrict__ kc,
        const float *__restrict__ vc, float *__restrict__ o, const float *__restrict__ gt, float *__restrict__ xs,
        CcAttn p, int mode, int nb, int win = 0, const float *__restrict__ snk = nullptr, int sink_off = 0, int ring = 0) {
    __shared__ float qs[256];
    __shared__ float sc[256];
    __shared__ float red[8];
    int tid = threadIdx.x, wp = tid >> 5, lane = tid & 31, h = blockIdx.x, hd = p.hd, stride = hd + 2;
    if (mode == 0) {
        int b = blockIdx.y, s = blockIdx.z, kvh = h / (p.H / p.KVH);
        int n = p.pos_base + s + 1, first = win > 0 && n > win ? n - win : 0;   /* a window: its last win positions */
        int L = cc_attn_dec_len(n - first, p.H), t0 = first + b * L, cnt = min(L, n - t0);
        size_t xo = ((size_t)(s * p.H + h) * nb + b) * stride;
        if (cnt <= 0) {   /* past the row's positions: an empty block */
            if (tid == 0) { xs[xo] = -3.0e38f; xs[xo + 1] = 0.f; }
            for (int d = tid; d < hd; d += 256) xs[xo + 2 + d] = 0.f;
            return;
        }
        int qb = p.q_off + s * p.q_row + h * p.q_seg;
        for (int d = tid; d < hd; d += 256) qs[d] = q[qb + d];
        __syncthreads();
        int kbase = kvh * p.cap;
        for (int j = wp; j < cnt; j += 8) {
            const float *kr = kc + p.k_off + (size_t)(kbase + (ring ? (t0 + j) % ring : t0 + j)) * hd;
            float a = 0.f;
            for (int d = lane; d < hd; d += 32) a += qs[d] * kr[d];
            for (int m = 16; m > 0; m >>= 1) a += cc_shfl_xor(a, m);
            if (lane == 0) sc[j] = a * p.scale;
        }
        __syncthreads();
        float mx = -3.0e38f;
        for (int j = tid; j < cnt; j += 256) mx = fmaxf(mx, sc[j]);
        for (int m = 16; m > 0; m >>= 1) mx = fmaxf(mx, cc_shfl_xor(mx, m));
        if (lane == 0) red[wp] = mx;
        __syncthreads();
        float M = red[0];
        for (int g = 1; g < 8; g++) M = fmaxf(M, red[g]);
        __syncthreads();
        float e = 0.f;
        for (int j = tid; j < cnt; j += 256) { float w = expf(sc[j] - M); sc[j] = w; e += w; }
        for (int m = 16; m > 0; m >>= 1) e += cc_shfl_xor(e, m);
        if (lane == 0) red[wp] = e;
        __syncthreads();
        float l = 0.f;
        for (int g = 0; g < 8; g++) l += red[g];
        for (int d = tid; d < hd; d += 256) {
            float acc = 0.f;
            for (int j = 0; j < cnt; j++) acc += sc[j] * vc[p.v_off + (size_t)(kbase + (ring ? (t0 + j) % ring : t0 + j)) * hd + d];
            xs[xo + 2 + d] = acc;
        }
        if (tid == 0) { xs[xo] = M; xs[xo + 1] = l; }
        return;
    }
    int s = blockIdx.y, n = p.pos_base + s + 1, first = win > 0 && n > win ? n - win : 0;
    int L = cc_attn_dec_len(n - first, p.H), nbr = (n - first + L - 1) / L;
    size_t xo = (size_t)(s * p.H + h) * nb * stride;
    float M = snk ? snk[sink_off + h] : -3.0e38f;   /* the sink joins the max and the sum, with no value */
    for (int b = 0; b < nbr; b++) M = fmaxf(M, xs[xo + (size_t)b * stride]);
    float Lt = snk ? expf(snk[sink_off + h] - M) : 0.f;
    for (int b = 0; b < nbr; b++) Lt += xs[xo + (size_t)b * stride + 1] * expf(xs[xo + (size_t)b * stride] - M);
    float inv = Lt > 0.f ? 1.f / Lt : 0.f;
    int ob = p.o_off + s * p.o_row + h * hd, gb = p.g_off + s * p.g_row + h * p.g_seg;
    for (int d = tid; d < hd; d += 256) {
        float v = 0.f;
        for (int b = 0; b < nbr; b++) v += xs[xo + (size_t)b * stride + 2 + d] * expf(xs[xo + (size_t)b * stride] - M);
        v *= inv;
        if (p.has_gate) v *= 1.f / (1.f + expf(-gt[gb + d]));
        o[ob + d] = v;
    }
}
static int cc_attn(CcBuf *q, CcBuf *kc, CcBuf *vc, CcBuf *o, CcBuf *gate, CcBuf *sel, const CcAttn *p) {
    CcCtx *c = live();
    if (!c || !q || !kc || !vc || !o || !p || p->S < 1 || p->H < 1 || p->KVH < 1 || p->H % p->KVH || p->hd < 1 || p->hd > 256) return 0;
    if (p->has_gate && !gate) return 0;
    if (p->sel_row > 0 && !sel) return 0;
    /* a prompt chunk's plain causal attention on the tensor cores (cc_attn_fk), from
     * COLI_CUDA_CHAIN_FLASH rows (16; 0 = never): sm_70 and later, head dim 64, 128 or 256,
     * the query heads of a kv head dividing 16, no selection list */
    static int fl = -1;
    if (fl < 0) { const char *e = getenv("COLI_CUDA_CHAIN_FLASH"); fl = e && *e ? atoi(e) : 16; if (fl < 0) fl = 0; }
    int G = p->H / p->KVH;
#if !defined(__HIP__) && !defined(__HIP_PLATFORM_AMD__)
    if (fl > 0 && p->S >= fl && c->major >= 7 && p->sel_row <= 0 && 16 % G == 0 && (p->hd == 64 || p->hd == 128 || p->hd == 256)) {
        dim3 fg((unsigned)((p->S + 16 / G - 1) / (16 / G)), (unsigned)p->KVH);
        if (p->hd == 64) cc_attn_fk<64><<<fg, 128, 0, c->stream>>>(fp(q), fp(kc), fp(vc), fpw(o), fp(gate), *p, G);
        else if (p->hd == 128) cc_attn_fk<128><<<fg, 128, 0, c->stream>>>(fp(q), fp(kc), fp(vc), fpw(o), fp(gate), *p, G);
        else if (((p->k_off | p->v_off) & 3) == 0 && (((size_t)fp(kc) | (size_t)fp(vc)) & 15) == 0)   /* float4 loads */
            cc_attn_fk<256, true><<<fg, 128, 0, c->stream>>>(fp(q), fp(kc), fp(vc), fpw(o), fp(gate), *p, G);
        else cc_attn_fk<256><<<fg, 128, 0, c->stream>>>(fp(q), fp(kc), fp(vc), fpw(o), fp(gate), *p, G);
        return launched(c, "attn (tensor cores)");
    }
#endif
    (void)fl; (void)G;
    /* a decode's or a verify's rows over many positions: split over them */
    static int dec = -1;
    if (dec < 0) { const char *e = getenv("COLI_CUDA_ATTN_DEC"); dec = !(e && *e == '0'); }
    int last = p->pos_base + p->S;
    /* only where a prompt chunk takes cc_attn_fk (f16 on the tensor cores): elsewhere a row's
     * bits do not depend on how a forward is cut into steps, and a prefix resumed answers as
     * the prompt recomputed does */
    int fk = 0;
#if !defined(__HIP__) && !defined(__HIP_PLATFORM_AMD__)
    fk = fl > 0 && c->major >= 7 && 16 % G == 0 && (p->hd == 64 || p->hd == 128 || p->hd == 256);
#endif
    if (dec && fk && p->S <= CC_ATTN_DEC_ROWS && p->sel_row <= 0 && last >= CC_ATTN_DEC_MIN) {
        /* each row as a decode step of it alone: its own blocks of positions (cc_attn_dec_len),
         * and the rows whose positions are fewer than CC_ATTN_DEC_MIN through cc_attn_k */
        int k = CC_ATTN_DEC_MIN - (p->pos_base + 1);
        if (k > 0) {
            CcAttn a = *p; a.S = k;
            cc_attn_k<<<dim3((unsigned)p->H, (unsigned)k), 128, 0, c->stream>>>(fp(q), fp(kc), fp(vc), fpw(o), fp(gate), nullptr, a);
            if (!launched(c, "attn")) return 0;
            CcAttn b = *p; b.S -= k; b.pos_base += k; b.q_off += k * p->q_row; b.o_off += k * p->o_row; b.g_off += k * p->g_row;
            return cc_attn(q, kc, vc, o, gate, sel, &b);
        }
        int nb = 1;
        for (int r = 0; r < p->S; r++) {
            int n = p->pos_base + r + 1, L = cc_attn_dec_len(n, p->H), m = (n + L - 1) / L;
            if (m > nb) nb = m;
        }
        size_t need = (size_t)p->S * p->H * nb * (p->hd + 2) * sizeof(float);
        if (need > c->adec_bytes) {   /* rare: the context past what it was; the frames before done first */
            if (c->adec) {   /* retired, not freed: the frames before may still read it, and a sync here can deadlock a hybrid layer */
                if (c->adec_n_old == 4) { if (!cc_ok(c, cudaStreamSynchronize(c->stream), "attn scratch")) return 0; cc_adec_reap(c); }
                c->adec_old[c->adec_n_old++] = c->adec;
            }
            c->adec = nullptr; c->adec_bytes = 0;
            size_t want = need + need / 2;
            if (cudaMalloc((void **)&c->adec, want) == cudaSuccess) c->adec_bytes = want;
            else { c->adec = nullptr; (void)cudaGetLastError(); }
        }
        if (c->adec) {
            cc_attn_dec_k<<<dim3((unsigned)p->H, (unsigned)nb, (unsigned)p->S), 256, 0, c->stream>>>(
                fp(q), fp(kc), fp(vc), fpw(o), fp(gate), c->adec, *p, 0, nb);
            cc_attn_dec_k<<<dim3((unsigned)p->H, (unsigned)p->S), 256, 0, c->stream>>>(
                fp(q), fp(kc), fp(vc), fpw(o), fp(gate), c->adec, *p, 1, nb);
            return launched(c, "attn (positions in blocks)");
        }
    }
    dim3 grid((unsigned)p->H, (unsigned)p->S);
    cc_attn_k<<<grid, 128, 0, c->stream>>>(fp(q), fp(kc), fp(vc), fpw(o), fp(gate), sel ? (const int *)sel->d : nullptr, *p);
    return launched(c, "attn");
}
static int cc_dnconv(CcBuf *in, CcBuf *w, CcBuf *ring, CcBuf *out, CcBuf *snap, const CcDnConv *p) {
    CcCtx *c = live();
    if (!c || !in || !w || !ring || !out || !p || p->S < 1 || p->CD < 1 || p->CK < 2 || p->CK > 9) return 0;
    if (p->snap_row >= 0 && !snap) return 0;
    cc_dnconv_k<<<grid1((size_t)p->CD, 64), 64, 0, c->stream>>>(fp(in), fp(w), fpw(ring), fpw(out), fpw(snap), *p);
    return launched(c, "dnconv");
}
static int cc_dnrec(int KD, CcBuf *cv, CcBuf *ab, CcBuf *z, CcBuf *st, CcBuf *prm, CcBuf *y, CcBuf *snap, const CcDnRec *p) {
    CcCtx *c = live();
    if (!c || !cv || !ab || !z || !st || !prm || !y || !p || p->S < 1 || p->VH < 1 || p->KH < 1 || p->VH % p->KH ||
        p->VD < 1 || p->VD > 128) return 0;
    if (p->snap_row >= 0 && !snap) return 0;
    const float *a = fp(cv), *b = fp(ab), *zz = fp(z), *pr = fp(prm);
    float *s = fpw(st), *yy = fpw(y), *sn = fpw(snap);
    static int rows_on = -1;
    if (rows_on < 0) { const char *e = getenv("COLI_CUDA_DNREC_ROWS"); rows_on = !(e && *e == '0'); }
    if (rows_on && p->S >= CC_DNREC_ROWS) {   /* a prompt chunk's rows, a block at a time */
        switch (KD) {
        case 4:   cc_dnrec_rows_k<4, 16><<<(unsigned)p->VH, 128, 0, c->stream>>>(a, b, zz, s, pr, yy, sn, *p); return launched(c, "dnrec rows");
        case 8:   cc_dnrec_rows_k<8, 16><<<(unsigned)p->VH, 128, 0, c->stream>>>(a, b, zz, s, pr, yy, sn, *p); return launched(c, "dnrec rows");
        case 16:  cc_dnrec_rows_k<16, 16><<<(unsigned)p->VH, 128, 0, c->stream>>>(a, b, zz, s, pr, yy, sn, *p); return launched(c, "dnrec rows");
        case 32:  cc_dnrec_rows_k<32, 16><<<(unsigned)p->VH, 128, 0, c->stream>>>(a, b, zz, s, pr, yy, sn, *p); return launched(c, "dnrec rows");
        case 64:  cc_dnrec_rows_k<64, 16><<<(unsigned)p->VH, 128, 0, c->stream>>>(a, b, zz, s, pr, yy, sn, *p); return launched(c, "dnrec rows");
        case 128: cc_dnrec_rows_k<128, 16><<<(unsigned)p->VH, 128, 0, c->stream>>>(a, b, zz, s, pr, yy, sn, *p); return launched(c, "dnrec rows");
        case 256: cc_dnrec_rows_k<256, 8><<<(unsigned)p->VH, 128, 0, c->stream>>>(a, b, zz, s, pr, yy, sn, *p); return launched(c, "dnrec rows");
        default: return 0;
        }
    }
    switch (KD) {
    case 4:   cc_dnrec_k<4><<<(unsigned)p->VH, 128, 0, c->stream>>>(a, b, zz, s, pr, yy, sn, *p); break;
    case 8:   cc_dnrec_k<8><<<(unsigned)p->VH, 128, 0, c->stream>>>(a, b, zz, s, pr, yy, sn, *p); break;
    case 16:  cc_dnrec_k<16><<<(unsigned)p->VH, 128, 0, c->stream>>>(a, b, zz, s, pr, yy, sn, *p); break;
    case 32:  cc_dnrec_k<32><<<(unsigned)p->VH, 128, 0, c->stream>>>(a, b, zz, s, pr, yy, sn, *p); break;
    case 64:  cc_dnrec_k<64><<<(unsigned)p->VH, 128, 0, c->stream>>>(a, b, zz, s, pr, yy, sn, *p); break;
    case 128: cc_dnrec_k<128><<<(unsigned)p->VH, 128, 0, c->stream>>>(a, b, zz, s, pr, yy, sn, *p); break;
    case 256: cc_dnrec_k<256><<<(unsigned)p->VH, 128, 0, c->stream>>>(a, b, zz, s, pr, yy, sn, *p); break;
    default: return 0;
    }
    return launched(c, "dnrec");
}
static int cc_ew(CcBuf *y, CcBuf *a, CcBuf *b, CcBuf *cb, CcBuf *e, const CcEw *p) {
    CcCtx *c = live();
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
    default: return 0;
    }
    cc_ew_k<<<grid1((size_t)p->n, 256), 256, 0, c->stream>>>(fpw(y), fp(a), fp(b), fp(cb), fp(e), *p);
    return launched(c, "ew");
}
static int cc_qsa(CcBuf *src, CcBuf *w, CcBuf *pk, CcBuf *cs, CcBuf *sc, CcBuf *sel, const CcQsa *p) {
    CcCtx *c = live();
    if (!c || !src || !pk || !p || p->ID < 1 || p->ID > 256 || p->R < 1) return 0;
    if (p->mode == 0) {
        if (!w || !cs || p->nb < 1 || 2 * p->half_ > p->ID) return 0;
        cc_qsa_k<<<(unsigned)p->nb, 128, 0, c->stream>>>(fp(src), fp(w), fpw(pk), fp(cs), nullptr, nullptr, *p);
    } else if (p->mode == 1) {
        if (!sc || !sel || p->S < 1 || p->IQ < 1 || p->nbmax < 1 || p->sel_row < 1) return 0;
        cc_qsa_k<<<(unsigned)p->S, 128, 0, c->stream>>>(fp(src), nullptr, fpw(pk), nullptr, fpw(sc), (int *)sel->d, *p);
    } else return 0;
    return launched(c, "qsa");
}
static int cc_ple(CcBuf *keys, CcBuf *hyp, CcBuf *val, CcBuf *prm, CcBuf *gated, CcBuf *normv, CcBuf *conv, CcBuf *ring, const CcPle *p) {
    CcCtx *c = live();
    if (!c || !hyp || !gated || !normv || !p || p->S < 1 || p->C < 1 || p->H < 1) return 0;
    if (p->mode == 0) {
        if (!keys || !val || !prm) return 0;
        cc_ple_k<<<(unsigned)(p->S * p->C), 256, 0, c->stream>>>(fp(keys), fpw(hyp), fp(val), fp(prm), fpw(gated), fpw(normv), nullptr, nullptr, *p);
    } else if (p->mode == 1) {
        if (!conv || !ring || p->CK < 2 || (p->CK - 1) * p->NG > 32) return 0;
        cc_ple_k<<<grid1((size_t)p->C * p->H, 256), 256, 0, c->stream>>>(nullptr, fpw(hyp), nullptr, nullptr, fpw(gated), fpw(normv), fp(conv), fpw(ring), *p);
    } else return 0;
    return launched(c, "ple");
}
static int cc_route(CcBuf *l, CcBuf *bias, CcBuf *ids, CcBuf *w, const CcRoute *p) {
    CcCtx *c = live();
    if (!c || !l || !ids || !w || !p || p->S < 1 || p->E < 1 || p->E > CC_ROUTE_EMAX || p->K < 1 || p->K > CC_ROUTE_KMAX ||
        p->K > p->E || p->kind < 0 || p->kind > 2 || p->has_bias < 0 || p->has_bias > 2 || (p->has_bias && !bias) || !(p->temp > 0.f) ||
        (p->n_group > 1 && (p->n_group > 64 || p->E % p->n_group || p->topk_group < 1 || p->topk_group > p->n_group))) return 0;
    if ((size_t)(p->l_off + (p->S - 1) * p->l_row + p->E) * 4 > l->bytes || (size_t)(p->i_off + (p->S - 1) * p->i_row + p->K) * 4 > ids->bytes ||
        (size_t)(p->w_off + (p->S - 1) * p->w_row + p->K) * 4 > w->bytes) return 0;
    cc_route_k<<<(unsigned)p->S, 256, 0, c->stream>>>(fp(l), fp(bias), (int *)ids->d, fpw(w), *p);
    return launched(c, "route");
}
static int cc_moe(CcBuf *tab, CcBuf *x, CcBuf *ids, CcBuf *w, CcBuf *hid, CcBuf *y, CcBuf *hy, const CcMoe *p) {
    CcCtx *c = live();
    if (!c || !tab || !ids || !hid || !p || p->parts < 1 || p->parts > 3 || p->S < 1 || p->K < 1) return 0;
    if ((p->parts & 1) && !x) return 0;
    if ((p->parts & 2) && (!w || !y || (p->h_off >= 0 && !hy))) return 0;
    if (cc_moe_hid_floats(p->S, p->K, p->D, p->I, p->E) * 4 > hid->bytes || (p->S > CC_MOE_FRAME_ROWS && p->E < 1)) return 0;
    g_st.ops++;
    if (!coli_cuda_moe_async(c->device, (const CcExpert *)tab->d, fp(x), (const int *)ids->d, fp(w), fpw(hid), fpw(y), fp(hy), p,
                             (void *)c->stream)) { c->lost = 1; return 0; }
    return 1;
}
static int cc_signal(CcBuf *flag, size_t word, unsigned v) {
    CcCtx *c = live();
    if (!c || !flag || flag->kind != CC_DOWN || (word + 1) * 4 > flag->bytes) return 0;
    cc_signal_k<<<1, 1, 0, c->stream>>>((volatile unsigned *)flag->d + word, v);
    return launched(c, "signal");
}
static int cc_wait_host(CcBuf *flag, size_t word, unsigned v) {
    CcCtx *c = live();
    if (!c || !flag || flag->kind != CC_UP || (word + 1) * 4 > flag->bytes) return 0;
    cc_wait_k<<<1, 1, 0, c->stream>>>((volatile unsigned *)flag->d + word, v);
    return launched(c, "wait host");
}
static int cc_moe_stage_op(CcBuf *b, size_t slot_bytes, int slots) {
    CcCtx *c = live();
    if (!c || (b && (b->kind != CC_DEV || slots < 1 || CC_MOE_STAGE_MAP + (size_t)slots * slot_bytes > b->bytes))) return 0;
    return coli_cuda_moe_stage(c->device, b ? (void *)b->d : nullptr, b ? slot_bytes : 0, b ? slots : 0);
}
static void cc_stats(CcStats *st) { if (st) *st = g_st; }

/* ---- staging: copies from page-locked host memory beside the frame ---------------- */
static int cc_copy_stream(CcCtx *c) {
    if (c->cs_ok) return 1;
    if (!cc_ok(c, cudaStreamCreateWithFlags(&c->cs, cudaStreamNonBlocking), "copy stream")) return 0;
    if (cudaEventCreateWithFlags(&c->ev_fetched, cudaEventDisableTiming) != cudaSuccess) {
        (void)cudaStreamDestroy(c->cs); return cc_ok(c, cudaErrorMemoryAllocation, "copy event");
    }
    if (cudaEventCreateWithFlags(&c->ev_free, cudaEventDisableTiming) != cudaSuccess) {
        (void)cudaEventDestroy(c->ev_fetched); (void)cudaStreamDestroy(c->cs); return cc_ok(c, cudaErrorMemoryAllocation, "copy event");
    }
    c->cs_ok = 1;
    return 1;
}
static int cc_fetch(CcBuf *dst, const CcFetch *r, int n) {
    CcCtx *c = live();
    if (!c || !dst || dst->kind != CC_DEV || n < 0 || (n && !r)) return 0;
    for (int i = 0; i < n; i++) if (!r[i].src || r[i].dst > dst->bytes || r[i].bytes > dst->bytes - r[i].dst) return 0;
    if (!cc_copy_stream(c)) return 0;
    /* an event never recorded is no wait: the first fetch starts at once */
    if (!cc_ok(c, cudaStreamWaitEvent(c->cs, c->ev_free, 0), "fetch's reader")) return 0;
    for (int i = 0; i < n; i++) {
        if (!r[i].bytes) continue;
        if (!cc_ok(c, cudaMemcpyAsync((char *)dst->d + r[i].dst, r[i].src, r[i].bytes, cudaMemcpyHostToDevice, c->cs), "fetch")) return 0;
        g_st.bytes_up += r[i].bytes;
    }
    g_st.ops++;
    return cc_ok(c, cudaEventRecord(c->ev_fetched, c->cs), "fetch event");
}
static int cc_fetch_wait(void) {
    CcCtx *c = live();
    if (!c || !c->cs_ok) return 0;
    g_st.ops++;
    return cc_ok(c, cudaStreamWaitEvent(c->stream, c->ev_fetched, 0), "fetch wait");
}
static int cc_fetch_done(void) {
    CcCtx *c = live();
    if (!c || !c->cs_ok) return 0;
    return cc_ok(c, cudaEventRecord(c->ev_free, c->stream), "fetch done");
}
static int cc_host_up(void *dst, const void *src, size_t bytes, unsigned *ticket) {
    CcCtx *c = live();
    if (!c || !dst || !src || !ticket) return 0;
    if (!cc_copy_stream(c)) return 0;
    if (!c->up_made) {
        if (cudaEventCreateWithFlags(&c->up_after, cudaEventDisableTiming) != cudaSuccess) return cc_ok(c, cudaErrorMemoryAllocation, "host up event");
        for (int i = 0; i < CC_UP_MARKS; i++)
            if (cudaEventCreateWithFlags(&c->up_ev[i], cudaEventDisableTiming) != cudaSuccess) {
                for (int j = 0; j < i; j++) (void)cudaEventDestroy(c->up_ev[j]);
                (void)cudaEventDestroy(c->up_after);
                return cc_ok(c, cudaErrorMemoryAllocation, "host up event");
            }
        c->up_made = 1;
    }
    unsigned t = ++c->up_seq;
    cudaEvent_t ev = c->up_ev[t % CC_UP_MARKS];
    if (t > CC_UP_MARKS && !cc_ok(c, cudaEventSynchronize(ev), "host up ring")) return 0;   /* the mark it reuses landed */
    if (!cc_ok(c, cudaEventRecord(c->up_after, c->stream), "host up after") ||
        !cc_ok(c, cudaStreamWaitEvent(c->cs, c->up_after, 0), "host up wait") ||
        !cc_ok(c, cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, c->cs), "host up") ||
        !cc_ok(c, cudaEventRecord(ev, c->cs), "host up mark")) return 0;
    g_st.bytes_up += bytes;
    *ticket = t;
    return 1;
}
static int cc_host_up_done(unsigned ticket) {
    CcCtx *c = live();
    if (!c || !c->up_made || !ticket || ticket > c->up_seq) return 0;
    if (c->up_seq - ticket >= CC_UP_MARKS) return 1;   /* its mark reused: waited for then */
    cudaError_t e = cudaEventQuery(c->up_ev[ticket % CC_UP_MARKS]);
    if (e == cudaErrorNotReady) { (void)cudaGetLastError(); return 0; }
    return cc_ok(c, e, "host up done");
}
static const void *cc_dev_addr(const CcBuf *b) { return b ? (const void *)b->d : nullptr; }   /* a mapped UP/DOWN buffer's: where the device reads it */
/* cc_attn with a window and sinks (Flint), the same three forms cc_attn picks: a prompt chunk
 * on the tensor cores (cc_attn_fk), a decode's or a verify's rows split over their positions
 * (cc_attn_dec_k, only where a prompt takes cc_attn_fk, as in cc_attn), else cc_attn_k; each
 * row's positions the window's, the sink in each softmax's denominator. */
static int cc_attn_w2(CcBuf *q, CcBuf *kc, CcBuf *vc, CcBuf *o, CcBuf *gate, CcBuf *snk, const CcAttnW *w, int ring) {
    CcCtx *c = live();
    if (!w) return 0;
    const CcAttn *p = &w->a;
    if (!c || !q || !kc || !vc || !o || p->S < 1 || p->H < 1 || p->KVH < 1 || p->H % p->KVH || p->hd < 1 || p->hd > 256 ||
        p->sel_row > 0 || w->win < 0 || (w->sink && !snk) || (p->has_gate && !gate) || ring < 0 ||
        (ring && (ring != p->cap || w->win < 1 || ring < w->win + p->S - 1))) return 0;   /* a ring holds the step's rows and their windows */
    if (!w->win && !w->sink) return cc_attn(q, kc, vc, o, gate, nullptr, p);
    const float *sk = w->sink ? fp(snk) : nullptr;
    const int win = w->win, so = w->sink_off;
    static int fl = -1, dec = -1;
    if (fl < 0) { const char *e = getenv("COLI_CUDA_CHAIN_FLASH"); fl = e && *e ? atoi(e) : 16; if (fl < 0) fl = 0; }
    if (dec < 0) { const char *e = getenv("COLI_CUDA_ATTN_DEC"); dec = !(e && *e == '0'); }
    int G = p->H / p->KVH, fk = 0;
#if !defined(__HIP__) && !defined(__HIP_PLATFORM_AMD__)
    fk = fl > 0 && c->major >= 7 && 16 % G == 0 && (p->hd == 64 || p->hd == 128 || p->hd == 256);
    if (fk && p->S >= fl) {
        dim3 fg((unsigned)((p->S + 16 / G - 1) / (16 / G)), (unsigned)p->KVH);
        if (p->hd == 64) cc_attn_fk<64><<<fg, 128, 0, c->stream>>>(fp(q), fp(kc), fp(vc), fpw(o), fp(gate), *p, G, win, sk, so, ring);
        else if (p->hd == 128) cc_attn_fk<128><<<fg, 128, 0, c->stream>>>(fp(q), fp(kc), fp(vc), fpw(o), fp(gate), *p, G, win, sk, so, ring);
        else cc_attn_fk<256><<<fg, 128, 0, c->stream>>>(fp(q), fp(kc), fp(vc), fpw(o), fp(gate), *p, G, win, sk, so, ring);
        return launched(c, "attn (tensor cores, window, sinks)");
    }
#endif
    /* a row's positions: the window's at most */
    #define CC_WPOS(r) (win > 0 && p->pos_base + (r) + 1 > win ? win : p->pos_base + (r) + 1)
    if (dec && fk && p->S <= CC_ATTN_DEC_ROWS && CC_WPOS(p->S - 1) >= CC_ATTN_DEC_MIN) {
        int k = 0;   /* the rows with fewer than CC_ATTN_DEC_MIN positions: cc_attn_k, as in cc_attn */
        while (k < p->S && CC_WPOS(k) < CC_ATTN_DEC_MIN) k++;
        if (k > 0) {
            CcAttn a = *p; a.S = k;
            cc_attn_k<<<dim3((unsigned)p->H, (unsigned)k), 128, 0, c->stream>>>(fp(q), fp(kc), fp(vc), fpw(o), fp(gate), nullptr, a,
                                                                                win, sk, so, ring);
            if (!launched(c, "attn (window, sinks)")) return 0;
            CcAttnW b = *w; b.a.S -= k; b.a.pos_base += k; b.a.q_off += k * p->q_row; b.a.o_off += k * p->o_row; b.a.g_off += k * p->g_row;
            return cc_attn_w2(q, kc, vc, o, gate, snk, &b, ring);
        }
        int nb = 1;
        for (int r = 0; r < p->S; r++) {
            int n = CC_WPOS(r), L = cc_attn_dec_len(n, p->H), m = (n + L - 1) / L;
            if (m > nb) nb = m;
        }
        size_t need = (size_t)p->S * p->H * nb * (p->hd + 2) * sizeof(float);
        if (need > c->adec_bytes) {   /* the frames before done first, as in cc_attn */
            if (c->adec) {   /* retired, not freed: the frames before may still read it, and a sync here can deadlock a hybrid layer */
                if (c->adec_n_old == 4) { if (!cc_ok(c, cudaStreamSynchronize(c->stream), "attn scratch")) return 0; cc_adec_reap(c); }
                c->adec_old[c->adec_n_old++] = c->adec;
            }
            c->adec = nullptr; c->adec_bytes = 0;
            size_t want = need + need / 2;
            if (cudaMalloc((void **)&c->adec, want) == cudaSuccess) c->adec_bytes = want;
            else { c->adec = nullptr; (void)cudaGetLastError(); }
        }
        if (c->adec) {
            cc_attn_dec_k<<<dim3((unsigned)p->H, (unsigned)nb, (unsigned)p->S), 256, 0, c->stream>>>(
                fp(q), fp(kc), fp(vc), fpw(o), fp(gate), c->adec, *p, 0, nb, win, sk, so, ring);
            cc_attn_dec_k<<<dim3((unsigned)p->H, (unsigned)p->S), 256, 0, c->stream>>>(
                fp(q), fp(kc), fp(vc), fpw(o), fp(gate), c->adec, *p, 1, nb, win, sk, so, ring);
            return launched(c, "attn (positions in blocks, window, sinks)");
        }
    }
    #undef CC_WPOS
    cc_attn_k<<<dim3((unsigned)p->H, (unsigned)p->S), 128, 0, c->stream>>>(fp(q), fp(kc), fp(vc), fpw(o), fp(gate), nullptr, *p,
                                                                         win, sk, so, ring);
    return launched(c, "attn (window, sinks)");
}
static int cc_attn_w(CcBuf *q, CcBuf *kc, CcBuf *vc, CcBuf *o, CcBuf *gate, CcBuf *snk, const CcAttnW *w) {
    return cc_attn_w2(q, kc, vc, o, gate, snk, w, 0);
}

static int cc_muse(void) { return 1; }   /* the norm's PRESCALE, ew's SIGMUL and RESGATE */
static int cc_kv16(void) { return 1; }   /* ew's F16 */
static int cc_pick(CcBuf *logits, CcBuf *out, int n) {
    CcCtx *c = live();
    if (!c || !logits || !out || n < 1 || (size_t)n * 4 > logits->bytes || out->bytes < 8) return 0;
    cc_pick_k<<<1, 256, 0, c->stream>>>(fp(logits), out->d, n);
    return launched(c, "pick");
}
static int cc_matmul_rows(ColiCudaTensor *t, CcBuf *ids, CcBuf *x, CcBuf *y, int n) {
    CcCtx *c = live();
    if (!c || !t || !ids || !x || !y || n < 1) return 0;
    int fmt, I, O; const void *w = nullptr; const float *sc = nullptr;
    if (!coli_cuda_tensor_shape(t, &fmt, &I, &O) || coli_cuda_tensor_device(t) != c->device || (fmt != 0 && fmt != 1)) return 0;
    if (!coli_cuda_tensor_dev(t, &w, &sc, nullptr) || (fmt == 1 && !sc)) return 0;
    if ((size_t)I * 4 > x->bytes || (size_t)n * 4 > ids->bytes || (size_t)n * 4 > y->bytes) return 0;
    g_st.matmuls++;
    unsigned grid = grid1((size_t)n, 8);
    if (fmt == 1 && I % 16 == 0 && I <= 12288) {
        size_t smem = (size_t)(I / 4 + I / 32 + 1) * sizeof(float4);
        if (smem <= 48 * 1024)
            cc_matmul_rows_i8_k<1><<<grid, 256, smem, c->stream>>>(fpw(y), fp(x), (const int8_t *)w, sc, (const int *)ids->d, I, O, n);
        else
            cc_matmul_rows_i8_k<0><<<grid, 256, (size_t)I * sizeof(float), c->stream>>>(fpw(y), fp(x), (const int8_t *)w, sc, (const int *)ids->d, I, O, n);
    } else {
        cc_matmul_rows_k<<<grid, 256, 0, c->stream>>>(fpw(y), fp(x), w, sc, (const int *)ids->d, fmt, I, O, n);
    }
    return launched(c, "matmul_rows");
}
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
};
extern "C" const ColiCudaChainOps *coli_cuda_chain_ops(void) { return &g_cc_ops; }
