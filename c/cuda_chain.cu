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
    int major;   /* the device's compute capability: cc_attn_fk's tensor cores from 7, cc_attn_f2's from 8 */
    int f2_big;  /* cc_attn_f2's head dim 256 (66 KiB of shared memory): 1 allowed, -1 refused, 0 not asked */
    int f2_big_h;   /* the same for its float16 cache's kernel */
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
    COLI_PDL_WAIT();
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
    COLI_PDL_WAIT();
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
    COLI_PDL_WAIT();
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
/* A cache's elements as floats: float32, or float16 (attn_h: a float16 KV cache holds the
 * numbers a float16 host cache holds, so the attention reads the same values). cc_kv4: four
 * of them at an offset that is a multiple of four. */
__device__ __forceinline__ float cc_kvf(const float *p, size_t i) { return p[i]; }
__device__ __forceinline__ float4 cc_kv4(const float *p, size_t i) { return *reinterpret_cast<const float4 *>(p + i); }
#if !defined(__HIP__) && !defined(__HIP_PLATFORM_AMD__)
__device__ __forceinline__ float cc_kvf(const __half *p, size_t i) { return __half2float(p[i]); }
__device__ __forceinline__ float4 cc_kv4(const __half *p, size_t i) {
    uint2 u = *reinterpret_cast<const uint2 *>(p + i);
    float2 a = __half22float2(*reinterpret_cast<const __half2 *>(&u.x)), b = __half22float2(*reinterpret_cast<const __half2 *>(&u.y));
    return make_float4(a.x, a.y, b.x, b.y);
}
#endif
template <typename KT> static const KT *kvp(CcBuf *b) { return b ? reinterpret_cast<const KT *>(b->d) : nullptr; }
/* one block of 128 per (head, row); K and V rows read once per tile of 128 positions;
 * hd <= 256: two dims per thread */
/* win > 0: a sliding window, the row's last win positions; snk: a sink logit per head at
 * snk[sink_off + h], in the denominator only (CcAttnW); ring > 0: the cache a ring of that
 * many rows, position t in row t % ring (a sliding layer's, cc_attn_wr) */
template <typename KT = float>
__global__ static void __launch_bounds__(128) cc_attn_k(const float *__restrict__ q, const KT *__restrict__ kc,
        const KT *__restrict__ vc, float *__restrict__ o, const float *__restrict__ gt, const int *__restrict__ sel, CcAttn p,
        int win = 0, const float *__restrict__ snk = nullptr, int sink_off = 0, int ring = 0) {
    COLI_PDL_WAIT();
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
            for (int d = 0; d < p.hd; d++) a += qs[d] * cc_kvf(kc, kb + d);
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
            if (d0 < p.hd) acc0 += w * cc_kvf(vc, vb + d0);
            if (d1 < p.hd) acc1 += w * cc_kvf(vc, vb + d1);
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
template<int HD, typename KT = float>
__global__ static void __launch_bounds__(128) cc_attn_fk(const float *__restrict__ q, const KT *__restrict__ kc,
        const KT *__restrict__ vc, float *__restrict__ o, const float *__restrict__ gt, CcAttn p, int GQ,
        int win = 0, const float *__restrict__ snk = nullptr, int sink_off = 0, int ring = 0) {   /* CcAttnW's window and sinks, a ring */
    COLI_PDL_WAIT();
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
            for (int e = tid; e < CC_FA_BC * HD; e += 128) {   /* K of keys k0 .. k0 + 63, zero past nt */
                int kl = e / HD, d = e - kl * HD, key = k0 + kl;
                float v = key < nt ? cc_kvf(kc, p.k_off + (size_t)(g * p.cap + (ring ? key % ring : key)) * HD + d) : 0.f;
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
            for (int e = tid; e < CC_FA_BC * HD; e += 128) {   /* V of the same keys */
                int kl = e / HD, d = e - kl * HD, key = k0 + kl;
                float v = key < nt ? cc_kvf(vc, p.v_off + (size_t)(g * p.cap + (ring ? key % ring : key)) * HD + d) : 0.f;
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
/* A prompt chunk's causal attention in one pass on the tensor cores' m16n8k16 f16 MMA (sm_80
 * and later), FlashAttention-2's online softmax. A block of NW warps takes 16 * NW query rows:
 * 16 * NW / GQ tokens x the GQ query heads of one kv head, warp w rows 16w .. 16w + 15, so each
 * key block of BC keys is staged once for all of them (K and V rounded to f16 as they are
 * staged, rows of HD + 8 halves). A warp's scores stay in its registers: per row the running
 * max m and sum l rescale its output fragments as the blocks go by, and P = exp(s - m) goes
 * from the score fragments straight into the A operand of P V (the accumulator's layout is
 * the A operand's, two key tiles a k-step). Q is scaled before its f16 rounding and held in
 * registers (QREG), or read from shared memory each block (HD 256: the registers hold O).
 * Keys: t <= pos_base + token(r), with a window t > pos_base + token(r) - win, none past nt;
 * a warp skips the blocks none of its rows sees. A sink joins the denominator from the start
 * (m = sink, l = 1); key t of the cache at row t % ring with a ring. cc_attn_fk took 9.6 s of
 * a 16K prompt on Flint (RTX PRO 4000): 16 rows a block re-read K and V from L2 for every 16
 * rows, twice. Not bit-identical to cc_attn_fk (one pass, the sums in another order).
 * Grid: x = ceil(S / (16 * NW / GQ)) token groups, y = kv heads; dynamic shared memory
 * cc_f2_smem_bytes. */
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 && !defined(__HIP__) && !defined(__HIP_PLATFORM_AMD__)
#define CC_F2_BODY 1
__device__ __forceinline__ void cc_f2_mma(float *c, const uint32_t *a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void cc_f2_ldm(uint32_t *r, const __half *p) {
    unsigned s = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}
__device__ __forceinline__ void cc_f2_ldmt(uint32_t *r, const __half *p) {
    unsigned s = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}
__device__ __forceinline__ uint32_t cc_f2_pack(float a, float b) {
    __half2 h = __floats2half2_rn(a, b);
    return *reinterpret_cast<uint32_t *>(&h);
}
/* four floats as four halves at d (8-byte aligned) */
__device__ __forceinline__ void cc_f2_put(__half *d, float4 v, float s) {
    uint2 u = make_uint2(cc_f2_pack(v.x * s, v.y * s), cc_f2_pack(v.z * s, v.w * s));
    *reinterpret_cast<uint2 *>(d) = u;
}
#else
#define CC_F2_BODY 0
#endif
template<int HD, int NW, int BC, bool QREG>
static constexpr size_t cc_f2_smem_bytes() { return ((size_t)2 * BC * (HD + 8) + (QREG ? 0 : (size_t)16 * NW * (HD + 8))) * 2; }
template<int HD, int NW, int BC, bool QREG, typename KT = float>
__global__ static void __launch_bounds__(32 * NW) cc_attn_f2(const float *__restrict__ q, const KT *__restrict__ kc,
        const KT *__restrict__ vc, float *__restrict__ o, const float *__restrict__ gt, CcAttn p, int GQ,
        int win, const float *__restrict__ snk, int sink_off, int ring) {
    COLI_PDL_WAIT();
#if CC_F2_BODY
    constexpr int LD = HD + 8, BR = 16 * NW, NT = 32 * NW, NKT = BC / 8, NDT = HD / 8, KS = HD / 16;
    static_assert(!QREG || BR <= 2 * BC, "Q is staged where K and V go");
    extern __shared__ __align__(16) unsigned char cc_f2_sm[];
    __half *sK = reinterpret_cast<__half *>(cc_f2_sm), *sV = sK + BC * LD, *sQ = QREG ? sK : sV + BC * LD;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gid = lane >> 2, tig = lane & 3;
    const int TPB = BR / GQ, g = blockIdx.y, t0 = blockIdx.x * TPB, S = p.S, pb = p.pos_base, nt = pb + S;
    const int tlast = min(t0 + TPB, S) - 1;
    for (int e = tid; e < BR * HD / 4; e += NT) {   /* Q, scaled, row r = (token t0 + r / GQ, head g * GQ + r % GQ) */
        int r = e / (HD / 4), d = (e - r * (HD / 4)) * 4, tk = t0 + r / GQ, h = g * GQ + r % GQ;
        float4 v = tk < S ? *reinterpret_cast<const float4 *>(q + p.q_off + (size_t)tk * p.q_row + (size_t)h * p.q_seg + d)
                          : make_float4(0.f, 0.f, 0.f, 0.f);
        cc_f2_put(sQ + r * LD + d, v, p.scale);
    }
    __syncthreads();
    const int ar = (lane & 7) + ((lane >> 3) & 1) * 8, ac = (lane >> 4) * 8;   /* an A operand's ldmatrix row and column */
    const int br = (lane & 7) + ((lane >> 4) << 3), bc = ((lane >> 3) & 1) * 8;  /* K's: two key tiles, two halves of a k-step */
    const int vr = (lane & 7) + (((lane >> 3) & 1) << 3), vcol = (lane >> 4) << 3; /* V's (transposed): two halves of 16 keys, two dim tiles */
    uint32_t qa[QREG ? KS : 1][4];
    if (QREG) {
        #pragma unroll
        for (int ks = 0; ks < KS; ks++) cc_f2_ldm(qa[ks], sQ + (warp * 16 + ar) * LD + ks * 16 + ac);
        __syncthreads();   /* sQ is where the first K block goes */
    }
    const int r0 = warp * 16 + gid, r1 = r0 + 8;
    const int qp0 = pb + min(t0 + r0 / GQ, S - 1), qp1 = pb + min(t0 + r1 / GQ, S - 1);
    const int wlo = pb + min(t0 + (warp * 16) / GQ, S - 1), whi = pb + min(t0 + (warp * 16 + 15) / GQ, S - 1);
    float m0 = -1e30f, m1 = -1e30f, l0 = 0.f, l1 = 0.f;
    if (snk) {   /* the sink: m = its logit, and one term of l (on one lane of the four that share a row) */
        m0 = snk[sink_off + g * GQ + r0 % GQ]; m1 = snk[sink_off + g * GQ + r1 % GQ];
        if (tig == 0) { l0 = 1.f; l1 = 1.f; }
    }
    float oa[NDT][4];
    #pragma unroll
    for (int dt = 0; dt < NDT; dt++) oa[dt][0] = oa[dt][1] = oa[dt][2] = oa[dt][3] = 0.f;
    const int kb_lo = win > 0 && pb + t0 - win + 1 > 0 ? (pb + t0 - win + 1) / BC : 0, kb_hi = (pb + tlast) / BC;
    for (int kb = kb_lo; kb <= kb_hi; kb++) {
        const int k0 = kb * BC;
        for (int e = tid; e < BC * HD / 4; e += NT) {   /* K and V of keys k0 .. k0 + BC - 1, zero past nt */
            int kl = e / (HD / 4), d = (e - kl * (HD / 4)) * 4, key = k0 + kl;
            float4 kv = make_float4(0.f, 0.f, 0.f, 0.f), vv = kv;
            if (key < nt) {
                size_t at = (size_t)(g * p.cap + (ring ? key % ring : key)) * HD + d;
                kv = cc_kv4(kc, p.k_off + at);
                vv = cc_kv4(vc, p.v_off + at);
            }
            cc_f2_put(sK + kl * LD + d, kv, 1.f);
            cc_f2_put(sV + kl * LD + d, vv, 1.f);
        }
        __syncthreads();
        /* none of this warp's rows sees a key of the block: past its last row, or before its
         * first row's window */
        if (k0 <= whi && (win <= 0 || k0 + BC - 1 + win > wlo)) {
            float sc[NKT][4];
            #pragma unroll
            for (int j = 0; j < NKT; j++) sc[j][0] = sc[j][1] = sc[j][2] = sc[j][3] = 0.f;
            #pragma unroll
            for (int ks = 0; ks < KS; ks++) {
                uint32_t a[4];
                if (QREG) { a[0] = qa[ks][0]; a[1] = qa[ks][1]; a[2] = qa[ks][2]; a[3] = qa[ks][3]; }
                else cc_f2_ldm(a, sQ + (warp * 16 + ar) * LD + ks * 16 + ac);
                #pragma unroll
                for (int j = 0; j < NKT; j += 2) {
                    uint32_t b[4];
                    cc_f2_ldm(b, sK + (j * 8 + br) * LD + ks * 16 + bc);
                    cc_f2_mma(sc[j], a, b[0], b[1]);
                    cc_f2_mma(sc[j + 1], a, b[2], b[3]);
                }
            }
            /* the masks only where a row's position or window cuts the block */
            const bool edge = !(k0 + BC - 1 <= wlo && (win <= 0 || k0 + win > whi));
            float x0 = -1e30f, x1 = -1e30f;
            #pragma unroll
            for (int j = 0; j < NKT; j++)
                #pragma unroll
                for (int u = 0; u < 2; u++) {
                    if (edge) {
                        int key = k0 + j * 8 + tig * 2 + u;
                        if (key > qp0 || (win > 0 && key + win <= qp0)) sc[j][u] = -1e30f;
                        if (key > qp1 || (win > 0 && key + win <= qp1)) sc[j][2 + u] = -1e30f;
                    }
                    x0 = fmaxf(x0, sc[j][u]); x1 = fmaxf(x1, sc[j][2 + u]);
                }
            x0 = fmaxf(x0, __shfl_xor_sync(0xffffffffu, x0, 1)); x0 = fmaxf(x0, __shfl_xor_sync(0xffffffffu, x0, 2));
            x1 = fmaxf(x1, __shfl_xor_sync(0xffffffffu, x1, 1)); x1 = fmaxf(x1, __shfl_xor_sync(0xffffffffu, x1, 2));
            const float n0 = fmaxf(m0, x0), n1 = fmaxf(m1, x1), c0 = __expf(m0 - n0), c1 = __expf(m1 - n1);
            m0 = n0; m1 = n1; l0 *= c0; l1 *= c1;
            #pragma unroll
            for (int dt = 0; dt < NDT; dt++) { oa[dt][0] *= c0; oa[dt][1] *= c0; oa[dt][2] *= c1; oa[dt][3] *= c1; }
            uint32_t pa[BC / 16][4];
            #pragma unroll
            for (int j = 0; j < NKT; j++) {
                float p0 = sc[j][0] > -1e29f ? __expf(sc[j][0] - m0) : 0.f, p1 = sc[j][1] > -1e29f ? __expf(sc[j][1] - m0) : 0.f;
                float p2 = sc[j][2] > -1e29f ? __expf(sc[j][2] - m1) : 0.f, p3 = sc[j][3] > -1e29f ? __expf(sc[j][3] - m1) : 0.f;
                l0 += p0 + p1; l1 += p2 + p3;
                pa[j >> 1][(j & 1) * 2] = cc_f2_pack(p0, p1);
                pa[j >> 1][(j & 1) * 2 + 1] = cc_f2_pack(p2, p3);
            }
            #pragma unroll
            for (int kk = 0; kk < BC / 16; kk++)
                #pragma unroll
                for (int dt = 0; dt < NDT; dt += 2) {
                    uint32_t b[4];
                    cc_f2_ldmt(b, sV + (kk * 16 + vr) * LD + dt * 8 + vcol);
                    cc_f2_mma(oa[dt], pa[kk], b[0], b[1]);
                    cc_f2_mma(oa[dt + 1], pa[kk], b[2], b[3]);
                }
        }
        __syncthreads();   /* sK and sV are rewritten next block */
    }
    l0 += __shfl_xor_sync(0xffffffffu, l0, 1); l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 1); l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
    const float i0 = l0 > 0.f ? 1.f / l0 : 0.f, i1 = l1 > 0.f ? 1.f / l1 : 0.f;
    #pragma unroll
    for (int half = 0; half < 2; half++) {
        const int r = half ? r1 : r0, tk = t0 + r / GQ, h = g * GQ + r % GQ;
        if (tk >= S) continue;
        const float inv = half ? i1 : i0;
        float *orow = o + p.o_off + (size_t)tk * p.o_row + (size_t)h * HD;
        const float *grow = gt + p.g_off + (size_t)tk * p.g_row + (size_t)h * p.g_seg;
        #pragma unroll
        for (int dt = 0; dt < NDT; dt++) {
            const int d = dt * 8 + tig * 2;
            float v0 = oa[dt][half * 2] * inv, v1 = oa[dt][half * 2 + 1] * inv;
            if (p.has_gate) { v0 *= 1.f / (1.f + __expf(-grow[d])); v1 *= 1.f / (1.f + __expf(-grow[d + 1])); }
            orow[d] = v0; orow[d + 1] = v1;
        }
    }
#endif
}
/* cc_attn_f2 for a prompt chunk where it applies (1: launched): sm_80 and later, head dim 64,
 * 128 or 256, the query heads of a kv head dividing 64, rows and caches at float4 offsets;
 * COLI_CUDA_CHAIN_FLASH2=0 keeps cc_attn_fk */
template <typename KT>
static int cc_attn_f2_go(CcCtx *c, CcBuf *q, CcBuf *kc, CcBuf *vc, CcBuf *o, CcBuf *gate, const CcAttn *p, int G,
                         int win, const float *sk, int so, int ring) {
#if !defined(__HIP__) && !defined(__HIP_PLATFORM_AMD__)
    static int on = -1;
    if (on < 0) { const char *e = getenv("COLI_CUDA_CHAIN_FLASH2"); on = !(e && *e == '0'); }
    if (!on || c->major < 8 || 64 % G || (p->hd != 64 && p->hd != 128 && p->hd != 256) ||
        (p->q_off | p->q_row | p->q_seg | p->k_off | p->v_off) % 4) return 0;
    dim3 fg((unsigned)((p->S + 64 / G - 1) / (64 / G)), (unsigned)p->KVH);
    if (p->hd == 64) {
        coli_launch(cc_attn_f2<64, 4, 64, true, KT>, fg, 128, cc_f2_smem_bytes<64, 4, 64, true>(), c->stream, 
            fp(q), kvp<KT>(kc), kvp<KT>(vc), fpw(o), fp(gate), *p, G, win, sk, so, ring);
    } else if (p->hd == 128) {
        coli_launch(cc_attn_f2<128, 4, 64, true, KT>, fg, 128, cc_f2_smem_bytes<128, 4, 64, true>(), c->stream, 
            fp(q), kvp<KT>(kc), kvp<KT>(vc), fpw(o), fp(gate), *p, G, win, sk, so, ring);
    } else {
        const size_t sm = cc_f2_smem_bytes<256, 4, 32, false>();
        int &big = sizeof(KT) == 2 ? c->f2_big_h : c->f2_big;   /* a float16 cache's kernel is another function */
        if (!big) {
            int optin = 0;
            big = cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, c->device) == cudaSuccess &&
                  (size_t)optin >= sm &&
                  cudaFuncSetAttribute(cc_attn_f2<256, 4, 32, false, KT>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm) == cudaSuccess
                  ? 1 : -1;
            (void)cudaGetLastError();
        }
        if (big < 0) return 0;
        coli_launch(cc_attn_f2<256, 4, 32, false, KT>, fg, 128, sm, c->stream, fp(q), kvp<KT>(kc), kvp<KT>(vc), fpw(o), fp(gate), *p, G, win, sk, so, ring);
    }
    return 1;
#else
    (void)c; (void)q; (void)kc; (void)vc; (void)o; (void)gate; (void)p; (void)G; (void)win; (void)sk; (void)so; (void)ring;
    return 0;
#endif
}
/* one thread per channel, the rows in order, the channel's ring in registers */
__global__ static void cc_dnconv_k(const float *__restrict__ xin, const float *__restrict__ w, float *__restrict__ ring,
                                   float *__restrict__ o, float *__restrict__ snap, CcDnConv p) {
    COLI_PDL_WAIT();
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
    COLI_PDL_WAIT();
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
    COLI_PDL_WAIT();
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
    COLI_PDL_WAIT();
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
    COLI_PDL_WAIT();
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
    COLI_PDL_WAIT();
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
    COLI_PDL_WAIT();
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
    COLI_PDL_WAIT();
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
    COLI_PDL_WAIT();
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
    COLI_PDL_WAIT();
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
    COLI_PDL_WAIT();
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
    if (!c || !dst || !src || (n > 0 && !r)) return 0;
    for (int i = 0; i < n; i++)
        if ((r[i].dst + r[i].n) * 4 > dst->bytes || (r[i].src + r[i].n) * 4 > src->bytes) return 0;
    CcGrid g;
    const int grids = dst != src && dst->d && src->d;
    for (int i = 0; i < n; ) {
        int k = grids ? cc_grid_run(r + i, n - i) : 1;
        if (k >= 2 && cc_grid_of(r + i, k, &g)) {   /* one kernel for each grid */
            long long tot = (long long)g.n * g.len;
            coli_launch(cc_copy_grid_k, grid1((size_t)tot, 256), 256, 0, c->stream, dst->d, src->d, g);
            if (!launched(c, "copy regions (grid)")) return 0;
            i += k;
            continue;
        }
        if (r[i].n && !cc_ok(c, cudaMemcpyAsync(dst->d + r[i].dst, src->d + r[i].src, r[i].n * 4, cudaMemcpyDefault, c->stream),
                             "copy regions"))
            return 0;
        i++;
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
    coli_launch(cc_norm_k, grid1((size_t)p->nseg, 1), 256, 0, c->stream, fp(x), fp(w), fpw(y), *p);
    return launched(c, "norm");
}
static int cc_rope(CcBuf *x, CcBuf *cs, const CcRope *p) {
    CcCtx *c = live();
    if (!c || !x || !cs || !p || p->nseg < 1 || p->half_ < 1 || p->per_row < 1) return 0;
    coli_launch(cc_rope_k, grid1((size_t)p->nseg * p->half_, 64), 64, 0, c->stream, fpw(x), fp(cs), *p);
    return launched(c, "rope");
}
static int cc_rope_f16(CcBuf *x, CcBuf *v, CcBuf *cs, const CcRope *p, int seg) {
    CcCtx *c = live();
    if (!c || !x || !v || !cs || !p || p->nseg < 1 || p->half_ < 1 || p->per_row < 1 || seg < 2 * p->half_) return 0;
    coli_launch(cc_rope_f16_k, grid1((size_t)p->nseg * seg, 128), 128, 0, c->stream, fpw(x), fpw(v), fp(cs), *p, seg);
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
template <typename KT = float>
__global__ static void __launch_bounds__(256) cc_attn_dec_k(const float *__restrict__ q, const KT *__restrict__ kc,
        const KT *__restrict__ vc, float *__restrict__ o, const float *__restrict__ gt, float *__restrict__ xs,
        CcAttn p, int mode, int nb, int win = 0, const float *__restrict__ snk = nullptr, int sink_off = 0, int ring = 0,
        int hc = 0) {   /* hc: the blocks' rule's heads (cc_attn_dec_len), 0 = H */
    COLI_PDL_WAIT();
    __shared__ float qs[256];
    __shared__ float sc[256];
    __shared__ float red[8];
    int tid = threadIdx.x, wp = tid >> 5, lane = tid & 31, h = blockIdx.x, hd = p.hd, stride = hd + 2;
    if (hc < 1) hc = p.H;
    if (mode == 0) {
        int b = blockIdx.y, s = blockIdx.z, kvh = h / (p.H / p.KVH);
        int n = p.pos_base + s + 1, first = win > 0 && n > win ? n - win : 0;   /* a window: its last win positions */
        int L = cc_attn_dec_len(n - first, hc), t0 = first + b * L, cnt = min(L, n - t0);
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
            const KT *kr = kc + p.k_off + (size_t)(kbase + (ring ? (t0 + j) % ring : t0 + j)) * hd;
            float a = 0.f;
            for (int d = lane; d < hd; d += 32) a += qs[d] * cc_kvf(kr, d);
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
            for (int j = 0; j < cnt; j++) acc += sc[j] * cc_kvf(vc, p.v_off + (size_t)(kbase + (ring ? (t0 + j) % ring : t0 + j)) * hd + d);
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
    int s = blockIdx.y, n = p.pos_base + s + 1, first = win > 0 && n > win ? n - win : 0;
    int L = cc_attn_dec_len(n - first, hc), nbr = (n - first + L - 1) / L;
    size_t xo = (size_t)(s * p.H + h) * nb * stride;
    float mm = -3.0e38f;
    for (int b = tid; b < nbr; b += 256) mm = fmaxf(mm, xs[xo + (size_t)b * stride]);
    for (int m = 16; m > 0; m >>= 1) mm = fmaxf(mm, cc_shfl_xor(mm, m));
    if (lane == 0) red[wp] = mm;
    __syncthreads();
    float M = snk ? snk[sink_off + h] : -3.0e38f;   /* the sink joins the max and the sum, with no value */
    for (int g = 0; g < 8; g++) M = fmaxf(M, red[g]);
    float Lt = snk ? expf(snk[sink_off + h] - M) : 0.f, v = 0.f;
    for (int b0 = 0; b0 < nbr; b0 += 128) {
        int nc = min(128, nbr - b0);
        __syncthreads();   /* red, then the last chunk's weights, read */
        if (tid < nc) { size_t xb = xo + (size_t)(b0 + tid) * stride; float w = expf(xs[xb] - M); sc[tid] = w; sc[128 + tid] = xs[xb + 1] * w; }
        __syncthreads();
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
    __syncthreads();
    Lt = red[0];
    float inv = Lt > 0.f ? 1.f / Lt : 0.f;
    int ob = p.o_off + s * p.o_row + h * hd, gb = p.g_off + s * p.g_row + h * p.g_seg;
    if (tid < hd) {
        v *= inv;
        if (p.has_gate) v *= 1.f / (1.f + expf(-gt[gb + tid]));
        o[ob + tid] = v;
    }
}
/* cc_attn_dec_k's mode 0 for the G query heads of a KV head at once: a block a (KV head, block
 * of positions, row), each K and V row read once for the G heads instead of once a head, four
 * floats a load (chain_attn_dec.comp's mode 2). Qwen3.6's eight heads a KV head after an 18K
 * prompt read 74 MB of rows a layer eight times. The scores a thread or two a position; V a float4
 * column a thread, the positions over 256 / (hd / 4) groups of threads, their sums added group by group.
 * The blocks' rule counts four blocks a KV head (hc); the join is cc_attn_dec_k's mode 1 with
 * that rule. Dynamic shared memory: the heads' q rows and scores, G * (hd + 256) floats. */
template <int G, typename KT = float>
__global__ static void __launch_bounds__(256) cc_attn_decg_k(const float *__restrict__ q, const KT *__restrict__ kc,
        const KT *__restrict__ vc, float *__restrict__ xs, CcAttn p, int nb, int win, int ring, int hc, int tpp) {
    COLI_PDL_WAIT();
    extern __shared__ float4 cc_decg_sm[];
    __shared__ float red[G * 8];
    const int hd = p.hd, stride = hd + 2, NV = hd / 4, NG = 256 / NV;
    float4 *qv = cc_decg_sm;                       /* G * NV: the heads' q rows, then the groups' V sums */
    float *sc = (float *)(cc_decg_sm + G * NV);    /* G * 256 */
    int tid = threadIdx.x, wp = tid >> 5, lane = tid & 31, kvh = blockIdx.x, b = blockIdx.y, s = blockIdx.z;
    int n = p.pos_base + s + 1, first = win > 0 && n > win ? n - win : 0;
    int L = cc_attn_dec_len(n - first, hc), t0 = first + b * L, cnt = min(L, n - t0);
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
        qv[i] = make_float4(x[0], x[1], x[2], x[3]);
    }
    __syncthreads();
    int kbase = kvh * p.cap;
    /* the scores. tpp 0: a K row over a warp's lanes, two float4s a lane, the heads' sums by
     * shuffles; tpp 1, 2 or 4: that many threads a position (adjacent lanes, a float4 each in
     * turn along the K row) against the G q rows, their sums joined across the tpp lanes, the max
     * on the way (chain_attn_dec.comp's mode 2), the loop's trips the block's so every lane meets
     * the shuffle. cc_decg_tpp chooses */
    float M[G], e[G];
#pragma unroll
    for (int g = 0; g < G; g++) M[g] = -3.0e38f;
    if (tpp == 0) {
        for (int j = wp; j < cnt; j += 8) {
            size_t kr = p.k_off + (size_t)(kbase + (ring ? (t0 + j) % ring : t0 + j)) * hd;
            float4 kk[2];
#pragma unroll
            for (int i = 0; i < 2; i++) { int c = lane + 32 * i; kk[i] = c < NV ? cc_kv4(kc, kr + 4 * c) : make_float4(0.f, 0.f, 0.f, 0.f); }
            float a[G];
#pragma unroll
            for (int g = 0; g < G; g++) a[g] = 0.f;
#pragma unroll
            for (int i = 0; i < 2; i++) {
                int c = lane + 32 * i;
                if (c < NV)
#pragma unroll
                    for (int g = 0; g < G; g++) {
                        float4 x = qv[g * NV + c];
                        a[g] += x.x * kk[i].x + x.y * kk[i].y + x.z * kk[i].z + x.w * kk[i].w;
                    }
            }
#pragma unroll
            for (int g = 0; g < G; g++) {
                float x = a[g];
                for (int m = 16; m > 0; m >>= 1) x += cc_shfl_xor(x, m);
                if (lane == 0) sc[g * 256 + j] = x * p.scale;
            }
        }
        __syncthreads();
        for (int j = tid; j < cnt; j += 256)
#pragma unroll
            for (int g = 0; g < G; g++) M[g] = fmaxf(M[g], sc[g * 256 + j]);
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
                    float4 kk = cc_kv4(kc, kr + 4 * c);
#pragma unroll
                    for (int g = 0; g < G; g++) {
                        float4 x = qv[g * NV + c];
                        a[g] += x.x * kk.x + x.y * kk.y + x.z * kk.z + x.w * kk.w;
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
                    M[g] = fmaxf(M[g], x);
                }
        }
    }
#pragma unroll
    for (int g = 0; g < G; g++) {
        float x = M[g];
        for (int m = 16; m > 0; m >>= 1) x = fmaxf(x, cc_shfl_xor(x, m));
        if (lane == 0) red[g * 8 + wp] = x;
    }
    __syncthreads();
#pragma unroll
    for (int g = 0; g < G; g++) { M[g] = red[g * 8]; for (int k = 1; k < 8; k++) M[g] = fmaxf(M[g], red[g * 8 + k]); }
    __syncthreads();
#pragma unroll
    for (int g = 0; g < G; g++) e[g] = 0.f;
    for (int j = tid; j < cnt; j += 256)
#pragma unroll
        for (int g = 0; g < G; g++) { float w = expf(sc[g * 256 + j] - M[g]); sc[g * 256 + j] = w; e[g] += w; }
#pragma unroll
    for (int g = 0; g < G; g++) {
        float x = e[g];
        for (int m = 16; m > 0; m >>= 1) x += cc_shfl_xor(x, m);
        if (lane == 0) red[g * 8 + wp] = x;
    }
    __syncthreads();
    int c = tid % NV, grp = tid / NV;
    float4 acc[G];
#pragma unroll
    for (int g = 0; g < G; g++) acc[g] = make_float4(0.f, 0.f, 0.f, 0.f);
    /* four positions' loads in flight, then their sums in position order (the bits of one at a time) */
    int j = grp;
    for (; j + 3 * NG < cnt; j += 4 * NG) {
        float4 v[4];
#pragma unroll
        for (int u = 0; u < 4; u++) { int t = t0 + j + u * NG; v[u] = cc_kv4(vc, p.v_off + (size_t)(kbase + (ring ? t % ring : t)) * hd + 4 * c); }
#pragma unroll
        for (int u = 0; u < 4; u++)
#pragma unroll
            for (int g = 0; g < G; g++) {
                float w = sc[g * 256 + j + u * NG];
                acc[g].x += w * v[u].x; acc[g].y += w * v[u].y; acc[g].z += w * v[u].z; acc[g].w += w * v[u].w;
            }
    }
    for (; j < cnt; j += NG) {
        float4 v = cc_kv4(vc, p.v_off + (size_t)(kbase + (ring ? (t0 + j) % ring : t0 + j)) * hd + 4 * c);
#pragma unroll
        for (int g = 0; g < G; g++) {
            float w = sc[g * 256 + j];
            acc[g].x += w * v.x; acc[g].y += w * v.y; acc[g].z += w * v.z; acc[g].w += w * v.w;
        }
    }
    for (int k = NG - 1; k >= 1; k--) {   /* the groups' sums into group 0's, the last group first */
        if (grp == k)
#pragma unroll
            for (int g = 0; g < G; g++) {
                float4 x = qv[g * NV + c];
                qv[g * NV + c] = k == NG - 1 ? acc[g] : make_float4(x.x + acc[g].x, x.y + acc[g].y, x.z + acc[g].z, x.w + acc[g].w);
            }
        __syncthreads();
    }
    if (grp == 0)
#pragma unroll
        for (int g = 0; g < G; g++) {
            float4 v = acc[g];
            if (NG > 1) { float4 x = qv[g * NV + c]; v = make_float4(v.x + x.x, v.y + x.y, v.z + x.z, v.w + x.w); }
            float *o = xs + xo0 + g * xg + 2 + 4 * c;
            o[0] = v.x; o[1] = v.y; o[2] = v.z; o[3] = v.w;
        }
    if (tid < G) {
        float l = 0.f;
        for (int k = 0; k < 8; k++) l += red[tid * 8 + k];
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
    dim3 g((unsigned)p.KVH, (unsigned)nb, (unsigned)p.S);
    size_t sm = (size_t)G * (p.hd + 256) * sizeof(float);
    int tpp = cc_decg_tpp();
    switch (G) {
    case 2: coli_launch(cc_attn_decg_k<2, KT>, g, 256, sm, c->stream, q, kc, vc, xs, p, nb, win, ring, hc, tpp); break;
    case 4: coli_launch(cc_attn_decg_k<4, KT>, g, 256, sm, c->stream, q, kc, vc, xs, p, nb, win, ring, hc, tpp); break;
    case 8: coli_launch(cc_attn_decg_k<8, KT>, g, 256, sm, c->stream, q, kc, vc, xs, p, nb, win, ring, hc, tpp); break;
    default: coli_launch(cc_attn_decg_k<16, KT>, g, 256, sm, c->stream, q, kc, vc, xs, p, nb, win, ring, hc, tpp); break;
    }
}
template <typename KT>
static int cc_attn_t(CcBuf *q, CcBuf *kc, CcBuf *vc, CcBuf *o, CcBuf *gate, CcBuf *sel, const CcAttn *p) {
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
    if (fl > 0 && p->S >= fl && p->sel_row <= 0 && cc_attn_f2_go<KT>(c, q, kc, vc, o, gate, p, G, 0, nullptr, 0, 0))
        return launched(c, "attn (tensor cores, one pass)");
    if (fl > 0 && p->S >= fl && c->major >= 7 && p->sel_row <= 0 && 16 % G == 0 && (p->hd == 64 || p->hd == 128 || p->hd == 256)) {
        dim3 fg((unsigned)((p->S + 16 / G - 1) / (16 / G)), (unsigned)p->KVH);
        if (p->hd == 64) coli_launch(cc_attn_fk<64, KT>, fg, 128, 0, c->stream, fp(q), kvp<KT>(kc), kvp<KT>(vc), fpw(o), fp(gate), *p, G, 0, nullptr, 0, 0);
        else if (p->hd == 128) coli_launch(cc_attn_fk<128, KT>, fg, 128, 0, c->stream, fp(q), kvp<KT>(kc), kvp<KT>(vc), fpw(o), fp(gate), *p, G, 0, nullptr, 0, 0);
        else coli_launch(cc_attn_fk<256, KT>, fg, 128, 0, c->stream, fp(q), kvp<KT>(kc), kvp<KT>(vc), fpw(o), fp(gate), *p, G, 0, nullptr, 0, 0);
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
            coli_launch(cc_attn_k<KT>, dim3((unsigned)p->H, (unsigned)k), 128, 0, c->stream, fp(q), kvp<KT>(kc), kvp<KT>(vc), fpw(o), fp(gate), nullptr, a, 0, nullptr, 0, 0);
            if (!launched(c, "attn")) return 0;
            CcAttn b = *p; b.S -= k; b.pos_base += k; b.q_off += k * p->q_row; b.o_off += k * p->o_row; b.g_off += k * p->g_row;
            return cc_attn_t<KT>(q, kc, vc, o, gate, sel, &b);
        }
        /* a KV head's query heads a block where cc_attn_decg_k takes them (offsets of whole float4s):
         * its rule counts four blocks a KV head */
        int gq = p->k_off % 4 == 0 && p->v_off % 4 == 0 ? cc_decg_heads(G, p->hd) : 0, hc = gq ? 4 * p->KVH : p->H, nb = 1;
        for (int r = 0; r < p->S; r++) {
            int n = p->pos_base + r + 1, L = cc_attn_dec_len(n, hc), m = (n + L - 1) / L;
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
            if (gq) cc_decg_launch(c, gq, fp(q), kvp<KT>(kc), kvp<KT>(vc), c->adec, *p, nb, 0, 0, hc);
            else coli_launch(cc_attn_dec_k<KT>, dim3((unsigned)p->H, (unsigned)nb, (unsigned)p->S), 256, 0, c->stream, 
                fp(q), kvp<KT>(kc), kvp<KT>(vc), fpw(o), fp(gate), c->adec, *p, 0, nb, 0, nullptr, 0, 0, 0);
            coli_launch(cc_attn_dec_k<KT>, dim3((unsigned)p->H, (unsigned)p->S), 256, 0, c->stream, 
                fp(q), kvp<KT>(kc), kvp<KT>(vc), fpw(o), fp(gate), c->adec, *p, 1, nb, 0, nullptr, 0, 0, hc);
            return launched(c, "attn (positions in blocks)");
        }
    }
    dim3 grid((unsigned)p->H, (unsigned)p->S);
    coli_launch(cc_attn_k<KT>, grid, 128, 0, c->stream, fp(q), kvp<KT>(kc), kvp<KT>(vc), fpw(o), fp(gate), sel ? (const int *)sel->d : nullptr, *p, 0, nullptr, 0, 0);
    return launched(c, "attn");
}
static int cc_attn(CcBuf *q, CcBuf *kc, CcBuf *vc, CcBuf *o, CcBuf *gate, CcBuf *sel, const CcAttn *p) {
    return cc_attn_t<float>(q, kc, vc, o, gate, sel, p);
}
static int cc_dnconv(CcBuf *in, CcBuf *w, CcBuf *ring, CcBuf *out, CcBuf *snap, const CcDnConv *p) {
    CcCtx *c = live();
    if (!c || !in || !w || !ring || !out || !p || p->S < 1 || p->CD < 1 || p->CK < 2 || p->CK > 9) return 0;
    if (p->snap_row >= 0 && !snap) return 0;
    coli_launch(cc_dnconv_k, grid1((size_t)p->CD, 64), 64, 0, c->stream, fp(in), fp(w), fpw(ring), fpw(out), fpw(snap), *p);
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
        case 4:   coli_launch(cc_dnrec_rows_k<4, 16>, (unsigned)p->VH, 128, 0, c->stream, a, b, zz, s, pr, yy, sn, *p); return launched(c, "dnrec rows");
        case 8:   coli_launch(cc_dnrec_rows_k<8, 16>, (unsigned)p->VH, 128, 0, c->stream, a, b, zz, s, pr, yy, sn, *p); return launched(c, "dnrec rows");
        case 16:  coli_launch(cc_dnrec_rows_k<16, 16>, (unsigned)p->VH, 128, 0, c->stream, a, b, zz, s, pr, yy, sn, *p); return launched(c, "dnrec rows");
        case 32:  coli_launch(cc_dnrec_rows_k<32, 16>, (unsigned)p->VH, 128, 0, c->stream, a, b, zz, s, pr, yy, sn, *p); return launched(c, "dnrec rows");
        case 64:  coli_launch(cc_dnrec_rows_k<64, 16>, (unsigned)p->VH, 128, 0, c->stream, a, b, zz, s, pr, yy, sn, *p); return launched(c, "dnrec rows");
        case 128: coli_launch(cc_dnrec_rows_k<128, 16>, (unsigned)p->VH, 128, 0, c->stream, a, b, zz, s, pr, yy, sn, *p); return launched(c, "dnrec rows");
        case 256: coli_launch(cc_dnrec_rows_k<256, 8>, (unsigned)p->VH, 128, 0, c->stream, a, b, zz, s, pr, yy, sn, *p); return launched(c, "dnrec rows");
        default: return 0;
        }
    }
    switch (KD) {
    case 4:   coli_launch(cc_dnrec_k<4>, (unsigned)p->VH, 128, 0, c->stream, a, b, zz, s, pr, yy, sn, *p); break;
    case 8:   coli_launch(cc_dnrec_k<8>, (unsigned)p->VH, 128, 0, c->stream, a, b, zz, s, pr, yy, sn, *p); break;
    case 16:  coli_launch(cc_dnrec_k<16>, (unsigned)p->VH, 128, 0, c->stream, a, b, zz, s, pr, yy, sn, *p); break;
    case 32:  coli_launch(cc_dnrec_k<32>, (unsigned)p->VH, 128, 0, c->stream, a, b, zz, s, pr, yy, sn, *p); break;
    case 64:  coli_launch(cc_dnrec_k<64>, (unsigned)p->VH, 128, 0, c->stream, a, b, zz, s, pr, yy, sn, *p); break;
    case 128: coli_launch(cc_dnrec_k<128>, (unsigned)p->VH, 128, 0, c->stream, a, b, zz, s, pr, yy, sn, *p); break;
    case 256: coli_launch(cc_dnrec_k<256>, (unsigned)p->VH, 128, 0, c->stream, a, b, zz, s, pr, yy, sn, *p); break;
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
    coli_launch(cc_ew_k, grid1((size_t)p->n, 256), 256, 0, c->stream, fpw(y), fp(a), fp(b), fp(cb), fp(e), *p);
    return launched(c, "ew");
}
static int cc_qsa(CcBuf *src, CcBuf *w, CcBuf *pk, CcBuf *cs, CcBuf *sc, CcBuf *sel, const CcQsa *p) {
    CcCtx *c = live();
    if (!c || !src || !pk || !p || p->ID < 1 || p->ID > 256 || p->R < 1) return 0;
    if (p->mode == 0) {
        if (!w || !cs || p->nb < 1 || 2 * p->half_ > p->ID) return 0;
        coli_launch(cc_qsa_k, (unsigned)p->nb, 128, 0, c->stream, fp(src), fp(w), fpw(pk), fp(cs), nullptr, nullptr, *p);
    } else if (p->mode == 1) {
        if (!sc || !sel || p->S < 1 || p->IQ < 1 || p->nbmax < 1 || p->sel_row < 1) return 0;
        coli_launch(cc_qsa_k, (unsigned)p->S, 128, 0, c->stream, fp(src), nullptr, fpw(pk), nullptr, fpw(sc), (int *)sel->d, *p);
    } else return 0;
    return launched(c, "qsa");
}
static int cc_ple(CcBuf *keys, CcBuf *hyp, CcBuf *val, CcBuf *prm, CcBuf *gated, CcBuf *normv, CcBuf *conv, CcBuf *ring, const CcPle *p) {
    CcCtx *c = live();
    if (!c || !hyp || !gated || !normv || !p || p->S < 1 || p->C < 1 || p->H < 1) return 0;
    if (p->mode == 0) {
        if (!keys || !val || !prm) return 0;
        coli_launch(cc_ple_k, (unsigned)(p->S * p->C), 256, 0, c->stream, fp(keys), fpw(hyp), fp(val), fp(prm), fpw(gated), fpw(normv), nullptr, nullptr, *p);
    } else if (p->mode == 1) {
        if (!conv || !ring || p->CK < 2 || (p->CK - 1) * p->NG > 32) return 0;
        coli_launch(cc_ple_k, grid1((size_t)p->C * p->H, 256), 256, 0, c->stream, nullptr, fpw(hyp), nullptr, nullptr, fpw(gated), fpw(normv), fp(conv), fpw(ring), *p);
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
    coli_launch(cc_route_k, (unsigned)p->S, 256, 0, c->stream, fp(l), fp(bias), (int *)ids->d, fpw(w), *p);
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
template <typename KT>
static int cc_attn_w2_t(CcBuf *q, CcBuf *kc, CcBuf *vc, CcBuf *o, CcBuf *gate, CcBuf *snk, const CcAttnW *w, int ring) {
    CcCtx *c = live();
    if (!w) return 0;
    const CcAttn *p = &w->a;
    if (!c || !q || !kc || !vc || !o || p->S < 1 || p->H < 1 || p->KVH < 1 || p->H % p->KVH || p->hd < 1 || p->hd > 256 ||
        p->sel_row > 0 || w->win < 0 || (w->sink && !snk) || (p->has_gate && !gate) || ring < 0 ||
        (ring && (ring != p->cap || w->win < 1 || ring < w->win + p->S - 1))) return 0;   /* a ring holds the step's rows and their windows */
    if (!w->win && !w->sink) return cc_attn_t<KT>(q, kc, vc, o, gate, nullptr, p);
    const float *sk = w->sink ? fp(snk) : nullptr;
    const int win = w->win, so = w->sink_off;
    static int fl = -1, dec = -1;
    if (fl < 0) { const char *e = getenv("COLI_CUDA_CHAIN_FLASH"); fl = e && *e ? atoi(e) : 16; if (fl < 0) fl = 0; }
    if (dec < 0) { const char *e = getenv("COLI_CUDA_ATTN_DEC"); dec = !(e && *e == '0'); }
    int G = p->H / p->KVH, fk = 0;
#if !defined(__HIP__) && !defined(__HIP_PLATFORM_AMD__)
    fk = fl > 0 && c->major >= 7 && 16 % G == 0 && (p->hd == 64 || p->hd == 128 || p->hd == 256);
    if (fl > 0 && p->S >= fl && cc_attn_f2_go<KT>(c, q, kc, vc, o, gate, p, G, win, sk, so, ring))
        return launched(c, "attn (tensor cores, one pass, window, sinks)");
    if (fk && p->S >= fl) {
        dim3 fg((unsigned)((p->S + 16 / G - 1) / (16 / G)), (unsigned)p->KVH);
        if (p->hd == 64) coli_launch(cc_attn_fk<64, KT>, fg, 128, 0, c->stream, fp(q), kvp<KT>(kc), kvp<KT>(vc), fpw(o), fp(gate), *p, G, win, sk, so, ring);
        else if (p->hd == 128) coli_launch(cc_attn_fk<128, KT>, fg, 128, 0, c->stream, fp(q), kvp<KT>(kc), kvp<KT>(vc), fpw(o), fp(gate), *p, G, win, sk, so, ring);
        else coli_launch(cc_attn_fk<256, KT>, fg, 128, 0, c->stream, fp(q), kvp<KT>(kc), kvp<KT>(vc), fpw(o), fp(gate), *p, G, win, sk, so, ring);
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
            coli_launch(cc_attn_k<KT>, dim3((unsigned)p->H, (unsigned)k), 128, 0, c->stream, fp(q), kvp<KT>(kc), kvp<KT>(vc), fpw(o), fp(gate), nullptr, a,
                                                                                win, sk, so, ring);
            if (!launched(c, "attn (window, sinks)")) return 0;
            CcAttnW b = *w; b.a.S -= k; b.a.pos_base += k; b.a.q_off += k * p->q_row; b.a.o_off += k * p->o_row; b.a.g_off += k * p->g_row;
            return cc_attn_w2_t<KT>(q, kc, vc, o, gate, snk, &b, ring);
        }
        int gq = p->k_off % 4 == 0 && p->v_off % 4 == 0 ? cc_decg_heads(G, p->hd) : 0, hc = gq ? 4 * p->KVH : p->H, nb = 1;   /* as in cc_attn */
        for (int r = 0; r < p->S; r++) {
            int n = CC_WPOS(r), L = cc_attn_dec_len(n, hc), m = (n + L - 1) / L;
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
            if (gq) cc_decg_launch(c, gq, fp(q), kvp<KT>(kc), kvp<KT>(vc), c->adec, *p, nb, win, ring, hc);
            else coli_launch(cc_attn_dec_k<KT>, dim3((unsigned)p->H, (unsigned)nb, (unsigned)p->S), 256, 0, c->stream, 
                fp(q), kvp<KT>(kc), kvp<KT>(vc), fpw(o), fp(gate), c->adec, *p, 0, nb, win, sk, so, ring, 0);
            coli_launch(cc_attn_dec_k<KT>, dim3((unsigned)p->H, (unsigned)p->S), 256, 0, c->stream, 
                fp(q), kvp<KT>(kc), kvp<KT>(vc), fpw(o), fp(gate), c->adec, *p, 1, nb, win, sk, so, ring, hc);
            return launched(c, "attn (positions in blocks, window, sinks)");
        }
    }
    #undef CC_WPOS
    coli_launch(cc_attn_k<KT>, dim3((unsigned)p->H, (unsigned)p->S), 128, 0, c->stream, fp(q), kvp<KT>(kc), kvp<KT>(vc), fpw(o), fp(gate), nullptr, *p,
                                                                         win, sk, so, ring);
    return launched(c, "attn (window, sinks)");
}
static int cc_attn_w2(CcBuf *q, CcBuf *kc, CcBuf *vc, CcBuf *o, CcBuf *gate, CcBuf *snk, const CcAttnW *w, int ring) {
    return cc_attn_w2_t<float>(q, kc, vc, o, gate, snk, w, ring);
}
static int cc_attn_w(CcBuf *q, CcBuf *kc, CcBuf *vc, CcBuf *o, CcBuf *gate, CcBuf *snk, const CcAttnW *w) {
    return cc_attn_w2(q, kc, vc, o, gate, snk, w, 0);
}

static int cc_muse(void) { return 1; }   /* the norm's PRESCALE, ew's SIGMUL and RESGATE */
static int cc_kv16(void) { return 1; }   /* ew's F16 */
static int cc_pick(CcBuf *logits, CcBuf *out, int n) {
    CcCtx *c = live();
    if (!c || !logits || !out || n < 1 || (size_t)n * 4 > logits->bytes || out->bytes < 8) return 0;
    coli_launch(cc_pick_k, 1, 256, 0, c->stream, fp(logits), out->d, n);
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
            coli_launch(cc_matmul_rows_i8_k<1>, grid, 256, smem, c->stream, fpw(y), fp(x), (const int8_t *)w, sc, (const int *)ids->d, I, O, n);
        else
            coli_launch(cc_matmul_rows_i8_k<0>, grid, 256, (size_t)I * sizeof(float), c->stream, fpw(y), fp(x), (const int8_t *)w, sc, (const int *)ids->d, I, O, n);
    } else {
        coli_launch(cc_matmul_rows_k, grid, 256, 0, c->stream, fpw(y), fp(x), w, sc, (const int *)ids->d, fmt, I, O, n);
    }
    return launched(c, "matmul_rows");
}
/* copy_regions_h: float32 regions of src (offsets and lengths in floats) into a float16 cache
 * (in halves), each value rounded to the nearest float16: a step's K and V rows into the
 * mirrors of a float16 KV cache, whose values rope_f16 has already rounded */
#if !defined(__HIP__) && !defined(__HIP_PLATFORM_AMD__)
__global__ static void cc_copy_grid_h_k(__half *__restrict__ d, const float *__restrict__ s, CcGrid g) {
    COLI_PDL_WAIT();
    long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= (long long)g.n * g.len) return;
    long long i = gid / g.len, j = gid - i * g.len, q = i / g.per, h = i - q * g.per;
    d[g.d0 + q * g.dq + h * g.dh + j] = __float2half_rn(s[g.s0 + q * g.sq + h * g.sh + j]);
}
static int cc_copy_regions_h(CcBuf *dst, CcBuf *src, const CcRegion *r, int n) {
    CcCtx *c = live();
    if (!c || !dst || !src || !dst->d || !src->d || dst == src || (n > 0 && !r)) return 0;
    for (int i = 0; i < n; i++)
        if ((r[i].dst + r[i].n) * 2 > dst->bytes || (r[i].src + r[i].n) * 4 > src->bytes) return 0;
    __half *d = reinterpret_cast<__half *>(dst->d);
    for (int i = 0; i < n; ) {
        int k = cc_grid_run(r + i, n - i);
        CcGrid g;
        if (!(k >= 2 && cc_grid_of(r + i, k, &g))) { k = 1; g = CcGrid{(long long)r[i].dst, (long long)r[i].src, 0, 0, 0, 0, (int)r[i].n, 1, 1}; }
        long long tot = (long long)g.n * g.len;
        if (tot > 0) {
            coli_launch(cc_copy_grid_h_k, grid1((size_t)tot, 256), 256, 0, c->stream, d, src->d, g);
            if (!launched(c, "copy regions (float16)")) return 0;
        }
        i += k;
    }
    g_st.ops++;
    return 1;
}
/* attn_h, attn_wh: attn and attn_wr over a float16 cache (its offsets in halves) */
static int cc_attn_h(CcBuf *q, CcBuf *kc, CcBuf *vc, CcBuf *o, CcBuf *gate, CcBuf *sel, const CcAttn *p) {
    return cc_attn_t<__half>(q, kc, vc, o, gate, sel, p);
}
static int cc_attn_wh(CcBuf *q, CcBuf *kc, CcBuf *vc, CcBuf *o, CcBuf *gate, CcBuf *snk, const CcAttnW *w, int ring) {
    return cc_attn_w2_t<__half>(q, kc, vc, o, gate, snk, w, ring);
}
static int cc_kvh(void) { return 1; }
#else
static int cc_copy_regions_h(CcBuf *, CcBuf *, const CcRegion *, int) { return 0; }
static int cc_attn_h(CcBuf *, CcBuf *, CcBuf *, CcBuf *, CcBuf *, CcBuf *, const CcAttn *) { return 0; }
static int cc_attn_wh(CcBuf *, CcBuf *, CcBuf *, CcBuf *, CcBuf *, CcBuf *, const CcAttnW *, int) { return 0; }
static int cc_kvh(void) { return 0; }
#endif
/* ---- the decision engines' encoders: shaders/chain_enc.comp ------------------------
 * Its six modes as kernels, the shader's arithmetic but for two choices that keep a row
 * nearer the engines' own: the LayerNorm's mean and variance are summed in double, as
 * laya.c and gliner_decide.c sum them, and the GELU is erff's (the shader fits its own
 * erfc, GLSL having none). */
__device__ static inline float cc_enc_act(float x, int k) {
    if (k == 1) return 0.5f * x * (1.0f + erff(x * 0.70710678118654752f));
    if (k == 2) return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
    if (k == 3) return fmaxf(x, 0.0f);
    return x;
}
__device__ static inline double cc_enc_sum256(double v, double *red, int tid) {
    red[tid] = v;
    __syncthreads();
    for (int off = 128; off > 0; off >>= 1) {
        if (tid < off) red[tid] += red[tid + off];
        __syncthreads();
    }
    double s = red[0];
    __syncthreads();
    return s;
}
/* mode 0: a block a row, its values in registers before any write (y may be a or b) */
__global__ static void cc_enc_norm_k(float *a, const float *b, const float *prm, float *y, CcEncNorm p) {
    COLI_PDL_WAIT();
    __shared__ double red[256];
    int r = blockIdx.x, tid = threadIdx.x;
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
    float rstd = (float)(1.0 / sqrt(var + (double)p.eps));
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
__global__ static void cc_enc_bias_k(float *y, const float *prm, CcEncBias p) {
    COLI_PDL_WAIT();
    long long i = (long long)blockIdx.x * 256 + threadIdx.x;
    if (i >= p.n) return;
    int r = (int)(i / p.O), o = (int)(i - (long long)r * p.O);
    size_t at = (size_t)p.y_off + (size_t)r * p.y_row + o;
    float v = y[at];
    if (p.b_off >= 0) v = v + prm[p.b_off + o];
    y[at] = cc_enc_act(v, p.act);
}
/* mode 2 */
__global__ static void cc_enc_geglu_k(const float *a, const float *prm, float *y, CcEncGeglu p) {
    COLI_PDL_WAIT();
    long long i = (long long)blockIdx.x * 256 + threadIdx.x;
    if (i >= p.n) return;
    int r = (int)(i / p.I), j = (int)(i - (long long)r * p.I);
    size_t ab = (size_t)p.a_off + (size_t)r * p.a_row;
    float u = a[ab + j], g = a[ab + p.I + j];
    if (p.b_off >= 0) { u = u + prm[p.b_off + j]; g = g + prm[p.b_off + p.I + j]; }
    y[(size_t)p.y_off + (size_t)r * p.y_row + j] = cc_enc_act(u, p.act) * g;
}
/* mode 3: a thread a (row, query or key, head, pair) */
__global__ static void cc_enc_rope_k(float *x, const float *tab, const int *pos, CcEncRope p) {
    COLI_PDL_WAIT();
    long long i = (long long)blockIdx.x * 256 + threadIdx.x;
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
__global__ static void cc_enc_addrow_k(float *y, const float *tab, const int *idx, CcEncAddRow p) {
    COLI_PDL_WAIT();
    long long i = (long long)blockIdx.x * 256 + threadIdx.x;
    if (i >= p.n) return;
    int r = (int)(i / p.D), d = (int)(i - (long long)r * p.D);
    y[(size_t)p.y_off + (size_t)r * p.y_row + d] += tab[(size_t)p.t_off + (size_t)idx[r] * p.D + d];
}
/* mode 5: a block a (head, 16 rows), the union of their ranges in tiles of 32 rows, the
 * softmax online (a running max and sum a row) */
#define CC_EBQ 16
#define CC_EBK 32
#define CC_EHD 128
__global__ static void __launch_bounds__(256) cc_enc_attn_k(const float *qkv, const float *c2p, const float *p2c, float *o,
                                                            const int *rng, const int *ridx, const int *pos, CcEncAttn p) {
    COLI_PDL_WAIT();
    __shared__ float qs[CC_EBQ * CC_EHD];
    __shared__ float kv[CC_EBK * (CC_EHD + 1)];
    __shared__ float sc[CC_EBQ * CC_EBK];
    __shared__ float mrow[CC_EBQ], lrow[CC_EBQ], crow[CC_EBQ];
    __shared__ int rlo[CC_EBQ], rhi[CC_EBQ], rpos[CC_EBQ];
    __shared__ int klo, khi;
    const int S = p.S, hd = p.hd, HDP = CC_EHD + 1, rel = p.flags & (CC_ENC_C2P | CC_ENC_P2C);
    int h = blockIdx.x, b0 = blockIdx.y * CC_EBQ, tid = threadIdx.x;
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
    __syncthreads();
    if (tid == 0) {
        int lo = 0x7fffffff, hi = -1;
        for (int i = 0; i < nrow; i++) if (rlo[i] <= rhi[i]) { lo = min(lo, rlo[i]); hi = max(hi, rhi[i]); }
        klo = lo; khi = hi;
    }
    __syncthreads();
    int pi = tid >> 4, pd = tid & 15;   /* the PV step: row pi, dims pd + 16k */
    float acc[8];
    for (int k = 0; k < 8; k++) acc[k] = 0.0f;
    for (int j0 = klo; j0 <= khi; j0 += CC_EBK) {
        int nk = khi - j0 + 1 < CC_EBK ? khi - j0 + 1 : CC_EBK;
        for (int e = tid; e < nk * hd; e += 256) {
            int jj = e / hd, d = e - jj * hd;
            kv[jj * HDP + d] = qkv[(size_t)p.k_off + (size_t)(j0 + jj) * p.kv_row + (size_t)h * hd + d];
        }
        __syncthreads();
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
        __syncthreads();
        if (tid < CC_EBQ) {
            int i = tid;
            float m = mrow[i], mn = m;
            int jlo = rlo[i] - j0 > 0 ? rlo[i] - j0 : 0, jhi = rhi[i] - j0 < nk - 1 ? rhi[i] - j0 : nk - 1;
            for (int jj = jlo; jj <= jhi; jj++) mn = fmaxf(mn, sc[i * CC_EBK + jj]);
            float l = 0.0f;
            for (int jj = 0; jj < CC_EBK; jj++) {
                float e = jj >= jlo && jj <= jhi ? expf(sc[i * CC_EBK + jj] - mn) : 0.0f;
                sc[i * CC_EBK + jj] = e;
                l += e;
            }
            float cr = expf(m - mn);
            crow[i] = cr;
            lrow[i] = lrow[i] * cr + l;
            mrow[i] = mn;
        }
        __syncthreads();
        for (int e = tid; e < nk * hd; e += 256) {
            int jj = e / hd, d = e - jj * hd;
            kv[jj * HDP + d] = qkv[(size_t)p.v_off + (size_t)(j0 + jj) * p.kv_row + (size_t)h * hd + d];
        }
        __syncthreads();
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
        __syncthreads();
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
__global__ static void __launch_bounds__(256) cc_enc_rel_k(const float *x, const float *pt, float *y, CcEncRel p) {
    COLI_PDL_WAIT();
    __shared__ float xs[CC_EBQ * (CC_EDC + 1)];
    __shared__ float ps[CC_EBR * (CC_EDC + 1)];
    int t0 = blockIdx.x * CC_EBR, i0 = blockIdx.y * CC_EBQ, h = blockIdx.z;
    int tid = threadIdx.x, tt = tid & 63, iq = tid >> 6;   /* bucket tt, rows 4iq .. 4iq+3 */
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
        __syncthreads();
        for (int d = 0; d < dn; d++) {
            float pv = ps[tt * (CC_EDC + 1) + d];
            for (int k = 0; k < 4; k++) acc[k] += xs[(4 * iq + k) * (CC_EDC + 1) + d] * pv;
        }
        __syncthreads();
    }
    if (t0 + tt < p.nr)
        for (int k = 0; k < 4; k++) {
            int i = i0 + 4 * iq + k;
            if (i < p.S) y[(size_t)p.y_off + ((size_t)h * p.S + i) * p.nr + t0 + tt] = acc[k];
        }
}
static int cc_enc_norm(CcBuf *a, CcBuf *b, CcBuf *prm, CcBuf *y, const CcEncNorm *p) {
    CcCtx *c = live();
    if (!c || !a || !prm || !y || !p || p->rows < 1 || p->D < 1 || p->D > 4096 || ((p->flags & CC_ENC_ADD) && !b)) return 0;
    coli_launch(cc_enc_norm_k, (unsigned)p->rows, 256, 0, c->stream, fpw(a), fp(b), fp(prm), fpw(y), *p);
    return launched(c, "encoder norm");
}
static int cc_enc_bias(CcBuf *y, CcBuf *prm, const CcEncBias *p) {
    CcCtx *c = live();
    if (!c || !y || !p || p->n < 1 || p->O < 1 || (p->b_off >= 0 && !prm)) return 0;
    coli_launch(cc_enc_bias_k, grid1((size_t)p->n, 256), 256, 0, c->stream, fpw(y), fp(prm), *p);
    return launched(c, "encoder bias");
}
static int cc_enc_geglu(CcBuf *a, CcBuf *prm, CcBuf *y, const CcEncGeglu *p) {
    CcCtx *c = live();
    if (!c || !a || !y || !p || p->n < 1 || p->I < 1 || (p->b_off >= 0 && !prm)) return 0;
    coli_launch(cc_enc_geglu_k, grid1((size_t)p->n, 256), 256, 0, c->stream, fp(a), fp(prm), fpw(y), *p);
    return launched(c, "encoder geglu");
}
static int cc_enc_rope(CcBuf *x, CcBuf *tab, CcBuf *pos, const CcEncRope *p) {
    CcCtx *c = live();
    if (!c || !x || !tab || !pos || !p || p->n < 1 || p->H < 1 || p->hd < 2) return 0;
    coli_launch(cc_enc_rope_k, grid1((size_t)p->n, 256), 256, 0, c->stream, fpw(x), fp(tab), (const int *)pos->d, *p);
    return launched(c, "encoder rope");
}
static int cc_enc_addrow(CcBuf *y, CcBuf *tab, CcBuf *idx, const CcEncAddRow *p) {
    CcCtx *c = live();
    if (!c || !y || !tab || !idx || !p || p->n < 1 || p->D < 1) return 0;
    coli_launch(cc_enc_addrow_k, grid1((size_t)p->n, 256), 256, 0, c->stream, fpw(y), fp(tab), (const int *)idx->d, *p);
    return launched(c, "encoder add row");
}
static int cc_enc_attn(CcBuf *qkv, CcBuf *c2p, CcBuf *p2c, CcBuf *o, CcBuf *rng, CcBuf *ridx, CcBuf *pos, const CcEncAttn *p) {
    CcCtx *c = live();
    int rel = p ? p->flags & (CC_ENC_C2P | CC_ENC_P2C) : 0;
    if (!c || !qkv || !o || !rng || !p || p->S < 1 || p->H < 1 || p->hd < 1 || p->hd > CC_EHD || (p->S + CC_EBQ - 1) / CC_EBQ > 65535 ||
        (rel && (!ridx || !pos)) || ((p->flags & CC_ENC_C2P) && !c2p) || ((p->flags & CC_ENC_P2C) && !p2c)) return 0;
    dim3 g((unsigned)p->H, (unsigned)((p->S + CC_EBQ - 1) / CC_EBQ));
    coli_launch(cc_enc_attn_k, g, 256, 0, c->stream, fp(qkv), fp(c2p), fp(p2c), fpw(o), (const int *)rng->d,
                                           ridx ? (const int *)ridx->d : nullptr, pos ? (const int *)pos->d : nullptr, *p);
    return launched(c, "encoder attention");
}
static int cc_enc_rel(CcBuf *x, CcBuf *pt, CcBuf *y, const CcEncRel *p) {
    CcCtx *c = live();
    if (!c || !x || !pt || !y || !p || p->S < 1 || p->H < 1 || p->hd < 1 || p->nr < 1 || p->H > 65535 ||
        (p->S + CC_EBQ - 1) / CC_EBQ > 65535) return 0;
    dim3 g((unsigned)((p->nr + CC_EBR - 1) / CC_EBR), (unsigned)((p->S + CC_EBQ - 1) / CC_EBQ), (unsigned)p->H);
    coli_launch(cc_enc_rel_k, g, 256, 0, c->stream, fp(x), fp(pt), fpw(y), *p);
    return launched(c, "encoder relative products");
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
    cc_kvh,
    cc_attn_h,
    cc_attn_wh,
    cc_copy_regions_h,
    cc_enc_norm, cc_enc_bias, cc_enc_geglu, cc_enc_rope, cc_enc_addrow, cc_enc_attn, cc_enc_rel,
};
extern "C" const ColiCudaChainOps *coli_cuda_chain_ops(void) { return &g_cc_ops; }
