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

struct CcBuf { int kind, dev; size_t bytes; float *d; void *h; };

namespace {
struct CcCtx {
    int device, up, lost, in_frame;
    cudaStream_t stream;
    cudaEvent_t ev; int ev_ok, ev_pending;
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
    for (int i = tid; i < p.D; i += 256) { float v = x[xb + i]; acc += v * v; }
    red[tid] = acc;
    __syncthreads();
    for (int off = 128; off > 0; off >>= 1) {
        if (tid < off) red[tid] += red[tid + off];
        __syncthreads();
    }
    float r = (p.flags & 4) ? 1.f / sqrtf(red[0] + p.eps) : 1.f / sqrtf(red[0] / (float)p.D + p.eps);
    for (int i = tid; i < p.D; i += 256) {
        float wv = (p.flags & 2) ? 1.f : ((p.flags & 1) ? 1.f + w[wb + i] : w[wb + i]);
        y[yb + i] = x[xb + i] * r * wv * p.post;
    }
}
__global__ static void cc_rope_k(float *__restrict__ x, const float *__restrict__ cs, CcRope p) {
    long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= (long long)p.nseg * p.half_) return;
    int seg = (int)(gid / p.half_), j = (int)(gid - (long long)seg * p.half_);
    int row = seg / p.per_row, hh = seg - row * p.per_row;
    int base = p.x_off + row * p.x_row + hh * p.x_seg;
    int cb = p.cs_off + row * p.cs_row + 2 * j;
    float c = cs[cb], s = cs[cb + 1];
    float a = x[base + j], b = x[base + j + p.half_];
    x[base + j] = a * c - b * s;
    x[base + j + p.half_] = b * c + a * s;
}
/* one block of 128 per (head, row); K and V rows read once per tile of 128 positions;
 * hd <= 256: two dims per thread */
__global__ static void __launch_bounds__(128) cc_attn_k(const float *__restrict__ q, const float *__restrict__ kc,
        const float *__restrict__ vc, float *__restrict__ o, const float *__restrict__ gt, const int *__restrict__ sel, CcAttn p) {
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
    for (int t0 = 0; t0 < n; t0 += 128) {
        int i = t0 + tid;
        float sv = -3.0e38f;
        if (i < n) {
            int t = list ? sel[lb + 1 + i] : i;
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
    int kh = h / (p.VH / p.KH);
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
        float decay = expf(-expf(alog) * cc_softplus(av + dtb));
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
__device__ static float cc_sig(float x) { return 1.f / (1.f + expf(-x)); }
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
        }
        (void)cudaGetLastError();
        c->up = 0; c->ev_ok = 0;
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
        if (cudaMalloc((void **)&b->d, bytes) != cudaSuccess || cudaMemset(b->d, 0, bytes) != cudaSuccess) {
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
    return ok;
}
static int cc_finish(void) {
    CcCtx *c = live();
    if (!c) return 0;
    double t0 = cc_now_ms();
    int ok = cc_ok(c, cudaStreamSynchronize(c->stream), "finish");
    g_st.waits++; g_st.wait_ms += cc_now_ms() - t0;
    c->ev_pending = 0;
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
static int cc_copy_regions(CcBuf *dst, CcBuf *src, const CcRegion *r, int n) {
    CcCtx *c = live();
    if (!c || !dst || !src || (n > 0 && !r)) return 0;
    for (int i = 0; i < n; i++) {
        if ((r[i].dst + r[i].n) * 4 > dst->bytes || (r[i].src + r[i].n) * 4 > src->bytes) return 0;
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
static int cc_attn(CcBuf *q, CcBuf *kc, CcBuf *vc, CcBuf *o, CcBuf *gate, CcBuf *sel, const CcAttn *p) {
    CcCtx *c = live();
    if (!c || !q || !kc || !vc || !o || !p || p->S < 1 || p->H < 1 || p->KVH < 1 || p->H % p->KVH || p->hd < 1 || p->hd > 256) return 0;
    if (p->has_gate && !gate) return 0;
    if (p->sel_row > 0 && !sel) return 0;
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
    switch (KD) {
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
    case CC_EW_HC_LOW: case CC_EW_HC_INJ: case CC_EW_SCALE: if (!a) return 0; break;
    case CC_EW_HC_MIX: case CC_EW_HC_APPLY: if (!a || !b || p->D < 1 || p->C < 1) return 0; break;
    case CC_EW_GATE_ADD: if (!a || !b || !e || p->D < 1) return 0; break;
    default: return 0;
    }
    cc_ew_k<<<grid1((size_t)p->n, 256), 256, 0, c->stream>>>(fpw(y), fp(a), fp(b), fp(cb), fp(e), *p);
    return launched(c, "ew");
}
static void cc_stats(CcStats *st) { if (st) *st = g_st; }

static const ColiCudaChainOps g_cc_ops = {
    sizeof(ColiCudaChainOps),
    cc_init, cc_ready, cc_lost_, cc_shutdown, cc_device, cc_device_now,
    cc_buf, cc_free, cc_reserve, cc_ptr, cc_bytes,
    cc_begin, cc_submit, cc_finish,
    cc_copy, cc_zero, cc_copy_regions, cc_write, cc_read,
    cc_matmul, cc_norm, cc_rope, cc_attn, cc_dnconv, cc_dnrec, cc_ew,
    cc_stats,
};
extern "C" const ColiCudaChainOps *coli_cuda_chain_ops(void) { return &g_cc_ops; }
