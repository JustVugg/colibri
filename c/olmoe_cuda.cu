/* olmoe_cuda.cu -- GPU-resident decode step for the OLMoE engine (olmoe.c).
 *
 * Built as its own library (coli_cuda_olmoe.dll on Windows, linked directly on Linux), the same split
 * backend_cuda_dsv4.cu uses: the ABI is olmoe_cuda.h. Without the library or without a GPU the engine
 * keeps running on the CPU; nothing here is reachable from the default build.
 *
 * Scope: batch-1 decode (S == 1). Prefill stays on the CPU path; the engine uploads the prompt's KV
 * rows afterwards. Everything is FP32 activations x (FP16 dense | int8 per-row-scaled expert) weights,
 * the same operations and weight values as the CPU path (no requantisation). The GPU's float summation
 * order differs from the CPU's, so results are not bit-identical to the CPU; the engine's verify mode
 * (COLI_OLMOE_CUDA_VERIFY=1) measures per-token logit divergence and argmax agreement against the CPU.
 *
 * Why these kernels (measured, not guessed): a cached CPU decode step streams ~2.3 GB of resident
 * weights per token at the DRAM rate; a consumer GPU has ~10x that bandwidth. Decode is bandwidth-bound,
 * so every kernel is a warp-per-row GEMV with 16-byte vector loads and the activation vector staged in
 * shared memory.
 *
 * Expert weights live in VRAM "slots" (blob = gate|up|down int8, scales = gs|us|ds float). A device table
 * maps (layer, expert) -> slot (-1 = not resident). The router kernel resolves slots on the device; when
 * every expert is resident the host issues a whole token with no synchronisation. Otherwise the host
 * fetches the routed ids after the router, uploads the misses, and passes explicit slots.
 */
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "olmoe_cuda.h"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "[olmoe_cuda] %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); return -1; } } while (0)

#define MAXK 16          /* max routed experts per token */
#define GEMV_WARPS 8     /* rows per block */

struct LayerDev {
    __half *q, *k, *v, *o;           /* [D][D] FP16 */
    float *in_ln, *post_ln, *qn, *kn;/* [D] */
    float *gate;                     /* [E][D] FP32 */
    float *K, *V;                    /* [H][max_t][hd] */
};

struct G {
    int ready = 0, dev = 0;
    int L = 0, D = 0, H = 0, hd = 0, I = 0, E = 0, K = 0, V = 0, max_t = 0, n_slots = 0, norm_topk = 0;
    float eps = 1e-5f, theta = 10000.f;
    LayerDev *layers = nullptr;
    __half *lm_head = nullptr; float *final_norm = nullptr;
    int8_t *slot_w = nullptr; float *slot_s = nullptr;   /* [n_slots][3*I*D] / [n_slots][2*I+D] */
    int *slot_of = nullptr;                              /* [L*E] */
    size_t w_bytes = 0, s_elems = 0;
    /* scratch */
    float *x, *xn, *q, *k, *v, *ctx, *logits_e, *logits;
    float *act;                                          /* [MAXK][I] */
    int *sel_slots, *sel_ids; float *sel_w;              /* device routing result */
    int *miss;                                           /* device flag: a routed expert was not resident */
    float *kv_stage;                                     /* [L][2][D] rows appended this token */
    int pos = 0;
    cudaStream_t st = nullptr;
    /* pinned host */
    int *h_sel_ids; float *h_sel_w; int *h_miss; float *h_logits; float *h_kv_stage;
    int8_t *h_w_stage;
};
static G g;

/* ------------------------------ kernels ------------------------------ */

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    return v;
}

/* y[row] (+)= dot(W[row,:], x), W FP16 [O][I], I % 256 == 0 */
__global__ void k_gemv_f16(const __half *__restrict__ W, const float *__restrict__ x, float *__restrict__ y,
                           int I, int O, int add) {
    extern __shared__ float xs[];
    for (int i = threadIdx.x; i < I; i += blockDim.x) xs[i] = x[i];
    __syncthreads();
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int row = blockIdx.x * GEMV_WARPS + warp;
    if (row >= O) return;
    const uint4 *wp = reinterpret_cast<const uint4 *>(W + (size_t)row * I);
    float acc = 0.f;
    int n = I >> 3;
    #pragma unroll 4
    for (int j = lane; j < n; j += 32) {
        uint4 v = __ldg(wp + j);
        const __half2 *h = reinterpret_cast<const __half2 *>(&v);
        const float *xj = xs + (j << 3);
        float2 a = __half22float2(h[0]), b = __half22float2(h[1]), c = __half22float2(h[2]), d = __half22float2(h[3]);
        acc += a.x * xj[0] + a.y * xj[1] + b.x * xj[2] + b.y * xj[3]
             + c.x * xj[4] + c.y * xj[5] + d.x * xj[6] + d.y * xj[7];
    }
    acc = warp_sum(acc);
    if (lane == 0) y[row] = add ? y[row] + acc : acc;
}

/* router logits: W FP32 [O][I] */
__global__ void k_gemv_f32(const float *__restrict__ W, const float *__restrict__ x, float *__restrict__ y, int I, int O) {
    extern __shared__ float xs[];
    for (int i = threadIdx.x; i < I; i += blockDim.x) xs[i] = x[i];
    __syncthreads();
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int row = blockIdx.x * GEMV_WARPS + warp;
    if (row >= O) return;
    const float4 *wp = reinterpret_cast<const float4 *>(W + (size_t)row * I);
    float acc = 0.f;
    for (int j = lane; j < (I >> 2); j += 32) {
        float4 w = __ldg(wp + j);
        const float *xj = xs + (j << 2);
        acc += w.x * xj[0] + w.y * xj[1] + w.z * xj[2] + w.w * xj[3];
    }
    acc = warp_sum(acc);
    if (lane == 0) y[row] = acc;
}

/* out = x * r * w, r = 1/sqrt(mean(x^2)+eps); double accumulation like the CPU reference.
 * Two rows (blockIdx.y) so q and k norms run in one launch: in/out/w are arrays of 2. */
__global__ void k_rmsnorm2(float *out0, const float *x0, const float *w0,
                           float *out1, const float *x1, const float *w1, int D, float eps) {
    const float *x = blockIdx.y ? x1 : x0; const float *w = blockIdx.y ? w1 : w0; float *out = blockIdx.y ? out1 : out0;
    __shared__ double sm[32];
    double a = 0;
    for (int i = threadIdx.x; i < D; i += blockDim.x) a += (double)x[i] * x[i];
    for (int o = 16; o > 0; o >>= 1) a += __shfl_down_sync(0xffffffffu, a, o);
    if ((threadIdx.x & 31) == 0) sm[threadIdx.x >> 5] = a;
    __syncthreads();
    if (threadIdx.x == 0) {
        double t = 0; for (int i = 0; i < (blockDim.x >> 5); i++) t += sm[i];
        sm[0] = t;
    }
    __syncthreads();
    float r = 1.f / sqrtf((float)(sm[0] / D) + eps);
    for (int i = threadIdx.x; i < D; i += blockDim.x) out[i] = x[i] * r * w[i];
}

/* rotate-half RoPE on q and k (per head), then append k, v to the layer KV cache at pos and
 * stage the new rows for the host mirror. One thread per (head, j). */
__global__ void k_rope_append(float *q, float *k, const float *v, float *Kc, float *Vc, float *stage,
                              int H, int hd, int max_t, int pos, float theta, int D) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int half = hd >> 1;
    if (t >= H * half) return;
    int h = t / half, j = t % half;
    float inv = powf(theta, -2.0f * j / hd);
    float ang = pos * inv, cs = cosf(ang), sn = sinf(ang);
    float *qh = q + h * hd, *kh = k + h * hd;
    float a = qh[j], b = qh[j + half];
    qh[j] = a * cs - b * sn; qh[j + half] = b * cs + a * sn;
    a = kh[j]; b = kh[j + half];
    float ka = a * cs - b * sn, kb = b * cs + a * sn;
    kh[j] = ka; kh[j + half] = kb;
    size_t base = ((size_t)h * max_t + pos) * hd;
    Kc[base + j] = ka; Kc[base + j + half] = kb;
    Vc[base + j] = v[h * hd + j]; Vc[base + j + half] = v[h * hd + j + half];
    stage[h * hd + j] = ka; stage[h * hd + j + half] = kb;
    stage[D + h * hd + j] = v[h * hd + j]; stage[D + h * hd + j + half] = v[h * hd + j + half];
}

/* causal MHA decode, one block per head, T = pos + 1. Scores in dynamic shared memory (T floats). */
__global__ void k_attn(const float *q, const float *Kc, const float *Vc, float *ctx, int hd, int max_t, int T, float scale) {
    extern __shared__ float sc[];
    __shared__ float red[32];
    int h = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
    const float *qh = q + h * hd;
    const float *Kh = Kc + (size_t)h * max_t * hd, *Vh = Vc + (size_t)h * max_t * hd;
    float q4[4];                      /* hd == 128: 4 floats per lane */
    for (int i = 0; i < 4; i++) q4[i] = qh[lane * 4 + i];
    for (int t = warp; t < T; t += nw) {
        float4 kv = *reinterpret_cast<const float4 *>(Kh + (size_t)t * hd + lane * 4);
        float d = kv.x * q4[0] + kv.y * q4[1] + kv.z * q4[2] + kv.w * q4[3];
        d = warp_sum(d);
        if (lane == 0) sc[t] = d * scale;
    }
    __syncthreads();
    float m = -1e30f;
    for (int t = threadIdx.x; t < T; t += blockDim.x) m = fmaxf(m, sc[t]);
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_down_sync(0xffffffffu, m, o));
    if (lane == 0) red[warp] = m;
    __syncthreads();
    if (threadIdx.x == 0) { float mm = red[0]; for (int i = 1; i < nw; i++) mm = fmaxf(mm, red[i]); red[0] = mm; }
    __syncthreads();
    m = red[0];
    __syncthreads();
    float s = 0.f;
    for (int t = threadIdx.x; t < T; t += blockDim.x) { float e = expf(sc[t] - m); sc[t] = e; s += e; }
    s = warp_sum(s);
    if (lane == 0) red[warp] = s;
    __syncthreads();
    if (threadIdx.x == 0) { float ss = 0; for (int i = 0; i < nw; i++) ss += red[i]; red[0] = ss; }
    __syncthreads();
    s = red[0];
    for (int t = threadIdx.x; t < T; t += blockDim.x) sc[t] /= s;
    __syncthreads();
    if (threadIdx.x < hd) {
        float acc = 0.f;
        for (int t = 0; t < T; t++) acc += sc[t] * Vh[(size_t)t * hd + threadIdx.x];
        ctx[h * hd + threadIdx.x] = acc;
    }
}

/* softmax over E logits + top-K (first max wins, strict '>', lowest id on ties: moe_router.h),
 * then map experts to VRAM slots. One block; K sequential argmax passes by thread 0 (E is small). */
__global__ void k_route(const float *logits, const int *slot_of_layer, float *sel_w, int *sel_ids, int *sel_slots,
                        int *miss, int E, int K, int norm_topk) {
    __shared__ float p[1024];
    for (int e = threadIdx.x; e < E; e += blockDim.x) p[e] = logits[e];
    __syncthreads();
    if (threadIdx.x == 0) {
        float m = -1e30f; for (int e = 0; e < E; e++) if (p[e] > m) m = p[e];
        float s = 0.f; for (int e = 0; e < E; e++) { p[e] = expf(p[e] - m); s += p[e]; }
        for (int e = 0; e < E; e++) p[e] /= s;
        int taken[MAXK]; float val[MAXK]; int any_miss = 0;
        for (int k = 0; k < K; k++) {
            int best = -1; float bv = -1e30f;
            for (int e = 0; e < E; e++) {
                int t = 0; for (int j = 0; j < k; j++) if (taken[j] == e) { t = 1; break; }
                if (!t && p[e] > bv) { bv = p[e]; best = e; }
            }
            if (best < 0) { best = 0; while (true) { int t = 0; for (int j = 0; j < k; j++) if (taken[j] == best) t = 1; if (!t) break; best++; } bv = 0.f; }
            taken[k] = best; val[k] = bv;
        }
        if (norm_topk) { float sm = 0; for (int k = 0; k < K; k++) sm += val[k]; for (int k = 0; k < K; k++) val[k] /= sm; }
        for (int k = 0; k < K; k++) {
            int sl = slot_of_layer[taken[k]];
            sel_ids[k] = taken[k]; sel_w[k] = val[k]; sel_slots[k] = sl;
            if (sl < 0) any_miss = 1;
        }
        *miss = any_miss;
    }
}

/* per-expert int8 dot on a row: 16 int8 per lane-load */
__device__ __forceinline__ float dot_i8_row(const int8_t *__restrict__ w, const float *xs, int I, int lane) {
    const uint4 *wp = reinterpret_cast<const uint4 *>(w);
    float acc = 0.f;
    int n = I >> 4;
    #pragma unroll 2
    for (int j = lane; j < n; j += 32) {
        uint4 v = __ldg(wp + j);
        const int8_t *b = reinterpret_cast<const int8_t *>(&v);
        const float *xj = xs + (j << 4);
        #pragma unroll
        for (int i = 0; i < 16; i++) acc += (float)b[i] * xj[i];
    }
    return warp_sum(acc);
}

/* act[k][o] = silu(gate_k[o] * gs) * (up_k[o] * us) for the K routed experts. grid = (I/8, K). */
__global__ void k_expert_gateup(const int8_t *slot_w, const float *slot_s, const int *sel_slots, const float *x,
                                float *act, int D, int I, size_t w_bytes, size_t s_elems) {
    extern __shared__ float xs[];
    int k = blockIdx.y; int sl = sel_slots[k];
    for (int i = threadIdx.x; i < D; i += blockDim.x) xs[i] = x[i];
    __syncthreads();
    if (sl < 0) return;
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int row = blockIdx.x * GEMV_WARPS + warp;
    if (row >= I) return;
    const int8_t *w = slot_w + (size_t)sl * w_bytes;
    const float *s = slot_s + (size_t)sl * s_elems;
    float g = dot_i8_row(w + (size_t)row * D, xs, D, lane);
    float u = dot_i8_row(w + (size_t)I * D + (size_t)row * D, xs, D, lane);
    if (lane == 0) {
        g *= s[row]; u *= s[I + row];
        act[k * I + row] = (g / (1.f + expf(-g))) * u;
    }
}

/* x[h] += sum_k w_k * (ds[h] * dot(down_k[h,:], act[k])), summed in fixed k order (as the CPU loop). */
__global__ void k_expert_down(const int8_t *slot_w, const float *slot_s, const int *sel_slots, const float *sel_w,
                              const float *act, float *x, int D, int I, int K, size_t w_bytes, size_t s_elems) {
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int row = blockIdx.x * GEMV_WARPS + warp;
    if (row >= D) return;
    float out = 0.f;
    for (int k = 0; k < K; k++) {
        int sl = sel_slots[k];
        if (sl < 0) continue;
        const int8_t *w = slot_w + (size_t)sl * w_bytes + (size_t)2 * I * D + (size_t)row * I;
        const uint4 *wp = reinterpret_cast<const uint4 *>(w);
        const float *a = act + k * I;
        float acc = 0.f;
        for (int j = lane; j < (I >> 4); j += 32) {
            uint4 v = __ldg(wp + j);
            const int8_t *b = reinterpret_cast<const int8_t *>(&v);
            #pragma unroll
            for (int i = 0; i < 16; i++) acc += (float)b[i] * a[(j << 4) + i];
        }
        acc = warp_sum(acc);
        acc = __shfl_sync(0xffffffffu, acc, 0);
        out += sel_w[k] * (acc * slot_s[(size_t)sl * s_elems + 2 * I + row]);
    }
    if (lane == 0) x[row] += out;
}


/* ------------------------------ host API ------------------------------ */


#define UP(dst, src, bytes) do { CK(cudaMalloc(&(dst), (bytes))); CK(cudaMemcpy((dst), (src), (bytes), cudaMemcpyHostToDevice)); } while (0)

OLMOE_CUDA_API int olmoe_cuda_available(void) { int n = 0; return cudaGetDeviceCount(&n) == cudaSuccess && n > 0; }

OLMOE_CUDA_API int olmoe_cuda_init(int device, int layers, int hidden, int heads, int head_dim, int inter, int nexp, int topk,
                   int vocab, int max_t, float eps, float theta, int norm_topk) {
    if (g.ready) return 1;
    if (head_dim != 128 || topk > MAXK || nexp > 1024 || hidden % 256 || inter % 16) return -2;
    CK(cudaSetDevice(device));
    g.dev = device; g.L = layers; g.D = hidden; g.H = heads; g.hd = head_dim; g.I = inter; g.E = nexp; g.K = topk;
    g.V = vocab; g.max_t = max_t; g.eps = eps; g.theta = theta; g.norm_topk = norm_topk;
    g.w_bytes = (size_t)3 * inter * hidden; g.s_elems = (size_t)2 * inter + hidden;
    g.layers = (LayerDev *)calloc(layers, sizeof(LayerDev));
    CK(cudaStreamCreateWithFlags(&g.st, cudaStreamNonBlocking));
    size_t kvb = (size_t)heads * max_t * head_dim * sizeof(float);
    for (int l = 0; l < layers; l++) { CK(cudaMalloc(&g.layers[l].K, kvb)); CK(cudaMalloc(&g.layers[l].V, kvb)); }
    CK(cudaMalloc(&g.x, hidden * 4)); CK(cudaMalloc(&g.xn, hidden * 4)); CK(cudaMalloc(&g.q, hidden * 4));
    CK(cudaMalloc(&g.k, hidden * 4)); CK(cudaMalloc(&g.v, hidden * 4)); CK(cudaMalloc(&g.ctx, hidden * 4));
    CK(cudaMalloc(&g.logits_e, nexp * 4)); CK(cudaMalloc(&g.logits, (size_t)vocab * 4));
    CK(cudaMalloc(&g.act, (size_t)MAXK * inter * 4));
    CK(cudaMalloc(&g.sel_slots, MAXK * 4)); CK(cudaMalloc(&g.sel_ids, MAXK * 4)); CK(cudaMalloc(&g.sel_w, MAXK * 4));
    CK(cudaMalloc(&g.miss, 4));
    CK(cudaMalloc(&g.kv_stage, (size_t)layers * 2 * hidden * 4));
    CK(cudaMalloc(&g.slot_of, (size_t)layers * nexp * 4));
    { int *neg = (int *)malloc((size_t)layers * nexp * 4); for (int i = 0; i < layers * nexp; i++) neg[i] = -1;
      CK(cudaMemcpy(g.slot_of, neg, (size_t)layers * nexp * 4, cudaMemcpyHostToDevice)); free(neg); }
    CK(cudaHostAlloc((void **)&g.h_sel_ids, MAXK * 4, 0)); CK(cudaHostAlloc((void **)&g.h_sel_w, MAXK * 4, 0));
    CK(cudaHostAlloc((void **)&g.h_miss, 4, 0));
    CK(cudaHostAlloc((void **)&g.h_logits, (size_t)vocab * 4, 0));
    CK(cudaHostAlloc((void **)&g.h_kv_stage, (size_t)layers * 2 * hidden * 4, 0));
    CK(cudaHostAlloc((void **)&g.h_w_stage, g.w_bytes + g.s_elems * 4, 0));
    g.ready = 1;
    return 1;
}

OLMOE_CUDA_API int olmoe_cuda_set_layer(int l, const void *q, const void *k, const void *v, const void *o,
                        const float *in_ln, const float *post_ln, const float *qn, const float *kn, const float *gate) {
    LayerDev *L = &g.layers[l]; size_t dd = (size_t)g.D * g.D * 2, d4 = (size_t)g.D * 4;
    UP(L->q, q, dd); UP(L->k, k, dd); UP(L->v, v, dd); UP(L->o, o, dd);
    UP(L->in_ln, in_ln, d4); UP(L->post_ln, post_ln, d4); UP(L->qn, qn, d4); UP(L->kn, kn, d4);
    UP(L->gate, gate, (size_t)g.E * g.D * 4);
    return 1;
}

OLMOE_CUDA_API int olmoe_cuda_set_head(const void *lm_head, const float *final_norm) {
    UP(g.lm_head, lm_head, (size_t)g.V * g.D * 2);
    UP(g.final_norm, final_norm, (size_t)g.D * 4);
    return 1;
}

OLMOE_CUDA_API void olmoe_cuda_mem_info(size_t *free_b, size_t *total_b) { cudaMemGetInfo(free_b, total_b); }

/* allocate n expert slots; returns how many were actually allocated (stops when VRAM runs out) */
OLMOE_CUDA_API int olmoe_cuda_alloc_slots(int n) {
    g.n_slots = 0;
    if (cudaMalloc(&g.slot_w, g.w_bytes * n) != cudaSuccess) { cudaGetLastError(); g.slot_w = nullptr; }
    if (!g.slot_w) {
        /* fall back to the largest n that fits, in steps of 16 slots */
        while (n > 16 && !g.slot_w) { n -= 16; if (cudaMalloc(&g.slot_w, g.w_bytes * n) != cudaSuccess) { cudaGetLastError(); g.slot_w = nullptr; } }
        if (!g.slot_w) return 0;
    }
    if (cudaMalloc(&g.slot_s, g.s_elems * 4 * n) != cudaSuccess) { cudaGetLastError(); cudaFree(g.slot_w); g.slot_w = nullptr; return 0; }
    g.n_slots = n;
    return n;
}

/* copy one expert (merged int8 blob + scales) into a VRAM slot and publish it in the device table */
OLMOE_CUDA_API int olmoe_cuda_upload_expert(int layer, int eid, int slot, const int8_t *w, const float *s) {
    if (slot < 0 || slot >= g.n_slots) return -3;
    CK(cudaMemcpy(g.slot_w + (size_t)slot * g.w_bytes, w, g.w_bytes, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(g.slot_s + (size_t)slot * g.s_elems, s, g.s_elems * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(g.slot_of + (size_t)layer * g.E + eid, &slot, 4, cudaMemcpyHostToDevice));
    return 1;
}

/* evict: mark (layer, eid) non-resident */
OLMOE_CUDA_API int olmoe_cuda_unmap_expert(int layer, int eid) {
    int neg = -1;
    CK(cudaMemcpy(g.slot_of + (size_t)layer * g.E + eid, &neg, 4, cudaMemcpyHostToDevice));
    return 1;
}

OLMOE_CUDA_API void *olmoe_cuda_stage_ptr(void) { return g.h_w_stage; }   /* pinned buffer for w (w_bytes) then s (s_elems floats) */

/* upload rows [from,to) of a host KV cache laid out [H][host_max_t][hd] */
OLMOE_CUDA_API int olmoe_cuda_kv_upload(int layer, const float *K, const float *V, int host_max_t, int from, int to) {
    if (to > g.max_t) return -4;
    size_t rowb = (size_t)(to - from) * g.hd * 4;
    for (int h = 0; h < g.H; h++) {
        CK(cudaMemcpy(g.layers[layer].K + ((size_t)h * g.max_t + from) * g.hd, K + ((size_t)h * host_max_t + from) * g.hd, rowb, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(g.layers[layer].V + ((size_t)h * g.max_t + from) * g.hd, V + ((size_t)h * host_max_t + from) * g.hd, rowb, cudaMemcpyHostToDevice));
    }
    return 1;
}

OLMOE_CUDA_API int olmoe_cuda_begin_token(const float *x_embed, int pos) {
    if (pos >= g.max_t) return -4;
    g.pos = pos;
    CK(cudaMemcpyAsync(g.x, x_embed, (size_t)g.D * 4, cudaMemcpyHostToDevice, g.st));
    return 1;
}

OLMOE_CUDA_API int olmoe_cuda_layer_attn(int l) {
    LayerDev *L = &g.layers[l]; int D = g.D;
    size_t sm = (size_t)D * 4; int gb = (D + GEMV_WARPS - 1) / GEMV_WARPS;
    k_rmsnorm2<<<dim3(1, 1), 256, 0, g.st>>>(g.xn, g.x, L->in_ln, g.xn, g.x, L->in_ln, D, g.eps);
    k_gemv_f16<<<gb, 32 * GEMV_WARPS, sm, g.st>>>(L->q, g.xn, g.q, D, D, 0);
    k_gemv_f16<<<gb, 32 * GEMV_WARPS, sm, g.st>>>(L->k, g.xn, g.k, D, D, 0);
    k_gemv_f16<<<gb, 32 * GEMV_WARPS, sm, g.st>>>(L->v, g.xn, g.v, D, D, 0);
    k_rmsnorm2<<<dim3(1, 2), 256, 0, g.st>>>(g.q, g.q, L->qn, g.k, g.k, L->kn, D, g.eps);
    k_rope_append<<<(g.H * (g.hd / 2) + 127) / 128, 128, 0, g.st>>>(g.q, g.k, g.v, L->K, L->V,
                                                                     g.kv_stage + (size_t)l * 2 * D, g.H, g.hd, g.max_t, g.pos, g.theta, D);
    int T = g.pos + 1;
    k_attn<<<g.H, 256, (size_t)T * 4, g.st>>>(g.q, L->K, L->V, g.ctx, g.hd, g.max_t, T, 1.f / sqrtf((float)g.hd));
    k_gemv_f16<<<gb, 32 * GEMV_WARPS, sm, g.st>>>(L->o, g.ctx, g.x, D, D, 1);
    CK(cudaGetLastError());
    return 1;
}

OLMOE_CUDA_API int olmoe_cuda_layer_route(int l) {
    LayerDev *L = &g.layers[l]; int D = g.D;
    k_rmsnorm2<<<dim3(1, 1), 256, 0, g.st>>>(g.xn, g.x, L->post_ln, g.xn, g.x, L->post_ln, D, g.eps);
    k_gemv_f32<<<(g.E + GEMV_WARPS - 1) / GEMV_WARPS, 32 * GEMV_WARPS, (size_t)D * 4, g.st>>>(L->gate, g.xn, g.logits_e, D, g.E);
    k_route<<<1, 64, 0, g.st>>>(g.logits_e, g.slot_of + (size_t)l * g.E, g.sel_w, g.sel_ids, g.sel_slots, g.miss, g.E, g.K, g.norm_topk);
    CK(cudaGetLastError());
    return 1;
}

/* host needs the routed experts (only when not every expert is resident) */
OLMOE_CUDA_API int olmoe_cuda_route_fetch(int *ids, float *w, int *miss) {
    CK(cudaMemcpyAsync(g.h_sel_ids, g.sel_ids, g.K * 4, cudaMemcpyDeviceToHost, g.st));
    CK(cudaMemcpyAsync(g.h_sel_w, g.sel_w, g.K * 4, cudaMemcpyDeviceToHost, g.st));
    CK(cudaMemcpyAsync(g.h_miss, g.miss, 4, cudaMemcpyDeviceToHost, g.st));
    CK(cudaStreamSynchronize(g.st));
    memcpy(ids, g.h_sel_ids, g.K * 4); memcpy(w, g.h_sel_w, g.K * 4); *miss = *g.h_miss;
    return 1;
}

/* explicit slots (after miss handling); NULL = use the device-resolved slots */
OLMOE_CUDA_API int olmoe_cuda_layer_experts(const int *slots) {
    int D = g.D, I = g.I;
    if (slots) CK(cudaMemcpyAsync(g.sel_slots, slots, g.K * 4, cudaMemcpyHostToDevice, g.st));
    k_expert_gateup<<<dim3((I + GEMV_WARPS - 1) / GEMV_WARPS, g.K), 32 * GEMV_WARPS, (size_t)D * 4, g.st>>>(
        g.slot_w, g.slot_s, g.sel_slots, g.xn, g.act, D, I, g.w_bytes, g.s_elems);
    k_expert_down<<<(D + GEMV_WARPS - 1) / GEMV_WARPS, 32 * GEMV_WARPS, 0, g.st>>>(
        g.slot_w, g.slot_s, g.sel_slots, g.sel_w, g.act, g.x, D, I, g.K, g.w_bytes, g.s_elems);
    CK(cudaGetLastError());
    return 1;
}

/* final norm + lm_head; copies logits and this token's appended KV rows back to the host */
OLMOE_CUDA_API int olmoe_cuda_finish(float *logits_out, float *kv_stage_out) {
    int D = g.D;
    k_rmsnorm2<<<dim3(1, 1), 256, 0, g.st>>>(g.xn, g.x, g.final_norm, g.xn, g.x, g.final_norm, D, g.eps);
    k_gemv_f16<<<(g.V + GEMV_WARPS - 1) / GEMV_WARPS, 32 * GEMV_WARPS, (size_t)D * 4, g.st>>>(g.lm_head, g.xn, g.logits, D, g.V, 0);
    CK(cudaMemcpyAsync(g.h_logits, g.logits, (size_t)g.V * 4, cudaMemcpyDeviceToHost, g.st));
    CK(cudaMemcpyAsync(g.h_kv_stage, g.kv_stage, (size_t)g.L * 2 * D * 4, cudaMemcpyDeviceToHost, g.st));
    CK(cudaStreamSynchronize(g.st));
    memcpy(logits_out, g.h_logits, (size_t)g.V * 4);
    memcpy(kv_stage_out, g.h_kv_stage, (size_t)g.L * 2 * D * 4);
    return 1;
}

OLMOE_CUDA_API void olmoe_cuda_shutdown(void) { if (g.ready) { cudaDeviceSynchronize(); cudaDeviceReset(); g.ready = 0; } }
