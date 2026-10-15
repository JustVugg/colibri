/* olmoe_cuda.cu -- GPU-resident decode and prefill step for the OLMoE engine (olmoe.c).
 *
 * Built as an opt-in standalone library (coli_olmoe_cuda.dll on Windows, libcoli_olmoe_cuda.so
 * on Linux) that the engine loads at runtime via LoadLibrary/dlopen, exactly like coli_cuda.dll:
 * no CUDA symbols are linked into olmoe, and if the library or an NVIDIA GPU is absent or if
 * anything fails, the engine keeps running on the CPU path with zero disruption.
 *
 * Scope: batch-1 decode (S == 1) and batched prefill (S > 1, chunks of up to 128 tokens).
 * Everything runs as FP32 activations x (FP16 dense | int8 per-row-scaled expert) weights,
 * matching the CPU path operations and weight scales without requantization.
 * GPU floating-point reduction order differs from the CPU's, so output is not bit-identical;
 * the engine's COLI_GPU_VERIFY mode measures per-token logit divergence and argmax agreement
 * against the CPU reference.
 *
 * Kernel design:
 *  - Cached CPU decode streams ~2.3 GB of resident weights per token at ~20-30 GB/s DRAM;
 *    a consumer GPU (e.g. RTX 4060 Laptop) has ~200+ GB/s VRAM bandwidth. Because decode is
 *    memory bandwidth-bound, dense and expert kernels are warp-per-row GEMVs with 16-byte vector
 *    loads and shared-memory activation staging.
 *  - Expert weights reside in VRAM "slots" (int8 gate|up|down blob + float scales). A device
 *    residency table maps (layer, expert) -> slot. The on-device router resolves slots directly;
 *    when all experts are resident, forward passes run entirely on GPU with zero host-device sync.
 *    When VRAM is smaller than the model, the host checks routed IDs, uploads misses into LRU
 *    slots on demand, and continues execution seamlessly.
 */
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#define OC_API extern "C" __declspec(dllexport)
#else
#define OC_API extern "C" __attribute__((visibility("default")))
#endif

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
    int *h_sel_ids; float *h_sel_w; int *h_miss; float *h_logits; float *h_kv_stage; void *h_stage;
    int8_t *h_w_stage;
};
static G g;

/* ---- optional per-phase timing (COLI_GPU_PROF=1 on the engine side) --------------------------------
 * Event pairs bracket each phase; they are resolved after the token's stream sync, so enabling this only
 * costs the event records. Phases: dense GEMVs (q/k/v/o), attention core, router, expert gate+up,
 * expert down, final norm + lm_head. */
enum { P_DENSE, P_ATTN, P_ROUTE, P_GATEUP, P_DOWN, P_HEAD, P_N };
static int prof_on = 0;
static cudaEvent_t prof_ev[512][2]; static int prof_cat[512], prof_n = 0, prof_made = 0;
static double prof_ms[P_N]; static long prof_tokens = 0;
static void prof_begin(int cat) {
    if (!prof_on || prof_n >= 512) return;
    if (!prof_made) { for (int i = 0; i < 512; i++) { cudaEventCreate(&prof_ev[i][0]); cudaEventCreate(&prof_ev[i][1]); } prof_made = 1; }
    prof_cat[prof_n] = cat; cudaEventRecord(prof_ev[prof_n][0], g.st);
}
static void prof_end(void) { if (!prof_on || prof_n >= 512) return; cudaEventRecord(prof_ev[prof_n][1], g.st); prof_n++; }
static void prof_flush(void) {
    if (!prof_on) return;
    for (int i = 0; i < prof_n; i++) { float ms = 0; cudaEventElapsedTime(&ms, prof_ev[i][0], prof_ev[i][1]); prof_ms[prof_cat[i]] += ms; }
    prof_n = 0; prof_tokens++;
}

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

OC_API int oc_init(int device, int layers, int hidden, int heads, int head_dim, int inter, int nexp, int topk,
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
    CK(cudaMalloc(&g.logits_e, nexp * 4)); CK(cudaMalloc(&g.logits, (size_t)vocab * 4 * 8));   /* up to 8 rows for batched verification */
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

OC_API int oc_set_layer(int l, const void *q, const void *k, const void *v, const void *o,
                        const float *in_ln, const float *post_ln, const float *qn, const float *kn, const float *gate) {
    LayerDev *L = &g.layers[l]; size_t dd = (size_t)g.D * g.D * 2, d4 = (size_t)g.D * 4;
    UP(L->q, q, dd); UP(L->k, k, dd); UP(L->v, v, dd); UP(L->o, o, dd);
    UP(L->in_ln, in_ln, d4); UP(L->post_ln, post_ln, d4); UP(L->qn, qn, d4); UP(L->kn, kn, d4);
    UP(L->gate, gate, (size_t)g.E * g.D * 4);
    return 1;
}

OC_API int oc_set_head(const void *lm_head, const float *final_norm) {
    UP(g.lm_head, lm_head, (size_t)g.V * g.D * 2);
    UP(g.final_norm, final_norm, (size_t)g.D * 4);
    return 1;
}

OC_API void oc_mem_info(size_t *free_b, size_t *total_b) { cudaMemGetInfo(free_b, total_b); }

/* allocate n expert slots; returns how many were actually allocated (stops when VRAM runs out) */
OC_API int oc_alloc_slots(int n) {
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
OC_API int oc_upload_expert(int layer, int eid, int slot, const int8_t *w, const float *s) {
    if (slot < 0 || slot >= g.n_slots) return -3;
    CK(cudaMemcpy(g.slot_w + (size_t)slot * g.w_bytes, w, g.w_bytes, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(g.slot_s + (size_t)slot * g.s_elems, s, g.s_elems * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(g.slot_of + (size_t)layer * g.E + eid, &slot, 4, cudaMemcpyHostToDevice));
    return 1;
}

/* evict: mark (layer, eid) non-resident */
OC_API int oc_unmap_expert(int layer, int eid) {
    int neg = -1;
    CK(cudaMemcpy(g.slot_of + (size_t)layer * g.E + eid, &neg, 4, cudaMemcpyHostToDevice));
    return 1;
}

OC_API void *oc_stage_ptr(void) { return g.h_w_stage; }   /* pinned buffer for w (w_bytes) then s (s_elems floats) */

/* upload rows [from,to) of a host KV cache laid out [H][host_max_t][hd] */
OC_API int oc_kv_upload(int layer, const float *K, const float *V, int host_max_t, int from, int to) {
    if (to > g.max_t) return -4;
    size_t rowb = (size_t)(to - from) * g.hd * 4;
    for (int h = 0; h < g.H; h++) {
        CK(cudaMemcpy(g.layers[layer].K + ((size_t)h * g.max_t + from) * g.hd, K + ((size_t)h * host_max_t + from) * g.hd, rowb, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(g.layers[layer].V + ((size_t)h * g.max_t + from) * g.hd, V + ((size_t)h * host_max_t + from) * g.hd, rowb, cudaMemcpyHostToDevice));
    }
    return 1;
}

OC_API int oc_begin_token(const float *x_embed, int pos) {
    if (pos >= g.max_t) return -4;
    g.pos = pos;
    CK(cudaMemcpyAsync(g.x, x_embed, (size_t)g.D * 4, cudaMemcpyHostToDevice, g.st));
    return 1;
}

OC_API int oc_layer_attn(int l) {
    LayerDev *L = &g.layers[l]; int D = g.D;
    size_t sm = (size_t)D * 4; int gb = (D + GEMV_WARPS - 1) / GEMV_WARPS;
    k_rmsnorm2<<<dim3(1, 1), 256, 0, g.st>>>(g.xn, g.x, L->in_ln, g.xn, g.x, L->in_ln, D, g.eps);
    prof_begin(P_DENSE);
    k_gemv_f16<<<gb, 32 * GEMV_WARPS, sm, g.st>>>(L->q, g.xn, g.q, D, D, 0);
    k_gemv_f16<<<gb, 32 * GEMV_WARPS, sm, g.st>>>(L->k, g.xn, g.k, D, D, 0);
    k_gemv_f16<<<gb, 32 * GEMV_WARPS, sm, g.st>>>(L->v, g.xn, g.v, D, D, 0);
    prof_end();
    prof_begin(P_ATTN);
    k_rmsnorm2<<<dim3(1, 2), 256, 0, g.st>>>(g.q, g.q, L->qn, g.k, g.k, L->kn, D, g.eps);
    k_rope_append<<<(g.H * (g.hd / 2) + 127) / 128, 128, 0, g.st>>>(g.q, g.k, g.v, L->K, L->V,
                                                                     g.kv_stage + (size_t)l * 2 * D, g.H, g.hd, g.max_t, g.pos, g.theta, D);
    int T = g.pos + 1;
    k_attn<<<g.H, 256, (size_t)T * 4, g.st>>>(g.q, L->K, L->V, g.ctx, g.hd, g.max_t, T, 1.f / sqrtf((float)g.hd));
    prof_end();
    prof_begin(P_DENSE);
    k_gemv_f16<<<gb, 32 * GEMV_WARPS, sm, g.st>>>(L->o, g.ctx, g.x, D, D, 1);
    prof_end();
    CK(cudaGetLastError());
    return 1;
}

OC_API int oc_layer_route(int l) {
    LayerDev *L = &g.layers[l]; int D = g.D;
    prof_begin(P_ROUTE);
    k_rmsnorm2<<<dim3(1, 1), 256, 0, g.st>>>(g.xn, g.x, L->post_ln, g.xn, g.x, L->post_ln, D, g.eps);
    k_gemv_f32<<<(g.E + GEMV_WARPS - 1) / GEMV_WARPS, 32 * GEMV_WARPS, (size_t)D * 4, g.st>>>(L->gate, g.xn, g.logits_e, D, g.E);
    k_route<<<1, 64, 0, g.st>>>(g.logits_e, g.slot_of + (size_t)l * g.E, g.sel_w, g.sel_ids, g.sel_slots, g.miss, g.E, g.K, g.norm_topk);
    prof_end();
    CK(cudaGetLastError());
    return 1;
}

/* host needs the routed experts (only when not every expert is resident) */
OC_API int oc_route_fetch(int *ids, float *w, int *miss) {
    CK(cudaMemcpyAsync(g.h_sel_ids, g.sel_ids, g.K * 4, cudaMemcpyDeviceToHost, g.st));
    CK(cudaMemcpyAsync(g.h_sel_w, g.sel_w, g.K * 4, cudaMemcpyDeviceToHost, g.st));
    CK(cudaMemcpyAsync(g.h_miss, g.miss, 4, cudaMemcpyDeviceToHost, g.st));
    CK(cudaStreamSynchronize(g.st));
    memcpy(ids, g.h_sel_ids, g.K * 4); memcpy(w, g.h_sel_w, g.K * 4); *miss = *g.h_miss;
    return 1;
}

/* explicit slots (after miss handling); NULL = use the device-resolved slots */
OC_API int oc_layer_experts(const int *slots) {
    int D = g.D, I = g.I;
    if (slots) CK(cudaMemcpyAsync(g.sel_slots, slots, g.K * 4, cudaMemcpyHostToDevice, g.st));
    prof_begin(P_GATEUP);
    k_expert_gateup<<<dim3((I + GEMV_WARPS - 1) / GEMV_WARPS, g.K), 32 * GEMV_WARPS, (size_t)D * 4, g.st>>>(
        g.slot_w, g.slot_s, g.sel_slots, g.xn, g.act, D, I, g.w_bytes, g.s_elems);
    prof_end();
    prof_begin(P_DOWN);
    k_expert_down<<<(D + GEMV_WARPS - 1) / GEMV_WARPS, 32 * GEMV_WARPS, 0, g.st>>>(
        g.slot_w, g.slot_s, g.sel_slots, g.sel_w, g.act, g.x, D, I, g.K, g.w_bytes, g.s_elems);
    prof_end();
    CK(cudaGetLastError());
    return 1;
}

/* final norm + lm_head; copies logits and this token's appended KV rows back to the host */
OC_API int oc_finish(float *logits_out, float *kv_stage_out) {
    int D = g.D;
    prof_begin(P_HEAD);
    k_rmsnorm2<<<dim3(1, 1), 256, 0, g.st>>>(g.xn, g.x, g.final_norm, g.xn, g.x, g.final_norm, D, g.eps);
    k_gemv_f16<<<(g.V + GEMV_WARPS - 1) / GEMV_WARPS, 32 * GEMV_WARPS, (size_t)D * 4, g.st>>>(g.lm_head, g.xn, g.logits, D, g.V, 0);
    prof_end();
    CK(cudaMemcpyAsync(g.h_logits, g.logits, (size_t)g.V * 4, cudaMemcpyDeviceToHost, g.st));
    CK(cudaMemcpyAsync(g.h_kv_stage, g.kv_stage, (size_t)g.L * 2 * D * 4, cudaMemcpyDeviceToHost, g.st));
    CK(cudaStreamSynchronize(g.st));
    prof_flush();
    memcpy(logits_out, g.h_logits, (size_t)g.V * 4);
    memcpy(kv_stage_out, g.h_kv_stage, (size_t)g.L * 2 * D * 4);
    return 1;
}


/* ============================ batched path (prefill / speculative verify) ============================
 * S tokens at once (S <= PF_MAX). Dense projections and expert MLPs are "rows" GEMVs: a block owns 8
 * weight rows and up to PF_R token vectors, so every weight row is read once per PF_R tokens and the
 * re-reads by the other token chunks are served from L2. Tokens are grouped per expert so each routed
 * expert's weights stream once per chunk of its tokens. Each (token, k) expert output is written to its own
 * slot of tbuf and summed per token in fixed k order, i.e. the same accumulation order as decode/CPU. */
#define PF_MAX 128
#define PF_R 8
struct PF {
    int S = 0, pos = 0, ok = 0;
    float *x, *xn, *q, *k, *v, *ctx, *logits_e, *w_sel, *act, *tbuf, *kvl;
    int *ids, *cnt, *start, *cursor, *list, *used;
    float *hkv;       /* pinned host [L][2][PF_MAX][D] */
    int *h_used;
    float *h_logits_rows;
};
static PF pf;

template <int R>
__global__ void k_gemv_f16_rows(const __half *__restrict__ W, const float *__restrict__ X, float *__restrict__ Y,
                                int I, int O, int S, int add) {
    extern __shared__ float xs[];
    int s0 = blockIdx.y * R, nr = min(R, S - s0);
    for (int idx = threadIdx.x; idx < nr * I; idx += blockDim.x) xs[idx] = X[(size_t)s0 * I + idx];
    __syncthreads();
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int row = blockIdx.x * GEMV_WARPS + warp;
    if (row >= O) return;
    const uint4 *wp = reinterpret_cast<const uint4 *>(W + (size_t)row * I);
    float acc[R];
    #pragma unroll
    for (int r = 0; r < R; r++) acc[r] = 0.f;
    for (int j = lane; j < (I >> 3); j += 32) {
        uint4 v = __ldg(wp + j);
        const __half2 *h = reinterpret_cast<const __half2 *>(&v);
        float wf[8];
        #pragma unroll
        for (int t = 0; t < 4; t++) { float2 f = __half22float2(h[t]); wf[2 * t] = f.x; wf[2 * t + 1] = f.y; }
        #pragma unroll
        for (int r = 0; r < R; r++) {
            if (r < nr) {
                const float *xj = xs + (size_t)r * I + (j << 3);
                float a = 0.f;
                #pragma unroll
                for (int t = 0; t < 8; t++) a += wf[t] * xj[t];
                acc[r] += a;
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < R; r++) {
        float a = warp_sum(acc[r]);
        if (lane == 0 && r < nr) { size_t o = (size_t)(s0 + r) * O + row; Y[o] = add ? Y[o] + a : a; }
    }
}

__global__ void k_gemv_f32_rows(const float *__restrict__ W, const float *__restrict__ X, float *__restrict__ Y, int I, int O, int S) {
    int sidx = blockIdx.y; extern __shared__ float xs[];
    for (int i = threadIdx.x; i < I; i += blockDim.x) xs[i] = X[(size_t)sidx * I + i];
    __syncthreads();
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int row = blockIdx.x * GEMV_WARPS + warp;
    if (row >= O) return;
    const float4 *wp = reinterpret_cast<const float4 *>(W + (size_t)row * I);
    float acc = 0.f;
    for (int j = lane; j < (I >> 2); j += 32) {
        float4 w = __ldg(wp + j); const float *xj = xs + (j << 2);
        acc += w.x * xj[0] + w.y * xj[1] + w.z * xj[2] + w.w * xj[3];
    }
    acc = warp_sum(acc);
    if (lane == 0) Y[(size_t)sidx * O + row] = acc;
}

/* one block per token row: out = x*r*w */
__global__ void k_rmsnorm_rows(float *out, const float *x, const float *w, int D, float eps) {
    __shared__ double sm[32];
    const float *xr = x + (size_t)blockIdx.x * D; float *o = out + (size_t)blockIdx.x * D;
    double a = 0;
    for (int i = threadIdx.x; i < D; i += blockDim.x) a += (double)xr[i] * xr[i];
    for (int off = 16; off > 0; off >>= 1) a += __shfl_down_sync(0xffffffffu, a, off);
    if ((threadIdx.x & 31) == 0) sm[threadIdx.x >> 5] = a;
    __syncthreads();
    if (threadIdx.x == 0) { double t = 0; for (int i = 0; i < (blockDim.x >> 5); i++) t += sm[i]; sm[0] = t; }
    __syncthreads();
    float r = 1.f / sqrtf((float)(sm[0] / D) + eps);
    for (int i = threadIdx.x; i < D; i += blockDim.x) o[i] = xr[i] * r * w[i];
}

/* q,k rows: RoPE at pos_base+s, then append K,V rows to the layer cache and to the per-layer staging buffer */
__global__ void k_rope_append_rows(float *q, float *k, const float *v, float *Kc, float *Vc, float *stage,
                                   int H, int hd, int max_t, int pos_base, float theta, int D, int S) {
    int s = blockIdx.y; int t = blockIdx.x * blockDim.x + threadIdx.x;
    int half = hd >> 1;
    if (t >= H * half) return;
    int h = t / half, j = t % half, pos = pos_base + s;
    float inv = powf(theta, -2.0f * j / hd);
    float ang = pos * inv, cs = cosf(ang), sn = sinf(ang);
    float *qh = q + (size_t)s * D + h * hd, *kh = k + (size_t)s * D + h * hd;
    float a = qh[j], b = qh[j + half];
    qh[j] = a * cs - b * sn; qh[j + half] = b * cs + a * sn;
    a = kh[j]; b = kh[j + half];
    float ka = a * cs - b * sn, kb = b * cs + a * sn;
    kh[j] = ka; kh[j + half] = kb;
    size_t base = ((size_t)h * max_t + pos) * hd;
    const float *vh = v + (size_t)s * D + h * hd;
    Kc[base + j] = ka; Kc[base + j + half] = kb;
    Vc[base + j] = vh[j]; Vc[base + j + half] = vh[j + half];
    stage[(size_t)s * D + h * hd + j] = ka; stage[(size_t)s * D + h * hd + j + half] = kb;
    stage[(size_t)S * D + (size_t)s * D + h * hd + j] = vh[j]; stage[(size_t)S * D + (size_t)s * D + h * hd + j + half] = vh[j + half];
}

/* causal MHA for every (head, row): T = pos_base + s + 1 */
__global__ void k_attn_rows(const float *q, const float *Kc, const float *Vc, float *ctx, int hd, int max_t, int pos_base, int D, float scale) {
    extern __shared__ float sc[];
    __shared__ float red[32];
    int h = blockIdx.x, s = blockIdx.y, T = pos_base + s + 1;
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
    const float *qh = q + (size_t)s * D + h * hd;
    const float *Kh = Kc + (size_t)h * max_t * hd, *Vh = Vc + (size_t)h * max_t * hd;
    float q4[4];
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
    float sum = 0.f;
    for (int t = threadIdx.x; t < T; t += blockDim.x) { float e = expf(sc[t] - m); sc[t] = e; sum += e; }
    sum = warp_sum(sum);
    if (lane == 0) red[warp] = sum;
    __syncthreads();
    if (threadIdx.x == 0) { float ss = 0; for (int i = 0; i < nw; i++) ss += red[i]; red[0] = ss; }
    __syncthreads();
    sum = red[0];
    for (int t = threadIdx.x; t < T; t += blockDim.x) sc[t] /= sum;
    __syncthreads();
    if (threadIdx.x < hd) {
        float acc = 0.f;
        for (int t = 0; t < T; t++) acc += sc[t] * Vh[(size_t)t * hd + threadIdx.x];
        ctx[(size_t)s * D + h * hd + threadIdx.x] = acc;
    }
}

/* softmax + top-K per token; counts tokens per expert */
__global__ void k_route_rows(const float *logits, float *sel_w, int *sel_ids, int *cnt, int *used, int E, int K, int norm_topk) {
    __shared__ float p[1024];
    int s = blockIdx.x;
    for (int e = threadIdx.x; e < E; e += blockDim.x) p[e] = logits[(size_t)s * E + e];
    __syncthreads();
    if (threadIdx.x == 0) {
        float m = -1e30f; for (int e = 0; e < E; e++) if (p[e] > m) m = p[e];
        float sm = 0.f; for (int e = 0; e < E; e++) { p[e] = expf(p[e] - m); sm += p[e]; }
        for (int e = 0; e < E; e++) p[e] /= sm;
        int taken[MAXK]; float val[MAXK];
        for (int k = 0; k < K; k++) {
            int best = -1; float bv = -1e30f;
            for (int e = 0; e < E; e++) {
                int t = 0; for (int j = 0; j < k; j++) if (taken[j] == e) { t = 1; break; }
                if (!t && p[e] > bv) { bv = p[e]; best = e; }
            }
            if (best < 0) { best = 0; while (true) { int t = 0; for (int j = 0; j < k; j++) if (taken[j] == best) t = 1; if (!t) break; best++; } bv = 0.f; }
            taken[k] = best; val[k] = bv;
        }
        if (norm_topk) { float t = 0; for (int k = 0; k < K; k++) t += val[k]; for (int k = 0; k < K; k++) val[k] /= t; }
        for (int k = 0; k < K; k++) { sel_ids[s * K + k] = taken[k]; sel_w[s * K + k] = val[k]; atomicAdd(&cnt[taken[k]], 1); used[taken[k]] = 1; }
    }
}
__global__ void k_scan(const int *cnt, int *start, int *cursor, int E) {
    if (threadIdx.x == 0) { int a = 0; for (int e = 0; e < E; e++) { start[e] = a; a += cnt[e]; cursor[e] = 0; } }
}
__global__ void k_scatter(const int *ids, const int *start, int *cursor, int *list, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int e = ids[i]; list[start[e] + atomicAdd(&cursor[e], 1)] = i;   /* i = s*K + k */
}

/* grid (I/8, chunks, E): chunk c of expert e covers list positions start[e] + c*R .. */
__global__ void k_expert_gateup_rows(const int8_t *slot_w, const float *slot_s, const int *slot_of, const int *cnt, const int *start,
                                     const int *list, const float *X, float *act, int D, int I, int K, size_t w_bytes, size_t s_elems) {
    extern __shared__ float xs[];
    int e = blockIdx.z, c = blockIdx.y, ne = cnt[e], j0 = c * PF_R;
    if (j0 >= ne) return;
    int sl = slot_of[e]; if (sl < 0) return;
    int nr = min(PF_R, ne - j0);
    for (int r = 0; r < nr; r++) {
        int tok = list[start[e] + j0 + r] / K;
        for (int i = threadIdx.x; i < D; i += blockDim.x) xs[r * D + i] = X[(size_t)tok * D + i];
    }
    __syncthreads();
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int row = blockIdx.x * GEMV_WARPS + warp;
    if (row >= I) return;
    const int8_t *w = slot_w + (size_t)sl * w_bytes; const float *sc = slot_s + (size_t)sl * s_elems;
    const uint4 *gp = reinterpret_cast<const uint4 *>(w + (size_t)row * D);
    const uint4 *up = reinterpret_cast<const uint4 *>(w + (size_t)I * D + (size_t)row * D);
    float ag[PF_R], au[PF_R];
    #pragma unroll
    for (int r = 0; r < PF_R; r++) { ag[r] = 0.f; au[r] = 0.f; }
    for (int j = lane; j < (D >> 4); j += 32) {
        uint4 gv = __ldg(gp + j), uv = __ldg(up + j);
        const int8_t *gb = reinterpret_cast<const int8_t *>(&gv), *ub = reinterpret_cast<const int8_t *>(&uv);
        #pragma unroll
        for (int r = 0; r < PF_R; r++) {
            if (r < nr) {
                const float *xj = xs + (size_t)r * D + (j << 4);
                float a = 0.f, b = 0.f;
                #pragma unroll
                for (int t = 0; t < 16; t++) { a += (float)gb[t] * xj[t]; b += (float)ub[t] * xj[t]; }
                ag[r] += a; au[r] += b;
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < PF_R; r++) {
        float g = warp_sum(ag[r]), u = warp_sum(au[r]);
        if (lane == 0 && r < nr) {
            g *= sc[row]; u *= sc[I + row];
            act[(size_t)(start[e] + j0 + r) * I + row] = (g / (1.f + expf(-g))) * u;
        }
    }
}

__global__ void k_expert_down_rows(const int8_t *slot_w, const float *slot_s, const int *slot_of, const int *cnt, const int *start,
                                   const int *list, const float *act, float *tbuf, int D, int I, size_t w_bytes, size_t s_elems) {
    extern __shared__ float xs[];
    int e = blockIdx.z, c = blockIdx.y, ne = cnt[e], j0 = c * PF_R;
    if (j0 >= ne) return;
    int sl = slot_of[e]; if (sl < 0) return;
    int nr = min(PF_R, ne - j0);
    for (int idx = threadIdx.x; idx < nr * I; idx += blockDim.x) xs[idx] = act[(size_t)(start[e] + j0) * I + idx];
    __syncthreads();
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int row = blockIdx.x * GEMV_WARPS + warp;
    if (row >= D) return;
    const uint4 *dp = reinterpret_cast<const uint4 *>(slot_w + (size_t)sl * w_bytes + (size_t)2 * I * D + (size_t)row * I);
    float ad[PF_R];
    #pragma unroll
    for (int r = 0; r < PF_R; r++) ad[r] = 0.f;
    for (int j = lane; j < (I >> 4); j += 32) {
        uint4 v = __ldg(dp + j); const int8_t *b = reinterpret_cast<const int8_t *>(&v);
        #pragma unroll
        for (int r = 0; r < PF_R; r++) {
            if (r < nr) {
                const float *xj = xs + (size_t)r * I + (j << 4);
                float a = 0.f;
                #pragma unroll
                for (int t = 0; t < 16; t++) a += (float)b[t] * xj[t];
                ad[r] += a;
            }
        }
    }
    float ds = slot_s[(size_t)sl * s_elems + 2 * I + row];
    #pragma unroll
    for (int r = 0; r < PF_R; r++) {
        float a = warp_sum(ad[r]);
        if (lane == 0 && r < nr) tbuf[(size_t)list[start[e] + j0 + r] * D + row] = a * ds;
    }
}

/* x[s] += sum_k w[s,k] * tbuf[s,k]  (k in order, as the CPU loop) */
__global__ void k_combine(float *x, const float *tbuf, const float *sel_w, int D, int K) {
    int s = blockIdx.x;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float acc = 0.f;
        for (int k = 0; k < K; k++) acc += sel_w[s * K + k] * tbuf[((size_t)s * K + k) * D + d];
        x[(size_t)s * D + d] += acc;
    }
}

static int pf_alloc(void) {
    int D = g.D, K = g.K, S = PF_MAX;
    size_t sd = (size_t)S * D * 4;
    CK(cudaMalloc(&pf.x, sd)); CK(cudaMalloc(&pf.xn, sd)); CK(cudaMalloc(&pf.q, sd)); CK(cudaMalloc(&pf.k, sd));
    CK(cudaMalloc(&pf.v, sd)); CK(cudaMalloc(&pf.ctx, sd));
    CK(cudaMalloc(&pf.logits_e, (size_t)S * g.E * 4)); CK(cudaMalloc(&pf.w_sel, (size_t)S * K * 4));
    CK(cudaMalloc(&pf.ids, (size_t)S * K * 4)); CK(cudaMalloc(&pf.list, (size_t)S * K * 4));
    CK(cudaMalloc(&pf.act, (size_t)S * K * g.I * 4)); CK(cudaMalloc(&pf.tbuf, (size_t)S * K * D * 4));
    CK(cudaMalloc(&pf.kvl, 2 * sd));
    CK(cudaMalloc(&pf.cnt, g.E * 4)); CK(cudaMalloc(&pf.start, g.E * 4)); CK(cudaMalloc(&pf.cursor, g.E * 4)); CK(cudaMalloc(&pf.used, g.E * 4));
    CK(cudaHostAlloc((void **)&pf.hkv, (size_t)g.L * 2 * sd, 0));
    CK(cudaHostAlloc((void **)&pf.h_used, g.E * 4, 0));
    CK(cudaHostAlloc((void **)&pf.h_logits_rows, (size_t)PF_R * g.V * 4, 0));
    CK(cudaFuncSetAttribute(k_gemv_f16_rows<PF_R>, cudaFuncAttributeMaxDynamicSharedMemorySize, PF_R * g.D * 4));
    CK(cudaFuncSetAttribute(k_expert_gateup_rows, cudaFuncAttributeMaxDynamicSharedMemorySize, PF_R * g.D * 4));
    CK(cudaFuncSetAttribute(k_expert_down_rows, cudaFuncAttributeMaxDynamicSharedMemorySize, PF_R * g.I * 4));
    pf.ok = 1;
    return 1;
}

/* x_rows: S embedding rows (host). Positions pos_base .. pos_base+S-1. */
OC_API int oc_pf_begin(const float *x_rows, int S, int pos_base) {
    if (S < 1 || S > PF_MAX || pos_base + S > g.max_t) return -4;
    if (!pf.ok && pf_alloc() < 0) return -1;
    pf.S = S; pf.pos = pos_base;
    CK(cudaMemcpyAsync(pf.x, x_rows, (size_t)S * g.D * 4, cudaMemcpyHostToDevice, g.st));
    return 1;
}

OC_API int oc_pf_attn(int l) {
    LayerDev *L = &g.layers[l]; int D = g.D, S = pf.S;
    dim3 gg((D + GEMV_WARPS - 1) / GEMV_WARPS, (S + PF_R - 1) / PF_R);
    size_t sm = (size_t)PF_R * D * 4;
    k_rmsnorm_rows<<<S, 256, 0, g.st>>>(pf.xn, pf.x, L->in_ln, D, g.eps);
    k_gemv_f16_rows<PF_R><<<gg, 32 * GEMV_WARPS, sm, g.st>>>(L->q, pf.xn, pf.q, D, D, S, 0);
    k_gemv_f16_rows<PF_R><<<gg, 32 * GEMV_WARPS, sm, g.st>>>(L->k, pf.xn, pf.k, D, D, S, 0);
    k_gemv_f16_rows<PF_R><<<gg, 32 * GEMV_WARPS, sm, g.st>>>(L->v, pf.xn, pf.v, D, D, S, 0);
    k_rmsnorm_rows<<<S, 256, 0, g.st>>>(pf.q, pf.q, L->qn, D, g.eps);
    k_rmsnorm_rows<<<S, 256, 0, g.st>>>(pf.k, pf.k, L->kn, D, g.eps);
    k_rope_append_rows<<<dim3((g.H * (g.hd / 2) + 127) / 128, S), 128, 0, g.st>>>(pf.q, pf.k, pf.v, L->K, L->V, pf.kvl,
                                                                                  g.H, g.hd, g.max_t, pf.pos, g.theta, D, S);
    /* per-layer KV rows back to the host mirror (stream-ordered: pf.kvl is reused by the next layer) */
    CK(cudaMemcpyAsync(pf.hkv + (size_t)l * 2 * PF_MAX * D, pf.kvl, (size_t)2 * S * D * 4, cudaMemcpyDeviceToHost, g.st));
    k_attn_rows<<<dim3(g.H, S), 256, (size_t)(pf.pos + S) * 4, g.st>>>(pf.q, L->K, L->V, pf.ctx, g.hd, g.max_t, pf.pos, D, 1.f / sqrtf((float)g.hd));
    k_gemv_f16_rows<PF_R><<<gg, 32 * GEMV_WARPS, sm, g.st>>>(L->o, pf.ctx, pf.x, D, D, S, 1);
    CK(cudaGetLastError());
    return 1;
}

/* routing + per-expert token grouping. If `used_out` is non-NULL it is filled (host sync) so the engine can
 * make every used expert resident before oc_pf_experts. */
OC_API int oc_pf_route(int l, int *used_out) {
    LayerDev *L = &g.layers[l]; int D = g.D, S = pf.S, E = g.E, K = g.K;
    k_rmsnorm_rows<<<S, 256, 0, g.st>>>(pf.xn, pf.x, L->post_ln, D, g.eps);
    CK(cudaMemsetAsync(pf.cnt, 0, E * 4, g.st)); CK(cudaMemsetAsync(pf.used, 0, E * 4, g.st));
    k_gemv_f32_rows<<<dim3((E + GEMV_WARPS - 1) / GEMV_WARPS, S), 32 * GEMV_WARPS, (size_t)D * 4, g.st>>>(L->gate, pf.xn, pf.logits_e, D, E, S);
    k_route_rows<<<S, 64, 0, g.st>>>(pf.logits_e, pf.w_sel, pf.ids, pf.cnt, pf.used, E, K, g.norm_topk);
    k_scan<<<1, 32, 0, g.st>>>(pf.cnt, pf.start, pf.cursor, E);
    k_scatter<<<(S * K + 127) / 128, 128, 0, g.st>>>(pf.ids, pf.start, pf.cursor, pf.list, S * K);
    CK(cudaGetLastError());
    if (used_out) {
        CK(cudaMemcpyAsync(pf.h_used, pf.used, E * 4, cudaMemcpyDeviceToHost, g.st));
        CK(cudaStreamSynchronize(g.st));
        memcpy(used_out, pf.h_used, E * 4);
    }
    return 1;
}

OC_API int oc_pf_experts(int l) {
    int D = g.D, I = g.I, S = pf.S, E = g.E, K = g.K;
    int chunks = (S + PF_R - 1) / PF_R;   /* an expert appears at most once per token, so ne <= S */
    const int *slot_of = g.slot_of + (size_t)l * E;
    k_expert_gateup_rows<<<dim3((I + GEMV_WARPS - 1) / GEMV_WARPS, chunks, E), 32 * GEMV_WARPS, (size_t)PF_R * D * 4, g.st>>>(
        g.slot_w, g.slot_s, slot_of, pf.cnt, pf.start, pf.list, pf.xn, pf.act, D, I, K, g.w_bytes, g.s_elems);
    k_expert_down_rows<<<dim3((D + GEMV_WARPS - 1) / GEMV_WARPS, chunks, E), 32 * GEMV_WARPS, (size_t)PF_R * I * 4, g.st>>>(
        g.slot_w, g.slot_s, slot_of, pf.cnt, pf.start, pf.list, pf.act, pf.tbuf, D, I, g.w_bytes, g.s_elems);
    k_combine<<<S, 256, 0, g.st>>>(pf.x, pf.tbuf, pf.w_sel, D, K);
    CK(cudaGetLastError());
    return 1;
}

/* final norm + lm_head for `nrows` rows starting at row `first` (1 row for prefill, up to PF_R for verification);
 * logits_out is [nrows][V]. *kv_out points at the pinned [L][2][PF_MAX][D] K/V rows of this chunk (S valid rows). */
OC_API int oc_pf_finish(int first, int nrows, float *logits_out, float **kv_out) {
    int D = g.D;
    if (nrows > PF_R) return -5;
    if (nrows > 0) {
        k_rmsnorm_rows<<<pf.S, 256, 0, g.st>>>(pf.xn, pf.x, g.final_norm, D, g.eps);
        dim3 gg((g.V + GEMV_WARPS - 1) / GEMV_WARPS, 1);
        k_gemv_f16_rows<PF_R><<<gg, 32 * GEMV_WARPS, (size_t)PF_R * D * 4, g.st>>>(g.lm_head, pf.xn + (size_t)first * D, g.logits, D, g.V, nrows, 0);
        CK(cudaMemcpyAsync(pf.h_logits_rows, g.logits, (size_t)nrows * g.V * 4, cudaMemcpyDeviceToHost, g.st));
    }
    CK(cudaStreamSynchronize(g.st));
    if (nrows > 0) memcpy(logits_out, pf.h_logits_rows, (size_t)nrows * g.V * 4);
    *kv_out = pf.hkv;
    return 1;
}

/* achievable device-memory traffic (D2D copy, read+write bytes) and pinned host->device bandwidth, GB/s; used by the
 * engine's bottleneck diagnostics as the measured ceilings */
OC_API int oc_bench_bw(double *d2d_gbs, double *h2d_gbs) {
    size_t n = (size_t)64 << 20; void *a = nullptr, *b = nullptr, *h = nullptr;
    cudaEvent_t e0, e1; float ms = 0;
    CK(cudaMalloc(&a, n)); CK(cudaMalloc(&b, n)); CK(cudaHostAlloc(&h, n, 0));
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    CK(cudaMemcpyAsync(b, a, n, cudaMemcpyDeviceToDevice, g.st));
    CK(cudaEventRecord(e0, g.st));
    for (int i = 0; i < 8; i++) CK(cudaMemcpyAsync(b, a, n, cudaMemcpyDeviceToDevice, g.st));
    CK(cudaEventRecord(e1, g.st)); CK(cudaEventSynchronize(e1)); CK(cudaEventElapsedTime(&ms, e0, e1));
    *d2d_gbs = 8.0 * 2.0 * n / (ms * 1e-3) / 1e9;
    CK(cudaEventRecord(e0, g.st));
    for (int i = 0; i < 8; i++) CK(cudaMemcpyAsync(a, h, n, cudaMemcpyHostToDevice, g.st));
    CK(cudaEventRecord(e1, g.st)); CK(cudaEventSynchronize(e1)); CK(cudaEventElapsedTime(&ms, e0, e1));
    *h2d_gbs = 8.0 * n / (ms * 1e-3) / 1e9;
    cudaEventDestroy(e0); cudaEventDestroy(e1); cudaFree(a); cudaFree(b); cudaFreeHost(h);
    return 1;
}

OC_API void oc_shutdown(void) { if (g.ready) { cudaDeviceSynchronize(); cudaDeviceReset(); g.ready = 0; } }

OC_API void oc_prof_enable(int on) { prof_on = on; }
/* ms[P_N] = total ms per phase over `tokens` decode tokens: dense gemv, attention core, router, expert gate+up, expert down, head */
OC_API int oc_prof_get(double *ms, long *tokens) { for (int i = 0; i < P_N; i++) ms[i] = prof_ms[i]; *tokens = prof_tokens; return P_N; }
