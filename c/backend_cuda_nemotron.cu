/* backend_cuda_nemotron.cu -- CUDA kernels for Nemotron-H / Nemotron-3.5-Super-VL
 * (NVFP4, FP8 E4M3, BF16, F32 GEMV + Mamba-2 SSM + NoPE GQA Attention).
 */
#include "backend_cuda_nemotron.h"
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>

static cudaStream_t g_stream = NULL;

__constant__ float c_e4m3_lut[256];
__constant__ float c_fp4_lut[16];

static __device__ __forceinline__ float d_bf16_to_f32(uint16_t b) {
    uint32_t u = ((uint32_t)b) << 16;
    return __uint_as_float(u);
}

static __device__ __forceinline__ float d_silu(float x) {
    return x / (1.0f + expf(-x));
}

static __device__ __forceinline__ float d_softplus(float x) {
    return x > 20.0f ? x : log1pf(expf(x));
}

static __device__ __forceinline__ float warp_sum(float v) {
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, off);
    return v;
}

/* ----------------------------------------------------- Elementwise / Norm ---- */

__global__ void k_rmsnorm(float *out, const float *in, const float *w, int n, float eps) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    float ss = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float v = in[i];
        ss += v * v;
    }
    ss = warp_sum(ss);
    int wid = tid >> 5, lane = tid & 31;
    if (lane == 0) smem[wid] = ss;
    __syncthreads();
    int nw = blockDim.x >> 5;
    if (wid == 0) {
        float v = (lane < nw) ? smem[lane] : 0.0f;
        v = warp_sum(v);
        if (lane == 0) smem[0] = rsqrtf(v / (float)n + eps);
    }
    __syncthreads();
    float inv = smem[0];
    for (int i = tid; i < n; i += blockDim.x)
        out[i] = in[i] * inv * w[i];
}

__global__ void k_add_inplace(float *a, const float *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] += b[i];
}

/* ----------------------------------------------------- GEMV Kernels ---------- */
/* 4 warps per block = 4 output rows per block, 32 threads per row. */

__global__ void k_matvec_f32(float *y, const float *w, const float *x,
                             int O, int I, int act_relu2, float accum_scale) {
    int row = blockIdx.x * 4 + (threadIdx.x >> 5);
    int lane = threadIdx.x & 31;
    if (row >= O) return;
    const float *wr = w + (size_t)row * I;
    float acc = 0.0f;
    int i = lane * 4;
    for (; i + 3 < I; i += 128) {
        float4 wv = *reinterpret_cast<const float4 *>(wr + i);
        float4 xv = *reinterpret_cast<const float4 *>(x + i);
        acc += wv.x * xv.x + wv.y * xv.y + wv.z * xv.z + wv.w * xv.w;
    }
    for (int j = (I & ~3) + lane; j < I; j += 32) acc += wr[j] * x[j];
    acc = warp_sum(acc);
    if (lane == 0) {
        if (act_relu2) acc = acc > 0.0f ? acc * acc : 0.0f;
        if (accum_scale != 0.0f) y[row] += accum_scale * acc;
        else y[row] = acc;
    }
}

__global__ void k_matvec_bf16(float *y, const uint16_t *w, const float *x,
                              int O, int I, int act_relu2, float accum_scale) {
    int row = blockIdx.x * 4 + (threadIdx.x >> 5);
    int lane = threadIdx.x & 31;
    if (row >= O) return;
    const uint16_t *wr = w + (size_t)row * I;
    float acc = 0.0f;
    int i = lane * 4;
    for (; i + 3 < I; i += 128) {
        uint2 w2 = *reinterpret_cast<const uint2 *>(wr + i);
        float4 xv = *reinterpret_cast<const float4 *>(x + i);
        acc += d_bf16_to_f32((uint16_t)(w2.x & 0xffffu)) * xv.x
             + d_bf16_to_f32((uint16_t)(w2.x >> 16))     * xv.y
             + d_bf16_to_f32((uint16_t)(w2.y & 0xffffu)) * xv.z
             + d_bf16_to_f32((uint16_t)(w2.y >> 16))     * xv.w;
    }
    for (int j = (I & ~3) + lane; j < I; j += 32) acc += d_bf16_to_f32(wr[j]) * x[j];
    acc = warp_sum(acc);
    if (lane == 0) {
        if (act_relu2) acc = acc > 0.0f ? acc * acc : 0.0f;
        if (accum_scale != 0.0f) y[row] += accum_scale * acc;
        else y[row] = acc;
    }
}

__global__ void k_matvec_fp8(float *y, const uint8_t *w, const float *x,
                             float scale, int O, int I, int act_relu2, float accum_scale) {
    int row = blockIdx.x * 4 + (threadIdx.x >> 5);
    int lane = threadIdx.x & 31;
    if (row >= O) return;
    const uint8_t *wr = w + (size_t)row * I;
    float acc = 0.0f;
    int i = lane * 4;
    for (; i + 3 < I; i += 128) {
        uchar4 w4 = *reinterpret_cast<const uchar4 *>(wr + i);
        float4 xv = *reinterpret_cast<const float4 *>(x + i);
        acc += c_e4m3_lut[w4.x] * xv.x + c_e4m3_lut[w4.y] * xv.y
             + c_e4m3_lut[w4.z] * xv.z + c_e4m3_lut[w4.w] * xv.w;
    }
    for (int j = (I & ~3) + lane; j < I; j += 32) acc += c_e4m3_lut[wr[j]] * x[j];
    acc = warp_sum(acc) * scale;
    if (lane == 0) {
        if (act_relu2) acc = acc > 0.0f ? acc * acc : 0.0f;
        if (accum_scale != 0.0f) y[row] += accum_scale * acc;
        else y[row] = acc;
    }
}

/* NVFP4: 16 weights = 8 packed U8 bytes (uint2), 1 E4M3 scale per 16 weights, * scale2.
 * Each lane processes 1 group of 16 weights per iteration (32 lanes = 512 weights/iter). */
__global__ void k_matvec_nvfp4(float *y, const uint8_t *q, const uint8_t *sc,
                               const float *x, float scale2, int O, int I,
                               int act_relu2, float accum_scale) {
    int row = blockIdx.x * 4 + (threadIdx.x >> 5);
    int lane = threadIdx.x & 31;
    if (row >= O) return;
    int ng = I >> 4;
    const uint2 *qr = reinterpret_cast<const uint2 *>(q + (size_t)row * (I >> 1));
    const uint8_t *sr = sc + (size_t)row * ng;
    const float4 *x4 = reinterpret_cast<const float4 *>(x);
    float acc = 0.0f;
    for (int g = lane; g < ng; g += 32) {
        float gs = c_e4m3_lut[sr[g]];
        uint2 qb = qr[g];
        uint32_t w0 = qb.x, w1 = qb.y;
        float4 xv0 = x4[g * 4 + 0];
        float4 xv1 = x4[g * 4 + 1];
        float4 xv2 = x4[g * 4 + 2];
        float4 xv3 = x4[g * 4 + 3];
        float gsum =
            c_fp4_lut[w0 & 0xfu]         * xv0.x + c_fp4_lut[(w0 >> 4)  & 0xfu] * xv0.y +
            c_fp4_lut[(w0 >> 8)  & 0xfu] * xv0.z + c_fp4_lut[(w0 >> 12) & 0xfu] * xv0.w +
            c_fp4_lut[(w0 >> 16) & 0xfu] * xv1.x + c_fp4_lut[(w0 >> 20) & 0xfu] * xv1.y +
            c_fp4_lut[(w0 >> 24) & 0xfu] * xv1.z + c_fp4_lut[(w0 >> 28) & 0xfu] * xv1.w +
            c_fp4_lut[w1 & 0xfu]         * xv2.x + c_fp4_lut[(w1 >> 4)  & 0xfu] * xv2.y +
            c_fp4_lut[(w1 >> 8)  & 0xfu] * xv2.z + c_fp4_lut[(w1 >> 12) & 0xfu] * xv2.w +
            c_fp4_lut[(w1 >> 16) & 0xfu] * xv3.x + c_fp4_lut[(w1 >> 20) & 0xfu] * xv3.y +
            c_fp4_lut[(w1 >> 24) & 0xfu] * xv3.z + c_fp4_lut[(w1 >> 28) & 0xfu] * xv3.w;
        acc += gsum * gs;
    }
    acc = warp_sum(acc) * scale2;
    if (lane == 0) {
        if (act_relu2) acc = acc > 0.0f ? acc * acc : 0.0f;
        if (accum_scale != 0.0f) y[row] += accum_scale * acc;
        else y[row] = acc;
    }
}

/* ----------------------------------------------------- Mamba-2 Kernels ------- */

__global__ void k_mamba_conv_silu(float *xbc_out, float *conv_state, const float *xbc_in,
                                  const float *conv_w, const float *conv_b, int C_dim, int K) {
    int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= C_dim) return;
    float *s = conv_state + (size_t)ch * K;
    for (int k = 0; k < K - 1; k++) s[k] = s[k + 1];
    s[K - 1] = xbc_in[ch];
    const float *w = conv_w + (size_t)ch * K;
    float acc = conv_b ? conv_b[ch] : 0.0f;
    for (int k = 0; k < K; k++) acc += s[k] * w[k];
    xbc_out[ch] = d_silu(acc);
}

/* 1 warp (32 threads) per (h, p) element; threads stride over N (ssm_state=128 -> 4 iters). */
__global__ void k_mamba_ssm_step(float *y, float *ssm_state, const float *xbc,
                                 const float *dt_raw, const float *dt_bias,
                                 const float *A_neg_exp, const float *D,
                                 int H, int P, int N, int G, float dt_min, float dt_max) {
    int hp = blockIdx.x * 4 + (threadIdx.x >> 5);
    int lane = threadIdx.x & 31;
    int D_in = H * P;
    if (hp >= D_in) return;
    int h = hp / P;
    int p = hp - h * P;
    int hpg = H / G;
    int g = h / hpg;

    float dt = d_softplus(dt_raw[h] + dt_bias[h]);
    if (dt < dt_min) dt = dt_min;
    if (dt > dt_max) dt = dt_max;
    float dA = expf(A_neg_exp[h] * dt);
    float xp = xbc[hp];
    float dt_xp = dt * xp;

    const float *Bg = xbc + D_in + g * N;
    const float *Cg = xbc + D_in + G * N + g * N;
    float *shp = ssm_state + ((size_t)h * P + p) * N;

    float yacc = 0.0f;
    for (int n = lane; n < N; n += 32) {
        float nv = shp[n] * dA + dt_xp * Bg[n];
        shp[n] = nv;
        yacc += nv * Cg[n];
    }
    yacc = warp_sum(yacc);
    if (lane == 0) y[hp] = yacc + D[h] * xp;
}

/* 1 block per group (G=8, group_size=1024). */
__global__ void k_mamba_gated_rmsnorm(float *y, const float *z, const float *norm_w,
                                      int D_in, int G, float eps) {
    extern __shared__ float smem[];
    int g = blockIdx.x;
    int tid = threadIdx.x;
    int gsz = D_in / G;
    int off = g * gsz;
    float ss = 0.0f;
    for (int j = tid; j < gsz; j += blockDim.x) {
        float v = y[off + j] * d_silu(z[off + j]);
        y[off + j] = v;
        ss += v * v;
    }
    ss = warp_sum(ss);
    int wid = tid >> 5, lane = tid & 31;
    if (lane == 0) smem[wid] = ss;
    __syncthreads();
    int nw = blockDim.x >> 5;
    if (wid == 0) {
        float v = (lane < nw) ? smem[lane] : 0.0f;
        v = warp_sum(v);
        if (lane == 0) smem[0] = rsqrtf(v / (float)gsz + eps);
    }
    __syncthreads();
    float inv = smem[0];
    for (int j = tid; j < gsz; j += blockDim.x)
        y[off + j] = y[off + j] * inv * norm_w[off + j];
}

/* ----------------------------------------------------- NoPE GQA Attention ---- */

__global__ void k_kv_append(float *k_cache, float *v_cache, const float *k, const float *v,
                            int kv_dim, int pos) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < kv_dim) {
        k_cache[(size_t)pos * kv_dim + i] = k[i];
        v_cache[(size_t)pos * kv_dim + i] = v[i];
    }
}

/* 1 block (128 threads, 1 thread per head_dim element for hd<=128) per Q head, online softmax. */
__global__ void k_attn_nope_gqa(float *out, const float *k_cache, const float *v_cache,
                                const float *q, int H, int KVH, int hd, int seq) {
    extern __shared__ float smem[];
    int h = blockIdx.x;
    int d = threadIdx.x;
    if (h >= H) return;
    int gqa = H / KVH;
    int kh = h / gqa;
    int kv_dim = KVH * hd;
    float scale = rsqrtf((float)hd);
    float qd = (d < hd) ? q[h * hd + d] : 0.0f;

    float m_max = -1e30f;
    float l_sum = 0.0f;
    float acc = 0.0f;

    for (int t = 0; t < seq; t++) {
        const float *kt = k_cache + (size_t)t * kv_dim + kh * hd;
        const float *vt = v_cache + (size_t)t * kv_dim + kh * hd;
        float prod = (d < hd) ? (qd * kt[d]) : 0.0f;
        prod = warp_sum(prod);
        int wid = d >> 5, lane = d & 31;
        if (lane == 0) smem[wid] = prod;
        __syncthreads();
        if (wid == 0) {
            int nw = blockDim.x >> 5;
            float s = (lane < nw) ? smem[lane] : 0.0f;
            s = warp_sum(s);
            if (lane == 0) smem[0] = s * scale;
        }
        __syncthreads();
        float sc = smem[0];
        float new_max = fmaxf(m_max, sc);
        float alpha = expf(m_max - new_max);
        float w = expf(sc - new_max);
        l_sum = l_sum * alpha + w;
        if (d < hd) acc = acc * alpha + w * vt[d];
        m_max = new_max;
    }
    if (d < hd) out[h * hd + d] = acc / l_sum;
}

/* ----------------------------------------------------- Host C API ------------ */

static int nemo_cu_init(int device) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count <= 0) return -1;
    if (device < 0 || device >= count) device = 0;
    if (cudaSetDevice(device) != cudaSuccess) return -1;
    if (cudaStreamCreate(&g_stream) != cudaSuccess) return -1;

    float h_e4m3[256];
    for (int i = 0; i < 256; i++) {
        int s = (i >> 7) & 1, e = (i >> 3) & 0xf, m = i & 0x7;
        float v = (e == 0xf && m == 0x7) ? 0.0f :
                  (e == 0 ? ldexpf((float)m, -9) : ldexpf(1.0f + (float)m / 8.0f, e - 7));
        h_e4m3[i] = s ? -v : v;
    }
    static const float h_fp4[16] = {
         0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
    };
    if (cudaMemcpyToSymbol(c_e4m3_lut, h_e4m3, sizeof(h_e4m3)) != cudaSuccess) return -1;
    if (cudaMemcpyToSymbol(c_fp4_lut, h_fp4, sizeof(h_fp4)) != cudaSuccess) return -1;
    return 0;
}

static void nemo_cu_shutdown(void) {
    if (g_stream) { cudaStreamDestroy(g_stream); g_stream = NULL; }
}

static void nemo_cu_mem_info(size_t *free_b, size_t *total_b) {
    size_t f = 0, t = 0;
    cudaMemGetInfo(&f, &t);
    if (free_b) *free_b = f;
    if (total_b) *total_b = t;
}

static void *nemo_cu_alloc(size_t n) {
    void *p = NULL;
    if (cudaMalloc(&p, n ? n : 16) != cudaSuccess) return NULL;
    return p;
}

static void *nemo_cu_upload(const void *h, size_t n) {
    void *p = nemo_cu_alloc(n);
    if (!p) return NULL;
    if (cudaMemcpy(p, h, n, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(p);
        return NULL;
    }
    return p;
}

static void nemo_cu_free(void *d) { if (d) cudaFree(d); }

static int nemo_cu_h2d(void *dst, const void *src, size_t n) {
    return cudaMemcpyAsync(dst, src, n, cudaMemcpyHostToDevice, g_stream) == cudaSuccess ? 0 : -1;
}

static int nemo_cu_d2h(void *dst, const void *src, size_t n) {
    if (cudaMemcpyAsync(dst, src, n, cudaMemcpyDeviceToHost, g_stream) != cudaSuccess) return -1;
    return cudaStreamSynchronize(g_stream) == cudaSuccess ? 0 : -1;
}

static int nemo_cu_memset_zero(void *dst, size_t n) {
    return cudaMemsetAsync(dst, 0, n, g_stream) == cudaSuccess ? 0 : -1;
}

static int nemo_cu_sync(void) {
    return cudaStreamSynchronize(g_stream) == cudaSuccess ? 0 : -1;
}

static void nemo_cu_rmsnorm(float *d_out, const float *d_in, const float *d_w, int n, float eps) {
    int th = 256;
    while (th > n && th > 32) th >>= 1;
    k_rmsnorm<<<1, th, (size_t)(th / 32) * sizeof(float), g_stream>>>(d_out, d_in, d_w, n, eps);
}

static void nemo_cu_add_inplace(float *d_a, const float *d_b, int n) {
    int th = 256, bl = (n + th - 1) / th;
    k_add_inplace<<<bl, th, 0, g_stream>>>(d_a, d_b, n);
}

static void nemo_cu_matvec(float *d_y, const void *d_w, const uint8_t *d_sc, float scale_scalar,
                           int fmt, int O, int I, const float *d_x, int act_relu2, float accum_scale) {
    int bl = (O + 3) / 4;
    if (fmt == 0) {
        k_matvec_f32<<<bl, 128, 0, g_stream>>>(d_y, (const float *)d_w, d_x, O, I, act_relu2, accum_scale);
    } else if (fmt == 1) {
        k_matvec_bf16<<<bl, 128, 0, g_stream>>>(d_y, (const uint16_t *)d_w, d_x, O, I, act_relu2, accum_scale);
    } else if (fmt == 2) {
        k_matvec_fp8<<<bl, 128, 0, g_stream>>>(d_y, (const uint8_t *)d_w, d_x, scale_scalar, O, I, act_relu2, accum_scale);
    } else if (fmt == 3) {
        k_matvec_nvfp4<<<bl, 128, 0, g_stream>>>(d_y, (const uint8_t *)d_w, d_sc, d_x, scale_scalar, O, I, act_relu2, accum_scale);
    }
}

static void nemo_cu_mamba_conv_silu(float *d_xbc_out, float *d_conv_state, const float *d_xbc_in,
                                    const float *d_conv_w, const float *d_conv_b, int C_dim, int K) {
    int th = 256, bl = (C_dim + th - 1) / th;
    k_mamba_conv_silu<<<bl, th, 0, g_stream>>>(d_xbc_out, d_conv_state, d_xbc_in, d_conv_w, d_conv_b, C_dim, K);
}

static void nemo_cu_mamba_ssm_step(float *d_y, float *d_ssm_state, const float *d_xbc,
                                   const float *d_dt_raw, const float *d_dt_bias,
                                   const float *d_A_neg_exp, const float *d_D,
                                   int H, int P, int N, int G, float dt_min, float dt_max) {
    int bl = (H * P + 3) / 4;
    k_mamba_ssm_step<<<bl, 128, 0, g_stream>>>(d_y, d_ssm_state, d_xbc, d_dt_raw, d_dt_bias,
                                               d_A_neg_exp, d_D, H, P, N, G, dt_min, dt_max);
}

static void nemo_cu_mamba_gated_rmsnorm(float *d_y, const float *d_z, const float *d_norm_w,
                                        int D_in, int G, float eps) {
    int th = 256;
    int gsz = D_in / G;
    while (th > gsz && th > 32) th >>= 1;
    k_mamba_gated_rmsnorm<<<G, th, (size_t)(th / 32) * sizeof(float), g_stream>>>(d_y, d_z, d_norm_w, D_in, G, eps);
}

static void nemo_cu_attn_nope_gqa(float *d_out, float *d_k_cache, float *d_v_cache,
                                  const float *d_q, const float *d_k, const float *d_v,
                                  int H, int KVH, int hd, int pos) {
    int kv_dim = KVH * hd;
    int bl_kv = (kv_dim + 255) / 256;
    k_kv_append<<<bl_kv, 256, 0, g_stream>>>(d_k_cache, d_v_cache, d_k, d_v, kv_dim, pos);
    int th = 32;
    while (th < hd) th <<= 1;
    if (th > 256) th = 256;
    k_attn_nope_gqa<<<H, th, (size_t)(th / 32) * sizeof(float), g_stream>>>(
        d_out, d_k_cache, d_v_cache, d_q, H, KVH, hd, pos + 1);
}

static const ColiNemoCudaOps g_ops = {
    nemo_cu_init,
    nemo_cu_shutdown,
    nemo_cu_mem_info,
    nemo_cu_alloc,
    nemo_cu_upload,
    nemo_cu_free,
    nemo_cu_h2d,
    nemo_cu_d2h,
    nemo_cu_memset_zero,
    nemo_cu_sync,
    nemo_cu_rmsnorm,
    nemo_cu_add_inplace,
    nemo_cu_matvec,
    nemo_cu_mamba_conv_silu,
    nemo_cu_mamba_ssm_step,
    nemo_cu_mamba_gated_rmsnorm,
    nemo_cu_attn_nope_gqa
};

const ColiNemoCudaOps *coli_cuda_nemotron_ops(void) { return &g_ops; }
