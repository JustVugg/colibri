/* cuda_inkc.cuh -- Inkling's and Inkling-2's ops on the CUDA chain (cuda_chain.h's sconv, relattn,
 * ink2_mix, ink2_clip and ink2_dconv), included once by cuda_chain.cu after cuda_glm53c.cuh, whose
 * grid helper it uses. Each is the Vulkan chain's op of the same name (shaders/chain_sconv.comp,
 * chain_relattn.comp, chain_ink2.comp) with the same fields in the same order; vk_chain.h has
 * their contracts:
 *   sconv      mode 0 the depthwise causal short convolution with its ring (residual inside),
 *              mode 1 x *= fc, mode 2 x /= fc over n floats
 *   relattn    grouped attention with a relative-position bias bank, a per-row tau, a sliding
 *              window over a ring cache, the step's own rows from kvs, and with has_sink a sink
 *              logit per (row, head) in tau at sink_off that starts the online softmax
 *   ink2_mix   the attention residuals' mix of RMS-normalised block states and the stream
 *   ink2_clip  the L2 clip of each segment after an optional per-element scale
 *   ink2_dconv a global layer's dynamic short convolution with its state */

__global__ static void ccg_sconv_k(float *x, const float *w, float *ring, CcSconv p) {
    const long c = (long)(blockIdx.x + blockIdx.y * gridDim.x) * 64 + threadIdx.x;
    if (p.mode != 0) {
        if (c >= p.n) return;
        const float v = x[p.x_off + c];
        x[p.x_off + c] = p.mode == 1 ? v * p.fc : v / p.fc;
        return;
    }
    if (c >= p.C) return;
    const int nh = p.CK - 1;
    float hist[8];
#pragma unroll
    for (int k = 0; k < 8; k++) hist[k] = k < nh ? ring[p.ring_off + c * nh + k] : 0.f;
    const float *wc = w + p.w_off + c * p.CK;
    const float wl = wc[p.CK - 1];
    for (int s = 0; s < p.S; s++) {
        const long xi = p.x_off + (long)s * p.x_row + c;
        const float cur = x[xi];
        float acc = 0.f;
#pragma unroll
        for (int k = 0; k < 8; k++) if (k < nh) acc += wc[k] * hist[k];
        acc += wl * cur;
        const float outv = acc + cur;
        x[xi] = (p.fc != 1.f && p.fc != 0.f) ? outv / p.fc : outv;
#pragma unroll
        for (int k = 0; k < 7; k++) if (k + 1 < nh) hist[k] = hist[k + 1];
#pragma unroll
        for (int k = 0; k < 8; k++) if (k == nh - 1) hist[k] = cur;
    }
#pragma unroll
    for (int k = 0; k < 8; k++) if (k < nh) ring[p.ring_off + c * nh + k] = hist[k];
}

/* 2D parallel depthwise causal short convolution for prefill (S >= 16):
 * Pass 1 computes bit-exact outputs into tmp[S, C] across grid (ceil(C/128), ceil(S/16));
 * Pass 2 saves the last nh original rows of x into ring and writes tmp -> x (zero race). */
__global__ static void ccg_sconv_rows_k(const float *__restrict__ x, const float *__restrict__ w,
                                        const float *__restrict__ ring, float *__restrict__ tmp, CcSconv p) {
    const int c = (int)blockIdx.x * 128 + (int)threadIdx.x;
    const int s0 = (int)blockIdx.y * 16;
    if (c >= p.C || s0 >= p.S) return;
    const int nh = p.CK - 1;
    const int s1 = min(s0 + 16, p.S);
    const float *wc = w + p.w_off + (long)c * p.CK;
    float wt[9];
#pragma unroll
    for (int k = 0; k < 9; k++) wt[k] = k < p.CK ? __ldg(wc + k) : 0.f;
    float hist[8];
#pragma unroll
    for (int k = 0; k < 8; k++) {
        if (k < nh) {
            const int src_s = s0 - (nh - k);
            hist[k] = src_s < 0 ? __ldg(ring + p.ring_off + (long)c * nh + (nh + src_s))
                                : __ldg(x + p.x_off + (long)src_s * p.x_row + c);
        } else hist[k] = 0.f;
    }
    const float wl = wt[p.CK - 1];
    const bool div_fc = (p.fc != 1.f && p.fc != 0.f);
    for (int s = s0; s < s1; s++) {
        const float cur = __ldg(x + p.x_off + (long)s * p.x_row + c);
        float acc = 0.f;
#pragma unroll
        for (int k = 0; k < 8; k++) if (k < nh) acc += wt[k] * hist[k];
        acc += wl * cur;
        const float outv = acc + cur;
        tmp[(long)s * p.C + c] = div_fc ? outv / p.fc : outv;
#pragma unroll
        for (int k = 0; k < 7; k++) if (k + 1 < nh) hist[k] = hist[k + 1];
#pragma unroll
        for (int k = 0; k < 8; k++) if (k == nh - 1) hist[k] = cur;
    }
}
__global__ static void ccg_conv_commit_k(float *__restrict__ x, float *__restrict__ ring,
                                         const float *__restrict__ tmp,
                                         int S, int C, int nh, int x_off, int x_row, int ring_off) {
    const long idx = (long)(blockIdx.x + blockIdx.y * gridDim.x) * 256 + threadIdx.x;
    const long tot = (long)S * C;
    if (idx >= tot) return;
    const int s = (int)(idx / C);
    const int c = (int)(idx - (long)s * C);
    const long xi = (long)x_off + (long)s * x_row + c;
    if (s >= S - nh) ring[ring_off + (long)c * nh + (s - (S - nh))] = x[xi];
    x[xi] = tmp[idx];
}
__global__ static void ccg_conv_commit_f4_k(float *__restrict__ x, float *__restrict__ ring,
                                            const float *__restrict__ tmp,
                                            int S, int C, int nh, int x_off, int x_row, int ring_off) {
    const int C4 = C >> 2;
    const long idx = (long)(blockIdx.x + blockIdx.y * gridDim.x) * 256 + threadIdx.x;
    const long tot4 = (long)S * C4;
    if (idx >= tot4) return;
    const int s = (int)(idx / C4);
    const int c4 = (int)(idx - (long)s * C4);
    const int c0 = c4 << 2;
    float4 *xp = reinterpret_cast<float4 *>(x + (long)x_off + (long)s * x_row) + c4;
    if (s >= S - nh) {
        const float4 orig = *xp;
        const int slot = s - (S - nh);
        ring[ring_off + (long)(c0 + 0) * nh + slot] = orig.x;
        ring[ring_off + (long)(c0 + 1) * nh + slot] = orig.y;
        ring[ring_off + (long)(c0 + 2) * nh + slot] = orig.z;
        ring[ring_off + (long)(c0 + 3) * nh + slot] = orig.w;
    }
    *xp = reinterpret_cast<const float4 *>(tmp)[idx];
}

/* ==== Blackwell / Ampere Tensor-Core FlashRelAttn (hd == 128, d_rel <= 16, S >= 16) ====
 * Tile BM=64 query rows (4 warps x 16 rows) x BN=64 key/value rows, HD=128.
 *   - Q stays in registers across the entire KV loop (ldmatrix.x4).
 *   - K and V staged in 32 KB swizzled FP16 shared memory (zero bank conflicts).
 *   - Relative position bias bank R and relp staged in exact FP32 (4 KB + 8.5 KB, total 44.5 KB <= 48 KB)
 *     so relative-position logit biases have zero FP16 quantization error. */
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 && !defined(__HIP__) && !defined(__HIP_PLATFORM_AMD__)
#define CC_FA_BODY 1
#else
#define CC_FA_BODY 0
#endif
#if CC_FA_BODY
__device__ __forceinline__ unsigned cc_fa_swz(int r, int c) { return (unsigned)(r * 256 + ((c ^ (r & 7)) << 4)); }
__device__ __forceinline__ unsigned cc_fa_h2u(__half2 h) { return *reinterpret_cast<unsigned *>(&h); }
__device__ __forceinline__ float cc_fa_clamp(float f) { return f != f ? f : fminf(fmaxf(f, -65504.f), 65504.f); }
__device__ __forceinline__ uint4 cc_fa_f8_to_h8(const float4 a, const float4 b) {
    return make_uint4(
        cc_fa_h2u(__floats2half2_rn(cc_fa_clamp(a.x), cc_fa_clamp(a.y))),
        cc_fa_h2u(__floats2half2_rn(cc_fa_clamp(a.z), cc_fa_clamp(a.w))),
        cc_fa_h2u(__floats2half2_rn(cc_fa_clamp(b.x), cc_fa_clamp(b.y))),
        cc_fa_h2u(__floats2half2_rn(cc_fa_clamp(b.z), cc_fa_clamp(b.w)))
    );
}
__device__ __forceinline__ void cc_fa_ldm4(unsigned (&r)[4], unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
__device__ __forceinline__ void cc_fa_ldm4_trans(unsigned (&r)[4], unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
__device__ __forceinline__ void cc_fa_mma(float (&c)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
#endif
__global__ static void __launch_bounds__(128, 2) ccg_relattn_flash_hd128_k(
        const float *__restrict__ q, const float *__restrict__ kc, const float *__restrict__ vc,
        float *__restrict__ o, const float *__restrict__ kvs, const float *__restrict__ r,
        const float *__restrict__ relp, const float *__restrict__ tau, CcRelAttn p) {
#if CC_FA_BODY
    __shared__ __align__(128) unsigned char kv_sm[2 * 64 * 256];   /* K_sm (16 KB) + V_sm (16 KB) */
    __shared__ __align__(16) float R_sm[64][16];                    /* 4 KB exact FP32 */
    __shared__ __align__(16) float relp_s[128][17];                 /* 8.5 KB exact FP32 (padded stride 17) */
    unsigned char *K_sm = kv_sm;
    unsigned char *V_sm = kv_sm + 64 * 256;
    const unsigned ksm0 = (unsigned)__cvta_generic_to_shared(K_sm);
    const unsigned vsm0 = (unsigned)__cvta_generic_to_shared(V_sm);

    const int h = (int)blockIdx.x;
    const int q_start = (int)blockIdx.y * 64;
    if (q_start >= p.S) return;
    const int kvh = h / (p.H / p.KVH);
    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31, g = lane >> 2, tig = lane & 3;
    const int dr = p.d_rel;

    #pragma unroll
    for (int idx = tid; idx < 64 * 16; idx += 128) {
        const int rq = idx >> 4, c = idx & 15, sq = q_start + rq;
        uint4 h8 = make_uint4(0u, 0u, 0u, 0u);
        if (sq < p.S) {
            const float4 *qp = reinterpret_cast<const float4 *>(q + p.q_off + (long)sq * p.q_row + h * 128) + 2 * c;
            h8 = cc_fa_f8_to_h8(__ldg(qp), __ldg(qp + 1));
        }
        *reinterpret_cast<uint4 *>(K_sm + cc_fa_swz(rq, c)) = h8;
    }
    if (dr > 0) {
        for (int idx = tid; idx < 64 * dr; idx += 128) {
            const int rq = idx / dr, j = idx - rq * dr, sq = q_start + rq;
            R_sm[rq][j] = (sq < p.S) ? __ldg(r + p.r_off + (long)sq * p.r_row + h * dr + j) : 0.f;
        }
    }
    __syncthreads();

    unsigned qA[8][4];
    const int ar = lane & 15, ac = lane >> 4;
    #pragma unroll
    for (int kk = 0; kk < 8; kk++) cc_fa_ldm4(qA[kk], ksm0 + cc_fa_swz(warp * 16 + ar, 2 * kk + ac));

    const int r0 = warp * 16 + g, r1 = r0 + 8;
    const int sq0 = q_start + r0, sq1 = q_start + r1;
    const bool valid0 = sq0 < p.S, valid1 = sq1 < p.S;
    const int qpos0 = p.pos_base + sq0, qpos1 = p.pos_base + sq1;
    const int wmin0 = p.window > 0 ? max(0, qpos0 - p.window + 1) : 0;
    const int wmin1 = p.window > 0 ? max(0, qpos1 - p.window + 1) : 0;
    const float ts0 = valid0 ? __ldg(tau + p.tau_off + sq0) : 1.f;
    const float ts1 = valid1 ? __ldg(tau + p.tau_off + sq1) : 1.f;
    float r0_f[16], r1_f[16];
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        r0_f[j] = (j < dr) ? R_sm[r0][j] : 0.f;
        r1_f[j] = (j < dr) ? R_sm[r1][j] : 0.f;
    }

    float m0 = (p.has_sink && valid0) ? __ldg(tau + p.sink_off + sq0 * p.H + h) : -3.0e38f;
    float m1 = (p.has_sink && valid1) ? __ldg(tau + p.sink_off + sq1 * p.H + h) : -3.0e38f;
    float l0 = p.has_sink ? 1.f : 0.f;
    float l1 = p.has_sink ? 1.f : 0.f;
    float O[16][4];
    #pragma unroll
    for (int nd = 0; nd < 16; nd++) { O[nd][0] = 0.f; O[nd][1] = 0.f; O[nd][2] = 0.f; O[nd][3] = 0.f; }
    __syncthreads();

    const int q_end = min(q_start + 63, p.S - 1);
    const int t_min = p.window > 0 ? max(0, p.pos_base + q_start - p.window + 1) : 0;
    const int t_max = p.pos_base + q_end;
    const int bn = (lane & 7) + ((lane >> 4) << 3), bc = (lane >> 3) & 1;
    const int vk_off = lane & 15, vn_off = lane >> 4;

    for (int tb = t_min; tb <= t_max; tb += 64) {
        #pragma unroll
        for (int idx = tid; idx < 64 * 16; idx += 128) {
            const int rk = idx >> 4, c = idx & 15, t = tb + rk;
            uint4 kh8 = make_uint4(0u, 0u, 0u, 0u), vh8 = make_uint4(0u, 0u, 0u, 0u);
            if (t <= t_max) {
                const float *kr, *vr;
                if (t < p.pos_base) {
                    const long off = ((long)kvh * p.cap + t % p.cap) * 128;
                    kr = kc + p.k_off + off;
                    vr = vc + p.v_off + off;
                } else {
                    const long off = (long)(t - p.pos_base) * p.kv_row + kvh * 128;
                    kr = kvs + p.ks_off + off;
                    vr = kvs + p.vs_off + off;
                }
                const float4 *kp4 = reinterpret_cast<const float4 *>(kr) + 2 * c;
                const float4 *vp4 = reinterpret_cast<const float4 *>(vr) + 2 * c;
                kh8 = cc_fa_f8_to_h8(__ldg(kp4), __ldg(kp4 + 1));
                vh8 = cc_fa_f8_to_h8(__ldg(vp4), __ldg(vp4 + 1));
            }
            const unsigned sw = cc_fa_swz(rk, c);
            *reinterpret_cast<uint4 *>(K_sm + sw) = kh8;
            *reinterpret_cast<uint4 *>(V_sm + sw) = vh8;
        }

        const int d0 = (p.pos_base + q_start) - tb - 63;
        const bool use_rel = (dr > 0) && (d0 < p.ext) && (d0 + 126 >= 0);
        if (use_rel) {
            const int dist = d0 + tid;
            if (dist >= 0 && dist < p.ext) {
                #pragma unroll
                for (int j = 0; j < 16; j++)
                    if (j < dr) relp_s[tid][j] = __ldg(relp + p.relp_off + (long)j * p.ext + dist);
            } else {
                #pragma unroll
                for (int j = 0; j < 16; j++) if (j < dr) relp_s[tid][j] = 0.f;
            }
        }
        __syncthreads();

        float S[8][4];
        #pragma unroll
        for (int j = 0; j < 8; j++) { S[j][0] = 0.f; S[j][1] = 0.f; S[j][2] = 0.f; S[j][3] = 0.f; }
        #pragma unroll
        for (int kk = 0; kk < 8; kk++) {
            #pragma unroll
            for (int jp = 0; jp < 4; jp++) {
                unsigned kB[4];
                cc_fa_ldm4(kB, ksm0 + cc_fa_swz(jp * 16 + bn, 2 * kk + bc));
                cc_fa_mma(S[2 * jp],     qA[kk], kB[0], kB[1]);
                cc_fa_mma(S[2 * jp + 1], qA[kk], kB[2], kB[3]);
            }
        }

        float tm0 = -3.0e38f, tm1 = -3.0e38f;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            #pragma unroll
            for (int c = 0; c < 2; c++) {
                const int rk = j * 8 + tig * 2 + c;
                const int t = tb + rk;
                if (valid0 && t >= wmin0 && t <= qpos0) {
                    float b0 = 0.f;
                    if (use_rel && (qpos0 - t) < p.ext) {
                        const int ridx = r0 - rk + 63;
                        #pragma unroll
                        for (int jj = 0; jj < 16; jj++) if (jj < dr) b0 += r0_f[jj] * relp_s[ridx][jj];
                    }
                    const float sv = ts0 * (S[j][c] * p.scale + b0);
                    S[j][c] = sv;
                    tm0 = fmaxf(tm0, sv);
                } else {
                    S[j][c] = -3.0e38f;
                }
                if (valid1 && t >= wmin1 && t <= qpos1) {
                    float b1 = 0.f;
                    if (use_rel && (qpos1 - t) < p.ext) {
                        const int ridx = r1 - rk + 63;
                        #pragma unroll
                        for (int jj = 0; jj < 16; jj++) if (jj < dr) b1 += r1_f[jj] * relp_s[ridx][jj];
                    }
                    const float sv = ts1 * (S[j][2 + c] * p.scale + b1);
                    S[j][2 + c] = sv;
                    tm1 = fmaxf(tm1, sv);
                } else {
                    S[j][2 + c] = -3.0e38f;
                }
            }
        }
        tm0 = fmaxf(tm0, __shfl_xor_sync(0xffffffffu, tm0, 1));
        tm0 = fmaxf(tm0, __shfl_xor_sync(0xffffffffu, tm0, 2));
        tm1 = fmaxf(tm1, __shfl_xor_sync(0xffffffffu, tm1, 1));
        tm1 = fmaxf(tm1, __shfl_xor_sync(0xffffffffu, tm1, 2));

        const float mn0 = fmaxf(m0, tm0);
        const float mn1 = fmaxf(m1, tm1);
        const float corr0 = (m0 > -1.0e30f) ? expf(m0 - mn0) : 0.f;
        const float corr1 = (m1 > -1.0e30f) ? expf(m1 - mn1) : 0.f;
        #pragma unroll
        for (int nd = 0; nd < 16; nd++) {
            O[nd][0] *= corr0; O[nd][1] *= corr0;
            O[nd][2] *= corr1; O[nd][3] *= corr1;
        }

        float ps0 = 0.f, ps1 = 0.f;
        unsigned pA[4][4];
        #pragma unroll
        for (int kk = 0; kk < 4; kk++) {
            const int j0 = 2 * kk, j1 = j0 + 1;
            const float p00 = (S[j0][0] > -1.0e30f) ? expf(S[j0][0] - mn0) : 0.f;
            const float p01 = (S[j0][1] > -1.0e30f) ? expf(S[j0][1] - mn0) : 0.f;
            const float p10 = (S[j0][2] > -1.0e30f) ? expf(S[j0][2] - mn1) : 0.f;
            const float p11 = (S[j0][3] > -1.0e30f) ? expf(S[j0][3] - mn1) : 0.f;
            const float q00 = (S[j1][0] > -1.0e30f) ? expf(S[j1][0] - mn0) : 0.f;
            const float q01 = (S[j1][1] > -1.0e30f) ? expf(S[j1][1] - mn0) : 0.f;
            const float q10 = (S[j1][2] > -1.0e30f) ? expf(S[j1][2] - mn1) : 0.f;
            const float q11 = (S[j1][3] > -1.0e30f) ? expf(S[j1][3] - mn1) : 0.f;
            ps0 += p00 + p01 + q00 + q01;
            ps1 += p10 + p11 + q10 + q11;
            pA[kk][0] = cc_fa_h2u(__floats2half2_rn(p00, p01));
            pA[kk][1] = cc_fa_h2u(__floats2half2_rn(p10, p11));
            pA[kk][2] = cc_fa_h2u(__floats2half2_rn(q00, q01));
            pA[kk][3] = cc_fa_h2u(__floats2half2_rn(q10, q11));
        }
        ps0 += __shfl_xor_sync(0xffffffffu, ps0, 1);
        ps0 += __shfl_xor_sync(0xffffffffu, ps0, 2);
        ps1 += __shfl_xor_sync(0xffffffffu, ps1, 1);
        ps1 += __shfl_xor_sync(0xffffffffu, ps1, 2);
        l0 = l0 * corr0 + ps0; m0 = mn0;
        l1 = l1 * corr1 + ps1; m1 = mn1;

        #pragma unroll
        for (int kk = 0; kk < 4; kk++) {
            #pragma unroll
            for (int nd = 0; nd < 8; nd++) {
                unsigned vB[4];
                cc_fa_ldm4_trans(vB, vsm0 + cc_fa_swz(kk * 16 + vk_off, 2 * nd + vn_off));
                cc_fa_mma(O[2 * nd],     pA[kk], vB[0], vB[1]);
                cc_fa_mma(O[2 * nd + 1], pA[kk], vB[2], vB[3]);
            }
        }
        __syncthreads();
    }

    const float inv0 = (l0 > 0.f) ? (1.f / l0) : 0.f;
    const float inv1 = (l1 > 0.f) ? (1.f / l1) : 0.f;
    if (valid0) {
        float *o0 = o + p.o_off + (long)sq0 * p.o_row + h * 128;
        #pragma unroll
        for (int nd = 0; nd < 16; nd++)
            *reinterpret_cast<float2 *>(o0 + nd * 8 + tig * 2) = make_float2(O[nd][0] * inv0, O[nd][1] * inv0);
    }
    if (valid1) {
        float *o1 = o + p.o_off + (long)sq1 * p.o_row + h * 128;
        #pragma unroll
        for (int nd = 0; nd < 16; nd++)
            *reinterpret_cast<float2 *>(o1 + nd * 8 + tig * 2) = make_float2(O[nd][2] * inv1, O[nd][3] * inv1);
    }
#else
    (void)q; (void)kc; (void)vc; (void)o; (void)kvs; (void)r; (void)relp; (void)tau; (void)p;
#endif
}

/* tile b0 (128 positions from b0) of a (head, row)'s attention into its running max m, sum l and
 * weighted V (acc0: dims tid, acc1: tid + 128): chain_relattn.comp's online softmax step */
__device__ __forceinline__ void ccg_relattn_tile(const float *__restrict__ kc, const float *__restrict__ vc, const float *__restrict__ kvs,
        const float *__restrict__ relp, const CcRelAttn &p, const float *qs, const float *rs, float *pe, float *red, int *pt,
        int kvh, int qpos, int t0, int n, float ts, int b0, int tid, float &m, float &l, float &acc0, float &acc1) {
    const int i = b0 + tid;
    float sv = -3.0e38f;
    if (i < n) {
        const int t = t0 + i;
        pt[tid] = t;
        float a = 0.f;
        if (t < p.pos_base) { const float *kr = kc + p.k_off + ((long)kvh * p.cap + t % p.cap) * p.hd; for (int d = 0; d < p.hd; d++) a += qs[d] * kr[d]; }
        else { const float *kr = kvs + p.ks_off + (long)(t - p.pos_base) * p.kv_row + kvh * p.hd; for (int d = 0; d < p.hd; d++) a += qs[d] * kr[d]; }
        const int dist = qpos - t;
        float bias = 0.f;
        if (dist < p.ext) for (int j = 0; j < p.d_rel; j++) bias += rs[j] * relp[p.relp_off + j * p.ext + dist];
        sv = ts * (a * p.scale + bias);
    }
    red[tid] = sv;
    __syncthreads();
    for (int off = 64; off > 0; off >>= 1) { if (tid < off) red[tid] = fmaxf(red[tid], red[tid + off]); __syncthreads(); }
    const float mn = fmaxf(m, red[0]);
    __syncthreads();
    const float e = i < n ? expf(sv - mn) : 0.f;
    pe[tid] = e; red[tid] = e;
    __syncthreads();
    for (int off = 64; off > 0; off >>= 1) { if (tid < off) red[tid] += red[tid + off]; __syncthreads(); }
    const float corr = expf(m - mn);
    l = l * corr + red[0];
    acc0 *= corr; acc1 *= corr;
    const int cnt = min(128, n - b0), d0 = tid, d1 = tid + 128;
    for (int j = 0; j < cnt; j++) {
        const int t = pt[j];
        const float w = pe[j];
        const float *vr = t < p.pos_base ? vc + p.v_off + ((long)kvh * p.cap + t % p.cap) * p.hd
                                         : kvs + p.vs_off + (long)(t - p.pos_base) * p.kv_row + kvh * p.hd;
        if (d0 < p.hd) acc0 += w * vr[d0];
        if (d1 < p.hd) acc1 += w * vr[d1];
    }
    m = mn;
    __syncthreads();
}
/* Fast coalesced warp-per-token tile for hd==128 (4 warps = 128 threads):
 * each warp loads a full 128-float K row as 32 coalesced float4s. */
__device__ __forceinline__ void ccg_relattn_tile_hd128(
        const float *__restrict__ kc, const float *__restrict__ vc, const float *__restrict__ kvs,
        const float *__restrict__ relp, const CcRelAttn &p, float4 qv, const float *rs,
        float *pe, float *red, int *pt,
        int kvh, int qpos, int t0, int n, float ts, int b0, int tid,
        float &m, float &l, float &acc0) {
    const int warp = tid >> 5, lane = tid & 31;
    const int cnt = min(128, n - b0);
    const int t_start = t0 + b0;
    const int row0 = t_start % p.cap;
    const bool contig = (t_start + cnt <= p.pos_base) && (row0 + cnt <= p.cap);
    const long head_off = (long)kvh * p.cap;
    if (contig) {
        const bool any_rel = (p.d_rel > 0) && (qpos - (t_start + cnt - 1) < p.ext);
        for (int j = warp; j < cnt; j += 4) {
            const float *kr = kc + p.k_off + (head_off + row0 + j) * 128;
            const float4 kv = __ldg(reinterpret_cast<const float4 *>(kr) + lane);
            float a = qv.x * kv.x + qv.y * kv.y + qv.z * kv.z + qv.w * kv.w;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) a += __shfl_xor_sync(0xffffffffu, a, off);
            float bias = 0.f;
            if (any_rel) {
                const int dist = qpos - (t_start + j);
                if (dist < p.ext) {
                    if (p.d_rel <= 32) {
                        bias = lane < p.d_rel ? rs[lane] * __ldg(relp + p.relp_off + lane * p.ext + dist) : 0.f;
                        #pragma unroll
                        for (int off = 16; off > 0; off >>= 1) bias += __shfl_xor_sync(0xffffffffu, bias, off);
                    } else {
                        for (int rj = lane; rj < p.d_rel; rj += 32)
                            bias += rs[rj] * __ldg(relp + p.relp_off + rj * p.ext + dist);
                        #pragma unroll
                        for (int off = 16; off > 0; off >>= 1) bias += __shfl_xor_sync(0xffffffffu, bias, off);
                    }
                }
            }
            if (lane == 0) red[j] = ts * (a * p.scale + bias);
        }
    } else {
        for (int j = warp; j < cnt; j += 4) {
            const int t = t_start + j;
            const float *kr = t < p.pos_base
                ? kc + p.k_off + (head_off + t % p.cap) * 128
                : kvs + p.ks_off + (long)(t - p.pos_base) * p.kv_row + kvh * 128;
            const float4 kv = __ldg(reinterpret_cast<const float4 *>(kr) + lane);
            float a = qv.x * kv.x + qv.y * kv.y + qv.z * kv.z + qv.w * kv.w;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) a += __shfl_xor_sync(0xffffffffu, a, off);
            const int dist = qpos - t;
            float bias = 0.f;
            if (dist < p.ext && p.d_rel > 0) {
                if (p.d_rel <= 32) {
                    bias = lane < p.d_rel ? rs[lane] * __ldg(relp + p.relp_off + lane * p.ext + dist) : 0.f;
                    #pragma unroll
                    for (int off = 16; off > 0; off >>= 1) bias += __shfl_xor_sync(0xffffffffu, bias, off);
                } else {
                    for (int rj = lane; rj < p.d_rel; rj += 32)
                        bias += rs[rj] * __ldg(relp + p.relp_off + rj * p.ext + dist);
                    #pragma unroll
                    for (int off = 16; off > 0; off >>= 1) bias += __shfl_xor_sync(0xffffffffu, bias, off);
                }
            }
            if (lane == 0) {
                red[j] = ts * (a * p.scale + bias);
                pt[j] = t;
            }
        }
    }
    __syncthreads();
    const float sv = tid < cnt ? red[tid] : -3.0e38f;
    float wm = sv;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) wm = fmaxf(wm, __shfl_xor_sync(0xffffffffu, wm, off));
    __shared__ float wred[4];
    if (lane == 0) wred[warp] = wm;
    __syncthreads();
    float tmax = fmaxf(fmaxf(wred[0], wred[1]), fmaxf(wred[2], wred[3]));
    const float mn = fmaxf(m, tmax);
    const float e = tid < cnt ? expf(sv - mn) : 0.f;
    pe[tid] = e;
    float ws = e;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) ws += __shfl_xor_sync(0xffffffffu, ws, off);
    if (lane == 0) wred[warp] = ws;
    __syncthreads();
    const float sum_e = wred[0] + wred[1] + wred[2] + wred[3];
    const float corr = expf(m - mn);
    l = l * corr + sum_e;
    acc0 *= corr;
    if (contig) {
        const float *vp = vc + p.v_off + (head_off + row0) * 128 + tid;
        int j = 0;
        for (; j + 3 < cnt; j += 4) {
            const float w0 = pe[j], w1 = pe[j + 1], w2 = pe[j + 2], w3 = pe[j + 3];
            const float v0 = __ldg(vp + (long)j * 128);
            const float v1 = __ldg(vp + (long)(j + 1) * 128);
            const float v2 = __ldg(vp + (long)(j + 2) * 128);
            const float v3 = __ldg(vp + (long)(j + 3) * 128);
            acc0 += w0 * v0;
            acc0 += w1 * v1;
            acc0 += w2 * v2;
            acc0 += w3 * v3;
        }
        for (; j < cnt; j++) acc0 += pe[j] * __ldg(vp + (long)j * 128);
    } else {
        for (int j = 0; j < cnt; j++) {
            const int t = pt[j];
            const float w = pe[j];
            const float *vr = t < p.pos_base
                ? vc + p.v_off + (head_off + t % p.cap) * 128
                : kvs + p.vs_off + (long)(t - p.pos_base) * p.kv_row + kvh * 128;
            acc0 += w * __ldg(vr + tid);
        }
    }
    m = mn;
    __syncthreads();
}
/* one block of 128 per (head, row): chain_relattn.comp's online softmax over tiles of 128 positions.
 * part (the split, a decode step's rows): block z takes tiles z, z + gridDim.z, ... so contexts > gridDim.z * 128
 * never truncate; ccg_relattn_join_k joins the partial tiles. */
__global__ static void ccg_relattn_k(const float *q, const float *kc, const float *vc, float *o, const float *kvs,
                                     const float *r, const float *relp, const float *tau, CcRelAttn p, float *part,
                                     const int *pb_ptr) {
    if (pb_ptr) p.pos_base = *pb_ptr;
    const int h = (int)blockIdx.x, s = (int)blockIdx.y, tid = (int)threadIdx.x;
    const int kvh = h / (p.H / p.KVH);
    const int qpos = p.pos_base + s;
    const int t0 = p.window > 0 ? max(0, qpos - p.window + 1) : 0;
    const int n = qpos - t0 + 1;
    if (part && (int)blockIdx.z * 128 >= n) {
        float *pp = part + (((long)s * p.H + h) * gridDim.z + (int)blockIdx.z) * (p.hd + 2);
        if (tid == 0) { pp[0] = -3.0e38f; pp[1] = 0.f; }
        return;
    }
    __shared__ float qs[256], rs[64], pe[128], red[128];
    __shared__ int pt[128];
    const int qb = p.q_off + s * p.q_row + h * p.hd;
    const float ts = tau[p.tau_off + s];
    for (int j = tid; j < p.d_rel; j += 128) rs[j] = r[p.r_off + s * p.r_row + h * p.d_rel + j];
    const bool fast128 = (p.hd == 128) && ((p.q_off | p.q_row | p.k_off | p.ks_off | p.kv_row) & 3) == 0;
    if (fast128) {
        const float4 qv = __ldg(reinterpret_cast<const float4 *>(q + qb) + (tid & 31));
        __syncthreads();
        float m = -3.0e38f, l = 0.f, acc0 = 0.f;
        if (part) {
            const int z = (int)blockIdx.z, stride = (int)gridDim.z * 128;
            for (int b0 = z * 128; b0 < n; b0 += stride)
                ccg_relattn_tile_hd128(kc, vc, kvs, relp, p, qv, rs, pe, red, pt, kvh, qpos, t0, n, ts, b0, tid, m, l, acc0);
            float *pp = part + (((long)s * p.H + h) * gridDim.z + z) * 130;
            if (tid == 0) { pp[0] = m; pp[1] = l; }
            pp[2 + tid] = acc0;
            return;
        }
        if (p.has_sink) { m = tau[p.sink_off + s * p.H + h]; l = 1.f; }
        for (int b0 = 0; b0 < n; b0 += 128)
            ccg_relattn_tile_hd128(kc, vc, kvs, relp, p, qv, rs, pe, red, pt, kvh, qpos, t0, n, ts, b0, tid, m, l, acc0);
        const float inv = l > 0.f ? 1.f / l : 0.f;
        o[p.o_off + s * p.o_row + h * 128 + tid] = acc0 * inv;
        return;
    }
    for (int d = tid; d < p.hd; d += 128) qs[d] = q[qb + d];
    __syncthreads();
    float m = -3.0e38f, l = 0.f, acc0 = 0.f, acc1 = 0.f;
    if (part) {
        const int z = (int)blockIdx.z, stride = (int)gridDim.z * 128;
        for (int b0 = z * 128; b0 < n; b0 += stride)
            ccg_relattn_tile(kc, vc, kvs, relp, p, qs, rs, pe, red, pt, kvh, qpos, t0, n, ts, b0, tid, m, l, acc0, acc1);
        float *pp = part + (((long)s * p.H + h) * gridDim.z + z) * (p.hd + 2);
        if (tid == 0) { pp[0] = m; pp[1] = l; }
        if (tid < p.hd) pp[2 + tid] = acc0;
        if (tid + 128 < p.hd) pp[2 + tid + 128] = acc1;
        return;
    }
    if (p.has_sink) { m = tau[p.sink_off + s * p.H + h]; l = 1.f; }
    for (int b0 = 0; b0 < n; b0 += 128)
        ccg_relattn_tile(kc, vc, kvs, relp, p, qs, rs, pe, red, pt, kvh, qpos, t0, n, ts, b0, tid, m, l, acc0, acc1);
    const float inv = l > 0.f ? 1.f / l : 0.f;
    const int ob = p.o_off + s * p.o_row + h * p.hd;
    if (tid < p.hd) o[ob + tid] = acc0 * inv;
    if (tid + 128 < p.hd) o[ob + tid + 128] = acc1 * inv;
}
/* the split's join: a (head, row)'s nc tiles (and its sink) in tile order into its output */
__global__ static void ccg_relattn_join_k(const float *part, const float *tau, float *o, CcRelAttn p, int nc) {
    const int h = (int)blockIdx.x, s = (int)blockIdx.y, tid = (int)threadIdx.x;
    const float *pp = part + ((long)s * p.H + h) * nc * (p.hd + 2);
    float M = p.has_sink ? tau[p.sink_off + s * p.H + h] : -3.0e38f;
    for (int z = 0; z < nc; z++) if (pp[(long)z * (p.hd + 2) + 1] > 0.f) M = fmaxf(M, pp[(long)z * (p.hd + 2)]);
    float L = p.has_sink ? expf(tau[p.sink_off + s * p.H + h] - M) : 0.f, acc = 0.f;
    for (int z = 0; z < nc; z++) {
        const float *pz = pp + (long)z * (p.hd + 2);
        if (!(pz[1] > 0.f)) continue;
        const float w = expf(pz[0] - M);
        L += pz[1] * w;
        if (tid < p.hd) acc += pz[2 + tid] * w;
    }
    if (tid < p.hd) o[p.o_off + s * p.o_row + h * p.hd + tid] = L > 0.f ? acc / L : 0.f;
}

/* Fast single-pass vectorized float4 ink2_mix (D <= 4096): caches each vector's float4 in registers,
 * keeping 1.f / sqrtf() and sc[e] * (v * rr[e]) arithmetic matching ccg_ink2_mix_k. */
template<int MAXE>
__global__ static void ccg_ink2_mix_f4_k(const float *__restrict__ x, const float *__restrict__ blk,
                                         const float *__restrict__ prm, float *__restrict__ y, CcInk2Mix p) {
    const int s = (int)(blockIdx.x + blockIdx.y * gridDim.x);
    if (s >= p.S) return;
    const int tid = (int)threadIdx.x, wp = tid >> 5, ln = tid & 31, nwp = (int)blockDim.x >> 5;
    const int D4 = p.D >> 2;
    const bool act = tid < D4;
    const float4 *xb4 = reinterpret_cast<const float4 *>(x + p.x_off + (long)s * p.x_row);
    const float4 *bb4 = blk ? reinterpret_cast<const float4 *>(blk + p.b_off + (long)s * p.b_row) : xb4;
    const float4 *q4  = reinterpret_cast<const float4 *>(prm + p.w_off);
    const float4 qv = act ? __ldg(q4 + tid) : make_float4(0.f, 0.f, 0.f, 0.f);
    float4 vreg[MAXE];
    __shared__ float pms[32 * MAXE], pdt[32 * MAXE], sc[MAXE], rr[MAXE];
    const int nvec = p.nb + 1;
    #pragma unroll
    for (int e = 0; e < MAXE; e++) {
        if (e >= nvec) break;
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (act) v = __ldg(e < p.nb ? (bb4 + (long)e * D4 + tid) : (xb4 + tid));
        vreg[e] = v;
        float a = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        float b = v.x * qv.x + v.y * qv.y + v.z * qv.z + v.w * qv.w;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            a += __shfl_xor_sync(0xffffffffu, a, off);
            b += __shfl_xor_sync(0xffffffffu, b, off);
        }
        if (ln == 0) { pms[wp * MAXE + e] = a; pdt[wp * MAXE + e] = b; }
    }
    __syncthreads();
    if (tid < nvec) {
        float a = 0.f, b = 0.f;
        for (int g = 0; g < nwp; g++) { a += pms[g * MAXE + tid]; b += pdt[g * MAXE + tid]; }
        const float r = 1.f / sqrtf(a / (float)p.D + p.eps);
        rr[tid] = r; sc[tid] = (b * r) * p.scale;
    }
    __syncthreads();
    if (tid == 0) {
        float m = sc[0];
        for (int e = 1; e < nvec; e++) if (sc[e] > m) m = sc[e];
        float sum = 0.f;
        for (int e = 0; e < nvec; e++) { sc[e] = expf(sc[e] - m); sum += sc[e]; }
        for (int e = 0; e < nvec; e++) sc[e] /= sum;
    }
    __syncthreads();
    if (act) {
        float4 out = make_float4(0.f, 0.f, 0.f, 0.f);
        #pragma unroll
        for (int e = 0; e < MAXE; e++) {
            if (e >= nvec) break;
            const float s_e = sc[e], r_e = rr[e];
            out.x += s_e * (vreg[e].x * r_e); out.y += s_e * (vreg[e].y * r_e);
            out.z += s_e * (vreg[e].z * r_e); out.w += s_e * (vreg[e].w * r_e);
        }
        reinterpret_cast<float4 *>(y + p.y_off + (long)s * p.y_row)[tid] = out;
    }
}

/* one block of 256 per row: every vector's sums in one pass, the warps' joined in order */
__global__ static void ccg_ink2_mix_k(const float *x, const float *blk, const float *prm, float *y, CcInk2Mix p) {
    const int s = (int)(blockIdx.x + blockIdx.y * gridDim.x), tid = (int)threadIdx.x, wp = tid >> 5, ln = tid & 31;
    if (s >= p.S) return;
    __shared__ float pms[8 * 16], pdt[8 * 16], sc[16], rr[16];
    const float *xb = x + p.x_off + (long)s * p.x_row, *bb = blk ? blk + p.b_off + (long)s * p.b_row : xb;
    const float *q = prm + p.w_off;
    float ms[16], dt[16];
#pragma unroll
    for (int e = 0; e < 16; e++) { ms[e] = 0.f; dt[e] = 0.f; }
    for (int d = tid; d < p.D; d += 256) {
        const float qv = q[d];
#pragma unroll
        for (int e = 0; e < 16; e++) {
            if (e > p.nb) break;
            const float v = e < p.nb ? bb[(long)e * p.D + d] : xb[d];
            ms[e] += v * v; dt[e] += v * qv;
        }
    }
#pragma unroll
    for (int e = 0; e < 16; e++) {
        if (e > p.nb) break;
        float a = ms[e], b = dt[e];
        for (int off = 16; off > 0; off >>= 1) { a += __shfl_xor_sync(0xffffffffu, a, off); b += __shfl_xor_sync(0xffffffffu, b, off); }
        if (ln == 0) { pms[wp * 16 + e] = a; pdt[wp * 16 + e] = b; }
    }
    __syncthreads();
    if (tid <= p.nb) {
        float a = 0.f, b = 0.f;
        for (int g = 0; g < 8; g++) { a += pms[g * 16 + tid]; b += pdt[g * 16 + tid]; }
        const float r = 1.f / sqrtf(a / (float)p.D + p.eps);
        rr[tid] = r; sc[tid] = (b * r) * p.scale;
    }
    __syncthreads();
    if (tid == 0) {
        float m = sc[0];
        for (int e = 1; e <= p.nb; e++) if (sc[e] > m) m = sc[e];
        float sum = 0.f;
        for (int e = 0; e <= p.nb; e++) { sc[e] = expf(sc[e] - m); sum += sc[e]; }
        for (int e = 0; e <= p.nb; e++) sc[e] /= sum;
    }
    __syncthreads();
    float *yb = y + p.y_off + (long)s * p.y_row;
    for (int d = tid; d < p.D; d += 256) {
        float a = 0.f;
        for (int e = 0; e < p.nb; e++) a += sc[e] * (bb[(long)e * p.D + d] * rr[e]);
        yb[d] = a + sc[p.nb] * (xb[d] * rr[p.nb]);
    }
}

/* Fast warp-shuffle clip for n == 128 (8 segments/block, 1 warp/segment, float4) */
__global__ static void ccg_ink2_clip_w128_k(float *__restrict__ x, const float *__restrict__ w, CcInk2Clip p) {
    const int warp_global = ((int)(blockIdx.x + blockIdx.y * gridDim.x) << 3) + ((int)threadIdx.x >> 5);
    if (warp_global >= p.nseg) return;
    const int lane = (int)threadIdx.x & 31;
    const int j = warp_global % p.per_row;
    float4 *v4 = reinterpret_cast<float4 *>(x + p.x_off + (long)(warp_global / p.per_row) * p.x_row + (long)j * 128);
    float4 a = v4[lane];
    if (p.has_w) {
        const float4 sw = __ldg(reinterpret_cast<const float4 *>(w + p.w_off + (long)j * 128) + lane);
        a.x *= sw.x; a.y *= sw.y; a.z *= sw.z; a.w *= sw.w;
    }
    float ss = a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, off);
    const float f = p.thr / (sqrtf(ss) + 1e-6f);
    if (f < 1.f) { a.x *= f; a.y *= f; a.z *= f; a.w *= f; }
    v4[lane] = a;
}
/* Fast half-warp clip for n == 16 (16 segments/block, 16 lanes/segment) */
__global__ static void ccg_ink2_clip_w16_k(float *__restrict__ x, const float *__restrict__ w, CcInk2Clip p) {
    const int seg = ((int)(blockIdx.x + blockIdx.y * gridDim.x) << 4) + ((int)threadIdx.x >> 4);
    if (seg >= p.nseg) return;
    const int lane16 = (int)threadIdx.x & 15;
    const int j = seg % p.per_row;
    float *vp = x + p.x_off + (long)(seg / p.per_row) * p.x_row + (long)j * 16;
    float a = vp[lane16];
    if (p.has_w) a *= __ldg(w + p.w_off + (long)j * 16 + lane16);
    float ss = a * a;
    #pragma unroll
    for (int off = 8; off > 0; off >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, off, 16);
    const float f = p.thr / (sqrtf(ss) + 1e-6f);
    vp[lane16] = f < 1.f ? a * f : a;
}

/* one block of 256 per segment */
__global__ static void ccg_ink2_clip_k(float *x, const float *w, CcInk2Clip p) {
    const int g = (int)(blockIdx.x + blockIdx.y * gridDim.x), tid = (int)threadIdx.x;
    if (g >= p.nseg) return;
    __shared__ float red[256];
    const int j = g % p.per_row;
    float *v = x + p.x_off + (long)(g / p.per_row) * p.x_row + (long)j * p.n;
    const float *sw = p.has_w ? w + p.w_off + (long)j * p.n : nullptr;
    float ss = 0.f;
    for (int i = tid; i < p.n; i += 256) { float a = v[i]; if (sw) a *= sw[i]; ss += a * a; }
    red[tid] = ss;
    __syncthreads();
    for (int off = 128; off > 0; off >>= 1) { if (tid < off) red[tid] += red[tid + off]; __syncthreads(); }
    const float f = p.thr / (sqrtf(red[0]) + 1e-6f);
    for (int i = tid; i < p.n; i += 256) { float a = v[i]; if (sw) a *= sw[i]; v[i] = f < 1.f ? a * f : a; }
}

/* 2D parallel dynamic short convolution for prefill (S >= 16): bit-exact to ccg_ink2_dconv_k */
__global__ static void ccg_ink2_dconv_rows_k(const float *__restrict__ x, const float *__restrict__ dyn,
                                             const float *__restrict__ st, const float *__restrict__ ring,
                                             float *__restrict__ tmp, CcInk2Dconv p) {
    const int c = (int)blockIdx.x * 128 + (int)threadIdx.x;
    const int s0 = (int)blockIdx.y * 16;
    if (c >= p.C || s0 >= p.S) return;
    const int K = p.K, nh = K - 1, h = c / p.hd;
    const int s1 = min(s0 + 16, p.S);
    float sw[9];
#pragma unroll
    for (int lag = 0; lag < 9; lag++) sw[lag] = lag < K ? __ldg(st + p.st_off + (long)lag * p.C + c) : 0.f;
    float hist[8];
#pragma unroll
    for (int k = 0; k < 8; k++) {
        if (k < nh) {
            const int src_s = s0 - (nh - k);
            hist[k] = src_s < 0 ? __ldg(ring + p.ring_off + (long)c * nh + (nh + src_s))
                                : __ldg(x + p.x_off + (long)src_s * p.x_row + c);
        } else hist[k] = 0.f;
    }
    for (int s = s0; s < s1; s++) {
        const float cur = __ldg(x + p.x_off + (long)s * p.x_row + c);
        const float *dw = dyn + p.d_off + (long)s * p.d_row + h * K;
        float acc = cur * (sw[0] + __ldg(dw));
#pragma unroll
        for (int lag = 1; lag < 9; lag++) if (lag < K) acc += hist[nh - lag] * (sw[lag] + __ldg(dw + lag));
        tmp[(long)s * p.C + c] = (acc + cur) / p.fc;
#pragma unroll
        for (int k = 0; k < 7; k++) if (k + 1 < nh) hist[k] = hist[k + 1];
#pragma unroll
        for (int k = 0; k < 8; k++) if (k == nh - 1) hist[k] = cur;
    }
}

/* one thread per channel over the step's rows */
__global__ static void ccg_ink2_dconv_k(float *x, const float *dyn, const float *st, float *ring, CcInk2Dconv p) {
    const long c = (long)(blockIdx.x + blockIdx.y * gridDim.x) * 256 + threadIdx.x;
    if (c >= p.C) return;
    const int K = p.K, nh = K - 1, h = (int)(c / p.hd);
    float hist[8];
#pragma unroll
    for (int k = 0; k < 8; k++) hist[k] = k < nh ? ring[p.ring_off + c * nh + k] : 0.f;
    for (int s = 0; s < p.S; s++) {
        const long xi = p.x_off + (long)s * p.x_row + c;
        const float cur = x[xi];
        const float *dw = dyn + p.d_off + (long)s * p.d_row + h * K;
        float acc = cur * (st[p.st_off + c] + dw[0]);
#pragma unroll
        for (int lag = 1; lag < 9; lag++) if (lag < K) acc += hist[nh - lag] * (st[p.st_off + (long)lag * p.C + c] + dw[lag]);
        x[xi] = (acc + cur) / p.fc;
#pragma unroll
        for (int k = 0; k < 7; k++) if (k + 1 < nh) hist[k] = hist[k + 1];
#pragma unroll
        for (int k = 0; k < 8; k++) if (k == nh - 1) hist[k] = cur;
    }
#pragma unroll
    for (int k = 0; k < 8; k++) if (k < nh) ring[p.ring_off + c * nh + k] = hist[k];
}

static int cc_sconv(CcBuf *x, CcBuf *w, CcBuf *ring, const CcSconv *p) {
    CcCtx *c = live();
    if (!c || !p || !x || (p->mode == 0 && (p->CK < 1 || p->CK > 9 || !w || !ring))) return 0;
    const long n = p->mode == 0 ? p->C : p->n;
    if (n <= 0) return 1;
    if (p->mode == 0 && p->S >= 16 && !c->capturing) {
        const size_t need = (size_t)p->S * (size_t)p->C * sizeof(float);
        float *tmp = nullptr;
        if (cudaMallocAsync((void **)&tmp, need, c->stream) == cudaSuccess) {
            dim3 gr((unsigned)((p->C + 127) / 128), (unsigned)((p->S + 15) / 16));
            coli_launch(ccg_sconv_rows_k, gr, 128, 0, c->stream, fp(x), fp(w), fp(ring), tmp, *p);
            const int nh = p->CK - 1;
            if (((p->C | p->x_off | p->x_row) & 3) == 0) {
                dim3 gc; cg_grid(((long)p->S * (p->C >> 2) + 255) / 256, &gc);
                coli_launch(ccg_conv_commit_f4_k, gc, 256, 0, c->stream, fpw(x), fpw(ring), tmp, p->S, p->C, nh, p->x_off, p->x_row, p->ring_off);
            } else {
                dim3 gc; cg_grid(((long)p->S * p->C + 255) / 256, &gc);
                coli_launch(ccg_conv_commit_k, gc, 256, 0, c->stream, fpw(x), fpw(ring), tmp, p->S, p->C, nh, p->x_off, p->x_row, p->ring_off);
            }
            (void)cudaFreeAsync(tmp, c->stream);
            return launched(c, "sconv 2d");
        }
        (void)cudaGetLastError();
    }
    dim3 g; cg_grid((n + 63) / 64, &g);
    coli_launch(ccg_sconv_k, g, 64, 0, c->stream, fpw(x), p->mode == 0 ? fp(w) : fp(x), p->mode == 0 ? fpw(ring) : fpw(x), *p);
    return launched(c, "sconv");
}
static int cc_relattn(CcBuf *q, CcBuf *kc, CcBuf *vc, CcBuf *o, CcBuf *kvs, CcBuf *r, CcBuf *relp, CcBuf *tau, const CcRelAttn *p) {
    CcCtx *c = live();
    if (!c || !p || p->hd > 256 || p->hd < 1 || p->d_rel > 64 || p->d_rel < 0 || p->KVH < 1 || p->H % p->KVH || p->cap < 1) return 0;
    if (p->S <= 0) return 1;
#if !(defined(__HIP__) || defined(__HIP_PLATFORM_AMD__))
    if (cc_relattn_flash_on() && c->major >= 8 && p->S >= 16 && p->hd == 128 && p->d_rel <= 16 &&
        ((p->q_off | p->q_row | p->o_off | p->o_row | p->k_off | p->v_off | p->ks_off | p->vs_off | p->kv_row) & 3) == 0) {
        static unsigned fa_attr = 0;
        if (c->device >= 0 && c->device < 32 && !(fa_attr & (1u << c->device))) {
            cudaFuncSetAttribute(ccg_relattn_flash_hd128_k, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
            (void)cudaGetLastError();
            fa_attr |= 1u << c->device;
        }
        dim3 gf((unsigned)p->H, (unsigned)((p->S + 63) / 64));
        coli_launch(ccg_relattn_flash_hd128_k, gf, 128, 0, c->stream, fp(q), fp(kc), fp(vc), fpw(o),
                    fp(kvs), fp(r), fp(relp), fp(tau), *p);
        return launched(c, "relattn flash");
    }
#endif
    /* a decode step's rows (up to 8): each (head, row)'s positions split over blocks of a tile each, then joined
     * (COLI_CUDA_RELATTN_SPLIT=0: a block a (head, row) over every tile). Inkling-2-Mini Q4_K_XL at about 650
     * positions on an RTX PRO 4000: a block a head ran its tiles one after another */
    const int split = cc_relattn_split();
    const int nmax = p->window > 0 ? p->window : p->pos_base + p->S, nc = (nmax + 127) / 128;
    if (split && p->S <= 8 && nc >= 2) {
        float *part = NULL;
        const size_t pb = (size_t)p->S * p->H * nc * (p->hd + 2) * sizeof(float);
        if (cudaMallocAsync((void **)&part, pb, c->stream) == cudaSuccess) {
            coli_launch(ccg_relattn_k, dim3((unsigned)p->H, (unsigned)p->S, (unsigned)nc), 128, 0, c->stream, fp(q), fp(kc), fp(vc), fpw(o),
                        fp(kvs), fp(r), fp(relp), fp(tau), *p, part, (const int *)nullptr);
            coli_launch(ccg_relattn_join_k, dim3((unsigned)p->H, (unsigned)p->S), 256, 0, c->stream, (const float *)part, fp(tau), fpw(o), *p, nc);
            (void)cudaFreeAsync(part, c->stream);
            return launched(c, "relattn split");
        }
        (void)cudaGetLastError();
    }
    coli_launch(ccg_relattn_k, dim3((unsigned)p->H, (unsigned)p->S), 128, 0, c->stream, fp(q), fp(kc), fp(vc), fpw(o), fp(kvs),
                fp(r), fp(relp), fp(tau), *p, (float *)nullptr, (const int *)nullptr);
    return launched(c, "relattn");
}
/* CUDA-Graph-safe relattn + KV scatter reading pos_base from a 1-int device buffer (no host-side pb dependency,
 * no cudaMallocAsync inside capture). */
__global__ static void ccg_ink2_kv_put_k(float *kc, float *vc, const float *kvs, const int *pb_ptr,
                                         int S, int KV, int hd, int cap, int vs_off) {
    const int s0 = S > cap ? S - cap : 0;
    const int ns = S - s0;
    const long idx = (long)(blockIdx.x + blockIdx.y * gridDim.x) * 256 + threadIdx.x;
    const long kvo = (long)KV * hd;
    const long total = (long)ns * kvo;
    if (idx >= total) return;
    const int sr = (int)(idx / kvo);
    const int s = s0 + sr;
    const int rem = (int)(idx - (long)sr * kvo);
    const int h = rem / hd;
    const int d = rem - h * hd;
    const int pb = *pb_ptr;
    const long src = (long)s * kvo + (long)h * hd + d;
    const long dst = ((long)h * cap + (long)((pb + s) % cap)) * hd + d;
    kc[dst] = kvs[src];
    vc[dst] = kvs[(long)vs_off + src];
}
__global__ static void ccg_ink2_kv_put_f4_k(float4 *kc, float4 *vc, const float4 *kvs, const int *pb_ptr,
                                            int S, int KV, int hd4, int cap, int vs_off4) {
    const int s0 = S > cap ? S - cap : 0;
    const int ns = S - s0;
    const long idx = (long)(blockIdx.x + blockIdx.y * gridDim.x) * 256 + threadIdx.x;
    const long kvo4 = (long)KV * hd4;
    const long total = (long)ns * kvo4;
    if (idx >= total) return;
    const int sr = (int)(idx / kvo4);
    const int s = s0 + sr;
    const int rem = (int)(idx - (long)sr * kvo4);
    const int h = rem / hd4;
    const int d4 = rem - h * hd4;
    const int pb = *pb_ptr;
    const long src = (long)s * kvo4 + (long)h * hd4 + d4;
    const long dst = ((long)h * cap + (long)((pb + s) % cap)) * hd4 + d4;
    kc[dst] = kvs[src];
    vc[dst] = kvs[(long)vs_off4 + src];
}
static int cc_relattn_pb(CcBuf *q, CcBuf *kc, CcBuf *vc, CcBuf *o, CcBuf *kvs, CcBuf *r, CcBuf *relp, CcBuf *tau,
                         CcBuf *pb_buf, CcBuf *part_buf, int nc, const CcRelAttn *p) {
    CcCtx *c = live();
    if (!c || !p || !pb_buf || p->hd > 256 || p->hd < 1 || p->d_rel > 64 || p->d_rel < 0 || p->KVH < 1 || p->H % p->KVH || p->cap < 1) return 0;
    if (p->S <= 0) return 1;
    if (part_buf && nc >= 2) {
        coli_launch(ccg_relattn_k, dim3((unsigned)p->H, (unsigned)p->S, (unsigned)nc), 128, 0, c->stream, fp(q), fp(kc), fp(vc), fpw(o),
                    fp(kvs), fp(r), fp(relp), fp(tau), *p, fpw(part_buf), (const int *)fp(pb_buf));
        coli_launch(ccg_relattn_join_k, dim3((unsigned)p->H, (unsigned)p->S), 256, 0, c->stream, fp(part_buf), fp(tau), fpw(o), *p, nc);
        return launched(c, "relattn pb split");
    }
    coli_launch(ccg_relattn_k, dim3((unsigned)p->H, (unsigned)p->S), 128, 0, c->stream, fp(q), fp(kc), fp(vc), fpw(o), fp(kvs),
                fp(r), fp(relp), fp(tau), *p, (float *)nullptr, (const int *)fp(pb_buf));
    return launched(c, "relattn pb");
}
static int cc_ink2_kv_put(CcBuf *kc, CcBuf *vc, CcBuf *kvs, CcBuf *pb_buf, int S, int KV, int hd, int cap, int vs_off) {
    CcCtx *c = live();
    if (!c || !kc || !vc || !kvs || !pb_buf || KV < 1 || hd < 1 || cap < 1) return 0;
    if (S <= 0) return 1;
    const int ns = S > cap ? cap : S;
    if (((hd | vs_off) & 3) == 0) {
        const long total4 = (long)ns * KV * (hd >> 2);
        dim3 g; cg_grid((total4 + 255) / 256, &g);
        coli_launch(ccg_ink2_kv_put_f4_k, g, 256, 0, c->stream, reinterpret_cast<float4 *>(fpw(kc)),
                    reinterpret_cast<float4 *>(fpw(vc)), reinterpret_cast<const float4 *>(fp(kvs)),
                    (const int *)fp(pb_buf), S, KV, hd >> 2, cap, vs_off >> 2);
        return launched(c, "ink2 kv put f4");
    }
    const long total = (long)ns * KV * hd;
    dim3 g; cg_grid((total + 255) / 256, &g);
    coli_launch(ccg_ink2_kv_put_k, g, 256, 0, c->stream, fpw(kc), fpw(vc), fp(kvs), (const int *)fp(pb_buf), S, KV, hd, cap, vs_off);
    return launched(c, "ink2 kv put");
}
static int cc_ink2_mix(CcBuf *x, CcBuf *blk, CcBuf *prm, CcBuf *y, const CcInk2Mix *p) {
    CcCtx *c = live();
    if (!c || !p || !x || !prm || !y || p->S < 0 || p->D < 1 || p->nb < 0 || p->nb > 15 || (p->nb > 0 && !blk) || y == x || y == blk) return 0;
    if (p->S == 0) return 1;
    dim3 g; cg_grid(p->S, &g);
    if ((p->D & 3) == 0 && p->D <= 4096 &&
        ((p->x_off | p->x_row | p->b_off | p->b_row | p->w_off | p->y_off | p->y_row) & 3) == 0) {
        const int bdim = ((p->D >> 2) + 31) & ~31;
        if (p->nb < 6)
            coli_launch(ccg_ink2_mix_f4_k<6>, g, bdim, 0, c->stream, fp(x), p->nb > 0 ? fp(blk) : nullptr, fp(prm), fpw(y), *p);
        else
            coli_launch(ccg_ink2_mix_f4_k<16>, g, bdim, 0, c->stream, fp(x), p->nb > 0 ? fp(blk) : nullptr, fp(prm), fpw(y), *p);
        return launched(c, "ink2 mix f4");
    }
    coli_launch(ccg_ink2_mix_k, g, 256, 0, c->stream, fp(x), p->nb > 0 ? fp(blk) : nullptr, fp(prm), fpw(y), *p);
    return launched(c, "ink2 mix");
}
static int cc_ink2_clip(CcBuf *x, CcBuf *prm, const CcInk2Clip *p) {
    CcCtx *c = live();
    if (!c || !p || !x || p->nseg < 0 || p->n < 1 || p->per_row < 1 || (p->has_w && !prm)) return 0;
    if (p->nseg == 0) return 1;
    if (p->n == 128 && ((p->x_off | p->x_row | (p->has_w ? p->w_off : 0)) & 3) == 0) {
        dim3 g; cg_grid(((long)p->nseg + 7) / 8, &g);
        coli_launch(ccg_ink2_clip_w128_k, g, 256, 0, c->stream, fpw(x), p->has_w ? fp(prm) : nullptr, *p);
        return launched(c, "ink2 clip w128");
    }
    if (p->n == 16) {
        dim3 g; cg_grid(((long)p->nseg + 15) / 16, &g);
        coli_launch(ccg_ink2_clip_w16_k, g, 256, 0, c->stream, fpw(x), p->has_w ? fp(prm) : nullptr, *p);
        return launched(c, "ink2 clip w16");
    }
    dim3 g; cg_grid(p->nseg, &g);
    coli_launch(ccg_ink2_clip_k, g, 256, 0, c->stream, fpw(x), p->has_w ? fp(prm) : nullptr, *p);
    return launched(c, "ink2 clip");
}
static int cc_ink2_dconv(CcBuf *x, CcBuf *dyn, CcBuf *prm, CcBuf *ring, const CcInk2Dconv *p) {
    CcCtx *c = live();
    if (!c || !p || !x || !dyn || !prm || !ring || p->S < 0 || p->C < 1 || p->K < 1 || p->K > 9 || p->hd < 1 || p->C % p->hd) return 0;
    if (p->S == 0) return 1;
    if (p->S >= 16 && !c->capturing) {
        const size_t need = (size_t)p->S * (size_t)p->C * sizeof(float);
        float *tmp = nullptr;
        if (cudaMallocAsync((void **)&tmp, need, c->stream) == cudaSuccess) {
            dim3 gr((unsigned)((p->C + 127) / 128), (unsigned)((p->S + 15) / 16));
            coli_launch(ccg_ink2_dconv_rows_k, gr, 128, 0, c->stream, fp(x), fp(dyn), fp(prm), fp(ring), tmp, *p);
            const int nh = p->K - 1;
            if (((p->C | p->x_off | p->x_row) & 3) == 0) {
                dim3 gc; cg_grid(((long)p->S * (p->C >> 2) + 255) / 256, &gc);
                coli_launch(ccg_conv_commit_f4_k, gc, 256, 0, c->stream, fpw(x), fpw(ring), tmp, p->S, p->C, nh, p->x_off, p->x_row, p->ring_off);
            } else {
                dim3 gc; cg_grid(((long)p->S * p->C + 255) / 256, &gc);
                coli_launch(ccg_conv_commit_k, gc, 256, 0, c->stream, fpw(x), fpw(ring), tmp, p->S, p->C, nh, p->x_off, p->x_row, p->ring_off);
            }
            (void)cudaFreeAsync(tmp, c->stream);
            return launched(c, "ink2 dconv 2d");
        }
        (void)cudaGetLastError();
    }
    dim3 g; cg_grid(((long)p->C + 255) / 256, &g);
    coli_launch(ccg_ink2_dconv_k, g, 256, 0, c->stream, fpw(x), fp(dyn), fp(prm), fpw(ring), *p);
    return launched(c, "ink2 dconv");
}

/* ==== Inkling-2 Multi-Slot (KV_SLOTS > 1) Batched Mux Convs, Q8 KV Attention/Scatter & GPU Softcap ==== */

/* Parallel 2D static short conv for S independent mux rows (each row s uses its own slot ring base). */
__global__ static void ccg_sconv_mux_k(float *__restrict__ x, const float *__restrict__ w,
                                       float *__restrict__ ring, const int *__restrict__ ring_bases, CcSconv p) {
    const int c = (int)blockIdx.x * 128 + (int)threadIdx.x;
    const int s = (int)blockIdx.y;
    if (c >= p.C || s >= p.S) return;
    const int nh = p.CK - 1;
    const size_t roff = (size_t)ring_bases[s] + (size_t)p.ring_off + (size_t)c * nh;
    const float *wc = w + p.w_off + (long)c * p.CK;
    const long xi = (long)p.x_off + (long)s * p.x_row + c;
    const float cur = x[xi];
    float hist[8];
    float acc = 0.f;
#pragma unroll
    for (int k = 0; k < 8; k++) {
        if (k < nh) {
            hist[k] = ring[roff + k];
            acc += __ldg(wc + k) * hist[k];
        } else hist[k] = 0.f;
    }
    acc += __ldg(wc + p.CK - 1) * cur;
    const float outv = acc + cur;
    x[xi] = (p.fc != 1.f && p.fc != 0.f) ? outv / p.fc : outv;
#pragma unroll
    for (int k = 0; k < 7; k++) if (k + 1 < nh) ring[roff + k] = hist[k + 1];
    if (nh > 0) ring[roff + nh - 1] = cur;
}

/* Parallel 2D dynamic short conv for S independent mux rows (global layers). */
__global__ static void ccg_ink2_dconv_mux_k(float *__restrict__ x, const float *__restrict__ dyn,
                                            const float *__restrict__ st, float *__restrict__ ring,
                                            const int *__restrict__ ring_bases, CcInk2Dconv p) {
    const int c = (int)blockIdx.x * 128 + (int)threadIdx.x;
    const int s = (int)blockIdx.y;
    if (c >= p.C || s >= p.S) return;
    const int K = p.K, nh = K - 1, h = c / p.hd;
    const size_t roff = (size_t)ring_bases[s] + (size_t)p.ring_off + (size_t)c * nh;
    const long xi = (long)p.x_off + (long)s * p.x_row + c;
    const float cur = x[xi];
    const float *dw = dyn + p.d_off + (long)s * p.d_row + (long)h * K;
    float hist[8];
#pragma unroll
    for (int k = 0; k < 8; k++) hist[k] = k < nh ? ring[roff + k] : 0.f;
    float acc = cur * (__ldg(st + p.st_off + c) + __ldg(dw));
#pragma unroll
    for (int lag = 1; lag < 9; lag++)
        if (lag < K) acc += hist[nh - lag] * (__ldg(st + p.st_off + (long)lag * p.C + c) + __ldg(dw + lag));
    x[xi] = (acc + cur) / p.fc;
#pragma unroll
    for (int k = 0; k < 7; k++) if (k + 1 < nh) ring[roff + k] = hist[k + 1];
    if (nh > 0) ring[roff + nh - 1] = cur;
}

/* Per-head-row symmetric INT8 KV quantize + scatter (1 warp per (s, kv_head)).
 * mux == 1: row s writes to tab[s] at tab[s].pos % tab[s].cap.
 * mux == 0: rows [max(0, S - cap) .. S) write to tab[0] at (tab[0].pos + s) % tab[0].cap. */
__global__ static void ccg_ink2_kv_put_q8_k(const CcInk2KvRow *__restrict__ tab, const float *__restrict__ kvs,
                                            int S, int KV, int hd, int vs_off, int mux) {
    const int wid = ((int)(blockIdx.x + blockIdx.y * gridDim.x) << 3) + ((int)threadIdx.x >> 5);
    const int lane = (int)threadIdx.x & 31;
    const int total_w = S * KV;
    if (wid >= total_w) return;
    const int s = wid / KV;
    const int h = wid - s * KV;
    const CcInk2KvRow kv = tab[mux ? s : 0];
    if (!mux && s < S - kv.cap) return;
    const int pos = mux ? kv.pos : (kv.pos + s);
    const long row_idx = (long)h * kv.cap + (long)(pos % kv.cap);
    const long src_base = (long)s * KV * hd + (long)h * hd;
    if (hd == 128 && (vs_off & 3) == 0) {
        /* K */
        const float4 kf = __ldg(reinterpret_cast<const float4 *>(kvs + src_base) + lane);
        float amax = fmaxf(fmaxf(fabsf(kf.x), fabsf(kf.y)), fmaxf(fabsf(kf.z), fabsf(kf.w)));
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
        const float ksc = amax > 0.f ? (amax / 127.f) : 1.f;
        const float kinv = amax > 0.f ? (127.f / amax) : 0.f;
        const int kx = max(-127, min(127, __float2int_rn(kf.x * kinv)));
        const int ky = max(-127, min(127, __float2int_rn(kf.y * kinv)));
        const int kz = max(-127, min(127, __float2int_rn(kf.z * kinv)));
        const int kw = max(-127, min(127, __float2int_rn(kf.w * kinv)));
        reinterpret_cast<char4 *>(static_cast<signed char *>(kv.k_q) + row_idx * 128)[lane] =
            make_char4((signed char)kx, (signed char)ky, (signed char)kz, (signed char)kw);
        if (lane == 0) kv.k_sc[row_idx] = ksc;
        /* V */
        const float4 vf = __ldg(reinterpret_cast<const float4 *>(kvs + vs_off + src_base) + lane);
        float vamax = fmaxf(fmaxf(fabsf(vf.x), fabsf(vf.y)), fmaxf(fabsf(vf.z), fabsf(vf.w)));
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) vamax = fmaxf(vamax, __shfl_xor_sync(0xffffffffu, vamax, off));
        const float vsc = vamax > 0.f ? (vamax / 127.f) : 1.f;
        const float vinv = vamax > 0.f ? (127.f / vamax) : 0.f;
        const int vx = max(-127, min(127, __float2int_rn(vf.x * vinv)));
        const int vy = max(-127, min(127, __float2int_rn(vf.y * vinv)));
        const int vz = max(-127, min(127, __float2int_rn(vf.z * vinv)));
        const int vw = max(-127, min(127, __float2int_rn(vf.w * vinv)));
        reinterpret_cast<char4 *>(static_cast<signed char *>(kv.v_q) + row_idx * 128)[lane] =
            make_char4((signed char)vx, (signed char)vy, (signed char)vz, (signed char)vw);
        if (lane == 0) kv.v_sc[row_idx] = vsc;
        return;
    }
    /* Generic hd fallback */
    float kamax = 0.f, vamax = 0.f;
    for (int d = lane; d < hd; d += 32) {
        kamax = fmaxf(kamax, fabsf(kvs[src_base + d]));
        vamax = fmaxf(vamax, fabsf(kvs[vs_off + src_base + d]));
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        kamax = fmaxf(kamax, __shfl_xor_sync(0xffffffffu, kamax, off));
        vamax = fmaxf(vamax, __shfl_xor_sync(0xffffffffu, vamax, off));
    }
    const float ksc = kamax > 0.f ? (kamax / 127.f) : 1.f, kinv = kamax > 0.f ? (127.f / kamax) : 0.f;
    const float vsc = vamax > 0.f ? (vamax / 127.f) : 1.f, vinv = vamax > 0.f ? (127.f / vamax) : 0.f;
    signed char *kd = static_cast<signed char *>(kv.k_q) + row_idx * hd;
    signed char *vd = static_cast<signed char *>(kv.v_q) + row_idx * hd;
    for (int d = lane; d < hd; d += 32) {
        kd[d] = (signed char)max(-127, min(127, __float2int_rn(kvs[src_base + d] * kinv)));
        vd[d] = (signed char)max(-127, min(127, __float2int_rn(kvs[vs_off + src_base + d] * vinv)));
    }
    if (lane == 0) { kv.k_sc[row_idx] = ksc; kv.v_sc[row_idx] = vsc; }
}

/* Blackwell / Ampere Tensor-Core FlashRelAttn reading Q8 KV cache on prefix reuse (t < pos_base)
 * and FP32 kvs on current chunk (t >= pos_base). Single-slot prefill (mux == 0, S >= 16, hd == 128). */
__global__ static void __launch_bounds__(128, 2) ccg_relattn_flash_hd128_q8_k(
        const float *__restrict__ q, const CcInk2KvRow *__restrict__ tab,
        float *__restrict__ o, const float *__restrict__ kvs, const float *__restrict__ r,
        const float *__restrict__ relp, const float *__restrict__ tau, CcRelAttn p) {
#if CC_FA_BODY
    __shared__ __align__(128) unsigned char kv_sm[2 * 64 * 256];
    __shared__ __align__(16) float R_sm[64][16];
    __shared__ __align__(16) float relp_s[128][17];
    unsigned char *K_sm = kv_sm;
    unsigned char *V_sm = kv_sm + 64 * 256;
    const unsigned ksm0 = (unsigned)__cvta_generic_to_shared(K_sm);
    const unsigned vsm0 = (unsigned)__cvta_generic_to_shared(V_sm);

    const CcInk2KvRow kv = tab[0];
    p.pos_base = kv.pos;
    p.cap = kv.cap;
    const int h = (int)blockIdx.x;
    const int q_start = (int)blockIdx.y * 64;
    if (q_start >= p.S) return;
    const int kvh = h / (p.H / p.KVH);
    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31, g = lane >> 2, tig = lane & 3;
    const int dr = p.d_rel;

    #pragma unroll
    for (int idx = tid; idx < 64 * 16; idx += 128) {
        const int rq = idx >> 4, c = idx & 15, sq = q_start + rq;
        uint4 h8 = make_uint4(0u, 0u, 0u, 0u);
        if (sq < p.S) {
            const float4 *qp = reinterpret_cast<const float4 *>(q + p.q_off + (long)sq * p.q_row + h * 128) + 2 * c;
            h8 = cc_fa_f8_to_h8(__ldg(qp), __ldg(qp + 1));
        }
        *reinterpret_cast<uint4 *>(K_sm + cc_fa_swz(rq, c)) = h8;
    }
    if (dr > 0) {
        for (int idx = tid; idx < 64 * dr; idx += 128) {
            const int rq = idx / dr, j = idx - rq * dr, sq = q_start + rq;
            R_sm[rq][j] = (sq < p.S) ? __ldg(r + p.r_off + (long)sq * p.r_row + h * dr + j) : 0.f;
        }
    }
    __syncthreads();

    unsigned qA[8][4];
    const int ar = lane & 15, ac = lane >> 4;
    #pragma unroll
    for (int kk = 0; kk < 8; kk++) cc_fa_ldm4(qA[kk], ksm0 + cc_fa_swz(warp * 16 + ar, 2 * kk + ac));

    const int r0 = warp * 16 + g, r1 = r0 + 8;
    const int sq0 = q_start + r0, sq1 = q_start + r1;
    const bool valid0 = sq0 < p.S, valid1 = sq1 < p.S;
    const int qpos0 = p.pos_base + sq0, qpos1 = p.pos_base + sq1;
    const int wmin0 = p.window > 0 ? max(0, qpos0 - p.window + 1) : 0;
    const int wmin1 = p.window > 0 ? max(0, qpos1 - p.window + 1) : 0;
    const float ts0 = valid0 ? __ldg(tau + p.tau_off + sq0) : 1.f;
    const float ts1 = valid1 ? __ldg(tau + p.tau_off + sq1) : 1.f;
    float r0_f[16], r1_f[16];
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        r0_f[j] = (j < dr) ? R_sm[r0][j] : 0.f;
        r1_f[j] = (j < dr) ? R_sm[r1][j] : 0.f;
    }

    float m0 = (p.has_sink && valid0) ? __ldg(tau + p.sink_off + sq0 * p.H + h) : -3.0e38f;
    float m1 = (p.has_sink && valid1) ? __ldg(tau + p.sink_off + sq1 * p.H + h) : -3.0e38f;
    float l0 = p.has_sink ? 1.f : 0.f;
    float l1 = p.has_sink ? 1.f : 0.f;
    float O[16][4];
    #pragma unroll
    for (int nd = 0; nd < 16; nd++) { O[nd][0] = 0.f; O[nd][1] = 0.f; O[nd][2] = 0.f; O[nd][3] = 0.f; }
    __syncthreads();

    const int q_end = min(q_start + 63, p.S - 1);
    const int t_min = p.window > 0 ? max(0, p.pos_base + q_start - p.window + 1) : 0;
    const int t_max = p.pos_base + q_end;
    const int bn = (lane & 7) + ((lane >> 4) << 3), bc = (lane >> 3) & 1;
    const int vk_off = lane & 15, vn_off = lane >> 4;

    for (int tb = t_min; tb <= t_max; tb += 64) {
        #pragma unroll
        for (int idx = tid; idx < 64 * 16; idx += 128) {
            const int rk = idx >> 4, c = idx & 15, t = tb + rk;
            uint4 kh8 = make_uint4(0u, 0u, 0u, 0u), vh8 = make_uint4(0u, 0u, 0u, 0u);
            if (t <= t_max) {
                if (t < p.pos_base) {
                    const long ridx = (long)kvh * kv.cap + (long)(t % kv.cap);
                    const float ksc = __ldg(kv.k_sc + ridx);
                    const float vsc = __ldg(kv.v_sc + ridx);
                    const char4 *kq = reinterpret_cast<const char4 *>(static_cast<const signed char *>(kv.k_q) + ridx * 128) + 2 * c;
                    const char4 *vq = reinterpret_cast<const char4 *>(static_cast<const signed char *>(kv.v_q) + ridx * 128) + 2 * c;
                    const char4 ka = __ldg(kq), kb = __ldg(kq + 1);
                    const char4 va = __ldg(vq), vb = __ldg(vq + 1);
                    kh8 = cc_fa_f8_to_h8(
                        make_float4(ka.x * ksc, ka.y * ksc, ka.z * ksc, ka.w * ksc),
                        make_float4(kb.x * ksc, kb.y * ksc, kb.z * ksc, kb.w * ksc));
                    vh8 = cc_fa_f8_to_h8(
                        make_float4(va.x * vsc, va.y * vsc, va.z * vsc, va.w * vsc),
                        make_float4(vb.x * vsc, vb.y * vsc, vb.z * vsc, vb.w * vsc));
                } else {
                    const long off = (long)(t - p.pos_base) * p.kv_row + kvh * 128;
                    const float4 *kp4 = reinterpret_cast<const float4 *>(kvs + p.ks_off + off) + 2 * c;
                    const float4 *vp4 = reinterpret_cast<const float4 *>(kvs + p.vs_off + off) + 2 * c;
                    kh8 = cc_fa_f8_to_h8(__ldg(kp4), __ldg(kp4 + 1));
                    vh8 = cc_fa_f8_to_h8(__ldg(vp4), __ldg(vp4 + 1));
                }
            }
            const unsigned sw = cc_fa_swz(rk, c);
            *reinterpret_cast<uint4 *>(K_sm + sw) = kh8;
            *reinterpret_cast<uint4 *>(V_sm + sw) = vh8;
        }

        const int d0 = (p.pos_base + q_start) - tb - 63;
        const bool use_rel = (dr > 0) && (d0 < p.ext) && (d0 + 126 >= 0);
        if (use_rel) {
            const int dist = d0 + tid;
            if (dist >= 0 && dist < p.ext) {
                #pragma unroll
                for (int j = 0; j < 16; j++)
                    if (j < dr) relp_s[tid][j] = __ldg(relp + p.relp_off + (long)j * p.ext + dist);
            } else {
                #pragma unroll
                for (int j = 0; j < 16; j++) if (j < dr) relp_s[tid][j] = 0.f;
            }
        }
        __syncthreads();

        float S[8][4];
        #pragma unroll
        for (int j = 0; j < 8; j++) { S[j][0] = 0.f; S[j][1] = 0.f; S[j][2] = 0.f; S[j][3] = 0.f; }
        #pragma unroll
        for (int kk = 0; kk < 8; kk++) {
            #pragma unroll
            for (int jp = 0; jp < 4; jp++) {
                unsigned kB[4];
                cc_fa_ldm4(kB, ksm0 + cc_fa_swz(jp * 16 + bn, 2 * kk + bc));
                cc_fa_mma(S[2 * jp],     qA[kk], kB[0], kB[1]);
                cc_fa_mma(S[2 * jp + 1], qA[kk], kB[2], kB[3]);
            }
        }

        float tm0 = -3.0e38f, tm1 = -3.0e38f;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            #pragma unroll
            for (int c = 0; c < 2; c++) {
                const int rk = j * 8 + tig * 2 + c;
                const int t = tb + rk;
                if (valid0 && t >= wmin0 && t <= qpos0) {
                    float b0 = 0.f;
                    if (use_rel && (qpos0 - t) < p.ext) {
                        const int ridx = r0 - rk + 63;
                        #pragma unroll
                        for (int jj = 0; jj < 16; jj++) if (jj < dr) b0 += r0_f[jj] * relp_s[ridx][jj];
                    }
                    const float sv = ts0 * (S[j][c] * p.scale + b0);
                    S[j][c] = sv;
                    tm0 = fmaxf(tm0, sv);
                } else S[j][c] = -3.0e38f;
                if (valid1 && t >= wmin1 && t <= qpos1) {
                    float b1 = 0.f;
                    if (use_rel && (qpos1 - t) < p.ext) {
                        const int ridx = r1 - rk + 63;
                        #pragma unroll
                        for (int jj = 0; jj < 16; jj++) if (jj < dr) b1 += r1_f[jj] * relp_s[ridx][jj];
                    }
                    const float sv = ts1 * (S[j][2 + c] * p.scale + b1);
                    S[j][2 + c] = sv;
                    tm1 = fmaxf(tm1, sv);
                } else S[j][2 + c] = -3.0e38f;
            }
        }
        tm0 = fmaxf(tm0, __shfl_xor_sync(0xffffffffu, tm0, 1));
        tm0 = fmaxf(tm0, __shfl_xor_sync(0xffffffffu, tm0, 2));
        tm1 = fmaxf(tm1, __shfl_xor_sync(0xffffffffu, tm1, 1));
        tm1 = fmaxf(tm1, __shfl_xor_sync(0xffffffffu, tm1, 2));

        const float mn0 = fmaxf(m0, tm0), mn1 = fmaxf(m1, tm1);
        const float corr0 = (m0 > -1.0e30f) ? expf(m0 - mn0) : 0.f;
        const float corr1 = (m1 > -1.0e30f) ? expf(m1 - mn1) : 0.f;
        #pragma unroll
        for (int nd = 0; nd < 16; nd++) {
            O[nd][0] *= corr0; O[nd][1] *= corr0;
            O[nd][2] *= corr1; O[nd][3] *= corr1;
        }

        float ps0 = 0.f, ps1 = 0.f;
        unsigned pA[4][4];
        #pragma unroll
        for (int kk = 0; kk < 4; kk++) {
            const int j0 = 2 * kk, j1 = j0 + 1;
            const float p00 = (S[j0][0] > -1.0e30f) ? expf(S[j0][0] - mn0) : 0.f;
            const float p01 = (S[j0][1] > -1.0e30f) ? expf(S[j0][1] - mn0) : 0.f;
            const float p10 = (S[j0][2] > -1.0e30f) ? expf(S[j0][2] - mn1) : 0.f;
            const float p11 = (S[j0][3] > -1.0e30f) ? expf(S[j0][3] - mn1) : 0.f;
            const float q00 = (S[j1][0] > -1.0e30f) ? expf(S[j1][0] - mn0) : 0.f;
            const float q01 = (S[j1][1] > -1.0e30f) ? expf(S[j1][1] - mn0) : 0.f;
            const float q10 = (S[j1][2] > -1.0e30f) ? expf(S[j1][2] - mn1) : 0.f;
            const float q11 = (S[j1][3] > -1.0e30f) ? expf(S[j1][3] - mn1) : 0.f;
            ps0 += p00 + p01 + q00 + q01;
            ps1 += p10 + p11 + q10 + q11;
            pA[kk][0] = cc_fa_h2u(__floats2half2_rn(p00, p01));
            pA[kk][1] = cc_fa_h2u(__floats2half2_rn(p10, p11));
            pA[kk][2] = cc_fa_h2u(__floats2half2_rn(q00, q01));
            pA[kk][3] = cc_fa_h2u(__floats2half2_rn(q10, q11));
        }
        ps0 += __shfl_xor_sync(0xffffffffu, ps0, 1);
        ps0 += __shfl_xor_sync(0xffffffffu, ps0, 2);
        ps1 += __shfl_xor_sync(0xffffffffu, ps1, 1);
        ps1 += __shfl_xor_sync(0xffffffffu, ps1, 2);
        l0 = l0 * corr0 + ps0; m0 = mn0;
        l1 = l1 * corr1 + ps1; m1 = mn1;

        #pragma unroll
        for (int kk = 0; kk < 4; kk++) {
            #pragma unroll
            for (int nd = 0; nd < 8; nd++) {
                unsigned vB[4];
                cc_fa_ldm4_trans(vB, vsm0 + cc_fa_swz(kk * 16 + vk_off, 2 * nd + vn_off));
                cc_fa_mma(O[2 * nd],     pA[kk], vB[0], vB[1]);
                cc_fa_mma(O[2 * nd + 1], pA[kk], vB[2], vB[3]);
            }
        }
        __syncthreads();
    }

    const float inv0 = (l0 > 0.f) ? (1.f / l0) : 0.f;
    const float inv1 = (l1 > 0.f) ? (1.f / l1) : 0.f;
    if (valid0) {
        float *o0 = o + p.o_off + (long)sq0 * p.o_row + h * 128;
        #pragma unroll
        for (int nd = 0; nd < 16; nd++)
            *reinterpret_cast<float2 *>(o0 + nd * 8 + tig * 2) = make_float2(O[nd][0] * inv0, O[nd][1] * inv0);
    }
    if (valid1) {
        float *o1 = o + p.o_off + (long)sq1 * p.o_row + h * 128;
        #pragma unroll
        for (int nd = 0; nd < 16; nd++)
            *reinterpret_cast<float2 *>(o1 + nd * 8 + tig * 2) = make_float2(O[nd][2] * inv1, O[nd][3] * inv1);
    }
#else
    (void)q; (void)tab; (void)o; (void)kvs; (void)r; (void)relp; (void)tau; (void)p;
#endif
}

/* Vectorized char4 Q8 KV tile for hd==128 (4 warps = 128 threads, 4x less KV bandwidth than FP32). */
__device__ __forceinline__ void ccg_relattn_tile_q8_hd128(
        const CcInk2KvRow &kv, const float *__restrict__ kvs, const float *__restrict__ relp,
        const CcRelAttn &p, float4 qv, const float *rs, float *pe, float *red, int *pt,
        int kvh, int s, int pos_base_row, int qpos, int t0, int n, float ts, int b0, int tid, int mux,
        float &m, float &l, float &acc0) {
    const int warp = tid >> 5, lane = tid & 31;
    const int cnt = min(128, n - b0);
    const signed char *kq_base = static_cast<const signed char *>(kv.k_q);
    const signed char *vq_base = static_cast<const signed char *>(kv.v_q);
    const int t_start = t0 + b0;
    const int row0 = t_start % kv.cap;
    const bool contig = (t_start + cnt <= pos_base_row) && (row0 + cnt <= kv.cap);
    const long head_off = (long)kvh * kv.cap;
    if (contig) {
        const bool any_rel = (p.d_rel > 0) && (qpos - (t_start + cnt - 1) < p.ext);
        for (int j = warp; j < cnt; j += 4) {
            const long ridx = head_off + row0 + j;
            const char4 kc4 = __ldg(reinterpret_cast<const char4 *>(kq_base + ridx * 128) + lane);
            const float ksc = __ldg(kv.k_sc + ridx);
            float a = ksc * (qv.x * (float)kc4.x + qv.y * (float)kc4.y + qv.z * (float)kc4.z + qv.w * (float)kc4.w);
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) a += __shfl_xor_sync(0xffffffffu, a, off);
            float bias = 0.f;
            if (any_rel) {
                const int dist = qpos - (t_start + j);
                if (dist < p.ext) {
                    if (p.d_rel <= 32) {
                        bias = lane < p.d_rel ? rs[lane] * __ldg(relp + p.relp_off + lane * p.ext + dist) : 0.f;
                        #pragma unroll
                        for (int off = 16; off > 0; off >>= 1) bias += __shfl_xor_sync(0xffffffffu, bias, off);
                    } else {
                        for (int rj = lane; rj < p.d_rel; rj += 32)
                            bias += rs[rj] * __ldg(relp + p.relp_off + rj * p.ext + dist);
                        #pragma unroll
                        for (int off = 16; off > 0; off >>= 1) bias += __shfl_xor_sync(0xffffffffu, bias, off);
                    }
                }
            }
            if (lane == 0) red[j] = ts * (a * p.scale + bias);
        }
    } else {
        for (int j = warp; j < cnt; j += 4) {
            const int t = t_start + j;
            float a;
            if (t < pos_base_row) {
                const long ridx = head_off + (long)(t % kv.cap);
                const char4 kc4 = __ldg(reinterpret_cast<const char4 *>(kq_base + ridx * 128) + lane);
                const float ksc = __ldg(kv.k_sc + ridx);
                a = ksc * (qv.x * (float)kc4.x + qv.y * (float)kc4.y + qv.z * (float)kc4.z + qv.w * (float)kc4.w);
            } else {
                const int ks_row = mux ? s : (t - pos_base_row);
                const float4 kv4 = __ldg(reinterpret_cast<const float4 *>(kvs + p.ks_off + (long)ks_row * p.kv_row + kvh * 128) + lane);
                a = qv.x * kv4.x + qv.y * kv4.y + qv.z * kv4.z + qv.w * kv4.w;
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) a += __shfl_xor_sync(0xffffffffu, a, off);
            const int dist = qpos - t;
            float bias = 0.f;
            if (dist < p.ext && p.d_rel > 0) {
                if (p.d_rel <= 32) {
                    bias = lane < p.d_rel ? rs[lane] * __ldg(relp + p.relp_off + lane * p.ext + dist) : 0.f;
                    #pragma unroll
                    for (int off = 16; off > 0; off >>= 1) bias += __shfl_xor_sync(0xffffffffu, bias, off);
                } else {
                    for (int rj = lane; rj < p.d_rel; rj += 32)
                        bias += rs[rj] * __ldg(relp + p.relp_off + rj * p.ext + dist);
                    #pragma unroll
                    for (int off = 16; off > 0; off >>= 1) bias += __shfl_xor_sync(0xffffffffu, bias, off);
                }
            }
            if (lane == 0) {
                red[j] = ts * (a * p.scale + bias);
                pt[j] = t;
            }
        }
    }
    __syncthreads();
    const float sv = tid < cnt ? red[tid] : -3.0e38f;
    float wm = sv;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) wm = fmaxf(wm, __shfl_xor_sync(0xffffffffu, wm, off));
    __shared__ float wred[4];
    if (lane == 0) wred[warp] = wm;
    __syncthreads();
    const float tmax = fmaxf(fmaxf(wred[0], wred[1]), fmaxf(wred[2], wred[3]));
    const float mn = fmaxf(m, tmax);
    const float e = tid < cnt ? expf(sv - mn) : 0.f;
    if (contig) {
        /* Pre-multiply exp weight by v_sc once in parallel across the 128 threads (exact (w * vsc) bits) */
        pe[tid] = tid < cnt ? e * __ldg(kv.v_sc + head_off + row0 + tid) : 0.f;
    } else {
        pe[tid] = e;
    }
    float ws = e;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) ws += __shfl_xor_sync(0xffffffffu, ws, off);
    if (lane == 0) wred[warp] = ws;
    __syncthreads();
    const float sum_e = wred[0] + wred[1] + wred[2] + wred[3];
    const float corr = expf(m - mn);
    l = l * corr + sum_e;
    acc0 *= corr;
    if (contig) {
        const signed char *vp = vq_base + (head_off + row0) * 128 + tid;
        int j = 0;
        for (; j + 3 < cnt; j += 4) {
            const float w0 = pe[j], w1 = pe[j + 1], w2 = pe[j + 2], w3 = pe[j + 3];
            const signed char v0 = __ldg(vp + (long)j * 128);
            const signed char v1 = __ldg(vp + (long)(j + 1) * 128);
            const signed char v2 = __ldg(vp + (long)(j + 2) * 128);
            const signed char v3 = __ldg(vp + (long)(j + 3) * 128);
            acc0 += w0 * (float)v0;
            acc0 += w1 * (float)v1;
            acc0 += w2 * (float)v2;
            acc0 += w3 * (float)v3;
        }
        for (; j < cnt; j++) acc0 += pe[j] * (float)__ldg(vp + (long)j * 128);
    } else {
        for (int j = 0; j < cnt; j++) {
            const int t = pt[j];
            const float w = pe[j];
            if (t < pos_base_row) {
                const long ridx = head_off + (long)(t % kv.cap);
                const float vsc = __ldg(kv.v_sc + ridx);
                const signed char vq = __ldg(vq_base + ridx * 128 + tid);
                acc0 += (w * vsc) * (float)vq;
            } else {
                const int vs_row = mux ? s : (t - pos_base_row);
                acc0 += w * __ldg(kvs + p.vs_off + (long)vs_row * p.kv_row + kvh * 128 + tid);
            }
        }
    }
    m = mn;
    __syncthreads();
}

/* Generic Q8 KV tile for any hd <= 256. */
__device__ __forceinline__ void ccg_relattn_tile_q8(
        const CcInk2KvRow &kv, const float *__restrict__ kvs, const float *__restrict__ relp,
        const CcRelAttn &p, const float *qs, const float *rs, float *pe, float *red, int *pt,
        int kvh, int s, int pos_base_row, int qpos, int t0, int n, float ts, int b0, int tid, int mux,
        float &m, float &l, float &acc0, float &acc1) {
    const int i = b0 + tid;
    float sv = -3.0e38f;
    if (i < n) {
        const int t = t0 + i;
        pt[tid] = t;
        float a = 0.f;
        if (t < pos_base_row) {
            const long ridx = (long)kvh * kv.cap + (long)(t % kv.cap);
            const signed char *kr = static_cast<const signed char *>(kv.k_q) + ridx * p.hd;
            const float ksc = kv.k_sc[ridx];
            for (int d = 0; d < p.hd; d++) a += qs[d] * (float)kr[d];
            a *= ksc;
        } else {
            const int ks_row = mux ? s : (t - pos_base_row);
            const float *kr = kvs + p.ks_off + (long)ks_row * p.kv_row + kvh * p.hd;
            for (int d = 0; d < p.hd; d++) a += qs[d] * kr[d];
        }
        const int dist = qpos - t;
        float bias = 0.f;
        if (dist < p.ext) for (int j = 0; j < p.d_rel; j++) bias += rs[j] * relp[p.relp_off + j * p.ext + dist];
        sv = ts * (a * p.scale + bias);
    }
    red[tid] = sv;
    __syncthreads();
    for (int off = 64; off > 0; off >>= 1) { if (tid < off) red[tid] = fmaxf(red[tid], red[tid + off]); __syncthreads(); }
    const float mn = fmaxf(m, red[0]);
    __syncthreads();
    const float e = i < n ? expf(sv - mn) : 0.f;
    pe[tid] = e; red[tid] = e;
    __syncthreads();
    for (int off = 64; off > 0; off >>= 1) { if (tid < off) red[tid] += red[tid + off]; __syncthreads(); }
    const float corr = expf(m - mn);
    l = l * corr + red[0];
    acc0 *= corr; acc1 *= corr;
    const int cnt = min(128, n - b0), d0 = tid, d1 = tid + 128;
    for (int j = 0; j < cnt; j++) {
        const int t = pt[j];
        const float w = pe[j];
        if (t < pos_base_row) {
            const long ridx = (long)kvh * kv.cap + (long)(t % kv.cap);
            const signed char *vr = static_cast<const signed char *>(kv.v_q) + ridx * p.hd;
            const float wv = w * kv.v_sc[ridx];
            if (d0 < p.hd) acc0 += wv * (float)vr[d0];
            if (d1 < p.hd) acc1 += wv * (float)vr[d1];
        } else {
            const int vs_row = mux ? s : (t - pos_base_row);
            const float *vr = kvs + p.vs_off + (long)vs_row * p.kv_row + kvh * p.hd;
            if (d0 < p.hd) acc0 += w * vr[d0];
            if (d1 < p.hd) acc1 += w * vr[d1];
        }
    }
    m = mn;
    __syncthreads();
}

__global__ static void ccg_relattn_q8_k(const float *q, const CcInk2KvRow *tab, float *o, const float *kvs,
                                        const float *r, const float *relp, const float *tau, CcRelAttn p,
                                        float *part, int mux) {
    const int h = (int)blockIdx.x, s = (int)blockIdx.y, tid = (int)threadIdx.x;
    const int kvh = h / (p.H / p.KVH);
    const CcInk2KvRow kv = tab[mux ? s : 0];
    const int pos_base_row = kv.pos;
    const int qpos = mux ? kv.pos : (kv.pos + s);
    const int t0 = p.window > 0 ? max(0, qpos - p.window + 1) : 0;
    const int n = qpos - t0 + 1;
    if (part && (int)blockIdx.z * 128 >= n) {
        float *pp = part + (((long)s * p.H + h) * gridDim.z + (int)blockIdx.z) * (p.hd + 2);
        if (tid == 0) { pp[0] = -3.0e38f; pp[1] = 0.f; }
        return;
    }
    __shared__ float qs[256], rs[64], pe[128], red[128];
    __shared__ int pt[128];
    const int qb = p.q_off + s * p.q_row + h * p.hd;
    const float ts = tau[p.tau_off + s];
    for (int j = tid; j < p.d_rel; j += 128) rs[j] = r[p.r_off + s * p.r_row + h * p.d_rel + j];
    const bool fast128 = (p.hd == 128) && ((p.q_off | p.q_row | p.ks_off | p.vs_off | p.kv_row) & 3) == 0;
    if (fast128) {
        const float4 qv = __ldg(reinterpret_cast<const float4 *>(q + qb) + (tid & 31));
        __syncthreads();
        float m = -3.0e38f, l = 0.f, acc0 = 0.f;
        if (part) {
            const int z = (int)blockIdx.z, stride = (int)gridDim.z * 128;
            for (int b0 = z * 128; b0 < n; b0 += stride)
                ccg_relattn_tile_q8_hd128(kv, kvs, relp, p, qv, rs, pe, red, pt, kvh, s, pos_base_row, qpos, t0, n, ts, b0, tid, mux, m, l, acc0);
            float *pp = part + (((long)s * p.H + h) * gridDim.z + z) * 130;
            if (tid == 0) { pp[0] = m; pp[1] = l; }
            pp[2 + tid] = acc0;
            return;
        }
        if (p.has_sink) { m = tau[p.sink_off + s * p.H + h]; l = 1.f; }
        for (int b0 = 0; b0 < n; b0 += 128)
            ccg_relattn_tile_q8_hd128(kv, kvs, relp, p, qv, rs, pe, red, pt, kvh, s, pos_base_row, qpos, t0, n, ts, b0, tid, mux, m, l, acc0);
        const float inv = l > 0.f ? 1.f / l : 0.f;
        o[p.o_off + s * p.o_row + h * 128 + tid] = acc0 * inv;
        return;
    }
    for (int d = tid; d < p.hd; d += 128) qs[d] = q[qb + d];
    __syncthreads();
    float m = -3.0e38f, l = 0.f, acc0 = 0.f, acc1 = 0.f;
    if (part) {
        const int z = (int)blockIdx.z, stride = (int)gridDim.z * 128;
        for (int b0 = z * 128; b0 < n; b0 += stride)
            ccg_relattn_tile_q8(kv, kvs, relp, p, qs, rs, pe, red, pt, kvh, s, pos_base_row, qpos, t0, n, ts, b0, tid, mux, m, l, acc0, acc1);
        float *pp = part + (((long)s * p.H + h) * gridDim.z + z) * (p.hd + 2);
        if (tid == 0) { pp[0] = m; pp[1] = l; }
        if (tid < p.hd) pp[2 + tid] = acc0;
        if (tid + 128 < p.hd) pp[2 + tid + 128] = acc1;
        return;
    }
    if (p.has_sink) { m = tau[p.sink_off + s * p.H + h]; l = 1.f; }
    for (int b0 = 0; b0 < n; b0 += 128)
        ccg_relattn_tile_q8(kv, kvs, relp, p, qs, rs, pe, red, pt, kvh, s, pos_base_row, qpos, t0, n, ts, b0, tid, mux, m, l, acc0, acc1);
    const float inv = l > 0.f ? 1.f / l : 0.f;
    const int ob = p.o_off + s * p.o_row + h * p.hd;
    if (tid < p.hd) o[ob + tid] = acc0 * inv;
    if (tid + 128 < p.hd) o[ob + tid + 128] = acc1 * inv;
}

__global__ static void ccg_ink2_softcap_k(float *__restrict__ x, int n, float cap, float inv_cap) {
    const int n4 = n >> 2;
    for (int i = (int)blockIdx.x * blockDim.x + (int)threadIdx.x; i < n4; i += (int)gridDim.x * blockDim.x) {
        float4 v = reinterpret_cast<float4 *>(x)[i];
        v.x = cap * tanhf(v.x * inv_cap);
        v.y = cap * tanhf(v.y * inv_cap);
        v.z = cap * tanhf(v.z * inv_cap);
        v.w = cap * tanhf(v.w * inv_cap);
        reinterpret_cast<float4 *>(x)[i] = v;
    }
    if (blockIdx.x == 0 && threadIdx.x < (n & 3)) {
        const int idx = (n4 << 2) + (int)threadIdx.x;
        x[idx] = cap * tanhf(x[idx] * inv_cap);
    }
}

static int cc_sconv_mux(CcBuf *x, CcBuf *w, CcBuf *ring, CcBuf *ring_bases, const CcSconv *p) {
    CcCtx *c = live();
    if (!c || !p || !x || !w || !ring || !ring_bases || p->CK < 1 || p->CK > 9) return 0;
    if (p->S <= 0 || p->C <= 0) return 1;
    dim3 g((unsigned)((p->C + 127) / 128), (unsigned)p->S);
    coli_launch(ccg_sconv_mux_k, g, 128, 0, c->stream, fpw(x), fp(w), fpw(ring), (const int *)fp(ring_bases), *p);
    return launched(c, "sconv mux");
}
static int cc_ink2_dconv_mux(CcBuf *x, CcBuf *dyn, CcBuf *prm, CcBuf *ring, CcBuf *ring_bases, const CcInk2Dconv *p) {
    CcCtx *c = live();
    if (!c || !p || !x || !dyn || !prm || !ring || !ring_bases || p->C < 1 || p->K < 1 || p->K > 9 || p->hd < 1 || p->C % p->hd) return 0;
    if (p->S <= 0) return 1;
    dim3 g((unsigned)((p->C + 127) / 128), (unsigned)p->S);
    coli_launch(ccg_ink2_dconv_mux_k, g, 128, 0, c->stream, fpw(x), fp(dyn), fp(prm), fpw(ring), (const int *)fp(ring_bases), *p);
    return launched(c, "ink2 dconv mux");
}
static int cc_relattn_q8(CcBuf *q, CcBuf *kv_tab, size_t tab_off, CcBuf *o, CcBuf *kvs, CcBuf *r, CcBuf *relp, CcBuf *tau,
                         CcBuf *part_buf, int nc, const CcRelAttn *p, int mux) {
    CcCtx *c = live();
    if (!c || !p || !q || !kv_tab || !o || !kvs || !r || !relp || !tau || p->hd > 256 || p->hd < 1 ||
        p->d_rel > 64 || p->d_rel < 0 || p->KVH < 1 || p->H % p->KVH) return 0;
    if (p->S <= 0) return 1;
    const CcInk2KvRow *tab = reinterpret_cast<const CcInk2KvRow *>(kv_tab->d) + tab_off;
#if !(defined(__HIP__) || defined(__HIP_PLATFORM_AMD__))
    if (!mux && cc_relattn_flash_on() && c->major >= 8 && p->S >= 16 && p->hd == 128 && p->d_rel <= 16 &&
        ((p->q_off | p->q_row | p->o_off | p->o_row | p->ks_off | p->vs_off | p->kv_row) & 3) == 0) {
        static unsigned fa_q8_attr = 0;
        if (c->device >= 0 && c->device < 32 && !(fa_q8_attr & (1u << c->device))) {
            cudaFuncSetAttribute(ccg_relattn_flash_hd128_q8_k, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
            (void)cudaGetLastError();
            fa_q8_attr |= 1u << c->device;
        }
        dim3 gf((unsigned)p->H, (unsigned)((p->S + 63) / 64));
        coli_launch(ccg_relattn_flash_hd128_q8_k, gf, 128, 0, c->stream, fp(q), tab, fpw(o),
                    fp(kvs), fp(r), fp(relp), fp(tau), *p);
        return launched(c, "relattn flash q8");
    }
#endif
    if (part_buf && nc >= 2) {
        coli_launch(ccg_relattn_q8_k, dim3((unsigned)p->H, (unsigned)p->S, (unsigned)nc), 128, 0, c->stream,
                    fp(q), tab, fpw(o), fp(kvs), fp(r), fp(relp), fp(tau), *p, fpw(part_buf), mux);
        coli_launch(ccg_relattn_join_k, dim3((unsigned)p->H, (unsigned)p->S), 256, 0, c->stream,
                    fp(part_buf), fp(tau), fpw(o), *p, nc);
        return launched(c, "relattn q8 split");
    }
    coli_launch(ccg_relattn_q8_k, dim3((unsigned)p->H, (unsigned)p->S), 128, 0, c->stream,
                fp(q), tab, fpw(o), fp(kvs), fp(r), fp(relp), fp(tau), *p, (float *)nullptr, mux);
    return launched(c, "relattn q8");
}
static int cc_ink2_kv_put_q8(CcBuf *kv_tab, size_t tab_off, CcBuf *kvs, int S, int KV, int hd, int vs_off, int mux) {
    CcCtx *c = live();
    if (!c || !kv_tab || !kvs || KV < 1 || hd < 1) return 0;
    if (S <= 0) return 1;
    const CcInk2KvRow *tab = reinterpret_cast<const CcInk2KvRow *>(kv_tab->d) + tab_off;
    const long nwarps = (long)S * KV;
    dim3 g; cg_grid((nwarps + 7) / 8, &g);
    coli_launch(ccg_ink2_kv_put_q8_k, g, 256, 0, c->stream, tab, fp(kvs), S, KV, hd, vs_off, mux);
    return launched(c, "ink2 kv put q8");
}
static int cc_ink2_softcap(CcBuf *x, int n, float cap) {
    CcCtx *c = live();
    if (!c || !x || n < 0) return 0;
    if (n == 0 || !(cap > 0.f)) return 1;
    const int n4 = n >> 2;
    int blocks = (n4 + 255) / 256;
    if (blocks < 1) blocks = 1;
    if (blocks > 4096) blocks = 4096;
    coli_launch(ccg_ink2_softcap_k, (unsigned)blocks, 256, 0, c->stream, fpw(x), n, cap, 1.f / cap);
    return launched(c, "ink2 softcap");
}

/* ==== Packed Multi-Prompt Prefill Kernels (M independent prompt chunks in ONE GPU frame) ==== */

__global__ static void ccg_sconv_packed_k(float *__restrict__ x, const float *__restrict__ w,
                                          float *__restrict__ ring, const CcInk2Seg *__restrict__ segs, CcSconv p) {
    const int c = (int)blockIdx.x * 128 + (int)threadIdx.x;
    const int m = (int)blockIdx.y;
    if (c >= p.C) return;
    const CcInk2Seg seg = segs[m];
    const int nh = p.CK - 1;
    const size_t roff = (size_t)seg.ring_base + (size_t)p.ring_off + (size_t)c * nh;
    const float *wc = w + p.w_off + (long)c * p.CK;
    float wt[9];
#pragma unroll
    for (int k = 0; k < 9; k++) wt[k] = k < p.CK ? __ldg(wc + k) : 0.f;
    float hist[8];
#pragma unroll
    for (int k = 0; k < 8; k++) hist[k] = k < nh ? ring[roff + k] : 0.f;
    const float wl = wt[p.CK - 1];
    const bool div_fc = (p.fc != 1.f && p.fc != 0.f);
    for (int ls = 0; ls < seg.len; ls++) {
        const long xi = (long)p.x_off + (long)(seg.start + ls) * p.x_row + c;
        const float cur = x[xi];
        float acc = 0.f;
#pragma unroll
        for (int k = 0; k < 8; k++) if (k < nh) acc += wt[k] * hist[k];
        acc += wl * cur;
        const float outv = acc + cur;
        x[xi] = div_fc ? outv / p.fc : outv;
#pragma unroll
        for (int k = 0; k < 7; k++) if (k + 1 < nh) hist[k] = hist[k + 1];
#pragma unroll
        for (int k = 0; k < 8; k++) if (k == nh - 1) hist[k] = cur;
    }
#pragma unroll
    for (int k = 0; k < 8; k++) if (k < nh) ring[roff + k] = hist[k];
}

__global__ static void ccg_ink2_dconv_packed_k(float *__restrict__ x, const float *__restrict__ dyn,
                                               const float *__restrict__ st, float *__restrict__ ring,
                                               const CcInk2Seg *__restrict__ segs, CcInk2Dconv p) {
    const int c = (int)blockIdx.x * 128 + (int)threadIdx.x;
    const int m = (int)blockIdx.y;
    if (c >= p.C) return;
    const CcInk2Seg seg = segs[m];
    const int K = p.K, nh = K - 1, h = c / p.hd;
    const size_t roff = (size_t)seg.ring_base + (size_t)p.ring_off + (size_t)c * nh;
    float sw[9];
#pragma unroll
    for (int lag = 0; lag < 9; lag++) sw[lag] = lag < K ? __ldg(st + p.st_off + (long)lag * p.C + c) : 0.f;
    float hist[8];
#pragma unroll
    for (int k = 0; k < 8; k++) hist[k] = k < nh ? ring[roff + k] : 0.f;
    for (int ls = 0; ls < seg.len; ls++) {
        const int gs = seg.start + ls;
        const long xi = (long)p.x_off + (long)gs * p.x_row + c;
        const float cur = x[xi];
        const float *dw = dyn + p.d_off + (long)gs * p.d_row + (long)h * K;
        float acc = cur * (sw[0] + __ldg(dw));
#pragma unroll
        for (int lag = 1; lag < 9; lag++) if (lag < K) acc += hist[nh - lag] * (sw[lag] + __ldg(dw + lag));
        x[xi] = (acc + cur) / p.fc;
#pragma unroll
        for (int k = 0; k < 7; k++) if (k + 1 < nh) hist[k] = hist[k + 1];
#pragma unroll
        for (int k = 0; k < 8; k++) if (k == nh - 1) hist[k] = cur;
    }
#pragma unroll
    for (int k = 0; k < 8; k++) if (k < nh) ring[roff + k] = hist[k];
}

__global__ static void ccg_ink2_kv_put_q8_packed_k(const CcInk2KvRow *__restrict__ tab, const float *__restrict__ kvs,
                                                   const CcInk2Seg *__restrict__ segs, const int2 *__restrict__ row_seg,
                                                   int N, int KV, int hd, int vs_off) {
    const int wid = ((int)(blockIdx.x + blockIdx.y * gridDim.x) << 3) + ((int)threadIdx.x >> 5);
    const int lane = (int)threadIdx.x & 31;
    if (wid >= N * KV) return;
    const int gs = wid / KV;
    const int h = wid - gs * KV;
    const int2 rs = row_seg[gs];
    const CcInk2Seg seg = segs[rs.x];
    const CcInk2KvRow kv = tab[rs.x];
    const int ls = rs.y;
    if (ls < seg.len - kv.cap) return;
    const int pos = seg.pos_base + ls;
    const long row_idx = (long)h * kv.cap + (long)(pos % kv.cap);
    const long src_base = (long)gs * KV * hd + (long)h * hd;
    if (hd == 128 && (vs_off & 3) == 0) {
        const float4 kf = __ldg(reinterpret_cast<const float4 *>(kvs + src_base) + lane);
        float amax = fmaxf(fmaxf(fabsf(kf.x), fabsf(kf.y)), fmaxf(fabsf(kf.z), fabsf(kf.w)));
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
        const float ksc = amax > 0.f ? (amax / 127.f) : 1.f, kinv = amax > 0.f ? (127.f / amax) : 0.f;
        reinterpret_cast<char4 *>(static_cast<signed char *>(kv.k_q) + row_idx * 128)[lane] =
            make_char4((signed char)max(-127, min(127, __float2int_rn(kf.x * kinv))),
                       (signed char)max(-127, min(127, __float2int_rn(kf.y * kinv))),
                       (signed char)max(-127, min(127, __float2int_rn(kf.z * kinv))),
                       (signed char)max(-127, min(127, __float2int_rn(kf.w * kinv))));
        if (lane == 0) kv.k_sc[row_idx] = ksc;
        const float4 vf = __ldg(reinterpret_cast<const float4 *>(kvs + vs_off + src_base) + lane);
        float vamax = fmaxf(fmaxf(fabsf(vf.x), fabsf(vf.y)), fmaxf(fabsf(vf.z), fabsf(vf.w)));
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) vamax = fmaxf(vamax, __shfl_xor_sync(0xffffffffu, vamax, off));
        const float vsc = vamax > 0.f ? (vamax / 127.f) : 1.f, vinv = vamax > 0.f ? (127.f / vamax) : 0.f;
        reinterpret_cast<char4 *>(static_cast<signed char *>(kv.v_q) + row_idx * 128)[lane] =
            make_char4((signed char)max(-127, min(127, __float2int_rn(vf.x * vinv))),
                       (signed char)max(-127, min(127, __float2int_rn(vf.y * vinv))),
                       (signed char)max(-127, min(127, __float2int_rn(vf.z * vinv))),
                       (signed char)max(-127, min(127, __float2int_rn(vf.w * vinv))));
        if (lane == 0) kv.v_sc[row_idx] = vsc;
        return;
    }
    float kamax = 0.f, vamax = 0.f;
    for (int d = lane; d < hd; d += 32) { kamax = fmaxf(kamax, fabsf(kvs[src_base + d])); vamax = fmaxf(vamax, fabsf(kvs[vs_off + src_base + d])); }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) { kamax = fmaxf(kamax, __shfl_xor_sync(0xffffffffu, kamax, off)); vamax = fmaxf(vamax, __shfl_xor_sync(0xffffffffu, vamax, off)); }
    const float ksc = kamax > 0.f ? (kamax / 127.f) : 1.f, kinv = kamax > 0.f ? (127.f / kamax) : 0.f;
    const float vsc = vamax > 0.f ? (vamax / 127.f) : 1.f, vinv = vamax > 0.f ? (127.f / vamax) : 0.f;
    signed char *kd = static_cast<signed char *>(kv.k_q) + row_idx * hd;
    signed char *vd = static_cast<signed char *>(kv.v_q) + row_idx * hd;
    for (int d = lane; d < hd; d += 32) {
        kd[d] = (signed char)max(-127, min(127, __float2int_rn(kvs[src_base + d] * kinv)));
        vd[d] = (signed char)max(-127, min(127, __float2int_rn(kvs[vs_off + src_base + d] * vinv)));
    }
    if (lane == 0) { kv.k_sc[row_idx] = ksc; kv.v_sc[row_idx] = vsc; }
}

__global__ static void __launch_bounds__(128, 2) ccg_relattn_flash_hd128_q8_packed_k(
        const float *__restrict__ q, const CcInk2KvRow *__restrict__ tab,
        float *__restrict__ o, const float *__restrict__ kvs, const float *__restrict__ r,
        const float *__restrict__ relp, const float *__restrict__ tau,
        const CcInk2Seg *__restrict__ segs, const int2 *__restrict__ tile_map, CcRelAttn p) {
#if CC_FA_BODY
    __shared__ __align__(128) unsigned char kv_sm[2 * 64 * 256];
    __shared__ __align__(16) float R_sm[64][16];
    __shared__ __align__(16) float relp_s[128][17];
    unsigned char *K_sm = kv_sm;
    unsigned char *V_sm = kv_sm + 64 * 256;
    const unsigned ksm0 = (unsigned)__cvta_generic_to_shared(K_sm);
    const unsigned vsm0 = (unsigned)__cvta_generic_to_shared(V_sm);

    const int h = (int)blockIdx.x;
    const int2 tm = tile_map[blockIdx.y];
    const CcInk2Seg seg = segs[tm.x];
    const CcInk2KvRow kv = tab[tm.x];
    const int q_start = tm.y;
    const int S_seg = seg.len;
    const int pos_base = seg.pos_base;
    if (q_start >= S_seg) return;
    const int kvh = h / (p.H / p.KVH);
    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31, g = lane >> 2, tig = lane & 3;
    const int dr = p.d_rel;

    #pragma unroll
    for (int idx = tid; idx < 64 * 16; idx += 128) {
        const int rq = idx >> 4, c = idx & 15, sq = q_start + rq;
        uint4 h8 = make_uint4(0u, 0u, 0u, 0u);
        if (sq < S_seg) {
            const float4 *qp = reinterpret_cast<const float4 *>(q + p.q_off + (long)(seg.start + sq) * p.q_row + h * 128) + 2 * c;
            h8 = cc_fa_f8_to_h8(__ldg(qp), __ldg(qp + 1));
        }
        *reinterpret_cast<uint4 *>(K_sm + cc_fa_swz(rq, c)) = h8;
    }
    if (dr > 0) {
        for (int idx = tid; idx < 64 * dr; idx += 128) {
            const int rq = idx / dr, j = idx - rq * dr, sq = q_start + rq;
            R_sm[rq][j] = (sq < S_seg) ? __ldg(r + p.r_off + (long)(seg.start + sq) * p.r_row + h * dr + j) : 0.f;
        }
    }
    __syncthreads();

    unsigned qA[8][4];
    const int ar = lane & 15, ac = lane >> 4;
    #pragma unroll
    for (int kk = 0; kk < 8; kk++) cc_fa_ldm4(qA[kk], ksm0 + cc_fa_swz(warp * 16 + ar, 2 * kk + ac));

    const int r0 = warp * 16 + g, r1 = r0 + 8;
    const int sq0 = q_start + r0, sq1 = q_start + r1;
    const bool valid0 = sq0 < S_seg, valid1 = sq1 < S_seg;
    const int gsq0 = seg.start + sq0, gsq1 = seg.start + sq1;
    const int qpos0 = pos_base + sq0, qpos1 = pos_base + sq1;
    const int wmin0 = p.window > 0 ? max(0, qpos0 - p.window + 1) : 0;
    const int wmin1 = p.window > 0 ? max(0, qpos1 - p.window + 1) : 0;
    const float ts0 = valid0 ? __ldg(tau + p.tau_off + gsq0) : 1.f;
    const float ts1 = valid1 ? __ldg(tau + p.tau_off + gsq1) : 1.f;
    float r0_f[16], r1_f[16];
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        r0_f[j] = (j < dr) ? R_sm[r0][j] : 0.f;
        r1_f[j] = (j < dr) ? R_sm[r1][j] : 0.f;
    }

    float m0 = (p.has_sink && valid0) ? __ldg(tau + p.sink_off + gsq0 * p.H + h) : -3.0e38f;
    float m1 = (p.has_sink && valid1) ? __ldg(tau + p.sink_off + gsq1 * p.H + h) : -3.0e38f;
    float l0 = p.has_sink ? 1.f : 0.f, l1 = p.has_sink ? 1.f : 0.f;
    float O[16][4];
    #pragma unroll
    for (int nd = 0; nd < 16; nd++) { O[nd][0] = 0.f; O[nd][1] = 0.f; O[nd][2] = 0.f; O[nd][3] = 0.f; }
    __syncthreads();

    const int q_end = min(q_start + 63, S_seg - 1);
    const int t_min = p.window > 0 ? max(0, pos_base + q_start - p.window + 1) : 0;
    const int t_max = pos_base + q_end;
    const int bn = (lane & 7) + ((lane >> 4) << 3), bc = (lane >> 3) & 1;
    const int vk_off = lane & 15, vn_off = lane >> 4;

    for (int tb = t_min; tb <= t_max; tb += 64) {
        #pragma unroll
        for (int idx = tid; idx < 64 * 16; idx += 128) {
            const int rk = idx >> 4, c = idx & 15, t = tb + rk;
            uint4 kh8 = make_uint4(0u, 0u, 0u, 0u), vh8 = make_uint4(0u, 0u, 0u, 0u);
            if (t <= t_max) {
                if (t < pos_base) {
                    const long ridx = (long)kvh * kv.cap + (long)(t % kv.cap);
                    const float ksc = __ldg(kv.k_sc + ridx), vsc = __ldg(kv.v_sc + ridx);
                    const char4 *kq = reinterpret_cast<const char4 *>(static_cast<const signed char *>(kv.k_q) + ridx * 128) + 2 * c;
                    const char4 *vq = reinterpret_cast<const char4 *>(static_cast<const signed char *>(kv.v_q) + ridx * 128) + 2 * c;
                    const char4 ka = __ldg(kq), kb = __ldg(kq + 1), va = __ldg(vq), vb = __ldg(vq + 1);
                    kh8 = cc_fa_f8_to_h8(make_float4(ka.x * ksc, ka.y * ksc, ka.z * ksc, ka.w * ksc),
                                         make_float4(kb.x * ksc, kb.y * ksc, kb.z * ksc, kb.w * ksc));
                    vh8 = cc_fa_f8_to_h8(make_float4(va.x * vsc, va.y * vsc, va.z * vsc, va.w * vsc),
                                         make_float4(vb.x * vsc, vb.y * vsc, vb.z * vsc, vb.w * vsc));
                } else {
                    const long off = (long)(seg.start + (t - pos_base)) * p.kv_row + kvh * 128;
                    const float4 *kp4 = reinterpret_cast<const float4 *>(kvs + p.ks_off + off) + 2 * c;
                    const float4 *vp4 = reinterpret_cast<const float4 *>(kvs + p.vs_off + off) + 2 * c;
                    kh8 = cc_fa_f8_to_h8(__ldg(kp4), __ldg(kp4 + 1));
                    vh8 = cc_fa_f8_to_h8(__ldg(vp4), __ldg(vp4 + 1));
                }
            }
            const unsigned sw = cc_fa_swz(rk, c);
            *reinterpret_cast<uint4 *>(K_sm + sw) = kh8;
            *reinterpret_cast<uint4 *>(V_sm + sw) = vh8;
        }

        const int d0 = (pos_base + q_start) - tb - 63;
        const bool use_rel = (dr > 0) && (d0 < p.ext) && (d0 + 126 >= 0);
        if (use_rel) {
            const int dist = d0 + tid;
            if (dist >= 0 && dist < p.ext) {
                #pragma unroll
                for (int j = 0; j < 16; j++)
                    if (j < dr) relp_s[tid][j] = __ldg(relp + p.relp_off + (long)j * p.ext + dist);
            } else {
                #pragma unroll
                for (int j = 0; j < 16; j++) if (j < dr) relp_s[tid][j] = 0.f;
            }
        }
        __syncthreads();

        float S[8][4];
        #pragma unroll
        for (int j = 0; j < 8; j++) { S[j][0] = 0.f; S[j][1] = 0.f; S[j][2] = 0.f; S[j][3] = 0.f; }
        #pragma unroll
        for (int kk = 0; kk < 8; kk++) {
            #pragma unroll
            for (int jp = 0; jp < 4; jp++) {
                unsigned kB[4];
                cc_fa_ldm4(kB, ksm0 + cc_fa_swz(jp * 16 + bn, 2 * kk + bc));
                cc_fa_mma(S[2 * jp],     qA[kk], kB[0], kB[1]);
                cc_fa_mma(S[2 * jp + 1], qA[kk], kB[2], kB[3]);
            }
        }

        float tm0 = -3.0e38f, tm1 = -3.0e38f;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            #pragma unroll
            for (int c = 0; c < 2; c++) {
                const int rk = j * 8 + tig * 2 + c, t = tb + rk;
                if (valid0 && t >= wmin0 && t <= qpos0) {
                    float b0 = 0.f;
                    if (use_rel && (qpos0 - t) < p.ext) {
                        const int ridx = r0 - rk + 63;
                        #pragma unroll
                        for (int jj = 0; jj < 16; jj++) if (jj < dr) b0 += r0_f[jj] * relp_s[ridx][jj];
                    }
                    const float sv = ts0 * (S[j][c] * p.scale + b0);
                    S[j][c] = sv; tm0 = fmaxf(tm0, sv);
                } else S[j][c] = -3.0e38f;
                if (valid1 && t >= wmin1 && t <= qpos1) {
                    float b1 = 0.f;
                    if (use_rel && (qpos1 - t) < p.ext) {
                        const int ridx = r1 - rk + 63;
                        #pragma unroll
                        for (int jj = 0; jj < 16; jj++) if (jj < dr) b1 += r1_f[jj] * relp_s[ridx][jj];
                    }
                    const float sv = ts1 * (S[j][2 + c] * p.scale + b1);
                    S[j][2 + c] = sv; tm1 = fmaxf(tm1, sv);
                } else S[j][2 + c] = -3.0e38f;
            }
        }
        tm0 = fmaxf(tm0, __shfl_xor_sync(0xffffffffu, tm0, 1));
        tm0 = fmaxf(tm0, __shfl_xor_sync(0xffffffffu, tm0, 2));
        tm1 = fmaxf(tm1, __shfl_xor_sync(0xffffffffu, tm1, 1));
        tm1 = fmaxf(tm1, __shfl_xor_sync(0xffffffffu, tm1, 2));

        const float mn0 = fmaxf(m0, tm0), mn1 = fmaxf(m1, tm1);
        const float corr0 = (m0 > -1.0e30f) ? expf(m0 - mn0) : 0.f;
        const float corr1 = (m1 > -1.0e30f) ? expf(m1 - mn1) : 0.f;
        #pragma unroll
        for (int nd = 0; nd < 16; nd++) {
            O[nd][0] *= corr0; O[nd][1] *= corr0;
            O[nd][2] *= corr1; O[nd][3] *= corr1;
        }

        float ps0 = 0.f, ps1 = 0.f;
        unsigned pA[4][4];
        #pragma unroll
        for (int kk = 0; kk < 4; kk++) {
            const int j0 = 2 * kk, j1 = j0 + 1;
            const float p00 = (S[j0][0] > -1.0e30f) ? expf(S[j0][0] - mn0) : 0.f;
            const float p01 = (S[j0][1] > -1.0e30f) ? expf(S[j0][1] - mn0) : 0.f;
            const float p10 = (S[j0][2] > -1.0e30f) ? expf(S[j0][2] - mn1) : 0.f;
            const float p11 = (S[j0][3] > -1.0e30f) ? expf(S[j0][3] - mn1) : 0.f;
            const float q00 = (S[j1][0] > -1.0e30f) ? expf(S[j1][0] - mn0) : 0.f;
            const float q01 = (S[j1][1] > -1.0e30f) ? expf(S[j1][1] - mn0) : 0.f;
            const float q10 = (S[j1][2] > -1.0e30f) ? expf(S[j1][2] - mn1) : 0.f;
            const float q11 = (S[j1][3] > -1.0e30f) ? expf(S[j1][3] - mn1) : 0.f;
            ps0 += p00 + p01 + q00 + q01;
            ps1 += p10 + p11 + q10 + q11;
            pA[kk][0] = cc_fa_h2u(__floats2half2_rn(p00, p01));
            pA[kk][1] = cc_fa_h2u(__floats2half2_rn(p10, p11));
            pA[kk][2] = cc_fa_h2u(__floats2half2_rn(q00, q01));
            pA[kk][3] = cc_fa_h2u(__floats2half2_rn(q10, q11));
        }
        ps0 += __shfl_xor_sync(0xffffffffu, ps0, 1); ps0 += __shfl_xor_sync(0xffffffffu, ps0, 2);
        ps1 += __shfl_xor_sync(0xffffffffu, ps1, 1); ps1 += __shfl_xor_sync(0xffffffffu, ps1, 2);
        l0 = l0 * corr0 + ps0; m0 = mn0;
        l1 = l1 * corr1 + ps1; m1 = mn1;

        #pragma unroll
        for (int kk = 0; kk < 4; kk++) {
            #pragma unroll
            for (int nd = 0; nd < 8; nd++) {
                unsigned vB[4];
                cc_fa_ldm4_trans(vB, vsm0 + cc_fa_swz(kk * 16 + vk_off, 2 * nd + vn_off));
                cc_fa_mma(O[2 * nd],     pA[kk], vB[0], vB[1]);
                cc_fa_mma(O[2 * nd + 1], pA[kk], vB[2], vB[3]);
            }
        }
        __syncthreads();
    }

    const float inv0 = (l0 > 0.f) ? (1.f / l0) : 0.f;
    const float inv1 = (l1 > 0.f) ? (1.f / l1) : 0.f;
    if (valid0) {
        float *o0 = o + p.o_off + (long)gsq0 * p.o_row + h * 128;
        #pragma unroll
        for (int nd = 0; nd < 16; nd++)
            *reinterpret_cast<float2 *>(o0 + nd * 8 + tig * 2) = make_float2(O[nd][0] * inv0, O[nd][1] * inv0);
    }
    if (valid1) {
        float *o1 = o + p.o_off + (long)gsq1 * p.o_row + h * 128;
        #pragma unroll
        for (int nd = 0; nd < 16; nd++)
            *reinterpret_cast<float2 *>(o1 + nd * 8 + tig * 2) = make_float2(O[nd][2] * inv1, O[nd][3] * inv1);
    }
#else
    (void)q; (void)tab; (void)o; (void)kvs; (void)r; (void)relp; (void)tau; (void)segs; (void)tile_map; (void)p;
#endif
}

static int cc_sconv_packed(CcBuf *x, CcBuf *w, CcBuf *ring, CcBuf *segs, int nseg, const CcSconv *p) {
    CcCtx *c = live();
    if (!c || !p || !x || !w || !ring || !segs || nseg < 1 || p->CK < 1 || p->CK > 9) return 0;
    if (p->C <= 0) return 1;
    dim3 g((unsigned)((p->C + 127) / 128), (unsigned)nseg);
    coli_launch(ccg_sconv_packed_k, g, 128, 0, c->stream, fpw(x), fp(w), fpw(ring),
                reinterpret_cast<const CcInk2Seg *>(segs->d), *p);
    return launched(c, "sconv packed");
}
static int cc_ink2_dconv_packed(CcBuf *x, CcBuf *dyn, CcBuf *prm, CcBuf *ring, CcBuf *segs, int nseg, const CcInk2Dconv *p) {
    CcCtx *c = live();
    if (!c || !p || !x || !dyn || !prm || !ring || !segs || nseg < 1 || p->C < 1 || p->K < 1 || p->K > 9 || p->hd < 1 || p->C % p->hd) return 0;
    dim3 g((unsigned)((p->C + 127) / 128), (unsigned)nseg);
    coli_launch(ccg_ink2_dconv_packed_k, g, 128, 0, c->stream, fpw(x), fp(dyn), fp(prm), fpw(ring),
                reinterpret_cast<const CcInk2Seg *>(segs->d), *p);
    return launched(c, "ink2 dconv packed");
}
static int cc_relattn_q8_packed(CcBuf *q, CcBuf *kv_tab, size_t tab_off, CcBuf *o, CcBuf *kvs, CcBuf *r, CcBuf *relp, CcBuf *tau,
                                CcBuf *segs, CcBuf *tile_map, int ntiles, const CcRelAttn *p) {
    CcCtx *c = live();
    if (!c || !p || !q || !kv_tab || !o || !kvs || !r || !relp || !tau || !segs || !tile_map || ntiles < 1 ||
        p->hd != 128 || p->d_rel > 16 || p->KVH < 1 || p->H % p->KVH) return 0;
#if !(defined(__HIP__) || defined(__HIP_PLATFORM_AMD__))
    if (c->major < 8) return 0;
    static unsigned fa_pk_attr = 0;
    if (c->device >= 0 && c->device < 32 && !(fa_pk_attr & (1u << c->device))) {
        cudaFuncSetAttribute(ccg_relattn_flash_hd128_q8_packed_k, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
        (void)cudaGetLastError();
        fa_pk_attr |= 1u << c->device;
    }
    const CcInk2KvRow *tab = reinterpret_cast<const CcInk2KvRow *>(kv_tab->d) + tab_off;
    dim3 gf((unsigned)p->H, (unsigned)ntiles);
    coli_launch(ccg_relattn_flash_hd128_q8_packed_k, gf, 128, 0, c->stream, fp(q), tab, fpw(o),
                fp(kvs), fp(r), fp(relp), fp(tau), reinterpret_cast<const CcInk2Seg *>(segs->d),
                reinterpret_cast<const int2 *>(tile_map->d), *p);
    return launched(c, "relattn q8 packed");
#else
    return 0;
#endif
}
static int cc_ink2_kv_put_q8_packed(CcBuf *kv_tab, size_t tab_off, CcBuf *kvs, CcBuf *segs, CcBuf *row_seg, int N, int KV, int hd, int vs_off) {
    CcCtx *c = live();
    if (!c || !kv_tab || !kvs || !segs || !row_seg || N < 1 || KV < 1 || hd < 1) return 0;
    const CcInk2KvRow *tab = reinterpret_cast<const CcInk2KvRow *>(kv_tab->d) + tab_off;
    const long nwarps = (long)N * KV;
    dim3 g; cg_grid((nwarps + 7) / 8, &g);
    coli_launch(ccg_ink2_kv_put_q8_packed_k, g, 256, 0, c->stream, tab, fp(kvs),
                reinterpret_cast<const CcInk2Seg *>(segs->d), reinterpret_cast<const int2 *>(row_seg->d),
                N, KV, hd, vs_off);
    return launched(c, "ink2 kv put q8 packed");
}
