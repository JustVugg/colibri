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
        x[xi] = acc + cur;
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
/* Commit pass shared by both sconv and dconv: saves x[S - nh .. S - 1] into ring before writing tmp -> x. */
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
/* Vectorized float4 commit pass when C, x_off, x_row are multiples of 4. */
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
 *   - Relative position bias bank relp[d_rel, ext] staged transposed as half2[128][9] per tile,
 *     skipped entirely when tile distance >= ext; R*relp dot computed in packed __hfma2.
 *   - Online softmax reduces across 4-lane quads via __shfl_xor_sync (zero shared mem / zero barriers).
 *   - P converts directly from FP32 S accumulators to FP16 m16n8k16 A-fragments in registers,
 *     and P*V uses hardware-transposed ldmatrix.m8n8.x4.trans on V_sm. */
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
    __shared__ __align__(16) __half2 R_sm[64][8];                   /* 2 KB */
    __shared__ __align__(16) __half2 relp_s[128][9];                /* 4.6 KB (padded stride 9) */
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
    const int dr2 = p.d_rel >> 1;

    /* 1. Stage Q[q_start..+64][128] into K_sm (FP32 -> FP16) and R[q_start..+64][d_rel] into R_sm */
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
    if (dr2 > 0) {
        for (int idx = tid; idx < 64 * dr2; idx += 128) {
            const int rq = idx / dr2, j = idx - rq * dr2, sq = q_start + rq;
            __half2 rh = __float2half2_rn(0.f);
            if (sq < p.S) {
                const float *rp = r + p.r_off + (long)sq * p.r_row + h * p.d_rel + 2 * j;
                rh = __floats2half2_rn(cc_fa_clamp(__ldg(rp)), cc_fa_clamp(__ldg(rp + 1)));
            }
            R_sm[rq][j] = rh;
        }
    }
    __syncthreads();

    /* 2. Load this warp's 16x128 Q into registers qA[8][4] and lane's R rows into r0_h/r1_h */
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
    __half2 r0_h[8], r1_h[8];
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        r0_h[j] = (j < dr2) ? R_sm[r0][j] : __float2half2_rn(0.f);
        r1_h[j] = (j < dr2) ? R_sm[r1][j] : __float2half2_rn(0.f);
    }

    float m0 = (p.has_sink && valid0) ? __ldg(tau + p.sink_off + sq0 * p.H + h) : -3.0e38f;
    float m1 = (p.has_sink && valid1) ? __ldg(tau + p.sink_off + sq1 * p.H + h) : -3.0e38f;
    float l0 = p.has_sink ? 1.f : 0.f;
    float l1 = p.has_sink ? 1.f : 0.f;
    float O[16][4];
    #pragma unroll
    for (int nd = 0; nd < 16; nd++) { O[nd][0] = 0.f; O[nd][1] = 0.f; O[nd][2] = 0.f; O[nd][3] = 0.f; }
    __syncthreads();

    /* 3. Iterate over valid KV tiles [t_min .. t_max] in blocks of BN=64 */
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
        const bool use_rel = (dr2 > 0) && (d0 < p.ext) && (d0 + 126 >= 0);
        if (use_rel) {
            const int dist = d0 + tid;
            if (dist >= 0 && dist < p.ext) {
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    if (j < dr2) {
                        const float e0 = __ldg(relp + p.relp_off + (long)(2 * j) * p.ext + dist);
                        const float e1 = __ldg(relp + p.relp_off + (long)(2 * j + 1) * p.ext + dist);
                        relp_s[tid][j] = __floats2half2_rn(cc_fa_clamp(e0), cc_fa_clamp(e1));
                    }
                }
            } else {
                #pragma unroll
                for (int j = 0; j < 8; j++) if (j < dr2) relp_s[tid][j] = __float2half2_rn(0.f);
            }
        }
        __syncthreads();

        /* S[16x64] = Q[16x128] * K[64x128]^T */
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

        /* Add relative bias, apply causal + window masks, scale by tau, find quad max */
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
                        __half2 bh = __float2half2_rn(0.f);
                        #pragma unroll
                        for (int jj = 0; jj < 8; jj++) if (jj < dr2) bh = __hfma2(r0_h[jj], relp_s[ridx][jj], bh);
                        b0 = __low2float(bh) + __high2float(bh);
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
                        __half2 bh = __float2half2_rn(0.f);
                        #pragma unroll
                        for (int jj = 0; jj < 8; jj++) if (jj < dr2) bh = __hfma2(r1_h[jj], relp_s[ridx][jj], bh);
                        b1 = __low2float(bh) + __high2float(bh);
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

        /* O[16x128] += P[16x64] * V[64x128] */
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

    /* 4. Normalize and store coalesced float2 outputs */
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
/* one block of 128 per (head, row): chain_relattn.comp's online softmax over tiles of 128 positions.
 * part (the split, a decode step's rows): block z takes tile z alone, its max, sum and weighted V into
 * part (hd + 2 floats a (row, head, tile)), and ccg_relattn_join_k joins the tiles */
__global__ static void ccg_relattn_k(const float *q, const float *kc, const float *vc, float *o, const float *kvs,
                                     const float *r, const float *relp, const float *tau, CcRelAttn p, float *part) {
    const int h = (int)blockIdx.x, s = (int)blockIdx.y, tid = (int)threadIdx.x;
    const int kvh = h / (p.H / p.KVH);
    __shared__ float qs[256], rs[64], pe[128], red[128];
    __shared__ int pt[128];
    const int qb = p.q_off + s * p.q_row + h * p.hd;
    for (int d = tid; d < p.hd; d += 128) qs[d] = q[qb + d];
    for (int j = tid; j < p.d_rel; j += 128) rs[j] = r[p.r_off + s * p.r_row + h * p.d_rel + j];
    const int qpos = p.pos_base + s;
    const int t0 = p.window > 0 ? max(0, qpos - p.window + 1) : 0;
    const int n = qpos - t0 + 1;
    const float ts = tau[p.tau_off + s];
    __syncthreads();
    float m = -3.0e38f, l = 0.f, acc0 = 0.f, acc1 = 0.f;
    if (part) {
        const int z = (int)blockIdx.z, b0 = z * 128;
        if (b0 < n) ccg_relattn_tile(kc, vc, kvs, relp, p, qs, rs, pe, red, pt, kvh, qpos, t0, n, ts, b0, tid, m, l, acc0, acc1);
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

/* Fast warp-shuffle clip for n == 128 (8 segments/block, 1 warp/segment, float4, zero __syncthreads) */
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
/* Fast half-warp clip for n == 16 (16 segments/block, 16 lanes/segment, zero __syncthreads) */
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

static float *cc_conv_tmp_get(CcCtx *c, size_t bytes) {
    if (c->conv_tmp && c->conv_tmp_bytes >= bytes) return c->conv_tmp;
    if (c->conv_tmp) { (void)cudaStreamSynchronize(c->stream); cudaFree(c->conv_tmp); c->conv_tmp = nullptr; c->conv_tmp_bytes = 0; }
    size_t cap = bytes < (1u << 22) ? (1u << 22) : bytes;
    if (cudaMalloc((void **)&c->conv_tmp, cap) != cudaSuccess) { (void)cudaGetLastError(); return nullptr; }
    c->conv_tmp_bytes = cap;
    return c->conv_tmp;
}
static int cc_sconv(CcBuf *x, CcBuf *w, CcBuf *ring, const CcSconv *p) {
    CcCtx *c = live();
    if (!c || !p || !x || (p->mode == 0 && (p->CK < 1 || p->CK > 9 || !w || !ring))) return 0;
    const long n = p->mode == 0 ? p->C : p->n;
    if (n <= 0) return 1;
    if (p->mode == 0 && p->S >= 16) {
        const size_t need = (size_t)p->S * (size_t)p->C * sizeof(float);
        float *tmp = cc_conv_tmp_get(c, need);
        if (tmp) {
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
            return launched(c, "sconv 2d");
        }
    }
    dim3 g; cg_grid((n + 63) / 64, &g);
    coli_launch(ccg_sconv_k, g, 64, 0, c->stream, fpw(x), p->mode == 0 ? fp(w) : fp(x), p->mode == 0 ? fpw(ring) : fpw(x), *p);
    return launched(c, "sconv");
}
static int cc_relattn_flash_on(void) {
    static int on = -1;
    if (on < 0) { const char *e = getenv("COLI_CUDA_RELATTN_FLASH"); on = !(e && *e == '0'); }
    return on;
}
static int cc_relattn(CcBuf *q, CcBuf *kc, CcBuf *vc, CcBuf *o, CcBuf *kvs, CcBuf *r, CcBuf *relp, CcBuf *tau, const CcRelAttn *p) {
    CcCtx *c = live();
    if (!c || !p || p->hd > 256 || p->hd < 1 || p->d_rel > 64 || p->d_rel < 0 || p->KVH < 1 || p->H % p->KVH || p->cap < 1) return 0;
    if (p->S <= 0) return 1;
#if !(defined(__HIP__) || defined(__HIP_PLATFORM_AMD__))
    if (cc_relattn_flash_on() && c->major >= 8 && p->S >= 16 && p->hd == 128 && p->d_rel <= 16 && (p->d_rel & 1) == 0 &&
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
                        fp(kvs), fp(r), fp(relp), fp(tau), *p, part);
            coli_launch(ccg_relattn_join_k, dim3((unsigned)p->H, (unsigned)p->S), 256, 0, c->stream, (const float *)part, fp(tau), fpw(o), *p, nc);
            (void)cudaFreeAsync(part, c->stream);
            return launched(c, "relattn split");
        }
        (void)cudaGetLastError();
    }
    coli_launch(ccg_relattn_k, dim3((unsigned)p->H, (unsigned)p->S), 128, 0, c->stream, fp(q), fp(kc), fp(vc), fpw(o), fp(kvs),
                fp(r), fp(relp), fp(tau), *p, (float *)nullptr);
    return launched(c, "relattn");
}
static int cc_ink2_mix(CcBuf *x, CcBuf *blk, CcBuf *prm, CcBuf *y, const CcInk2Mix *p) {
    CcCtx *c = live();
    if (!c || !p || !x || !prm || !y || p->S < 0 || p->D < 1 || p->nb < 0 || p->nb > 15 || (p->nb > 0 && !blk) || y == x || y == blk) return 0;
    if (p->S == 0) return 1;
    dim3 g; cg_grid(p->S, &g);
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
    if (p->S >= 16) {
        const size_t need = (size_t)p->S * (size_t)p->C * sizeof(float);
        float *tmp = cc_conv_tmp_get(c, need);
        if (tmp) {
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
            return launched(c, "ink2 dconv 2d");
        }
    }
    dim3 g; cg_grid(((long)p->C + 255) / 256, &g);
    coli_launch(ccg_ink2_dconv_k, g, 256, 0, c->stream, fpw(x), fp(dyn), fp(prm), fpw(ring), *p);
    return launched(c, "ink2 dconv");
}
