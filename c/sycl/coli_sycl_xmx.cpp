/* coli_sycl_xmx.cpp — the XMX (DPAS) GEMM family of coli_sycl.h, section 4: what a prompt chunk's
 * rows go through in place of the CUDA build's WMMA tiles (hcg_dense_k, hcg2_dense_k, hcg_moe_*_k,
 * compiled out of the migration by COLI_GPU_PORTABLE_ONLY).
 *
 * The tile, after hcg_tile's: a work-group of 256 work-items takes 64 rows x 64 outputs of each of
 * the NW matrices (one, or gate and up with the GLU epilogue), K in steps of 64 through local
 * memory. The weights (int8 rows, int4 rows, int4 per gs inputs; the experts' planar int4) are
 * fetched as their codes into registers a step ahead, decoded to f16 and written into the step's
 * local tile while the matrix engines work on the current one (two tiles, one barrier a step).
 * The activations are not staged by the tile: a small kernel before the GEMM rounds them once to
 * f16 (hcg_h's rounding, held to +-65504) into a padded copy that every tile along N reads with
 * joint_matrix_load straight from global memory (each row tile is read by every output tile; the
 * rounding is paid once instead of once per output tile, and the local memory holds weights only).
 *
 * Sub-group 16, always: the matrix engines' N is the sub-group size on every Intel part (DPAS
 * computes 16 outputs a sub-group), and joint_matrix's Intel shapes are listed at 16 only. A
 * sub-group holds TM rows x 32 outputs (TM 8, the 8x16x16 shape every XMX part lists: the DPAS
 * repeat count is at most 8) or TM rows x 16 outputs (TM 16, the compound 16x16x16 shape pvc
 * and bmg list, two DPAS; COLI_SYCL_XMX_TM=16 where the profile has it). The f16 operands:
 *   A  joint_matrix<half, use::a, TM, 16, row_major>      from the f16 copy, stride K
 *   B  joint_matrix<half, use::b, 16, 16, ext_intel_packed> from the local tile, VNNI packed:
 *      the pair (k, k + 1) of output n is one 32-bit word at packed row k / 2, column 2n, so a
 *      work-item's 16 consecutive inputs of one output row are eight word stores down a column
 *   C  joint_matrix<float, use::accumulator, TM, 16>, f32 sums over the whole K
 * and the int8 ones (a_s8, fmt 1 and 2, where the device lists s8 x s8 -> s32 at 8x16x32): the
 * activations quantized once per row (absmax / 127) by the staging kernel,
 *   A  joint_matrix<int8_t, use::a, 8, 32>, B <int8_t, use::b, 32, 16, ext_intel_packed> (the
 *   quad (k .. k + 3) of output n one word at packed row k / 4, column 4n), C <int32_t, 8, 16>,
 * the row's and the weight row's scales applied in the epilogue (not bit-identical to the f16
 * path: moe's quantize_s4_rows policy).
 *
 * The epilogue reads the accumulators in place with the Intel joint_matrix_apply that names each
 * element's (row, column): int8 and int4 rows went in as their codes, exact in f16, and take the
 * row's scale here; int4 per gs took the group's scale at the staging (the product rounded to f16,
 * hcg_store_w's arithmetic); gate and up then silu(g) * u. Nothing of the tile goes back through
 * local memory. M tails: rows past M are read from the copy's zero padding and not stored; N
 * tails: weight rows past N stage as zeros and are not stored. K a multiple of 64.
 *
 * A device without the f16 shape (COLI_SYCL_XMX=0, a part whose matrix_combinations is empty) runs
 * the same staging through a plain tile (the activations' tile in local memory too, 16 outputs a
 * work-item in f32 FMAs over the same f16 values): the API takes every shape _ok takes.
 *
 * The routed experts (coli_sycl_xmx_moe): one work-group per (sort tile, output tile), the tile's
 * expert from the sort (te, tr), its rows base[e] + tr .. of the slot order; the staging kernel
 * gathers the slots' x rows into the f16 copy in slot order (phase 0) or converts hid's rows
 * (phase 1), so a tile's rows are consecutive in the copy and its outputs go to hid's places
 * base[e] + tr + r, as hcg_moe_gu_k / hcg_moe_dn_k wrote them.
 *
 * The profile decides (coli_sycl_profile.h): has_xmx and the shape flags from matrix_combinations
 * (coli_sycl_xmx_shape_flags), coli_sycl_xmx_supported over them; this unit only reads them.
 *
 * Host side: a grow-only scratch per (device, context) holds the activations' copy (an in-order
 * queue runs the staging and the GEMM back to back, so one copy serves every call on the queue;
 * growing waits for the queue first). Every submit is in a try block: a SYCL exception logs,
 * returns 0 and the caller launches its fallback. Nothing here touches a device at load time. */
#include "coli_sycl.h"

#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

namespace jm = sycl::ext::oneapi::experimental::matrix;
namespace jmi = sycl::ext::intel::experimental::matrix;

namespace {

/* ---- the tile ----------------------------------------------------------------------- */
constexpr int BM = 64, BN = 64, BK = 64;   /* rows, outputs, inputs a work-group step */
constexpr int WG = 256, SG = 16;           /* work-items a work-group, the sub-group: 16 sub-groups */
constexpr int LDB_H = BN * 2 + 8;          /* halves a packed B row: 272 bytes, the pad spreads the banks */
constexpr int LDB_S = BN * 4 + 16;         /* bytes a packed s8 B row: 272 bytes */
constexpr int LDA_H = BK + 8;              /* halves a row of the plain tile's A (fallback only): 144 bytes */
constexpr int PACK_ROWS = 8;               /* packed B rows a K step of 16 halves or 32 bytes (both 8) */

/* f16 rounding of an activation: NaN stays one, the rest held to the f16 range (hcg_h) */
inline sycl::half h_sat(float f) {
    return sycl::half(f != f ? f : sycl::fmin(sycl::fmax(f, -65504.f), 65504.f));
}
inline uint32_t pack2(sycl::half a, sycl::half b) {
    return (uint32_t)sycl::bit_cast<uint16_t>(a) | ((uint32_t)sycl::bit_cast<uint16_t>(b) << 16);
}
/* a signed nibble (the backend's layout: offset_to_signed_s4 at upload) and a planar one (v + 8) */
inline int nib_signed(unsigned s) { return (int)((s & 15u) ^ 8u) - 8; }
inline int nib_planar(unsigned s) { return (int)(s & 15u) - 8; }

/* ---- what a work-group's kernel takes (uniform) ----------------------------------------- */
struct XmxArgs {
    float *Y; long ldc;              /* row (a_row0 + r) of the output: Y + row * ldc + n */
    const void *A; long lda;         /* the activations' copy (f16 or s8), lda elements a row */
    const float *a_scale;            /* s8: the rows' scales (NULL for f16) */
    const uint8_t *W[2]; const float *sc[2];   /* dense: the matrices (NW of them) and their scales */
    int gs, ng, K, N;                /* fmt 4: a scale per gs inputs, ng a row; K inputs, N outputs */
    int M;                           /* dense: the rows */
    int moe, phase, t_off;           /* the routed experts: tiles of the sort, the table's entries */
    const CcExpert *tab;
    const int *base, *te, *tr, *ntile;
};

/* a work-item's share of a step's weights: row n = n0 + (t >> 2), inputs k0 + 16 (t & 3) .. + 15,
 * as hcg_fetch_w takes them (four lanes read one row's 32 or 64 contiguous bytes) */
struct WReg { uint32_t q[4]; float sv; };

template <int WF>
inline void fetch_w(WReg &r, const uint8_t *W, const float *sc, int gs, int ng, int n, int N, int k0, int K, int kq, int planar) {
    r.q[0] = r.q[1] = r.q[2] = r.q[3] = 0; r.sv = 0.f;
    if (n >= N) return;
    if (WF == 1) {
        const uint32_t *p = reinterpret_cast<const uint32_t *>(W + (size_t)n * K + k0 + kq);
        r.q[0] = p[0]; r.q[1] = p[1]; r.q[2] = p[2]; r.q[3] = p[3]; r.sv = 1.f;
    } else if (WF == 4 && planar) {   /* 16 bytes of block k0 / 64: low nibbles (kq < 32) or high */
        const uint32_t *p = reinterpret_cast<const uint32_t *>(W + ((size_t)n * K + k0) / 2 + (kq & 31));
        r.q[0] = p[0]; r.q[1] = p[1]; r.q[2] = p[2]; r.q[3] = p[3];
        r.sv = sc[(size_t)n * ng + (k0 + kq) / gs];
    } else {   /* int4 pairs: weight 2j low nibble, 2j + 1 high */
        const uint32_t *p = reinterpret_cast<const uint32_t *>(W + ((size_t)n * K + k0 + kq) / 2);
        r.q[0] = p[0]; r.q[1] = p[1];
        r.sv = WF == 4 ? sc[(size_t)n * ng + (k0 + kq) / gs] : 1.f;
    }
}
/* the 16 weights of the share as f16, into the packed tile at rows kq / 2 .., column 2n */
template <int WF>
inline void store_w_h(sycl::half *bs, const WReg &r, int n, int kq, int planar) {
    uint32_t *dst = reinterpret_cast<uint32_t *>(bs + (size_t)(kq / 2) * LDB_H + n * 2);
    if (WF == 4 && planar) {
        const int sh = kq < 32 ? 0 : 4;
        #pragma unroll
        for (int p = 0; p < 8; p++) {
            unsigned b0 = (r.q[p >> 1] >> (16 * (p & 1))) & 255u, b1 = (r.q[p >> 1] >> (16 * (p & 1) + 8)) & 255u;
            dst[(size_t)p * (LDB_H / 2)] = pack2(sycl::half((float)nib_planar(b0 >> sh) * r.sv), sycl::half((float)nib_planar(b1 >> sh) * r.sv));
        }
    } else if (WF == 1) {
        #pragma unroll
        for (int p = 0; p < 8; p++) {
            unsigned w = r.q[p >> 1] >> (16 * (p & 1));
            dst[(size_t)p * (LDB_H / 2)] = pack2(sycl::half((float)(int8_t)(w & 255u)), sycl::half((float)(int8_t)((w >> 8) & 255u)));
        }
    } else {
        #pragma unroll
        for (int p = 0; p < 8; p++) {
            unsigned b = (r.q[p >> 2] >> (8 * (p & 3))) & 255u;
            float a = (float)nib_signed(b), c = (float)nib_signed(b >> 4);
            if (WF == 4) { a *= r.sv; c *= r.sv; }
            dst[(size_t)p * (LDB_H / 2)] = pack2(sycl::half(a), sycl::half(c));
        }
    }
}
/* the same share as s8 codes (fmt 1 and 2: the scale is the row's, applied after the sum),
 * into the packed s8 tile at rows kq / 4 .., column 4n */
template <int WF>
inline void store_w_s(int8_t *bs, const WReg &r, int n, int kq) {
    uint32_t *dst = reinterpret_cast<uint32_t *>(bs + (size_t)(kq / 4) * LDB_S + n * 4);
    if (WF == 1) {
        #pragma unroll
        for (int p = 0; p < 4; p++) dst[(size_t)p * (LDB_S / 4)] = r.q[p];
    } else {
        #pragma unroll
        for (int p = 0; p < 4; p++) {
            unsigned w = (r.q[p >> 1] >> (16 * (p & 1))) & 0xffffu;   /* two bytes: four nibbles */
            uint32_t v = 0;
            #pragma unroll
            for (int j = 0; j < 4; j++) v |= (uint32_t)(uint8_t)nib_signed(w >> (4 * j)) << (8 * j);
            dst[(size_t)p * (LDB_S / 4)] = v;
        }
    }
}

/* ---- the GEMM kernel ------------------------------------------------------------------- */
/* WF the weights' format (1, 2, 4), NW 1 or 2 (gate and up: the GLU), TM the rows a sub-group
 * (8 or 16), TA the operands' type (sycl::half or int8_t), XMX the matrix engines (false: the
 * plain tile over the same staged values). Sub-group s of 16 takes rows TM (s % NRT) .. and
 * outputs CPG (s / NRT) .. of the 64 x 64 tile: TM 8: 8 row tiles x 2 column groups of 32 (two
 * accumulators a matrix); TM 16: 4 x 4 of 16 (one). */
template <int WF, int NW, int TM, typename TA, bool XMX>
struct XmxGemmK {
    XmxArgs a;
    sycl::local_accessor<TA, 1> bs;           /* [2][NW][PACK_ROWS * BK / KK][LDB] packed B */
    sycl::local_accessor<sycl::half, 1> as;   /* fallback: [2][BM][LDA_H] A; XMX: 1 element, unused */

    static constexpr bool S8 = std::is_same<TA, int8_t>::value;
    static constexpr int KK = S8 ? 32 : 16;                 /* inputs a matrix step */
    static constexpr int KP = S8 ? 4 : 2;                   /* operands a packed word */
    static constexpr int LDB = S8 ? LDB_S : LDB_H;          /* elements a packed row */
    static constexpr int BTILE = (BK / KP) * LDB;           /* elements a matrix's step tile */
    static constexpr int NRT = BM / TM, NCG = (WG / SG) / NRT, CPG = BN / NCG, NT = CPG / 16;
    using TC = typename std::conditional<S8, int32_t, float>::type;
    using MatA = jm::joint_matrix<sycl::sub_group, TA, jm::use::a, TM, KK, jm::layout::row_major>;
    using MatB = jm::joint_matrix<sycl::sub_group, TA, jm::use::b, KK, 16, jm::layout::ext_intel_packed>;
    using MatC = jm::joint_matrix<sycl::sub_group, TC, jm::use::accumulator, TM, 16>;
    using MatF = jm::joint_matrix<sycl::sub_group, float, jm::use::accumulator, TM, 16>;
    static constexpr size_t bs_elems() { return (size_t)2 * NW * BTILE; }
    static constexpr size_t as_elems() { return XMX ? 1 : (size_t)2 * BM * LDA_H; }
    static constexpr size_t slm_bytes() { return bs_elems() * sizeof(TA) + as_elems() * sizeof(sycl::half); }

    /* the epilogue of one accumulator pair (gate, and up with NW 2) at rows rt * TM .., outputs
     * c0 ..: the row scales (fmt 1, 2; the s8 rows'), the GLU, the stores to the rows' places */
    void epilogue(sycl::sub_group sg, MatF &g, MatF *u, int rt, int c0, int n0, int M, long a_row0,
                  const float *const *sc) const {
        const float *Y_sc0 = WF != 4 ? sc[0] : nullptr, *Y_sc1 = WF != 4 && NW == 2 ? sc[1] : nullptr;
        if (NW == 2) {
            if (Y_sc1) jmi::joint_matrix_apply(sg, *u, [&](float &v, size_t row, size_t col) {
                int n = n0 + c0 + (int)col; (void)row;
                if (n < a.N) v *= Y_sc1[n];
            });
            jmi::joint_matrix_apply(sg, g, [&](float &v, size_t row, size_t col) {
                int n = n0 + c0 + (int)col, r = rt * TM + (int)row;
                float x = v;
                if (Y_sc0 && n < a.N) x *= Y_sc0[n];
                if (a.a_scale) x *= a.a_scale[a_row0 + r];
                v = x / (1.f + sycl::exp(-x));
            });
            if (a.a_scale) jmi::joint_matrix_apply(sg, *u, [&](float &v, size_t row, size_t col) {
                (void)col; v *= a.a_scale[a_row0 + rt * TM + (int)row];
            });
            jm::joint_matrix_apply(sg, g, *u, [](float &x, float &y) { x *= y; });
            jmi::joint_matrix_apply(sg, g, [&](float &v, size_t row, size_t col) {
                int n = n0 + c0 + (int)col, r = rt * TM + (int)row;
                if (r < M && n < a.N) a.Y[(a_row0 + r) * a.ldc + n] = v;
            });
        } else {
            jmi::joint_matrix_apply(sg, g, [&](float &v, size_t row, size_t col) {
                int n = n0 + c0 + (int)col, r = rt * TM + (int)row;
                if (r < M && n < a.N) {
                    float x = v;
                    if (Y_sc0) x *= Y_sc0[n];
                    if (a.a_scale) x *= a.a_scale[a_row0 + r];
                    a.Y[(a_row0 + r) * a.ldc + n] = x;
                }
            });
        }
    }

    [[sycl::reqd_sub_group_size(SG)]] void operator()(sycl::nd_item<2> it) const {
        const int t = (int)it.get_local_id(1), tile = (int)it.get_group(0), n0 = (int)it.get_group(1) * BN;
        /* the work-group's rows: a dense row tile, or a sort tile's rows of one expert (uniform) */
        int M; long a_row0; const uint8_t *W[2]; const float *sc[2]; int planar = 0;
        if (a.moe) {
            if (tile >= *a.ntile) return;   /* a tile the sort did not fill: uniform, before any barrier */
            const int e = a.te[tile], r0 = a.tr[tile];
            M = a.base[e + 1] - a.base[e] - r0; if (M > BM) M = BM;
            a_row0 = (long)a.base[e] + r0;
            const CcExpert ex = a.tab[a.t_off + e];
            if (a.phase == 0) { W[0] = (const uint8_t *)ex.g; sc[0] = ex.gs; W[1] = (const uint8_t *)ex.u; sc[1] = ex.us; }
            else { W[0] = (const uint8_t *)ex.d; sc[0] = ex.ds; W[1] = nullptr; sc[1] = nullptr; }
            planar = WF == 4 && ex.planar;
        } else {
            const int m0 = tile * BM;
            M = a.M - m0 < BM ? a.M - m0 : BM;
            a_row0 = m0;
            W[0] = a.W[0]; sc[0] = a.sc[0]; W[1] = a.W[1]; sc[1] = a.sc[1];
        }
        const int n = n0 + (t >> 2), kq = (t & 3) * 16;   /* this work-item's staging share */
        TA *bs0 = bs.template get_multi_ptr<sycl::access::decorated::no>().get();
        sycl::half *as0 = as.template get_multi_ptr<sycl::access::decorated::no>().get();
        (void)as0;   /* the plain tile's */
        auto bptr = [&](int buf, int w) { return bs.template get_multi_ptr<sycl::access::decorated::no>() + (size_t)(buf * NW + w) * BTILE; };
        auto stage = [&](int buf, const WReg *rw) {
            #pragma unroll
            for (int w = 0; w < NW; w++) {
                if constexpr (S8) store_w_s<WF>(bs0 + (size_t)(buf * NW + w) * BTILE, rw[w], t >> 2, kq);
                else store_w_h<WF>(bs0 + (size_t)(buf * NW + w) * BTILE, rw[w], t >> 2, kq, planar);
            }
        };
        const TA *A = reinterpret_cast<const TA *>(a.A);
        WReg rw[NW];
        #pragma unroll
        for (int w = 0; w < NW; w++) fetch_w<WF>(rw[w], W[w], sc[w], a.gs, a.ng, n, a.N, 0, a.K, kq, planar);

        if constexpr (XMX) {
            sycl::sub_group sg = it.get_sub_group();
            const int s = t / SG, rt = s % NRT, cg = s / NRT;
            MatC acc[NW][NT];
            #pragma unroll
            for (int w = 0; w < NW; w++) for (int j = 0; j < NT; j++) jm::joint_matrix_fill(sg, acc[w][j], (TC)0);
            stage(0, rw);
            it.barrier(sycl::access::fence_space::local_space);
            int buf = 0;
            for (int k0 = 0; k0 < a.K; k0 += BK) {
                const bool next = k0 + BK < a.K;
                if (next) {
                    #pragma unroll
                    for (int w = 0; w < NW; w++) fetch_w<WF>(rw[w], W[w], sc[w], a.gs, a.ng, n, a.N, k0 + BK, a.K, kq, planar);
                }
                const TA *arow = A + (a_row0 + rt * TM) * a.lda + k0;
                #pragma unroll
                for (int kk = 0; kk < BK / KK; kk++) {
                    MatA ma;
                    jm::joint_matrix_load(sg, ma, sycl::address_space_cast<sycl::access::address_space::global_space,
                                                                            sycl::access::decorated::no>(arow + kk * KK), a.lda);
                    #pragma unroll
                    for (int w = 0; w < NW; w++) {
                        #pragma unroll
                        for (int j = 0; j < NT; j++) {
                            MatB mb;
                            jm::joint_matrix_load(sg, mb, bptr(buf, w) + (size_t)kk * PACK_ROWS * LDB + (cg * CPG + j * 16) * KP, LDB);
                            jm::joint_matrix_mad(sg, acc[w][j], ma, mb, acc[w][j]);
                        }
                    }
                }
                if (next) stage(buf ^ 1, rw);   /* the other tile: its last readers passed the previous barrier */
                it.barrier(sycl::access::fence_space::local_space);
                buf ^= 1;
            }
            #pragma unroll
            for (int j = 0; j < NT; j++) {
                if constexpr (S8) {
                    MatF fg, fu;
                    jm::joint_matrix_copy(sg, acc[0][j], fg);
                    if (NW == 2) jm::joint_matrix_copy(sg, acc[NW - 1][j], fu);
                    epilogue(sg, fg, &fu, rt, cg * CPG + j * 16, n0, M, a_row0, sc);
                } else {
                    epilogue(sg, acc[0][j], &acc[NW - 1][j], rt, cg * CPG + j * 16, n0, M, a_row0, sc);
                }
            }
        } else {
            /* the plain tile: the A step staged as f16 too (row t >> 2, inputs kq ..), then work-item
             * t sums 16 rows (16 (t >> 6) ..) of output t & 63 in f32 over the same values */
            const int arow_l = t >> 2, rn = t & 63, r0 = (t >> 6) * 16;
            auto stage_x = [&](int buf, int k0) {
                const sycl::half *src = reinterpret_cast<const sycl::half *>(a.A) + (a_row0 + arow_l) * a.lda + k0 + kq;
                uint32_t *dst = reinterpret_cast<uint32_t *>(as0 + (size_t)buf * BM * LDA_H + arow_l * LDA_H + kq);
                const uint32_t *s32 = reinterpret_cast<const uint32_t *>(src);
                #pragma unroll
                for (int i = 0; i < 8; i++) dst[i] = s32[i];
            };
            float acc[NW][16];
            #pragma unroll
            for (int w = 0; w < NW; w++) for (int r = 0; r < 16; r++) acc[w][r] = 0.f;
            stage(0, rw); stage_x(0, 0);
            it.barrier(sycl::access::fence_space::local_space);
            int buf = 0;
            for (int k0 = 0; k0 < a.K; k0 += BK) {
                const bool next = k0 + BK < a.K;
                if (next) {
                    #pragma unroll
                    for (int w = 0; w < NW; w++) fetch_w<WF>(rw[w], W[w], sc[w], a.gs, a.ng, n, a.N, k0 + BK, a.K, kq, planar);
                }
                const sycl::half *At = as0 + (size_t)buf * BM * LDA_H + r0 * LDA_H;
                for (int k = 0; k < BK; k++) {
                    float b[NW];
                    #pragma unroll
                    for (int w = 0; w < NW; w++) b[w] = (float)bs0[(size_t)(buf * NW + w) * BTILE + (size_t)(k / 2) * LDB_H + rn * 2 + (k & 1)];
                    #pragma unroll
                    for (int r = 0; r < 16; r++) {
                        const float av = (float)At[r * LDA_H + k];
                        #pragma unroll
                        for (int w = 0; w < NW; w++) acc[w][r] += av * b[w];
                    }
                }
                if (next) { stage(buf ^ 1, rw); stage_x(buf ^ 1, k0 + BK); }
                it.barrier(sycl::access::fence_space::local_space);
                buf ^= 1;
            }
            const int nn = n0 + rn;
            if (nn < a.N) {
                const float s0 = WF != 4 ? sc[0][nn] : 1.f, s1 = WF != 4 && NW == 2 ? sc[1][nn] : 1.f;
                for (int r = 0; r < 16 && r0 + r < M; r++) {
                    float g = acc[0][r] * s0;
                    if (NW == 2) { float u = acc[NW - 1][r] * s1; g = (g / (1.f + sycl::exp(-g))) * u; }
                    a.Y[(a_row0 + r0 + r) * a.ldc + nn] = g;
                }
            }
        }
    }
};

/* ---- the activations' copy ------------------------------------------------------------- */
/* rows of the f16 copy, 8 halves a work-item: mode 0 dense f32 (src + r * lda), 1 dense f16, 2
 * the routed slots' x rows in slot order (row p < *nplaced is x[x_off + (order[p] / Kslots) *
 * x_row]), 3 hid's rows (f32, src + p * lda) in slot order; rows past M / *nplaced are zero */
struct StageArgs {
    sycl::half *dst; int K, rows;
    int mode; const void *src; long lda; int M;
    const int *order, *nplaced; int Kslots, x_off, x_row;
};
struct AStageK {
    StageArgs s;
    void operator()(sycl::nd_item<1> it) const {
        const long i = (long)it.get_global_id(0), per_row = s.K / 8;
        const long r = i / per_row, k0 = (i - r * per_row) * 8;
        if (r >= s.rows) return;
        sycl::vec<sycl::half, 8> v;
        const int live = s.mode >= 2 ? *s.nplaced : s.M;
        if (r >= live) { for (int j = 0; j < 8; j++) v[j] = sycl::half(0.f); }
        else if (s.mode == 1) {
            const sycl::half *p = reinterpret_cast<const sycl::half *>(s.src) + r * s.lda + k0;
            for (int j = 0; j < 8; j++) v[j] = p[j];
        } else {
            const float *p;
            if (s.mode == 2) p = reinterpret_cast<const float *>(s.src) + s.x_off + (long)(s.order[r] / s.Kslots) * s.x_row + k0;
            else p = reinterpret_cast<const float *>(s.src) + r * s.lda + k0;
            for (int j = 0; j < 8; j++) v[j] = h_sat(p[j]);
        }
        *reinterpret_cast<sycl::vec<sycl::half, 8> *>(s.dst + r * (long)s.K + k0) = v;
    }
};
/* the s8 copy: a work-group a row, the row's absmax / 127 its scale (1 for a zero row), the
 * codes rounded to nearest and held to +-127; rows past M zero with scale 0 */
struct AQuantK {
    int8_t *dst; float *scale; const void *src; long lda; int a_f16, K, M;
    void operator()(sycl::nd_item<1> it) const {
        const int r = (int)it.get_group(0), t = (int)it.get_local_id(0), nt = (int)it.get_local_range(0);
        int8_t *d = dst + (long)r * K;
        if (r >= M) {
            for (int k = t; k < K; k += nt) d[k] = 0;
            if (t == 0) scale[r] = 0.f;
            return;
        }
        auto at = [&](int k) -> float {
            return a_f16 ? (float)reinterpret_cast<const sycl::half *>(src)[(long)r * lda + k]
                         : reinterpret_cast<const float *>(src)[(long)r * lda + k];
        };
        float m = 0.f;
        for (int k = t; k < K; k += nt) m = sycl::fmax(m, sycl::fabs(at(k)));
        m = sycl::reduce_over_group(it.get_group(), m, sycl::maximum<float>());
        const float sc = m > 0.f ? m / 127.f : 1.f;
        if (t == 0) scale[r] = sc;
        for (int k = t; k < K; k += nt) {
            float q = sycl::rint(at(k) / sc);
            q = sycl::fmin(sycl::fmax(q, -127.f), 127.f);
            d[k] = (int8_t)(int)q;
        }
    }
};

/* ---- the scratch ------------------------------------------------------------------------ */
/* The staging copy of the activations. On an open device's compute queue (the backend's and
 * the chain's calls) it is the device's arena, slot 3 (coli_sycl_scratch: grown with headroom,
 * the old block retired until the chain sees the frame end, freed by coli_sycl_close), so
 * nothing of this unit waits inside a frame or holds a device at exit. A foreign queue (the
 * standalone test's) keeps a private block keyed by queue, grown after a wait on that queue (it
 * has no frames), in a list on the heap that is never destroyed: destroying queue, device and
 * context copies during static destruction, after the runtime began its own shutdown, is a known
 * way to hang at exit (the per-device state of coli_sycl_profile.cpp is kept the same way). */
struct XmxScratch { sycl::queue q; void *p; size_t bytes; };
std::vector<XmxScratch> *g_scratch = new std::vector<XmxScratch>;
std::mutex g_scratch_mu;
#define XMX_SCRATCH_SLOT 3

void *xmx_scratch(sycl::queue &q, size_t bytes) {
    const int dev = coli_sycl_queue_device(q);
    if (dev >= 0 && coli_sycl_queue(dev) == q) return coli_sycl_scratch(dev, bytes, XMX_SCRATCH_SLOT);
    std::lock_guard<std::mutex> lk(g_scratch_mu);
    for (XmxScratch &s : *g_scratch) {
        if (!(s.q == q)) continue;
        if (s.bytes >= bytes) return s.p;
        q.wait();
        sycl::free(s.p, q.get_context());
        s.p = sycl::malloc_device(bytes, q.get_device(), q.get_context());
        s.bytes = s.p ? bytes : 0;
        return s.p;
    }
    void *p = sycl::malloc_device(bytes, q.get_device(), q.get_context());
    if (!p) return nullptr;
    g_scratch->push_back(XmxScratch{q, p, bytes});
    return p;
}

/* ---- the host side ---------------------------------------------------------------------- */
struct XmxPlan { int tm; bool xmx, s8; };

int env_int(const char *name, int dflt) {
    const char *e = getenv(name);
    return e && *e ? atoi(e) : dflt;
}
/* the shape the profile takes: the matrix engines with the f16 8x16x16 shape, 16x16x16 where
 * asked for (COLI_SYCL_XMX_TM=16) and listed, s8 where the caller asks and the format is a
 * row-scaled one; else the plain tile */
XmxPlan xmx_plan(const coli_sycl_profile &p, int fmt, int a_s8) {
    XmxPlan pl = {8, false, false};
    if (p.has_xmx && (p.xmx_shapes & COLI_SYCL_XMX_F16)) pl.xmx = true;
    if (pl.xmx && (p.xmx_shapes & COLI_SYCL_XMX_F16_16)) {
        static int tm = -1;
        if (tm < 0) { tm = env_int("COLI_SYCL_XMX_TM", 8); if (tm != 16) tm = 8; }
        pl.tm = tm;
    }
    if (pl.xmx && a_s8 && (fmt == 1 || fmt == 2) && (p.xmx_shapes & COLI_SYCL_XMX_INT8)) pl.s8 = true;
    return pl;
}
/* Isolate the grouped experts' tile from the dense trunk. Unset keeps the
 * existing global policy; explicit 16 needs the listed compound f16 shape. */
XmxPlan xmx_moe_plan(const coli_sycl_profile &p, int fmt, int phase = 0) {
    XmxPlan pl = xmx_plan(p, fmt, 0);
    static const int tm = env_int("COLI_SYCL_XMX_MOE_TM", 0);
    static const int dn_tm = env_int("COLI_SYCL_XMX_MOE_DN_TM", 0);
    if (pl.xmx && tm == 8) pl.tm = 8;
    if (pl.xmx && tm == 16 && (p.xmx_shapes & COLI_SYCL_XMX_F16_16)) pl.tm = 16;
    if (pl.xmx && phase == 1 && dn_tm == 8) pl.tm = 8;
    if (pl.xmx && phase == 1 && dn_tm == 16 && (p.xmx_shapes & COLI_SYCL_XMX_F16_16)) pl.tm = 16;
    return pl;
}
bool shape_ok(const coli_sycl_profile &p, int fmt, int gs, int K, int N, size_t slm_need) {
    if (fmt != 1 && fmt != 2 && fmt != 4) return false;
    if (fmt == 4 && (gs < 16 || gs % 16)) return false;   /* a work-item's 16 inputs in one group */
    if (K < BK || K % BK || N < 1) return false;
    if (p.sg_count && !coli_sycl_sg_offered(p, SG)) return false;
    if (p.slm_bytes && slm_need > p.slm_bytes) return false;
    return true;
}
size_t slm_need_for(const XmxPlan &pl, int nw) {   /* the largest the plan may take */
    size_t b = (size_t)2 * nw * (BK / 2) * LDB_H * sizeof(sycl::half);   /* the packed f16 tiles (the s8 ones are the same bytes) */
    if (!pl.xmx) b += (size_t)2 * BM * LDA_H * sizeof(sycl::half);
    return b;
}

template <int WF, int NW, int TM, typename TA, bool XMX>
void submit_k(sycl::queue &q, const XmxArgs &a, int tiles, int ntiles_n) {
    using K = XmxGemmK<WF, NW, TM, TA, XMX>;
    q.submit([&](sycl::handler &h) {
        sycl::local_accessor<TA, 1> bs(sycl::range<1>(K::bs_elems()), h);
        sycl::local_accessor<sycl::half, 1> as(sycl::range<1>(K::as_elems()), h);
        h.parallel_for(sycl::nd_range<2>(sycl::range<2>(tiles, (size_t)ntiles_n * WG), sycl::range<2>(1, WG)), K{a, bs, as});
    });
}
template <int WF, int NW>
void submit_plan(sycl::queue &q, const XmxArgs &a, const XmxPlan &pl, int tiles, int ntiles_n) {
    if (!pl.xmx) submit_k<WF, NW, 8, sycl::half, false>(q, a, tiles, ntiles_n);
    else if (pl.s8 && WF != 4) submit_k<WF, NW, 8, int8_t, true>(q, a, tiles, ntiles_n);
    else if (pl.tm == 16) submit_k<WF, NW, 16, sycl::half, true>(q, a, tiles, ntiles_n);
    else submit_k<WF, NW, 8, sycl::half, true>(q, a, tiles, ntiles_n);
}
template <int NW>
void submit_fmt(sycl::queue &q, const XmxArgs &a, const XmxPlan &pl, int fmt, int tiles, int ntiles_n) {
    if (fmt == 1) submit_plan<1, NW>(q, a, pl, tiles, ntiles_n);
    else if (fmt == 2) submit_plan<2, NW>(q, a, pl, tiles, ntiles_n);
    else submit_plan<4, NW>(q, a, pl, tiles, ntiles_n);
}
/* the activations' copy for `rows` rows (a multiple of BM) of K: staged, the pointers into a */
bool stage_a(sycl::queue &q, XmxArgs &a, const XmxPlan &pl, const StageArgs &st_in, const void *src, long lda, int a_f16, int M) {
    const size_t rows = (size_t)st_in.rows, K = (size_t)a.K;
    if (pl.s8) {
        const size_t codes = rows * K, bytes = codes + rows * sizeof(float);
        int8_t *d = reinterpret_cast<int8_t *>(xmx_scratch(q, bytes));
        if (!d) return false;
        float *sc = reinterpret_cast<float *>(d + codes);
        q.parallel_for(sycl::nd_range<1>(rows * 256, 256), AQuantK{d, sc, src, lda, a_f16, a.K, M});
        a.A = d; a.lda = a.K; a.a_scale = sc;
    } else {
        sycl::half *d = reinterpret_cast<sycl::half *>(xmx_scratch(q, rows * K * sizeof(sycl::half)));
        if (!d) return false;
        StageArgs st = st_in; st.dst = d;
        const size_t n = rows * (K / 8);
        q.parallel_for(sycl::nd_range<1>((n + 255) / 256 * 256, 256), AStageK{st});
        a.A = d; a.lda = a.K; a.a_scale = nullptr;
    }
    return true;
}
void xmx_log_fail(const char *what, const sycl::exception &e) {
    fprintf(stderr, "[SYCL] xmx %s: %s\n", what, e.what());
}
/* the dense GEMM (NW 1) and the GLU pair (NW 2) */
int dense_run(const coli_sycl_profile &p, sycl::queue &q, const coli_sycl_gemm &g, const void *Wu, const float *su, int nw) {
    if (!g.C || !g.A || !g.W || !g.scales || g.M < 1 || (nw == 2 && (!Wu || !su))) return 0;
    if (g.ldc < g.N || g.lda < g.K) return 0;
    XmxPlan pl = xmx_plan(p, g.fmt, g.a_s8);
    if (!shape_ok(p, g.fmt, g.gs, g.K, g.N, slm_need_for(pl, nw))) return 0;
    XmxArgs a; memset(&a, 0, sizeof a);
    a.Y = g.C; a.ldc = g.ldc; a.K = g.K; a.N = g.N; a.M = g.M; a.gs = g.gs; a.ng = g.fmt == 4 ? (g.K + g.gs - 1) / g.gs : 1;
    a.W[0] = (const uint8_t *)g.W; a.sc[0] = g.scales; a.W[1] = (const uint8_t *)Wu; a.sc[1] = su;
    const int tiles = (g.M + BM - 1) / BM, ntn = (g.N + BN - 1) / BN;
    StageArgs st; memset(&st, 0, sizeof st);
    st.K = g.K; st.rows = tiles * BM; st.mode = g.a_f16 ? 1 : 0; st.src = g.A; st.lda = g.lda; st.M = g.M;
    try {
        if (!stage_a(q, a, pl, st, g.A, g.lda, g.a_f16, g.M)) return 0;
        if (nw == 2) submit_fmt<2>(q, a, pl, g.fmt, tiles, ntn);
        else submit_fmt<1>(q, a, pl, g.fmt, tiles, ntn);
    } catch (const sycl::exception &e) { xmx_log_fail(nw == 2 ? "gemm glu" : "gemm", e); return 0; }
    return 1;
}
/* the moe's shape checks shared by _ok and the call */
bool moe_shape_ok(const coli_sycl_profile &p, const CcMoe &m, const XmxPlan &pl) {
    if (m.S < 1 || m.K < 1 || m.E < 1 || m.E > 4096 || m.D < BK || m.I < BK || m.D % BK || m.I % BK) return false;
    if (m.planar && !(m.fmt == 4 && m.gs == 64)) return false;   /* planar entries: int4 per 64 only (hcg's) */
    if (m.fmt_d && m.fmt_d != m.fmt) return false;               /* a GGUF's types: not this family */
    /* the rows are inside the layout (CcMoe.lay_S / lay_s0; cc_moe refuses the same). A parts 2
     * alone over a slice of the layout's rows (a prompt chunk's lanes: the rows' sums of 0..15
     * and 16..39 apart, each with its own offsets) is as valid as the whole: the sort is the
     * layout's, moe_rows_sum_k reads it at lay_s0 + s. Refusing LS != S here made the backend
     * take moe_sum_k (the unsorted layout) for such a lane: the sums were not the chunk's bits
     * (the chain test's "a chunk's two lanes' sums", found on the first card, MIGRATION.md §13). */
    const int LS = m.lay_S > 0 ? m.lay_S : m.S;
    if (m.lay_S < 0 || m.lay_s0 < 0 || m.lay_s0 > LS - m.S) return false;
    return shape_ok(p, m.fmt, m.gs, m.D, m.I, slm_need_for(pl, 2)) && shape_ok(p, m.fmt, m.gs, m.I, m.D, slm_need_for(pl, 1));
}

} // namespace

/* ---- coli_sycl.h's section 4 ------------------------------------------------------------- */

int coli_sycl_xmx_gemm_ok(const coli_sycl_profile &p, int fmt, int gs, int K, int N) {
    XmxPlan pl = xmx_plan(p, fmt, 0);
    return shape_ok(p, fmt, gs, K, N, slm_need_for(pl, 2)) ? 1 : 0;
}
int coli_sycl_xmx_gemm(const coli_sycl_profile &p, sycl::queue &q, const coli_sycl_gemm &g) {
    return dense_run(p, q, g, nullptr, nullptr, 1);
}
int coli_sycl_xmx_gemm_glu(const coli_sycl_profile &p, sycl::queue &q, const coli_sycl_gemm &gate, const void *Wu, const float *scales_u) {
    return dense_run(p, q, gate, Wu, scales_u, 2);
}

int coli_sycl_xmx_moe_ok(const coli_sycl_profile &p, const CcMoe &m) {
    return moe_shape_ok(p, m, xmx_moe_plan(p, m.fmt)) ? 1 : 0;
}
int coli_sycl_xmx_moe(const coli_sycl_profile &p, sycl::queue &q, const CcExpert *tab, const CcMoe &m,
                      const coli_sycl_moe_sort &sort, const float *x, float *hid, int phase) {
    XmxPlan pl = xmx_moe_plan(p, m.fmt, phase);
    if (!tab || !hid || !sort.order || !sort.base || !sort.te || !sort.tr || !sort.ntile || sort.T < 1 || sort.TR != BM) return 0;
    if ((phase != 0 && phase != 1) || (phase == 0 && !x) || !moe_shape_ok(p, m, pl)) return 0;
    if ((m.lay_S > 0 && m.lay_S != m.S) || m.lay_s0) return 0;   /* the GEMMs run over the layout's rows: a parts 1 is never a lane (cc_moe) */
    static bool said[2] = {};
    if (!said[phase]) {
        said[phase] = true;
        fprintf(stderr, "[SYCL] grouped MoE %s tile: TM%d, %s activations\n",
                phase ? "down" : "gate/up", pl.tm, pl.xmx ? "f16 XMX" : "f16 portable");
    }
    const long A_total = (long)m.S * m.K;
    const int rows = (int)((A_total + BM + BM - 1) / BM * BM);   /* every tile's 64 rows inside the copy */
    XmxArgs a; memset(&a, 0, sizeof a);
    a.moe = 1; a.phase = phase; a.t_off = m.t_off; a.tab = tab;
    a.base = sort.base; a.te = sort.te; a.tr = sort.tr; a.ntile = sort.ntile;
    a.gs = m.gs;
    if (phase == 0) { a.K = m.D; a.N = m.I; a.Y = hid; a.ldc = m.I; }
    else { a.K = m.I; a.N = m.D; a.Y = hid + A_total * m.I; a.ldc = m.D; }
    a.ng = m.fmt == 4 ? (a.K + m.gs - 1) / m.gs : 1;
    const int ntn = (a.N + BN - 1) / BN;
    StageArgs st; memset(&st, 0, sizeof st);
    st.K = a.K; st.rows = rows; st.nplaced = sort.base + m.E; st.order = sort.order; st.Kslots = m.K;
    if (phase == 0) { st.mode = 2; st.src = x; st.x_off = m.x_off; st.x_row = m.x_row; }
    else { st.mode = 3; st.src = hid; st.lda = m.I; }
    try {
        if (!stage_a(q, a, pl, st, st.src, st.lda, 0, 0)) return 0;
        if (phase == 0) submit_fmt<2>(q, a, pl, m.fmt, sort.T, ntn);
        else submit_fmt<1>(q, a, pl, m.fmt, sort.T, ntn);
    } catch (const sycl::exception &e) { xmx_log_fail(phase == 0 ? "moe gate/up" : "moe down", e); return 0; }
    return 1;
}
