/* Opt-in prompt attention. One workgroup owns eight query rows of one head.
 * QK and PV use the profile's f16 8x16x16 matrices; online softmax keeps f32
 * maxima, denominators and output accumulators. Every key block has the same
 * four workgroup barriers, including padded query rows. No device allocation,
 * atomics, inter-workgroup synchronization or host signaling. */
#include "coli_sycl.h"
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <atomic>

namespace jm = sycl::ext::oneapi::experimental::matrix;
namespace jmi = sycl::ext::intel::experimental::matrix;
namespace {
constexpr int QM = 8, WG = 128, SG = 16;
int key_tile() {
    const char *e = std::getenv("COLI_SYCL_XMX_ATTN_K");
    return e && !std::strcmp(e, "64") ? 64 : 32;
}
size_t slm_size(int hd, int kn) { return (QM + 2 * kn) * hd * sizeof(sycl::half) +
    QM * kn * (sizeof(float) + sizeof(sycl::half)) + 3 * QM * sizeof(float); }
sycl::half half_clamp(float x) {
    return sycl::half(sycl::fmin(sycl::fmax(x, -65504.f), 65504.f));
}

template<int HD, int KN, typename KV>
void attn_submit(sycl::queue &queue, coli_sycl_attn a) {
    const size_t tiles = ((size_t)a.p.S + QM - 1) / QM;
    coli_sycl_fprof_ev(queue.submit([&](sycl::handler &cgh) {
        sycl::local_accessor<sycl::half, 1> qs(QM * HD, cgh), ks(HD * KN, cgh),
            vs(KN * HD, cgh), probs(QM * KN, cgh);
        sycl::local_accessor<float, 1> scores(QM * KN, cgh), stats(3 * QM, cgh);
        cgh.parallel_for(sycl::nd_range<1>(tiles * a.p.H * WG, WG),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            const auto p = a.p;
            const int tid = it.get_local_id(0), sid = tid / SG, lane = tid % SG;
            const size_t block = it.get_group(0);
            const int h = block / tiles, row0 = (block % tiles) * QM, kh = h / a.G;
            const int rows = sycl::min(QM, p.S - row0), end = p.pos_base + row0 + rows;
            auto sg = it.get_sub_group();
            auto qp = qs.template get_multi_ptr<sycl::access::decorated::no>();
            auto kp = ks.template get_multi_ptr<sycl::access::decorated::no>();
            auto vp = vs.template get_multi_ptr<sycl::access::decorated::no>();
            auto pp = probs.template get_multi_ptr<sycl::access::decorated::no>();
            using A = jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::a, QM, 16, jm::layout::row_major>;
            using B = jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::b, 16, 16, jm::layout::ext_intel_packed>;
            using C = jm::joint_matrix<sycl::sub_group, float, jm::use::accumulator, QM, 16>;
            // Named accumulators avoid IGC's AccessChain lowering bug for a
            // matrix array carried through the online-softmax loop.
            C out0, out1;
            jm::joint_matrix_fill(sg, out0, 0.f);
            if constexpr (HD > WG) jm::joint_matrix_fill(sg, out1, 0.f);
            for (int i = tid; i < QM * HD; i += WG) {
                const int r = i / HD, d = i % HD;
                qs[i] = r < rows ? half_clamp(a.q[(size_t)p.q_off + (size_t)(row0 + r) * p.q_row + (size_t)h * p.q_seg + d]) : sycl::half(0);
            }
            if (tid < QM) { stats[tid] = -INFINITY; stats[QM + tid] = 0.f; stats[2 * QM + tid] = 0.f; }
            it.barrier(sycl::access::fence_space::local_space);
            for (int64_t key0 = 0; key0 < end; key0 += KN) {
                // K^T: [HD/2][KN*2], V: [KN/2][HD*2], VNNI half pairs.
                for (int i = tid; i < KN * HD; i += WG) {
                    const int t = i / HD, d = i % HD;
                    const size_t off = ((size_t)kh * p.cap + key0 + t) * HD + d;
                    const bool valid = key0 + t < end;
                    ks[(d / 2) * KN * 2 + t * 2 + d % 2] = valid ? half_clamp((float)((const KV *)a.kc)[(size_t)p.k_off + off]) : sycl::half(0);
                    vs[(t / 2) * HD * 2 + d * 2 + t % 2] = valid ? half_clamp((float)((const KV *)a.vc)[(size_t)p.v_off + off]) : sycl::half(0);
                }
                it.barrier(sycl::access::fence_space::local_space);
                if (sid < KN / 16) { // subgroup-uniform matrix operations
                    C dot; jm::joint_matrix_fill(sg, dot, 0.f);
                    for (int d = 0; d < HD; d += 16) {
                        A qa; B kb;
                        jm::joint_matrix_load(sg, qa, qp + d, HD);
                        jm::joint_matrix_load(sg, kb, kp + (d / 2) * KN * 2 + sid * 32, KN * 2);
                        jm::joint_matrix_mad(sg, dot, qa, kb, dot);
                    }
                    jmi::joint_matrix_apply(sg, dot, [&](float &v, size_t r, size_t col) {
                        scores[r * KN + sid * 16 + col] = v * p.scale;
                    });
                }
                it.barrier(sycl::access::fence_space::local_space);
                // One subgroup per query row; each lane handles KN/16 keys.
                // Invalid tail rows never evaluate -inf - -inf.
                float score[KN / SG], exps[KN / SG], mx = -INFINITY;
                const bool live = sid < rows;
                const int last = p.pos_base + row0 + sid;
                #pragma unroll
                for (int j = 0; j < KN / SG; j++) {
                    const int col = lane + j * SG;
                    score[j] = live && key0 + col <= last ? scores[sid * KN + col] : -INFINITY;
                    mx = sycl::fmax(mx, score[j]);
                }
                const float block_max = sycl::reduce_over_group(sg, mx, sycl::maximum<float>());
                const float old_max = stats[sid], old_sum = stats[QM + sid];
                const float next_max = live ? sycl::fmax(old_max, block_max) : 0.f;
                const float alpha = live ? sycl::exp(old_max - next_max) : 0.f;
                float sum = 0.f;
                #pragma unroll
                for (int j = 0; j < KN / SG; j++) {
                    exps[j] = live ? sycl::exp(score[j] - next_max) : 0.f;
                    sum += exps[j];
                    probs[sid * KN + lane + j * SG] = sycl::half(exps[j]);
                }
                const float block_sum = sycl::reduce_over_group(sg, sum, sycl::plus<float>());
                if (lane == 0) {
                    stats[sid] = next_max; stats[QM + sid] = old_sum * alpha + block_sum;
                    stats[2 * QM + sid] = alpha;
                }
                it.barrier(sycl::access::fence_space::local_space);
                auto pv_step = [&](C &acc, int col0) {
                    if (col0 < HD) {
                        jmi::joint_matrix_apply(sg, acc, [&](float &v, size_t r, size_t) { v *= stats[2 * QM + r]; });
                        #pragma unroll
                        for (int k = 0; k < KN; k += 16) {
                            A pa; B vb;
                            jm::joint_matrix_load(sg, pa, pp + k, KN);
                            jm::joint_matrix_load(sg, vb, vp + (k / 2) * HD * 2 + col0 * 2, HD * 2);
                            jm::joint_matrix_mad(sg, acc, pa, vb, acc);
                        }
                    }
                };
                pv_step(out0, sid * 16);
                if constexpr (HD > WG) pv_step(out1, sid * 16 + WG);
                it.barrier(sycl::access::fence_space::local_space);
            }
            auto store_out = [&](C &acc, int col0) {
                if (col0 < HD) jmi::joint_matrix_apply(sg, acc, [&](float &v, size_t r, size_t col) {
                    if (r < (size_t)rows) {
                        const size_t row = row0 + r, d = col0 + col;
                        float val = v / stats[QM + r];
                        if (p.has_gate) val *= 1.f / (1.f + sycl::exp(-a.gate[(size_t)p.g_off + row * p.g_row + (size_t)h * p.g_seg + d]));
                        a.o[(size_t)p.o_off + row * p.o_row + (size_t)h * HD + d] = val;
                    }
                });
            };
            store_out(out0, sid * 16);
            if constexpr (HD > WG) store_out(out1, sid * 16 + WG);
        });
    }));
}
}

int coli_sycl_xmx_attn_ok(const coli_sycl_profile &p, int hd, int G, int kv_f16, int sel) {
    const char *e = std::getenv("COLI_SYCL_XMX_ATTN");
    return e && !std::strcmp(e, "1") && (hd == 64 || hd == 128 || hd == 256) && p.has_xmx && (p.xmx_shapes & COLI_SYCL_XMX_F16) &&
        coli_sycl_sg_offered(p, SG) && p.max_wg_size >= WG && p.slm_bytes >= slm_size(hd, key_tile()) &&
        G > 0 && (kv_f16 == 0 || kv_f16 == 1) && !sel;
}

int coli_sycl_xmx_attn(const coli_sycl_profile &p, sycl::queue &q, const coli_sycl_attn &a) {
    const auto &v = a.p;
    if (!coli_sycl_xmx_attn_ok(p, v.hd, a.G, a.kv_f16, a.sel != nullptr || v.sel_row > 0) ||
        a.win || a.snk || a.ring || !a.q || !a.kc || !a.vc || !a.o || a.q == a.o || a.gate == a.o ||
        v.S < 1 || v.H < 1 || v.KVH < 1 || v.H % v.KVH || v.H / v.KVH != a.G ||
        v.pos_base < 0 || (int64_t)v.pos_base + v.S > v.cap || v.q_off < 0 || v.o_off < 0 ||
        v.k_off < 0 || v.v_off < 0 || v.q_seg < v.hd || (int64_t)v.q_row < (int64_t)(v.H - 1) * v.q_seg + v.hd ||
        (int64_t)v.o_row < (int64_t)v.H * v.hd || !std::isfinite(v.scale) ||
        (v.has_gate && (!a.gate || v.g_off < 0 || v.g_seg < v.hd || (int64_t)v.g_row < (int64_t)(v.H - 1) * v.g_seg + v.hd))) return 0;
    // Exceptions propagate to the chain's cc_try; never submit a fallback after
    // a device error. No unsupported case reaches the queue.
    static std::atomic<bool> said{false}; // host only
    const int kn = key_tile();
    if (!said.exchange(true)) std::fprintf(stderr, "[SYCL] prompt attention: XMX online 8x%d, SG16, f16 Q/K/V/P\n", kn);
#define RUN_K(H, K) if (a.kv_f16) attn_submit<H, K, sycl::half>(q, a); else attn_submit<H, K, float>(q, a);
#define RUN_HD(H) if (v.hd == H) { if (kn == 64) { RUN_K(H, 64) } else { RUN_K(H, 32) } }
    RUN_HD(64) else RUN_HD(128) else RUN_HD(256)
#undef RUN_HD
#undef RUN_K
    return 1;
}
