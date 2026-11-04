/* Qwen3.8's prompt experts on a CUDA device (cuda_q38_moe.cu, the chain's q38_moe and q38_sum) against
 * the CPU's arithmetic, compared bit for bit: qwen38_core.h's q38_fp8_rows for gate, up and down (per
 * 128-column block eight fused lanes, lane l the block's columns l, l+8, .., l+120 from +0, hsum256's
 * tree, then fma(part, scale, total) in double, rounded to float at the end), h = (g sigmoid(g)) u with
 * exp_f's bits (k3_exact.h's host code and its table, the device's own arithmetic checked there on every
 * float), and every row's experts fused in rank order from +0 (a slot without a place adds nothing).
 * Shapes: a small one and Qwen3.8-Flash-Next's (D 2560, I 640); experts of 1, 31, 32, 33 and 50+ rows
 * (partial tiles); two launches, the second's hidden rows from its first place on (hp0). NaN-free
 * random weights, block scales of a checkpoint's size.
 * Needs a device: make cuda-q38-moe-check CUDA=1 (Windows: nvcc with this file, cuda_q38_moe.cu).
 * CUDA only, as the kernels are: an HIP build of it says so and passes. */
#if defined(__HIP__) || defined(__HIP_PLATFORM_AMD__)
#include <cstdio>
int main() { std::printf("cuda_q38_moe.cu is CUDA only: skipped\n"); return 0; }
#else
#define COLI_CUDA_CHAIN_NO_WRAPPERS
#include "../cuda_chain.h"
#include "../k3_exact.h"
#include "../k3_exact_tables.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

extern "C" int coli_cuda_q38_moe(const float *x, const int *rows, const CcQ38Exp *tab, const CcQ38Moe *p,
                                 float *h, float *out, void *stream);
extern "C" int coli_cuda_q38_sum(const float *out, const int *pos, const float *gates, int S, int K, int D, float *y,
                                 void *stream);

static float dec(uint8_t b) {   /* quant.h's e4m3_decode8: the seven low bits at 20, the sign at 31, times 2^120 */
    const uint32_t lo = b & 0x7fu;
    if (lo == 0x7fu) return std::nanf("");
    uint32_t u = ((uint32_t)(b & 0x80u) << 24) | (lo << 20);
    float f;
    std::memcpy(&f, &u, 4);
    return f * 0x1p120f;
}
static float ref_expf(float x) {
    float f;
    if (k3x_expf_fast(x, &f)) return f;
    return k3x_table(k3x_EXP_in, k3x_EXP_out, K3X_EXP_N, x);
}
static float ref_sigmoid(float x) {   /* q38_sigmoid */
    if (x >= 0.f) { float z = ref_expf(-x); return 1.f / (1.f + z); }
    float z = ref_expf(x);
    return z / (1.f + z);
}
/* output o of an O x K e4m3 matrix (scales [ceil(O/128)][K/128]) for one input row: q38_fp8_rows */
static float ref_row(const uint8_t *w, const float *sc, int K, int o, const float *x) {
    const int nb = K / 128;
    double a = 0.0;
    for (int bi = 0; bi < nb; bi++) {
        float l[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (int i = 0; i < 128; i += 8)
            for (int k = 0; k < 8; k++) l[k] = std::fmaf(dec(w[(size_t)o * K + bi * 128 + i + k]), x[bi * 128 + i + k], l[k]);
        volatile float s04 = l[0] + l[4], s26 = l[2] + l[6], s15 = l[1] + l[5], s37 = l[3] + l[7];
        volatile float p0 = s04 + s26, p1 = s15 + s37;
        const float part = p0 + p1;
        a = std::fma((double)part, (double)sc[(o / 128) * nb + bi], a);
    }
    return (float)a;
}

static int fails;
static void check(int ok, const char *what) {
    std::printf("  %s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) fails++;
}

/* S rows, K slots each, over NE experts of D x I; the rows' routes skewed so that the experts get
 * 1 .. many rows; one slot of row 3 has no place (-1) */
static void run(int D, int I, int S, int K, int NE, unsigned seed) {
    std::printf("D %d, I %d: %d rows, %d slots each, %d experts\n", D, I, S, K, NE);
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_int_distribution<int> byte(0, 255);
    const size_t MB = (size_t)D * I;
    std::vector<uint8_t> W((size_t)NE * 3 * MB);
    for (auto &b : W) { int v; do v = byte(rng); while ((v & 0x7f) == 0x7f); b = (uint8_t)v; }
    const int SCN = (I / 128) * (D / 128);   /* a matrix's block scales: gate, up [I/128][D/128], down [D/128][I/128] */
    std::vector<float> SC((size_t)NE * 3 * SCN);
    for (auto &s : SC) s = std::ldexp(1.f + 0.5f * std::fabs(nd(rng)), -12 - (int)(rng() % 4));
    std::vector<float> X((size_t)S * D);
    for (auto &v : X) v = nd(rng);
    std::vector<int> idx((size_t)S * K);
    std::vector<float> gates((size_t)S * K);
    for (int s = 0; s < S; s++)
        for (int z = 0; z < K; z++) {
            int e = z == 0 ? (s < 1 ? NE - 1 : s % 3) : z == 1 ? NE - 2 : (s * 7 + 1) % 3;
            if (z == 2 && e == idx[(size_t)s * K]) e = (e + 1) % 3;
            if (z == 1 && s >= 31 && NE > 5) e = 3;   /* expert 3: rows 31 .. S-1 */
            idx[(size_t)s * K + z] = e;
            gates[(size_t)s * K + z] = 0.1f + 0.3f * std::fabs(nd(rng));
        }
    if (S > 3) idx[(size_t)3 * K + K - 1] = -1;   /* a slot no expert fills */
    std::vector<int> cnt(NE, 0), start(NE + 1, 0), cur(NE), place((size_t)S * K, -1), rowof((size_t)S * K);
    for (int j = 0; j < S * K; j++) if (idx[j] >= 0) cnt[idx[j]]++;
    for (int e = 0; e < NE; e++) start[e + 1] = start[e] + cnt[e];
    for (int e = 0; e < NE; e++) cur[e] = start[e];
    for (int j = 0; j < S * K; j++) if (idx[j] >= 0) { int p = cur[idx[j]]++; place[j] = p; rowof[p] = j / K; }
    const int NP = start[NE];
    std::printf("  rows an expert:");
    for (int e = 0; e < NE; e++) std::printf(" %d", cnt[e]);
    std::printf("\n");
    uint8_t *dW; float *dSC, *dX, *dH, *dH2, *dO, *dY, *dG; int *dR, *dP; CcQ38Exp *dT;
    cudaMalloc(&dW, W.size()); cudaMemcpy(dW, W.data(), W.size(), cudaMemcpyHostToDevice);
    cudaMalloc(&dSC, SC.size() * 4); cudaMemcpy(dSC, SC.data(), SC.size() * 4, cudaMemcpyHostToDevice);
    cudaMalloc(&dX, X.size() * 4); cudaMemcpy(dX, X.data(), X.size() * 4, cudaMemcpyHostToDevice);
    cudaMalloc(&dH, (size_t)NP * I * 4); cudaMalloc(&dH2, (size_t)NP * I * 4);
    cudaMalloc(&dO, (size_t)NP * D * 4); cudaMalloc(&dY, (size_t)S * D * 4);
    cudaMalloc(&dR, (size_t)NP * 4); cudaMemcpy(dR, rowof.data(), (size_t)NP * 4, cudaMemcpyHostToDevice);
    cudaMalloc(&dP, (size_t)S * K * 4); cudaMemcpy(dP, place.data(), (size_t)S * K * 4, cudaMemcpyHostToDevice);
    cudaMalloc(&dG, (size_t)S * K * 4); cudaMemcpy(dG, gates.data(), (size_t)S * K * 4, cudaMemcpyHostToDevice);
    cudaMalloc(&dT, (size_t)NE * sizeof(CcQ38Exp));
    /* two launches: experts 0 .. 2 (h from place 0), then the rest (their own h from their first place) */
    const int split = NE < 3 ? NE : 3;
    int ok = 1;
    for (int part = 0; part < 2; part++) {
        std::vector<CcQ38Exp> tab;
        int t = 0;
        for (int e = part ? split : 0; e < (part ? NE : split); e++) {
            if (!cnt[e]) continue;
            CcQ38Exp x{};
            const uint8_t *b = dW + (size_t)e * 3 * MB;
            x.g = b; x.u = b + MB; x.d = b + 2 * MB;
            x.gs = dSC + (size_t)e * 3 * SCN; x.us = x.gs + SCN; x.ds = x.gs + 2 * SCN;
            x.p0 = start[e]; x.n = cnt[e]; x.t0 = t;
            t += (cnt[e] + CC_Q38_TILE - 1) / CC_Q38_TILE;
            tab.push_back(x);
        }
        if (tab.empty()) continue;
        cudaMemcpy(dT, tab.data(), tab.size() * sizeof(CcQ38Exp), cudaMemcpyHostToDevice);
        const int hp0 = part ? start[split] : 0;
        CcQ38Moe p{(int)tab.size(), t, D, I, hp0, 0};
        ok &= coli_cuda_q38_moe(dX, dR, dT, &p, part ? dH2 : dH, dO, nullptr);
        ok &= cudaDeviceSynchronize() == cudaSuccess;
    }
    ok &= coli_cuda_q38_sum(dO, dP, dG, S, K, D, dY, nullptr);
    ok &= cudaDeviceSynchronize() == cudaSuccess;
    check(ok, "the launches ran");
    std::vector<float> H((size_t)NP * I), O((size_t)NP * D), Y((size_t)S * D);
    cudaMemcpy(H.data(), dH, (size_t)start[split] * I * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(H.data() + (size_t)start[split] * I, dH2, (size_t)(NP - start[split]) * I * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(O.data(), dO, O.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(Y.data(), dY, Y.size() * 4, cudaMemcpyDeviceToHost);
    std::vector<float> rh((size_t)NP * I), ro((size_t)NP * D);
    for (int j = 0; j < S * K; j++) {
        if (idx[j] < 0) continue;
        const int e = idx[j], p = place[j], s = j / K;
        const uint8_t *b = W.data() + (size_t)e * 3 * MB;
        const float *sc = SC.data() + (size_t)e * 3 * SCN;
        for (int o = 0; o < I; o++) {
            const float g = ref_row(b, sc, D, o, X.data() + (size_t)s * D), u = ref_row(b + MB, sc + SCN, D, o, X.data() + (size_t)s * D);
            volatile float sg = g * ref_sigmoid(g);
            rh[(size_t)p * I + o] = sg * u;
        }
        for (int o = 0; o < D; o++) ro[(size_t)p * D + o] = ref_row(b + 2 * MB, sc + 2 * SCN, I, o, rh.data() + (size_t)p * I);
    }
    long bh = 0, bo = 0, by = 0;
    for (size_t k = 0; k < H.size(); k++) bh += std::memcmp(&H[k], &rh[k], 4) != 0;
    for (size_t k = 0; k < O.size(); k++) bo += std::memcmp(&O[k], &ro[k], 4) != 0;
    for (int s = 0; s < S; s++)
        for (int d = 0; d < D; d++) {
            float acc = 0.f;
            for (int z = 0; z < K; z++) {
                const int p = place[(size_t)s * K + z];
                if (p >= 0) acc = std::fmaf(gates[(size_t)s * K + z], ro[(size_t)p * D + d], acc);
            }
            by += std::memcmp(&acc, &Y[(size_t)s * D + d], 4) != 0;
        }
    char msg[160];
    std::snprintf(msg, sizeof msg, "silu(gate) * up the CPU's bits (%ld of %zu differ)", bh, H.size()); check(!bh, msg);
    std::snprintf(msg, sizeof msg, "down the CPU's bits (%ld of %zu differ)", bo, O.size()); check(!bo, msg);
    std::snprintf(msg, sizeof msg, "the rows' rank-order sums the CPU's bits (%ld of %zu differ)", by, Y.size()); check(!by, msg);
    cudaFree(dW); cudaFree(dSC); cudaFree(dX); cudaFree(dH); cudaFree(dH2); cudaFree(dO); cudaFree(dY);
    cudaFree(dR); cudaFree(dP); cudaFree(dG); cudaFree(dT);
}

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n < 1) { std::printf("no CUDA device: skipped\n"); return 0; }
    run(256, 128, 40, 3, 4, 7u);
    run(2560, 640, 50, 3, 6, 12345u);   /* Qwen3.8-Flash-Next */
    if (fails) { std::printf("%d check(s) failed\n", fails); return 1; }
    std::printf("all checks passed\n");
    return 0;
}
#endif
