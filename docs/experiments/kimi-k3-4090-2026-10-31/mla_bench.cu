// mla_bench.cu -- the MLA core's kernels one by one at a 5,862-token prompt's calls (1,024 rows each)
// built against 04623a27's kernels: git show 04623a27:c/backend_cuda_k3.cu > backend_cuda_k3_04623a27.cu
#include "backend_cuda_k3_04623a27.cu"
#include <cstdio>
#include <cstdlib>
#include <vector>

static uint32_t rng = 1;
static float rndf() { rng = rng * 1664525u + 1013904223u; return ((float)(rng >> 8) / 16777216.0f - 0.5f) * 4.0f; }

int main(int argc, char **argv) {
    const int H = 96, qn = 128, qr = 64, vh = 128, kvl = 512, P = argc > 1 ? atoi(argv[1]) : 5862, CALL = 1024;
    const int qh = qn + qr, KD = kvl + qr, O = H * (qn + vh);
    std::vector<float> Lc((size_t)P * kvl), Rc((size_t)P * qr), qv((size_t)CALL * H * qh), gv((size_t)CALL * H * vh);
    std::vector<int8_t> kvb((size_t)O * kvl); std::vector<float> kvbs(O);
    for (auto &v : Lc) v = rndf(); for (auto &v : Rc) v = rndf(); for (auto &v : qv) v = rndf(); for (auto &v : gv) v = rndf();
    for (auto &v : kvb) v = (int8_t)((int)(rndf() * 60)); for (auto &v : kvbs) v = 0.004f;
    float *LR, *dqv, *dgv, *ds, *Q, *S, *cl, *ctx; int8_t *dw;
    cudaMalloc(&LR, (size_t)P * KD * 4); cudaMalloc(&dqv, qv.size() * 4); cudaMalloc(&dgv, gv.size() * 4);
    cudaMalloc(&dw, kvb.size()); cudaMalloc(&ds, O * 4); cudaMalloc(&ctx, (size_t)CALL * H * vh * 4);
    const int TQmax = (int)((128ll << 20) / ((long long)H * CALL * 4));
    cudaMalloc(&Q, (size_t)CALL * H * KD * 4); cudaMalloc(&S, (128ll << 20) + (size_t)H * P * 4 * 2); cudaMalloc(&cl, (size_t)CALL * H * kvl * 4);
    cudaMemcpy2D(LR, (size_t)KD * 4, Lc.data(), (size_t)kvl * 4, (size_t)kvl * 4, P, cudaMemcpyHostToDevice);
    cudaMemcpy2D(LR + kvl, (size_t)KD * 4, Rc.data(), (size_t)qr * 4, (size_t)qr * 4, P, cudaMemcpyHostToDevice);
    cudaMemcpy(dqv, qv.data(), qv.size() * 4, cudaMemcpyHostToDevice); cudaMemcpy(dgv, gv.data(), gv.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dw, kvb.data(), kvb.size(), cudaMemcpyHostToDevice); cudaMemcpy(ds, kvbs.data(), O * 4, cudaMemcpyHostToDevice);
    (void)TQmax;
    cudaEvent_t ev[6]; for (auto &e : ev) cudaEventCreate(&e);
    double tk[5] = {0, 0, 0, 0, 0};
    const char *nm[5] = {"k3_mla_q", "scores gemm", "softmax", "clat gemm", "k3_mla_out"};
    double mac_s = 0, mac_c = 0;
    for (int c0 = 0; c0 < P; c0 += CALL) {
        const int n = P - c0 < CALL ? P - c0 : CALL, N = c0 + n, pos0 = c0;
        long long tq = (128ll << 20) / ((long long)H * N * 4);
        const int TQ = tq < 1 ? 1 : tq > n ? n : (int)tq;
        for (int r0 = 0; r0 < n; r0 += TQ) {
            const int rows = n - r0 < TQ ? n - r0 : TQ, M = rows * H;
            cudaEventRecord(ev[0]);
            k3_mla_q<<<M, 256>>>(dqv + (size_t)r0 * H * qh, H, qh, qn, qr, vh, kvl, dw, ds, Q);
            cudaEventRecord(ev[1]);
            dim3 gs((unsigned)((N + K3G_T - 1) / K3G_T), (unsigned)((M + K3G_T - 1) / K3G_T));
            k3_gemm<0, 1><<<gs, 256>>>(M, N, KD, Q, KD, LR, KD, S, N, 0.07f);
            cudaEventRecord(ev[2]);
            k3_mla_softmax<<<M, 256>>>(S, N, H, pos0 + r0);
            cudaEventRecord(ev[3]);
            dim3 gp((unsigned)((kvl + K3G_T - 1) / K3G_T), (unsigned)((M + K3G_T - 1) / K3G_T));
            k3_gemm<1, 0><<<gp, 256>>>(M, kvl, N, S, N, LR, KD, cl, kvl, 0.0f);
            cudaEventRecord(ev[4]);
            k3_mla_out<<<M, 128>>>(cl, dgv + (size_t)r0 * H * vh, H, qn, vh, kvl, dw, ds, ctx + (size_t)r0 * H * vh);
            cudaEventRecord(ev[5]);
            cudaEventSynchronize(ev[5]);
            for (int k = 0; k < 5; k++) { float ms; cudaEventElapsedTime(&ms, ev[k], ev[k + 1]); tk[k] += ms; }
            mac_s += (double)M * N * KD; mac_c += (double)M * N * kvl;
        }
    }
    double tot = 0; for (int k = 0; k < 5; k++) tot += tk[k];
    printf("a layer of a %d-token prompt (calls of %d rows): %.1f ms; x24 layers %.2f s\n", P, CALL, tot, tot * 24 / 1000);
    for (int k = 0; k < 5; k++) printf("  %-12s %8.1f ms  (%4.1f%%)\n", nm[k], tk[k], 100 * tk[k] / tot);
    printf("  scores %.3g MAC: %.2f TMAC/s; clat %.3g MAC: %.2f TMAC/s\n", mac_s, mac_s / tk[1] / 1e9, mac_c, mac_c / tk[3] / 1e9);
    printf("%s\n", cudaGetErrorString(cudaGetLastError()));
    return 0;
}
