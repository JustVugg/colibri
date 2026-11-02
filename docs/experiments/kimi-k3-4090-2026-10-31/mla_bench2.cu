// mla_bench2.cu -- the new MLA core kernels one by one at a prompt's calls (1,024 rows each), as k3cu_mla runs them
#include "../../../c/backend_cuda_k3.cu"
#include <cstdio>
#include <cstdlib>
#include <vector>

static uint32_t rng = 1;
static float rndf() { rng = rng * 1664525u + 1013904223u; return ((float)(rng >> 8) / 16777216.0f - 0.5f) * 4.0f; }

int main(int argc, char **argv) {
    const int H = 96, qn = 128, qr = 64, vh = 128, kvl = 512, P = argc > 1 ? atoi(argv[1]) : 5862, CALL = 1024;
    const long long cap = (argc > 2 ? atoll(argv[2]) : 128) << 20;
    const int quant = argc > 3 ? atoi(argv[3]) : 0;
    const int qh = qn + qr, KD = kvl + qr, O = H * (qn + vh);
    if (!k3cu_init(0)) { printf("init failed\n"); return 1; }
    std::vector<float> Lc((size_t)P * kvl), Rc((size_t)P * qr), qv((size_t)CALL * H * qh), gv((size_t)CALL * H * vh);
    std::vector<int8_t> kvb((size_t)O * kvl); std::vector<float> kvbs(O);
    for (auto &v : Lc) v = rndf(); for (auto &v : Rc) v = rndf(); for (auto &v : qv) v = rndf() * 0.3f; for (auto &v : gv) v = rndf();
    for (auto &v : kvb) v = (int8_t)((int)(rndf() * 60)); for (auto &v : kvbs) v = 0.004f;
    const int ldSm = (P + 7) / 8 * 8, ldpm = (P + MG_T - 1) / MG_T;
    float *LR, *dqv, *dgv, *ds, *Q, *S, *pm, *sm, *cl, *ctx; int8_t *dw;
    cudaMalloc(&LR, (size_t)P * KD * 4); cudaMalloc(&dqv, qv.size() * 4); cudaMalloc(&dgv, gv.size() * 4);
    cudaMalloc(&dw, kvb.size()); cudaMalloc(&ds, O * 4); cudaMalloc(&ctx, (size_t)CALL * H * vh * 4);
    cudaMalloc(&Q, (size_t)CALL * H * KD * 4); cudaMalloc(&S, cap + (size_t)H * ldSm * 4 * 2); cudaMalloc(&cl, (size_t)CALL * H * kvl * 4);
    cudaMalloc(&pm, (size_t)CALL * H * ldpm * 4); cudaMalloc(&sm, (size_t)CALL * H * 4);
    cudaMemcpy2D(LR, (size_t)KD * 4, Lc.data(), (size_t)kvl * 4, (size_t)kvl * 4, P, cudaMemcpyHostToDevice);
    cudaMemcpy2D(LR + kvl, (size_t)KD * 4, Rc.data(), (size_t)qr * 4, (size_t)qr * 4, P, cudaMemcpyHostToDevice);
    cudaMemcpy(dqv, qv.data(), qv.size() * 4, cudaMemcpyHostToDevice); cudaMemcpy(dgv, gv.data(), gv.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dw, kvb.data(), kvb.size(), cudaMemcpyHostToDevice); cudaMemcpy(ds, kvbs.data(), O * 4, cudaMemcpyHostToDevice);
    cudaEvent_t ev[6]; for (auto &e : ev) cudaEventCreate(&e);
    double tk[5] = {0, 0, 0, 0, 0};
    const char *nm[5] = {"k3_mla_q", "scores", "soft", "clat", "k3_mla_out"};
    double mac_s = 0, mac_c = 0;
    const float scale = 1.0f / sqrtf((float)qh);
    for (int c0 = 0; c0 < P; c0 += CALL) {
        const int n = P - c0 < CALL ? P - c0 : CALL, N = c0 + n, pos0 = c0;
        const int ldS = (N + MG_K - 1) / MG_K * MG_K, ldp = (N + MG_T - 1) / MG_T;
        long long tq = cap / ((long long)H * ldS * 4);
        int TQ = tq < 1 ? 1 : tq > n ? n : (int)tq;
        if (quant) {   /* rows a chunk a multiple of the rows that fill the card's 2 blocks an SM in clat */
            int sms = 0; cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
            const int it = (kvl + MG_T - 1) / MG_T, q = 2 * sms * MG_T / (H * it);
            if (q >= 1 && TQ > q && TQ < n) TQ = TQ / q * q;
        }
        for (int r0 = 0; r0 < n; r0 += TQ) {
            const int rows = n - r0 < TQ ? n - r0 : TQ, M = rows * H, first = pos0 + r0;
            const unsigned rt = (rows + MG_T - 1) / MG_T, mt = (M + MG_T - 1) / MG_T, it = (kvl + MG_T - 1) / MG_T, dt = (vh + MG_T - 1) / MG_T;
            cudaEventRecord(ev[0]);
            k3_mla_q<<<dim3(it, rt, H), 256>>>(dqv + (size_t)r0 * H * qh, rows, H, qn, qr, vh, kvl, dw, ds, Q);
            cudaEventRecord(ev[1]);
            k3_mla_scores<<<dim3(ldp, mt), 256>>>(Q, LR, M, N, KD, H, first, scale, S, ldS, pm, ldp);
            cudaEventRecord(ev[2]);
            k3_mla_soft<<<(M + MS_R - 1) / MS_R, 256>>>(S, ldS, pm, ldp, M, H, first, sm);
            cudaEventRecord(ev[3]);
            k3_mla_clat<<<dim3(it, mt), 256>>>(S, ldS, sm, LR, M, N, KD, kvl, H, first, cl);
            cudaEventRecord(ev[4]);
            k3_mla_out<<<dim3(dt, rt, H), 256>>>(cl, dgv + (size_t)r0 * H * vh, rows, H, qn, vh, kvl, dw, ds, ctx + (size_t)r0 * H * vh);
            cudaEventRecord(ev[5]);
            cudaEventSynchronize(ev[5]);
            for (int k = 0; k < 5; k++) { float ms; cudaEventElapsedTime(&ms, ev[k], ev[k + 1]); tk[k] += ms; }
            for (int r = 0; r < rows; r++) { mac_s += (double)H * (first + r + 1) * KD; mac_c += (double)H * (first + r + 1) * kvl; }
        }
    }
    double tot = 0; for (int k = 0; k < 5; k++) tot += tk[k];
    printf("a layer of a %d-token prompt (calls of %d rows, scores %lld MB a chunk): %.1f ms; x24 layers %.2f s\n", P, CALL, cap >> 20, tot, tot * 24 / 1000);
    for (int k = 0; k < 5; k++) printf("  %-12s %8.1f ms  (%4.1f%%)\n", nm[k], tk[k], 100 * tk[k] / tot);
    printf("  useful MACs: scores %.3g at %.2f TMAC/s; clat %.3g at %.2f TMAC/s\n", mac_s, mac_s / tk[1] / 1e9, mac_c, mac_c / tk[3] / 1e9);
    printf("%s\n", cudaGetErrorString(cudaGetLastError()));
    return 0;
}
