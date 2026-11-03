// bench_b8.cu: the k3 DLL's tile kernels (backend_cuda_k3.cu included whole) against coli_cuda.dll's
// quant_matmul_blocked at Kimi K3's held shapes, the kernels alone (events), T multiply-adds a second.
//   nvcc -O3 -std=c++17 -arch=sm_89 -ftz=false -I<c> bench_b8.cu -o bench_b8.exe
#include "backend_cuda_k3.cu"
#include "qmb_kernel.cuh"
#include <vector>

#define CK(x) do{cudaError_t e_=(x); if(e_!=cudaSuccess){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e_)); exit(1);}}while(0)

static float run(int which, int fmt, const uint8_t *w, const float *s, const float *x, const float *xb, int S, int I, int O, float *y, int reps) {
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    auto go = [&]() {
        if (which == 0) k3_dense_launch(fmt, w, s, x, S, I, O, 64, y, 0, 1);
        else {
            const dim3 grid((unsigned)(O / QMB_RO), (unsigned)((S + QMB_RS - 1) / QMB_RS));
            if (fmt == 4 && I == 7168) quant_matmul_blocked<4, 7168><<<grid, 256>>>(y, xb, w, s, S, O);
            else if (fmt == 4) quant_matmul_blocked<4, 12288><<<grid, 256>>>(y, xb, w, s, S, O);
            else if (I == 7168) quant_matmul_blocked<1, 7168><<<grid, 256>>>(y, xb, w, s, S, O);
            else quant_matmul_blocked<1, 12288><<<grid, 256>>>(y, xb, w, s, S, O);
        }
    };
    go(); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    cudaEventRecord(a);
    for (int i = 0; i < reps; i++) go();
    cudaEventRecord(b); CK(cudaEventSynchronize(b));
    float ms; cudaEventElapsedTime(&ms, a, b);
    return ms / reps;
}

int main(int argc, char **argv) {
    const int reps = argc > 1 ? atoi(argv[1]) : 20;
    struct { const char *name; int O, I, fmt, qmb; } sh[] = {
        {"MLA g 12288x7168 int8", 12288, 7168, 1, 1}, {"MLA o 7168x12288 int8", 7168, 12288, 1, 1},
        {"MLA qb 18432x1536 int8", 18432, 1536, 1, 0}, {"MLA qa 1536x7168 int8", 1536, 7168, 1, 1},
        {"MLA kva 576x7168 int8", 576, 7168, 1, 0},
        {"KDA q 12288x7168 int4", 12288, 7168, 4, 1}, {"KDA o 7168x12288 int4", 7168, 12288, 4, 1}};
    const int rows[] = {256, 512, 1024};
    CK(cudaSetDevice(0));
    for (auto &h : sh) {
        const size_t wb = h.fmt == 1 ? (size_t)h.O * h.I : (size_t)h.O * h.I / 2, sn = h.fmt == 1 ? (size_t)h.O : (size_t)h.O * h.I / 64;
        std::vector<uint8_t> hw(wb); std::vector<float> hs(sn);
        for (auto &v : hw) v = (uint8_t)(rand() & 255);
        for (auto &v : hs) v = 1e-3f * (1 + (rand() & 7));
        uint8_t *dw; float *ds, *dx, *dxb, *dy;
        const int Smax = 1024;
        std::vector<float> hx((size_t)Smax * h.I);
        for (auto &v : hx) v = (float)(rand() % 2001 - 1000) / 1000.f;
        CK(cudaMalloc(&dw, wb)); CK(cudaMalloc(&ds, sn * 4)); CK(cudaMalloc(&dx, hx.size() * 4));
        CK(cudaMalloc(&dxb, (size_t)(Smax + 8) * h.I * 4)); CK(cudaMalloc(&dy, (size_t)Smax * h.O * 4));
        CK(cudaMemcpy(dw, hw.data(), wb, cudaMemcpyHostToDevice)); CK(cudaMemcpy(ds, hs.data(), sn * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dx, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice));
        printf("%-26s", h.name);
        for (int S : rows) {
            const double mac = (double)S * h.O * h.I;
            const float t0 = run(0, h.fmt, dw, ds, dx, dxb, S, h.I, h.O, dy, reps);
            printf("  %4d rows: k3 %5.1f", S, mac / (t0 * 1e-3) / 1e12);
            if (h.qmb) {
                const long long n = (long long)((S + QMB_RS - 1) / QMB_RS) * h.I;
                qmb_rows<<<(unsigned)((n + 255) / 256), 256>>>(dxb, dx, S, h.I, n);
                const float t1 = run(1, h.fmt, dw, ds, dx, dxb, S, h.I, h.O, dy, reps);
                printf(", qmb %5.1f", mac / (t1 * 1e-3) / 1e12);
            }
        }
        printf("  T/s\n");
        cudaFree(dw); cudaFree(ds); cudaFree(dx); cudaFree(dxb); cudaFree(dy);
    }
    return 0;
}
