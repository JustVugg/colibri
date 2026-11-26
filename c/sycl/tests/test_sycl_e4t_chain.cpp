/* fmt 9 (coli_cuda_tensor_upload_e4t): an e4m3 matrix with one ue8m0 byte a 32x32 tile, as the dense chain multiplies
 * it (the chain's table: T->matmul inside a frame, coli_cuda_tensor_gemm_async on the chain's queue). Every output must
 * be, byte for byte, deepseek_v41.c's mv8_rows_cpu on an AVX2 build: per output and row, per 32-column tile, lanes
 * j = 0..7 take columns j, j+8, j+16, j+24: acc = w0*x0 (a multiply), then three fused multiply-adds; h =
 * ((a0+a4)+(a2+a6)) + ((a1+a5)+(a3+a7)); sum = fma(h, tile, sum). The reference below replays that with each operation a
 * statement of its own (std::fma where the CPU fuses), so the compiler has nothing to contract. Shapes: DeepSeek V4.1's
 * trunk (wq_a, wq_b, wkv, a wo_a group, wo_b, the shared expert) and odd ones (O not a multiple of 32 or of the 8-row
 * pass), 1..32 rows, subnormal weights included (no 0x7F/0xFF: a NaN's payload is not a value to compare). Then the
 * one-row product timed over the trunk shapes (E4T-TIME): GB/s of weight bytes. "E4T SUITE PASS" at the end. */
#include "../../backend_cuda.h"
#include "../../cuda_chain.h"
#include "sycl_test_args.h"
#include <chrono>
#include <cmath>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static const ColiCudaChainOps *T;
static int fails;
static void require(bool ok, const char *what) {
    if (!ok) { std::fprintf(stderr, "FAIL E4T: %s\n", what); std::exit(1); }
}
static uint32_t state = 0x9e3779b9u;
static uint32_t rnd() { state ^= state << 13; state ^= state >> 17; state ^= state << 5; return state; }

static float e4(uint8_t b) {   /* quant.h's e4m3_decode8 for one byte (bits placed, times 2^120) */
    const uint32_t bits = ((uint32_t)(b & 0x80) << 24) | ((uint32_t)(b & 0x7f) << 20);
    float v; std::memcpy(&v, &bits, 4);
    return v * 0x1p120f;
}
static float ue8(uint8_t b) { const uint32_t u = (uint32_t)b << 23; float f; std::memcpy(&f, &u, 4); return f; }

static void reference(float *y, const uint8_t *q, const uint8_t *s, const float *x, int S, int I, int O) {
    const int tiles = I / 32;
    for (int r = 0; r < S; r++)
        for (int o = 0; o < O; o++) {
            const uint8_t *row = q + (size_t)o * I, *sc = s + (size_t)(o / 32) * tiles;
            const float *xr = x + (size_t)r * I;
            float sum = 0.0f;
            for (int t = 0; t < tiles; t++) {
                const int b = t * 32;
                float a[8];
                for (int j = 0; j < 8; j++) {
                    float acc = e4(row[b + j]) * xr[b + j];
                    acc = std::fma(e4(row[b + 8 + j]), xr[b + 8 + j], acc);
                    acc = std::fma(e4(row[b + 16 + j]), xr[b + 16 + j], acc);
                    acc = std::fma(e4(row[b + 24 + j]), xr[b + 24 + j], acc);
                    a[j] = acc;
                }
                const float p0 = a[0] + a[4], p1 = a[1] + a[5], p2 = a[2] + a[6], p3 = a[3] + a[7];
                const float q0 = p0 + p2, q1 = p1 + p3;
                const float h = q0 + q1;
                sum = std::fma(h, ue8(sc[t]), sum);
            }
            y[(size_t)r * O + o] = sum;
        }
}
static CcBuf *up(const std::vector<float> &v) {
    CcBuf *b = T->buf(v.size() * 4, CC_DEV);
    require(b && T->begin() && T->write(b, 0, v.data(), v.size() * 4) && T->submit(1), "upload to a chain buffer");
    return b;
}
static void make(std::vector<uint8_t> &q, std::vector<uint8_t> &s, int O, int I) {
    q.resize((size_t)O * I); s.resize((size_t)((O + 31) / 32) * (I / 32));
    for (auto &b : q) { uint8_t v; do v = (uint8_t)rnd(); while ((v & 0x7f) == 0x7f); b = v; }
    for (auto &b : s) b = (uint8_t)(110 + rnd() % 20);
}
static void shape(int dev, int O, int I, int S) {
    std::vector<uint8_t> q, s; make(q, s, O, I);
    std::vector<float> x((size_t)S * I), y((size_t)S * O), want((size_t)S * O);
    for (auto &v : x) v = (float)((int)(rnd() % 20001) - 10000) / 3371.0f;
    ColiCudaTensor *t = nullptr;
    require(coli_cuda_tensor_upload_e4t(&t, q.data(), s.data(), O, I, dev), "upload_e4t");
    CcBuf *xb = up(x), *yb = T->buf((size_t)S * O * 4, CC_DEV);
    require(yb && T->begin() && T->matmul(t, xb, 0, yb, 0, S) && T->submit(1), "the chain's matmul on fmt 9");
    require(T->read(yb, 0, y.data(), y.size() * 4), "read back");
    reference(want.data(), q.data(), s.data(), x.data(), S, I, O);
    /* COLI_SYCL_E4T_FAST (default 1): rows <= 4 take the fast product, held to a tolerance (max |err| / max |ref| < 1e-5);
     * otherwise (and with COLI_SYCL_E4T_FAST=0) the exact kernel, byte for byte */
    const char *fe = getenv("COLI_SYCL_E4T_FAST");
    const bool fast = !(fe && *fe == '0') && S <= 4;
    size_t diff = 0;
    double me = 0, mr = 0;
    for (size_t k = 0; k < y.size(); k++) {
        diff += std::memcmp(&y[k], &want[k], 4) != 0;
        me = std::fmax(me, std::fabs((double)y[k] - want[k])); mr = std::fmax(mr, std::fabs((double)want[k]));
    }
    const double rel = mr > 0 ? me / mr : me;
    if (fast) std::printf("E4T O=%d I=%d S=%d fast: max|err|/max|ref| %.3g (%zu of %zu not byte-equal)\n", O, I, S, rel, diff, y.size());
    else std::printf("E4T O=%d I=%d S=%d: %zu of %zu outputs differ\n", O, I, S, diff, y.size());
    std::fflush(stdout);
    if (fast ? !(rel < 1e-5) : diff != 0) fails++;
    T->free(xb); T->free(yb);
    coli_cuda_tensor_free(t);
}
/* matmul_grp (DeepSeek V4.1's wo_a groups): ng matrices of ol x I in one launch, inputs slices of each row, against
 * one T->matmul a matrix: the same bits (the same KS, the same arithmetic a row); then both timed (GRP-TIME) */
static void grouped(int dev, int ng, int ol, int I, int S) {
    std::vector<std::vector<uint8_t>> q(ng), s(ng);
    std::vector<ColiCudaTensor *> t(ng, nullptr);
    for (int k = 0; k < ng; k++) { make(q[k], s[k], ol, I); require(coli_cuda_tensor_upload_e4t(&t[k], q[k].data(), s[k].data(), ol, I, dev), "upload_e4t (grouped)"); }
    std::vector<float> x((size_t)S * ng * I), ya((size_t)S * ng * ol), yb((size_t)S * ng * ol);
    for (auto &v : x) v = (float)((int)(rnd() % 20001) - 10000) / 3371.0f;
    CcBuf *xb = up(x), *ga = T->buf(ya.size() * 4, CC_DEV), *gb = T->buf(yb.size() * 4, CC_DEV), *xg = T->buf(x.size() * 4, CC_DEV),
          *yg = T->buf(ya.size() * 4, CC_DEV);
    require(T->size >= offsetof(ColiCudaChainOps, matmul_grp) + sizeof T->matmul_grp && T->matmul_grp, "matmul_grp in the table");
    require(T->begin() && T->matmul_grp(t.data(), ng, xb, 0, ng * I, I, ga, 0, ng * ol, S) == 1 && T->submit(1), "matmul_grp");
    /* the per-matrix way: group-major copies (k, s), a matmul a group, back to row-major */
    std::vector<CcRegion> r1((size_t)S * ng), r2((size_t)S * ng);
    for (int r = 0; r < S; r++)
        for (int k = 0; k < ng; k++) {
            r1[r * ng + k] = CcRegion{(size_t)k * S * I + (size_t)r * I, (size_t)r * ng * I + (size_t)k * I, (size_t)I};
            r2[r * ng + k] = CcRegion{(size_t)r * ng * ol + (size_t)k * ol, (size_t)k * S * ol + (size_t)r * ol, (size_t)ol};
        }
    int ok = T->begin() && T->copy_regions(xg, xb, r1.data(), S * ng);
    for (int k = 0; ok && k < ng; k++) ok = T->matmul(t[k], xg, (size_t)k * S * I, yg, (size_t)k * S * ol, S);
    require(ok && T->copy_regions(gb, yg, r2.data(), S * ng) && T->submit(1), "per-matrix matmuls");
    require(T->read(ga, 0, ya.data(), ya.size() * 4) && T->read(gb, 0, yb.data(), yb.size() * 4), "read back (grouped)");
    size_t diff = 0;
    double me = 0, mr = 0;
    for (size_t i = 0; i < ya.size(); i++) {
        diff += std::memcmp(&ya[i], &yb[i], 4) != 0;
        me = std::fmax(me, std::fabs((double)ya[i] - yb[i])); mr = std::fmax(mr, std::fabs((double)yb[i]));
    }
    /* below COLI_SYCL_E4T_OB_MIN rows (default 3; the grouped launch takes the 4-output kernel from 2) a lone matrix runs the
     * KS rows kernel: two kernels by design, held to the fast products' tolerance; otherwise the same bits */
    const char *om = getenv("COLI_SYCL_E4T_OB_MIN");
    const int obmin = om && *om ? std::max(2, std::atoi(om)) : 3;
    const bool two = S >= 2 && S < obmin;
    const double rel = mr > 0 ? me / mr : me;
    if (two) std::printf("GRP %d x %dx%d S=%d: grouped vs per-matrix (two kernels), max|err|/max|ref| %.3g\n", ng, ol, I, S, rel);
    else std::printf("GRP %d x %dx%d S=%d: grouped vs per-matrix, %zu of %zu values differ\n", ng, ol, I, S, diff, ya.size());
    if (two ? !(rel < 1e-5) : diff != 0) fails++;
    for (int mode = 0; mode < 2; mode++) {
        const int n = 200;
        const auto t0 = std::chrono::steady_clock::now();
        require(T->begin(), "begin (timed)");
        for (int it = 0; it < n; it++) {
            if (mode == 0) require(T->matmul_grp(t.data(), ng, xb, 0, ng * I, I, ga, 0, ng * ol, S) == 1, "grouped (timed)");
            else for (int k = 0; k < ng; k++) require(T->matmul(t[k], xg, (size_t)k * S * I, yg, (size_t)k * S * ol, S), "per-matrix (timed)");
        }
        require(T->submit(1), "submit (timed)");
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / n;
        std::printf("GRP-TIME %d x %dx%d S=%d %s: %.1f us\n", ng, ol, I, S, mode == 0 ? "one launch" : "a launch a matrix", us);
    }
    for (CcBuf *b : {xb, ga, gb, xg, yg}) T->free(b);
    for (auto *tt : t) coli_cuda_tensor_free(tt);
}
static void timed(int dev, int S = 1) {   /* S rows, the trunk's shapes, many launches in a frame */
    const int sh[][2] = { {1024, 5120}, {32768, 1024}, {576, 5120}, {1024, 4096}, {5120, 8192}, {2048, 5120}, {5120, 2048} };
    for (auto &p : sh) {
        const int O = p[0], I = p[1], N = 200;
        std::vector<uint8_t> q, s; make(q, s, O, I);
        ColiCudaTensor *t = nullptr;
        require(coli_cuda_tensor_upload_e4t(&t, q.data(), s.data(), O, I, dev), "upload_e4t (timed)");
        std::vector<float> x((size_t)S * I, 0.25f);
        CcBuf *xb = up(x), *yb = T->buf((size_t)S * O * 4, CC_DEV);
        double best = 1e9, sub = 0;
        for (int r = 0; r < 4; r++) {
            auto t0 = std::chrono::steady_clock::now();
            bool ok = T->begin();
            for (int i = 0; i < N && ok; i++) ok = T->matmul(t, xb, 0, yb, 0, S);
            auto t1 = std::chrono::steady_clock::now();
            ok = T->submit(1) && ok;
            require(ok, "timed launches");
            double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / N;
            if (dt < best) { best = dt; sub = std::chrono::duration<double>(t1 - t0).count() / N; }
        }
        /* host: the launches' own time on the host (enqueue); a product's total at or near it means the host bounds it */
        std::printf("E4T-TIME O=%d I=%d S=%d: %.1f us a product (host enqueue %.1f us), %.0f GB/s of weights\n", O, I, S,
                    best * 1e6, sub * 1e6, (double)O * I / best / 1e9);
        std::fflush(stdout);
        T->free(xb); T->free(yb);
        coli_cuda_tensor_free(t);
    }
}
/* references for the timings: the backend's int8 rows (fmt 1, the tuned gemv_i8 path) on the same shapes and bytes,
 * and a 32x32 product (a launch's floor on this queue) */
static void timed_ref(int dev) {
    const int sh[][2] = { {32, 32}, {1024, 5120}, {32768, 1024}, {576, 5120}, {5120, 8192}, {5120, 2048} };
    for (auto &p : sh) {
        const int O = p[0], I = p[1], N = 200;
        std::vector<int8_t> q((size_t)O * I); for (auto &v : q) v = (int8_t)((int)(rnd() % 255) - 127);
        std::vector<float> sc(O, 0.01f);
        ColiCudaTensor *t = nullptr;
        require(coli_cuda_tensor_upload(&t, q.data(), sc.data(), 1, I, O, dev), "upload fmt 1 (timed)");
        std::vector<float> x(I, 0.25f);
        CcBuf *xb = up(x), *yb = T->buf((size_t)O * 4, CC_DEV);
        double best = 1e9;
        for (int r = 0; r < 4; r++) {
            auto t0 = std::chrono::steady_clock::now();
            bool ok = T->begin();
            for (int i = 0; i < N && ok; i++) ok = T->matmul(t, xb, 0, yb, 0, 1);
            ok = T->submit(1) && ok;
            require(ok, "timed fmt 1 launches");
            double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / N;
            if (dt < best) best = dt;
        }
        std::printf("I8REF-TIME O=%d I=%d S=1: %.1f us a product, %.0f GB/s of weights\n", O, I, best * 1e6, (double)O * I / best / 1e9);
        std::fflush(stdout);
        T->free(xb); T->free(yb);
        coli_cuda_tensor_free(t);
    }
}
int main(int argc, char **argv) {
    int dev[1]; int n = sycl_test_devices(argc, argv, dev, 1); if (n < 0) return 0;
    require(coli_cuda_init(dev, n), "init");
    T = coli_cuda_chain_ops();
    require(T && T->size == sizeof(ColiCudaChainOps) && T->init(dev[0]) && T->ready(), "the chain on the device");
    const int sh[][3] = { {64, 32, 1}, {1024, 5120, 1}, {1024, 5120, 4}, {577, 5120, 3}, {32768, 1024, 1}, {576, 5120, 8},
                          {1024, 4096, 9}, {5120, 8192, 2}, {2048, 5120, 32}, {5120, 2048, 7}, {33, 96, 17},
                          {1024, 4096, 4}, {32768, 1280, 4}, {2304, 5120, 3}, {1027, 2048, 2}, {64, 32, 4} };
    for (auto &p : sh) shape(dev[0], p[0], p[1], p[2]);
    grouped(dev[0], 8, 1024, 4096, 1); grouped(dev[0], 8, 1024, 4096, 4); grouped(dev[0], 3, 64, 96, 2);
    timed(dev[0]);
    timed(dev[0], 4);
    timed_ref(dev[0]);
    timed(dev[0], 2); timed(dev[0], 3);   /* last: the 2- and 3-row timings (the RNG streams above unchanged) */
    grouped(dev[0], 8, 1024, 4096, 2);
    std::printf(fails ? "E4T SUITE FAIL\n" : "E4T SUITE PASS\n");
    return fails ? 1 : 0;
}
