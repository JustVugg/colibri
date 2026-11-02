/* The dense chain's int8 GEMV (coli_cuda_tensor_gemm_async, gemv_i8_rows) on a speculative
 * verify's rows: every row of 2, 3 and 4 has the bits of the same row alone (a decode
 * step's), the rows sharing each weight load or not (a matrix of few blocks keeps a block a
 * row; a long row fits fewer rows in shared memory). BENCH=1 also times 1 to 4 rows, the
 * weights read once a verify against once a row (COLI_CUDA_GEMV_NR=1). Build from c/:
 *   nvcc -O2 -std=c++17 -arch=native tests/test_gemv_rows_cuda.cu -o tests/test_gemv_rows_cuda */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cuda_runtime.h>
#include "../backend_cuda.cu"

static int fails;
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

static void check(int I, int O, int bench) {
    int8_t *w = (int8_t *)malloc((size_t)O * I);
    float *sc = (float *)malloc((size_t)O * 4), *x = (float *)malloc((size_t)4 * I * 4);
    for (size_t i = 0; i < (size_t)O * I; i++) w[i] = (int8_t)((rand() % 255) - 127);
    for (int o = 0; o < O; o++) sc[o] = 0.001f + (rand() % 100) * 1e-5f;
    for (int i = 0; i < 4 * I; i++) x[i] = (rand() / (float)RAND_MAX - .5f) * 2.f;
    ColiCudaTensor *t = NULL;
    if (!coli_cuda_tensor_upload(&t, w, sc, 1, I, O, 0)) { printf("FAIL upload %d x %d\n", O, I); fails++; return; }
    float *xd, *yd, *one = (float *)malloc((size_t)4 * O * 4), *got = (float *)malloc((size_t)4 * O * 4);
    cudaMalloc(&xd, (size_t)4 * I * 4); cudaMalloc(&yd, (size_t)4 * O * 4);
    cudaMemcpy(xd, x, (size_t)4 * I * 4, cudaMemcpyHostToDevice);
    for (int s = 0; s < 4; s++) coli_cuda_tensor_gemm_async(t, yd + (size_t)s * O, xd + (size_t)s * I, 1, NULL);
    cudaDeviceSynchronize();
    cudaMemcpy(one, yd, (size_t)4 * O * 4, cudaMemcpyDeviceToHost);
    for (int S = 2; S <= 4; S++) {
        cudaMemset(yd, 0, (size_t)4 * O * 4);
        int ok = coli_cuda_tensor_gemm_async(t, yd, xd, S, NULL) && cudaDeviceSynchronize() == cudaSuccess;
        cudaMemcpy(got, yd, (size_t)S * O * 4, cudaMemcpyDeviceToHost);
        int same = ok && !memcmp(got, one, (size_t)S * O * 4);
        if (!same) { printf("FAIL %d x %d: %d rows differ from the rows alone\n", O, I, S); fails++; }
    }
    if (bench) {
        printf("int8 [%6d x %5d]:", O, I);
        for (int S = 1; S <= 4; S++) {
            double best = 1e9;
            for (int r = 0; r < 5; r++) {
                cudaDeviceSynchronize(); double t0 = now();
                for (int k = 0; k < 20; k++) coli_cuda_tensor_gemm_async(t, yd, xd, S, NULL);
                cudaDeviceSynchronize(); double dt = (now() - t0) / 20;
                if (dt < best) best = dt;
            }
            printf(" %d rows %.1f us", S, best * 1e6);
        }
        printf("\n");
    }
    cudaFree(xd); cudaFree(yd); coli_cuda_tensor_free(t);
    free(w); free(sc); free(x); free(one); free(got);
}

int main(void) {
    int devs[1] = {0};
    if (!coli_cuda_init(devs, 1)) { printf("SKIP: no CUDA device\n"); return 0; }
    int bench = getenv("BENCH") && *getenv("BENCH") == '1';
    int shapes[][2] = {{2048, 300}, {2048, 8192}, {4096, 2048}, {2048, 512}, {2048, 100}, {12288, 640}, {2048, 248320}};
    for (unsigned k = 0; k < sizeof shapes / sizeof shapes[0]; k++) check(shapes[k][0], shapes[k][1], bench);
    if (fails) { printf("test_gemv_rows_cuda: %d failure(s)\n", fails); return 1; }
    printf("OK test_gemv_rows_cuda: a verify's rows have a decode step's bits\n");
    return 0;
}
