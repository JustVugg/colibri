/* test_dll_loadlib.c — validates coli_cuda.dll loads via LoadLibrary +
 * GetProcAddress on Windows. This is the exact pattern compat.h will use in
 * Bloque 5 to wire the engine to the CUDA backend. Standalone C99, no
 * dependency on backend_cuda.h — we redeclare the types locally to prove
 * that a downstream consumer only needs the DLL + its ABI, not the .cu source. */

#include <windows.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>

typedef struct ColiCudaTensor ColiCudaTensor;

/* Function-pointer types matching backend_cuda.h signatures. */
typedef int  (*fn_init)(const int *devices, int count);
typedef void (*fn_shutdown)(void);
typedef int  (*fn_device_count)(void);
typedef int  (*fn_mem_info)(int device, size_t *free_bytes, size_t *total_bytes);
typedef int  (*fn_tensor_upload)(ColiCudaTensor **tensor, const void *weights,
                                 const float *scales, int fmt, int I, int O, int device);
typedef int  (*fn_matmul)(ColiCudaTensor **tensor, float *y, const float *x,
                          const void *weights, const float *scales,
                          int fmt, int S, int I, int O, int device);
typedef void (*fn_tensor_free)(ColiCudaTensor *tensor);

#define REQUIRE(cond, msg) do { if(!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return 1; } } while(0)

int main(void) {
    HMODULE dll = LoadLibraryA("coli_cuda.dll");
    if (!dll) {
        fprintf(stderr, "FAIL: LoadLibrary(coli_cuda.dll) returned NULL (err %lu)\n",
                GetLastError());
        return 1;
    }
    fprintf(stderr, "OK: DLL loaded at %p\n", (void *)dll);

    fn_init init = (fn_init)(void *)GetProcAddress(dll, "coli_cuda_init");
    fn_shutdown shutdown = (fn_shutdown)(void *)GetProcAddress(dll, "coli_cuda_shutdown");
    fn_device_count devcount = (fn_device_count)(void *)GetProcAddress(dll, "coli_cuda_device_count");
    fn_mem_info meminfo = (fn_mem_info)(void *)GetProcAddress(dll, "coli_cuda_mem_info");
    fn_tensor_upload upload = (fn_tensor_upload)(void *)GetProcAddress(dll, "coli_cuda_tensor_upload");
    fn_matmul matmul = (fn_matmul)(void *)GetProcAddress(dll, "coli_cuda_matmul");
    fn_tensor_free tfree = (fn_tensor_free)(void *)GetProcAddress(dll, "coli_cuda_tensor_free");

    REQUIRE(init && shutdown && devcount && meminfo && upload && matmul && tfree,
            "GetProcAddress for one of the 7 required symbols returned NULL");
    fprintf(stderr, "OK: 7 symbols resolved via GetProcAddress\n");

    int devices[] = {0};
    REQUIRE(init(devices, 1), "coli_cuda_init failed");
    fprintf(stderr, "OK: coli_cuda_init(devices=[0], count=1) succeeded\n");

    REQUIRE(devcount() == 1, "coli_cuda_device_count != 1");
    fprintf(stderr, "OK: coli_cuda_device_count() == 1\n");

    size_t free_b = 0, total_b = 0;
    REQUIRE(meminfo(0, &free_b, &total_b), "coli_cuda_mem_info failed");
    fprintf(stderr, "OK: coli_cuda_mem_info(dev=0) -> free=%.2f GB / total=%.2f GB\n",
            (double)free_b / 1e9, (double)total_b / 1e9);

    /* Numerical: 1x4 @ 4x2 int8 matmul. Same as author's test row 1. */
    const float x[4] = {1, -2, 3, -4};
    const int8_t q8[8] = {1, 2, 3, 4, -1, 2, -3, 4};
    const float scales[2] = {0.5f, 2.0f};
    const float expected[2] = {-5.0f, -60.0f};
    float y[2] = {0, 0};
    ColiCudaTensor *t = NULL;

    REQUIRE(matmul(&t, y, x, q8, scales, /*fmt=int8*/1, /*S=*/1, /*I=*/4, /*O=*/2, 0),
            "coli_cuda_matmul (int8) failed");
    for (int i = 0; i < 2; i++) {
        if (fabsf(y[i] - expected[i]) > 1e-4f) {
            fprintf(stderr, "FAIL: matmul[%d] = %.6f, expected %.6f\n", i, y[i], expected[i]);
            return 1;
        }
    }
    fprintf(stderr, "OK: int8 matmul via DLL matches expected (%.2f, %.2f)\n", y[0], y[1]);

    tfree(t);
    shutdown();
    FreeLibrary(dll);
    fprintf(stderr, "OK: shutdown + FreeLibrary clean\n\nALL_GREEN\n");
    return 0;
}
