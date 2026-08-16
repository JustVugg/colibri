/* backend_loader_dsv4.c — Windows runtime loader for the DS4 CUDA backend DLL.
 *
 * Why this exists (mirrors backend_loader.c / the GLM CUDA_DLL=1 split):
 * the DS4 engine (deepseek_v4.c) is built with MinGW-w64 (gcc), but the CUDA
 * kernels (backend_cuda_dsv4.cu) must be compiled with MSVC + nvcc. We cannot
 * link an nvcc/MSVC object into a MinGW gcc binary (MSVC-COFF references
 * __GSHandlerCheck/__security_cookie which MinGW libc does not provide). The
 * clean cross-toolchain split is: build the CUDA backend into a standalone
 * coli_dsv4_cuda.dll with nvcc+MSVC, then load it here at runtime via
 * LoadLibrary/GetProcAddress. The host (deepseek_v4.exe) never links cudart.
 *
 * On Linux this file is NOT compiled (Makefile.deepseek-v4 links
 * backend_cuda_dsv4.o directly). On Windows, when COLI_DSV4_CUDA is defined,
 * deepseek_v4.c calls the dsv4_cuda_* wrappers below, which forward through
 * function pointers resolved from the DLL at first use. If the DLL is absent,
 * every call safely returns the "not initialized" sentinel (0 / no-op) and
 * the engine falls back to CPU.
 *
 * ABI note: Dsv4CudaTensor* is opaque to the host (it stores the pointer,
 * never dereferences it), so the MSVC-allocated struct is safe to pass across
 * the boundary as an opaque handle. All scalar types (int, pointers) agree
 * between MSVC and MinGW-w64 on x86-64.
 */
#ifdef _WIN32

#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <windows.h>

#include "backend_cuda_dsv4.h"

#define DSV4_BACKEND_DLL "coli_dsv4_cuda.dll"

/* Function-pointer typedefs matching the exported symbols the engine uses. */
typedef int  (*fn_dsv4_init)(const int *devices, int count);
typedef void (*fn_dsv4_shutdown)(void);
typedef int  (*fn_dsv4_upload_fp4)(Dsv4CudaTensor **t, const uint8_t *w,
                                   const uint8_t *scale, int O, int I, int device);
typedef int  (*fn_dsv4_upload_fp8)(Dsv4CudaTensor **t, const uint8_t *w,
                                   const uint8_t *scale, int O, int I, int device);
typedef int  (*fn_dsv4_matvec)(Dsv4CudaTensor *t, float *y, const float *x);
typedef int  (*fn_dsv4_matvec_grouped)(Dsv4CudaTensor *t, float *y,
                                       const float *x, int groups);
typedef int  (*fn_dsv4_expert_group)(Dsv4CudaTensor *const *gate,
                                     Dsv4CudaTensor *const *up,
                                     Dsv4CudaTensor *const *down,
                                     const float *weights, int count,
                                     float limit, float *y, const float *x);
typedef void (*fn_dsv4_tensor_free)(Dsv4CudaTensor *t);

/* Resolved pointers plus a "load attempted" flag (attempt at most once). */
static struct {
    int loaded;     /* 1 = load attempted (success or fail) */
    int available;  /* 1 = DLL loaded and all symbols resolved */
    HMODULE dll;
    fn_dsv4_init             init;
    fn_dsv4_shutdown         shutdown;
    fn_dsv4_upload_fp4       upload_fp4;
    fn_dsv4_upload_fp8       upload_fp8;
    fn_dsv4_matvec           matvec;
    fn_dsv4_matvec_grouped   matvec_grouped;
    fn_dsv4_expert_group     expert_group;
    fn_dsv4_tensor_free      tensor_free;
} g_dsv4;

static const char *dsv4_win_loader_error_text(DWORD code){
    switch(code){
    case ERROR_FILE_NOT_FOUND:  return "the file was not found";
    case ERROR_PATH_NOT_FOUND:  return "the path was not found";
    case ERROR_ACCESS_DENIED:   return "access denied";
    case ERROR_MOD_NOT_FOUND:   return "module or one of its dependencies was not found";
    case ERROR_PROC_NOT_FOUND:  return "a required procedure was not found";
    case ERROR_BAD_EXE_FORMAT:  return "invalid executable format or architecture";
    default:                    return NULL;
    }
}

static int dsv4_cuda_load(void){
    if (g_dsv4.loaded) return g_dsv4.available;
    g_dsv4.loaded = 1;
    g_dsv4.dll = LoadLibraryA(DSV4_BACKEND_DLL);
    if (!g_dsv4.dll){
        DWORD e = GetLastError();
        const char *txt = dsv4_win_loader_error_text(e);
        fprintf(stderr, "[DSV4 CUDA] %s could not be loaded (%lu%s%s); "
                        "expert GPU tier disabled (CPU path remains active).\n",
                DSV4_BACKEND_DLL, (unsigned long)e,
                txt ? ": " : "", txt ? txt : "");
        return 0;
    }
#define DSV4_RESOLVE(name, type) \
    _Pragma("GCC diagnostic push") \
    _Pragma("GCC diagnostic ignored \"-Wcast-function-type\"") \
    g_dsv4.name = (type)GetProcAddress(g_dsv4.dll, "dsv4_cuda_" #name); \
    _Pragma("GCC diagnostic pop") \
    if (!g_dsv4.name){ \
        fprintf(stderr, "[DSV4 CUDA] %s missing symbol dsv4_cuda_" #name "\n", \
                DSV4_BACKEND_DLL); \
        FreeLibrary(g_dsv4.dll); g_dsv4.dll = NULL; return 0; \
    }
    DSV4_RESOLVE(init,          fn_dsv4_init);
    DSV4_RESOLVE(shutdown,      fn_dsv4_shutdown);
    DSV4_RESOLVE(upload_fp4,    fn_dsv4_upload_fp4);
    DSV4_RESOLVE(upload_fp8,    fn_dsv4_upload_fp8);
    DSV4_RESOLVE(matvec,        fn_dsv4_matvec);
    DSV4_RESOLVE(matvec_grouped,fn_dsv4_matvec_grouped);
    DSV4_RESOLVE(expert_group,  fn_dsv4_expert_group);
    DSV4_RESOLVE(tensor_free,   fn_dsv4_tensor_free);
#undef DSV4_RESOLVE
    g_dsv4.available = 1;
    fprintf(stderr, "[DSV4 CUDA] loaded %s\n", DSV4_BACKEND_DLL);
    return 1;
}

/* ---- dsv4_cuda_* wrappers: forward through the DLL, else CPU-fallback sentinel. ---- */

int dsv4_cuda_init(const int *devices, int count){
    return dsv4_cuda_load() ? g_dsv4.init(devices, count) : 0;
}
void dsv4_cuda_shutdown(void){
    if (dsv4_cuda_load()) g_dsv4.shutdown();
    if (g_dsv4.dll){ FreeLibrary(g_dsv4.dll); g_dsv4.dll = NULL; }
    g_dsv4.loaded = 0; g_dsv4.available = 0;
}
int dsv4_cuda_upload_fp4(Dsv4CudaTensor **t, const uint8_t *w,
                         const uint8_t *scale, int O, int I, int device){
    return dsv4_cuda_load() ? g_dsv4.upload_fp4(t, w, scale, O, I, device) : 0;
}
int dsv4_cuda_upload_fp8(Dsv4CudaTensor **t, const uint8_t *w,
                         const uint8_t *scale, int O, int I, int device){
    return dsv4_cuda_load() ? g_dsv4.upload_fp8(t, w, scale, O, I, device) : 0;
}
int dsv4_cuda_matvec(Dsv4CudaTensor *t, float *y, const float *x){
    return dsv4_cuda_load() ? g_dsv4.matvec(t, y, x) : 0;
}
int dsv4_cuda_matvec_grouped(Dsv4CudaTensor *t, float *y, const float *x,
                             int groups){
    return dsv4_cuda_load() ? g_dsv4.matvec_grouped(t, y, x, groups) : 0;
}
int dsv4_cuda_expert_group(Dsv4CudaTensor *const *gate, Dsv4CudaTensor *const *up,
                           Dsv4CudaTensor *const *down, const float *weights,
                           int count, float limit, float *y, const float *x){
    return dsv4_cuda_load() ? g_dsv4.expert_group(gate, up, down, weights, count,
                                                  limit, y, x) : 0;
}
void dsv4_cuda_tensor_free(Dsv4CudaTensor *t){
    if (dsv4_cuda_load()) g_dsv4.tensor_free(t);
}

#endif /* _WIN32 */
