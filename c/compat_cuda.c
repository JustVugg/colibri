/* compat_cuda.c — Definition of the runtime loader for coli_cuda.dll.
 *
 * Compiled only on Windows with COLI_CUDA=1. Owns the 11 function-pointer
 * definitions declared extern in compat_cuda.h and the LoadLibrary logic
 * that populates them. */

#if defined(_WIN32) && defined(COLI_CUDA)

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <stdio.h>

/* Include the shim header. Its call-site redirection macros do NOT interfere
 * with the pointer-variable definitions below because none of those lines
 * contain a bare coli_cuda_* identifier — they contain fn_coli_cuda_* (a
 * typedef name) and p_coli_cuda_* (the pointer variable), both of which are
 * distinct from the macro left-hand side. */
#include "compat_cuda.h"

fn_coli_cuda_init          p_coli_cuda_init          = NULL;
fn_coli_cuda_shutdown      p_coli_cuda_shutdown      = NULL;
fn_coli_cuda_device_count  p_coli_cuda_device_count  = NULL;
fn_coli_cuda_device_at     p_coli_cuda_device_at     = NULL;
fn_coli_cuda_mem_info      p_coli_cuda_mem_info      = NULL;
fn_coli_cuda_stats         p_coli_cuda_stats         = NULL;
fn_coli_cuda_tensor_upload p_coli_cuda_tensor_upload = NULL;
fn_coli_cuda_matmul        p_coli_cuda_matmul        = NULL;
fn_coli_cuda_tensor_free   p_coli_cuda_tensor_free   = NULL;
fn_coli_cuda_tensor_bytes  p_coli_cuda_tensor_bytes  = NULL;
fn_coli_cuda_tensor_device p_coli_cuda_tensor_device = NULL;

/* -1 = attempted and failed; 0 = not attempted yet; 1 = loaded OK. */
static int g_loaded_state = 0;
static HMODULE g_dll = NULL;

/* Resolve a single symbol. Uses stringification to avoid quoting the name
 * at the call site — safe here because ## and # inhibit macro expansion. */
#define RESOLVE(name) do {                                                     \
    p_##name = (fn_##name)(void *)GetProcAddress(g_dll, #name);                \
    if (!p_##name) {                                                           \
        fprintf(stderr, "[compat_cuda] symbol %s missing in coli_cuda.dll\n",  \
                #name);                                                        \
        goto fail;                                                             \
    }                                                                          \
} while (0)

int compat_cuda_load(void) {
    if (g_loaded_state) return g_loaded_state > 0;
    g_loaded_state = -1;  /* mark attempted */

    g_dll = LoadLibraryA("coli_cuda.dll");
    if (!g_dll) {
        fprintf(stderr,
            "[compat_cuda] LoadLibrary(\"coli_cuda.dll\") failed (Windows err %lu).\n"
            "  On Windows the CUDA backend lives in a separate DLL built with\n"
            "  MSVC + nvcc. Build it with:  make coli_cuda.dll CUDA=1\n"
            "  and ensure coli_cuda.dll is in the working directory or PATH.\n",
            GetLastError());
        return 0;
    }

    RESOLVE(coli_cuda_init);
    RESOLVE(coli_cuda_shutdown);
    RESOLVE(coli_cuda_device_count);
    RESOLVE(coli_cuda_device_at);
    RESOLVE(coli_cuda_mem_info);
    RESOLVE(coli_cuda_stats);
    RESOLVE(coli_cuda_tensor_upload);
    RESOLVE(coli_cuda_matmul);
    RESOLVE(coli_cuda_tensor_free);
    RESOLVE(coli_cuda_tensor_bytes);
    RESOLVE(coli_cuda_tensor_device);

    g_loaded_state = 1;
    return 1;

fail:
    FreeLibrary(g_dll);
    g_dll = NULL;
    return 0;
}

#undef RESOLVE

#endif /* _WIN32 && COLI_CUDA */
