/* Synthetic XDNA helper for hardware-free tests. Contains NO XRT and touches no
 * device: it implements the ABI generation 2 contract in plain C and computes
 * the GEMM on the CPU.
 *
 * This exists so the whole execution control flow -- gate order, artifact open,
 * userptr wrap, activation staging, padding, logical-row extraction, failure
 * classification and fallback -- is qualified on a machine with no NPU. The
 * physical qualification of the NUMERICS is a separate, explicitly hardware
 * run; nothing here substitutes for it.
 *
 * Build variants:
 *   (default)      complete, ABI 2
 *   -DFAKE_ABI1    reports ABI generation 1 -- must be refused outright
 *   -DFAKE_PARTIAL ABI 2 but missing one required export -- must be refused
 */

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#define API __declspec(dllexport)

#ifdef FAKE_ABI1
#define FAKE_ABI 1u
#else
#define FAKE_ABI 2u
#endif

enum {
    H_OK           =  0,
    H_E_DEVICE     = -1,
    H_E_ARTIFACT   = -2,
    H_E_NOT_OPEN   = -3,
    H_E_WRAP       = -4,
    H_E_SIZE       = -5,
    H_E_DISPATCH   = -6,
    H_E_COMPLETION = -7,
    H_E_EXCEPTION  = -8
};

/* Which stage the next call should fail at. 0 = none. */
enum { F_NONE = 0, F_DEVICE, F_ARTIFACT, F_WRAP, F_EXECUTE, F_COMPLETION };

static int g_fail;
static int g_open, g_wrap, g_exec, g_relw, g_shut;
static int g_is_open;
static unsigned short *g_w;             /* SNAPSHOT of the caller image, see wrap */
static const void *g_w_src;              /* the caller pointer it was taken from */
static size_t g_w_elems;
static size_t g_m, g_k, g_n;
static char g_err[256] = {0};

static float b2f(unsigned short b){
    unsigned int u = (unsigned int)b << 16;
    float f; memcpy(&f, &u, sizeof f); return f;
}

API unsigned int coli_xdna_helper_abi_version(void){ return FAKE_ABI; }
API const char *coli_xdna_helper_last_error(void){ return g_err; }

API int coli_xdna_helper_open(const char *xclbin, const char *insts,
                              uint32_t m, uint32_t k, uint32_t n){
    g_open++;
    if(g_fail == F_DEVICE){ snprintf(g_err,sizeof g_err,"injected device failure"); return H_E_DEVICE; }
    if(g_fail == F_ARTIFACT){ snprintf(g_err,sizeof g_err,"injected artifact failure"); return H_E_ARTIFACT; }
    /* The host is supposed to hand us real, already-verified paths. Checking
     * they are at least openable keeps the fake honest about that contract. */
    if(!xclbin || !insts) return H_E_ARTIFACT;
    { FILE *f = fopen(xclbin, "rb"); if(!f) return H_E_ARTIFACT; fclose(f); }
    { FILE *f = fopen(insts,  "rb"); if(!f) return H_E_ARTIFACT; fclose(f); }
    g_m = m; g_k = k; g_n = n; g_is_open = 1; free(g_w); g_w = NULL; g_w_src = NULL;
    return H_OK;
}

API int coli_xdna_helper_wrap_weight(void *bf16, uint64_t bytes){
    g_wrap++;
    if(!g_is_open) return H_E_NOT_OPEN;
    if(g_fail == F_WRAP){ snprintf(g_err,sizeof g_err,"injected wrap failure"); return H_E_WRAP; }
    if(!bf16 || bytes != (uint64_t)g_k * g_n * 2) return H_E_SIZE;
    /* The alignment the qualified userptr path requires. The fake enforces it
     * so a host regression that dropped the check would still be caught here. */
    if(((uintptr_t)bf16 % 4096u) != 0){ snprintf(g_err,sizeof g_err,"unaligned userptr"); return H_E_WRAP; }
    /* SNAPSHOT, deliberately. The real helper wraps this memory in an
     * xrt::ext::bo and immediately sync()s it BO_TO_DEVICE, so the device sees
     * the bytes as they were AT WRAP TIME. Reading the caller pointer live at
     * execute time would model a shortcut the real path does not take, and
     * would hide any host bug that fails to re-wrap after the image changes. */
    { size_t elems = (size_t)(bytes / 2);
      unsigned short *snap = (unsigned short *)malloc(bytes ? (size_t)bytes : 2);
      if(!snap) return H_E_WRAP;
      memcpy(snap, bf16, (size_t)bytes);
      free(g_w); g_w = snap; g_w_src = bf16; g_w_elems = elems; }
    return H_OK;
}

#ifndef FAKE_PARTIAL
API int coli_xdna_helper_execute(const void *a_bf16, uint64_t a_bytes,
                                 void *c_f32, uint64_t c_bytes){
    g_exec++;
    if(!g_is_open || !g_w) return H_E_NOT_OPEN;
    if(g_fail == F_EXECUTE){ snprintf(g_err,sizeof g_err,"injected dispatch failure"); return H_E_DISPATCH; }
    if(a_bytes != (uint64_t)g_m * g_k * 2) return H_E_SIZE;
    if(c_bytes != (uint64_t)g_m * g_n * 4) return H_E_SIZE;
    {
        const unsigned short *A = (const unsigned short *)a_bf16;
        float *C = (float *)c_f32;
        size_t i, j, kk;
        for(i = 0; i < g_m; i++)
            for(j = 0; j < g_n; j++){
                float acc = 0.0f;
                for(kk = 0; kk < g_k; kk++)
                    acc += b2f(A[i*g_k + kk]) * b2f(g_w[kk*g_n + j]);
                C[i*g_n + j] = acc;
            }
    }
    if(g_fail == F_COMPLETION){
        /* The hostile shape: the output buffer has already been written and we
         * then declare failure. The host must still not treat it as handled. */
        snprintf(g_err,sizeof g_err,"injected completion failure");
        return H_E_COMPLETION;
    }
    return H_OK;
}
#endif

API int coli_xdna_helper_release_weight(void){ g_relw++; free(g_w); g_w = NULL; g_w_src = NULL; return H_OK; }
API void coli_xdna_helper_shutdown(void){ g_shut++; g_is_open = 0; free(g_w); g_w = NULL; g_w_src = NULL; }

/* -- test controls, not part of the ABI ---------------------------------- */
API void fake_set_fail(int stage){ g_fail = stage; }
API void fake_reset(void){
    g_fail = 0; g_open = g_wrap = g_exec = g_relw = g_shut = 0;
    g_is_open = 0; g_w = NULL; g_err[0] = '\0';
}
API void fake_counts(int *open, int *wrap, int *exec, int *relw, int *shut){
    if(open) *open = g_open;  if(wrap) *wrap = g_wrap;  if(exec) *exec = g_exec;
    if(relw) *relw = g_relw;  if(shut) *shut = g_shut;
}
