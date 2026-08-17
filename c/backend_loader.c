/* backend_loader.c — Windows runtime loader for the GPU backend DLL.
 *
 * Why this exists: the engine is built with MinGW-w64 (gcc), but CUDA kernels
 * must be compiled with MSVC + nvcc. We cannot link a CUDA .o into a gcc binary
 * reliably across the MSVC/MinGW ABI, and nvcc requires cl.exe as its host
 * compiler. The clean cross-toolchain split is: build the CUDA backend into a
 * standalone coli_cuda.dll with nvcc+MSVC, then load it here at runtime via
 * LoadLibrary/GetProcAddress. The host (glm.exe) never links cudart directly.
 *
 * On Linux this file is not compiled (the Makefile links backend_cuda.o
 * directly). On Windows, when COLI_CUDA is defined, glm.c calls the
 * coli_cuda_* wrappers below, which forward through function pointers resolved
 * from the DLL at first use. If the DLL is absent, every call safely returns
 * the "not initialized" sentinel (0 / no-op) and the engine falls back to CPU.
 *
 * ABI note: ColiCudaTensor* is opaque to the host (it stores the pointer,
 * never dereferences it), so the MSVC-allocated struct is safe to pass across
 * the boundary as an opaque handle. All scalar types (int, size_t, pointers)
 * agree between MSVC and MinGW-w64 on x86-64.
 */
#ifdef _WIN32

/* FILE_ID_INFO / FileIdInfo (coli_win32_loader_probe, below) are gated behind
 * _WIN32_WINNT >= _WIN32_WINNT_WIN8 in mingw-w64's winbase.h. MSVC exposes
 * them unconditionally, so this only bites the MinGW build -- and only where
 * the file-identity block is actually compiled, i.e. under COLI_HIP_DLL or
 * COLI_LOADER_TEST_API. Set the floor before <windows.h> is pulled in. */
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0602   /* Windows 8 */
#endif

#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>
#include <windows.h>
#include <tlhelp32.h>

#include "backend_cuda.h"

/* Which backend DLL this host looks for, and how it labels its own messages.
 * The Makefile defines COLI_HIP_DLL for a HIP_DLL=1 host and leaves it undefined
 * for CUDA_DLL=1; the two builds are mutually exclusive there, so exactly one
 * arm is live. Every filename-coupled site below uses COLI_BACKEND_DLL —
 * including the path bounds check, which must derive from the same constant so
 * it can never drift from the name it is guarding.
 *
 * The EXPORTED symbol prefix stays coli_cuda_ for both vendors: one shared ABI,
 * one shared backend_cuda.cu (see GPU_BACKENDS.md). Only the container filename
 * and the diagnostic label differ. */
#ifdef COLI_HIP_DLL
#define COLI_BACKEND_DLL "coli_hip.dll"
#define COLI_BACKEND_DLL_W L"coli_hip.dll"   /* the HIP host loads it wide */
#define COLI_VENDOR_TAG  "[HIP]"
#else
#define COLI_BACKEND_DLL "coli_cuda.dll"
#define COLI_VENDOR_TAG  "[CUDA]"
#endif

/* Function-pointer typedefs matching each exported symbol. */
typedef int            (*fn_init)(const int *devices, int count);
typedef void           (*fn_shutdown)(void);
typedef int            (*fn_device_count)(void);
typedef int            (*fn_device_at)(int index);
typedef int            (*fn_mem_info)(int device, size_t *free_bytes, size_t *total_bytes);
typedef int            (*fn_device_integrated)(int device);
typedef void           (*fn_stats)(int device, size_t *tensor_count, size_t *tensor_bytes);
typedef void           (*fn_group_stats)(uint64_t *calls, uint64_t *experts, uint64_t *rows,
                                         double *h2d_ms, double *kernel_ms, double *d2h_ms);
typedef void           (*fn_group_stats_device)(int device, uint64_t *calls,
                                                uint64_t *experts, uint64_t *rows,
                                                double *h2d_ms, double *kernel_ms,
                                                double *d2h_ms);
typedef int            (*fn_expert_mlp)(ColiCudaTensor *gate, ColiCudaTensor *up,
                                        ColiCudaTensor *down, float *y, const float *x, int S);
typedef int            (*fn_expert_group)(ColiCudaTensor *const *gates, ColiCudaTensor *const *ups,
                                          ColiCudaTensor *const *downs, const int *rows, int count,
                                          float *y, const float *x);
typedef int            (*fn_expert_group_issue)(ColiCudaTensor *const *gates,
                                                ColiCudaTensor *const *ups,
                                                ColiCudaTensor *const *downs,
                                                const int *rows, int count, const float *x);
typedef const float *  (*fn_expert_group_take)(int device);
typedef int            (*fn_attention_absorb)(ColiCudaTensor *kv_b, float *ctx, const float *q,
                                              const float *latent, const float *rope, int H, int Q,
                                              int R, int V, int K, int T, float attention_scale);
typedef int            (*fn_tensor_upload)(ColiCudaTensor **tensor, const void *weights,
                                           const float *scales, int fmt, int I, int O, int device);
typedef int            (*fn_tensor_upload_g)(ColiCudaTensor **tensor, const void *weights, const float *scales, int fmt, int I, int O, int device, int gs);
typedef int            (*fn_e8_set_grid)(const void *grid);
typedef int            (*fn_fp8_set_lut)(const float *lut);
typedef int            (*fn_matmul)(ColiCudaTensor **tensor, float *y, const float *x,
                                    const void *weights, const float *scales,
                                    int fmt, int S, int I, int O, int device, int gs);
typedef void           (*fn_tensor_free)(ColiCudaTensor *tensor);
typedef size_t         (*fn_tensor_bytes)(const ColiCudaTensor *tensor);
typedef int            (*fn_tensor_device)(const ColiCudaTensor *tensor);

/* --- #111 GPU resident pipeline additions (matched to backend_cuda.h) --- */


/* --- #111 GPU resident pipeline additions (matched to backend_cuda.h) --- */
typedef int (*fn_attention_absorb_batch)(ColiCudaTensor *kv_b,float *ctx,const float *q, const float *latent,const float *rope,int S, int H,int Q,int R,int V,int K,int T, float attention_scale);
typedef int (*fn_attention_absorb_batch_dev)(ColiCudaTensor *kv_b_shard,float *ctx_dev, const float *q_dev,const float *latent_dev,const float *rope_dev, int S,int H,int Q,int R,int V,int K,int T,float scale);
typedef int (*fn_attention_absorb_kvdev)(ColiCudaTensor *kv_b,float *ctx,const float *q, const float *latent_dev,const float *rope_dev,int H,int Q,int R,int V,int K,int T, float scale);
typedef int (*fn_attention_project_batch)(ColiCudaTensor *kv_b,ColiCudaTensor *o_proj, float *out,const float *q,const float *latent, const float *rope,int S,int H,int Q,int R, int V,int K,int T,float attention_scale);
typedef int (*fn_attention_project_ragged)(ColiCudaTensor *kv_b,ColiCudaTensor *o_proj,
        float *out,const float *q,const void *const *keys,
        const float *const *latent,const float *const *rope,
        const int *lengths,int S,int H,int Q,int R,int V,int K,int max_t,float attention_scale);
typedef int (*fn_attention_project_batch_dev)(ColiCudaTensor *kv_b,ColiCudaTensor *o_proj, float *out,const float *q_dev,const float *latent_dev,const float *rope_dev, int S,int H,int Q,int R,int V,int K,int T,float scale);
typedef int (*fn_attention_project_batch_dev_out)(ColiCudaTensor *kv_b,ColiCudaTensor *o_proj, float *out_dev,const float *q_dev,const float *latent_dev,const float *rope_dev, int S,int H,int Q,int R,int V,int K,int T,float scale);
typedef int (*fn_pipe_add)(int device,float *x_dev,const float *t_dev,size_t n);
typedef void * (*fn_pipe_alloc)(int device,size_t bytes);
typedef int (*fn_pipe_copy2d)(int device,float *dst,int dpitch,const float *src, int spitch,int width,int height);
typedef int (*fn_pipe_download)(int device,const void *src,void *dst,size_t bytes);
typedef void (*fn_pipe_free)(int device,void *p);
typedef int (*fn_pipe_gemm)(ColiCudaTensor *t,float *y_dev,const float *x_dev,int S);
typedef int (*fn_pipe_peer_copy)(int dst_dev,float *dst,int src_dev, const float *src,size_t bytes);
typedef int (*fn_pipe_rmsnorm)(int device,float *y_dev,const float *x_dev, const float *w_dev,int S,int D,float eps);
typedef int (*fn_pipe_rmsnorm_s)(int device,float *y_dev,const float *x_dev, const float *w_dev,int S,int D,float eps, int xstride,int ystride);
typedef int (*fn_group_resident_issue)(ColiCudaTensor *const *gates,ColiCudaTensor *const *ups,ColiCudaTensor *const *downs,const float *weights,int count,int home_device,const float *x_src_dev,float *partial_slot_dev);
typedef int (*fn_group_resident_take)(int home_device,const int *devices,int n_issued,float *slots_dev,float *acc_dev,int D);
typedef int (*fn_pipe_router)(int device,const float *x_dev,const void *rw_dev,const void *rb_dev,int D,int E,int Ksel,float topp,int norm_topk,float routed_scale,int *idx_host,float *w_host,int *keff_host);
typedef int (*fn_pipe_rope)(int device,float *v_dev,const int *pos_dev,int rows, int stride,int offset,int R,int heads,float theta);
typedef int (*fn_pipe_rope_base)(int device,float *v_dev,int pos_base,int rows, int stride,int offset,int R,int heads,float theta);
typedef int (*fn_pipe_rows_add)(int device,float *x_dev,const float *partial_dev, const int *rows_dev,int nrows,int D);
typedef float * (*fn_pipe_scratch)(int device,int slot,size_t bytes);
typedef int (*fn_pipe_silu_mul)(int device,float *gate_dev,const float *up_dev,size_t n);
typedef int (*fn_pipe_sync)(int device);
typedef int (*fn_pipe_upload)(int device,void *dst,const void *src,size_t bytes);
typedef int (*fn_shared_mlp_w4a16)(ColiCudaTensor *gate, ColiCudaTensor *up, ColiCudaTensor *down, float *y, const float *x, int S);
typedef int (*fn_tensor_update)(ColiCudaTensor *tensor, const void *weights, const float *scales);

/* Resolved pointers, plus a flag so we attempt the load at most once. */
static struct {
    int loaded;        /* 1 = load attempted (success or fail), 0 = not yet */
    int available;     /* 1 = DLL loaded and all symbols resolved */
    HMODULE dll;
#ifdef COLI_HIP_DLL
    /* The one explicit reference this loader holds on the configured
     * amdhip64_7.dll. Non-NULL IS the ownership flag — it is set only after
     * everything below it succeeded, and released only in shutdown. The
     * backend's own import table also references the runtime, but that
     * reference belongs to the backend, not to us. */
    HMODULE hip_runtime;
#endif
    fn_init            init;
    fn_shutdown        shutdown;
    fn_device_count    device_count;
    fn_device_at       device_at;
    fn_mem_info        mem_info;
    fn_device_integrated device_integrated;
    fn_stats           stats;
    fn_group_stats     group_stats;
    fn_group_stats_device group_stats_device;
    fn_expert_mlp      expert_mlp;
    fn_expert_group    expert_group;
    fn_expert_group_issue expert_group_issue;
    fn_expert_group_take expert_group_take;
    fn_attention_absorb attention_absorb;
    fn_tensor_upload   tensor_upload;
    fn_tensor_upload_g tensor_upload_g;
    fn_e8_set_grid     e8_set_grid;
    fn_fp8_set_lut     fp8_set_lut;
    fn_matmul          matmul;
    fn_tensor_free     tensor_free;
    fn_tensor_bytes    tensor_bytes;
    fn_tensor_device   tensor_device;

    fn_attention_absorb_batch attention_absorb_batch;
    fn_attention_absorb_batch_dev attention_absorb_batch_dev;
    fn_attention_absorb_kvdev attention_absorb_kvdev;
    fn_attention_project_batch attention_project_batch;
    fn_attention_project_ragged attention_project_ragged;
    fn_attention_project_batch_dev attention_project_batch_dev;
    fn_attention_project_batch_dev_out attention_project_batch_dev_out;
    fn_pipe_add pipe_add;
    fn_pipe_alloc pipe_alloc;
    fn_pipe_copy2d pipe_copy2d;
    fn_pipe_download pipe_download;
    fn_pipe_free pipe_free;
    fn_pipe_gemm pipe_gemm;
    fn_pipe_peer_copy pipe_peer_copy;
    fn_pipe_rmsnorm pipe_rmsnorm;
    fn_pipe_rmsnorm_s pipe_rmsnorm_s;
    fn_group_resident_issue expert_group_resident_issue;
    fn_group_resident_take expert_group_resident_take;
    fn_pipe_router pipe_router;
    fn_pipe_rope pipe_rope;
    fn_pipe_rope_base pipe_rope_base;
    fn_pipe_rows_add pipe_rows_add;
    fn_pipe_scratch pipe_scratch;
    fn_pipe_silu_mul pipe_silu_mul;
    fn_pipe_sync pipe_sync;
    fn_pipe_upload pipe_upload;
    fn_shared_mlp_w4a16 shared_mlp_w4a16;
    fn_tensor_update tensor_update;
} g_cuda;

#ifdef COLI_HIP_DLL
/* ---- COLI_HIP_RUNTIME_DIR: the HIP runtime-directory contract ----
 *
 * A HIP host must be told, explicitly and unambiguously, which ROCm/HIP
 * runtime it is meant to use. This machine class routinely carries more than
 * one amdhip64_7.dll (a system-wide ROCm install plus whatever the user
 * unpacked), and Windows resolves an import by BASE NAME against whatever is
 * already loaded — so "whichever one the search order happens to find" is not
 * a contract, it is a coin flip with a native crash on the losing side.
 *
 * This slice validates the configuration only. It reads the variable, checks
 * that it names a real directory holding a file called amdhip64_7.dll, and
 * then hands over to the existing backend load unchanged. Nothing is loaded,
 * preloaded or pinned here, and no state survives the call: making the load
 * itself deterministic is a separate, self-contained change.
 *
 * Placement matters. The check lives inside coli_cuda_load(), the one-shot
 * loader entry, so it fires exactly when the GPU tier is actually requested.
 * A host that never asks for the GPU never reaches this code and therefore
 * never needs the variable. Under a CUDA_DLL build the whole block is
 * compiled out, so COLI_HIP_RUNTIME_DIR cannot influence CUDA behaviour even
 * when it is set to nonsense.
 *
 * Deliberately not done: no %VAR% expansion (the value is taken literally, so
 * a path containing a percent sign is not silently rewritten), no fallback to
 * the current directory, no PATH search, no process-environment mutation, and
 * no inspection of the runtime file beyond "it exists and is a file". */

#define COLI_HIP_RUNTIME_DIR_VAR L"COLI_HIP_RUNTIME_DIR"
#define COLI_HIP_RUNTIME_FILE    L"amdhip64_7.dll"

static int coli_hip_is_sep(wchar_t c){ return c == L'\\' || c == L'/'; }

/* Absolute in the Windows sense, using only the C runtime — pulling in
 * Shlwapi for PathIsRelativeW would add a link dependency to the host for one
 * predicate. Accepts a drive-absolute path (C:\x, C:/x) and anything rooted at
 * a double separator (\\server\share, \\?\C:\x, //server/share). Everything
 * else is rejected, including the two forms that LOOK rooted but are not:
 * "C:runtime" is relative to the drive's own current directory, and "\runtime"
 * is relative to the current drive. Resolving either against the CWD is
 * precisely the ambiguity this contract exists to remove. */
static int coli_hip_path_is_absolute(const wchar_t *p){
    if(!p || !p[0]) return 0;
    if(coli_hip_is_sep(p[0]) && coli_hip_is_sep(p[1])) return 1;
    if(((p[0] >= L'A' && p[0] <= L'Z') || (p[0] >= L'a' && p[0] <= L'z')) &&
       p[1] == L':' && coli_hip_is_sep(p[2])) return 1;
    return 0;
}

/* stderr is a narrow (byte-oriented) stream here, and mixing wide output into
 * it is undefined, so paths are converted rather than printed with %ls. */
static char *coli_hip_utf8(const wchar_t *w){
    char *s;
    int n = WideCharToMultiByte(CP_UTF8, 0, w, -1, NULL, 0, NULL, NULL);
    if(n <= 0) return NULL;
    s = (char *)malloc((size_t)n);
    if(!s) return NULL;
    if(WideCharToMultiByte(CP_UTF8, 0, w, -1, s, n, NULL, NULL) <= 0){
        free(s);
        return NULL;
    }
    return s;
}

/* One shape for every rejection: what was wrong, the offending value when
 * there is one, and the same "CPU path remains active" reassurance the
 * backend miss already prints. It never claims a DLL was loaded or tried. */
static void coli_hip_reject(const char *what, const wchar_t *value){
    char *utf8 = value ? coli_hip_utf8(value) : NULL;
    fprintf(stderr, COLI_VENDOR_TAG " %s%s%s; GPU tier disabled "
                    "(CPU path remains active).\n",
            what,
            value ? ": " : "",
            utf8 ? utf8 : (value ? "<unprintable path>" : ""));
    free(utf8);
}

/* Read COLI_HIP_RUNTIME_DIR into a fresh allocation.
 * Returns 1 with *out owned by the caller (possibly the empty string), 0 when
 * the variable is not set at all, and -1 on retrieval or allocation failure
 * with *err carrying the reason. The buffer is sized from the API rather than
 * assumed, so a value longer than MAX_PATH is read in full instead of being
 * truncated into a different path. */
static int coli_hip_env_dup(wchar_t **out, DWORD *err){
    DWORD cap = MAX_PATH;
    int attempt;

    *out = NULL;
    *err = 0;
    for(attempt = 0; attempt < 8; attempt++){
        DWORD n;
        wchar_t *buf = (wchar_t *)malloc((size_t)cap * sizeof(wchar_t));
        if(!buf){ *err = ERROR_NOT_ENOUGH_MEMORY; return -1; }
        /* Cleared first: a set-but-empty value returns 0 with the error code
         * untouched, which is the only way to tell it from "not set". */
        SetLastError(ERROR_SUCCESS);
        n = GetEnvironmentVariableW(COLI_HIP_RUNTIME_DIR_VAR, buf, cap);
        if(n == 0){
            DWORD e = GetLastError();
            if(e == ERROR_ENVVAR_NOT_FOUND){ free(buf); return 0; }
            if(e == ERROR_SUCCESS){ buf[0] = L'\0'; *out = buf; return 1; }
            free(buf);
            *err = e;
            return -1;
        }
        if(n < cap){ *out = buf; return 1; }
        free(buf);
        cap = n;   /* n is now the required size, terminator included */
    }
    /* Only reachable if the value keeps growing between calls. */
    *err = ERROR_MORE_DATA;
    return -1;
}

/* The validated directory, owned by the caller, or NULL after a diagnostic
 * naming the specific problem. Only the DIRECTORY is settled here; the runtime
 * file beneath it is joined, classified and identified in one step further
 * down, once the identity helpers are in scope. */
static wchar_t *coli_hip_runtime_dir_value(void){
    wchar_t *dir = NULL;
    DWORD err = 0, attrs;
    size_t len;
    int rc;

    rc = coli_hip_env_dup(&dir, &err);
    if(rc == 0){
        fprintf(stderr, COLI_VENDOR_TAG " COLI_HIP_RUNTIME_DIR is not set; "
                        "set it to the directory containing amdhip64_7.dll. "
                        "GPU tier disabled (CPU path remains active).\n");
        return NULL;
    }
    if(rc < 0){
        fprintf(stderr, COLI_VENDOR_TAG " could not read COLI_HIP_RUNTIME_DIR "
                        "(error %lu); GPU tier disabled "
                        "(CPU path remains active).\n", (unsigned long)err);
        return NULL;
    }

    len = wcslen(dir);
    if(len == 0){
        coli_hip_reject("COLI_HIP_RUNTIME_DIR is empty", NULL);
        goto fail;
    }
    if(!coli_hip_path_is_absolute(dir)){
        coli_hip_reject("COLI_HIP_RUNTIME_DIR must be an absolute path", dir);
        goto fail;
    }

    attrs = GetFileAttributesW(dir);
    if(attrs == INVALID_FILE_ATTRIBUTES){
        coli_hip_reject("COLI_HIP_RUNTIME_DIR does not exist", dir);
        goto fail;
    }
    if(!(attrs & FILE_ATTRIBUTE_DIRECTORY)){
        coli_hip_reject("COLI_HIP_RUNTIME_DIR is not a directory", dir);
        goto fail;
    }
    return dir;

fail:
    free(dir);
    return NULL;
}
#endif /* COLI_HIP_DLL */

/* ---- Truthful reporting of a failed backend load ----
 *
 * Both LoadLibraryEx attempts below can fail for reasons that have nothing to
 * do with the file being absent: a 32-bit DLL on a 64-bit host, a dependency
 * that cannot be resolved, a denied ACL. Reporting all of those as "not found"
 * sends the reader looking for a missing file that is sitting right there.
 *
 * These helpers are vendor-neutral on purpose — a CUDA host is misled by the
 * same message — so they live outside the COLI_HIP_DLL block.
 *
 * No FormatMessage: its text is localised, which would make any assertion on
 * it machine-dependent, and it needs an allocation to be used safely. A short
 * project-owned table is smaller, stable, and always carries the numeric code
 * so an unrecognised value is still actionable. */
static const char *coli_win_loader_error_text(DWORD code){
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

/* Renders one error into *buf*. Recognised codes gain a short explanation;
 * everything else still reports its number rather than being swallowed. */
static void coli_win_error_phrase(DWORD code, char *buf, size_t n){
    const char *text = coli_win_loader_error_text(code);
    if(text) snprintf(buf, n, "Windows error %lu (%s)", (unsigned long)code, text);
    else     snprintf(buf, n, "Windows loader error %lu", (unsigned long)code);
}

/* ===================== Windows runtime identity and inventory =============
 *
 * Foundation for deterministic HIP runtime binding. NOTHING HERE IS CALLED BY
 * THE LOADER YET: this slice adds the machinery and proves it, and a later one
 * wires it in. Keeping the two apart means the change that alters what the
 * process loads can be reviewed — and reverted — on its own.
 *
 * Three measured Windows facts shape all of it:
 *
 *   1. Two modules with the same base name, loaded from different absolute
 *      paths, really do coexist in one process, with distinct HMODULEs.
 *   2. GetModuleHandleExW asked for a BASE NAME returns only the first of them
 *      and silently hides the rest, so it cannot answer "what is loaded".
 *   3. Two different path strings can name one physical file (hard links, dot
 *      segments, case), and one path string can be re-pointed at a different
 *      file by a rename even while the old one is mapped.
 *
 * So: enumerate every match rather than ask for one, and decide equality on
 * physical file identity rather than on path text. Paths are kept only to tell
 * a human which file we meant.
 *
 * These functions are file-static and appear in no header: they live in
 * backend_loader.o, which links into the host executable only — coli_hip.dll
 * and coli_cuda.dll are built from backend_cuda.cu and never see this file.
 * The whole block is compiled only where something consumes it, so a CUDA host
 * carries none of it. */
#if defined(COLI_HIP_DLL) || defined(COLI_LOADER_TEST_API)

#define COLI_W32_RUNTIME_BASENAME L"amdhip64_7.dll"
#define COLI_W32_PATH_START  512u   /* wchar_t; grows, never a correctness bound */
#define COLI_W32_PATH_ROUNDS 8      /* 512 -> 65536 wchar_t before giving up */

static int coli_win32_loader_is_sep(wchar_t c){ return c == L'\\' || c == L'/'; }

/* --- physical file identity ------------------------------------------
 *
 * Two representations, because the newer one is not always obtainable.
 * FILE_ID_INFO carries a 128-bit ID that stays unique on ReFS;
 * BY_HANDLE_FILE_INFORMATION carries the classic 64-bit index. Both are
 * captured when available so a later comparison can pick a format the two
 * sides genuinely share — see coli_win32_loader_identity_equal. */
typedef struct {
    int         have_id_info;
    int         have_by_handle;
    DWORD       error;            /* why nothing could be captured */
    ULONGLONG   idi_volume;
    FILE_ID_128 idi_id;
    DWORD       bh_volume;
    DWORD       bh_index_high;
    DWORD       bh_index_low;
} ColiW32FileId;

static int coli_win32_loader_identity_valid(const ColiW32FileId *id){
    return id && (id->have_id_info || id->have_by_handle);
}

/* 1 = the same physical file, 0 = different files, -1 = cannot be determined.
 *
 * The fallback is SYMMETRIC: if either side is missing FILE_ID_INFO, BOTH
 * sides drop to the by-handle index. A 128-bit ID is never compared against a
 * 64-bit index, because they are different namespaces and a coincidental match
 * would be indistinguishable from a real one. When no shared format exists the
 * answer is "unknown" — never a text comparison, which would reject hard links
 * to the very file that was configured. */
static int coli_win32_loader_identity_equal(const ColiW32FileId *a, const ColiW32FileId *b){
    if(!a || !b) return -1;
    if(a->have_id_info && b->have_id_info)
        return (a->idi_volume == b->idi_volume &&
                memcmp(&a->idi_id, &b->idi_id, sizeof a->idi_id) == 0) ? 1 : 0;
    if(a->have_by_handle && b->have_by_handle)
        return (a->bh_volume     == b->bh_volume &&
                a->bh_index_high == b->bh_index_high &&
                a->bh_index_low  == b->bh_index_low) ? 1 : 0;
    return -1;
}

/* --- growing-buffer Unicode paths ------------------------------------
 *
 * Every one of these returns a malloc'd, NUL-terminated wide string the caller
 * frees, or NULL with *err set. None of them treats MAX_PATH as a limit. */

static int coli_w32_grow_ok(size_t cap){
    return cap <= ((size_t)-1) / sizeof(wchar_t) / 2;
}

/* Path of a loaded module, or of the running executable when mod is NULL.
 * GetModuleFileNameW signals truncation by returning the buffer size AND
 * setting ERROR_INSUFFICIENT_BUFFER; the length alone cannot be trusted,
 * which is why the error code is cleared first and then consulted. */
static wchar_t *coli_win32_loader_module_path(HMODULE mod, DWORD *err){
    size_t cap = COLI_W32_PATH_START;
    int round;
    if(err) *err = 0;
    for(round = 0; round < COLI_W32_PATH_ROUNDS; round++){
        wchar_t *buf;
        DWORD n;
        if(!coli_w32_grow_ok(cap)){ if(err) *err = ERROR_ARITHMETIC_OVERFLOW; return NULL; }
        buf = (wchar_t *)malloc(cap * sizeof(wchar_t));
        if(!buf){ if(err) *err = ERROR_NOT_ENOUGH_MEMORY; return NULL; }
        SetLastError(ERROR_SUCCESS);
        n = GetModuleFileNameW(mod, buf, (DWORD)cap);
        if(n == 0){
            DWORD e = GetLastError();
            free(buf);
            if(err) *err = e;
            return NULL;
        }
        if((size_t)n < cap && GetLastError() != ERROR_INSUFFICIENT_BUFFER){
            buf[n] = L'\0';
            return buf;
        }
        free(buf);
        cap *= 2;
    }
    if(err) *err = ERROR_INSUFFICIENT_BUFFER;
    return NULL;
}

static wchar_t *coli_win32_loader_exe_path(DWORD *err){
    return coli_win32_loader_module_path(NULL, err);
}

/* Canonical path behind an open handle, for DIAGNOSTICS ONLY. It resolves
 * case, dot segments, symlinks and junctions — but NOT hard links, which keep
 * their own names. That is exactly why equality is decided on file identity
 * and this string is only ever shown to a human. The VOLUME_NAME_DOS form is
 * returned as Windows gives it, including any \\?\ prefix: stripping that
 * prefix can change how a path is interpreted, so it is left intact. */
static wchar_t *coli_win32_loader_final_path(HANDLE handle, DWORD *err){
    size_t cap = COLI_W32_PATH_START;
    int round;
    if(err) *err = 0;
    for(round = 0; round < COLI_W32_PATH_ROUNDS; round++){
        wchar_t *buf;
        DWORD n;
        if(!coli_w32_grow_ok(cap)){ if(err) *err = ERROR_ARITHMETIC_OVERFLOW; return NULL; }
        buf = (wchar_t *)malloc(cap * sizeof(wchar_t));
        if(!buf){ if(err) *err = ERROR_NOT_ENOUGH_MEMORY; return NULL; }
        SetLastError(ERROR_SUCCESS);
        n = GetFinalPathNameByHandleW(handle, buf, (DWORD)cap,
                                      FILE_NAME_NORMALIZED | VOLUME_NAME_DOS);
        if(n == 0){
            DWORD e = GetLastError();
            free(buf);
            if(err) *err = e;
            return NULL;
        }
        if((size_t)n < cap){ buf[n] = L'\0'; return buf; }
        free(buf);
        /* n is the required size INCLUDING the terminator when it did not fit. */
        cap = (size_t)n + 1;
    }
    if(err) *err = ERROR_INSUFFICIENT_BUFFER;
    return NULL;
}

/* <dir> + one separator + <child>. Purely textual: nothing is resolved,
 * expanded or searched for. A value already ending in a separator does not
 * gain a second one, because "C:\x\\y" reads as a UNC root. */
static wchar_t *coli_win32_loader_join(const wchar_t *dir, const wchar_t *child, DWORD *err){
    size_t dl, cl, need;
    int sep;
    wchar_t *out;
    if(err) *err = 0;
    if(!dir || !child || !dir[0]){ if(err) *err = ERROR_INVALID_PARAMETER; return NULL; }
    dl = wcslen(dir);
    cl = wcslen(child);
    sep = !coli_win32_loader_is_sep(dir[dl - 1]);
    if(cl > ((size_t)-1) - dl - 2){ if(err) *err = ERROR_ARITHMETIC_OVERFLOW; return NULL; }
    need = dl + (size_t)(sep ? 1 : 0) + cl + 1;
    if(need > ((size_t)-1) / sizeof(wchar_t)){ if(err) *err = ERROR_ARITHMETIC_OVERFLOW; return NULL; }
    out = (wchar_t *)malloc(need * sizeof(wchar_t));
    if(!out){ if(err) *err = ERROR_NOT_ENOUGH_MEMORY; return NULL; }
    wcscpy(out, dir);
    if(sep) wcscat(out, L"\\");
    wcscat(out, child);
    return out;
}

/* One open, both answers. Zero requested access means "query metadata": it
 * needs no read permission and cannot disturb a mapped module. Full sharing
 * means the probe never blocks anyone else, and FILE_FLAG_BACKUP_SEMANTICS
 * lets the same call work if the target turns out to be a directory. */
static int coli_win32_loader_probe(const wchar_t *path, ColiW32FileId *id,
                            wchar_t **final_path, DWORD *err){
    HANDLE h;
    FILE_ID_INFO fii;
    BY_HANDLE_FILE_INFORMATION bhi;
    if(final_path) *final_path = NULL;
    if(err) *err = 0;
    if(!id) return 0;
    memset(id, 0, sizeof *id);
    if(!path){ id->error = ERROR_INVALID_PARAMETER; if(err) *err = id->error; return 0; }

    h = CreateFileW(path, 0,
                    FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                    NULL, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, NULL);
    if(h == INVALID_HANDLE_VALUE){
        id->error = GetLastError();
        if(err) *err = id->error;
        return 0;
    }
    if(GetFileInformationByHandleEx(h, FileIdInfo, &fii, (DWORD)sizeof fii)){
        id->have_id_info = 1;
        id->idi_volume   = fii.VolumeSerialNumber;
        id->idi_id       = fii.FileId;
    }
    if(GetFileInformationByHandle(h, &bhi)){
        id->have_by_handle  = 1;
        id->bh_volume       = bhi.dwVolumeSerialNumber;
        id->bh_index_high   = bhi.nFileIndexHigh;
        id->bh_index_low    = bhi.nFileIndexLow;
    }
    if(!coli_win32_loader_identity_valid(id)){
        id->error = GetLastError();
        if(err) *err = id->error;
        CloseHandle(h);
        return 0;
    }
    if(final_path) *final_path = coli_win32_loader_final_path(h, NULL);
    CloseHandle(h);
    return 1;
}

/* --- inventory of every loaded module with a given base name ---------- */

enum {
    COLI_W32_INV_OK = 0,
    COLI_W32_INV_SNAPSHOT_FAILED,
    COLI_W32_INV_ENUM_FAILED,
    COLI_W32_INV_ALLOC_FAILED,
    COLI_W32_INV_PATH_FAILED,
    COLI_W32_INV_IDENTITY_FAILED
};

typedef struct {
    HMODULE       module;
    wchar_t      *path;        /* authoritative, from GetModuleFileNameW */
    wchar_t      *final_path;  /* canonical, diagnostics only, may be NULL */
    wchar_t      *entry_path;  /* MODULEENTRY32W.szExePath, MAX_PATH, fallback */
    ColiW32FileId id;
} ColiW32Module;

typedef struct {
    int            status;
    DWORD          error;
    size_t         count;
    ColiW32Module *items;
} ColiW32Inventory;

static void coli_win32_loader_inventory_free(ColiW32Inventory *inv){
    size_t i;
    if(!inv) return;
    for(i = 0; i < inv->count; i++){
        free(inv->items[i].path);
        free(inv->items[i].final_path);
        free(inv->items[i].entry_path);
    }
    free(inv->items);
    inv->items = NULL;
    inv->count = 0;
}

static wchar_t *coli_w32_wdup(const wchar_t *s){
    size_t n;
    wchar_t *o;
    if(!s) return NULL;
    n = wcslen(s) + 1;
    o = (wchar_t *)malloc(n * sizeof(wchar_t));
    if(o) memcpy(o, s, n * sizeof(wchar_t));
    return o;
}

/* Enumerate, do not ask. A failed or partial enumeration is reported as such:
 * it must never be mistaken for "nothing is loaded", because the two lead to
 * opposite decisions. Reference counts are untouched — the HMODULEs recorded
 * here are observations, not owned references. */
static int coli_win32_loader_inventory(const wchar_t *basename, ColiW32Inventory *out){
    int attempt;
    if(!out) return COLI_W32_INV_ALLOC_FAILED;
    memset(out, 0, sizeof *out);
    if(!basename){ out->status = COLI_W32_INV_ENUM_FAILED; out->error = ERROR_INVALID_PARAMETER; return out->status; }

    for(attempt = 0; attempt < 8; attempt++){
        HANDLE snap;
        MODULEENTRY32W entry;
        BOOL more;

        snap = CreateToolhelp32Snapshot(TH32CS_SNAPMODULE | TH32CS_SNAPMODULE32,
                                        GetCurrentProcessId());
        if(snap == INVALID_HANDLE_VALUE){
            DWORD e = GetLastError();
            /* The module list changed underneath the snapshot; that is normal
             * and transient, so retry rather than report a failure. */
            if(e == ERROR_BAD_LENGTH) continue;
            out->status = COLI_W32_INV_SNAPSHOT_FAILED;
            out->error  = e;
            return out->status;
        }

        memset(&entry, 0, sizeof entry);
        entry.dwSize = sizeof entry;
        more = Module32FirstW(snap, &entry);
        if(!more){
            DWORD e = GetLastError();
            CloseHandle(snap);
            if(e == ERROR_BAD_LENGTH) continue;
            out->status = COLI_W32_INV_ENUM_FAILED;
            out->error  = e;
            return out->status;
        }
        do {
            ColiW32Module *grown, item;
            DWORD err = 0;
            if(_wcsicmp(entry.szModule, basename) != 0) continue;

            memset(&item, 0, sizeof item);
            item.module     = entry.hModule;
            item.entry_path = coli_w32_wdup(entry.szExePath);
            item.path       = coli_win32_loader_module_path(entry.hModule, &err);
            if(!item.path){
                free(item.entry_path);
                CloseHandle(snap);
                coli_win32_loader_inventory_free(out);
                out->status = COLI_W32_INV_PATH_FAILED;
                out->error  = err;
                return out->status;
            }
            if(!coli_win32_loader_probe(item.path, &item.id, &item.final_path, &err)){
                free(item.entry_path);
                free(item.path);
                CloseHandle(snap);
                coli_win32_loader_inventory_free(out);
                out->status = COLI_W32_INV_IDENTITY_FAILED;
                out->error  = err;
                return out->status;
            }
            grown = (ColiW32Module *)realloc(out->items,
                                             (out->count + 1) * sizeof *out->items);
            if(!grown){
                free(item.entry_path);
                free(item.path);
                free(item.final_path);
                CloseHandle(snap);
                coli_win32_loader_inventory_free(out);
                out->status = COLI_W32_INV_ALLOC_FAILED;
                out->error  = ERROR_NOT_ENOUGH_MEMORY;
                return out->status;
            }
            out->items = grown;
            out->items[out->count++] = item;
        } while(Module32NextW(snap, &entry));

        CloseHandle(snap);
        out->status = COLI_W32_INV_OK;
        out->error  = 0;
        return out->status;
    }
    out->status = COLI_W32_INV_SNAPSHOT_FAILED;
    out->error  = ERROR_BAD_LENGTH;
    return out->status;
}

/* --- the decision ----------------------------------------------------
 *
 * Deliberately pure: no Windows call, no allocation, no global state, nothing
 * loaded. It takes the comparison results rather than the identities so that
 * every branch — including the ones the filesystem will not produce on demand
 * — can be driven directly from a test. */
enum {
    COLI_BIND_NEED_EXACT_LOAD = 0,
    COLI_BIND_REUSE_EXACT_MATCH,
    COLI_BIND_REJECT_WRONG_RUNTIME,
    COLI_BIND_REJECT_MULTIPLE_RUNTIMES,
    COLI_BIND_REJECT_UNVERIFIABLE
};

static int coli_win32_loader_decide(int configured_valid, int inventory_status,
                             size_t count, const int *match_equality){
    if(!configured_valid)                    return COLI_BIND_REJECT_UNVERIFIABLE;
    if(inventory_status != COLI_W32_INV_OK)  return COLI_BIND_REJECT_UNVERIFIABLE;
    if(count == 0)                           return COLI_BIND_NEED_EXACT_LOAD;
    /* More than one same-basename runtime is refused even when one of them is
     * the configured file: which one an import binds to is decided by load
     * order, not by us, so the situation is not something to pick a winner in. */
    if(count > 1)                            return COLI_BIND_REJECT_MULTIPLE_RUNTIMES;
    if(!match_equality)                      return COLI_BIND_REJECT_UNVERIFIABLE;
    if(match_equality[0] == 1)               return COLI_BIND_REUSE_EXACT_MATCH;
    if(match_equality[0] == 0)               return COLI_BIND_REJECT_WRONG_RUNTIME;
    return COLI_BIND_REJECT_UNVERIFIABLE;    /* identity unknown -> refuse */
}

/* --- the configured runtime, as a short-lived fact -------------------
 *
 * Only what this slice can honestly establish: where the runtime is, what file
 * that is, and how to name it in a message. No module handle, no ownership
 * flag, no verification state — those become real when something is actually
 * loaded, and inventing them now would mean carrying fields that no code can
 * yet set correctly. */
typedef struct {
    wchar_t      *path;
    wchar_t      *final_path;
    ColiW32FileId id;
    int           valid;
    DWORD         error;
} ColiW32ConfiguredRuntime;

/* Idempotent, and safe on a zero-initialised object. */
static void coli_win32_loader_configured_free(ColiW32ConfiguredRuntime *cr){
    if(!cr) return;
    free(cr->path);
    free(cr->final_path);
    cr->path = NULL;
    cr->final_path = NULL;
    cr->valid = 0;
}

static int coli_win32_loader_configured(const wchar_t *dir, ColiW32ConfiguredRuntime *out){
    DWORD err = 0;
    if(!out) return 0;
    memset(out, 0, sizeof *out);
    out->path = coli_win32_loader_join(dir, COLI_W32_RUNTIME_BASENAME, &err);
    if(!out->path){ out->error = err; return 0; }
    if(!coli_win32_loader_probe(out->path, &out->id, &out->final_path, &err)){
        out->error = err;
        return 0;   /* path retained for diagnostics; free() is the caller's */
    }
    out->valid = 1;
    return 1;
}

#ifdef COLI_LOADER_TEST_API
/* ---- Test-only surface (never compiled into the shipped host) --------
 *
 * The loader tests run the REAL functions above by compiling this file into a
 * native harness with COLI_LOADER_TEST_API defined. Only that build has these
 * symbols; the normal host build contains none of them.
 *
 * Everything is passed by value or through caller-owned buffers on purpose:
 * no internal pointer, no struct layout and no mutable global escapes, so the
 * test surface cannot become a back door into loader state. */

static int coli_w32_test_utf8(const wchar_t *w, char *out, size_t cap){
    if(!out || cap == 0) return 0;
    out[0] = '\0';
    if(!w) return 0;
    return WideCharToMultiByte(CP_UTF8, 0, w, -1, out, (int)cap, NULL, NULL) > 0;
}

/* 1 same file, 0 different, -1 undeterminable — the production contract. */
int coli_loader_test_same_file(const wchar_t *a, const wchar_t *b){
    ColiW32FileId ia, ib;
    coli_win32_loader_probe(a, &ia, NULL, NULL);
    coli_win32_loader_probe(b, &ib, NULL, NULL);
    return coli_win32_loader_identity_equal(&ia, &ib);
}

/* Drives the symmetric fallback directly: the caller says which
 * representations each side has, so the mixed case can be exercised without
 * needing a filesystem on which one API fails. */
int coli_loader_test_identity_equal_synthetic(int a_idi, int a_bh, int b_idi, int b_bh,
                                              int same_idi, int same_bh){
    ColiW32FileId a, b;
    memset(&a, 0, sizeof a);
    memset(&b, 0, sizeof b);
    a.have_id_info = a_idi; a.have_by_handle = a_bh;
    b.have_id_info = b_idi; b.have_by_handle = b_bh;
    a.idi_volume = 1; b.idi_volume = same_idi ? 1 : 2;
    a.bh_volume  = 1; b.bh_volume  = same_bh  ? 1 : 2;
    return coli_win32_loader_identity_equal(&a, &b);
}

int coli_loader_test_final_path(const wchar_t *path, char *out, size_t cap){
    ColiW32FileId id;
    wchar_t *final_path = NULL;
    int ok;
    if(!coli_win32_loader_probe(path, &id, &final_path, NULL)) return 0;
    ok = coli_w32_test_utf8(final_path, out, cap);
    free(final_path);
    return ok;
}

int coli_loader_test_join(const wchar_t *dir, const wchar_t *child, char *out, size_t cap){
    DWORD err = 0;
    wchar_t *joined = coli_win32_loader_join(dir, child, &err);
    int ok;
    if(!joined) return 0;
    ok = coli_w32_test_utf8(joined, out, cap);
    free(joined);
    return ok;
}

int coli_loader_test_exe_path(char *out, size_t cap){
    DWORD err = 0;
    wchar_t *p = coli_win32_loader_exe_path(&err);
    int ok;
    if(!p) return 0;
    ok = coli_w32_test_utf8(p, out, cap);
    free(p);
    return ok;
}

int coli_loader_test_decide(int configured_valid, int inventory_status,
                            int count, const int *match_equality){
    return coli_win32_loader_decide(configured_valid, inventory_status,
                                    (size_t)count, match_equality);
}

int coli_loader_test_configured(const wchar_t *dir, char *path_out, size_t path_cap,
                                char *final_out, size_t final_cap, unsigned long *error){
    ColiW32ConfiguredRuntime cr;
    int valid;
    int ok = coli_win32_loader_configured(dir, &cr);
    coli_w32_test_utf8(cr.path, path_out, path_cap);
    coli_w32_test_utf8(cr.final_path, final_out, final_cap);
    if(error) *error = cr.error;
    valid = cr.valid;
    coli_win32_loader_configured_free(&cr);
    coli_win32_loader_configured_free(&cr);   /* proves the free is idempotent */
    (void)ok;
    return valid;
}

/* One inventory pass, flattened into caller-owned slots. *count is the true
 * match count even when it exceeds max_items, so a test can never mistake a
 * truncated report for a smaller inventory. */
int coli_loader_test_inventory(const wchar_t *configured_path,
                               int *status, unsigned long *error, int *count,
                               int max_items, int slot,
                               char *paths, char *finals,
                               unsigned long long *modules, int *equal_to_configured){
    ColiW32Inventory inv;
    ColiW32FileId configured;
    int have_configured = 0;
    size_t i;

    memset(&configured, 0, sizeof configured);
    if(configured_path)
        have_configured = coli_win32_loader_probe(configured_path, &configured, NULL, NULL);

    coli_win32_loader_inventory(COLI_W32_RUNTIME_BASENAME, &inv);
    if(status) *status = inv.status;
    if(error)  *error  = inv.error;
    if(count)  *count  = (int)inv.count;

    for(i = 0; i < inv.count && (int)i < max_items; i++){
        if(paths)  coli_w32_test_utf8(inv.items[i].path,       paths  + (size_t)slot * i, (size_t)slot);
        if(finals) coli_w32_test_utf8(inv.items[i].final_path, finals + (size_t)slot * i, (size_t)slot);
        if(modules) modules[i] = (unsigned long long)(uintptr_t)inv.items[i].module;
        if(equal_to_configured)
            equal_to_configured[i] = have_configured
                ? coli_win32_loader_identity_equal(&configured, &inv.items[i].id)
                : -1;
    }
    coli_win32_loader_inventory_free(&inv);
    return inv.status;
}
#endif /* COLI_LOADER_TEST_API */
#endif /* COLI_HIP_DLL || COLI_LOADER_TEST_API */

#ifdef COLI_HIP_DLL
/* ============ Fail-closed HIP runtime binding =============================
 *
 * A HIP host may only reach GPU work if it can show that the amdhip64_7.dll
 * bound to coli_hip.dll is exactly the one COLI_HIP_RUNTIME_DIR names. Anything
 * it cannot show — a different runtime already loaded, two of them, an
 * enumeration that did not complete — disables the GPU tier instead of
 * guessing. The CPU path stays available throughout.
 *
 * Honest limit: the runtime is verified AFTER it is mapped. LoadLibraryExW
 * takes a path, not an open handle, so a rename between identifying the file
 * and loading it cannot be prevented — only detected, which the post-load
 * inventory does. Nothing here is atomic and nothing claims to be. */

/* Everything the loader learned about the configured runtime, valid for the
 * length of one initialisation attempt. */
static int coli_hip_configure(ColiW32ConfiguredRuntime *out){
    wchar_t *dir;
    DWORD attrs;
    int ok;

    memset(out, 0, sizeof *out);
    dir = coli_hip_runtime_dir_value();     /* diagnoses its own failures */
    if(!dir) return 0;

    /* One join, then classify what it landed on. The path survives a failed
     * probe precisely so these messages can name it. */
    ok = coli_win32_loader_configured(dir, out);
    free(dir);
    if(!out->path){
        fprintf(stderr, COLI_VENDOR_TAG " could not build the amdhip64_7.dll "
                        "path under COLI_HIP_RUNTIME_DIR (error %lu); GPU tier "
                        "disabled (CPU path remains active).\n",
                (unsigned long)out->error);
        return 0;
    }
    /* Classified independently of the probe: identity inspection opens with
     * backup semantics and therefore SUCCEEDS on a directory, so "is it a
     * file" has to be asked separately or a directory named amdhip64_7.dll
     * would sail through and fail much later as an opaque access error. */
    attrs = GetFileAttributesW(out->path);
    if(attrs == INVALID_FILE_ATTRIBUTES){
        coli_hip_reject("amdhip64_7.dll not found in COLI_HIP_RUNTIME_DIR", out->path);
    } else if(attrs & FILE_ATTRIBUTE_DIRECTORY){
        coli_hip_reject("amdhip64_7.dll is a directory, not a file", out->path);
    } else if(!ok){
        /* A real file whose identity could not be read: nothing downstream
         * could be proven, so this is a refusal too. */
        coli_hip_reject("the identity of amdhip64_7.dll could not be established",
                        out->path);
    } else {
        return 1;
    }
    coli_win32_loader_configured_free(out);
    return 0;
}

/* Name a runtime in a message: the canonical path when we have one, otherwise
 * whatever the module list gave us. */
static void coli_hip_report_module(const char *label, const ColiW32Module *m){
    const wchar_t *shown = m->final_path ? m->final_path :
                          (m->path ? m->path : m->entry_path);
    char *utf8 = shown ? coli_hip_utf8(shown) : NULL;
    fprintf(stderr, COLI_VENDOR_TAG "   %s: %s\n", label,
            utf8 ? utf8 : "<path unavailable>");
    free(utf8);
}

static void coli_hip_report_configured(const ColiW32ConfiguredRuntime *cfg){
    const wchar_t *shown = cfg->final_path ? cfg->final_path : cfg->path;
    char *utf8 = shown ? coli_hip_utf8(shown) : NULL;
    fprintf(stderr, COLI_VENDOR_TAG "   configured: %s\n",
            utf8 ? utf8 : "<path unavailable>");
    free(utf8);
}

/* Exactly one amdhip64_7.dll, it is the configured file, and it is the module
 * this loader holds a reference to. Anything else fails closed. *stage* names
 * the moment so a reader can tell a bad starting state from one the backend
 * created. */
static int coli_hip_verify_bound(const ColiW32ConfiguredRuntime *cfg,
                                 HMODULE expected, const char *stage){
    ColiW32Inventory inv;
    int ok = 0;

    coli_win32_loader_inventory(COLI_W32_RUNTIME_BASENAME, &inv);
    if(inv.status != COLI_W32_INV_OK){
        fprintf(stderr, COLI_VENDOR_TAG " %s: the loaded-module list could not "
                        "be read (status %d, error %lu); GPU tier disabled "
                        "(CPU path remains active).\n",
                stage, inv.status, (unsigned long)inv.error);
    } else if(inv.count != 1){
        size_t i;
        fprintf(stderr, COLI_VENDOR_TAG " %s: expected exactly one "
                        "amdhip64_7.dll, found %u; GPU tier disabled "
                        "(CPU path remains active).\n",
                stage, (unsigned)inv.count);
        coli_hip_report_configured(cfg);
        for(i = 0; i < inv.count; i++) coli_hip_report_module("loaded", &inv.items[i]);
    } else if(!inv.items[0].path){
        fprintf(stderr, COLI_VENDOR_TAG " %s: the path of the loaded "
                        "amdhip64_7.dll is unavailable; GPU tier disabled "
                        "(CPU path remains active).\n", stage);
    } else if(inv.items[0].module != expected){
        fprintf(stderr, COLI_VENDOR_TAG " %s: the loaded amdhip64_7.dll is not "
                        "the module this process referenced; GPU tier disabled "
                        "(CPU path remains active).\n", stage);
        coli_hip_report_configured(cfg);
        coli_hip_report_module("loaded", &inv.items[0]);
    } else if(coli_win32_loader_identity_equal(&cfg->id, &inv.items[0].id) != 1){
        fprintf(stderr, COLI_VENDOR_TAG " %s: the loaded amdhip64_7.dll is not "
                        "the configured file; GPU tier disabled "
                        "(CPU path remains active).\n", stage);
        coli_hip_report_configured(cfg);
        coli_hip_report_module("loaded", &inv.items[0]);
    } else {
        ok = 1;
    }
    coli_win32_loader_inventory_free(&inv);
    return ok;
}

/* Decide against what is already loaded, then take one explicit reference to
 * the configured file. Returns the owned HMODULE, or NULL after diagnosing.
 *
 * Both acceptable decisions end in the same LoadLibraryExW on the same exact
 * absolute path: when the module is already present that call returns the
 * existing handle and raises its reference count, so "loaded it" and "adopted
 * it" produce one uniform ownership rule and one uniform release. */
static HMODULE coli_hip_acquire_runtime(const ColiW32ConfiguredRuntime *cfg){
    ColiW32Inventory inv;
    int *equality = NULL;
    HMODULE runtime = NULL;
    int decision;
    size_t i;

    coli_win32_loader_inventory(COLI_W32_RUNTIME_BASENAME, &inv);
    if(inv.status == COLI_W32_INV_OK && inv.count > 0){
        equality = (int *)malloc(inv.count * sizeof *equality);
        if(!equality){
            fprintf(stderr, COLI_VENDOR_TAG " out of memory comparing loaded "
                            "runtimes; GPU tier disabled "
                            "(CPU path remains active).\n");
            coli_win32_loader_inventory_free(&inv);
            return NULL;
        }
        for(i = 0; i < inv.count; i++)
            equality[i] = coli_win32_loader_identity_equal(&cfg->id, &inv.items[i].id);
    }

    decision = coli_win32_loader_decide(cfg->valid, inv.status, inv.count, equality);
    switch(decision){
    case COLI_BIND_REJECT_WRONG_RUNTIME:
        fprintf(stderr, COLI_VENDOR_TAG " a different amdhip64_7.dll is already "
                        "loaded in this process; GPU tier disabled "
                        "(CPU path remains active).\n");
        coli_hip_report_configured(cfg);
        coli_hip_report_module("already loaded", &inv.items[0]);
        break;
    case COLI_BIND_REJECT_MULTIPLE_RUNTIMES:
        fprintf(stderr, COLI_VENDOR_TAG " %u different amdhip64_7.dll modules "
                        "are already loaded in this process; which one an import "
                        "binds to is not ours to choose. GPU tier disabled "
                        "(CPU path remains active).\n", (unsigned)inv.count);
        coli_hip_report_configured(cfg);
        for(i = 0; i < inv.count; i++)
            coli_hip_report_module("already loaded", &inv.items[i]);
        break;
    case COLI_BIND_REJECT_UNVERIFIABLE:
        fprintf(stderr, COLI_VENDOR_TAG " the loaded amdhip64_7.dll state could "
                        "not be established (status %d, error %lu); GPU tier "
                        "disabled (CPU path remains active).\n",
                inv.status, (unsigned long)inv.error);
        coli_hip_report_configured(cfg);
        break;
    default: {
        /* NEED_EXACT_LOAD and REUSE_EXACT_MATCH take the same action. The path
         * is absolute and exact: no PATH, no current directory, no System32,
         * and never a bare name. */
        DWORD err;
        runtime = LoadLibraryExW(cfg->path, NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
        if(!runtime){
            char phrase[160];
            err = GetLastError();
            coli_win_error_phrase(err, phrase, sizeof phrase);
            fprintf(stderr, COLI_VENDOR_TAG " the configured amdhip64_7.dll "
                            "could not be loaded: %s; GPU tier disabled "
                            "(CPU path remains active).\n", phrase);
            coli_hip_report_configured(cfg);
            break;
        }
        if(!coli_hip_verify_bound(cfg, runtime, "runtime verification")){
            FreeLibrary(runtime);
            runtime = NULL;
        }
        break;
    }
    }
    free(equality);
    coli_win32_loader_inventory_free(&inv);
    return runtime;
}

/* <executable directory>\coli_hip.dll, Unicode and dynamically sized, loaded
 * by absolute path only. There is deliberately no bare-name fallback: a
 * fallback that can reach System32 would let some other coli_hip.dll satisfy
 * the load and quietly undo the runtime contract above. */
static HMODULE coli_hip_load_backend(void){
    DWORD err = 0, attrs;
    wchar_t *exe, *candidate = NULL, *slash, *p;
    HMODULE backend = NULL;
    char phrase[160], *shown;

    exe = coli_win32_loader_exe_path(&err);
    if(!exe){
        coli_win_error_phrase(err, phrase, sizeof phrase);
        fprintf(stderr, COLI_VENDOR_TAG " the executable path could not be "
                        "determined: %s; GPU tier disabled "
                        "(CPU path remains active).\n", phrase);
        return NULL;
    }
    /* Trim to the directory, keeping the separator so a drive root stays
     * "C:\" rather than becoming the drive-relative "C:". */
    slash = NULL;
    for(p = exe; *p; p++) if(coli_win32_loader_is_sep(*p)) slash = p;
    if(!slash){
        fprintf(stderr, COLI_VENDOR_TAG " the executable path has no directory "
                        "component; GPU tier disabled "
                        "(CPU path remains active).\n");
        free(exe);
        return NULL;
    }
    slash[1] = L'\0';

    candidate = coli_win32_loader_join(exe, COLI_BACKEND_DLL_W, &err);
    if(!candidate){
        coli_win_error_phrase(err, phrase, sizeof phrase);
        fprintf(stderr, COLI_VENDOR_TAG " " COLI_BACKEND_DLL " path could not be "
                        "built: %s; GPU tier disabled "
                        "(CPU path remains active).\n", phrase);
        free(exe);
        return NULL;
    }

    backend = LoadLibraryExW(candidate, NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
    if(!backend){
        err = GetLastError();                 /* before anything else runs */
        coli_win_error_phrase(err, phrase, sizeof phrase);
        attrs = GetFileAttributesW(candidate);
        shown = coli_hip_utf8(candidate);
        fprintf(stderr, COLI_VENDOR_TAG " " COLI_BACKEND_DLL " could not be "
                        "loaded; GPU tier disabled (CPU path remains active).\n");
        fprintf(stderr, COLI_VENDOR_TAG "   candidate %s: %s; load failed with %s\n",
                shown ? shown : "<unprintable path>",
                attrs == INVALID_FILE_ATTRIBUTES ? "does not exist" :
                (attrs & FILE_ATTRIBUTE_DIRECTORY) ? "exists but is a directory" :
                "exists as a file",
                phrase);
        free(shown);
    }
    free(candidate);
    free(exe);
    return backend;
}
#endif /* COLI_HIP_DLL */

/* Resolve the DLL and all 11 symbols. Returns 1 on success, 0 otherwise.
 * Idempotent: the first call (success or fail) sticks; later calls are no-ops
 * that return the cached result. The engine treats a 0 return as "CUDA
 * unavailable" and falls back to the CPU path without aborting. */
static int coli_cuda_load(void){
#ifdef COLI_HIP_DLL
    ColiW32ConfiguredRuntime cfg;
    HMODULE runtime = NULL;      /* owned here until the final transfer */
#endif
    if(g_cuda.loaded) return g_cuda.available;
    /* Set before any work, and never cleared: one attempt per process.
     *
     * COLI_HIP_RUNTIME_DIR is start-up configuration, and a wrong or ambiguous
     * set of loaded runtimes cannot be repaired from inside a running process
     * — a retry would re-run a deterministic failure against the same module
     * state. It is also what keeps this function safe without a lock: see the
     * single-entry note below. */
    g_cuda.loaded = 1;

#ifdef COLI_HIP_DLL
    /* No synchronisation here, and that is a traced conclusion rather than an
     * assumption: coli_cuda_load has exactly two callers, coli_cuda_init and
     * coli_cuda_attention_project_ragged. colibri.c calls init on the main
     * thread before model initialisation and therefore before any worker
     * thread exists, and the ragged path is reachable only after that init
     * succeeded. If a third caller ever appears, or the guard in colibri.c
     * moves, this needs an explicit lock instead. */
    if(!coli_hip_configure(&cfg)) return 0;
    runtime = coli_hip_acquire_runtime(&cfg);
    if(!runtime){
        coli_win32_loader_configured_free(&cfg);
        return 0;
    }
    g_cuda.dll = coli_hip_load_backend();
    if(!g_cuda.dll){
        FreeLibrary(runtime);
        coli_win32_loader_configured_free(&cfg);
        return 0;
    }
    /* The backend is mapped, which pulled in its own dependencies. Ask again:
     * if loading it introduced a second amdhip64_7.dll, or displaced the one
     * we referenced, nothing below this point can be trusted. */
    if(!coli_hip_verify_bound(&cfg, runtime, "post-backend verification")){
        FreeLibrary(g_cuda.dll);
        g_cuda.dll = NULL;
        FreeLibrary(runtime);
        coli_win32_loader_configured_free(&cfg);
        return 0;
    }
    coli_win32_loader_configured_free(&cfg);
#else
    /* Load the backend DLL from the engine's OWN directory, by absolute path —
     * never a bare name. LoadLibraryA(COLI_BACKEND_DLL) searches the current
     * working directory (and, without SafeDllSearchMode, other writable dirs):
     * an attacker who drops a same-named DLL where the user launches glm.exe (or
     * inside a downloaded model directory the user cd's into) would get their
     * DllMain run at load — classic DLL hijacking -> arbitrary code execution.
     * Resolving the path next to glm.exe and loading THAT specific file with
     * LOAD_WITH_ALTERED_SEARCH_PATH anchors both the DLL and its dependency
     * search to the trusted install directory instead of the CWD. */
    char dllpath[MAX_PATH];
    /* Diagnostic-only bookkeeping. Each error is taken immediately after the
     * call that produced it, with no intervening Windows API — GetLastError is
     * clobbered by almost anything, and the two attempts fail for genuinely
     * different reasons often enough that collapsing them loses information. */
    int   candidate_built = 0;
    DWORD primary_err = 0, fallback_err = 0;
    DWORD mn = GetModuleFileNameA(NULL, dllpath, (DWORD)sizeof(dllpath));
    if(mn > 0 && mn < sizeof(dllpath)){
        char *slash = strrchr(dllpath, '\\');
        if(slash && (size_t)(slash + 1 - dllpath) + sizeof(COLI_BACKEND_DLL) <= sizeof(dllpath)){
            strcpy(slash + 1, COLI_BACKEND_DLL);
            candidate_built = 1;
            g_cuda.dll = LoadLibraryExA(dllpath, NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
            if(!g_cuda.dll) primary_err = GetLastError();
        }
    }
    if(!g_cuda.dll){
        /* fallback (GetModuleFileNameA praticamente non fallisce): cerca solo
         * nella dir dell'applicazione e in System32, MAI la CWD. */
        g_cuda.dll = LoadLibraryExA(COLI_BACKEND_DLL, NULL,
            LOAD_LIBRARY_SEARCH_APPLICATION_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32);
        if(!g_cuda.dll) fallback_err = GetLastError();
    }
    if(!g_cuda.dll){
        /* Report what was actually observed, and nothing beyond it. The
         * candidate is classified here, after both attempts, so this can never
         * influence whether or in what order they happened. Windows does not
         * tell us WHICH dependency or WHICH procedure was missing, so neither
         * does this message. */
        char primary_msg[160], fallback_msg[160];
        coli_win_error_phrase(fallback_err, fallback_msg, sizeof(fallback_msg));
        fprintf(stderr, COLI_VENDOR_TAG " " COLI_BACKEND_DLL " could not be loaded; "
                        "GPU tier disabled (CPU path remains active).\n");
        if(candidate_built){
            DWORD attrs = GetFileAttributesA(dllpath);
            coli_win_error_phrase(primary_err, primary_msg, sizeof(primary_msg));
            if(attrs == INVALID_FILE_ATTRIBUTES){
                DWORD why = GetLastError();
                if(why == ERROR_FILE_NOT_FOUND || why == ERROR_PATH_NOT_FOUND)
                    fprintf(stderr, COLI_VENDOR_TAG "   candidate %s: does not exist; "
                                    "load failed with %s\n", dllpath, primary_msg);
                else
                    fprintf(stderr, COLI_VENDOR_TAG "   candidate %s: could not be classified "
                                    "(Windows error %lu); load failed with %s\n",
                            dllpath, (unsigned long)why, primary_msg);
            } else if(attrs & FILE_ATTRIBUTE_DIRECTORY){
                fprintf(stderr, COLI_VENDOR_TAG "   candidate %s: exists but is a directory; "
                                "load failed with %s\n", dllpath, primary_msg);
            } else {
                fprintf(stderr, COLI_VENDOR_TAG "   candidate %s: exists as a file; "
                                "load failed with %s\n", dllpath, primary_msg);
            }
        } else {
            /* The MAX_PATH-bounded construction above is the only way to get
             * here; it is a real limitation, so say so instead of implying the
             * file was looked for and missed. The HIP host no longer has it —
             * its path is built dynamically — but CUDA's flow is unchanged. */
            fprintf(stderr, COLI_VENDOR_TAG "   candidate path could not be constructed "
                            "(executable directory unavailable or longer than MAX_PATH); "
                            "no primary load was attempted\n");
        }
        fprintf(stderr, COLI_VENDOR_TAG "   fallback by name " COLI_BACKEND_DLL ": %s\n",
                fallback_msg);
        return 0;
    }
#endif /* COLI_HIP_DLL */

    /* A mandatory symbol is missing: the backend goes, and so does the runtime
     * reference the HIP path acquired. The CUDA build expands the release to
     * nothing, so its behaviour is byte-for-byte what it was. */
#ifdef COLI_HIP_DLL
    #define COLI_RELEASE_RUNTIME_ON_FAIL FreeLibrary(runtime);
#else
    #define COLI_RELEASE_RUNTIME_ON_FAIL
#endif

    #define RESOLVE(name, type) \
        /* GetProcAddress returns FARPROC (void(*)(void)); casting it to a   \
         * specific function-pointer type is the standard LoadLibrary idiom. \
         * -Wcast-function-type flags it but it is safe: the DLL exported     \
         * the symbol with extern "C" and the exact signature we expect. */   \
        _Pragma("GCC diagnostic push") \
        _Pragma("GCC diagnostic ignored \"-Wcast-function-type\"") \
        g_cuda.name = (type)GetProcAddress(g_cuda.dll, "coli_cuda_" #name); \
        _Pragma("GCC diagnostic pop") \
        if(!g_cuda.name){ \
            fprintf(stderr, COLI_VENDOR_TAG " " COLI_BACKEND_DLL \
                            " missing symbol coli_cuda_" #name "\n"); \
            FreeLibrary(g_cuda.dll); g_cuda.dll=NULL; \
            COLI_RELEASE_RUNTIME_ON_FAIL \
            return 0; }

    /* Optional symbol: a DLL predating it leaves the pointer NULL and only that
     * feature degrades (fmt=6 tensors stay CPU-side), instead of taking the whole
     * GPU backend down over one missing entry point. */
    #define RESOLVE_OPT(name, type) \
        _Pragma("GCC diagnostic push") \
        _Pragma("GCC diagnostic ignored \"-Wcast-function-type\"") \
        g_cuda.name = (type)GetProcAddress(g_cuda.dll, "coli_cuda_" #name); \
        _Pragma("GCC diagnostic pop")

    RESOLVE(init,           fn_init)
    RESOLVE(shutdown,       fn_shutdown)
    RESOLVE(device_count,   fn_device_count)
    RESOLVE(device_at,      fn_device_at)
    RESOLVE(mem_info,       fn_mem_info)
    /* Optional: a DLL predating #653 leaves this NULL; the wrapper then reports
     * "not integrated" (0), so the RAM-budget correction simply doesn't apply
     * rather than taking the whole GPU backend down over one missing symbol. */
    RESOLVE_OPT(device_integrated, fn_device_integrated)
    RESOLVE(stats,          fn_stats)
    RESOLVE(group_stats,    fn_group_stats)
    RESOLVE(group_stats_device, fn_group_stats_device)
    RESOLVE(expert_mlp,     fn_expert_mlp)
    RESOLVE(expert_group,   fn_expert_group)
    RESOLVE(expert_group_issue, fn_expert_group_issue)
    RESOLVE(expert_group_take, fn_expert_group_take)
    RESOLVE(attention_absorb, fn_attention_absorb)
    RESOLVE(tensor_upload,  fn_tensor_upload)
    RESOLVE(tensor_upload_g, fn_tensor_upload_g)
    RESOLVE_OPT(e8_set_grid, fn_e8_set_grid)
    RESOLVE_OPT(fp8_set_lut, fn_fp8_set_lut)
    RESOLVE(matmul,         fn_matmul)
    RESOLVE(tensor_free,    fn_tensor_free)
    RESOLVE(tensor_bytes,   fn_tensor_bytes)
    RESOLVE(tensor_device,  fn_tensor_device)

    RESOLVE(attention_absorb_batch, fn_attention_absorb_batch)
    RESOLVE(attention_absorb_batch_dev, fn_attention_absorb_batch_dev)
    RESOLVE(attention_absorb_kvdev, fn_attention_absorb_kvdev)
    RESOLVE(attention_project_batch, fn_attention_project_batch)
    RESOLVE(attention_project_ragged, fn_attention_project_ragged)
    RESOLVE(attention_project_batch_dev, fn_attention_project_batch_dev)
    RESOLVE(attention_project_batch_dev_out, fn_attention_project_batch_dev_out)
    RESOLVE(pipe_add, fn_pipe_add)
    RESOLVE(pipe_alloc, fn_pipe_alloc)
    RESOLVE(pipe_copy2d, fn_pipe_copy2d)
    RESOLVE(pipe_download, fn_pipe_download)
    RESOLVE(pipe_free, fn_pipe_free)
    RESOLVE(pipe_gemm, fn_pipe_gemm)
    RESOLVE(pipe_peer_copy, fn_pipe_peer_copy)
    RESOLVE(pipe_rmsnorm, fn_pipe_rmsnorm)
    RESOLVE(pipe_rmsnorm_s, fn_pipe_rmsnorm_s)
    RESOLVE(expert_group_resident_issue, fn_group_resident_issue)
    RESOLVE(expert_group_resident_take, fn_group_resident_take)
    RESOLVE(pipe_router, fn_pipe_router)
    RESOLVE(pipe_rope, fn_pipe_rope)
    RESOLVE(pipe_rope_base, fn_pipe_rope_base)
    RESOLVE(pipe_rows_add, fn_pipe_rows_add)
    RESOLVE(pipe_scratch, fn_pipe_scratch)
    RESOLVE(pipe_silu_mul, fn_pipe_silu_mul)
    RESOLVE(pipe_sync, fn_pipe_sync)
    RESOLVE(pipe_upload, fn_pipe_upload)
    RESOLVE(shared_mlp_w4a16, fn_shared_mlp_w4a16)
    RESOLVE(tensor_update, fn_tensor_update)
    #undef RESOLVE
    #undef COLI_RELEASE_RUNTIME_ON_FAIL

#ifdef COLI_HIP_DLL
    /* Everything above proved: the configured file is the only amdhip64_7.dll
     * in the process, it is the module we referenced, it survived the backend
     * load, and every mandatory symbol resolved. Only now does the reference
     * stop being local, so no earlier failure path can leak or double-free it. */
    g_cuda.hip_runtime = runtime;
#endif
    g_cuda.available = 1;
    return 1;
}

/* ---- Public wrappers: match backend_cuda.h signatures exactly.
 * Each forwards to the resolved pointer; if the DLL never loaded, return the
 * "not initialized" result the engine already handles (init returns 0, matmul
 * returns 0 so the caller marks the tensor cuda_failed and uses CPU). ---- */

int coli_cuda_init(const int *devices, int count){
    if(!coli_cuda_load()) return 0;
    return g_cuda.init(devices, count);
}

void coli_cuda_shutdown(void){
    if(g_cuda.available && g_cuda.shutdown) g_cuda.shutdown();
#ifdef COLI_HIP_DLL
    /* Backend first, then the runtime it imports: releasing the runtime while
     * the backend is still mapped would leave the backend holding the only
     * reference to a module we no longer track. Safe after a failed
     * initialisation too, because both handles are NULL then.
     *
     * Note for the record, not a defect introduced here: colibri.c only calls
     * this on the ANS-pack early exit, so in ordinary runs the practical
     * cleanup path is process teardown. Adding the missing normal-exit call is
     * a separate lifecycle question. */
    if(g_cuda.dll){ FreeLibrary(g_cuda.dll); g_cuda.dll = NULL; }
    if(g_cuda.hip_runtime){ FreeLibrary(g_cuda.hip_runtime); g_cuda.hip_runtime = NULL; }
    g_cuda.available = 0;
#endif
}

int coli_cuda_device_count(void){
    if(!g_cuda.available) return 0;
    return g_cuda.device_count();
}

int coli_cuda_device_at(int index){
    if(!g_cuda.available) return -1;
    return g_cuda.device_at(index);
}

int coli_cuda_mem_info(int device, size_t *free_bytes, size_t *total_bytes){
    if(!g_cuda.available){ if(free_bytes)*free_bytes=0; if(total_bytes)*total_bytes=0; return 0; }
    return g_cuda.mem_info(device, free_bytes, total_bytes);
}

int coli_cuda_device_integrated(int device){
    if(!g_cuda.available || !g_cuda.device_integrated) return 0;
    return g_cuda.device_integrated(device);
}

void coli_cuda_stats(int device, size_t *tensor_count, size_t *tensor_bytes){
    if(!g_cuda.available){ if(tensor_count)*tensor_count=0; if(tensor_bytes)*tensor_bytes=0; return; }
    g_cuda.stats(device, tensor_count, tensor_bytes);
}

void coli_cuda_group_stats(uint64_t *calls, uint64_t *experts, uint64_t *rows,
                           double *h2d_ms, double *kernel_ms, double *d2h_ms){
    if(!g_cuda.available){
        if(calls)*calls=0; if(experts)*experts=0; if(rows)*rows=0;
        if(h2d_ms)*h2d_ms=0; if(kernel_ms)*kernel_ms=0; if(d2h_ms)*d2h_ms=0;
        return;
    }
    g_cuda.group_stats(calls, experts, rows, h2d_ms, kernel_ms, d2h_ms);
}

void coli_cuda_group_stats_device(int device, uint64_t *calls, uint64_t *experts,
                                  uint64_t *rows, double *h2d_ms,
                                  double *kernel_ms, double *d2h_ms){
    if(!g_cuda.available){
        if(calls)*calls=0; if(experts)*experts=0; if(rows)*rows=0;
        if(h2d_ms)*h2d_ms=0; if(kernel_ms)*kernel_ms=0; if(d2h_ms)*d2h_ms=0;
        return;
    }
    g_cuda.group_stats_device(device, calls, experts, rows, h2d_ms, kernel_ms, d2h_ms);
}

int coli_cuda_expert_mlp(ColiCudaTensor *gate, ColiCudaTensor *up,
                         ColiCudaTensor *down, float *y, const float *x, int S){
    if(!g_cuda.available) return 0;
    return g_cuda.expert_mlp(gate, up, down, y, x, S);
}

int coli_cuda_expert_group(ColiCudaTensor *const *gates, ColiCudaTensor *const *ups,
                           ColiCudaTensor *const *downs, const int *rows, int count,
                           float *y, const float *x){
    if(!g_cuda.available) return 0;
    return g_cuda.expert_group(gates, ups, downs, rows, count, y, x);
}

int coli_cuda_expert_group_issue(ColiCudaTensor *const *gates,
                                 ColiCudaTensor *const *ups,
                                 ColiCudaTensor *const *downs,
                                 const int *rows, int count, const float *x){
    if(!g_cuda.available) return 0;
    return g_cuda.expert_group_issue(gates, ups, downs, rows, count, x);
}

const float *coli_cuda_expert_group_take(int device){
    if(!g_cuda.available) return NULL;
    return g_cuda.expert_group_take(device);
}

int coli_cuda_attention_absorb(ColiCudaTensor *kv_b, float *ctx, const float *q,
                               const float *latent, const float *rope, int H, int Q,
                               int R, int V, int K, int T, float attention_scale){
    if(!g_cuda.available) return 0;
    return g_cuda.attention_absorb(kv_b, ctx, q, latent, rope, H, Q, R, V, K, T, attention_scale);
}

int coli_cuda_tensor_upload(ColiCudaTensor **tensor, const void *weights,
                            const float *scales, int fmt, int I, int O, int device){
    if(!g_cuda.available) return 0;
    return g_cuda.tensor_upload(tensor, weights, scales, fmt, I, O, device);
}

int coli_cuda_tensor_upload_g(ColiCudaTensor **tensor, const void *weights, const float *scales, int fmt, int I, int O, int device, int gs){
    if(!g_cuda.available || !g_cuda.tensor_upload_g){ return 0; }
    return g_cuda.tensor_upload_g(tensor, weights, scales, fmt, I, O, device, gs);
}

int coli_cuda_e8_set_grid(const void *grid){
    if(!g_cuda.available || !g_cuda.e8_set_grid) return 0;   /* fmt=6 stays CPU-side */
    return g_cuda.e8_set_grid(grid);
}

int coli_cuda_fp8_set_lut(const float *lut){
    if(!g_cuda.available || !g_cuda.fp8_set_lut) return 0;   /* fmt=8 stays CPU-side */
    return g_cuda.fp8_set_lut(lut);
}

int coli_cuda_matmul(ColiCudaTensor **tensor, float *y, const float *x,
                     const void *weights, const float *scales,
                     int fmt, int S, int I, int O, int device, int gs){
    if(!g_cuda.available) return 0;
    return g_cuda.matmul(tensor, y, x, weights, scales, fmt, S, I, O, device, gs);
}

void coli_cuda_tensor_free(ColiCudaTensor *tensor){
    if(g_cuda.available && g_cuda.tensor_free) g_cuda.tensor_free(tensor);
}

size_t coli_cuda_tensor_bytes(const ColiCudaTensor *tensor){
    if(!g_cuda.available) return 0;
    return g_cuda.tensor_bytes(tensor);
}

int coli_cuda_tensor_device(const ColiCudaTensor *tensor){
    if(!g_cuda.available) return -1;
    return g_cuda.tensor_device(tensor);
}

/* ---- #111 pipeline wrappers ---- */


/* ---- #111 pipeline wrappers (see header for semantics) ---- */

int coli_cuda_attention_absorb_batch(ColiCudaTensor *kv_b,float *ctx,const float *q, const float *latent,const float *rope,int S, int H,int Q,int R,int V,int K,int T, float attention_scale){
    if(!g_cuda.available){ return 0; }
    return g_cuda.attention_absorb_batch(kv_b, ctx, q, latent, rope, S, H, Q, R, V, K, T, attention_scale);
}

int coli_cuda_attention_absorb_batch_dev(ColiCudaTensor *kv_b_shard,float *ctx_dev, const float *q_dev,const float *latent_dev,const float *rope_dev, int S,int H,int Q,int R,int V,int K,int T,float scale){
    if(!g_cuda.available){ return 0; }
    return g_cuda.attention_absorb_batch_dev(kv_b_shard, ctx_dev, q_dev, latent_dev, rope_dev, S, H, Q, R, V, K, T, scale);
}

int coli_cuda_attention_absorb_kvdev(ColiCudaTensor *kv_b,float *ctx,const float *q, const float *latent_dev,const float *rope_dev,int H,int Q,int R,int V,int K,int T, float scale){
    if(!g_cuda.available){ return 0; }
    return g_cuda.attention_absorb_kvdev(kv_b, ctx, q, latent_dev, rope_dev, H, Q, R, V, K, T, scale);
}

int coli_cuda_attention_project_batch(ColiCudaTensor *kv_b,ColiCudaTensor *o_proj, float *out,const float *q,const float *latent, const float *rope,int S,int H,int Q,int R, int V,int K,int T,float attention_scale){
    if(!g_cuda.available){ return 0; }
    return g_cuda.attention_project_batch(kv_b, o_proj, out, q, latent, rope, S, H, Q, R, V, K, T, attention_scale);
}

int coli_cuda_attention_project_ragged(ColiCudaTensor *kv_b,ColiCudaTensor *o_proj,
        float *out,const float *q,const void *const *keys,
        const float *const *latent,const float *const *rope,
        const int *lengths,int S,int H,int Q,int R,int V,int K,int max_t,float attention_scale){
    if(!coli_cuda_load()) return 0;
    return g_cuda.attention_project_ragged(kv_b,o_proj,out,q,keys,latent,rope,lengths,
        S,H,Q,R,V,K,max_t,attention_scale);
}

int coli_cuda_attention_project_batch_dev(ColiCudaTensor *kv_b,ColiCudaTensor *o_proj, float *out,const float *q_dev,const float *latent_dev,const float *rope_dev, int S,int H,int Q,int R,int V,int K,int T,float scale){
    if(!g_cuda.available){ return 0; }
    return g_cuda.attention_project_batch_dev(kv_b, o_proj, out, q_dev, latent_dev, rope_dev, S, H, Q, R, V, K, T, scale);
}

int coli_cuda_attention_project_batch_dev_out(ColiCudaTensor *kv_b,ColiCudaTensor *o_proj, float *out_dev,const float *q_dev,const float *latent_dev,const float *rope_dev, int S,int H,int Q,int R,int V,int K,int T,float scale){
    if(!g_cuda.available){ return 0; }
    return g_cuda.attention_project_batch_dev_out(kv_b, o_proj, out_dev, q_dev, latent_dev, rope_dev, S, H, Q, R, V, K, T, scale);
}

int coli_cuda_pipe_add(int device,float *x_dev,const float *t_dev,size_t n){
    if(!g_cuda.available){ return 0; }
    return g_cuda.pipe_add(device, x_dev, t_dev, n);
}

void * coli_cuda_pipe_alloc(int device,size_t bytes){
    if(!g_cuda.available){ return NULL; }
    return g_cuda.pipe_alloc(device, bytes);
}

int coli_cuda_pipe_copy2d(int device,float *dst,int dpitch,const float *src, int spitch,int width,int height){
    if(!g_cuda.available){ return 0; }
    return g_cuda.pipe_copy2d(device, dst, dpitch, src, spitch, width, height);
}

int coli_cuda_pipe_download(int device,const void *src,void *dst,size_t bytes){
    if(!g_cuda.available){ return 0; }
    return g_cuda.pipe_download(device, src, dst, bytes);
}

void coli_cuda_pipe_free(int device,void *p){
    if(!g_cuda.available){ return; }
    g_cuda.pipe_free(device, p);
}

int coli_cuda_pipe_gemm(ColiCudaTensor *t,float *y_dev,const float *x_dev,int S){
    if(!g_cuda.available){ return 0; }
    return g_cuda.pipe_gemm(t, y_dev, x_dev, S);
}

int coli_cuda_pipe_peer_copy(int dst_dev,float *dst,int src_dev, const float *src,size_t bytes){
    if(!g_cuda.available){ return 0; }
    return g_cuda.pipe_peer_copy(dst_dev, dst, src_dev, src, bytes);
}

int coli_cuda_pipe_rmsnorm(int device,float *y_dev,const float *x_dev, const float *w_dev,int S,int D,float eps){
    if(!g_cuda.available){ return 0; }
    return g_cuda.pipe_rmsnorm(device, y_dev, x_dev, w_dev, S, D, eps);
}

int coli_cuda_expert_group_resident_issue(ColiCudaTensor *const *gates,ColiCudaTensor *const *ups,ColiCudaTensor *const *downs,const float *weights,int count,int home_device,const float *x_src_dev,float *partial_slot_dev){
    if(!g_cuda.available || !g_cuda.expert_group_resident_issue){ return 0; }
    return g_cuda.expert_group_resident_issue(gates, ups, downs, weights, count, home_device, x_src_dev, partial_slot_dev);
}

int coli_cuda_expert_group_resident_take(int home_device,const int *devices,int n_issued,float *slots_dev,float *acc_dev,int D){
    if(!g_cuda.available || !g_cuda.expert_group_resident_take){ return 0; }
    return g_cuda.expert_group_resident_take(home_device, devices, n_issued, slots_dev, acc_dev, D);
}

int coli_cuda_pipe_router(int device,const float *x_dev,const void *rw_dev,const void *rb_dev,int D,int E,int Ksel,float topp,int norm_topk,float routed_scale,int *idx_host,float *w_host,int *keff_host){
    if(!g_cuda.available || !g_cuda.pipe_router){ return 0; }
    return g_cuda.pipe_router(device, x_dev, rw_dev, rb_dev, D, E, Ksel, topp, norm_topk, routed_scale, idx_host, w_host, keff_host);
}

int coli_cuda_pipe_rmsnorm_s(int device,float *y_dev,const float *x_dev, const float *w_dev,int S,int D,float eps, int xstride,int ystride){
    if(!g_cuda.available){ return 0; }
    return g_cuda.pipe_rmsnorm_s(device, y_dev, x_dev, w_dev, S, D, eps, xstride, ystride);
}

int coli_cuda_pipe_rope(int device,float *v_dev,const int *pos_dev,int rows, int stride,int offset,int R,int heads,float theta){
    if(!g_cuda.available){ return 0; }
    return g_cuda.pipe_rope(device, v_dev, pos_dev, rows, stride, offset, R, heads, theta);
}

int coli_cuda_pipe_rope_base(int device,float *v_dev,int pos_base,int rows, int stride,int offset,int R,int heads,float theta){
    if(!g_cuda.available){ return 0; }
    return g_cuda.pipe_rope_base(device, v_dev, pos_base, rows, stride, offset, R, heads, theta);
}

int coli_cuda_pipe_rows_add(int device,float *x_dev,const float *partial_dev, const int *rows_dev,int nrows,int D){
    if(!g_cuda.available){ return 0; }
    return g_cuda.pipe_rows_add(device, x_dev, partial_dev, rows_dev, nrows, D);
}

float * coli_cuda_pipe_scratch(int device,int slot,size_t bytes){
    if(!g_cuda.available){ return NULL; }
    return g_cuda.pipe_scratch(device, slot, bytes);
}

int coli_cuda_pipe_silu_mul(int device,float *gate_dev,const float *up_dev,size_t n){
    if(!g_cuda.available){ return 0; }
    return g_cuda.pipe_silu_mul(device, gate_dev, up_dev, n);
}

int coli_cuda_pipe_sync(int device){
    if(!g_cuda.available){ return 0; }
    return g_cuda.pipe_sync(device);
}

int coli_cuda_pipe_upload(int device,void *dst,const void *src,size_t bytes){
    if(!g_cuda.available){ return 0; }
    return g_cuda.pipe_upload(device, dst, src, bytes);
}

int coli_cuda_shared_mlp_w4a16(ColiCudaTensor *gate, ColiCudaTensor *up, ColiCudaTensor *down, float *y, const float *x, int S){
    if(!g_cuda.available){ return 0; }
    return g_cuda.shared_mlp_w4a16(gate, up, down, y, x, S);
}

int coli_cuda_tensor_update(ColiCudaTensor *tensor, const void *weights, const float *scales){
    if(!g_cuda.available){ return 0; }
    return g_cuda.tensor_update(tensor, weights, scales);
}

#endif /* _WIN32 */
