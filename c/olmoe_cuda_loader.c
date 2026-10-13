/* olmoe_cuda_loader.c -- Windows runtime loader for the OLMoE CUDA library (olmoe_cuda.h).
 *
 * olmoe.exe is built with MinGW-w64, but the kernels in olmoe_cuda.cu need nvcc + MSVC, and the two ABIs cannot be linked
 * together. The kernels are therefore built into coli_cuda_olmoe.dll (symbols exported through olmoe_cuda.def) and
 * resolved here with LoadLibrary/GetProcAddress; the engine never links cudart. Same arrangement as
 * backend_loader_dsv4.c, and the same contract: if the DLL (or a symbol) is missing every wrapper returns the failure
 * sentinel, olmoe_cuda_available() answers 0, and the engine stays on the CPU.
 *
 * Search order: COLI_OLMOE_CUDA_DLL (a full path), then the directory of the running executable. Not the system
 * search path: a stray copy of the DLL elsewhere must not be picked up silently.
 *
 * Not compiled off Windows: Linux links olmoe_cuda.o directly. */
#ifdef _WIN32
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#include "olmoe_cuda.h"

#define OLMOE_CUDA_DLL_NAME "coli_cuda_olmoe.dll"

typedef int   (*fn_available)(void);
typedef int   (*fn_init)(int, int, int, int, int, int, int, int, int, int, float, float, int);
typedef int   (*fn_set_layer)(int, const void *, const void *, const void *, const void *, const float *, const float *,
                              const float *, const float *, const float *);
typedef int   (*fn_set_head)(const void *, const float *);
typedef void  (*fn_mem_info)(size_t *, size_t *);
typedef int   (*fn_alloc_slots)(int);
typedef void *(*fn_stage_ptr)(void);
typedef int   (*fn_upload_expert)(int, int, int, const int8_t *, const float *);
typedef int   (*fn_unmap_expert)(int, int);
typedef int   (*fn_kv_upload)(int, const float *, const float *, int, int, int);
typedef int   (*fn_begin_token)(const float *, int);
typedef int   (*fn_layer)(int);
typedef int   (*fn_route_fetch)(int *, float *, int *);
typedef int   (*fn_layer_experts)(const int *);
typedef int   (*fn_finish)(float *, float *);
typedef void  (*fn_shutdown)(void);

static struct {
    int tried, ok;
    fn_available available; fn_init init; fn_set_layer set_layer; fn_set_head set_head; fn_mem_info mem_info;
    fn_alloc_slots alloc_slots; fn_stage_ptr stage_ptr; fn_upload_expert upload_expert; fn_unmap_expert unmap_expert;
    fn_kv_upload kv_upload; fn_begin_token begin_token; fn_layer layer_attn; fn_layer layer_route;
    fn_route_fetch route_fetch; fn_layer_experts layer_experts; fn_finish finish; fn_shutdown shutdown;
} L;

static int load(void) {
    if (L.tried) return L.ok;
    L.tried = 1;
    char path[1024];
    const char *ov = getenv("COLI_OLMOE_CUDA_DLL");
    if (ov && *ov) snprintf(path, sizeof path, "%s", ov);
    else {
        DWORD n = GetModuleFileNameA(NULL, path, sizeof path);
        while (n > 0 && path[n - 1] != '\\' && path[n - 1] != '/') n--;
        path[n] = 0;
        strncat(path, OLMOE_CUDA_DLL_NAME, sizeof path - strlen(path) - 1);
    }
    HMODULE h = LoadLibraryExA(path, NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!h) return 0;
#define BIND(f) do { FARPROC p_ = GetProcAddress(h, "olmoe_cuda_" #f); if (!p_) return 0; memcpy(&L.f, &p_, sizeof p_); } while (0)
    BIND(available); BIND(init); BIND(set_layer); BIND(set_head); BIND(mem_info); BIND(alloc_slots); BIND(stage_ptr);
    BIND(upload_expert); BIND(unmap_expert); BIND(kv_upload); BIND(begin_token); BIND(layer_attn); BIND(layer_route);
    BIND(route_fetch); BIND(layer_experts); BIND(finish); BIND(shutdown);
#undef BIND
    L.ok = 1;
    return 1;
}

int olmoe_cuda_available(void) { return load() && L.available(); }
int olmoe_cuda_init(int device, int layers, int hidden, int heads, int head_dim, int inter, int nexp, int topk, int vocab,
                    int max_t, float eps, float theta, int norm_topk) {
    return load() ? L.init(device, layers, hidden, heads, head_dim, inter, nexp, topk, vocab, max_t, eps, theta, norm_topk) : -1;
}
int olmoe_cuda_set_layer(int l, const void *q, const void *k, const void *v, const void *o, const float *in_ln,
                         const float *post_ln, const float *qn, const float *kn, const float *gate) {
    return load() ? L.set_layer(l, q, k, v, o, in_ln, post_ln, qn, kn, gate) : -1;
}
int olmoe_cuda_set_head(const void *lm_head, const float *final_norm) { return load() ? L.set_head(lm_head, final_norm) : -1; }
void olmoe_cuda_mem_info(size_t *f, size_t *t) { if (load()) L.mem_info(f, t); else { *f = 0; *t = 0; } }
int olmoe_cuda_alloc_slots(int n) { return load() ? L.alloc_slots(n) : 0; }
void *olmoe_cuda_stage_ptr(void) { return load() ? L.stage_ptr() : NULL; }
int olmoe_cuda_upload_expert(int layer, int eid, int slot, const int8_t *w, const float *s) {
    return load() ? L.upload_expert(layer, eid, slot, w, s) : -1;
}
int olmoe_cuda_unmap_expert(int layer, int eid) { return load() ? L.unmap_expert(layer, eid) : -1; }
int olmoe_cuda_kv_upload(int layer, const float *K, const float *V, int host_max_t, int from, int to) {
    return load() ? L.kv_upload(layer, K, V, host_max_t, from, to) : -1;
}
int olmoe_cuda_begin_token(const float *x, int pos) { return load() ? L.begin_token(x, pos) : -1; }
int olmoe_cuda_layer_attn(int l) { return load() ? L.layer_attn(l) : -1; }
int olmoe_cuda_layer_route(int l) { return load() ? L.layer_route(l) : -1; }
int olmoe_cuda_route_fetch(int *ids, float *w, int *miss) { return load() ? L.route_fetch(ids, w, miss) : -1; }
int olmoe_cuda_layer_experts(const int *slots) { return load() ? L.layer_experts(slots) : -1; }
int olmoe_cuda_finish(float *logits, float *kv) { return load() ? L.finish(logits, kv) : -1; }
void olmoe_cuda_shutdown(void) { if (L.ok) L.shutdown(); }
#endif
