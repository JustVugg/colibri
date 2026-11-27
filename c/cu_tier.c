/* cu_tier.c -- vk_tier.c's routed-expert tier on a CUDA device (cu_tier.h).
 *
 * The same file compiled again: its calls to backend_vulkan.c go to the CUDA backend's
 * table (cu_xb.cu, cu_tier_ops.h), its public names take the cut_ prefix, its lines the
 * "[CUDA]" tag. backend_vulkan.h comes first, so its declarations keep their names, and
 * cu_tier.h before the definitions, so the compiler checks its declarations against them. */
#ifdef COLI_CUDA
#include <stddef.h>
#include "backend_vulkan.h"
#include "cu_tier_ops.h"
#include "cu_tier.h"

/* Without the table (a CUDA DLL from before it), the sizes the engines ask before vkt_init
 * answer 0 and the tier stays off. */
static int cut_none_available(void) { return 0; }
static const char *cut_none_name(void) { return "no CUDA tier"; }
static size_t cut_none_align(void) { return 256; }
static size_t cut_none_row_bytes(int fmt, int I) { (void)fmt; (void)I; return 0; }
static size_t cut_none_scale_count(int fmt, int I, int O, int gs) { (void)fmt; (void)I; (void)O; (void)gs; return 0; }
static size_t cut_none_alloc_bytes(int fmt, int I, int O, int gs, size_t *sb) { (void)fmt; (void)I; (void)O; (void)gs; if (sb) *sb = 0; return 0; }
static int cut_none_block_elems(int fmt) { (void)fmt; return 0; }
static const CuTierOps g_cut_none = {
    .version = CU_TIER_OPS_VERSION, .available = cut_none_available, .device_name = cut_none_name,
    .buffer_alignment = cut_none_align, .tensor_row_bytes = cut_none_row_bytes, .tensor_scale_count = cut_none_scale_count,
    .tensor_alloc_bytes = cut_none_alloc_bytes, .block_elems = cut_none_block_elems,
};
static const CuTierOps *g_cut_ops;
static const CuTierOps *cut_ops(void) {
    if (!g_cut_ops) {
        const CuTierOps *o = coli_cuda_tier_ops();
        g_cut_ops = o && o->version == CU_TIER_OPS_VERSION ? o : &g_cut_none;
    }
    return g_cut_ops;
}

/* backend_vulkan.c's calls, on the table */
#define coli_vk_available(...) (cut_ops()->available())
#define coli_vk_device_name(...) (cut_ops()->device_name(__VA_ARGS__))
#define coli_vk_device_shares_ram(...) (cut_ops()->device_shares_ram(__VA_ARGS__))
#define coli_vk_device_local_bytes(...) (cut_ops()->device_local_bytes(__VA_ARGS__))
#define coli_vk_mem_budget(...) (cut_ops()->mem_budget(__VA_ARGS__))
#define coli_vk_dev_info(...) (cut_ops()->dev_info(__VA_ARGS__))
#define coli_vk_dev2_open_env(...) (cut_ops()->dev2_open_env(__VA_ARGS__))
#define coli_vk_block_bytes(...) (cut_ops()->block_bytes(__VA_ARGS__))
#define coli_vk_block_bytes_dev(...) (cut_ops()->block_bytes_dev(__VA_ARGS__))
#define coli_vk_buffer_alignment(...) (cut_ops()->buffer_alignment(__VA_ARGS__))
#define coli_vk_pool_stats(...) (cut_ops()->pool_stats(__VA_ARGS__))
#define coli_vk_tier_pool_limit(...) (cut_ops()->tier_pool_limit(__VA_ARGS__))
#define coli_vk_tier_pool_limit_dev(...) (cut_ops()->tier_pool_limit_dev(__VA_ARGS__))
#define coli_vk_tier_extra_pool_limit(...) (cut_ops()->tier_extra_pool_limit(__VA_ARGS__))
#define coli_vk_tier_host_pool(...) (cut_ops()->tier_host_pool(__VA_ARGS__))
/* the CUDA tier's host pool is page-locked memory of its own (cudaHostAlloc), not a Vulkan
 * heap the driver sizes: no room reported, the pool sized from the RAM as before */
#define coli_vk_tier_host_heap_room() ((size_t)0)
/* the CUDA tier has no Vulkan mapped-weights window: the trunk-first ask never ran, the
 * host pool keeps its own sizing */
#define coli_vk_window_trunk_first_on() (0)
#define VKT_NO_GIVE_BACK 1   /* vkt_give_back: the dense chain that asks is Vulkan's */
#define coli_vk_block_elems(...) (cut_ops()->block_elems(__VA_ARGS__))
#define coli_vk_tensor_row_bytes(...) (cut_ops()->tensor_row_bytes(__VA_ARGS__))
#define coli_vk_tensor_scale_count(...) (cut_ops()->tensor_scale_count(__VA_ARGS__))
#define coli_vk_tensor_payload(...) (cut_ops()->tensor_payload(__VA_ARGS__))
#define coli_vk_tensor_alloc_bytes(...) (cut_ops()->tensor_alloc_bytes(__VA_ARGS__))
#define coli_vk_tier_tensor_pad(...) (cut_ops()->tier_tensor_pad(__VA_ARGS__))
#define coli_vk_tier_tensor_extra(...) (cut_ops()->tier_tensor_extra(__VA_ARGS__))
#define coli_vk_tier_tensor_like(...) (cut_ops()->tier_tensor_like(__VA_ARGS__))
#define coli_vk_tensor_commit(...) (cut_ops()->tensor_commit(__VA_ARGS__))
#define coli_vk_tensor_refill(...) (cut_ops()->tensor_refill(__VA_ARGS__))
#define coli_vk_tensor_refill_as(...) (cut_ops()->tensor_refill_as(__VA_ARGS__))
#define coli_vk_tensor_copy(...) (cut_ops()->tensor_copy(__VA_ARGS__))
#define coli_vk_tensor_free(...) (cut_ops()->tensor_free(__VA_ARGS__))
#define coli_vk_tensor_bytes(...) (cut_ops()->tensor_bytes(__VA_ARGS__))
#define coli_vk_tensor_info(...) (cut_ops()->tensor_info(__VA_ARGS__))
#define coli_vk_tensor_host_rows(...) (cut_ops()->tensor_host_rows(__VA_ARGS__))
#define coli_vk_xb_init(...) (cut_ops()->xb_init(__VA_ARGS__))
#define coli_vk_xb_init_dev(...) (cut_ops()->xb_init_dev(__VA_ARGS__))
#define coli_vk_xb_ready(...) (cut_ops()->xb_ready(__VA_ARGS__))
#define coli_vk_xb_ready_dev(...) (cut_ops()->xb_ready_dev(__VA_ARGS__))
#define coli_vk_xb_queue_shared(...) (cut_ops()->xb_queue_shared(__VA_ARGS__))
#define coli_vk_xb_expert(...) (cut_ops()->xb_expert(__VA_ARGS__))
#define coli_vk_xb_expert_free(...) (cut_ops()->xb_expert_free(__VA_ARGS__))
#define coli_vk_xb_expert_tensors(...) (cut_ops()->xb_expert_tensors(__VA_ARGS__))
#define coli_vk_xb_expert_entry(...) (cut_ops()->xb_expert_entry(__VA_ARGS__))
#define coli_vk_xb_issue_w(...) (cut_ops()->xb_issue_w(__VA_ARGS__))
#define coli_vk_xb_issue_dev(...) (cut_ops()->xb_issue_dev(__VA_ARGS__))
#define coli_vk_xb_join(...) (cut_ops()->xb_join(__VA_ARGS__))
#define coli_vk_xb_join_dev(...) (cut_ops()->xb_join_dev(__VA_ARGS__))
#define coli_vk_xb_busy_dev(...) (cut_ops()->xb_busy_dev(__VA_ARGS__))
#define coli_vk_xb_stats(...) (cut_ops()->xb_stats(__VA_ARGS__))
#define coli_vk_xb_step_begin(...) (cut_ops()->xb_step_begin(__VA_ARGS__))
#define coli_vk_xb_step_sum(...) (cut_ops()->xb_step_sum(__VA_ARGS__))
#define coli_vk_xb_step_end(...) (cut_ops()->xb_step_end(__VA_ARGS__))
#define coli_vk_xb_sub_fit(...) (cut_ops()->xb_sub_fit(__VA_ARGS__))
#define coli_vk_xb_sub_reserve(...) (cut_ops()->xb_sub_reserve(__VA_ARGS__))
#define coli_vk_xb_sub_issue(...) (cut_ops()->xb_sub_issue(__VA_ARGS__))
#define coli_vk_xb_sub_issue_step(...) (cut_ops()->xb_sub_issue_step(__VA_ARGS__))
#define coli_vk_xb_sub_join(...) (cut_ops()->xb_sub_join(__VA_ARGS__))
#define coli_vk_xb_sub_busy(...) (cut_ops()->xb_sub_busy(__VA_ARGS__))
#define coli_vk_xb_sub_copy(...) (cut_ops()->xb_sub_copy(__VA_ARGS__))
#define coli_vk_xb_sub_copy_drop(...) (cut_ops()->xb_sub_copy_drop(__VA_ARGS__))

/* the tier's names */
#define vkt_begin_forward cut_begin_forward
#define vkt_dev_changes cut_dev_changes
#define vkt_dev_count cut_dev_count
#define vkt_dev_entry cut_dev_entry
#define vkt_dev_host cut_dev_host
#define vkt_dev_table cut_dev_table
#define vkt_dev_table_hybrid cut_dev_table_hybrid
#define vkt_devices cut_devices
#define vkt_epoch cut_epoch
#define vkt_expert_bytes cut_expert_bytes
#define vkt_give_back cut_give_back
#define vkt_host_pairs cut_host_pairs
#define vkt_host_pairs_w cut_host_pairs_w
#define vkt_host_pool_on cut_host_pool_on
#define vkt_init cut_init
#define vkt_issue cut_issue
#define vkt_issue_w cut_issue_w
#define vkt_join cut_join
#define vkt_join_sum cut_join_sum
#define vkt_layers cut_layers
#define vkt_note cut_note
#define vkt_plan cut_plan
#define vkt_put cut_put
#define vkt_put_done cut_put_done
#define vkt_ram_first cut_ram_first
#define vkt_ram_gave cut_ram_gave
#define vkt_ready cut_ready
#define vkt_report cut_report
#define vkt_resident cut_resident
#define vkt_score cut_score
#define vkt_settle cut_settle
#define vkt_shutdown cut_shutdown
#define vkt_stage_room cut_stage_room
#define vkt_step_rows cut_step_rows
#define vkt_stream_prefetch cut_stream_prefetch
#define vkt_take_back cut_take_back
#define vkt_wanted cut_wanted
#define vkt_wants cut_wants
#define vkt_warm_report cut_warm_report

#define VKT_TAG "[CUDA]"
#ifndef COLI_VULKAN
#define COLI_VULKAN 1   /* vk_tier.c's body */
#define CUT_VULKAN_DEFINED 1
#endif
#include "vk_tier.c"

/* ---- direct-to-device MXFP4 (fmt 7, E8M0) warm-start hooks ------------------
 * Lets WEX4 (WNR4/WNR3) warm-start decode experts directly into primary CUDA
 * device VRAM (pool 1) via coli_wex_cuda_decode_batch_to_device, eliminating
 * D2H -> host slab -> CPU E8M0 conversion -> second H2D bounce. Resolves the
 * two optional cu_xb.cu exports lazily (gracefully returns 0 on older DLLs,
 * HIP builds, or fake-tier unit tests). */
typedef int (*CutAllocMxfp4Fn)(int, ColiVkTensor **, int, int, int, size_t, size_t,
                               int *, unsigned char **, unsigned char **);
typedef int (*CutExpandE8m0Fn)(ColiVkTensor *const *, int);

#ifdef _WIN32
static pthread_once_t g_cut_direct_once = PTHREAD_ONCE_INIT;
static struct { CutAllocMxfp4Fn alloc; CutExpandE8m0Fn expand; } g_cut_direct;
static void cut_direct_load_once(void) {
#ifndef COLI_HIP_DLL
    HMODULE h = GetModuleHandleA("coli_cuda.dll");
    if (!h) return;
#if defined(__GNUC__) || defined(__clang__)
    _Pragma("GCC diagnostic push")
    _Pragma("GCC diagnostic ignored \"-Wcast-function-type\"")
#endif
    CutAllocMxfp4Fn a = (CutAllocMxfp4Fn)GetProcAddress(h, "coli_cuda_tier_alloc_mxfp4");
    CutExpandE8m0Fn e = (CutExpandE8m0Fn)GetProcAddress(h, "coli_cuda_tier_expand_e8m0");
#if defined(__GNUC__) || defined(__clang__)
    _Pragma("GCC diagnostic pop")
#endif
    if (a && e) { g_cut_direct.alloc = a; g_cut_direct.expand = e; }
#endif
}
static int cut_direct_load(void) {
    pthread_once(&g_cut_direct_once, cut_direct_load_once);
    return g_cut_direct.alloc != NULL && g_cut_direct.expand != NULL;
}
#define cut_direct_alloc_fn  g_cut_direct.alloc
#define cut_direct_expand_fn g_cut_direct.expand
#elif defined(__GNUC__) || defined(__clang__)
__attribute__((weak)) extern int coli_cuda_tier_alloc_mxfp4(int, ColiVkTensor **, int, int, int, size_t, size_t,
                                                            int *, unsigned char **, unsigned char **);
__attribute__((weak)) extern int coli_cuda_tier_expand_e8m0(ColiVkTensor *const *, int);
static int cut_direct_load(void) {
    return coli_cuda_tier_alloc_mxfp4 != NULL && coli_cuda_tier_expand_e8m0 != NULL;
}
#define cut_direct_alloc_fn  coli_cuda_tier_alloc_mxfp4
#define cut_direct_expand_fn coli_cuda_tier_expand_e8m0
#else
static int cut_direct_load(void) { return 0; }
static int cut_direct_alloc_stub(int d, ColiVkTensor **t, int I, int O, int g, size_t pw, size_t ps,
                                 int *cd, unsigned char **w, unsigned char **s) {
    (void)d; (void)t; (void)I; (void)O; (void)g; (void)pw; (void)ps; (void)cd; (void)w; (void)s; return 0;
}
static int cut_direct_expand_stub(ColiVkTensor *const *t, int n) { (void)t; (void)n; return 0; }
#define cut_direct_alloc_fn  cut_direct_alloc_stub
#define cut_direct_expand_fn cut_direct_expand_stub
#endif

int cut_warm_direct_ready(int *cuda_dev_out) {
    if (!T.on || T.full || T.c.hidden < 32 || (T.c.hidden % 32) != 0 || T.c.inter < 32 || (T.c.inter % 32) != 0)
        return 0;
    if (T.form[0].gu.kind != VKT_SRC_MXFP4_E8M0 || T.form[0].dn.kind != VKT_SRC_MXFP4_E8M0 ||
        T.form[0].gu_gs != 32 || T.form[0].dn_gs != 32)
        return 0;
    if (!cut_direct_load()) return 0;
    int dev = coli_cuda_device_at(0);
    if (dev < 0) return 0;
    if (cuda_dev_out) *cuda_dev_out = dev;
    return 1;
}

int cut_warm_direct_alloc(int layer, int eid, CutWarmDirect *out,
                          unsigned char *dev_w[3], unsigned char *dev_s[3]) {
    if (!T.on || T.full || !out || !dev_w || !dev_s) return 0;
    if (layer < 0 || layer >= T.main_layers || eid < 0 || eid >= T.c.experts) return 0;
    if (place_of(slot(layer, eid)) != 0) return 0;
    const __typeof__(T.form[0]) *fm = &T.form[form_of(layer)];
    if (fm->gu.kind != VKT_SRC_MXFP4_E8M0 || fm->dn.kind != VKT_SRC_MXFP4_E8M0 ||
        fm->gu_gs != 32 || fm->dn_gs != 32)
        return 0;
    if (!cut_direct_load()) return 0;
    const int H = T.c.hidden, F = T.c.inter;
    ColiVkTensor *t[3] = {NULL, NULL, NULL};
    for (int k = 0; k < 3; k++) {
        const int I = k < 2 ? H : F, O = k < 2 ? F : H;
        if (!cut_direct_alloc_fn(0, &t[k], I, O, 32, pad_w_of(layer, k == 2), pad_s_of(layer, k == 2),
                                 NULL, &dev_w[k], &dev_s[k])) {
            for (int j = 0; j < k; j++) coli_vk_tensor_free(t[j]);
            return 0;
        }
    }
    out->layer = layer;
    out->eid = eid;
    for (int k = 0; k < 3; k++) out->t[k] = t[k];
    return 1;
}

int cut_warm_direct_commit(CutWarmDirect *items, const int *ok, int n) {
    if (!T.on || !items || !ok || n <= 0) return 0;
    ColiVkTensor *stack_t[192];
    ColiVkTensor **all_t = n <= 64 ? stack_t : (ColiVkTensor **)malloc((size_t)n * 3u * sizeof(*all_t));
    int nt = 0, committed = 0;
    int expand_ok = 0;
    if (all_t) {
        for (int i = 0; i < n; i++) {
            if (!ok[i] || !items[i].t[0] || !items[i].t[1] || !items[i].t[2]) continue;
            for (int k = 0; k < 3; k++) all_t[nt++] = (ColiVkTensor *)items[i].t[k];
        }
        expand_ok = nt == 0 ? 1 : (cut_direct_load() && cut_direct_expand_fn(all_t, nt));
        if (all_t != stack_t) free(all_t);
    }
    for (int i = 0; i < n; i++) {
        ColiVkTensor *t[3] = {
            (ColiVkTensor *)items[i].t[0], (ColiVkTensor *)items[i].t[1], (ColiVkTensor *)items[i].t[2]
        };
        items[i].t[0] = items[i].t[1] = items[i].t[2] = NULL;
        if (expand_ok && ok[i] && t[0] && t[1] && t[2]) {
            push_done(items[i].layer, items[i].eid, 1, 0, t, NULL);
            __atomic_add_fetch(&T.warm, 1, __ATOMIC_RELAXED);
            committed++;
        } else {
            for (int k = 0; k < 3; k++) if (t[k]) coli_vk_tensor_free(t[k]);
        }
    }
    return committed;
}
#endif /* COLI_CUDA */
