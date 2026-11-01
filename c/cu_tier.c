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
#define coli_vk_xb_join_sum(...) (cut_ops()->xb_join_sum(__VA_ARGS__))
#define coli_vk_xb_sum_arm(...) (cut_ops()->xb_sum_arm(__VA_ARGS__))
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
#define vkt_cs_armed cut_cs_armed
#define vkt_bank_fill cut_bank_fill
#define vkt_bank_switch cut_bank_switch
#define vkt_dev_changes cut_dev_changes
#define vkt_dev_count cut_dev_count
#define vkt_dev_entry cut_dev_entry
#define vkt_dev_host cut_dev_host
#define vkt_dev_table cut_dev_table
#define vkt_dev_table_hybrid cut_dev_table_hybrid
#define vkt_devices cut_devices
#define vkt_epoch cut_epoch
#define vkt_expert_bytes cut_expert_bytes
#define vkt_forkdec_store_line cut_forkdec_store_line
#define vkt_forkdec_wanted cut_forkdec_wanted
#define vkt_give_back cut_give_back
#define vkt_host_pairs cut_host_pairs
#define vkt_host_pairs_w cut_host_pairs_w
#define vkt_host_pool_on cut_host_pool_on
#define vkt_host_room cut_host_room
#define vkt_init cut_init
#define vkt_issue cut_issue
#define vkt_issue_w cut_issue_w
#define vkt_issue_group cut_issue_group
#define vkt_join cut_join
#define vkt_join_merged cut_join_merged
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
#define vkt_cs_served cut_cs_served
#define vkt_score cut_score
#define vkt_settle cut_settle
#define vkt_slab_forget cut_slab_forget
#define vkt_slabreg_slot cut_slabreg_slot
#define vkt_slabreg_slot_free cut_slabreg_slot_free
#define vkt_shutdown cut_shutdown
#define vkt_stage_room cut_stage_room
#define vkt_step_rows cut_step_rows
#define vkt_stream_prefetch cut_stream_prefetch
#define vkt_wanted cut_wanted
#define vkt_wants cut_wants
#define vkt_warm_report cut_warm_report

/* the Vulkan strict gate (COLI_VK_STRICT, backend_vulkan.c) has no CUDA side: the
 * tier's refusal lines above keep their own fprintf, behaviour unchanged */
#define coli_vk_degrade(cls, op, reason) ((void)0)
/* CS-1b's slab-alias capability is the Vulkan backend's own (VK_EXT_external_memory_host
 * imports of the store's host pages); a CUDA backend has no such import, so the calls
 * refuse and the compute site's group degrades its aliased experts to the CPU -- the
 * lever's refusal ladder, never a drop */
#define coli_vk_slab_import(...) (0)
#define coli_vk_tensor_alias(...) (0)
#define coli_vk_tensor_alias_wide(...) (0)
#define coli_vk_slab_unref(...) ((void)0)
#define coli_vk_slab_ref(...) ((void)0)
#define coli_vk_slab_bytes(...) (0)
#define coli_vk_slab_cap() (0)
#define coli_vk_slab_forget(...) ((void)0)
/* the forked decode path's registry is the Vulkan backend's own (coli_vk_slabreg_*,
 * backend_vulkan.c); a CUDA backend has no such registry, so every call declines and
 * the store's slots keep the malloc path, the CS alias the import branch (itself
 * shimmed above) -- the lever's refusal ladder, never a drop */
#define coli_vk_slabreg_wanted() (0)
#define coli_vk_slabreg_status(...) (0)   /* never reached: wanted() declines first */
#define coli_vk_slabreg_alloc(...) (0)
#define coli_vk_slabreg_free(...) ((void)0)
#define coli_vk_slabreg_lookup(...) (0)
#define coli_vk_slabreg_alias(...) (0)
/* the Vulkan backend's CS diagnostics (backend_vulkan.h); the CUDA tier has no slab
 * import, so vk_tier.c's report reads an all-zero snapshot */
#define coli_vk_cs_diag(out) (memset((out), 0, sizeof *(out)))
#define VKT_TAG "[CUDA]"
#ifndef COLI_VULKAN
#define COLI_VULKAN 1   /* vk_tier.c's body */
#define CUT_VULKAN_DEFINED 1
#endif
#include "vk_tier.c"
#endif /* COLI_CUDA */
