/* vk_chain.c -- a layer's dense chain recorded into one Vulkan submission (vk_chain.h).
 *
 * The backend (backend_vulkan.c) owns the device and the resident tensors; this file
 * builds its own pipelines from the same shader directory, with one pipeline layout
 * for all of them (eight storage buffers, 128 bytes of push constants), and records
 * them into a ring of frames. A frame is a command buffer, its fence, its descriptor
 * pools and the staging and temporary buffers its copies read; it is reused once its
 * fence has signalled.
 *
 * Barriers: one global memory barrier (compute and transfer, both ways) whenever an op
 * touches a buffer the ops since the last barrier wrote, or writes one they read. The
 * frame opens with that barrier (it orders the frame after everything submitted before
 * it on the queue) and closes with one to the host. Weights are never written, so they
 * never cause one. */
#include "vk_chain.h"
#include "vk_tier.h"
#include "vk_alloc.h"
#include "vk_load.h"
#include "shaders/chain_qsa.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "compat.h"

/* Every device memory block goes through the backend's books (COLI_VK_DEVICE_CAP_MB,
 * coli_vk_mem_alloc): the chain's buffers count against the device's budget as its
 * weights and the tier's experts do. */
static VkResult vkc_mem_alloc(VkDevice d, const VkMemoryAllocateInfo *ai, const VkAllocationCallbacks *cb, VkDeviceMemory *m) {
    (void)cb;
    return (VkResult)coli_vk_mem_alloc((void *)d, ai, m);
}
static void vkc_mem_free(VkDevice d, VkDeviceMemory m, const VkAllocationCallbacks *cb) {
    (void)cb;
    coli_vk_mem_free((void *)d, &m);
}
#undef vkAllocateMemory   /* vk_load.h's: the chain's go through the books above */
#undef vkFreeMemory
#define vkAllocateMemory vkc_mem_alloc
#define vkFreeMemory vkc_mem_free

static double vkc_now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }

/* ---- pipelines ------------------------------------------------------------------ */
enum { P_NORM, P_ROPE, P_ATTN, P_DNCONV, P_EW, P_QSA, P_PLE, P_GEMV, P_NPIPE };
static const char *const pipe_file[P_NPIPE] = {
    "chain_norm.spv", "chain_rope.spv", "chain_attn.spv", "chain_dnconv.spv",
    "chain_ew.spv", "chain_qsa.spv", "chain_ple.spv", "qmatmul.spv"};
#define VKC_GEMM_MAX 4
#define VKC_DNREC_MAX 8
#define VKC_FA_MAX 32  /* chain_attn_flash pipelines, a (head dim, query heads a KV head, cache type) each */
#define VKC_ADG_MAX 32 /* chain_attn_dec's grouped pipelines (mode 2), a (head dim, query heads a KV head, cache type) each */
#define VKC_FA_ROWS 64 /* chain_attn_flash's query rows a workgroup (its R) */
#define VKC_FA_SLICE 16 /* chain_attn_flash's units of COLI_VK_ATTN_SLICE's budget to chain_attnb's one */
#define VKC_FRAMES 16
#define VKC_STAGE0 (((size_t)16 << 20) / VKC_FRAMES)   /* a frame's first staging buffer: 16 MiB over the frames, as when they were 4 of 4 MiB */
#define VKC_BIND 8
#define VKC_TRACK 64
#define VKC_PROF_Q 512
/* COLI_VK_CHAIN_PROF=1: device time per kind of op (timestamps after every op; ops
 * that run side by side between two barriers share the time unevenly) */
enum { PK_GEMV8, PK_GEMV, PK_GEMM, PK_NORM, PK_ROPE, PK_ATTN, PK_DNCONV, PK_DNREC, PK_EW, PK_QSA, PK_PLE, PK_COPY,
       PK_MLA, PK_MLAW, PK_DSA, PK_KDA, PK_MHC, PK_ARES, PK_ROUTE, PK_MOE, PK_N };
static const char *const pk_name[PK_N] = {"GEMV int8", "GEMV other", "tiled GEMM", "norm", "rope", "attention",
                                          "dn conv", "dn recurrence", "element-wise", "qsa", "ple", "copy",
                                          "mla", "mla weights", "dsa", "kda", "mhc", "attnres", "route", "experts"};
/* the MLA, KDA, mHC and AttnRes pipelines: optional (an engine without such layers
 * needs none of them, and a missing one turns off only its own ops) */
enum { PM_MLA, PM_HGEMV, PM_DSA, PM_KDA, PM_MHC, PM_ARES, PM_ROUTE, PM_N };
static const char *const mla_file[PM_N] = {"chain_mla.spv", "chain_hgemv.spv", "chain_dsa.spv", "chain_kda.spv",
                                           "chain_mhc.spv", "chain_ares.spv", "chain_route.spv"};
#define VKC_KDA_MAX 8

/* ---- memory: blocks per kind, buffers bound at offsets inside them -------------- */
typedef struct { VkDeviceMemory mem; uint8_t *map; } VkcBlock;
typedef struct { VkaPool p; uint32_t memtype; int mapped; } VkcPool;
struct VkcBuf {
    VkBuffer buf;
    VkDeviceMemory mem;               /* VKC_HOST: the imported pages */
    int kind;
    size_t bytes;
    VkaRange r;
    uint8_t *ptr;
    unsigned long long last_frame;   /* serial of the last frame that bound it */
    int d;                           /* the device (vkc_device) it was made on */
};

typedef struct {
    VkCommandBuffer cmd;
    VkFence fence;
    VkDescriptorPool *pools; int npools, cur_pool;
    VkcBuf **tmp; int ntmp, ctmp;     /* staging / temporaries, freed when the frame comes back */
    VkcBuf *stage; size_t stage_used;
    int inflight, open;
    unsigned long long serial;
    uint64_t wait_h, sig_g;           /* hybrid steps: the host's value this frame waits for, the value it signals */
    VkQueryPool qp; int nq; unsigned char kind[VKC_PROF_Q];   /* COLI_VK_CHAIN_PROF: a timestamp after each op */
    uint32_t shape[VKC_PROF_Q];   /* COLI_VK_CHAIN_PROF=2: a matmul's (fmt, O, I) slot in prof_sh, else 0 */
} VkcFrame;

typedef struct {
    int ready, lost;
    int misuse;                       /* a bind of the other device's buffer: the record fails */
    ColiVkCore core;
    VkDevice dev;
    VkQueue queue;
    uint64_t fence_ns;                /* a frame's wait before the device counts as lost (vkc_init) */
    VkDescriptorSetLayout dsl;
    VkPipelineLayout pl;
    VkShaderModule mod[P_NPIPE], mod_gemm, mod_dnrec;
    VkShaderModule mod_gv2; VkPipeline gv2; int gv2_floats;   /* chain_gemv2.comp: int8 decode GEMVs (COLI_VK_CHAIN_GEMV2), x staged up to gv2_floats */
    VkShaderModule mod_bd; VkPipeline bd;   /* qmatmul_bd.comp: the block formats (a GGUF's matrices, blockdec.glsl); none without the file */
    VkShaderModule mod_bdv; VkPipeline bdv[4][64][10]; uint8_t bdv_tried[4][64][10]; int bdv_xs;   /* chain_gemv_bd.comp: a pipeline per (rows, ggml type, staging size), made on first use */
    VkShaderModule mod_bdv8; VkPipeline bdv8[4][64][10]; uint8_t bdv8_tried[4][64][10];   /* chain_gemv_bd_i8.spv: the same in the CPU's integer form, by its codes' size */
    VkShaderModule mod_bdg; VkPipeline bdg[64]; uint8_t bdg_tried[64];   /* qmatmul_gemm_bd.comp: a prompt's rows, a pipeline per ggml type */
    VkShaderModule mod_fetch; VkPipeline fetch; uint8_t fetch_tried;      /* chain_fetch.comp (vkc_fetch) */
    VkShaderModule mod_moe; VkPipeline moe[2][10];            /* chain_moe.comp: the routed experts, by address (bufferDeviceAddress),
                                                                 [1] with the block formats' paths (BD = 1), by staging size */
    int moe_xs, moe_xs_bd; uint8_t moe_tried[2][10];          /* their largest staging floats (0: the device cannot); made once asked for */
    VkPipeline moe_f[3][10]; uint8_t moe_f_tried[3][10];      /* chain_moe.comp for one of our formats (FMTC 1, 2, 4), by staging size */
    VkShaderModule mod_moe_gv; VkPipeline moe_gv[3][10]; uint8_t moe_gv_tried[3][10];   /* chain_moe_gv.comp: a decode's passes in our formats */
    VkPipeline moe_gvq[10][17]; uint8_t moe_gvq_tried[10][17];   /* its int4 fast path (HQ), by staging and steps a lane */
    VkShaderModule mod_moe_i8; VkPipeline moe_i8[10]; uint8_t moe_i8_tried[10];   /* chain_moe_i8.spv, by its codes' size */
    struct { int8_t i8, b; uint8_t tg, td, tried; VkPipeline p; } moe_bt[32]; int moe_nbt;   /* the block formats' of one layer's types (BTG, BTD) */
    int idot, idot_set;                                       /* the integer form on (vkc_moe_set_idot: -1 = the device's default) */
    /* hybrid steps (vkc_hybrid_*): a timeline the device signals (its frames' sig_g) and one
     * the host signals (their wait_h), and the next frame's wait */
    VkSemaphore tsem_g, tsem_h; uint64_t next_wait_h;
    /* a prompt chunk's routed experts (vkc_moe_rows): chain_moe_rows.comp, and the block
     * formats' grouped GEMMs (qmatmul_grp_bd.comp: [0] down, [1] gate+up) a pipeline per
     * (pass, ggml type), made on first use */
    VkShaderModule mod_rows, mod_grp[2]; VkPipeline rows; VkPipeline grp[2][64]; uint8_t grp_tried[2][64];
    VkShaderModule mod_grpc[2]; VkPipeline grpc[2][64]; uint8_t grpc_tried[2][64];   /* qmatmul_grp*_bdc.comp: the same on the matrix units */
    VkShaderModule mod_grpw[2]; VkPipeline grpw[2][64]; uint8_t grpw_tried[2][64];   /* qmatmul_grp*_bdcw.comp: their wider item */
    VkShaderModule mod_bdgc; VkPipeline bdgc[64]; uint8_t bdgc_tried[64];             /* qmatmul_gemm_bdc.comp: a GGUF's prompt GEMM on them */
    VkShaderModule mod_bdgw; VkPipeline bdgw[64]; uint8_t bdgw_tried[64];             /* qmatmul_gemm_bdcw.comp: its 128 tile */
    VkShaderModule mod_grpq[2]; VkPipeline grpq[2]; uint8_t grpq_tried[2];      /* our formats' (1, 2, 4) in floats: qmatmul_grp_q.comp */
    VkShaderModule mod_grpqc[2]; VkPipeline grpqc[2]; uint8_t grpqc_tried[2];   /* and on the matrix units: qmatmul_grp_qc.comp */
    int rows_tried;
    int discrete;                     /* a discrete card: vkc_prewarm's default */
    int gv2_all;                      /* chain_gemv2 for every int8 decode matrix it takes (COLI_VK_CHAIN_GEMV2=1, or not NVIDIA) */
    VkPipeline pipe[P_NPIPE];
    VkPipeline gemm[VKC_GEMM_MAX]; int gemm_bm[VKC_GEMM_MAX], gemm_bn[VKC_GEMM_MAX], ngemm;
    VkPipeline dnrec[VKC_DNREC_MAX]; int dnrec_kd[VKC_DNREC_MAX], ndnrec;
    VkShaderModule mod_dnrr; int dnrr_tried;   /* chain_dnrec_rows.spv: a prompt chunk's rows, made on first use */
    VkPipeline dnrr[VKC_DNREC_MAX]; int dnrr_kd[VKC_DNREC_MAX], ndnrr;
    VkShaderModule mod_gemv4; VkPipeline gemv4; int gemv4_xs;   /* chain_gemv.comp: the vectorized decode GEMV */
    VkPipeline gemv4x;                /* chain_gemv.comp with x read from its buffer (a row past the staging) */
    VkPipeline gemv4n, gemv4xn;       /* the same taking up to 4 rows of x a workgroup (a verify's), NR 4 */
    VkShaderModule mod_adec; VkPipeline adec; int adec_tried; VkcBuf *adec_x;   /* chain_attn_dec.comp and its scratch */
    VkPipeline adg[VKC_ADG_MAX]; int adg_hd[VKC_ADG_MAX], adg_gq[VKC_ADG_MAX], adg_kv[VKC_ADG_MAX], nadg;   /* its grouped mode's */
    VkPipeline attn_h, adec_h; int attn_h_tried, adec_h_tried;   /* chain_attn and chain_attn_dec over a float16 cache (KV16) */
    VkShaderModule mod_mdec; VkPipeline mdec; int mdec_tried;   /* chain_mla_dec.comp (the same scratch) */
    /* chain_gemm.comp: int8/int4 prompt GEMMs, x rounded to f16 once (COLI_VK_CHAIN_GEMM) */
    VkShaderModule mod_tg; VkPipeline tg;
    VkShaderModule mmod[PM_N]; VkPipeline mpipe[PM_N]; int mla_ok;   /* chain_mla, chain_hgemv, chain_dsa */
    /* chain_attn_flash.comp: the attention core on the matrix units for prompt chunks,
     * one pipeline per (head dim, query heads per kv head) */
    VkShaderModule mod_fa; VkPipeline fa[VKC_FA_MAX]; int fa_hd[VKC_FA_MAX], fa_gq[VKC_FA_MAX], fa_kv[VKC_FA_MAX], nfa;
    VkPipeline kdarec[VKC_KDA_MAX]; int kdarec_kd[VKC_KDA_MAX], nkdarec;   /* chain_kda's recurrence per key dim */
    VkShaderModule mod_kdar; int kdar_tried;   /* chain_kda_rows.spv: a prompt chunk's rows, made on first use */
    VkPipeline kdar[VKC_KDA_MAX]; int kdar_kd[VKC_KDA_MAX], nkdar;
    VkCommandPool cpool;
    VkcFrame fr[VKC_FRAMES];
    int cur;                          /* the open frame, -1 none */
    unsigned long long serial;        /* frames begun so far */
    VkcPool pool[3];
    VkcBuf *dummy;
    size_t align;
    int gemm_rows;                    /* -1 the backend's rule, 0 never, else the rows */
    VkPipeline bound;
    /* hazard tracking since the last barrier */
    VkBuffer wr[VKC_TRACK], rd[VKC_TRACK]; int nwr, nrd;
    VkcStats st;
    int prof, kind; float ts_period; uint32_t ts_mask;
    double prof_ms[PK_N]; unsigned long long prof_n[PK_N];
    /* COLI_VK_CHAIN_PROF=2: the matmuls' time by (fmt, rows of x, O, I), the slots' keys */
    int prof_shapes; uint64_t prof_key[64]; double prof_sh_ms[64]; unsigned long long prof_sh_n[64];
} VkcCtx;
/* One context per device (vkc_device): 0 the primary, 1 COLI_VK_DEV2's, for the layers an
 * engine places there. Every call works on the current device's; a buffer remembers its
 * device, and binding or copying one of the other device's is refused. */
static VkcCtx g_kctx[2] = {{.cur = -1, .gemm_rows = -1}, {.cur = -1, .gemm_rows = -1}};
static int g_kd;
static int g_kvh;   /* the attention call in progress reads a float16 cache (vkc_attn_h) */
#define KC (g_kctx[g_kd])

/* a weight tensor the current device holds (one on the other device is refused) */
static int vkc_tensor_info(const ColiVkTensor *t, ColiVkTensorInfo *ti) {
    int d = 0;
    return coli_vk_tensor_info_dev(t, ti, &d) && d == g_kd;
}

int vkc_ready(void) { return KC.ready && !KC.lost; }
int vkc_device(int d) { int was = g_kd; g_kd = d == 1 ? 1 : 0; return was; }
int vkc_device_now(void) { return g_kd; }
void vkc_shutdown(void);
void vkc_shutdown_all(void) {   /* both devices' contexts, the second first (at exit) */
    int was = g_kd;
    g_kd = 1; vkc_shutdown();
    g_kd = 0; vkc_shutdown();
    g_kd = was;
}
int vkc_lost(void) { return KC.lost; }

static void lose(const char *what, VkResult r) {
    if (!KC.lost) fprintf(stderr, "[VK] chain: %s failed (%d): the device is lost\n", what, (int)r);
    KC.lost = 1;
    coli_vk_mark_lost_dev(g_kd);
}

/* ---- blocks ---------------------------------------------------------------------- */
#define VKC_BLOCK ((uint64_t)64 << 20)
static int pool_grow(VkcPool *P, uint64_t need) {
    uint64_t cap = vka_block_size_for(&P->p, need);
    VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = cap, .memoryTypeIndex = P->memtype};
#ifdef VK_EXT_memory_priority
    VkMemoryPriorityAllocateInfoEXT pri = {.sType = VK_STRUCTURE_TYPE_MEMORY_PRIORITY_ALLOCATE_INFO_EXT,
        .priority = 1.0f};   /* rides every chain submit: never evicted before weights */
    if (KC.core.has_prio) ai.pNext = &pri;
#endif
    VkcBlock *bk = calloc(1, sizeof *bk);
    if (!bk) return 0;
    if (vkAllocateMemory(KC.dev, &ai, NULL, &bk->mem) != VK_SUCCESS) { free(bk); return 0; }
    if (P->mapped && vkMapMemory(KC.dev, bk->mem, 0, cap, 0, (void **)&bk->map) != VK_SUCCESS) {
        vkFreeMemory(KC.dev, bk->mem, NULL); free(bk); return 0;
    }
    if (vka_pool_add_block(&P->p, cap, bk) < 0) {
        if (bk->map) vkUnmapMemory(KC.dev, bk->mem);
        vkFreeMemory(KC.dev, bk->mem, NULL); free(bk); return 0;
    }
    return 1;
}
static uint32_t memtype_of(int kind) {
    return kind == VKC_UP ? KC.core.memtype_host : kind == VKC_DOWN ? KC.core.memtype_cached : KC.core.memtype_dev;
}
static int memtype_host_visible(uint32_t t) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties((VkPhysicalDevice)KC.core.phys, &mp);
    return (mp.memoryTypes[t].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
}

static void track_reset(void) { KC.nwr = KC.nrd = 0; }
/* the hybrid steps' timeline functions (Vulkan 1.2 core, taken from the device, not the
 * loader's list: an older loader keeps the chain without them), the primary device's */
static PFN_vkCreateSemaphore p_vkCreateSemaphore; static PFN_vkDestroySemaphore p_vkDestroySemaphore;
static PFN_vkGetSemaphoreCounterValue p_vkGetSemaphoreCounterValue; static PFN_vkWaitSemaphores p_vkWaitSemaphores;
static PFN_vkSignalSemaphore p_vkSignalSemaphore;
static VkDevice g_hyb_dev; static VkSemaphore g_hyb_g, g_hyb_h;
static int hyb_load(VkDevice dev) {
    p_vkCreateSemaphore = (PFN_vkCreateSemaphore)vkGetDeviceProcAddr(dev, "vkCreateSemaphore");
    p_vkDestroySemaphore = (PFN_vkDestroySemaphore)vkGetDeviceProcAddr(dev, "vkDestroySemaphore");
    p_vkGetSemaphoreCounterValue = (PFN_vkGetSemaphoreCounterValue)vkGetDeviceProcAddr(dev, "vkGetSemaphoreCounterValue");
    p_vkWaitSemaphores = (PFN_vkWaitSemaphores)vkGetDeviceProcAddr(dev, "vkWaitSemaphores");
    p_vkSignalSemaphore = (PFN_vkSignalSemaphore)vkGetDeviceProcAddr(dev, "vkSignalSemaphore");
    return p_vkCreateSemaphore && p_vkDestroySemaphore && p_vkGetSemaphoreCounterValue && p_vkWaitSemaphores && p_vkSignalSemaphore;
}
static void zero_now(VkcBuf *b);

VkcBuf *vkc_buf(size_t bytes, int kind) {
    if (!vkc_ready() || kind < 0 || kind > 2) return NULL;
    if (!bytes) bytes = 4;
    bytes = (bytes + 3) & ~(size_t)3;
    if (KC.core.ssbo_range && bytes > KC.core.ssbo_range) return NULL;   /* one binding could not address it */
    VkcBuf *b = calloc(1, sizeof *b);
    if (!b) return NULL;
    b->kind = kind; b->bytes = bytes; b->d = g_kd;
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = bytes,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                 VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,   /* vkc_moe_rows' dispatches */
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    if (vkCreateBuffer(KC.dev, &bi, NULL, &b->buf) != VK_SUCCESS) { free(b); return NULL; }
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(KC.dev, b->buf, &req);
    VkcPool *P = &KC.pool[kind];
    uint64_t align = req.alignment > KC.align ? req.alignment : KC.align;
    if (!(req.memoryTypeBits & (1u << P->memtype)) ||
        (!vka_alloc(&P->p, req.size, align, &b->r) &&
         !(pool_grow(P, req.size + align) && vka_alloc(&P->p, req.size, align, &b->r)))) {
        vkDestroyBuffer(KC.dev, b->buf, NULL); free(b); return NULL;
    }
    VkcBlock *bk = P->p.b[b->r.block].user;
    if (vkBindBufferMemory(KC.dev, b->buf, bk->mem, b->r.off) != VK_SUCCESS) {
        if (vka_free(&P->p, b->r)) { VkcBlock *e = P->p.b[b->r.block].user;
            if (e) { if (e->map) vkUnmapMemory(KC.dev, e->mem); vkFreeMemory(KC.dev, e->mem, NULL); free(e); }
            vka_pool_drop_block(&P->p, b->r.block); }
        vkDestroyBuffer(KC.dev, b->buf, NULL); free(b); return NULL;
    }
    b->ptr = bk->map ? bk->map + b->r.off : NULL;
    KC.st.dev_bytes += bytes;
    if (b->ptr) memset(b->ptr, 0, bytes);
    else zero_now(b);
    return b;
}

static void buf_release(VkcBuf *b) {
    if (b->kind == VKC_HOST) { coli_vk_host_buffer_free((void *)b->buf, (void *)b->mem, b->bytes); free(b); return; }
    if (b->kind == VKC_ADDR) {
        vkDestroyBuffer(KC.dev, b->buf, NULL); vkFreeMemory(KC.dev, b->mem, NULL);
        KC.st.dev_bytes -= b->bytes; free(b); return;
    }
    VkcPool *P = &KC.pool[b->kind];
    vkDestroyBuffer(KC.dev, b->buf, NULL);
    if (vka_free(&P->p, b->r)) {
        VkcBlock *e = P->p.b[b->r.block].user;
        if (e) { if (e->map) vkUnmapMemory(KC.dev, e->mem); vkFreeMemory(KC.dev, e->mem, NULL); free(e); }
        vka_pool_drop_block(&P->p, b->r.block);
    }
    KC.st.dev_bytes -= b->bytes;
    free(b);
}

/* a frame that may still read b must finish first */
static int frame_wait(VkcFrame *f);
static void vkc_free_here(VkcBuf *b) {
    if (!KC.dev) { free(b); return; }
    if (!KC.lost)
        for (int i = 0; i < VKC_FRAMES; i++) {
            VkcFrame *f = &KC.fr[i];
            if ((f->inflight || f->open) && f->serial == b->last_frame) {
                if (f->open) vkc_submit(1); else frame_wait(f);
            }
        }
    for (int i = 0; i < KC.nwr; i++) if (KC.wr[i] == b->buf) KC.wr[i] = VK_NULL_HANDLE;
    for (int i = 0; i < KC.nrd; i++) if (KC.rd[i] == b->buf) KC.rd[i] = VK_NULL_HANDLE;
    buf_release(b);
}
void vkc_free(VkcBuf *b) {   /* on the device that made it, whichever is current */
    if (!b) return;
    int was = g_kd;
    g_kd = b->d;
    vkc_free_here(b);
    g_kd = was;
}
int vkc_reserve(VkcBuf **b, size_t bytes, int kind) {
    if (*b && (*b)->bytes >= bytes && (*b)->kind == kind) return 1;
    size_t want = bytes;
    if (*b) { want = (*b)->bytes + (*b)->bytes / 2; if (want < bytes) want = bytes; vkc_free(*b); }
    *b = vkc_buf(want, kind);
    return *b != NULL;
}
void *vkc_ptr(const VkcBuf *b) { return b ? b->ptr : NULL; }
size_t vkc_bytes(const VkcBuf *b) { return b ? b->bytes : 0; }
VkcBuf *vkc_buf_addr(size_t bytes, uint64_t *addr) {
    if (!vkc_ready() || g_kd || !KC.core.bda || !addr) return NULL;
    bytes = (bytes + 15) & ~(size_t)15;
    if (!bytes) bytes = 16;
    if (KC.core.ssbo_range && bytes > KC.core.ssbo_range) return NULL;   /* bound as one too */
    VkcBuf *b = calloc(1, sizeof *b);
    if (!b) return NULL;
    b->kind = VKC_ADDR; b->bytes = bytes; b->d = g_kd;
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = bytes,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                 VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    if (vkCreateBuffer(KC.dev, &bi, NULL, &b->buf) != VK_SUCCESS) { free(b); return NULL; }
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(KC.dev, b->buf, &req);
    VkMemoryAllocateFlagsInfo fl = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
        .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT};
    VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .pNext = &fl,
        .allocationSize = req.size, .memoryTypeIndex = KC.core.memtype_dev};
    if (!(req.memoryTypeBits & (1u << KC.core.memtype_dev)) || vkAllocateMemory(KC.dev, &ai, NULL, &b->mem) != VK_SUCCESS) {
        vkDestroyBuffer(KC.dev, b->buf, NULL); free(b); return NULL;
    }
    if (vkBindBufferMemory(KC.dev, b->buf, b->mem, 0) != VK_SUCCESS) {
        vkFreeMemory(KC.dev, b->mem, NULL); vkDestroyBuffer(KC.dev, b->buf, NULL); free(b); return NULL;
    }
    VkBufferDeviceAddressInfo da = {.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .buffer = b->buf};
    *addr = (uint64_t)vkGetBufferDeviceAddress(KC.dev, &da);
    KC.st.dev_bytes += bytes;
    return b;
}
VkcBuf *vkc_host(const void *ptr, size_t bytes, size_t *off) {
    /* the primary device's import: on the second device's chain the host's part of a
     * split stays on the CPU (vkc_kv_shadow says so) */
    if (!vkc_ready() || !ptr || !bytes || g_kd) return NULL;
    void *buf = NULL, *mem = NULL;
    size_t o = 0;
    if (!coli_vk_host_buffer(ptr, bytes, &buf, &mem, &o)) return NULL;
    size_t al = coli_vk_import_alignment(), sz = (o + bytes + al - 1) / al * al;   /* the pages, as imported */
    VkcBuf *b = calloc(1, sizeof *b);
    if (!b) { coli_vk_host_buffer_free(buf, mem, sz); return NULL; }
    b->kind = VKC_HOST; b->buf = (VkBuffer)buf; b->mem = (VkDeviceMemory)mem;
    b->bytes = sz;
    *off = o;
    return b;
}

/* ---- init ------------------------------------------------------------------------ */
/* The length of a path's directory with its separator, 0 without one: the shaders'
 * directory from qmatmul.spv's path (a backslash ends it too on Windows). */
static size_t dir_prefix(const char *path) {
    const char *sl = strrchr(path, '/');
#ifdef _WIN32
    const char *bs = strrchr(path, '\\');
    if (bs && (!sl || bs > sl)) sl = bs;
#endif
    return sl ? (size_t)(sl - path) + 1 : 0;
}
static VkShaderModule load_module(const char *dir_spv, const char *file) {
    char path[1200];
    size_t pre = dir_prefix(dir_spv);
    if (pre + strlen(file) + 1 >= sizeof path) return VK_NULL_HANDLE;
    memcpy(path, dir_spv, pre); strcpy(path + pre, file);
    FILE *f = fopen(path, "rb");
    if (!f) return VK_NULL_HANDLE;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    if (n <= 0 || n % 4) { fclose(f); return VK_NULL_HANDLE; }
    uint32_t *code = malloc((size_t)n);
    if (!code || fread(code, 1, (size_t)n, f) != (size_t)n) { fclose(f); free(code); return VK_NULL_HANDLE; }
    fclose(f);
    VkShaderModuleCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = (size_t)n, .pCode = code};
    VkShaderModule m = VK_NULL_HANDLE;
    if (vkCreateShaderModule(KC.dev, &ci, NULL, &m) != VK_SUCCESS) m = VK_NULL_HANDLE;
    free(code);
    if (!m) fprintf(stderr, "[VK] chain: cannot load %s\n", path);
    return m;
}
/* The chain's shaders size their work by gl_SubgroupSize and gl_SubgroupID, so each
 * pipeline runs at the subgroup size the device reports (ColiVkCore.pin_sg), not one
 * the compiler picks: an Intel Iris Xe (subgroups of 8 to 32) compiled chain_gemv and
 * chain_gemv2 at another width than the shader saw, and every decode GEMV came out
 * wrong. 0 (no subgroup size control, COLI_VK_SUBGROUP=0): the driver's choice. */
static VkPipeline make_pipe(VkShaderModule m, const VkSpecializationInfo *si) {
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT rss = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO_EXT,
        .requiredSubgroupSize = (uint32_t)KC.core.pin_sg};
    VkComputePipelineCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .pNext = KC.core.pin_sg > 0 ? &rss : NULL,
                  .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = m, .pName = "main", .pSpecializationInfo = si},
        .layout = KC.pl};
    VkPipeline p = VK_NULL_HANDLE;
    if (vkCreateComputePipelines(KC.dev, VK_NULL_HANDLE, 1, &ci, NULL, &p) != VK_SUCCESS) return VK_NULL_HANDLE;
    return p;
}

/* A pipeline of the matrix units' shaders: at the device's cooperative-matrix subgroup
 * size (64 or 32), required, in full subgroups; NULL elsewhere or when it fails. */
static VkPipeline coop_pipe(VkShaderModule m, const VkSpecializationInfo *si) {
    int sg = KC.core.coop_sg;
    if (!m || (sg != 64 && sg != 32)) return VK_NULL_HANDLE;
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT rss = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO_EXT,
        .requiredSubgroupSize = (uint32_t)sg};
    VkComputePipelineCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .pNext = &rss,
                  .flags = VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT_EXT,
                  .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = m, .pName = "main", .pSpecializationInfo = si},
        .layout = KC.pl};
    VkPipeline p = VK_NULL_HANDLE;
    if (vkCreateComputePipelines(KC.dev, VK_NULL_HANDLE, 1, &ci, NULL, &p) != VK_SUCCESS) return VK_NULL_HANDLE;
    return p;
}
/* COLI_VK_<name>=0 keeps a family of matrix-unit shaders off */
static int coop_on(const char *env) {
    const char *e = getenv(env);
    return (KC.core.coop_sg == 64 || KC.core.coop_sg == 32) && !(e && *e == '0');
}

static void dsv4_init(void);       /* chain_dsv4.comp (DeepSeek V4.1 / V4), below */
static void dsv4_shutdown(void);
static void kvs_shutdown(void);    /* chain_kvs.comp (the split KV cache), below */
/* The compute shared memory a workgroup may declare: the device's
 * maxComputeSharedMemorySize, or COLI_VK_SHARED_LIMIT (tests: Lavapipe reports 32 KiB of
 * what is host memory, and the op tests run the 38 KiB full attention there). */
static uint32_t kc_shared_limit(void) {
    const char *e = getenv("COLI_VK_SHARED_LIMIT");
    if (e && *e) return (uint32_t)strtoul(e, NULL, 10);
    VkPhysicalDeviceProperties pp; vkGetPhysicalDeviceProperties((VkPhysicalDevice)KC.core.phys, &pp);
    return pp.limits.maxComputeSharedMemorySize;
}

/* qmatmul_gemm_bd.comp's fixed shared memory, held to the compiled shader's by
 * tests/test_vk_shared_memory.py: a device with less keeps the GEMV for a prompt's rows */
#define GEMM_BD_SHARED 44544u
/* and qmatmul_gemm_bdc.comp's (its tiles, its partial-tile patch and bdstep's tables): a device with
 * less keeps the floats' GEMM there */
#define GEMM_BDC_SHARED 33344u
/* and qmatmul_gemm_bdcw.comp's, its 128 tile (no IQ grid in its tables) */
#define GEMM_BDCW_SHARED 41056u
/* and qmatmul_grp*_bdcw.comp's, the experts' wider item (its tiles, patches and tables) */
#define GRP_BDCW_SHARED 41536u
/* an ASan or UBSan build (gcc defines __SANITIZE_ADDRESS__; clang has __has_feature) */
#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_UNDEFINED__)
#define VKC_SANITIZED 1
#elif defined(__has_feature)
#if __has_feature(address_sanitizer) || __has_feature(undefined_behavior_sanitizer)
#define VKC_SANITIZED 1
#endif
#endif
#ifndef VKC_SANITIZED
#define VKC_SANITIZED 0
#endif
int vkc_init(void) {
    if (KC.ready) return !KC.lost;
    if (!coli_vk_core_dev(g_kd, &KC.core)) return 0;
    KC.dev = (VkDevice)KC.core.device; KC.queue = (VkQueue)KC.core.queue;
    /* 20 s tells a GPU that hangs. A software device (llvmpipe) generates a shader's code
     * at its first launch, inside the frame's wait, and caches it on disk: on a CI runner's
     * empty cache chain_moe's first frame took more than 20 s (4 s on a laptop), and the
     * device was given up for a compile. There a frame waits up to 5 minutes, and 15 in a
     * sanitized build: under ASan the first frame of a GGUF's chain (chain_moe's block
     * formats and the GEMVs of five types) went past the 5 minutes on a CI runner. */
    {
        VkPhysicalDeviceProperties pp; vkGetPhysicalDeviceProperties((VkPhysicalDevice)KC.core.phys, &pp);
        KC.fence_ns = pp.deviceType != VK_PHYSICAL_DEVICE_TYPE_CPU ? 20000000000ULL
                    : VKC_SANITIZED ? 900000000000ULL : 300000000000ULL;
        KC.discrete = pp.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
        /* chain_gemv2 takes every int8 decode matrix but on NVIDIA, where chain_gemv.comp
         * was faster for all but a few long rows (matmul_in) */
        const char *g2 = getenv("COLI_VK_CHAIN_GEMV2");
        KC.gv2_all = (g2 && *g2 == '1') || pp.vendorID != 0x10DE;
    }
    KC.align = KC.core.ssbo_align > 16 ? KC.core.ssbo_align : 16;
    VkDescriptorSetLayoutBinding b[VKC_BIND];
    for (int i = 0; i < VKC_BIND; i++) b[i] = (VkDescriptorSetLayoutBinding){.binding = (uint32_t)i,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT};
    VkDescriptorSetLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = VKC_BIND, .pBindings = b};
    VkPushConstantRange pr = {.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = 128};
    if (vkCreateDescriptorSetLayout(KC.dev, &li, NULL, &KC.dsl) != VK_SUCCESS) return 0;
    VkPipelineLayoutCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1, .pSetLayouts = &KC.dsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &pr};
    if (vkCreatePipelineLayout(KC.dev, &pi, NULL, &KC.pl) != VK_SUCCESS) return 0;
    for (int i = 0; i < P_NPIPE; i++) {
        KC.mod[i] = load_module(KC.core.spv_path, pipe_file[i]);
        if (!KC.mod[i] || !(KC.pipe[i] = make_pipe(KC.mod[i], NULL))) {
            fprintf(stderr, "[VK] chain: shader %s unavailable, the chain stays off\n", pipe_file[i]);
            return 0;
        }
    }
    KC.mod_dnrec = load_module(KC.core.spv_path, "chain_dnrec.spv");
    if (!KC.mod_dnrec) return 0;
    /* the vectorized decode GEMV, its x staging as large as the device's shared memory
     * allows (16 KiB floats at most); COLI_VK_CHAIN_GEMV=0 keeps qmatmul.comp's */
    {
        const char *e = getenv("COLI_VK_CHAIN_GEMV");
        VkPhysicalDeviceProperties pp; vkGetPhysicalDeviceProperties((VkPhysicalDevice)KC.core.phys, &pp);
        int xs = (int)((pp.limits.maxComputeSharedMemorySize - 512) / 16);   /* past the subgroups' 32 vec4 sums */
        if (xs > 4096) xs = 4096;
        if (!(e && *e == '0') && xs >= 256 && (KC.mod_gemv4 = load_module(KC.core.spv_path, "chain_gemv.spv"))) {
            /* and its variant for a row past the staging, x read from its buffer */
            VkSpecializationMapEntry me[3] = {{0, 0, 4}, {1, 4, 4}, {2, 8, 4}};
            int32_t v[3] = {xs, 0, 1}, vx[3] = {xs, 1, 1}, vn[3] = {xs, 0, 4}, vxn[3] = {xs, 1, 4};
            VkSpecializationInfo si = {3, me, sizeof v, v}, six = {3, me, sizeof vx, vx},
                                 sin = {3, me, sizeof vn, vn}, sixn = {3, me, sizeof vxn, vxn};
            if ((KC.gemv4 = make_pipe(KC.mod_gemv4, &si))) KC.gemv4_xs = xs;
            KC.gemv4x = KC.gemv4 ? make_pipe(KC.mod_gemv4, &six) : VK_NULL_HANDLE;
            /* a verify's rows: COLI_VK_CHAIN_GEMV_NR=1 keeps them a workgroup each */
            const char *nr = getenv("COLI_VK_CHAIN_GEMV_NR");
            if (!(nr && *nr == '1')) {
                KC.gemv4n = KC.gemv4 ? make_pipe(KC.mod_gemv4, &sin) : VK_NULL_HANDLE;
                KC.gemv4xn = KC.gemv4x ? make_pipe(KC.mod_gemv4, &sixn) : VK_NULL_HANDLE;
            }
        }
    }
    /* the int8/int4 decode GEMV with lanes per row and rows per workgroup chosen per
     * matrix (chain_gemv2.comp): needs clustered subgroup operations. Its x staging is
     * sized from the device's shared memory as chain_gemv's is: a fixed 64 KiB lost the
     * device on a 48 KiB GPU (#1998, a Quadro P2000), where it now stages 48 KiB and a
     * matrix whose rows of x do not fit takes chain_gemv. */
    {
        const char *e = getenv("COLI_VK_CHAIN_GEMV2");
        VkPhysicalDeviceSubgroupProperties sgp = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
        VkPhysicalDeviceProperties2 p2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &sgp};
        vkGetPhysicalDeviceProperties2((VkPhysicalDevice)KC.core.phys, &p2);
        int xs = (int)(kc_shared_limit() / 16);   /* vec4s */
        if (xs > 4096) xs = 4096;
        int ok = (sgp.supportedOperations & VK_SUBGROUP_FEATURE_CLUSTERED_BIT) && (sgp.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) &&
                 sgp.subgroupSize >= 16 && xs >= 256 && !(e && *e == '0');
        if (ok && (KC.mod_gv2 = load_module(KC.core.spv_path, "chain_gemv2.spv"))) {
            int32_t v = xs;
            VkSpecializationMapEntry me = {0, 0, 4};
            VkSpecializationInfo si = {1, &me, 4, &v};
            if ((KC.gv2 = make_pipe(KC.mod_gv2, &si))) {
                KC.gv2_floats = 4 * xs;
                if (xs < 4096)
                    fprintf(stderr, "[VK] chain: chain_gemv2 stages x in %d KiB of shared memory, the device's limit\n", xs / 64);
            }
        }
    }
    /* the block formats' GEMV (a GGUF's dense matrices): qmatmul_bd.spv beside the chain's
     * shaders, optional; without it a block-format matrix is refused and the engine's chain
     * declines that layer */
    if ((KC.mod_bd = load_module(KC.core.spv_path, "qmatmul_bd.spv"))) KC.bd = make_pipe(KC.mod_bd, NULL);
    /* and the types chain_gemv_bd.comp decodes on its own, x staged in what shared memory
     * allows past its 64-byte table (16 KiB floats at most); COLI_VK_CHAIN_GEMV_BD=0 keeps
     * qmatmul_bd.comp's for every type */
    {
        const char *e = getenv("COLI_VK_CHAIN_GEMV_BD");
        VkPhysicalDeviceProperties pp; vkGetPhysicalDeviceProperties((VkPhysicalDevice)KC.core.phys, &pp);
        int xs = (int)((pp.limits.maxComputeSharedMemorySize - 64 - 2176 * 4) / 16);   /* past bdstep.glsl's tables */
        if (xs > 4096) xs = 4096;
        if (KC.bd && !(e && *e == '0') && xs >= 256 && (KC.mod_bdv = load_module(KC.core.spv_path, "chain_gemv_bd.spv")))
            KC.bdv_xs = xs;
        /* the tiled GEMM for a prompt's rows: its tiles and bdstep's tables, 43.5 KiB */
        const char *g = getenv("COLI_VK_CHAIN_GEMM_BD");
        if (KC.bd && !(g && *g == '0') && pp.limits.maxComputeSharedMemorySize >= GEMM_BD_SHARED)
            KC.mod_bdg = load_module(KC.core.spv_path, "qmatmul_gemm_bd.spv");
    }
    /* the routed experts by address (chain_moe.comp): a device with bufferDeviceAddress
     * and clustered subgroup adds; the input rows staged in what shared memory allows */
    VkPhysicalDeviceSubgroupProperties msg = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
    VkPhysicalDeviceProperties2 mp2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &msg};
    vkGetPhysicalDeviceProperties2((VkPhysicalDevice)KC.core.phys, &mp2);
    /* built at the first vkc_moe_ready (moe_build): only an engine that runs its experts
     * in the frame pays for it. Built here for every engine, it cost a software device an
     * LLVM compile whose memory the process kept: GLM's resident set at exit, +100 MiB on
     * Lavapipe, no longer fell by what its dense weights gave back */
    if (KC.core.bda && (msg.supportedOperations & VK_SUBGROUP_FEATURE_CLUSTERED_BIT) &&
        (msg.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT)) {
        int32_t xs = (int32_t)(kc_shared_limit() / 4) & ~3;
        KC.moe_xs = xs > 8192 ? 8192 : xs >= 1024 ? xs : 0;
        /* the block formats' pipeline: past bdstep.glsl's tables (2192 words) */
        xs = (int32_t)((kc_shared_limit() > 8768 ? kc_shared_limit() - 8768 : 0) / 4) & ~3;
        KC.moe_xs_bd = xs > 8192 ? 8192 : xs >= 1024 ? xs : 0;
    }
    KC.idot = g_kd == 0 ? coli_vk_idot() : 0; KC.idot_set = -1;
    { const char *ie = getenv("COLI_VK_IDOT"); if (ie && *ie == '1') KC.idot_set = 1; }   /* where supported, not only accelerated (tests) */
    /* the MLA, KDA and mHC shaders, optional: without one only its ops decline */
    for (int i = 0; i < PM_N; i++) {
        char path[1200];
        size_t pre = dir_prefix(KC.core.spv_path);
        FILE *f = NULL;
        if (pre + strlen(mla_file[i]) + 1 < sizeof path) {
            memcpy(path, KC.core.spv_path, pre); strcpy(path + pre, mla_file[i]);
            f = fopen(path, "rb");
        }
        if (f) fclose(f);
        if (f && (KC.mmod[i] = load_module(KC.core.spv_path, mla_file[i]))) KC.mpipe[i] = make_pipe(KC.mmod[i], NULL);
    }
    KC.mla_ok = KC.mpipe[PM_MLA] && KC.mpipe[PM_HGEMV] && KC.mpipe[PM_DSA];
    /* the attention core on the matrix units (vkc_attn_flash_rows): cooperative matrices at
     * subgroups of 64 or 32, the shader's tiling four subgroups' (the first four of eight at 32) */
    if (KC.core.coop_sg == 64 || KC.core.coop_sg == 32) KC.mod_fa = load_module(KC.core.spv_path, "chain_attn_flash.spv");
    dsv4_init();
    /* the int8/int4 prompt GEMM on the matrix units (chain_gemm.comp, 256 invocations:
     * four subgroups of 64, or eight of 32 as NVIDIA's cooperative matrices run);
     * COLI_VK_CHAIN_GEMM=0 leaves those formats to the GEMMs below */
    {
        const char *e = getenv("COLI_VK_CHAIN_GEMM");
        if ((KC.core.coop_sg == 64 || KC.core.coop_sg == 32) && !(e && *e == '0') &&
            (KC.mod_tg = load_module(KC.core.spv_path, "chain_gemm.spv"))) {
            int32_t tt = 4;
            VkSpecializationMapEntry me = {0, 0, 4};
            VkSpecializationInfo si = {1, &me, 4, &tt};
            VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT rss = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO_EXT,
                .requiredSubgroupSize = (uint32_t)KC.core.coop_sg};
            VkComputePipelineCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .pNext = &rss,
                          .flags = VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT_EXT,
                          .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = KC.mod_tg, .pName = "main",
                          .pSpecializationInfo = &si}, .layout = KC.pl};
            if (vkCreateComputePipelines(KC.dev, VK_NULL_HANDLE, 1, &ci, NULL, &KC.tg) != VK_SUCCESS) KC.tg = VK_NULL_HANDLE;
        }
    }
    /* the fp32 tiled GEMM at the backend's tiles; none = every S on the GEMV */
    KC.mod_gemm = KC.core.gemm_tiles ? load_module(KC.core.spv_path, "qmatmul_gemm.spv") : VK_NULL_HANDLE;
    for (int k = 0; KC.mod_gemm && k < KC.core.gemm_tiles && k < VKC_GEMM_MAX; k++) {
        const int *t = KC.core.gemm_tile[k];
        int32_t sv[7] = {t[0], t[1], t[2], t[3], t[4], (t[0] / t[3]) * (t[1] / t[4]), t[5] ? 1 : 0};
        VkSpecializationMapEntry me[7];
        for (int i = 0; i < 7; i++) me[i] = (VkSpecializationMapEntry){(uint32_t)i, (uint32_t)(i * 4), 4};
        VkSpecializationInfo si = {7, me, sizeof sv, sv};
        if (!(KC.gemm[k] = make_pipe(KC.mod_gemm, &si))) break;
        KC.gemm_bm[k] = t[0]; KC.gemm_bn[k] = t[1]; KC.ngemm = k + 1;
    }
    VkCommandPoolCreateInfo cp = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = KC.core.qfam};
    if (vkCreateCommandPool(KC.dev, &cp, NULL, &KC.cpool) != VK_SUCCESS) return 0;
    for (int i = 0; i < VKC_FRAMES; i++) {
        VkCommandBufferAllocateInfo ca = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = KC.cpool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
        VkFenceCreateInfo fi = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        if (vkAllocateCommandBuffers(KC.dev, &ca, &KC.fr[i].cmd) != VK_SUCCESS ||
            vkCreateFence(KC.dev, &fi, NULL, &KC.fr[i].fence) != VK_SUCCESS) return 0;
    }
    if (g_kd == 0 && coli_vk_timeline() && hyb_load(KC.dev)) {
        VkSemaphoreTypeCreateInfo st = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO, .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE};
        VkSemaphoreCreateInfo sci = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = &st};
        if (p_vkCreateSemaphore(KC.dev, &sci, NULL, &KC.tsem_g) != VK_SUCCESS || p_vkCreateSemaphore(KC.dev, &sci, NULL, &KC.tsem_h) != VK_SUCCESS) {
            if (KC.tsem_g) p_vkDestroySemaphore(KC.dev, KC.tsem_g, NULL);
            KC.tsem_g = KC.tsem_h = VK_NULL_HANDLE;
        }
        g_hyb_dev = KC.dev; g_hyb_g = KC.tsem_g; g_hyb_h = KC.tsem_h;
    }
    {
        const char *e = getenv("COLI_VK_CHAIN_PROF");
        uint32_t nqf = 0; vkGetPhysicalDeviceQueueFamilyProperties((VkPhysicalDevice)KC.core.phys, &nqf, NULL);
        VkQueueFamilyProperties qf[16]; if (nqf > 16) nqf = 16;
        vkGetPhysicalDeviceQueueFamilyProperties((VkPhysicalDevice)KC.core.phys, &nqf, qf);
        VkPhysicalDeviceProperties pp; vkGetPhysicalDeviceProperties((VkPhysicalDevice)KC.core.phys, &pp);
        uint32_t bits = KC.core.qfam < nqf ? qf[KC.core.qfam].timestampValidBits : 0;
        if (e && (*e == '1' || *e == '2') && bits && pp.limits.timestampPeriod > 0) {
            KC.prof = *e == '2' ? 2 : 1; KC.ts_period = pp.limits.timestampPeriod;   /* 2: the matmuls by shape too */
            KC.ts_mask = bits >= 32 ? 0xffffffffu : (1u << bits) - 1;
            for (int i = 0; i < VKC_FRAMES; i++) {
                VkQueryPoolCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
                    .queryType = VK_QUERY_TYPE_TIMESTAMP, .queryCount = VKC_PROF_Q};
                if (vkCreateQueryPool(KC.dev, &qi, NULL, &KC.fr[i].qp) != VK_SUCCESS) KC.prof = 0;
            }
        }
    }
    for (int k = 0; k < 3; k++) {
        KC.pool[k].memtype = memtype_of(k);
        KC.pool[k].mapped = memtype_host_visible(KC.pool[k].memtype);
        vka_pool_init(&KC.pool[k].p, coli_vk_block_bytes_dev(g_kd, VKC_BLOCK), 0);   /* smaller under COLI_VK_DEVICE_CAP_MB */
    }
    if (!KC.pool[VKC_UP].mapped || !KC.pool[VKC_DOWN].mapped) return 0;
    KC.ready = 1;
    KC.dummy = vkc_buf(256, VKC_DEV);
    if (!KC.dummy) { KC.ready = 0; return 0; }
    return 1;
}

/* ---- frames ---------------------------------------------------------------------- */
static void frame_reclaim(VkcFrame *f) {
    for (int i = 0; i < f->ntmp; i++) buf_release(f->tmp[i]);
    f->ntmp = 0;
    f->stage_used = 0;
    for (int i = 0; i < f->npools; i++) vkResetDescriptorPool(KC.dev, f->pools[i], 0);
    f->cur_pool = 0;
}
/* A frame is milliseconds of work and the thread waiting on it has nothing else to do:
 * poll the fence for up to COLI_VK_CHAIN_SPIN_US before blocking, whose wake-up costs 50 to
 * 150 us per frame, two frames a layer. On a discrete card 50 ms, a whole decode step: at 2 ms
 * the host slept through each step's last frames and woke after the device had run dry
 * (Qwen3.6 UD-Q4_K_M on an RTX PRO 4000 decoded 76.5 tok/s, 90.7 polling to the end; the
 * device idle about 1 ms a token between steps). An integrated GPU shares the package's power
 * with the cores: 2 ms there. */
static long chain_spin_us(void) {
    static long v = -1;
    if (v < 0) {
        const char *e = getenv("COLI_VK_CHAIN_SPIN_US");
        VkPhysicalDeviceProperties pp; vkGetPhysicalDeviceProperties((VkPhysicalDevice)KC.core.phys, &pp);
        v = e && *e ? atol(e) : pp.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? 50000 : 2000;
        if (v < 0) v = 0;
    }
    return v;
}
static int frame_wait(VkcFrame *f) {
    if (!f->inflight) return 1;
    double t0 = vkc_now_ms();
    VkResult r = VK_NOT_READY;
    for (long spin = chain_spin_us(); spin > 0 && r == VK_NOT_READY && (vkc_now_ms() - t0) * 1000.0 < spin; )
        r = vkGetFenceStatus(KC.dev, f->fence);
    if (r == VK_NOT_READY) r = vkWaitForFences(KC.dev, 1, &f->fence, VK_TRUE, KC.fence_ns);
    KC.st.wait_ms += vkc_now_ms() - t0; KC.st.waits++;
    f->inflight = 0;
    if (r != VK_SUCCESS) { lose("fence wait", r); return 0; }
    if (KC.prof && f->qp && f->nq > 1) {
        uint64_t ts[VKC_PROF_Q];
        if (vkGetQueryPoolResults(KC.dev, f->qp, 0, (uint32_t)f->nq, sizeof ts, ts, sizeof ts[0],
                                  VK_QUERY_RESULT_64_BIT) == VK_SUCCESS)
            for (int i = 1; i < f->nq; i++) {
                uint64_t d = (ts[i] - ts[i - 1]) & (KC.ts_mask == 0xffffffffu ? ~0ull : (uint64_t)KC.ts_mask);
                KC.prof_ms[f->kind[i]] += (double)d * KC.ts_period / 1e6; KC.prof_n[f->kind[i]]++;
                if (f->shape[i]) { KC.prof_sh_ms[f->shape[i] - 1] += (double)d * KC.ts_period / 1e6; KC.prof_sh_n[f->shape[i] - 1]++; }
            }
        f->nq = 0;
    }
    frame_reclaim(f);
    return 1;
}
static uint64_t g_prof_shape;   /* the matmul being recorded: (fmt, S, O, I), 0 outside one */
static void prof_mark(VkcFrame *f, int kind) {
    if (!KC.prof || !f->qp || f->nq >= VKC_PROF_Q) return;
    vkCmdWriteTimestamp(f->cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, f->qp, (uint32_t)f->nq);
    uint32_t slot = 0;
    if (KC.prof == 2 && g_prof_shape) {
        int k = 0;
        while (k < KC.prof_shapes && KC.prof_key[k] != g_prof_shape) k++;
        if (k == KC.prof_shapes && k < 64) KC.prof_key[KC.prof_shapes++] = g_prof_shape;
        if (k < 64) slot = (uint32_t)k + 1;
    }
    f->shape[f->nq] = slot;
    f->kind[f->nq++] = (unsigned char)kind;
}
int vkc_finish(void) {
    if (!KC.ready) return 0;
    if (KC.cur >= 0 && !vkc_submit(0)) return 0;
    for (int i = 0; i < VKC_FRAMES; i++) if (!frame_wait(&KC.fr[i])) return 0;
    return !KC.lost;
}

static void barrier_now(VkCommandBuffer c, VkPipelineStageFlags dst_stage, VkAccessFlags dst) {
    VkMemoryBarrier mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, .dstAccessMask = dst};
    vkCmdPipelineBarrier(c, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, dst_stage,
                         0, 1, &mb, 0, NULL, 0, NULL);
    KC.st.barriers++;
}
static const VkAccessFlags ALL_RW = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                                    VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
#define ALL_STAGES (VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT)

int vkc_begin(void) {
    /* the backend found the device lost (a staged upload's fence): this frame fails as a
     * lost device's would, and the engine takes over on the CPU */
    if (KC.ready && !KC.lost && !coli_vk_available_dev(g_kd)) lose("the backend's device", VK_ERROR_DEVICE_LOST);
    if (!vkc_ready()) return 0;
    if (KC.cur >= 0) return 1;                     /* already open */
    int i = (int)(KC.serial % VKC_FRAMES);
    VkcFrame *f = &KC.fr[i];
    if (!frame_wait(f)) return 0;
    frame_reclaim(f);
    if (vkResetCommandBuffer(f->cmd, 0) != VK_SUCCESS) { lose("command buffer reset", VK_ERROR_DEVICE_LOST); return 0; }
    VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    vkBeginCommandBuffer(f->cmd, &bi);
    f->serial = ++KC.serial;
    f->open = 1;
    KC.cur = i;
    KC.bound = VK_NULL_HANDLE;
    f->wait_h = KC.next_wait_h; f->sig_g = 0; KC.next_wait_h = 0;
    /* after everything submitted earlier on this queue (and the host's writes) */
    barrier_now(f->cmd, ALL_STAGES, ALL_RW);
    if (f->wait_h) {   /* and the host's writes made after the submission, before its signal (vkc_hybrid_signal) */
        VkMemoryBarrier hb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_HOST_WRITE_BIT, .dstAccessMask = ALL_RW};
        vkCmdPipelineBarrier(f->cmd, VK_PIPELINE_STAGE_HOST_BIT, ALL_STAGES, 0, 1, &hb, 0, NULL, 0, NULL);
    }
    track_reset();
    if (KC.prof && f->qp) {
        vkCmdResetQueryPool(f->cmd, f->qp, 0, VKC_PROF_Q);
        f->nq = 0;
        vkCmdWriteTimestamp(f->cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, f->qp, 0);
        f->kind[0] = 0; f->nq = 1;
    }
    return 1;
}
int vkc_submit(int wait) {
    if (KC.cur < 0) return vkc_ready();
    VkcFrame *f = &KC.fr[KC.cur];
    barrier_now(f->cmd, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT);
    vkEndCommandBuffer(f->cmd);
    f->open = 0; KC.cur = -1;
    if (KC.lost) return 0;
    {   /* COLI_VK_CHAIN_FAULT=n (tests): the primary device's n-th submission fails as a lost
         * device would; COLI_VK_CHAIN_FAULT2 the second device's */
        static long fault[2] = {-2, -2};
        if (fault[g_kd] == -2) { const char *e = getenv(g_kd ? "COLI_VK_CHAIN_FAULT2" : "COLI_VK_CHAIN_FAULT"); fault[g_kd] = e && *e ? atol(e) : -1; }
        if (fault[g_kd] > 0 && (long)KC.st.frames + 1 >= fault[g_kd])
            { lose(g_kd ? "queue submit (COLI_VK_CHAIN_FAULT2)" : "queue submit (COLI_VK_CHAIN_FAULT)", VK_ERROR_DEVICE_LOST); return 0; }
    }
    vkResetFences(KC.dev, 1, &f->fence);
    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &f->cmd};
    /* a hybrid step's frame: it waits for the host's value, or signals the device's */
    VkTimelineSemaphoreSubmitInfo tsi = {.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
    VkPipelineStageFlags wst = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    if (f->wait_h || f->sig_g) {
        si.pNext = &tsi;
        if (f->wait_h) { tsi.waitSemaphoreValueCount = 1; tsi.pWaitSemaphoreValues = &f->wait_h;
                         si.waitSemaphoreCount = 1; si.pWaitSemaphores = &KC.tsem_h; si.pWaitDstStageMask = &wst; }
        if (f->sig_g) { tsi.signalSemaphoreValueCount = 1; tsi.pSignalSemaphoreValues = &f->sig_g;
                        si.signalSemaphoreCount = 1; si.pSignalSemaphores = &KC.tsem_g; }
    }
    VkResult r = (VkResult)coli_vk_queue_submit_dev(g_kd, KC.queue, &si, f->fence);   /* the backend's lock, staged uploads */
    if (r != VK_SUCCESS) { lose("queue submit", r); return 0; }
    f->inflight = 1;
    KC.st.frames++;
    return wait ? frame_wait(f) : 1;
}

/* ---- hybrid steps (vk_chain.h) ---------------------------------------------------- */
int vkc_hybrid_ready(void) { return vkc_ready() && KC.tsem_g && KC.tsem_h; }
int vkc_hybrid_submit(uint64_t g) {
    if (KC.cur < 0 || !KC.tsem_g) return 0;
    KC.fr[KC.cur].sig_g = g;
    return vkc_submit(0);
}
void vkc_hybrid_wait_next(uint64_t h) { KC.next_wait_h = h; }
/* the host's side, from any thread: the primary device's two timelines */
int vkc_hybrid_host_wait(uint64_t g) {
    if (!g_hyb_dev) return 0;
    uint64_t v = 0;
    for (int i = 0; i < 20000; i++) {   /* a few hundred microseconds spinning, then the driver's wait */
        if (p_vkGetSemaphoreCounterValue(g_hyb_dev, g_hyb_g, &v) != VK_SUCCESS) return 0;
        if (v >= g) return 1;
    }
    VkSemaphoreWaitInfo wi = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO, .semaphoreCount = 1, .pSemaphores = &g_hyb_g, .pValues = &g};
    return p_vkWaitSemaphores(g_hyb_dev, &wi, 60ull * 1000000000ull) == VK_SUCCESS;
}
int vkc_hybrid_host_signal(uint64_t h) {
    if (!g_hyb_dev) return 0;
    VkSemaphoreSignalInfo si = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO, .semaphore = g_hyb_h, .value = h};
    return p_vkSignalSemaphore(g_hyb_dev, &si) == VK_SUCCESS;
}

/* Measured on an RTX PRO 4000 (Qwen3.6-35B-A3B int4-g64, every expert on the device, 128
 * tokens): one frame a forward 44.4 tok/s, a frame every 1, 2 or 4 layers 58.3, every 8
 * 52.0, every 20 48.8; the same text in every case. */
int vkc_split_layers(void) {
    static int n = -1;
    if (n < 0) { const char *e = getenv("COLI_VK_CHAIN_SPLIT"); n = e && *e ? atoi(e) : 4; if (n < 0) n = 0; }
    return n;
}
int vkc_layer_recorded(int *layers) {
    const int every = vkc_split_layers();
    if (every < 1 || ++*layers < every) return 1;
    *layers = 0;
    return vkc_submit(0) && vkc_begin();
}

/* ---- recording helpers ----------------------------------------------------------- */
static VkcFrame *open_frame(void) { return KC.cur >= 0 ? &KC.fr[KC.cur] : NULL; }

static int in_set(const VkBuffer *s, int n, VkBuffer b) { for (int i = 0; i < n; i++) if (s[i] == b) return 1; return 0; }
/* a barrier before an op reading rd[] and writing wr[] when it conflicts with the ops
 * recorded since the last one */
static void hazard(VkcFrame *f, const VkBuffer *rd, int nr, const VkBuffer *wr, int nw) {
    int need = 0;
    for (int i = 0; i < nr && !need; i++) need = in_set(KC.wr, KC.nwr, rd[i]);
    for (int i = 0; i < nw && !need; i++) need = in_set(KC.wr, KC.nwr, wr[i]) || in_set(KC.rd, KC.nrd, wr[i]);
    if (need || KC.nwr + nw > VKC_TRACK || KC.nrd + nr > VKC_TRACK) {
        barrier_now(f->cmd, ALL_STAGES, ALL_RW);
        track_reset();
    }
    for (int i = 0; i < nr; i++) if (!in_set(KC.rd, KC.nrd, rd[i])) KC.rd[KC.nrd++] = rd[i];
    for (int i = 0; i < nw; i++) if (!in_set(KC.wr, KC.nwr, wr[i])) KC.wr[KC.nwr++] = wr[i];
}

static VkDescriptorSet alloc_set(VkcFrame *f) {
    for (;;) {
        if (f->cur_pool < f->npools) {
            VkDescriptorSetAllocateInfo da = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                .descriptorPool = f->pools[f->cur_pool], .descriptorSetCount = 1, .pSetLayouts = &KC.dsl};
            VkDescriptorSet s;
            if (vkAllocateDescriptorSets(KC.dev, &da, &s) == VK_SUCCESS) return s;
            f->cur_pool++;
            continue;
        }
        VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 512 * VKC_BIND};
        VkDescriptorPoolCreateInfo pc = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .maxSets = 512, .poolSizeCount = 1, .pPoolSizes = &ps};
        VkDescriptorPool *n = realloc(f->pools, (size_t)(f->npools + 1) * sizeof *n);
        if (!n) return VK_NULL_HANDLE;
        f->pools = n;
        if (vkCreateDescriptorPool(KC.dev, &pc, NULL, &f->pools[f->npools]) != VK_SUCCESS) return VK_NULL_HANDLE;
        f->npools++;
    }
}

/* One dispatch. bufs[i] NULL: the dummy; offs/ranges in bytes (range 0 = to the end). */
typedef struct { VkBuffer buf; VkDeviceSize off, range; int written; } VkcBind;
static int record(VkPipeline pipe, const VkcBind *bd, int n, const void *pc, size_t pcb,
                  uint32_t gx, uint32_t gy, uint32_t gz) {
    VkcFrame *f = open_frame();
    if (KC.misuse) { KC.misuse = 0; return 0; }
    if (!f || KC.lost || !gx || !gy || !gz) return f && !KC.lost;
    VkBuffer rd[VKC_BIND], wr[VKC_BIND]; int nr = 0, nw = 0;
    VkDescriptorBufferInfo bi[VKC_BIND];
    VkWriteDescriptorSet w[VKC_BIND];
    VkDescriptorSet set = alloc_set(f);
    if (!set) { fprintf(stderr, "[VK] chain: out of descriptor sets\n"); return 0; }
    for (int i = 0; i < VKC_BIND; i++) {
        if (i < n && bd[i].buf) {
            bi[i] = (VkDescriptorBufferInfo){bd[i].buf, bd[i].off, bd[i].range ? bd[i].range : VK_WHOLE_SIZE};
            if (bd[i].written) wr[nw++] = bd[i].buf; else rd[nr++] = bd[i].buf;
        } else bi[i] = (VkDescriptorBufferInfo){KC.dummy->buf, 0, VK_WHOLE_SIZE};
        w[i] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set,
            .dstBinding = (uint32_t)i, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &bi[i]};
    }
    vkUpdateDescriptorSets(KC.dev, VKC_BIND, w, 0, NULL);
    hazard(f, rd, nr, wr, nw);
    if (KC.bound != pipe) { vkCmdBindPipeline(f->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe); KC.bound = pipe; }
    vkCmdBindDescriptorSets(f->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, KC.pl, 0, 1, &set, 0, NULL);
    if (pcb) vkCmdPushConstants(f->cmd, KC.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, (uint32_t)pcb, pc);
    vkCmdDispatch(f->cmd, gx, gy, gz);
    KC.st.ops++;
    prof_mark(f, KC.kind);
    return 1;
}
static VkcBind B(VkcBuf *b, int written) {
    if (b && b->d != g_kd) {   /* the other device's: the record that binds it fails */
        if (!KC.misuse) fprintf(stderr, "[VK] chain: a buffer of device %d bound on device %d\n", b->d, g_kd);
        KC.misuse = 1;
        return (VkcBind){VK_NULL_HANDLE, 0, 0, written};
    }
    if (b) b->last_frame = KC.serial;
    return (VkcBind){b ? b->buf : VK_NULL_HANDLE, 0, 0, written};
}
/* groups over x, spilling into y past the 65535 limit */
static void grid(uint64_t groups, uint32_t *gx, uint32_t *gy) {
    if (!groups) { *gx = *gy = 0; return; }
    uint64_t x = groups < 65535 ? groups : 65535;
    *gx = (uint32_t)x; *gy = (uint32_t)((groups + x - 1) / x);
}

/* ---- transfers ------------------------------------------------------------------- */
static int copy_bytes(VkcBuf *dst, size_t doff, VkcBuf *src, size_t soff, size_t bytes) {
    VkcFrame *f = open_frame();
    if (!f || KC.lost) return 0;
    if (!bytes) return 1;
    if (doff + bytes > dst->bytes || soff + bytes > src->bytes) { fprintf(stderr, "[VK] chain: copy out of range\n"); return 0; }
    if (dst->d != g_kd || src->d != g_kd) { fprintf(stderr, "[VK] chain: a copy across devices\n"); return 0; }
    VkBuffer r = src->buf, w = dst->buf;
    hazard(f, &r, 1, &w, 1);
    VkBufferCopy c = {soff, doff, bytes};
    vkCmdCopyBuffer(f->cmd, src->buf, dst->buf, 1, &c);
    prof_mark(f, PK_COPY);
    src->last_frame = dst->last_frame = KC.serial;
    return 1;
}
int vkc_barrier(void) {
    VkcFrame *f = open_frame();
    if (!f || KC.lost) return 0;
    barrier_now(f->cmd, ALL_STAGES, ALL_RW);
    track_reset();
    return 1;
}
/* chain_fetch.comp: a workgroup per slice of a region, the regions along y */
#define VKC_FETCH_SLICE 65536   /* words a workgroup: 256 KiB */
int vkc_fetch(VkcBuf *reg, int first, int n, uint32_t max_words, VkcBuf *dst) {
    if (!vkc_ready() || !KC.core.bda || !reg || !dst || first < 0 || n < 0) return 0;
    if (!n) return 1;
    if ((size_t)(first + n) * 16 > reg->bytes) return 0;
    if (!KC.fetch_tried) {
        KC.fetch_tried = 1;
        if ((KC.mod_fetch = load_module(KC.core.spv_path, "chain_fetch.spv"))) KC.fetch = make_pipe(KC.mod_fetch, NULL);
    }
    if (!KC.fetch) return 0;
    KC.kind = PK_COPY;
    VkcBind bd[2] = {B(reg, 0), B(dst, 1)};
    struct { int first, n, slice; } pc = {first, n, VKC_FETCH_SLICE};
    uint32_t gx = (max_words + VKC_FETCH_SLICE - 1) / VKC_FETCH_SLICE;
    return record(KC.fetch, bd, 2, &pc, sizeof pc, gx ? gx : 1, (uint32_t)n, 1);
}
int vkc_copy(VkcBuf *dst, size_t doff, VkcBuf *src, size_t soff, size_t n) {
    return copy_bytes(dst, doff * 4, src, soff * 4, n * 4);
}
/* ---- a grid of regions in one dispatch (chain_copy2d.comp), made on first use ----
 * A prompt chunk's K and V rows go into the head-major cache as a region per (row, head):
 * a copy command with 16384 regions of 512 bytes took 3.3 ms on an RTX PRO 4000 (OLMoE,
 * a chunk of 1024 rows, four of them a layer). Regions on a grid (one length; the
 * destination and the source moving by fixed steps along an inner and an outer index)
 * take one dispatch from VKC_CP2_MIN regions; any other list keeps the copy command. A
 * shorter list takes one dispatch when it is a grid whole: a decode step's 16 regions of 512
 * bytes (OLMoE's heads) took 8.3 us as a copy command, 1.46 us as a dispatch. A list that is
 * no grid keeps the copy command rather than many small dispatches. */
#define VKC_CP2_MIN 64
static struct { VkShaderModule mod; VkPipeline pipe, pipe_h; int tried, tried_h; } g_cp2[2];
#define CP2 (g_cp2[g_kd])
static VkPipeline cp2_pipe(void) {
    if (CP2.pipe || CP2.tried || !vkc_ready()) return CP2.pipe;
    CP2.tried = 1;
    if ((CP2.mod = load_module(KC.core.spv_path, "chain_copy2d.spv"))) CP2.pipe = make_pipe(CP2.mod, NULL);
    return CP2.pipe;
}
/* its H16 twin (vkc_copy_regions_h): into a float16 cache */
static VkPipeline cp2h_pipe(void) {
    if (CP2.pipe_h || CP2.tried_h || !cp2_pipe()) return CP2.pipe_h;
    CP2.tried_h = 1;
    static const VkSpecializationMapEntry me = {0, 0, 4};
    static const VkBool32 on = VK_TRUE;
    VkSpecializationInfo si = {1, &me, 4, &on};
    return CP2.pipe_h = make_pipe(CP2.mod, &si);
}
static void cp2_shutdown(void) {
    if (CP2.pipe_h) vkDestroyPipeline(KC.dev, CP2.pipe_h, NULL);
    if (CP2.pipe) vkDestroyPipeline(KC.dev, CP2.pipe, NULL);
    if (CP2.mod) vkDestroyShaderModule(KC.dev, CP2.mod, NULL);
    memset(&CP2, 0, sizeof CP2);
}
typedef struct { uint32_t d0, s0; int32_t dh, sh, dq, sq; uint32_t len, per, n; } VkcCp2;
/* the regions' grid, when they are one: region i = (q, h) = (i / per, i % per) */
static int cp2_grid(const VkcRegion *r, int n, VkcCp2 *g, int min) {
    if (n < min || n < 2 || !r[0].n) return 0;
    long long L = (long long)r[0].n, d0 = (long long)r[0].dst, s0 = (long long)r[0].src;
    long long dh = (long long)r[1].dst - d0, sh = (long long)r[1].src - s0;
    int per = 1;
    while (per < n && (long long)r[per].dst - (long long)r[per - 1].dst == dh && (long long)r[per].src - (long long)r[per - 1].src == sh) per++;
    if (n % per) return 0;
    int Q = n / per;
    long long dq = Q > 1 ? (long long)r[per].dst - d0 : 0, sq = Q > 1 ? (long long)r[per].src - s0 : 0;
    const long long I32 = 2147483647LL;
    if (llabs(dh) * (per - 1) + llabs(dq) * (Q - 1) > I32 || llabs(sh) * (per - 1) + llabs(sq) * (Q - 1) > I32) return 0;
    if ((long long)n * L > 4294967295LL) return 0;
    for (int i = 0; i < n; i++) {
        long long q = i / per, h = i % per;
        if ((long long)r[i].n != L || (long long)r[i].dst != d0 + q * dq + h * dh || (long long)r[i].src != s0 + q * sq + h * sh) return 0;
        if ((long long)r[i].dst + L > 4294967295LL || (long long)r[i].src + L > 4294967295LL) return 0;
    }
    *g = (VkcCp2){(uint32_t)d0, (uint32_t)s0, (int32_t)dh, (int32_t)sh, (int32_t)dq, (int32_t)sq, (uint32_t)L, (uint32_t)per, (uint32_t)n};
    return 1;
}
/* how many of the regions from r[0] make one grid, in whole rows of it (cuda_chain.cu's
 * cc_grid_run): a ring's rows break where the ring wraps, and each side is a grid */
static int cp2_run(const VkcRegion *r, int n) {
    if (n < 2) return n;
    long long L = (long long)r[0].n, d0 = (long long)r[0].dst, s0 = (long long)r[0].src;
    long long dh = (long long)r[1].dst - d0, sh = (long long)r[1].src - s0;
    int per = 1;
    while (per < n && (long long)r[per].n == L && (long long)r[per].dst - (long long)r[per - 1].dst == dh &&
           (long long)r[per].src - (long long)r[per - 1].src == sh) per++;
    if (per == n) return n;
    long long dq = (long long)r[per].dst - d0, sq = (long long)r[per].src - s0;
    int Q = 1;
    for (; (Q + 1) * per <= n; Q++) {
        int ok = 1;
        for (int h = 0; h < per && ok; h++) {
            const VkcRegion *x = &r[Q * per + h];
            ok = (long long)x->n == L && (long long)x->dst == d0 + Q * dq + h * dh && (long long)x->src == s0 + Q * sq + h * sh;
        }
        if (!ok) break;
    }
    return Q * per;
}
int vkc_copy_regions(VkcBuf *dst, VkcBuf *src, const VkcRegion *r, int n) {
    VkcFrame *f = open_frame();
    if (!f || KC.lost) return 0;
    if (n < 1) return 1;
    for (int i = 0; i < n; i++)
        if ((r[i].dst + r[i].n) * 4 > dst->bytes || (r[i].src + r[i].n) * 4 > src->bytes) {
            fprintf(stderr, "[VK] chain: copy region out of range\n"); return 0;
        }
    VkcCp2 g;
    const char *e = getenv("COLI_VK_COPY_GRID");   /* 0: every list through the copy command */
    if (dst != src && !(e && *e == '0') && n >= 2 && n < VKC_CP2_MIN && cp2_pipe() && cp2_grid(r, n, &g, 2)) {
        KC.kind = PK_COPY;   /* a short list that is one grid */
        VkcBind bd[2] = {B(src, 0), B(dst, 1)};
        uint32_t gx, gy; grid(((uint64_t)g.n * g.len + 255) / 256, &gx, &gy);
        if (!record(CP2.pipe, bd, 2, &g, sizeof g, gx, gy, 1)) return 0;
        KC.st.copy_grids++;
        return 1;
    }
    const int grids = dst != src && !(e && *e == '0') && n >= VKC_CP2_MIN && cp2_pipe();
    VkBufferCopy *c = NULL;
    int nc = 0;
    for (int i = 0; i < n; ) {   /* each grid of VKC_CP2_MIN regions or more in a launch, the rest in one copy */
        int k = grids ? cp2_run(r + i, n - i) : n - i;
        if (grids && k >= VKC_CP2_MIN && cp2_grid(r + i, k, &g, VKC_CP2_MIN)) {
            KC.kind = PK_COPY;
            VkcBind bd[2] = {B(src, 0), B(dst, 1)};
            uint32_t gx, gy; grid(((uint64_t)g.n * g.len + 255) / 256, &gx, &gy);
            if (!record(CP2.pipe, bd, 2, &g, sizeof g, gx, gy, 1)) { free(c); return 0; }
            KC.st.copy_grids++;
            i += k;
            continue;
        }
        if (!c && !(c = malloc((size_t)(n - i) * sizeof *c))) return 0;
        for (int j = i; j < i + k; j++) c[nc++] = (VkBufferCopy){r[j].src * 4, r[j].dst * 4, r[j].n * 4};
        i += k;
    }
    if (!nc) { free(c); return 1; }
    n = nc;
    f = open_frame();   /* a launch above may have closed the frame */
    if (!f) { free(c); return 0; }
    VkBuffer rb = src->buf, wb = dst->buf;
    hazard(f, &rb, 1, &wb, 1);
    vkCmdCopyBuffer(f->cmd, src->buf, dst->buf, (uint32_t)n, c);
    prof_mark(f, PK_COPY);
    src->last_frame = dst->last_frame = KC.serial;
    free(c);
    return 1;
}
/* float32 regions of src (offsets and lengths in floats) into a float16 cache (in halves, even),
 * each value rounded to the nearest float16: chain_copy2d's H16 twin, a grid a launch and each
 * region outside one a launch of its own */
int vkc_copy_regions_h(VkcBuf *dst, VkcBuf *src, const VkcRegion *r, int n) {
    VkcFrame *f = open_frame();
    if (!f || KC.lost || dst == src) return 0;
    if (n < 1) return 1;
    VkPipeline hp = cp2h_pipe();
    if (!hp) return 0;
    for (int i = 0; i < n; i++)
        if ((r[i].dst + r[i].n) * 2 > dst->bytes || (r[i].src + r[i].n) * 4 > src->bytes || (r[i].dst | r[i].n) % 2) {
            fprintf(stderr, "[VK] chain: float16 copy region out of range\n"); return 0;
        }
    for (int i = 0; i < n; ) {
        int k = cp2_run(r + i, n - i);
        VkcCp2 g;
        if (!cp2_grid(r + i, k, &g, 2)) {
            k = 1;
            g = (VkcCp2){(uint32_t)r[i].dst, (uint32_t)r[i].src, 0, 0, 0, 0, (uint32_t)r[i].n, 1, 1};
        }
        KC.kind = PK_COPY;
        VkcBind bd[2] = {B(src, 0), B(dst, 1)};
        uint32_t gx, gy; grid(((uint64_t)g.n * (g.len / 2) + 255) / 256, &gx, &gy);
        if (g.n && g.len && !record(hp, bd, 2, &g, sizeof g, gx, gy, 1)) return 0;
        i += k;
    }
    return 1;
}
/* vkc_attn and vkc_attn_w over a float16 cache (kc, vc and their offsets in halves) */
int vkc_attn_h(VkcBuf *q, VkcBuf *kc, VkcBuf *vc, VkcBuf *o, VkcBuf *gate, VkcBuf *sel, const VkcAttn *p) {
    g_kvh = 1;
    int ok = vkc_attn(q, kc, vc, o, gate, sel, p);
    g_kvh = 0;
    return ok;
}
int vkc_attn_wh(VkcBuf *q, VkcBuf *kc, VkcBuf *vc, VkcBuf *o, VkcBuf *gate, VkcBuf *sel, VkcBuf *snk, const VkcAttnW *p) {
    g_kvh = 1;
    int ok = vkc_attn_w(q, kc, vc, o, gate, sel, snk, p);
    g_kvh = 0;
    return ok;
}
/* the device takes a float16 cache: the grid copy's twin made (every attention shader has one) */
int vkc_kvh_ready(void) { return vkc_ready() && cp2h_pipe() != VK_NULL_HANDLE; }
int vkc_zero(VkcBuf *dst, size_t off, size_t n) {
    VkcFrame *f = open_frame();
    if (!f || KC.lost) return 0;
    if (!n) return 1;
    if ((off + n) * 4 > dst->bytes) return 0;
    VkBuffer w = dst->buf;
    hazard(f, NULL, 0, &w, 1);
    vkCmdFillBuffer(f->cmd, dst->buf, off * 4, n * 4, 0);
    dst->last_frame = KC.serial;
    return 1;
}
static void zero_now(VkcBuf *b) {
    /* a device-only buffer starts zeroed too: its own little frame */
    int was_open = KC.cur >= 0;
    if (!was_open && !vkc_begin()) return;
    vkc_zero(b, 0, b->bytes / 4);
    if (!was_open) vkc_submit(1);
}
int vkc_write(VkcBuf *dst, size_t off, const void *src, size_t bytes) {
    VkcFrame *f = open_frame();
    if (!f || KC.lost) return 0;
    if (!bytes) return 1;
    /* always through the staging, even into host-visible memory: the copy runs in
     * frame order, after the ops recorded before it (a direct memcpy would not) */
    size_t at = (f->stage_used + 15) & ~(size_t)15;
    if (!f->stage || at + bytes > f->stage->bytes) {
        /* a new staging buffer; the old one stays alive until the frame comes back */
        size_t want = f->stage ? 2 * f->stage->bytes : VKC_STAGE0;
        while (want < bytes) want *= 2;
        VkcBuf *n = vkc_buf(want, VKC_UP);
        if (!n) return 0;
        if (f->stage) {
            if (f->ntmp == f->ctmp) {
                int c = f->ctmp ? 2 * f->ctmp : 8;
                VkcBuf **t = realloc(f->tmp, (size_t)c * sizeof *t);
                if (!t) { buf_release(n); return 0; }
                f->tmp = t; f->ctmp = c;
            }
            f->tmp[f->ntmp++] = f->stage;
        }
        f->stage = n; at = 0;
    }
    memcpy(f->stage->ptr + at, src, bytes);
    f->stage_used = at + bytes;
    KC.st.bytes_up += bytes;
    return copy_bytes(dst, off * 4, f->stage, at, bytes);
}
int vkc_read(VkcBuf *src, size_t off, void *dst, size_t bytes) {
    if (!vkc_ready()) return 0;
    if (!bytes) return 1;
    if (!vkc_finish()) return 0;
    if (src->ptr && src->kind != VKC_UP) {        /* the device wrote it into host-readable memory */
        memcpy(dst, src->ptr + off * 4, bytes);
        KC.st.bytes_down += bytes;
        return 1;
    }
    VkcBuf *rb = vkc_buf(bytes, VKC_DOWN);
    if (!rb) return 0;
    int ok = vkc_begin() && copy_bytes(rb, 0, src, off * 4, bytes) && vkc_submit(1);
    if (ok) { memcpy(dst, rb->ptr, bytes); KC.st.bytes_down += bytes; }
    vkc_free(rb);
    return ok;
}

/* ---- matmul ---------------------------------------------------------------------- */
void vkc_gemm_rows(int rows) { KC.gemm_rows = rows; }
struct VkcPC { int fmt, S, I, O, rowWords, gs; };
/* Staging sizes: a pipeline that stages x in the smallest of these (vec4s) holding its
 * rows, not in what the device allows, takes the shared memory it uses, and more of its
 * workgroups fit an SM. Measured on an RTX PRO 4000 (2 workgroups an SM at the device's
 * 48 KiB, 4 at 12 KiB of x and bdstep.glsl's 8.5 KiB of tables): chain_moe at Qwen3.8's
 * and Qwen3.6's shapes 10-30% faster for every format and 1 to 3 rows (IQ2_S/Q2_0 at
 * D 2560: 112 -> 96, 207 -> 167, 311 -> 243 us a layer); chain_gemv_bd only for one row of
 * the IQ2 grids (bdv_pipe). */
static const int kc_stage[9] = {256, 512, 768, 1024, 1536, 2048, 2560, 3072, 4096};
#define KC_STAGES 9
static int stage_of(int need, int cap) {   /* a bucket: the smallest that holds need, KC_STAGES = the cap */
    for (int b = 0; b < KC_STAGES; b++) if (kc_stage[b] >= need && kc_stage[b] < cap) return b;
    return KC_STAGES;
}
static int stage_vec4(int b, int cap) { return b < KC_STAGES ? kc_stage[b] : cap; }
/* the integer form's sizes for a row's codes (16 inputs a uvec4): the smallest of these
 * that holds the row; KC_STAGES past them (the form declines) */
static const int kc_codes[9] = {16, 32, 64, 128, 160, 192, 256, 384, 512};
static int codes_of(int I) {
    int need = (I + 15) / 16;
    for (int b = 0; b < KC_STAGES; b++) if (kc_codes[b] >= need) return b;
    return KC_STAGES;
}
/* A block-format matrix (fmt >= 100, a GGUF's): qmatmul_bd.comp, its table at binding 2,
 * one row of outputs per workgroup row as the expert batch dispatches it; every row's
 * products as a decode step's. */
/* chain_gemv_bd.comp's pipeline for ggml type t, nr rows of x (1..4) and inputs of I, made
 * on first use; NULL for a type it does not decode (Q4_K and Q5_K also want rows of whole
 * 16-byte steps) */
static VkPipeline bdv_pipe(int t, int nr, int rowWords, int I) {
    static const int types[] = {8, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 29, 42};
    int on = 0;
    for (size_t k = 0; k < sizeof types / sizeof *types; k++) on |= types[k] == t;
    if (!on || !KC.mod_bdv || nr < 1 || nr > 4 || ((t == 12 || t == 13) && rowWords % 4)) return VK_NULL_HANDLE;
    /* the smallest staging only for one row of IQ2_XXS, IQ2_XS, IQ3_XXS or IQ2_S, whose decoders wait on
     * their grid reads: twice the workgroups an SM hide them, 10240 x 2560 at 61.2 -> 48.4
     * and 66.4 -> 52.0 us. The others read at the memory's rate, and more workgroups cost
     * them: IQ4_XS and IQ4_NL 3-4% for one row, Q8_0 40% for two. */
    int i4 = nr * (I / 4), b = nr == 1 && (t == 16 || t == 17 || t == 18 || t == 22) ? stage_of(i4 + i4 / 8, KC.bdv_xs) : KC_STAGES;
    if (!KC.bdv_tried[nr - 1][t][b]) {
        KC.bdv_tried[nr - 1][t][b] = 1;
        int32_t v[3] = {stage_vec4(b, KC.bdv_xs), t, nr};
        VkSpecializationMapEntry me[3] = {{0, 0, 4}, {1, 4, 4}, {2, 8, 4}};
        VkSpecializationInfo si = {3, me, sizeof v, v};
        KC.bdv[nr - 1][t][b] = make_pipe(KC.mod_bdv, &si);
    }
    return KC.bdv[nr - 1][t][b];
}
/* chain_gemv_bd_i8.spv's pipeline (the integer form, where it is on: moe_i8_pipe's rule) for
 * ggml type t, nr rows of x and inputs of I: their codes in the smallest kc_codes bucket of
 * uvec4s that holds a row, within the device's shared memory. NULL: the float form. */
static int moe_i8_type(int fmt);
static VkPipeline bdv8_pipe(int t, int nr, int rowWords, int I) {
    if (!(KC.idot_set >= 0 ? KC.idot_set && KC.idot : KC.idot == 2) || !KC.mod_bd || nr < 1 || nr > 4 || I % 32 ||
        !moe_i8_type(100 + t) || ((t == 12 || t == 13) && rowWords % 4)) return VK_NULL_HANDLE;
    /* IQ4_XS and IQ4_NL look their values up: one row ran slower so (10240 x 2560 on an RTX
     * PRO 4000, 44.2 -> 46.8 us), two or more faster (three 80.8 -> 58.0) */
    if ((t == 20 || t == 23) && nr < 2 && KC.idot_set < 0) return VK_NULL_HANDLE;
    int b = codes_of(I);
    if (b >= KC_STAGES) return VK_NULL_HANDLE;
    int xq = kc_codes[b];
    size_t sh = (size_t)nr * (xq + xq / 8 + 1) * 16 + (size_t)nr * (xq / 2 + 1) * 4 + (size_t)nr * (xq + 1) * 4 + 8768 + 64 + 64;
    if (sh > kc_shared_limit()) return VK_NULL_HANDLE;
    if (!KC.bdv8_tried[nr - 1][t][b]) {
        KC.bdv8_tried[nr - 1][t][b] = 1;
        if (!KC.mod_bdv8) KC.mod_bdv8 = load_module(KC.core.spv_path, "chain_gemv_bd_i8.spv");
        if (KC.mod_bdv8) {
            int32_t v[4] = {4, t, nr, xq};
            VkSpecializationMapEntry me[4] = {{0, 0, 4}, {1, 4, 4}, {2, 8, 4}, {3, 12, 4}};
            VkSpecializationInfo si = {4, me, sizeof v, v};
            KC.bdv8[nr - 1][t][b] = make_pipe(KC.mod_bdv8, &si);
        }
    }
    return KC.bdv8[nr - 1][t][b];
}
int vkc_gemm_bd_ready(void) { return vkc_ready() && KC.mod_bdg != VK_NULL_HANDLE; }
/* qmatmul_gemm_bdcw.comp's pipeline for ggml type t: the matrix units' GEMM at 128 x 128, for the
 * types with no IQ grid in shared memory (its tables hold four words); made on first use,
 * COLI_VK_GEMM_WIDE=0 never */
static VkPipeline bdgw_pipe(int t) {
    static int on = -1;
    if (on < 0) { const char *e = getenv("COLI_VK_GEMM_WIDE"); on = !(e && *e == '0'); }
    if (!on || t < 0 || t >= 64 || t == 16 || t == 17 || t == 18 || t == 21 || t == 22 || !KC.bdgc[t]) return VK_NULL_HANDLE;
    if (!KC.bdgw_tried[t]) {
        KC.bdgw_tried[t] = 1;
        if (kc_shared_limit() < GEMM_BDCW_SHARED) return VK_NULL_HANDLE;
        if (!KC.mod_bdgw) KC.mod_bdgw = load_module(KC.core.spv_path, "qmatmul_gemm_bdcw.spv");
        int32_t v[2] = {t, KC.core.coop_sg};
        VkSpecializationMapEntry me[2] = {{0, 0, 4}, {1, 4, 4}};
        VkSpecializationInfo si = {2, me, sizeof v, v};
        if (KC.mod_bdgw) KC.bdgw[t] = coop_pipe(KC.mod_bdgw, &si);
    }
    return KC.bdgw[t];
}
/* qmatmul_gemm_bd.comp's pipeline for ggml type t, made on first use */
static VkPipeline bdg_pipe(int t, int rowWords) {
    if (t < 0 || t >= 64 || !(t == 8 || t == 10 || t == 11 || t == 12 || t == 13 || t == 14 || t == 16 || t == 17 || t == 18 || t == 19 ||
        t == 20 || t == 21 || t == 22 || t == 23 || t == 29 || t == 42) || ((t == 12 || t == 13) && rowWords % 4)) return VK_NULL_HANDLE;
    /* on the matrix units where the device has them (qmatmul_gemm_bdc.comp; COLI_VK_CHAIN_GEMM=0 off) */
    if (!KC.bdgc_tried[t]) {
        KC.bdgc_tried[t] = 1;
        if (KC.bd && coop_on("COLI_VK_CHAIN_GEMM") && kc_shared_limit() >= GEMM_BDC_SHARED) {
            if (!KC.mod_bdgc) KC.mod_bdgc = load_module(KC.core.spv_path, "qmatmul_gemm_bdc.spv");
            int32_t v = t;
            VkSpecializationMapEntry me = {0, 0, 4};
            VkSpecializationInfo si = {1, &me, 4, &v};
            KC.bdgc[t] = coop_pipe(KC.mod_bdgc, &si);
        }
    }
    if (KC.bdgc[t]) return KC.bdgc[t];
    if (!KC.mod_bdg) return VK_NULL_HANDLE;
    if (!KC.bdg_tried[t]) {
        KC.bdg_tried[t] = 1;
        int32_t v = t;
        VkSpecializationMapEntry me = {0, 0, 4};
        VkSpecializationInfo si = {1, &me, 4, &v};
        KC.bdg[t] = make_pipe(KC.mod_bdg, &si);
    }
    return KC.bdg[t];
}
static int matmul_bd(const ColiVkTensorInfo *ti, VkcBuf *x, size_t xb, VkcBuf *y, size_t yb, int S) {
    VkBuffer tab = (VkBuffer)coli_vk_block_table_dev(g_kd);
    if (!KC.bd || !tab) return 0;
    VkcBind bd[4] = {{x->buf, xb, 0, 0}, {(VkBuffer)ti->wbuf, 0, 0, 0}, {tab, 0, 0, 0}, {y->buf, yb, 0, 1}};
    x->last_frame = y->last_frame = KC.serial;
    KC.kind = PK_GEMV;
    /* a prompt's rows: the tiled GEMM, each weight decoded once for 64 rows of x; not a
     * speculative verify's (vkc_gemm_rows(0)): its rows past 4 the GEMV a row each, a decode
     * step's bits (the tiles gave a verify of 5 or 6 rows other bits, and speculative decoding's
     * output then followed the drafts' count) */
    VkPipeline gp = S > 4 && KC.gemm_rows != 0 && ti->fmt < 164 && ti->I % 64 == 0 ? bdg_pipe(ti->fmt - 100, ti->rowWords) : VK_NULL_HANDLE;
    if (gp) {
        struct VkcPC pc = {ti->fmt, S, ti->I, ti->O, ti->rowWords, ti->gs};
        KC.kind = PK_GEMM;
        /* 128 x 128 where the rows fill it, on the matrix units */
        VkPipeline wp = S >= 128 && gp == KC.bdgc[ti->fmt - 100] ? bdgw_pipe(ti->fmt - 100) : VK_NULL_HANDLE;
        if (wp) {
            int ok = record(wp, bd, 4, &pc, sizeof pc, (uint32_t)((ti->O + 127) / 128), (uint32_t)((S + 127) / 128), 1);
            KC.st.matmuls += ok; KC.st.gemms += ok; KC.st.tile_gemms += ok; KC.st.tile_wide += ok;
            return ok;
        }
        int ok = record(gp, bd, 4, &pc, sizeof pc, (uint32_t)((ti->O + 63) / 64), (uint32_t)((S + 63) / 64), 1);
        KC.st.matmuls += ok; KC.st.gemms += ok;
        if (gp == KC.bdgc[ti->fmt - 100]) KC.st.tile_gemms += ok;   /* on the matrix units */
        return ok;
    }
    /* a speculative verify's rows (S <= 4) share each weight's decode */
    int nr = S <= 4 ? S : 1;
    VkPipeline v = ti->fmt < 164 && ti->I % 32 == 0 ? bdv8_pipe(ti->fmt - 100, nr, ti->rowWords, ti->I) : VK_NULL_HANDLE;
    if (v) KC.st.gemv_i8++;
    else if (ti->fmt < 164 && ti->I % 32 == 0) v = bdv_pipe(ti->fmt - 100, nr, ti->rowWords, ti->I);
    int ok;
    if (v) {
        /* lanes per row: about six 32-weight steps each, at most a subgroup (the shader
         * caps it at its own); rows a workgroup 256 / lanes, a grid-stride beyond 65535.
         * Measured on an RTX PRO 4000 against four and eight: 10240 x 2560 Q4_K 53 -> 46 us
         * for one row, 108 -> 82 us for three; 2560 x 6144 stays at 32 lanes, 31 us */
        const char *e = getenv("COLI_VK_CHAIN_GEMV_LPR");
        int steps = ti->I / 32, lpr = 1;
        if (e && *e) lpr = atoi(e); else while (lpr < 32 && lpr * 6 < steps) lpr *= 2;
        if (lpr < 1) lpr = 1;
        struct { int fmt, S, I, O, rowWords, gs, lpr, nr; } pc = {ti->fmt, S, ti->I, ti->O, ti->rowWords, ti->gs, lpr, nr};
        /* at most 256 workgroups, the rest by grid-stride: each stages x once for more rows
         * (an RTX PRO 4000: Q4_K 10240 x 2560 65 -> 57 us, IQ2_S 116 -> 95) */
        int rows_wg = 256 / lpr, wg = (ti->O + rows_wg - 1) / rows_wg;
        if (wg > 256) wg = 256;
        ok = record(v, bd, 4, &pc, sizeof pc, (uint32_t)wg, (uint32_t)((S + nr - 1) / nr), 1);
    } else {
        struct VkcPC pc = {ti->fmt, S, ti->I, ti->O, ti->rowWords, ti->gs};
        ok = record(KC.bd, bd, 4, &pc, sizeof pc, (uint32_t)((ti->O + 7) / 8), (uint32_t)S, 1);
    }
    KC.st.matmuls += ok;
    return ok;
}
static int matmul_aligned(const ColiVkTensorInfo *ti, VkcBuf *x, size_t xb, VkcBuf *y, size_t yb, int S) {
    if (ti->fmt >= 100) return matmul_bd(ti, x, xb, y, yb, S);
    int path = -1;   /* GEMM slot, -1 the GEMV */
    if (KC.ngemm && S >= 2) {
        int take = KC.gemm_rows < 0 ? (KC.core.gemm_min_s && S >= KC.core.gemm_min_s &&
                                      (int64_t)S * ti->O >= KC.core.gemm_min_so)
                                   : (KC.gemm_rows > 0 && S >= KC.gemm_rows);
        if (take) { path = 0; while (path + 1 < KC.ngemm && S > KC.gemm_bn[path]) path++; }
    }
    VkcBind bd[4] = {{x->buf, xb, 0, 0}, {(VkBuffer)ti->wbuf, 0, 0, 0}, {(VkBuffer)ti->sbuf, 0, 0, 0},
                     {y->buf, yb, 0, 1}};
    x->last_frame = y->last_frame = KC.serial;
    struct VkcPC pc = {ti->fmt, S, ti->I, ti->O, ti->rowWords, ti->gs};
    int ok;
    /* the vectorized GEMV where the row is whole 16-byte steps: x staged where the row fits
     * the staging with its pads, else read from its buffer (xg) */
    int per = ti->fmt == 1 ? 16 : ti->fmt == 4 ? 32 : ti->fmt == 11 || ti->fmt == 14 ? 8 : ti->fmt == 10 ? 4 : 0;
    int xg = (ti->rowWords / 4) * per / 4 + (ti->rowWords / 4) * per / 32 > KC.gemv4_xs;
    int v4 = path < 0 && KC.gemv4 && per && ti->rowWords % 4 == 0 && ti->I % 4 == 0 &&
             (ti->fmt != 4 || (ti->gs > 0 && ti->gs % 32 == 0)) && (xg ? KC.gemv4x : KC.gemv4);
    KC.kind = path >= 0 ? PK_GEMM : ti->fmt == 1 ? PK_GEMV8 : PK_GEMV;
    int xv = ti->I / 4 + ti->I / 32;   /* a row of x staged, with chain_gemv2's padding */
    /* a matrix of few long rows: a row a workgroup with every lane on it (an RTX PRO 4000,
     * Qwen3.8's 320 x 10240 hyper-connection mixer: 20 workgroups of 16 rows 34 us, 320 of
     * one row 15) */
    int few = (ti->O + 15) / 16 < 128 && ti->I / 16 >= 256;
    /* On NVIDIA chain_gemv.comp takes the other int8 matrices: an RTX PRO 4000 ran them
     * faster there (Qwen3.8's 10240 x 320 hyper-connection rows 21 -> 14 us, its 512 x 2560
     * router 11 -> 5, 10240 x 2560 90 -> 71), decode Qwen3.8 IQ2_XS 39.5/44.4 -> 45.5/45.6
     * tok/s and Qwen3.6's int8 trunk 62.2/61.9 -> 65.2/65.0. A Radeon 8060S measured
     * chain_gemv2 faster (docs/vulkan.md). */
    int g2 = path < 0 && KC.gv2 && S <= 4 && ti->fmt == 1 &&   /* int4 measured slower than chain_gemv here */
             ti->I % 32 == 0 && xv <= KC.gv2_floats / 4 && ti->rowWords % 4 == 0 && (KC.gv2_all || few);
    if (g2) {
        /* rows a workgroup: short matrices in small blocks, so they still fill the device;
         * 0 for a matrix of few long rows */
        int rpw = ti->O <= 1024 ? 16 : ti->O <= 4096 ? 32 : 64;
        if (few) rpw = 0;
        /* a speculative verify's rows (S <= 4) share each weight load */
        int nr = S <= 4 && S * xv <= KC.gv2_floats / 4 ? S : 1;
        struct { int fmt, S, I, O, rowWords, gs, rpw, nr; } pc8 = {ti->fmt, S, ti->I, ti->O, ti->rowWords, ti->gs, rpw, nr};
        ok = record(KC.gv2, bd, 4, &pc8, sizeof pc8, (uint32_t)(rpw ? (ti->O + rpw - 1) / rpw : ti->O), (uint32_t)((S + nr - 1) / nr), 1);
    }
    else if (v4) {
        /* each workgroup stages x once: give it enough rows that the staging does not
         * rival the weights, while keeping enough workgroups to fill the device */
        /* lanes per row: about sixteen 16-byte steps each (COLI_VK_CHAIN_GEMV_LPR overrides) */
        const char *e = getenv("COLI_VK_CHAIN_GEMV_LPR");
        int nq = ti->rowWords / 4, lpr = 4;
        if (e && *e) lpr = atoi(e); else while (lpr < 64 && lpr * 16 < nq) lpr *= 2;
        if (lpr < 1) lpr = 1;
        struct { int fmt, S, I, O, rowWords, gs, lpr, nr; } pc7 = {ti->fmt, S, ti->I, ti->O, ti->rowWords, ti->gs, lpr, 1};
        int rows_wg = 256 / lpr;   /* at least, at subgroups of 64 */
        int wg = (ti->O + rows_wg - 1) / rows_wg; if (wg > 1024) wg = 1024;
        /* a few long rows: a row a workgroup, every lane on it (lpr 0), where the rows would
         * leave the device short of workgroups: a decode's or a verify's rows (a prompt's
         * would stage x once a row of W) */
        if (!(e && *e) && S <= 4 && wg < 128 && nq >= 256) { pc7.lpr = 0; wg = ti->O; }
        /* a verify's rows (S <= 4) in one workgroup, every weight load shared, as many as the
         * staging holds (x read from its buffer: all of them), where the matrix has the
         * workgroups to fill the device without its rows (512 x 2048 int8, 16 workgroups:
         * 9 us -> 11 to 15 sharing) */
        int xrow = (ti->rowWords / 4) * per / 4, nr = S <= 4 && wg >= 64 ? S : 1;
        xrow += xrow / 8 + 1;
        while (nr > 1 && !xg && nr * xrow > KC.gemv4_xs) nr--;
        if (!(xg ? KC.gemv4xn : KC.gemv4n)) nr = 1;
        pc7.nr = nr;
        KC.st.gemv_xg += xg;
        VkPipeline gp = nr > 1 ? (xg ? KC.gemv4xn : KC.gemv4n) : xg ? KC.gemv4x : KC.gemv4;
        ok = record(gp, bd, 4, &pc7, sizeof pc7, (uint32_t)wg, (uint32_t)((S + nr - 1) / nr), 1);
    }
    else if (path >= 0 && KC.tg && !ti->wide && (ti->fmt == 1 || ti->fmt == 2 || (ti->fmt == 4 && ti->gs >= 8 && ti->gs % 8 == 0)) &&
             ti->I % 64 == 0 && ti->O % 128 == 0 && ti->rowWords % 4 == 0) {
        ok = record(KC.tg, bd, 4, &pc, sizeof pc, (uint32_t)((S + 63) / 64), (uint32_t)(ti->O / 128), 1);
        KC.st.tile_gemms += ok;
    }
    else if (path >= 0)
        ok = record(KC.gemm[path], bd, 4, &pc, sizeof pc, (uint32_t)((ti->O + KC.gemm_bm[path] - 1) / KC.gemm_bm[path]),
                    (uint32_t)((S + KC.gemm_bn[path] - 1) / KC.gemm_bn[path]), 1);
    else ok = record(KC.pipe[P_GEMV], bd, 4, &pc, sizeof pc, (uint32_t)((ti->O + 7) / 8), (uint32_t)S, 1);
    KC.st.matmuls += ok; KC.st.gemms += ok && path >= 0;
    return ok;
}
static int vkc_matmul_in(ColiVkTensor *t, VkcBuf *x, size_t xo, VkcBuf *y, size_t yo, int S);
int vkc_matmul(ColiVkTensor *t, VkcBuf *x, size_t xo, VkcBuf *y, size_t yo, int S) {
    ColiVkTensorInfo ti;
    if (KC.prof == 2 && vkc_tensor_info(t, &ti))
        g_prof_shape = (uint64_t)(ti.fmt & 0xff) << 56 | (uint64_t)(S > 255 ? 255 : S) << 48 |
                       (uint64_t)(ti.O & 0xffffff) << 24 | (uint64_t)(ti.I & 0xffffff);
    int ok = vkc_matmul_in(t, x, xo, y, yo, S);
    g_prof_shape = 0;
    return ok;
}
static int vkc_matmul_in(ColiVkTensor *t, VkcBuf *x, size_t xo, VkcBuf *y, size_t yo, int S) {
    ColiVkTensorInfo ti;
    if (!open_frame() || KC.lost || S < 1 || S > 65535 || !vkc_tensor_info(t, &ti)) return 0;
    size_t xb = xo * 4, yb = yo * 4, xn = (size_t)S * ti.I * 4, yn = (size_t)S * ti.O * 4;
    if (xb + xn > x->bytes || yb + yn > y->bytes) { fprintf(stderr, "[VK] chain: matmul out of range\n"); return 0; }
    if (xb % KC.align == 0 && yb % KC.align == 0) return matmul_aligned(&ti, x, xb, y, yb, S);
    /* an offset the device cannot bind: through temporaries that live with the frame */
    VkcFrame *f = open_frame();
    VkcBuf *tx = vkc_buf(xn, VKC_DEV), *ty = vkc_buf(yn, VKC_DEV);
    if (!tx || !ty) { if (tx) buf_release(tx); if (ty) buf_release(ty); return 0; }
    if (f->ntmp + 2 > f->ctmp) {
        int c = f->ctmp ? 2 * f->ctmp : 8; while (c < f->ntmp + 2) c *= 2;
        VkcBuf **n = realloc(f->tmp, (size_t)c * sizeof *n);
        if (!n) { buf_release(tx); buf_release(ty); return 0; }
        f->tmp = n; f->ctmp = c;
    }
    f->tmp[f->ntmp++] = tx; f->tmp[f->ntmp++] = ty;
    return copy_bytes(tx, 0, x, xb, xn) && matmul_aligned(&ti, tx, 0, ty, 0, S) && copy_bytes(y, yb, ty, 0, yn);
}

/* ---- the chain's shaders ----------------------------------------------------------- */
int vkc_norm(VkcBuf *x, VkcBuf *w, VkcBuf *y, const VkcNorm *p) {
    KC.kind = PK_NORM;
    VkcBind bd[3] = {B(x, x == y), B(w, 0), B(y, 1)};
    uint32_t gx, gy; grid((uint64_t)p->nseg, &gx, &gy);
    return record(KC.pipe[P_NORM], bd, 3, p, sizeof *p, gx, gy, 1);
}
int vkc_rope(VkcBuf *x, VkcBuf *cs, const VkcRope *p) {
    KC.kind = PK_ROPE;
    VkcBind bd[2] = {B(x, 1), B(cs, 0)};
    uint32_t gx, gy; grid(((uint64_t)p->nseg * p->half_ + 63) / 64, &gx, &gy);
    return record(KC.pipe[P_ROPE], bd, 2, p, sizeof *p, gx, gy, 1);
}
int vkc_attn(VkcBuf *q, VkcBuf *kc, VkcBuf *vc, VkcBuf *o, VkcBuf *gate, VkcBuf *sel, const VkcAttn *p) {
    VkcAttnW w;
    memset(&w, 0, sizeof w);
    w.a = *p;
    return vkc_attn_w(q, kc, vc, o, gate, sel, NULL, &w);
}
/* ---- a float16 KV cache (vkc_attn_h, vkc_attn_wh, vkc_copy_regions_h) ------------
 * The attention shaders read K and V two halves a word where their specialization constant 7
 * (KV16) is set: each pipeline below has a float16 cache's twin, made on first use and
 * picked while g_kvh says the call in progress reads such a cache. */
static const VkSpecializationMapEntry g_kv16_me = {7, 0, 4};
static const VkBool32 g_kv16_on = VK_TRUE;
static const VkSpecializationInfo g_kv16_si = {1, &g_kv16_me, 4, &g_kv16_on};
static VkPipeline attn_h_pipe(void) {
    if (!KC.attn_h && !KC.attn_h_tried) { KC.attn_h_tried = 1; KC.attn_h = make_pipe(KC.mod[P_ATTN], &g_kv16_si); }
    return KC.attn_h;
}
/* ---- blocked attention for prompt chunks (chain_attnb.comp), made on first use ----
 * chain_attn reads a row's positions once per (head, row): a chunk of S rows reads its
 * KV cache S * H times, which held a 2048-row chunk of Qwen3.6 at 1.2 s a layer on a
 * Radeon 780M. From COLI_VK_ATTN_BLOCK rows (16; 0 = never) a call takes chain_attnb,
 * one workgroup per KV head and block of rows: each K and V row read once per block.
 * Below it, and for decode and an MTP verify, chain_attn's bits as before. */
static struct { VkShaderModule mod; VkPipeline pipe, pipe_h; int tried, tried_h, bc; } g_kab[2];
#define KAB (g_kab[g_kd])
static VkPipeline kab_pipe(void) {
    if (g_kvh && KAB.pipe && !KAB.pipe_h && !KAB.tried_h) {   /* a float16 cache's twin */
        KAB.tried_h = 1;
        int32_t v[2] = {KAB.bc, 1};
        VkSpecializationMapEntry me[2] = {{0, 0, 4}, {7, 4, 4}};
        VkSpecializationInfo si = {2, me, sizeof v, v};
        KAB.pipe_h = make_pipe(KAB.mod, &si);
    }
    if (KAB.pipe || KAB.tried || !vkc_ready()) return g_kvh ? KAB.pipe_h : KAB.pipe;
    KAB.tried = 1;
    /* the tile of positions as the device's shared memory allows: K and V rows of 256
     * floats, the partial dots and the scores, about 3.2 KiB a position */
    VkPhysicalDeviceProperties pp; vkGetPhysicalDeviceProperties((VkPhysicalDevice)KC.core.phys, &pp);
    int bc = 16;
    while (bc >= 4 && (size_t)bc * (2 * 256 + 32 * 8 + 32 + 1) * 4 > pp.limits.maxComputeSharedMemorySize) bc /= 2;
    if (bc < 4) return VK_NULL_HANDLE;
    if ((KAB.mod = load_module(KC.core.spv_path, "chain_attnb.spv"))) {
        int32_t v = bc;
        VkSpecializationMapEntry me = {0, 0, 4};
        VkSpecializationInfo si = {1, &me, 4, &v};
        if ((KAB.pipe = make_pipe(KAB.mod, &si))) KAB.bc = bc;
    }
    return g_kvh && KAB.pipe ? kab_pipe() : KAB.pipe;
}
static void kab_shutdown(void) {
    if (KAB.pipe) vkDestroyPipeline(KC.dev, KAB.pipe, NULL);
    if (KAB.pipe_h) vkDestroyPipeline(KC.dev, KAB.pipe_h, NULL);
    if (KAB.mod) vkDestroyShaderModule(KC.dev, KAB.mod, NULL);
    memset(&KAB, 0, sizeof KAB);
}
/* ---- attention over a prompt chunk, cut over several submissions ----------------
 * The attention ops' work grows with rows x positions: a chunk of thousands of rows makes
 * one dispatch of seconds (8192 rows of Qwen3.6 at its full context: about 5 s on a Radeon
 * 780M), past what a driver lets one submission run (amdgpu resets a ring after 10 s;
 * Windows' TDR after 2 s). An attention whose rows x positions x heads x head dim passes
 * COLI_VK_ATTN_SLICE (2^32; 0 = never) is recorded in slices of rows, each ending the
 * frame it is in (submitted, not waited for) and the next opening a frame of its own: the
 * next frame's first barrier orders it after the slice. A slice starts at a multiple of
 * `align` (the blocked attention's rows per workgroup), so its blocks are the ones the
 * whole dispatch would have made, and no row's arithmetic changes. */
static long long attn_slice_budget(void) {
    const char *e = getenv("COLI_VK_ATTN_SLICE");
    long long v = e && *e ? atoll(e) : (1LL << 32);
    return v < 0 ? 0 : v;
}
long long vkc_attn_slice_budget(void) { return attn_slice_budget(); }
static int attn_slice_rows(int S, int pos_base, double width, int align) {
    long long b = attn_slice_budget();
    if (align < 1) align = 1;
    if (!b || S <= align) return S;
    double per_row = (double)(pos_base + S) * (width > 1 ? width : 1);
    long long rr = (long long)((double)b / per_row);
    rr = rr / align * align;
    if (rr < align) rr = align;
    return rr >= S ? S : (int)rr;
}
static int attn_slice_next(void) {   /* the open frame out, a new one in */
    KC.st.attn_slices++;
    return vkc_submit(0) && vkc_begin();
}
/* chain_attn_flash.comp's pipeline for this head dim and grouping, made on first use;
 * NULL when the shader is missing or the pipeline fails */
static VkPipeline fa_pipe(int hd, int gq) {
    for (int i = 0; i < KC.nfa; i++) if (KC.fa_hd[i] == hd && KC.fa_gq[i] == gq && KC.fa_kv[i] == g_kvh) return KC.fa[i];
    if (!KC.mod_fa || KC.nfa == VKC_FA_MAX) return VK_NULL_HANDLE;
    /* keys a block (64, 32 at head dim 256: its scores, P and K/V tiles in 28 KiB of shared
     * memory) and the subgroups that share a row tile's work (256 invocations, 4 row tiles);
     * a float16 cache's twin (KV16) */
    int32_t sv[5] = {hd, gq, hd > 128 ? 32 : 64, 256 / KC.core.coop_sg / 4, g_kvh};
    VkSpecializationMapEntry me[5] = {{0, 0, 4}, {1, 4, 4}, {2, 8, 4}, {3, 12, 4}, {7, 16, 4}};
    VkSpecializationInfo si = {5, me, sizeof sv, sv};
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT rss = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO_EXT,
        .requiredSubgroupSize = (uint32_t)KC.core.coop_sg};
    VkComputePipelineCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .pNext = &rss,
                  .flags = VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT_EXT,
                  .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = KC.mod_fa, .pName = "main",
                  .pSpecializationInfo = &si}, .layout = KC.pl};
    VkPipeline p = VK_NULL_HANDLE;
    if (vkCreateComputePipelines(KC.dev, VK_NULL_HANDLE, 1, &ci, NULL, &p) != VK_SUCCESS) {
        vkDestroyShaderModule(KC.dev, KC.mod_fa, NULL); KC.mod_fa = VK_NULL_HANDLE;   /* not again */
        return VK_NULL_HANDLE;
    }
    KC.fa[KC.nfa] = p; KC.fa_hd[KC.nfa] = hd; KC.fa_kv[KC.nfa] = g_kvh; KC.fa_gq[KC.nfa++] = gq;
    return p;
}
/* From COLI_VK_CHAIN_FLASH rows (16; 0 = never) a causal attention without a selection
 * list, position-major rows or a second V width (head dim a multiple of 64; a window, a
 * ring and sinks as chain_attnb takes them) takes chain_attn_flash where the device has it,
 * ahead of the blocked shader: Flint's (sliding layers with sinks, the others with sinks)
 * took chain_attnb for 100.2 of a 16K prompt's 106 s of device time on an RTX PRO 4000. */
int vkc_attn_flash_rows(void) {
    const char *e = getenv("COLI_VK_CHAIN_FLASH");
    int v = e && *e ? atoi(e) : 16;
    return v < 0 ? 0 : v;
}
int vkc_attn_block_rows(void) {
    const char *e = getenv("COLI_VK_ATTN_BLOCK");
    int v = e && *e ? atoi(e) : 16;
    return v < 0 ? 0 : v;
}
/* ---- a decode's attention split over its positions (chain_attn_dec.comp) ----------
 * chain_attn gives a (head, row) one workgroup that walks the positions in turn: a decode
 * step's few rows leave the device all but idle (Qwen3.6's 16 heads, about 160 us a
 * layer at 700 positions on an RTX PRO 4000). From VKC_ATTN_DEC_MIN positions a call of
 * up to VKC_ATTN_DEC_ROWS rows without a selection list runs in blocks of L positions, a
 * workgroup a (head, block, row), the blocks then joined; each row's blocks as its own
 * positions choose them (attn_dec_len), the rows short of VKC_ATTN_DEC_MIN positions through
 * chain_attn, as a decode step of each alone would run: a verify's rows get their decode
 * steps' bits (a verify of 5 rows took chain_attn before, and speculative decoding's output
 * then followed the drafts' count). COLI_VK_ATTN_DEC=0 keeps chain_attn. */
#define VKC_ATTN_DEC_MIN 128
#define VKC_ATTN_DEC_ROWS 8
static int attn_dec_len(int n, int H) {   /* chain_attn_dec.comp's dec_len, H the workgroups of a block */
    int L = 64, nb = (n + L - 1) / L;
    while (L < 256 && (int64_t)H * nb > 1024) { L *= 2; nb = (n + L - 1) / L; }
    return L;
}
static VkPipeline adec_pipe(void) {
    static int on = -1;
    if (on < 0) { const char *e = getenv("COLI_VK_ATTN_DEC"); on = !(e && *e == '0'); }
    if (on && !KC.adec && !KC.adec_tried) {
        KC.adec_tried = 1;
        if ((KC.mod_adec = load_module(KC.core.spv_path, "chain_attn_dec.spv"))) KC.adec = make_pipe(KC.mod_adec, NULL);
    }
    if (!on || !KC.adec || !g_kvh) return KC.adec;
    if (!KC.adec_h && !KC.adec_h_tried) { KC.adec_h_tried = 1; KC.adec_h = make_pipe(KC.mod_adec, &g_kv16_si); }   /* KV16 */
    return KC.adec_h;
}
/* chain_attn_dec's grouped pipeline (mode 2) for gq query heads a KV head of head dim hd, made
 * on first use; NULL where the shared memory it takes (the heads' q rows, scores and sums) is
 * past the device's, or COLI_VK_ATTN_DEC_GQA=0 */
static VkPipeline adg_pipe(int hd, int gq) {
    static int on = -1;
    if (on < 0) { const char *e = getenv("COLI_VK_ATTN_DEC_GQA"); on = !(e && *e == '0'); }
    if (!on || gq < 1 || gq > 16 || hd < 1 || hd > 256 || !adec_pipe()) return VK_NULL_HANDLE;
    for (int i = 0; i < KC.nadg; i++) if (KC.adg_hd[i] == hd && KC.adg_gq[i] == gq && KC.adg_kv[i] == g_kvh) return KC.adg[i];
    if (KC.nadg == VKC_ADG_MAX) return VK_NULL_HANDLE;
    VkPhysicalDeviceProperties pp; vkGetPhysicalDeviceProperties((VkPhysicalDevice)KC.core.phys, &pp);
    if (((size_t)gq * (hd + 256 + 64) + hd) * 4 > pp.limits.maxComputeSharedMemorySize) return VK_NULL_HANDLE;
    /* two threads a position for the scores at head dims of 256, one below; the same for
     * either cache, whose bits stay equal. One layer after 18K positions (float16): a Radeon
     * 780M's 64 lanes of 512-byte rows 1463 -> 1025 us at two, its 128 dims 1241 -> 1452 (one
     * kept); an RTX PRO 4000 125.5 -> 121.8 us, its 128 dims 131.1 -> 129.6 */
    const char *te = getenv("COLI_VK_ATTN_DEC_TPP");
    int tpp = te && (*te == '1' || *te == '2') ? *te - '0' : hd >= 256 ? 2 : 1;
    int32_t sv[4] = {gq, hd, g_kvh, tpp};
    VkSpecializationMapEntry me[4] = {{0, 0, 4}, {1, 4, 4}, {7, 8, 4}, {2, 12, 4}};
    VkSpecializationInfo si = {4, me, sizeof sv, sv};
    VkPipeline p = make_pipe(KC.mod_adec, &si);
    if (!p) return VK_NULL_HANDLE;
    KC.adg[KC.nadg] = p; KC.adg_hd[KC.nadg] = hd; KC.adg_kv[KC.nadg] = g_kvh; KC.adg_gq[KC.nadg++] = gq;
    return p;
}
/* the blocks' scratch, at least `bytes`: a smaller one goes with the open frame (freed when
 * the frame comes back), so a call never waits on the frames before it */
static VkcBuf *adec_scratch(size_t bytes) {
    if (KC.adec_x && KC.adec_x->bytes >= bytes) return KC.adec_x;
    VkcFrame *f = open_frame();
    if (!f) return NULL;
    VkcBuf *nb = vkc_buf(bytes + bytes / 2, VKC_DEV);
    if (!nb) return NULL;
    if (KC.adec_x) {
        if (f->ntmp == f->ctmp) {
            int c = f->ctmp ? 2 * f->ctmp : 8;
            VkcBuf **t = realloc(f->tmp, (size_t)c * sizeof *t);
            if (!t) { buf_release(nb); return NULL; }
            f->tmp = t; f->ctmp = c;
        }
        f->tmp[f->ntmp++] = KC.adec_x;
    }
    return KC.adec_x = nb;
}
static int attn_dec(VkcBuf *q, VkcBuf *kc, VkcBuf *vc, VkcBuf *o, VkcBuf *gate, VkcBuf *snk, const VkcAttnW *w) {
    int S = w->a.S, last = w->a.pos_base + S;   /* the last row's positions */
    int n = w->win > 0 && w->win < last ? w->win : last;
    /* Only where a prompt chunk of this attention would take chain_attn_flash (f16, so a
     * row's bits already depend on how a forward is cut into steps): elsewhere the prompt's
     * chain_attnb gives a row the bits chain_attn gives it alone, and a prefix resumed must
     * answer to the last digit as the prompt recomputed does (tests/test_brio_serve.py).
     * Windows, rings and sinks too, since the flash shader takes them: Flint's decode after a
     * 17K prompt read its full layers' positions a workgroup a head, 15.4 tok/s on an
     * RTX PRO 4000 */
    int G = w->a.H / w->a.KVH;
    if (!(KC.mod_fa && vkc_attn_flash_rows() > 0 && !w->kv_pm && w->vd == w->a.hd &&
          w->a.hd % 64 == 0 && VKC_FA_ROWS % G == 0 && (!g_kvh || (w->a.k_off % 8 == 0 && w->a.v_off % 8 == 0)) &&
          fa_pipe(w->a.hd, G))) return 0;
    if (S < 1 || S > VKC_ATTN_DEC_ROWS || n < VKC_ATTN_DEC_MIN || w->a.pos_base + 1 < VKC_ATTN_DEC_MIN || !adec_pipe()) return 0;
    /* a KV head's query heads in one workgroup a block (mode 2), each K and V row read once
     * for them, four floats a load (the offsets whole vec4s). Its blocks' rule counts four
     * workgroups a KV head: Qwen3.6 after 18K positions on an RTX PRO 4000, 256 positions a
     * block, 279 us a layer; 64 (a workgroup a KV head) 325 us, its join alone 60 */
    VkPipeline gp = G >= 1 && w->a.hd == w->vd && w->a.k_off % 4 == 0 && w->a.v_off % 4 == 0 ? adg_pipe(w->a.hd, G) : VK_NULL_HANDLE;
    int hc = gp ? 4 * w->a.KVH : w->a.H, nb = 1;
    for (int r = 0; r < S; r++) {   /* the most blocks a row takes, of its window's positions as the shader counts them */
        int nr = w->a.pos_base + r + 1;
        if (w->win > 0 && nr > w->win) nr = w->win;
        int Lr = attn_dec_len(nr, hc), m = (nr + Lr - 1) / Lr;
        if (m > nb) nb = m;
    }
    VkcBuf *xs = adec_scratch((size_t)S * w->a.H * nb * (w->vd + 2) * 4);
    if (!xs) return 0;
    struct { VkcAttnW w; int mode, L, nb; } pc = {*w, gp ? 2 : 0, hc, nb};
    VkcBind bd[8] = {B(q, 0), B(kc, 0), B(vc, 0), B(o, 1), B(gate, 0), B(NULL, 0), B(snk, 0), B(xs, 1)};
    VkPipeline p0 = adec_pipe();   /* a float16 cache's mode 0 where g_kvh */
    int ok = gp ? record(gp, bd, 8, &pc, sizeof pc, (uint32_t)w->a.KVH, (uint32_t)nb, (uint32_t)S)
                : p0 && record(p0, bd, 8, &pc, sizeof pc, (uint32_t)w->a.H, (uint32_t)nb, (uint32_t)S);
    KC.st.attn_dec_gqa += ok && gp;
    pc.mode = 1;
    bd[7].written = 0;
    ok = ok && record(KC.adec, bd, 8, &pc, sizeof pc, (uint32_t)w->a.H, (uint32_t)S, 1);
    KC.st.attn_dec += ok;
    return ok;
}
int vkc_attn_w(VkcBuf *q, VkcBuf *kc, VkcBuf *vc, VkcBuf *o, VkcBuf *gate, VkcBuf *sel, VkcBuf *snk,
               const VkcAttnW *p) {
    KC.kind = PK_ATTN;
    VkcAttnW w = *p;   /* the shader's push constants: VkcAttn's fields, then these */
    if (w.vd <= 0) w.vd = w.a.hd;
    if (w.a.hd > 256 || w.vd > 256 || w.a.H % w.a.KVH || w.win < 0 || w.ring < 0 || (w.sink && !snk)) return 0;
    VkcBind bd[7] = {B(q, 0), B(kc, 0), B(vc, 0), B(o, 1), B(gate, 0), B(sel, 0), B(snk, 0)};
    int G = w.a.H / w.a.KVH, blk = vkc_attn_block_rows(), fl = vkc_attn_flash_rows();
    /* a prompt chunk's causal attention on the matrix units (chain_attn_flash), VKC_FA_ROWS / G
     * tokens a workgroup; else the blocked shader, else chain_attn */
    VkPipeline fp = KC.mod_fa && fl > 0 && w.a.S >= fl && !(sel && w.a.sel_row > 0) && !w.kv_pm &&
                    w.vd == w.a.hd && w.a.hd % 64 == 0 && VKC_FA_ROWS % G == 0 && (!g_kvh || (w.a.k_off % 8 == 0 && w.a.v_off % 8 == 0))
                    ? fa_pipe(w.a.hd, G) : VK_NULL_HANDLE;
    int blocked = !fp && blk > 0 && w.a.S >= blk && G <= 32 && kab_pipe(), br = w.a.sel_row > 0 ? 1 : 32 / G;
    /* a decode's or a verify's rows over many positions: split over them; a verify across
     * VKC_ATTN_DEC_MIN positions in two calls, its rows before it as their decode steps run */
    if (!fp && !blocked && !(sel && w.a.sel_row > 0) && w.a.S > 1 && w.a.S <= VKC_ATTN_DEC_ROWS &&
        w.a.pos_base + 1 < VKC_ATTN_DEC_MIN && w.a.pos_base + w.a.S >= VKC_ATTN_DEC_MIN) {
        int k = VKC_ATTN_DEC_MIN - (w.a.pos_base + 1);
        VkcAttnW lo = *p, hi = *p;
        lo.a.S = k;
        hi.a.S -= k; hi.a.pos_base += k; hi.a.q_off += k * p->a.q_row; hi.a.o_off += k * p->a.o_row; hi.a.g_off += k * p->a.g_row;
        return vkc_attn_w(q, kc, vc, o, gate, sel, snk, &lo) && vkc_attn_w(q, kc, vc, o, gate, sel, snk, &hi);
    }
    if (!fp && !blocked && !(sel && w.a.sel_row > 0) && attn_dec(q, kc, vc, o, gate, snk, &w)) return 1;
    /* the slices' budget counts chain_attnb's work: the matrix units do a unit of it 8.5 to 13
     * times faster (a Radeon 8060S, 186 -> 22 ms; Flint's 16K prompt on an RTX PRO 4000,
     * 100 -> 7.8 s), so a slice of theirs takes VKC_FA_SLICE times the units, about twice
     * chain_attnb's time a submission. At the budget's own size Qwen3.6's 16K prompt was cut
     * into slices of 64 rows, 16 workgroups and a submission each: 5.9 s of attention */
    int S = w.a.S, rr = attn_slice_rows(S, w.a.pos_base, (double)w.a.H * (w.a.hd > w.vd ? w.a.hd : w.vd) / (fp ? VKC_FA_SLICE : 1),
                                        fp ? VKC_FA_ROWS / G : blocked ? br : 1);
    int ok = 1;
    for (int r0 = 0; ok && r0 < S; r0 += rr) {
        VkcAttnW x = w;
        int n = S - r0 < rr ? S - r0 : rr;
        x.a.S = n; x.a.pos_base = w.a.pos_base + r0;
        x.a.q_off += r0 * w.a.q_row; x.a.g_off += r0 * w.a.g_row; x.a.o_off += r0 * w.a.o_row;
        if (w.a.sel_row > 0) x.a.sel_off += r0 * w.a.sel_row;
        if (r0 && !attn_slice_next()) return 0;
        if (fp) {
            ok = record(fp, bd, 7, &x, sizeof x, (uint32_t)((x.a.S + VKC_FA_ROWS / G - 1) / (VKC_FA_ROWS / G)), (uint32_t)x.a.KVH, 1);
            KC.st.attn_flash += ok;
        } else if (blocked) {
            struct { VkcAttnW w; int br, B, anchor, part, st_off; } pc = {x, br, 0, 0, 0, 0};
            ok = record(kab_pipe(), bd, 7, &pc, sizeof pc, (uint32_t)x.a.KVH, (uint32_t)((x.a.S + br - 1) / br), 1);
            KC.st.attn_blocked += ok;
        } else {
            VkPipeline ap = g_kvh ? attn_h_pipe() : KC.pipe[P_ATTN];
            ok = ap && record(ap, bd, 7, &x, sizeof x, (uint32_t)x.a.H, (uint32_t)x.a.S, 1);
        }
    }
    return ok;
}
static int attn_part_rec(VkcBuf *q, VkcBuf *kc, VkcBuf *vc, VkcBuf *o, const VkcAttnW *p, int blk, int anchor, int st_off,
                         int ch, int nz, int o_z, int st_z) {
    KC.kind = PK_ATTN;
    VkcAttnW w = *p;
    if (w.vd <= 0) w.vd = w.a.hd;
    int G = w.a.KVH > 0 ? w.a.H / w.a.KVH : 0;
    if (w.a.hd > 256 || w.vd > 256 || G < 1 || G > 32 || w.a.H % w.a.KVH || w.a.sel_row > 0 || w.sink || w.ring || blk < 1 ||
        anchor < 0 || ch < 0 || (ch > 0 && (ch % 16 || nz < 1 || nz > 65535)) || !kab_pipe()) return 0;
    w.a.has_gate = 0; w.a.g_off = ch; w.a.g_row = o_z; w.a.g_seg = st_z;   /* the chunks (unused by part otherwise) */
    VkcBind bd[7] = {B(q, 0), B(kc, 0), B(vc, 0), B(o, 1), B(NULL, 0), B(NULL, 0), B(NULL, 0)};
    int br = 32 / G, S = w.a.S, rr = attn_slice_rows(S, w.a.pos_base, (double)w.a.H * (w.a.hd > w.vd ? w.a.hd : w.vd), br);
    int ok = 1;
    for (int r0 = 0; ok && r0 < S; r0 += rr) {
        VkcAttnW x = w;
        int n = S - r0 < rr ? S - r0 : rr;
        x.a.S = n; x.a.pos_base = w.a.pos_base + r0;
        x.a.q_off += r0 * w.a.q_row; x.a.o_off += r0 * w.a.o_row;
        if (r0 && !attn_slice_next()) return 0;
        struct { VkcAttnW w; int br, B, anchor, part, st_off; } pc = {x, br, blk, anchor, 1, st_off + r0 * w.a.H * 2};
        ok = record(KAB.pipe, bd, 7, &pc, sizeof pc, (uint32_t)x.a.KVH, (uint32_t)((x.a.S + br - 1) / br), (uint32_t)(ch > 0 ? nz : 1));
        KC.st.attn_blocked += ok;
    }
    return ok;
}
int vkc_attn_part(VkcBuf *q, VkcBuf *kc, VkcBuf *vc, VkcBuf *o, const VkcAttnW *p, int blk, int anchor, int st_off) {
    return attn_part_rec(q, kc, vc, o, p, blk, anchor, st_off, 0, 1, 0, 0);
}
int vkc_attn_part_chunks(VkcBuf *q, VkcBuf *kc, VkcBuf *vc, VkcBuf *o, const VkcAttnW *p, int blk, int anchor, int st_off,
                         int ch, int nz, int o_z, int st_z) {
    return ch > 0 && attn_part_rec(q, kc, vc, o, p, blk, anchor, st_off, ch, nz, o_z, st_z);
}
int vkc_dnconv(VkcBuf *in, VkcBuf *w, VkcBuf *ring, VkcBuf *out, VkcBuf *snap, const VkcDnConv *p) {
    KC.kind = PK_DNCONV;
    if (p->CK < 1 || p->CK > 9) return 0;
    VkcBind bd[5] = {B(in, 0), B(w, 0), B(ring, 1), B(out, 1), B(snap, snap != NULL && snap != ring)};
    if (snap == ring) bd[4].written = 0;   /* the same buffer: tracked once, as written */
    uint32_t gx, gy; grid(((uint64_t)p->CD + 63) / 64, &gx, &gy);
    return record(KC.pipe[P_DNCONV], bd, 5, p, sizeof *p, gx, gy, 1);
}
static VkPipeline dnrec_pipe(int KD) {
    for (int i = 0; i < KC.ndnrec; i++) if (KC.dnrec_kd[i] == KD) return KC.dnrec[i];
    if (KC.ndnrec == VKC_DNREC_MAX) return VK_NULL_HANDLE;
    /* lanes a column: each thread's part of it at most 16 values (more spilled: 73 us a
     * layer for KD 128 in one lane, 21 in eight) */
    int nspl = 1;
    while (nspl < 8 && KD / nspl > 16 && KD % (2 * nspl) == 0) nspl *= 2;
    int32_t v[3] = {KD, nspl, 128 * nspl};
    VkSpecializationMapEntry me[3] = {{0, 0, 4}, {1, 4, 4}, {2, 8, 4}};
    VkSpecializationInfo si = {3, me, sizeof v, v};
    VkPipeline p = make_pipe(KC.mod_dnrec, &si);
    if (!p) return VK_NULL_HANDLE;
    KC.dnrec[KC.ndnrec] = p; KC.dnrec_kd[KC.ndnrec++] = KD;
    return p;
}
/* chain_dnrec_rows.comp for KD: its rows a block (R) as many as the shared memory holds, at
 * most 16 (NULL past VKC_DNREC_MAX keys, or when COLI_VK_DNREC_ROWS=0) */
static VkPipeline dnrr_pipe(int KD) {
    static int on = -1;
    if (on < 0) { const char *e = getenv("COLI_VK_DNREC_ROWS"); on = !(e && *e == '0'); }
    if (!on) return VK_NULL_HANDLE;
    for (int i = 0; i < KC.ndnrr; i++) if (KC.dnrr_kd[i] == KD) return KC.dnrr[i];
    if (KC.ndnrr == VKC_DNREC_MAX) return VK_NULL_HANDLE;
    if (!KC.mod_dnrr) {
        if (KC.dnrr_tried) return VK_NULL_HANDLE;
        KC.dnrr_tried = 1;
        if (!(KC.mod_dnrr = load_module(KC.core.spv_path, "chain_dnrec_rows.spv"))) return VK_NULL_HANDLE;
    }
    int nspl = 1;   /* as dnrec_pipe */
    while (nspl < 8 && KD / nspl > 16 && KD % (2 * nspl) == 0) nspl *= 2;
    size_t row = (size_t)2 * (KD + nspl) * 4 + 2 * 128 * 4 + 2 * 4;   /* q, k (each part padded a word), v, o, beta, decay */
    int R = 16;
    while (R > 1 && (size_t)R * row > kc_shared_limit()) R /= 2;
    int32_t v[4] = {KD, nspl, 128 * nspl, R};
    VkSpecializationMapEntry me[4] = {{0, 0, 4}, {1, 4, 4}, {2, 8, 4}, {3, 12, 4}};
    VkSpecializationInfo si = {4, me, sizeof v, v};
    VkPipeline p = make_pipe(KC.mod_dnrr, &si);
    KC.dnrr[KC.ndnrr] = p; KC.dnrr_kd[KC.ndnrr++] = KD;   /* a failed make is remembered as NULL */
    return p;
}
int vkc_dnrec(int KD, VkcBuf *cv, VkcBuf *ab, VkcBuf *z, VkcBuf *st, VkcBuf *prm, VkcBuf *y, VkcBuf *snap, const VkcDnRec *p) {
    KC.kind = PK_DNREC;
    if (p->VD > 128 || KD < 1 || KD > 256 || p->VH % p->KH) return 0;
    /* a prompt chunk's rows: the norms a block of rows at a time, the rows in order with no
     * barrier (chain_dnrec_rows.comp); a decode's or a verify's few through chain_dnrec */
    VkPipeline pipe = p->S >= VKC_DNREC_ROWS ? dnrr_pipe(KD) : VK_NULL_HANDLE;
    if (!pipe) pipe = dnrec_pipe(KD);
    if (!pipe) return 0;
    VkcBind bd[7] = {B(cv, 0), B(ab, 0), B(z, 0), B(st, 1), B(prm, 0), B(y, 1), B(snap, snap != NULL && snap != st)};
    if (snap == st) bd[6].written = 0;
    return record(pipe, bd, 7, p, sizeof *p, (uint32_t)p->VH, 1, 1);
}
int vkc_ew(VkcBuf *y, VkcBuf *a, VkcBuf *b, VkcBuf *c, VkcBuf *e, const VkcEw *p) {
    KC.kind = PK_EW;
    VkcBind bd[5] = {B(y, 1), B(a == y ? NULL : a, 0), B(b == y ? NULL : b, 0), B(c == y ? NULL : c, 0), B(e == y ? NULL : e, 0)};
    /* an input that is y itself reads through binding 0's alias: bind it there too */
    if (a == y) bd[1] = (VkcBind){y->buf, 0, 0, 0};
    if (b == y) bd[2] = (VkcBind){y->buf, 0, 0, 0};
    if (c == y) bd[3] = (VkcBind){y->buf, 0, 0, 0};
    if (e == y) bd[4] = (VkcBind){y->buf, 0, 0, 0};
    uint32_t gx, gy; grid(((uint64_t)p->n + 255) / 256, &gx, &gy);
    return record(KC.pipe[P_EW], bd, 5, p, sizeof *p, gx, gy, 1);
}
enum { QSA_SCORES = 1, QSA_SELECT };   /* the modes of chain_qsa.comp that VkcQsa mode 1 runs in turn */
int vkc_qsa(VkcBuf *src, VkcBuf *w, VkcBuf *pk, VkcBuf *cs, VkcBuf *sc, VkcBuf *sel, const VkcQsa *p) {
    KC.kind = PK_QSA;
    if (p->ID > 256) return 0;
    if (p->mode == 0) {
        VkcBind bd[4] = {B(src, 0), B(w, 0), B(pk, 1), B(cs, 0)};
        return record(KC.pipe[P_QSA], bd, 4, p, sizeof *p, (uint32_t)p->nb, 1, 1);
    }
    VkcBind none = {VK_NULL_HANDLE, 0, 0, 0};
    VkcBind scores[6] = {B(src, 0), none, B(pk, 0), none, B(sc, 1), none};
    VkcBind select[6] = {none, none, none, none, B(sc, 0), B(sel, 1)};
    int blocks = (p->pos_base + p->S) / p->R;   /* complete blocks in the last row */
    VkcQsa pass = *p;
    pass.mode = QSA_SCORES;
    if (!record(KC.pipe[P_QSA], scores, 6, &pass, sizeof pass, (uint32_t)p->S, (uint32_t)((blocks + QSA_WG - 1) / QSA_WG), 1)) return 0;
    pass.mode = QSA_SELECT;
    return record(KC.pipe[P_QSA], select, 6, &pass, sizeof pass, (uint32_t)p->S, 1, 1);
}
int vkc_ple(VkcBuf *keys, VkcBuf *hyp, VkcBuf *val, VkcBuf *prm, VkcBuf *gated, VkcBuf *normv,
            VkcBuf *conv, VkcBuf *ring, const VkcPle *p) {
    KC.kind = PK_PLE;
    if (p->mode == 0) {
        VkcBind bd[6] = {B(keys, 0), B(hyp, 0), B(val, 0), B(prm, 0), B(gated, 1), B(normv, 1)};
        return record(KC.pipe[P_PLE], bd, 6, p, sizeof *p, (uint32_t)(p->S * p->C), 1, 1);
    }
    if ((p->CK - 1) * p->NG > 32) return 0;
    VkcBind bd[8] = {{VK_NULL_HANDLE, 0, 0, 0}, B(hyp, 1), {VK_NULL_HANDLE, 0, 0, 0}, {VK_NULL_HANDLE, 0, 0, 0},
                     B(gated, 0), B(normv, 0), B(conv, 0), B(ring, 1)};
    uint32_t gx, gy; grid(((uint64_t)p->C * p->H + 255) / 256, &gx, &gy);
    return record(KC.pipe[P_PLE], bd, 8, p, sizeof *p, gx, gy, 1);
}

/* ---- multi-head latent attention ------------------------------------------------- */
int vkc_mla_ready(void) { return vkc_ready() && KC.mla_ok; }

/* ---- a decode's MLA core split over its positions (chain_mla_dec.comp) -------------
 * chain_mla.comp's core gives a (head, row) one workgroup that walks every position (64
 * heads of a 512-float latent: 1.75 ms a layer at 2000 positions on an RTX PRO 4000, the
 * same with 128 heads). From VKC_MLA_DEC_MIN positions a call of up to 4 rows runs in
 * blocks of positions, as chain_attn_dec.comp does (selection lists too: blocks of the
 * list's entries). COLI_VK_MLA_DEC=0 keeps chain_mla.comp's. */
#define VKC_MLA_DEC_MIN 128
static VkPipeline mdec_pipe(void) {
    static int on = -1;
    if (on < 0) { const char *e = getenv("COLI_VK_MLA_DEC"); on = !(e && *e == '0'); }
    if (!on || KC.mdec || KC.mdec_tried) return KC.mdec;
    KC.mdec_tried = 1;
    if ((KC.mod_mdec = load_module(KC.core.spv_path, "chain_mla_dec.spv"))) KC.mdec = make_pipe(KC.mod_mdec, NULL);
    return KC.mdec;
}
static int mla_dec(VkcBuf *qabs, VkcBuf *qr, VkcBuf *lat, VkcBuf *rope, VkcBuf *sel, VkcBuf *clat, const VkcMlaCore *b) {
    int S = b->S, n = b->pos_base + S - b->kv_start;
    if (b->sel_row > 0 && b->sel_row - 1 > n) n = b->sel_row - 1;   /* a list holds at most its row's room */
    if (S > 4 || n < VKC_MLA_DEC_MIN || !mdec_pipe()) return 0;
    int L = 64, nb = (n + L - 1) / L;
    while (L < 256 && (int64_t)b->H * nb > 4096) { L *= 2; nb = (n + L - 1) / L; }
    VkcBuf *xs = adec_scratch((size_t)S * b->H * nb * (b->K + 2) * 4);
    if (!xs) return 0;
    struct { int mode; VkcMlaCore b; int dmode, L, nb; } pc = {0, *b, 0, L, nb};
    VkcBind bd[8] = {B(qabs, 0), B(qr, 0), B(lat, 0), B(b->R > 0 ? rope : NULL, 0), B(b->sel_row > 0 ? sel : NULL, 0),
                     B(clat, 1), B(NULL, 0), B(xs, 1)};
    int ok = record(KC.mdec, bd, 8, &pc, sizeof pc, (uint32_t)b->H, (uint32_t)nb, (uint32_t)S);
    pc.dmode = 1;
    bd[7].written = 0;
    ok = ok && record(KC.mdec, bd, 8, &pc, sizeof pc, (uint32_t)b->H, (uint32_t)S, 1);
    KC.st.attn_dec += ok;
    return ok;
}
int vkc_mla_core(VkcBuf *qabs, VkcBuf *qr, VkcBuf *lat, VkcBuf *rope, VkcBuf *sel, VkcBuf *clat, const VkcMlaCore *p) {
    KC.kind = PK_MLA;
    if (!KC.mla_ok || p->S < 1 || p->H < 1 || p->K < 1 || p->K > 1024 || p->R < 0 || p->R > 128 ||
        (p->R > 0 && !rope) || p->pos_base < 0 || p->kv_start < 0 || p->kv_start > p->pos_base) return 0;
    typedef struct { int mode; VkcMlaCore b; } CorePC;
    CorePC pc = {0, *p};
    if (!sel) pc.b.sel_row = 0;
    if (mla_dec(qabs, qr, lat, rope, sel, clat, &pc.b)) return 1;   /* a decode's rows: in blocks of positions */
    VkcBind bd[6] = {B(qabs, 0), B(qr, 0), B(lat, 0), B(p->R > 0 ? rope : NULL, 0), B(pc.b.sel_row > 0 ? sel : NULL, 0),
                     B(clat, 1)};
    /* a long prompt chunk in slices of rows, a submission each (attn_slice_rows) */
    int S = p->S, rr = attn_slice_rows(S, p->pos_base, (double)p->H * (p->K + p->R), 1), ok = 1;
    for (int r0 = 0; ok && r0 < S; r0 += rr) {
        CorePC x = pc;
        x.b.S = S - r0 < rr ? S - r0 : rr; x.b.pos_base = p->pos_base + r0;
        x.b.qa_off += r0 * p->qa_row; x.b.qr_off += r0 * p->qr_row; x.b.o_off += r0 * p->o_row;
        if (pc.b.sel_row > 0) x.b.sel_off += r0 * pc.b.sel_row;
        if (r0 && !attn_slice_next()) return 0;
        ok = record(KC.mpipe[PM_MLA], bd, 6, &x, sizeof x, (uint32_t)p->H, (uint32_t)x.b.S, 1);
    }
    return ok;
}
static int mla_rowop(int mode, VkcBuf *x, VkcBuf *aux, VkcBuf *y, const VkcMlaRow *p) {
    if (!KC.mla_ok || !open_frame() || KC.lost || p->nseg < 0) return 0;
    if (p->nseg == 0) return 1;
    if (mode == 1 && (p->rd < 2 || p->rd > 256 || (p->rd & 1) || p->rd > p->seg_len)) return 0;
    if (mode == 2 && p->seg_len < 1) return 0;
    struct { int mode; VkcMlaRow b; int copy_rest; } pc = {mode, *p, x != y};
    VkcBind bd[6] = {B(x, x == y), B(aux, 0), B(NULL, 0), B(NULL, 0), B(NULL, 0), B(y, 1)};
    uint32_t gx, gy; grid((uint64_t)p->nseg, &gx, &gy);
    return record(KC.mpipe[PM_MLA], bd, 6, &pc, sizeof pc, gx, gy, 1);
}
int vkc_mla_rope(VkcBuf *x, VkcBuf *cs, VkcBuf *y, const VkcMlaRow *p) { KC.kind = PK_ROPE; return mla_rowop(1, x, cs, y, p); }
int vkc_mla_lnorm(VkcBuf *x, VkcBuf *prm, VkcBuf *y, const VkcMlaRow *p) { KC.kind = PK_NORM; return mla_rowop(2, x, prm, y, p); }

int vkc_mla_hgemv(ColiVkTensor *t, VkcBuf *x, VkcBuf *y, VkcBuf *gate, const VkcHgemv *p) {
    KC.kind = PK_MLAW;
    ColiVkTensorInfo ti;
    if (!KC.mla_ok || !open_frame() || KC.lost || !vkc_tensor_info(t, &ti) || p->S < 1 || p->S > 65535 ||
        p->H < 1 || p->H > 65535 || p->n < 1 || (p->trans && p->n > 1024)) return 0;
    int f = ti.fmt;
    if (!(f == 1 || f == 2 || f == 4 || f == 5 || f == 7 || f == 10 || f == 11 || f == 12 || f == 13)) return 0;
    if ((f == 4 || f == 7 || f == 12 || f == 13) && ti.gs < 1) return 0;
    if ((int64_t)(p->H - 1) * p->hstride + p->hoff + p->n > (int64_t)ti.O) return 0;   /* the rows exist */
    struct { int mode, fmt, I, rowWords, gs, S, H, n, hstride, hoff, x_off, x_row, x_seg, y_off, y_row, y_seg,
             g_off, g_row, has_gate; } pc = {p->trans, f, ti.I, ti.rowWords, ti.gs, p->S, p->H, p->n, p->hstride, p->hoff,
             p->x_off, p->x_row, p->x_seg, p->y_off, p->y_row, p->y_seg, p->g_off, p->g_row, gate && p->has_gate};
    VkcBind bd[5] = {B(x, 0), {(VkBuffer)ti.wbuf, 0, 0, 0}, {(VkBuffer)ti.sbuf, 0, 0, 0}, B(y, 1), B(pc.has_gate ? gate : NULL, 0)};
    if (p->trans) return record(KC.mpipe[PM_HGEMV], bd, 5, &pc, sizeof pc, (uint32_t)p->H, (uint32_t)p->S, 1);
    /* rows: a subgroup each, grid-stride (at least four subgroups a workgroup) */
    int64_t rows = (int64_t)p->H * p->n, wg = (rows + 7) / 8;
    if (wg > 4096) wg = 4096;
    return record(KC.mpipe[PM_HGEMV], bd, 5, &pc, sizeof pc, (uint32_t)wg, (uint32_t)p->S, 1);
}

#define VKC_DSA_GRID 32   /* positions (pools, columns) from which the indexers score on a grid */
static int dsa_grid_on(void) {   /* COLI_VK_DSA_GRID=0: every row's scores in its own workgroup, as before */
    static int on = -1;
    if (on < 0) { const char *e = getenv("COLI_VK_DSA_GRID"); on = !(e && *e == '0'); }
    return on;
}
int vkc_dsa_select(VkcBuf *iq, VkcBuf *hw, VkcBuf *keys, VkcBuf *sc, VkcBuf *sel, const VkcDsa *p) {
    KC.kind = PK_DSA;
    if (!KC.mla_ok || p->S < 1 || p->S > 65535 || p->IH < 1 || p->IH > 64 || p->ID < 1 || p->IH * p->ID > 4096 ||
        p->topk < 1 || p->pos_base < 0 || p->sc_row < p->pos_base + p->S ||
        p->sel_row < 1 + (p->topk < p->pos_base + p->S ? p->topk : p->pos_base + p->S)) return 0;
    struct { int mode; VkcDsa b; int scored; } pc = {0, *p, 0};
    VkcBind bd[5] = {B(iq, 0), B(hw, 0), B(keys, 0), B(sc, 1), B(sel, 1)};
    /* past VKC_DSA_GRID positions the scores on a grid of 256 positions a workgroup first
     * (mode 3, the same bits), then each row's selection from them */
    int nk = p->pos_base + p->S;
    if (dsa_grid_on() && nk > p->topk && nk >= VKC_DSA_GRID) {
        pc.mode = 3;
        int P = 256 / p->IH > 0 ? 256 / p->IH : 1;   /* positions a workgroup: a thread a (position, head) */
        if (!record(KC.mpipe[PM_DSA], bd, 5, &pc, sizeof pc, (uint32_t)((nk + P - 1) / P), (uint32_t)p->S, 1)) return 0;
        pc.mode = 0; pc.scored = 1;
    }
    return record(KC.mpipe[PM_DSA], bd, 5, &pc, sizeof pc, (uint32_t)p->S, 1, 1);
}

/* k-pooling (chain_dsa.comp modes 1 and 2) */
int vkc_dsa_pool_keys(VkcBuf *keys, VkcBuf *gates, VkcBuf *prm, VkcBuf *pk, const VkcDsaPool *p) {
    KC.kind = PK_DSA;
    if (!KC.mla_ok || p->np < 0 || p->pool < 1 || p->p0 < 0 || p->ID < 1) return 0;
    if (p->np == 0) return open_frame() && !KC.lost;
    struct { int mode; VkcDsaPool b; } pc = {1, *p};
    VkcBind bd[8] = {B(NULL, 0), B(NULL, 0), B(keys, 0), B(NULL, 0), B(NULL, 0), B(pk, 1), B(gates, 0), B(prm, 0)};
    return record(KC.mpipe[PM_DSA], bd, 8, &pc, sizeof pc, (uint32_t)p->np, 1, 1);
}
int vkc_dsa_pool_select(VkcBuf *iq, VkcBuf *hw, VkcBuf *pk, VkcBuf *sc, VkcBuf *sel, const VkcDsaPick *p) {
    KC.kind = PK_DSA;
    if (!KC.mla_ok || p->S < 1 || p->S > 65535 || p->IH < 1 || p->IH > 64 || p->ID < 1 || p->IH * p->ID > 4096 ||
        p->pool < 1 || p->topk < p->pool || p->topk % p->pool || p->topk / p->pool > 1024 || p->pos_base < 0 ||
        p->sc_row < (p->pos_base + p->S) / p->pool || p->sel_row < 1 + p->topk + (p->tail ? p->pool - 1 : 0)) return 0;
    struct { int mode; VkcDsaPick b; int scored; } pc = {2, *p, 0};
    VkcBind bd[6] = {B(iq, 0), B(hw, 0), B(NULL, 0), B(sc, 1), B(sel, 1), B(pk, 0)};
    /* past VKC_DSA_GRID pools their scores on a grid first (mode 4, the same bits) */
    int np = (p->pos_base + p->S) / p->pool;
    if (dsa_grid_on() && np >= VKC_DSA_GRID) {
        pc.mode = 4;
        int P = 256 / p->IH > 0 ? 256 / p->IH : 1;   /* pools a workgroup: a thread a (pool, head) */
        if (!record(KC.mpipe[PM_DSA], bd, 6, &pc, sizeof pc, (uint32_t)((np + P - 1) / P), (uint32_t)p->S, 1)) return 0;
        pc.mode = 2; pc.scored = 1;
    }
    return record(KC.mpipe[PM_DSA], bd, 6, &pc, sizeof pc, (uint32_t)p->S, 1, 1);
}

/* ---- Kimi Delta Attention (chain_kda.comp) ------------------------------------------ */
int vkc_kda_ready(void) { return vkc_ready() && KC.mpipe[PM_KDA]; }
int vkc_kda_conv(VkcBuf *in, VkcBuf *w, VkcBuf *win, VkcBuf *out, const VkcKdaConv *p) {
    KC.kind = PK_KDA;
    if (!KC.mpipe[PM_KDA] || p->K < 1 || p->K > 8 || p->P < 1 || p->C < 1 || p->S < 1) return 0;
    struct { int mode; VkcKdaConv b; } pc = {0, *p};
    VkcBind bd[4] = {B(in, 0), B(w, 0), B(win, 1), B(out, 1)};
    uint32_t gx, gy; grid(((uint64_t)p->C + 127) / 128, &gx, &gy);
    return record(KC.mpipe[PM_KDA], bd, 4, &pc, sizeof pc, gx, gy, 1);
}
static VkPipeline kda_pipe(int KD) {
    for (int i = 0; i < KC.nkdarec; i++) if (KC.kdarec_kd[i] == KD) return KC.kdarec[i];
    if (KC.nkdarec == VKC_KDA_MAX || !KC.mmod[PM_KDA]) return VK_NULL_HANDLE;
    int32_t kd = KD;
    VkSpecializationMapEntry me = {0, 0, 4};
    VkSpecializationInfo si = {1, &me, 4, &kd};
    VkPipeline p = make_pipe(KC.mmod[PM_KDA], &si);
    if (!p) return VK_NULL_HANDLE;
    KC.kdarec[KC.nkdarec] = p; KC.kdarec_kd[KC.nkdarec++] = KD;
    return p;
}
/* chain_kda_rows.spv for KD: its rows a block (R) as many as the shared memory holds, at
 * most 16 (NULL past VKC_KDA_MAX keys, or when COLI_VK_KDA_ROWS=0) */
static VkPipeline kdar_pipe(int KD) {
    static int on = -1;
    if (on < 0) { const char *e = getenv("COLI_VK_KDA_ROWS"); on = !(e && *e == '0'); }
    if (!on) return VK_NULL_HANDLE;
    for (int i = 0; i < KC.nkdar; i++) if (KC.kdar_kd[i] == KD) return KC.kdar[i];
    if (KC.nkdar == VKC_KDA_MAX) return VK_NULL_HANDLE;
    if (!KC.mod_kdar) {
        if (KC.kdar_tried) return VK_NULL_HANDLE;
        KC.kdar_tried = 1;
        if (!(KC.mod_kdar = load_module(KC.core.spv_path, "chain_kda_rows.spv"))) return VK_NULL_HANDLE;
    }
    size_t row = ((size_t)3 * KD + 2 * 128 + 4) * 4;   /* q, k, alpha, v, o and four scalars a row */
    int R = 16;
    while (R > 1 && (size_t)R * row > kc_shared_limit()) R /= 2;
    int32_t v[2] = {KD, R};
    VkSpecializationMapEntry me[2] = {{0, 0, 4}, {1, 4, 4}};
    VkSpecializationInfo si = {2, me, sizeof v, v};
    VkPipeline p = make_pipe(KC.mod_kdar, &si);
    KC.kdar[KC.nkdar] = p; KC.kdar_kd[KC.nkdar++] = KD;   /* a failed make is remembered as NULL */
    return p;
}
int vkc_kda_rec(int KD, VkcBuf *m, VkcBuf *f, VkcBuf *b, VkcBuf *g, VkcBuf *prm, VkcBuf *st, VkcBuf *y, const VkcKdaRec *p) {
    return vkc_kda_rec_flags(KD, m, f, b, g, prm, st, y, p, 0);
}
int vkc_kda_rec_flags(int KD, VkcBuf *m, VkcBuf *f, VkcBuf *b, VkcBuf *g, VkcBuf *prm, VkcBuf *st, VkcBuf *y,
                      const VkcKdaRec *p, int flags) {
    KC.kind = PK_KDA;
    if (!KC.mpipe[PM_KDA] || KD < 1 || KD > 256 || p->VD < 1 || p->VD > 128 || p->H < 1 || p->S < 1 ||
        (flags & ~(VKC_KDA_EXP_A | VKC_KDA_K3))) return 0;
    /* a prompt chunk's rows a block at a time (chain_kda_rows: the same bits); a decode's
     * or a verify's few through chain_kda */
    VkPipeline pipe = p->S >= VKC_KDA_ROWS ? kdar_pipe(KD) : VK_NULL_HANDLE;
    if (!pipe) pipe = kda_pipe(KD);
    if (!pipe) return 0;
    struct { int mode; VkcKdaRec b; int flags; } pc = {1, *p, flags};
    VkcBind bd[7] = {B(m, 0), B(prm, 0), B(st, 1), B(y, 1), B(f, 0), B(b, 0), B(g, 0)};
    return record(pipe, bd, 7, &pc, sizeof pc, (uint32_t)p->H, 1, 1);
}

/* ---- manifold-constrained hyper-connections (chain_mhc.comp) ------------------------ */
int vkc_mhc_ready(void) { return vkc_ready() && KC.mpipe[PM_MHC]; }
int vkc_mhc(int mode, VkcBuf *x, VkcBuf *m, VkcBuf *hp, VkcBuf *prm, VkcBuf *y, const VkcMhc *p) {
    KC.kind = PK_MHC;
    if (!KC.mpipe[PM_MHC] || mode < 0 || mode > 4 || p->S < 0 || (mode != 4 && (p->H < 1 || p->H > 8 || p->D < 1))) return 0;
    if (mode == 0 && p->iters < 1) return 0;
    struct { int mode; VkcMhc b; } pc = {mode, *p};
    VkcBind bd[5] = {B(x, 0), B(m, 0), B(hp, mode == 0), B(prm, 0), B(y, mode != 0)};
    uint64_t n = mode == 4 ? (uint64_t)p->n : mode == 2 ? (uint64_t)p->S * p->H * p->D : (uint64_t)p->S * p->D;
    if (mode == 0) return p->S == 0 || record(KC.mpipe[PM_MHC], bd, 5, &pc, sizeof pc, (uint32_t)p->S, 1, 1);
    uint32_t gx, gy; grid((n + 255) / 256, &gx, &gy);
    return n == 0 || record(KC.mpipe[PM_MHC], bd, 5, &pc, sizeof pc, gx, gy, 1);
}

/* ---- attention residuals and SiTU-GLU (chain_ares.comp) ---------------------------- */
/* the router's pick (chain_route.comp): one workgroup per row */
int vkc_route_ready(void) { return vkc_ready() && KC.mpipe[PM_ROUTE]; }
int vkc_route(VkcBuf *logits, VkcBuf *bias, VkcBuf *ids, VkcBuf *w, const VkcRoute *p) {
    KC.kind = PK_ROUTE;
    if (!KC.mpipe[PM_ROUTE] || p->S < 1 || p->E < 1 || p->E > VKC_ROUTE_EMAX || p->K < 1 || p->K > VKC_ROUTE_KMAX ||
        p->K > p->E || p->kind < 0 || p->kind > 2 || p->has_bias < 0 || p->has_bias > 2 || (p->has_bias && !bias) || !(p->temp > 0.0f) ||
        (p->n_group > 1 && (p->n_group > 64 || p->E % p->n_group || p->topk_group < 1 || p->topk_group > p->n_group)))
        return 0;
    /* without a bias the logits stand in for it, at their own offset: a device that reads
     * both sides of the shader's choice stays in bounds (the 256-byte dummy would not) */
    VkcRoute q = *p;
    if (!q.has_bias) q.b_off = q.l_off;
    static int sg = -1;   /* COLI_VK_ROUTE_SG=0: every row on the workgroup's path */
    if (sg < 0) { const char *e = getenv("COLI_VK_ROUTE_SG"); sg = !(e && *e == '0'); }
    if (!sg) q.wg = 1;
    VkcBind bd[4] = {B(logits, 0), B(q.has_bias ? bias : logits, 0), B(ids, 1), B(w, 1)};
    uint32_t gx, gy; grid((uint64_t)q.S, &gx, &gy);
    return record(KC.mpipe[PM_ROUTE], bd, 4, &q, sizeof q, gx, gy, 1);
}
/* the routed experts (chain_moe.comp): gate and up of every (row, slot), then down and
 * the weighted sum of each row's slots */
/* chain_moe's pipeline (bd: the block formats' paths) staging rows of up to n floats (its
 * stage(): n plus a vec4 of padding every 8; n < 0 the device's largest), made on first use */
/* fmt: one of our formats (1, 2 or 4) that a pipeline of its own decodes (FMTC), else 0
 * for the pipeline of every format; the block formats' pipeline (bd) has every format */
static int moe_fi(int fmt) { return fmt == 1 ? 1 : fmt == 2 ? 2 : fmt == 4 ? 3 : 0; }
/* the slot of a block formats' pipeline (i8: the integer form's) of staging b for a layer's
 * types tg, td (a GGUF type plus one: chain_moe.comp's BTG, BTD); NULL when the slots ran
 * out, the pipeline of every type then */
static VkPipeline *moe_bt_slot(int i8, int b, int tg, int td, uint8_t **tried) {
    int i = 0;
    while (i < KC.moe_nbt && !(KC.moe_bt[i].i8 == i8 && KC.moe_bt[i].b == b && KC.moe_bt[i].tg == tg && KC.moe_bt[i].td == td)) i++;
    if (i == KC.moe_nbt) {
        if (i == (int)(sizeof KC.moe_bt / sizeof *KC.moe_bt)) return NULL;
        KC.moe_bt[i].i8 = (int8_t)i8; KC.moe_bt[i].b = (int8_t)b; KC.moe_bt[i].tg = (uint8_t)tg; KC.moe_bt[i].td = (uint8_t)td;
        KC.moe_bt[i].tried = 0; KC.moe_bt[i].p = VK_NULL_HANDLE;
        KC.moe_nbt++;
    }
    *tried = &KC.moe_bt[i].tried;
    return &KC.moe_bt[i].p;
}
/* tg, td: the block formats' pipeline of those types alone (moe_bt_slot), 0 every type's */
static VkPipeline moe_pipe_f(int bd, int n, int fmt, int tg, int td) {
    int cap = (bd ? KC.moe_xs_bd : KC.moe_xs) / 4;
    if (!cap) return VK_NULL_HANDLE;
    int np = (n + 255) & ~255;   /* the 16-byte way pads a row to its words (wide()) */
    int need = n > 0 ? (np + np / 8 + 4 + 3) / 4 : cap, b = stage_of(need, cap);
    int fi = bd ? 0 : moe_fi(fmt);
    VkPipeline *pp = fi ? &KC.moe_f[fi - 1][b] : &KC.moe[bd][b];
    uint8_t *tried = fi ? &KC.moe_f_tried[fi - 1][b] : &KC.moe_tried[bd][b];
    if (!bd || !tg || !td) tg = td = 0;
    if (tg) {
        uint8_t *t2; VkPipeline *p2 = moe_bt_slot(0, b, tg, td, &t2);
        if (p2) { pp = p2; tried = t2; } else tg = td = 0;
    }
    if (*pp || *tried) return *pp;
    *tried = 1;
    if (!KC.mod_moe && !(KC.mod_moe = load_module(KC.core.spv_path, "chain_moe.spv"))) return VK_NULL_HANDLE;
    int32_t v[5] = {4 * stage_vec4(b, cap), bd, fi ? fmt : 0, tg, td};
    VkSpecializationMapEntry me[5] = {{0, 0, 4}, {1, 4, 4}, {3, 8, 4}, {4, 12, 4}, {5, 16, 4}};
    VkSpecializationInfo si = {5, me, sizeof v, v};
    double t0 = vkc_now_ms();
    *pp = make_pipe(KC.mod_moe, &si);
    double dt = vkc_now_ms() - t0;
    if (dt > 5000.0) fprintf(stderr, "[VK] chain: chain_moe.spv%s took %.1f s to build\n", bd ? " (block formats)" : "", dt / 1e3);
    return *pp;
}
static VkPipeline moe_pipe(int bd, int n) { return moe_pipe_f(bd, n, 0, 0, 0); }
/* chain_moe_gv.comp for format fmt (1, 2, 4) and input rows of n floats (a multiple of 32):
 * NULL where the row does not fit the staging or COLI_VK_MOE_GV=0. int4 with a scale per
 * gs inputs (4) whose rows are a whole number of steps a lane at coli_vk_gv_lanes, up to 16
 * (Qwen3.6's 2048 and 512 inputs: 8): the shader's HQ build, each lane's steps loaded ahead
 * with the loop's bits (a Radeon 8060S's Qwen3.6 layer, one row: 160 -> 100 us).
 * COLI_VK_MOE_GV_HQ=0 keeps the loop. */
static int g_gv_hq = -1;   /* the fast path: -1 from COLI_VK_MOE_GV_HQ (vkc_moe_gv_hq) */
void vkc_moe_gv_hq(int on) { g_gv_hq = on; }
static VkPipeline moe_gv_pipe(int fmt, int n) {
    static int on = -1;
    if (on < 0) { const char *e = getenv("COLI_VK_MOE_GV"); on = !(e && *e == '0'); }
    int hq = g_gv_hq;
    if (hq < 0) { const char *e = getenv("COLI_VK_MOE_GV_HQ"); hq = !(e && *e == '0'); }
    int fi = moe_fi(fmt), cap = KC.moe_xs / 4;
    if (!on || !fi || !cap || n < 32 || n % 32) return VK_NULL_HANDLE;
    int need = n / 4 + n / 32 + 1, b = stage_of(need, cap);
    if (need > stage_vec4(b, cap)) return VK_NULL_HANDLE;
    int steps = n / 32, l = coli_vk_gv_lanes(fmt, n), nq = hq && fmt == 4 && steps % l == 0 ? steps / l : 0;
    if (nq > 16) nq = 0;
    VkPipeline *pp = nq ? &KC.moe_gvq[b][nq] : &KC.moe_gv[fi - 1][b];
    uint8_t *tried = nq ? &KC.moe_gvq_tried[b][nq] : &KC.moe_gv_tried[fi - 1][b];
    if (*pp || *tried) return *pp;
    *tried = 1;
    if (!KC.mod_moe_gv && !(KC.mod_moe_gv = load_module(KC.core.spv_path, "chain_moe_gv.spv"))) return VK_NULL_HANDLE;
    int32_t v[4] = {stage_vec4(b, cap), fmt, nq > 0, nq ? nq : 8};
    VkSpecializationMapEntry me[4] = {{0, 0, 4}, {1, 4, 4}, {2, 8, 4}, {3, 12, 4}};
    VkSpecializationInfo si = {4, me, sizeof v, v};
    return *pp = make_pipe(KC.mod_moe_gv, &si);
}
/* The routed experts in the CPU's integer form (chain_moe_i8.spv, bdstep_i8.glsl): the
 * types it decodes, and its pipeline for rows of up to n inputs (their codes in the
 * smallest kc_stage bucket of uvec4s that holds them, the float staging the least). On by
 * default where the device's packed 4 x 8 dot is accelerated. */
static int moe_i8_type(int fmt) {
    int t = fmt - 100;
    return t == 8 || t == 10 || t == 11 || t == 12 || t == 13 || t == 14 || t == 16 || t == 17 || t == 18 || t == 19 || t == 20 || t == 21 ||
           t == 22 || t == 23 || t == 29 || t == 42;
}
static VkPipeline moe_i8_pipe(int n, int tg, int td) {
    if (!(KC.idot_set >= 0 ? KC.idot_set && KC.idot : KC.idot == 2) || !KC.moe_xs_bd) return VK_NULL_HANDLE;
    int b = codes_of(n > 0 ? n : 8192);
    if (b >= KC_STAGES) return VK_NULL_HANDLE;
    int xq = kc_codes[b];
    /* its shared memory: the codes, scales and sums, bdstep.glsl's tables */
    if ((size_t)(xq + xq / 8 + 1) * 16 + (size_t)(xq / 2 + 1) * 4 + (size_t)(xq + 1) * 4 + 8768 + 16 > kc_shared_limit()) return VK_NULL_HANDLE;
    VkPipeline *pp = &KC.moe_i8[b];
    uint8_t *tried = &KC.moe_i8_tried[b];
    if (!tg || !td) tg = td = 0;
    if (tg) {
        uint8_t *t2; VkPipeline *p2 = moe_bt_slot(1, b, tg, td, &t2);
        if (p2) { pp = p2; tried = t2; } else tg = td = 0;
    }
    if (*pp || *tried) return *pp;
    *tried = 1;
    if (!KC.mod_moe_i8 && !(KC.mod_moe_i8 = load_module(KC.core.spv_path, "chain_moe_i8.spv"))) return VK_NULL_HANDLE;
    int32_t v[5] = {4, 1, xq, tg, td};
    VkSpecializationMapEntry me[5] = {{0, 0, 4}, {1, 4, 4}, {2, 8, 4}, {4, 12, 4}, {5, 16, 4}};
    VkSpecializationInfo si = {5, me, sizeof v, v};
    *pp = make_pipe(KC.mod_moe_i8, &si);
    return *pp;
}
void vkc_moe_set_idot(int on) { KC.idot_set = on; }
int vkc_moe_idot(void) { return KC.idot; }
/* chain_moe can run here: its module and a staging size (no pipeline made yet) */
static int moe_build(void) {
    if (!KC.moe_xs) return 0;
    if (!KC.mod_moe) KC.mod_moe = load_module(KC.core.spv_path, "chain_moe.spv");
    return KC.mod_moe != VK_NULL_HANDLE;
}

/* The pipelines this file makes at first use, made now (vk_chain.h, vkc_prewarm): the block
 * formats' GEMV (one to four rows of x) and tiled GEMM for the types asked, chain_moe's
 * pipeline, DeltaNet's recurrence. Each but the recurrence is launched once, on work whose
 * output nothing reads. Made at first use, after the expert tier and a prompt's scratch had
 * filled a discrete card's memory, a pipeline got its code and its first launch's resources
 * in system memory, and every launch after read them over the link: an RTX PRO 4000 with
 * Qwen3.8-Flash-Next's tier filling its 24 GiB ran chain_moe's three passes in 190, 115 and
 * 65 us a layer, against 65, 35 and 6 in a process with room, and its one-row decode went
 * from 13.9 to 27.6 tok/s with them made before the tier. */
static int rows_make(void);
static VkPipeline grp_pipe(int v, int t, int coop);
static VkPipeline grpq_pipe(int v, int coop);
static VkPipeline grpw_pipe(int v, int t);
static int grp_wide_rows(void);
int vkc_prewarm(const VkcPrewarm *pw) {
    const char *e = getenv("COLI_VK_CHAIN_PREWARM");   /* 1 on any device (tests), 0 never */
    if (!vkc_ready() || !pw || !(e && *e ? *e != '0' : KC.discrete)) return 0;
    double t0 = vkc_now_ms();
    int n = 0;
    VkcBuf *x = vkc_buf(64 * 256 * 4, VKC_DEV), *w = vkc_buf((size_t)64 << 10, VKC_DEV), *y = vkc_buf(64 * 4, VKC_DEV),
           *ids = vkc_buf(16, VKC_DEV), *rw = vkc_buf(16, VKC_DEV), *tab = vkc_buf(COLI_VK_XB_ENTRY * 4, VKC_DEV),
           *hid = vkc_buf(2 * 64 * 4, VKC_DEV);
    VkcBuf *fo = NULL;   /* the attention's output (attn_hd) */
    int ok = x && w && y && ids && rw && tab && hid && vkc_begin();
    VkBuffer bt = ok && pw->types ? (VkBuffer)coli_vk_block_table_dev(g_kd) : VK_NULL_HANDLE;
    if (ok && KC.bd && bt) {
        VkcBind bd[4] = {{x->buf, 0, 0, 0}, {w->buf, 0, 0, 0}, {bt, 0, 0, 0}, {y->buf, 0, 0, 1}};
        KC.kind = PK_GEMV;
        for (int t = 0; t < 64 && ok; t++) {
            if (!(pw->types >> t & 1)) continue;
            /* one row of 256 weights (the zeroed buffer), in the staging size of each width the
             * model's matrices have (256 when none is named) */
            for (int nr = 1; nr <= 4 && ok; nr++) {
                VkPipeline made[KC_STAGES + 1]; int nm = 0;
                for (int k = 0; k < (pw->n_gemv_i > 0 ? pw->n_gemv_i : 1) && ok; k++) {
                    int I = pw->n_gemv_i > 0 ? pw->gemv_i[k] : 256;
                    VkPipeline v = bdv8_pipe(t, nr, 64, I);
                    if (!v) v = bdv_pipe(t, nr, 64, I);
                    int seen = 0;
                    for (int j = 0; j < nm; j++) seen |= made[j] == v;
                    if (seen) continue;
                    if (nm <= KC_STAGES) made[nm++] = v;
                    struct { int fmt, S, I, O, rowWords, gs, lpr, nr; } pc = {100 + t, nr, 256, 1, 64, 0, 1, nr};
                    if (v) { ok = record(v, bd, 4, &pc, sizeof pc, 1, 1, 1); n += ok; }
                }
            }
            VkPipeline g = KC.mod_bdg ? bdg_pipe(t, 64) : VK_NULL_HANDLE;   /* the same row against 64 rows of x */
            struct VkcPC pc = {100 + t, 64, 256, 1, 64, 0};
            if (g && ok) { ok = record(g, bd, 4, &pc, sizeof pc, 1, 1, 1); n += ok; }
            VkPipeline gw = g && g == KC.bdgc[t] ? bdgw_pipe(t) : VK_NULL_HANDLE;   /* and on the 128 tile */
            if (gw && ok) { ok = record(gw, bd, 4, &pc, sizeof pc, 1, 1, 1); n += ok; }
        }
    }
    /* the experts' (1 colibri's formats, 2 the block formats'): a table without a resident
     * entry, so every workgroup leaves at once */
    int mn = pw->moe_d > pw->moe_i ? pw->moe_d : pw->moe_i > 0 ? pw->moe_i : -1;   /* the experts' staging */
    /* the block formats': a pipeline a pair of the file's gate/up and down types (the
     * integer form's where both have it, as vkc_moe picks), every type's past 16 pairs */
    int npr = 0;
    for (int g = 0; g < 64 && pw->moe == 2; g++) for (int d = 0; d < 64; d++) npr += (pw->ex_gu >> g & 1) && (pw->ex_dn >> d & 1);
    if (npr > 16) npr = 0;
    if (ok && (pw->moe == 1 || pw->moe == 2) && (npr || moe_pipe(pw->moe == 2, mn))) {
        VkcMoe p = {.S = 1, .K = 1, .D = 64, .I = 64, .x_row = 64, .i_row = 1, .w_row = 1, .y_row = 64, .blocks = pw->moe == 2,
                    .stage_n = mn};
        for (int g = 0; g < 64 && npr && ok; g++) for (int d = 0; d < 64 && ok; d++) {
            if (!((pw->ex_gu >> g & 1) && (pw->ex_dn >> d & 1))) continue;
            p.gfmt = 100 + g; p.dfmt = 100 + d;
            ok = vkc_moe(x, ids, rw, tab, hid, y, &p); n += ok;
        }
        p.gfmt = p.dfmt = 0;
        if (!npr) { ok = ok && vkc_moe(x, ids, rw, tab, hid, y, &p); n += ok; }
        /* our formats' pipelines of one format each (1, 2 and 4: what a container's experts are) */
        for (int f = 1; f <= 4 && ok && pw->moe == 1; f++) {
            if (!moe_fi(f)) continue;
            p.gfmt = p.dfmt = f;
            ok = vkc_moe(x, ids, rw, tab, hid, y, &p); n += ok;
            /* and chain_moe_gv.comp's at the layer's widths (its staging sized by them) */
            if (pw->moe_d > 0 && moe_gv_pipe(f, pw->moe_d)) n++;
            if (pw->moe_i > 0 && moe_gv_pipe(f, pw->moe_i)) n++;
        }
    }
    /* DeltaNet's recurrence, made (its launch wants a layer's state) */
    if (ok && pw->dn_kd > 0 && pw->dn_kd <= 256) n += (dnrec_pipe(pw->dn_kd) != VK_NULL_HANDLE) + (dnrr_pipe(pw->dn_kd) != VK_NULL_HANDLE);
    /* a prompt chunk's experts (vkc_moe_rows), made and launched once with no rows: a
     * pipeline made later, with the tier holding the card, slowed every launch after it,
     * the decode's too (Qwen3.8-Flash-Next on an RTX PRO 4000: 37.7 -> 25.5 tok/s once its
     * first prompt made these), and one made here but first launched there still cost the
     * decode 6 to 16% (38.9 -> 36.6, 40.6 -> 34.2 tok/s). The plan with no pairs leaves at
     * once; the grouped GEMMs get an item of expert 0 with no rows and no input (I 0), whose
     * table entry has only the block table's address for the decoders' tables: nothing is
     * read past them, nothing written. */
    if (ok && (pw->ex_gu || pw->ex_dn || pw->ex_ours) && rows_make()) {
        struct { int mode, S, K, E, D, I, t_off, imax, i_off, i_row, w_off, w_row, y_off, y_row; uint32_t tab_lo, tab_hi; int tm, bo1, bo0; } rp =
            {0, 0, 1, 1, 64, 64, 0, 1, 0, 1, 0, 1, 0, 64, 0, 0, 32, 32, 64};
        VkcBind rb[7] = {B(ids, 0), B(tab, 0), B(hid, 1), B(y, 1), B(w, 1), B(x, 1), B(rw, 1)};
        ok = record(KC.rows, rb, 7, &rp, sizeof rp, 1, 1, 1); n += ok;
        uint64_t ta = coli_vk_block_table_addr(g_kd);
        uint32_t et[16] = {0};
        et[14] = (uint32_t)ta; et[15] = (uint32_t)(ta >> 32);
        ok = ok && vkc_zero(w, 0, 4) && vkc_write(tab, 0, et, sizeof et);
        struct { int I, O; float limit; int ibase, kgat, rpw; } gp = {0, 1, 0.f, 0, 0, 0};
        VkcBind gb[5] = {B(x, 0), B(w, 0), B(tab, 0), B(hid, 1), B(ids, 0)};
        for (int t = 0; t < 64 && ok; t++)
            for (int v = 0; v < 2 && ok; v++) {
                if (!((v ? pw->ex_gu : pw->ex_dn) >> t & 1)) continue;
                for (int c = 0; c < 3 && ok; c++) {   /* in floats, on the matrix units and at their wider item */
                    VkPipeline g = c < 2 ? grp_pipe(v, t, c) : grp_wide_rows() > 0 ? grpw_pipe(v, t) : VK_NULL_HANDLE;
                    if (g) { ok = record(g, gb, 5, &gp, sizeof gp, 1, 1, 1); n += ok; }
                }
            }
        /* our formats' grouped GEMMs: the same empty item (no rows, I 0: nothing read) */
        for (int v = 0; v < 2 && ok && pw->ex_ours; v++) {
            for (int c = 0; c < 2 && ok; c++) {
                VkPipeline g = grpq_pipe(v, c);
                if (g) { ok = record(g, gb, 5, &gp, sizeof gp, 1, 1, 1); n += ok; }
            }
        }
    }
    /* a prompt chunk's attention on the matrix units (chain_attn_flash), launched on one row */
    if (ok && KC.mod_fa && pw->attn_hd > 0 && pw->attn_hd <= 256 && pw->attn_hd % 64 == 0 && pw->attn_gq > 0 && 16 % pw->attn_gq == 0) {
        VkPipeline fp = fa_pipe(pw->attn_hd, pw->attn_gq);
        if (fp) fo = vkc_buf((size_t)16 * 256 * 4, VKC_DEV);
        if (fo) {
            int hd = pw->attn_hd, G = pw->attn_gq;
            VkcAttnW a; memset(&a, 0, sizeof a);
            a.a = (VkcAttn){.S = 1, .H = G, .KVH = 1, .hd = hd, .cap = 1, .q_row = G * hd, .q_seg = hd, .o_row = G * hd, .scale = 1.f};
            a.vd = hd;
            VkcBind ab[7] = {B(x, 0), B(x, 0), B(x, 0), B(fo, 1), B(NULL, 0), B(NULL, 0), B(NULL, 0)};
            KC.kind = PK_ATTN;
            ok = record(fp, ab, 7, &a, sizeof a, 1, 1, 1); n += ok;
        }
    }
    /* a decode's attention over its positions' blocks (chain_attn_dec), its join launched on
     * one row of one position */
    if (ok && pw->attn_hd > 0 && pw->attn_gq > 0) adg_pipe(pw->attn_hd, pw->attn_gq);   /* its grouped mode: made, not run */
    if (ok && pw->attn_hd > 0 && adec_pipe()) {
        struct { VkcAttnW w; int mode, L, nb; } pc;
        memset(&pc, 0, sizeof pc);
        pc.w.a = (VkcAttn){.S = 1, .H = 1, .KVH = 1, .hd = 64, .cap = 1, .q_row = 64, .q_seg = 64, .o_row = 64, .scale = 1.f};
        pc.w.vd = 64; pc.mode = 1; pc.L = 64; pc.nb = 1;
        VkcBind ab[8] = {B(x, 0), B(x, 0), B(x, 0), B(y, 1), B(NULL, 0), B(NULL, 0), B(NULL, 0), B(x, 0)};
        KC.kind = PK_ATTN;
        ok = record(KC.adec, ab, 8, &pc, sizeof pc, 1, 1, 1); n += ok;
    }
    /* a prompt chunk's K/V rows into the cache (vkc_copy_regions' grid), launched with no regions */
    if (ok && cp2_pipe()) {
        VkcCp2 g = {0};
        g.len = g.per = 1;
        VkcBind cb[2] = {B(x, 0), B(y, 1)};
        KC.kind = PK_COPY;
        ok = record(CP2.pipe, cb, 2, &g, sizeof g, 1, 1, 1); n += ok;
    }
    ok = ok && vkc_submit(1);
    VkcBuf *all[] = {x, w, y, ids, rw, tab, hid, fo};
    for (size_t k = 0; k < sizeof all / sizeof *all; k++) if (all[k]) vkc_free(all[k]);
    if (!ok) return 0;
    if (n) fprintf(stderr, "[VK] chain: %d pipelines made before the expert tier, in %.1f s (COLI_VK_CHAIN_PREWARM=0: at first use)\n",
                   n, (vkc_now_ms() - t0) / 1e3);
    return n;
}
int vkc_moe_ready(void) { return vkc_ready() && moe_build(); }
int vkc_moe(VkcBuf *x, VkcBuf *ids, VkcBuf *w, VkcBuf *table, VkcBuf *hid, VkcBuf *y, const VkcMoe *p) {
    KC.kind = PK_MOE;
    int sn = p->stage_n ? p->stage_n : p->D > p->I ? p->D : p->I;
    /* the integer form where both of the layer's formats have it */
    /* a layer of known block types: the pipeline of those alone (every entry has them) */
    int tg = p->blocks && p->gfmt >= 100 && p->gfmt < 164 ? p->gfmt - 99 : 0, td = p->blocks && p->dfmt >= 100 && p->dfmt < 164 ? p->dfmt - 99 : 0;
    VkPipeline mp = p->blocks && moe_i8_type(p->gfmt) && moe_i8_type(p->dfmt) && p->D % 32 == 0 && p->I % 32 == 0
                    ? moe_i8_pipe(sn, tg, td) : VK_NULL_HANDLE;
    if (mp) KC.st.moe_i8++;
    else mp = moe_pipe_f(p->blocks != 0, sn, 0, tg, td);
    /* our formats: each pass on the pipeline of its matrices' format (chain_moe.comp's FMTC),
     * where the layer's is known and one of 1, 2, 4 */
    VkPipeline mg = mp, md = mp;
    if (!p->blocks) {
        VkPipeline g = moe_fi(p->gfmt) ? moe_pipe_f(0, sn, p->gfmt, 0, 0) : VK_NULL_HANDLE;
        VkPipeline d = moe_fi(p->dfmt) ? moe_pipe_f(0, sn, p->dfmt, 0, 0) : VK_NULL_HANDLE;
        if (g) mg = g;
        if (d) md = d;
    }
    if (!mp || p->S < 1 || p->K < 1 || p->K > VKC_ROUTE_KMAX || p->D < 1 || p->I < 1 || p->act < 0 || p->act > 1 ||
        (p->act == 1 && !(p->a > 0.0f && p->b > 0.0f)) || p->t_off < 0 || !x || !ids || !w || !table || !hid || !y ||
        hid == x || hid == y || y == x) return 0;
    if ((size_t)p->S * p->K * ((size_t)p->I + p->D) * sizeof(float) > hid->bytes) return 0;
    struct { int mode, S, K, D, I, act; float limit, a, b;
             int t_off, x_off, x_row, i_off, i_row, w_off, w_row, y_off, y_row, rpw, spl; } pc =
        {0, p->S, p->K, p->D, p->I, p->act, p->limit, p->a, p->b,
         p->t_off, p->x_off, p->x_row, p->i_off, p->i_row, p->w_off, p->w_row, p->y_off, p->y_row,
         p->rpw > 0 && p->rpw <= 256 ? p->rpw : 64, 8};
    /* The block formats' rows a workgroup and 32-weight steps a lane, pass by pass (an RTX
     * PRO 4000, 256 experts, a route of their own each layer; Qwen3.6's Q4_K/Q5_K at D 2048,
     * I 512, K 8 and Qwen3.8's IQ2_S/Q2_0 at D 2560, I 640, K 10):
     *   gate and up of the K-quants and Q8_0: 64 rows at 16 steps (Q4_K one row 38.8 ->
     *   37.4 us against 64 at 8); of the grid types and Q2_0, 32 rows at 8 steps while 64
     *   would leave fewer than 128 workgroups (IQ2_S one row 84.8 -> 74.7 us), 64 at 16
     *   past that (three rows 323.5 -> 305.0 us a layer);
     *   down: 128 rows at 8 steps, every lane of the workgroup on a row (Q5_K 36.5 -> 25.3
     *   us, Q2_0 35.2 -> 31.5).
     * Qwen3.6's layer, one row: 82.8 -> 70 us. A rpw of the caller's keeps both passes.
     * The choice by the layer's shape, not the step's rows: rows and steps a workgroup set a
     * row's lanes (the shader's lpr) and so its sums' order, and a speculative verify's rows
     * must get their decode steps' bits (by the rows, a verify of 2 took the wide passes and
     * a decode step the narrow ones: speculative decoding's output then followed the drafts'
     * count, and the three-row gain above is given up for that). */
    int rg = pc.rpw, sg = 8, rd = pc.rpw, sd = 8;
    if (p->blocks && p->rpw <= 0 && p->gfmt >= 100) {
        int t = p->gfmt - 100, kq = t == 8 || t == 10 || t == 11 || t == 12 || t == 13 || t == 14;
        int wide = kq || (long long)p->K * ((p->I + 63) / 64) >= 128;
        rg = wide ? 64 : 32; sg = wide ? 16 : 8; rd = 128; sd = 8;
        /* Q6_K's down at 4 steps: its step's decode wants more lanes a row (an RTX PRO 4000,
         * D 2048 I 512 K 8: 27.4 -> 20.9 us; D 2560 I 640 K 10 34.1 -> 32.3; the other types
         * the same or slower) */
        if (p->dfmt == 114) sd = 4;
    }
    /* gate and up, then down, each (pair, rpw outputs) a workgroup; then the rank-order sum.
     * Binding 6: the block formats' table (a GGUF's experts), a dummy without one. */
    VkBuffer tabbuf = p->blocks ? (VkBuffer)coli_vk_block_table_dev(g_kd) : VK_NULL_HANDLE;   /* made with the first block tensor */
    VkcBind bt = tabbuf ? (VkcBind){tabbuf, 0, 0, 0} : B(NULL, 0);
    VkcBind bd[7] = {B(x, 0), B(ids, 0), B(w, 0), B(table, 0), B(hid, 1), B(NULL, 0), bt};
    uint32_t gx, gy;
    /* our formats with whole 16-byte rows: chain_moe_gv.comp's passes (a dense GEMV's layout) */
    VkPipeline gvg = !p->blocks ? moe_gv_pipe(p->gfmt, p->D) : VK_NULL_HANDLE;
    VkPipeline gvd = gvg && p->I % 32 == 0 ? moe_gv_pipe(p->dfmt, p->I) : VK_NULL_HANDLE;
    if (gvg && gvd) {
        /* lanes a row: about eight 16-byte steps each (Qwen3.6's 2048 inputs 8, its 512 2:
         * an RTX PRO 4000's layer 47-50 us, against 51 at 4 and 4, 53 at 32) */
        int l0 = coli_vk_gv_lanes(p->gfmt, p->D), l1 = coli_vk_gv_lanes(p->dfmt, p->I);
        /* down's rows into hid past the S * K hidden rows (binding 5) */
        struct { int mode, S, K, D, I, act; float limit, a, b; int t_off, x_off, x_row, i_off, i_row, lpr, tiles, y_off, y_row; } gp =
            {0, p->S, p->K, p->D, p->I, p->act, p->limit, p->a, p->b, p->t_off, p->x_off, p->x_row, p->i_off, p->i_row, l0, 1,
             p->S * p->K * p->I, p->D};
        VkcBind bg[7] = {B(x, 0), B(ids, 0), B(w, 0), B(table, 0), B(hid, 1), B(hid, 1), bt};
        int upw = 256 / l0, nwg = (p->K * p->I + upw - 1) / upw;   /* units a workgroup, at subgroups of 64 */
        if (nwg > 1024) nwg = 1024;
        if (!record(gvg, bg, 7, &gp, sizeof gp, (uint32_t)nwg, (uint32_t)p->S, 1)) return 0;
        int rpp = 256 / l1;   /* a down tile: the rows one pass of the workgroup takes */
        gp.mode = 1; gp.lpr = l1; gp.tiles = (p->D + rpp - 1) / rpp;
        grid((uint64_t)p->S * p->K * gp.tiles, &gx, &gy);
        if (!record(gvd, bg, 7, &gp, sizeof gp, gx, gy, 1)) return 0;
        KC.st.moe_gv++;
    } else {
    pc.rpw = rg; pc.spl = sg;
    grid((uint64_t)p->S * p->K * ((p->I + rg - 1) / rg), &gx, &gy);
    if (!record(mg, bd, 7, &pc, sizeof pc, gx, gy, 1)) return 0;
    pc.mode = 1; pc.rpw = rd; pc.spl = sd;
    grid((uint64_t)p->S * p->K * ((p->D + rd - 1) / rd), &gx, &gy);
    if (!record(md, bd, 7, &pc, sizeof pc, gx, gy, 1)) return 0;
    }
    pc.mode = 2;
    VkcBind bd2[7] = {B(NULL, 0), B(ids, 0), B(w, 0), B(table, 0), B(hid, 0), B(y, 1), bt};
    grid((uint64_t)p->S * ((p->D + 255) / 256), &gx, &gy);
    return record(mg, bd2, 7, &pc, sizeof pc, gx, gy, 1);
}
/* ---- a prompt chunk's routed experts (chain_moe_rows.comp, qmatmul_grp*_bd.comp) ------ */
/* the ggml types bdstep.glsl decodes */
static int bd_type(int t) { return t == 8 || t == 10 || t == 11 || t == 12 || t == 13 || t == 14 || t == 16 || t == 17 || t == 18 ||
                                                    t == 19 || t == 20 || t == 21 || t == 22 || t == 23 || t == 29 || t == 42; }
/* the grouped GEMM's pipeline for pass v (1 gate+up, 0 down) and ggml type t: on the matrix
 * units (coop, qmatmul_grp*_bdc.comp, where the device has cooperative matrices at subgroups
 * of 64 or 32, unless COLI_VK_GRP_COOP=0) or in floats (qmatmul_grp*_bd.comp) */
static VkPipeline grp_pipe(int v, int t, int coop) {
    static const char *file[2] = {"qmatmul_grp_bd.spv", "qmatmul_grp_gate_up_bd.spv"};
    static const char *cfile[2] = {"qmatmul_grp_bdc.spv", "qmatmul_grp_gate_up_bdc.spv"};
    if (t < 0 || t >= 64 || !bd_type(t)) return VK_NULL_HANDLE;
    int32_t c = t;
    VkSpecializationMapEntry me = {0, 0, 4};
    VkSpecializationInfo si = {1, &me, 4, &c};
    if (coop) {
        if (!KC.grpc_tried[v][t]) {
            KC.grpc_tried[v][t] = 1;
            if (coop_on("COLI_VK_GRP_COOP")) {
                if (!KC.mod_grpc[v]) KC.mod_grpc[v] = load_module(KC.core.spv_path, cfile[v]);
                KC.grpc[v][t] = coop_pipe(KC.mod_grpc[v], &si);
            }
        }
        return KC.grpc[v][t];
    }
    if (!KC.grp_tried[v][t]) {
        KC.grp_tried[v][t] = 1;
        if (!KC.mod_grp[v]) KC.mod_grp[v] = load_module(KC.core.spv_path, file[v]);
        if (KC.mod_grp[v]) KC.grp[v][t] = make_pipe(KC.mod_grp[v], &si);
    }
    return KC.grp[v][t];
}
/* the grouped GEMM of our own formats ([0] down, [1] gate+up): one pipeline for 1, 2 and 4,
 * the format the expert table's; on the matrix units (coop, qmatmul_grp_qc.comp, as
 * grp_pipe's) or in floats (qmatmul_grp_q.comp) */
static VkPipeline grpq_pipe(int v, int coop) {
    static const char *file[2] = {"qmatmul_grp_q.spv", "qmatmul_grp_gate_up_q.spv"};
    static const char *cfile[2] = {"qmatmul_grp_qc.spv", "qmatmul_grp_gate_up_qc.spv"};
    if (coop) {
        if (!KC.grpqc_tried[v]) {
            KC.grpqc_tried[v] = 1;
            if (coop_on("COLI_VK_GRP_COOP") && (KC.mod_grpqc[v] = load_module(KC.core.spv_path, cfile[v])))
                KC.grpqc[v] = coop_pipe(KC.mod_grpqc[v], NULL);
        }
        return KC.grpqc[v];
    }
    if (!KC.grpq_tried[v]) {
        KC.grpq_tried[v] = 1;
        if ((KC.mod_grpq[v] = load_module(KC.core.spv_path, file[v]))) KC.grpq[v] = make_pipe(KC.mod_grpq[v], NULL);
    }
    return KC.grpq[v];
}
static int our_fmt(int f) { return f == 1 || f == 2 || f == 4; }
/* pass v's (1 gate+up, 0 down) pipeline for a matrix of format f: a GGUF's block type, or ours */
static VkPipeline rows_pipe1(int v, int f, int coop) {
    return f >= 100 ? grp_pipe(v, f - 100, coop) : our_fmt(f) ? grpq_pipe(v, coop) : VK_NULL_HANDLE;
}
/* coop asks for the matrix units, which a format or a device without them answers in floats;
 * without the floats' pipeline (too little shared memory for it), the matrix units' */
static VkPipeline rows_pipe(int v, int f, int coop) {
    VkPipeline p = coop ? rows_pipe1(v, f, 1) : VK_NULL_HANDLE;
    if (!p) p = rows_pipe1(v, f, 0);
    return p ? p : rows_pipe1(v, f, 1);
}
static int rows_coop(int v, int f, VkPipeline p) {
    return p && (f >= 100 && f < 164 ? p == KC.grpc[v][f - 100] : our_fmt(f) && p == KC.grpqc[v]);
}
/* the grouped GEMM's wider item for pass v and ggml type t (qmatmul_grp*_bdcw.comp, 64 rows
 * by 128 decoded rows): where the matrix units take the type, made on first use */
static VkPipeline grpw_pipe(int v, int t) {
    static const char *file[2] = {"qmatmul_grp_bdcw.spv", "qmatmul_grp_gate_up_bdcw.spv"};
    if (t < 0 || t >= 64 || !grp_pipe(v, t, 1)) return VK_NULL_HANDLE;
    if (!KC.grpw_tried[v][t]) {
        KC.grpw_tried[v][t] = 1;
        if (kc_shared_limit() >= GRP_BDCW_SHARED) {
            if (!KC.mod_grpw[v]) KC.mod_grpw[v] = load_module(KC.core.spv_path, file[v]);
            int32_t c[2] = {t, KC.core.coop_sg};
            VkSpecializationMapEntry me[2] = {{0, 0, 4}, {1, 4, 4}};
            VkSpecializationInfo si = {2, me, sizeof c, c};
            if (KC.mod_grpw[v]) KC.grpw[v][t] = coop_pipe(KC.mod_grpw[v], &si);
        }
    }
    return KC.grpw[v][t];
}
/* the experts' rows a chunk's pairs give each on average from which the wider item takes both
 * passes: COLI_VK_GRP_WIDE (0 never) */
static int grp_wide_rows(void) {
    static int v = -1;
    if (v < 0) { const char *e = getenv("COLI_VK_GRP_WIDE"); v = e && *e ? atoi(e) : 32; if (v < 0) v = 0; }
    return v;
}
/* One dispatch whose group counts the device wrote (args at byte off of a buffer an earlier
 * dispatch of the frame wrote): the bindings' hazards as record's, and a barrier the
 * indirect read waits on. */
static int record_indirect(VkPipeline pipe, const VkcBind *bd, int n, const void *pc, size_t pcb, VkcBuf *args, size_t off) {
    VkcFrame *f = open_frame();
    if (!f || KC.lost) return 0;
    VkMemoryBarrier mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT};
    vkCmdPipelineBarrier(f->cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    KC.st.barriers++;
    /* record's own work with an empty grid, then the indirect dispatch in its place */
    VkDescriptorSet set = alloc_set(f);
    if (!set) { fprintf(stderr, "[VK] chain: out of descriptor sets\n"); return 0; }
    VkBuffer rd[VKC_BIND], wr[VKC_BIND]; int nr = 0, nw = 0;
    VkDescriptorBufferInfo bi[VKC_BIND];
    VkWriteDescriptorSet w[VKC_BIND];
    for (int i = 0; i < VKC_BIND; i++) {
        if (i < n && bd[i].buf) {
            bi[i] = (VkDescriptorBufferInfo){bd[i].buf, bd[i].off, bd[i].range ? bd[i].range : VK_WHOLE_SIZE};
            if (bd[i].written) wr[nw++] = bd[i].buf; else rd[nr++] = bd[i].buf;
        } else bi[i] = (VkDescriptorBufferInfo){KC.dummy->buf, 0, VK_WHOLE_SIZE};
        w[i] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set,
            .dstBinding = (uint32_t)i, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &bi[i]};
    }
    rd[nr++] = args->buf;
    vkUpdateDescriptorSets(KC.dev, VKC_BIND, w, 0, NULL);
    hazard(f, rd, nr, wr, nw);
    if (KC.bound != pipe) { vkCmdBindPipeline(f->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe); KC.bound = pipe; }
    vkCmdBindDescriptorSets(f->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, KC.pl, 0, 1, &set, 0, NULL);
    if (pcb) vkCmdPushConstants(f->cmd, KC.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, (uint32_t)pcb, pc);
    vkCmdDispatchIndirect(f->cmd, args->buf, off);
    args->last_frame = KC.serial;
    KC.st.ops++;
    prof_mark(f, KC.kind);
    return 1;
}
/* chain_moe_rows.comp's pipeline, made on first use (a device with bufferDeviceAddress, the
 * block formats' table and the grouped GEMMs' shared memory); COLI_VK_CHAIN_GEMM_BD=0 keeps a
 * prompt's experts on the host's step as its dense rows */
#define ROWS_GRP_SHARED 35648u   /* qmatmul_grp_gate_up_bd.comp's tiles and tables (backend_vulkan.c's GRP_BD_SHARED) */
static int rows_make(void) {
    if (!vkc_ready() || !KC.core.bda) return 0;
    if (!KC.rows_tried) {
        KC.rows_tried = 1;
        const char *g = getenv("COLI_VK_CHAIN_GEMM_BD");
        VkPhysicalDeviceProperties pp; vkGetPhysicalDeviceProperties((VkPhysicalDevice)KC.core.phys, &pp);
        if (!(g && *g == '0') && pp.limits.maxComputeSharedMemorySize >= ROWS_GRP_SHARED &&
            (KC.mod_rows = load_module(KC.core.spv_path, "chain_moe_rows.spv"))) KC.rows = make_pipe(KC.mod_rows, NULL);
    }
    return KC.rows != VK_NULL_HANDLE;
}
int vkc_moe_rows_ready(int gfmt, int dfmt) {
    /* a GGUF's block types (their decoders read the block table), or our formats (both matrices) */
    int bd = gfmt >= 100 && dfmt >= 100 && coli_vk_block_table_addr(g_kd), q = our_fmt(gfmt) && our_fmt(dfmt);
    return (bd || q) && rows_make() && rows_pipe(1, gfmt, 0) && rows_pipe(0, dfmt, 0);
}
void vkc_moe_rows_free(VkcMoeRows *r) {
    VkcBuf **b[7] = {&r->cnt, &r->amap, &r->items, &r->etab, &r->aux, &r->hid, &r->yk};
    for (int k = 0; k < 7; k++) if (*b[k]) { vkc_free(*b[k]); *b[k] = NULL; }
}
int vkc_moe_rows(VkcBuf *x, VkcBuf *ids, VkcBuf *w, VkcBuf *table, VkcMoeRows *r, VkcBuf *y, const VkcMoe *p) {
    KC.kind = PK_MOE;
    int E = p->E;
    if (p->S < 1 || p->K < 1 || p->K > VKC_ROUTE_KMAX || E < 1 || E > VKC_ROUTE_EMAX || p->act != 0 || p->D % 64 ||
        p->I % 64 || p->t_off < 0 || p->x_off || !x || !ids || !w || !table || !y || !r || !vkc_moe_rows_ready(p->gfmt, p->dfmt)) return 0;
    size_t pairs = (size_t)p->S * p->K;
    /* the wider item when the experts' rows fill it, both passes on the matrix units */
    int wr = grp_wide_rows();
    int wide = wr > 0 && !p->floats && p->gfmt >= 100 && p->dfmt >= 100 && pairs >= (size_t)wr * E &&
               grpw_pipe(1, p->gfmt - 100) && grpw_pipe(0, p->dfmt - 100);
    int tm = wide ? 64 : 32, bo1 = wide ? 64 : 32, bo0 = wide ? 128 : 64;
    int nob1 = (p->I + bo1 - 1) / bo1, nob0 = (p->D + bo0 - 1) / bo0;
    /* a pass's items at most: every expert's partial tile and the pairs' whole ones */
    size_t imax = ((pairs + tm - 1) / tm + (size_t)E) * (size_t)(nob1 > nob0 ? nob1 : nob0);
    if (imax > 65535 * 16) return 0;
    uint64_t ta = coli_vk_block_table_addr(g_kd);
    if (!vkc_reserve(&r->cnt, (size_t)3 * E * 4, VKC_DEV) || !vkc_reserve(&r->amap, pairs * 4, VKC_DEV) ||
        !vkc_reserve(&r->items, 2 * imax * 16, VKC_DEV) || !vkc_reserve(&r->etab, (size_t)32 * E * 4, VKC_DEV) ||
        !vkc_reserve(&r->aux, 32, VKC_DEV) || !vkc_reserve(&r->hid, pairs * p->I * 4, VKC_DEV) ||
        !vkc_reserve(&r->yk, pairs * p->D * 4, VKC_DEV)) return 0;
    struct { int mode, S, K, E, D, I, t_off, imax, i_off, i_row, w_off, w_row, y_off, y_row; uint32_t tab_lo, tab_hi; int tm, bo1, bo0; } pc =
        {0, p->S, p->K, E, p->D, p->I, p->t_off, (int)imax, p->i_off, p->i_row, p->w_off, p->w_row, p->y_off, p->y_row,
         (uint32_t)ta, (uint32_t)(ta >> 32), tm, bo1, bo0};
    uint32_t gp = (uint32_t)((pairs + 255) / 256);
    /* count, plan, scatter: their tables in r */
    VkcBind sb[7] = {B(ids, 0), B(table, 0), B(r->cnt, 1), B(r->amap, 1), B(r->items, 1), B(r->etab, 1), B(r->aux, 1)};
    int ok = vkc_zero(r->cnt, 0, (size_t)E) && record(KC.rows, sb, 7, &pc, sizeof pc, gp, 1, 1);
    pc.mode = 1; ok = ok && record(KC.rows, sb, 7, &pc, sizeof pc, 1, 1, 1);
    pc.mode = 2; ok = ok && record(KC.rows, sb, 7, &pc, sizeof pc, gp, 1, 1);
    /* gate+up: the tokens' rows gathered by amap into the packed hidden rows */
    struct { int I, O; float limit; int ibase, kgat, rpw; } pg = {p->D, p->I, p->limit, 0, p->K, 0};
    size_t eb0 = (size_t)16 * E * 4;   /* the down pass's table */
    VkcBind gb[5] = {B(x, 0), B(r->items, 0), {r->etab->buf, 0, eb0, 0}, B(r->hid, 1), B(r->amap, 0)};
    r->etab->last_frame = KC.serial;
    /* on the matrix units unless the caller keeps the floats (p->floats: the tier's host pool,
     * whose experts the GEMMs read over PCIe; there the floats' were faster) */
    VkPipeline pgu = wide ? grpw_pipe(1, p->gfmt - 100) : rows_pipe(1, p->gfmt, !p->floats),
               pdn = wide ? grpw_pipe(0, p->dfmt - 100) : rows_pipe(0, p->dfmt, !p->floats);
    ok = ok && record_indirect(pgu, gb, 5, &pg, sizeof pg, r->aux, 0);
    /* down: the packed hidden rows into the pairs' rows (amap) */
    struct { int I, O; float limit; int ibase, kgat, rpw; } pd = {p->I, p->D, 0.f, (int)imax, p->K, 0};
    VkcBind db[5] = {B(r->hid, 0), B(r->items, 0), {r->etab->buf, eb0, eb0, 0}, B(r->yk, 1), B(r->amap, 0)};
    ok = ok && record_indirect(pdn, db, 5, &pd, sizeof pd, r->aux, 16);
    if (ok) KC.st.grp_coop += wide ? 2 : rows_coop(1, p->gfmt, pgu) + rows_coop(0, p->dfmt, pdn);
    if (ok && wide) KC.st.grp_wide += 2;
    /* the rows' sums over their experts, in rank order */
    pc.mode = 3;
    VkcBind yb[7] = {B(ids, 0), B(table, 0), B(NULL, 0), B(NULL, 0), B(w, 0), B(r->yk, 0), B(y, 1)};
    ok = ok && record(KC.rows, yb, 7, &pc, sizeof pc, (uint32_t)p->S * (uint32_t)((p->D + 255) / 256), 1, 1);
    return ok;
}
int vkc_ares_ready(void) { return vkc_ready() && KC.mpipe[PM_ARES]; }
int vkc_ares_mix(VkcBuf *x, VkcBuf *blk, VkcBuf *prm, VkcBuf *y, const VkcAres *p) {
    KC.kind = PK_ARES;
    if (!KC.mpipe[PM_ARES] || p->S < 0 || p->D < 1 || p->nb < 0 || p->nb > 15 || (p->nb > 0 && !blk) || y == x || y == blk) return 0;
    if (p->S == 0) return open_frame() && !KC.lost;
    struct { int mode; VkcAres b; } pc = {0, *p};
    VkcBind bd[4] = {B(x, 0), B(p->nb > 0 ? blk : NULL, 0), B(prm, 0), B(y, 1)};
    uint32_t gx, gy; grid((uint64_t)p->S, &gx, &gy);
    return record(KC.mpipe[PM_ARES], bd, 4, &pc, sizeof pc, gx, gy, 1);
}
int vkc_situ(VkcBuf *g, VkcBuf *u, VkcBuf *y, const VkcSitu *p) {
    KC.kind = PK_ARES;
    if (!KC.mpipe[PM_ARES] || p->n < 0) return 0;
    if (p->n == 0) return open_frame() && !KC.lost;
    int32_t pc[13] = {1, p->n, 0, 0, p->g_off, 0, p->u_off, 0, 0, p->y_off, 0, 0, 0};
    memcpy(&pc[11], &p->b1, 4); memcpy(&pc[12], &p->b2, 4);
    VkcBind bd[4] = {B(g, 0), B(u, 0), B(NULL, 0), B(y, 1)};
    uint32_t gx, gy; grid(((uint64_t)p->n + 255) / 256, &gx, &gy);
    return record(KC.mpipe[PM_ARES], bd, 4, pc, sizeof pc, gx, gy, 1);
}

int vkc_mla_scratch(VkcMlaScratch *s, const VkcMla *m, int rows) {
    if (s->rows >= rows && s->q) return 1;
    size_t r = (size_t)rows, H = (size_t)m->H;
    int ok = vkc_reserve(&s->qa, (m->q_lora > 0 ? r * m->q_lora : 1) * 4, VKC_DEV) &&
             vkc_reserve(&s->q, r * H * (m->Q + m->R) * 4, VKC_DEV) &&
             vkc_reserve(&s->kv, r * (m->K + m->R) * 4, VKC_DEV) &&
             vkc_reserve(&s->qabs, r * H * m->K * 4, VKC_DEV) &&
             vkc_reserve(&s->clat, r * H * m->K * 4, VKC_DEV) &&
             vkc_reserve(&s->ctx, r * H * m->V * 4, VKC_DEV);
    s->rows = ok ? rows : 0;
    return ok;
}
void vkc_mla_scratch_free(VkcMlaScratch *s) {
    vkc_free(s->qa); vkc_free(s->q); vkc_free(s->kv); vkc_free(s->qabs); vkc_free(s->clat); vkc_free(s->ctx);
    memset(s, 0, sizeof *s);
}
/* the tensor is there and has the shape [O x I] */
static int mla_shape(ColiVkTensor *t, int O, int I) {
    ColiVkTensorInfo ti;
    return t && vkc_tensor_info(t, &ti) && ti.I == I && ti.O == O;
}
int vkc_mla_qkv(const VkcMla *m, VkcMlaScratch *s, VkcBuf *x, size_t x_off, int S, int pos_base,
                VkcBuf *cs, VkcMlaCache *c, VkcBuf *down, size_t down_off) {
    int H = m->H, QR = m->Q + m->R, KR = m->K + m->R;
    if (!KC.mla_ok || S < 1 || S > s->rows || pos_base < 0 || (int64_t)pos_base + S > c->cap || m->R < 0 ||
        (m->R & 1) || m->R > 128 || m->K < 1 || m->K > 1024 || (m->R > 0 && (!cs || !c->rope)) ||
        !mla_shape(m->q_b, H * QR, m->q_lora > 0 ? m->q_lora : m->D) || !mla_shape(m->kv_a, KR, m->D) ||
        (m->q_lora > 0 && !mla_shape(m->q_a, m->q_lora, m->D))) return 0;
    int ok;
    if (m->q_lora > 0) {
        VkcNorm qn = {S, m->q_lora, 1, 0, m->q_lora, m->q_lora, 0, m->q_lora, m->q_lora, (int)m->q_norm, 0, 0, m->eps, 1.f};
        ok = vkc_matmul(m->q_a, x, x_off, s->qa, 0, S) && vkc_norm(s->qa, m->prm, s->qa, &qn) &&
             vkc_matmul(m->q_b, s->qa, 0, s->q, 0, S);
    } else ok = vkc_matmul(m->q_b, x, x_off, s->q, 0, S);
    ok = ok && vkc_matmul(m->kv_a, x, x_off, s->kv, 0, S);
    VkcNorm ln = {S, m->K, 1, 0, KR, KR, pos_base * m->K, m->K, m->K, (int)m->kv_norm, 0, 0, m->eps, 1.f};
    ok = ok && vkc_norm(s->kv, m->prm, c->lat, &ln);
    if (ok && m->R > 0) {
        VkcMlaRow rq = {S * H, H, m->R, m->R, m->rope_style, m->Q, H * QR, QR, m->Q, H * QR, QR, 0, m->R, 0, 0, 0, 0.f};
        VkcMlaRow rk = {S, 1, m->R, m->R, m->rope_style, m->K, KR, 0, pos_base * m->R, m->R, 0, 0, m->R, 0, 0, 0, 0.f};
        ok = vkc_mla_rope(s->q, cs, s->q, &rq) && vkc_mla_rope(s->kv, cs, c->rope, &rk);
    }
    if (ok && m->kv16) {   /* the rows as the host's float16 cache holds them, before anything reads them */
        VkcEw rl = {VKC_EW_F16, S * m->K, m->K, 0, 0, 0, pos_base * m->K, pos_base * m->K, 0, 0, 0, 1.f};
        VkcEw rr = {VKC_EW_F16, S * m->R, m->R, 0, 0, 0, pos_base * m->R, pos_base * m->R, 0, 0, 0, 1.f};
        ok = vkc_ew(c->lat, c->lat, NULL, NULL, NULL, &rl) && (m->R == 0 || vkc_ew(c->rope, c->rope, NULL, NULL, NULL, &rr));
    }
    if (ok && down)
        ok = vkc_copy(down, down_off, c->lat, (size_t)pos_base * m->K, (size_t)S * m->K) &&
             (m->R == 0 || vkc_copy(down, down_off + (size_t)S * m->K, c->rope, (size_t)pos_base * m->R, (size_t)S * m->R));
    return ok;
}
int vkc_mla_attn(const VkcMla *m, VkcMlaScratch *s, int S, int pos_base, int kv_start, VkcMlaCache *c,
                 VkcBuf *sel, size_t sel_off, int sel_row, VkcBuf *gate, size_t gate_off, VkcBuf *out, size_t out_off) {
    int H = m->H, QR = m->Q + m->R, HK = H * m->K, HV = H * m->V;
    if (!KC.mla_ok || S < 1 || S > s->rows || (int64_t)pos_base + S > c->cap) return 0;
    if (m->kv_b ? !mla_shape(m->kv_b, H * (m->Q + m->V), m->K)
                : !(mla_shape(m->k_abs, HK, m->Q) && mla_shape(m->v_abs, HV, m->K))) return 0;
    if (m->o && out && !mla_shape(m->o, m->D, HV)) return 0;
    int ok;
    if (m->kv_b) {   /* qa[s][h][i] = sum_d kv_b[h*(Q+V) + d][i] q[s][h][d], d < Q */
        VkcHgemv a = {1, S, H, m->Q, m->Q + m->V, 0, 0, H * QR, QR, 0, HK, m->K, 0, 0, 0};
        ok = vkc_mla_hgemv(m->kv_b, s->q, s->qabs, NULL, &a);
    } else {         /* qa[s][h][i] = k_abs[h*K + i] . q[s][h][0..Q) */
        VkcHgemv a = {0, S, H, m->K, m->K, 0, 0, H * QR, QR, 0, HK, m->K, 0, 0, 0};
        ok = vkc_mla_hgemv(m->k_abs, s->q, s->qabs, NULL, &a);
    }
    VkcMlaCore cp = {S, H, m->K, m->R, pos_base, kv_start, 0, HK, m->K, m->Q, H * QR, QR, 0, m->K, 0, m->R,
                     (int)sel_off, sel ? sel_row : 0, 0, HK, m->K, m->scale};
    ok = ok && vkc_mla_core(s->qabs, s->q, c->lat, c->rope, sel, s->clat, &cp);
    VkcHgemv v = {0, S, H, m->V, m->kv_b ? m->Q + m->V : m->V, m->kv_b ? m->Q : 0, 0, HK, m->K, 0, HV, m->V,
                  (int)gate_off, HV, gate != NULL};
    ok = ok && vkc_mla_hgemv(m->kv_b ? m->kv_b : m->v_abs, s->clat, s->ctx, gate, &v);
    if (ok && m->o && out) ok = vkc_matmul(m->o, s->ctx, 0, out, out_off, S);
    return ok;
}
/* vkc_mla_attn for rows of different sequences: row s attends its own cache c[s] over
 * positions 0..pos[s] (or sel's list at sel_off + s*sel_row); the absorption, the values
 * and o_proj take every row at once, the core one row at a time. */
int vkc_mla_attn_rows(const VkcMla *m, VkcMlaScratch *s, int S, const int *pos, VkcMlaCache *const *c,
                      VkcBuf *sel, size_t sel_off, int sel_row, VkcBuf *out, size_t out_off) {
    int H = m->H, QR = m->Q + m->R, HK = H * m->K, HV = H * m->V;
    if (!KC.mla_ok || S < 1 || S > s->rows) return 0;
    for (int r = 0; r < S; r++) if (!c[r] || pos[r] < 0 || (int64_t)pos[r] + 1 > c[r]->cap) return 0;
    if (m->kv_b ? !mla_shape(m->kv_b, H * (m->Q + m->V), m->K)
                : !(mla_shape(m->k_abs, HK, m->Q) && mla_shape(m->v_abs, HV, m->K))) return 0;
    if (m->o && out && !mla_shape(m->o, m->D, HV)) return 0;
    int ok;
    if (m->kv_b) {
        VkcHgemv a = {1, S, H, m->Q, m->Q + m->V, 0, 0, H * QR, QR, 0, HK, m->K, 0, 0, 0};
        ok = vkc_mla_hgemv(m->kv_b, s->q, s->qabs, NULL, &a);
    } else {
        VkcHgemv a = {0, S, H, m->K, m->K, 0, 0, H * QR, QR, 0, HK, m->K, 0, 0, 0};
        ok = vkc_mla_hgemv(m->k_abs, s->q, s->qabs, NULL, &a);
    }
    for (int r = 0; ok && r < S; r++) {
        VkcMlaCore cp = {1, H, m->K, m->R, pos[r], 0, r * HK, HK, m->K, m->Q + r * H * QR, H * QR, QR, 0, m->K, 0, m->R,
                         (int)sel_off + (sel ? r * sel_row : 0), sel ? sel_row : 0, r * HK, HK, m->K, m->scale};
        ok = vkc_mla_core(s->qabs, s->q, c[r]->lat, c[r]->rope, sel, s->clat, &cp);
    }
    VkcHgemv v = {0, S, H, m->V, m->kv_b ? m->Q + m->V : m->V, m->kv_b ? m->Q : 0, 0, HK, m->K, 0, HV, m->V, 0, HV, 0};
    ok = ok && vkc_mla_hgemv(m->kv_b ? m->kv_b : m->v_abs, s->clat, s->ctx, NULL, &v);
    if (ok && m->o && out) ok = vkc_matmul(m->o, s->ctx, 0, out, out_off, S);
    return ok;
}
int vkc_mla(const VkcMla *m, VkcMlaScratch *s, VkcBuf *x, size_t x_off, int S, int pos_base, int kv_start,
            VkcBuf *cs, VkcMlaCache *c, VkcBuf *out, size_t out_off) {
    return vkc_mla_qkv(m, s, x, x_off, S, pos_base, cs, c, NULL, 0) &&
           vkc_mla_attn(m, s, S, pos_base, kv_start, c, NULL, 0, 0, NULL, 0, out, out_off);
}
/* ---- inkling's ops: chain_sconv.comp and chain_relattn.comp, made on first use ---- */
static struct { VkShaderModule mod[2]; VkPipeline pipe[2]; int tried[2]; } g_kx[2];
#define KX (g_kx[g_kd])
static VkPipeline kx_pipe(int k) {
    static const char *const file[2] = {"chain_sconv.spv", "chain_relattn.spv"};
    if (KX.pipe[k] || KX.tried[k] || !vkc_ready()) return KX.pipe[k];
    KX.tried[k] = 1;
    if ((KX.mod[k] = load_module(KC.core.spv_path, file[k]))) KX.pipe[k] = make_pipe(KX.mod[k], NULL);
    return KX.pipe[k];
}
static void kx_shutdown(void) {
    for (int k = 0; k < 2; k++) {
        if (KX.pipe[k]) vkDestroyPipeline(KC.dev, KX.pipe[k], NULL);
        if (KX.mod[k]) vkDestroyShaderModule(KC.dev, KX.mod[k], NULL);
    }
    memset(&KX, 0, sizeof KX);
}
int vkc_sconv_ready(void) { return kx_pipe(0) != VK_NULL_HANDLE; }
int vkc_relattn_ready(void) { return kx_pipe(1) != VK_NULL_HANDLE; }
int vkc_sconv(VkcBuf *x, VkcBuf *w, VkcBuf *ring, const VkcSconv *p) {
    VkPipeline pipe = kx_pipe(0);
    if (!pipe || (p->mode == 0 && (p->CK < 1 || p->CK > 9))) return 0;
    KC.kind = p->mode == 0 ? PK_DNCONV : PK_EW;
    uint32_t gx, gy;
    if (p->mode == 0) {
        VkcBind bd[3] = {B(x, 1), B(w, 0), B(ring, 1)};
        grid(((uint64_t)p->C + 63) / 64, &gx, &gy);
        return record(pipe, bd, 3, p, sizeof *p, gx, gy, 1);
    }
    VkcBind bd[1] = {B(x, 1)};
    grid(((uint64_t)p->n + 63) / 64, &gx, &gy);
    return record(pipe, bd, 1, p, sizeof *p, gx, gy, 1);
}
int vkc_relattn(VkcBuf *q, VkcBuf *kc, VkcBuf *vc, VkcBuf *o, VkcBuf *kvs, VkcBuf *r, VkcBuf *relp, VkcBuf *tau,
                const VkcRelAttn *p) {
    VkPipeline pipe = kx_pipe(1);
    if (!pipe || p->hd > 256 || p->d_rel > 64 || p->d_rel < 0 || p->KVH < 1 || p->H % p->KVH || p->cap < 1) return 0;
    KC.kind = PK_ATTN;
    VkcBind bd[8] = {B(q, 0), B(kc, 0), B(vc, 0), B(o, 1), B(kvs, 0), B(r, 0), B(relp, 0), B(tau, 0)};
    /* a long prompt chunk in slices of rows, a submission each (attn_slice_rows): the
     * shader's s0 is the slice's first row, pos_base stays the step's */
    int S = p->S, rr = attn_slice_rows(S, p->pos_base, (double)p->H * p->hd, 1), ok = 1;
    for (int r0 = 0; ok && r0 < S; r0 += rr) {
        struct { VkcRelAttn a; int s0; } x = {*p, r0};
        x.a.S = S - r0 < rr ? S - r0 : rr;
        x.a.q_off += r0 * p->q_row; x.a.o_off += r0 * p->o_row; x.a.r_off += r0 * p->r_row; x.a.tau_off += r0;
        if (r0 && !attn_slice_next()) return 0;
        ok = record(pipe, bd, 8, &x, sizeof x, (uint32_t)p->H, (uint32_t)x.a.S, 1);
    }
    return ok;
}

/* ---- the decision engines' encoders: chain_enc.comp, made on first use ---- */
static struct { VkShaderModule mod; VkPipeline pipe; int tried; } g_ke[2];
#define KE (g_ke[g_kd])
/* chain_attn_full.comp (a diffusion step's attention): made on first use, a pipeline per
 * head dim (its specialization constant) */
static struct { VkShaderModule mod; VkPipeline pipe[3]; int tried; VkShaderModule cmod; VkPipeline cpipe[2]; int ctried; } g_kf[2];
#define KF (g_kf[g_kd])
static void kf_shutdown(void) {
    for (int i = 0; i < 3; i++) if (KF.pipe[i]) vkDestroyPipeline(KC.dev, KF.pipe[i], NULL);
    for (int i = 0; i < 2; i++) if (KF.cpipe[i]) vkDestroyPipeline(KC.dev, KF.cpipe[i], NULL);
    if (KF.mod) vkDestroyShaderModule(KC.dev, KF.mod, NULL);
    if (KF.cmod) vkDestroyShaderModule(KC.dev, KF.cmod, NULL);
    memset(&KF, 0, sizeof KF);
}
/* chain_attn_full.comp's shared arrays have a fixed size: a device with less shared
 * memory declines the op (the CPU runs it) instead of getting an invalid pipeline, as
 * chain_gemv2 did on a 48 KiB GPU (#1998). tests/test_vk_shared_memory.py holds this
 * number to the compiled shader's. */
#define ATTN_FULL_SHARED 38912u
static void kf_load(void) {
    if (KF.mod || KF.tried) return;
    KF.tried = 1;
    if (kc_shared_limit() < ATTN_FULL_SHARED) return;
    KF.mod = load_module(KC.core.spv_path, "chain_attn_full.spv");
}
int vkc_attn_full(VkcBuf *qkv, VkcBuf *o, const VkcAttnFull *p) {
    if (!vkc_ready() || p->S < 1 || p->T < 1 || p->H < 1 || (p->hd != 32 && p->hd != 64 && p->hd != 128) ||
        (int64_t)p->S > 65535LL * 64 || p->H > 65535) return 0;
    int k = p->hd == 32 ? 0 : p->hd == 64 ? 1 : 2;
    if (!KF.pipe[k]) {
        kf_load();
        if (!KF.mod) return 0;
        int hd = p->hd;
        VkSpecializationMapEntry me = {0, 0, sizeof(int)};
        VkSpecializationInfo si = {1, &me, sizeof(int), &hd};
        if (!(KF.pipe[k] = make_pipe(KF.mod, &si))) return 0;
    }
    KC.kind = PK_ATTN;
    VkcBind bd[2] = {B(qkv, 0), B(o, 1)};
    return record(KF.pipe[k], bd, 2, p, sizeof *p, (uint32_t)((p->S + 63) / 64), (uint32_t)p->H, 1);
}
/* The same on the matrix units (chain_attn_coop.comp): a device with cooperative matrices
 * at subgroups of 32 or 64, head dims 64 and 128; its pipeline requires that subgroup
 * size. 0: not here (nothing recorded). */
static VkPipeline kf_coop_pipe(int hd) {
    int k = hd == 128, sg = KC.core.coop_sg;
    if (KF.cpipe[k]) return KF.cpipe[k];
    if ((sg != 32 && sg != 64) || (hd != 64 && hd != 128)) return VK_NULL_HANDLE;
    if (!KF.cmod) {
        if (KF.ctried) return VK_NULL_HANDLE;
        KF.ctried = 1;
        if (!(KF.cmod = load_module(KC.core.spv_path, "chain_attn_coop.spv"))) return VK_NULL_HANDLE;
    }
    int32_t sv[2] = {hd, sg};
    VkSpecializationMapEntry me[2] = {{0, 0, 4}, {1, 4, 4}};
    VkSpecializationInfo si = {2, me, sizeof sv, sv};
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT rss = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO_EXT,
        .requiredSubgroupSize = (uint32_t)sg};
    VkComputePipelineCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .pNext = &rss,
                  .flags = VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT_EXT,
                  .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = KF.cmod, .pName = "main",
                  .pSpecializationInfo = &si}, .layout = KC.pl};
    if (vkCreateComputePipelines(KC.dev, VK_NULL_HANDLE, 1, &ci, NULL, &KF.cpipe[k]) != VK_SUCCESS) KF.cpipe[k] = VK_NULL_HANDLE;
    return KF.cpipe[k];
}
int vkc_attn_full_coop(VkcBuf *qkv, VkcBuf *o, const VkcAttnFull *p) {
    if (!vkc_ready() || p->S < 1 || p->T < 1 || p->H < 1 || (int64_t)p->S > 65535LL * 64 || p->H > 65535) return 0;
    VkPipeline pipe = kf_coop_pipe(p->hd);
    if (!pipe) return 0;
    KC.kind = PK_ATTN;
    VkcBind bd[2] = {B(qkv, 0), B(o, 1)};
    return record(pipe, bd, 2, p, sizeof *p, (uint32_t)((p->S + 63) / 64), (uint32_t)p->H, 1);
}
int vkc_attn_full_coop_ready(int hd) { return vkc_ready() && kf_coop_pipe(hd) != VK_NULL_HANDLE; }
int vkc_attn_full_ready(void) {
    if (!vkc_ready()) return 0;
    kf_load();
    return KF.mod != VK_NULL_HANDLE;
}

/* chain_vae.comp (Qwen-Image's VAE: a band's 3x3 taps, the DupUp3D shortcut): made on first use */
static struct { VkShaderModule mod; VkPipeline pipe; int tried; } g_kva[2];
#define KVA (g_kva[g_kd])
static void kva_shutdown(void) {
    if (KVA.pipe) vkDestroyPipeline(KC.dev, KVA.pipe, NULL);
    if (KVA.mod) vkDestroyShaderModule(KC.dev, KVA.mod, NULL);
    memset(&KVA, 0, sizeof KVA);
}
int vkc_vae_ready(void) {
    if (!vkc_ready()) return 0;
    if (!KVA.pipe && !KVA.tried) {
        KVA.tried = 1;
        if ((KVA.mod = load_module(KC.core.spv_path, "chain_vae.spv"))) KVA.pipe = make_pipe(KVA.mod, NULL);
    }
    return KVA.pipe != VK_NULL_HANDLE;
}
int vkc_vae(VkcBuf *x, VkcBuf *o, VkcBuf *tab, const VkcVae *p) {
    if (p->n < 1 || !vkc_vae_ready()) return 0;
    KC.kind = PK_EW;
    VkcBind bd[3] = {B(x, 0), B(o, 1), B(p->mode == 1 ? tab : NULL, 0)};
    uint32_t gx, gy; grid(((uint64_t)p->n + 255) / 256, &gx, &gy);
    return record(KVA.pipe, bd, 3, p, sizeof *p, gx, gy, 1);
}

static VkPipeline ke_pipe(void) {
    if (KE.pipe || KE.tried || !vkc_ready()) return KE.pipe;
    KE.tried = 1;
    if ((KE.mod = load_module(KC.core.spv_path, "chain_enc.spv"))) KE.pipe = make_pipe(KE.mod, NULL);
    return KE.pipe;
}
static void ke_shutdown(void) {
    if (KE.pipe) vkDestroyPipeline(KC.dev, KE.pipe, NULL);
    if (KE.mod) vkDestroyShaderModule(KC.dev, KE.mod, NULL);
    memset(&KE, 0, sizeof KE);
}
int vkc_enc_ready(void) { return ke_pipe() != VK_NULL_HANDLE; }
static int ke_ew(int kind, int mode, void *pc, size_t bytes, const VkcBind *bd, int nb, uint64_t n) {
    VkPipeline pipe = ke_pipe();
    if (!pipe) return 0;
    *(int *)pc = mode;
    KC.kind = kind;
    uint32_t gx, gy; grid((n + 255) / 256, &gx, &gy);
    return record(pipe, bd, nb, pc, bytes, gx, gy, 1);
}
int vkc_enc_norm(VkcBuf *a, VkcBuf *b, VkcBuf *prm, VkcBuf *y, const VkcEncNorm *p) {
    VkPipeline pipe = ke_pipe();
    if (!pipe || p->D < 1 || p->D > 4096) return 0;
    VkcEncNorm c = *p;
    c.mode = 0;
    KC.kind = PK_NORM;
    VkcBind bd[4] = {B(a, (p->flags & VKC_ENC_SUM) != 0), B((p->flags & VKC_ENC_ADD) ? b : NULL, 0), B(prm, 0), B(y, 1)};
    uint32_t gx, gy; grid((uint64_t)p->rows, &gx, &gy);
    return record(pipe, bd, 4, &c, sizeof c, gx, gy, 1);
}
int vkc_enc_bias(VkcBuf *y, VkcBuf *prm, const VkcEncBias *p) {
    VkcEncBias c = *p;
    VkcBind bd[4] = {B(NULL, 0), B(NULL, 0), B(prm, 0), B(y, 1)};
    return ke_ew(PK_EW, 1, &c, sizeof c, bd, 4, (uint64_t)p->n);
}
int vkc_enc_geglu(VkcBuf *a, VkcBuf *prm, VkcBuf *y, const VkcEncGeglu *p) {
    VkcEncGeglu c = *p;
    VkcBind bd[4] = {B(a, 0), B(NULL, 0), B(prm, 0), B(y, 1)};
    return ke_ew(PK_EW, 2, &c, sizeof c, bd, 4, (uint64_t)p->n);
}
int vkc_enc_rope(VkcBuf *x, VkcBuf *tab, VkcBuf *pos, const VkcEncRope *p) {
    if (p->hd < 2 || p->hd % 2) return 0;
    VkcEncRope c = *p;
    VkcBind bd[5] = {B(x, 1), B(tab, 0), B(NULL, 0), B(NULL, 0), B(pos, 0)};
    return ke_ew(PK_ROPE, 3, &c, sizeof c, bd, 5, (uint64_t)p->n);
}
int vkc_enc_addrow(VkcBuf *y, VkcBuf *tab, VkcBuf *idx, const VkcEncAddRow *p) {
    VkcEncAddRow c = *p;
    VkcBind bd[5] = {B(NULL, 0), B(NULL, 0), B(tab, 0), B(y, 1), B(idx, 0)};
    return ke_ew(PK_EW, 4, &c, sizeof c, bd, 5, (uint64_t)p->n);
}
int vkc_enc_attn(VkcBuf *qkv, VkcBuf *c2p, VkcBuf *p2c, VkcBuf *o, VkcBuf *rng, VkcBuf *ridx, VkcBuf *pos,
                 const VkcEncAttn *p) {
    VkPipeline pipe = ke_pipe();
    if (!pipe || p->hd < 1 || p->hd > 128 || p->S < 1 || p->H < 1) return 0;
    VkcEncAttn c = *p;
    c.mode = 5;
    KC.kind = PK_ATTN;
    VkcBind bd[7] = {B(qkv, 0), B(c2p, 0), B(p2c, 0), B(o, 1), B(rng, 0), B(ridx, 0), B(pos, 0)};
    return record(pipe, bd, 7, &c, sizeof c, (uint32_t)p->H, (uint32_t)((p->S + 15) / 16), 1);
}
int vkc_enc_rel(VkcBuf *x, VkcBuf *pt, VkcBuf *y, const VkcEncRel *p) {
    VkPipeline pipe = ke_pipe();
    if (!pipe || p->hd < 1 || p->S < 1 || p->H < 1 || p->nr < 1) return 0;
    VkcEncRel c = *p;
    c.mode = 6;
    KC.kind = PK_ATTN;
    VkcBind bd[4] = {B(x, 0), B(pt, 0), B(NULL, 0), B(y, 1)};
    return record(pipe, bd, 4, &c, sizeof c, (uint32_t)((p->nr + 63) / 64), (uint32_t)((p->S + 15) / 16), (uint32_t)p->H);
}

void vkc_stats(VkcStats *st) { *st = KC.st; }

/* ---- big prefill chunks ------------------------------------------------------------
 * The rows of a prompt chunk (vk_chain.h, vkc_chunk_rows). The free device memory is
 * read once, at an engine's first call: a later reading would count the chain's own
 * scratch as used and shrink the chunk from one forward to the next. */
int vkc_chunk_auto(void) {
    const char *e = getenv("COLI_VK_CHAIN_ROWS");
    return !(e && *e) || !strcmp(e, "auto");
}
int vkc_chunk_rows(const char *engine, size_t row_bytes) {
    const char *e = getenv("COLI_VK_CHAIN_ROWS");
    if (e && *e && strcmp(e, "auto") != 0) {
        int v = atoi(e);
        return v < 1 ? 1 : v > 65535 ? 65535 : v;
    }
    static char said[16][24]; static int nsaid, rows_said[16], said_d[16];
    for (int i = 0; i < nsaid; i++) if (!strcmp(said[i], engine ? engine : "") && said_d[i] == g_kd) return rows_said[i];
    const char *mx = getenv("COLI_VK_CHAIN_ROWS_MAX");
    int cap = mx && *mx ? atoi(mx) : 8192;
    if (cap < 1) cap = 1;
    if (cap > 65535) cap = 65535;
    double used = 0, bud = 0, free_b;
    int have = coli_vk_mem_budget_dev(g_kd, &used, &bud);
    free_b = have ? (bud - used) * 1e9 : (double)coli_vk_device_local_bytes_dev(g_kd) / 4;
    /* Expert outputs and CPU fallback scratch also grow with the chunk on a
     * discrete device. Bound both memories, including Windows commit headroom. */
    double avail = compat_mem_available_gb() * 1e9;
    if (avail > 0 && avail < free_b) free_b = avail;
    /* the room the tier's budget left for a chunk's host pool experts (vk_dmoe.h), not
     * for rows: Qwen3.8's chunks took it and every prompt after the first read the pool */
    if (g_kd == 0) free_b -= (double)vkt_stage_room();
    if (free_b < 0) free_b = 0;
    double fit = row_bytes ? free_b / 2 / (double)row_bytes : cap;
    int rows = fit >= cap ? cap : (int)fit;
    if (rows < cap && rows >= 256) rows = rows / 256 * 256;
    if (rows < 1) rows = 1;
    if (nsaid < 16) {
        snprintf(said[nsaid], sizeof said[nsaid], "%s", engine ? engine : "");
        said_d[nsaid] = g_kd; rows_said[nsaid++] = rows;
        fprintf(stderr, "[VK] %s chain: prompt chunks of up to %d rows (%.0f KiB a row; %.2f GiB free on the device%s, "
                "half of it at most; COLI_VK_CHAIN_ROWS sets it, COLI_VK_CHAIN_ROWS_MAX caps it at %d)\n",
                engine ? engine : "engine", rows, row_bytes / 1024.0, free_b / 1073741824.0,
                avail > 0 ? " or in RAM" : "", cap);
    }
    return rows;
}
/* ---- a partial chain (vk_chain.h, vkc_fit) ------------------------------------------- */
static const char *fit_size(size_t b, char *buf, size_t n) {
    if (b >= ((size_t)1 << 30)) snprintf(buf, n, "%.2f GiB", (double)b / 1073741824.0);
    else snprintf(buf, n, "%.1f MiB", (double)b / 1048576.0);
    return buf;
}
static size_t fit_up(size_t b, size_t a) { return a > 1 ? (b + a - 1) / a * a : b; }
size_t vkc_fit_reserve(void) {
    const char *e = getenv("COLI_VK_TIER_RESERVE_GB");
    double g = e && *e ? atof(e) : 1.0;
    return g > 0 ? (size_t)(g * 1073741824.0) : 0;
}
/* A weight pool's block, a block of each of the chain's three pools, and the frames'
 * first staging buffers: what the pools may hold beyond the bytes asked for. */
size_t vkc_fit_pools(void) {
    return coli_vk_block_bytes_dev(g_kd, (size_t)256 << 20) + 3 * (size_t)coli_vk_block_bytes_dev(g_kd, VKC_BLOCK) +
           (size_t)VKC_FRAMES * VKC_STAGE0;
}
int vkc_fit_rows(int def) {
    const char *e = getenv("COLI_VK_CHAIN_ROWS");
    if (e && *e && strcmp(e, "auto") != 0) {
        int v = atoi(e);
        return v < 1 ? 1 : v > 65535 ? 65535 : v;
    }
    return def < 1 ? 1 : def;
}
size_t vkc_fit_tensor(int fmt, int I, int O, int gs) {
    size_t a = coli_vk_buffer_alignment_dev(g_kd);
    size_t rows = (coli_vk_tensor_row_bytes(fmt, I) + 3) / 4 * 4 * (size_t)O;
    size_t sc = coli_vk_tensor_scale_count(fmt, I, O, gs) * sizeof(float);
    return fit_up(rows ? rows : 4, a) + fit_up(sc ? sc : 4, a);
}
size_t vkc_fit_buf(size_t bytes) {
    size_t a = KC.align > 16 ? KC.align : 16, b = coli_vk_buffer_alignment_dev(g_kd);
    if (b > a) a = b;
    size_t v = bytes ? bytes : 4;
    return fit_up((v + 3) & ~(size_t)3, a);
}
int vkc_fit(const char *engine, int L, const size_t *layer_bytes, const size_t *matrix_bytes,
            size_t fixed_bytes, size_t tail_bytes, VkcFit *fit) {
    const char *eng = engine ? engine : "engine";
    memset(fit, 0, sizeof *fit);
    fit->L = L < 0 ? 0 : L;
    fit->per = calloc((size_t)(fit->L ? fit->L : 1), sizeof(size_t));
    fit->mat = calloc((size_t)(fit->L ? fit->L : 1), sizeof(size_t));
    fit->marks = calloc((size_t)(fit->L ? fit->L : 1), sizeof(unsigned long long));
    size_t all = 0;
    for (int i = 0; i < fit->L; i++) {
        size_t b = layer_bytes ? layer_bytes[i] : 0;
        if (fit->per) fit->per[i] = b;
        if (fit->mat && matrix_bytes) fit->mat[i] = matrix_bytes[i];
        all += b;
    }
    fit->free_b = coli_vk_free_bytes_dev(g_kd);
    fit->reserve = vkc_fit_reserve();
    fit->pools = vkc_fit_pools();
    fit->engine_fixed = fixed_bytes;
    fit->fixed = fixed_bytes + fit->pools;
    fit->tail_b = tail_bytes;
    size_t room = fit->free_b > fit->reserve ? fit->free_b - fit->reserve : 0;
    /* forced: COLI_VK_CHAIN_LAYERS on the primary, COLI_VK_CHAIN_LAYERS2 on the second device */
    const char *fvar = g_kd ? "COLI_VK_CHAIN_LAYERS2" : "COLI_VK_CHAIN_LAYERS", *e = getenv(fvar);
    char forced[48] = "";
    if (e && *e && strcmp(e, "auto") != 0) {
        int v = atoi(e);
        fit->n = v < 0 ? 0 : v > fit->L ? fit->L : v;
        fit->forced = 1;
        fit->tail = fit->n == fit->L && fit->L > 0;
        snprintf(forced, sizeof forced, "%s=%s", fvar, e);
    } else if (fit->fixed + all <= room) {
        fit->n = fit->L;
        fit->tail = fit->fixed + all + tail_bytes <= room;
    } else {
        size_t acc = fit->fixed;
        while (fit->n < fit->L && layer_bytes && acc + layer_bytes[fit->n] <= room) acc += layer_bytes[fit->n++];
    }
    for (int i = 0; i < fit->n; i++) fit->used_b += layer_bytes ? layer_bytes[i] : 0;
    /* the numbers, exactly (a test predicts N from them) */
    fprintf(stderr, "[VK] %s chain fit: free %zu B, reserve %zu B, fixed %zu B (the engine's %zu B, the pools' %zu B), tail %zu B, layers",
            eng, fit->free_b, fit->reserve, fit->fixed, fit->engine_fixed, fit->pools, fit->tail_b);
    for (int i = 0; i < fit->L; i++) fprintf(stderr, " %zu", layer_bytes ? layer_bytes[i] : (size_t)0);
    fprintf(stderr, " B");
    if (matrix_bytes) {
        fprintf(stderr, ", matrices");
        for (int i = 0; i < fit->L; i++) fprintf(stderr, " %zu", matrix_bytes[i]);
        fprintf(stderr, " B");
    }
    fprintf(stderr, "\n");
    char a[32], f[32], r[32], t[32], x[32], why[160];
    if (fit->forced) snprintf(why, sizeof why, "%s", forced);
    else snprintf(why, sizeof why, "free %s, reserve %s", fit_size(fit->free_b, f, sizeof f), fit_size(fit->reserve, r, sizeof r));
    if (fit->n == 0)
        fprintf(stderr, "[VK] %s chain: 0 of %d layers on the device: the chain stays off (%s%s%s%s%s)\n", eng, fit->L, why,
                fit->forced || !fit->L ? "" : "; layer 0 takes ", fit->forced || !fit->L ? "" : fit_size(fit->per ? fit->per[0] : 0, a, sizeof a),
                fit->forced || !fit->L ? "" : " beside ", fit->forced || !fit->L ? "" : fit_size(fit->fixed, x, sizeof x));
    else
        fprintf(stderr, "[VK] %s chain: %d of %d layers on the device (%s), %d on the CPU%s%s%s (%s)\n", eng, fit->n, fit->L,
                fit_size(fit->used_b, a, sizeof a), fit->L - fit->n,
                !fit->tail && tail_bytes ? ", the head and what goes with it (" : "",
                !fit->tail && tail_bytes ? fit_size(tail_bytes, t, sizeof t) : "", !fit->tail && tail_bytes ? ") on the CPU" : "", why);
    return fit->n;
}
void vkc_fit_shrink(const char *engine, VkcFit *fit, int n, const char *why) {
    if (!fit || n >= fit->n) return;
    if (n < 0) n = 0;
    const char *eng = engine ? engine : "engine";
    fit->n = n; fit->tail = 0; fit->used_b = 0;
    for (int i = 0; i < n && fit->per; i++) fit->used_b += fit->per[i];
    char a[32];
    if (n == 0)
        fprintf(stderr, "[VK] %s chain: 0 of %d layers on the device: the chain stays off (layer 0 did not reach the device: %s; "
                        "what it had placed was freed)\n", eng, fit->L, why ? why : "refused");
    else
        fprintf(stderr, "[VK] %s chain: %d of %d layers on the device (%s), %d on the CPU (layer %d did not reach the device: %s; "
                        "what it had placed was freed)\n", eng, n, fit->L, fit_size(fit->used_b, a, sizeof a), fit->L - n, n,
                why ? why : "refused");
}
void vkc_fit_mark(VkcFit *fit, int layer) {
    if (fit && fit->marks && layer >= 0 && layer < fit->L) fit->marks[layer] = coli_vk_fault_reached();
}
void vkc_fit_placed(const char *engine, const VkcFit *fit) {
    if (!fit) return;
    size_t w = 0, nt = 0, m = 0;
    coli_vk_mem_info_dev(g_kd, &w, &nt);
    /* matrices only: the block formats' decode table goes up with the first of them, beside them */
    size_t tab = coli_vk_block_table_bytes(g_kd);
    w = w > tab ? w - tab : 0;
    for (int i = 0; i < fit->n && fit->mat; i++) m += fit->mat[i];
    fprintf(stderr, "[VK] %s chain: %d of %d layers placed: %zu B of matrices on the device (the fit counted %zu B for these "
                    "layers), chain buffers %zu B, device memory held %zu B", engine ? engine : "engine", fit->n, fit->L, w, m,
            KC.ready ? KC.st.dev_bytes : (size_t)0, coli_vk_device_used_dev(g_kd));
    if (coli_vk_fault_set() && fit->marks) {
        fprintf(stderr, "; COLI_VK_STAGED_FAULT's point reached");
        for (int i = 0; i < fit->L; i++) fprintf(stderr, "%s%llu", i ? "," : " ", fit->marks[i]);
        fprintf(stderr, " times by the end of each layer");
    }
    fprintf(stderr, "\n");
}
int vkc_fit_partial(const VkcFit *fit) { return fit && fit->L > 0 && (fit->n < fit->L || !fit->tail); }
void vkc_layer_free(ColiVkTensor **const *t, int nt, VkcBuf **const *b, int nb) {
    if (vkc_ready()) vkc_finish();   /* no frame may still read them */
    for (int i = 0; i < nt; i++) if (t[i] && *t[i]) { coli_vk_tensor_free(*t[i]); *t[i] = NULL; }
    for (int i = 0; i < nb; i++) if (b[i] && *b[i]) { vkc_free(*b[i]); *b[i] = NULL; }
}

void vkc_prof_print(void) {
    if (!KC.prof) return;
    double tot = 0; for (int k = 0; k < PK_N; k++) tot += KC.prof_ms[k];
    fprintf(stderr, "[VK] chain profile: %.1f ms of device time", tot);
    for (int k = 0; k < PK_N; k++)
        if (KC.prof_n[k]) fprintf(stderr, " | %s %.1f ms (%llu)", pk_name[k], KC.prof_ms[k], KC.prof_n[k]);
    fprintf(stderr, "\n");
    /* COLI_VK_CHAIN_PROF=2: the matmuls by shape, the costliest first */
    for (int done = 0; done < KC.prof_shapes && done < 24; done++) {
        int b = -1;
        for (int k = 0; k < KC.prof_shapes; k++) if (KC.prof_sh_n[k] && (b < 0 || KC.prof_sh_ms[k] > KC.prof_sh_ms[b])) b = k;
        if (b < 0) break;
        uint64_t key = KC.prof_key[b];
        fprintf(stderr, "[VK] chain profile: matmul fmt %d S %d [%d x %d]: %.1f ms over %llu, %.1f us each\n",
                (int)(key >> 56), (int)((key >> 48) & 0xff), (int)((key >> 24) & 0xffffff), (int)(key & 0xffffff),
                KC.prof_sh_ms[b], KC.prof_sh_n[b], KC.prof_sh_ms[b] * 1e3 / (double)KC.prof_sh_n[b]);
        KC.prof_sh_n[b] = 0;   /* printed (the report runs once, at the end) */
    }
}

void vkc_shutdown(void) {
    if (!KC.ready) return;
    if (!KC.lost && coli_vk_available_dev(g_kd)) vkc_finish();
    else vkDeviceWaitIdle(KC.dev);
    for (int i = 0; i < VKC_FRAMES; i++) {
        VkcFrame *f = &KC.fr[i];
        frame_reclaim(f);
        if (f->stage) buf_release(f->stage);
        free(f->tmp);
        for (int k = 0; k < f->npools; k++) vkDestroyDescriptorPool(KC.dev, f->pools[k], NULL);
        free(f->pools);
        if (f->fence) vkDestroyFence(KC.dev, f->fence, NULL);
        if (f->qp) vkDestroyQueryPool(KC.dev, f->qp, NULL);
    }
    if (KC.tsem_g) p_vkDestroySemaphore(KC.dev, KC.tsem_g, NULL);
    if (KC.tsem_h) p_vkDestroySemaphore(KC.dev, KC.tsem_h, NULL);
    if (g_kd == 0) { g_hyb_dev = VK_NULL_HANDLE; g_hyb_g = g_hyb_h = VK_NULL_HANDLE; }
    KC.tsem_g = KC.tsem_h = VK_NULL_HANDLE;
    if (KC.dummy) buf_release(KC.dummy);
    if (KC.adec_x) buf_release(KC.adec_x);
    for (int k = 0; k < 3; k++) {
        VkaPool *P = &KC.pool[k].p;
        for (int b = 0; b < P->nb; b++) {
            if (!P->b[b].present) continue;
            VkcBlock *e = P->b[b].user;
            if (e) { if (e->map) vkUnmapMemory(KC.dev, e->mem); vkFreeMemory(KC.dev, e->mem, NULL); free(e); }
        }
        vka_pool_destroy(P);
    }
    if (KC.cpool) vkDestroyCommandPool(KC.dev, KC.cpool, NULL);
    kx_shutdown();
    kab_shutdown();
    cp2_shutdown();
    ke_shutdown();
    kf_shutdown();
    kva_shutdown();
    for (int i = 0; i < P_NPIPE; i++) {
        if (KC.pipe[i]) vkDestroyPipeline(KC.dev, KC.pipe[i], NULL);
        if (KC.mod[i]) vkDestroyShaderModule(KC.dev, KC.mod[i], NULL);
    }
    for (int i = 0; i < KC.nkdarec; i++) vkDestroyPipeline(KC.dev, KC.kdarec[i], NULL);
    for (int i = 0; i < KC.nkdar; i++) if (KC.kdar[i]) vkDestroyPipeline(KC.dev, KC.kdar[i], NULL);
    if (KC.mod_kdar) vkDestroyShaderModule(KC.dev, KC.mod_kdar, NULL);
    for (int i = 0; i < PM_N; i++) {
        if (KC.mpipe[i]) vkDestroyPipeline(KC.dev, KC.mpipe[i], NULL);
        if (KC.mmod[i]) vkDestroyShaderModule(KC.dev, KC.mmod[i], NULL);
    }
    for (int i = 0; i < KC.ngemm; i++) vkDestroyPipeline(KC.dev, KC.gemm[i], NULL);
    if (KC.tg) vkDestroyPipeline(KC.dev, KC.tg, NULL);
    if (KC.mod_tg) vkDestroyShaderModule(KC.dev, KC.mod_tg, NULL);
    for (int i = 0; i < KC.ndnrec; i++) vkDestroyPipeline(KC.dev, KC.dnrec[i], NULL);
    for (int i = 0; i < KC.ndnrr; i++) if (KC.dnrr[i]) vkDestroyPipeline(KC.dev, KC.dnrr[i], NULL);
    if (KC.mod_dnrr) vkDestroyShaderModule(KC.dev, KC.mod_dnrr, NULL);
    if (KC.gemv4) vkDestroyPipeline(KC.dev, KC.gemv4, NULL);
    if (KC.gemv4x) vkDestroyPipeline(KC.dev, KC.gemv4x, NULL);
    if (KC.gemv4n) vkDestroyPipeline(KC.dev, KC.gemv4n, NULL);
    if (KC.gemv4xn) vkDestroyPipeline(KC.dev, KC.gemv4xn, NULL);
    if (KC.adec) vkDestroyPipeline(KC.dev, KC.adec, NULL);
    if (KC.mdec) vkDestroyPipeline(KC.dev, KC.mdec, NULL);
    if (KC.mod_mdec) vkDestroyShaderModule(KC.dev, KC.mod_mdec, NULL);
    if (KC.mod_adec) vkDestroyShaderModule(KC.dev, KC.mod_adec, NULL);
    if (KC.mod_gemv4) vkDestroyShaderModule(KC.dev, KC.mod_gemv4, NULL);
    if (KC.gv2) vkDestroyPipeline(KC.dev, KC.gv2, NULL);
    if (KC.mod_gv2) vkDestroyShaderModule(KC.dev, KC.mod_gv2, NULL);
    if (KC.bd) vkDestroyPipeline(KC.dev, KC.bd, NULL);
    for (int r = 0; r < 4; r++) for (int t = 0; t < 64; t++) for (int b = 0; b <= KC_STAGES; b++)
        if (KC.bdv[r][t][b]) vkDestroyPipeline(KC.dev, KC.bdv[r][t][b], NULL);
    for (int t = 0; t < 64; t++) if (KC.bdg[t]) vkDestroyPipeline(KC.dev, KC.bdg[t], NULL);
    if (KC.mod_bdg) vkDestroyShaderModule(KC.dev, KC.mod_bdg, NULL);
    if (KC.fetch) vkDestroyPipeline(KC.dev, KC.fetch, NULL);
    if (KC.mod_fetch) vkDestroyShaderModule(KC.dev, KC.mod_fetch, NULL);
    for (int t = 0; t < 64; t++) if (KC.bdgc[t]) vkDestroyPipeline(KC.dev, KC.bdgc[t], NULL);
    if (KC.mod_bdgc) vkDestroyShaderModule(KC.dev, KC.mod_bdgc, NULL);
    for (int i = 0; i < 64; i++) if (KC.bdgw[i]) vkDestroyPipeline(KC.dev, KC.bdgw[i], NULL);
    if (KC.mod_bdgw) vkDestroyShaderModule(KC.dev, KC.mod_bdgw, NULL);
    for (int v = 0; v < 2; v++) {
        for (int t = 0; t < 64; t++) if (KC.grpc[v][t]) vkDestroyPipeline(KC.dev, KC.grpc[v][t], NULL);
        if (KC.mod_grpc[v]) vkDestroyShaderModule(KC.dev, KC.mod_grpc[v], NULL);
        for (int t = 0; t < 64; t++) if (KC.grpw[v][t]) vkDestroyPipeline(KC.dev, KC.grpw[v][t], NULL);
        if (KC.mod_grpw[v]) vkDestroyShaderModule(KC.dev, KC.mod_grpw[v], NULL);
    }
    if (KC.mod_bdv) vkDestroyShaderModule(KC.dev, KC.mod_bdv, NULL);
    if (KC.mod_bd) vkDestroyShaderModule(KC.dev, KC.mod_bd, NULL);
    for (int d = 0; d < 2; d++) for (int b = 0; b <= KC_STAGES; b++) if (KC.moe[d][b]) vkDestroyPipeline(KC.dev, KC.moe[d][b], NULL);
    for (int f = 0; f < 3; f++) for (int b = 0; b <= KC_STAGES; b++) if (KC.moe_f[f][b]) vkDestroyPipeline(KC.dev, KC.moe_f[f][b], NULL);
    for (int b = 0; b <= KC_STAGES; b++) if (KC.moe_i8[b]) vkDestroyPipeline(KC.dev, KC.moe_i8[b], NULL);
    for (int i = 0; i < KC.moe_nbt; i++) if (KC.moe_bt[i].p) vkDestroyPipeline(KC.dev, KC.moe_bt[i].p, NULL);
    for (int r = 0; r < 4; r++) for (int t = 0; t < 64; t++) for (int b = 0; b <= KC_STAGES; b++)
        if (KC.bdv8[r][t][b]) vkDestroyPipeline(KC.dev, KC.bdv8[r][t][b], NULL);
    if (KC.mod_bdv8) vkDestroyShaderModule(KC.dev, KC.mod_bdv8, NULL);
    if (KC.mod_moe_i8) vkDestroyShaderModule(KC.dev, KC.mod_moe_i8, NULL);
    if (KC.rows) vkDestroyPipeline(KC.dev, KC.rows, NULL);
    if (KC.mod_rows) vkDestroyShaderModule(KC.dev, KC.mod_rows, NULL);
    for (int v = 0; v < 2; v++) {
        for (int t = 0; t < 64; t++) if (KC.grp[v][t]) vkDestroyPipeline(KC.dev, KC.grp[v][t], NULL);
        if (KC.mod_grp[v]) vkDestroyShaderModule(KC.dev, KC.mod_grp[v], NULL);
        if (KC.grpq[v]) vkDestroyPipeline(KC.dev, KC.grpq[v], NULL);
        if (KC.mod_grpq[v]) vkDestroyShaderModule(KC.dev, KC.mod_grpq[v], NULL);
        if (KC.grpqc[v]) vkDestroyPipeline(KC.dev, KC.grpqc[v], NULL);
        if (KC.mod_grpqc[v]) vkDestroyShaderModule(KC.dev, KC.mod_grpqc[v], NULL);
    }
    if (KC.mod_moe) vkDestroyShaderModule(KC.dev, KC.mod_moe, NULL);
    for (int f = 0; f < 3; f++) for (int b = 0; b < 10; b++) if (KC.moe_gv[f][b]) vkDestroyPipeline(KC.dev, KC.moe_gv[f][b], NULL);
    for (int b = 0; b < 10; b++) for (int q = 0; q < 17; q++) if (KC.moe_gvq[b][q]) vkDestroyPipeline(KC.dev, KC.moe_gvq[b][q], NULL);
    if (KC.mod_moe_gv) vkDestroyShaderModule(KC.dev, KC.mod_moe_gv, NULL);
    if (KC.mod_gemm) vkDestroyShaderModule(KC.dev, KC.mod_gemm, NULL);
    if (KC.mod_dnrec) vkDestroyShaderModule(KC.dev, KC.mod_dnrec, NULL);
    for (int i = 0; i < KC.nfa; i++) vkDestroyPipeline(KC.dev, KC.fa[i], NULL);
    for (int i = 0; i < KC.nadg; i++) vkDestroyPipeline(KC.dev, KC.adg[i], NULL);
    KC.nadg = 0;
    if (KC.mod_fa) vkDestroyShaderModule(KC.dev, KC.mod_fa, NULL);
    if (KC.pl) vkDestroyPipelineLayout(KC.dev, KC.pl, NULL);
    if (KC.dsl) vkDestroyDescriptorSetLayout(KC.dev, KC.dsl, NULL);
    dsv4_shutdown();
    kvs_shutdown();
    memset(&KC, 0, sizeof KC);
    KC.cur = -1; KC.gemm_rows = -1;
}

/* ---- DeepSeek V4.1 Flash and DeepSeek V4 attention (chain_dsv4.comp) ------------------
 * Its own optional pipeline: without the shader only these ops decline. */
static struct { VkShaderModule mod; VkPipeline pipe; } g_d4[2];
#define D4 (g_d4[g_kd])
static void dsv4_init(void) {
    char path[1200];
    size_t pre = dir_prefix(KC.core.spv_path);
    const char *file = "chain_dsv4.spv";
    if (pre + strlen(file) + 1 >= sizeof path) return;
    memcpy(path, KC.core.spv_path, pre); strcpy(path + pre, file);
    FILE *f = fopen(path, "rb");
    if (!f) return;
    fclose(f);
    if ((D4.mod = load_module(KC.core.spv_path, file))) D4.pipe = make_pipe(D4.mod, NULL);
}
static void dsv4_shutdown(void) {
    if (D4.pipe) vkDestroyPipeline(KC.dev, D4.pipe, NULL);
    if (D4.mod) vkDestroyShaderModule(KC.dev, D4.mod, NULL);
    memset(&D4, 0, sizeof D4);
}
int vkc_dsv4_ready(void) { return vkc_ready() && D4.pipe; }

/* the shader reads one block of 32 words: the mode, then the op's fields */
typedef struct { int mode; int w[31]; } Dsv4PC;
static int dsv4_rec(int kind, int mode, const void *b, size_t bytes, VkcBind *bd, int nb, uint32_t gx, uint32_t gy) {
    if (!D4.pipe || !open_frame() || KC.lost || bytes > sizeof(int) * 31) return 0;
    Dsv4PC pc; memset(&pc, 0, sizeof pc);
    pc.mode = mode; memcpy(pc.w, b, bytes);
    KC.kind = kind;
    return record(D4.pipe, bd, nb, &pc, sizeof pc, gx, gy, 1);
}
int vkc_dsv4_attn(VkcBuf *q, VkcBuf *win, VkcBuf *cmp, VkcBuf *list, VkcBuf *prm, VkcBuf *out, const VkcDsAttn *p) {
    if (p->S < 1 || p->S > 65535 || p->H < 1 || p->H > 65535 || p->hd < 1 || p->hd > 1024 || p->cnt < 0 || p->cnt > 3072)
        return 0;
    if (p->cnt == 0) return 0;
    VkcBind bd[6] = {B(q, 0), B(win, 0), B(cmp ? cmp : win, 0), B(list, 0), B(prm, 0), B(out, 1)};
    return dsv4_rec(PK_ATTN, 0, p, sizeof *p, bd, 6, (uint32_t)p->H, (uint32_t)p->S);
}
int vkc_dsv4_rope(VkcBuf *x, VkcBuf *cs, const VkcDsRope *p) {
    if (p->nseg < 0 || p->per_row < 1 || p->rd < 2 || (p->rd & 1)) return 0;
    if (p->nseg == 0) return open_frame() && !KC.lost;
    VkcBind bd[6] = {B(NULL, 0), B(NULL, 0), B(NULL, 0), B(NULL, 0), B(cs, 0), B(x, 1)};
    uint32_t gx, gy; grid(((uint64_t)p->nseg * (p->rd / 2) + 255) / 256, &gx, &gy);
    return dsv4_rec(PK_ROPE, 1, p, sizeof *p, bd, 6, gx, gy);
}
int vkc_dsv4_compress(VkcBuf *kv, VkcBuf *sc, VkcBuf *ring, VkcBuf *prm, VkcBuf *out, const VkcDsComp *p) {
    if (p->S < 0 || p->ratio < 1 || p->P < 1 || p->D < 1 || p->D > p->P || (p->overlap && p->P < 2 * p->D)) return 0;
    if (p->S == 0) return open_frame() && !KC.lost;
    VkcBind bd[8] = {B(kv, 0), B(sc, 0), B(NULL, 0), B(NULL, 0), B(p->ape_off >= 0 ? prm : NULL, 0), B(out, 1), B(NULL, 0),
                     B(ring, 1)};
    return dsv4_rec(PK_DSA, 2, p, sizeof *p, bd, 8, 1, 1);
}
int vkc_dsv4_score(VkcBuf *iq, VkcBuf *hw, VkcBuf *keys, VkcBuf *mask, VkcBuf *sc, const VkcDsScore *p) {
    /* queries above 4096 floats: mode 9, the same sums with the queries read from memory */
    int wide = p->IH > 64 || p->IH * p->ID > 4096;
    if (p->S < 1 || p->S > 65535 || p->IH < 1 || p->IH > 4096 || p->ID < 1 || p->ratio < 1 ||
        p->width < 0 || p->sc_row < p->width || (p->mask_row > 0 && !mask)) return 0;
    if (p->width == 0) return open_frame() && !KC.lost;
    VkcBind bd[6] = {B(iq, 0), B(hw, 0), B(keys, 0), B(p->mask_row > 0 ? mask : NULL, 0), B(NULL, 0), B(sc, 1)};
    /* many columns: on a grid of 256 / IH columns a workgroup (mode 10, the same bits) */
    if (dsa_grid_on() && p->IH <= 256 && p->width >= VKC_DSA_GRID) {
        int P = 256 / p->IH;
        return dsv4_rec(PK_DSA, 10, p, sizeof *p, bd, 6, (uint32_t)((p->width + P - 1) / P), (uint32_t)p->S);
    }
    return dsv4_rec(PK_DSA, wide ? 9 : 3, p, sizeof *p, bd, 6, (uint32_t)p->S, 1);
}
int vkc_dsv4_cand(VkcBuf *sc, VkcBuf *mask, const VkcDsCand *p) {
    if (p->S < 1 || p->S > 65535 || p->block < 1 || p->ratio < 1 || p->width < 0 ||
        (p->width + p->block - 1) / p->block > 4096 || p->mask_row < p->width) return 0;
    VkcBind bd[7] = {B(sc, 0), B(NULL, 0), B(NULL, 0), B(NULL, 0), B(NULL, 0), B(NULL, 0), B(mask, 1)};
    return dsv4_rec(PK_DSA, 4, p, sizeof *p, bd, 7, (uint32_t)p->S, 1);
}
int vkc_dsv4_topk(VkcBuf *sc, VkcBuf *list, const VkcDsTopk *p) {
    if (p->S < 1 || p->S > 65535 || p->topk < 0 || p->width < 0 || (p->order && p->topk > 4096)) return 0;
    if (p->topk == 0) return open_frame() && !KC.lost;
    VkcBind bd[7] = {B(sc, 0), B(NULL, 0), B(NULL, 0), B(NULL, 0), B(NULL, 0), B(NULL, 0), B(list, 1)};
    return dsv4_rec(PK_DSA, 5, p, sizeof *p, bd, 7, (uint32_t)p->S, 1);
}
int vkc_dsv4_engram(VkcBuf *kv, VkcBuf *prm, VkcBuf *x, const VkcDsEngram *p) {
    if (p->S < 1 || p->S > 65535 || p->H < 1 || p->D < 1) return 0;
    VkcBind bd[6] = {B(kv, 0), B(prm, 0), B(NULL, 0), B(NULL, 0), B(NULL, 0), B(x, 1)};
    return dsv4_rec(PK_EW, 6, p, sizeof *p, bd, 6, (uint32_t)p->H, (uint32_t)p->S);
}
int vkc_dsv4_round(VkcBuf *x, VkcBuf *y, const VkcDsRound *p) {
    if (!x || !y || p->kind < VKC_DS_BF16 || p->kind > VKC_DS_HADAMARD || p->nseg < 0 || p->per_row < 1 || p->len < 1)
        return 0;
    if ((p->kind == VKC_DS_E4M3 || p->kind == VKC_DS_E2M1) && (p->block < 1 || p->block > 256)) return 0;
    if (p->kind == VKC_DS_HADAMARD && (p->len > 4096 || (p->len & (p->len - 1)))) return 0;
    if (p->nseg == 0) return open_frame() && !KC.lost;
    /* the fields, then whether it runs in place (x is read where y is written), then hscale */
    int w[14];
    memcpy(w, p, 12 * sizeof(int));
    w[12] = x == y;
    memcpy(&w[13], &p->hscale, sizeof(float));
    uint64_t groups = p->kind == VKC_DS_BF16 ? ((uint64_t)p->nseg * p->len + 255) / 256
                    : p->kind == VKC_DS_HADAMARD ? (uint64_t)p->nseg
                    : (uint64_t)p->nseg * ((p->len + p->block - 1) / p->block);
    uint32_t gx, gy; grid(groups, &gx, &gy);
    VkcBind bd[6] = {B(x == y ? NULL : x, 0), B(NULL, 0), B(NULL, 0), B(NULL, 0), B(NULL, 0), B(y, 1)};
    return dsv4_rec(PK_EW, 7, w, sizeof w, bd, 6, gx, gy);
}
int vkc_dsv4_swiglu(VkcBuf *a, VkcBuf *b, VkcBuf *y, const VkcDsSwiglu *p) {
    if (!a || !b || !y || p->n < 0) return 0;
    if (p->n == 0) return open_frame() && !KC.lost;
    uint32_t gx, gy; grid(((uint64_t)p->n + 255) / 256, &gx, &gy);
    VkcBind bd[6] = {B(a, 0), B(b, 0), B(NULL, 0), B(NULL, 0), B(NULL, 0), B(y, 1)};
    return dsv4_rec(PK_EW, 8, p, sizeof *p, bd, 6, gx, gy);
}

/* ---- a KV cache split between the device and the host (chain_kvs.comp) ------------
 * Its own optional pipeline, as DeepSeek's: without the shader only these ops decline
 * (and an engine's split with them: past the budget its chain declines as before). */
static struct { VkShaderModule mod; VkPipeline pipe; int tried; } g_ks[2];
#define KS (g_ks[g_kd])
static VkPipeline kvs_pipe(void) {
    if (KS.pipe || KS.tried || !vkc_ready()) return KS.pipe;
    KS.tried = 1;
    char path[1200];
    size_t pre = dir_prefix(KC.core.spv_path);
    const char *file = "chain_kvs.spv";
    if (pre + strlen(file) + 1 >= sizeof path) return VK_NULL_HANDLE;
    memcpy(path, KC.core.spv_path, pre); strcpy(path + pre, file);
    FILE *f = fopen(path, "rb");
    if (!f) return VK_NULL_HANDLE;
    fclose(f);
    if ((KS.mod = load_module(KC.core.spv_path, file))) KS.pipe = make_pipe(KS.mod, NULL);
    return KS.pipe;
}
static void kvs_shutdown(void) {
    if (KS.pipe) vkDestroyPipeline(KC.dev, KS.pipe, NULL);
    if (KS.mod) vkDestroyShaderModule(KC.dev, KS.mod, NULL);
    memset(&KS, 0, sizeof KS);
}
int vkc_kvs_ready(void) { return kvs_pipe() != VK_NULL_HANDLE; }
int vkc_kvs_attn(VkcBuf *q, VkcBuf *kc, VkcBuf *vc, VkcBuf *out, VkcBuf *gate, VkcBuf *sel, VkcBuf *snk, VkcBuf *tab,
                 const VkcKvsAttn *p) {
    VkPipeline pipe = kvs_pipe();
    if (!pipe || !q || !kc || !vc || !out || !tab || p->S < 1 || p->S > 65535 || p->H < 1 || p->H > 65535 ||
        p->KVH < 1 || p->H % p->KVH || p->hd < 1 || p->hd > 256 || p->vd < 1 || p->vd > 256 || p->B < 1 || p->ns < 1 ||
        p->nblk < 0 || p->anchor < 0 || p->win < 0 || (p->sink && !snk) || (p->sel_row > 0 && !sel) ||
        ((p->flags & VKC_KVS_GATE) && (!gate || !(p->flags & VKC_KVS_FIN)))) return 0;
    struct { int mode; VkcKvsAttn b; } pc = {0, *p};
    KC.kind = PK_ATTN;
    VkcBind bd[8] = {B(q, 0), B(kc, 0), B(vc, 0), B(out, 1), B(p->flags & VKC_KVS_GATE ? gate : NULL, 0),
                     B(p->sel_row > 0 ? sel : NULL, 0), B(p->sink ? snk : NULL, 0), B(tab, 0)};
    return record(pipe, bd, 8, &pc, sizeof pc, (uint32_t)p->H, (uint32_t)p->S, 1);
}
int vkc_kvs_mla(VkcBuf *qa, VkcBuf *qr, VkcBuf *lat, VkcBuf *rope, VkcBuf *sel, VkcBuf *out, VkcBuf *tab, const VkcKvsMla *p) {
    VkPipeline pipe = kvs_pipe();
    if (!pipe || !qa || !lat || !out || !tab || p->S < 1 || p->S > 65535 || p->H < 1 || p->H > 65535 || p->K < 1 ||
        p->K > 1024 || p->R < 0 || p->R > 128 || (p->R > 0 && (!rope || !qr)) || p->B < 1 || p->ns < 1 || p->nblk < 0 ||
        p->anchor < 0 ||
        p->kv_start < 0 || (p->sel_row > 0 && !sel)) return 0;
    struct { int mode; VkcKvsMla b; } pc = {1, *p};
    KC.kind = PK_MLA;
    VkcBind bd[8] = {B(qa, 0), B(p->R > 0 ? qr : NULL, 0), B(lat, 0), B(p->R > 0 ? rope : NULL, 0), B(NULL, 0),
                     B(p->sel_row > 0 ? sel : NULL, 0), B(out, 1), B(tab, 0)};
    return record(pipe, bd, 8, &pc, sizeof pc, (uint32_t)p->H, (uint32_t)p->S, 1);
}
int vkc_kvs_merge(VkcBuf *dev, VkcBuf *cpu, VkcBuf *gate, VkcBuf *out, const VkcKvsMerge *p) {
    VkPipeline pipe = kvs_pipe();
    if (!pipe || !dev || !cpu || !out || p->n < 0 || p->H < 1 || p->d < 1 || ((p->flags & VKC_KVS_GATE) && !gate)) return 0;
    if (p->n == 0) return open_frame() && !KC.lost;
    struct { int mode; VkcKvsMerge b; } pc = {2, *p};
    KC.kind = PK_ATTN;
    VkcBind bd[4] = {B(dev, 0), B(cpu, 0), B(p->flags & VKC_KVS_GATE ? gate : NULL, 0), B(out, 1)};
    uint32_t gx, gy; grid((uint64_t)p->n, &gx, &gy);
    return record(pipe, bd, 4, &pc, sizeof pc, gx, gy, 1);
}
int vkc_kvs_join(VkcBuf *parts, VkcBuf *out, const VkcKvsJoin *p) {
    VkPipeline pipe = kvs_pipe();
    if (!pipe || !parts || !out || p->n < 0 || p->d < 1 || p->nz < 0) return 0;
    if (p->n == 0) return open_frame() && !KC.lost;
    struct { int mode; VkcKvsJoin b; } pc = {5, *p};
    KC.kind = PK_ATTN;
    VkcBind bd[4] = {B(parts, 0), B(NULL, 0), B(NULL, 0), B(out, 1)};
    uint32_t gx, gy; grid((uint64_t)p->n, &gx, &gy);
    return record(pipe, bd, 4, &pc, sizeof pc, gx, gy, 1);
}
int vkc_kvs_rel(VkcBuf *q, VkcBuf *kc, VkcBuf *vc, VkcBuf *out, VkcBuf *r, VkcBuf *tau, VkcBuf *relp, VkcBuf *tab,
                const VkcKvsRel *p) {
    VkPipeline pipe = kvs_pipe();
    if (!pipe || !q || !kc || !vc || !out || !tau || !tab || p->S < 1 || p->S > 65535 || p->H < 1 || p->H > 65535 ||
        p->KVH < 1 || p->H % p->KVH || p->hd < 1 || p->hd > 256 || p->d_rel < 0 || p->d_rel > 64 ||
        (p->d_rel > 0 && (!r || !relp)) || p->B < 1 || p->nblk < 0 || p->anchor < 0) return 0;
    struct { int mode; VkcKvsRel b; } pc = {3, *p};
    KC.kind = PK_ATTN;
    VkcBind bd[8] = {B(q, 0), B(kc, 0), B(vc, 0), B(out, 1), B(p->d_rel > 0 ? r : NULL, 0), B(tau, 0),
                     B(p->d_rel > 0 ? relp : NULL, 0), B(tab, 0)};
    return record(pipe, bd, 8, &pc, sizeof pc, (uint32_t)p->H, (uint32_t)p->S, 1);
}
int vkc_kvs_ds_cold(VkcBuf *q, VkcBuf *win, VkcBuf *cmp, VkcBuf *cold, VkcBuf *out, VkcBuf *prm, VkcBuf *list, VkcBuf *tab, const VkcKvsDs *p) {
    VkPipeline pipe = kvs_pipe();
    if (!pipe || !q || !win || !out || !prm || !list || !tab || p->S < 1 || p->S > 65535 || p->H < 1 || p->H > 65535 ||
        p->hd < 1 || p->hd > 1024 || p->cnt < 0 || p->B < 1 || p->ns < 1 || p->nblk < 0 ||
        ((p->flags & VKC_KVS_DSCOLD) && (!cold || !(p->flags & VKC_KVS_DSFIN))) ||
        ((p->flags & VKC_KVS_V4) && !(p->flags & VKC_KVS_DSFIN))) return 0;
    struct { int mode; VkcKvsDs b; } pc = {4, *p};
    KC.kind = PK_ATTN;
    VkcBind bd[8] = {B(q, 0), B(win, 0), B(cmp ? cmp : win, 0), B(out, 1), B(prm, 0), B(list, 0), B(cold, 0), B(tab, 0)};
    return record(pipe, bd, 8, &pc, sizeof pc, (uint32_t)p->H, (uint32_t)p->S, 1);
}
int vkc_kvs_ds(VkcBuf *q, VkcBuf *win, VkcBuf *cmp, VkcBuf *out, VkcBuf *prm, VkcBuf *list, VkcBuf *tab, const VkcKvsDs *p) {
    return vkc_kvs_ds_cold(q, win, cmp, NULL, out, prm, list, tab, p);
}
unsigned long long vkc_serial(void) {
    unsigned long long s = 0;
    for (int i = 0; i < VKC_FRAMES; i++)
        if (!KC.fr[i].open && KC.fr[i].serial > s) s = KC.fr[i].serial;
    return s;
}
int vkc_wait_serial(unsigned long long serial) {
    if (!vkc_ready()) return 0;
    for (int i = 0; i < VKC_FRAMES; i++) {
        VkcFrame *f = &KC.fr[i];
        if (f->serial == serial && !f->open) return frame_wait(f) && !KC.lost;
    }
    return !KC.lost;   /* reused since: it completed long ago */
}
