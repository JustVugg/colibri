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
#include "vk_alloc.h"
#include <vulkan/vulkan.h>
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
#define VKC_FRAMES 4
#define VKC_BIND 8
#define VKC_TRACK 64
#define VKC_PROF_Q 512
/* COLI_VK_CHAIN_PROF=1: device time per kind of op (timestamps after every op; ops
 * that run side by side between two barriers share the time unevenly) */
enum { PK_GEMV8, PK_GEMV, PK_GEMM, PK_NORM, PK_ROPE, PK_ATTN, PK_DNCONV, PK_DNREC, PK_EW, PK_QSA, PK_PLE, PK_COPY,
       PK_MLA, PK_MLAW, PK_DSA, PK_KDA, PK_MHC, PK_ARES, PK_N };
static const char *const pk_name[PK_N] = {"GEMV int8", "GEMV other", "tiled GEMM", "norm", "rope", "attention",
                                          "dn conv", "dn recurrence", "element-wise", "qsa", "ple", "copy",
                                          "mla", "mla weights", "dsa", "kda", "mhc", "attnres"};
/* the MLA, KDA, mHC and AttnRes pipelines: optional (an engine without such layers
 * needs none of them, and a missing one turns off only its own ops) */
enum { PM_MLA, PM_HGEMV, PM_DSA, PM_KDA, PM_MHC, PM_ARES, PM_N };
static const char *const mla_file[PM_N] = {"chain_mla.spv", "chain_hgemv.spv", "chain_dsa.spv", "chain_kda.spv",
                                           "chain_mhc.spv", "chain_ares.spv"};
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
    VkQueryPool qp; int nq; unsigned char kind[VKC_PROF_Q];   /* COLI_VK_CHAIN_PROF: a timestamp after each op */
} VkcFrame;

typedef struct {
    int ready, lost;
    int misuse;                       /* a bind of the other device's buffer: the record fails */
    ColiVkCore core;
    VkDevice dev;
    VkQueue queue;
    VkDescriptorSetLayout dsl;
    VkPipelineLayout pl;
    VkShaderModule mod[P_NPIPE], mod_gemm, mod_dnrec;
    VkShaderModule mod_gv2; VkPipeline gv2;   /* chain_gemv2.comp: int8 decode GEMVs (COLI_VK_CHAIN_GEMV2) */
    VkPipeline pipe[P_NPIPE];
    VkPipeline gemm[VKC_GEMM_MAX]; int gemm_bm[VKC_GEMM_MAX], gemm_bn[VKC_GEMM_MAX], ngemm;
    VkPipeline dnrec[VKC_DNREC_MAX]; int dnrec_kd[VKC_DNREC_MAX], ndnrec;
    VkShaderModule mod_gemv4; VkPipeline gemv4; int gemv4_xs;   /* chain_gemv.comp: the vectorized decode GEMV */
    VkPipeline gemv4_1;   /* chain_gemv.comp for a decode step: one row's registers */
    VkShaderModule mmod[PM_N]; VkPipeline mpipe[PM_N]; int mla_ok;   /* chain_mla, chain_hgemv, chain_dsa */
    VkPipeline kdarec[VKC_KDA_MAX]; int kdarec_kd[VKC_KDA_MAX], nkdarec;   /* chain_kda's recurrence per key dim */
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
} VkcCtx;
/* One context per device (vkc_device): 0 the primary, 1 COLI_VK_DEV2's, for the layers an
 * engine places there. Every call works on the current device's; a buffer remembers its
 * device, and binding or copying one of the other device's is refused. */
static VkcCtx g_kctx[2] = {{.cur = -1, .gemm_rows = -1}, {.cur = -1, .gemm_rows = -1}};
static int g_kd;
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
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
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
static VkShaderModule load_module(const char *dir_spv, const char *file) {
    char path[1200];
    const char *sl = strrchr(dir_spv, '/');
    size_t pre = sl ? (size_t)(sl - dir_spv) + 1 : 0;
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
static VkPipeline make_pipe(VkShaderModule m, const VkSpecializationInfo *si) {
    VkComputePipelineCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                  .module = m, .pName = "main", .pSpecializationInfo = si}, .layout = KC.pl};
    VkPipeline p = VK_NULL_HANDLE;
    if (vkCreateComputePipelines(KC.dev, VK_NULL_HANDLE, 1, &ci, NULL, &p) != VK_SUCCESS) return VK_NULL_HANDLE;
    return p;
}

static void dsv4_init(void);       /* chain_dsv4.comp (DeepSeek V4.1 / V4), below */
static void dsv4_shutdown(void);
static void kvs_shutdown(void);    /* chain_kvs.comp (the split KV cache), below */
int vkc_init(void) {
    if (KC.ready) return !KC.lost;
    if (!coli_vk_core_dev(g_kd, &KC.core)) return 0;
    KC.dev = (VkDevice)KC.core.device; KC.queue = (VkQueue)KC.core.queue;
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
        int xs = (int)(pp.limits.maxComputeSharedMemorySize / 16);
        if (xs > 4096) xs = 4096;
        if (!(e && *e == '0') && xs >= 256 && (KC.mod_gemv4 = load_module(KC.core.spv_path, "chain_gemv.spv"))) {
            /* XS, rows a workgroup at most, steps a lane loads ahead (COLI_VK_CHAIN_GEMV_UNR) */
            const char *ue = getenv("COLI_VK_CHAIN_GEMV_UNR");
            const char *u1 = getenv("COLI_VK_CHAIN_GEMV_UNR1");
            int32_t v[3] = {xs, 4, ue && *ue ? atoi(ue) : 4};
            VkSpecializationMapEntry me[3] = {{0, 0, 4}, {1, 4, 4}, {2, 8, 4}};
            VkSpecializationInfo si = {3, me, 12, v};
            if ((KC.gemv4 = make_pipe(KC.mod_gemv4, &si))) KC.gemv4_xs = xs;
            v[1] = 1; v[2] = u1 && *u1 ? atoi(u1) : 8;   /* a decode step's: one row's registers */
            if (KC.gemv4) KC.gemv4_1 = make_pipe(KC.mod_gemv4, &si);
        }
    }
    /* the int8/int4 decode GEMV with lanes per row and rows per workgroup chosen per
     * matrix (chain_gemv2.comp): needs clustered subgroup operations */
    {
        const char *e = getenv("COLI_VK_CHAIN_GEMV2");
        VkPhysicalDeviceSubgroupProperties sgp = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
        VkPhysicalDeviceProperties2 p2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &sgp};
        vkGetPhysicalDeviceProperties2((VkPhysicalDevice)KC.core.phys, &p2);
        int ok = (sgp.supportedOperations & VK_SUBGROUP_FEATURE_CLUSTERED_BIT) && (sgp.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) &&
                 sgp.subgroupSize >= 16 && !(e && *e == '0');
        if (ok && (KC.mod_gv2 = load_module(KC.core.spv_path, "chain_gemv2.spv"))) KC.gv2 = make_pipe(KC.mod_gv2, NULL);
    }
    /* the MLA, KDA and mHC shaders, optional: without one only its ops decline */
    for (int i = 0; i < PM_N; i++) {
        char path[1200];
        const char *sl = strrchr(KC.core.spv_path, '/');
        size_t pre = sl ? (size_t)(sl - KC.core.spv_path) + 1 : 0;
        FILE *f = NULL;
        if (pre + strlen(mla_file[i]) + 1 < sizeof path) {
            memcpy(path, KC.core.spv_path, pre); strcpy(path + pre, mla_file[i]);
            f = fopen(path, "rb");
        }
        if (f) fclose(f);
        if (f && (KC.mmod[i] = load_module(KC.core.spv_path, mla_file[i]))) KC.mpipe[i] = make_pipe(KC.mmod[i], NULL);
    }
    KC.mla_ok = KC.mpipe[PM_MLA] && KC.mpipe[PM_HGEMV] && KC.mpipe[PM_DSA];
    dsv4_init();
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
    {
        const char *e = getenv("COLI_VK_CHAIN_PROF");
        uint32_t nqf = 0; vkGetPhysicalDeviceQueueFamilyProperties((VkPhysicalDevice)KC.core.phys, &nqf, NULL);
        VkQueueFamilyProperties qf[16]; if (nqf > 16) nqf = 16;
        vkGetPhysicalDeviceQueueFamilyProperties((VkPhysicalDevice)KC.core.phys, &nqf, qf);
        VkPhysicalDeviceProperties pp; vkGetPhysicalDeviceProperties((VkPhysicalDevice)KC.core.phys, &pp);
        uint32_t bits = KC.core.qfam < nqf ? qf[KC.core.qfam].timestampValidBits : 0;
        if (e && *e == '1' && bits && pp.limits.timestampPeriod > 0) {
            KC.prof = 1; KC.ts_period = pp.limits.timestampPeriod;
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
 * poll the fence for up to COLI_VK_CHAIN_SPIN_US (2000 us) before blocking, whose
 * wake-up costs 50 to 150 us per frame, two frames a layer. */
static long chain_spin_us(void) {
    static long v = -1;
    if (v < 0) { const char *e = getenv("COLI_VK_CHAIN_SPIN_US"); v = e && *e ? atol(e) : 2000; if (v < 0) v = 0; }
    return v;
}
static int frame_wait(VkcFrame *f) {
    if (!f->inflight) return 1;
    double t0 = vkc_now_ms();
    VkResult r = VK_NOT_READY;
    for (long spin = chain_spin_us(); spin > 0 && r == VK_NOT_READY && (vkc_now_ms() - t0) * 1000.0 < spin; )
        r = vkGetFenceStatus(KC.dev, f->fence);
    if (r == VK_NOT_READY) r = vkWaitForFences(KC.dev, 1, &f->fence, VK_TRUE, 20000000000ULL);
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
            }
        f->nq = 0;
    }
    frame_reclaim(f);
    return 1;
}
static void prof_mark(VkcFrame *f, int kind) {
    if (!KC.prof || !f->qp || f->nq >= VKC_PROF_Q) return;
    vkCmdWriteTimestamp(f->cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, f->qp, (uint32_t)f->nq);
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
    /* after everything submitted earlier on this queue (and the host's writes) */
    barrier_now(f->cmd, ALL_STAGES, ALL_RW);
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
    VkResult r = (VkResult)coli_vk_queue_submit_dev(g_kd, KC.queue, &si, f->fence);   /* the backend's lock, staged uploads */
    if (r != VK_SUCCESS) { lose("queue submit", r); return 0; }
    f->inflight = 1;
    KC.st.frames++;
    return wait ? frame_wait(f) : 1;
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
int vkc_copy(VkcBuf *dst, size_t doff, VkcBuf *src, size_t soff, size_t n) {
    return copy_bytes(dst, doff * 4, src, soff * 4, n * 4);
}
int vkc_copy_regions(VkcBuf *dst, VkcBuf *src, const VkcRegion *r, int n) {
    VkcFrame *f = open_frame();
    if (!f || KC.lost) return 0;
    if (n < 1) return 1;
    VkBufferCopy *c = malloc((size_t)n * sizeof *c);
    if (!c) return 0;
    for (int i = 0; i < n; i++) {
        if ((r[i].dst + r[i].n) * 4 > dst->bytes || (r[i].src + r[i].n) * 4 > src->bytes) {
            fprintf(stderr, "[VK] chain: copy region out of range\n"); free(c); return 0;
        }
        c[i] = (VkBufferCopy){r[i].src * 4, r[i].dst * 4, r[i].n * 4};
    }
    VkBuffer rb = src->buf, wb = dst->buf;
    hazard(f, &rb, 1, &wb, 1);
    vkCmdCopyBuffer(f->cmd, src->buf, dst->buf, (uint32_t)n, c);
    prof_mark(f, PK_COPY);
    src->last_frame = dst->last_frame = KC.serial;
    free(c);
    return 1;
}
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
        size_t want = f->stage ? 2 * f->stage->bytes : (size_t)4 << 20;
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
static int matmul_aligned(const ColiVkTensorInfo *ti, VkcBuf *x, size_t xb, VkcBuf *y, size_t yb, int S) {
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
    /* the vectorized GEMV where the row is whole 16-byte steps and fits the staging */
    int per = ti->fmt == 1 ? 16 : ti->fmt == 4 ? 32 : ti->fmt == 11 || ti->fmt == 14 ? 8 : ti->fmt == 10 ? 4 : 0;
    int v4 = path < 0 && KC.gemv4 && per && ti->rowWords % 4 == 0 && ti->I % 4 == 0 &&
             (ti->fmt != 4 || (ti->gs > 0 && ti->gs % 32 == 0)) && (ti->rowWords / 4) * per / 4 <= KC.gemv4_xs;
    KC.kind = path >= 0 ? PK_GEMM : ti->fmt == 1 ? PK_GEMV8 : PK_GEMV;
    int g2 = path < 0 && KC.gv2 && S <= 4 && ti->fmt == 1 &&   /* int4 measured slower than chain_gemv here */
             ti->I % 32 == 0 && ti->I <= 16384 && ti->rowWords % 4 == 0;
    if (g2) {
        /* rows a workgroup: short matrices in small blocks, so they still fill the device */
        int rpw = ti->O <= 1024 ? 16 : ti->O <= 4096 ? 32 : 64;
        /* a speculative verify's rows (S <= 4) share each weight load */
        int nr = S <= 4 && S * ti->I <= 16384 ? S : 1;
        struct { int fmt, S, I, O, rowWords, gs, rpw, nr; } pc8 = {ti->fmt, S, ti->I, ti->O, ti->rowWords, ti->gs, rpw, nr};
        ok = record(KC.gv2, bd, 4, &pc8, sizeof pc8, (uint32_t)((ti->O + rpw - 1) / rpw), (uint32_t)((S + nr - 1) / nr), 1);
    }
    else if (v4) {
        /* each workgroup stages x once: give it enough rows that the staging does not
         * rival the weights, while keeping enough workgroups to fill the device */
        /* lanes per row: about sixteen 16-byte steps each, eight for int4, whose steps
         * carry twice the arithmetic (measured on a Radeon 8060S: int4 4096 -> 2048
         * 52 -> 43 us, 2048 -> 8192 71 -> 62 us); COLI_VK_CHAIN_GEMV_LPR overrides */
        const char *e = getenv("COLI_VK_CHAIN_GEMV_LPR");
        int nq = ti->rowWords / 4, lpr = 4, st = ti->fmt == 4 ? 8 : 16;
        if (e && *e) lpr = atoi(e); else while (lpr < 64 && lpr * st < nq) lpr *= 2;
        if (lpr < 1) lpr = 1;
        /* a verify's rows (S <= 4) in one workgroup each weight step is loaded once for;
         * a decode step is nr = 1 through the same code, so the bits agree */
        int nx4 = (ti->rowWords / 4) * per / 4, nr = S <= 4 && S * nx4 <= KC.gemv4_xs ? S : 1;
        struct { int fmt, S, I, O, rowWords, gs, lpr, nr; } pc8 = {ti->fmt, S, ti->I, ti->O, ti->rowWords, ti->gs, lpr, nr};
        int rows_wg = 256 / lpr;   /* at least, at subgroups of 64 */
        int wg = (ti->O + rows_wg - 1) / rows_wg; if (wg > 1024) wg = 1024;
        ok = record(nr == 1 && KC.gemv4_1 ? KC.gemv4_1 : KC.gemv4, bd, 4, &pc8, sizeof pc8, (uint32_t)wg, (uint32_t)((S + nr - 1) / nr), 1);
    }
    else if (path >= 0)
        ok = record(KC.gemm[path], bd, 4, &pc, sizeof pc, (uint32_t)((ti->O + KC.gemm_bm[path] - 1) / KC.gemm_bm[path]),
                    (uint32_t)((S + KC.gemm_bn[path] - 1) / KC.gemm_bn[path]), 1);
    else ok = record(KC.pipe[P_GEMV], bd, 4, &pc, sizeof pc, (uint32_t)((ti->O + 7) / 8), (uint32_t)S, 1);
    KC.st.matmuls += ok; KC.st.gemms += ok && path >= 0;
    return ok;
}
int vkc_matmul(ColiVkTensor *t, VkcBuf *x, size_t xo, VkcBuf *y, size_t yo, int S) {
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
/* ---- blocked attention for prompt chunks (chain_attnb.comp), made on first use ----
 * chain_attn reads a row's positions once per (head, row): a chunk of S rows reads its
 * KV cache S * H times, which held a 2048-row chunk of Qwen3.6 at 1.2 s a layer on a
 * Radeon 780M. From COLI_VK_ATTN_BLOCK rows (16; 0 = never) a call takes chain_attnb,
 * one workgroup per KV head and block of rows: each K and V row read once per block.
 * Below it, and for decode and an MTP verify, chain_attn's bits as before. */
static struct { VkShaderModule mod; VkPipeline pipe; int tried, bc; } g_kab[2];
#define KAB (g_kab[g_kd])
static VkPipeline kab_pipe(void) {
    if (KAB.pipe || KAB.tried || !vkc_ready()) return KAB.pipe;
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
    return KAB.pipe;
}
static void kab_shutdown(void) {
    if (KAB.pipe) vkDestroyPipeline(KC.dev, KAB.pipe, NULL);
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
int vkc_attn_block_rows(void) {
    const char *e = getenv("COLI_VK_ATTN_BLOCK");
    int v = e && *e ? atoi(e) : 16;
    return v < 0 ? 0 : v;
}
int vkc_attn_w(VkcBuf *q, VkcBuf *kc, VkcBuf *vc, VkcBuf *o, VkcBuf *gate, VkcBuf *sel, VkcBuf *snk,
               const VkcAttnW *p) {
    KC.kind = PK_ATTN;
    VkcAttnW w = *p;   /* the shader's push constants: VkcAttn's fields, then these */
    if (w.vd <= 0) w.vd = w.a.hd;
    if (w.a.hd > 256 || w.vd > 256 || w.a.H % w.a.KVH || w.win < 0 || w.ring < 0 || (w.sink && !snk)) return 0;
    VkcBind bd[7] = {B(q, 0), B(kc, 0), B(vc, 0), B(o, 1), B(gate, 0), B(sel, 0), B(snk, 0)};
    int G = w.a.H / w.a.KVH, blk = vkc_attn_block_rows();
    int blocked = blk > 0 && w.a.S >= blk && G <= 32 && kab_pipe(), br = w.a.sel_row > 0 ? 1 : 32 / G;
    int S = w.a.S, rr = attn_slice_rows(S, w.a.pos_base, (double)w.a.H * (w.a.hd > w.vd ? w.a.hd : w.vd), blocked ? br : 1);
    int ok = 1;
    for (int r0 = 0; ok && r0 < S; r0 += rr) {
        VkcAttnW x = w;
        int n = S - r0 < rr ? S - r0 : rr;
        x.a.S = n; x.a.pos_base = w.a.pos_base + r0;
        x.a.q_off += r0 * w.a.q_row; x.a.g_off += r0 * w.a.g_row; x.a.o_off += r0 * w.a.o_row;
        if (w.a.sel_row > 0) x.a.sel_off += r0 * w.a.sel_row;
        if (r0 && !attn_slice_next()) return 0;
        if (blocked) {
            struct { VkcAttnW w; int br, B, anchor, part, st_off; } pc = {x, br, 0, 0, 0, 0};
            ok = record(KAB.pipe, bd, 7, &pc, sizeof pc, (uint32_t)x.a.KVH, (uint32_t)((x.a.S + br - 1) / br), 1);
            KC.st.attn_blocked += ok;
        } else ok = record(KC.pipe[P_ATTN], bd, 7, &x, sizeof x, (uint32_t)x.a.H, (uint32_t)x.a.S, 1);
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
    int32_t kd = KD;
    VkSpecializationMapEntry me = {0, 0, 4};
    VkSpecializationInfo si = {1, &me, 4, &kd};
    VkPipeline p = make_pipe(KC.mod_dnrec, &si);
    if (!p) return VK_NULL_HANDLE;
    KC.dnrec[KC.ndnrec] = p; KC.dnrec_kd[KC.ndnrec++] = KD;
    return p;
}
int vkc_dnrec(int KD, VkcBuf *cv, VkcBuf *ab, VkcBuf *z, VkcBuf *st, VkcBuf *prm, VkcBuf *y, VkcBuf *snap, const VkcDnRec *p) {
    KC.kind = PK_DNREC;
    if (p->VD > 128 || KD < 1 || KD > 256 || p->VH % p->KH) return 0;
    VkPipeline pipe = dnrec_pipe(KD);
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
int vkc_qsa(VkcBuf *src, VkcBuf *w, VkcBuf *pk, VkcBuf *cs, VkcBuf *sc, VkcBuf *sel, const VkcQsa *p) {
    KC.kind = PK_QSA;
    if (p->ID > 256) return 0;
    if (p->mode == 0) {
        VkcBind bd[4] = {B(src, 0), B(w, 0), B(pk, 1), B(cs, 0)};
        return record(KC.pipe[P_QSA], bd, 4, p, sizeof *p, (uint32_t)p->nb, 1, 1);
    }
    VkcBind bd[6] = {B(src, 0), {VK_NULL_HANDLE, 0, 0, 0}, B(pk, 0), {VK_NULL_HANDLE, 0, 0, 0}, B(sc, 1), B(sel, 1)};
    return record(KC.pipe[P_QSA], bd, 6, p, sizeof *p, (uint32_t)p->S, 1, 1);
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

int vkc_mla_core(VkcBuf *qabs, VkcBuf *qr, VkcBuf *lat, VkcBuf *rope, VkcBuf *sel, VkcBuf *clat, const VkcMlaCore *p) {
    KC.kind = PK_MLA;
    if (!KC.mla_ok || p->S < 1 || p->H < 1 || p->K < 1 || p->K > 1024 || p->R < 0 || p->R > 128 ||
        (p->R > 0 && !rope) || p->pos_base < 0 || p->kv_start < 0 || p->kv_start > p->pos_base) return 0;
    typedef struct { int mode; VkcMlaCore b; } CorePC;
    CorePC pc = {0, *p};
    if (!sel) pc.b.sel_row = 0;
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

int vkc_dsa_select(VkcBuf *iq, VkcBuf *hw, VkcBuf *keys, VkcBuf *sc, VkcBuf *sel, const VkcDsa *p) {
    KC.kind = PK_DSA;
    if (!KC.mla_ok || p->S < 1 || p->S > 65535 || p->IH < 1 || p->IH > 64 || p->ID < 1 || p->IH * p->ID > 4096 ||
        p->topk < 1 || p->pos_base < 0 || p->sc_row < p->pos_base + p->S ||
        p->sel_row < 1 + (p->topk < p->pos_base + p->S ? p->topk : p->pos_base + p->S)) return 0;
    struct { int mode; VkcDsa b; } pc = {0, *p};
    VkcBind bd[5] = {B(iq, 0), B(hw, 0), B(keys, 0), B(sc, 1), B(sel, 1)};
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
    struct { int mode; VkcDsaPick b; } pc = {2, *p};
    VkcBind bd[6] = {B(iq, 0), B(hw, 0), B(NULL, 0), B(sc, 1), B(sel, 1), B(pk, 0)};
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
int vkc_kda_rec(int KD, VkcBuf *m, VkcBuf *f, VkcBuf *b, VkcBuf *g, VkcBuf *prm, VkcBuf *st, VkcBuf *y, const VkcKdaRec *p) {
    return vkc_kda_rec_flags(KD, m, f, b, g, prm, st, y, p, 0);
}
int vkc_kda_rec_flags(int KD, VkcBuf *m, VkcBuf *f, VkcBuf *b, VkcBuf *g, VkcBuf *prm, VkcBuf *st, VkcBuf *y,
                      const VkcKdaRec *p, int flags) {
    KC.kind = PK_KDA;
    if (!KC.mpipe[PM_KDA] || KD < 1 || KD > 256 || p->VD < 1 || p->VD > 128 || p->H < 1 || p->S < 1 ||
        (flags & ~(VKC_KDA_EXP_A | VKC_KDA_K3))) return 0;
    VkPipeline pipe = kda_pipe(KD);
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
           (size_t)VKC_FRAMES * ((size_t)4 << 20);
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
    if (KC.dummy) buf_release(KC.dummy);
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
    ke_shutdown();
    for (int i = 0; i < P_NPIPE; i++) {
        if (KC.pipe[i]) vkDestroyPipeline(KC.dev, KC.pipe[i], NULL);
        if (KC.mod[i]) vkDestroyShaderModule(KC.dev, KC.mod[i], NULL);
    }
    for (int i = 0; i < KC.nkdarec; i++) vkDestroyPipeline(KC.dev, KC.kdarec[i], NULL);
    for (int i = 0; i < PM_N; i++) {
        if (KC.mpipe[i]) vkDestroyPipeline(KC.dev, KC.mpipe[i], NULL);
        if (KC.mmod[i]) vkDestroyShaderModule(KC.dev, KC.mmod[i], NULL);
    }
    for (int i = 0; i < KC.ngemm; i++) vkDestroyPipeline(KC.dev, KC.gemm[i], NULL);
    for (int i = 0; i < KC.ndnrec; i++) vkDestroyPipeline(KC.dev, KC.dnrec[i], NULL);
    if (KC.gemv4) vkDestroyPipeline(KC.dev, KC.gemv4, NULL);
    if (KC.gemv4_1) vkDestroyPipeline(KC.dev, KC.gemv4_1, NULL);
    if (KC.mod_gemv4) vkDestroyShaderModule(KC.dev, KC.mod_gemv4, NULL);
    if (KC.gv2) vkDestroyPipeline(KC.dev, KC.gv2, NULL);
    if (KC.mod_gv2) vkDestroyShaderModule(KC.dev, KC.mod_gv2, NULL);
    if (KC.mod_gemm) vkDestroyShaderModule(KC.dev, KC.mod_gemm, NULL);
    if (KC.mod_dnrec) vkDestroyShaderModule(KC.dev, KC.mod_dnrec, NULL);
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
    const char *sl = strrchr(KC.core.spv_path, '/');
    size_t pre = sl ? (size_t)(sl - KC.core.spv_path) + 1 : 0;
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
    const char *sl = strrchr(KC.core.spv_path, '/');
    size_t pre = sl ? (size_t)(sl - KC.core.spv_path) + 1 : 0;
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
