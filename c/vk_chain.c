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
    int kind;
    size_t bytes;
    VkaRange r;
    uint8_t *ptr;
    unsigned long long last_frame;   /* serial of the last frame that bound it */
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

static struct {
    int ready, lost;
    ColiVkCore core;
    VkDevice dev;
    VkQueue queue;
    VkDescriptorSetLayout dsl;
    VkPipelineLayout pl;
    VkShaderModule mod[P_NPIPE], mod_gemm, mod_dnrec;
    VkPipeline pipe[P_NPIPE];
    VkPipeline gemm[VKC_GEMM_MAX]; int gemm_bm[VKC_GEMM_MAX], gemm_bn[VKC_GEMM_MAX], ngemm;
    VkPipeline dnrec[VKC_DNREC_MAX]; int dnrec_kd[VKC_DNREC_MAX], ndnrec;
    VkShaderModule mod_gemv4; VkPipeline gemv4; int gemv4_xs;   /* chain_gemv.comp: the vectorized decode GEMV */
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
} K = {.cur = -1, .gemm_rows = -1};

int vkc_ready(void) { return K.ready && !K.lost; }
int vkc_lost(void) { return K.lost; }

static void lose(const char *what, VkResult r) {
    if (!K.lost) fprintf(stderr, "[VK] chain: %s failed (%d): the device is lost\n", what, (int)r);
    K.lost = 1;
    coli_vk_mark_lost();
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
    if (K.core.has_prio) ai.pNext = &pri;
#endif
    VkcBlock *bk = calloc(1, sizeof *bk);
    if (!bk) return 0;
    if (vkAllocateMemory(K.dev, &ai, NULL, &bk->mem) != VK_SUCCESS) { free(bk); return 0; }
    if (P->mapped && vkMapMemory(K.dev, bk->mem, 0, cap, 0, (void **)&bk->map) != VK_SUCCESS) {
        vkFreeMemory(K.dev, bk->mem, NULL); free(bk); return 0;
    }
    if (vka_pool_add_block(&P->p, cap, bk) < 0) {
        if (bk->map) vkUnmapMemory(K.dev, bk->mem);
        vkFreeMemory(K.dev, bk->mem, NULL); free(bk); return 0;
    }
    return 1;
}
static uint32_t memtype_of(int kind) {
    return kind == VKC_UP ? K.core.memtype_host : kind == VKC_DOWN ? K.core.memtype_cached : K.core.memtype_dev;
}
static int memtype_host_visible(uint32_t t) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties((VkPhysicalDevice)K.core.phys, &mp);
    return (mp.memoryTypes[t].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
}

static void track_reset(void) { K.nwr = K.nrd = 0; }
static void zero_now(VkcBuf *b);

VkcBuf *vkc_buf(size_t bytes, int kind) {
    if (!vkc_ready() || kind < 0 || kind > 2) return NULL;
    if (!bytes) bytes = 4;
    bytes = (bytes + 3) & ~(size_t)3;
    if (K.core.ssbo_range && bytes > K.core.ssbo_range) return NULL;   /* one binding could not address it */
    VkcBuf *b = calloc(1, sizeof *b);
    if (!b) return NULL;
    b->kind = kind; b->bytes = bytes;
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = bytes,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    if (vkCreateBuffer(K.dev, &bi, NULL, &b->buf) != VK_SUCCESS) { free(b); return NULL; }
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(K.dev, b->buf, &req);
    VkcPool *P = &K.pool[kind];
    uint64_t align = req.alignment > K.align ? req.alignment : K.align;
    if (!(req.memoryTypeBits & (1u << P->memtype)) ||
        (!vka_alloc(&P->p, req.size, align, &b->r) &&
         !(pool_grow(P, req.size + align) && vka_alloc(&P->p, req.size, align, &b->r)))) {
        vkDestroyBuffer(K.dev, b->buf, NULL); free(b); return NULL;
    }
    VkcBlock *bk = P->p.b[b->r.block].user;
    if (vkBindBufferMemory(K.dev, b->buf, bk->mem, b->r.off) != VK_SUCCESS) {
        if (vka_free(&P->p, b->r)) { VkcBlock *e = P->p.b[b->r.block].user;
            if (e) { if (e->map) vkUnmapMemory(K.dev, e->mem); vkFreeMemory(K.dev, e->mem, NULL); free(e); }
            vka_pool_drop_block(&P->p, b->r.block); }
        vkDestroyBuffer(K.dev, b->buf, NULL); free(b); return NULL;
    }
    b->ptr = bk->map ? bk->map + b->r.off : NULL;
    K.st.dev_bytes += bytes;
    if (b->ptr) memset(b->ptr, 0, bytes);
    else zero_now(b);
    return b;
}

static void buf_release(VkcBuf *b) {
    VkcPool *P = &K.pool[b->kind];
    vkDestroyBuffer(K.dev, b->buf, NULL);
    if (vka_free(&P->p, b->r)) {
        VkcBlock *e = P->p.b[b->r.block].user;
        if (e) { if (e->map) vkUnmapMemory(K.dev, e->mem); vkFreeMemory(K.dev, e->mem, NULL); free(e); }
        vka_pool_drop_block(&P->p, b->r.block);
    }
    K.st.dev_bytes -= b->bytes;
    free(b);
}

/* a frame that may still read b must finish first */
static int frame_wait(VkcFrame *f);
void vkc_free(VkcBuf *b) {
    if (!b) return;
    if (!K.dev) { free(b); return; }
    if (!K.lost)
        for (int i = 0; i < VKC_FRAMES; i++) {
            VkcFrame *f = &K.fr[i];
            if ((f->inflight || f->open) && f->serial == b->last_frame) {
                if (f->open) vkc_submit(1); else frame_wait(f);
            }
        }
    for (int i = 0; i < K.nwr; i++) if (K.wr[i] == b->buf) K.wr[i] = VK_NULL_HANDLE;
    for (int i = 0; i < K.nrd; i++) if (K.rd[i] == b->buf) K.rd[i] = VK_NULL_HANDLE;
    buf_release(b);
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
    if (vkCreateShaderModule(K.dev, &ci, NULL, &m) != VK_SUCCESS) m = VK_NULL_HANDLE;
    free(code);
    if (!m) fprintf(stderr, "[VK] chain: cannot load %s\n", path);
    return m;
}
static VkPipeline make_pipe(VkShaderModule m, const VkSpecializationInfo *si) {
    VkComputePipelineCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                  .module = m, .pName = "main", .pSpecializationInfo = si}, .layout = K.pl};
    VkPipeline p = VK_NULL_HANDLE;
    if (vkCreateComputePipelines(K.dev, VK_NULL_HANDLE, 1, &ci, NULL, &p) != VK_SUCCESS) return VK_NULL_HANDLE;
    return p;
}

int vkc_init(void) {
    if (K.ready) return !K.lost;
    if (!coli_vk_core(&K.core)) return 0;
    K.dev = (VkDevice)K.core.device; K.queue = (VkQueue)K.core.queue;
    K.align = K.core.ssbo_align > 16 ? K.core.ssbo_align : 16;
    VkDescriptorSetLayoutBinding b[VKC_BIND];
    for (int i = 0; i < VKC_BIND; i++) b[i] = (VkDescriptorSetLayoutBinding){.binding = (uint32_t)i,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT};
    VkDescriptorSetLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = VKC_BIND, .pBindings = b};
    VkPushConstantRange pr = {.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = 128};
    if (vkCreateDescriptorSetLayout(K.dev, &li, NULL, &K.dsl) != VK_SUCCESS) return 0;
    VkPipelineLayoutCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1, .pSetLayouts = &K.dsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &pr};
    if (vkCreatePipelineLayout(K.dev, &pi, NULL, &K.pl) != VK_SUCCESS) return 0;
    for (int i = 0; i < P_NPIPE; i++) {
        K.mod[i] = load_module(K.core.spv_path, pipe_file[i]);
        if (!K.mod[i] || !(K.pipe[i] = make_pipe(K.mod[i], NULL))) {
            fprintf(stderr, "[VK] chain: shader %s unavailable, the chain stays off\n", pipe_file[i]);
            return 0;
        }
    }
    K.mod_dnrec = load_module(K.core.spv_path, "chain_dnrec.spv");
    if (!K.mod_dnrec) return 0;
    /* the vectorized decode GEMV, its x staging as large as the device's shared memory
     * allows (16 KiB floats at most); COLI_VK_CHAIN_GEMV=0 keeps qmatmul.comp's */
    {
        const char *e = getenv("COLI_VK_CHAIN_GEMV");
        VkPhysicalDeviceProperties pp; vkGetPhysicalDeviceProperties((VkPhysicalDevice)K.core.phys, &pp);
        int xs = (int)(pp.limits.maxComputeSharedMemorySize / 16);
        if (xs > 4096) xs = 4096;
        if (!(e && *e == '0') && xs >= 256 && (K.mod_gemv4 = load_module(K.core.spv_path, "chain_gemv.spv"))) {
            int32_t v = xs;
            VkSpecializationMapEntry me = {0, 0, 4};
            VkSpecializationInfo si = {1, &me, 4, &v};
            if ((K.gemv4 = make_pipe(K.mod_gemv4, &si))) K.gemv4_xs = xs;
        }
    }
    /* the MLA, KDA and mHC shaders, optional: without one only its ops decline */
    for (int i = 0; i < PM_N; i++) {
        char path[1200];
        const char *sl = strrchr(K.core.spv_path, '/');
        size_t pre = sl ? (size_t)(sl - K.core.spv_path) + 1 : 0;
        FILE *f = NULL;
        if (pre + strlen(mla_file[i]) + 1 < sizeof path) {
            memcpy(path, K.core.spv_path, pre); strcpy(path + pre, mla_file[i]);
            f = fopen(path, "rb");
        }
        if (f) fclose(f);
        if (f && (K.mmod[i] = load_module(K.core.spv_path, mla_file[i]))) K.mpipe[i] = make_pipe(K.mmod[i], NULL);
    }
    K.mla_ok = K.mpipe[PM_MLA] && K.mpipe[PM_HGEMV] && K.mpipe[PM_DSA];
    /* the fp32 tiled GEMM at the backend's tiles; none = every S on the GEMV */
    K.mod_gemm = K.core.gemm_tiles ? load_module(K.core.spv_path, "qmatmul_gemm.spv") : VK_NULL_HANDLE;
    for (int k = 0; K.mod_gemm && k < K.core.gemm_tiles && k < VKC_GEMM_MAX; k++) {
        const int *t = K.core.gemm_tile[k];
        int32_t sv[7] = {t[0], t[1], t[2], t[3], t[4], (t[0] / t[3]) * (t[1] / t[4]), t[5] ? 1 : 0};
        VkSpecializationMapEntry me[7];
        for (int i = 0; i < 7; i++) me[i] = (VkSpecializationMapEntry){(uint32_t)i, (uint32_t)(i * 4), 4};
        VkSpecializationInfo si = {7, me, sizeof sv, sv};
        if (!(K.gemm[k] = make_pipe(K.mod_gemm, &si))) break;
        K.gemm_bm[k] = t[0]; K.gemm_bn[k] = t[1]; K.ngemm = k + 1;
    }
    VkCommandPoolCreateInfo cp = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = K.core.qfam};
    if (vkCreateCommandPool(K.dev, &cp, NULL, &K.cpool) != VK_SUCCESS) return 0;
    for (int i = 0; i < VKC_FRAMES; i++) {
        VkCommandBufferAllocateInfo ca = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = K.cpool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
        VkFenceCreateInfo fi = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        if (vkAllocateCommandBuffers(K.dev, &ca, &K.fr[i].cmd) != VK_SUCCESS ||
            vkCreateFence(K.dev, &fi, NULL, &K.fr[i].fence) != VK_SUCCESS) return 0;
    }
    {
        const char *e = getenv("COLI_VK_CHAIN_PROF");
        uint32_t nqf = 0; vkGetPhysicalDeviceQueueFamilyProperties((VkPhysicalDevice)K.core.phys, &nqf, NULL);
        VkQueueFamilyProperties qf[16]; if (nqf > 16) nqf = 16;
        vkGetPhysicalDeviceQueueFamilyProperties((VkPhysicalDevice)K.core.phys, &nqf, qf);
        VkPhysicalDeviceProperties pp; vkGetPhysicalDeviceProperties((VkPhysicalDevice)K.core.phys, &pp);
        uint32_t bits = K.core.qfam < nqf ? qf[K.core.qfam].timestampValidBits : 0;
        if (e && *e == '1' && bits && pp.limits.timestampPeriod > 0) {
            K.prof = 1; K.ts_period = pp.limits.timestampPeriod;
            K.ts_mask = bits >= 32 ? 0xffffffffu : (1u << bits) - 1;
            for (int i = 0; i < VKC_FRAMES; i++) {
                VkQueryPoolCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
                    .queryType = VK_QUERY_TYPE_TIMESTAMP, .queryCount = VKC_PROF_Q};
                if (vkCreateQueryPool(K.dev, &qi, NULL, &K.fr[i].qp) != VK_SUCCESS) K.prof = 0;
            }
        }
    }
    for (int k = 0; k < 3; k++) {
        K.pool[k].memtype = memtype_of(k);
        K.pool[k].mapped = memtype_host_visible(K.pool[k].memtype);
        vka_pool_init(&K.pool[k].p, VKC_BLOCK, 0);
    }
    if (!K.pool[VKC_UP].mapped || !K.pool[VKC_DOWN].mapped) return 0;
    K.ready = 1;
    K.dummy = vkc_buf(256, VKC_DEV);
    if (!K.dummy) { K.ready = 0; return 0; }
    return 1;
}

/* ---- frames ---------------------------------------------------------------------- */
static void frame_reclaim(VkcFrame *f) {
    for (int i = 0; i < f->ntmp; i++) buf_release(f->tmp[i]);
    f->ntmp = 0;
    f->stage_used = 0;
    for (int i = 0; i < f->npools; i++) vkResetDescriptorPool(K.dev, f->pools[i], 0);
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
        r = vkGetFenceStatus(K.dev, f->fence);
    if (r == VK_NOT_READY) r = vkWaitForFences(K.dev, 1, &f->fence, VK_TRUE, 20000000000ULL);
    K.st.wait_ms += vkc_now_ms() - t0; K.st.waits++;
    f->inflight = 0;
    if (r != VK_SUCCESS) { lose("fence wait", r); return 0; }
    if (K.prof && f->qp && f->nq > 1) {
        uint64_t ts[VKC_PROF_Q];
        if (vkGetQueryPoolResults(K.dev, f->qp, 0, (uint32_t)f->nq, sizeof ts, ts, sizeof ts[0],
                                  VK_QUERY_RESULT_64_BIT) == VK_SUCCESS)
            for (int i = 1; i < f->nq; i++) {
                uint64_t d = (ts[i] - ts[i - 1]) & (K.ts_mask == 0xffffffffu ? ~0ull : (uint64_t)K.ts_mask);
                K.prof_ms[f->kind[i]] += (double)d * K.ts_period / 1e6; K.prof_n[f->kind[i]]++;
            }
        f->nq = 0;
    }
    frame_reclaim(f);
    return 1;
}
static void prof_mark(VkcFrame *f, int kind) {
    if (!K.prof || !f->qp || f->nq >= VKC_PROF_Q) return;
    vkCmdWriteTimestamp(f->cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, f->qp, (uint32_t)f->nq);
    f->kind[f->nq++] = (unsigned char)kind;
}
int vkc_finish(void) {
    if (!K.ready) return 0;
    if (K.cur >= 0 && !vkc_submit(0)) return 0;
    for (int i = 0; i < VKC_FRAMES; i++) if (!frame_wait(&K.fr[i])) return 0;
    return !K.lost;
}

static void barrier_now(VkCommandBuffer c, VkPipelineStageFlags dst_stage, VkAccessFlags dst) {
    VkMemoryBarrier mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, .dstAccessMask = dst};
    vkCmdPipelineBarrier(c, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, dst_stage,
                         0, 1, &mb, 0, NULL, 0, NULL);
    K.st.barriers++;
}
static const VkAccessFlags ALL_RW = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                                    VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
#define ALL_STAGES (VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT)

int vkc_begin(void) {
    if (!vkc_ready()) return 0;
    if (K.cur >= 0) return 1;                     /* already open */
    int i = (int)(K.serial % VKC_FRAMES);
    VkcFrame *f = &K.fr[i];
    if (!frame_wait(f)) return 0;
    frame_reclaim(f);
    if (vkResetCommandBuffer(f->cmd, 0) != VK_SUCCESS) { lose("command buffer reset", VK_ERROR_DEVICE_LOST); return 0; }
    VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    vkBeginCommandBuffer(f->cmd, &bi);
    f->serial = ++K.serial;
    f->open = 1;
    K.cur = i;
    K.bound = VK_NULL_HANDLE;
    /* after everything submitted earlier on this queue (and the host's writes) */
    barrier_now(f->cmd, ALL_STAGES, ALL_RW);
    track_reset();
    if (K.prof && f->qp) {
        vkCmdResetQueryPool(f->cmd, f->qp, 0, VKC_PROF_Q);
        f->nq = 0;
        vkCmdWriteTimestamp(f->cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, f->qp, 0);
        f->kind[0] = 0; f->nq = 1;
    }
    return 1;
}
int vkc_submit(int wait) {
    if (K.cur < 0) return vkc_ready();
    VkcFrame *f = &K.fr[K.cur];
    barrier_now(f->cmd, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT);
    vkEndCommandBuffer(f->cmd);
    f->open = 0; K.cur = -1;
    if (K.lost) return 0;
    {   /* COLI_VK_CHAIN_FAULT=n (tests): the n-th submission fails as a lost device would */
        static long fault = -2;
        if (fault == -2) { const char *e = getenv("COLI_VK_CHAIN_FAULT"); fault = e && *e ? atol(e) : -1; }
        if (fault > 0 && (long)K.st.frames + 1 >= fault) { lose("queue submit (COLI_VK_CHAIN_FAULT)", VK_ERROR_DEVICE_LOST); return 0; }
    }
    vkResetFences(K.dev, 1, &f->fence);
    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &f->cmd};
    VkResult r = vkQueueSubmit(K.queue, 1, &si, f->fence);
    if (r != VK_SUCCESS) { lose("queue submit", r); return 0; }
    f->inflight = 1;
    K.st.frames++;
    return wait ? frame_wait(f) : 1;
}

/* ---- recording helpers ----------------------------------------------------------- */
static VkcFrame *open_frame(void) { return K.cur >= 0 ? &K.fr[K.cur] : NULL; }

static int in_set(const VkBuffer *s, int n, VkBuffer b) { for (int i = 0; i < n; i++) if (s[i] == b) return 1; return 0; }
/* a barrier before an op reading rd[] and writing wr[] when it conflicts with the ops
 * recorded since the last one */
static void hazard(VkcFrame *f, const VkBuffer *rd, int nr, const VkBuffer *wr, int nw) {
    int need = 0;
    for (int i = 0; i < nr && !need; i++) need = in_set(K.wr, K.nwr, rd[i]);
    for (int i = 0; i < nw && !need; i++) need = in_set(K.wr, K.nwr, wr[i]) || in_set(K.rd, K.nrd, wr[i]);
    if (need || K.nwr + nw > VKC_TRACK || K.nrd + nr > VKC_TRACK) {
        barrier_now(f->cmd, ALL_STAGES, ALL_RW);
        track_reset();
    }
    for (int i = 0; i < nr; i++) if (!in_set(K.rd, K.nrd, rd[i])) K.rd[K.nrd++] = rd[i];
    for (int i = 0; i < nw; i++) if (!in_set(K.wr, K.nwr, wr[i])) K.wr[K.nwr++] = wr[i];
}

static VkDescriptorSet alloc_set(VkcFrame *f) {
    for (;;) {
        if (f->cur_pool < f->npools) {
            VkDescriptorSetAllocateInfo da = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                .descriptorPool = f->pools[f->cur_pool], .descriptorSetCount = 1, .pSetLayouts = &K.dsl};
            VkDescriptorSet s;
            if (vkAllocateDescriptorSets(K.dev, &da, &s) == VK_SUCCESS) return s;
            f->cur_pool++;
            continue;
        }
        VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 512 * VKC_BIND};
        VkDescriptorPoolCreateInfo pc = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .maxSets = 512, .poolSizeCount = 1, .pPoolSizes = &ps};
        VkDescriptorPool *n = realloc(f->pools, (size_t)(f->npools + 1) * sizeof *n);
        if (!n) return VK_NULL_HANDLE;
        f->pools = n;
        if (vkCreateDescriptorPool(K.dev, &pc, NULL, &f->pools[f->npools]) != VK_SUCCESS) return VK_NULL_HANDLE;
        f->npools++;
    }
}

/* One dispatch. bufs[i] NULL: the dummy; offs/ranges in bytes (range 0 = to the end). */
typedef struct { VkBuffer buf; VkDeviceSize off, range; int written; } VkcBind;
static int record(VkPipeline pipe, const VkcBind *bd, int n, const void *pc, size_t pcb,
                  uint32_t gx, uint32_t gy, uint32_t gz) {
    VkcFrame *f = open_frame();
    if (!f || K.lost || !gx || !gy || !gz) return f && !K.lost;
    VkBuffer rd[VKC_BIND], wr[VKC_BIND]; int nr = 0, nw = 0;
    VkDescriptorBufferInfo bi[VKC_BIND];
    VkWriteDescriptorSet w[VKC_BIND];
    VkDescriptorSet set = alloc_set(f);
    if (!set) { fprintf(stderr, "[VK] chain: out of descriptor sets\n"); return 0; }
    for (int i = 0; i < VKC_BIND; i++) {
        if (i < n && bd[i].buf) {
            bi[i] = (VkDescriptorBufferInfo){bd[i].buf, bd[i].off, bd[i].range ? bd[i].range : VK_WHOLE_SIZE};
            if (bd[i].written) wr[nw++] = bd[i].buf; else rd[nr++] = bd[i].buf;
        } else bi[i] = (VkDescriptorBufferInfo){K.dummy->buf, 0, VK_WHOLE_SIZE};
        w[i] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set,
            .dstBinding = (uint32_t)i, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &bi[i]};
    }
    vkUpdateDescriptorSets(K.dev, VKC_BIND, w, 0, NULL);
    hazard(f, rd, nr, wr, nw);
    if (K.bound != pipe) { vkCmdBindPipeline(f->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe); K.bound = pipe; }
    vkCmdBindDescriptorSets(f->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, K.pl, 0, 1, &set, 0, NULL);
    if (pcb) vkCmdPushConstants(f->cmd, K.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, (uint32_t)pcb, pc);
    vkCmdDispatch(f->cmd, gx, gy, gz);
    K.st.ops++;
    prof_mark(f, K.kind);
    return 1;
}
static VkcBind B(VkcBuf *b, int written) {
    if (b) b->last_frame = K.serial;
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
    if (!f || K.lost) return 0;
    if (!bytes) return 1;
    if (doff + bytes > dst->bytes || soff + bytes > src->bytes) { fprintf(stderr, "[VK] chain: copy out of range\n"); return 0; }
    VkBuffer r = src->buf, w = dst->buf;
    hazard(f, &r, 1, &w, 1);
    VkBufferCopy c = {soff, doff, bytes};
    vkCmdCopyBuffer(f->cmd, src->buf, dst->buf, 1, &c);
    prof_mark(f, PK_COPY);
    src->last_frame = dst->last_frame = K.serial;
    return 1;
}
int vkc_copy(VkcBuf *dst, size_t doff, VkcBuf *src, size_t soff, size_t n) {
    return copy_bytes(dst, doff * 4, src, soff * 4, n * 4);
}
int vkc_copy_regions(VkcBuf *dst, VkcBuf *src, const VkcRegion *r, int n) {
    VkcFrame *f = open_frame();
    if (!f || K.lost) return 0;
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
    src->last_frame = dst->last_frame = K.serial;
    free(c);
    return 1;
}
int vkc_zero(VkcBuf *dst, size_t off, size_t n) {
    VkcFrame *f = open_frame();
    if (!f || K.lost) return 0;
    if (!n) return 1;
    if ((off + n) * 4 > dst->bytes) return 0;
    VkBuffer w = dst->buf;
    hazard(f, NULL, 0, &w, 1);
    vkCmdFillBuffer(f->cmd, dst->buf, off * 4, n * 4, 0);
    dst->last_frame = K.serial;
    return 1;
}
static void zero_now(VkcBuf *b) {
    /* a device-only buffer starts zeroed too: its own little frame */
    int was_open = K.cur >= 0;
    if (!was_open && !vkc_begin()) return;
    vkc_zero(b, 0, b->bytes / 4);
    if (!was_open) vkc_submit(1);
}
int vkc_write(VkcBuf *dst, size_t off, const void *src, size_t bytes) {
    VkcFrame *f = open_frame();
    if (!f || K.lost) return 0;
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
    K.st.bytes_up += bytes;
    return copy_bytes(dst, off * 4, f->stage, at, bytes);
}
int vkc_read(VkcBuf *src, size_t off, void *dst, size_t bytes) {
    if (!vkc_ready()) return 0;
    if (!bytes) return 1;
    if (!vkc_finish()) return 0;
    if (src->ptr && src->kind != VKC_UP) {        /* the device wrote it into host-readable memory */
        memcpy(dst, src->ptr + off * 4, bytes);
        K.st.bytes_down += bytes;
        return 1;
    }
    VkcBuf *rb = vkc_buf(bytes, VKC_DOWN);
    if (!rb) return 0;
    int ok = vkc_begin() && copy_bytes(rb, 0, src, off * 4, bytes) && vkc_submit(1);
    if (ok) { memcpy(dst, rb->ptr, bytes); K.st.bytes_down += bytes; }
    vkc_free(rb);
    return ok;
}

/* ---- matmul ---------------------------------------------------------------------- */
void vkc_gemm_rows(int rows) { K.gemm_rows = rows; }
struct VkcPC { int fmt, S, I, O, rowWords, gs; };
static int matmul_aligned(const ColiVkTensorInfo *ti, VkcBuf *x, size_t xb, VkcBuf *y, size_t yb, int S) {
    int path = -1;   /* GEMM slot, -1 the GEMV */
    if (K.ngemm && S >= 2) {
        int take = K.gemm_rows < 0 ? (K.core.gemm_min_s && S >= K.core.gemm_min_s &&
                                      (int64_t)S * ti->O >= K.core.gemm_min_so)
                                   : (K.gemm_rows > 0 && S >= K.gemm_rows);
        if (take) { path = 0; while (path + 1 < K.ngemm && S > K.gemm_bn[path]) path++; }
    }
    VkcBind bd[4] = {{x->buf, xb, 0, 0}, {(VkBuffer)ti->wbuf, 0, 0, 0}, {(VkBuffer)ti->sbuf, 0, 0, 0},
                     {y->buf, yb, 0, 1}};
    x->last_frame = y->last_frame = K.serial;
    struct VkcPC pc = {ti->fmt, S, ti->I, ti->O, ti->rowWords, ti->gs};
    int ok;
    /* the vectorized GEMV where the row is whole 16-byte steps and fits the staging */
    int per = ti->fmt == 1 ? 16 : ti->fmt == 4 ? 32 : ti->fmt == 11 ? 8 : ti->fmt == 10 ? 4 : 0;
    int v4 = path < 0 && K.gemv4 && per && ti->rowWords % 4 == 0 && ti->I % 4 == 0 &&
             (ti->fmt != 4 || (ti->gs > 0 && ti->gs % 32 == 0)) && (ti->rowWords / 4) * per / 4 <= K.gemv4_xs;
    K.kind = path >= 0 ? PK_GEMM : ti->fmt == 1 ? PK_GEMV8 : PK_GEMV;
    if (v4) {
        /* each workgroup stages x once: give it enough rows that the staging does not
         * rival the weights, while keeping enough workgroups to fill the device */
        /* lanes per row: about sixteen 16-byte steps each (COLI_VK_CHAIN_GEMV_LPR overrides) */
        const char *e = getenv("COLI_VK_CHAIN_GEMV_LPR");
        int nq = ti->rowWords / 4, lpr = 4;
        if (e && *e) lpr = atoi(e); else while (lpr < 64 && lpr * 16 < nq) lpr *= 2;
        if (lpr < 1) lpr = 1;
        struct { int fmt, S, I, O, rowWords, gs, lpr; } pc7 = {ti->fmt, S, ti->I, ti->O, ti->rowWords, ti->gs, lpr};
        int rows_wg = 256 / lpr;   /* at least, at subgroups of 64 */
        int wg = (ti->O + rows_wg - 1) / rows_wg; if (wg > 1024) wg = 1024;
        ok = record(K.gemv4, bd, 4, &pc7, sizeof pc7, (uint32_t)wg, (uint32_t)S, 1);
    }
    else if (path >= 0)
        ok = record(K.gemm[path], bd, 4, &pc, sizeof pc, (uint32_t)((ti->O + K.gemm_bm[path] - 1) / K.gemm_bm[path]),
                    (uint32_t)((S + K.gemm_bn[path] - 1) / K.gemm_bn[path]), 1);
    else ok = record(K.pipe[P_GEMV], bd, 4, &pc, sizeof pc, (uint32_t)((ti->O + 7) / 8), (uint32_t)S, 1);
    K.st.matmuls += ok; K.st.gemms += ok && path >= 0;
    return ok;
}
int vkc_matmul(ColiVkTensor *t, VkcBuf *x, size_t xo, VkcBuf *y, size_t yo, int S) {
    ColiVkTensorInfo ti;
    if (!open_frame() || K.lost || S < 1 || S > 65535 || !coli_vk_tensor_info(t, &ti)) return 0;
    size_t xb = xo * 4, yb = yo * 4, xn = (size_t)S * ti.I * 4, yn = (size_t)S * ti.O * 4;
    if (xb + xn > x->bytes || yb + yn > y->bytes) { fprintf(stderr, "[VK] chain: matmul out of range\n"); return 0; }
    if (xb % K.align == 0 && yb % K.align == 0) return matmul_aligned(&ti, x, xb, y, yb, S);
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
    K.kind = PK_NORM;
    VkcBind bd[3] = {B(x, x == y), B(w, 0), B(y, 1)};
    uint32_t gx, gy; grid((uint64_t)p->nseg, &gx, &gy);
    return record(K.pipe[P_NORM], bd, 3, p, sizeof *p, gx, gy, 1);
}
int vkc_rope(VkcBuf *x, VkcBuf *cs, const VkcRope *p) {
    K.kind = PK_ROPE;
    VkcBind bd[2] = {B(x, 1), B(cs, 0)};
    uint32_t gx, gy; grid(((uint64_t)p->nseg * p->half_ + 63) / 64, &gx, &gy);
    return record(K.pipe[P_ROPE], bd, 2, p, sizeof *p, gx, gy, 1);
}
int vkc_attn(VkcBuf *q, VkcBuf *kc, VkcBuf *vc, VkcBuf *o, VkcBuf *gate, VkcBuf *sel, const VkcAttn *p) {
    VkcAttnW w;
    memset(&w, 0, sizeof w);
    w.a = *p;
    return vkc_attn_w(q, kc, vc, o, gate, sel, NULL, &w);
}
int vkc_attn_w(VkcBuf *q, VkcBuf *kc, VkcBuf *vc, VkcBuf *o, VkcBuf *gate, VkcBuf *sel, VkcBuf *snk,
               const VkcAttnW *p) {
    K.kind = PK_ATTN;
    VkcAttnW w = *p;   /* the shader's push constants: VkcAttn's fields, then these */
    if (w.vd <= 0) w.vd = w.a.hd;
    if (w.a.hd > 256 || w.vd > 256 || w.a.H % w.a.KVH || w.win < 0 || w.ring < 0 || (w.sink && !snk)) return 0;
    VkcBind bd[7] = {B(q, 0), B(kc, 0), B(vc, 0), B(o, 1), B(gate, 0), B(sel, 0), B(snk, 0)};
    return record(K.pipe[P_ATTN], bd, 7, &w, sizeof w, (uint32_t)w.a.H, (uint32_t)w.a.S, 1);
}
int vkc_dnconv(VkcBuf *in, VkcBuf *w, VkcBuf *ring, VkcBuf *out, VkcBuf *snap, const VkcDnConv *p) {
    K.kind = PK_DNCONV;
    if (p->CK < 1 || p->CK > 9) return 0;
    VkcBind bd[5] = {B(in, 0), B(w, 0), B(ring, 1), B(out, 1), B(snap, snap != NULL && snap != ring)};
    if (snap == ring) bd[4].written = 0;   /* the same buffer: tracked once, as written */
    uint32_t gx, gy; grid(((uint64_t)p->CD + 63) / 64, &gx, &gy);
    return record(K.pipe[P_DNCONV], bd, 5, p, sizeof *p, gx, gy, 1);
}
static VkPipeline dnrec_pipe(int KD) {
    for (int i = 0; i < K.ndnrec; i++) if (K.dnrec_kd[i] == KD) return K.dnrec[i];
    if (K.ndnrec == VKC_DNREC_MAX) return VK_NULL_HANDLE;
    int32_t kd = KD;
    VkSpecializationMapEntry me = {0, 0, 4};
    VkSpecializationInfo si = {1, &me, 4, &kd};
    VkPipeline p = make_pipe(K.mod_dnrec, &si);
    if (!p) return VK_NULL_HANDLE;
    K.dnrec[K.ndnrec] = p; K.dnrec_kd[K.ndnrec++] = KD;
    return p;
}
int vkc_dnrec(int KD, VkcBuf *cv, VkcBuf *ab, VkcBuf *z, VkcBuf *st, VkcBuf *prm, VkcBuf *y, VkcBuf *snap, const VkcDnRec *p) {
    K.kind = PK_DNREC;
    if (p->VD > 128 || KD < 1 || KD > 256 || p->VH % p->KH) return 0;
    VkPipeline pipe = dnrec_pipe(KD);
    if (!pipe) return 0;
    VkcBind bd[7] = {B(cv, 0), B(ab, 0), B(z, 0), B(st, 1), B(prm, 0), B(y, 1), B(snap, snap != NULL && snap != st)};
    if (snap == st) bd[6].written = 0;
    return record(pipe, bd, 7, p, sizeof *p, (uint32_t)p->VH, 1, 1);
}
int vkc_ew(VkcBuf *y, VkcBuf *a, VkcBuf *b, VkcBuf *c, VkcBuf *e, const VkcEw *p) {
    K.kind = PK_EW;
    VkcBind bd[5] = {B(y, 1), B(a == y ? NULL : a, 0), B(b == y ? NULL : b, 0), B(c == y ? NULL : c, 0), B(e == y ? NULL : e, 0)};
    /* an input that is y itself reads through binding 0's alias: bind it there too */
    if (a == y) bd[1] = (VkcBind){y->buf, 0, 0, 0};
    if (b == y) bd[2] = (VkcBind){y->buf, 0, 0, 0};
    if (c == y) bd[3] = (VkcBind){y->buf, 0, 0, 0};
    if (e == y) bd[4] = (VkcBind){y->buf, 0, 0, 0};
    uint32_t gx, gy; grid(((uint64_t)p->n + 255) / 256, &gx, &gy);
    return record(K.pipe[P_EW], bd, 5, p, sizeof *p, gx, gy, 1);
}
int vkc_qsa(VkcBuf *src, VkcBuf *w, VkcBuf *pk, VkcBuf *cs, VkcBuf *sc, VkcBuf *sel, const VkcQsa *p) {
    K.kind = PK_QSA;
    if (p->ID > 256) return 0;
    if (p->mode == 0) {
        VkcBind bd[4] = {B(src, 0), B(w, 0), B(pk, 1), B(cs, 0)};
        return record(K.pipe[P_QSA], bd, 4, p, sizeof *p, (uint32_t)p->nb, 1, 1);
    }
    VkcBind bd[6] = {B(src, 0), {VK_NULL_HANDLE, 0, 0, 0}, B(pk, 0), {VK_NULL_HANDLE, 0, 0, 0}, B(sc, 1), B(sel, 1)};
    return record(K.pipe[P_QSA], bd, 6, p, sizeof *p, (uint32_t)p->S, 1, 1);
}
int vkc_ple(VkcBuf *keys, VkcBuf *hyp, VkcBuf *val, VkcBuf *prm, VkcBuf *gated, VkcBuf *normv,
            VkcBuf *conv, VkcBuf *ring, const VkcPle *p) {
    K.kind = PK_PLE;
    if (p->mode == 0) {
        VkcBind bd[6] = {B(keys, 0), B(hyp, 0), B(val, 0), B(prm, 0), B(gated, 1), B(normv, 1)};
        return record(K.pipe[P_PLE], bd, 6, p, sizeof *p, (uint32_t)(p->S * p->C), 1, 1);
    }
    if ((p->CK - 1) * p->NG > 32) return 0;
    VkcBind bd[8] = {{VK_NULL_HANDLE, 0, 0, 0}, B(hyp, 1), {VK_NULL_HANDLE, 0, 0, 0}, {VK_NULL_HANDLE, 0, 0, 0},
                     B(gated, 0), B(normv, 0), B(conv, 0), B(ring, 1)};
    uint32_t gx, gy; grid(((uint64_t)p->C * p->H + 255) / 256, &gx, &gy);
    return record(K.pipe[P_PLE], bd, 8, p, sizeof *p, gx, gy, 1);
}

/* ---- multi-head latent attention ------------------------------------------------- */
int vkc_mla_ready(void) { return vkc_ready() && K.mla_ok; }

int vkc_mla_core(VkcBuf *qabs, VkcBuf *qr, VkcBuf *lat, VkcBuf *rope, VkcBuf *sel, VkcBuf *clat, const VkcMlaCore *p) {
    K.kind = PK_MLA;
    if (!K.mla_ok || p->S < 1 || p->H < 1 || p->K < 1 || p->K > 1024 || p->R < 0 || p->R > 128 ||
        (p->R > 0 && !rope) || p->pos_base < 0 || p->kv_start < 0 || p->kv_start > p->pos_base) return 0;
    struct { int mode; VkcMlaCore b; } pc = {0, *p};
    if (!sel) pc.b.sel_row = 0;
    VkcBind bd[6] = {B(qabs, 0), B(qr, 0), B(lat, 0), B(p->R > 0 ? rope : NULL, 0), B(pc.b.sel_row > 0 ? sel : NULL, 0),
                     B(clat, 1)};
    return record(K.mpipe[PM_MLA], bd, 6, &pc, sizeof pc, (uint32_t)p->H, (uint32_t)p->S, 1);
}
static int mla_rowop(int mode, VkcBuf *x, VkcBuf *aux, VkcBuf *y, const VkcMlaRow *p) {
    if (!K.mla_ok || !open_frame() || K.lost || p->nseg < 0) return 0;
    if (p->nseg == 0) return 1;
    if (mode == 1 && (p->rd < 2 || p->rd > 256 || (p->rd & 1) || p->rd > p->seg_len)) return 0;
    if (mode == 2 && p->seg_len < 1) return 0;
    struct { int mode; VkcMlaRow b; int copy_rest; } pc = {mode, *p, x != y};
    VkcBind bd[6] = {B(x, x == y), B(aux, 0), B(NULL, 0), B(NULL, 0), B(NULL, 0), B(y, 1)};
    uint32_t gx, gy; grid((uint64_t)p->nseg, &gx, &gy);
    return record(K.mpipe[PM_MLA], bd, 6, &pc, sizeof pc, gx, gy, 1);
}
int vkc_mla_rope(VkcBuf *x, VkcBuf *cs, VkcBuf *y, const VkcMlaRow *p) { K.kind = PK_ROPE; return mla_rowop(1, x, cs, y, p); }
int vkc_mla_lnorm(VkcBuf *x, VkcBuf *prm, VkcBuf *y, const VkcMlaRow *p) { K.kind = PK_NORM; return mla_rowop(2, x, prm, y, p); }

int vkc_mla_hgemv(ColiVkTensor *t, VkcBuf *x, VkcBuf *y, VkcBuf *gate, const VkcHgemv *p) {
    K.kind = PK_MLAW;
    ColiVkTensorInfo ti;
    if (!K.mla_ok || !open_frame() || K.lost || !coli_vk_tensor_info(t, &ti) || p->S < 1 || p->S > 65535 ||
        p->H < 1 || p->H > 65535 || p->n < 1 || (p->trans && p->n > 1024)) return 0;
    int f = ti.fmt;
    if (!(f == 1 || f == 2 || f == 4 || f == 5 || f == 7 || f == 10 || f == 11 || f == 12 || f == 13)) return 0;
    if ((f == 4 || f == 7 || f == 12 || f == 13) && ti.gs < 1) return 0;
    if ((int64_t)(p->H - 1) * p->hstride + p->hoff + p->n > (int64_t)ti.O) return 0;   /* the rows exist */
    struct { int mode, fmt, I, rowWords, gs, S, H, n, hstride, hoff, x_off, x_row, x_seg, y_off, y_row, y_seg,
             g_off, g_row, has_gate; } pc = {p->trans, f, ti.I, ti.rowWords, ti.gs, p->S, p->H, p->n, p->hstride, p->hoff,
             p->x_off, p->x_row, p->x_seg, p->y_off, p->y_row, p->y_seg, p->g_off, p->g_row, gate && p->has_gate};
    VkcBind bd[5] = {B(x, 0), {(VkBuffer)ti.wbuf, 0, 0, 0}, {(VkBuffer)ti.sbuf, 0, 0, 0}, B(y, 1), B(pc.has_gate ? gate : NULL, 0)};
    if (p->trans) return record(K.mpipe[PM_HGEMV], bd, 5, &pc, sizeof pc, (uint32_t)p->H, (uint32_t)p->S, 1);
    /* rows: a subgroup each, grid-stride (at least four subgroups a workgroup) */
    int64_t rows = (int64_t)p->H * p->n, wg = (rows + 7) / 8;
    if (wg > 4096) wg = 4096;
    return record(K.mpipe[PM_HGEMV], bd, 5, &pc, sizeof pc, (uint32_t)wg, (uint32_t)p->S, 1);
}

int vkc_dsa_select(VkcBuf *iq, VkcBuf *hw, VkcBuf *keys, VkcBuf *sc, VkcBuf *sel, const VkcDsa *p) {
    K.kind = PK_DSA;
    if (!K.mla_ok || p->S < 1 || p->S > 65535 || p->IH < 1 || p->IH > 64 || p->ID < 1 || p->IH * p->ID > 4096 ||
        p->topk < 1 || p->pos_base < 0 || p->sc_row < p->pos_base + p->S ||
        p->sel_row < 1 + (p->topk < p->pos_base + p->S ? p->topk : p->pos_base + p->S)) return 0;
    struct { int mode; VkcDsa b; } pc = {0, *p};
    VkcBind bd[5] = {B(iq, 0), B(hw, 0), B(keys, 0), B(sc, 1), B(sel, 1)};
    return record(K.mpipe[PM_DSA], bd, 5, &pc, sizeof pc, (uint32_t)p->S, 1, 1);
}

/* k-pooling (chain_dsa.comp modes 1 and 2) */
int vkc_dsa_pool_keys(VkcBuf *keys, VkcBuf *gates, VkcBuf *prm, VkcBuf *pk, const VkcDsaPool *p) {
    K.kind = PK_DSA;
    if (!K.mla_ok || p->np < 0 || p->pool < 1 || p->p0 < 0 || p->ID < 1) return 0;
    if (p->np == 0) return open_frame() && !K.lost;
    struct { int mode; VkcDsaPool b; } pc = {1, *p};
    VkcBind bd[8] = {B(NULL, 0), B(NULL, 0), B(keys, 0), B(NULL, 0), B(NULL, 0), B(pk, 1), B(gates, 0), B(prm, 0)};
    return record(K.mpipe[PM_DSA], bd, 8, &pc, sizeof pc, (uint32_t)p->np, 1, 1);
}
int vkc_dsa_pool_select(VkcBuf *iq, VkcBuf *hw, VkcBuf *pk, VkcBuf *sc, VkcBuf *sel, const VkcDsaPick *p) {
    K.kind = PK_DSA;
    if (!K.mla_ok || p->S < 1 || p->S > 65535 || p->IH < 1 || p->IH > 64 || p->ID < 1 || p->IH * p->ID > 4096 ||
        p->pool < 1 || p->topk < p->pool || p->topk % p->pool || p->topk / p->pool > 1024 || p->pos_base < 0 ||
        p->sc_row < (p->pos_base + p->S) / p->pool || p->sel_row < 1 + p->topk + (p->tail ? p->pool - 1 : 0)) return 0;
    struct { int mode; VkcDsaPick b; } pc = {2, *p};
    VkcBind bd[6] = {B(iq, 0), B(hw, 0), B(NULL, 0), B(sc, 1), B(sel, 1), B(pk, 0)};
    return record(K.mpipe[PM_DSA], bd, 6, &pc, sizeof pc, (uint32_t)p->S, 1, 1);
}

/* ---- Kimi Delta Attention (chain_kda.comp) ------------------------------------------ */
int vkc_kda_ready(void) { return vkc_ready() && K.mpipe[PM_KDA]; }
int vkc_kda_conv(VkcBuf *in, VkcBuf *w, VkcBuf *win, VkcBuf *out, const VkcKdaConv *p) {
    K.kind = PK_KDA;
    if (!K.mpipe[PM_KDA] || p->K < 1 || p->K > 8 || p->P < 1 || p->C < 1 || p->S < 1) return 0;
    struct { int mode; VkcKdaConv b; } pc = {0, *p};
    VkcBind bd[4] = {B(in, 0), B(w, 0), B(win, 1), B(out, 1)};
    uint32_t gx, gy; grid(((uint64_t)p->C + 127) / 128, &gx, &gy);
    return record(K.mpipe[PM_KDA], bd, 4, &pc, sizeof pc, gx, gy, 1);
}
static VkPipeline kda_pipe(int KD) {
    for (int i = 0; i < K.nkdarec; i++) if (K.kdarec_kd[i] == KD) return K.kdarec[i];
    if (K.nkdarec == VKC_KDA_MAX || !K.mmod[PM_KDA]) return VK_NULL_HANDLE;
    int32_t kd = KD;
    VkSpecializationMapEntry me = {0, 0, 4};
    VkSpecializationInfo si = {1, &me, 4, &kd};
    VkPipeline p = make_pipe(K.mmod[PM_KDA], &si);
    if (!p) return VK_NULL_HANDLE;
    K.kdarec[K.nkdarec] = p; K.kdarec_kd[K.nkdarec++] = KD;
    return p;
}
int vkc_kda_rec(int KD, VkcBuf *m, VkcBuf *f, VkcBuf *b, VkcBuf *g, VkcBuf *prm, VkcBuf *st, VkcBuf *y, const VkcKdaRec *p) {
    return vkc_kda_rec_flags(KD, m, f, b, g, prm, st, y, p, 0);
}
int vkc_kda_rec_flags(int KD, VkcBuf *m, VkcBuf *f, VkcBuf *b, VkcBuf *g, VkcBuf *prm, VkcBuf *st, VkcBuf *y,
                      const VkcKdaRec *p, int flags) {
    K.kind = PK_KDA;
    if (!K.mpipe[PM_KDA] || KD < 1 || KD > 256 || p->VD < 1 || p->VD > 128 || p->H < 1 || p->S < 1 ||
        (flags & ~(VKC_KDA_EXP_A | VKC_KDA_K3))) return 0;
    VkPipeline pipe = kda_pipe(KD);
    if (!pipe) return 0;
    struct { int mode; VkcKdaRec b; int flags; } pc = {1, *p, flags};
    VkcBind bd[7] = {B(m, 0), B(prm, 0), B(st, 1), B(y, 1), B(f, 0), B(b, 0), B(g, 0)};
    return record(pipe, bd, 7, &pc, sizeof pc, (uint32_t)p->H, 1, 1);
}

/* ---- manifold-constrained hyper-connections (chain_mhc.comp) ------------------------ */
int vkc_mhc_ready(void) { return vkc_ready() && K.mpipe[PM_MHC]; }
int vkc_mhc(int mode, VkcBuf *x, VkcBuf *m, VkcBuf *hp, VkcBuf *prm, VkcBuf *y, const VkcMhc *p) {
    K.kind = PK_MHC;
    if (!K.mpipe[PM_MHC] || mode < 0 || mode > 4 || p->S < 0 || (mode != 4 && (p->H < 1 || p->H > 8 || p->D < 1))) return 0;
    if (mode == 0 && p->iters < 1) return 0;
    struct { int mode; VkcMhc b; } pc = {mode, *p};
    VkcBind bd[5] = {B(x, 0), B(m, 0), B(hp, mode == 0), B(prm, 0), B(y, mode != 0)};
    uint64_t n = mode == 4 ? (uint64_t)p->n : mode == 2 ? (uint64_t)p->S * p->H * p->D : (uint64_t)p->S * p->D;
    if (mode == 0) return p->S == 0 || record(K.mpipe[PM_MHC], bd, 5, &pc, sizeof pc, (uint32_t)p->S, 1, 1);
    uint32_t gx, gy; grid((n + 255) / 256, &gx, &gy);
    return n == 0 || record(K.mpipe[PM_MHC], bd, 5, &pc, sizeof pc, gx, gy, 1);
}

/* ---- attention residuals and SiTU-GLU (chain_ares.comp) ---------------------------- */
int vkc_ares_ready(void) { return vkc_ready() && K.mpipe[PM_ARES]; }
int vkc_ares_mix(VkcBuf *x, VkcBuf *blk, VkcBuf *prm, VkcBuf *y, const VkcAres *p) {
    K.kind = PK_ARES;
    if (!K.mpipe[PM_ARES] || p->S < 0 || p->D < 1 || p->nb < 0 || p->nb > 15 || (p->nb > 0 && !blk) || y == x || y == blk) return 0;
    if (p->S == 0) return open_frame() && !K.lost;
    struct { int mode; VkcAres b; } pc = {0, *p};
    VkcBind bd[4] = {B(x, 0), B(p->nb > 0 ? blk : NULL, 0), B(prm, 0), B(y, 1)};
    uint32_t gx, gy; grid((uint64_t)p->S, &gx, &gy);
    return record(K.mpipe[PM_ARES], bd, 4, &pc, sizeof pc, gx, gy, 1);
}
int vkc_situ(VkcBuf *g, VkcBuf *u, VkcBuf *y, const VkcSitu *p) {
    K.kind = PK_ARES;
    if (!K.mpipe[PM_ARES] || p->n < 0) return 0;
    if (p->n == 0) return open_frame() && !K.lost;
    int32_t pc[13] = {1, p->n, 0, 0, p->g_off, 0, p->u_off, 0, 0, p->y_off, 0, 0, 0};
    memcpy(&pc[11], &p->b1, 4); memcpy(&pc[12], &p->b2, 4);
    VkcBind bd[4] = {B(g, 0), B(u, 0), B(NULL, 0), B(y, 1)};
    uint32_t gx, gy; grid(((uint64_t)p->n + 255) / 256, &gx, &gy);
    return record(K.mpipe[PM_ARES], bd, 4, pc, sizeof pc, gx, gy, 1);
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
    return t && coli_vk_tensor_info(t, &ti) && ti.I == I && ti.O == O;
}
int vkc_mla_qkv(const VkcMla *m, VkcMlaScratch *s, VkcBuf *x, size_t x_off, int S, int pos_base,
                VkcBuf *cs, VkcMlaCache *c, VkcBuf *down, size_t down_off) {
    int H = m->H, QR = m->Q + m->R, KR = m->K + m->R;
    if (!K.mla_ok || S < 1 || S > s->rows || pos_base < 0 || (int64_t)pos_base + S > c->cap || m->R < 0 ||
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
    if (!K.mla_ok || S < 1 || S > s->rows || (int64_t)pos_base + S > c->cap) return 0;
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
static struct { VkShaderModule mod[2]; VkPipeline pipe[2]; int tried[2]; } KX;
static VkPipeline kx_pipe(int k) {
    static const char *const file[2] = {"chain_sconv.spv", "chain_relattn.spv"};
    if (KX.pipe[k] || KX.tried[k] || !vkc_ready()) return KX.pipe[k];
    KX.tried[k] = 1;
    if ((KX.mod[k] = load_module(K.core.spv_path, file[k]))) KX.pipe[k] = make_pipe(KX.mod[k], NULL);
    return KX.pipe[k];
}
static void kx_shutdown(void) {
    for (int k = 0; k < 2; k++) {
        if (KX.pipe[k]) vkDestroyPipeline(K.dev, KX.pipe[k], NULL);
        if (KX.mod[k]) vkDestroyShaderModule(K.dev, KX.mod[k], NULL);
    }
    memset(&KX, 0, sizeof KX);
}
int vkc_sconv_ready(void) { return kx_pipe(0) != VK_NULL_HANDLE; }
int vkc_relattn_ready(void) { return kx_pipe(1) != VK_NULL_HANDLE; }
int vkc_sconv(VkcBuf *x, VkcBuf *w, VkcBuf *ring, const VkcSconv *p) {
    VkPipeline pipe = kx_pipe(0);
    if (!pipe || (p->mode == 0 && (p->CK < 1 || p->CK > 9))) return 0;
    K.kind = p->mode == 0 ? PK_DNCONV : PK_EW;
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
    K.kind = PK_ATTN;
    VkcBind bd[8] = {B(q, 0), B(kc, 0), B(vc, 0), B(o, 1), B(kvs, 0), B(r, 0), B(relp, 0), B(tau, 0)};
    return record(pipe, bd, 8, p, sizeof *p, (uint32_t)p->H, (uint32_t)p->S, 1);
}

void vkc_stats(VkcStats *st) { *st = K.st; }
void vkc_prof_print(void) {
    if (!K.prof) return;
    double tot = 0; for (int k = 0; k < PK_N; k++) tot += K.prof_ms[k];
    fprintf(stderr, "[VK] chain profile: %.1f ms of device time", tot);
    for (int k = 0; k < PK_N; k++)
        if (K.prof_n[k]) fprintf(stderr, " | %s %.1f ms (%llu)", pk_name[k], K.prof_ms[k], K.prof_n[k]);
    fprintf(stderr, "\n");
}

void vkc_shutdown(void) {
    if (!K.ready) return;
    if (!K.lost && coli_vk_available()) vkc_finish();
    else vkDeviceWaitIdle(K.dev);
    for (int i = 0; i < VKC_FRAMES; i++) {
        VkcFrame *f = &K.fr[i];
        frame_reclaim(f);
        if (f->stage) buf_release(f->stage);
        free(f->tmp);
        for (int k = 0; k < f->npools; k++) vkDestroyDescriptorPool(K.dev, f->pools[k], NULL);
        free(f->pools);
        if (f->fence) vkDestroyFence(K.dev, f->fence, NULL);
        if (f->qp) vkDestroyQueryPool(K.dev, f->qp, NULL);
    }
    if (K.dummy) buf_release(K.dummy);
    for (int k = 0; k < 3; k++) {
        VkaPool *P = &K.pool[k].p;
        for (int b = 0; b < P->nb; b++) {
            if (!P->b[b].present) continue;
            VkcBlock *e = P->b[b].user;
            if (e) { if (e->map) vkUnmapMemory(K.dev, e->mem); vkFreeMemory(K.dev, e->mem, NULL); free(e); }
        }
        vka_pool_destroy(P);
    }
    if (K.cpool) vkDestroyCommandPool(K.dev, K.cpool, NULL);
    kx_shutdown();
    for (int i = 0; i < P_NPIPE; i++) {
        if (K.pipe[i]) vkDestroyPipeline(K.dev, K.pipe[i], NULL);
        if (K.mod[i]) vkDestroyShaderModule(K.dev, K.mod[i], NULL);
    }
    for (int i = 0; i < K.nkdarec; i++) vkDestroyPipeline(K.dev, K.kdarec[i], NULL);
    for (int i = 0; i < PM_N; i++) {
        if (K.mpipe[i]) vkDestroyPipeline(K.dev, K.mpipe[i], NULL);
        if (K.mmod[i]) vkDestroyShaderModule(K.dev, K.mmod[i], NULL);
    }
    for (int i = 0; i < K.ngemm; i++) vkDestroyPipeline(K.dev, K.gemm[i], NULL);
    for (int i = 0; i < K.ndnrec; i++) vkDestroyPipeline(K.dev, K.dnrec[i], NULL);
    if (K.gemv4) vkDestroyPipeline(K.dev, K.gemv4, NULL);
    if (K.mod_gemv4) vkDestroyShaderModule(K.dev, K.mod_gemv4, NULL);
    if (K.mod_gemm) vkDestroyShaderModule(K.dev, K.mod_gemm, NULL);
    if (K.mod_dnrec) vkDestroyShaderModule(K.dev, K.mod_dnrec, NULL);
    if (K.pl) vkDestroyPipelineLayout(K.dev, K.pl, NULL);
    if (K.dsl) vkDestroyDescriptorSetLayout(K.dev, K.dsl, NULL);
    memset(&K, 0, sizeof K);
    K.cur = -1; K.gemm_rows = -1;
}
