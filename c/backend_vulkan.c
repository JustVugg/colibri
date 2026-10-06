// Vulkan compute backend for colibri's quantized matmul, targeting the
// Strix Halo iGPU (RADV gfx1151). Mirrors backend_cuda.c's contract but
// exploits unified memory: weight "uploads" write into HOST_VISIBLE|
// DEVICE_LOCAL memory — the same physical RAM the iGPU reads — so there is
// no PCIe copy. That is what makes offloading *streamed experts* profitable
// here, which the discrete-CUDA path deliberately avoids.
//
// M2 scope: correctness + a standalone GPU-vs-CPU test harness. Synchronous
// submit/wait per call; async queues and zero-copy import come in M4.
#include "backend_vulkan.h"
#include "vk_alloc.h"
#include <vulkan/vulkan.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#ifdef __linux__
#include <unistd.h>
#endif
static double vk_now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec*1000.0 + t.tv_nsec/1e6; }

#define VKCHECK(x, what) do { VkResult _r = (x); if (_r != VK_SUCCESS) { \
    fprintf(stderr, "[VK] %s failed: %d\n", what, _r); return 0; } } while (0)

/* ---- device memory books, and COLI_VK_DEVICE_CAP_MB (docs/vulkan.md, "A partial chain") --
 * Every device memory allocation of the process goes through vk_mem_alloc / vk_mem_free
 * (the macros below rename the Vulkan calls in this file; vk_chain.c reaches them through
 * coli_vk_mem_alloc / coli_vk_mem_free): the bytes in a device-local heap are counted per
 * device (0 = G, 1 = G2). COLI_VK_DEVICE_CAP_MB=n (tests; a fraction is taken) makes a
 * device hold at most n MiB of them: an allocation past it fails as
 * VK_ERROR_OUT_OF_DEVICE_MEMORY, as a full card's does, and the budget the engines read
 * (coli_vk_mem_budget, _budget2, coli_vk_free_bytes, coli_vk_device_local_bytes, the
 * expert batch's heap room) is the cap and those bytes. So Lavapipe behaves like a small
 * card. Host memory imported in place (coli_vk_tensor_import) is the host's and is not
 * counted. Under the cap the pools take small blocks (coli_vk_block_bytes) so that their
 * granularity does not decide what fits. */
typedef struct { uint64_t key, bytes; int dev; } VkMemRec;
static struct {
    pthread_mutex_t mx;
    VkDevice dev[2];
    uint32_t dl_types[2];          /* memory types whose heap is device-local */
    uint64_t cap[2], used[2], peak[2];
    unsigned long long refused[2];
    VkMemRec *rec; size_t nrec, crec;
} g_mem = {.mx = PTHREAD_MUTEX_INITIALIZER};
static uint64_t vk_mem_key(VkDeviceMemory m) { uint64_t k = 0; memcpy(&k, &m, sizeof m); return k; }
static int vk_mem_dev(VkDevice d) { return d && d == g_mem.dev[1] ? 1 : 0; }
static int vk_mem_imported(const VkMemoryAllocateInfo *ai) {
    for (const VkBaseInStructure *p = ai->pNext; p; p = p->pNext)
        if (p->sType == VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT) return 1;
    return 0;
}
static VkResult vk_mem_alloc(VkDevice d, const VkMemoryAllocateInfo *ai, const VkAllocationCallbacks *cb,
                             VkDeviceMemory *m) {
    int k = vk_mem_dev(d);
    int counted = ai->memoryTypeIndex < 32 && (g_mem.dl_types[k] >> ai->memoryTypeIndex & 1u) && !vk_mem_imported(ai);
    uint64_t n = ai->allocationSize;
    if (counted) {
        pthread_mutex_lock(&g_mem.mx);
        if (g_mem.cap[k] && g_mem.used[k] + n > g_mem.cap[k]) {
            g_mem.refused[k]++;
            pthread_mutex_unlock(&g_mem.mx);
            return VK_ERROR_OUT_OF_DEVICE_MEMORY;
        }
        g_mem.used[k] += n;
        if (g_mem.used[k] > g_mem.peak[k]) g_mem.peak[k] = g_mem.used[k];
        pthread_mutex_unlock(&g_mem.mx);
    }
    VkResult r = vkAllocateMemory(d, ai, cb, m);
    if (!counted) return r;
    pthread_mutex_lock(&g_mem.mx);
    if (r != VK_SUCCESS) g_mem.used[k] -= n;
    else {
        if (g_mem.nrec == g_mem.crec) {
            size_t c = g_mem.crec ? 2 * g_mem.crec : 256;
            VkMemRec *nr = realloc(g_mem.rec, c * sizeof *nr);
            if (nr) { g_mem.rec = nr; g_mem.crec = c; }
        }
        if (g_mem.nrec < g_mem.crec) g_mem.rec[g_mem.nrec++] = (VkMemRec){vk_mem_key(*m), n, k};
    }
    pthread_mutex_unlock(&g_mem.mx);
    return r;
}
static void vk_mem_free(VkDevice d, VkDeviceMemory m, const VkAllocationCallbacks *cb) {
    if (m != VK_NULL_HANDLE) {
        uint64_t key = vk_mem_key(m);
        int k = vk_mem_dev(d);
        pthread_mutex_lock(&g_mem.mx);
        for (size_t i = g_mem.nrec; i-- > 0;)
            if (g_mem.rec[i].key == key && g_mem.rec[i].dev == k) {
                g_mem.used[k] -= g_mem.rec[i].bytes;
                g_mem.rec[i] = g_mem.rec[--g_mem.nrec];
                break;
            }
        pthread_mutex_unlock(&g_mem.mx);
    }
    vkFreeMemory(d, m, cb);
}
#define vkAllocateMemory vk_mem_alloc
#define vkFreeMemory vk_mem_free
/* A device just created: which of its memory types count, and the cap. */
static void vk_mem_device(int k, VkPhysicalDevice phys, VkDevice d) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    uint32_t mask = 0;
    for (uint32_t i = 0; i < mp.memoryTypeCount && i < 32; i++)
        if (mp.memoryHeaps[mp.memoryTypes[i].heapIndex].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) mask |= 1u << i;
    const char *e = getenv("COLI_VK_DEVICE_CAP_MB");
    double mb = e && *e ? atof(e) : 0.0;
    pthread_mutex_lock(&g_mem.mx);
    g_mem.dev[k] = d; g_mem.dl_types[k] = mask;
    g_mem.cap[k] = mb > 0 ? (uint64_t)(mb * 1048576.0) : 0;
    g_mem.used[k] = g_mem.peak[k] = 0; g_mem.refused[k] = 0;
    pthread_mutex_unlock(&g_mem.mx);
    if (g_mem.cap[k])
        fprintf(stderr, "[VK] COLI_VK_DEVICE_CAP_MB=%s: %s takes at most %.3f MiB of device-local memory\n",
                e, k ? "the second device" : "the device", (double)g_mem.cap[k] / 1048576.0);
}
/* The device is gone (shutdown): its books go with it. */
static void vk_mem_forget(int k) {
    pthread_mutex_lock(&g_mem.mx);
    for (size_t i = g_mem.nrec; i-- > 0;)
        if (g_mem.rec[i].dev == k) g_mem.rec[i] = g_mem.rec[--g_mem.nrec];
    g_mem.dev[k] = VK_NULL_HANDLE; g_mem.dl_types[k] = 0; g_mem.used[k] = 0;
    pthread_mutex_unlock(&g_mem.mx);
}
static uint64_t vk_mem_room(int k) {   /* lock-free read is enough for a budget */
    uint64_t c = g_mem.cap[k], u = __atomic_load_n(&g_mem.used[k], __ATOMIC_RELAXED);
    return c > u ? c - u : 0;
}
static void vk_pool_blocks(int k);   /* the weight pools' block size, below */
size_t coli_vk_device_cap(void) { return (size_t)g_mem.cap[0]; }
size_t coli_vk_device_used(void) { return (size_t)__atomic_load_n(&g_mem.used[0], __ATOMIC_RELAXED); }
size_t coli_vk_block_bytes_dev(int dev, size_t def) {
    int k = dev == 1 ? 1 : 0;
    if (!g_mem.cap[k]) return def;
    size_t b = (size_t)64 << 10;   /* at most about 4096 blocks fill the cap */
    while ((uint64_t)b < g_mem.cap[k] / 4096) b <<= 1;
    return b < def ? b : def;
}
size_t coli_vk_block_bytes(size_t def) { return coli_vk_block_bytes_dev(0, def); }
int coli_vk_mem_alloc(void *device, const void *info, void *memory) {
    return (int)vk_mem_alloc((VkDevice)device, (const VkMemoryAllocateInfo *)info, NULL, (VkDeviceMemory *)memory);
}
void coli_vk_mem_free(void *device, const void *memory) {
    vk_mem_free((VkDevice)device, *(const VkDeviceMemory *)memory, NULL);
}

typedef struct VkWPool VkWPool;
struct ColiVkTensor {
    VkBuffer wbuf, sbuf;
    size_t wbytes;
    int fmt, I, O, rowWords, gs;
    int dev;               /* 0 = primary device, 1 = the COLI_VK_DEV2 expert-tier device */
    VkWPool *pool;         /* the weight pool its two ranges came from */
    VkaRange wr, sr;       /* where the rows and the scales sit in the pool's blocks */
    struct ColiVkTensor *next_free;   /* deferred-free list (a free while async work is in flight) */
    uint8_t *img;          /* staged uploads: the host image a tier tensor is filled in, until committed */
    uint8_t img_al;        /* img came from the aligned allocator (up_img_alloc) */
    uint8_t *wmap, *smap;  /* mapped memory: the rows' and the scales' mapping (coli_vk_tensor_refill) */
    VkDeviceMemory imp;    /* coli_vk_tensor_import: the host memory its rows are read from, in place */
    struct ColiVkTensor *imp_next;   /* the live imports, freed at shutdown */
};

/* ---- memory placement without Resizable BAR (docs/vulkan.md) ------------------------
 * Resident data (the weights, the expert tier, the MLA KV mirror) is written by the host
 * once and read by the device for the rest of the run. The mapped path puts it in the
 * HOST_VISIBLE|DEVICE_LOCAL type and writes it through the mapping: on an integrated GPU,
 * a CPU device or a discrete card with Resizable BAR that type covers the device's
 * memory. Without Resizable BAR a discrete card exposes it as a window of about 256 MB:
 * NVIDIA refuses allocations past it, RADV puts them in system RAM, read over PCIe.
 * Staged uploads put resident data in a DEVICE_LOCAL type the host does not map and copy
 * it there from a host staging buffer (vkCmdCopyBuffer), on a queue of their own when the
 * device has a spare one (a transfer-only family first), else on the main queue, every
 * submit on which then takes a lock. One uploader per device (index 0 = G, 1 = G2). */
#define VK_UP_SLOTS 2
#define VK_UP_SLOT ((size_t)16 << 20)
typedef struct {
    int on;                           /* staged uploads on this device */
    uint32_t mt_dev, mt_stage;        /* the device-local target type, the host staging type */
    VkDevice dev; VkQueue q; uint32_t fam; int shared;   /* shared: q is the main queue, locked */
    uint32_t fams[3], nfams;          /* the families that touch device-local tensors (CONCURRENT when > 1) */
    pthread_mutex_t mx;               /* one upload at a time: the slots and their command buffers */
    VkCommandPool cpool; VkCommandBuffer cmd[VK_UP_SLOTS]; VkFence fence[VK_UP_SLOTS];
    int pending[VK_UP_SLOTS];         /* submitted and not waited for */
    int cur, open; size_t used;       /* the slot being filled: commands recorded, bytes used */
    VkBuffer sbuf; VkDeviceMemory smem; uint8_t *sptr;   /* VK_UP_SLOTS slots of VK_UP_SLOT bytes */
    int fill;                         /* zero-fill a fresh device-local block before its first use */
    int failed;                       /* the upload under way failed (cleared by up_finish) */
    int lost;                         /* a fence wait failed: the device is gone, for good */
    VkResult err; const char *what;   /* the failure, for the message */
    unsigned long long bytes, copies, submits, blocks_filled;
    char why[200];
    /* copies straight from host memory (up_import, COLI_VK_UP_IMPORT): the source pages
     * imported as a transfer source (VK_EXT_external_memory_host), no copy into the staging
     * buffer. imp_align 0 = none on this device; each import lives until its slot is done */
    size_t imp_align; void *imp_props;
    struct { VkBuffer buf; VkDeviceMemory mem; } *imp[VK_UP_SLOTS];
    int nimp[VK_UP_SLOTS], cimp[VK_UP_SLOTS];
    unsigned long long imp_bytes, imp_copies, imp_refused;
} VkUp;
static VkUp g_up[2] = {{.mx = PTHREAD_MUTEX_INITIALIZER, .fill = 1}, {.mx = PTHREAD_MUTEX_INITIALIZER, .fill = 1}};
static pthread_mutex_t g_qmx[2] = {PTHREAD_MUTEX_INITIALIZER, PTHREAD_MUTEX_INITIALIZER};
/* Every submit of the backend and the chain: when the uploader shares the main queue
 * its thread submits there too, and vkQueueSubmit wants the queue externally synced. */
static VkResult vk_submit(int dev, VkQueue q, const VkSubmitInfo *si, VkFence f) {
    if (!g_up[dev].shared || q != g_up[dev].q) return vkQueueSubmit(q, 1, si, f);
    pthread_mutex_lock(&g_qmx[dev]);
    VkResult r = vkQueueSubmit(q, 1, si, f);
    pthread_mutex_unlock(&g_qmx[dev]);
    return r;
}
/* COLI_VK_STAGED_FAULT=<point>[:n] (tests): the n-th time (the first by default) a staged
 * upload reaches <point> it fails there, as a driver can: stage (the uploader's staging
 * buffer), pwstage (the KV mirror's), block (a weight pool's device-local block), kvbuf (a
 * KV mirror or norm-weight buffer), record (a command buffer's begin or end), submit, wait
 * (a fence wait: the device is then taken as lost), commit (a tier expert's commit, after
 * its first matrix). One stderr line says when it fired. */
static char g_fault_point[16];
static long g_fault_at, g_fault_n;
static void up_fault_init(void) {
    const char *e = getenv("COLI_VK_STAGED_FAULT"), *c = e ? strchr(e, ':') : NULL;
    size_t n = e ? (c ? (size_t)(c - e) : strlen(e)) : 0;
    g_fault_point[0] = 0; g_fault_n = 0;
    if (!n || n >= sizeof g_fault_point) return;
    memcpy(g_fault_point, e, n); g_fault_point[n] = 0;
    g_fault_at = c ? atol(c + 1) : 1;
    if (g_fault_at < 1) g_fault_at = 1;
}
static int up_fault(const char *point) {
    if (!g_fault_point[0] || strcmp(point, g_fault_point)) return 0;
    if (__atomic_add_fetch(&g_fault_n, 1, __ATOMIC_RELAXED) != g_fault_at) return 0;
    fprintf(stderr, "[VK] COLI_VK_STAGED_FAULT: %s #%ld fails\n", point, g_fault_at);
    return 1;
}
/* How many times the COLI_VK_STAGED_FAULT point was reached so far (0 when unset). */
unsigned long long coli_vk_fault_reached(void) { return (unsigned long long)__atomic_load_n(&g_fault_n, __ATOMIC_RELAXED); }
int coli_vk_fault_set(void) { return g_fault_point[0] != 0; }

typedef struct {
    VkBuffer buf; VkDeviceMemory mem; void *ptr; size_t cap;
} Scratch;

/* Tile of the fp32 GEMM pipeline: BM outputs x BN activation rows per workgroup, BK
 * inputs per shared-memory step, TM x TN accumulators per thread; pf = fetch the next
 * step's words while this one's products run. */
typedef struct { int bm, bn, bk, tm, tn, pf; } VkGemmTile;
/* Tile of the cooperative-matrix pipeline: BM outputs x BN activation rows per
 * workgroup, WM x WN of it per subgroup (multiples of 16), BK inputs per staged step. */
typedef struct { int bm, bn, wm, wn, bk; } VkCoopTile;
#define VK_GEMM_SLOTS 2
#define VK_COOP_SLOTS 3

/* Persistent device-side KV latent/rope cache for one layer (MLA attention).
 * Host appends rows as tokens decode (absolute-position indexing); the absorb
 * kernel reads them in place. Allocated once at max_t rows, like the CUDA
 * kv_dev shadow. */
#define VK_KV_LAYERS 160
typedef struct {
    VkBuffer bl, br; VkDeviceMemory ml, mr; void *pl, *pr;
    int rows, K, R;
} VkKvLayer;

static struct {
    int ready;
    VkInstance inst;
    VkPhysicalDevice phys;
    VkDevice dev;
    VkQueue queue;
    uint32_t qfam;
    uint32_t memtype;            // HOST_VISIBLE|HOST_COHERENT (prefer DEVICE_LOCAL) — for inputs/weights
    uint32_t memtype_cached;     // HOST_CACHED — for buffers the CPU reads back (outputs)
    VkDescriptorSetLayout dsl;
    VkPipelineLayout plyt;
    VkPipeline pipe;
    VkShaderModule shader;
    VkDescriptorPool dpool;
    VkDescriptorSet dset;
    /* Tiled GEMMs for prefill-sized S, with the GEMV's 4 bindings and push constants so
     * they bind through G.plyt/G.dset: the fp32 one (qmatmul_gemm.spv) and, when the
     * device has a 16x16x16 fp16 -> fp32 subgroup MMA and a settable subgroup size
     * (coop_sg), the cooperative-matrix one (qmatmul_coop.spv) for the formats whose
     * decoded weights are fp16 values. Each is built at a few tile widths (specialization
     * constants, slots of growing BN) and a call takes the narrowest that covers its S.
     * A call takes a GEMM when S >= gemm_min_s (0 = GEMV for every S) and S*O >=
     * gemm_min_so; coop_off is the harness's switch to the fp32 GEMM. */
    VkShaderModule shader_gemm, shader_coop;
    VkPipeline pipe_gemm[VK_GEMM_SLOTS], pipe_coop[VK_COOP_SLOTS];
    VkGemmTile gemm_t[VK_GEMM_SLOTS];
    VkCoopTile coop_t[VK_COOP_SLOTS];
    int gemm_min_s, gemm_min_so, has_coop, coop_sg, coop_off;
    int has_bda;           /* bufferDeviceAddress on: the expert batch's grouped GEMM reads weights by address */
    /* fused dual gate+up+silu pipeline (6 bindings): x, Wg, gscale, Wu, uscale, hidden */
    VkShaderModule shader_gu; VkDescriptorSetLayout dsl_gu; VkPipelineLayout plyt_gu;
    VkPipeline pipe_gu; VkDescriptorPool dpool_gu; VkDescriptorSet dset_gu;
    /* MLA absorb attention core (7 bindings): q, W, scales, Lcache, Rcache, scores, ctx */
    VkShaderModule shader_att; VkDescriptorSetLayout dsl_att; VkPipelineLayout plyt_att;
    VkPipeline pipe_att; VkDescriptorPool dpool_att; VkDescriptorSet dset_att;
    VkCommandPool cpool;
    VkCommandBuffer cmd;
    VkFence fence;
    Scratch x, y, h;   /* h = fused gate+up hidden output */
    /* full expert-group scratch: activations/hidden/output for K experts + per-expert
     * descriptor sets (gate_up: dsl_gu, down: dsl), so gate_up->down runs on-device in
     * one submit with hidden never leaving the GPU. */
    Scratch eg_x, eg_h, eg_y;
    VkDescriptorPool eg_pool; VkDescriptorSet eg_gu[64], eg_dn[64]; int eg_nsets;
    /* expert-group ASYNC state: its own command buffer + fence so an in-flight group
     * never collides with the main cmd/fence (dense matmuls, absorb) — issue() returns
     * immediately, the CPU computes its share, take() joins. */
    VkCommandBuffer eg_cmd; VkFence eg_fence; int eg_inflight; size_t eg_pending_yb;
    double eg_t0, eg_t1, eg_t2, eg_t3; int eg_prof;
    /* q-prep chain (pair -> rmsnorm -> q_b in ONE submit): norm pipeline (3 bindings),
     * a 3rd matmul set + norm set, GPU-only latent intermediates, per-layer resident
     * norm-weight buffers (tiny, uploaded once like the KV mirror). */
    VkShaderModule shader_nrm; VkDescriptorSetLayout dsl_nrm; VkPipelineLayout plyt_nrm;
    VkPipeline pipe_nrm; VkDescriptorPool qprep_pool; VkDescriptorSet dset_qp3, dset_nrm;
    Scratch qp1, qp2;
    VkBuffer lnbuf[VK_KV_LAYERS]; VkDeviceMemory lnmem[VK_KV_LAYERS]; int lnlen[VK_KV_LAYERS];
    Scratch att_sc;              /* attention score scratch (GPU-only) */
    Scratch att_ctx;             /* fused absorb+o: ctx stays on device (GPU-only) */
    Scratch y2;                  /* second output of the fused matmul pair (readback) */
    VkDescriptorPool pair_pool; VkDescriptorSet dset_pair;   /* 4-binding set for the pair's 2nd matmul */
    VkKvLayer kv[VK_KV_LAYERS];  /* per-layer resident KV latent/rope cache */
    /* resubmit cache: skip vkUpdateDescriptorSets + command re-record when the bound
     * tensor / shape / scratch buffers are unchanged from the previous call (the hot-
     * expert-called-repeatedly pattern). The synchronous fence wait each call means no
     * submission is ever in flight, so rebinding/re-recording only when something
     * actually changed is safe. */
    ColiVkTensor *bound_tensor; int bound_S, bound_I, bound_O, bound_gemm, cmd_ready;
    VkBuffer bound_xbuf, bound_ybuf;
    size_t used_bytes, tensor_count;
    /* VRAM pressure-proofing: with VK_EXT_memory_priority the attention working set
     * (KV mirror, scratches) outranks bulk expert weights, so an oversubscribed heap
     * evicts cold tier experts instead of thrashing the per-token attention submits
     * (measured: decode attention 7.8s at 7.6 GB resident -> 17.8s at 15.2 GB).
     * VK_EXT_memory_budget lets the tier fill stop at a reserve instead of guessing. */
    int has_prio, has_budget, has_hostmem;
    float prio;                  /* priority applied to the NEXT allocations (class knob) */
    /* the routed-expert tier's queue (coli_vk_xb_*): family, index, whether it is the
     * main queue after all, and its timestamp bits; device limits its batches respect */
    VkQueue tqueue; uint32_t tq_fam, tq_idx, tq_ts; int tq_shared;
    size_t ssbo_align, ssbo_range; float ts_period;
    size_t buf_align;            /* what a storage buffer's memory requirements align to */
    uint32_t memtype_dev;        /* DEVICE_LOCAL, for scratch only the device touches */
    char spv_path[1024];         /* the main shader, its siblings are found beside it */
} G;

struct PC { int fmt, S, I, O, rowWords, gs; };
/* gate_up shader only: the activation (COLI_VK_ACT_*) and its parameters: act 0's
 * SwiGLU clamp (limit<=0 keeps the unclamped path, GLM-5.2 / colibri.c), act 1's
 * two bounds. Must ride per dispatch: the same pipeline serves callers whose
 * correct value is "no clamp". */
struct PCGU { int fmt, S, I, O, rowWords, gs; float limit; int act; float a, b; };
static float g_swiglu_limit = 0.f;
void coli_vk_set_swiglu_limit(float limit) { g_swiglu_limit = limit > 0.f ? limit : 0.f; }
static struct PCGU pcgu(int fmt, int S, int I, int O, int rowWords, int gs) {
    return (struct PCGU){fmt, S, I, O, rowWords, gs, g_swiglu_limit, 0, 0.f, 0.f};
}
/* expert_act.comp: the activation on its own, for the tiled-GEMM route */
struct PCAct { int n, act; float limit, a, b; };
static void async_begin(int dev);
static void async_end(int dev);
struct PCN { int S, D; float eps; };
/* Push constants of the absorb attention kernel (must match attention_absorb.comp). */
struct PCAttn { int fmt, S, H, Q, R, V, K, st0, T, rowWords, cap; float scale; int gs; };

static int pick_memtype(VkPhysicalDevice phys) {
    VkPhysicalDeviceMemoryProperties m;
    vkGetPhysicalDeviceMemoryProperties(phys, &m);
    int best = -1;
    for (uint32_t i = 0; i < m.memoryTypeCount; i++) {
        VkMemoryPropertyFlags f = m.memoryTypes[i].propertyFlags;
        if ((f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
            (f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            if (f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) return (int)i; // ideal on APU
            if (best < 0) best = (int)i;
        }
    }
    return best;
}

/* Cached+coherent host-visible type for buffers the CPU READS BACK. pick_memtype prefers
 * DEVICE_LOCAL host-visible = write-combined VRAM over ReBAR, which the CPU writes fast but
 * reads catastrophically slowly (~40 MB/s). Outputs must be HOST_CACHED for cheap readback. */
static int pick_memtype_cached(VkPhysicalDevice phys) {
    VkPhysicalDeviceMemoryProperties m;
    vkGetPhysicalDeviceMemoryProperties(phys, &m);
    for (uint32_t i = 0; i < m.memoryTypeCount; i++) {
        VkMemoryPropertyFlags f = m.memoryTypes[i].propertyFlags;
        if ((f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) && (f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) &&
            (f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT)) return (int)i;
    }
    return pick_memtype(phys);   /* no cached type -> fall back (no worse than before) */
}

/* DEVICE_LOCAL for buffers only the device reads and writes (the expert batch's
 * intermediates); the host-visible type when there is none. */
static int pick_memtype_device(VkPhysicalDevice phys) {
    VkPhysicalDeviceMemoryProperties m;
    vkGetPhysicalDeviceMemoryProperties(phys, &m);
    for (uint32_t i = 0; i < m.memoryTypeCount; i++)
        if (m.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) return (int)i;
    return pick_memtype(phys);
}

/* Staged or mapped (see VkUp above). Staged when COLI_VK_STAGED=1, or, unset, when the
 * host-visible device-local heap (the heap of mt_host, the type the mapped path writes,
 * when that is device-local; none otherwise) holds less than a quarter of the largest
 * device-local heap: a discrete card without Resizable BAR (256 MB of 8 GB), never one
 * with it, an integrated GPU or Lavapipe (the whole heap is host-visible).
 * COLI_VK_HOST_VISIBLE_CAP_MB=N treats the host-visible heap as at most N MiB, which
 * lets a device with Resizable BAR or unified memory take the decision a card without
 * it takes. The target type: device-local and not host-visible on the largest
 * device-local heap (the host never maps it), else that heap's device-local type
 * (Lavapipe has one type for everything); vendor-specific types (uncached, coherent)
 * are passed over. The staging type: host-visible and coherent, not device-local
 * where there is one, so staging never takes the window. */
static int place_plain(VkMemoryPropertyFlags f) {
    return !(f & (VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT | VK_MEMORY_PROPERTY_PROTECTED_BIT | 0xC0u));
}
static int place_decide(VkPhysicalDevice phys, int mt_host, VkUp *u) {
    VkPhysicalDeviceMemoryProperties m;
    vkGetPhysicalDeviceMemoryProperties(phys, &m);
    VkDeviceSize dl_max = 0, hv = 0;
    for (uint32_t i = 0; i < m.memoryHeapCount; i++)
        if ((m.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) && m.memoryHeaps[i].size > dl_max)
            dl_max = m.memoryHeaps[i].size;
    if (mt_host >= 0 && (m.memoryTypes[mt_host].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
        hv = m.memoryHeaps[m.memoryTypes[mt_host].heapIndex].size;
    const char *cap = getenv("COLI_VK_HOST_VISIBLE_CAP_MB");
    if (cap && *cap && (VkDeviceSize)atoll(cap) << 20 < hv) hv = (VkDeviceSize)atoll(cap) << 20;
    int dev = -1, any = -1, st = -1;
    for (uint32_t i = 0; i < m.memoryTypeCount; i++) {
        VkMemoryPropertyFlags f = m.memoryTypes[i].propertyFlags;
        if (!place_plain(f)) continue;
        int big = (f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) && m.memoryHeaps[m.memoryTypes[i].heapIndex].size == dl_max;
        if (big && any < 0) any = (int)i;
        if (big && !(f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) && dev < 0) dev = (int)i;
        if ((f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) && (f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) &&
            !(f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) && st < 0) st = (int)i;
    }
    if (dev < 0) dev = any;
    if (st < 0) st = mt_host;
    const char *e = getenv("COLI_VK_STAGED");
    int on;
    if (e && *e) { on = atoi(e) != 0; snprintf(u->why, sizeof u->why, "COLI_VK_STAGED=%s", e); }
    else {
        on = dl_max > 0 && hv * 4 < dl_max;
        snprintf(u->why, sizeof u->why, "%llu of %llu MiB of device-local memory is host-visible%s",
                 (unsigned long long)(hv >> 20), (unsigned long long)(dl_max >> 20),
                 cap && *cap ? " (COLI_VK_HOST_VISIBLE_CAP_MB)" : "");
    }
    if (on && (dev < 0 || st < 0)) {
        snprintf(u->why, sizeof u->why, "no device-local or no host-visible coherent memory type");
        on = 0;
    }
    u->on = on;
    if (on) { u->mt_dev = (uint32_t)dev; u->mt_stage = (uint32_t)st; }
    return on;
}
/* The uploader's command buffers, fences and staging slots (u->dev, u->fam set). */
/* A slot's imports, once its copies are done (or it was never submitted). */
static void up_imp_release(VkUp *u, int s) {
    for (int i = 0; i < u->nimp[s]; i++) {
        vkDestroyBuffer(u->dev, u->imp[s][i].buf, NULL);
        vkFreeMemory(u->dev, u->imp[s][i].mem, NULL);
    }
    u->nimp[s] = 0;
}
/* The host image a tier tensor is filled in (coli_vk_tier_tensor, staged uploads): aligned
 * and rounded to the device's import alignment, so up_import can copy from it in place. */
static uint8_t *up_img_alloc(int dev, size_t bytes, uint8_t *aligned) {
    size_t al = g_up[dev].imp_align;
    *aligned = al != 0;
    if (!al) return calloc(1, bytes);
    void *p = NULL;
    size_t sz = (bytes + al - 1) / al * al;
#ifdef _WIN32
    p = _aligned_malloc(sz, al);
#else
    if (posix_memalign(&p, al, sz)) p = NULL;
#endif
    if (p) memset(p, 0, sz);
    return p;
}
static void up_img_free(uint8_t *p, int aligned) {
    if (!p) return;
#ifdef _WIN32
    if (aligned) { _aligned_free(p); return; }
#else
    (void)aligned;
#endif
    free(p);
}
/* Copies straight from host memory on this device (up_import): VK_EXT_external_memory_host
 * enabled, staged uploads on, COLI_VK_UP_IMPORT not 0. After up_init. */
static void up_import_init(VkUp *u, VkPhysicalDevice phys, int has_hostmem, const char *who) {
    u->imp_align = 0; u->imp_props = NULL;
#ifdef VK_EXT_external_memory_host
    const char *e = getenv("COLI_VK_UP_IMPORT");
    if (!u->on || !u->dev || !has_hostmem || (e && *e == '0')) return;
    VkPhysicalDeviceExternalMemoryHostPropertiesEXT hp = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT};
    VkPhysicalDeviceProperties2 p2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &hp};
    vkGetPhysicalDeviceProperties2(phys, &p2);
    u->imp_props = (void *)vkGetDeviceProcAddr(u->dev, "vkGetMemoryHostPointerPropertiesEXT");
    if (!u->imp_props) return;
    u->imp_align = hp.minImportedHostPointerAlignment ? (size_t)hp.minImportedHostPointerAlignment : 4096;
    fprintf(stderr, "[VK] %sstaged uploads: copied straight from host memory where its pages import "
            "(%zu KiB pages; COLI_VK_UP_IMPORT=0 stages every copy)\n", who, u->imp_align >> 10);
#else
    (void)phys; (void)has_hostmem; (void)who;
#endif
}
static int up_init(VkUp *u) {
    VkCommandPoolCreateInfo cp = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = u->fam};
    if (vkCreateCommandPool(u->dev, &cp, NULL, &u->cpool) != VK_SUCCESS) return 0;
    VkCommandBufferAllocateInfo ca = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = u->cpool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = VK_UP_SLOTS};
    if (vkAllocateCommandBuffers(u->dev, &ca, u->cmd) != VK_SUCCESS) return 0;
    VkFenceCreateInfo fi = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    for (int s = 0; s < VK_UP_SLOTS; s++)
        if (vkCreateFence(u->dev, &fi, NULL, &u->fence[s]) != VK_SUCCESS) return 0;
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = VK_UP_SLOTS * VK_UP_SLOT,
        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT, .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    if (vkCreateBuffer(u->dev, &bi, NULL, &u->sbuf) != VK_SUCCESS) return 0;
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(u->dev, u->sbuf, &req);
    if (!(req.memoryTypeBits & (1u << u->mt_stage)) || up_fault("stage")) return 0;
    VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = req.size, .memoryTypeIndex = u->mt_stage};
    if (vkAllocateMemory(u->dev, &ai, NULL, &u->smem) != VK_SUCCESS) return 0;
    if (vkBindBufferMemory(u->dev, u->sbuf, u->smem, 0) != VK_SUCCESS ||
        vkMapMemory(u->dev, u->smem, 0, VK_WHOLE_SIZE, 0, (void **)&u->sptr) != VK_SUCCESS) return 0;
    return 1;
}
static void up_destroy(VkUp *u) {
    if (!u->dev) return;
    for (int s = 0; s < VK_UP_SLOTS; s++) { up_imp_release(u, s); free(u->imp[s]); u->imp[s] = NULL; u->cimp[s] = 0; }
    if (u->sbuf) vkDestroyBuffer(u->dev, u->sbuf, NULL);
    if (u->smem) vkFreeMemory(u->dev, u->smem, NULL);
    for (int s = 0; s < VK_UP_SLOTS; s++) if (u->fence[s]) vkDestroyFence(u->dev, u->fence[s], NULL);
    if (u->cpool) vkDestroyCommandPool(u->dev, u->cpool, NULL);
    u->sbuf = VK_NULL_HANDLE; u->smem = VK_NULL_HANDLE; u->sptr = NULL; u->cpool = VK_NULL_HANDLE;
    for (int s = 0; s < VK_UP_SLOTS; s++) { u->fence[s] = VK_NULL_HANDLE; u->cmd[s] = VK_NULL_HANDLE; u->pending[s] = 0; }
    u->open = 0; u->used = 0; u->cur = 0; u->failed = u->lost = 0; u->what = NULL;
}

static int alloc_buf_mt(size_t bytes, VkBuffer *buf, VkDeviceMemory *mem, void **ptr, uint32_t memtype,
                        VkBufferUsageFlags usage) {
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = bytes, .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    VKCHECK(vkCreateBuffer(G.dev, &bi, NULL, buf), "vkCreateBuffer");
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(G.dev, *buf, &req);
    VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = req.size, .memoryTypeIndex = memtype};
#ifdef VK_EXT_memory_priority
    VkMemoryPriorityAllocateInfoEXT pri = {.sType = VK_STRUCTURE_TYPE_MEMORY_PRIORITY_ALLOCATE_INFO_EXT,
        .priority = G.prio};
    if (G.has_prio) ai.pNext = &pri;
#endif
    VKCHECK(vkAllocateMemory(G.dev, &ai, NULL, mem), "vkAllocateMemory");
    VKCHECK(vkBindBufferMemory(G.dev, *buf, *mem, 0), "vkBindBufferMemory");
    if (ptr) VKCHECK(vkMapMemory(G.dev, *mem, 0, bytes, 0, ptr), "vkMapMemory");
    return 1;
}
static int alloc_hostvis_mt(size_t bytes, VkBuffer *buf, VkDeviceMemory *mem, void **ptr, uint32_t memtype) {
    return alloc_buf_mt(bytes, buf, mem, ptr, memtype, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
}
/* Priority class of subsequent allocations (VK_EXT_memory_priority; no-op without it).
 * Scratches/KV force 1.0 internally; weight uploads take whatever is current — the
 * engine sets 0.4 around the bulk expert-tier fill, dense stays at the 0.75 default. */
void coli_vk_alloc_priority(float p) { G.prio = p < 0 ? 0 : p > 1 ? 1 : p; }

/* Device-local heap usage/budget in GB (VK_EXT_memory_budget). Returns 0 when the
 * extension is absent — callers then keep their count-based caps unchanged. */
int coli_vk_mem_budget(double *used_gb, double *budget_gb) {
    if (g_mem.cap[0] && G.phys) {   /* COLI_VK_DEVICE_CAP_MB: the cap and this process's bytes */
        if (used_gb) *used_gb = (double)__atomic_load_n(&g_mem.used[0], __ATOMIC_RELAXED) / 1e9;
        if (budget_gb) *budget_gb = (double)g_mem.cap[0] / 1e9;
        return 1;
    }
#ifdef VK_EXT_memory_budget
    if (!G.has_budget || !G.phys) return 0;
    VkPhysicalDeviceMemoryBudgetPropertiesEXT bud = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
    VkPhysicalDeviceMemoryProperties2 mp2 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2, .pNext = &bud};
    vkGetPhysicalDeviceMemoryProperties2(G.phys, &mp2);
    double u = 0, b = 0;
    for (uint32_t i = 0; i < mp2.memoryProperties.memoryHeapCount; i++)
        if (mp2.memoryProperties.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            u += (double)bud.heapUsage[i]; b += (double)bud.heapBudget[i];
        }
    if (used_gb) *used_gb = u / 1e9;
    if (budget_gb) *budget_gb = b / 1e9;
    return b > 0;
#else
    (void)used_gb; (void)budget_gb; return 0;
#endif
}
/* Free device memory now: the cap less what this process holds (COLI_VK_DEVICE_CAP_MB), else
 * VK_EXT_memory_budget's budget less its usage, else the largest device-local heap less
 * what this process holds. */
size_t coli_vk_free_bytes(void) {
    if (!G.phys) return 0;
    if (g_mem.cap[0]) return (size_t)vk_mem_room(0);
    double used = 0, bud = 0;
    if (coli_vk_mem_budget(&used, &bud)) return bud > used ? (size_t)((bud - used) * 1e9) : 0;
    size_t dl = coli_vk_device_local_bytes(), u = coli_vk_device_used();
    return dl > u ? dl - u : 0;
}
static int alloc_hostvis(size_t bytes, VkBuffer *buf, VkDeviceMemory *mem, void **ptr) {
    return alloc_hostvis_mt(bytes, buf, mem, ptr, G.memtype);
}

static int scratch_reserve_mt(Scratch *s, size_t bytes, uint32_t memtype) {
    if (s->cap >= bytes) return 1;
    if (s->buf) { vkDestroyBuffer(G.dev, s->buf, NULL); vkFreeMemory(G.dev, s->mem, NULL); }
    s->buf = VK_NULL_HANDLE; s->cap = 0; s->ptr = NULL;
    /* The driver may hand the next buffer the destroyed one's handle value, so the
     * matmul binding cache cannot tell them apart by handle: drop it on every regrow.
     * (Same tensor, growing S, e.g. a prefill chunk after decode, left the descriptor
     * on the freed memory: a GPUVM fault on RADV.) */
    G.bound_tensor = NULL; G.cmd_ready = 0;
    float p0 = G.prio; G.prio = 1.0f;            /* scratches ride every submit: never evict */
    int ok = alloc_hostvis_mt(bytes, &s->buf, &s->mem, &s->ptr, memtype);
    G.prio = p0;
    if (!ok) return 0;
    s->cap = bytes;
    return 1;
}
static int scratch_reserve(Scratch *s, size_t bytes) { return scratch_reserve_mt(s, bytes, G.memtype); }

/* Bytes of one weight row on the CPU side. fmt 10 (f32) and 11 (bf16) are the plain
 * float weights the newer engines keep resident; they carry no scales. */
static size_t cpu_row_bytes(int fmt, int I) {
    return fmt == 1 || fmt == 12 || fmt == 13 ? (size_t)I   // int8, fp8: one byte per weight
         : fmt == 5  ? ((size_t)I + 63) / 64 * 24            // int3-g64: 24B per 64-group
         : fmt == 10 ? (size_t)I * 4
         : fmt == 11 || fmt == 14 ? (size_t)I * 2            // bf16, f16
         : (size_t)(I + 1) / 2;
}
static int rowwords(int fmt, int I) {
    return (int)((cpu_row_bytes(fmt, I) + 3) / 4);           // padded to uint32 (24|4: exact)
}
/* Scale floats per tensor: per-row formats carry O, int3-g64 carries O*ceil(I/64)
 * (one f32 per 64-input group). upload_tensor and tensor_free must agree on this. */
static size_t scale_floats(int fmt, int I, int O, int gs) {
    if (fmt == 10 || fmt == 11 || fmt == 14) return 1;    // unused by the shader; bound anyway
    if (fmt == 5) return (size_t)O * (((size_t)I + 63) / 64);
    if (fmt == 4 || fmt == 7 || fmt == 12 || fmt == 13)
        return (size_t)O * (((size_t)I + gs - 1) / gs);   // per-group [O,ng]
    return (size_t)O;
}

static VkShaderModule load_spv(VkDevice dev, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[VK] cannot open %s\n", path); return VK_NULL_HANDLE; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    if (n <= 0 || n % 4 != 0) {   // SPIR-V is a stream of uint32; empty/non-seekable/odd size is invalid
        fprintf(stderr, "[VK] bad SPIR-V size %ld in %s\n", n, path); fclose(f); return VK_NULL_HANDLE; }
    uint32_t *code = malloc((size_t)n);
    if (!code) { fclose(f); return VK_NULL_HANDLE; }
    if (fread(code, 1, n, f) != (size_t)n) { fclose(f); free(code); return VK_NULL_HANDLE; }
    fclose(f);
    VkShaderModuleCreateInfo si = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = n, .pCode = code};
    VkShaderModule m;
    VkResult r = vkCreateShaderModule(dev, &si, NULL, &m);
    free(code);
    return r == VK_SUCCESS ? m : VK_NULL_HANDLE;
}

/* Build a compute pipeline + descriptor pool/set for nbind storage buffers with a
 * pc_size-byte push constant. Used by the 4-binding matmul, 6-binding gate_up and
 * 7-binding absorb attention pipelines. */
static int build_pipeline(VkDevice dev, int nbind, size_t pc_size, VkShaderModule shader,
                          VkDescriptorSetLayout *dsl, VkPipelineLayout *plyt, VkPipeline *pipe,
                          VkDescriptorPool *dpool, VkDescriptorSet *dset) {
    VkDescriptorSetLayoutBinding b[8];
    for (int i = 0; i < nbind; i++) b[i] = (VkDescriptorSetLayoutBinding){
        .binding = (uint32_t)i, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT};
    VkDescriptorSetLayoutCreateInfo dsli = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = (uint32_t)nbind, .pBindings = b};
    VKCHECK(vkCreateDescriptorSetLayout(dev, &dsli, NULL, dsl), "descSetLayout");
    VkPushConstantRange pcr = {.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = (uint32_t)pc_size};
    VkPipelineLayoutCreateInfo pli = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1, .pSetLayouts = dsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr};
    VKCHECK(vkCreatePipelineLayout(dev, &pli, NULL, plyt), "pipelineLayout");
    VkComputePipelineCreateInfo cpi = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                  .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = shader, .pName = "main"},
        .layout = *plyt};
    VKCHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpi, NULL, pipe), "pipeline");
    VkDescriptorPoolSize ps = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = (uint32_t)nbind};
    VkDescriptorPoolCreateInfo dpi = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &ps};
    VKCHECK(vkCreateDescriptorPool(dev, &dpi, NULL, dpool), "descPool");
    VkDescriptorSetAllocateInfo dsa = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = *dpool, .descriptorSetCount = 1, .pSetLayouts = dsl};
    VKCHECK(vkAllocateDescriptorSets(dev, &dsa, dset), "allocDescSet");
    return 1;
}

/* "…/qmatmul.spv" -> "…/qmatmul<suffix>" (sibling of the main shader). */
static void derive_sibling(const char *spv, const char *suffix, char *out, size_t n) {
    const char *dot = strstr(spv, ".spv");
    if (dot && (size_t)(dot - spv) + strlen(suffix) + 1 < n) {
        size_t pre = (size_t)(dot - spv);
        memcpy(out, spv, pre); strcpy(out + pre, suffix);
    } else snprintf(out, n, "%s", spv);
}
/* "…/qmatmul.spv" -> "…/attention_absorb.spv" (same directory). */
static void derive_dir_file(const char *spv, const char *fname, char *out, size_t n) {
    const char *sl = strrchr(spv, '/');
    size_t pre = sl ? (size_t)(sl - spv) + 1 : 0;
    if (pre + strlen(fname) + 1 < n) { memcpy(out, spv, pre); strcpy(out + pre, fname); }
    else snprintf(out, n, "%s", fname);
}

static int gemm_tile_ok(VkGemmTile t, const VkPhysicalDeviceLimits *lim) {
    if (t.bm < 1 || t.bn < 1 || t.tm < 1 || t.tn < 1 || t.bk < 8 || t.bk % 8 ||
        t.bm % t.tm || t.bn % t.tn) return 0;
    long threads = (long)(t.bm / t.tm) * (t.bn / t.tn);
    long lds = (long)(t.bm + t.bn) * (t.bk / 4 + 1) * 16;     /* wsh + xsh, vec4 quads */
    return threads >= 32 && threads <= (long)lim->maxComputeWorkGroupInvocations &&
           threads <= (long)lim->maxComputeWorkGroupSize[0] && lds <= (long)lim->maxComputeSharedMemorySize;
}
static int gemm_pipeline(VkGemmTile t, int slot) {
    int32_t sv[7] = {t.bm, t.bn, t.bk, t.tm, t.tn, (t.bm / t.tm) * (t.bn / t.tn), t.pf ? 1 : 0};
    VkSpecializationMapEntry me[7];
    for (int i = 0; i < 7; i++) me[i] = (VkSpecializationMapEntry){(uint32_t)i, (uint32_t)(i * 4), 4};
    VkSpecializationInfo si = {7, me, sizeof(sv), sv};
    VkComputePipelineCreateInfo cpi = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                  .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = G.shader_gemm, .pName = "main",
                  .pSpecializationInfo = &si},
        .layout = G.plyt};
    VkPipeline pipe;
    VKCHECK(vkCreateComputePipelines(G.dev, VK_NULL_HANDLE, 1, &cpi, NULL, &pipe), "gemm pipeline");
    if (G.pipe_gemm[slot]) vkDestroyPipeline(G.dev, G.pipe_gemm[slot], NULL);
    G.pipe_gemm[slot] = pipe; G.gemm_t[slot] = t;
    G.cmd_ready = 0;                     /* a recorded GEMM dispatch names the old pipeline */
    return 1;
}

/* qmatmul_coop.comp's EST: floats per epilogue row of its accumulator staging (a
 * multiple of 4 floats, so coopMatStore's stride is 16-byte aligned; #1908). */
#define VK_COOP_EST 20
/* Subgroups of exactly G.coop_sg lanes; BK a multiple of 32 (group sizes are). */
static int coop_tile_ok(VkCoopTile t, const VkPhysicalDeviceLimits *lim) {
    if (t.wm < 16 || t.wn < 16 || t.wm % 16 || t.wn % 16 || t.bm % t.wm || t.bn % t.wn ||
        t.bk < 32 || t.bk % 32 || G.coop_sg < 16 || 256 % G.coop_sg) return 0;
    long nw = (long)(t.bm / t.wm) * (t.bn / t.wn), threads = nw * G.coop_sg;
    long lds = (long)(t.bm + 2 * t.bn) * (t.bk + 8) * 2 + (long)t.bm * 4 + nw * 16 * VK_COOP_EST * 4;
    return threads <= (long)lim->maxComputeWorkGroupInvocations &&
           threads <= (long)lim->maxComputeWorkGroupSize[0] && lds <= (long)lim->maxComputeSharedMemorySize;
}
static int coop_pipeline(VkCoopTile t, int slot) {
#ifdef VK_KHR_cooperative_matrix
    int32_t sv[7] = {t.bm, t.bn, t.wm, t.wn, t.bk, G.coop_sg, (t.bm / t.wm) * (t.bn / t.wn) * G.coop_sg};
    VkSpecializationMapEntry me[7];
    for (int i = 0; i < 7; i++) me[i] = (VkSpecializationMapEntry){(uint32_t)i, (uint32_t)(i * 4), 4};
    VkSpecializationInfo si = {7, me, sizeof(sv), sv};
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT rss = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO_EXT,
        .requiredSubgroupSize = (uint32_t)G.coop_sg};
    VkComputePipelineCreateInfo cpi = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .pNext = &rss,
                  .flags = VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT_EXT,
                  .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = G.shader_coop, .pName = "main",
                  .pSpecializationInfo = &si},
        .layout = G.plyt};
    VkPipeline pipe;
    VKCHECK(vkCreateComputePipelines(G.dev, VK_NULL_HANDLE, 1, &cpi, NULL, &pipe), "coop pipeline");
    if (G.pipe_coop[slot]) vkDestroyPipeline(G.dev, G.pipe_coop[slot], NULL);
    G.pipe_coop[slot] = pipe; G.coop_t[slot] = t;
    G.cmd_ready = 0;
    return 1;
#else
    (void)t; (void)slot; return 0;
#endif
}

/* The tiles, narrowest first, measured on a Radeon 780M (RDNA3, RADV) over Qwen3.8's
 * shapes: BN = 32 serves a 32-row prefill chunk, the wider ones longer prompts.
 * COLI_VK_GEMM_TILE=bm,bn,bk,tm,tn[,pf] and COLI_VK_COOP_TILE=bm,bn,wm,wn,bk put one
 * tile in every slot (bench). */
static const VkGemmTile gemm_tiles[VK_GEMM_SLOTS] = {{128, 32, 32, 4, 4, 1}, {64, 64, 32, 4, 4, 0}};
static const VkCoopTile coop_tiles[VK_COOP_SLOTS] = {{128, 32, 32, 32, 32}, {128, 64, 32, 32, 32},
                                                     {128, 128, 64, 32, 32}};
/* The narrowest slot whose BN covers S, else the widest. */
static int gemm_slot(int S) {
    int k = 0;
    while (k + 1 < VK_GEMM_SLOTS && G.pipe_gemm[k + 1] && S > G.gemm_t[k].bn) k++;
    return k;
}
static int coop_slot(int S) {
    int k = 0;
    while (k + 1 < VK_COOP_SLOTS && G.pipe_coop[k + 1] && S > G.coop_t[k].bn) k++;
    return k;
}

static void vk_prof_paths(void);
static int g_vk_prof;

int coli_vk_init(const char *spv_path) {
    if (G.ready) return 1;
    up_fault_init();
    VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .apiVersion = VK_API_VERSION_1_2};
    VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app};
    VKCHECK(vkCreateInstance(&ici, NULL, &G.inst), "vkCreateInstance");

    uint32_t nd = 0;
    vkEnumeratePhysicalDevices(G.inst, &nd, NULL);
    if (!nd) { fprintf(stderr, "[VK] no devices\n"); return 0; }
    VkPhysicalDevice devs[8]; if (nd > 8) nd = 8;
    vkEnumeratePhysicalDevices(G.inst, &nd, devs);
    // Prefer a real GPU over a CPU/software device (llvmpipe) on multi-adapter hosts:
    // discrete > integrated > virtual > other/cpu. Falls back to devs[0] if all equal.
    // COLI_VK_DEV=<index> takes that entry of the enumeration instead (the order
    // vulkaninfo lists as GPU0, GPU1, ...): with two discrete GPUs the ranking always
    // took the first one (#1908). Out of range or not a number: a line, then the ranking.
    G.phys = devs[0];
    int bestrank = -1, forced = 0;
    const char *dv = getenv("COLI_VK_DEV");
    if (dv && *dv) {
        char *end = NULL; long k = strtol(dv, &end, 10);
        if (end != dv && *end == 0 && k >= 0 && k < (long)nd) { G.phys = devs[k]; forced = 1; }
        else fprintf(stderr, "[VK] COLI_VK_DEV=%s ignored: %u device%s, numbered from 0\n", dv, nd, nd == 1 ? "" : "s");
    }
    for (uint32_t i = 0; i < nd && !forced; i++) {
        VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(devs[i], &p);
        int rank = p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU   ? 4 :
                   p.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? 3 :
                   p.deviceType == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU    ? 2 :
                   p.deviceType == VK_PHYSICAL_DEVICE_TYPE_OTHER          ? 1 : 0; // CPU last
        if (rank > bestrank) { bestrank = rank; G.phys = devs[i]; }
    }

    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(G.phys, &nq, NULL);
    VkQueueFamilyProperties qf[16]; if (nq > 16) nq = 16;
    vkGetPhysicalDeviceQueueFamilyProperties(G.phys, &nq, qf);
    G.qfam = UINT32_MAX;
    for (uint32_t i = 0; i < nq; i++)
        if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { G.qfam = i; break; }
    if (G.qfam == UINT32_MAX) { fprintf(stderr, "[VK] no compute queue\n"); return 0; }

    /* A second queue for the routed-expert tier's async batches (coli_vk_xb_*), so the
     * synchronous dense matmuls of the same layer do not wait behind a batch: another
     * queue of the main family when it has two (NVIDIA, Intel), else a queue of another
     * compute family, a compute-only one first (RADV's async compute), else the main
     * queue is shared (Lavapipe) and the two serialize. COLI_VK_TIER_QUEUE=0 shares it
     * on purpose. Nothing else changes for the engines that never open a batch. */
    float qprio[3] = {1.0f, 1.0f, 1.0f};
    VkDeviceQueueCreateInfo qis[3] = {{.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = G.qfam, .queueCount = 1, .pQueuePriorities = qprio}};
    uint32_t nqi = 1;
    G.tq_fam = G.qfam; G.tq_idx = 0;
    {
        const char *tq = getenv("COLI_VK_TIER_QUEUE");
        if (!(tq && *tq == '0')) {
            if (qf[G.qfam].queueCount >= 2) { qis[0].queueCount = 2; G.tq_idx = 1; }
            else {
                uint32_t best = UINT32_MAX;
                for (uint32_t i = 0; i < nq; i++) {
                    if (i == G.qfam || !(qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT)) continue;
                    if (best == UINT32_MAX || ((qf[best].queueFlags & VK_QUEUE_GRAPHICS_BIT) &&
                                               !(qf[i].queueFlags & VK_QUEUE_GRAPHICS_BIT))) best = i;
                }
                if (best != UINT32_MAX) {
                    qis[1] = (VkDeviceQueueCreateInfo){.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                        .queueFamilyIndex = best, .queueCount = 1, .pQueuePriorities = qprio};
                    nqi = 2; G.tq_fam = best;
                }
            }
        }
        G.tq_ts = qf[G.tq_fam].timestampValidBits;
    }
    /* Staged uploads (VkUp), decided before the device exists since their queue is one of
     * its queues: a transfer-only family (a copy engine), else a spare queue of a family
     * already asked for, else the main queue, shared under g_qmx[0]. */
    uint32_t up_fam = G.qfam, up_idx = 0; int up_shared = 1;
    if (place_decide(G.phys, pick_memtype(G.phys), &g_up[0])) {
        uint32_t t = UINT32_MAX;
        for (uint32_t i = 0; i < nq && t == UINT32_MAX; i++)
            if ((qf[i].queueFlags & VK_QUEUE_TRANSFER_BIT) && qf[i].queueCount &&
                !(qf[i].queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) && i != G.qfam && i != G.tq_fam) t = i;
        if (t != UINT32_MAX) {
            qis[nqi++] = (VkDeviceQueueCreateInfo){.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                .queueFamilyIndex = t, .queueCount = 1, .pQueuePriorities = qprio};
            up_fam = t; up_shared = 0;
        } else
            for (uint32_t k = 0; k < nqi && up_shared; k++)
                if (qf[qis[k].queueFamilyIndex].queueCount > qis[k].queueCount) {
                    up_fam = qis[k].queueFamilyIndex; up_idx = qis[k].queueCount++; up_shared = 0;
                }
    }
    /* Pressure-proofing extensions (both optional, detected at runtime):
     * memory_priority ranks allocations for the kernel's eviction order,
     * memory_budget exposes how much VRAM a new allocation can still take. */
    const char *dext[6]; uint32_t ndext = 0;
    int has_cm = 0, has_ssc = 0;
    {
        uint32_t ne = 0;
        vkEnumerateDeviceExtensionProperties(G.phys, NULL, &ne, NULL);
        VkExtensionProperties *ep = ne ? malloc(ne * sizeof(*ep)) : NULL;
        if (ep) {
            vkEnumerateDeviceExtensionProperties(G.phys, NULL, &ne, ep);
            for (uint32_t i = 0; i < ne; i++) {
#ifdef VK_EXT_memory_priority
                if (!strcmp(ep[i].extensionName, VK_EXT_MEMORY_PRIORITY_EXTENSION_NAME)) G.has_prio = 1;
#endif
#ifdef VK_EXT_memory_budget
                if (!strcmp(ep[i].extensionName, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME)) G.has_budget = 1;
#endif
#ifdef VK_EXT_external_memory_host
                if (!strcmp(ep[i].extensionName, VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME)) G.has_hostmem = 1;
#endif
#ifdef VK_KHR_cooperative_matrix
                if (!strcmp(ep[i].extensionName, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME)) has_cm = 1;
                if (!strcmp(ep[i].extensionName, VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME)) has_ssc = 1;
#endif
            }
            free(ep);
        }
    }
    /* Cooperative-matrix GEMM (optional, COLI_VK_COOP=0 turns it off): the KHR extension
     * with a 16x16x16 fp16 x fp16 -> fp32 subgroup shape, shaderFloat16, the Vulkan
     * memory model, and a subgroup size the pipeline can require. All of it or none. */
#ifdef VK_KHR_cooperative_matrix
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR cmf = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
    VkPhysicalDeviceSubgroupSizeControlFeaturesEXT sscf = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES_EXT, .pNext = &cmf};
    VkPhysicalDeviceVulkan12Features v12f = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, .pNext = &sscf};
    {
        const char *e = getenv("COLI_VK_COOP");
        VkPhysicalDeviceProperties pp; vkGetPhysicalDeviceProperties(G.phys, &pp);
        if (has_cm && has_ssc && pp.apiVersion >= VK_API_VERSION_1_2 && !(e && *e == '0')) {
            VkPhysicalDeviceFeatures2 f2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &v12f};
            vkGetPhysicalDeviceFeatures2(G.phys, &f2);
            VkPhysicalDeviceSubgroupSizeControlPropertiesEXT ssp = {
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES_EXT};
            VkPhysicalDeviceProperties2 p2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &ssp};
            vkGetPhysicalDeviceProperties2(G.phys, &p2);
            int shape = 0;
            PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR cmp = (PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)
                vkGetInstanceProcAddr(G.inst, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");
            uint32_t nc = 0;
            if (cmp && cmp(G.phys, &nc, NULL) == VK_SUCCESS && nc) {
                VkCooperativeMatrixPropertiesKHR *cp = calloc(nc, sizeof(*cp));
                if (cp) {
                    for (uint32_t i = 0; i < nc; i++) cp[i].sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
                    if (cmp(G.phys, &nc, cp) == VK_SUCCESS)
                        for (uint32_t i = 0; i < nc; i++)
                            if (cp[i].MSize == 16 && cp[i].NSize == 16 && cp[i].KSize == 16 &&
                                cp[i].AType == VK_COMPONENT_TYPE_FLOAT16_KHR && cp[i].BType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                                cp[i].CType == VK_COMPONENT_TYPE_FLOAT32_KHR && cp[i].ResultType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                                cp[i].scope == VK_SCOPE_SUBGROUP_KHR) shape = 1;
                    free(cp);
                }
            }
            /* Subgroups of 64 where the device allows (RDNA3: the wave64 MMA measured
             * fastest), else the largest it can require; COLI_VK_COOP_SG overrides. */
            int sgz = ssp.maxSubgroupSize >= 64 && ssp.minSubgroupSize <= 64 ? 64 : (int)ssp.maxSubgroupSize;
            const char *se = getenv("COLI_VK_COOP_SG");
            if (se && atoi(se) >= (int)ssp.minSubgroupSize && atoi(se) <= (int)ssp.maxSubgroupSize) sgz = atoi(se);
            if (shape && cmf.cooperativeMatrix && v12f.shaderFloat16 && v12f.vulkanMemoryModel &&
                sscf.subgroupSizeControl && sscf.computeFullSubgroups &&
                (ssp.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT) && sgz >= 16) {
                G.has_coop = 1; G.coop_sg = sgz;
                /* the grouped expert GEMM (qmatmul_grp.comp) addresses the weights;
                 * COLI_VK_BDA=0 leaves it off */
                const char *be = getenv("COLI_VK_BDA");
                G.has_bda = v12f.bufferDeviceAddress && !(be && *be == '0');
            }
        }
    }
    /* enable exactly what the cooperative-matrix shader needs, nothing else */
    memset(&v12f, 0, sizeof(v12f)); memset(&sscf, 0, sizeof(sscf)); memset(&cmf, 0, sizeof(cmf));
    cmf.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR; cmf.cooperativeMatrix = VK_TRUE;
    sscf.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES_EXT; sscf.pNext = &cmf;
    sscf.subgroupSizeControl = VK_TRUE; sscf.computeFullSubgroups = VK_TRUE;
    v12f.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES; v12f.pNext = &sscf;
    v12f.shaderFloat16 = VK_TRUE; v12f.vulkanMemoryModel = VK_TRUE;
    v12f.bufferDeviceAddress = G.has_bda ? VK_TRUE : VK_FALSE;
#endif
    VkDeviceCreateInfo di = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = nqi, .pQueueCreateInfos = qis};
#ifdef VK_EXT_memory_priority
    VkPhysicalDeviceMemoryPriorityFeaturesEXT prif = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PRIORITY_FEATURES_EXT,
        .memoryPriority = VK_TRUE};
    if (G.has_prio) { dext[ndext++] = VK_EXT_MEMORY_PRIORITY_EXTENSION_NAME; prif.pNext = (void *)di.pNext; di.pNext = &prif; }
#endif
#ifdef VK_EXT_memory_budget
    if (G.has_budget) dext[ndext++] = VK_EXT_MEMORY_BUDGET_EXTENSION_NAME;
#endif
#ifdef VK_EXT_external_memory_host
    /* host memory the device reads in place (no copy): measured by the harness
     * (COLI_VK_TEST_HOSTMEM), see docs/vulkan.md on integrated GPUs */
    if (G.has_hostmem) dext[ndext++] = VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME;
#endif
    di.enabledExtensionCount = ndext; di.ppEnabledExtensionNames = ndext ? dext : NULL;
    G.prio = 0.75f;                              /* default class: dense/resident weights */
#ifdef VK_KHR_cooperative_matrix
    if (G.has_coop) {   /* a device that refuses the extra features still comes up without them */
        VkDeviceCreateInfo dc = di;
        const char *cext[8]; uint32_t nc = 0;
        for (uint32_t i = 0; i < ndext; i++) cext[nc++] = dext[i];
        cext[nc++] = VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME;
        cext[nc++] = VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME;
        cmf.pNext = (void *)di.pNext; dc.pNext = &v12f;
        dc.enabledExtensionCount = nc; dc.ppEnabledExtensionNames = cext;
        if (vkCreateDevice(G.phys, &dc, NULL, &G.dev) != VK_SUCCESS) { G.has_coop = 0; G.has_bda = 0; G.dev = VK_NULL_HANDLE; }
    }
    if (!G.dev)
#endif
    VKCHECK(vkCreateDevice(G.phys, &di, NULL, &G.dev), "vkCreateDevice");
    vk_mem_device(0, G.phys, G.dev);   /* the device memory books, and COLI_VK_DEVICE_CAP_MB */
    vk_pool_blocks(0);
    vkGetDeviceQueue(G.dev, G.qfam, 0, &G.queue);
    vkGetDeviceQueue(G.dev, G.tq_fam, G.tq_idx, &G.tqueue);
    G.tq_shared = G.tqueue == G.queue;
    if (g_up[0].on) {
        VkUp *u = &g_up[0];
        u->dev = G.dev; u->fam = up_fam; u->shared = up_shared;
        if (up_shared) u->q = G.queue; else vkGetDeviceQueue(G.dev, up_fam, up_idx, &u->q);
        uint32_t fs[3] = {G.qfam, G.tq_fam, up_fam};
        for (int k = 0; k < 3; k++) {
            int seen = 0;
            for (uint32_t j = 0; j < u->nfams; j++) seen |= u->fams[j] == fs[k];
            if (!seen) u->fams[u->nfams++] = fs[k];
        }
    }
    {   /* what the expert batch's scratch offsets must respect */
        VkPhysicalDeviceProperties pp; vkGetPhysicalDeviceProperties(G.phys, &pp);
        G.ssbo_align = (size_t)pp.limits.minStorageBufferOffsetAlignment;
        G.ssbo_range = (size_t)pp.limits.maxStorageBufferRange;
        G.ts_period = pp.limits.timestampPeriod;
        /* a weight range starts at this alignment (D3D12 behind Dozen: 64 KiB) */
        VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = 4096,
            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
        VkBuffer probe;
        G.buf_align = 256;
        if (vkCreateBuffer(G.dev, &bi, NULL, &probe) == VK_SUCCESS) {
            VkMemoryRequirements req; vkGetBufferMemoryRequirements(G.dev, probe, &req);
            if (req.alignment > G.buf_align) G.buf_align = (size_t)req.alignment;
            vkDestroyBuffer(G.dev, probe, NULL);
        }
    }
    if (G.has_prio || G.has_budget)
        fprintf(stderr, "[VK] VRAM pressure-proofing: memory_priority %s, memory_budget %s\n",
                G.has_prio ? "on" : "absent", G.has_budget ? "on" : "absent");

    int mt = pick_memtype(G.phys);
    if (mt < 0) { fprintf(stderr, "[VK] no host-visible memory\n"); return 0; }
    G.memtype = (uint32_t)mt;
    G.memtype_cached = (uint32_t)pick_memtype_cached(G.phys);
    G.memtype_dev = (uint32_t)pick_memtype_device(G.phys);

    /* Resizable-BAR sanity (#523): on discrete cards the weight tiers want
     * HOST_VISIBLE|DEVICE_LOCAL. With ReBAR disabled that combination exists only in a
     * ~256 MB BAR window (or not at all), so tier allocations silently land in system
     * RAM and every access crosses PCIe — measurably SLOWER than the CPU path, while
     * the resident-experts log still reports an apparently healthy VRAM tier. Staged
     * uploads (VkUp) are the answer: the warnings below are for a device where they are
     * off (COLI_VK_STAGED=0, or no staging buffer). */
    {
        VkPhysicalDeviceMemoryProperties mp;
        vkGetPhysicalDeviceMemoryProperties(G.phys, &mp);
        VkDeviceSize dl_max = 0;
        for (uint32_t i = 0; i < mp.memoryHeapCount; i++)
            if ((mp.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) &&
                mp.memoryHeaps[i].size > dl_max) dl_max = mp.memoryHeaps[i].size;
        VkMemoryPropertyFlags cf = mp.memoryTypes[G.memtype].propertyFlags;
        VkDeviceSize hv_dl = mp.memoryHeaps[mp.memoryTypes[G.memtype].heapIndex].size;
        if (g_up[0].on && !up_init(&g_up[0])) {
            fprintf(stderr, "[VK] staged uploads unavailable (no staging buffer): resident data stays in mapped memory\n");
            up_destroy(&g_up[0]);
            g_up[0].on = g_up[0].shared = 0;
        }
        up_import_init(&g_up[0], G.phys, G.has_hostmem, "");
        if (g_up[0].on) {
            VkUp *u = &g_up[0];
            fprintf(stderr, "[VK] memory: staged uploads, resident data in device-local memory (type %u, %llu MiB heap) "
                    "copied from host staging memory (type %u) on %s (%s)\n", u->mt_dev,
                    (unsigned long long)(mp.memoryHeaps[mp.memoryTypes[u->mt_dev].heapIndex].size >> 20), u->mt_stage,
                    u->shared ? "the main queue" : u->fam != G.qfam && u->fam != G.tq_fam ? "a transfer queue" : "a queue of its own",
                    u->why);
        } else if (dl_max && !(cf & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
            fprintf(stderr, "[VK] warning: no host-visible+device-local memory type — weight tiers "
                    "will live in system RAM and every access crosses PCIe (expect slower than "
                    "CPU-only). On a discrete card, enable Resizable BAR in the BIOS or leave "
                    "COLI_VK_STAGED unset.\n");
        else if (dl_max && hv_dl * 4 < dl_max)
            fprintf(stderr, "[VK] warning: only %llu of %llu MB VRAM is host-visible (Resizable BAR "
                    "appears disabled) — allocations beyond the %llu MB window fall back to system "
                    "RAM and will be slow. Enable Resizable BAR / Smart Access Memory in the BIOS, "
                    "or leave COLI_VK_STAGED unset.\n",
                    (unsigned long long)(hv_dl >> 20), (unsigned long long)(dl_max >> 20),
                    (unsigned long long)(hv_dl >> 20));
    }

    G.shader = load_spv(G.dev, spv_path);
    if (!G.shader) return 0;
    snprintf(G.spv_path, sizeof G.spv_path, "%s", spv_path);
    if (!build_pipeline(G.dev, 4, sizeof(struct PC), G.shader, &G.dsl, &G.plyt, &G.pipe, &G.dpool, &G.dset)) return 0;

    /* Optional tiled GEMMs: an absent shader or a tile the device cannot hold leaves
     * that slot empty; without the fp32 one every S stays on the GEMV. The cooperative-
     * matrix one only comes up on devices that passed the checks above, and alongside
     * the fp32 one, which takes the formats it does not. */
    {
        VkPhysicalDeviceProperties pp; vkGetPhysicalDeviceProperties(G.phys, &pp);
        char gm_path[512]; derive_sibling(spv_path, "_gemm.spv", gm_path, sizeof(gm_path));
        G.shader_gemm = load_spv(G.dev, gm_path);
        const char *e = getenv("COLI_VK_GEMM_TILE");
        VkGemmTile u = {0, 0, 0, 0, 0, 0};
        int one = e && *e && sscanf(e, "%d,%d,%d,%d,%d,%d", &u.bm, &u.bn, &u.bk, &u.tm, &u.tn, &u.pf) >= 5 &&
                  gemm_tile_ok(u, &pp.limits);
        if (e && *e && !one) fprintf(stderr, "[VK] COLI_VK_GEMM_TILE=%s rejected\n", e);
        for (int k = 0; G.shader_gemm && k < (one ? 1 : VK_GEMM_SLOTS); k++) {
            VkGemmTile t = one ? u : gemm_tiles[k];
            if (gemm_tile_ok(t, &pp.limits)) gemm_pipeline(t, k);
        }
        if (G.shader_gemm && !G.pipe_gemm[0]) {
            vkDestroyShaderModule(G.dev, G.shader_gemm, NULL); G.shader_gemm = VK_NULL_HANDLE;
            for (int k = 1; k < VK_GEMM_SLOTS; k++)
                if (G.pipe_gemm[k]) { vkDestroyPipeline(G.dev, G.pipe_gemm[k], NULL); G.pipe_gemm[k] = VK_NULL_HANDLE; }
        }
        if (G.has_coop && G.pipe_gemm[0]) {
            char cm_path[512]; derive_sibling(spv_path, "_coop.spv", cm_path, sizeof(cm_path));
            G.shader_coop = load_spv(G.dev, cm_path);
            e = getenv("COLI_VK_COOP_TILE");
            VkCoopTile c = {0, 0, 0, 0, 0};
            one = e && *e && sscanf(e, "%d,%d,%d,%d,%d", &c.bm, &c.bn, &c.wm, &c.wn, &c.bk) == 5 &&
                  coop_tile_ok(c, &pp.limits);
            if (e && *e && !one) fprintf(stderr, "[VK] COLI_VK_COOP_TILE=%s rejected\n", e);
            /* every slot shares BK: coop_fmt checks group sizes against it */
            for (int k = 0; G.shader_coop && k < (one ? 1 : VK_COOP_SLOTS); k++) {
                VkCoopTile t = one ? c : coop_tiles[k];
                if (coop_tile_ok(t, &pp.limits)) coop_pipeline(t, k);
            }
        }
        if (!G.pipe_coop[0]) {
            for (int k = 1; k < VK_COOP_SLOTS; k++)
                if (G.pipe_coop[k]) { vkDestroyPipeline(G.dev, G.pipe_coop[k], NULL); G.pipe_coop[k] = VK_NULL_HANDLE; }
            if (G.shader_coop) vkDestroyShaderModule(G.dev, G.shader_coop, NULL);
            G.shader_coop = VK_NULL_HANDLE; G.has_coop = 0;
        }
        /* Where the GEMM overtakes the GEMV, measured on the 780M over Qwen3.8's shapes
         * (int8, int4-g64, bf16): from S = 2 on a wide matrix (O = 2560: 0.65 against
         * 0.90 ms in int8, 1.0 against 2.2 in bf16), but a narrow one fills too few
         * tiles (O = 48: the GEMV wins up to S = 32, loses from 128). The rule that
         * fits: S >= 2 and S*O >= 4096. COLI_VK_GEMM_MIN_S=N puts every S >= N on the
         * GEMM, 0 none; S == 1 (decode) always stays on the GEMV. */
        const char *m = getenv("COLI_VK_GEMM_MIN_S");
        G.gemm_min_s = !G.pipe_gemm[0] ? 0 : m && *m ? atoi(m) : 2;
        G.gemm_min_so = m && *m ? 0 : 4096;
        if (G.gemm_min_s == 1) G.gemm_min_s = 2;
        if (G.gemm_min_s < 0) G.gemm_min_s = 0;
    }

    /* Optional fused gate+up pipeline: skip gracefully if its shader isn't present
     * (single-matmul path keeps working). */
    char gu_path[512]; derive_sibling(spv_path, "_gate_up.spv", gu_path, sizeof(gu_path));
    G.shader_gu = load_spv(G.dev, gu_path);
    if (G.shader_gu && !build_pipeline(G.dev, 6, sizeof(struct PCGU), G.shader_gu, &G.dsl_gu, &G.plyt_gu, &G.pipe_gu, &G.dpool_gu, &G.dset_gu))
        return 0;

    /* Optional MLA absorb attention pipeline (same directory as the main shader). */
    /* Optional rmsnorm pipeline: enables the pair->norm->q_b single-submit chain
     * (coli_vk_attn_qprep); absent -> callers keep the 3-submit path. */
    char nrm_path[512]; derive_dir_file(spv_path, "rmsnorm.spv", nrm_path, sizeof(nrm_path));
    G.shader_nrm = load_spv(G.dev, nrm_path);
    if (G.shader_nrm) {
        VkDescriptorPool np; VkDescriptorSet ns;
        if (!build_pipeline(G.dev, 3, sizeof(struct PCN), G.shader_nrm, &G.dsl_nrm, &G.plyt_nrm, &G.pipe_nrm, &np, &ns))
            return 0;
        G.dset_nrm = ns;
        /* one extra 4-binding matmul set for the chain's 3rd matmul (dset+dset_pair serve 1+2) */
        VkDescriptorPoolSize ps3 = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 4};
        VkDescriptorPoolCreateInfo dpi3 = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &ps3};
        VKCHECK(vkCreateDescriptorPool(G.dev, &dpi3, NULL, &G.qprep_pool), "qprep descPool");
        VkDescriptorSetAllocateInfo dsa3 = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = G.qprep_pool, .descriptorSetCount = 1, .pSetLayouts = &G.dsl};
        VKCHECK(vkAllocateDescriptorSets(G.dev, &dsa3, &G.dset_qp3), "qprep descSet");
    }
    char att_path[512]; derive_dir_file(spv_path, "attention_absorb.spv", att_path, sizeof(att_path));
    G.shader_att = load_spv(G.dev, att_path);
    if (G.shader_att && !build_pipeline(G.dev, 7, sizeof(struct PCAttn), G.shader_att, &G.dsl_att, &G.plyt_att, &G.pipe_att, &G.dpool_att, &G.dset_att))
        return 0;

    VkCommandPoolCreateInfo cpci = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = G.qfam};
    VKCHECK(vkCreateCommandPool(G.dev, &cpci, NULL, &G.cpool), "cmdPool");
    VkCommandBufferAllocateInfo cbi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = G.cpool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
    VKCHECK(vkAllocateCommandBuffers(G.dev, &cbi, &G.cmd), "cmdBuf");
    VKCHECK(vkAllocateCommandBuffers(G.dev, &cbi, &G.eg_cmd), "eg cmdBuf");
    VkFenceCreateInfo fi = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VKCHECK(vkCreateFence(G.dev, &fi, NULL, &G.fence), "fence");
    VKCHECK(vkCreateFence(G.dev, &fi, NULL, &G.eg_fence), "eg fence");

    G.ready = 1;
    if ((g_vk_prof = getenv("VK_PROF") != NULL)) atexit(vk_prof_paths);
    VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(G.phys, &p);
    fprintf(stderr, "[VK] ready: %s, compute qfam %u, memtype %u%s%s", p.deviceName, G.qfam, G.memtype,
            G.shader_gu ? ", fused gate+up" : "", G.shader_att ? ", absorb attention" : "");
    if (G.gemm_min_s) fprintf(stderr, ", tiled GEMM from S=%d%s%s", G.gemm_min_s,
                              G.gemm_min_so ? " and S*O>=4096" : "", G.pipe_coop[0] ? " (cooperative matrix)" : "");
    if (g_up[0].on) fprintf(stderr, ", staged uploads");
    fprintf(stderr, "\n");
    return 1;
}

int coli_vk_available(void) { return G.ready; }

void coli_vk_mem_info(size_t *used, size_t *count) {
    if (used) *used = G.used_bytes;
    if (count) *count = G.tensor_count;
}

/* Weight memory: many tensors share a few big VkDeviceMemory blocks.
 * WHY blocks: per-submit driver cost measures LINEAR in the number of distinct
 * device-memory objects the queue actively references (~0.35 ms/submit extra at a
 * 950-expert tier's ~5.7k allocations: decode attention 7.9s @420 -> 17.6s @950, flat
 * once the GPU paths moved to CPU; IDLE allocations cost nothing until first
 * referenced — harness ballast probe). Packing uploads into 256 MB blocks keeps the
 * referenced-BO count in the dozens.
 * Each tensor is a VkBuffer bound at an offset that vk_alloc.h hands out, and a
 * freed tensor gives its two ranges back (best fit, coalesced), so a tier can evict
 * and reuse the space. Three pools: the resident weights of every engine (dense,
 * GLM's registry, MiMo's experts), the expert tier's own (its budget is the pool's
 * limit, its blocks at the 0.4 eviction priority), and COLI_VK_DEV2's. A pool's
 * lock covers its books only: the uploader thread of the tier allocates while the
 * engine thread uploads dense matrices. A free while async work is in flight on the
 * tensor's device waits for that work (coli_vk_tensor_free). */
typedef struct { VkDeviceMemory mem; uint8_t *base; } VkBlk;
struct VkWPool {
    VkaPool p;
    pthread_mutex_t mx;
    int dev;                 /* 0 = G, 1 = G2 */
    float prio;              /* eviction priority of its blocks; < 0 = G.prio at creation */
    int counted;             /* 1 = its tensors are coli_vk_mem_info's (the resident weights) */
    size_t bytes, tensors;   /* payload of its live tensors (rows + scales) */
};
#define VK_WBLOCK ((size_t)256 << 20)
static VkWPool g_wpool  = {.p = {.block_bytes = VK_WBLOCK}, .mx = PTHREAD_MUTEX_INITIALIZER, .dev = 0, .prio = -1.f, .counted = 1};
static VkWPool g_tpool  = {.p = {.block_bytes = VK_WBLOCK}, .mx = PTHREAD_MUTEX_INITIALIZER, .dev = 0, .prio = 0.4f};
static VkWPool g_wpool2 = {.p = {.block_bytes = VK_WBLOCK}, .mx = PTHREAD_MUTEX_INITIALIZER, .dev = 1, .prio = -1.f};
static VkWPool g_tpool2 = {.p = {.block_bytes = VK_WBLOCK}, .mx = PTHREAD_MUTEX_INITIALIZER, .dev = 1, .prio = -1.f};   /* the tier's experts on COLI_VK_DEV2 */
/* the tier's extra layers (an MTP head's), whose experts have another size than the
 * main ones': a pool of their own, so neither leaves holes the other cannot use */
static VkWPool g_xpool  = {.p = {.block_bytes = VK_WBLOCK}, .mx = PTHREAD_MUTEX_INITIALIZER, .dev = 0, .prio = 0.4f};
/* Device k's pools take VK_WBLOCK blocks, smaller ones under COLI_VK_DEVICE_CAP_MB. */
static void vk_pool_blocks(int k) {
    size_t bb = VK_WBLOCK;
    if (g_mem.cap[k]) { bb = (size_t)64 << 10; while ((uint64_t)bb < g_mem.cap[k] / 4096 && bb < VK_WBLOCK) bb <<= 1; }
    if (k == 0) { g_wpool.p.block_bytes = bb; g_tpool.p.block_bytes = bb; g_xpool.p.block_bytes = bb; }
    else { g_wpool2.p.block_bytes = bb; g_tpool2.p.block_bytes = bb; }
}
static VkDevice pool_device(const VkWPool *P);
static uint32_t pool_memtype(const VkWPool *P);
static int pool_has_prio(const VkWPool *P);
static VkResult vk_fence_wait(VkDevice dev, VkFence f);
static void up_lost(int dev);

/* ---- the uploader at work (staged uploads, VkUp above) ---------------------------
 * The caller holds u->mx from its first up_add to its up_finish. The bytes go through
 * the slots: a slot fills up with copies and is submitted, the next one fills while it
 * runs, and up_finish submits the last and waits for every slot, so what was added is
 * on the device when it returns (its fence waited: any queue may read it next).
 * Failures: a command buffer that would not record or a submit refused fails this upload
 * only (up_finish still waits for the slots already sent, so the caller may free the
 * tensors, and the next upload starts clean); a fence wait that fails means the copy may
 * still run: the device is taken as lost (up_lost), as for every other wait. */
typedef void (*UpFill)(uint8_t *dst, size_t off, size_t n, const void *ctx);
static int up_fail(VkUp *u, VkResult r, const char *what) {
    if (!u->failed) { u->err = r; u->what = what; }
    u->failed = 1;
    if (r == VK_ERROR_DEVICE_LOST) u->lost = 1;
    return 0;
}
static int up_wait(VkUp *u, int s) {
    if (!u->pending[s]) { up_imp_release(u, s); return 1; }   /* never submitted: its imports go */
    u->pending[s] = 0;
    VkResult r = vk_fence_wait(u->dev, u->fence[s]);
    if (r == VK_SUCCESS && up_fault("wait")) r = VK_TIMEOUT;
    up_imp_release(u, s);   /* the copies are done (or the device is gone) */
    if (r == VK_SUCCESS) return 1;
    up_fail(u, r, "fence wait");
    u->lost = 1;
    return 0;
}
static int up_open(VkUp *u) {
    if (u->failed || u->lost) return 0;
    if (u->open) return 1;
    if (!up_wait(u, u->cur)) return 0;
    VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    VkResult r = vkResetCommandBuffer(u->cmd[u->cur], 0);
    if (r == VK_SUCCESS && up_fault("record")) r = VK_ERROR_OUT_OF_HOST_MEMORY;
    if (r == VK_SUCCESS) r = vkBeginCommandBuffer(u->cmd[u->cur], &bi);
    if (r != VK_SUCCESS) return up_fail(u, r, "command buffer");
    u->open = 1; u->used = 0;
    return 1;
}
static int up_submit(VkUp *u, int dev) {
    if (!u->open) return !u->failed && !u->lost;
    VkCommandBuffer c = u->cmd[u->cur];
    u->open = 0;
    if (u->failed) { vkEndCommandBuffer(c); return 0; }   /* out of recording; never submitted */
    /* the copies' writes made available to every later access, on whatever queue */
    VkMemoryBarrier mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT};
    vkCmdPipelineBarrier(c, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &c};
    VkResult r = vkEndCommandBuffer(c);
    if (r == VK_SUCCESS && up_fault("record")) r = VK_ERROR_OUT_OF_HOST_MEMORY;
    if (r != VK_SUCCESS) return up_fail(u, r, "command buffer");
    r = vkResetFences(u->dev, 1, &u->fence[u->cur]);
    if (r == VK_SUCCESS && up_fault("submit")) r = VK_ERROR_OUT_OF_DEVICE_MEMORY;
    if (r == VK_SUCCESS) r = vk_submit(dev, u->q, &si, u->fence[u->cur]);
    if (r != VK_SUCCESS) return up_fail(u, r, "submit");
    u->pending[u->cur] = 1; u->submits++;
    u->cur = (u->cur + 1) % VK_UP_SLOTS;
    return 1;
}
/* `bytes` that fill() writes, copied to dst at dst_off */
static int up_add(VkUp *u, int dev, VkBuffer dst, size_t dst_off, size_t bytes, UpFill fill, const void *ctx) {
    for (size_t off = 0; off < bytes; ) {
        if (!up_open(u)) return 0;
        size_t room = VK_UP_SLOT - u->used, n = bytes - off < room ? bytes - off : room;
        size_t so = (size_t)u->cur * VK_UP_SLOT + u->used;
        fill(u->sptr + so, off, n, ctx);
        VkBufferCopy c = {so, dst_off + off, n};
        vkCmdCopyBuffer(u->cmd[u->cur], u->sbuf, dst, 1, &c);
        u->used = (u->used + n + 255) & ~(size_t)255;
        u->bytes += n; u->copies++; off += n;
        if (u->used >= VK_UP_SLOT && !up_submit(u, dev)) return 0;
    }
    return 1;
}
/* `bytes` at src, imported in place as a transfer source in the slot being recorded: *buf
 * holds them at *off. The import spans the whole pages around them; it lives until the
 * slot's fence (up_wait), so src must outlive up_finish. 0 = not here (no imports on this
 * device, the driver refused these pages, COLI_VK_STAGED_FAULT=import): the caller stages. */
static int up_import(VkUp *u, const void *src, size_t bytes, VkBuffer *buf, VkDeviceSize *off) {
#ifdef VK_EXT_external_memory_host
    if (!u->imp_align || !u->imp_props || !bytes || !up_open(u)) return 0;
    if (up_fault("import")) { u->imp_refused++; return 0; }
    size_t al = u->imp_align;
    uintptr_t p = (uintptr_t)src, base = p / al * al, end = (p + bytes + al - 1) / al * al;
    size_t sz = end - base;
    VkMemoryHostPointerPropertiesEXT mp = {.sType = VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
    if (((PFN_vkGetMemoryHostPointerPropertiesEXT)u->imp_props)(u->dev, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
                                                                (void *)base, &mp) != VK_SUCCESS || !mp.memoryTypeBits) {
        u->imp_refused++; return 0;
    }
    VkExternalMemoryBufferCreateInfo eb = {.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT};
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .pNext = &eb, .size = sz,
        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT, .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    VkBuffer b;
    if (vkCreateBuffer(u->dev, &bi, NULL, &b) != VK_SUCCESS) { u->imp_refused++; return 0; }
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(u->dev, b, &req);
    uint32_t bits = mp.memoryTypeBits & req.memoryTypeBits, mt = 0;
    if (!bits || req.size > sz) { vkDestroyBuffer(u->dev, b, NULL); u->imp_refused++; return 0; }
    while (!(bits & (1u << mt))) mt++;
    VkImportMemoryHostPointerInfoEXT imp = {.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT,
        .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, .pHostPointer = (void *)base};
    VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .pNext = &imp, .allocationSize = sz,
        .memoryTypeIndex = mt};
    VkDeviceMemory m;
    if (vkAllocateMemory(u->dev, &ai, NULL, &m) != VK_SUCCESS) { vkDestroyBuffer(u->dev, b, NULL); u->imp_refused++; return 0; }
    if (vkBindBufferMemory(u->dev, b, m, 0) != VK_SUCCESS) {
        vkDestroyBuffer(u->dev, b, NULL); vkFreeMemory(u->dev, m, NULL); u->imp_refused++; return 0;
    }
    int s = u->cur;
    if (u->nimp[s] == u->cimp[s]) {
        int nc = u->cimp[s] ? 2 * u->cimp[s] : 16;
        void *n = realloc(u->imp[s], (size_t)nc * sizeof(*u->imp[s]));
        if (!n) { vkDestroyBuffer(u->dev, b, NULL); vkFreeMemory(u->dev, m, NULL); u->imp_refused++; return 0; }
        u->imp[s] = n; u->cimp[s] = nc;
    }
    u->imp[s][u->nimp[s]].buf = b; u->imp[s][u->nimp[s]].mem = m; u->nimp[s]++;
    *buf = b; *off = (VkDeviceSize)(p - base);
    return 1;
#else
    (void)u; (void)src; (void)bytes; (void)buf; (void)off;
    return 0;
#endif
}
/* Copies of n ranges of one host region (src: base, bytes in all), each region's
 * src_off[i], len[i] to dst[i] at dst_off[i], straight from the imported pages. 0 = not
 * imported (the caller stages); nothing was recorded then. */
static int up_add_host(VkUp *u, const void *src, size_t bytes, int n, const VkBuffer *dst, const size_t *dst_off,
                       const size_t *src_off, const size_t *len) {
    VkBuffer b; VkDeviceSize o;
    if (!up_import(u, src, bytes, &b, &o)) return 0;
    for (int i = 0; i < n; i++) {
        if (!len[i]) continue;
        VkBufferCopy c = {o + src_off[i], dst_off[i], len[i]};
        vkCmdCopyBuffer(u->cmd[u->cur], b, dst[i], 1, &c);
        u->imp_bytes += len[i]; u->imp_copies++;
    }
    return 1;
}
/* Always called, also after a failed up_add: the slots sent are waited for. 1 = all on
 * the device; 0 = this upload failed (u->lost: and the device with it). */
static int up_finish(VkUp *u, int dev) {
    int ok = up_submit(u, dev);
    for (int s = 0; s < VK_UP_SLOTS; s++) ok &= up_wait(u, s);
    ok = ok && !u->failed && !u->lost;
    if (!u->lost) u->failed = 0;   /* the next upload starts clean */
    return ok;
}
/* An upload failed (u->what says where): the device lost with it, or only this one. */
static void up_failed(int dev, const char *what) {
    VkUp *u = &g_up[dev];
    if (u->lost) { up_lost(dev); return; }
    fprintf(stderr, "[VK] %sstaged upload failed (%s: %d): %s\n", dev ? "dev2 " : "",
            u->what ? u->what : "?", (int)u->err, what);
}
typedef struct { const uint8_t *w; size_t cpu_rb, stride; } UpRows;
/* the rows at their padded stride, zeros past each row's bytes */
static void up_fill_rows(uint8_t *dst, size_t off, size_t n, const void *ctx) {
    const UpRows *r = ctx;
    for (size_t end = off + n; off < end; ) {
        size_t o = off / r->stride, in = off - o * r->stride, take = r->stride - in;
        if (take > end - off) take = end - off;
        size_t data = in < r->cpu_rb ? r->cpu_rb - in : 0;
        if (data > take) data = take;
        if (data) memcpy(dst, r->w + o * r->cpu_rb + in, data);
        if (take > data) memset(dst + data, 0, take - data);
        dst += take; off += take;
    }
}
static void up_fill_bytes(uint8_t *dst, size_t off, size_t n, const void *ctx) {
    memcpy(dst, (const uint8_t *)ctx + off, n);
}
/* A buffer of the device-local tensors: every family that touches them shares it. */
static void up_sharing(const VkUp *u, VkBufferCreateInfo *bi) {
    if (u->nfams > 1) {
        bi->sharingMode = VK_SHARING_MODE_CONCURRENT;
        bi->queueFamilyIndexCount = u->nfams; bi->pQueueFamilyIndices = u->fams;
    }
}
/* A fresh device-local block, filled with zeros before its first tensor. A contributor
 * measured on an RX 580 (RADV, Polaris) that without this first touch the results
 * read from a fresh block differed slightly and from run to run (#1338); the fill value
 * did not matter. Cheap: one fill per 256 MB block. A failure only skips the fill. */
static void up_zero(int dev, VkDeviceMemory mem, uint64_t cap) {
    VkUp *u = &g_up[dev];
    if (!u->fill) return;
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = cap,
        .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT, .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    up_sharing(u, &bi);
    VkBuffer b;
    if (vkCreateBuffer(u->dev, &bi, NULL, &b) != VK_SUCCESS) return;
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(u->dev, b, &req);
    int lost = 0;
    if (req.size <= cap && (req.memoryTypeBits & (1u << u->mt_dev)) && vkBindBufferMemory(u->dev, b, mem, 0) == VK_SUCCESS) {
        pthread_mutex_lock(&u->mx);
        if (up_open(u)) vkCmdFillBuffer(u->cmd[u->cur], b, 0, VK_WHOLE_SIZE, 0);
        if (up_finish(u, dev)) u->blocks_filled++;
        else up_failed(dev, "the block is used unfilled");
        lost = u->lost;
        pthread_mutex_unlock(&u->mx);
    }
    if (!lost) vkDestroyBuffer(u->dev, b, NULL);   /* a failed wait: the fill may still run, leave it */
}

/* The primary device's pools are addressable when bufferDeviceAddress is on (the
 * expert batch's grouped GEMM reads expert weights through an address table). */
static int pool_bda(const VkWPool *P) { return P->dev == 0 && G.has_bda; }
/* Block memory for a pool (lock held). */
static VkBlk *pool_new_block(VkWPool *P, uint64_t cap) {
    VkBlk *bk = calloc(1, sizeof(*bk));
    if (!bk) return NULL;
    VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = cap, .memoryTypeIndex = pool_memtype(P)};
#ifdef VK_EXT_memory_priority
    VkMemoryPriorityAllocateInfoEXT pri = {.sType = VK_STRUCTURE_TYPE_MEMORY_PRIORITY_ALLOCATE_INFO_EXT,
        .priority = P->prio >= 0.f ? P->prio : G.prio};
    if (pool_has_prio(P)) ai.pNext = &pri;
#endif
    VkMemoryAllocateFlagsInfo afl = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
        .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT};
    if (pool_bda(P)) { afl.pNext = ai.pNext; ai.pNext = &afl; }
    VkDevice dev = pool_device(P);
    int staged = g_up[P->dev].on;   /* device-local, never mapped: the uploader fills it */
    if (staged && up_fault("block")) { free(bk); return NULL; }
    if (vkAllocateMemory(dev, &ai, NULL, &bk->mem) != VK_SUCCESS ||
        (!staged && vkMapMemory(dev, bk->mem, 0, cap, 0, (void **)&bk->base) != VK_SUCCESS)) {
        if (bk->mem) vkFreeMemory(dev, bk->mem, NULL);
        free(bk); return NULL;
    }
    if (staged) up_zero(P->dev, bk->mem, cap);
    return bk;
}
static void pool_release_block(VkWPool *P, int k) {   /* lock held, block empty */
    VkBlk *bk = P->p.b[k].user;
    VkDevice dev = pool_device(P);
    if (bk) { if (bk->base) vkUnmapMemory(dev, bk->mem); vkFreeMemory(dev, bk->mem, NULL); free(bk); }
    vka_pool_drop_block(&P->p, k);
}

/* A buffer of `bytes` bound inside one of the pool's blocks; *ptr is its mapping.
 * 0 when the device has no room or the pool is at its limit. */
static int pool_suballoc(VkWPool *P, size_t bytes, VkBuffer *buf, void **ptr, VkaRange *r) {
    VkDevice dev = pool_device(P);
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = bytes ? bytes : 4, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    if (g_up[P->dev].on) { bi.usage |= VK_BUFFER_USAGE_TRANSFER_DST_BIT; up_sharing(&g_up[P->dev], &bi); }
    if (pool_bda(P)) bi.usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    VKCHECK(vkCreateBuffer(dev, &bi, NULL, buf), "vkCreateBuffer");
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(dev, *buf, &req);
    if (!(req.memoryTypeBits & (1u << pool_memtype(P)))) { vkDestroyBuffer(dev, *buf, NULL); *buf = VK_NULL_HANDLE; return 0; }
    uint64_t align = req.alignment ? req.alignment : 256;
    pthread_mutex_lock(&P->mx);
    int ok = vka_alloc(&P->p, req.size, align, r);
    if (!ok) {
        /* a new block: the usual size, or what is left under the limit when that is
         * smaller but still holds this request (a budget is rarely a block multiple) */
        uint64_t cap = vka_block_size_for(&P->p, req.size);
        if (P->p.limit && P->p.total + cap > P->p.limit && P->p.limit > P->p.total &&
            P->p.limit - P->p.total >= req.size + align)
            cap = (P->p.limit - P->p.total) & ~(uint64_t)4095;
        VkBlk *bk = vka_pool_can_grow(&P->p, cap) && cap >= req.size ? pool_new_block(P, cap) : NULL;
        if (bk) {
            if (vka_pool_add_block(&P->p, cap, bk) < 0) {
                if (bk->base) vkUnmapMemory(dev, bk->mem);
                vkFreeMemory(dev, bk->mem, NULL); free(bk);
            } else ok = vka_alloc(&P->p, req.size, align, r);
        } else P->p.refusals++;
    }
    VkBlk *blk = ok ? P->p.b[r->block].user : NULL;
    pthread_mutex_unlock(&P->mx);
    if (!ok) { vkDestroyBuffer(dev, *buf, NULL); *buf = VK_NULL_HANDLE; return 0; }
    if (vkBindBufferMemory(dev, *buf, blk->mem, r->off) != VK_SUCCESS) {
        vkDestroyBuffer(dev, *buf, NULL); *buf = VK_NULL_HANDLE;
        pthread_mutex_lock(&P->mx);
        if (vka_free(&P->p, *r)) pool_release_block(P, r->block);
        pthread_mutex_unlock(&P->mx);
        return 0;
    }
    if (ptr) *ptr = blk->base ? blk->base + r->off : NULL;
    return 1;
}
static void pool_free_range(VkWPool *P, VkaRange r) {
    if (!r.len) return;
    pthread_mutex_lock(&P->mx);
    if (vka_free(&P->p, r)) pool_release_block(P, r.block);
    pthread_mutex_unlock(&P->mx);
}
static void pool_destroy(VkWPool *P) {
    pthread_mutex_lock(&P->mx);
    VkDevice dev = pool_device(P);
    for (int k = 0; k < P->p.nb; k++) {
        if (!P->p.b[k].present) continue;
        VkBlk *bk = P->p.b[k].user;
        if (bk) { if (dev) { if (bk->base) vkUnmapMemory(dev, bk->mem); vkFreeMemory(dev, bk->mem, NULL); } free(bk); }
    }
    uint64_t bb = P->p.block_bytes, lim = P->p.limit;
    vka_pool_destroy(&P->p);
    P->p.block_bytes = bb ? bb : VK_WBLOCK; P->p.limit = lim;
    P->bytes = P->tensors = 0;
    pthread_mutex_unlock(&P->mx);
}

/* Lay out a tensor's two ranges in pool P: rows at their padded stride, zeroed, and
 * the scales; *wptr and *sptr are the mappings to fill. With staged uploads the ranges
 * are device-local: *wptr and *sptr are then a zeroed host image of the same layout
 * that coli_vk_tensor_commit copies over (or, wptr NULL, nothing: upload_tensor_pool
 * streams the rows itself). */
static ColiVkTensor *tensor_alloc(VkWPool *P, int fmt, int I, int O, int gs, void **wptr, void **sptr) {
    ColiVkTensor *t = calloc(1, sizeof(*t));
    if (!t) return NULL;
    t->fmt = fmt; t->I = I; t->O = O; t->rowWords = rowwords(fmt, I);
    t->gs = (fmt == 4 || fmt == 7 || fmt == 12 || fmt == 13) ? gs : 0;
    t->dev = P->dev; t->pool = P;
    t->wbytes = (size_t)t->rowWords * 4 * (size_t)O;
    size_t sbytes = scale_floats(fmt, I, O, gs) * sizeof(float);
    void *wp = NULL, *sp = NULL;
    if (!pool_suballoc(P, t->wbytes, &t->wbuf, &wp, &t->wr)) { free(t); return NULL; }
    if (!pool_suballoc(P, sbytes, &t->sbuf, &sp, &t->sr)) {
        vkDestroyBuffer(pool_device(P), t->wbuf, NULL); pool_free_range(P, t->wr); free(t); return NULL;
    }
    if (g_up[P->dev].on) {
        if (wptr && !(t->img = up_img_alloc(P->dev, t->wbytes + sbytes, &t->img_al))) {
            vkDestroyBuffer(pool_device(P), t->wbuf, NULL); vkDestroyBuffer(pool_device(P), t->sbuf, NULL);
            pool_free_range(P, t->wr); pool_free_range(P, t->sr); free(t); return NULL;
        }
        if (wptr) { *wptr = t->img; *sptr = t->img + t->wbytes; }
    } else {
        *wptr = wp; *sptr = sp;
        memset(*wptr, 0, t->wbytes);
        t->wmap = wp; t->smap = sp;
    }
    __atomic_add_fetch(&P->bytes, t->wbytes + sbytes, __ATOMIC_RELAXED);
    __atomic_add_fetch(&P->tensors, 1, __ATOMIC_RELAXED);
    if (P->counted) {
        // Counters are touched concurrently: frees run from expert_load under
        // `#pragma omp parallel`, so RMW them atomically (torn counts otherwise).
        __atomic_add_fetch(&G.used_bytes, t->wbytes + sbytes, __ATOMIC_RELAXED);
        __atomic_add_fetch(&G.tensor_count, 1, __ATOMIC_RELAXED);
    }
    return t;
}
static int fmt_uploadable(int fmt, int gs) {
    return fmt == 1 || fmt == 2 || fmt == 5 || fmt == 10 || fmt == 11 || fmt == 14 ||   /* fmt=4/7: word-aligned groups only */
           ((fmt == 4 || fmt == 7) && gs >= 8 && gs % 8 == 0) ||
           ((fmt == 12 || fmt == 13) && gs >= 4 && gs % 4 == 0);           /* fp8 / int8: 4 per word */
}

static void tensor_release(ColiVkTensor *t);
static int upload_tensor_pool(VkWPool *P, ColiVkTensor **out, const void *weights, const float *scales,
                              int fmt, int I, int O, int gs) {
    if (*out) return (*out)->fmt == fmt && (*out)->I == I && (*out)->O == O;
    if (!fmt_uploadable(fmt, gs)) return 0;
    if (g_up[P->dev].on) {   /* staged: the rows and the scales through the uploader */
        ColiVkTensor *t = tensor_alloc(P, fmt, I, O, gs, NULL, NULL);
        if (!t) return 0;
        static const float one = 1.0f;   /* float weights: no scales */
        UpRows rows = {weights, cpu_row_bytes(fmt, I), (size_t)t->rowWords * 4};
        size_t sb = scale_floats(fmt, I, O, gs) * sizeof(float);
        VkUp *u = &g_up[P->dev];
        pthread_mutex_lock(&u->mx);
        /* rows without padding: straight from the weights when their pages import */
        size_t z = 0;
        int ok = ((rows.cpu_rb == rows.stride && up_add_host(u, weights, t->wbytes, 1, &t->wbuf, &z, &z, &t->wbytes)) ||
                  up_add(u, P->dev, t->wbuf, 0, t->wbytes, up_fill_rows, &rows)) &&
                 up_add(u, P->dev, t->sbuf, 0, sb, up_fill_bytes, fmt == 10 || fmt == 11 || fmt == 14 ? (const void *)&one : scales);
        ok = up_finish(u, P->dev) && ok;
        pthread_mutex_unlock(&u->mx);
        if (!ok) {   /* the matrix stays on the CPU (with the device, if it was lost) */
            up_failed(P->dev, "a matrix stays on the CPU");
            tensor_release(t);
            return 0;
        }
        *out = t;
        return 1;
    }
    void *wptr, *sptr;
    ColiVkTensor *t = tensor_alloc(P, fmt, I, O, gs, &wptr, &sptr);
    if (!t) return 0;
    size_t stride = (size_t)t->rowWords * 4;         // padded row bytes
    size_t cpu_rb = cpu_row_bytes(fmt, I);
    for (int o = 0; o < O; o++)                        // copy row-by-row into padded layout
        memcpy((uint8_t *)wptr + (size_t)o * stride,
               (const uint8_t *)weights + (size_t)o * cpu_rb, cpu_rb);
    size_t sfl = scale_floats(fmt, I, O, gs);            // fmt=5: O*ceil(I/64) group scales
    if (fmt == 10 || fmt == 11 || fmt == 14) ((float *)sptr)[0] = 1.0f;   /* float weights: no scales */
    else memcpy(sptr, scales, sfl * sizeof(float));
    *out = t;
    return 1;
}
static int upload_tensor(ColiVkTensor **out, const void *weights, const float *scales,
                         int fmt, int I, int O, int gs) {
    return upload_tensor_pool(&g_wpool, out, weights, scales, fmt, I, O, gs);
}

/* Upload a resident tensor without computing (for the expert tier: gate/up/down are
 * uploaded once, then driven by coli_vk_expert_group). Returns 0 on failure. */
int coli_vk_tensor_ensure(ColiVkTensor **tensor, const void *weights, const float *scales, int fmt, int I, int O, int grp) {
    if (!G.ready) return 0;
    return upload_tensor(tensor, weights, scales, fmt, I, O, grp);
}

/* ---- resident rows read in place (VK_EXT_external_memory_host) ----------------------
 * On a device that shares the CPU's RAM a resident copy is a second copy of the same
 * bytes: a 27B trunk held twice does not fit beside the host's on a 64 GB box. An import
 * hands the device the host's own rows instead: the pages are wrapped in a VkDeviceMemory
 * of the type vkGetMemoryHostPointerPropertiesEXT allows and a buffer is bound to it at
 * offset 0, so the shaders read the matrix where the engine keeps it. The scales (a few
 * floats a row) are copied into the weight pool as usual. Only where the rows need no
 * padding (the shaders read rowWords words a row: the CPU's row length must already be a
 * whole number of words) and the allocation is aligned to the device's import alignment
 * and spans whole units of it. Not with staged uploads: those are for a card whose memory
 * is not the host's. */
static ColiVkTensor *g_imports;            /* live imports, freed at shutdown */
static pthread_mutex_t g_imports_mx = PTHREAD_MUTEX_INITIALIZER;
static size_t g_import_bytes;
size_t coli_vk_import_alignment(void) {
#ifdef VK_EXT_external_memory_host
    if (!G.ready || !G.has_hostmem || g_up[0].on) return 0;
    static size_t al;
    if (!al) {
        VkPhysicalDeviceExternalMemoryHostPropertiesEXT hp = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT};
        VkPhysicalDeviceProperties2 p2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &hp};
        vkGetPhysicalDeviceProperties2(G.phys, &p2);
        al = hp.minImportedHostPointerAlignment ? (size_t)hp.minImportedHostPointerAlignment : 4096;
    }
    return al;
#else
    return 0;
#endif
}
size_t coli_vk_imported_bytes(void) { return __atomic_load_n(&g_import_bytes, __ATOMIC_RELAXED); }
/* A host range read in place by a shader (the chain's VKC_HOST): the pages around it,
 * so the caller's allocation need not be aligned; the pages past its ends are the
 * process's own and are never written. */
int coli_vk_host_buffer(const void *ptr, size_t bytes, void **buf_out, void **mem_out, size_t *off) {
#ifdef VK_EXT_external_memory_host
    size_t al = coli_vk_import_alignment();
    if (!al || !ptr || !bytes) return 0;
    uintptr_t p = (uintptr_t)ptr, base = p / al * al, end = (p + bytes + al - 1) / al * al;
    size_t sz = end - base;
    if (sz > G.ssbo_range) return 0;
    static PFN_vkGetMemoryHostPointerPropertiesEXT gp;
    if (!gp) gp = (PFN_vkGetMemoryHostPointerPropertiesEXT)vkGetDeviceProcAddr(G.dev, "vkGetMemoryHostPointerPropertiesEXT");
    VkMemoryHostPointerPropertiesEXT mp = {.sType = VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
    if (!gp || gp(G.dev, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, (void *)base, &mp) != VK_SUCCESS ||
        !mp.memoryTypeBits) return 0;
    VkExternalMemoryBufferCreateInfo eb = {.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT};
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .pNext = &eb, .size = sz,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    VkBuffer b;
    if (vkCreateBuffer(G.dev, &bi, NULL, &b) != VK_SUCCESS) return 0;
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(G.dev, b, &req);
    uint32_t bits = mp.memoryTypeBits & req.memoryTypeBits, mt = 0;
    if (!bits || req.size > sz) { vkDestroyBuffer(G.dev, b, NULL); return 0; }
    while (!(bits & (1u << mt))) mt++;
    VkImportMemoryHostPointerInfoEXT imp = {.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT,
        .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, .pHostPointer = (void *)base};
    VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .pNext = &imp, .allocationSize = sz,
        .memoryTypeIndex = mt};
    VkDeviceMemory m;
    if (vkAllocateMemory(G.dev, &ai, NULL, &m) != VK_SUCCESS) { vkDestroyBuffer(G.dev, b, NULL); return 0; }
    if (vkBindBufferMemory(G.dev, b, m, 0) != VK_SUCCESS) { vkDestroyBuffer(G.dev, b, NULL); vkFreeMemory(G.dev, m, NULL); return 0; }
    *buf_out = (void *)b; *mem_out = (void *)m; *off = (size_t)(p - base);
    __atomic_add_fetch(&g_import_bytes, sz, __ATOMIC_RELAXED);
    return 1;
#else
    (void)ptr; (void)bytes; (void)buf_out; (void)mem_out; (void)off;
    return 0;
#endif
}
void coli_vk_host_buffer_free(void *buf, void *mem, size_t bytes) {
    if (!G.dev) return;
    if (buf) vkDestroyBuffer(G.dev, (VkBuffer)buf, NULL);
    if (mem) vkFreeMemory(G.dev, (VkDeviceMemory)mem, NULL);
    if (bytes) __atomic_sub_fetch(&g_import_bytes, bytes, __ATOMIC_RELAXED);
}
int coli_vk_tensor_import(ColiVkTensor **tensor, const void *weights, size_t alloc_bytes, const float *scales,
                          int fmt, int I, int O, int gs) {
    if (*tensor) return (*tensor)->fmt == fmt && (*tensor)->I == I && (*tensor)->O == O;
#ifdef VK_EXT_external_memory_host
    size_t al = coli_vk_import_alignment();
    if (!al || !fmt_uploadable(fmt, gs) || O < 1 || I < 1) return 0;
    size_t rb = cpu_row_bytes(fmt, I);
    if (rb != (size_t)rowwords(fmt, I) * 4) return 0;          /* rows the shaders would read padded */
    size_t wbytes = rb * (size_t)O, sz = (wbytes + al - 1) / al * al;
    if ((uintptr_t)weights % al || sz > alloc_bytes || wbytes > G.ssbo_range) return 0;
    static PFN_vkGetMemoryHostPointerPropertiesEXT gp;
    if (!gp) gp = (PFN_vkGetMemoryHostPointerPropertiesEXT)vkGetDeviceProcAddr(G.dev, "vkGetMemoryHostPointerPropertiesEXT");
    VkMemoryHostPointerPropertiesEXT mp = {.sType = VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
    if (!gp || gp(G.dev, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, weights, &mp) != VK_SUCCESS ||
        !mp.memoryTypeBits) return 0;
    VkExternalMemoryBufferCreateInfo eb = {.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT};
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .pNext = &eb, .size = wbytes,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    VkBuffer buf;
    if (vkCreateBuffer(G.dev, &bi, NULL, &buf) != VK_SUCCESS) return 0;
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(G.dev, buf, &req);
    uint32_t bits = mp.memoryTypeBits & req.memoryTypeBits, mt = 0;
    if (!bits) { vkDestroyBuffer(G.dev, buf, NULL); return 0; }
    while (!(bits & (1u << mt))) mt++;
    VkImportMemoryHostPointerInfoEXT imp = {.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT,
        .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, .pHostPointer = (void *)weights};
    VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .pNext = &imp, .allocationSize = sz,
        .memoryTypeIndex = mt};
    VkDeviceMemory mem;
    if (vkAllocateMemory(G.dev, &ai, NULL, &mem) != VK_SUCCESS) { vkDestroyBuffer(G.dev, buf, NULL); return 0; }
    if (vkBindBufferMemory(G.dev, buf, mem, 0) != VK_SUCCESS) {
        vkDestroyBuffer(G.dev, buf, NULL); vkFreeMemory(G.dev, mem, NULL); return 0;
    }
    ColiVkTensor *t = calloc(1, sizeof(*t));
    size_t sbytes = scale_floats(fmt, I, O, gs) * sizeof(float);
    void *sp = NULL;
    if (!t || !pool_suballoc(&g_wpool, sbytes, &t->sbuf, &sp, &t->sr) || !sp) {
        if (t && t->sbuf) { vkDestroyBuffer(G.dev, t->sbuf, NULL); pool_free_range(&g_wpool, t->sr); }
        free(t); vkDestroyBuffer(G.dev, buf, NULL); vkFreeMemory(G.dev, mem, NULL); return 0;
    }
    t->fmt = fmt; t->I = I; t->O = O; t->rowWords = rowwords(fmt, I);
    t->gs = (fmt == 4 || fmt == 7 || fmt == 12 || fmt == 13) ? gs : 0;
    t->dev = 0; t->pool = &g_wpool; t->wbuf = buf; t->wbytes = wbytes; t->imp = mem;
    if (fmt == 10 || fmt == 11 || fmt == 14) ((float *)sp)[0] = 1.0f;
    else memcpy(sp, scales, sbytes);
    __atomic_add_fetch(&g_wpool.bytes, sbytes, __ATOMIC_RELAXED);
    __atomic_add_fetch(&g_wpool.tensors, 1, __ATOMIC_RELAXED);
    __atomic_add_fetch(&G.used_bytes, sbytes, __ATOMIC_RELAXED);
    __atomic_add_fetch(&G.tensor_count, 1, __ATOMIC_RELAXED);
    __atomic_add_fetch(&g_import_bytes, wbytes, __ATOMIC_RELAXED);
    pthread_mutex_lock(&g_imports_mx);
    t->imp_next = g_imports; g_imports = t;
    pthread_mutex_unlock(&g_imports_mx);
    *tensor = t;
    return 1;
#else
    (void)tensor; (void)weights; (void)alloc_bytes; (void)scales; (void)fmt; (void)I; (void)O; (void)gs;
    return 0;
#endif
}

/* Sync-path fence wait. A blocked vkWaitForFences pays a scheduler wake on
 * signal (~50-150 us) — and the engine fences ~2 sync submits per layer per
 * token, so the wakes alone cost seconds per run. Spin on vkGetFenceStatus for
 * a short budget first (the common decode dispatch completes in 0.5-2 ms),
 * then fall back to the blocking wait. The spinning thread is stalled on the
 * GPU result anyway. COLI_VK_SPIN_US=0 restores the pure blocking wait. */
static long g_vk_spin_us = -1;
static VkResult vk_fence_wait_us(VkDevice dev, VkFence f, long spin_us) {
    if (spin_us > 0) {
        double t0 = vk_now();
        do {
            VkResult r = vkGetFenceStatus(dev, f);
            if (r != VK_NOT_READY) return r;   /* VK_SUCCESS or a real error */
        } while ((vk_now() - t0) * 1000.0 < (double)spin_us);
    }
    return vkWaitForFences(dev, 1, &f, VK_TRUE, 10000000000ULL);
}
static long vk_spin_us(void) {
    if (g_vk_spin_us < 0) {
        const char *e = getenv("COLI_VK_SPIN_US");
        g_vk_spin_us = e ? atol(e) : 300;
        if (g_vk_spin_us < 0) g_vk_spin_us = 0;
    }
    return g_vk_spin_us;
}
static VkResult vk_fence_wait(VkDevice dev, VkFence f) { return vk_fence_wait_us(dev, f, vk_spin_us()); }
/* A tiled GEMM runs for milliseconds, past the spin budget, and the blocked wait then
 * wakes up to a millisecond late: measured on the 780M, a 0.6 ms S=32 GEMM took 0.6 or
 * 1.05 ms per call at random. Its wait spins through the dispatch, bounded at 50 ms;
 * COLI_VK_SPIN_US=0 still blocks. */
static VkResult vk_fence_wait_gemm(VkDevice dev, VkFence f) {
    return vk_fence_wait_us(dev, f, vk_spin_us() > 0 ? 50000 : 0);
}

/* Global submit/wait totals across EVERY synchronous GPU path (VK_PROF=1) — the
 * per-path counters miss traffic that flows through the fused pair/absorb/group
 * entries, so the tier-size-linear per-submit tax is localized here instead. */
static double g_vsub_ms, g_vwait_ms; static long g_vsub_n;
static void vkprof_tick(void) {
    if ((++g_vsub_n & 2047) == 0)
        fprintf(stderr, "[VK_PROF sub] n=%ld | submit %.0f | wait %.0f ms\n", g_vsub_n, g_vsub_ms, g_vwait_ms);
}

static unsigned long long g_vk_matmul_calls;   /* successful coli_vk_matmul calls */
static unsigned long long g_vk_gemm_calls;     /* ... of which through a tiled GEMM */
static unsigned long long g_vk_coop_calls;     /* ... of which the cooperative-matrix one */

/* VK_PROF=1: at exit, how the matmuls and their wall time (upload to readback) split
 * between the GEMV and the two tiled GEMMs. */
static double g_vk_path_ms[3];
static unsigned long long g_vk_path_n[3];
static void vk_prof_paths(void) {
    fprintf(stderr, "[VK_PROF] %llu matmuls: GEMV %llu in %.0f ms, fp32 GEMM %llu in %.0f ms, "
            "cooperative-matrix GEMM %llu in %.0f ms\n", g_vk_matmul_calls,
            g_vk_path_n[0], g_vk_path_ms[0], g_vk_path_n[1], g_vk_path_ms[1], g_vk_path_n[2], g_vk_path_ms[2]);
}

/* Formats the cooperative-matrix GEMM takes: decoded weights exact in fp16, and group
 * boundaries on its staged steps (one group scale per step). */
static int coop_fmt(const ColiVkTensor *t) {
    int gs = t->fmt == 5 ? 64 : t->gs;
    return t->fmt == 1 || t->fmt == 2 ||
           ((t->fmt == 4 || t->fmt == 5 || t->fmt == 7 || t->fmt == 12) && gs % G.coop_t[0].bk == 0);
}
/* Its x upload: the rows, then their powers of two rs[s] = 2^(15-e) with
 * max|x[s,:]| < 2^e, so a scaled row stays under 2^15 (fp16 max 65504) and its hi/lo
 * fp16 split keeps 22 bits. One pass copies and takes the integer max of the magnitude
 * bits (vectorizes without fast-math); a row holding Inf or NaN keeps rs = 1 and its
 * non-finite result, as on the GEMV. */
static void coop_stage_x(float *restrict dst, const float *restrict x, int S, int I) {
    float *rs = dst + (size_t)S * I;
    for (int s = 0; s < S; s++) {
        const float *restrict r = x + (size_t)s * I;
        float *restrict d = dst + (size_t)s * I;
        uint32_t m = 0;
        for (int i = 0; i < I; i++) {
            uint32_t u; memcpy(&u, &r[i], 4);
            d[i] = r[i]; u &= 0x7fffffffu; m = u > m ? u : m;
        }
        int e = (int)(m >> 23) - 126;                 /* m < 2^e for a normal m */
        if (m >= 0x7f800000u) e = 15;
        if (e < -100) e = -100;                       /* zero and subnormal rows */
        uint32_t b = (uint32_t)(15 - e + 127) << 23;
        memcpy(&rs[s], &b, sizeof(b));
    }
}

int coli_vk_matmul(ColiVkTensor **tensor, float *y, const float *x,
                   const void *weights, const float *scales,
                   int fmt, int S, int I, int O, int gs) {
    if (!G.ready || S < 1 || !upload_tensor(tensor, weights, scales, fmt, I, O, gs)) return 0;
    ColiVkTensor *t = *tensor;
    /* VK_PROF=1: phase split of the dense per-call cost, printed every 8192 calls —
     * separates our code (memcpy/desc/record) from the driver (submit) and the GPU
     * (fence wait) to localize the tier-size-linear tax. */
    static double p_x, p_desc, p_rec, p_sub, p_wait, p_y; static long p_n;
    double t0 = G.eg_prof ? vk_now() : 0, tA, tp = g_vk_prof ? vk_now() : 0;
    /* Prefill-sized S takes a tiled GEMM: one BM x BN tile of y per workgroup, so each
     * weight is fetched ceil(S/BN) times instead of S; the cooperative-matrix one where
     * the device and the format allow (path 2), else the fp32 one (1). Decode (S = 1),
     * and a matrix too narrow to fill the device at this S, keep the GEMV (0). */
    int path = 0;
    if (G.gemm_min_s && S >= G.gemm_min_s && (int64_t)S * O >= G.gemm_min_so)
        path = G.pipe_coop[0] && !G.coop_off && coop_fmt(t) ? 2 : 1;
    size_t xb = (size_t)S * I * sizeof(float), yb = (size_t)S * O * sizeof(float);
    VkBuffer old_x = G.x.buf, old_y = G.y.buf;
    if (!scratch_reserve(&G.x, xb + (path == 2 ? (size_t)S * sizeof(float) : 0)) ||   /* + rs[S] */
        !scratch_reserve_mt(&G.y, yb, G.memtype_cached)) return 0;  /* y read back */
    if (path == 2) coop_stage_x(G.x.ptr, x, S, I);
    else memcpy(G.x.ptr, x, xb);
    if (G.eg_prof) { tA = vk_now(); p_x += tA - t0; t0 = tA; }

    /* Rebind descriptors only when the tensor or a scratch buffer changed (a realloc
     * makes the old VkBuffer handle stale); otherwise the previous binding is still valid. */
    int rebind = G.bound_tensor != t || G.x.buf != old_x || G.y.buf != old_y
              || G.bound_xbuf != G.x.buf || G.bound_ybuf != G.y.buf;
    if (rebind) {
        VkDescriptorBufferInfo bi[4] = {
            {.buffer = G.x.buf, .range = VK_WHOLE_SIZE},
            {.buffer = t->wbuf, .range = VK_WHOLE_SIZE},
            {.buffer = t->sbuf, .range = VK_WHOLE_SIZE},
            {.buffer = G.y.buf, .range = VK_WHOLE_SIZE}};
        VkWriteDescriptorSet w[4];
        for (int i = 0; i < 4; i++) w[i] = (VkWriteDescriptorSet){
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = G.dset,
            .dstBinding = i, .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &bi[i]};
        vkUpdateDescriptorSets(G.dev, 4, w, 0, NULL);
        G.bound_tensor = t; G.bound_xbuf = G.x.buf; G.bound_ybuf = G.y.buf;
    }
    if (G.eg_prof) { tA = vk_now(); p_desc += tA - t0; t0 = tA; }

    /* Re-record the command buffer only when the binding or the dispatch shape changed.
     * Recorded WITHOUT one-time-submit so the same buffer can be resubmitted verbatim —
     * for repeated calls to the same expert this drops setup to a bare submit+wait. */
    if (rebind || !G.cmd_ready || G.bound_S != S || G.bound_I != I || G.bound_O != O || G.bound_gemm != path) {
        VKCHECK(vkResetCommandBuffer(G.cmd, 0), "resetCmd");
        VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        VKCHECK(vkBeginCommandBuffer(G.cmd, &begin), "beginCmd");
        int slot = path == 2 ? coop_slot(S) : path ? gemm_slot(S) : 0;
        vkCmdBindPipeline(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          path == 2 ? G.pipe_coop[slot] : path ? G.pipe_gemm[slot] : G.pipe);
        vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt, 0, 1, &G.dset, 0, NULL);
        struct PC pc = {fmt, S, I, O, t->rowWords, t->gs};
        vkCmdPushConstants(G.cmd, G.plyt, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        if (path) {
            int bm = path == 2 ? G.coop_t[slot].bm : G.gemm_t[slot].bm;
            int bn = path == 2 ? G.coop_t[slot].bn : G.gemm_t[slot].bn;
            vkCmdDispatch(G.cmd, (uint32_t)((O + bm - 1) / bm), (uint32_t)((S + bn - 1) / bn), 1);
        } else
            /* Grid-stride shader: one subgroup per output row (~8 rows/workgroup at wave32).
             * Launch ~O/8 workgroups for occupancy; the shader loops to cover any O / wave width. */
            vkCmdDispatch(G.cmd, (uint32_t)((O + 7) / 8), (uint32_t)S, 1);
        VKCHECK(vkEndCommandBuffer(G.cmd), "endCmd");
        G.cmd_ready = 1; G.bound_S = S; G.bound_I = I; G.bound_O = O; G.bound_gemm = path;
    }
    if (G.eg_prof) { tA = vk_now(); p_rec += tA - t0; t0 = tA; }

    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1, .pCommandBuffers = &G.cmd};
    VKCHECK(vkResetFences(G.dev, 1, &G.fence), "resetFence");
    VKCHECK(vk_submit(0, G.queue, &si, G.fence), "queueSubmit");
    if (G.eg_prof) { tA = vk_now(); p_sub += tA - t0; g_vsub_ms += tA - t0; t0 = tA; }
    // Bounded wait: a GPU hang/TDR must never wedge the process. 10s is orders of
    // magnitude over a single-GEMV dispatch; on timeout/device-loss disable VK for
    // the rest of the run and fall back to CPU (the caller degrades on our 0 return).
    VkResult wr = path ? vk_fence_wait_gemm(G.dev, G.fence) : vk_fence_wait(G.dev, G.fence);
    if (wr != VK_SUCCESS) {
        fprintf(stderr, "[VK] fence wait failed: %d — disabling GPU offload, staying on CPU\n", wr);
        G.ready = 0;
        return 0;
    }
    if (G.eg_prof) { tA = vk_now(); p_wait += tA - t0; g_vwait_ms += tA - t0; t0 = tA; vkprof_tick(); }
    memcpy(y, G.y.ptr, yb);
    if (G.eg_prof) {
        p_y += vk_now() - t0;
        if ((++p_n & 8191) == 0)
            fprintf(stderr, "[VK_PROF dense] n=%ld | memcpy_x %.0f | desc %.0f | record %.0f | submit %.0f | wait %.0f | memcpy_y %.0f ms\n",
                    p_n, p_x, p_desc, p_rec, p_sub, p_wait, p_y);
    }
    g_vk_matmul_calls++;
    g_vk_gemm_calls += path != 0;
    g_vk_coop_calls += path == 2;
    if (g_vk_prof) { g_vk_path_ms[path] += vk_now() - tp; g_vk_path_n[path]++; }
    return 1;
}

unsigned long long coli_vk_matmul_calls(void) { return g_vk_matmul_calls; }

/* The shader path every engine resolves the same way (#523): COLI_VK_SHADERS may be the
 * qmatmul.spv file or the directory holding it; unset, the shaders/ directory next to the
 * binary (the build layout), then the historical path relative to the working directory. */
const char *coli_vk_shader_path(char *buf, size_t n) {
    const char *env = getenv("COLI_VK_SHADERS");
    struct stat st;
    if (env && *env) {
        if (!stat(env, &st) && S_ISDIR(st.st_mode)) { snprintf(buf, n, "%s/qmatmul.spv", env); return buf; }
        return env;
    }
#ifdef __linux__
    ssize_t k = readlink("/proc/self/exe", buf, n - 1);
    if (k > 0) {
        buf[k] = 0;
        char *sl = strrchr(buf, '/');
        if (sl && (size_t)(sl + 1 - buf) + sizeof("shaders/qmatmul.spv") <= n) {
            strcpy(sl + 1, "shaders/qmatmul.spv");
            if (!stat(buf, &st)) return buf;
        }
    }
#endif
    return "shaders/qmatmul.spv";
}

/* The dense matrices' place (coli_vk_dense_decide): the last decision, and the one
 * last printed (-1 none). */
static int g_dense_on = 1, g_dense_said = -1;
static int dense_choice(int tier_on, int def, char *why, size_t n) {
    const char *e = getenv("COLI_VK_DENSE");
    if (e && *e) { snprintf(why, n, "COLI_VK_DENSE=%s", e); return atoi(e) != 0; }
    if (def && tier_on && coli_vk_device_shares_ram()) {
        snprintf(why, n, "%s shares the CPU's RAM and the expert tier is on; COLI_VK_DENSE=1 puts them on the device",
                 coli_vk_device_integrated() ? "an integrated GPU" : "a CPU device");
        return 0;
    }
    snprintf(why, n, "default");
    return def != 0;
}
int coli_vk_dense_decide(const char *engine, int tier_on, int def) {
    char why[192];
    g_dense_on = dense_choice(tier_on, def, why, sizeof why);
    if (engine && g_dense_on != g_dense_said) {
        fprintf(stderr, "[VK] %s: dense matrices on the %s (%s)\n", engine, g_dense_on ? "device" : "CPU", why);
        g_dense_said = g_dense_on;
    }
    return g_dense_on;
}
int coli_vk_dense(void) { return g_dense_on; }

/* The dense weights' host copies (coli_vk_dense_host_decide): device only or not, and
 * what the engine dropped and read back since. Counters are touched from parallel
 * regions (a CPU fallback may reload inside one): atomics. */
static int g_dho, g_dho_atexit, g_dho_on_device;
static char g_dho_engine[32];
static unsigned long long g_dho_drop_n, g_dho_drop_b, g_dho_load_n, g_dho_load_b;
static int g_dho_lay_dev, g_dho_lay_all;   /* a partial chain: the first lay_dev of lay_all layers dropped theirs */
/* " (the n of L layers ...)" for the lines, empty unless the engine said a partial chain */
static const char *dho_layers_note(char *buf, size_t n, int placed) {
    buf[0] = 0;
    if (g_dho_lay_all > 0 && g_dho_lay_dev < g_dho_lay_all) {
        if (placed) snprintf(buf, n, " (the %d of %d layers on the device; the %d on the CPU keep theirs)",
                             g_dho_lay_dev, g_dho_lay_all, g_dho_lay_all - g_dho_lay_dev);
        else snprintf(buf, n, ", the %d of %d layers on the device", g_dho_lay_dev, g_dho_lay_all);
    }
    return buf;
}
static double dho_gib(unsigned long long b) { return (double)b / 1073741824.0; }
/* A size for the [VK] lines: GiB from 1 GiB, MiB below (a test fixture's KiB show as 0.0 MiB). */
static const char *dho_size(unsigned long long b, char *buf, size_t n) {
    if (b >= 1073741824ull) snprintf(buf, n, "%.2f GiB", dho_gib(b));
    else snprintf(buf, n, "%.1f MiB", (double)b / 1048576.0);
    return buf;
}
/* This process's resident set in GiB (Linux; -1 elsewhere). */
static double dho_rss_gib(void) {
#ifdef __linux__
    FILE *f = fopen("/proc/self/statm", "r");
    long pages = -1, rss = -1;
    if (f) { if (fscanf(f, "%ld %ld", &pages, &rss) != 2) rss = -1; fclose(f); }
    long pg = sysconf(_SC_PAGESIZE);
    return rss >= 0 && pg > 0 ? (double)rss * (double)pg / 1073741824.0 : -1.0;
#else
    return -1.0;
#endif
}
static void dho_exit_report(void) {
    double rss = dho_rss_gib();
    char r[48] = "";
    if (rss >= 0) snprintf(r, sizeof r, "; RSS %.2f GiB", rss);
    if (!g_dho) {   /* the same line with the host copies kept, for a comparison of the two */
        fprintf(stderr, "[VK] %s: dense weights at exit: on the device and in host RAM%s\n", g_dho_engine, r);
        return;
    }
    char a[32], b[32], ln[96];
    fprintf(stderr, "[VK] %s: dense weights at exit: %llu matrices on the device only (%s of host copies "
            "dropped%s), %llu read back from disk for the CPU (%s)%s\n", g_dho_engine,
            __atomic_load_n(&g_dho_drop_n, __ATOMIC_RELAXED), dho_size(__atomic_load_n(&g_dho_drop_b, __ATOMIC_RELAXED), a, sizeof a),
            dho_layers_note(ln, sizeof ln, 0),
            __atomic_load_n(&g_dho_load_n, __ATOMIC_RELAXED), dho_size(__atomic_load_n(&g_dho_load_b, __ATOMIC_RELAXED), b, sizeof b), r);
}
static int dho_choice(int dense_on_device, size_t dense_bytes, char *why, size_t n) {
    const char *e = getenv("COLI_VK_DENSE_HOST");
    if (!dense_on_device) {
        snprintf(why, n, "the dense part runs on the CPU%s", e && *e && atoi(e) == 0 ? ", so COLI_VK_DENSE_HOST=0 has nothing to drop" : "");
        return 0;
    }
    if (e && *e) { snprintf(why, n, "COLI_VK_DENSE_HOST=%s", e); return atoi(e) == 0; }
    if (coli_vk_device_integrated()) {
        snprintf(why, n, "an integrated GPU: its memory is the CPU's RAM, the host copy would hold them twice; "
                 "COLI_VK_DENSE_HOST=1 keeps it");
        return 1;
    }
    if (coli_vk_device_shares_ram()) {
        snprintf(why, n, "a CPU device; COLI_VK_DENSE_HOST=0 drops the host copies");
        return 0;
    }
    /* a discrete GPU: device only when the dense weights fit what it has free, with the
     * expert tier's default reserve (1 GiB) left for scratch, mirrors and the driver */
    double used = 0, budget = 0, fit;
    if (coli_vk_mem_budget(&used, &budget)) fit = (budget - used) * 1e9;
    else fit = (double)coli_vk_device_local_bytes();
    fit -= 1073741824.0;
    if ((double)dense_bytes <= fit) {
        snprintf(why, n, "a discrete GPU with room for them: %.2f of %.2f GiB free; COLI_VK_DENSE_HOST=1 keeps the host copy",
                 dho_gib(dense_bytes), fit > 0 ? fit / 1073741824.0 : 0.0);
        return 1;
    }
    snprintf(why, n, "a discrete GPU without room for all of them (%.2f GiB, %.2f GiB free): the host copy stays the "
             "fallback; COLI_VK_DENSE_HOST=0 drops it", dho_gib(dense_bytes), fit > 0 ? fit / 1073741824.0 : 0.0);
    return 0;
}
int coli_vk_dense_host_decide(const char *engine, int dense_on_device, size_t dense_bytes) {
    char why[256];
    g_dho = G.ready && dho_choice(dense_on_device, dense_bytes, why, sizeof why);
    if (!G.ready) snprintf(why, sizeof why, "no device");
    snprintf(g_dho_engine, sizeof g_dho_engine, "%s", engine ? engine : "engine");
    if (engine) fprintf(stderr, "[VK] %s: dense weights %s (%s)\n", engine,
                        g_dho ? "on the device only" : "on the device and in host RAM", why);
    g_dho_on_device = G.ready && dense_on_device;
    if (g_dho_on_device && !g_dho_atexit) { g_dho_atexit = 1; atexit(dho_exit_report); }
    return g_dho;
}
int coli_vk_dense_device_only(void) { return g_dho; }
void coli_vk_dense_host_dropped(size_t bytes) {
    __atomic_add_fetch(&g_dho_drop_n, 1, __ATOMIC_RELAXED);
    __atomic_add_fetch(&g_dho_drop_b, (unsigned long long)bytes, __ATOMIC_RELAXED);
}
void coli_vk_dense_host_reloaded(size_t bytes) {
    __atomic_add_fetch(&g_dho_load_n, 1, __ATOMIC_RELAXED);
    __atomic_add_fetch(&g_dho_load_b, (unsigned long long)bytes, __ATOMIC_RELAXED);
}
void coli_vk_dense_host_placed(const char *engine, const char *kept) {
    if (!g_dho) return;
    double rss = dho_rss_gib();
    char r[48] = "";
    if (rss >= 0) snprintf(r, sizeof r, "; RSS now %.2f GiB", rss);
    char a[32], ln[96];
    fprintf(stderr, "[VK] %s: %llu dense matrices on the device only, %s of host RAM given back%s%s%s%s\n",
            engine, __atomic_load_n(&g_dho_drop_n, __ATOMIC_RELAXED),
            dho_size(__atomic_load_n(&g_dho_drop_b, __ATOMIC_RELAXED), a, sizeof a), dho_layers_note(ln, sizeof ln, 1),
            kept && *kept ? "; kept on the host: " : "", kept && *kept ? kept : "", r);
}
void coli_vk_dense_host_layers(int on_device, int layers) {
    g_dho_lay_dev = on_device < 0 ? 0 : on_device; g_dho_lay_all = layers < 0 ? 0 : layers;
}
unsigned long long coli_vk_dense_host_dropped_bytes(void) { return __atomic_load_n(&g_dho_drop_b, __ATOMIC_RELAXED); }

/* Where a layer's dense chain runs (vk_chain.c): the whole layer recorded into one
 * submission with the residual stream and the recurrent/KV state on the device, the
 * alternative to one synchronous coli_vk_matmul per matrix. Returns COLI_VK_CHAIN_OFF,
 * _ON (every forward) or _PREFILL (forwards of more than two rows: prompts, not decode
 * steps or an MTP verify).
 *   COLI_VK_CHAIN set and non-empty: 0 off, 2 prefill only, any other number on.
 *   Unset, COLI_VK_DENSE=0: off (the trunk stays on the CPU, as that variable says; the
 *   chain's device copy of it would also take the expert tier's budget: on an 8 GB card
 *   with qwen38 the tier got 1.87 GiB instead of 6 and decode lost 24%, #1900).
 *   Unset, a discrete GPU: on.
 *   Unset, an integrated GPU: `igpu` when the engine runs the expert tier (tier_on),
 *   else off; an engine never timed on one passes COLI_VK_CHAIN_UNMEASURED, which is
 *   off and says so. `igpu` is what the engine measured on one (a Radeon 780M, docs/vulkan.md,
 *   "The dense chain"): qwen36 passes ON (its chain won decode, 9.9 against 8.0 tok/s,
 *   and prefill, 9.5 against 12.2 s); qwen38 OFF (its chain won prefill, 30.1 against
 *   38.7 s, and lost decode, 3.2 against 3.8 tok/s: the device's GEMV at the GPU's
 *   800 MHz floor is slower than the CPU's int8 one on a 4.3 GB trunk; prompts only
 *   still lost 5% of decode, the trunk's device copy taking from the tier's budget).
 *   olmoe ON (OLMoE-1B-7B: the chain decoded 17.3 against 12.8 tok/s and prefilled 512
 *   tokens in 5.5 against 6.4 s; the CPU alone decodes 23.1 tok/s, its trunk being f32).
 *   inkling passes COLI_VK_CHAIN_UNMEASURED (no checkpoint of it runs on the box).
 *   Unset, a CPU device (Lavapipe): off.
 * Printed as a [VK] line with an engine name (NULL: silent). */
int coli_vk_chain_decide(const char *engine, int tier_on, int igpu) {
    const char *e = getenv("COLI_VK_CHAIN"), *dn = getenv("COLI_VK_DENSE");
    char why[192];
    int on;
    if (e && *e) {
        int v = atoi(e);
        on = v == 0 ? COLI_VK_CHAIN_OFF : v == 2 ? COLI_VK_CHAIN_PREFILL : COLI_VK_CHAIN_ON;
        snprintf(why, sizeof why, "COLI_VK_CHAIN=%s", e);
    } else if (dn && *dn == '0') {
        on = COLI_VK_CHAIN_OFF;
        snprintf(why, sizeof why, "COLI_VK_DENSE=0: the trunk on the CPU; COLI_VK_CHAIN=1 turns the chain on");
    } else if (coli_vk_device_integrated()) {
        int measured = igpu != COLI_VK_CHAIN_UNMEASURED;
        on = tier_on && measured ? igpu : COLI_VK_CHAIN_OFF;
        snprintf(why, sizeof why, "an integrated GPU%s: %s; COLI_VK_CHAIN=0 off, 1 on, 2 prompts only",
                 tier_on ? " with the expert tier" : " without the expert tier",
                 on == COLI_VK_CHAIN_ON ? "measured faster on decode and prefill"
                 : on == COLI_VK_CHAIN_PREFILL ? "measured faster on prefill, slower on decode"
                 : tier_on && measured ? "measured slower on decode" : "not measured");
    } else if (coli_vk_device_shares_ram()) {
        on = COLI_VK_CHAIN_OFF;
        snprintf(why, sizeof why, "a CPU device; COLI_VK_CHAIN=1 turns it on");
    } else { on = COLI_VK_CHAIN_ON; snprintf(why, sizeof why, "a discrete GPU%s", tier_on ? ", beside the expert tier" : ""); }
    if (engine) fprintf(stderr, "[VK] %s: dense chain %s (%s)\n", engine,
                        on == COLI_VK_CHAIN_ON ? "on" : on == COLI_VK_CHAIN_PREFILL ? "on for prompts" : "off", why);
    return on;
}

int coli_vk_init_env_tier(const char *engine, int tier_on) {
    const char *on = getenv("COLI_VULKAN");
    if (!on || !atoi(on)) return 0;
    char buf[1024];
    const char *spv = coli_vk_shader_path(buf, sizeof buf);
    int ok = coli_vk_init(spv) && coli_vk_available();
    if (ok) {
        char why[192];
        g_dense_on = g_dense_said = dense_choice(tier_on, 1, why, sizeof why);
        fprintf(stderr, "[VK] %s: device ready, dense matrices on the %s (%s)\n", engine,
                g_dense_on ? "device" : "CPU", why);
    } else fprintf(stderr, "[VK] %s: no usable Vulkan device (shaders %s), running on the CPU\n",
                   engine, spv);
    return ok;
}
int coli_vk_init_env(const char *engine) { return coli_vk_init_env_tier(engine, 0); }

/* Fused first half of the expert MLP: hidden = silu(gate(x)) * up(x), computed in ONE
 * dispatch that reads x once for both projections. gate/up are resident (uploaded on
 * first call). D = input (hidden) dim, I = moe_inter. Returns 0 -> caller falls back. */
int coli_vk_gate_up(ColiVkTensor **gate, ColiVkTensor **up, float *hidden, const float *x,
                    const void *gw, const float *gs, const void *uw, const float *us,
                    int fmt, int S, int D, int I, int grp) {
    if (!G.ready || !G.shader_gu || S < 1 || D > 6144) return 0;   /* shader stages x in xsh[6144] */
    if (!upload_tensor(gate, gw, gs, fmt, D, I, grp) || !upload_tensor(up, uw, us, fmt, D, I, grp)) return 0;
    ColiVkTensor *tg = *gate, *tu = *up;
    size_t xb = (size_t)S * D * sizeof(float), hb = (size_t)S * I * sizeof(float);
    if (!scratch_reserve(&G.x, xb) || !scratch_reserve_mt(&G.h, hb, G.memtype_cached)) return 0;  /* hidden read back */
    memcpy(G.x.ptr, x, xb);

    VkDescriptorBufferInfo bi[6] = {
        {.buffer = G.x.buf, .range = VK_WHOLE_SIZE}, {.buffer = tg->wbuf, .range = VK_WHOLE_SIZE},
        {.buffer = tg->sbuf, .range = VK_WHOLE_SIZE}, {.buffer = tu->wbuf, .range = VK_WHOLE_SIZE},
        {.buffer = tu->sbuf, .range = VK_WHOLE_SIZE}, {.buffer = G.h.buf, .range = VK_WHOLE_SIZE}};
    VkWriteDescriptorSet w[6];
    for (int i = 0; i < 6; i++) w[i] = (VkWriteDescriptorSet){
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = G.dset_gu,
        .dstBinding = (uint32_t)i, .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &bi[i]};
    vkUpdateDescriptorSets(G.dev, 6, w, 0, NULL);

    VKCHECK(vkResetCommandBuffer(G.cmd, 0), "resetCmd");
    VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    VKCHECK(vkBeginCommandBuffer(G.cmd, &begin), "beginCmd");
    vkCmdBindPipeline(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe_gu);
    vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt_gu, 0, 1, &G.dset_gu, 0, NULL);
    struct PCGU pc = pcgu(fmt, S, D, I, tg->rowWords, tg->gs);   // PC.I = input D, PC.O = moe_inter I
    vkCmdPushConstants(G.cmd, G.plyt_gu, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(G.cmd, (uint32_t)((I + 7) / 8), (uint32_t)S, 1);
    VKCHECK(vkEndCommandBuffer(G.cmd), "endCmd");

    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &G.cmd};
    VKCHECK(vkResetFences(G.dev, 1, &G.fence), "resetFence");
    double vp0 = G.eg_prof ? vk_now() : 0;
    VKCHECK(vk_submit(0, G.queue, &si, G.fence), "queueSubmit");
    if (G.eg_prof) { double vp1 = vk_now(); g_vsub_ms += vp1 - vp0; vp0 = vp1; }
    if (vk_fence_wait(G.dev, G.fence) != VK_SUCCESS) { G.ready = 0; return 0; }
    if (G.eg_prof) { g_vwait_ms += vk_now() - vp0; vkprof_tick(); }
    memcpy(hidden, G.h.ptr, hb);
    G.cmd_ready = 0; G.bound_tensor = NULL;   /* the shared command buffer/binding was clobbered */
    return 1;
}

static void wr_desc_dev(VkDevice dev, VkDescriptorSet set, int n, const VkDescriptorBufferInfo *bi) {
    VkWriteDescriptorSet w[6];
    for (int i = 0; i < n; i++) w[i] = (VkWriteDescriptorSet){
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set, .dstBinding = (uint32_t)i,
        .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &bi[i]};
    vkUpdateDescriptorSets(dev, (uint32_t)n, w, 0, NULL);
}
static void wr_desc(VkDescriptorSet set, int n, const VkDescriptorBufferInfo *bi) {
    wr_desc_dev(G.dev, set, n, bi);
}

/* Full batched expert MLP for `count` experts, hidden staying on-device:
 * for each c, hidden_c = silu(gate_c(x_c))*up_c(x_c) (fused), then y_c = down_c(hidden_c).
 * x/y are packed [sum(rows)*D]; experts are resident VkTensors (gate/up: D->I, down: I->D).
 * Mirrors coli_cuda_expert_group. Split into prepare+submit / take so the caller can
 * overlap the GPU batch with its own CPU share (issue -> CPU rows -> take); the group
 * runs on its OWN command buffer + fence, so in-flight work never collides with the
 * main pipeline (dense matmuls, absorb attention). Returns 0 -> caller falls back. */
static int eg_prepare_submit(ColiVkTensor *const *gates, ColiVkTensor *const *ups,
                             ColiVkTensor *const *downs, const int *rows, int count,
                             const float *x) {
    if (!G.ready || !G.shader_gu || count < 1 || count > 64) return 0;
    ColiVkTensor *g0 = gates[0]; if (!g0) return 0;
    int D = g0->I, I = g0->O, fmt = g0->fmt, total = 0, off[64];
    if (D > 6144) return 0;   /* gate_up shader stages x in xsh[6144] */
    int dfmt = downs[0]->fmt;   /* down may be a different quant than gate/up (per-projection
                                 * containers, e.g. --up-bits 3); gate/up must MATCH — the
                                 * fused gate_up shader decodes both with one fmt. */
    for (int c = 0; c < count; c++) {
        off[c] = total; total += rows[c];
        if (rows[c] < 1 || gates[c]->I != D || gates[c]->O != I || gates[c]->fmt != fmt ||
            ups[c]->I != D || ups[c]->O != I || ups[c]->fmt != fmt ||
            downs[c]->I != I || downs[c]->O != D || downs[c]->fmt != dfmt) return 0;
    }
    size_t xb = (size_t)total*D*4, hb = (size_t)total*I*4, yb = (size_t)total*D*4;
    if (!scratch_reserve(&G.eg_x, xb) || !scratch_reserve(&G.eg_h, hb) ||
        !scratch_reserve_mt(&G.eg_y, yb, G.memtype_cached)) return 0;   /* eg_y is read back -> cached */
    G.eg_prof = getenv("VK_PROF") != NULL;
    if (G.eg_prof) G.eg_t0 = vk_now();
    memcpy(G.eg_x.ptr, x, xb);
    if (G.eg_prof) G.eg_t1 = vk_now();

    if (!G.eg_pool) {   /* one-time: 64 gate_up (6-binding) + 64 down (4-binding) sets */
        VkDescriptorPoolSize ps = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 64*6 + 64*4};
        VkDescriptorPoolCreateInfo dpi = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 128, .poolSizeCount = 1, .pPoolSizes = &ps};
        VKCHECK(vkCreateDescriptorPool(G.dev, &dpi, NULL, &G.eg_pool), "eg descPool");
        VkDescriptorSetLayout lg[64], ld[64];
        for (int c = 0; c < 64; c++) { lg[c] = G.dsl_gu; ld[c] = G.dsl; }
        VkDescriptorSetAllocateInfo ag = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = G.eg_pool, .descriptorSetCount = 64, .pSetLayouts = lg};
        VkDescriptorSetAllocateInfo ad = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = G.eg_pool, .descriptorSetCount = 64, .pSetLayouts = ld};
        VKCHECK(vkAllocateDescriptorSets(G.dev, &ag, G.eg_gu), "eg gu sets");
        VKCHECK(vkAllocateDescriptorSets(G.dev, &ad, G.eg_dn), "eg dn sets");
        G.eg_nsets = 64;
    }
    for (int c = 0; c < count; c++) {
        VkDeviceSize xo = (VkDeviceSize)off[c]*D*4, ho = (VkDeviceSize)off[c]*I*4, yo = (VkDeviceSize)off[c]*D*4;
        VkDescriptorBufferInfo gi[6] = {
            {G.eg_x.buf, xo, (VkDeviceSize)rows[c]*D*4}, {gates[c]->wbuf, 0, VK_WHOLE_SIZE},
            {gates[c]->sbuf, 0, VK_WHOLE_SIZE}, {ups[c]->wbuf, 0, VK_WHOLE_SIZE},
            {ups[c]->sbuf, 0, VK_WHOLE_SIZE}, {G.eg_h.buf, ho, (VkDeviceSize)rows[c]*I*4}};
        wr_desc(G.eg_gu[c], 6, gi);
        VkDescriptorBufferInfo di[4] = {
            {G.eg_h.buf, ho, (VkDeviceSize)rows[c]*I*4}, {downs[c]->wbuf, 0, VK_WHOLE_SIZE},
            {downs[c]->sbuf, 0, VK_WHOLE_SIZE}, {G.eg_y.buf, yo, (VkDeviceSize)rows[c]*D*4}};
        wr_desc(G.eg_dn[c], 4, di);
    }
    if (G.eg_prof) G.eg_t2 = vk_now();

    VKCHECK(vkResetCommandBuffer(G.eg_cmd, 0), "eg resetCmd");
    VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    VKCHECK(vkBeginCommandBuffer(G.eg_cmd, &begin), "eg beginCmd");
    VkMemoryBarrier mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_SHADER_READ_BIT};
    /* phase 1: fused gate+up+silu -> hidden (per expert, bound to its x/hidden slices) */
    vkCmdBindPipeline(G.eg_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe_gu);
    for (int c = 0; c < count; c++) {
        struct PCGU pc = pcgu(fmt, rows[c], D, I, gates[c]->rowWords, gates[c]->gs);
        vkCmdBindDescriptorSets(G.eg_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt_gu, 0, 1, &G.eg_gu[c], 0, NULL);
        vkCmdPushConstants(G.eg_cmd, G.plyt_gu, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(G.eg_cmd, (uint32_t)((I + 7) / 8), (uint32_t)rows[c], 1);
    }
    vkCmdPipelineBarrier(G.eg_cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    /* phase 2: down projection hidden -> y */
    vkCmdBindPipeline(G.eg_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe);
    for (int c = 0; c < count; c++) {
        struct PC pc = {dfmt, rows[c], I, D, downs[c]->rowWords, downs[c]->gs};
        vkCmdBindDescriptorSets(G.eg_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt, 0, 1, &G.eg_dn[c], 0, NULL);
        vkCmdPushConstants(G.eg_cmd, G.plyt, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(G.eg_cmd, (uint32_t)((D + 7) / 8), (uint32_t)rows[c], 1);
    }
    VKCHECK(vkEndCommandBuffer(G.eg_cmd), "eg endCmd");
    if (G.eg_prof) G.eg_t3 = vk_now();

    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &G.eg_cmd};
    VKCHECK(vkResetFences(G.dev, 1, &G.eg_fence), "eg resetFence");
    { double vp0 = G.eg_prof ? vk_now() : 0;
      VKCHECK(vk_submit(0, G.queue, &si, G.eg_fence), "eg queueSubmit");
      if (G.eg_prof) g_vsub_ms += vk_now() - vp0; }
    G.eg_pending_yb = yb; G.eg_inflight = 1; async_begin(0);
    return 1;
}

/* Issue a group asynchronously: submit and return WITHOUT waiting, so the caller
 * computes its CPU share concurrently. Exactly one group may be in flight. */
int coli_vk_expert_group_issue(ColiVkTensor *const *gates, ColiVkTensor *const *ups,
                               ColiVkTensor *const *downs, const int *rows, int count,
                               const float *x) {
    if (G.eg_inflight) return 0;
    return eg_prepare_submit(gates, ups, downs, rows, count, x);
}

/* Join the in-flight group and read back the packed outputs. */
int coli_vk_expert_group_take(float *y) {
    if (!G.eg_inflight) return 0;
    G.eg_inflight = 0;
    if (vk_fence_wait(G.dev, G.eg_fence) != VK_SUCCESS) {
        fprintf(stderr, "[VK] expert-group fence wait failed — disabling GPU offload\n");
        G.ready = 0; async_end(0); return 0;
    }
    async_end(0);
    double t4 = G.eg_prof ? vk_now() : 0;
    memcpy(y, G.eg_y.ptr, G.eg_pending_yb);
    if (G.eg_prof) {
        double t5 = vk_now();
        fprintf(stderr, "[VK_PROF] memcpy_x %.3f | desc %.3f | record %.3f | issue->take %.3f | memcpy_y %.3f ms\n",
                G.eg_t1-G.eg_t0, G.eg_t2-G.eg_t1, G.eg_t3-G.eg_t2, t4-G.eg_t3, t5-t4);
    }
    return 1;
}

/* Synchronous form (shared expert, harness): issue + take in one call. */
int coli_vk_expert_group(ColiVkTensor *const *gates, ColiVkTensor *const *ups,
                         ColiVkTensor *const *downs, const int *rows, int count,
                         float *y, const float *x) {
    if (G.eg_inflight) return 0;
    if (!eg_prepare_submit(gates, ups, downs, rows, count, x)) return 0;
    return coli_vk_expert_group_take(y);
}

/* ==================== SECOND DEVICE: expert tier only (COLI_VK_DEV2) ============
 * A self-contained context for a second Vulkan GPU (e.g. an RX 580 beside the
 * RX 9070) that hosts ONLY resident tier experts and runs ONLY the async
 * expert-group path (fused gate_up -> down). Attention, dense, q-prep and the
 * KV mirror stay on device 0. Deliberately separate from G so the device-0 hot
 * path is untouched and both groups can be in flight simultaneously. The same
 * physical device as dev0 is allowed when forced by index (a second logical
 * device — the pre-hardware test mode); `auto` requires a distinct real GPU. */
static struct {
    int ready;
    VkPhysicalDevice phys; VkDevice dev; VkQueue queue; uint32_t qfam;
    uint32_t memtype, memtype_cached;
    VkShaderModule sh_qmm, sh_gu;
    VkDescriptorSetLayout dsl, dsl_gu; VkPipelineLayout plyt, plyt_gu;
    VkPipeline pipe, pipe_gu;
    VkCommandPool cpool; VkCommandBuffer cmd; VkFence fence;
    VkDescriptorPool pool; VkDescriptorSet gu[64], dn[64]; int nsets;
    Scratch x, h, y;
    int inflight; size_t pending_yb;
    int has_budget, has_hostmem;
    /* what the expert batch's context on this device needs (xb_bind) */
    uint32_t memtype_dev, ts_bits; float ts_period;
    size_t ssbo_align, ssbo_range, buf_align;
    int integrated, shares_ram;
    char name[256];
} G2;

static int alloc_hostvis_d2(size_t bytes, VkBuffer *buf, VkDeviceMemory *mem, void **ptr, uint32_t memtype) {
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = bytes, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    VKCHECK(vkCreateBuffer(G2.dev, &bi, NULL, buf), "d2 vkCreateBuffer");
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(G2.dev, *buf, &req);
    VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = req.size, .memoryTypeIndex = memtype};
    VKCHECK(vkAllocateMemory(G2.dev, &ai, NULL, mem), "d2 vkAllocateMemory");
    VKCHECK(vkBindBufferMemory(G2.dev, *buf, *mem, 0), "d2 vkBindBufferMemory");
    if (ptr) VKCHECK(vkMapMemory(G2.dev, *mem, 0, bytes, 0, ptr), "d2 vkMapMemory");
    return 1;
}
static int scratch_reserve_d2(Scratch *s, size_t bytes, uint32_t memtype) {
    if (s->cap >= bytes) return 1;
    if (s->buf) { vkDestroyBuffer(G2.dev, s->buf, NULL); vkFreeMemory(G2.dev, s->mem, NULL); }
    s->buf = VK_NULL_HANDLE; s->cap = 0; s->ptr = NULL;
    if (!alloc_hostvis_d2(bytes, &s->buf, &s->mem, &s->ptr, memtype)) return 0;
    s->cap = bytes;
    return 1;
}
static VkDevice pool_device(const VkWPool *P) { return P->dev ? G2.dev : G.dev; }
static uint32_t pool_memtype(const VkWPool *P) {
    return g_up[P->dev].on ? g_up[P->dev].mt_dev : P->dev ? G2.memtype : G.memtype;
}
/* An upload's fence failed: the device is gone, as for any other wait. */
static void up_lost(int dev) {
    fprintf(stderr, "[VK] %sstaged upload failed (%s: %d): the device is lost, disabling %s\n", dev ? "dev2 " : "",
            g_up[dev].what ? g_up[dev].what : "?", (int)g_up[dev].err, dev ? "dev2 offload" : "GPU offload");
    /* another thread (the tier's uploader) may be the one that finds out */
    if (dev) __atomic_store_n(&G2.ready, 0, __ATOMIC_RELEASE); else __atomic_store_n(&G.ready, 0, __ATOMIC_RELEASE);
}
static int pool_has_prio(const VkWPool *P) { return P->dev ? 0 : G.has_prio; }
static int upload_tensor_d2(ColiVkTensor **out, const void *weights, const float *scales,
                            int fmt, int I, int O, int gs) {
    return upload_tensor_pool(&g_wpool2, out, weights, scales, fmt, I, O, gs);
}

/* Bring up the second device. devidx: -1 = auto (best-ranked real GPU that is NOT
 * device 0; fails if none), >=0 = that enumeration index (same-physical-device
 * allowed with a warning — the pre-hardware test mode). Requires coli_vk_init. */
int coli_vk_init_dev2(const char *spv_path, int devidx) {
    if (G2.ready) return 1;
    if (!G.ready) return 0;
    uint32_t nd = 0;
    vkEnumeratePhysicalDevices(G.inst, &nd, NULL);
    VkPhysicalDevice devs[8]; if (nd > 8) nd = 8;
    if (!nd) return 0;
    vkEnumeratePhysicalDevices(G.inst, &nd, devs);
    if (devidx >= 0) {
        if ((uint32_t)devidx >= nd) { fprintf(stderr, "[VK] dev2: index %d out of range (%u devices)\n", devidx, nd); return 0; }
        G2.phys = devs[devidx];
        if (G2.phys == G.phys)
            fprintf(stderr, "[VK] dev2: SAME physical device as dev0 — second logical device (test mode)\n");
    } else {
        int bestrank = -1;
        for (uint32_t i = 0; i < nd; i++) {
            if (devs[i] == G.phys) continue;
            VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(devs[i], &p);
            int rank = p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU   ? 4 :
                       p.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? 3 : -1;
            if (rank > bestrank) { bestrank = rank; G2.phys = devs[i]; }
        }
        if (bestrank < 0) { fprintf(stderr, "[VK] dev2=auto: no second real GPU found\n"); return 0; }
    }
    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(G2.phys, &nq, NULL);
    VkQueueFamilyProperties qf[16]; if (nq > 16) nq = 16;
    vkGetPhysicalDeviceQueueFamilyProperties(G2.phys, &nq, qf);
    G2.qfam = UINT32_MAX;
    for (uint32_t i = 0; i < nq; i++)
        if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { G2.qfam = i; break; }
    if (G2.qfam == UINT32_MAX) { fprintf(stderr, "[VK] dev2: no compute queue\n"); return 0; }
    float qprio = 1.0f;
    VkDeviceQueueCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = G2.qfam, .queueCount = 1, .pQueuePriorities = &qprio};
    const char *dext[2]; uint32_t ndext = 0;
    {
        uint32_t ne = 0;
        vkEnumerateDeviceExtensionProperties(G2.phys, NULL, &ne, NULL);
        VkExtensionProperties *ep = ne ? malloc(ne * sizeof(*ep)) : NULL;
        if (ep) {
            vkEnumerateDeviceExtensionProperties(G2.phys, NULL, &ne, ep);
            for (uint32_t i = 0; i < ne; i++) {
#ifdef VK_EXT_memory_budget
                if (!strcmp(ep[i].extensionName, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME)) G2.has_budget = 1;
#endif
#ifdef VK_EXT_external_memory_host
                if (!strcmp(ep[i].extensionName, VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME)) G2.has_hostmem = 1;
#endif
            }
            free(ep);
        }
#ifdef VK_EXT_memory_budget
        if (G2.has_budget) dext[ndext++] = VK_EXT_MEMORY_BUDGET_EXTENSION_NAME;
#endif
#ifdef VK_EXT_external_memory_host
        if (G2.has_hostmem) dext[ndext++] = VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME;
#endif
    }
    VkDeviceCreateInfo di = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1, .pQueueCreateInfos = &qi,
        .enabledExtensionCount = ndext, .ppEnabledExtensionNames = ndext ? dext : NULL};
    VKCHECK(vkCreateDevice(G2.phys, &di, NULL, &G2.dev), "d2 vkCreateDevice");
    vk_mem_device(1, G2.phys, G2.dev);
    vk_pool_blocks(1);
    vkGetDeviceQueue(G2.dev, G2.qfam, 0, &G2.queue);
    int mt = pick_memtype(G2.phys);
    if (mt < 0) { fprintf(stderr, "[VK] dev2: no host-visible memory\n"); return 0; }
    G2.memtype = (uint32_t)mt;
    G2.memtype_cached = (uint32_t)pick_memtype_cached(G2.phys);
    G2.memtype_dev = (uint32_t)pick_memtype_device(G2.phys);
    {   /* the expert batch's limits on this device, as coli_vk_init reads them for G */
        VkPhysicalDeviceProperties pp; vkGetPhysicalDeviceProperties(G2.phys, &pp);
        G2.ssbo_align = (size_t)pp.limits.minStorageBufferOffsetAlignment;
        G2.ssbo_range = (size_t)pp.limits.maxStorageBufferRange;
        G2.ts_period = pp.limits.timestampPeriod;
        G2.ts_bits = qf[G2.qfam].timestampValidBits;
        G2.integrated = pp.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;
        G2.shares_ram = G2.integrated || pp.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU;
        snprintf(G2.name, sizeof G2.name, "%s", pp.deviceName);
        VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = 4096,
            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
        VkBuffer probe;
        G2.buf_align = 256;
        if (vkCreateBuffer(G2.dev, &bi, NULL, &probe) == VK_SUCCESS) {
            VkMemoryRequirements req; vkGetBufferMemoryRequirements(G2.dev, probe, &req);
            if (req.alignment > G2.buf_align) G2.buf_align = (size_t)req.alignment;
            vkDestroyBuffer(G2.dev, probe, NULL);
        }
    }
    /* its experts as device 0's: staged uploads by the same rule, on its one queue */
    if (place_decide(G2.phys, mt, &g_up[1])) {
        VkUp *u = &g_up[1];
        u->dev = G2.dev; u->q = G2.queue; u->fam = G2.qfam; u->shared = 1;
        u->fams[0] = G2.qfam; u->nfams = 1;
        if (!up_init(u)) {
            fprintf(stderr, "[VK] dev2: staged uploads unavailable (no staging buffer): its experts stay in mapped memory\n");
            up_destroy(u); u->on = u->shared = 0;
        } else fprintf(stderr, "[VK] dev2 memory: staged uploads, experts in device-local memory (type %u) copied from "
                       "host staging memory (type %u) (%s)\n", u->mt_dev, u->mt_stage, u->why);
        up_import_init(u, G2.phys, G2.has_hostmem, "dev2 ");
    }
    G2.sh_qmm = load_spv(G2.dev, spv_path);
    if (!G2.sh_qmm) return 0;
    char gu_path[512]; derive_sibling(spv_path, "_gate_up.spv", gu_path, sizeof(gu_path));
    G2.sh_gu = load_spv(G2.dev, gu_path);
    if (!G2.sh_gu) { fprintf(stderr, "[VK] dev2: gate_up shader required for the tier\n"); return 0; }
    VkDescriptorPool dp; VkDescriptorSet ds;   /* build_pipeline's singleton set: unused here */
    if (!build_pipeline(G2.dev, 4, sizeof(struct PC), G2.sh_qmm, &G2.dsl, &G2.plyt, &G2.pipe, &dp, &ds)) return 0;
    if (!build_pipeline(G2.dev, 6, sizeof(struct PCGU), G2.sh_gu, &G2.dsl_gu, &G2.plyt_gu, &G2.pipe_gu, &dp, &ds)) return 0;
    VkCommandPoolCreateInfo cpci = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = G2.qfam};
    VKCHECK(vkCreateCommandPool(G2.dev, &cpci, NULL, &G2.cpool), "d2 cmdPool");
    VkCommandBufferAllocateInfo cbi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = G2.cpool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
    VKCHECK(vkAllocateCommandBuffers(G2.dev, &cbi, &G2.cmd), "d2 cmdBuf");
    VkFenceCreateInfo fi = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VKCHECK(vkCreateFence(G2.dev, &fi, NULL, &G2.fence), "d2 fence");
    G2.ready = 1;
    VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(G2.phys, &p);
    fprintf(stderr, "[VK] dev2 ready: %s (expert tier only), compute qfam %u, memtype %u\n",
            p.deviceName, G2.qfam, G2.memtype);
    return 1;
}

int coli_vk_dev2_available(void) { return G2.ready; }
int coli_vk_tensor_dev(const ColiVkTensor *t) { return t ? t->dev : 0; }

int coli_vk_mem_budget2(double *used_gb, double *budget_gb) {
    if (g_mem.cap[1] && G2.phys) {   /* COLI_VK_DEVICE_CAP_MB, as coli_vk_mem_budget */
        if (used_gb) *used_gb = (double)__atomic_load_n(&g_mem.used[1], __ATOMIC_RELAXED) / 1e9;
        if (budget_gb) *budget_gb = (double)g_mem.cap[1] / 1e9;
        return 1;
    }
#ifdef VK_EXT_memory_budget
    if (!G2.has_budget || !G2.phys) return 0;
    VkPhysicalDeviceMemoryBudgetPropertiesEXT bud = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
    VkPhysicalDeviceMemoryProperties2 mp2 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2, .pNext = &bud};
    vkGetPhysicalDeviceMemoryProperties2(G2.phys, &mp2);
    double u = 0, b = 0;
    for (uint32_t i = 0; i < mp2.memoryProperties.memoryHeapCount; i++)
        if (mp2.memoryProperties.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            u += (double)bud.heapUsage[i]; b += (double)bud.heapBudget[i];
        }
    if (used_gb) *used_gb = u / 1e9;
    if (budget_gb) *budget_gb = b / 1e9;
    return b > 0;
#else
    (void)used_gb; (void)budget_gb; return 0;
#endif
}

/* COLI_VK_DEV2=auto|<index> for the routed-expert tier (vk_tier.c): the second device up
 * with the shaders G was opened with. 1 when it is up (before, or now); 0 when the
 * variable is unset or names no usable device (coli_vk_init_dev2's line says why). */
int coli_vk_dev2_open_env(void) {
    if (G2.ready) return 1;
    const char *d2 = getenv("COLI_VK_DEV2");
    if (!d2 || !*d2 || !G.ready) return 0;
    return coli_vk_init_dev2(G.spv_path, strcmp(d2, "auto") ? atoi(d2) : -1);
}

/* What the tier sizes itself by, for either device. */
int coli_vk_dev_info(int dev, ColiVkDevInfo *o) {
    memset(o, 0, sizeof *o);
    if (dev == 0) {
        if (!G.ready) return 0;
        double u = 0, b = 0;
        o->name = coli_vk_device_name(); o->integrated = coli_vk_device_integrated();
        o->shares_ram = coli_vk_device_shares_ram(); o->local_bytes = coli_vk_device_local_bytes();
        o->has_budget = coli_vk_mem_budget(&u, &b); o->free_bytes = coli_vk_free_bytes();
        o->buf_align = coli_vk_buffer_alignment();
        return 1;
    }
    if (dev != 1 || !G2.ready) return 0;
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(G2.phys, &mp);
    VkDeviceSize m = 0;
    for (uint32_t i = 0; i < mp.memoryHeapCount; i++)
        if ((mp.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) && mp.memoryHeaps[i].size > m) m = mp.memoryHeaps[i].size;
    if (g_mem.cap[1] && g_mem.cap[1] < m) m = g_mem.cap[1];   /* COLI_VK_DEVICE_CAP_MB */
    o->name = G2.name; o->integrated = G2.integrated; o->shares_ram = G2.shares_ram;
    o->local_bytes = (size_t)m; o->buf_align = G2.buf_align;
    double u = 0, b = 0;
    o->has_budget = coli_vk_mem_budget2(&u, &b);
    size_t held = (size_t)__atomic_load_n(&g_mem.used[1], __ATOMIC_RELAXED);
    o->free_bytes = g_mem.cap[1] ? (size_t)vk_mem_room(1)
                  : o->has_budget ? (b > u ? (size_t)((b - u) * 1e9) : 0)
                  : o->local_bytes > held ? o->local_bytes - held : 0;
    return 1;
}

int coli_vk_tensor_ensure2(ColiVkTensor **tensor, const void *weights, const float *scales, int fmt, int I, int O, int grp) {
    if (!G2.ready) return 0;
    return upload_tensor_d2(tensor, weights, scales, fmt, I, O, grp);
}

/* dev2 mirror of eg_prepare_submit: identical structure on G2's pipelines/scratches. */
static int eg2_prepare_submit(ColiVkTensor *const *gates, ColiVkTensor *const *ups,
                              ColiVkTensor *const *downs, const int *rows, int count,
                              const float *x) {
    if (!G2.ready || count < 1 || count > 64) return 0;
    ColiVkTensor *g0 = gates[0]; if (!g0) return 0;
    int D = g0->I, I = g0->O, fmt = g0->fmt, total = 0, off[64];
    if (D > 6144) return 0;
    int dfmt = downs[0]->fmt;
    for (int c = 0; c < count; c++) {
        off[c] = total; total += rows[c];
        if (rows[c] < 1 || gates[c]->I != D || gates[c]->O != I || gates[c]->fmt != fmt ||
            ups[c]->I != D || ups[c]->O != I || ups[c]->fmt != fmt ||
            downs[c]->I != I || downs[c]->O != D || downs[c]->fmt != dfmt) return 0;
    }
    size_t xb = (size_t)total*D*4, hb = (size_t)total*I*4, yb = (size_t)total*D*4;
    /* VK_PROF=1: phase split of the dev2 issue cost (same scheme as the dense path) —
     * localizes the per-block tax between our copy, descriptors, recording and the
     * driver's submit on the chipset-x4 Polaris path. */
    static double q_x, q_desc, q_rec, q_sub; static long q_n;
    double t0 = G.eg_prof ? vk_now() : 0, tA;
    if (!scratch_reserve_d2(&G2.x, xb, G2.memtype) || !scratch_reserve_d2(&G2.h, hb, G2.memtype) ||
        !scratch_reserve_d2(&G2.y, yb, G2.memtype_cached)) return 0;
    memcpy(G2.x.ptr, x, xb);
    if (G.eg_prof) { tA = vk_now(); q_x += tA - t0; t0 = tA; }
    if (!G2.pool) {
        VkDescriptorPoolSize ps = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 64*6 + 64*4};
        VkDescriptorPoolCreateInfo dpi = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 128, .poolSizeCount = 1, .pPoolSizes = &ps};
        VKCHECK(vkCreateDescriptorPool(G2.dev, &dpi, NULL, &G2.pool), "d2 eg descPool");
        VkDescriptorSetLayout lg[64], ld[64];
        for (int c = 0; c < 64; c++) { lg[c] = G2.dsl_gu; ld[c] = G2.dsl; }
        VkDescriptorSetAllocateInfo ag = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = G2.pool, .descriptorSetCount = 64, .pSetLayouts = lg};
        VkDescriptorSetAllocateInfo ad = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = G2.pool, .descriptorSetCount = 64, .pSetLayouts = ld};
        VKCHECK(vkAllocateDescriptorSets(G2.dev, &ag, G2.gu), "d2 eg gu sets");
        VKCHECK(vkAllocateDescriptorSets(G2.dev, &ad, G2.dn), "d2 eg dn sets");
        G2.nsets = 64;
    }
    for (int c = 0; c < count; c++) {
        VkDeviceSize xo = (VkDeviceSize)off[c]*D*4, ho = (VkDeviceSize)off[c]*I*4, yo = (VkDeviceSize)off[c]*D*4;
        VkDescriptorBufferInfo gi[6] = {
            {G2.x.buf, xo, (VkDeviceSize)rows[c]*D*4}, {gates[c]->wbuf, 0, VK_WHOLE_SIZE},
            {gates[c]->sbuf, 0, VK_WHOLE_SIZE}, {ups[c]->wbuf, 0, VK_WHOLE_SIZE},
            {ups[c]->sbuf, 0, VK_WHOLE_SIZE}, {G2.h.buf, ho, (VkDeviceSize)rows[c]*I*4}};
        wr_desc_dev(G2.dev, G2.gu[c], 6, gi);
        VkDescriptorBufferInfo di[4] = {
            {G2.h.buf, ho, (VkDeviceSize)rows[c]*I*4}, {downs[c]->wbuf, 0, VK_WHOLE_SIZE},
            {downs[c]->sbuf, 0, VK_WHOLE_SIZE}, {G2.y.buf, yo, (VkDeviceSize)rows[c]*D*4}};
        wr_desc_dev(G2.dev, G2.dn[c], 4, di);
    }
    if (G.eg_prof) { tA = vk_now(); q_desc += tA - t0; t0 = tA; }
    VKCHECK(vkResetCommandBuffer(G2.cmd, 0), "d2 eg resetCmd");
    VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    VKCHECK(vkBeginCommandBuffer(G2.cmd, &begin), "d2 eg beginCmd");
    VkMemoryBarrier mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_SHADER_READ_BIT};
    vkCmdBindPipeline(G2.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G2.pipe_gu);
    for (int c = 0; c < count; c++) {
        struct PCGU pc = pcgu(fmt, rows[c], D, I, gates[c]->rowWords, gates[c]->gs);
        vkCmdBindDescriptorSets(G2.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G2.plyt_gu, 0, 1, &G2.gu[c], 0, NULL);
        vkCmdPushConstants(G2.cmd, G2.plyt_gu, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(G2.cmd, (uint32_t)((I + 7) / 8), (uint32_t)rows[c], 1);
    }
    vkCmdPipelineBarrier(G2.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    vkCmdBindPipeline(G2.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G2.pipe);
    for (int c = 0; c < count; c++) {
        struct PC pc = {dfmt, rows[c], I, D, downs[c]->rowWords, downs[c]->gs};
        vkCmdBindDescriptorSets(G2.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G2.plyt, 0, 1, &G2.dn[c], 0, NULL);
        vkCmdPushConstants(G2.cmd, G2.plyt, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(G2.cmd, (uint32_t)((D + 7) / 8), (uint32_t)rows[c], 1);
    }
    VKCHECK(vkEndCommandBuffer(G2.cmd), "d2 eg endCmd");
    if (G.eg_prof) { tA = vk_now(); q_rec += tA - t0; t0 = tA; }
    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &G2.cmd};
    VKCHECK(vkResetFences(G2.dev, 1, &G2.fence), "d2 eg resetFence");
    VKCHECK(vk_submit(1, G2.queue, &si, G2.fence), "d2 eg queueSubmit");
    if (G.eg_prof) { tA = vk_now(); q_sub += tA - t0;
        if ((++q_n & 2047) == 0)
            fprintf(stderr, "[VK_PROF d2iss] n=%ld | memcpy_x %.0f | desc %.0f | record %.0f | submit %.0f ms\n",
                    q_n, q_x, q_desc, q_rec, q_sub);
    }
    G2.pending_yb = yb; G2.inflight = 1; async_begin(1);
    return 1;
}

int coli_vk_expert_group_issue2(ColiVkTensor *const *gates, ColiVkTensor *const *ups,
                                ColiVkTensor *const *downs, const int *rows, int count,
                                const float *x) {
    if (G2.inflight) return 0;
    return eg2_prepare_submit(gates, ups, downs, rows, count, x);
}
int coli_vk_expert_group_take2(float *y) {
    if (!G2.inflight) return 0;
    G2.inflight = 0;
    if (vk_fence_wait(G2.dev, G2.fence) != VK_SUCCESS) {
        fprintf(stderr, "[VK] dev2 expert-group fence wait failed — disabling dev2 offload\n");
        G2.ready = 0; async_end(1); return 0;
    }
    async_end(1);
    memcpy(y, G2.y.ptr, G2.pending_yb);
    return 1;
}
int coli_vk_expert_group2(ColiVkTensor *const *gates, ColiVkTensor *const *ups,
                          ColiVkTensor *const *downs, const int *rows, int count,
                          float *y, const float *x) {
    if (G2.inflight) return 0;
    if (!eg2_prepare_submit(gates, ups, downs, rows, count, x)) return 0;
    return coli_vk_expert_group_take2(y);
}

/* ---- MLA absorb attention core -------------------------------------------------
 * The KV latent (L, [rows,K]) and rope (R, [rows,Rd]) caches live in persistent
 * per-layer device buffers, appended row-by-row as tokens decode (the host stays
 * canonical; glm.c tracks a valid-watermark and re-appends after invalidation).
 * Rows are indexed by ABSOLUTE position, so kv_start windows just skip rows. */

/* ---- staged writes on the main queue (the KV mirror, the q-prep norm weights) -------
 * With staged uploads these buffers are device-local and the host's writes go through
 * this staging buffer: each is a pending copy, recorded at the head of the next absorb or
 * q-prep command buffer (their only readers) and done when its fence has signalled. A
 * row written again before then (a rewound cache) first sends what is pending, so no two
 * pending copies overlap. Engine thread only, as the KV mirror's calls are. */
typedef struct { VkBuffer dst; VkBufferCopy c; } PwCopy;
static struct {
    VkBuffer buf; VkDeviceMemory mem; uint8_t *ptr; size_t cap, used;
    PwCopy *cp; int n, ccp, rec;
    int last1[VK_KV_LAYERS];      /* per layer, 1 + the last row pending (0: none) */
    size_t bytes;                 /* staged so far */
} PW;
static void pw_record(VkCommandBuffer cmd) {
    if (!PW.n) return;
    for (int i = 0; i < PW.n; i++) vkCmdCopyBuffer(cmd, PW.buf, PW.cp[i].dst, 1, &PW.cp[i].c);
    VkMemoryBarrier mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT, .dstAccessMask = VK_ACCESS_SHADER_READ_BIT};
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    PW.rec = PW.n;
}
static void pw_done(void) {   /* the command buffer that recorded them has finished */
    if (!PW.rec) return;
    PW.n = PW.rec = 0; PW.used = 0;
    memset(PW.last1, 0, sizeof PW.last1);
}
static int pw_flush(void) {   /* send what is pending now, in a command buffer of its own */
    if (!PW.n) return 1;
    if (!G.ready) { PW.n = PW.rec = 0; PW.used = 0; return 0; }
    G.cmd_ready = 0; G.bound_tensor = NULL;   /* the matmul's recorded command buffer goes */
    VKCHECK(vkResetCommandBuffer(G.cmd, 0), "resetCmd");
    VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    VKCHECK(vkBeginCommandBuffer(G.cmd, &begin), "beginCmd");
    pw_record(G.cmd);
    VKCHECK(vkEndCommandBuffer(G.cmd), "endCmd");
    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &G.cmd};
    VKCHECK(vkResetFences(G.dev, 1, &G.fence), "resetFence");
    VKCHECK(vk_submit(0, G.queue, &si, G.fence), "queueSubmit");
    if (vk_fence_wait(G.dev, G.fence) != VK_SUCCESS) { G.ready = 0; return 0; }
    pw_done();
    return 1;
}
static int pw_add(VkBuffer dst, size_t dst_off, const void *src, size_t bytes) {
    if (PW.used + bytes > PW.cap) {
        if (!pw_flush()) return 0;
        if (bytes > PW.cap) {
            if (PW.buf) { vkDestroyBuffer(G.dev, PW.buf, NULL); vkFreeMemory(G.dev, PW.mem, NULL); }
            PW.buf = VK_NULL_HANDLE; PW.mem = VK_NULL_HANDLE; PW.ptr = NULL; PW.cap = 0;
            size_t cap = (size_t)4 << 20;
            while (cap < bytes) cap *= 2;
            void *p;
            if (up_fault("pwstage") ||
                !alloc_buf_mt(cap, &PW.buf, &PW.mem, &p, g_up[0].mt_stage, VK_BUFFER_USAGE_TRANSFER_SRC_BIT)) {
                PW.buf = VK_NULL_HANDLE; PW.mem = VK_NULL_HANDLE; return 0;
            }
            PW.ptr = p; PW.cap = cap;
        }
    }
    if (PW.n == PW.ccp) {
        int c = PW.ccp ? 2 * PW.ccp : 256;
        PwCopy *n = realloc(PW.cp, (size_t)c * sizeof *n);
        if (!n) return 0;
        PW.cp = n; PW.ccp = c;
    }
    memcpy(PW.ptr + PW.used, src, bytes);
    PW.cp[PW.n++] = (PwCopy){dst, {PW.used, dst_off, bytes}};
    PW.used = (PW.used + bytes + 15) & ~(size_t)15;
    PW.bytes += bytes;
    return 1;
}
/* A device-local buffer the main queue writes (PW) and reads, never mapped. */
static int alloc_dev(size_t bytes, VkBuffer *buf, VkDeviceMemory *mem) {
    if (up_fault("kvbuf")) return 0;
    return alloc_buf_mt(bytes, buf, mem, NULL, g_up[0].mt_dev,
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
}

int coli_vk_kv_ensure(int layer, int max_rows, int K, int Rd) {
    if (!G.ready || layer < 0 || layer >= VK_KV_LAYERS || max_rows < 1 || K < 1 || Rd < 1) return 0;
    VkKvLayer *v = &G.kv[layer];
    if (v->bl) return v->rows >= max_rows && v->K == K && v->R == Rd;  /* resize goes through coli_vk_kv_reset */
    float p0 = G.prio; G.prio = 1.0f;            /* KV mirror rides every attention submit */
    int ok1, ok;
    if (g_up[0].on) {   /* staged: device-local, the rows through PW */
        ok1 = alloc_dev((size_t)max_rows * K * 4, &v->bl, &v->ml);
        ok = ok1 && alloc_dev((size_t)max_rows * Rd * 4, &v->br, &v->mr);
    } else {
        ok1 = alloc_hostvis((size_t)max_rows * K * 4, &v->bl, &v->ml, &v->pl);
        ok = ok1 && alloc_hostvis((size_t)max_rows * Rd * 4, &v->br, &v->mr, &v->pr);
    }
    G.prio = p0;
    if (!ok) {
        if (ok1) { vkDestroyBuffer(G.dev, v->bl, NULL); vkFreeMemory(G.dev, v->ml, NULL); }
        memset(v, 0, sizeof(*v)); return 0;
    }
    v->rows = max_rows; v->K = K; v->R = Rd;
    return 1;
}

/* Mirror one host cache row into the device copy (write-combined memory: the CPU
 * only ever WRITES these buffers, the GPU reads them). */
int coli_vk_kv_row(int layer, int pos, const float *L, const float *R) {
    if (layer < 0 || layer >= VK_KV_LAYERS) return 0;
    VkKvLayer *v = &G.kv[layer];
    if (!v->bl || pos < 0 || pos >= v->rows) return 0;
    if (g_up[0].on) {
        if (pos < PW.last1[layer] && !pw_flush()) return 0;
        if (!pw_add(v->bl, (size_t)pos * v->K * 4, L, (size_t)v->K * 4) ||
            !pw_add(v->br, (size_t)pos * v->R * 4, R, (size_t)v->R * 4)) return 0;
        PW.last1[layer] = pos + 1;
        return 1;
    }
    memcpy((float *)v->pl + (size_t)pos * v->K, L, (size_t)v->K * 4);
    memcpy((float *)v->pr + (size_t)pos * v->R, R, (size_t)v->R * 4);
    return 1;
}

/* Drop all per-layer KV device caches (cache resize in kv_alloc). */
void coli_vk_kv_reset(void) {
    if (PW.n) pw_flush();   /* nothing pending may name a buffer that goes */
    for (int i = 0; i < VK_KV_LAYERS; i++) {
        VkKvLayer *v = &G.kv[i];
        if (!v->bl) continue;
        if (G.ready) {   /* dead device: leak GPU handles like coli_vk_tensor_free */
            vkDestroyBuffer(G.dev, v->bl, NULL); vkFreeMemory(G.dev, v->ml, NULL);
            vkDestroyBuffer(G.dev, v->br, NULL); vkFreeMemory(G.dev, v->mr, NULL);
        }
        memset(v, 0, sizeof(*v));
    }
}

/* Decode MLA absorption core for S causal query rows of one sequence, one submit:
 * ctx[s,h,:] = softmax((Wnope_h^T q_nope).L_t + q_rope.R_t) weighted latent context
 * projected through the value rows of kv_b. kv_b ([H*(Q+V), K]) uploads on first
 * call and stays resident; L/R rows [st0, T) must already be mirrored via
 * coli_vk_kv_row. Returns 0 -> caller falls back to CPU. */
int coli_vk_attention_absorb(ColiVkTensor **kvb, const void *w, const float *sc, int fmt, int grp,
                             float *ctx, const float *q, int layer, int S, int H,
                             int Q, int R, int V, int K, int st0, int T, float scale) {
    if (!G.ready || !G.pipe_att || S < 1 || H < 1 || layer < 0 || layer >= VK_KV_LAYERS) return 0;
    if (Q > 256 || R > 64 || K > 512 || st0 < 0 || T - S - st0 < 0) return 0;  /* shared-array limits */
    VkKvLayer *kv = &G.kv[layer];
    if (!kv->bl || kv->rows < T || kv->K != K || kv->R != R) return 0;
    if (!upload_tensor(kvb, w, sc, fmt, K, H * (Q + V), grp)) return 0;
    ColiVkTensor *t = *kvb;
    int cap = T - st0;
    size_t qb = (size_t)S * H * (Q + R) * 4, cb = (size_t)S * H * V * 4;
    size_t sb = (size_t)S * H * cap * 4;
    if (!scratch_reserve(&G.x, qb) || !scratch_reserve_mt(&G.y, cb, G.memtype_cached) ||
        !scratch_reserve(&G.att_sc, sb)) return 0;    /* y (ctx) is read back -> cached */
    memcpy(G.x.ptr, q, qb);

    VkDescriptorBufferInfo bi[7] = {
        {.buffer = G.x.buf, .range = VK_WHOLE_SIZE}, {.buffer = t->wbuf, .range = VK_WHOLE_SIZE},
        {.buffer = t->sbuf, .range = VK_WHOLE_SIZE}, {.buffer = kv->bl, .range = VK_WHOLE_SIZE},
        {.buffer = kv->br, .range = VK_WHOLE_SIZE}, {.buffer = G.att_sc.buf, .range = VK_WHOLE_SIZE},
        {.buffer = G.y.buf, .range = VK_WHOLE_SIZE}};
    VkWriteDescriptorSet wd[7];
    for (int i = 0; i < 7; i++) wd[i] = (VkWriteDescriptorSet){
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = G.dset_att,
        .dstBinding = (uint32_t)i, .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &bi[i]};
    vkUpdateDescriptorSets(G.dev, 7, wd, 0, NULL);

    VKCHECK(vkResetCommandBuffer(G.cmd, 0), "resetCmd");
    VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    VKCHECK(vkBeginCommandBuffer(G.cmd, &begin), "beginCmd");
    pw_record(G.cmd);   /* staged: the mirror's pending rows first */
    vkCmdBindPipeline(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe_att);
    vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt_att, 0, 1, &G.dset_att, 0, NULL);
    struct PCAttn pc = {fmt, S, H, Q, R, V, K, st0, T, t->rowWords, cap, scale, t->gs};
    vkCmdPushConstants(G.cmd, G.plyt_att, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(G.cmd, (uint32_t)H, (uint32_t)S, 1);     /* one workgroup per (head, row) */
    VKCHECK(vkEndCommandBuffer(G.cmd), "endCmd");

    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &G.cmd};
    VKCHECK(vkResetFences(G.dev, 1, &G.fence), "resetFence");
    double vp0 = G.eg_prof ? vk_now() : 0;
    VKCHECK(vk_submit(0, G.queue, &si, G.fence), "queueSubmit");
    if (G.eg_prof) { double vp1 = vk_now(); g_vsub_ms += vp1 - vp0; vp0 = vp1; }
    if (vk_fence_wait(G.dev, G.fence) != VK_SUCCESS) { G.ready = 0; return 0; }
    pw_done();
    if (G.eg_prof) { g_vwait_ms += vk_now() - vp0; vkprof_tick(); }
    memcpy(ctx, G.y.ptr, cb);
    G.cmd_ready = 0; G.bound_tensor = NULL;   /* the shared command buffer/binding was clobbered */
    return 1;
}

/* Two resident matmuls sharing the SAME input x in ONE submit (q_a + kv_a in the
 * attention prologue): one x staging, two dispatches, one fence — replaces two
 * full submit+wait roundtrips. Outputs y1 [S,O1] and y2 [S,O2] read back from
 * cached memory. Returns 0 -> caller falls back to the single-matmul path. */
int coli_vk_matmul_pair(ColiVkTensor **t1p, float *y1, const void *w1, const float *s1, int O1,
                        ColiVkTensor **t2p, float *y2, const void *w2, const float *s2, int O2,
                        int fmt, const float *x, int S, int I, int grp) {
    if (!G.ready || S < 1) return 0;
    if (!upload_tensor(t1p, w1, s1, fmt, I, O1, grp) || !upload_tensor(t2p, w2, s2, fmt, I, O2, grp)) return 0;
    ColiVkTensor *t1 = *t1p, *t2 = *t2p;
    size_t xb = (size_t)S * I * 4, yb1 = (size_t)S * O1 * 4, yb2 = (size_t)S * O2 * 4;
    if (!scratch_reserve(&G.x, xb) || !scratch_reserve_mt(&G.y, yb1, G.memtype_cached) ||
        !scratch_reserve_mt(&G.y2, yb2, G.memtype_cached)) return 0;
    memcpy(G.x.ptr, x, xb);

    if (!G.pair_pool) {   /* one-time: a second 4-binding set (G.dset serves the first) */
        VkDescriptorPoolSize ps = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 4};
        VkDescriptorPoolCreateInfo dpi = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &ps};
        VKCHECK(vkCreateDescriptorPool(G.dev, &dpi, NULL, &G.pair_pool), "pair descPool");
        VkDescriptorSetAllocateInfo dsa = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = G.pair_pool, .descriptorSetCount = 1, .pSetLayouts = &G.dsl};
        VKCHECK(vkAllocateDescriptorSets(G.dev, &dsa, &G.dset_pair), "pair descSet");
    }
    VkDescriptorBufferInfo b1[4] = {
        {.buffer = G.x.buf, .range = VK_WHOLE_SIZE}, {.buffer = t1->wbuf, .range = VK_WHOLE_SIZE},
        {.buffer = t1->sbuf, .range = VK_WHOLE_SIZE}, {.buffer = G.y.buf, .range = VK_WHOLE_SIZE}};
    VkDescriptorBufferInfo b2[4] = {
        {.buffer = G.x.buf, .range = VK_WHOLE_SIZE}, {.buffer = t2->wbuf, .range = VK_WHOLE_SIZE},
        {.buffer = t2->sbuf, .range = VK_WHOLE_SIZE}, {.buffer = G.y2.buf, .range = VK_WHOLE_SIZE}};
    wr_desc(G.dset, 4, b1);
    wr_desc(G.dset_pair, 4, b2);

    VKCHECK(vkResetCommandBuffer(G.cmd, 0), "resetCmd");
    VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    VKCHECK(vkBeginCommandBuffer(G.cmd, &begin), "beginCmd");
    vkCmdBindPipeline(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe);
    struct PC pc1 = {fmt, S, I, O1, t1->rowWords, t1->gs};
    vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt, 0, 1, &G.dset, 0, NULL);
    vkCmdPushConstants(G.cmd, G.plyt, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc1), &pc1);
    vkCmdDispatch(G.cmd, (uint32_t)((O1 + 7) / 8), (uint32_t)S, 1);
    struct PC pc2 = {fmt, S, I, O2, t2->rowWords, t2->gs};
    vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt, 0, 1, &G.dset_pair, 0, NULL);
    vkCmdPushConstants(G.cmd, G.plyt, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc2), &pc2);
    vkCmdDispatch(G.cmd, (uint32_t)((O2 + 7) / 8), (uint32_t)S, 1);
    VKCHECK(vkEndCommandBuffer(G.cmd), "endCmd");

    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &G.cmd};
    VKCHECK(vkResetFences(G.dev, 1, &G.fence), "resetFence");
    double vp0 = G.eg_prof ? vk_now() : 0;
    VKCHECK(vk_submit(0, G.queue, &si, G.fence), "queueSubmit");
    if (G.eg_prof) { double vp1 = vk_now(); g_vsub_ms += vp1 - vp0; vp0 = vp1; }
    if (vk_fence_wait(G.dev, G.fence) != VK_SUCCESS) { G.ready = 0; return 0; }
    if (G.eg_prof) { g_vwait_ms += vk_now() - vp0; vkprof_tick(); }
    memcpy(y1, G.y.ptr, yb1);
    memcpy(y2, G.y2.ptr, yb2);
    G.cmd_ready = 0; G.bound_tensor = NULL;   /* the shared command buffer/binding was clobbered */
    return 1;
}


/* q-prep chain: [q_a + kv_a pair] -> rmsnorm(q_latent) -> [q_b], recorded in ONE
 * command buffer with compute barriers — one submit+fence where the engine paid
 * three (the middle CPU norm forced two roundtrips). Only q [S,Oqb] and the kv
 * latent [S,Okva] return to the host (RoPE + canonical KV append stay CPU-side).
 * The per-layer norm weights upload once into a tiny resident buffer (KV-mirror
 * pattern). All three tensors must share fmt (dense io is int8 in practice).
 * Returns 0 -> caller runs the 3-submit path (also when rmsnorm.spv is absent). */
int coli_vk_attn_qprep(int layer,
                       ColiVkTensor **qa,  const void *wqa,  const float *sqa,  int Oqa,
                       ColiVkTensor **kva, const void *wkva, const float *skva, int Okva,
                       ColiVkTensor **qb,  const void *wqb,  const float *sqb,  int Oqb,
                       int fmt, int grp, const float *lnw, float eps,
                       const float *x, int S, int I, float *q_out, float *kv_out, float *lat_out) {
    if (!G.ready || !G.shader_nrm || S < 1 || layer < 0 || layer >= VK_KV_LAYERS) return 0;
    if (!upload_tensor(qa, wqa, sqa, fmt, I, Oqa, grp) || !upload_tensor(kva, wkva, skva, fmt, I, Okva, grp) ||
        !upload_tensor(qb, wqb, sqb, fmt, Oqa, Oqb, grp)) return 0;
    ColiVkTensor *tqa = *qa, *tkv = *kva, *tqb = *qb;
    if (!G.lnbuf[layer]) {                       /* resident norm weights, uploaded once */
        void *lp = NULL; float p0 = G.prio; G.prio = 1.0f;
        int ok = g_up[0].on ? alloc_dev((size_t)Oqa * 4, &G.lnbuf[layer], &G.lnmem[layer])
                            : alloc_hostvis((size_t)Oqa * 4, &G.lnbuf[layer], &G.lnmem[layer], &lp);
        G.prio = p0;
        if (!ok) { G.lnbuf[layer] = VK_NULL_HANDLE; return 0; }
        if (lp) memcpy(lp, lnw, (size_t)Oqa * 4);
        else if (!pw_add(G.lnbuf[layer], 0, lnw, (size_t)Oqa * 4)) {   /* staged: rides this submit */
            vkDestroyBuffer(G.dev, G.lnbuf[layer], NULL); vkFreeMemory(G.dev, G.lnmem[layer], NULL);
            G.lnbuf[layer] = VK_NULL_HANDLE; return 0;
        }
        G.lnlen[layer] = Oqa;
    }
    if (G.lnlen[layer] != Oqa) return 0;
    size_t xb = (size_t)S * I * 4, qb_b = (size_t)S * Oqb * 4, kvb_b = (size_t)S * Okva * 4;
    size_t lat = (size_t)S * Oqa * 4;
    if (!scratch_reserve(&G.x, xb) || !scratch_reserve_mt(&G.y, qb_b, G.memtype_cached) ||
        !scratch_reserve_mt(&G.y2, kvb_b, G.memtype_cached) ||
        !scratch_reserve(&G.qp1, lat) ||
        !scratch_reserve_mt(&G.qp2, lat, G.memtype_cached)) return 0;   /* normed latent reads back (DSA indexer) */
    memcpy(G.x.ptr, x, xb);

    VkDescriptorBufferInfo b1[4] = {
        {.buffer = G.x.buf, .range = VK_WHOLE_SIZE}, {.buffer = tqa->wbuf, .range = VK_WHOLE_SIZE},
        {.buffer = tqa->sbuf, .range = VK_WHOLE_SIZE}, {.buffer = G.qp1.buf, .range = VK_WHOLE_SIZE}};
    VkDescriptorBufferInfo b2[4] = {
        {.buffer = G.x.buf, .range = VK_WHOLE_SIZE}, {.buffer = tkv->wbuf, .range = VK_WHOLE_SIZE},
        {.buffer = tkv->sbuf, .range = VK_WHOLE_SIZE}, {.buffer = G.y2.buf, .range = VK_WHOLE_SIZE}};
    VkDescriptorBufferInfo bn[3] = {
        {.buffer = G.qp1.buf, .range = VK_WHOLE_SIZE}, {.buffer = G.lnbuf[layer], .range = VK_WHOLE_SIZE},
        {.buffer = G.qp2.buf, .range = VK_WHOLE_SIZE}};
    VkDescriptorBufferInfo b3[4] = {
        {.buffer = G.qp2.buf, .range = VK_WHOLE_SIZE}, {.buffer = tqb->wbuf, .range = VK_WHOLE_SIZE},
        {.buffer = tqb->sbuf, .range = VK_WHOLE_SIZE}, {.buffer = G.y.buf, .range = VK_WHOLE_SIZE}};
    if (!G.pair_pool) {   /* the chain reuses the pair's 2nd matmul set */
        VkDescriptorPoolSize ps = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 4};
        VkDescriptorPoolCreateInfo dpi = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &ps};
        VKCHECK(vkCreateDescriptorPool(G.dev, &dpi, NULL, &G.pair_pool), "pair descPool");
        VkDescriptorSetAllocateInfo dsa = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = G.pair_pool, .descriptorSetCount = 1, .pSetLayouts = &G.dsl};
        VKCHECK(vkAllocateDescriptorSets(G.dev, &dsa, &G.dset_pair), "pair descSet");
    }
    wr_desc(G.dset, 4, b1); wr_desc(G.dset_pair, 4, b2); wr_desc(G.dset_qp3, 4, b3);
    wr_desc(G.dset_nrm, 3, bn);

    VKCHECK(vkResetCommandBuffer(G.cmd, 0), "resetCmd");
    VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    VKCHECK(vkBeginCommandBuffer(G.cmd, &begin), "beginCmd");
    pw_record(G.cmd);   /* staged: the norm weights (and any pending mirror rows) first */
    VkMemoryBarrier mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_SHADER_READ_BIT};
    vkCmdBindPipeline(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe);
    struct PC pc1 = {fmt, S, I, Oqa, tqa->rowWords, tqa->gs};
    vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt, 0, 1, &G.dset, 0, NULL);
    vkCmdPushConstants(G.cmd, G.plyt, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc1), &pc1);
    vkCmdDispatch(G.cmd, (uint32_t)((Oqa + 7) / 8), (uint32_t)S, 1);
    struct PC pc2 = {fmt, S, I, Okva, tkv->rowWords, tkv->gs};
    vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt, 0, 1, &G.dset_pair, 0, NULL);
    vkCmdPushConstants(G.cmd, G.plyt, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc2), &pc2);
    vkCmdDispatch(G.cmd, (uint32_t)((Okva + 7) / 8), (uint32_t)S, 1);
    vkCmdPipelineBarrier(G.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &mb, 0, NULL, 0, NULL);
    vkCmdBindPipeline(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe_nrm);
    struct PCN pcn = {S, Oqa, eps};
    vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt_nrm, 0, 1, &G.dset_nrm, 0, NULL);
    vkCmdPushConstants(G.cmd, G.plyt_nrm, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pcn), &pcn);
    vkCmdDispatch(G.cmd, (uint32_t)S, 1, 1);
    vkCmdPipelineBarrier(G.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &mb, 0, NULL, 0, NULL);
    vkCmdBindPipeline(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe);
    struct PC pc3 = {fmt, S, Oqa, Oqb, tqb->rowWords, tqb->gs};
    vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt, 0, 1, &G.dset_qp3, 0, NULL);
    vkCmdPushConstants(G.cmd, G.plyt, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc3), &pc3);
    vkCmdDispatch(G.cmd, (uint32_t)((Oqb + 7) / 8), (uint32_t)S, 1);
    VKCHECK(vkEndCommandBuffer(G.cmd), "endCmd");

    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &G.cmd};
    VKCHECK(vkResetFences(G.dev, 1, &G.fence), "resetFence");
    double vp0 = G.eg_prof ? vk_now() : 0;
    VKCHECK(vk_submit(0, G.queue, &si, G.fence), "queueSubmit");
    if (G.eg_prof) { double vp1 = vk_now(); g_vsub_ms += vp1 - vp0; vp0 = vp1; }
    if (vk_fence_wait(G.dev, G.fence) != VK_SUCCESS) { G.ready = 0; return 0; }
    pw_done();
    if (G.eg_prof) { g_vwait_ms += vk_now() - vp0; vkprof_tick(); }
    memcpy(q_out, G.y.ptr, qb_b);
    memcpy(kv_out, G.y2.ptr, kvb_b);
    if (lat_out) memcpy(lat_out, G.qp2.ptr, lat);
    G.cmd_ready = 0; G.bound_tensor = NULL;   /* the shared command buffer/binding was clobbered */
    return 1;
}

/* Fused absorb attention + o-projection in ONE submit: the absorb kernel writes ctx
 * [S,H*V] to a device-only scratch, a barrier, then the resident o_proj ([Dout, H*V])
 * runs on it via the plain matmul pipeline — only out [S,Dout] returns to the host.
 * Kills the per-layer ctx readback + re-upload + second submit of the unfused path.
 * Returns 0 -> caller falls back (plain absorb or CPU). */
int coli_vk_attention_absorb_project(ColiVkTensor **kvb, const void *w, const float *sc, int fmt, int grp,
                                     ColiVkTensor **ot, const void *ow, const float *osc, int ofmt, int ogrp,
                                     float *out, const float *q, int layer, int S, int H,
                                     int Q, int R, int V, int K, int st0, int T, float scale,
                                     int Dout) {
    if (!G.ready || !G.pipe_att || S < 1 || H < 1 || layer < 0 || layer >= VK_KV_LAYERS) return 0;
    if (Q > 256 || R > 64 || K > 512 || st0 < 0 || T - S - st0 < 0 || Dout < 1) return 0;
    VkKvLayer *kv = &G.kv[layer];
    if (!kv->bl || kv->rows < T || kv->K != K || kv->R != R) return 0;
    if (!upload_tensor(kvb, w, sc, fmt, K, H * (Q + V), grp)) return 0;
    if (!upload_tensor(ot, ow, osc, ofmt, H * V, Dout, ogrp)) return 0;
    ColiVkTensor *t = *kvb, *to = *ot;
    int cap = T - st0;
    size_t qb = (size_t)S * H * (Q + R) * 4, cb = (size_t)S * H * V * 4;
    size_t sb = (size_t)S * H * cap * 4, ob = (size_t)S * Dout * 4;
    if (!scratch_reserve(&G.x, qb) || !scratch_reserve(&G.att_ctx, cb) ||
        !scratch_reserve(&G.att_sc, sb) || !scratch_reserve_mt(&G.y, ob, G.memtype_cached)) return 0;
    memcpy(G.x.ptr, q, qb);

    VkDescriptorBufferInfo bi[7] = {
        {.buffer = G.x.buf, .range = VK_WHOLE_SIZE}, {.buffer = t->wbuf, .range = VK_WHOLE_SIZE},
        {.buffer = t->sbuf, .range = VK_WHOLE_SIZE}, {.buffer = kv->bl, .range = VK_WHOLE_SIZE},
        {.buffer = kv->br, .range = VK_WHOLE_SIZE}, {.buffer = G.att_sc.buf, .range = VK_WHOLE_SIZE},
        {.buffer = G.att_ctx.buf, .range = VK_WHOLE_SIZE}};
    VkWriteDescriptorSet wd[7];
    for (int i = 0; i < 7; i++) wd[i] = (VkWriteDescriptorSet){
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = G.dset_att,
        .dstBinding = (uint32_t)i, .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &bi[i]};
    vkUpdateDescriptorSets(G.dev, 7, wd, 0, NULL);
    VkDescriptorBufferInfo oi[4] = {
        {.buffer = G.att_ctx.buf, .range = VK_WHOLE_SIZE}, {.buffer = to->wbuf, .range = VK_WHOLE_SIZE},
        {.buffer = to->sbuf, .range = VK_WHOLE_SIZE}, {.buffer = G.y.buf, .range = VK_WHOLE_SIZE}};
    wr_desc(G.dset, 4, oi);

    VKCHECK(vkResetCommandBuffer(G.cmd, 0), "resetCmd");
    VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    VKCHECK(vkBeginCommandBuffer(G.cmd, &begin), "beginCmd");
    pw_record(G.cmd);   /* staged: the mirror's pending rows first */
    vkCmdBindPipeline(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe_att);
    vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt_att, 0, 1, &G.dset_att, 0, NULL);
    struct PCAttn pc = {fmt, S, H, Q, R, V, K, st0, T, t->rowWords, cap, scale, t->gs};
    vkCmdPushConstants(G.cmd, G.plyt_att, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(G.cmd, (uint32_t)H, (uint32_t)S, 1);
    VkMemoryBarrier mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_SHADER_READ_BIT};
    vkCmdPipelineBarrier(G.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    vkCmdBindPipeline(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe);
    vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt, 0, 1, &G.dset, 0, NULL);
    struct PC opc = {ofmt, S, H * V, Dout, to->rowWords, to->gs};
    vkCmdPushConstants(G.cmd, G.plyt, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(opc), &opc);
    vkCmdDispatch(G.cmd, (uint32_t)((Dout + 7) / 8), (uint32_t)S, 1);
    VKCHECK(vkEndCommandBuffer(G.cmd), "endCmd");

    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &G.cmd};
    VKCHECK(vkResetFences(G.dev, 1, &G.fence), "resetFence");
    double vp0 = G.eg_prof ? vk_now() : 0;
    VKCHECK(vk_submit(0, G.queue, &si, G.fence), "queueSubmit");
    if (G.eg_prof) { double vp1 = vk_now(); g_vsub_ms += vp1 - vp0; vp0 = vp1; }
    if (vk_fence_wait(G.dev, G.fence) != VK_SUCCESS) { G.ready = 0; return 0; }
    pw_done();
    if (G.eg_prof) { g_vwait_ms += vk_now() - vp0; vkprof_tick(); }
    memcpy(out, G.y.ptr, ob);
    G.cmd_ready = 0; G.bound_tensor = NULL;   /* the shared command buffer/binding was clobbered */
    return 1;
}

/* A tensor's buffers and ranges back to its pool. If the device was lost or disabled
 * (the fence-timeout path sets ready=0), a submission may still reference these
 * buffers: do NOT vkDestroy into a dead device (GPU-side UAF) nor hand the range out
 * again; leak the GPU side (we are degrading to CPU for the rest of the run) and
 * reclaim the host struct and counters only. */
static void tensor_release(ColiVkTensor *t) {
    VkWPool *P = t->pool;
    if (t->imp) {   /* rows imported in place: their memory object, the scales' range */
        pthread_mutex_lock(&g_imports_mx);
        for (ColiVkTensor **pp = &g_imports; *pp; pp = &(*pp)->imp_next) if (*pp == t) { *pp = t->imp_next; break; }
        pthread_mutex_unlock(&g_imports_mx);
        size_t sb = scale_floats(t->fmt, t->I, t->O, t->gs) * sizeof(float);
        if (G.ready) {
            vkDestroyBuffer(G.dev, t->wbuf, NULL);
            vkFreeMemory(G.dev, t->imp, NULL);
            if (t->sbuf) vkDestroyBuffer(G.dev, t->sbuf, NULL);
            pool_free_range(P, t->sr);
        }
        __atomic_sub_fetch(&P->bytes, sb, __ATOMIC_RELAXED);
        __atomic_sub_fetch(&P->tensors, 1, __ATOMIC_RELAXED);
        __atomic_sub_fetch(&G.tensor_count, 1, __ATOMIC_RELAXED);
        __atomic_sub_fetch(&G.used_bytes, sb, __ATOMIC_RELAXED);
        __atomic_sub_fetch(&g_import_bytes, t->wbytes, __ATOMIC_RELAXED);
        free(t);
        return;
    }
    if (t->dev == 1 ? G2.ready : G.ready) {
        VkDevice dev = pool_device(P);
        if (t->wbuf) vkDestroyBuffer(dev, t->wbuf, NULL);
        if (t->sbuf) vkDestroyBuffer(dev, t->sbuf, NULL);
        pool_free_range(P, t->wr);
        pool_free_range(P, t->sr);
    }
    // Mirror tensor_alloc exactly (rows + scales), atomically: frees run from
    // expert_load under `#pragma omp parallel` and from the tier's uploader.
    size_t b = t->wbytes + scale_floats(t->fmt, t->I, t->O, t->gs) * sizeof(float);
    __atomic_sub_fetch(&P->bytes, b, __ATOMIC_RELAXED);
    __atomic_sub_fetch(&P->tensors, 1, __ATOMIC_RELAXED);
    if (P->counted) {
        __atomic_sub_fetch(&G.tensor_count, 1, __ATOMIC_RELAXED);
        __atomic_sub_fetch(&G.used_bytes, b, __ATOMIC_RELAXED);
    }
    up_img_free(t->img, t->img_al);
    free(t);
}
/* Async work in flight per device (the GLM expert group, the dev2 group, the expert
 * tier's batch). A tensor freed meanwhile may be one that work still reads, and its
 * range would be the next upload's: it waits on this list until the device is idle
 * again (tensor_reap, at every join). */
static int g_async_inflight[2];
static ColiVkTensor *g_deferred;
static pthread_mutex_t g_deferred_mx = PTHREAD_MUTEX_INITIALIZER;
static void async_begin(int dev) { __atomic_add_fetch(&g_async_inflight[dev], 1, __ATOMIC_ACQ_REL); }
static void tensor_reap(int dev) {
    if (__atomic_load_n(&g_async_inflight[dev], __ATOMIC_ACQUIRE)) return;
    pthread_mutex_lock(&g_deferred_mx);
    ColiVkTensor *mine = NULL, **pp = &g_deferred;
    while (*pp) {
        ColiVkTensor *t = *pp;
        if (t->dev == dev) { *pp = t->next_free; t->next_free = mine; mine = t; }
        else pp = &t->next_free;
    }
    pthread_mutex_unlock(&g_deferred_mx);
    while (mine) { ColiVkTensor *n = mine->next_free; tensor_release(mine); mine = n; }
}
static void async_end(int dev) {
    __atomic_sub_fetch(&g_async_inflight[dev], 1, __ATOMIC_ACQ_REL);
    tensor_reap(dev);
}

void coli_vk_tensor_free(ColiVkTensor *t) {
    if (!t) return;
    if (t->dev == 0 && t->pool != &g_tpool && G.bound_tensor == t) { G.bound_tensor = NULL; G.cmd_ready = 0; }  /* drop stale cache */
    if (__atomic_load_n(&g_async_inflight[t->dev], __ATOMIC_ACQUIRE)) {
        pthread_mutex_lock(&g_deferred_mx);
        t->next_free = g_deferred; g_deferred = t;
        pthread_mutex_unlock(&g_deferred_mx);
        return;
    }
    tensor_release(t);
}

static VkWPool *pool_of(int which) {
    return which == 1 ? &g_tpool : which == 2 ? &g_wpool2 : which == 3 ? &g_tpool2 : which == 4 ? &g_xpool : &g_wpool;
}
void coli_vk_pool_stats(int which, ColiVkPoolStats *st) {
    VkWPool *P = pool_of(which);
    VkaStats v;
    pthread_mutex_lock(&P->mx);
    vka_stats(&P->p, &v);
    st->limit = P->p.limit; st->refusals = P->p.refusals;
    st->allocs = P->p.allocs; st->frees = P->p.frees;
    pthread_mutex_unlock(&P->mx);
    st->blocks = v.blocks; st->live = v.live; st->total = v.total; st->used = v.used;
    st->free = v.free; st->largest_free = v.largest_free; st->peak_used = v.peak_used; st->frag = v.frag;
    st->tensors = __atomic_load_n(&P->tensors, __ATOMIC_RELAXED);
    st->payload = __atomic_load_n(&P->bytes, __ATOMIC_RELAXED);
}
static void tier_pool_limit(VkWPool *P, size_t bytes) {
    pthread_mutex_lock(&P->mx);
    P->p.limit = bytes;
    /* blocks no larger than the budget, so a small budget is not one refused block */
    size_t bb = coli_vk_block_bytes_dev(P->dev, VK_WBLOCK);
    P->p.block_bytes = bytes && bytes < bb ? ((bytes + 4095) & ~(size_t)4095) : bb;
    /* the empty block the pool keeps as a spare (vka_free) goes too: a tier started
     * again after a shutdown would otherwise fill it past its new, smaller budget */
    for (int k = 0; k < P->p.nb; k++)
        if (P->p.b[k].present && !P->p.b[k].live) pool_release_block(P, k);
    pthread_mutex_unlock(&P->mx);
}
void coli_vk_tier_pool_limit(size_t bytes) { tier_pool_limit(&g_tpool, bytes); }
void coli_vk_tier_pool_limit_dev(int dev, size_t bytes) { tier_pool_limit(dev == 1 ? &g_tpool2 : &g_tpool, bytes); }
void coli_vk_tier_extra_pool_limit(size_t bytes) { tier_pool_limit(&g_xpool, bytes); }

size_t coli_vk_tensor_bytes(const ColiVkTensor *t) { return t ? t->wbytes : 0; }

/* Is the device we picked an integrated GPU? The backend already computes this
 * at device selection (see the deviceType ranking above) but only uses it to
 * rank candidates. The RAM planner needs it too: on an integrated GPU our
 * HOST_VISIBLE|DEVICE_LOCAL allocations are the same physical memory the host
 * expert cache draws from, so whoever sizes that cache has to know. Mirrors
 * coli_cuda_device_integrated() (#653) for the Vulkan path. */
int coli_vk_device_integrated(void) {
    if (!G.phys) return 0;
    VkPhysicalDeviceProperties p;
    vkGetPhysicalDeviceProperties(G.phys, &p);
    return p.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? 1 : 0;
}
/* An integrated GPU or a CPU device (Lavapipe): its "device memory" is host RAM. */
int coli_vk_device_shares_ram(void) {
    if (!G.phys) return 0;
    VkPhysicalDeviceProperties p;
    vkGetPhysicalDeviceProperties(G.phys, &p);
    return p.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU || p.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU;
}
/* The largest DEVICE_LOCAL heap, for a budget when VK_EXT_memory_budget is absent. */
size_t coli_vk_device_local_bytes(void) {
    if (!G.phys) return 0;
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(G.phys, &mp);
    VkDeviceSize m = 0;
    for (uint32_t i = 0; i < mp.memoryHeapCount; i++)
        if ((mp.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) && mp.memoryHeaps[i].size > m) m = mp.memoryHeaps[i].size;
    if (g_mem.cap[0] && g_mem.cap[0] < m) m = g_mem.cap[0];   /* COLI_VK_DEVICE_CAP_MB */
    return (size_t)m;
}
const char *coli_vk_device_name(void) {
    static VkPhysicalDeviceProperties p;
    if (!G.phys) return "";
    vkGetPhysicalDeviceProperties(G.phys, &p);
    return p.deviceName;
}

/* ==================== ASYNC EXPERT BATCH (the routed-expert tier) ====================
 * One layer step of resident routed experts in ONE submit that the caller does not
 * wait for: per expert, its activation rows run gate+up+act then down, every expert
 * of the step recorded into one command buffer on the tier queue (G.tqueue), and
 * coli_vk_xb_join hands back one output row per assignment for the caller to add in
 * its own routing order. vk_tier.c drives it; nothing else does.
 *
 * The batch has a context per device (XbCtx): g_xb[0] on the primary device G, g_xb[1]
 * on COLI_VK_DEV2's second device G2 (its only queue, its own pipelines and scratch),
 * so the tier can hold experts on both and run a step's two batches at once. The
 * second device takes no cooperative-matrix route (its device is created without the
 * features) and no sub-batches (streaming runs on the primary device).
 *
 * Two routes per expert, chosen by its row count:
 *   - up to gemm_rows-1 rows: the fused gate+up+act GEMV (qmatmul_gate_up.comp, one
 *     subgroup per output row), then the down GEMV (qmatmul.comp). A row's bits do not
 *     depend on how many rows share the dispatch, so a decode step and a two-row MTP
 *     verify give an expert the same output;
 *   - from gemm_rows rows (prefill): gate and up through the tiled GEMM
 *     (qmatmul_gemm.comp), the activation on its own (expert_act.comp), down through
 *     the GEMM again: each weight is fetched once per tile instead of once per row.
 * Every format qmatmul.comp reads works on both routes (gate and up share one).
 *
 * The cost per step is kept to recording: an expert's descriptor sets are written
 * once, when it becomes resident (coli_vk_xb_expert). Its weights are static
 * bindings; the scratch (x rows, gate/up/hidden rows, y rows) are DYNAMIC storage
 * buffers whose offsets ride vkCmdBindDescriptorSets, each expert's slice at a
 * minStorageBufferOffsetAlignment boundary. A scratch that has to grow rewrites the
 * sets of every live expert, once. */
#define XB_POOL_EXPERTS 128
struct ColiVkExpert {
    ColiVkTensor *g, *u, *d;
    VkDescriptorSet s_gu, s_dn, s_g, s_u;
    int dpool;
    ColiVkExpert *prev, *next;
    struct XbCtx *X;               /* the device's batch it belongs to */
};
typedef struct { VkBuffer buf; VkDeviceMemory mem; void *ptr; size_t region; } XbBuf;
typedef struct XbCtx {
    /* the device (0 = G, 1 = G2) and what the batch needs of it, set by xb_bind */
    int d;
    VkPhysicalDevice phys; VkDevice dev; VkQueue q;
    uint32_t q_fam, ts_bits; float ts_period; int q_shared, has_budget;
    size_t ssbo_align, ssbo_range;
    uint32_t mt_host, mt_dev, mt_cached;
    VkGemmTile gemm_t[VK_GEMM_SLOTS]; int gemm_has[VK_GEMM_SLOTS], gemm_shader;
    char spv[1024];
    int ready, inflight;
    int D, I, act; float limit, a, b;
    VkCommandPool cpool; VkCommandBuffer cmd; VkFence fence;
    VkQueryPool qpool; int has_ts;
    VkShaderModule sh_act, sh_gu, sh_mv, sh_mm, sh_coop;
    VkDescriptorSetLayout dsl6, dsl4, dsl3;
    VkPipelineLayout pl6, pl4, pl3;
    VkPipeline p_gu, p_mv, p_act, p_mm[VK_GEMM_SLOTS], p_coop[VK_COOP_SLOTS];
    VkDescriptorPool *dpools; int ndpools;
    VkDescriptorPool act_pool; VkDescriptorSet act_set;
    XbBuf x, g, u, h, y;          /* x host-written, g/u/h device-only, y host-read */
    size_t align;
    int gemm_rows;                /* rows from which an expert takes the GEMM route (0 = never) */
    int coop_rows;
    unsigned long long cooperative_matmuls;
    ColiVkExpert *live;
    /* the batch in flight: where each assignment row's output sits */
    size_t *yoff; int nrows, cyoff;
    unsigned long long batches, experts, rows, gemm_experts;
    double dev_ms;
    /* COLI_VK_ACT_SWIGLU_V4: expert_act_v4.spv between the activation and down, its
     * row weights in the x buffer after each expert's rows (xb_v4_init) */
    VkShaderModule sh_v4; VkDescriptorSetLayout dsl_v4; VkPipelineLayout pl_v4; VkPipeline p_v4;
    VkDescriptorPool v4_pool; VkDescriptorSet v4_set;
    /* sub-batches (coli_vk_xb_sub_*, the tier's big prefill steps): two halves of the
     * scratch, each with its own command buffer, fence and timestamps, up to two in flight */
    struct { VkCommandBuffer cmd; VkFence fence; VkQueryPool qp; int inflight, nrows, cyoff; size_t *yoff; } sub[2];
    size_t sub_x, sub_i, sub_y;           /* a half's bytes of each scratch (0: not reserved) */
    unsigned long long sub_batches, sub_experts, sub_rows, sub_gemm;
    double sub_ms;
    /* The grouped GEMM (qmatmul_grp.comp, [0] down, [1] gate+up+SwiGLU): every expert of
     * a batch in one dispatch per phase, the weights read through an address table. The
     * primary device with cooperative matrices, subgroups of 64 and bufferDeviceAddress
     * (xb_grp_init). Tables and sets per batch that can be in flight: [0] the single
     * batch, [1 + h] sub-batch half h. */
    VkShaderModule sh_grp[2]; VkDescriptorSetLayout dsl_grp; VkPipelineLayout pl_grp;
    VkPipeline p_grp[2]; VkDescriptorPool grp_pool; VkDescriptorSet grp_set[3][2];
    VkShaderModule sh_gv[2]; VkPipeline p_gv[2];   /* qmatmul_grp_gemv.comp: the same batches below grp_rows */
    XbBuf grp_it[3][2], grp_et[3][2], grp_map[3];
    int grp_rows;
    unsigned long long grp_batches, gv_batches;
    /* a big step's whole-step buffers (coli_vk_xb_step_*): its token rows once (xt), every
     * assignment's output row in place (ys, s*K + k), the routed sum (out) and its inputs */
    struct {
        int on, S, K; const float *xh;
        XbBuf xt, ys, out, w, use;
        VkShaderModule sh; VkDescriptorSetLayout dsl; VkPipelineLayout pl; VkPipeline pipe;
        VkDescriptorPool pool; VkDescriptorSet set;
        unsigned long long steps, sums; double sum_ms;
    } st;
} XbCtx;
static XbCtx g_xb[2];
static int xb_dev_ready(const XbCtx *X) { return X->d ? G2.ready : G.ready; }
struct PCV4 { int rows, n; };
struct PCGRP { int I, O; float limit; int ibase; int kgat; int rpw; };   /* qmatmul_grp.comp, qmatmul_grp_gemv.comp (rpw) */
#define XB_GV_RPW 64   /* outputs per grouped-GEMV workgroup */
#define XB_GRP_TT 4   /* 16-row tiles of an expert per workgroup: 64 rows (2, 4 or 8) */

static size_t xb_up(size_t v, size_t a) { return (v + a - 1) / a * a; }

static int xb_buf(XbCtx *X, XbBuf *b, size_t region, uint32_t memtype, int map) {
    if (b->buf) { vkDestroyBuffer(X->dev, b->buf, NULL); vkFreeMemory(X->dev, b->mem, NULL); }
    memset(b, 0, sizeof(*b));
    int ok;
    if (X->d) ok = alloc_hostvis_d2(2 * region, &b->buf, &b->mem, map ? &b->ptr : NULL, memtype);
    else {
        float p0 = G.prio; G.prio = 1.0f;        /* rides every batch: never evict */
        ok = alloc_hostvis_mt(2 * region, &b->buf, &b->mem, map ? &b->ptr : NULL, memtype);
        G.prio = p0;
    }
    if (!ok) { memset(b, 0, sizeof(*b)); return 0; }
    b->region = region;
    return 1;
}

/* An expert's four sets: gate+up GEMV (x, Wg, sg, Wu, su, h), down (h, Wd, sd, y), and
 * the GEMM route's gate (x, Wg, sg, g) and up (x, Wu, su, u). Dynamic windows are the
 * buffers' regions; the offsets come at bind time. */
static void xb_write_sets(ColiVkExpert *e) {
    XbCtx *X = e->X;
    VkDescriptorBufferInfo gu[6] = {
        {X->x.buf, 0, X->x.region}, {e->g->wbuf, 0, VK_WHOLE_SIZE}, {e->g->sbuf, 0, VK_WHOLE_SIZE},
        {e->u->wbuf, 0, VK_WHOLE_SIZE}, {e->u->sbuf, 0, VK_WHOLE_SIZE}, {X->h.buf, 0, X->h.region}};
    VkDescriptorBufferInfo dn[4] = {
        {X->h.buf, 0, X->h.region}, {e->d->wbuf, 0, VK_WHOLE_SIZE}, {e->d->sbuf, 0, VK_WHOLE_SIZE},
        {X->y.buf, 0, X->y.region}};
    VkDescriptorBufferInfo mg[4] = {
        {X->x.buf, 0, X->x.region}, {e->g->wbuf, 0, VK_WHOLE_SIZE}, {e->g->sbuf, 0, VK_WHOLE_SIZE},
        {X->g.buf, 0, X->g.region}};
    VkDescriptorBufferInfo mu[4] = {
        {X->x.buf, 0, X->x.region}, {e->u->wbuf, 0, VK_WHOLE_SIZE}, {e->u->sbuf, 0, VK_WHOLE_SIZE},
        {X->u.buf, 0, X->u.region}};
    VkWriteDescriptorSet w[18]; int n = 0;
    struct { VkDescriptorSet set; VkDescriptorBufferInfo *bi; int nb, d0, d1; } sets[4] = {
        {e->s_gu, gu, 6, 0, 5}, {e->s_dn, dn, 4, 0, 3}, {e->s_g, mg, 4, 0, 3}, {e->s_u, mu, 4, 0, 3}};
    for (int k = 0; k < 4; k++)
        for (int b = 0; b < sets[k].nb; b++)
            w[n++] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = sets[k].set, .dstBinding = (uint32_t)b, .descriptorCount = 1,
                .descriptorType = (b == sets[k].d0 || b == sets[k].d1) ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC
                                                                        : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo = &sets[k].bi[b]};
    vkUpdateDescriptorSets(X->dev, (uint32_t)n, w, 0, NULL);
}
static void xb_write_act_set(XbCtx *X) {
    VkDescriptorBufferInfo bi[3] = {{X->g.buf, 0, X->g.region}, {X->u.buf, 0, X->u.region}, {X->h.buf, 0, X->h.region}};
    VkWriteDescriptorSet w[3];
    for (int b = 0; b < 3; b++)
        w[b] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = X->act_set,
            .dstBinding = (uint32_t)b, .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, .pBufferInfo = &bi[b]};
    vkUpdateDescriptorSets(X->dev, 3, w, 0, NULL);
}
/* expert_act_v4's set: the hidden rows in place, the weights from the x buffer */
static void xb_write_v4_set(XbCtx *X) {
    VkDescriptorBufferInfo bi[2] = {{X->h.buf, 0, X->h.region}, {X->x.buf, 0, X->x.region}};
    VkWriteDescriptorSet w[2];
    for (int b = 0; b < 2; b++)
        w[b] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = X->v4_set,
            .dstBinding = (uint32_t)b, .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, .pBufferInfo = &bi[b]};
    vkUpdateDescriptorSets(X->dev, 2, w, 0, NULL);
}

/* Make every scratch region hold `xr`/`ir`/`yr` bytes (offsets of a batch stay below
 * its region; the buffer is twice that, so offset + window always fits). Grows by
 * half again at least; rewrites every live expert's sets when a buffer moved. */
static int xb_sizes(XbCtx *X, size_t xr, size_t ir, size_t yr, size_t *nx, size_t *ni, size_t *ny) {
    *nx = xr > X->x.region ? xb_up(xr + xr / 2, 4096) : X->x.region;
    *ni = ir > X->g.region ? xb_up(ir + ir / 2, 4096) : X->g.region;
    *ny = yr > X->y.region ? xb_up(yr + yr / 2, 4096) : X->y.region;
    if (*nx > X->ssbo_range) *nx = X->ssbo_range & ~(size_t)4095;
    if (*ni > X->ssbo_range) *ni = X->ssbo_range & ~(size_t)4095;
    if (*ny > X->ssbo_range) *ny = X->ssbo_range & ~(size_t)4095;
    return xr <= *nx && ir <= *ni && yr <= *ny;
}
static int xb_reserve(XbCtx *X, size_t xr, size_t ir, size_t yr) {
    if (xr <= X->x.region && ir <= X->g.region && yr <= X->y.region) return 1;
    size_t nx, ni, ny;
    if (!xb_sizes(X, xr, ir, yr, &nx, &ni, &ny)) return 0;   /* one window cannot address it */
    int ok = 1;
    if (nx != X->x.region) ok &= xb_buf(X, &X->x, nx, X->mt_host, 1);
    if (ni != X->g.region) ok &= xb_buf(X, &X->g, ni, X->mt_dev, 0) && xb_buf(X, &X->u, ni, X->mt_dev, 0) &&
                                 xb_buf(X, &X->h, ni, X->mt_dev, 0);
    if (ny != X->y.region) ok &= xb_buf(X, &X->y, ny, X->mt_cached, 1);
    if (!ok) {   /* out of device memory: no batch from here on */
        fprintf(stderr, "[VK] %sexpert batch: scratch of %.1f MiB failed, %s\n", X->d ? "dev2 " : "",
                (double)(nx + 3 * ni + ny) * 2 / 1048576.0, X->d ? "the second device stops" : "the tier stops");
        X->ready = 0;
        return 0;
    }
    for (ColiVkExpert *e = X->live; e; e = e->next) xb_write_sets(e);
    xb_write_act_set(X);
    if (X->v4_set) xb_write_v4_set(X);
    return 1;
}

static int xb_layout(XbCtx *X, int nbind, unsigned dynmask, size_t pc, VkDescriptorSetLayout *dsl, VkPipelineLayout *pl) {
    VkDescriptorSetLayoutBinding b[8];
    for (int i = 0; i < nbind; i++) b[i] = (VkDescriptorSetLayoutBinding){.binding = (uint32_t)i,
        .descriptorType = (dynmask >> i) & 1 ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT};
    VkDescriptorSetLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = (uint32_t)nbind, .pBindings = b};
    VKCHECK(vkCreateDescriptorSetLayout(X->dev, &li, NULL, dsl), "xb set layout");
    VkPushConstantRange r = {.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = (uint32_t)pc};
    VkPipelineLayoutCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1, .pSetLayouts = dsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &r};
    VKCHECK(vkCreatePipelineLayout(X->dev, &pi, NULL, pl), "xb pipeline layout");
    return 1;
}
static int xb_pipeline(XbCtx *X, VkShaderModule sh, VkPipelineLayout pl, const VkSpecializationInfo *si, VkPipeline *pipe) {
    VkComputePipelineCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                  .module = sh, .pName = "main", .pSpecializationInfo = si}, .layout = pl};
    VKCHECK(vkCreateComputePipelines(X->dev, VK_NULL_HANDLE, 1, &ci, NULL, pipe), "xb pipeline");
    return 1;
}

static void xb_coop_init(XbCtx *X) {
#ifdef VK_KHR_cooperative_matrix
    if (X->d) return;   /* the primary device's pipelines and tiles; the second one has none */
    const char *e = getenv("COLI_VK_TIER_COOP");
    if (!G.pipe_coop[0] || !X->p_mm[0] || (e && !atoi(e))) return;
    char path[1100]; derive_sibling(X->spv, "_coop.spv", path, sizeof path);
    X->sh_coop = load_spv(X->dev, path);
    if (!X->sh_coop) return;
    VkPhysicalDeviceProperties pp; vkGetPhysicalDeviceProperties(X->phys, &pp);
    for (int k = 0; k < VK_COOP_SLOTS; k++) {
        if (!G.pipe_coop[k]) continue;
        VkCoopTile t = G.coop_t[k];
        int nw = (t.bm / t.wm) * (t.bn / t.wn), nt = nw * G.coop_sg;
        size_t shared = (size_t)(t.bm + 2 * t.bn) * (t.bk + 8) * 2 +
                        (size_t)t.bm * 4 + (size_t)nw * 16 * VK_COOP_EST * 4 + (size_t)(nt + t.bn) * 4;
        if (shared > pp.limits.maxComputeSharedMemorySize || nt % (nt < t.bn ? 1 : nt / t.bn)) continue;
        int32_t sv[8] = {t.bm, t.bn, t.wm, t.wn, t.bk, G.coop_sg, nt, 1};
        VkSpecializationMapEntry me[8];
        for (int i = 0; i < 8; i++) me[i] = (VkSpecializationMapEntry){(uint32_t)i, (uint32_t)(i * 4), 4};
        VkSpecializationInfo si = {8, me, sizeof sv, sv};
        VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT rss = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO_EXT,
            .requiredSubgroupSize = (uint32_t)G.coop_sg};
        VkComputePipelineCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .pNext = &rss,
                .flags = VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT_EXT,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = X->sh_coop, .pName = "main",
                .pSpecializationInfo = &si}, .layout = X->pl4};
        if (vkCreateComputePipelines(X->dev, VK_NULL_HANDLE, 1, &ci, NULL, &X->p_coop[k]) != VK_SUCCESS)
            X->p_coop[k] = VK_NULL_HANDLE;
    }
    e = getenv("COLI_VK_TIER_COOP_ROWS");
    X->coop_rows = e && *e ? atoi(e) : 64;
    if (X->coop_rows < 1) X->coop_rows = 1;
#endif
}

/* The narrow tile wins on per-row and integer-group weights in expert shapes.
 * FP8/MXFP4 benefit from wider tiles, but rounding an awkward row count up to
 * another wide tile can cost more than the FP32 GEMM. Keep that fallback when
 * the cooperative tile would process more padded rows. */
static int xb_coop_slot(XbCtx *X, const ColiVkTensor *t, int rows, int gemm) {
    int k = t->fmt == 1 || t->fmt == 2 || t->fmt == 4 || t->fmt == 5 ? 0 : coop_slot(rows);
    if (!X->coop_rows || rows < X->coop_rows || !X->p_coop[k] || !coop_fmt(t)) return -1;
    if (t->fmt == 7 || t->fmt == 12) {
        int cbn = G.coop_t[k].bn, fbn = X->gemm_t[gemm].bn;
        if ((rows + cbn - 1) / cbn * cbn > (rows + fbn - 1) / fbn * fbn) return -1;
    }
    return k;
}

/* Keep each projection's format check: mixed gate/up/down quantization may make
 * only some projections eligible. Small verifies retain their scalar GEMV path. */
static VkPipeline xb_mm_pipeline(XbCtx *X, const ColiVkTensor *t, int rows, int slot,
                                 int *bm, int *bn, int *cooperative) {
    int k = xb_coop_slot(X, t, rows, slot);
    if (k >= 0) {
        *bm = G.coop_t[k].bm; *bn = G.coop_t[k].bn; (*cooperative)++;
        return X->p_coop[k];
    }
    *bm = X->gemm_t[slot].bm; *bn = X->gemm_t[slot].bn;
    return X->p_mm[slot];
}

/* COLI_VK_ACT_SWIGLU_V4's pass (expert_act_v4.spv), made the first time that
 * activation is asked for; 0 when its shader is missing. */
static int xb_v4_init(XbCtx *X) {
    if (X->p_v4) return 1;
    char path[1100];
    derive_dir_file(X->spv, "expert_act_v4.spv", path, sizeof path);
    if (!(X->sh_v4 = load_spv(X->dev, path))) return 0;
    if (!xb_layout(X, 2, 3u, sizeof(struct PCV4), &X->dsl_v4, &X->pl_v4) ||
        !xb_pipeline(X, X->sh_v4, X->pl_v4, NULL, &X->p_v4)) return 0;
    VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 2};
    VkDescriptorPoolCreateInfo dp = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &ps};
    VKCHECK(vkCreateDescriptorPool(X->dev, &dp, NULL, &X->v4_pool), "xb v4 pool");
    VkDescriptorSetAllocateInfo da = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = X->v4_pool, .descriptorSetCount = 1, .pSetLayouts = &X->dsl_v4};
    VKCHECK(vkAllocateDescriptorSets(X->dev, &da, &X->v4_set), "xb v4 set");
    if (X->h.buf && X->x.buf) xb_write_v4_set(X);
    return 1;
}

/* The device a context runs on, read at its first init: the primary device's tier
 * queue and the GEMM tiles its pipelines took, or the second device's one queue and
 * the default tiles its limits hold. 0 when that device is not up. */
static int xb_bind(XbCtx *X, int d) {
    if (d ? !G2.ready : !G.ready) return 0;
    X->d = d;
    if (!d) {
        X->phys = G.phys; X->dev = G.dev; X->q = G.tqueue; X->q_fam = G.tq_fam; X->ts_bits = G.tq_ts;
        X->ts_period = G.ts_period; X->q_shared = G.tq_shared; X->has_budget = G.has_budget;
        X->ssbo_align = G.ssbo_align; X->ssbo_range = G.ssbo_range;
        X->mt_host = G.memtype; X->mt_dev = G.memtype_dev; X->mt_cached = G.memtype_cached;
        for (int k = 0; k < VK_GEMM_SLOTS; k++) { X->gemm_t[k] = G.gemm_t[k]; X->gemm_has[k] = G.pipe_gemm[k] != VK_NULL_HANDLE; }
        X->gemm_shader = G.shader_gemm != VK_NULL_HANDLE;
        snprintf(X->spv, sizeof X->spv, "%s", G.spv_path);
        return G.shader_gu != VK_NULL_HANDLE;
    }
    X->phys = G2.phys; X->dev = G2.dev; X->q = G2.queue; X->q_fam = G2.qfam; X->ts_bits = G2.ts_bits;
    X->ts_period = G2.ts_period; X->q_shared = 0; X->has_budget = G2.has_budget;
    X->ssbo_align = G2.ssbo_align; X->ssbo_range = G2.ssbo_range;
    X->mt_host = G2.memtype; X->mt_dev = G2.memtype_dev; X->mt_cached = G2.memtype_cached;
    VkPhysicalDeviceProperties pp; vkGetPhysicalDeviceProperties(G2.phys, &pp);
    for (int k = 0; k < VK_GEMM_SLOTS; k++) { X->gemm_t[k] = gemm_tiles[k]; X->gemm_has[k] = gemm_tile_ok(gemm_tiles[k], &pp.limits); }
    X->gemm_shader = X->gemm_has[0];
    snprintf(X->spv, sizeof X->spv, "%s", G.spv_path);
    return 1;
}
/* The narrowest of the context's GEMM slots whose BN covers S, else the widest. */
static int xb_gemm_slot(const XbCtx *X, int S) {
    int k = 0;
    while (k + 1 < VK_GEMM_SLOTS && X->gemm_has[k + 1] && S > X->gemm_t[k].bn) k++;
    return k;
}

/* The grouped GEMM's pipelines (qmatmul_grp*.spv beside the main shader); leaves
 * X->grp_rows at 0 when the device or the files do not allow it. */
static void xb_grp_init(XbCtx *X) {
    const char *e = getenv("COLI_VK_XB_GROUPED");
    if (X->d || !G.has_bda || !G.has_coop || G.coop_sg != 64 || (e && *e == '0')) return;
    const char *nm[2] = {"qmatmul_grp.spv", "qmatmul_grp_gate_up.spv"};
    for (int v = 0; v < 2; v++) {
        char path[1100];
        derive_dir_file(X->spv, nm[v], path, sizeof path);
        if (!(X->sh_grp[v] = load_spv(X->dev, path))) return;
    }
    if (!xb_layout(X, 5, 0u, sizeof(struct PCGRP), &X->dsl_grp, &X->pl_grp)) return;
    int32_t tt = XB_GRP_TT;
    VkSpecializationMapEntry me = {0, 0, sizeof(int32_t)};
    VkSpecializationInfo si = {1, &me, sizeof tt, &tt};
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT rss = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO_EXT,
        .requiredSubgroupSize = 64};
    for (int v = 0; v < 2; v++) {
        VkComputePipelineCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .pNext = &rss,
                      .flags = VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT_EXT,
                      .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = X->sh_grp[v], .pName = "main",
                      .pSpecializationInfo = &si}, .layout = X->pl_grp};
        if (vkCreateComputePipelines(X->dev, VK_NULL_HANDLE, 1, &ci, NULL, &X->p_grp[v]) != VK_SUCCESS) {
            X->p_grp[v] = VK_NULL_HANDLE; return;
        }
    }
    /* the grouped GEMV for the small batches (COLI_VK_XB_GEMV=0: per-expert GEMVs) */
    const char *ge = getenv("COLI_VK_XB_GEMV");
    const char *gnm[2] = {"qmatmul_grp_gemv.spv", "qmatmul_grp_gemv_gate_up.spv"};
    for (int v = 0; v < 2 && !(ge && *ge == '0'); v++) {
        char path[1100];
        derive_dir_file(X->spv, gnm[v], path, sizeof path);
        if (!(X->sh_gv[v] = load_spv(X->dev, path))) { X->p_gv[0] = VK_NULL_HANDLE; break; }
        VkComputePipelineCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .pNext = &rss,
                      .flags = VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT_EXT,
                      .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = X->sh_gv[v], .pName = "main",
                      .pSpecializationInfo = &si}, .layout = X->pl_grp};
        if (vkCreateComputePipelines(X->dev, VK_NULL_HANDLE, 1, &ci, NULL, &X->p_gv[v]) != VK_SUCCESS) {
            X->p_gv[v] = VK_NULL_HANDLE; X->p_gv[0] = VK_NULL_HANDLE; break;
        }
    }
    if (!X->p_gv[0] || !X->p_gv[1]) X->p_gv[0] = X->p_gv[1] = VK_NULL_HANDLE;
    VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 30};
    VkDescriptorPoolCreateInfo dp = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 6, .poolSizeCount = 1, .pPoolSizes = &ps};
    if (vkCreateDescriptorPool(X->dev, &dp, NULL, &X->grp_pool) != VK_SUCCESS) { X->grp_pool = VK_NULL_HANDLE; return; }
    VkDescriptorSetLayout l6[6] = {X->dsl_grp, X->dsl_grp, X->dsl_grp, X->dsl_grp, X->dsl_grp, X->dsl_grp};
    VkDescriptorSetAllocateInfo da = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = X->grp_pool, .descriptorSetCount = 6, .pSetLayouts = l6};
    if (vkAllocateDescriptorSets(X->dev, &da, &X->grp_set[0][0]) != VK_SUCCESS) return;
    const char *r = getenv("COLI_VK_XB_GROUPED_ROWS");
    X->grp_rows = r && *r ? atoi(r) : 64;
    if (X->grp_rows < 1) X->grp_rows = 1;
    /* the step's routed sum (expert_sum.spv); without it big steps keep their row copies */
    char sp[1100];
    derive_dir_file(X->spv, "expert_sum.spv", sp, sizeof sp);
    const char *se = getenv("COLI_VK_XB_STEP");
    if ((se && *se == '0') || !(X->st.sh = load_spv(X->dev, sp))) return;
    if (!xb_layout(X, 4, 0u, 3 * sizeof(int), &X->st.dsl, &X->st.pl) ||
        !xb_pipeline(X, X->st.sh, X->st.pl, NULL, &X->st.pipe)) { X->st.pipe = VK_NULL_HANDLE; return; }
    VkDescriptorPoolSize ps2 = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4};
    VkDescriptorPoolCreateInfo dp2 = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &ps2};
    VkDescriptorSetAllocateInfo da2 = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorSetCount = 1, .pSetLayouts = &X->st.dsl};
    if (vkCreateDescriptorPool(X->dev, &dp2, NULL, &X->st.pool) != VK_SUCCESS) { X->st.pool = VK_NULL_HANDLE; X->st.pipe = VK_NULL_HANDLE; return; }
    da2.descriptorPool = X->st.pool;
    if (vkAllocateDescriptorSets(X->dev, &da2, &X->st.set) != VK_SUCCESS) X->st.pipe = VK_NULL_HANDLE;
}

static int xb_init(XbCtx *X, int d, int D, int I, int act, float limit, float a, float b) {
    if (X->ready) {   /* the activation rides the push constants; the geometry changes
                       * only while no expert is allocated (scratch grows by bytes) */
        if (X->inflight || X->sub[0].inflight || X->sub[1].inflight || ((X->D != D || X->I != I) && X->live)) return 0;
        X->D = D; X->I = I;
        if (act == COLI_VK_ACT_SWIGLU_V4 && !xb_v4_init(X)) return 0;
        X->act = act; X->limit = limit > 0.f ? limit : 0.f; X->a = a; X->b = b;
        return 1;
    }
    if (D < 1 || I < 1 || !xb_bind(X, d)) return 0;
    X->D = D; X->I = I; X->act = act; X->limit = limit > 0.f ? limit : 0.f; X->a = a; X->b = b;
    X->align = X->ssbo_align > 16 ? X->ssbo_align : 16;
    if (!xb_layout(X, 6, (1u << 0) | (1u << 5), sizeof(struct PCGU), &X->dsl6, &X->pl6) ||
        !xb_layout(X, 4, (1u << 0) | (1u << 3), sizeof(struct PC), &X->dsl4, &X->pl4) ||
        !xb_layout(X, 3, 7u, sizeof(struct PCAct), &X->dsl3, &X->pl3)) return 0;
    /* Own modules, loaded from G's files: MoltenVK with Metal argument buffers runs a
     * pipeline built from a module that a static-layout pipeline already used without
     * that pipeline's dynamic offsets. Own modules keep the batch's windows apart. */
    char gu_path[1100], mm_path[1100];
    derive_sibling(X->spv, "_gate_up.spv", gu_path, sizeof gu_path);
    derive_sibling(X->spv, "_gemm.spv", mm_path, sizeof mm_path);
    X->sh_gu = load_spv(X->dev, gu_path);
    X->sh_mv = load_spv(X->dev, X->spv);
    X->sh_mm = X->gemm_shader ? load_spv(X->dev, mm_path) : VK_NULL_HANDLE;
    if (!X->sh_gu || !X->sh_mv || (!X->d && X->gemm_shader && !X->sh_mm)) return 0;   /* the second device: GEMVs only */
    if (!xb_pipeline(X, X->sh_gu, X->pl6, NULL, &X->p_gu) || !xb_pipeline(X, X->sh_mv, X->pl4, NULL, &X->p_mv)) return 0;
    /* the GEMM route: the backend's tiles, and expert_act.spv beside the main shader */
    char act_path[1100];
    derive_dir_file(X->spv, "expert_act.spv", act_path, sizeof act_path);
    X->sh_act = X->sh_mm && X->gemm_has[0] ? load_spv(X->dev, act_path) : VK_NULL_HANDLE;
    if (X->sh_act && xb_pipeline(X, X->sh_act, X->pl3, NULL, &X->p_act)) {
        for (int k = 0; k < VK_GEMM_SLOTS; k++) {
            if (!X->gemm_has[k]) continue;
            VkGemmTile t = X->gemm_t[k];
            int32_t sv[7] = {t.bm, t.bn, t.bk, t.tm, t.tn, (t.bm / t.tm) * (t.bn / t.tn), t.pf ? 1 : 0};
            VkSpecializationMapEntry me[7];
            for (int i = 0; i < 7; i++) me[i] = (VkSpecializationMapEntry){(uint32_t)i, (uint32_t)(i * 4), 4};
            VkSpecializationInfo si = {7, me, sizeof(sv), sv};
            if (!xb_pipeline(X, X->sh_mm, X->pl4, &si, &X->p_mm[k])) X->p_mm[k] = VK_NULL_HANDLE;
        }
    }
    const char *e = getenv("COLI_VK_TIER_GEMM_ROWS");
    X->gemm_rows = X->p_mm[0] && X->p_act ? (e && *e ? atoi(e) : 16) : 0;
    if (X->gemm_rows < 0) X->gemm_rows = 0;
    xb_coop_init(X);
    xb_grp_init(X);
    VkCommandPoolCreateInfo cp = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = X->q_fam};
    VKCHECK(vkCreateCommandPool(X->dev, &cp, NULL, &X->cpool), "xb cmd pool");
    VkCommandBufferAllocateInfo cb = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = X->cpool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
    VKCHECK(vkAllocateCommandBuffers(X->dev, &cb, &X->cmd), "xb cmd buffer");
    VkFenceCreateInfo fi = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VKCHECK(vkCreateFence(X->dev, &fi, NULL, &X->fence), "xb fence");
    if (X->ts_bits && X->ts_period > 0.f) {
        VkQueryPoolCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
            .queryType = VK_QUERY_TYPE_TIMESTAMP, .queryCount = 2};
        X->has_ts = vkCreateQueryPool(X->dev, &qi, NULL, &X->qpool) == VK_SUCCESS;
    }
    VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 3};
    VkDescriptorPoolCreateInfo dp = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &ps};
    VKCHECK(vkCreateDescriptorPool(X->dev, &dp, NULL, &X->act_pool), "xb act pool");
    VkDescriptorSetAllocateInfo da = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = X->act_pool, .descriptorSetCount = 1, .pSetLayouts = &X->dsl3};
    VKCHECK(vkAllocateDescriptorSets(X->dev, &da, &X->act_set), "xb act set");
    /* a first scratch for 64 rows of 16 experts; grows with the first prefill */
    if (!xb_reserve(X, 64 * (size_t)D * 4 + 16 * X->align, 64 * (size_t)I * 4 + 16 * X->align,
                    64 * (size_t)D * 4 + 16 * X->align)) return 0;
    if (act == COLI_VK_ACT_SWIGLU_V4 && !xb_v4_init(X)) return 0;
    X->ready = 1;
    return 1;
}
int coli_vk_xb_init(int D, int I, int act, float limit, float a, float b) {
    return xb_init(&g_xb[0], 0, D, I, act, limit, a, b);
}
int coli_vk_xb_init_dev(int dev, int D, int I, int act, float limit, float a, float b) {
    return dev == 0 || dev == 1 ? xb_init(&g_xb[dev], dev, D, I, act, limit, a, b) : 0;
}
/* a device lost elsewhere stops the tier too */
int coli_vk_xb_ready(void) { return g_xb[0].ready && G.ready; }
int coli_vk_xb_ready_dev(int dev) { return dev == 0 || dev == 1 ? g_xb[dev].ready && xb_dev_ready(&g_xb[dev]) : 0; }
int coli_vk_xb_queue_shared(void) { return G.tq_shared; }

static int xb_busy(XbCtx *X);
ColiVkExpert *coli_vk_xb_expert(ColiVkTensor *g, ColiVkTensor *u, ColiVkTensor *d) {
    if (!g || !u || !d || u->dev != g->dev || d->dev != g->dev) return NULL;
    XbCtx *X = &g_xb[g->dev ? 1 : 0];   /* the batch of the device the tensors live on */
    if (!X->ready || xb_busy(X)) return NULL;
    if (g->I != X->D || g->O != X->I || u->I != X->D || u->O != X->I || g->fmt != u->fmt || g->gs != u->gs ||
        d->I != X->I || d->O != X->D) return NULL;
    ColiVkExpert *e = calloc(1, sizeof(*e));
    if (!e) return NULL;
    VkDescriptorSetLayout ls[4] = {X->dsl6, X->dsl4, X->dsl4, X->dsl4};
    VkDescriptorSet sets[4];
    int k = 0;
    for (;; k++) {
        if (k == X->ndpools) {
            VkDescriptorPoolSize ps[2] = {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 10 * XB_POOL_EXPERTS},
                                          {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 8 * XB_POOL_EXPERTS}};
            VkDescriptorPoolCreateInfo dp = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
                .maxSets = 4 * XB_POOL_EXPERTS, .poolSizeCount = 2, .pPoolSizes = ps};
            VkDescriptorPool *n = realloc(X->dpools, (size_t)(X->ndpools + 1) * sizeof(*n));
            if (!n) { free(e); return NULL; }
            X->dpools = n;
            if (vkCreateDescriptorPool(X->dev, &dp, NULL, &X->dpools[X->ndpools]) != VK_SUCCESS) { free(e); return NULL; }
            X->ndpools++;
        }
        VkDescriptorSetAllocateInfo da = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = X->dpools[k], .descriptorSetCount = 4, .pSetLayouts = ls};
        if (vkAllocateDescriptorSets(X->dev, &da, sets) == VK_SUCCESS) break;
    }
    e->g = g; e->u = u; e->d = d; e->dpool = k; e->X = X;
    e->s_gu = sets[0]; e->s_dn = sets[1]; e->s_g = sets[2]; e->s_u = sets[3];
    xb_write_sets(e);
    e->next = X->live; if (X->live) X->live->prev = e; X->live = e;
    return e;
}

/* The expert's sets and its three tensors. Not while a batch is in flight (the
 * batch may name it); the tier frees at its join. */
static void xb_expert_unlink(ColiVkExpert *e) {   /* its sets, not its tensors */
    XbCtx *X = e->X;
    if (e->prev) e->prev->next = e->next; else X->live = e->next;
    if (e->next) e->next->prev = e->prev;
    if (xb_dev_ready(X) && e->dpool < X->ndpools) {
        VkDescriptorSet sets[4] = {e->s_gu, e->s_dn, e->s_g, e->s_u};
        vkFreeDescriptorSets(X->dev, X->dpools[e->dpool], 4, sets);
    }
}
void coli_vk_xb_expert_free(ColiVkExpert *e) {
    if (!e) return;
    xb_expert_unlink(e);
    coli_vk_tensor_free(e->g); coli_vk_tensor_free(e->u); coli_vk_tensor_free(e->d);
    free(e);
}

static void xb_barrier(VkCommandBuffer c) {
    VkMemoryBarrier mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_SHADER_READ_BIT};
    vkCmdPipelineBarrier(c, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &mb, 0, NULL, 0, NULL);
}

/* One batch's layout in the scratch: each expert's x, hidden and y offsets from the bases
 * x0, i0, y0 (its x rows, then with the V4 activation its row weights), the ends and the
 * row total. 0 when an expert is missing or a row count is out of range. */
static int xb_plan(XbCtx *X, ColiVkExpert *const *ex, const int *rows, int count, int v4, size_t x0, size_t i0, size_t y0,
                   size_t *off, size_t *xe, size_t *ie, size_t *ye, int *total) {
    const size_t a = X->align, dr = (size_t)X->D * 4, ir = (size_t)X->I * 4;
    size_t xo = x0, io = i0, yo = y0; int t = 0;
    for (int c = 0; c < count; c++) {
        if (!ex[c] || rows[c] < 1 || rows[c] > 65535) return 0;
        off[3 * c] = xo; off[3 * c + 1] = io; off[3 * c + 2] = yo;
        xo += xb_up((size_t)rows[c] * dr, a); io += xb_up((size_t)rows[c] * ir, a); yo += xb_up((size_t)rows[c] * dr, a);
        if (v4) xo += xb_up((size_t)rows[c] * 4, a);
        t += rows[c];
    }
    *xe = xo; *ie = io; *ye = yo; *total = t;
    return 1;
}
static uint64_t vk_addr(VkBuffer b) {
    VkBufferDeviceAddressInfo ai = {.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .buffer = b};
    return (uint64_t)vkGetBufferDeviceAddress(G.dev, &ai);
}
/* The grouped route of a batch (X->p_grp) into cmd: gate+up+SwiGLU of every expert in
 * one dispatch, a barrier, down of every expert in another. The packed rows stay where
 * xb_plan put them: with row sizes on the buffers' alignment an expert's x, hidden and y
 * rows start at the same row index past the batch's bases x0, i0, y0 (its scratch half),
 * which the sets bind as their offsets. `slot`: whose tables and sets (0 the single
 * batch, 1 + h a sub-batch half). 0 = not taken, nothing recorded. */
/* Whether the grouped route takes this batch (and its row total). */
/* Whether the grouped route takes this batch (and its row total): 1 the grouped GEMM
 * (grp_rows rows or more), 2 the grouped GEMV (fewer), 0 neither. */
static int xb_grp_ok(const XbCtx *X, ColiVkExpert *const *ex, const int *rows, int count, int *total_out) {
    const int D = X->D, I = X->I;
    const size_t dr = (size_t)D * 4, ir = (size_t)I * 4;
    int total = 0;
    for (int c = 0; c < count; c++) total += rows[c];
    if (total_out) *total_out = total;
    if (!X->grp_rows || X->act != COLI_VK_ACT_SWIGLU || dr % X->align || ir % X->align) return 0;
    int mode = total >= X->grp_rows ? 1 : X->p_gv[0] ? 2 : 0;
    if (mode == 1 && (D % 128 || I % 128)) return 0;
    if (mode == 2 && (D % 32 || I % 32 || D > 8192 || I > 8192)) return 0;
    if (!mode) return 0;
    for (int c = 0; c < count; c++) {
        const ColiVkTensor *t[3] = {ex[c]->g, ex[c]->u, ex[c]->d};
        for (int k = 0; k < 3; k++) {
            int f = t[k]->fmt;
            if (t[k]->dev || !t[k]->pool || !(f == 1 || f == 2 || f == 4) || t[k]->rowWords % 4 ||
                (f == 4 && (t[k]->gs < 8 || t[k]->gs % (mode == 2 ? 32 : 8)))) return 0;
        }
        if (ex[c]->u->fmt != ex[c]->g->fmt || ex[c]->u->gs != ex[c]->g->gs) return 0;
    }
    return mode;
}
/* amap (a big step's sub-batch, X->st.on): packed row j is assignment amap[j]; x comes from
 * the step's token rows and y goes to the step's output rows (the scratch's x and y unused). */
static int xb_grp_record(XbCtx *X, VkCommandBuffer cmd, int slot, ColiVkExpert *const *ex, const int *rows,
                         int count, const size_t *off, size_t x0, size_t i0, size_t y0, const int *amap) {
    const int D = X->D, I = X->I;
    const size_t dr = (size_t)D * 4, ir = (size_t)I * 4;
    int total = 0, mode = xb_grp_ok(X, ex, rows, count, &total);
    if (!mode || (mode == 2 && amap)) return 0;
    if (amap) {
        XbBuf *mb = &X->grp_map[slot];
        size_t nb = (size_t)total * 4;
        if (nb > mb->region && !xb_buf(X, mb, xb_up(nb + nb / 2, 4096), X->mt_host, 1)) { X->grp_rows = 0; return 0; }
        memcpy(mb->ptr, amap, nb);
    }
    /* items: (expert, first row, 128-row output block), 64 rows a workgroup; the tiles
     * of one (expert, block) adjacent, so workgroups in flight share weights */
    /* the GEMV's items: (expert, row, XB_GV_RPW outputs) */
    const int tm = mode == 1 ? 16 * XB_GRP_TT : 1, bo = mode == 1 ? 128 : XB_GV_RPW;
    int nit[2] = {0, 0};
    for (int c = 0; c < count; c++) {
        int nt = (rows[c] + tm - 1) / tm;
        nit[0] += nt * ((D + bo - 1) / bo); nit[1] += nt * ((I + bo - 1) / bo);
    }
    for (int v = 0; v < 2; v++) {
        XbBuf *ib = &X->grp_it[slot][v], *eb = &X->grp_et[slot][v];
        size_t nb = (size_t)nit[v] * 16, ne = (size_t)count * 64;
        if ((nb > ib->region && !xb_buf(X, ib, xb_up(nb + nb / 2, 4096), X->mt_host, 1)) ||
            (ne > eb->region && !xb_buf(X, eb, xb_up(ne + ne / 2, 4096), X->mt_host, 1))) {
            X->grp_rows = 0;   /* no room for the tables: the per-expert route from here on */
            return 0;
        }
    }
    for (int v = 0; v < 2; v++) {   /* v 1: gate+up (out I), v 0: down (out D) */
        uint32_t et[16], *it = X->grp_it[slot][v].ptr;
        int n = 0, O = v ? I : D;
        for (int c = 0; c < count; c++) {
            const ColiVkTensor *a = v ? ex[c]->g : ex[c]->d, *b = v ? ex[c]->u : NULL;
            uint64_t ad[4] = {vk_addr(a->wbuf), vk_addr(a->sbuf), b ? vk_addr(b->wbuf) : 0, b ? vk_addr(b->sbuf) : 0};
            memset(et, 0, sizeof et);
            for (int k = 0; k < 4; k++) { et[2 * k] = (uint32_t)ad[k]; et[2 * k + 1] = (uint32_t)(ad[k] >> 32); }
            et[8] = (uint32_t)rows[c]; et[9] = (uint32_t)((off[3 * c] - x0) / dr);
            et[10] = (uint32_t)a->fmt; et[11] = (uint32_t)a->rowWords; et[12] = (uint32_t)a->gs;
            memcpy((uint8_t *)X->grp_et[slot][v].ptr + (size_t)c * 64, et, sizeof et);   /* written, never read back */
            for (int ob = 0; ob < (O + bo - 1) / bo; ob++)
                for (int t0 = 0; t0 < rows[c]; t0 += tm) {
                    /* the GEMM takes a block index, the GEMV its first output */
                    uint32_t q[4] = {(uint32_t)c, (uint32_t)t0, (uint32_t)(mode == 1 ? ob : ob * bo), 1};
                    memcpy(it + (size_t)n * 4, q, sizeof q); n++;
                }
        }
    }
    /* windows from the bases: x rows, hidden rows, y rows of this batch */
    size_t total_d = (size_t)total * dr, total_i = (size_t)total * ir;
    VkDescriptorBufferInfo in[2] = {{X->h.buf, i0, total_i}, {X->x.buf, x0, total_d}};
    VkDescriptorBufferInfo out[2] = {{X->y.buf, y0, total_d}, {X->h.buf, i0, total_i}};
    VkDescriptorBufferInfo mp = {X->grp_it[slot][0].buf, 0, 16};   /* unread without amap */
    if (amap) {
        in[1] = (VkDescriptorBufferInfo){X->st.xt.buf, 0, (size_t)X->st.S * dr};
        out[0] = (VkDescriptorBufferInfo){X->st.ys.buf, 0, (size_t)X->st.S * X->st.K * dr};
        mp = (VkDescriptorBufferInfo){X->grp_map[slot].buf, 0, (size_t)total * 4};
    }
    for (int step = 0; step < 2; step++) {
        int v = 1 - step;   /* gate+up first, then down */
        VkDescriptorBufferInfo bi[5] = {in[v], {X->grp_it[slot][v].buf, 0, X->grp_it[slot][v].region},
                                        {X->grp_et[slot][v].buf, 0, X->grp_et[slot][v].region}, out[v], mp};
        VkWriteDescriptorSet w[5];
        for (int b = 0; b < 5; b++)
            w[b] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = X->grp_set[slot][v],
                .dstBinding = (uint32_t)b, .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &bi[b]};
        vkUpdateDescriptorSets(X->dev, 5, w, 0, NULL);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, mode == 1 ? X->p_grp[v] : X->p_gv[v]);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, X->pl_grp, 0, 1, &X->grp_set[slot][v], 0, NULL);
        for (int i0n = 0; i0n < nit[v]; i0n += 65535) {   /* the grid's x limit every device has */
            struct PCGRP pc = {v ? D : I, v ? I : D, X->limit, i0n, amap ? X->st.K : 0, XB_GV_RPW};
            vkCmdPushConstants(cmd, X->pl_grp, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, &pc);
            vkCmdDispatch(cmd, (uint32_t)(nit[v] - i0n < 65535 ? nit[v] - i0n : 65535), 1, 1);
        }
        if (!step) xb_barrier(cmd);
    }
    if (mode == 1) X->grp_batches++; else X->gv_batches++;
    return 1;
}
/* Record one batch laid out at off[] (its x rows already written) into cmd, with its
 * timestamps in qp when the device has them, and submit it with fence on the tier queue.
 * *ngemm: how many experts took the GEMM route. 0 = nothing was submitted. */
static int xb_record_submit(XbCtx *X, VkCommandBuffer cmd, VkFence fence, VkQueryPool qp, ColiVkExpert *const *ex,
                            const int *rows, int count, const size_t *off, int v4, int *ngemm_out,
                            int slot, size_t x0, size_t i0, size_t y0, const int *amap) {
    const int D = X->D, I = X->I;
    const size_t a = X->align, dr = (size_t)D * 4;
    if (vkResetCommandBuffer(cmd, 0) != VK_SUCCESS) return 0;
    VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    vkBeginCommandBuffer(cmd, &bi);
    if (X->has_ts) {
        vkCmdResetQueryPool(cmd, qp, 0, 2);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qp, 0);
    }
    /* the route of each expert: its GEMM slot, or -1 for the GEMVs */
    int *route = malloc((size_t)count * sizeof(int));
    if (!route) { vkEndCommandBuffer(cmd); return 0; }
    int ngemm = 0, cooperative = 0, grouped = !v4 && xb_grp_record(X, cmd, slot, ex, rows, count, off, x0, i0, y0, amap);
    if (grouped) ngemm = count;
    else {
        for (int c = 0; c < count; c++) {
            int k = X->gemm_rows && rows[c] >= X->gemm_rows ? xb_gemm_slot(X, rows[c]) : -1;
            route[c] = k >= 0 && X->p_mm[k] ? k : -1;
            ngemm += route[c] >= 0;
        }
        /* phase 1: gate and up of every expert (fused GEMV, or two GEMMs) */
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, X->p_gu);
        for (int c = 0; c < count; c++) {
            if (route[c] >= 0) continue;
            const ColiVkExpert *e = ex[c];
            uint32_t dyn[2] = {(uint32_t)off[3 * c], (uint32_t)off[3 * c + 1]};
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, X->pl6, 0, 1, &e->s_gu, 2, dyn);
            struct PCGU pc = {e->g->fmt, rows[c], D, I, e->g->rowWords, e->g->gs, X->limit, X->act, X->a, X->b};
            vkCmdPushConstants(cmd, X->pl6, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            vkCmdDispatch(cmd, (uint32_t)((I + 7) / 8), (uint32_t)rows[c], 1);
        }
        if (ngemm) {
            VkPipeline bound = VK_NULL_HANDLE;
            for (int c = 0; c < count; c++) {
                if (route[c] < 0) continue;
                const ColiVkExpert *e = ex[c];
                int k = route[c];
                const ColiVkTensor *gu[2] = {e->g, e->u};
                VkDescriptorSet su[2] = {e->s_g, e->s_u};
                for (int q = 0; q < 2; q++) {
                    int bm, bn;
                    VkPipeline pipe = xb_mm_pipeline(X, gu[q], rows[c], k, &bm, &bn, &cooperative);
                    if (pipe != bound) { vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe); bound = pipe; }
                    uint32_t dyn[2] = {(uint32_t)off[3 * c], (uint32_t)off[3 * c + 1]};
                    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, X->pl4, 0, 1, &su[q], 2, dyn);
                    struct PC pc = {gu[q]->fmt, rows[c], D, I, gu[q]->rowWords, gu[q]->gs};
                    vkCmdPushConstants(cmd, X->pl4, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
                    vkCmdDispatch(cmd, (uint32_t)((I + bm - 1) / bm), (uint32_t)((rows[c] + bn - 1) / bn), 1);
                }
            }
            xb_barrier(cmd);
            /* phase 2: the activation of the GEMM experts */
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, X->p_act);
            for (int c = 0; c < count; c++) {
                if (route[c] < 0) continue;
                uint32_t dyn[3] = {(uint32_t)off[3 * c + 1], (uint32_t)off[3 * c + 1], (uint32_t)off[3 * c + 1]};
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, X->pl3, 0, 1, &X->act_set, 3, dyn);
                int n = rows[c] * I;
                struct PCAct pc = {n, X->act, X->limit, X->a, X->b};
                vkCmdPushConstants(cmd, X->pl3, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
                uint32_t groups = (uint32_t)((n + 255) / 256), gx = groups < 65535 ? groups : 65535;
                vkCmdDispatch(cmd, gx, (groups + gx - 1) / gx, 1);
            }
        }
        xb_barrier(cmd);
        if (v4) {   /* DeepSeek V4: route weight, bf16 and E4M3 per 128 on the hidden rows */
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, X->p_v4);
            for (int c = 0; c < count; c++) {
                uint32_t dyn[2] = {(uint32_t)off[3 * c + 1], (uint32_t)(off[3 * c] + xb_up((size_t)rows[c] * dr, a))};
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, X->pl_v4, 0, 1, &X->v4_set, 2, dyn);
                struct PCV4 pc = {rows[c], I};
                vkCmdPushConstants(cmd, X->pl_v4, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
                vkCmdDispatch(cmd, (uint32_t)((I + 127) / 128), (uint32_t)rows[c], 1);
            }
            xb_barrier(cmd);
        }
        /* phase 3: down of every expert */
        VkPipeline bound = VK_NULL_HANDLE;
        for (int c = 0; c < count; c++) {
            const ColiVkExpert *e = ex[c];
            int k = route[c], gemm = k >= 0;
            int bm = 8, bn = 1;
            VkPipeline pipe = gemm ? xb_mm_pipeline(X, e->d, rows[c], k, &bm, &bn, &cooperative) : X->p_mv;
            if (pipe != bound) { vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe); bound = pipe; }
            uint32_t dyn[2] = {(uint32_t)off[3 * c + 1], (uint32_t)off[3 * c + 2]};
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, X->pl4, 0, 1, &e->s_dn, 2, dyn);
            struct PC pc = {e->d->fmt, rows[c], I, D, e->d->rowWords, e->d->gs};
            vkCmdPushConstants(cmd, X->pl4, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            if (gemm) {
                vkCmdDispatch(cmd, (uint32_t)((D + bm - 1) / bm), (uint32_t)((rows[c] + bn - 1) / bn), 1);
            } else vkCmdDispatch(cmd, (uint32_t)((D + 7) / 8), (uint32_t)rows[c], 1);
        }
    }
    /* the host reads y after the fence */
    VkMemoryBarrier hb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_HOST_READ_BIT};
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &hb, 0, NULL, 0, NULL);
    if (X->has_ts) vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qp, 1);
    free(route);
    if (vkEndCommandBuffer(cmd) != VK_SUCCESS) return 0;
    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cmd};
    if (vkResetFences(X->dev, 1, &fence) != VK_SUCCESS) return 0;
    if (vk_submit(X->d, X->q, &si, fence) != VK_SUCCESS) {
        fprintf(stderr, "[VK] %sexpert batch: submit failed, %s\n", X->d ? "dev2 " : "",
                X->d ? "the second device stops" : "the tier stops");
        X->ready = 0; return 0;
    }
    *ngemm_out = ngemm;
    X->cooperative_matmuls += (unsigned long long)cooperative;
    return 1;
}
/* the single batch or a sub-batch in flight: the scratch and the experts' sets are in use */
static int xb_busy(XbCtx *X) { return X->inflight || X->sub[0].inflight || X->sub[1].inflight; }

static int xb_issue(XbCtx *X, ColiVkExpert *const *ex, const int *rows, int count, const float *const *xrows,
                    const float *wrows) {
    if (!X->ready || xb_busy(X) || count < 1) return 0;
    const size_t a = X->align, dr = (size_t)X->D * 4;
    /* COLI_VK_ACT_SWIGLU_V4: each expert's row weights follow its x rows */
    const int v4 = X->act == COLI_VK_ACT_SWIGLU_V4 && X->p_v4;
    size_t *off = malloc((size_t)count * 3 * sizeof(size_t));
    if (!off) return 0;
    size_t xo, io, yo; int total;
    if (!xb_plan(X, ex, rows, count, v4, 0, 0, 0, off, &xo, &io, &yo, &total)) { free(off); return 0; }
    if (!xb_reserve(X, xo, io, yo)) { free(off); return 0; }
    if (total > X->cyoff) {
        size_t *n = realloc(X->yoff, (size_t)total * sizeof(*n));
        if (!n) { free(off); return 0; }
        X->yoff = n; X->cyoff = total;
    }
    /* the activation rows, packed per expert; where each output row will be (the copies
     * in parallel for a prompt step, as coli_vk_xb_sub_issue's) */
    int *first = malloc((size_t)count * sizeof(int));
    if (!first) { free(off); return 0; }
    for (int c = 0, j = 0; c < count; j += rows[c], c++) first[c] = j;
    #pragma omp parallel for schedule(dynamic, 4) if (total >= 256)
    for (int c = 0; c < count; c++)
        for (int r = 0; r < rows[c]; r++) {
            int j = first[c] + r;
            memcpy((uint8_t *)X->x.ptr + off[3 * c] + (size_t)r * dr, xrows[j], dr);
            X->yoff[j] = off[3 * c + 2] + (size_t)r * dr;
            if (v4) ((float *)((uint8_t *)X->x.ptr + off[3 * c] + xb_up((size_t)rows[c] * dr, a)))[r] = wrows ? wrows[j] : 1.0f;
        }
    free(first);
    X->nrows = total;
    int ngemm = 0;
    int ok = xb_record_submit(X, X->cmd, X->fence, X->qpool, ex, rows, count, off, v4, &ngemm, 0, 0, 0, 0, NULL);
    free(off);
    if (!ok) return 0;
    X->inflight = 1; async_begin(X->d);
    X->batches++; X->experts += (unsigned long long)count; X->rows += (unsigned long long)total;
    X->gemm_experts += (unsigned long long)ngemm;
    return 1;
}
int coli_vk_xb_issue(ColiVkExpert *const *ex, const int *rows, int count, const float *const *xrows) {
    return xb_issue(&g_xb[0], ex, rows, count, xrows, NULL);
}
int coli_vk_xb_issue_w(ColiVkExpert *const *ex, const int *rows, int count, const float *const *xrows,
                       const float *wrows) {
    return xb_issue(&g_xb[0], ex, rows, count, xrows, wrows);
}
int coli_vk_xb_issue_dev(int dev, ColiVkExpert *const *ex, const int *rows, int count, const float *const *xrows,
                         const float *wrows) {
    if (dev != 0 && dev != 1) return 0;
    for (int c = 0; c < count; c++) if (!ex[c] || ex[c]->X != &g_xb[dev]) return 0;   /* every expert on this device */
    return xb_issue(&g_xb[dev], ex, rows, count, xrows, wrows);
}

/* COLI_VK_DEV2_FAULT=n (tests): the n-th batch joined on the second device fails there,
 * as a lost device's does. One stderr line says when it fired. */
static int dev2_fault(void) {
    static long at = -1, n;
    if (at < 0) { const char *e = getenv("COLI_VK_DEV2_FAULT"); at = e && *e ? atol(e) : 0; }
    if (at <= 0 || ++n != at) return 0;
    fprintf(stderr, "[VK] COLI_VK_DEV2_FAULT: batch #%ld on the second device fails\n", at);
    return 1;
}
static int xb_join(XbCtx *X, const float **yrows, double *device_ms) {
    if (!X->inflight) return 0;
    X->inflight = 0;
    VkResult r = vk_fence_wait(X->dev, X->fence);
    if (r == VK_SUCCESS && X->d && dev2_fault()) r = VK_ERROR_DEVICE_LOST;
    async_end(X->d);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "[VK] %sexpert batch: fence wait failed (%d), %s\n", X->d ? "dev2 " : "", r,
                X->d ? "the second device stops" : "the tier stops");
        X->ready = 0;
        if (X->d) __atomic_store_n(&G2.ready, 0, __ATOMIC_RELEASE);   /* lost: nothing else runs there */
        return 0;
    }
    double ms = 0;
    if (X->has_ts) {
        uint64_t ts[2];
        if (vkGetQueryPoolResults(X->dev, X->qpool, 0, 2, sizeof ts, ts, sizeof ts[0], VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
            uint64_t mask = X->ts_bits >= 64 ? ~0ull : ((1ull << X->ts_bits) - 1);
            ms = (double)((ts[1] - ts[0]) & mask) * X->ts_period / 1e6;
        }
    }
    X->dev_ms += ms;
    if (device_ms) *device_ms = ms;
    for (int j = 0; j < X->nrows; j++) yrows[j] = (const float *)((const uint8_t *)X->y.ptr + X->yoff[j]);
    return 1;
}
int coli_vk_xb_join(const float **yrows, double *device_ms) { return xb_join(&g_xb[0], yrows, device_ms); }
int coli_vk_xb_join_dev(int dev, const float **yrows, double *device_ms) {
    return dev == 0 || dev == 1 ? xb_join(&g_xb[dev], yrows, device_ms) : 0;
}
int coli_vk_xb_busy_dev(int dev) { return (dev == 0 || dev == 1) && g_xb[dev].inflight; }

static void xb_stats(const XbCtx *X, ColiVkXbStats *st) {
    memset(st, 0, sizeof(*st));
    st->batches = X->batches; st->experts = X->experts; st->rows = X->rows; st->gemm_experts = X->gemm_experts;
    st->device_ms = X->dev_ms; st->timestamps = X->has_ts; st->queue_shared = X->q_shared;
    st->gemm_rows = X->gemm_rows;
    st->scratch_bytes = 2 * (X->x.region + 3 * X->g.region + X->y.region);
    st->sub_batches = X->sub_batches; st->sub_experts = X->sub_experts; st->sub_rows = X->sub_rows;
    st->sub_gemm = X->sub_gemm; st->sub_ms = X->sub_ms;
    st->cooperative_matmuls = X->cooperative_matmuls; st->grouped_batches = X->grp_batches; st->gemv_batches = X->gv_batches;
}
void coli_vk_xb_stats(ColiVkXbStats *st) { xb_stats(&g_xb[0], st); }
void coli_vk_xb_stats_dev(int dev, ColiVkXbStats *st) {
    if (dev == 0 || dev == 1) xb_stats(&g_xb[dev], st); else memset(st, 0, sizeof(*st));
}

/* ---- sub-batches: a big prefill step as several batches, two in flight ----------------
 * The single batch above holds a whole step in the scratch at once. A big prefill step
 * (vk_tier.c's streaming: thousands of rows, every expert of the layer) instead runs as
 * a sequence of bounded batches: batch k uses half k % 2 of each scratch, so the host
 * writes the rows of the next while the device computes the one before, and the scratch
 * stays the size of two halves whatever the step's size (a card without Resizable BAR
 * keeps its x rows inside the host-visible window). Each half has its own command buffer,
 * fence and timestamps on the tier queue; its outputs are copied out when it is joined.
 * The recording is the single batch's (xb_record_submit). */
static int xb_sub_make(XbCtx *X, int h) {
    if (X->sub[h].cmd) return 1;
    VkCommandBufferAllocateInfo cb = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = X->cpool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
    VkFenceCreateInfo fi = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    if (vkAllocateCommandBuffers(X->dev, &cb, &X->sub[h].cmd) != VK_SUCCESS) { X->sub[h].cmd = VK_NULL_HANDLE; return 0; }
    if (vkCreateFence(X->dev, &fi, NULL, &X->sub[h].fence) != VK_SUCCESS) {
        vkFreeCommandBuffers(X->dev, X->cpool, 1, &X->sub[h].cmd); X->sub[h].cmd = VK_NULL_HANDLE; X->sub[h].fence = VK_NULL_HANDLE;
        return 0;
    }
    if (X->has_ts) {
        VkQueryPoolCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
            .queryType = VK_QUERY_TYPE_TIMESTAMP, .queryCount = 2};
        if (vkCreateQueryPool(X->dev, &qi, NULL, &X->sub[h].qp) != VK_SUCCESS) X->sub[h].qp = VK_NULL_HANDLE;
    }
    return 1;
}
/* Logical bytes of one half; xb_reserve adds its descriptor window and growth. */
static void xb_sub_sizes(XbCtx *X, int rows, int experts, size_t *hx, size_t *hi, size_t *hy) {
    const size_t a = X->align, dr = (size_t)X->D * 4, ir = (size_t)X->I * 4, pad = (size_t)experts * a;
    const int v4 = X->act == COLI_VK_ACT_SWIGLU_V4 && X->p_v4;
    *hx = xb_up((size_t)rows * dr + pad + (v4 ? (size_t)rows * 4 + pad : 0), a);
    *hi = xb_up((size_t)rows * ir + pad, a);
    *hy = xb_up((size_t)rows * dr + pad, a);
}
/* Choose before allocating: an OOM in xb_reserve disables the batch backend.
 * Account for both halves, the doubled descriptor windows, growth by 1.5 and
 * all five buffers. xb_buf frees an old buffer before replacing it, so only
 * positive growth must fit the remaining budget. Reused scratch costs zero. */
int coli_vk_xb_sub_fit(int rows, int experts, size_t extra_budget) {
    XbCtx *X = &g_xb[0];   /* sub-batches: the primary device only */
    if (!X->ready || xb_busy(X) || rows < 1 || experts < 1) return 0;
    if (rows > 65535) rows = 65535;
    VkPhysicalDeviceMemoryProperties2 mp = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
#ifdef VK_EXT_memory_budget
    VkPhysicalDeviceMemoryBudgetPropertiesEXT bud = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
    if (X->has_budget) mp.pNext = &bud;
#endif
    vkGetPhysicalDeviceMemoryProperties2(X->phys, &mp);
    size_t room[VK_MAX_MEMORY_HEAPS];
    for (uint32_t h = 0; h < mp.memoryProperties.memoryHeapCount; h++) {
        room[h] = (size_t)mp.memoryProperties.memoryHeaps[h].size / 8;
#ifdef VK_EXT_memory_budget
        if (X->has_budget) room[h] = bud.heapBudget[h] > bud.heapUsage[h] ? (size_t)((bud.heapBudget[h] - bud.heapUsage[h]) / 2) : 0;
#endif
        if (g_mem.cap[0] && (mp.memoryProperties.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) &&
            room[h] > vk_mem_room(0) / 2) room[h] = (size_t)(vk_mem_room(0) / 2);   /* COLI_VK_DEVICE_CAP_MB */
    }
    uint32_t xheap = mp.memoryProperties.memoryTypes[X->mt_host].heapIndex;
    uint32_t iheap = mp.memoryProperties.memoryTypes[X->mt_dev].heapIndex;
    uint32_t yheap = mp.memoryProperties.memoryTypes[X->mt_cached].heapIndex;
    int lo = 0, hi = rows;
    while (lo < hi) {
        int mid = lo + (hi - lo + 1) / 2;
        size_t hx, hh, hy, nx, ni, ny;
        xb_sub_sizes(X, mid, experts, &hx, &hh, &hy);
        int ok = xb_sizes(X, 2 * hx, 2 * hh, 2 * hy, &nx, &ni, &ny);
        if (ok) {
            size_t dx = 2 * (nx - X->x.region), di = 6 * (ni - X->g.region), dy = 2 * (ny - X->y.region);
            size_t need[VK_MAX_MEMORY_HEAPS] = {0};
            need[xheap] += dx; need[iheap] += di; need[yheap] += dy;
            ok = dx + di + dy <= extra_budget;
            for (uint32_t h = 0; ok && h < mp.memoryProperties.memoryHeapCount; h++) ok = need[h] <= room[h];
        }
        if (ok) lo = mid; else hi = mid - 1;
    }
    return lo;
}
int coli_vk_xb_sub_reserve(int rows, int experts) {
    XbCtx *X = &g_xb[0];   /* sub-batches: the primary device only */
    if (!X->ready || xb_busy(X) || rows < 1 || experts < 1) return 0;
    if (!xb_sub_make(X, 0) || !xb_sub_make(X, 1)) return 0;
    if (X->has_ts && (!X->sub[0].qp || !X->sub[1].qp)) return 0;
    size_t hx, hi, hy;
    xb_sub_sizes(X, rows, experts, &hx, &hi, &hy);
    if (!xb_reserve(X, 2 * hx, 2 * hi, 2 * hy)) return 0;
    X->sub_x = hx; X->sub_i = hi; X->sub_y = hy;
    return 1;
}
int coli_vk_xb_sub_busy(int h) { return h >= 0 && h < 2 && g_xb[0].sub[h].inflight; }
int coli_vk_xb_sub_issue(int h, ColiVkExpert *const *ex, const int *rows, int count, const float *const *xrows,
                         const float *wrows) {
    XbCtx *X = &g_xb[0];
    for (int c = 0; c < count; c++) if (!ex[c] || ex[c]->X != X) return 0;
    if (!X->ready || h < 0 || h > 1 || X->inflight || X->sub[h].inflight || !X->sub_x || count < 1) return 0;
    const size_t a = X->align, dr = (size_t)X->D * 4;
    const int v4 = X->act == COLI_VK_ACT_SWIGLU_V4 && X->p_v4;
    size_t *off = malloc((size_t)count * 3 * sizeof(size_t));
    if (!off) return 0;
    size_t x0 = h * X->sub_x, i0 = h * X->sub_i, y0 = h * X->sub_y, xe, ie, ye; int total;
    if (!xb_plan(X, ex, rows, count, v4, x0, i0, y0, off, &xe, &ie, &ye, &total) ||
        xe > x0 + X->sub_x || ie > i0 + X->sub_i || ye > y0 + X->sub_y) { free(off); return 0; }
    if (total > X->sub[h].cyoff) {
        size_t *n = realloc(X->sub[h].yoff, (size_t)total * sizeof(*n));
        if (!n) { free(off); return 0; }
        X->sub[h].yoff = n; X->sub[h].cyoff = total;
    }
    /* the activation rows, packed per expert (the copies in parallel: thousands of rows) */
    int *first = malloc((size_t)count * sizeof(int));
    if (!first) { free(off); return 0; }
    for (int c = 0, j = 0; c < count; j += rows[c], c++) first[c] = j;
    size_t *yoff = X->sub[h].yoff;
    uint8_t *xp = X->x.ptr;
    #pragma omp parallel for schedule(dynamic, 4) if (total >= 256)
    for (int c = 0; c < count; c++)
        for (int r = 0; r < rows[c]; r++) {
            int j = first[c] + r;
            memcpy(xp + off[3 * c] + (size_t)r * dr, xrows[j], dr);
            yoff[j] = off[3 * c + 2] + (size_t)r * dr;
            if (v4) ((float *)(xp + off[3 * c] + xb_up((size_t)rows[c] * dr, a)))[r] = wrows ? wrows[j] : 1.0f;
        }
    free(first);
    X->sub[h].nrows = total;
    int ngemm = 0;
    int ok = xb_record_submit(X, X->sub[h].cmd, X->sub[h].fence, X->sub[h].qp, ex, rows, count, off, v4, &ngemm, 1 + h, x0, i0, y0, NULL);
    free(off);
    if (!ok) return 0;
    X->sub[h].inflight = 1; async_begin(X->d);
    X->sub_batches++; X->sub_experts += (unsigned long long)count; X->sub_rows += (unsigned long long)total;
    X->sub_gemm += (unsigned long long)ngemm;
    return 1;
}
int coli_vk_xb_sub_join(int h, float *const *yout, double *device_ms) {
    XbCtx *X = &g_xb[0];   /* sub-batches: the primary device only */
    if (device_ms) *device_ms = 0;
    if (h < 0 || h > 1 || !X->sub[h].inflight) return 0;
    X->sub[h].inflight = 0;
    VkResult r = vk_fence_wait_gemm(X->dev, X->sub[h].fence);
    async_end(X->d);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "[VK] expert batch: fence wait failed (%d), the tier stops\n", r);
        X->ready = 0; return 0;
    }
    double ms = 0;
    if (X->has_ts && X->sub[h].qp) {
        uint64_t ts[2];
        if (vkGetQueryPoolResults(X->dev, X->sub[h].qp, 0, 2, sizeof ts, ts, sizeof ts[0], VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
            uint64_t mask = X->ts_bits >= 64 ? ~0ull : ((1ull << X->ts_bits) - 1);
            ms = (double)((ts[1] - ts[0]) & mask) * X->ts_period / 1e6;
        }
    }
    X->sub_ms += ms;
    if (device_ms) *device_ms = ms;
    const size_t dr = (size_t)X->D * 4;
    const size_t *yoff = X->sub[h].yoff;
    const uint8_t *yp = X->y.ptr;
    int n = X->sub[h].nrows;
    #pragma omp parallel for schedule(static) if (n >= 256)
    for (int j = 0; j < n; j++) memcpy(yout[j], yp + yoff[j], dr);
    return 1;
}

/* ---- a big step's whole-step buffers (the tier's prompt steps on the primary device) ----
 * coli_vk_xb_step_begin uploads the step's S token rows once and hands out ys, S*K output
 * rows in assignment order (s*K + k). Sub-batches issued with coli_vk_xb_sub_issue_step
 * then take the grouped route reading their x rows through the assignment map and
 * writing their y rows straight to ys: no packed x rows (K copies of each token row) and
 * no copy back at the join. A batch the grouped route does not take is packed from the
 * caller's rows as before and copied into ys at its join. coli_vk_xb_step_sum adds every
 * used row of a token in rank order on the device, so the host reads S rows, not S*K. */
static int st_buf(XbCtx *X, XbBuf *b, size_t bytes, uint32_t memtype) {
    if (bytes <= b->region) return 1;
    if (b->buf) { vkDestroyBuffer(X->dev, b->buf, NULL); vkFreeMemory(X->dev, b->mem, NULL); }
    memset(b, 0, sizeof(*b));
    size_t r = xb_up(bytes + bytes / 4, 65536);
    if (!alloc_hostvis_mt(r, &b->buf, &b->mem, &b->ptr, memtype)) { memset(b, 0, sizeof(*b)); return 0; }
    b->region = r;
    return 1;
}
float *coli_vk_xb_step_begin(int S, int K, const float *x) {
    XbCtx *X = &g_xb[0];
    if (!X->ready || !X->st.pipe || !X->grp_rows || xb_busy(X) || S < 1 || K < 1) return NULL;
    const size_t dr = (size_t)X->D * 4;
    if ((size_t)S * K * dr > X->ssbo_range) return NULL;
    if (!st_buf(X, &X->st.xt, (size_t)S * dr, X->mt_host) || !st_buf(X, &X->st.ys, (size_t)S * K * dr, X->mt_cached) ||
        !st_buf(X, &X->st.out, (size_t)S * dr, X->mt_cached) || !st_buf(X, &X->st.w, (size_t)S * K * 4, X->mt_host) ||
        !st_buf(X, &X->st.use, (size_t)S * K * 4, X->mt_host)) return NULL;
    uint8_t *xt = X->st.xt.ptr;
    #pragma omp parallel for schedule(static) if (S >= 256)
    for (int s = 0; s < S; s++) memcpy(xt + (size_t)s * dr, x + (size_t)s * X->D, dr);
    X->st.on = 1; X->st.S = S; X->st.K = K; X->st.xh = x; X->st.steps++;
    return X->st.ys.ptr;
}
void coli_vk_xb_step_end(void) { g_xb[0].st.on = 0; }
int coli_vk_xb_sub_issue_step(int h, ColiVkExpert *const *ex, const int *rows, int count, const int *assign,
                              const float *wrows) {
    XbCtx *X = &g_xb[0];
    if (!X->st.on || h < 0 || h > 1 || count < 1) return 0;
    for (int c = 0; c < count; c++) if (!ex[c] || ex[c]->X != X) return 0;
    int total = 0;
    if (xb_grp_ok(X, ex, rows, count, &total) != 1) {   /* packed as before, copied into ys at the join */
        const float **xr = malloc((size_t)total * sizeof(*xr));
        if (!xr) return 0;
        for (int j = 0; j < total; j++) xr[j] = X->st.xh + (size_t)(assign[j] / X->st.K) * X->D;
        int ok = coli_vk_xb_sub_issue(h, ex, rows, count, xr, wrows);
        free(xr);
        return ok;
    }
    if (!X->ready || X->inflight || X->sub[h].inflight || !X->sub_x) return 0;
    size_t *off = malloc((size_t)count * 3 * sizeof(size_t));
    if (!off) return 0;
    size_t x0 = h * X->sub_x, i0 = h * X->sub_i, y0 = h * X->sub_y, xe, ie, ye; int t2;
    if (!xb_plan(X, ex, rows, count, 0, x0, i0, y0, off, &xe, &ie, &ye, &t2) ||
        xe > x0 + X->sub_x || ie > i0 + X->sub_i || ye > y0 + X->sub_y) { free(off); return 0; }
    X->sub[h].nrows = 0;   /* the rows are in ys already: nothing to copy at the join */
    int ngemm = 0;
    int ok = xb_record_submit(X, X->sub[h].cmd, X->sub[h].fence, X->sub[h].qp, ex, rows, count, off, 0, &ngemm,
                              1 + h, x0, i0, y0, assign);
    free(off);
    if (!ok) return 0;
    X->sub[h].inflight = 1; async_begin(X->d);
    X->sub_batches++; X->sub_experts += (unsigned long long)count; X->sub_rows += (unsigned long long)total;
    X->sub_gemm += (unsigned long long)ngemm;
    return 1;
}
const float *coli_vk_xb_step_sum(const float *w, const uint8_t *use, double *device_ms) {
    XbCtx *X = &g_xb[0];
    if (device_ms) *device_ms = 0;
    if (!X->st.on || xb_busy(X)) return NULL;
    const int S = X->st.S, K = X->st.K, D = X->D, n = S * K;
    float *wd = X->st.w.ptr; uint32_t *ud = X->st.use.ptr;
    for (int i = 0; i < n; i++) { wd[i] = w ? w[i] : 1.0f; ud[i] = use[i] != 0; }
    VkDescriptorBufferInfo bi[4] = {{X->st.ys.buf, 0, (size_t)n * D * 4}, {X->st.w.buf, 0, (size_t)n * 4},
                                    {X->st.use.buf, 0, (size_t)n * 4}, {X->st.out.buf, 0, (size_t)S * D * 4}};
    VkWriteDescriptorSet wr[4];
    for (int b = 0; b < 4; b++)
        wr[b] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = X->st.set,
            .dstBinding = (uint32_t)b, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &bi[b]};
    vkUpdateDescriptorSets(X->dev, 4, wr, 0, NULL);
    VkCommandBuffer cmd = X->cmd;
    if (vkResetCommandBuffer(cmd, 0) != VK_SUCCESS) return NULL;
    VkCommandBufferBeginInfo cb = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    vkBeginCommandBuffer(cmd, &cb);
    if (X->has_ts) { vkCmdResetQueryPool(cmd, X->qpool, 0, 2); vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, X->qpool, 0); }
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, X->st.pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, X->st.pl, 0, 1, &X->st.set, 0, NULL);
    int pc[3] = {S, K, D};
    vkCmdPushConstants(cmd, X->st.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, pc);
    vkCmdDispatch(cmd, (uint32_t)((D + 255) / 256), (uint32_t)S, 1);
    VkMemoryBarrier hb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_HOST_READ_BIT};
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &hb, 0, NULL, 0, NULL);
    if (X->has_ts) vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, X->qpool, 1);
    if (vkEndCommandBuffer(cmd) != VK_SUCCESS) return NULL;
    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cmd};
    if (vkResetFences(X->dev, 1, &X->fence) != VK_SUCCESS || vk_submit(X->d, X->q, &si, X->fence) != VK_SUCCESS) return NULL;
    async_begin(X->d);
    VkResult r = vk_fence_wait_gemm(X->dev, X->fence);
    async_end(X->d);
    if (r != VK_SUCCESS) { fprintf(stderr, "[VK] expert batch: the step's sum failed (%d)\n", r); X->ready = 0; return NULL; }
    double ms = 0;
    if (X->has_ts) {
        uint64_t ts[2];
        if (vkGetQueryPoolResults(X->dev, X->qpool, 0, 2, sizeof ts, ts, sizeof ts[0], VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
            uint64_t mask = X->ts_bits >= 64 ? ~0ull : ((1ull << X->ts_bits) - 1);
            ms = (double)((ts[1] - ts[0]) & mask) * X->ts_period / 1e6;
        }
    }
    X->st.sums++; X->st.sum_ms += ms;
    if (device_ms) *device_ms = ms;
    return X->st.out.ptr;
}

static void xb_shutdown(XbCtx *X) {
    if (!X->dev) return;
    if (X->inflight) { vk_fence_wait(X->dev, X->fence); X->inflight = 0; async_end(X->d); }
    for (int h = 0; h < 2; h++) {
        if (X->sub[h].inflight) { vk_fence_wait(X->dev, X->sub[h].fence); X->sub[h].inflight = 0; async_end(X->d); }
        if (X->sub[h].fence) vkDestroyFence(X->dev, X->sub[h].fence, NULL);
        if (X->sub[h].qp) vkDestroyQueryPool(X->dev, X->sub[h].qp, NULL);
        free(X->sub[h].yoff);
    }
    while (X->live) coli_vk_xb_expert_free(X->live);
    XbBuf *bufs[17] = {&X->x, &X->g, &X->u, &X->h, &X->y};
    for (int k = 0; k < 6; k++) bufs[5 + k] = &X->grp_it[k / 2][k % 2], bufs[11 + k] = &X->grp_et[k / 2][k % 2];
    for (int k = 0; k < 17; k++)
        if (bufs[k]->buf) { vkDestroyBuffer(X->dev, bufs[k]->buf, NULL); vkFreeMemory(X->dev, bufs[k]->mem, NULL); }
    for (int v = 0; v < 2; v++) {
        if (X->p_grp[v]) vkDestroyPipeline(X->dev, X->p_grp[v], NULL);
        if (X->sh_grp[v]) vkDestroyShaderModule(X->dev, X->sh_grp[v], NULL);
    }
    if (X->grp_pool) vkDestroyDescriptorPool(X->dev, X->grp_pool, NULL);
    XbBuf *sb[8] = {&X->st.xt, &X->st.ys, &X->st.out, &X->st.w, &X->st.use, &X->grp_map[0], &X->grp_map[1], &X->grp_map[2]};
    for (int k = 0; k < 8; k++) if (sb[k]->buf) { vkDestroyBuffer(X->dev, sb[k]->buf, NULL); vkFreeMemory(X->dev, sb[k]->mem, NULL); }
    if (X->st.pipe) vkDestroyPipeline(X->dev, X->st.pipe, NULL);
    if (X->st.sh) vkDestroyShaderModule(X->dev, X->st.sh, NULL);
    if (X->st.pool) vkDestroyDescriptorPool(X->dev, X->st.pool, NULL);
    if (X->st.pl) vkDestroyPipelineLayout(X->dev, X->st.pl, NULL);
    if (X->st.dsl) vkDestroyDescriptorSetLayout(X->dev, X->st.dsl, NULL);
    for (int v = 0; v < 2; v++) {
        if (X->p_gv[v]) vkDestroyPipeline(X->dev, X->p_gv[v], NULL);
        if (X->sh_gv[v]) vkDestroyShaderModule(X->dev, X->sh_gv[v], NULL);
    }
    if (X->pl_grp) vkDestroyPipelineLayout(X->dev, X->pl_grp, NULL);
    if (X->dsl_grp) vkDestroyDescriptorSetLayout(X->dev, X->dsl_grp, NULL);
    for (int k = 0; k < X->ndpools; k++) vkDestroyDescriptorPool(X->dev, X->dpools[k], NULL);
    free(X->dpools);
    if (X->act_pool) vkDestroyDescriptorPool(X->dev, X->act_pool, NULL);
    VkPipeline ps[3 + VK_GEMM_SLOTS] = {X->p_gu, X->p_mv, X->p_act};
    for (int k = 0; k < VK_GEMM_SLOTS; k++) ps[3 + k] = X->p_mm[k];
    for (int k = 0; k < 3 + VK_GEMM_SLOTS; k++) if (ps[k]) vkDestroyPipeline(X->dev, ps[k], NULL);
    if (X->sh_act) vkDestroyShaderModule(X->dev, X->sh_act, NULL);
    if (X->sh_gu) vkDestroyShaderModule(X->dev, X->sh_gu, NULL);
    if (X->sh_mv) vkDestroyShaderModule(X->dev, X->sh_mv, NULL);
    if (X->sh_mm) vkDestroyShaderModule(X->dev, X->sh_mm, NULL);
    for (int k = 0; k < VK_COOP_SLOTS; k++) if (X->p_coop[k]) vkDestroyPipeline(X->dev, X->p_coop[k], NULL);
    if (X->sh_coop) vkDestroyShaderModule(X->dev, X->sh_coop, NULL);
    if (X->pl6) vkDestroyPipelineLayout(X->dev, X->pl6, NULL);
    if (X->pl4) vkDestroyPipelineLayout(X->dev, X->pl4, NULL);
    if (X->pl3) vkDestroyPipelineLayout(X->dev, X->pl3, NULL);
    if (X->dsl6) vkDestroyDescriptorSetLayout(X->dev, X->dsl6, NULL);
    if (X->dsl4) vkDestroyDescriptorSetLayout(X->dev, X->dsl4, NULL);
    if (X->dsl3) vkDestroyDescriptorSetLayout(X->dev, X->dsl3, NULL);
    if (X->p_v4) vkDestroyPipeline(X->dev, X->p_v4, NULL);
    if (X->sh_v4) vkDestroyShaderModule(X->dev, X->sh_v4, NULL);
    if (X->pl_v4) vkDestroyPipelineLayout(X->dev, X->pl_v4, NULL);
    if (X->dsl_v4) vkDestroyDescriptorSetLayout(X->dev, X->dsl_v4, NULL);
    if (X->v4_pool) vkDestroyDescriptorPool(X->dev, X->v4_pool, NULL);
    if (X->qpool) vkDestroyQueryPool(X->dev, X->qpool, NULL);
    if (X->fence) vkDestroyFence(X->dev, X->fence, NULL);
    if (X->cpool) vkDestroyCommandPool(X->dev, X->cpool, NULL);
    free(X->yoff);
    memset(X, 0, sizeof(*X));
}

/* A tensor in the expert tier's pool, laid out for the caller to fill in place: O
 * rows of cpu_row_bytes at *stride bytes apart (the padding is zero), and the
 * scales (fmt 10/11: one float the caller sets to 1). The tier's uploader thread
 * calls this; the pool's lock is the only shared state it touches. */
int coli_vk_tier_tensor_dev(int dev, ColiVkTensor **t, int fmt, int I, int O, int gs,
                            uint8_t **rows, size_t *stride, float **scales) {
    if ((dev == 1 ? !G2.ready : !G.ready) || !fmt_uploadable(fmt, gs)) return 0;
    void *w, *s;
    ColiVkTensor *n = tensor_alloc(dev == 1 ? &g_tpool2 : &g_tpool, fmt, I, O, gs, &w, &s);
    if (!n) return 0;
    *t = n; *rows = w; *stride = (size_t)n->rowWords * 4; *scales = s;
    return 1;
}
int coli_vk_tier_tensor(ColiVkTensor **t, int fmt, int I, int O, int gs,
                        uint8_t **rows, size_t *stride, float **scales) {
    return coli_vk_tier_tensor_dev(0, t, fmt, I, O, gs, rows, stride, scales);
}
int coli_vk_tier_tensor_extra(ColiVkTensor **t, int fmt, int I, int O, int gs,
                              uint8_t **rows, size_t *stride, float **scales) {
    if (!G.ready || !fmt_uploadable(fmt, gs)) return 0;
    void *w, *s;
    ColiVkTensor *n = tensor_alloc(&g_xpool, fmt, I, O, gs, &w, &s);
    if (!n) return 0;
    *t = n; *rows = w; *stride = (size_t)n->rowWords * 4; *scales = s;
    return 1;
}
/* Staged uploads: the host images of tensors filled in place (coli_vk_tier_tensor) go
 * to their device-local ranges, all of them before this returns, and are freed. Mapped
 * memory: nothing to do. Any thread; the tensors of one call on one device. */
int coli_vk_tensor_commit(ColiVkTensor *const *t, int n) {
    int dev = -1;
    for (int k = 0; k < n; k++) if (t[k] && t[k]->img) dev = t[k]->dev;
    if (dev < 0) return 1;
    VkUp *u = &g_up[dev];
    pthread_mutex_lock(&u->mx);
    int ok = 1;
    for (int k = 0; k < n && ok; k++) {
        if (!t[k] || !t[k]->img) continue;
        size_t sb = scale_floats(t[k]->fmt, t[k]->I, t[k]->O, t[k]->gs) * sizeof(float);
        /* straight from the host image when its pages import (up_import), else staged */
        VkBuffer dst[2] = {t[k]->wbuf, t[k]->sbuf};
        size_t doff[2] = {0, 0}, soff[2] = {0, t[k]->wbytes}, len[2] = {t[k]->wbytes, sb};
        ok = up_add_host(u, t[k]->img, t[k]->wbytes + sb, 2, dst, doff, soff, len) ||
             (up_add(u, dev, t[k]->wbuf, 0, t[k]->wbytes, up_fill_bytes, t[k]->img) &&
              up_add(u, dev, t[k]->sbuf, 0, sb, up_fill_bytes, t[k]->img + t[k]->wbytes));
        if (ok && k == 0 && n > 1 && up_fault("commit")) ok = up_fail(u, VK_ERROR_OUT_OF_DEVICE_MEMORY, "commit");
    }
    ok = up_finish(u, dev) && ok;
    if (!ok) up_failed(dev, "the tensors are not resident");
    pthread_mutex_unlock(&u->mx);
    for (int k = 0; k < n; k++) if (t[k]) { up_img_free(t[k]->img, t[k]->img_al); t[k]->img = NULL; }
    return ok;
}
/* A tier tensor filled again in place (a staging slot of the tier's streaming): mapped
 * memory hands its mapping back (the padding past each row is still the zeros of its
 * allocation, and the caller writes the rows only), staged uploads a fresh zeroed host
 * image that coli_vk_tensor_commit copies over and frees. Not while a batch reads it. */
int coli_vk_tensor_refill(ColiVkTensor *t, uint8_t **rows, size_t *stride, float **scales) {
    if (!G.ready || !t || t->pool != &g_tpool) return 0;
    if (g_up[t->dev].on) {
        size_t sb = scale_floats(t->fmt, t->I, t->O, t->gs) * sizeof(float);
        up_img_free(t->img, t->img_al);
        if (!(t->img = up_img_alloc(t->dev, t->wbytes + sb, &t->img_al))) return 0;
        *rows = t->img; *scales = (float *)(t->img + t->wbytes);
    } else {
        if (!t->wmap || !t->smap) return 0;
        *rows = t->wmap; *scales = (float *)t->smap;
    }
    *stride = (size_t)t->rowWords * 4;
    return 1;
}
int coli_vk_staged(void) { return G.ready && g_up[0].on; }
size_t coli_vk_tensor_row_bytes(int fmt, int I) { return cpu_row_bytes(fmt, I); }
size_t coli_vk_tensor_payload(int fmt, int I, int O, int gs) {
    if (fmt == 4 || fmt == 7 || fmt == 12 || fmt == 13) { if (gs < 1) return 0; } else gs = 0;
    return (size_t)rowwords(fmt, I) * 4 * (size_t)O + scale_floats(fmt, I, O, gs) * sizeof(float);
}
size_t coli_vk_buffer_alignment(void) { return G.buf_align ? G.buf_align : 256; }
size_t coli_vk_tensor_scale_count(int fmt, int I, int O, int gs) { return scale_floats(fmt, I, O, gs); }

/* Staged uploads: where the resident data ended up, at exit (the tests read it). */
static void place_report(void) {
    VkUp *u = &g_up[0];
    if (!u->on) return;
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(G.phys, &mp);
    ColiVkPoolStats w, t, x;
    coli_vk_pool_stats(0, &w); coli_vk_pool_stats(1, &t); coli_vk_pool_stats(4, &x);
    t.peak_used += x.peak_used;   /* the expert tier: its extra layers' pool too */
    size_t kv = 0, ln = 0;
    for (int l = 0; l < VK_KV_LAYERS; l++) {
        if (G.kv[l].bl) kv += (size_t)G.kv[l].rows * (G.kv[l].K + G.kv[l].R) * 4;
        if (G.lnbuf[l]) ln += (size_t)G.lnlen[l] * 4;
    }
    VkMemoryPropertyFlags fd = mp.memoryTypes[u->mt_dev].propertyFlags, fc = mp.memoryTypes[G.memtype_dev].propertyFlags;
    size_t host = ((fd & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) ? 0 : w.peak_used + t.peak_used + kv + ln);
    const double M = 1048576.0;
    fprintf(stderr, "[VK] memory at exit: weights %.1f MiB, expert tier %.1f MiB (peaks), KV mirror %.1f MiB in "
            "device-local memory type %u (%s); the dense chain's state in type %u (%s); %.1f MiB staged in %llu copies "
            "(%llu submits), %.1f MiB of them straight from host memory in %llu copies (%llu not imported), "
            "%llu blocks zero-filled; resident data in host memory: %.1f MiB\n",
            w.peak_used / M, t.peak_used / M, (kv + ln) / M, u->mt_dev,
            (fd & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) ? "host-visible" : "not host-visible", G.memtype_dev,
            (fc & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) ? "device-local" : "host memory",
            (u->bytes + PW.bytes + u->imp_bytes) / M, u->copies + u->imp_copies, u->submits, u->imp_bytes / M,
            u->imp_copies, u->imp_refused, u->blocks_filled, host / M);
}

/* COLI_VK_DEV2's device and everything on it: the tier's batch and pool, the registry's
 * pool, the expert group's pipelines. After vkt_shutdown (its experts are freed). */
static void dev2_shutdown(void) {
    if (!G2.dev) return;
    pthread_mutex_lock(&g_up[1].mx); pthread_mutex_lock(&g_qmx[1]);
    vkDeviceWaitIdle(G2.dev);
    pthread_mutex_unlock(&g_qmx[1]); pthread_mutex_unlock(&g_up[1].mx);
    xb_shutdown(&g_xb[1]);
    G2.inflight = 0;
    g_async_inflight[1] = 0;   /* idle: the deferred frees run now */
    tensor_reap(1);
    Scratch *sc[3] = {&G2.x, &G2.h, &G2.y};
    for (int k = 0; k < 3; k++) if (sc[k]->buf) { vkDestroyBuffer(G2.dev, sc[k]->buf, NULL); vkFreeMemory(G2.dev, sc[k]->mem, NULL); }
    if (G2.pool) vkDestroyDescriptorPool(G2.dev, G2.pool, NULL);
    if (G2.fence) vkDestroyFence(G2.dev, G2.fence, NULL);
    if (G2.cpool) vkDestroyCommandPool(G2.dev, G2.cpool, NULL);
    if (G2.pipe) vkDestroyPipeline(G2.dev, G2.pipe, NULL);
    if (G2.pipe_gu) vkDestroyPipeline(G2.dev, G2.pipe_gu, NULL);
    if (G2.plyt) vkDestroyPipelineLayout(G2.dev, G2.plyt, NULL);
    if (G2.plyt_gu) vkDestroyPipelineLayout(G2.dev, G2.plyt_gu, NULL);
    if (G2.dsl) vkDestroyDescriptorSetLayout(G2.dev, G2.dsl, NULL);
    if (G2.dsl_gu) vkDestroyDescriptorSetLayout(G2.dev, G2.dsl_gu, NULL);
    if (G2.sh_qmm) vkDestroyShaderModule(G2.dev, G2.sh_qmm, NULL);
    if (G2.sh_gu) vkDestroyShaderModule(G2.dev, G2.sh_gu, NULL);
    pool_destroy(&g_wpool2);
    pool_destroy(&g_tpool2);
    up_destroy(&g_up[1]);
    g_up[1].on = g_up[1].shared = 0; g_up[1].failed = 0; g_up[1].nfams = 0; g_up[1].dev = VK_NULL_HANDLE;
    vkDestroyDevice(G2.dev, NULL);
    memset(&G2, 0, sizeof(G2));
    vk_mem_forget(1);
    vk_pool_blocks(1);
}

void coli_vk_shutdown(void) {
    if (!G.ready) return;
    place_report();
    /* the uploader's thread may still be in a submit on a shared queue */
    pthread_mutex_lock(&g_up[0].mx); pthread_mutex_lock(&g_qmx[0]);
    vkDeviceWaitIdle(G.dev);
    pthread_mutex_unlock(&g_qmx[0]); pthread_mutex_unlock(&g_up[0].mx);
    xb_shutdown(&g_xb[0]);
    if (G.x.buf) { vkDestroyBuffer(G.dev, G.x.buf, NULL); vkFreeMemory(G.dev, G.x.mem, NULL); }
    if (G.y.buf) { vkDestroyBuffer(G.dev, G.y.buf, NULL); vkFreeMemory(G.dev, G.y.mem, NULL); }
    if (G.h.buf) { vkDestroyBuffer(G.dev, G.h.buf, NULL); vkFreeMemory(G.dev, G.h.mem, NULL); }
    if (G.eg_x.buf) { vkDestroyBuffer(G.dev, G.eg_x.buf, NULL); vkFreeMemory(G.dev, G.eg_x.mem, NULL); }
    if (G.eg_h.buf) { vkDestroyBuffer(G.dev, G.eg_h.buf, NULL); vkFreeMemory(G.dev, G.eg_h.mem, NULL); }
    if (G.eg_y.buf) { vkDestroyBuffer(G.dev, G.eg_y.buf, NULL); vkFreeMemory(G.dev, G.eg_y.mem, NULL); }
    if (G.att_sc.buf) { vkDestroyBuffer(G.dev, G.att_sc.buf, NULL); vkFreeMemory(G.dev, G.att_sc.mem, NULL); }
    if (G.att_ctx.buf) { vkDestroyBuffer(G.dev, G.att_ctx.buf, NULL); vkFreeMemory(G.dev, G.att_ctx.mem, NULL); }
    if (G.y2.buf) { vkDestroyBuffer(G.dev, G.y2.buf, NULL); vkFreeMemory(G.dev, G.y2.mem, NULL); }
    if (G.qp1.buf) { vkDestroyBuffer(G.dev, G.qp1.buf, NULL); vkFreeMemory(G.dev, G.qp1.mem, NULL); }
    if (G.qp2.buf) { vkDestroyBuffer(G.dev, G.qp2.buf, NULL); vkFreeMemory(G.dev, G.qp2.mem, NULL); }
    for (int l = 0; l < VK_KV_LAYERS; l++)   /* per-layer resident norm weights (attn_qprep) */
        if (G.lnbuf[l]) { vkDestroyBuffer(G.dev, G.lnbuf[l], NULL); vkFreeMemory(G.dev, G.lnmem[l], NULL); }
    if (G.pair_pool) vkDestroyDescriptorPool(G.dev, G.pair_pool, NULL);
    coli_vk_kv_reset();
    if (G.eg_pool) vkDestroyDescriptorPool(G.dev, G.eg_pool, NULL);
    vkDestroyFence(G.dev, G.fence, NULL);
    vkDestroyFence(G.dev, G.eg_fence, NULL);
    vkDestroyCommandPool(G.dev, G.cpool, NULL);
    vkDestroyDescriptorPool(G.dev, G.dpool, NULL);
    vkDestroyPipeline(G.dev, G.pipe, NULL);
    for (int k = 0; k < VK_GEMM_SLOTS; k++) if (G.pipe_gemm[k]) vkDestroyPipeline(G.dev, G.pipe_gemm[k], NULL);
    for (int k = 0; k < VK_COOP_SLOTS; k++) if (G.pipe_coop[k]) vkDestroyPipeline(G.dev, G.pipe_coop[k], NULL);
    if (G.shader_gemm) vkDestroyShaderModule(G.dev, G.shader_gemm, NULL);
    if (G.shader_coop) vkDestroyShaderModule(G.dev, G.shader_coop, NULL);
    vkDestroyPipelineLayout(G.dev, G.plyt, NULL);
    vkDestroyDescriptorSetLayout(G.dev, G.dsl, NULL);
    vkDestroyShaderModule(G.dev, G.shader, NULL);
    if (G.shader_gu) {
        vkDestroyDescriptorPool(G.dev, G.dpool_gu, NULL);
        vkDestroyPipeline(G.dev, G.pipe_gu, NULL);
        vkDestroyPipelineLayout(G.dev, G.plyt_gu, NULL);
        vkDestroyDescriptorSetLayout(G.dev, G.dsl_gu, NULL);
        vkDestroyShaderModule(G.dev, G.shader_gu, NULL);
    }
    if (G.shader_att) {
        vkDestroyDescriptorPool(G.dev, G.dpool_att, NULL);
        vkDestroyPipeline(G.dev, G.pipe_att, NULL);
        vkDestroyPipelineLayout(G.dev, G.plyt_att, NULL);
        vkDestroyDescriptorSetLayout(G.dev, G.dsl_att, NULL);
        vkDestroyShaderModule(G.dev, G.shader_att, NULL);
    }
    g_async_inflight[0] = 0;   /* idle after vkDeviceWaitIdle: the deferred frees run now */
    tensor_reap(0);
    pthread_mutex_lock(&g_imports_mx);   /* imported rows: their memory objects go before the device */
    for (ColiVkTensor *t = g_imports; t; t = t->imp_next) {
        vkDestroyBuffer(G.dev, t->wbuf, NULL); vkFreeMemory(G.dev, t->imp, NULL);
        t->wbuf = VK_NULL_HANDLE; t->imp = VK_NULL_HANDLE;
    }
    g_imports = NULL;
    pthread_mutex_unlock(&g_imports_mx);
    pool_destroy(&g_wpool);    /* weight blocks: unmapped/freed with the device */
    pool_destroy(&g_tpool);
    pool_destroy(&g_xpool);
    if (PW.buf) { vkDestroyBuffer(G.dev, PW.buf, NULL); vkFreeMemory(G.dev, PW.mem, NULL); }
    free(PW.cp);
    memset(&PW, 0, sizeof PW);
    up_destroy(&g_up[0]);
    g_up[0].on = g_up[0].shared = 0; g_up[0].failed = 0; g_up[0].nfams = 0; g_up[0].dev = VK_NULL_HANDLE;
    dev2_shutdown();   /* a device of this instance: it goes first */
    vkDestroyDevice(G.dev, NULL);
    vkDestroyInstance(G.inst, NULL);
    memset(&G, 0, sizeof(G));
    vk_mem_forget(0);
    vk_pool_blocks(0);
}

/* ---- the dense chain's view of the device (vk_chain.c) -------------------------
 * The chain records its own command buffers on the main queue, from the engine
 * thread, with pipelines of its own built from the same shader directory; it reads
 * the resident tensors the engines already uploaded (their VkBuffers below). */
int coli_vk_core(ColiVkCore *o) {
    if (!G.ready || !o) return 0;
    memset(o, 0, sizeof *o);
    o->instance = (void *)G.inst; o->phys = (void *)G.phys; o->device = (void *)G.dev;
    o->queue = (void *)G.queue; o->qfam = G.qfam;
    /* staged uploads: the chain's host-written buffers (its frames' staging, the rows
     * each layer step uploads) in host staging memory, not in the small window */
    o->memtype_host = g_up[0].on ? g_up[0].mt_stage : G.memtype;
    o->memtype_cached = G.memtype_cached; o->memtype_dev = G.memtype_dev;
    o->ssbo_align = G.ssbo_align; o->ssbo_range = G.ssbo_range; o->spv_path = G.spv_path;
    o->has_prio = G.has_prio;
    for (int k = 0; k < VK_GEMM_SLOTS && k < 4; k++) {
        if (!G.pipe_gemm[k]) break;
        VkGemmTile t = G.gemm_t[k];
        o->gemm_tile[k][0] = t.bm; o->gemm_tile[k][1] = t.bn; o->gemm_tile[k][2] = t.bk;
        o->gemm_tile[k][3] = t.tm; o->gemm_tile[k][4] = t.tn; o->gemm_tile[k][5] = t.pf;
        o->gemm_tiles = k + 1;
    }
    o->gemm_min_s = G.gemm_min_s; o->gemm_min_so = G.gemm_min_so;
    o->integrated = coli_vk_device_integrated(); o->shares_ram = coli_vk_device_shares_ram();
    o->coop_sg = G.has_coop ? G.coop_sg : 0;
    return 1;
}
int coli_vk_tensor_info(const ColiVkTensor *t, ColiVkTensorInfo *o) {
    if (!t || !o || t->dev != 0) return 0;
    o->wbuf = (void *)t->wbuf; o->sbuf = (void *)t->sbuf;
    o->fmt = t->fmt; o->I = t->I; o->O = t->O; o->rowWords = t->rowWords; o->gs = t->gs;
    return 1;
}
/* A fence the chain waited on failed: the device is gone, everyone falls back. */
void coli_vk_mark_lost(void) { G.ready = 0; }

/* ---- the dense chain on either device (vk_chain.c keeps a context per device) --------
 * d = 0 is the primary device, what the calls without _dev answer; d = 1 is COLI_VK_DEV2's
 * (G2), which until now held only routed experts. The chain's layers on it run the same
 * shaders: the core below is G2's, with the primary's GEMM tiles (a tile is a speed
 * choice; every tile computes the same bits). */
int coli_vk_core_dev(int d, ColiVkCore *o) {
    if (d == 0) return coli_vk_core(o);
    if (d != 1 || !G2.ready || !G.ready || !o) return 0;
    if (!coli_vk_core(o)) return 0;   /* the tiles, the shader path: the primary's */
    o->phys = (void *)G2.phys; o->device = (void *)G2.dev; o->queue = (void *)G2.queue; o->qfam = G2.qfam;
    o->memtype_host = g_up[1].on ? g_up[1].mt_stage : G2.memtype;
    o->memtype_cached = G2.memtype_cached; o->memtype_dev = G2.memtype_dev;
    o->ssbo_align = G2.ssbo_align; o->ssbo_range = G2.ssbo_range;
    o->has_prio = 0;
    o->integrated = G2.integrated; o->shares_ram = G2.shares_ram;
    o->coop_sg = 0;   /* the second device's cooperative matrices are not probed: its chain keeps the plain shaders */
    return 1;
}
int coli_vk_available_dev(int d) { return d == 1 ? G2.ready : G.ready; }
void coli_vk_mark_lost_dev(int d) { if (d == 1) G2.ready = 0; else G.ready = 0; }
int coli_vk_queue_submit_dev(int d, void *queue, const void *submit_info, void *fence) {
    return (int)vk_submit(d == 1 ? 1 : 0, (VkQueue)queue, (const VkSubmitInfo *)submit_info, (VkFence)fence);
}
int coli_vk_tensor_info_dev(const ColiVkTensor *t, ColiVkTensorInfo *o, int *dev) {
    if (!t || !o || t->dev < 0 || t->dev > 1) return 0;
    o->wbuf = (void *)t->wbuf; o->sbuf = (void *)t->sbuf;
    o->fmt = t->fmt; o->I = t->I; o->O = t->O; o->rowWords = t->rowWords; o->gs = t->gs;
    if (dev) *dev = t->dev;
    return 1;
}
int coli_vk_mem_budget_dev(int d, double *used_gb, double *budget_gb) {
    return d == 1 ? coli_vk_mem_budget2(used_gb, budget_gb) : coli_vk_mem_budget(used_gb, budget_gb);
}
size_t coli_vk_device_used_dev(int d) { return (size_t)__atomic_load_n(&g_mem.used[d == 1 ? 1 : 0], __ATOMIC_RELAXED); }
size_t coli_vk_device_local_bytes_dev(int d) {
    if (d != 1) return coli_vk_device_local_bytes();
    if (!G2.phys) return 0;
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(G2.phys, &mp);
    VkDeviceSize m = 0;
    for (uint32_t i = 0; i < mp.memoryHeapCount; i++)
        if ((mp.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) && mp.memoryHeaps[i].size > m) m = mp.memoryHeaps[i].size;
    if (g_mem.cap[1] && g_mem.cap[1] < m) m = g_mem.cap[1];   /* COLI_VK_DEVICE_CAP_MB */
    return (size_t)m;
}
size_t coli_vk_free_bytes_dev(int d) {
    if (d != 1) return coli_vk_free_bytes();
    if (!G2.phys) return 0;
    if (g_mem.cap[1]) return (size_t)vk_mem_room(1);
    double used = 0, bud = 0;
    if (coli_vk_mem_budget2(&used, &bud)) return bud > used ? (size_t)((bud - used) * 1e9) : 0;
    size_t dl = coli_vk_device_local_bytes_dev(1), u = coli_vk_device_used_dev(1);
    return dl > u ? dl - u : 0;
}
void coli_vk_mem_info_dev(int d, size_t *used, size_t *count) {
    if (d != 1) { coli_vk_mem_info(used, count); return; }
    if (used) *used = (size_t)__atomic_load_n(&g_wpool2.bytes, __ATOMIC_RELAXED);
    if (count) *count = (size_t)__atomic_load_n(&g_wpool2.tensors, __ATOMIC_RELAXED);
}
size_t coli_vk_buffer_alignment_dev(int d) {
    if (d != 1) return coli_vk_buffer_alignment();
    return G2.buf_align ? G2.buf_align : 256;
}
/* The chain's submits: through the lock the staged uploader takes on a shared queue. */
int coli_vk_queue_submit(void *queue, const void *submit_info, void *fence) {
    return (int)vk_submit(0, (VkQueue)queue, (const VkSubmitInfo *)submit_info, (VkFence)fence);
}

#ifdef VK_TEST
// ---- standalone GPU-vs-CPU validation + microbench --------------------------
#include <math.h>
#include <time.h>

static double now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9; }

static int g_ref_gs = 64;   /* fmt=4 group size the harness cases use */
/* A digest (FNV-1a) of the device's results in every case below: mapped memory and
 * staged uploads must print the same one (tests/vulkan_engines.sh staged). */
static uint64_t g_digest = 1469598103934665603ULL;
static void digest(const void *p, size_t n) {
    const uint8_t *b = p;
    for (size_t i = 0; i < n; i++) { g_digest ^= b[i]; g_digest *= 1099511628211ULL; }
}
static size_t ref_rowbytes(int fmt, int I) {
    return fmt == 1 || fmt == 12 || fmt == 13 ? (size_t)I : fmt == 5 ? (size_t)((I + 63) / 64) * 24
         : fmt == 10 ? (size_t)I * 4 : fmt == 11 || fmt == 14 ? (size_t)I * 2 : (size_t)(I + 1) / 2;
}
static size_t ref_scales(int fmt, int I, int O) {   // scale COUNT (per-group for fmt 4/5)
    if (fmt == 10 || fmt == 11 || fmt == 14) return 1;
    if (fmt == 5) return (size_t)O * (size_t)((I + 63) / 64);
    if (fmt == 4 || fmt == 7 || fmt == 12 || fmt == 13) return (size_t)O * (size_t)((I + g_ref_gs - 1) / g_ref_gs);
    return (size_t)O;
}
static float deq(const uint8_t *row, int fmt, int i) {
    if (fmt == 1 || fmt == 13) { int b = ((const int8_t *)row)[i]; return (float)b; }
    if (fmt == 10) { float f; memcpy(&f, row + (size_t)i * 4, 4); return f; }
    if (fmt == 12) { uint8_t b = row[i]; int e = (b >> 3) & 15, m = b & 7;   /* E4M3_LUT semantics */
                     if ((b & 0x7f) == 0x7f) return NAN;
                     float v = e == 0 ? m / 512.0f : ldexpf(1.0f + m / 8.0f, e - 7);
                     return (b & 0x80) ? -v : v; }
    if (fmt == 11) { uint16_t h; memcpy(&h, row + (size_t)i * 2, 2); uint32_t u = (uint32_t)h << 16;
                     float f; memcpy(&f, &u, 4); return f; }
    if (fmt == 14) { uint16_t h; memcpy(&h, row + (size_t)i * 2, 2);   /* f16: normals and subnormals */
                     int e = (h >> 10) & 31, m = h & 1023;
                     float v = e == 0 ? ldexpf((float)m, -24) : ldexpf(1.0f + m / 1024.0f, e - 15);
                     return (h & 0x8000) ? -v : v; }
    if (fmt == 5) {   // int3-g64: 16B low plane (2 bits) + 8B high plane (1 bit), v+4
        const uint8_t *lo = row + (size_t)(i >> 6) * 24, *hi = lo + 16; int j = i & 63;
        unsigned u = ((lo[j >> 2] >> ((j & 3) * 2)) & 3u) | (((hi[j >> 3] >> (j & 7)) & 1u) << 2);
        return (float)((int)u - 4); }
    uint8_t v = row[i >> 1]; int nib = (i & 1) ? (v >> 4) : (v & 15);
    if (fmt == 7) { static const float e2m1[8] = {0, 0.5f, 1, 1.5f, 2, 3, 4, 6};   /* MXFP4 */
                    return (nib & 8) ? -e2m1[nib & 7] : e2m1[nib & 7]; }
    return (float)(nib - 8);
}

static void cpu_ref(float *y, const float *x, const uint8_t *w, const float *sc,
                    int fmt, int S, int I, int O) {
    size_t rb = ref_rowbytes(fmt, I);
    int gw2 = (fmt == 4 || fmt == 7 || fmt == 12 || fmt == 13) ? g_ref_gs : 64, ng = (I + gw2 - 1) / gw2;
    for (int s = 0; s < S; s++) for (int o = 0; o < O; o++) {
        double sum = 0; const uint8_t *row = w + (size_t)o * rb;
        if (fmt == 5 || fmt == 4 || fmt == 7 || fmt == 12 || fmt == 13) {   // per-group scales fold inside the sum
            for (int g = 0; g < ng; g++) {
                double a = 0; int end = (g + 1) * gw2 < I ? (g + 1) * gw2 : I;
                for (int i = g * gw2; i < end; i++) a += x[s * I + i] * deq(row, fmt, i);
                sum += a * sc[(size_t)o * ng + g];
            }
            y[s * O + o] = (float)sum;
        } else {
            for (int i = 0; i < I; i++) sum += x[s * I + i] * deq(row, fmt, i);
            y[s * O + o] = (float)(fmt == 10 || fmt == 11 || fmt == 14 ? sum : sum * sc[o]);
        }
    }
}

/* dequant dot of one weight row against x with that row's scales applied —
 * per-row for fmt 1/2, per 64-group for fmt=5. scb = tensor scale array, o = row. */
static double ref_dot(const float *x, const uint8_t *row, const float *scb, int o, int fmt, int I) {
    if (fmt == 5 || fmt == 4) {
        int gw2 = fmt == 4 ? g_ref_gs : 64;
        int ng = (I + gw2 - 1) / gw2; double sum = 0;
        for (int g = 0; g < ng; g++) {
            double a = 0; int end = (g + 1) * gw2 < I ? (g + 1) * gw2 : I;
            for (int i = g * gw2; i < end; i++) a += x[i] * deq(row, fmt, i);
            sum += a * scb[(size_t)o * ng + g];
        }
        return sum;
    }
    double sum = 0;
    for (int i = 0; i < I; i++) sum += x[i] * deq(row, fmt, i);
    return sum * scb[o];
}

/* Random activations, weights and scales for one case: x[S,I] in [-1,1), raw weight
 * bytes (no NaN/Inf for the float and fp8 formats), scales in [0.01,0.02). */
static void case_fill(int fmt, int S, int I, int O, float **x, uint8_t **w, float **sc) {
    size_t rb = ref_rowbytes(fmt, I), nsc = ref_scales(fmt, I, O);
    *x = malloc((size_t)S * I * sizeof(float));
    *w = malloc(rb * O);
    *sc = malloc(nsc * sizeof(float));
    for (size_t i = 0; i < (size_t)S * I; i++) (*x)[i] = (float)((rand() % 200 - 100) / 100.0);
    for (size_t i = 0; i < rb * O; i++) (*w)[i] = rand() & 0xff;
    if (fmt == 12)                         /* no NaN bytes: they would poison the comparison */
        for (size_t i = 0; i < rb * O; i++) if (((*w)[i] & 0x7f) == 0x7f) (*w)[i] ^= 1;
    if (fmt == 10 || fmt == 11)            /* float weights: random bytes could be NaN/Inf */
        for (size_t i = 0; i < (size_t)I * O; i++) {
            float f = (float)((rand() % 2001 - 1000) / 1000.0);
            if (fmt == 10) memcpy(*w + i * 4, &f, 4);
            else { uint32_t u; memcpy(&u, &f, 4); uint16_t h = (uint16_t)(u >> 16); memcpy(*w + i * 2, &h, 2); }
        }
    if (fmt == 14)                         /* f16: the same values as 10 and 11, rounded to f16 */
        for (size_t i = 0; i < (size_t)I * O; i++) {
            float f = (float)((rand() % 2001 - 1000) / 1000.0), a = fabsf(f);
            int e; float m = frexpf(a, &e);                 /* a = m 2^e, m in [0.5, 1) */
            uint16_t h = 0;
            if (a >= ldexpf(1.0f, -14)) {                   /* normal: 10 bits after the leading one */
                int q = (int)lrintf(ldexpf(m, 11));        /* in [1024, 2048] */
                if (q == 2048) { q = 1024; e++; }
                h = (uint16_t)(((e - 1 + 15) << 10) | (q - 1024));
            } else h = (uint16_t)lrintf(ldexpf(a, 24));     /* subnormal */
            if (f < 0) h |= 0x8000;
            memcpy(*w + i * 2, &h, 2);
        }
    for (size_t o = 0; o < nsc; o++) (*sc)[o] = 0.01f + (rand() % 100) / 10000.0f;
}

static int run_case(int fmt, int S, int I, int O, int iters) {
    float *x, *sc; uint8_t *w;
    case_fill(fmt, S, I, O, &x, &w, &sc);
    float *yg = malloc((size_t)S * O * sizeof(float));
    float *yc = malloc((size_t)S * O * sizeof(float));

    ColiVkTensor *t = NULL;
    if (!coli_vk_matmul(&t, yg, x, w, sc, fmt, S, I, O, g_ref_gs)) { printf("matmul failed\n"); return 1; }
    digest(yg, (size_t)S * O * sizeof(float));
    const char *path = G.bound_gemm == 2 ? "coop" : G.bound_gemm ? "gemm" : "gemv";
    double c0 = now(); cpu_ref(yc, x, w, sc, fmt, S, I, O); double cpu_ms = (now() - c0) * 1000;
    double maxerr = 0, maxrel = 0;
    for (int i = 0; i < S * O; i++) {
        double e = fabs(yg[i] - yc[i]); if (e > maxerr) maxerr = e;
        if (fabs(yc[i]) > 1e-2) { double r = e / fabs(yc[i]); if (r > maxrel) maxrel = r; }
    }
    // microbench (GPU)
    double t0 = now();
    for (int k = 0; k < iters; k++) coli_vk_matmul(&t, yg, x, w, sc, fmt, S, I, O, g_ref_gs);
    double gpu_ms = (now() - t0) * 1000 / iters;
    char gsb[16] = "";
    if (fmt == 4 || fmt == 7 || fmt == 12 || fmt == 13) snprintf(gsb, sizeof(gsb), " gs=%d", g_ref_gs);
    printf("fmt=%d%s S=%d I=%d O=%d %s | maxerr=%.4g maxrel=%.4g | gpu=%.3f ms  cpu_ref=%.3f ms\n",
           fmt, gsb, S, I, O, path, maxerr, maxrel, gpu_ms, cpu_ms);
    coli_vk_tensor_free(t);
    free(x); free(w); free(sc); free(yg); free(yc);
    return maxrel > 1e-3 ? 1 : 0;
}

/* The tiled GEMMs against the same reference, forced for this S whatever the device
 * threshold, and required to have really run: a missing qmatmul_gemm.spv must fail
 * here, not pass on the GEMV. The fp32 GEMM always; then, on a device with the
 * cooperative-matrix pipeline, that one too for the formats it takes. gs is the group
 * size of fmt 4, 7 and 12. */
static int coop_takes(int fmt, int gs) {
    ColiVkTensor t = {.fmt = fmt, .gs = gs};
    return G.pipe_coop[0] && coop_fmt(&t);
}
static int run_gemm_case(int fmt, int S, int I, int O, int gs) {
    if (!G.pipe_gemm[0]) { printf("gemm fmt=%d: no tiled GEMM pipeline (qmatmul_gemm.spv missing?)\n", fmt); return 1; }
    int keep_s = G.gemm_min_s, keep_so = G.gemm_min_so, keep_gs = g_ref_gs, bad = 0;
    G.gemm_min_s = 2; G.gemm_min_so = 0; g_ref_gs = gs;
    G.coop_off = 1;                                     /* the fp32 GEMM */
    unsigned long long n0 = g_vk_gemm_calls;
    bad |= run_case(fmt, S, I, O, 2);
    if (g_vk_gemm_calls == n0) { printf("  ^ did not take the tiled GEMM\n"); bad = 1; }
    G.coop_off = 0;
    if (coop_takes(fmt, gs)) {                          /* the cooperative-matrix one */
        n0 = g_vk_coop_calls;
        bad |= run_case(fmt, S, I, O, 2);
        if (g_vk_coop_calls == n0) { printf("  ^ did not take the cooperative-matrix GEMM\n"); bad = 1; }
    }
    G.gemm_min_s = keep_s; G.gemm_min_so = keep_so; g_ref_gs = keep_gs;
    return bad;
}

/* Seconds per coli_vk_matmul call with the GEMM threshold at min_s (0 = GEMV): x upload,
 * dispatch and y readback, what an engine pays; and, in *gpu, per resubmit of the
 * recorded command buffer, the dispatch alone plus the submit roundtrip. */
static double time_matmul(ColiVkTensor **t, float *y, const float *x, const void *w, const float *sc,
                          int fmt, int S, int I, int O, int gs, int min_s, double *gpu) {
    int keep = G.gemm_min_s, keep_so = G.gemm_min_so; G.gemm_min_s = min_s; G.gemm_min_so = 0;
    coli_vk_matmul(t, y, x, w, sc, fmt, S, I, O, gs);            /* record + warm */
    int it = 0; double t0 = now(), el;
    do { coli_vk_matmul(t, y, x, w, sc, fmt, S, I, O, gs); it++; } while ((el = now() - t0) < 0.25 && it < 500);
    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &G.cmd};
    int ig = 0; double g0 = now(), gl;
    do { vkResetFences(G.dev, 1, &G.fence); vkQueueSubmit(G.queue, 1, &si, G.fence);
         vk_fence_wait_gemm(G.dev, G.fence); ig++;
    } while ((gl = now() - g0) < 0.25 && ig < 500);
    G.gemm_min_s = keep; G.gemm_min_so = keep_so;
    *gpu = gl / ig;
    return el / it;
}

/* COLI_VK_TEST_GEMM_BENCH=1: GEMV against the tiled GEMM, format by format, over S on
 * one engine-sized matrix (COLI_VK_TEST_GEMM_SHAPE=I,O; default 2560,6144, Qwen3.8's
 * DeltaNet z projection). GFLOP/s per call and per resubmit; the S where the GEMM
 * overtakes is what COLI_VK_GEMM_MIN_S defaults to. COLI_VK_TEST_GEMM_FMT=a,b,... and
 * COLI_VK_TEST_GEMM_S=a,b,... restrict the formats and the S values. Runs instead of
 * every other case. */
static void bench_gemm(void) {
    int I = 2560, O = 6144, fm[8] = {1, 2, 4, 5, 7, 10, 11, 12}, nf = 8;
    const char *e = getenv("COLI_VK_TEST_GEMM_SHAPE");
    if (e) sscanf(e, "%d,%d", &I, &O);
    if ((e = getenv("COLI_VK_TEST_GEMM_FMT"))) {
        nf = 0;
        for (const char *c = e; *c && nf < 8; ) { fm[nf++] = atoi(c); while (*c && *c != ',') c++; if (*c) c++; }
    }
    int Ss[16] = {2, 4, 8, 16, 32, 64, 128, 256, 512}, ns = 9, Smax = 0;
    if ((e = getenv("COLI_VK_TEST_GEMM_S"))) {         /* the S values, a,b,... */
        ns = 0;
        for (const char *c = e; *c && ns < 16; ) { Ss[ns++] = atoi(c); while (*c && *c != ',') c++; if (*c) c++; }
    }
    for (int k = 0; k < ns; k++) if (Ss[k] > Smax) Smax = Ss[k];
    printf("GEMM bench I=%d O=%d, threshold now S>=%d and S*O>=%d; GEMM tiles", I, O, G.gemm_min_s, G.gemm_min_so);
    for (int k = 0; k < VK_GEMM_SLOTS; k++) if (G.pipe_gemm[k]) printf(" %dx%d", G.gemm_t[k].bm, G.gemm_t[k].bn);
    printf("; COOP tiles (subgroup %d)", G.coop_sg);
    for (int k = 0; k < VK_COOP_SLOTS; k++) if (G.pipe_coop[k]) printf(" %dx%d", G.coop_t[k].bm, G.coop_t[k].bn);
    printf("\n");
    for (int f = 0; f < nf; f++) {
        int fmt = fm[f];
        g_ref_gs = fmt == 7 ? 32 : 64;
        float *x, *sc; uint8_t *w;
        case_fill(fmt, Smax, I, O, &x, &w, &sc);
        float *y = malloc((size_t)Smax * O * sizeof(float));
        ColiVkTensor *t = NULL;
        if (!coli_vk_matmul(&t, y, x, w, sc, fmt, 1, I, O, g_ref_gs)) { printf("fmt=%d: upload failed\n", fmt); continue; }
        for (int k = 0; k < ns; k++) {
            int S = Ss[k];
            double gv, gm, gvg, gmg, cm = 0, cmg = 0;
            gv = time_matmul(&t, y, x, w, sc, fmt, S, I, O, g_ref_gs, 0, &gvg);
            G.coop_off = 1;
            gm = time_matmul(&t, y, x, w, sc, fmt, S, I, O, g_ref_gs, 2, &gmg);
            G.coop_off = 0;
            if (coop_takes(fmt, g_ref_gs)) cm = time_matmul(&t, y, x, w, sc, fmt, S, I, O, g_ref_gs, 2, &cmg);
            double fl = 2.0 * S * I * O / 1e9;
            printf("fmt=%-2d S=%-4d | GEMV %8.3f ms %7.1f GFLOP/s (gpu %8.3f) | GEMM %8.3f ms %7.1f GFLOP/s (gpu %8.3f)",
                   fmt, S, gv * 1e3, fl / gv, gvg * 1e3, gm * 1e3, fl / gm, gmg * 1e3);
            if (cm > 0) printf(" | COOP %8.3f ms %7.1f GFLOP/s (gpu %8.3f)", cm * 1e3, fl / cm, cmg * 1e3);
            printf("\n");
        }
        coli_vk_tensor_free(t);
        free(x); free(w); free(sc); free(y);
    }
    g_ref_gs = 64;
}

/* Batched throughput: record N dispatches in ONE command buffer, one submit + one
 * fence wait — the amortized per-matmul cost with the submit roundtrip spread across
 * the batch (how the real expert tier would drive it). Reuses the descriptor binding
 * left by a prior coli_vk_matmul call for this tensor/shape. */
static double bench_batched(ColiVkTensor *t, const float *x, int fmt, int S, int I, int O, int N) {
    size_t xb = (size_t)S * I * sizeof(float);
    memcpy(G.x.ptr, x, xb);
    vkResetCommandBuffer(G.cmd, 0);
    VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkBeginCommandBuffer(G.cmd, &begin);
    vkCmdBindPipeline(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe);
    vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt, 0, 1, &G.dset, 0, NULL);
    struct PC pc = {fmt, S, I, O, t->rowWords, t->gs};
    vkCmdPushConstants(G.cmd, G.plyt, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    /* Every dispatch in the loop rewrites the same output, so the barrier must also
     * order the next write after the previous one (WAW), not only the reads. */
    VkMemoryBarrier mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
    for (int i = 0; i < N; i++) {
        vkCmdDispatch(G.cmd, (uint32_t)((O + 7) / 8), (uint32_t)S, 1);
        vkCmdPipelineBarrier(G.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    }
    vkEndCommandBuffer(G.cmd);
    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &G.cmd};
    for (int warm = 0; warm < 2; warm++) {
        vkResetFences(G.dev, 1, &G.fence); vkQueueSubmit(G.queue, 1, &si, G.fence);
        vkWaitForFences(G.dev, 1, &G.fence, VK_TRUE, 10000000000ULL);
    }
    int iters = 10; double t0 = now();
    for (int k = 0; k < iters; k++) {
        vkResetFences(G.dev, 1, &G.fence); vkQueueSubmit(G.queue, 1, &si, G.fence);
        vkWaitForFences(G.dev, 1, &G.fence, VK_TRUE, 10000000000ULL);
    }
    G.cmd_ready = 0;   /* we clobbered the cached command buffer */
    return (now() - t0) * 1000.0 / iters / N;   /* ms per matmul, roundtrip amortized */
}

/* Fused gate+up correctness vs CPU ref: hidden = silu(gate*x)*(up*x). */
static int run_gate_up(int fmt, int S, int D, int I) {
    size_t rb = ref_rowbytes(fmt, D), nsc = ref_scales(fmt, D, I);
    float *x = malloc((size_t)S*D*4); uint8_t *gw = malloc(rb*I), *uw = malloc(rb*I);
    float *gs = malloc(nsc*4), *us = malloc(nsc*4);
    float *hg = malloc((size_t)S*I*4), *hc = malloc((size_t)S*I*4);
    for (int i = 0; i < S*D; i++) x[i] = (rand()%200-100)/100.0f;
    for (size_t i = 0; i < rb*I; i++) { gw[i] = rand()&0xff; uw[i] = rand()&0xff; }
    for (size_t o = 0; o < nsc; o++) { gs[o] = 0.01f+(rand()%100)/10000.0f; us[o] = 0.01f+(rand()%100)/10000.0f; }
    ColiVkTensor *tg = NULL, *tu = NULL;
    if (!coli_vk_gate_up(&tg, &tu, hg, x, gw, gs, uw, us, fmt, S, D, I, g_ref_gs)) { printf("gate_up failed\n"); return 1; }
    digest(hg, (size_t)S * I * 4);
    for (int s = 0; s < S; s++) for (int o = 0; o < I; o++) {
        float gt = (float)ref_dot(x+(size_t)s*D, gw+(size_t)o*rb, gs, o, fmt, D);
        float ut = (float)ref_dot(x+(size_t)s*D, uw+(size_t)o*rb, us, o, fmt, D);
        hc[s*I+o] = (gt/(1.0f+expf(-gt)))*ut;
    }
    double maxrel = 0;
    for (int i = 0; i < S*I; i++) { double e = fabs(hg[i]-hc[i]); if (fabs(hc[i])>1e-2) { double r = e/fabs(hc[i]); if (r>maxrel) maxrel = r; } }
    printf("gate_up(fused) fmt=%d S=%d D=%d I=%d | maxrel=%.4g\n", fmt, S, D, I, maxrel);
    coli_vk_tensor_free(tg); coli_vk_tensor_free(tu);
    free(x); free(gw); free(uw); free(gs); free(us); free(hg); free(hc);
    return maxrel > 1e-3 ? 1 : 0;
}

/* Batched throughput of the fused gate_up (N dispatches / one submit). */
static double bench_gu_batched(ColiVkTensor *tg, const float *x, int fmt, int S, int D, int I, int N) {
    if (!G.pipe_gu) return -1;   /* gate_up shader not loaded: run_gate_up already reported it */
    memcpy(G.x.ptr, x, (size_t)S*D*sizeof(float));
    vkResetCommandBuffer(G.cmd, 0);
    VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkBeginCommandBuffer(G.cmd, &begin);
    vkCmdBindPipeline(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe_gu);
    vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt_gu, 0, 1, &G.dset_gu, 0, NULL);
    struct PCGU pc = pcgu(fmt, S, D, I, tg->rowWords, tg->gs);
    vkCmdPushConstants(G.cmd, G.plyt_gu, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    /* Every dispatch in the loop rewrites the same output, so the barrier must also
     * order the next write after the previous one (WAW), not only the reads. */
    VkMemoryBarrier mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
    for (int i = 0; i < N; i++) {
        vkCmdDispatch(G.cmd, (uint32_t)((I+7)/8), (uint32_t)S, 1);
        vkCmdPipelineBarrier(G.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    }
    vkEndCommandBuffer(G.cmd);
    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &G.cmd};
    for (int w = 0; w < 2; w++) { vkResetFences(G.dev,1,&G.fence); vkQueueSubmit(G.queue,1,&si,G.fence); vkWaitForFences(G.dev,1,&G.fence,VK_TRUE,10000000000ULL); }
    int iters = 10; double t0 = now();
    for (int k = 0; k < iters; k++) { vkResetFences(G.dev,1,&G.fence); vkQueueSubmit(G.queue,1,&si,G.fence); vkWaitForFences(G.dev,1,&G.fence,VK_TRUE,10000000000ULL); }
    return (now()-t0)*1000.0/iters/N;
}

/* FAIR fused-gate_up throughput: cycle K DISTINCT experts (own descriptor set each) so
 * weights come from VRAM, not L2 — matching ROCm's expert_group reading distinct experts.
 * Returns ms per gate_up (one expert). */
static double bench_experts_fair(int fmt, int D, int I, int K, int Npass) {
    if (!G.pipe_gu) return -1;   /* gate_up shader not loaded: run_gate_up already reported it */
    if (K > 32) K = 32;
    size_t rb = ref_rowbytes(fmt, D), nsc = ref_scales(fmt, D, I);
    ColiVkTensor *tg[32] = {0}, *tu[32] = {0};
    float *h = malloc((size_t)I*4), *x = malloc((size_t)D*4);
    for (int i = 0; i < D; i++) x[i] = (rand()%200-100)/100.0f;
    for (int c = 0; c < K; c++) {
        uint8_t *gw = malloc(rb*I), *uw = malloc(rb*I); float *gs = malloc(nsc*4), *us = malloc(nsc*4);
        for (size_t i = 0; i < rb*I; i++) { gw[i] = rand()&0xff; uw[i] = rand()&0xff; }
        for (size_t o = 0; o < nsc; o++) { gs[o] = 0.01f; us[o] = 0.01f; }
        coli_vk_gate_up(&tg[c], &tu[c], h, x, gw, gs, uw, us, fmt, 1, D, I, g_ref_gs);   /* uploads distinct experts */
        free(gw); free(uw); free(gs); free(us);
    }
    VkDescriptorPoolSize ps = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = (uint32_t)(6*K)};
    VkDescriptorPoolCreateInfo dpi = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = (uint32_t)K, .poolSizeCount = 1, .pPoolSizes = &ps};
    VkDescriptorPool pool; vkCreateDescriptorPool(G.dev, &dpi, NULL, &pool);
    VkDescriptorSetLayout lays[32]; VkDescriptorSet sets[32]; for (int c = 0; c < K; c++) lays[c] = G.dsl_gu;
    VkDescriptorSetAllocateInfo dsa = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = pool, .descriptorSetCount = (uint32_t)K, .pSetLayouts = lays};
    vkAllocateDescriptorSets(G.dev, &dsa, sets);
    memcpy(G.x.ptr, x, (size_t)D*4);
    for (int c = 0; c < K; c++) {
        VkDescriptorBufferInfo bi[6] = {{.buffer=G.x.buf,.range=VK_WHOLE_SIZE},{.buffer=tg[c]->wbuf,.range=VK_WHOLE_SIZE},{.buffer=tg[c]->sbuf,.range=VK_WHOLE_SIZE},{.buffer=tu[c]->wbuf,.range=VK_WHOLE_SIZE},{.buffer=tu[c]->sbuf,.range=VK_WHOLE_SIZE},{.buffer=G.h.buf,.range=VK_WHOLE_SIZE}};
        VkWriteDescriptorSet w[6]; for (int i = 0; i < 6; i++) w[i] = (VkWriteDescriptorSet){.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,.dstSet=sets[c],.dstBinding=(uint32_t)i,.descriptorCount=1,.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.pBufferInfo=&bi[i]};
        vkUpdateDescriptorSets(G.dev, 6, w, 0, NULL);
    }
    vkResetCommandBuffer(G.cmd, 0);
    VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkBeginCommandBuffer(G.cmd, &begin);
    vkCmdBindPipeline(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe_gu);
    struct PCGU pc = pcgu(fmt, 1, D, I, tg[0]->rowWords, tg[0]->gs);
    vkCmdPushConstants(G.cmd, G.plyt_gu, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    /* K experts write the same hidden slice in turn: order each write after the
     * previous one (WAW), not only after its reads. */
    VkMemoryBarrier mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                          .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
    for (int pass = 0; pass < Npass; pass++) for (int c = 0; c < K; c++) {
        vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt_gu, 0, 1, &sets[c], 0, NULL);
        vkCmdDispatch(G.cmd, (uint32_t)((I+7)/8), 1, 1);
        vkCmdPipelineBarrier(G.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    }
    vkEndCommandBuffer(G.cmd);
    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &G.cmd};
    for (int w = 0; w < 2; w++) { vkResetFences(G.dev,1,&G.fence); vkQueueSubmit(G.queue,1,&si,G.fence); vkWaitForFences(G.dev,1,&G.fence,VK_TRUE,10000000000ULL); }
    int iters = 10; double t0 = now();
    for (int k = 0; k < iters; k++) { vkResetFences(G.dev,1,&G.fence); vkQueueSubmit(G.queue,1,&si,G.fence); vkWaitForFences(G.dev,1,&G.fence,VK_TRUE,10000000000ULL); }
    double ms = (now()-t0)*1000.0/iters/((double)Npass*K);
    vkDestroyDescriptorPool(G.dev, pool, NULL);
    for (int c = 0; c < K; c++) { coli_vk_tensor_free(tg[c]); coli_vk_tensor_free(tu[c]); }
    free(h); free(x); G.cmd_ready = 0; G.bound_tensor = NULL;
    return ms;
}

/* Full expert-group correctness (vs CPU ref) + fair throughput: K distinct experts,
 * one submit, hidden on-device. The real comparison to ROCm's coli_cuda_expert_group. */
static int run_expert_group(int fmt, int D, int I, int K) {
    if (K > 64) K = 64;
    size_t gu_rb = ref_rowbytes(fmt, D), gu_sc = ref_scales(fmt, D, I);
    size_t d_rb  = ref_rowbytes(fmt, I), d_sc  = ref_scales(fmt, I, D);
    ColiVkTensor *tg[64] = {0}, *tu[64] = {0}, *td[64] = {0};
    uint8_t *hgw[64], *huw[64], *hdw[64]; float *hgs[64], *hus[64], *hds[64];
    float *x = malloc((size_t)K*D*4), *yg = malloc((size_t)K*D*4), *yc = malloc((size_t)K*D*4);
    float *tmp = malloc((size_t)(D > I ? D : I) * 4);
    for (int i = 0; i < K*D; i++) x[i] = (rand()%200-100)/100.0f;
    for (int c = 0; c < K; c++) {
        hgw[c] = malloc(gu_rb*I); huw[c] = malloc(gu_rb*I); hdw[c] = malloc(d_rb*D);
        for (size_t i = 0; i < gu_rb*I; i++) { hgw[c][i] = rand()&0xff; huw[c][i] = rand()&0xff; }
        for (size_t i = 0; i < d_rb*D; i++) hdw[c][i] = rand()&0xff;
        hgs[c] = malloc(gu_sc*4); hus[c] = malloc(gu_sc*4); hds[c] = malloc(d_sc*4);
        for (size_t o = 0; o < gu_sc; o++) { hgs[c][o] = 0.01f+(rand()%100)/10000.0f; hus[c][o] = 0.01f+(rand()%100)/10000.0f; }
        for (size_t o = 0; o < d_sc; o++) hds[c][o] = 0.01f+(rand()%100)/10000.0f;
        coli_vk_matmul(&tg[c], tmp, x, hgw[c], hgs[c], fmt, 1, D, I, g_ref_gs);   /* upload gate  (D->I) */
        coli_vk_matmul(&tu[c], tmp, x, huw[c], hus[c], fmt, 1, D, I, g_ref_gs);   /* upload up    (D->I) */
        coli_vk_matmul(&td[c], tmp, x, hdw[c], hds[c], fmt, 1, I, D, g_ref_gs);   /* upload down  (I->D) */
    }
    int rows[64]; for (int c = 0; c < K; c++) rows[c] = 1;
    if (!coli_vk_expert_group(tg, tu, td, rows, K, yg, x)) { printf("expert_group failed\n"); return 1; }
    digest(yg, (size_t)K * D * 4);
    float *hid = malloc((size_t)I*4);
    for (int c = 0; c < K; c++) {
        float *xc = x + (size_t)c*D;
        for (int o = 0; o < I; o++) {
            float gt = (float)ref_dot(xc, hgw[c]+(size_t)o*gu_rb, hgs[c], o, fmt, D);
            float ut = (float)ref_dot(xc, huw[c]+(size_t)o*gu_rb, hus[c], o, fmt, D);
            hid[o] = (gt/(1.0f+expf(-gt)))*ut;
        }
        for (int d = 0; d < D; d++)
            yc[c*D+d] = (float)ref_dot(hid, hdw[c]+(size_t)d*d_rb, hds[c], d, fmt, I);
    }
    double maxrel = 0;
    for (int i = 0; i < K*D; i++) { double e = fabs(yg[i]-yc[i]); if (fabs(yc[i])>1e-2) { double r = e/fabs(yc[i]); if (r>maxrel) maxrel = r; } }
    coli_vk_expert_group(tg, tu, td, rows, K, yg, x);   /* warm + leaves G.eg_cmd recorded */
    /* async issue/take must reproduce the sync result exactly (same buffers/records) */
    {
        float *ya = malloc((size_t)K*D*4);
        if (!coli_vk_expert_group_issue(tg, tu, td, rows, K, x) || !coli_vk_expert_group_take(ya)) {
            printf("expert_group issue/take failed\n"); maxrel = 1;
        } else {
            double amr = 0;
            for (int i = 0; i < K*D; i++) { double e = fabs(ya[i]-yg[i]); if (fabs(yg[i])>1e-2) { double r = e/fabs(yg[i]); if (r>amr) amr = r; } }
            if (amr > 1e-6) { printf("issue/take deviates from sync: %.4g\n", amr); maxrel = 1; }
        }
        free(ya);
    }
    int iters = 20; double t0 = now();
    for (int k = 0; k < iters; k++) coli_vk_expert_group(tg, tu, td, rows, K, yg, x);
    double ms = (now()-t0)*1000.0/iters/K;
    /* GPU-only: re-submit the already-recorded command buffer (skips per-call host setup:
     * descriptor updates + recording), isolating raw GPU throughput. */
    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &G.eg_cmd};
    for (int w = 0; w < 2; w++) { vkResetFences(G.dev,1,&G.eg_fence); vkQueueSubmit(G.queue,1,&si,G.eg_fence); vkWaitForFences(G.dev,1,&G.eg_fence,VK_TRUE,10000000000ULL); }
    double g0 = now();
    for (int k = 0; k < iters; k++) { vkResetFences(G.dev,1,&G.eg_fence); vkQueueSubmit(G.queue,1,&si,G.eg_fence); vkWaitForFences(G.dev,1,&G.eg_fence,VK_TRUE,10000000000ULL); }
    double gpums = (now()-g0)*1000.0/iters/K;
    printf("FULL VK expert_group fmt=%d %2d experts | maxrel=%.4g | per-call %.4f  GPU-only %.4f ms/expert (ROCm 0.179)\n",
           fmt, K, maxrel, ms, gpums);
    free(hid); free(x); free(yg); free(yc); free(tmp);
    for (int c = 0; c < K; c++) {
        coli_vk_tensor_free(tg[c]); coli_vk_tensor_free(tu[c]); coli_vk_tensor_free(td[c]);
        free(hgw[c]); free(huw[c]); free(hdw[c]); free(hgs[c]); free(hus[c]); free(hds[c]);
    }
    /* The gate_up -> down chain accumulates fp32 rounding across two long reductions
     * (D=6144 then I) vs the double-precision CPU ref; 1-2e-3 shows up at any K
     * depending on the random draw, and is fine for greedy argmax. 3e-3 keeps the
     * gate honest without flagging the known fp behavior as a failure. */
    return maxrel > 3e-3 ? 1 : 0;
}

/* MLA absorb attention vs a CPU ref that mirrors glm.c's absorb loop exactly:
 * qabs = sum_d q[d]*deq(row rbase+d)*ws, scores over cache rows [st0, T-S+s],
 * softmax, weighted latent, value-row projection. */
static int g_absorb_rewind;   /* write every row twice, first garbage: a rewound mirror */
static int run_absorb(int fmt, int S, int H, int Q, int R, int V, int K, int st0, int T, int layer) {
    size_t rb = ref_rowbytes(fmt, K);
    int O = H * (Q + V), ngK = (K + 63) / 64;
    size_t nws = ref_scales(fmt, K, O);
    uint8_t *w = malloc(rb * O); float *ws = malloc(nws * 4);
    float *q = malloc((size_t)S * H * (Q + R) * 4);
    float *L = malloc((size_t)T * K * 4), *Rr = malloc((size_t)T * R * 4);
    float *cg = malloc((size_t)S * H * V * 4), *cc = malloc((size_t)S * H * V * 4);
    for (size_t i = 0; i < rb * (size_t)O; i++) w[i] = rand() & 0xff;
    for (size_t o = 0; o < nws; o++) ws[o] = 0.01f + (rand() % 100) / 10000.0f;
    for (int i = 0; i < S * H * (Q + R); i++) q[i] = (rand() % 200 - 100) / 100.0f;
    for (int i = 0; i < T * K; i++) L[i] = (rand() % 200 - 100) / 100.0f;
    for (int i = 0; i < T * R; i++) Rr[i] = (rand() % 200 - 100) / 100.0f;
    float scale = 0.13f;
    if (!coli_vk_kv_ensure(layer, T, K, R)) { printf("kv_ensure failed\n"); return 1; }
    if (g_absorb_rewind) {   /* garbage rows, then the real ones from the middle and from 0 */
        for (int t = 0; t < T; t++)
            if (!coli_vk_kv_row(layer, t, Rr, L)) { printf("kv_row failed\n"); return 1; }
        for (int t = T / 2; t < T; t++)
            if (!coli_vk_kv_row(layer, t, L + (size_t)t * K, Rr + (size_t)t * R)) { printf("kv_row failed\n"); return 1; }
    }
    for (int t = 0; t < T; t++)
        if (!coli_vk_kv_row(layer, t, L + (size_t)t * K, Rr + (size_t)t * R)) { printf("kv_row failed\n"); return 1; }
    ColiVkTensor *kvb = NULL;
    if (!coli_vk_attention_absorb(&kvb, w, ws, fmt, g_ref_gs, cg, q, layer, S, H, Q, R, V, K, st0, T, scale)) {
        printf("absorb failed\n"); return 1; }
    digest(cg, (size_t)S * H * V * 4);
    float *qabs = malloc((size_t)K * 4), *clat = malloc((size_t)K * 4), *sc = malloc((size_t)(T - st0) * 4);
    for (int s = 0; s < S; s++) for (int h = 0; h < H; h++) {
        const float *qp = q + ((size_t)s * H + h) * (Q + R), *qr = qp + Q;
        int rbase = h * (Q + V), nt = (T - S + s + 1) - st0;
        for (int i = 0; i < K; i++) qabs[i] = 0;
        for (int d = 0; d < Q; d++) { const uint8_t *row = w + (size_t)(rbase + d) * rb;
            for (int i = 0; i < K; i++) {
                float sw = fmt == 5 ? ws[(size_t)(rbase + d) * ngK + (i >> 6)]
                         : fmt == 4 ? ws[(size_t)(rbase + d) * ((K + g_ref_gs - 1) / g_ref_gs) + i / g_ref_gs]
                         : ws[rbase + d];
                qabs[i] += qp[d] * deq(row, fmt, i) * sw; } }
        for (int j = 0; j < nt; j++) { int t = st0 + j;
            double a = 0;
            for (int i = 0; i < K; i++) a += qabs[i] * L[(size_t)t * K + i];
            for (int d = 0; d < R; d++) a += qr[d] * Rr[(size_t)t * R + d];
            sc[j] = (float)(a * scale); }
        float mx = sc[0]; for (int j = 1; j < nt; j++) if (sc[j] > mx) mx = sc[j];
        double sum = 0; for (int j = 0; j < nt; j++) { sc[j] = expf(sc[j] - mx); sum += sc[j]; }
        for (int i = 0; i < K; i++) clat[i] = 0;
        for (int j = 0; j < nt; j++) { float a = (float)(sc[j] / sum);
            for (int i = 0; i < K; i++) clat[i] += a * L[(size_t)(st0 + j) * K + i]; }
        for (int v = 0; v < V; v++)
            cc[((size_t)s * H + h) * V + v] =
                (float)ref_dot(clat, w + (size_t)(rbase + Q + v) * rb, ws, rbase + Q + v, fmt, K);
    }
    double maxrel = 0, maxerr = 0;
    for (int i = 0; i < S * H * V; i++) { double e = fabs(cg[i] - cc[i]); if (e > maxerr) maxerr = e;
        if (fabs(cc[i]) > 1e-2) { double r = e / fabs(cc[i]); if (r > maxrel) maxrel = r; } }
    double t0 = now(); int iters = 20;   /* per-call cost, the engine pattern (one submit/layer) */
    for (int k = 0; k < iters; k++)
        coli_vk_attention_absorb(&kvb, w, ws, fmt, g_ref_gs, cg, q, layer, S, H, Q, R, V, K, st0, T, scale);
    double ms = (now() - t0) * 1000 / iters;
    printf("absorb fmt=%d S=%d H=%d Q=%d R=%d V=%d K=%d st0=%d T=%d | maxerr=%.4g maxrel=%.4g | %.3f ms/call\n",
           fmt, S, H, Q, R, V, K, st0, T, maxerr, maxrel, ms);
    /* fused absorb+project vs (CPU ctx ref) @ (CPU o ref) */
    int Dout = 512; size_t orb = ref_rowbytes(fmt, H * V), onsc = ref_scales(fmt, H * V, Dout);
    uint8_t *owt = malloc(orb * Dout); float *osc = malloc(onsc * 4);
    float *og = malloc((size_t)S * Dout * 4), *oc = malloc((size_t)S * Dout * 4);
    for (size_t i = 0; i < orb * (size_t)Dout; i++) owt[i] = rand() & 0xff;
    for (size_t o = 0; o < onsc; o++) osc[o] = 0.01f + (rand() % 100) / 10000.0f;
    ColiVkTensor *ot = NULL;
    if (!coli_vk_attention_absorb_project(&kvb, w, ws, fmt, g_ref_gs, &ot, owt, osc, fmt, g_ref_gs,
            og, q, layer, S, H, Q, R, V, K, st0, T, scale, Dout)) { printf("absorb_project failed\n"); return 1; }
    digest(og, (size_t)S * Dout * 4);
    cpu_ref(oc, cc, owt, osc, fmt, S, H * V, Dout);
    double pmaxrel = 0;
    for (int i = 0; i < S * Dout; i++) { double e = fabs(og[i] - oc[i]);
        if (fabs(oc[i]) > 1e-2) { double r = e / fabs(oc[i]); if (r > pmaxrel) pmaxrel = r; } }
    t0 = now();
    for (int k = 0; k < iters; k++)
        coli_vk_attention_absorb_project(&kvb, w, ws, fmt, g_ref_gs, &ot, owt, osc, fmt, g_ref_gs,
            og, q, layer, S, H, Q, R, V, K, st0, T, scale, Dout);
    printf("absorb+project fused                  | maxrel=%.4g | %.3f ms/call (unfused absorb was %.3f)\n",
           pmaxrel, (now() - t0) * 1000 / iters, ms);
    coli_vk_tensor_free(ot); free(owt); free(osc); free(og); free(oc);
    coli_vk_tensor_free(kvb);
    free(w); free(ws); free(q); free(L); free(Rr); free(cg); free(cc); free(qabs); free(clat); free(sc);
    return (maxrel > 2e-3 || pmaxrel > 5e-3) ? 1 : 0;
}


/* q-prep chain vs CPU ref: matmul(qa) -> rmsnorm -> matmul(qb), kv_a alongside. */
static int run_qprep(int fmt, int S, int I, int Oqa, int Okva, int Oqb) {
    size_t rba = ref_rowbytes(fmt, I), rbb = ref_rowbytes(fmt, Oqa);
    size_t nsa = ref_scales(fmt, I, Oqa), nsk = ref_scales(fmt, I, Okva), nsb = ref_scales(fmt, Oqa, Oqb);
    uint8_t *wa = malloc(rba * Oqa), *wk = malloc(rba * Okva), *wb = malloc(rbb * Oqb);
    float *sa = malloc(nsa * 4), *sk = malloc(nsk * 4), *sb = malloc(nsb * 4);
    float *ln = malloc((size_t)Oqa * 4), *x = malloc((size_t)S * I * 4);
    float *qg = malloc((size_t)S * Oqb * 4), *kvg = malloc((size_t)S * Okva * 4);
    float *lat = malloc((size_t)S * Oqa * 4), *qc = malloc((size_t)S * Oqb * 4), *kvc = malloc((size_t)S * Okva * 4);
    for (size_t i = 0; i < rba * (size_t)Oqa; i++) wa[i] = rand() & 0xff;
    for (size_t i = 0; i < rba * (size_t)Okva; i++) wk[i] = rand() & 0xff;
    for (size_t i = 0; i < rbb * (size_t)Oqb; i++) wb[i] = rand() & 0xff;
    for (size_t o = 0; o < nsa; o++) sa[o] = 0.01f + (rand() % 100) / 10000.0f;
    for (size_t o = 0; o < nsk; o++) sk[o] = 0.01f + (rand() % 100) / 10000.0f;
    for (size_t o = 0; o < nsb; o++) sb[o] = 0.01f + (rand() % 100) / 10000.0f;
    for (int i = 0; i < Oqa; i++) ln[i] = 0.5f + (rand() % 100) / 100.0f;
    for (int i = 0; i < S * I; i++) x[i] = (rand() % 2000 - 1000) / 500.0f;
    ColiVkTensor *ta = NULL, *tk = NULL, *tb = NULL;
    static int qp_layer = 140;         /* distinct high slot per case: the per-layer ln
                                        * cache is upload-once by design (engine weights
                                        * are immutable); reuse here would mix cases */
    int layer = qp_layer++;
    if (!coli_vk_attn_qprep(layer, &ta, wa, sa, Oqa, &tk, wk, sk, Okva, &tb, wb, sb, Oqb,
                            fmt, g_ref_gs, ln, 1e-6f, x, S, I, qg, kvg, NULL)) {
        printf("qprep unavailable (rmsnorm.spv missing?)\n"); return 1; }
    digest(qg, (size_t)S * Oqb * 4); digest(kvg, (size_t)S * Okva * 4);
    cpu_ref(lat, x, wa, sa, fmt, S, I, Oqa);
    cpu_ref(kvc, x, wk, sk, fmt, S, I, Okva);
    for (int s = 0; s < S; s++) {                       /* rmsnorm rows like colibri.c */
        double ms = 0; float *r = lat + (size_t)s * Oqa;
        for (int i = 0; i < Oqa; i++) ms += (double)r[i] * r[i];
        float rr = 1.0f / sqrtf((float)(ms / Oqa) + 1e-6f);
        for (int i = 0; i < Oqa; i++) r[i] = r[i] * rr * ln[i];
    }
    cpu_ref(qc, lat, wb, sb, fmt, S, Oqa, Oqb);
    float mq = 0, mk = 0;
    for (int i = 0; i < S * Oqb; i++) { float d = fabsf(qg[i] - qc[i]) / (fabsf(qc[i]) + 1e-3f); if (d > mq) mq = d; }
    for (int i = 0; i < S * Okva; i++) { float d = fabsf(kvg[i] - kvc[i]) / (fabsf(kvc[i]) + 1e-3f); if (d > mk) mk = d; }
    printf("qprep fmt=%d S=%d I=%d (%d->%d, kv %d) | maxrel q %.4g kv %.4g\n", fmt, S, I, Oqa, Oqb, Okva, mq, mk);
    coli_vk_tensor_free(ta); coli_vk_tensor_free(tk); coli_vk_tensor_free(tb);
    free(wa); free(wk); free(wb); free(sa); free(sk); free(sb); free(ln); free(x);
    free(qg); free(kvg); free(lat); free(qc); free(kvc);
    /* q crosses TWO quantized reductions + the norm; random +-8-nibble rows are
     * cancellation-heavy, so fp32-vs-f64 divergence amplifies ~10x vs one GEMV
     * (same reasoning as the expert_group 3e-3 threshold). Engine-level logit
     * comparison on real weights is the tight check. */
    return mq > 1e-2f || mk > 1e-3f;
}

/* f16 weights (fmt 14) and rows imported in place (coli_vk_tensor_import), after the
 * digest above, which stays the one of the formats before them (tests/vulkan_engines.sh
 * staged compares it across builds), with a digest line of their own. The imports: an
 * int8, an f32 and an f16 matrix in page-aligned host memory, through the GEMV and the
 * fp32 GEMM, against the reference and bit for bit against the same matrix uploaded as
 * a copy; a device without the extension (or staged uploads) says so and skips them. */
static int run_import_case(int fmt, int S, int I, int O) {
    size_t al = coli_vk_import_alignment();
    if (!al) { printf("import fmt=%d: not possible on this device, skipped\n", fmt); return 0; }
    float *x, *sc; uint8_t *w0;
    case_fill(fmt, S, I, O, &x, &w0, &sc);
    size_t rb = ref_rowbytes(fmt, I), bytes = rb * O, sz = (bytes + al - 1) / al * al;
    void *w = NULL;
    if (posix_memalign(&w, al, sz)) { printf("import: out of memory\n"); return 1; }
    memset(w, 0, sz); memcpy(w, w0, bytes);
    float *yi = malloc((size_t)S * O * 4), *yc = malloc((size_t)S * O * 4), *yr = malloc((size_t)S * O * 4);
    ColiVkTensor *ti = NULL, *tc = NULL;
    int bad = 0, keep_s = G.gemm_min_s, keep_so = G.gemm_min_so;
    if (S > 1) { G.gemm_min_s = 2; G.gemm_min_so = 1; }   /* a block of rows: the fp32 GEMM */
    size_t before = coli_vk_imported_bytes();
    if (!coli_vk_tensor_import(&ti, w, sz, sc, fmt, I, O, 0)) { printf("import fmt=%d: refused\n", fmt); bad = 1; }
    else if (coli_vk_imported_bytes() != before + bytes) { printf("import fmt=%d: the bytes were not counted\n", fmt); bad = 1; }
    else if (!coli_vk_matmul(&ti, yi, x, w, sc, fmt, S, I, O, 0) || !coli_vk_matmul(&tc, yc, x, w0, sc, fmt, S, I, O, 0)) {
        printf("import fmt=%d: matmul failed\n", fmt); bad = 1;
    } else {
        digest(yi, (size_t)S * O * 4);
        cpu_ref(yr, x, w0, sc, fmt, S, I, O);
        double maxrel = 0;
        for (int i = 0; i < S * O; i++)
            if (fabs(yr[i]) > 1e-2) { double r = fabs(yi[i] - yr[i]) / fabs(yr[i]); if (r > maxrel) maxrel = r; }
        int same = !memcmp(yi, yc, (size_t)S * O * 4);
        printf("import fmt=%d S=%d I=%d O=%d %s | maxrel=%.4g | the copy's bits: %s\n", fmt, S, I, O,
               G.bound_gemm ? "gemm" : "gemv", maxrel, same ? "same" : "DIFFERENT");
        bad = maxrel > 1e-3 || !same;
    }
    G.gemm_min_s = keep_s; G.gemm_min_so = keep_so;
    coli_vk_tensor_free(ti); coli_vk_tensor_free(tc);
    if (coli_vk_imported_bytes() != before) { printf("import fmt=%d: the bytes were not given back\n", fmt); bad = 1; }
    free(w); free(w0); free(x); free(sc); free(yi); free(yc); free(yr);
    return bad;
}
static int run_gemm_case(int fmt, int S, int I, int O, int gs);
/* COLI_VK_TEST_IMPORT_BENCH=1: one Qwen3.8-27B MLP matrix (I 5120, O 17408) in int8 and
 * f16, as a copy in the weight pool and imported in place, at S = 1 (the GEMV) and S =
 * 300 (a decision's prompt, the tiled GEMM): the wall time of a call, best of 5. */
static void bench_import(void) {
    size_t al = coli_vk_import_alignment();
    if (!al) { printf("import bench: imports not possible on this device\n"); return; }
    const int I = 5120, O = 17408, Ss[2] = {1, 300}, fm[2] = {1, 14};
    for (int f = 0; f < 2; f++) {
        int fmt = fm[f];
        float *x, *sc; uint8_t *w0;
        case_fill(fmt, 300, I, O, &x, &w0, &sc);
        size_t bytes = ref_rowbytes(fmt, I) * O, sz = (bytes + al - 1) / al * al;
        void *w = NULL;
        if (posix_memalign(&w, al, sz)) return;
        memcpy(w, w0, bytes);
        float *y = malloc((size_t)300 * O * 4);
        ColiVkTensor *tc = NULL, *ti = NULL;
        if (!coli_vk_tensor_ensure(&tc, w0, sc, fmt, I, O, 0) || !coli_vk_tensor_import(&ti, w, sz, sc, fmt, I, O, 0)) {
            printf("import bench fmt=%d: no tensor\n", fmt); return;
        }
        for (int k = 0; k < 2; k++) {
            double best[2] = {1e30, 1e30};
            int keep_s = G.gemm_min_s, keep_so = G.gemm_min_so;
            if (Ss[k] > 1) { G.gemm_min_s = 2; G.gemm_min_so = 1; }   /* the harness keeps GEMVs; a prompt takes the GEMM */
            for (int which = 0; which < 2; which++) {
                ColiVkTensor **t = which ? &ti : &tc;
                coli_vk_matmul(t, y, x, which ? w : w0, sc, fmt, Ss[k], I, O, 0);
                for (int r = 0; r < 5; r++) {
                    double t0 = now();
                    coli_vk_matmul(t, y, x, which ? w : w0, sc, fmt, Ss[k], I, O, 0);
                    double ms = (now() - t0) * 1000;
                    if (ms < best[which]) best[which] = ms;
                }
            }
            printf("import bench fmt=%d S=%d (%s): copy %.2f ms, imported %.2f ms (%.1f MB of rows)\n", fmt, Ss[k],
                   G.bound_gemm == 2 ? "coop" : G.bound_gemm ? "gemm" : "gemv", best[0], best[1], bytes / 1e6);
            G.gemm_min_s = keep_s; G.gemm_min_so = keep_so;
        }
        coli_vk_tensor_free(tc); coli_vk_tensor_free(ti);
        free(w); free(w0); free(x); free(sc); free(y);
    }
}
static int run_late_cases(void) {
    if (getenv("COLI_VK_TEST_IMPORT_BENCH") && atoi(getenv("COLI_VK_TEST_IMPORT_BENCH"))) bench_import();
    uint64_t keep = g_digest;
    g_digest = 1469598103934665603ULL;
    int bad = 0;
    bad |= run_case(14, 1, 6144, 1536, 20);
    bad |= run_case(14, 8, 2048, 512, 10);
    bad |= run_case(14, 1, 16384, 512, 5);
    bad |= run_case(14, 3, 37, 5, 5);       // odd I: the last word holds one f16
    bad |= run_gemm_case(14, 16, 1001, 193, 64);
    bad |= run_gemm_case(14, 64, 1536, 97, 64);
    bad |= run_gemm_case(14, 512, 520, 131, 64);
    bad |= run_import_case(1, 1, 2048, 384);
    bad |= run_import_case(1, 64, 2048, 384);
    bad |= run_import_case(10, 3, 1024, 100);
    bad |= run_import_case(14, 1, 4096, 257);
    bad |= run_import_case(14, 96, 4096, 257);
    printf("f16 and import digest %016llx\n", (unsigned long long)g_digest);
    g_digest = keep;
    return bad;
}

/* ---- the expert tier's device side: weight pool and async expert batch ----------
 * Reference dot of one weight row in any format, its scales applied the way the
 * shader does (per row for 1/2, per gs-group for 4/7/12/13, per 64 for 5, none for
 * 10/11), in double. */
static double xref_dot(const float *x, const uint8_t *row, const float *sc, int o, int fmt, int I, int gs) {
    if (fmt == 10 || fmt == 11) { double a = 0; for (int i = 0; i < I; i++) a += x[i] * deq(row, fmt, i); return a; }
    if (fmt == 1 || fmt == 2) { double a = 0; for (int i = 0; i < I; i++) a += x[i] * deq(row, fmt, i); return a * sc[o]; }
    int g = fmt == 5 ? 64 : gs, ng = (I + g - 1) / g; double sum = 0;
    for (int k = 0; k < ng; k++) {
        double a = 0; int end = (k + 1) * g < I ? (k + 1) * g : I;
        for (int i = k * g; i < end; i++) a += x[i] * deq(row, fmt, i);
        sum += a * sc[(size_t)o * ng + k];
    }
    return sum;
}
static float xref_act(float g, float u, int act, float limit, float a, float b) {
    if (act == COLI_VK_ACT_SITU) return a * tanhf(g / a) * (1.0f / (1.0f + expf(-g))) * (b * tanhf(u / b));
    if (limit > 0) { if (g > limit) g = limit; if (u > limit) u = limit; if (u < -limit) u = -limit; }
    return (g / (1.0f + expf(-g))) * u;
}
/* One weight matrix of the test: random codes (no NaN for fp8, finite floats for
 * 10/11) and scales, kept on the host for the reference and copied in place into a
 * tier tensor. */
typedef struct { uint8_t *w; float *s; int fmt, I, O, gs; ColiVkTensor *t; } XMat;
static int xmat_make(XMat *m, int fmt, int I, int O, int gs) {
    memset(m, 0, sizeof(*m));
    m->fmt = fmt; m->I = I; m->O = O; m->gs = gs;
    int keep = g_ref_gs; g_ref_gs = gs;
    float *x; case_fill(fmt, 1, I, O, &x, &m->w, &m->s); free(x);
    g_ref_gs = keep;
    uint8_t *rows; size_t stride; float *scales;
    if (!coli_vk_tier_tensor(&m->t, fmt, I, O, gs, &rows, &stride, &scales)) return 0;
    size_t rb = coli_vk_tensor_row_bytes(fmt, I), ns = coli_vk_tensor_scale_count(fmt, I, O, gs);
    for (int o = 0; o < O; o++) memcpy(rows + (size_t)o * stride, m->w + (size_t)o * rb, rb);
    if (fmt == 10 || fmt == 11) scales[0] = 1.0f; else memcpy(scales, m->s, ns * sizeof(float));
    return coli_vk_tensor_commit(&m->t, 1);
}
static void xmat_drop(XMat *m) { free(m->w); free(m->s); m->w = NULL; m->s = NULL; }

/* K experts in format fmt (gate/up) and dfmt (down) through coli_vk_xb_issue/join,
 * against the CPU in double: rows per expert from 1 to past the GEMM threshold, so
 * both routes run in one batch. Then the GEMV route's row independence (a row alone
 * and the same row among others give the same bits) and a second batch on the same
 * experts. */
static int run_xbatch(int fmt, int dfmt, int gs, int D, int I, int act, float limit) {
    enum { K = 5 };
    const char *large = getenv("COLI_VK_TEST_XB_LARGE");
    int rows[K] = {1, 3, 2, 20, 1};
    if (large && atoi(large)) { rows[2] = 64; rows[3] = 257; rows[4] = 20; }
    XMat mg[K], mu[K], md[K]; ColiVkExpert *ex[K];
    g_xb[0].act = act; g_xb[0].limit = limit; g_xb[0].a = 4.0f; g_xb[0].b = 25.0f;
    for (int c = 0; c < K; c++) {
        if (!xmat_make(&mg[c], fmt, D, I, gs) || !xmat_make(&mu[c], fmt, D, I, gs) ||
            !xmat_make(&md[c], dfmt, I, D, gs) || !(ex[c] = coli_vk_xb_expert(mg[c].t, mu[c].t, md[c].t))) {
            printf("xbatch fmt=%d/%d: expert %d could not be made resident\n", fmt, dfmt, c); return 1;
        }
    }
    int total = 0; for (int c = 0; c < K; c++) total += rows[c];
    float *x = malloc((size_t)total * D * 4), *ref = malloc((size_t)total * D * 4), *hid = malloc((size_t)I * 4);
    const float **xr = malloc((size_t)total * sizeof(*xr)), **yr = malloc((size_t)total * sizeof(*yr));
    for (int i = 0; i < total * D; i++) x[i] = (rand() % 200 - 100) / 100.0f;
    for (int j = 0; j < total; j++) xr[j] = x + (size_t)j * D;
    int j = 0;
    for (int c = 0; c < K; c++)
        for (int r = 0; r < rows[c]; r++, j++) {
            if (large && atoi(large) && c >= 1) {
                /* Device-side scaling must preserve zero and tiny rows as well
                 * as values that would overflow an unscaled fp16 conversion. */
                static const int powers[] = {-40, 0, 10, 20};
                for (int d = 0; d < D; d++)
                    x[(size_t)j * D + d] = r ? ldexpf(x[(size_t)j * D + d], powers[(r + 3) % 4]) : 0;
            }
            const float *xs = x + (size_t)j * D;
            size_t grb = ref_rowbytes(fmt, D), drb = ref_rowbytes(dfmt, I);
            for (int o = 0; o < I; o++) {
                float g = (float)xref_dot(xs, mg[c].w + (size_t)o * grb, mg[c].s, o, fmt, D, gs);
                float u = (float)xref_dot(xs, mu[c].w + (size_t)o * grb, mu[c].s, o, fmt, D, gs);
                hid[o] = xref_act(g, u, act, limit, g_xb[0].a, g_xb[0].b);
            }
            for (int d = 0; d < D; d++) ref[(size_t)j * D + d] = (float)xref_dot(hid, md[c].w + (size_t)d * drb, md[c].s, d, dfmt, I, gs);
        }
    int bad = 0; double dms = 0;
    unsigned long long coop_before = g_xb[0].cooperative_matmuls, coop_expected = 0;
    for (int c = 0; c < K; c++) {
        int slot = gemm_slot(rows[c]);
        if (g_xb[0].gemm_rows && rows[c] >= g_xb[0].gemm_rows && g_xb[0].p_mm[slot])
            coop_expected += (xb_coop_slot(&g_xb[0], mg[c].t, rows[c], slot) >= 0) +
                             (xb_coop_slot(&g_xb[0], mu[c].t, rows[c], slot) >= 0) +
                             (xb_coop_slot(&g_xb[0], md[c].t, rows[c], slot) >= 0);
    }
    if (!coli_vk_xb_issue(ex, rows, K, xr) || !coli_vk_xb_join(yr, &dms)) { printf("xbatch fmt=%d/%d: issue/join failed\n", fmt, dfmt); bad = 1; }
    unsigned long long coop_done = g_xb[0].cooperative_matmuls - coop_before;
    if (coop_done != coop_expected) { printf("xbatch: cooperative matmuls %llu, expected %llu\n", coop_done, coop_expected); bad = 1; }
    for (int jj = 0; !bad && jj < total; jj++) digest(yr[jj], (size_t)D * 4);
    double maxrel = 0;
    for (int jj = 0; !bad && jj < total; jj++) {
        double scale = 0;
        for (int d = 0; d < D; d++) if (fabs(ref[(size_t)jj * D + d]) > scale) scale = fabs(ref[(size_t)jj * D + d]);
        for (int d = 0; d < D; d++) {
            double r = ref[(size_t)jj * D + d], e = fabs(yr[jj][d] - r);
            if (!isfinite(yr[jj][d])) { bad = 1; break; }
            if (scale == 0) { if (yr[jj][d] != 0) bad = 1; continue; }
            /* relative to the row's own magnitude where it is not tiny; cancellation-heavy
             * random rows make single elements near zero meaningless */
            double rel = e / (fabs(r) > 1e-2 * scale ? fabs(r) : 1e-2 * scale);
            if (rel > maxrel) maxrel = rel;
        }
    }
    /* the first expert's row alone, then the same row as the 2nd of 3: same bits */
    float *y1 = malloc((size_t)D * 4);
    int same = 1;
    if (!bad) {
        int one = 1;
        memcpy(y1, yr[0], (size_t)D * 4);
        if (!coli_vk_xb_issue(ex, &one, 1, xr) || !coli_vk_xb_join(yr, NULL)) bad = 1;
        else same = !memcmp(y1, yr[0], (size_t)D * 4);
        const float *xx[3] = {xr[2], xr[0], xr[3]}; int three = 3;
        if (!bad && (!coli_vk_xb_issue(ex, &three, 1, xx) || !coli_vk_xb_join(yr, NULL))) bad = 1;
        else if (!bad) same &= !memcmp(y1, yr[1], (size_t)D * 4);
    }
    const char *an = act == COLI_VK_ACT_SITU ? "situ" : limit > 0 ? "swiglu-limit" : "swiglu";
    printf("xbatch fmt=%d/%d gs=%d D=%d I=%d %s, rows %d/%d/%d/%d/%d (GEMM from %d, %llu cooperative matmuls) | maxrel=%.3g | row bits independent of the batch: %s | device %.3f ms\n",
           fmt, dfmt, gs, D, I, an, rows[0], rows[1], rows[2], rows[3], rows[4], g_xb[0].gemm_rows,
           coop_done, maxrel, same ? "yes" : "NO", dms);
    if (maxrel > 3e-3 || !same) bad = 1;
    for (int c = 0; c < K; c++) { coli_vk_xb_expert_free(ex[c]); xmat_drop(&mg[c]); xmat_drop(&mu[c]); xmat_drop(&md[c]); }
    free(x); free(ref); free(hid); free(xr); free(yr); free(y1);
    return bad;
}

/* The tier pool on the device: fill to the budget, free, fill again into the same
 * memory; a free while a batch is in flight waits for the join. */
static int run_tier_pool(void) {
    int bad = 0, D = g_xb[0].D, I = g_xb[0].I;
    size_t one = 0;
    ColiVkPoolStats st;
    coli_vk_pool_stats(1, &st);
    if (st.live) { printf("tier pool: not empty at the start of its test\n"); return 1; }
    {   /* the bytes one expert takes in a block, alignment gaps included: two experts in
         * a fresh pool, the distance between their first ranges */
        coli_vk_tier_pool_limit(0);
        pool_destroy(&g_tpool);
        ColiVkTensor *m[6]; void *w, *sc;
        for (int k = 0; k < 6; k++) m[k] = tensor_alloc(&g_tpool, 4, k % 3 == 2 ? I : D, k % 3 == 2 ? D : I, 64, &w, &sc);
        if (m[0] && m[3] && m[0]->wr.block == m[3]->wr.block) one = (size_t)(m[3]->wr.off - m[0]->wr.off);
        for (int k = 0; k < 6; k++) if (m[k]) coli_vk_tensor_free(m[k]);
        pool_destroy(&g_tpool);
        if (!one) { printf("tier pool: could not measure an expert's footprint\n"); return 1; }
    }
    coli_vk_tier_pool_limit(10 * one + one / 2);   /* room for 10 experts, one block of that size */
    enum { N = 16 };
    XMat mg[N], mu[N], md[N]; ColiVkExpert *ex[N]; int n = 0;
    memset(mg, 0, sizeof mg); memset(mu, 0, sizeof mu); memset(md, 0, sizeof md);
    while (n < N && xmat_make(&mg[n], 4, D, I, 64) && xmat_make(&mu[n], 4, D, I, 64) && xmat_make(&md[n], 4, I, D, 64)) {
        if (!(ex[n] = coli_vk_xb_expert(mg[n].t, mu[n].t, md[n].t))) { printf("tier pool: expert %d has no sets\n", n); return 1; }
        n++;
    }
    coli_vk_pool_stats(1, &st);
    int refused_first = st.refusals > 0;
    /* the partial expert that hit the budget: give back what it got */
    if (n < N) {
        if (mg[n].t) coli_vk_tensor_free(mg[n].t);
        if (mu[n].t) coli_vk_tensor_free(mu[n].t);
        if (md[n].t) coli_vk_tensor_free(md[n].t);
        xmat_drop(&mg[n]); xmat_drop(&mu[n]); xmat_drop(&md[n]);
    }
    printf("tier pool: %d experts fit a budget of 10.5 (%zu bytes each), refusals %llu, %d block(s), frag %.3f\n",
           n, one, st.refusals, st.blocks, st.frag);
    if (n != 10 || !refused_first || st.blocks != 1) { printf("  ^ the budget did not bound the pool\n"); bad = 1; }
    if (n < 4) {   /* the rest needs four experts */
        for (int k = 0; k < n; k++) { coli_vk_xb_expert_free(ex[k]); xmat_drop(&mg[k]); xmat_drop(&mu[k]); xmat_drop(&md[k]); }
        coli_vk_tier_pool_limit(0);
        return 1;
    }
    /* a batch in flight while three experts are freed: their memory stays until the join */
    float *x = malloc((size_t)D * 4); for (int i = 0; i < D; i++) x[i] = 0.01f * (i % 17);
    const float *xr[1] = {x}, *yr[1]; int r1 = 1;
    ColiVkPoolStats before, during, after;
    coli_vk_pool_stats(1, &before);
    int issued = coli_vk_xb_issue(&ex[0], &r1, 1, xr);
    for (int k = 0; k < 3; k++) {   /* not ex[0]: it is in the batch; the tensors only, the sets stay */
        coli_vk_tensor_free(mg[n - 1 - k].t); coli_vk_tensor_free(mu[n - 1 - k].t); coli_vk_tensor_free(md[n - 1 - k].t);
    }
    coli_vk_pool_stats(1, &during);
    int joined = issued && coli_vk_xb_join(yr, NULL);
    coli_vk_pool_stats(1, &after);
    printf("tier pool: frees during a batch held %zu -> %zu bytes, after the join %zu\n",
           (size_t)before.used, (size_t)during.used, (size_t)after.used);
    if (!issued || !joined || during.used != before.used || after.used >= before.used) {
        printf("  ^ a free during a batch was not deferred to its join\n"); bad = 1;
    }
    /* the three freed experts' room takes three new ones */
    int again = 0;
    for (int k = 0; k < 3; k++) {
        ColiVkTensor *g = NULL, *u = NULL, *d = NULL;
        ColiVkExpert *e = ex[n - 1 - k];   /* its tensors went above: drop its sets */
        xb_expert_unlink(e); free(e); ex[n - 1 - k] = NULL;
        xmat_drop(&mg[n - 1 - k]); xmat_drop(&mu[n - 1 - k]); xmat_drop(&md[n - 1 - k]);
        XMat a, b, c;
        if (xmat_make(&a, 4, D, I, 64) && xmat_make(&b, 4, D, I, 64) && xmat_make(&c, 4, I, D, 64)) {
            g = a.t; u = b.t; d = c.t;
            if ((ex[n - 1 - k] = coli_vk_xb_expert(g, u, d))) again++;
        }
        xmat_drop(&a); xmat_drop(&b); xmat_drop(&c);
    }
    coli_vk_pool_stats(1, &st);
    printf("tier pool: %d of 3 experts uploaded again into the freed room, %d blocks, %zu of %zu bytes used\n",
           again, st.blocks, (size_t)st.used, (size_t)st.total);
    if (again != 3) { printf("  ^ freed room was not reused\n"); bad = 1; }
    for (int k = 0; k < n; k++) { coli_vk_xb_expert_free(ex[k]); xmat_drop(&mg[k]); xmat_drop(&mu[k]); xmat_drop(&md[k]); }
    coli_vk_pool_stats(1, &st);
    printf("tier pool: all freed, %zu bytes used, %d live ranges, %d block(s) kept\n", (size_t)st.used, st.live, st.blocks);
    if (st.used || st.live) bad = 1;
    coli_vk_tier_pool_limit(0);
    free(x);
    return bad;
}

/* COLI_VK_ACT_SWIGLU_V4 bit for bit. Experts whose sums are exact in f32 (f32 weights,
 * one input that is a power of two, the others zero) and whose gate is at least 24, so
 * that its sigmoid is exactly 1 on both sides: the device and the reference below then
 * compute the same numbers through every rounding the activation makes -- gate and up
 * to bf16, their product, the route weight and bf16 again, each block's power-of-two
 * E4M3 scale (amax exactly 448 x 2^k among them) and every E4M3 rounding (ties to
 * even, subnormal codes and their ties, values under half the smallest, the 448
 * clamp), a 32-input tail block. Down is the identity, so a row of y IS the hidden
 * row after the pass. Rows 1 and 3 take the GEMV route, 20 the GEMM. The reference is
 * DeepSeek V4's CPU arithmetic (coli_bf16_round, coli_fp8_activation_qdq_ref), written
 * out again here. */
static float v4r_bf16(float v) {
    uint32_t b; memcpy(&b, &v, 4);
    if ((b & 0x7f800000u) != 0x7f800000u) b += 0x7fffu + ((b >> 16) & 1u);
    b &= 0xffff0000u; memcpy(&v, &b, 4);
    return v;
}
static float v4r_e4m3(float value) {   /* encode (nearest even) then decode */
    int neg = signbit(value) != 0;
    float a = fabsf(value), r;
    if (!a) return value;
    if (a >= 448.0f) r = 448.0f;
    else if (a < 0.015625f) {
        float sc = a * 512.0f; unsigned q = (unsigned)sc; float fr = sc - (float)q;
        if (fr > 0.5f || (fr == 0.5f && (q & 1))) q++;
        r = ldexpf((float)q, -9);
    } else {
        uint32_t b; memcpy(&b, &a, 4);
        int E = (int)((b >> 23) & 0xff) - 127;
        uint32_t sig = 0x800000u | (b & 0x7fffffu), q = sig >> 20, rem = sig & 0xfffffu;
        if (rem > 0x80000u || (rem == 0x80000u && (q & 1u))) q++;
        r = ldexpf((float)q, E - 3);
    }
    return neg ? -r : r;
}
static void v4r_qdq(float *v, int n) {
    for (int base = 0; base < n; base += 128) {
        int cnt = n - base < 128 ? n - base : 128;
        float mx = 0.0f;
        for (int i = 0; i < cnt; i++) mx = fmaxf(mx, fabsf(v[base + i]));
        mx = fmaxf(mx, 1e-4f);
        int ex; float fr = frexpf(mx / 448.0f, &ex);
        int e = fr == 0.5f ? ex - 1 : ex;
        if (e < -127) e = -127;
        if (e > 127) e = 127;
        float scale = ldexpf(1.0f, e);
        for (int i = 0; i < cnt; i++)
            v[base + i] = v4r_e4m3(fmaxf(-448.0f, fminf(448.0f, v[base + i] / scale))) * scale;
    }
}
static int run_xbatch_v4(void) {
    enum { D = 160, I = 160, K = 3 };   /* one block of 128 and a tail of 32 */
    static const int rows[K] = {1, 3, 20};
    if (!coli_vk_xb_init(D, I, COLI_VK_ACT_SWIGLU_V4, 0.f, 0.f, 0.f)) {
        printf("xbatch v4: no pass (expert_act_v4.spv missing beside the main shader?)\n"); return 1;
    }
    /* special values of the hidden row (x 2^-4: the block's amax 28 makes its scale
     * 2^-4): 448, ties of the normal and the subnormal codes, under half the smallest */
    static const float tval[] = {448.0f, 1.0625f, 1.1875f, -2.125f, 15.5f, -15.5f, 447.0f, 440.0f,
                                 3.5f / 512, 2.5f / 512, -1.5f / 512, 0.4f / 512, 0.75f / 512, 0.0f, 300.0f, -0.5f};
    const int nt = (int)(sizeof tval / sizeof *tval);
    float *gw[K], *uw[K], *dw = calloc((size_t)D * I, sizeof(float));
    ColiVkExpert *ex[K];
    int bad = 0;
    for (int d = 0; d < D; d++) dw[(size_t)d * I + d] = 1.0f;
    for (int c = 0; c < K; c++) {
        gw[c] = calloc((size_t)I * D, sizeof(float)); uw[c] = calloc((size_t)I * D, sizeof(float));
        for (int o = 0; o < I; o++) {
            /* column 0 only: g = 24..63 with few bits, u random -- or, in expert 0's
             * first block, g = 32 and u = t / 512 (h = t / 16) for the special values
             * and the rest under them, so the block's amax is 28 */
            float g = 24.0f + (float)(rand() % 40), u = ((float)rand() / RAND_MAX - 0.5f) * ldexpf(1.0f, rand() % 12 - 8);
            if (c == 0 && o < 128) u = ((float)rand() / RAND_MAX - 0.5f) * ldexpf(1.0f, -(rand() % 12) - 6);
            if (c == 0 && o < nt) { g = 32.0f; u = tval[o] / 512.0f; }
            gw[c][(size_t)o * D] = g; uw[c][(size_t)o * D] = u;
        }
        ColiVkTensor *t[3]; const float *src[3] = {gw[c], uw[c], dw};
        for (int k = 0; k < 3; k++) {
            uint8_t *rws; size_t stride; float *sc;
            if (!coli_vk_tier_tensor(&t[k], 10, k < 2 ? D : I, k < 2 ? I : D, 0, &rws, &stride, &sc)) {
                printf("xbatch v4: no tensor\n"); return 1;
            }
            for (int o = 0; o < (k < 2 ? I : D); o++) memcpy(rws + (size_t)o * stride, src[k] + (size_t)o * D, (size_t)D * 4);
            sc[0] = 1.0f;
        }
        if (!coli_vk_tensor_commit(t, 3)) { printf("xbatch v4: commit failed\n"); return 1; }
        if (!(ex[c] = coli_vk_xb_expert(t[0], t[1], t[2]))) { printf("xbatch v4: no expert\n"); return 1; }
    }
    int total = 0; for (int c = 0; c < K; c++) total += rows[c];
    float *x = calloc((size_t)total * D, sizeof(float)), *w = malloc((size_t)total * sizeof(float));
    const float **xr = malloc((size_t)total * sizeof(*xr)), **yr = malloc((size_t)total * sizeof(*yr));
    static const float xs[4] = {1.0f, 2.0f, 0.5f, 1.0f}, ws[5] = {1.0f, 0.75f, 1.5f, 0.3f, 1.0f};
    for (int j = 0; j < total; j++) { x[(size_t)j * D] = j == 0 ? 1.0f : xs[j % 4]; w[j] = j == 0 ? 1.0f : ws[j % 5]; xr[j] = x + (size_t)j * D; }
    int mism = 0, sub = 0; double dms = 0;
    if (!coli_vk_xb_issue_w(ex, rows, K, xr, w) || !coli_vk_xb_join(yr, &dms)) { printf("xbatch v4: issue/join failed\n"); bad = 1; }
    for (int j = 0; !bad && j < total; j++) digest(yr[j], (size_t)I * 4);
    float h[I];
    for (int c = 0, j = 0; !bad && c < K; c++)
        for (int r = 0; r < rows[c]; r++, j++) {
            for (int o = 0; o < I; o++) {
                float g = v4r_bf16(x[(size_t)j * D] * gw[c][(size_t)o * D]), u = v4r_bf16(x[(size_t)j * D] * uw[c][(size_t)o * D]);
                float sg = g >= 0.0f ? 1.0f / (1.0f + expf(-g)) : expf(g) / (1.0f + expf(g));
                h[o] = v4r_bf16(g * sg * u * w[j]);
            }
            v4r_qdq(h, I);
            for (int o = 0; o < I; o++) {
                if (yr[j][o] != h[o]) { if (mism < 5) printf("  v4 mismatch: expert %d row %d [%d]: device %.9g, reference %.9g\n", c, r, o, yr[j][o], h[o]); mism++; }
                if (c == 0 && j == 0 && o < nt) sub += h[o] != 0.0f && fabsf(h[o]) < ldexpf(1.0f, -10);   /* subnormal codes */
            }
        }
    printf("xbatch v4 D=%d I=%d, rows 1/3/20 (GEMM from %d) | %d of %d values differ from the CPU's arithmetic | device %.3f ms\n",
           D, I, g_xb[0].gemm_rows, mism, total * I, dms);
    if (mism || !sub) bad = 1;
    for (int c = 0; c < K; c++) { coli_vk_xb_expert_free(ex[c]); free(gw[c]); free(uw[c]); }
    free(dw); free(x); free(w); free(xr); free(yr);
    xb_shutdown(&g_xb[0]);
    return bad;
}

/* Every format through the batch, both routes, the two activations, a hidden size
 * past the gate_up staging array, odd widths and a down format other than gate/up's. */
static int run_xbatch_all(void) {
    int bad = 0;
    if (!coli_vk_xb_init(320, 160, COLI_VK_ACT_SWIGLU, 0.f, 4.f, 25.f)) { printf("xbatch: init failed (gate_up.spv missing?)\n"); return 1; }
    printf("xbatch: tier queue %s, GEMM route from %d rows, timestamps %s\n",
           G.tq_shared ? "shared with the main queue" : "of its own", g_xb[0].gemm_rows, g_xb[0].has_ts ? "on" : "off");
    bad |= run_tier_pool();   /* first: the tier pool is still empty, its budget is exact */
    static const int fm[9] = {1, 2, 4, 5, 7, 10, 11, 12, 13};
    for (int f = 0; f < 9; f++) {
        int gs = fm[f] == 7 ? 32 : 64;
        bad |= run_xbatch(fm[f], fm[f], gs, 320, 160, COLI_VK_ACT_SWIGLU, 0.f);
    }
    bad |= run_xbatch(4, 1, 64, 320, 160, COLI_VK_ACT_SWIGLU, 0.f);      /* int4 gate/up, int8 down */
    bad |= run_xbatch(4, 10, 64, 320, 160, COLI_VK_ACT_SWIGLU, 0.f);     /* cooperative gate/up, fp32 down */
    bad |= run_xbatch(10, 1, 64, 320, 160, COLI_VK_ACT_SWIGLU, 0.f);     /* fp32 gate/up, cooperative down */
    bad |= run_xbatch(4, 4, 64, 320, 160, COLI_VK_ACT_SWIGLU, 0.7f);     /* the clamped SwiGLU */
    bad |= run_xbatch(7, 7, 32, 320, 160, COLI_VK_ACT_SITU, 0.f);        /* SiTU-GLU on MXFP4 */
    xb_shutdown(&g_xb[0]);
    /* odd widths: D, I not multiples of the groups, offsets past odd row sizes */
    if (!coli_vk_xb_init(200, 72, COLI_VK_ACT_SWIGLU, 0.f, 0.f, 0.f)) return 1;
    bad |= run_xbatch(12, 12, 8, 200, 72, COLI_VK_ACT_SWIGLU, 0.f);
    bad |= run_xbatch(2, 2, 0, 200, 72, COLI_VK_ACT_SWIGLU, 0.f);
    xb_shutdown(&g_xb[0]);
    /* a hidden size the gate_up shader cannot stage (> 6144) */
    if (!coli_vk_xb_init(6400, 64, COLI_VK_ACT_SWIGLU, 0.f, 0.f, 0.f)) return 1;
    bad |= run_xbatch(4, 4, 64, 6400, 64, COLI_VK_ACT_SWIGLU, 0.f);
    xb_shutdown(&g_xb[0]);
    bad |= run_xbatch_v4();   /* DeepSeek V4's activation, bit for bit */
    return bad;
}

/* COLI_VK_TEST_HOSTMEM=1: the expert batch reading its experts from the tier pool
 * (device memory the tier fills by a copy) against host memory the device reads in
 * place (VK_EXT_external_memory_host, no copy). Batches of 10 experts of Qwen3.8's
 * shape (hidden 2560, inter 640, int4-g64) cycle through 48 distinct experts so no
 * expert is in a cache; the device time comes from the batch's timestamps. Also
 * what the copy into the tier pool costs per expert, against a copy into malloc'd
 * memory, and whether both memories give the same bits. */
typedef struct { void *host[2]; VkDeviceMemory mem[2]; } HostImp;
static int host_import(size_t bytes, void **host, VkDeviceMemory *mem, VkBuffer *buf) {
#ifdef VK_EXT_external_memory_host
    VkPhysicalDeviceExternalMemoryHostPropertiesEXT hp = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT};
    VkPhysicalDeviceProperties2 p2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &hp};
    vkGetPhysicalDeviceProperties2(G.phys, &p2);
    size_t al = hp.minImportedHostPointerAlignment ? hp.minImportedHostPointerAlignment : 4096;
    size_t sz = (bytes + al - 1) / al * al;
    if (posix_memalign(host, al, sz)) return 0;
    memset(*host, 0, sz);
    PFN_vkGetMemoryHostPointerPropertiesEXT gp = (PFN_vkGetMemoryHostPointerPropertiesEXT)
        vkGetDeviceProcAddr(G.dev, "vkGetMemoryHostPointerPropertiesEXT");
    VkMemoryHostPointerPropertiesEXT mp = {.sType = VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
    if (!gp || gp(G.dev, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, *host, &mp) != VK_SUCCESS || !mp.memoryTypeBits) return 0;
    uint32_t mt = 0; while (!(mp.memoryTypeBits & (1u << mt))) mt++;
    VkImportMemoryHostPointerInfoEXT imp = {.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT,
        .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, .pHostPointer = *host};
    VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .pNext = &imp, .allocationSize = sz, .memoryTypeIndex = mt};
    if (vkAllocateMemory(G.dev, &ai, NULL, mem) != VK_SUCCESS) return 0;
    VkExternalMemoryBufferCreateInfo eb = {.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT};
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .pNext = &eb, .size = bytes,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    if (vkCreateBuffer(G.dev, &bi, NULL, buf) != VK_SUCCESS) return 0;
    if (vkBindBufferMemory(G.dev, *buf, *mem, 0) != VK_SUCCESS) return 0;
    static int said;
    if (!said++) printf("hostmem: imported with alignment %zu, memory type %u\n", al, mt);
    return 1;
#else
    (void)bytes; (void)host; (void)mem; (void)buf; return 0;
#endif
}
static ColiVkTensor *host_tensor(int fmt, int I, int O, int gs, HostImp *hi, uint8_t **rows, float **sc) {
    ColiVkTensor *t = calloc(1, sizeof(*t));
    t->fmt = fmt; t->I = I; t->O = O; t->rowWords = rowwords(fmt, I); t->gs = gs;
    t->wbytes = (size_t)t->rowWords * 4 * O;
    size_t sb = scale_floats(fmt, I, O, gs) * 4;
    if (!host_import(t->wbytes, &hi->host[0], &hi->mem[0], &t->wbuf) || !host_import(sb, &hi->host[1], &hi->mem[1], &t->sbuf)) {
        free(t); return NULL;
    }
    *rows = hi->host[0]; *sc = hi->host[1];
    return t;
}
static void bench_hostmem(void) {
    enum { N = 48, B = 10, R = 60 };
    const int D = 2560, I = 640, gs = 64;
    if (!G.has_hostmem) { printf("hostmem: VK_EXT_external_memory_host absent on this device\n"); return; }
    xb_shutdown(&g_xb[0]);
    if (!coli_vk_xb_init(D, I, COLI_VK_ACT_SWIGLU, 0.f, 0.f, 0.f)) { printf("hostmem: no expert batch\n"); return; }
    coli_vk_tier_pool_limit(0);
    size_t rbg = (size_t)(D / 2), rbd = (size_t)(I / 2);
    size_t sg = (size_t)I * (D / gs), sd = (size_t)D * (I / gs);
    size_t ebytes = 2 * (rbg * I + sg * 4) + rbd * D + sd * 4;
    uint8_t *src = malloc(2 * rbg * I + rbd * D); float *ssrc = malloc((2 * sg + sd) * 4);
    for (size_t i = 0; i < 2 * rbg * I + rbd * D; i++) src[i] = (uint8_t)rand();
    for (size_t i = 0; i < 2 * sg + sd; i++) ssrc[i] = 0.001f * (1 + rand() % 7);
    ColiVkExpert *ex[2][N]; HostImp hi[N][3];
    double copy_ms = 0, malloc_ms = 0;
    uint8_t *mbuf = malloc(ebytes);
    for (int kind = 0; kind < 2; kind++)
        for (int e = 0; e < N; e++) {
            ColiVkTensor *t[3]; uint8_t *rows; float *sc; size_t stride;
            for (int k = 0; k < 3; k++) {
                int In = k < 2 ? D : I, On = k < 2 ? I : D;
                const uint8_t *cs = src + (k < 2 ? k * rbg * I : 2 * rbg * I);
                const float *ss = ssrc + (k < 2 ? k * sg : 2 * sg);
                size_t rb = k < 2 ? rbg : rbd, ns = k < 2 ? sg : sd;
                if (kind == 0) {
                    if (!coli_vk_tier_tensor(&t[k], 4, In, On, gs, &rows, &stride, &sc)) { printf("hostmem: tier pool full\n"); return; }
                } else {
                    t[k] = host_tensor(4, In, On, gs, &hi[e][k], &rows, &sc); stride = rb;
                    if (!t[k]) { printf("hostmem: import failed\n"); return; }
                }
                double c0 = vk_now();
                for (int o = 0; o < On; o++) memcpy(rows + (size_t)o * stride, cs + (size_t)o * rb, rb);
                memcpy(sc, ss, ns * 4);
                if (kind == 0) { coli_vk_tensor_commit(&t[k], 1); copy_ms += vk_now() - c0; }
            }
            if (kind == 0) { double c0 = vk_now(); memcpy(mbuf, src, 2 * rbg * I + rbd * D); memcpy(mbuf + 2 * rbg * I + rbd * D, ssrc, (2 * sg + sd) * 4);
                         __asm__ volatile("" :: "r"(mbuf) : "memory");   /* the copy is kept: nothing reads mbuf */
                         malloc_ms += vk_now() - c0; }
            ex[kind][e] = coli_vk_xb_expert(t[0], t[1], t[2]);
        }
    float *x = malloc((size_t)D * 4); for (int i = 0; i < D; i++) x[i] = 0.01f * (i % 13) - 0.06f;
    const float *xr[B], *yr[B]; int rows1[B];
    for (int j = 0; j < B; j++) { xr[j] = x; rows1[j] = 1; }
    double ms[2] = {0, 0}, wall[2] = {0, 0};
    float *y0 = malloc((size_t)D * 4 * B);
    int same = 1;
    for (int kind = 0; kind < 2; kind++)
        for (int r = 0; r < R; r++) {
            ColiVkExpert *b[B];
            for (int j = 0; j < B; j++) b[j] = ex[kind][(r * B + j) % N];
            double t0 = vk_now(), dm = 0;
            coli_vk_xb_issue(b, rows1, B, xr); coli_vk_xb_join(yr, &dm);
            if (r) { ms[kind] += dm; wall[kind] += vk_now() - t0; }   /* the first batch warms the pipelines */
            if (r == 0) for (int j = 0; j < B; j++) {
                if (kind == 0) memcpy(y0 + (size_t)j * D, yr[j], (size_t)D * 4);
                else same &= !memcmp(y0 + (size_t)j * D, yr[j], (size_t)D * 4);
            }
        }
    double gb = (double)ebytes * B * (R - 1) / 1e9;
    printf("hostmem: %d batches of %d experts (%.2f MB each): device memory %.2f ms/batch = %.1f GB/s (wall %.2f ms), "
           "imported host memory %.2f ms/batch = %.1f GB/s (wall %.2f ms); same bits: %s\n",
           R - 1, B, ebytes / 1e6, ms[0] / (R - 1), gb / (ms[0] / 1e3), wall[0] / (R - 1),
           ms[1] / (R - 1), gb / (ms[1] / 1e3), wall[1] / (R - 1), same ? "yes" : "NO");
    printf("hostmem: one expert copied into the tier pool %.3f ms (%.1f GB/s), into malloc'd memory %.3f ms\n",
           copy_ms / N, ebytes / 1e6 / copy_ms * N, malloc_ms / N);
    for (int e = 0; e < N; e++) {
        coli_vk_xb_expert_free(ex[0][e]);
        ColiVkExpert *h = ex[1][e];
        ColiVkTensor *ts[3] = {h->g, h->u, h->d};
        xb_expert_unlink(h); free(h);
        for (int k = 0; k < 3; k++) {
            vkDestroyBuffer(G.dev, ts[k]->wbuf, NULL); vkDestroyBuffer(G.dev, ts[k]->sbuf, NULL);
            vkFreeMemory(G.dev, hi[e][k].mem[0], NULL); vkFreeMemory(G.dev, hi[e][k].mem[1], NULL);
            free(hi[e][k].host[0]); free(hi[e][k].host[1]); free(ts[k]);
        }
    }
    free(src); free(ssrc); free(mbuf); free(x); free(y0);
    xb_shutdown(&g_xb[0]);
}

int main(int argc, char **argv) {
    const char *spv = argc > 1 ? argv[1] : "shaders/qmatmul.spv";
    if (!coli_vk_init(spv)) { printf("vk init failed\n"); return 1; }
    /* COLI_VK_TEST_NOFILL=1: staged uploads without the zero fill of a fresh block (the
     * Polaris finding of #1338: does it still matter on this device?) */
    if (getenv("COLI_VK_TEST_NOFILL") && atoi(getenv("COLI_VK_TEST_NOFILL"))) g_up[0].fill = 0;
    printf("memory: %s\n", g_up[0].on ? g_up[0].fill ? "staged uploads" : "staged uploads, no zero fill" : "mapped");
    srand(1234);
    int bad = 0;
    /* COLI_VK_TEST_BALLAST=N: allocate N idle 4 MB device buffers before benching.
     * Probes whether per-submit cost scales with the process's ALLOCATION COUNT
     * (RADV/amdgpu CS buffer-list accounting) independent of bytes — the suspected
     * mechanism behind decode attention degrading with expert-tier size even with
     * VRAM to spare (7.9s @2.6k BOs -> 15.6s @4.3k with 2.9 GB free). */
    {
        int nb = getenv("COLI_VK_TEST_BALLAST") ? atoi(getenv("COLI_VK_TEST_BALLAST")) : 0;
        for (int i = 0; i < nb; i++) {
            VkBuffer b; VkDeviceMemory m;
            if (!alloc_hostvis_mt(4u << 20, &b, &m, NULL, G.memtype)) { printf("ballast stop at %d\n", i); break; }
        }
        if (nb) printf("ballast: %d x 4 MB idle allocations\n", nb);
    }
    if (getenv("COLI_VK_TEST_GEMM_BENCH") && atoi(getenv("COLI_VK_TEST_GEMM_BENCH"))) {
        bench_gemm();
        coli_vk_shutdown();
        return 0;
    }
    if (getenv("COLI_VK_TEST_HOSTMEM") && atoi(getenv("COLI_VK_TEST_HOSTMEM"))) {
        bench_hostmem();
        coli_vk_shutdown();
        return 0;
    }
    G.gemm_min_s = 0;   /* every case below is the GEMV's, but for run_gemm_case's */
    bad |= run_case(1, 1, 6144, 1536, 50);   // int8 expert gate/up shape (S=1 decode)
    bad |= run_case(2, 1, 6144, 1536, 50);   // int4 expert
    bad |= run_case(1, 1, 1536, 6144, 50);   // down proj shape
    bad |= run_case(2, 8, 6144, 1536, 20);   // prefill/MTP batch
    bad |= run_case(1, 1, 512, 512, 100);    // small
    bad |= run_case(2, 1, 16384, 6144, 20);  // o_proj shape: I > xsh capacity (unstaged path)
    /* int3-g64 (fmt=5): per-group scales through every dense shape incl. tail groups */
    bad |= run_case(5, 1, 6144, 2048, 50);   // int3 expert gate/up shape
    bad |= run_case(5, 1, 2048, 6144, 50);   // int3 down proj shape
    bad |= run_case(5, 8, 6144, 2048, 20);   // int3 batch
    bad |= run_case(5, 1, 100, 64, 20);      // partial tail group (I%64 != 0)
    bad |= run_case(5, 1, 16384, 6144, 20);  // int3 o_proj shape (unstaged path)
    /* f32 (fmt=10) and bf16 (fmt=11) weights: staged, batched, unstaged, odd widths */
    bad |= run_case(10, 1, 6144, 1536, 20);
    bad |= run_case(10, 8, 2048, 512, 10);
    bad |= run_case(10, 1, 16384, 512, 5);
    bad |= run_case(10, 3, 37, 5, 5);
    bad |= run_case(11, 1, 6144, 1536, 20);
    bad |= run_case(11, 8, 2048, 512, 10);
    bad |= run_case(11, 1, 16384, 512, 5);
    bad |= run_case(11, 3, 37, 5, 5);      // odd I: the last word holds one bf16
    /* fp8 e4m3 (fmt=12), grouped scales: 64 (harness default), then 32 and 128 below */
    bad |= run_case(12, 1, 6144, 1536, 20);
    bad |= run_case(12, 8, 2048, 512, 10);
    bad |= run_case(12, 1, 16384, 512, 5);
    bad |= run_case(12, 3, 100, 5, 5);     // partial tail group
    g_ref_gs = 32;  bad |= run_case(12, 2, 4096, 256, 5);
    g_ref_gs = 128; bad |= run_case(12, 2, 4096, 256, 5);
    g_ref_gs = 64;
    /* int8 with grouped scales (fmt=13), the qwen36 int8 gs64 container's experts */
    bad |= run_case(13, 1, 6144, 1536, 20);
    bad |= run_case(13, 8, 2048, 512, 10);
    bad |= run_case(13, 1, 16384, 512, 5);
    bad |= run_case(13, 3, 100, 5, 5);     // partial tail group
    g_ref_gs = 128; bad |= run_case(13, 2, 4096, 256, 5);
    g_ref_gs = 64;
    /* Tiled GEMM (prefill-sized S), every format: S = 16, 64 and 512 against odd O and
     * odd or unaligned I, so the last output, row and input tiles are all partial, the
     * last group is a tail, and x rows come both on and off a 16-byte quad. */
    {
        static const int gf[9] = {1, 2, 4, 5, 7, 10, 11, 12, 13};
        for (int f = 0; f < 9; f++) {
            int gs = gf[f] == 7 ? 32 : 64;
            bad |= run_gemm_case(gf[f], 16, 1001, 193, gs);   // odd I: tail group, x off a quad
            bad |= run_gemm_case(gf[f], 64, 1536, 97, gs);    // I % 4 == 0: x by quads
            bad |= run_gemm_case(gf[f], 512, 520, 131, gs);   // 8-input tail group
        }
        bad |= run_gemm_case(1, 37, 999, 65, 0);      // odd S, a 1-output last tile
        bad |= run_gemm_case(4, 33, 1001, 70, 8);     // the smallest grouped-int4 group
        bad |= run_gemm_case(4, 33, 1001, 70, 24);    // a group that is no power of two
        bad |= run_gemm_case(12, 33, 1001, 70, 12);   // fp8: the two words of a unit in two groups
        bad |= run_gemm_case(12, 33, 1001, 70, 128);
        bad |= run_gemm_case(13, 33, 1001, 70, 12);   // int8: the two words of a unit in two groups
    }
    /* The routed-expert tier's device side: every format through the async expert
     * batch against the CPU, and its weight pool (budget, reuse, deferred frees). */
    bad |= run_xbatch_all();
    /* COLI_VK_TEST_MATMUL_ONLY=1: stop after the per-format matmul and expert-batch
     * cases above. CI runs this on Lavapipe, where the benches below say nothing and
     * take most of the time. */
    if (getenv("COLI_VK_TEST_MATMUL_ONLY") && atoi(getenv("COLI_VK_TEST_MATMUL_ONLY"))) {
        printf("outputs digest %016llx\n", (unsigned long long)g_digest);
        bad |= run_late_cases();
        printf(bad ? "FAIL\n" : "PASS\n");
        coli_vk_shutdown();
        return bad;
    }
    /* Batched (amortized) throughput on the int4 expert shapes — the real expert-tier pattern. */
    {
        int I = 6144, O = 2048;   /* our gate/up dims */
        float *x = malloc((size_t)I * 4); uint8_t *w = malloc((size_t)(I + 1) / 2 * O); float *sc = malloc((size_t)O * 4);
        for (int i = 0; i < I; i++) x[i] = (rand() % 200 - 100) / 100.0f;
        for (size_t i = 0; i < (size_t)(I + 1) / 2 * O; i++) w[i] = rand() & 0xff;
        for (int o = 0; o < O; o++) sc[o] = 0.01f + (rand() % 100) / 10000.0f;
        float *y = malloc((size_t)O * 4);
        ColiVkTensor *t = NULL; coli_vk_matmul(&t, y, x, w, sc, 2, 1, I, O, 0);   /* bind */
        printf("BATCHED int4 S=1 6144->2048 (our gate/up): %.4f ms/matmul (N=64, one submit)\n",
               bench_batched(t, x, 2, 1, I, O, 64));
        coli_vk_tensor_free(t); free(x); free(w); free(sc); free(y);
        I = 2048; O = 6144;   /* our down dims */
        x = malloc((size_t)I * 4); w = malloc((size_t)(I + 1) / 2 * O); sc = malloc((size_t)O * 4);
        for (int i = 0; i < I; i++) x[i] = (rand() % 200 - 100) / 100.0f;
        for (size_t i = 0; i < (size_t)(I + 1) / 2 * O; i++) w[i] = rand() & 0xff;
        for (int o = 0; o < O; o++) sc[o] = 0.01f + (rand() % 100) / 10000.0f;
        y = malloc((size_t)O * 4);
        t = NULL; coli_vk_matmul(&t, y, x, w, sc, 2, 1, I, O, 0);
        printf("BATCHED int4 S=1 2048->6144 (our down):    %.4f ms/matmul (N=64, one submit)\n",
               bench_batched(t, x, 2, 1, I, O, 64));
        coli_vk_tensor_free(t); free(x); free(w); free(sc); free(y);
    }
    /* FUSED gate+up: correctness + batched throughput (vs 2x separate gate/up). */
    {
        int D = 6144, I = 2048;
        bad |= run_gate_up(2, 1, D, I);
        bad |= run_gate_up(5, 1, D, I);   // int3-g64 fused pair (per-group scales)
        size_t rb = (size_t)(D + 1) / 2;
        float *x = malloc((size_t)D*4); uint8_t *gw = malloc(rb*I), *uw = malloc(rb*I);
        float *gs = malloc((size_t)I*4), *us = malloc((size_t)I*4), *h = malloc((size_t)I*4);
        for (int i = 0; i < D; i++) x[i] = (rand()%200-100)/100.0f;
        for (size_t i = 0; i < rb*I; i++) { gw[i] = rand()&0xff; uw[i] = rand()&0xff; }
        for (int o = 0; o < I; o++) { gs[o] = 0.01f; us[o] = 0.01f; }
        ColiVkTensor *tg = NULL, *tu = NULL; coli_vk_gate_up(&tg, &tu, h, x, gw, gs, uw, us, 2, 1, D, I, 0);
        printf("BATCHED fused gate_up int4 6144->2048:     %.4f ms (N=64, SAME expert = L2-cached)\n",
               bench_gu_batched(tg, x, 2, 1, D, I, 64));
        coli_vk_tensor_free(tg); coli_vk_tensor_free(tu); free(x); free(gw); free(uw); free(gs); free(us); free(h);
    }
    /* FAIR: cycle 8 distinct experts (VRAM reads, not L2) — matches ROCm expert_group. */
    printf("FAIR fused gate_up int4 6144->2048 (8 distinct experts): %.4f ms/expert\n",
           bench_experts_fair(2, 6144, 2048, 8, 8));
    printf("FAIR fused gate_up int3 6144->2048 (8 distinct experts): %.4f ms/expert\n",
           bench_experts_fair(5, 6144, 2048, 8, 8));
    /* FULL expert_group: the real primitive. Sweep K to see if per-expert cost is fixed
     * per-call overhead (drops with K) or per-dispatch (constant). */
    bad |= run_expert_group(2, 6144, 2048, 1);
    bad |= run_expert_group(2, 6144, 2048, 8);
    bad |= run_expert_group(2, 6144, 2048, 32);
    /* int3-g64 expert group: correctness + the 0.86x-bytes throughput question.
     * count=1 included — it is the SHARED-expert path shape in the engine. */
    bad |= run_qprep(1, 1, 6144, 1536, 576, 16384);   /* GLM q_a/kv_a/q_b decode shapes */
    bad |= run_qprep(1, 11, 6144, 1536, 576, 16384);  /* prefill batch */
    bad |= run_qprep(1, 2, 6144, 1536, 576, 16384);   /* S=2 (MTP verify) */
    bad |= run_qprep(2, 1, 6144, 1536, 576, 16384);   /* int4 dense variant */
    /* fmt=4 grouped int4 (#298 semantics), gs=64 across real shapes + gs=32 sanity */
    g_ref_gs = 64;
    bad |= run_case(4, 1, 6144, 2048, 50);
    bad |= run_case(4, 1, 2048, 6144, 50);
    bad |= run_case(4, 8, 6144, 1536, 20);
    bad |= run_case(4, 1, 16384, 6144, 10);
    g_ref_gs = 32;
    bad |= run_case(4, 1, 6144, 2048, 20);
    g_ref_gs = 64;
    bad |= run_expert_group(4, 6144, 2048, 8);
    bad |= run_expert_group(5, 6144, 2048, 1);
    bad |= run_expert_group(5, 6144, 2048, 8);
    bad |= run_expert_group(5, 6144, 2048, 32);
    /* Fused same-input matmul pair (the q_a + kv_a prologue pattern), int4 AND int3. */
    for (int pi = 0; pi < 3; pi++) {
        int pf = pi == 0 ? 2 : pi == 1 ? 5 : 4;
        int I = 6144, O1 = 2048, O2 = 576, S = 1;
        size_t rb = ref_rowbytes(pf, I);
        size_t n1 = ref_scales(pf, I, O1), n2 = ref_scales(pf, I, O2);
        uint8_t *w1 = malloc(rb * O1), *w2 = malloc(rb * O2);
        float *s1 = malloc(n1 * 4), *s2 = malloc(n2 * 4), *x = malloc((size_t)I * 4);
        float *y1 = malloc((size_t)O1 * 4), *y2 = malloc((size_t)O2 * 4);
        float *c1 = malloc((size_t)O1 * 4), *c2 = malloc((size_t)O2 * 4);
        for (size_t i = 0; i < rb * (size_t)O1; i++) w1[i] = rand() & 0xff;
        for (size_t i = 0; i < rb * (size_t)O2; i++) w2[i] = rand() & 0xff;
        for (size_t o = 0; o < n1; o++) s1[o] = 0.01f + (rand() % 100) / 10000.0f;
        for (size_t o = 0; o < n2; o++) s2[o] = 0.01f + (rand() % 100) / 10000.0f;
        for (int i = 0; i < I; i++) x[i] = (rand() % 200 - 100) / 100.0f;
        ColiVkTensor *t1 = NULL, *t2 = NULL;
        if (!coli_vk_matmul_pair(&t1, y1, w1, s1, O1, &t2, y2, w2, s2, O2, pf, x, S, I, g_ref_gs)) {
            printf("matmul_pair fmt=%d failed\n", pf); bad = 1;
        } else {
            digest(y1, (size_t)O1 * 4); digest(y2, (size_t)O2 * 4);
            cpu_ref(c1, x, w1, s1, pf, S, I, O1); cpu_ref(c2, x, w2, s2, pf, S, I, O2);
            double mr = 0;
            for (int i = 0; i < O1; i++) { double e = fabs(y1[i]-c1[i]); if (fabs(c1[i])>1e-2) { double r=e/fabs(c1[i]); if (r>mr) mr=r; } }
            for (int i = 0; i < O2; i++) { double e = fabs(y2[i]-c2[i]); if (fabs(c2[i])>1e-2) { double r=e/fabs(c2[i]); if (r>mr) mr=r; } }
            double t0 = now();
            for (int k = 0; k < 30; k++) coli_vk_matmul_pair(&t1, y1, w1, s1, O1, &t2, y2, w2, s2, O2, pf, x, S, I, g_ref_gs);
            printf("matmul_pair fmt=%d 6144->(2048,576)   | maxrel=%.4g | %.3f ms/pair-call\n", pf, mr, (now()-t0)*1000/30);
            bad |= mr > 1e-3;
        }
        coli_vk_tensor_free(t1); coli_vk_tensor_free(t2);
        free(w1); free(w2); free(s1); free(s2); free(x); free(y1); free(y2); free(c1); free(c2);
    }
    /* MLA absorb attention core (GLM decode shape + window/causal/int8 variants). */
    if (G.pipe_att) {
        bad |= run_absorb(2, 1, 64, 192, 64, 256, 512, 0, 300, 0);    // GLM-5.2 decode
        bad |= run_absorb(2, 1, 64, 192, 64, 256, 512, 17, 300, 1);   // kv_start window
        bad |= run_absorb(2, 2, 64, 192, 64, 256, 512, 0, 300, 2);    // S=2 causal (MTP verify)
        bad |= run_absorb(1, 1, 8, 128, 32, 64, 256, 0, 64, 3);       // int8, odd dims
        bad |= run_absorb(2, 1, 64, 192, 64, 256, 512, 0, 2000, 4);   // long context
        bad |= run_absorb(5, 1, 64, 192, 64, 256, 512, 0, 300, 5);    // int3-g64 kv_b + o
        bad |= run_absorb(5, 2, 64, 192, 64, 256, 512, 17, 300, 6);   // int3 S=2 causal + window
        bad |= run_absorb(4, 1, 64, 192, 64, 256, 512, 0, 300, 7);    // grouped-int4 kv_b + o
        bad |= run_absorb(4, 2, 64, 192, 64, 256, 512, 17, 300, 8);   // fmt=4 S=2 causal + window
        g_absorb_rewind = 1;   /* every row written twice, out of order: a rewound mirror */
        bad |= run_absorb(2, 1, 64, 192, 64, 256, 512, 0, 300, 9);
        g_absorb_rewind = 0;
    }
    printf("outputs digest %016llx\n", (unsigned long long)g_digest);
    bad |= run_late_cases();
    printf(bad ? "FAIL\n" : "PASS\n");
    coli_vk_shutdown();
    return bad;
}
#endif
