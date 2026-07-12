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
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VKCHECK(x, what) do { VkResult _r = (x); if (_r != VK_SUCCESS) { \
    fprintf(stderr, "[VK] %s failed: %d\n", what, _r); return 0; } } while (0)

struct ColiVkTensor {
    VkBuffer wbuf, sbuf;
    VkDeviceMemory wmem, smem;
    size_t wbytes;
    int fmt, I, O, rowWords;
};

typedef struct {
    VkBuffer buf; VkDeviceMemory mem; void *ptr; size_t cap;
} Scratch;

static struct {
    int ready;
    VkInstance inst;
    VkPhysicalDevice phys;
    VkDevice dev;
    VkQueue queue;
    uint32_t qfam;
    uint32_t memtype;            // HOST_VISIBLE|HOST_COHERENT (prefer DEVICE_LOCAL)
    VkDescriptorSetLayout dsl;
    VkPipelineLayout plyt;
    VkPipeline pipe;
    VkShaderModule shader;
    VkDescriptorPool dpool;
    VkDescriptorSet dset;
    VkCommandPool cpool;
    VkCommandBuffer cmd;
    VkFence fence;
    Scratch x, y;
    size_t used_bytes, tensor_count;
} G;

struct PC { int fmt, S, I, O, rowWords; };

static int pick_memtype(void) {
    VkPhysicalDeviceMemoryProperties m;
    vkGetPhysicalDeviceMemoryProperties(G.phys, &m);
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

static int alloc_hostvis(size_t bytes, VkBuffer *buf, VkDeviceMemory *mem, void **ptr) {
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = bytes, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    VKCHECK(vkCreateBuffer(G.dev, &bi, NULL, buf), "vkCreateBuffer");
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(G.dev, *buf, &req);
    VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = req.size, .memoryTypeIndex = G.memtype};
    VKCHECK(vkAllocateMemory(G.dev, &ai, NULL, mem), "vkAllocateMemory");
    VKCHECK(vkBindBufferMemory(G.dev, *buf, *mem, 0), "vkBindBufferMemory");
    if (ptr) VKCHECK(vkMapMemory(G.dev, *mem, 0, bytes, 0, ptr), "vkMapMemory");
    return 1;
}

static int scratch_reserve(Scratch *s, size_t bytes) {
    if (s->cap >= bytes) return 1;
    if (s->buf) { vkDestroyBuffer(G.dev, s->buf, NULL); vkFreeMemory(G.dev, s->mem, NULL); }
    s->buf = VK_NULL_HANDLE; s->cap = 0; s->ptr = NULL;
    if (!alloc_hostvis(bytes, &s->buf, &s->mem, &s->ptr)) return 0;
    s->cap = bytes;
    return 1;
}

static int rowwords(int fmt, int I) {
    size_t rb = fmt == 1 ? (size_t)I : (size_t)(I + 1) / 2; // bytes/row on CPU side
    return (int)((rb + 3) / 4);                              // padded to uint32
}

static VkShaderModule load_spv(const char *path) {
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
    VkResult r = vkCreateShaderModule(G.dev, &si, NULL, &m);
    free(code);
    return r == VK_SUCCESS ? m : VK_NULL_HANDLE;
}

int coli_vk_init(const char *spv_path) {
    if (G.ready) return 1;
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
    G.phys = devs[0];
    int bestrank = -1;
    for (uint32_t i = 0; i < nd; i++) {
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

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = G.qfam, .queueCount = 1, .pQueuePriorities = &prio};
    VkDeviceCreateInfo di = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1, .pQueueCreateInfos = &qi};
    VKCHECK(vkCreateDevice(G.phys, &di, NULL, &G.dev), "vkCreateDevice");
    vkGetDeviceQueue(G.dev, G.qfam, 0, &G.queue);

    int mt = pick_memtype();
    if (mt < 0) { fprintf(stderr, "[VK] no host-visible memory\n"); return 0; }
    G.memtype = (uint32_t)mt;

    G.shader = load_spv(spv_path);
    if (!G.shader) return 0;

    VkDescriptorSetLayoutBinding b[4];
    for (int i = 0; i < 4; i++) b[i] = (VkDescriptorSetLayoutBinding){
        .binding = i, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT};
    VkDescriptorSetLayoutCreateInfo dsli = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 4, .pBindings = b};
    VKCHECK(vkCreateDescriptorSetLayout(G.dev, &dsli, NULL, &G.dsl), "descSetLayout");

    VkPushConstantRange pcr = {.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0, .size = sizeof(struct PC)};
    VkPipelineLayoutCreateInfo pli = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1, .pSetLayouts = &G.dsl,
        .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr};
    VKCHECK(vkCreatePipelineLayout(G.dev, &pli, NULL, &G.plyt), "pipelineLayout");

    VkComputePipelineCreateInfo cpi = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                  .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = G.shader, .pName = "main"},
        .layout = G.plyt};
    VKCHECK(vkCreateComputePipelines(G.dev, VK_NULL_HANDLE, 1, &cpi, NULL, &G.pipe), "pipeline");

    VkDescriptorPoolSize ps = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 4};
    VkDescriptorPoolCreateInfo dpi = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &ps};
    VKCHECK(vkCreateDescriptorPool(G.dev, &dpi, NULL, &G.dpool), "descPool");
    VkDescriptorSetAllocateInfo dsa = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = G.dpool, .descriptorSetCount = 1, .pSetLayouts = &G.dsl};
    VKCHECK(vkAllocateDescriptorSets(G.dev, &dsa, &G.dset), "allocDescSet");

    VkCommandPoolCreateInfo cpci = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = G.qfam};
    VKCHECK(vkCreateCommandPool(G.dev, &cpci, NULL, &G.cpool), "cmdPool");
    VkCommandBufferAllocateInfo cbi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = G.cpool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
    VKCHECK(vkAllocateCommandBuffers(G.dev, &cbi, &G.cmd), "cmdBuf");
    VkFenceCreateInfo fi = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VKCHECK(vkCreateFence(G.dev, &fi, NULL, &G.fence), "fence");

    G.ready = 1;
    VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(G.phys, &p);
    fprintf(stderr, "[VK] ready: %s, compute qfam %u, memtype %u\n", p.deviceName, G.qfam, G.memtype);
    return 1;
}

int coli_vk_available(void) { return G.ready; }

void coli_vk_mem_info(size_t *used, size_t *count) {
    if (used) *used = G.used_bytes;
    if (count) *count = G.tensor_count;
}

static int upload_tensor(ColiVkTensor **out, const void *weights, const float *scales,
                         int fmt, int I, int O) {
    if (*out) return (*out)->fmt == fmt && (*out)->I == I && (*out)->O == O;
    if (fmt != 1 && fmt != 2) return 0;
    ColiVkTensor *t = calloc(1, sizeof(*t));
    if (!t) return 0;
    t->fmt = fmt; t->I = I; t->O = O; t->rowWords = rowwords(fmt, I);
    size_t stride = (size_t)t->rowWords * 4;         // padded row bytes
    size_t cpu_rb = fmt == 1 ? (size_t)I : (size_t)(I + 1) / 2;
    t->wbytes = stride * (size_t)O;
    void *wptr;
    if (!alloc_hostvis(t->wbytes, &t->wbuf, &t->wmem, &wptr)) { free(t); return 0; }
    memset(wptr, 0, t->wbytes);
    for (int o = 0; o < O; o++)                        // copy row-by-row into padded layout
        memcpy((uint8_t *)wptr + (size_t)o * stride,
               (const uint8_t *)weights + (size_t)o * cpu_rb, cpu_rb);
    void *sptr;
    if (!alloc_hostvis((size_t)O * sizeof(float), &t->sbuf, &t->smem, &sptr)) {
        vkDestroyBuffer(G.dev, t->wbuf, NULL); vkFreeMemory(G.dev, t->wmem, NULL); free(t); return 0;
    }
    memcpy(sptr, scales, (size_t)O * sizeof(float));
    // Counters are touched concurrently: frees run from expert_load under
    // `#pragma omp parallel`, so RMW them atomically (torn counts otherwise).
    __atomic_add_fetch(&G.used_bytes, t->wbytes + (size_t)O * sizeof(float), __ATOMIC_RELAXED);
    __atomic_add_fetch(&G.tensor_count, 1, __ATOMIC_RELAXED);
    *out = t;
    return 1;
}

int coli_vk_matmul(ColiVkTensor **tensor, float *y, const float *x,
                   const void *weights, const float *scales,
                   int fmt, int S, int I, int O) {
    if (!G.ready || S < 1 || !upload_tensor(tensor, weights, scales, fmt, I, O)) return 0;
    ColiVkTensor *t = *tensor;
    size_t xb = (size_t)S * I * sizeof(float), yb = (size_t)S * O * sizeof(float);
    if (!scratch_reserve(&G.x, xb) || !scratch_reserve(&G.y, yb)) return 0;
    memcpy(G.x.ptr, x, xb);

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

    VKCHECK(vkResetCommandBuffer(G.cmd, 0), "resetCmd");
    VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    VKCHECK(vkBeginCommandBuffer(G.cmd, &begin), "beginCmd");
    vkCmdBindPipeline(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe);
    vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt, 0, 1, &G.dset, 0, NULL);
    struct PC pc = {fmt, S, I, O, t->rowWords};
    vkCmdPushConstants(G.cmd, G.plyt, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(G.cmd, (uint32_t)O, (uint32_t)S, 1);
    VKCHECK(vkEndCommandBuffer(G.cmd), "endCmd");

    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1, .pCommandBuffers = &G.cmd};
    VKCHECK(vkResetFences(G.dev, 1, &G.fence), "resetFence");
    VKCHECK(vkQueueSubmit(G.queue, 1, &si, G.fence), "queueSubmit");
    // Bounded wait: a GPU hang/TDR must never wedge the process. 10s is orders of
    // magnitude over a single-GEMV dispatch; on timeout/device-loss disable VK for
    // the rest of the run and fall back to CPU (the caller degrades on our 0 return).
    VkResult wr = vkWaitForFences(G.dev, 1, &G.fence, VK_TRUE, 10000000000ULL);
    if (wr != VK_SUCCESS) {
        fprintf(stderr, "[VK] fence wait failed: %d — disabling GPU offload, staying on CPU\n", wr);
        G.ready = 0;
        return 0;
    }
    memcpy(y, G.y.ptr, yb);
    return 1;
}

void coli_vk_tensor_free(ColiVkTensor *t) {
    if (!t) return;
    // If the device was lost/disabled (the fence-timeout path sets G.ready=0), a submission
    // may still reference these buffers — do NOT vkDestroy into a dead device (GPU-side UAF).
    // Leak the GPU handles (we're degrading to CPU for the rest of the run) and reclaim the
    // host struct + counters only.
    if (G.ready) {
        if (t->wbuf) { vkDestroyBuffer(G.dev, t->wbuf, NULL); vkFreeMemory(G.dev, t->wmem, NULL); }
        if (t->sbuf) { vkDestroyBuffer(G.dev, t->sbuf, NULL); vkFreeMemory(G.dev, t->smem, NULL); }
    }
    // Mirror upload_tensor exactly (weights + scales), atomically — otherwise
    // used_bytes leaks the O*float scales buffer on every free and drifts upward.
    __atomic_sub_fetch(&G.tensor_count, 1, __ATOMIC_RELAXED);
    __atomic_sub_fetch(&G.used_bytes, t->wbytes + (size_t)t->O * sizeof(float), __ATOMIC_RELAXED);
    free(t);
}

size_t coli_vk_tensor_bytes(const ColiVkTensor *t) { return t ? t->wbytes : 0; }

void coli_vk_shutdown(void) {
    if (!G.ready) return;
    vkDeviceWaitIdle(G.dev);
    if (G.x.buf) { vkDestroyBuffer(G.dev, G.x.buf, NULL); vkFreeMemory(G.dev, G.x.mem, NULL); }
    if (G.y.buf) { vkDestroyBuffer(G.dev, G.y.buf, NULL); vkFreeMemory(G.dev, G.y.mem, NULL); }
    vkDestroyFence(G.dev, G.fence, NULL);
    vkDestroyCommandPool(G.dev, G.cpool, NULL);
    vkDestroyDescriptorPool(G.dev, G.dpool, NULL);
    vkDestroyPipeline(G.dev, G.pipe, NULL);
    vkDestroyPipelineLayout(G.dev, G.plyt, NULL);
    vkDestroyDescriptorSetLayout(G.dev, G.dsl, NULL);
    vkDestroyShaderModule(G.dev, G.shader, NULL);
    vkDestroyDevice(G.dev, NULL);
    vkDestroyInstance(G.inst, NULL);
    memset(&G, 0, sizeof(G));
}

#ifdef VK_TEST
// ---- standalone GPU-vs-CPU validation + microbench --------------------------
#include <math.h>
#include <time.h>

static double now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9; }

static float deq(const uint8_t *row, int fmt, int i) {
    if (fmt == 1) { int b = ((const int8_t *)row)[i]; return (float)b; }
    uint8_t v = row[i >> 1]; int nib = (i & 1) ? (v >> 4) : (v & 15); return (float)(nib - 8);
}

static void cpu_ref(float *y, const float *x, const uint8_t *w, const float *sc,
                    int fmt, int S, int I, int O) {
    size_t rb = fmt == 1 ? (size_t)I : (size_t)(I + 1) / 2;
    for (int s = 0; s < S; s++) for (int o = 0; o < O; o++) {
        double sum = 0; const uint8_t *row = w + (size_t)o * rb;
        for (int i = 0; i < I; i++) sum += x[s * I + i] * deq(row, fmt, i);
        y[s * O + o] = (float)(sum * sc[o]);
    }
}

static int run_case(int fmt, int S, int I, int O, int iters) {
    size_t rb = fmt == 1 ? (size_t)I : (size_t)(I + 1) / 2;
    float *x = malloc((size_t)S * I * sizeof(float));
    uint8_t *w = malloc(rb * O);
    float *sc = malloc((size_t)O * sizeof(float));
    float *yg = malloc((size_t)S * O * sizeof(float));
    float *yc = malloc((size_t)S * O * sizeof(float));
    for (int i = 0; i < S * I; i++) x[i] = (float)((rand() % 200 - 100) / 100.0);
    for (size_t i = 0; i < rb * O; i++) w[i] = rand() & 0xff;
    for (int o = 0; o < O; o++) sc[o] = 0.01f + (rand() % 100) / 10000.0f;

    ColiVkTensor *t = NULL;
    if (!coli_vk_matmul(&t, yg, x, w, sc, fmt, S, I, O)) { printf("matmul failed\n"); return 1; }
    cpu_ref(yc, x, w, sc, fmt, S, I, O);
    double maxerr = 0, maxrel = 0;
    for (int i = 0; i < S * O; i++) {
        double e = fabs(yg[i] - yc[i]); if (e > maxerr) maxerr = e;
        if (fabs(yc[i]) > 1e-2) { double r = e / fabs(yc[i]); if (r > maxrel) maxrel = r; }
    }
    // microbench (GPU)
    double t0 = now();
    for (int k = 0; k < iters; k++) coli_vk_matmul(&t, yg, x, w, sc, fmt, S, I, O);
    double gpu_ms = (now() - t0) * 1000 / iters;
    // microbench (CPU ref, 1 iter — it's slow)
    double c0 = now(); cpu_ref(yc, x, w, sc, fmt, S, I, O); double cpu_ms = (now() - c0) * 1000;
    printf("fmt=%d S=%d I=%d O=%d | maxerr=%.4g maxrel=%.4g | gpu=%.3f ms  cpu_ref=%.3f ms\n",
           fmt, S, I, O, maxerr, maxrel, gpu_ms, cpu_ms);
    coli_vk_tensor_free(t);
    free(x); free(w); free(sc); free(yg); free(yc);
    return maxrel > 1e-3 ? 1 : 0;
}

int main(int argc, char **argv) {
    const char *spv = argc > 1 ? argv[1] : "shaders/qmatmul.spv";
    if (!coli_vk_init(spv)) { printf("vk init failed\n"); return 1; }
    srand(1234);
    int bad = 0;
    bad |= run_case(1, 1, 6144, 1536, 50);   // int8 expert gate/up shape (S=1 decode)
    bad |= run_case(2, 1, 6144, 1536, 50);   // int4 expert
    bad |= run_case(1, 1, 1536, 6144, 50);   // down proj shape
    bad |= run_case(2, 8, 6144, 1536, 20);   // prefill/MTP batch
    bad |= run_case(1, 1, 512, 512, 100);    // small
    printf(bad ? "FAIL\n" : "PASS\n");
    coli_vk_shutdown();
    return bad;
}
#endif
