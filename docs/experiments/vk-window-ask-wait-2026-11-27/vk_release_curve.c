/* vk_release_curve: how fast a Vulkan driver reports freed device memory as free again.
 *
 * It does what colibri's vk_window_decide does at a start (backend_vulkan.c): allocate 1 GiB
 * blocks of the host-visible device-local type up to N, free them all, and then reads the
 * heap's free memory (VK_EXT_memory_budget: budget less usage) every POLL microseconds for
 * up to WAIT ms. Per round it prints:
 *   - the heap's free before the ask, after the blocks, and after the free;
 *   - when the free came back to within 64 MiB of where it was (vk_window_decide's rule),
 *     and when each GiB of it came back;
 *   - the free at the moments an engine might size its expert tier (TIER_MS, default
 *     50,100,150,200,300 ms after the free), and what colibri's tier budget would be from it:
 *     free - COLI_VK_TIER_RESERVE_GB (1) - DENSE_GB - CHAIN_GB (Inkling-2-Mini UD-Q3_K_XL at
 *     CTX_MAX=73728: 0.89 dense + 2.44 chain, from the engine's own startup lines).
 *
 *   cc -O2 vk_release_curve.c -lvulkan -o vk_release_curve
 *   ./vk_release_curve [rounds] [blocks]        env: DEV_ID=0x7590 POLL_US=1000 WAIT_MS=3000
 *                                                     GAP_MS=2000 TIER_MS=50,100,150,200,300
 *                                                     DENSE_GB=0.89 CHAIN_GB=2.44 RESERVE_GB=1
 */
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define GIB (1024.0 * 1024.0 * 1024.0)

static double now_ms(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}
static double envd(const char *k, double d) { const char *e = getenv(k); return e && *e ? atof(e) : d; }

static VkPhysicalDevice phys;
static uint32_t heap;
/* the heap's free bytes now: its budget less its usage (VK_EXT_memory_budget) */
static double heap_free(void) {
    VkPhysicalDeviceMemoryBudgetPropertiesEXT b = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
    VkPhysicalDeviceMemoryProperties2 m = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2, .pNext = &b};
    vkGetPhysicalDeviceMemoryProperties2(phys, &m);
    return b.heapBudget[heap] > b.heapUsage[heap] ? (double)(b.heapBudget[heap] - b.heapUsage[heap]) : 0.0;
}

int main(int argc, char **argv) {
    const int rounds = argc > 1 ? atoi(argv[1]) : 5, want = argc > 2 ? atoi(argv[2]) : 13;
    const double poll_us = envd("POLL_US", 1000), wait_ms = envd("WAIT_MS", 3000), gap_ms = envd("GAP_MS", 2000);
    const double dense = envd("DENSE_GB", 0.89) * GIB, chain = envd("CHAIN_GB", 2.44) * GIB, reserve = envd("RESERVE_GB", 1) * GIB;
    const unsigned dev_id = (unsigned)strtoul(getenv("DEV_ID") ? getenv("DEV_ID") : "0", NULL, 0);
    double tier_ms[16]; int ntier = 0;
    {
        char buf[256]; snprintf(buf, sizeof buf, "%s", getenv("TIER_MS") ? getenv("TIER_MS") : "50,100,150,200,300");
        for (char *s = strtok(buf, ","); s && ntier < 16; s = strtok(NULL, ",")) tier_ms[ntier++] = atof(s);
    }

    VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .pApplicationName = "vk_release_curve",
                             .apiVersion = VK_API_VERSION_1_2};
    VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app};
    VkInstance inst;
    if (vkCreateInstance(&ici, NULL, &inst) != VK_SUCCESS) { fprintf(stderr, "vkCreateInstance failed\n"); return 1; }
    uint32_t n = 0; vkEnumeratePhysicalDevices(inst, &n, NULL);
    VkPhysicalDevice *pd = calloc(n, sizeof *pd); vkEnumeratePhysicalDevices(inst, &n, pd);
    VkPhysicalDeviceProperties pp;
    for (uint32_t i = 0; i < n && !phys; i++) {
        vkGetPhysicalDeviceProperties(pd[i], &pp);
        if (dev_id ? pp.deviceID == dev_id : pp.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) phys = pd[i];
    }
    if (!phys) { fprintf(stderr, "no matching device (DEV_ID=0x%x)\n", dev_id); return 1; }
    vkGetPhysicalDeviceProperties(phys, &pp);

    /* the host-visible device-local type colibri's window asks (its G.memtype) */
    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    const VkMemoryPropertyFlags need = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    int type = -1;
    for (uint32_t i = 0; i < mp.memoryTypeCount && type < 0; i++)
        if ((mp.memoryTypes[i].propertyFlags & need) == need) type = (int)i;
    if (type < 0) { fprintf(stderr, "no host-visible device-local memory type\n"); return 1; }
    heap = mp.memoryTypes[type].heapIndex;

    uint32_t ne = 0; vkEnumerateDeviceExtensionProperties(phys, NULL, &ne, NULL);
    VkExtensionProperties *ep = calloc(ne, sizeof *ep); vkEnumerateDeviceExtensionProperties(phys, NULL, &ne, ep);
    int has_budget = 0;
    for (uint32_t i = 0; i < ne; i++) has_budget |= !strcmp(ep[i].extensionName, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
    if (!has_budget) { fprintf(stderr, "%s has no VK_EXT_memory_budget\n", pp.deviceName); return 1; }
    const float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = 0,
                                   .queueCount = 1, .pQueuePriorities = &prio};
    const char *exts[] = {VK_EXT_MEMORY_BUDGET_EXTENSION_NAME};
    VkDeviceCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .queueCreateInfoCount = 1,
                              .pQueueCreateInfos = &qci, .enabledExtensionCount = 1, .ppEnabledExtensionNames = exts};
    VkDevice dev;
    if (vkCreateDevice(phys, &dci, NULL, &dev) != VK_SUCCESS) { fprintf(stderr, "vkCreateDevice failed\n"); return 1; }

    printf("%s, memory type %d (heap %u, %.2f GiB), %d rounds of %d x 1 GiB, polled every %.0f us for %.0f ms\n",
           pp.deviceName, type, heap, mp.memoryHeaps[heap].size / GIB, rounds, want, poll_us, wait_ms);
    printf("tier budget = free - %.2f reserve - %.2f dense - %.2f chain (GiB)\n\n", reserve / GIB, dense / GIB, chain / GIB);

    VkDeviceMemory *blk = calloc((size_t)want, sizeof *blk);
    const int maxs = (int)(wait_ms * 1000 / poll_us) + 16;
    double *ts = malloc(sizeof(double) * maxs), *fs = malloc(sizeof(double) * maxs);
    for (int r = 1; r <= rounds; r++) {
        const double free0 = heap_free();
        int got = 0;
        double t0 = now_ms();
        for (; got < want; got++) {
            VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                       .allocationSize = (VkDeviceSize)1 << 30, .memoryTypeIndex = (uint32_t)type};
            if (vkAllocateMemory(dev, &ai, NULL, &blk[got]) != VK_SUCCESS) break;
        }
        const double t_alloc = now_ms() - t0, free_held = heap_free();
        for (int i = 0; i < got; i++) vkFreeMemory(dev, blk[i], NULL);
        const double tf = now_ms();
        int ns = 0;
        for (;;) {
            const double t = now_ms() - tf;
            ts[ns] = t; fs[ns] = heap_free(); ns++;
            if (t >= wait_ms || ns >= maxs) break;
            usleep((useconds_t)poll_us);
        }
        printf("round %d: free %.2f GiB before, %d GiB allocated in %.1f ms (free then %.2f), freed\n",
               r, free0 / GIB, got, t_alloc, free_held / GIB);
        printf("  just after the free: %.2f GiB (%.2f of %d GiB still counted)\n",
               fs[0] / GIB, (free0 - fs[0]) / GIB, got);
        int back = -1;
        for (int i = 0; i < ns && back < 0; i++) if (fs[i] + 64.0 * 1024 * 1024 >= free0) back = i;
        if (back >= 0) printf("  back within 64 MiB of before: %.1f ms\n", ts[back]);
        else printf("  NOT back within %.0f ms (last %.2f GiB, %.2f still counted)\n", wait_ms, fs[ns - 1] / GIB, (free0 - fs[ns - 1]) / GIB);
        printf("  GiB still counted over time:");
        double last = -1;
        for (int i = 0; i < ns; i++) {
            const double held = (free0 - fs[i]) / GIB;
            const double rh = held < 0 ? 0 : (double)(int)(held * 10 + 0.5) / 10;
            if (rh != last) { printf(" %.1f@%.0fms", rh, ts[i]); last = rh; }
        }
        printf("\n  if the tier sized itself at:");
        for (int k = 0; k < ntier; k++) {
            int i = 0;
            while (i + 1 < ns && ts[i + 1] <= tier_ms[k]) i++;
            double b = fs[i] - reserve - dense - chain;
            if (b < 0) b = 0;
            printf(" %.0f ms -> %.2f GiB%s", tier_ms[k], b / GIB, b == 0 ? " (NO ROOM)" : "");
        }
        printf("\n\n");
        fflush(stdout);
        if (r < rounds) usleep((useconds_t)(gap_ms * 1000));
    }
    vkDestroyDevice(dev, NULL);
    vkDestroyInstance(inst, NULL);
    return 0;
}
