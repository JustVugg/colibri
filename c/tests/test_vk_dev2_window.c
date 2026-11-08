/* Exercise the real initialization paths on two logical devices. Lavapipe is
 * sufficient: the existing probe override emulates a restricted mapped window.
 * Include the backend to inspect uploader state without adding a public API. */
#include "../backend_vulkan.c"
#include <assert.h>

/* The probe answers only for a device that is not staged already (docs/vulkan.md):
 * a discrete card whose host-visible device-local window is under a quarter of its
 * device-local heap (no Resizable BAR) stages every upload whatever the probe says.
 * There the probe's cases can only check that both devices follow the heaps. *window
 * is that window's heap, 0 when the card has none. */
static int heaps_stage(VkPhysicalDevice phys, VkDeviceSize *window) {
    *window = 0;
    if (vk_integrated(phys)) return 0;
    VkPhysicalDeviceMemoryProperties m;
    vkGetPhysicalDeviceMemoryProperties(phys, &m);
    VkDeviceSize dl = 0, hv = 0;
    for (uint32_t i = 0; i < m.memoryHeapCount; i++)
        if ((m.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) && m.memoryHeaps[i].size > dl)
            dl = m.memoryHeaps[i].size;
    int mt = pick_memtype(phys, COLI_VK_MEM_CPU_WRITES, ~0u);
    if (mt >= 0 && (m.memoryTypes[mt].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
        hv = m.memoryHeaps[m.memoryTypes[mt].heapIndex].size;
    *window = hv;
    return dl > 0 && hv * 4 < dl;
}

int main(int argc, char **argv) {
    const char *spv = argc > 1 ? argv[1] : "shaders/qmatmul.spv";
    static const struct { const char *probe, *forced; int staged; } cases[] = {
        {"1", NULL, 1}, {"100000000", NULL, 0},
        {"1", "0", 0}, {"100000000", "1", 1}
    };
    unsetenv("COLI_VK_HOST_VISIBLE_CAP_MB");
    unsetenv("COLI_VK_WINDOW_ASK_MB");
    int heaps = -1;
    VkDeviceSize window = 0;
    for (size_t i = 0; i < sizeof cases / sizeof *cases; i++) {
        /* The shared probe still selects the primary's path correctly. */
        setenv("COLI_VK_WINDOW_PROBE_MB", cases[i].probe, 1);
        if (cases[i].forced) setenv("COLI_VK_STAGED", cases[i].forced, 1);
        else unsetenv("COLI_VK_STAGED");
        assert(coli_vk_init(spv));
        if (heaps < 0) {
            heaps = heaps_stage(G.phys, &window);
            if (heaps) puts("the heaps stage this device's uploads (no Resizable BAR): "
                            "the probe's cases check that both devices follow them");
        }
        const int staged = cases[i].forced ? cases[i].staged : cases[i].staged || heaps;
        assert(g_up[0].on == staged);
        assert(!g_up[0].later);
        coli_vk_shutdown();
        /* The primary stays mapped; only the secondary sees the probe answer. */
        setenv("COLI_VK_STAGED", "0", 1);
        assert(coli_vk_init(spv));
        setenv("COLI_VK_WINDOW_PROBE_MB", cases[i].probe, 1);
        if (cases[i].forced) setenv("COLI_VK_STAGED", cases[i].forced, 1);
        else unsetenv("COLI_VK_STAGED");
        assert(coli_vk_init_dev2(spv, 0));
        assert(!g_up[0].on);
        assert(g_up[1].on == staged);
        assert(!g_up[1].later);
        if (staged) {
            assert(g_up[1].dev == G2.dev && g_up[1].shared);
        } else {
            assert(!g_up[1].dev && !g_up[1].shared && !g_up[1].nfams);
        }
        if (!staged && heaps && window < VK_WBLOCK) {
            /* mapped memory forced on a window smaller than one pool block: the block
             * does not fit, on either device, so there is nothing to upload into */
            printf("COLI_VK_STAGED=0: the %llu MiB window holds no %zu MiB pool block, no upload\n",
                   (unsigned long long)(window >> 20), VK_WBLOCK >> 20);
            coli_vk_shutdown();
            continue;
        }
        ColiVkTensor *t = NULL;
        uint8_t *rows; size_t stride; float *scales;
        assert(coli_vk_tier_tensor_dev(1, &t, 1, 32, 4, 0, &rows, &stride, &scales));
        memset(rows, 1, stride * 4);
        for (int j = 0; j < 4; j++) scales[j] = 1.f;
        assert(coli_vk_tensor_commit(&t, 1));
        coli_vk_tensor_free(t);
        coli_vk_shutdown();
    }
    puts("dev2 mapped-window selection and uploads: PASS");
    return 0;
}
