/* The functions of vk_memtype.h on memory-property tables written here, so the test needs the
 * Vulkan headers and no device:
 *   - an integrated GPU with a system heap gets host-visible, coherent system memory, write-
 *     combined for what the CPU writes and cached for the rest, on the table of a Radeon APU,
 *     on a table with the types in another order and on a table with a large device-local heap;
 *   - an integrated GPU without such a heap, a discrete GPU with and without Resizable BAR
 *     and a CPU device get the types chosen for a discrete GPU;
 *   - an allocation of system memory is retried in a device-local type on an integrated GPU only;
 *   - the size, budget and room of the device memory count every heap of an integrated GPU and
 *     the device-local heaps of the others.
 *
 *   make vk-memtype-check VK=1 */
#include "../vk_memtype.h"
#include <stdio.h>

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fails++; printf("FAIL %s:%d: ", __FILE__, __LINE__); \
                                       printf(__VA_ARGS__); printf("\n"); } } while (0)

enum {
    DL = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    HV = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
    COH = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    CACHED = VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
    AMD = VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD,
    LOCAL_HEAP = VK_MEMORY_HEAP_DEVICE_LOCAL_BIT
};
#define GIB ((VkDeviceSize)1 << 30)
typedef struct { VkDeviceSize size; VkMemoryHeapFlags flags; } Heap;
typedef struct { VkMemoryPropertyFlags flags; uint32_t heap; } Type;

static VkPhysicalDeviceMemoryProperties table(const Heap *h, uint32_t nh, const Type *t, uint32_t nt) {
    VkPhysicalDeviceMemoryProperties m = {.memoryHeapCount = nh, .memoryTypeCount = nt};
    for (uint32_t i = 0; i < nh; i++) m.memoryHeaps[i] = (VkMemoryHeap){h[i].size, h[i].flags};
    for (uint32_t i = 0; i < nt; i++) m.memoryTypes[i] = (VkMemoryType){t[i].flags, t[i].heap};
    return m;
}

/* The types picked for the three uses, in the order of the enum. */
static void expect(const char *name, const VkPhysicalDeviceMemoryProperties *m, int integrated, int gpu_only,
                   int cpu_writes, int cpu_reads) {
    const int want[] = {gpu_only, cpu_writes, cpu_reads};
    const char *use[] = {"GPU only", "CPU writes", "CPU reads"};
    for (int u = COLI_VK_MEM_GPU_ONLY; u <= COLI_VK_MEM_CPU_READS; u++) {
        const int got = vk_pick_memtype(m, integrated, (ColiVkMemUse)u);
        CHECK(got == want[u], "%s, %s: type %d, expected %d", name, use[u], got, want[u]);
    }
}

int main(void) {
    /* A Radeon 8060S under RADV: a system heap and a device-local heap that is twice as large,
     * since RADV reports two thirds of the memory of an APU as device-local, with the
     * AMD-specific types after the plain ones. */
    const Heap apu_h[] = {{35 * GIB, 0}, {69 * GIB, LOCAL_HEAP}};
    const Type apu_t[] = {{DL, 1}, {HV | COH, 0}, {DL | HV | COH, 1}, {HV | COH | CACHED, 0},
                          {HV | COH | AMD, 0}, {DL | HV | COH | AMD, 1}};
    const VkPhysicalDeviceMemoryProperties apu = table(apu_h, 2, apu_t, 6);
    expect("APU, integrated", &apu, 1, 3, 1, 3);
    expect("APU, picked as a discrete GPU", &apu, 0, 0, 2, 3);

    /* the same kind of device with the cached type first */
    const Heap apu2_h[] = {{8 * GIB, LOCAL_HEAP}, {24 * GIB, 0}};
    const Type apu2_t[] = {{DL, 0}, {DL | HV | COH, 0}, {HV | COH | CACHED, 1}, {HV | COH, 1}};
    const VkPhysicalDeviceMemoryProperties apu2 = table(apu2_h, 2, apu2_t, 4);
    expect("APU, cached type first", &apu2, 1, 2, 3, 2);

    /* system memory with one kind of type is used for every use */
    const Type cached_t[] = {{DL, 0}, {HV | COH | CACHED, 1}};
    const VkPhysicalDeviceMemoryProperties cached_only = table(apu2_h, 2, cached_t, 2);
    expect("system heap, cached type only", &cached_only, 1, 1, 1, 1);
    const Type uncached_t[] = {{DL, 0}, {HV | COH, 1}};
    const VkPhysicalDeviceMemoryProperties uncached_only = table(apu2_h, 2, uncached_t, 2);
    expect("system heap, uncached type only", &uncached_only, 1, 1, 1, 1);

    /* an integrated GPU whose one heap is device-local */
    const Heap one_h[] = {{16 * GIB, LOCAL_HEAP}};
    const Type one_t[] = {{DL, 0}, {DL | HV | COH | CACHED, 0}};
    const VkPhysicalDeviceMemoryProperties one = table(one_h, 1, one_t, 2);
    expect("integrated, one device-local heap", &one, 1, 0, 1, 1);

    /* a card with Resizable BAR */
    const Heap bar_h[] = {{16 * GIB, LOCAL_HEAP}, {32 * GIB, 0}};
    const Type bar_t[] = {{DL, 0}, {DL | HV | COH, 0}, {HV | COH, 1}, {HV | COH | CACHED, 1}};
    const VkPhysicalDeviceMemoryProperties bar = table(bar_h, 2, bar_t, 4);
    expect("discrete with Resizable BAR", &bar, 0, 0, 1, 3);

    /* a card without it: the visible VRAM is a 256 MiB window in a separate heap */
    const Heap win_h[] = {{8 * GIB, LOCAL_HEAP}, {32 * GIB, 0}, {256 << 20, LOCAL_HEAP}};
    const Type win_t[] = {{DL, 0}, {HV | COH, 1}, {DL | HV | COH, 2}, {HV | COH | CACHED, 1}};
    const VkPhysicalDeviceMemoryProperties win = table(win_h, 3, win_t, 4);
    expect("discrete without Resizable BAR", &win, 0, 0, 2, 3);

    /* Lavapipe: one type for everything */
    const Heap lvp_h[] = {{64 * GIB, LOCAL_HEAP}};
    const Type lvp_t[] = {{DL | HV | COH | CACHED, 0}};
    const VkPhysicalDeviceMemoryProperties lvp = table(lvp_h, 1, lvp_t, 1);
    expect("Lavapipe", &lvp, 0, 0, 0, 0);

    /* no host-visible type: nothing to map */
    const Type blind_t[] = {{DL, 0}};
    const VkPhysicalDeviceMemoryProperties blind = table(one_h, 1, blind_t, 1);
    expect("no host-visible type, integrated", &blind, 1, 0, -1, -1);
    expect("no host-visible type, discrete", &blind, 0, 0, -1, -1);

    /* a BIOS with a large frame buffer: the device-local heap is much larger than the system heap */
    const Heap big_h[] = {{8 * GIB, 0}, {48 * GIB, LOCAL_HEAP}};
    const VkPhysicalDeviceMemoryProperties big = table(big_h, 2, apu_t, 6);
    expect("integrated, large device-local heap", &big, 1, 3, 1, 3);

    /* the device memory */
    CHECK(vk_heap_is_device_memory(&apu, 1, 0) && vk_heap_is_device_memory(&apu, 1, 1),
          "every heap of an integrated GPU is device memory");
    CHECK(!vk_heap_is_device_memory(&apu, 0, 0) && vk_heap_is_device_memory(&apu, 0, 1),
          "only the device-local heap of another GPU is device memory");
    CHECK(vk_device_memory_size(&apu, 1) == 104 * GIB, "an integrated GPU: the heaps together");
    CHECK(vk_device_memory_size(&apu, 0) == 69 * GIB, "another GPU: the device-local heap");
    CHECK(vk_device_memory_size(&win, 0) == 8 * GIB, "another GPU: the largest device-local heap");

    /* the retry of an allocation that is out of memory: system memory types only, integrated only */
    CHECK(vk_retry_memtype(&apu, 1, 1) == 2 && vk_retry_memtype(&apu, 1, 3) == 2 &&
          vk_retry_memtype(&big, 1, 1) == 2, "system memory retries in the device-local host-visible type");
    CHECK(vk_retry_memtype(&apu, 1, 0) == -1 && vk_retry_memtype(&apu, 1, 2) == -1,
          "a device-local type has no retry");
    CHECK(vk_retry_memtype(&apu, 0, 1) == -1, "another GPU has no retry");
    CHECK(vk_retry_memtype(&uncached_only, 1, 1) == -1, "no host-visible device-local type to retry in");

    /* budget and room: the usage of the system heap exceeds the budget reported for it, as on
     * RADV once a tier is larger than the heap, and the device-local heap has room. */
    const VkDeviceSize usage[VK_MAX_MEMORY_HEAPS] = {10 * GIB, 2 * GIB};
    const VkDeviceSize budget[VK_MAX_MEMORY_HEAPS] = {6 * GIB, 30 * GIB};
    VkDeviceSize used_sum, budget_sum, room[VK_MAX_MEMORY_HEAPS];
    vk_device_memory_budget(&apu, 1, usage, budget, &used_sum, &budget_sum);
    CHECK(used_sum == 12 * GIB && budget_sum == 40 * GIB,
          "an integrated GPU: the usage of a heap over its budget is its budget (%llu used, %llu budget)",
          (unsigned long long)used_sum, (unsigned long long)budget_sum);
    vk_device_memory_budget(&apu, 0, usage, budget, &used_sum, &budget_sum);
    CHECK(used_sum == 2 * GIB && budget_sum == 30 * GIB, "another GPU: the device-local heap only");
    const VkDeviceSize over[VK_MAX_MEMORY_HEAPS] = {0, 10 * GIB};
    const VkDeviceSize cap[VK_MAX_MEMORY_HEAPS] = {0, 8 * GIB};
    vk_device_memory_budget(&apu, 0, over, cap, &used_sum, &budget_sum);
    CHECK(used_sum == 10 * GIB && budget_sum == 8 * GIB, "another GPU: a budget below the usage is kept");

    vk_batch_room(&apu, 1, usage, budget, room);
    CHECK(room[0] == 14 * GIB && room[1] == 0, "an integrated GPU: the room of all heaps is in heap 0");
    vk_batch_room(&apu, 0, usage, budget, room);
    CHECK(room[0] == 0 && room[1] == 14 * GIB, "another GPU: the room of each heap");
    vk_batch_room(&apu, 1, NULL, NULL, room);
    CHECK(room[0] == 13 * GIB && room[1] == 0, "without a budget: an eighth of the size of the heaps");
    const VkDeviceSize big_usage[VK_MAX_MEMORY_HEAPS] = {9 * GIB};
    const VkDeviceSize big_budget[VK_MAX_MEMORY_HEAPS] = {8 * GIB, 40 * GIB};
    vk_batch_room(&big, 1, big_usage, big_budget, room);
    CHECK(room[0] == 20 * GIB, "a large device-local heap: its room is counted when the system heap is full");
    CHECK(vk_batch_slot(1, 1) == 0 && vk_batch_slot(0, 1) == 1, "the heaps of an integrated GPU share entry 0");

    if (fails) {
        printf("FAILED: %d\n", fails);
        return 1;
    }
    printf("vk_memtype: all cases passed\n");
    return 0;
}
