/* vk_memtype.h -- the memory type of a Vulkan buffer, chosen by what the buffer is used for,
 * and the size, budget and room of the device memory (backend_vulkan.c). The functions depend
 * on the memory properties of the device only, so tests/test_vk_memtype.c checks them on tables.
 *
 * An integrated GPU with a heap of system memory gets host-visible, coherent system memory
 * instead of the BIOS-reserved VRAM (the carve-out), which also contains the GPU page tables.
 * A system memory allocation that runs out of memory is retried in a device-local type. */
#ifndef COLI_VK_MEMTYPE_H
#define COLI_VK_MEMTYPE_H

#include <stdint.h>
#include <vulkan/vulkan.h>

typedef enum {
    COLI_VK_MEM_GPU_ONLY,     /* only the GPU reads and writes it */
    COLI_VK_MEM_CPU_WRITES,   /* the CPU writes it, the GPU reads it */
    COLI_VK_MEM_CPU_READS     /* the CPU reads it back */
} ColiVkMemUse;

/* True where the heap is part of the memory used by the GPU (the device memory): a device-local
 * heap, or any heap of an integrated GPU, which share one memory with the CPU. */
static inline int vk_heap_is_device_memory(const VkPhysicalDeviceMemoryProperties *m, int integrated,
                                           uint32_t heap) {
    return integrated || (m->memoryHeaps[heap].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT);
}

/* The first type with every flag of `want` and none of `avoid`, or -1. */
static inline int vk_first_type(const VkPhysicalDeviceMemoryProperties *m, VkMemoryPropertyFlags want,
                                VkMemoryPropertyFlags avoid) {
    for (uint32_t i = 0; i < m->memoryTypeCount; i++) {
        const VkMemoryPropertyFlags f = m->memoryTypes[i].propertyFlags;
        if ((f & want) == want && !(f & avoid)) return (int)i;
    }
    return -1;
}

/* A host-visible, coherent type outside the device-local heaps, cached or write-combined, else
 * the other kind. Lazily allocated, protected and AMD-specific types (device-coherent,
 * device-uncached) are skipped. */
static inline int vk_system_memtype(const VkPhysicalDeviceMemoryProperties *m, int cached) {
    const VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    const VkMemoryPropertyFlags avoid =
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT |
        VK_MEMORY_PROPERTY_PROTECTED_BIT | VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD |
        VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD;
    const int t = cached ? vk_first_type(m, want | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, avoid)
                         : vk_first_type(m, want, avoid | VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
    return t >= 0 ? t : vk_first_type(m, want, avoid);
}

/* The memory type for a buffer used as `use`, or -1 when the device has none. */
static inline int vk_pick_memtype(const VkPhysicalDeviceMemoryProperties *m, int integrated, ColiVkMemUse use) {
    if (integrated) {
        /* Write-combined memory is enough for what the CPU writes and never reads back. The dense
         * chain also maps buffers used by the GPU alone and may read them on the CPU, so those
         * get cached memory. */
        const int t = vk_system_memtype(m, use != COLI_VK_MEM_CPU_WRITES);
        if (t >= 0) return t;
    }
    /* The CPU writes through a host-visible, device-local type where there is one. On a discrete
     * GPU that is write-combined VRAM, which the CPU reads at about 40 MB/s, so what it reads
     * back needs a host-cached type. */
    const VkMemoryPropertyFlags visible = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    int t = -1;
    switch (use) {
    case COLI_VK_MEM_GPU_ONLY:
        t = vk_first_type(m, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0);
        break;
    case COLI_VK_MEM_CPU_READS:
        t = vk_first_type(m, visible | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, 0);
        break;
    case COLI_VK_MEM_CPU_WRITES:
        break;
    }
    if (t < 0) t = vk_first_type(m, visible | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0);
    return t >= 0 ? t : vk_first_type(m, visible, 0);
}

/* The type to retry an allocation in when `type`, a system memory type picked for an integrated
 * GPU, is out of memory: the first host-visible, coherent, device-local type, or -1. */
static inline int vk_retry_memtype(const VkPhysicalDeviceMemoryProperties *m, int integrated, uint32_t type) {
    if (!integrated || (m->memoryTypes[type].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) return -1;
    return vk_first_type(m, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0);
}

/* The size of the device memory: the largest device-local heap, or all heaps of an integrated GPU
 * together. */
static inline VkDeviceSize vk_device_memory_size(const VkPhysicalDeviceMemoryProperties *m, int integrated) {
    VkDeviceSize size = 0;
    for (uint32_t h = 0; h < m->memoryHeapCount; h++) {
        if (!vk_heap_is_device_memory(m, integrated, h)) continue;
        const VkDeviceSize s = m->memoryHeaps[h].size;
        size = integrated ? size + s : s > size ? s : size;
    }
    return size;
}

/* The usage and the budget of the device memory from the usage and the budget reported for each
 * heap. On an integrated GPU, a heap whose usage exceeds the budget reported for it counts the
 * usage as the budget. RADV caps the budget of the system heap at the size of the heap, and a
 * tier can contain more than that, so the heap would lower the free memory of the other heaps. */
static inline void vk_device_memory_budget(const VkPhysicalDeviceMemoryProperties *m, int integrated,
                                           const VkDeviceSize *usage, const VkDeviceSize *budget,
                                           VkDeviceSize *used_sum, VkDeviceSize *budget_sum) {
    *used_sum = *budget_sum = 0;
    for (uint32_t h = 0; h < m->memoryHeapCount; h++) {
        if (!vk_heap_is_device_memory(m, integrated, h)) continue;
        *used_sum += usage[h];
        *budget_sum += integrated && usage[h] > budget[h] ? usage[h] : budget[h];
    }
}

/* The entry of a heap in the table of vk_batch_room. The heaps of an integrated GPU are parts of
 * one memory and share entry 0. */
static inline uint32_t vk_batch_slot(int integrated, uint32_t heap) { return integrated ? 0 : heap; }

/* The room for an expert batch in each heap: half of the budget left in the heap, or an eighth of
 * its size where there is no budget (NULL usage and budget). */
static inline void vk_batch_room(const VkPhysicalDeviceMemoryProperties *m, int integrated,
                                 const VkDeviceSize *usage, const VkDeviceSize *budget, VkDeviceSize *room) {
    for (uint32_t h = 0; h < VK_MAX_MEMORY_HEAPS; h++) room[h] = 0;
    for (uint32_t h = 0; h < m->memoryHeapCount; h++) {
        VkDeviceSize r = m->memoryHeaps[h].size / 8;
        if (budget) r = budget[h] > usage[h] ? (budget[h] - usage[h]) / 2 : 0;
        room[vk_batch_slot(integrated, h)] += r;
    }
}

#endif
