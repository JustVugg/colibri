#ifndef COLIBRI_BACKEND_VULKAN_INTERNAL_H
#define COLIBRI_BACKEND_VULKAN_INTERNAL_H

#include "backend_vulkan.h"

#include <stdatomic.h>
#include <vulkan/vulkan.h>

struct ColiVkBuffer {
    VkBuffer buffer;
    VkDeviceMemory memory;
    VkDeviceSize allocation_size;
    size_t bytes;
    int device;
};

typedef struct {
    int active;
    int ordinal;
    VkPhysicalDevice physical;
    VkDevice logical;
    VkQueue queue;
    uint32_t queue_family;
    VkCommandPool command_pool;
    VkCommandBuffer command_buffer;
    VkFence fence;
    VkPhysicalDeviceProperties properties;
    VkPhysicalDeviceMemoryProperties memory_properties;
    VkBuffer staging_buffer;
    VkDeviceMemory staging_memory;
    void *staging_map;
    VkDeviceSize staging_size;
    VkMemoryPropertyFlags staging_flags;
    size_t device_heap_bytes;
    size_t allocated_bytes;
    atomic_flag transfer_lock;
} ColiVkDevice;

extern VkInstance coli_vk_instance;
extern ColiVkDevice coli_vk_devices[COLI_VK_MAX_DEVICES];
extern int coli_vk_n_devices;
extern atomic_flag coli_vk_state_lock;
extern size_t coli_vk_buffer_count_value;
extern size_t coli_vk_buffer_bytes_value;
extern uint64_t coli_vk_upload_count;
extern uint64_t coli_vk_download_count;
extern uint64_t coli_vk_uploaded_bytes_value;
extern uint64_t coli_vk_downloaded_bytes_value;

void coli_vk_lock(atomic_flag *flag);
void coli_vk_unlock(atomic_flag *flag);
int coli_vk_failf(const char *fmt, ...);
ColiVkDevice *coli_vk_device_by_ordinal(int ordinal);
uint32_t coli_vk_find_memory_type(const ColiVkDevice *device,
                                  uint32_t type_bits,
                                  VkMemoryPropertyFlags required,
                                  VkMemoryPropertyFlags preferred,
                                  VkMemoryPropertyFlags *chosen_flags);
int coli_vk_device_init(ColiVkDevice *device, VkPhysicalDevice physical,
                        int ordinal);
void coli_vk_device_destroy(ColiVkDevice *device);
int coli_vk_ensure_staging(ColiVkDevice *device, VkDeviceSize bytes);
int coli_vk_copy_buffer(ColiVkDevice *device, VkBuffer src, VkBuffer dst,
                        VkDeviceSize bytes);
int coli_vk_flush_staging(ColiVkDevice *device);
int coli_vk_invalidate_staging(ColiVkDevice *device);

#endif
