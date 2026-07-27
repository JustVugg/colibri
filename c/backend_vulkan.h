#ifndef COLIBRI_BACKEND_VULKAN_H
#define COLIBRI_BACKEND_VULKAN_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define COLI_VK_MAX_DEVICES 16
#define COLI_VK_DEVICE_NAME_MAX 256

typedef struct ColiVkBuffer ColiVkBuffer;

typedef struct {
    int ordinal;                 /* Vulkan physical-device ordinal. */
    uint32_t vendor_id;
    uint32_t device_id;
    uint32_t api_version;
    uint32_t driver_version;
    uint32_t device_type;
    size_t device_local_bytes;   /* Sum of device-local memory heaps. */
    char name[COLI_VK_DEVICE_NAME_MAX];
} ColiVkDeviceInfo;

/*
 * Initialize one or more Vulkan devices for transfer-only expert caching.
 * `devices` contains Vulkan physical-device ordinals. When devices==NULL or
 * count==0, ordinal 0 is selected. No compute shaders or vendor extensions are
 * required: the backend deliberately targets Vulkan 1.0 core transfer support.
 */
int coli_vk_init(const int *devices, int count);
void coli_vk_shutdown(void);
int coli_vk_device_count(void);
int coli_vk_device_at(int index);
int coli_vk_device_info(int device, ColiVkDeviceInfo *info);

/*
 * `free_bytes` is a conservative backend-local estimate: device-local heap
 * capacity minus allocations owned by this backend. Core Vulkan 1.0 has no
 * portable process-wide free-memory query, so other applications/allocators
 * are intentionally not guessed here.
 */
int coli_vk_mem_info(int device, size_t *free_bytes, size_t *total_bytes);

/* Allocate a device-local transfer buffer and upload `bytes` from host memory. */
int coli_vk_buffer_upload(ColiVkBuffer **buffer, const void *src,
                          size_t bytes, int device);

/* Replace all bytes in an existing buffer without reallocating its VRAM slot. */
int coli_vk_buffer_update(ColiVkBuffer *buffer, const void *src, size_t bytes);

/* Download bytes from a device-local cache buffer into host memory. */
int coli_vk_buffer_download(const ColiVkBuffer *buffer, void *dst, size_t bytes);

void coli_vk_buffer_free(ColiVkBuffer *buffer);
size_t coli_vk_buffer_bytes(const ColiVkBuffer *buffer);
int coli_vk_buffer_device(const ColiVkBuffer *buffer);

/* Aggregate cache statistics. Any output pointer may be NULL. */
void coli_vk_stats(size_t *buffer_count, size_t *buffer_bytes,
                   uint64_t *uploads, uint64_t *downloads,
                   uint64_t *uploaded_bytes, uint64_t *downloaded_bytes);

/* Thread-local description of the most recent failure on the calling thread. */
const char *coli_vk_last_error(void);

#ifdef __cplusplus
}
#endif

#endif
