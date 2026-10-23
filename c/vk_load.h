/* The Vulkan functions the backend calls (backend_vulkan.c, vk_chain.c), taken from the
 * loader at run time instead of linked: no binary built with VK=1 names libvulkan.so.1
 * or vulkan-1.dll among the libraries it needs, so it starts on a machine with no Vulkan
 * at all (the release archives are built with VK=1) and runs on the CPU there.
 * coli_vk_load (backend_vulkan.c) opens the loader the first time a device is asked for
 * and fills one pointer per function below; the macros at the end send every call in the
 * files that include this header through them. A function the backend starts calling
 * joins COLI_VK_FUNCS and gets its #define, or the build fails (no prototype, and no
 * library to link it from). */
#ifndef COLI_VK_LOAD_H
#define COLI_VK_LOAD_H
#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <vulkan/vulkan.h>

#define COLI_VK_FUNCS(X) \
    X(vkAllocateCommandBuffers) X(vkAllocateDescriptorSets) X(vkAllocateMemory) \
    X(vkBeginCommandBuffer) X(vkBindBufferMemory) X(vkCmdBindDescriptorSets) \
    X(vkCmdBindPipeline) X(vkCmdCopyBuffer) X(vkCmdDispatch) X(vkCmdFillBuffer) \
    X(vkCmdPipelineBarrier) X(vkCmdPushConstants) X(vkCmdResetQueryPool) \
    X(vkCmdWriteTimestamp) X(vkCreateBuffer) X(vkCreateCommandPool) \
    X(vkCreateComputePipelines) X(vkCreateDescriptorPool) X(vkCreateDescriptorSetLayout) \
    X(vkCreateDevice) X(vkCreateFence) X(vkCreateInstance) X(vkCreatePipelineLayout) \
    X(vkCreateQueryPool) X(vkCreateShaderModule) X(vkDestroyBuffer) X(vkDestroyCommandPool) \
    X(vkDestroyDescriptorPool) X(vkDestroyDescriptorSetLayout) X(vkDestroyDevice) \
    X(vkDestroyFence) X(vkDestroyInstance) X(vkDestroyPipeline) X(vkDestroyPipelineLayout) \
    X(vkDestroyQueryPool) X(vkDestroyShaderModule) X(vkDeviceWaitIdle) X(vkEndCommandBuffer) \
    X(vkEnumerateDeviceExtensionProperties) X(vkEnumeratePhysicalDevices) \
    X(vkFreeCommandBuffers) X(vkFreeDescriptorSets) X(vkFreeMemory) \
    X(vkGetBufferDeviceAddress) X(vkGetBufferMemoryRequirements) X(vkGetDeviceProcAddr) \
    X(vkGetDeviceQueue) X(vkGetFenceStatus) X(vkGetInstanceProcAddr) \
    X(vkGetPhysicalDeviceFeatures2) X(vkGetPhysicalDeviceMemoryProperties) \
    X(vkGetPhysicalDeviceMemoryProperties2) X(vkGetPhysicalDeviceProperties) \
    X(vkGetPhysicalDeviceProperties2) X(vkGetPhysicalDeviceQueueFamilyProperties) \
    X(vkGetQueryPoolResults) X(vkMapMemory) X(vkQueueSubmit) X(vkResetCommandBuffer) \
    X(vkResetDescriptorPool) X(vkResetFences) X(vkUnmapMemory) X(vkUpdateDescriptorSets) \
    X(vkWaitForFences)

#define COLI_VK_EXTERN(f) extern PFN_##f coli_##f;
COLI_VK_FUNCS(COLI_VK_EXTERN)
#undef COLI_VK_EXTERN

/* Open the loader (COLI_VK_LOADER, else the system's) and take every function above:
 * 1 when all are there; 0 with a line saying what is missing, every later call too. */
int coli_vk_load(void);

#define vkAllocateCommandBuffers coli_vkAllocateCommandBuffers
#define vkAllocateDescriptorSets coli_vkAllocateDescriptorSets
#define vkAllocateMemory coli_vkAllocateMemory
#define vkBeginCommandBuffer coli_vkBeginCommandBuffer
#define vkBindBufferMemory coli_vkBindBufferMemory
#define vkCmdBindDescriptorSets coli_vkCmdBindDescriptorSets
#define vkCmdBindPipeline coli_vkCmdBindPipeline
#define vkCmdCopyBuffer coli_vkCmdCopyBuffer
#define vkCmdDispatch coli_vkCmdDispatch
#define vkCmdFillBuffer coli_vkCmdFillBuffer
#define vkCmdPipelineBarrier coli_vkCmdPipelineBarrier
#define vkCmdPushConstants coli_vkCmdPushConstants
#define vkCmdResetQueryPool coli_vkCmdResetQueryPool
#define vkCmdWriteTimestamp coli_vkCmdWriteTimestamp
#define vkCreateBuffer coli_vkCreateBuffer
#define vkCreateCommandPool coli_vkCreateCommandPool
#define vkCreateComputePipelines coli_vkCreateComputePipelines
#define vkCreateDescriptorPool coli_vkCreateDescriptorPool
#define vkCreateDescriptorSetLayout coli_vkCreateDescriptorSetLayout
#define vkCreateDevice coli_vkCreateDevice
#define vkCreateFence coli_vkCreateFence
#define vkCreateInstance coli_vkCreateInstance
#define vkCreatePipelineLayout coli_vkCreatePipelineLayout
#define vkCreateQueryPool coli_vkCreateQueryPool
#define vkCreateShaderModule coli_vkCreateShaderModule
#define vkDestroyBuffer coli_vkDestroyBuffer
#define vkDestroyCommandPool coli_vkDestroyCommandPool
#define vkDestroyDescriptorPool coli_vkDestroyDescriptorPool
#define vkDestroyDescriptorSetLayout coli_vkDestroyDescriptorSetLayout
#define vkDestroyDevice coli_vkDestroyDevice
#define vkDestroyFence coli_vkDestroyFence
#define vkDestroyInstance coli_vkDestroyInstance
#define vkDestroyPipeline coli_vkDestroyPipeline
#define vkDestroyPipelineLayout coli_vkDestroyPipelineLayout
#define vkDestroyQueryPool coli_vkDestroyQueryPool
#define vkDestroyShaderModule coli_vkDestroyShaderModule
#define vkDeviceWaitIdle coli_vkDeviceWaitIdle
#define vkEndCommandBuffer coli_vkEndCommandBuffer
#define vkEnumerateDeviceExtensionProperties coli_vkEnumerateDeviceExtensionProperties
#define vkEnumeratePhysicalDevices coli_vkEnumeratePhysicalDevices
#define vkFreeCommandBuffers coli_vkFreeCommandBuffers
#define vkFreeDescriptorSets coli_vkFreeDescriptorSets
#define vkFreeMemory coli_vkFreeMemory
#define vkGetBufferDeviceAddress coli_vkGetBufferDeviceAddress
#define vkGetBufferMemoryRequirements coli_vkGetBufferMemoryRequirements
#define vkGetDeviceProcAddr coli_vkGetDeviceProcAddr
#define vkGetDeviceQueue coli_vkGetDeviceQueue
#define vkGetFenceStatus coli_vkGetFenceStatus
#define vkGetInstanceProcAddr coli_vkGetInstanceProcAddr
#define vkGetPhysicalDeviceFeatures2 coli_vkGetPhysicalDeviceFeatures2
#define vkGetPhysicalDeviceMemoryProperties coli_vkGetPhysicalDeviceMemoryProperties
#define vkGetPhysicalDeviceMemoryProperties2 coli_vkGetPhysicalDeviceMemoryProperties2
#define vkGetPhysicalDeviceProperties coli_vkGetPhysicalDeviceProperties
#define vkGetPhysicalDeviceProperties2 coli_vkGetPhysicalDeviceProperties2
#define vkGetPhysicalDeviceQueueFamilyProperties coli_vkGetPhysicalDeviceQueueFamilyProperties
#define vkGetQueryPoolResults coli_vkGetQueryPoolResults
#define vkMapMemory coli_vkMapMemory
#define vkQueueSubmit coli_vkQueueSubmit
#define vkResetCommandBuffer coli_vkResetCommandBuffer
#define vkResetDescriptorPool coli_vkResetDescriptorPool
#define vkResetFences coli_vkResetFences
#define vkUnmapMemory coli_vkUnmapMemory
#define vkUpdateDescriptorSets coli_vkUpdateDescriptorSets
#define vkWaitForFences coli_vkWaitForFences
#endif
