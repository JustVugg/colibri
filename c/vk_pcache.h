/* The compute pipelines' cache on disk (backend_vulkan.c). Every pipeline of the backend
 * and of the dense chain (vk_chain.c) is made through coli_vk_pipe_create, against one
 * VkPipelineCache per device, which starts from the file the last run left and is written
 * back after every pipeline that took long to make, and when the device goes away. A
 * driver that compiles a big shader slowly then does it once: AMD's Windows driver took
 * about 20 minutes for each chain_moe.spv pipeline on a Radeon RX 580, at every start
 * (#2777). The file is one per device and driver (vendor, device, driver version and
 * the driver's cache UUID in its name), and its header is checked against the device
 * before the driver sees it.
 *
 * COLI_VK_PIPELINE_CACHE: unset, the files go to %LOCALAPPDATA%\colibri\vk-pipelines on
 * Windows and $XDG_CACHE_HOME/colibri/vk-pipelines (~/.cache) elsewhere; a directory
 * puts them there; 0 keeps no cache. */
#ifndef COLI_VK_PCACHE_H
#define COLI_VK_PCACHE_H
#include "vk_load.h"

/* vkCreateComputePipelines of one pipeline through the device's cache. `what` names it
 * for a line every minute while a build runs past 10 s (NULL: no line). */
VkResult coli_vk_pipe_create(VkDevice dev, const VkComputePipelineCreateInfo *ci, VkPipeline *pipe, const char *what);
/* The file of the device's cache ("" without one), for tests and the lines above. */
const char *coli_vk_pcache_path(VkDevice dev);
#endif
