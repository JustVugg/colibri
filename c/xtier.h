/* xtier.h -- the routed-expert tier on the device the engine runs: Vulkan's (vk_tier.h) or
 * CUDA's (cu_tier.h, the same tier compiled over the CUDA backend).
 *
 * XT(issue)(layer, x, S, K, idx, taken) is vkt_issue or cut_issue, chosen once per process
 * with xt_use_cuda(1) before the tier starts (the engine's CUDA device open, its Vulkan
 * tier not wanted); every call after that goes to the same tier. The two have the same
 * declarations, and in a build without one of the backends its calls are vk_tier.h's or
 * cu_tier.h's stubs, so an engine writes the tier once. */
#ifndef COLI_XTIER_H
#define COLI_XTIER_H
#include "vk_tier.h"
#include "cu_tier.h"

static int g_xt_cuda;   /* one per engine translation unit: the engine is one */
static inline void xt_use_cuda(int on) { g_xt_cuda = on != 0; }
static inline int xt_cuda(void) { return g_xt_cuda; }
#define XT(f) (g_xt_cuda ? cut_##f : vkt_##f)
/* the tag of the engine's own tier lines */
#define XT_TAG (g_xt_cuda ? "[CUDA]" : "[VK]")
#endif
