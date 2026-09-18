/* int8-expert scenario for the Qwen3.6 Vulkan tier gate: the same three
 * checks as test_qwen36_tier_vk.c, run against raw int8-promoted experts
 * (fmt=1, no packing) instead of packed int4. The Vulkan backend cannot be
 * re-initialised in-process (arenas outlive coli_vk_shutdown), so this runs
 * as a separate executable rather than a second qt_init() call alongside
 * the int4 scenario. */
#define TIER_VK_INT8 1
#include "test_qwen36_tier_vk.c"
