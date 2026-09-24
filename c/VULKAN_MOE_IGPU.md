# Vulkan MoE GEMV backend for integrated / AMD GPUs (DRAFT)

A self-contained Vulkan compute backend for the routed-expert GEMVs used by
the `qwen36` engine (Qwen3.6-35B-A3B). This PR isolates the Vulkan path from
the engine so it can be reviewed on its own.

## What it does

- `vulkan_gemv.c` (+ `vulkan_gemv.h`, `vulkan_core.h`, `vulkan_gemv_spv.h`,
  `vulkan_gemv_int4_spv.h`, `vulkan_gemv_idp_spv.h`) implements a Vulkan
  storage-buffer GEMV for the MoE expert forward.
- **int4 weights** are unpacked in-shader (2 nibbles/byte) with a plain
  `float` GEMV, so it runs on drivers whose int8 / dot-product compiler path
  is broken (e.g. AMD Radeon 780M `0x800184` segfaults on `OpSDotKHR`).
- An **int8 float path** and an optional **`OpSDotKHR` IDP shader** are also
  provided; the IDP shader is blacklisted on known-broken drivers.
- The engine probes for a Vulkan compute device at startup; if `vg_init()`
  fails or is disabled (`COLIBRI_GPU=0`), it silently falls back to the CPU
  MoE path.

## Relation to #418

`#418` (steve-m) is the project's Vulkan backend, oriented at **CUDA /
NVIDIA** (expert tier + dense + MLA attention). This backend targets
**integrated / AMD GPUs** — shared system memory, no CUDA, no dedicated
VRAM. The value proposition is complementary (iGPU vs dGPU), not a competing
second backend. Happy to coordinate with @steve-m / maintainers so the repo
does not end up maintaining two Vulkan implementations if unification is
preferred.

## Status: DRAFT

- Depends on **#712** (qwen36 engine) as the caller. Not yet wired into `dev`'s
  build, because the engine itself is landing via #712. The integration point
  is `vg_init()` / `vg_expert_ensure()` called from the engine's MoE forward.
- Until #712 lands, these files are additive (no change to existing `dev`
  build/targets).

## Known limitation (and the fix applied here)

Slot byte offsets are `uint32_t`, so a weights pool **>= 4 GB** would
wrap/alias (concretely, layers 32-39 alias layers 0-7 at `cache>=256`).
**Fix:** `vg_init()` now refuses to enable the backend and falls back to the
CPU MoE path when the pool would exceed 4 GB, instead of silently producing
wrong results. 64-bit shader offsets (via `VK_KHR_buffer_device_address`) are
a follow-up.

## Also fixed (engine side, via #712)

`qwen36.c`'s `gpu_probe()` used `VK_QUEUE_COMPUTE_BIT = 0x20` (which is
`VIDEO_DECODE_BIT`); the correct value is `0x00000002u`. That meant the GPU
was never probed on NVIDIA/Linux. Tracked in #712.
