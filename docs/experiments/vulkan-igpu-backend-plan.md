# colibri Vulkan iGPU backend — plan (HX 370 / Radeon 890M, portable)

Date: 2026-07-18. Goal: make colibri run expert matmul + attention on the
integrated GPU instead of (or alongside) CPU, starting on the AMD Ryzen AI HX
370 (Radeon 890M, RDNA 3.5 / GFX1150) and portable to ANY Vulkan device.

## STATUS (2026-07-18)
- [x] Confirmed Vulkan sees the 890M: `deviceName = AMD Radeon 890M Graphics
  (RADV GFX1150)`, `apiVersion=1.4.318`, `shaderFloat16`/`shaderInt8` available,
  64 KB shared mem, timestamp queries available.
- [x] **PoC PASSED**: `c/vk_poc/vk_int4_poc.c` runs int4-unpack + group-scale
  dequant + matmul on the 890M via RADV. GPU vs CPU-dequant:
  `max_abs_err=3.66e-07, cosine=1.000000` — the math path is proven correct on
  this GPU. Build: `cc vk_int4_poc.c -o vk_int4_poc -lvulkan -lm`, run with
  `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json ./vk_int4_poc`.
- [ ] Implement `backend_vulkan.c` exposing the `coli_cuda_*` ABI (matmul,
  expert_mlp, expert_group, pipe_*).
- [ ] Wire `glm.c` dispatch to select Vulkan when `--gpu vulkan` / env set.
- [ ] End-to-end: load GLM-5.2 int4, run decode on iGPU, benchmark tok/s vs CPU-only.
- [ ] Portability: same backend runs on any Vulkan device (Intel Arc, other
  Radeon, nvk) — this is the multi-machine lever.

## Why Vulkan (not ROCm, not CUDA, not Metal)

Hardware probe on the HX 370 (2026-07-18):
- **No ROCm**: `/opt/rocm` absent, no `rocminfo`/`hipcc`. The 890M is RDNA
  (consumer), not CDNA — ROCm does not support it. ROCm is OUT.
- **No CUDA** (Linux x86, no NVIDIA). **No Metal** (not macOS).
- **Vulkan 1.4 via Mesa RADV WORKS**: `vulkaninfo` →
  `deviceName = AMD Radeon 890M Graphics (RADV GFX1150)`,
  `driverName = radv`, `apiVersion = 1.4.318`, `vendorID 0x1002`,
  `deviceID 0x150e`.
- Vulkan features available: `VK_KHR_shader_float16_int8` (int8 + float16 in
  shaders), `shaderInt16 = true`, `maxComputeSharedMemorySize = 64 KB`,
  `maxComputeWorkGroupInvocations = 1024`, `timestampComputeAndGraphics = true`.

**Decision: implement `backend_vulkan.c` exposing the existing `coli_cuda_*`
ABI against Vulkan.** Rationale:
1. It is the ONLY compute path the 890M exposes.
2. Vulkan is cross-vendor → the same backend runs on Intel Arc, other Radeon,
   and (via nvk / official) NVIDIA. This directly serves the multi-device
   benchmark goal: one backend, many GPUs. ROCm would exclude the 890M and
   lock us to CDNA.

## The backend ABI (already designed — we implement it, not invent it)

`c/backend_cuda.h` defines a backend-agnostic interface `glm.c` calls under
`#ifdef COLI_CUDA`, with a CPU fallback when the call returns 0. We mirror it
exactly in Vulkan so the engine core is untouched. Surface to implement:

- init/shutdown/device_count/mem_info/stats
- `coli_cuda_tensor_upload` (weights + FP32 group scales → VkBuffer)
- `coli_cuda_matmul` (y[S,O] = x[S,I] @ W[O,I]^T; fmt 0=f32,1=int8,2=int4,3=int2)
- `coli_cuda_expert_mlp` (fused down(silu(gate(x))*up(x)))
- `coli_cuda_shared_mlp_w4a16` (INT4 weights, FP16 acts, FP32 acc)
- `coli_cuda_expert_group` (packed same-shape experts)
- `coli_cuda_attention_absorb*` / `attention_project*` (MLA attention)
- `coli_cuda_pipe_*` resident pipeline: rmsnorm, rope, silu_mul, gemm, copy2d,
  peer_copy, sync

`glm.c` dispatch (e.g. `glm.c:996` `coli_cuda_matmul(...)` returns → fallback)
is flag-gated. We add `COLI_VULKAN` so the same symbols resolve to the Vulkan
impl. Build: compile `backend_vulkan.c` + link `libvulkan.so.1`.

## Compute strategy (per kernel)

- **INT4 matmul** (`w4a16` style): unpack 2 nibbles per byte in the shader,
  dequant with the row's FP32 group scale (128x128 groups per config's
  `weight_block_size`), multiply by FP16 activation, accumulate FP32, store
  FP16 (or FP32) output. Mirrors CUDA `TC_INT4` / `shared_mlp_w4a16`.
- **INT8 matmul** (`w8a16`): native `int8` mul (VK_KHR_shader_float16_int8) or
  uint unpack; used by the int8 MTP head.
- **Tile sizing**: 128x128 or 64x64 tiles in shared memory (64 KB budget),
  wavefront-64 (RDNA3). Aim for coalesced global loads.
- **Residency**: weights already live in the 96 GB unified RAM. Vulkan still
  needs a device VkBuffer + copy (the `pipe_upload`/`pipe_alloc` API models
  this). Because unified memory, no PCIe stall — the copy is RAM→VRAM-on-die.

## Why this moves the needle (revisited from glm52-hx370-2026-07-18.md)

Current HX 370 decode: 0.77 tok/s, 71% disk wait, 4s matmul / 24s attention
(CPU). The iGPU helps two ways:
1. **Speed**: RDNA3 compute (12 CU) >> 12 Zen5 threads for matmul. Even with
   disk-bound decode, the 4s matmul slice shrinks, and if residency rises
   (below) the matmul-dominated ceiling (~2.7 tok/s at 100% residency) climbs.
2. **Quality (the bigger win)**: unified memory means a Vulkan-resident hot
   expert set is addressed by the GPU in the SAME 96 GB. Pinning hot experts
   device-side raises hit rate toward 100% WITHOUT more RAM → kills the disk
   wait driving both slowness and the ~50% routing noise.

Caveat: disk streaming coalescing (random small expert reads underusing 7
GB/s) is a SEPARATE task; the Vulkan backend is necessary but not sufficient
for a big number. Both are tracked.

## Phase A — Proof of concept (de-risk, standalone)

Standalone `c/vk_poc/` (or a temp dir): a minimal Vulkan app that
1. instance + device on the 890M (VK_ICD_FILENAMES=radeon_icd.json),
2. uploads a small INT4 weight matrix + FP32 scales + one FP16 activation row,
3. runs a compute shader `y = x @ W^T` (int4 unpack → FP16 mul → FP32 acc),
4. downloads y, checks vs a CPU reference (max rel err, cosine sim).
Pass = math path correct on this GPU. Gate for Phase B.

## Phase B — backend_vulkan.c

Implement the ABI surface above. Start with `matmul` + `expert_mlp` +
`tensor_upload` + `pipe_*` minimal (cover the decode path); attention can
follow. Flag `COLI_VULKAN`; `glm.c` calls resolve to Vulkan impl. CPU
fallback retained for any unimplemented entry point (return 0 → engine falls
back), so we can land incrementally.

## Phase C — integrate + benchmark

Build with Vulkan, run the same fixed TEMP=0 / TOPP=0.95 benches from
glm52-hx370-2026-07-18.md, compare tok/s + hit rate + disk wait. Record in
the same doc (new section). Then the backend is portable: set
VK_ICD_FILENAMES per machine and bench across devices.

## Open risks

- RADV INT4 perf may be modest vs CUDA Tensor Cores (no native int4 TC on
  RDNA3 — manual unpack). Still >> CPU for matmul.
- The `coli_cuda_pipe_*` resident pipeline is large; landing the full ABI is
  real work. Plan lands matmul + expert_mlp first, attention later.
- Unified-memory copy semantics: confirm RADV exposes the expected behavior;
  if not, fall back to explicit staging buffers.
