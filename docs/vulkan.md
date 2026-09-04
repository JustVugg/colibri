# Vulkan backend (any GPU with a Vulkan 1.2 driver)

colibrì includes an opt-in Vulkan compute backend that runs the whole GLM
decode compute path on any GPU a Vulkan driver can see — no CUDA, no ROCm.
That includes cards the vendor stacks have dropped (ROCm 7 removed Polaris:
an RX 580 runs here via RADV) and, measured on an RX 9070 (RDNA4), it is
*faster* than the ROCm/HIP backend on the same card.

```bash
cd c
make glm VK=1                # needs libvulkan + glslc (shaderc) for the shaders
COLI_VULKAN=1 COLI_VK_DENSE=1 COLI_VK_ATTN=1 \
PIN=<model>/.coli_usage PIN_GB=0 COLI_NO_OMP_TUNE=1 \
./coli run "Hello" --topp 0.7
```

Requirements: `libvulkan` and a Vulkan **1.2** ICD with
`GL_KHR_shader_subgroup_arithmetic` (any Mesa RADV, AMDVLK, NVIDIA or Intel
ANV driver from the last several years), plus `glslc` at build time. The
backend picks the most capable physical device (discrete > integrated) and
degrades to the CPU path on any failure — a wedged GPU can slow a run, never
corrupt it.

Polaris/gfx803 validated on real hardware (RX 580 8 GB, Mesa 25.2.8 RADV, no
Resizable BAR, GPU clocks not pinned — no root to set
`power_dpm_force_performance_level`; commit bb16ab3): the shaders' dynamic
subgroup sizes ran unmodified, wave64-safe by construction. Qwen3.6-35B-A3B
int4-gs64, 64-token decode: staged uploads 5.99 tok/s vs the mapped path
(`COLI_VK_STAGED=0`) 2.44 tok/s; frozen-heat runs were token-identical to
each other and to the CPU-only baseline.

Set `COLI_NO_OMP_TUNE=1` on multi-core boxes: the engine's OMP self-tune
(active spin-wait) is skipped under `COLI_CUDA`/`COLI_METAL` but not under
Vulkan, and spinning worker threads starve the async I/O pool (measured
CPU expert bandwidth 28 → 5 GB/s without it).

**Resizable BAR is faster; staged uploads make it optional.** The weight tiers
prefer HOST_VISIBLE|DEVICE_LOCAL memory. With ReBAR disabled that combination
only exists in a ~256 MB BAR window, and the driver silently places everything
beyond it in system RAM — the tier then *reports* resident experts while every
access crosses PCIe, slower than the CPU path (measured 0.11 vs 0.24 tok/s
either side of the BIOS toggle on an RX 9070 XT). When the backend sees a small
host-visible slice it now switches to **staged uploads**: resident weights go
to plain DEVICE_LOCAL memory through a host staging buffer and
`vkCmdCopyBuffer` (`[VK] weights: staged device-local uploads` in the banner).
`COLI_VK_STAGED=1` forces the staged path on any card, `=0` forces the mapped
path. Each new device-local block is filled once on creation: without that
first GPU-side write, results computed from a fresh block differ slightly and
non-deterministically (measured on an RX 580; the cause is not yet
identified). In the correctness harness's batched int4 matmuls, the staged
path ran roughly 4× faster than the mapped path on the RX 580 (0.17 vs 0.66
ms/matmul) — a harness-only number for the kernel itself, not end-to-end
throughput. Enable Resizable BAR / Smart Access Memory in the BIOS when you
can; it removes the copy at warmstart. Unified-memory APUs have no
device-local-only memory and are unaffected either way.

The compiled shaders are found via `COLI_VK_SHADERS` (either the
`qmatmul.spv` file or the directory holding the `.spv` set); unset, the
engine looks in `shaders/` next to the binary, then relative to the CWD.

## What runs on the GPU

| Piece | Env | Mechanism |
|---|---|---|
| Routed experts (hot set) | `COLI_VK_EXPERTS=N` (default 320) | Top-N experts by `.coli_usage` heat uploaded **once at startup** into a VRAM registry; at decode they are served from VRAM with **no RAM slot, no disk read, no prefetch**, as one async fused batch (gate+up+silu→down, hidden on-device) overlapped with the CPU computing the remaining experts. Shown as the `vk` bucket in the hit-rate line. |
| Dense projections | `COLI_VK_DENSE=1` | q_a+kv_a fused into one submit, q_b, o; shared expert as a single fused expert-group submit. Resident int4/int8 weights upload once. |
| MLA attention core | `COLI_VK_ATTN=1` | One dispatch per layer: absorbed query, scores over the KV window, softmax, weighted latent, value rows, **fused with the o-projection** (the context vector never leaves the GPU). The latent/rope KV lives in a persistent per-layer device mirror, appended ~2.3 KB/token/layer with the same invalidation points as the CUDA KV shadow. |

The `PIN_GB=0` (with `PIN` still set) in the example is deliberate: the VRAM
registry holds the same hot experts a RAM pin would, so the pin's RAM is
better spent on the adaptive LRU cache. Keep `PIN` set so AUTOPIN does not
re-pin from history.

## Correctness

- `gcc -O3 -DVK_TEST backend_vulkan.c -o test_vk -lvulkan -lm && ./test_vk
  shaders/qmatmul.spv` runs a CPU-reference exactness harness over every
  primitive (GEMV int4/int8 across shapes incl. the long-row o-projection,
  fused gate+up, the full expert group sync and async, the matmul pair, and
  the absorb attention core incl. causal S=2, kv_start windows, int8, and
  long-context cases). Typical maxrel ~1e-5..2e-3 (fp32 reduction order).
- Engine-level: greedy decode with the full stack matches the pure-CPU
  engine token-for-token on the validation prompt.
- int4 weights decode as offset-binary (nibble−8), byte-identical layout to
  the CPU path — no repacking.

## Measured performance (AMD RX 9070, RDNA4, RADV/Mesa 26.1)

Expert-MLP primitive (K experts, int4 6144→2048→6144, per-call incl. readback):
Vulkan **0.11–0.13 ms/expert** vs the production ROCm/HIP expert group
**0.179 ms/expert** — ~35% faster. The decode MLA attention core runs 3.7×
faster than the HIP kernel on the same card. End-to-end GLM-5.2 (744B int4,
NVMe-streamed) decode on a 12-core Zen2 + RX 9070 box: Vulkan
**1.7–1.8 tok/s** (64-token) / **1.6** (256-token) / **1.58 sustained**
(512-token) vs the HIP backend at 1.5–1.55 on identical settings.
The two write-combined-memory rules that make this possible: buffers the CPU
reads back must be HOST_CACHED (ReBAR VRAM reads at ~40 MB/s otherwise), and
everything else lives HOST_VISIBLE|DEVICE_LOCAL.

## Benchmarking against other backends

Two defaults will silently skew any Vulkan-vs-CUDA/HIP comparison:

- **MTP speculation**: CUDA/HIP builds disable model drafts by default
  (`DRAFT` auto-resolves to 0 under `COLI_CUDA=1`, see #163), while CPU and
  Vulkan runs keep `DRAFT=3`. The arms then execute different decode loops —
  the speculative arm routes ~2× the expert positions per emitted token
  (rejected draft positions still pay their expert I/O), which dominates on
  storage-bound boxes. Output is identical either way (greedy verify is
  lossless), so nothing looks wrong. Pin `DRAFT=0` (or `DRAFT=3
  COLI_CUDA_MTP=1`) explicitly on **both** arms.
- **GPU clocks**: decode dispatches are microsecond bursts that never ramp
  DPM on their own; the memory clock can sit parked through an entire run.
  Pin `power_dpm_force_performance_level=high` (both arms) or disclose it.

Also note `experts loaded/token` in the run stats counts *routed positions*
(including rejected speculative ones) before any cache/tier is consulted —
it does not fall when the VK tier serves a hit; the `vk` bucket in the
hit-rate line is the tier-effectiveness number.

## Limits and future work

- Decode-focused: the expert tier and attention core serve `S<=4`; prefill
  uses the CPU/batched paths (dense projections do run on VK at prefill).
- DSA top-k selection, ragged multi-slot serving, and quantized-KV caches
  fall back to the CPU attention path.
- Not yet done: cooperative-matrix (coopmat) prefill kernels, a fully
  resident-layer pipeline.
