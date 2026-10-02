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

Set `COLI_NO_OMP_TUNE=1` on multi-core boxes: the engine's OMP self-tune
(active spin-wait) is skipped under `COLI_CUDA`/`COLI_METAL` but not under
Vulkan, and spinning worker threads starve the async I/O pool (measured
CPU expert bandwidth 28 → 5 GB/s without it).

**Discrete cards need Resizable BAR.** The weight tiers allocate
HOST_VISIBLE|DEVICE_LOCAL memory; with ReBAR disabled that combination only
exists in a ~256 MB BAR window, and the driver silently places everything
beyond it in system RAM — the tier then *reports* resident experts while every
access crosses PCIe, slower than the CPU path (measured 0.11 vs 0.24 tok/s
either side of the BIOS toggle on an RX 9070 XT). The engine now warns at init
when the host-visible slice of VRAM is small; if you see that warning, enable
Resizable BAR / Smart Access Memory in the BIOS. Unified-memory APUs are
unaffected.

The compiled shaders are found via `COLI_VK_SHADERS` (either the
`qmatmul.spv` file or the directory holding the `.spv` set); unset, the
engine looks in `shaders/` next to the binary, then relative to the CWD.

### Windows (MSYS2)

In the MSYS2 **UCRT64** shell ([quickstart.md](quickstart.md)), add the Vulkan
headers, the loader's import library and `glslc`, then build:

```bash
pacman -S --needed mingw-w64-ucrt-x86_64-vulkan-headers \
  mingw-w64-ucrt-x86_64-vulkan-loader mingw-w64-ucrt-x86_64-shaderc
cd c
make colibri.exe VK=1
```

The binary is statically linked like the default Windows build, plus one
import: `vulkan-1.dll`, the loader the GPU driver installs in `System32`, so it
runs outside MSYS2 with nothing added to `PATH`. The next-to-the-binary shader
lookup above is Linux-only: on Windows run from `c\` or set `COLI_VK_SHADERS`.
To check the driver before downloading a model, point `SNAP` at a folder
holding only a `config.json`, as the CI's Lavapipe job does; the backend
initialises before any weight is read:

```powershell
# in c\
New-Item -ItemType Directory -Force $env:TEMP\vkprobe | Out-Null
'{"model_type":"glm_moe_dsa"}' | Set-Content $env:TEMP\vkprobe\config.json
$env:SNAP = "$env:TEMP\vkprobe"; $env:COLI_VULKAN = "1"; $env:COLI_NO_OMP_TUNE = "1"
.\colibri.exe    # prints "[VK] ready: <GPU>", then exits: there is no model
```

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

## The other engines

Every engine links the same backend in a `VK=1` build (`make <engine> VK=1`;
`make deepseek-v4 VK=1` for DeepSeek V4) and opens it with `COLI_VULKAN=1` once its
weights are loaded. Kimi K3 has its own expert tier (`K3_VK`, see
[ENVIRONMENT.md](ENVIRONMENT.md)) and glm53 its own section in
[glm53-flash.md](glm53-flash.md). For the engines below, a missing device or missing
shaders prints `[VK] <engine>: no usable Vulkan device ..., running on the CPU` and
the run continues on the CPU. That differs from the GLM engine above, which exits.

What these engines put on the device is their **resident** matrices, in the form
they already hold in RAM, uploaded at the first multiply (MiMo uploads them at
startup). Routed experts arrive from disk on every miss and stay on the CPU, with
one opt-in exception for MiMo.

| Engine | On the device | Weight formats | Stays on the CPU |
|---|---|---|---|
| qwen36 (Qwen3.6, Qwen3-Coder, Qwen3.8-27B) | the dense trunk | int8 rows; int4-g64 with `COLI_DENSE_BITS=4`; f32 with `COLI_DENSE_I8=0` | DeltaNet `dn_a`/`dn_b`, vision tower, routed experts |
| qwen38 (Qwen3.8 Flash Next) | the trunk | int8 trunk rows, bf16, f32 (`Q38_NATIVE_BF16=0`) | routed FP8 experts |
| inkling | dense and shared-expert matrices | int8 and int4-g64 (dense-int4g64 container), f32, bf16 | routed experts, embedding and audio lookups, CUDA residents; bf16 on CPUs with the AVX512-BF16 dot (see below) |
| olmoe | attention q/k/v/o, router, lm_head | f32 | routed experts, embedding |
| deepseek_v41 | the trunk, vision included | fp8 in 32x32 ue8m0 tiles, bf16 | routed experts |
| deepseek_v4 | resident dense layers, head, router, compressors | fp8 in 128x128 blocks, bf16 | routed experts, the indexer's `weights_proj`, DSpark stages, the `--oracle` path |
| mimo | trunk and vision tower; up to `MIMO_VK_EXPERTS=N` routed experts | native fp8/bf16, int8, f32 (`MIMO_DENSE_BITS`); experts as MXFP4 | router |
| qwenimage | the DiT's matrices | int8, bf16, f32 (`COLI_IMG_BITS`) | text encoder, VAE, attention |

Each engine ends a run, and each serve turn, with
`[VK] <engine>: N matmuls on the GPU`. That count is how you tell a path that ran
from one that only initialised.

**Arithmetic.** The device reads the same weights the CPU reads and multiplies
them by f32 activations.
- Where the CPU's default kernel also uses f32 activations, the two differ only in
  the order of the sums.
- Where the CPU kernel rounds activations first, the device result instead matches
  the CPU's f32-activation setting, so tokens can drift from the CPU default after
  a few steps. These kernels are:
  - qwen36's int8 dot (`COLI_DENSE_IDOT`, on by default);
  - qwen38's int8 trunk;
  - qwenimage's `COLI_IMG_ACT8`.
- Two engines keep the CPU's exact arithmetic instead:
  - deepseek_v4 rounds activations to E4M3 on the host before the call, as its CPU
    kernel does.
  - inkling leaves its bf16 matrices on the CPU when the build has the AVX512-BF16
    dot (Zen 4/5, Sapphire Rapids), because that dot rounds activations to bf16.

**Memory.** The host copy stays as the CPU fallback. On an integrated GPU or APU,
which shares RAM with the CPU, the resident set is therefore held twice: size
`RAM_GB`/caps with that in mind. The weight arena does not return freed tensors'
memory, which is also why MiMo's expert tier never evicts.

**Status.** CI checks every engine above on Lavapipe (`tests/vulkan_engines.sh`, the
`vulkan-engines` job): each configuration gives the CPU run's tokens, and its matmul
count is above zero. That proves correctness, not speed. None of these engines has
been measured on a real GPU yet.

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
- Khronos validation layers: the backend never enables them, so the loader
  does. `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation` turns on the core
  checks; add `VK_LAYER_VALIDATE_SYNC=true` for synchronization validation
  (or point `VK_LAYER_SETTINGS_PATH` at a directory holding a file named
  exactly `vk_layer_settings.txt`). The harness above reports no hazards
  under it. Known layer defect, SDK 1.4.357.1 on MoltenVK: submit-time
  synchronization validation segfaults inside the layer at `vkDeviceWaitIdle`
  during shutdown; set `VK_LAYER_SYNCVAL_SUBMIT_TIME_VALIDATION=false`, or
  read stdout through a pty, since the crash lands in an `atexit` handler
  before stdio flushes.

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
  resident-layer pipeline, Polaris/gfx803 validation on real hardware (the
  shaders use dynamic subgroup sizes and are wave64-safe by construction).
