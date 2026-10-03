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
| Routed experts | on with `COLI_VULKAN=1` (`COLI_VK_TIER=0` turns it off) | The shared routed-expert tier ([below](#the-routed-expert-tier-vk_tierc)): warm from `.coli_usage` at startup, then adapting while you chat, under a budget (`COLI_VK_TIER_GB`). The resident experts of each MoE step, prefill included, run as one async batch while the CPU loads and computes the others; they need **no RAM slot and no disk read**. Shown as the `vk` bucket in the hit-rate line, and in a `[VK] tier colibri run:` line at exit. `COLI_VK_EXPERTS=N`, the fixed top-N set this engine used to upload, is a deprecated alias: N caps the tier at N experts, `0` turns it off. |
| Dense projections | `COLI_VK_DENSE=1` | q_a+kv_a fused into one submit, q_b, o; shared expert as a single fused expert-group submit. Resident int4/int8 weights upload once. |
| MLA attention core | `COLI_VK_ATTN=1` | One dispatch per layer: absorbed query, scores over the KV window, softmax, weighted latent, value rows, **fused with the o-projection** (the context vector never leaves the GPU). The latent/rope KV lives in a persistent per-layer device mirror, appended ~2.3 KB/token/layer with the same invalidation points as the CUDA KV shadow. |

The `PIN_GB=0` (with `PIN` still set) in the example is deliberate: the expert
tier warms from the same history a RAM pin would, so the pin's RAM is better
spent on the adaptive LRU cache. Keep `PIN` set so AUTOPIN does not re-pin from
history.

## The other engines

Every engine links the same backend in a `VK=1` build (`make <engine> VK=1`;
`make deepseek-v4 VK=1` for DeepSeek V4) and opens it with `COLI_VULKAN=1` once its
weights are loaded. Kimi K3 also reads its older switch `K3_VK=1` and MiMo its
`MIMO_VK_EXPERTS` (see [ENVIRONMENT.md](ENVIRONMENT.md)), and glm53 has its own section in
[glm53-flash.md](glm53-flash.md). For the engines below, a missing device or missing
shaders prints `[VK] <engine>: no usable Vulkan device ..., running on the CPU` and
the run continues on the CPU. That differs from the GLM engine above, which exits.
A device that opens prints `[VK] <engine>: device ready, dense matrices on the
device` (or `on the CPU`), and why.

**Where the dense matrices go** is one rule, the backend's `coli_vk_dense_decide()`,
for every engine below. `COLI_VK_DENSE=1` puts them on the device, `COLI_VK_DENSE=0`
keeps them on the CPU. Unset, they go to the device, except where the device shares
the CPU's RAM (an integrated GPU, or a CPU device such as Lavapipe) and the engine
runs the routed-expert tier ([below](#the-routed-expert-tier-vk_tierc)): there they
stay on the CPU and the device takes the experts. On a Radeon 780M the dense matmuls,
one synchronous call each at the GPU's 800 MHz floor, cost more than the tier gained
(Qwen3.8 decode at 2.35 tok/s with them on the device, 3.80 without). That case is
every engine below with routed experts, all of them but qwenimage, and any engine
added to the tier inherits it. A discrete GPU keeps the dense matrices
on the device by default. The GLM engine above reads the same variable through the
same function with its own default, off.

What these engines put on the device is their **resident** matrices, in the form
they already hold in RAM, uploaded at the first multiply (MiMo and Kimi K3 upload
them at startup). Routed experts arrive from disk on every miss; every engine below
that has them, and the GLM engine above, keeps a cache of them on the device with
the shared expert tier ([below](#the-routed-expert-tier-vk_tierc)).

| Engine | On the device | Weight formats | Stays on the CPU |
|---|---|---|---|
| qwen36 (Qwen3.6, Qwen3-Coder, Qwen3.8-27B) | the dense trunk; routed experts on the expert tier | int8 rows; int4-g64 with `COLI_DENSE_BITS=4`; f32 with `COLI_DENSE_I8=0`; experts int4-g64, int4 per row, int8 per row or gs64 | DeltaNet `dn_a`/`dn_b`, vision tower, the experts the tier does not hold |
| qwen38 (Qwen3.8 Flash Next) | the trunk; routed experts on the expert tier | int8 trunk rows, bf16, f32 (`Q38_NATIVE_BF16=0`); experts int4-g64 (sidecar), FP8 128x128 blocks, bf16 | the MTP head's experts, the experts the tier does not hold |
| inkling | dense and shared-expert matrices; routed experts on the expert tier | int8 and int4-g64 (dense-int4g64 container), f32, bf16; experts int4 or int8 per row (container or runtime quantization), f32 | embedding and audio lookups, CUDA residents (with CUDA or Metal on, the experts too); bf16 on CPUs with the AVX512-BF16 dot (see below); the experts the tier does not hold |
| olmoe | attention q/k/v/o, router, lm_head; routed experts on the expert tier | f32; experts int8 per row | embedding, the experts the tier does not hold |
| kimi_k3 (Kimi K3) | the shared experts' matrices (one row at a time: decode); routed experts on the expert tier | shared experts int8 rows, int4-g64, f32 (`K3_BITS`); experts MXFP4 with ue8m0 scales (fmt 7), SiTU-GLU in the latent space | KDA, MLA, the latent projections, router, head, prefill's shared experts, the experts the tier does not hold |
| mimo | trunk and vision tower; routed experts on the expert tier | native fp8/bf16, int8, f32 (`MIMO_DENSE_BITS`); experts MXFP4 with e8m0 scales (fmt 7) | router, the experts the tier does not hold |
| deepseek_v41 | the trunk, vision included; routed experts on the expert tier | fp8 in 32x32 ue8m0 tiles, bf16; experts MXFP4 (fp4, a ue8m0 scale per 32) | the DSpark stages and their experts, the experts the tier does not hold |
| deepseek_v4 | resident dense layers, head, router, compressors; routed experts on the expert tier | fp8 in 128x128 blocks, bf16; experts MXFP4 (fp4, a ue8m0 scale per 32) with [an activation of its own](#deepseek-v4s-activation) | the indexer's `weights_proj`, DSpark stages, the `--oracle` path's dense layers, the experts the tier does not hold |
| glm53 (GLM-5.3 Flash) | the resident matrices (an f32 checkpoint's experts among them); the streaming container's routed experts on the expert tier | int8 and int4-g64 (`GLM53_BITS`); experts int4-gs64 | f32 matrices (`GLM53_BITS=32`), the streamed experts the tier does not hold, all of them when `swiglu_limit` is 0 |
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
  - qwen36's int8 dot (`COLI_DENSE_IDOT`, on by default) and its routed-expert
    kernel (`QWEN_EXPERT_ACT`, int8 by default): the expert tier's experts match
    `QWEN_EXPERT_ACT=f32`, to 2.5e-7 of the logits on the test fixture;
  - qwen38's int8 trunk;
  - qwenimage's `COLI_IMG_ACT8`;
  - Kimi K3's routed-expert kernel (`K3_IDOT`, on by default): the tier's experts
    match `K3_IDOT=0`, the configuration of its vendor oracle;
  - MiMo's `MIMO_IDOT=1`, and inkling's and olmoe's `IDOT=1` (all three off by
    default).
- Two engines keep the CPU's exact arithmetic instead:
  - deepseek_v4 rounds activations to E4M3 on the host before the call, as its CPU
    kernel does; its routed experts on the expert tier make every rounding its CPU
    expert makes ([DeepSeek V4's activation](#deepseek-v4s-activation)).
  - inkling leaves its bf16 matrices on the CPU when the build has the AVX512-BF16
    dot (Zen 4/5, Sapphire Rapids), because that dot rounds activations to bf16.

**Memory.** The host copy stays as the CPU fallback. On an integrated GPU or APU,
which shares RAM with the CPU, the resident set is therefore held twice: size
`RAM_GB`/caps with that in mind. Freed tensors give their device memory back (see
the expert tier's section).

**Status.** CI checks every engine above on Lavapipe (`tests/vulkan_engines.sh`, the
`vulkan-engines` job): each configuration gives the CPU run's tokens, and its matmul
count is above zero. That proves correctness, not speed.

The first real GPU measured is an integrated Radeon 780M (RADV) in a Ryzen 7 PRO
8700GE (16 threads, 61 GiB DDR5, NVMe): same binaries, cold page cache, load under 2.
The shader harness (`tests/vulkan_engines.sh shader`) passes every format case there.
The engines are correct on it and slower than the CPU today:

| Workload | CPU | Vulkan, 780M |
|---|---|---|
| Qwen3.8 Flash Next int4, decode 100 tokens | 3.55 tok/s | 1.53 tok/s |
| Qwen3.8 Flash Next, prefill 512 tokens | 51 s | 109 s |
| Qwen3.6-35B-A3B, decode | 5.97 tok/s | 3.06 tok/s (identical output) |
| Qwen3.6-35B-A3B, prefill 512 tokens | 39.8 s | 60.7 s |

Why, measured:
- **Decode**: every matmul is a synchronous submit, about 726 per token, and an
  integrated GPU reads the same RAM as the CPU.
- **Prefill**: the shader is a per-row GEMV, so each weight is read once per prompt
  row.

## Prefill: the tiled GEMMs

`qmatmul.comp` is a GEMV per activation row: at S rows every weight is fetched and
decoded S times, which holds a Radeon 780M at 80 GFLOP/s whatever S is. From S = 2,
`coli_vk_matmul` (the resident-matrix path of every engine) takes a tiled GEMM
instead, so the API and the callers are unchanged:

- `qmatmul_gemm.comp`, fp32, every format: a workgroup owns a BM-output x BN-row tile
  of y, decodes its weight rows into shared memory once per 32-input step and runs
  them against BN activation rows staged beside them, a TM x TN block per thread.
  Group scales fold into the decoded weight; each step sums into a fresh partial
  (blocked summation), which keeps the error at the GEMV's level.
- `qmatmul_coop.comp`, where the device has `VK_KHR_cooperative_matrix` with a
  16x16x16 fp16 x fp16 -> fp32 subgroup shape and a settable subgroup size: the
  formats whose weights decode exactly to fp16 (int8, int4, int3-g64, MXFP4, fp8;
  grouped ones with gs % 32 == 0). Nothing is rounded to fp16: each activation row is
  scaled by a power of two the host computes during the upload, split into an fp16
  high part (top 11 bits) and an fp16 low part, and the MMA runs on both; every
  16-input product then joins fp32 accumulators, times the group scale. The harness
  holds it to the fp32 GEMM's bound.
- Each shader is built at a few tile widths (BN 32 for a 32-row prefill chunk, 64,
  128); a call takes the narrowest that covers its S. The threshold, measured on the
  780M: S >= 2 and S*O >= 4096 (a 48-output matrix stays on the GEMV up to S = 32).
  `COLI_VK_GEMM_MIN_S` overrides it, `COLI_VK_COOP=0` keeps the fp32 GEMM.

Measured on a Radeon 780M (RDNA3, RADV, Mesa 26.0) sharing DDR5 with a Ryzen 7 PRO
8700GE, `OMP_NUM_THREADS=8`. One matrix, I = 2560, O = 6144, S = 512, back to back
(the harness, `COLI_VK_TEST_GEMM_BENCH=1`; the CPU column is the kernel an engine
runs for that storage, `-march=native`):

| fmt | GEMV | tiled GEMM (fp32) | cooperative matrix | CPU, 8 threads |
|---|---|---|---|---|
| 1 int8 | 82 GFLOP/s | 1431 | 1909 | 841 (int8 activations, VNNI) |
| 2 int4 | 104 | 1486 | 2164 | 224 |
| 4 int4-g64 | 86 | 1406 | 2026 | 488 (int8 activations, VNNI) |
| 5 int3-g64 | 86 | 1441 | 2039 | 125 |
| 7 MXFP4 | 64 | 1390 | 1928 | 199 |
| 10 f32 | 29 | 948 | | 224 |
| 11 bf16 | 51 | 1222 | | 25 |
| 12 fp8 | 43 | 1259 | 1735 | 33 |

End to end, prefill of 512 tokens (`N_NEW=1`, cold page cache, same binaries):

| | CPU | Vulkan, GEMV | Vulkan, tiled GEMM |
|---|---|---|---|
| Qwen3.8 Flash Next (int8 trunk, bf16) | 50.7 s | 111.1 s | 53.8 s |
| same, `Q38_PREFILL_BATCH_ROWS=512` | 44.7 s | | 44.3 s |
| Qwen3.6-35B-A3B (int8 dense) | 40.9 s | 62.0 s | 56.3 s |

On this APU the GEMM brings Qwen3.8's Vulkan prefill from 111 s to the CPU's
level: level with it with 512-row prefill chunks (44.3 against 44.7 s), 6% behind
at the default 32-row chunks; the generated token is the CPU's on both models.
`VK_PROF=1` shows where the time goes. Qwen3.8 at the default chunks spends 8.4 s in
resident matmuls on the device against the CPU's 6.5 s: 6.8 s in 5,052
cooperative-matrix GEMMs (the DeltaNet projections at S = 32, 3.1 s; shared experts
and router, 1.7 s; attention and gated residuals at S = 512, 2.0 s) and 1.6 s in
2,273 GEMVs (1,024 one-token PLE projections, 0.86 s; matrices too narrow for the
GEMM, 0.7 s). Two things measured there hold it at parity:
- A clock stuck at 800 MHz. The engines call the device synchronously, one matrix
  at a time between CPU phases, and with `power_dpm_force_performance_level=auto`
  the GPU stays at its 800 MHz floor through those millisecond bursts (over 97% of
  the samples during the Qwen3.8 runs): a GEMM that takes 1.1 ms back to back takes
  2.3 ms after a 3 ms CPU gap. Fewer, larger calls (512-row chunks) are what lift
  it to parity.
- Per-token calls. Qwen3.6 without the CUDA tier projected its DeltaNet inputs one
  token at a time (`deltanet()` in qwen36.c): 46,081 of its 46,281 device matmuls
  were S = 1 and stayed on the GEMV (20 s of its 56).

The engines now project per block of rows wherever prefill used to call the device
once per token, recurrences and gathers still consuming the rows in order, and with
CPU outputs byte-identical to the per-token build: Qwen3.6's DeltaNet inputs and
out_proj, Qwen3.8's PLE keys and values, DeepSeek V4.1's router, indexer, index keys,
compressor and vision tower, DeepSeek V4's router, compressors and index queries,
GLM-5.3's device-resident projections, Kimi K3's DSA index keys and Inkling's
per-position heads. Same 780M, same runs:

| | CPU | Vulkan, tiled GEMM | device matmuls (GEMV / GEMM) |
|---|---|---|---|
| Qwen3.6-35B-A3B, per token | 40.9 s | 56.3 s | 46,081 / 200 |
| Qwen3.6-35B-A3B, per block | 36.0 s | 37.2 s | 1 / 380 |
| Qwen3.8 Flash Next, per token | 50.7 s | 53.8 s | 2,273 / 5,054 |
| Qwen3.8 Flash Next, per block | 52.8 s | 55.3 s | 1,249 / 5,086 |

The CPU gains too where the block lets a matrix stay in cache across rows (Qwen3.6's
DeltaNet: 7.1 to 2.5 s). Qwen3.8's block saves its 0.9 s of PLE GEMVs, inside the
run-to-run spread of its expert reads (cold page cache, about 2 s).

## The routed-expert tier (`vk_tier.c`)

The engines above keep their dense matrices on the device (on a discrete GPU; on
one that shares the CPU's RAM they stay on the CPU while this tier is on, see
[the other engines](#the-other-engines)); the routed experts are the other half of
a MoE model, and the one that does not fit. The tier keeps a cache
of them on the device the way a GPU-equipped PC should use its card:

- **What is resident adapts while you chat.** At startup the tier fills its budget
  from the expert history (`.coli_usage`, the hottest experts first, read from disk
  in parallel). After that, every expert the CPU computes is a candidate: it is
  promoted while there is room, or when it is hotter than the coldest resident by
  `tier.h`'s LFRU margin (25% + 4 routings), which is evicted. Heat is one per
  routing, halved every 1024 tokens; the history starts at 32 for a layer's hottest
  expert and in proportion below, so an expert of a new workload displaces the
  history's coldest residents after a few dozen routings of its own (it needs more
  than 1.25 x their heat + 4). A promotion copies the expert's bytes once on the engine thread
  (at most `COLI_VK_TIER_RATE` per token, 16) and an uploader thread writes it to
  the device; it serves from the next layer step on.
- **The device and the CPU compute at the same time.** For each MoE layer step the
  routed (row, expert) pairs whose expert is resident go to the device as ONE
  submit that nobody waits for: per expert, its rows run gate+up and the activation
  then down, all experts in one command buffer, on a queue of their own when the
  device has a second one (RADV's async compute, a second queue on NVIDIA and Intel),
  so the dense matmuls of the same layer do not wait behind it. Meanwhile the CPU
  loads and computes the other experts and the shared expert. Then the step joins.
- **Neither side waits for the other more than it must.** When a join keeps
  waiting (the device is the slower side: an integrated GPU at its floor clock),
  the tier hands the CPU the step's resident experts that the CPU also holds in RAM,
  beyond the device's share of the step's rows, and takes them back when the device
  finishes early; an expert only the device holds stays there (the CPU would read
  it from disk). A discrete card that finishes first keeps everything.
- **The sum does not depend on what was resident.** Every expert's output joins its
  row in routing (rank) order, the device's and the CPU's alike: the same order as a
  CPU-only run, so the device's experts differ from the CPU's only by their own
  summation order. A device row's bits do not depend on how many rows share its
  dispatch either (up to 15 rows an expert takes the per-row GEMV route), so an MTP
  verify's two rows get a decode step's bits. From 16 rows (prefill) an expert takes
  the tiled GEMM for gate, up and down.
- **Without `COLI_VULKAN`, nothing changes.** In a `VK=1` build with `COLI_VULKAN`
  unset, and in a build without `VK=1`, the stdout and the last logits of every
  qwen36 and qwen38 fixture configuration are the bytes of the build before the
  tier (110 configurations: bf16, FP8, int4-g64, int8, the MTP head, every prefill
  mode, both qwen36 expert kernels, the mixed container, four model geometries).
  The same holds for inkling and olmoe (36 configurations, stdout, stderr and every
  logit vector), kimi_k3 (30 configurations, every step's logits), mimo (32, text and
  picture, every prompt position's logits), deepseek_v41 and deepseek_v4 (40
  comparisons per build: every tiny oracle, the 40-token prompt, DSpark at every
  forced acceptance, the three V4 cases' oracle records on the 4- and the 8-expert
  fixture, served logprob echoes, and the engines' C tests), and colibri and glm53
  (94 configurations: every expert format of both, stdout and the teacher-forced
  logits). One default moved with
  it: a `VK=1` build of Kimi K3 used to open the device without being asked (`K3_VK=1`
  was the default); it now waits for `COLI_VULKAN=1` (or `K3_VK=1`) like every engine.

**Engines on the tier: every MoE engine.** Each describes its experts in the form its
RAM already holds them (nothing is requantized), adds every expert of a row in the
order its CPU-only run does, and keeps a history for the warm start where it has one:

| Engine | Experts in RAM (`VktSrc`), device format | Activation | Warm start from | Of its own |
|---|---|---|---|---|
| qwen36 (Qwen3.6, Qwen3-Coder, the 2.4T geometry) | int8 per row `I8_ROW` or gs64 `I8_GS` (fmt 1, 13); int4 per row or gs64 from the int8-slot kernel, `I8_AS_I4_ROW` / `I8_AS_I4_GS` (fmt 2, 4); planar int4-g64 from the int4 kernel, `I4U_PLANAR64` (fmt 4); the mixed container's int4 gate/up and int8 down | SwiGLU | `COLI_USAGE` (default `<snap>/.coli_usage`), kept only while the tier is on | the tier's experts match `QWEN_EXPERT_ACT=f32` (the default kernel rounds activations to int8) |
| qwen38 (Qwen3.8 Flash Next) | the int4-g64 sidecar `I4U_PLANAR64` (fmt 4); the release's FP8 in 128x128 blocks `FP8_BLOCK` (fmt 12); `BF16` (fmt 11); `F32` (fmt 10) | SwiGLU | `COLI_USAGE` (default `<snap>/.coli_usage`), always kept | the MTP head's layer stays on the CPU (its experts are FP8 beside an int4 sidecar) |
| inkling | int4 container `I4U_PAIRS_ROW` (fmt 2); int8 container `I8_ROW` (fmt 1); runtime int8 rows, `I8_AS_I4_ROW` at 2 to 4 bits (fmt 2) and `I8_ROW` above; `F32` at `bits=0` (fmt 10). Gate and up come from the fused `gate_up` tensor, up I rows in | SwiGLU | `<snap>/.coli_usage` or `PIN=<path>`, in the generate and serve modes; the ref.json oracle reads none | [Inkling and OLMoE](#inkling-and-olmoe) |
| olmoe | int8 rows `I8_ROW` (fmt 1), gate, up and down as the merged container holds them | SwiGLU | `COLI_USAGE` only | [Inkling and OLMoE](#inkling-and-olmoe) |
| kimi_k3 (Kimi K3) | the checkpoint's MXFP4 with ue8m0 scales `MXFP4_E8M0` 32 (fmt 7): gate `w1`, up `w3`, down `w2`, in the latent space | SiTU-GLU (`VKT_ACT_SITU`) | `COLI_USAGE` (default `<snap>/.coli_usage`) | [Kimi K3 and MiMo](#kimi-k3-and-mimo) |
| mimo (MiMo-V2.6 Flash and Pro) | the release's MXFP4 `MXFP4_E8M0` 32 (fmt 7) | SwiGLU | none: the tier fills as experts pass by | [Kimi K3 and MiMo](#kimi-k3-and-mimo) |
| deepseek_v41 (DeepSeek V4.1 Flash) | MXFP4 `MXFP4_E8M0` 32 (fmt 7), as the checkpoint stores it | SwiGLU with `swiglu_limit` | `COLI_USAGE` (default `<snap>/.coli_usage`), kept only while the tier is on | the backbone's layers; the DSpark stages keep their own experts on the CPU |
| deepseek_v4 (DeepSeek V4 Flash) | MXFP4 `MXFP4_E8M0` 32 (fmt 7); pinned experts unpacked from rows16 | `VKT_ACT_SWIGLU_V4` | the store's own `<model>/.coli_usage` | [DeepSeek V4's activation](#deepseek-v4s-activation) |
| colibri (GLM-5.2) | `F32` (fmt 10), int8 `I8_ROW` (fmt 1), int4 per row `I4U_PAIRS_ROW` (fmt 2), int4-gs `I4U_PAIRS_GS` (fmt 4), int3-g64 `I3_G64` (fmt 5); down may have its own | SwiGLU | `<snap>/.coli_usage` | [GLM-5.2 and GLM-5.3 Flash](#glm-52-and-glm-53-flash-on-the-tier) |
| glm53 (GLM-5.3 Flash) | the streaming container's int4-gs64 `I4U_PAIRS_GS` 64 (fmt 4) | SwiGLU with `swiglu_limit`, run only above 0 | `COLI_USAGE` (default `<snap>/.coli_usage`) | [GLM-5.2 and GLM-5.3 Flash](#glm-52-and-glm-53-flash-on-the-tier) |

With the CUDA expert tier built and on (`COLI_CUDA=1`) as well, **CUDA wins**: the
Vulkan tier stays off and says so (`[VK] tier <engine>: the CUDA expert tier is on
and wins`). The Vulkan dense trunk keeps running, on the device by default whatever
the device, since no Vulkan tier runs.

| Variable | Default | Effect |
|---|---|---|
| `COLI_VK_TIER` | on with `COLI_VULKAN=1` | `0`: no tier, the routed experts stay on the CPU (the dense trunk still uses the device). |
| `COLI_VK_TIER_GB` | measured | The tier's budget in GiB, within what the device can hold. Unset: below. |
| `COLI_VK_TIER_RESERVE_GB` | `1` | Device memory left to everything else (scratch, KV mirrors, the driver) on top of the dense weights the engine still has to place. |
| `COLI_VK_TIER_RATE` | `16` | Promotions per token at most (a prompt's forward gets this many per prompt token): each copies one expert on the engine thread. |
| `COLI_VK_TIER_BALANCE` | on | `0`: the device takes every resident expert of a step even when it is the slower side (see below). |
| `COLI_VK_TIER_WARM` | on | `0`: no warm start; the tier fills as experts pass by. |
| `COLI_VK_TIER_SYNC` | `0` | `1`: each layer step first waits for the uploads staged so far: residency then follows the routing alone (with `COLI_VK_TIER_BALANCE=0`, the run is reproducible). A promotion that displaces a resident while a batch is in flight waits for the join to free it. For tests and debugging. |
| `COLI_VK_TIER_GEMM_ROWS` | `16` | Rows from which an expert of a step takes the tiled GEMM instead of the per-row GEMV; `0` never. |
| `COLI_VK_TIER_QUEUE` | a second queue | `0`: the tier shares the main queue (its batches and the dense matmuls then serialize). |
| `COLI_VK_DENSE` | on, but off on a device sharing the CPU's RAM while the tier is on | `0`: the dense trunk stays on the CPU and the device takes the routed experts only; `1`: the trunk on the device whatever the device. Unset: on a discrete GPU, or with the tier off, on the device; on an integrated GPU or Lavapipe with the tier on, on the CPU. The startup line says which and why. (The GLM engine reads it through the same rule with its own default, off.) |
| `COLI_USAGE` | `<snap>/.coli_usage` | The history the warm start reads; each engine's is in the table above. qwen36 and deepseek_v41 keep it only while the tier is on, and save it at the end of every run and serve turn. olmoe reads one only when this is set, inkling also takes `PIN=<path>`, deepseek_v4 reads its expert store's own (`<model>/.coli_usage`, always kept) and MiMo none. |

**The budget.** On a discrete GPU: what `VK_EXT_memory_budget` says is free in
device-local memory, less the reserve and the dense weights the engine is about to
place there (Qwen3.8 puts 4.1 GiB of trunk on the device). On an integrated GPU (and on
Lavapipe), device memory IS the CPU's RAM: RADV on the Radeon 780M reports a 21 GiB
device-local heap and a 10.5 GiB host heap, together the 512 MiB carve-out and the
31 GiB of system RAM the kernel lets the GPU map; an allocation in either takes RAM
the CPU's cache and the page cache would otherwise have. There the default is a quarter of what
`MemAvailable` leaves once the engine's expert cache has grown to its configured
size (cap x layers x expert) and the dense weights are placed, less 2 GiB: the tier
never takes what that cache needs. `COLI_VK_TIER_GB` sets it explicitly. The startup
line says which rule applied:

```
[VK] tier qwen38: on, AMD Radeon 780M Graphics (RADV PHOENIX), budget 9.62 GiB = 3734 experts of 2.6 MiB (fmt 4 gs 64, down fmt 4 gs 64), shared RAM: a quarter of what the expert cache leaves, own queue, up to 16 promotions per token, balanced against the CPU
[VK] tier qwen38: warm start, 3734 experts from the history in 2.6s
```

**Memory that is given back.** Weight tensors are VkBuffers bound at offsets inside
256 MB device-memory blocks (one memory object per tensor makes every submit pay for
thousands of referenced allocations). `vk_alloc.h` hands those offsets out best-fit
and takes them back on free, coalesced; an emptied block goes back to the driver.
The tier's experts live in a pool of their own whose limit is the budget, at the
lower eviction priority (`VK_EXT_memory_priority`), so a pressed heap evicts experts,
never scratch or the dense trunk. A free while a batch may still read the tensor
waits for that batch's join.

**One line per run and serve turn** (stderr, beside the engine's own `[VK]` line):

```
[VK] tier qwen38 run: device 29464 of 59520 routed experts (49.5%; this run 29464 of 59520) | CPU RAM hits 18619, disk loads 8513 | resident 3734 (budget 3734, 9.61 GiB of 9.62 GiB, 39 blocks, frag 0.54) | uploads 4756 (10.89 GiB, 3734 warm), evictions 1022, skipped 0 queue + 2242 rate, failed 0 | device 8686.4 ms, CPU share 11136.6 ms, waited 3031.6 ms (65% of device time hidden) | balance: device share 0.05, 3869 rows handed to the CPU
```

- *device / routed*: where each routed (row, expert) pair ran; the rest is split by
  the engine's RAM cache into *RAM hits* and *disk loads*.
- *resident*, *budget*, the pool's blocks and fragmentation (`1 - largest free
  extent / free bytes`).
- *uploads* (warm-start ones included), *evictions*, promotions *skipped* because
  the upload queue was full or the per-forward rate was spent, uploads that
  *failed* (the device refused memory: the planned residency shrinks to what is
  there).
- *device*: the batches' device time (timestamps); *CPU share*: what the engine did
  between issue and join; *waited*: what the join then waited. The device time not
  waited for was hidden behind the CPU.
- *balance* (with the balancer on): the share of a step's resident rows the device
  keeps at the moment when the CPU holds them too, and the rows handed to the CPU
  since startup.

The dashboard's expert map (`EMAP`) shows a device-resident expert as tier 2 (VRAM),
the experts a device step served still light up in `HITS`, and qwen36's
`CACHE_ROUTE` ranks them like CUDA-resident ones.

### Inkling and OLMoE

Both run their routed experts on this tier with `COLI_VULKAN=1`, in the form their
RAM caches hold them (the table above). inkling's two shared experts run beside the
batch, on the CPU or, with the dense matrices, on the device; `TOPP`'s trimmed ranks
reach neither side; with CUDA or Metal on, the Vulkan tier stays off. olmoe's `PILOT`
prefetcher skips experts the device holds, and a slot is handed to the tier only
under the cache lock, while it still holds that expert.

Each layer step routes every row first, then runs per block of 64 rows: the block's
resident experts go to the device as one batch, the CPU computes the other pairs with
the kernels and the cache rounds it always uses, and every rank joins its row in
routing order. The device multiplies f32 activations, as both engines' default CPU
kernels do; `IDOT=1` (opt-in on both) rounds the CPU's activations and the device's
not. With `COLI_VULKAN` unset, or in a build without `VK=1`, the stdout and the
stderr (timings aside) and every logit vector of 36 configurations per build (each
expert format, caps 1 to 8, `TOPP`, `IDOT`, the generate and serve modes, `PPL`,
`ROUTE_TRACE`) are the bytes of the build before the tier.

### Kimi K3 and MiMo

Both had a tier of their own, which filled once and never evicted; they joined this
one, and their switches are read as its own ([ENVIRONMENT.md](ENVIRONMENT.md)): Kimi
K3's `K3_VK` (`1` opens the device as `COLI_VULKAN=1` does, `0` keeps it closed),
`K3_VK_GB` (the budget, as `COLI_VK_TIER_GB`) and `K3_VK_UP` (promotions per token, as
`COLI_VK_TIER_RATE`), MiMo's `MIMO_VK_EXPERTS` (`N` sizes the budget at N experts, `0`
keeps the experts on the CPU). Kimi K3's experts run SiTU-GLU on the device with the
config's two constants, in the latent space, and match `K3_IDOT=0` (the CPU's default
kernel rounds activations to int8); MiMo's run SwiGLU. Both add every pair of a row in
the CPU run's order (Kimi K3: its union's disk-offset order; MiMo: the row's routing
order), and offer the experts the CPU computed after the join, those still in the RAM
cache, so that a promotion that displaces a resident frees it at once and its upload
starts on a pool with room.

### DeepSeek V4's activation

DeepSeek V4's CPU expert rounds more than the others do: gate and up to bf16 before
the SwiGLU, the activation times the route weight to bf16, then down's input to E4M3
with one power-of-two scale per 128 values (its fp4 kernel rounds every input that
way), and down's output to bf16. Left to the plain SwiGLU, a device row would differ
from the CPU's by those roundings, not by a summation order. So the tier has a third
activation for it, `VKT_ACT_SWIGLU_V4` (the backend's `COLI_VK_ACT_SWIGLU_V4`):

- the engine rounds x to E4M3 per 128 before `vkt_issue_w`, as its kernel rounds it,
  and hands the route weights with the routing;
- the activation shaders (both routes) round gate and up to bf16 and compute the
  clamped SwiGLU in the CPU's form;
- `expert_act_v4.comp`, one pass between the activation and down, multiplies each row
  by its weight, rounds to bf16, and rounds each block of 128 to E4M3 with the CPU's
  scale (the smallest power of two that brings the block's maximum under 448);
- the engine rounds each returned row to bf16 and adds it with no weight of its own,
  in its CPU path's order (ascending expert id, then rank).

What is left between the two sides is the projections' summation order and the
device's `exp`; on everything measured the roundings absorbed both, though a value
that lands on a rounding boundary can still go the other way on another driver.
Measured on Lavapipe: the harness's
exact-input case gives the CPU's values bit for bit through every rounding (0 of 3840
differ, and three mutants of the pass each fail it); `vk-tier-check` gives 264 of 264
device rows bit-identical to the CPU arithmetic on random MXFP4 experts; a served
71-token prompt with every routed expert on the device gives the CPU's per-position
logprob echoes byte for byte. On an Intel Iris Xe through Mesa's Dozen the harness
case is bit for bit too, and every deepseek_v4 and deepseek_v41 tier configuration
gives the CPU's tokens.

Two things of the engine's own: the store keeps its pinned hot experts in a 16-row
interleaved layout (rows16), which the tier unpacks to rows only when it will take
the expert (`vkt_wants`); and a routing the device served is counted as a store
lookup counts one (pins, HITS and heat, `.coli_usage`), so the history stays the
routing's whichever side computed it.

### GLM-5.2 and GLM-5.3 Flash on the tier

**GLM-5.2 (`colibri`).** With `COLI_VULKAN=1` the routed experts go to the tier, in the
form the loader holds them: f32 (`./colibri <cap> 16`), int8 and int4 per row and
int3-g64 when a bf16 or FP8 checkpoint is quantized at load, the int4-gs container
(`convert_fp8_to_int4.py`'s default, gs 64) and int3-g64. Gate and up share one format,
down may have its own (`--down-bits 3`; an int4-g64 container's rows narrower than the
group stay per row). E8/IQ3 experts (fmt 6, whose input is rotated), int2 and fp8 have
no device form and stay on the CPU (`[VK] tier colibri: experts in fmt 6/6/6 ... stay
on the CPU`), and so does the MTP head's layer (int8). The tier serves every row count:
decode, the MTP and n-gram verify rows and the batched prefill, by blocks of 64 rows; the
fixed set it replaces served S <= 4 only.

- *The sum.* Without `COLI_VULKAN`, `moe()` adds a token's experts in the order of the
  batch's union, as before (the default build's stdout and the teacher-forced logits of
  every fixture configuration are the base commit's bytes). With the tier on, every
  expert of a row joins it in routing (rank) order, the device's and the CPU's alike,
  then the shared expert.
- *What the CPU computes beside the batch* goes through the same pin set, LRU and disk
  loads (`PIPE` included) as the CPU path, and every expert it computes is offered to
  the tier. A device-served expert takes no RAM slot and no disk read.
- *Arithmetic.* The device multiplies f32 activations. The CPU's int8 dot (`IDOT`,
  on by default) rounds the activations to int8 for int8 experts, and for int4-per-row
  experts from two rows (from one on AVX-512 VNNI and ARM dot-product builds): there
  the tier's experts match `IDOT=0`, which the tests set.
- *The old names.* `COLI_VK_EXPERTS` (the count of the fixed set, 320 by default) is a
  deprecated alias: `N` caps the tier at N experts, `0` turns it off as `COLI_VK_TIER=0`
  does, and a line says so. `COLI_VK_RESERVE_GB` (the old reserve, 3 GB) adds what it
  asks beyond `COLI_VK_TIER_RESERVE_GB`. With `COLI_VK_ATTN=1` the absorb core's KV
  mirror (`CTX` rows a layer) is reserved too. The dense set (`COLI_VK_DENSE=1`) is
  uploaded before the tier sizes itself, so the budget is what remains; the trunk's
  default here stays off. On an integrated GPU the tier takes its default share of RAM
  after the expert cache's (the startup reservation of the fixed set is gone).
- *`COLI_VK_DEV2`*: the second device's registry, fixed at startup, takes the hottest
  experts of the history that the tier does not hold (with the tier off, the hottest
  ones), up to `COLI_VK_EXPERTS2`; a step sends them there as one group on a worker
  thread while the tier's batch and the CPU run.

**GLM-5.3 Flash (`glm53`).** The streaming container's int4-gs64 experts go to the tier;
the resident matrices follow the dense rule above (on a device that shares the CPU's RAM
they stay on the CPU while the tier is on; before, `COLI_VULKAN=1` always put them on the
device). An f32 checkpoint, whose experts are resident matrices, has no tier. The
experts' SwiGLU is clamped (`swiglu_limit`): `swiglu_clamped` clamps at any limit,
including 0, where it zeroes every routed expert's output, while the shader clamps only
above 0; so the tier runs only for a limit above 0 and says why otherwise. The shared
expert is written first, then every rank of every row in routing order. Its tiny
fixtures have a single MoE layer, whose forwards the tier cannot tell apart by the
layer index going back: the engine marks each forward's start (`vkt_begin_forward`).

Both print `[VK] tier colibri|glm53 run:` at the end of a run and a `turn` line after
each serve turn; `EMAP` shows a tier-resident expert as tier 2. No GPU has run GLM
weights on the tier yet: `tests/vulkan_engines.sh glm` proves the tokens on Lavapipe,
nothing about speed.

### Measured on a Radeon 780M

Same box as above (Ryzen 7 PRO 8700GE, 16 threads, 61 GiB DDR5, NVMe, RADV), one
quiet run each after the model files were dropped from the page cache, the same
binary for every arm, `OMP_NUM_THREADS=8`. Qwen3.8 runs the int4-g64 sidecar at
cap 96, Qwen3.6 the int4 gs64 container at cap 64; decode is 100 tokens after a
25-token prompt (prompt included, as above), prefill a 512-token prompt
(`N_NEW=1`; Qwen3.8 with `Q38_PREFILL_BATCH_ROWS=512`).

Each arm set `COLI_VK_DENSE` explicitly; the trunk on the CPU is now this device's
default while the tier is on. Every tier arm starts from a history of one unrelated
conversation (a 231-token
prompt about planning a bakery's week, 100 tokens generated), as a user's would be;
*no warm start* starts from nothing. A first round, run from histories that had
seen the benchmark's own prompts, is at the end. Decode, 100 tokens (the rate the
engine reports; in brackets the whole process, load and warm start included):

| | Qwen3.8 Flash Next, int4 | Qwen3.6-35B-A3B |
|---|---|---|
| CPU | 3.51 tok/s (36.4 s) | 6.02 tok/s (23.6 s) |
| tier, trunk on the CPU (`COLI_VK_DENSE=0`) | 3.80 tok/s (36.9 s) | 8.03 tok/s (22.5 s) |
| the same, `COLI_VK_TIER_BALANCE=0` | 3.71 tok/s (37.6 s) | 7.86 tok/s (22.7 s) |
| the same, no warm start | 3.45 tok/s (37.0 s) | 6.10 tok/s (23.4 s) |
| tier and trunk on the device (`COLI_VK_DENSE=1`) | 2.35 tok/s (53.0 s) | 7.32 tok/s (23.7 s) |
| trunk on the device, no tier (`COLI_VK_TIER=0`, first round) | 1.60 tok/s (70.4 s) | 3.22 tok/s (38.1 s) |

Prefill of a 512-token prompt (time to the first token; in brackets the whole
process):

| | Qwen3.8 Flash Next, int4 | Qwen3.6-35B-A3B |
|---|---|---|
| CPU (first round) | 43.9 s (51.8 s) | 35.7 s (42.7 s) |
| tier, trunk on the CPU | 38.3 s (48.9 s) | 12.3 s (22.4 s) |
| the same, no warm start | 44.3 s (52.4 s) | 21.0 s (28.2 s) |
| tier and trunk on the device | 38.4 s (48.8 s) | 12.5 s (22.7 s) |
| trunk on the device, no tier (first round) | 43.2 s (51.2 s) | 37.1 s (44.1 s) |

What the numbers say, and what they do not:

- **The tier wins by the reads it saves.** At these caps the CPU's RAM cache misses
  often and every miss is an NVMe read; an expert on the device is neither read nor
  computed by the CPU. Qwen3.6's budget (12.5 GiB) holds 74% of its 10,240 experts
  and served 91% of the decode's routed pairs and 75% of the prefill's; Qwen3.8's
  (9.6 GiB) holds 15% of its 24,576 and served 50% of the decode's pairs and 16% of
  the prefill's. Hence 1.33x on Qwen3.6's decode and 2.9x on its time to the first
  token, against 1.08x and 1.15x on Qwen3.8.
- **The warm start is paid at startup.** Reading the history's experts took 2.3 to
  3.4 s (7.6 to 11.1 GiB) in every warm arm. The whole-process times include it, the
  rates do not: over a 100-token run Qwen3.8's tier comes out even with the CPU
  (36.9 s against 36.4 s), Qwen3.6's 1 s ahead. Without a history the tier fills as
  experts pass, at most 16 per token: not enough over 100 tokens for Qwen3.8 (3.45
  against 3.51 tok/s); Qwen3.6's prefill still gains (21.0 s against 35.7 s).
- **The trunk on the device costs more than the tier gains, here.** The trunk's
  synchronous matmuls at the 800 MHz floor (see above) make "trunk on the device, no
  tier" the slowest arm, and "tier and trunk" sits between it and the tier alone.
  Hence the default: on a device that shares the CPU's RAM, with the tier on, the
  trunk stays on the CPU (`COLI_VK_DENSE=1` puts it back). A discrete card keeps it
  on the device by default; nothing here measures one.
- **Overlap.** On Qwen3.8's decode the device computed 8.7 s of experts and the
  joins waited 3.0 s of it: 65% ran behind the CPU's share of the step. On Qwen3.6
  the device is the slower side (6.8 s of device time against 1.9 s of CPU share);
  the balancer moved the share to its floor, but the experts it holds are mostly not
  in the CPU's 64-slot cache, so there was little to hand back. With the balancer off
  the rates were 2% lower on both models, inside what one run to the next varies on
  this box.
- **The clock.** In the runs with the trunk on the CPU the GPU sat at its 800 MHz
  floor in 97 to 100% of the samples, except Qwen3.6's prefill (72% warm, 87%
  cold). Nothing here was run with the clock pinned.
- **Memory.** On this APU the tier's device memory is RAM, and it does not show in
  the process's RSS: the lowest `MemAvailable` during a decode fell from 52 to
  42 GiB (Qwen3.6) and from 40 to 31 GiB (Qwen3.8) with the tier on.
- **The text.** Qwen3.6 printed the CPU's text in every arm, decode and prefill.
  Qwen3.8 with the trunk on the CPU printed the CPU's 100 decode tokens in two of four
  runs; in the other two (balancer off, no warm start) the text left the CPU's at the
  72nd word. Its first token after the 512-token prompt was the CPU's in four of five
  runs. Its experts compute in f32 on both sides, so the device's and the CPU's
  differ only in summation order, about 1e-7 of the logits on the test fixtures. Over
  48 layers of top-10-of-512 routing and 512 tokens such differences flip routing
  near-ties and grow: after that prompt the logits differ from the CPU's by 0.36 on
  average (KL 0.10), and by as much between the tier's own two routes for prefill
  rows (the GEMM against the per-row GEMV: 0.34, KL 0.07). The trunk on the device
  moves them further (0.76, KL 0.49) through its int8 trunk (see Arithmetic above).
  A run with the tier is also not bit-reproducible by default: which experts a step
  finds resident depends on when the uploader finished, and the balancer on measured
  times (`COLI_VK_TIER_SYNC=1 COLI_VK_TIER_BALANCE=0` removes both).

The first round (histories that had seen the benchmark's prompts: Qwen3.8's own
`.coli_usage` from earlier work, and for Qwen3.6 a run on the prefill prompt):
Qwen3.8 with the trunk on the CPU decoded at 3.70 tok/s with 56% of the pairs on the
device and reached the first token of the 512-token prompt in 32.9 s with 63% (the
second round's 38.3 s had 16%); Qwen3.6 decoded at 8.04 tok/s and prefilled in
12.8 s. A history that has seen the prompt helps Qwen3.8 and hardly matters for
Qwen3.6, whose budget holds most of its experts anyway.

### What was not measured

No discrete GPU was available. On one, the tier's experts sit in VRAM and the device
reads them at VRAM bandwidth, several times what the CPU gets from DDR; that is the
case the design is for, and nothing above is a prediction of it. What the 780M does
not have and a discrete card does: its own memory (here the device's experts and
the CPU's cache share the same DDR5 channels and the same 61 GiB), a clock that
leaves its floor under bursty work (the 780M's mostly did not), and PCIe uploads
(here an upload is a RAM copy). What the 780M does show is that the machinery
holds: the batches overlap the CPU, the history fills the budget in a few seconds,
eviction keeps the budget, and the fixtures' tokens are the CPU's.

### Integrated GPUs: reading the RAM cache in place

`VK_EXT_external_memory_host` lets the device read host memory where it is, with no
second copy: on an APU that would make the tier's experts the RAM cache's own slots.
The harness measures it (`COLI_VK_TEST_HOSTMEM=1 ./vk_test`): batches of 10
experts of Qwen3.8's shape (int4-g64, 2.76 MB each) cycling over 48 distinct
experts, once from the tier's device memory and once from page-aligned host memory
imported in place. On the 780M, three runs:

| | device time per batch |
|---|---|
| experts in the tier's device memory | 2.57 to 2.59 ms |
| experts in imported host memory | 3.24 to 3.77 ms (same bits) |

The copy the import would save costs 0.095 ms per expert into the tier's memory
(29 GB/s; 0.063 ms into ordinary memory), on the uploader thread, off the engine's
path. Reading in place is slower on every batch to save a copy paid once per
promotion, so the tier copies, on APUs too. Serving the RAM cache's own slots would
also need slots that are page-aligned, slots held against the cache's LRU while a
batch reads them, and slots laid out the way the shaders read them (today the copy
converts `expert_ffn.h`'s planar int4 and spreads Qwen3.8's FP8 block scales). The
backend enables the extension when the device has it; nothing outside the harness
uses it.

## The dense chain (`vk_chain.c`)

With the dense matrices on the device one `coli_vk_matmul` at a time, each matrix is
a submit and a host round trip: about 726 of them per Qwen3.8 decode token. On an
integrated GPU that made the dense part slower on the device than on the CPU (see
[the other engines](#the-other-engines)). The chain records a whole layer into one
command buffer instead, and keeps the residual stream on the device from one layer
to the next. qwen36 (Qwen3.6, Qwen3-Coder, Qwen3.8-27B) and qwen38 (Qwen3.8 Flash
Next) run it: by default on a discrete GPU, and for qwen36 on an integrated one with
the expert tier (see [the default](#the-chain-on-a-radeon-780m)); `COLI_VK_CHAIN=1`
anywhere. olmoe and inkling run it as well ([OLMoE and Inkling](#olmoe-and-inkling)).
colibri (GLM-5.2), glm53 (GLM-5.3 Flash) and kimi_k3 run it too, with the MLA,
KDA and hyper-connection ops ([below](#glm-52-and-glm-53-flash-on-the-chain)).

**What runs where, per layer** (S rows: one at decode, a prompt chunk at prefill):

| | qwen36 | qwen38 adds |
|---|---|---|
| device, frame A1 | the previous layer's MoE output joining the residual (in the CPU's order: routed experts in rank order, then the gated shared expert, then the add); the input RMSNorm; the gated attention (q/k/v, per-head q/k norm, RoPE, the new K/V rows into the device cache, attention with the output gate, o_proj) or the Gated DeltaNet (qkv/z/b/a, the causal convolution with its ring, the recurrence with its state, the gated norm, out_proj); the residual add; the post-attention norm; the router logits | the four hyper-connection streams: each block's gated-residual read (per-stream norm, the low-rank pair, the stream mix, the inject weights) and its write-back; the PLE layer's projections, gate and dilated convolution (the n-gram table rows come up from the host); QSA with its indexer: each block's pooled key computed once on the device when a step completes it, and the top-k selection per query row |
| host | the router's softmax and top-k; the routed experts (the expert tier's device batch and the CPU's share, joined in rank order); the new K/V rows copied into the host's cache | the new index-key rows too |
| device, frame A2 (not waited for) | the shared expert and its gate, while the host computes the routed experts | |
| last frame | the final norm and lm_head on the last row | the final mixer; lm_head on the last one or two rows (an MTP verify) |

Per layer the host gets the normalized rows the routed experts read (D floats a row)
and the router logits (E floats a row), and the new K/V (and index-key) rows of an
attention layer; it sends the routed sum back (D floats a row). A model without
routed experts (Qwen3.8-27B) has no host step: its whole forward is one frame.

**The state, and who owns it.**
- The residual stream: on the device for the whole forward.
- The attention KV cache (and Qwen3.8's index keys): the host's copy stays canonical,
  as with the GLM engine's KV mirror. The device holds a mirror per layer with a
  watermark: rows below it equal the host's. A step from `pos_base` first uploads
  the rows between the watermark and `pos_base`; a step that runs on the CPU lowers
  the watermark to its `pos_base`; a cache that grows is mirrored again. Qwen3.8's
  pooled block keys have a watermark of their own, lowered to the first block a step
  rewrites (a rejected draft's block is recomputed when a step completes it again).
- The DeltaNet recurrent state and conv rings, and Qwen3.8's PLE ring: on the device
  while the chain runs (60 MB on Qwen3.6-35B, too much to copy per token). The host's
  copy is brought back before anything reads it there (a pinned snapshot, the prompt
  cache, a CPU step) and pushed up after anything writes it there (a reset: a fill
  with zeros on the device; a restored snapshot: an upload).
- An MTP verify (S = 2) snapshots the DeltaNet states, the conv rings and the PLE ring
  after its first row on the device (the shaders write the snapshot as they pass
  that row); a rejected draft swaps the device buffers, as the CPU swaps its own.
  Its matrices take the per-row GEMV, so its rows get a decode step's bits.

Prompt-cache and prefix reuse need nothing else: a reused prefix is rows below the
watermark and a recurrent state that already sits where the next step expects it.
The serve and prefix tests run with the chain on (below).

**What stays on the CPU.** The routed experts the tier does not hold, the router's
top-k, the embedding gather and the vision tower's rows, Qwen3.8's n-gram table reads
and the MTP head (its experts are FP8 beside the int4 sidecar, two rows per draft).
The chain declines, and the per-matrix path runs with the state synced first, under
the CUDA expert tier (CUDA keeps its priority), a qpack container, PILOT prefetch, or
a geometry outside its shaders (head dim above 256, a DeltaNet value head above 128 or
key head above 256, a conv kernel above 9). A device lost while the chain holds the
recurrent state does not stop the engine: it rebuilds that state on the CPU from the
prefix record (the ids the state was built from; the KV rows are the host's already),
a prefill's worth of CPU work, and runs on the CPU from there. Only a state the ids do
not describe (a turn with an image) cannot be rebuilt: that stops the engine with a
message. `COLI_VK_CHAIN_FAULT=n` fakes the loss at the n-th frame (the tests use it).

**The shaders** (`shaders/chain_*.comp`, each documented at its top): RMSNorm over
segments (rows, heads with a gate between them, streams with a weight slice each,
zero-centred or not, L2); RoPE from a host table (the CPU's own cosf/sinf at the CPU's
angles, so M-RoPE is just another table); grouped-query attention with an online
softmax over tiles of 128 positions, the output gate and an optional selection list;
the DeltaNet convolution and the recurrence (one workgroup per value head, a column of
the state in registers per thread, the gated norm fused); the element-wise steps; the
QSA block keys and selection; the PLE gate and convolution; and a decode GEMV for the
trunk's formats (int8 rows, int4-g64, bf16, f32) that reads 16 bytes per lane per step
and spreads a row over a cluster of lanes. The matrices of a prefill chunk take the
backend's fp32 tiled GEMM from its threshold (S ≥ 2 and S·O ≥ 4096). Activations are
f32 throughout, as the CPU's f32 path.

| Variable | Default | Effect |
|---|---|---|
| `COLI_VK_CHAIN` | on for a discrete GPU; on an integrated GPU with the expert tier, what the engine measured (qwen36 and olmoe on, qwen38 off; mimo, inkling, colibri, glm53 and kimi_k3 off: not measured); off on a CPU device | `1`: every layer's dense chain on the device; `2`: prompts only (forwards of more than two rows; decode and MTP verifies on the per-matrix path, the state moving between the two); `0`: the per-matrix path. The `[VK] <engine>: dense chain ...` line says which and why. |
| `COLI_VK_CHAIN_ROWS` | `512` | Prompt rows per chunk: a longer prompt runs every layer chunk by chunk (the device's scratch is sized for one chunk). |
| `COLI_VK_CHAIN_GEMV` | on | `0`: the decode matrices take `qmatmul.comp`'s GEMV instead of `chain_gemv.comp`'s. |
| `COLI_VK_CHAIN_SPIN_US` | `2000` | How long a wait on a chain frame polls the fence before blocking. |
| `COLI_VK_CHAIN_PROF` | off | `1`: one `[VK] chain profile` line of device time per kind of op (timestamps). |

Each run and serve turn prints `[VK] <engine> chain: N forwards, F frames (ops,
matmuls, tiled GEMM), the time spent waiting for the device, the routed experts' host
time and the device memory the chain holds`.

### The chain on a Radeon 780M

The box and the method of [the tier's measurements](#measured-on-a-radeon-780m): Ryzen
7 PRO 8700GE, RADV, `OMP_NUM_THREADS=8`, every run after the model files were dropped
from the page cache, 1-min load under 2, every tier arm from the same history of one
unrelated conversation, the same binary for every arm; Qwen3.8 runs the int4-g64
sidecar at cap 96, Qwen3.6 the int4 gs64 container at cap 64. Decode is 100 tokens
after a 25-token prompt (the rate the engine reports; in brackets the whole process),
prefill a 512-token prompt (`N_NEW=1`; Qwen3.8 with `Q38_PREFILL_BATCH_ROWS=512`).
Where a cell lists two or three numbers, they are separate rounds.

| Decode | Qwen3.8 Flash Next, int4 | Qwen3.6-35B-A3B |
|---|---|---|
| CPU | 3.49 tok/s (36.5 s) | 6.01 tok/s (23.6 s) |
| tier, trunk on the CPU | 3.81, 3.81, 3.84 tok/s (36.8 s) | 8.03, 8.06 tok/s (22.5 s) |
| tier and chain (`COLI_VK_CHAIN=1`) | 3.18, 3.18 tok/s (41.7 s) | 9.94, 9.92, 9.97 tok/s (20.2 s) |
| tier and chain on prompts only (`COLI_VK_CHAIN=2`) | 3.64 tok/s | |

| Prefill, 512 tokens | Qwen3.8 Flash Next, int4 | Qwen3.6-35B-A3B |
|---|---|---|
| CPU | 43.6 s (51.5 s) | 35.7 s (42.8 s) |
| tier, trunk on the CPU | 38.7, 38.6 s (49.4 s) | 12.2 s (22.3 s) |
| tier and chain (prompts: `COLI_VK_CHAIN=1` or `2`) | 30.1, 30.1, 30.1 s (40.5 s) | 9.5, 9.5 s (19.7 s) |

What the numbers say:

- **Qwen3.6: the chain wins both.** Decode 24% over the tier alone, 65% over the CPU;
  the first token 22% sooner than the tier alone. A decode token is 82 frames (two a
  layer and the head) where the trunk on the device took about 290 synchronous
  submits; the host waited 53 ms a token for the device's frames and spent 40 on the
  routed experts (the tier's batch, the CPU's share, their join).
- **Qwen3.8: prefill wins, decode loses.** The first token comes 22% sooner, but
  decode is 17% slower than the tier alone. Its trunk is 3.6 G weights in the layers
  and 0.6 G in lm_head, int8 rows, read whole every token (more than its routed
  experts); the device's decode GEMV, at the 800 MHz floor the GPU mostly sits at,
  reads int8 at 30 to 43 GB/s back to back, where the CPU's integer kernel takes the
  trunk in about 84 ms a token (73 ms of resident matmuls and 11 of lm_head in the tier
  arm's timers), about 50 GB/s. `COLI_VK_CHAIN_PROF=1` put 77% of the chain's device
  time in those GEMVs. Running only the prompts on the device (`COLI_VK_CHAIN=2`) keeps the prefill
  gain and still loses 5% of decode: on shared RAM the trunk's device copy (1.1 GiB of
  the budget here) is taken from the tier.
- **The clock.** With the trunk on the CPU the GPU sat at its 800 MHz floor in 97 to
  100% of the samples during decode. The chain keeps it busy enough to leave the floor
  part of the time: 53% of the samples at 800 MHz on Qwen3.6's decode, 39% on
  Qwen3.8's, 60% and 84% on the prefills. Nothing here pinned the clock.
- **The text.** Qwen3.6's chain printed the CPU's 100 decode tokens word for word (the
  tier alone left them at the 22nd word in this round); Qwen3.8's left them at the
  59th word of 78 (the tier alone at the 43rd). The chain multiplies f32 activations
  where the CPU's int8 kernels round them (see Arithmetic above); after the 512-token
  prompts every arm gave the CPU's first token.
- **Memory.** The chain holds 160 to 290 MiB on the device for Qwen3.6 and 350 to
  720 MiB for Qwen3.8 (state, mirrors and scratch for a 512-row chunk), beside the
  trunk's device copy (1.9 GiB of int8 for Qwen3.6). On an integrated GPU both come
  out of the tier's budget: 12.05 instead of 12.48 GiB on Qwen3.6, 8.4 instead of
  9.6 GiB on Qwen3.8.

**The default** (`coli_vk_chain_decide`, next to `coli_vk_dense_decide`): on a discrete
GPU the chain is on. On an integrated GPU with the expert tier on, each engine passes
what it measured here: qwen36 on, qwen38 off (decode, a chat's steady state, is slower
in both modes; `COLI_VK_CHAIN=2` is the choice for long prompts). Without the tier, and
on a CPU device such as Lavapipe, it is off. An engine not timed on an integrated GPU
passes `COLI_VK_CHAIN_UNMEASURED`: off there, and the line says "not measured". The
startup line says which and why:

```
[VK] qwen36: dense chain on (an integrated GPU with the expert tier: measured faster on decode and prefill; COLI_VK_CHAIN=0 off, 1 on, 2 prompts only)
```

**What was not measured.** No discrete GPU was available. On one the trunk sits in
VRAM, read at several times the CPU's bandwidth, the clock is not held at a floor
between bursts, and the host round trip per layer crosses PCIe (a few KB a row each
way); that is the case the chain's default is set for, and nothing above is a
prediction of it. Not timed either: contexts past 2048 tokens (where Qwen3.8's QSA
selects blocks instead of attending to all of them), serve sessions, Qwen3-Coder and
Qwen3.8-27B (no checkpoints on the box); their correctness is the Lavapipe gates'.

### MiMo-V2.6 on the chain

`mimo_chain.h` runs MiMo-V2.6's layers (Flash and Pro) as frames on the device, as
qwen36's chain does, with what this model has instead of a DeltaNet:

| | mimo |
|---|---|
| device, one frame per MoE layer | the previous layer's routed sum joining the residual (the CPU's `h += out`); the input RMSNorm; the fused qkv projection in its dense form (`MIMO_DENSE_BITS`: the release's FP8 with its 128-column block scales, int8 rows or f32); partial RoPE on q and k (rotate-half on the first `rope_dim` dims, from a host table of the CPU's own `cosf`/`sinf`, one per layer kind: the two kinds have their own theta); the value scale (`chain_ew` SCALE); the new K/V rows into the device's cache; attention with the layer's sliding window and its sink logits (`chain_attn.comp` with a window, a ring and a sink); o_proj; the residual add; the post-attention RMSNorm. The dense layer (layer 0 on the release) runs its MLP there too and has no host step |
| host | `moe()` as the CPU runs it, on the device's normalized rows: the router (sigmoid scores, the correction bias picks, top-k renormalized) and the routed experts (the tier's batch and the CPU's share, joined in each row's routing order). There is no shared expert, so nothing runs beside it |
| last frame | the final norm and lm_head on the last row, every row for a read-out (`MIMO_LOGITS`, logprobs) |

**The caches.** The host's stay canonical, in the layout the CPU keeps: a
full-attention layer `[ctx][kvh][hd]` (V `[ctx][kvh][vd]`), a windowed layer a ring of
`min(window, ctx)` rows with position `p` in row `p % rows`. The device mirrors each
one in exactly that layout (`chain_attn.comp` reads position-major rows and a ring) behind
a watermark per layer: positions below it (for a ring, the last `rows` of them) equal
the host's. A step lowers it to its first position and uploads what the window still
sees; a CPU step lowers it to its own; a restored photo (`pin_state_load` rewrites the
host's rings) drops the rings' watermarks to 0, so the next step uploads them again.
Prefix reuse needs nothing else.

**A windowed layer's attention.** One decode row reads the ring in place: its window
is exactly the ring's rows. A block of rows cannot (its later rows would overwrite slots
its earlier rows still see), so it gathers the window's earlier rows from the ring and
its own new rows into one position-major scratch, as the CPU's `attention()` copies
them, attends there, and writes its last `rows` rows into the ring. Both visit the
positions in the same order, so a row gets the same bits either way: with every matrix
on the per-row GEMV (`COLI_VK_GEMM_MIN_S=0`) and the tier off, the logits of every
position are the same bytes for a prompt in one block, one token at a time, and in
blocks of 3, 8 (the fixture's window) and 9 rows.

**A lost device.** MiMo has no recurrent state, and the new K/V rows reach the host's
caches only when the step's last frame has completed: a device lost in the middle of a
step loses nothing the host holds, and the CPU runs the step's remaining rows from where
the caches end. Nothing is rebuilt, and a turn with a picture recovers too. The line is
`[VK] mimo chain: the device was lost at position P; ...`.

**What stays off the device.** The router's top-k and the routed experts the tier does
not hold, the embedding gather, the vision tower (on the device only with
`COLI_VK_DENSE=1`, through the per-matrix path), and runs with `MIMO_TRACE` (the CPU's
per-layer dump), which the chain declines. MiMo has no MTP head. The chain also
declines a head dim above 256 or a dense matrix that did not reach the device.

**Correctness, on the tiny fixture** (6 layers: 4 windowed with a window of 8 and a sink
logit, 2 full, partial RoPE with two thetas, a value scale of 0.707, a dense layer 0, a
vision tower), on Lavapipe and on the Radeon 780M (`tests/vulkan_engines.sh
mimo-chain`): Xiaomi's vendor oracle passes with the chain on (greedy and teacher-forced
in every case, the native FP8/BF16 trunk, prefill in blocks of 3 and of 1, the picture);
every configuration gives the CPU's tokens and every position's logits within 1e-4 of
the largest one (measured over 42 configurations at most 2.0e-6 on Lavapipe and 2.7e-6
on the 780M, both with the picture's tower on the device; 1.7e-6 without it); serve
sessions give the CPU's frames (logprobs within 2.9e-4 on Lavapipe and 1.8e-4 on the
780M, on logits that reach 300); the prefix-reuse and photo tests pass bit for bit
against a cold engine. With `COLI_VULKAN` unset, the tokens and the logits bytes of 36
configurations (`MIMO_DENSE_BITS` 32, 0 and 8, blocks of 64, 3 and 1, four cases with
the picture) are the previous build's; with `COLI_VULKAN=1` and `COLI_VK_CHAIN=0` or
unset, those of 54 (the tier balanced deterministically, `COLI_VK_TIER_BALANCE=0`, since
its balance moves bits from run to run on either build).

**The default.** The chain's speed on a real MiMo model is not measured: no MiMo
checkpoint is on the test box (the smallest, Flash, has 309B parameters). The chain is
on for a discrete GPU (the common rule), off on an integrated GPU (`COLI_VK_CHAIN=1`
turns it on) and on a CPU device:

```
[VK] mimo: dense chain off (an integrated GPU with the expert tier: not measured; COLI_VK_CHAIN=0 off, 1 on, 2 prompts only)
```

### OLMoE and Inkling

olmoe (`olmoe_chain.h`) and inkling (`inkling_chain.h`) run the chain too, with the same
knobs, default rule and `[VK] <engine> chain: ...` line as qwen36.

| | olmoe | inkling |
|---|---|---|
| device, frame A1 | the layer before's routed sum joining the residual; the input norm; q/k/v; OLMoE's q and k norms over the whole projection; RoPE from the CPU's table; the K/V rows into the device mirror; attention; o_proj; the residual add; the post-attention norm; the router logits | the layer before's MoE output joining the residual (the routed sum, then each shared expert times its combine weight, `moe()`'s order, then the MLP's short convolution and the add); the input norm; q/k/v and the bias projection r; the short convolutions on K and V with their rings; the per-head q/k norms; the attention (`chain_relattn.comp`); the K/V rows into the device ring; o_proj; the attention's short convolution; the residual add; the post-attention norm; the router logits (routed and shared). A dense-MLP layer runs its MLP, global scale, short convolution and add in the same frame |
| host | softmax, top-k, the routed experts (the tier's batch and the CPU's share in rank order: the code `moe()` runs); the K/V rows into the host's cache; `PILOT`'s prefetch (below) | the sigmoid router with its bias, top-k, the joint combine weights, `TOPP`; the routed experts (`moe_ex`: `moe()`'s code, the device's logits handed in, the shared experts left out); the K/V rows into the host's cache |
| device, frame A2 (not waited for) | none: OLMoE has no shared expert | the shared experts, unweighted |
| last frame | the final norm and lm_head on the last row | the final norm, the division by the width multiplier, lm_head; the per-position heads (logprobs, teacher forcing) read the final rows on the host |

Inkling's attention is not qwen36's: a learned relative-position bias (an r projection
per row mixed through a per-layer bank, one bias per backward distance), the log-length
scale tau, a sliding window on five layers of six over a ring of K/V rows, and short
depthwise convolutions (residual inside) on K, V, the attention output and the MLP
output. Two shaders do it, made on first use so that the qwen chains never depend on
them: `chain_relattn.comp` (grouped attention with the bias bank mixed in the CPU's
order, tau from a host table computed as the CPU computes it, the window, the ring, and
the step's own rows read from its K/V scratch as `attention()` reads them, so a step
that wraps the ring never loses a row an earlier query of it reads) and
`chain_sconv.comp` (the convolution with its ring, in `sconv_apply`'s order, plus the
scalar multiply and divide of the dense MLP's global scale and the logits' width
multiplier). The shared experts join through `chain_ew.comp`'s `HC_APPLY` over one stream
(y += w[row] * x), the CPU's `os[d] += w * hh[d]`. Nothing in the chain stages a row
in shared memory past what the device allows: Inkling's hidden size (6144) and its
24576-wide dense MLP run through `chain_gemv.comp` or, past its staging, `qmatmul.comp`'s
GEMV reading from the buffer; `tools/make_tiny_inkling.py --wide` makes a two-layer
model at D = 6144 for the tests.

**State.** Both keep the host's K/V cache canonical and copy each step's rows back.
olmoe mirrors it behind qwen36's watermark. Inkling's mirror has the host's layout, a
ring on sliding layers, and a range of positions per layer that the host wrote alone
(a CPU step): the next chain step uploads the rows of its last `cap` positions first,
whatever positions they now hold, so a rewind to a pinned snapshot over a wrapped ring
reads on the device what the CPU reads. Inkling's four convolution states per layer run
on the device and come back at the end of every chain step (a few KB a layer), so the
host's copy is always current; a reset, a restored snapshot or a CPU step sends them
up again before the next chain step.

**Where they decline** (the per-matrix path, the state marked as the host's): inkling
under CUDA or Metal; inkling's bf16 matrices on a CPU whose bf16 dot rounds the
activations (AVX512-BF16), which the per-matrix path keeps on the CPU for the same
reason (`[VK] inkling chain: ... bf16 stays on the CPU`); a geometry outside the shaders
(head dim above 256, a bias bank wider than 64, more than 9 taps). olmoe runs `PILOT`:
the rows after attention come down with the frame and the prefetch reads them, the
rows after the MoE are those plus the routed sum (the float add the device makes).
Running PILOT under the chain showed a race of the CPU path's own: the prefetcher could
take the slot the forward pass was multiplying with as its LRU victim (at cap 1 the only
slot) and read another expert into it mid-matmul. A slot now counts its readers, and
neither eviction takes one being read. A device lost mid-step: olmoe redoes the step on
the CPU (attention only, nothing to rebuild); inkling rebuilds its K/V and convolution
states on the CPU from the prefix record, as qwen36 does.

**OLMoE-1B-7B on the Radeon 780M** (`allenai/OLMoE-1B-7B-0924`, converted with
`tools/convert_olmoe_merged.py`: int8 experts, f32 trunk; cap 64, `OMP_NUM_THREADS=8`,
1-min load under 2, no other engine running, the same binary for every arm, every tier
arm from the same history of an unrelated prompt). Cold: the model files dropped from
the page cache first (`posix_fadvise`, as the qwen bench does); warm: the run after.
Decode is 100 steps after a 25-token prompt: the engine's time for 101 new tokens minus
its time for 1 (the prefill alone), both from `TUNE decode`; prefill is a 512-token
prompt with one new token. Where a warm cell lists two numbers, they are separate
rounds.

| Decode, 100 tokens | cold | warm |
|---|---|---|
| CPU | 22.2 tok/s | 23.1, 23.2 tok/s |
| tier, trunk on the CPU | 12.6 tok/s | 12.8, 12.8 tok/s |
| tier and chain (`COLI_VK_CHAIN=1`) | 16.8 tok/s | 17.3, 17.2 tok/s |
| chain without the tier (`COLI_VK_TIER=0`) | 12.0 tok/s | 12.8 tok/s |

| Prefill, 512 tokens | cold | warm |
|---|---|---|
| CPU | 11.7 s | 10.5, 10.6 s |
| tier, trunk on the CPU | 6.3 s | 6.4, 6.4 s |
| tier and chain | 5.5 s | 5.5, 5.5 s |
| chain without the tier | 12.7 s | 11.4 s |

What the numbers say:
- **Against the tier alone the chain wins both**: decode 35% faster, the 512-token
  prompt 14% sooner. That is the comparison `coli_vk_chain_decide` makes on an
  integrated GPU with the tier on, so olmoe passes ON there.
- **Against the CPU, decode loses**: 17.3 against 23.1 tok/s. OLMoE's trunk is f32
  (1.49 GB read every token, more than its eight routed experts' 0.8 GB of int8); the CPU
  reads it faster than the GPU does at the clock it mostly holds. Prefill wins: 5.5
  against 10.5 s. `COLI_VULKAN=1` is opt-in, and on this box the CPU alone is the faster
  way to decode OLMoE; with the device on, the chain is the better of its two modes.
  `COLI_VK_CHAIN_PROF=1` put 95% of the chain's device time in the trunk's f32 GEMVs,
  34.8 ms a token (about 43 GB/s for its 1.49 GB); the GPU sat at its 800 MHz floor in
  78% of the clock samples with the chain, 95% with the tier alone.
- **Without the tier** the chain only moves the f32 trunk to the device and leaves every
  expert on the CPU: slower than the CPU in both. The default keeps it off there.
- **The text.** Every arm, cold and warm, printed the same 100 decode tokens, and the
  same first token after the 512-token prompt.

**Not measured**: a discrete GPU; OLMoE in serve sessions, past a 537-token context, with
`PILOT` (its experts fit in RAM here) or with the dense trunk on the device beside the
chain (`COLI_VK_DENSE=1`).

**Inkling: not measured.** No Inkling checkpoint runs on the box (the model is 975B),
so its integrated-GPU default is off (`COLI_VK_CHAIN_UNMEASURED`: the `[VK]` line says
"not measured"); `COLI_VK_CHAIN=1` turns it on. Its correctness is the tiny fixtures',
on Lavapipe and on the 780M (below), including the two-layer model at D = 6144.

**Tests.** `tests/vulkan_engines.sh inkling-olmoe-chain` gates every configuration on the
CPU run's tokens and every forward's logits within 1e-4 of the largest (`DUMP=<path>` in
both engines' ref mode writes them; measured 2e-7 relative on the fixtures, 6e-7 at
D = 6144): inkling's f32, dense-int4g64 and bf16 snapshots (on an AVX512-BF16 host the
bf16 one checks the clean decline instead), its int4 and int8 expert containers and
runtime quantizations, `TOPP`, the tier off, the trunk's device copies shared with the
per-matrix path, prefill in chunks of 3, the tiled GEMM, the per-row GEMV, prompts only,
D = 6144 (and the tier's expert batch at that width); olmoe's caps, 4-bit experts,
`PILOT` at caps 1 and 2 and `PILOT=3`, the tier off, the shared trunk, an eviction
budget, chunks, the GEMM, the per-row GEMV, prompts only; the device lost mid-decode in
both (inkling in a shared-expert frame and at a router); `tests/vulkan_chain_serve.py`
sessions (pins, prompt-cache extensions, a divergent prompt, logprobs) frame for frame
against the CPU, with the chain and with prompts only; and both engines' prefix-reuse,
dashboard and Brio tests with the chain on. `inkling-olmoe-chain-sanitize` runs the
chain's ops, a set of those configurations and a serve session of each engine under
ASan and UBSan. On the 780M (RADV) every one of those configurations and serve sessions
gave the CPU's tokens.

### GLM-5.2 and GLM-5.3 Flash on the chain

`glm_chain.h` (colibri) and `glm53_chain.h` (glm53) follow the recipe below with the
[MLA ops](#multi-head-latent-attention-on-the-chain-vkc_mla). What runs where, per
layer:

| | colibri (GLM-5.2) | glm53 (GLM-5.3 Flash) |
|---|---|---|
| device, frame A1 | the previous layer's MoE output joining the residual (routed, then the shared expert, then the add: the CPU's order); the input RMSNorm; the MLA attention: q_a, its norm, q_b, kv_a, the latent norm, interleaved RoPE, the new latent and rope rows into the device cache, on a DSA layer the index key (wk, LayerNorm, RoPE) into its cache and, past `index_topk` (or with `DSA_FORCE`), each row's top-k, reused by the shared layers after it; the absorbed core over the cache or the selection, the value rows, o_proj; the add; the post-attention norm. A dense layer runs its MLP here and has no host step | the previous layer's FFN branch (the routed sum plus the shared expert) written back into the hc_mult streams; every site through mHC (the mix, the split with Sinkhorn, the collapse, the write back); the input RMSNorm; a KDA layer (its projections, the short convolution with its window, the delta rule with its state, the output norm and gate, o) or an MLA layer (the projections into the cache, the k-pooled indexer: index keys and pool gates into their caches, each completed pool's key, every row's pools; the absorbed core over the selection, the values, o); the FFN site's entry and norm; a dense layer's MLP (clamped SwiGLU) |
| host | moe() on the normalized rows without the shared expert: the f32 router with every routing option, the routed experts (the tier's batch and the CPU's share); the new KV rows copied into the host's cache | the router and the routed experts (the tier's batch and the CPU's share); the new MLA rows copied into the host's cache |
| device, frame A2 (not waited for) | the shared expert | the shared expert (clamped SwiGLU) |
| after the last layer | the final rows back to the host, which runs the final norm and lm_head as before | the final streams back to the host, which collapses them and runs the final norm and the head as before |

**The state.** The KV caches (GLM-5.2's latent, rope keys and index keys; GLM-5.3's
latent, index keys and pool gates) stay the host's: each step copies its new rows back,
and the device mirror has a watermark that every host write lowers (the CPU's attention,
`kv_alloc`, a slot adopting another slot's rows, another KV state or session bound, a
pin restored). GLM-5.3's pool keys have a watermark of their own. MLA has no recurrent
state, so a rejected draft is rows the next step rewrites. GLM-5.3's KDA state and
convolution windows stay on the device while the chain runs, for one session at a time:
the host's copy is brought back before a pin or a state capture reads it, before a CPU
forward of the session and when another session takes the device, and goes up after a
pin or a state is restored.

**Drafts and the MTP head.** colibri's speculative decode runs as before: the verify
rows go through the chain, the MTP head stays on the CPU and reads the chain's final
rows, and n-gram drafts work the same way. glm53 has no draft path.

**A lost device.** The forward that failed runs again on the CPU from its input (the
chain keeps the caller's rows untouched until its last chunk is through), and the CPU
runs from there. colibri has nothing to rebuild (its cache is the host's). glm53
rebuilds the KDA state on the CPU from the input rows the chain records since the host's
copy was last current (embedding rows, or the vision tower's), a prefill's worth of CPU
work.

**Declined** (the CPU path runs, the state synced first): a ragged multi-slot decode
batch (a single-slot serve's one-row batch takes the chain), a layer range (a segment),
a quantized KV cache (KV8, KV_TQ), PILOT, LOOKA, the exact verify of
`COLI_EXACT_VERIFY`, the CUDA backend, matrices with no device form (int2, E8/IQ3, fp8
dense matrices), and geometries past the ops' limits. With the chain on, the per-matrix
switches (`COLI_VK_DENSE`, `COLI_VK_ATTN`, `COLI_VK_DEV2`) keep working beside it: the
chain's tensors are their device copies where they made one.

**Arithmetic.** f32 activations throughout, as the CPU's f32 paths: colibri's CPU int8
dot (`IDOT`, on by default for int8 and, from two rows, int4 rows) rounds activations,
so the tests set `IDOT=0` for those trunks, as for the tier. On the fixtures every
configuration gives the CPU's tokens, and every logits row is within 2e-6 of the largest
logit (Lavapipe: 1.9e-6 at worst, colibri's int4-g64 experts on the tier, 6.3e-7 and
below everywhere else).

**The default**: both engines pass `COLI_VK_CHAIN_UNMEASURED`, so the chain is off on
an integrated GPU (`COLI_VK_CHAIN=1` turns it on, `2` for prompts only) and on a
discrete GPU follows the rule above. No GLM checkpoint was run: the 780M box has none,
and both models are hundreds of GB. Speed is not measured; the tests prove the tokens on
the tiny fixtures, on Lavapipe and on the 780M:

```
[VK] colibri: dense chain off (an integrated GPU with the expert tier: not measured; COLI_VK_CHAIN=0 off, 1 on, 2 prompts only)
```

### Kimi K3 on the chain

`kimi_k3_chain.h` (kimi_k3) follows the recipe below with the
[MLA ops](#multi-head-latent-attention-on-the-chain-vkc_mla), GLM-5.3's KDA ops and three
of its own. Kimi K3's residual is AttnRes: per row a running prefix and a snapshot every
`attn_res_block_size` layers, mixed by a softmax before the attention and before the MLP
of every layer and once at the end. What runs where, per layer:

| | kimi_k3 |
|---|---|
| device, frame A1 | the previous layer's MoE output joining the prefix (the routed sum's RMSNorm and latent up-projection, plus the shared experts, then the add: `moe_forward`'s order); the attention site's residual mix and a block boundary's snapshot; the input RMSNorm; a KDA layer (q, k, v, the full-rank output gate, the decay's two f32 matrices and beta's; the short convolution with its window, the delta rule with its state, the output norm and gate, o_proj) or a gated MLA layer (q_a, its norm, q_b, kv_a, the latent norm, the new rows into the cache; the absorbed core over the cache, the values times the sigmoid gate, o_proj); the prefix update; the MLP site's residual mix and post-attention norm; a dense layer's SiTU-GLU MLP and its add (no host step), or the f32 router's logits and the latent down-projection |
| host | the router (sigmoid, the top-k with the correction bias, `K3_TOPP`) and the routed experts in the latent: the tier's batch and the CPU's share, added in the union's order; the new MLA rows copied into the host's cache |
| device, frame A2 (not waited for) | the shared experts (SiTU-GLU at full width) |
| after the last layer | the output residual mix and the final RMSNorm of every row, lm_head on the last; the normalized rows come back when the host wants the logits of every row (`K3_LOGITS`, `K3_VAL_LOGITS`, a logprobs request) and the host runs lm_head on them |

Kimi K3's MLA is NoPE: the qk_rope parts of the query and of the shared key are used as
the projections leave them. `vkc_mla_qkv` rotates them by angle 0 (a table of cos 1 and
sin 0), which in float is the identity.

| Op | Shader | What it does |
|---|---|---|
| `vkc_ares_mix` | `chain_ares` (mode 0) | `res_mix`: per row, the softmax of `(v . w) / sqrt(mean(v^2) + eps)` over the block snapshots and the prefix, and their weighted sum, in snapshot order (up to 15 snapshots) |
| `vkc_situ` | `chain_ares` (mode 1) | SiTU-GLU, `b1*tanh(g/b1)*sigmoid(g)*b2*tanh(u/b2)`, in the CPU's order |
| `vkc_kda_rec_flags` | `chain_kda` | the KDA recurrence with Kimi K3's two differences from GLM-5.3: the decay's `exp(A_log)` given as the engine keeps it (`VKC_KDA_EXP_A`), and `kda_forward`'s order of the l2 norms (the eps after the squares, q normalized, then scaled) and of the update, `k * ((v - mem) * beta)` (`VKC_KDA_K3`). `vkc_kda_rec` is the same op with no flags, unchanged for glm53 |

`make vk-chain-check VK=1` runs Kimi K3's KDA layer against `kda_forward`'s arithmetic
over two submissions (the state and the window carried, inputs small enough that the
l2 eps counts), `res_mix` over 0 to 15 snapshots at row strides (D up to 7168) and
SiTU-GLU at Kimi K3's constants. The KDA layer within 3.8e-7 of its largest output on
Lavapipe, 5.0e-7 on the Radeon 780M and 5.4e-7 on the Iris Xe (Dozen); the residual mix
within 2.4e-7, 3.3e-7 and 3.3e-7; SiTU-GLU within the test's 1e-6 on all three.

**The state.** The MLA caches (`Lc`, the normalized latent, and `Rc`, the qk_rope part)
stay the host's: each step copies its new rows back, and the device mirror has a
watermark per layer that a CPU forward, a reset and a grown cache (`kv_alloc`) lower.
The KDA state and the three convolution windows of every KDA layer stay on the device
while the chain runs (96 heads of 128 x 128 floats and three windows of 12288 x 4: 6.9
MB a layer on the full model): the host's copy is brought back before a recurrent-state
checkpoint (`COLI_K3_CKPT`, a `pin=1` photo) or a CPU forward reads it, and goes up
after a reset (a fill with zeros on the device) or a restored photo. Prefix reuse needs
nothing more: the reused positions are rows below the watermark and a KDA state that
already sits where the next step expects it.

**Drafts.** kimi_k3 has no MTP head and no draft path.

**A lost device.** The forward that failed runs again on the CPU from its input rows
(the chain never writes them). If the device held the newest KDA state, that state is
rebuilt on the CPU first: from the host's copy, current at the position where it last
went up or came back, through the prefix record's ids up to where the device was, a
prefill's worth of CPU work; from zeros if the loss interrupted a copy of the state to
the host after part of it had landed. The CPU runs from there. (Both rebuilds were
checked on the 780M, where that copy takes frames, by faults placed inside it: a serve
session with checkpoints gave the CPU's frames either way.)

**Declined** (the CPU path runs, the state synced first): the CUDA expert tier,
`KIMI_DSA_INDEXER=1` (its index cache is filled on the CPU), the validation dumps that
read every layer on the host (`K3_TRACE`, `K3_VALIDATE_LAYER`, `K3_DEBUG_OUT`), a model
without its head (`K3_LAYERS`), a Segment's layer range, and geometries past the ops'
limits (a KDA head above 128 floats, a convolution above 8 taps, `kv_lora` above 1024,
`qk_rope` above 128 or odd). The chain's tensors are its own except the shared experts'
under `COLI_VK_DENSE`, which it shares with the per-matrix path: a forward the chain
declines (`COLI_VK_CHAIN=2`'s decode) runs on the CPU as before.

**Arithmetic.** f32 activations, as the CPU's dense kernels (int8 rows and int4-g64
alike); the routed experts are the tier's or the CPU's, as without the chain. The tiny
fixture amplifies rounding at a few positions: the CPU against itself, with only its
RMSNorm's sum taken in float instead of double, moves the logits by up to 1.4e-4 of the
largest one on the f32 trunk and 1.0e-3 on the 8-bit one, and the served logprobs by up
to 4.1e-3, at the positions where the chain moves them most (Lavapipe: 1.8e-4, 4.5e-4
and 6.1e-3; the 780M: 2.3e-4, 9.9e-4 and 3.3e-3). The tests hold every logits row within
2e-3 of the largest and the logprobs within 2e-2; the tokens are the CPU's in every
configuration. With the CPU's int8 expert activations (`K3_IDOT=1`, tier off) a flipped
int8 step moved the logits by 8.3e-3 on Lavapipe (no step flipped on the 780M), the
tokens unchanged; that configuration is gated on its tokens.

**The default**: kimi_k3 passes `COLI_VK_CHAIN_UNMEASURED`: off on an integrated GPU
(`COLI_VK_CHAIN=1` turns it on, `2` for prompts only), on a discrete GPU the rule above.
No Kimi K3 checkpoint was run (1.56 TB; the 780M box has none): speed is not measured,
and the tests prove the tokens on the tiny fixture, on Lavapipe and on the 780M:

```
[VK] kimi_k3: dense chain off (an integrated GPU with the expert tier: not measured; COLI_VK_CHAIN=0 off, 1 on, 2 prompts only)
```

### Adding an engine to the chain

The recipe qwen36_chain.h and qwen38_chain.h follow, for the engines still on the
per-matrix path:

1. **Parameters and tensors once.** Pack every norm weight and small parameter vector
   into one device buffer (`vkc_buf` + one `vkc_write`) and pass offsets; resolve every
   matrix to the device copy the per-matrix path already uploads (`coli_vk_tensor_ensure`
   into the engine's own `vk` field), so the two paths never hold a matrix twice.
2. **Own the state explicitly.** Attention caches: keep the host's canonical, copy each
   step's new rows back (a few KB a layer), mirror on the device behind a watermark
   lowered by every CPU write. Recurrent state too large to copy per token: on the
   device, with a "who holds the newest copy" flag synced at every host read (snapshots,
   prompt caches) and every host write (resets, restores). A speculative verify writes a
   snapshot of the recurrent state at its first row in the shader that walks the rows,
   and a rejection swaps buffers.
3. **One frame per layer up to the first thing the host must decide** (the router's
   top-k for the routed experts), the CPU-independent tail (the shared expert) in a
   frame nobody waits for, and the join (routed sum + shared, in the CPU's order) at the
   head of the next layer's frame.
4. **The CPU's arithmetic order wherever it is cheap to keep** (the conv sum order, the
   MoE combine, the RoPE angles from a host table), f32 activations, and a gate per
   configuration in `tests/vulkan_engines.sh` against the CPU's tokens and logits.

What each remaining architecture needs on top of today's shaders:

| Engine | Attention / mixer | New pieces |
|---|---|---|
| deepseek_v41, kimi_k3 (MLA layers) | MLA: q_a/kv_a, the latent norms, q_b, RoPE on the rope dims, a latent + rope cache | the [MLA ops](#multi-head-latent-attention-on-the-chain-vkc_mla) take any geometry (q_lora or none, NoPE, the scale and cos/sin table from the engine, a gate on the values for Kimi K3); colibri and glm53 run them ([above](#glm-52-and-glm-53-flash-on-the-chain)). DeepSeek V4.1's attention has sinks and its own sparse selection: those are not in the ops yet |
| deepseek_v4 | MLA with compressed (CSA) and hierarchical (HCA) KV, mHC | the manifold hyper-connections are `vkc_mhc` (GLM-5.3's, the same arithmetic) and the clamped SwiGLU too; the compressors' rolling windows are rings like the conv's, snapshotted the same way; the CPU rounds activations to E4M3 before its fp8 matmuls, so the chain needs that rounding as an element-wise op to keep the same arithmetic |
| kimi_k3 (KDA layers) | Kimi Delta Attention: a gated delta rule whose decay is a vector over the key channels | on the chain ([above](#kimi-k3-on-the-chain)): `vkc_kda_conv` and `vkc_kda_rec_flags` with Kimi K3's options; its full-rank output gate and low-rank decay are matmuls before the op |
| mimo | sliding-window attention (and full layers) | done: [MiMo-V2.6 on the chain](#mimo-v26-on-the-chain) (`chain_attn.comp` with a window, a ring of W rows, V's own head dim and a sink) |
| inkling | grouped attention with a relative-position bias, a sliding window and short convolutions; MoE with shared experts | in the chain ([OLMoE and Inkling](#olmoe-and-inkling)): two shaders of its own, `chain_relattn.comp` and `chain_sconv.comp`; the shared experts join through `HC_APPLY` |
| olmoe | attention with q/k norm, MoE without a shared expert | in the chain ([OLMoE and Inkling](#olmoe-and-inkling)) with qwen36's ops as they are |

### Multi-head latent attention on the chain (`vkc_mla`)

The MLA layers (GLM-5.2, GLM-5.3, the DeepSeek V3 family's attention, Kimi K3's MLA
layers) have chain ops of their own, in `vk_chain.h`, written for any geometry: H heads
of Q no-position and R rotated query floats (R = 0 for NoPE), V value floats, a latent of
K floats (`kv_lora`), q_lora from 0 (q straight from the hidden rows) up, RoPE as rotate
-half or as interleaved pairs, the softmax scale and the cos/sin table from the caller (a
YaRN model passes its scaled frequencies, its mscale on the table and mscale squared on
the scale). The cache on the device holds, per position, the normalized latent and the
rotated shared key; the engine owns it, mirrors the host's rows behind a watermark as the
GQA engines do, and copies each step's new rows back.

| Op | Shader | What it does |
|---|---|---|
| `vkc_mla_qkv` | `chain_norm`, `chain_mla` (mode 1) and `vkc_matmul` | q_a, its RMSNorm and q_b (or q_b alone), kv_a, the latent's RMSNorm into the cache row, RoPE on q's rotated part and on the shared key into the cache row, a copy of the new rows for the host |
| `vkc_mla_attn` | `chain_hgemv`, `chain_mla` (mode 0) | the absorbed query (each head's Q values times its key rows of `kv_b`, read transposed, or a `[H*K x Q]` matrix), the attention core over the cache with an online softmax (the causal range or a selection list with skipped entries), the value rows on the softmax-weighted latent, an optional sigmoid gate, o_proj |
| `vkc_mla_rope`, `vkc_mla_lnorm` | `chain_mla` (modes 1, 2) | RoPE over segments from a host table, in place or into another buffer; LayerNorm with weight and bias (an indexer's key norm) |
| `vkc_dsa_select` | `chain_dsa` | a token-level DSA indexer: each position's score `(sum_h [d_h > 0] w_h d_h) * wscale`, `d_h = (q_h . k_t) * qscale`, and the top-k in the CPU's order (above the k-th score in position order, then the ties in position order), or "every position" while the context is within top-k |

The weights stay in the format the engine already holds them in (int8 and int4 rows,
int4 and int8 and fp8 in groups, int3-g64, MXFP4, f32, bf16): the per-head blocks read
them where the backend uploaded them. Between `vkc_mla_qkv` and `vkc_mla_attn` an engine
records what reads the projections (a DSA indexer reads the normalized q latent the first
leaves in its scratch). Limits: K up to 1024, R up to 128, Q up to 1024 where kv_b's key
rows are read transposed, an indexer of up to 64 heads and 4096 query floats.

GLM-5.3 Flash adds three more things, as ops of their own in `vk_chain.h`, which the
DeepSeek V4 and Kimi K3 chains can take as they are:

| Op | Shader | What it does |
|---|---|---|
| `vkc_dsa_pool_keys`, `vkc_dsa_pool_select` | `chain_dsa` (modes 1, 2) | the k-pooled indexer (`sparse_index.h`): a pool's key, the per-channel softmax mixture of its members' keys under their gate logits plus a position bias, computed once when a step completes the pool; each row's top pools in rank order (score, then the lower pool), their positions, the incomplete tail, -1 in the unused slots |
| `vkc_kda_conv`, `vkc_kda_rec` | `chain_kda` | Kimi Delta Attention (`delta_attention.h`): the short convolution with its window, the gated delta rule with a decay per key row and the state on the device, GLM-5.3's output RMSNorm and sigmoid gate; the l2 sums and the norm's sum in the CPU's order |
| `vkc_mhc` | `chain_mhc` | manifold-constrained hyper-connections (`hyper_connections.h`): the mix logits' split with Sinkhorn, the collapse, the write back, the plain mean of the last collapse; and a clamped SwiGLU |

`make vk-chain-check VK=1` runs them against double-precision references of colibri.c's
absorbed attention: every weight format both ways through the per-head blocks, RoPE in
both styles in place and into a cache row, LayerNorm with and without a bias, and the
layer op over eight geometries (q latent or none, NoPE, `kv_b` or the split halves,
selection lists with skipped entries, a gate, prefill rows after earlier ones, a long
context from a nonzero start, GLM-5.2's head shape, a latent of 1024). Measured on
Lavapipe and on an Intel Iris Xe (Mesa's Dozen): the layer's output within 5e-7 of its
largest value and the new cache rows within 7e-7 (the test's bound is 2e-5). The
indexer's selection is checked bit for bit, ties included, on scores that are exact in
float on both sides. The k-pooled selection is checked slot for slot against `sparse_index.h` the same
way, the KDA layer against `delta_attention.h` over two submissions (the state and the
window carried between them, inputs small enough that the l2 eps counts), the mHC ops
against `hyper_connections.h`; each within 3e-7 on Lavapipe and on the Iris Xe.

## Adding an engine to the tier

Every MoE engine here is on the tier ([the table above](#the-routed-expert-tier-vk_tierc));
these are the steps the next one takes. They are in [`c/vk_tier.h`](../c/vk_tier.h);
in short:

1. **Describe the experts** (`VktConfig`): geometry, how RAM holds gate/up and down
   (`VktSrc`: int8 per row or grouped, the int8 copy of an int4 container, int4
   pairs signed or `v+8`, `expert_ffn.h`'s planar int4-g64, int3-g64, MXFP4 with f32
   or ue8m0 scales, fp8 per group or in square blocks, bf16, f32), the activation
   (`VKT_ACT_SWIGLU` with an optional clamp, `VKT_ACT_SITU`, `VKT_ACT_SWIGLU_V4`), the most assignments a
   step carries, the RAM the expert cache may still take and the dense bytes still to
   come to the device; optionally `.max_experts`, a count the engine's users already
   size its device tier in (GLM-5.2's `COLI_VK_EXPERTS`). `atexit(coli_vk_shutdown)`,
   then `vkt_init(&cfg, rt_counts_all())` after the device and the history, then
   `atexit(vkt_shutdown)` when it succeeds: at exit the tier lets go of its experts
   first and the device is destroyed before the drivers unload, whether or not the
   tier started (`vkt_init` makes the expert batch's pipelines before it can refuse).
2. **Warm start** (optional): `vkt_plan`, read each planned expert into a buffer of
   the loader's own (any number of threads), `vkt_put`, then `vkt_put_done`.
3. **Every MoE step**: `vkt_issue(layer, x, S, K, idx, taken)`; compute the pairs not
   taken on the CPU into rows of their own, and `vkt_note` every expert whose bytes
   are in RAM; the shared expert; `vkt_join` (when the issue took any); then add
   every rank of every row in order, the device's row where `taken`. A failed join
   (device lost) leaves the taken pairs to the CPU and turns the tier off.
   `vkt_issue_w` also hands the route weights (for an activation that applies them on
   the device); `vkt_wants(l, e)` says whether `vkt_note` would take an expert, for an
   engine that must convert its RAM form first; `vkt_begin_forward()` marks a
   forward's start for a model whose layer index never goes back (a single MoE layer).
4. **Report**: `vkt_report("run"|"turn", ram_hits, disk_loads)` beside the engine's
   `[VK]` line; `vkt_resident(l, e)` gives EMAP its tier 2.

## Correctness

- `gcc -O3 -DVK_TEST backend_vulkan.c -o test_vk -lvulkan -lm && ./test_vk
  shaders/qmatmul.spv` runs a CPU-reference exactness harness over every
  primitive (GEMV int4/int8 across shapes incl. the long-row o-projection,
  fused gate+up, the full expert group sync and async, the matmul pair, and
  the absorb attention core incl. causal S=2, kv_start windows, int8, and
  long-context cases), and both tiled GEMMs for every weight format at S = 16, 64
  and 512 with odd I and O, tail groups and odd group sizes; a GEMM case fails if the
  call did not take the GEMM it names. Typical maxrel ~1e-5..2e-3 (fp32 reduction
  order). `COLI_VK_TEST_MATMUL_ONLY=1` stops after the GEMV and GEMM format cases.
- Engine-level: greedy decode with the full stack matches the pure-CPU
  engine token-for-token on the validation prompt.
- The expert tier: `tests/test_vk_alloc` (in `make check`) runs the sub-allocator
  against a byte map, 40,000 random steps included; `make vk-tier-check VK=1` runs
  `vk_tier.c` against a CPU reference for every expert source format, the warm
  start, adaptation with eviction, partial batches, and with `COLI_VK_TIER_SYNC=1` a
  promotion that displaces a resident while a batch is in flight (it must wait for
  the join's free, not fail and shrink the budget); the harness runs the expert
  batch for every weight format and every activation, a row's bits checked
  independent of the batch, and the tier pool's budget with frees while a batch is
  in flight. `tests/vulkan_engines.sh qwen` gives the CPU's tokens with the tier on
  in every qwen36 and qwen38 expert format, under eviction, with MTP and with the
  trunk on the CPU; `qwen-sanitize` runs the same under ASan and UBSan.
  `tests/vulkan_engines.sh inkling-olmoe` does the same for inkling (every expert
  format above, under the f32, bf16 and dense-int4g64 snapshots, `TOPP`) and olmoe
  (with `PILOT`'s worker), under eviction, with a warm start, and through both
  engines' serve tests (prefix reuse, the dashboard, Brio); `inkling-olmoe-sanitize`
  runs them under ASan and UBSan. `kimi` and the mimo half of `mimo-qwenimage` do it
  for Kimi K3 and MiMo: their vendor oracles with the tier on, the CPU's
  tokens in every dense format with the trunk on the device and on the CPU, a budget
  of two experts that must evict, the old switches, Kimi K3's warm start;
  `kimi-mimo-sanitize` runs the tier's configurations under ASan and UBSan.
  `tests/vulkan_engines.sh deepseek` does it for deepseek_v41 (every tiny oracle, the
  40-token prompt, DSpark, eviction) and deepseek_v4 (the three oracle cases on a 4-
  and an 8-expert fixture with pinned rows16 experts, eviction, the warm start, the
  served logprobs), and `deepseek-sanitize` under ASan and UBSan. DeepSeek V4's
  activation has a case of its own in the harness, checked bit for bit, and in
  `vk-tier-check`.
  `tests/vulkan_engines.sh glm` does the same for colibri and glm53 (every expert
  format, warm and cold, prefill, eviction, the trunk and attention core on the
  device, `COLI_VK_DEV2`), `glm-sanitize` under ASan and UBSan.
- The dense chain: `make vk-chain-check VK=1` (`tests/test_vk_chain.c`) runs every chain
  op against a CPU reference; `tests/vulkan_engines.sh qwen-chain` gives the CPU's tokens
  with the chain in every qwen36 geometry and expert container and every qwen38 format,
  the last logits within 1e-4 of the largest one where both sides use f32 activations
  (measured 2e-7 and below on the fixtures), prefill in chunks, an image, MTP drafts
  rejected, accepted and alternating, a device lost mid-run, the qwen38 oracle targets,
  the prefix-reuse contract and serve sessions frame for frame; `qwen-chain-sanitize`
  runs the chain under ASan and UBSan. `mimo-chain` does it for MiMo: the vendor
  oracle with the chain on, the CPU's tokens and every position's logits in every dense
  form, block size, case and tier setting, the window boundary bit for bit across block
  sizes, prompts only, a device lost in four places, the prefix-reuse and photo tests
  and serve sessions; `mimo-chain-sanitize` under ASan and UBSan.
  `inkling-olmoe-chain` and
  `inkling-olmoe-chain-sanitize` do the same for inkling and olmoe
  ([OLMoE and Inkling](#olmoe-and-inkling)); `vk-chain-check` covers their ops
  (`chain_sconv.comp`, `chain_relattn.comp`, the GEMV and the norm at D = 6144).
  `glm-chain` does the
  same for colibri and glm53: every expert format and trunk, prefill in chunks, the DSA
  selection active, n-gram and MTP drafts accepted and rejected, glm53's image, KDA state
  and swiglu_limit 0, a device lost (glm53's KDA state rebuilt), serve sessions with pins
  and two KV slots, and glm53's pin-branch harness; `glm-chain-sanitize` runs them under
  ASan and UBSan.
  `kimi-chain` does it for Kimi K3: Moonshot's
  oracle with the chain on, every dense format, prefill a token at a time and in chunks,
  the tier's eviction, prompts only, a device lost in a prompt, inside a chunked forward
  and mid-decode (the KDA state rebuilt), serve sessions with prefix reuse and
  recurrent-state checkpoints in RAM and on disk; `kimi-chain-sanitize` under ASan and
  UBSan.
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

- GLM-5.2 (this section's engine): the per-matrix attention core serves `S<=4`; prefill
  uses the CPU/batched attention paths (dense projections do run on VK at prefill),
  except with the dense chain (`COLI_VK_CHAIN=1`), which runs the whole layer, prefill
  and the DSA selection included. Its routed experts are on the shared expert tier,
  which serves prefill too.
- The expert tier's uploads are host writes into host-visible device memory: a
  discrete card needs Resizable BAR for them (above). A staging copy on a transfer
  queue, for cards without it, is not written.
- Without the dense chain, DSA top-k selection, ragged multi-slot serving, and
  quantized-KV caches fall back to the CPU attention path; with it, the DSA selection
  runs on the device.
- Not yet done: a fully resident-layer pipeline for the engines other than qwen36,
  qwen38, colibri and glm53 ([the dense chain](#the-dense-chain-vk_chainc) is theirs), Polaris/gfx803 validation on real
  hardware (the shaders use dynamic subgroup sizes and are wave64-safe by
  construction). The cooperative-matrix GEMM is measured on RDNA3 only.
