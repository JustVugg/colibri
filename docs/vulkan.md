# Vulkan backend (any GPU with a Vulkan 1.2 driver)

colibrì includes an opt-in Vulkan compute backend that runs the whole GLM
decode compute path on any GPU a Vulkan driver can see — no CUDA, no ROCm.
That includes cards the vendor stacks have dropped (ROCm 7 removed Polaris:
an RX 580 runs here via RADV) and, measured on an RX 9070 (RDNA4), it is
*faster* than the ROCm/HIP backend on the same card.

The same models on the CPU, Vulkan and CUDA, one machine and one build: [gpu-parity.md](gpu-parity.md).

```bash
cd c
make glm VK=1                # needs the Vulkan headers + glslc (shaderc) for the shaders
COLI_VULKAN=1 COLI_VK_DENSE=1 COLI_VK_ATTN=1 \
PIN=<model>/.coli_usage PIN_GB=0 COLI_NO_OMP_TUNE=1 \
./coli run "Hello" --topp 0.7
```

Requirements: a Vulkan **1.2** driver with
`GL_KHR_shader_subgroup_arithmetic` (any Mesa RADV, AMDVLK, NVIDIA or Intel
ANV driver from the last several years), plus the Vulkan headers and `glslc`
at build time. On Debian and Ubuntu that is `sudo apt install libvulkan-dev glslc`
(`glslc` is its own package there, not part of `libvulkan-dev` or `vulkan-tools`;
without it `make VK=1` stops at `glslc not found`). The
backend picks the most capable physical device (discrete > integrated) and
degrades to the CPU path on any failure — a wedged GPU can slow a run, never
corrupt it.

Nothing links the Vulkan loader: the backend opens it (`libvulkan.so.1`,
`vulkan-1.dll`, on macOS `libvulkan.1.dylib` or MoltenVK; `COLI_VK_LOADER`
names another) when `COLI_VULKAN=1` asks for a device. A `VK=1` binary therefore
starts on a machine with no Vulkan at all and runs on the CPU there, and with
`COLI_VULKAN=1` says `[VK] no Vulkan loader (...)`. The release archives for
Linux and Windows are built this way, with the shaders in `shaders/` next to the
engines: unpacked, they run on a Vulkan GPU with nothing to build, and
`coli setup` turns the GPU on when it finds one.

Set `COLI_NO_OMP_TUNE=1` on multi-core boxes: the engine's OMP self-tune
(active spin-wait) is skipped under `COLI_CUDA`/`COLI_METAL` but not under
Vulkan, and spinning worker threads starve the async I/O pool (measured
CPU expert bandwidth 28 → 5 GB/s without it).

**Discrete cards without Resizable BAR** expose HOST_VISIBLE|DEVICE_LOCAL memory
only as a ~256 MB window. Writing the weights through a mapping of that window, as
the backend does on every other device, either fails past it (NVIDIA) or lands in
system RAM read over PCIe (RADV: measured 0.11 against 0.24 tok/s either side of
the BIOS toggle on an RX 9070 XT). On such a card the backend now copies resident
data into device-local memory through a staging buffer instead, on its own; see
[Memory placement without Resizable BAR](#memory-placement-without-resizable-bar).
Unified-memory APUs and cards with Resizable BAR keep the mapped path.

The compiled shaders are found via `COLI_VK_SHADERS` (either the
`qmatmul.spv` file or the directory holding the `.spv` set); unset, the
engine looks in `shaders/` next to the binary (Linux, Windows and macOS),
then relative to the CWD.

### NVIDIA in a container

In a CUDA container (seen with `nvidia/cuda:12.8.1-devel-ubuntu24.04`, driver
580) the driver's Vulkan file, `/etc/vulkan/icd.d/nvidia_icd.json`, names the
GLX library, `libGLX_nvidia.so.0`. The loader skips it (`libXext.so.6: cannot
open shared object file`), and with `libxext6` installed it still cannot start
it (`Could not get 'vkCreateInstance' via 'vk_icdGetInstanceProcAddr' for ICD
libGLX_nvidia.so.0`): `vulkaninfo` shows `llvmpipe`, the CPU, not the card.
The driver's EGL library starts with no display: a file like it naming
`libEGL_nvidia.so.0`, given with `VK_DRIVER_FILES` (`VK_ICD_FILENAMES` for an
older loader), shows the card. The `api_version` below is driver 580's; take
yours from the original. Set the variable for every command that opens Vulkan,
`coli plan` included.

```bash
cat > nvidia_icd_egl.json <<'EOF'
{ "file_format_version" : "1.0.1",
  "ICD": { "library_path": "libEGL_nvidia.so.0", "api_version" : "1.4.312" } }
EOF
VK_DRIVER_FILES=$PWD/nvidia_icd_egl.json vulkaninfo --summary   # lists the card
```

### Windows (MSYS2)

The release's `windows-x86_64.zip` has the engines built this way. To build
them, in the MSYS2 **UCRT64** shell ([quickstart.md](quickstart.md)) add the
Vulkan headers and `glslc`:

```bash
pacman -S --needed mingw-w64-ucrt-x86_64-vulkan-headers mingw-w64-ucrt-x86_64-shaderc
cd c
make colibri.exe VK=1
```

The binary is statically linked like the default Windows build and imports
nothing for Vulkan: it opens `vulkan-1.dll`, the loader every GPU driver
installs in `System32`, from the program's directory or `System32` (never the
current one), so it runs outside MSYS2 with nothing added to `PATH`, and on a
machine without a Vulkan driver it runs on the CPU. It finds `shaders\` next to
itself wherever it is started from.
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
| MLA attention core | `COLI_VK_ATTN=1` | One dispatch per layer: absorbed query, scores over the KV window, softmax, weighted latent, value rows, **fused with the o-projection** (the context vector never leaves the GPU). The latent/rope KV lives in a persistent per-layer device mirror, appended ~2.3 KB/token/layer with the same invalidation points as the CUDA KV shadow. The mirror grows with the conversation (256 rows a layer, then powers of two up to the context), filled again from the host when it grows; a layer the device has no room for computes its attention on the CPU, and one line says so. |

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
| qwen36 (Qwen3.6, Qwen3-Coder, Qwen3.8-27B, Clef) | the dense trunk; routed experts on the expert tier | int8 rows; int4-g64 with `COLI_DENSE_BITS=4`; f16 (fmt 14) with `COLI_DENSE_BITS=16`; f32 with `COLI_DENSE_I8=0`; experts int4-g64, int4 per row, int8 per row or gs64 | DeltaNet `dn_a`/`dn_b`, vision tower, Clef's joint head, the experts the tier does not hold |
| qwen38 (Qwen3.8 Flash Next) | the trunk; routed experts on the expert tier, the MTP head's too on a discrete GPU | int8 trunk rows, bf16, f32 (`Q38_NATIVE_BF16=0`); experts int4-g64 (sidecar), FP8 128x128 blocks, bf16 | the experts the tier does not hold |
| inkling | dense and shared-expert matrices; routed experts on the expert tier | int8 and int4-g64 (dense-int4g64 container), f32, bf16; experts int4 or int8 per row (container or runtime quantization), f32 | embedding and audio lookups, CUDA residents (with CUDA or Metal on, the experts too); bf16 on CPUs with the AVX512-BF16 dot (see below); the experts the tier does not hold |
| olmoe | attention q/k/v/o, router, lm_head; routed experts on the expert tier | f32; experts int8 per row | embedding, the experts the tier does not hold |
| kimi_k3 (Kimi K3) | the shared experts' matrices (one row at a time: decode); routed experts on the expert tier | shared experts int8 rows, int4-g64, f32 (`K3_BITS`); experts MXFP4 with ue8m0 scales (fmt 7), SiTU-GLU in the latent space | KDA, MLA, the latent projections, router, head, prefill's shared experts, the experts the tier does not hold |
| mimo | trunk and vision tower; routed experts on the expert tier | native fp8/bf16, int8, f32 (`MIMO_DENSE_BITS`); experts MXFP4 with e8m0 scales (fmt 7) | router, the experts the tier does not hold |
| deepseek_v41 | the trunk, vision included; routed experts on the expert tier | fp8 in 32x32 ue8m0 tiles, bf16; experts MXFP4 (fp4, a ue8m0 scale per 32) | the DSpark stages (their experts on the tier on a discrete GPU), the experts the tier does not hold |
| deepseek_v4 | resident dense layers, head, router, compressors; routed experts on the expert tier | fp8 in 128x128 blocks, bf16; experts MXFP4 (fp4, a ue8m0 scale per 32) with [an activation of its own](#deepseek-v4s-activation) | the indexer's `weights_proj`, DSpark stages, the `--oracle` path's dense layers, the experts the tier does not hold |
| glm53 (GLM-5.3 Flash) | the resident matrices (an f32 checkpoint's experts among them); the streaming container's routed experts on the expert tier | int8 and int4-g64 (`GLM53_BITS`); experts int4-gs64 | f32 matrices (`GLM53_BITS=32`), the streamed experts the tier does not hold, all of them when `swiglu_limit` is 0 |
| qwenimage | the DiT: every block of a step on the chain (`qwenimage_chain.h`), or its matrices one by one | int8, bf16, f32 (`COLI_IMG_BITS`) | text encoder, VAE |
| laya (Laya) | the whole forward: encoder and decision head layers ([below](#the-decision-engines)); with `COLI_VK_CHAIN=0` their matrices | f16 (fmt 14, the release's F16 values), f32 | the scorer and the act head on the marker rows, the tokenizer |
| gliner_decide (GLiNER2.5-Decide) | the whole encoder ([below](#the-decision-engines)); with `COLI_VK_CHAIN=0` its matrices | f32 | the classifier on the `[L]` rows, the tokenizer |

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

**Run to run.** The dense chain alone gives the same tokens on every run. With the
routed-expert tier on, a run is not bit-reproducible by default: which experts a step
finds on the device depends on when the uploader finished, and the balancer moves rows
between the device and the CPU on measured times. The device's experts compute as
above, so a different split changes the last bits, and over many layers of routing
near-ties that can change a greedy token. `COLI_VK_TIER_SYNC=1
COLI_VK_TIER_BALANCE=0` make the residency and the split follow the routing alone
(each step waits for its uploads, so it is slower): the runs then repeat (#2081).

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

## The decision engines

Laya (`laya.c`), GLiNER2.5-Decide (`gliner_decide.c`) and Clef (`qwen36.c` with
`clef_head.h`) answer a decision with one forward over the request's rows and
generate no text. Their encoder work runs as a batch of matrix products.
Build them with `VK=1` (`make laya gliner_decide qwen36 VK=1`) and set `COLI_VULKAN=1`.

**Laya and GLiNER2.5-Decide** (`decide_vk.h`). The device opens once the weights are
loaded, and every matrix of the encoder and the head goes up then, so the first request
pays nothing for it: Laya's as f16 (fmt 14), since the release stores F16 and every value
is one (the same values in half the bytes), GLiNER2.5-Decide's as f32 (fmt 10). A forward
then runs one of two ways:

- **On the device** (the default on a GPU, discrete or integrated; `COLI_VK_CHAIN=1`
  anywhere). The engine records its whole forward as frames of the dense chain, one
  submission per forward: the embedding rows go up, the encoder (and Laya's two head
  layers) run in the CPU path's order, and only the rows the scorer reads come back
  (Laya's markers and each sequence's first row, GLiNER2.5-Decide's `[L]` rows). The
  scorer, the act head and the classifier, a few rows each, stay on the CPU.
- **Matrix by matrix** (`COLI_VK_CHAIN=0`): every dense matrix through `coli_vk_matmul`
  (the tiled GEMM from the backend's threshold), the norms, the attention and the rest
  on the CPU. `COLI_VK_DENSE=0` as well keeps everything on the CPU.

What runs between the matrices is `chain_enc.comp` (`vkc_enc_*` in `vk_chain.h`):

| Op | What it does |
|---|---|
| `vkc_enc_norm` | LayerNorm with weight and bias, the residual add fused (`x += h; h = LN(x)` for ModernBERT's pre-norm layers, `x = LN(h + x)` for DeBERTa's post-norm ones) |
| `vkc_enc_bias` | a matrix's bias, then GELU (erf or tanh), ReLU or nothing, in place |
| `vkc_enc_geglu` | ModernBERT's GeGLU: the first half of `Wi`'s row through GELU, times the second |
| `vkc_enc_rope` | rotate-half RoPE on the query and key heads, positions from each sequence's start, cos and sin from the host's own table (global and local theta) |
| `vkc_enc_addrow` | a table row per row: Laya's type embedding by the row's question type |
| `vkc_enc_attn` | bidirectional attention over a range of rows per row: its sequence (a request's questions are separate sequences in one batch), or Laya's +-64 window inside it; DeBERTa's two relative terms added by bucket, an online softmax over tiles of 32 rows, 16 query rows a workgroup |
| `vkc_enc_rel` | DeBERTa's relative terms as its CPU attention computes them: each row's query against the key projection of the relative embeddings (content to position) and its key against their query projection (position to content), per head, for the buckets the forward can reach |

The arithmetic is the CPU path's in f32: the two differ in the order of the sums, GELU's
`erf` comes from a fit of `erfc` with a relative error under 1.2e-7, and LayerNorm sums in
float where the engines sum in double. A device lost in the middle of a forward leaves
that forward to the CPU, which recomputes it from the ids: a decision has no state to
rebuild. Each request ends with `[VK] <engine>: N matmuls on the GPU` and
`[VK] <engine> chain: F forwards on the device ...`.

**Clef** runs through qwen36's own Vulkan paths, which DECIDE now uses: the backbone's
prefill over the record matrix by matrix (the default on an integrated GPU, where a
model without routed experts keeps its dense matrices on the device and the chain off)
or as the dense chain (`COLI_VK_CHAIN=1`; the default on a discrete GPU), whose read-out
brings back every row's hidden state for the head. The joint head stays on the CPU. A `[clef]` line reports backbone and head
time separately so an end-to-end measurement can identify the limiting stage. Two things were missing for it:

- **f16 rows.** `COLI_DENSE_BITS=16` keeps the trunk in the container's f16, and
  qwen36's device path knew the int4, int8 and f32 copies only. The backend now takes f16
  rows as fmt 14 (no scales, low half the even column, like bf16's 11) in the GEMV, the
  fp32 tiled GEMM and the chain's decode GEMV; the cooperative-matrix GEMM keeps its
  formats.
- **The rows read in place** (`COLI_VK_IMPORT`). A resident copy on a device that shares
  the CPU's RAM is the matrix held twice: Clef's trunk is 27 GB in int8 and 54 GB in
  f16, and the f16 one does not fit twice in 61 GB. `coli_vk_tensor_import` hands the
  device the host's own pages (`VK_EXT_external_memory_host`): qwen36 allocates its int8
  and f16 rows page-aligned when `COLI_VULKAN` is set, and the device reads them where
  they are; their scales, a float a row, are copied. On by default for a model without
  routed experts on an integrated GPU or Lavapipe; every other model copies as before.
  The harness checks imported rows bit for bit against a copy (int8, f32, f16, the GEMV
  and the GEMM).

The in-place import falls back to an ordinary resident copy if the extension, alignment
or buffer requirements cannot be met. It is disabled with staged uploads. The host
allocation remains alive while the device reads it. This saves the duplicate weight
storage on a shared-memory GPU; it is not a claim of faster matrix multiplication.
On an integrated GPU this also applies with `COLI_VK_DENSE_HOST=0`: imported rows
already have one physical copy. Keeping them in place avoids moving the entire
trunk into a potentially smaller device-local heap and exhausting its scratch budget.
`COLI_VK_IMPORT=0` forces independent device copies and permits their host pages to
be released. Only pages actually released contribute to the host-memory drop counter.

The dense Qwen/Clef backbone submits each layer separately, retaining the residual
stream on the device. Without routed experts there is no intermediate host readback
to divide the work; recording all layers of a large prompt in one submission can
exceed the driver's watchdog. The regression compares CPU answers across repeated
requests and checks that the submission count grows with the number of layers.

On the real Clef 27B int8 checkpoint, Radeon 780M/RADV completed six requests
(254–755 tokens) with 401 independent device matrices, 23.86 GiB of host copies
released and no CPU reloads. The shared-page configuration completed the same six
requests with 24,412.5 MiB read in place; its maximum absolute logit difference from
the independent copies was 3.5e-7. Against two CPU references with the same float
activation arithmetic (`COLI_DENSE_IDOT=0`), the maximum difference was 3.42e-6.
The shared-page budget selected 3,328 prompt rows automatically. These are correctness
and memory checks on one integrated GPU; they do not establish throughput on a
discrete GPU or under Windows.

### Validation and performance

`bash c/tests/vulkan_engines.sh decide` checks all three engines against their CPU
outputs: identical decisions and refusals, probabilities within 1e-5 and logits within
1e-4 relative to the largest magnitude (at least 1). It also verifies that work was
actually dispatched. The matrix and chain paths, f16 and int8 Clef rows, import and
copy, staged uploads, CPU fallback after a device loss, serve sessions and SDK requests
are covered. `decide-sanitize` runs the comparison and serve tests under ASan and UBSan.
With both `COLI_VK_CHAIN=0` and `COLI_VK_DENSE=0`, answers must equal the CPU's byte for
byte apart from elapsed times. These gates have passed on Lavapipe; they establish
correctness, not GPU speed.

For latency, measure complete requests after a warm-up in the same running engine,
using identical records, weight precision and CPU thread count. Compare the CPU with
`COLI_VULKAN=1 COLI_VK_CHAIN=0` and `COLI_VULKAN=1 COLI_VK_CHAIN=1`, and report the median
`engine_ms`, input lengths, GPU, available RAM and competing workload. Clef's
backbone/head timing identifies how much of the request Vulkan can accelerate. No
end-to-end speedup for all three real checkpoints is established by the fixture gates,
and no discrete GPU measurement is included here.

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
- **Also when the device computes every expert.** With the host pool (system RAM the
  device reads over PCIe, below) the device computes every routed expert and the CPU
  none, so no expert passes by the CPU any more and the set on the device would stay
  the warm start's. Where RAM holds every expert, the host pool now takes the device's
  experts too (the warm start puts each in both places). After a decode step or a
  verify (up to 8 rows: a prompt chunk routes nearly every expert once), a routed expert
  the device read from the pool moves up into a free place on the device or the coldest
  resident's, when it beats that one by two routings, up to `COLI_VK_TIER_PROMOTE` (32) a
  forward. The resident goes back to its pool copy at once and nothing is copied back;
  the newcomer is copied up by the device's copy engine from the pool
  (`coli_vk_tensor_copy`: no byte through the CPU) while the engine goes on, and takes
  its place in the tables at the next frame's start (`vkt_settle`). The tier logs which
  entries each change touched (`vkt_dev_changes`), and the dense chain rewrites those
  entries only: read whole at every change, the tables cost Qwen3.8's decode a fifth
  (the code prompt below: 37.5-40.3 tok/s).
  Measured on an RTX PRO 4000 with Qwen3.8-Flash-Next IQ2_XS (34.6 GiB of experts in
  the pool, 10443 places on the card), each server on a copy of one history, three
  runs:

  | prompt | decode, `COLI_VK_TIER_PROMOTE=0` | decode, the default | the decode's experts in VRAM |
  |---|---|---|---|
  | code (not in the history) | 30.3-31.6 tok/s | 46.4-48.0 tok/s | 66.7% -> 92.7% |
  | Italian prose (not in the history) | 54.4-56.1 tok/s | 61.7-64.0 tok/s | 79.9% -> 93.2% |
  | the meeting report that wrote the history | 49.6-50.8 tok/s | 48.7-51.4 tok/s | 93.9% -> 93.3% |

  Without the two routings' margin the last one lost 4% to swaps that found nothing
  more on the card; at 16 a forward the code prompt reached 89% in VRAM, not 93%. The
  greedy text is the same with and without (Qwen3.8 IQ2_XS, Qwen3.6 Q4_K_M with
  `COLI_VK_TIER_GB=8`, where the second took 8.6 s against 11.1 s for 400 tokens).
  Without the RAM for every expert, or beside a second device, the tier keeps the warm
  start's set as before.
- **The host pool and the engine's RAM cache share the RAM.** When the pool cannot hold
  every expert the device leaves (a model bigger than RAM and VRAM together), the CPU
  computes the others from the engine's own RAM cache, which the engine sized from the
  same free RAM before the tier started. The pool now takes that cache's least
  (`VktConfig.ram_floor`, top-k slots a layer) out of its room, and once sized hands the
  engine its bytes (`VktConfig.ram_given`): every engine but DeepSeek V4 lowers its cache
  by as much, to no less than top-k slots a layer; one without the callback keeps its
  `ram_reserve` and the pool takes what is left. On GLM-5.3 Flash UD-IQ2_XXS (95 GB) with
  62 GB of RAM the two had asked for 52.8 and 48 GB, and a 711-token prompt was killed by
  the OOM killer; the pool now takes 45 GiB and the cache 10 slots a layer instead of 127,
  and the same prompt runs at 11.2 tok/s with the process at 58 GB. The pool beats the
  cache there: with `COLI_VK_TIER_HOST=0` (every free byte the cache's) the decode ran at
  0.7-1.4 tok/s against 3.2-3.3. DeepSeek V4's expert store keeps its `ram_reserve` on
  purpose: its CPU computes a step's cold experts fast enough that the split helps, and
  on DeepSeek V4 Flash 0731 UD-IQ2_XXS (85 GB) a pool of 41 GiB with the store cut to 22
  slots a layer decoded at 5.5-5.6 tok/s against 5.9-6.2 with the store whole and a pool
  of 5 GiB. `COLI_VK_TIER_HOST_GB` sizes the pool and leaves the
  cache as it was.
- **A GGUF's few bigger layers keep their own rooms.** Published GGUFs type a few expert
  layers bigger than the rest, and the tier takes each layer's experts in its own types
  (`VktConfig.layer_gate_up`, `layer_down`). Every expert used to take the room of the
  biggest form, so that a freed expert's place held any other: on the files below, 9.4
  MiB for experts of 7.3 (IQ2_XXS gate and up, IQ3_XXS down in all but two or three
  layers). Now the layers of the form most layers have keep their experts at their own
  size, and only the others take the room of the biggest among them. A newcomer
  displaces a resident of its own class first, and one of the other class only when
  that one's room holds it: a smaller form's expert in a bigger form's room, never the
  other way. The device and the host pool hold 25-31% more experts. Measured on an
  RTX PRO 4000 (24 GB) with 62 GB of RAM, each server on a copy of one history, 128
  tokens, a warm-up and a run:

  | model | experts in VRAM, in the pool | routed experts on the device | disk loads | decode | prompt |
  |---|---|---|---|---|---|
  | GLM-5.3 Flash UD-IQ2_XXS (95 GB), before | 1707, 4808 | 89.4% | 13198 | 2.50, 3.14 tok/s | 8.3, 9.2 tok/s |
  | the same, now | 2167, 6064 | 95.0% | 6502 | 2.92, 3.40 tok/s | 9.3, 9.3 tok/s |
  | DeepSeek V4 Flash 0731 UD-IQ2_XXS (85 GB), before | 1721, 465 | 69.4% | 3367 | 5.25, 5.35 tok/s | 9.3, 17.3 tok/s |
  | the same, now | 2250, 580 | 74.1% | 1469 | 6.18, 6.25 tok/s | 19.7, 20.4 tok/s |
  | Inkling-Small UD-IQ2_XXS (77 GB), before | 2375, 5431 | 99.8% | 186 | 7.87, 8.11 tok/s | 12.2, 22.8 tok/s |
  | the same, now | 3098, 7003 | 100.0% | 14 | 11.53, 10.92 tok/s | 15.1, 33.1 tok/s |

  The startup line says `N layers in M forms other than most layers', their experts in
  the room of the biggest of those, the rest at their own size`.
- **A fresh install starts warm.** A model downloaded a minute ago has no history,
  so its first runs would fill the tier one routing at a time. `coli setup` copies a
  starting history into the model folder when it has none, for the catalog models
  that ship one (`c/profiles/<catalog id>.coli_usage`: Qwen3.6-35B-A3B and Qwen3.8
  Flash Next). It comes from a calibration session (16 prompts: chat in five
  languages, code, reasoning, JSON), and the engine adds every run's routing to it,
  so your own use takes over; an existing history is never replaced. Measured on a
  Radeon 780M with Qwen3.6-35B-A3B, two prompts outside the calibration set, 128
  tokens, two rounds each: the first run went from 5.9-6.2 tok/s without it to
  10.9-11.1 with it, the device serving 92-94% of the routed experts instead of
  15-16%, for a warm start of about 3 s.
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
  dispatch either (up to 15 rows an expert takes the per-row GEMV route), so a
  speculative verify's rows (two to six, [speculative.md](speculative.md)) get a decode
  step's bits. From 16 rows (prefill) an expert takes
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
| qwen38 (Qwen3.8 Flash Next) | the int4-g64 sidecar `I4U_PLANAR64` (fmt 4); the release's FP8 in 128x128 blocks `FP8_BLOCK` (fmt 12); `BF16` (fmt 11); `F32` (fmt 10) | SwiGLU | `COLI_USAGE` (default `<snap>/.coli_usage`), always kept | the MTP head's layer is an extra layer of its own form (FP8 beside an int4 sidecar), [below](#the-mtp-heads-layer-on-the-tier-coli_vk_tier_mtp) |
| inkling | int4 container `I4U_PAIRS_ROW` (fmt 2); int8 container `I8_ROW` (fmt 1); runtime int8 rows, `I8_AS_I4_ROW` at 2 to 4 bits (fmt 2) and `I8_ROW` above; `F32` at `bits=0` (fmt 10). Gate and up come from the fused `gate_up` tensor, up I rows in | SwiGLU | `<snap>/.coli_usage` or `PIN=<path>`, in the generate and serve modes; the ref.json oracle reads none | [Inkling and OLMoE](#inkling-and-olmoe) |
| olmoe | int8 rows `I8_ROW` (fmt 1), gate, up and down as the merged container holds them | SwiGLU | `COLI_USAGE` only | [Inkling and OLMoE](#inkling-and-olmoe) |
| kimi_k3 (Kimi K3) | the checkpoint's MXFP4 with ue8m0 scales `MXFP4_E8M0` 32 (fmt 7): gate `w1`, up `w3`, down `w2`, in the latent space | SiTU-GLU (`VKT_ACT_SITU`) | `COLI_USAGE` (default `<snap>/.coli_usage`) | [Kimi K3 and MiMo](#kimi-k3-and-mimo) |
| mimo (MiMo-V2.6 Flash and Pro) | the release's MXFP4 `MXFP4_E8M0` 32 (fmt 7); a GGUF's other block types as the file has them (100 + the ggml type), each layer its own | SwiGLU | `COLI_USAGE` (default `<snap>/.coli_usage`) | [Kimi K3 and MiMo](#kimi-k3-and-mimo) |
| deepseek_v41 (DeepSeek V4.1 Flash) | MXFP4 `MXFP4_E8M0` 32 (fmt 7), as the checkpoint stores it | SwiGLU with `swiglu_limit` | `COLI_USAGE` (default `<snap>/.coli_usage`), kept only while the tier is on | the backbone's layers; the DSpark stages, with caches of their own, are extra layers ([below](#the-mtp-heads-layer-on-the-tier-coli_vk_tier_mtp)) |
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
| `COLI_VK_TIER_RESERVE_GB` | `1` | Device memory left to everything else (scratch, KV mirrors, the driver) on top of the dense weights the engine still has to place and, on qwen38, the chain's KV mirror and a prompt chunk. |
| `COLI_VK_TIER_RATE` | `16` | Promotions per token at most (a prompt's forward gets this many per prompt token): each copies one expert on the engine thread. |
| `COLI_VK_TIER_HOST` | on (a discrete GPU) | `0`: no host pool. The pool shares system RAM with the engine's RAM expert cache (below). |
| `COLI_VK_TIER_HOST_GB` | from free RAM | The host pool's size in GiB; the engine's cache keeps its size then. |
| `COLI_VK_TIER_PROMOTE` | `32` | With a host pool that holds every expert: the most routed experts that move up from it to the device a forward (below). `0`: the warm start's set. |
| `COLI_VK_TIER_BALANCE` | on | `0`: the device takes every resident expert of a step even when it is the slower side (see below). |
| `COLI_VK_TIER_WARM` | on | `0`: no warm start; the tier fills as experts pass by. |
| `COLI_VK_TIER_FILL` | on | The warm start fills the device: the history's experts first (hottest first), then the room left with the experts no history named, one of each layer in turn. A first run used to start with an empty device and fill it at the promotion rate (Qwen3.6 on a 3090 Ti: 16.5% of the routed experts on the device and the CPU's speed, then 97% and three times it from the second run, #2081). A model whose experts all fit now goes up whole at start. `0`: the history's experts only. |
| `COLI_VK_TIER_SYNC` | `0` | `1`: each layer step first waits for the uploads staged so far: residency then follows the routing alone (with `COLI_VK_TIER_BALANCE=0`, the run is reproducible). A promotion that displaces a resident while a batch is in flight waits for the join to free it. For tests and debugging. |
| `COLI_VK_TIER_GEMM_ROWS` | `16` | Rows from which an expert of a step takes the tiled GEMM instead of the per-row GEMV; `0` never. |
| `COLI_VK_XB_GROUPED` | on where supported | `0`: a step's experts keep their own dispatches where the grouped GEMM would take them (cooperative matrices at subgroup size 64, `bufferDeviceAddress`, widths multiples of 128, int8/int4, SwiGLU). |
| `COLI_VK_XB_GROUPED_ROWS` | `64` | Assignments in a batch from which the grouped GEMM takes it. |
| `COLI_VK_XB_STEP` | on where supported | `0`: a prompt step packs its rows per expert and sums on the host, as before the whole-step buffers. On a Radeon 8060S (Qwen3.6, 1011-token prompt) the grouped GEMM and the step buffers took the experts' device time from 762 to 344 ms and the first token from 2.08 to 1.49 s, perplexity unchanged (7.65). |
| `COLI_VK_XB_GEMV` | on where supported | `0`: a decode step's experts keep a GEMV dispatch each instead of the grouped GEMV (`qmatmul_grp_gemv.comp`). On a Radeon 8060S (Qwen3.6, a 256-token answer, tier 24 GB) the grouped GEMV took decode from 37.2 to 42.5 tok/s with the int8 trunk and from 45.0 to 53.5 with the int4 trunk. It also keeps the experts below 16 rows the grouped routes leave (any device without them: NVIDIA, Intel, Lavapipe; MXFP4 and SiTU everywhere) on the per-expert GEMVs instead of the dense chain's decode kernel; see "The tier's small experts on the chain's kernel". |
| `COLI_VK_TIER_QUEUE` | a second queue | `0`: the tier shares the main queue (its batches and the dense matmuls then serialize). |
| `COLI_VK_TIER_STREAM` | on | A prompt step's cold experts streamed to the device ([below](#big-prompt-chunks-and-expert-streaming)); `0` off. `_SLOTS`, `_ROWS`, `_HALF`, `_PAR` there. |
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

### The tier's small experts on the chain's kernel

A decode step, an MTP verify or a prompt chunk's rarely routed experts give the batch
experts with a row or a few. Where no grouped route takes the batch (the grouped GEMM
and GEMV need cooperative matrices at subgroups of 64: NVIDIA, Intel and Lavapipe have
no grouped route; on a Radeon the grouped GEMV takes int8, int4 and MXFP4 with SwiGLU or
SiTU, 17-26% faster there than this kernel on a 780M, and MXFP4 now 2530 -> 2085 us a step
at MiMo's shape), each of those experts below the per-expert GEMM's 16 rows runs the dense
chain's decode kernel, `chain_moe_gv.comp`, with every row of the batch a pair of its
own: two dispatches for the step instead of two per expert, each a GEMV of 8 outputs a
workgroup. Formats int8, int4 (a scale a row or a group) and MXFP4; SwiGLU, clamped or
not, and SiTU. DeepSeek V4's activation and the block formats keep their routes.

The chain and the tier take the lanes a row from the same rule (`coli_vk_gv_lanes`) and
pin the same subgroup size, so a row has the same bits on either: a prompt chunk's
experts through the tier and the next tokens' through the chain (OLMoE and Qwen, the
frame's own experts for short steps) give a resumed prefix the cold forward's numbers.
tests/test_vk_chain compares them pair by pair.

A decode step's experts on an RTX PRO 4000 (`COLI_VK_CHAIN_BENCH=xb` in
tests/test_vk_chain, device time a layer, 48 layers of distinct experts):

| shape | per-expert GEMVs | chain kernel |
|---|---|---|
| GLM-5.3: 8 int4-g64, 6144 x 2048 | 1595 us | 442 us |
| Kimi K3: 8 MXFP4, 7168 x 2048, SiTU | 1677 us | 572 us |
| DeepSeek V4.1: 6 MXFP4, 4096 x 2048 | 1604 us | 256 us |
| MiMo: 8 MXFP4, 4096 x 2048 | 1411 us | 338 us |
| 8 int4 with a scale a row (Inkling), 7168 x 2048 | 1332 us | 463 us |
| 8 int8, 6144 x 2048 | 1333 us | 762 us |

A GGUF's experts (the block formats) take the same way for a batch whose experts all have 4
rows or fewer (a decode step, an MTP verify): `chain_moe.comp`'s block path (bdstep.glsl, its
float form, a pipeline per pair of types), every row a pair, down's rows copied to the
batch's. The grouped GEMM of the block formats works in 32-row tiles and the per-expert
GEMVs a row a subgroup:

| shape | grouped GEMM | chain kernel |
|---|---|---|
| GLM: 8 Q4_K, 6144 x 2048 | 2694 us | 512 us |
| GLM: 8 Q6_K, 6144 x 2048 | 3277 us | 830 us |
| Kimi K3: 8 IQ2_S/Q2_0, 7168 x 2048 | 3136 us | 434 us |
| DeepSeek: 6 Q4_K, 4096 x 2048 | 1393 us | 303 us |

A prompt chunk's experts keep the grouped GEMM. `COLI_VK_XB_GEMV=0` keeps the per-expert GEMVs.

### RAM and VRAM without the same experts (`COLI_VK_TIER_EXCLUSIVE`)

With RAM short of the working set, the engine's RAM expert cache and the tier would
otherwise hold many of the same experts: the CPU reads an expert, computes it, the tier
promotes it, and its RAM copy stays until the LRU reaches it. Exclusive caching (on unless
`COLI_VK_TIER_EXCLUSIVE=0`) gives that copy up first:

- **When the RAM cache must evict**, it first takes, among the slots it may evict, the
  least recently used one whose expert the tier holds on a device (`vkt_ram_first`), and
  only then its usual LRU choice. With RAM to spare nothing is evicted and nothing
  changes; with little RAM the two caches hold different experts, so together they hold
  more of them. An expert routed in its layer's current step is never offered, even on a
  device: the CPU may be computing it from that slot (the balance handed it back, or the
  batch had no room for its rows).
- **The prefetchers that read experts into RAM** (the PILOT workers of qwen36, olmoe and
  colibri) skip what the device holds. qwen38's only advises the page cache, and only of
  the experts the CPU will compute.
- Every MoE engine does it in its own cache: qwen36, qwen38, olmoe, inkling, mimo,
  kimi_k3, deepseek_v41, deepseek_v4's expert store, colibri and glm53. Pinned slots and
  slots being read or computed are never taken, as before.
- The run's line says how many RAM copies were given up for it:
  `| exclusive: N RAM copies of device experts given up first`.

The cost: an expert the device holds is no longer in RAM to hand back to the CPU when the
device is the slower side of a step (`COLI_VK_TIER_BALANCE`), so on an integrated GPU at its
floor clock the balance has fewer experts to move. `COLI_VK_TIER_EXCLUSIVE=0` keeps the
copies.

The `dev2` families' `excl` and `noexcl` cases run every engine with a RAM cache of a slot
or two against its CPU run. On DeepSeek V4's 16-expert fixture with 6 slots a layer, the
run with it gave up 21 RAM copies and read 46 experts from disk, against 51 without it.

### The MTP head's layer on the tier (`COLI_VK_TIER_MTP`)

Qwen3.8's MTP head drafts with a MoE layer of its own (index `layers`), whose experts the
snapshot keeps in their own form: FP8 in 128x128 blocks beside the int4-g64 sidecar, which
covers the model's layers only. With `COLI_VK_TIER_MTP=1`, the default on a discrete GPU,
the tier takes that layer as an extra layer (`VktConfig.extra_layers`, with
`extra_gate_up` and `extra_down`): its experts are promoted
as the drafts pass by, served by the same batches as any layer's, and given up to the
exclusive RAM cache the same way. What differs:

- **A pool of its own.** An FP8 expert is about twice an int4 one, and in one pool the
  holes an evicted int4 expert leaves are too small for it. The extra layer's experts sit
  in a pool of their own on the primary device, where a newcomer displaces only another
  extra expert, so each pool holds one size. The pool gets every extra expert when the
  budget holds them beside every main one, else the extra layers' share of the budget
  (their layers over all the layers, at least one expert), and its bytes leave the main
  experts' count. They never go to a second device.
- **No history, no streaming.** The history and the warm start cover the model's layers;
  the drafts fill the extra layer, with `COLI_VK_TIER_RATE` promotions per token of its
  own. It comes last in a forward, after the model's layers have spent theirs: on the
  release they always do, and with one shared rate the head's layer got no expert at
  all. Big prompt steps never reach it (the head drafts a few rows at a time), so it
  takes no streaming slots.
- **The lines.** The startup line adds `; the extra layers' experts (1, fmt 12): X of S
  in a pool of P`, the run's line `| extra layers (1): N of M routed experts on the
  device, resident R (budget X)`.

**Measured** on the Radeon 780M of [speculative.md](speculative.md#measured) with the same
command as its Vulkan runs (the tier alone, int4-g64 sidecar, three drafts, cap 170, the
code-edit prompt, 128 tokens, the same starting history, model pages evicted before every
run), three runs each, alternating:

| MTP head's experts | tok/s | head's routed experts on the device | model's layers on the device |
|---|---|---|---|
| on the CPU (`COLI_VK_TIER_MTP=0`) | 3.48, 3.49, 3.46 | | 90382-90410 of 176640 |
| on the tier (`COLI_VK_TIER_MTP=1`) | 3.44, 3.44, 3.46 | 1095-1160 of 4280 (31 resident, 4.8 MiB each) | 89773-90343 of 176640 |

All six answers were byte-identical, and so was the head's acceptance (95 of 95 drafts).
On this integrated GPU the head's experts on the device bought nothing: their pool (149
MiB) came out of the model's layers' budget, whose share on the device fell a little. So
the default puts them on the tier on a discrete GPU only;
`COLI_VK_TIER_MTP=1` or `0` decides either way. A discrete GPU was not measured here.

When the head's experts have no device form (`[VK] tier qwen38: the extra layers' expert
format ... has no device form`), mix formats or find no room, they stay on the CPU. The
tokens are the CPU run's either way: the MTP layer's experts join the row in rank order
like the others'.

**The other engines with a drafting head.**

| Engine | Extra layers | Their experts |
|---|---|---|
| colibri (GLM-5.2) | the MTP head's layer, index `n_layers` | as the container keeps them (int8 beside int4 in a converted GLM-5.2), checked like the model's |
| deepseek_v41 | the DSpark stages, `n_layers + stage` | the backbone's MXFP4; each stage's cache is the RAM side the balance and the exclusive RAM cache read |
| deepseek_v4 | none | its full DSpark drafter (three stages, the DSpark supplement) keeps its experts on the CPU: the tiny fixture has one MTP layer, so no test reaches that path |
| glm53 | none | its MTP layer has no routed experts |

The same `COLI_VK_TIER_MTP` decides, with the same default.

**Tests.** `tests/test_vk_tier`'s `extra` case gates the pools (an f32 extra layer, eight
times an int4 expert: each pool evicts its own kind, no upload refused, an extra expert
gets in beside a full main pool on its own promotions). The `qwen` family runs the head's
layer on the device on both MTP fixtures and its default on Lavapipe; `glm` colibri's
head with f32 and 4-bit experts at two drafts, and its default; `deepseek` V4.1's two
DSpark stages at three forced acceptances, the default at five. Each must give the CPU's
tokens and serve experts of the extra layers from the device; the sanitizer families run
one of each.

### A second device (`COLI_VK_DEV2`)

The dense chain can put layers there too: [Layers on two devices](#layers-on-two-devices).

A machine with two GPUs (a V100 beside a GTX 1070, an RX 9070 beside an RX 580) can
give the tier the memory of both. `COLI_VK_DEV2=auto` takes the best GPU that is not the
primary device (a discrete card before an integrated one); `COLI_VK_DEV2=<index>` takes
that entry of the Vulkan enumeration (`COLI_VK_DEV` picks the primary one the same way).
Every engine on the tier gets it, with nothing else to set:

- **What goes where.** The primary device keeps its budget as before. The second
  device's is its free memory less `COLI_VK_RESERVE2_GB` (0.5 GiB: its batch's scratch
  and the driver), at most `COLI_VK_EXPERTS2` experts when that is set, and never more
  experts than the primary device leaves: a model whose experts all fit on the primary
  device leaves the second one empty, and a line says so. An integrated GPU as the
  second device takes a quarter of what the RAM leaves, as the primary one does.
- **The warm start** fills the primary device with the hottest experts of the history
  and the second one with the next ones. **While you chat**, a promoted expert goes
  where there is room (the primary device first), or takes the place of the coldest
  resident on either device, by the same LFRU rule.
- **Every step** sends each device the batch of its own resident experts. Both are in
  flight at once while the CPU computes the rest, and the rows come back to be added
  in routing order, as with one device. The second device has its own pipelines,
  scratch and queue. It takes the GEMV and fp32 GEMM routes, not the
  cooperative-matrix one, because its device is created without those features.
- **Big prompt steps.** Their sub-batches and the streaming of cold experts stay on
  the primary device. The second device's residents of such a step run there as one
  batch beside them, up to 128 MiB of rows; past that, the streaming rule decides for
  them.
- **A failure** of the second device (a batch that fails, the device lost) hands that
  step's rows back to the CPU and frees its experts, and the tier goes on with the
  primary device:
  `[VK] tier <engine>: a batch on the second device failed, its experts go back to the
  CPU and the tier goes on with <device>`.
- **The lines.** At startup:
  `[VK] tier <engine>: second device <name>, budget B = N experts of S (device memory),
  the experts after the primary device's`. The run's line ends with
  `| second device <name>: resident R (budget N, used of B), X rows in Y batches
  (Z beside prompt steps), uploads U, T ms`. Its other fields stay the primary
  device's.
- **colibri's own registry.** With the tier on, colibri's fixed second-device registry
  stays off: the tier holds the experts there, in every format. With `COLI_VK_TIER=0`
  the registry works as before (the hottest experts of the history, int4 and int3,
  fixed at startup).

The families `dev2`, `dev2-deepseek-kimi-mimo` and `dev2-sanitize`
(`c/tests/vulkan_dev2.sh`) run every MoE engine on Lavapipe opened twice. `COLI_VK_DEV2=0`
is the test mode: a second logical device on the same physical one. The primary device
is held to a few experts, and four cases run against the CPU's tokens:
- decode;
- a warm start before big prompt steps;
- both devices full and evicting;
- the second device failing at its second batch (`COLI_VK_DEV2_FAULT=2`).

Not measured: speed on two real GPUs. Lavapipe shows that the rows come back right and
land where they should. How much a second card adds depends on its memory and its bus.

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
on the CPU`). The MTP head's layer (its experts int8 in a converted container) is an extra
layer ([above](#the-mtp-heads-layer-on-the-tier-coli_vk_tier_mtp)). The tier serves every row count:
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
  mirror (`CTX` rows a layer) is reserved too, at most half of what the device has free:
  the gateway's automatic context (131072 tokens on a large RAM) asked GLM-5.3 for 23.6 GB
  of mirror on a 24 GB card and left the tier no expert (#2314). The dense set (`COLI_VK_DENSE=1`) is
  uploaded before the tier sizes itself, so the budget is what remains; the trunk's
  default here stays off. On an integrated GPU the tier takes its default share of RAM
  after the expert cache's (the startup reservation of the fixed set is gone).
- *`COLI_VK_DEV2`*: with the tier on, the tier holds experts on the second device
  ([A second device](#a-second-device-coli_vk_dev2)). With `COLI_VK_TIER=0`, colibri's
  own registry there, fixed at startup, takes the hottest experts of the history, up to
  `COLI_VK_EXPERTS2`; a step sends them there as one group on a worker thread while
  the CPU runs.

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

The decode GEMVs' int8 path (`chain_gemv.comp`) and the routed experts' rows
(`chain_moe_gv.comp`) issue four 16-byte steps' loads before the first is summed: a loop
that loaded one step and summed it waited for every load in turn. The sums go into the same
accumulators in the same order, so a row's bits are unchanged (tests/test_vk_chain). An RTX
3070: int8 8192 x 2048 78 -> 66 us a matrix in `COLI_VK_CHAIN_BENCH=1`, Qwen3.6-35B int4 decode
18.5 -> 17.5 ms a token; the experts' bench (every expert in VRAM) unchanged.

With the dense matrices on the device one `coli_vk_matmul` at a time, each matrix is
a submit and a host round trip: about 726 of them per Qwen3.8 decode token. On an
integrated GPU that made the dense part slower on the device than on the CPU (see
[the other engines](#the-other-engines)). The chain records a whole layer into one
command buffer instead, and keeps the residual stream on the device from one layer
to the next. qwen36 (Qwen3.6, Qwen3-Coder, Qwen3.8-27B) and qwen38 (Qwen3.8 Flash
Next) run it: by default on a discrete GPU, and for qwen36 on an integrated one with
the expert tier, and for qwen38 on one with 32 or more compute units too (see [the
default](#the-chain-on-a-radeon-780m)); `COLI_VK_CHAIN=1`
anywhere. `COLI_VK_DENSE=0` keeps it off too, unless `COLI_VK_CHAIN` says otherwise: the
chain's copy of the trunk would take the expert tier's budget (on an 8 GB RTX 4070
Laptop with qwen38, disk-bound, the tier got 1.87 GiB instead of 6.00 and decode went
from 0.84 to 0.64 tok/s, #1900). olmoe and inkling run it as well ([OLMoE and Inkling](#olmoe-and-inkling)).
colibri (GLM-5.2), glm53 (GLM-5.3 Flash) and kimi_k3 run it too, with the MLA,
KDA and hyper-connection ops ([below](#glm-52-and-glm-53-flash-on-the-chain)), and
deepseek_v41 and deepseek_v4 with DeepSeek's own ([below](#deepseek-v41-flash-and-deepseek-v4-on-the-chain)).

**What runs where, per layer** (S rows: one at decode, a prompt chunk at prefill):

| | qwen36 | qwen38 adds |
|---|---|---|
| device, frame A1 | the previous layer's MoE output joining the residual (in the CPU's order: routed experts in rank order, then the gated shared expert, then the add); the input RMSNorm; the gated attention (q/k/v, per-head q/k norm, RoPE, the new K/V rows into the device cache, attention with the output gate, o_proj) or the Gated DeltaNet (qkv/z/b/a, the causal convolution with its ring, the recurrence with its state, the gated norm, out_proj); the residual add; the post-attention norm; the router logits | the four hyper-connection streams: each block's gated-residual read (per-stream norm, the low-rank pair, the stream mix, the inject weights) and its write-back; the PLE layer's projections, gate and dilated convolution (the n-gram table rows come up from the host); QSA with its indexer: each block's pooled key computed once on the device when a step completes it, and the top-k selection per query row |
| host | the router's softmax and top-k; the routed experts (the expert tier's device batch and the CPU's share, joined in rank order); the new K/V rows copied into the host's cache | the new index-key rows too |
| device, frame A2 (not waited for) | the shared expert and its gate, while the host computes the routed experts | |
| last frame | the final norm and lm_head on the last row, or on every row of a speculative verify | the final mixer; lm_head on the last row, or on every row of a verify |

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
  the watermark to its `pos_base`; a cache that grows is mirrored again. Past the
  device's budget the mirror keeps part of the cache and the CPU attends over the
  rest ([below](#a-kv-cache-past-the-devices-budget)). Qwen3.8's
  pooled block keys have a watermark of their own, lowered to the first block a step
  rewrites (a rejected draft's block is recomputed when a step completes it again).
- The DeltaNet recurrent state and conv rings, and Qwen3.8's PLE ring: on the device
  while the chain runs (60 MB on Qwen3.6-35B, too much to copy per token). The host's
  copy is brought back before anything reads it there (a pinned snapshot, the prompt
  cache, a CPU step) and pushed up after anything writes it there (a reset: a fill
  with zeros on the device; a restored snapshot: an upload).
- A speculative verify (S = 2 to 6: the token and its MTP or prompt-lookup drafts,
  [speculative.md](speculative.md)) copies the DeltaNet states, the conv rings and the
  PLE ring on the device after each of its rows but the last, one slot per row. One
  copy the shaders write as they pass its row; more split the convolution and the
  recurrence into one dispatch per copy, each ending on its row (the same bits). A draft
  rejected after row r swaps slot r's device buffers in, as the CPU swaps its own, and
  the KV and pooled-key watermarks come down to the standing rows. Its matrices take
  the decode GEMV, so its rows get a decode step's bits; up to four of them share a
  workgroup and every weight load (`chain_gemv.comp`'s NR 4 pipeline, each row's sums
  the decode's operations in their order; `chain_gemv2.comp` does the same for int8
  where it runs). On an RTX PRO 4000 a verify of 2 / 3 / 4 rows of Qwen3.6's int8
  lm_head took 2.50 / 3.74 / 4.97 ms when each row read the weights, 1.32 / 1.42 / 1.65
  shared, and an 8192 x 2048 matrix 63 / 81 / 107 us against 51 / 55 / 62. A code edit
  of 400 tokens (prompt lookup, Qwen3.6 int4-g64 on the chain) went from 3.71-3.77 to
  3.16-3.20 s, the same text; chat decode unchanged (84.4-85.4 against 84.3-85.9 tok/s).
  A matrix of fewer than 64 workgroups keeps a workgroup a row (sharing them left the
  device short: 512 x 2048, 9 -> 11 to 15 us). Slots past the first are
  allocated the first time a verify that deep runs (113 MiB each on Qwen3.8 Flash Next,
  63 MiB on Qwen3.6-35B).

Prompt-cache and prefix reuse need nothing else: a reused prefix is rows below the
watermark and a recurrent state that already sits where the next step expects it.
The serve and prefix tests run with the chain on (below).

**qwen36 with a partial chain** ([a partial chain](#a-partial-chain)). When its layers
do not all fit the device, qwen36 places layers 0..N-1 (their matrices, the DeltaNet's
b|a rows and the shared expert's gate row, the DeltaNet state and conv rings, the K/V
mirrors) and the CPU runs layers N..L-1 and the head. The fit counts a layer's K/V
mirror at the split's floor of three blocks of `COLI_VK_KV_BLOCK` positions; past it the
split decides from what is free at the first forward. qwen36 decides N at start, before
the dense-host decision and the tier, and sets the chain up there and then instead of at
its first forward, one whole layer at a time; the tier then sizes itself from what is
left (its `dense_bytes` is 0: nothing is still to come). A forward runs the N layers
chunk by chunk as above and copies every row's residual after layer N-1 to the host once
per chunk; the CPU then runs its layers over all the rows, layer by layer, and the head.
The device's layers keep their state on the device (the K/V mirrors and their
watermarks, the DeltaNet state and a verify's copies), the CPU's layers on the host (their
K/V rows, their DeltaNet state and the CPU's verify snapshots): a rejected draft rolls each
side back with its own copies. A device lost mid-forward rebuilds the N layers' DeltaNet
state from the prefix record; the CPU's layers have not run that forward, so theirs is
already the host's. With the dense weights on the device only, a layer's host copies go
once the whole layer is on the device; the CPU's layers keep theirs and never read a
matrix back. The matrices of the CPU's layers (and the head's, when it stays) are marked
so that the per-matrix path never uploads them either. `tests/vulkan_engines.sh
partial-qwen36-olmoe` gates it on Lavapipe ([below](#olmoe-and-inkling)).

**What stays on the CPU.** The routed experts the tier does not hold, the router's
top-k, the embedding gather and the vision tower's rows, Qwen3.8's n-gram table reads
and the MTP head's layer apart from its matrices (with the full chain the per-matrix path
runs those on the device, and its routed experts are on the expert tier as an extra
layer).
The chain declines, and the per-matrix path runs with the state synced first, under
the CUDA expert tier (CUDA keeps its priority), a qpack container, PILOT prefetch, or
a geometry outside its shaders (head dim above 256, a DeltaNet value head above 128 or
key head above 256, a conv kernel above 9). A device lost while the chain holds the
recurrent state does not stop the engine: it rebuilds that state on the CPU from the
prefix record (the ids the state was built from; the KV rows are the host's already),
a prefill's worth of CPU work, and runs on the CPU from there. Only a state the ids do
not describe (a turn with an image) cannot be rebuilt: that stops the engine with a
message. `COLI_VK_CHAIN_FAULT=n` fakes the loss at the n-th frame (the tests use it).

**Qwen3.8 on a partial chain** ([A partial chain](#a-partial-chain); `vkc_fit`,
`COLI_VK_CHAIN_LAYERS`). When its layers do not all fit the device,
qwen38's chain takes the first N. At startup, before any upload, the dense-host pass and the
expert tier, the fit counts each layer's matrices in the format they go up in (the trunk's
int8 rows, bf16 or f32) with its shared expert's gate; its state at its starting size (a
DeltaNet layer's recurrent state and conv ring with a verify's first copy of each, an
attention layer's K/V and index-key mirrors at the KV split's floor of three blocks and its
rows of the chunk's K/V read-back, the PLE ring with the PLE layer); and its parameters. The
fixed part is the scratch of one 256-row chunk. The tail is the final mixer, lm_head and the
MTP head's matrices. A partial chain is placed then and there, layer by layer, so the tier
sizes itself after it. The full chain keeps its setup at the first forward, so with every
layer fitting the uploads and the tier's budget are the ones before. A forward runs the N
layers chunk by chunk on the device. After layer N-1 every row's four hyper-connection
streams come back to the host once per chunk, and the CPU runs layers N.. over all the
rows, the final mixer and lm_head. The device's layers keep their DeltaNet, conv, K/V,
index-key and pooled-key state on the device. A verify copies their state into the chain's
slots. The CPU's layers keep theirs on the host, and the CPU's copies roll them back. The PLE
ring and its n-gram history go with whichever side holds the PLE layer. The MTP head runs
outside the chain and reads the final streams from whichever side ran the last layer: with a
partial chain its matrices stay on the CPU, with the full chain and the per-matrix path on
they answer from the device one at a time, as before. A lost
device rebuilds only the device's layers from the prefix record. The vision tower's rows and
the n-gram table stay on the host as before. With `COLI_VK_DENSE_HOST=0` only the N layers'
host copies go (with the full chain fitted, each layer once all of it is on the device). The
tiny fixture on Lavapipe under `COLI_VK_DEVICE_CAP_MB=58.758690` with
`COLI_VK_TIER_RESERVE_GB=0.04`:

```
[VK] qwen38 chain fit: free 61612952 B, reserve 42949672 B, fixed 18569232 B (the engine's 1529872 B, the pools' 17039360 B), tail 8960 B, layers 47664 92768 24112 92768 B, matrices 27080 21184 16832 21184 B
[VK] qwen38 chain: 1 of 4 layers on the device (0.0 MiB), 3 on the CPU, the head and what goes with it (0.0 MiB) on the CPU (free 58.8 MiB, reserve 41.0 MiB)
[VK] qwen38 chain: 1 of 4 layers placed: 27080 B of matrices on the device (the fit counted 27080 B for these layers), chain buffers 4210736 B, device memory held 4329472 B
[VK] qwen38 chain: 1 layers on the device (0 QSA), 18 matrices resident, 0.0 MiB of parameters; the CPU runs the other layers, the final mixer and lm_head
```

`tests/vulkan_engines.sh partial-qwen38` (and `-sanitize`) covers it on Lavapipe. No
Qwen3.8 checkpoint ran on a partial chain and no discrete GPU was available, so what N a
real card takes for the 4.1 GiB trunk is not measured.

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
| `COLI_VK_CHAIN` | on for a discrete GPU; on an integrated GPU with the expert tier, what the engine measured (qwen36 and olmoe on, qwen38 on with 32 or more compute units and off with fewer; mimo, inkling, colibri, glm53, kimi_k3, deepseek_v41 and deepseek_v4 off: not measured); off on a CPU device | `1`: every layer's dense chain on the device; `2`: prompts only (forwards of more than two rows that are not a speculative verify; decode and verifies on the per-matrix path, the state moving between the two); `0`: the per-matrix path. The `[VK] <engine>: dense chain ...` line says which and why. |
| `COLI_VK_CHAIN_ROWS` | from the budget, up to 8192 | Prompt rows per chunk: a longer prompt runs every layer chunk by chunk (the device's scratch is sized for one chunk). Unset or `auto`: the most rows whose buffers fit half the free device memory, up to `COLI_VK_CHAIN_ROWS_MAX` (8192); see [big prompt chunks](#big-prompt-chunks-and-expert-streaming). It was 512 before. |
| `COLI_VK_CHAIN_LAYERS` | what fits | The engines on [the partial chain](#a-partial-chain): the first N layers on the device, the CPU the rest and the head. Unset or `auto`: the most layers the device's free memory holds (every layer when they fit, the chain as before); `n`: n layers; `0`: the chain off. |
| `COLI_VK_CHAIN_FLASH` | `16` | Rows from which a forward's causal attention runs on the matrix units (`chain_attn_flash.comp`: the query heads of one kv head folded into 64 rows a workgroup, four row tiles of 16, K/V staged once a block as f16 for all of them, every subgroup on matrix work, two passes); `0` never. Needs cooperative matrices at subgroup size 64 or 32 (at 32 the first four of the workgroup's eight subgroups take the matrix work, all of them stage and run the softmax); otherwise, and for a selection list, the blocked shader or `chain_attn.comp`. A sliding window, a ring and sinks run here too (Flint on an RTX PRO 4000: 100.2 of a 16K prompt's 106 s of device time had been `chain_attnb`'s; the prompt went from 133 to 1164 tok/s). Workgroups of 16 rows staged every K/V block four times as often: at 64 rows Flint's 16K prompt went from 1148 to 1440 tok/s on the same card, Qwen3.6's from 1119 to 1534 with `COLI_VK_ATTN_SLICE` counted in the shader's own units (below); the Radeon 780M runs the same shader at subgroups of 64. On a Radeon 8060S (Qwen3.6, 1011-token prompt) the attention went from 186 ms (`chain_attnb`) to 22 ms, the first token from 2.08 to 1.85 s, perplexity unchanged (7.65). On an RTX PRO 4000 (Qwen3.6 int4-g64, 467-token prompt) the prefill went from 1413-1477 to 1521-1580 tok/s. |
| `COLI_VK_CHAIN_GEMV` | on | `0`: the decode matrices take `qmatmul.comp`'s GEMV instead of `chain_gemv.comp`'s. |
| `COLI_VK_CHAIN_GEMV_NR` | on | `1`: a speculative verify's rows take `chain_gemv.comp` a workgroup a row, each reading every weight, instead of up to four sharing every weight load ([the dense chain](#the-dense-chain-vk_chainc)). |
| `COLI_VK_DNREC_ROWS` | on | A prompt chunk's DeltaNet recurrence (from 8 rows) on `chain_dnrec_rows.comp`: per block of up to 16 rows the q and k norms, beta and decay in parallel, then the rows in order with no barrier (the column's parts joined by shuffles), then the outputs' gated norm in parallel: three barriers a block where `chain_dnrec.comp` takes four a row. On an RTX PRO 4000 (Qwen3.6 int4-g64, 467-token prompt) the prefill went from 1501-1579 to 1758-1873 tok/s, the decode unchanged (76.4-77.3); Qwen3.8 IQ2_S/Q2_0 with 19 GiB in the host pool 143.7-144.9 to 147.1-148.5. `0`: `chain_dnrec.comp` for every row count. |
| `COLI_VK_ATTN_DEC` | on | A decode step's or a verify's attention (up to 4 rows, no selection list) over 128 positions or more on `chain_attn_dec.comp`: the positions in blocks of 64 (more while the blocks would pass 1024 workgroups), a workgroup a (head, block, row) for the block's scores (a subgroup a position), max, sum and weighted values, then a pass that joins the blocks (the sink's logit with them) and applies the gate. `chain_attn.comp` gives a head one workgroup that walks every position: Qwen3.6's 16 heads left an RTX PRO 4000 all but idle, about 160 us a layer at 700 positions. Decode Qwen3.6 int4-g64 76.5-77.6 -> 85.3-86.3 tok/s; Qwen3.8's QSA layers (a selection list) keep `chain_attn.comp`. Only where a prompt chunk of the same attention takes `chain_attn_flash.comp` (f16, so a row's bits already depend on how a forward is cut into steps; windows, rings and sinks too since it takes them: Flint's decode after a 17K-token prompt read its sink layers a workgroup a head, 15.4 tok/s): elsewhere (a position-major cache, a device without the matrix units) `chain_attnb.comp` gives a prompt's row the bits `chain_attn.comp` gives it alone, and a prefix resumed must answer to the last digit as the prompt recomputed does (`tests/test_brio_serve.py`, OLMoE and MiMo on Lavapipe). `0`: `chain_attn.comp` always. |
| `COLI_VK_ATTN_DEC_GQA` | on | `chain_attn_dec.comp`'s mode 2: a workgroup a (KV head, block, row) for all of its query heads, 1 to 16 (a pipeline a head dim and grouping, spec constants GQ, HDQ and TPP), each K and V row read once for them. OLMoE's one query head a KV head took mode 0 before: its decode after 2.7K tokens 148.6-148.9 -> 161.1-161.3 tok/s. The scores a thread a position (two at head dims of 256: `COLI_VK_ATTN_DEC_TPP`), eight values a load against the q rows in shared memory, the max on the way; V a column of eight values a thread with the positions over groups of threads, the same groups for a float16 or a float32 cache. The join (mode 1) takes each block's weight once into shared memory, not once a value. The blocks' rule counts four workgroups a KV head. One layer after 18K positions in float16 on an RTX PRO 4000: Qwen3.6's 181 -> 122 us (a K row over a subgroup's lanes, five shuffles a head a position, before), Flint's 32/4 heads of 128 229 -> 131; after 60K 573 -> 372. Decode after an 18K-token prompt: Qwen3.6 UD-Q4_K_M 74.5-75.7 -> 79.3-79.8 tok/s, Flint UD-Q3_K_XL 70.7-71.0 -> 80.5-80.8 (92.1-92.2 after 1K, from 88.6-88.9). `0`: mode 0, a workgroup a head. |
| `COLI_VK_ATTN_DEC_TPP` | `2` at head dims of 256, `1` below | Threads a position for mode 2's scores, adjacent lanes taking a K row's loads in turn, their sums joined by a shuffle. A Radeon 780M's 64 lanes reading 64 rows of 512 bytes at once overflowed its cache: one layer of Qwen3.6 after 18K positions 1216-1303 us before, 1463-1627 at one thread a position, 1025-1094 at two; Flint's heads of 128 1241-1406 at one, 1452-1495 at two. RTX PRO 4000: 125.5 -> 121.8 us at two (heads of 256). |
| `COLI_VK_KV16` | on | qwen36's chain keeps its KV mirrors in float16 where the host's cache is float16 (the default; `Q36_KV=f32` keeps both float32), and olmoe's where its host's values are float16's (`KV16`, [OLMoE and Inkling](#olmoe-and-inkling)): every attention shader (`chain_attn`, `chain_attnb`, `chain_attn_flash`, `chain_attn_dec`) has a twin with its specialization constant 7 (KV16) that reads K and V two halves a word (`vkc_attn_h`, `vkc_attn_wh`), and a step's rows go in through `chain_copy2d.comp`'s H16 twin (`vkc_copy_regions_h`), which rounds by kv16.h's integer arithmetic before it packs: `packHalf2x16` rounds toward zero on RADV (a Radeon 780M), where the copy's halves differed from the host's. The values are the host's, so every path gives a float32 mirror's bits (`tests/test_vk_chain.c`), in half the memory and half the reads. One conversation's chain only (`KV_SLOTS` places share buffers with it) and while the mirror does not split (`chain_kvs.comp` reads float32: a plan that splits takes the mirrors back to float32). RTX PRO 4000, after an 18K-token prompt: Qwen3.6 UD-Q4_K_M decode 60.0 -> 63.1 tok/s, prompt 1706 -> 1774; Flint UD-Q3_K_XL decode 64.2 -> 67.0, prompt 1735 -> 1874; the same greedy answers. `0`: float32 mirrors. |
| `COLI_VK_MLA_DEC` | on | The same for the MLA core (`vkc_mla_core`: GLM-5.2/5.3, Kimi K3, DeepSeek V4.1): a decode's or a verify's rows (up to 4) over 128 positions or more on `chain_mla_dec.comp`, the positions (or a selection list's entries) in blocks of 64 (more while the blocks would pass 4096 workgroups), then joined. `chain_mla.comp`'s core walks every position in a workgroup a head with about twenty barriers a tile, and its time did not change from 64 heads to 128: at 2000 positions 1747 -> 323 us a layer, at 8000 11058 -> 1758 (RTX PRO 4000, a 512-float latent and a 64-float rope). A decode row's bits then differ in the last digits from the same row inside a prompt chunk (the prompt keeps `chain_mla.comp`'s core). `0`: `chain_mla.comp` always. |
| `COLI_VK_CHAIN_MOE` | on where supported | A decode step's (or a short verify's) MoE layer whose experts the tier holds all of runs its routing and experts in the frame, with no host step, and a GGUF's prompt chunk too on qwen36 and qwen38 ([below](#the-routed-experts-in-the-frame-vk_dmoeh)); `0`: the host routes as before. |
| `COLI_VK_SUBGROUP` | on where the size varies | Every chain pipeline requires the subgroup size the device reports (`VK_EXT_subgroup_size_control`), where the device has more than one. Without it an Intel Iris Xe (Windows driver 101.7076) compiled the chain's GEMVs at another size than the shaders read: Qwen3.6 answered garbage with the chain on and correctly with it off. `0`: the driver's choice. `n` (a power of two the device allows): pin `n` instead, here and on the GEMV and expert-batch pipelines that take the same size; on Intel Meteor Lake (Mesa 26.2) the reported 32 left `chain_moe.spv` compiling for over 38 minutes, 16 takes 72-195 s and `test_vk_chain` passes. |
| `COLI_VK_CHAIN_GEMV2` | on where supported | `0`: the int8 decode matrices keep `chain_gemv.comp` instead of `chain_gemv2.comp` (rows a workgroup by the matrix's length). On a Radeon 8060S (Qwen3.6, a 256-token answer, int8 trunk) decode went from 36.8 to 39.0 tok/s; at int4 (`COLI_DENSE_BITS=4`) it measured slower, so it takes int8 only. It stages x in the device's shared memory, up to 64 KiB (#1998). On NVIDIA it takes only a matrix of few long rows (Qwen3.8's 320 x 10240 hyper-connection mixer) and `chain_gemv.comp` the rest, which an RTX PRO 4000 ran faster (decode Qwen3.8 IQ2_XS 39.5/44.4 to 45.5/45.6 tok/s, Qwen3.6's int8 trunk 62.2/61.9 to 65.2/65.0); `1` gives it every int8 matrix there too. |
| `COLI_VK_CHAIN_SPIN_US` | `50000` on a discrete card, `2000` elsewhere | How long a wait on a chain frame polls the fence before blocking (whose wake-up costs 50 to 150 us). At 2 ms a decode step's host slept through its last frames and woke after the device had run dry, about 1 ms a token: Qwen3.6 UD-Q4_K_M on an RTX PRO 4000 decoded 76.5 tok/s, 83.7 at 5 ms, 90.6 from 10 ms (a step's time) on; Flint UD-Q3_K_XL 86 -> 89.6. An integrated GPU shares the package's power with the cores and keeps 2 ms. |
| `COLI_VK_CHAIN_PROF` | off | `1`: one `[VK] chain profile` line of device time per kind of op (timestamps). |
| `COLI_VK_KV_SPLIT` | on | `0`: never split the KV cache past the device's budget ([below](#a-kv-cache-past-the-devices-budget)); the whole mirrors, and past the budget the chain declines. |
| `COLI_VK_KV_DEVICE_ROWS` | from the budget | The positions a split layer keeps on the device; set, the cache splits whenever it is longer. |
| `COLI_VK_KV_BLOCK` | `64` | Positions per block of the split's block table. |
| `COLI_VK_KV_PIN` | off | `1`: enable read-based block pins; QSA/DSA/pooled MLA rounding can then depend on read history. DeepSeek preserves its sparse arithmetic in either mode. |
| `COLI_VK_KV_COLD` | unset | `device`: a split layer's host part attended on the device too, from a shadow of the host's rows ([below](#a-kv-cache-past-the-devices-budget)); off by default, because on the measured integrated GPU it slowed decode. |

Each run and serve turn prints `[VK] <engine> chain: N forwards, F frames (ops,
matmuls, tiled GEMM), the time spent waiting for the device, the routed experts' host
time, the layer steps whose routed experts ran in the frame and the device memory the
chain holds`.

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
what it measured here: qwen36 on, qwen38 off with fewer than 32 compute units (decode, a
chat's steady state, is slower in both modes; `COLI_VK_CHAIN=2` is the choice for long
prompts) and on with 32 or more. Without the tier, and on a CPU device such as Lavapipe, it
is off. An engine not timed on an integrated GPU passes `COLI_VK_CHAIN_UNMEASURED`: off
there, and the line says "not measured". The startup line says which and why:

```
[VK] qwen36: dense chain on (an integrated GPU with the expert tier: measured faster on decode and prefill; COLI_VK_CHAIN=0 off, 1 on, 2 prompts only)
```

**Compute units.** An integrated GPU reports its active compute units in
`VK_AMD_shader_core_properties2`. From 32 units on, qwen38 turns the chain on by default. With
the MTP head on beside the chain and the expert tier, qwen38 also drafts one token per verify
and gives the expert cache of the head a slot for every expert. The tier reserves RAM for that
cache. `Q38_MTP_DRAFTS` and `Q38_MTP_CAP` set either. A chain that holds every layer leaves the
head off unless `Q38_MTP=1` asks for it ([speculative.md](speculative.md)). `coli plan` reads
the units through the Vulkan loader. Where only `vulkaninfo` is available, it warns and prices
fewer than 32 units.

A verify of two drafts is three rows and took 1.67 times a one-row forward, against 1.22 for
one draft. It returned 2.20 tokens per forward against 1.76. A cache of 32 slots misses the FP8
experts of the head, which are then read from disk on every draft (2,707 to 3,111 loads a run
against 614 with a slot for every expert). Qwen3.8 Flash Next with the int4-g64 sidecar on a Radeon 8060S (40 compute
units), `COLI_VULKAN=1`, the head's layer on the CPU (`Q38_MTP_CHAIN=0` keeps it there), an 84-token prompt and 300 new
tokens, greedy, single runs:

| Arm | tok/s |
|---|---|
| trunk on the CPU, 2 drafts, head cache 32 | 8.3 |
| chain on, 2 drafts, head cache 32 | 17.4 |
| chain on, 1 draft, head cache 32 | 19.8 |
| chain on, 2 drafts, head cache 512 | 20.0 |
| chain on, 1 draft, head cache 512 | 22.2 to 23.4 |

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

**A GGUF of block types** (ggml-org's MiMo-V2.6-Flash Q2_K_S, say: K-quant experts and
trunk). The engine used to take a GGUF only with MXFP4 experts and a BF16 trunk. A trunk
matrix of a ggml block type now stays the file's rows (`DW_GQ`): the CPU multiplies it in
its integer form (`ggml_quants.h`), the chain uploads it as the device's block format, and
the per-matrix path leaves it to the CPU. The experts are read as the file has them (a slot
`[down | gate | up]`, each layer its types), computed on the CPU with the same kernels and
taken by the tier as their blocks, the commonest pair of types as the main form. A block
embedding becomes the bf16 rows the engine keeps.

**The expert history.** mimo now counts its routing (`route_trace.h`) and keeps it in
`COLI_USAGE` (default `<model>/.coli_usage`), and the tier starts warm from it
(`vkt_plan`, `vkt_put`). Before, every start had the tier empty, filling as experts passed
by. On MiMo-V2.6-Flash Q2_K_S (106 GB) with an RTX PRO 4000 and 62 GB of RAM, three
120-word prompts of 128 tokens on one server: decode 2.0, 2.9, 3.8 tok/s cold, then 7.8 from
the first request once a run had written the history (85% of the routed experts on the
device, about 1230 disk loads a turn instead of about 21500).

**A prompt chunk's scratch.** The chain reserved its scratch for each prompt's rows, so a
longer prompt than the first had to grow it after the tier's promotions had filled the
budget: on the server above the second request's 172 rows were refused and the chain was
off for the rest of the session (decode 7.9 then 3.4-3.9 tok/s, prefill 7.9 then 6.0 and 3.9). The
first prompt now reserves the whole chunk (`mc_chunk_rows`, sized from the room free at
the first forward), and a refused growth runs the prompt in chunks of the rows the chain
holds instead of turning it off: prefill 8.1-8.2 and decode 7.8-7.9 tok/s on every request.

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

**A partial chain.** MiMo-V2.6's dense part (about 24 GB on Flash, 32 GB on Pro in its
native FP8/BF16 form) does not fit most cards, so the chain takes the first N layers
that do (`vkc_fit`, decided at start-up before anything goes up; `COLI_VK_CHAIN_LAYERS`
forces N) and the CPU runs the others and the head. A layer's bytes are its matrices in
their `MIMO_DENSE_BITS` form (the fused qkv, o_proj, and the dense layer 0's MLP), its
K/V mirror at the size the chain allocates (a sliding layer's ring; a full layer's
context, or `COLI_VK_KV_DEVICE_ROWS` when that asks for fewer: the split covers the
rest) and its norms and sink logits; the head is the tail, with the vision tower when
the per-matrix path would take it (`COLI_VK_DENSE`). A partial chain places its layers
before the tier sizes itself; with everything fitting, the layers go up after the tier
as before. Each prompt chunk crosses the N device layers, its residual rows come down
with the frame that ends it (the buffer the normed rows use), and the CPU runs layers
N.. and the head on those rows before the next chunk; a chunk counts as computed once
both sides ran it, so a device lost in a later chunk loses nothing either side holds,
and the CPU runs on from where the host's caches end. The device's layers keep their
mirrors behind the watermarks above; the CPU's layers keep their caches on the host
only, and nothing of them, of the head or of the tower goes up through the per-matrix
path. A layer that does not fully reach the device is freed and the chain keeps the
layers before it. With `COLI_VK_DENSE_HOST=0` only the N layers drop their host copies,
and `coli plan` (`resource_plan.py`, MiMo's layout) predicts the same N from the
checkpoint's header and config and credits those layers alone.
`tests/vulkan_engines.sh partial-inkling-mimo` gates it on the fixture (every N from 0
to 6, text and picture, the three dense forms, N from a device cap and from an upload
failing inside a layer, chunks of 3 rows, one token at a time, prompts only, the KV
split, a lost device, device-only weights, serve sessions and the prefix-reuse tests)
on the CPU's tokens and logits within 1e-4 of the largest, and `coli plan`'s numbers on
the engine's (free, per-layer and fixed bytes, N) under each cap. Lavapipe only: no
discrete GPU was available, so the fit's choice on a real card is not measured.

**The default.** The chain's speed on a real MiMo model is not measured: no MiMo
checkpoint is on the test box (the smallest, Flash, has 309B parameters). The chain is
on for a discrete GPU (the common rule), off on an integrated GPU (`COLI_VK_CHAIN=1`
turns it on) and on a CPU device:

```
[VK] mimo: dense chain off (an integrated GPU with the expert tier: not measured; COLI_VK_CHAIN=0 off, 1 on, 2 prompts only)
```

### OLMoE and Inkling

olmoe (`olmoe_chain.h`) and inkling (`inkling_chain.h`) run the chain too, with the same
knobs, default rule and `[VK] <engine> chain: ...` line as qwen36. A GGUF's matrices go up
as the file's blocks (`olc_gq_tensor`, `inkc_tensor`: 100 + the ggml type, rows the
loader interleaved by 16 back to the file's first, whole groups of them for Inkling's
head, whose unpadded vocabulary stops inside the last one), OLMoE's experts in the frame
with their block decoders; outside the chain (the per-matrix path, the dense weights on
the device only) a GGUF's dense part stays on the CPU. Before, both chains were off for a
GGUF. On an RTX PRO 4000: OLMoE-1B-7B Q4_K_M decode 71-73 -> 210 tok/s and a 176-token
prompt 549 -> 2454 tok/s (the CPU's greedy text); Inkling-Small UD-IQ2_XXS (77 GB, 62 GB
of RAM) decode 6.9-7.1 -> 9.4-10.4 tok/s and a 173-token prompt 22.2-22.7 -> 17.6-29.2.

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
olmoe mirrors it behind qwen36's watermark. Its K and V values are float16's (`KV16`, the
default): the CPU rounds a step's rows before anything reads them and keeps them in its
float32 arrays, and the chain's mirror holds them in float16 (`COLI_VK_KV16`), its rows
rounded on the way in (`vkc_copy_regions_h`) and again by the host as they come down, so
both sides hold the same numbers. A mirror that splits goes back to float32, the device
then rounding the rows first (`chain_ew.comp`'s op 11). On an RTX PRO 4000, OLMoE-1B-7B
Q4_K_M: one layer's decode attention after 1280 positions 61.5 -> 35.4 us, decode after
1K / 2.7K tokens 244.9-245.0 / 175.6-175.9 -> 273.7-275.2 / 220.8-222.1 tok/s, six 1K
prompts a server start 4506-6414 against 4529-6406 tok/s before. Inkling's mirror has the host's layout, a
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

**olmoe with a partial chain** ([a partial chain](#a-partial-chain)). olmoe decides N in
`model_init`, before the dense-host decision and the tier: it takes the chain's on/off
decision there from the same inputs, silently, as the dense-host decision already did
(the `[VK] olmoe: dense chain ...` line still follows the tier), brings the chain's
pipelines up and sets the N layers up there and then. A forward runs the N layers and
copies every row's residual to the host; the CPU runs layers N..L-1 and the head over
all the rows. Only the N layers mirror their K/V rows (the split covers those); the
CPU's layers keep theirs on the host. `PILOT` keeps prefetching the model's next layers
from the chain's rows, wherever those layers run. A device lost mid-step: the CPU redoes
the step from its embedding rows, as before (the CPU's layers had not run it). With the
dense weights on the device only, the automatic cache grows by the N layers' host copies
alone.

**Partial-chain tests** (`tests/vulkan_engines.sh partial-qwen36-olmoe`, and
`partial-qwen36-olmoe-sanitize` under ASan and UBSan). For qwen36 (the hybrid, Qwen3-Coder,
the 27B dense geometry, int8 and int4-g64 rows, an image) and olmoe: `COLI_VK_CHAIN_LAYERS`
from 0 to L against the CPU's tokens and logits, with the matrices on the device after
setup the N layers' exactly; a `COLI_VK_DEVICE_CAP_MB` computed from a probe so that
exactly k layers fit; an upload failing inside layer k's setup (N = k, nothing of layer k
left on the device); and with N < L prompt chunks with expert streaming, the tiled GEMM,
the tier off, prompts only, prompt-lookup verifies (against the CPU and byte for byte
against the same run without drafts), the KV split with a host part, a lost device
mid-decode, serve sessions with pins and prefix reuse, Clef's oracle, olmoe's `PILOT`,
and the dense weights on the device only (the N layers' matrices dropped alone, none read
back on a healthy run, only theirs after a lost device). On Lavapipe only: no discrete
GPU was available, so the placement on a real card's budget is the cap's emulation.

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

**Inkling's partial chain.** Inkling's dense part is 49 GB in bf16 (about 15 GB in the
dense-int4g64 container), so on any consumer card its chain is partial: `vkc_fit` decides
at start-up, before anything goes up and before the expert cache is sized, how many
layers from the first fit (`COLI_VK_CHAIN_LAYERS` forces N), and the CPU runs the others
and lm_head. A layer's bytes are its matrices in the form each goes up in (q, k, v, r,
o_proj; the dense MLP's three, or the router's f32 copy and each shared expert's three),
its share of the parameter arena, its four convolution rings and its K/V mirror at the
window's rows (a global layer's grows with the context: the KV split covers that). No fit
is made when the chain would decline anyway: bf16 matrices on a CPU whose bf16 dot
rounds the activations go to the device in no form (the `[VK] inkling: 0 resident
matrices go to the GPU` case), and the chain declines as before. A partial chain places
its layers at start-up, before the tier sizes itself (so does every fit with
`COLI_VK_DENSE_HOST` dropping the host copies, which drops each layer's only once all of
it is on the device); with everything fitting and the host copies kept, the layers go up
at the first forward as before. Every chunk of a forward crosses the N layers, the
residual rows come down once a chunk, and the CPU runs layers N.. over all the rows
layer by layer, as its own forward does, then the head and the per-position heads. The
device's layers keep their mirrors, rings and convolution states as above; the CPU's
layers keep theirs on the host, so a CPU step lowers and a lost device rebuilds the
device's layers alone (`rebuilding the state of P positions ... (the device's layers;
the CPU's have theirs)`). `coli plan` (`resource_plan.py`, inkling's layout, which reads
the dense-int4g64 container's forms when it is there) predicts the same N.
`tests/vulkan_engines.sh partial-inkling-mimo` gates it on the tiny fixtures (every N
from 0 to 8, the dense-int4g64 container, the expert containers, bf16, D = 6144 with N
from 0 to 2, N from a device cap and from an upload failing inside a layer, chunks of 3,
prompts only, the KV split, a lost device, device-only weights, serve sessions and the
prefix-reuse and dashboard tests) on the CPU's tokens and every forward's logits within
1e-4 of the largest, and `coli plan`'s numbers on the engine's under each cap, on
Lavapipe; no discrete GPU was available.

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

**A GGUF (glm53).** GLM-5.3 Flash's matrices from a GGUF go up as the file's blocks
(100 + the ggml type, the device's block formats; rows the CPU keeps interleaved by 16
go back to the file's first), as qwen36's and qwen38's chains take them. The MLA's
per-head `k_b` and `v_b`, which the absorbed core reads a head at a time
(`vkc_mla_hgemv`, no block form), go up dequantized as floats. Before, a GGUF's chain
stopped at "a matrix with no device form" and the whole trunk ran on the CPU (UD-IQ2_XXS
on an RTX PRO 4000: 63% of the CPU's time in its Q5_K/Q6_K rows). Now every layer is on
the device (5.96 GiB): decode 0.8-1.1 -> 2.8-3.2 tok/s, a 171-token prompt 2.2 -> 4.8-6.3
tok/s, with 62 GB of RAM and two NVMe drives for the 95 GB the RAM and VRAM do not hold.
This model's greedy text forks at near ties where the CPU's own two forms fork too: on
an 8-token prompt the CPU's integer and float sums (`GQ_EXACT=1`) put the logits 28 to
37% apart, and the chain against the CPU's float sums 2e-3 on the first row and up to 8%
after it.

**A GGUF (colibri).** GLM-5.2's matrices from a GGUF go up the same way: the file's blocks
(100 + the ggml type), rows the CPU keeps interleaved by 16 back to the file's first
(`glmc_gq_rows`). A GGUF splits `kv_b` into the per-head `k_b` and `v_b`, which are the
absorbed core's `k_abs` and `v_abs`: they go up dequantized as floats, as glm53's, and the
per-matrix path never takes them for a whole `kv_b`. Before, the chain had no form for
either and every layer stayed on the CPU. UD-IQ1_S (202 GB) on an RTX PRO 4000 with 62 GB
of RAM and two NVMe drives, a 166-token prompt and 128 tokens: the trunk on the CPU with
the tier on the whole card (2165 experts) gave 2.6 tok/s of prompt and 0.67-0.68 of
decode, the trunk on the device (78 layers, 16 GiB) with the tier on the rest (580
experts) 2.8-2.9 and 0.72-0.74 (the CPU alone 1.9 and 0.46). The model is three times the
RAM: the CPU's share of the experts read from the disk is the cost either way, and
giving the tier more of the card measured no better (39 layers on the device and 1313
experts: 2.7 and 0.67-0.85; 20 layers and 1685 experts: 2.5-2.7 and 0.62-0.68). With
the device's float form (`GQ_EXACT=1 COLI_VK_IDOT=0 COLI_VK_CHAIN_GEMM=0`) the greedy
text is the CPU's float form's for 40 tokens, and the first 4 tokens' top-10 logprobs
agree within 1e-4; the integer forms fork at a tie, as the CPU's own two forms do.

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
rows go through the chain, the MTP head runs on the CPU (its routed experts on the
expert tier on a discrete GPU) and reads the chain's final rows, and n-gram drafts work
the same way. glm53 has no draft path.

**A lost device.** The forward that failed runs again on the CPU from its input (the
chain keeps the caller's rows untouched until its last chunk is through), and the CPU
runs from there. colibri has nothing to rebuild (its cache is the host's). glm53
rebuilds the KDA state on the CPU from the input rows the chain records since the host's
copy was last current (embedding rows, or the vision tower's), a prefill's worth of CPU
work.

**Several conversations at once (qwen36).** With `KV_SLOTS` above 1 the chain keeps each
conversation's state on the device (its KV mirrors, DeltaNet state and conv ring), in as
many places as the device's free memory takes once the expert tier has its budget
(`COLI_VK_CHAIN_MUX` caps them). A prompt runs on the chain in its conversation's place,
which keeps its state for the next turn; a decode step runs one row of every active
conversation in one forward, the projections, the router and the routed experts over all
the rows, each row's attention and DeltaNet on its own place, its new K/V rows to its
conversation's host cache. A conversation without a place takes the least recently used
one (that conversation's DeltaNet state goes to the host first); a step with a row whose
conversation has no place runs on the host's path. A device lost mid-step rebuilds every
conversation's state that was on the device from its prefix record, on the CPU. The end
of a turn prints `[VK] qwen36 chain: KV_SLOTS=n: s decode steps of several conversations
in the chain (r rows, ... a step), h on the host's path; p conversations' state on the
device, e handed over`. The second device's chain, a partial chain and the KV split keep
the older way: the chain runs each conversation's prompts, its state handed over through
the host when the conversation changes, and the decode steps run on the host's path.

**Several conversations at once (the other engines).** qwen38, OLMoE, Inkling,
Kimi K3, MiMo, DeepSeek V4 and V4.1 decode `KV_SLOTS`' conversations together too, but
their dense chain keeps one conversation's state on the device, so it stays off with
more than one slot (a `[VK] <engine>: KV_SLOTS=n: the dense chain is off` line says
so): the dense part runs on the CPU, and the routed-expert tier serves the experts of
every conversation's row. `tests/vulkan_engines.sh mux` checks each engine's frames
against the same requests served alone, on the CPU and with the tier.

**Several conversations at once (colibri).** `KV_SLOTS`' batched decode (one row from
each active conversation, each at its own position) runs on the chain: the norms, the
projections and the MoE take the batch's rows as one; each row's new KV rows go to its
own conversation's mirror, and its attention core runs over that mirror
(`vkc_mla_attn_rows`) with its own DSA list over its own index keys. A conversation's
mirror is the chain's own when the chain holds it (the last one a prompt or a one-row
step ran for), else one of up to `COLI_VK_CHAIN_MUX` (default 16) beside it: whole,
with its own watermarks, which a turn that goes back in its conversation lowers as on
the host. Past that count the mirror read longest ago gives way, and its conversation's
rows go up again from the host when it comes back. A mirror holds `positions x
(kv_lora + qk_rope)` floats a layer (and `positions x index_hd` on a full DSA layer),
positions in powers of two up to the context, and is placed only within four fifths of
the device's free memory. A step whose conversations the mirrors cannot all hold, a
step with the KV split on, and `COLI_VK_CHAIN_MUX=0` run on the CPU as before. With a
second device each chain runs its layers for the whole batch, the DSA lists crossing as
in one conversation's step; a partial chain hands the rows (and their lists) to the CPU
after its layers. The `[VK] colibri chain: n multiplexed steps (r rows), ...` line at
exit says what ran. `tests/vulkan_chain_mux.py` compares every frame with the CPU's:
waves of requests on 3 and 4 slots sent at once (the batches grow and shrink; the
second wave extends, rewinds and replaces each conversation), DSA top-4, one mirror
beside the chain's, the device lost in a multiplexed step, the partial chain and two
devices with a shared indexer at their edges. Not yet measured with a real checkpoint.
glm53 still runs such steps on the CPU.

**Declined** (the CPU path runs, the state synced first): glm53's multi-slot decode
batch (a single-slot serve's one-row batch takes the chain), a layer range (a segment),
a quantized KV cache (KV8, KV_TQ), PILOT, LOOKA, the exact verify of
`COLI_EXACT_VERIFY`, the CUDA backend, matrices with no device form (int2, E8/IQ3, fp8
dense matrices), and geometries past the ops' limits. With the chain on, the per-matrix
switches (`COLI_VK_DENSE`, `COLI_VK_ATTN`, `COLI_VK_DEV2`) keep working beside it: the
chain's tensors are their device copies where they made one.

**A partial chain** ([the fit](#a-partial-chain)). GLM-5.2's dense part is 9.9 GB, more
than an 8 or 12 GB card holds. Both engines decide N right after the load, before the
chain's pipelines take their first blocks and before colibri's pins and `cap_for_ram`,
glm53's expert cache and the tier size themselves, and place layers 0..N-1 one at a time
(`COLI_VK_CHAIN_LAYERS=n` forces N). A forward runs all
its chunks through the N device layers; then the residual rows (colibri) or the hc_mult
streams (glm53) come back to the host once per chunk, and the CPU's own layer loop runs
layers N..L-1, the final norm and the head on them. When colibri's layer N is a shared
DSA indexer layer, the selection made by the device's last full layer comes back with
the rows (1 + `index_topk` ints a row) into the CPU's, so layer N reads the same positions
it would on the device. The KV mirror, its watermarks, the KV split and glm53's KDA state
on the device cover the N layers; the CPU layers' KV rows and KDA state are the host's
alone. After a lost device colibri runs the forward again on the CPU, and glm53 rebuilds
the KDA state of the device's layers only (the CPU layers ran every position already).
With `COLI_VK_DENSE_HOST=0` only the N layers drop their host copies, each once all of it
reached the device; colibri's `resident dense`, and with it the pins and `cap_for_ram`,
counts only what was dropped, and glm53's expert cache sizes itself after them. `coli
plan` predicts glm53's N from the same numbers (`_glm53_chain_layout` in
`resource_plan.py`, checked against the engine's fit line); colibri's dense formats come
from its command line, which the plan does not see, so it gives colibri no device-only
credit and no N. lm_head,
and colibri's MTP layer and eh_proj, go to the device only with every layer there and
room for them. Until then the per-matrix path (`COLI_VK_DENSE`, `COLI_VK_ATTN`) multiplies
on the device only what the chain placed, and the line
`[VK] <engine> chain: N of L layers held at exit: M B of matrices on the device` says so at
exit. The `partial-glm` and `partial-glm-sanitize` families
(`tests/vulkan_partial_glm.sh`) check it on the tiny fixtures, with glm_tiny's shared
indexer layers and a six-layer GLM-5.3 whose KDA and MLA layers alternate, so MLA layers
run on both sides of the handoff. On Lavapipe every configuration gives the CPU's tokens,
every logits row within 6.7e-7 (colibri, 43 configurations) and 7.8e-7 (glm53, 38) of the
largest logit, and `coli plan` predicts glm53's N, free, per-layer and fixed bytes as the
engine prints them under the four caps it is aimed with; the sanitized runs report no
diagnostic. No discrete GPU was available, and the 780M box was busy with other work: the
fit on a real card's budget is not measured.

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

### DeepSeek V4.1 Flash and DeepSeek V4 on the chain

`deepseek_v41_chain.h` (deepseek_v41) follows the recipe below with the
[DeepSeek ops](#deepseek-v41-flash-and-deepseek-v4s-attention-vkc_dsv4), the mHC ops and
the per-head blocks of `vkc_mla_hgemv`. The residual is `hc_mult` streams per position,
and V4.1 collapses a site with the mix the site before it computed. What runs where, per
layer:

| | deepseek_v41 (DeepSeek V4.1 Flash) |
|---|---|
| device, frame A1 | the previous layer's FFN branch (the routed sum plus the shared expert) written back into the streams; on an engram layer the engram (`eng_wkv` over the n-gram rows the host looked up, the gate into each stream); on a DSpark target layer the streams' mean for the draft head; the attention site's mix, split with Sinkhorn, collapse and norm; the attention: `wq_a`, its norm, `wq_b`, `wkv`, its norm, RoPE on interleaved pairs, the new rows into the window ring; on a kv_source layer the compressor's rolling group, the pooled latent's norm, the index keys and the compressed rows' RoPE; on an index source the indexer (its queries and RoPE, `weights_proj`, the scores against the keys the CPU would read, the candidate blocks, the top-k); the sparse attention with the sink over the window and the selection, the inverse RoPE, the grouped `wo_a`, `wo_b`; the write back; the FFN site's mix, collapse and norm |
| host | `moe_run_at` without the shared expert: the router and the routed experts (the tier's batch and the CPU's share); an engram layer's n-gram rows are looked up (on disk) as its frame is recorded |
| device, frame A2 (not waited for) | the shared expert (clamped SwiGLU) |
| after the last layer | the final streams and the last site's mix back to the host, which collapses them and runs the final norm and the head as before |

**The state.** The host's stays canonical. What a forward changes there, the window ring
and its position map, the compressed rows and index keys, the compressor's group and the
published index keys, is written back once the forward's last frame is through, exactly
as the CPU would have left it, a speculative verify's undo rows included. The device
mirrors the window ring (window + chunk rows, so a chunk never overwrites a row one of
its earlier rows still reads) behind a watermark, each kv_source layer's compressed rows
and index keys behind one of their own (grown in powers of two), and takes the
compressor's group up at every forward. Every host write lowers the watermarks: a CPU
forward, a rejected draft, a reset. The index keys each layer scores follow the engine's
published-key rule (the released behaviour; per row on a verify) and `V41_INDEX_OWNER=1`.

**Drafts.** DSpark's stages stay on the CPU and read the host's window rings and the
chain's means; a verify's rows go through the chain (its matrices on the per-row GEMV,
so each row gets a decode step's bits), a rejection is the host's rollback and the
watermarks follow.

**A lost device.** The forward that failed runs again on the CPU from its input (the
chain leaves it, and the host's state, untouched until every chunk is through), and the
CPU runs from there. There is nothing to rebuild.

**Declined** (the CPU runs the layers, the watermarks follow): `V41_TRACE`, prompts only
when the forward has two rows or fewer, and a model the ops or the chain do not take: a
head above 1024 floats, a window plus top-k above 3072 entries, an indexer above 64 heads
or 4096 query floats, more than 8 streams, a compressed layer reading the index list of
another ratio (or of none), a candidate mask read across ratios.

**Arithmetic.** V4.1's CPU multiplies f32 activations everywhere, so the chain does the
same arithmetic in another order. On the fixtures every configuration gives the CPU's
tokens, and every logits row is within 1e-6 of the largest logit: 9.3e-7 at worst on
Lavapipe, 1.1e-6 on a Radeon 780M (RADV) and 1.0e-6 on an Intel Iris Xe (Mesa's Dozen,
four configurations).

**A partial chain** ([below](#a-partial-chain)): when the dense layers do not all fit the
device, deepseek_v41 chains the first N and `forward_full`'s CPU loop runs the rest from
layer N (`v41c_forward` returns the layers it ran). The handoff gives back every row's
`hc_mult` streams and the last FFN site's mix, and the host state the CPU's layers read
from a chain layer within the same forward: the candidate mask (`m->candidates`, when the
candidate source is a chain layer and an index source after N reads it) and the index
list the last chain index source published (`m->shared_topk`, which a compressed layer
after N reads before the next index source runs). A chain layer that reads a CPU layer's
index keys (the published-key rule: the layer that published last, or an earlier row's
layer in a verify) reads the host's rows as the CPU's layer order has them at that point,
from a mirror of those keys lowered after every forward. The chain's layers write their
rings, compressed rows, keys and groups back at the end of the forward as before; the
CPU's layers keep theirs on the host. The fit counts each layer's matrices (fp8 as fmt 12
with a scale per 32 inputs, the compressor's and the indexer's bf16, the mHC mixes in
f32, an engram layer's `eng_wkv`; for prompts only with the dense weights on the device
only, wo_a per output group too), its window ring, its compressor's group and 64
compressed rows and keys, its share of the parameters and of a forward's pull buffer; the
fixed part is the scratch of one 512-row chunk, the tail (with `COLI_VK_DENSE`) the head
and the routers. With the dense weights on the device only the fit runs before the
layers are read and each chained layer goes up whole as it is read (its matrices, an
engram layer's projection, its state) before its host pages are given back, so a layer
that does not reach the device reads nothing back from disk. DSpark's stages take the
means of their target layers from whichever side ran them.

**A GGUF.** The published deepseek41 GGUFs' matrices go up as the file's blocks (100 +
the ggml type), rows the CPU keeps interleaved by 16 back to the file's first
(`v41c_gq`), and their BF16 compressors and indexers as bf16 (fmt 11). The grouped
output projection `wo_a` runs on `vkc_mla_hgemv`, which reads no blocks: it goes up as
int8 with a scale per 32 inputs (fmt 13) from Q8_0 and as floats (fmt 10) from the other
types. Before, the chain refused a GGUF and every layer stayed on the CPU. UD-IQ1_S (233
GB) on an RTX PRO 4000 (24 GB) with 62 GB of RAM, a 163-token prompt and 128 tokens, a
warm-up and three runs, each server on a copy of one expert history:

| | prompt | decode |
|---|---|---|
| the CPU | 6.0-6.7 tok/s | 2.21-2.23 tok/s |
| the chain off (the per-matrix path holds 60 matrices, the rest of the trunk on the CPU), the tier on the card (2893 experts) | 16.6-16.9 tok/s | 2.91-3.16 tok/s |
| the chain (40 layers, 9.5 GiB), the tier on the rest (1631 experts) | 17.3-18.6 tok/s | 5.00-5.23 tok/s |

The greedy text is the CPU's for 40 tokens in both forms: the integer forms, and the
device's float form (`GQ_EXACT=1 COLI_VK_IDOT=0 COLI_VK_CHAIN_GEMM=0`) against the CPU's.
`tests/vulkan_engines.sh deepseek-chain` runs a small file of the same types and shapes
(`tests/test_gguf_import.py` `write_deepseek41`) against the CPU's float form: the logits
within 1.4e-6 of the largest on the 780M.

**The default**: `COLI_VK_CHAIN_UNMEASURED`, so the chain is off on an integrated GPU
(`COLI_VK_CHAIN=1` turns it on, `2` for prompts only) and on a discrete GPU follows the
rule above. No DeepSeek checkpoint fits the 780M box (V4.1 Flash is 510 GB, the
published UD-IQ1_S 233 GB): there the tests prove the tokens on the tiny fixtures, on
Lavapipe, the 780M and the Iris Xe:

```
[VK] deepseek_v41: dense chain off (an integrated GPU with the expert tier: not measured; COLI_VK_CHAIN=0 off, 1 on, 2 prompts only)
```

### DeepSeek V4 on the chain

`deepseek_v4_chain.h` runs DeepSeek V4's layers on the dense chain (`make deepseek-v4
VK=1`, `COLI_VK_CHAIN=1`), every forward the engine makes: prompts, decode steps, the
verify of a draft, the teacher-forced pass of `--record-oracle`, serve turns. What runs
where, per layer:

| | deepseek_v4 |
|---|---|
| device, frame A1 | the previous layer's FFN branch (the routed sum from the host plus the shared expert, to bf16) written back into the hc_mult streams; the attention site (hc_attn_fn, the split with Sinkhorn, the collapse with the site's own pre, bf16, the norm, bf16); the attention: the input to E4M3, wq_a, q_norm; on a compressed layer the compressor (wkv and wgate in bf16 on the f32 input, the ring with its position bias, the pooled rows' norm, RoPE at the group's first position with YaRN's table, the no-position part to E4M3 per 64); on a ratio-4 layer the indexer (its own compressor into the keys with the Hadamard transform and E2M1 per 32; the queries from the q latent, RoPE, Hadamard, E2M1; weights_proj; the scores and the top-k in score order); wq_b and the per-head RMS without weight, wkv and kv_norm, RoPE, the key's no-position part to E4M3 per 64, the new rows into the window ring; the sparse attention with the sink over the window and the compressed rows (V4's bf16 weights and output), the inverse RoPE, wo_a per group, wo_b; mHC's exit; the FFN site; every bf16 and E4M3 rounding where the CPU makes it |
| host | the router (bf16, or the hash router on the token ids) and the routed experts: the expert tier's batch and the CPU's share, summed in the CPU block's order (ascending expert, then rank) |
| device, frame A2 (not waited for) | the shared expert: the input to E4M3, w1 and w3, V4's SwiGLU between bf16 roundings, the input to E4M3, w2, bf16 |
| after the last layer | the streams back to the host, which runs the final collapse, the norm and the head as before (and DSpark's taps of the last three layers) |

The matrices are the per-matrix path's device copies, found in the same map (fp8 as
fmt 12, bf16 as fmt 11), and the mHC mixes in f32 (fmt 10), which only the chain
multiplies. A GGUF's go up in their own types (`v4c_gq_fmt`): the device's block
formats as the file has them (rows the loader interleaved by 16 back to the file's
first), a float view as stored; the grouped `wo_a`, which `vkc_mla_hgemv` reads a group
at a time and which has no block form there, goes up as int8 with a scale per 32 from a
Q8_0 (fmt 13, the same numbers) and as floats from any other type. Before, a GGUF's chain
was off and the trunk ran on the CPU: on DeepSeek V4 Flash 0731 UD-IQ2_XXS (85 GB, an RTX
PRO 4000 with 62 GB of RAM) decode 2.2-2.5 -> 5.6-6.2 tok/s and a 163-token prompt 6-10 ->
16-19 tok/s, a 703-token one at 20-21.5; the logits within the model's own spread (its
CPU's integer and float sums 6-9% apart, the chain 4-9% from the floats, the same greedy
tokens). `make deepseek-v4 VK=1` on its own builds the block formats' shaders too now (the
target depends on the parent's `VK_SPV`): its own list had no `qmatmul_bd.spv`, and such a
build ran DeepSeek V4 Flash UD-IQ1_S (82.5 GB) with the chain and the tier off, 0 matmuls on
the GPU. Built whole, on the same card: the CPU 7.9 tok/s of a 163-token prompt and 2.77-2.79
of decode, the chain (43 layers, 6.0 GiB) and the tier (2382 experts) 20-22 and 6.62-6.94,
the same greedy text. A whole chain (every layer and the head) gives the tier no dense bytes still to
come: its matrices were counted twice before, and the tier's budget went from 10.3 to
16.5 GiB. Every matrix takes the per-row GEMV, never the tiled GEMM: a row's bits then
do not depend on how a forward is cut into chunks, nor on whether the row is a prompt
row, a decode step or a verify row, which is what the CPU gives (its batched and
per-token kernels agree bit for bit), and what makes a reused prefix give the bits of a
cold prefill. A prompt's matrices are slower for it than they could be.

Four roundings are ops of their own, `vkc_dsv4_round` (bf16 over segments; E4M3 per
block, the scale the smallest power of two that brings the block's maximum under 448;
E2M1 per block, the scale the smallest that brings it under 6; the Hadamard transform
with its bf16), each the engine's C bit for bit, and `vkc_dsv4_swiglu`. The indexer's
scores past 4096 query floats (V4's 64 heads of 128) read the queries from memory instead
of staging them, the same sums.

**The state.** The host's stays canonical. What a forward changes there (the window ring,
the compressed rows and their count, each compressor's ring, the indexer's keys and count
and its compressor's ring) is written back once the forward's last frame is through, as
the CPU would have left it. The device mirrors it: the window ring (window + chunk rows)
behind a watermark, with the position each row holds, so a row a rejected draft longer
than a chunk overwrote goes up again; the compressed rows and the index keys of each layer
behind watermarks of their own; each compressor's ring behind a flag. Every host write
lowers them: a CPU forward, a restored snapshot (a rejected draft), a reset, a prefix
checkpoint or a pin restored, another attention state, a forward that does not start
where the last one ended.

**Drafts.** n-gram drafts (`V4_DRAFT`) verify through the chain and replay the accepted
rows through it after a rejection. The full DSpark drafter (three MTP stages) stays on the
CPU, reading the taps of the target's last three layers that the chain copies back; the
tiny fixture has one MTP layer, so that path ran target-only in every test here.

**A lost device.** The forward that failed runs again on the CPU from its input (the chain
leaves the host's state and the caller's rows untouched until the last frame), and the CPU
runs from there: nothing to rebuild.

**Declined** (the CPU path runs, the device's copies follow): no resident dense layers (a
low-memory plan reloads them per forward, and the `--oracle` path's own copies), the CUDA
tier, prompts only (`COLI_VK_CHAIN=2`) for forwards of two rows or fewer, a forward whose
attention list (the window plus every compressed row of a layer without an indexer) would
pass 3072 entries (from there on in that session), and geometries past the shaders (head
dim above 1024, an indexer head that is not a power of two or above 4096, a top-k above
4096, more than 8 streams).

**Arithmetic.** The device sums in other orders than the CPU (the GEMV, the norms' and
mHC's reductions) and its exp and sqrt are not glibc's, so an intermediate value now and
then lands on the other side of a bf16 rounding. V4 then amplifies it: the next E4M3
rounding of a block that holds that value can move a whole step (one part in 8 to 16),
and the indexer's top-k can pick another compressed row. Measured on Lavapipe: given the
same input, a layer on the chain gives the CPU's output bit for bit apart from such
single-value flips (checked by feeding the CPU the chain's layer output); on the fixtures
51 to 100% of the logits rows are bit-identical to the CPU's, and the worst row moves by
up to 0.32 of its largest logit (the 2-output-group fixture's 72-token case, after one
flipped norm value at layer 0 moved a key row by an E4M3 step); in one of 1,500
teacher-forced positions the argmax moved, never in a generated stream. So the gates
are: the CPU's generated tokens exactly (and the reference's), every logits row within
0.5 of its largest |logit|, and two checks no driver can blur: the chain against itself
(chunks of 1, 2, 3 or the default, prefill chunks of 3, the tier off, drafts rejected and
accepted give the same bits) and each decode row equal to the teacher-forced row at its
position. A stale row in the device's ring after a rejected draft (a bug found while
writing this) moved the logits by 0.39 to 0.53 of the largest, inside what a
rounding flip can do, and failed the self-consistency check at once.

**A partial chain** ([below](#a-partial-chain)): when the dense layers do not all fit the
device, deepseek_v4 chains the first N and the CPU's `target_batch` and `target_token` run
the rest from the streams the chain hands back (`v4c_forward` returns the layers it ran).
The fit counts each layer's matrices (fp8 as fmt 12 with its block scales, wo_a per output
group, the compressors' and the indexer's bf16, the mHC mixes in f32), its window ring, its
compressors' rings and 64 compressed rows and index keys, and its parameters; the fixed part
is the scratch of one 128-row block. The RAM plan asks for N
(`coli_v4_dense_device_decide`, which sets `dense_resident.device_layers`) before the expert
store opens, so only those layers' fp8 matrices and compressor projections leave RAM; a
chained layer is built whole in its placement before its host pages are given back. With
part of the model on the CPU the per-matrix path uploads nothing new (the head and the
routers stay on the CPU), and after a lost device only the device's layers are read per
forward.

**The default.** `COLI_VK_CHAIN_UNMEASURED`: off on an integrated GPU (`COLI_VK_CHAIN=1`
turns it on, `2` for prompts only), on a discrete GPU the rule above. No DeepSeek V4
checkpoint was run (the 780M box has none; the model is far past its disk), so speed is
not measured; the tests prove the tokens on the tiny fixtures.

```
[VK] deepseek_v4: dense chain off (an integrated GPU with the expert tier: not measured; COLI_VK_CHAIN=0 off, 1 on, 2 prompts only)
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

**A partial chain** ([below](#a-partial-chain)): kimi_k3 chains the first N layers and
`step_chunk_ex` runs the rest with `k3_layers_forward_range` from layer N. The handoff
moves each row's AttnRes state after layer N-1: the prefix, the block snapshots so far
and their count, written to the caller's buffers once every chunk is through (a lost
device reruns the forward from its untouched input). The chain's layers keep their KDA
state, convolution windows and MLA mirrors on the device as before (the "who holds the
newest" flag, its syncs and pushes and the watermarks cover them alone); the CPU's layers
keep theirs on the host, so after a lost device only the chain layers' KDA state is
rebuilt from the prefix record (layers 0..N-1 replayed). The fit runs once the device is
open and before the expert cache is sized: each layer's matrices in the form the loader
made of them (int4-g64, int8 rows or f32; the router and the KDA decay and beta
projections in f32), the KDA state and windows, an MLA layer's mirror and down rows at one
256-row chunk's positions, its share of the parameters; the fixed part is the scratch of
one chunk and the output mix, the tail the head. With every layer on the device and not
the head, the chain hands the CPU its final normalized rows and the CPU multiplies the
head. With the dense weights on the device only, only the chain's layers' matrices (and
the head with the tail) give back their host copies, after the setup, so a layer that
does not reach the device has nothing to read back; the expert cache's RAM plan counts
only those.

**The default**: kimi_k3 passes `COLI_VK_CHAIN_UNMEASURED`: off on an integrated GPU
(`COLI_VK_CHAIN=1` turns it on, `2` for prompts only), on a discrete GPU the rule above.
No Kimi K3 checkpoint was run (1.56 TB; the 780M box has none): speed is not measured,
and the tests prove the tokens on the tiny fixture, on Lavapipe and on the 780M:

```
[VK] kimi_k3: dense chain off (an integrated GPU with the expert tier: not measured; COLI_VK_CHAIN=0 off, 1 on, 2 prompts only)
```

### The routed experts in the frame (`vk_dmoe.h`)

A chain stops its frame at every MoE layer: the router's logits go down, the host routes,
the tier's batch and the CPU compute the experts, the sum comes back up. When the tier
holds every expert of a layer on the chain's device (the model fits: the warm start's
fill puts it there before the first token), that round trip buys nothing: the device
can route and compute the layer itself. For a step of up to 8 rows (decode, an MTP or
lookup verify) such a layer runs in the frame and the frame goes on; with every layer
so, a token is one frame.

- `vkc_route` (`chain_route.comp`): the router's softmax, sigmoid or sqrt(softplus), a
  correction bias for the pick (or, qwen36's, into the logits), group-limited top-k, the
  weights normalized in rank order and scaled; every engine's kind, checked against each
  engine's CPU routing in `tests/test_vk_chain.c` (ids exact, the lower expert on a tie).
- `vkc_moe` (`chain_moe.comp`): the experts by device address from a table the tier
  gives (`vkt_dev_table`, one entry per expert: its gate, up and down rows and scales),
  in three passes: gate and up with the activation per (row, rank), down per (row, rank),
  the rank-order sum. A row's 16-byte weight steps are spread over a power-of-two group
  of lanes with clustered adds, one compiled body per weight format, so every format of
  the tier (int8, int4, MXFP4, int3, f32, bf16, FP8, int8 in groups) takes it. Each row
  is computed alone: a verify's rows have a decode step's bits. Our container formats (1
  int8 rows, 2 int4 rows, 4 int4 per group) have pipelines of their own besides
  (`FMTC`, a specialization constant): each pass runs on its matrices' format's, so the
  other bodies are not compiled in. With every body in, Qwen3.6 int4-g64's layer took 112
  us on an RTX PRO 4000 (the gate and up pass alone 77); with fmt 4's alone, 73 (42), and
  the decode went from 56.6-57.7 to 67.2-67.4 tok/s. The layer's formats come from the
  tier's table (its first resident entry); `vkc_prewarm` makes and launches the three.
- Where D and I are whole 16-byte steps (multiples of 32), our formats' gate and up pass
  and down pass run on `chain_moe_gv.comp` instead, laid out as the dense decode GEMV
  (`chain_gemv.comp`) lays a matrix: the gate and up pass stages x once a workgroup and
  takes (expert, output) units across the row's experts, gate's and up's rows of a unit
  against the same x loads, row s's experts looked up once a workgroup; the down pass
  stages a pair's hidden row and takes its rows; about eight 16-byte steps a lane in both
  (chain_moe.comp spread a 2048-input row over 32 lanes, two steps each). Qwen3.6 int4-g64
  on an RTX PRO 4000: a layer 47-53 us (as the CUDA chain's 51), the gate and up pass 80
  -> 66 us and down 29 -> 17 in the chain profile, the decode 67.3-68.4 -> 76.5-78.3 tok/s,
  the teacher-forced loss the same (1.7576). `COLI_VK_MOE_GV=0` keeps chain_moe.comp's.
- int4 per group (4) with groups of whole steps: the loop of a lane's steps waits for each
  load in turn, which on a Radeon 8060S (RADV) left a Qwen3.6 layer at 160 us, 78 GB/s.
  Where a row is a whole number of steps a lane at the lanes chosen above (Qwen3.6's 2048
  and 512 inputs: eight), the pipeline built with `HQ` (a specialization constant, with the
  steps a lane in `NQ`) loads every step and scale of the lane first, gate's and up's
  together, then sums them; each step's product is rounded before it is added in both that
  path and the loop (`precise`), so the two give the same bits (tests/test_vk_chain checks
  it) and the tier's small experts, on the loop, stay the chain's. Gate and up as two
  loops instead of one interleaved took the loop alone 160 -> 119 us. Radeon 8060S,
  Qwen3.6's shapes (`COLI_VK_CHAIN_BENCH=moe`, K 8 of 256): one row 160 -> 100 us, two
  282 -> 166, three 385 -> 239; Qwen3.6-35B-A3B int4-g64 decode 45.0 -> 50.3 tok/s with the
  same text. `COLI_VK_MOE_GV_HQ=0` keeps the loop.
- The books follow at the frame's end, from the ids that come down with it: the history
  (`.coli_usage`), the HITS bitmap, the tier's heat (`vkt_dev_count`); the tier's line
  counts them as `in the dense chain: N routed experts with no host step`.

A prompt chunk (more than 8 rows) takes the frame the same way on qwen36 and qwen38, a
GGUF's experts and our own formats': `vkc_moe_rows` (`chain_moe_rows.comp`) counts the chunk's (row, expert) pairs by
expert on the device, plans each expert's tiles of 32 rows and writes the two passes'
dispatch sizes, then the block formats' grouped GEMMs (`qmatmul_grp_gate_up_bd.comp`,
`qmatmul_grp_bd.comp`, the tier's own; for int8 rows, int4 rows and int4 per gs,
`qmatmul_grp_q.comp` and its `GATE_UP` build, the same structure with `qmatmul.comp`'s
formats and the scales from the expert table) run gate and up, then down, over those
tiles with indirect dispatches, and a last pass sums each row's experts in rank order.
The tier's own grouped GEMM for our formats (`qmatmul_grp.comp`, the matrix units) needs
subgroups of 64, so on NVIDIA a prompt's int4 experts went one expert at a time through
the host's step. Where the device has cooperative matrices at subgroups of 64 or 32, our
formats' tiles run on the matrix units (`qmatmul_grp_qc.comp` and its `GATE_UP` build: the
same items and table, weights times their scales and the inputs rounded to f16, sums in
f32, the item's eight 16 x 16 accumulators shared out among the subgroups;
`COLI_VK_GRP_COOP=0` keeps the floats). A GGUF's experts take the same way
(`qmatmul_grp_bdc.comp` and its `GATE_UP` build: the block decoders write f16 into the
tiles), and so do its dense matrices for a prompt's rows (`qmatmul_gemm_bdc.comp`, 64
outputs x 64 rows a workgroup, `COLI_VK_CHAIN_GEMM=0` keeps the floats). On an RTX PRO
4000, Qwen3.6-35B-A3B, a 467-token prompt, two starts each: UD-Q4_K_M 694-726 -> 1300-1373
tok/s, UD-IQ2_XXS 733-757 -> 1403-1505 tok/s, the decode unchanged (66-68, 86-88); the
teacher-forced loss after a 600-token prompt 3.2141 -> 3.2211. With the tier's host pool on
(experts in system RAM the device reads over PCIe) a chunk's grouped GEMMs keep the floats,
whose reads over PCIe went faster: Qwen3.8 Next IQ2_XS with 19 GiB of experts in the pool, a
507-token prompt, 93-94 tok/s on the matrix units against 115-116 in floats; without the pool
the two were even. No row leaves
the device. A layer whose types have no grouped GEMM (a GGUF's IQ1_M layers) keeps the
host's step for that chunk; the rest of the forward stays in the frame. It needs the
grouped GEMMs' 34.8 KiB of shared memory (not Lavapipe's 32 KiB, where every layer keeps
the host's step) and turns off with `COLI_VK_CHAIN_GEMM_BD=0`. Its pipelines are made and
launched once, with no rows, before the tier fills the card (`vkc_prewarm`, from the file's
expert types): made after, they slowed every kernel of the frame, decode included, and
made before but first launched after, they still cost the decode 6 to 16%.

Both GEMMs have a wider tile where a prompt fills it. A GGUF's dense matrices take
`qmatmul_gemm_bdcw.comp` from 128 rows: 128 outputs x 128 rows a workgroup, every thread
decoding one of the 128 rows' two 32-weight steps, so each weight step serves twice the rows
and each input step twice the outputs (`COLI_VK_GEMM_WIDE=0` keeps the 64 tile). Its tables
keep four words, so the types whose decoders read an IQ grid from shared memory (IQ2_XXS,
IQ2_XS, IQ3_XXS, IQ3_S, IQ2_S) keep the 64 tile; a shared array of a size that is not whole
16 bytes misaligned the matrix loads after it and broke every type. RTX PRO 4000, one
4096 x 2048 matrix against 4096 rows: Q4_K 3.30 -> 1.97 ms (20.8 -> 34.9 TFLOP/s), Q5_K
3.53 -> 2.23, Q6_K 4.85 -> 2.39, Q8_0 4.71 -> 2.70. A chunk's experts take
`qmatmul_grp_bdcw.comp` (and its `GATE_UP` build) when they average 32 rows or more
(`COLI_VK_GRP_WIDE` sets that, `0` never): an item 64 rows of an expert by 128 decoded rows
(64 outputs of gate and 64 of up, or 128 of down), the items planned at those sizes by
`chain_moe_rows.comp`. With gate and up, a subgroup holds a gate tile and its up tile, so
silu(gate) * up is taken on the accumulators; a tile of whole rows goes to memory as it is,
the down pass's (rows to their pairs) and a partial one through a subgroup's patch. One
layer of Qwen3.6's experts in Q4_K (256 of 2048 x 512, top 8) on the RTX PRO 4000: 512 rows
2.90 -> 3.28 ms (kept on the narrow item), 1024 rows 4.40 -> 3.69, 4096 rows 13.8 -> 9.5. Both
pipelines are made and launched with the others before the tier fills the card. Prompt
reading, tok/s at 1K / 4K / 16K tokens, two runs each:

| | before | dense 128 tile | and the experts' wider item |
|---|---|---|---|
| Qwen3.6 UD-Q4_K_M | 1779-1801 / 1960-1979 / 1770-1785 | 2097 / 2344-2358 / 2090-2104 | 2095-2117 / 2621-2637 / 2305-2318 |
| Flint UD-Q3_K_XL | 1678-1692 / 1970-1976 / 1906-1907 | 1876-1895 / 2296-2310 / 2234 | 1873-1901 / 2520-2529 / 2445-2448 |

Measured on an RTX PRO 4000 Blackwell SFF (70 W, a virtual GPU whose speed moves from one
server start to the next), three runs a start. Qwen3.6-35B-A3B, a prompt of 467 tokens,
every expert on the card:

| | prefill before | prefill now | decode now |
|---|---|---|---|
| UD-Q4_K_M | 450 to 485 tok/s | 676 to 681 tok/s | 70.2 to 71.2 tok/s |
| UD-IQ2_XXS | 460 to 515 tok/s | 706 to 721 tok/s | 82.3 to 83.8 tok/s |
| int4-g64 (ours) | 308 to 313 tok/s | 740 to 750 tok/s | 57.7 to 60.4 tok/s |

The int4-g64 row's teacher-forced loss after a 600-token prompt (125 tokens scored,
`PPL=1`) is 3.2181 nats a token either way.

Measured again on the same card once the matrix units ran at subgroups of 32 and the
frame took the experts' grouped GEMMs, two series alternated, the same 467-token prompt and
every expert on the card: colibri's int4-g64 container 1796-1902 tok/s of prefill and
83.8-84.6 of decode, the GGUF Q4_K_M 1412-1556 and 73.7-81.9. Our container (22 GB) now leads
the GGUF (21 GB) in both phases.

The matrix units at subgroups of 32 then took the same prompt further, each step started
on its own server twice, three runs a start (prefill / decode):

| int4-g64 | prefill | decode |
|---|---|---|
| every GEMM in floats | 734 to 756 tok/s | 57.5 to 57.7 tok/s |
| the dense GEMMs on `chain_gemm.comp` | 865 to 889 tok/s | 56.9 to 60.5 tok/s |
| and the experts on `qmatmul_grp_qc.comp` | 1410 to 1477 tok/s | 57.2 to 60.3 tok/s |
| and the attention on `chain_attn_flash.comp` | 1521 to 1580 tok/s | 57.1 to 58.4 tok/s |

The loss with f16 operands: 3.2378 (dense GEMMs), 3.2308 (and the experts), 3.2266 (and the
attention), against 3.2181 in floats and 3.2340 on the CPU: a change of rounding moves it
by that much. OLMoE-1B-7B's geometry in int8 (random weights, a 1024-token prompt) went
from 1.3 s (the host's step) to 0.32 s.

Qwen3.8 Next IQ2_XS, a prompt of 507 tokens, 19.7 GiB of experts on the card and 15 GiB
in the host pool (prompt chunks of 151 rows), the chunk's experts on the host's step and
in the frame started one after the other, four pairs (prefill / decode):

| pair | host's step | in the frame |
|---|---|---|
| 1 | 66.5 / 25.8 tok/s | 92.2 / 36.7 tok/s |
| 2 | 80.9 / 37.0 tok/s | 92.8 / 36.3 tok/s |
| 3 | 73.7 / 32.8 tok/s | 85.6 / 32.8 tok/s |
| 4 | 85.6 / 39.2 tok/s | 90.3 / 36.2 tok/s |

The prefill gains in every pair (5 to 39%); the decode stays within what the card moves by
between starts. With 151-row chunks each chunk reads the host pool's experts over PCIe
again: larger chunks are the next lever there.

A layer the tier holds only part of keeps the host's step, in the same forward: the frame
ends there as before. The table follows the tier: an eviction or an upload changes its
epoch (`vkt_epoch`), and the next step writes the table again. It is off for PILOT (its
prefetch and the routing momentum it reads want every row's logits on the host),
`ROUTE_TRACE`, qwen36's cache routing and agreement meter, the CUDA tier or a qpack
container, the second device (the tier's experts are the primary's), and a device
without `bufferDeviceAddress` or clustered subgroup adds. `COLI_VK_CHAIN_MOE=0` turns it
off. Engines: olmoe, qwen36 (Qwen3.6, Qwen3-Coder), qwen38; the others' models do not
fit a device whole, where their layers keep the tier's step.

The device opens with `bufferDeviceAddress` wherever it has it now (before, only with
the matrix units, for the grouped GEMM); `COLI_VK_BDA=0` still opens it without.

**Hybrid steps: the host pool's experts on the CPU.** The tier's host pool holds the
experts the card has no room for in system RAM, which the device reads over PCIe: 10 to
11 GB/s on an RTX PRO 4000 at PCIe 4.0 x8, at any size (`COLI_VK_BENCH_HOST`). With
Qwen3.8 Next IQ2_XS on that card about 7% of a decode step's pairs and 11% of a 3-row
verify's come from there, and each layer that has one waits for it: skipping them
(wrong tokens, the right times) took decode from 38.5 to 50.1 tok/s and an MTP verify's
forward from 70 to about 45 ms. In a step of up to 8 rows (`vk_dmoe.h`), where the device
has timeline semaphores, the CPU now computes those pairs in place while the device
computes the ones it holds:
- frame A ends at the route, the input rows copied down, and signals the device's
  timeline;
- frame B runs the device's experts from a table where the host pool's are zero;
- frame C waits for the host's timeline and adds the CPU's sum;
- a worker thread takes the steps in order: it waits for frame A, computes the pairs as
  `ggml_quants.h` does a GGUF's matmuls (int8 activations, integer dots; the float
  reference with `GQ_EXACT=1`), from the host pool's rows as the file has them, writes
  their weighted sum and signals.

A round trip of the two timelines measured 7 us on the card. The host pool is cached
system RAM (it was write-combined, which made the CPU's reads 25 times slower; the
device reads both at the same speed). A prompt chunk keeps the full table and reads the
host pool over PCIe. A GGUF's block types take it, and colibri's int8 (a scale per row
or per group), int4 pairs (per row or per group), int3-g64, MXFP4 and FP8 formats, with
the CPU's kernels for them in `vk_tier.c` (int8 activations per 32 and integer dots, as
`ggml_quants.h`'s; FP8, and MXFP4 past 2048 groups a row, against the float row; the
float reference of every one with `GQ_EXACT=1`). The low-bit formats (fmt 200 to 203)
keep the device's reads. `COLI_VK_HYBRID=0` turns it off, `COLI_VK_HYBRID_THREADS` sizes the
worker's team (OpenMP's by default). The chain's line says how many steps and pairs it
took and the CPU's time. A share of the host pool's experts (`COLI_VK_HYBRID_PCIE`, by a
hash of layer and expert) can stay with the device over PCIe; it is 0 by default, every
one on the CPU. The CPU reads a GGUF's rows eight at a time (`ggml_quants.h`
`gq_rows_f8`: the rows' bytes transposed in registers into the R16 forms' codes, or the
grid types' rows with their signs on the activation, the R16 kernels' numerics) at its
memory's speed, faster than the device reads them over PCIe, and a step's few pairs split by
a hash leave the device the slower side on the steps that drew two. Warm 400-token
answers, the share at 0.3 against 0: Flint UD-Q3_K_XL with 7.4 GB of the card 44.6
against 49.4 tok/s (0.15 47.5, 0.45 40.8), with 12 GB 51.6 against 56.2; Qwen3.6-35B-A3B
Q4_K_M 47.7 against 55.5 and 71.3 against 74.4; Qwen3.8 IQ2_XS on the whole card 49.1
against 49.0. The worker sums the pairs in its own memory and copies the sum to the
device's buffer once: that buffer is written by the host and read by the device, memory
the CPU does not cache, and reading it back as the sum grew cost about 70 us a step of
Flint's three pairs (decode 39.6 against 44.6 tok/s at 7.4 GB).

Measured on the RTX PRO 4000 with an i5-13500 (Qwen3.8 IQ2_XS, the defaults, on and off
started one after the other): decode 42.2/42.5 against 40.4/41.0 tok/s with the team on
OpenMP's 14 threads, 44 to 45 with the regions' rows in chunks; the same greedy tokens
(160 of them) on and off. The CPU spends about 155 us a pair (IQ2_S gate and up 86 us,
Q2_0 down 28 us): the device's own experts take less, so a step waits for the CPU.

**The tier's own steps take the hybrid too.** Where a model does not fit the card whole
(GLM-5.3, Inkling), a layer's routed experts run in the tier's batch, outside the chain's
frame, and the device read the host pool's experts of that batch over the bus: 8 GB/s for GLM-5.3's (7.5 ms a layer of 8 against 0.25 ms from
VRAM, `COLI_VK_CHAIN_BENCH=xb` with `COLI_VK_BENCH_HOST=1`). In a step of up to 8 rows the
tier now leaves those of a GGUF's block types or of colibri's formats above out of the device's batch and the CPU computes
them in place at the join (`vkt_host_pairs`, the same code as the chain's worker), while the
device computes the ones it holds; the join hands their rows back as the device's, so no
engine changes. The same share stays with the device over PCIe, a prompt step keeps them
all on the device, and `COLI_VK_HYBRID=0` turns it off. DeepSeek V4's expert takes it with
the device's roundings (`hy_act_v4`, as `shaders/expert_act_v4.comp`: gate and up to bf16,
the clamped SwiGLU, times the route weight and to bf16, E4M3 per 128, then down), the weight
in the row as in the device's; the tier copies a CPU pair's input row at the issue, since
that engine frees its rounded rows as the issue returns. Kimi K3's SiTU stays on the device.
**Whether the hybrid pays is measured, not assumed** (`vk_hyb_gate.h`). On an RTX 3070 at
PCIe 4.0 x16 with Qwen3.6 int4 (2756 of 10240 experts on the card, about 1.4 host pool pairs
a decode step) the hybrid steps took a token from 18.5 ms to 33 to 50: the CPU's pair cost
300 to 500 us at one to twelve threads where the device read the same expert over the bus
in about 130, and a step's three frames and timeline round trip came on top. So the chain
times its decode forwards with the device's reads for three forwards, then with the hybrid
steps for three (the first forward of each warms it), keeps the faster (the hybrid where it
is not slower by a twentieth) and measures again after 1024 forwards; the run's line says
what it measured and what it kept. `COLI_VK_HYBRID_GATE=0` keeps the steps on, as before.
The same card had come up without timeline semaphores at all when the build's headers
predated the cooperative matrix (the feature was asked for inside that extension's block),
so the steps never ran there; every device is asked now.

The run's line adds `hybrid: N steps,
P host pool pairs computed by the CPU in place (T ms, U us a pair)`. Measured on the RTX PRO
4000 with the i5-13500, each server on a copy of one history, 128 tokens, a warm-up and a run:

| model | decode, `COLI_VK_HYBRID=0` | decode, the default | the CPU a pair |
|---|---|---|---|
| GLM-5.3 Flash UD-IQ2_XXS (95 GB) | 3.22, 3.31 tok/s | 3.53, 4.97 tok/s | 508 us |
| the same, 199 tokens after a 59-token prompt, two runs | 3.22, 3.20 tok/s | 5.06, 5.24 tok/s | 494 us |
| Inkling-Small UD-IQ2_XXS (77 GB) | 11.10, 10.54 tok/s | 15.06, 15.75 tok/s | 862 us |
| GLM-5.3 int4-g64, colibri's (391 GB), a warm-up and three runs, twice in each order | 0.50 to 0.63 tok/s (0.56 on average) | 0.59 to 0.66 tok/s (0.62) | 852 to 882 us |
| DeepSeek V4 Flash UD-IQ1_S (82.5 GB), a warm-up and four runs, twice in each order | 6.62 to 6.88 tok/s | 6.62 to 6.94 tok/s | 382 to 385 us |

The prompts are unchanged (GLM-5.3 int4-g64: 1.4 to 1.7 tok/s off, 1.6 to 1.8 on, the model three times the RAM, so the disk decides); GLM-5.3 Flash's warm-up still overlaps the warm start's uploads. With GLM-5.3 int4-g64 the device's time over the runs went from 277 to 108 s. DeepSeek V4 Flash's host pool is 4 GiB of its 76 GB of experts, its pairs one or two a step: the same either way. GLM-5.3
gave the same greedy text (40 tokens) on and off, in 17.7 s against 26.7 s; with the device's
share of the pool at 0, 0.3 and 0.5 its 199-token runs decoded at 5.01/4.96, 5.06/5.24 and
3.92/4.02 tok/s. Alone (the bench
above), the CPU takes 300 to 340 us a pair of GLM-5.3's at 12 to 20 threads, the device 700
to 940 over the bus.

**A prompt chunk's host pool experts in VRAM.** A prompt chunk routes nearly every expert
of a layer, and the grouped GEMMs of a GGUF's block types read the host pool in place at
about 4 GB/s. Before a chunk's layer runs, its host pool experts of block types are now
copied into a staging buffer in VRAM (`chain_fetch.comp`: regions read by address, 16-byte
loads, several in flight a lane), and the layer's GEMMs read them there:
- the room comes out of the tier's budget when the host pool will hold experts (a layer's
  share of them and a quarter more; the tier's line says how much), and the prompt chunks'
  rows are sized without it. Allocated beside a full budget, the buffer took the card to
  its last memory, and the driver moved other allocations out: decode after a prompt lost
  4 to 15%;
- between prompt chunks the buffer holds the host pool's hottest experts of block types
  (the tier's own order), as more of the device's: the decode's tables point there and the
  CPU's hybrid share leaves them, so the decode gets the room back;
- colibri's formats keep reading the pool in place: their GEMMs read it near the bus's
  rate, and a copy before them only added its time (Qwen3.6 int4 with 9 GiB in the pool,
  a 507-token prompt: 513 against 394 tok/s).

Measured on the RTX PRO 4000 (PCIe 4.0 x8), a 507-token prompt, servers started one after
the other:

| model | host pool | prefill, staging off | prefill, staging on |
|---|---|---|---|
| Qwen3.8 IQ2_S/Q2_0, the default budget | 19 GiB | 114 to 119 tok/s | 143 to 150 tok/s |
| Qwen3.6 Q4_K_M, `COLI_VK_TIER_GB=8` | 12 GiB | 279 to 282 tok/s | 333 to 338 tok/s |

The decode's tok/s in such a bench moves with the text: the prompt on the matrix units
continues differently from the floats' it took over PCIe, and the continuation routes to
other experts. With the same tokens (Qwen3.8, a 2000-token prompt then 200 single steps,
teacher-forced): 31.4 s against 24.0 s, the CPU's host pool pairs 31920 against 23642, the
loss 1.4458 against 1.4274. `COLI_VK_STAGE=0` turns it off (the tier keeps its room too).

**The integer form.** A GGUF's experts run, where the device accelerates packed 4 x 8
integer dot products (`VK_KHR_shader_integer_dot_product`,
`integerDotProduct4x8BitPackedSignedAccelerated`), in the CPU's integer form
(`chain_moe_i8.spv`, `bdstep_i8.glsl`), as `ggml_quants.h` computes them by default:
- each input row (x for gate and up, the activation for down) is quantized to int8 in
  blocks of 32 as `gq_quant_act_ref` does it (d = amax / 127, codes rounded to nearest
  even and clamped to +-127, their sum per 16), in shared memory instead of the float row;
- each type's weights are the integers and scales of `gq_unpack_row` (a scale and a min
  per 16), and a group's sum `dx * (sw * sum(w8 * xq) - mw * xs)` is formed with exact
  integer inner sums (`dotPacked4x8EXT`).

Against the CPU's integer form, `tests/test_vk_chain`'s `moe i8` cases agree to about
2e-7 for every type it decodes (Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ1_S, IQ1_M, IQ2_XXS,
IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS, Q2_0), where the float form differs by about 1e-2: on the device the experts now
give what the CPU gives. Both of a layer's formats must have the form (else the float
pipeline runs that layer); a prompt chunk's grouped GEMMs keep their float sums.
`COLI_VK_IDOT=0` leaves the extension off; a device that supports it without
acceleration (Lavapipe) keeps the float form unless a test asks (`vkc_moe_set_idot`).
The chain's line counts the layer steps that took it (`N in the integer form`).

Both block formats' pipelines (the float form's and the integer form's) are made for a
layer's pair of types (`chain_moe.comp`'s `BTG` and `BTD`), so each holds those two
decoders alone: Qwen3.8's IQ2_S/Q2_0 decode 50.0-50.8 -> 51.1-51.9 tok/s on the RTX PRO
4000, and a software device no longer builds every type's decoders at a layer's first
launch (minutes on Lavapipe). The engines' prewarm makes one a pair of the file's gate/up
and down types (every type's past 16 pairs); a step with no known types keeps the
pipeline of every type.

Measured on the RTX PRO 4000 (`COLI_VK_CHAIN_BENCH=moe`, the float form with
`COLI_VK_IDOT=0`), a layer's experts in microseconds:

| experts (gate+up / down) | D | 1 row | 2 rows | 3 rows |
|---|---|---|---|---|
| IQ2_S / Q2_0 | 2560 | 98.6 -> 67.2 | 170 -> 118 | 244 -> 178 |
| IQ2_XXS / Q6_K | 2048 | 65.8 -> 60.4 | 117 -> 105 | 177 -> 143 |
| Q4_K / Q6_K | 2048 | 75.6 -> 69.3 | 128 -> 120 | 193 -> 165 |
| IQ4_XS / IQ4_XS | 2560 | 106 -> 99.6 | 187 -> 180 | 268 -> 283 |

A GGUF's dense GEMV (`chain_gemv_bd_i8.spv`, `chain_gemv_bd.comp` with `-DIDOT`) takes the
same form there: up to four rows of x (a verify's) quantized into shared memory, and each
32-weight step decoded once for all of them (`bsi_step` loops over `BSI_NR` rows; the
experts' kernel is its one-row case). The rows' codes take a quarter of the floats' room,
so a verify's rows no longer fill the shared memory, and the integer products cost less
than the conversions. IQ4_XS and IQ4_NL look their values up and keep the float form for
one row (44.2 us against 46.8). The codes are the CPU's to the bit: a Vulkan division
may be 2.5 ulp off and an fma need not be fused (Lavapipe's are not), and either moved a
code at a tie now and then, so `bsi_div` makes `127 / amax` and `amax / 127` exact with
the integer remainder of the mantissas' quotient. `tests/test_vk_chain`'s `matmul_bd i8`
cases, every type at 1, 3 and 4 rows, agree with `gq_matmul`'s integer path to under
2e-5, on the RTX and on Lavapipe. A GEMV of 10240 x 2560 in microseconds (float form ->
integer form):

| type | 1 row | 2 rows | 3 rows | 4 rows |
|---|---|---|---|---|
| IQ2_S | 52.0 -> 29.9 | 67.2 -> 39.7 | 87.3 -> 46.2 | 232.7 -> 58.3 |
| IQ2_XXS | 48.1 -> 25.4 | 60.8 -> 36.8 | 81.7 -> 46.4 | 232.2 -> 54.0 |
| Q2_0 | 40.0 -> 29.0 | 51.6 -> 32.8 | 72.7 -> 39.8 | 217.8 -> 47.2 |
| Q4_K | 46.9 -> 42.2 | 62.5 -> 45.0 | 82.1 -> 50.2 | 149.1 -> 59.0 |
| Q6_K | 68.1 -> 63.1 | 90.0 -> 67.3 | 113.2 -> 72.1 | 262.4 -> 81.6 |
| Q8_0 | 74.6 -> 73.2 | 87.0 -> 75.6 | 104.8 -> 78.9 | 272.5 -> 98.9 |
| IQ4_XS | 44.5 (kept) | 59.7 -> 51.2 | 80.8 -> 58.0 | 231.1 -> 66.2 |

**Staging sizes.** `chain_moe.comp` stages a row of x in shared memory, and so does
`chain_gemv_bd.comp` (a GGUF's dense GEMV). Both pipelines used to be built at the
device's largest staging (48 KiB on NVIDIA, with bdstep.glsl's 8.5 KiB of tables), which
left two workgroups an SM. Now each is made per size: the smallest of 256 to 4096 vec4s
that holds its rows (`kc_stage` in `vk_chain.c`), so four workgroups fit an SM at
Qwen3.8's and Qwen3.6's widths. `vkc_prewarm` gets the widths from the engine (each
matrix's input width, the experts' D and I) and makes and launches those sizes before the
tier fills the card.

Measured on the RTX PRO 4000 (`tests/test_vk_chain`'s `COLI_VK_CHAIN_BENCH=moe`, one binary
with the staging forced to the old size and to the new), a layer's experts in
microseconds (K 10 of 256 at D 2560 I 640, K 8 of 256 at D 2048 I 512):

| experts (gate+up / down) | D | 1 row | 2 rows | 3 rows |
|---|---|---|---|---|
| IQ2_S / Q2_0 | 2560 | 112 -> 96 | 207 -> 167 | 311 -> 243 |
| Q4_K / Q6_K | 2560 | 146 -> 119 | 253 -> 222 | 397 -> 325 |
| IQ2_XXS / Q6_K | 2048 | 77 -> 66 | 135 -> 119 | 197 -> 178 |
| Q8_0 / Q8_0 | 2048 | 103 -> 98 | 194 -> 169 | 292 -> 284 |

The dense GEMV takes the small sizes only for one row of IQ2_XXS or IQ2_S, whose decoders
wait on their grid reads (10240 x 2560: 61.2 -> 48.4 and 66.4 -> 52.0 us). The other
formats read at the memory's rate and lose with more workgroups an SM (IQ4_XS and IQ4_NL
3 to 4% for one row, Q8_0 40% for two), so they keep the largest.

In the engine (decode tok/s, a 467-token prompt and 127 tokens, the old sizes and the new
started one after the other, two pairs, three runs each):

| model | old sizes | new sizes |
|---|---|---|
| Qwen3.6 UD-IQ2_XXS | 83.6, 82.0 | 87.0, 86.7 |
| Qwen3.6 UD-Q4_K_M | 70.6, 71.1 | 71.9, 71.4 |
| Qwen3.6 int4-g64 (colibri's container) | 61.5, 61.7 | 61.6, 62.1 |
| Qwen3.8 Next IQ2_XS (507-token prompt) | 35.1, 38.1 | 41.4, 38.2 |

The prefill did not move. On Qwen3.8 the card's own spread between starts is as large as
the change.

Measured on a Radeon 780M (RADV, OLMoE-1B-7B int8, every expert on the device, a
25-token prompt and 100 decode steps, warm, the same binary):

| | decode |
|---|---|
| the routed experts in the frame | 33.2 tok/s (166 frames for 101 tokens) |
| the host's routing and the tier's batch (`COLI_VK_CHAIN_MOE=0`) | 28.2 to 30.1 tok/s (1762 frames) |
| the tier alone (`COLI_VK_CHAIN=0`) | 19.1 tok/s |
| the CPU | 22.9 tok/s |

The experts take 13.0 ms of device time a token in the frame against 14.4 ms in the
tier's grouped GEMV; the rest is the dense chain's GEMVs (14.4 ms). The 101 tokens are
the CPU's in all three. A first version of the shader (one 32-bit word a lane, a row a
subgroup, the format chosen per word) ran at 13.6 tok/s: the per-format bodies and the
16-byte steps are what made it faster than the host's step.

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
| deepseek_v41 | MQA over a window ring and compressed rows, a sink, an indexer with candidate blocks | on the chain ([above](#deepseek-v41-flash-and-deepseek-v4-on-the-chain)), with the [DeepSeek ops](#deepseek-v41-flash-and-deepseek-v4s-attention-vkc_dsv4) |
| deepseek_v4 | MQA over a window ring and compressed (CSA, HCA) rows, mHC, bf16 and E4M3 roundings | on the chain ([above](#deepseek-v4-on-the-chain)), with the DeepSeek ops and their rounding modes |
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

The indexers' scores run on a grid (modes 3 and 4): a workgroup a block of 256 / IH
positions (or pools) and a row, a thread a (position, head) for the head's dot, then a
thread a position joining the heads, each sum in the order mode 0 or 2 makes it, so the
same bits; the selection then reads them. A row's one workgroup computing every
position's IH x ID products took a decode step on an RTX PRO 4000 (32 index heads of 128,
top 2048) 16.2 ms a layer at 8000 positions and 63.8 at 32000; now 345 us and 1.25 ms,
the k-pooled form (pools of 8) 2.19 ms -> 86 us at 8000 positions. The selection's
counts are subgroup sums (integers: the same totals in any order).
DeepSeek V4/V4.1's indexer scores (`chain_dsv4`, modes 3 and 9) take the same grid
(mode 10, the queries in shared memory or read where they are as modes 3 and 9 do): its 64
heads of 128 over 2000 compressed columns 8.1 ms -> 141 us, over 8000 31.6 ms -> 470 us.
`COLI_VK_DSA_GRID=0` keeps every row's scores in its own workgroup.

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

### DeepSeek V4.1 Flash and DeepSeek V4's attention (`vkc_dsv4`)

DeepSeek's attention is not the absorbed MLA above: it is MQA over one KV row per
position, the same row key and value, read from a sliding window of raw rows and from
compressed rows (a compressor pools `ratio` positions into one) that a DSA indexer picks
per query, with an attention sink. Its ops are in `vk_chain.h` (`vkc_dsv4_*`), one shader
of their own (`chain_dsv4.comp`), optional like the MLA ones:

| Op | What it does |
|---|---|
| `vkc_dsv4_attn` | the sparse attention of `sparse_attn.h`: per row a list of window rows, compressed rows and skipped entries; the sink in the denominator only, the value sum and the denominator in list order; optionally DeepSeek V4's roundings (the weights and the output to bf16) |
| `vkc_dsv4_rope` | RoPE on interleaved pairs in place from a host table, forward or inverse (the attention output's un-rotation) |
| `vkc_dsv4_compress` | the compressor's rolling group: each row's kv and score rows into the ring slot of its position, the per-channel softmax pooling when a group completes; DeepSeek V4's overlapping form (two halves, a position bias per slot) too |
| `vkc_dsv4_score` | the indexer's scores: the relu-gated, head-weighted dot of each reachable compressed row, a candidate mask, -inf past the row's reach |
| `vkc_dsv4_cand` | DeepSeek V4.1's candidate blocks: each block's best score, the newest block pinned, the best blocks kept whole |
| `vkc_dsv4_topk` | the top-k of the CPU's selection (ties to the lower column), in column order (V4.1) or by rank (V4), padded with skipped entries |
| `vkc_dsv4_engram` | DeepSeek V4.1's engram gate on the residual streams |

`make vk-chain-check VK=1` checks them against references transcribed from
`deepseek_v41.c` (and V4's rounding variant): the attention over lists of window and
compressed rows with skipped entries and a row with none (within 1.1e-6 of the largest
output on Lavapipe), RoPE both ways, the compressor over three calls with its ring
carried (ratios 1 to 4, the overlapping form with its bias, within 2e-6), the scores,
the candidate mask and the top-k list slot for slot on scores exact in float on both
sides, ties included, with and without a mask, and the engram gate (within 2e-6).

## Big prompt chunks and expert streaming

A long prompt used to cross the chain in chunks of 512 rows, and a chunk's MoE step reached
the tier in blocks of 32 to 128 rows (the engines' own prefill blocks): each block re-read
the experts it routed, and the tier could take only the experts it already held. The CPU
computed the rest, on a long prompt most of them, for every block. Two changes together:

- **Big chunks.** The chain's chunk now comes from the device's free memory, up to 8192
  rows, and the engines hand the tier a chunk's whole MoE step at once. One pass over a
  layer's experts then serves every row of the chunk.
- **Expert streaming.** In a step that big, a cold expert (not resident) is routed to
  dozens or hundreds of rows. The tier uploads it into a staging slot on the device and
  runs its rows as a GEMM there, instead of leaving it to the CPU. The slots are reused,
  and the resident set and its LFRU are not touched.

### The chunk (`vkc_chunk_rows`)

`COLI_VK_CHAIN_ROWS=n` sets the chunk as before. Unset (or `auto`), each engine counts
its chain's scratch per row from its own reservations (a counting pass over the same
`vkc_reserve` calls), adds the simultaneous tier and CPU expert outputs and the
engines' CPU gather/activation workspaces, and takes the most rows, up to
`COLI_VK_CHAIN_ROWS_MAX` (8192), whose buffers fit half of the device memory free
at its first forward. Free memory is
`VK_EXT_memory_budget`'s budget less its use (without the extension, a quarter of the
device-local heap). The CPU buffers must fit too: available physical RAM (and
Windows commit headroom) also caps the budget. Qwen3.8 includes its simultaneous
CPU expert input, gate, up and output buffers even when all experts run on the
device. Chunks of at least 256 rows round down to a multiple of 256; tighter
budgets keep the rows that fit, down to one. The decision
is taken once per engine, after the tier has filled, so on a discrete card the room the
tier leaves (`COLI_VK_TIER_RESERVE_GB`) is what the chunk gets. One line says what it
took:

```
[VK] qwen36 chain: prompt chunks of up to 8192 rows (325 KiB a row; 7.30 GiB free on the device or in RAM, half of it at most; COLI_VK_CHAIN_ROWS sets it, COLI_VK_CHAIN_ROWS_MAX caps it at 8192)
```

On qwen38 the tier also leaves what the chain allocates at its first forward: the KV
mirror whole at the context's capacity (at most a quarter of the device's free memory;
past that `vk_kvsplit.h` splits it anyway) and a chunk of 512 rows. The tier's line says
how much (`3.83 GiB left for the dense chain's KV mirror and a prompt chunk`). Before,
the tier took all but `COLI_VK_TIER_RESERVE_GB`, the chain then split its KV mirror (the
attention took twice the time) and its chunks were 151 rows. Qwen3.8 Next IQ2_XS on an
RTX PRO 4000, 64K of context, a 507-token prompt, the defaults, two pairs started one
after the other:

| | tier | prefill | decode |
|---|---|---|---|
| the tier takes the device | 19.78 GiB, KV split, chunks of 151 rows | 90.9, 88.8 tok/s | 36.8, 36.7 tok/s |
| it leaves the chain's KV and a chunk | 15.95 GiB, KV whole, chunks of 512 rows | 105.8, 111.3 tok/s | 39.4, 43.1 tok/s |

A sweep of `COLI_VK_TIER_GB` on the same card put the best decode at 16 GiB (40.4 tok/s;
18 GiB 37.5, 14 GiB 37.8, 10 GiB 31.8): smaller, more experts go over PCIe from the host
pool; larger, the chain's KV no longer fits.

The engines whose prefill blocks the prompt before the chain hand it the chain's chunk
when the chunk comes from the budget: MiMo instead of `MIMO_CHUNK`'s 64 rows, GLM-5.3
instead of `GLM53_PREFILL_CHUNK`'s 128, Kimi K3 instead of `K3_CHUNK`'s 32, DeepSeek V4
instead of its 128-row CPU block (its chained MoE step, the expert union, takes any
batch). Each of those variables, when set, keeps its blocks, as before. DeepSeek V4.1
and V4 grow their window rings to the window plus the chunk. The indexers' score
scratch keeps its cap (32M floats a forward on DeepSeek V4.1 and GLM-5.2).

**What did not scale, and the fixes.** Every chain op takes up to 65535 rows; the
dispatches over rows stay within the device limits at 8192. Two things did not scale:

- **Attention reads.** `chain_attn.comp` reads a row's positions once per (head, row), so
  a chunk of S rows read its KV cache S x H times: 1.2 s a layer for 2048 rows of
  Qwen3.6 on the 780M. From `COLI_VK_ATTN_BLOCK` rows (16) `vkc_attn` takes
  `chain_attnb.comp` instead. A workgroup takes the query heads of one KV head for a
  block of rows (up to 32 (head, row) pairs). It walks the positions in tiles, each
  tile's K and V rows staged in shared memory once for every pair: tiles of 4 to 16
  positions, as the device's shared memory allows. It takes every form `chain_attn`
  takes (a window, a ring, position-major rows, a sink, selection lists). Below 16 rows,
  so for decode and MTP verifies, `chain_attn` runs as before with the same bits.
- **Submission length.** The attention's work grows with rows x positions, and one
  submission of a big chunk ran for seconds. On the 780M a frame of a 5632-row chunk at
  8192 positions was reset by the driver (`ring comp_1.1.0 timeout` in the kernel log,
  the device lost). Past `COLI_VK_ATTN_SLICE` (2^32 rows x positions x heads x head
  dim), `vkc_attn`, `vkc_attn_w`, `vkc_mla_core` and `vkc_relattn` record their rows in
  slices. Each slice ends its frame, submitted and not waited for, and the next starts a
  new one, which its first barrier orders after the slice. A slice starts on a multiple
  of the blocked attention's rows, so no row's arithmetic changes.

### Streaming (`vk_tier.c`)

An engine that gives the tier `VktConfig.load` and `.release` (and `.load_batch` when it
reads experts in parallel) streams. Those hooks hand the tier an expert's bytes as the
CPU path would get them: through the engine's RAM cache, else from disk. Every MoE
engine has them. For a step of at least `COLI_VK_TIER_GEMM_ROWS` rows (16),
`vkt_step_rows` gives the engine the whole step instead of its usual block, and
`vkt_issue` runs it as follows:

- **The rule**, per cold expert of the step. Stream it when it has at least R rows:
  R = max(G, ceil(upload / cpu_row)). G is the rows from which the device takes the
  tiled GEMM (16; fewer rows would run the per-row GEMV). `upload` is the time one expert
  takes to reach a slot, its bytes at the bandwidth the uploads so far measured.
  `cpu_row` is the time the CPU spent per (row, expert) pair it computed in the prefill
  steps so far, its share of a step between issue and join, loads included. Below R the
  CPU keeps the expert: its rows cost the CPU less than the upload. Until both are
  measured, R = G. `COLI_VK_TIER_STREAM_ROWS=n` fixes R at n. One line per forward says
  what the rule gave:

  ```
  [VK] tier qwen36 stream: a step of 2560 rows; cold experts with 16 rows or more go to the device (an upload of 1.7 MiB at 8.2 GB/s takes 0.217 ms, the CPU 0.2280 ms a row; the GEMM from 16 rows); layer 0: 245 resident, 11 streamed, 0 kept on the CPU (0 rows)
  ```

- **Sub-batches** (backend: `coli_vk_xb_sub_*`). The step's device work runs as a
  sequence of bounded batches: the resident experts first, then the streamed ones, with
  at most half the slots and `COLI_VK_TIER_STREAM_HALF` rows a batch. That cap keeps a
  batch's x rows within 32 MiB. Before allocating, the backend can reduce the batch
  further to fit half the free budget of every affected memory heap and available
  RAM. This calculation includes both halves, gate/up/hidden/output buffers, the
  doubled descriptor windows and allocation growth; existing buffers count only
  their additional bytes. A tight budget therefore reduces rows before an allocation
  failure could disable the tier. An expert with more rows than a batch holds is cut
  into parts.
  Batch k goes to half k % 2 of the scratch, so while the device computes one, the engine
  thread loads and uploads the experts of the next (double buffering), and a batch's
  outputs are copied out when it is joined. The recording is the single batch's,
  through the same `xb_record_submit`.
- **The staging slots.** `COLI_VK_TIER_STREAM_SLOTS` (64) experts' worth of tensors, made
  at the first step that streams and filled again for each expert they carry
  (`coli_vk_tensor_refill`). On mapped memory that is the tensor's own mapping. With
  staged uploads it is a host image that `coli_vk_tensor_commit` copies over. The slots
  come out of the tier's budget: the residents give up what the budget cannot hold
  beside them (64 slots of Qwen3.6's 7200 residents). The conversion into a slot is the
  tier's own byte work (`convert`). A group the engine loaded together converts in
  parallel, one expert per thread; a lone one converts with its rows split over the
  threads (`COLI_VK_TIER_STREAM_PAR=0` keeps it on the engine thread).
- **The host pool's experts through the slots too.** A resident of the host pool is not
  cold, so a big step's GEMMs read it in place over PCIe, a GGUF's block types at about 4
  GB/s; on GLM-5.3, where a layer's step routes 100 to 140 of them, those reads were most
  of the device's 115.7 s over three prompts. With the host pool on, a big step's host pool experts of
  block types now go through the slots like the streamed ones: each is copied from the
  pool into a slot at the head of the batch that first reads it, a barrier between the
  copies and its dispatches (`coli_vk_xb_sub_copy`: the slot takes the expert's format
  first; with mapped memory the CPU copies through the mapping), so the bytes cross the
  bus once, at its rate, and the engine thread goes on to the step's cold experts. A slot
  that does not take one leaves it to be read in place, never to the CPU. `COLI_VK_TIER_POOL_COPY=0` reads them in place.
  Measured on the RTX PRO 4000 (PCIe 4.0 x8), a 700-token prompt, a warm-up and a run,
  each server on a copy of one history:

  | model | prefill, `COLI_VK_TIER_POOL_COPY=0` | prefill, the default | the copies |
  |---|---|---|---|
  | GLM-5.3 Flash UD-IQ2_XXS (95 GB) | 11.9, 12.3 tok/s | 17.9, 18.2 tok/s | 110 GiB a prompt |
  | Inkling-Small UD-IQ2_XXS (77 GB) | 65.9, 70.2 tok/s | 97.2, 111.1 tok/s | 17 GiB a prompt |
  | DeepSeek V4 Flash 0731 UD-IQ2_XXS (85 GB) | 19.5, 23.4 tok/s | 22.4, 23.3 tok/s | 4.5 GiB a prompt |

  Copied by the copy engine and waited for, before each batch, the same bytes moved at 12.3
  to 12.5 GB/s and GLM-5.3's prompt ran at 15.2 and 17.4 tok/s, Inkling's at 94.0 and 107.6:
  the engine thread then waited for the copies instead of reading the step's cold experts.
  On GLM-5.3 the bus is now the floor (a layer step copies about 0.94 GB); DeepSeek V4's CPU share
  of a step (the cold experts its store reads) was the longer side already. GLM-5.3's greedy text after a long prompt is the same with and without (40 tokens in 39.3 s against 46.9);
  Inkling's forks at a near tie after 36 words ("beautification" against "initiatives"):
  a pool expert's rows now run in the slots' batches, where the backend may pick another
  of its GEMM routes for them than in the residents' batch, as for a streamed expert.
- **A pool expert of a row or two stays on the CPU.** A prompt step's host pool experts of
  block types with at most `COLI_VK_HYBRID_BIG_ROWS` rows (2) are computed by the CPU in
  place at the join (`vkt_host_pairs`, as the decode's hybrid steps), while the device
  computes the step's batches, instead of crossing the bus for a row or two. An expert's
  pairs go together: its rows are read once for all of them and each pair keeps the bits
  it has on its own (the kernels of `ggml_quants.h` take each activation row apart). The
  tier's line adds `, and S prompt steps' P pairs` to its hybrid part. Measured on the RTX
  PRO 4000, MiMo-V2.6-Flash Q2_K_S (106 GB), a 464-token prompt, three runs each on a copy
  of one history:

  | `COLI_VK_HYBRID_BIG_ROWS` | prefill | the device's time | the CPU waiting for it | the prompt's pairs on the CPU |
  |---|---|---|---|---|
  | 0 | 36.9-37.8 tok/s | 27.9 s | 25.3 s | 0 |
  | 2 (the default) | 38.5-39.4 tok/s | 23.7 s | 21.2 s | 6756 |
  | 4 | 38.3-39.1 tok/s | 21.2 s | 18.8 s | 16145 |
  | 8 | 37.0-37.8 tok/s | 18.2 s | 16.0 s | 37030 |
  | 16 | 28.3-29.4 tok/s | 13.8 s | 11.6 s | 102413 |

  Past 4 rows the CPU becomes the longer side. Grouped, a pair costs 290-380 us against
  410 for a decode step's (each pair its own rows). On GLM-5.2 UD-IQ1_S, whose prompt the
  CPU's cold experts hold (230 s of CPU against 26 s of device time), 0, 1, 2 and 4 rows
  gave the same 2.9-3.5 tok/s.
- **Nothing pollutes the resident set.** A streamed expert is not offered for promotion:
  a prompt's whole working set passes through the slots and would otherwise flush the
  residents. Its routings count as heat, as every routing does, and the slots are free
  again when their batch is joined.
- **Prefetch while attention runs.** The routing of a layer is known only after that
  layer's attention and router, in every engine here: the router reads the post-attention
  residual, so neither layer L's nor layer L+1's experts are known while L's attention
  runs. What is known is a prediction. The chain engines submit the frame that computes
  a layer's routing without waiting, call `vkt_stream_prefetch(layer, rows)`, then wait.
  The tier fills its free slots with that layer's likely streamed experts while the
  device works: the routing of the previous big step at the same layer when there was
  one (a long prompt's earlier chunk), else the history's heat. A prediction that misses
  costs an upload; an expert it does not cover is streamed after the routing as usual.
- **Device work.** Streamed and resident experts use the tiled GEMM. From 64 rows
  of one expert, devices supporting `VK_KHR_cooperative_matrix` also use its
  cooperative path for int8, int4, int3, MXFP4 and FP8 projections with compatible
  group sizes. Eligibility is checked separately for gate, up and down; other
  formats retain the FP32 GEMM. Activation scales are computed on the device,
  including the intermediate hidden rows, without a CPU readback. The hi/lo
  activation split and FP32 accumulators follow the per-matrix path. Decode and
  small draft-verification batches retain the GEMV. Integer weights use the
  narrower tile; FP8 and MXFP4 retain the FP32 GEMM when a wider cooperative
  tile would process extra padded rows. `COLI_VK_TIER_COOP=0` disables
  this path and `COLI_VK_TIER_COOP_ROWS` changes its row threshold. The tier's
  `cooperative matmuls` counter counts projections actually submitted.

An isolated Radeon 780M/RADV expert benchmark (D=2560, intermediate=640,
64 rows, two warmups and the median of nine alternating issue/join measurements)
measured 1.410→1.153 ms for int8, 0.793→0.644 ms for int4-g64, and
0.965→0.833 ms for FP8. These are expert-batch times, not model tokens/second.
The correctness harness exercises mixed projection formats, incomplete tiles,
zero and widely scaled activations, streaming and DeepSeek V4's rounding path.
Formats or shapes that do not qualify retain the existing kernels.

The run's line counts what streamed:

```
[VK] tier qwen36 run: ... | stream: 40 steps, 2188 cold experts (3.61 GiB, 10.72 GB/s) and 796154 rows streamed in 742 sub-batches, prefetched 0 (0 used), 929 cold experts (5548 rows) kept on the CPU
```

and, with the host pool's experts copied into the slots, `, N host pool experts copied into
the slots (S, R GB/s)` after it. The forward's stream line says how many of a layer's streamed
experts came from the host pool (`135 streamed (135 of them copied from the host pool)`).

| Engine | Its loads (`load` / `load_batch`) | The step the tier sees |
|---|---|---|
| qwen36 | `expert_hold` (and the int8 copy for the int8-slot kernel), one at a time | `moe_vk_run`'s block becomes the chunk |
| qwen38 | the layer cache, `q38_expert_get_batch` for a group | `q38_moe_prefill`'s rows the chunk |
| olmoe | `expert_get` / `expert_put` | `moe_vk_run`'s block the chunk |
| inkling | the slot cache, a group's misses filled in parallel | `ink_vk_moe`'s block the chunk |
| mimo | `experts_ensure` for a group (parallel reads) | the chain's chunk instead of `MIMO_CHUNK` |
| colibri | pins, the LRU, the misses read into the working set in parallel and promoted after the copy | `moe_vk`'s block the chunk |
| glm53 | `expert_block_read` for a group | the chain's chunk instead of `GLM53_PREFILL_CHUNK` |
| kimi_k3 | the layer cache, the misses read into the working set in parallel and promoted after the copy | the chain's chunk instead of `K3_CHUNK` |
| deepseek_v41 | `expert_slots_at` for a group | the backbone's MoE takes the chunk (its arrays on the heap) |
| deepseek_v4 | the store's lease (pinned rows16 experts unpacked as for a promotion) | the chain's chunk instead of the 128-row block |

| Variable | Default | Effect |
|---|---|---|
| `COLI_VK_CHAIN_ROWS` | from the budget, up to 8192 | Prompt rows per chain chunk; `auto` as unset. Set, it also keeps the engines' own prompt blocks (above). |
| `COLI_VK_CHAIN_ROWS_MAX` | `8192` | The most rows the budget's chunk takes. |
| `COLI_VK_ATTN_BLOCK` | `16` | Rows from which the chain's attention takes `chain_attnb.comp`; `0` never. |
| `COLI_VK_CHAIN_GEMM` | on where supported | The chain's int8/int4 prompt matmuls (`I % 64`, `O % 128`) on `chain_gemm.comp` (cooperative matrices at subgroup size 64 or 32, x rounded to f16 once, 128 outputs x 64 rows a workgroup); `0` keeps the fp32 GEMM. On a Radeon 8060S (Qwen3.6, 1011-token prompt): 517-546 -> 282-287 ms of tiled GEMM, the first token 2.08 -> 1.85 s, perplexity unchanged (7.65). On an RTX PRO 4000 (Qwen3.6 int4-g64, 467-token prompt): an 8192 x 2048 GEMM of 1024 rows 3.64 -> 1.53 ms, the prefill 734-756 -> 865-889 tok/s. |
| `COLI_VK_GEMM_WIDE` | on where supported | A GGUF's dense prompt matmuls from 128 rows on `qmatmul_gemm_bdcw.comp` (128 outputs x 128 rows a workgroup) instead of `qmatmul_gemm_bdc.comp`'s 64, for the types with no IQ grid in shared memory. RTX PRO 4000: a 4096 x 2048 Q4_K matrix against 4096 rows 3.30 -> 1.97 ms; Qwen3.6 UD-Q4_K_M's prompt at 4K tokens 1960-1979 -> 2344-2358 tok/s. `0`: the 64 tile. |
| `COLI_VK_GRP_WIDE` | `32` | A prompt chunk's GGUF experts on `qmatmul_grp_bdcw.comp` (64 rows by 128 decoded rows an item) when the chunk's rows give each expert this many on average, both passes on the matrix units. One layer of Qwen3.6's Q4_K experts: 1024 rows 4.40 -> 3.69 ms, 512 rows 2.90 -> 3.28 (so the threshold); its prompt at 4K tokens 2366-2371 -> 2621-2637 tok/s. `0`: never. |
| `COLI_VK_ATTN_SLICE` | `4294967296` | Rows x positions x heads x head dim past which an attention is cut over several submissions; `0` never. The matrix units' attention (`chain_attn_flash.comp`) counts a unit as a sixteenth: it runs one 8.5 to 13 times faster than `chain_attnb`. |
| `COLI_VK_TIER_STREAM` | on | `0`: no streaming; the tier takes a prompt step in the engine's usual blocks and leaves the cold experts to the CPU, as before. |
| `COLI_VK_TIER_STREAM_SLOTS` | `64` | Staging slots (experts) for streaming, out of the tier's budget. |
| `COLI_VK_TIER_STREAM_ROWS` | the rule | Fix R, the rows from which a cold expert streams (tests). |
| `COLI_VK_TIER_STREAM_HALF` | x rows within 32 MiB | Rows of a sub-batch (tests: small ones cut experts into parts). |
| `COLI_VK_TIER_STREAM_PAR` | on | `0`: a streamed expert's conversion on the engine thread alone. |
| `COLI_VK_TIER_POOL_COPY` | on with the host pool and streaming | `0`: a big step's host pool experts of block types are read in place over PCIe again, not copied into the slots. |
| `COLI_VK_HYBRID_BIG_ROWS` | `2` | A big step's host pool experts of block types with at most this many rows are computed by the CPU in place; `0` none. |

### Validation and timing

The `prefill-*` families in `c/tests/vulkan_engines.sh` compare all ten MoE engines
against their CPU path. They exercise the default chunk, forced small streaming
slots and sub-batches, staged uploads, prefetch across chunks, and streaming off.
The streaming cases require device work and streamed experts; a silent CPU fallback
does not pass. Sanitized cases check the host allocations and lifetime handling.
Lavapipe checks correctness and cannot establish hardware throughput.

For a timing comparison, use the same model, quantization, prompt IDs, output length,
RAM cache capacity and usage-history file in each run. Compare the former 512-row
chain (`COLI_VK_CHAIN_ROWS=512 COLI_VK_ATTN_BLOCK=0 COLI_VK_TIER_STREAM=0`) with the
default chunk and streaming. Also run the default chunk with only
`COLI_VK_TIER_STREAM=0` to separate larger chunks from streaming. Record the selected
chunk, resident and streamed experts, CPU rows, upload bytes, device wait time, peak
RAM and device memory, time to first token and decode tokens per second. Repeat
with both cold and warm caches on an otherwise idle machine. A shorter prefill does
not imply faster decode; staging slots also consume room in the expert tier.

A paired real-model check on Radeon 780M/RADV used Qwen3.8 Flash Next's int4-g64
expert sidecar, a 2,048-token prompt, one generated token, eight CPU threads,
16 RAM slots per layer and the same starting usage history. Each checkpoint was
advised out of the filesystem cache before its run; no other inference process ran.

| Configuration | Prompt plus one token, excluding load | Whole process | Expert assignments on GPU |
|---|---:|---:|---:|
| Former 512-row chunks, streaming off, host dense copies retained | 217.9 s | 229.1 s | 27.6% |
| Automatic chunks, streaming, device-only dense weights; FP32 expert GEMM | 44.8 s | 56.3 s | 96.0% |
| Same, cooperative expert GEMM enabled | 43.6 s | 55.3 s | 96.0% |

The new budget selected 2,304 rows, released 4.07 GiB of dense host copies, and
streamed about 21.8 GiB of cold experts. All three runs generated the same token.
This is one paired prefill measurement per configuration. It measures the combined
chunk, streaming and memory changes; the small difference between the last two
rows does not establish an end-to-end cooperative-kernel gain by itself.

For native FP8 decode on the same machine, a separate short prompt with 16
generated tokens, CAP=16 and three runs per mode (one cold, two warm) produced identical text.
Median decode speeds were 1.10 tok/s on the CPU, 1.42 with the expert tier and
CPU dense work, and 1.14 with the full dense chain. The tier-only configuration
won that workload. These measurements do not predict discrete-GPU or Windows
performance, and the prefill result uses a different expert quantization.

Performance depends on the device, memory bandwidth, model dimensions, expert
coverage and prompt length. FP32 tiled GEMM does not imply that the expert batch
uses matrix acceleration hardware. Measure each model and target GPU before
choosing nondefault chunk or streaming settings.

Additional real-model checks used the same Radeon 780M, eight threads, 64 generated
tokens, identical starting expert histories within each comparison, and advisory
eviction of checkpoint file pages before each process. Each configuration ran once
with no concurrent inference. The values below are generated tokens divided by
prompt plus generation time, excluding model load.

| Model and starting history | CPU | Expert tier, CPU dense work | Expert tier and dense chain |
|---|---:|---:|---:|
| Qwen3.6-35B-A3B, existing int4-g64 container, cap 64 | 5.37 tok/s | 7.70 tok/s | 10.22 tok/s |
| OlMoE-1B-7B, existing 8-bit container, cap 64, empty history | 13.30 tok/s | 10.97 tok/s | 9.43 tok/s |
| Same OlMoE, common history learned from the CPU run | 13.19 tok/s | 11.77 tok/s | 16.35 tok/s |

Qwen3.6's CPU and dense-chain output matched byte for byte; the tier-only output
differed. Dynamic CPU/GPU expert placement changes reduction order and can change
a greedy choice between close logits. All OlMoE configurations produced the same
64 token IDs. Without history, the OlMoE tier uploaded 687 experts (4.03 GiB) during
generation; with history it could warm the selected experts before the timed prompt.
That startup distinction reverses which configuration wins. These are workload
checks, not evidence that enabling every GPU path always improves latency.

The [speculative decoding measurements](speculative.md#measured) also compare
Qwen3.8's existing int4-g64 sidecar with three MTP drafts, prompt lookup and both
together, and distinguish prompt time from generation time.

### A KV cache past the device's budget

Every chain engine keeps the host's KV cache canonical and mirrors each attention
layer's cache on the device. A long context makes those mirrors large (Qwen3.6-35B-A3B's
ten attention layers hold 40 KiB a position, 2.5 GiB at 64K), and past the device's
budget the chain could not hold them. Then each split layer keeps only part of its cache
on the device, the host's RAM holds the rest (it holds every row anyway), and a step's
attention runs on both at once, the two results merged through their softmax
statistics. Below the budget nothing changes: the whole mirrors, the same bits as
before. The pieces are shared (`vk_kvsplit.h`, the `vkc_kvs_*` ops of `vk_chain.h`,
`shaders/chain_kvs.comp`), so every attention form takes the same path.

**What sits where.** A split layer has `ns` slots of `B` positions on the device (a
block table, `bt`, maps a block to its slot). `nw` slots are a window over the newest
blocks: a step's own rows always land there. Where the engine selects positions (Qwen3.8's
QSA, GLM-5.2's DSA, GLM-5.3's pooled lists, DeepSeek's indexer), with `COLI_VK_KV_PIN=1` the other half of the
slots holds the blocks the lists read most, counted per step and pinned two at a time
ahead of pinned blocks read less (hits halve every 128 steps). A step from `pos_base`
uploads the resident rows below it that the host wrote since (a watermark, as for the
whole mirrors), and the whole of a block that just got its slot.

**The partition, and why it is the row's own.** A row at position `pos` attends on the
device over the blocks `pos/B - anchor .. pos/B` (and, from a list, the pinned blocks),
and on the CPU over the rest. A step writes at most `chunk` rows, so every block of a
row's share is in the window while any step holds the row, and the share covers the
step's earlier rows, which reach the host's cache only after the step. `chunk` is the
largest both allow, half the window less a block: a prompt's chunk is a step, and each
chunk carries its own frames and expert loads.

| Qwen3.6-35B-A3B, 7579-token prompt, Radeon 780M, the chain | first token after |
|---|---|
| the whole mirrors | 77 s |
| 1024 positions a layer on the device, chunks of an eighth of the window (before) | 170 s |
| the same, a quarter | 141 s |
| the same, half the window less a block (now) | 122 s |
| 4096 positions a layer, an eighth (before) | 183 s |
| 4096 positions, a quarter | 165 s |
| 4096 positions, half less a block (now) | 142 s |

(`COLI_VK_KV_DEVICE_ROWS` forced the split, `OMP_NUM_THREADS=8`, cap 256, 32 new tokens,
the model in the page cache; two runs of the eighth agreed within 1%, one run each of the
others.) A row's share shrinks with it (896 of 1024 positions before, 576 now), so a
decode step's host part grows; the 31 decode tokens took 3.4 to 3.6 s either way. Because the share depends on the row's position only, and the host's part is cut
into chunks fixed by position and joined in order, a row gets the same bits however a
forward is cut into steps: a prompt in one chunk or many, a decode step, a verify, a
resumed prefix or a cold one. Pins follow the history of reads, so a listed row's bits
can depend on it; pins are therefore off by default and enabled explicitly with
`COLI_VK_KV_PIN=1`. DeepSeek stages only the missing compressed rows named by its
selection, then runs the complete sparse attention on the device in the original list
order. Its arithmetic remains independent of cache residency and pins.

**A step.** One that sees nothing outside the device's share records the attention as
before, in the layer's frame (`VKC_KVS_FIN`: the device's arithmetic gives the same
bits as a merge with an empty host part). Otherwise the frame (F1) is submitted with the
queries (and lists) copied down; the device's part is recorded in a frame of its own
(F2) and submitted; the host waits for F1 only and computes its part over the host's
rows while F2 runs (OpenMP over rows, heads and chunks of 2048 positions); a third frame
(F3) uploads the host's part, merges (`vkc_kvs_merge`: `M = max(m_d, m_c)`, `out =
(e^(m_d-M) acc_d + e^(m_c-M) acc_c) / (e^(m_d-M) l_d + e^(m_c-M) l_c)`, the gate after,
DeepSeek V4's bf16 rounding after) and carries the rest of the layer. The extra host
round trip is paid only on layers with a host part.

| Op | Shader | What it does |
|---|---|---|
| `vkc_kvs_attn` | `chain_kvs` (mode 0) | grouped-query attention (chain_attn's arithmetic) over the device's share: head-major or position-major rows, V's own head dim, a window, a sink (the device's part holds it), lists |
| `vkc_kvs_mla` | `chain_kvs` (mode 1) | the MLA core (chain_mla's) over the device's share of the latent and rope rows, causal or listed |
| `vkc_kvs_rel` | `chain_kvs` (mode 3) | Inkling's attention (chain_relattn's): the relative-position bias and tau |
| `vkc_kvs_ds` | `chain_kvs` (mode 4) | DeepSeek's sparse attention with its sink over the window rows and the resident compressed rows; with every listed row resident, `vkc_dsv4_attn`'s bits |
| `vkc_kvs_merge` | `chain_kvs` (mode 2) | the two parts joined, a gate or V4's bf16 rounding after |

**The host's part on the device (`COLI_VK_KV_COLD=device`, off by default).** The
host's rows can also be attended on the device, so a step with a host part runs in one
frame with no round trip:
- **The copy.** Each split layer keeps a shadow of the host's rows in page-aligned memory
  the device reads in place (an imported host allocation, `VK_EXT_external_memory_host`).
  The shadow follows the host's cache row by row and is lowered with it on a rewind or a
  rollback.
- **The ops.** `chain_kvs` reads the shadow with its COLD flag (the GQA, MLA and Inkling
  forms; position t is row t). The prompt's rows take the blocked attention
  (`chain_attnb`'s part mode) in chunks of 512 positions, one workgroup each, and mode 5
  of `chain_kvs` joins them in order.
- **The bits.** The chunks are fixed by position, so a row's bits do not depend on how a
  forward is cut into steps, as on the CPU path.
- **Where it applies.** It needs a device that imports host memory, no read-based pins
  and no staged uploads. DeepSeek's sparse forms keep the host's part on the CPU, and so
  does the second device's chain ([Layers on two devices](#layers-on-two-devices)): the
  import is the primary's. Where it cannot run, the line says so and the CPU computes the
  host's part.

The run's report adds `| the host's part on the device: N layer steps, R rows copied
to the shadow in T ms (M MiB held)`.

| Qwen3.6-35B-A3B, 7579-token prompt, Radeon 780M, the chain | first token after | 31 decode tokens |
|---|---|---|
| 1024 positions a layer on the device, the host's part on the CPU | 121 s | 3.5 s |
| the same, the host's part on the device | 114 s | 20 s |
| 4096 positions a layer, on the CPU | 143 s | |
| the same, on the device | 137 s | |

The prompt gains 4 to 6%. Decode is 5.7 times slower: with the default, the CPU
computes the host's part while the device runs its own, and on this integrated GPU the
decode attention is slower than the CPU's. Hence the default. A dedicated GPU, whose
attention outruns the CPU's by more, has not been measured.

`vk_kvsplit.h` carries the rest: the plan, the tables, the uploads and the stores of a
step's rows, the host's part (`vkc_kv_host_attn`, with the same bias, tau and V4
rounding), and one call per form for an engine (`vkc_kv_gqa`, `vkc_kv_mla`,
`vkc_kv_mla_qkv` and `vkc_kv_mla_attn`, `vkc_kv_rel`, `vkc_kv_ds`).

**The plan** (`vkc_kv_plan`, when an engine makes or grows its mirrors): the whole
mirrors when they fit four fifths of the device's free budget (`VK_EXT_memory_budget`,
or the largest device-local heap less what is allocated), else the rows that do, at
least three blocks. `COLI_VK_KV_DEVICE_ROWS=N` sets the rows itself (the tests' small
device, a measurement's split), `COLI_VK_KV_BLOCK` the block (64), `COLI_VK_KV_SPLIT=0`
turns the split off (past the budget the chain declines as before), `COLI_VK_KV_PIN=1`
enables read-based pins. The line says what was decided:

```
[VK] qwen36 chain: the KV cache split past the device's budget: 4096 of 65568 positions a layer on the device (64 blocks of 64, 0 for pins by reads; a row's newest 2112 positions on the device), the rest in RAM (160.0 MiB on the device instead of 2561.2)
```

and each run reports the split's steps: `[VK] <engine> chain: KV split: N layer steps, M
with a host part (time waiting for the queries, host work over P positions, Q sparse rows staged), K
blocks pinned by reads`. Sparse staging reuses at most 32 MiB of device scratch per
query, accounted for by the chain allocator. Allocation or scratch-cap failure takes
the existing CPU fallback; it never allocates another complete compressed cache.

**What splits, per engine.**

| Engine | Split | Stays whole on the device |
|---|---|---|
| qwen36 | the K/V of the attention layers | the DeltaNet state |
| qwen38 | the K/V of the attention layers; QSA's lists reach both parts, pins follow them | the index keys and pooled block keys (every row scores them every step), the DeltaNet and PLE state |
| olmoe | every layer's K/V | |
| mimo | the full-attention layers' K/V (position-major, the sink in the device's part) | the sliding layers' rings (bounded by the window) |
| inkling | the global layers' K/V (the bias and tau on both sides) | the sliding layers' rings, the convolution states |
| colibri | the latent and rope rows; DSA's lists reach both parts, pins follow them | the DSA index keys |
| glm53 | the MLA latent rows; the pooled lists reach both parts, pins follow them | the index keys, pool gates and pooled keys, the KDA state |
| kimi_k3 | the gated MLA layers' latent rows | the KDA state |
| deepseek_v41, deepseek_v4 | the compressed rows the sparse attention reads; the indexer's picks pin blocks; each chunk's new compressed rows reach the host's array as the chunk ends | the window ring, the index keys, the compressors' rings |

Every engine's host cache already holds a step's rows before the next step runs, so the
host's part reads only rows the host has; DeepSeek's chains, which wrote their state
back at the end of a forward, now copy each chunk's new compressed rows as the chunk
ends.

**Arithmetic.** f32 throughout: the device and the host sum in other orders than the
CPU's attention and than each other, so a split row agrees with the CPU's to rounding.
DeepSeek V4 rounds its attention weights to bf16 against the maximum of the complete
selection. The split preserves that maximum and the original accumulation order by
staging missing selected rows. Independent partial bf16 softmaxes are not equivalent
and are not used. Both DeepSeek variants retain the full-device result bit for bit;
V4's chunk, decode versus teacher-forced, and draft-rejection checks remain enabled.

**A lost device** is handled as each engine handles it without the split: the KV rows
are the host's, so the CPU redoes the step (and an engine with a recurrent state
rebuilds it from the prefix record, as before).

**Tests.** `make vk-chain-check VK=1` runs the ops through the engines' sequence of
steps (the window placed, the rows below uploaded, the step's rows stored, the parts
merged), decode and prefill, against a double-precision reference of the full attention:
GQA with a gate, a sink, position-major rows, a window and lists with pins; MLA (NoPE, a
nonzero start, lists, latents up to 1024); Inkling's bias and tau; DeepSeek's lists
(V4's rounding too, both split and whole caches equal `vkc_dsv4_attn` bit for bit); the MLA
layer test of every geometry over a split cache; the host's cache filled as an engine's
is (a step's rows only after the step, NaN before); and the same bits for every
position in steps of 1, 2, 3 and 4. `tests/vulkan_engines.sh kv-split` gates every chain
engine but DeepSeek's under an emulated small device (`COLI_VK_KV_DEVICE_ROWS` of 8 to 32
positions in blocks of 2 to 8, contexts of 80 positions and more, most steps with a host
part): the CPU's tokens, logits within each family's tolerance, chunks of 3, prompts
only, the split off, a lost device, pins, MTP and n-gram drafts, serve sessions with
pins, prompt-cache extensions and divergent prompts, and the prefix-reuse tests;
`kv-split-deepseek` the same for deepseek_v41 and deepseek_v4; `kv-split-sanitize` and
`kv-split-deepseek-sanitize` a set of each under ASan and UBSan. `kv-split-cold` and
`kv-split-cold-sanitize` run `kv-split` and `kv-split-sanitize` again with
`COLI_VK_KV_COLD=device`, and every gate must have run the host's part on the device.

Validation on the local Lavapipe device covers numerical correctness, not hardware
throughput. A discrete GPU has not been measured for this change.

## Adding an engine to the tier

Every MoE engine here is on the tier ([the table above](#the-routed-expert-tier-vk_tierc));
these are the steps the next one takes. They are in [`c/vk_tier.h`](../c/vk_tier.h);
in short:

1. **Describe the experts** (`VktConfig`): geometry, how RAM holds gate/up and down
   (`VktSrc`: int8 per row or grouped, the int8 copy of an int4 container, int4
   pairs signed or `v+8`, `expert_ffn.h`'s planar int4-g64, int3-g64, MXFP4 with f32
   or ue8m0 scales, fp8 per group or in square blocks, bf16, f32), the activation
   (`VKT_ACT_SWIGLU` with an optional clamp, `VKT_ACT_SITU`, `VKT_ACT_SWIGLU_V4`), the most assignments a
   step carries, the RAM the expert cache may still take (and, so the host pool can
   share that RAM, the least the cache needs and a callback that lowers it:
   `.ram_floor`, `.ram_given`) and the dense bytes still to come to the device;
   optionally `.max_experts`, a count the engine's users already
   size its device tier in (GLM-5.2's `COLI_VK_EXPERTS`), and `.extra_layers` with
   their own `VktFmt` for layers past the model's whose experts RAM holds in another
   form (an MTP head's: [above](#the-mtp-heads-layer-on-the-tier-coli_vk_tier_mtp)). `atexit(coli_vk_shutdown)`,
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

## Memory placement without Resizable BAR

Resident data (the dense weights, the routed-expert tier, the MLA KV mirror) is written
by the host once and read by the device for the rest of the run. The mapped path puts it
in the memory type for what the CPU writes (HOST_VISIBLE|DEVICE_LOCAL, or system memory
on an integrated GPU) and writes it through a mapping. On an integrated GPU, a CPU device
(Lavapipe) or a discrete card with Resizable BAR, that type covers the device's memory. A
discrete card without Resizable BAR (every Turing card, every Ampere card on its launch
VBIOS, older AMD cards with the option off) exposes it as a window of about 256 MB of 8 GB
or more. NVIDIA's driver refuses allocations past the window: on an RTX 3070 with its
launch VBIOS the backend warned "only 246 of 8192 MB VRAM is host-visible", placed no
matrix, every tier upload failed and the chain stopped at its first matrix. RADV places
them in system RAM instead, where every access crosses PCIe.

**Memory types by use.** The backend chooses the memory type of a buffer by what the buffer
is used for: only the GPU uses it, the CPU writes it, or the CPU reads it back. A discrete
GPU and Lavapipe get device-local memory where the use allows it, and a host-cached type
for what the CPU reads back. An integrated GPU with a heap of system memory gets
host-visible system memory for all three uses, so no buffer is placed in the VRAM
carve-out reserved by the BIOS. An allocation that runs out of memory there is retried in a
device-local type. `COLI_VK_DEVICE_CAP_MB`, the budget and the expert batch count the
device memory: the device-local heaps, and every heap of an integrated GPU.

**Staged uploads** put resident data in a DEVICE_LOCAL memory type the host does not map
and copy it there from a host staging buffer with `vkCmdCopyBuffer`.

**Straight from host memory.** Where the device has `VK_EXT_external_memory_host`, a
staged upload skips the staging buffer: the source pages are imported as a transfer
source and the device copies from them directly, so the CPU does not copy the bytes a
second time. It applies to:
- the tier's experts (and the streaming slots' refills), from the host image they are
  converted in, which is allocated aligned to the device's import alignment;
- the trunk's rows from the weights themselves, when they need no padding.

An import lives until its command buffer's fence. Pages the driver will not import (some
file-backed mappings, a refusal) are staged as before. The second device (`COLI_VK_DEV2`)
does the same. `COLI_VK_UP_IMPORT=0` stages every copy; a `[VK] staged uploads: copied
straight from host memory` line says imports are on, and the exit report counts the
imported bytes and copies and the refused ones. On Lavapipe the results are the same bits
either way. On a discrete card it removes a host copy of every uploaded byte, which matters
most for the tier's warm start and the trunk's placement; its speed there was not measured.

**The rule** (`place_decide` in `backend_vulkan.c`). `COLI_VK_STAGED=1` stages,
`COLI_VK_STAGED=0` keeps the mapped path. Unset: staged when the host-visible
device-local heap holds less than a quarter of the largest device-local heap, or there is
no host-visible device-local type at all. That is a card without Resizable BAR (256 MB of
8 GB) and never a card with it, an integrated GPU or Lavapipe, whose host-visible heap is
the whole device-local heap. `COLI_VK_HOST_VISIBLE_CAP_MB=N` treats the host-visible heap
as at most N MiB, so a device with Resizable BAR or unified memory takes the decision a
card without it would (the tests use 246). The device-local target is a type that is not
host-visible on the largest device-local heap (else that heap's device-local type:
Lavapipe has one type for everything); the staging type is host-visible and coherent and
not device-local where one exists, so staging never takes the window. Vendor types
(AMD's uncached and device-coherent ones) are passed over.

**A window smaller than the card.** A card whose host-visible device-local type is as big as its
heap can still be given less of it. On a Colab A100 80 GB (NVIDIA's 580 driver) the engine's device is
given 39.75 GiB of it, mapped or not, and then a block is refused with `VK_ERROR_OUT_OF_DEVICE_MEMORY`,
while the heap says 80 GiB and `VK_EXT_memory_budget` 79.21; on a Colab L4 (24 GB) it is given 9.75 GiB
of 22.5.
On the mapped path the expert tier and the weights add up to about that window, and the allocations
made after them (the streaming slots, the expert batch's scratch) are refused: with Qwen3.8's int4
experts, `origin/dev` held 13,784 of 24,576 experts, and a prompt of 512 tokens fell to the CPU and
took 98 s to the first token. On the L4 the pool asks the type for another block for each tensor
that does not fit (`pool_suballoc`; read in the code, the stuck runs did not count their refusals), a
refused block took about 0.6 s in a probe program, and the start-up of Qwen3.6's expert tier (10,240
experts) had not finished after 15 minutes in four runs, with five threads waiting for the pool's mutex
and one inside the driver; staged it takes 21 s. So when `COLI_VK_STAGED` is unset and the card is discrete, `coli_vk_init`
asks once, as soon as the device exists (`vk_window_decide`), by allocating 1 GiB blocks of the type
up to 9/10 of its heap in whole GiB (less if `VK_EXT_memory_budget` says less is free, so memory
another process holds is not read as a small window; without the extension it is), and takes the
staged path when it is refused before that. The blocks go back all together when the loop ends. That took 2.3 to 2.4 s at the engine's start on that A100
(39 GiB obtained, a refusal, the frees; about 50 ms for each GiB allocated and freed) and 1.0 s on
the L4, and another process on the card can see allocations refused while they are held. For this
the device is made with the queue a staged card would ask for, as the rule above would.
`COLI_VK_STAGED=0` skips the probe. The probe's blocks are counted by the device-memory books like
any other allocation, so `COLI_VK_DEVICE_CAP_MB` limits them. `COLI_VK_WINDOW_PROBE_MB=N` answers N
MiB without asking, and `COLI_VK_WINDOW_ASK_MB=N` asks for N MiB at most so that the loop runs for
real (tests; a device with a host-visible device-local type that is not staged already). The log says which way it went; on that A100:

```
[VK] memory: staged uploads, resident data in device-local memory (type 1, 81920 MiB heap) copied from host staging memory (type 3) on a transfer queue (the driver gave 39 of the 72 GiB asked of the host-visible device-local memory (9/10 of its heap, less if its budget leaves less), found in 2.3 s)
```

and when the type gives what was asked, the line starts `[VK] memory: mapped weights, the host-visible
device-local memory gave the N GiB asked`.

**What moves where, staged:**

| Data | Mapped path | Staged |
|---|---|---|
| Dense resident tensors (`coli_vk_tensor_ensure`, `coli_vk_matmul`'s first call) | weight pool, host-visible | weight pool in device-local blocks; rows and scales streamed through two 16 MiB staging slots, the upload complete when the call returns |
| The routed-expert tier's pool (`vk_tier.c`) | tier pool, host-visible, filled in place by the uploader thread | device-local; the uploader fills a host image (`coli_vk_tier_tensor`) and `coli_vk_tensor_commit` copies it, all three matrices of an expert in one submission |
| `COLI_VK_DEV2`'s experts | its pool, host-visible | its pool in its device-local memory, by the same rule on that device |
| MLA KV mirror and q-prep norm weights (`COLI_VK_ATTN`) | host-visible, written per row | device-local; each row is a pending copy recorded at the head of the next absorb or q-prep command buffer, their only readers |
| The dense chain: state, KV caches, parameters (`VKC_DEV`) | device-local already, written through the frame's staging | unchanged |
| The chain's host-written buffers (`VKC_UP`: frame staging, the rows each layer step uploads) | the host-visible device-local type | host staging memory, out of the window |
| Readbacks (`VKC_DOWN`, `y` scratches) | host-visible, cached | unchanged |
| Per-call input scratches, device-only scratches | unchanged | unchanged (a few MB, they fit the window) |

**Synchronization.** One uploader per device, its own command pool, two command buffers
and two fences (a slot is filled while the other copies), one upload at a time under its
mutex. Its queue: a transfer-only family (a copy engine) when the device has one, else a
spare queue of a family the backend already uses (the 780M's second compute queue, a third
queue of the Iris Xe's main family), else the main queue (Lavapipe has one queue), in
which case every submit of the backend and of the chain takes the same lock
(`vk_submit`, `coli_vk_queue_submit`). An upload waits for its fences before it returns,
so a tensor is complete before any queue reads it: the tier's uploader thread hands an
expert to the engine thread only after its commit returned. When the uploader's family
differs from the main or the tier queue's, the staged tensors' buffers are created
`VK_SHARING_MODE_CONCURRENT` over those families. The KV mirror's pending copies ride the
main queue in the same command buffer as their reader; a row written again before that
(a rewound cache) first sends what is pending, so no two pending copies overlap.

**When an upload fails.** A staging buffer that cannot be had at startup leaves the
mapped path on (with a line saying so). A device-local block or buffer the driver refuses
is out of memory: that matrix stays on the CPU, the tier takes fewer experts, the KV
mirror's layer runs its attention on the CPU. A command buffer that would not record or a
submit refused fails that upload only: the uploader waits for what it had sent, frees the
tensors and starts the next upload clean; the matrix stays on the CPU, and an expert whose
commit failed stays on the CPU with the tier's budget intact (it may be promoted again). A
fence wait that fails means the copy may still run: the device is taken as lost, as for
every other wait, so the dense matrices, the tier (its batches stop) and the chain (at its
next frame, rebuilding the state on the CPU) all move to the CPU. `COLI_VK_STAGED_FAULT`
injects each of these, and `tests/vulkan_engines.sh staged-faults` (with
`staged-faults-sanitize` under ASan and UBSan) runs qwen36's matrices, its tier awaited and
with the uploader thread free, its chain, and colibri's KV mirror through every point,
gated on the CPU's tokens.

**A fresh device-local block is zero-filled** (`vkCmdFillBuffer`) before its first
tensor. On an RX 580 (RADV, Polaris) the author of #1338, where this approach comes from,
measured results that differed slightly from run to run when read from a block the GPU had
never touched, and identical ones after a fill of any value. Skipping the fill
(`COLI_VK_TEST_NOFILL=1` in the harness) changed nothing on Lavapipe, the Iris Xe or the
780M: the harness's digest of every result was the same with and without it. It costs one
fill per 256 MB block and stays.

**Lines it prints.** At startup, staged only:

```
[VK] memory: staged uploads, resident data in device-local memory (type 0, 21466 MiB heap) copied from host staging memory (type 2) on a queue of its own (246 of 21466 MiB of device-local memory is host-visible (COLI_VK_HOST_VISIBLE_CAP_MB))
```

and at exit, where the data ended up (the tests read the last field; qwen36's tiny
fixture with the chain on the 780M, `COLI_VK_STAGED=1`):

```
[VK] memory at exit: weights 0.8 MiB, expert tier 0.4 MiB (peaks), KV mirror 0.0 MiB in device-local memory type 0 (not host-visible); the dense chain's state in type 0 (device-local); 1.2 MiB staged in 530 copies, 2 blocks zero-filled; resident data in host memory: 0.0 MiB
```

With `COLI_VK_STAGED=0` on a small-window card, the old warnings stay.

**Tested.** None of our devices lacks Resizable BAR, so the path is forced
(`COLI_VK_STAGED=1`) or the decision emulated (`COLI_VK_HOST_VISIBLE_CAP_MB=246`):

- the `VK_TEST` harness prints a digest of every result the device returns (every
  format, the tiled GEMMs, the expert batch, and in the full run the gate_up, the expert
  group, the absorb core, the q-prep chain and a rewound KV mirror): mapped, staged,
  staged without the zero fill and under the emulated window, the digest is the same on
  Lavapipe, the Iris Xe through Dozen and the Radeon 780M, and for the format and
  expert-batch cases the same as the backend's before this change;
- `tests/vulkan_engines.sh staged`: that comparison, then the tier (`test_vk_tier`) and
  the chain's ops (`test_vk_chain`) staged, each ending with no resident data in host
  memory; `<family>-staged` runs a family with `COLI_VK_STAGED=1`, and `qwen-staged`
  first runs qwen36's tier and chain under the emulated window with the decision left to
  the backend (CI: the staged family and the shader family staged in the Vulkan job,
  `qwen-staged` and `qwen-chain-staged` in the engines matrix);
- on the 780M, qwen36's tiny fixture with the tier and the trunk on the device, and with
  the chain: the CPU's tokens, and logits bit for bit the mapped run's, staged and under
  the emulated window.

**Measured on the 780M**, where staging is not needed (unified memory: the default stays
mapped), to see what it costs: Qwen3.6-35B-A3B, the method of [the tier's
measurements](#measured-on-a-radeon-780m) (int4 gs64 at cap 64, `OMP_NUM_THREADS=8`, the
model files evicted from the page cache before each run, 1-min load under 2, every arm
from the same history), the same binary, `COLI_VK_STAGED=0` against `1`. Decode is 100
tokens after a 25-token prompt (in brackets the whole process, the warm start's 11 GiB of
staged experts included), prefill a 512-token prompt (time to the first token):

| | mapped | staged |
|---|---|---|
| decode, tier (trunk on the CPU) | 7.94 tok/s (22.96 s) | 7.96 tok/s (22.60 s) |
| decode, tier and chain (`COLI_VK_CHAIN=1`) | 9.91 tok/s (20.16 s) | 9.96 tok/s (20.09 s) |
| prefill, tier | 12.26, 12.22 s | 12.29, 12.15 s |
| prefill, tier and chain | 9.53 s | 9.43 s |

The same within what one run to the next varies on this box, and each pair printed the
same text. On
unified memory a device-local copy is a RAM copy and the mapped path reads the same RAM,
so this says only that staging costs nothing here. **Not measured: a discrete card without
Resizable BAR**, the case staging is for (none is available); there the copies cross PCIe
once per upload and the device then reads VRAM instead of the window or system RAM. The
contributor who reported the RTX 3070 offered to run it.

## f32 matrices that came from bf16 or f16

Several engines keep dense matrices as f32 on the host although the checkpoint stores
them in two bytes a value (bf16, or f16 in some OLMoE releases): OLMoE's attention,
routers and lm_head, Kimi K3's routers and KDA decay and beta projections (and its whole
trunk at `K3_BITS=32`), Inkling's routers, GLM-5.3's f32 trunk. Every such float is a
bf16 or an f16 widened, so the device copy goes up in two bytes a value, as bf16 (fmt 11)
or f16 (fmt 14): the same values in half the bytes. On OLMoE-1B-7B the dense part takes
0.75 GB instead of 1.5 GB, and the expert tier has the rest. The check is exact and runs
once per matrix at upload (`coli_vk_f32_fmt`): bf16 when every float has a zero low half,
else f16 when an f16 holds every float, else the matrix stays f32 (fmt 10). The chain's
fit, the expert tier's dense budget and the dense-host decision count each matrix at the
format it will take, and so does `coli plan`: a BF16 or F16 tensor from the checkpoint's
header, an F32 one from its values (the one read of tensor values the planner makes).

The values are the same, so the tokens are the CPU's; the logits can differ from an
f32 upload in the last bits, because the shaders sum a 2-byte row in steps of 8 rather
than 4.

## Dense weights on the device only

Without this mode, an engine whose dense part runs on the device (the chain, or the
per-matrix path with `COLI_VK_DENSE`) keeps the host copy of every resident dense
matrix as the CPU's fallback. That costs the dense part's RAM twice on an integrated
GPU (the device's memory is the same RAM), and on any machine it takes from the
experts' RAM cache: the cap or the plan counts the dense weights in RAM. On a 32 GB
machine with a 16 GB card the difference decides the plan. DeepSeek V4 Flash there
(#1852) planned its dense layers as not resident (reloaded from disk every forward),
so its chain declined and the CPU did the dense work while the GPU sat at 9%.

`COLI_VK_DENSE_HOST=0` puts the dense weights on the device only:

- **At start**, once the device and the chain are decided and before the expert tier
  sizes its budget, every resident dense matrix goes up (staged or mapped, as
  [above](#memory-placement-without-resizable-bar)) and its host copy is given back.
  The per-matrix path and the chain share those tensors. The tier's budget sees them
  placed.
- **Every step the chain does not run** (prompts only, a declined step, a head or a
  draft the engine runs per matrix) takes the device too: the CPU has no copy. In this
  mode the engine forces the per-matrix path on, whatever `COLI_VK_DENSE` says.
- **The CPU fallback still works.** When the CPU needs a matrix the device holds alone
  (a lost device, `COLI_VK_CHAIN_FAULT`, a staged upload that failed), the engine reads
  it back from the checkpoint, matrix by matrix as each is first needed, with the bytes
  the load made (the same conversion: Qwen3.8's int8 rows, GLM's quantization, ...),
  and keeps it from there on. The tokens are the CPU's.
- **What stays on the host** is what the CPU reads directly every step: the embedding
  tables (their rows are gathered on the CPU), norms and small vectors, and per engine
  the pieces listed below. Qwen/Clef's imported weight pages on an integrated GPU
  also stay alive: those rows are shared with the device and already occupy RAM once.
- **The RAM goes to the experts.** Each engine that sizes its own expert cache from
  the RAM it sees counts the dense weights as not in RAM in this mode, and so does
  `coli plan` (`resource_plan.py`) credits only recognized matrices and formats
  that the engine can release. It keeps the embedding, CPU-readable vectors,
  norms and unrecognized components in the RAM reservation. Packed formats whose
  runtime layout cannot be established from the checkpoint, GLM's CLI-selected
  QT formats, and Inkling's hardware-dependent BF16 path receive no advance RAM
  credit; the runtime may release more after successful uploads. Quantized matrices
  receive a conservative credit for their loaded size, not the larger source
  tensor. Qwen3.8's CPU-int8 estimate follows `Q38_TRUNK_SKIP` and
  `Q38_TRUNK_MIN_KB` and includes the float scales per row. On an integrated
  GPU the device copy still occupies physical RAM: the planner charges it once in
  `shared_device_dense_bytes`, inside `runtime_bytes`. Dropping the host copy frees
  one copy of the weights; it does not remove the device's memory from the budget.
  On a discrete GPU the fit decision uses the device budget minus its current heap
  usage and a 1 GiB reserve. The shared-device reservation and fit estimate remain
  conservative when a component's device representation is unknown. The resulting
  expert capacity depends on that device, context size and available memory.

For GLM-5.3 the RAM estimate applies `GLM53_BITS` to recognized dense matrices
individually. Int4 requires columns divisible by 64; narrower matrices use int8
rows with a float scale per row, as in the engine. Embeddings, norms, routers,
vision weights and unknown floating components retain an F32 reservation. The
absorbed `kv_b` matrices also retain a conservative F32 reservation. Packed source
weights and their scales keep their stored size. Cached checkpoint scans do not
cache environment settings: changing `GLM53_BITS` updates the next plan without
requiring a rescan.

**The default** (`coli_vk_dense_host_decide`, beside `coli_vk_dense_decide` and
`coli_vk_chain_decide`): with the dense part on the device, an integrated GPU drops
the host copies (its memory is the same RAM: the copy would hold the dense part
twice); a discrete GPU drops them when its free memory (`VK_EXT_memory_budget`), less
1 GiB, holds them, else keeps them as the fallback; a CPU device (Lavapipe) keeps them.
`COLI_VK_DENSE_HOST=1` keeps them anywhere. With the dense part on the CPU there is
nothing to drop. The lines it prints:

```
[VK] qwen38: dense weights on the device only (an integrated GPU: its memory is the CPU's RAM, the host copy would hold them twice; COLI_VK_DENSE_HOST=1 keeps it)
[VK] qwen38: 1007 dense matrices on the device only, 4.21 GiB of host RAM given back; kept on the host: the embedding (its rows are gathered on the CPU), the vision tower, norms; RSS now 2.10 GiB
...
[VK] qwen38: dense weights at exit: 1007 matrices on the device only (4.21 GiB of host copies dropped), 0 read back from disk for the CPU (0.0 MiB); RSS 24.60 GiB
```

The `dense-only-*` test families cover Qwen3.6, Qwen3.8, OLMoE, Inkling, MiMo,
Kimi K3, GLM-5.2, GLM-5.3, DeepSeek V4 and V4.1, including staged uploads,
device loss and sanitizer variants. On Kimi K3 the generated steps' logits must
match the run retaining its host copies byte for byte. Earlier prompt logprobs
use a CPU head with host copies and a GPU head without them, so they retain the
existing CPU-reference numerical bound; they are not an identical-arithmetic
comparison. A trace that declines the chain must still exercise the per-matrix
GPU fallback.

## A partial chain

On a card too small for every dense layer the chain used to be decided on with no check
that the layers fit. Setup uploaded matrix after matrix and gave up at the first one that
did not fit, leaving what it had uploaded on the device, used only by the slow
per-matrix path; the expert tier then sized itself from what was left. DeepSeek V4 Flash
(7.9 GB of dense layers) on an 8 GB card ended there with no chain, half its dense part in
VRAM and no expert tier: #1852's case on a smaller card.

The chain now takes what fits: the first N layers run on the device as every layer did
before (their matrices, the residual and their state), and the CPU path runs layers N
to L-1 and the head. N = L is the full chain. N = 0 is the chain off with nothing of the
dense part on the device, every byte left to the tier.

**How N is chosen** (`vkc_fit`, `vk_chain.h`), once at startup, before any upload and
before the RAM plan sizes the expert cache. The room is the device's free memory
(`coli_vk_free_bytes`: `VK_EXT_memory_budget`'s budget less its usage, else the largest
device-local heap less what the process holds) less the reserve, `COLI_VK_TIER_RESERVE_GB`
(1 GiB). Out of it come, in this order:

- the chain's fixed bytes: the engine's (the scratch of one prompt chunk at the engine's
  block, buffers every layer shares) and the pools' granularity (a weight block, a block
  of each of the chain's three pools, the frames' first staging buffers);
- each layer's bytes: its matrices as uploaded (each of their two ranges aligned), its
  state at its first size (the KV split covers its growth), its part of the parameters;
- the tail: what the device takes only with every layer there (a head, matrices the
  per-matrix path uploads as it meets them).

N = L when the fixed bytes and every layer fit, the tail going up too when it fits beside
them. Otherwise N is the most layers from layer 0 that fit, and the tail stays on the CPU.
`COLI_VK_CHAIN_LAYERS=n` forces N (at most L; `0`: the chain off). When everything fits
the result is the chain as before. Two lines say what was decided (DeepSeek V4's six-layer
test fixture on Lavapipe under `COLI_VK_DEVICE_CAP_MB=59.691730` with
`COLI_VK_TIER_RESERVE_GB=0.04`):

```
[VK] deepseek_v4 chain fit: free 62591315 B, reserve 42949672 B, fixed 18931200 B (the engine's 1891840 B, the pools' 17039360 B), tail 0 B, layers 159720 278760 192104 159720 159720 278760 B, matrices 138888 213404 155280 138888 138888 213404 B
[VK] deepseek_v4 chain: 3 of 6 layers on the device (0.6 MiB), 3 on the CPU (free 59.7 MiB, reserve 41.0 MiB)
```

and, once the layers are on the device and before the tier sizes itself:

```
[VK] deepseek_v4 chain: 3 of 6 layers placed: 507572 B of matrices on the device (the fit counted 507572 B for these layers), chain buffers 4265784 B, device memory held 4853760 B
```

**A layer that does not reach the device.** The budget can promise what a driver then
refuses, or an upload can fail. Setup goes layer by layer: a layer whose matrices or state
do not all reach the device is freed whole (its tensors, its buffers), its matrices stay
the CPU's, and the chain keeps the layers before it. Nothing of the failed layer stays on
the device: the placed line's bytes are the kept layers' exactly.

```
[VK] deepseek_v4 chain: 2 of 6 layers on the device (0.4 MiB), 4 on the CPU (layer 2 did not reach the device: a matrix the device refused; what it had placed was freed)
[VK] deepseek_v4 chain: 2 of 6 layers placed: 352292 B of matrices on the device (the fit counted 352292 B for these layers), chain buffers 4253936 B, device memory held 436207616 B; COLI_VK_STAGED_FAULT's point reached 11,27,0,0,0,0 times by the end of each layer
```

(the same fixture with `COLI_VK_STAGED=1 COLI_VK_STAGED_FAULT=submit:29 COLI_VK_CHAIN_LAYERS=6`:
the 29th staged submit, the second inside layer 2, fails)

A lost device keeps its handling: the CPU runs the forward again and from there on.

**The handoff.** A forward runs the chain's N layers on the device chunk by chunk as before;
after layer N-1 the residual (and the engine's per-token stream state: DeepSeek's hc_mult
streams) comes back to the host once per chunk, and the CPU's layer loop runs from layer N
on it, then the head. The device's layers keep their state on the device (the mirrors
behind their watermarks), the CPU's layers on the host. While part of the model stays on
the CPU, the per-matrix path uploads nothing new: a CPU layer's or the head's matrices are
never put on the device behind the fit's back.

**Host copies, per layer.** With `COLI_VK_DENSE_HOST=0` (or its rule) only the N layers
give their host copies back; the CPU's layers keep theirs and never read a matrix back from
disk. The engine's RAM plan counts only the N layers' matrices out of RAM, and so does
`coli plan` (`resource_plan.py`'s `vk_chain_fit`, which predicts N with the same rule from
the device's budget and the checkpoint's header). The dense-host lines say it:

```
[VK] deepseek_v4: 31 dense matrices on the device only, 0.4 MiB of host RAM given back (the 3 of 6 layers on the device; the 3 on the CPU keep theirs); kept on the host: norms, block scales, the router, the indexer's weights_proj, the mHC mixes, the head; RSS now 0.13 GiB
[VK] deepseek_v4: dense weights at exit: 31 matrices on the device only (0.4 MiB of host copies dropped, the 3 of 6 layers on the device), 0 read back from disk for the CPU (0.0 MiB); RSS 0.14 GiB
```

`coli plan` counts every buffer at the backend's least alignment, 256 bytes. A device that
aligns storage buffers more coarsely makes the engine count more: Mesa's Dozen (the Iris Xe
through D3D12) aligns them to 64 KiB, and its fit counted 1,379,048 bytes for the six-layer
fixture's first layer, which Lavapipe and the plan count at 159,720. On a real model, whose
matrices are megabytes each, that is a few 64 KiB per matrix; on such a device the plan can
predict a layer more than the engine places.

**The tier** sizes itself after the chain's uploads, as before, and takes what is left.

| Engine | What the handoff moves | On the partial chain |
|---|---|---|
| deepseek_v4 | the hc_mult streams; DSpark's taps of the last three layers come from whichever side ran them | yes |
| qwen36 (Qwen3.6, Qwen3-Coder, Qwen3.8-27B, Clef) | the residual rows alone; a prompt-lookup verify rolls each side back with its own copies (the device's DeltaNet slots, the CPU's snapshots) ([qwen36](#the-dense-chain-vk_chainc)) | yes |
| olmoe | the residual rows alone; `PILOT` keeps prefetching the next layers from the chain's rows ([olmoe](#olmoe-and-inkling)) | yes |
| qwen38 | the four hyper-connection streams of every row; the MTP head reads the final streams from whichever side ran the last layer, and the PLE ring and n-gram history stay with the PLE layer's side | yes |
| colibri (GLM-5.2) | the residual rows; when layer N is a shared DSA indexer layer, the selection the device's last full layer made (1 + `index_topk` ints a row) | yes ([GLM](#glm-52-and-glm-53-flash-on-the-chain)) |
| glm53 (GLM-5.3 Flash) | the hc_mult streams | yes ([GLM](#glm-52-and-glm-53-flash-on-the-chain)) |
| deepseek_v41 | the hc_mult streams and the last site's mix; the candidate mask and the published index list a chain layer made for the CPU's layers; DSpark's target means from whichever side ran them | yes |
| kimi_k3 | the AttnRes prefix, the block snapshots and their count; with every layer but not the head, the final rows for the CPU's head | yes |

**`COLI_VK_DEVICE_CAP_MB=n`** (tests) limits the device to n MiB of device
memory (a fraction is taken): every allocation of the backend and the chain (tensors, the
tier's pool, the chain's buffers, staging, the second device's) is counted, one past the
cap fails as out of device memory would, and every budget the engines read reports the cap.
Under it the pools take small blocks, so their granularity does not decide what fits. With
it Lavapipe behaves like a small card.

**Tests.** `tests/vulkan_engines.sh partial-<group>` (one file per engine group,
`tests/vulkan_partial_<group>.sh`; `partial-deepseek` for DeepSeek V4) and its
`-sanitize` variant (ASan and UBSan). On Lavapipe, DeepSeek V4: `COLI_VK_CHAIN_LAYERS` at 0,
1, 3, 5 and 6 of a six-layer fixture and on the other test geometries (the CPU's tokens,
logits within V4's bound, the placed bytes the kept layers' exactly, nothing more on the
device at exit than at setup); `COLI_VK_DEVICE_CAP_MB` aimed at 0, 1, 2, 3 and 5 layers
from a probe's numbers (the line's N and `coli plan`'s are the aimed one); an upload
failing inside layers 0, 1, 2 and 4 (N = that layer); prompt chunks, prefill blocks, expert
streaming, n-gram drafts accepted and rejected, prompts only, the per-matrix path beside,
the KV split, a lost device mid-decode, in the prompt and between drafts, serve sessions
and the prefix-reuse tests, and the chain against itself (chunks, prefill blocks, drafts)
with N < L; the dense weights on the device only with N < L. A discrete GPU has not been
measured: none is available here, so the fit's behaviour on one (the budget a real driver
reports, its allocation granularity) is not verified.

`partial-dsk` does the same for DeepSeek V4.1 Flash and Kimi K3 on Lavapipe:
`COLI_VK_CHAIN_LAYERS` at every k of their six-layer fixtures (V4.1's cuts at 2 and 4 hand
the CPU's layers the index list and the candidate mask; Kimi K3's at 1, 3 and 5 cut inside
an AttnRes block of two layers); `COLI_VK_DEVICE_CAP_MB` aimed at 0, 1, 3 and 5 layers and
at every layer but the head (the line's N and `coli plan`'s, with the same free, per-layer
and fixed bytes); an upload failing inside layers 0, 2 and 4, and with the dense weights on
the device only inside layer 3, where nothing is read back from disk; prompt chunks, expert
streaming, DSpark drafts (V4.1), prompts only, the tier off, the per-matrix path beside,
the KV split, a lost device (V4.1: the forward again on the CPU; Kimi K3: the chain layers'
KDA state rebuilt from the prefix record), serve sessions with prefix reuse, Kimi K3's
recurrent-state photos and V4.1's images with N < L; the dense weights on the device only
with N < L.

### Layers on two devices

With a second GPU (`COLI_VK_DEV2`) the layers the primary device leaves do not have to go
to the CPU: a second chain takes them on that device. The primary keeps its first N
layers (its fit, as above); the second device's chain takes the layers from N on with a
fit of its own over that device's free memory (`COLI_VK_CHAIN_LAYERS2` forces how many);
the CPU runs what is left, and the head stays on the host. Every chain engine does it:
qwen36, qwen38, OLMoE, MiMo, Inkling, GLM-5.2 (colibri), GLM-5.3, Kimi K3, DeepSeek V4.1
and DeepSeek V4. It needs a partial chain on the primary: when every layer fits there,
the second device holds experts only, as before.

A forward runs every row through the primary's layers, brings them back, and runs them
through the second device's layers from there: what crosses is what the CPU's next layer
would have read. That is the residual (the hc_mult streams on the mHC engines; Kimi K3's
AttnRes prefix, block snapshots and their count), and on GLM-5.2 the DSA selection of the
primary's last full layer when the second device's first layer shares its indexer. The
second device's matrices are copies of its own, which the per-matrix path never reads.
Each device's KV mirror, KV split and watermarks are its own.

Recurrent state (qwen36's and qwen38's DeltaNet, Inkling's convolution rings, GLM-5.3's
and Kimi K3's KDA) stays where its layers run. The host's copy is made current for both
devices or for neither, the second device's read first, so a read that fails leaves the
primary's as it was.

DeepSeek V4.1's second device starts only at a layer that reads nothing the layers
before it make in a forward. Its first compressed layer owns its compressed rows and runs
its own indexer, and no candidate mask crosses. On V4.1 Flash those are layers 2, 8, 14
and 20. The fit comes down to the last such layer, with a line that says so; a forced
`COLI_VK_CHAIN_LAYERS` that is not one stays, and the second device stays off.

The lines (Kimi K3's six-layer fixture, Lavapipe opened twice: `COLI_VK_DEV2=0`):

```
[VK] kimi_k3 chain: 2 of 6 layers on the device (0.5 MiB), 4 on the CPU, the head and what goes with it (0.2 MiB) on the CPU (COLI_VK_CHAIN_LAYERS=2)
[VK] kimi_k3 dev2 chain fit: free 23872688128 B, reserve 1073741824 B, fixed 489255168 B (...), tail 0 B, layers 190280 304512 304512 304512 B, ...
[VK] kimi_k3 dev2 chain: 4 of 4 layers on the device (1.1 MiB), 0 on the CPU (COLI_VK_CHAIN_LAYERS2=4)
[VK] kimi_k3 chain: layers 2..5 on the second device (1 KDA, 3 MLA, 0 dense MLP), 3 AttnRes blocks, ...
[VK] kimi_k3 dev2 chain: 8 forwards, 41 frames (...), ...
```

**A lost second device.** Losing either device turns both chains off. An engine whose
state is the host's (OLMoE, MiMo, GLM-5.2, DeepSeek V4.1 and V4) has the CPU run the
forward again, or run the second device's layers from the primary's output. An engine with
recurrent state rebuilds the state its devices held from its record, as on one device.
`COLI_VK_CHAIN_FAULT2=n` fakes the loss at the second device's n-th frame; frame 1 is its
setup, whose layers then stay on the CPU from the start.

`COLI_VK_CHAIN_DEV2=0` keeps the layers off the second device (its experts stay there).

Tested on Lavapipe only, opened twice: every split of every engine's fixtures against its
own CPU run. The families are `layers-dev2` (qwen36, qwen38, OLMoE, MiMo, Inkling),
`layers-dev2-mla` (colibri, GLM-5.3, Kimi K3) and `layers-dev2-deepseek`, each also under
ASan and UBSan. They cover:

- the tokens, and every logits row within each engine's chain tolerance;
- prompt chunks, drafts and MTP accepted and rejected;
- prompts only, the KV split on both devices, experts on both devices;
- the second device lost at its setup, in a prompt and mid-decode;
- serve sessions with pins and the prompt cache.

Two real GPUs have not been measured: none is available here. So whether a forward is
faster than the same layers on one device and the CPU depends on the card and the link,
and is not verified. The rows cross through host memory once per forward, chunk by chunk.

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
- Big prompt chunks and expert streaming: `make vk-tier-check VK=1` (`test_vk_tier`'s
  `stream` section) runs big steps through the sub-batches against its CPU reference in
  six source formats and all three activations (DeepSeek V4's roundings bit for bit):
  experts cut into parts, the staging slots filled through both hooks, prefetch, cold
  experts below the rule's rows left to the CPU, and the resident set the warm start's
  before and after. `vk-chain-check` runs the blocked attention against the same cases
  as `chain_attn` (windows, rings, position-major rows, sinks, lists), and every
  attention op cut in slices. `tests/vulkan_engines.sh prefill-qwen`,
  `prefill-inkling-olmoe`, `prefill-mimo-kimi`, `prefill-glm` and `prefill-deepseek`
  run each chain engine on a prompt past its usual block (110 to 300 tokens), four ways:
  the chunk from the budget with the rule's streaming; a forced chunk of 40 to 100 rows
  with a forced threshold of 3 rows, four slots and sub-batches of 24 rows; the same
  staged; the feature off (`COLI_VK_CHAIN_ROWS=512 COLI_VK_TIER_STREAM=0
  COLI_VK_ATTN_BLOCK=0`). Each one gives the CPU's tokens and logits, the streamed runs
  show streamed experts, and a run over several chunks shows prefetched experts used.
  Also qwen36's device lost while a prompt's sub-batches are in flight (the state
  rebuilt on the CPU). The forced runs run again under ASan and UBSan, staged and
  mapped (`prefill-qwen-sanitize` and the families' own sanitized halves).
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
- On a discrete card without Resizable BAR, resident data (the dense weights, the
  expert tier, the KV mirror) goes through staged uploads
  ([above](#memory-placement-without-resizable-bar)); that path is tested by forcing
  it and by emulating the small window, and not yet measured on such a card.
- Streaming copies each assignment's input row into the scratch and each output row
  out of it (the expert batch's layout). A gather of the chunk's rows by index in the
  GEMM shaders would save both copies; it is not written.
- Without the dense chain, DSA top-k selection, ragged multi-slot serving, and
  quantized-KV caches fall back to the CPU attention path; with it, the DSA selection
  and colibri's multi-slot serving run on the device.
- Not yet done: a fully resident-layer pipeline for the engines other than qwen36,
  qwen38, colibri and glm53 ([the dense chain](#the-dense-chain-vk_chainc) is theirs), Polaris/gfx803 validation on real
  hardware (the shaders use dynamic subgroup sizes and are wave64-safe by
  construction). The cooperative-matrix GEMM is measured on RDNA3 only.
