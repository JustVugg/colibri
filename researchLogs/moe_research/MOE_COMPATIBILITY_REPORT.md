# MoE Compatibility and M4 Hardware Assessment

## Executive summary

**Host:** Apple M4 MacBook Air, 16 GiB unified memory (17,179,869,184 bytes). CPU and
GPU draw from the same memory pool; “GPU memory” is not an additional 16 GiB. The
host’s currently available memory varies with macOS and other applications, so
physical capacity is not the same as a safe engine budget.

**Best practical starting point:** Qwen3-Coder-30B-A3B-Instruct in the local
int4-gs64 container, with an intentionally small expert cache. The repository
documents 6.5 GB resident at cache cap 32, measured on a Linux system, not this
Mac. A cap of 16 is a conservative starting estimate for 16 GiB, not an M4
measurement. Its expert path is not accelerated by the branch’s native
IQ2_XXS Metal GEMV.

**Best experiment for the current branch’s Metal change:** Flint 40B, using
`flint-UD-IQ2_XXS.gguf` and Qwen36’s opt-in IQ2_XXS Metal expert GEMVs. This is
the only identified candidate that directly exercises this branch-specific
routed-expert path. It is not a full-GPU MoE path: only eligible IQ2_XXS
projections on single-row decode dispatch to Metal; other formats/operations
remain CPU-side. The candidate is aggressively quantized, and the repository
reports a quality penalty for this file. No Flint M4 inference has been run.

**Largest-model stretch option:** DeepSeek V4 Flash REAP 150B (85 GB on disk)
uses a streaming planner and is architecturally more plausible than models
whose dense weights alone exceed memory. It still has no M4/16-GiB validation;
the planner’s minimum expert working set, runtime reserve, and available-memory
check make fit conditional. Its cold routed-expert traffic is several GB per
token, so fitting is not the same as being interactive.

**Do not use as evidence of MoE support:** the existing local Qwen3.6-27B
IQ2_XXS Metal validation is a dense model with zero experts. It verifies the
GEMV implementation and host behavior, but says nothing about routing,
expert-cache behavior, or end-to-end MoE speed.

## Scope and evidence standard

This is a read-only assessment of the current checkout and public model
metadata. Local documentation and engine/source paths are treated as evidence
of implementation; external parameter counts and file sizes are attributed to
the named public model repositories. A family having a loader, a format being
decodable, or a Metal kernel existing does not by itself prove a particular
checkpoint’s routed experts use that backend.

Labels used below:

- **Verified locally:** documented or established by the current local source,
  configuration, or prior local report.
- **External metadata:** public repository configuration/file metadata checked
  without fetching model weights.
- **Estimate:** arithmetic or a hardware judgment, not a measured result.
- **Unknown:** not established by this checkout’s evidence.

No model weights were downloaded, no model was converted or run, and no build,
compile, benchmark, or inference was performed for this assessment. The
commands in the experiment section are proposals only.

## Checkout and host

- Repository: `JustVugg/colibri`
- Branch: `feature-IQ2_XXS`
- HEAD: `c92e16eda788d435b630b93e77ca6aa9d09a4ec8`
- Existing user changes observed before report creation: modified
  `c/Makefile`, `c/qwen36.c`, `c/qwen36_mtp.h`; untracked `.meta-store/`,
  `c/tests/test_qwen36_phase2_stats.c`, `c/tools/gguf_inspect.c`, and `logs/`.
  These are not part of this research and must be preserved.
- `logs/moe_research/MOE_COMPATIBILITY_REPORT.md` was absent before this
  report was created.
- Host memory was confirmed as 16 GiB unified memory. There is no separate
  discrete GPU-memory budget.

## What “fits” means on this machine

For an MoE model, total parameter count and the file’s disk size are not RAM
requirements by themselves. A useful budget must include:

1. Resident dense weights (and any resident shared experts, embeddings, and
   output head).
2. Routed-expert cache capacity. Where the engine needs a full routing step
   resident, that is at least `top-k × sparse layers × bytes per expert`;
   some engines can process smaller batches/rounds, while others cannot.
3. Attention/KV or recurrent state at the selected context, temporary
   activations/workspace, tokenizer/runtime allocations, and the OS.
4. Unified-memory pressure from Metal and macOS. A cache that fits the
   theoretical physical total can still cause compression or memory pressure.

File sizes below use the units reported by the cited repository/docs. Decimal
GB and binary GiB are not interchangeable. Estimates are deliberately not
presented as measured M4 RSS or throughput.

### Hardware-level conclusion

16 GiB is sufficient to investigate MoE inference with a bounded cache; it
does not make the entire expert bank resident for most large models. SSD
streaming is possible only where the engine actually implements it, and
decode performance can then be dominated by expert reads. On Apple Silicon,
Metal access to shared memory avoids a PCIe copy but does not expand RAM.
Reducing expert cache size lowers residency and may increase disk reads; it
does not reduce the model’s total stored weights.

## Candidate comparison

| Candidate / local engine | Architecture and checkpoint facts | Storage and memory implications | M4 / backend fit |
|---|---|---|---|
| **Qwen3-Coder-30B-A3B-Instruct** / `c/qwen36.c` | 30B total, about 3B active; 48 attention layers; 128 routed experts, top-8, no shared expert. **Verified locally** in `docs/qwen36.md` and the model-type/config path. | Official BF16 snapshot 61.1 GB; converted int4-gs64 container 19 GB. Repository’s Linux run reports 6.5 GB resident at cap 32, 5.1 tok/s on a Ryzen 7 PRO 8700GE, and 15.2 GB at cap 128. These are not M4 measurements. A rough cap-16 resident estimate is about 5 GB by linear interpolation of the published cap-32/cap-128 measurements; use only as a planning estimate. | Best balanced practical candidate. `--cap` is the expert-cache control; `--ram` is not honored by qwen36. Its int4-gs64 expert path is not the raw IQ2_XXS path, so do not expect the branch’s native IQ2_XXS expert Metal GEMV. |
| **Flint 40B** / `c/qwen36.c` | 40B total, 2.5B active; 49 attention layers (two dense, 47 MoE); 256 routed experts, top-6, plus an ungated shared expert. **Verified locally** in `docs/qwen36.md`. | Public `unsloth/flint-GGUF` metadata lists `flint-UD-IQ2_XXS.gguf` at 11,855,541,696 bytes (about 11.86 GB decimal). The file is not the process RSS: dense weights, expert cache, state, and workspace remain. No M4 RSS was found. | **Most direct Metal experiment.** qwen36 streams GGUF experts through its LRU. Metal can run only the routed matrix projections that are IQ2_XXS and eligible single-row decode operations; mixed/other GGUF types fall back to CPU. The repository reports NLL 1.045 for UD-IQ2_XXS on a specific 2,048-token text, versus 0.534 for Flint BF16; this is a quality warning, not a general quality score. |
| **OLMoE-1B-7B** / `c/olmoe.c` | 7B total / about 1B active; local GGUF/import path exists. `c/olmoe.c` reads expert counts/top-k from config rather than hardcoding one checkpoint geometry. Public parity docs identify the OLMoE-1B-7B Q4_K_M file. | Q4_K_M GGUF about 4 GB. Source notes describe an expert set of about 6.5 GB for this model; dense weights and context add to that. The engine has RAM-aware cache planning and on-demand `pread`/LRU expert loading. | Lowest-risk small-model baseline, but not a large-total-parameter choice. CPU/Vulkan paths are present; no routed-expert Metal dispatch is wired in `olmoe.c`. |
| **Qwen3.6-35B-A3B** / `c/qwen36.c` | 35B total / ~3B active; 40 layers, 256 experts top-8 plus shared expert, hybrid attention/DeltaNet. | Converted int4-gs64 container about 23 GB; repository quickstart examples say it occupies about 20 GB when run with the full model/cache on a 27 GB machine. 16 GiB is not a safe full-residency target. | Supported and streamable, but no direct implication that its converted int4 experts use IQ2_XXS Metal. A separate local 27B dense GGUF validation is not evidence for this MoE model. |
| **Qwen3.8-Flash-Next** / `c/qwen38.c` | 125B ordinary parameters / 6B active, plus 51B PLE; 48 layers, 512 routed experts top-10 and one shared expert. | Official model repository about 185.5 GB; fixed native-BF16 resident weights 9.2 GiB, plus 28 MiB scale bank, 226 MiB recurrent/PLE state, 54 KiB per context token, up to 1.1 GiB workspace, and routed cache. Docs explicitly say 16 GB is below the useful-cache practical floor. | Native FP8/safetensors path and optional int4-g64 expert sidecar. No Qwen38 Metal MoE implementation or Metal link in the `qwen38` target was found. Not recommended for 16 GiB. |
| **Qwen3.8-2.4T-A95B** / `c/qwen36.c` geometry | 2.4T total / 95B active; 92 layers, 512 experts top-10. | Local docs estimate about 1.4 TB of int4 experts for the all-expert warm start and explicitly distinguish this from a disk-streaming design. | Geometry resolves, but that is not practical support for this host: the engine’s warm-start behavior defeats a 16-GiB fit. |
| **DeepSeek V4 Flash REAP 150B** / `c/deepseek_v4.c` | Expert-pruned 43-layer checkpoint with 132 experts per layer; same engine family as official V4 Flash. | 85 GB on disk; docs report about 76 GB of experts. The resource planner reserves runtime/system memory and a minimum working set, then can omit dense/head residency if they do not fit. At top-6 across 43 sparse layers and roughly 12.6 MB per expert, a cold step can request roughly 3.0 GiB of expert bytes (estimate; cache hits reduce SSD reads). | Largest plausible **conditional** streaming experiment in this comparison. CPU-first; official docs list arm64 macOS CPU support. No Metal expert path. A 16-GiB fit is not verified and depends on OS-available memory, context, planner inputs, and disk. Expect a storage-bound workload. |
| **DeepSeek V4 Flash (official)** / `c/deepseek_v4.c` | 284B official checkpoint; 43 layers; 256 routed experts plus one shared expert, top-6. | 167 GB on disk; about 6.27 GiB dense weights, ~1.06 GiB BF16 output head, and ~137 GiB routed experts. The V4 planner can stream experts and may leave dense/head weights nonresident, but it requires runtime reserves and at least the routed top-k working set. | Supported CPU streaming path, but no assurance the planner will accept this specific 16-GiB host. M4 throughput is unknown; expert traffic and memory pressure make it a stretch candidate, not a first experiment. |
| **GLM-5.2** / `c/colibri.c` | 744B MoE; local `glm` family engine and on-demand expert cache. | Quickstart cites roughly 430 GB for its int4 model. An OLMoE source comment describes the RAM-cache strategy as making GLM-5.2 fit within a 15-GB total-RAM target; it is not a resident-dense-only measurement or an M4 result. Cache/context/OS remain part of the budget. | Technically supported, but not a sensible first experiment on 16 GiB. Generic Metal MoE is wired for supported internal packed formats, not a blanket accelerator for every checkpoint format. |
| **GLM-5.3-Flash** / `c/glm53.c` | 321B total; 45 layers plus MTP; 42 sparse layers, top-8 routed experts (plus shared expert per source config). | Converted model 194.7 GB; dense set is 9.7B BF16 parameters (~18 GB in the release); docs report ~12 GB resident when `GLM53_BITS=4`, before cache/context/system. One token’s nominal cold expert demand is ~4.8 GB. | Batched Metal MoE is wired in `glm53.c` for eligible containers, with CPU fallback. At 16 GiB, dense + cache + macOS leaves little margin; not a first fit recommendation. |
| **Mistral-Large-4-1T-A52B** / `c/colibri.c` | 1T total / 52B active; 72 layers; 128 routed experts, top-4, plus one shared expert. | Smallest listed unsloth UD-IQ1_S is 233 GB; the locally documented dense part is 17.9 GB (about 16.7 GiB), already above physical memory before OS/cache. | GGUF is supported by the GLM-5.2 engine family, but the dense footprint rules it out on this host. Do not infer GGUF experts use Metal’s internal packed-format path. |
| **Kimi K3** / `c/kimi_k3.c` | 2.8T total / 104B active; 93 layers; 896 routed experts, top-16, and two full-width shared experts. | Routed MXFP4 experts: about 1.45 TB total, 17.55 MB each; dense tensors are also substantial. The cold expert demand implied by 16 experts across 93 layers is tens of GB per token. | The source’s `moe_forward` does not call the generic `coli_metal_moe_*` expert API; routed MXFP4 is a Vulkan path, not Metal. Prior Kimi Metal documents contain aspirational/contradictory status text; current source wiring is the controlling evidence. Not viable for 16 GiB. |
| **MiMo-V2.6 Flash / Pro** / `c/mimo.c` | Flash 309B / 15B active, 48 layers, 256 experts top-8; Pro 1.02T / 42B active, 70 layers, 384 experts top-8. | Flash 177.7 GB, Pro 573.5 GB on disk. Released MXFP4 experts are streamed through per-layer RAM LRU; cold Flash expert demand is about 5 GB per token per local docs. | CPU/Vulkan paths; no wired Metal expert path found in `mimo.c`. Flash is storage-bound at a small cache; Pro is far outside host memory. |
| **Inkling** / `c/inkling.c` | 975B total / 41B active; local expert-streaming engine. | Original BF16 resident set 49.4 GB (peaks near 99 GB while expanded); even the dense-int4 path is 15.3 GB decimal, before cache/context/OS. Routed expert bank is about 464 GB. | `inkling.c` does wire batched Metal MoE, but the 16-GiB machine cannot safely hold its documented resident weights plus working set. Metal support does not fix that memory floor. |
| **Muse Spark Open 816B-A42B** / `c/qwen36.c` | 816B total / 42B active; 62 attention layers; 256 experts top-8. | The smallest documented UD-IQ1_S set is 175.6 GB. Its dense part alone is 17.3 GB decimal, larger than 16 GiB. | Qwen36 GGUF path exists; dense residency excludes this host. Only eligible IQ2_XXS projections could use the qwen36 GEMV path; no evidence of a complete Metal MoE path. |
| **OLMoE plus other GGUF/import variants** | The GGUF registry and engines are architecture-gated; local support is not universal GGUF support. | Model-specific metadata/format must match the engine’s loader. | `docs/gguf.md` lists supported architectures and engine routes. A recognized architecture or decodable dense block does not guarantee the routed-expert format is accepted by Metal/Vulkan. |

### Other locally supported routes and explicit limits

- **Qwen3.6-27B:** dense, not MoE. Local IQ2_XXS/Metal reports show successful
  M4 inference and roughly 8.4–8.95 GB peak RSS for that particular dense
  checkpoint. Do not extrapolate its RSS/speed to an MoE cache.
- **Qwen3-Coder-REAP-25B-A3B:** geometry is named in local docs and resolves to
  qwen36; the same docs say this variant was not run locally.
- **Qwen3.5/Qwen3.6 MoE GGUF and Qwen3-Coder GGUF:** `docs/gguf.md` maps
  `qwen35moe` and `qwen3moe` to qwen36. Model-specific tensors and format still
  need to pass the loader’s shape and type checks.
- **GLM-5.2/Mistral-Large-4 GGUF, OLMoE, Muse Spark, and Qwen3.8 dense GGUF:**
  listed in the import registry; only the routed models above are relevant to
  this MoE comparison.
- Local engines also include **DeepSeek V4.1 Flash** (`c/deepseek_v41.c`):
  552B total, 510 GB on disk, ~289 GB routed experts, and ~18 GB dense weights
  (docs). Dense weights alone exceed the Mac’s physical 16 GiB. Not a candidate.
- **Flint** is an additional qwen36 model type, not a separate engine. Its
  architecture and GGUF importer support are documented under “Flint” in
  `docs/qwen36.md`.

## Metal, CPU, and cache compatibility

### What the current source actually wires

| Path | Routed expert execution on this branch | What that does and does not establish |
|---|---|---|
| Generic `c/colibri.c` Metal path | Batched zero-copy expert blocks through `coli_metal_moe_block*`; see `moe()` and its batch-building path. | `backend_metal.mm::moe_submit_impl` accepts internal packed formats 1, 2, 4, 5, and 6. This is not a generic decoder for every raw GGUF block type. Metal shares unified memory with the CPU. |
| GLM-5.3 | Batched Metal MoE call in `glm53.c` (`coli_metal_moe_block_clamped`) with CPU fallback. | Requires the loaded expert representation, activation/clamp, and pointer registration to satisfy backend checks. |
| Inkling | Batched Metal MoE calls in `inkling.c::moe_ex`. | Implements supported internal quantized experts; this does not make its resident footprint fit 16 GiB. |
| Qwen36 raw GGUF | `q36_iq2_xxs_metal_try*` plus calls in `moe_gq_decode`; only `GQ_IQ2_XXS` and one-row (`S==1`) GEMV qualify. | This is the branch-specific experiment path. It is not batched expert execution, and non-IQ2_XXS projections remain CPU-side. The engine retains explicit CPU fallback. |
| Kimi K3 | Source `moe_forward` computes routed experts without a call to `coli_metal_moe_*`. | The local Metal planning document is not proof of working MoE dispatch. `mxfp4` is listed as Vulkan-only in `docs/FORMATS.md`. |
| Qwen3.8 | No Qwen38 routed-expert Metal dispatch or Metal object in its target was found. | CPU/native FP8 path plus the documented CUDA/Vulkan tiers; do not assume macOS Metal execution. |
| OLMoE, MiMo, DeepSeek V4/V4.1 | No routed-expert Metal dispatch found in their engine paths. | Their documented CPU, Vulkan, or CUDA routes do not imply Metal support. |

The generic Metal MoE backend works on registered/shared host slabs; it is not
a separate VRAM cache. For any of these models, Metal can increase memory
pressure without making a too-large resident set fit.

### Formats and activation shape

- `c/ggml_quants.h::gq_blk_decoder` is the CPU-side GGUF block decoder. The
  existence of a CPU decoder does not establish that the internal Metal MoE
  kernel accepts that GGUF type.
- qwen36’s native IQ2_XXS GEMV gate checks all of: `GQ_IQ2_XXS`, `S==1`,
  runtime opt-in, and integer-kernel geometry. `moe_gq_decode` calls it for
  eligible gate/up and down projections. A mixed GGUF may use Metal for only a
  subset of its projections.
- qwen36 allocates/registers eligible IQ2_XXS slabs for Metal; on unsupported
  type, geometry, dispatch, or initialization, the path falls back to CPU.
- Generic Metal MoE dispatch has ~5 ms submit latency per the local Metal docs;
  its batched block path is designed to avoid one submission per projection.
  The qwen36 IQ2_XXS MoE path dispatches individual GEMVs, so dispatch overhead
  is a material experiment risk.
- `QWEN36_QPACK` (MLX-affine) and the standard qwen36 int4-gs64 container are
  separate paths. Neither should be described as the raw IQ2_XXS Metal path
  without the relevant tensor type and dispatch actually being established.

### Cache behavior

- **qwen36:** routed experts are demand-read from their container/GGUF and kept
  in an LRU; `--cap N` controls experts per layer. Its docs explicitly state
  `--ram` is ignored. Avoid setting a cap based on the entire model size.
- **DeepSeek V4:** source `coli_v4_resource_plan_compute` accounts for a system
  reserve, runtime reserve, top-k minimum expert working set, and available
  OS memory. The later resident-tier plan can leave dense weights and the head
  unresident rather than exceed its budget. The planner’s success is conditional
  on actual available memory, not merely physical RAM.
- **DeepSeek V4.1:** dense/vision weights remain resident; routed experts and
  engram tables stream. Its ~18 GB dense estimate alone makes it unsuitable
  for this host.
- **OLMoE:** uses on-demand `pread`, per-layer LRU and RAM-aware slot sizing;
  low cache capacity is correct but can be slow.
- **Qwen3.8:** routed expert cache is explicit and context/workspace are
  separately budgeted. Its fixed 9.2 GiB native-BF16 resident set leaves too
  little room for a useful expert cache on 16 GiB.
- **Mimo/Kimi/Inkling/GLM family:** caches and warm sets are engine-specific;
  a `--cap`, `--ram`, or Metal flag from one engine must not be generalized to
  another.

## Three rankings

These are engineering rankings for this host, not quality or speed benchmarks.

### 1. Practicality on 16 GiB

1. **Qwen3-Coder-30B-A3B int4-gs64, cap 16 initially** — best balance of model
   size, active parameters, and a documented low-cap resident measurement.
   The cap-16 number is estimated; check actual macOS memory pressure.
2. **OLMoE-1B-7B Q4_K_M** — least demanding baseline, but only 7B total; useful
   for verifying ordinary streamed MoE behavior, not the large-model goal.
3. **Flint 40B UD-IQ2_XXS** — plausible from its quantized GGUF and cache
   design, but exact M4 resident use is unknown and quality is a concern.
4. **DeepSeek V4 Flash REAP 150B** — conditional disk-streaming stretch only;
   planner acceptance and practical speed on this host are unknown.
5. **Qwen3.6-35B-A3B, GLM-5.3-Flash, official DeepSeek V4 Flash** — potentially
   streamable, but no evidence justifies calling them safe 16-GiB choices.
6. **Qwen3.8 Flash-Next, Qwen3.8-2.4T, GLM-5.2, DeepSeek V4.1, Kimi K3,
   MiMo Pro, Inkling, Muse Spark, Mistral Large 4** — excluded or not
   recommended due fixed residency, cache floor, or expert-bank scale.

### 2. Best candidate for a branch-specific M4 Metal experiment

1. **Flint UD-IQ2_XXS** — actual qwen36 raw-IQ2_XXS routed projection path;
   low cache can be bounded, but mixed-type fallback and quantization quality
   must be measured.
2. **Qwen3-Coder int4-gs64** — strongest practical non-Metal control candidate;
   useful to characterize CPU/cache behavior but does not test the special
   IQ2_XXS expert kernel.
3. **OLMoE Q4_K_M** — small engine and baseline; no routed-expert Metal path.
4. **GLM-5.3 / Inkling internal packed experts** — batched Metal code is wired,
   but their memory requirements make them poor 16-GiB first experiments.

### 3. Largest total model worth investigating after the practical baseline

1. **DeepSeek V4 Flash REAP 150B** — 85 GB disk checkpoint and explicit
   streaming/resource planner, but M4 fit and token speed remain unknown.
2. **Official DeepSeek V4 Flash, 284B** — source has a conditional streaming
   planner, but minimum working set plus runtime/system reserves may exceed
   available memory and cold traffic is high.
3. **GLM-5.2, 744B** — engine supports disk-streamed experts, but its documented
   15-GB total-RAM target leaves little safe 16-GiB headroom.
4. **All larger fixed-resident/large-bank candidates** — not viable under the
   current host budget; see the candidate table.

## Proposed first experiment (not run)

**Question:** On the M4, does qwen36’s existing native IQ2_XXS Metal path
improve routed-expert decode for an actual supported MoE checkpoint, and what
fraction of expert projections dispatch versus fall back?

**Checkpoint:** `unsloth/flint-GGUF` → `flint-UD-IQ2_XXS.gguf`
(public metadata size 11,855,541,696 bytes). Use `--cap 8`, which is above
Flint’s top-6 routing requirement while keeping the per-layer cache bounded.
This cache choice is a conservative test setting, not a demonstrated
performance optimum.

**Proposed commands, intentionally not executed:**

```sh
hf download unsloth/flint-GGUF flint-UD-IQ2_XXS.gguf \
  --local-dir ~/Models/flint-gguf
./c/coli import --out ~/Models/flint-iq2 \
  ~/Models/flint-gguf/flint-UD-IQ2_XXS.gguf
make -C c qwen36 METAL=1

# CPU control; use a fixed prompt and the same generation limit for both runs.
COLI_METAL_IQ2_XXS=0 ./c/coli chat --model ~/Models/flint-iq2 --cap 8

# Metal treatment.
COLI_METAL_IQ2_XXS=1 ./c/coli chat --model ~/Models/flint-iq2 --cap 8
```

In the eventual experiment, record exact model/importer banner and tensor
types, macOS version, memory pressure and peak RSS, prompt/context, cap,
time-to-first-token, steady decode rate, and the qwen36 IQ2_XXS Metal success
versus CPU-fallback counters. Run cold-cache and warm-cache trials separately;
alternate CPU/Metal order to reduce thermal/order bias. Compare greedy output
for behavioral changes, but do not assume output identity is a substitute for
coverage or performance measurement. If the imported Flint checkpoint does
not expose eligible IQ2_XXS expert matrices, stop: the treatment would not be
testing the intended route.

**Quality caveat:** the local Flint quantization table reports NLL 1.045 for
UD-IQ2_XXS on its specified familiar text, versus 0.534 for BF16. If output
quality is more important than testing the new Metal path, use
Qwen3-Coder-30B-A3B int4-gs64 as a separate CPU/cache experiment instead; it
does not exercise the qwen36 IQ2_XXS expert Metal dispatch.

## Source and documentation anchors

### Core support and dispatch

- `c/qwen36.c`: `q36_iq2_metal_enabled` (~131),
  `q36_iq2_gq_alloc` (~177), `q36_iq2_xxs_metal_try*` (~2245),
  `load_cfg` (~2534), `expert_fetch`/`expert_get` (~4801/4858),
  `moe_gq_decode` (~5709), `moe_gq_run` (~5800), `moe_muse` (~6148).
- `c/ggml_quants.h`: GGUF block decoder and CPU matmul paths
  (`gq_blk_decoder`, `gq_matmul_f32`).
- `c/backend_metal.mm`: `moe_submit_impl` (~2157), which checks the internal
  packed format allowlist; `coli_metal_moe_block*` wrappers (~2290 onward).
- `c/colibri.c`: `expert_load_impl` (~3461), `moe_vk` (~6674), generic `moe`
  (~6755), Metal block submission (~7319, ~7378, ~7382).
- `c/glm53.c`: expert cache init (~1712), stable Metal-aligned slots
  (~2057–2107), batched Metal expert block (~3335–3375).
- `c/inkling.c`: `load_cfg` (~805), cache and `moe_ex` (~2513), batched
  Metal submission (~2647, ~2730–2788).
- `c/kimi_k3.c`: `load_cfg` (~947), `moe_forward` (~2898); routed MXFP4
  path does not call `coli_metal_moe_*`.
- `c/olmoe.c`: config loading (~646), RAM/cache planning (~826–942),
  `load_expert_gguf` (~1104), `moe_vk_run`/`moe_routed` (~1862–1943).
- `c/mimo.c`: config loading (~183), `moe_cpu` (~1384), `moe` (~1494).
- `c/deepseek_v4.c`: `coli_v4_resource_plan_compute` (~1355),
  `expert_record_bytes` (~1561), `build_runtime_plan` (~1615), runtime
  resident-tier selection (~1714 onward).
- `c/deepseek_v41.c`: config parsing (~175 onward), cache slots and
  `moe_run_at` (~3552).
- `c/qwen38.c`: expert memory accounting (`q38_segment_expert_layout`,
  `q38_expert_report` ~3220–3365) and CPU/native FP8 runtime path.
- `c/Makefile`: Metal is opt-in via `METAL=1`; qwen36, GLM-5.3, Inkling,
  Kimi K3, and generic colibri targets include the Metal object; qwen38 does
  not.

### Model and validation documentation

- `docs/qwen36.md`: Qwen3.6 model geometry (~1–20), family geometry table
  (~410–436), Qwen3-Coder facts and measurements (~511–566), Flint facts and
  quantization table (~568–630), Muse Spark (~632 onward), and explicit
  `--ram`/`--cap` semantics (~851–866).
- `docs/qwen38.md`: Qwen3.8-Flash-Next resident/cache/context/workspace budget
  and 16-GB caveat (~108 onward).
- `docs/deepseek-v4.md`: official V4 geometry and memory plan (~1–70), REAP
  150B size and behavior (~71–100), supported macOS CPU target (~51).
- `docs/deepseek-v41.md`: 552B/510-GB model breakdown and 18-GB dense set
  (~1–20).
- `docs/glm53-flash.md`: 321B architecture, conversion, resident-memory table,
  per-token expert traffic, and Vulkan path (~1–115).
- `docs/inkling.md`: 975B/41B geometry, 49.4-GB resident weights,
  dense-int4 15.3-GB container and cache discussion (~1–80); Metal/Vulkan
  expert tiers (~198–220).
- `docs/kimi_k3.md`: 2.8T/104B geometry, 896 experts top-16, MXFP4 weights,
  cache/streaming design (~1–75).
- `docs/mimo.md`: Flash/Pro geometry, file sizes and MXFP4 streaming (~1–60).
- `docs/mistral-large4.md`: 1T/52B geometry, GGUF sizes/types, and 17.9-GB
  dense-set measurement (~1–100).
- `docs/quickstart.md`: 16-GB minimum/24-GB recommended, model fit policy and
  27.3-GB example recommendations (~1–105).
- `docs/metal.md`: Metal behavior, supported batching, memory model, and
  measured results on different Apple hardware (~1–55).
- `docs/FORMATS.md`: internal quant format/backend mapping, including
  MXFP4’s Vulkan-only status (~60–75, ~130–145).
- `docs/gguf.md`: GGUF engine registry and model-specific format notes.
- `logs/iq2_validation/REPORT.md`: prior dense Qwen3.6-27B M4 IQ2_XXS report.
- `logs/phase2_mtp/REPORT.md`: prior dense-model M4 CPU/Metal comparison;
  explicitly notes absence of a loadable MTP head.
- Public metadata checked without fetching weights: [Qwen3-Coder
  checkpoint](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct),
  [Flint GGUF files](https://huggingface.co/unsloth/flint-GGUF), and
  [Qwen3.6-35B-A3B checkpoint](https://huggingface.co/Qwen/Qwen3.6-35B-A3B).

## Bottom line

Use Qwen3-Coder with a low cache as the practical MoE baseline; use Flint
UD-IQ2_XXS only when the primary goal is a focused test of this branch’s
M4-specific routed-expert Metal GEMV. Treat the 150B DeepSeek V4 REAP model as
a later SSD-streaming stress test, not a known 16-GiB fit. Existing dense
Qwen3.6 M4 results are useful hardware context but are not MoE validation.
