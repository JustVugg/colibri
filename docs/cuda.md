# CUDA backend (Linux)

colibrì includes an opt-in CUDA backend for model-resident tensors. Streaming
experts deliberately remain on the original CPU path: copying an expert from
NVMe to the GPU on every use would only replace the disk bottleneck with a PCIe
bottleneck. Resident quantized tensors are uploaded lazily once and reused.

```bash
cd c
make cuda-test CUDA=1                  # q8/q4/q2/f32 kernel correctness
make CUDA=1
# optional dense-path experiment (hot experts are configured below)
COLI_CUDA=1 COLI_GPU=0 CUDA_DENSE=1 SNAP=/nvme/glm52_i4 ./colibri 64 4 4
```

Requirements: Linux, an NVIDIA driver, and a CUDA Toolkit under
`/usr/local/cuda` (override with `CUDA_HOME=/path/to/cuda`).
`CUDA_ARCH=native` builds for the GPU in the current machine. Requesting CUDA
with a CPU-only binary, an invalid device, or an unavailable runtime fails at
startup instead of silently falling back. For Windows, see
[windows.md](windows.md) (runtime DLL path).

## The VRAM expert tier

A measured `PIN` profile promotes its hottest experts into a persistent VRAM
tier while keeping the rest in RAM:

```bash
STATS=stats.txt SNAP=/nvme/glm52_i4 ./colibri 64 4 4   # collect routing frequencies first
COLI_CUDA=1 COLI_GPU=0 CUDA_EXPERT_GB=16 \
PIN=stats.txt PIN_GB=160 SNAP=/nvme/glm52_i4 ./colibri 64 4 4

# multi-GPU expert tier, 150 GB total budget across six 32 GB devices
COLI_CUDA=1 COLI_GPUS=0,1,2,3,4,5 CUDA_EXPERT_GB=150 \
CUDA_DENSE=1 PIN=stats.txt PIN_GB=300 RAM_GB=226 \
SNAP=/nvme/glm52_i4 ./colibri 64 4 4

# large-RAM host: fill safe VRAM, then keep every remaining expert in RAM
COLI_CUDA=1 COLI_GPUS=0,1,2,3,4,5 CUDA_EXPERT_GB=auto \
CUDA_DENSE=1 COLI_CUDA_ATTN=1 PIN=stats.txt PIN_GB=all RAM_GB=auto \
SNAP=/nvme/glm52_i4 ./colibri 64 4 4
```

Selected experts are uploaded during startup, so capacity failures occur before
inference. The budget is clamped against free VRAM after reserving the projected
dense resident set and 2 GB of runtime headroom per device. With `COLI_GPUS`,
`CUDA_EXPERT_GB` is a total budget across the device set; experts are assigned
whole to the least-loaded device that can hold them. Multi-GPU runs default to
`PIN_FILL=1` (measured hot set first, then unused VRAM filled with zero-heat
experts) and `CUDA_RELEASE_HOST=1` (RAM copy released after upload, reloaded
from disk only if CUDA later fails).

`CUDA_EXPERT_GB=auto` fills each device up to measured free memory minus
projected dense tensors and headroom. `PIN_GB=all` then loads the remaining
routed experts into RAM **up to the `--ram` budget** (it clamps — [#229](https://github.com/JustVugg/colibri/issues/229)),
eliminating decode-time disk misses when capacity permits. This mode is intended
for dedicated high-memory inference hosts.

### Full-residency reference result (6× RTX 5090, 251 GiB host)

`CUDA_EXPERT_GB=auto PIN_GB=all` selected a 176.7 GB VRAM tier + 191.3 GB RAM
tier (all 19,456 experts resident), adapting the VRAM tier every 16 tokens.
With the GPU-resident pipeline (`COLI_CUDA_PIPE=2`) and Tensor-Core W4A16
dispatch (`COLI_CUDA_TC_W4A16=1`), 96-token greedy decode measured
**5.8–6.8 tok/s** (TTFT ~13 s; 1571-token prefill ~122 s then 4.2 tok/s).
Full experiment log: [experiments/glm52-6x5090-2026-07-12.md](experiments/glm52-6x5090-2026-07-12.md).
These are host-specific capacity results, not portable defaults.

## Experimental lossless compressed expert tier

An optional [DietGPU](https://github.com/facebookresearch/dietgpu) ANS tier keeps
warm int4 experts compressed in VRAM and decodes the selected tensors directly
to reusable device scratch before grouped expert matmul. It does not quantize,
prune, substitute, or move expert weights over PCIe during decode.

This path is disabled in normal builds. DietGPU must first be built as shared
libraries, then supplied explicitly:

```bash
make CUDA=1 COLI_ANS=1 DIETGPU_ROOT=/opt/dietgpu
```

Create the sidecar once using the same model, hot-expert profile, device set,
and placement budget intended for inference:

```bash
COLI_CUDA=1 COLI_GPUS=0,1,2,3 CUDA_EXPERT_GB=auto \
PIN=stats.txt PIN_GB=all CUDA_RAW_EXPERTS=2500 \
COLI_ANS_SIDECAR=/models/glm52-experts.ans COLI_ANS_PACK=1 \
SNAP=/models/glm52-int4 ./colibri 1 4 4
```

Remove `COLI_ANS_PACK` to load the sidecar. Records are checked against the
expected format, dimensions, raw size, codec bound, and file length. A sidecar
is tied to its exact expert placement order; changing the model, profile,
budgets, raw-prefix size, or device set requires repacking it.
On Linux, `COLI_ANS_DIRECT=1` uses aligned direct reads while retaining the same
sidecar format. `COLI_ANS_PROFILE=1` prints a one-line load-time breakdown.

On a 6× RTX 5090 host with GLM-5.2 int4, a 2,500-expert raw tier plus 8,128
compressed experts increased VRAM capacity from 9,335 to 10,628 experts
(+13.9%). The original controlled runs measured 6.19→7.12 tok/s over 32 tokens
(+15.0%), and 6.75→7.31 tok/s over 128 tokens (+8.3%). A later fixed-token
revalidation on current `dev` measured a smaller 8.17→8.54 tok/s median gain
(+4.5%) against the full 176.6 GB raw tier. Pinned asynchronous staging reduced
placement from 197 to 157 seconds; aligned direct reads reduced it further to
80–103 seconds across two runs, with the 110 GB archive read accounting for
18.94–19.23 seconds.

The archive itself is lossless: every decoded weight byte is verified against
the original tensor and incompressible records are rejected. That does **not**
guarantee identical generated text versus the raw-tier placement. Moving more
experts from CPU execution into CUDA changes floating-point accumulation order;
a current greedy A/B diverged after a near tie even though both weight paths
were byte-exact. Treat output identity as a measured workload result, not a
property of the compression format. The archived DietGPU dependency therefore
remains an experimental build-time option. Live `REPIN` is disabled because it
would invalidate the sidecar's fixed expert order.

## The GPU-resident pipeline (`COLI_CUDA_PIPE`)

`COLI_CUDA_PIPE=2` keeps the residual stream on-device across layers: rmsnorms,
residual adds, router GEMMs and the shared expert run on the GPU while the CPU
expert loop runs uninterrupted, with batched attention and grouped expert
uploads at prefill. On a single-GPU host this also pays at decode (S=1):
**+49%** measured on a 5070 Ti ([#273](https://github.com/JustVugg/colibri/issues/273)/#274);
on multi-GPU hosts the per-layer P2P hops cancel the gain, so the decode gate is
device-count aware. `COLI_CUDA_TC_W4A16=1` enables Tensor-Core int4×fp16 mixed
dispatch for batched rows (pays at ≥16 rows).

qwen36 has its own form of this, shaped like the Vulkan chain: `COLI_CUDA_CHAIN=1`
runs every layer as one chain on the card (`c/cuda_chain.h`, `c/qwen36_cuda_chain.h`;
docs/qwen36-cuda-tier.md, "The dense chain on the card").

## The engines' routed-expert tier on CUDA (`cu_tier`)

The routed-expert tier the Vulkan engines share (`c/vk_tier.c`, [vulkan.md](vulkan.md#the-routed-expert-tier-vk_tierc)) also runs on a CUDA device: `c/cu_tier.c` compiles the same file again over a CUDA backend (`c/cu_xb.cu`, through the table in `c/cu_tier_ops.h`), so the history, the warm start, the promotions, the host pool and the hybrid steps are the Vulkan tier's, and its lines read `[CUDA] tier <engine>`. It takes the same `COLI_VK_TIER*` settings.

- **Formats:** colibri's (int8 a row or per group, int4 pairs, int3-g64, MXFP4, FP8, bf16, f32) and a GGUF's block types as the file has them, through the same block decoders as `cuda_bd.cuh`'s GEMVs, in the CPU's integer form (`GQ_EXACT=1`: the float form).
- **The host pool:** the experts the card leaves sit in mapped page-locked RAM that the card reads over the bus; the CPU computes a small step's pairs from it in place (the hybrid, `COLI_VK_HYBRID=0` off).
- **Not on CUDA yet:** the dense chain's expert table, a second device, streaming a prompt's cold experts, and the low-bit formats (fmt 200-203). Only Qwen3.8's sidecars use those, and Qwen3.8's own CUDA tier streams FP8 experts only: a build with Vulkan too hands the sidecars' experts to the Vulkan tier on the same card ([qwen38.md](qwen38.md#routed-experts-as-int4-g64)).
- **GLM-5.3 Flash, DeepSeek V4.1 and MiMo** (`COLI_CUDA=1`, no Vulkan tier): the CUDA device opens for the routed experts, as their checkpoint or a GGUF holds them. Their dense matrices go to the same card where they fit (below, `cu_dense.h`): GLM-5.3's resident ones, a DeepSeek V4.1 GGUF's, and MiMo's in f32, int8 or a GGUF's blocks (its bf16 and FP8 stay on the CPU). A build with Vulkan too runs these engines on Vulkan unless `COLI_VULKAN=0`, since Vulkan's dense chain runs a layer's work on the card in one frame. `tests/cuda_tier_engines.sh` runs each of these engines and Kimi K3 on its tiny fixture against its CPU run on a CUDA card (the tokens, DeepSeek V4.1's oracle, Kimi K3's logits).
- **Kimi K3** (`COLI_CUDA=1`, `K3_CUDA=1`, no Vulkan tier): the routed experts, a checkpoint's MXFP4 or a GGUF's blocks, resident in the tier from the history, with SiTU on the card. The per-call path that uploaded every routed MXFP4 expert at every decode step stays off once the tier is on; the experts it has not taken run on the CPU. Its dense matrices go to the card where they fit (below), which UD-IQ1_S's 41.6 GiB does not on a 24 GB card. unsloth's UD-IQ1_S (594 GB) on an RTX PRO 4000 (24 GB) with 62 GB of RAM, a 181-token prompt and 128 tokens: 0.6 tok/s of prompt and 0.26 of decode, against 0.4-0.5 and 0.18-0.22 on the CPU and 1.0 and 0.28 on Vulkan, whose chain also puts 45 of the 93 layers on the card; most experts come from the disk, which sets the pace.
- **DeepSeek V4** (`make deepseek-v4 CUDA=1`, `COLI_CUDA=1` with no Vulkan tier) uses it for a GGUF's experts, and puts the GGUF's dense matrices on the same card (below).

### The dense matrices on the card, a matrix at a time (`cu_dense.h`)

The tier leaves an engine's dense part on the CPU, which reads all of it every token. `cu_dense.h` puts it on the tier's card: each matrix goes up the first time it is used, in the backend's formats (f32, int8 rows, int4 in groups, a GGUF's blocks, rows the CPU holds interleaved by 16 back to the file's), its handle in the engine's matrix struct, and runs there from then on (the GEMV for a decode step, the tensor cores' tiles for a prompt chunk). DeepSeek V4 and V4.1 (a GGUF), Kimi K3, GLM-5.3 and MiMo take it; `COLI_CUDA_DENSE=0` keeps the dense part on the CPU.

- **The budget:** the whole dense part where it fits in half of the card's free memory less `COLI_VK_TIER_RESERVE_GB`, else none, and the tier leaves it that room. Kimi K3 UD-IQ1_S with its first 16.7 of 41.6 GiB on the card (`COLI_CUDA_DENSE_GB`) left the tier 865 experts instead of 3463; the card's share of the routed experts fell from 36% to 16.5%, more of them came from the disk, and a request had not finished in the time it took with the dense part on the CPU.
- **The rules** are the Vulkan per-matrix path's: the engine thread, outside OpenMP regions, and only matrices the engine keeps. DeepSeek V4 and V4.1 find a GGUF view by its address (`cud_entry`): V4 in its resident layers, V4.1 for `wo_a`'s groups, a view made for each call; a routed expert's slot, which holds one expert and then another, never takes this path.
- `GQ_EXACT=1` runs a GGUF's blocks in the float form on the card too (`backend_cuda.cu`), as the CPU does.

### DeepSeek V4 with a GGUF on CUDA

The FP4 checkpoint's own CUDA path mirrors native FP4 experts and FP8 and bf16 matrices only, so under it a GGUF's experts and its dense part stayed on the CPU. With `COLI_CUDA=1` the tier above takes the experts, and the dense matrices the engine keeps resident (attention, the compressors and the indexer, the shared expert, the head) go to the same card as the file's blocks: `cuda_bd.cuh`'s GEMV for a decode step, the tensor cores' tiles for a prompt chunk's rows. The engine reads its layers at start, so that the tier's budget leaves them their room, and each matrix goes up the first time it is used. `COLI_CUDA_DENSE=0` keeps them on the CPU.

DeepSeek V4 Flash UD-IQ1_S (82.5 GB) on an RTX PRO 4000 (24 GB, PCIe 4.0 x8) with 62 GB of RAM and an i5-13500, a 120-word prompt (163 tokens) and 128 tokens, a warm-up and four runs, each server on a copy of one expert history:

| | prompt | decode |
|---|---|---|
| CUDA, the experts on the CPU (the tier off) | 7.8-8.4 tok/s | 2.67-2.68 tok/s |
| CUDA, the tier, the dense part on the CPU (`COLI_CUDA_DENSE=0`) | 14.0-14.1 tok/s | 4.11-4.19 tok/s |
| CUDA, the tier and the dense part on the card | 17.4-18.8 tok/s | 7.23-7.70 tok/s |
| CUDA, and the batched attention on the card (the default now) | 23.2-24.4 tok/s | 7.79-8.14 tok/s |
| Vulkan (the tier and the dense chain) | 20-22 tok/s | 6.6-6.9 tok/s |

- The CPU read the dense part (791 matrices, 5.98 GiB) every token: half of a decode step went to it.
- On the card it leaves the tier 2333 experts instead of 3280, and the host pool 614 (4.25 GiB) instead of 7709: the pool takes what RAM has free less the room of the engine's expert cache, and the CPU computes the other experts from that cache (78.5% of the routed experts on the card, 95.6% before).
- A pool that takes the cache's room instead (the cache down to its least, 22 slots a layer: 6215 experts in the pool, 509 disk reads against 4612) measured the same decode (7.60-7.69) and a slower prompt (15.1-15.4), so the cache keeps its room.
- The batched attention (`COLI_CUDA_ATTN_BATCH`): the attention core, the indexer, the KV ring and the mHC mixes on the card. It used to find its device through a layer's FP8 `wq_a` mirror, which a GGUF never has, and stayed on the CPU (about 2.6 s of an 8.3 s prompt); a layer without that mirror takes the device the engine opened, and with the dense part on the card it is on unless `COLI_CUDA_ATTN_BATCH=0`. Two pairs of servers alternated: prompt 23.2-24.4 tok/s against 18.2-18.9 without it, decode 7.79-8.14 against 7.56-7.70.
- In the float forms (`GQ_EXACT=1 COLI_CUDA_BD_FLOAT=1 COLI_CUDA_BD_TILES=0`, and `COLI_VK_TIER_WARM=0`: the warm start places experts from many threads, so its pool can leave out a different few each run) and `COLI_CUDA_ATTN_BATCH=0`, the 40-token greedy text is the CPU's. The batched attention sums in another order than the CPU's, and its text parts from the CPU's within those 40 tokens; take `COLI_CUDA_ATTN_BATCH=0` for a comparison with the CPU.

The same models on the CPU, Vulkan and CUDA, one machine and one build: [gpu-parity.md](gpu-parity.md).

## OLMoE on the card (`olmoe_cuda_chain.h`)

OLMoE-1B-7B is small enough to hold whole on a card (Q4_K_M: 3.9 GiB of experts and 0.6 GiB of
the rest), so a CUDA build (`make olmoe CUDA=1`) with `COLI_CUDA=1` (`coli serve --gpu`) puts
every layer, every routed expert and lm_head on the device and runs a step of any rows as one
frame: the norms, q/k/v (one matrix where their forms agree), OLMoE's q and k norms over the
whole projection, RoPE, the new K/V rows into a float16 mirror, the attention, o_proj, the router,
its softmax and top-k, the experts and the residual adds, then the final norm and lm_head. The
dense matrices stay in their GGUF blocks (or f32), the experts in their blocks or the merged
checkpoint's int8 rows. It needs the whole model, a prompt chunk's scratch and the KV mirror in
the card's free memory, and declines otherwise (the CPU runs it), as with `KV_SLOTS` > 1.

While it runs, the device's KV cache is the one that counts: the host's catches up only when the
CPU needs it (a step the chain declines reads the missing rows down), and a device lost mid-step
rebuilds them on the CPU from the token ids the chain recorded, the CPU running from then on
(`COLI_CUDA_CHAIN_FAULT=n` fakes the loss at the n-th frame).

Measured on an RTX PRO 4000 (PCIe gen 4 x8), OLMoE-1B-7B-0125-Instruct Q4_K_M, `coli serve`,
tok/s:

| | prompt 1K / 3K tokens | decode after 1K |
|---|---|---|
| the CPU (before: no CUDA path) | 101 / 79 | 32.5 |
| this chain | 9798-10107 / 8584-8599 | 316.5-316.6 |
| our Vulkan chain, the same build | 4634 / 5140 | 275.0 |
| the reference GGUF engine, CUDA | 10690 / 8403 | 325.9 |

Where the time went, and what changed it (decode, a token): a RoPE reading its cos/sin table from
host memory over the bus took 22.5 us a call (0.7 ms of 3.9), now copied to the device a frame;
bringing every step's K/V rows down into the host's cache cost a 1K-token prompt more than its
kernels, now they stay on the device; q, k and v in one matrix (and the q and k norms and RoPE
in one launch each) took the decode from 293.6 to 315.0. `COLI_CUDA_CHAIN_QKV=0` keeps the three
matrices apart.

`tests/cuda_olmoe_compare.py` runs the tiny fixture's oracle on the CPU and on the device
(`KV16=1`, `COLI_CUDA_CHAIN_ROWS=3`, `COLI_CUDA_CHAIN_QKV=0`, `COLI_CUDA_CHAIN_FAULT=n`): the
reference's tokens and the last logits within 1e-5 of the CPU's.

## The decision engines

Laya (`laya.c`) and GLiNER2.5-Decide (`gliner_decide.c`) answer a decision with one forward
over the request's rows and generate no text ([systemone.md](systemone.md#decision-engines)).
In a CUDA build (`make laya gliner_decide CUDA=1`) with `COLI_CUDA=1` (`coli serve --gpu`),
`decide_cuda.h` opens the device once the weights are loaded, uploads every matrix of the
encoder and of Laya's head as f32 (the checkpoint's values), and runs each forward as frames
of the CUDA chain, one submission a forward: the embedding rows go up, the activations stay on
the device, and only the rows the scorer reads come back (Laya's markers and each sequence's
first row, GLiNER2.5-Decide's `[L]` rows). The scorer, the act head and the classifier stay on
the CPU. It is `decide_vk.h`'s forward, op for op, on the chain's encoder ops (`cc_enc_*` in
`cuda_chain.h`, the arithmetic of `chain_enc.comp`): the LayerNorm with the residual add, a
matrix's bias and its GELU or ReLU, ModernBERT's GeGLU and RoPE, a type row per row, the
bidirectional attention over each row's range of rows with DeBERTa's two relative terms, and
those terms' products. The LayerNorm sums in double as the engines do, and the GELU is
`erff`'s.

The device is `COLI_GPUS`' first (device 0 without it); `COLI_CUDA_CHAIN=0` keeps the
forward on the CPU, and with `COLI_VULKAN=1` as well the Vulkan device takes it. A device lost
mid-forward (`COLI_CUDA_CHAIN_FAULT=n` fakes it at the n-th frame) leaves that forward and
every later one to the CPU, which recomputes it from the ids. Each request ends with
`[CUDA] <engine> chain: F forwards on the device ..., N on the CPU`.

Measured on an RTX PRO 4000 (the CPU an i5-13500, 6 performance-core threads), the release
checkpoints, requests of 12 questions over a 1,700-character email:

| | CPU, a request | CUDA, a request | the decisions | probabilities |
|---|---|---|---|---|
| Laya (ModernBERT-large, 28 layers; 6,144 rows a request) | 11.4-13.0 s | 0.69 s | the CPU's | within 3.3e-6 |
| GLiNER2.5-Decide (DeBERTa-v3-large, 24 layers; one sequence of about 1,100 rows) | 2.0-2.6 s | 0.16-0.21 s | the CPU's | within 5.4e-7 |

`tests/test_cuda_chain.cu` checks every encoder op against a CPU reference;
`tests/cuda_decide_compare.py laya|gliner` runs each tiny fixture's cases on the CPU and on
the device and holds the answers to the CPU's (the decisions, the refusals, every
probability within 1e-5), with `COLI_CUDA_CHAIN_FAULT=n` a loss in the middle of the run.

## Kernels that start while the one before ends (`COLI_CUDA_PDL`)

A decode step runs its kernels one after another on one stream: about 800 a token on
Qwen3.6-35B, 300 on OLMoE. With a plain launch the device sits idle between two of them while
the next grid is set up and the last blocks of the one before drain. On sm_90 and later the
chain's kernels start through programmatic dependent launch (`coli_launch`,
`backend_gpu_compat.h`): the next kernel is set up while the one before ends, and its first
statement waits on the device (`griddepcontrol.wait`) until the grids before it have completed
and their writes are visible. Each kernel reads and writes what it did after a plain launch, so
every result keeps its bits.

It covers the CUDA chain's ops, the GGUF block GEMVs, and the backend kernels the chain reaches:
the dense GEMVs and GEMMs of every format, the routed experts' kernels, their sort and their
sum. HIP, cards before sm_90, a build whose code for the card was compiled below sm_90, and the
legacy default stream launch plainly. `COLI_CUDA_PDL=0` does so everywhere.

Measured on an RTX PRO 4000 (sm_120), alternating server starts, `dev` against this, tok/s:

| | decode after 1K | decode after a long prompt | prompt |
|---|---|---|---|
| Qwen3.6-35B UD-Q4_K_M, every expert on the card | 101.1-102.3 -> 103.8-104.1 | 18K: 90.1-90.9 -> 92.4-92.7 | the same |
| Qwen3.6-35B, colibri's int8 trunk and int4 experts | 105.0-105.5 -> 109.8-110.0 | 18K: 93.8-94.0 -> 97.2-97.4 | the same |
| OLMoE-1B-7B Q4_K_M | 312.5-313.4 -> 322.4-322.8 | 3.8K: 239.8-240.6 -> 246.4-247.0 | the same |
| Qwen3.8-Flash-Next IQ2_XS with the host pool | 40.8-41.1 -> 40.8-42.2 | 4.8K: 29.3-29.4 -> 28.7-30.0 | the same |

The prompt column compares `COLI_CUDA_PDL=0` and `1` on one build, three alternated starts
each (OLMoE: prompts of 1K and 3.4K tokens; Qwen3.6: 1K and 4.4K). Letting each kernel release
the next one at its start (`griddepcontrol.launch_dependents`) measured slower: Qwen3.6
103.3 / 102.7 tok/s after 1K against 104.8 / 103.9 with the wait alone.

## Notes and limitations

- Text-mode timing reports prefill separately from decode.
- MTP speculation defaults off on CUDA (cold draft routes increase expert
  traffic); explicit `DRAFT=n` overrides. Since #294, `SPEC_PIN=1` keeps
  draft/verify kernels consistent when speculation is on.
- Devices use independent contexts; a single expert is not sharded. Kernels are
  correctness-first custom kernels.
- Profile quality matters more than raw VRAM capacity: the same 150 GB tier
  measured 0.94–1.64 tok/s hot-first vs 0.29 tok/s filled without routing heat.
- The GPU tier earns its VRAM only when the CPU is the weak link — a tuned
  AVX-512 CPU can match a 5090 on expert matmul
  ([#101](https://github.com/JustVugg/colibri/issues/101)).

## Reproducible backend A/B without the full checkpoint

```bash
cd c
python tools/make_glm_bench_model.py --output /nvme/colibri-bench-medium --device cuda
python tools/benchmark_cuda_fixture.py --model /nvme/colibri-bench-medium --gpu 0
```

The 313M-parameter fixture has random weights and is not a language model. It
preserves the real MLA/MoE/streaming shapes to compare CPU streaming, dense-only
CUDA, CPU hot-store, and CUDA hot-expert execution with identical replay tokens.
