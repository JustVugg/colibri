# Track A — colibri's Vulkan DeepSeek V4.1 prefill vs DeepSeek's DeepInfer

Scope: (1) prefill VRAM / memory management, (2) operation fusion + kernel/command batching.
Read-only on colibri; no benchmarks run. All line numbers are from the two checkouts on this box.

Sources
- DeepInfer: `deepseek-ai/DeepInfer`, commit `a679bb86ce36e883a98a97733492b686e5b9a08e` (2026-11-03, "docs: update
  V4.1 Flash Q&A to v0.2"), cloned to `/root/Refs/DeepInfer`.
- colibri: `/root/Projects/colibri` (unmodified).

**How to read this document.** Every claim carries `file:line`. Section 4 separates what I verified in
source from what is inference. Numbers marked *(derived)* are arithmetic on cited code, not measurements.

---

## 0. The one framing fact that decides everything

DeepInfer is a **disaggregated multi-GPU server for SM100-class parts** (README: prefill benchmark on 4
SM100 GPUs; `doc/inference_deployment_qa.md:143` describes prefill engines as 8-GPU nodes). Its whole model
of memory is "everything resident, EP-sharded, KV fills the rest of the HBM envelope". **It has no
weight-streaming, no expert paging, no host pool, and no device-memory split** — the only offload in the
open-source tree is the Engram n-gram table (host DRAM + RDMA, `deepinfer/model_executor/parallel_utils/userbuffer.py:147`,
`doc/inference_deployment_qa.md:85`).

colibri's prefill runs on **one 16 GiB RX 6900 XT (RADV)** with 7.62 GiB of trunk resident, a 5 GiB mapped
window, a 3.31 GiB expert tier and expert reads coming off disk at ~2.6 GB/s (`docs/deepseek-v41.md:529-540`).

So the useful comparison is **not** "copy their memory policy" — their policy is "the model fits". It is:

1. their *disciplines* that survive when memory is tight: one plan-time budget, zero runtime allocation,
   zero mid-forward syncs, quantized KV, bounded-replay truncation, block-compacted index scratch;
2. their *kernel composition*: one kernel per pipeline stage, not per op.

Two of those disciplines are directly transplantable, and one of them (bounded replay) colibri already has and
has implemented **more exactly** than DeepSeek ships (bounded-replay row in §1, and §4).

---

## 1. Side-by-side: prefill memory management

| | colibri (Vulkan chain, `deepseek_v41_chain.h`) | DeepInfer (V4.1) |
|---|---|---|
| What is resident on the device | trunk 40/40 layers, 7.62 GiB (`docs/deepseek-v41.md:505-540`); expert tier 3.31 GiB ≈ 84 experts; window rings + compressed KV mirror + index keys; per-chunk activation scratch | entire model, EP-sharded: each rank holds `n_routed_experts / ep_size` expert rows (`v4_modules/moe.py:1-16`, `:94-107`); engram tables **not** resident on GPU (host RDMA, see below) |
| Expert fetch during prefill | host `moe_run_at` from disk/RAM cache, overlapped with device frame A2 (`c/deepseek_v41_chain.h:1229-1234`); tier streams cold experts per chunk (`docs/vulkan.md:2927-2960`) | none — experts are on-device; tokens move instead (DeepEP all-to-all, `v4_modules/mlp.py:461-511`) |
| Chunk / batch sizing | `v41c_chunk_rows` (`:752-768`): counting pass over `v41c_scratch`, then `vkc_chunk_rows` = min(cap 8192, **half** of (budget−use), RAM-capped) (`c/vk_chain.c:3367-3400`); then a **quadratic-scratch proxy cap `rows × ctx ≤ 32M` floats** (`:991-992`); then the KV-split clamp (`:993-996`) | `max_num_batched_tokens`, default `max(max_model_len, 4096, …)` (`deepinfer/config.py:219-228`); open-source scheduler chunks on that + `prefill_max_sum_seqlen_k` + CP alignment (`csrc/engine/scheduler.hpp:462-505`); **production** solves the chunk from a *time* budget: `cost(q,c)` quadratic, cap 32,768 tokens, min-reduced across DP (`doc/inference_deployment_qa.md:143`) |
| Prefill scratch budgeted how | per-row counting pass, then half of free device memory; ring grows to `W+rows` and is re-mirrored on growth (`:770-782`) | **profile once at startup**: dummy worst-case prompt batch + a *real* temp KV cache, `peak_torch + non_torch` measured, then `total*utilization − peak − state` → `num_gpu_blocks` (`deepinfer/worker/base.py:209-268`, `worker/model_runner.py:405-514`); with **forced MoE imbalance 1.8×** during profiling (`model_runner.py:511-514`) so the KV allocation survives the worst routing |
| KV on device | fp32: a window ring of `W+chunk` rows × 512 floats per layer (`:510`, `:775`) and, for the **4 kv-source layers only** (`kv_source_layer_ids [2,8,14,20]`, `ckpt config.json`; `sli[]` at `:647`), compressed rows `cap*hd*4` (`:739`) + index keys `cap*128*4` (`:742`); `v41c_res` counts 4 B/element (`:123-126`) | quantized: main KV **528 B/token** MXFP8 (`V41_FORMAT`, `tilelang_ops/cache_common.py:119-127`; `doc/inference_deployment_qa.md:223`) or **288 B/token** FP4 (`V41_FP4_FORMAT`, `:129-137`); index KV 132 B (MXFP8) / 68 B (FP4) (`_get_qk_topk_bytes`, `worker/cache_engine.py:73-84`); paged uint8 blocks, token-interleaved |
| KV structure | per-layer window ring + per-owner compressed rows + index keys; a windowed **split** (`vk_kvsplit.h`) when the full mirror would grow (`v41c_plan`, `:700-730`) — and a split costs 77 s → 170 s on a 7579-token prompt (`docs/vulkan.md:3168-3198`) | SWA rings live in a **state cache** with one fixed-size `StateBlock` per sequence, `[stored SWA \| dropped SWA \| compression state]` (`worker/cache_engine.py:206-310`, `doc/inference_deployment_qa.md:217-223`); sized `*2` for store+copy buffer (`worker/base.py:239-246`) |
| Prefix/context ceiling | `max_positions` from the checkpoint (1,048,576 on this checkpoint), capped by `CTX` (`c/deepseek_v41.c:227-244`); the device mirror grows with the conversation and splits when it must | the ceiling is whatever `num_gpu_blocks` the startup profile allowed; past it, block KV is offloaded to 3FS (72 h TTL, `doc/inference_deployment_qa.md:167-168, 191`) and swapped in per-layer via GDR — never resident-limit-driven |
| Bounded replay of decoder windows | **exact tail**: `19×127+1 = 2414` rows for V4.1 (`docs/deepseek-v41.md:379-396`), bit-identical to running every row; `V41_PREFILL_TAIL` can set DeepSeek's 140 for speed at a KL cost | YOCO prefill truncation: `DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE=compact`, `KEEP_TOKENS=140` (window 128 + backtracking), which **physically compacts the token dimension** at the boundary layer (`csrc/engine/input_metadata/prepare_tensor_input_v4.cpp:129-230`, `v4_modules/yoco.py:12-20`), with the guard `keep_tokens ≥ window + max_backtracking` (`:350-356`) |
| Activation width | fp32 everywhere (`v41c_res`, `:123-126`) | bf16 acts + FP8 activations with per-32 UE8M0 scales, written straight into the dispatch buffer (`v4_modules/mlp.py:274-301`, `linear.py` fp8 cast params) |

### The one constraint that matters most

colibri prints the row cost it budgets: **1293 KiB a row** on the 16 GiB box
(`dsv41-serve.log`: `[VK] deepseek_v41 chain: prompt chunks of up to 1280 rows (1293 KiB a row; 3.72 GiB free
on the device or in RAM, half of it at most …)`), and the chunk is `min(8192, half_free / 1293 KiB)` =
1280-1536 rows. That is the whole story **up to ~26k context**. Above it, a second, hard-coded cap takes over:

```
int CH = v41c_chunk_rows(ch, m, E), rows;
{ int64_t lim = ((int64_t)32 << 20) / (E > 1 ? E : 1); if (lim < CH) CH = lim < 1 ? 1 : (int)lim; }
```
`c/deepseek_v41_chain.h:991-992`, `E` = the forward's end position.

`lim` is `32M / ctx` **rows** — a proxy for "rows × context ≤ 32M float-elements", which is exactly the shape
of the indexer's two buffers: `isc` (scores) and `mask` (candidate mask), each reserved at `rows × wcap`
4-byte elements with `wcap = ctx` (`:794`, `:123-126`, `:783-800`). The measured 1293 KiB was taken at a first forward (small `E`), so treat it
as the row's *fixed* cost and add `8·ctx` bytes for those two buffers *(derived, half of a 3.72 GiB free
budget):*

| context | cap `32M/ctx` | row bytes | free-budget rows | chunk used | quadratic share |
|---|---|---|---|---|---|
| 4k | 8192 | ~1.36 MB | ~1470 | **1470** | 2% |
| 16k | 2048 | ~1.45 MB | ~1375 | **1375** | 9% |
| 32k | 1024 | ~1.59 MB | ~1255 | **1024** | 16% |
| 64k | 512 | ~1.85 MB | ~1080 | **512** | 28% |
| 128k | 256 | ~2.37 MB | ~840 | **256** | 44% |

So from ~26k context on, **colibri's prefill chunk shrinks as 1/ctx** and every expert pass serves fewer rows
— the opposite of the "big chunks" win it measured (`docs/vulkan.md:2843-2860`, 512 rows → 8192 rows):
**1470 → 256 rows, 5.7× worse, at 128k**. Below ~26k nothing is wrong: the free budget binds and the chunk
sits at its steady state.

DeepInfer hits the same wall (its indexer score scratch is documented as a 32M-float cap too,
`docs/vulkan.md:2903-2904`) but **decouples it from the step size**: `candidate_block_max(logits, valid_lens,
block_size)` takes `[num_rows, max_model_len]` and an explicit `valid_lens[num_rows]`
(`tilelang_ops/candidate_blocks.py:11-49`), i.e. it is written to be called on a **tile** of rows, stores
**block maxima** (`[num_rows, max_model_len/block_size]` bf16, 1024-byte row alignment) rather than a
per-column int32 mask — 16-32× smaller per row — and allocates it per call. colibri instead reserves
`isc`/`mask` for the whole chunk (`:794`) and shrinks the chunk to fit them.

---

## 2. Side-by-side: fusion and launch batching

| | colibri | DeepInfer |
|---|---|---|
| Submission model | one command buffer per phase; frame ring of 16 (`c/vk_chain.c:55`, `vkc_begin` `:820-853`, `vkc_submit` `:854-892`); a global memory barrier at the head of every frame (`barrier_now` `:806-813`); ops recorded via `record()` (`:969`) | plain eager PyTorch: one launch per op, all async on one stream; no batching layer, no graphs for prefill |
| Per layer, prefill | **2 frames**: A1 (attention+mix+prologue) submitted (`deepseek_v41_chain.h:1209`), then a **full-device wait** `vkc_finish()` (`:1213`) because the host needs `nrm` for the router; A2 (shared expert) submitted and *not* waited (`:1229`); host `moe_run_at` overlaps A2 (`:1231-1234`) | one unsynchronised launch sequence; **CUDA graphs decode-only** (`run_cuda_graph = not is_prompt and …`, `worker/model_runner.py:182`; graphs disabled entirely for the PROMPT engine, `:73`) |
| Mid-forward syncs | **1 full-device drain per layer** (`vkc_finish` waits `VKC_FRAMES` fences, `c/vk_chain.c:802-807`) → 40/chunk | **enforced zero**: `execute_main_model`/`execute_sample` run under `forbid_cuda_sync()` which *raises* on any torch-visible CPU↔GPU sync (`worker/model_runner.py:311-330`, `deepinfer/utils.py:119-132`), covered by `tests/test_cuda_sync_guards.py` |
| Dispatches per attention stage | ~17-25: `v41c_pre` 4 (`:822-829`), q_a matmul, q_a norm, q_b matmul, wkv matmul, kv norm, rope ×2, ring `vkc_copy_regions`, compressor (`comp_wkv`+`comp_wgate`+`vkc_dsv4_compress`+norm), index (`idx_wk`+norm+rope ×2), indexer (`idx_wq_b`+rope+`idx_wproj`+`vkc_dsv4_score`+`vkc_dsv4_cand`+`vkc_dsv4_topk`), `vkc_dsv4_attn`, inverse rope, `vkc_mla_hgemv` wo_a, wo_b matmul, `v41c_post` (`:855-950`) | ~1 per stage: `mhc_pre_big_fuse` **one kernel** for norm+fn GEMM+mixes+Sinkhorn+collapse+FP8 cast (`v4_modules/hc.py:317-338`, `mhc_modeling.py:1-10`); KV row transform **one kernel**: `fused_norm_rope_hadamard_cast_store_kvcache` = RMSNorm+RoPE+Hadamard+FP8/FP4 cast+paged store (`tilelang_ops/fused_transform.py:189-235`, called at `v4_modules/attn.py:433-451`, `:946-960`); q-norm+rope fed *into* the attention kernel and inverse-rope+wv-einsum fused into its epilogue (`v4_modules/attn.py:87-93, 159-172, 191-200`) |
| MoE | router GEMV on host; whole chunk to the tier (`c/deepseek_v41.c:3616-3631`) or 32-row blocks on the CPU path (`:3633`); per-expert GEMMs; per-draw contributions kept apart and summed in rank order (`docs/deepseek-v41.md:345-364`) | gate → DeepEP dispatch → grouped FP8/FP4 GEMM with **SwiGLU + per-token cast fused between the two GEMMs** (`v4_modules/mlp.py:406-460`, `:109-140`) → combine; dispatch/combine on a comm stream with `async_with_compute_stream=True` (`:469-511`); shared expert runs **under an SM cap concurrently with the combine** (`:596-610`); optional `mega_moe` fuses dispatch+gate_up+activation+down+shared+combine into one call (`:528-560`) |
| CPU/launch-time hiding | none — the host thread must finish the MoE before the next device frame can start | zero-overhead scheduling: the previous step's token handling overlaps the next step's scheduling (`DEEPINFER_ZERO_OVERHEAD_SCHEDULING`, `engine/llm_engine.py:826-980`, `csrc/engine/zos.cpp`) |
| Per-row variance in a kernel | `vkc_dsv4_score` loops **one dispatch per row** when the chunk's rows score different key owners (`deepseek_v41_chain.h:906-916`) | per-row `valid_lens` inside one kernel (`candidate_blocks.py:11-49`) |

---

## 3. Ranked recommendations

Benefit is expressed in colibri's own units (VRAM freed, submits/drains removed, expert passes per prompt).
Effort S ≤ a day, M ≈ a few days, L ≈ a week+.

### R1 — Decouple the prefill chunk from the indexer's quadratic scratch (tile the scores, block the mask)
**DeepInfer**: `candidate_block_max` is written to run on a row *tile* with per-row `valid_lens`, stores
block maxima instead of a per-column mask, in bf16 with 1024-byte row alignment
(`tilelang_ops/candidate_blocks.py:11-49`); the indexer score scratch is a capped, reusable tile while the
MoE/attention step size is set by `max_num_batched_tokens` (32,768 tokens in production,
`doc/inference_deployment_qa.md:143`).
**colibri analogue**: the cap `lim = 32 MiB / E` (`c/deepseek_v41_chain.h:991-992`) exists only because
`isc` and `mask` are reserved at `rows × wcap` for the whole chunk (`:794`), with `wcap = ctx`
(`:783-800`). `vkc_dsv4_score` already takes a row count (`:915`), so it can be called per tile.
**Expected benefit** *(derived, from the table in §1)*: the chunk returns to its low-context steady state at
every context — **256 → ~1470 rows at 128k (5.7×)**, **512 → ~1470 at 64k (2.9×)**, 1024 → ~1470 at 32k
(1.4×), nothing below ~26k. Expert-union passes per prompt fall by the same factor, which is the lever
colibri already measured as its 512→8192 "big chunks" win (`docs/vulkan.md:2843-2860`). VRAM: the same
buffers, re-tiled — no new memory, and it *frees* the chunk-sized `isc` reservation (`:794`).
A second, independent 2× sits behind the same code: the row's other 1.3 MB is fp32 activation scratch, and
DeepInfer runs those buffers in bf16/FP8 (`v4_modules/mlp.py:274-301`). That halves `row_bytes` — so ~2900
rows everywhere — but it cannot be bit-exact.
**Interaction with VRAM**: a bigger chunk grows every layer's window ring to `W+chunk` rows
(`v41c_grow_win`, `:770-782`): 40 × 8320 × 512 × 4 B ≈ **681 MB at 8192 rows** against 115 MB today, i.e.
+566 MB that comes straight out of the expert tier. Pair R1 with R2 (fp8 rings: the same rows for 175 MB) or
land R1 with the ring rows left at the chunk size the budget can afford.
**Correctness risk**: none for the mask change *if* the mask stays derived from the same whole-block
selection (`vkc_dsv4_cand`: "the best blocks kept whole", `docs/vulkan.md:2831`): tiling and block-indexing
change no row's arithmetic. bf16 scores would change bits — keep f32 unless an explicit approximate mode is
added.
**Effort**: M (score kernel gains a row tile + a block-index mask lookup; the mask→host copy at `:1220-1226`
becomes a block table).

### R2 — Store the device KV mirror in the checkpoint's quantized KV format instead of fp32
**DeepInfer**: main KV 528 B/token MXFP8 or 288 B/token FP4, index KV 132/68 B/token, per-block UE8M0 scales,
token-interleaved, paged uint8 (`tilelang_ops/cache_common.py:119-155`, `worker/cache_engine.py:46-84`,
`doc/inference_deployment_qa.md:223`); the cast is fused into the kernel that *writes* the row
(`fused_transform.py:189-235`).
**colibri analogue**: `ch->win[i]` (`:510`, `:775`), `ch->ckv[i]` (`:739`) and `ch->ikey[i]` (`:742`) are all
`floats × sizeof(float)` — **2048 B/token** for the ratio-1 layer 20's compressed row (`head_dim = 512` in the
checkpoint's `config.json`), 1024 B/token for layers 2/8/14 at ratio 2, 512/256 B/token of index keys, and
2048 B/token per layer of window ring.
**Expected benefit** *(derived; 4 owning layers, `W = 128`, `sliding_window: 128`)*: the mirrors total
`6400 × ctx` bytes today, `1660 × ctx` at FP8 and `900 × ctx` at FP4 (528 B / 288 B per token):

| context | fp32 mirrors today | fp8 | freed | ≈ resident experts at 21.1 MiB |
|---|---|---|---|---|
| 16k | 105 MB | 27 MB | **78 MB** | +3 |
| 64k | 419 MB | 108 MB | **311 MB** | +14 |
| 128k | 838 MB | 216 MB | **622 MB** | +29 |
| 1M (the checkpoint's `max_position_embeddings`) | 6.4 GB | 1.66 GB | **4.7 GB** | +220 |

At 16k this is small; from 128k up it decides whether the mirror fits the 5 GiB window at all, i.e. whether
`vkc_kv_plan_need` splits (`:700-730`) and the prompt pays 2.2× (`docs/vulkan.md:3168-3198`: 77 s → 170 s on a
7579-token prompt) — and it is what pays for R1's ring growth (see R1).
**Correctness risk**: **high — breaks bit-exactness vs the CPU oracle** (the CPU keeps fp32 KV). It needs the
same kind of explicit opt-in as `V41_PREFILL_TAIL=140` (`docs/deepseek-v41.md:419-424`): a documented
approximation with its KL measured, not a default change.
**Effort**: L (three new store/load paths + the attention and compressor shaders reading a quantized row).

### R3 — One kernel for the KV row transform: norm → rope → ring store → compress
**DeepInfer**: `fused_norm_rope_hadamard_cast_store_kvcache` does RMSNorm, RoPE, Hadamard, quantize and the
paged store of one row batch in a single kernel, with `KVCacheOutArgs(slot_mapping)` as the sink
(`tilelang_ops/fused_transform.py:189-235`; used on the SWA path at `v4_modules/attn.py:433-451`). The q-side
equivalent is eaten by the attention kernel itself (`attn.py:87-93`, `:159-172`).
**colibri analogue**: `vkc_matmul(wkv)` → `v41c_norm(kv)` → `v41c_rope(kv)` → `vkc_copy_regions(win)` =
4 dispatches, and the compressor path adds `comp_wkv`, `comp_wgate`, `vkc_dsv4_compress`, a second norm and
two more ropes (`c/deepseek_v41_chain.h:860-905`).
**Expected benefit**: 4-9 fewer dispatches per layer × 40 layers = **160-360 fewer dispatches per chunk**,
and 2-4 fewer full passes over `nr × hd` floats (~2.9 MB per pass at 1280 rows × 512 × 4 B; ~470 MB of
round-trip traffic per chunk at 1280 rows). Expect **single-digit milliseconds per chunk** at the current
chunk sizes — it becomes worth more once R1 raises the chunk. Removes no VRAM.
**Correctness risk**: none if the fused shader reproduces fp32 rounding per element (no reassociation, no
different accumulation order). Bit-exactness is achievable; verify per row against the CPU path.
**Effort**: M.

### R4 — Remove the per-layer full-device drain; wait only on the frame the host reads
**DeepInfer**: a forward is **forbidden** to sync — `execute_main_model` under `forbid_cuda_sync()` raises on
any torch-visible sync, with the invariant in the docstring and a test
(`worker/model_runner.py:311-330`, `deepinfer/utils.py:119-132`, `tests/test_cuda_sync_guards.py`). Nothing in
a 40-layer prefill step idles the device for the CPU.
**colibri analogue**: `vkc_finish()` at `:1213` submits and then waits **all 16 frames' fences**
(`c/vk_chain.c:802-807`) — including A2 of the previous layer, which nothing on the host reads — once per
layer, 40 times per chunk. The host MoE for the chunk is the only true dependency.
**Expected benefit**: removes the spurious drains (≤ 40 per chunk); the real cost per drain is the
`vkc_finish` semantics, not the fence — so expect **2-8 ms per chunk** *(derived: 40 × 50-200 µs)*, plus it
stops `chain_spin_us()` from burning a core that the disk reads want (`c/vk_chain.c:766-772`). Order-of-
magnitude smaller than R1/R2 on a disk-bound prefill; listed because it is nearly free.
**Correctness risk**: none (the wait set only shrinks; any buffer read by the host must keep its fence).
**Effort**: S.

### R5 — One dispatch for the indexer score, per-row owner in the kernel
**DeepInfer**: candidate/score kernels take `valid_lens[num_rows]` and process the whole step in one launch
(`tilelang_ops/candidate_blocks.py:11-49`).
**colibri analogue**: `same` is computed over the chunk's rows and, if the rows' key owners differ,
`vkc_dsv4_score` is recorded **once per row** (`c/deepseek_v41_chain.h:906-916`) — up to `nr` dispatches per
index-source layer per chunk.
**Expected benefit**: up to `nr-1` dispatches per index-source layer removed in the mixed-owner case; in the
common prompt case (`same == 1`) zero, and on a verify it caps at `W` (128) rows. Bounded, but the worst case
is large and the fix is local.
**Correctness risk**: none (same rows, same scores).
**Effort**: M (a per-row owner indirection in the shader, or an array-of-buffers binding).

### R6 — Give the prefill path a fixed, plan-time buffer set (no runtime reallocation, no re-mirror)
**DeepInfer**: every prefill-sized buffer is allocated once from `max_num_batched_tokens`
(`model_loader.py:119-120`, `v4_modules/mlp.py:173-175`), the KV is sized once from a worst-case profile
(`worker/base.py:209-268`), and nothing is allocated during a forward.
**colibri analogue**: the window ring is re-allocated and **re-mirrored** whenever `W+rows` exceeds `Wd`
(`v41c_grow_win`, `:770-782`), which drops `win_valid` and forces a re-upload of every layer's ring on the
next forward; `v41c_cache` reallocates compressed-row buffers as the mirror grows (`:731-748`); and
`v41c_attention` does a host `malloc/free` of `nr` regions per layer per chunk (`:865-870`).
**Expected benefit**: one avoided ring re-upload ≈ 40 layers × 1408 rows × 512 × 4 B ≈ **115 MB** *(derived)*,
tens of ms once; the per-layer `malloc` is noise. Structural value is that R1 changes the chunk size, so a
single plan-time sizing removes a class of "the chunk grew and everything re-mirrored" events.
**Correctness risk**: none.
**Effort**: S.

---

## 4. Verified facts vs inference

**Verified in source (DeepInfer)** — everything cited in §1/§2 with these files read directly:
`deepinfer/config.py`, `deepinfer/worker/{base,cache_engine,model_runner}.py`,
`deepinfer/model_executor/models/{decoder_layer.py,v4_modules/{moe,mlp,attn,hc,yoco,engram,engram_parameters}.py}`,
`deepinfer/model_executor/layers/tilelang_ops/{cache_common,fused_transform,candidate_blocks}.py`,
`deepinfer/model_executor/{parallel_utils/userbuffer.py,input_metadata/yoco.py}`,
`csrc/engine/{scheduler.hpp,block_manager.hpp,block_load_plan.cpp,zos.cpp}`,
`csrc/engine/input_metadata/prepare_tensor_input_v4.cpp`, `doc/inference_deployment_qa.md`, `README.md`.
Notably verified: no weight streaming / host offload anywhere in `deepinfer/` except Engram
(`grep offload|swap_space|cpu_offload` returns only pinned-memory sampler/mask allocations); CUDA graph
capture is gated off for prompt/prefill steps; `forbid_cuda_sync` wraps the two launch entry points.

**Verified in source (colibri)** — `c/deepseek_v41_chain.h`, `c/vk_chain.c`, `c/deepseek_v41.c`,
`docs/deepseek-v41.md:345-540`, `docs/vulkan.md:1934-2040, 2817-2960`.

**Inference (not verified)**
- That DeepInfer's 140-row YOCO truncation is *exact* for their topology. The mechanism is verified
  (compaction + `keep_tokens ≥ window + backtracking`); whether 140 rows reproduces every-logit behaviour
  depends on the consumer layers sharing the boundary layer's KV/hidden
  (`v4_modules/yoco.py:12-55`), which colibri's docs claim makes DeepSeek's tail approximate
  (`docs/deepseek-v41.md:419-424`). I did not settle this contradiction.
- All chunk-row and GB numbers in §1/R1/R2/R6: arithmetic on cited reservations (`head_dim = 512`,
  `sliding_window = 128`, key owners `[2,8,14,20]`, `compress_ratios` 2/2/2/1 — all read from
  `/root/Models/DeepSeek-V4.1-Flash-1017/config.json`) plus the measured **1293 KiB/row** colibri prints
  (`dsv41-serve.log`). Not measurements of my own; no GPU work was run.
- The *device-side* dynamic cost is corroborated but not cleanly separated: an untracked probe artifact in the
  tree (`vram-a1-16k-short.csv`, 25 Hz samples of `/sys/class/drm/card2/.../mem_info_vram_used` over one
  16k-context turn) shows device usage moving **12.15 → 14.36 GiB**, i.e. ~2.2 GiB of growth per turn at 16k.
  That number bundles the KV split's uploads, the ring, and the per-chunk scratch; I did not decompose it.
  (That file and `prefill-vram.py` were written into the tree at 19:13 by a process other than this task — 25 Hz
  × 1038 samples ≈ 42 s, i.e. a live `prefill-vram.py --ctx 16384` run. Treat the figure as indicative.)
- The "expert-union passes per prompt ∝ 1/chunk" claim assumes routing saturates well before 8192 rows, which
  colibri's own §"Big prompt chunks" argues and its measured table supports, but which I did not re-derive
  for V4.1-Flash's 384 experts / top-6 (from `config.json`).
- Dispatch counts per stage are counts of *recorded ops* read off the source, not profiler traces.

**Checked and deliberately not recommended**
- *Fold the shared expert into A1* (DeepInfer's `mega_moe`, `mlp.py:528-560`): colibri's separate A2 frame
  exists precisely to overlap the host MoE (`:1229-1234`); folding it would serialize the two. Close it.
- *Pipe two chunks through the device* (DeepInfer-style async scheduling): impossible — prompt rows are
  causally ordered, chunk k+1's attention reads chunk k's KV in A1, so the host MoE of chunk k cannot hide
  behind chunk k+1's attention. Only same-chunk cross-layer pipelining could, and that is blocked by the
  router round trip.
- *Raise `MOE_ROW_CHUNK` (32, `c/deepseek_v41.c:3184`) to cover a whole chunk on the CPU expert path* (to match
  DeepInfer's one grouped GEMM per step, `mlp.py:406-460`): the CPU path keeps a `draws × dim` fp32
  contribution buffer (`c/deepseek_v41.c:3699`), which at 1280 rows × top-6 × 5120 × 4 B would be
  **~150 MiB** *(derived)*, and expert-major accumulation cannot preserve the per-draw rank-ordered sum
  without it (`docs/deepseek-v41.md:358-364`). The tier's whole-step path already gets this win when VK is on.
- *Time-budgeted chunk sizing* (DeepInfer production, `Q&A:143`): colibri's prefill is disk-bound, so the
  cost model is "bytes of the expert union", which is monotone in rows — the memory cap is already the right
  controller. No action.
- *`fadvise(WILLNEED)` / reader-thread prefetch*: already built, measured and removed by colibri
  (`docs/deepseek-v41.md:426-448`).

---

## 5. Bottom line

1. **R1 is the only change here that plausibly moves colibri's prefill by a large factor on long prompts**
   (5.7× at 128k, 2.9× at 64k, 1.4× at 32k, in expert-union passes; nothing below ~26k), it costs no VRAM, and
   it is bit-exact. DeepInfer's indexer is structurally built to be tiled; colibri's is structurally
   whole-chunk — which is why colibri's own chunk silently falls to 256 rows exactly where its prompts get
   long.
2. **R2 is the only VRAM lever that scales with context** (78 MB at 16k, 0.62 GiB at 128k, 4.7 GiB at the
   checkpoint's 1M ceiling — the difference between splitting the KV cache and not), and it is what pays for
   R1's ring growth; but it is the only recommendation that cannot be bit-exact, so it belongs behind an
   explicit approximate mode.
3. **R3-R6 are hygiene**: real but second-order against a 5.9 s disk bill. DeepInfer's fusion and
   zero-sync disciplines are worth adopting only after R1 changes the chunk, which is what makes each
   dispatch and each drain matter more.
