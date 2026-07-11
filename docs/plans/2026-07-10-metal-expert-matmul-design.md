# Metal backend for the MoE expert matmul (Apple Silicon)

Date: 2026-07-10 · Branch: `metal-backend` · Target: Apple M4 Max (128 GB, unified memory)

## Problem

On Apple Silicon, colibrì's decode profile is **matmul-bound, not disk-bound**: a 96 GB
run measured `expert-matmul 123.7 s` vs `expert-disk 74.2 s` per 32 tokens (49% vs 29%).
The engine has a CUDA backend but **no Apple-GPU path**, so the M4 Max's 40-core GPU is idle.
Unified memory means there is no PCIe copy tax — the reason CUDA keeps streaming experts on
the CPU does not apply here. This is a genuine opportunity to accelerate the bottleneck.

## Key empirical findings (microbenchmarks, `scratchpad/mbench.mm`, `mbatch.mm`)

1. **Runtime-compiled Metal works with Command Line Tools only** (no Xcode / no `metal` CLI):
   `newLibraryWithSource:` compiles the shader at load. Backend requires no Xcode.
2. **Winning kernel = "V3": vectorized `float4` loads + threadgroup reduction + simd_sum.**
   Correct vs CPU dequant→f32 reference (normalized err ~2e-6, all formats).
3. **Metal has a hard per-dispatch latency wall (~150 µs from an idle GPU; ~20 µs empty
   command buffer).** Synchronous per-matmul dispatch (what a faithful CUDA mirror does)
   therefore *loses to the CPU* for expert-sized matmuls — a hot expert's 3 sync dispatches
   ≈ 450 µs of pure launch latency vs ~300 µs total on CPU. CUDA gets away with per-call
   dispatch because its launches are ~5–10 µs; Metal's are 20–150 µs.
4. **The win is entirely in batching.** Kept busy, V3 sustains 300+ GB/s (~75% of the
   ~410 GB/s ceiling) and 460–785 GFLOP/s.
5. **Batched full-layer dispatch** (8 experts' gate+up+silu+down in ONE command buffer,
   `mbatch.mm`): 854 µs/layer, 707 GFLOP/s, 177 GB/s, correct. Extrapolates to
   **~64 ms/token** for the expert matmul (75 layers) if experts are resident, vs the
   measured CPU `t_emm` of 3.86 s/token.

## Decision: batched expert dispatch (not a synchronous CUDA mirror)

Accelerate the **routed-expert SwiGLU** by dispatching a whole block of experts as one Metal
command buffer, reading expert weights **zero-copy** from the RAM slabs they already occupy.
Disk streaming (`expert_load`) is unchanged; only the matmul moves to the GPU.

At decode (S=1) all top-k experts multiply the **same** token vector, so a layer's expert
work is a batched GEMV: shared `x`, per-expert weights. For S>1 (prefill/MTP) each expert has
`nr` rows; the kernel loops rows internally (weight reused across rows).

### Kernel (`backend_metal.metal`, compiled at runtime)
- `gemv`: grid over flattened `(expert*O + o)`; threadgroup reduces over K via `float4` loads
  + `simd_sum`; dequant int8/int4/int2 inline (mirror of CUDA `weight_at`); `* scale[e,o]`.
  (1D grid — Metal requires grid-position builtins to share shape; flatten (O,E)→O*E.)
- `silu`: elementwise `silu(gate)*up`.
- Per layer, one command buffer: gate-all → up-all → silu → down-all → (weighted combine).

### Numerics
Dequant → f32 MAC, matching the CPU non-IDOT path (`matmul_q`/`matmul_i4`) and the CUDA
kernel. Validation bar: token-exact vs CPU on the fixture (same standard the CUDA tier meets).

## Expected impact (from the 96 GB profile)

| scenario | est. tok/s |
|---|---|
| current CPU | 0.13 |
| + GPU expert matmul, cold cache | ~0.24 (~1.9×) |
| + warm cache (disk→~0) | ~0.55 (~4×) |
| + attention on GPU (future phase) | higher |

Expert-matmul offload alone is ~2× cold, ~4× warm; it then exposes disk streaming and
attention as the next bottlenecks (future work).

## Integration plan (`c/`)

New files, mirroring the CUDA backend's shape:
- `backend_metal.h` — `extern "C"` API: `coli_metal_init/shutdown`, `coli_metal_moe_layer(...)`
  (batched entry, not per-tensor), `coli_metal_mem/stats`.
- `backend_metal.mm` — Objective-C++: owns `MTLDevice`, queue, pipelines (runtime-compiled),
  persistent zero-copy `MTLBuffer`s wrapping expert slabs and the activation/scratch buffers.
- `backend_metal.metal` — shader source (embedded as a string in the .mm, or a sibling file).

Edits:
- `c/Makefile` — `METAL=1` branch inside the Darwin section: compile `.mm` with
  `clang++ -x objective-c++ -fobjc-arc`, `-framework Metal -framework Foundation`,
  `-DCOLI_METAL`. Default build unchanged; error if `METAL=1` on non-Darwin.
- `c/glm.c` — `#ifdef COLI_METAL`: env init (`COLI_METAL=1`), and in `moe()` replace the
  serial per-expert `matmul_qt` block (glm.c:1230–1247) with a batched Metal dispatch when
  enabled, gathering the block's resident expert slab pointers. CPU path stays the default.

### Zero-copy requirement
Expert slabs must be GPU-addressable. Options: (a) allocate slabs page-aligned and wrap with
`newBufferWithBytesNoCopy:`; (b) allocate slab storage as `MTLBuffer` and point `ESlot.slab`
at its `.contents`. Prefer (b) for cleanliness once proven; (a) is less invasive for a first
cut. Wrap once per slot (stable address); contents change as experts load.

## Validation
- `make metal-test METAL=1`: standalone kernel correctness vs CPU (all formats), like
  `make cuda-test`.
- End-to-end token-exactness: `TF=1` teacher-forcing with `COLI_METAL=1` must match the CPU
  run (argmax-stable; the tiny f32 reassociation deltas do not change tokens).
- A/B tok/s on the real model, warm and cold cache.

## Status
- [x] Approach validated empirically (kernel + batched dispatch + correctness + throughput).
- [x] `backend_metal.{h,mm}` + Makefile `METAL=1` branch (runtime-compiled shader).
- [x] `moe()` batched integration + zero-copy slabs (page-aligned, mutex-guarded registry).
- [x] Kernel-correctness test (`make metal-test`) + token-exact end-to-end (identical greedy output).
- [x] A/B benchmark on the real model (below).

## Measured results (M4 Max, 96 GB budget, warm cache ~55% hit, greedy, 10 tok)

| | CPU | Metal | 
|---|---|---|
| end-to-end | 45.5 s (0.22 tok/s) | 36.4 s (0.27 tok/s), ~1.23x |
| expert-matmul | 11.6 s | 8.5 s, ~1.34x |
| experts on GPU | — | 100% (0 CPU fallback) |

All routed experts (pinned + cached + streamed) run on the GPU. Token-exact vs CPU.

**Key diagnostic:** of the ~8.3 s GPU wall-time, only ~3.1 s is actual GPU kernel
execution — **~62% is idle/scheduling latency** (~13 ms/block over 396 sporadic
submits). The GPU powers down between blocks because attention runs on the CPU
each layer, forcing a sync. Expert-matmul FLOPs are not the bottleneck; submit
latency and (still) disk streaming are.

## Next levers (measured, in impact order)
1. **Offload attention to the GPU** — removes the per-layer CPU sync so the GPU stays
   hot; would reclaim most of the ~5 s latency AND move attention (5-10 s) off the CPU.
   Biggest win; substantial (MLA + RoPE + softmax + DSA indexing on Metal).
2. **Reduce per-block submit overhead** — persistent encoders / fewer command buffers;
   partial help while attention stays on CPU.
3. **Warm cache further** — disk streaming (~17 s) still co-dominates at ~55% hit.

---

# Phase 2: Fused decode attention (in progress)

Goal: run the whole S=1 decode attention for a layer in ONE Metal command buffer
(keeps the GPU hot across the layer → speeds attention AND reclaims the ~5s expert
latency). DSA top-2048 selection stays on CPU (passed in as an index list); prefill
falls back to CPU.

Real GLM-5.2 attention dims (config.json): hidden=6144, H=64, q_lora=2048,
kv_lora=512, qk_nope=192, qk_rope=64, v_head=256 → qk_head=256; kv_b=[28672,512];
o_proj=[6144,16384]; attn_scale=1/16; theta=10000; index_nh=32 hd=128 topk=2048.

Exact math captured from glm.c: projections 1024-1037 (q_a→rmsnorm→q_b→rope; kv_a→
split→latent rmsnorm + krot rope → cache); absorption core 1097-1126 (qabs via
qt_addrow over kv_b nope-rows, T-scoring qabs·Lc + qr·Rc ×1/16, softmax, clat=Σa·Lc,
ctx via qt_matvec_rows over kv_b V-rows); o_proj 1127. Helpers: rmsnorm 587,
rope_interleave 604 (half=32, interleaved-in/split-out), softmax 598.

Kernel plan (all one command buffer, barriers between): q_a matmul → rmsnorm →
q_b matmul → rope → kv_a matmul → (cache write) → [qabs → score → softmax → clat →
ctx] absorption core → o_proj matmul.

## Done
- [x] **Absorption core kernel** validated (nerr ~1e-6, 0.37-0.68 ms/layer).
- [x] rmsnorm + RoPE + copy kernels; projection matmuls reuse mm_gemv.
- [x] `coli_metal_attn_decode`: full S=1 decode attention in ONE command buffer.
      Attention weights uploaded+cached; Lc/Rc page-aligned + registered in kv_alloc.
- [x] Integrated into attention() behind COLI_METAL (S=1 absorb, st0==0, no active DSA
      selection, GLM-5.2 int4 dims; else CPU fallback). DSA index-key stays on CPU.
- [x] **Token-exact vs CPU** (identical greedy output), verified.

## Honest measured result (DRAFT=0, all-S=1 decode, ~60% hit, 8 tok)
The fused attention **triggers correctly** (546 layer-calls = full decode coverage) but is
**submit-latency-bound at short context and yields no speedup**:

| | CPU | Metal |
|---|---|---|
| attention (t_attn) | 8.43 s | 7.93 s (~neutral) |
| end-to-end | 0.30 tok/s | 0.31 tok/s |

`METAL-ATTN: gpu-wall 3.70s (kernel 0.63s)` → **83% of the attention GPU time is idle
latency** (546 sporadic command buffers × ~5.6 ms). The compute is genuinely fast (0.63 s),
but the per-layer submit latency cancels the projection-matmul savings. The earlier
"16.5 -> 10.5 (MTP on)" number was run-to-run variance, not the offload — corrected here.

## Why, and the real lever
Same wall as the experts: Metal's ~5 ms cold-GPU submit latency dominates sporadic,
dependency-chained per-layer dispatches, and attention *adds* a submit per layer (2/layer
total). The win needs **fewer submits / a hotter GPU**, not faster kernels:
- Fuse attention + experts into one command buffer per layer (1 submit/layer, GPU stays hot).
- Do the residual add / routing on GPU too so there's no CPU glue forcing a sync.
- Or handle S<=4 to amortize (covers MTP verify) — but latency, not compute, is the ceiling.
Fused attention should help at **long context** (kernel time grows past the fixed latency).



## CLEAN warm A/B (2026-07-11, 96 GB, MTP on, ~55% hit, machine idle) — the real result
Earlier "attention neutral" was DRAFT=0 + machine contention. In the realistic MTP config,
warm and uncontended, the combined offload (experts + S<=4 attention) is a genuine win:

| | CPU | Metal | speedup |
|---|---|---|---|
| end-to-end | 49.9s (0.20 tok/s) | 35.1s (0.28 tok/s) | ~1.4x |
| attention | 15.2s | 8.0s | ~1.9x |
| expert-matmul | 12.0s | 7.3s | ~1.65x |
| disk (matched) | 16.2s | 16.4s | - |

Token-exact. GPU still latency-bound (attn kernel 0.70s of 3.04s wall) so more upside exists
if submit count drops, but GPU already beats CPU on both. Measure MTP-on + warm + idle machine;
DRAFT=0 and contention both mislead.

## Status of the offload overall
Expert matmul: real ~1.2-1.3x warm. Attention: correct + token-exact but latency-neutral
at short context. Both are gated by Metal submit latency; reducing submit count is the
single highest-leverage remaining work.

## Iteration 2 (loop, 2026-07-11)
- Interleaved attention q/kv paths, 7->4 barriers: attn gpu-wall 3.04->2.73s. Committed.
- PILOT=1 router-lookahead prefetch: NO effect with Metal (disk 19.5 vs 19.8s) — SPEC=1
  readahead already covers it; the residual disk cost is the CPU pread/slab copy.
- User observation confirmed: GPU sits at idle clock (338 MHz, 23%, 0.56W) — fed in short
  sporadic bursts while CPU streams experts. Structural for a disk-streaming MoE.
- Swap (~46 GB) is normal idle-app paging per user; model is on a separate volume.
- **Zero-copy attention weights** (qalloc: page-align+register all dense QT weights):
  removes ~6 GB GPU-side duplication + upload copies. Token-exact, RSS -3 GB,
  35.1s -> 29.7s (0.34 tok/s) warm — best number yet.
- **Shared-expert fusion**: Phase E folded into the first Metal moe_block as an extra
  expert (rw=1.0, all S rows) — 3 fewer CPU matmuls/layer, bigger GPU submits.
- Next candidate: router on GPU (score matmul + sigmoid + top-8 of 256) to merge the
  attention and expert command buffers into ONE per layer (halves submit count).

## Iteration 2 final A/B (interleaved M/C/M/C, identical speculation + hit-rate)
Metal 33.6/30.9s vs CPU 49.4/51.1s -> **0.31 vs 0.20 tok/s = ~1.56x end-to-end**, token-exact.

## Iteration 3 plan: overlap disk with GPU inside the layer
Router-on-GPU is NOT the right lever: the top-8 result must return to the CPU to decide
which experts to LOAD FROM DISK, so a CPU sync after routing is unavoidable and the CBs
cannot merge on miss layers (~99% of layers at 58% hit). Instead, split each layer's
expert block into two submits: (1) immediately submit the CB for RESIDENT experts
(pin/LRU hits) and let the GPU work while (2) the CPU preads the missed experts in
parallel, then (3) submit the misses CB and combine. Overlaps ~8ms GPU compute with
~38ms disk per block: est. ~3s off a 32s run, and keeps the GPU warmer between submits.
Needs: two-phase moe_block API (begin/end) + non-shared scratch (2-slot ring).

## Iteration 3 result: disk/GPU overlap (begin/end two-phase moe_block)
Resident experts (+fused shared) submit to GPU BEFORE the missed experts' preads;
missed subset follows in a second submit. Token-exact. Warm 96GB:
- expert-matmul 8.96 -> 4.92s (resident compute hidden inside the disk window)
- expert idle latency ~5.7s -> ~0.9s (GPU no longer cold-starts after the load)
- total 28.97s = 0.35 tok/s vs CPU 50.2s = **~1.73x end-to-end**
Loop progression: 0.20 (CPU) -> 0.28 (iter1) -> 0.31 (iter2) -> 0.35 tok/s (iter3).
Build gotcha: `make glm METAL=1` after a default build does not rebuild — touch/clean first.
Remaining: attention CB still ~4s idle (submitted after CPU residual+rmsnorm+DSA-key with
nothing to hide behind); disk ~16s is now the dominant cost and is true model streaming.
