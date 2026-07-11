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
- [ ] `backend_metal.{h,mm,metal}` + Makefile branch.
- [ ] `moe()` batched integration + zero-copy slabs.
- [ ] Kernel-correctness test + token-exact end-to-end validation.
- [ ] A/B benchmark on the real model.
