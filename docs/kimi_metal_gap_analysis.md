# Kimi K3 Metal Gap Analysis

Every Kimi K3 operation with CPU, Vulkan, and Metal implementation status.
All operations originate in `c/kimi_k3.c` unless noted otherwise.

Legend: = implemented, = CPU only, = no GPU backend

---

## 1. Forward Pass (`step_chunk` / `kimima_forward`)

| Op | Description | CPU | Vulkan | Metal |
|---|---|---|---|---|
| Embedding lookup | `st_read_slice_f32` reads `model.embed_tokens.weight` by token id | | | |
| res_mix (AttnRes) | Softmax mix over block snapshots + prefix_sum | | | |
| rmsnorm_ | Per-row RMSNorm with gamma weights | | | |
| kda_forward | KDA attention: conv, recurrence, O projection | | | |
| mla_forward | MLA attention: Q/KV/G projections, cache, absorption, softmax, decode, gate | | | |
| moe_forward | Stable LatentMoE: router, top-K, latent, experts, shared | | partial | partial |
| dense_forward | Dense FFN: gate, up, SiTU-GLU, down | | | |
| Residual add | `prefix_sum = mlp` | | | |
| Head forward | res_mix, RMSNorm, lm_head projection | | | |
| Sampling | softmax, top-p, temperature, random weighted pick | | | |

---

## 2. KDA Attention (`kda_forward`)

| Op | Description | CPU | Vulkan | Metal |
|---|---|---|---|---|
| Conv4 + SiLU | Causal convolve q/k/v withconv state management | | | |
| L2 normalize q/k | Per-head L2 normalization | | | |
| Decay compute | `gk = -5*(exp(A_log)*(W_fb W_fa x + dt_bias))` — per-channel gate | | | |
| Recurrence | `S = (I - kkT)*Diag(e^gk)*S + kvT` — per-head 128x128 state | | | |
| O projection | `W_o[sig(W_g x) * RMSNorm_head(St*q)] — 3 matmuls + gate | | | |

---

## 3. MLA Attention (`mla_forward`)

| Op | Description | CPU | Vulkan | Metal |
|---|---|---|---|---|
| Q projection | Q projection matmul | | | |
| Q decode | Q decode matmul | | | |
| KV projection + NoPE | KV projection matmul + NoPE (rotary-style, not true RoPE) | | | |
| Cache write | Write L (latent) and R (context) rows to KV cache | | | |
| G projection + abs gate | G projection matmul, then sigmoid gate on O projection | | | |
| Absorption | `qr[t,h] = (x[t].q * v_b)` — row dot product, int4 dequant | | | |
| Scoring | `soc[t,h,u] = x[t].q * L[u] + qr[t,h] * R[u]` | | | |
| Softmax (causal) | Standard softmax over valid attention mask | | | |
| Context decode (latent) | `hd_v[t] = ctx[t,h] * v_b` — per-head int4 dequant + reduce | | | |
| Context decode (full) | `ctx[t,vb] = ctx[t,h] * V[vb*vb:h*vb+h]` — per-head int4 dequant | | | |
| O projection | gated output projection matmul | | | |

---

## 4. MoE (`moe_forward`)

| Op | Description | CPU | Vulkan | Metal |
|---|---|---|---|---|
| Router matmul | `sig[t,e] += bias[e]` — full sigmoid on scores + score-correction bias | | | |
| Top-K select | `rtop8(sig, E, K, Ksel, topp, 1)` — top-16 on biased scores, renormalized | | | |
| Shared expert gate | `W * x` — Latent up projection for latent space Ops | | | |
| Latent up projection | `W * x` — per-token 3584 | | | |

---

## 5. Expert Apply (`expert_apply` / `experts_apply_union`)

| Op | Description | CPU | Vulkan | Metal |
|---|---|---|---|---|
| W1 (gate, MXFP4+int8 | `W1 * x` — MXFP4 quantized layer (QAT) | | | |
| W2 (down) matmul | `W2 * situf_(g*u)` — MXFP4 quantized layer (QAT) | | | |
| W3 (shared-expert branch) matmul | `W3 * situf_(((W1 * x) - W3))` — MXFP4 quantized layer (QAT) | | | |
| SiTU-GLU activation | `4*tanh(g/4)*σ(g) * 25*tanh(u/25)` elementwise activation | | | |

---

## 6. Shared Experts (in `moe_forward`)

| Op | Description | CPU | Vulkan | Metal |
|---|---|---|---|---|
| sh_gate matmul | `W_sh_gate * x` — full-width shared gate | | | |
| sh_up matmul | `W_sh_up * x` — full-width shared up | | | |
| SiTU-GLU (shared) | Elementwise on gate + up | | | |
| sh_down matmul | `W_sh_down * activation` — full-width shared down | | | |
| Residual add (shared) | `out += sd` — add shared expert output to MoE output | | | |

---

## 7. Misc / Infrastructure

| Op | Description | CPU | Vulkan | Metal |
|---|---|---|---|---|
| Token saving | `st_write_slice_f32` — write embedding row to slot for mixed input | | | |
| KV cache alloc | `kv_alloc` — allocate L and R cache arrays per non-KDA layer | | | |

---

## Metal Detail: Current Capabilities

### `backend_metal.h` API

| Function | Status | Notes |
|---|---|---|
| `coli_metal_init()` | | Init device, queue, residency set |
| `coli_metal_reload()` | | Re-compile shaders after source change |
| `coli_metal_register()` / `unregister()` | | Pin/unpin host pointer for zero-copy |
| `coli_metal_matmul()` | | fmt=1 (int8), fmt=2 (int4-g64), fmt=3 (int2), fmt=4 (grouped int4) |
| `coli_metal_gemm()` | | Large row-batch GEMM (prefill), fmt=1/2/4, chunked dispatch |
| `coli_metal_layer_decode()` | | Full MLA decode layer: RMSNorm, attention, shared exp, router, top-8, down |
| `coli_metal_moe_block()` | | Batched routed expert SwiGLU, fmt=1/2 |
| `coli_metal_moe_block_begin()` / `_end()` | | Async two-phase: submit without wait, then scatter |
| `coli_metal_rtop8()` | | Standalone top-8 select (serial + parallel variants) |
| `coli_metal_stats()` | | Reset counters |
| `coli_metal_mem_info()` | | Residency, usage, slab info |
| `coli_metal_shutdown()` | | Teardown |

### Inline Shader Kernels (SIL)

| Kernel | Purpose | Formats |
|---|---|---|
| `mm_gemv` | Quantized GEMV, one threadgroup per 4 output elements | fmt 0..4 |
| `moe_gemv` | Batched routed GEMV (indirect via erow[] address arrays) | fmt 0..4 |
| `r_router` | Router GEMV: `logit[s][e] = x[s].w_e` | f32 weights |
| `r_top8` / `r_top8_par` | Top-K: sigmoid+bias, top-p truncation, renormalize, routed scale | f32 |

### Shader Pipeline States

| Pipeline | Purpose |
|---|---|
| `g_gemv` | Quantized GEMV (`mm_gemv`) |
| `g_moe_gemv` | Batched routed GEMV (`moe_gemv`) |
| `g_r_router` | Router logits (`r_router`) |
| `g_r_top8` | Serial top-8 select (`r_top8`) |
| `g_r_top8p` | Parallel top-8 select (`r_top8_par`) |
| `g_a_rms` | RMSNorm (attention path) |
| `g_a_copy` | Row copy helper (extract single row from S×AH buffer) |
| `g_a_add` | Residual add (`axr_ += aout_`) |
| `g_moe_silu` | Dual silu activation (gate, up) |

### Metal `layer_decode` Attention Kernels

The fused decode layer (`coli_metal_layer_decode`) encodes attention inline via
`encode_attention()` which dispatches per-halbkhead loops. These are not separate
pipelines — they are encoded into the same command buffer as a loop over the 96
half-heads (each half-head is 64 dims for KDA / the MLA equivalent). The kernels
used are compiled at runtime via `MTLLibrary` source strings embedded in the
function body. These handle:

- RMSNorm of Q and KV inputs
- RoPE / NoPE via `generate_rope_phase()`
- Q × decoded-KV (absorption)
- Causal softmax
- Context decode (latent + full)
- O projection with sigmoid gate

---

## Vulkan Detail: Current Capabilities

### `backend_vulkan.h` API

| Function | Status | Notes |
|---|---|---|
| `coli_vk_init()` | | Init device, queue, pipeline |
| `coli_vk_matmul()` | | fmt=1 (int8), fmt=2 (int4-g64), fmt=7 (MXFP4 integrated) |
| `coli_vk_expert_upload()` | | Upload expert tier to VRAM |
| `coli_vk_expert_apply()` | | Single expert apply (W1, W2, W3) |
| `coli_vk_matmul_pair()` | | Paired W1/W3 matmul for GLU |
| `coli_vk_set_budget()` | | Set VRAM pressure budget |
| `coli_vk_mem_info()` | | Memory and tier stats |
| `coli_vk_shutdown()` | | Teardown |

### Key Gap: fmt=7 MXFP4

Vulkan supports MXFP4+int8 via `fmt=7` in `coli_vk_matmul` and `coli_vk_pair`.
Metal does **not** support `fmt=7` today — only fmt=1/2/3/4. This means the
expert W1/W2/W3 MXFP4 weights cannot run on Metal and fall back to CPU.

---

## Summary of Gaps

### Critical (blocks full Metal inference)

1. **KDA attention** — No Metal kernel for any of KDA's 96 conv4, recurrence, decay, or
   O projection. All 69 KDA layers run on CPU.
2. **MLA attention (non-fused)** — `coli_metal_layer_decode` covers only S1 (decode,
   single token). Prefill (S>1, batched MLA) and the per-halbkhead inline kernels
   have not been extracted to a standalone `coli_metal_mla_forward()` API.
3. **MXFP4 matmul** — Metal `mm_gemv` does not include `fmt=7`. Vulkan has it.
4. **rmsnorm_, SiTU-`g` — No Metal kernel for standalone RMSNorm on arbitrary tensors.
   The `g_a_rms` pipeline exists but is only callable inside `layer_decode`.
5. **res_mix (AttnRes)** — No Metal kernel for the block-snapshot softmax mix.
6. **Sampling** — No GPU softmax + top-p + temperature over 262k vocab. Trivial matmul, but not implemented.

### Important (performance blockers)

7. **Stand-alone matmul wrapper** — `coli_metal_matmul()` exists and works, but
   `kimi_k3.c` has no `#ifdef COLI_METAL` dispatch to call it from `w_matmul()`.
   The GLM engine (`colibri.c`) has this dispatch; K3 does not.
8. **Dense forward** — 24 MLA layers use `dense_forward` (3 matmuls + SiTU-GLU).
   No Metal wrapper exists.
9. **Head forward** — Final res_mix + RMSNorm + lm_head. No Metal path.
10. **Conv4 + SiLU** — KDA conv state update. No GPU kernel.

### Lower priority

11. **Embedding lookup** — Random-access row read from safetensors. Trivial to GPU
    but I/O-bound anyway.
12. **KV cache alloc/write** — CPU-side allocation. Could be moved to unified memory.
13. **L2 normalize** — Used in KDA for q/k. Small kernel, but gated by KDA support.

### What would make Metal 24/93 layers on GPU today?

| Required | Count | Impact |
|---|---|---|
| Standalone RMSNorm kernel | 1 | Enables dense, head, and any arbitrary tensor |
| `w_matmul` → `coli_metal_matmul` dispatch in `kimi_k3.c` | 1 | Enables dense_forward and head forward |
| SiTU-GLU kernel | 1 | Needed for dense + shared experts on GPU |
| Embedding on GPU | 1 | Avoids CPU-GPU boundary for first layer |
| Head forward Metal path | 1 | End-to-end for the 24 MLA layers |
| **Subtotal** | **5** | **24/93 layers (MLA-only models or MLA-only passages)** |

### What would make Metal 93/93 layers on GPU?

The above 5, plus:

| Required | Count | Impact |
|---|---|---|
| KDA full kernel (conv + recurrence + decay + O) | 1 | The single largest gap — 69 layers |
| MXFP4 matmul (`fmt=7`) | 1 | Expert W1/W2/W3 |
| res_mix kernel | 1 | AttnRes every layer, every forward |
| MLA full forward (S>1) | 1 | Prefill for MLA layers |
| **Subtotal** | **11** | **93/93 layers, decode + prefill, everything on GPU** |

---

## Phase 2: Reusable Metal Kernels — Vulkan vs Metal

Cross-references every Vulkan shader / API call against Metal's existing kernels and
API surface. Identifies what can be reused immediately, what needs extraction from a
larger Metal function, and what must be implemented from scratch.

| Operation | Vulkan | Metal | Reusable? | Action |
|---|---|---|---|---|
| Quantized GEMV | `qmatmul.comp` → `coli_vk_matmul()` (fmt 1,2,4,5,7) | `mm_gemv` → `coli_metal_matmul()` (fmt 1,2,3,4) | **Yes** — same algorithm, more fmt coverage | REUSE Metal. Add `#ifdef COLI_METAL` dispatch in `kimi_k3.c:w_matmul()` to call `coli_metal_matmul()` for fmt 1,2,3,4 |
| RMSNorm | `rmsnorm.comp` → `coli_vk_attn_qprep()` chain | `g_a_rms` pipeline → inside `layer_decode` only | **Partial** — pipeline exists, no standalone API | EXTEND: extract `g_a_rms` to `coli_metal_rmsnorm(src, dst, w, n, d)` standalone call |
| Fused gate+up+silu | `qmatmul_gate_up.comp` → `coli_vk_gate_up()` (fmt 3,6,5,7) | `moe_gemv` + `g_moe_silu` inside `moe_block()` | **Partial** — kernels exist inside MoE path | EXTEND: extract `coli_metal_gate_up(x, wgate, wup, out, fmt, n, d)` standalone API |
| Batched expert (MoE) | `coli_vk_expert_group()` → `_issue()` / `_take()` | `moe_block()` / `moe_block_begin()` / `_end()` | **Yes** — Metal has both sync + async variants | REUSE Metal. Wire `moe_block_begin/end` into `kimi_k3.c:expert_apply_union()` |
| Router GEMV | General `qmatmul` path | `r_router` → `coli_metal_rtop8()`, serial/parallel | **Yes** | REUSE Metal (already supports top-K with sigmoid+bias+renorm) |
| Top-K select | No dedicated kernel | `r_top8` / `r_top8_par` pipelines | **Yes** (Metal-only) | REUSE Metal already covers this |
| KV cache management | `VkKvLayer` → `kv_ensure()` / `kv_row()` / `kv_reset()` | Encoded inside `layer_decode()` inline | **Partial** — cache logic exists but not standalone | EXTEND: extract `coli_metal_kv_ensure(layer, ntoks, nheadd, ldim, rdim)`, `kv_row()`, `kv_reset()` |
| MLA attention absorb | `attention_absorb.comp` → `coli_vk_attention_absorb()` + `_project()` | Inline in `layer_decode` (per-halbkhead loop) | **Partial** — logic exists, embedded | EXTEND: extract to `coli_metal_attn_absorb(q, v_b, L, R, ctx, out)` standalone API |
| MoE silu activation | Fused inside `qmatmul_gate_up.comp` | `g_moe_silu` pipeline | **Yes** | REUSE Metal's `g_moe_silu` covers the elementwise SiLU dual activation |
| MXFP4 matmul | `qmatmul.comp` fmt=7 → `coli_vk_matmul()` | **Not supported** (fmt 0..4 only) | **No** — Vulkan-only | IMPLEMENT: add `fmt=7` block to `mm_gemv` and `moe_gemv` shaders + grow `coli_metal_matmul` switch. Critical for Kimi K3 expert weights (W1/W2/W3) |
| Dual-GEMV pair | `coli_vk_matmul_pair()` — two resident matmuls on one input | **No equivalent** | **No** — Vulkan-only | IMPLEMENT: `coli_metal_matmul_pair(src, w1, w3, out, fmt, n, d1, d3)`. Needed for KDA conv+recurrence projections and GLU gate+up fusion |
| Second device (dual GPU) | `coli_vk_init_dev2()` — expert tier on second GPU | Metal only supports one device | **No** | N/A — Apple Silicon has single GPU; not applicable |
| VRAM pressure-proofing | `mem_budget()` + `alloc_priority()` | **No equivalent** | **No** — Vulkan-only | N/A — Metal uses residency sets with automatic eviction; not a gap |
| Large-batch GEMM | `qmatmul` with 2D block dispatch | `coli_metal_gemm()` (full chunked 2D dispatch) | **Yes** — Metal has it, Vulkan doesn't | REUSE Metal. Wire into `kimi_k3.c:w_matmul()` for prefill S>1 |
| SiLU activation | Fused inside `gate_up` | `g_moe_silu` inside `moe_block` | **Partial** — exists but not standalone | EXTEND: extract to `coli_metal_silu(src, dst, n)` for dense + shared expert paths |

### Summary by Action Category

| Action | Count | Impact |
|---|---|---|
| **REUSE immediately** (dispatch from `kimi_k3.c`) | 6 | quantized matmul, MoE block, router, top-K, GEMM, silu |
| **EXTEND** (extract from larger Metal function) | 5 | RMSNorm, gate_up, KV cache, attn_absorb, silu standalone |
| **IMPLEMENT** (new kernel + API) | 2 | MXFP4 matmul (fmt=7), dual-GEMV pair |
| **N/A** (not applicable to Metal) | 2 | second device, VRAM pressure |
| **TOTAL** | **15** | **Cross-references every Vulkan op against Metal** |

### Critical Path to 93/93

The single most impactful implementation is **MXFP4 matmul (`fmt=7`)** in Metal.
Every Kimi K3 expert layer uses MXFP4 weights for W1/W2/W3. Without it, expert
computation falls back to CPU. Once `fmt=7` is added, the existing `mm_gemv`,
`moe_gemv`, and `moe_block` pipelines can cover all expert matmuls with minimal
additional code — the formatting logic is already there, just needs a new
dequantization pattern.

---

## Build Status

| Configuration | Result |
|---|---|
| `make kimi_k3` | Compiles, runs CPU-only |
| `make METAL=1 kimi_k3` | Compiles, Metal scaffolding wired |
| `K3_METAL=1 make METAL=1 kimi_k3` | Metal init/shutdown active; forward NOT IMPLEMENTED, falls through to CPU |

### Phase 4 Changes (scaffolding only)

- `W` struct: added `void *metal` alongside `void *vk`
- `kimima_forward_metal()`: entry point exists, returns 0 (NOT IMPLEMENTED)
- `w_matmul()`: `#ifdef COLI_METAL` dispatch hook present, falls through to CPU
- `model_init()`: reads `K3_METAL` env, calls `coli_metal_init()`, logs status
- `main()`: three exit paths all call `coli_metal_shutdown()` on cleanup
- `Makefile`: `$(METAL_OBJ)` added to kimi_k3 link line
- `K3_METAL` env flag documented in header comment
