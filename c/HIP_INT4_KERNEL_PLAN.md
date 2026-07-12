# HIP int4 expert-matmul kernel: bandwidth port plan (gfx1100)

Status: documented, not started. Colibri's GPU expert matmul on AMD works
correctly but is bandwidth-inefficient. This is the plan to close the gap by
reusing the tuned kernel work from the vLLM ROCm fork.

## Objective

Replace colibri's naive `quant_matmul` (fmt=2 / int4) on gfx1100 with a
bandwidth-optimal kernel, for the resident-expert decode path. Target ~3x the
weight-streaming bandwidth (matching the vLLM skinny kernel's ~40% of peak).

## Measured baseline (RX 7900 XTX, gfx1100)

Microbench of the current `quant_matmul` at real expert shapes (peak ~960 GB/s):

    gate/up  O=2048 I=6144 S=1 : 0.045 ms/call, ~140 GB/s (~15% of peak)
    down     O=6144 I=2048 S=1 : 0.057 ms/call, ~110 GB/s (~12% of peak)
    batched  O=2048 I=6144 S=3 : 0.123 ms/call,  ~51 GB/s

Activations stay resident in Infinity Cache (an "activations re-read from DRAM"
model exceeds peak), so weight streaming is the real bottleneck. There is ~3x
headroom.

A cheap colibri-native fix (vectorized 16-byte uint4 loads, each packed byte
read once) was measured and does NOT help: 0.95x / 0.60x / 1.28x. For S=1 it is
slower because fewer threads each doing more work reduces memory-level
parallelism. The 3x is not reachable by a simple rewrite; it needs the full
sophisticated kernel.

## Reusable source (vLLM ROCm fork)

Fork: `/mnt/scratch/vllm/vllm-src`, branch `main` (was `port-rdna-hybrid-w4a16`).
- `csrc/rocm/skinny_gemms_int4.cu` — `wvSplitK_int4_hf_sml_` / `wvSplitK_int4_hf_`
  and entry `wvSplitK_int4_g`. Tuned skinny int4 GEMV, ~40-50% of peak at M<=5.
  Supports `GROUP_SIZE=0` (per-row scale) and symmetric `uint4b8`.
- `csrc/rocm/attention.cu` — RDNA3 WMMA typedefs/intrinsics
  (`__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`, `floatx8`, `bit16x16`) in the
  `#elif defined(__GFX11__)` block, for a future batched (M=8-64) matrix-core path.
- `PHASE2_WMMA_HANDOFF.md` (repo root) — full design notes.

Key techniques that produce the win (all in the skinny kernel):
`__builtin_nontemporal_load` vectorized weight loads, `__builtin_amdgcn_fdot2`
packed fp16 dot products, `REDUCE_SUM_DPP_WAVE32` DPP reductions, LDS activation
staging, and YTILE (multiple output rows per workgroup reuse one staged
activation).

## Format gaps (why it is not a drop-in)

Colibri and the vLLM kernel differ on all three axes; a one-time re-encode at
`coli_cuda_tensor_upload` is required:

1. Encoding: colibri fmt=2 is signed two's-complement s4 (`n&8 ? n-16 : n`,
   range -8..7); the vLLM kernel expects `uint4b8` (nibble-8). Convert with an
   XOR 0x88 (see `offset_to_signed_s4` already in `backend_cuda.cu`).
2. Packing: colibri packs sequential 2-nibbles/byte `[O, (I+1)/2]`; the vLLM
   kernel uses an ExLlama-style interleaved uint32 layout. Repack at upload.
3. Dtype: colibri is fp32 activations / scales / output; the kernel is fp16/bf16.
   Convert `x` fp32->fp16 per call, cache fp16 scales at upload, convert output
   back to fp32.

## Port plan

1. `c/skinny_int4_hip.cu` (hipcc-only, `#if defined(__HIP__GFX1X__)`): the two
   `__global__` kernels + macros, `scalar_t` -> `half`, drop the torch/at::Tensor
   entry. Plain launcher `coli_hip_int4_gemv(half *C, const half *A,
   const uint8_t *Wrepacked, const half *scale, int Nout, int K, int Mrows)`.
   Add a Makefile rule to compile it under HIP=1 and link the object.
2. `backend_cuda.cu`: in `coli_cuda_tensor_upload` for fmt==2 under HIP, store a
   repacked+re-encoded weight buffer and fp16 scales alongside (or instead of)
   the raw tensor. Add fp32<->fp16 convert kernels around the call.
3. Wire into the single-tensor and grouped-expert matmul paths, gated behind
   `COLI_HIP_SKINNY=1` (default off); fall back to `quant_matmul`.
4. Validate: `make cuda-test HIP=1` must stay green (extend it with an int4
   shape large enough to exercise the kernel), and diff outputs against the
   naive kernel within tolerance.
5. Tune YTILE/UNRL/A_CHUNK for gfx1100 (96 CUs, WGP=48 — note the sYT heuristic
   miscalibration called out in the handoff). Microbench each shape.
6. Later (optional): the batched M=8-64 matrix-core path via the WMMA intrinsics,
   for #80's grouped-expert continuous-batching path.

## Payoff and when to do it

The kernel only helps when experts are resident (no disk) and the matmul is the
bottleneck. Colibri's usual decode is disk-bound (low hit rate), where this will
not move tok/s. It is worth doing for fully-resident configs
(`PIN_GB=all` / `CUDA_EXPERT_GB=auto` filling RAM+VRAM), where a high hit rate
makes GPU matmul the ceiling and the ~3x lands on the compute fraction.
