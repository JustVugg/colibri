# HIP int4 expert-matmul kernel: bandwidth port plan (gfx1100)

Status: steps 1-5 landed (kernel ported to `c/skinny_int4_hip.cu`, repack +
fp16 glue, wired into matmul & fused expert paths behind `COLI_HIP_SKINNY=1`, validated
by `make cuda-test HIP=1`). Step 6 (WMMA path) is implemented and validated but opt-in (needs perf work); rows>8 defaults to chunked skinny.

Colibri's GPU expert matmul on AMD works
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

Steps 1-4 done; 5-6 remain.

1. [done] `c/skinny_int4_hip.cu` (hipcc-only, `#if defined(__HIP__GFX1X__)`): the two
   `__global__` kernels + macros, `scalar_t` -> `half`, drop the torch/at::Tensor
   entry. Plain launcher `coli_hip_int4_gemv(half *C, const half *A,
   const uint8_t *Wrepacked, const half *scale, int Nout, int K, int Mrows)`.
   Add a Makefile rule to compile it under HIP=1 and link the object.
2. [done] `backend_cuda.cu`: in `coli_cuda_tensor_upload` for fmt==2 under HIP, store a
   repacked+re-encoded weight buffer and fp16 scales alongside (or instead of)
   the raw tensor. Add fp32<->fp16 convert kernels around the call.
3. [done] Wire into the single-tensor and grouped-expert matmul paths, gated behind
   `COLI_HIP_SKINNY=1` (default off); fall back to `quant_matmul`.
4. [done] Validate: `make cuda-test HIP=1` must stay green (extend it with an int4
   shape large enough to exercise the kernel), and diff outputs against the
   naive kernel within tolerance.
5. [done] Tuned YTILE/UNRL for gfx1100 (multiProcessorCount reports 48, not 96 —
   the plan's WGP note was right). Sweep confirms YTILE=4/UNRL=2 as the default;
   see results below.
6. [partial] Batched WMMA (RDNA3 matrix-core) path for rows>8, in
   `c/skinny_int4_hip.cu` (`wmma_int4` via rocWMMA, dequants straight from the
   signed-s4 weights + fp32 scale, no repack). Correct and validated, but the
   first-correct version is occupancy/latency-bound and slower than chunked
   skinny at real shapes, so it is opt-in (`COLI_HIP_WMMA=1`); see below.

## Result (step 6: WMMA, opt-in)

`coli_hip_int4_wmma` computes C[rows,Nout]=A@dequant(W)^T with rocWMMA
16x16x16 f16 tiles, one wave per output tile looping K once. It is numerically
correct (relRMS ~2e-4 vs a CPU reference; wired path matches naive within fp16
tolerance) but not yet fast. Pure-kernel time vs chunked skinny (RX 7900 XTX):

    Nout=2048 K=6144 rows=16/32/64 : wmma 589/877/1740us  skinny 67/126/249us
    Nout=6144 K=2048 rows=16/32/64 : wmma 199/300/572us   skinny 43/86/173us

WMMA is 3-10x slower (~4-32 GB/s effective). Root causes and the work needed to
reach the ~40% (~384 GB/s) the handoff targets:
  - Occupancy: one wave per workgroup, and only Nout/16 output tiles of
    parallelism (128 for Nout=2048). Needs split-K (partial sums + atomic add)
    to expose more waves, and/or multiple waves per block sharing a staged
    activation tile.
  - Latency: a __syncthreads every 16 K-elements with no double-buffering, so
    the int4 dequant (VALU) is fully exposed instead of overlapping the WMMA.
  - Weight loads are byte-wise and uncoalesced; should be vectorized (uint4).
  - Epilogue stages to LDS for the per-column scale; bounds-safe direct global
    stores would cut LDS pressure.
Until that lands, the rows>8 default stays chunked skinny (correct, ~3x naive).

## Result (steps 1-5)

Wired: single-tensor `coli_cuda_matmul`, fused `coli_cuda_expert_mlp`, and
grouped `coli_cuda_expert_group` (all behind `COLI_HIP_SKINNY=1`, fall back to
`quant_matmul`/grouped kernels when a shape is ineligible: rows>8, K%16!=0, or
the tile exceeds LDS). The grouped/fused paths reuse one skinny helper and cross
PCIe once per MLP.

Pure kernel time (device events, 500 iters, RX 7900 XTX, CuCount=48), swept over
YTILE x UNRL. Weight bytes per call = O*K/2 = 6.3 MB (fits Infinity Cache, so
absolute GB/s is optimistic vs a fully DRAM-resident stream; the ranking holds):

    O=2048 K=6144 S=1 : YT=4 best, UN=4 7.7us / UN=2 7.8us  (~810 GB/s)
    O=6144 K=2048 S=1 : YT=4 best, UN=4 6.8us / UN=2 7.1us  (~900 GB/s)
    O=2048 K=6144 S=3 : YT=4 UN=2 12.8us  (~490 GB/s)
    O=2048 K=6144 S=5 : YT=4 UN=2 18.1us  (~350 GB/s)

YTILE=4 wins at every shape; UNRL=4 is marginally faster at S=1 but slower at
S>=3, so the default stays YTILE=4/UNRL=2 (override via COLI_INT4_YTILE /
COLI_INT4_UNRL). Naive baseline for the same shapes was ~110-140 GB/s, so the
kernel-only speedup is ~6x.

## End-to-end wall time (single-tensor path)

End-to-end `coli_cuda_matmul` wall time, naive vs skinny (RX 7900 XTX, PCIe
copy overhead identical on both paths):

    O=2048 I=6144 S=1 : 117 -> 85 us   1.38x
    O=6144 I=2048 S=1 : 106 -> 73 us   1.45x
    O=2048 I=6144 S=3 : 180 -> 61 us   2.95x
    O=2048 I=6144 S=5 : 278 -> 74 us   3.74x

S=1 is copy/launch-latency bound end to end (kernel-only gain is larger); the
win grows with S because skinny batches N rows in one launch that reuses the
LDS-staged activation, where the naive kernel does S separate grid launches.
The single-tensor, fused-`expert_mlp`, and grouped-`expert_group` paths are all
wired. Remaining: making the opt-in WMMA path (step 6, `COLI_HIP_WMMA=1`) beat
chunked skinny via split-K + double-buffering + coalesced weight loads.

## Payoff and when to do it

The kernel only helps when experts are resident (no disk) and the matmul is the
bottleneck. Colibri's usual decode is disk-bound (low hit rate), where this will
not move tok/s. It is worth doing for fully-resident configs
(`PIN_GB=all` / `CUDA_EXPERT_GB=auto` filling RAM+VRAM), where a high hit rate
makes GPU matmul the ceiling and the ~3x lands on the compute fraction.
