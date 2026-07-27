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
(+13.9%). Greedy decode was byte-identical to the raw-tier baseline and measured
6.19→7.12 tok/s over 32 tokens (+15.0%), and 6.75→7.31 tok/s over 128 tokens
(+8.3%). Pinned asynchronous staging reduced placement from 197 to 157 seconds;
aligned direct reads reduced it further to 80–103 seconds across two runs, with
the 110 GB archive read accounting for 18.94–19.15 seconds. These results are a
general compression guarantee: incompressible records are rejected, and the
archived DietGPU dependency makes this an experimental build-time option.
Live `REPIN` is disabled because it would invalidate the sidecar's fixed expert
order.

## The GPU-resident pipeline (`COLI_CUDA_PIPE`)

`COLI_CUDA_PIPE=2` keeps the residual stream on-device across layers: rmsnorms,
residual adds, router GEMMs and the shared expert run on the GPU while the CPU
expert loop runs uninterrupted, with batched attention and grouped expert
uploads at prefill. On a single-GPU host this also pays at decode (S=1):
**+49%** measured on a 5070 Ti ([#273](https://github.com/JustVugg/colibri/issues/273)/#274);
on multi-GPU hosts the per-layer P2P hops cancel the gain, so the decode gate is
device-count aware. `COLI_CUDA_TC_W4A16=1` enables Tensor-Core int4×fp16 mixed
dispatch for batched rows (pays at ≥16 rows).

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
