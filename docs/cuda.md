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
COLI_CUDA=1 COLI_GPU=0 CUDA_DENSE=1 SNAP=/nvme/glm52_i4 ./glm 64 4 4
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
STATS=stats.txt SNAP=/nvme/glm52_i4 ./glm 64 4 4   # collect routing frequencies first
COLI_CUDA=1 COLI_GPU=0 CUDA_EXPERT_GB=16 \
PIN=stats.txt PIN_GB=160 SNAP=/nvme/glm52_i4 ./glm 64 4 4

# multi-GPU expert tier, 150 GB total budget across six 32 GB devices
COLI_CUDA=1 COLI_GPUS=0,1,2,3,4,5 CUDA_EXPERT_GB=150 \
CUDA_DENSE=1 PIN=stats.txt PIN_GB=300 RAM_GB=226 \
SNAP=/nvme/glm52_i4 ./glm 64 4 4

# large-RAM host: fill safe VRAM, then keep every remaining expert in RAM
COLI_CUDA=1 COLI_GPUS=0,1,2,3,4,5 CUDA_EXPERT_GB=auto \
CUDA_DENSE=1 COLI_CUDA_ATTN=1 PIN=stats.txt PIN_GB=all RAM_GB=auto \
SNAP=/nvme/glm52_i4 ./glm 64 4 4
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
