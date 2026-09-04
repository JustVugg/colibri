# qwen36: VRAM expert tier (CUDA or Vulkan)

Applies colibri's placement concept ("route -> place -> overlap -> learn") to
Qwen3.6-35B-A3B one level up from the GLM disk tier: all 10,240 experts live
in RAM, the **hot** ones are promoted into DEVICE_LOCAL VRAM across one or
more GPUs and computed there through the existing shared CUDA or Vulkan
backend (`backend_cuda.cu` / `backend_vulkan.c` expert-group API — no new
backend).

## How it works

- **Home device:** expert `eid` lives on GPU `eid % n_gpus`; no duplicates.
- **Placement:** routing heat decides who earns VRAM (LFRU semantics from
  `tier.h`, 25%+4 hysteresis). Runtime heat is halved every 1024 decode ticks,
  so a long-lived process can replace experts from an old workload instead of
  permanently freezing its initial hot set. A parallel **warmstart** fills the per-device
  budget before the first token — ordered by a persisted heat table
  (`HEAT_FILE`) when present, so a second run starts fully placed.
- **Decode:** per (token, layer) the resident experts are issued as async
  groups on all devices (`coli_cuda_expert_group_issue/take`); VRAM misses
  fall back to the CPU int8 path and overlap with the in-flight groups, as
  does the shared expert. Placement never changes routing or precision.
- **Memory:** the warmstart frees the RAM int8 copies of VRAM-resident
  experts (rematerialized from the packed int4 copy on LFRU eviction; no
  container access). Peak RSS for the 35B int4 container: ~29 GB with two
  8 GB GPUs.

## Usage

```bash
make -C c qwen36 CUDA=1 CUDA_ARCH=native   # NVCC=/usr/bin/nvcc on distro CUDA
COLI_CUDA=1 COLI_GPUS=0,1 HEAT_FILE=heat.bin CUDA_EXPERT_GB=auto \
OMP_NUM_THREADS=<physical cores> OMP_WAIT_POLICY=ACTIVE OMP_PROC_BIND=close \
SNAP=<container> N_NEW=200 ./c/qwen36 256 4 prompt.txt
```

`cap` (argv[1]) must equal `n_experts` (full RAM residency). int4 containers
only (the int8 container keeps the CPU path). `COLI_TIMERS=1` prints
per-phase timings and tier telemetry.

## Vulkan tier (`make -C c qwen36 VK=1`)

Any Vulkan 1.2 device — AMD via Mesa/RADV (including Polaris cards ROCm
dropped), Intel ANV, NVIDIA. Needs `libvulkan` and `glslc` at build time, like
the GLM Vulkan backend (see [vulkan.md](vulkan.md)).

```bash
make -C c qwen36 VK=1
COLI_VULKAN=1 HEAT_FILE=heat.bin VK_EXPERT_GB=auto \
OMP_NUM_THREADS=<physical cores> COLI_NO_OMP_TUNE=1 \
SNAP=<container> N_NEW=200 ./c/qwen36 256 4 prompt.txt
```

Differences from the CUDA tier:

- **Single device.** `COLI_GPUS` is not read; the backend picks the most
  capable Vulkan device (discrete > integrated).
- **Fill once.** Residency is decided at warmstart — `HEAT_FILE` order when the
  file exists, natural order otherwise — up to `VK_EXPERT_GB` (`auto` = the
  driver's device-local budget minus 1 GB). There are no runtime LFRU swaps:
  the Vulkan weight arena never reclaims a freed slice, so each swap would leak
  one expert of VRAM. Heat still accumulates and saves at exit, so the second
  run starts hot. `QT_NO_WARMSTART=1` switches to filling on first use, which
  is still fill-once.
- **No Resizable BAR needed.** Discrete cards without ReBAR get real VRAM
  residency through the backend's staged uploads (`COLI_VK_STAGED`, see
  [vulkan.md](vulkan.md)).
- **CUDA wins** when a binary is built with both `CUDA=1` and `VK=1`.
- Numerics: the same offset-binary int4 layout as the CUDA upload, so
  `test_qwen36_tier_vk` (built into `make check`) holds the GPU output to
  within 2e-3 relative of the CPU int4 path.

## Measured (Threadripper 3945WX 12C, RTX 3070 8 GB + Quadro RTX 4000 8 GB, Qwen3.6-35B-A3B int4, 200-token decode)

| | 1 GPU (8 GB) | 2 GPUs (16 GB) |
|---|---|---|
| decode tok/s (cold / warm heat) | 9.2 / 9.9 | 10.6 / **11.3** |
| VRAM-resident experts | 4,391 (43 %) | 8,532 (83 %) |
| VRAM hit rate (cold / warm) | 44 % / 95 % | 85 % / 100 % |
| peak RSS | 40 GB | **29 GB** |
| reference: Ollama q4_K_M, same box | 7.5 | 10.5 |

CPU-only baseline of this engine before the tier: 0.35 tok/s.
Numerics: logits cosine vs the f32 CPU reference 0.9992 (dense int8 on),
bit-identical GPU-vs-CPU on the same container (cosine 1.0000001).

## Measured (RX 580 8 GB, i7-7700K, Qwen3.6-35B-A3B int4-gs64, 64-token decode)

Hardware: AMD Radeon RX 580 8 GB (Polaris10, gfx803), Intel Core i7-7700K
(8 threads), Mesa 25.2.8 RADV, no Resizable BAR. GPU clocks were not pinned
(no root on this box to set `power_dpm_force_performance_level`). Commit
56f019b. Prompt: 15 tokens, `N_NEW=64`, greedy decode, container
`qwen36_i4_gs64` (grouped-scale int4, gs=64) — this is the tier's first run
against a gs64 container.

| | CPU-only | Vulkan cold heat | Vulkan warm heat (staged, frozen) | Vulkan mapped path (`COLI_VK_STAGED=0`, frozen) |
|---|---|---|---|---|
| decode tok/s | 0.63 | 6.46 | 7.4 (7.52 / 7.35 across 2 runs) | 4.16 |
| TTFT | 44.65 s | 1.48 s | 1.14 s / 1.31 s | 2.36 s |
| VRAM-resident experts | — | 3,644/10,240 (35.6 %) | 3,644/10,240 (35.6 %) | 3,661/10,240 (35.8 %) |
| VRAM hit rate | — | 35.8 % | 96.7 % | 96.8 % |
| peak RSS | 17.75 GB | 40.78 GB | 40.72 GB | 40.67 GB |

The two frozen-heat staged runs were token-identical to each other and to the
CPU-only baseline (`diff` clean both ways): the gs64 grouped-scale int4
upload path holds bit-for-bit on this card. Staged uploads ran ~1.8x the
mapped-path throughput warm (7.4 vs 4.16 tok/s) — the expected cost of every
non-resident-window access crossing PCIe without ReBAR.
