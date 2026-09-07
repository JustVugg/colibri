# qwen36: Vulkan VRAM expert tier

The same expert tier as [qwen36-cuda-tier.md](qwen36-cuda-tier.md) — all
10,240 Qwen3.6-35B-A3B experts live in RAM, the **hot** ones are promoted into
DEVICE_LOCAL VRAM and computed there — built against the shared Vulkan backend
(`backend_vulkan.c` expert-group API, no new backend) instead of CUDA. The
placement logic in `qwen36_tier.c` is backend-agnostic; only the operations
behind its backend shim differ. Read the CUDA document for how the tier works
(home device, heat, warmstart, the RSS behaviour per container) and for the
placement calibration; this page covers what is different on Vulkan.

## Usage (`make -C c qwen36 VK=1`)

Any Vulkan 1.2 device — AMD via Mesa/RADV (including Polaris cards ROCm
dropped), Intel ANV, NVIDIA. Needs `libvulkan` and `glslc` at build time, like
the GLM Vulkan backend (see [vulkan.md](vulkan.md)).

```bash
make -C c qwen36 VK=1
COLI_VULKAN=1 HEAT_FILE=heat.bin VK_EXPERT_GB=auto \
OMP_NUM_THREADS=<physical cores> COLI_NO_OMP_TUNE=1 \
SNAP=<container> N_NEW=200 ./c/qwen36 256 4 prompt.txt
```

`cap` (argv[1]) must equal `n_experts` (full RAM residency). int4 and int8
containers are promoted; a grouped-scale int8 container is refused with
`[qtier] int8 experts with grouped scales (gs=%d) cannot be expressed on the
GPU (fmt=1 is per-row only) -> CPU path`. `COLI_TIMERS=1` prints per-phase
timings and tier telemetry.

## Differences from the CUDA tier

- **Single device.** `COLI_GPUS` / `COLI_GPU` are not read; the backend picks
  the most capable Vulkan device (discrete > integrated). `COLI_VK_DEV=<index>`
  selects the Vulkan device — the backend's existing selector, documented in
  [ENVIRONMENT.md](ENVIRONMENT.md).
- **Fill once.** Residency is decided at warmstart — `HEAT_FILE` order when the
  file exists, natural order otherwise — up to `VK_EXPERT_GB` (`auto` = the
  driver's device-local budget minus 1 GB). There are no runtime LFRU swaps:
  the Vulkan weight arena never reclaims a freed slice, so each swap would leak
  one expert of VRAM. Heat still accumulates and saves at exit, so the second
  run starts hot. `QT_NO_WARMSTART=1` switches to filling on first use, which
  is still fill-once.
- **Budget accounting** charges experts with the same allocation-granularity
  table the CUDA tier measured for `cudaMalloc`. On Vulkan that is an
  approximation of the arena's real footprint, on the conservative side.
- **Experts only.** The resident dense trunk (`COLI_PLACE`, `COLI_LMHEAD_GPU`)
  and the fp8 streaming mode are CUDA-only today: on a Vulkan build the tier
  refuses them with one stderr line and those pieces stay on the CPU path.
- **No Resizable BAR needed.** Discrete cards without ReBAR get real VRAM
  residency through the backend's staged uploads (`COLI_VK_STAGED`, see
  [vulkan.md](vulkan.md)).
- **CUDA wins** when a binary is built with both `CUDA=1` and `VK=1`.
- Numerics: the same offset-binary int4 layout as the CUDA upload, so
  `test_qwen36_tier_vk` (built into `make check`) holds the GPU output to
  within 2e-3 relative of the CPU int4 path.

## Measured (RX 580 8 GB, i7-7700K, Qwen3.6-35B-A3B int4-gs64, 64-token decode)

Hardware: AMD Radeon RX 580 8 GB (Polaris10, gfx803), Intel Core i7-7700K
(8 threads), Mesa 25.2.8 RADV, no Resizable BAR. GPU clocks were not pinned
(no root on this box to set `power_dpm_force_performance_level`). Commit
bb16ab3. Prompt: 15 tokens, `N_NEW=64`, greedy decode, container
`qwen36_i4_gs64` (grouped-scale int4, gs=64) — this is the tier's first run
against a gs64 container.

| | CPU-only | Vulkan cold heat | Vulkan warm heat (staged, frozen) | Vulkan mapped path (`COLI_VK_STAGED=0`, frozen) |
|---|---|---|---|---|
| decode tok/s | 0.63 | 6.40 | 5.99 (6.38 / 5.60 across 2 runs) | 2.44 |
| TTFT | 44.65 s | 1.50 s | 1.48 s / 1.50 s | 2.48 s |
| VRAM-resident experts | — | 3,663/10,240 (35.8 %) | 3,663 and 3,655/10,240 (35.8 / 35.7 %) | 3,655/10,240 (35.7 %) |
| VRAM hit rate | — | 36.1 % | 96.8 % | 96.8 % |
| peak RSS | 17.75 GB | 40.73 GB | 40.67 / 40.69 GB | 40.69 GB |

The two frozen-heat staged runs were token-identical to each other and to the
CPU-only baseline (`diff` clean both ways), and so was the mapped-path run:
the gs64 grouped-scale int4 upload path holds bit-for-bit on this card.
Staged uploads ran ~2.5x the mapped-path throughput warm (5.99 vs 2.44
tok/s) — the expected cost of every non-resident-window access crossing PCIe
without ReBAR. With the clocks unpinned the two warm runs spread 12 %
(6.38 / 5.60), and the resident count moves a few experts run to run because
the budget follows the live `VK_EXT_memory_budget` free figure.
