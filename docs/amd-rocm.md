# Running colibrì on AMD GPUs with ROCm

This guide documents a community-tested path for running colibrì on **AMD
Radeon / Radeon Instinct** cards via ROCm/HIP. It was validated on a **Vega 20
(Radeon Pro Vega II / Radeon Instinct MI50, `gfx906`)** — a card with no matrix
cores, and one of the slower HIP expert-tier targets. The same recipe should
apply to `gfx908`, `gfx90a`, `gfx1030`, `gfx1100`, and `gfx1101` after
changing `HIP_ARCH` in the Dockerfile.

> **Scope.** This is the *ROCm/HIP* path — the CUDA tier compiled through HIP.
> For AMD cards without working ROCm support (or when you prefer RADV), see
> [docs/vulkan.md](vulkan.md) instead.

## The hardware tested

| Component | Value |
|---|---|
| GPU | AMD Vega 20 (`gfx906`), 32 GB HBM2 |
| ROCm runtime | 6.3.3 (patched gfx906 image) |
| Host | x86-64, Linux, 16 GB RAM |
| Docker | 24.x with compose v2 |

## Why a Dockerfile

The upstream tree already ships a HIP path (`HIP=1`, `HIP_ARCH=<arch>`, see
`GPU_BACKENDS.md`), but two things block a stock `make deepseek-v4 HIP=1`:

1. **`c/Makefile`'s HIP branch omits two engine-specific CUDA objects.**
   The `CUDA=1` branch sets `INK_CUDA_OBJ` and `DSV4_CUDA_OBJ`; the `HIP=1`
   branch sets neither, so `inkling` and `deepseek_v4` link without their GPU
   objects and silently fall back to CPU.

2. **`c/backend_cuda_ink.cu` and `c/backend_cuda_dsv4.cu` include CUDA headers
   directly** (`<cuda_runtime.h>`, `<cuda_fp8.h>`, `<cublasLt.h>`, …) and use
   raw CUDA names, unlike `c/backend_cuda.cu`, which routes through
   `c/backend_gpu_compat.h`. Under `hipcc` those headers are absent and the
   file does not compile.

The `Dockerfile.colibri` in the repo root works around both **without patching
any source file**: a small `docker-shims/` directory supplies CUDA→HIP shim
headers, and the build passes `INK_CUDA_OBJ=` / `DSV4_CUDA_OBJ=` to `make`.

## Build

From the repo root:

    docker build -t colibri-rocm -f Dockerfile.colibri .

Takes ~4 minutes on a modern x86-64 host. Two `hipcc` passes compile
`backend_cuda_ink.o` and `backend_cuda_dsv4.o`, then `make install` builds
every other engine with `HIP=1 HIP_ARCH=gfx906`.

**Change the arch.** Edit `Dockerfile.colibri` and replace every `gfx906`
with your card's target (`rocminfo | grep gfx` prints it). The string appears
in three `hipcc` commands and one `NVCCFLAGS`.

## Run

    docker run --rm -it \
      --device /dev/kfd \
      --device /dev/dri/card1 \
      --device /dev/dri/renderD129 \
      --group-add video \
      --group-add render \
      --security-opt seccomp=unconfined \
      -v /path/to/models:/models \
      -e COLI_MODEL=/models/<model-dir> \
      -e COLI_CUDA=1 \
      -e COLI_GPUS=0 \
      -e V4_LOADER_LANES=3 \
      -e CUDA_DENSE=1 \
      colibri-rocm \
      run --gpu 0 "Say hi"

Replace `card1` / `renderD129` with the AMD nodes your host exposes. List
them with `ls -l /dev/dri/by-path/`. On the tested host, PCI `03:00.0` was
the AMD card; Intel iGPUs and NVIDIA GPUs must not be passed in (ROCm ignores
them but they add noise).

## Environment variables that matter

| Variable | Value | Why |
|---|---|---|
| `COLI_CUDA` | `1` | Enables the GPU backend. Required. |
| `COLI_GPUS` | `0` | Device index list. **Do not** use `COLI_GPU=1` — the launcher reads it as a device index, not a boolean. |
| `CUDA_DENSE` | `1` | Keeps dense weights on the GPU, freeing host RAM for the expert cache. |
| `V4_LOADER_LANES` | `3` | GPU-tier default. Higher values spill host memory. |
| `DSV4_HYBRID` | `1` | Splits expert misses between GPU and CPU using measured bandwidth. Off by default. |
| `DSV4_CUDA_EXPERT_MIRRORS` | `2048`–`4096` | Upper bound on VRAM expert mirrors. Growth is bounded by free VRAM. |
| `DSV4_CUDA_VRAM_RESERVE_MB` | `600` | Free VRAM kept for transient prefill. Lower = more mirrors. |

## What works, and what does not

**Confirmed working** on gfx906:

- HIP runtime detects the card (`[DSV4 CUDA] device 0: AMD Radeon Graphics 34.3 GB sm_90`).
- Dense matvec tier active (`v4_gpu tier=dense-matvec device=0`).
- Full generation: text is produced, tokens come out correctly.

**Known limitation.** The routed-expert tier is effectively CPU-bound on
`gfx906`. Two factors compound:

- No matrix cores. `GPU_BACKENDS.md` documents that WMMA kernels are compiled
  out for `gfx906`; the portable fallback kernels are slower than AVX2 for
  expert matvec on this class of card.
- The expert tier is **all-or-nothing per token**. A single routed expert
  missing from the VRAM mirror cache sends the *whole token's* MoE to the
  CPU. On a 32 GB card holding an 85 GB model with a small host RAM budget,
  hit rate stays below the threshold where the GPU path can engage.

The result is a working but slow pipeline (~0.03–0.05 tok/s on the tested
Vega 20 for DeepSeek V4 Flash 284B). Dense-tier acceleration is real; expert
acceleration is not, on this hardware class.

For a faster AMD path, the recommended alternatives are:

- **`gfx90a` / `gfx942`** (MI200/MI300): matrix cores, real expert-tier speedup.
- **`gfx1100` / `gfx1101`** (RDNA3): WMMA via rocWMMA; see the upstream
  measurements in `docs/benchmarks.md`.
- **Vulkan** (`docs/vulkan.md`): a separate AMD backend that does not depend
  on ROCm, and works on cards ROCm dropped.

## Model files

The Docker image is model-agnostic. Bind-mount the model directory read-write
(note: **not** `:ro`) — the engine writes a small `.coli_usage` autopin history
and a `.coli_kv*` file next to the model on first run.
If the model directory is `:ro`, the engine still runs but cannot persist
autopin state between sessions and will warn on every start.

## Files added by this guide

- `Dockerfile.colibri` — the build recipe.
- `docker-shims/` — CUDA→HIP compatibility headers, one per CUDA header the
  two `.cu` files include.
- `docs/amd-rocm.md` — this file.

Nothing under `c/` is modified. All workarounds live in the build stage.

## Acknowledgements for this path

The AMD/ROCm path exists because of upstream work that made it possible:

- **Vincenzo Fornaro** ([@JustVugg](https://github.com/JustVugg)) and every
  contributor to **colibrì** — for an engine that already ships a HIP backend,
  a CUDA→HIP compatibility header (`c/backend_gpu_compat.h`), and the design
  notes that made the two Makefile gaps findable.
- **[@mixa3607](https://hub.docker.com/u/mixa3607)** — for the patched gfx906
  ROCm Docker images (`mixa3607/rocm-gfx906`, `mixa3607/llama.cpp-gfx906`)
  that provide a working ROCm 6.3.3 toolchain for cards upstream ROCm has
  dropped.
- The **ROCm/rocWMMA** maintainers at AMD — for the compatibility layer that
  lets a single `.cu` compile for both vendors.
- The **llama.cpp** community — for the gfx906 patches that keep this class
  of card alive in the ROCm ecosystem.

Thank you.
