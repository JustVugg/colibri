# DeepSeek V4 Flash — CUDA/VRAM expert tiering (colibri)

This document describes the DS4 engine with the **CUDA/VRAM expert tiering**
added on top of the target-only CPU engine (`docs/deepseek-v4.md`). Same source
builds on Linux and Windows with different commands; the engine
(`c/deepseek_v4.c`) is shared and untoched by the tier.

## Scope

- **VRAM expert tier**: the top-M hottest routed experts per layer (ranked by
  usage history) stay resident in VRAM via `coli_dsv4_cuda.dll` / the Linux CUDA
  link; the rest comes from the RAM cache (LRU + usage pinning, `.coli_usage`
  history in the model dir) and cold streaming from disk (O_DIRECT).
- Dense layers (43, ~6.27 GiB fp8) and attention stay on **CPU/RAM**, like the
  target-only build. Only routed experts are GPU-eligible.
- Numerics are identical to the CPU path (verified A/B on real checkpoints).
- Missing `coli_dsv4_cuda.dll` (Windows) → silent CPU fallback, no error.
- **Tool use / function calling**: OpenAI `tools` are supported with the
  checkpoint's native DSML format (`｜DSML｜tool_calls` blocks, `<tool_result>`
  merged into user turns) — including streaming, multi-turn tool results,
  `tool_choice` none/auto/required/forced, and the Anthropic `/v1/messages`
  translation. Python-only (`openai_server.py`), engine untouched.

**Not wired up (yet):** on-disk KV persistence (`.coli_kv`), DSpark/MTP
speculation (`V4_MTP=0` default), non-greedy sampling and more than one KV
slot in serve mode.

## Download

```bash
hf download deepseek-ai/DeepSeek-V4-Flash-0731 \
  --local-dir /path/to/DeepSeek-V4-Flash
```

Official HF checkpoint, **no weight conversion** (routed experts fp4 + dense
fp8-e4m3, consumed natively). First run creates `.coli_usage` next to the model.

## Build — Linux

Toolchain: `gcc`/`make`, CUDA Toolkit (`nvcc`), OpenMP. From the repo `c/` dir:

```bash
make -f Makefile.deepseek-v4 deepseek-v4 CUDA=1 CUDA_ARCH=sm_86
```

- `CUDA_ARCH`: `sm_86` (RTX 30xx), `sm_89` (RTX 40xx), or `portable`
  (multi-gencode sm_80→sm_120); default `native` (SASS for the local GPU).
- `CUDA_HOME` defaults to `/usr/local/cuda` (else set it).
- Expected: `deepseek_v4` links `libcudart`/`libcublasLt`; first run prints
  `[DSV4 CUDA] device 0: <GPU> ... sm_XX` and `v4_cuda_tier ...`.

## Build — Windows

Toolchain: MinGW-w64 (`w64devkit` or MSYS2: `gcc`+`make`+`sh.exe`) for the
engine; **MSVC Build Tools 2022** + **CUDA Toolkit** (≥ 12.8) for the CUDA DLL.

Why the split: nvcc+MSVC emit a COFF object MinGW cannot link (MSVC runtime
refs), so the kernels live in a standalone `coli_dsv4_cuda.dll` loaded at
runtime by the MinGW host via `LoadLibrary` (`c/backend_loader_dsv4.c`) —
the same `CUDA_DLL=1` pattern GLM uses. `deepseek_v4.c` is untouched.

```cmd
:: 1) CUDA DLL — from the "x64 Native Tools Command Prompt for VS 2022",
::    with nvcc on PATH and the MSYS2/w64devkit bin dir on PATH (for sh.exe):
cd c
make -f Makefile.deepseek-v4 dsv4-cuda-dll CUDA=1 CUDA_ARCH=sm_86

:: 2) Engine host — any shell with make+gcc (w64devkit needs LTO=0):
make -f Makefile.deepseek-v4 deepseek-v4 CUDA=1 CUDA_ARCH=sm_86 LTO=0
```

- Runtime PATH: add `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\x64`
  (CUDA 13.x keeps the runtime DLLs in `bin\x64`, not `bin`).
- Expected: `coli_dsv4_cuda.dll` with exactly the 5 engine-facing exports
  (`dsv4_cuda_init`, `dsv4_cuda_shutdown`, `dsv4_cuda_tensor_free`,
  `dsv4_cuda_upload_fp4`, `dsv4_cuda_expert_group`); `deepseek_v4.exe` runs
  even without the DLL (CPU fallback).
- Note: the top-level `Makefile` target `deepseek-v4` does **not** propagate
  `CUDA=1`; always build through `Makefile.deepseek-v4`.

## Use

```bash
# interactive chat (native TUI, same look as GLM: tok/s · hit% · RSS · elapsed)
python coli chat --model /path/to/DeepSeek-V4-Flash --gpu 0 --vram 8 --ram 50

# OpenAI-compatible server (persistent, multi-turn)
python coli serve --model /path/to/DeepSeek-V4-Flash --gpu 0 --vram 8 --ram 50 \
  --host 0.0.0.0 --allowed-host <LAN-IP> --cors-origin <ORIGIN>
```

- `--gpu`/`--vram`/`--ram` map to `COLI_GPUS`/`CUDA_EXPERT_GB`/`RAM_GB`
  (same flags as GLM). Keep `--vram` below your free VRAM.
- Windows: identical commands (run `coli` from `c\`, CUDA `bin\x64` on PATH).
- All other flags/env vars: see the colibri docs (`README`, `SETTINGS.md`,
  `deepseek-v4.md` for the CPU engine, `windows.md` for the GLM GPU walkthrough).

## Validation

- A/B CPU vs CUDA on a real checkpoint: token-identical outputs.
- Tier active: `v4_cuda_tier uploads>0`, expert hit rate climbs run over run
  (`.coli_usage` learning); VRAM returns to idle after exit.
- Fallback: without the DLL, `uploads=0 drops=0` and the engine still runs.