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
- Dense layers (43, ~6.27 GiB fp8) and attention stay on **CPU/RAM** by
  default, like the target-only build. With `COLI_DSV4_DENSE_CUDA=1` the five
  dense matmuls per layer (wq_a/wq_b/wkv/wo_a/wo_b, native fp8 weights) move
  to the GPU with identical numerics — measured **+18% decode** on the
  reference hardware (see the dense-on-GPU section below). Only routed
  experts are GPU-eligible by default.
- Numerics are identical to the CPU path (verified A/B on real checkpoints).
- Missing `coli_dsv4_cuda.dll` (Windows) → silent CPU fallback, no error.
- **Tool use / function calling**: OpenAI `tools` are supported with the
  checkpoint's native DSML format (`｜DSML｜tool_calls` blocks, `<tool_result>`
  merged into user turns) — including streaming, multi-turn tool results,
  `tool_choice` none/auto/required/forced, and the Anthropic `/v1/messages`
  translation. Python-only (`openai_server.py`), engine untouched.
- **Cross-session KV persistence (`.coli_kv`)**: the conversation's attention
  state (window + compressed + recurrent compressor/indexer state) is
  snapshotted to `<model_dir>/.coli_kv` after every turn (temp+rename, atomic)
  and resumed at the next serve start — the first request skips re-prefilling
  the history (`[KV] resumed conversation from disk: N tokens`). `KVSAVE=0`
  disables saving/resume; delete the file to start clean.

**Not wired up (yet):** DSpark/MTP speculation (`V4_MTP=0` default),
non-greedy sampling and more than one KV slot in serve mode.

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
- cuBLASLt < 12.8 (e.g. Jetson JetPack with CUDA 12.6) lacks the MXFP8
  block-scaling APIs used by the optional `DSV4_CUDA_TC` tensor-core path
  (default off). Build with `NVCCFLAGS=" -DCOLI_DSV4_NO_TC"` to compile it
  out; the custom FP4 expert path always builds, unchanged:

```bash
make -f Makefile.deepseek-v4 deepseek-v4 CUDA=1 CUDA_ARCH=sm_87 \
  NVCCFLAGS="-O3 -std=c++17 -arch=sm_87 -DCOLI_DSV4_NO_TC" -j$(nproc)
```

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
- Runtime alternative (no PATH change): copy ALL THREE runtime DLLs next to the
  binary — `cublasLt64_13.dll` (the DLL's direct dependency), `cublas64_13.dll`
  and `cudart64_13.dll`. A partial copy (e.g. cudart alone) still falls back to
  CPU silently; only the complete set works. Verified 2026-08-15 (clean run,
  PATH without CUDA).
- Expected: `coli_dsv4_cuda.dll` with exactly the 5 engine-facing exports
  (`dsv4_cuda_init`, `dsv4_cuda_shutdown`, `dsv4_cuda_tensor_free`,
  `dsv4_cuda_upload_fp4`, `dsv4_cuda_expert_group`); `deepseek_v4.exe` runs
  even without the DLL (CPU fallback).
- Note: the top-level `Makefile` target `deepseek-v4` does **not** propagate
  `CUDA=1`; always build through `Makefile.deepseek-v4`.

## Use

All commands go through the `c/coli` dispatcher (the same one GLM uses); the
raw engine is one-shot or serve-protocol only. Run `python coli --help` from
the repo `c/` dir for the full flag list.

### Baseline — CPU only (no GPU required)

```bash
python coli chat --model /path/to/DeepSeek-V4-Flash --gpu none --ram <N>
```

- `--gpu none` maps to `COLI_DSV4_CUDA=0`: pure CPU, works on any machine
  (no CUDA build needed). The engine streams cold experts from disk.
- `--ram N` is the RAM budget for the pinned expert cache (the single most
  impactful knob). **Rule of thumb: `N ≈ total RAM − 12 GB`** (12 GB covers
  the OS, the resident dense layers ~6.3 GiB and the head). Examples: 16 GB
  RAM → `--ram 4` (slow, disk-bound, but it runs); 64 GB → `--ram 52`;
  128 GB → `--ram 116`.

### Baseline — CPU + CUDA (NVIDIA GPU with the CUDA build)

```bash
python coli chat --model /path/to/DeepSeek-V4-Flash --gpu 0 --vram <V> --ram <N>
```

```bash
# OpenAI-compatible server (persistent, multi-turn, KV-resume)
python coli serve --model /path/to/DeepSeek-V4-Flash --gpu 0 --vram <V> --ram <N> \
  --host 0.0.0.0 --allowed-host <LAN-IP> --cors-origin <ORIGIN>
```

- `--gpu 0` / `--vram V` / `--ram N` map to `COLI_GPUS` / `CUDA_EXPERT_GB` /
  `RAM_GB` (same flags as GLM). Without the CUDA DLL/link the engine silently
  falls back to CPU.
- `--vram V` is the VRAM budget for the expert tier. **Rule of thumb:
  `V ≈ 50% of your GPU's VRAM`, rounded down** — the rest is needed for the KV
  context, temporary allocations and the desktop. Never fill the card: the
  speedup is not proportional to the budget (see measured results below).
- Windows: identical commands (run `coli` from `c\`, CUDA `bin\x64` on PATH).
- First run on a fresh model dir creates `.coli_usage` (usage history used by
  the pinning); each run appends to it, and `.coli_kv` persists the
  conversation across restarts.

## Performance tuning

The knobs below are ordered by measured impact (see the reference-hardware
results section). Defaults are sane; change one knob at a time and re-measure
on your own hardware.

| Flag / env | What it does | How to set it |
|---|---|---|
| `--ram N` / `RAM_GB` | RAM budget for pinned experts → hit rate → disk read volume | `N ≈ total RAM − 12 GB`. The #1 lever. |
| `COLI_V4_PREWARM=1` | Pin the usage-history experts **before** READY instead of interleaved with the first prefill | Slightly faster decode + hit; the time-to-first-token is unchanged (the pin cost moves into the startup). Harmless to enable. |
| `--vram V` / `CUDA_EXPERT_GB` | VRAM budget for the hottest experts | `V ≈ 50%` of VRAM, never above ~75% (KV/context margin). Above the safe point the gain is small and OOM risk grows. |
| `COLI_DSV4_DENSE_CUDA=1` | Move the dense layers (5 matmuls/layer, ~6.27 GiB fp8) to the GPU; the expert tier stays unchanged | Measured **+18% decode** and −1.6 CPU cores, at the cost of ~9.6 GB VRAM (dense + experts) and ~+10 s TTFT (H2D preload at startup). **Default off**: enable for long sessions, keep off for short one-shots. |
| `COLI_V4_DENSE_DEBUG=1` | Log every dense-GPU call (`v4_dense_dbg ... after ok=1/0`) to count CPU fallbacks | Diagnostic only, leave off. With the tier active expect `ok=1` on every call and 0 fallbacks. |
| `--gpu none` / `COLI_DSV4_CUDA=0` | Disable the CUDA tier (pure CPU) | CPU-only machines. |
| `--gpu N` / `COLI_GPUS` | Select the CUDA device(s) | `0`, or `0,1` for multi-GPU. |
| `--ctx N` / `CTX` | Context window (default 4096) | Raise only if you need long context; the KV lives in RAM. |
| `OMP_NUM_THREADS` | OpenMP threads (default = physical cores) | **Leave it**: extra threads (SMT oversubscription) do not help this engine. |
| `KVSAVE` | `.coli_kv` cross-session persistence (default on) | `KVSAVE=0` disables save+resume. |
| `COLI_V4_AUTOPIN` / `COLI_V4_SAVE_USAGE` | Usage-history learning + saving (default on) | Leave it: the pinning learns your usage pattern run over run. |
| `COLI_V4_DIRECT` | Expert reads O_DIRECT (default: autodetected) | **Do not set `0`**: buffered reads measured *slower* (page-cache pressure with a large pin set). |
| `V4_MTP` / `V4_DRAFT` / `V4_NGRAM` / `COLI_V4_MARKOV_*` | Speculative decoding (default off) | Experimental; measured net-negative on this engine. Not recommended. |

## Reference measurements (2026-08, dev hardware)

Benchmarked with the raw serve protocol (same prompt, 167 tokens, 200
generated, greedy, `OMP_NUM_THREADS=12`, `KVSAVE=0`; each row = mean of 2-4
runs, σ ≈ 0.01 tok/s). These are **reference numbers for one machine**, not
guarantees — they show the *direction and rough magnitude* of each knob.
Benchmarked code: `feat/ds4-cuda-tier` @ `0d17c77` (pre-sync with
`upstream/dev`; post-sync smoke runs measure 1.1-1.2 tok/s on the same
config, in line with these numbers).

Hardware: Ryzen 9 5900X (12c/24t), 64 GB RAM, RTX 3080 10.7 GB (sm_86),
NVMe Kingston KC3000 (~2-3 GB/s random read), Windows 11 native build.

| Config | decode tok/s | TTFT (167-tok prompt) | hit | disk read volume |
|---|---|---|---|---|
| `--ram 40 --vram 4` (defaults) | 0.91 | 147 s | 77% | 272 GB / 200 tok |
| `--ram 52 --vram 4` | 0.97 (+7%) | 139 s | 83% | 208 GB (−23%) |
| `--ram 52 --vram 4` + `COLI_V4_PREWARM=1` | **1.01 (+11%)** | 134 s* | 84% | 208 GB |
| `--vram 8.6` (on top of ram 40) | 0.96 | 136 s | 77% | 286 GB |
| buffered (`COLI_V4_DIRECT=0`) | 0.82 (−10%) | 156 s | 77% | n/a (page cache) |
| CPU-only (`--gpu none`) | 0.87 | 149 s | 78% | 267 GB |
| GLM-5.2 788B IQ3 (same harness) | 0.41 | 234 s | 47% | ~0 (resident RAM) |

\* time-to-first-token unchanged: pre-READY grows by ~5 s, TTFT shrinks by ~5 s.

### Dense-on-GPU tier (2026-08-16, same hardware)

A/B of `COLI_DSV4_DENSE_CUDA` with the same raw-serve harness, same prompt
(124 tokens — different from the 167-token table above, so compare rows
within this section only), `--ram 52 --vram 4`, `OMP_NUM_THREADS=12`,
`KVSAVE=0`; `.coli_usage` restored to the identical backup before each run.

| Config | decode tok/s | TTFT | hit | CPU cores (decode) | GPU util (decode) | disk (decode) | VRAM peak |
|---|---|---|---|---|---|---|---|
| dense CPU (`=0`) | 0.955 | 95 s | 72% | 5.54 | 5.6% | 622 MB/s | 5.3 GB |
| **dense GPU (`=1`)** | **1.126 (+18%)** | 105 s | 73% | **3.97** | **10.0%** | 681 MB/s | **9.7 GB** |

- The dense matmuls were previously estimated "not the bottleneck" (~15 ms /
  token); the A/B shows they cost **~1.6 CPU cores** in decode and moving them
  to the GPU is worth **+18% tok/s** — a real lever, not a rounding error.
- Verified integrity: 42,785 dense-GPU calls `ok=1`, **0 CPU fallbacks** over
  the 200-token run (43 layers × 5 matmuls × ~199 tokens); numerics identical
  to the CPU path (token-identical outputs).
- Cost: ~9.6 GB VRAM (6.27 GiB dense + expert tier) on a 10.7 GB card — check
  your KV margin — and ~+10 s TTFT (H2D preload of the 43 layers at startup).
  Amortized on long sessions; a short one-shot pays it back only partially.
- GLM comparison, same harness: GLM reads **~7× more disk per token** than DS4
  (4.6 vs 0.65 GB/token) because it routes 2.4× more (78 layers × 8 top-k vs
  43 × 6) with a ~44% hit rate (its RAM tier holds only ~692 of ~20k experts),
  yet its tok/s is unchanged by I/O — GLM's bottleneck is per-token pipeline
  latency, not disk (it measured 0.38-0.42 tok/s with both ~0 and ~1.7 GB/s of
  disk I/O). `CUDA_EXPERT_GB` 3 vs 5 does not change the GLM VRAM tier (0/692
  experts both ways: the dense+KV already fill the card).

Takeaways:

- **RAM is the lever**: 40→52 GB = +7% decode, −23% disk volume. The engine is
  disk-bound (≈1 GB of expert reads per generated token); every GB of pinned
  RAM buys hit rate.
- **VRAM is not the bottleneck**: 4→8.6 GB VRAM budget = only +6% (and 97%
  VRAM usage, 2× upload/drop churn, OOM risk). The expert matmul is not the
  cost; the disk transfer is.
- **GPU vs CPU-only**: +4-5% for the expert CUDA tier alone (the wall is the
  disk). Moving the **dense** layers to the GPU as well (`COLI_DSV4_DENSE_CUDA=1`)
  adds another +18% on top — the dense matmuls were a hidden CPU cost.
- **Buffered reads are slower** on this hardware: with 40-52 GB pinned, the
  page cache has ~10 GB left for a 92 GB prefill → thrash + extra copy.
  O_DIRECT stays the default for a reason.
- The path to 2-3 tok/s on this class of hardware is reducing the *disk read
  volume* (mmap-style page-cache reuse of expert weights, or smarter
  prefetch), not more VRAM or more threads.

## Validation

- A/B CPU vs CUDA on a real checkpoint: token-identical outputs.
- Tier active: `v4_cuda_tier uploads>0`, expert hit rate climbs run over run
  (`.coli_usage` learning); VRAM returns to idle after exit.
- Fallback: without the DLL, `uploads=0 drops=0` and the engine still runs.
