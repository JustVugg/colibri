# Hy3 (Tencent 295B MoE) on colibri

`c/hy3.c` runs [Tencent Hy3](https://huggingface.co/tencent/Hy3) (295B total / 21B active)
with colibri's expert-streaming approach: dense weights resident in RAM (~5 GB),
routed experts streamed from disk with an LRU + pinned cache. Auto-detected from
`config.json` (`model_type: hy_v3`); same `coli chat` / `coli serve` / `coli web` front end.

## Pre-converted weights (Hugging Face)

**https://huggingface.co/UnderstandLing/Hy3-colibri-int4**

~142 GB int4 Colibri container (107 main shards + 6 MTP shards). **Not** GGUF / AWQ / vLLM — only loads in this engine.

```bash
hf download UnderstandLing/Hy3-colibri-int4 --local-dir /path/to/hy3_i4
```

## Quick start

```bash
cd c
./setup.sh          # build hy3 + run hy3_tiny oracle self-test (32/32)

# download or convert model (see below), then:
COLI_MODEL=/path/to/hy3_i4 ./coli chat --ram 12 --gpu 0 --vram 14
```

First reply on a **cold cache** is slow (minutes). Stay in the same session for faster follow-ups. Use `--verbose` to see engine stderr.

## Memory profiles

Hy3 does **not** load 295B parameters into RAM. Only the dense “active” stack (~5 GB) stays resident; MoE experts live on disk and are fetched per token. More RAM/VRAM = larger hot caches = fewer disk reads = faster — not a hard requirement.

### Lean — 16 GB system RAM + ~16 GB VRAM GPU (recommended demo)

Proves the point: a laptop-class machine runs a 295B MoE without 64 GB of system memory.

```bash
cd c && make hy3 CUDA=1
COLI_MODEL=/path/to/hy3_i4 ./coli chat --ram 12 --gpu 0 --vram 14 --auto-tier
```

Typical footprint with this budget:

| Tier | What | Size |
|------|------|------|
| **RAM (always)** | Dense weights (embed, attn, norms, lm_head, 1 dense layer) | **~5 GB** |
| **RAM (runtime)** | KV cache + attention scratch (grows with `--ctx`) | **~2–3 GB** at 4k context |
| **RAM (pin)** | Hot experts in RAM backing store (autopin from `.coli_usage`) | **~1–3 GB** (budget-limited) |
| **VRAM (optional)** | Same hot experts uploaded for GPU matmul | up to `--vram` |
| **Disk** | Full int4 container | **~142 GB** on NVMe |

At startup you should see something like:

```
[CUDA] mode: routed experts only (resident dense on CPU)
[PIN] hot store: N experts in RAM (… GB)
[CUDA] hot expert tier: N experts, VRAM … GB
[RAM_GB=12.0] cap=…/layer (peak ~12–14 GB)
```

Use `coli plan` and `coli doctor` before a long run:

```bash
COLI_MODEL=/path/to/hy3_i4 ./coli plan --ram 12 --gpu 0 --vram 14
COLI_MODEL=/path/to/hy3_i4 ./coli doctor --ram 12 --gpu 0 --vram 14
```

### Comfortable — 32–62 GB system RAM (faster, less disk)

```bash
COLI_MODEL=/path/to/hy3_i4 ./coli chat --ram 56 --gpu 0 --vram 16 --auto-tier --temp 0.7
```

- **`--ram 56`** — large expert LRU (51+ slots/layer vs a handful on lean)
- Autopin can keep **hundreds** of frequent experts in RAM/VRAM (e.g. ~6 GB hot tier)
- Still only ~5 GB dense resident — the extra RAM is **cache**, not “loading the full model”

### CPU-only (no GPU)

```bash
make hy3    # no CUDA
COLI_MODEL=/path/to/hy3_i4 ./coli chat --ram 12
```

Works on 16 GB RAM; experts matmul on CPU. Slower decode, same memory story.

## Tuning knobs

| Flag / env | Effect |
|------------|--------|
| **`--ram N`** | Engine memory budget (GB). Caps LRU slots/layer and autopin size. On a 16 GB machine use **12–14** (leave headroom for the OS). |
| **`--gpu 0`** / **`COLI_GPU`** | Enable CUDA on device 0. Requires `make hy3 CUDA=1`. |
| **`--vram N`** / **`CUDA_EXPERT_GB`** | VRAM budget for hot pinned experts. Try **12–14** on a 16 GB GPU. |
| **`--auto-tier`** | Runs `coli plan` and applies RAM/VRAM/device env vars automatically. |
| **`--ctx N`** | Max context (default 4096). Lower (e.g. 2048) saves KV RAM on tight machines. |
| **`AUTOPIN=0`** | Disable learning-based RAM pin (debug only; usually keep on). |
| **`PIN_GB=N`** | Cap manual/autopin hot store size in GB. |
| **`REPIN=N`** | Live swap of cold VRAM/RAM pins every N emitted tokens. |
| **`CUDA_DENSE=1`** | Upload resident dense tensors to GPU (small win on Hy3: 1 dense layer). |
| **`CUDA_ATTN=1`** | Run GQA attention on GPU during decode (float KV only; incompatible with `KV_I8=1`). |
| **`DIRECT=1`** | `O_DIRECT` disk reads on NVMe. |
| **`PIPE=1`** | Async expert prefetch via thread pool (helps when cache is small). |
| **`PIPE=2`** | io_uring expert reads (needs `make hy3 IOURING=1`; falls back to `PIPE=1`). |
| **`KV_I8=1`** | int8 KV cache (~4× smaller KV RAM; float in-flight row for current position). |
| **`PERF=1`** | Every 100 emitted tokens, print `[perf] attn=…% disk=…% expert_mm=…% head=…%` on stderr. |
| **`TREE_DRAFT=1`** | Tree MTP speculative decode (needs `out-mtp-*.safetensors`; default linear draft otherwise). |
| **`--verbose`** | Show `[CUDA]` / `[PIN]` / `[RAM_GB]` engine lines in the terminal. |

**Trade-offs on 16 GB RAM:** smaller `--ram` → fewer LRU slots and fewer pinned experts → more disk reads per token → slower, but peak RSS stays near **~10–14 GB** instead of 50+ GB. Pair with GPU VRAM so the experts you *do* pin run matmul on the card during **decode** (not during prefill layer progress).

Model path on **native ext4** (WSL: use `/home/...`, not `/mnt/c`).

## Tiny oracle fixtures

These are **not** the full Hy3 model (~142 GB). They are a 5-layer random `hy_v3` checkpoint (~3 MB) used to validate `hy3.c` and `convert_hy3.py` without downloading Tencent weights.

| Path | In git? | Purpose |
|------|---------|---------|
| `c/hy3_tiny/config.json` (+ `generation_config.json`) | **yes** | model config only; weight shards are **not** committed (`*.safetensors` is gitignored) |
| `c/hy3_tiny/*.safetensors` | **no** | ~3 MB fp32 weights — regenerate with `make_hy3_oracle.py` (CI does this automatically) |
| `c/ref_hy3.json` | **yes** | expected prompt/full token IDs and TF logits from the HF reference run |
| `c/hy3_tiny_i4/` | **no** | int4 Colibri container derived from `hy3_tiny`; tests the converter + int4 load path |

`hy3_tiny_i4` is intentionally **not** committed (regenerable in seconds). Add `c/hy3_tiny_i4/` to your local `.gitignore` if you like.

### Regenerate `hy3_tiny` (+ `ref_hy3.json`)

Run from `c/` if the fixture is missing or you need to refresh the oracle:

```bash
cd c
pip install 'transformers>=5.6.0' torch safetensors
python3 tools/make_hy3_oracle.py   # writes hy3_tiny/ and ref_hy3.json
make hy3
SNAP=./hy3_tiny TF=1 ./hy3 64 16 16    # expect 32/32 positions
```

Greedy generate (no teacher forcing):

```bash
SNAP=./hy3_tiny ./hy3 64 16 16         # expect 20/20 new tokens vs ref_hy3.json
```

Use **`64 16 16`** on fp32 `hy3_tiny`. Running `./hy3 64 4 4` quantizes fp32 weights on load and **breaks** the oracle.

### Regenerate `hy3_tiny_i4` (local only)

After `hy3_tiny` exists:

```bash
cd c
./coli convert --repo ./hy3_tiny --model ./hy3_tiny_i4
# or: python3 tools/convert_hy3.py --indir hy3_tiny --outdir hy3_tiny_i4 --ebits 4

SNAP=./hy3_tiny_i4 TF=1 ./hy3 64 4 8    # expect 32/32 positions
```

Use **`64 4 8`** for the int4 tiny fixture (4-bit weights, 8-bit scales).

## Convert Hy3 yourself

From Hugging Face FP8 (resumable, never needs full checkpoint on disk):

```bash
pip install torch safetensors huggingface_hub numpy
./coli convert --repo tencent/Hy3-FP8 --model /path/to/hy3_i4
```

To smoke-test conversion on the tiny fixture before a full Hy3 run, see [Tiny oracle fixtures](#tiny-oracle-fixtures) (`hy3_tiny` → `hy3_tiny_i4`).

## Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| **RAM** | **16 GB** (dense ~5 GB + small cache; lean profile) | **32–62 GB** for large LRU / many pinned experts |
| **VRAM** | none (CPU path) | **8–16 GB** NVIDIA GPU + `make hy3 CUDA=1` |
| **Disk** | ~150 GB model + fast NVMe | ext4, not WSL 9p `/mnt/c` |
| **CPU** | gcc, OpenMP, **AVX2** | more cores help CPU matmul when cache is cold |

Check your machine:

```bash
# lean laptop profile
COLI_MODEL=/path/to/hy3_i4 ./coli plan --ram 12 --gpu 0 --vram 14
COLI_MODEL=/path/to/hy3_i4 ./coli doctor --ram 12 --gpu 0 --vram 14

# comfortable desktop profile
COLI_MODEL=/path/to/hy3_i4 ./coli plan --ram 56 --gpu 0 --vram 16
COLI_MODEL=/path/to/hy3_i4 ./coli doctor --ram 56
```

**What actually sits in RAM:** ~**5 GB** dense weights (the always-active parameters) + KV/runtime (~2–3 GB at 4k ctx) + optional expert cache/pin (the rest of your `--ram` budget). The 295B MoE weights stay on **disk** unless cached.

## Performance (honest)

Hy3 cold decode reads **~5–6 GB of experts per token** from disk (79 MoE layers × 8 experts). Expect:

| Phase | Lean (16 GB RAM + GPU) | Comfortable (56 GB RAM + GPU) |
|-------|------------------------|-------------------------------|
| First prompt (cold) | ~0.05–0.15 tok/s | ~0.05–0.15 tok/s |
| Warm cache | ~0.1–0.3 tok/s | ~0.2–0.5 tok/s |
| Same chat session, turn 2+ | faster as `.coli_usage` + LRU warm up | often several× faster |

Tuning: `--ram` sizes the cache budget; `--vram` + `--gpu` move hot experts to the card; autopin/PIN from `.coli_usage`; `DIRECT=1` on NVMe; `PIPE=1` or `PIPE=2` (io_uring); optional `CUDA_ATTN=1`, `KV_I8=1`, `PERF=1`.

**Throughput numbers:** mid-stream `[t=N] … tok/s` and the chat footer’s **decode** rate measure generation only (from first emitted token). The footer’s **total** rate includes prefill for that turn (full prompt on turn 1, incremental new tokens on follow-ups). Example: `└─ 38 tok · 0.10 tok/s total · 0.24 tok/s decode · …`.

### ROCm / AMD GPU (optional)

ROCm builds use the HIP runtime and keep the existing accelerator ABI and
`--gpu` / `--vram` interface. On Linux/WSL, keep the model on a native
ext4/NVMe path. On Windows, use the AMD HIP SDK and make sure `hipcc` is on
`PATH` (or set `ROCM_HOME`/`HIP_PATH`). `gfx1101` is the default target for an
RX 7800 XT; override `ROCM_ARCH` for another AMD GPU.

    cd c
    make hy3 ROCM=1 ROCM_ARCH=gfx1101
    make rocm-test ROCM=1 ROCM_ARCH=gfx1101

    COLI_MODEL=/home/me/hy3_i4 ./coli chat --ram 12 --gpu 0 --vram 14 --auto-tier --verbose

Windows HIP SDK example (PowerShell; native builds also need GNU make and
MinGW gcc for the C host objects):

    $env:HIP_PATH = 'C:\Program Files\AMD\ROCm\7.1'
    $env:ROCM_PATH = $env:HIP_PATH
    $env:ROCM_HOME = $env:HIP_PATH
    make hy3 ROCM=1 ROCM_ARCH=gfx1101
    make rocm-test ROCM=1 ROCM_ARCH=gfx1101

The ROCm path disables the NVIDIA-only WMMA kernel and uses the generic packed
int4 path, preserving correctness while leaving room for an AMD-specific
MFMA/rocWMMA optimization later. `coli doctor` and the resource planner also
recognize `libamdhip64`; Linux GPU discovery uses `rocm-smi`, while the
Windows HIP SDK fallback uses `hipInfo`.

For the 295B `hy_v3` GGUF distribution, keep the GGUF under the LM Studio model
directory, but use a current upstream `llama.cpp` ROCm build for inference.
The bundled LM Studio AMD runtime may index `hy_v3` while still reporting
`unknown model architecture: 'hy_v3'` at load time. `c/rocm_tune.py` reads the
GGUF tensor table and calculates whole-layer offload after reserving the KV
cache, compute buffers, and a safety margin. This avoids the frequent
host/device transfers produced by fractional MoE-layer fitting:

    python c/rocm_tune.py `
      --model "$env:USERPROFILE\.lmstudio\models\AngelSlim\Hy3-GGUF\Hy3-IQ1_M.gguf" `
      --vram-gib 15.98 --context 2048 --target-mib 512 --compute-mib 256 `
      --cache-type-k q8_0 --cache-type-v q8_0

The printed command uses the recommended integer `-ngl`, `--fit off`, and
q8_0 K/V cache. The output separates the total KV cache from its VRAM and RAM
parts. The same plan is available as `coli rocm-plan --model ...`.
The standalone tool's `--vram-gib` is binary GiB; the existing `coli --vram`
flag is decimal GB, so pass the value appropriate to the tool you use.
The speed-oriented default reserves 512 MiB of free VRAM; use
`--target-mib 1024` when a larger desktop/allocator margin is preferred.
The upstream `llama.cpp --fit on` mode remains useful as a safe fallback, but
it may select partial MoE layers for maximum occupancy rather than maximum
decode speed. KV offload is enabled by default in llama.cpp; q8_0 K/V is a
practical RX 7800 XT starting point at 2K context (Hy3's measured cache is
about 340 MiB for one sequence).

For this RX 7800 XT / 96 GB RAM machine, the context trade-off is:

| Context | KV type | Recommended `-ngl` | KV total | Note |
|---:|---|---:|---:|---|
| 2K | q8_0 | 14 | 340 MiB | fastest measured: 2.21 decode tok/s |
| 64K | q8_0 | 12 | 10.6 GiB | tested successfully: 1.83 decode tok/s |
| 128K | q8_0 | 11 | 21.25 GiB | tested successfully, but 1.23 decode tok/s |
| 128K | q4_0 | 12 | 11.25 GiB | lower KV precision; more VRAM/RAM headroom |

For 64K or 128K, pass the desired context to `rocm_tune.py`; do not reuse the
2K `-ngl 14` setting. A large context can fit in capacity while making prefill
very slow, especially when q8_0 KV keeps most of the cache in host RAM.

### CUDA expert tier (optional)

Same three-tier model as GLM: **disk → RAM → VRAM**. Hot pinned experts can run GPU matmul during **token generation** (not during `[prefill] layer …` progress, which is mostly CPU attention + disk I/O).

```bash
cd c
make hy3 CUDA=1                    # requires nvcc + NVIDIA driver
make hy3 CUDA=1 IOURING=1          # + io_uring for PIPE=2 expert loads

# lean: 16 GB system RAM + 16 GB GPU
COLI_MODEL=/path/to/hy3_i4 ./coli chat --ram 12 --gpu 0 --vram 14 --auto-tier

# fast: large RAM cache + GPU + optional perf knobs
COLI_VERBOSE=1 DIRECT=1 PIPE=2 CUDA_ATTN=1 \
  COLI_MODEL=/path/to/hy3_i4 ./coli chat --ram 56 --gpu 0 --vram 14 --auto-tier --verbose
```

Or set env vars directly: `COLI_CUDA=1`, `COLI_GPU=0`, `CUDA_EXPERT_GB=14`, plus autopin from `.coli_usage`.
At startup, confirm you see `[CUDA] hot expert tier: N experts, VRAM … GB` — if that line is missing, CUDA is not active (rebuild with `CUDA=1` and pass `--gpu` / `--vram`).
Optional `CUDA_DENSE=1` uploads resident dense weights to GPU (uses VRAM; leaves less room for expert tier). `CUDA_ATTN=1` offloads GQA attention to GPU (best with float KV; do not combine with `KV_I8=1`). `REPIN=N` swaps cold VRAM pins for hot experts during chat.

## OpenAI-compatible API

```bash
COLI_MODEL=/path/to/hy3_i4 ./coli serve --ram 12 --gpu 0 --vram 14 --host 127.0.0.1 --port 8000
```

Model id: `hy3-colibri`.

## What we contribute (vs upstream Hy3)

| Piece | Description |
|-------|-------------|
| **Colibri int4 format** | `out-*.safetensors` + per-row `.qs` scales — incompatible with GGUF/AWQ |
| **`convert_hy3.py`** | Hy3-FP8 per-tensor `weight_scale` dequant, `shared_mlp` → `shared_experts` naming |
| **`hy3.c`** | GQA, sigmoid+bias router, route norm, batch-union MoE, LRU/pin cache |
| **Chat template** | Hunyuan `｜` delimiters, `coli chat` + `coli serve` |

Base model: [tencent/Hy3](https://huggingface.co/tencent/Hy3) / [Hy3-FP8](https://huggingface.co/tencent/Hy3-FP8) (Apache 2.0).  
Same idea as [GLM-5.2-colibri-int4](https://huggingface.co/jlnsrk/GLM-5.2-colibri-int4), applied to Hy3.

## Architecture (Hy3 vs GLM in colibrì)

| | GLM-5.2 | Hy3 |
|--|---------|-----|
| Attention | MLA (compressed KV) | **GQA** (64Q / 8KV) |
| Experts | 256 × 75 layers | **192 × 79 layers** |
| Expert size (int4) | ~19 MB | **~9 MB** |
| Dense prefix | 3 layers | **1 layer** |
| Disk container | ~370 GB | **~142 GB** |

## Status & roadmap

**Working today**

- FP8 → int4 conversion (full 80-layer model)
- `hy3_tiny` oracle **32/32** (weights regenerated in CI via `make_hy3_oracle.py`); `hy3_tiny_i4` oracle **32/32** (regenerate locally)
- Full-model chat (coherent output with `IDOT=0`, fixed serve KV alloc)
- MTP speculative decode (auto-enabled when `out-mtp-*.safetensors` present; `DRAFT=3` default; optional `TREE_DRAFT=1`)
- Performance knobs: AVX2 attention, `KV_I8=1` int8 KV, `PERF=1` breakdown, `PIPE=2` io_uring loads, `CUDA_ATTN=1` GPU attention
- `coli chat` / `coli serve` / `coli convert` / `coli plan` / `coli doctor`

**Not yet**

- KV disk persistence (GQA layout)
- int8 IDOT fast path (disabled by default until oracle-clean on all shapes)

## Build

```bash
cd c && make hy3
# optional CUDA build (Linux + nvcc):
make hy3 CUDA=1
# optional io_uring expert loads (PIPE=2):
make hy3 IOURING=1
# both:
make hy3 CUDA=1 IOURING=1
# optional self-test (requires hy3_tiny/ — see Tiny oracle fixtures):
SNAP=./hy3_tiny TF=1 ./hy3 64 16 16
# MTP smoke test (full converted weights with out-mtp-*.safetensors):
# SNAP=/path/to/hy3-int4 TEMP=0 DRAFT=3 PROMPT=1 ./hy3 64 8 4
# Expect stderr: [MTP] active: native speculative decoding (draft=3)
# and ~2+ tokens/forward in the speculation stats line
```

## Provenance & license

- Engine fork: colibrì ([MIT](https://github.com/JustVugg/colibri))
- Hy3 weights: [Tencent Hy3](https://huggingface.co/tencent/Hy3) (Apache 2.0)
- This port's converted weights: [UnderstandLing/Hy3-colibri-int4](https://huggingface.co/UnderstandLing/Hy3-colibri-int4) (Apache 2.0 derivative)

## Citation

Credit [Tencent Hy3](https://huggingface.co/tencent/Hy3) for the base model and [colibrì](https://github.com/JustVugg/colibri) for the streaming engine.
