# DeepSeek V4 target engine (CPU + optional CUDA tier)

[简体中文](deepseek-v4.zh-CN.md) (older; this English page is the reference)

DeepSeek V4 Flash runs from the official sharded safetensors checkpoint with
no conversion: routed experts stay native fp4, the dense set stays fp8-e4m3
with UE8M0 block scales, MLA + DSA sparse attention, 43 layers, 256 routed
experts + 1 shared, top-6. The engine is CPU-first and CPU-canonical: every
piece of state lives in host memory, and an optional CUDA tier accelerates
prefill and decode stage by stage, falling back to the CPU reference for any
stage it declines. Every GPU stage was accepted only when it reproduced the
engine's CPU reference on greedy text; what "identical" can and cannot mean
across kernels and cache states is spelled out in [Validation](#validation).

## Scope

- Production code is in `c/deepseek_v4.c` (amalgamated units); the public
  engine/session API is `c/deepseek_v4.h`. The CUDA tier is
  `c/backend_cuda_dsv4.cu` (+ `backend_loader_dsv4.c` on Windows).
- Official sharded safetensors checkpoints load through shared `st.h`;
  standard MXFP4 matmul uses shared `quant.h`.
- Unified `c/coli` routes `run`, `chat`, `serve`, and `web` to V4; `serve` is
  the OpenAI/Anthropic-compatible gateway (`c/openai_server.py`): it renders
  OpenAI and Anthropic tools into V4's native prompt contract and parses DSML
  call blocks back into each protocol (see the
  [per-engine API matrix](api.md#tool-calling-support)); grammar remains
  unsupported.
- Serving is greedy, one KV slot; speculative drafting exists and is off by
  default (`V4_DRAFT`, `V4_MTP`).
- Build targets: x86-64/aarch64 Linux, Windows/MSYS2, arm64 macOS (CPU);
  CUDA tier on Windows (runtime DLL) and Linux (`CUDA=1`, direct link;
  verified under WSL2).

Destroy every session before destroying its engine.

## Memory policy

A checkpoint has 43 transformer layers, hidden size 4096, and 256 routed
experts per sparse layer with top-k 6. Dense weights occupy about 6.27 GiB and
a resident BF16 output head about 1.06 GiB. Routed-expert weights (~137 GiB
on disk, ~12.6 MB each) are streamed and cached according to the RAM budget:
the planner reserves workspace and a minimum expert working set, then enables
dense/head residency and grows the expert cache when memory permits.
`--ram GiB` is a planner budget, not an OS-enforced limit; without it the
budget is derived from available OS memory. `COLI_MODEL_MIRROR=<dir>` (a
second copy of the checkpoint on another SSD) splits expert reads across two
drives — measured 2× read bandwidth; on this workload the two NVMe are the
limit for both prefill refill and decode (see Performance).

## Download

```bash
hf download deepseek-ai/DeepSeek-V4-Flash-0731 \
  --local-dir /path/to/DeepSeek-V4-Flash
```

A download can finish with a truncated shard even when the client reports
success. If `st.h` rejects a shard as out of bounds, compare every local shard
size with the Hugging Face repository before treating it as an engine failure.

## Build

CPU engine (all platforms):

```bash
cd c
make deepseek-v4            # ARCH=native for the local CPU (default x86-64-v3)
```

### Windows CUDA tier

The engine is MinGW-built (MSYS2 **UCRT64** shell: `MSYSTEM=UCRT64`; the plain
MSYS shell picks the cygwin gcc and fails on `getrusage`), while the CUDA
kernels must be built by nvcc + MSVC into a DLL that the engine loads at
runtime. Two flavours of the same source ship side by side and the loader
picks at start-up by GPU:

| DLL | build target | kernels | GPUs |
|---|---|---|---|
| `coli_cuda_dsv4_dg.dll` | `make cuda-dsv4-dg-dll` (from a vcvars64 shell; fetches the pinned DeepGEMM headers on first use) | DeepGEMM tensor-core prefill paths + generic kernels | compute capability 12.x (RTX 50-series) |
| `coli_cuda_dsv4.dll` | `make cuda-dsv4-dll CUDA_ARCH=portable` | generic CUDA kernels only (fp32 compute from fp8/fp4 decode) | any sm_80+ (RTX 30/40/50, A/H series) |

The DeepGEMM sm120 headers the DLL needs are not in the repository:
`c/tools/fetch_deepgemm.sh` checks out the community sm120 port of DeepGEMM
(MIT; upstream DeepGEMM has no sm120 kernels) plus its CUTLASS/CuTe submodule
(BSD-3) at a pinned commit into the gitignored `c/third_party/deepgemm/`,
verifies both commit ids, and applies `c/patches-deepgemm-sm120-msvc.patch`
(MSVC ignores `alignas(64)` on by-value TMA descriptors — the patch pads
them). `make cuda-dsv4-dg-dll` and `DEEPGEMM=1` run it on first build
(`make deepgemm-fetch` runs it alone; ~40 MB, git required);
`DEEPGEMM_HOME=<dir>` points at an existing tree instead and `DEEPGEMM_PIN`
bumps the commit. Attribution: `THIRD_PARTY_NOTICES.md`. Loader rules:
try `_dg` first, ask it whether device 0 fits (`dsv4_cuda_backend_arch_ok`),
otherwise load the generic DLL; `COLI_DSV4_DLL=<file name>` forces one. The
start-up banner names the choice: `[DSV4 CUDA] backend=coli_cuda_dsv4_dg.dll
(deepgemm-sm120)`. Deploy = `deepseek_v4.exe` + both DLLs next to it.

### Linux CUDA tier

```bash
cd c
make -f Makefile.deepseek-v4 deepseek-v4 CUDA=1 CUDA_ARCH=portable   # generic kernels, any sm_80+
make -f Makefile.deepseek-v4 deepseek-v4 CUDA=1 DEEPGEMM=1           # DeepGEMM flavour, sm_120a only (fetches the pinned headers)
make -f Makefile.deepseek-v4 deepseek-v4 CUDA=1 CUDA_ARCH=sm_86 NO_TC=1   # toolkit older than 12.8
```

`CUDA_HOME` defaults to `/usr/local/cuda` (`CUDA_HOME=/usr/local/cuda-13.3`
to pick a versioned install). `CUDA_ARCH` is `native` (the local card),
`portable` (sm_80/86/89/90/120 + PTX) or one `sm_XX`; `DEEPGEMM=1` ignores it
and builds sm_120a only — the DeepGEMM kernels use block-scaled MMA PTX that
exists for no other target. `NO_TC=1` (`-DCOLI_DSV4_NO_TC`) compiles out the
opt-in cuBLASLt MXFP8 path (`DSV4_CUDA_TC=1`), whose block-scaling API needs
cuBLASLt ≥ 12.8; use it for CUDA 12.6 toolkits (Jetson JetPack). `libcuda`
is linked from the toolkit's `lib64/stubs` (driver installs and WSL2 ship
only `libcuda.so.1`), and the binary carries an rpath to `$(CUDA_HOME)/lib64`.

Verified on Ubuntu 22.04 under WSL2 (gcc 11, CUDA 13.3.1 + 12.6.85, RTX
5080): the CPU-only build (6 s), the generic fat binary, the DeepGEMM
binary and the 12.6 `NO_TC=1` object all compile; without the gate the 12.6
compile fails on exactly the four cuBLASLt 12.8 symbols. At run time both
CUDA binaries bring the tier up (`[DSV4 CUDA] device 0: ... sm_120`,
`v4_gpu tier=dense-matvec`) and generate; see [Validation](#validation) for
the WSL2 numbers.

There is no loader on Linux: one flavour is linked directly. A DeepGEMM
binary on a non-sm120 card logs `v4_gpu warning=backend ... does not support
device` and runs CPU-only. Host-slab pinning for expert uploads is
Windows-only for now (`VirtualQuery`); Linux uses pageable copies (correct,
slower refill).

## Run

Canonical serve line (Windows, PowerShell; the environment used for every
number below):

```powershell
$env:COLI_MODEL_MIRROR = "D:\models\DeepSeek-V4-Flash"   # optional second copy
$env:COLI_CUDA = "1"; $env:COLI_GPU = "0"; $env:CUDA_DENSE = "1"; $env:COLI_CUDA_PIPE = "2"
$env:COLI_CUDA_ATTN_BATCH = "1"; $env:COLI_CUDA_MOE_BATCH = "1"
$env:V4_MOE_REFILL_GROUP = "12"; $env:DSV4_CUDA_EXPERT_MIRRORS = "688"
$env:V4_LOADER_LANES = "3"   # GPU tier: 9-lane default tuned for the CPU path, see the env table
python ./coli serve --model C:\models\DeepSeek-V4-Flash --ram 32 --ctx 20000
```

`coli run|chat|web` take the same environment. `--ngen` is a ceiling, not a
target (answers end at EOS; an oversized ceiling is clamped to the context
with a stderr note). `CTX`/`--ctx` sets the context window.

What the knobs do (full table in [Environment reference](#environment-reference)):
`COLI_CUDA_ATTN_BATCH=1` puts the batched prefill attention block on the GPU
(compressor/indexer projections, sparse attention on a persistent device KV
ring, wo, mHC) and enables the GPU decode attention/indexer paths;
`COLI_CUDA_MOE_BATCH=1` runs prefill MoE on a transient VRAM expert bank
(route-aware refill, expert-grouped GEMM); `DSV4_CUDA_EXPERT_MIRRORS` is the
decode-side VRAM expert cache; `V4_MOE_REFILL_GROUP` the refill lookup width.
All default to CPU-safe values when unset; the GPU tier itself is on when the
DLL loads (`DSV4_CUDA=0` disables it).

## VRAM budget and per-card settings

What the tier keeps in VRAM (measured on a 16 GB RTX 5080): dense/attention
mirrors **6.3 GB** (needed for GPU attention and the GPU dense matvecs in
decode), the transient prefill expert bank **2.2 GB** (`COLI_CUDA_MOE_BATCH`,
freed when decode starts), attention/KV/work buffers **~0.5 GB**, and the
rest as decode expert mirrors at **~8 MB each**. `DSV4_CUDA_EXPERT_MIRRORS`
is only an upper bound (default 4096): the cache grows while at least
`DSV4_CUDA_VRAM_RESERVE_MB` (default 2800 with the bank enabled, 600
without) stays free, so free VRAM sizes it at run time — you normally do not
need to set it. Decode is disk-bound, so more mirrors help in proportion to
the hit rate they add (~11 000 experts total; 600 mirrors ≈ 5 %); more RAM
(`--ram`) buys more than more VRAM.

| VRAM | dense mirrors | prefill bank | recommended env | expected |
|---|---|---|---|---|
| 6 GB | no (6.3 GB does not fit) | yes (2.2 GB) | `CUDA_DENSE=0 COLI_CUDA_MOE_BATCH=1 COLI_CUDA_ATTN_BATCH=0` | GPU MoE prefill (generic kernels on non-RTX50), CPU attention/decode; ~150 mirrors between prefills |
| 8 GB | yes | no (1.7 GB left) | `CUDA_DENSE=1 COLI_CUDA_ATTN_BATCH=1 COLI_CUDA_MOE_BATCH=0` | GPU attention + GPU decode dense; MoE prefill on the CPU union; ~100 mirrors |
| 10–12 GB | yes | yes (tight) | `CUDA_DENSE=1 COLI_CUDA_ATTN_BATCH=1 COLI_CUDA_MOE_BATCH=1 DSV4_CUDA_VRAM_RESERVE_MB=2500` | full GPU prefill; 100–350 decode mirrors after the bank is released |
| 16 GB (this box) | yes | yes | the canonical line above (mirror cap can be left unset) | ~600 mirrors, the numbers in this document |
| 24 GB | yes | yes | canonical | ~1 700 mirrors (~15 %); decode +10–15 % vs 16 GB |
| 32 GB | yes | yes | canonical | ~2 700 mirrors (~25 %); decode +25–35 %; raise `--ram` first if RAM allows |

On non-Blackwell cards the generic DLL is selected automatically (fp32
kernels): same settings, prefill roughly 2–3× slower than the DeepGEMM
numbers, still far ahead of CPU.

## Prefill: segments, chunks, checkpoints

- **Chunks/segments.** Prefill runs layer-major over 128-token chunks
  (`V4_PREFILL_CHUNK`) inside 4096-token segments (`V4_PREFILL_SEGMENT`).
  Segments are atomic; the client-cancel poll runs between them and completed
  segments are recorded, so an identical retry resumes instead of restarting.
  Each segment re-sweeps every layer's routed experts through the transient
  bank, so fewer/larger segments = fewer disk sweeps, at the price of cancel
  latency (~2.5 min worst case at 4096). Progress lines `v4_prefill N/M tokens`
  print per segment on multi-segment prompts.
- **Prefix checkpoints** (`V4_PREFIX_CKPT`, min length `V4_PREFIX_CKPT_MIN`=512).
  The window/compressor/indexer state cannot rewind, so reuse across
  conversations needs a snapshot at the shared boundary. Three capture rules:
  (1) the gateway tells the engine where the rendered system turn ends
  (optional 8th `SUBMIT` header field) → snapshot there on the very first
  request; (2) fallback: the longest common prefix of two successive fresh
  prompts; (3) **prompt-end** snapshot after every prefill, because agent
  clients re-render the assistant reply, so strict "extends everything fed"
  reuse fails at the reply boundary. `V4_PREFIX_CKPT_SLOTS` (4) in-memory
  slots, LRU with prompt-end evicted first. Log lines: `v4_ckpt store
  prefix=N` / `prompt_end=N`, `v4_ckpt hit prefix=N`.
- **Persistence.** Prefix captures are written to `<model>/.coli_ckpt/`
  (`V4_PREFIX_CKPT_DISK`=1; `2` also persists prompt-end captures, `0` off;
  ~140 MB for an 8.3k-token prefix, config-fingerprinted, temp+rename) and
  loaded lazily on the first request after a restart — a fresh serve no
  longer re-prefills the system prompt.

## Performance (RTX 5080 16 GB, 2× NVMe mirror, 32 GB RAM, i9-class CPU)

| scenario | before this tier | now |
|---|---|---|
| 826-token prefill (cold) | 578 s (CPU) | ~40 s |
| 3324-token prefill (cold) | 271 s (early GPU) | **90 s** (DeepGEMM), 225 s (generic DLL) |
| 8.3k opencode system prompt, first turn after start | ~600 s | **~4 min** — once per model (checkpoint on disk) |
| every later session with the same system prompt | 300 s | **6–9 s** |
| follow-up turn in a conversation (tool result) | 677 s | **6 s** |
| decode at 3.3k context | 0.6 tok/s | **1.5–1.6 tok/s** |

Where the time goes now: prefill ≈ 35 % expert-bank refill (disk-bound,
~6 GB/s across two NVMe), ≈ 35 % attention block (many small GPU stages +
host↔device syncs), ≈ 5 % MoE GEMM, rest indexer/dense/mHC. Decode ≈ 75 %
routed-expert reads (~230 × 12.6 MB per token from disk at 6 % VRAM hit
rate), rest GPU stages. More RAM/VRAM (higher expert hit rate) is the only
lever left below the disk limit; a multi-GPU design exists on paper only.

Profilers: `DSV4_ATTN_PROF=1` (per chunk-layer `attnprof`/`blockprof`/
`idxprof`), `DSV4_DECODE_PROF=1` (per-token `decprof`), `DSV4_CUDA_MOE_PROF=1`,
`DSV4_IDX_VERIFY=1` (GPU indexer projection/scoring vs CPU, bitwise),
`V4_PREFIX_LOG=1` (checkpoint/hint decisions), `v4_mirror` per-drive I/O
telemetry after each request.

## GPU coverage

| build | GPUs | prefill | decode |
|---|---|---|---|
| CPU only (no DLL / `DSV4_CUDA=0` / Linux default) | — | CPU reference | CPU reference |
| generic DLL / `CUDA=1` | any sm_80+ | GPU attention block, indexer, generic batched MoE on the VRAM bank | GPU attention, indexer, expert mirrors |
| DeepGEMM DLL / `CUDA=1 DEEPGEMM=1` | compute 12.x | as generic + tensor-core dense/MoE GEMMs | same as generic |
| multi-GPU | — | single device today (`DSV4_CUDA_DEVICE` selects); expert-parallel design drafted, not implemented | — |

## Environment reference (V4 engine)

Defaults in parentheses; all read by `c/deepseek_v4.c` unless noted `.cu`.

**GPU tier**
| var | meaning |
|---|---|
| `DSV4_CUDA` (1) | master switch for the V4 GPU tier; `0` = CPU only |
| `DSV4_CUDA_DEVICE` (0) | CUDA device ordinal |
| `COLI_DSV4_DLL` | Windows: force a backend DLL file name (loader) |
| `COLI_CUDA_ATTN_BATCH` (0) | `1` = GPU batched prefill attention block + GPU decode attention/indexer |
| `COLI_CUDA_MOE_BATCH` (0) | `1` = prefill MoE on the transient VRAM expert bank |
| `COLI_CUDA_MOE_BATCH_MIN` (256) | min fresh tokens to engage the bank |
| `DSV4_CUDA_EXPERT_MIRRORS` (4096) | upper bound on decode VRAM expert mirrors (~8 MB each); free VRAM sizes the cache at run time |
| `DSV4_CUDA_VRAM_RESERVE_MB` (2800 with the bank, else 600) | VRAM kept free while mirrors grow (bank + attention buffers) |
| `V4_MTP_GPU_MIRRORS` (16) | separate mirror cache for the MTP drafter |
| `DSV4_CUDA_PIN_HOST` (1, `.cu`) | page-lock expert-cache slabs for DMA uploads |
| `DSV4_CUDA_MOE_GROUPED` (1, `.cu`) | expert-grouped MoE GEMM rows; `0` = legacy replicated |
| `V4_MOE_BANK_FULL` (off) | `=N`: prefetch a whole layer's experts above N tokens (measured worse) |

**Prefill / checkpoints**
| var | meaning |
|---|---|
| `V4_PREFILL_CHUNK` (128) | chunk width, clamp [1,128] |
| `V4_PREFILL_SEGMENT` (4096) | atomic segment length (cancel/resume granularity, expert re-sweeps) |
| `V4_PREFIX_CKPT` (1) / `V4_PREFIX_CKPT_MIN` (512) / `V4_PREFIX_CKPT_SLOTS` (4) / `V4_PREFIX_CKPT_DISK` (1) | prefix checkpoints, see above |
| `V4_PREFIX_LOG` | log hint boundary / reuse decisions |
| `V4_IDX_BATCH` (1) | batched indexer selection in prefill; `0` = legacy per-token |
| `V4_IDX_IDENTITY` (0) | `1` = skip indexer scoring when every candidate fits under `index_topk` (index order instead of upstream's score order; ~8 % faster prefill, changes rounding) |
| `COLI_V4_ROWS16` (1) | `0` = never repack hot experts into the rows16 layout (reference matvec for all; numerics comparisons) |
| `CTX` (4096) / `NGEN` | context window / generation ceiling (CLI + serve) |

**Experts / I/O**
| var | meaning |
|---|---|
| `COLI_MODEL_MIRROR`, `SNAP_MIRROR`, `COLI_DISK_WEIGHTS` | dual-SSD read split (see ENVIRONMENT.md) |
| `V4_MOE_REFILL_GROUP` (6, max 16) | parallel lookups per bank refill group (bounded by pin slots; 12 measured best) |
| `V4_LOADER_LANES` (9, max 16) | persistent expert-loader threads for decode. The default suits the CPU path (1.41× decode, #1097); **with the GPU tier set `3`**: mirrors/RAM absorb most decode misses (9 lanes ≈ +4 % decode) while the extra readers contend with the MoE bank refill for the NVMe queues (~10 % slower prefill; measured on the reference box, v1.7.0) |
| `V4_EXPERT_UNION` (1) | per-chunk expert union batching on the CPU path |
| `COLI_V4_DIRECT` (1) | O_DIRECT streaming reads |
| `COLI_V4_AUTOPIN`, `COLI_V4_PREWARM`, `COLI_V4_SAVE_USAGE` | hot-expert pinning from `.coli_usage` history |

**Speculative decoding** (off by default): `V4_DRAFT`, `V4_MTP`, `V4_MTP_GB`,
`V4_MTP_MIN`, `V4_MTP_DRAFT`, `V4_MTP_PARTIAL_KEEP`, `V4_NGRAM` (1),
`V4_NGRAM_PARTIAL_KEEP`, `COLI_V4_MARKOV_SPEC`, `COLI_V4_MARKOV_BLOCK`,
`COLI_V4_MARKOV_KEEP`.

**Diagnostics**: `DSV4_ATTN_PROF`, `DSV4_DECODE_PROF`, `DSV4_IDX_VERIFY`,
`DSV4_CUDA_MOE_PROF`, `DSV4_CUDA_DG_PROFILE`/`_AB`/`_DUMP` (`.cu`),
`COLI_NO_OMP_TUNE`, `OMP_NUM_THREADS`.

Note: `COLI_CUDA`, `COLI_GPU`, `CUDA_DENSE`, `COLI_CUDA_PIPE` in the serve
line are read by the CLI/`colibri.c` conventions; the V4 engine's own switch is
`DSV4_CUDA` and the two `COLI_CUDA_*_BATCH` gates above.

## Validation

Tiny fixture (generated locally, ignored):

```bash
python -m pip install -r tools/requirements-deepseek-v4-tiny.txt
make deepseek-v4-tiny-check
```

Real checkpoint oracle:

```bash
make deepseek-v4-oracle MODEL=/path/to/DeepSeek-V4-Flash \
  MEMORY_GB=32 ORACLE_TEACHER_FORCING=32 ORACLE_GREEDY=20
```

GPU tier evidence (2026-08): each stage was accepted with greedy text
byte-identical to the engine's own reference chain at the time (826- and
3324-token benchmark prompts, 8 tokens; strip the CLI's trailing
`TUNE decode:` line before comparing); the GPU indexer projection/scoring and
the CPU-replica fp8 matmul are checked bitwise by `DSV4_IDX_VERIFY=1`; the CUDA
numerics test is `c/tests/test_dsv4_sparse_attn_batch_cuda.c`. Different
kernels are not bit-identical to each other in general — a GPU run and a CPU
run of the same prompt diverge by a rounding flip after some tokens, exactly
as two CPU runs with different hot-expert sets do (next section) — so text
identity is a regression check within one configuration, not a proof across
configurations.

### Linux CUDA tier under WSL2 (2026-08-16)

Ubuntu 22.04 on WSL2 (15 GB guest RAM, model read over `/mnt/c` + `/mnt/d`,
`--memory-gb 11`, canonical GPU env), prompt "Say hello in exactly one short
sentence.", 12-token ceiling. Both Linux binaries produced `Hello!` + EOS,
the same as the Windows engine on the same prompt; the wall-clock gap is the
9p file system (expert streaming), not the tier:

| build | tier | ttft | 3 decode tokens |
|---|---|---|---|
| Linux `CUDA=1 DEEPGEMM=1` (sm_120a) | `[DSV4 CUDA] device 0 ... sm_120`, dense mirrors 6.27 GiB | 119.9 s | 32.4 s |
| Linux `CUDA=1 CUDA_ARCH=portable` (generic) | same | 122.1 s | 33.2 s |
| Windows `deepseek_v4.exe` + `coli_cuda_dsv4_dg.dll` (NTFS, 2× NVMe) | same | 16.8 s | 2.0 s |

A native Linux box with the checkpoint on local NVMe is still the missing
measurement (host-slab pinning is Windows-only, so refill is pageable there).

### CPU-only audit (2026-08-15/16)

Question: does the CPU path still compute what upstream `dev` computes after
the GPU work? Method: 826-token prompt, 48 greedy tokens, real checkpoint,
upstream `dev` engine built from its own tree vs this tree's engine built
without the GPU tier (the Linux/macOS build shape) and with the tier
disabled.

- Compiles clean without `COLI_V4_GPU_TIER`; the tiny torch-reference suite
  (`tests/test_deepseek_v4_tiny.py`) is token-exact on the CPU-only build for
  every target case.
- On the real checkpoint, greedy text is **not** a stable identity check
  across configurations, in upstream itself: hot (cache-resident) experts run
  the vectorized `rows16` kernel while cold ones run the reference matvec, the
  two accumulate in different orders, and which experts are hot depends on the
  autopin history (`.coli_usage`, rewritten by every run), the chunk width and
  the hit pattern. Upstream produced different 48-token texts with and without
  its history; so did this tree; every variant is a coherent answer differing
  by a rounding flip a few tokens in.
- With that variable removed on both sides (`COLI_V4_ROWS16=0`
  `COLI_V4_AUTOPIN=0 COLI_V4_SAVE_USAGE=0`: reference kernels only, no
  history) **upstream and this tree's CPU-only build produce identical
  48-token text** — the CPU math is unchanged.
- One change of this branch did alter CPU accumulation order and is now off
  by default: the indexer identity short-circuit for `count <= index_topk`
  (returned candidates in index order; upstream returns them in score order,
  and the sparse attention reference sums in that order). `V4_IDX_IDENTITY=1`
  re-enables it (~8 % faster prefill on GPU, e.g. 96 s -> 88 s at 3.3k).
- CPU-only prefill speed: chunk 128 vs 64 identical within noise (788 vs
  797 s for 826 tokens); the pinned-slab, pipelined refill and all kernel
  work are GPU-only and do not touch the CPU path.
- Degenerate config guarded: a decode expert mirror cache below 8 entries
  (`DSV4_CUDA_EXPERT_MIRRORS<8`) recycled a slot still referenced within one
  layer and produced garbage; such caches now stay on the CPU.

## Follow-ups

- Non-greedy sampling and more serving slots.
- Linux CUDA tier: measure on a native Linux box (WSL2 verified), POSIX host pinning.
- Multi-GPU expert-parallel tier (design draft, untracked until built).
- Shared replacements for the two temporary private quant paths (rows16 cache).
