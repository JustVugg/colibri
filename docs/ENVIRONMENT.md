# Environment Variables

Reference for the environment variables read by the colibrì engine.

**Baseline generated from `dev @ def8419`** by scanning every `getenv()` / `getenv_utf8()` site in `c/*.c`, `c/*.h`, `c/*.cu` and `c/*.mm`. Individual entries are also maintained with their owning source. Defaults and behavior are taken from the source; see [MAINTAINING-DOCS.md](MAINTAINING-DOCS.md) to regenerate the full inventory after the code changes.

## Which program reads these?

**There are ten engine binaries, and they do not share a knob set.** The main
engine `c/colibri` (built from `c/colibri.c`, formerly `glm.c`) reads most of
what follows, but the sister engines read their own:

| Engine | Source | Its own variables |
|---|---|---|
| `colibri` | `c/colibri.c` | everything below except the three sections named for another engine |
| `glm53` | `c/glm53.c` | the `GLM53_*` family and `COLI_MAP_EXPERTS`: see [GLM-5.3-Flash engine](#glm-53-flash-engine-glm53) |
| `kimi_k3` | `c/kimi_k3.c` | the `K3_*` family, and the Vulkan expert tier's `COLI_VK_TIER*` — see [Kimi K3 engine](#kimi-k3-engine-kimi_k3) |
| `inkling` | `c/inkling.c` | `INK_*`, plus `CTX_MAX`, `PIN_N`, `REP_PEN`, `GPU_DEV`, `NOGPU`, the Vulkan expert tier's `COLI_VK_TIER*` — see [Inkling engine](#inkling-engine-inkling) |
| `qwen36` | `c/qwen36.c` | `QWEN_*`, `Q36_*`, its dense/CUDA-tier controls, the `CACHE_ROUTE` family (VRAM tier over RAM cache), and the Vulkan expert tier's `COLI_VK_TIER*` — see [Qwen3.6 engine](#qwen36-engine-qwen36) |
| `qwen38` | `c/qwen38.c` | `Q38_MAXT`, `Q38_EOS`, `Q38_NATIVE_FP8`, `Q38_NATIVE_BF16`, `Q38_EXPERT_INT4`, `Q38_MTP`, `Q38_MTP_DRAFTS`, `COLI_LOOKUP`, `Q38_PREFILL_BATCH`, `Q38_TRUNK_CPU_INT8`, `Q38_FP8_KERNEL`, `COLI_TIMERS`, the Vulkan expert tier's `COLI_VK_TIER*` — see [Qwen3.8 engine](#qwen38-engine-qwen38) |
| `olmoe` | `c/olmoe.c` | `HOT`, `WIDE`, `SMOOTH`, `CONF_LIMIT`, `MAX_NEW`, `CHAT`, `EXPERT_DROP`, `WARMUP`, the Vulkan expert tier's `COLI_VK_TIER*` — see [OLMoE engine](#olmoe-engine-olmoe) |
| `deepseek_v4` | `c/deepseek_v4.c` | `CTX`, the `V4_*` / `DSV4_*` families, the two `COLI_CUDA_*_BATCH` gates and the Vulkan expert tier's `COLI_VK_TIER*` — see [DeepSeek V4 engine](#deepseek-v4-engine-deepseek_v4); note that the CUDA section below describes `colibri.c` knobs (`COLI_CUDA`, `CUDA_DENSE`, ...) which the V4 engine does not read — its GPU switch is `DSV4_CUDA` |
| `deepseek_v41` | `c/deepseek_v41.c` | the `V41_*` family and the Vulkan expert tier's `COLI_VK_TIER*`: see [DeepSeek V4.1 engine](#deepseek-v41-engine-deepseek_v41) |
| `mimo` | `c/mimo.c` | the `MIMO_*` family, and the Vulkan expert tier's `COLI_VK_TIER*`: see [MiMo-V2.6 engine](#mimo-v26-engine-mimo) |

Setting an `INK_*` variable while running `colibri` does nothing, and vice
versa; nothing warns you about it. A few variables are genuinely shared because
they live in headers every engine includes (`COLI_USAGE`, `USAGE_SAVE`,
`COLI_USAGE_DECAY` in `route_trace.h`; `RANS_*` in `rans.h`;
`COLI_NO_OMP_TUNE` / `OMP_NUM_THREADS` in `omp_tune.h`).

You rarely export any of them by hand — the `coli` CLI and `openai_server.py`
translate most of their flags into these variables before launching the engine
(e.g. `--temp` → `TEMP`, `--ctx` → `CTX`). See [SETTINGS.md](SETTINGS.md) for
the flag → variable mapping. Export a variable directly only to reach a knob the
CLI doesn't surface, or to override what the CLI would set.

Format: `VAR` — default — effect.

---

## Common — everyday use

| Variable | Default | Effect |
|---|---|---|
| `RAM_GB` | `0` (auto ≈ 88% of free RAM) | RAM budget in GB for the resident/streamed expert working set. Higher → more experts stay hot → higher cache hit rate. Read by colibri, kimi_k3, glm53 and olmoe; on olmoe it sizes the expert cache once the dense weights are resident, and only when no `--cap` was given. |
| `CTX` | `4096` | Maximum context length (tokens) the KV cache is sized for. |
| `COLI_PREFILL_CHUNK` | `0` (off) | Run a long prompt through the layers in N-token slices instead of one pass. Every S-scaled activation buffer shrinks from prompt-sized to chunk-sized, which is the remedy when a long prompt exhausts CUDA scratch. Byte-identical output (verified at N=256). Skipped under an active MTP draft. **Cost:** a slice of 512 tokens already routes to essentially every expert of every layer (`P(miss) = (1-topk/n_experts)^N`), so each slice re-reads the whole non-resident expert set -- prefer the largest N that still fits your scratch. |
| `NGEN` | `256` (engine) | Max tokens to generate before stopping (stop tokens can end sooner). `coli --ngen` defaults to `1024`. |
| `COLI_TEMP` | `-1` (auto: `1.0` for chat/text, greedy elsewhere) | Sampling temperature. **`COLI_TEMP=0` = greedy/argmax = deterministic.** `TEMP` still works as a deprecated alias, but only if fully numeric: `$TEMP` is the temp-*directory* path on Windows and for the ROCm runtime (#509), so prefer `COLI_TEMP`. |
| `NUCLEUS` | `0.90` | Nucleus (top-p) mass kept when sampling. Slightly tighter than the official 0.95 because the int4 tail is noisy. |
| `TOPK` | `0` (off) | Top-k filter on the sampling distribution (`0` = no limit). |
| `TOPP` | `0` (off) | Top-p filter (`0` = use `NUCLEUS`). |
| `SEED` | unset → seeded from clock + PID | RNG seed for sampling. **Unset = different every run.** Set a fixed value for reproducible sampling. |
| `KVSAVE` | `1` (on) | Persist the KV cache to `<model>/.coli_kv` so a conversation reopens warm. `KVSAVE=0` disables save+load (lossless round-trip; does not change output). |
| `KV_SLOTS` | `1` | Number of independent KV conversation slots (1–16), used in serve mode. |
| `KV8` | `0` (off) | Store the MLA latent KV cache in fp8 e4m3 with a per-row scale: ~3.9× less KV RAM, and `.coli_kv` shrinks ~4× (saved as the v2 format; f32 v1 files are quantized on resume and rewritten). Adds DeepSeek-V3-class KV quantization noise to attention. CPU attention path only for now: the CUDA/Metal fused-attention fast paths read f32 KV rows, so under KV8 they fall back to the CPU consumer (native fp8 decode; a one-time notice is printed under `COLI_CUDA_ATTN=1`). Forces `COLI_CUDA_PIPE=0`. Native CUDA/Metal fp8-KV kernels are follow-up PRs. |
| `KV_TQ` | `0` (off) | Sub-byte MLA latent KV quantization, mutually exclusive with `KV8` (`KV_TQ` wins). `KV_TQ=4` is the recommended tier: rotated-int4 codec (randomized-Hadamard rotation + Lloyd codebook, per-row radius as the scale), ~7.6× less KV RAM than f32. `KV_TQ=2|3|5|6` selects the PolarQuant codec at that bit width (`KV_TQ_POLAR=1` forces PolarQuant at 4 bits too). Requires power-of-two row widths (`kv_lora`/`qk_rope`; the GLM MLA shapes 512/64 qualify) — on a model whose shapes don't, the engine refuses to start rather than silently zeroing the cache. A value below the 2–6 grid (e.g. `KV_TQ=1`) is treated as the recommended `4` with a notice, not as the most aggressive tier. `.coli_kv` is saved as the v3 format; a file saved under a different KV mode, codec, or bit width is refused with an explicit message and the cache restarts. Same CPU-only status as `KV8`: GPU fast paths fall back to the CPU consumer; native kernels are follow-up PRs. Forces `COLI_CUDA_PIPE=0`. |
| `THINK` | `0` (off) | Emit a `<think>` reasoning block. `THINK=1` turns on visible reasoning. |
| `MTP` | on | Multi-Token Prediction (speculative draft head). `MTP=0` disables it. |

---

## Performance / tuning

| Variable | Default | Effect |
|---|---|---|
| `COLI_METAL` | off | Enable the Apple-Silicon Metal GPU backend. Requires a `make METAL=1` build. |
| `COLI_METAL_GEMM_MIN` | `16` | Minimum matmul rows to dispatch a GEMM to the GPU (below this, stays on CPU). |
| `COLI_METAL_SPIN` | off | Keep a GPU keep-alive spinner running (reduces dispatch latency; costs power). |
| `COLI_METAL_PREFILL` | `0` (off) | `=1` runs S>4 (prefill) attention on the GPU. Off by default because the CPU path is bit-exact; this one is an opt-in speed/exactness trade. |
| `COLI_GEMM_CHUNK` | `1` (on) | Split a large GEMM dispatch into ≤2^25-thread chunks. `=0` restores the single full dispatch (the pre-fix behaviour), so the fix can be A/B'd on one binary. |
| `COLI_RTOP8` | `1` (on) | Parallel top-8 router kernel. `=0` falls back to the serial one. |
| `COLI_METAL_RESSET` | off | `=1` uses an `MTLResidencySet` (macOS 15+) for the resident buffers instead of per-dispatch `useResource` calls. |
| `PIPE` | `0` (off) | Overlap expert disk-load with matmul via I/O worker threads. Byte-identical output; reorders I/O. `PIPE=1` opts in. |
| `PIPE_WORKERS` | `8` | Number of pthread loaders when `PIPE=1`, or the io-wq worker maximum per ring when `URING=1` (capped at 64). Tune to SSD queue depth and available cores. |
| `COLI_PIPE_BLOCK` | `0` (spin) | `=1` makes `pipe_wait` block instead of spinning. Spinning wins on an idle box; blocking is better when the cores are contended. |
| `PILOT_WORKERS` | `1` | Pilot loader threads on the blocking (non-`URING`) `PILOT_REAL` path, via an SPMC ring. `>1` raises NVMe queue depth. Clamped to [1,16]; `1` is byte-identical to the historic behaviour. |
| `PILOT_EVICT_GUARD` | `1` (on) | Keep pilot-prefetched experts from being evicted before they are used. `=0` restores plain LRU eviction (A/B). Also read by `olmoe`. |
| `RSS_GUARD_GB` | the resolved RAM budget | Resident-set ceiling (GB) checked every 16 emitted tokens; the cache is trimmed when it is crossed. Set explicitly to guard tighter or looser than the RAM budget. |
| `XEXP` | `0` (off) | `=1` runs ONE OpenMP region across all experts of a batch-union block instead of ~2 fork/joins per expert. Engages only at S=1 with an all-resident int4 block, off the speculation window, and with the int4-IDOT S=1 family (`I4S<=1`); output is byte-identical to that family. Measured +11.6% on a 2-socket 48-core Ice Lake, but neutral-to-negative on a 24-core box — hence opt-in. Measure on your host. |
| `COLI_KV_SHARE` | `0` (off) | `=1` lets a new serve slot adopt an existing slot's KV prefix instead of re-prefilling it. Measured on 6x5090 with a 675-token shared prefix: slot TTFT 50.1s → 1.7s, generated tokens identical. |
| `KVB_FLASH_MB` | `2048` | Ceiling (MB) for the one-shot `kvb_all` k/v reconstruction buffer in prefill attention (#768 — 30.1 GB at ctx 262144, and `cap_for_ram` reserved it permanently). Above the ceiling the reconstruction is tiled with an online (flash-style) softmax: same rebuild total, ~tile-sized transient, output may differ from one-shot by rounding (same divergence class as the CUDA/Metal attention arms). `=0` disables tiling (always one-shot). DSA-selected rows always take the one-shot path. |
| `KVB_TILE_MB` | `512` | Tile size (MB) for the tiled reconstruction above. |
| `KVB_FLASH` | unset | `=1` forces the tiled path at any size, `=0` forces one-shot — overrides the `KVB_FLASH_MB` trigger (A/B switch). |
| `COLI_GROUP_ASYNC` | `0` (off) | `=1` issues and collects CUDA expert groups asynchronously so CPU and GPU overlap at decode (S≤4). |
| `COLI_DISKCLASS_WINDOW` | see source | Recency window (in ticks) for the DISK-CLASS heat statistic. |
| `URING` | `0` (off) | Linux-only queued expert I/O. `URING=1` implies `PIPE=1`, forces cold reads through io-wq (`IOSQE_ASYNC`), replaces blocking loader pthreads and spin waits with batched SQEs/CQEs, and batches `PILOT_REAL` loads on a separate ring. Use `DIRECT=1` for cold NVMe to avoid page-cache copy/readahead limits. Fails clearly if the kernel denies io_uring; incompatible with `COLI_MMAP=1`. |
| `DIRECT` | `0` (off) | Use `O_DIRECT`/unbuffered reads for expert slabs. **Drive-dependent — measure it on your hardware.** On real NVMe with DRAM cache and headroom it is often a large win (measured +34% decode with `PIPE=1` on a Blackwell/Windows box, and 4.25→9.69 GB/s in iobench on a GB10); on QLC/DRAM-less drives or slow/virtualised disks it can be neutral to negative. Helps sustained NVMe; keeps the zero-copy GPU path. |
| `COLI_NO_OMP_TUNE` | off | **Kill-switch** for the OpenMP hot-thread tuning (`OMP_WAIT_POLICY=active` spin + proc-bind). Set `=1` when the CPU is mostly waiting on the GPU (Metal) so spin doesn't steal the shared power budget. Hybrid CUDA/CPU hosts may test an explicit user-owned policy only with controlled profiling; see [tuning.md](tuning.md#hybrid-cudacpu-openmp-override). |
| `COLI_NUMA` | auto in generated plans on multi-socket Linux; otherwise off | `COLI_NUMA=1` selectively interleaves large expert and dense slabs across NUMA nodes via `mbind` (raw syscall, no libnuma). Helps multi-socket hosts (+7–40% expert matmul); silent no-op on single-node or non-Linux. Explicit `COLI_NUMA=0` overrides the generated plan. |
| `MLOCK` | `-1` (auto: on for macOS) | Wire the streamed expert cache into physical RAM (`mlock`) to dodge the memory compressor. `0` off, `1` force. |
| `CAP` | unset | Expert-cache cap (slots/layer) when no CLI positional was given. Precedence: explicit `--cap`/positional > `CAP` > platform default > historic default (#379). Mainly for direct `./glm` use — `coli` users should prefer `--cap`. |
| `CAP_RAISE` | `1` (on); `0` on Metal + macOS + fast model volume (#379) | Let the engine raise the expert-cache cap above `topk` when RAM allows (bigger batches). `0` fixes the cap. When the platform-aware Metal cache default engages (F_NOCACHE probe measured the model volume fast), the *default* flips to `0` — auto-raise re-creates the Metal residency churn the minimal cache avoids. An explicit `CAP_RAISE` always wins. |
| `COLI_SSD_FAST_GBS` | `4.0` | Threshold (GB/s, measured F_NOCACHE, cached in `<model>/.coli_ssd` — see [The `.coli_ssd` probe cache](#the-coli_ssd-probe-cache) below) at or above which the model volume counts as "fast" for the platform-aware Metal cache defaults (#379). |
| `PREFETCH` | `0` | Prefetch depth for streamed experts. |
| `COLI_MMAP` | `0` | colibri: serve the routed experts as read-only `mmap` views of their shards instead of copying each one into a cache slab, so the page cache is the expert cache. Experts only; for the dense weights see `TRUNK_RESIDENT_LAYERS`. Linux, macOS and FreeBSD: elsewhere (Windows) the experts are read into slabs as without it. Incompatible with `URING=1`. See [Weights from disk instead of RAM](#weights-from-disk-instead-of-ram). |
| `TRUNK_RESIDENT_LAYERS` | unset (the whole trunk resident) | colibri: keep the dense tensors of only the top N layers resident. In the layers below, the attention projections, the dense MLP and the shared expert become read-only `mmap` views of the shards, paged in from disk when the OS has evicted them; `0` maps every layer. Embeddings, LM head, norms, router, MTP layer and DSA indexer stay resident, and so does a tensor without its `.qs` companion (a bf16 checkpoint maps nothing). CPU-only: exits 2 with `COLI_METAL`, `COLI_VULKAN` or `COLI_CUDA` set. Linux, macOS and FreeBSD; on Windows the trunk stays resident. The RAM it frees goes to the expert cache: lower `RAM_GB` or `CAP` to lower the total (#1399). |
| `PIN` | unset | Path to a `.coli_usage`/stats file; pins the hottest experts into a resident "hot store" at startup. **`PIN=auto`** seeds from the model dir's live `.coli_usage` (appended after every turn, so each restart's pin placement follows the accumulated real workload) with `stats.txt` as the fallback for a virgin model dir; neither present → no pin this run. |
| `PIN_GB` | `10.0` | Size budget (GB) for the pinned hot store when `PIN` is set. |
| `AUTOPIN` | `1` (on) | Auto-pin the hot store from usage history once ≥5000 selections are recorded. Automatic pinning is capped so it cannot reduce the adaptive LRU capacity that fits before pinning; explicit `PIN`/`PIN_GB` settings remain authoritative. |
| `REPIN` | `0` (off) | Live re-pin the hot store every N emitted tokens (RFC). |
| `PILOT` | `0` (off) | Router-piloted cross-layer expert prefetch. |
| `PILOT_REAL` | `0` (off) | Value-preserving real cross-layer prefetch loads (`PILOT_REAL=1` opts in). |
| `PILOT_K` | `6` if `PILOT_REAL` else `8` | Number of experts the pilot prefetches per step. |
| `PILOT_TWO` | `0` (off) | Two-step shared-expert-corrected router prediction for the pilot. |
| `COUPLE` | unset | Path to a coupling-score file driving cross-layer expert prefetch (#176). When set, `couple_load` reads it. |
| `COUPLE_K` | `8` | Top-K coupled experts per layer when `COUPLE` is set. |
| `COUPLE_D` | `1` | Coupling lookahead depth (`1` or `2`) when `COUPLE` is set. |
| `CACHE_ROUTE` | `0` (off) | Opt-in max-rank cache-aware MoE routing (pin∪LRU prefer within top-M). Also read by `qwen36`, where the VRAM tier outranks the RAM cache. See [CACHE_ROUTE.md](CACHE_ROUTE.md). |
| `ROUTE_J` | `2` | Sacred top ranks always taken when `CACHE_ROUTE=1`. |
| `ROUTE_M` | `12` | Max-rank window for resident preference when `CACHE_ROUTE=1`. |
| `ROUTE_P` | `0` | Cumulative mass window for CACHE_ROUTE (`0` = fixed M). |
| `ROUTE_ALPHA` | `1` | Scale gate mass of substituted experts before renorm (`1` = off). |
| `ROUTE_AGREE` | auto | Overlap% + KL vs true top-K; auto-on when `CACHE_ROUTE=1`. Alone it changes nothing and prints the meters (always 100% / 0). |
| `ROUTE_TRACE` | unset | If set to a path, logs every routing decision there (testing/analysis). |
| `ABSORB` | `-1` (auto: absorbed for S≤4) | MLA attention absorption mode. |
| `IDOT` | `1` | Integer dot-product kernel. `IDOT=0` uses exact f32 kernels (for A/B numerical checks). |
| `COLI_POLICY` | `quality` | Resource policy: `quality`, `balanced`, or `experimental-fast`. |
| `PROF` | `0` (off) | Performance profile: a startup header (machine + effective config), then per run — or per turn in serve mode, on stderr — forward-latency percentiles (p50/p90/p99/max), expert-I/O totals and cache-tier fill, phase shares of wall time, and a verdict naming the knob most likely to help on this machine. Output is additive; `PROF` unset changes nothing. |
| `COLI_NO_FUSED_PAIR` | `0` (off) | `=1` disables the fused-pair matmul kernel. |
| `DISK_SPLIT` | `0` (off) | `=1` splits the reported disk-load time across the draft/absorb/forward phases in stats. |
| `I4S` | per-ISA (`1` on AVX-512-VNNI / NEON-dotprod, `2` elsewhere) | Engage the int4 `IDOT` kernel for batch `S>=<n>`. `I4S=1` turns IDOT on at decode too: int8-quantized activations on expert matmuls — **not bit-identical** to the f32 decode path (measured 0.39% of scale on the gate output; the same numerics prefill already uses at `S>=2`, and the shipped default on AVX-512-VNNI, measured +5.5% end-to-end there). Attention projections always stay exact regardless. A default flip on AVX-VNNI awaits the quality ablation. |
| `IDOT_GS` | `0` (off) | **Opt-in** grouped planar IDOT for `fmt=4` (gs64/gs128) tensors: int8 activations with the K1 plane layout, one integer dot per scale group. Same numerics family as `I4S=1` — not bit-identical to the f32 grouped kernel, hence off until the ablation. Requires the planar family (AVX2 or AVX-512 build — on AVX-512 only the fmt=4 tensors planarize, fmt=2 keeps the pair layout — no GPU backend, no `XEXP`). Multi-row calls (prefill batch-union, serve-mux decode) take a 1×4 row tile that pays each weight block's unpack once per 4 rows; on AVX-512-VNNI whole 64-element groups go through single `vpdpbusd` zmm ops. All shapes are bit-identical to each other and to the pure-C reference (integer group dots, same per-row fmaf order). Activation prints `[K1b]` once. |
| `AMX` | `1` (on where armable) | `=0` disables the K1c AMX int8 tile kernel inside the `IDOT_GS=1` family (Sapphire Rapids+; Linux arms tile state via `ARCH_REQ_XCOMP_PERM`, Windows 11 via `EnableProcessOptionalXStateFeatures`; other OSes fail closed). With gs a multiple of 64, one `tdpbssd` tile-multiply covers a scale group for 16 output rows × up to 16 activation rows; bit-identical to the vector K1b path. Arming prints `[K1c]` once. |
| `AMX_S_MIN` | `8` | Row threshold for the AMX tile kernel: below it the B-tile unpack does not amortize and the vector 1×4 tile is the better kernel. Measure on your host — the break-even depends on cache level and core count. |
| `SPEC_PIN` | `1` (on) | Speculation gate mode. `0` reverts to the legacy S-dependent speculation gates (#163). |
| `COLI_RAM_OVERCOMMIT` | off | `=1` overrides the "projected peak > MemAvailable → exit(2)" guard so a run that risks kernel OOM-kill is allowed to proceed. |

## Weights from disk instead of RAM

Every engine streams the routed experts from disk and keeps a cache of them in
RAM. Some can also leave weights on disk instead of copying them, which is the
lever when a model does not fit even with the smallest expert cache (#1764).
What each engine can do:

| Engine | Dense weights | Routed experts |
|---|---|---|
| `colibri` (GLM-5.2) | `TRUNK_RESIDENT_LAYERS=0`: attention, dense MLP and shared expert of every layer mapped from disk. CPU only; Linux, macOS, FreeBSD. | `COLI_MMAP=1`: mapped, the page cache is the cache. Linux, macOS, FreeBSD. |
| `kimi_k3` | `K3_MMAP=1`: every prepared matrix mapped, LM head included. The embedding is read one row per token in any case. CPU only. | Cache of at least one slot per layer (`K3_EXPERT_GB`). |
| `deepseek_v4` | Automatic: when the dense trunk or the BF16 head does not fit in `RAM_GB`, it is read from disk again on every use. The `ram_tiers` line on stderr says `dense=streamed`. | Cache of at least the top-k slots per layer. |
| `glm53` | Resident (`GLM53_BITS` picks 4, 8 or 32 bits). | `COLI_MAP_EXPERTS=1`: views of a per-shard mapping, CPU runs only. |
| `qwen38` | Resident (`Q38_TRUNK_CPU_INT8` keeps it as int8). | `COLI_MAP_EXPERTS=1`, native FP8 experts or the int4-g64 sidecar's records. |
| `qwen36`, `inkling`, `deepseek_v41`, `olmoe` | Resident. | Cache of at least one slot per layer. |

A mapped weight costs a disk read whenever the OS has evicted it, so in the
worst case every token reads every mapped byte: this is how a model runs at
all, not how it runs fast. Mapped bytes also do not count as resident, and the
engine hands the RAM they free to the expert cache; to lower the total, lower
`RAM_GB` (`coli --ram`) or `CAP` too. For GLM-5.2 on a machine where not even
the trunk fits: `TRUNK_RESIDENT_LAYERS=0 COLI_MMAP=1 RAM_GB=2`.

## The `.coli_ssd` probe cache

On Metal + macOS the engine's first startup measures the model volume with an
honest F_NOCACHE random-read probe (#379) and caches the result in
`<model>/.coli_ssd`, so every later startup reads a file instead of
re-measuring. Details that matter when you meet this file in the wild:

- **Cold-range steering.** `F_NOCACHE` bypasses the page cache only for pages
  that are not already resident, so probing a freshly-read (warm) shard would
  measure RAM, not the disk. The probe snapshots residency with `mincore` and
  reads only 4 MB windows that are entirely cold.
- **Contamination veto.** If the shard offers fewer than 64 MB of such cold
  windows, the measurement is refused: nothing is cached, one stderr line
  explains the deferral, the conservative (slow-storage) defaults hold, and
  the probe simply retries on the next, colder, startup. The same veto (with
  its own honest message) fires for an under-allocated shard — a sparse or
  still-downloading file whose "cold" pages are holes that would measure as
  RAM-speed zero-fill — and for a shard too small to ever offer 64 MB of
  probe windows. The probe measures the largest `.safetensors` in the dir.
- **Format (v2).** One line, `v2 <gbs> <st_dev>` — the measured GB/s and the
  `st_dev` of the model dir's volume at measurement time. The grammar is
  strict (plain digits, `0 < gbs < 1000`; no inf/nan/hex/exponents) and both
  readers — the C engine and `coli doctor`/`coli plan` — accept exactly the
  same bytes; anything else is ignored and re-probed, never trusted.
- **Volume identity (best-effort).** The cache is honored only while its
  recorded `st_dev` matches the model dir's current volume, so copying or
  rsyncing the model dir (including this hidden file) to another drive
  normally triggers a re-probe there instead of inheriting the old drive's
  number; doctor/plan likewise stop showing the stale value. This is
  best-effort, not an identity guarantee: macOS recycles `st_dev` values, so
  a cache carried to an external volume that happens to be assigned the old
  device id (e.g. drives attached one after another in the same slot) will be
  wrongly trusted until deleted. When in doubt after moving a model dir,
  delete `.coli_ssd`. True volume-UUID identity is a named follow-up.
- **Legacy upgrade.** A pre-v2 bare-number cache (written before steering
  existed, so possibly warm-contaminated) is re-measured once on the next
  startup and rewritten as v2.
- **Deleting the file is always safe** — the only cost is one ~0.35 s re-probe.
- **Split/mirror layouts:** the probe measures the **primary** model dir only
  (`COLI_MODEL`), and its verdict sets the cache defaults for the whole run.
  With `COLI_MODEL_DIRS`/`COLI_MODEL_MIRROR` spreading shards across drives of
  different speeds, that single-drive verdict is an approximation; revisit if
  mixed-speed split setups become common (the `COLI_DISK_WEIGHTS` startup
  probe already measures every drive, but feeds the split ratio, not the
  cache defaults).

---

## Dual-SSD streaming

| Variable | Default | Effect |
|---|---|---|
| `COLI_MODEL_DIRS` | unset | SPLIT the model across 2+ drives: a `;`/`,`-separated list of extra directories, each holding a **distinct** subset of the `.safetensors` shards (no duplication). Shards act as a search path — every shard is read from whichever drive holds it, so concurrent expert loads parallelise across drives and combined capacity is used. Scales to N drives. Metadata (config/tokenizer/`.coli_usage`) stays in the primary `COLI_MODEL` dir. Pairs well with `PIPE=1` (concurrent loaders) + `DIRECT=1`. Distinct from — and composable with — `COLI_MODEL_MIRROR`: the mirror is matched per-shard by basename against the merged (split) index, so a mirror dir may hold a copy of any subset of the split's shards. |
| `COLI_MODEL_MIRROR` | unset | `;`/`,`-separated list of directories, each a byte-identical (read-only) copy of the model on another drive; expert reads split across the primary and every mirror. Partial mirrors work (only the shards present are used). |
| `COLI_DISK_WEIGHTS` | unset (startup bandwidth probe) | Split ratio `<primary>,<mirror>[,<mirror2>...]` — one positive weight per drive (e.g. `1,1` for 50/50, `9,3` for a fast+slow pair, `1,1,1` for a 3-way mirror). Unset = probe every drive with the engine's own access pattern at startup. |
| `SNAP_MIRROR` | unset | Legacy alias for `COLI_MODEL_MIRROR`, consulted only when that is unset or empty. |
| `COLI_MIR_STRIPE` | see source | Stripe granularity for splitting a single expert read across mirror replicas. |

Per-drive byte counts are reported in a `MIRROR:` stats line. Combine with `DIRECT=1` so the two copies never compete for page cache.

## Vulkan (any GPU with a Vulkan 1.2 driver)

| Variable | Default | Effect |
|---|---|---|
| `COLI_VULKAN` | off | Enable the Vulkan backend. Requires a `make VK=1` build. The GLM engine fails at startup (no silent fallback) if libvulkan or the compiled shaders are missing. The other engines (qwen36, qwen38, inkling, olmoe, deepseek_v41, deepseek_v4, kimi_k3, mimo, glm53, qwenimage) print one line and run on the CPU. What each one puts on the device: [vulkan.md](vulkan.md#the-other-engines). |
| `COLI_VK_DEV` | unset | The primary Vulkan device by its index in the enumeration (0, 1, ...: the order `vulkaninfo --summary` lists as GPU0, GPU1, ...), for a machine with two GPUs of the same kind. Without it, the backend prefers a discrete GPU, then integrated/virtual devices, and with two discrete GPUs takes the first. An index out of range is ignored with a `[VK] COLI_VK_DEV=... ignored` line. `[VK] ready:` names the device chosen. |
| `COLI_VK_SHADERS` | auto | Path to the compiled `qmatmul.spv` **or** the directory holding the `.spv` set; the other shaders are found next to it. Unset: `shaders/` next to the binary, then CWD-relative `shaders/qmatmul.spv`. |
| `COLI_VK_EXPERTS` | unset | Deprecated. The GLM engine's routed experts are on the shared expert tier (below), sized by its budget. `N`: the tier holds at most N experts (it was the size of a fixed top-N set, 320 by default); `0`: no tier, as `COLI_VK_TIER=0`. ~19 MB per int4 expert. |
| `COLI_VK_DENSE` | `0` in this engine | Run the resident dense matmuls (attention projections, shared expert) on the GPU. The other engines read the same variable with a default of on (see the expert tier below); all of them go through `coli_vk_dense_decide()` in the backend. |
| `COLI_VK_ATTN` | `0` | Run the S≤4 MLA absorb attention core (+ fused o-projection) on the GPU, with a persistent device-side KV mirror. |
| `COLI_VK_QPREP` | `1` (on) | Fuse the Q-prep step (RMSNorm + rope + compress) into one GPU dispatch instead of splitting it, which cost three fences where one suffices. `0` restores the split path; `2` additionally keeps CPU reference copies of Q and comp for A/B comparison. |
| `COLI_VK_RESERVE_GB` | unset | Deprecated, GLM engine. Device memory (GB) the expert tier leaves to everything else, as `COLI_VK_TIER_RESERVE_GB`, of which it is now an alias: the larger of the two applies. With `COLI_VK_ATTN=1` the tier also leaves room for the absorb core's KV mirror (`CTX` rows a layer). |
| `COLI_VK_SPIN_US` | `300` | Microseconds to spin-poll a fence before blocking. `0` always blocks — lower latency at idle, at the cost of a core spinning. A tiled GEMM call spins through its dispatch (up to 50 ms) unless this is `0`: a blocked wait wakes up to a millisecond late, which doubled a 0.6 ms prefill GEMM on a Radeon 780M. |
| `COLI_VK_GEMM_MIN_S` | measured | When a resident matmul (`coli_vk_matmul`, every engine) takes the tiled GEMM instead of the per-row GEMV. Unset: S ≥ 2 and S·O ≥ 4096, measured on a Radeon 780M (a narrow matrix at small S stays on the GEMV). `N`: every call with S ≥ N. `0`: the GEMV for every S. S = 1 (decode) always stays on the GEMV. See [vulkan.md](vulkan.md#prefill-the-tiled-gemms). |
| `COLI_VK_COOP` | on where supported | The cooperative-matrix GEMM (`VK_KHR_cooperative_matrix`, 16x16x16 fp16 → fp32) for the formats whose weights decode exactly to fp16: int8, int4, int3-g64, MXFP4 and fp8, the grouped ones with a group size that is a multiple of 32. `0` keeps every format on the fp32 GEMM. |
| `COLI_VK_COOP_SG` | `64` where allowed | Subgroup size the cooperative-matrix pipeline requires (RDNA3: wave64 measured fastest). Tuning only. |
| `COLI_VK_CHAIN_GEMM` | on where supported | The dense chain's int8 and int4 prompt matmuls (`I` a multiple of 64, `O` of 128) on the matrix units (`chain_gemm.comp`: x rounded to f16 once) on a device with cooperative matrices at subgroup size 64. `0` keeps them on the chain's other GEMMs. |
| `COLI_VK_CHAIN_GEMM_TT` / `COLI_VK_CHAIN_GEMM_OXF` | `8` / `1` | `chain_gemm.comp`'s 16-row tiles a workgroup (8: 128 token rows) and the grid's x over the output blocks (1) instead of the token blocks: the workgroups running together share their x rows and each weight block is dequantised half as often. A 4k-token prompt of Qwen3.6-35B-A3B on a Radeon 8060S: trunk GEMMs 1551 -> 1093 ms. `4` / `0`: the old 64 rows and order. |
| `COLI_VK_GEMM_TILE` | measured | `bm,bn,bk,tm,tn[,pf]`: one tile for every width of the fp32 GEMM instead of the measured pair. Tuning only (`COLI_VK_TEST_GEMM_BENCH`). |
| `COLI_VK_COOP_TILE` | measured | `bm,bn,wm,wn,bk`: one tile for every width of the cooperative-matrix GEMM. Tuning only. |
| `COLI_VK_STAGED` | auto | Staged uploads: resident data (the weights, the expert tier, the MLA KV mirror, `COLI_VK_DEV2`'s experts) copied into device-local memory the host does not map, through a host staging buffer, instead of written through a mapping of the host-visible device-local memory. `1` on, `0` off. Unset: on when the host-visible device-local heap is under a quarter of the largest device-local heap, as on a discrete card without Resizable BAR (about 256 MB of 8 GB, where the mapped path fails or spills to system RAM); off on a card with Resizable BAR, an integrated GPU or Lavapipe. See [vulkan.md](vulkan.md#memory-placement-without-resizable-bar). |
| `COLI_VK_STAGED_FAULT` | unset | Tests: `<point>[:n]` makes the n-th staged upload (the first by default) fail at `stage` (the staging buffer), `pwstage` (the KV mirror's), `block` (a device-local block), `kvbuf` (a KV mirror or norm-weight buffer), `record` (a command buffer's begin or end), `submit`, `wait` (a fence wait: the device is then lost), `commit` (a tier expert's commit after its first matrix) or `import` (an import of host memory, which is then staged); a `[VK] COLI_VK_STAGED_FAULT` line says when it fired. |
| `COLI_VK_UP_IMPORT` | on | Staged uploads copy straight from host memory where the device has `VK_EXT_external_memory_host` (the source pages imported as a transfer source, no copy into the staging buffer). `0` stages every copy ([vulkan.md](vulkan.md#memory-placement-without-resizable-bar)). |
| `COLI_VK_HOST_VISIBLE_CAP_MB` | unset | Tests: treat the host-visible device-local heap as at most this many MiB in the `COLI_VK_STAGED` decision, so a device with Resizable BAR or unified memory takes the decision a card without it would (`246` emulates an RTX 3070 on its launch VBIOS). |

### The routed-expert tier (`vk_tier.c`, every MoE engine)

With `COLI_VULKAN=1` every MoE engine (qwen36, qwen38, inkling, olmoe, kimi_k3, mimo, deepseek_v41, deepseek_v4, colibri, glm53) keeps a cache of routed experts on the device that adapts while you chat, and computes the resident ones of each layer step while the CPU computes the rest. See [vulkan.md](vulkan.md#the-routed-expert-tier-vk_tierc) (each engine's expert formats and history there), and for the two GLM engines (the formats, `COLI_VK_EXPERTS`, glm53's `swiglu_limit`) [vulkan.md](vulkan.md#glm-52-and-glm-53-flash-on-the-tier). With the CUDA expert tier also on, CUDA wins and this tier stays off.

| Variable | Default | Effect |
|---|---|---|
| `COLI_VK_TIER` | on with `COLI_VULKAN=1` | `0`: no tier; the routed experts stay on the CPU (the dense trunk still uses the device). |
| `COLI_VK_TIER_GB` | measured | The tier's budget in GiB, within what the device can hold. Unset: on a discrete GPU, the free device-local memory (`VK_EXT_memory_budget`) less `COLI_VK_TIER_RESERVE_GB` and the dense weights still to be placed; on an integrated GPU or Lavapipe, whose device memory is the host's RAM, a quarter of what `MemAvailable` leaves once the engine's expert cache has grown to its size and the dense weights are placed, less 2 GiB. |
| `COLI_VK_TIER_RESERVE_GB` | `1` | Device memory left to everything but the experts (scratch, KV mirrors, the driver), on top of the dense weights. |
| `COLI_VK_TIER_RATE` | `16` | Promotions per token at most (a prompt's forward gets this many per prompt token); each copies one expert on the engine thread before the uploader thread writes it to the device. |
| `COLI_VK_TIER_WARM` | on | `0`: no warm start from the history; the tier fills as experts pass by. |
| `COLI_VK_TIER_MTP` | on a discrete GPU, off on one that shares the RAM | `1` puts a drafting head's routed experts on the tier as extra layers, in the form the snapshot keeps them: qwen38's MTP head (FP8 beside the int4 sidecar), colibri's MTP head (int8 beside int4), deepseek_v41's DSpark stages; `0` keeps them on the CPU. On a Radeon 780M it measured 3.44 tok/s against 3.48 without, hence the default there ([vulkan.md](vulkan.md#the-mtp-heads-layer-on-the-tier-coli_vk_tier_mtp)). |
| `COLI_VK_TIER_SYNC` | `0` | `1`: every layer step waits for the uploads staged so far before it runs, so which experts are resident depends on the routing alone, not on when the uploader thread ran (the tests set it: a tiny fixture's run can end before that thread is first scheduled). With `COLI_VK_TIER_BALANCE=0` as well, a run with the tier is reproducible. A promotion that displaces a resident while a batch is in flight waits for the batch's join to free it. Costs the overlap of uploads with compute. |
| `COLI_VK_TIER_BALANCE` | on | `0`: the device takes every resident expert of a step. On, when a join keeps waiting for the device, the step's resident experts the CPU also holds in RAM go back to the CPU beyond the device's share, which moves with every join (down when the CPU waited more than a tenth of its own time, up when the device finished before 80% of it). |
| `COLI_VK_TIER_GEMM_ROWS` | `16` | Rows from which an expert of a step takes the tiled GEMM (prefill) instead of the per-row GEMV; `0` never. Below it a row's bits do not depend on how many rows share the dispatch. |
| `COLI_VK_XB_GROUPED` | on where supported | A step's experts as one grouped GEMM (`qmatmul_grp.comp`): gate+up with the SwiGLU in one dispatch, down in another, the weights read by device address. Needs cooperative matrices at subgroup size 64, `bufferDeviceAddress`, hidden and expert widths that are multiples of 128, SwiGLU, and int8 or int4 experts (device fmt 1, 2, 4); otherwise the per-expert route. `0` keeps the per-expert route. |
| `COLI_VK_XB_GROUPED_ROWS` | `64` | Assignments (rows x top-k) in a batch from which the grouped GEMM takes it. Below it (decode, an MTP verify) the per-expert route keeps a row's bits independent of the batch. |
| `COLI_VK_XB_STEP` | on where supported | A big (prompt) step on the grouped GEMM: its token rows uploaded once, every expert's output written in place, and when every routed assignment ran on the device the routed sum per row computed there (`expert_sum.comp`, rank order). `0` keeps the packed rows and the host sum. |
| `COLI_VK_XB_GEMV` | on where supported | A batch below `COLI_VK_XB_GROUPED_ROWS` (decode, an MTP or lookup verify) as one grouped GEMV per phase (`qmatmul_grp_gemv.comp`: an (expert, row, 64 outputs) item a workgroup, the weights by address) instead of a GEMV dispatch per expert; a row's bits do not depend on the batch. Same device and expert conditions as the grouped GEMM, int4 groups a multiple of 32. `0` keeps the per-expert GEMVs. |
| `COLI_VK_BDA` | on where supported | `0`: the device is opened without `bufferDeviceAddress`, so the grouped GEMM is off. |
| `COLI_VK_TIER_COOP` | on where supported | Cooperative-matrix GEMM for resident and streamed expert batches. The same format and group-size checks as the per-matrix path apply to each gate, up and down projection independently; other projections keep the FP32 GEMM. `0` disables it; `COLI_VK_COOP=0` disables all cooperative pipelines. |
| `COLI_VK_TIER_COOP_ROWS` | `64` | Minimum rows of one expert for the cooperative path, also subject to `COLI_VK_TIER_GEMM_ROWS`. Small decode and draft-verification batches retain their GEMV arithmetic. |
| `COLI_VK_TIER_QUEUE` | a second queue | `0`: the tier's batches share the main queue with the dense matmuls (they then serialize). Unset: a second queue of the main family, else a compute-only family's (RADV), else shared (Lavapipe). |
| `COLI_VK_TIER_STREAM` | on | Streaming: a prompt step's cold (not resident) experts with enough rows are loaded as the CPU path would load them, uploaded into staging slots and computed on the device as GEMMs, the engine handing the tier the whole step (see [vulkan.md](vulkan.md#big-prompt-chunks-and-expert-streaming)). `0`: off, the tier as before. Needs an engine's load hook: every MoE engine has one. |
| `COLI_VK_TIER_STREAM_SLOTS` | `64` | Staging slots (experts), out of the tier's budget. |
| `COLI_VK_TIER_STREAM_ROWS` | the rule | Rows from which a cold expert streams. Unset: the larger of `COLI_VK_TIER_GEMM_ROWS` and the measured upload time over the measured CPU time per (row, expert); the `[VK] tier <engine> stream:` line says what it gave. Tests fix it. |
| `COLI_VK_TIER_STREAM_HALF` | x rows within 32 MiB | Rows of a streaming sub-batch (tests: small ones cut experts into parts). |
| `COLI_VK_TIER_STREAM_PAR` | on | `0`: a streamed expert's conversion into its slot runs on the engine thread alone. |
| `COLI_VK_DENSE` | on; off on a device sharing the CPU's RAM while the tier is on | Where the dense (non-expert) matrices of qwen36, qwen38, inkling, olmoe, kimi_k3 (its shared experts), mimo, deepseek_v41, deepseek_v4, glm53, qwenimage, and laya and gliner_decide with `COLI_VK_CHAIN=0`, run. `0`: on the CPU (with the tier on, the device takes the routed experts only); `1`: on the device whatever the device. Unset: on the device, except on an integrated GPU or a CPU device (Lavapipe) while the engine runs the expert tier (every one of them but qwenimage, which has no routed experts), where they stay on the CPU: there the dense matmuls cost more than the tier gains (measured on a Radeon 780M). The `[VK] <engine>: device ready, dense matrices on the ...` line says which and why. The GLM engine reads it through the same rule with its own default, off (see above). |
| `COLI_VK_CHAIN` | off with `COLI_VK_DENSE=0`; else on for a discrete GPU; qwen36 and olmoe on, qwen38 off, and mimo, inkling, colibri, glm53, kimi_k3, deepseek_v41 and deepseek_v4 off (not measured) on an integrated GPU with the tier; off on a CPU device | qwen36, qwen38, olmoe, inkling, mimo, colibri (GLM-5.2), glm53 (GLM-5.3 Flash), kimi_k3, deepseek_v41 (DeepSeek V4.1 Flash) and deepseek_v4: every layer's dense chain recorded into one submission per layer, the residual stream and the recurrent and KV state on the device (see [vulkan.md](vulkan.md#the-dense-chain-vk_chainc)). `0` off, `1` on, `2` on for prompts only (decode on the CPU). The `[VK] <engine>: dense chain ...` line says which and why. |
| `COLI_VK_DENSE_HOST` | auto | Whether the dense weights the device holds (the chain on, or `COLI_VK_DENSE` on) keep a copy in host RAM. `0`: the device only. Each resident dense matrix goes up at start and its host copy is given back; the steps the chain does not run take the device too; the CPU reads a matrix back from disk only when it needs one (a lost device), and the engine's expert cache no longer reserves the released host copies. `coli plan` still charges device allocations to physical RAM on integrated GPUs. `1`: the host copy stays as the CPU fallback, as before. Unset: the device only on an integrated GPU (its memory is the same RAM) and on a discrete GPU whose free memory, less 1 GiB, holds the dense weights; the host copy stays on a CPU device (Lavapipe), on a discrete GPU without room, and whenever the dense part runs on the CPU. The `[VK] <engine>: dense weights ...` line says which and why, a second one how much RAM was given back, and one at exit what was read back and the resident set. See [vulkan.md](vulkan.md#dense-weights-on-the-device-only). |
| `COLI_VK_CHAIN_ROWS` | from the budget, up to 8192 | Prompt rows per chunk of the chain's prefill. Unset or `auto`: the most rows whose scratch fits half the device memory free at the engine's first forward (also bounded by available physical RAM and Windows commit headroom), up to `COLI_VK_CHAIN_ROWS_MAX`; MiMo, GLM-5.3, Kimi K3 and DeepSeek V4 then hand the chain blocks of that chunk instead of `MIMO_CHUNK`, `GLM53_PREFILL_CHUNK`, `K3_CHUNK` and the 128-row block, when those are unset. Set: that many rows, the engines' own blocks as before. See [vulkan.md](vulkan.md#big-prompt-chunks-and-expert-streaming). |
| `COLI_VK_CHAIN_ROWS_MAX` | `8192` | The most rows the budget's chunk takes. |
| `COLI_VK_CHAIN_LAYERS` | from the device's free memory | How many layers the chain places on the device, the first N of them; the CPU runs the rest and the head (the partial chain, [vulkan.md](vulkan.md#a-partial-chain)). Unset or `auto`: the most layers from layer 0 that fit the device's free memory less `COLI_VK_TIER_RESERVE_GB`, the chain's fixed buffers and one prompt chunk's scratch; every layer (the chain as before) when they all fit. `n`: n layers, at most every one; `0`: the chain off, nothing of the dense part on the device. The `[VK] <engine> chain: N of L layers on the device ...` line says what was decided and why (`COLI_VK_CHAIN_LAYERS=n` when forced); `[VK] <engine> chain fit:` gives the numbers. With `COLI_VK_DENSE_HOST=0` (or its rule) only the N layers give back their host copies, and the RAM plan and `coli plan` count only those. Engines not yet on the partial chain keep every layer. |
| `COLI_VK_CHAIN_LAYERS2` | from the second device's free memory | With `COLI_VK_DEV2` and a partial chain: how many of the layers after the primary's the second device's chain places ([vulkan.md](vulkan.md#layers-on-two-devices)). Unset or `auto`: the most that fit that device's free memory less `COLI_VK_TIER_RESERVE_GB` and the chain's fixed buffers; `n`: n layers. The head stays on the host. The `[VK] <engine> dev2 chain: n of m layers on the device ...` line says what was decided. DeepSeek V4.1 starts the second device only at a layer that reads nothing the layers before it make (the primary's fit comes down to one). |
| `COLI_VK_CHAIN_DEV2` | `1` | `0` keeps the dense chain's layers off the second device (`COLI_VK_DEV2` then holds experts only). |
| `COLI_VK_CHAIN_FAULT2` | unset | Tests: the n-th frame on the second device's chain fails as a lost device's would (frame 1 is its setup: its layers stay on the CPU). Both chains go off; the CPU runs from there, the recurrent state rebuilt where a device held it. |
| `COLI_VK_CHAIN_MUX` | `16` | colibri: how many conversations' KV mirrors the chain keeps on the device beside its own for `KV_SLOTS`' batched decode ([vulkan.md](vulkan.md#glm-52-and-glm-53-flash-on-the-chain)), up to 16; past that the one read longest ago gives way. `0`: those steps run on the CPU, as before. |
| `COLI_VK_DEVICE_CAP_MB` | unset | Tests: the device holds at most this many MiB of device-local memory (a fraction is taken). An allocation past it fails as out of device memory, and every budget the engines read (`VK_EXT_memory_budget`'s, the expert tier's, the chunk's, the partial chain's fit) reports the cap and this process's bytes, so Lavapipe behaves like a small card. The second device (`COLI_VK_DEV2`) gets the same cap. A `[VK] COLI_VK_DEVICE_CAP_MB=...` line says it is on. |
| `COLI_VK_ATTN_BLOCK` | `16` | Rows from which the chain's attention takes the blocked shader (`chain_attnb.comp`: K and V read once per block of rows); `0` never. Below it, and at decode, `chain_attn.comp` as before. |
| `COLI_VK_ATTN_SLICE` | `4294967296` | Rows x positions x heads x head dim past which the chain's attention ops are recorded in slices of rows, a submission each, so no submission of a big prompt chunk runs past a driver's job timeout; `0` never. |
| `COLI_VK_CHAIN` (laya, gliner_decide) | on for a GPU, discrete or integrated; off on a CPU device | The decision engines' encoder forward (and Laya's head layers) recorded as one submission on the device, the activations staying there (`decide_vk.h`, [vulkan.md](vulkan.md#the-decision-engines)). `0`: matrix by matrix through the device (`COLI_VK_DENSE` decides, as above), the rest on the CPU; `1`: on. The `[VK] <engine>: forward on the device ...` line says which and why. Clef's is qwen36's chain, as above. |
| `COLI_VK_IMPORT` | on for a model without routed experts on a device sharing the CPU's RAM; off otherwise | qwen36: the device reads the int8 and f16 dense rows where the host keeps them (`VK_EXT_external_memory_host`, `coli_vk_tensor_import`) instead of holding a copy of each. `1` requests import, `0` copies. On an integrated GPU, imported rows remain shared even with `COLI_VK_DENSE_HOST=0`: they already have one physical copy, and their pages stay alive while Vulkan reads them. `COLI_VK_IMPORT=0` forces an independent device copy whose host pages can be released; only those released pages count as RAM given back. On other devices, `COLI_VK_DENSE_HOST=0` disables import. Unset, on for the dense 27B family (Qwen3.8-27B, Clef) on an integrated GPU or Lavapipe, whose trunk is the whole model and would be held twice (Clef's f16 trunk, 54 GB, twice past a 61 GB box); a copy elsewhere. Where the device cannot import (no extension, staged uploads) the rows are copied. See [vulkan.md](vulkan.md#the-decision-engines). |
| `COLI_VK_CHAIN_FLASH` | `16` | Rows of a forward from which the chain's causal attention (no selection list, window, ring or sink, head dim a multiple of 64) runs on the matrix units (`chain_attn_flash.comp`, f16 operands) on a device with cooperative matrices at subgroup size 64. Ahead of `COLI_VK_ATTN_BLOCK`'s blocked shader. `0` never. |
| `COLI_VK_CHAIN_GEMV` | on | `0`: the chain's decode matrices take `qmatmul.comp`'s GEMV instead of `chain_gemv.comp`'s. |
| `COLI_VK_CHAIN_GEMV_UNR` | `8` | 16-byte weight steps a lane of `chain_gemv.comp` loads before it decodes them (1..8; another value keeps the default). More loads in flight, the same sums in the same order. A decode step and a verify's rows run the same pipeline, so a row's bits do not depend on the driver. |
| `COLI_VK_CHAIN_GEMV2` | on where supported | The chain's int8 decode matrices on `chain_gemv2.comp`: rows a workgroup from the matrix's length (16 up to 1024 outputs, 32 up to 4096, else 64), a row's 16-byte words over a power-of-two lane group with clustered adds, up to four rows of x (a verify's) sharing each weight load with a decode step's bits. Needs clustered subgroup operations. `0` keeps `chain_gemv.comp`. |
| `COLI_VK_CHAIN_MOE` | `0` | `1`: a Qwen3.6 decode step's routed experts run inside the chain's frame when the expert tier holds every expert: routing (`chain_moe_route.comp`, one subgroup a row), the experts' gate\|up and down as grouped GEMVs over a table of the resident experts' buffer addresses, and their sum in rank order. One submission a token instead of a host round trip a layer; verifies up to eight rows. The routing weights come from the device's softmax, so the text can differ from the host path's in the last bits (decode perplexity of Qwen3.6-35B-A3B over 1024 tokens: 7.69 host, 7.60 device, 7.66 with the one-wave kernels). Needs subgroups of 64 and buffer device addresses. |
| `COLI_VK_MOE_WAVE` | on | With `COLI_VK_CHAIN_MOE=1`: Qwen3.6-35B-A3B's expert shapes (int4 g64, 2048 -> 512 and 512 -> 2048) on `qmatmul_grp_gemv_wave.comp`, one wave of 64 a workgroup, 16 / 32 outputs an item. `0`: `qmatmul_grp_gemv.comp` for every shape. |
| `COLI_VK_MOE_RPW_G` / `COLI_VK_MOE_RPW_D` | `16` / `32` with the one-wave kernels, else `64` | Output rows a grouped-GEMV item of the device MoE for gate\|up / down. |
| `COLI_VK_CHAIN_IDOT` | `0` | `1`: the chain's int4 decode GEMVs (groups of 32 or more) as W4A8 on `chain_gemv_idot.comp`: x quantized per 32 values to int8 with its own scale, eight packed int8 dot products a 16-byte weight step. A row's bits do not depend on the rows beside it, so verifies stay exact. Needs `VK_KHR_shader_integer_dot_product` (taken when the device has it). Qwen3.6-35B-A3B, int4 trunk, Radeon 8060S: decode +12 %, perplexity 8.05 -> 8.05 (with `COLI_INT4_GRID`: 7.64 -> 7.60). |
| `COLI_VK_CHAIN_FLASH_MR` | on | A prompt chunk's attention on the matrix units with four 16-row tiles a workgroup sharing each K/V block (`chain_attn_flash_mr.comp`). `0`: `chain_attn_flash.comp`, one tile a workgroup. |
| `COLI_VK_CHAIN_SPIN_US` | `2000` | How long a wait on a chain frame polls before blocking. |
| `COLI_VK_CHAIN_PROF` | off | `1`: device time per kind of chain op, one line per run. |
| `COLI_VK_KV_SPLIT` | on | Past the device's budget a chain engine keeps part of each attention layer's KV cache on the device and attends over the rest on the CPU, the two parts merged (DeepSeek stages only selected missing rows to preserve sparse-attention arithmetic) (see [vulkan.md](vulkan.md#a-kv-cache-past-the-devices-budget)). `0`: never split: the whole mirrors as before, and past the budget the chain declines. |
| `COLI_VK_KV_DEVICE_ROWS` | from the device's budget | The positions a split layer keeps on the device; set, the cache splits whenever it is longer (the tests' small device, a measurement's split). |
| `COLI_VK_KV_BLOCK` | `64` | Positions per block of the split cache's block table. |
| `COLI_VK_KV_PIN` | off | `1`: pin blocks by sparse-selection reads. For QSA/DSA/pooled MLA this can change rounding with read history; the default keeps the partition independent of chunking and prefix reuse. DeepSeek stages missing selected rows and preserves its arithmetic in either mode. |
| `COLI_VK_KV_COLD` | unset | `device`: the host's part of a split KV cache attended on the device too, from a shadow of the host's rows (GQA, MLA and Inkling forms; DeepSeek keeps it on the CPU). Off by default: on the measured integrated GPU it made the prompt 4-6% faster and decode 5.7x slower (see [vulkan.md](vulkan.md#a-kv-cache-past-the-devices-budget)). |
| `COLI_VK_CHAIN_FAULT` | off | Tests: the n-th chain frame fails as a lost device would; the engine rebuilds the state on the CPU and runs there (colibri, mimo and the DeepSeek engines have none to rebuild: the CPU continues from its host caches). |
| `DUMP` | unset | colibri, glm53, deepseek_v41 and deepseek_v4 in a `VK=1` build: every logits row the forward computes, appended to this file as raw f32 (the Vulkan gates compare the device's with the CPU's). |

### Second Vulkan device (opt-in)

Every engine on the routed-expert tier. A second GPU holds the experts after the primary device's: filled from the history at startup (the hottest on the primary device, the next ones there), then adapting as on the primary device; each step sends each device its own batch, both in flight at once. Its own pipelines, scratch and queue, so the primary device's hot path is untouched.

| Variable | Default | Effect |
|---|---|---|
| `COLI_VK_DEV2` | unset (off) | A second GPU for the routed-expert tier, every engine on it: the experts after the primary device's ([vulkan.md](vulkan.md#a-second-device-coli_vk_dev2)). A number selects that device index; `auto` picks a distinct real GPU, a discrete one first. A second *logical* device on the same physical GPU is accepted only when forced by index: that is the test mode. With `COLI_VK_TIER=0`, colibri keeps its own fixed registry there. |
| `COLI_VK_EXPERTS2` | tier: unset (the budget decides); colibri's registry: `512` | Expert count cap on the second device. |
| `COLI_VK_RESERVE2_GB` | `0.5` | Device memory (GiB) held back on the second device, as `COLI_VK_TIER_RESERVE_GB` is for the primary one. |
| `COLI_VK_TIER_EXCLUSIVE` | on | Exclusive RAM/VRAM: an engine's RAM expert cache that must evict gives up first a slot whose expert the Vulkan tier holds, and the prefetchers that read experts into RAM skip such an expert, so the two caches hold different experts when RAM is short. `0` keeps the copies ([vulkan.md](vulkan.md#ram-and-vram-without-the-same-experts-coli_vk_tier_exclusive)). |
| `COLI_V4_EXPERT_SLOTS` | unset | Tests: DeepSeek V4's RAM expert cache holds at most this many experts a layer (never below its minimum of 6, or every expert of a smaller layer). |
| `COLI_VK_DEV2_FAULT` | unset | Tests: the n-th batch joined on the second device fails there, as a lost device's does; the tier gives its experts back to the CPU and goes on with the primary device. |

See [docs/vulkan.md](vulkan.md). On multi-core boxes also set `COLI_NO_OMP_TUNE=1` (see that doc for why).

## CUDA (NVIDIA)

| Variable | Default | Effect |
|---|---|---|
| `COLI_CUDA` | off | Enable the CUDA backend. Requires a CUDA build. An explicit `COLI_CUDA=0` disables it **and suppresses the Windows bare-run auto-enable** (before this, Windows "CPU" runs with `COLI_CUDA=0` silently got a VRAM expert tier). The CLI flag `--gpu none` is the canonical hard off-switch on every platform. |
| `COLI_GPU` / `COLI_GPUS` | unset | Device selection (`auto`, `none`, or a list like `0,1`). Requires `COLI_CUDA=1`. |
| `CUDA_DENSE` | `0` | Place dense (non-expert) matmuls on the GPU. Off by default the engine reports `routed experts only (resident dense on CPU)`: on a host where the CPU is the limiter this leaves the dense path of every layer on the CPU while the VRAM tier serves experts only. Measured x2.8 on a 4x A6000 / 24-core host (1.53 -> 4.26 tok/s). |
| `CUDA_EXPERT_GB` | `0` | VRAM budget (GB) for caching experts on the GPU. Also accepts `auto`. |
| `CUDA_RESERVE_GB` | `2.0` | VRAM (GB) held back from the expert tier for activations, scratch and the KV cache. |
| `CUDA_EXPERT_LOAD_BALANCE` | `0` (off) | Experimental multi-GPU expert assignment: keep the same frequency-ranked GPU prefix, but greedily distribute it by accumulated profile weight instead of resident bytes alone. On one 6×RTX 5090 fixed replay its three-run median was +2.9%, with large variance; leave off unless validated on the target workload. |
| `CUDA_RELEASE_HOST` | auto (`1` if >1 device) | Release host-side copies after upload. |
| `COLI_CUDA_ROUTER` | `0` (off) | `=1` runs the MoE router (logits + top-k select) on the GPU at S=1. Skipped while a routing trace is being recorded, under `CACHE_ROUTE`, and above 4096 experts / topk 64. |
| `COLI_CUDA_RESID` | `0` (off) | `=1` keeps the residual stream on the device between layers instead of copying it back to the host each time. |
| `COLI_DSA_GATHER` | `0` (off) | `=1` gathers the DSA-selected KV rows on the GPU. With `DSA_FORCE=1` (identity selection) the output is byte-identical to the dense CUDA path, which is how the gather is validated. |
| `COLI_CUDA_ATTN` | off | Run S≤4 attention on the GPU. |
| `COLI_CUDA_ATTN_PREFIX` | off | Reuse one uploaded decode activation across `q_a` and `kv_a` while preserving the stock CPU RMSNorm path. |
| `COLI_CUDA_ATTN_SHARD` | off | `=1` splits KV-b heads across devices during attention load (multi-GPU). |
| `COLI_CUDA_PROFILE` | off | Emit CUDA timing. |
| `COLI_MTP_GUARD_PCT` | `70` | Pause MTP after the guard window when recent acceptance falls below this percentage. |
| `COLI_MTP_GUARD_WINDOW` | `24` | Number of MTP proposals used by the soft acceptance guard. |
| `COLI_CUDA_PIPE` | `0` (off) | `1` engages the multi-step attention pipeline; `2` enables the pipe2 path. |
| `COLI_CUDA_PIPE_SHARD` | off | `=1` runs the multi-device P2P head-shard attention path (opt-in for NVLink topologies; serializes ~95 MB/layer over a star PCIe topology). |
| `COLI_CUDA_PIPE_S_MIN` | `1` single-GPU, `8` multi-GPU | Minimum prefill batch S to engage the pipe2 CUDA path. |
| `COLI_CUDA_MTP` | `0` (off) | `=1` opts into MTP speculation under CUDA (off by default: cold streaming experts run on CPU where the fused-pair/IDOT kernels diverge in FP order, collapsing draft acceptance, #163/#292 — though #467 measured acceptance holding at 49% on sm_120). When set explicitly, the resource planner skips its `DRAFT=0` export so the engine's auto path can engage draft=3 — no need to also set `DRAFT`. Note the measured trade-off (#467): at ~85% hit the widened S=4 expert union costs more than speculation saves (−32%); the opt-in pays only near-full residency (~99% hit). |
| `COLI_CUDA_ASYNC` | on | `=0` forces synchronous `cudaMemcpy` instead of async + pinned host staging. |
| `COLI_CUDA_DUAL_PROJ` | on | `=0` issues gate+up as two separate launches instead of one fused `grouped_hidden_w4_dual`. |
| `COLI_CUDA_W4_PACKED` | on | `=0` disables the grouped packed-int4 path. |
| `COLI_CUDA_F8_WARP` | on (CUDA), off (HIP) | fmt=8 (fp8-e4m3) kernel selector. Default on CUDA: warp-per-row kernels with shared-memory LUT decode and reference-mirroring accumulation (f32 per 128-block, double across blocks, like the CPU `matmul_fp8`). `=0` restores the original fmt=8 kernels everywhere they run — grouped AND the dense `quant_matmul` branch. `=2` routes the warp kernels' decode through cuda_fp8.h: a real hardware `cvt` only on sm_89+, the header's bit-manip emulation below that, and plain `=1` behavior where cuda_fp8.h is absent (HIP); experimental until the 256-value sweep certifies it on the target silicon. Non-numeric values select the default. HIP defaults to `=0` because the warp kernels' wave64 width-32 shuffle sub-grouping is not yet validated on AMD silicon. |
| `COLI_CUDA_TC_INT4` | off | `=1` uses the W4A4 WMMA Tensor Core path (when all expert tensors are int4 and dims divide). |
| `COLI_CUDA_TC_MIN_ROWS` | `8` | Min rows-per-expert to engage the W4A4 Tensor Core path. |
| `COLI_CUDA_TC_W4A16` | off | `=1` uses the lossless W4A16 Tensor Core path (compute capability ≥7). |
| `COLI_CUDA_TC_W4A16_MIN` | `16` | Per-expert row threshold above which W4A16 TC tiles dispatch (smaller batches fall back to the naive kernel). |
| `COLI_CUDA_SHARED_W4A16` | off | `=1` uploads shared-expert weights and runs the shared-MLP W4A16 Tensor Core kernel. |
| `COLI_CUDA_SHARED_W4A16_MIN_ROWS` | `32` | Min row count to engage the shared-MLP W4A16 kernel. |
| `CUDA_RAW_EXPERTS` | unset | Experimental ANS build only: keep this many hottest experts raw, then store subsequent VRAM experts losslessly compressed. Requires `COLI_ANS_SIDECAR`. |
| `COLI_ANS_SIDECAR` | unset | Experimental ANS build only: path to the sequential compressed-expert sidecar. |
| `COLI_ANS_PACK` | `0` | Experimental ANS build only: `=1` creates `COLI_ANS_SIDECAR` during pinning and exits before inference. |
| `COLI_ANS_DIRECT` | `0` | Experimental ANS build on Linux: `=1` reads the sidecar with aligned `O_DIRECT`, bypassing page-cache overhead. Falls back to buffered I/O if unavailable. |
| `COLI_ANS_PROFILE` | `0` | Experimental ANS build: print sidecar header, read, staging/allocation, and H2D enqueue timings on first use. |
| `COLI_METAL_UNTRACKED` | off (Metal only) | `=1` sets `MTLResourceHazardTrackingModeUntracked` on Metal buffers (reduces hazard-tracking overhead). |

> **Windows note.** On Windows, a bare `coli chat` / `coli run` / `coli serve`
> (no `--gpu`/`--vram`/`--auto-tier`) **auto-enables the GPU** when it detects a
> CUDA build (`coli_cuda.dll` next to the engine) and at least one GPU via
> `nvidia-smi`. The expert-tier VRAM budget is then sized automatically from the
> card's free VRAM (same computation as `--auto-tier`). If `nvidia-smi` is not on
> `PATH` the run falls back to CPU with a warning — pass `--vram N` (or add
> `nvidia-smi` to `PATH`) to enable CUDA in that case. `--gpu none` forces
> CPU-only. (Linux/macOS behaviour is unchanged: pass a flag to enable CUDA.)

---

## Advanced / experimental / debug

These are for testing, benchmarking, or internal use — not part of the everyday surface, and some may change without notice.

| Variable | Default | Effect |
|---|---|---|
| `SPEC` | `1` | Speculative decoding on/off. |
| `DRAFT` | `-1` (auto: 3 with MTP, else 0) | Number of speculative draft tokens per step. |
| `GRAMMAR` | unset | Path to a GBNF grammar file to constrain generation. Takes precedence over `SCHEMA`. |
| `SCHEMA` | unset | Path to a JSON-Schema file compiled to GBNF to constrain generation (consulted only when `GRAMMAR` is empty). |
| `GRAMMAR_DRAFT` | unset | Max grammar-forced draft span length. |
| `COLI_LOOKUP` | `1` (on) | colibri without an MTP head: takes the n-gram drafts from the prompt lookup of [speculative.md](speculative.md) (the longest suffix of 4 down to 2 tokens seen before, its most recent occurrence) instead of the last bigram, up to `DRAFT` per verify (5 when `DRAFT` is unset), gated by the measured acceptance and forward times (`COLI_SPEC_GATE=0` drafts every proposal). Unset: the bigram source as before. |
| `COLI_DRAFT_CORPUS` | unset | Path to a file of frozen token ids (whitespace-separated, `-1` separates spans) used as a speculative draft source: the engine proposes the continuation that followed the longest suffix of the live context found in the corpus. Off when unset. Build one from any run with `TOKENS=1`. See [corpus-draft.md](corpus-draft.md). |
| `COLI_CORPUS_K` | `8` (max 48) | Proposal depth for `COLI_DRAFT_CORPUS`. Deeper raises the forward multiplier and the per-forward cost. |
| `COLI_CORPUS_MINACC` | `50` | Acceptance floor (percent) for the corpus source. Below it over a 24-proposal window the source pauses for 256 tokens, then re-arms — rejected drafts cost real time. |
| `EXPERT_BUDGET` | `0` (off) | Cap experts loaded per layer (MoE-Spec). **Quarantined:** silently forced to `0` unless `EXPERT_BUDGET_EXPERIMENTAL` is set — every tested value is either no faster or incoherent (issue #303). |
| `EXPERT_BUDGET_EXPERIMENTAL` | unset | Setting it (any value) allows `EXPERT_BUDGET>0` to actually take effect (expect garbage, #294). |
| `DEGRADE_ZERO` | `0` (off) | **Opt-in approximate mode:** miss slots with per-position gate weight < `DEGRADE_TAU` are zero-filled instead of triggering a blocking disk read. Decode-only (`S≤4`). Changes output — must be set explicitly. Measured on OLMoE-1B-7B: `tau=0.03` → +2.9% ppl, 21.8% slots zeroed; `tau=0.05` → +41% ppl. GLM-5.2 and Kimi K3 router contracts are unmeasured — treat `tau=0.03` as OLMoE-calibrated and tune per-model. `[PROF]` footer reports zeroed slot count and top-3 layers by drop share. See PR #906, issue #865. Calibration assumes a warm expert cache; cold-start transient is not characterized. |
| `DEGRADE_TAU` | `0.03` | Gate weight threshold for `DEGRADE_ZERO` (clamped to `(0, 1]`). Compared per-position, post-`norm_topk`, pre-`routed_scale` — i.e. as a fraction of each position's routed mass. |
| `DSA` | on | Dynamic Sparse Attention indexer. `DSA=0` disables. |
| `DSA_FORCE` | `0` | Force the DSA path on. |
| `DSA_TOPK` | model value | Override the DSA index top-k (testing). |
| `LOOKA` | `0` | Measure router predictability (instrumentation). |
| `I4_ACC512` / `I4_ACC512_TEST` | off | int4 512-wide accumulator kernel toggle / self-test. |
| `NOPACK` | off | Disable weight packing. |
| `DROP` | off | Drop-related debug toggle. |
| `PIN_FILL` | `0` | Fill the pinned store even without usage data. |
| `MTP_DEBUG` / `MTP_PRENORM` / `MTP_SWAP` | off | MTP head debugging / ablations. |
| `STATS` | unset | Write an expert-usage histogram to `STATS=<file>` at end of run. |
| `TOKENS` | unset | If set, dumps generated token ids to stderr for A/B comparison. |
| `SCORE` | unset | Scoring/eval mode over `SCORE=<file>`. |
| `SCORE_PREFIX` | on | If unset or `≠0`, prepends `[gMASK]<sop>` to scoring contexts (GLM-family only). |
| `REPIN_VERBOSE` | off | If set, prints per-swap `[REPIN]` diagnostics during VRAM repin. |
| `REF` / `REF_FORCE` | `ref_glm.json` | Reference-output comparison mode. |
| `REPLAY` | unset | Replay mode. |
| `TF` | unset | Teacher-forcing mode. |
| `ORACLE_STRICT` | unset (off) | `colibri` only, env-only. `=1` makes failed teacher-forcing (`TF`) and greedy oracle comparisons exit with status 1. Token-exact by default; only TF can use the mismatch allowance below. Non-finite logits and incomplete generation always fail strict mode; modes that bypass comparison are rejected. Unset or `0` keeps completed comparisons report-only. Invalid reference JSON/arrays fail regardless of this setting. See [CONTRIBUTING.md](../CONTRIBUTING.md) for the strict oracle commands. |
| `ORACLE_TF_MAX_MISMATCHES` | `0` | `colibri` only, env-only. Maximum token mismatches accepted with `ORACLE_STRICT=1` and `TF` set. Must be a nonnegative decimal integer smaller than the number of TF positions. CI uses `2` for the 32-position tiny fixture (30–32 matches); unset or `0` requires exact agreement. Mismatches remain visible in diagnostics. Ignored outside strict TF mode; cannot relax greedy comparison or non-finite-output checks. |
| `CHAT_TEMPLATE` | `1` | Apply the GLM chat template (`0` = raw prompt). |
| `PPL` | off (`olmoe.c` and `qwen38.c` only) | `PPL=1` enters teacher-forced NLL/perplexity meter mode in the OLMoE and Qwen3.8 sister engines. |
| `ABLATE_SCORE` | unset | Causal-ablation sweep over the manifest named by `ABLATE_SCORE=<file>`: one teacher-forced prefill per item, with a final-logit read-out at every target position. Runs before `SCORE` and exits when done, returning 0 only if every item completed. One item per line, fields separated by single spaces: `item T n_prompt mode ncells (layer expert swap){ncells} token{T}`. `mode` is 0 baseline, 1 contribution, 2 route-around, 3 module-swap; `swap` is `-1` except in mode 3. Every field is checked against the loaded config before anything runs, and a manifest that does not fit the format is refused whole, with the offending line named on stderr — no part of it is executed. One item's declared token count is refused above `min(CTX, 1,048,576)` tokens — the same context length `CTX` already sizes the KV cache for, not a bare constant — so a corrupt or hostile manifest cannot size the loader's per-item token array, the run's KV allocation, or its prefill buffer beyond what this engine's configured context could ever actually need. CRLF line endings and a missing final newline are accepted. |
| `ABLATE_OUT` | unset | Where the ablation sweep writes its read-out, as JSON lines: a header naming the run's vocabulary, layer/expert geometry, the SHA-256 of the loaded `config.json` and of the manifest, and the item and target counts to expect; one record per item; one record per target position carrying the gold token's negative log-likelihood, logits, margin, argmax and top-k; and a final record with the counts actually completed, so a truncated file can be told from a complete one. The path must not already exist — an existing file, link or directory entry is refused rather than overwritten. Optional: without `ABLATE_OUT` the sweep runs but writes nothing, and an accompanying `ROUTE_TRACE` records the post-ablation router trace. `c/tools/check_ablate_evidence.py` validates a finished file against the manifest and config it names. |
| `DEBUG_LOGITS` | unset | In reference-comparison mode, dump per-position logit diagnostics. |
| `COLI_LOGIT_DUMP` | unset | `=1` prints the top-5 `id:logit` pairs per step to stderr — for comparing two engine configs on identical forced context (backend-exactness triage). |
| `I3_AVX512` | auto | Force the AVX-512 int3 kernel on (`1`) or off (`0`). |
| `I3_AVX512_TEST` | unset | Run the AVX-512 int3 self-test and exit. |
| `COLI_GPU_FAIL_AFTER` | unset | Fault injection: make GPU compute calls start failing after N of them, to exercise the CPU fallback without real hardware faults. Uploads and queries are not gated. |
| `COLI_VK_TEST_BALLAST` | `0` | Allocate N extra dummy Vulkan buffers to reproduce decode attention degrading with expert-tier size even when VRAM is free (measured 7.9s @2.6k buffer objects → 15.6s @4.3k with 2.9 GB still free). |
| `COLI_VK_TEST_GEMM_BENCH` | unset | In the `VK_TEST` harness, time the GEMV against the fp32 and the cooperative-matrix GEMM per weight format and S, in GFLOP/s, instead of running the cases. `COLI_VK_TEST_GEMM_FMT=a,b,...`, `COLI_VK_TEST_GEMM_S=a,b,...` and `COLI_VK_TEST_GEMM_SHAPE=I,O` (default `2560,6144`) narrow it. |
| `COLI_VK_TEST_NOFILL` | unset | In the `VK_TEST` harness with staged uploads, skip the zero fill of a fresh device-local block (the run-to-run difference #1338 measured on Polaris without it). |
| `COLI_VK_TEST_HOSTMEM` | unset | In the `VK_TEST` harness, time the expert batch reading Qwen3.8-shaped int4-g64 experts from the tier's device memory against host memory imported with `VK_EXT_external_memory_host` (no copy), and what a copy into the tier costs, instead of running the cases. |
| `COLI_SERVE_ALL_STOPS` | unset | In batched serve mode, keep every stop token instead of filtering to the EOS-like ones. Trades the #401 tool-call safety for behaviour some non-tool clients prefer. |
| `VK_PROF` | unset | If set, time the Vulkan expert-group path and report it, and print at exit how the resident matmuls split between the GEMV, the fp32 GEMM and the cooperative-matrix GEMM, with their wall time. |
| `COLI_USAGE` | `<model>/.coli_usage` | Path to the expert-usage history to seed the ranking from, and to write back to. Shared by every engine (`route_trace.h`). |
| `COLI_USAGE_DECAY` | `1.0` (no decay) | Per-run multiplier applied to the recorded counts before ranking, i.e. a half-life. Without one the ranking freezes: after ~18M recorded selections one more turn moves it by 0.2% and the profile stops following the workload (#780). Values outside `(0,1]` are ignored. |
| `USAGE_SAVE` | `1` (on) | `=0` runs read-only — the usage history is loaded but never written back. For benchmark loops that would otherwise skew the profile they are measuring. |
| `RANS_PATH` | auto (best available) | Force a specific rANS kernel (`scalar`, `neon`, `avx512`, …). An unavailable choice yields `invalid` and fails loudly — never a silent downgrade. |
| `RANS_NEON` | on where built | `=0` kill-switch for the NEON rANS path. |
| `RANS_AVX512` | on where built | `=0` kill-switch for the AVX-512 rANS path. |
| `OMP_NUM_THREADS` | unset | Standard OpenMP variable. Setting it disables the engine's own OpenMP hot-thread tuning entirely — the user is assumed to be in charge. |

---

## GLM-5.3-Flash engine (`glm53`)

Read **only** by `c/glm53.c`. Like the other siblings it has its own loader,
cache and precision selection and shares none of the `colibri` knobs above.
See `docs/glm53-flash.md`.

| Variable | Default | Effect |
|---|---|---|
| `GLM53_BITS` | `4` | Precision of the resident dense weights: 4, 8 or 32. Routed experts are not affected — they arrive already quantized in the container and are never requantized. |
| `GLM53_EXPERT_GB` | measured | RAM budget (GB) for the expert LRU cache; per-layer slots are derived from it. Unset, it is taken from reclaimable physical memory after the weights are loaded (Linux `MemAvailable`, Windows available physical memory, macOS free+inactive+purgeable pages), minus a 3 GB margin. A fixed number is wrong in both directions: too small on a large machine leaves memory idle while the disk does all the work. |
| `GLM53_MAXT` | `8192` | KV state capacity in tokens, and the session size in serve mode. |
| `GLM53_PREFILL_CHUNK` | `128` | Prefill chunk size in tokens. Smaller keeps the workspace smaller; too small re-reads experts once per chunk per layer instead of amortizing them. In serve mode, CANCEL is checked before each prefill chunk. Cancellation waits for any chunk already running to finish. |
| `GLM53_REWIND` | `0` | Avoid reprocessing the prompt when a continue request trims cached trailing whitespace, by rewinding recurrent state with a snapshot (~149 MiB per slot, outside the `GLM53_EXPERT_GB` budget), allocated on first use, retained across slot resets, and copied once per generated whitespace run, whether or not the reply is ever continued. Off by default for that reason; exact-match continuations need no snapshot and reuse the cache either way. |
| `GLM53_MAX_IMAGE_TOKENS` | checkpoint's (8000) | Ceiling on tokens per image. Each covers 28×28 pixels, so 256 keeps ordinary text legible and 64 keeps shapes and colours. The image is shrunk, not cropped. Lower it: 8000 is 2691 tokens for a 1080p photo, i.e. a prefill nobody will sit through. |
| `GLM53_VERBOSE` | unset | Print the parsed geometry, the expert budget and the per-token cache cost to stderr. |
| `GLM53_DUMP_INDEX` | unset | Print the rows the sparse indexer selected. The first place to look when the engine diverges only at certain lengths. |
| `COLI_MAP_EXPERTS` | `0` | Serve the routed-expert pieces as read-only views of a per-shard mapping instead of copying each miss into a slab. CPU runs only: with Metal active the slots keep owned slabs, because the batched Metal MoE cannot register a view that does not start on its mapping's base. Also read by `qwen38`. See [Weights from disk instead of RAM](#weights-from-disk-instead-of-ram). |
| `COLI_VULKAN` | `0` | Open the shared Vulkan backend. Needs a `VK=1` build and the compiled shaders (`COLI_VK_SHADERS`). The streaming container's routed experts go to the shared expert tier (`COLI_VK_TIER*`, see the Vulkan section), which keeps the hot ones on the device instead of uploading each one as it arrives; it runs only when the checkpoint's `swiglu_limit` is above 0, the only clamp the device applies as the CPU does. The resident matrices follow `COLI_VK_DENSE`: unset, on the device, except on a device sharing the CPU's RAM while the tier runs. |

## Kimi K3 engine (`kimi_k3`)

Read **only** by `c/kimi_k3.c`. The K3 engine has its own loader, cache and quantization selection, so it does not share the `colibri` knobs above.

| Variable | Default | Effect |
|---|---|---|
| `K3_BITS` | `4` | Expert quantization width. Setting it at all also pins the choice (the engine otherwise infers it from the container). |
| `K3_MLA_BITS` | `8` | Quantization width for the MLA attention tensors. |
| `K3_HEAD_BITS` | `8` | Quantization width for the LM head. |
| `K3_MMAP` | `0` (off) | Map fully prepared U8 matrices and F32 sidecars read-only. CPU-only; refuses conversion and enabled GPU backends rather than falling back. |
| `K3_EXPERT_GB` | `8.0` | RAM budget (GB) for the expert LRU cache; per-layer slots are derived from it. |
| `K3_LAYERS` | `0` (all) | Load only the first N layers — for smoke tests and trace-only runs. |
| `K3_MAXT` | `np + ngen` one-shot, `8192` in serve | KV cache capacity in tokens. In serve mode it is also the prompt-rejection bound. |
| `K3_CHUNK` | `32` | Prefill chunk size in tokens. Clamped to [1,512]. |
| `K3_DIRECT` | `1` (on) | Use `O_DIRECT`/unbuffered reads for expert loads. `=0` for buffered. |
| `K3_IDOT` | `1` (on) | Integer dot-product kernels. `=0` uses exact f32 (A/B numerical checks). |
| `K3_PIPE` | `1` (on) | Overlap expert disk-load with compute. `=0` serializes. |
| `K3_LOAD_THREADS` | `4` | Loader threads for the pipe path. Clamped to [1,16]. |
| `K3_DIRS` | unset | Extra shard directories (`;`/`,`-separated) for a multi-drive split, as `COLI_MODEL_DIRS` is for `colibri`. |
| `K3_TOPP` | `0` (off) | Prune routed experts to this cumulative gate weight. A quality lever — A/B it against `K3_LOGITS`. |
| `K3_THINK` | `1` (on) | Emit a reasoning block. `=0` disables. |
| `COLI_VULKAN` | `0` | `VK=1` build: the routed experts on the shared expert tier (`COLI_VK_TIER*`, see [Vulkan](#vulkan-any-gpu-with-a-vulkan-12-driver)), as the checkpoint's MXFP4 with SiTU-GLU on the device, warm-started from the expert history `COLI_USAGE`; the shared experts' matrices on the device where `COLI_VK_DENSE` puts the dense matrices. The tier's experts match `K3_IDOT=0` (the CPU's default kernel rounds activations to int8). With `K3_CUDA=1` too, CUDA wins and the Vulkan tier stays off. |
| `K3_VK` | unset (follow `COLI_VULKAN`) | The switch of the engine's tier before the shared one, kept as an alias: `1` opens the device as `COLI_VULKAN=1` does; `0` keeps it closed whatever `COLI_VULKAN` says. Before the shared tier a `VK=1` build opened the device without being asked (`1` was the default); it now waits for one of the two. |
| `K3_VK_GB` | unset | Alias: read as `COLI_VK_TIER_GB` (the routed experts' budget, GiB) when that is unset. |
| `K3_VK_UP` | unset | Alias: a number is read as `COLI_VK_TIER_RATE` (promotions per token) when that is unset. `auto` (and `K3_VK_FILL_FRAC`) is accepted and does nothing: the shared tier uploads on a thread of its own and never holds a step. |
| `K3_PREFIX_LOG` | unset | Log the KV-prefix reuse decision either way, with the reason when it is "no" — "it did not get faster" is otherwise indistinguishable from "reuse is off". |
| `K3_CHAT_IDS` | unset | Print the chat-template token ids for the built prompt, then continue. |
| `K3_TRACE` | unset | Write a routing trace to `K3_TRACE=<file>`. |
| `K3_LOGITS` | unset | Write per-step logits to `K3_LOGITS=<file>`. |
| `K3_X0` | unset | Read input rows `[T, hidden]` as f32 from this file, bypassing the embedding — for feeding activations captured elsewhere. |

## Inkling engine (`inkling`)

Read **only** by `c/inkling.c`.

| Variable | Default | Effect |
|---|---|---|
| `CTX_MAX` | `8192` | Served KV bound. A prompt plus its requested generation beyond this is rejected rather than truncated. |
| `PIN_N` | `cap / 2` | Experts pinned per layer. Measured on the 975B: `cap/4` (19/layer) gave 83.6% hit / 0.32 tok/s, 40/layer gave 95.6% / 0.80 tok/s — decode fills run at queue depth ~1, so every pinned expert removes a ~35 ms stall. Clamped to `cap - 8`. |
| `REP_PEN` | `1.1` | Repetition penalty over a 128-token history (prompt tail + emitted). |
| `INK_DENSE_Q4` | auto | Use the `dense-int4g64/` sidecar for dense weights when that directory exists. `=0` forces the unquantized dense path. |
| `INK_SHARED_BATCH` | auto | Prefill rows per shared-expert batch, bounded to 64 MiB of scratch. `=0` restores the scalar per-token path for A/B/debugging; a positive value caps the chunk size. Decode (`S=1`) is unchanged. |
| `INK_METAL_MIN_S` | `1` | Minimum batch S to send the MoE block to Metal. `=2` restores the prefill-only gate (which mattered when the residency set was absent and per-block `useResource` churn cost ~135 ms). |
| `INK_PREFIX_LOG` | unset | Log the KV-prefix reuse decision and its reason, as `K3_PREFIX_LOG` does for K3. |
| `COLI_PREFIX_LOG` | unset | Same line for the engines that take the shared record (Qwen3.6, OLMoE): reports how many prompt tokens were reused, or why none were. |
| `COLI_KV_PREFIX` | on, except DeepSeek V4.1 | `0` disables KV-prefix reuse; on `deepseek_v41` reuse is OFF until you set `1`. That engine reads one set of index keys for a prefilled position and another for a decoded one, both the vendor's, so a prefix holding an earlier turn's generated tokens answers differently than the same text read cold. A resumed prefill is exact. |
| `GPU_DEV` | `0` | CUDA device index for the inkling CUDA backend. |
| `NOGPU` | unset | If set, skip GPU init entirely (both CUDA and Metal), regardless of the other GPU variables. |
| `COLI_VULKAN` | `0` | `VK=1` build: the dense and shared-expert matrices on the Vulkan device, and the routed experts on the shared expert tier (`COLI_VK_TIER*`, `COLI_VK_DENSE`, see [Vulkan](#vulkan-any-gpu-with-a-vulkan-12-driver)) in the form the cache holds them: the int4 or int8 container, the runtime int8 quantization, f32 at `bits=0`. The warm start reads the cache-warming history (`<snap>/.coli_usage` or `PIN=<path>`) in the generate and serve modes; the `ref.json` oracle reads none. With the CUDA backend or Metal on, the Vulkan tier stays off. |

## Qwen3.6 engine (`qwen36`)

Read **only** by `c/qwen36.c`. See [qwen36.md](qwen36.md) for the model layout
and the CPU/GPU execution split.

| Variable | Default | Effect |
|---|---|---|
| `COLI_DENSE_I8` | `1` (on) | Quantize resident dense matrices to per-row int8 at startup. `=0` keeps the f32 reference path for quality A/Bs. |
| `COLI_DENSE_IDOT` | `1` (on) | The dense trunk's GEMVs (DeltaNet projections and out_proj, attention q/k/v/o, shared expert, lm_head) quantize the activation to int8 once per call and run integer dot products (maddubs on AVX2, vpdpbusd on AVX-VNNI / AVX-512 VNNI) instead of converting every int8 weight to f32. Not bit-identical to the f32 path; measured +1.0% perplexity, lm_head 12.6 to 10.2 ms/token. `=0` restores the f32-activation kernel. |
| `QWEN_EXPERT_ACT` | `i8` | The routed experts' activation quantized to int8 once per row (expert_ffn.h mode 1). Measured +0.1% perplexity, expert compute 22.7 to 15.9 ms/token. `=f32` restores f32 activations and the bit-identical contract with the pair kernels. |
| `COLI_DENSE_BITS` | `8` (a Clef checkpoint under the gateway: `16` when the planner's RAM budget holds it, docs/clef.md) | `=4` stores the dense trunk as int4 in blocks of 64 with one scale per block (the K1b planar layout, half the bytes), served by the grouped integer kernel; implies the integer dot. Opt-in: on the 35B it costs +10% perplexity on the whole trunk, +2.4% on lm_head alone (see `COLI_DENSE_INT4`). qwen36 `=16` keeps the container's f16 values, no quantization, twice int8's RAM: on Clef it moves a probability by 0.012 where int8 moves it by 0.22. |
| `COLI_INT4_GRID` / `COLI_INT4_GRID_STEP` | `0` / `0.1` | int4 trunk packing (`COLI_DENSE_BITS=4`): the least-squares scale refinement of each group of 64 also starts from `amax / (7 ± step·i)`, `i = 1..n`, and `amax / 8`, the best kept. Measured on Qwen3.6-35B-A3B, decode perplexity (1024 tokens, single steps): 8.05 at `0` -> 7.60 at `5` / `0.8` with the W4A8 GEMV (int8 trunk 7.65), same size and speed. The search costs about 20 s at start; `COLI_INT4_CACHE` keeps its result. |
| `COLI_INT4_CACHE` | `~/.cache/colibri-int4` | Where `COLI_INT4_GRID`'s packed matrices are kept, one file a matrix keyed by a sample of its weights, its shape and the search (`0`: off). About 1.1 GB for Qwen3.6-35B-A3B; a later start reads them in a few seconds. |
| `Q36_MAX_IMAGE_TOKENS` | the checkpoint's preprocessor ceiling | Gateway, qwen36 containers with a vision tower (Qwen3.8-27B): ceiling on the tokens one image costs. The image is shrunk, not cropped. Without one a 1080p photo is about 2000 tokens of tower and prefill on the CPU. |
| `COLI_DENSE_KEEP_I8` | `0` (off) | qwen36: a matrix that got its int4 copy (`COLI_DENSE_BITS=4`) frees its int8 copy, which only the CUDA placer reads; `=1` keeps both. With `COLI_CUDA=1` both are kept anyway. Measured on Qwen3.8-27B: 42.4 GB resident with both copies, 18.8 GB without. |
| `COLI_DENSE_INT4` | all components | With `COLI_DENSE_BITS=4`, a comma list of the components that take int4: `lmhead`, `dnproj`, `dnout`, `attn`, `shexp`, `router`. Measured on the 35B: `lmhead` +2.4% perplexity for 254 MB less per token; `lmhead,dnproj,dnout` +5.6%; everything +10%. |
| `QWEN_EXPERT_KERNEL` | `1` (on) | Routed experts run through the shared `expert_ffn.h` kernel: the int4 stays packed in RAM (planar layout, half the expert-cache RSS of the int8 unpack), gate+up are one pass, and a layer is two OpenMP regions over (expert, row-chunk) items instead of 3 x top-k GEMV regions. Takes effect on an int4 gs=64 container whose hidden and expert widths are multiples of 64, and not under the CUDA expert tier. `=0` restores the unpack-to-int8 path; the two produce the same tokens (1024-token decode on the real container byte-identical; pinned on the tiny int4 fixture in CI), only the f32 accumulation order inside a dot differs. Measured at cap 256 on the real container: 12.8 -> 15.7 tok/s, peak RSS 29 -> 17 GB. |
| `Q36_DN_SNAP1` | `1` (on) | Vulkan chain: a speculative verify's DeltaNet copies (the convolution ring and the recurrent state after each of its rows) are written by the one convolution and the one recurrence dispatch of the layer, into slots behind the state in its own buffer; a rejection copies its slot back on the device. `0`: one dispatch a copy, as before. |
| `QWEN_DENSE_BATCH` | `1` (on) | On AVX2/FMA, reuse each dense-int8 weight decode across two prompt rows. `=0` restores one GEMV call per row. Decode `S=1` is unchanged. |
| `QWEN_SHARED_BATCH` | bounded by 32 MiB scratch | Batch the CPU shared expert across prompt rows. `=0` restores scalar calls; a positive integer caps rows per chunk. The CUDA-tier overlap path is unchanged. |
| `COLI_LOOKUP` | `1` (on) | Prompt-lookup drafts (`0` turns them off). When the recent tokens repeat an n-gram (4 down to 2 tokens) of the prompt or the output, the tokens that followed it are drafted and checked in one verify forward; the output is that of plain decoding, greedy or sampled. A gate drafts only where the measured acceptance and verify cost say it pays. Off under the CUDA tier, `CACHE_ROUTE` and qpack. See [speculative.md](speculative.md). |
| `Q36_MTP` | `0` | `1`: load the release's `mtp.*` block from the converted container and draft beside prompt lookup. Missing head: plain decoding with a diagnostic. Off under CUDA tier, `CACHE_ROUTE`, qpack. See [qwen36.md](qwen36.md#the-mtp-head-q36_mtp). |
| `Q36_MTP_DRAFTS` | `1` | Draft depth `1`..`7`, or `auto` / `0` for the gate's choice; each deeper row reads the head's previous pre-norm residual. |
| `Q36_MTP_PMIN` | `0` | Stop deeper drafts when the newest draft probability is below this value (`0`..`1`). With a vocabulary list, probability is conditional on that list. |
| `Q36_MTP_VOCAB_IDS` | unset | File of whitespace-separated token IDs to score with the shared `lm_head`; duplicates removed, malformed/out-of-range IDs refused. Verify always uses the full vocabulary. |
| `Q36_MTP_GPU` | `1` | With the Vulkan chain, run the head on the device. `0`: CPU head. |
| `Q36_MTP_DMOE` | `1` | Device MoE routing/grouped experts when supported; `0`: host routing and per-expert device matrix operations. |
| `Q36_MTP_FORCE` | unset | Tests: `reject`, `accept`, `mixed`, `cycle`, `row1`..`row7`. In `ref.json` mode, replace proposals with reference tokens except forced wrong rows; `reject` also rejects every draft in text/serve mode. |
| `Q36_MTP_DUMP` | unset | Tests: append `(int32 backbone row, int32 next token, float32[vocab] draft logits)` for each first draft to this file. |
| `COLI_LOOKUP_DRAFTS` | `5` | Lookup drafts per verify at most (1..5); a verify of `k` drafts copies the DeltaNet state after `k` rows (63 MiB each on the 35B). |
| `COLI_SPEC_GATE` | `1` (on) | `=0`: every proposal drafted in full, the gate's estimates ignored (tests). |
| `COLI_LOOKUP_FORCE` | unset | Tests only, `ref.json` mode: the reference's tokens as the proposal; `accept`, `mixed`, `cycle`, `row1`..`row5` as for `Q38_MTP_FORCE`. |
| `Q36_MAXT` | conservative engine default (16384 for a Clef checkpoint, its own `max_length`) | Lower the served/context capacity; it cannot raise the model's compiled safety ceiling. |
| `COLI_VULKAN` | `0` | `VK=1` build: the dense trunk on the Vulkan device, and the routed experts on the shared expert tier (`COLI_VK_TIER*`, `COLI_VK_DENSE`, see [Vulkan](#vulkan-any-gpu-with-a-vulkan-12-driver)). With the tier on, the engine keeps the expert history `COLI_USAGE` (default `<snap>/.coli_usage`), saved at every run and serve turn end; it keeps none otherwise. |
| `Q36_DN_GPU` | `0` (off) | CUDA expert tier: a decode token runs every DeltaNet layer whose in_proj and out_proj sit on one card end to end on that card (conv, recurrence, gated norm; state resident in VRAM). Measured on the 35B, 3070, trunk in VRAM: DeltaNet 16.2 -> 10.8 ms/token, the token 39.4 -> 33.3 ms. See [qwen36-cuda-tier.md](qwen36-cuda-tier.md#the-deltanet-layer-on-the-card-q36_dn_gpu1). |
| `QT_HOME` | `expert` | CUDA expert tier on two or more cards: `layer` homes every expert of a layer on one card (layer ranges weighted by allowance / probed dense-GEMV time per card, `QT_LAYER_SPLIT=<n>` layers on the first card overrides), with the layer's trunk and, for the last layer, lm_head alongside -- a pipeline instead of a per-layer join. See [qwen36-cuda-tier.md](qwen36-cuda-tier.md#two-cards-as-a-pipeline-qt_homelayer). |
| `Q36_OFFER_SHEXP` | `0` (off) | Offer the shared expert to the VRAM placer. Off by default: on the card it costs 6.3 ms/token (3070) to 11-12.7 ms (with a slower second card) against 3.7-4.0 on the CPU, where it hides behind the expert group. |
| `QT_PREFILL_REPLAN` | `0` (off) | CUDA expert tier: after each prefill layer's routing, swap VRAM residents this prompt never routed to for its most-routed non-residents of that layer, budget-neutral, uploads overlapping the rest of the prefill. Measured on the 35B: decode VRAM hit rate 60 -> 73 % against a heat file from other prompts, 38 -> 75 % from a cold start. See [qwen36-cuda-tier.md](qwen36-cuda-tier.md#the-residents-follow-the-prompt-qt_prefill_replan1). |
| `QT_PREFILL_REPLAN_MAX` | `24` | Cap on the swaps planned per prefill layer by `QT_PREFILL_REPLAN` (offline at B=96 the uncapped plan's hit rate with two thirds of the swaps). |

## Qwen3.8 engine (`qwen38`)

Read **only** by `c/qwen38.c`. See [qwen38.md](qwen38.md) for the native FP8
checkpoint layout and the text-only capability boundary.

| Variable | Default | Effect |
|---|---|---|
| `Q38_MAXT` | `8192` | Served context capacity. Values above the model's native 262,144-token limit are clamped; malformed or non-positive values restore the default. |
| `Q38_EOS` | tokenizer/config stop IDs | Override the served end-of-sequence token ID for controlled experiments. Normally the engine stops on the tokenizer's `<|im_end|>` / `<|endoftext|>` IDs, falling back to `eos_token_id`. |
| `Q38_NATIVE_FP8` | `1` (on) | Keep routed E4M3 expert bytes and their F32 128×128 block scales native in the LRU. `=0` restores expanded-FP32 slots for A/B validation. |
| `Q38_NATIVE_BF16` | `1` (on) | Keep resident and routed BF16 matrices in two-byte storage while retaining FP32 activations/accumulation. `=0` restores the expanded-FP32 reference. |
| `Q38_EXPERT_INT4` | auto | Routed experts from `<snap>/experts-int4g64/`, the int4-g64 sidecar `tools/convert_qwen38_experts_int4.py` writes next to the FP8 shards: 2.76 MB per expert instead of 4.92 MB, so the same RAM holds 1.78x the experts and a miss reads 56% of the bytes. Unset: used when its `index.json` says the conversion is complete (an incomplete one is skipped with a line on stderr; one that disagrees with the model is refused). `=0` keeps the snapshot's FP8 experts; `=1` refuses to start without a complete sidecar. The CUDA expert tier streams FP8 only and stays off with int4 experts; `coli plan` / `--auto-tier` size the cap with the sidecar's records. See [qwen38.md](qwen38.md#routed-experts-as-int4-g64). |
| `Q38_MTP` | on when the checkpoint has the head | Loads the checkpoint's MTP head and decodes speculatively (`0`: plain decoding, the head unread; `1`: required, a checkpoint without it is refused; unset, a checkpoint without it decodes plainly with a line): the head drafts the next token and one forward over two rows verifies it; a rejected draft rolls the DeltaNet/PLE state back to a copy taken after the first row. The output is that of plain decoding, greedy or sampled. A checkpoint without an MTP head is refused. Acceptance and tokens/forward go to stderr at the end of a run and after every served turn. Measured on the release with the int4-g64 sidecar: 1.94 tokens/forward and 12-14% more tokens/s, the expert reads from disk being the same with and without it. See [qwen38.md](qwen38.md#speculative-decoding-with-the-mtp-head). |
| `Q38_MTP_WIRING` | `b` | How the MTP head normalizes the model's four streams before `fc_hidden`, which the tensors leave open: `b` one norm per stream (the reading the trained weights agree with: 94-96% of drafts accepted on the release), `a` one norm over all four streams as one vector (85-92%; a diagnostic). Only the acceptance rate depends on it, never the output. |
| `Q38_MTP_CAP` | the cap | Expert cache of the MTP head's layer. Its experts are the snapshot's FP8 even beside the int4-g64 sidecar; the startup line prints what the cache costs full. |
| `Q38_MTP_DRAFTS` | `2` | With the MTP head on: the MTP head's drafts per verify, `1` to `3` (the deeper ones read the head's own streams), or `0` / `auto` to let the gate pick 0 to 3 per verify from the measured acceptance and verify cost. A verify of `k` drafts copies the recurrent state after `k` rows (113 MiB each). The default 2 measured fastest on a Ryzen 7 PRO 8700GE, on the CPU and in the Vulkan dense chain; `1` is the one-draft verify that came before. See [speculative.md](speculative.md). |
| `Q38_MTP_FORCE`, `Q38_MTP_DUMP` | unset | Tests only: `reject` rolls back every draft, `accept` / `mixed` draft the reference's next token (always, every other verify) in `ref.json` mode, `cycle` moves the wrong draft over the rows verify by verify, `row1`..`row3` makes that draft wrong; `Q38_MTP_DUMP=<file>` writes every first draft's row, token and logits. |
| `COLI_LOOKUP`, `COLI_LOOKUP_DRAFTS`, `COLI_SPEC_GATE`, `COLI_LOOKUP_FORCE` | `1`, `5`, `1`, unset | Prompt-lookup drafts, as for qwen36 below; on qwen38 combined with the MTP head (a verify carries the proposal worth more per unit of time). |
| `Q38_PREFILL_BATCH` | `1` (on) | Route prompt rows in bounded expert-major chunks and batch resident shared-expert/DeltaNet projections. `=0` restores row-at-a-time prompt execution for A/B diagnosis; decode is unchanged. |
| `Q38_TRUNK_CPU_INT8` | `1` (on) | The dense trunk (DeltaNet and attention projections, hyper-connection mixers, shared expert, router, lm_head; every matrix of at least `Q38_TRUNK_MIN_KB`) is kept on the CPU as int8 rows with one scale per row and the BF16 copy is released; `q38_weight_matmul` quantizes the activation to int8 and uses the integer kernels of `idot.h` for decode and prefill. `=0` keeps the BF16 rows and the f32 kernel (the numeric reference). See [qwen38.md](qwen38.md#the-trunk-on-the-cpu-int8-rows). |
| `Q38_DN_GPU` | `0` (off) | CUDA expert tier with the trunk in VRAM: a decode token runs every DeltaNet layer whose `dnqkv`, `dnz` and `dnout` sit on one card end to end on that card (conv, recurrence, gated norm; state resident in VRAM). See [qwen38.md](qwen38.md#the-deltanet-layer-on-the-card-q38_dn_gpu1). |
| `COLI_MAP_EXPERTS` | `0` | Point the native-FP8 routed-expert slots (or the int4-g64 sidecar's records) at a read-only mapping of their shard instead of copying 14 MB per miss into a slab. Same variable as in `glm53`. |
| `Q38_FP8_KERNEL` | vector | The routed experts' e4m3 blocks are decoded eight at a time in registers and multiplied with FMA (AVX2 builds); `scalar` restores `quant.h`'s table kernel, which differs only by float summation order inside a block. |
| `COLI_TIMERS` | `0` (off) | Set to `1` for the detailed Qwen3.8 phase breakdown on stderr. The shared per-request `PROF` frame is emitted regardless. |
| `COLI_VULKAN` | `0` | `VK=1` build: the trunk on the Vulkan device, and the routed experts of the model's layers on the shared expert tier (`COLI_VK_TIER*`, `COLI_VK_DENSE`, see [Vulkan](#vulkan-any-gpu-with-a-vulkan-12-driver)); the MTP head's layer too on a discrete GPU (`COLI_VK_TIER_MTP`). |

## DeepSeek V4 engine (`deepseek_v4`)

The V4 engine has its own knob set (~70 variables: GPU tier, prefill segments/
chunks, prefix checkpoints, expert I/O, speculative decoding, profilers). It is
documented with defaults in
[deepseek-v4.md — Environment reference](deepseek-v4.md#environment-reference-v4-engine);
the ones you are most likely to set: `DSV4_CUDA` (GPU tier on/off),
`COLI_CUDA_ATTN_BATCH=1`, `COLI_CUDA_MOE_BATCH=1`, `DSV4_CUDA_EXPERT_MIRRORS`,
`V4_MOE_REFILL_GROUP`, `V4_PREFILL_SEGMENT`, `V4_PREFIX_CKPT*`, `CTX`.
`COLI_V4_SAVE_USAGE=0` is an engine-specific alias that disables only V4's
usage rewrite; the shared `USAGE_SAVE=0` covers this engine too.

| Variable | Default | Effect |
|---|---|---|
| `COLI_VULKAN` | `0` | `make deepseek-v4 VK=1` build: the resident dense layers, head, router and compressors on the Vulkan device, and the routed experts on the shared expert tier (`COLI_VK_TIER*`, `COLI_VK_DENSE`, see [Vulkan](#vulkan-any-gpu-with-a-vulkan-12-driver)) with DeepSeek V4's own activation, which makes the CPU kernel's bf16 and E4M3 roundings on the device ([vulkan.md](vulkan.md#deepseek-v4s-activation)). The tier's history is the store's `<model>/.coli_usage`; a routing the device served counts in it as one the CPU computed. With the CUDA tier on (`DSV4_CUDA`), CUDA wins and the Vulkan tier stays off. |
| `COLI_V4_ROWS16` | `1` (on) | Repack hot-pinned experts into the vectorized `rows16` layout. **While this is on, greedy output varies run to run on the same machine** (#1136): rows16 and the reference matvec accumulate in different orders, and which experts take which kernel follows the expert-cache state. `=0` runs the reference matvec for every expert — slower, but the kernel variable is gone. **Set `=0` for any quality A/B on this engine**; throughput A/Bs do not need it. |

**Reproducible greedy runs (#1136):** greedy text on this engine varies with
the expert-cache state — hot experts run the vectorized `rows16` kernel, cold
ones run the reference matvec, the two accumulate in different orders, and
which experts are hot follows the autopin history (`.coli_usage`, rewritten by
every run). This is a known defect, not a documented trade-off — the house
rule since the olmoe/inkling IDOT cases (#1044, #1080) is that a fast path
which changes tokens is opt-in, and a convergence fix (reference path adopting
rows16's accumulation order) is planned under #1136. Until it lands: for
byte-identical output across runs, either freeze the history (`USAGE_SAVE=0`,
after seeding it once) or remove the variable entirely
(`COLI_V4_ROWS16=0 COLI_V4_AUTOPIN=0 USAGE_SAVE=0`: reference kernels only, no
history). Details in [deepseek-v4.md — CPU-only behaviour](deepseek-v4.md).

## DeepSeek V4.1 engine (`deepseek_v41`)

Read **only** by `c/deepseek_v41.c`. See [deepseek-v41.md](deepseek-v41.md).

| Variable | Default | Effect |
|---|---|---|
| `V41_ENGRAM_ROWS` | 65536 | DeepSeek V4.1: rows of engram cache per table. The n-gram traffic is Zipfian, so a small cache absorbs most of it; 65536 rows is 64 MB per table on the released head_dim. |
| `COLI_VULKAN` | `0` | DeepSeek V4.1, `VK=1` build: the trunk (vision tower included) on the Vulkan device, and the backbone's routed experts on the shared expert tier (`COLI_VK_TIER*`, `COLI_VK_DENSE`, see [Vulkan](#vulkan-any-gpu-with-a-vulkan-12-driver)); the DSpark stages' experts too on a discrete GPU (`COLI_VK_TIER_MTP`). With the tier on, the engine keeps the expert history `COLI_USAGE` (default `<snap>/.coli_usage`), saved at every run and serve turn end; it keeps none otherwise. |
| `V41_INDEX_OWNER` | unset | DeepSeek V4.1: score each layer against its OWN index keys instead of the last published cache. The default reproduces the released inference code; this changes the model's behaviour, see docs/deepseek-v41.md. |
| `V41_MAX_IMAGE_TOKENS` | the checkpoint's `max_image_tokens` | DeepSeek V4.1: ceiling on what one image costs in prompt tokens. |
| `V41_TRACE` | unset | DeepSeek V4.1: print per-sublayer checksums, matching tools/dsv41_ref.py's, to locate a divergence by diffing two columns. `2` follows the first row of a speculative step rather than the last. |
| `V41_DSPARK` | on when the checkpoint carries the head | DeepSeek V4.1: `0` disables the DSpark draft head, which is then not loaded. Drafts never change what a turn produces, only how many forwards it takes: measured +17% on the real checkpoint from a cold cache (24 tokens in 99.3 s against 116.6). |
| `V41_DSPARK_MAX` | the checkpoint's `dspark_block_size` | DeepSeek V4.1: how many drafted tokens go in front of the main model per round. Fewer costs less when a round is rejected and caps the win when it is not. |
| `V41_DSPARK_MINACC` | 60 | DeepSeek V4.1: percent of drafts that must be accepted over a window of ten before drafting pauses for 64 tokens. 60 is the measured break-even. |
| `V41_SPEC_FORCE` | unset | DeepSeek V4.1, oracle mode only: draft the reference's own tokens (`1`), corrupt the last one (`2`), keep the head's (`3`), corrupt the first one (`4`) or a different one each round (`5`), so the verification path runs on a fixture whose draft head is random noise. |

## MiMo-V2.6 engine (`mimo`)

Read **only** by `c/mimo.c` (and `MIMO_MAX_IMAGE_TOKENS` by the gateway). See [mimo.md](mimo.md).

| Variable | Default | Effect |
|---|---|---|
| `MIMO_DENSE_BITS` | 0 | MiMo: 0 keeps the dense weights as released (FP8, BF16), exact; 8 is int8 per row, less RAM, not exact; 32 is f32, the oracle's configuration. |
| `MIMO_IDOT` | 0 | MiMo: 1 quantizes the expert matmuls' activations to int8 per 32 (faster, not exact). |
| `MIMO_DIRECT` | 1 | MiMo: 0 reads experts through the page cache instead of `O_DIRECT`. |
| `MIMO_READ_THREADS` | 8 | MiMo: parallel expert reads per layer. |
| `MIMO_CHUNK` | 64 | MiMo: prompt tokens per prefill block. |
| `MIMO_CTX` | `CTX`, else 8192 | MiMo: context the KV cache is sized for, capped by the checkpoint. |
| `MIMO_CAP` | 64 | MiMo: expert cache slots per layer when no argument gives one; never below one routing step. |
| `MIMO_MAX_IMAGE_TOKENS` | unset | MiMo, gateway: ceiling on what one picture costs in prompt tokens. |
| `MIMO_LOGITS` | unset | MiMo, oracle: dump every prompt position's logits (f32) to this file. |
| `MIMO_TRACE` | unset | MiMo, oracle: dump the residual after every sublayer of the first block. |
| `MIMO_DIRS` | unset | MiMo: extra directories holding shards. |
| `MIMO_STATS` | unset | MiMo: report the vision tower's time per picture. |
| `COLI_VULKAN` | `0` | MiMo, `VK=1` build: the routed experts on the shared expert tier (`COLI_VK_TIER*`, see [Vulkan](#vulkan-any-gpu-with-a-vulkan-12-driver)) as the release's MXFP4, and the dense matrices of the trunk and of the vision tower on the device where `COLI_VK_DENSE` puts them, in their `MIMO_DENSE_BITS` form (FP8 and BF16, int8 or f32). The router stays on the CPU. MiMo keeps no expert history: the tier fills as experts pass by. |
| `MIMO_VK_EXPERTS` | unset | MiMo, with `COLI_VULKAN=1`: the switch of the engine's own tier before the shared one, kept as an alias. `0` keeps the routed experts on the CPU (no tier); `N` sizes the shared tier's budget at N experts (`COLI_VK_TIER_GB`, which wins when set); the tier then evicts as the routing moves, where the old one stopped at N and never let go. Unset: the shared tier with its own budget. |

## OLMoE engine (`olmoe`)

Read **only** by `c/olmoe.c`. This is the sister engine used for streaming-cache research, so most of these are experiment knobs.

| Variable | Default | Effect |
|---|---|---|
| `CHAT` | unset | Interactive chat mode; bypasses the `ref.json` harness entirely. |
| `MAX_NEW` | `512` | Max tokens to generate in chat mode. |
| `HOT` | `0` | Number of hottest experts to pin at startup. |
| `WARMUP` | `5` | Tokens observed before the hot set is considered learned. |
| `WIDE` | `1` | Router width multiplier for the prefetch prediction. Clamped to [1,4]. |
| `SMOOTH` | `0.3` | EMA factor for routing momentum (gate logits smoothed across tokens). Clamped to [0, 0.95]. |
| `CONF_LIMIT` | `0.92` | Confidence ceiling for the router prediction. Clamped to [0.1, 1.0]. |
| `EXPERT_DROP` | `0` (off) | Drop experts below the confidence threshold instead of loading them (quality/speed experiment). |
| `COLI_VULKAN` | `0` | `VK=1` build: attention, router and lm_head on the Vulkan device, and the routed experts (int8 rows) on the shared expert tier (`COLI_VK_TIER*`, `COLI_VK_DENSE`, see [Vulkan](#vulkan-any-gpu-with-a-vulkan-12-driver)). The warm start reads the history `COLI_USAGE` names; without it the tier fills as experts pass by. |

## Laya engine (`laya`)

Read **only** by `c/laya.c`, the decision engine ([laya.md](laya.md)).

| Variable | Default | Effect |
|---|---|---|
| `COLI_LAYA_MAX_LEN` | the checkpoint's `max_len` | Tokens per question sequence (512 on the English checkpoint, capped at the encoder's 8192 positions). |
| `COLI_LAYA_HEAD_MAX_LEN` | the checkpoint's `head_max_len` | Tokens shared by a question's instructions and options; raise it for questions with many options. |

With a `VK=1` build, `COLI_VULKAN=1` runs the forward on a Vulkan device (`COLI_VK_CHAIN`, `COLI_VK_DENSE` above; [vulkan.md](vulkan.md#the-decision-engines)).

## GLiNER2.5-Decide engine (`gliner_decide`)

Read **only** by `c/gliner_decide.c`, the decision engine ([gliner_decide.md](gliner_decide.md)).

| Variable | Default | Effect |
|---|---|---|
| `COLI_GLINER_MAX_LEN` | `4096` | Tokens of the one sequence a request becomes: every question with its options, then the state. The state is cut at the last whole word that fits; questions that alone exceed it are refused with a 422. |

With a `VK=1` build, `COLI_VULKAN=1` runs the forward on a Vulkan device (`COLI_VK_CHAIN`, `COLI_VK_DENSE` above; [vulkan.md](vulkan.md#the-decision-engines)).

---

## Server / CLI (`openai_server.py`, `coli`)

These are read by the Python programs (not the `glm` engine), so they don't appear in `glm.c`. They cover the OpenAI-compatible server, tool calling, and the debug view.

| Variable | Default | Effect |
|---|---|---|
| `COLI_DEBUG` | `0` (off) | Tee the engine transaction to stderr, by level. **`1`** = decoded model output stream only (byte-by-byte, on both the tool-call and plain paths). **`2`** = both sides — the fully-rendered prompt the engine received *and* the output, bracketed and correlated by request id, so stderr reads as the whole conversation. Invaluable for seeing what the model received vs. emitted during an OpenCode session. |
| `COLI_TOOL_SALVAGE` | `0` (off) | Opt-in de-mangler: reconstruct a malformed int4 tool call by mapping its lone payload onto the tool's primary parameter. Never rewrites well-formed output; recommended for int4 deployments. |
| `COLI_THINK` | `0` (off) | Make thinking the default when the client sends *neither* `reasoning_effort` nor `enable_thinking`. Any explicit client value still wins. |
| `COLI_CONTINUE_ASSISTANT` | `1` (on) | On the OpenAI- and Anthropic-compatible chat endpoints, continue a trailing `assistant` message — render its turn open and resume from it, dropping the turn terminator and the generation cue — instead of opening a new turn, the same contract as Anthropic's API. On by default: a message list ending in a non-empty `assistant` turn continues. Set `0` to restore the old behavior (append a fresh generation cue). Refused with `tools`/`tool_calls`, and the turn must carry text not ending in whitespace. Every shipped family supports it, Kimi K3 included (its open turn is framed engine-side in `kimi_k3.c`). Unrelated to `COLI_PREFILL_CHUNK`, which is the compute phase. |
| `COLI_MODEL` | unset | Default model directory (fallback for `--model`). |
| `COLI_MODEL_ID` | `glm-5.2-colibri` | Model id reported by the API. |
| `COLI_API_KEY` | unset | Required bearer token for the server. |
| `COLI_IMAGE_ROOT` | unset (local paths denied) | Directory under which an `image_url.url` naming a local path or `file://` URI may be read. Unset, the server refuses local paths: a client sends images as base64 `data:` URIs (`coli chat` and `coli web` do), because a file read here happens with the server's own rights and an inference client is not the operator. Set it to allow paths under one directory only; symlinks are resolved before the check. |
| `COLI_ALLOWED_HOSTS` | unset | Comma-separated hostnames or IP addresses accepted by the DNS-rebinding guard in addition to loopback and the bind address. Equivalent to repeating `--allowed-host`. |
| `COLI_MAX_QUEUE` | `8` | Max queued requests. |
| `COLI_QUEUE_TIMEOUT` | `300` | Seconds a request may wait in the queue. |
| `COLI_KV_SLOTS` | `1` | Independent KV conversation slots (→ engine `KV_SLOTS`). |
| `COLI_POLICY` | `quality` | Resource policy (shared with the engine): `quality` \| `balanced` \| `experimental-fast`. |
| `COLI_CHAT_STATS` | `full` | Default for `coli chat --stats`: the footer after each answer. `full` = tokens, seconds, tok/s; `compact` = tokens, tok/s; `off` = no footer. Counts are exact (no `~`) when the server reports `completion_tokens` in the streamed usage block, the chars/4 estimate otherwise. The flag wins over the variable. |
| `COLI_COLOR` | auto (TTY) | `COLI_COLOR=1` forces colored `coli` output when not a TTY. |
| `COLI_RAW` | `0` | `coli` raw output mode. |

> **Debugging an OpenCode session:** `COLI_DEBUG=1` watches the model's output stream; `COLI_DEBUG=2` shows both sides (prompt + output) as a transcript. Add `COLI_TOOL_SALVAGE=1` on int4 to catch mangled tool calls.

## Set by the CLI (don't usually set by hand)

`coli` / `openai_server.py` set these internally to select a run mode or pass through a flag:

- `SNAP` — model snapshot directory (required by `glm`; set from `--model`).
- `SERVE`, `SERVE_BATCH` — select serve / batched-serve mode.
- `PROMPT` — one-shot text mode (the engine also honors `COLI_PROMPT`, preferred cross-platform; `PROMPT` is ignored on Windows if it contains cmd.exe `$`-metacharacters).
- `COLI_OMP_TUNED` — internal sentinel guarding the OMP re-exec (see `COLI_NO_OMP_TUNE`); not user-facing.

---

## Worked example — the fast, reproducible Apple-Silicon config

```bash
# fast (sampling, non-deterministic by design):
COLI_METAL=1 DIRECT=1 COLI_NO_OMP_TUNE=1 PIPE=1 PIPE_WORKERS=6 MTP=0 \
  ./coli run --model /path/to/model --ram 113 "your prompt"

# same, but reproducible (greedy):
COLI_TEMP=0 COLI_METAL=1 DIRECT=1 COLI_NO_OMP_TUNE=1 PIPE=1 PIPE_WORKERS=6 MTP=0 \
  ./coli run --model /path/to/model --ram 113 "your prompt"
```
