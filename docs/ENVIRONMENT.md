# Environment Variables

Reference for the environment variables read by the colibrì engine.

**Generated from `dev @ 7fb1159`** by scanning every `getenv()` / `getenv_utf8()` site in `c/*.c`, `c/*.h`, `c/*.cu` and `c/*.mm`. Defaults and behavior are taken from the source; see [MAINTAINING-DOCS.md](MAINTAINING-DOCS.md) to regenerate this after the code changes.

## Which program reads these?

**There are five engine binaries, and they do not share a knob set.** The main
engine `c/colibri` (built from `c/colibri.c`, formerly `glm.c`) reads most of
what follows, but the sister engines read their own:

| Engine | Source | Its own variables |
|---|---|---|
| `colibri` | `c/colibri.c` | everything below except the three sections named for another engine |
| `kimi_k3` | `c/kimi_k3.c` | the `K3_*` family — see [Kimi K3 engine](#kimi-k3-engine-kimi_k3) |
| `inkling` | `c/inkling.c` | `INK_*`, plus `CTX_MAX`, `PIN_N`, `REP_PEN`, `GPU_DEV`, `NOGPU` — see [Inkling engine](#inkling-engine-inkling) |
| `olmoe` | `c/olmoe.c` | `HOT`, `WIDE`, `SMOOTH`, `CONF_LIMIT`, `MAX_NEW`, `CHAT`, `EXPERT_DROP`, `WARMUP` — see [OLMoE engine](#olmoe-engine-olmoe) |
| `deepseek_v4` | `c/deepseek_v4.c` | `CTX`, the `V4_*` / `DSV4_*` families and the two `COLI_CUDA_*_BATCH` gates — see [DeepSeek V4 engine](#deepseek-v4-engine-deepseek_v4); note that the CUDA section below describes `colibri.c` knobs (`COLI_CUDA`, `CUDA_DENSE`, ...) which the V4 engine does not read — its GPU switch is `DSV4_CUDA` |

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
| `RAM_GB` | `0` (auto ≈ 88% of free RAM) | RAM budget in GB for the resident/streamed expert working set. Higher → more experts stay hot → higher cache hit rate. |
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
| `COLI_GROUP_ASYNC` | `0` (off) | `=1` issues and collects CUDA expert groups asynchronously so CPU and GPU overlap at decode (S≤4). |
| `COLI_DISKCLASS_WINDOW` | see source | Recency window (in ticks) for the DISK-CLASS heat statistic. |
| `URING` | `0` (off) | Linux-only queued expert I/O. `URING=1` implies `PIPE=1`, forces cold reads through io-wq (`IOSQE_ASYNC`), replaces blocking loader pthreads and spin waits with batched SQEs/CQEs, and batches `PILOT_REAL` loads on a separate ring. Use `DIRECT=1` for cold NVMe to avoid page-cache copy/readahead limits. Fails clearly if the kernel denies io_uring; incompatible with `COLI_MMAP=1`. |
| `DIRECT` | `0` (off) | Use `O_DIRECT`/unbuffered reads for expert slabs. **Drive-dependent — measure it on your hardware.** On real NVMe with DRAM cache and headroom it is often a large win (measured +34% decode with `PIPE=1` on a Blackwell/Windows box, and 4.25→9.69 GB/s in iobench on a GB10); on QLC/DRAM-less drives or slow/virtualised disks it can be neutral to negative. Helps sustained NVMe; keeps the zero-copy GPU path. |
| `COLI_NO_OMP_TUNE` | off | **Kill-switch** for the OpenMP hot-thread tuning (`OMP_WAIT_POLICY=active` spin + proc-bind). Set `=1` when the CPU is mostly waiting on the GPU (Metal) so spin doesn't steal the shared power budget. |
| `COLI_NUMA` | auto in generated plans on multi-socket Linux; otherwise off | `COLI_NUMA=1` selectively interleaves large expert and dense slabs across NUMA nodes via `mbind` (raw syscall, no libnuma). Helps multi-socket hosts (+7–40% expert matmul); silent no-op on single-node or non-Linux. Explicit `COLI_NUMA=0` overrides the generated plan. |
| `MLOCK` | `-1` (auto: on for macOS) | Wire the streamed expert cache into physical RAM (`mlock`) to dodge the memory compressor. `0` off, `1` force. |
| `CAP` | unset | Expert-cache cap (slots/layer) when no CLI positional was given. Precedence: explicit `--cap`/positional > `CAP` > platform default > historic default (#379). Mainly for direct `./glm` use — `coli` users should prefer `--cap`. |
| `CAP_RAISE` | `1` (on); `0` on Metal + macOS + fast model volume (#379) | Let the engine raise the expert-cache cap above `topk` when RAM allows (bigger batches). `0` fixes the cap. When the platform-aware Metal cache default engages (F_NOCACHE probe measured the model volume fast), the *default* flips to `0` — auto-raise re-creates the Metal residency churn the minimal cache avoids. An explicit `CAP_RAISE` always wins. |
| `COLI_SSD_FAST_GBS` | `4.0` | Threshold (GB/s, measured F_NOCACHE, cached in `<model>/.coli_ssd` — see [The `.coli_ssd` probe cache](#the-coli_ssd-probe-cache) below) at or above which the model volume counts as "fast" for the platform-aware Metal cache defaults (#379). |
| `PREFETCH` | `0` | Prefetch depth for streamed experts. |
| `COLI_MMAP` | `0` | `mmap` the weights instead of read()-ing into slabs. |
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
| `CACHE_ROUTE` | `0` (off) | Opt-in max-rank cache-aware MoE routing (pin∪LRU prefer within top-M). See [CACHE_ROUTE.md](CACHE_ROUTE.md). |
| `ROUTE_J` | `2` | Sacred top ranks always taken when `CACHE_ROUTE=1`. |
| `ROUTE_M` | `12` | Max-rank window for resident preference when `CACHE_ROUTE=1`. |
| `ROUTE_P` | `0` | Cumulative mass window for CACHE_ROUTE (`0` = fixed M). |
| `ROUTE_ALPHA` | `1` | Scale gate mass of substituted experts before renorm (`1` = off). |
| `ROUTE_AGREE` | auto | Overlap% + KL vs true top-K; auto-on when `CACHE_ROUTE=1`. |
| `ROUTE_TRACE` | unset | If set to a path, logs every routing decision there (testing/analysis). |
| `ABSORB` | `-1` (auto: absorbed for S≤4) | MLA attention absorption mode. |
| `IDOT` | `1` | Integer dot-product kernel. `IDOT=0` uses exact f32 kernels (for A/B numerical checks). |
| `COLI_POLICY` | `quality` | Resource policy: `quality`, `balanced`, or `experimental-fast`. |
| `PROF` | `0` (off) | Performance profile: a startup header (machine + effective config), then per run — or per turn in serve mode, on stderr — forward-latency percentiles (p50/p90/p99/max), expert-I/O totals and cache-tier fill, phase shares of wall time, and a verdict naming the knob most likely to help on this machine. Output is additive; `PROF` unset changes nothing. |
| `COLI_NO_FUSED_PAIR` | `0` (off) | `=1` disables the fused-pair matmul kernel. |
| `DISK_SPLIT` | `0` (off) | `=1` splits the reported disk-load time across the draft/absorb/forward phases in stats. |
| `I4S` | per-ISA (`1` on AVX-512-VNNI / NEON-dotprod, `2` elsewhere) | Engage the int4 `IDOT` kernel for batch `S>=<n>`. `I4S=1` turns IDOT on at decode too: int8-quantized activations on expert matmuls — **not bit-identical** to the f32 decode path (measured 0.39% of scale on the gate output; the same numerics prefill already uses at `S>=2`, and the shipped default on AVX-512-VNNI, measured +5.5% end-to-end there). Attention projections always stay exact regardless. A default flip on AVX-VNNI awaits the quality ablation. |
| `IDOT_GS` | `0` (off) | **Opt-in** grouped planar IDOT for `fmt=4` (gs64/gs128) tensors: int8 activations with the K1 plane layout, one integer dot per scale group. Same numerics family as `I4S=1` — not bit-identical to the f32 grouped kernel, hence off until the ablation. Requires the planar family (AVX2 build, no GPU backend, no `XEXP`). Activation prints `[K1b]` once. |
| `SPEC_PIN` | `1` (on) | Speculation gate mode. `0` reverts to the legacy S-dependent speculation gates (#163). |
| `COLI_RAM_OVERCOMMIT` | off | `=1` overrides the "projected peak > MemAvailable → exit(2)" guard so a run that risks kernel OOM-kill is allowed to proceed. |

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
| `COLI_VULKAN` | off | Enable the Vulkan backend. Requires a `make VK=1` build; fails at startup (no silent fallback) if libvulkan or the compiled shaders are missing. |
| `COLI_VK_DEV` | unset | Select the primary Vulkan physical-device enumeration index. Without it, the backend prefers a discrete GPU, then integrated/virtual devices. |
| `COLI_VK_SHADERS` | auto | Path to the compiled `qmatmul.spv` **or** the directory holding the `.spv` set; the other shaders are found next to it. Unset: `shaders/` next to the binary, then CWD-relative `shaders/qmatmul.spv`. |
| `COLI_VK_EXPERTS` | `320` | Pinned VRAM expert tier size: top-N experts by `.coli_usage` heat uploaded once at startup and served from VRAM with no RAM slot or disk read. `0` disables the tier (experts stay on the CPU path). ~19 MB VRAM per int4 expert. |
| `COLI_VK_DENSE` | `0` | Run the resident dense matmuls (attention projections, shared expert) on the GPU. |
| `COLI_VK_ATTN` | `0` | Run the S≤4 MLA absorb attention core (+ fused o-projection) on the GPU, with a persistent device-side KV mirror. |
| `COLI_VK_QPREP` | `1` (on) | Fuse the Q-prep step (RMSNorm + rope + compress) into one GPU dispatch instead of splitting it, which cost three fences where one suffices. `0` restores the split path; `2` additionally keeps CPU reference copies of Q and comp for A/B comparison. |
| `COLI_VK_RESERVE_GB` | `3.0` | VRAM (GB) held back from the expert tier for the lazily-allocated dense weights, KV mirror and staging buffers (measured ~1.7 GB at 4k ctx, growing with `max_t`). Only meaningful when the driver reports `VK_EXT_memory_budget`; without it the `COLI_VK_EXPERTS` count cap applies alone. |
| `COLI_VK_SPIN_US` | `300` | Microseconds to spin-poll a fence before blocking. `0` always blocks — lower latency at idle, at the cost of a core spinning. |

### Second Vulkan device (opt-in)

A second GPU can hold the *next* heat-ranked experts after dev0's budget stops. Deliberately separate from the first device so the dev0 hot path is untouched and both groups can be in flight at once.

| Variable | Default | Effect |
|---|---|---|
| `COLI_VK_DEV2` | unset (off) | Enable the second device tier. A number selects that physical device index; `auto` picks a distinct real GPU (a second *logical* device on the same physical GPU is accepted only when forced by index — that is the pre-hardware test mode). |
| `COLI_VK_EXPERTS2` | `512` | Expert count cap for the dev2 tier (only read when `COLI_VK_DEV2` brought a device up). |
| `COLI_VK_RESERVE2_GB` | `0.5` | VRAM (GB) held back on dev2, as `COLI_VK_RESERVE_GB` is for dev0. |

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
| `COLI_DRAFT_CORPUS` | unset | Path to a file of frozen token ids (whitespace-separated, `-1` separates spans) used as a speculative draft source: the engine proposes the continuation that followed the longest suffix of the live context found in the corpus. Off when unset. Build one from any run with `TOKENS=1`. See [corpus-draft.md](corpus-draft.md). |
| `COLI_CORPUS_K` | `8` (max 48) | Proposal depth for `COLI_DRAFT_CORPUS`. Deeper raises the forward multiplier and the per-forward cost. |
| `COLI_CORPUS_MINACC` | `50` | Acceptance floor (percent) for the corpus source. Below it over a 24-proposal window the source pauses for 256 tokens, then re-arms — rejected drafts cost real time. |
| `EXPERT_BUDGET` | `0` (off) | Cap experts loaded per layer (MoE-Spec). **Quarantined:** silently forced to `0` unless `EXPERT_BUDGET_EXPERIMENTAL` is set — every tested value is either no faster or incoherent (issue #303). |
| `EXPERT_BUDGET_EXPERIMENTAL` | unset | Setting it (any value) allows `EXPERT_BUDGET>0` to actually take effect (expect garbage, #294). |
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
| `CHAT_TEMPLATE` | `1` | Apply the GLM chat template (`0` = raw prompt). |
| `PPL` | off (`olmoe.c` only) | `PPL=1` enters teacher-forced NLL/perplexity meter mode in the OLMoE sister engine. |
| `ABLATE_SCORE` | unset | Causal-ablation sweep over `ABLATE_SCORE=<file>`, with a per-target-position final-logit read-out. Runs before `SCORE` and exits when done. |
| `ABLATE_OUT` | unset | Where the ablation sweep writes its logit read-out. Pair with `ABLATE_SCORE`; an optional `ROUTE_TRACE` records the post-ablation router trace. |
| `DEBUG_LOGITS` | unset | In reference-comparison mode, dump per-position logit diagnostics. |
| `COLI_LOGIT_DUMP` | unset | `=1` prints the top-5 `id:logit` pairs per step to stderr — for comparing two engine configs on identical forced context (backend-exactness triage). |
| `I3_AVX512` | auto | Force the AVX-512 int3 kernel on (`1`) or off (`0`). |
| `I3_AVX512_TEST` | unset | Run the AVX-512 int3 self-test and exit. |
| `COLI_GPU_FAIL_AFTER` | unset | Fault injection: make GPU compute calls start failing after N of them, to exercise the CPU fallback without real hardware faults. Uploads and queries are not gated. |
| `COLI_VK_TEST_BALLAST` | `0` | Allocate N extra dummy Vulkan buffers to reproduce decode attention degrading with expert-tier size even when VRAM is free (measured 7.9s @2.6k buffer objects → 15.6s @4.3k with 2.9 GB still free). |
| `COLI_SERVE_ALL_STOPS` | unset | In batched serve mode, keep every stop token instead of filtering to the EOS-like ones. Trades the #401 tool-call safety for behaviour some non-tool clients prefer. |
| `VK_PROF` | unset | If set, time the Vulkan expert-group path and report it. |
| `COLI_USAGE` | `<model>/.coli_usage` | Path to the expert-usage history to seed the ranking from, and to write back to. Shared by every engine (`route_trace.h`). |
| `COLI_USAGE_DECAY` | `1.0` (no decay) | Per-run multiplier applied to the recorded counts before ranking, i.e. a half-life. Without one the ranking freezes: after ~18M recorded selections one more turn moves it by 0.2% and the profile stops following the workload (#780). Values outside `(0,1]` are ignored. |
| `USAGE_SAVE` | `1` (on) | `=0` runs read-only — the usage history is loaded but never written back. For benchmark loops that would otherwise skew the profile they are measuring. |
| `RANS_PATH` | auto (best available) | Force a specific rANS kernel (`scalar`, `neon`, `avx512`, …). An unavailable choice yields `invalid` and fails loudly — never a silent downgrade. |
| `RANS_NEON` | on where built | `=0` kill-switch for the NEON rANS path. |
| `RANS_AVX512` | on where built | `=0` kill-switch for the AVX-512 rANS path. |
| `OMP_NUM_THREADS` | unset | Standard OpenMP variable. Setting it disables the engine's own OpenMP hot-thread tuning entirely — the user is assumed to be in charge. |

---

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
| `K3_VK` | `1` (on where built) | Vulkan expert tier. `=0` forces CPU-only. |
| `K3_VK_GB` | `0` (driver budget) | VRAM cap (GB) for the K3 Vulkan expert tier. |
| `K3_VK_UP` | `8` | Expert uploads allowed per step while filling the VRAM tier. |
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
| `INK_METAL_MIN_S` | `1` | Minimum batch S to send the MoE block to Metal. `=2` restores the prefill-only gate (which mattered when the residency set was absent and per-block `useResource` churn cost ~135 ms). |
| `INK_PREFIX_LOG` | unset | Log the KV-prefix reuse decision and its reason, as `K3_PREFIX_LOG` does for K3. |
| `GPU_DEV` | `0` | CUDA device index for the inkling CUDA backend. |
| `NOGPU` | unset | If set, skip GPU init entirely (both CUDA and Metal), regardless of the other GPU variables. |

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

---

## Server / CLI (`openai_server.py`, `coli`)

These are read by the Python programs (not the `glm` engine), so they don't appear in `glm.c`. They cover the OpenAI-compatible server, tool calling, and the debug view.

| Variable | Default | Effect |
|---|---|---|
| `COLI_DEBUG` | `0` (off) | Tee the engine transaction to stderr, by level. **`1`** = decoded model output stream only (byte-by-byte, on both the tool-call and plain paths). **`2`** = both sides — the fully-rendered prompt the engine received *and* the output, bracketed and correlated by request id, so stderr reads as the whole conversation. Invaluable for seeing what the model received vs. emitted during an OpenCode session. |
| `COLI_TOOL_SALVAGE` | `0` (off) | Opt-in de-mangler: reconstruct a malformed int4 tool call by mapping its lone payload onto the tool's primary parameter. Never rewrites well-formed output; recommended for int4 deployments. |
| `COLI_THINK` | `0` (off) | Make thinking the default when the client sends *neither* `reasoning_effort` nor `enable_thinking`. Any explicit client value still wins. |
| `COLI_MODEL` | unset | Default model directory (fallback for `--model`). |
| `COLI_MODEL_ID` | `glm-5.2-colibri` | Model id reported by the API. |
| `COLI_API_KEY` | unset | Required bearer token for the server. |
| `COLI_ALLOWED_HOSTS` | unset | Comma-separated hostnames or IP addresses accepted by the DNS-rebinding guard in addition to loopback and the bind address. Equivalent to repeating `--allowed-host`. |
| `COLI_MAX_QUEUE` | `8` | Max queued requests. |
| `COLI_QUEUE_TIMEOUT` | `300` | Seconds a request may wait in the queue. |
| `COLI_KV_SLOTS` | `1` | Independent KV conversation slots (→ engine `KV_SLOTS`). |
| `COLI_POLICY` | `quality` | Resource policy (shared with the engine): `quality` \| `balanced` \| `experimental-fast`. |
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
