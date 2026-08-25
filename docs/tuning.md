# Tuning & runtime knobs

Everything here is opt-in; the defaults are chosen so a plain `./coli chat`
is safe on any machine. See also [SETTINGS.md](SETTINGS.md) and
[ENVIRONMENT.md](ENVIRONMENT.md) for the full variable inventory.

## The knobs that matter most

| knob | what it does |
|---|---|
| `--temp T` | token sampling temperature (default 0.7 + nucleus 0.90 — tuned for int4; 0 = greedy) |
| `--topp 0.7` | adaptive expert top-p (30–40% less disk; lossy — prints a warning) |
| `--ngen N` | max tokens per answer (`:more` in chat continues a truncated one) |
| `--repin N` | adapt RAM/VRAM hot experts every N emitted tokens |
| `RAM_GB=<n>` | claim more RAM for the expert cache than the conservative auto-detect |
| `PIN=stats PIN_GB=g` | pin the hottest experts from a measured usage profile |
| `DRAFT=n` | MTP draft depth (0 disables speculation) |
| `GRAMMAR=g.gbnf` | grammar-forced drafts for constrained JSON/NDJSON output ([docs](grammar-draft.md)) |
| `THINK=1` | enable GLM-5.2's reasoning block |
| `PILOT=1` | router-lookahead disk prefetch (see below) |
| `URING=1` | Linux-only batched expert I/O (implies `PIPE=1`) |
| `PIPE=0` | disable the async expert-load pool (default ON — overlaps `pread` with matmul, −18% disk service) |
| `DIRECT=1` | O_DIRECT expert reads (measured **+65%** alone on a Strix Halo, [#200](https://github.com/JustVugg/colibri/issues/200)) |
| `COLI_NUMA=1` | interleave resident weights across NUMA nodes on multi-socket hosts ([#82](https://github.com/JustVugg/colibri/issues/82)) |
| `CACHE_ROUTE=1` | cache-aware max-rank routing (opt-in, [#199](https://github.com/JustVugg/colibri/issues/199)) |
| `AUTOPIN=0` | disable the learning cache's auto-pin |
| `CAP_RAISE=0` | don't auto-grow the expert cache |
| `KVSAVE=0` | disable KV-cache persistence |
| `TF=1` | teacher-forcing validation |

Automatic history pinning and the adaptive LRU share the same expert RAM
budget. Colibri caps automatic pinning to preserve the no-pin LRU capacity;
explicit `PIN` and `PIN_GB` settings remain authoritative.

## Resource policy

`coli plan` reports the planned hot (VRAM), warm (RAM), and cold backing (disk)
tiers, the reason for each placement, and the expected bottleneck. The default
`--policy quality` and `--policy balanced` modes preserve checkpoint quantization
and router decisions unless `--topk` or `--topp` is passed; those explicit lossy
overrides print a warning and proceed.

Auto-tier plans size OpenMP from physical cores and bind workers across cores.
Memory-bound quantized kernels can regress sharply when SMT siblings compete for
limited memory channels. The GLM, Kimi K3, and OLMoE engines also apply that
physical-core cap when launched directly; explicit `OMP_NUM_THREADS` and the
`COLI_NO_OMP_TUNE` kill switch always take precedence.

> Note (#471): exporting `OMP_PROC_BIND`/`OMP_PLACES` used to interact badly with
> the engine's one-time OpenMP tuning re-exec on Linux — the re-exec'd image
> inherited the first image's place-0 thread binding and the whole team landed on
> one core (~20× slowdown). The engine now resets its affinity to all online CPUs
> right before the re-exec, so explicit `OMP_*` pinning works as documented.
> `COLI_OMP_TUNED=1` remains the escape hatch that skips the re-exec entirely.

```bash
coli plan --model /models/glm52_i4 --policy quality
coli run --auto-tier --policy quality "Explain MoE offloading"
# Explicit research-only router reduction:
coli run --policy experimental-fast --topk 4 "Benchmark prompt"
```

## Measured machine profiles

`coli plan` chooses a safe starting point from capacity and topology.
`coli tune` measures the remaining scheduling choices on the actual model and
machine, then saves a hardware/model/engine-specific profile:

```bash
coli tune --model /models/glm52_i4
coli run --model /models/glm52_i4 --auto-tier "Explain MoE offloading"
```

GLM generates one calibration continuation and teacher-forces it for every
candidate. The sibling engines use their common serving protocol instead: one
engine stays alive for all requests of a candidate, and deterministic prompts
rotate in a fixed order. That preserves the expert cache and measures a real
mixed chat instead of repeatedly loading one process or teaching the cache one
exact prompt. Greedy output is compared per prompt across every candidate; any
byte drift disqualifies the scheduling change.

The bounded sweep includes execution knobs such as OpenMP thread count, NUMA
placement, I/O overlap, direct I/O, CUDA pipelining, and DeepSeek V4 expert
loader lanes. When cold experts remain on disk it also tests 75% and 50% of
the planner's expert-cache allowance. Dense weights, KV/workspace reservations,
and the OS reserve are never reduced, and no candidate may exceed the planner's
safe baseline. GLM/Inkling/OLMoE/Qwen receive the measured slots-per-layer cap;
V4 receives a whole-process `RAM_GB` ceiling; Kimi receives `K3_EXPERT_GB` under
its unchanged whole-process ceiling. The sweep never changes weights,
quantization, router decisions, `TOPK`, `TOPP`, or sampling. Add repeatable
`--rotate-prompt` options to replace the secondary built-in workload prompt.

A candidate is saved only when median throughput improves by at least 3% while
expert hit rate remains within 0.5 percentage points and reported TTFT/p99
latency stay within 20% of the baseline. The winner is then rerun before a final
baseline; this reverse-order gate gives the baseline any remaining warm-cache
advantage and rejects startup drift. Otherwise the baseline is recorded and no
override is applied. Saved profiles are loaded by `--auto-tier`; `--ram`,
`--cap`, `RAM_GB`, `K3_EXPERT_GB`, and other explicit settings always win. A
resource winner is rechecked against the current plan at every launch, so a
profile measured with more free RAM is ignored rather than overcommitted later.
Use `--no-tune-profile` to bypass a saved profile.

Profiles live under `$XDG_CONFIG_HOME/colibri/tuning` (normally
`~/.config/colibri/tuning`) or `%LOCALAPPDATA%\colibri\tuning` on Windows. A
change to the engine binary, model metadata, CPU topology, or GPU inventory
produces a new fingerprint instead of reusing stale measurements.

Disk is an immutable recovery source, not a normal decode target. If the plan
leaves cold expert bytes on disk, speed depends on cache hit rate; output quality
does not.

Cold expert reads can use a deferred pipeline: resident RAM/VRAM experts execute
while missing experts are loaded in a bounded background I/O pool, then the cold
results join before the layer completes. The pool engages only under `PIPE=1`;
`PIPE_WORKERS=n` sets its worker count (default 8). Profiling reports both disk
service time and the smaller foreground-visible wait time so overlap is explicit.

`--policy balanced` enables lossless live placement (`REPIN=64`). At safe request
boundaries, a per-layer LFRU score combines decaying session frequency with recent
access and replaces at most four sufficiently colder pinned experts. `--policy
quality` leaves live replacement off by default; `REPIN=0` always disables it.

## The learning cache

The engine records which experts your usage actually routes to (`.coli_usage`
next to the model, updated every turn) and at startup automatically pins the
hottest ones in spare RAM — colibrì literally gets faster the more you use it.
`PIN=auto` seeds the pin directly from the live usage history
([#301](https://github.com/JustVugg/colibri/pull/301)).

**The expert cache auto-sizes to your RAM** (since 2026-07-10): the engine
*raises* the LRU cap to fill your `--ram` budget instead of only lowering it.
If you benchmarked colibrì before that date, rerun — your numbers were capped.

**Live tier adaptation** (`--repin N`, opt-in): at safe turn boundaries, a
decaying session heat map replaces cold pinned experts with hotter streamed
experts. A 25% hysteresis and a four-swap limit prevent tier thrashing.
Persistent `.coli_usage` remains the long-term signal and is not decayed.

The history's on-disk format, what happens when one engine is handed another
engine's history, and how `PIN=<file>` differs from `PIN=auto` in how much it
trusts a file are documented in
[routing-telemetry.md](routing-telemetry.md).

## Router-lookahead prefetch (`PILOT=1`, experimental)

GLM-5.2's expert routing is measurably predictable *ahead of time* — applying
layer L+1's router to layer L's post-attention state recalls **71.6%** of the
true top-8 (vs 41.3% for "same experts as last token"). `PILOT=1` issues
next-layer expert readahead from a dedicated I/O thread while the current layer
computes. `PILOT_REAL=1` moves the prefetched loads off the critical path
(measured +11pp hit rate on a big-cache host), and `PILOT_TWO=1` folds the
computed shared-expert into the prediction (+3% recall,
[#200](https://github.com/JustVugg/colibri/issues/200)). On disk-saturated
hosts hint-only PILOT can be net negative — measure on yours.

## Speculation and reproducibility

Speculative decoding requires that the draft and verify paths compute the same
function — `SPEC_PIN=1` (default since [#294](https://github.com/JustVugg/colibri/pull/294))
pins every forward issued while drafts are live to the platform's S=1 kernel
family. For byte-exact reproducibility across runs: `DRAFT=0`, plus `IDOT=0
COLI_CUDA=0` if you also want kernel-family/GPU independence. Acceptance
percentages are not comparable across engine versions under `--topp`
([#163](https://github.com/JustVugg/colibri/issues/163) has the full story).

## Conversations reopen warm

`coli chat` persists the compressed MLA KV-cache to disk after every turn
(`.coli_kv`, ~182 KB/token, appended incrementally, crash-safe). Close the chat,
reopen it tomorrow — the model still remembers the whole conversation and **zero
re-prefill happens**: validated byte-identical to an uninterrupted session.
`:reset` clears it, `KVSAVE=0` disables it.
