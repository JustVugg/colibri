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

## Resource policy

`coli plan` reports the planned hot (VRAM), warm (RAM), and cold backing (disk)
tiers, the reason for each placement, and the expected bottleneck. The default
`--policy quality` and `--policy balanced` modes preserve checkpoint quantization
and router decisions unless `--topk` or `--topp` is passed; those explicit lossy
overrides print a warning and proceed.

Auto-tier plans size OpenMP from physical cores and bind workers across cores.
Memory-bound quantized kernels can regress sharply when SMT siblings compete for
limited memory channels; explicit `OMP_*` settings always take precedence.

```bash
coli plan --model /models/glm52_i4 --policy quality
coli run --auto-tier --policy quality "Explain MoE offloading"
# Explicit research-only router reduction:
coli run --policy experimental-fast --topk 4 "Benchmark prompt"
```

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
