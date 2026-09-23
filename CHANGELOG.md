# Changelog

All notable changes to colibrì are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/).

## [1.12.1] — 2026-09-22

70 pull requests since v1.12.0, 58 of them from contributors. Two tokenizers
brought back to the reference, brio on the ninth engine, `coli chat` working
again at the default context on two families, and a placement decision that
is now measured on the card in front of it instead of predicted.

### Tokenizers, measured against the reference

- **#1654**: qwen36 tokenized differently from HF `tokenizers` in two ways.
  An added token right after punctuation was encoded as text (`X.<|im_end|>`
  was 7 tokens instead of 3, every chat turn ending in punctuation paid +4,
  #1653), and a whitespace run followed by a non-space was one piece where
  the regex's `\s+(?!\S)` leaves the last char to the next one, so every
  indented line of code tokenized differently. Measured on the real
  vocabulary: 2,803 lines and blocks of code, Markdown, Chinese and
  Japanese went from 757 identical to 2,803, with 5.4% fewer tokens.
- **#1656**: OLMoE's `tokenizer.json` has no Split, a bare ByteLevel with
  `use_regex`, for which HF runs the original GPT-2 pattern; `tok.h` applied
  cl100k. A GPT-2 family in `tok.h`: 1,560/1,708 identical before, 1,708/1,708
  after. The same measurement on GLM-5.2/5.3/5.3-Flash, DeepSeek V4 and V4.1,
  Inkling and Qwen3.8 came back identical on every case.

### Brio and the serve contract

- **#1662**: `POST /v1/systemone`, the request and the reply of TypeSafe's
  Jev API, served by the brio channel: a client written for it points at
  colibri and changes the base URL. `noul` is a yes/no question, `choice`
  scores the labels with their descriptions in the text, `score` the level
  numbers with the expected value and the legend; `confidence` by their
  documented formula. Any `model` name is accepted on that route. Measured on
  the real Qwen3.6: the three-question example of the docs in 1m46 with the
  state read once.
- **#1655**: the DeepSeek V4 engine speaks the numeric channel (`logprobs=k`,
  `pin=1`, `max_tokens=0`), so `/v1/brio` works on the ninth engine instead
  of answering 500 (#1648). The head that used to keep only its argmax now
  returns the whole row; `ECHO` per prompt position during prefill, the
  prompt-end scores kept with a state snapshot on `pin`, and a logprob tail
  on every `DATA` frame during generation. The tiny fixture pins that the
  best `ECHO` token equals the greedy token from the same prefix, and that
  the pinned predictor equals a cold prefill's.
- **#1659**: qwen36 and qwen38 refused a request when `prompt + max_tokens`
  exceeded the context, and the gateway's default budget for these two
  families is 8192, the whole default context: every request without
  `max_tokens` and every `coli chat` message answered 400 on a two-token
  prompt (#1641). `max_tokens` is now a ceiling, clamped to the room the
  prompt leaves, as GLM and DeepSeek V4 already did; only a prompt that does
  not fit is refused. `docs/api.md` states the rule, `docs/qwen38.md` names
  `Q38_MAXT` as the variable `--ctx` becomes.

### The dense trunk in VRAM, measured before it is placed

- **#1657**: qwen36 offers the rest of its dense trunk to the VRAM placer:
  the DeltaNet out_proj (`dnout`), the attention q/k/v/o (`attnproj`) and
  the shared expert (`shexp`), about 650 MB more of int8 on the 35B beside
  `lmhead` and `dnproj`. Measured on four Tesla M10 by the reporter of
  #1652, every placed component ran slower than the CPU (lm_head 68.8 ms
  against 41.7), so the engine now times one GEMV both ways at startup and
  withdraws the whole automatic placement when the GPU loses, giving the
  VRAM back to the experts: `auto` equals `off` on that box, byte-identical
  output. A hand-written `COLI_PLACE` stands; `COLI_TRUNK_PROBE=0` trusts
  the placer.

### Performance

- **#1664**: qwen36's dense trunk and routed experts multiply with integer
  dot products. The activation is quantized to int8 once per call and the
  weights, int8 rows or int4 planar blocks, meet it with maddubs / vpdpbusd
  instead of a float conversion per weight; the integer kernels move from
  `quant.h` into `idot.h`, shared by every engine. Measured on the 35B, 8
  threads, every expert resident: decode 6.71 to 8.23 tok/s (+22.6%), lm_head
  12.6 to 10.1 ms/token, the expert compute 22.7 to 15.6, for +1.3%
  perplexity on 4 x 512 tokens. Both are the default (`COLI_DENSE_IDOT=0`,
  `QWEN_EXPERT_ACT=f32` restore the f32 kernels). `COLI_DENSE_BITS=4` with
  `COLI_DENSE_INT4=<components>` stores part of the trunk as int4 in blocks
  of 64: opt-in, with the perplexity it costs per component in the docs
  (lm_head alone +2.4%, everything +10%).
- **#1668**: qwen38's dense trunk (553 matrices, 3.6 G weights, 8 GiB of
  BF16 read on every token, more than the ten routed experts) is kept on the
  CPU as int8 rows with the BF16 copy released, and multiplied with the same
  integer kernels; the routed experts' e4m3 blocks are decoded eight at a
  time in registers and multiplied with FMA instead of one table lookup per
  weight. Measured on the released Qwen3.8-Flash-Next-FP8, 8 threads, RAM
  LRU 96 per layer: decode 0.61 to 1.42 tok/s, the trunk 434 to 85 ms/token,
  lm_head 76 to 13, the expert GEMVs 388 to 140, peak RSS 32.2 to 28.5 GB;
  prefill of 512 tokens 495 to 149 s. Perplexity on 4 x 512 tokens +0.5%
  (two chunks lower, two higher); the vector FP8 kernel alone reproduces the
  BF16 run to four decimals. Both are the default (`Q38_TRUNK_CPU_INT8=0`
  keeps the BF16 trunk, `Q38_FP8_KERNEL=scalar` the table kernel).

### Performance, from contributors

- **#1606**: the K1b grouped int4 family gets a multi-row tile and AVX-512
  and AMX arms, and is no longer switched off on AVX-512 builds; exact on
  all eight engines on a 16-core AVX-512 host.
- **#1239**: an SSE4.1 tier for the olmoe and qwen36 int8 GEMV, for hosts
  with SSE4.1 but no AVX2; on the Sandy Bridge of #1652 decode went from
  2.45 to 3.54 tok/s.
- **#1313**: `matmul_fp8` computes four output rows per pass under clang,
  where the contraction makes it bit-exact; GCC keeps the one-row kernel.
- **#1612**: qwen36 gains the GLM engine's `CACHE_ROUTE` lever, with the
  VRAM tier as the first residency level, opt-in.
- **#906**: `DEGRADE_ZERO`, an opt-in policy that zero-fills a missed
  expert slot below a gate-weight threshold instead of blocking on the
  load (#865).
- **#1677**: qwen36 projects a prompt's DeltaNet inputs (qkv and z) on the
  card in blocks of up to 256 rows instead of one row at a time; a 259-row
  prefill makes 2 projection calls instead of 259, with the convolution
  history and the recurrent state checked against the CPU run. The paired
  microbenchmark on an RTX 4070 read 4 to 10x per projection.
- **#1674**: qwen36's attention projections the tier placed in VRAM answer
  a whole prompt batch with one call per matrix; a failed call turns only
  that handle off and the prompt continues on the CPU.
- **#1676**: Kimi K3's streaming CUDA expert keeps the gate, up and SiTU
  intermediates on the device and applies down there (one fused entry
  point, optional in the DLL: an older backend keeps the three-call path).
- **#1673**: the streaming MXFP4 matmul reuses one grow-only device scratch
  per card instead of allocating and freeing weights and scales on every
  call.
- **#1559** (kreuzzelg): `convert_qwen36.py --down-bits 8` writes the mixed
  expert layout, int4 gs64 gate/up and int8 down in one slab (5.7 bits per
  weight against gs64's 4.5); the engine tells it apart by size and reads
  each matrix in its own format on the CPU path, and refuses the VRAM tier
  with a line. It is the knob behind the #1370 numbers: on wikitext-2 the
  int8 down alone recovers a quarter of the gap between gs64 and all-int8,
  the rest sits in gate/up. A measurement tool and a middle step, not the
  answer to the gap.
- **#1686** (DebugSultan): qwen38's prefill chunk (`Q38_PREFILL_BATCH_ROWS`)
  and workspace (`Q38_PREFILL_WORKSPACE_MIB`) are runtime knobs, the expert
  load batch is no longer capped at top-k, and the QSA ranking and
  attention run per position in parallel at prefill. Measured on the
  released checkpoint on top of the int8 trunk: output byte-identical at
  every chunk width, no speed change on our 16-core server; the knobs are
  there for hardware where the chunk binds.

### Fixed

- **#1650** (bokiko): a Qwen3.8 pin snapshots the recurrent and PLE state
  but reuses the live attention and indexer rows; after an unrelated prompt
  overwrote those rows, returning to the pin could change brio logprobs
  without a warning. The engine now records the token identity of the live
  rows (`kv_prefix.h`) and refuses a stale pin or prefix restore; image rows
  are tainted. Wire regressions run on the BF16 and FP8 fixtures.
- **#1626**: `SNAP` is the model directory for every non-GLM engine, so
  `coli run` stops handing them a leftover environment (#1600).
- **#1604**: glm53 honours `Mat.resident` in the Vulkan gate and frees the
  Vulkan handle in `mat_release`.
- **#1321**: glm53 sizes its expert cache around the model rather than
  around `MemAvailable`, which the page cache had been inflating.
- **#1588**: qwen36 refuses loudly on a failed encode-buffer realloc instead
  of writing through NULL.
- **#1546**: `coli convert` routes OLMoE to `convert_olmoe_merged.py`.
- **#1630**: olmoe emits the `ROUTE_TRACE` records it announced; the stream
  used to be a zero-byte file.
- **#1610**: `v41_dsml.py` is staged during installation (and the nix flake
  bumped).
- **#1511**: the GPU test suite builds under HIP on gfx1151.
- **#1658**: `test_mem_available` compared two reads of available memory
  with `==` and failed on a busy Windows runner; a quarter of a GB of
  tolerance.
- **#1670**: DeepSeek V4.1 read only its argv cache cap, so `RAM_GB=120`
  on a 128 GB box left the engine at eight expert slots per layer and
  23.8 GB of RSS (#1666). With `--cap` omitted, `coli chat`, `coli serve`
  and `coli web` now size the cache from the resource plan, with `RAM_GB`
  or `--ram` as the budget; an explicit `--cap`, a measured profile and an
  auto-tier plan keep precedence.
- **#1671**: DeepSeek V4.1 treats `max_tokens` as a ceiling like the other
  engines (#1641): a fitting prompt with a large request generates what
  the context leaves, a score-only prompt may fill the context, and a
  prompt one token over it is refused instead of silently truncated.
- **#1675**: resident MXFP4 tensors on CUDA carried O float scales where
  the kernel reads O x ceil(I/32) exponent bytes: short buffers were
  over-read and long ones truncated. One format-aware size for upload,
  refresh, accounting and release.
- **#1678**, **#1679**, **#1680**, **#1682**, **#1683**, **#1684**: the
  Qwen CUDA tier's lifecycle, end to end. Shutdown releases every resident
  expert, projection handle and host table after parked callers resume;
  a failed gate, up or down upload frees what it had already allocated;
  the expert budget charges the three scale buffers at their own sizes
  (two experts used to be admitted where one fit); a failed result
  collection stops inference instead of publishing a partial MoE sum; a
  failed or explicitly disabled tier start unwinds its storage and
  synchronization objects, and a second init cannot overwrite a running
  tier; the CUDA backend validates the whole device list before touching
  state and keeps live contexts on a repeated init. Fault-injected on the
  fake backend, then run together on an RTX 4070 under compute-sanitizer
  with zero errors and zero bytes leaked.
- **#1669**: `test_systemone_api` imports its scoring engine relative to
  its package, so an installed `tests` package no longer breaks discovery.
- **#1697** (kevin9327): the dashboard redesign had dropped the reasoning
  stream: thinking tokens arrived on `delta.reasoning_content` and vanished,
  the bubble stayed empty until the answer and a stop during thinking lost
  the turn. The stream is read again, rendered as its own folding block,
  counted in the rate and the time to first token, and a unit test pins the
  split.
- **#1693** (namespaceMarcello): with `PILOT` on, OLMoE could read the same
  expert twice, once from the prefetcher and once from the forward pass,
  into two slots; a slot being read now keeps a reservation in the index
  (the `colibri.c` pattern) and the second caller waits for the first read
  to publish. Three model-free scenarios pin it.
- **#1695** (namespaceMarcello): the prefill echo state and `serve_echo` sit
  under the same `QWEN36_NO_MAIN` guard, so the segment build no longer
  warns about a function it never gets; the full build is byte-identical.
- **#1622**: DeepSeek V4's REAP checkpoints store each expert as six
  per-matrix records; the engine read them through buffered pread and
  counted every one as a direct-I/O fallback (36% of expert reads on the
  150B, #1615). Each segment now goes through the aligned direct window,
  with a regression on a generated per-matrix fixture.
- **#1597**: a replayed tool call whose `arguments` parsed as JSON but was
  not an object (`"[1, 2]"`, `"5"`) answered HTTP 500 from the GLM renderers
  before the engine was asked anything; it renders the call without
  arguments, as every other renderer already did.
- **#1624**: the five gcc 13 warnings left in `make check` are gone, and
  `st_index_load` refuses an index path that would not fit its buffer
  instead of opening a truncated one, with a long-path case in the tests.
- **#1651**: a `pyflakes` pass over the launcher, autotune, the family
  registry and the qwen36 converter: a `measure()` defined twice, a
  `readline` import without a fallback, a stray f-string, dead variables.
- **#1580**: `make qwen36 CUDA_DLL=1` on Windows reached GNU make's implicit
  rule and built a CPU-only binary; a bare `qwen36` alias, a `.build-config`
  prerequisite so a CUDA_DLL change rebuilds, a loader-against-header parity
  test, and the Windows CUDA tier documented.
- **#1556**: `coli plan` on macOS said "no supported GPU detected" on every
  Mac; it now lists the Metal device by name, as identity only, without
  pretending unified memory is a VRAM budget.
- **#1691**: the installed launcher invoked as `/bin/coli` or `/sbin/coli`
  on a merged-/usr system derived `/libexec/colibri` instead of
  `/usr/libexec/colibri`, because `abspath` kept the alias (#1689,
  florin65's patch): `realpath` first. A test runs the launcher through
  such an alias, and another checks that every root module the launcher
  reaches is in the `make install` list, the gap #1610 closed by hand.

### Tools and the gateway

- **#1425**: a general GGUF reader, pure stdlib, and a converter from GGUF
  OLMoE checkpoints to a colibri container, with the numerical evidence in
  its own CI job.
- **#1497**: opt-in prompt-injected tool calling for the families without
  native tool tokens (OLMoE, Qwen3.6), behind `COLI_TOOL_FALLBACK=1`, with a
  two-turn end-to-end test.
- **#1355**, **#1357**: durable per-request results and strict stdout
  classification in the eval harness, and the logprob-gap check gated on
  the engine preamble.
- **#1687**: `GET /metrics` in Prometheus text format, behind the API key:
  four gauges, six outcome counters and four histograms (queue wait, slot
  occupancy, first output, engine call), no request labels, no new
  dependency. The admission scheduler distinguishes completion, failure
  and cancellation, lets a request use a free slot that no earlier waiter
  reserved, and joins the keepalive pump before the slot is released.
- **#1402** (enitimeago): a request whose last message is a non-empty
  `assistant` turn continues that turn instead of answering in a new one, on
  `/v1/chat/completions` and `/v1/messages`, for all nine families (Kimi K3
  frames the open turn engine-side); the prompt ends inside the turn as the
  official template renders it without a generation cue. On by default,
  `COLI_CONTINUE_ASSISTANT=0` restores the old behaviour; refused together
  with tools or a turn ending in whitespace, with a 400 that says why. Each
  renderer is pinned against the vendored template (#1401).
- **#1605**: `ORACLE_STRICT=1` makes a GLM oracle comparison exit non-zero
  when it fails, token-exact by default with `ORACLE_TF_MAX_MISMATCHES` for
  the documented teacher-forcing allowance; references are validated before
  the comparison and non-finite logits cannot pass. Both oracle CI jobs run
  real-process regressions against it.
- **#1705**: `tools/benchmark_baseline.py`, a collection protocol on top of
  the HTTP harness for a repeated three-engine serving baseline: one frozen
  manifest (hardware, model and template identity, per-engine launch
  settings, cache and speculation policy), a rotating plan over a
  concurrency matrix, one collector per engine and round that manages no
  server, and a comparison that keeps failed and missing cells visible and
  distinguishes matched artifacts from deployment comparisons. No results
  are bundled and no ranking is emitted.
- **#1688**: `tools/benchmark_http_serving.py`, a stdlib HTTP streaming
  benchmark over fixed JSONL conversations: closed-loop or paced arrivals
  (periodic or Poisson, seeded), warmup separated from measurement,
  first-output and duration SLOs, latency percentiles and usage-based
  token throughput; a truncated or malformed stream is a failure, not a
  sample.

### Docs

- **#1639**, **#1644**: the README shows brio mode and the dashboard as it
  is: the workspace, the Brain page (the measured expert atlas as a cortex,
  and a region inside it) and the Profiling page, in four languages;
  `docs/api.md` describes the four pages instead of the old console.
- **#1492**, **#1643**: a Japanese README, and its banner at the shipping
  version, which the banner test now checks in every language.
- **#1634**: the multi-disk guide states measured gains and limits instead
  of "twice the bandwidth", with Bash and PowerShell examples.
- **#1617**: connecting the pi coding agent to `coli serve`.
- **#1619**: `expected_bytes` identity versus physical extent for
  int4-rans256-g0 (#1273).
- **#1618** (bherald): `docs/qwen38.md` no longer calls the engine text-only;
  the vision tower and the gateway image path shipped in 1.12.0.
- **#1649**, **#1647** (Suraj2105-1): the musl CI job runs on Alpine 3.24
  and the release pipeline on Node.js 22, ahead of the 3.21 and Node 20
  end of life.
- **#1568** (Yoruxyv): an Indonesian translation of the dashboard.
- **#1681** (XBold): the README and `docs/qwen38.md` no longer say Qwen3.8
  has no GPU backend; the CUDA VRAM expert tier and the int8 trunk in VRAM
  shipped in 1.12.0.

## [1.12.0] — 2026-09-20

81 pull requests since v1.11.0. A new way to ask a model a closed question,
a redesigned dashboard and landing page, and a long run of small failures
that used to answer 500 or die on a locale.

### Brio mode: score a closed set instead of generating

- **#1632**: brio mode, on all nine engines. Hand the engine the options an
  answer is allowed to take and it reports the probability of each one
  instead of writing the answer: the option tokens are read, not sampled,
  so `completion_tokens` is 0 and no reply can fall outside the list. It is
  opt-in per request through two `SUBMIT` keys, `logprobs=k` and `pin=1`;
  without them every frame is byte-identical to before, which is asserted
  per engine rather than claimed. `max_tokens=0` became legal, but only
  together with `logprobs>0`.
- The shared contract is three headers: `decode_batch.h` for the logprob
  tail, `serve_codec.h` for the wire, and `pin_pool.h` for nested state
  snapshots, so a request can keep one photograph of the shared
  instructions and a deeper one of the instructions plus the question. With
  one level the question is re-read once per option; with two, four items
  cost 176 tokens instead of 496.
- `kv_prefix_holds()` is the guard that makes the snapshots safe: a
  snapshot is reused only when the prefix record still holds the same ids,
  because a K/V bank can be dropped between two options.
- Exactness is measured, not asserted: the logprobs read from a snapshot
  match a cold recomputation to 0.00e+00 on all nine tiny fixtures, and on
  the two real checkpoints available (qwen36 at 22 GB and DeepSeek V4.1
  Flash at 476 GB).
- Three clients: `POST /v1/brio` on the gateway, `/brio` in `coli chat`,
  and a page in the dashboard. `docs/brio.md` has the request, the reply,
  the normalisation choice and the cases where the mode does not help.
- The endpoint takes three forms. `options` is one closed question.
  `questions` is many questions on one state, each with its own options,
  with the state photographed once and the snapshots ordered by the server.
  `schema` is an object of field to allowed values: the server writes the
  JSON skeleton and fills it one field at a time, so the object is valid by
  construction and every value is one the caller allowed, each with its own
  probability and entropy. Both were measured before they were exposed: the
  5.7x and the 2.4x above are these two forms.

### The dashboard and the landing page

- **#1633**: the web dashboard is redesigned around a workspace and a
  navigation dock, with a brio page that reads a document once and asks it
  several questions, each with its own option set, and a light and dark
  theme.
- **#1635**: the landing page lists the models as a searchable list instead
  of a card grid, gains a brio section, and gets a light and dark theme
  built on the mechanism proposed in #1551. The shipping version and the
  family names are generated from `c/version.py` and `c/family_registry.py`,
  so the page cannot claim eight families again while nine are running.
  The hero plays a recorded session rather than a mock-up.

### Performance

- **#1477**: one routed-expert kernel, `expert_ffn.h`, shared by the MoE
  engines and used by qwen36 first: int4 kept planar in RAM instead of
  unpacked to int8, and two OpenMP regions per layer instead of three per
  expert. 12.8 to 15.7 tok/s with the resident set down from 29 to 17 GB,
  output identical.
- **#1424**: a CUDA VRAM tier for Qwen3.8-Flash-Next, 43 % on one 8 GB card
  and 60 % on two.
- **#1553** and **#1558**: the previous turn's KV is reused instead of
  re-prefilling the transcript, on qwen36, olmoe and DeepSeek V4.1, with a
  resumed prefill that is exact.
- **#1521** and **#1524**: the mHC mix and the KDA convolution and head
  loops run in parallel, bit-identical, for 3.9 % and 3.0 %.
- **#1517**: the four engines that never sized their OpenMP team now do.
- **#1522**: the DeepSeek V4.1 expert stream can be read from more than one
  drive; **#1455** weights glm53 expert reads by disk and honours
  `COLI_MODEL_MIRROR`; **#1543** fixes the V4 mirror cache on unified memory.
- **#1495**: less overhead in the sparse router's top-K selection.

### Fixed

- Requests that answered `500 engine failed` and now answer properly:
  a non-object `json_schema` (**#1587**), an unpaired surrogate escape
  (**#1589**), a non-object `tool_choice.function` (**#1598**), and a tool
  call whose arguments are not an object.
- **#1595**: the grammar draft walker switched itself off after about sixty
  repetitions. **#1596**: a nested array schema doubled the compiled
  grammar at every level.
- **#1483** and **#1514**: olmoe, inkling and glm53 refuse an over-long
  prompt with `CONTEXT_EXCEEDED` instead of failing late.
- **#1549**: the lazy shard-mapping table in `st.h` was published without
  synchronisation and glm53 reached it from an OpenMP region.
- **#1557**: a truncated multibyte tail made the qwen36 tokenizer read past
  the prompt. **#1530**: an integer overflow check in `kimi_k3.c`.
- **#1631**: `coli serve` reaches the expert-history save when it is stopped
  with SIGTERM, not only on Ctrl-C. **#1628**: the expert grid is resent
  after every turn, so the dashboard stops showing the cold snapshot
  forever.
- **#1608**: `coli doctor` matches core tensor roles by component instead of
  by one family's spelling.
- **#1526**, **#1584**, **#1585**, **#1536**: the planner, the auto-tuner
  and the registry agree about which engine is running, and stop advising
  GLM-only knobs on every family.
- **#1489**: `coli bench` ran the GLM engine on any model. **#1503**: every
  engine is handed its model directory through `SNAP`. **#1486** and
  **#1488**: a directory without `config.json` names no engine, and
  `coli convert` refuses an output directory that already holds one.

### Windows, macOS and locales

- **#1515**, **#1560**, **#1561**, **#1562**, **#1516**: the chat, `coli
  run`, `coli tune` and the datapoint harness no longer die when an engine
  line, a prompt or a dash does not fit the console code page.
- **#1537**, **#1579**, **#1535**: the qwen36 CUDA_DLL build on Windows gets
  its tier, is recognised, and the device probe loads the backend before
  counting devices. **#1563**: the Inkling dense int4 converter finishes.
- **#1599**: MacPorts libomp built every macOS engine single-threaded.
  **#1548**: Metal falls back to `MTLCopyAllDevices` when there is no
  system default device. **#1504**: olmoe measures available RAM on macOS
  and Windows too.

### Diagnostics

- **#1494**, **#1506**, **#1512**: `PROF` reports block time, head time,
  matmul, attention and lm_head, at microsecond resolution, on deepseek_v4,
  olmoe and qwen36. **#1555**: `CONSIST=1` checks prefill against decode.
- **#1482**: configurable generation statistics in `coli chat`, with exact
  token throughput when the server reports it.
- **#1468**, **#1487**, **#1480**: a failure in the MoE step, a missing core
  tensor and duplicate tensor names are diagnosed by name instead of by a
  generic message.

### Docs and licence

- **#1540** and **#1541**: the copyright holder is named and a `NOTICE` file
  is added.
- **#1294**: a reproducible benchmarking protocol; **#1508**: a GPU-backend
  datapoint needs a correctness line to be accepted.
- **#1471** and **#1473**: the expired Discord invite is replaced in all
  four READMEs and on the site.

## [1.11.0] — 2026-09-13

56 pull requests since v1.10.2. A ninth model family, five real bugs closed
across four engines, and the two platforms the C tests never built on now
building them in CI.

### A ninth engine: DeepSeek V4.1 Flash

- **#1453**: DeepSeek V4.1 Flash (552B, 510 GB on disk) runs on a CPU box
  streaming experts from an SSD, with no conversion: the released checkpoint
  is read natively, fp8 dense with 32x32 ue8m0 tiles and fp4 experts whose
  layout is byte-identical to the mxfp4 the Kimi K3 engine already reads.
  Everything in the architecture that is not V4 is in: Engram (two n-gram
  memories of 384M rows, 203 GB, that never enter RAM), the DSA indexer with
  its two-level candidate source, hyper-connections, the compressor, the
  32-layer vision tower with its aligner, DSpark speculative decoding (3
  stages, blocks of 5, verified in one batched forward with a rollback for
  the rejected rows) and tool calling in the checkpoint's own DSML format.
- Held token-exact in CI against a torch-only CPU reference
  (`c/tools/dsv41_ref.py`, written because the vendor's own forward needs
  tilelang GPU kernels), at three cache capacities, on a short and a
  40-token prompt, under all three speculative modes; the vision tower is
  matched to 5e-06. The CI also asserts that the vendor's index-key
  republication policy and the tidy reading disagree, so the shipped default
  cannot be "corrected" by accident (`V41_INDEX_OWNER`).
- Measured on the released checkpoint, cold, caches dropped before every run,
  same prompt and seed: a turn went from 78.7 s to 25.1 s during the work
  (0.305 to 0.957 tok/s) through batched expert reads (`V41_READ_DEPTH`,
  default 8 is the measured knee), attention matrices read once per block of
  positions instead of once per token, and an expert-major MoE. Every step
  is bit-exact against what it replaced. A five-turn chat session runs at
  1.14 to 1.58 tok/s; an `image_url` part goes through the gateway into the
  vision tower on the same engine.
- Two ways of hiding the expert reads behind the matmuls were built,
  measured worse, and removed; both are written up in `docs/deepseek-v41.md`
  with the numbers, so the next attempt starts from them.
- `coli chat`, `coli serve` and `coli web` work; `coli run` is deliberately
  unwired for this family, as for qwen36 and qwen38. Per-turn accounting is
  behind `V41_STATS`, off by default, so the chat stays clean.

### Fixed

- **#1390** (@crichalchemist): the qwen36 GPU tier's `qt_fill_wait` returned
  when the upload queue was empty, but the uploader frees the ring slot at
  dequeue, before the backend copies anything, so the warmstart freed the RAM
  int8 copy of experts still in flight. An in-flight count that the uploader
  decrements only once the slot is resident (#1360).
- **#1434** (@njloof): `fmt=0` is raw f32 and has no scale array, but the
  Metal sizing helper returned a per-row scale size for it, so
  `coli_metal_matmul` wrapped a NULL pointer and segfaulted inside
  `newBufferWithBytes`, or silently produced all-zero output when the size
  happened to be a page multiple. The kernel no longer applies a scale for
  `fmt=0`, and a fail-closed guard sized off the same helper falls back to
  the CPU path for any format that needs scales and got none.
- **#1389** (@bherald): GLM-5.3 teardown leaked the lazily allocated dashboard
  HITS table.
- **#1461**: `coli cluster worker` could never start without an explicit
  `--cap`: the default is `None`, and `str(None)` reached the engine's
  argument check as the literal `"None"` (#1452). The worker now resolves the
  cap the way every other launcher does.
- **#1460**: olmoe's serve-mode `PROF` line published five literal zeros, so
  `/profile` reported `expert_disk_s = 0.0` on turns where the expert reads
  were the workload. The disk phase is measured now; the other four fields
  stay zero because they are unmeasured, not because a guess would look
  better (#1449, half of it).
- **#1445**: olmoe's `--ram` was inert and "no `--cap`" meant a constant eight
  slots per layer regardless of memory; the cap is derived from the RAM budget
  (#1443). On the reporter's box a 2,291-token prompt went from 675 s to 353 s
  with no flags (#1442).
- **#1377**: available RAM is measured on every platform (glm53 read
  `/proc/meminfo` unconditionally, so on Windows and macOS it read 0 and the
  expert budget collapsed to 1 GB, which is how `coli tune` went OOM in
  #1375), and the Windows commit limit is respected.
- **#1381**: the context ceiling is announced, a request that cannot be
  honoured is refused up front, and the number reported is the right one
  (#1376).
- **#1393**: GLM-5.3 replies always start inside the think block (#1278).
- **#1414** (@dajiaohuang): the GLM-5.3 serve loop honours `CANCEL` while the
  turn is still running (#1332).
- **#1423** (@kreuzzelg): the lazy HITS table is built privately and published
  under a lock in qwen36, kimi_k3, glm53 and qwen38; the parallel warmstart
  segfaulted on first touch about one run in twelve (#1422).
- **#1404** (@Petsku01): the qwen36 tier offers int8 experts on the decode
  path (#1391). **#1388** (@crichalchemist): `QT_MAX_ROWS` replaces seven
  literal 32s that had to agree, and the int8 copy is freed by ownership
  rather than by format.
- **#1421** (@bherald): `coli` no longer overwrites an explicit
  `CUDA_EXPERT_GB` when it decides to place the dense trunk on the card.
- **#1244** (@Unknown-Findout): the CUDA expert tier is charged real VRAM, not
  logical bytes (#687). **#1394**: the VRAM prefix is priced per row, not at
  the container's widest (#1351). **#1410**, **#1411**: on a single GPU the
  VRAM prefix is sized from the VRAM budget and the measured headroom, not
  from the RAM pin plan capped by the LRU reserve (#1409, #1405).
- **#1447** (@trigger2k20): `coli mirror verify` accepts a complete mirror
  copied by other means, with no receipt, and fails closed on a short or
  corrupt shard.
- **#1456** (@trigger2k20): the planner reports `memory.unified` from the
  host, not from whether the selected engine has a placement-capable GPU, so
  a CPU-only engine on Apple Silicon no longer reads as non-unified; and
  `coli tune` stops suggesting `DRAFT`/`PIPE`/`PIN`/`NUMA` to glm53, which
  ignores them.
- **#1396** (@dmoraesrs): the web reasoning selector drops "Medium" for
  GLM 5.3, which the engine renders identically to "High".
- **#1428**, **#1435** (@iiEliJas), **#1433** and **#1432** (@texasich),
  **#1440**, **#1459**, **#1458** (@trigger2k20), **#1463**: `setenv` visible
  to `getenv` in-process on Windows, `malloc_trim` guarded to glibc so the
  engines build on musl, a missing bench recipe restored, the per-test
  `_putenv_s` helpers dropped, tests including `compat.h` directly, and
  Clang warnings cleaned up.

### Added

- **#1462** (@trigger2k20): an opt-in Metal path for GLM-5.3-Flash's routed
  experts on Apple Silicon (`COLI_METAL=1`): gate, up, clamped SwiGLU, down
  and the route-weighted scatter on the GPU, CPU path unchanged when Metal
  is off. Validated against the CPU implementation to 2.8e-09 and measured
  on an M4 Max: 0.469 to 0.621 tok/s at 32 tokens, up to 2.15 tok/s with
  the expert cache tuned. Also fixes `coli run --cap N` being silently
  ignored on the glm53 one-shot path.
- **#1457** (@trigger2k20): GLM-5.3-Flash feeds the shared routing telemetry
  and writes `.coli_usage`, so usage-driven placement and partial-mirror
  planning have data for it.
- **#1454** (@yuripourre): the qwen36 GPU tier compiles with `HIP=1`.
- **#1399** (@SebaWag): `TRUNK_RESIDENT_LAYERS=N` streams the dense trunk
  through mmap (#826).
- **#1361**, **#1374** (@kreuzzelg): the qwen36 dense trunk places itself
  (`COLI_PLACE=auto`, priced in bytes saved per byte of VRAM), fp8 streaming
  mode, VRAM accounting at allocator granularity, thread affinity, `COLI_GPU`.
- **#1383**, **#1385**, **#1386**, **#1387**: Brain and Profile tabs (EMAP,
  HITS, PROF) on every engine. **#1382** (@dmoraesrs): a reasoning depth
  selector in the web UI (#1311).
- **#1407**, **#1408**, **#1406**: CI builds the portable Windows CUDA DLL and
  publishes it as an artifact; the Makefile refuses the 32-bit `cl.exe` before
  nvcc runs (#1405).
- **#1436**: CI builds against musl (Alpine) and compiles the environment
  tests on Windows, the two holes that let #1430 and #1420 ship.
- **#1360** (@kreuzzelg): tier invariants on the fake backend, and the
  reservation leak they found. **#1419**, **#1415**, **#1418**
  (@dajiaohuang): the V4 and fp8 test harnesses build on Windows.
- **#1439** (@bherald): the planner tests no longer materialise four 3 GB
  zero-filled payloads. **#1358** (@monotophic): a raw-evidence adapter for
  the scoring corpus. **#1238** (@ZacharyZcR): reproducible performance
  records are validated.

### Docs

- **#1446**: the qwen36 `--ram` section said the engine does not stream at
  all. **#1392**: Azure GLM-5.2 benchmark rows (#1379, #1380, #1384), the rANS
  cross-reference (#1273), and the int8 probe pinned in CI (#1331).
- **#1427** (@ZH1995): README.zh-CN formatting. **#1438** (@iiEliJas): the
  `malloc_trim` comment.

## [1.10.2] — 2026-09-06

Patch release. Three of these fixes answer reports made against 1.10.1 in the
days after it shipped.

### Fixed

- `coli convert` picks the converter from the checkpoint's `config.json`. It ran
  GLM-5.2's converter on everything; on GLM-5.3-Flash that quantized the nested
  embedding and the engine refused the result hours later inside `coli web`.
  An option the target converter does not take is refused, not dropped (#1368,
  #1369). The converter also refuses a checkpoint it cannot serve, with a
  per-family pointer (#1305).
- `coli doctor` no longer reports two missing core tensors on a healthy
  GLM-5.3-Flash: it matches roles, prefix-agnostically, instead of GLM-5.2's
  literal names (#1365, #1366).
- The release archive ships every file `coli` reaches: `iq3_pack.py`, its grid
  data file, and `tools/convert_glm53.py`, which had never been packaged
  (#1359, #1364).
- The RSS guard counts anonymous memory, not reclaimable page cache, so a mapped
  container no longer evicts experts to free memory it was not using (#1350).
- `serve` honors CANCEL while a turn is still running (#1336), and the
  disconnect scenario is built rather than hoped for (#1329).
- Metal: bit-exact fp8-e4m3 decode (#1346). Qwen3.6: tokenizer merges in both
  spellings (#1319). macOS: Homebrew prefixes found when `brew` is off the PATH
  (#1320). DeepSeek V4 on macOS: real CPU and memory in HWINFO (#1308).
- `coli doctor` omits the GPU plan for a CPU-only engine (#1322); a missing core
  tensor explains itself on every fatal path (#1318); GLM-5.3-Flash `--no-think`
  is the template's lowest effort level, not a shape of ours (#1327).
- GLM-5.3-Flash `serve` can use the 16 KV slots the engine has; the registry
  declared 1 (#1283).
- `image_url` local reads: `..` is refused, and with `COLI_IMAGE_ROOT` set a
  path must resolve inside it, so an authenticated client of a non-loopback
  server cannot read arbitrary files through the image API. Error messages no
  longer confirm a path or its permissions (#1354).

### Changed

- The GLM family is named `GLM-5.2/5.3`: the two checkpoints share the base
  model and cannot be told apart from their configuration (#1367).
- DeepSeek V4 Flash REAP-150B (85 GB, 132 of 256 experts) loads with the same
  engine (#1310), is documented, and is announced by its measured geometry
  rather than the official checkpoint's 284B.
- The qwen36 VRAM tier promotes int8 experts instead of reserving for nothing
  (#1334), and the three tier bugs that surfaced with it are fixed: an `is_x`
  overrun with two or more GPUs, a shutdown that could hang, and a
  use-after-free on int8 containers (#1339, #1340, #1341, #1344). Kimi K3
  stops paying for a DSA indexer nothing reads (#1335).
- Opt-in: `COLI_MAP_EXPERTS=1` serves experts through a per-shard file mapping
  (#1325). Off by default; output is byte-identical either way.

### Docs

- Windows DeepSeek V4 users are led to the release launcher (`coli.cmd`)
  instead of a source build (#1291).

### Build and CI

- CI workflows run with `contents: read`, and third-party actions are pinned
  by commit SHA (#1354).
- Makefile lists the headers each engine includes as prerequisites (#1284,
  #1349). Site and READMEs carry the eight families with real RAM figures under
  a contract test (#1302).

## [1.10.1] — 2026-08-31

Packaging repair for the prebuilt archives; no engine changes.

## [1.10.0] — 2026-08-31

### A seventh engine: Qwen3.8-Flash-Next

- Added complete text-only inference for the official Qwen3.8-Flash-Next FP8
  checkpoint: four-stream Gated Residual, Gated DeltaNet, Qwen Sparse Attention,
  pageable hashed n-gram embeddings, top-10 routed MoE, and the shared expert.
- The original 131 safetensors shards run directly. Vision and MTP are not
  loaded or advertised; tools and non-text gateway content are refused.
- Added family/planner/doctor/build/release integration and a pinned upstream
  tiny oracle covering sparse selection, cached decode, LRU eviction, and
  sanitizer runs.
- Kept native FP8 expert payloads and normalized scale banks bounded, added
  cache-sized parallel demand loading, and made prompt MoE execution
  expert-major with bounded shared-expert and causal DeltaNet batching.
- Added exact single-slot hybrid prompt-prefix reuse across QSA, DeltaNet and
  PLE state, with persistent state and bounded workspace reflected explicitly
  in planner RAM accounting.
- Added the boundary-only Qwen3.8 Edge adapter and the seventh-family real
  Edge -> Segment -> Edge oracle gate without loading transformer, vision or
  MTP tensors into the Edge process.

## [1.7.0] — 2026-08-19
71 pull requests since v1.6.2. A sixth model family with its GPU tier, a
rebuilt expert-matmul path, and the CI that would have caught the class of bug
we shipped twice.

### A sixth engine: Qwen3.6-35B-A3B — CPU and GPU
- **#712** (@kreuzzelg) — hybrid Gated Attention + Gated DeltaNet + streaming
  MoE, in `c/qwen36.c`. Pre-converted containers published (int4-gs64
  recommended: cosine to the int8 anchor 0.98777 → 0.99313, KL 0.109 → 0.080
  vs per-row). The engine takes any architecture-identical checkpoint
  unchanged — KAT-Coder v2.5 runs on it with no code path of its own.
- **#713** (@kreuzzelg) — CUDA VRAM expert tier with heat-based placement
  across GPUs: **1.44 → 10.05 tok/s (7.0×) on 2× 8 GB cards, output
  bit-identical to CPU** (`cmp` over the full 200-token generation), measured
  cold with no heat table. A `qt_ready()` gate keeps CPU-only builds from
  allocating the packed int4 buffers they never read: **7.33 GB saved**.

### The expert matmul path, rebuilt (all bit-identical)
- **#1071 / #1075 / #1076 / #1077** — activation quantization hoisted to layer
  level across GLM, Kimi K3 and DeepSeek V4: the same vector was being
  re-quantized ~16× per layer, serially. Removes ~5.2 ms/token of serial time
  and every per-call `malloc` from the hot path.
- **#1079 / #1086** — **K1: plane-nibble int4 layout + unsigned-VNNI dot.**
  Storing element *k* and *k+32* in one byte deletes the unpack, and since
  nibbles are stored unsigned, `dot(v,x) = dot(u,x) − 8·Σx` feeds `vpdpbusd`
  natively. 8 uops per 64 MACs instead of 32: **1.45–2.65× on the IDOT
  kernels**, zero bytes added.
- **#1088** — **K2: 1×4 union tile.** The prefill union hands each expert
  2–16 rows; the weight block's load+mask is now paid once per four rows
  instead of once per row: **2.67–3.10× at S=4** (peak 253 GMAC/s).
- **#1093** — parallel silu and the down-side activation hoist.
- **#1094** — **K1b: grouped planar IDOT for gs64 containers** (`IDOT_GS=1`,
  opt-in): the recommended container format could not reach the integer
  kernels at any batch size before this.
- **#1082** (@outtodata) — `FUSED3=1` opt-in fused AVX2 expert matmul.

### Streaming and I/O
- **#1097** — DeepSeek V4 expert-loader pool default 3 → 9 lanes:
  **1.41× decode** on the real V4-Flash checkpoint (8 interleaved runs on a
  quiet 25 GB box). `V4_LOADER_LANES` still overrides.
- **#1056** (@dcutugno) — DeepGEMM sm120 headers fetched at a pinned commit on
  first build: 2.5× prefill on sm120 with nothing vendored in-tree.
- **#988 / #1054 / #1055** — DeepSeek V4 CUDA tier and dual-SSD mirror.

### Correctness and CI
- **#1083** — **ARM CI job** (`ubuntu-24.04-arm`) plus an integer-kernel
  bit-exactness gate that runs on both ISAs. Every tiny-oracle job ran on x86
  before this, so NEON-divergent paths were invisible — which is how the IDOT
  defaults below shipped. Closes #1081.
- **#1044 / #1080** — IDOT made opt-in in olmoe and inkling: the fast path is
  x86-only and quantizes activations, so the same model produced different
  tokens on x86 and ARM by default.
- **#1109** (@SebaWag) — ARM64 dotprod probed by *compiling* the intrinsic:
  GCC 11 defines `__ARM_FEATURE_DOTPROD` for a base it cannot emit
  `vdotq_s32` for. Fixes #1104.
- **#1111** — `__syncwarp()` after `grouped_s4_wmma`'s store (reported by
  @monotophic with `compute-sanitizer` evidence). Fixes #1099.
- **#1073 / #1074** (@bherald) — Kimi K3 cancels prefill between layers
  instead of holding the engine for a minutes-long prompt; cancelled requests
  no longer count as completed.

### Apple Silicon
- **#790 → #1113** (@RDouglasSharp) — **Metal backend for Kimi K3**: KDA state and
  window buffers aligned, wrap-once buffer cache, CPU-side MLA KV cache. 1.7×/2.4×
  on the compute-bound phases (KDA attention + projections dispatched to the GPU);
  MoE experts stay on the CPU. Kimi K3's first GPU backend.

### More correctness fixes
- **#1098** (@monotophic) — `__syncthreads()` missing from the absorb softmax
  reduction, with a determinism test that reproduces the hazard.
- **#1101** (@monotophic) — allocation and `snprintf` results checked on the
  checkpoint-load path (#798).
- **#1100 / #1108** (@monotophic) — fmt=8/fmt=6 scale-byte accounting in
  `tensor_bytes`/`tensor_free`, and `weights_owned` set before the host-to-device
  copy so a failed upload frees its buffer. Each ships with its own regression
  test; all four of this contributor's CUDA fixes landed in this release.
- **#1122** (@ZacharyZcR) — `USAGE_SAVE=0` honoured in every engine (#1039): the
  history was loaded but written back anyway, which quietly contaminated any A/B
  that shared a usage file between arms.
- **#1121** (@ZacharyZcR) — LRU victim selection now respects a lowered `ecap`
  (#1034): after an RSS-guard reduction the cache kept evicting against the old
  capacity.
- **#1123** (@ZacharyZcR) — the v1.6.2 warning-cleanup patches landed (#1032).
- **#1106** (@monotophic) — duplicate tensor names across indexed shards are now
  refused rather than silently resolved to one of them (untrusted containers).

### Interfaces
- **#829** (@aaristov) — GPU-vs-fallback counters and chat status made visible in
  `coli serve`: the tier's behaviour is now observable instead of inferred.
- **#1095** (@benmaster82) — OLMoE planner geometry adapter, and **#1103**
  (@SebaWag) — Kimi K3, Inkling and DeepSeek V4 adapters with 23 tests: every
  family now has real planner geometry, so `coli plan` stops guessing (#1066).
- **#1096** (@terrizoaguimor) — DeepSeek V4 serve framing on the shared codec,
  completing the codec migration across OLMoE, Kimi K3 and V4.
- **#1063 / #1068** (@terrizoaguimor) — model families are registry-owned:
  `coli`, the gateway, `doctor` and the planner read one descriptor table.
- **#1087 / #1090 / #1096 / #1116** (@terrizoaguimor) — a shared serve framing
  codec, now adopted by **every engine**: OLMoE, Kimi K3, DeepSeek V4 and
  Inkling (whose audio payload rides as an opaque extension). Each migration
  landed behind a byte-exact wire-transcript freeze, so the gateway contract is
  provably unchanged. Byte framing had been duplicated five times, which is how
  Windows binary mode silently disappeared from sibling engines (#748).
- **#1036** (@lineape) — distributed expert workers (LAN, opt-in via
  `CLUSTER_WORKERS`).
- Planner: DeepSeek V4 expert naming now recognized, so `coli plan` and
  `coli doctor` stop counting every routed expert as dense (fixes #1110).

## [1.6.2] — 2026-08-14
Security release: **six privately-reported memory-safety issues fixed**, all reachable
from attacker-controlled input (malicious model file / `config.json`, or the kimi_k3
SERVE stdin). Every fix validates at the trust boundary — no behavioural change on
well-formed models or requests. Advisories: GHSA-gf38-c8fx-ppvv (kimi_k3 SERVE OOB
write), GHSA-2qrj-xjmh-mv74 (json.h OOB read), GHSA-w696-h9p7-6rgc (inkling audio
OOB), GHSA-7654-r78q-vc3r (deepseek_v4 indexer OOB R/W), and two more in the same
class. See the GitHub Security Advisories for details.

## [1.6.1] — 2026-08-13
- `--allowed-host '*'` lets an operator reach a public bind deliberately (#990, #993)
- OLMoE streaming no longer drops the answer into the reasoning channel (#984, #985)
- chat reasoning-channel fixes and pty test hardening (#980)

## [1.6.0] — 2026-08-12
**If you are on v1.5.0, update:** it shipped a performance regression on GLM-5.2
(#856), left up to ~60 GB of RAM unused with a 13-point expert hit-rate loss (#885),
and broke Kimi K3 outright on some machines (#888).
- #869 — the planner priced every expert row at the container's *widest* width;
  mixed-width containers had their cache silently halved
- #914 (bherald) — pin budgets accounted correctly
- prefill batch-union: each distinct expert is read **once** per chunk

## [1.5.0] — 2026-08-05
51 pull requests from 15 contributors.
- **Fifth engine: DeepSeek V4 Flash** (@DrewZt, #165) — MLA + DSA sparse attention,
  43 layers, 256 routed experts + 1 shared, top-6; official checkpoint streams with
  no conversion (fp4 experts, fp8-e4m3 dense with UE8M0 block scales)
- #839 — the rows16 fp4 fast path no longer requires AVX-512: consumer Intel/AMD
  CPUs since Alder Lake get the fast path

## [1.4.0] — 2026-08-01
162 commits, 29 pull requests (25 from contributors).
- **Release archives now contain every engine** — v1.3.0 archives shipped `c/colibri`
  alone while the README promised four families (#720); `inkling` and `kimi_k3` are
  built, packaged and smoke-tested per platform
- Third GPU backend lands

## [1.3.0] — 2026-07-29
Three MoE families on one engine, 744B → 2.8T.
- **Kimi K3** (#676) — 2.8T/104B active: KDA + gated-NoPE-MLA + AttnRes + LatentMoE,
  streams Moonshot's QAT MXFP4 experts straight from the original HF shards
- **Inkling** — a 975B model answers on a 25 GB machine

## [1.2.0] — 2026-07-28
- GB10 / DGX Spark unified-memory OOM fix (#653); AMD/ROCm recognized by `doctor`
  and `resource_plan` (#662, #663)
- AVX2 `matmul_e8` — fmt=6 was 92% of decode on the scalar kernel (#654); native
  SIMD fmt=6 encoder, 15× over numpy
- chat stop-set fix (#633/#381), async packed-int4 parity (#632), stable KV slot
  per conversation (#634, #639)

## [1.1.1] — 2026-07-23

A same-day patch release. **Windows users on v1.1.0 should upgrade**: Microsoft
Defender flags the v1.1.0 Windows binary, and the cause was ours.

### Fixed

- **107 KB of zeros were shipped inside every binary** (#527, #532) — and that is
  what antivirus ML heuristics were reacting to. `static GrDraft g_grd={.max=24};`
  looks harmless, but `GrDraft` is ~107 KB (the grammar's 1024 static rules plus the
  PDA walker) and **any** initializer moves the whole struct out of `.bss` and into
  `.data`, writing 106,848 bytes of near-zero-entropy data into the file — in a
  *writable* section, which is the classic shape of an unpacking buffer for a packed
  payload. Section forensics against v1.0.0 (clean on the same Defender definitions)
  isolated it: identical toolchain, identical PE layout, `.data` 1,840 → 108,752
  bytes. A Windows build with the fix scans clean where v1.1.0 does not. Every
  platform's binary also gets smaller: the Linux engine drops 474,904 → 368,016
  bytes, **-22.5%**.
- **`python3 openai_server.py` was broken on a clean checkout** (#526) — the gateway
  still looked for an engine named `glm` after the #391 rename. It resolves
  `colibri`/`colibri.exe` first now, falling back to `glm` for older trees.
  `coli serve` was unaffected. Spotted by @RDouglasSharp while debugging #488.

### Added

- **Anthropic Messages API** on `/v1/messages` (#343, #525) — clients that only speak
  to Anthropic endpoints, Claude Code above all, now work against colibri with no
  shim: same port, nothing to enable. It is a translation layer over the same
  generation path, so tools, streaming and the KV cache behave exactly as on
  `/v1/chat/completions`. Covers system prompts, `text`/`tool_use`/`tool_result`
  blocks, `input_schema` tools, every `tool_choice` mode, the full named-event SSE
  sequence with protocol `ping` keepalives, `stop_reason` mapping, extended thinking,
  and `x-api-key` auth (`Bearer` still works). `stop_sequences`, `top_k` and non-text
  blocks are refused explicitly rather than silently ignored.
- **`SHA256SUMS.txt` published with every release** (#530) — verify a download is
  exactly what CI built from the tagged source.
- **The Windows engine is uploaded as a CI artifact** (#532) — an antivirus report can
  now be verified on a pull request instead of only after a release is published.

### Changed

- **Docs: "Get started" now starts by getting the program** (#521). The README told
  newcomers to download the 372 GB model *before* it told them how to obtain colibri —
  and for Linux/macOS it never told them at all. New order: get colibri (prebuilt
  archive or build from source) → get the model (372 GB stated up front) → run it.
  The obsolete "rename the engine to `glm.exe`" step is gone; archives have shipped a
  plainly-named `colibri.exe` since #508. Applied in all four languages.

## [1.1.0] — 2026-07-22

A community release. 27 pull requests from more than 20 contributors, 216 commits since
v1.0.0. Most of what follows was found, measured, or fixed by people who do not work on
this project and had nothing to gain from it.

### Added

- **AMD GPU support (HIP/ROCm)** (#339) — single-source `backend_gpu_compat.h` with a WMMA
  dispatch gate, so one codebase builds for CUDA and HIP. Validated on an RX 9070 XT
  (RDNA4, ROCm 7.2): token-exact against CPU on a real fmt=4 gs64 container, with resident
  dense *and* with routed experts in VRAM, plus a fail-injection control proving the GPU
  actually executed the work.
- **Dual-SSD streaming** (`COLI_MODEL_MIRROR`, #421) — read the model from two drives at
  once, roughly doubling streaming bandwidth on a disk-bound host.
- **N-drive shard split** (`COLI_MODEL_DIRS`, #469) — capacity aggregation: run a container
  no single drive can hold, spread across several with no duplication.
- **fmt=5 (int3-g64)** (#168) — 3-bit weights with per-64 group scales: measured 3.3x lower
  outlier-row error than per-row int4 at 25% fewer bytes.
- **fmt=6 (E8/IQ3 lattice)** (#465) — CPU decode kernel and dispatch; index codec tooling (#458).
- **`tools/try_tool_calling.py`** — dependency-free two-turn tool-calling probe that doubles
  as a smoke test.

### Fixed

- **Tool calling in coding clients** (#401), root cause found, in two parts:
  - **#506** — the engine capped prompt encoding at `CTX-2`, and the tokenizer stops dead at
    its limit *without reporting anything*. A prompt longer than the context was therefore
    silently truncated to its first `CTX-2` tokens and answered anyway. With the 4096 default
    that is 4094 — exactly the `prefill 4094` in the field report. The dropped tail was the
    tool instructions and the user's actual turn, so the model emitted a bare `<` and stopped;
    and because clients append to the *end* while truncation keeps the *head*, every retry
    re-sent a byte-identical prompt. Now refused with a 400 `context_length_exceeded`.
  - **#505** — a tool call whose closing `</tool_call>` never arrived was dropped whole,
    because the parser required both tags. Now recovered when unambiguous, on both the
    streamed and non-streamed paths.
  - **#437** — non-EOS role markers were armed as hard stops in serve mode and cut generation
    the instant a tool block started.
- **Grouped-int4 (fmt=4) produced garbage output on CUDA** with `CUDA_DENSE=1` (#298) — the
  dense and attention kernels applied per-group scales as if they were per-row. Hardware-verified.
- **OpenMP tuning re-exec preserved the CPU affinity mask** (#476), jailing every thread onto
  one core when `OMP_PROC_BIND`/`OMP_PLACES` were set: roughly a 20x slowdown.
- **Pilot eviction guard dropped ~100% of speculations** once the cache filled (#497),
  collapsing `PILOT_REAL` to a hint-only path.
- **Silent budget clamp** capped the CUDA expert tier at ~109 experts regardless of
  `CUDA_EXPERT_GB` (#495).
- fmt=4 guard at the per-row-only CUDA entry points (#464/#470); `COLI_CUDA_MTP=1` and
  `COLI_CUDA=0` are now honoured over implicit defaults (#468).

### Security

Threat model: model files come from mirrors that are not trusted.

- **#368** — server hardening, JSON and tokenizer parser hardening, build flags, downloader
  and dependency pinning.
- **#413** — the quant layout is resolved *and* validated against the on-disk byte counts
  (unknown layouts are refused rather than falling through to int2), shape-product overflow
  is rejected, and the olmoe dtype-3 path no longer trusts a crafted `nbytes` (heap overflow).

### Performance — all byte-identical

- **#481** +4.7x on the MLA-absorb score and value-mix reductions
- **#477** +13% decode on AVX-512 (`qt_addrow` / `qt_matvec_rows`)
- **#475** +11.6% with opt-in `XEXP=1` (one OpenMP region per expert block at S=1 full residency)
- **#473** +5.5% int4 IDOT at S=1 on AVX-512 VNNI

### Changed

- `glm.c` is now `colibri.c` plus header modules (#391); `make glm` remains as an alias.
- Serve stage 2 (#192): `response_format`, per-request grammars, grammar-forced drafts.

### Upgrade notes

- **`CTX` still defaults to 4096.** Coding clients send far more than that in a single system
  prompt. Use `CTX=32768`. Before this release an over-long prompt was silently truncated;
  now you get a clear 400 instead.

## [1.0.0] — 2026-07-19

First tagged release. The engine has been running in production since late June
2026; this tag marks the baseline for semantic versioning going forward.

### Highlights

- **GLM-5.2 (744B MoE)** runs on ~25 GB RAM in pure C, streaming experts from disk
- **Three-tier placement**: VRAM (hot) / RAM (warm) / NVMe (cold), with a learning
  cache that pins your workload's hottest experts automatically
- **CUDA backend**: multi-GPU expert tier, dense tensor distribution, batched
  ragged attention, resident pipeline (`COLI_CUDA_PIPE=2`)
- **Metal backend** (Apple Silicon): batched expert SwiGLU + fused decode attention
  on unified memory GPU
- **MTP speculation**: native GLM-5.2 draft heads, grammar-forced drafts, kernel-
  pinned verification (`SPEC_PIN=1`)
- **OpenAI-compatible API**: `coli serve` with SSE streaming, KV slots, bounded
  queue, web dashboard (`coli web`)
- **Web UI**: chat with live metrics, expert cortex brain page, profiling breakdown,
  expert atlas 3-D galaxy
- **Cross-platform**: Linux, macOS, Windows 11 (native MinGW), PowerPC; CI on all three
- **Auto-tune**: `coli plan --auto-tier` classifies the bottleneck and derives
  MTP/PIPE/NUMA/PIN settings with explanations

### Engine

- Token-exact validation against `transformers` oracle (teacher-forcing 32/32)
- Compressed MLA KV cache (576 floats/token, 57× smaller), persisted across
  restarts (`.coli_kv`, zero re-prefill)
- DSA sparse attention (lightning indexer), faithfully implemented
- Router-lookahead prefetch (`PILOT=1`, 71.6% predictive)
- Async expert I/O pool (`PIPE=1`), io_uring batching (`URING=1`)
- NUMA-aware expert placement (`COLI_NUMA=1`, +13–40% on multi-socket)
- AVX2 / AVX-512 / AVX-VNNI / ARM NEON / NEON-i8mm / POWER VSX kernels
- int4 / int8 / int2 / grouped-int4 (fmt=4) quantization formats

### Tools

- `coli convert` — FP8→int4 one-shard-at-a-time converter
- `coli doctor` — read-only setup diagnostics
- `coli plan` — resource planner with auto-tune prescription
- `coli bench` — MMLU / HellaSwag / ARC quality benchmarks
- Expert atlas (`tools/analyze.py --web`) — measured topic affinity for 19,456 experts

### Community

- 30+ hardware datapoints in the benchmark tracker
- Contributions from 20+ authors across engine, docs, tooling, and ports
