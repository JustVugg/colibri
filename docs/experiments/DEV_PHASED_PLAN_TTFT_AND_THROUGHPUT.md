# Colibri Development Phased Plan: TTFT, Prefill and Throughput

**Status:** Active development plan
**Target branch:** `feature-IQ2_XXS`
**Prepared:** 2026-11-10
**Baseline inspected:** `bc4b92509b214687423294a1ee320d3933faa018`
**Scope:** Interactive and agent inference performance, primarily Qwen3.6 35B-A3B MoE and 27B dense.

> Implementation notes and observations are recorded in Section 7. A completed instrumentation change is not a performance result; real-model baselines are still required.

## 1. Objective

Move Colibri from being primarily decode-throughput optimised to being **interactive-agent optimised**.

For Hermes, Brewer and Copilot, the first response matters. A model that generates quickly after a long silent prefill can still feel slow.

The programme has four objectives:

1. Reduce time-to-first-token (TTFT), especially for long prompts.
2. Avoid reprocessing conversation history when existing model state can be reused safely.
3. Use available RAM deliberately without causing memory pressure or swap.
4. Preserve decode throughput while evaluating model and quantisation alternatives.

Initial prompt sizes: **1K, 4K, 12K, 32K and 66K tokens**. These are workload points, not a claim that every model/configuration supports 66K. Verify each model's effective context limit first.

## 2. Baseline and known evidence

The following are historical observations in `logs/DEV_PLAN_10TO15_TOKS.md`, not fresh measurements from this plan.

### Qwen3.6 35B-A3B IQ2_XXS

- 40 layers, 256 routed experts, top-8.
- C11 fused ARM IQ2/IQ3 kernels reduced recorded MoE block time from about 42.7 ms to 28.4 ms.
- Recorded plain decode after C11: approximately 14.43–14.59 tok/s.
- Recorded MTP-on decode: approximately 12.99–13.59 tok/s, with D1 the best tested draft depth.
- MTP acceptance was useful, but draft-head cost became a net loss after the main model's verify path became faster.
- Current practical baseline: MTP off/unset unless a new end-to-end test proves otherwise.

### Qwen3.6 27B dense IQ2_XXS

- Dense model, not MoE.
- Branch log records a 9.56 GB model and a working native GGUF MTP head.
- CPU and Metal results are substantially below the 35B MoE decode result.
- Still valuable as a controlled workload for dense prefill, Metal GEMV and MTP-head experiments.

### Existing mechanisms to build on

- `kv_prefix.h`: records token IDs actually fed into the engine and permits exact-prefix reuse where engine state is valid.
- `c/tools/benchmark_http_serving.py`: streaming benchmark with client-observed first-output distributions (p50/p95/p99), request duration and optional arrival-rate/SLO data.
- `prompt_break.h` and Qwen36 prompt control: safe prompt-stop boundaries.
- C3 mmap expert access: file-backed expert weights with OS read-ahead/discard hints.
- Latest branch commit adds opt-in `Q36_HYBRID_GB`: selected hot experts can be promoted to owned RAM copies while cold experts remain mmap-backed. This is new and has no performance conclusion yet.

**Measurement distinction:** client-observed first-output latency is not a complete diagnosis of engine TTFT. It includes some combination of queueing, HTTP, tokenisation, prompt preparation, prefill, logits, sampling and stream delivery. Gateway timings and Qwen3.6 tokenization, prefix/snapshot, prefill-operation, first-sample and first-text-emission measurements are implemented. The C12 workload matrix and real-model baseline collection remain outstanding.

## 3. Programme principles

1. Measure before changing kernels.
2. Optimise real agent sessions, not only isolated prompts.
3. Separate cold, warm and prefix-reuse behaviour.
4. Preserve correctness. KV reuse and recurrent-state changes require explicit validation.
5. Treat memory as a budget, not a guaranteed physical reservation. macOS may compress memory or reclaim file-backed pages.
6. Keep model comparisons fair: same prompt tokens, context settings, sampling, output limit, warmup and concurrency.
7. Use repeated runs and report distributions.
8. Do not regress established decode performance for a synthetic prompt win without an explicit trade-off decision.

## 4. Phased roadmap

### C12 — TTFT instrumentation and long-prompt baseline

**Priority:** P0 · **Status:** Gateway and Qwen3.6 stage timings implemented; one-request real-model pilots completed at 1K, 4K, 12K, 32K and 64K; repeated representative baselines, cold/warm/restored-prefix separation and instrumentation-overhead measurement remain pending

**Goal:** Identify where time is spent between request arrival and the first streamed token.

Add low-overhead, request-correlated timings for:

- Request accepted / queue wait.
- Tokenisation and chat-template processing, where applicable.
- Prefix matching and reusable-token count.
- Prompt embedding / input preparation.
- Prefill total time, token count, chunk count and per-chunk duration.
- Final normalisation / logits computation.
- First-token sampling and first non-empty text-frame emission after prefill.
- End-to-end client-observed first-output time.

Keep server timings separate from client timings. Do not log prompt contents, secrets or tool payloads.

Extend the existing benchmark workflow. Store JSON results with model identity, quantisation, context length, build/commit, environment, prompt token count, workload hash and cold/warm state.

**Workload matrix**

| Prompt length | Purpose |
|---:|---|
| 1K | Small interactive request |
| 4K | Ordinary conversation |
| 12K | Common coding-agent context |
| 32K | Long engineering session |
| 66K | Heavy agent context, only where supported |

For each size, test: cold process/first request; warm process/first request; same conversation prefix plus a short appended turn; changed system prompt or deliberately divergent prefix; and repeated realistic session turns.

Start at concurrency 1. Add realistic concurrency after single-request timing is understood.

**Deliverables:** stage-level TTFT timing; reproducible prompt workload files and hashes; baseline JSON reports for both reference models where context permits; bottleneck report by prompt size.

**Acceptance:** timings reconcile with client-observed first-output latency within documented overhead/tolerance; p50/p95/p99 reported across repeated runs; cold, warm and prefix-hit cases remain separate; instrumentation overhead measured.

### C13 — Conversation prefix reuse validation and optimisation

**Priority:** P0 after C12 · **Status:** Telemetry implemented; real-model validation passed for disk-restore, live-reuse and divergent cases (2026-11-11), the three-fix phase-3 sequence, and the phase-4 truncation/reset/4K-repeat/append cases (2026-11-12). Code preserves informative live-state miss labels, supports exact-repeat disk restore using saved logits, lowers the default snapshot threshold to 256 tokens, and prunes by mtime with restore-touch LRU under both a file-count cap and a total-bytes budget (`COLI_PROMPT_CACHE_GB`, default 8). The macOS build and targeted fixture-backed runtime tests pass (16 passed, 1 skipped); a MinGW-w64 cross-build from macOS also produced a verified PE32+ `qwen36.exe`. This verifies compilation/linking, not execution on Windows. Multiplexed serving (`KV_SLOTS=2`) and the byte-budget retention are validated on the live stack (2026-11-12, phase 5). Remaining cases include the native Windows runtime test.

**Goal:** Avoid processing prompt tokens whose model state is already available and valid.

1. Audit `kv_prefix.h` integration in Qwen36 serving and other relevant engines.
2. Measure exact-prefix hit rate, reused token count/percentage, time saved, and miss reasons.
3. Validate that prefix records always describe tokens actually fed into the model.
4. Preserve the conservative recurrent-state rule: do not resume from an arbitrary partial common prefix unless the exact recurrent state for that position can be restored.
5. Evaluate stable prompt construction for agent clients. Avoid dynamic timestamps or changing system text in reusable prefixes where practical, without changing application semantics.
6. Consider snapshots/checkpoints only if measurements show exact-prefix reuse is insufficient and state cost is justified.

**Deliverables:** prefix hit/miss telemetry without prompt-content logging; agent-session reuse benchmark; correctness tests for append, divergence, truncation, reset, context growth and recurrent state; prioritised fixes based on measured miss causes.

**Acceptance:** reused state always matches the exact recorded prefix; divergent/tainted inputs never reuse invalid state; realistic sessions measurably reduce prefill work where expected; token output correctness is preserved.

### C14 — Budgeted hybrid RAM residency

**Priority:** P1 · **Status:** Initial implementation exists; benchmark and harden

**Goal:** Use user-approved memory headroom to keep frequently used routed experts in owned RAM copies while retaining mmap-backed access for cold experts.

The latest commit introduces `Q36_HYBRID_GB`. It estimates a per-layer hot-expert count from a GB budget, collects route frequency and promotes selected mmap-backed experts to owned copies. Cold experts remain mapped.

This is promising, but code landing is not proof of a speedup.

Test budgets **0, 1, 2, 3 and 5 GB** using the same prompt/session workload. Record:

- Actual resident memory and memory footprint.
- Compressed memory, swap growth and memory pressure.
- Expert hit/miss and promotion counts.
- Expert read/promotion time.
- Decode tok/s, prefill tok/s and TTFT.
- Stability over a sustained multi-turn session.

Keep MTP-head memory accounting separate from main-model expert residency. The MTP expert slab previously measured about 278 MB in the recorded 35B configuration; this does not justify an arbitrary fixed reserve without measuring all associated weights and buffers.

Review promotion/eviction for concurrency safety, pinned-slot limits, allocation failure and fallback behaviour. Confirm the budget calculation cannot silently exceed its requested allowance.

**Design direction:** expose a clear user-configurable upper budget, not a hard RAM lock; preserve OS/application headroom; promote hot experts only when measured reuse justifies it; observe MTP weights/buffers separately; avoid aggressive prefetch that causes cache pollution.

**Acceptance:** retain a budget only if repeated measurements improve a target workload without unacceptable TTFT, decode, memory-pressure or swap regressions. 5 GB is a test point, not an assumed default.

### C15 — Long-prompt prefill acceleration

**Priority:** P1 after C12 diagnosis · **Status:** Proposed, profile-guided

**Goal:** Reduce long-prompt processing time without harming decode or recurrent-state correctness.

Investigation order:

1. Use C12 timings to separate tokenisation, embedding, attention, DeltaNet/recurrent work, MoE, dense projections and output-head costs.
2. Identify compute, bandwidth, expert loading, scheduling or chunk-size bottlenecks.
3. Evaluate prompt chunk sizing and batching. Larger chunks are not automatically better; recurrent state and memory footprint constrain useful sizes.
4. Inspect ARM fused-row kernels for prompt batches, not just single-token decode.
5. Evaluate Metal only where shapes amortise transfer/dispatch overhead. The sparse MoE Metal path previously lost to CPU in recorded tests; do not generalise this to dense layers or every prompt shape.
6. Re-run correctness/parity tests after recurrent-layer batching or state changes.

**Deliverables:** per-operation prefill profile; narrowly scoped kernel/scheduling experiments; before/after results at 4K, 12K, 32K and supported 66K; decode regression report.

**Acceptance:** repeatable long-prompt TTFT or prefill improvement, preserved output/state correctness and documented single-token decode effect.

### C16 — Q4_K_M quality/performance comparison

**Priority:** P2 · **Status:** Proposed

**Goal:** Determine whether Q4_K_M offers a worthwhile quality, memory and speed trade-off against IQ2_XXS on target hardware and agent workloads.

The branch's GGUF type table supports Q4_K, Q5_K and Q6_K tensor types. Q4_K_M is a mixed-quantisation recipe; individual tensors may use different types. The Qwen36 loader accounts for mixed expert tensor types. This does not prove every Q4_K_M layout is fully supported or optimised. Validate the exact candidate GGUF metadata and tensor types first.

Compare same architecture/family where possible:

- Load success and tensor coverage.
- Model size and resident memory.
- Prefill throughput and TTFT.
- Decode throughput.
- Expert read volume and cache behaviour.
- Output quality on a fixed agent-task evaluation set.
- Long-context stability.

The current ARM fused-row path explicitly accelerates IQ2_XXS and IQ3_XXS. Q4_K currently follows a different generic path in the inspected dispatch. Consider a dedicated ARM Q4_K kernel only if profiling shows a material bottleneck and the model comparison justifies it.

**Acceptance:** select quantisation based on measured quality/performance, not bits-per-weight or a single tok/s result.

### C17 — 27B dense model and C8 Metal investigation

**Priority:** P2 · **Status:** Candidate experiment

**Goal:** Use Qwen3.6 27B dense as a controlled workload for dense GEMV, MTP head and Metal-path experiments.

1. Re-establish clean CPU and Metal baselines after C12 instrumentation.
2. Measure prefill and decode separately.
3. Inspect which dense operations dispatch to Metal and which remain on CPU.
4. Revisit C8 IQ2 Metal GEMV for the dense MTP head, measuring dispatch overhead, transfer cost and row shape.
5. Measure MTP on/off with identical prompts and sampling. Acceptance rate alone does not prove benefit.
6. Compare with 35B MoE without conflating architecture, quantisation or workload.

**Acceptance:** proceed with C8 or related Metal work only if the dense workload has a repeatable end-to-end gain. A kernel-level win without request-latency improvement is insufficient.

## 5. Cross-phase test protocol

Every report should include:

- Git commit SHA and build flags.
- Machine/OS and thermal/power state where available.
- Model filename, architecture, quantisation and context limit.
- Prompt token count and workload hash.
- Cold/warm state and prefix hit/miss.
- Memory budget and relevant environment variables.
- Concurrency and request-arrival pattern.
- TTFT p50/p95/p99.
- Prefill tok/s and total prefill duration.
- Decode tok/s.
- Peak/steady resident memory, compressed memory and swap delta.
- Correctness/parity outcome.

Use repeated runs and report distributions. Keep warmup separate. Never compare a cold run with a warm run as if they were equivalent.

### Deterministic agent-session workload

Maintain at least one multi-turn workload resembling an agent:

1. Large stable system prompt.
2. Initial user request.
3. Assistant analysis/tool-call response.
4. Tool result.
5. Assistant response.
6. Follow-up user request.
7. Repeated turns with both stable and deliberately changed prefixes.

This is the primary test for the combined value of TTFT, prefix reuse and RAM residency.

**Implementation status (2026-11-12):** the phase-5 retention driver (`logs/logs/colibri-c12-workloads/c13_prefix_phase5.py`, mode `retain`) implements this shape against the live stack — three multi-turn sessions with a stable opening prompt, appended follow-ups, and deliberate cross-session divergence — and every assistant turn replays the engine's own generated text, which the 2026-11-12 finding makes mandatory for the live-reuse path to be exercised at all. Extension to system-prompt variants and tool-result turns is still open.

## 6. Development observations and progress

### 2026-11-10 — Initial serving observations

The supplied Qwen3.6-35B-A3B IQ2_XXS serving log covers 36 turns. It shows the rendered prompt growing from about 1.3K to 23.2K tokens while the engine saves prompt-cache snapshots across turns. A later request reports restoring 1,265 of 8,411 prompt tokens from the disk prompt cache, so snapshot reuse is present but is not yet demonstrated to cover the full live conversation prefix. The speculative lookup path reports roughly 1.00–1.39 tokens per forward in this sample; this is not a controlled decode tok/s comparison and does not establish a speedup or regression.

The logged T(1) component ranges from about 62 ms to 344 ms as turns progress. It is only one engine component, not end-to-end TTFT. The excerpt has no corresponding client-observed TTFT distribution, prefill duration, prompt-cache source/miss reason, or controlled decode-throughput measurement. No kernel, speculation, or residency change should be justified from this run alone.

### 2026-11-10 — C12 initial instrumentation

- The HTTP benchmark retains client-observed first-output latency and full-batch completion-token throughput, and also collects server-provided stage timing metadata without recording prompt contents.
- Colibri chat/completion responses now report request preparation, scheduler queue wait, engine submission-to-acceptance, acceptance-to-first-output, engine-call duration, request-to-engine-done, and decode tokens/sec. Streaming metrics are attached to the final completion chunk; non-streaming metrics are top-level.
- Qwen3.6 reports tokenization, live-prefix matching, snapshot lookup/restore/store, fresh-prefill wall time and prompt/reused-token counts. Snapshot-store duration is measured separately because storage happens after prefill and before engine TTFT timing starts. Fresh-prefill operation accumulators cover DeltaNet, attention, MoE, shared/router subsets and LM head; the first sampling call is timed independently, and elapsed time to first non-empty text emission after prefill is reported. The emission interval can include sampling, decoding and UTF-8 buffering. Operation accumulators are diagnostic, not additive wall-time parts.
- Benchmark summaries and baseline comparisons expose client and server timings separately, including snapshot-store, first-sample and post-prefill emission distributions. API and baseline documentation describe their measurement boundaries.
- The engine embeds the prompt rows before its layer forward. The CPU path processes the fresh prompt in one host pass; Vulkan/CUDA paths split rows into backend chunks. The follow-up instrumentation now reports embedding duration and chunk count plus average/maximum chunk duration.
- Qwen3.6 `KV_SLOTS>1` multiplexed serving now emits the same stage timing fields as single-slot serving, including first-sample and first-text-emission timings. Focused validation passed after this change: `cd c && make qwen36` (CPU build); `cd c && PYTHONPATH=. python3 -m pytest tests/test_openai_server.py tests/test_benchmark_http_serving.py tests/test_benchmark_baseline.py tests/test_qwen36_prefix_serve.py -q` (517 passed, 646 subtests, 12 fixture-dependent skipped); and `git diff --check`. Multiplexed runtime behavior was not independently exercised by the available fixture; Vulkan/CUDA chunk instrumentation has not been compiled or exercised on this host.

### 2026-11-10 — C12 first real-model timing pilot

- Ran the built serving stack with Qwen3.6-35B-A3B IQ2_XXS, `OMP_NUM_THREADS=6`, context 66K, concurrency 1, greedy sampling, and 128 generated tokens. The API was bound to loopback; `COLI_PROMPT_CACHE=0` prevented this matrix from restoring or writing prompt snapshots. The 1K request was the first inference after server startup; 4K and 12K followed in the same process. Prompt text was synthetic, length-controlled, and not a representative coding-agent session. Qwen3.6 speculative lookup remained at its default setting, so these are exploratory observations, not the clean baseline.
- One request per size completed successfully. Actual token counts differ slightly from the target labels:

  | Target | Prompt tokens | Client first output (s) | Engine TTFT (s) | Prefill (s) | Decode (tok/s) |
  |---:|---:|---:|---:|---:|---:|
  | 1K | 1,045 | 34.326 | 34.313 | 34.304 | 14.30 |
  | 4K | 4,105 | 172.503 | 172.489 | 172.475 | 9.43 |
  | 12K | 12,231 | 785.007 | 784.993 | 784.941 | 6.92 |

- Client-observed first output exceeded engine TTFT by 12–14 ms across these three requests. Prefill dominated TTFT; the operation diagnostics reported MoE as the largest component at 1K/4K, while attention rose to 263 s at 12K. Operation timers overlap and are not additive. Engine decode rate was measured over 128 output tokens; it is a single-request observation, not a repeated sustained-throughput distribution.
- After adding embedding/chunk timing, one more 1,045-token request ran with `COLI_LOOKUP=0`: client first output 35.212 s, engine TTFT 35.201 s, prefill 35.191 s, embedding preparation 0.000565 s, one CPU layer chunk averaging/maxing 35.160 s, and decode 14.01 tok/s. This validates the new fields through a real CPU-path response; it is one observation and not an instrumentation-overhead comparison.
- The 12K request took 13.4 minutes. Its attention timer reached about 263 s, making attention a plausible long-prompt investigation target, but one synthetic run cannot select a specific optimization: operation timers overlap, and there is no controlled repeated baseline or request-aligned memory capture. A helper now runs prepared 32K and 64K workloads sequentially into separate reports; those lengthy runs are deferred, not prerequisites for identifying this initial direction.
- No speedup is established. Remaining C12 work: controlled repeated p50/p95/p99 runs, a representative prompt set, GPU chunk-field verification where applicable, optional 32K/64K scaling reports, cold/warm/restored-prefix separation, memory aligned to TTFT/decode, and instrumentation-overhead measurement. Preserve the JSON reports and hashed workloads in session artifacts; they are not checked into the repository.

### 2026-11-10 — C13 prefix telemetry in progress

- Qwen3.6 serving statistics now distinguish live state, pin snapshot, disk snapshot, and no reuse; misses are classified with bounded reason labels such as disabled, tainted state, divergent prefix, non-extension, unavailable sliding window, or no matching disk snapshot.
- Gateway and benchmark paths preserve and summarize those labels. They carry token counts and controlled enums only; they do not include prompt text.
- Next validation: compile and run the new protocol/parser tests, then run a deterministic multi-turn Qwen3.6 session with stable-prefix, appended-turn, divergent-prefix, and restart/cold cases. Compare prompt/reused-token counts and prefill time while checking generated output against cold-prefill references. The tiny fixture is currently unavailable, and no real-model result should be claimed until that workload is run.

### 2026-11-11 — C12 32K/64K scaling run attempt

- The prepared `prompt-32k.jsonl` and `prompt-64k.jsonl` workloads are on disk under `logs/logs/colibri-c12-workloads/`. An earlier helper attempt at 05:27 produced `prompt-32k.json` / `prompt-64k.json` reports in `benchmarks/`, but all requests failed with HTTP 401 (missing API key); no model inference occurred.
- Server relaunched for this attempt (same config as the pilot: Qwen3.6-35B-A3B IQ2_XXS, `OMP_NUM_THREADS=6`, `--ctx 66000`, loopback port 8001, concurrency 1, greedy, 128 output tokens). Difference vs the pilot: `COLI_PROMPT_CACHE=0` is not set this time, so snapshot store/restore is enabled. Both long prompts are fresh text, so each first request should still be a cold prefill for that prefix.
- Rerunning `tools/benchmark_long_prompts.py` with `--repeats 1` into a fresh `benchmarks-scaling/` output directory (separate `prompt-32k.json` and `prompt-64k.json` reports). Single observations matching the pilot style; repeated p50/p95/p99 distributions remain a separate pending C12 item. Results will be recorded here when the run completes.
- Two helper iterations failed before inference started, both instantly: first with HTTP 401 (no API key supplied; all 5 requests failed), then with HTTP 404 `model_not_found` because `--model` must be the server's advertised model ID (`qwen3.6-colibri`, per `GET /v1/models`), not the GGUF path used in `serve --model`. A correct run is now in progress with `--model qwen3.6-colibri`.
- The correct run completed. Both requests succeeded, one observation per size, same server/process, 32K first then 64K, `repeats=1`, greedy, 128 output tokens. Wall time was 3,391 s (32K) and 11,892 s (64K).

  | Target | Prompt tokens | Client first output (s) | Engine TTFT (s) | Prefill (s) | Prefill tok/s | Decode (tok/s) |
  |---:|---:|---:|---:|---:|---:|---:|
  | 32K | 32,012 | 3,358.641 | 3,358.627 | 3,357.184 | 9.54 | 3.98 |
  | 64K | 64,012 | 11,845.178 | 11,845.139 | 11,837.822 | 5.41 | 2.75 |

- Client-observed first output exceeded engine TTFT by 14 ms (32K) and 38 ms (64K), continuing the 12–14 ms reconciliation seen at 1K–12K. Prefill again dominated TTFT. Prefix reuse was 0 with miss reason `no_matching_disk_snapshot` for both (fresh prompts, snapshots written after prefill: 1.38 s at 32K, 7.23 s at 64K; snapshot restore 0.0026 s / 0.0022 s). Tokenization (0.054 s / 0.112 s), embedding (0.046 s / 0.056 s, one CPU layer chunk each), queue wait and first-sample (≈20 µs) and first-emission (≈0.3 ms) costs are negligible at these sizes.
- Prefill operation accumulators (overlapping, not additive): 32K = attention 1,852 s, MoE 1,186 s, DeltaNet 309 s, shared 40 s, router 26 s; 64K = attention 8,352 s, MoE 2,749 s, DeltaNet 712 s, shared 89 s, router 68 s. Attention grows from ~34% of the recorded components at 12K to 55% at 32K and 71% at 64K, and per-token attention time rises 21.5 → 57.9 → 130.5 ms across 12K/32K/64K, consistent with a quadratic component. MoE per-token time is roughly flat between 32K (37.1 ms) and 64K (42.9 ms) — MoE prefill scales near-linearly with tokens, attention does not.
- Combined with the pilot, prefill cost per token is now superlinear across the full matrix: 32.8 / 42.0 / 64.2 / 104.9 / 184.9 ms per token at 1K/4K/12K/32K/64K (30.5 / 23.8 / 15.6 / 9.5 / 5.4 prefill tok/s), and decode tok/s falls 14.30 / 9.43 / 6.92 / 3.98 / 2.75 as context grows. Long-prompt TTFT is now measurably attention-dominated, which upgrades attention prefill batching/chunking to the leading C15 target; this is still one synthetic observation per size with speculation default-on and no memory capture, so it selects a direction, not a kernel change.

### 2026-11-11 — C13 first real-model validation sequence

- Ran a 7-request deterministic sequence against the live serving stack (same config as the 32K/64K run: snapshots enabled, ctx 66K, greedy, 48 output tokens) with prompts of 516–1,580 tokens. Driver and raw rows are session artifacts (`logs/logs/colibri-c12-workloads/c13_prefix_sequence.py`); prompt text is synthetic and hashed, not logged.
- **Disk-snapshot extension reuse works end to end:** the turn-3 request (1,580 tokens) restored 1,048 tokens (66.6%) from the turn-2 request's disk snapshot (`source=disk_snapshot`, restore 0.123 s) and prefilled only the 532-token tail (28.1 s, 52.7 ms/token — consistent with fresh-prefill rates at this size). This is the first measured prefill-work saving from the prefix path on the real model.
- **Original exact-repeat limitation and label masking:** the 2026-11-11 sequence showed exact repeats paid full prefill and the disk-cache branch masked `prompt_not_longer` as `no_matching_disk_snapshot`. Subsequent code changes now allow full-length disk snapshot matches, return saved logits for zero-prefill resume, and preserve informative live-state miss labels when disk lookup fails. Fixture-backed tests now verify exact-repeat output parity against cold prefill and preserve `prompt_not_longer` with disk caching enabled.
- **Short-prompt snapshot policy changed:** the original default minimum of 1,024 tokens meant the 516-token requests stored no snapshot. The current code lowers the default to 256 tokens (with an explicit minimum clamp of 64); prompts below 256 still do not get a default disk snapshot. Fixture-backed tests confirmed that 200-token prompts do not store snapshots and 256-token prompts do. Validation against real interactive workloads and storage/latency costs remains pending.
- **Output correctness in phase 1:** identical prompts produced identical greedy outputs on cold repeats (r2==r1, r4==r3, sha256 prefix match), and divergent prompts produced different outputs. The follow-up fixture-backed exact-repeat test now confirms the disk-restored output matches a cold-prefill reference.
- A mid-divergence case (r7) reported no reuse, but the probe was flawed: after r5 the live state held 1,580 tokens and r7's prompt was only 1,044, so it hit the same no-rewind rule rather than exercising `divergent_prefix` classification. A follow-up must send a longer-than-state prompt that diverges early.
- Prefill cost per token drifted upward across the sequence (34.7 ms/token on the first 516-token cold prefill vs 55.2 ms/token on a later 516-token one). Single observations with an intervening larger context; no conclusion — consistent with the unexplained variance the repeated C12 baselines are meant to bound.

### 2026-11-11 — C13 restored/reused-state correctness (phase 2 and 2b)

- Restarted the serving stack with `COLI_PROMPT_CACHE=0` (no disk snapshots, empty live state) to obtain cold-prefill references for the phase-1 restored runs. Same build, sampling and prompt text as phase 1.
- **Restored-state output equality: PASS.** A cold prefill of the phase-1 turn-3 prompt (1,580 tokens, 52.4 s) produced output identical (sha256 prefix match) to phase-1 r5, which had served it from a disk-snapshot restore in a different process with snapshots enabled. The disk-restore path preserves output behavior on this case.
- **Live-reused-state output equality: PASS.** An appended turn-4 request (2,112 tokens) reused 1,594 tokens from live state (`source=live_state`, 75.5%) and prefilled only the 518-token tail (21.7 s); a forced cold prefill of the exact same prompt afterward (2,112 tokens, 110.2 s) produced identical output. TTFT comparison for the same 2,112-token prompt: 21.7 s reused vs 110.2 s cold — the first controlled same-prompt time-saved measurement on the real model (~5x TTFT reduction for this append case).
- **Miss labels surface correctly when the disk cache is off:** `no_live_state` (fresh process) and `prompt_not_longer` (equal-or-shorter prompt than state) were reported verbatim, confirming that phase 1's `no_matching_disk_snapshot` masking is specific to the disk-cache branch overwriting the live-state miss.
- **`divergent_prefix` classification: PASS** (phase 2b). A 2,650-token prompt that diverges early (a wrong assistant turn) against a 2,112-token live state reported `source=none`, `miss=divergent_prefix`, 0 reused tokens, and a full 111.5 s prefill; its genuinely different final question also produced the expected different answer. Divergent inputs do not reuse invalid state.
- Probe-design notes for future runs: equal-length variants hit the no-rewind rule (`prompt_not_longer`) rather than the divergence path, so divergence probes must be strictly longer than held state; and synthetic prompts whose turns ask the same question make output-difference checks vacuous — vary the question when checking divergence by output.
- What remains for C13 correctness: truncation, context growth past the window, reset and multiplexed-serving cases; repeat the append comparison at larger prompt sizes; and measure the 256-token policy with real interactive workloads.
- Review findings on the current code change: the generated-fixture suite reported 15 passed and 1 skipped; exact-repeat restore matched cold greedy output, the cache-enabled miss retained `prompt_not_longer`, and 200/256-token prompts verified the default threshold. `make qwen36`, Python syntax checks, and `git diff --check` pass on macOS. LRU pruning uses `st_mtime` plus the platform-specific touch wrapper, and comments match the policy. A MinGW-w64 cross-build from macOS produced a verified PE32+ `qwen36.exe`; this verifies compilation/linking, not execution on Windows.

### 2026-11-12 — C13 phase 3 real-model validation (three-fix sequence)

- Re-ran the prepared `c13_prefix_phase3.py` sequence against the live serving stack (fresh restart of the same config as the 2026-11-11 runs: snapshots enabled with the new 256-token default minimum, ctx 66K, greedy, 48 output tokens). Focused fixture suite re-verified green first (15 passed, 1 skipped). Rows are saved in `logs/logs/colibri-c12-workloads/phase3_results.jsonl`; the earlier attempt on 2026-11-11 (snapshots written 00:31–00:45) was never recorded and its stdout is unrecoverable, so this run supersedes it.
- **Exact-repeat full disk restore: PASS.** p1 (341 tokens, unique, cold): 0 reused, 12.84 s prefill (37.6 ms/token), TTFT 12.97 s, snapshot stored. p2 (exact repeat): 341/341 restored from disk in 0.085 s, **prefill 0.0 s, engine TTFT 0.088 s** — about 147× lower TTFT than the cold case, with identical output (sha `fb965acd73e5`). This is the first measured exact-repeat cold-vs-restored latency pair on the real model; fixture tests had only established output parity before.
- **Repeat after divergent live state: PASS.** p4 (exact repeat of p1 while live state held p3's different 271-token prompt): 341/341 restored from disk, prefill 0.0 s, output identical to p1. The disk path recovers even when live state cannot be reused.
- **Miss-label survival with disk caching on: PASS.** p3 (271-token new prompt, shorter than the 341-token live state) reported `prompt_not_longer` verbatim and did a full 10.68 s prefill — the disk branch no longer masks the live-state miss reason.
- **Divergent longer prompt: consistent behavior.** p5 (exact repeat of the phase-1/2 1,580-token t3 prompt) reported 0 reused, miss `divergent_prefix`, and a full 55.9 s prefill (35.4 ms/token) — but output still matched the twice-verified cold reference (sha `49ad13729105`). Root cause is benign and informative: the 1,580-token disk snapshot from phase 1 had been pruned away under the default `COLI_PROMPT_CACHE_MAX=8` mtime-LRU cap (the directory holds ~3.4–4.3 GB across 8 files), so the only candidate state was the divergent live state and the miss label is correct. Consequence worth measuring: the default 8-file cap evicts useful agent-session snapshots quickly at 400–575 MB per file; retention policy/size budget is an open storage-cost question.
- **Append path after a longer divergent state: works via disk.** p6 (p1 conversation + assistant turn + new question, 628 tokens) could not rewind live state (it held p5's 1,580 tokens → `prompt_not_longer`), so it restored the 341-token p1 snapshot from disk and prefilled only the 287-token tail (14.17 s, TTFT 14.45 s) versus roughly 23–24 s for a cold 628-token estimate. Correct behavior, though it shows the live-state extension path is bypassed whenever the previous live prompt is longer than the incoming one — a multiplexed/alternating-session case to watch.
- Probe-design notes confirmed on the real model: p3's output check is vacuous (the templated question produces the same answer); divergence checks need distinct questions. Prefill per-token rate at ~1.5K context now spans 35.4–55.2 ms/token across runs — the unexplained variance the repeated C12 baselines must bound.

### 2026-11-12 — C13 retention fix: size-budgeted snapshot pruning

- The phase-3 pruning observation became a code change: eviction is now enforced under **both** a file-count cap (`COLI_PROMPT_CACHE_MAX`, default 8) and a total-bytes budget (`COLI_PROMPT_CACHE_GB`, default 8, 0 = unlimited), pruning oldest-mtime-first while either cap is exceeded. Rationale: snapshots scale with prompt length (~272 KB/token at this model), so a count cap is blind to the actual storage cost and evicted a reusable 1,580-token state within a day of real use.
- Fixture-backed test added (`test_disk_snapshot_retention_prunes_by_count_and_bytes`): verifies eviction to the count cap, eviction by byte budget when the count cap would have kept both files, and that the retained snapshot still restores (`disk_snapshot`). Suite now reports 16 passed, 1 skipped. `make qwen36` and `git diff --check` pass on macOS. The running server still uses the pre-fix binary; the new default applies from the next restart (no restart was forced mid-validation).

### 2026-11-12 — C13 phase 4: truncation, reset, 4K exact repeat and the larger append comparison

- Ran against the live stack (later restarted onto the retention-fix build with `COLI_PREFIX_LOG=1`; server logs are session artifacts). Drivers and rows: `c13_prefix_phase4.py`, `c13_prefix_phase4b.py`, `phase4*_results.jsonl` under `logs/logs/colibri-c12-workloads/`.
- **Truncation: PASS.** After a two-turn state (594 tokens), the 378-token prompt that drops the assistant turn reported 0 reused, miss `prompt_not_longer`, and a full 13.8 s prefill. Truncated history never rewinds or reuses invalid state.
- **Reset (system-prompt change): PASS.** With a strictly longer prompt (816 > 594) whose system message differs at position 0: 0 reused, miss `divergent_prefix`, full 26.5 s prefill, and a different answer. The first reset attempt (536 < 594) correctly hit `prompt_not_longer` instead — reset probes must be longer than held state, the same lesson as phase 1's r7.
- **4K exact repeat: PASS and large.** Cold 3,458-token prompt: 123.4 s prefill (35.7 ms/token), TTFT 123.6 s. Exact repeat: 3,458/3,458 restored from disk in 0.160 s, prefill 0.0 s, **TTFT 0.170 s** (~726× lower), identical output. Restores scale with snapshot size as expected (0.085 s at 341 tokens → 0.16 s at 3,458).
- **Larger append comparison (3,852 tokens): PASS, ~7.3×.** Appended turn after the 3,458-token prompt: 3,458 reused (disk snapshot), tail prefill 17.9 s, TTFT 18.29 s; forced cold prefill of the same prompt in a cache-off process: TTFT 133.06 s, identical output (sha `a85ee064470b`). This is the larger-size counterpart to phase 2's 2,112-token ~5× case.
- **Why the append used `disk_snapshot` rather than `live_state` — explained, not a bug.** With `COLI_PREFIX_LOG=1`, the live-state miss analysis shows: (a) after any request the held state includes the generated tokens (594 prompt + 15 generated = held 609), so an exact repeat or a shorter new turn is always declined by the no-rewind rule, and the disk path with saved logits is the correct zero-prefill mechanism; (b) the phase-4 append declined live reuse because the driver's scripted assistant turn differed from the model's actual generated text at one token (`…the snapshot only after…` vs the script's `…the snapshot after…`) — divergence at the assistant turn's first token, so live reuse was correctly refused. Real agent clients resend the model's actual output text, so real sessions live-extend (phase 2 measured exactly that). **Probe-design rule going forward: synthetic multi-turn drivers must replay the engine's own generated text for assistant turns**, otherwise the live path is silently skipped in favour of the disk snapshot.
- Per-token prefill at ~3.5K context: 35.7 and 34.5 ms/token (cold 3,458 and 3,852) — consistent with the pilot's 42 ms/token at 4K; variance across runs remains the open C12 item.

### 2026-11-12 — C13 phase 5: multiplexed serving (KV_SLOTS=2) and the retention budget on a real multi-turn session

- Drivers and rows: `c13_prefix_phase5.py`, `phase5_mux_results.jsonl`, `phase5_muxcoldref_results.jsonl`/`_checks.json`, `phase5_retain_results.jsonl` under `logs/logs/colibri-c12-workloads/`; server logs `serve-phase5-mux.log`, `serve-phase5-retain.log`.
- **Multiplexed correctness: PASS.** Two distinct conversations ran with both turns of a round in flight concurrently (`--kv-slots 2`; the resource planner picked a 16,384-token context for 9.2 GB of planned conversation state on this 16 GB host). Turn 2 of each conversation replayed that conversation's actual engine output and reused the **whole** held live state (`live_state`, 449 of 689 and 412 of 647 tokens — the held state includes the generated tokens, and the replay matched through them), cutting concurrent-turn TTFT to 7.9–16.2 s vs the 20.4–21.1 s full-prefill colliding run. Output identity held: all four mux outputs (t1 and t2 of both conversations) are sha- and prompt-token-identical to the same conversations run alone on a single-slot, cache-off server — interleaved 2-slot decoding did not change greedy results.
- **The pcache gate held in mux mode:** the scratch snapshot directory stayed empty and no request ever reported `disk_snapshot` — the disk cache is deliberately disabled at `KV_SLOTS>1` (restore and store both check `g_q36_mux_slots == 1`), so multiplexed serving gets live-state reuse only. Lifting that gate is a possible future step, not a defect.
- **Slot affinity is hash-based, and collisions degrade gracefully (observed).** The gateway routes a conversation to `sha1(system+first user message) mod kv_slots` (openai_server.py's stable-slot hash, the #634 fix). The first mux attempt used two prompts that hashed to the same slot: the rounds serialized on one slot's state, each request correctly refused the previous conversation's state by miss reason (`prompt_not_longer`, `divergent_prefix`) and paid a full prefill — the documented degrade path, correctness intact. The driver now pins non-colliding prompts (hash checked offline) so the concurrent case is what runs.
- **Retention budget on a real multi-turn session: PASS.** Three 3-turn agent-style sessions (assistant turns replaying actual outputs) against a scratch cache directory with `COLI_PROMPT_CACHE_GB=0.5`: the directory plateaued at 6 files / ~486 MB and never exceeded the budget; every store over budget pruned the **oldest** file by mtime (`[PCACHE] pruned … byte budget`: 71.1 → 80.7 → 91.6 → 71.0 MB as sessions A, B, C churned). Session B and C's second turns restored their session's t1 snapshot from disk (373 of 857 and 378 of 864 tokens, 0.08–0.09 s) — see the live-vs-disk note below for why they did not live-reuse.
- **Restore after churn, and eviction cost:** the exact repeat of the newest session's last prompt fully restored (1,414 of 1,414 tokens, TTFT 0.129 s vs 19.35 s cold, identical output). The exact repeat of the oldest session's first prompt — evicted three stores earlier — correctly reported reuse 0 (`prompt_not_longer` from the longer live state, no matching disk snapshot) and still produced the byte-identical output of its first occurrence (sha `4c982b2658ea`) after a full 14.0 s prefill. Snapshot restore time continues to scale with size (0.082 s at 373 tokens → 0.124 s at 1,414).
- **Live-vs-disk reuse choice varies per turn, both correct:** session A's t2 reused live state (416 tokens) while sessions B and C's t2s fell to the disk snapshot of their t1 prefix — the live path declines when the re-rendered assistant turn does not reproduce the held generated ids exactly (the phase-4c boundary), and the disk snapshot of the prompt-only prefix covers it. No output was wrong in any case; the engine picks the first mechanism that applies.
- Prune events are now logged (`[PCACHE] pruned <path> (<size> MB) <reason>`, behind `COLI_PCACHE_LOG=1`) — added before this run so eviction order would be observable; `make qwen36` clean.

### 2026-11-12 — C12/C13 repeated representative matrix: 1K/4K/12K, cold/restore/append, speculation on/off

- Driver `c13_matrix.py`, summarizer `c13_matrix_summary.py`; rows `phaseM_cold_speon_results.jsonl`, `phaseM_cold_speoff_results.jsonl`, `phaseM_reuse_results.jsonl`, `phaseM_restart_results.jsonl`, `phaseM_decode_speon_results.jsonl`, `phaseM_decode_speoff_results.jsonl`; summary `phaseM_summary.jsonl`. All under `logs/logs/colibri-c12-workloads/`. Settings: Qwen3.6-35B-A3B IQ2_XXS, `OMP_NUM_THREADS=6`, `--ctx 66000`, concurrency 1, greedy, 128 output tokens, temperature 0. Prompt sizes calibrated on this model+template from phases 4–5 (`tokens ≈ 98 + 35×pads`): 1,010 / 4,090 / 12,315 tokens, verified from the engine's own counts.
- **Cold-state prefill distributions (full prefill, cache off).** Spec on: 1K p50 33.0 s (n=5, 32.7–37.1 ms/token), 4K p50 158.5 s (n=3, 37.6–40.0 ms/token), 12K p50 677.1 s (n=3, 55.0–58.5 ms/token). Spec off (`COLI_LOOKUP=0`): 1K p50 45.4 s (n=5), 4K p50 211.5 s, 12K p50 722.6 s. Per-token cost rises with context in both configs — the pilot's superlinear profile reproduced with n>1.
- **Measurement confound found — drift dominates the speculation comparison on prefill.** Prompt-lookup drafting cannot change prefill (it drafts during decode), yet the spec-off run's prefills are 20–30% slower, and the 1K spec-off cell drifts 34.2 → 53.8 s across its five repeats (spec-on: 31.1 → 37.5). The gap is machine/process state (thermal or background load or allocator growth), not configuration. Consequence for the plan: cross-run prefill comparisons on this host need interleaved or simultaneous A/B designs, or the drift must be quantified per run; a single cross-config delta is not interpretable. Decode rate is where speculation should show, see below.
- **Exact-repeat disk restore across a process restart (n=3 per size, extremely tight):** 1K TTFT p50 0.102 s, 4K 0.189 s, 12K 0.416 s — restore time ≈ 0.034 ms/token (0.097/0.180/0.389 s restore components), zero prefill, identical outputs. Vs the same-day spec-on cold p50s: ~324× at 1K, ~839× at 4K, ~1,628× at 12K. This is now a distribution, not a single observation.
- **Appended turns with live reuse (n=1 per size, actual-output replay):** tail of ~242 new tokens on a reused 1,041/4,121/12,346-token prefix cost 10.7 / 14.0 / 18.7 s TTFT — i.e. ~44 → 58 → 77 ms per tail token as context grows. The tail's attention reads the whole context, so append cost is superlinear in context length at fixed tail size: direct confirmation that C15's target (attention over accumulated context in prefill) also dominates the incremental-turn case, not just the first request.
- **Speculation (prompt lookup, MTP head not loaded — `Q36_MTP=1` was not set):** decode rate on restored 1K state 15.5 tok/s p50 with lookup vs 16.3 tok/s p50 without — **no measurable speedup** on this workload, whose highly repetitive filler should favor lookup drafts. One cold-state spec-on row ran at 10.1 tok/s (n too small to interpret). The default gate's own earlier logs show it already refusing most draft steps (rows/step 1.02–1.88). Prompt lookup remains default-on for its zero-cost gate; no change justified by this data.
- Single-observation cells remain: the reuse-mode colds (4K 168.0 s, 12K 681.7 s — consistent with the cold-config p50s) and the three append rows. Peak memory was not captured in this matrix; that and the memory/TTFT correlation stay open.

### Current measurement gaps and next order

1. C13 first-pass validation is done: disk-restore, live-reuse and divergent cases are output-exact vs cold prefill; phase 3 (2026-11-12) verified the three fixes end to end; phase 4 (2026-11-12) added truncation, reset, a 4K exact repeat (~726× TTFT) and the larger append comparison (3,852 tokens, ~7.3×); phase 5 (2026-11-12) validated multiplexed serving (`KV_SLOTS=2`, sha-identical outputs, live reuse of the whole held state) and the `COLI_PROMPT_CACHE_GB` retention budget on a real multi-turn session (oldest-first pruning, ≤ 486 MB at a 0.5 GB budget, evicted repeats still output-exact). The eviction-policy gap is fixed with size-budgeted retention (`COLI_PROMPT_CACHE_GB`, fixture-tested), and the live-vs-disk source choice is explained by state-contains-generation plus scripted-assistant divergence — not a bug. Remaining cases: native Windows runtime test (parked — see Section 9); system-prompt/tool-result turns in the agent-session workload.
2. Cold/warm/restored-prefix TTFT separation now has controlled same-prompt comparisons at three sizes (2,112-token append ~5×; 341-token exact repeat ~147×; 3,458-token exact repeat ~726×; 3,852-token append ~7.3×) **and, since the 2026-11-12 matrix, distributions with n>1**: cross-process restore p50s 0.102/0.189/0.416 s at 1K/4K/12K (n=3 each, ~324×/~839×/~1,628× vs same-day cold) and cold-state prefill p50s at the same sizes.
3. The repeated 1K/4K/12K matrix is done (2026-11-12): cold/restore/append separation, speculation on and off, percentiles in `phaseM_summary.jsonl`. Remaining from this item: re-run 32K/64K under identical settings before folding them in, and use interleaved A/B designs for any cross-config prefill comparison (drift finding).
4. Measure memory alongside TTFT and sustained decode for the same runs; verify timing reconciliation and GPU chunk summaries where applicable, and measure instrumentation overhead.
5. Only after these measurements choose a C14 residency or C15 kernel/chunk experiment. Preserve the same prompts, sampling, output limit and environment across comparisons.

## 7. Priority summary

| Phase | Focus | Priority | Gate | Status / evidence |
|---|---|---:|---|---|
| C12 | TTFT instrumentation and baselines | P0 | Identify bottleneck | **Instrumentation:** gateway/Qwen3.6 stage timings plus embedding and backend chunk summaries; single-slot and multiplexed STAT paths include first-sample and first-text-emission timings. CPU build and focused suites passed (521 tests, 646 subtests, 12 fixture-dependent skips); multiplexed runtime behavior and Vulkan/CUDA chunk fields remain unverified. A real 1,045-token request verified the CPU fields (embedding 0.000565 s; one 35.160 s chunk). **Real-model pilot:** 1,045/4,105/12,231/32,012/64,012-token prompts produced 34.3/172.5/785.0/3,358.6/11,845.1 s engine TTFT and 14.30/9.43/6.92/3.98/2.75 tok/s over 128 output tokens; client/server first-output delta stayed 12–38 ms across all sizes. Prefill cost per token is superlinear (32.8→184.9 ms/tok) and prefill is attention-dominated at long context (attention ≈55% of recorded components at 32K, ≈71% at 64K; per-token attention 21.5→57.9→130.5 ms across 12K/32K/64K, while MoE per-token time is roughly flat) — direction selected for C15, not a kernel change. **Repeated matrix (2026-11-12):** cold-state prefill now has distributions (spec on: 1K p50 33.0 s n=5, 4K p50 158.5 s n=3, 12K p50 677.1 s n=3; per-token 33→39→55 ms, superlinear profile reproduced), cross-process disk restores are tight distributions (0.102/0.189/0.416 s TTFT p50, n=3 per size, ≈0.034 ms/token, ~324×/~839×/~1,628× vs same-day cold), and appended turns with live reuse show the incremental tail cost rising 44→58→77 ms per tail token as context grows at a fixed ~242-token tail. Prompt-lookup speculation showed no measurable decode speedup (15.5 vs 16.3 tok/s p50). **New finding:** prefill drifts up to ~30% across same-day runs and monotonically within a process (34→54 s over five 1K repeats), so cross-config prefill deltas are not interpretable without interleaved A/B designs — the open variance item is now bounded and characterized. **Pending:** GPU chunk verification, memory correlation (peak RSS not captured in the matrix), instrumentation overhead, interleaved A/B for any config comparison. |
| C13 | Prefix reuse validation | P0 | Correct reuse and measurable savings | **Telemetry implemented:** Qwen3.6 records reuse source, token count, and bounded miss reason; benchmark reports reuse fraction and categories. **Real-model validation (2026-11-11), phase 1–2b:** disk-snapshot extension reuse works (1,048 of 1,580 tokens restored in 0.123 s); live-state reuse works (1,594 of 2,112, tail prefill only); restored and reused state produced output identical to cold prefills (sha-matched, across processes and cache configs); divergent prompts reuse nothing and report `divergent_prefix`. **Phase-3 three-fix sequence (2026-11-12):** exact repeat of a 341-token prompt fully restored from disk with 0.0 s prefill (TTFT 0.088 s vs 12.97 s cold, identical output); repeat after a divergent live state also fully restored; `prompt_not_longer` survives the disk branch with caching on; a pruned-snapshot repeat correctly reported `divergent_prefix` and still matched the cold output reference. **Phase-4 cases (2026-11-12):** truncation declines reuse (`prompt_not_longer`, full prefill); system-prompt reset declines reuse (`divergent_prefix`, longer-than-state probe); 3,458-token exact repeat restores with TTFT 0.170 s vs 123.6 s cold (~726×); 3,852-token append costs 18.29 s vs 133.06 s cold (~7.3×), output sha-equal — with the live-vs-disk source choice explained by the no-rewind rule over state that includes generated tokens and by scripted-assistant text divergence in the driver, not a defect. **Phase-5 (2026-11-12): multiplexed serving and retention validated.** Two concurrent conversations on `--kv-slots 2` reused their whole held live state on turn 2 (449/689, 412/647 via `live_state` with actual-output replay), all four outputs sha-identical to single-slot cold references, scratch snapshot directory stayed empty (the disk cache is deliberately gated to `KV_SLOTS==1`), and a same-slot collision degraded to correctly-labeled full prefills. The `COLI_PROMPT_CACHE_GB=0.5` budget on a three-session agent-style workload held the directory at ≤ 486 MB, pruned oldest-first, restored mid-session prefixes (373/857, 378/864) and the newest session's last prompt in full (1,414/1,414, TTFT 0.129 s vs 19.35 s cold), and an evicted prompt's repeat still produced its exact original output. Prune events are now logged for observability. **Implemented and targeted-tested:** default snapshot minimum is 256 tokens; oldest-mtime-first pruning with touch-on-restore under a file-count cap **and** a total-bytes budget (`COLI_PROMPT_CACHE_GB`, default 8, fixture-tested for both caps). Suite: 16 passed, 1 skipped. macOS build, Python syntax checks, and diff checks pass. MinGW-w64 cross-build produced a verified PE32+ `qwen36.exe`; execution on Windows itself remains unverified. **Pending:** native Windows runtime test, system-prompt/tool-result turns in the agent-session workload, repeated representative matrix. |
| C14 | Hybrid RAM residency | P1 | Better end-to-end performance without pressure | Existing opt-in `Q36_HYBRID_GB` implementation; no budget sweep or memory-pressure validation recorded here. **Pending:** 0/1/2/3/5 GB repeated workload tests, including decode and TTFT. No speed or memory result established. |
| C15 | Long-prompt prefill acceleration | P1 | Profile-guided improvement | Not started; waiting on C12 operation-level profile. No tests or speed improvements measured. |
| C16 | Q4_K_M comparison | P2 | Quality/performance evidence | Not started; exact candidate layout, tensor coverage, quality, memory, TTFT and decode comparison pending. No result established. |
| C17 | 27B dense / C8 Metal | P2 | Dense end-to-end gain | Candidate experiment only; clean post-instrumentation baselines and end-to-end Metal/MTP comparison pending. No new speed result established. |

## 8. Explicit non-goals

- Do not chase decode tok/s at the expense of agent TTFT without measuring the trade-off.
- Do not assume MTP should be enabled by default; current 35B measurements show it can lose after C11.
- Do not reserve or lock the full user-offered 5 GB blindly.
- Do not build a second prefix cache before measuring `kv_prefix.h`.
- Do not implement Q4_K_M-specific kernels before profiling the exact model.
- Do not assume Metal is automatically faster than the M4 CPU for sparse MoE.
- Do not accept a benchmark improvement without correctness and repeatability.

## 9. Immediate next actions

1. C13: native Windows runtime test — **parked (2026-11-12, user decision): nothing downstream is blocked by it** (the MinGW cross-build already verifies compile/link, and retention/mux/matrix validation runs on macOS and the fixture suite). It is an isolated portability check of the restore-touch `_utime` path. When unparked: the user's Windows laptop can run the full codebase plus the model — copy the repo and the GGUF container there, build, and run `c/tests/test_qwen36_prefix_serve.py` natively. Multiplexed serving (`KV_SLOTS=2`) and the `COLI_PROMPT_CACHE_GB` retention budget are already validated on the live stack (2026-11-12, phase 5): concurrent conversations live-reuse their held state with sha-identical outputs, and the budget pruned oldest-first while an evicted prompt's repeat stayed output-exact. Extend the deterministic agent-session workload (Section 5) with large system prompts and tool-result turns. Note: the disk prompt cache remains deliberately gated to `KV_SLOTS==1`; lifting that gate is a scoping question, not a defect.
2. Repeated representative matrix — **done (2026-11-12)**: 1K/4K/12K cold-state prefill distributions (spec on and off), cross-process restore distributions, append-with-live-reuse cases, and the speculation decode probe; percentiles in `phaseM_summary.jsonl`. Remaining from this item: the 32K/64K observations join only if re-run under identical settings, and any cross-config prefill comparison needs an interleaved A/B design (drift finding).
3. Measure memory, TTFT and sustained decode together for the same runs; verify timing reconciliation and instrumentation overhead. Reconcile the baseline SHA before interpreting comparisons. Memory correlation is the next measurement gap (the matrix captured no peak RSS).
4. Use the completed 1K–64K profile — attention is the dominant and superlinear long-prompt prefill component — to scope a C15 attention prefill experiment, and only then choose C14 residency work, without regressing either first-token latency or sustained generation speed.

**Decision rule:** C12 tells us where the time goes. C13 removes repeated work. C14 reduces avoidable data movement. C15 accelerates the remaining prefill bottleneck. C16 and C17 broaden model/hardware options after the primary agent workload is measurable.
