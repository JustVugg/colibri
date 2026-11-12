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

**Priority:** P0 after C12 · **Status:** Telemetry implemented; real-model validation passed for disk-restore, live-reuse and divergent cases (2026-11-11) and for the three-fix phase-3 sequence (2026-11-12). Code preserves informative live-state miss labels, supports exact-repeat disk restore using saved logits, lowers the default snapshot threshold to 256 tokens, and prunes by mtime with restore-touch LRU (default cap 8 files). The macOS build and targeted fixture-backed runtime tests pass; a MinGW-w64 cross-build from macOS also produced a verified PE32+ `qwen36.exe`. LRU comments now match the policy. This verifies compilation/linking, not execution on Windows. Remaining cases include truncation, reset and multiplexed serving; snapshot-retention policy under the 8-file default cap needs measurement.

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

### Current measurement gaps and next order

1. C13 first-pass validation is done (2026-11-11): disk-restore, live-reuse and divergent cases are output-exact vs cold prefill, and the append case measured ~5x TTFT reduction on one 2,112-token case. The 2026-11-12 phase-3 run verified the three fixes end to end on the real model: exact-repeat zero-prefill restore (TTFT 0.088 s vs 12.97 s cold on a 341-token prompt), informative miss-label survival with disk caching on, and the 256-token default threshold in live serving. The eviction policy gap found by that run is fixed with a size-budgeted retention (`COLI_PROMPT_CACHE_GB`, fixture-tested). Fixture-backed tests verify exact-repeat output parity, cache-enabled miss-label preservation, and the 256-token default threshold. MinGW-w64 cross-compilation passes; native Windows runtime remains unverified. Next cover truncation, reset, multiplexed serving and a larger append case.
2. Cold/warm/restored-prefix TTFT separation is partially done: cold, restored-extension, reused-append and exact-repeat-restored cases have real numbers, including the controlled same-prompt reused-vs-cold comparisons (2,112-token append ~5x; 341-token exact repeat ~147x). Next: a larger append case; retention policy now has a byte budget but its real-workload eviction/restore behavior still needs a measuring run.
3. Repeat the 1K/4K/12K prompt matrix with controlled settings and representative workloads; the 32K and 64K scaling observations are done (2026-11-11), so fold them into repeated p50/p95/p99 runs only if the full matrix is re-run under identical settings.
4. Measure memory alongside TTFT and sustained decode for the same runs; verify timing reconciliation and GPU chunk summaries where applicable, and measure instrumentation overhead.
5. Only after these measurements choose a C14 residency or C15 kernel/chunk experiment. Preserve the same prompts, sampling, output limit and environment across comparisons.

## 7. Priority summary

| Phase | Focus | Priority | Gate | Status / evidence |
|---|---|---:|---|---|
| C12 | TTFT instrumentation and baselines | P0 | Identify bottleneck | **Instrumentation:** gateway/Qwen3.6 stage timings plus embedding and backend chunk summaries; single-slot and multiplexed STAT paths include first-sample and first-text-emission timings. CPU build and focused suites passed (521 tests, 646 subtests, 12 fixture-dependent skips); multiplexed runtime behavior and Vulkan/CUDA chunk fields remain unverified. A real 1,045-token request verified the CPU fields (embedding 0.000565 s; one 35.160 s chunk). **Real-model pilot:** 1,045/4,105/12,231/32,012/64,012-token prompts produced 34.3/172.5/785.0/3,358.6/11,845.1 s engine TTFT and 14.30/9.43/6.92/3.98/2.75 tok/s over 128 output tokens; client/server first-output delta stayed 12–38 ms across all sizes. Prefill cost per token is superlinear (32.8→184.9 ms/tok) and prefill is attention-dominated at long context (attention ≈55% of recorded components at 32K, ≈71% at 64K; per-token attention 21.5→57.9→130.5 ms across 12K/32K/64K, while MoE per-token time is roughly flat) — direction selected for C15, not a kernel change. One synthetic sample per size, speculation default-on, snapshots enabled but fresh prompts (0 reuse, `no_matching_disk_snapshot`), no peak-memory capture; these are not baseline distributions or a speedup. **Pending:** representative repeated runs, speculation-off baseline, cold/warm/restored-prefix separation, GPU chunk verification, memory correlation, instrumentation overhead and C13 prefix correctness. |
| C13 | Prefix reuse validation | P0 | Correct reuse and measurable savings | **Telemetry implemented:** Qwen3.6 records reuse source, token count, and bounded miss reason; benchmark reports reuse fraction and categories. **Real-model validation (2026-11-11), phase 1–2b:** disk-snapshot extension reuse works (1,048 of 1,580 tokens restored in 0.123 s); live-state reuse works (1,594 of 2,112, tail prefill only); restored and reused state produced output identical to cold prefills (sha-matched, across processes and cache configs); divergent prompts reuse nothing and report `divergent_prefix`. **First controlled time-saved measurement:** same 2,112-token prompt, appended turn with reuse TTFT 21.7 s vs forced cold prefill 110.2 s (~5x) — single case, not a distribution. **Phase-3 three-fix sequence on the real model (2026-11-12):** exact repeat of a 341-token prompt fully restored from disk with 0.0 s prefill (TTFT 0.088 s vs 12.97 s cold, identical output); repeat after a divergent live state also fully restored; `prompt_not_longer` survives the disk branch with caching on; a pruned-snapshot repeat correctly reported `divergent_prefix` and still matched the cold output reference; an append after a longer divergent live state reused the disk snapshot and prefilled only a 287-token tail (14.45 s TTFT). **Implemented and targeted-tested:** default snapshot minimum is 256 tokens (200-token prompt stores nothing; 256-token prompt stores); oldest-mtime-first pruning with touch-on-restore, default cap 8 files. The generated tiny fixture enabled the focused suite: 15 passed, 1 skipped. macOS build, Python syntax checks, and diff checks pass. Windows `_utime` portability code is in place, and a MinGW-w64 cross-build produced a verified PE32+ `qwen36.exe`; execution on Windows itself remains unverified. **Pending:** native Windows runtime test, truncation/reset/multiplexed cases, larger-size repeats, and the snapshot retention/size-budget policy (the 8-file cap evicts 400–575 MB agent-session snapshots quickly). |
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

1. C13: run the cache-focused tests on a native Windows runner to validate runtime `_utime` behavior; then cover truncation, context growth past the window, reset and multiplexed serving. Measure the new `COLI_PROMPT_CACHE_GB` retention on a real multi-turn session (eviction order, restore hits, disk footprint).
2. Repeat the reused-vs-cold append comparison at a larger prompt size, then measure the 256-token snapshot policy against real interactive workloads and storage/latency cost.
3. Measure real Qwen3.6 35B cold, warm and restored-prefix requests separately, including snapshot restore/store, TTFT, prefill and decode.
4. Run the repeated 1K/4K/12K prompt matrix under controlled settings (66K only where supported); the single-observation 32K/64K scaling run is complete — see the 2026-11-11 entry.
5. Measure memory, TTFT and sustained decode together; verify timing reconciliation and instrumentation overhead. Reconcile the baseline SHA before interpreting comparisons.
6. Use the completed 1K–64K profile — attention is the dominant and superlinear long-prompt prefill component — to scope a C15 attention prefill experiment, and only then choose C14 residency work, without regressing either first-token latency or sustained generation speed.

**Decision rule:** C12 tells us where the time goes. C13 removes repeated work. C14 reduces avoidable data movement. C15 accelerates the remaining prefill bottleneck. C16 and C17 broaden model/hardware options after the primary agent workload is measurable.
