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

**Priority:** P0 · **Status:** Gateway and Qwen3.6 stage timings implemented; representative workload matrix and real-model baseline collection pending

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

**Priority:** P0 after C12 · **Status:** Exact-prefix timing and source/miss telemetry implemented for Qwen3.6; correctness and real-session validation pending

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
- Qwen3.6 reports tokenization, live-prefix matching, snapshot lookup/restore, fresh-prefill wall time and prompt/reused-token counts. Fresh-prefill operation accumulators cover DeltaNet, attention, MoE, shared/router subsets and LM head; the first sampling call is timed independently, and elapsed time to first non-empty text emission after prefill is reported. The emission interval can include sampling, decoding and UTF-8 buffering. Operation accumulators are diagnostic, not additive wall-time parts.
- Benchmark summaries and baseline comparisons expose client and server timings separately, including first-sample and post-prefill emission distributions. API and baseline documentation describe their measurement boundaries.
- Embedding/input preparation and explicit per-prefill-chunk timing are not yet separately exposed; confirm whether the serving path splits prompts into multiple measured chunks and add those fields if applicable.
- Focused validation passed: `cd c && make qwen36`; `cd c && PYTHONPATH=. python3 -m pytest tests/test_openai_server.py tests/test_benchmark_http_serving.py tests/test_benchmark_baseline.py tests/test_qwen36_prefix_serve.py -q` (517 passed, 646 subtests, 12 fixture-dependent skipped); and `git diff --check`. A real-model inference has not validated the new engine timing values, and no speed improvement has been measured.

### 2026-11-10 — C13 prefix telemetry in progress

- Qwen3.6 serving statistics now distinguish live state, pin snapshot, disk snapshot, and no reuse; misses are classified with bounded reason labels such as disabled, tainted state, divergent prefix, non-extension, unavailable sliding window, or no matching disk snapshot.
- Gateway and benchmark paths preserve and summarize those labels. They carry token counts and controlled enums only; they do not include prompt text.
- Next validation: compile and run the new protocol/parser tests, then run a deterministic multi-turn Qwen3.6 session with stable-prefix, appended-turn, divergent-prefix, and restart/cold cases. Compare prompt/reused-token counts and prefill time while checking generated output against cold-prefill references. The tiny fixture is currently unavailable, and no real-model result should be claimed until that workload is run.

### Current measurement gaps and next order

1. Finish C13 telemetry validation, including cache-snapshot versus live-state source and correctness on divergence/context growth.
2. Collect repeated 1K/4K/12K/32K workloads and 66K only when supported, separating cold, warm and prefix-hit runs; add explicit embedding/input-preparation and per-prefill-chunk timing if the actual execution path has multiple chunks, and verify timing reconciliation and instrumentation overhead. Include client TTFT p50/p95/p99, server TTFT, stage timings, and sustained decode tokens/sec.
4. Only then choose a measured C14 hybrid-residency or C15 kernel/chunk experiment. Preserve the same prompts, sampling, output limit and environment across comparisons.

## 7. Priority summary

| Phase | Focus | Priority | Gate | Status / evidence |
|---|---|---:|---|---|
| C12 | TTFT instrumentation and baselines | P0 | Identify bottleneck | **Stage instrumentation implemented:** gateway reports request/engine timings and decode rate; Qwen3.6 reports tokenization, prefix/snapshot, aggregate and operation-level prefill, first sampling and first text emission. Tests: Qwen3.6 build passed; targeted suites passed (517 tests, 646 subtests, 12 fixture-dependent skips); diff check passed. **Pending:** separately expose embedding/input-preparation and per-chunk timings if applicable; real-model protocol verification; repeated cold/warm/prefix workloads; 1K–66K prompt matrix; timing reconciliation and instrumentation-overhead measurement. No real-model speed improvement measured. |
| C13 | Prefix reuse validation | P0 | Correct reuse and measurable savings | **Telemetry implemented:** Qwen3.6 records reuse source, token count, and bounded miss reason; benchmark reports reuse fraction and categories. **Pending:** real multi-turn correctness/performance validation; tiny converted fixture unavailable (12 tests skipped). Existing log showed one disk restore of 1,265/8,411 tokens, but no controlled time-saved or speedup measurement. |
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

1. Validate C13 prefix source/miss telemetry with a deterministic multi-turn session and document its measured behavior.
2. Reconcile local changes and the baseline SHA before claiming historical comparisons are current.
3. Run real Qwen3.6 35B single-request cold/warm/prefix-hit tests, recording TTFT, prefill, decode throughput and memory.
4. Complete the 1K/4K/12K/32K C12 matrix (66K where supported), confirm timing reconciliation and assess instrumentation overhead.
5. Use the resulting profile to prioritize C14/C15 without regressing either first-token latency or sustained generation speed.

**Decision rule:** C12 tells us where the time goes. C13 removes repeated work. C14 reduces avoidable data movement. C15 accelerates the remaining prefill bottleneck. C16 and C17 broaden model/hardware options after the primary agent workload is measurable.
