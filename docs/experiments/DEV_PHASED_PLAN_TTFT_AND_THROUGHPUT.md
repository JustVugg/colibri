# Colibri Development Phased Plan: TTFT, Prefill and Throughput

**Status:** Proposed roadmap / experiment plan  
**Target branch:** `feature-IQ2_XXS`  
**Prepared:** 2026-11-10  
**Baseline inspected:** `bc4b92509b214687423294a1ee320d3933faa018`  
**Scope:** Interactive and agent inference performance, primarily Qwen3.6 35B-A3B MoE and 27B dense.

> This is a plan, not a claim that the work is implemented. Re-run historical baselines after synchronising the developer's laptop changes.

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

**Measurement distinction:** client-observed first-output latency is not a complete diagnosis of engine TTFT. It includes some combination of queueing, HTTP, tokenisation, prompt preparation, prefill, logits, sampling and stream delivery. C12 must add engine-side stage timings.

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

**Priority:** P0 · **Status:** Proposed · **Start here**

**Goal:** Identify where time is spent between request arrival and the first streamed token.

Add low-overhead, request-correlated timings for:

- Request accepted / queue wait.
- Tokenisation and chat-template processing, where applicable.
- Prefix matching and reusable-token count.
- Prompt embedding / input preparation.
- Prefill total time, token count, chunk count and per-chunk duration.
- Final normalisation / logits computation.
- Sampling and first-token stream emission.
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

**Priority:** P0 after C12 · **Status:** Existing mechanism to validate

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

## 6. Priority summary

| Phase | Focus | Priority | Gate |
|---|---|---:|---|
| C12 | TTFT instrumentation and baselines | P0 | Identify bottleneck |
| C13 | Prefix reuse validation | P0 | Correct reuse and measurable savings |
| C14 | Hybrid RAM residency | P1 | Better end-to-end performance without pressure |
| C15 | Long-prompt prefill acceleration | P1 | Profile-guided improvement |
| C16 | Q4_K_M comparison | P2 | Quality/performance evidence |
| C17 | 27B dense / C8 Metal | P2 | Dense end-to-end gain |

## 7. Explicit non-goals

- Do not chase decode tok/s at the expense of agent TTFT without measuring the trade-off.
- Do not assume MTP should be enabled by default; current 35B measurements show it can lose after C11.
- Do not reserve or lock the full user-offered 5 GB blindly.
- Do not build a second prefix cache before measuring `kv_prefix.h`.
- Do not implement Q4_K_M-specific kernels before profiling the exact model.
- Do not assume Metal is automatically faster than the M4 CPU for sparse MoE.
- Do not accept a benchmark improvement without correctness and repeatability.

## 8. Immediate next actions

1. Synchronise the developer's laptop work with `feature-IQ2_XXS` and reconcile uncommitted/unpushed changes before treating this document's baseline SHA as current.
2. Re-inspect updated Markdown plans, benchmark logs and source changes on the branch.
3. Re-run the current 35B decode baseline and existing long-prompt benchmark before changing code.
4. Implement C12 instrumentation and generate the first 1K–66K baseline matrix.
5. Use those measurements to decide the exact C13 and C15 implementation tasks.

**Decision rule:** C12 tells us where the time goes. C13 removes repeated work. C14 reduces avoidable data movement. C15 accelerates the remaining prefill bottleneck. C16 and C17 broaden model/hardware options after the primary agent workload is measurable.
