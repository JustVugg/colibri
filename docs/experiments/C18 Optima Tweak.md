# C18 Optima Tweak: Experimental Dynamic Workforce

## Objective
Explore whether parallel preparation and demand-driven worker allocation can improve C18 expert-matmul performance beyond the existing decode-once optimization.

**Non-negotiable:** Preserve the current C18 implementation as the immutable baseline. All experimental work must be independently switchable and removable.

## 1. Experimental Architecture

Introduce an isolated experimental execution path:

`C18_BASE` → Existing decode-once implementation  
`C18_OPTIMA` → Experimental workforce implementation

The experiment introduces three logical workloads:

1. **UNPACK:** Parallel decoding of independent quantized weight segments.
2. **SHAPE:** Organizing completed segments into the layout required by the matmul operation.
3. **PROCESS:** Consuming prepared weight data across activation rows.

Prepared weight data should be reusable across activation rows. Avoid redundant decoding, unnecessary copies and premature synchronization.

## 2. Workforce Coordinator

Introduce one lightweight coordinator responsible for worker allocation.

| Policy | Responsibility |
|---|---|
| CAP | Global thread ceiling |
| MAX | Per-workload worker ceiling |
| RESERVED | Protected capacity for urgent work |
| DEMAND | Pending work and processing pressure |
| COORDINATOR | Dynamic allocation and role transitions |

Requirements:
- Allocate workers according to demand.
- Support anticipatory reservations for upcoming stages.
- Reassign workers between workloads.
- Respect CAP, MAX and RESERVED at all times.
- Avoid excessive worker migration and synchronization.
- Prevent oversubscription with existing Colibri thread pools.

Initially, keep resource governance scoped to C18.

## 3. Experimental Controls

Provide independent runtime controls:

- `C18_OPTIMA=0|1`
- Configurable CAP, MAX and RESERVED policies.
- Experimental allocation telemetry.
- Immediate fallback to `C18_BASE`.

No experimental setting should alter baseline behaviour.

## 4. Benchmark Protocol

Run identical workloads under controlled conditions:

1. Execute C18 baseline.
2. Enable Optima and execute the experiment.
3. Disable Optima and re-run baseline.
4. Repeat interleaved A/B rounds.
5. Compare results.

Measure:
- TTFT and MoE prefill time.
- Unpacking, shaping and processing time.
- Worker utilization and allocation changes.
- Synchronization overhead.
- Memory consumption.
- Decode throughput.
- Output SHA consistency.

## 5. Experiment Lifecycle

Every experiment must be independently identifiable and disposable.

`CREATE → ENABLE → BENCHMARK → DISABLE → COMPARE → VERDICT`

Possible verdicts: KEEP, MODIFY or DISCARD.

## 6. Acceptance Criteria

- Baseline remains unchanged.
- Identical outputs.
- No decode regression.
- No memory or thread-safety regressions.
- Repeatable end-to-end improvement over C18 baseline.

Do not promote experimental code into the default execution path without a successful benchmark verdict.

## Expected Outcome

Determine whether a centrally coordinated, demand-driven workforce can outperform C18's existing decode-once implementation.

This is an experimental optimization track, not a replacement mandate.
