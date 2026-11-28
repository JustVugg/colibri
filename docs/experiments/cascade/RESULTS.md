# CASCADE-SAMPLE-001 result

**Decision: retain NO_CHANGE in production. Bounded CPU mechanism evidence only.**
Reuses ADAPTIVE_TRANSFER_DOE and G0–G5; this is not model G2/G3/G4/G5.
Default-off generated header, no production or installed change.

All 18 cases have exact token/next-RNG agreement across all arms.
Baseline, candidate-off and candidate-on each pass the existing 1612-draw oracle.
Complete-rank/fallback/scratch-refusal/TLS checks: 609 checks × 3 modes, zero failures.

Primary effect: median of five **same-build off/on** sampler-rate contrasts.
Range is observed paired min/max, **not a confidence interval**. Each process
has one warmup and three draws. Vocabulary, shape and sampling policy are
distinct workload blocks, never pooled as independent replicates.

| Vocab | Shape | T | p | Same-build rate change | Paired range | Original-reference rate change |
|---:|---|---:|---:|---:|---:|---:|
| 131072 | flat_tied | 0.0 | 0.9 | -0.14% | [-6.79, +185.94]% | +0.00% |
| 131072 | flat_tied | 0.7 | 0.9 | +134.12% | [+103.99, +152.20]% | +143.25% |
| 131072 | flat_tied | 0.7 | 1.0 | +199.84% | [+191.98, +204.27]% | +206.06% |
| 131072 | peaked | 0.0 | 0.9 | -0.14% | [-0.42, +0.14]% | +0.00% |
| 131072 | peaked | 0.7 | 0.9 | -0.04% | [-5.66, +1.05]% | -0.08% |
| 131072 | peaked | 0.7 | 1.0 | +826.19% | [+769.42, +893.20]% | +830.23% |
| 131072 | broad | 0.0 | 0.9 | +0.27% | [+0.00, +7.46]% | +0.13% |
| 131072 | broad | 0.7 | 0.9 | +224.52% | [+201.35, +228.00]% | +221.74% |
| 131072 | broad | 0.7 | 1.0 | +747.14% | [+694.88, +797.73]% | +731.95% |
| 248320 | flat_tied | 0.0 | 0.9 | -0.44% | [-6.66, +20.98]% | +0.22% |
| 248320 | flat_tied | 0.7 | 0.9 | +149.65% | [+147.56, +152.27]% | +151.95% |
| 248320 | flat_tied | 0.7 | 1.0 | +225.16% | [+206.65, +229.70]% | +218.19% |
| 248320 | peaked | 0.0 | 0.9 | -0.30% | [-0.81, +5.04]% | -0.15% |
| 248320 | peaked | 0.7 | 0.9 | -3.60% | [-25.05, +10.89]% | -2.40% |
| 248320 | peaked | 0.7 | 1.0 | +745.56% | [+686.63, +810.22]% | +748.58% |
| 248320 | broad | 0.0 | 0.9 | -5.34% | [-6.30, +4.56]% | -1.10% |
| 248320 | broad | 0.7 | 0.9 | +234.05% | [+220.16, +239.22]% | +232.50% |
| 248320 | broad | 0.7 | 1.0 | +718.06% | [+691.00, +767.61]% | +725.03% |

## Interpretation and next action

Sort-heavy benefit supports the qualitative prior; peaked nucleus and greedy
do not establish a useful gain. Small contrasts include host drift; do not
increase replicates to rescue these lines. The original-reference arm exposes
build/codegen drift; do not attribute a cross-build difference solely to radix.
No aggregate model gain exists: no model was run. Sort speedups are not additive
with GPU gains, and unknown end-to-end sampler fraction prevents a numeric
Amdahl estimate. Peak RAM, cold load, forward, serving and agent metrics remain
unmeasured. Extra radix scratch is a risk, not a measured peak-memory result.

Experimental cost: 270 fresh timed processes, 810 timed draws; summed sampler
ROI 3.155640 s. Three full sampler-oracle processes plus three edge processes.
GPU cost: zero. Total build/setup/developer wall time was not instrumented;
it is unknown, not substituted with ROI. No experimental-hours saving claimed.

Failures retained: initial build lacked required Windows _FILE_OFFSET_BITS=64
(fixed before timing); initial edge invocation omitted fixture mode (exit 2,
corrected before passing checks). No failed algorithm test was relabelled.
Plan frozen at 3fbd5c8f; Windows build repair at 5eabfe3b precedes timing.
Later ownership clarification and registry interface alignment change no
recorded prediction, input, generated header or result.

Next ranked test: parent-admitted held-out Qwen3.6 sampled serving, one fixed
model/quantisation and two histories, same-build opt-in off/on; first measure
sampler time fraction. Stop if diluted below useful resolution. Retain exact
text/RNG/count/stop; measure TTFT/ITL, throughput, peak RAM/VRAM, load and
cancel/recovery before G2 acceptance. No GPU work is admitted by this report.

Rollback: production unchanged; revert isolated Cascade commits to remove
the proposal. Parent integrates any future shared patch, through maintainer
review and matching G0–G5 gates. No CALR/AST compliance or assurance streak credit.
