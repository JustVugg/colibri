"""Derive same-build contrasts; never pool distinct workload blocks."""
import json
from pathlib import Path
import statistics

OUT = Path(__file__).resolve().parents[3] / "docs/experiments/cascade"


def paired(case, control):
    pairs = []
    for repeat in range(5):
        means = {r["mode"]: statistics.mean(r["ms"])
                 for r in case["runs"] if r["repeat"] == repeat}
        pairs.append(100 * (means[control] / means["on"] - 1))
    return pairs


def main():
    raw = json.loads((OUT / "sample-screen.json").read_text())
    report = ["# CASCADE-SAMPLE-001 result", "",
        "**Decision: retain NO_CHANGE in production. Bounded CPU mechanism evidence only.**",
        "Reuses ADAPTIVE_TRANSFER_DOE and G0–G5; this is not model G2/G3/G4/G5.",
        "Default-off generated header, no production or installed change.", "",
        "All 18 cases have exact token/next-RNG agreement across all arms.",
        "Baseline, candidate-off and candidate-on each pass the existing 1612-draw oracle.",
        "Complete-rank/fallback/scratch-refusal/TLS checks: 609 checks × 3 modes, zero failures.", "",
        "Primary effect: median of five **same-build off/on** sampler-rate contrasts.",
        "Range is observed paired min/max, **not a confidence interval**. Each process",
        "has one warmup and three draws. Vocabulary, shape and sampling policy are",
        "distinct workload blocks, never pooled as independent replicates.", "",
        "| Vocab | Shape | T | p | Same-build rate change | Paired range | Original-reference rate change |",
        "|---:|---|---:|---:|---:|---:|---:|"]
    derived = []
    timed = 0.
    for c in raw["cases"]:
        pairs = paired(c, "off")
        median = statistics.median(pairs)
        report.append(f'| {c["vocab"]} | {c["shape"]} | {c["temperature"]} | {c["top_p"]} | '
                      f'{median:+.2f}% | [{min(pairs):+.2f}, {max(pairs):+.2f}]% | '
                      f'{c["median_pair_rate_change_pct"]:+.2f}% |')
        derived.append({k: c[k] for k in ("vocab", "shape", "temperature", "top_p")} |
                       {"same_build_pairs_pct": pairs, "same_build_median_pct": median,
                        "original_reference_median_pct": c["median_pair_rate_change_pct"]})
        timed += sum(sum(r["ms"]) for r in c["runs"]) / 1000.
    report += ["", "## Interpretation and next action", "",
        "Sort-heavy benefit supports the qualitative prior; peaked nucleus and greedy",
        "do not establish a useful gain. Small contrasts include host drift; do not",
        "increase replicates to rescue these lines. The original-reference arm exposes",
        "build/codegen drift; do not attribute a cross-build difference solely to radix.",
        "No aggregate model gain exists: no model was run. Sort speedups are not additive",
        "with GPU gains, and unknown end-to-end sampler fraction prevents a numeric",
        "Amdahl estimate. Peak RAM, cold load, forward, serving and agent metrics remain",
        "unmeasured. Extra radix scratch is a risk, not a measured peak-memory result.", "",
        f"Experimental cost: 270 fresh timed processes, 810 timed draws; summed sampler",
        f"ROI {timed:.6f} s. Three full sampler-oracle processes plus three edge processes.",
        "GPU cost: zero. Total build/setup/developer wall time was not instrumented;",
        "it is unknown, not substituted with ROI. No experimental-hours saving claimed.", "",
        "Failures retained: initial build lacked required Windows _FILE_OFFSET_BITS=64",
        "(fixed before timing); initial edge invocation omitted fixture mode (exit 2,",
        "corrected before passing checks). No failed algorithm test was relabelled.",
        "Plan frozen at 3fbd5c8f; Windows build repair at 5eabfe3b precedes timing.",
        "Later ownership clarification and registry interface alignment change no",
        "recorded prediction, input, generated header or result.", "",
        "Next ranked test: parent-admitted held-out Qwen3.6 sampled serving, one fixed",
        "model/quantisation and two histories, same-build opt-in off/on; first measure",
        "sampler time fraction. Stop if diluted below useful resolution. Retain exact",
        "text/RNG/count/stop; measure TTFT/ITL, throughput, peak RAM/VRAM, load and",
        "cancel/recovery before G2 acceptance. No GPU work is admitted by this report.", "",
        "Rollback: production unchanged; revert isolated Cascade commits to remove",
        "the proposal. Parent integrates any future shared patch, through maintainer",
        "review and matching G0–G5 gates. No CALR/AST compliance or assurance streak credit."]
    (OUT / "RESULTS.md").write_text("\n".join(report) + "\n")
    (OUT / "contrasts.json").write_text(json.dumps(
        {"experiment": raw["experiment"], "same_build_primary": True,
         "timed_roi_seconds": timed, "total_wall_seconds": None,
         "cases": derived}, indent=2) + "\n")
    print("\n".join(report[:35]))


if __name__ == "__main__":
    main()
