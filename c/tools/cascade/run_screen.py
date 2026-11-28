"""Bounded CPU-only transfer screen. No production writes; stdlib only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[3]
DORY = "f84e14b1"


def run(args, **kwargs):
    try:
        return subprocess.check_output([str(a) for a in args], timeout=120, **kwargs)
    except subprocess.CalledProcessError as exc:
        print(exc.output.decode(errors="replace"))
        raise


def digest(data):
    return hashlib.sha256(data).hexdigest()


def generate_header():
    baseline = (ROOT / "c/qwen36_sample.h").read_bytes()
    donor = run(["git", "show", f"{DORY}:c/qwen36_sample.h"], cwd=ROOT)
    text = donor.decode()
    helpers = text[text.index("/* Explicit reference/fallback:"):text.index("/* The whole vocabulary sorted. */")]
    candidate = baseline.decode().replace(
        "/* The whole vocabulary sorted. */",
        "#include <float.h>\n#include <limits.h>\n" + helpers + "/* The whole vocabulary sorted. */")
    candidate = candidate.replace(
        "qsort(rank,(size_t)V,sizeof(SampleProb),sample_prob_desc);",
        "sample_prob_sort(rank,(size_t)V);").replace(
        "qsort(rank,(size_t)m,sizeof(SampleProb),sample_prob_desc);",
        "sample_prob_sort(rank,(size_t)m);")
    assert candidate.count("sample_prob_sort(rank,") == 2
    return baseline, donor, candidate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cc", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline, donor, candidate = generate_header()
    env = dict(os.environ)
    env.pop("COLI_LOGIT_DUMP", None)
    env.pop("DORY_CPU_SAMPLE_RADIX", None)
    results = {
        "experiment": "CASCADE-SAMPLE-001", "schema_version": 1,
        "scope": "synthetic CPU sampler only; not model inference",
        "source_revision": run(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
        "donor_revision": run(["git", "rev-parse", DORY], cwd=ROOT).decode().strip(),
        "platform": platform.platform(), "cpu": platform.processor(),
        "compiler": run([args.cc, "--version"]).decode().splitlines()[0],
        "flags": ["-std=c11", "-O2", "-D_FILE_OFFSET_BITS=64",
                  "-fno-fast-math", "-ffp-contract=off"],
        "sha256": {"baseline_header": digest(baseline), "donor_header": digest(donor),
                   "generated_header": digest(candidate.encode()),
                   "harness": digest(Path(__file__).with_name("sample_screen.c").read_bytes()),
                   "runner": digest(Path(__file__).read_bytes())},
        "conditions": {"threads": 1, "concurrency": 1, "warmups_per_process": 1,
                       "paired_repetitions": 5, "draws_per_process": 3,
                       "gpu": "not used", "model_quantisation": "not applicable",
                       "prompt_context": "deterministic synthetic logits; no KV",
                       "cold_load": "not measured", "clock": "monotonic wall clock"},
        "oracle": [], "cases": [], "unmeasured": [
            "model prefill/decode tokens/s", "HTTP end-to-end/tail latency",
            "total model throughput", "peak VRAM/RAM", "load time",
            "model quality", "serving/recovery/cancellation stability"],
    }
    with tempfile.TemporaryDirectory(prefix="cascade-sample-") as tmp:
        tmp = Path(tmp)
        generated = tmp / "candidate.h"
        generated.write_text(candidate)
        executables = {}
        for mode, header in [("baseline", ROOT / "c/qwen36_sample.h"), ("candidate", generated)]:
            exe = tmp / (mode + ".exe")
            command = [args.cc, *results["flags"], "-I", ROOT / "c",
                       f'-DCASCADE_HEADER="{header.as_posix()}"',
                       Path(__file__).with_name("sample_screen.c"), "-o", exe, "-lm"]
            run(command, env=env, stderr=subprocess.STDOUT)
            executables[mode] = exe
            results["sha256"][mode + "_executable"] = digest(exe.read_bytes())
            oracle = tmp / (mode + "_oracle.c")
            oracle.write_text((ROOT / "c/tests/test_qwen36_sample.c").read_text().replace(
                '#include "../qwen36_sample.h"', f'#include "{header.as_posix()}"'))
            oracle_exe = tmp / (mode + "_oracle.exe")
            run([args.cc, *results["flags"], "-I", ROOT / "c", oracle,
                 "-o", oracle_exe, "-lm"], env=env, stderr=subprocess.STDOUT)
            for flag in ([0] if mode == "baseline" else [0, 1]):
                test_env = dict(env, DORY_CPU_SAMPLE_RADIX=str(flag))
                results["oracle"].append({"mode": mode, "flag": flag,
                    "output": run([oracle_exe], env=test_env).decode().strip()})
        for vocab in (131072, 248320):
            for shape, name in enumerate(("flat_tied", "peaked", "broad")):
                for temp, top in ((0., .9), (.7, .9), (.7, 1.)):
                    record = {"vocab": vocab, "shape": name, "temperature": temp,
                              "top_p": top, "runs": []}
                    identities = []
                    by_mode = {m: [] for m in ("baseline", "off", "on")}
                    pair_effects = []
                    for repeat in range(5):
                        modes = ["baseline", "off", "on"]
                        if repeat % 2:
                            modes.reverse()
                        means = {}
                        for mode in modes:
                            exe = executables["baseline" if mode == "baseline" else "candidate"]
                            obs = json.loads(run([exe, vocab, shape, temp, top],
                                env=dict(env, DORY_CPU_SAMPLE_RADIX="1" if mode == "on" else "0")))
                            identities.append((obs["tokens"], obs["next_rng"]))
                            by_mode[mode].extend(obs["ms"])
                            means[mode] = statistics.mean(obs["ms"])
                            record["runs"].append({"repeat": repeat, "mode": mode, **obs})
                        pair_effects.append(100 * (means["baseline"] / means["on"] - 1))
                    assert all(i == identities[0] for i in identities), record
                    record["exact_tokens_and_rng"] = True
                    record["sampler_rate_change_pct_pairs"] = pair_effects
                    record["median_pair_rate_change_pct"] = statistics.median(pair_effects)
                    record["summary"] = {mode: {
                        "mean_ms": statistics.mean(values),
                        "median_ms": statistics.median(values),
                        "min_ms": min(values), "max_ms": max(values),
                        "p95_ms_nearest_rank": sorted(values)[-1],
                        "stdev_ms": statistics.stdev(values)}
                        for mode, values in by_mode.items()}
                    results["cases"].append(record)
    results["passed"] = True
    results["promoted"] = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({"passed": True, "oracle_runs": len(results["oracle"]),
                      "cases": len(results["cases"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
