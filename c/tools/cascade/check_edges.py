"""Reuse Dory's complete-rank/edge oracle against the generated recipient header."""
import argparse
import json
import os
from pathlib import Path
import tempfile
from run_screen import ROOT, DORY, digest, generate_header, run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cc", required=True)
    args = p.parse_args()
    output = ROOT / "docs/experiments/cascade"
    evidence = json.loads((output / "sample-screen.json").read_text())
    _, _, candidate = generate_header()
    assert digest(candidate.encode()) == evidence["sha256"]["generated_header"]
    fixture = run(["git", "show", f"{DORY}:c/tools/dory_agent_path_radix_test.c"], cwd=ROOT)
    env = dict(os.environ)
    env.pop("COLI_LOGIT_DUMP", None)
    results = {"scope": "G1 complete rank, guards, fallback, scratch refusal, RNG and TLS checks only",
               "donor_fixture_sha256": digest(fixture),
               "generated_header_sha256": digest(candidate.encode()), "runs": []}
    with tempfile.TemporaryDirectory(prefix="cascade-edges-") as tmp:
        tmp = Path(tmp)
        header = tmp / "candidate.h"
        header.write_text(candidate)
        source = tmp / "edges.c"
        source.write_text(fixture.decode().replace(
            '#include "../qwen36_sample.h"', f'#include "{header.as_posix()}"'))
        exe = tmp / "edges.exe"
        run([args.cc, *evidence["flags"], "-I", ROOT / "c", source, "-o", exe, "-lm"],
            stderr=-2, env=env)
        results["executable_sha256"] = digest(exe.read_bytes())
        for flag in (None, "1", "other"):
            e = dict(env)
            e.pop("DORY_CPU_SAMPLE_RADIX", None)
            if flag is not None:
                e["DORY_CPU_SAMPLE_RADIX"] = flag
            results["runs"].append({"selector": flag, "output": run(
                [exe, "radix" if flag == "1" else "qsort"], env=e).decode().strip()})
    results["passed"] = True
    (output / "sample-edges.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
