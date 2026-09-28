#!/usr/bin/env python3
"""Compare Gemma 4's generated fixture with its independent reference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np


def run(command):
    result = subprocess.run([str(x) for x in command], text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        raise SystemExit(f"command failed ({result.returncode}): {' '.join(map(str, command))}\n"
                         f"{result.stdout}\n{result.stderr}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", type=Path, default=Path("./gemma4"))
    parser.add_argument("--fixture", type=Path, default=Path("gemma4_tiny"))
    parser.add_argument("--cache", type=int, required=True)
    args = parser.parse_args()
    exe, fixture = args.exe.resolve(), args.fixture.resolve()
    ref = json.loads((fixture / "ref.json").read_text(encoding="utf-8"))
    model, packed = fixture / "model.gguf", fixture / "packed"
    with tempfile.TemporaryDirectory(prefix="gemma4-tiny-") as directory:
        logits_path = Path(directory) / "logits.f32"
        ranked = run([exe, "next-token", model, packed, ref["prompt"],
                      "--top", "3", "--expert-cache", args.cache,
                      "--logits-f32", logits_path])
        actual = np.fromfile(logits_path, dtype="<f4")
        expected = np.asarray(ref["logits"], dtype=np.float32)
        if actual.shape != expected.shape:
            raise SystemExit(f"logit shape differs: {actual.shape} != {expected.shape}")
        delta = actual - expected
        maximum = float(np.max(np.abs(delta)))
        denominator = float(np.linalg.norm(actual) * np.linalg.norm(expected))
        cosine = float(np.dot(actual, expected) / denominator) if denominator else 1.0
        if maximum > 0.02 or cosine < 0.99999:
            raise SystemExit(f"reference logits differ: max_abs={maximum:.7g} cosine={cosine:.9f}\n"
                             f"{ranked.stdout}\n{ranked.stderr}")
        generated = run([exe, "generate", model, packed, ref["prompt"],
                         "--max-new", len(ref["generated_tokens"]),
                         "--show-special", "--expert-cache", args.cache])
        marker = "generated:\n"
        if marker not in generated.stdout:
            raise SystemExit(f"generation marker missing:\n{generated.stdout}")
        text = generated.stdout.split(marker, 1)[1].splitlines()[0]
        if text != ref["generated_text"]:
            raise SystemExit(f"token stream differs: {text!r} != {ref['generated_text']!r}")
        print(f"PASS cap={args.cache}: token-exact, logits max_abs={maximum:.7g} cosine={cosine:.9f}")


if __name__ == "__main__":
    main()
