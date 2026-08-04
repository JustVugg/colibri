#!/usr/bin/env python3
"""Build / validate a DeepSeek V4 oracle fixture for colibri.

This is intentional heavy validation — not part of light `make check`.

Comparison contract (documented in the JSON and docs/deepseek-v4.md):
  - top-1 token exact match for teacher-forcing and greedy windows
  - logits max-abs / top-k ranking are reserved for a future transformers
    oracle (`source=transformers`); coli-self fixtures only claim token match

Preferred source when transformers + DeepseekV4ForCausalLM are available:
  record reference tokens from the official implementation.

Fallback (current default on most machines):
  drive the target-only C engine with --record-oracle (source=coli-self).
  That proves reproducibility, not bit-exact parity with Hugging Face.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def try_transformers_oracle(model: str, prompt: str, max_new: int) -> dict | None:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # noqa: BLE001 — optional dependency
        print(f"[oracle] transformers unavailable ({exc}); using coli-self",
              file=sys.stderr)
        return None

    try:
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        model_obj = AutoModelForCausalLM.from_pretrained(
            model, trust_remote_code=True, torch_dtype=torch.bfloat16
        ).eval()
    except Exception as exc:  # noqa: BLE001
        print(f"[oracle] cannot load HF DeepSeek V4 ({exc}); using coli-self",
              file=sys.stderr)
        return None

    # Match colibri's default non-thinking chat template as closely as practical.
    encoded = tokenizer(prompt, return_tensors="pt")
    prompt_ids = encoded["input_ids"][0].tolist()
    with torch.no_grad():
        out = model_obj.generate(
            **encoded, max_new_tokens=max_new, do_sample=False, use_cache=True
        )
        full = out[0].tolist()
        logits = model_obj(torch.tensor([full]), use_cache=False).logits[0]
        tf_pred = logits.argmax(-1).tolist()
    return {
        "source": "transformers",
        "model": model,
        "prompt": prompt,
        "comparison": {
            "top1_token": "exact",
            "logits": "max abs error may be reported by future tooling",
        },
        "prompt_ids": prompt_ids,
        "full_ids": full,
        "tf_pred": tf_pred,
    }


def run_c(binary: Path, args: list[str], env: dict | None = None) -> subprocess.CompletedProcess:
    cmd = [str(binary), *args]
    print("[oracle] exec:", " ".join(cmd), file=sys.stderr)
    merged = os.environ.copy()
    if env:
        merged.update(env)
    return subprocess.run(cmd, check=False, env=merged)


def coli_self_oracle(
    binary: Path,
    model: str,
    prompt: str,
    output: Path,
    max_new: int,
    memory_gb: float | None,
) -> dict:
    args = [model, prompt, "--max-tokens", str(max_new),
            "--record-oracle", str(output)]
    if memory_gb is not None:
        args.extend(["--memory-gb", str(memory_gb)])
    proc = run_c(binary, args)
    if proc.returncode != 0:
        raise SystemExit(f"coli-self record failed with exit {proc.returncode}")
    return json.loads(output.read_text(encoding="utf-8"))


def validate(
    binary: Path,
    model: str,
    oracle_path: Path,
    teacher_forcing: int,
    greedy: int,
    memory_gb: float | None,
) -> int:
    args = [
        model,
        "--oracle", str(oracle_path),
        "--teacher-forcing", str(teacher_forcing),
        "--greedy", str(greedy),
    ]
    if memory_gb is not None:
        args.extend(["--memory-gb", str(memory_gb)])
    proc = run_c(binary, args)
    if proc.returncode != 0:
        print("[oracle] token validation FAILED", file=sys.stderr)
        return proc.returncode

    print("[oracle] target token validation OK", file=sys.stderr)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--binary", default="./deepseek_v4")
    parser.add_argument("--prompt", default="What is the capital of France?")
    parser.add_argument("--max-new", type=int, default=32)
    parser.add_argument("--memory-gb", type=float, default=None)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--teacher-forcing", type=int, default=32)
    parser.add_argument("--greedy", type=int, default=20)
    parser.add_argument(
        "--prefer-transformers",
        action="store_true",
        help="Try Hugging Face first; fall back to coli-self on failure",
    )
    args = parser.parse_args()

    binary = Path(args.binary)
    if os.name == "nt" and binary.suffix == "" and not binary.exists():
        candidate = binary.with_suffix(".exe")
        if candidate.exists():
            binary = candidate
    if not binary.exists():
        raise SystemExit(f"binary not found: {binary}")
    # Path("./deepseek_v4") stringifies to "deepseek_v4". subprocess would
    # then search PATH instead of executing the binary in the current directory.
    binary = binary.resolve()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    fixture = None
    if args.prefer_transformers:
        fixture = try_transformers_oracle(args.model, args.prompt, args.max_new)
        if fixture is not None:
            output.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")
            print(f"[oracle] wrote transformers fixture → {output}", file=sys.stderr)

    if fixture is None:
        fixture = coli_self_oracle(
            binary, args.model, args.prompt, output, args.max_new, args.memory_gb
        )
        print(f"[oracle] wrote coli-self fixture → {output}", file=sys.stderr)

    if not args.validate:
        return 0
    return validate(
        binary,
        args.model,
        output,
        args.teacher_forcing,
        args.greedy,
        args.memory_gb,
    )


if __name__ == "__main__":
    sys.exit(main())
