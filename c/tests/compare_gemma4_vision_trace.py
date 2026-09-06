#!/usr/bin/env python3
"""Compare native and llama.cpp Gemma 4 vision operation traces."""

from __future__ import annotations

import argparse
import pathlib

import numpy as np


OPERATIONS = (
    "attention-input-norm",
    "query-norm",
    "key-norm",
    "value",
    "query-rope",
    "key-rope",
    "value-norm",
    "attention-context",
    "attention-output",
    "attention-post-norm",
    "attention-residual",
    "ffn-input-norm",
    "ffn-up",
    "ffn-gate",
    "ffn-activated",
    "ffn-output",
    "ffn-post-norm",
)


def metrics(native: np.ndarray, reference: np.ndarray) -> tuple[float, float, float]:
    if native.shape != reference.shape:
        raise ValueError(f"shape mismatch: {native.shape} != {reference.shape}")
    delta = native.astype(np.float64) - reference.astype(np.float64)
    maximum = float(np.max(np.abs(delta)))
    rms = float(np.sqrt(np.mean(delta * delta)))
    cosine = float(np.dot(native.astype(np.float64), reference.astype(np.float64)) /
                   (np.linalg.norm(native.astype(np.float64)) *
                    np.linalg.norm(reference.astype(np.float64))))
    return maximum, rms, cosine


def load(path: pathlib.Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"missing trace: {path}")
    values = np.fromfile(path, dtype="<f4")
    if not values.size:
        raise ValueError(f"empty trace: {path}")
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite trace: {path}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=pathlib.Path)
    parser.add_argument("--layer", type=int, default=17)
    parser.add_argument("--reference-directory", type=pathlib.Path)
    args = parser.parse_args()
    reference_directory = args.reference_directory or args.directory
    if args.layer < 0:
        parser.error("--layer must be non-negative")
    native_input = load(
        args.directory / f"native-vision-layer{args.layer}-input.f32")
    if args.layer == 0:
        reference_input_path = reference_directory / "llama-vision-input.f32"
    else:
        reference_input_path = reference_directory / (
            f"llama-vision-layer{args.layer}.f32")
    if reference_input_path.exists():
        reference_input = load(reference_input_path)
        maximum, rms, cosine = metrics(native_input, reference_input)
        print(f"{'input':24s} max={maximum:.8g} rms={rms:.8g} "
              f"cosine={cosine:.10g}")
    for operation in OPERATIONS:
        native_path = args.directory / (
            f"native-vision-layer{args.layer}-{operation}.f32")
        llama_path = reference_directory / (
            f"llama-vision-layer{args.layer}-{operation}.f32")
        native = load(native_path)
        reference = load(llama_path)
        maximum, rms, cosine = metrics(native, reference)
        print(f"{operation:24s} max={maximum:.8g} rms={rms:.8g} "
              f"cosine={cosine:.10g}")
    native = load(args.directory /
                  f"native-vision-layer{args.layer}-output.f32")
    reference = load(reference_directory /
                     f"llama-vision-layer{args.layer + 1}.f32")
    maximum, rms, cosine = metrics(native, reference)
    print(f"{'output':24s} max={maximum:.8g} rms={rms:.8g} "
          f"cosine={cosine:.10g}")


if __name__ == "__main__":
    main()
