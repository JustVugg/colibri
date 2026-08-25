#!/usr/bin/env python3
"""Compare Colibri's Gemma 4 vision tower with llama.cpp."""

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import tempfile

import numpy as np


TRACE_OPERATIONS = (
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


def run(command, env=None):
    completed = subprocess.run(command, env=env, text=True,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT)
    if completed.returncode:
        print(completed.stdout)
        raise SystemExit(f"command failed ({completed.returncode}): {command[0]}")


def metrics(actual, reference, label):
    if actual.shape != reference.shape:
        raise SystemExit(f"{label}: shape mismatch {actual.shape} != {reference.shape}")
    if not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise SystemExit(f"{label}: non-finite values")
    difference = actual.astype(np.float64) - reference.astype(np.float64)
    maximum = float(np.max(np.abs(difference)))
    rms = float(np.sqrt(np.mean(difference * difference)))
    a = actual.astype(np.float64).ravel()
    b = reference.astype(np.float64).ravel()
    cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    print(f"{label}: max={maximum:.7g} rms={rms:.7g} cosine={cosine:.9f}")
    return maximum, rms, cosine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("executable", type=pathlib.Path)
    parser.add_argument("oracle", type=pathlib.Path)
    parser.add_argument("model", type=pathlib.Path)
    parser.add_argument("packed", type=pathlib.Path)
    parser.add_argument("mmproj", type=pathlib.Path)
    parser.add_argument("--llama-bin", required=True, type=pathlib.Path)
    args = parser.parse_args()
    env = os.environ.copy()
    env["PATH"] = str(args.llama_bin) + os.pathsep + env.get("PATH", "")
    with tempfile.TemporaryDirectory(prefix="gemma4-vision-") as temporary:
        directory = pathlib.Path(temporary)
        trace_directory = directory / "operation-trace"
        trace_directory.mkdir()
        native_layer = directory / "native-layer1.f32"
        native_scaled_input = directory / "native-scaled-input.f32"
        native_layer9 = directory / "native-layer9.f32"
        native_layer18 = directory / "native-layer18.f32"
        native_exact_layer18 = directory / "native-exact-layer18.f32"
        native_last_layer = directory / "native-layer27.f32"
        native_projected = directory / "native-projected.f32"
        native_logits = directory / "native-image-logits.f32"
        decoder_logits = directory / "native-decoder-from-llama-image.f32"
        image = directory / "synthetic.ppm"
        pixels = bytes((index * 37 + 11) & 255 for index in range(48 * 48 * 3))
        image.write_bytes(b"P6\n48 48\n255\n" + pixels)
        run([str(args.executable), "vision-encode-probe", str(args.mmproj),
             "48", "48", "--layers", "1", "--prepared-f32",
             str(native_scaled_input), "--output-f32",
             str(native_layer)])
        run([str(args.oracle), str(args.model), str(args.mmproj), "48", "48",
             str(directory)], env=env)
        run([str(args.oracle), str(args.model), str(args.mmproj), "48", "48",
             str(trace_directory), "--trace-operations"], env=env)
        run([str(args.executable), "vision-encode-probe", str(args.mmproj),
             "48", "48", "--layers", "18", "--start-layer", "17",
             "--input-f32", str(trace_directory / "llama-vision-layer17.f32"),
             "--trace-layer", "17", "--trace-dir", str(trace_directory),
             "--output-f32", str(native_exact_layer18)])
        run([str(args.executable), "vision-encode-probe", str(args.mmproj),
             "48", "48", "--layers", "9", "--output-f32",
             str(native_layer9)])
        run([str(args.executable), "vision-encode-probe", str(args.mmproj),
             "48", "48", "--layers", "18", "--output-f32",
             str(native_layer18)])
        run([str(args.executable), "vision-encode-probe", str(args.mmproj),
             "48", "48", "--layers", "27", "--output-f32",
             str(native_last_layer)])
        run([str(args.executable), "vision-encode-probe", str(args.mmproj),
             "48", "48", "--output-f32", str(native_projected)])
        run([str(args.executable), "next-token", str(args.model),
             str(args.packed), "Describe this image.", "--chat",
             "--image", str(image), "--mmproj", str(args.mmproj),
             "--top", "1", "--logits-f32", str(native_logits)])
        run([str(args.executable), "next-token", str(args.model),
             str(args.packed), "Describe this image.", "--chat",
             "--image", str(image), "--mmproj", str(args.mmproj),
             "--image-embeddings-f32",
             str(directory / "llama-vision-projected.f32"),
             "--top", "1", "--logits-f32", str(decoder_logits)])
        native_scaled = np.fromfile(native_scaled_input, dtype="<f4")
        llama_scaled = np.fromfile(
            trace_directory / "llama-vision-scaled-input.f32", dtype="<f4")
        scaled = metrics(native_scaled, llama_scaled, "vision scaled input")
        if not np.array_equal(native_scaled, llama_scaled):
            raise SystemExit("vision scaled input differs from llama.cpp")
        layer = metrics(np.fromfile(native_layer, dtype="<f4"),
                        np.fromfile(directory / "llama-vision-layer1.f32",
                                    dtype="<f4"),
                        "vision layer 0")
        layer8 = metrics(np.fromfile(native_layer9, dtype="<f4"),
                         np.fromfile(directory / "llama-vision-layer9.f32",
                                     dtype="<f4"),
                         "vision layer 8")
        layer17 = metrics(np.fromfile(native_layer18, dtype="<f4"),
                          np.fromfile(directory / "llama-vision-layer18.f32",
                                      dtype="<f4"),
                          "vision layer 17")
        exact_layer17 = metrics(
            np.fromfile(native_exact_layer18, dtype="<f4"),
            np.fromfile(trace_directory / "llama-vision-layer18.f32",
                        dtype="<f4"),
            "vision layer 17 from exact llama.cpp input")
        for operation in TRACE_OPERATIONS:
            metrics(
                np.fromfile(
                    trace_directory / f"native-vision-layer17-{operation}.f32",
                    dtype="<f4"),
                np.fromfile(
                    trace_directory / f"llama-vision-layer17-{operation}.f32",
                    dtype="<f4"),
                f"vision layer 17 {operation}")
        last_layer = metrics(np.fromfile(native_last_layer, dtype="<f4"),
                             np.fromfile(directory / "llama-vision-layer27.f32",
                                         dtype="<f4"),
                             "vision layer 26")
        projected = metrics(np.fromfile(native_projected, dtype="<f4"),
                            np.fromfile(directory / "llama-vision-projected.f32",
                                        dtype="<f4"),
                            "vision projected")
        oracle_output = np.fromfile(directory / "llama-vision-output.f32",
                                    dtype="<f4")
        oracle_projected = np.fromfile(
            directory / "llama-vision-projected.f32", dtype="<f4")
        if not np.array_equal(oracle_output, oracle_projected):
            raise SystemExit("llama.cpp callback and public output differ")
        native_logit_values = np.fromfile(native_logits, dtype="<f4")
        llama_logit_values = np.fromfile(
            directory / "llama-image-logits.f32", dtype="<f4")
        logits = metrics(native_logit_values, llama_logit_values,
                         "image-conditioned logits")
        decoder_logit_values = np.fromfile(decoder_logits, dtype="<f4")
        decoder_logits_metric = metrics(
            decoder_logit_values, llama_logit_values,
            "decoder logits from llama.cpp image vectors")
        native_top = np.argsort(-native_logit_values)[:10]
        llama_top = np.argsort(-llama_logit_values)[:10]
        print("image-conditioned native top 10:",
              " ".join(f"{token}:{native_logit_values[token]:.6f}"
                       for token in native_top))
        print("image-conditioned llama top 10:",
              " ".join(f"{token}:{llama_logit_values[token]:.6f}"
                       for token in llama_top))
        if set(native_top) != set(llama_top):
            raise SystemExit(
                "image-conditioned top-10 candidate set differs from llama.cpp")
        if native_top[0] != llama_top[0]:
            raise SystemExit("image-conditioned top-1 token differs from llama.cpp")
        decoder_top = np.argsort(-decoder_logit_values)[:10]
        if not np.array_equal(decoder_top, llama_top):
            raise SystemExit("decoder top-10 ranking differs from llama.cpp")
    if layer[1] > 0.003 or layer[2] < 0.999999 or layer[0] > 0.2:
        raise SystemExit("vision layer-0 comparison failed")
    for checkpoint, label in ((layer8, "8"), (layer17, "17")):
        if checkpoint[1] > 1.0 or checkpoint[2] < 0.9999 or checkpoint[0] > 50.0:
            raise SystemExit(f"vision layer-{label} comparison failed")
    if exact_layer17[1] > 0.01 or exact_layer17[0] > 0.2:
        raise SystemExit("vision layer-17 exact-input operation trace failed")
    if last_layer[1] > 1.0 or last_layer[2] < 0.9999 or last_layer[0] > 50.0:
        raise SystemExit("vision layer-26 comparison failed")
    if projected[1] > 0.02 or projected[2] < 0.9999 or projected[0] > 0.3:
        raise SystemExit("vision projection comparison failed")
    if logits[1] > 0.25 or logits[2] < 0.999 or logits[0] > 2.0:
        raise SystemExit("image-conditioned logit comparison failed")
    if decoder_logits_metric[2] < 0.999 or decoder_logits_metric[0] > 2.0:
        raise SystemExit("image-conditioned decoder comparison failed")
    if os.environ.get("COLI_GEMMA4_VISION_COMPAT") == "llama-avx2":
        for checkpoint, label in (
                (layer, "0"), (layer8, "8"), (layer17, "17"),
                (exact_layer17, "17 exact-input"),
                (last_layer, "26"), (projected, "projection")):
            if checkpoint[0] != 0.0 or checkpoint[1] != 0.0:
                raise SystemExit(
                    f"llama-avx2 vision {label} is not bit-identical")


if __name__ == "__main__":
    main()
