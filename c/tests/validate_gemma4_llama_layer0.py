#!/usr/bin/env python3
"""Cross-check Colibri's first Gemma 4 decoder layer against llama.cpp."""

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import tempfile

import numpy as np

import validate_gemma4_attention as common
import validate_gemma4_layer as layer


LIMITS = {
    "query": 0.05,
    "key": 0.005,
    "value": 0.05,
    "attention_postnorm": 0.5,
    "dense_input": 0.2,
    "router_probability": 0.00001,
    "layer_output": 0.5,
}
MINIMUM_COSINE = 0.999


def run(command, environment=None, quiet=False):
    result = subprocess.run(
        [str(part) for part in command], env=environment,
        text=True, stdout=subprocess.PIPE if quiet else None,
        stderr=subprocess.PIPE if quiet else None,
    )
    if result.returncode:
        detail = "\n".join((result.stdout or "", result.stderr or "")).strip()
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(map(str, command))}\n{detail[-4000:]}")
    return result


def metric(actual: np.ndarray, reference: np.ndarray):
    actual = actual.astype(np.float32, copy=False).reshape(-1)
    reference = reference.astype(np.float32, copy=False).reshape(-1)
    delta = actual - reference
    denominator = float(np.linalg.norm(actual) * np.linalg.norm(reference))
    cosine = float(np.dot(actual, reference) / denominator) if denominator else 1.0
    cosine = min(1.0, max(-1.0, cosine))
    return (float(np.max(np.abs(delta))),
            float(np.sqrt(np.mean(delta * delta, dtype=np.float64))),
            cosine)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("executable", type=pathlib.Path)
    parser.add_argument("oracle", type=pathlib.Path)
    parser.add_argument("model", type=pathlib.Path)
    parser.add_argument("packed", type=pathlib.Path)
    parser.add_argument("--llama-bin", type=pathlib.Path,
                        help="directory containing llama.cpp runtime DLLs")
    parser.add_argument("--prompt", default="A",
                        help="oracle prompt; the default tokenizes to BOS plus one token")
    arguments = parser.parse_args()
    executable = arguments.executable.resolve()
    oracle = arguments.oracle.resolve()
    model = arguments.model.resolve()
    packed = arguments.packed.resolve()

    data_offset, tensors, metadata = common.tensor_index(model)
    prefix = "blk.0."
    width = int(tensors[prefix + "post_attention_norm.weight"][1][0])
    query_width = int(tensors[prefix + "attn_q.weight"][1][1])
    key_width = int(tensors[prefix + "attn_k.weight"][1][1])
    value_width = int(tensors[prefix + "attn_v.weight"][1][1])
    epsilon = np.float32(metadata["gemma4.attention.layer_norm_rms_epsilon"])
    oracle_environment = os.environ.copy()
    if arguments.llama_bin:
        oracle_environment["PATH"] = (str(arguments.llama_bin.resolve()) +
                                      os.pathsep + oracle_environment.get("PATH", ""))

    results = []
    with tempfile.TemporaryDirectory(prefix="gemma4-llama-") as directory:
        temporary = pathlib.Path(directory)
        oracle_result = run([oracle, model, arguments.prompt, temporary],
                            environment=oracle_environment, quiet=True)
        for line_text in (oracle_result.stdout or "").splitlines():
            if line_text.startswith(("captured ", "tokens:")):
                print(line_text)

        inputs = np.fromfile(temporary / "llama-input.f32", dtype="<f4")
        if inputs.size % width:
            raise ValueError("llama.cpp input tensor has an invalid size")
        inputs = inputs.reshape(-1, width)
        token_count = inputs.shape[0]
        references = {
            "query": np.fromfile(temporary / "llama-query.f32", dtype="<f4").reshape(token_count, query_width),
            "key": np.fromfile(temporary / "llama-key.f32", dtype="<f4").reshape(token_count, key_width),
            "value": np.fromfile(temporary / "llama-value.f32", dtype="<f4").reshape(token_count, value_width),
        }
        actual = {name: [] for name in references}
        for position, values in enumerate(inputs):
            input_path = temporary / f"input-{position}.f32"
            query_path = temporary / f"colibri-query-{position}.f32"
            key_path = temporary / f"colibri-key-{position}.f32"
            value_path = temporary / f"colibri-value-{position}.f32"
            values.tofile(input_path)
            run([executable, "attention-proj", model, "--layer", 0,
                 "--position", position, "--input-f32", input_path,
                 "--query-f32", query_path, "--key-f32", key_path,
                 "--value-f32", value_path], quiet=True)
            actual["query"].append(np.fromfile(query_path, dtype="<f4"))
            actual["key"].append(np.fromfile(key_path, dtype="<f4"))
            actual["value"].append(np.fromfile(value_path, dtype="<f4"))
        for name in references:
            results.append((name, *metric(np.stack(actual[name]), references[name])))

        attention_path = temporary / "colibri-attention.f32"
        run([executable, "attention-seq", model, "--layer", 0,
             "--tokens", token_count, "--input-f32", temporary / "llama-input.f32",
             "--output-f32", attention_path], quiet=True)
        raw_attention = np.fromfile(attention_path, dtype="<f4").reshape(token_count, width)
        post_weight = layer.read_f32_tensor(
            model, data_offset, tensors[prefix + "post_attention_norm.weight"])
        post_attention = np.stack([
            layer.rmsnorm(values, post_weight, epsilon)
            for values in raw_attention
        ])
        reference_post = np.fromfile(
            temporary / "llama-attention-postnorm.f32", dtype="<f4").reshape(token_count, width)
        results.append(("attention_postnorm", *metric(post_attention, reference_post)))

        after_attention = inputs + reference_post
        dense_weight = layer.read_f32_tensor(
            model, data_offset, tensors[prefix + "ffn_norm.weight"])
        dense_input = np.stack([
            layer.rmsnorm(values, dense_weight, epsilon)
            for values in after_attention
        ])
        reference_dense = np.fromfile(
            temporary / "llama-dense-input.f32", dtype="<f4").reshape(token_count, width)
        results.append(("dense_input", *metric(dense_input, reference_dense)))

        router_logits = np.fromfile(
            temporary / "llama-router-logits.f32", dtype="<f4").reshape(token_count, -1)
        router_actual = []
        for position, values in enumerate(after_attention):
            input_path = temporary / f"router-input-{position}.f32"
            probability_path = temporary / f"router-probability-{position}.f32"
            values.astype(np.float32).tofile(input_path)
            run([executable, "route", model, "--layer", 0,
                 "--input-f32", input_path,
                 "--probabilities-f32", probability_path], quiet=True)
            router_actual.append(np.fromfile(probability_path, dtype="<f4"))
        shifted = router_logits - np.max(router_logits, axis=1, keepdims=True)
        router_reference = np.exp(shifted).astype(np.float32)
        router_reference /= np.sum(router_reference, axis=1, keepdims=True,
                                   dtype=np.float32)
        router_actual = np.stack(router_actual)
        if not np.array_equal(np.argsort(-router_actual, axis=1)[:, :8],
                              np.argsort(-router_reference, axis=1)[:, :8]):
            raise SystemExit("Colibri and llama.cpp selected different top-8 experts")
        results.append(("router_probability",
                        *metric(router_actual, router_reference)))

        layer_path = temporary / "colibri-layer0.f32"
        run([executable, "layer-seq", model, packed, "--layer", 0,
             "--tokens", token_count, "--input-f32", temporary / "llama-input.f32",
             "--output-f32", layer_path], quiet=True)
        actual_layer = np.fromfile(layer_path, dtype="<f4").reshape(token_count, width)
        reference_layer = np.fromfile(
            temporary / "llama-layer0.f32", dtype="<f4").reshape(token_count, width)
        results.append(("layer_output", *metric(actual_layer, reference_layer)))

    failed = False
    print("\nboundary                 max_abs          rms       cosine")
    for name, maximum, rms, cosine in results:
        print(f"{name:22s} {maximum:12.6g} {rms:12.6g} {cosine:12.9f}")
        failed |= maximum > LIMITS[name] or cosine < MINIMUM_COSINE
    if failed:
        raise SystemExit("llama.cpp oracle comparison exceeded its tolerance")
    print("llama.cpp layer-0 oracle: PASS")


if __name__ == "__main__":
    main()
