#!/usr/bin/env python3
"""Cross-check Gemma 4's resident dense MLP and decoder block composition."""

from __future__ import annotations

import argparse
import pathlib
import tempfile

import numpy as np

import validate_gemma4_attention as common


def read_f32_tensor(model: pathlib.Path, data_offset: int, tensor):
    tensor_type, dimensions, offset = tensor
    if tensor_type != 0:
        raise ValueError("expected an F32 tensor")
    return np.fromfile(model, dtype="<f4", count=int(np.prod(dimensions)),
                       offset=data_offset + offset).copy()


def rmsnorm(values: np.ndarray, weight: np.ndarray, epsilon: float):
    sum_squares = np.float32(0.0)
    for value in values:
        sum_squares = np.float32(sum_squares + np.float32(value * value))
    mean = np.float32(sum_squares / np.float32(values.size))
    inverse = np.float32(1.0) / np.sqrt(np.float32(mean + epsilon))
    return ((values * inverse).astype(np.float32) * weight).astype(np.float32)


def dense_reference(model: pathlib.Path, data_offset: int, tensors,
                    layer: int, inputs: np.ndarray):
    prefix = f"blk.{layer}."
    gate = common.q4_0_matvec(model, data_offset,
                              tensors[prefix + "ffn_gate.weight"], inputs)
    up = common.q4_0_matvec(model, data_offset,
                            tensors[prefix + "ffn_up.weight"], inputs)
    coefficient = np.float32(0.7978845608028654)
    cubic = np.float32(0.044715)
    activated = np.float32(0.5) * gate * (
        np.float32(1.0) + np.tanh(coefficient *
                                  (gate + cubic * gate * gate * gate)))
    hidden = (activated * up).astype(np.float32)
    return common.q4_0_matvec(model, data_offset,
                              tensors[prefix + "ffn_down.weight"], hidden)


def difference(actual: np.ndarray, reference: np.ndarray):
    delta = actual - reference
    return (float(np.max(np.abs(delta))),
            float(np.sqrt(np.mean(delta * delta, dtype=np.float64))))


def validate_layer(executable: pathlib.Path, model: pathlib.Path,
                   packed: pathlib.Path, layer: int, inputs: np.ndarray,
                   data_offset: int, tensors, epsilon: float):
    prefix = f"blk.{layer}."
    width = inputs.shape[1]
    norms = {
        name: read_f32_tensor(model, data_offset, tensors[prefix + name])
        for name in (
            "post_attention_norm.weight",
            "ffn_norm.weight",
            "post_ffw_norm_1.weight",
            "pre_ffw_norm_2.weight",
            "post_ffw_norm_2.weight",
            "post_ffw_norm.weight",
        )
    }
    layer_scale = read_f32_tensor(
        model, data_offset, tensors[prefix + "layer_output_scale.weight"])[0]
    with tempfile.TemporaryDirectory(prefix="gemma4-layer-") as directory:
        temporary = pathlib.Path(directory)
        input_path = temporary / "inputs.f32"
        attention_path = temporary / "attention.f32"
        layer_path = temporary / "layer.f32"
        inputs.tofile(input_path)
        common.run([executable, "attention-seq", model, "--layer", layer,
                    "--tokens", 2, "--input-f32", input_path,
                    "--output-f32", attention_path])
        common.run([executable, "layer-seq", model, packed, "--layer", layer,
                    "--tokens", 2, "--input-f32", input_path,
                    "--output-f32", layer_path])
        attention = np.fromfile(attention_path, dtype="<f4").reshape(2, width)
        reference = []
        dense_errors = []
        for token in range(2):
            after_attention = (inputs[token] + rmsnorm(
                attention[token], norms["post_attention_norm.weight"],
                epsilon)).astype(np.float32)
            dense_input = rmsnorm(after_attention, norms["ffn_norm.weight"],
                                  epsilon)
            dense_input_path = temporary / f"dense-input-{token}.f32"
            dense_output_path = temporary / f"dense-output-{token}.f32"
            dense_input.tofile(dense_input_path)
            common.run([executable, "dense-mlp", model, "--layer", layer,
                        "--input-f32", dense_input_path,
                        "--output-f32", dense_output_path])
            dense_output = np.fromfile(dense_output_path, dtype="<f4")
            dense_errors.append(difference(
                dense_output,
                dense_reference(model, data_offset, tensors, layer, dense_input)))

            routed_input_path = temporary / f"routed-input-{token}.f32"
            routed_output_path = temporary / f"routed-output-{token}.f32"
            after_attention.tofile(routed_input_path)
            common.run([executable, "routed-mlp", model, packed,
                        "--layer", layer, "--input-f32", routed_input_path,
                        "--output-f32", routed_output_path])
            routed_output = np.fromfile(routed_output_path, dtype="<f4")
            dense_normalized = rmsnorm(
                dense_output, norms["post_ffw_norm_1.weight"], epsilon)
            routed_normalized = rmsnorm(
                routed_output, norms["post_ffw_norm_2.weight"], epsilon)
            combined = (dense_normalized + routed_normalized).astype(np.float32)
            block = rmsnorm(combined, norms["post_ffw_norm.weight"], epsilon)
            reference.append(((after_attention + block) * layer_scale).astype(np.float32))
        actual = np.fromfile(layer_path, dtype="<f4").reshape(2, width)
    return max(error[0] for error in dense_errors), difference(
        actual.reshape(-1), np.stack(reference).reshape(-1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("executable", type=pathlib.Path)
    parser.add_argument("model", type=pathlib.Path)
    parser.add_argument("packed", type=pathlib.Path)
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 5])
    arguments = parser.parse_args()
    executable = arguments.executable.resolve()
    model = arguments.model.resolve()
    packed = arguments.packed.resolve()
    data_offset, tensors, metadata = common.tensor_index(model)
    epsilon = float(metadata["gemma4.attention.layer_norm_rms_epsilon"])
    width = int(tensors["token_embd.weight"][1][0])
    indexes = np.arange(2 * width, dtype=np.float32)
    inputs = (np.sin(indexes * np.float32(0.011)) * np.float32(0.035) +
              np.cos(indexes * np.float32(0.007)) * np.float32(0.015))
    inputs = inputs.reshape(2, width).astype("<f4")
    for layer in arguments.layers:
        dense_maximum, (layer_maximum, layer_rms) = validate_layer(
            executable, model, packed, layer, inputs, data_offset, tensors,
            epsilon)
        print(f"layer {layer}: dense_max_abs={dense_maximum:.9g} "
              f"block_max_abs={layer_maximum:.9g} block_rms={layer_rms:.9g}")


if __name__ == "__main__":
    main()
