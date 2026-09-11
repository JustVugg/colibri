#!/usr/bin/env python3
"""Cross-check two-token Gemma 4 causal attention against NumPy."""

from __future__ import annotations

import argparse
import pathlib
import struct
import subprocess
import tempfile

import numpy as np


SCALAR_FORMATS = {
    0: "B",
    1: "b",
    2: "H",
    3: "h",
    4: "I",
    5: "i",
    6: "f",
    7: "?",
    10: "Q",
    11: "q",
    12: "d",
}


def read_exact(handle, count: int) -> bytes:
    value = handle.read(count)
    if len(value) != count:
        raise ValueError("unexpected end of GGUF")
    return value


def read_scalar(handle, value_type: int):
    try:
        value_format = SCALAR_FORMATS[value_type]
    except KeyError as error:
        raise ValueError(f"unsupported GGUF metadata type {value_type}") from error
    return struct.unpack("<" + value_format,
                         read_exact(handle, struct.calcsize(value_format)))[0]


def read_string(handle) -> str:
    length = read_scalar(handle, 10)
    return read_exact(handle, length).decode("utf-8")


def read_value(handle, value_type: int):
    if value_type == 8:
        return read_string(handle)
    if value_type == 9:
        element_type = read_scalar(handle, 4)
        length = read_scalar(handle, 10)
        return [read_value(handle, element_type) for _ in range(length)]
    return read_scalar(handle, value_type)


def tensor_index(path: pathlib.Path):
    with path.open("rb") as handle:
        if read_exact(handle, 4) != b"GGUF":
            raise ValueError("not a GGUF file")
        version, tensor_count, metadata_count = struct.unpack(
            "<IQQ", read_exact(handle, 20))
        if version not in (2, 3):
            raise ValueError(f"unsupported GGUF version {version}")
        alignment = 32
        selected_metadata = {}
        for _ in range(metadata_count):
            key = read_string(handle)
            value_type = read_scalar(handle, 4)
            value = read_value(handle, value_type)
            if key == "general.alignment":
                alignment = int(value)
            if key in ("gemma4.attention.head_count",
                       "gemma4.attention.layer_norm_rms_epsilon"):
                selected_metadata[key] = value
        tensors = {}
        for _ in range(tensor_count):
            name = read_string(handle)
            dimension_count = read_scalar(handle, 4)
            dimensions = tuple(read_scalar(handle, 10)
                               for _ in range(dimension_count))
            tensor_type = read_scalar(handle, 4)
            offset = read_scalar(handle, 10)
            tensors[name] = (tensor_type, dimensions, offset)
        data_offset = (handle.tell() + alignment - 1) // alignment * alignment
    return data_offset, tensors, selected_metadata


def q4_0_matvec(model: pathlib.Path, data_offset: int, tensor, vector):
    tensor_type, dimensions, offset = tensor
    if tensor_type != 2 or len(dimensions) != 2:
        raise ValueError("attention output must be a two-dimensional Q4_0 tensor")
    columns, rows = map(int, dimensions)
    if columns % 32 or vector.shape != (columns,):
        raise ValueError("attention output dimensions do not match context")
    blocks = columns // 32
    byte_count = rows * blocks * 18
    with model.open("rb") as handle:
        handle.seek(data_offset + offset)
        encoded = np.frombuffer(read_exact(handle, byte_count), dtype=np.uint8)
    encoded = encoded.reshape(rows, blocks, 18)
    scales = encoded[:, :, :2].copy().view("<f2").reshape(rows, blocks).astype(np.float32)
    packed = encoded[:, :, 2:]
    low = (packed & 15).astype(np.int16) - 8
    high = (packed >> 4).astype(np.int16) - 8
    blocked = vector.reshape(blocks, 32)
    dots = np.einsum("rbi,bi->rb", low.astype(np.float32), blocked[:, :16],
                     dtype=np.float32)
    dots += np.einsum("rbi,bi->rb", high.astype(np.float32), blocked[:, 16:],
                      dtype=np.float32)
    return np.sum(scales * dots, axis=1, dtype=np.float32)


def run(command):
    subprocess.run([str(part) for part in command], check=True,
                   stdout=subprocess.DEVNULL)


def validate_layer(executable: pathlib.Path, model: pathlib.Path, layer: int,
                   inputs: np.ndarray, data_offset: int, tensors,
                   query_heads: int):
    width = inputs.shape[1]
    prefix = f"blk.{layer}."
    output_tensor = tensors[prefix + "attn_output.weight"]
    context_width = int(output_tensor[1][0])
    query_tensor = tensors[prefix + "attn_q.weight"]
    key_tensor = tensors[prefix + "attn_k.weight"]
    query_width = int(query_tensor[1][1])
    kv_width = int(key_tensor[1][1])
    head_dim = query_width // query_heads
    kv_heads = kv_width // head_dim
    if context_width != query_width or query_heads % kv_heads:
        raise ValueError(f"invalid attention geometry for layer {layer}")
    queries_per_kv = query_heads // kv_heads

    with tempfile.TemporaryDirectory(prefix="gemma4-attention-") as directory:
        temporary = pathlib.Path(directory)
        input_path = temporary / "inputs.f32"
        output_path = temporary / "attention.f32"
        inputs.tofile(input_path)
        run([executable, "attention-seq", model, "--layer", layer,
             "--tokens", 2, "--input-f32", input_path,
             "--output-f32", output_path])
        queries, keys, values = [], [], []
        for position in range(2):
            token_path = temporary / f"input-{position}.f32"
            query_path = temporary / f"query-{position}.f32"
            key_path = temporary / f"key-{position}.f32"
            value_path = temporary / f"value-{position}.f32"
            inputs[position].tofile(token_path)
            run([executable, "attention-proj", model, "--layer", layer,
                 "--position", position, "--input-f32", token_path,
                 "--query-f32", query_path, "--key-f32", key_path,
                 "--value-f32", value_path])
            queries.append(np.fromfile(query_path, dtype="<f4").reshape(query_heads, head_dim))
            keys.append(np.fromfile(key_path, dtype="<f4").reshape(kv_heads, head_dim))
            values.append(np.fromfile(value_path, dtype="<f4").reshape(kv_heads, head_dim))

        reference = []
        for position in range(2):
            context = np.empty((query_heads, head_dim), dtype=np.float32)
            for head in range(query_heads):
                kv_head = head // queries_per_kv
                scores = np.asarray([
                    np.sum(queries[position][head] * keys[past][kv_head],
                           dtype=np.float32)
                    for past in range(position + 1)
                ], dtype=np.float32)
                weights = np.exp(scores - np.max(scores)).astype(np.float32)
                weights /= np.sum(weights, dtype=np.float32)
                context[head] = np.sum(
                    np.stack([weights[past] * values[past][kv_head]
                              for past in range(position + 1)]),
                    axis=0, dtype=np.float32)
            reference.append(q4_0_matvec(model, data_offset, output_tensor,
                                          context.reshape(context_width)))
        reference = np.concatenate(reference)
        actual = np.fromfile(output_path, dtype="<f4")
    difference = actual - reference
    return (float(np.max(np.abs(difference))),
            float(np.sqrt(np.mean(difference * difference, dtype=np.float64))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("executable", type=pathlib.Path)
    parser.add_argument("model", type=pathlib.Path)
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 5])
    arguments = parser.parse_args()
    executable = arguments.executable.resolve()
    model = arguments.model.resolve()
    data_offset, tensors, metadata = tensor_index(model)
    query_heads = metadata["gemma4.attention.head_count"]
    width = int(tensors["token_embd.weight"][1][0])
    indexes = np.arange(2 * width, dtype=np.float32)
    inputs = (np.sin(indexes * np.float32(0.017)) * np.float32(0.04) +
              np.cos(indexes * np.float32(0.003)) * np.float32(0.01))
    inputs = inputs.reshape(2, width).astype("<f4")
    for layer in arguments.layers:
        maximum, rms = validate_layer(executable, model, layer, inputs,
                                      data_offset, tensors, query_heads)
        print(f"layer {layer}: max_abs={maximum:.9g} rms={rms:.9g}")


if __name__ == "__main__":
    main()
