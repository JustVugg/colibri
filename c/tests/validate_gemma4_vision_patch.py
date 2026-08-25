#!/usr/bin/env python3
"""Cross-check Gemma 4 patch convolution and X/Y positions against NumPy."""

from __future__ import annotations

import argparse
import pathlib
import struct
import subprocess
import tempfile

import numpy as np


SCALARS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i",
           6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


def exact(handle, count):
    data = handle.read(count)
    if len(data) != count:
        raise ValueError("unexpected end of GGUF")
    return data


def scalar(handle, kind):
    fmt = SCALARS[kind]
    return struct.unpack("<" + fmt, exact(handle, struct.calcsize(fmt)))[0]


def string(handle):
    return exact(handle, scalar(handle, 10)).decode("utf-8")


def value(handle, kind):
    if kind == 8:
        return string(handle)
    if kind == 9:
        element = scalar(handle, 4)
        return [value(handle, element) for _ in range(scalar(handle, 10))]
    return scalar(handle, kind)


def index_gguf(path):
    with path.open("rb") as handle:
        if exact(handle, 4) != b"GGUF":
            raise ValueError("not a GGUF")
        version, tensor_count, metadata_count = struct.unpack("<IQQ", exact(handle, 20))
        if version not in (2, 3):
            raise ValueError("unsupported GGUF version")
        metadata = {}
        for _ in range(metadata_count):
            key = string(handle)
            metadata[key] = value(handle, scalar(handle, 4))
        tensors = {}
        for _ in range(tensor_count):
            name = string(handle)
            dims = tuple(scalar(handle, 10) for _ in range(scalar(handle, 4)))
            tensors[name] = (scalar(handle, 4), dims, scalar(handle, 10))
        alignment = int(metadata.get("general.alignment", 32))
        data_offset = (handle.tell() + alignment - 1) // alignment * alignment
    return data_offset, metadata, tensors


def read_f32(path, data_offset, tensor):
    kind, dims, offset = tensor
    if kind != 0:
        raise ValueError("expected F32 tensor")
    count = int(np.prod(dims))
    return np.memmap(path, dtype="<f4", mode="r",
                     offset=data_offset + offset, shape=(count,))


def smart_size(width, height, alignment=48, minimum=40, maximum=280):
    aligned_w = max(alignment, round(width / alignment) * alignment)
    aligned_h = max(alignment, round(height / alignment) * alignment)
    min_pixels = minimum * alignment * alignment
    max_pixels = maximum * alignment * alignment
    if aligned_w * aligned_h > max_pixels:
        beta = np.sqrt(np.float32(width * height) / np.float32(max_pixels))
        aligned_w = max(alignment, int(np.floor(np.float32(width) / beta /
                                               alignment)) * alignment)
        aligned_h = max(alignment, int(np.floor(np.float32(height) / beta /
                                               alignment)) * alignment)
    elif aligned_w * aligned_h < min_pixels:
        beta = np.sqrt(np.float32(min_pixels) / np.float32(width * height))
        aligned_w = int(np.ceil(np.float32(width) * beta / alignment)) * alignment
        aligned_h = int(np.ceil(np.float32(height) * beta / alignment)) * alignment
    return aligned_w, aligned_h


def resize_reference(rgb, width, height):
    source_h, source_w, _ = rgb.shape
    result = np.empty((height, width, 3), dtype=np.uint8)
    x_ratio = np.float32(source_w - 1) / np.float32(width - 1) if width > 1 else np.float32(0)
    y_ratio = np.float32(source_h - 1) / np.float32(height - 1) if height > 1 else np.float32(0)
    for y in range(height):
        py = np.float32(y) * y_ratio
        y0 = min(int(py), source_h - 1)
        y1 = min(y0 + 1, source_h - 1)
        yf = py - np.float32(y0)
        for x in range(width):
            px = np.float32(x) * x_ratio
            x0 = min(int(px), source_w - 1)
            x1 = min(x0 + 1, source_w - 1)
            xf = px - np.float32(x0)
            top = rgb[y0, x0].astype(np.float32) + (
                rgb[y0, x1].astype(np.float32) - rgb[y0, x0]) * xf
            bottom = rgb[y1, x0].astype(np.float32) + (
                rgb[y1, x1].astype(np.float32) - rgb[y1, x0]) * xf
            result[y, x] = (top + (bottom - top) * yf).astype(np.uint8)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("executable", type=pathlib.Path)
    parser.add_argument("mmproj", type=pathlib.Path)
    args = parser.parse_args()
    data_offset, metadata, tensors = index_gguf(args.mmproj)
    source_w = source_h = 48
    source = ((np.arange(source_w * source_h * 3, dtype=np.uint64) * 37 + 11) & 255)
    source = source.astype(np.uint8).reshape(source_h, source_w, 3)
    target_w, target_h = smart_size(source_w, source_h)
    resized = resize_reference(source, target_w, target_h).astype(np.float32) / np.float32(255)
    mean = np.asarray(metadata["clip.vision.image_mean"], dtype=np.float32)
    std = np.asarray(metadata["clip.vision.image_std"], dtype=np.float32)
    scaled = np.float32(2) * ((resized - mean) / std) - np.float32(1)
    patch = int(metadata["clip.vision.patch_size"])
    model_width = int(metadata["clip.vision.embedding_length"])
    columns, rows = target_w // patch, target_h // patch
    patches = np.stack([
        scaled[y*patch:(y+1)*patch, x*patch:(x+1)*patch]
        .transpose(2, 0, 1).reshape(-1)
        for y in range(rows) for x in range(columns)
    ]).astype(np.float32)
    kernels = read_f32(args.mmproj, data_offset,
                       tensors["v.patch_embd.weight"]).reshape(model_width, -1)
    positions = read_f32(args.mmproj, data_offset,
                         tensors["v.position_embd.weight"])
    table_rows = tensors["v.position_embd.weight"][1][1]
    positions = positions.reshape(2, table_rows, model_width)
    reference = patches @ kernels.T
    for y in range(rows):
        for x in range(columns):
            reference[y * columns + x] += positions[0, x] + positions[1, y]
    with tempfile.TemporaryDirectory(prefix="gemma4-vision-patch-") as directory:
        output = pathlib.Path(directory) / "patches.f32"
        subprocess.run([str(args.executable), "vision-patch-probe", str(args.mmproj),
                        str(source_w), str(source_h), "--output-f32", str(output)],
                       check=True, stdout=subprocess.DEVNULL)
        actual = np.fromfile(output, dtype="<f4").reshape(reference.shape)
    difference = actual - reference
    maximum = float(np.max(np.abs(difference)))
    rms = float(np.sqrt(np.mean(difference * difference)))
    cosine = float(np.dot(actual.ravel(), reference.ravel()) /
                   (np.linalg.norm(actual) * np.linalg.norm(reference)))
    selected = [(0, 0), (0, 71), (0, 1151),
                (len(patches)//2, 1), (len(patches)-1, 511)]
    scalar_errors = []
    for patch_index, output_index in selected:
        y, x = divmod(patch_index, columns)
        total = np.float32(positions[0, x, output_index] +
                           positions[1, y, output_index])
        for input_index in range(patches.shape[1]):
            total = np.float32(total + np.float32(
                patches[patch_index, input_index] * kernels[output_index, input_index]))
        scalar_errors.append(abs(float(actual[patch_index, output_index] - total)))
    scalar_maximum = max(scalar_errors)
    print(f"patch embedding: blas-max={maximum:.6g} rms={rms:.6g} "
          f"cosine={cosine:.9f} scalar-max={scalar_maximum:.6g}")
    # BLAS is free to reassociate 768 float32 products, so its absolute maximum
    # is informational. The selected scalar oracle preserves the native loop's
    # accumulation order and catches layout/position errors directly.
    if scalar_maximum > 2e-4 or rms > 5e-4 or cosine < 0.9999995:
        raise SystemExit("patch embedding comparison failed")


if __name__ == "__main__":
    main()
