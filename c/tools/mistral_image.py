#!/usr/bin/env python3
"""A picture to the patches Mistral-Large-4's vision tower (Pixtral) reads.

The engine takes patches already cut and normalised; decoding, resizing and the
normalisation live here, beside the chat template, as for the other towers.

What it does, in the order of the release's processor (PixtralImageProcessor with
Mistral3's spatial merge):
  1. decodes the picture and converts it to RGB;
  2. scales it down, keeping its proportions, when a side passes the tower's
     longest edge (1540 pixels: floor, as the reference does), then to whole
     merged tokens: each side rounded up to a multiple of patch * merge (28);
  3. resizes it there (bicubic), rescales by 1/255 and normalises with the CLIP
     mean and std;
  4. cuts 14 x 14 patches in row-major order of the patch grid, each one channel,
     row, column (the layout of the mmproj's v.patch_embd.weight).

grid_h x grid_w are the merged tokens: 2 x 2 patches each, as many [IMG] tokens in
the prompt, a row of them closed by [IMG_BREAK] and the last by [IMG_END].

`max_tokens` is ours, not the release's: a picture at the tower's own 1540 pixels is
3025 tokens through a 30-layer tower and then a 1T model's prompt. Past it the
picture is shrunk, never cropped. The bicubic is Pillow's: the reference resizes
with Pillow too (PILImageResampling.BICUBIC).

USE:
  python3 tools/mistral_image.py picture.png --json
  python3 tools/mistral_image.py picture.png --out patches.f32
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

PATCH = 14
MERGE = 2
LONGEST_EDGE = 1540
MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)
MAX_TOKENS = 1024   # merged tokens; ML4_MAX_IMAGE_TOKENS changes it in the gateway


def load_config(model_dir):
    """The tower's geometry from the model directory (written by coli import beside a linked
    mmproj), else the release's numbers."""
    if model_dir:
        path = Path(model_dir) / "preprocessor_config.json"
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    return {}


def canvas(height, width, patch=PATCH, merge=MERGE, longest_edge=LONGEST_EDGE, max_tokens=None):
    """(canvas height, canvas width, grid_h, grid_w): the reference's resize, then ours."""
    if height < 1 or width < 1:
        raise ValueError("an empty picture")
    unit = patch * merge
    ratio = max(height / longest_edge, width / longest_edge)
    if ratio > 1:
        height, width = int(math.floor(height / ratio)), int(math.floor(width / ratio))
    while True:
        gh, gw = (max(height, 1) - 1) // unit + 1, (max(width, 1) - 1) // unit + 1
        if not max_tokens or gh * gw <= max_tokens or (height <= unit and width <= unit):
            return gh * unit, gw * unit, gh, gw
        shrink = math.sqrt(max_tokens / (gh * gw))
        height, width = max(1, int(math.floor(height * shrink))), max(1, int(math.floor(width * shrink)))


def patchify(pixels, numpy, patch=PATCH):
    """[3, H, W] -> [H/p * W/p, 3 * p * p]: the patches row-major, each channel, row, column."""
    c, h, w = pixels.shape
    gh, gw = h // patch, w // patch
    blocks = pixels.reshape(c, gh, patch, gw, patch).transpose(1, 3, 0, 2, 4)
    return numpy.ascontiguousarray(blocks.reshape(gh * gw, c * patch * patch), dtype=numpy.float32)


def preprocess(source, model_dir=None, max_tokens=None):
    """A picture (path, bytes or PIL image) -> (patches float32 [P, 588], grid_h, grid_w)."""
    try:
        import numpy
        from PIL import Image
    except ImportError as exc:                    # pragma: no cover
        raise ImportError(f"Mistral-Large-4's pictures need Pillow and numpy: {exc}") from exc
    settings = load_config(model_dir)
    patch = int(settings.get("patch_size", PATCH))
    merge = int(settings.get("spatial_merge_size", MERGE))
    longest = int(settings.get("longest_edge", LONGEST_EDGE))
    mean = tuple(settings.get("image_mean", MEAN))
    std = tuple(settings.get("image_std", STD))
    if isinstance(source, (bytes, bytearray)):
        import io
        source = io.BytesIO(source)
    image = Image.open(source) if not hasattr(source, "mode") else source
    image = image.convert("RGB")
    width, height = image.size
    target_h, target_w, gh, gw = canvas(height, width, patch, merge, longest, max_tokens)
    if (target_w, target_h) != (width, height):
        image = image.resize((target_w, target_h), Image.BICUBIC)
    pixels = numpy.asarray(image, dtype=numpy.float32) / 255.0                 # [H, W, 3]
    pixels = (pixels - numpy.array(mean, dtype=numpy.float32)) / numpy.array(std, dtype=numpy.float32)
    pixels = numpy.ascontiguousarray(pixels.transpose(2, 0, 1))                # [3, H, W]
    return patchify(pixels, numpy, patch), gh, gw


def placeholder(grid_h, grid_w, image="[IMG]", brk="[IMG_BREAK]", end="[IMG_END]"):
    """The prompt's text for a picture of grid_h x grid_w merged tokens."""
    rows = [image * grid_w + brk] * grid_h
    return "".join(rows)[: -len(brk)] + end


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=Path, default=None, help="the model directory (its preprocessor_config.json)")
    parser.add_argument("--out", type=Path, default=None, help="the patches, raw float32")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--json", action="store_true", help="the grid only, on stdout")
    arguments = parser.parse_args()
    try:
        patches, grid_h, grid_w = preprocess(arguments.image, arguments.model, arguments.max_tokens)
    except ImportError as missing:
        print(missing, file=sys.stderr)
        return 1
    if arguments.json:
        print(json.dumps({"grid_h": grid_h, "grid_w": grid_w, "patches": int(patches.shape[0]),
                          "features": int(patches.shape[1])}))
    if arguments.out:
        arguments.out.write_bytes(patches.tobytes())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
