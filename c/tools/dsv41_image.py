#!/usr/bin/env python3
"""Turn an image into DeepSeek-V4.1-Flash ViT patches, gateway side.

The engine reads patches, not pixels: this is the vendor's inference/image_processor.py
(`plan_image_grid` + `load_image`) with its resize rules kept exactly, because the grid
it chooses decides how many tokens the image costs and the engine reconstructs the span
layout from that same grid.

Returns (patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w):
  patches      float32 [n_vit_h * n_vit_w, 3 * patch * patch], the ViT's input
  n_vit_*      the patch grid
  n_llm_*      the token grid after the aligner's downsample, which is what the prompt
               span has to be sized for: 1 + (n_llm_w + 1) * n_llm_h + 1 placeholders
"""
from __future__ import annotations

import io
import json
import math
from pathlib import Path

DEFAULTS = dict(patch_size=14, downsample_ratio=3, max_image_tokens=1024,
                min_pixels=295936, max_wh_ratio=None)


def load_config(model_dir) -> dict:
    settings = dict(DEFAULTS)
    if model_dir:
        path = Path(model_dir) / "config.json"
        if path.is_file():
            vision = json.loads(path.read_text()).get("vision_config") or {}
            for key in DEFAULTS:
                if key in vision:
                    settings[key] = vision[key]
    return settings


def _llm_grid(height: int, width: int, patch: int, ratio: int):
    return math.ceil((height // patch) / ratio), math.ceil((width // patch) / ratio)


def _token_count(n_llm_h: int, n_llm_w: int) -> int:
    """image_processor.num_image_tokens: start, a newline per row, end."""
    return 1 + (n_llm_w + 1) * n_llm_h + 1


def _solve_resize(height, width, patch, ratio, max_tokens):
    """image_processor.solve_resize_ratio: the largest aspect-preserving size that fits."""
    r = height / width
    max_w = math.sqrt((max_tokens - 2) / r + 0.25) - 0.5
    max_h = max_w * r
    cell = patch * ratio
    if max_w < 1.0:
        return (max_tokens - 2) // 2 * cell, cell
    if max_h < 1.0:
        return cell, (max_tokens - 3) * cell
    beta = min(math.floor(max_w) * cell / width, math.floor(max_h) * cell / height)
    return (math.floor(height * beta / patch) * patch,
            math.floor(width * beta / patch) * patch)


def plan_grid(width: int, height: int, settings: dict, max_tokens=None):
    """image_processor.plan_image_grid + safe_resize."""
    patch = int(settings["patch_size"])
    ratio = int(settings["downsample_ratio"])
    budget = int(max_tokens or settings["max_image_tokens"])
    wh_ratio = settings.get("max_wh_ratio")
    if wh_ratio is not None and width > height * wh_ratio:
        width = int(height * wh_ratio)
    min_pixels = int(settings["min_pixels"])
    if 0 < width * height < min_pixels:
        scale = (min_pixels / (width * height)) ** 0.5
        width, height = int(width * scale), int(height * scale)
    best_w = math.ceil(width / patch) * patch
    best_h = math.ceil(height / patch) * patch
    n_llm_h, n_llm_w = _llm_grid(best_h, best_w, patch, ratio)
    if _token_count(n_llm_h, n_llm_w) > budget:
        best_h, best_w = _solve_resize(height, width, patch, ratio, budget)
        n_llm_h, n_llm_w = _llm_grid(best_h, best_w, patch, ratio)
    return n_llm_h, n_llm_w, best_h, best_w


def preprocess(source, model_dir=None, max_tokens=None):
    try:
        import numpy
        from PIL import Image, ImageOps
    except ImportError as exc:                                  # pragma: no cover
        raise SystemExit(f"image input needs Pillow and numpy: {exc}")
    settings = load_config(model_dir)
    patch = int(settings["patch_size"])
    data = source if isinstance(source, (bytes, bytearray)) else Path(source).read_bytes()
    with Image.open(io.BytesIO(data)) as opened:
        image = opened.convert("RGB")
        n_llm_h, n_llm_w, best_h, best_w = plan_grid(image.width, image.height, settings, max_tokens)
        wh_ratio = settings.get("max_wh_ratio")
        if wh_ratio is not None and image.width >= wh_ratio * image.height:
            image = image.resize((best_w, best_h))
        else:
            # pad, not crop: what a letterboxed image loses is canvas, not content
            image = ImageOps.pad(image, (best_w, best_h), color=(127, 127, 127))
        pixels = numpy.asarray(image, dtype=numpy.float32) / 255.0
    pixels = (pixels - 0.5) / 0.5                                # the vendor's normalization
    n_vit_h, n_vit_w = best_h // patch, best_w // patch
    # [H, W, 3] -> [n_h, n_w, 3, p, p] -> flat rows of 3 * p * p, the order PatchEmbed reads
    grid = pixels.reshape(n_vit_h, patch, n_vit_w, patch, 3).transpose(0, 2, 4, 1, 3)
    patches = numpy.ascontiguousarray(grid.reshape(n_vit_h * n_vit_w, 3 * patch * patch),
                                      dtype=numpy.float32)
    return patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w


def span_tokens(n_llm_h: int, n_llm_w: int) -> int:
    """How many placeholder ids the prompt must carry for this image."""
    return _token_count(n_llm_h, n_llm_w)
