"""Inkling-2's image processor (transformers' Inkling2ImageProcessor) with Pillow and numpy.

A picture keeps its own resolution: it is padded to whole tiles of patch * merge pixels (40 on
Inkling-2-Small) with -1 on the 0-255 scale, rescaled by 1/255, normalized with mean and std 0.5,
and cut into one tile per language-model token, [rows * cols][tile][tile][3], the tiles in raster
order. A picture with transparency is composited onto the processor's checkerboard (8-pixel
squares of 204 and 153) first. The one change of ours: a picture of more than max_tokens tiles is
scaled down (bicubic) until it fits, since the tower and the prompt pay for every tile.
"""
import io
import json
import math
import os

DEFAULT_TILE = 40


def tile_of(model_dir):
    """patch * merge from config.json's vision_config (either spelling of the two keys)."""
    try:
        with open(os.path.join(model_dir, "config.json")) as f:
            vc = json.load(f).get("vision_config") or {}
    except (OSError, ValueError, TypeError):
        return DEFAULT_TILE
    patch = vc.get("patch_size", vc.get("native_patch_size", 20))
    merge = vc.get("spatial_merge_size", vc.get("adapter_merge_size", 2))
    return int(patch) * int(merge)


def _rgb(img, np):
    """RGB uint8 [H][W][3], transparency composited as Inkling2ImageProcessor.convert_to_rgb does."""
    if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info):
        rgba = np.asarray(img.convert("RGBA"), dtype=np.int32)
        if rgba[..., 3].min() < 255:
            h, w = rgba.shape[:2]
            bg = np.where(((np.arange(h)[:, None] // 8) + (np.arange(w)[None, :] // 8)) % 2 == 0, 204, 153)
            alpha = rgba[..., 3:4]
            return ((rgba[..., :3] * alpha + bg[..., None] * (255 - alpha) + 127) // 255).astype(np.uint8)
        return rgba[..., :3].astype(np.uint8)
    return np.asarray(img.convert("RGB"), dtype=np.uint8)


def preprocess(data, model_dir, max_tokens=None):
    """(tiles float32 [rows * cols][tile][tile][3], rows, cols) for one picture's bytes."""
    import numpy as np
    from PIL import Image
    img = Image.open(io.BytesIO(data))
    img.load()
    rgb = _rgb(img, np)
    tile = tile_of(model_dir) if model_dir else DEFAULT_TILE
    h, w = rgb.shape[:2]
    rows, cols = math.ceil(h / tile), math.ceil(w / tile)
    if max_tokens and rows * cols > max_tokens:
        scale = math.sqrt(max_tokens / (rows * cols))
        while True:
            nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
            if math.ceil(nh / tile) * math.ceil(nw / tile) <= max_tokens:
                break
            scale *= 0.97
        rgb = np.asarray(Image.fromarray(rgb).resize((nw, nh), Image.BICUBIC), dtype=np.uint8)
        h, w = nh, nw
        rows, cols = math.ceil(h / tile), math.ceil(w / tile)
    padded = np.full((rows * tile, cols * tile, 3), -1.0, dtype=np.float32)
    padded[:h, :w] = rgb
    x = (padded * np.float32(1 / 255) - np.float32(0.5)) / np.float32(0.5)
    tiles = x.reshape(rows, tile, cols, tile, 3).transpose(0, 2, 1, 3, 4).reshape(rows * cols, tile, tile, 3)
    return np.ascontiguousarray(tiles, dtype=np.float32), rows, cols
