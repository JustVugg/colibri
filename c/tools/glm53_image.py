#!/usr/bin/env python3
"""Da un'immagine alle patch che la torre vision di GLM-5.3 si aspetta.

Il motore C prende patch gia' estratte e non sa niente di JPEG, di
ridimensionamenti o di normalizzazioni: quel lavoro sta qui, dove Python c'e'
gia' per il template di chat e per il gateway.

Cosa fa, nell'ordine del processore ufficiale:

  1. decodifica e converte in RGB
  2. sceglie la tela con smart_resize -- aritmetica intera, allineata a 28
     (patch 14 per merge 2), dentro al budget di token del checkpoint
  3. ridimensiona, riscala di 1/255 e normalizza con media e deviazione CLIP
  4. spezza in patch nell'ordine che vuole la torre: a blocchi di merge,
     e dentro ogni patch canale, ripetizione temporale, riga, colonna

Un avvertimento onesto sul punto 3. Il processore di riferimento ridimensiona
con torchvision, e riprodurne il bicubico senza portarsi dietro torch non e'
possibile: qui si usa quello di Pillow. Geometria, ordine delle patch e
normalizzazione sono identici e verificati contro il riferimento; i valori dei
pixel differiscono di quanto differiscono due bicubici, che il test misura e
scrive invece di dichiararlo zero.

USO:
  python3 tools/glm53_image.py foto.jpg --out patches.f32
  python3 tools/glm53_image.py foto.jpg --json          # solo la griglia
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

# Dal processor_config.json del checkpoint. Restano qui come default e vengono
# sovrascritti da quel file quando lo si passa: due copie che divergono in
# silenzio sarebbero peggio di nessuna.
PATCH = 14
MERGE = 2
TEMPORAL = 2
MIN_TOKENS = 16
MAX_TOKENS = 8000
MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)


def smart_resize(height, width, *, temporal=TEMPORAL, factor=PATCH * MERGE,
                 min_tokens=MIN_TOKENS, max_tokens=MAX_TOKENS, frames=None):
    """La tela allineata dentro al budget. Aritmetica intera, come l'originale."""
    # Il riferimento passa num_frames = temporal_factor, non 1: conta nel
    # ramo che allarga un'immagine troppo piccola.
    if frames is None: frames = temporal
    pixels_per_token = temporal * factor ** 2
    min_pixels = min_tokens * pixels_per_token
    max_pixels = max_tokens * pixels_per_token

    def align(value):
        return math.ceil(value / factor) * factor

    aligned_frames = max(temporal, round(frames / temporal) * temporal)
    aligned_height, aligned_width = align(height), align(width)
    budget = aligned_frames * aligned_height * aligned_width

    if budget < min_pixels:
        scale = math.sqrt(min_pixels / (frames * height * width))
        aligned_height = align(max(1, math.ceil(height * scale)))
        aligned_width = align(max(1, math.ceil(width * scale)))
        budget = aligned_frames * aligned_height * aligned_width

    if budget > max_pixels:
        minimum = aligned_frames * factor ** 2
        if max_pixels < minimum:
            raise ValueError(f"max_tokens={max_tokens} non basta per una sola patch")
        low, high = 1, height
        best = (factor, factor)
        while low <= high:
            content_height = (low + high) // 2
            content_width = max(1, math.floor(width * content_height / height))
            candidate = (align(content_height), align(content_width))
            if aligned_frames * candidate[0] * candidate[1] <= max_pixels:
                best = candidate
                low = content_height + 1
            else:
                high = content_height - 1
        aligned_height, aligned_width = best

    return aligned_height, aligned_width


def patchify(pixels, numpy, patch=PATCH, merge=MERGE, temporal=TEMPORAL):
    """[3, H, W] normalizzato -> [griglia, 3*T*14*14], nell'ordine della torre.

    L'ordine non e' quello della griglia riga per riga: e' a blocchi di merge,
    perche' e' cosi' che il merger 2x2 poi li richiude. Dentro ogni patch i
    valori vanno canale, ripetizione temporale, riga, colonna."""
    channels, height, width = pixels.shape
    grid_h, grid_w = height // patch, width // patch
    blocks = pixels.reshape(channels, grid_h // merge, merge, patch,
                            grid_w // merge, merge, patch)
    # [gh/m, gw/m, m, m, C, p, p]
    blocks = blocks.transpose(1, 4, 2, 5, 0, 3, 6)
    blocks = numpy.repeat(blocks[..., None, :, :], temporal, axis=-3)
    return numpy.ascontiguousarray(
        blocks.reshape(grid_h * grid_w, channels * temporal * patch * patch),
        dtype=numpy.float32), grid_h, grid_w


def load_config(model_dir):
    """I parametri veri del checkpoint, quando c'e'.

    processor_config.json e' la fonte giusta, ma un export che non ce l'ha ha
    comunque vision_config dentro config.json: patch, merge e ripetizione
    temporale devono venire dal modello che si sta usando, non da costanti
    scritte qui, o su un checkpoint con una geometria diversa si taglierebbero
    patch della misura sbagliata senza che nulla protesti."""
    settings = {}
    if not model_dir:
        return settings
    root = Path(model_dir)
    processor = root / "processor_config.json"
    if processor.exists():
        settings.update(json.loads(processor.read_text()).get("image_processor", {}))
    config = root / "config.json"
    if config.exists():
        vision = json.loads(config.read_text()).get("vision_config") or {}
        for their, ours in (("patch_size", "patch_size"),
                            ("spatial_merge_size", "merge_size"),
                            ("temporal_patch_size", "temporal_patch_size")):
            if their in vision:
                settings.setdefault(ours, vision[their])
    return settings


def preprocess(source, model_dir=None):
    """Immagine (percorso, bytes o oggetto PIL) -> (patch, grid_h, grid_w)."""
    import numpy
    from PIL import Image

    settings = load_config(model_dir)
    mean = tuple(settings.get("image_mean", MEAN))
    std = tuple(settings.get("image_std", STD))
    min_tokens = settings.get("min_image_tokens", MIN_TOKENS)
    max_tokens = settings.get("max_image_tokens", MAX_TOKENS)
    # Il tetto del checkpoint e' 8000 token per immagine, che su una macchina
    # che streamma gli esperti da disco vuol dire un prefill lunghissimo per
    # una foto qualunque. GLM53_MAX_IMAGE_TOKENS lo abbassa: l'immagine viene
    # rimpicciolita, non tagliata, quindi si perde dettaglio e non pezzi.
    override = os.environ.get("GLM53_MAX_IMAGE_TOKENS")
    if override:
        try:
            max_tokens = max(min_tokens, int(override))
        except ValueError:
            pass
    patch = settings.get("patch_size", PATCH)
    merge = settings.get("merge_size", MERGE)
    temporal = settings.get("temporal_patch_size", TEMPORAL)

    if isinstance(source, (str, Path)):
        image = Image.open(source)
    elif isinstance(source, bytes):
        import io
        image = Image.open(io.BytesIO(source))
    else:
        image = source
    image = image.convert("RGB")

    height, width = image.height, image.width
    target_h, target_w = smart_resize(height, width, temporal=temporal,
                                      factor=patch * merge, min_tokens=min_tokens,
                                      max_tokens=max_tokens)

    # L'immagine NON viene stirata sulla tela: si scala mantenendo il rapporto
    # e il resto della tela resta nero. Stirarla cambierebbe le proporzioni di
    # tutto quello che il modello guarda, e non darebbe nessun errore.
    scale = min(target_h / height, target_w / width)
    pixels_per_token = temporal * (patch * merge) ** 2
    if temporal * height * width >= pixels_per_token * min_tokens:
        scale = min(1.0, scale)                 # gia' abbastanza grande: non si ingrandisce
    content_h = max(1, min(target_h, math.floor(height * scale)))
    content_w = max(1, min(target_w, math.floor(width * scale)))
    if (content_h, content_w) != (height, width):
        image = image.resize((content_w, content_h), Image.BICUBIC)

    # Il riempimento e' a zero PRIMA della normalizzazione, quindi nella tela
    # finita vale (0 - media)/deviazione e non zero.
    canvas = numpy.zeros((target_h, target_w, 3), numpy.float32)
    canvas[:content_h, :content_w] = numpy.asarray(image, dtype=numpy.float32)
    pixels = canvas.transpose(2, 0, 1) / 255.0
    for channel in range(3):
        pixels[channel] = (pixels[channel] - mean[channel]) / std[channel]
    return patchify(pixels, numpy, patch, merge, temporal)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--out", type=Path, help="dove scrivere le patch f32")
    parser.add_argument("--model", type=Path, help="checkpoint da cui leggere i parametri")
    parser.add_argument("--json", action="store_true", help="stampa solo la griglia")
    arguments = parser.parse_args()

    patches, grid_h, grid_w = preprocess(arguments.image, arguments.model)
    merge = load_config(arguments.model).get("merge_size", MERGE)
    tokens = (grid_h // merge) * (grid_w // merge)
    if arguments.out:
        arguments.out.write_bytes(patches.tobytes())
    if arguments.json:
        print(json.dumps({"grid_h": grid_h, "grid_w": grid_w,
                          "patches": int(patches.shape[0]),
                          "image_tokens": tokens}))
    else:
        print(f"{arguments.image.name}: griglia {grid_h}x{grid_w}, "
              f"{patches.shape[0]} patch da {patches.shape[1]}, "
              f"{tokens} token immagine")
    return 0


if __name__ == "__main__":
    sys.exit(main())
