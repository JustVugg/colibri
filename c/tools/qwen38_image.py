#!/usr/bin/env python3
"""Da un'immagine alle patch che la torre vision di Qwen3.8 si aspetta.

Il motore C prende patch gia' estratte e non sa niente di JPEG, di
ridimensionamenti o di normalizzazioni: quel lavoro sta qui, dove Python c'e'
gia' per il template di chat e per il gateway.

Cosa fa, nell'ordine del processore ufficiale (`Qwen2VLImageProcessor`):

  1. decodifica e converte in RGB
  2. sceglie la tela con smart_resize -- aritmetica intera, allineata a 32
     (patch 16 per merge 2), dentro alla finestra di pixel del checkpoint
  3. ridimensiona, riscala di 1/255 e normalizza con media e deviazione 0.5
  4. spezza in patch nell'ordine che vuole la torre: a blocchi di merge,
     e dentro ogni patch canale, ripetizione temporale, riga, colonna

Due differenze da GLM-5.3 che contano, e che sono la ragione per cui questo
file esiste invece di riusare glm53_image.py.

**La risoluzione e' dinamica, non fissa.** GLM-5.3 porta tutto su una tela di
448 e riempie con zeri. Qwen conserva le proporzioni e sceglie la tela in modo
che l'AREA cada dentro [shortest_edge, longest_edge] pixel: niente padding, ma
il numero di token dipende dall'immagine. Un 1080p produce 2040 token, che su un
motore che legge gli esperti dal disco e' un prefill che nessuno aspetta -- lo
stesso motivo per cui GLM53_MAX_IMAGE_TOKENS esiste.

**La normalizzazione e' 0.5/0.5**, non le costanti CLIP.

Un avvertimento onesto sul punto 3, identico a quello di GLM-5.3. Il processore
di riferimento ridimensiona con torchvision, e riprodurne il bicubico senza
portarsi dietro torch non e' possibile: qui si usa quello di Pillow. Geometria,
ordine delle patch e normalizzazione sono identici e verificati contro il
riferimento; i valori dei pixel differiscono di quanto differiscono due
bicubici, che il test misura e scrive invece di dichiararlo zero.

USO:
  python3 tools/qwen38_image.py foto.jpg --out patches.f32
  python3 tools/qwen38_image.py foto.jpg --json          # solo la griglia
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

PATCH = 16
MERGE = 2
TEMPORAL = 2
# La finestra in PIXEL della tela, non in token: e' cosi' che la scrive il
# checkpoint (size.shortest_edge / size.longest_edge).
MIN_PIXELS = 65536
MAX_PIXELS = 16777216
MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)
# Il riferimento rifiuta oltre questo rapporto invece di produrre una tela
# degenere. Rifiutare e' la risposta giusta: un'immagine 1x500 non ha una
# rappresentazione sensata a patch quadrate.
MAX_RATIO = 200


def smart_resize(height, width, *, factor=PATCH * MERGE,
                 min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS):
    """La tela allineata a `factor`, con l'area dentro alla finestra.

    Aritmetica intera e nell'ordine del riferimento: prima si arrotonda al
    multiplo di factor, POI si corregge se l'area e' fuori finestra. Invertire i
    due passi da' tele diverse di una patch su certe forme, e una patch di
    differenza e' una griglia diversa, cioe' un numero di token diverso."""
    if height <= 0 or width <= 0:
        raise ValueError(f"dimensioni non valide: {width}x{height}")
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"rapporto {max(height, width) / min(height, width):.1f} oltre {MAX_RATIO}: "
            f"un'immagine cosi' allungata non ha una tela sensata a patch quadrate")
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def patchify(pixels, numpy, patch=PATCH, merge=MERGE, temporal=TEMPORAL):
    """[3, H, W] normalizzato -> [griglia, 3*T*16*16], nell'ordine della torre.

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

    preprocessor_config.json e' la fonte giusta, ma un export che non ce l'ha ha
    comunque vision_config dentro config.json: patch, merge e ripetizione
    temporale devono venire dal modello che si sta usando, non da costanti
    scritte qui, o su un checkpoint con una geometria diversa si taglierebbero
    patch della misura sbagliata senza che nulla protesti."""
    settings = {}
    if not model_dir:
        return settings
    root = Path(model_dir)
    processor = root / "preprocessor_config.json"
    if processor.exists():
        raw = json.loads(processor.read_text())
        for key in ("patch_size", "merge_size", "temporal_patch_size",
                    "image_mean", "image_std"):
            if key in raw:
                settings[key] = raw[key]
        size = raw.get("size") or {}
        if "shortest_edge" in size:
            settings["min_pixels"] = size["shortest_edge"]
        if "longest_edge" in size:
            settings["max_pixels"] = size["longest_edge"]
    config = root / "config.json"
    if config.exists():
        vision = json.loads(config.read_text()).get("vision_config") or {}
        for their, ours in (("patch_size", "patch_size"),
                            ("spatial_merge_size", "merge_size"),
                            ("temporal_patch_size", "temporal_patch_size")):
            if their in vision:
                settings.setdefault(ours, vision[their])
    return settings


def preprocess(source, model_dir=None, max_tokens=None):
    """Immagine -> (patch float32, grid_h, grid_w).

    `max_tokens` e' un tetto NOSTRO, non del checkpoint: la finestra ufficiale
    arriva a 2040 token per un 1080p, e su un motore che legge gli esperti dal
    disco quel prefill non lo aspetta nessuno. Rimpicciolisce, non ritaglia:
    quello che si perde e' dettaglio, non pezzi di immagine."""
    try:
        import numpy
        from PIL import Image
    except ImportError as exc:                    # pragma: no cover
        raise SystemExit(f"serve Pillow e numpy: {exc}")

    settings = load_config(model_dir)
    patch = int(settings.get("patch_size", PATCH))
    merge = int(settings.get("merge_size", MERGE))
    temporal = int(settings.get("temporal_patch_size", TEMPORAL))
    mean = tuple(settings.get("image_mean", MEAN))
    std = tuple(settings.get("image_std", STD))
    min_pixels = int(settings.get("min_pixels", MIN_PIXELS))
    max_pixels = int(settings.get("max_pixels", MAX_PIXELS))

    if max_tokens:
        # Un token copre merge*patch per lato, quindi un tetto in token e' un
        # tetto in pixel di tela.
        ceiling = int(max_tokens) * (merge * patch) ** 2
        max_pixels = min(max_pixels, ceiling)
        min_pixels = min(min_pixels, max_pixels)

    image = Image.open(source) if not hasattr(source, "mode") else source
    image = image.convert("RGB")
    width, height = image.size
    target_h, target_w = smart_resize(height, width, factor=patch * merge,
                                      min_pixels=min_pixels, max_pixels=max_pixels)
    if (target_w, target_h) != (width, height):
        image = image.resize((target_w, target_h), Image.BICUBIC)

    pixels = numpy.asarray(image, dtype=numpy.float32) / 255.0     # [H, W, 3]
    pixels = (pixels - numpy.array(mean, dtype=numpy.float32)) / numpy.array(
        std, dtype=numpy.float32)
    pixels = numpy.ascontiguousarray(pixels.transpose(2, 0, 1))    # [3, H, W]
    return patchify(pixels, numpy, patch=patch, merge=merge, temporal=temporal)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=Path, default=None,
                        help="cartella del checkpoint, per prendere la geometria vera")
    parser.add_argument("--out", type=Path, default=None, help="patch float32 grezze")
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--json", action="store_true", help="solo la griglia, su stdout")
    arguments = parser.parse_args()

    patches, grid_h, grid_w = preprocess(arguments.image, arguments.model,
                                         arguments.max_tokens)
    if arguments.json:
        print(json.dumps({"grid_h": grid_h, "grid_w": grid_w,
                          "patches": int(patches.shape[0]),
                          "features": int(patches.shape[1]),
                          "tokens": grid_h * grid_w // (MERGE * MERGE)}))
    if arguments.out:
        arguments.out.write_bytes(patches.tobytes())
        print(f"{patches.shape[0]} patch x {patches.shape[1]} -> {arguments.out}",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
