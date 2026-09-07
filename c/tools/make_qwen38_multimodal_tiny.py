#!/usr/bin/env python3
"""Fixture multimodale tiny per Qwen3.8: testo piu' torre vision, in un modello solo.

Unisce quello che gia' sanno fare make_qwen38_tiny.py (il modello di testo) e
make_qwen38_vision_tiny.py (la torre) in un checkpoint che il motore carica
davvero, cosi' il percorso completo -- frame IMAGE, torre, sostituzione degli
embedding, generazione -- si puo' provare senza i 185 GB.

Il test che ci gira sopra non verifica che il modello risponda: con pesi casuali
risponderebbe comunque, e risponderebbe anche ignorando del tutto l'immagine.
Verifica che DUE IMMAGINI DIVERSE DIANO DUE RISPOSTE DIVERSE, che e' l'unica
domanda a cui una fixture casuale sa rispondere onestamente.

USO:
  python3 tools/make_qwen38_multimodal_tiny.py --out ./qwen38_mm_tiny
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SEED = 20260830


def build(out: Path, *, grid_h=4, grid_w=4, seed=SEED):
    import torch
    from safetensors.torch import save_file
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpVisionConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpVisionModel

    import make_qwen38_tiny

    out.mkdir(parents=True, exist_ok=True)
    text_dir = out / "_text"
    make_qwen38_tiny.build(text_dir, emit_ref=False)
    text_config = json.loads((text_dir / "config.json").read_text())

    vocab = text_config.get("vocab_size") or text_config["text_config"]["vocab_size"]
    hidden = text_config.get("hidden_size") or text_config["text_config"]["hidden_size"]
    # Il segnaposto immagine deve stare nel vocabolario del modello di testo, o
    # il motore rifiuterebbe il token prima ancora di arrivare alla torre.
    image_token = vocab - 1

    vision = Qwen4ExpVisionConfig(
        depth=2, hidden_size=64, num_heads=4, intermediate_size=128,
        patch_size=16, spatial_merge_size=2, temporal_patch_size=2,
        in_channels=3, out_hidden_size=hidden, num_position_embeddings=64,
        hidden_act="gelu_pytorch_tanh",
    )
    torch.manual_seed(seed)
    tower = Qwen4ExpVisionModel(vision).eval()
    with torch.no_grad():
        for parameter in tower.parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.6)

    from safetensors.torch import load_file
    tensors = dict(load_file(str(text_dir / "model.safetensors")))
    for name, value in tower.state_dict().items():
        tensors[f"model.visual.{name}"] = value.detach().contiguous()
    save_file(tensors, str(out / "model.safetensors"))

    merged = dict(text_config)
    merged["vision_config"] = vision.to_dict()
    merged["image_token_id"] = image_token
    (out / "config.json").write_text(json.dumps(merged, indent=2))
    for extra in ("generation_config.json",):
        source = text_dir / extra
        if source.exists():
            (out / extra).write_text(source.read_text())

    features = vision.in_channels * vision.temporal_patch_size * vision.patch_size ** 2
    tokens = (grid_h * grid_w) // (vision.spatial_merge_size ** 2)
    (out / "vision_meta.json").write_text(json.dumps({
        "grid_h": grid_h, "grid_w": grid_w, "features": features,
        "image_token": image_token, "tokens": tokens,
    }))
    print(f"  fixture multimodale: vocab {vocab}, hidden {hidden}, "
          f"segnaposto immagine {image_token}, {tokens} token per immagine")
    print(f"  scritta in {out}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        build(arguments.out)
    except ImportError as exc:
        raise SystemExit(f"serve transformers e safetensors: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
