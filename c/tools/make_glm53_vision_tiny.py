#!/usr/bin/env python3
"""Oracolo tiny del tower vision di GLM-5.3-Flash.

Il testo ha gia' il suo oracolo; questo copre l'altra meta' del modello, che
nessun test raggiungeva: patch embed, RoPE 2D, i blocchi ViT con SwiGLU
clampata, la norma finale, il downsample 2x2 e il patch merger.

Il fixture e' prodotto dal `Glm5NextVisionModel` ufficiale di Hugging Face, non
da noi: il motore C ha cosi' un bersaglio indipendente invece di validare se
stesso. Pesi e input sono deterministici (linspace su un seed fisso), quindi il
file e' riproducibile bit per bit e la CI non ha bisogno della rete.

Perche' un tower minuscolo e non quello vero: 24 blocchi da 1024 sono 563 M
parametri, inutili per provare che la matematica e' giusta. Qui bastano 2
blocchi e un'immagine da 4x4 patch, e ogni meccanismo viene comunque esercitato
almeno una volta - inclusa la fusione 2x2, che con una griglia 4x4 produce
esattamente 4 token di uscita.

USO:
  python3 tools/make_glm53_vision_tiny.py --output c/glm53_vision_tiny
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

EXPECTED_TRANSFORMERS = "5.16.1"

DEPTH = 2
HIDDEN = 64
HEADS = 4
INTERMEDIATE = 128
PATCH = 4
TEMPORAL = 2
MERGE = 2
OUT_HIDDEN = 96
PROJ_INTERMEDIATE = 128
IN_CHANNELS = 3
GRID_H = 4
GRID_W = 4
SEED = 1253


def deterministic_(tensor, torch, spread=0.05):
    """Riempie in modo riproducibile e indipendente dall'ordine di init."""
    flat = torch.linspace(-spread, spread, tensor.numel(), dtype=torch.float32)
    with torch.no_grad():
        tensor.copy_(flat.view_as(tensor))


def build(output: Path) -> int:
    import torch
    import transformers
    from safetensors.torch import save_file
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextVisionConfig
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextVisionModel

    if transformers.__version__ != EXPECTED_TRANSFORMERS:
        print(f"expected Transformers {EXPECTED_TRANSFORMERS}, "
              f"found {transformers.__version__}")
        return 2

    torch.manual_seed(SEED)
    config = Glm5NextVisionConfig(
        depth=DEPTH,
        hidden_size=HIDDEN,
        num_heads=HEADS,
        intermediate_size=INTERMEDIATE,
        patch_size=PATCH,
        temporal_patch_size=TEMPORAL,
        spatial_merge_size=MERGE,
        out_hidden_size=OUT_HIDDEN,
        projection_intermediate_size=PROJ_INTERMEDIATE,
        in_channels=IN_CHANNELS,
        image_size=PATCH * GRID_H,
    )
    model = Glm5NextVisionModel(config).eval()
    for parameter in model.parameters():
        deterministic_(parameter, torch)
    for name, buffer in model.named_buffers():
        if buffer.dtype.is_floating_point and "inv_freq" not in name:
            deterministic_(buffer, torch)

    patches = GRID_H * GRID_W
    width = IN_CHANNELS * TEMPORAL * PATCH * PATCH
    pixels = torch.linspace(-1.0, 1.0, patches * width, dtype=torch.float32).view(patches, width)
    grid_thw = torch.tensor([[1, GRID_H, GRID_W]], dtype=torch.long)

    with torch.no_grad():
        outputs = model(pixels, grid_thw=grid_thw)
    merged = outputs.pooler_output.float()

    output.mkdir(parents=True, exist_ok=True)
    tensors = {name: parameter.detach().contiguous().to(torch.float32)
               for name, parameter in model.state_dict().items()
               if parameter.dtype.is_floating_point}
    tensors["input.pixel_values"] = pixels.contiguous()
    save_file(tensors, str(output / "model.safetensors"), metadata={"format": "pt"})
    (output / "config.json").write_text(
        json.dumps({"model_type": "glm5_next_vision", "vision_config": config.to_dict()},
                   indent=2, sort_keys=True, default=str),
        encoding="utf-8")

    reference = {
        "schema_version": 1,
        "source": "transformers",
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "seed": SEED,
        "grid_thw": grid_thw.tolist(),
        "input_shape": list(pixels.shape),
        "output_shape": list(merged.shape),
        "output_sum": float(merged.sum()),
        "output_square_sum": float(merged.square().sum()),
        # Ogni token di uscita per intero: un confronto per statistiche sole
        # nasconderebbe un errore che sposta valori fra token senza cambiarne
        # somma e norma, ed e' esattamente cosi' che sbaglia una fusione 2x2.
        "output": merged.flatten().tolist(),
    }
    (output / "ref.json").write_text(json.dumps(reference, indent=2), encoding="utf-8")
    print(f"wrote {output} ({len(tensors)} tensors, "
          f"{merged.shape[0]} merged tokens x {merged.shape[1]})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[1] / "glm53_vision_tiny")
    arguments = parser.parse_args()
    return build(arguments.output)


if __name__ == "__main__":
    raise SystemExit(main())
