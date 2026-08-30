#!/usr/bin/env python3
"""Fixture e oracolo per la torre vision di Qwen3.8, senza scaricare i pesi.

La torre viene costruita da `Qwen4ExpVisionModel` upstream con dimensioni
minuscole e pesi casuali deterministici, e il suo forward viene registrato. Il
motore C legge la stessa fixture e deve produrre gli stessi numeri.

Perche' la torre viene isolata dal modello di testo invece di generare una
fixture multimodale intera: la torre ha una geometria sua (interpolazione delle
posizioni apprese, RoPE 2D, merge di 4 token adiacenti) e sbagliarne un pezzo si
vede come uno scarto piccolo dopo il merger, non come un errore. Se il confronto
avvenisse solo sui token finali del modello, un difetto nella torre arriverebbe
mescolato a tutto il resto e sarebbe molto piu' difficile da localizzare.

La fixture NON richiede il checkpoint: 240k parametri casuali, qualche centinaio
di kB. Serve solo transformers.

USO:
  python3 tools/make_qwen38_vision_tiny.py --out ./qwen38_vision_tiny
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

    # Piccole ma non degeneri: piu' di una testa perche' lo split per testa e' un
    # punto in cui si sbaglia, due blocchi perche' con uno solo un residuo
    # scambiato non si vede, e una griglia 4x4 perche' con 2x2 il merge coprirebbe
    # tutta l'immagine e l'ordine dei blocchi non verrebbe esercitato.
    config = Qwen4ExpVisionConfig(
        depth=2, hidden_size=64, num_heads=4, intermediate_size=128,
        patch_size=16, spatial_merge_size=2, temporal_patch_size=2,
        in_channels=3, out_hidden_size=32, num_position_embeddings=64,
        hidden_act="gelu_pytorch_tanh",
    )
    torch.manual_seed(seed)
    model = Qwen4ExpVisionModel(config).eval()
    # La scala dei pesi NON e' un dettaglio estetico. A 0.05 i prodotti q.k sono
    # cosi' piccoli che il softmax esce praticamente uniforme: l'attenzione
    # diventa la media dei valori e smette di dipendere dai punteggi. Con quella
    # fixture una torre SENZA RoPE passava il confronto -- verificato, non
    # temuto. A 0.6 i punteggi hanno un intervallo vero e la rope si vede.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.6)

    features = config.in_channels * config.temporal_patch_size * config.patch_size ** 2
    pixels = torch.randn(grid_h * grid_w, features, generator=torch.Generator().manual_seed(seed + 1))
    grid = torch.tensor([[1, grid_h, grid_w]])
    with torch.no_grad():
        result = model(pixels, grid_thw=grid)

    out.mkdir(parents=True, exist_ok=True)
    tensors = {f"model.visual.{k}": v.detach().contiguous()
               for k, v in model.state_dict().items()}
    save_file(tensors, str(out / "model.safetensors"))
    (out / "config.json").write_text(json.dumps(
        {"model_type": "qwen4_exp", "vision_config": config.to_dict()}, indent=2))

    reference = {
        "grid": [1, grid_h, grid_w],
        "features": features,
        "pixels": pixels.flatten().tolist(),
        "tokens": int(result.pooler_output.shape[0]),
        "out_hidden": int(result.pooler_output.shape[1]),
        "output": result.pooler_output.flatten().tolist(),
        # Anche l'uscita PRIMA del merger: se solo questa combacia il difetto e'
        # nel merger, se non combacia nessuna delle due e' nei blocchi. Un solo
        # numero finale non distinguerebbe i due casi.
        "last_hidden": result.last_hidden_state.flatten().tolist(),
        "hidden_size": config.hidden_size,
    }
    (out / "ref_vision.json").write_text(json.dumps(reference))
    print(f"  fixture: {len(tensors)} tensori, griglia {grid_h}x{grid_w}, "
          f"{result.pooler_output.shape[0]} token da {result.pooler_output.shape[1]}")
    print(f"  scritta in {out}")
    return reference


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--grid-h", type=int, default=4)
    parser.add_argument("--grid-w", type=int, default=4)
    arguments = parser.parse_args()
    try:
        build(arguments.out, grid_h=arguments.grid_h, grid_w=arguments.grid_w)
    except ImportError as exc:
        raise SystemExit(f"serve transformers e safetensors: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
