#!/usr/bin/env python3
"""Da un fixture GLM-5.3 f32, la coppia che prova lo streaming degli esperti.

Il motore ha due strade per gli esperti. Con un checkpoint f32 li carica tutti
in RAM; col contenitore int4 li lascia su disco e ne legge uno slot per volta,
con tabella degli indirizzi, cache LRU per layer e matrici che puntano dentro
allo slot. La seconda strada e' quella che serve sul modello vero, ed e' anche
quella dove si sbaglia un offset, una forma o una vittima della cache senza che
il programma se ne accorga: legge byte diversi e continua a rispondere.

Qui si scrivono due fixture con gli stessi identici numeri:

  <out>-i4    gli esperti nel contenitore int4 gs64 (`nome` U8 + `nome.qs` F32)
  <out>-deq   gli stessi esperti in f32, dequantizzati da quel contenitore

Il resto dei pesi non viene toccato. Il motore deve rispondere allo stesso modo
sui due: la differenza fra le due esecuzioni non e' la quantizzazione, che e'
gia' avvenuta in entrambe, ma solo da dove arrivano i byte.

USO:
  python3 tools/make_glm53_streaming_pair.py --fixture ~/glm53_mm_tiny \\
                                             --output ~/glm53_stream
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

EXPERT_MARK = ".mlp.experts."
GROUP = 64


def quantize(weight, numpy):
    """int4 con una scala ogni 64 colonne, la stessa di convert_glm53.py."""
    rows, columns = weight.shape
    groups = (columns + GROUP - 1) // GROUP
    padded = numpy.zeros((rows, groups * GROUP), numpy.float32)
    padded[:, :columns] = weight
    blocks = padded.reshape(rows, groups, GROUP)
    amax = numpy.abs(blocks).max(axis=2, keepdims=True)
    step = numpy.maximum(amax / 7.0, 1e-8)
    level = numpy.clip(numpy.rint(blocks / step), -8, 7).astype(numpy.int32)
    level = level.reshape(rows, groups * GROUP)[:, :columns]

    packed = numpy.zeros((rows, (columns + 1) // 2), numpy.uint8)
    low = (level[:, 0::2] + 8).astype(numpy.uint8)
    packed[:, :low.shape[1]] = low
    if columns > 1:
        high = (level[:, 1::2] + 8).astype(numpy.uint8)
        packed[:, :high.shape[1]] |= (high << 4)
    dequantized = (level * step.reshape(rows, groups).repeat(GROUP, axis=1)[:, :columns])
    return packed.reshape(-1), step[:, :, 0].astype(numpy.float32).reshape(-1), \
        dequantized.astype(numpy.float32)


def build(fixture: Path, output: Path) -> int:
    import numpy
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    quantized_dir = Path(str(output) + "-i4")
    dequantized_dir = Path(str(output) + "-deq")
    for directory in (quantized_dir, dequantized_dir):
        directory.mkdir(parents=True, exist_ok=True)

    quantized, dequantized = {}, {}
    experts = 0
    with safe_open(str(fixture / "model.safetensors"), "pt") as handle:
        for name in handle.keys():
            tensor = handle.get_tensor(name)
            if EXPERT_MARK not in name or not name.endswith(".weight") or tensor.dim() != 2:
                quantized[name] = tensor
                dequantized[name] = tensor
                continue
            packed, step, plain = quantize(tensor.float().numpy(), numpy)
            quantized[name] = torch.from_numpy(packed)
            quantized[name + ".qs"] = torch.from_numpy(step)
            dequantized[name] = torch.from_numpy(plain)
            experts += 1

    if not experts:
        print(f"{fixture}: nessun esperto da convertire")
        return 2

    save_file(quantized, str(quantized_dir / "model.safetensors"), metadata={"format": "pt"})
    save_file(dequantized, str(dequantized_dir / "model.safetensors"), metadata={"format": "pt"})
    for directory in (quantized_dir, dequantized_dir):
        for extra in ("config.json", "ref.json", "patches.f32"):
            source = fixture / extra
            if source.exists():
                shutil.copy(source, directory / extra)

    print(f"scritti {quantized_dir.name} e {dequantized_dir.name} "
          f"({experts} matrici di esperti)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    return build(arguments.fixture, arguments.output)


if __name__ == "__main__":
    raise SystemExit(main())
