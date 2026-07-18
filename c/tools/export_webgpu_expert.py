#!/usr/bin/env python3
"""Export selected dense MoE experts to browser-uploadable f32 buffers.

This first exporter intentionally targets readable floating-point source
checkpoints. Colibrì's packed int4 container needs a later dequantizing path;
keeping that conversion explicit avoids silently changing model quality.
"""

import argparse
import json
from pathlib import Path


def parse_indices(values):
    result = []
    for value in values:
        for part in value.split(","):
            if "-" in part:
                start, end = (int(x) for x in part.split("-", 1))
                result.extend(range(start, end + 1))
            elif part.strip():
                result.append(int(part))
    return sorted(set(result))


def export(source, output, layers, experts):
    try:
        import numpy as np
        from safetensors import safe_open
    except ImportError as error:
        raise SystemExit("install exporter dependencies with: uv pip install numpy safetensors") from error

    source = Path(source)
    output = Path(output)
    shards = sorted(source.glob("*.safetensors"))
    if not shards:
        raise SystemExit(f"no safetensors files found in {source}")
    index = {}
    for shard in shards:
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            for name in handle.keys():
                index.setdefault(name, shard)

    manifest = {"schema_version": 1, "dtype": "f32", "experts": {}}
    for layer in layers:
        for expert in experts:
            prefix = f"model.layers.{layer}.mlp.experts.{expert}"
            names = {key: f"{prefix}.{key}_proj.weight" for key in ("gate", "up", "down")}
            missing = [name for name in names.values() if name not in index]
            if missing:
                raise SystemExit(f"missing tensors for {layer}:{expert}: {', '.join(missing)}")
            arrays = {}
            for key, name in names.items():
                with safe_open(str(index[name]), framework="pt", device="cpu") as handle:
                    arrays[key] = np.asarray(handle.get_tensor(name).detach().cpu().numpy(), dtype="<f4")
            gate, up, down = arrays["gate"], arrays["up"], arrays["down"]
            if gate.ndim != 2 or up.shape != gate.shape or down.shape != (gate.shape[1], gate.shape[0]):
                raise SystemExit(f"unexpected shapes for {layer}:{expert}: {gate.shape}, {up.shape}, {down.shape}")
            manifest.setdefault("hidden", int(gate.shape[1]))
            manifest.setdefault("intermediate", int(gate.shape[0]))
            if (manifest["hidden"], manifest["intermediate"]) != (gate.shape[1], gate.shape[0]):
                raise SystemExit("all exported experts must have the same dimensions")
            directory = output / "experts" / f"layer-{layer:04d}" / f"expert-{expert:04d}"
            directory.mkdir(parents=True, exist_ok=True)
            paths = {}
            for key, array in arrays.items():
                path = directory / f"{key}.f32"
                array.tofile(path)
                paths[key] = str(path.relative_to(output))
            manifest["experts"][f"{layer}:{expert}"] = paths
    (output / "webgpu-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"manifest": str(output / "webgpu-manifest.json"),
                      "experts": len(manifest["experts"]),
                      "hidden": manifest["hidden"],
                      "intermediate": manifest["intermediate"]}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="source checkpoint directory")
    parser.add_argument("--output", required=True, help="directory served to the browser")
    parser.add_argument("--layer", action="append", required=True, help="layer id/range, repeatable")
    parser.add_argument("--expert", action="append", required=True, help="expert id/list, repeatable")
    args = parser.parse_args()
    export(args.source, args.output, parse_indices(args.layer), parse_indices(args.expert))


if __name__ == "__main__":
    main()
