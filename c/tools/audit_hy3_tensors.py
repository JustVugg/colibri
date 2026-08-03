#!/usr/bin/env python3
"""Audit tencent/Hy3 (or converted colibri) safetensors: dense vs expert bytes, key naming."""
import argparse, json, re, statistics, sys
from pathlib import Path

EXPERT_RE = re.compile(r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.")
DENSE_HINTS = (
    "embed", "lm_head", "norm", "self_attn", "shared_mlp", "shared_experts",
    "mlp.gate", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "expert_bias",
    "e_score_correction", "eh_proj",
)


def tensor_sizes(path):
    with path.open("rb") as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError(f"short header: {path}")
        length = int.from_bytes(raw, "little")
        header = json.loads(stream.read(length))
    file_size = path.stat().st_size
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        start, end = meta["data_offsets"]
        yield name, end - start, meta.get("dtype", "?")


def audit(model_dir):
    model_dir = Path(model_dir)
    cfg = json.loads((model_dir / "config.json").read_text())
    index_path = model_dir / "model.safetensors.index.json"
    shards = sorted(model_dir.glob("*.safetensors"))
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        wm = index.get("weight_map", index)
        names = sorted(wm.keys()) if isinstance(wm, dict) else sorted(index.keys())
        print(f"index: {index_path.name} ({len(names)} tensors)")
    else:
        names = sorted({n for shard in shards for n, _, _ in tensor_sizes(shard)})
        print(f"shards: {len(shards)} (no index.json — scanned headers)")

    experts = {}
    for name in names:
        m = EXPERT_RE.search(name)
        if m:
            experts[tuple(map(int, m.groups()))] = experts.get(tuple(map(int, m.groups())), 0) + 1

    print(f"\nmodel_type: {cfg.get('model_type')}  layers: {cfg.get('num_hidden_layers')}  "
          f"experts: {cfg.get('num_experts', cfg.get('n_routed_experts'))}")
    print(f"unique expert tensor groups: {len(experts)}")
    layers = sorted({k[0] for k in experts})
    print(f"expert layers: {layers[:5]}...{layers[-3:] if len(layers) > 8 else layers}")

    router = [n for n in names if "gate.weight" in n and "gate_proj" not in n]
    bias = [n for n in names if "bias" in n.lower() and "mlp" in n]
    shared = [n for n in names if "shared_mlp" in n or "shared_experts" in n]
    mtp = [n for n in names if re.search(r"layers\.80\.", n)]
    attn = [n for n in names if "q_norm" in n or "k_norm" in n]

    print(f"\nrouter weights ({len(router)}):")
    for n in router[:5]:
        print(f"  {n}")
    if len(router) > 5:
        print(f"  ... +{len(router)-5} more")

    print(f"\nrouter bias ({len(bias)}):")
    for n in bias[:8]:
        print(f"  {n}")

    print(f"\nshared expert ({len(shared)} tensors):")
    for n in sorted(set(".".join(x.split(".")[:4]) for x in shared))[:4]:
        print(f"  {n}.*")

    print(f"\nMTP layer 80 ({len(mtp)}):")
    for n in sorted(mtp):
        print(f"  {n}")

    print(f"\nq_norm/k_norm ({len(attn)}):")
    for n in attn[:4]:
        print(f"  {n}")

    if shards:
        dense_bytes = expert_bytes = 0
        expert_groups = {}
        for shard in shards:
            for name, size, _ in tensor_sizes(shard):
                m = EXPERT_RE.search(name)
                if m:
                    expert_bytes += size
                    expert_groups.setdefault(tuple(map(int, m.groups())), []).append(size)
                else:
                    dense_bytes += size
        per_layer = {layer: sum(v) for (layer, _), v in expert_groups.items()}
        layer_medians = [statistics.median(v) for v in
                         {layer: [sum(expert_groups[k]) for k in expert_groups if k[0] == layer]
                          for layer in {k[0] for k in expert_groups}}.values() if v]
        med = int(statistics.median(layer_medians)) if layer_medians else 0
        topk = cfg.get("num_experts_per_tok", 8)
        print(f"\nbyte estimate from local shards:")
        print(f"  dense: {dense_bytes/1e9:.2f} GB")
        print(f"  experts: {expert_bytes/1e9:.2f} GB")
        print(f"  median expert (3 tensors): {med/1e6:.2f} MB")
        print(f"  cold decode ~ {len(per_layer)*topk*med/1e9:.1f} GB/token")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default=".")
    args = ap.parse_args()
    try:
        audit(args.model)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
