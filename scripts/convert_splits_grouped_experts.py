"""Convert AirLLM's per-expert Qwen3Moe split shards to the grouped-tensor
layout transformers 5.x expects (Qwen3MoeExperts: gate_up_proj/down_proj 3D).

Reads  <src>/model.layers.N.safetensors   (experts.<i>.{gate,up,down}_proj.weight)
Writes <dst>/splitted_model/model.layers.N.safetensors
       (mlp.experts.gate_up_proj [E,2I,H], mlp.experts.down_proj [E,H,I])
Non-expert tensors and the embed/norm/lm_head shards are copied unchanged.
"""
from __future__ import annotations

import re
import shutil
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC = Path(
    r"D:\.cache\huggingface\hub\models--Qwen--Qwen3-Coder-30B-A3B-Instruct"
    r"\snapshots\b2cff646eb4bb1d68355c01b18ae02e7cf42d120\splitted_model"
)
DST = Path(r"D:\.cache\huggingface\airllm-qwen3moe-grouped\splitted_model")
EXPERT_RE = re.compile(r"^(model\.layers\.\d+\.mlp\.experts)\.(\d+)\.(gate|up|down)_proj\.weight$")


def convert_layer(src: Path, dst: Path) -> None:
    tensors: dict[str, torch.Tensor] = {}
    experts: dict[str, dict[int, dict[str, torch.Tensor]]] = {}
    with safe_open(src, framework="pt") as f:
        for key in f.keys():
            m = EXPERT_RE.match(key)
            if m:
                prefix, idx, proj = m.group(1), int(m.group(2)), m.group(3)
                experts.setdefault(prefix, {}).setdefault(idx, {})[proj] = f.get_tensor(key)
            else:
                tensors[key] = f.get_tensor(key)
    for prefix, per_expert in experts.items():
        idxs = sorted(per_expert)
        assert idxs == list(range(len(idxs))), f"{src.name}: non-contiguous experts"
        gate_up, down = [], []
        for i in idxs:
            e = per_expert[i]
            gate_up.append(torch.cat([e["gate"], e["up"]], dim=0))
            down.append(e["down"])
        tensors[prefix + ".gate_up_proj"] = torch.stack(gate_up)
        tensors[prefix + ".down_proj"] = torch.stack(down)
    save_file(tensors, str(dst), metadata={"format": "pt"})
    dst.with_suffix(dst.suffix + ".done").touch()


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    layers = sorted(SRC.glob("model.layers.*.safetensors"),
                    key=lambda p: int(p.name.split(".")[2]))
    for n, src in enumerate(layers):
        dst = DST / src.name
        if dst.with_suffix(dst.suffix + ".done").exists():
            print(f"skip {src.name}", flush=True)
            continue
        convert_layer(src, dst)
        print(f"[{n + 1}/{len(layers)}] {src.name} ({time.time() - t0:.0f}s)", flush=True)
    for name in ("model.embed_tokens.safetensors", "model.norm.safetensors", "lm_head.safetensors"):
        dst = DST / name
        if not dst.exists():
            shutil.copyfile(SRC / name, dst)
        dst.with_suffix(dst.suffix + ".done").touch()
    print(f"DONE in {time.time() - t0:.0f}s -> {DST}", flush=True)


if __name__ == "__main__":
    sys.exit(main())