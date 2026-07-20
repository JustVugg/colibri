#!/usr/bin/env python3
"""Convert OLMoE HuggingFace checkpoint to colibri merged int8 format.

Consolidates gate_proj, up_proj, and down_proj into a single merged tensor per expert.
This allows olmoe.c to load an expert in a single disk read call instead of 3.

Usage:
  python tools/convert_olmoe_merged.py --repo allenai/OLMoE-1B-7B-0125-Instruct --out ./olmoe_merged
"""

import argparse, json, os, sys, re
from pathlib import Path

# Windows: force UTF-8 output
if sys.platform == "win32":
    for s in (sys.stdout, sys.stderr):
        try: s.reconfigure(encoding="utf-8")
        except (AttributeError, OSError): pass

try:
    import torch
    from safetensors.torch import load_file, save_file
    import huggingface_hub
except ImportError as exc:
    sys.exit(f"Missing dependencies: {exc}. Install: pip install torch safetensors huggingface_hub")

EXPERT_KEY_RE = r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight"

def quantize_row(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Row-wise int8 quantization. Returns (int8_weights, float32_scales)."""
    w_f32 = w.float()
    row_max = w_f32.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    scales = row_max / 127.0
    q = (w_f32 / scales).round().clamp(-128, 127).to(torch.int8)
    return q, scales.squeeze(1)

def main():
    ap = argparse.ArgumentParser(description="Convert OLMoE HF checkpoint -> colibri merged int8")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--repo", help="HuggingFace repo ID")
    src.add_argument("--model", help="Local HF checkpoint directory")
    ap.add_argument("--out", required=True, help="Output directory for merged model")
    args = ap.parse_args()

    if args.repo:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError
        print(f"Downloading/Resolving {args.repo}...")
        try:
            src_dir = snapshot_download(args.repo, local_files_only=True, max_workers=4)
        except LocalEntryNotFoundError:
            src_dir = None
        if src_dir is None or not any(Path(src_dir).glob("*.safetensors")):
            print("Downloading safetensors...")
            src_dir = snapshot_download(args.repo, max_workers=4)
    else:
        src_dir = args.model

    src = Path(src_dir)
    if not src.is_dir():
        sys.exit(f"Model directory not found: {src}")
    if not (src / "config.json").is_file():
        sys.exit(f"config.json missing in {src}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Copy config.json
    import shutil
    shutil.copy2(src / "config.json", out / "config.json")
    print(f"config.json -> {out}")

    # Process safetensors
    shards = sorted(src.glob("*.safetensors"))
    if not shards:
        sys.exit(f"No safetensors found in {src}")

    print("Loading all shards to build complete state dict...")
    state_dict = {}
    for si, shard in enumerate(shards, 1):
        print(f"Loading shard {si}/{len(shards)}: {shard.name}...")
        tensors = load_file(str(shard))
        state_dict.update(tensors)

    # Gather experts
    experts = {}
    for name in list(state_dict.keys()):
        m = re.match(EXPERT_KEY_RE, name)
        if m:
            layer_idx, expert_idx, proj = m.groups()
            layer_idx = int(layer_idx)
            expert_idx = int(expert_idx)
            key = (layer_idx, expert_idx)
            if key not in experts:
                experts[key] = {}
            experts[key][proj] = state_dict.pop(name)

    print(f"Found {len(experts)} experts to merge.")

    # Process and merge experts
    out_tensors = {}
    total_expert_f32 = 0
    total_expert_q = 0

    for (layer, expert), projs in sorted(experts.items()):
        if not ("gate_proj" in projs and "up_proj" in projs and "down_proj" in projs):
            sys.exit(f"Missing projection for layer {layer} expert {expert}!")

        gate = projs["gate_proj"]
        up = projs["up_proj"]
        down = projs["down_proj"]

        total_expert_f32 += (gate.numel() + up.numel() + down.numel()) * gate.element_size()

        # Quantize each projection separately
        q_gate, s_gate = quantize_row(gate)
        q_up, s_up = quantize_row(up)
        q_down, s_down = quantize_row(down)

        # Merge weights and scales contiguously
        merged_q = torch.cat([q_gate.flatten(), q_up.flatten(), q_down.flatten()])
        merged_scales = torch.cat([s_gate, s_up, s_down])

        total_expert_q += merged_q.numel() * 1 + merged_scales.numel() * 4

        # Save to output
        out_tensors[f"model.layers.{layer}.mlp.experts.{expert}.merged_weight"] = merged_q
        out_tensors[f"model.layers.{layer}.mlp.experts.{expert}.qs"] = merged_scales

    # Copy remaining dense tensors
    print(f"Adding remaining {len(state_dict)} dense tensors...")
    out_tensors.update(state_dict)

    # Save to a single output safetensors file for simpler loading
    out_file = out / "model.safetensors"
    print(f"Saving merged safetensors model to {out_file}...")
    save_file(out_tensors, str(out_file))

    ratio = total_expert_q / max(total_expert_f32, 1) * 100
    print(f"\nDone. {len(experts)} experts successfully merged and saved.")
    print(f"Expert storage: {total_expert_f32/1e9:.1f} GB -> {total_expert_q/1e9:.1f} GB ({ratio:.0f}%)")
    print(f"Model ready at: {out}")

if __name__ == "__main__":
    main()
