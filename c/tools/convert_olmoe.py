#!/usr/bin/env python3
"""Convert OLMoE HuggingFace checkpoint to colibri int4 format.

Downloads or converts a local OLMoE checkpoint (e.g., allenai/OLMoE-1B-7B-0125-Instruct).
Dense weights stay as-is (engine reads BF16/F16 → F32 on load).
Expert weights get row-wise int8 quantization with float32 scales.

Usage:
  python tools/convert_olmoe.py --repo allenai/OLMoE-1B-7B-0125-Instruct --out ./olmoe_i4
  python tools/convert_olmoe.py --model ./OLMoE-1B-7B-0125-Instruct --out ./olmoe_i4
"""

import argparse, json, math, os, struct, sys
from pathlib import Path

# Windows: force UTF-8 output
if sys.platform == "win32":
    for s in (sys.stdout, sys.stderr):
        try: s.reconfigure(encoding="utf-8")
        except (AttributeError, OSError): pass

try:
    import torch
    from safetensors.torch import load_file, save_file
except ImportError as exc:
    sys.exit(f"Missing dependencies: {exc}. Install: pip install torch safetensors")


EXPERT_KEY_RE = r"model\.layers\.\d+\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)\.weight"


def quantize_row(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Row-wise int8 quantization. Returns (int8_weights, float32_scales)."""
    w_f32 = w.float()
    row_max = w_f32.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    scales = row_max / 127.0
    q = (w_f32 / scales).round().clamp(-128, 127).to(torch.int8)
    return q, scales.squeeze(1)


def is_expert_weight(name: str) -> bool:
    import re
    return bool(re.search(EXPERT_KEY_RE, name))


def main():
    ap = argparse.ArgumentParser(description="Convert OLMoE HF checkpoint -> colibri int4")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--repo", help="HuggingFace repo ID")
    src.add_argument("--model", help="Local HF checkpoint directory")
    ap.add_argument("--out", required=True, help="Output directory for int4 model")
    ap.add_argument("--ebits", type=int, default=4,
                    help="Expert quant bits (4 or 8, default 4)")
    args = ap.parse_args()

    if args.repo:
        from huggingface_hub import snapshot_download
        print(f"Downloading {args.repo}...")
        src_dir = snapshot_download(args.repo, local_files_only=True, max_workers=4)
        if not any(Path(src_dir).glob("*.safetensors")):
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

    expert_count = 0
    total_expert_f32 = total_expert_q = 0

    for si, shard in enumerate(shards, 1):
        print(f"[{si}/{len(shards)}] {shard.name}...", end=" ", flush=True)
        tensors = load_file(str(shard))
        out_tensors = {}
        for name, tensor in tensors.items():
            if is_expert_weight(name):
                expert_count += 1
                q, scales = quantize_row(tensor)
                total_expert_f32 += tensor.numel() * tensor.element_size()
                total_expert_q += q.numel() * 1 + scales.numel() * 4
                out_tensors[name] = q
                out_tensors[name + ".qs"] = scales
            else:
                out_tensors[name] = tensor

        out_shard = out / shard.name
        save_file(out_tensors, str(out_shard))
        ratio = total_expert_q / max(total_expert_f32, 1) * 100
        print(f"ok")

    print(f"\nDone. {expert_count} expert tensors quantized.")
    print(f"Expert storage: {total_expert_f32/1e9:.1f} GB -> {total_expert_q/1e9:.1f} GB ({ratio:.0f}%)")
    print(f"Model ready at: {out}")
    print(f"\nRun: SNAP={out} ./olmoe.exe 32 4 16")


if __name__ == "__main__":
    main()
