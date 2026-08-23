#!/usr/bin/env python3
"""Generate reference token IDs for Qwen3.6 (or a Qwen3-MoE-shaped model).

Two modes:
  --mode attention_only  (default, for Phase 1)
      Replace each DeltaNet layer's self_attn + mlp with a zero-returning
      module, so the layer is a clean identity (residual + 0). The C engine
      (qwen36.c) runs Gated Attention + MoE ONLY on the attention layers
      (i%4==3) and treats the others as identity, so its output must match.
  --mode full            (for the eventual full hybrid engine)
      Run the whole model unmodified.

For local validation on a 24 GB laptop, pair this with tools/make_qwen36_tiny.py
(a tiny Qwen3.6-shaped model that fits in RAM) -- the 35B bf16 checkpoint needs
~70 GB and cannot be loaded here.

Usage:
  python tools/make_qwen36_oracle.py --model ./qwen36_tiny --out ref_qwen36.json
  python tools/make_qwen36_oracle.py --repo Qwen/Qwen3.6-35B-A3B --out ref_qwen36.json
  python tools/make_qwen36_oracle.py --model ./qwen36_tiny --out ref_qwen36.json --prompt "The capital of France is"
"""
import argparse, json, sys
from pathlib import Path

if sys.platform == "win32":
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass

try:
    import torch
    import torch.nn as nn
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError as exc:
    sys.exit(f"Missing deps: {exc}. Run: pip install torch transformers")


class Zero(nn.Module):
    """Returns zeros_like(input) regardless of extra args.

    Replacing a decoder layer's self_attn/mlp with this makes the layer a clean
    identity: h = residual + self_attn(norm(h)) = residual + 0, and likewise for
    the MLP branch. The residual/layernorm wiring is preserved, so the layer's
    return tuple stays well-formed for the surrounding model.
    """

    def forward(self, hidden_states, *args, **kwargs):
        if torch.is_tensor(hidden_states):
            return torch.zeros_like(hidden_states)
        # some layers forward a tuple; zero the first tensor element
        if isinstance(hidden_states, (tuple, list)):
            return tuple(
                torch.zeros_like(t) if torch.is_tensor(t) else t for t in hidden_states
            )
        return hidden_states


def detect_layer_type(model, i: int) -> str:
    """A layer is 'attention' iff its index i%4==3 (Qwen3.6 layout)."""
    return "attn" if i % 4 == 3 else "delta"


def main():
    ap = argparse.ArgumentParser(description="Qwen3.6 reference token generator")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--repo", help="HuggingFace repo ID")
    src.add_argument("--model", help="Local HF checkpoint directory")
    ap.add_argument("--out", required=True, help="Output ref JSON path")
    ap.add_argument("--mode", choices=["attention_only", "full"], default="attention_only")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    args = ap.parse_args()

    if args.repo:
        from huggingface_hub import snapshot_download
        print(f"Resolving {args.repo} ...")
        try:
            mdir = snapshot_download(args.repo, local_files_only=True)
        except Exception:
            mdir = snapshot_download(args.repo)
    else:
        mdir = args.model

    print(f"Loading tokenizer from {mdir} ...")
    tokenizer = AutoTokenizer.from_pretrained(mdir)

    print("Encoding prompt ...")
    enc = tokenizer(args.prompt, return_tensors="pt")
    prompt_ids = enc["input_ids"][0].tolist()
    print(f"  Prompt IDs ({len(prompt_ids)}): {prompt_ids}")

    print(f"Loading model from {mdir} ...")
    print("  (large models need a lot of RAM -- be patient)")
    model = AutoModelForCausalLM.from_pretrained(
        mdir, torch_dtype=torch.bfloat16, device_map="cpu", low_cpu_mem_usage=True
    )
    model.eval()
    print("  Model loaded!")

    if args.mode == "attention_only":
        n = len(model.model.layers)
        replaced = 0
        for i in range(n):
            if detect_layer_type(model, i) == "delta":
                layer = model.model.layers[i]
                layer.self_attn = Zero()
                layer.mlp = Zero()
                replaced += 1
        print(f"  attention_only: replaced {replaced} DeltaNet layers with identity")

    print(f"Generating {args.max_new_tokens} tokens ...")
    with torch.no_grad():
        out_ids = model.generate(
            enc["input_ids"],
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    full_ids = out_ids[0].tolist()
    gen_ids = full_ids[len(prompt_ids):]

    print(f"Prompt IDs  : {prompt_ids}")
    print(f"Full IDs    : {full_ids}")
    print(f"Generated   : {gen_ids}")
    try:
        print(f"Text        : {tokenizer.decode(gen_ids, skip_special_tokens=True)!r}")
    except Exception:
        pass

    payload = {"prompt_ids": prompt_ids, "full_ids": full_ids,
               "mode": args.mode, "model": mdir}
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"\nSaved reference to {args.out}")


if __name__ == "__main__":
    main()
