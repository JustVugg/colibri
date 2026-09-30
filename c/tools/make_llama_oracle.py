#!/usr/bin/env python3
"""Greedy reference token ids for a Llama checkpoint, from HuggingFace transformers.

The oracle for llama.c. It never touches the engine under test: the ids come
from LlamaForCausalLM running the checkpoint's own weights (BF16 on disk,
upcast to float32 for the forward pass, which is what llama.c computes in).

Writes {prompt_ids, full_ids, n_new, ...} -- the same prompt_ids/full_ids pair
the olmoe harness reads -- plus the provenance needed to reproduce it.

Usage:
  python tools/make_llama_oracle.py --model DIR --out ref_llama.json \\
      [--prompt TEXT] [--n-new 64] [--repo ID --revision SHA]
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

try:
    import torch
    from transformers import AutoTokenizer, LlamaForCausalLM
except ImportError as exc:
    sys.exit(f"Missing deps: {exc}. Run: pip install torch transformers")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--prompt-ids", default="",
                    help="comma-separated ids; skips the tokenizer (checkpoints without one)")
    ap.add_argument("--n-new", type=int, default=64)
    ap.add_argument("--repo", default="")
    ap.add_argument("--revision", default="")
    args = ap.parse_args()
    model_dir = Path(args.model)

    torch.manual_seed(0)
    tok = None
    if args.prompt_ids:
        prompt_ids = [int(t) for t in args.prompt_ids.split(",")]
    else:
        tok = AutoTokenizer.from_pretrained(model_dir)
        prompt_ids = tok(args.prompt)["input_ids"]
    model = LlamaForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        out = model.generate(torch.tensor([prompt_ids]), max_new_tokens=args.n_new,
                             min_new_tokens=args.n_new, do_sample=False, use_cache=True)
    full_ids = out[0].tolist()
    gen = full_ids[len(prompt_ids):]
    if len(gen) != args.n_new:
        sys.exit(f"generated {len(gen)} tokens, wanted {args.n_new}")
    import transformers
    payload = {
        "repo": args.repo,
        "revision": args.revision,
        "shards": {p.name: sha256(p) for p in sorted(model_dir.glob("*.safetensors"))},
        "prompt": None if tok is None else args.prompt,
        "prompt_ids": prompt_ids,
        "n_new": args.n_new,
        "generated_ids": gen,
        "full_ids": full_ids,
        "compute_dtype": "float32",
        "decoding": "greedy",
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "text": None if tok is None else tok.decode(gen),
    }
    Path(args.out).write_text(json.dumps(payload, indent=1) + "\n")
    print(f"prompt_ids {prompt_ids}\ngenerated  {gen}\ntext {payload['text']!r}")
    print(f"sha256 {sha256(args.out)}  {args.out}")


if __name__ == "__main__":
    main()
