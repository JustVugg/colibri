#!/usr/bin/env python3
"""Build a tiny Qwen3-SHAPED model for local validation of qwen3.c.

The real Qwen3-8B is ~16 GB in bf16; this script synthesizes a *tiny* model
with the SAME dense layout (GQA + SwiGLU, q/k RMSNorm, full RoPE) and the SAME
tensor names, but with toy dimensions:

    hidden=64, n_layers=4, q_heads=4, kv_heads=2, head_dim=16,
    inter=128, vocab=320.

Because the layout is identical, convert_qwen3_dense.py + qwen3.c treat it
exactly like the big model, so you can validate token-exactness end-to-end.

It also emits ref.json from greedy generate() on fixed prompt ids. No tokenizer
needed (the ref harness feeds token ids directly).

Usage:
  python tools/make_qwen3_tiny.py --out ./qwen3_tiny
  python tools/make_qwen3_tiny.py --out ./qwen3_tiny   # writes qwen3_tiny/ref.json
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
except ImportError as exc:
    sys.exit(f"Missing deps: {exc}. Run: pip install torch transformers")


def get_classes():
    """Resolve the dense Qwen3 model/config classes via the declared
    config_class (NOT a name guess), same discipline as make_qwen36_tiny.py."""
    import transformers
    mcls = getattr(transformers, "Qwen3ForCausalLM", None)
    if mcls is None:
        sys.exit("Qwen3ForCausalLM not found in this transformers build. Upgrade transformers.")
    return mcls, mcls.config_class


def build(out: Path, hidden=64, n_layers=4, q_heads=4, kv_heads=2,
          head_dim=16, inter=128, vocab=320, max_new=16, prompt_ids=None,
          emit_ref=None, seed=20260916):
    # Fixed seed: an unseeded draw makes the gate flaky (see make_qwen36_tiny.py).
    torch.manual_seed(seed)
    ModelCls, ConfigCls = get_classes()
    base = dict(
        vocab_size=vocab, hidden_size=hidden, intermediate_size=inter,
        num_hidden_layers=n_layers, num_attention_heads=q_heads,
        num_key_value_heads=kv_heads, head_dim=head_dim,
        max_position_embeddings=512, rms_norm_eps=1e-6,
        rope_theta=10000.0, tie_word_embeddings=False,
        attention_bias=False, hidden_act="silu",
    )
    try:
        cfg = ConfigCls(**base)
    except TypeError:
        import inspect
        allowed = set(inspect.signature(ConfigCls.__init__).parameters) - {"self"}
        cfg = ConfigCls(**{k: v for k, v in base.items() if k in allowed})

    for _name, _val in (("pad_token_id", 0), ("bos_token_id", 1), ("eos_token_id", vocab - 1)):
        if not hasattr(cfg, _name):
            try:
                setattr(cfg, _name, _val)
            except Exception:
                pass

    model = ModelCls(cfg)
    model.eval()
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    print(f"Tiny model saved at {out}  (params ~ {sum(p.numel() for p in model.parameters())/1e6:.1f}M)")

    if emit_ref is not None:
        if prompt_ids is None:
            prompt_ids = [1, 2, 3, 4, 5]
        input_ids = torch.tensor([prompt_ids])
        with torch.no_grad():
            out_ids = model.generate(input_ids, max_new_tokens=max_new, do_sample=False,
                                     use_cache=True)
        full = out_ids[0].tolist()
        payload = {"prompt_ids": prompt_ids, "full_ids": full,
                   "mode": "full", "model": "qwen3_tiny"}
        Path(emit_ref).write_text(json.dumps(payload, indent=2))
        print(f"ref.json -> {emit_ref}")
        print(f"  prompt_ids={prompt_ids}")
        print(f"  full_ids ={full}")


def main():
    ap = argparse.ArgumentParser(description="Build a tiny dense-Qwen3-shaped model")
    ap.add_argument("--out", required=True, help="Output model dir")
    ap.add_argument("--emit-ref", default="ref.json",
                    help="Also emit this ref.json, relative to --out. Set '' to skip.")
    ap.add_argument("--seed", type=int, default=20260916,
                    help="RNG seed for the random init; fixed so the gate is reproducible")
    ap.add_argument("--max-new", type=int, default=16)
    ap.add_argument("--prompt-ids", default=None,
                    help="Comma-separated token ids for the prompt (default 1,2,3,4,5)")
    args = ap.parse_args()

    prompt_ids = None
    if args.prompt_ids:
        prompt_ids = [int(x) for x in args.prompt_ids.split(",") if x.strip() != ""]
    emit = args.emit_ref if args.emit_ref else None
    if emit is not None and not Path(emit).is_absolute():
        emit = str(Path(args.out) / emit)

    build(Path(args.out), max_new=args.max_new, prompt_ids=prompt_ids, emit_ref=emit,
          seed=args.seed)


if __name__ == "__main__":
    main()
