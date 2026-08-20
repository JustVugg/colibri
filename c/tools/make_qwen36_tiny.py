#!/usr/bin/env python3
"""Build a tiny Qwen3.6-SHAPED model for local validation of qwen36.c (Phase 0/1).

The real Qwen3.6-35B-A3B is ~70 GB in bf16 and cannot be loaded on a 24 GB
laptop. This script synthesizes a *tiny* model with the SAME hybrid layout
(10 x (3 x Gated DeltaNet -> MoE, 1 x Gated Attention -> MoE)) and the SAME
tensor names, but with toy dimensions:
    hidden=64, n_layers=8 (-> 2 attention layers at idx 3,7),
    q_heads=4, kv_heads=2, head_dim=16, rope_dim=8,
    n_experts=8, topk=2, n_group=1, topk_group=1, inter=32, vocab=320.

Because the layout is identical, convert_qwen36.py + qwen36.c treat it exactly
like the big model, so you can validate token-exactness end-to-end on a laptop.

It also emits ref.json in attention_only mode (DeltaNet layers -> identity),
matching what qwen36.c Phase 1 computes. No tokenizer needed.

Usage:
  python tools/make_qwen36_tiny.py --out ./qwen36_tiny
  python tools/make_qwen36_tiny.py --out ./qwen36_tiny --emit-ref ref_qwen36.json
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
except ImportError as exc:
    sys.exit(f"Missing deps: {exc}. Run: pip install torch transformers")


def get_classes():
    """Resolve the Qwen3-MoE model/config classes across transformers versions.

    Uses each model class's declared `config_class` (NOT a name guess): in
    transformers >=5 the text model's config is `Qwen3_5MoeTextConfig`, while the
    same-named `Qwen3_5MoeConfig` is the vision-language wrapper and lacks the
    text fields (vocab_size, head_dim, ...). Guessing by name picks the wrong one.
    """
    candidates = [
        "Qwen3_5MoeForCausalLM",
        "Qwen3MoeForCausalLM",
        "Qwen3NextMoeForCausalLM",
    ]
    import transformers
    for mcls in candidates:
        mc = getattr(transformers, mcls, None)
        if mc is not None:
            cc = getattr(mc, "config_class", None)
            if cc is not None:
                return mc, cc
    sys.exit("No Qwen3-MoE model class found in this transformers build. Upgrade transformers.")


class Zero(nn.Module):
    def forward(self, hidden_states, *args, **kwargs):
        if torch.is_tensor(hidden_states):
            return torch.zeros_like(hidden_states)
        if isinstance(hidden_states, (tuple, list)):
            return tuple(torch.zeros_like(t) if torch.is_tensor(t) else t for t in hidden_states)
        return hidden_states


def build(out: Path, hidden=64, n_layers=8, q_heads=4, kv_heads=2,
          head_dim=16, rope_dim=8, n_experts=8, topk=2, inter=32,
          vocab=320, max_new=16, prompt_ids=None, emit_ref=None,
          ref_mode="attention_only", seed=20260817):
    # Fixed by default: an unseeded draw makes the gate flaky. Measured --
    # three local draws passed, one CI draw failed at 11/16, with identical
    # code. A gate that reddens at random gets muted within a week.
    torch.manual_seed(seed)
    ModelCls, ConfigCls = get_classes()
    layer_types = ["full_attention" if i % 4 == 3 else "linear_attention"
                   for i in range(n_layers)]
    base = dict(
        vocab_size=vocab, hidden_size=hidden, intermediate_size=hidden * 2,
        num_hidden_layers=n_layers, num_attention_heads=q_heads,
        num_key_value_heads=kv_heads, num_experts=n_experts,
        num_experts_per_tok=topk, moe_intermediate_size=inter,
        shared_expert_intermediate_size=inter, max_position_embeddings=512,
        rms_norm_eps=1e-6, rope_theta=10000.0, tie_word_embeddings=False,
        head_dim=head_dim, linear_conv_kernel_dim=4,
        linear_key_head_dim=8, linear_value_head_dim=8,
        linear_num_key_heads=q_heads, linear_num_value_heads=q_heads * 2,
        layer_types=layer_types, hidden_act="silu",
        attention_bias=False, attention_dropout=0.0, use_cache=True,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
    )
    # Qwen3_5MoeConfig uses **kwargs, so pass everything; fall back to filtered
    # only if a build rejects an unknown key.
    try:
        cfg = ConfigCls(**base)
    except TypeError:
        import inspect
        allowed = set(inspect.signature(ConfigCls.__init__).parameters) - {"self"}
        cfg = ConfigCls(**{k: v for k, v in base.items() if k in allowed})

    # Some transformers builds require pad/bos/eos token ids on Qwen3-MoE configs;
    # if missing, save_pretrained()/generate() crash on attribute access.
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
        # attention_only replaces the DeltaNet layers (i%4!=3) with identity,
        # which is what the Phase-1 engine computed. The Phase-2 engine runs
        # BOTH layer kinds, so a gate built on that reference would pass while
        # never touching 30 of 40 layers -- hence ref_mode="full", which leaves
        # the model whole.
        #
        # This lives here rather than in make_qwen36_oracle.py because that
        # script encodes a text prompt through AutoTokenizer.from_pretrained(),
        # and this fixture is synthetic: it has weights and no tokenizer. The
        # oracle script stays the tool for real checkpoints.
        if ref_mode == "attention_only":
            replaced = 0
            for i in range(n_layers):
                if i % 4 != 3:
                    model.model.layers[i].self_attn = Zero()
                    model.model.layers[i].mlp = Zero()
                    replaced += 1
            print(f"attention_only: replaced {replaced} DeltaNet layers with identity")
        else:
            print("full: both layer kinds active (Gated Attention + Gated DeltaNet)")
        if prompt_ids is None:
            prompt_ids = [1, 2, 3, 4, 5]
        input_ids = torch.tensor([prompt_ids])
        with torch.no_grad():
            out_ids = model.generate(input_ids, max_new_tokens=max_new, do_sample=False,
                                     use_cache=True)
        full = out_ids[0].tolist()
        payload = {"prompt_ids": prompt_ids, "full_ids": full,
                   "mode": ref_mode, "model": "qwen36_tiny"}
        Path(emit_ref).write_text(json.dumps(payload, indent=2))
        print(f"ref.json -> {emit_ref}")
        print(f"  prompt_ids={prompt_ids}")
        print(f"  full_ids ={full}")


def main():
    ap = argparse.ArgumentParser(description="Build a tiny Qwen3.6-shaped model")
    ap.add_argument("--out", required=True, help="Output model dir")
    ap.add_argument("--emit-ref", default="ref_qwen36.json",
                    help="Also emit this ref.json (attention_only). Set '' to skip.")
    ap.add_argument("--seed", type=int, default=20260817,
                    help="RNG seed for the random init; fixed so the gate is "
                         "reproducible")
    ap.add_argument("--ref-mode", choices=["attention_only", "full"],
                    default="attention_only",
                    help="full keeps both layer kinds (Phase-2 engine); "
                         "attention_only replaces DeltaNet with identity (Phase 1)")
    ap.add_argument("--max-new", type=int, default=16)
    ap.add_argument("--prompt-ids", default=None,
                    help="Comma-separated token ids for the prompt (default 1,2,3,4,5)")
    args = ap.parse_args()

    prompt_ids = None
    if args.prompt_ids:
        prompt_ids = [int(x) for x in args.prompt_ids.split(",") if x.strip() != ""]
    emit = args.emit_ref if args.emit_ref else None

    build(Path(args.out), max_new=args.max_new, prompt_ids=prompt_ids, emit_ref=emit,
          ref_mode=args.ref_mode, seed=args.seed)


if __name__ == "__main__":
    main()
