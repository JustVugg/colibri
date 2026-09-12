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

Geometry presets (--geometry): the structural numbers of a real checkpoint at
toy widths, so the config-shape guards, the layer pattern, the routing and the
converter's tensor contract are exercised without a single real weight:
  qwen36-35b   8 layers (2 attention), 8 experts top-2, GQA 2:1  (default)
  qwen38-2p4t  92 layers (23 attention), 512 experts top-10, GQA 16:1,
               DeltaNet value:key heads 8:1, FUSED expert tensors and an
               mtp.* head, exactly as Qwen/Qwen3.8-2.4T-A95B ships them
--fused-experts rewrites the saved shard into the per-layer gate_up_proj /
down_proj layout the real checkpoints use (transformers saves per-expert);
--mtp adds a one-layer multi-token-prediction head the engine must skip.
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


GEOMETRIES = {
    # the shipped fixture: what CI has gated since #712
    "qwen36-35b": dict(hidden=64, n_layers=8, q_heads=4, kv_heads=2, head_dim=16,
                       rope_dim=8, n_experts=8, topk=2, inter=32,
                       dn_key_heads=4, dn_value_heads=8,
                       fused_experts=False, mtp=False),
    # Qwen/Qwen3.8-2.4T-A95B, config.json: 92 layers / full_attention_interval 4
    # / 512 experts top-10 / 64:4 attention heads / 128:16 DeltaNet heads /
    # mtp_num_hidden_layers 1 / fused experts. Widths shrunk, ratios kept.
    "qwen38-2p4t": dict(hidden=32, n_layers=92, q_heads=16, kv_heads=1, head_dim=16,
                        rope_dim=8, n_experts=512, topk=10, inter=16,
                        dn_key_heads=2, dn_value_heads=16,
                        fused_experts=True, mtp=True),
}


def _rewrite_shard(out: Path, n_layers, fused_experts, mtp, hidden, seed):
    """Post-process the transformers shard into the real checkpoints' layout.

    transformers saves routed experts one tensor per expert and never writes an
    mtp head. The real 35B and 2.4T checkpoints do the opposite on both counts,
    and those are the two paths the converter's tensor contract has to be seen
    handling: splitting the fused tensor, and skipping mtp.* on purpose.
    """
    from safetensors.torch import load_file, save_file
    path = out / "model.safetensors"
    tens = load_file(str(path))
    if fused_experts:
        for i in range(n_layers):
            pre = f"model.layers.{i}.mlp.experts."
            gate, up, down = [], [], []
            e = 0
            while f"{pre}{e}.gate_proj.weight" in tens:
                gate.append(tens.pop(f"{pre}{e}.gate_proj.weight"))
                up.append(tens.pop(f"{pre}{e}.up_proj.weight"))
                down.append(tens.pop(f"{pre}{e}.down_proj.weight"))
                e += 1
            if e == 0:
                sys.exit(f"layer {i}: no per-expert tensors to fuse")
            # [E, 2*inter, H] with gate rows first, then up -- the split the
            # converter performs is gu[:, :inter] / gu[:, inter:]
            tens[f"{pre}gate_up_proj"] = torch.stack(
                [torch.cat([g, u], dim=0) for g, u in zip(gate, up)]).contiguous()
            tens[f"{pre}down_proj"] = torch.stack(down).contiguous()   # [E, H, inter]
    if mtp:
        # One MTP layer shaped like the last transformer layer (an attention
        # layer, since (n_layers-1) % 4 == 3 for every geometry here), plus the
        # four glue tensors the real checkpoints carry. Random, never read.
        g = torch.Generator().manual_seed(seed + 1)
        src = f"model.layers.{n_layers - 1}."
        for k in list(tens):
            if k.startswith(src):
                tens["mtp.layers.0." + k[len(src):]] = torch.randn(
                    tens[k].shape, generator=g, dtype=torch.float32).to(tens[k].dtype) * 0.02
        tens["mtp.fc.weight"] = torch.randn((hidden, 2 * hidden), generator=g) * 0.02
        for k in ("mtp.norm.weight", "mtp.pre_fc_norm_embedding.weight",
                  "mtp.pre_fc_norm_hidden.weight"):
            tens[k] = torch.ones(hidden)
        cfg_path = out / "config.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        cfg["mtp_num_hidden_layers"] = 1
        cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    save_file(tens, str(path), metadata={"format": "pt"})
    print(f"shard rewritten: fused_experts={fused_experts} mtp={mtp} ({len(tens)} tensors)")


def build(out: Path, hidden=64, n_layers=8, q_heads=4, kv_heads=2,
          head_dim=16, rope_dim=8, n_experts=8, topk=2, inter=32,
          vocab=320, max_new=16, prompt_ids=None, emit_ref=None,
          ref_mode="attention_only", seed=20260817,
          dn_key_heads=None, dn_value_heads=None,
          fused_experts=False, mtp=False):
    if dn_key_heads is None:
        dn_key_heads = q_heads
    if dn_value_heads is None:
        dn_value_heads = q_heads * 2
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
        linear_num_key_heads=dn_key_heads, linear_num_value_heads=dn_value_heads,
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
    if fused_experts or mtp:
        # after the reference: generate() ran on the in-memory model, the
        # rewrite only changes what is on disk
        _rewrite_shard(out, n_layers, fused_experts, mtp, hidden, seed)


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
    ap.add_argument("--geometry", choices=sorted(GEOMETRIES), default="qwen36-35b",
                    help="structural numbers of a real checkpoint at toy widths")
    for name in ("layers", "experts", "topk", "q-heads", "kv-heads", "hidden", "inter",
                 "dn-key-heads", "dn-value-heads"):
        ap.add_argument(f"--{name}", type=int, default=None,
                        help=f"override the preset's {name.replace('-', '_')}")
    ap.add_argument("--fused-experts", action="store_true", default=None,
                    help="save experts as per-layer gate_up_proj/down_proj (real layout)")
    ap.add_argument("--mtp", action="store_true", default=None,
                    help="add a one-layer mtp.* head and mtp_num_hidden_layers=1")
    args = ap.parse_args()
    geo = dict(GEOMETRIES[args.geometry])
    for arg, key in (("layers", "n_layers"), ("experts", "n_experts"), ("topk", "topk"),
                     ("q_heads", "q_heads"), ("kv_heads", "kv_heads"), ("hidden", "hidden"),
                     ("inter", "inter"), ("dn_key_heads", "dn_key_heads"),
                     ("dn_value_heads", "dn_value_heads"), ("fused_experts", "fused_experts"),
                     ("mtp", "mtp")):
        if getattr(args, arg) is not None:
            geo[key] = getattr(args, arg)
    print(f"geometry {args.geometry}: " + ", ".join(f"{k}={v}" for k, v in geo.items()))

    prompt_ids = None
    if args.prompt_ids:
        prompt_ids = [int(x) for x in args.prompt_ids.split(",") if x.strip() != ""]
    emit = args.emit_ref if args.emit_ref else None

    build(Path(args.out), max_new=args.max_new, prompt_ids=prompt_ids, emit_ref=emit,
          ref_mode=args.ref_mode, seed=args.seed, **geo)


if __name__ == "__main__":
    main()
