#!/usr/bin/env python3
"""Create a deterministic, text-only Qwen4-Exp-shaped Qwen3.8 fixture.

The fixture deliberately uses :class:`Qwen4ExpForCausalLM` (the upstream text
class), rather than the multimodal ``Qwen4ExpForConditionalGeneration``
wrapper.  Consequently its state-dict names start at ``model.embed_tokens``;
the released multimodal checkpoint adds the extra ``model.language_model``
prefix.  This keeps vision and MTP weights out while retaining the exact text
module names (``layers.*.linear_attn``, ``layers.*.self_attn``, ``layers.*.ple``
and ``layers.*.mlp``) an engine needs to support.

Usage::

    python tools/make_qwen38_tiny.py --out ./qwen38_tiny

Requires Transformers 5.16.1, the first release with Qwen4-Exp support.
"""

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass

try:
    import torch
except ImportError as exc:
    sys.exit(f"Missing dependency: {exc}. Install torch and transformers.")


TRANSFORMERS_VERSION = "5.16.1"
SEED = 20260826


def _classes():
    try:
        import transformers
        from transformers import Qwen4ExpForCausalLM, Qwen4ExpTextConfig
    except ImportError as exc:
        sys.exit(
            f"Missing Qwen4-Exp support. Install transformers=={TRANSFORMERS_VERSION}: {exc}"
        )
    if transformers.__version__ != TRANSFORMERS_VERSION:
        sys.exit(
            f"Qwen3.8 fixture generation requires transformers=={TRANSFORMERS_VERSION}; "
            f"found {transformers.__version__}."
        )
    return transformers, Qwen4ExpForCausalLM, Qwen4ExpTextConfig


def _digest(tensor):
    data = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def _reference(model, prompt_ids, max_new):
    model.eval()
    input_ids = torch.tensor([prompt_ids], dtype=torch.long)
    with torch.no_grad():
        prefill = model(
            input_ids=input_ids,
            use_cache=True,
            output_hidden_states=True,
            output_router_logits=True,
        )
        next_id = prefill.logits[:, -1:, :].argmax(dim=-1)
        decode = model(
            input_ids=next_id,
            past_key_values=prefill.past_key_values,
            use_cache=True,
            output_hidden_states=True,
            output_router_logits=True,
        )
        # Fixed-length greedy decode is intentional.  The seeded fixture emits
        # EOS before max_new, and continuing through it exercises the PLE
        # history reset that an ordinary generate() call would stop before.
        generated = input_ids
        current = input_ids
        past = None
        for _ in range(max_new):
            step = model(input_ids=current, past_key_values=past, use_cache=True)
            token = step.logits[:, -1:, :].argmax(dim=-1)
            generated = torch.cat((generated, token), dim=1)
            current = token
            past = step.past_key_values

    def summary(outputs):
        logits = outputs.logits
        hidden = outputs.hidden_states[-1] if outputs.hidden_states else None
        return {
            "logits_shape": list(logits.shape),
            "logits_sha256": _digest(logits),
            "last_logits_argmax": int(logits[:, -1, :].argmax(dim=-1).item()),
            "last_hidden_shape": list(hidden.shape) if hidden is not None else None,
            "last_hidden_sha256": _digest(hidden) if hidden is not None else None,
            "router_layers": len(outputs.router_logits) if outputs.router_logits else 0,
        }

    full_ids = generated[0].tolist()
    return {
        "prefill": summary(prefill),
        "decode": {"input_ids": next_id[0].tolist(), **summary(decode)},
        "prompt_ids": prompt_ids,
        "full_ids": full_ids,
        "generated_ids": full_ids[len(prompt_ids) :],
        # Token equality is necessary but not sufficient: retain the logits
        # that predicted the final generated ID for a numeric C-engine gate.
        "final_logits": step.logits[0, -1, :].float().cpu().tolist(),
        "cache_class": type(prefill.past_key_values).__name__,
    }


FP8_BLOCK = 128
FP8_E4M3_MAX = 448.0


def _fp8_block_quant(w):
    """Quantize a 2-D BF16/F32 matrix to e4m3 with one scale per 128x128 block,
    the released checkpoint's layout (`weight_block_size [128, 128]`,
    `weight_scale_inv` = the multiplier that restores the value). Returns
    (e4m3 tensor, scale_inv tensor [ceil(O/128), ceil(I/128)], dequantized w).
    The dequantized copy is what goes back into the model, so the reference
    the fixture emits is the arithmetic of the bytes on disk."""
    w32 = w.float()
    O, I = w32.shape
    nb_o, nb_i = (O + FP8_BLOCK - 1) // FP8_BLOCK, (I + FP8_BLOCK - 1) // FP8_BLOCK
    scale_inv = torch.empty(nb_o, nb_i, dtype=torch.float32)
    q = torch.empty(O, I, dtype=torch.float8_e4m3fn)
    deq = torch.empty(O, I, dtype=torch.float32)
    for bo in range(nb_o):
        for bi in range(nb_i):
            blk = w32[bo * FP8_BLOCK:(bo + 1) * FP8_BLOCK, bi * FP8_BLOCK:(bi + 1) * FP8_BLOCK]
            amax = blk.abs().max().clamp(min=1e-12)
            s = (amax / FP8_E4M3_MAX).item()
            qb = (blk / s).to(torch.float8_e4m3fn)
            q[bo * FP8_BLOCK:(bo + 1) * FP8_BLOCK, bi * FP8_BLOCK:(bi + 1) * FP8_BLOCK] = qb
            deq[bo * FP8_BLOCK:(bo + 1) * FP8_BLOCK, bi * FP8_BLOCK:(bi + 1) * FP8_BLOCK] = qb.float() * s
            scale_inv[bo, bi] = s
    return q, scale_inv, deq


def _fp8_experts_in_model(model):
    """Fake-quantize every routed expert matrix in place (dequantized values
    back into the BF16 parameters) and return {saved_name: (q, scale_inv)} for
    the shard rewrite. In memory the experts are two fused parameters per
    layer, `mlp.experts.gate_up_proj` [E, 2I, H] and `mlp.experts.down_proj`
    [E, H, I]; save_pretrained splits them into the per-expert
    `mlp.experts.<e>.{gate,up,down}_proj.weight` the release ships, so the
    returned names are the saved ones. Shared expert, router and everything
    dense stay BF16 like the release."""
    packed = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith(".mlp.experts.gate_up_proj"):
                base = name[: -len("gate_up_proj")]              # ...mlp.experts.
                E, twoI, H = param.shape; I = twoI // 2
                for e in range(E):
                    for kind, sl in (("gate_proj", slice(0, I)), ("up_proj", slice(I, twoI))):
                        q, scale_inv, deq = _fp8_block_quant(param.data[e, sl, :])
                        param.data[e, sl, :] = deq.to(param.dtype)
                        packed[f"{base}{e}.{kind}.weight"] = (q, scale_inv)
            elif name.endswith(".mlp.experts.down_proj"):
                base = name[: -len("down_proj")]
                E = param.shape[0]
                for e in range(E):
                    q, scale_inv, deq = _fp8_block_quant(param.data[e])
                    param.data[e] = deq.to(param.dtype)
                    packed[f"{base}{e}.down_proj.weight"] = (q, scale_inv)
    if not packed:
        raise RuntimeError("no routed expert parameters found to quantize")
    return packed


def _rewrite_shard_fp8(out: Path, packed):
    """Replace the BF16 expert tensors in model.safetensors with e4m3 bytes and
    BF16 `weight_scale_inv` sidecars, the way the release ships them. The
    safetensors writer groups tensors by dtype and sorts by name inside a
    group, which is what puts gate_proj.weight and up_proj.weight next to each
    other -- the adjacency the engine's native FP8 path requires."""
    from safetensors.torch import load_file, save_file
    path = out / "model.safetensors"
    tensors = load_file(str(path))
    for name, (q, scale_inv) in packed.items():
        assert name in tensors, name
        tensors[name] = q.contiguous()
        tensors[name + "_scale_inv"] = scale_inv.to(torch.bfloat16).contiguous()
    # The release keeps a layer's gate/up expert tensors and its down_proj
    # tensors in different shards. Because the writer sorts BF16 before F8 and
    # by name within a dtype, that is what makes every layer's gate/up
    # weight_scale_inv sidecars one compact byte range and the down sidecars
    # another -- the invariant behind the engine's resident scale bank
    # (q38_prepare_expert_scale_bank). One file would interleave
    # down/gate/up per expert and silently send the engine down the
    # per-matrix fallback, so the fixture ships two shards plus the index.
    down = {k: v for k, v in tensors.items() if ".mlp.experts." in k and ".down_proj." in k}
    rest = {k: v for k, v in tensors.items() if k not in down}
    shards = {"model-00001-of-00002.safetensors": rest, "model-00002-of-00002.safetensors": down}
    weight_map = {}
    for fname, group in shards.items():
        save_file(group, str(out / fname), metadata={"format": "pt"})
        for k in group: weight_map[k] = fname
    path.unlink()
    (out / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": sum(v.numel() * v.element_size() for v in tensors.values())},
         "weight_map": weight_map}, indent=1), encoding="utf-8")
    cfg_path = out / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["quantization_config"] = {"quant_method": "fp8", "activation_scheme": "dynamic",
                                  "weight_block_size": [FP8_BLOCK, FP8_BLOCK],
                                  "fixture_note": "routed experts only, as in the release"}
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    print(f"fp8 experts: {len(packed)} matrices rewritten as F8_E4M3 + BF16 weight_scale_inv")


def build(out: Path, prompt_ids=None, max_new=8, seed=SEED, emit_ref=True, fp8_experts=False):
    if max_new < 1:
        raise ValueError("max_new must be at least 1")
    random.seed(seed)
    torch.manual_seed(seed)
    transformers, ModelCls, ConfigCls = _classes()

    # Four layers exercise both token mixers; layer 0 is a PLE-enabled GDN
    # layer and layers 1/3 are QSA layers.  All dimensions are intentionally
    # small, but the relationships are the production relationships.
    config = ConfigCls(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.5,
        },
        partial_rotary_factor=0.5,
        layer_types=[
            "linear_attention",
            "qwen_sparse_attention",
            "linear_attention",
            "qwen_sparse_attention",
        ],
        linear_conv_kernel_dim=4,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        hc_count=4,
        hc_lowrank=8,
        ngram_size=3,
        heads_per_ngram=2,
        ngram_vocab_size_base=31,
        make_ngram_vocab_size_divisible_by=4,
        split_ngram_parts=2,
        ple_layer_ids=[1],  # one-based; layer 0 is GDN
        ple_embed_dim=32,
        ple_conv_kernel_size=4,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=4,
        indexer_budget=4,
        indexer_compress_ratio=2,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        norm_topk_prob=True,
        output_gate_type="sigmoid",
        attention_bias=False,
        attention_dropout=0.0,
        tie_word_embeddings=False,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    model = ModelCls(config)
    # The released checkpoint is BF16.  Materialize the fixture in BF16 before
    # saving so the digest emitted here is exactly reproducible after a
    # save/load round trip (and follows the production arithmetic path).
    model = model.to(dtype=torch.bfloat16)
    model.eval()
    packed = _fp8_experts_in_model(model) if fp8_experts else None
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out), safe_serialization=True)
    if packed:
        _rewrite_shard_fp8(out, packed)

    if prompt_ids is None:
        prompt_ids = [1, 3, 4, 5, 6]
    if not prompt_ids or any(token < 0 or token >= config.vocab_size for token in prompt_ids):
        raise ValueError(f"prompt ids must be in [0, {config.vocab_size}) and non-empty")

    payload = {
        "schema_version": 2,
        "model": "qwen38_tiny",
        "seed": seed,
        "transformers_version": transformers.__version__,
        "text_only": True,
        "fp8_experts": bool(fp8_experts),
        "naming": "upstream Qwen4ExpForCausalLM (no model.language_model prefix)",
        "config_summary": {
            "hidden_size": config.hidden_size,
            "num_hidden_layers": config.num_hidden_layers,
            "layer_types": config.layer_types,
            "ple_layer_ids": config.ple_layer_ids,
            "num_experts": config.num_experts,
            "num_experts_per_tok": config.num_experts_per_tok,
            "tie_word_embeddings": config.tie_word_embeddings,
        },
    }
    payload.update(_reference(model, prompt_ids, max_new))
    if emit_ref:
        ref_path = out / "ref.json"
        ref_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Reference written to {ref_path}")
    print(f"Tiny Qwen3.8 fixture written to {out}")
    print(f"prompt_ids={payload['prompt_ids']}")
    print(f"full_ids={payload['full_ids']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path, help="Output model directory")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-new", type=int, default=8)
    parser.add_argument("--prompt-ids", help="Comma-separated token IDs")
    parser.add_argument("--no-ref", action="store_true", help="Do not emit ref.json")
    parser.add_argument("--fp8-experts", action="store_true",
                        help="routed experts as F8_E4M3 with 128x128 block weight_scale_inv "
                             "sidecars (the release layout); the reference uses the same "
                             "quantized values")
    args = parser.parse_args()
    prompt = [int(x) for x in args.prompt_ids.split(",") if x.strip()] if args.prompt_ids else None
    build(args.out, prompt_ids=prompt, max_new=args.max_new, seed=args.seed, emit_ref=not args.no_ref,
          fp8_experts=args.fp8_experts)


if __name__ == "__main__":
    main()
