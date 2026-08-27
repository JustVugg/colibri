#!/usr/bin/env python3
"""Generate the deterministic GLM-5.3-Flash text oracle contract.

The fixture is deliberately produced by Hugging Face's official
Glm5NextTextModel.  It covers KDA, DSA, mHC, dense FFN and routed MoE before a
Colibri runtime exists, so the future engine has an independent target instead
of validating itself.
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import OrderedDict
from pathlib import Path


SEED = 1243
VOCAB = 128
HIDDEN = 128
LAYERS = 4
HEADS = 4
HEAD_DIM = 32
EXPERTS = 4
TOPK = 2
MOE = 128


def require_dependencies():
    try:
        import torch
        import transformers
        from safetensors.torch import save_file
        from transformers import Glm5NextTextConfig
        from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextModel
    except Exception as exc:  # pragma: no cover - regeneration diagnostic
        raise SystemExit(
            "GLM-5.3 tiny generation requires the pinned PyTorch, "
            "Transformers and safetensors dependencies"
        ) from exc
    if transformers.__version__ != "5.16.1":
        raise SystemExit(
            f"expected Transformers 5.16.1, found {transformers.__version__}"
        )
    return torch, transformers, save_file, Glm5NextTextConfig, Glm5NextTextModel


def config_kwargs() -> dict[str, object]:
    return {
        "vocab_size": VOCAB,
        "hidden_size": HIDDEN,
        "intermediate_size": 256,
        "moe_intermediate_size": MOE,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": HEADS,
        "num_key_value_heads": HEADS,
        "n_shared_experts": 1,
        "n_routed_experts": EXPERTS,
        "num_experts_per_tok": TOPK,
        "kv_lora_rank": 64,
        "q_lora_rank": 128,
        "qk_rope_head_dim": 0,
        "qk_nope_head_dim": HEAD_DIM,
        "v_head_dim": HEAD_DIM,
        "max_position_embeddings": 128,
        "layer_types": ["linear_attention"] * 3 + ["deepseek_sparse_attention"],
        "mlp_layer_types": ["dense"] * 3 + ["sparse"],
        "indexer_types": ["full"] * LAYERS,
        "index_topk": 4,
        "index_kpool": 2,
        "index_head_dim": HEAD_DIM,
        "index_n_heads": 2,
        "linear_head_dim": HEAD_DIM,
        "linear_num_heads": HEADS,
        "linear_conv_kernel_dim": 4,
        "hc_mult": 2,
        "hc_sinkhorn_iters": 3,
        "pad_token_id": None,
        "bos_token_id": 0,
        "eos_token_id": 1,
        "tie_word_embeddings": False,
    }


def runtime_config(version: str) -> dict[str, object]:
    text = config_kwargs()
    text.update({
        "model_type": "glm5_next_text",
        "linear_attn_config": {
            "num_heads": HEADS,
            "head_dim": HEAD_DIM,
            "short_conv_kernel_size": 4,
            "kda_layers": [0, 1, 2],
        },
        "num_nextn_predict_layers": 0,
    })
    return {
        "architectures": ["Glm5NextForConditionalGeneration"],
        "model_type": "glm5_next",
        "transformers_version": version,
        "torch_dtype": "float32",
        "text_config": text,
    }


def production_layout(torch, model, head) -> OrderedDict[str, object]:
    """Use production prefixes and split HF's fused tiny expert tensors."""
    output: OrderedDict[str, object] = OrderedDict()
    for name, tensor in model.state_dict().items():
        prefix = "model.language_model."
        if name.endswith("mlp.experts.gate_up_proj"):
            base = prefix + name.removesuffix("gate_up_proj")
            for expert in range(EXPERTS):
                output[f"{base}{expert}.gate_proj.weight"] = tensor[expert, :MOE].contiguous()
                output[f"{base}{expert}.up_proj.weight"] = tensor[expert, MOE:].contiguous()
        elif name.endswith("mlp.experts.down_proj"):
            base = prefix + name.removesuffix("down_proj")
            for expert in range(EXPERTS):
                output[f"{base}{expert}.down_proj.weight"] = tensor[expert].contiguous()
        else:
            output[prefix + name] = tensor.detach().contiguous()
    output["lm_head.weight"] = head.weight.detach().contiguous()
    return output


def make_tokenizer() -> dict[str, object]:
    added = [
        {"id": token, "content": f"<t{token:03d}>", "single_word": False,
         "lstrip": False, "rstrip": False, "normalized": False, "special": True}
        for token in range(VOCAB)
    ]
    return {
        "version": "1.0", "truncation": None, "padding": None,
        "added_tokens": added, "normalizer": None, "pre_tokenizer": None,
        "post_processor": None, "decoder": None,
        "model": {"type": "BPE", "dropout": None, "unk_token": None,
                  "continuing_subword_prefix": "", "end_of_word_suffix": "",
                  "fuse_unk": False, "byte_fallback": False,
                  "ignore_merges": True, "vocab": {"x": VOCAB - 1}, "merges": []},
    }


def oracle(torch, transformers, model, head) -> dict[str, object]:
    prompt = [5, 7, 9, 11, 13, 17, 19, 23]
    ids = torch.tensor([prompt], dtype=torch.long)
    with torch.no_grad():
        full_hidden = model(ids, use_cache=False).last_hidden_state
        full_logits = head(full_hidden)
        prefix = model(ids[:, :-1], use_cache=True)
        incremental = model(
            ids[:, -1:], use_cache=True, past_key_values=prefix.past_key_values
        )
        incremental_logits = head(incremental.last_hidden_state)[:, -1]
    delta = float((full_logits[:, -1] - incremental_logits).abs().max())
    if delta > 1e-5:
        raise RuntimeError(f"prefill/incremental cache mismatch: {delta}")
    top_values, top_indices = full_logits[0, -1].topk(8)
    cache_shapes = []
    for layer in prefix.past_key_values.layers:
        if hasattr(layer, "recurrent_states"):
            cache_shapes.append({
                "kind": "kda",
                "conv": list(layer.conv_states[0].shape),
                "recurrent": list(layer.recurrent_states[0].shape),
            })
        else:
            cache_shapes.append({
                "kind": "dsa",
                "keys": list(layer.keys.shape),
                "values": list(layer.values.shape),
                "indexer": list(layer.indexer_keys.shape),
            })
    return {
        "schema_version": 1,
        "source": "transformers",
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "seed": SEED,
        "prompt_ids": prompt,
        "teacher_forcing_ids": full_logits[0].argmax(-1).tolist(),
        "last_logits_top_ids": top_indices.tolist(),
        "last_logits_top_values": top_values.tolist(),
        "cache_max_abs_delta": delta,
        "cache_shapes": cache_shapes,
    }


def write_c_array(stream, name: str, tensor, dimensions: str) -> None:
    values = tensor.detach().float().reshape(-1).tolist()
    stream.write(f"static const float {name}{dimensions} = {{\n")
    for start in range(0, len(values), 8):
        row = ", ".join(f"{value:.9g}f" for value in values[start:start + 8])
        stream.write(f"  {row},\n")
    stream.write("};\n\n")


def write_kda_case(torch, model, output: Path) -> None:
    """Emit a C fixture from the official recurrent KDA fallback."""
    from transformers.models.glm5_next.modeling_glm5_next import (
        causal_conv1d_fn, recurrent_kimi_delta_attention,
    )
    attention = model.layers[0].self_attn
    hidden = model.embed_tokens(torch.tensor([[5, 7]], dtype=torch.long))
    raw = torch.cat([
        attention.q_proj(hidden), attention.k_proj(hidden), attention.v_proj(hidden)
    ], dim=-1)
    convolved = causal_conv1d_fn(
        raw.transpose(1, 2), attention.conv1d.weight.squeeze(1),
        bias=None, activation=attention.activation,
    ).transpose(1, 2)
    query, key, value = convolved.split(HEADS * HEAD_DIM, dim=-1)
    decay = attention.forget_gate(hidden).view(1, 2, HEADS, HEAD_DIM)
    beta = torch.sigmoid(attention.b_proj(hidden))
    expected, _ = recurrent_kimi_delta_attention(
        query.view(1, 2, HEADS, HEAD_DIM),
        key.view(1, 2, HEADS, HEAD_DIM),
        value.view(1, 2, HEADS, HEAD_DIM),
        g=decay, beta=beta, initial_state=None, output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    with output.open("w", encoding="utf-8") as stream:
        stream.write("#ifndef COLIBRI_GLM53_KDA_CASE_H\n#define COLIBRI_GLM53_KDA_CASE_H\n\n")
        stream.write(f"#define GLM53_KDA_STEPS 2\n#define GLM53_KDA_HEADS {HEADS}\n")
        stream.write(f"#define GLM53_KDA_DIM {HEAD_DIM}\n#define GLM53_KDA_KERNEL 4\n\n")
        write_c_array(stream, "glm53_kda_qkv", raw[0],
                      "[2 * 3 * GLM53_KDA_HEADS * GLM53_KDA_DIM]")
        write_c_array(stream, "glm53_kda_conv", attention.conv1d.weight.squeeze(1),
                      "[3 * GLM53_KDA_HEADS * GLM53_KDA_DIM * GLM53_KDA_KERNEL]")
        write_c_array(stream, "glm53_kda_decay", decay[0],
                      "[2 * GLM53_KDA_HEADS * GLM53_KDA_DIM]")
        write_c_array(stream, "glm53_kda_beta", beta[0], "[2 * GLM53_KDA_HEADS]")
        write_c_array(stream, "glm53_kda_output", expected[0],
                      "[2 * GLM53_KDA_HEADS * GLM53_KDA_DIM]")
        stream.write("#endif\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default = Path(__file__).resolve().parents[1] / "glm53_tiny"
    parser.add_argument("--output", type=Path, default=default)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    torch, transformers, save_file, Config, Model = require_dependencies()
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    model = Model(Config(**config_kwargs())).eval()
    head = torch.nn.Linear(HIDDEN, VOCAB, bias=False).eval()
    weights = production_layout(torch, model, head)
    reference = oracle(torch, transformers, model, head)
    output = args.output.resolve()
    if output.exists():
        if not args.force:
            raise SystemExit(f"output exists (use --force): {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    (output / "config.json").write_text(
        json.dumps(runtime_config(transformers.__version__), indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "tokenizer.json").write_text(
        json.dumps(make_tokenizer(), separators=(",", ":")) + "\n", encoding="utf-8"
    )
    save_file(weights, output / "model.safetensors", metadata={
        "format": "pt", "generator": "c/tools/make_glm53_tiny.py"
    })
    (output / "ref.json").write_text(
        json.dumps(reference, indent=2) + "\n", encoding="utf-8"
    )
    write_kda_case(torch, model, output / "glm53_kda_case.h")
    size = sum(path.stat().st_size for path in output.iterdir())
    print(f"wrote {output} ({len(weights)} tensors, {size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
