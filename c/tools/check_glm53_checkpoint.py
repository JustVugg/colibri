#!/usr/bin/env python3
"""Validate the tensors consumed by the GLM-5.3 text runtime."""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path


def read_header(path: Path) -> dict[str, object]:
    with path.open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        return json.loads(stream.read(size))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--headers-json", type=Path)
    parser.add_argument("--require-fp8", action="store_true")
    args = parser.parse_args()
    if not args.model and not (args.config and args.headers_json):
        parser.error("provide --model or both --config and --headers-json")
    config_path = args.config or args.model / "config.json"
    config = json.loads(config_path.read_text())
    text = config["text_config"]
    if args.headers_json:
        tensors = json.loads(args.headers_json.read_text())
    else:
        tensors = {}
        for shard in sorted(args.model.glob("*.safetensors")):
            tensors.update({name: spec for name, spec in read_header(shard).items()
                            if name != "__metadata__"})
    failures: list[str] = []

    def tensor(name: str, shape: list[int], dtypes=("BF16", "F32")) -> None:
        actual = tensors.get(name)
        if not actual:
            failures.append(f"missing {name}")
        elif actual["shape"] != shape or actual["dtype"] not in dtypes:
            failures.append(f"invalid {name}: {actual['dtype']} {actual['shape']}, expected {dtypes} {shape}")

    def matrix(name: str, rows: int, columns: int, fp8: bool) -> None:
        actual = tensors.get(name)
        if not actual:
            failures.append(f"missing {name}")
            return
        if actual["shape"] != [rows, columns]:
            failures.append(f"invalid shape {name}: {actual['shape']}")
            return
        if actual["dtype"] == "F8_E4M3":
            scale = name + "_scale_inv"
            tensor(scale, [(rows + 127) // 128, (columns + 127) // 128], ("F32",))
        elif actual["dtype"] not in ("BF16", "F32") or (fp8 and args.require_fp8):
            failures.append(f"invalid dtype {name}: {actual['dtype']}")

    hidden = text["hidden_size"]
    layers = text["num_hidden_layers"]
    heads = text["num_attention_heads"]
    key_dim = text["qk_nope_head_dim"] + text["qk_rope_head_dim"]
    value_dim = text["v_head_dim"]
    q_rank = text["q_lora_rank"]
    kv_rank = text["kv_lora_rank"]
    index_heads = text["index_n_heads"]
    index_dim = text["index_head_dim"]
    hc = text["hc_mult"]
    linear = text["linear_attn_config"]
    kda_heads, kda_dim = linear["num_heads"], linear["head_dim"]
    kda_width = kda_heads * kda_dim
    prefix = "model.language_model."
    tensor(prefix + "embed_tokens.weight", [text["vocab_size"], hidden])
    tensor(prefix + "norm.weight", [hidden])
    tensor("lm_head.weight", [text["vocab_size"], hidden])
    for layer in range(layers):
        base = f"{prefix}layers.{layer}."
        tensor(base + "input_layernorm.weight", [hidden])
        tensor(base + "post_attention_layernorm.weight", [hidden])
        mix = (2 + hc) * hc
        for site in ("attn", "ffn"):
            tensor(base + f"hc_{site}_fn", [mix, hc * hidden])
            tensor(base + f"hc_{site}_base", [mix])
            tensor(base + f"hc_{site}_scale", [3])
        if text["layer_types"][layer] == "linear_attention":
            for projection in ("q", "k", "v"):
                matrix(base + f"self_attn.{projection}_proj.weight", kda_width, hidden, False)
                tensor(base + f"self_attn.{projection}_conv1d.weight",
                       [kda_width, 1, linear["short_conv_kernel_size"]])
            matrix(base + "self_attn.f_a_proj.weight", kda_dim, hidden, False)
            matrix(base + "self_attn.f_b_proj.weight", kda_width, kda_dim, False)
            tensor(base + "self_attn.dt_bias", [kda_width], ("F32", "BF16"))
            tensor(base + "self_attn.A_log", [kda_heads], ("F32", "BF16"))
            matrix(base + "self_attn.b_proj.weight", kda_heads, hidden, False)
            matrix(base + "self_attn.g_a_proj.weight", kda_dim, hidden, False)
            matrix(base + "self_attn.g_b_proj.weight", kda_width, kda_dim, False)
            tensor(base + "self_attn.o_norm.weight", [kda_dim])
            matrix(base + "self_attn.o_proj.weight", hidden, kda_width, False)
        else:
            matrix(base + "self_attn.q_a_proj.weight", q_rank, hidden, True)
            tensor(base + "self_attn.q_a_layernorm.weight", [q_rank])
            matrix(base + "self_attn.q_b_proj.weight", heads * key_dim, q_rank, True)
            matrix(base + "self_attn.kv_a_proj_with_mqa.weight", kv_rank, hidden, True)
            tensor(base + "self_attn.kv_a_layernorm.weight", [kv_rank])
            matrix(base + "self_attn.kv_b_proj.weight", heads * (key_dim + value_dim), kv_rank, False)
            matrix(base + "self_attn.o_proj.weight", hidden, heads * value_dim, True)
            matrix(base + "self_attn.indexer.wq_b.weight", index_heads * index_dim, q_rank, False)
            matrix(base + "self_attn.indexer.wk.weight", index_dim, hidden, False)
            tensor(base + "self_attn.indexer.k_norm.weight", [index_dim])
            tensor(base + "self_attn.indexer.k_norm.bias", [index_dim])
            matrix(base + "self_attn.indexer.index_kpool_compress_gate", index_dim, hidden, False)
            matrix(base + "self_attn.indexer.weights_proj.weight", index_heads, hidden, False)
            tensor(base + "self_attn.indexer.index_kpool_compress_ape", [text["index_kpool"], index_dim])
        if text["mlp_layer_types"][layer] == "dense":
            intermediate = text["intermediate_size"]
            matrix(base + "mlp.gate_proj.weight", intermediate, hidden, True)
            matrix(base + "mlp.up_proj.weight", intermediate, hidden, True)
            matrix(base + "mlp.down_proj.weight", hidden, intermediate, True)
        else:
            intermediate = text["moe_intermediate_size"]
            matrix(base + "mlp.gate.weight", text["n_routed_experts"], hidden, False)
            tensor(base + "mlp.gate.e_score_correction_bias", [text["n_routed_experts"]])
            for part, rows, columns in (("gate", intermediate, hidden), ("up", intermediate, hidden),
                                        ("down", hidden, intermediate)):
                matrix(base + f"mlp.shared_experts.{part}_proj.weight", rows, columns, True)
                for expert in range(text["n_routed_experts"]):
                    matrix(base + f"mlp.experts.{expert}.{part}_proj.weight", rows, columns, True)
    if failures:
        raise SystemExit("GLM-5.3 checkpoint contract failed:\n" + "\n".join(failures[:50]))
    print(f"PASS GLM-5.3 checkpoint contract: {layers} layers, {len(tensors)} tensors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
