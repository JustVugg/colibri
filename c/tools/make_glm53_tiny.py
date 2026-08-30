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
import types
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
        "hc_eps": 1.0e-6,
        "rms_norm_eps": 1.0e-5,
        "routed_scaling_factor": 2.5,
        "swiglu_limit": 10.0,
        "pad_token_id": None,
        "bos_token_id": 0,
        "eos_token_id": 1,
        "tie_word_embeddings": False,
    }


def runtime_config(version: str, fp8_mlp: bool = False) -> dict[str, object]:
    text = config_kwargs()
    text.update({
        "model_type": "glm5_next_text",
        "linear_attn_config": {
            "num_heads": HEADS,
            "head_dim": HEAD_DIM,
            "short_conv_kernel_size": 4,
            "gate_lower_bound": -5.0,
            "kda_layers": [0, 1, 2],
        },
        "num_nextn_predict_layers": 0,
    })
    result = {
        "architectures": ["Glm5NextForConditionalGeneration"],
        "model_type": "glm5_next",
        "transformers_version": version,
        "torch_dtype": "float32",
        "text_config": text,
    }
    if fp8_mlp:
        result["quantization_config"] = {
            "quant_method": "fp8",
            "fmt": "e4m3",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
        }
    return result


def production_layout(torch, model, head) -> OrderedDict[str, object]:
    """Recreate the official checkpoint's pre-Transformers disk layout."""
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
        elif name.endswith("self_attn.conv1d.weight"):
            base = prefix + name.removesuffix("conv1d.weight")
            q_conv, k_conv, v_conv = tensor.chunk(3, dim=0)
            output[f"{base}q_conv1d.weight"] = q_conv.contiguous()
            output[f"{base}k_conv1d.weight"] = k_conv.contiguous()
            output[f"{base}v_conv1d.weight"] = v_conv.contiguous()
        else:
            disk_name = name
            disk_name = disk_name.replace("attn_hc.fn", "hc_attn_fn")
            disk_name = disk_name.replace("attn_hc.base", "hc_attn_base")
            disk_name = disk_name.replace("attn_hc.scale", "hc_attn_scale")
            disk_name = disk_name.replace("ffn_hc.fn", "hc_ffn_fn")
            disk_name = disk_name.replace("ffn_hc.base", "hc_ffn_base")
            disk_name = disk_name.replace("ffn_hc.scale", "hc_ffn_scale")
            disk_name = disk_name.replace("self_attn.forget_gate.", "self_attn.")
            output[prefix + disk_name] = tensor.detach().contiguous()
    output["lm_head.weight"] = head.weight.detach().contiguous()
    return output


def initialize_contract_weights(torch, model, head) -> None:
    """Remove random-init degeneracies that make top-k ties implementation-defined."""
    with torch.no_grad():
        embeddings = model.embed_tokens.weight.detach().clone()
        for token in range(VOCAB):
            head.weight[(token + 1) % VOCAB].copy_(embeddings[token])
        indexer = model.layers[3].self_attn.indexer
        indexer.index_kpool_compress_gate.copy_(
            torch.linspace(-0.01, 0.01, indexer.index_kpool_compress_gate.numel()).view_as(
                indexer.index_kpool_compress_gate
            )
        )
        indexer.index_kpool_compress_ape.copy_(
            torch.linspace(-0.1, 0.1, indexer.index_kpool_compress_ape.numel()).view_as(
                indexer.index_kpool_compress_ape
            )
        )
        indexer.weights_proj.weight.copy_(
            torch.linspace(-0.02, 0.03, indexer.weights_proj.weight.numel()).view_as(
                indexer.weights_proj.weight
            )
        )
        indexer.wq_b.weight.zero_()
        indexer.wk.weight.zero_()
        for dim in range(indexer.head_dim):
            indexer.wk.weight[dim, dim] = 1.0
            for index_head in range(indexer.n_heads):
                indexer.wq_b.weight[index_head * indexer.head_dim + dim, dim] = 1.0


def quantize_block_weight(torch, weight):
    import torch.nn.functional as F

    rows, columns = weight.shape
    padded_rows = (rows + 127) // 128 * 128
    padded_columns = (columns + 127) // 128 * 128
    padded = F.pad(weight.float(), (0, padded_columns - columns, 0, padded_rows - rows))
    blocks = padded.reshape(padded_rows // 128, 128, padded_columns // 128, 128)
    maximum = blocks.abs().amax(dim=(1, 3))
    scales = torch.where(maximum > 0, maximum / 448.0, torch.ones_like(maximum))
    quantized = (blocks / scales[:, None, :, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    quantized_padded = quantized.reshape(padded_rows, padded_columns)
    dequantized_padded = (quantized.float() * scales[:, None, :, None]).reshape(padded_rows, padded_columns)
    quantized = quantized_padded[:rows, :columns].contiguous()
    dequantized = dequantized_padded[:rows, :columns].contiguous()
    return quantized, scales, dequantized


def dynamic_fp8_qdq(torch, value):
    shape = value.shape
    blocks = value.float().reshape(-1, shape[-1] // 128, 128)
    maximum = blocks.abs().amax(dim=-1)
    scales = torch.where(maximum > 0, maximum / 448.0, torch.ones_like(maximum))
    result = (blocks / scales[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn).float()
    return (result * scales[..., None]).reshape(shape)


def apply_fp8_mlp_oracle(torch, model) -> None:
    import torch.nn.functional as F

    def quantized_linear(module):
        _, _, dequantized = quantize_block_weight(torch, module.weight.detach())
        module.weight.copy_(dequantized)

        def forward(self, value):
            return F.linear(dynamic_fp8_qdq(torch, value), self.weight)

        module.forward = types.MethodType(forward, module)

    with torch.no_grad():
        for layer in model.layers:
            if layer.block_type == "deepseek_sparse_attention":
                quantized_linear(layer.self_attn.q_a_proj)
                quantized_linear(layer.self_attn.q_b_proj)
                quantized_linear(layer.self_attn.kv_a_proj_with_mqa)
                quantized_linear(layer.self_attn.o_proj)
            mlp = layer.mlp
            if hasattr(mlp, "experts"):
                quantized_linear(mlp.shared_experts.gate_proj)
                quantized_linear(mlp.shared_experts.up_proj)
                quantized_linear(mlp.shared_experts.down_proj)
                experts = mlp.experts
                gate, up = experts.gate_up_proj.chunk(2, dim=1)
                for expert in range(EXPERTS):
                    _, _, gate[expert] = quantize_block_weight(torch, gate[expert])
                    _, _, up[expert] = quantize_block_weight(torch, up[expert])
                    _, _, experts.down_proj[expert] = quantize_block_weight(torch, experts.down_proj[expert])

                def expert_forward(self, hidden_states, top_k_index, top_k_weights):
                    final = torch.zeros_like(hidden_states)
                    mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
                    for expert_index in torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero().flatten():
                        top_k_pos, token_index = torch.where(mask[expert_index])
                        current = dynamic_fp8_qdq(torch, hidden_states[token_index])
                        current = self._apply_gate(F.linear(current, self.gate_up_proj[expert_index]))
                        current = F.linear(
                            dynamic_fp8_qdq(torch, current), self.down_proj[expert_index]
                        ) * top_k_weights[token_index, top_k_pos, None]
                        final.index_add_(0, token_index, current.to(final.dtype))
                    return final

                experts.forward = types.MethodType(expert_forward, experts)
            else:
                quantized_linear(mlp.gate_proj)
                quantized_linear(mlp.up_proj)
                quantized_linear(mlp.down_proj)


def quantize_disk_weights(torch, model, weights: OrderedDict[str, object]) -> None:
    replacements = []
    for name, tensor in weights.items():
        mlp = name.endswith(("gate_proj.weight", "up_proj.weight", "down_proj.weight"))
        dsa_suffix = name.endswith(("self_attn.q_a_proj.weight", "self_attn.q_b_proj.weight",
                                    "self_attn.kv_a_proj_with_mqa.weight", "self_attn.o_proj.weight"))
        layer_index = int(name.split(".layers.", 1)[1].split(".", 1)[0]) if ".layers." in name else -1
        dsa = dsa_suffix and layer_index >= 0 and model.layers[layer_index].block_type == "deepseek_sparse_attention"
        if not (mlp or dsa):
            continue
        quantized, scales, _ = quantize_block_weight(torch, tensor)
        replacements.append((name, quantized, scales))
    for name, quantized, scales in replacements:
        weights[name] = quantized
        weights[name + "_scale_inv"] = scales


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
        full_output = model(ids, use_cache=False, output_hidden_states=True)
        full_hidden = full_output.last_hidden_state
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
    generated = []
    sequence = list(prompt)
    with torch.no_grad():
        for _ in range(4):
            greedy_ids = torch.tensor([sequence], dtype=torch.long)
            next_logits = head(model(greedy_ids, use_cache=False).last_hidden_state)[0, -1]
            token = int(next_logits.argmax())
            generated.append(token)
            sequence.append(token)
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
        "last_logits": full_logits[0, -1].tolist(),
        "greedy_new_ids": generated,
        "hidden_state_stats": [
            {"sum": float(state.float().sum()), "square_sum": float(state.float().square().sum())}
            for state in full_output.hidden_states
        ],
        "cache_max_abs_delta": delta,
        "cache_shapes": cache_shapes,
    }


def write_c_array(stream, name: str, tensor, dimensions: str) -> None:
    values = tensor.detach().float().reshape(-1).tolist()
    stream.write(f"static const float {name}{dimensions} = {{\n")
    for start in range(0, len(values), 8):
        literals = []
        for value in values[start:start + 8]:
            literal = f"{value:.9g}"
            if "." not in literal and "e" not in literal:
                literal += ".0"
            literals.append(literal + "f")
        row = ", ".join(literals)
        stream.write(f"  {row},\n")
    stream.write("};\n\n")


def write_u8_array(torch, stream, name: str, tensor, dimensions: str) -> None:
    values = tensor.detach().view(torch.uint8).reshape(-1).tolist()
    stream.write(f"static const unsigned char {name}{dimensions} = {{\n")
    for start in range(0, len(values), 16):
        stream.write("  " + ", ".join(str(value) for value in values[start:start + 16]) + ",\n")
    stream.write("};\n\n")


def write_fp8_case(torch, model, output: Path) -> None:
    weight = model.layers[0].mlp.gate_proj.weight.detach().float()
    rows, columns = weight.shape
    blocks = weight.reshape(rows // 128, 128, columns // 128, 128)
    maximum = blocks.abs().amax(dim=(1, 3))
    scales = torch.where(maximum > 0, maximum / 448.0, torch.ones_like(maximum))
    quantized = (blocks / scales[:, None, :, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    quantized = quantized.reshape_as(weight)
    activation = torch.linspace(-1.25, 1.75, columns)
    activation_blocks = activation.reshape(-1, 128)
    activation_maximum = activation_blocks.abs().amax(dim=1)
    activation_scales = torch.where(
        activation_maximum > 0, activation_maximum / 448.0, torch.ones_like(activation_maximum)
    )
    activation_qdq = (
        (activation_blocks / activation_scales[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn).float()
        * activation_scales[:, None]
    ).reshape(-1)
    dequantized = quantized.float().reshape(rows // 128, 128, columns // 128, 128)
    dequantized = (dequantized * scales[:, None, :, None]).reshape_as(weight)
    expected = torch.mv(dequantized, activation_qdq)
    with output.open("w", encoding="utf-8") as stream:
        stream.write("#ifndef COLIBRI_GLM53_FP8_CASE_H\n#define COLIBRI_GLM53_FP8_CASE_H\n\n")
        stream.write(f"#define GLM53_FP8_ROWS {rows}\n#define GLM53_FP8_COLUMNS {columns}\n\n")
        write_u8_array(torch, stream, "glm53_fp8_weight", quantized,
                       "[GLM53_FP8_ROWS * GLM53_FP8_COLUMNS]")
        write_c_array(stream, "glm53_fp8_scales", scales,
                      "[(GLM53_FP8_ROWS / 128) * (GLM53_FP8_COLUMNS / 128)]")
        write_c_array(stream, "glm53_fp8_input", activation, "[GLM53_FP8_COLUMNS]")
        write_c_array(stream, "glm53_fp8_expected", expected, "[GLM53_FP8_ROWS]")
        stream.write("#endif\n")


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


def write_indexer_case(torch, model, output: Path) -> None:
    attention = model.layers[3].self_attn
    indexer = attention.indexer
    ids = torch.arange(8).view(1, -1)
    values = torch.arange(8 * HIDDEN, dtype=torch.float32).view(1, 8, HIDDEN)
    hidden = torch.sin(values * 0.013) + 0.2 * torch.cos(values * 0.031)
    base = torch.linspace(-1.0, 1.0, indexer.head_dim)
    perturb = torch.sin(torch.arange(indexer.head_dim, dtype=torch.float32) * 0.37)
    for token in range(ids.shape[1]):
        hidden[0, token, :indexer.head_dim] = base + (token + 1) * 0.01 * perturb
    q_resid = torch.zeros_like(hidden)
    q_resid[..., :indexer.head_dim] = base
    with torch.no_grad():
        indexer.wk.weight.zero_()
        indexer.wq_b.weight.zero_()
        for dim in range(indexer.head_dim):
            indexer.wk.weight[dim, dim] = 1.0
            for head in range(indexer.n_heads):
                indexer.wq_b.weight[head * indexer.head_dim + dim, dim] = 1.0
        indexer.weights_proj.weight.copy_(
            torch.linspace(0.001, 0.01, indexer.weights_proj.weight.numel()).view_as(
                indexer.weights_proj.weight
            )
        )
        indexer.index_kpool_compress_gate.copy_(
            torch.linspace(-0.01, 0.01, indexer.index_kpool_compress_gate.numel()).view_as(
                indexer.index_kpool_compress_gate
            )
        )
        indexer.index_kpool_compress_ape.copy_(
            torch.linspace(-0.1, 0.1, indexer.index_kpool_compress_ape.numel()).view_as(
                indexer.index_kpool_compress_ape
            )
        )
    queries = indexer.wq_b(q_resid).view(1, ids.shape[1], indexer.n_heads, indexer.head_dim)
    keys = indexer.k_norm(indexer.wk(hidden))
    gates = torch.nn.functional.linear(hidden, indexer.index_kpool_compress_gate)
    weights = indexer.weights_proj(hidden) * (indexer.n_heads ** -0.5)
    valid = torch.ones_like(ids, dtype=torch.bool)
    with torch.no_grad():
        expected = indexer(hidden, q_resid, valid, None)
    with output.open("w", encoding="utf-8") as stream:
        stream.write("#ifndef COLIBRI_GLM53_INDEXER_CASE_H\n#define COLIBRI_GLM53_INDEXER_CASE_H\n\n")
        stream.write(f"#define GLM53_INDEX_SEQUENCE {ids.shape[1]}\n")
        stream.write(f"#define GLM53_INDEX_HEADS {indexer.n_heads}\n")
        stream.write(f"#define GLM53_INDEX_DIM {indexer.head_dim}\n")
        stream.write(f"#define GLM53_INDEX_POOL {indexer.index_kpool}\n")
        stream.write(f"#define GLM53_INDEX_TOPK {indexer.index_topk}\n")
        stream.write("#define GLM53_INDEX_WIDTH (GLM53_INDEX_TOPK + GLM53_INDEX_POOL - 1)\n\n")
        write_c_array(stream, "glm53_index_queries", queries,
                      "[GLM53_INDEX_SEQUENCE * GLM53_INDEX_HEADS * GLM53_INDEX_DIM]")
        write_c_array(stream, "glm53_index_keys", keys,
                      "[GLM53_INDEX_SEQUENCE * GLM53_INDEX_DIM]")
        write_c_array(stream, "glm53_index_gates", gates,
                      "[GLM53_INDEX_SEQUENCE * GLM53_INDEX_DIM]")
        write_c_array(stream, "glm53_index_weights", weights,
                      "[GLM53_INDEX_SEQUENCE * GLM53_INDEX_HEADS]")
        write_c_array(stream, "glm53_index_ape", indexer.index_kpool_compress_ape,
                      "[GLM53_INDEX_POOL * GLM53_INDEX_DIM]")
        valid_values = ", ".join("1" if value else "0" for value in valid.reshape(-1).tolist())
        stream.write(f"static const unsigned char glm53_index_valid[GLM53_INDEX_SEQUENCE] = {{{valid_values}}};\n")
        expected_values = ", ".join(str(value) for value in expected.reshape(-1).tolist())
        stream.write("static const int glm53_index_expected[GLM53_INDEX_SEQUENCE * GLM53_INDEX_WIDTH] = {\n")
        stream.write(f"  {expected_values}\n}};\n\n#endif\n")


def write_sparse_attention_case(torch, model, output: Path) -> None:
    attention = model.layers[3].self_attn
    ids = torch.tensor([[5, 7, 9, 11, 13, 17, 19, 23]], dtype=torch.long)
    hidden = model.embed_tokens(ids)
    q_resid = attention.q_a_layernorm(attention.q_a_proj(hidden))
    queries = attention.q_b_proj(q_resid).view(1, 8, HEADS, HEAD_DIM)
    compressed = attention.kv_a_proj_with_mqa(hidden)
    latent = attention.kv_a_layernorm(compressed).view(1, 1, 8, 64)
    keys, values = attention.expand_kv(latent, compressed.new_empty(1, 1, 8, 0))
    keys = keys.transpose(1, 2).contiguous()
    values = values.transpose(1, 2).contiguous()
    valid = torch.ones_like(ids, dtype=torch.bool)
    indices = attention.indexer(hidden, q_resid, valid, None)
    width = indices.shape[-1]
    expected = torch.zeros(1, 8, HEADS, HEAD_DIM)
    with torch.no_grad():
        for position in range(8):
            selected = indices[0, position]
            selected = selected[selected.ge(0)]
            for head in range(HEADS):
                scores = queries[0, position, head] @ keys[0, selected, head].T
                probabilities = torch.softmax(scores.float() / (HEAD_DIM ** 0.5), dim=-1)
                expected[0, position, head] = probabilities @ values[0, selected, head].float()
    with output.open("w", encoding="utf-8") as stream:
        stream.write("#ifndef COLIBRI_GLM53_SPARSE_ATTENTION_CASE_H\n#define COLIBRI_GLM53_SPARSE_ATTENTION_CASE_H\n\n")
        stream.write("#define GLM53_SA_SEQUENCE 8\n")
        stream.write(f"#define GLM53_SA_HEADS {HEADS}\n#define GLM53_SA_KEY_DIM {HEAD_DIM}\n")
        stream.write(f"#define GLM53_SA_VALUE_DIM {HEAD_DIM}\n#define GLM53_SA_WIDTH {width}\n\n")
        write_c_array(stream, "glm53_sa_queries", queries,
                      "[GLM53_SA_SEQUENCE * GLM53_SA_HEADS * GLM53_SA_KEY_DIM]")
        write_c_array(stream, "glm53_sa_keys", keys,
                      "[GLM53_SA_SEQUENCE * GLM53_SA_HEADS * GLM53_SA_KEY_DIM]")
        write_c_array(stream, "glm53_sa_values", values,
                      "[GLM53_SA_SEQUENCE * GLM53_SA_HEADS * GLM53_SA_VALUE_DIM]")
        index_values = ", ".join(str(value) for value in indices.reshape(-1).tolist())
        stream.write("static const int glm53_sa_indices[GLM53_SA_SEQUENCE * GLM53_SA_WIDTH] = {\n")
        stream.write(f"  {index_values}\n}};\n\n")
        write_c_array(stream, "glm53_sa_expected", expected,
                      "[GLM53_SA_SEQUENCE * GLM53_SA_HEADS * GLM53_SA_VALUE_DIM]")
        stream.write("#endif\n")


def write_mhc_case(torch, model, output: Path) -> None:
    mhc = model.layers[0].attn_hc
    streams = model.embed_tokens(torch.tensor([[5]])).unsqueeze(2).expand(-1, -1, 2, -1).contiguous()
    with torch.no_grad():
        post, comb, collapsed = mhc(streams)
        branch = torch.sin(torch.arange(HIDDEN, dtype=torch.float32) * 0.07).view(1, 1, HIDDEN)
        expanded = post.to(branch.dtype).unsqueeze(-1) * branch.unsqueeze(-2) + torch.matmul(
            comb.to(streams.dtype), streams.unsqueeze(-3)
        ).squeeze(-2)
    with output.open("w", encoding="utf-8") as stream:
        stream.write("#ifndef COLIBRI_GLM53_MHC_CASE_H\n#define COLIBRI_GLM53_MHC_CASE_H\n\n")
        stream.write(f"#define GLM53_MHC_COPIES 2\n#define GLM53_MHC_DIM {HIDDEN}\n")
        stream.write("#define GLM53_MHC_ITERATIONS 3\n\n")
        write_c_array(stream, "glm53_mhc_streams", streams,
                      "[GLM53_MHC_COPIES * GLM53_MHC_DIM]")
        write_c_array(stream, "glm53_mhc_function", mhc.fn,
                      "[((2 + GLM53_MHC_COPIES) * GLM53_MHC_COPIES) * GLM53_MHC_COPIES * GLM53_MHC_DIM]")
        write_c_array(stream, "glm53_mhc_scale", mhc.scale, "[3]")
        write_c_array(stream, "glm53_mhc_base", mhc.base,
                      "[(2 + GLM53_MHC_COPIES) * GLM53_MHC_COPIES]")
        write_c_array(stream, "glm53_mhc_branch", branch, "[GLM53_MHC_DIM]")
        write_c_array(stream, "glm53_mhc_collapsed", collapsed, "[GLM53_MHC_DIM]")
        write_c_array(stream, "glm53_mhc_post", post, "[GLM53_MHC_COPIES]")
        write_c_array(stream, "glm53_mhc_comb", comb,
                      "[GLM53_MHC_COPIES * GLM53_MHC_COPIES]")
        write_c_array(stream, "glm53_mhc_output", expanded,
                      "[GLM53_MHC_COPIES * GLM53_MHC_DIM]")
        stream.write("#endif\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default = Path(__file__).resolve().parents[1] / "glm53_tiny"
    parser.add_argument("--output", type=Path, default=default)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--fp8-mlp", action="store_true")
    args = parser.parse_args()
    torch, transformers, save_file, Config, Model = require_dependencies()
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    model = Model(Config(**config_kwargs())).eval()
    head = torch.nn.Linear(HIDDEN, VOCAB, bias=False).eval()
    initialize_contract_weights(torch, model, head)
    if args.fp8_mlp:
        apply_fp8_mlp_oracle(torch, model)
    weights = production_layout(torch, model, head)
    if args.fp8_mlp:
        quantize_disk_weights(torch, model, weights)
    reference = oracle(torch, transformers, model, head)
    output = args.output.resolve()
    if output.exists():
        if not args.force:
            raise SystemExit(f"output exists (use --force): {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    (output / "config.json").write_text(
        json.dumps(runtime_config(transformers.__version__, args.fp8_mlp), indent=2) + "\n",
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
    write_sparse_attention_case(torch, model, output / "glm53_sparse_attention_case.h")
    write_mhc_case(torch, model, output / "glm53_mhc_case.h")
    write_indexer_case(torch, model, output / "glm53_indexer_case.h")
    write_fp8_case(torch, model, output / "glm53_fp8_case.h")
    size = sum(path.stat().st_size for path in output.iterdir())
    print(f"wrote {output} ({len(weights)} tensors, {size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
