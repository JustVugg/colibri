#!/usr/bin/env python3
"""Generate the deterministic tiny DeepSeek V4 target oracle fixture.

The dedicated V4 CI job runs this with pinned dependencies.  It deliberately
requires PyTorch and a Transformers release that contains the official
DeepseekV4ForCausalLM implementation. Reference tokens always come from that
implementation; there is no C-engine fallback and the generated safetensors
file is never committed.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import struct
from collections import OrderedDict
from pathlib import Path


SEED = 1234
SCHEMA_VERSION = 1
GENERATOR_VERSION = "2"
VOCAB = 128
HIDDEN = 128
LAYERS = 3
HEADS = 4
HEAD_DIM = 32
Q_RANK = 128
O_GROUPS = 1
O_RANK = 128
EXPERTS = 4
TOPK = 2
MOE = 128
HC = 2
INDEX_HEADS = 2
INDEX_DIM = 32
SLIDING = 8
COMPRESS_RATIOS = [0, 4, 8]
MAX_POSITIONS = 128

E2M1 = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)


def require_dependencies():
    try:
        import torch
        import transformers
        from transformers import DeepseekV4Config, DeepseekV4ForCausalLM
    except Exception as exc:  # pragma: no cover - regeneration-only diagnostic
        raise SystemExit(
            "DeepSeek V4 tiny generation requires PyTorch and "
            "an official Transformers build with DeepseekV4ForCausalLM"
        ) from exc
    if not hasattr(transformers, "DeepseekV4ForCausalLM"):
        raise SystemExit(
            f"Transformers {transformers.__version__} has no DeepSeek V4 support"
        )
    return torch, transformers, DeepseekV4Config, DeepseekV4ForCausalLM


def ceil_scale_exponent(maximum: float, limit: float) -> int:
    if maximum <= 0.0:
        return -127
    return max(-127, min(127, math.ceil(math.log2(maximum / limit))))


def quantize_fp8(torch, value):
    """Production 128x128 E4M3 + UE8M0 block round-trip."""
    value = value.detach().to(torch.float32).contiguous()
    if value.ndim != 2:
        raise ValueError("FP8 tensors must be matrices")
    rows, columns = value.shape
    if columns % 128:
        raise ValueError(f"FP8 contraction dimension must be 128-aligned: {value.shape}")
    quantized = torch.empty_like(value, dtype=torch.float8_e4m3fn)
    scales = torch.empty(
        ((rows + 127) // 128, (columns + 127) // 128),
        dtype=torch.float8_e8m0fnu,
    )
    dequantized = torch.empty_like(value)
    for row in range(0, rows, 128):
        for column in range(0, columns, 128):
            block = value[row : row + 128, column : column + 128]
            exponent = ceil_scale_exponent(float(block.abs().max()), 448.0)
            scale = math.ldexp(1.0, exponent)
            q = (block / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
            quantized[row : row + 128, column : column + 128] = q
            scales[row // 128, column // 128] = scale
            dequantized[row : row + 128, column : column + 128] = q.float() * scale
    return quantized, scales, dequantized


def quantize_fp4(torch, value):
    """Production native packed E2M1 + per-row/per-32 UE8M0 round-trip."""
    value = value.detach().to(torch.float32).contiguous()
    if value.ndim != 2:
        raise ValueError("FP4 tensors must be matrices")
    rows, columns = value.shape
    if columns % 128:
        raise ValueError(f"FP4 contraction dimension must be 128-aligned: {value.shape}")
    packed = torch.empty((rows, columns // 2), dtype=torch.uint8)
    scales = torch.empty((rows, columns // 32), dtype=torch.float8_e8m0fnu)
    dequantized = torch.empty_like(value)
    lut = torch.tensor(E2M1, dtype=torch.float32)
    for row in range(rows):
        codes = torch.empty(columns, dtype=torch.uint8)
        for column in range(0, columns, 32):
            block = value[row, column : column + 32]
            exponent = ceil_scale_exponent(float(block.abs().max()), 6.0)
            scale = math.ldexp(1.0, exponent)
            normalized = (block / scale).clamp(-6.0, 6.0)
            code = (normalized[:, None] - lut[None, :]).abs().argmin(dim=1)
            codes[column : column + 32] = code.to(torch.uint8)
            scales[row, column // 32] = scale
            dequantized[row, column : column + 32] = lut[code] * scale
        packed[row] = codes[0::2] | (codes[1::2] << 4)
    # Safetensors must advertise I8, while the runtime treats the bits as packed U8.
    return packed.view(torch.int8), scales, dequantized


def bf16_round(torch, value):
    return value.detach().to(torch.bfloat16).contiguous().float()


def tensor_bytes(torch, tensor) -> bytes:
    return tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def safetensors_dtype(torch, tensor) -> str:
    mapping = {
        torch.float32: "F32",
        torch.bfloat16: "BF16",
        torch.float8_e4m3fn: "F8_E4M3",
        torch.float8_e8m0fnu: "F8_E8M0",
        torch.int64: "I64",
        torch.int8: "I8",
    }
    try:
        return mapping[tensor.dtype]
    except KeyError as exc:
        raise ValueError(f"unsupported safetensors dtype: {tensor.dtype}") from exc


def write_safetensors(torch, path: Path, tensors: OrderedDict[str, object]) -> None:
    """Write a small standard safetensors file while preserving payload order.

    ExpertStore requires w1/w2/w3 scales to form one contiguous range and the
    three packed weights to form a second contiguous range.  Preserving the
    insertion order makes that layout explicit and reproducible.
    """
    header: OrderedDict[str, object] = OrderedDict()
    payloads: list[bytes] = []
    offset = 0
    header["__metadata__"] = {
        "format": "pt",
        "generator": "c/tools/make_deepseek_v4_tiny.py",
    }
    for name, tensor in tensors.items():
        payload = tensor_bytes(torch, tensor)
        header[name] = {
            "dtype": safetensors_dtype(torch, tensor),
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + len(payload)],
        }
        payloads.append(payload)
        offset += len(payload)
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((-len(encoded)) % 8)
    with path.open("wb") as stream:
        stream.write(struct.pack("<Q", len(encoded)))
        stream.write(encoded)
        for payload in payloads:
            stream.write(payload)


def make_hf_config(DeepseekV4Config):
    return DeepseekV4Config(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        moe_intermediate_size=MOE,
        num_hidden_layers=LAYERS,
        num_attention_heads=HEADS,
        num_key_value_heads=1,
        head_dim=HEAD_DIM,
        q_lora_rank=Q_RANK,
        num_experts_per_tok=TOPK,
        n_routed_experts=EXPERTS,
        n_shared_experts=1,
        scoring_func="sqrtsoftplus",
        norm_topk_prob=True,
        routed_scaling_factor=1.5,
        max_position_embeddings=MAX_POSITIONS,
        rope_theta=10000.0,
        layer_types=[
            "sliding_attention",
            "compressed_sparse_attention",
            "heavily_compressed_attention",
        ],
        compress_rates={
            "compressed_sparse_attention": 4,
            "heavily_compressed_attention": 8,
        },
        compress_rope_theta=40000.0,
        hc_mult=HC,
        hc_sinkhorn_iters=3,
        hc_eps=1.0e-6,
        mlp_layer_types=["hash_moe", "moe", "moe"],
        swiglu_limit=10.0,
        sliding_window=SLIDING,
        o_groups=O_GROUPS,
        o_lora_rank=O_RANK,
        index_n_heads=INDEX_HEADS,
        index_head_dim=INDEX_DIM,
        index_topk=2,
        num_nextn_predict_layers=1,
        initializer_range=0.02,
        rms_norm_eps=1.0e-6,
        bos_token_id=0,
        eos_token_id=1,
        tie_word_embeddings=False,
        partial_rotary_factor=0.5,
        rope_parameters={
            "main": {
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 0.5,
            },
            "compress": {
                "rope_type": "default",
                "rope_theta": 40000.0,
                "partial_rotary_factor": 0.5,
            },
        },
    )


def make_runtime_config(transformers_version: str) -> dict:
    config = {
        "architectures": ["DeepseekV4ForCausalLM"],
        "model_type": "deepseek_v4",
        "transformers_version": transformers_version,
        "torch_dtype": "bfloat16",
        "expert_dtype": "fp4",
        "scoring_func": "sqrtsoftplus",
        "topk_method": "noaux_tc",
        "hidden_size": HIDDEN,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": HEADS,
        "num_key_value_heads": 1,
        "head_dim": HEAD_DIM,
        "q_lora_rank": Q_RANK,
        "qk_rope_head_dim": HEAD_DIM // 2,
        "o_groups": O_GROUPS,
        "o_lora_rank": O_RANK,
        "sliding_window": SLIDING,
        "index_n_heads": INDEX_HEADS,
        "index_head_dim": INDEX_DIM,
        "index_topk": 2,
        "n_routed_experts": EXPERTS,
        "num_experts_per_tok": TOPK,
        "n_shared_experts": 1,
        "moe_intermediate_size": MOE,
        "num_hash_layers": 1,
        "num_nextn_predict_layers": 1,
        "hc_mult": HC,
        "hc_sinkhorn_iters": 3,
        "vocab_size": VOCAB,
        "max_position_embeddings": MAX_POSITIONS,
        "rms_norm_eps": 1.0e-6,
        "hc_eps": 1.0e-6,
        "routed_scaling_factor": 1.5,
        "swiglu_limit": 10.0,
        "rope_theta": 10000.0,
        "compress_rope_theta": 40000.0,
        "compress_ratios": [*COMPRESS_RATIOS, 0],
        "rope_scaling": {
            "type": "yarn",
            "factor": 1.0,
            "original_max_position_embeddings": MAX_POSITIONS,
            "beta_fast": 32,
            "beta_slow": 1,
        },
        "quantization_config": {
            "activation_scheme": "dynamic",
            "fmt": "e4m3",
            "scale_fmt": "ue8m0",
            "quant_method": "fp8",
            "weight_block_size": [128, 128],
        },
    }
    return config


def initialize_router_coverage(torch, model) -> None:
    with torch.no_grad():
        # Tie the output rows to a one-token cyclic shift of the input
        # embeddings.  The residual stream therefore has a wide, deterministic
        # top-1 margin instead of relying on accidental near-ties in a random
        # language-model head.  Generated IDs still vary at every step, while
        # production activation quantization cannot flip an otherwise fragile
        # oracle result.
        embeddings = model.model.embed_tokens.weight.detach().clone()
        for token in range(VOCAB):
            model.lm_head.weight[(token + 1) % VOCAB].copy_(embeddings[token])
        for layer_id, layer in enumerate(model.model.layers):
            gate = layer.mlp.gate
            values = torch.arange(gate.weight.numel(), dtype=torch.float32)
            gate.weight.copy_((0.025 * torch.sin(values * 0.017 + layer_id)).reshape_as(gate.weight))
            if hasattr(gate, "tid2eid"):
                table = torch.empty((VOCAB, TOPK), dtype=torch.int64)
                for token in range(VOCAB):
                    table[token, 0] = (token + layer_id) % EXPERTS
                    table[token, 1] = (token * 3 + layer_id + 1) % EXPERTS
                    if table[token, 1] == table[token, 0]:
                        table[token, 1] = (table[token, 1] + 1) % EXPERTS
                gate.tid2eid.copy_(table)
            if hasattr(gate, "e_score_correction_bias"):
                gate.e_score_correction_bias.copy_(
                    torch.linspace(-0.03, 0.03, EXPERTS)
                )
        # EOS is deliberately unattractive so every fixed-length regression is
        # capable of detecting early truncation instead of accepting a prefix.
        model.lm_head.weight[1].zero_()


def add_target_tensors(torch, model):
    state = model.state_dict()
    output: OrderedDict[str, object] = OrderedDict()

    def add_bf16(name: str, key: str) -> None:
        rounded = bf16_round(torch, state[key])
        state[key].copy_(rounded)
        output[name] = rounded.to(torch.bfloat16)

    def add_f32(name: str, key: str) -> None:
        output[name] = state[key].detach().float().contiguous()

    def add_fp8(name: str, key: str) -> None:
        quantized, scales, dequantized = quantize_fp8(torch, state[key])
        state[key].copy_(dequantized)
        output[name + ".weight"] = quantized
        output[name + ".scale"] = scales

    add_bf16("embed.weight", "model.embed_tokens.weight")
    add_bf16("head.weight", "lm_head.weight")
    add_bf16("norm.weight", "model.norm.weight")
    add_f32("hc_head_fn", "model.hc_head.hc_fn")
    add_f32("hc_head_base", "model.hc_head.hc_base")
    add_f32("hc_head_scale", "model.hc_head.hc_scale")

    for layer_id in range(LAYERS):
        hf = f"model.layers.{layer_id}"
        c = f"layers.{layer_id}"
        add_f32(f"{c}.attn.attn_sink", f"{hf}.self_attn.sinks")
        add_bf16(f"{c}.attn.kv_norm.weight", f"{hf}.self_attn.kv_norm.weight")
        add_bf16(f"{c}.attn.q_norm.weight", f"{hf}.self_attn.q_a_norm.weight")
        add_fp8(f"{c}.attn.wkv", f"{hf}.self_attn.kv_proj.weight")
        add_fp8(f"{c}.attn.wo_a", f"{hf}.self_attn.o_a_proj.weight")
        add_fp8(f"{c}.attn.wo_b", f"{hf}.self_attn.o_b_proj.weight")
        add_fp8(f"{c}.attn.wq_a", f"{hf}.self_attn.q_a_proj.weight")
        add_fp8(f"{c}.attn.wq_b", f"{hf}.self_attn.q_b_proj.weight")
        add_bf16(f"{c}.attn_norm.weight", f"{hf}.input_layernorm.weight")

        if COMPRESS_RATIOS[layer_id]:
            cp = f"{hf}.self_attn.compressor"
            add_f32(f"{c}.attn.compressor.ape", f"{cp}.position_bias")
            add_bf16(f"{c}.attn.compressor.norm.weight", f"{cp}.kv_norm.weight")
            add_bf16(f"{c}.attn.compressor.wgate.weight", f"{cp}.gate_proj.weight")
            add_bf16(f"{c}.attn.compressor.wkv.weight", f"{cp}.kv_proj.weight")
            if COMPRESS_RATIOS[layer_id] == 4:
                ip = f"{cp}.indexer"
                add_f32(f"{c}.attn.indexer.compressor.ape", f"{ip}.position_bias")
                add_bf16(f"{c}.attn.indexer.compressor.norm.weight", f"{ip}.kv_norm.weight")
                add_bf16(f"{c}.attn.indexer.compressor.wgate.weight", f"{ip}.gate_proj.weight")
                add_bf16(f"{c}.attn.indexer.compressor.wkv.weight", f"{ip}.kv_proj.weight")
                add_bf16(f"{c}.attn.indexer.weights_proj.weight", f"{ip}.scorer.weights_proj.weight")
                add_fp8(f"{c}.attn.indexer.wq_b", f"{ip}.q_b_proj.weight")

        add_bf16(f"{c}.ffn.gate.weight", f"{hf}.mlp.gate.weight")
        if layer_id == 0:
            output[f"{c}.ffn.gate.tid2eid"] = state[
                f"{hf}.mlp.gate.tid2eid"
            ].detach().to(torch.int64).contiguous()
        else:
            add_f32(
                f"{c}.ffn.gate.bias",
                f"{hf}.mlp.gate.e_score_correction_bias",
            )
        add_fp8(f"{c}.ffn.shared_experts.w1", f"{hf}.mlp.shared_experts.gate_proj.weight")
        add_fp8(f"{c}.ffn.shared_experts.w2", f"{hf}.mlp.shared_experts.down_proj.weight")
        add_fp8(f"{c}.ffn.shared_experts.w3", f"{hf}.mlp.shared_experts.up_proj.weight")
        add_bf16(f"{c}.ffn_norm.weight", f"{hf}.post_attention_layernorm.weight")
        add_f32(f"{c}.hc_attn_base", f"{hf}.attn_hc.base")
        add_f32(f"{c}.hc_attn_fn", f"{hf}.attn_hc.fn")
        add_f32(f"{c}.hc_attn_scale", f"{hf}.attn_hc.scale")
        add_f32(f"{c}.hc_ffn_base", f"{hf}.ffn_hc.base")
        add_f32(f"{c}.hc_ffn_fn", f"{hf}.ffn_hc.fn")
        add_f32(f"{c}.hc_ffn_scale", f"{hf}.ffn_hc.scale")

        gate_up_key = f"{hf}.mlp.experts.gate_up_proj"
        down_key = f"{hf}.mlp.experts.down_proj"
        for expert in range(EXPERTS):
            matrices = [
                state[gate_up_key][expert, :MOE],
                state[down_key][expert],
                state[gate_up_key][expert, MOE:],
            ]
            names = ("w1", "w2", "w3")
            converted = [quantize_fp4(torch, matrix) for matrix in matrices]
            for matrix, (_, _, dequantized) in zip(matrices, converted):
                matrix.copy_(dequantized)
            # ExpertStore depends on these two contiguous triples.
            for name, (_, scales, _) in zip(names, converted):
                output[f"{c}.ffn.experts.{expert}.{name}.scale"] = scales
            for name, (packed, _, _) in zip(names, converted):
                output[f"{c}.ffn.experts.{expert}.{name}.weight"] = packed
    return output


def make_tokenizer() -> dict:
    added = [
        {
            "id": token,
            "content": f"<t{token:03d}>",
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        }
        for token in range(VOCAB)
    ]
    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added,
        "normalizer": None,
        "pre_tokenizer": None,
        "post_processor": None,
        "decoder": None,
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": "",
            "end_of_word_suffix": "",
            "fuse_unk": False,
            "byte_fallback": False,
            "ignore_merges": True,
            "vocab": {"x": VOCAB - 1},
            "merges": [],
        },
    }


def greedy_reference(torch, model, prompt: list[int], max_new: int) -> tuple[list[int], list[int]]:
    sequence = list(prompt)
    with torch.no_grad():
        for _ in range(max_new):
            inputs = torch.tensor([sequence], dtype=torch.long)
            logits = model(input_ids=inputs, use_cache=False).logits[0, -1]
            sequence.append(int(logits.argmax()))
        full = torch.tensor([sequence], dtype=torch.long)
        teacher = model(input_ids=full, use_cache=False).logits[0].argmax(-1).tolist()
    generated = sequence[len(prompt) :]
    if len(generated) != max_new or 1 in generated:
        raise RuntimeError(
            f"reference generation truncated or produced EOS: prompt={prompt} generated={generated}"
        )
    return sequence, teacher


def make_reference(torch, transformers, model) -> dict:
    prompts = {
        "short": [5, 7, 9, 11, 13, 17, 19, 23],
        "compressed": [5 + (index * 7) % 97 for index in range(20)],
        "long": [5 + (index * 11) % 97 for index in range(72)],
    }
    max_new = {"short": 8, "compressed": 4, "long": 4}
    cases = {}
    for name, prompt in prompts.items():
        full, teacher = greedy_reference(torch, model, prompt, max_new[name])
        cases[name] = {
            "prompt_ids": prompt,
            "teacher_forcing_ids": teacher,
            "greedy_full_ids": full,
            "greedy_new_ids": full[len(prompt) :],
            "max_new_tokens": max_new[name],
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "seed": SEED,
        "source": "transformers",
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "quantization_format": {
            "dense": "E4M3 weight + UE8M0 128x128 scale, dequantized in reference",
            "routed_experts": "packed E2M1 + UE8M0 per-row/per-32 scale, dequantized in reference",
            "bf16": "round-tripped before reference",
        },
        "config_summary": {
            "vocab_size": VOCAB,
            "hidden_size": HIDDEN,
            "num_hidden_layers": LAYERS,
            "num_attention_heads": HEADS,
            "head_dim": HEAD_DIM,
            "q_lora_rank": Q_RANK,
            "o_lora_rank": O_RANK,
            "hc_mult": HC,
            "n_routed_experts": EXPERTS,
            "num_experts_per_tok": TOPK,
            "n_shared_experts": 1,
            "moe_intermediate_size": MOE,
            "sliding_window": SLIDING,
            "compress_ratios": COMPRESS_RATIOS,
            "index_n_heads": INDEX_HEADS,
            "index_head_dim": INDEX_DIM,
            "index_topk": 2,
            "num_nextn_predict_layers": 1,
        },
        "prompt_ids_short": prompts["short"],
        "prompt_ids_compressed": prompts["compressed"],
        "prompt_ids_long": prompts["long"],
        "cases": cases,
    }


def print_manifest(label: str, tensors: OrderedDict[str, object]) -> None:
    print(f"[{label}] {len(tensors)} tensors")
    for name, tensor in tensors.items():
        print(f"  {name}: {list(tensor.shape)} {tensor.dtype}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default = Path(__file__).resolve().parents[1] / "deepseek_v4_tiny"
    parser.add_argument("--output", type=Path, default=default)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    torch, transformers, Config, Model = require_dependencies()
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    model = Model(make_hf_config(Config)).eval()
    initialize_router_coverage(torch, model)

    target = add_target_tensors(torch, model)
    reference = make_reference(torch, transformers, model)

    output = args.output.resolve()
    if output.exists():
        if not args.force:
            raise SystemExit(f"output exists (use --force): {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    (output / "config.json").write_text(
        json.dumps(make_runtime_config(transformers.__version__), indent=2) + "\n",
        encoding="utf-8",
    )
    tokenizer = json.dumps(make_tokenizer(), separators=(",", ":")) + "\n"
    (output / "tokenizer.json").write_text(tokenizer, encoding="utf-8")
    write_safetensors(torch, output / "model.safetensors", target)
    (output / "ref.json").write_text(
        json.dumps(reference, indent=2) + "\n", encoding="utf-8"
    )

    print_manifest("target", target)
    total = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    print(f"wrote {output} ({total} bytes, transformers={transformers.__version__})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
