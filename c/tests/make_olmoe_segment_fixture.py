#!/usr/bin/env python3
"""Create a deterministic merged-int8 checkpoint for OLMoE Segment tests.

Standard library only. Synthetic weights exercise loading and public-runtime
state consistency; this is not a trained model or an independent accuracy oracle.
No tokenizer or model-edge tensors are needed by the layer-range adapter.
"""
import argparse
import json
import math
import random
import struct
from pathlib import Path


def create(output: Path):
    hidden, inter, experts, layers = 32, 16, 4, 4
    rng = random.Random(20261106)
    tensors, data = {}, bytearray()

    def add(name, shape, values, dtype="F32"):
        count = math.prod(shape)
        payload = struct.pack(f"<{count}{'f' if dtype == 'F32' else 'b'}", *values)
        tensors[name] = {"dtype": dtype, "shape": shape,
                         "data_offsets": [len(data), len(data) + len(payload)]}
        data.extend(payload)

    for layer in range(layers):
        prefix = f"model.layers.{layer}."
        for norm in ("input_layernorm", "post_attention_layernorm",
                     "self_attn.q_norm", "self_attn.k_norm"):
            add(prefix + norm + ".weight", [hidden], [1.0] * hidden)
        for projection in ("q", "k", "v", "o"):
            add(prefix + f"self_attn.{projection}_proj.weight", [hidden, hidden],
                [rng.uniform(-0.04, 0.04) for _ in range(hidden * hidden)])
        add(prefix + "mlp.gate.weight", [experts, hidden],
            [rng.uniform(-0.1, 0.1) for _ in range(experts * hidden)])
        for expert in range(experts):
            name = prefix + f"mlp.experts.{expert}."
            count = 3 * inter * hidden
            add(name + "merged_weight", [count],
                [rng.randint(-20, 20) for _ in range(count)], "I8")
            count = 2 * inter + hidden
            add(name + "qs", [count], [0.001 + rng.random() * 0.002 for _ in range(count)])

    config = dict(hidden_size=hidden, intermediate_size=inter, num_hidden_layers=layers,
                  num_attention_heads=4, num_key_value_heads=4, num_experts=experts,
                  num_experts_per_tok=2, vocab_size=128, norm_topk_prob=True,
                  rope_theta=10000.0, rms_norm_eps=1e-5)
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config) + "\n", encoding="utf-8")
    header = json.dumps(tensors, separators=(",", ":")).encode()
    header += b" " * (-len(header) % 8)
    (output / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + data)
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    create(parser.parse_args().output)
