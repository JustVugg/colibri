#!/usr/bin/env python3
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open

import vllm.third_party.deep_gemm as deep_gemm
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    deepgemm_post_process_weight_scale_block,
)


MODEL = Path("/model")
LAYER = 0
GROUPS = 256
MAX_M = 16


def load(weight_map: dict[str, str], name: str) -> torch.Tensor:
    with safe_open(MODEL / weight_map[name], framework="pt") as shard:
        return shard.get_tensor(name)


def main() -> None:
    fc2 = os.getenv("DSV4_FC2") == "1"
    k = 2048 if fc2 else 4096
    weight_map = json.loads((MODEL / "model.safetensors.index.json").read_text())["weight_map"]
    weights, scales = [], []
    for expert in range(GROUPS):
        ew, es = [], []
        for projection in (("w2",) if fc2 else ("w1", "w3")):
            stem = f"layers.{LAYER}.ffn.experts.{expert}.{projection}"
            ew.append(load(weight_map, stem + ".weight"))
            es.append(load(weight_map, stem + ".scale").float())
        weights.append(torch.cat(ew))
        scales.append(torch.cat(es))

    weight = torch.stack(weights).cuda().contiguous()
    scale = deepgemm_post_process_weight_scale_block(
        torch.stack(scales).cuda().contiguous(),
        mn=4096,
        k=k,
        quant_block_shape=(1, 32),
        num_groups=GROUPS,
    )
    activation = torch.full(
        (GROUPS, MAX_M, k), 0x38, dtype=torch.uint8, device="cuda"
    ).view(torch.float8_e4m3fn)
    activation_scale = torch.ones((GROUPS, MAX_M, k // 128), dtype=torch.float32, device="cuda")
    output = torch.empty((GROUPS, MAX_M, 4096), dtype=torch.bfloat16, device="cuda")
    masked_m = torch.zeros(GROUPS, dtype=torch.int32, device="cuda")
    masked_m[:6] = 1

    for _ in range(3):
        deep_gemm.m_grouped_fp8_fp4_gemm_nt_masked(
            (activation, activation_scale), (weight, scale), output, masked_m, 1,
            recipe_a=(1, 128), recipe_b=(1, 32),
        )
    torch.cuda.synchronize()
    print("MASKED_OK", output[:6, 0, :4].float().cpu().tolist())


if __name__ == "__main__":
    main()
