#!/usr/bin/env python3
import json
from pathlib import Path

import torch
from safetensors import safe_open

from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    deepgemm_post_process_fp8_weight_block,
)
from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import (
    fused_inv_rope_fp8_quant,
)
from vllm.utils.deep_gemm import fp8_einsum


model = Path("/model")
weight_map = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]


def load(name: str) -> torch.Tensor:
    with safe_open(model / weight_map[name], framework="pt") as shard:
        return shard.get_tensor(name)


stem = "layers.0.attn.wo_a"
weight = load(stem + ".weight").cuda().contiguous()
scale = load(stem + ".scale").cuda().contiguous()
weight, scale = deepgemm_post_process_fp8_weight_block(
    weight, scale, (128, 128), use_e8m0=True, is_bmm=True, bmm_batch_size=8
)
o = torch.zeros((1, 64, 512), dtype=torch.bfloat16, device="cuda")
positions = torch.zeros((1,), dtype=torch.int64, device="cuda")
cos_sin = torch.zeros((1, 64), dtype=torch.float32, device="cuda")
o_fp8, o_scale = fused_inv_rope_fp8_quant(
    o, positions, cos_sin, n_groups=8, heads_per_group=8,
    nope_dim=448, rope_dim=64, tma_aligned_scales=True,
)
z = torch.empty((1, 8, 1024), dtype=torch.bfloat16, device="cuda")
fp8_einsum(
    "bhr,hdr->bhd", (o_fp8, o_scale), (weight, scale), z,
    recipe=(1, 1, 128),
)
torch.cuda.synchronize()
print("WO_A_PROBE", tuple(o_fp8.shape), tuple(o_scale.shape), tuple(weight.shape), tuple(scale.shape), tuple(z.shape))
