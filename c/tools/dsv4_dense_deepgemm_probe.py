#!/usr/bin/env python3
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open

from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    deepgemm_post_process_fp8_weight_block,
    per_token_group_quant_fp8_packed_for_deepgemm,
)
from vllm.utils.deep_gemm import fp8_gemm_nt


model = Path("/model")
weight_map = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]


def load(name: str) -> torch.Tensor:
    with safe_open(model / weight_map[name], framework="pt") as shard:
        return shard.get_tensor(name)


projection = os.environ.get("PROJECTION", "w1")
stem = os.environ.get("STEM", f"layers.0.ffn.shared_experts.{projection}")
weight = load(stem + ".weight").cuda().contiguous()
scale = load(stem + ".scale").cuda().contiguous()
weight, scale = deepgemm_post_process_fp8_weight_block(
    weight, scale, (128, 128), use_e8m0=True
)
x_path = Path("/fixture/dsv4-dense-x.f32") if projection == "w1" else Path("/missing")
x_width = weight.shape[1]
x = (
    torch.frombuffer(bytearray(x_path.read_bytes()), dtype=torch.float32)
    .to(torch.bfloat16)
    .view(1, x_width)
    .cuda()
    if x_path.exists()
    else torch.zeros((1, x_width), dtype=torch.bfloat16, device="cuda")
)
q, qs = per_token_group_quant_fp8_packed_for_deepgemm(
    x, 128, use_ue8m0=True
)
out = torch.empty((1, weight.shape[0]), dtype=torch.bfloat16, device="cuda")
fp8_gemm_nt((q, qs), (weight, scale), out)
torch.cuda.synchronize()
print("DENSE_PROBE", stem, tuple(weight.shape), tuple(out.shape), out[0, 0].item())
if x_path.exists():
    def raw(name: str, dtype: torch.dtype) -> torch.Tensor:
        return torch.frombuffer(bytearray((Path("/fixture") / name).read_bytes()), dtype=dtype)

    cq = raw("dsv4-dense-q.fp8", torch.uint8)
    csfa = raw("dsv4-dense-sfa.bin", torch.int32).view(8, 4)
    csfb = raw("dsv4-dense-sfb.bin", torch.int32).view(8, 2048)
    cout = raw("dsv4-dense-out.f32", torch.float32)
    oq = q.view(torch.uint8).cpu().flatten()
    osfa = qs.t().contiguous().cpu()
    osfb = scale.t().contiguous().cpu()
    oout = out.cpu().float().flatten()
    print("Q", torch.equal(cq, oq), (cq != oq).sum().item())
    print("SFA", torch.equal(csfa, osfa), (csfa != osfa).sum().item())
    print("SFB", torch.equal(csfb, osfb), (csfb != osfb).sum().item())
    delta = (cout - oout).abs()
    print("OUT", delta.max().item(), delta.mean().item())
