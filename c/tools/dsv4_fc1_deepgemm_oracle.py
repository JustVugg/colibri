#!/usr/bin/env python3
import json
from pathlib import Path

import torch
from safetensors import safe_open

import vllm.third_party.deep_gemm as deep_gemm
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    deepgemm_post_process_weight_scale_block,
    per_token_group_quant_fp8_packed_for_deepgemm,
)


MODEL = Path("/model")
FIXTURE = Path("/fixture")


def raw(name: str, dtype: torch.dtype) -> torch.Tensor:
    return torch.frombuffer(bytearray((FIXTURE / name).read_bytes()), dtype=dtype)


def load(weight_map: dict[str, str], name: str) -> torch.Tensor:
    with safe_open(MODEL / weight_map[name], framework="pt") as shard:
        return shard.get_tensor(name)


def main() -> None:
    weight_map = json.loads((MODEL / "model.safetensors.index.json").read_text())["weight_map"]
    ids = raw("dsv4-dg-ids.i32", torch.int32).long()
    x = raw("dsv4-dg-x.f32", torch.float32).cuda()
    rows = 768
    replicated = x.to(torch.bfloat16).repeat(rows, 1)
    q, scale = per_token_group_quant_fp8_packed_for_deepgemm(
        replicated, 128, use_ue8m0=True
    )

    native_q = raw("dsv4-dg-a1.fp8", torch.uint8).view(rows, 4096)
    native_scale = raw("dsv4-dg-sfa1.bin", torch.int32).view(8, rows)
    official_q = q.view(torch.uint8).cpu()
    official_scale = scale.t().contiguous().cpu()
    print(
        "quant",
        f"bytes_equal={torch.equal(native_q, official_q)}",
        f"byte_mismatch={(native_q != official_q).sum().item()}",
        f"scales_equal={torch.equal(native_scale, official_scale)}",
        f"scale_mismatch={(native_scale != official_scale).sum().item()}",
    )

    weights, scales = [], []
    for expert in ids.tolist():
        ew, es = [], []
        for projection in ("w1", "w3"):
            stem = f"layers.0.ffn.experts.{expert}.{projection}"
            ew.append(load(weight_map, stem + ".weight"))
            es.append(load(weight_map, stem + ".scale").float())
        weights.append(torch.cat(ew))
        scales.append(torch.cat(es))
    weight = torch.stack(weights).cuda().contiguous()
    scale_w = deepgemm_post_process_weight_scale_block(
        torch.stack(scales).cuda().contiguous(),
        mn=4096,
        k=4096,
        quant_block_shape=(1, 32),
        num_groups=6,
    )
    layout = torch.arange(6, dtype=torch.int32, device="cuda").repeat_interleave(128)
    output = torch.empty((rows, 4096), dtype=torch.bfloat16, device="cuda")
    deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
        (q, scale), (weight, scale_w), output, layout,
        recipe_a=(1, 128), recipe_b=(1, 32),
    )
    torch.cuda.synchronize()
    native = raw("dsv4-dg-fc1.bf16", torch.bfloat16).view(rows, 4096).float()
    official = output.cpu().float()
    delta = (native - official).abs()
    print(
        "fc1",
        f"exact={torch.equal(native, official)}",
        f"max_abs={delta.max().item():.9g}",
        f"mae={delta.mean().item():.9g}",
        f"mismatch={(delta != 0).sum().item()}",
    )
    for name, part in (("gate", slice(0, 2048)), ("up", slice(2048, 4096))):
        d = delta[:, part]
        print(
            name,
            f"exact={torch.equal(native[:, part], official[:, part])}",
            f"max_abs={d.max().item():.9g}",
            f"mae={d.mean().item():.9g}",
            f"mismatch={(d != 0).sum().item()}",
        )


if __name__ == "__main__":
    main()
