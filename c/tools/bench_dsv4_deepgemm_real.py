#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path

import torch
from safetensors import safe_open

import vllm.third_party.deep_gemm as deep_gemm
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    deepgemm_post_process_weight_scale_block,
    silu_mul_quant_fp8_packed_triton,
)


def load_tensor(model: Path, weight_map: dict[str, str], name: str) -> torch.Tensor:
    with safe_open(model / weight_map[name], framework="pt") as shard:
        return shard.get_tensor(name)


def dequant_row(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    table = torch.tensor(
        [0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6],
        dtype=torch.float32,
    )
    raw = packed.view(torch.uint8)
    values = torch.empty(raw.numel() * 2, dtype=torch.float32)
    values[0::2] = table[(raw & 15).long()]
    values[1::2] = table[(raw >> 4).long()]
    return values * scale.float().repeat_interleave(32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--experts", type=int, default=6)
    parser.add_argument("--alignment", type=int, default=128)
    args = parser.parse_args()
    weight_map = json.loads((args.model / "model.safetensors.index.json").read_text())["weight_map"]

    weights = []
    scales = []
    reference = []
    weights2 = []
    scales2 = []
    reference2 = []
    for expert in range(args.experts):
        parts = []
        part_scales = []
        part_reference = []
        for projection in ("w1", "w3"):
            stem = f"layers.{args.layer}.ffn.experts.{expert}.{projection}"
            weight = load_tensor(args.model, weight_map, stem + ".weight")
            scale = load_tensor(args.model, weight_map, stem + ".scale")
            parts.append(weight)
            part_scales.append(scale.float())
            if projection == "w1":
                part_reference.append(torch.stack([dequant_row(w, s) for w, s in zip(weight[:32], scale[:32])]))
        weights.append(torch.cat(parts))
        scales.append(torch.cat(part_scales))
        reference.append(part_reference[0].sum(dim=1))
        stem = f"layers.{args.layer}.ffn.experts.{expert}.w2"
        weight2 = load_tensor(args.model, weight_map, stem + ".weight")
        scale2 = load_tensor(args.model, weight_map, stem + ".scale")
        weights2.append(weight2)
        scales2.append(scale2.float())
        reference2.append(torch.stack([
            dequant_row(w, s) for w, s in zip(weight2[:32], scale2[:32])
        ]).sum(dim=1))

    weight = torch.stack(weights).cuda().contiguous()
    scale = torch.stack(scales).cuda().contiguous()
    scale = deepgemm_post_process_weight_scale_block(
        ws=scale, mn=4096, k=4096, quant_block_shape=(1, 32),
        num_groups=args.experts,
    )
    weight2 = torch.stack(weights2).cuda().contiguous()
    scale2 = torch.stack(scales2).cuda().contiguous()
    scale2 = deepgemm_post_process_weight_scale_block(
        ws=scale2, mn=4096, k=2048, quant_block_shape=(1, 32),
        num_groups=args.experts,
    )
    alignment = args.alignment
    rows = args.experts * alignment
    activation = torch.full((rows, 4096), 0x38, dtype=torch.uint8, device="cuda").view(torch.float8_e4m3fn)
    activation_scale = torch.ones((rows, 32), dtype=torch.float32, device="cuda")
    layout = torch.arange(args.experts, dtype=torch.int32, device="cuda").repeat_interleave(alignment)
    output = torch.empty((rows, 4096), dtype=torch.bfloat16, device="cuda")
    activation2 = torch.full((rows, 2048), 0x38, dtype=torch.uint8, device="cuda").view(torch.float8_e4m3fn)
    activation_scale2 = torch.ones((rows, 16), dtype=torch.float32, device="cuda")
    output2 = torch.empty((rows, 4096), dtype=torch.bfloat16, device="cuda")
    activation2_chain = torch.empty(
        (rows, 2048), dtype=torch.float8_e4m3fn, device="cuda"
    )

    def run() -> None:
        deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
            (activation, activation_scale), (weight, scale), output, layout,
            recipe_a=(1, 128), recipe_b=(1, 32),
        )

    def run2() -> None:
        deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
            (activation2, activation_scale2), (weight2, scale2), output2, layout,
            recipe_a=(1, 128), recipe_b=(1, 32),
        )

    def activate() -> tuple[torch.Tensor, torch.Tensor]:
        return silu_mul_quant_fp8_packed_triton(
            input=output, group_size=128, output_q=activation2_chain,
            clamp_limit=10.0,
        )

    def chain() -> None:
        run()
        a2q, a2q_scale = activate()
        deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
            (a2q, a2q_scale), (weight2, scale2), output2, layout,
            recipe_a=(1, 128), recipe_b=(1, 32),
        )

    for _ in range(10):
        run()
    torch.cuda.synchronize()
    begin = time.perf_counter()
    for _ in range(100):
        run()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - begin) * 10
    for _ in range(10):
        run2()
    torch.cuda.synchronize()
    begin = time.perf_counter()
    for _ in range(100):
        run2()
    torch.cuda.synchronize()
    elapsed2 = (time.perf_counter() - begin) * 10
    got2 = output2[::alignment, :32].float().cpu()
    for _ in range(10):
        activate()
    torch.cuda.synchronize()
    begin = time.perf_counter()
    for _ in range(100):
        activate()
    torch.cuda.synchronize()
    elapsed_activation = (time.perf_counter() - begin) * 10
    for _ in range(10):
        chain()
    torch.cuda.synchronize()
    begin = time.perf_counter()
    for _ in range(100):
        chain()
    torch.cuda.synchronize()
    elapsed_chain = (time.perf_counter() - begin) * 10

    got = output[::alignment, :32].float().cpu()
    expected = torch.stack(reference)
    absolute = (got - expected).abs()
    relative = absolute / expected.abs().clamp_min(1)
    worst = absolute.argmax().item()
    worst_expert, worst_row = divmod(worst, absolute.shape[1])
    print(f"layer={args.layer} experts={args.experts} fc1={elapsed:.4f} ms/launch")
    print(f"max_abs={absolute.max().item():.6g} max_rel={relative.max().item():.6g}")
    print(f"worst=({worst_expert},{worst_row}) got={got[worst_expert,worst_row].item():.6g} expected={expected[worst_expert,worst_row].item():.6g}")
    print("pair_mae", [[round((got[i] - expected[j]).abs().mean().item(), 4)
                        for j in range(args.experts)] for i in range(args.experts)])
    print("sample", got[0, :4].tolist(), expected[0, :4].tolist())
    expected2 = torch.stack(reference2)
    absolute2 = (got2 - expected2).abs()
    relative2 = absolute2 / expected2.abs().clamp_min(1)
    print(f"fc2={elapsed2:.4f} ms/launch max_abs={absolute2.max().item():.6g} max_rel={relative2.max().item():.6g}")
    print("fc2_sample", got2[0, :4].tolist(), expected2[0, :4].tolist())
    print(f"silu_requant={elapsed_activation:.4f} ms/launch")
    print(f"fc1+silu_requant+fc2={elapsed_chain:.4f} ms/launch")


if __name__ == "__main__":
    main()
