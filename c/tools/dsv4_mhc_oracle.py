#!/usr/bin/env python3
"""Dump the official first-token, first-layer mHC-pre result."""
import struct
import sys

import torch
from safetensors import safe_open

model, token, output = sys.argv[1], int(sys.argv[2]), sys.argv[3]
index = __import__("json").load(open(model + "/model.safetensors.index.json"))["weight_map"]

def tensor(name):
    with safe_open(model + "/" + index[name], framework="pt", device="cpu") as f:
        return f.get_tensor(name)

x = tensor("embed.weight")[token].repeat(4, 1)
fn = tensor("layers.0.hc_attn_fn").float()
scale = tensor("layers.0.hc_attn_scale").float()
base = tensor("layers.0.hc_attn_base").float()
flat = x.flatten().float()
mix = torch.mv(fn, flat) * torch.rsqrt(flat.square().mean() + 1e-6)
pre = torch.sigmoid(mix[:4] * scale[0] + base[:4]) + 1e-6
post = torch.sigmoid(mix[4:8] * scale[1] + base[4:8]) * 2.0
comb = torch.softmax((mix[8:] * scale[2] + base[8:]).view(4, 4), dim=-1) + 1e-6
comb = comb / (comb.sum(dim=-2, keepdim=True) + 1e-6)
for _ in range(19):
    comb = comb / (comb.sum(dim=-1, keepdim=True) + 1e-6)
    comb = comb / (comb.sum(dim=-2, keepdim=True) + 1e-6)
layer_input = (pre[:, None] * x.float()).sum(0).bfloat16().float()
with open(output, "wb") as f:
    for value in (layer_input, post, comb.flatten()):
        f.write(struct.pack(f"<{value.numel()}f", *value.tolist()))
print(layer_input[0].item(), post[0].item(), comb.flatten()[0].item())
